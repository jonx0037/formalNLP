"""n-Gram language models and smoothing — the reference implementation for formalNLP's third topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. The zero-count problem: the MLE is the relative frequency, and it gives held-out text
   probability 0 as soon as one held-out n-gram is unseen.
2. Smoothing as shrinkage: the hierarchical Dirichlet estimate is an interpolation with weight
   α/(c(u·) + α); α is fit by Type-II maximum likelihood (the Dirichlet–multinomial evidence).
3. Good–Turing: E[M_r(n)] = (r+1)/(n+1)·E[N_{r+1}(n+1)] exactly; r* = (r+1)N_{r+1}/N_r is Robbins'
   formula with word types as policy holders; Simple Good–Turing smooths N_r where it is sparse.
4. Backoff: Katz's backoff weight makes the model normalize.
5. Absolute discounting: the leave-one-out likelihood is strictly concave, with optimum
   D* ≤ n₁/(n₁ + 2n₂).
6. Kneser–Ney: with one discount per order, continuation counts are the unique lower-order
   distribution that preserves the empirical marginals.
7. Kneser–Ney as hierarchical Pitman–Yor: with θ = 0 and one table per word type, the HPYLM
   predictive is interpolated Kneser–Ney with D = d.

Cross-checks: the Dirichlet–multinomial evidence against ``scipy.stats.dirichlet_multinomial``; our
Simple Good–Turing against ``nltk.probability.SimpleGoodTuringProbDist``; our interpolated
Kneser–Ney against ``nltk.lm.KneserNeyInterpolated`` in the configuration nltk implements (one
discount at every order, an undiscounted continuation unigram as the base).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with nltk==3.10.4 python notebooks/n-gram-models-smoothing/n_gram_models_smoothing.py
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "finance-corpus"))
sys.path.insert(0, str(_HERE.parent / "zipf-heaps-vocabulary-statistics"))
import finance_corpus as fc  # noqa: E402
import zipf_heaps_vocabulary_statistics as zhv  # noqa: E402

# ONE stream for every random-instance claim below (STARTER-PROMPT A1).
SEED = 20261023

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
ORDER = 3  # the page's models are trigram models
NGram = tuple[str, ...]


# --------------------------------------------------------------------------------------------
# 1. Data, counts, and the zero-count problem.
# --------------------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[NGram]:
    """Sentences on terminal punctuation, tokenized as Topics 1–2 tokenize (fc._tokens)."""
    return [tuple(t) for s in _SENTENCE_END.split(text) if (t := fc._tokens(s))]


@lru_cache(maxsize=1)
def corpus() -> dict[str, tuple[NGram, ...]]:
    """train: generate_corpus(300, 0) (Topics 1–2's corpus); held: generate_corpus(100, 1) (Topic 2's
    held-out split); edgar: the public MD&A excerpts, out of domain. Raw tokens, no <unk> mapping."""

    def sents(docs) -> tuple[NGram, ...]:
        return tuple(s for d in docs for s in split_sentences(d.text))

    return {"train": sents(fc.generate_corpus(300, 0)), "held": sents(fc.generate_corpus(100, 1)),
            "edgar": sents(fc.load_edgar_excerpts())}


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    """The predicted vocabulary V: training word types, </s>, and <unk>. <s> is only ever context."""
    return frozenset(w for s in corpus()["train"] for w in s) | {EOS, UNK}


def with_unk(sents, vocab: frozenset[str] | None = None) -> tuple[NGram, ...]:
    vocab = vocabulary() if vocab is None else vocab
    return tuple(tuple(w if w in vocab else UNK for w in s) for s in sents)


def events(sents, n: int):
    """The order-n events: for every predicted token (each word and the final </s>), the n-gram ending
    at it, with n − 1 <s> pads in front."""
    for s in sents:
        p = (BOS,) * (n - 1) + tuple(s) + (EOS,)
        for i in range(n - 1, len(p)):
            yield p[i - n + 1 : i + 1]


def ngram_counts(sents, n: int) -> dict[int, Counter]:
    """Raw k-gram counts for k = 1, …, n, each a marginal of the order-n events: every predicted token
    contributes the k-gram ending at it. So c(u·) at order k equals Σ_x c(x u ·) at order k + 1, and the
    k-gram counts do not depend on n."""
    out = {k: Counter() for k in range(1, n + 1)}
    for g in events(sents, n):
        for k in range(1, n + 1):
            out[k][g[n - k :]] += 1
    return out


def context_totals(counts: Counter) -> Counter:
    """c(u·) = Σ_w c(u w) for every context u of one order's counts."""
    tot: Counter = Counter()
    for g, c in counts.items():
        tot[g[:-1]] += c
    return tot


def count_of_counts(counts: Counter) -> dict[int, int]:
    """N_r, the number of distinct n-grams seen exactly r times — Topic 1's frequency spectrum."""
    return zhv.frequency_spectrum(np.fromiter(counts.values(), dtype=float))


@lru_cache(maxsize=1)
def train_counts() -> dict[int, Counter]:
    return ngram_counts(corpus()["train"], ORDER)


@lru_cache(maxsize=1)
def train_totals() -> dict[int, Counter]:
    return {k: context_totals(c) for k, c in train_counts().items()}


@lru_cache(maxsize=4)
def heldout_events(name: str = "held") -> tuple[NGram, ...]:
    return tuple(events(with_unk(corpus()[name]), ORDER))


def log_loss(prob, evs) -> float:
    """Average −log P(w | u) over events, nats per token; ∞ as soon as one probability is 0.
    prob(word, context) → float."""
    total = 0.0
    for g in evs:
        p = prob(g[-1], g[:-1])
        if p <= 0:
            return math.inf
        total -= math.log(p)
    return total / len(evs)


def mle_prob(word: str, context: NGram) -> float:
    """P_ML(w | u) = c(u w)/c(u·) on the training counts; 0 when the context was never seen."""
    k = len(context) + 1
    tot = train_totals()[k].get(context, 0)
    return train_counts()[k].get(context + (word,), 0) / tot if tot else 0.0


def unseen_shares(name: str = "held") -> dict[int, float]:
    """Share of the held-out events whose k-gram suffix never occurs in training, k = 1, …, ORDER."""
    evs, counts = heldout_events(name), train_counts()
    return {k: float(np.mean([g[ORDER - k :] not in counts[k] for g in evs])) for k in range(1, ORDER + 1)}


# --------------------------------------------------------------------------------------------
# 2. Smoothing as shrinkage: additive and hierarchical Dirichlet; Type-II α.
# --------------------------------------------------------------------------------------------


def additive_prob(word: str, context: NGram, delta: float = 1.0) -> float:
    """Lidstone add-δ (Laplace at δ = 1): (c(u w) + δ)/(c(u·) + δ|V|) — Dirichlet(δ, …, δ) posterior mean."""
    k = len(context) + 1
    c = train_counts()[k].get(context + (word,), 0)
    return (c + delta) / (train_totals()[k].get(context, 0) + delta * len(vocabulary()))


def level_arrays(grams, counts=None, totals=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """For k-grams g (all one length k): [(c(g_j), c(u_j ·)) for j = 1, …, k], where g_j is g's last j
    words and u_j its context. The hierarchical Dirichlet predictive is a function of these alone."""
    counts = train_counts() if counts is None else counts
    totals = train_totals() if totals is None else totals
    k = len(grams[0])
    return [(np.array([counts[j].get(g[k - j :], 0) for g in grams], float),
             np.array([totals[j].get(g[k - j : -1], 0) for g in grams], float)) for j in range(1, k + 1)]


def dirichlet_probs(levels, alphas, V: int) -> np.ndarray:
    """MacKay–Peto hierarchical Dirichlet predictive along each gram's chain of suffixes:
    P_1 = (c + α₁/V)/(N + α₁),  P_j = (c_j + α_j·P_{j−1})/(c(u_j·) + α_j)."""
    p = np.full(len(levels[0][0]), 1.0 / V)
    for (c, t), a in zip(levels, alphas):
        p = (c + a * p) / (t + a)
    return p


def dirichlet_log_evidence(alpha: float, c: np.ndarray, m: np.ndarray, ctx_totals: np.ndarray) -> float:
    """Thm 1: log p(the word sequences after each context | α, m) under a Dirichlet(α·m) prior per
    context — Σ_u [log Γ(α) − log Γ(α + c(u·))] + Σ_{(u,w) seen} [log Γ(α m_w + c(u w)) − log Γ(α m_w)].
    The sequence probability, so no multinomial coefficient; unseen (u, w) contribute 0."""
    return float(np.sum(special.gammaln(alpha) - special.gammaln(alpha + ctx_totals))
                 + np.sum(special.gammaln(alpha * m + c) - special.gammaln(alpha * m)))


def _evidence_terms(k: int, alphas_below) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(c, m, c(u·)) for Thm 1 at order k, with m = P_{k−1} under the already-fitted lower α's."""
    counts, V = train_counts(), len(vocabulary())
    grams = list(counts[k])
    c = np.array([counts[k][g] for g in grams], float)
    m = (np.full(len(grams), 1.0 / V) if k == 1
         else dirichlet_probs(level_arrays([g[1:] for g in grams]), alphas_below, V))
    return c, m, np.array(list(train_totals()[k].values()), float)


@lru_cache(maxsize=1)
def type2_alphas() -> tuple[float, ...]:
    """α_k maximizing Thm 1's evidence, order by order from the unigram up (Type-II MLE)."""
    alphas: list[float] = []
    for k in range(1, ORDER + 1):
        c, m, tot = _evidence_terms(k, tuple(alphas))
        res = optimize.minimize_scalar(lambda la: -dirichlet_log_evidence(math.exp(la), c, m, tot),
                                       bounds=(math.log(1e-3), math.log(1e7)), method="bounded",
                                       options={"xatol": 1e-9})
        alphas.append(math.exp(res.x))
    return tuple(alphas)


@lru_cache(maxsize=4)
def _held_levels(name: str = "held"):
    return level_arrays(list(heldout_events(name)))


def dirichlet_log_loss(alphas, name: str = "held") -> float:
    return float(-np.mean(np.log(dirichlet_probs(_held_levels(name), alphas, len(vocabulary())))))


@lru_cache(maxsize=1)
def heldout_optimal_alphas() -> tuple[float, ...]:
    """The oracle benchmark: α's minimizing held-out log-loss (coordinate descent in log α)."""
    alphas = list(type2_alphas())
    for _ in range(3):
        for k in range(ORDER):
            def f(la, k=k):
                trial = alphas.copy()
                trial[k] = math.exp(la)
                return dirichlet_log_loss(tuple(trial))
            alphas[k] = math.exp(optimize.minimize_scalar(f, bounds=(math.log(1e-3), math.log(1e7)),
                                                          method="bounded", options={"xatol": 1e-9}).x)
    return tuple(alphas)


# --------------------------------------------------------------------------------------------
# 3. Good–Turing: Robbins' formula on word counts.
# --------------------------------------------------------------------------------------------

# Efron & Hastie, CASI Table 6.1 — 9,461 auto-insurance policy holders by claims x = 0, …, 7 (the
# counts formalStatistics Topic 33 §33.3 Example 3 runs Robbins' formula on).
CASI_CLAIMS = (7840, 1317, 239, 42, 14, 4, 4, 1)
CASI_ROBBINS = (0.168, 0.363, 0.527, 1.33, 1.43, 6.00, 1.75)  # as printed there, x = 0, …, 6


def turing_rstar(N: dict[int, float]) -> dict[int, float]:
    """Turing's adjusted counts r* = (r + 1)·N_{r+1}/N_r for every r with N_r > 0."""
    return {r: (r + 1) * N.get(r + 1, 0) / n for r, n in sorted(N.items()) if n > 0}


def expected_spectrum(p: np.ndarray, n: int, r: int) -> float:
    """E[N_r(n)] = Σ_w C(n, r) p_w^r (1 − p_w)^{n−r} for n i.i.d. draws from p."""
    return float(np.sum(stats.binom.pmf(r, n, p)))


def expected_mass(p: np.ndarray, n: int, r: int) -> float:
    """E[M_r(n)] = Σ_w p_w·P(c_w = r): the expected total probability of the types seen r times."""
    return float(np.sum(p * stats.binom.pmf(r, n, p)))


@dataclass(frozen=True)
class SimpleGoodTuring:
    """Gale & Sampson (1995): Turing's r* while it differs significantly from the log-linear fit to the
    averaged spectrum Z_r, the fit's r* from then on, renormalized so the seen types share 1 − N₁/N."""

    N: dict[int, int]
    total: int
    slope: float
    intercept: float
    switch_at: int
    renorm: float

    def smoothed(self, r: float) -> float:
        return math.exp(self.intercept + self.slope * math.log(r))

    def rstar(self, r: int) -> float:
        if r < self.switch_at:
            return (r + 1) * self.N.get(r + 1, 0) / self.N[r]
        return (r + 1) * self.smoothed(r + 1) / self.smoothed(r)

    def prob(self, r: int) -> float:
        """Probability of ONE type seen r times (r ≥ 1); r = 0 returns the total unseen mass N₁/N."""
        return self.N.get(1, 0) / self.total if r == 0 else self.rstar(r) / self.total * self.renorm


def simple_good_turing(N: dict[int, int]) -> SimpleGoodTuring:
    rs = sorted(r for r, n in N.items() if r > 0 and n > 0)
    z = []
    for j, r in enumerate(rs):
        q = rs[j - 1] if j > 0 else 0
        t = rs[j + 1] if j + 1 < len(rs) else 2 * r - q
        z.append(2.0 * N[r] / (t - q))
    slope, intercept = np.polyfit(np.log(rs), np.log(z), 1)
    if slope >= -1:
        raise ValueError(f"log-linear slope {slope:.3f} ≥ −1: the smoothed r* would not shrink")
    total = sum(r * N[r] for r in rs)
    fit = SimpleGoodTuring(dict(N), total, float(slope), float(intercept), rs[-1], 1.0)
    for j, r in enumerate(rs):
        if j + 1 == len(rs) or rs[j + 1] != r + 1:
            switch = r
            break
        x = (r + 1) * N[r + 1] / N[r]
        y = (r + 1) * fit.smoothed(r + 1) / fit.smoothed(r)
        if abs(x - y) <= 1.96 * math.sqrt((r + 1) ** 2 * N[r + 1] / N[r] ** 2 * (1 + N[r + 1] / N[r])):
            switch = r
            break
    fit = SimpleGoodTuring(dict(N), total, float(slope), float(intercept), switch, 1.0)
    seen = sum(N[r] * fit.rstar(r) / total for r in rs)
    return SimpleGoodTuring(dict(N), total, float(slope), float(intercept), switch,
                            (1 - N.get(1, 0) / total) / seen)


# --------------------------------------------------------------------------------------------
# 4. Katz backoff.
# --------------------------------------------------------------------------------------------

KATZ_CUTOFF = 5  # Katz discounts counts r ≤ k and trusts larger counts as they are


def katz_discounts(N: dict[int, int], k: int = KATZ_CUTOFF) -> dict[int, float]:
    """d_r = (r*/r − (k+1)N_{k+1}/N₁)/(1 − (k+1)N_{k+1}/N₁) for 1 ≤ r ≤ k, r* Turing's (Katz 1987)."""
    g = (k + 1) * N[k + 1] / N[1]
    return {r: ((r + 1) * N[r + 1] / (r * N[r]) - g) / (1 - g) for r in range(1, k + 1)}


@dataclass(frozen=True)
class KatzModel:
    order: int
    counts: dict[int, Counter]
    totals: dict[int, Counter]
    d: dict[int, dict[int, float]]
    alpha: dict[NGram, float]  # backoff weight of every seen context (filled order by order)
    vocab_size: int
    exhausted: set[NGram]  # contexts whose seen followers hold all the lower order's mass: kept at the MLE


def katz_prob(word: str, context: NGram, m: KatzModel) -> float:
    """Discounted MLE for seen n-grams; α(u)·P_katz(w | u⁻) otherwise; Laplace at the unigram level."""
    k = len(context) + 1
    if k == 1:
        return (m.counts[1].get((word,), 0) + 1) / (m.totals[1][()] + m.vocab_size)
    c = m.counts[k].get(context + (word,), 0)
    if c:
        d = 1.0 if context in m.exhausted else m.d[k].get(c, 1.0)
        return d * c / m.totals[k][context]
    return m.alpha.get(context, 1.0) * katz_prob(word, context[1:], m)


def katz_fit(sents, order: int = ORDER) -> KatzModel:
    """Prop 4: α(u) = (1 − Σ_{seen w} P*(w|u)) / (1 − Σ_{seen w} P_katz(w|u⁻)) makes P_katz(·|u) sum to 1.
    When the denominator is 0 — every word the lower order can predict after u⁻ was already seen after
    u (templated text does this) — the discounted mass has nowhere to go; such a context keeps its MLE."""
    counts = ngram_counts(sents, order)
    totals = {k: context_totals(c) for k, c in counts.items()}
    d = {k: katz_discounts(count_of_counts(counts[k])) for k in range(2, order + 1)}
    m = KatzModel(order, counts, totals, d, {}, len(vocabulary()), set())
    for k in range(2, order + 1):
        followers: dict[NGram, list[tuple[str, int]]] = defaultdict(list)
        for g, c in counts[k].items():
            followers[g[:-1]].append((g[-1], c))
        for u, ws in followers.items():
            kept = sum(d[k].get(c, 1.0) * c for _, c in ws) / totals[k][u]
            lower = sum(katz_prob(w, u[1:], m) for w, _ in ws)
            if 1 - lower < 1e-12:
                m.exhausted.add(u)
                m.alpha[u] = 0.0
            else:
                m.alpha[u] = (1 - kept) / (1 - lower)
    return m


@lru_cache(maxsize=1)
def katz_model() -> KatzModel:
    return katz_fit(corpus()["train"])


# --------------------------------------------------------------------------------------------
# 5. Absolute discounting and the leave-one-out discount.
# --------------------------------------------------------------------------------------------


def ney_discount(N: dict[int, int]) -> float:
    """Ney, Essen & Kneser's D = n₁/(n₁ + 2n₂) — the upper bound of Thm 3."""
    return N[1] / (N[1] + 2 * N[2])


def loo_spectrum(counts: Counter) -> dict[int, int]:
    """The spectrum Thm 3 needs: n_r for r ≥ 2 as usual, but n₁ counts only singletons whose context
    survives their removal (c(u·) ≥ 2). A singleton alone in its context leaves it unseen, and its
    leave-one-out probability is then the backoff distribution's, which does not involve D."""
    tot = context_totals(counts)
    N = dict(count_of_counts(counts))
    N[1] = sum(1 for g, c in counts.items() if c == 1 and tot[g[:-1]] >= 2)
    return N


def loo_objective(D: float, N: dict[int, int]) -> float:
    """Thm 3: the D-dependent part of the leave-one-out log-likelihood of backing-off absolute
    discounting, F(D) = n₁ log D + Σ_{r≥2} r·n_r·log(r − 1 − D)."""
    return N.get(1, 0) * math.log(D) + sum(r * n * math.log(r - 1 - D) for r, n in N.items() if r >= 2)


def loo_discount(N: dict[int, int]) -> float:
    """The unique root of F'(D) = n₁/D − Σ_{r≥2} r n_r/(r − 1 − D) in (0, 1)."""
    def dF(D):
        return N[1] / D - sum(r * n / (r - 1 - D) for r, n in N.items() if r >= 2)

    return float(optimize.brentq(dF, 1e-12, 1 - 1e-12, xtol=1e-15))


def loo_loglik_exact(D: float, counts: Counter, beta: dict[str, float]) -> float:
    """Exact leave-one-out log-likelihood of backing-off absolute discounting with a fixed lower-order
    distribution β: remove each training occurrence in turn and score it with the remaining counts.
        r ≥ 2:              (r − 1 − D)/(c(u·) − 1)
        r = 1, c(u·) ≥ 2:   D·(N₁₊(u•) − 1)/(c(u·) − 1) · β(w)/(1 − S_u + β(w)),  S_u = Σ_{seen w'} β(w')
        r = 1, c(u·) = 1:   β(w)  (the context is gone; full backoff)"""
    tot = context_totals(counts)
    types: Counter = Counter(g[:-1] for g in counts)
    S: Counter = Counter()
    for g in counts:
        S[g[:-1]] += beta[g[-1]]
    ll = 0.0
    for g, r in counts.items():
        u, w = g[:-1], g[-1]
        if r >= 2:
            ll += r * math.log((r - 1 - D) / (tot[u] - 1))
        elif tot[u] >= 2:
            ll += math.log(D * (types[u] - 1) / (tot[u] - 1) * beta[w] / (1 - S[u] + beta[w]))
        else:
            ll += math.log(beta[w])
    return ll


# --------------------------------------------------------------------------------------------
# 6. Kneser–Ney.
# --------------------------------------------------------------------------------------------


def continuation_counts(counts: Counter) -> Counter:
    """N₁₊(• v): for the k-gram counts of ONE order k ≥ 2, the number of distinct words x with
    c(x v) > 0, keyed by the (k−1)-gram suffix v.

    Example: bigram counts {(halvorsen, industrial): 284, (the, plan): 3, (a, plan): 1} give
    {(industrial,): 1, (plan,): 2} — 284 tokens of "industrial" but one context it continues.
    Entries with count 0 must not count (a Counter can hold them). Thm 4 says these are the counts
    the lower orders must use; kn_fit calls this once per lower order.
    """
    out: Counter = Counter()
    for g, c in counts.items():
        if c > 0:
            out[g[1:]] += 1
    return out


def mkn_discounts(N: dict[int, int]) -> tuple[float, float, float]:
    """Chen & Goodman's modified-KN discounts with Y = n₁/(n₁ + 2n₂):
    D₁ = 1 − 2Y n₂/n₁, D₂ = 2 − 3Y n₃/n₂, D₃₊ = 3 − 4Y n₄/n₃."""
    Y = N[1] / (N[1] + 2 * N[2])
    return (1 - 2 * Y * N[2] / N[1], 2 - 3 * Y * N[3] / N[2], 3 - 4 * Y * N[4] / N[3])


@dataclass(frozen=True)
class KNModel:
    """Everything the interpolated recursion reads. Contexts of different orders have different lengths,
    so one dict per statistic holds every order."""

    order: int
    counts: dict[int, Counter]  # order k → k-gram counts KN uses there: raw at k = order, continuation below
    totals: dict[NGram, float]  # c(u·) = Σ_w counts[len(u)+1][u w] for every context u that occurs
    freed: dict[NGram, float]  # Σ_{w: c(u w) > 0} D(c(u w)) — the count mass discounting frees in context u
    discounts: dict[int, tuple[float, ...]]  # order k → (D,) or modified KN's (D₁, D₂, D₃₊)
    vocab_size: int  # |V| of the uniform distribution the unigram level interpolates with

    def discount(self, k: int, c: int) -> float:
        """D_k(c): 0 for an unseen n-gram, else the discount for a count-c n-gram at order k."""
        if c <= 0:
            return 0.0
        ds = self.discounts[k]
        return ds[min(c, len(ds)) - 1]


def kn_fit(sents, order: int = ORDER, discounts="ney", *, continuation: bool = True) -> KNModel:
    """Fit interpolated Kneser–Ney (continuation=True) or plain interpolated absolute discounting
    (continuation=False: raw counts at every order — the one change KN makes). discounts: "ney" (one
    D_k = n₁/(n₁ + 2n₂) per order), "mkn" (Chen–Goodman's three), or an explicit {k: tuple}; the
    count-of-counts are those of the counts each order actually uses."""
    raw = ngram_counts(sents, order)
    counts = {order: raw[order]}
    for k in range(1, order):
        counts[k] = continuation_counts(raw[k + 1]) if continuation else raw[k]
    if discounts == "ney":
        ds = {k: (ney_discount(count_of_counts(c)),) for k, c in counts.items()}
    elif discounts == "mkn":
        ds = {k: mkn_discounts(count_of_counts(c)) for k, c in counts.items()}
    else:
        ds = dict(discounts)
    model = KNModel(order, counts, {}, {}, ds, len(vocabulary()))
    totals: Counter = Counter()
    freed: Counter = Counter()
    for k, cnt in counts.items():
        for g, c in cnt.items():
            totals[g[:-1]] += c
            freed[g[:-1]] += model.discount(k, c)
    model.totals.update(totals)
    model.freed.update(freed)
    return model


def interpolated_kn_prob(word: str, context: NGram, model: KNModel) -> float:
    """Chen–Goodman interpolated Kneser–Ney, recursive in the context length k − 1 = len(context):

        P(w | u) = [ max(c(u w) − D_k(c(u w)), 0) + freed(u) · P(w | u⁻) ] / c(u·)

    where u⁻ = u without its first (oldest) word, c = model.counts[k] (raw at the top order,
    continuation counts below), c(u·) = model.totals[u], freed(u) = model.freed[u], and D_k(c) =
    model.discount(k, c). Two edge cases carry the design:
      • c(u·) = 0 (u never occurred as a context): defer entirely to P(w | u⁻).
      • the empty context u = () is the last level: its "P(w | u⁻)" is the uniform 1/model.vocab_size.
    With one discount D per order, freed(u) = D·N₁₊(u•), the textbook form.
    """
    k = len(context) + 1
    lower = 1.0 / model.vocab_size if k == 1 else interpolated_kn_prob(word, context[1:], model)
    total = model.totals.get(context, 0)
    if not total:
        return lower
    c = model.counts[k].get(context + (word,), 0)
    return (max(c - model.discount(k, c), 0.0) + model.freed[context] * lower) / total


# --------------------------------------------------------------------------------------------
# 7. Kneser–Ney as hierarchical Pitman–Yor.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Seating:
    """A seating arrangement of the hierarchical Pitman–Yor LM (Teh 2006): one Chinese restaurant per
    context u, customers c_uw and tables t_uw per word. A table in restaurant u sends one customer to
    its parent restaurant u⁻ (u without its first word)."""

    order: int
    customers: dict[NGram, Counter]
    tables: dict[NGram, Counter]
    cust_total: dict[NGram, int]
    table_total: dict[NGram, int]


def one_table_seating(sents, order: int = ORDER) -> Seating:
    """Bottom restaurants (contexts of length order − 1) get the raw counts as customers; every word
    sits at exactly one table, t_uw = min(1, c_uw); each table sends one customer up."""
    customers: dict[NGram, Counter] = defaultdict(Counter)
    tables: dict[NGram, Counter] = defaultdict(Counter)
    for g, c in ngram_counts(sents, order)[order].items():
        customers[g[:-1]][g[-1]] += c
    for depth in range(order - 1, -1, -1):
        for u in [u for u in customers if len(u) == depth]:
            for w, c in customers[u].items():
                tables[u][w] = min(1, c)
                if depth:
                    customers[u[1:]][w] += tables[u][w]
    return Seating(order, dict(customers), dict(tables),
                   {u: sum(cu.values()) for u, cu in customers.items()},
                   {u: sum(tu.values()) for u, tu in tables.items()})


def hpylm_prob(word: str, context: NGram, s: Seating, d: dict[int, float], theta: dict[int, float],
               vocab_size: int) -> float:
    """The HPYLM predictive given a seating (Teh 2006, eq. 8), restaurant u with |u| = k − 1:
    P(w|u) = (c_uw − d_k t_uw)/(θ_k + c_u·) + (θ_k + d_k t_u·)/(θ_k + c_u·)·P(w|u⁻), base uniform."""
    k = len(context) + 1
    parent = 1.0 / vocab_size if k == 1 else hpylm_prob(word, context[1:], s, d, theta, vocab_size)
    cu = s.customers.get(context)
    if not cu:
        return parent
    c_dot, t_dot = s.cust_total[context], s.table_total[context]
    return ((cu[word] - d[k] * s.tables[context][word]) + (theta[k] + d[k] * t_dot) * parent) / (theta[k] + c_dot)


HPYLM_SWEEPS, HPYLM_BURN_IN = 40, 20  # Gibbs sweeps over every training token; predictive averaged after burn-in
_D_GRID = np.linspace(0.025, 0.975, 39)
_THETA_GRID = np.geomspace(1e-3, 50.0, 34)


class _Uniforms:
    """Uniform draws from ONE numpy stream, handed out in blocks (Python-speed sampling, A1's stream rule)."""

    def __init__(self, rng: np.random.Generator, block: int = 1 << 18) -> None:
        self.rng, self.block, self.buf, self.i = rng, block, [], 0

    def __call__(self) -> float:
        if self.i == len(self.buf):
            self.buf, self.i = self.rng.random(self.block).tolist(), 0
        self.i += 1
        return self.buf[self.i - 1]


class HPYLMSampler:
    """Teh's (2006) Gibbs sampler for the hierarchical Pitman–Yor LM in the Chinese-restaurant
    representation: every training token is a customer in its bottom restaurant; re-seating it may open
    or close a table, which adds or removes a customer one restaurant up. (d_k, θ_k) are resampled per
    order from their grid posterior (Beta(1, 1) on d, Gamma(1, 1) on θ) given the seating."""

    def __init__(self, sents, order: int = ORDER, *, seed=(SEED, 13)) -> None:
        self.order, self.V = order, len(vocabulary())
        self.rng = np.random.default_rng(list(seed))
        self.u01 = _Uniforms(self.rng)
        raw = ngram_counts(sents, order)
        self.customers = [(g[:-1], g[-1]) for g in events(sents, order)]
        self.tables: dict[tuple[NGram, str], list[int]] = {}
        self.cw: Counter = Counter()  # customers of w in restaurant u
        self.tw: Counter = Counter()  # tables of w in restaurant u
        self.cu: Counter = Counter()
        self.tu: Counter = Counter()
        # Start from the one-table-per-type seating — the state in which the predictive is Kneser–Ney.
        seat = one_table_seating(sents, order)
        for u, cust in seat.customers.items():
            for w, c in cust.items():
                self.tables[(u, w)] = [c]
                self.cw[(u, w)], self.tw[(u, w)] = c, 1
                self.cu[u] += c
                self.tu[u] += 1
        assert sum(self.cw[(g[:-1], g[-1])] for g in raw[order]) == len(self.customers)
        self.d = {k: ney_discount(count_of_counts(c)) for k, c in kn_fit(sents, order).counts.items()}
        self.theta = {k: 1.0 for k in range(1, order + 1)}

    def prob(self, w: str, u: NGram) -> float:
        k = len(u) + 1
        parent = 1.0 / self.V if k == 1 else self.prob(w, u[1:])
        c = self.cu.get(u, 0)
        if not c:
            return parent
        d, th = self.d[k], self.theta[k]
        return (self.cw.get((u, w), 0) - d * self.tw.get((u, w), 0) + (th + d * self.tu[u]) * parent) / (th + c)

    def _add(self, w: str, u: NGram) -> None:
        k = len(u) + 1
        d, th = self.d[k], self.theta[k]
        tabs = self.tables.setdefault((u, w), [])
        parent = 1.0 / self.V if k == 1 else self.prob(w, u[1:])
        new = (th + d * self.tu[u]) * parent
        x = self.u01() * (self.cw[(u, w)] - d * len(tabs) + new)
        for i, s in enumerate(tabs):
            x -= s - d
            if x < 0:
                tabs[i] += 1
                break
        else:  # a new table, which sends a customer to the parent restaurant
            tabs.append(1)
            self.tw[(u, w)] += 1
            self.tu[u] += 1
            if k > 1:
                self._add(w, u[1:])
        self.cw[(u, w)] += 1
        self.cu[u] += 1

    def _remove(self, w: str, u: NGram) -> None:
        k = len(u) + 1
        tabs = self.tables[(u, w)]
        x = self.u01() * self.cw[(u, w)]
        for i, s in enumerate(tabs):
            x -= s
            if x < 0:
                break
        tabs[i] -= 1
        if tabs[i] == 0:  # the table closes and its customer leaves the parent restaurant
            tabs.pop(i)
            self.tw[(u, w)] -= 1
            self.tu[u] -= 1
            if k > 1:
                self._remove(w, u[1:])
        self.cw[(u, w)] -= 1
        self.cu[u] -= 1

    def sweep(self) -> None:
        for i in self.rng.permutation(len(self.customers)):
            u, w = self.customers[i]
            self._remove(w, u)
            self._add(w, u)
        for k in range(1, self.order + 1):
            self._resample_hyper(k)

    def _resample_hyper(self, k: int) -> None:
        """Draw (d_k, θ_k) from the grid posterior ∝ prior × Π_u EPPF(seating of u | d, θ):
        Π_{i<t_u}(θ + i d) / Π_{i<c_u}(θ + i) · Π_tables Π_{j<size}(j − d)."""
        rest = Counter((self.tu[u], self.cu[u]) for u in self.cu if len(u) == k - 1 and self.cu[u] > 0)
        sizes = Counter(s for (u, _), tabs in self.tables.items() if len(u) == k - 1 for s in tabs)
        t, c, nr = (np.array(x, float) for x in zip(*((t, c, n) for (t, c), n in rest.items())))
        s, ns = (np.array(x, float) for x in zip(*sizes.items()))
        D, TH = np.meshgrid(_D_GRID, _THETA_GRID, indexing="ij")
        D, TH = D[..., None], TH[..., None]
        ll = np.sum(nr * ((t - 1) * np.log(D) + special.gammaln(TH / D + t) - special.gammaln(TH / D + 1)
                          - special.gammaln(TH + c) + special.gammaln(TH + 1)), axis=-1)
        ll += np.sum(ns * (special.gammaln(s - D) - special.gammaln(1 - D)), axis=-1)
        ll += -_THETA_GRID[None, :] + np.log(_THETA_GRID[None, :])  # Gamma(1,1) prior × log-grid Jacobian
        p = np.exp(ll - ll.max()).ravel()
        j = int(np.searchsorted(np.cumsum(p), self.u01() * p.sum(), side="right"))
        self.d[k], self.theta[k] = float(_D_GRID[j // len(_THETA_GRID)]), float(_THETA_GRID[j % len(_THETA_GRID)])

    def mean_tables_by_count(self, max_r: int = 10) -> list[float]:
        """Mean tables per (context, word) among bottom-restaurant pairs with c_uw = r, r = 1, …, max_r."""
        by_r: dict[int, list[int]] = defaultdict(list)
        for (u, w), c in self.cw.items():
            if len(u) == self.order - 1 and 1 <= c <= max_r:
                by_r[c].append(self.tw[(u, w)])
        return [float(np.mean(by_r[r])) for r in range(1, max_r + 1)]


EFFECTIVE_R = 8  # Panel B's counts; the sampled curve is seed-stable to two decimals through r = 6


@lru_cache(maxsize=1)
def hpylm_run() -> dict:
    """HPYLM_SWEEPS sweeps from the Kneser–Ney seating; after burn-in, average the predictive on both
    test sets and record d_k, θ_k, and the effective top-order discount d·E[t_uw | c_uw = r]."""
    smp = HPYLMSampler(corpus()["train"])
    tests = {name: heldout_events(name) for name in ("held", "edgar")}
    probs: dict[str, list] = {name: [] for name in tests}
    eff, hyper = [], []
    for sweep in range(HPYLM_SWEEPS):
        smp.sweep()
        if sweep >= HPYLM_BURN_IN:
            for name, evs in tests.items():
                probs[name].append([smp.prob(g[-1], g[:-1]) for g in evs])
            eff.append([smp.d[ORDER] * t for t in smp.mean_tables_by_count(EFFECTIVE_R)])
            hyper.append((dict(smp.d), dict(smp.theta)))
    return {"logloss": {name: float(-np.mean(np.log(np.mean(p, axis=0)))) for name, p in probs.items()},
            "effective_discount": np.mean(eff, axis=0).tolist(),
            "d": {k: float(np.mean([h[0][k] for h in hyper])) for k in range(1, ORDER + 1)},
            "theta": {k: float(np.mean([h[1][k] for h in hyper])) for k in range(1, ORDER + 1)},
            "sampler": smp}


# --------------------------------------------------------------------------------------------
# The smoothers side by side.
# --------------------------------------------------------------------------------------------

SMOOTHERS = ("mle", "laplace", "dirichlet", "katz", "absdisc", "ikn", "mkn")


def rediscount(model: KNModel, discounts: dict[int, tuple[float, ...]]) -> KNModel:
    """The same counts with new discounts (only freed(u) changes)."""
    out = KNModel(model.order, model.counts, model.totals, {}, dict(discounts), model.vocab_size)
    freed: Counter = Counter()
    for k, cnt in model.counts.items():
        for g, c in cnt.items():
            freed[g[:-1]] += out.discount(k, c)
    out.freed.update(freed)
    return out


@lru_cache(maxsize=1)
def kn_models() -> dict[str, KNModel]:
    """Absolute discounting, interpolated KN (one Ney D per order), modified KN (three per order)."""
    tr = corpus()["train"]
    return {"absdisc": kn_fit(tr, continuation=False), "ikn": kn_fit(tr), "mkn": kn_fit(tr, discounts="mkn")}


@lru_cache(maxsize=32)
def single_d_model(name: str, D: float) -> KNModel:
    """Absolute discounting or interpolated KN with ONE discount D at every order (the lab's slider)."""
    return rediscount(kn_models()[name], {k: (D,) for k in range(1, ORDER + 1)})


def prob_fn(name: str, D: float | None = None):
    """prob(word, context) for a smoother by name; D (abs. discounting / IKN only) overrides Ney's."""
    if name == "mle":
        return mle_prob
    if name == "laplace":
        return additive_prob
    if name == "dirichlet":
        alphas, V = type2_alphas(), len(vocabulary())
        return lambda w, u: float(dirichlet_probs(level_arrays([u + (w,)]), alphas, V)[0])
    if name == "katz":
        m = katz_model()
        return lambda w, u: katz_prob(w, u, m)
    m = kn_models()[name] if D is None else single_d_model(name, D)
    return lambda w, u: interpolated_kn_prob(w, u, m)


@lru_cache(maxsize=1)
def smoother_table() -> dict[str, dict[str, float]]:
    """Held-out log-loss (nats/token) of every smoother on the in-domain and EDGAR test sets; ∞ where a
    model gives some test event probability 0 (Katz's zero-reserve contexts, the MLE)."""
    out: dict[str, dict[str, float]] = {}
    for name in SMOOTHERS:
        out[name] = {}
        for test in ("held", "edgar"):
            evs = heldout_events(test)
            out[name][test] = (dirichlet_log_loss(type2_alphas(), test) if name == "dirichlet"
                               else log_loss(prob_fn(name), evs))
    out["dirichlet_oracle"] = {"held": dirichlet_log_loss(heldout_optimal_alphas())}
    out["hpylm"] = dict(hpylm_run()["logloss"])
    return out


# --------------------------------------------------------------------------------------------
# 8. Finance case study.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def continuation_run(n_more: int = 100) -> dict[str, tuple]:
    """The training corpus's own generator, run on for n_more documents: the first 300 documents are
    generate_corpus(300, 0) exactly; the next n_more continue the same Pitman–Yor restaurants. That is
    the text Good–Turing's N₁/N forecasts — a fresh generate_corpus(100, 1) restarts the restaurants from
    the same seed nouns, and EDGAR is another source altogether."""
    gen = fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    first = tuple(gen.doc(i, int(t)).text for i, t in enumerate(tones))
    more_tones = gen.rng.permutation(np.resize([-1, 0, 1], n_more))
    more = [gen.doc(300 + i, int(t)) for i, t in enumerate(more_tones)]
    return {"first": first, "continuation": tuple(s for d in more for s in split_sentences(d.text))}


def good_turing_unseen_mass() -> float:
    """N₁/N on the training tokens (words and </s>): Good–Turing's mass for word types not yet seen."""
    c1 = train_counts()[1]
    return sum(1 for n in c1.values() if n == 1) / sum(c1.values())


def oov_share(sents) -> float:
    """Share of predicted tokens (words and </s>) whose type never occurs in training."""
    toks = [w for s in sents for w in (*s, EOS)]
    return float(np.mean([w not in vocabulary() for w in toks]))


def raw_vs_continuation(min_count: int = 100, top: int = 6) -> list[tuple[str, int, int]]:
    """Frequent words that continue very few distinct contexts: (word, raw count, N₁₊(• w))."""
    raw, cont = train_counts()[1], continuation_counts(train_counts()[2])
    rows = [(g[0], n, cont[g]) for g, n in raw.items() if n >= min_count]
    return sorted(rows, key=lambda r: (r[2] / r[1], -r[1]))[:top]


def unigram_rank(word: str, model: KNModel) -> int:
    """1-based rank of word in the model's unigram level P(· | ())."""
    V = sorted(vocabulary())
    p = {w: interpolated_kn_prob(w, (), model) for w in V}
    return 1 + sum(q > p[word] for q in p.values())


@lru_cache(maxsize=1)
def finance_summary() -> dict:
    ab, kn = kn_models()["absdisc"], kn_models()["ikn"]
    word = "industrial"
    return {
        "halvorsen_raw": train_counts()[1][(word,)],
        "halvorsen_continuation": continuation_counts(train_counts()[2])[(word,)],
        "p_unigram_absdisc": interpolated_kn_prob(word, (), ab),
        "p_unigram_ikn": interpolated_kn_prob(word, (), kn),
        "rank_absdisc": unigram_rank(word, ab), "rank_ikn": unigram_rank(word, kn),
        "gt_unseen_mass": good_turing_unseen_mass(),
        "oov_continuation": oov_share(continuation_run()["continuation"]),
        "oov_fresh_run": oov_share(corpus()["held"]),
        "oov_edgar": oov_share(corpus()["edgar"]),
        "edgar_unseen_shares": unseen_shares("edgar"),
    }


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX's JSON).
# --------------------------------------------------------------------------------------------

LAB_D = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # the shared slider: one D at every order
# (Thm 4 needs D ≤ 1; held-out log-loss bottoms out near D = 0.9 for both models, so 1.0 shows the turn)
LAB_CONTEXTS = (
    ("within", "industrials"),  # in-domain: 26 tokens, 7 follower types (company names)
    ("machinery", "as"),  # seen once
    ("in", "osaka"),  # 7 tokens, always "</s>": Katz reserves nothing
    ("and", "components"),  # unseen trigram context; the bigram "components ·" is seen
    ("the", UNK),  # after an unknown word: only the unigram level is left
)
LAB_TOP = 8
LAB_WATCH = "industrial"  # 284 tokens, one context: the Halvorsen effect


@lru_cache(maxsize=1)
def _followers() -> dict[NGram, frozenset[str]]:
    by: dict[NGram, set] = defaultdict(set)
    for g in train_counts()[ORDER]:
        by[g[:-1]].add(g[-1])
    return {u: frozenset(ws) for u, ws in by.items()}


def context_distribution(name: str, u: NGram, D: float | None = None) -> dict[str, float] | None:
    """P(· | u) over V for one smoother (None for the MLE in an unseen context, where it is undefined)."""
    V = sorted(vocabulary())
    if name == "mle" and not train_totals()[ORDER].get(u):
        return None
    if name == "dirichlet":
        p = dirichlet_probs(level_arrays([u + (w,) for w in V]), type2_alphas(), len(V))
        return dict(zip(V, p.tolist()))
    f = prob_fn(name, D)
    return {w: f(w, u) for w in V}


def _summarize(p: dict[str, float] | None, u: NGram) -> dict | None:
    """Top LAB_TOP words as [word, probability, training count c(u w)] (0 = never seen after u), then the
    rest split into seen and unseen mass, the unseen mass in all, and the watch word's [probability, rank]."""
    if p is None:
        return None
    seen, c3 = _followers().get(u, frozenset()), train_counts()[ORDER]
    order = sorted(p, key=lambda w: (-p[w], w))
    top = order[:LAB_TOP]
    rest = order[LAB_TOP:]
    return {"top": [[w, round(p[w], 6), c3.get(u + (w,), 0)] for w in top],
            "rest_seen": round(sum(p[w] for w in rest if w in seen), 6),
            "rest_unseen": round(sum(p[w] for w in rest if w not in seen), 6),
            "unseen_mass": round(sum(q for w, q in p.items() if w not in seen), 6),
            "watch": [round(p[LAB_WATCH], 8), 1 + sum(q > p[LAB_WATCH] for q in p.values())]}  # ties share a rank


def _r(x: float, nd: int = 4):
    return None if not math.isfinite(x) else round(x, nd)


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    tot, table = train_totals()[ORDER], smoother_table()
    contexts = []
    for u in LAB_CONTEXTS:
        fixed = {name: _summarize(context_distribution(name, u), u) for name in ("mle", "laplace", "dirichlet",
                                                                                    "katz", "mkn")}
        by_d = {name: [_summarize(context_distribution(name, u, D), u) for D in LAB_D]
                for name in ("absdisc", "ikn")}
        contexts.append({"context": list(u), "count": tot.get(u, 0), "types": len(_followers().get(u, ())),
                         "fixed": fixed, "by_d": by_d})
    held = heldout_events()
    zeros = set(katz_zero_events())
    N3 = count_of_counts(train_counts()[ORDER])
    raw, sgt = turing_rstar(N3), simple_good_turing(N3)
    rs = list(range(1, EFFECTIVE_R + 1))
    return {
        "D": list(LAB_D), "watch": LAB_WATCH, "contexts": contexts,
        "logloss": {**{name: _r(table[name]["held"]) for name in SMOOTHERS},
                    "katz_finite": _r(log_loss(prob_fn("katz"), [g for g in held if g not in zeros])),
                    "katz_zero_events": len(zeros), "hpylm": _r(table["hpylm"]["held"]),
                    "by_d": {name: [_r(log_loss(prob_fn(name, D), held)) for D in LAB_D]
                             for name in ("absdisc", "ikn")}},
        "discount": {"r": rs, "gt": [_r(r - raw[r]) for r in rs], "sgt": [_r(r - sgt.rstar(r)) for r in rs],
                     "ney": _r(kn_models()["ikn"].discounts[ORDER][0]),
                     "mkn": [_r(x) for x in kn_models()["mkn"].discounts[ORDER]],
                     "hpylm": [_r(x) for x in hpylm_run()["effective_discount"]]},
        "spectrum": {"r": list(range(1, 31)), "n": [N3.get(r, 0) for r in range(1, 31)]},
    }


# --------------------------------------------------------------------------------------------
# Summary — the numbers the page quotes.
# --------------------------------------------------------------------------------------------


def summary() -> dict[str, float]:
    c, held = train_counts(), heldout_events()
    sh = unseen_shares()
    out: dict[str, float] = {"train_tokens": sum(c[1].values()), "train_sentences": len(corpus()["train"]),
                             "V": len(vocabulary()), "held_events": len(held),
                             "edgar_events": len(heldout_events("edgar"))}
    for k in range(1, ORDER + 1):
        out[f"unseen_share_{k}"] = sh[k]
        out[f"ney_D_{k}"] = ney_discount(count_of_counts(c[k]))
    for k, a in enumerate(type2_alphas(), 1):
        out[f"type2_alpha_{k}"] = a
    for k, a in enumerate(heldout_optimal_alphas(), 1):
        out[f"oracle_alpha_{k}"] = a
    for k in (2, 3):
        out[f"loo_D_{k}"] = loo_discount(loo_spectrum(c[k]))
    zeros = set(katz_zero_events())
    out["katz_zero_events"] = len(zeros)
    out["logloss_katz_finite_events"] = log_loss(prob_fn("katz"), [g for g in held if g not in zeros])
    for name, row in smoother_table().items():
        for test, v in row.items():
            out[f"logloss_{name}_{test}"] = v
    h = hpylm_run()
    for k in range(1, ORDER + 1):
        out[f"hpylm_d_{k}"], out[f"hpylm_theta_{k}"] = h["d"][k], h["theta"][k]
    for k, v in finance_summary().items():
        if not isinstance(v, dict):
            out[f"fin_{k}"] = v
    return out


# --------------------------------------------------------------------------------------------
# Verification harness — every claim the page makes.
# --------------------------------------------------------------------------------------------


def _random_corpus(rng: np.random.Generator, n_sents: int = 60, vocab: int = 12) -> list[NGram]:
    """Tiny Zipfian corpora for exhaustive checks (word i drawn ∝ 1/(i+1))."""
    p = zhv.zipf_pmf(1.0, vocab)
    words = np.array([f"w{i}" for i in range(vocab)])
    return [tuple(words[rng.choice(vocab, size=int(rng.integers(1, 8)), p=p)]) for _ in range(n_sents)]


def test_counts_are_marginals():
    """ngram_counts' orders are marginals of the top-order events: c(u·) at order k = Σ_x c(x u ·)."""
    c, tot = train_counts(), train_totals()
    for k in range(1, ORDER):
        up: Counter = Counter()
        for g, n in c[k + 1].items():
            up[g[1:]] += n
        assert up == c[k]
        assert sum(tot[k].values()) == sum(c[1].values())


def test_mle_maximizes_likelihood():
    """Prop 1: Σ_w c(uw) log P̂(w|u) − Σ_w c(uw) log q(w) = c(u·)·KL(P̂ ‖ q) ≥ 0 for every q on the simplex."""
    rng = np.random.default_rng([SEED, 1])
    c3, tot = train_counts()[3], train_totals()[3]
    by_ctx: dict[NGram, dict[str, int]] = defaultdict(dict)
    for g, n in c3.items():
        by_ctx[g[:-1]][g[-1]] = n
    ctxs = [u for u in by_ctx if len(by_ctx[u]) >= 3]
    for i in rng.choice(len(ctxs), 50, replace=False):
        u = ctxs[i]
        cnt = np.array(list(by_ctx[u].values()), float)
        phat = cnt / cnt.sum()
        for _ in range(5):
            q = rng.dirichlet(np.ones(len(cnt)))
            gap = cnt @ np.log(phat) - cnt @ np.log(q)
            assert gap >= 0 and abs(gap - tot[u] * np.sum(phat * np.log(phat / q))) < 1e-8


def test_zero_frequency_catastrophe():
    """Prop 2: the MLE's held-out log-loss is ∞ iff some held-out n-gram is unseen — and most are."""
    sh = unseen_shares()
    assert all(0 < sh[k] < 1 for k in sh) and sh[1] < sh[2] < sh[3]
    assert log_loss(mle_prob, heldout_events()) == math.inf
    seen = [g for g in heldout_events() if g in train_counts()[ORDER]]
    assert math.isfinite(log_loss(mle_prob, seen))


def test_dirichlet_is_interpolation():
    """Prop 3: (c + α P_lower)/(c(u·) + α) = (1 − λ_u)·P_ML + λ_u·P_lower with λ_u = α/(c(u·) + α)."""
    alphas, V = type2_alphas(), len(vocabulary())
    held = list(heldout_events())[:3000]
    levels = level_arrays(held)
    p3, p2 = dirichlet_probs(levels, alphas, V), dirichlet_probs(levels[:2], alphas[:2], V)
    c, t = levels[2]
    lam = alphas[2] / (t + alphas[2])
    pml = np.divide(c, t, out=np.zeros_like(c), where=t > 0)
    assert np.allclose(p3, (1 - lam) * pml + lam * p2, rtol=0, atol=1e-14)
    assert np.all(lam[t == 0] == 1)  # an unseen context is all prior


def test_evidence_matches_scipy():
    """Thm 1 against scipy.stats.dirichlet_multinomial (which includes the multinomial coefficient)."""
    rng = np.random.default_rng([SEED, 2])
    for _ in range(30):
        V = int(rng.integers(2, 9))
        m, alpha = rng.dirichlet(np.ones(V)), float(rng.uniform(0.1, 50))
        x = rng.multinomial(int(rng.integers(1, 40)), rng.dirichlet(np.ones(V)))
        ours = dirichlet_log_evidence(alpha, x[x > 0].astype(float), m[x > 0], np.array([x.sum()], float))
        coef = special.gammaln(x.sum() + 1) - special.gammaln(x + 1).sum()
        assert abs(ours + coef - stats.dirichlet_multinomial.logpmf(x, alpha * m, x.sum())) < 1e-9


def test_type2_alpha_is_a_maximum():
    """Type-II α's are interior maxima of the evidence, and each beats Laplace's prior weight α = |V|."""
    alphas = type2_alphas()
    for k in range(1, ORDER + 1):
        c, m, tot = _evidence_terms(k, alphas[: k - 1])
        e = dirichlet_log_evidence(alphas[k - 1], c, m, tot)
        for f in (0.9, 1.1):
            assert e > dirichlet_log_evidence(alphas[k - 1] * f, c, m, tot)
    assert dirichlet_log_loss(alphas) < log_loss(additive_prob, heldout_events())


def test_good_turing_identity_exact():
    """Thm 2: E[M_r(n)] = (r+1)/(n+1)·E[N_{r+1}(n+1)], exactly, for any p and n."""
    rng = np.random.default_rng([SEED, 3])
    for _ in range(40):
        V = int(rng.integers(2, 200))
        p = rng.dirichlet(np.full(V, rng.uniform(0.05, 2)))
        n = int(rng.integers(1, 300))
        for r in range(0, min(n, 6)):
            lhs, rhs = expected_mass(p, n, r), (r + 1) / (n + 1) * expected_spectrum(p, n + 1, r + 1)
            assert abs(lhs - rhs) <= 1e-12 * max(1.0, lhs), (n, r, lhs, rhs)


def test_missing_mass_bias_bound():
    """Cor: 0 ≤ E[N₁(n)]/n − E[M₀(n)] ≤ 1/n — exactly on random p, and in simulation from a Zipf source."""
    rng = np.random.default_rng([SEED, 4])
    for _ in range(40):
        p = rng.dirichlet(np.full(int(rng.integers(2, 500)), rng.uniform(0.05, 2)))
        n = int(rng.integers(1, 2000))
        bias = expected_spectrum(p, n, 1) / n - expected_mass(p, n, 0)
        assert -1e-15 <= bias <= 1 / n
    p, n, reps = zhv.zipf_pmf(1.1, 5000), 2000, 400
    est, miss = [], []
    for _ in range(reps):
        c = np.bincount(rng.choice(len(p), size=n, p=p), minlength=len(p))
        est.append(np.sum(c == 1) / n)
        miss.append(p[c == 0].sum())
    exact = expected_spectrum(p, n, 1) / n - expected_mass(p, n, 0)
    se = np.std(np.subtract(est, miss), ddof=1) / math.sqrt(reps)
    assert abs(np.mean(est) - np.mean(miss) - exact) < 4 * se and 0 <= exact <= 1 / n


def test_good_turing_is_robbins():
    """Robbins' formula is Turing's r*: run on Topic 33's insurance table, our r* reproduces its row; and
    under Poissonization E[λ | c = r] over the types is exactly (r+1)·E[N_{r+1}]/E[N_r]."""
    rstar = turing_rstar(dict(enumerate(CASI_CLAIMS)))
    for x, printed in enumerate(CASI_ROBBINS):
        assert round(rstar[x], 3 if x < 3 else 2) == printed, (x, rstar[x])
    rng = np.random.default_rng([SEED, 5])
    for _ in range(20):
        lam = rng.dirichlet(np.full(300, 0.3)) * rng.integers(50, 5000)
        for r in range(6):
            f = stats.poisson.pmf(r, lam)
            post = lam @ f / f.sum()
            assert abs(post - (r + 1) * stats.poisson.pmf(r + 1, lam).sum() / f.sum()) < 1e-10 * max(1, post)


def test_sgt_matches_nltk():
    """Our Simple Good–Turing equals nltk's on the bigram spectrum, seen probabilities and unseen mass."""
    from nltk.probability import FreqDist, SimpleGoodTuringProbDist

    c2 = train_counts()[2]
    ours = simple_good_turing(count_of_counts(c2))
    theirs = SimpleGoodTuringProbDist(FreqDist(dict(c2)), bins=len(c2) + 1)
    by_r: dict[int, NGram] = {}
    for g, r in c2.items():
        by_r.setdefault(r, g)
    for r, g in sorted(by_r.items()):
        assert abs(ours.prob(r) - theirs.prob(g)) < 1e-15, (r, ours.prob(r), theirs.prob(g))
    assert abs(ours.prob(0) - theirs.prob(("never", "seen"))) < 1e-15
    total = sum(n * ours.prob(r) for r, n in ours.N.items()) + ours.prob(0)
    assert abs(total - 1) < 1e-12


def test_katz_normalizes():
    """Prop 4: Σ_{w ∈ V} P_katz(w | u) = 1 for seen, partly seen, and unseen contexts."""
    m, V = katz_model(), sorted(vocabulary())
    rng = np.random.default_rng([SEED, 6])
    ctxs = list(train_totals()[3])
    picks = [ctxs[i] for i in rng.choice(len(ctxs), 12, replace=False)]
    picks += [("halvorsen", "industrial"), (UNK, "the"), ("zz", "zz"), (BOS, BOS)]
    for u in picks:
        assert abs(sum(katz_prob(w, u, m) for w in V) - 1) < 1e-9, u
    assert all(0 < d < 1 for dk in m.d.values() for d in dk.values())


def katz_zero_events() -> list[NGram]:
    """Held-out events Katz gives probability 0: their context's followers all occurred more than
    KATZ_CUTOFF times, so nothing was discounted and it reserved no mass — α = 0 from the cutoff, or
    α = 0/0 in an exhausted context kept at its MLE. The zero-count problem, back through the cutoff."""
    m = katz_model()
    return [g for g in heldout_events() if katz_prob(g[-1], g[:-1], m) <= 0]


def test_katz_cutoff_reintroduces_zeros():
    m, zeros = katz_model(), katz_zero_events()
    assert sum(a == 0 for a in m.alpha.values()) > 1000  # templated contexts reserve nothing
    assert 0 < len(zeros) < 10 and log_loss(lambda w, u: katz_prob(w, u, m), heldout_events()) == math.inf
    assert all(m.alpha[g[:-1]] == 0 for g in zeros)
    # The page's breakdown: 2,170 zero-reserve contexts = 564 from the cutoff (α = 0) + 1,606 exhausted
    # trigram contexts (α = 0/0, kept at the MLE), 1,289 of which would free mass under discounting.
    zero = {u for u, a in m.alpha.items() if a == 0}
    assert (len(m.alpha), len(zero), len(zero - m.exhausted), len(m.exhausted)) == (17269, 2170, 564, 1606)
    assert all(len(u) == 2 for u in m.exhausted)
    followers: dict[NGram, list[int]] = defaultdict(list)
    for g, c in m.counts[ORDER].items():
        followers[g[:-1]].append(c)
    frees = sum(sum(m.d[ORDER].get(c, 1.0) * c for c in followers[u]) < m.totals[ORDER][u] for u in m.exhausted)
    assert frees == 1289
    assert {g[:-1]: g[:-1] in m.exhausted for g in zeros} == {
        ("across", "aerospace"): False, ("in", "osaka"): True, ("of", "fairhaven"): False}


def test_loo_concave_and_bound():
    """Thm 3: F is strictly concave on (0, 1) with a unique maximizer D* ≤ n₁/(n₁ + 2n₂) — on every
    corpus order and on random spectra."""
    rng = np.random.default_rng([SEED, 7])
    spectra = [loo_spectrum(train_counts()[k]) for k in (2, 3)]
    for _ in range(200):
        N = {r: int(rng.integers(1, 5000)) for r in range(1, int(rng.integers(3, 12)))}
        spectra.append(N)
    grid = np.linspace(0.01, 0.99, 99)
    for N in spectra:
        F = np.array([loo_objective(D, N) for D in grid])
        assert np.all(np.diff(F, 2) < 0)  # strictly concave on the grid
        D = loo_discount(N)
        assert 0 < D <= N[1] / (N[1] + 2 * N[2]) + 1e-12
        assert loo_objective(D, N) >= F.max() - 1e-9


def test_loo_objective_is_exact():
    """Thm 3's F(D) differs from the exact leave-one-out log-likelihood by a constant (the derivation)."""
    c2 = train_counts()[2]
    c1, V = train_counts()[1], vocabulary()
    beta = {w: (c1.get((w,), 0) + 1) / (sum(c1.values()) + len(V)) for w in V}
    N = loo_spectrum(c2)
    gaps = [loo_loglik_exact(D, c2, beta) - loo_objective(D, N) for D in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert np.ptp(gaps) < 1e-6 * abs(gaps[0]), gaps


def _brute_continuation(counts: Counter) -> Counter:
    before: dict[NGram, set] = defaultdict(set)
    for g, c in counts.items():
        if c > 0:
            before[g[1:]].add(g[0])
    return Counter({v: len(xs) for v, xs in before.items()})


def test_continuation_counts_bruteforce():
    rng = np.random.default_rng([SEED, 8])
    for _ in range(30):
        counts = ngram_counts(_random_corpus(rng), 3)
        for k in (2, 3):
            cnt = Counter(counts[k])
            cnt[("w0",) * k] = 0  # an explicit zero must not count
            assert continuation_counts(cnt) == _brute_continuation(cnt)
    got = continuation_counts(train_counts()[2])
    assert got[("industrial",)] == 1 and train_counts()[1][("industrial",)] == 284


def _kn_test_contexts(rng) -> list[NGram]:
    ctxs = list(train_totals()[3])
    return ([ctxs[i] for i in rng.choice(len(ctxs), 10, replace=False)]
            + [("halvorsen", "industrial"), (UNK, "the"), ("zz", "zz"), (BOS, BOS), ("the", UNK)])


def test_ikn_normalizes():
    """Σ_{w ∈ V} P(w | u) = 1 for interpolated KN (Ney and modified discounts) and for absolute
    discounting, at every order, for seen and unseen contexts."""
    rng = np.random.default_rng([SEED, 9])
    V, sents = sorted(vocabulary()), corpus()["train"]
    for model in (kn_fit(sents), kn_fit(sents, discounts="mkn"), kn_fit(sents, continuation=False)):
        for u in _kn_test_contexts(rng):
            for ctx in (u, u[1:], ()):
                assert abs(sum(interpolated_kn_prob(w, ctx, model) for w in V) - 1) < 1e-9, ctx


def test_kn_marginal_constraint():
    """Thm 4 (one discount per order): with the lower order left undiscounted, interpolated KN preserves
    the empirical marginals — Σ_v c(v·) P(w | v) = c(w) (bigram) and Σ_x c(x v ·) P(w | x v) = c(v w)
    (trigram) — exactly."""
    rng = np.random.default_rng([SEED, 10])
    sents, c, tot = corpus()["train"], train_counts(), train_totals()
    bi = kn_fit(sents, 2, discounts={2: (0.75,), 1: (0.0,)})
    words = [g[0] for g in c[1]]
    for i in rng.choice(len(words), 25, replace=False):
        w = words[i]
        got = sum(t * interpolated_kn_prob(w, v, bi) for v, t in tot[2].items())
        assert abs(got - c[1][(w,)]) < 1e-8 * c[1][(w,)], w
    tri = kn_fit(sents, 3, discounts={3: (0.75,), 2: (0.0,), 1: (0.0,)})
    by_v: dict[str, list[NGram]] = defaultdict(list)
    for xv in tot[3]:
        by_v[xv[1]].append(xv)
    pairs = list(c[2])
    for i in rng.choice(len(pairs), 25, replace=False):
        v, w = pairs[i]
        got = sum(tot[3][xv] * interpolated_kn_prob(w, xv, tri) for xv in by_v[v])
        assert abs(got - c[2][(v, w)]) < 1e-8 * c[2][(v, w)], (v, w)


def test_marginals_need_discount_weighted_lower_order():
    """Thm 4, general form: with count-dependent discounts the marginal-preserving lower order is
    β(w | v) ∝ Σ_x D(c(x v w)), not the continuation count — so modified KN's lower orders (Chen–Goodman
    use continuation counts) do not preserve marginals exactly. For one D the two coincide."""
    c, tot = train_counts(), train_totals()
    D = mkn_discounts(count_of_counts(c[2]))

    def disc(n):
        return D[min(n, 3) - 1]

    weighted: Counter = Counter()
    for (v, w), n in c[2].items():
        weighted[(w,)] += disc(n)
    freed_total = sum(weighted.values())
    beta = {w: x / freed_total for (w,), x in weighted.items()}
    cont = _brute_continuation(c[2])
    cont_total = sum(cont.values())
    worst_weighted = worst_cont = 0.0
    # Σ_v c(v·) P(w|v) = Σ_v (c(vw) − D(c(vw))) + freed_total·β(w)  (algebra of the interpolated form)
    kept: Counter = Counter()
    for (v, w), n in c[2].items():
        kept[(w,)] += n - disc(n)
    for (w,), cw in c[1].items():
        worst_weighted = max(worst_weighted, abs(kept[(w,)] + freed_total * beta[w] - cw) / cw)
        worst_cont = max(worst_cont, abs(kept[(w,)] + freed_total * cont[(w,)] / cont_total - cw) / cw)
    assert worst_weighted < 1e-12 and worst_cont > 0.01, (worst_weighted, worst_cont)
    assert abs(sum(tot[2].values()) - sum(c[1].values())) == 0


def test_ikn_matches_nltk():
    """Our recursion equals nltk.lm.KneserNeyInterpolated when configured as nltk implements it: one
    discount at every order and an undiscounted continuation unigram (D₁ = 0), fed our own n-grams."""
    from nltk.lm import KneserNeyInterpolated

    sents, D = corpus()["train"], 0.75
    ours = kn_fit(sents, ORDER, discounts={1: (0.0,), **{k: (D,) for k in range(2, ORDER + 1)}})
    lm = KneserNeyInterpolated(ORDER, discount=D)
    lm.fit([[g[ORDER - k :] for g in events([s], ORDER) for k in range(1, ORDER + 1)] for s in sents],
           vocabulary_text=[w for s in sents for w in (BOS, *s, EOS)])
    evs = heldout_events()
    rng = np.random.default_rng([SEED, 11])
    checked = 0
    for i in rng.choice(len(evs), 60, replace=False):
        g = evs[i]
        if g[-1] == UNK:
            continue
        a, b = lm.score(g[-1], list(g[:-1])), interpolated_kn_prob(g[-1], g[:-1], ours)
        assert abs(a - b) < 1e-12, (g, a, b)
        checked += 1
    assert checked >= 50  # 60 of 60 on this draw


def test_hpylm_one_table_is_ikn():
    """Prop 6: with θ = 0 and one table per word type, the HPYLM predictive IS interpolated KN with
    D_k = d_k — parent customers are continuation counts — on every query."""
    sents, V = corpus()["train"], len(vocabulary())
    d = {1: 0.6, 2: 0.7, 3: 0.8}
    seat = one_table_seating(sents)
    kn = kn_fit(sents, discounts={k: (dk,) for k, dk in d.items()})
    theta = {k: 0.0 for k in d}
    evs = heldout_events()
    rng = np.random.default_rng([SEED, 12])
    for i in rng.choice(len(evs), 300, replace=False):
        g = evs[i]
        assert abs(hpylm_prob(g[-1], g[:-1], seat, d, theta, V) - interpolated_kn_prob(g[-1], g[:-1], kn)) < 1e-13
    for k in (1, 2):  # the seating's parent customers are the continuation counts, level by level
        got = Counter({u + (w,): n for u, cu in seat.customers.items() if len(u) == k - 1 for w, n in cu.items()})
        assert got == kn.counts[k]


def test_hpylm_sampler_invariants():
    """After every sweep the seating is legal: each restaurant's counts match its table lists, every open
    table sends exactly one customer up, and the bottom restaurants hold exactly the training tokens."""
    s = hpylm_run()["sampler"]
    up: Counter = Counter()
    for (u, w), tabs in s.tables.items():
        assert s.cw[(u, w)] == sum(tabs) and s.tw[(u, w)] == len(tabs) and all(t > 0 for t in tabs)
        if u:
            up[(u[1:], w)] += len(tabs)
    for (u, w), c in s.cw.items():
        if len(u) < ORDER - 1:
            assert c == up[(u, w)], (u, w)
    raw = train_counts()[ORDER]
    assert all(s.cw[(g[:-1], g[-1])] == n for g, n in raw.items())
    assert sum(c for (u, _), c in s.cw.items() if len(u) == ORDER - 1) == sum(raw.values())


def test_hpylm_discount_grows_with_count():
    """A singleton sits at one table, so its discount is d; frequent n-grams split across tables, and the
    effective discount d·E[t | r] rises with r — past modified KN's top step by r = 3."""
    h = hpylm_run()
    eff, d3 = h["effective_discount"], h["d"][ORDER]
    assert abs(eff[0] - d3) < 1e-12  # E[t | r = 1] = 1 exactly
    assert all(b > a for a, b in zip(eff[:6], eff[1:6]))
    assert all(e > kn_models()["mkn"].discounts[ORDER][2] for e in eff[2:6])
    assert all(h["theta"][k] < 0.05 for k in (2, 3))  # higher orders: pure discounting, KN's regime


def test_heldout_ordering():
    """In domain: HPYLM < modified KN < interpolated KN < absolute discounting < Dirichlet (Type-II)
    < Laplace, and Katz and the MLE are ∞. Out of domain (EDGAR) Laplace's heavy smoothing wins."""
    t = smoother_table()
    held = [t[n]["held"] for n in ("hpylm", "mkn", "ikn", "absdisc", "dirichlet", "laplace")]
    assert all(a < b for a, b in zip(held, held[1:])), held
    assert t["katz"]["held"] == math.inf and t["mle"]["held"] == math.inf
    assert t["dirichlet_oracle"]["held"] < t["dirichlet"]["held"]
    # The evidence under-smooths only the lower orders: the oracle α is ~62× Type-II's at k = 1, 2 and
    # smaller at k = 3.
    t2, orc = type2_alphas(), heldout_optimal_alphas()
    assert all(50 < o / a < 75 for o, a in zip(orc[:2], t2[:2])) and orc[2] < t2[2], (t2, orc)
    finite = {n: r["edgar"] for n, r in t.items() if "edgar" in r and math.isfinite(r["edgar"])}
    assert min(finite, key=finite.get) == "laplace", finite


def test_continuation_rebuild_matches_corpus():
    assert continuation_run()["first"] == tuple(d.text for d in fc.generate_corpus(300, 0))


def test_good_turing_forecasts_the_continuation():
    """N₁/N forecasts the OOV rate of the same process continued (within 10%), not of a fresh run that
    restarts the restaurants (far lower) nor of another source (EDGAR, far higher)."""
    f = finance_summary()
    gt = f["gt_unseen_mass"]
    assert abs(f["oov_continuation"] - gt) < 0.1 * gt
    assert f["oov_fresh_run"] < gt / 2 and f["oov_edgar"] > 10 * gt


def test_halvorsen_effect():
    """`industrial` (284 tokens, always after `halvorsen`) is a top-30 word to absolute discounting's
    unigram level and below rank 1,000 to Kneser–Ney's."""
    f = finance_summary()
    assert (f["halvorsen_raw"], f["halvorsen_continuation"]) == (284, 1)
    assert f["rank_absdisc"] <= 30 < 1000 < f["rank_ikn"]
    assert 60 < f["p_unigram_absdisc"] / f["p_unigram_ikn"] < 62  # the page's "about sixty-one times"


def test_viz_payload():
    p = viz_payload()
    for ctx in p["contexts"]:
        dists = [d for d in ctx["fixed"].values() if d] + [d for ds in ctx["by_d"].values() for d in ds]
        for d in dists:
            total = sum(q for _, q, _ in d["top"]) + d["rest_seen"] + d["rest_unseen"]
            assert abs(total - 1) < 1e-5
        if ctx["count"] == 0:
            assert ctx["fixed"]["mle"] is None
    osaka = p["contexts"][2]
    assert osaka["fixed"]["katz"]["unseen_mass"] == 0 and osaka["by_d"]["ikn"][6]["unseen_mass"] > 0
    unk = p["contexts"][4]
    assert unk["by_d"]["absdisc"][6]["watch"][1] < 50 < 1000 < unk["by_d"]["ikn"][6]["watch"][1]
    assert p["logloss"]["katz"] is None and p["logloss"]["katz_zero_events"] > 0
    ikn = p["logloss"]["by_d"]["ikn"]
    assert 0 < ikn.index(min(ikn)) < len(ikn) - 1  # the slider shows an interior minimum


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "n-gram-smoothing-lab.json"


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    import json

    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def _run_harness() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    print("\nSummary (train generate_corpus(300, 0), held out generate_corpus(100, 1)):")
    for k, v in summary().items():
        print(f"  {k:28} {v:.4f}" if isinstance(v, float) else f"  {k:28} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
