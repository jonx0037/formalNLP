"""Lexicon sentiment and the Loughran–McDonald dictionary — the reference implementation for formalNLP's eighteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. A lexicon is a linear classifier whose weights were fixed before any labeled document was seen. The
sections follow the page:

1. The picture. A lexicon L = (P, N, …) gives a word a weight in {−1, 0, +1} (Def 1); a document becomes a point of
   the category simplex (its rates p̂₊, p̂₋ of positive and negative hits), and net tone p̂₊ − p̂₋ projects that point onto
   one direction. When each tone draws token categories i.i.d., the Bayes rule given the counts cuts the simplex by
   lines; it is a threshold rule on net tone exactly when every pair of tones has antisymmetric log-rate ratios, as in a
   symmetric-tilt world, where the cutpoints are ±τ (Prop 1). On the finance corpus the two coordinates beat their
   difference by 0.09.
2. Length. When each token is a positive hit with rate a and a negative hit with rate b, the raw score's sign is wrong
   with probability at most (1 − (√a − √b)²)^L, and that exponent is the exact large-deviation rate (Thm 1).
3. Contamination. Adding tone-independent words of rate c to the negative list leaves the mean gap between tones
   unchanged and raises both variances by exactly c(1 − c ± 2(a − b)), so the signal-to-noise ratio falls strictly
   (Prop 2): Loughran and McDonald's case against general-purpose negative lists in finance.
4. What no lexicon sees: aspect valence × direction (Topic 17's XOR), the valence trap ("costs exceeded expectations"),
   and negation (Def 2, a k-token negation scope). Uncertainty and litigious words are not neutrality.
5. The lexicon as a prior. Logistic regression with penalty (λ/2) min_β ‖W_k − Aβ_k‖² for a lexicon subspace span(A)
   (A = w_L for net tone, A = [1_P, 1_N] for the categories) is ridge on the part of each class's weights orthogonal to
   span(A); as λ → ∞ that part is at most (mean ‖x‖)/λ and the fit converges to a softmax on the lexicon counts Aᵀx,
   the dictionary with fitted weights and cutpoints (Thm 2).
6. Inducing a lexicon. The harmonic function on a word graph with seed boundary values exists, is unique, obeys the
   maximum principle, and equals the random walk's absorption probabilities (Thm 3). On document co-occurrence it
   recovers most stand-in polarities' signs from two seeds per pole, but it scores the finance polysemes almost as far
   from zero as the polar words and ranks topical words at both extremes: induction scores candidates; it does not
   select them.

Licensing. The Loughran–McDonald Master Dictionary is free for academic research and is NEVER committed to this
repository. ``load_lm_dictionary`` reads a copy the reader supplies (or, on explicit opt-in, downloads one into
~/.cache, outside the repo). Nothing in the harness or the lab uses it: every assert runs against the hand-written
``fc.STANDIN_LEXICON``, which is not derived from it.

Cross-checks: exact trinomial tails and numerical minimization for Thm 1; exact moments for Prop 2; the direct
multinomial likelihood for Prop 1; iterative label propagation and a Monte Carlo random walk for Thm 3.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python notebooks/lexicon-sentiment-loughran-mcdonald/lexicon_sentiment_loughran_mcdonald.py
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import optimize
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "linear-text-classifiers"))
import linear_text_classifiers as ltc  # noqa: E402 — a module object: its test_* stay out of this harness

fc = ltc.fc

# ONE stream per random-instance claim and per sampled grid, each from default_rng([SEED, k]); k ≥ 3100 keeps this
# module's streams apart from Topics 1–17.
SEED = ltc.SEED
_r = ltc._r


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


def _g(x) -> str:
    return f"{x:g}"


LEX = fc.STANDIN_LEXICON
GEN = fc.GENERAL_STANDIN_LEXICON
CATEGORIES = ("positive", "negative", "uncertainty", "litigious")
TONES = (-1, 0, 1)  # class index t + 1


# --------------------------------------------------------------------------------------------
# 1. A lexicon is a linear functional (Def 1), and the Bayes rule of a tilt world (Prop 1).
# --------------------------------------------------------------------------------------------


def lexicon_weights(voc: dict[str, int], lexicon=LEX) -> np.ndarray:
    """Def 1's w_L ∈ {−1, 0, 1}^V over a feature vocabulary: +1 on positive words, −1 on negative ones."""
    w = np.zeros(len(voc))
    for cat, sign in (("positive", 1.0), ("negative", -1.0)):
        for word in lexicon[cat]:
            if word in voc:
                w[voc[word]] = sign
    return w


def category_counts(tokens: list[str], lexicon=LEX) -> np.ndarray:
    """A document's hits in each of the four categories, then its length: the corners of the category simplex."""
    return np.array([sum(t in lexicon[c] for t in tokens) for c in CATEGORIES] + [len(tokens)], dtype=float)


def tilt_world(theta: np.ndarray, P: np.ndarray, N: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """Prop 1's world. Base rates θ with equal mass m on the positive set P and the negative set N; positive documents
    scale P's rates by e^β and N's by e^{−β}, negative documents the reverse, neutral documents keep θ. Rows are the
    class distributions in tone order (−, 0, +); Z = 1 + 2m(cosh β − 1) normalizes both polar rows."""
    m = theta[P].sum()
    assert abs(theta[N].sum() - m) < 1e-12, "Prop 1 needs equal base mass on P and N"
    tilt = np.zeros_like(theta)
    tilt[P], tilt[N] = 1.0, -1.0
    Z = 1 + 2 * m * (math.cosh(beta) - 1)
    rows = np.vstack([theta * np.exp(-beta * tilt), theta, theta * np.exp(beta * tilt)])
    assert np.allclose(rows.sum(1), [Z, 1, Z])
    return rows / rows.sum(1, keepdims=True), Z


def tilt_log_posterior(x: np.ndarray, P: np.ndarray, N: np.ndarray, beta: float, Z: float) -> np.ndarray:
    """Prop 1's closed form with equal priors: logits (−β s − L log Z, 0, β s − L log Z), s = pos − neg, normalized."""
    s, L = x[P].sum() - x[N].sum(), x.sum()
    z = np.array([-beta * s - L * math.log(Z), 0.0, beta * s - L * math.log(Z)])
    return z - logsumexp(z)


def tilt_threshold(beta: float, Z: float) -> float:
    """τ = log Z / β: the Bayes rule says + iff (pos − neg)/L > τ, − iff < −τ, neutral otherwise."""
    return math.log(Z) / beta


def category_rates(C: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Prop 1's plug-in category world: per tone, the pooled rates (q₊, q₋, q₀) of positive hits, negative hits, and
    every other token."""
    Q = np.array([[C[y == k, 0].sum(), C[y == k, 1].sum(), (C[y == k, 4] - C[y == k, 0] - C[y == k, 1]).sum()]
                  for k in range(3)])
    return Q / Q.sum(1, keepdims=True)


def category_bayes(C: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Prop 1(a) with equal priors: argmax_t of pos·log q_t₊ + neg·log q_t₋ + other·log q_t₀. Divided by L, each class's
    score is affine in (p̂₊, p̂₋), so the regions are cut by lines."""
    X = np.c_[C[:, 0], C[:, 1], C[:, 4] - C[:, 0] - C[:, 1]]
    return (X @ np.log(Q).T).argmax(1)


def pairwise_slopes(Q: np.ndarray) -> dict:
    """Prop 1(b): for each pair of tones (t, t'), the coefficients (u₊, u₋) of p̂₊ and p̂₋ in the log-odds, relative to
    the 'other' category: u_c = log(q_tc/q_t'c) − log(q_t0/q_t'0). Net tone decides the pair iff u₊ = −u₋."""
    lq = np.log(Q)
    out = {}
    for t, s in ((2, 1), (1, 0), (2, 0)):
        d = lq[t] - lq[s]
        out[(t - 1, s - 1)] = (float(d[0] - d[2]), float(d[1] - d[2]))
    return out


# --------------------------------------------------------------------------------------------
# 2. How long a document must be (Thm 1).
# --------------------------------------------------------------------------------------------


def chernoff_rho(a: float, b: float) -> float:
    """Thm 1's per-token factor: min_t E e^{−tX} = 1 − (√a − √b)², at e^t = √(a/b)."""
    return 1.0 - (math.sqrt(a) - math.sqrt(b)) ** 2


def score_pmf(a: float, b: float, L: int) -> np.ndarray:
    """Exact law of S_L = Σ X_i, X_i = +1, −1, 0 w.p. a, b, 1 − a − b: index s + L holds P(S_L = s). Every term of the
    L-fold convolution is nonnegative, so the tail keeps its relative accuracy."""
    step = np.array([b, 1.0 - a - b, a])
    out = np.array([1.0])
    for bit in bin(L)[2:]:  # square-and-multiply
        out = np.convolve(out, out)
        if bit == "1":
            out = np.convolve(out, step)
    return out


def sign_error(a: float, b: float, L: int) -> float:
    """P(S_L ≤ 0): the raw score fails to call a positive document positive (a tie counts as a failure)."""
    return float(score_pmf(a, b, L)[: L + 1].sum())


L_GRID = tuple(int(v) for v in np.unique(np.geomspace(1, 3000, 36).round()))


# --------------------------------------------------------------------------------------------
# 3. Contamination (Prop 2).
# --------------------------------------------------------------------------------------------


def token_moments(p_pos: float, p_neg: float) -> tuple[float, float]:
    """Mean and variance of one token's score X ∈ {+1, −1, 0}."""
    mu = p_pos - p_neg
    return mu, p_pos + p_neg - mu * mu


def contamination_moments(a: float, b: float, c: float) -> dict:
    """Prop 2 by direct computation: per-token (mean, variance) for positive documents (rates a, b) and negative ones
    (b, a), before and after adding a tone-independent rate c to the negative list, and Fisher's ratio."""
    out = {}
    for tag, cc in (("clean", 0.0), ("contaminated", c)):
        (m1, v1), (m0, v0) = token_moments(a, b + cc), token_moments(b, a + cc)
        out[tag] = {"mean+": m1, "mean-": m0, "var+": v1, "var-": v0, "snr": (m1 - m0) ** 2 / ((v1 + v0) / 2)}
    return out


# --------------------------------------------------------------------------------------------
# The finance corpus: Topic 17's split, scored by lexicons.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def split_counts(lexicon_name: str = "finance") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Category counts (n × 5: four categories and length) of Topic 17's train, dev, and test documents."""
    lexicon = {"finance": LEX, "general": GEN}[lexicon_name]
    Ttr, _, Tdv, _, Tte, _ = ltc.fin_split()
    return tuple(np.array([category_counts(t, lexicon) for t in T]) for T in (Ttr, Tdv, Tte))  # type: ignore[return-value]


def net_tone(C: np.ndarray) -> np.ndarray:
    """Loughran and McDonald's proportional net tone (pos − neg)/L: fc.lexicon_score on the count rows."""
    return (C[:, 0] - C[:, 1]) / np.maximum(C[:, 4], 1)


def tune_thresholds(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The cutpoints (lo, hi) maximizing three-class accuracy of − / 0 / + = score < lo / in between / > hi, over
    midpoints of the sorted distinct scores (exhaustive: every distinct rule is reached)."""
    u = np.unique(score)
    cuts = np.r_[u[0] - 1.0, (u[1:] + u[:-1]) / 2, u[-1] + 1.0]
    order = np.argsort(score)
    s, ys = score[order], y[order]
    pos = np.searchsorted(s, cuts)  # documents strictly below each cut
    cum = np.vstack([np.zeros(3, int), np.cumsum(np.eye(3, dtype=int)[ys], 0)])
    below = cum[pos]  # class counts below each cut
    tot = cum[-1]
    best, arg = -1, (0, 0)
    for i in range(len(cuts)):  # lo = cuts[i], hi = cuts[j ≥ i]
        neg_ok = below[i, 0]
        mid = below[i:, 1] - below[i, 1]
        top = tot[2] - below[i:, 2]
        acc = neg_ok + mid + top
        j = int(np.argmax(acc))
        if acc[j] > best:
            best, arg = int(acc[j]), (i, i + j)
    return float(cuts[arg[0]]), float(cuts[arg[1]])


def apply_thresholds(score: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(score < lo, 0, np.where(score > hi, 2, 1))


@lru_cache(maxsize=None)
def lexicon_classifier(lexicon_name: str = "finance") -> dict:
    """Net tone with cutpoints tuned on the training split, scored on the test split."""
    ytr, yte = ltc.fin_split()[1], ltc.fin_split()[5]
    Ctr, _, Cte = split_counts(lexicon_name)
    lo, hi = tune_thresholds(net_tone(Ctr), ytr)
    st = net_tone(Cte)
    pred = apply_thresholds(st, lo, hi)
    polar = yte != 1
    return {"lo": lo, "hi": hi, "acc": float((pred == yte).mean()),
            "sign_acc": float((np.sign(st[polar]) == yte[polar] - 1).mean()),
            "tie_share": float((st[polar] == 0).mean())}


@lru_cache(maxsize=None)
def category_classifier() -> dict:
    """Prop 1(a) on the corpus: the plug-in category-world Bayes rule (rates from the training split) on the test split."""
    ytr, yte = ltc.fin_split()[1], ltc.fin_split()[5]
    Ctr, _, Cte = split_counts("finance")
    Q = category_rates(Ctr, ytr)
    return {"Q": Q, "acc": float((category_bayes(Cte, Q) == yte).mean()), "slopes": pairwise_slopes(Q)}


def tone_rates(lexicon_name: str = "finance") -> dict:
    """Per-tone hit rates on the training split: positive, negative, uncertainty, litigious hits per token."""
    ytr = ltc.fin_split()[1]
    C = split_counts(lexicon_name)[0]
    return {t: C[ytr == t + 1, :4].sum(0) / C[ytr == t + 1, 4].sum() for t in TONES}


def score_moments(lexicon_name: str = "finance") -> dict:
    """Prop 2 on the corpus: mean and variance of net tone by tone (test split), and Fisher's ratio between the polar
    tones."""
    yte = ltc.fin_split()[5]
    s = net_tone(split_counts(lexicon_name)[2])
    m = {t: float(s[yte == t + 1].mean()) for t in TONES}
    v = {t: float(s[yte == t + 1].var()) for t in TONES}
    return {"mean": m, "var": v, "snr": (m[1] - m[-1]) ** 2 / ((v[1] + v[-1]) / 2)}


def length_terciles() -> dict:
    """Sign accuracy of the finance net tone on polar test documents, by length tercile."""
    yte = ltc.fin_split()[5]
    C = split_counts("finance")[2]
    polar = yte != 1
    L, s, y = C[polar, 4], net_tone(C[polar]), yte[polar] - 1
    edges = np.quantile(L, [1 / 3, 2 / 3])
    bins = np.digitize(L, edges)
    return {"mean_len": [float(L[bins == k].mean()) for k in range(3)],
            "sign_acc": [float((np.sign(s[bins == k]) == y[bins == k]).mean()) for k in range(3)]}


@lru_cache(maxsize=None)
def fin_test_docs() -> tuple:
    """The test split as FinanceDoc objects (gold sentences, aspects, document types); same seed as ltc.FIN_TEST."""
    return tuple(fc.generate_corpus(*ltc.FIN_TEST))


# --------------------------------------------------------------------------------------------
# 4. What no lexicon sees: constructions, negation (Def 2), and the non-polar categories.
# --------------------------------------------------------------------------------------------

_DIRECTION = frozenset("rose increased climbed fell declined decreased".split())
_COMMENT = re.compile(r"described as|company called|viewed as|characterized as")
NEGATORS = frozenset({"not", "no", "never"})
CONSTRUCTIONS = ("adjective", "direction + comment", "direction only", "negated adjective", "verb, up-is-good",
                 "verb, cost-like", "noun phrase")


def construction(text: str, aspects) -> str:
    """Which generator form a single-polarity aspect sentence uses (the corpus's own templates)."""
    low = text.lower()
    toks = set(fc._tokens(text))
    if low.startswith("management pointed to"):
        return "noun phrase"
    if " not " in low:
        return "negated adjective"
    if "expectations" in toks:
        cost_like = any(fc.ASPECT_CATEGORIES[a.category][0] < 0 for a in aspects)
        return "verb, cost-like" if cost_like else "verb, up-is-good"
    if toks & _DIRECTION:
        return "direction + comment" if _COMMENT.search(low) else "direction only"
    return "adjective"


def negated_counts(tokens: list[str], k: int, lexicon=LEX) -> tuple[int, int, int, int]:
    """Def 2: a positive or negative hit preceded by a negator within k tokens counts for the other side.
    Returns (pos, neg, flips, flips at distance > 1). This corpus negates only "not ADJ", so a flip at distance > 1
    is a false flip."""
    pos = neg = flips = far = 0
    last = -10**9
    for i, t in enumerate(tokens):
        if t in NEGATORS:
            last = i
            continue
        sign = 1 if t in lexicon["positive"] else -1 if t in lexicon["negative"] else 0
        if sign == 0:
            continue
        if k > 0 and i - last <= k:
            sign, flips, far = -sign, flips + 1, far + (i - last > 1)
        pos, neg = pos + (sign > 0), neg + (sign < 0)
    return pos, neg, flips, far


NEG_K = (0, 1, 2, 3, 5, 8)


@lru_cache(maxsize=None)
def aspect_sentences() -> tuple:
    """Every test sentence whose aspects share one nonzero polarity and no contrast: (construction, polarity, tokens)."""
    out = []
    for d in fin_test_docs():
        for (s0, s1) in d.sentences:
            asp = [a for a in d.aspects if s0 <= a.start < s1]
            pols = {a.polarity for a in asp}
            text = d.text[s0:s1]
            if not asp or len(pols) != 1 or 0 in pols or ", but " in text:
                continue
            out.append((construction(text, asp), int(pols.pop()), tuple(fc._tokens(text))))
    return tuple(out)


def construction_table(k: int = 0) -> dict:
    """Sentence-level lexicon accuracy (sign of pos − neg against the gold polarity; a zero is an abstention) per
    construction, with negation scope k."""
    rows = {c: {"n": 0, "right": 0, "abstain": 0} for c in CONSTRUCTIONS}
    for c, pol, toks in aspect_sentences():
        p, n, _, _ = negated_counts(list(toks), k)
        s = int(np.sign(p - n))
        rows[c]["n"] += 1
        rows[c]["right"] += s == pol
        rows[c]["abstain"] += s == 0
    for r in rows.values():
        r["acc"] = r["right"] / r["n"] if r["n"] else float("nan")
        r["abstain_share"] = r["abstain"] / r["n"] if r["n"] else float("nan")
    total = sum(r["n"] for r in rows.values())
    rows["all"] = {"n": total, "acc": sum(r["right"] for r in rows.values()) / total,
                   "abstain_share": sum(r["abstain"] for r in rows.values() if "abstain" in r) / total}
    return rows


@lru_cache(maxsize=None)
def negation_sweep() -> dict:
    """Document-level three-class accuracy (cutpoints re-tuned on train for each k) and flip counts, by scope k."""
    Ttr, ytr, _, _, Tte, yte = ltc.fin_split()
    out = {"k": list(NEG_K), "acc": [], "flips": [], "far": [], "sentence_acc": []}
    for k in NEG_K:
        def scores(T):
            rows = np.array([negated_counts(t, k) for t in T], dtype=float)
            return (rows[:, 0] - rows[:, 1]) / np.array([max(len(t), 1) for t in T]), rows
        s_tr, _ = scores(Ttr)
        s_te, rows = scores(Tte)
        lo, hi = tune_thresholds(s_tr, ytr)
        out["acc"].append(float((apply_thresholds(s_te, lo, hi) == yte).mean()))
        out["flips"].append(int(rows[:, 2].sum()))
        out["far"].append(int(rows[:, 3].sum()))
        out["sentence_acc"].append(construction_table(k)["all"]["acc"])
    return out


def category_profile() -> dict:
    """Mean hits per 100 tokens of each category, by tone and by document type (test split)."""
    docs = fin_test_docs()
    C = split_counts("finance")[2]
    rate = 100 * C[:, :4] / C[:, 4:5]
    tone = np.array([d.tone for d in docs])
    dtype = np.array([d.doc_type for d in docs])
    return {"by_tone": {t: rate[tone == t].mean(0) for t in TONES},
            "by_type": {dt: rate[dtype == dt].mean(0) for dt in ("earnings-call", "mdna", "8-k")}}


@lru_cache(maxsize=None)
def category_features_accuracy() -> dict:
    """Three-class LR on category rates: polarity categories only vs all four (λ = 10⁻³ on rates per 100 tokens), and
    the corpus's 'any uncertainty word → neutral' rule."""
    ytr, yte = ltc.fin_split()[1], ltc.fin_split()[5]
    Ctr, _, Cte = split_counts("finance")
    out = {}
    for tag, cols in (("polarity", [0, 1]), ("all four", [0, 1, 2, 3])):
        Xtr = sp.csr_matrix(100 * Ctr[:, cols] / Ctr[:, 4:5])
        Xte = sp.csr_matrix(100 * Cte[:, cols] / Cte[:, 4:5])
        W, b, _ = ltc.lr_fit(Xtr, ytr, 3, 1e-3)
        out[tag] = float((ltc.log_posterior(Xte, W, b).argmax(1) == yte).mean())
    out["uncertainty rule"] = float(np.mean([fc.lexicon_rule(d.text) == d.tone for d in fin_test_docs()]))
    return out


def edgar_polysemes() -> list[dict]:
    """On the real EDGAR excerpts (stand-in lists only): negative hits of the finance stand-in, and the extra hits a
    general-purpose list adds through the finance polysemes."""
    out = []
    for d in fc.load_edgar_excerpts():
        toks = fc._tokens(d.text)
        poly = [t for t in toks if t in fc.FINANCE_POLYSEMES]
        out.append({"company": d.company, "tokens": len(toks),
                    "negative": sum(t in LEX["negative"] for t in toks), "polysemes": len(poly),
                    "words": sorted(set(poly))})
    return out


# --------------------------------------------------------------------------------------------
# 5. The lexicon as a prior (Thm 2).
# --------------------------------------------------------------------------------------------


KINDS = ("net", "categories")  # Thm 2's lexicon subspaces: span(w_L) and span(1_P, 1_N)


@lru_cache(maxsize=None)
def prior_design(kind: str = "net"):
    """Topic 17's unigram features (train/dev/test) with the lexicon counts Aᵀx appended as the last r columns: A = w_L
    (r = 1, the raw score pos − neg) or A = [1_P, 1_N] (r = 2, the raw positive and negative counts)."""
    X, Xd, Xt, voc = ltc.fin_features(False)
    wL = lexicon_weights(voc)
    A = wL[:, None] if kind == "net" else np.c_[wL > 0, wL < 0].astype(float)

    def aug(M):
        return sp.hstack([M, sp.csr_matrix(M @ A)]).tocsr()

    return aug(X), aug(Xd), aug(Xt), A


def prior_fit(Xa, y: np.ndarray, r: int, lam: float, eta: float = 0.0, x0=None):
    """Thm 2's estimator: logits x·Δ_k + (Aᵀx)·β_k + b_k, penalty (λ/2)‖Δ‖² + (η/2)‖β‖². Returns (Δ (K×V), β (K×r), b,
    θ)."""
    V = Xa.shape[1] - r
    W, b, th = ltc.lr_fit(Xa, y, 3, np.r_[np.full(V, lam), np.full(r, eta)], x0)
    return W[:, :V], W[:, V:], b, th


def lexicon_softmax(Xa, y: np.ndarray, r: int, eta: float = 0.0):
    """Thm 2's λ → ∞ limit: softmax on the r lexicon counts with free intercepts (the dictionary with fitted weights and
    cutpoints). Returns (β (K×r), b)."""
    W, b, _ = ltc.lr_fit(Xa[:, -r:], y, 3, eta)
    return W, b


def thm2_constant(Xa, r: int) -> float:
    """G = (1/n) Σ ‖x_i‖ over the unigram block: Thm 2(b) gives ‖Δ_k‖ ≤ G/λ for every class k."""
    X = Xa[:, :-r]
    return float(np.sqrt(np.asarray(X.multiply(X).sum(1))).mean())


THM2_LAMBDAS = (1.0, 10.0, 100.0, 1000.0)


@lru_cache(maxsize=None)
def thm2_limit(kind: str = "net") -> dict:
    """Thm 2 on the full training split: AᵀΔ_k = 0, λ‖Δ_k‖ against G, and the gap to the limit's test probabilities as
    λ grows."""
    Xa, _, Xta, A = prior_design(kind)
    r = A.shape[1]
    ytr = ltc.fin_split()[1]
    B_inf, b_inf = lexicon_softmax(Xa, ytr, r)
    Ct = np.asarray(Xta[:, -r:].todense())
    z_inf = Ct @ B_inf.T + b_inf
    p_inf = np.exp(z_inf - logsumexp(z_inf, axis=1, keepdims=True))
    rows, x0 = [], None
    for lam in THM2_LAMBDAS:
        D, B, b, x0 = prior_fit(Xa, ytr, r, lam, 0.0, x0)
        z = np.asarray(Xta[:, :-r] @ D.T) + Ct @ B.T + b
        p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
        nD = np.linalg.norm(D, axis=1).max()
        rows.append({"lam": lam, "lam_norm": float(lam * nD),
                     "orth": float(np.abs(D @ A).max() / (nD * np.linalg.norm(A, axis=0).max())),
                     "gap": float(np.abs(p - p_inf).max()), "beta": B.tolist()})
    return {"G": thm2_constant(Xa, r), "rows": rows, "beta_inf": B_inf.tolist()}


CURVE_N = (10, 30, 100, 300, 1000, 1500)
CURVE_REPS = {10: 10, 30: 10, 100: 10, 300: 3, 1000: 1, 1500: 1}
PRIOR_LAMBDAS = (1.0, 0.1, 0.01)
SUPPORT_LAMBDAS = (0.1, 0.01, 0.001)
ETA = 1e-4  # a tiny ridge on the lexicon-score slopes: a 10-document sample can separate on s alone
METHODS = ("net", "categories", "support", "lr", "prior_net", "prior_categories")
TUNED = ("support", "lr", "prior_net", "prior_categories")


@lru_cache(maxsize=None)
def learning_curves() -> dict:
    """Test accuracy against the number of labeled training documents: the dictionary with fitted cutpoints (net tone,
    and the two category counts: Thm 2's λ = ∞ ends), LR on the lexicon's words only (word power), unigram LR (Topic
    17's λ grid), and the two lexicon-centered priors. Every hyperparameter is chosen on the 600-document dev set; small
    n average CURVE_REPS random subsets (one stream)."""
    _, ytr, _, ydv, _, yte = ltc.fin_split()
    design = {k: prior_design(k) for k in KINDS}
    Xa, Xda, Xta, A = design["net"]
    X, Xd, Xt = Xa[:, :-1], Xda[:, :-1], Xta[:, :-1]
    V = X.shape[1]
    sup = np.flatnonzero(A[:, 0] != 0)
    rng = _rng(3102)

    def acc(M, W, b, y):
        return float((ltc.log_posterior(M, W, b).argmax(1) == y).mean())

    def tuned(fit, lams):
        best, x0 = None, None
        for lam in lams:
            W, b, x0, Md, Mt = fit(lam, x0)
            d = acc(Md, W, b, ydv)
            if best is None or d > best[0]:
                best = (d, acc(Mt, W, b, yte), lam)
        return best[1], best[2]

    out = {m: [] for m in METHODS} | {f"{m}_hp": [] for m in TUNED}
    for n in CURVE_N:
        res = {m: [] for m in METHODS} | {f"{m}_hp": [] for m in TUNED}
        for _ in range(CURVE_REPS[n]):
            idx = np.arange(n) if n == len(ytr) else rng.choice(len(ytr), n, replace=False)
            y = ytr[idx]
            for k in KINDS:
                Ma, _, Mta, Ak = design[k]
                r = Ak.shape[1]
                B, b = lexicon_softmax(Ma[idx], y, r, ETA)
                res[k].append(acc(Mta[:, -r:], B, b, yte))

            def fit_support(lam, x0):
                W, b_, th = ltc.lr_fit(X[idx][:, sup], y, 3, lam, x0)
                return W, b_, th, Xd[:, sup], Xt[:, sup]

            def fit_lr(lam, x0):
                W, b_, th = ltc.lr_fit(X[idx], y, 3, lam, x0)
                return W, b_, th, Xd, Xt

            def fit_prior(kind):
                Ma, Mda, Mta, Ak = design[kind]
                r = Ak.shape[1]

                def fit(lam, x0):
                    W, b_, th = ltc.lr_fit(Ma[idx], y, 3, np.r_[np.full(V, lam), np.full(r, ETA)], x0)
                    return W, b_, th, Mda, Mta
                return fit

            for m, fit, lams in (("support", fit_support, SUPPORT_LAMBDAS), ("lr", fit_lr, ltc.FIN_LAMBDAS),
                                 ("prior_net", fit_prior("net"), PRIOR_LAMBDAS),
                                 ("prior_categories", fit_prior("categories"), PRIOR_LAMBDAS)):
                a, hp = tuned(fit, lams)
                res[m].append(a)
                res[f"{m}_hp"].append(hp)
        for m in METHODS:
            out[m].append(float(np.mean(res[m])))
        for m in TUNED:
            out[f"{m}_hp"].append(float(np.median(res[f"{m}_hp"])))
    out["n"] = list(CURVE_N)
    out["crossover"] = {k: next((n for n, l, x in zip(CURVE_N, out[k], out["lr"]) if x >= l), None) for k in KINDS}
    return out


LAMBDA_PATH = (1e4, 1e3, 100.0, 10.0, 1.0, 0.1, 0.01, 1e-3)


@lru_cache(maxsize=None)
def lambda_path(kind: str = "net") -> dict:
    """Test accuracy of the prior fit along λ on the full training split, from the dictionary end (λ = 10⁴) down."""
    Xa, _, Xta, A = prior_design(kind)
    r = A.shape[1]
    ytr, yte = ltc.fin_split()[1], ltc.fin_split()[5]
    accs, x0 = [], None
    for lam in LAMBDA_PATH:
        V = Xa.shape[1] - r
        W, b, x0 = ltc.lr_fit(Xa, ytr, 3, np.r_[np.full(V, lam), np.full(r, ETA)], x0)
        accs.append(float((ltc.log_posterior(Xta, W, b).argmax(1) == yte).mean()))
    return {"lam": list(LAMBDA_PATH), "acc": accs}


# --------------------------------------------------------------------------------------------
# 6. Inducing a lexicon (Thm 3).
# --------------------------------------------------------------------------------------------


def harmonic(W: np.ndarray, seeds: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Thm 3: f = seed values on the seeds, and on the other vertices of the seeds' connected components the unique
    solution of (D − W)_UU f_U = W_US f_S. Vertices in components with no seed are left NaN (f is not determined
    there). Returns (f, U)."""
    S = np.array(list(seeds))
    fS = np.array([seeds[i] for i in S])
    _, comp = connected_components(sp.csr_matrix(W), directed=False)
    U = np.setdiff1d(np.flatnonzero(np.isin(comp, comp[S])), S)
    Lap = np.diag(W.sum(1)) - W
    f = np.full(W.shape[0], np.nan)
    f[S] = fS
    f[U] = np.linalg.solve(Lap[np.ix_(U, U)], W[np.ix_(U, S)] @ fS)
    return f, U


def propagate(W: np.ndarray, seeds: dict[int, float], iters: int) -> np.ndarray:
    """Zhu and Ghahramani's label propagation: repeat f ← D⁻¹Wf with the seeds clamped (starting from 0)."""
    f = np.zeros(W.shape[0])
    S = np.array(list(seeds))
    fS = np.array([seeds[i] for i in S])
    d = W.sum(1)
    live = d > 0
    for _ in range(iters):
        f[live] = (W[live] @ f) / d[live]
        f[S] = fS
    return f


def walk_absorption(W: np.ndarray, seeds: dict[int, float], start: int, walks: int, rng) -> float:
    """Monte Carlo: the probability that a random walk from ``start`` (step i → j w.p. W_ij/d_i) hits a +1 seed before
    a −1 seed."""
    P = W / W.sum(1, keepdims=True)
    cum = np.cumsum(P, 1)
    hit = 0
    for _ in range(walks):
        v = start
        while v not in seeds:
            v = int(np.searchsorted(cum[v], rng.random() * cum[v, -1]))
        hit += seeds[v] > 0
    return hit / walks


INDUCE_MIN_DF = 30
SEEDS = (("strong", "robust"), ("weak", "adverse"))


@lru_cache(maxsize=None)
def cooccurrence_graph(min_df: int = INDUCE_MIN_DF) -> tuple[np.ndarray, dict[str, int]]:
    """Document-level PPMI between word types of the training split with df ≥ min_df (zero diagonal)."""
    X, voc = ltc.featurize(ltc.fin_split()[0], min_df=min_df)
    B = ltc.binarize(X).astype(float)
    n = B.shape[0]
    C = np.asarray((B.T @ B).todense())
    df = np.diag(C).copy()
    np.fill_diagonal(C, 0.0)
    with np.errstate(divide="ignore"):
        pmi = np.log(C * n / np.outer(df, df))
    return np.where(C > 0, np.maximum(pmi, 0.0), 0.0), voc


def induced(seeds=SEEDS, min_df: int = INDUCE_MIN_DF) -> dict:
    """Propagate ±1 from the seed words; center by the median and scale by the sd over determined vertices (z)."""
    W, voc = cooccurrence_graph(min_df)
    lab = {voc[w]: 1.0 for w in seeds[0]} | {voc[w]: -1.0 for w in seeds[1]}
    f, U = harmonic(W, lab)
    z = (f - np.median(f[U])) / f[U].std()
    return {"z": z, "f": f, "voc": voc, "U": U, "seeds": lab}


def induction_scores(res: dict) -> dict:
    """Sign accuracy on the stand-in's polar words outside the seeds, mean |z| per category, and the precision of the
    ten most extreme words at each pole."""
    voc, z, U = res["voc"], res["z"], set(res["U"].tolist())
    seeds = res["seeds"]
    inv = {j: w for w, j in voc.items()}

    def words(ws):
        return [w for w in ws if w in voc and voc[w] in U]

    polar = words(LEX["positive"] | LEX["negative"])
    acc = float(np.mean([(z[voc[w]] > 0) == (w in LEX["positive"]) for w in polar]))
    mean_abs = {c: float(np.mean(np.abs([z[voc[w]] for w in words(LEX[c])]))) for c in CATEGORIES}
    mean_abs["polar"] = float(np.mean(np.abs([z[voc[w]] for w in polar])))
    mean_abs["polysemes"] = float(np.mean(np.abs([z[voc[w]] for w in words(fc.FINANCE_POLYSEMES)])))
    Ul = sorted(U)
    order = sorted(Ul, key=lambda j: z[j])
    top, bot = [inv[j] for j in order[-10:]], [inv[j] for j in order[:10]]
    return {"sign_acc": acc, "n_polar": len(polar), "mean_abs": mean_abs,
            "prec_pos": float(np.mean([w in LEX["positive"] for w in top])),
            "prec_neg": float(np.mean([w in LEX["negative"] for w in bot])),
            "top": top[::-1], "bottom": bot, "n_vertices": len(U) + len(seeds), "V": len(voc)}


SEED_DRAWS = 200


@lru_cache(maxsize=None)
def seed_robustness() -> list[float]:
    """Sign accuracy for SEED_DRAWS random seed sets of two positive and two negative stand-in words (one stream)."""
    W, voc = cooccurrence_graph()
    pos = sorted(w for w in LEX["positive"] if w in voc)
    neg = sorted(w for w in LEX["negative"] if w in voc)
    rng = _rng(3103)
    out = []
    for _ in range(SEED_DRAWS):
        sp_ = tuple(rng.choice(pos, 2, replace=False).tolist())
        sn = tuple(rng.choice(neg, 2, replace=False).tolist())
        out.append(induction_scores(induced((sp_, sn)))["sign_acc"])
    return out


# --------------------------------------------------------------------------------------------
# The real Loughran–McDonald dictionary: opt-in, never committed, never asserted against.
# --------------------------------------------------------------------------------------------

LM_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
LM_DRIVE_ID = "1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"  # Loughran-McDonald_MasterDictionary_1993-2025.csv (March 2026)
LM_CACHE = Path.home() / ".cache" / "formalnlp" / "lm" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
_LM_COLUMNS = {"positive": "Positive", "negative": "Negative", "uncertainty": "Uncertainty", "litigious": "Litigious"}


def parse_lm_csv(text: str) -> dict[str, frozenset[str]]:
    """The Master Dictionary's CSV → {category: frozenset of lowercase words}. A category column holds the year a word
    entered the list (0 if never; a negative year if it was later removed), so a word belongs when the value is > 0."""
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.lower(): c for c in reader.fieldnames or []}
    out: dict[str, set[str]] = {cat: set() for cat in _LM_COLUMNS}
    for row in reader:
        word = row[cols["word"]].strip().lower()
        for cat, col in _LM_COLUMNS.items():
            if float(row[cols[col.lower()]] or 0) > 0:
                out[cat].add(word)
    return {cat: frozenset(ws) for cat, ws in out.items()}


def load_lm_dictionary(path: str | Path | None = None, *, fetch: bool = False) -> dict[str, frozenset[str]] | None:
    """The real dictionary in fc.lexicon_score's shape, or None. Looks at ``path``, then $LM_MASTER_DICTIONARY, then
    ~/.cache/formalnlp/lm/. Downloads into that cache only when ``fetch`` is True or FORMALNLP_FETCH_LM=1; the
    dictionary is free for academic research (commercial use needs a license: see LM_PAGE)."""
    candidates = [Path(p) for p in (path, os.environ.get("LM_MASTER_DICTIONARY")) if p] + [LM_CACHE]
    for p in candidates:
        if p.is_file():
            return parse_lm_csv(p.read_text(encoding="utf-8", errors="replace"))
    if not (fetch or os.environ.get("FORMALNLP_FETCH_LM") == "1"):
        return None
    import urllib.request

    url = f"https://drive.usercontent.google.com/download?id={LM_DRIVE_ID}&export=download&confirm=t"
    try:
        LM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 — a fixed https URL
            data = r.read().decode("utf-8", errors="replace")
        if not data.lower().startswith("word"):
            raise ValueError("the download did not return the CSV")
        LM_CACHE.write_text(data, encoding="utf-8")
        return parse_lm_csv(data)
    except Exception as exc:  # noqa: BLE001 — explain the manual route instead
        print(f"Could not fetch the dictionary ({exc}). Download the CSV from {LM_PAGE} and set "
              f"LM_MASTER_DICTIONARY to its path, or save it as {LM_CACHE}.")
        return None


# --------------------------------------------------------------------------------------------
# The lab's data.
# --------------------------------------------------------------------------------------------

SIMPLEX_POINTS = 600  # test documents drawn for the simplex scatter (one stream)


def _simplex_payload() -> dict:
    yte = ltc.fin_split()[5]
    C = split_counts("finance")[2]
    pick = np.sort(_rng(3106).choice(len(yte), SIMPLEX_POINTS, replace=False))
    rates = 100 * C[pick, :2] / C[pick, 4:5]
    cc = category_classifier()
    lc = lexicon_classifier("finance")
    return {"points": [[_r(x, 2), _r(yv, 2), int(t) - 1] for (x, yv), t in zip(rates, yte[pick])],
            "logQ": _r(np.log(cc["Q"]), 5), "net": {"lo": _r(100 * lc["lo"], 3), "hi": _r(100 * lc["hi"], 3),
                                                   "acc": _r(lc["acc"])},
            "bayes_acc": _r(cc["acc"]), "lr_acc": _r(ltc.fin_fit(ltc.FinConfig("lr", False))["acc"]),
            "slopes": {f"{t:+d}/{u:+d}": _r(v) for (t, u), v in cc["slopes"].items()}}


def _length_payload() -> dict:
    rf, rg = tone_rates("finance")[1], tone_rates("general")[1]
    presets = [
        {"id": "finance", "label": "finance stand-in", "a": float(rf[0]), "b": float(rf[1])},
        {"id": "general", "label": "general-purpose list", "a": float(rg[0]), "b": float(rg[1])},
        {"id": "dense", "label": "ten times denser", "a": 10 * float(rf[0]), "b": 10 * float(rf[1])},
    ]
    for q in presets:
        q["rho"] = chernoff_rho(q["a"], q["b"])
        q["exact"] = _r([sign_error(q["a"], q["b"], L) for L in L_GRID], 4)
        # Thm 1 needs a > b; with the general list b > a in positive documents, and the sign error tends to 1
        q["bound"] = _r([min(1.0, q["rho"] ** L) for L in L_GRID], 4) if q["a"] > q["b"] else None
        q["a"], q["b"], q["rho"] = _r(q["a"], 4), _r(q["b"], 4), _r(q["rho"], 5)
    yte = ltc.fin_split()[5]
    C = split_counts("finance")[2]
    pos_docs = yte == 2
    Lbar = C[pos_docs, 4].mean()
    return {"L": list(L_GRID), "presets": presets,
            "doc": {"L": _r(Lbar, 1), "err": _r(float((net_tone(C[pos_docs]) <= 0).mean())),
                    "iid": _r(sign_error(rf[0], rf[1], int(round(Lbar)))),
                    "bound": _r(chernoff_rho(rf[0], rf[1]) ** Lbar)}}


HIST_EDGES = np.linspace(-0.06, 0.06, 25)


def _contamination_payload() -> dict:
    yte = ltc.fin_split()[5]
    out = {"edges": _r(100 * HIST_EDGES, 2)}
    for name in ("finance", "general"):
        s = np.clip(net_tone(split_counts(name)[2]), HIST_EDGES[0], HIST_EDGES[-1] - 1e-12)
        lc = lexicon_classifier(name)
        mo = score_moments(name)
        out[name] = {"hist": {str(t): np.histogram(s[yte == t + 1], HIST_EDGES)[0].tolist() for t in TONES},
                     "lo": _r(100 * lc["lo"], 3), "hi": _r(100 * lc["hi"], 3), "acc": _r(lc["acc"]),
                     "sign_acc": _r(lc["sign_acc"]), "snr": _r(mo["snr"]),
                     "mean": {str(t): _r(100 * mo["mean"][t], 3) for t in TONES}}
    rf, rg = tone_rates("finance"), tone_rates("general")
    out["c"] = {str(t): _r(100 * (rg[t][1] - rf[t][1]), 3) for t in TONES}
    return out


def _prior_payload() -> dict:
    c = learning_curves()
    return {"n": c["n"], **{m: _r(c[m]) for m in METHODS}, "crossover": c["crossover"],
            "ceiling": _r(ltc.bayes_ceiling()),
            "path": {k: {"lam": [_g(v) for v in lambda_path(k)["lam"]], "acc": _r(lambda_path(k)["acc"])}
                     for k in KINDS},
            "limit": {k: {"G": _r(thm2_limit(k)["G"]),
                          "rows": [{"lam": _g(r["lam"]), "lam_norm": _r(r["lam_norm"]), "gap": _r(r["gap"], 4)}
                                   for r in thm2_limit(k)["rows"]]} for k in KINDS}}


def _induction_payload() -> dict:
    res = induced()
    sc = induction_scores(res)
    voc, z = res["voc"], res["z"]
    words = []
    for cat in CATEGORIES:
        for w in sorted(LEX[cat]):
            if w in voc and not np.isnan(z[voc[w]]):
                seed = voc[w] in res["seeds"]  # a seed is a boundary value (±1), not a score: z is undefined there
                words.append({"w": w, "cat": cat, "z": None if seed else _r(z[voc[w]]), "seed": seed})
    for w in sorted(fc.FINANCE_POLYSEMES):
        if w in voc and not np.isnan(z[voc[w]]):
            words.append({"w": w, "cat": "polyseme", "z": _r(z[voc[w]]), "seed": False})
    rob = np.array(seed_robustness())
    return {"words": words, "sign_acc": _r(sc["sign_acc"]), "n_polar": sc["n_polar"], "V": sc["V"],
            "mean_abs": {k: _r(v) for k, v in sc["mean_abs"].items()}, "top": sc["top"], "bottom": sc["bottom"],
            "prec_pos": _r(sc["prec_pos"]), "prec_neg": _r(sc["prec_neg"]),
            "robust": {"median": _r(np.median(rob)), "p10": _r(np.quantile(rob, 0.1)), "min": _r(rob.min())},
            "seeds": [list(SEEDS[0]), list(SEEDS[1])], "min_df": INDUCE_MIN_DF}


def _constructions_payload() -> dict:
    ct0, ct1 = construction_table(0), construction_table(1)
    ng = negation_sweep()
    return {"rows": [{"c": c, "n": ct0[c]["n"], "acc": _r(ct0[c]["acc"]), "abstain": _r(ct0[c]["abstain_share"]),
                      "acc_k1": _r(ct1[c]["acc"])} for c in CONSTRUCTIONS],
            "all": {"acc": _r(ct0["all"]["acc"]), "acc_k1": _r(ct1["all"]["acc"])},
            "negation": {"k": ng["k"], "acc": _r(ng["acc"]), "far": ng["far"], "flips": ng["flips"]}}


def viz_payload() -> dict:
    return {"simplex": _simplex_payload(), "length": _length_payload(), "contamination": _contamination_payload(),
            "constructions": _constructions_payload(), "prior": _prior_payload(), "induction": _induction_payload()}


VIZ_JSON = (Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "lexicon-sentiment-lab.json")


def summary() -> dict:
    out = {}
    for name in ("finance", "general"):
        r = tone_rates(name)
        for t in TONES:
            out[f"{name} hit rates (pos, neg, unc, lit) in tone {t:+d}"] = " / ".join(f"{v:.6f}" for v in r[t])
        lc = lexicon_classifier(name)
        mo = score_moments(name)
        out[f"{name}: 3-class acc / polar sign acc / tie share"] = (
            f"{lc['acc']:.6f} / {lc['sign_acc']:.6f} / {lc['tie_share']:.6f}")
        out[f"{name}: cutpoints lo / hi"] = f"{lc['lo']:.6f} / {lc['hi']:.6f}"
        out[f"{name}: test means by tone"] = " / ".join(f"{mo['mean'][t]:.6f}" for t in TONES)
        out[f"{name}: test variances by tone"] = " / ".join(f"{mo['var'][t]:.3e}" for t in TONES)
        out[f"{name}: Fisher ratio (+ vs −)"] = mo["snr"]
    cc = category_classifier()
    out["Prop 1: category-world Bayes accuracy"] = cc["acc"]
    for (t, u), (up, un) in cc["slopes"].items():
        out[f"Prop 1: slopes (u+, u-) for {t:+d} vs {u:+d}"] = f"{up:.6f} / {un:.6f}"
    out["Topic 17 unigram LR accuracy"] = ltc.fin_fit(ltc.FinConfig("lr", False))["acc"]
    r = tone_rates("finance")[1]
    rho = chernoff_rho(r[0], r[1])
    out["Thm 1: rho (finance, positive docs)"] = rho
    yte = ltc.fin_split()[5]
    C = split_counts("finance")[2]
    Lbar = C[yte == 2, 4].mean()
    out["Thm 1: mean length of positive docs"] = Lbar
    out["Thm 1: bound / iid exact at mean length / measured"] = (
        f"{rho ** Lbar:.6f} / {sign_error(r[0], r[1], int(round(Lbar))):.6f} / "
        f"{float((net_tone(C[yte == 2]) <= 0).mean()):.6f}")
    out["Thm 1: expected hits per positive doc"] = (r[0] + r[1]) * Lbar
    lt = length_terciles()
    out["length terciles: mean length"] = " / ".join(f"{v:.2f}" for v in lt["mean_len"])
    out["length terciles: sign accuracy"] = " / ".join(f"{v:.6f}" for v in lt["sign_acc"])
    for k in (0, 1):
        ct = construction_table(k)
        for c in CONSTRUCTIONS + ("all",):
            out[f"sentences k={k} {c}: n / acc / abstain"] = (
                f"{ct[c]['n']} / {ct[c]['acc']:.6f} / {ct[c]['abstain_share']:.6f}")
    ng = negation_sweep()
    out[f"negation k {NEG_K}: doc acc"] = " / ".join(f"{v:.6f}" for v in ng["acc"])
    out[f"negation k {NEG_K}: flips / far flips"] = " / ".join(f"{a}:{b}" for a, b in zip(ng["flips"], ng["far"]))
    cp = category_profile()
    for t in TONES:
        out[f"category rates per 100 tokens, tone {t:+d}"] = " / ".join(f"{v:.6f}" for v in cp["by_tone"][t])
    for dt, v in cp["by_type"].items():
        out[f"category rates per 100 tokens, {dt}"] = " / ".join(f"{x:.6f}" for x in v)
    for k, v in category_features_accuracy().items():
        out[f"category LR / rule: {k}"] = v
    for e in edgar_polysemes():
        out[f"EDGAR {e['company']}: tokens / negative / polysemes"] = (
            f"{e['tokens']} / {e['negative']} / {e['polysemes']} {e['words']}")
    for kind in KINDS:
        th = thm2_limit(kind)
        out[f"Thm 2 ({kind}): G"] = th["G"]
        for row in th["rows"]:
            out[f"Thm 2 ({kind}) lam={row['lam']:g}: lam*max|D_k| / orth / gap"] = (
                f"{row['lam_norm']:.6f} / {row['orth']:.2e} / {row['gap']:.6f}")
    c = learning_curves()
    for m in METHODS:
        out[f"curves {CURVE_N}: {m}"] = " / ".join(f"{v:.6f}" for v in c[m])
    out["curves: crossover n (LR >= net / categories)"] = f"{c['crossover']['net']} / {c['crossover']['categories']}"
    for m in TUNED:
        out[f"curves: median chosen lambda, {m}"] = " ".join(f"{v:g}" for v in c[f"{m}_hp"])
    for kind in KINDS:
        out[f"lambda path ({kind}) {LAMBDA_PATH}"] = " / ".join(f"{v:.6f}" for v in lambda_path(kind)["acc"])
    sc = induction_scores(induced())
    out["Thm 3: vertices / V"] = f"{sc['n_vertices']} / {sc['V']}"
    out["Thm 3: polar words scored"] = sc["n_polar"]
    out["Thm 3: sign accuracy on polar words"] = sc["sign_acc"]
    out["Thm 3: mean |z| by category"] = " ".join(f"{k}={v:.6f}" for k, v in sc["mean_abs"].items())
    out["Thm 3: precision of top/bottom 10"] = f"{sc['prec_pos']:.6f} / {sc['prec_neg']:.6f}"
    out["Thm 3: top 10"] = " ".join(sc["top"])
    out["Thm 3: bottom 10"] = " ".join(sc["bottom"])
    res = induced()
    out["Thm 3: polyseme z"] = " ".join(f"{w}={res['z'][res['voc'][w]]:+.3f}" for w in sorted(fc.FINANCE_POLYSEMES)
                                       if w in res["voc"])
    rob = np.array(seed_robustness())
    out["Thm 3: random seeds sign acc median / p10 / min"] = (
        f"{np.median(rob):.6f} / {np.quantile(rob, 0.1):.6f} / {rob.min():.6f}")
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: one test per claim on the page.
# --------------------------------------------------------------------------------------------


def test_lexicon_is_a_linear_functional():
    """Def 1: w_L · x is pos − neg, and net tone equals fc.lexicon_score on the text."""
    docs = fin_test_docs()[:50]
    T = [fc._tokens(d.text) for d in docs]
    X, voc = ltc.featurize(T)
    wL = lexicon_weights(voc)
    C = np.array([category_counts(t) for t in T])
    assert np.array_equal(X @ wL, C[:, 0] - C[:, 1])
    assert np.allclose(net_tone(C), [fc.lexicon_score(d.text) for d in docs])


def test_prop1_category_world_is_linear():
    """Prop 1(a): in a category world the Bayes posterior given the full count vector depends on it only through the
    category counts, and equals the collapsed rule; (b) net tone decides a pair iff u₊ = −u₋."""
    rng = _rng(3107)
    for _ in range(20):
        V = int(rng.integers(6, 15))
        P, N = np.arange(2), np.arange(2, 4)
        base = rng.dirichlet(np.ones(V))
        within = [base[P] / base[P].sum(), base[N] / base[N].sum(), base[4:] / base[4:].sum()]
        Q = rng.dirichlet(np.ones(3), size=3)  # per tone (q₊, q₋, q₀)
        rows = np.array([np.r_[Q[t, 0] * within[0], Q[t, 1] * within[1], Q[t, 2] * within[2]] for t in range(3)])
        for _ in range(10):
            L = int(rng.integers(1, 60))
            x = rng.multinomial(L, rows[rng.integers(3)])
            direct = ltc.nb_log_posterior_direct(x, rows, np.log(np.ones(3) / 3))
            cnt = np.array([x[P].sum(), x[N].sum(), x[4:].sum()])
            z = cnt @ np.log(Q).T
            assert np.allclose(direct, z - logsumexp(z), atol=1e-10)
    m, beta = 0.04, 0.7  # the symmetric tilt world's category rates: antisymmetric, so net tone suffices
    q = np.array([[m * math.exp(-beta * s), m * math.exp(beta * s), 1 - 2 * m] for s in (1, 0, -1)])
    u = pairwise_slopes(q / q.sum(1, keepdims=True))
    assert all(abs(up + un) < 1e-12 for up, un in u.values()), u
    assert abs(u[(1, 0)][0] - beta) < 1e-12


def test_prop1_tilt_world_bayes_rule():
    """Prop 1(c): in a symmetric-tilt world the NB posterior equals the closed form, the ± log-odds is 2β(pos − neg),
    and the Bayes decision is the normalized score against ±τ."""
    rng = _rng(3100)
    for _ in range(20):
        V = int(rng.integers(6, 15))
        P, N = np.arange(2), np.arange(2, 4)
        theta = rng.dirichlet(np.ones(V))
        m = rng.uniform(0.05, 0.2)
        theta[P] *= m / theta[P].sum()
        theta[N] *= m / theta[N].sum()
        rest = np.arange(4, V)
        theta[rest] *= (1 - 2 * m) / theta[rest].sum()
        beta = rng.uniform(0.2, 2.0)
        rows, Z = tilt_world(theta, P, N, beta)
        tau = tilt_threshold(beta, Z)
        for _ in range(10):
            L = int(rng.integers(1, 60))
            x = rng.multinomial(L, rows[rng.integers(3)])
            direct = ltc.nb_log_posterior_direct(x, rows, np.log(np.ones(3) / 3))
            closed = tilt_log_posterior(x, P, N, beta, Z)
            assert np.allclose(direct, closed, atol=1e-10)
            s = x[P].sum() - x[N].sum()
            assert abs((closed[2] - closed[0]) - 2 * beta * s) < 1e-10
            r = s / L
            if abs(abs(r) - tau) > 1e-9:  # off the boundary the rule is unambiguous
                assert int(np.argmax(closed)) == (2 if r > tau else 0 if r < -tau else 1)


def test_net_tone_discards_a_coordinate():
    """Prop 1 on the corpus: the tones' log-rate ratios are far from antisymmetric (neutral documents have few polar
    words of either kind), and the category-world rule beats net-tone thresholds by more than 0.07."""
    cc = category_classifier()
    up, un = cc["slopes"][(1, 0)]
    assert up > 0.5 and abs(un) < 0.5 * up, (up, un)
    assert cc["acc"] > lexicon_classifier("finance")["acc"] + 0.07


def test_thm1_bound_holds():
    """Thm 1: P(S_L ≤ 0) ≤ ρ^L on a grid and on random instances, and ρ is the minimum of the moment generating
    function at e^t = √(a/b)."""
    rng = _rng(3101)
    cases = [(a, b, L) for a in (0.02, 0.05, 0.2, 0.4) for b in (0.005, 0.01, 0.1) if b < a
             for L in (1, 5, 20, 100, 500)]
    cases += [(a, a * rng.uniform(0.05, 0.95), int(rng.integers(1, 400)))
              for a in rng.uniform(0.005, 0.45, 200)]
    for a, b, L in cases:
        assert sign_error(a, b, L) <= chernoff_rho(a, b) ** L * (1 + 1e-12), (a, b, L)
        res = optimize.minimize_scalar(lambda t: a * math.exp(-t) + b * math.exp(t) + 1 - a - b)
        assert abs(res.fun - chernoff_rho(a, b)) < 1e-9 and abs(res.x - 0.5 * math.log(a / b)) < 1e-4
    assert abs(score_pmf(0.3, 0.1, 37).sum() - 1) < 1e-12


def test_thm1_exponent_is_tight():
    """Thm 1 (Cramér): (log P(S_2L ≤ 0) − log P(S_L ≤ 0))/L → log ρ, with error shrinking like (log L)/L."""
    a, b = 0.3, 0.1
    lr = math.log(chernoff_rho(a, b))
    errs = [abs((math.log(sign_error(a, b, 2 * L)) - math.log(sign_error(a, b, L))) / L - lr)
            for L in (125, 250, 500, 1000)]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-3, errs


def test_length_on_the_corpus():
    """Thm 1's picture on the corpus: at the mean length the bound is loose (0.67) but valid, and the i.i.d. model's
    exact error matches the measured one to within 0.02."""
    r = tone_rates("finance")[1]
    yte = ltc.fin_split()[5]
    C = split_counts("finance")[2]
    Lbar = C[yte == 2, 4].mean()
    measured = float((net_tone(C[yte == 2]) <= 0).mean())
    assert measured <= chernoff_rho(r[0], r[1]) ** Lbar
    assert abs(sign_error(r[0], r[1], int(round(Lbar))) - measured) < 0.02


def test_prop2_contamination_moments():
    """Prop 2: the mean gap is unchanged, each variance rises by exactly c(1 − c ± 2(a − b)), the ratio falls."""
    rng = _rng(3104)
    for _ in range(500):
        b = rng.uniform(0.001, 0.1)
        a = b + rng.uniform(0.0, 0.1)
        c = rng.uniform(0.0005, 0.1)
        m = contamination_moments(a, b, c)
        cl, co = m["clean"], m["contaminated"]
        assert abs((co["mean+"] - co["mean-"]) - (cl["mean+"] - cl["mean-"])) < 1e-15
        assert abs(co["var+"] - cl["var+"] - c * (1 - c + 2 * (a - b))) < 1e-14
        assert abs(co["var-"] - cl["var-"] - c * (1 - c - 2 * (a - b))) < 1e-14
        if a > b:
            assert co["snr"] < cl["snr"]


def test_general_lexicon_misreads_finance():
    """Prop 2 on the corpus: the polysemes add a nearly tone-independent negative rate c, shift every tone's mean by
    about −c, lower Fisher's ratio, and cost accuracy even with re-tuned cutpoints."""
    rf, rg = tone_rates("finance"), tone_rates("general")
    c = np.array([rg[t][1] - rf[t][1] for t in TONES])
    assert c.min() > 0.8 * c.max(), c
    mf, mg = score_moments("finance"), score_moments("general")
    for t in TONES:
        assert abs((mg["mean"][t] - mf["mean"][t]) + c.mean()) < 0.003
        assert mg["var"][t] > mf["var"][t]
    assert mg["snr"] < mf["snr"]
    lf, lg = lexicon_classifier("finance"), lexicon_classifier("general")
    assert lg["acc"] < lf["acc"] - 0.03 and lg["sign_acc"] < lf["sign_acc"] - 0.1


def test_edgar_has_polysemes():
    """Real MD&A text (stand-in lists only): the polysemes a general list would count as negative outnumber the finance
    stand-in's negative hits."""
    rows = edgar_polysemes()
    assert len(rows) >= 3
    assert sum(r["polysemes"] for r in rows) > sum(r["negative"] for r in rows)


def test_constructions_a_lexicon_cannot_read():
    """Without negation scope the lexicon is wrong on every negated adjective and every cost-like 'exceeded
    expectations', abstains on every bare direction, and is right on plain adjectives and comments."""
    ct = construction_table(0)
    assert ct["negated adjective"]["acc"] == 0.0 and ct["verb, cost-like"]["acc"] == 0.0
    assert ct["direction only"]["abstain_share"] == 1.0
    assert ct["adjective"]["acc"] == 1.0 and ct["direction + comment"]["acc"] == 1.0
    assert min(r["n"] for c, r in ct.items() if c != "all") >= 20


def test_negation_scope():
    """Def 2: k = 1 repairs every negated adjective; growing k only adds false flips, and the document accuracy peaks
    at a short scope."""
    assert construction_table(1)["negated adjective"]["acc"] == 1.0
    ng = negation_sweep()
    assert ng["far"][1] == 0 and all(f2 >= f1 for f1, f2 in zip(ng["far"][1:], ng["far"][2:]))
    assert ng["far"][-1] > 0
    assert ng["acc"][1] > ng["acc"][0] and max(ng["acc"]) == max(ng["acc"][:3])


def test_uncertainty_is_not_neutrality():
    """Uncertainty words occur at similar rates in every tone, litigious words occur only in 8-Ks, the 'uncertain →
    neutral' rule fails, and the non-polar categories add nothing to a classifier on category rates."""
    cp = category_profile()
    unc = [cp["by_tone"][t][2] for t in TONES]
    assert min(unc) > 0.9 * max(unc), unc
    lit = cp["by_type"]
    assert lit["8-k"][3] > 0 and lit["earnings-call"][3] == lit["mdna"][3] == 0
    acc = category_features_accuracy()
    assert acc["uncertainty rule"] < lexicon_classifier("finance")["acc"]
    assert abs(acc["all four"] - acc["polarity"]) < 0.01


def test_thm2_prior_limit():
    """Thm 2, for both subspaces: AᵀΔ = 0 at every λ, λ‖Δ_k‖ ≤ G, and the test probabilities approach the λ = ∞
    softmax at rate ~1/λ."""
    for kind in KINDS:
        th = thm2_limit(kind)
        for row in th["rows"]:
            assert row["orth"] < 1e-6, (kind, row)
            assert row["lam_norm"] <= th["G"], (kind, row)
        gaps = [r["gap"] for r in th["rows"]]
        assert all(g2 < g1 / 5 for g1, g2 in zip(gaps[1:], gaps[2:])), (kind, gaps)
        assert np.allclose(th["rows"][-1]["beta"], th["beta_inf"], atol=5e-3), kind


def test_dictionary_versus_labels():
    """The learning curves: the dictionary with fitted cutpoints plateaus; unigram LR overtakes net tone only after
    hundreds of labels and the category counts later still; each lexicon-centered prior is at least as good as LR and
    its own dictionary end at every training size."""
    c = learning_curves()
    net, cat, lr = np.array(c["net"]), np.array(c["categories"]), np.array(c["lr"])
    assert net[1:].max() - net[1:].min() < 0.06
    assert c["crossover"]["net"] is not None and c["crossover"]["net"] >= 100
    assert np.all(cat[1:] > net[1:])
    for kind, end in (("net", net), ("categories", cat)):
        pr = np.array(c[f"prior_{kind}"])
        assert np.all(pr >= np.maximum(end, lr) - 0.01), (kind, pr, end, lr)
    assert c["prior_net"][-1] > lr[-1] + 0.01
    assert c["support"][-1] > net[-1] + 0.05


def test_thm3_harmonic_properties():
    """Thm 3 on random weighted graphs: the solve is unique (L_UU positive definite), obeys the maximum principle,
    equals clamped label propagation's limit, and equals 2·P(absorb at +) − 1 by Monte Carlo."""
    rng = _rng(3105)
    for _ in range(10):
        n = int(rng.integers(8, 20))
        A = np.triu(rng.random((n, n)) < 0.35, 1) * rng.uniform(0.1, 2.0, (n, n))
        A = A + A.T
        A[0, 1:] = A[1:, 0] = np.where(A[0, 1:] > 0, A[0, 1:], 0.05)  # connected through vertex 0
        seeds = {0: 1.0, n - 1: -1.0}
        f, U = harmonic(A, seeds)
        assert len(U) == n - 2
        Lap = np.diag(A.sum(1)) - A
        assert np.linalg.eigvalsh(Lap[np.ix_(U, U)]).min() > 0
        assert np.all(f <= 1 + 1e-12) and np.all(f >= -1 - 1e-12)
        assert np.allclose(propagate(A, seeds, 20000), f, atol=1e-8)
    A = np.array([[0, 2, 1, 0, 0], [2, 0, 1, 1, 0], [1, 1, 0, 2, 1], [0, 1, 2, 0, 3], [0, 0, 1, 3, 0]], float)
    seeds = {0: 1.0, 4: -1.0}
    f, _ = harmonic(A, seeds)
    for v in (1, 2, 3):
        h = walk_absorption(A, seeds, v, 20000, rng)
        assert abs((2 * h - 1) - f[v]) < 4 * 2 * math.sqrt(0.25 / 20000), (v, h, f[v])


def test_induction_on_the_corpus():
    """Two seeds per pole recover most stand-in polarities' signs, and the result survives random seeds; but the
    polysemes score almost as far from zero as the polar words, and the most extreme words are mostly not lexicon
    words: induction scores a candidate list, it does not select one."""
    sc = induction_scores(induced())
    assert sc["sign_acc"] >= 0.85, sc["sign_acc"]
    rob = np.array(seed_robustness())
    assert np.median(rob) >= 0.75, np.median(rob)
    ma = sc["mean_abs"]
    assert ma["polysemes"] > 0.7 * min(ma["positive"], ma["negative"]), ma
    assert ma["uncertainty"] < min(ma["positive"], ma["negative"]), ma
    assert sc["prec_pos"] <= 0.2 and sc["prec_neg"] <= 0.2


def test_lm_csv_parser():
    """The opt-in loader's parser on a synthetic three-row file in the Master Dictionary's layout (no real data)."""
    text = ("Word,Seq_num,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining\n"
            "ABANDON,1,2009,0,0,0,0,0,0\nACHIEVE,2,0,2009,0,0,0,0,0\nAPPROXIMATE,3,0,0,-2011,0,0,0,0\n")
    lex = parse_lm_csv(text)
    assert lex["negative"] == {"abandon"} and lex["positive"] == {"achieve"} and not lex["uncertainty"]
    assert fc.lexicon_score("we achieve", lex) == 0.5


def test_lm_dictionary_not_committed():
    """The licensing rule, enforced: no file in the repository looks like the Master Dictionary."""
    root = Path(__file__).resolve().parents[2]
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    bad = [f for f in files if re.search(r"(?i)master_?dictionary|loughran.*\.(csv|xlsx?)$", f)]
    assert not bad, bad


def test_viz_payload():
    """The lab narrates: the finance preset first, the crossovers, and every category in the induction view."""
    p = viz_payload()
    assert p["length"]["presets"][0]["id"] == "finance"
    gen = p["length"]["presets"][1]
    assert gen["bound"] is None and gen["b"] > gen["a"] and gen["exact"][-1] > 0.98  # contamination flips a − b
    assert p["prior"]["crossover"] == learning_curves()["crossover"]
    assert len(p["prior"]["n"]) == len(p["prior"]["prior_net"]) == len(CURVE_N)
    assert {w["cat"] for w in p["induction"]["words"]} == set(CATEGORIES) | {"polyseme"}
    assert len(p["simplex"]["points"]) == SIMPLEX_POINTS


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with --write-payload after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


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
    print("\nSummary:")
    for k, v in summary().items():
        print(f"  {k:62} {v:.6f}" if isinstance(v, float) else f"  {k:62} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__" and "--dev" in sys.argv:
    import time

    t0 = time.time()
    for k, v in summary().items():
        print(f"  {k:62} {v:.6f}" if isinstance(v, float) else f"  {k:62} {v}")
    print("time", round(time.time() - t0, 1))
elif __name__ == "__main__":
    raise SystemExit(_run_harness())
