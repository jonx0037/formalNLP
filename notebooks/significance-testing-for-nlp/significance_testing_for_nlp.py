"""Significance testing for NLP — the reference implementation for formalNLP's twenty-eighth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. The unit of resampling (Defs 1–3). A paired test set has n segments, each carrying both systems' outputs and their
   sufficient statistics; a corpus metric is a function g of the pooled statistics, and corpus BLEU is homogeneous of
   degree 0 in them, so it is a surface over the simplex of segment weights. The paired bootstrap draws a random
   multinomial point of that simplex; approximate randomization draws a random vertex of the swap cube {0, 1}ⁿ.
2. Randomization is exact for any metric (Thm 1, Prop 1). Under the sharp null (each segment's two outputs are
   exchangeable), the Monte Carlo p-value (1 + #{T_b ≥ T})/(B + 1) has P(p ≤ α) ≤ α for every statistic and every B.
   Topic 14's decoding prompts give an exact A/A null: two trigram samples per prompt are i.i.d., so disjoint prompt
   blocks are independent experiments with the sharp null true. On 0/1 losses the full-group randomization test is
   exactly McNemar's exact test (Prop 1).
3. The bootstrap's two p-values, and where it breaks (Prop 2, Thm 2, Prop 3). Koehn's "A wins in ≥ 1 − α of the
   resamples" inverts the percentile interval; Berg-Kirkpatrick et al.'s δ* ≥ 2δ̂ inverts the basic one; they coincide
   when the bootstrap law is symmetric about δ̂. Away from c = r, corpus BLEU's difference is asymptotically linear in
   per-segment influences ∇g·Z_i, so the bootstrap variance matches the linearization (Thm 2). At c = r the brevity
   penalty's kink makes the limit min(0, aW), and the bootstrap's coverage there falls below nominal (Prop 3).
4. Clustered segments (Def 4, Thm 3). Equicorrelated clusters multiply the variance of a mean by 1 + (m − 1)ρ, which
   the segment bootstrap misses. On Topic 19's aspect rows an aspect-blind reader is right on exactly one of the two
   aspects in every contrast sentence (ρ = −1 within the sentence), and pairing absorbs most of the document effect.
5. What a test set cannot tell you (Def 5, Thm 4). Var(δ̂) = E[Var(δ̂ | D)] + Var(Δ(D)) over training draws D; a
   consistent test-set test of one trained pair rejects the procedure-level null with probability → 1 − π₀. In Topic
   17's bursty world near the NB–LR crossover, test-set McNemar rejects in both directions across training draws.
6. Power and multiplicity (Prop 4, Prop 5). Exact McNemar power is a double binomial sum, and Connor's normal
   approximation lands near 80% at its own n; reporting the best of k metrics has size between α and kα.
7. Finance case study: tone classifiers on 1,500 filings (McNemar), and the 45 summarizer pairs on 90 filings
   (approximate randomization on corpus BLEU and fact F1, raw, Holm, and Benjamini–Hochberg).

Cross-checks: scipy's binomtest (exact McNemar), scipy's permutation_test with permutation_type="samples" (the full
swap group), Topic 27's corpus BLEU (the vectorized pooled BLEU), finite differences (the BLEU gradient).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python notebooks/significance-testing-for-nlp/significance_testing_for_nlp.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
for _dep in ("overlap-metrics-bleu-rouge", "linear-text-classifiers", "aspect-based-sentiment"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import overlap_metrics_bleu_rouge as om  # noqa: E402 — module objects: their test_* stay out of this harness
import linear_text_classifiers as ltc  # noqa: E402
import aspect_based_sentiment as ab  # noqa: E402

fc, ck = om.fc, om.ck

# ONE stream per random-instance claim, each from default_rng([SEED, k]); k ≥ 5000 keeps this module's streams apart
# from Topics 1–27.
SEED = om.SEED
VIZ_JSON = _HERE.parents[1] / "src" / "components" / "viz" / "data" / "significance-testing-lab.json"
N = om.N_MAX  # BLEU's maximum n-gram order
K = 2 + 2 * N  # one segment's BLEU statistics: c, r, m_1..N, t_1..N
ALPHA = 0.05


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


def _r(x, nd: int = 3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _tie(t0: float) -> float:
    """Floating-point tolerance for 'T_b ≥ T': resampled statistics equal to the observed one in exact arithmetic can
    differ from it in the last bits (sums taken in another order)."""
    return 1e-12 * max(1.0, abs(t0))


# --------------------------------------------------------------------------------------------
# 1. The unit of resampling (Defs 1–3).
# --------------------------------------------------------------------------------------------


def stat_matrix(stats_) -> np.ndarray:
    """Def 1: one row of BLEU statistics (c, r, m_1..N, t_1..N) per segment."""
    return np.array([[s.c, s.r, *s.m, *s.t] for s in stats_], dtype=float)


def bleu_pooled(P: np.ndarray) -> np.ndarray:
    """Corpus BLEU (Topic 27's Def 2, unsmoothed) of pooled statistics along the last axis: exp(min(0, 1 − r/c) +
    (1/N) Σ log(m_n/t_n)), and 0 when c = 0 or any m_n = 0. Homogeneous of degree 0: any positive weights work."""
    c, r, m, t = P[..., 0], P[..., 1], P[..., 2:2 + N], P[..., 2 + N:]
    ok = (c > 0) & (m > 0).all(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logb = np.minimum(0.0, 1 - r / c) + (np.log(m) - np.log(t)).mean(-1)
        out = np.exp(logb)
    return np.where(ok, out, 0.0)


def mean_pooled(P: np.ndarray) -> np.ndarray:
    """A decomposable metric as a pooled statistic: rows (x_i, 1), so g(P) = Σ w x / Σ w is a weighted mean."""
    return P[..., 0] / P[..., 1]


def mean_rows(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, np.ones_like(x)])


def weighted(g, S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """g of the W-weighted pooled statistics, one value per row of W (Def 1's M_w)."""
    return g(W @ S)


def boot_counts(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Def 2: B multinomial(n, 1/n) count vectors, i.e. B random points n·w* of the simplex."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)


def bootstrap_deltas(g, SA: np.ndarray, SB: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Def 2: δ* = M_w*(A) − M_w*(B), the same weights for both systems (the paired bootstrap)."""
    return weighted(g, SA, W) - weighted(g, SB, W)


def swap_deltas(g, SA: np.ndarray, SB: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Def 3: δ under the swaps ε (rows of E, 1 = swap segment i's two outputs)."""
    E = E.astype(float)
    F = 1.0 - E
    return g(F @ SA + E @ SB) - g(E @ SA + F @ SB)


def swap_matrix(B: int, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((B, n)) < 0.5


def full_group(n: int) -> np.ndarray:
    """Every vertex of the swap cube {0, 1}ⁿ (small n only)."""
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=bool)


def mc_pvalue(t0: float, tb: np.ndarray) -> float:
    """Thm 1's Monte Carlo p-value (1 + #{b : T_b ≥ T})/(B + 1)."""
    return float((1 + np.sum(tb >= t0 - _tie(t0))) / (len(tb) + 1))


def group_pvalue(t0: float, tg: np.ndarray) -> float:
    """formalML rank-tests Thm 3's full-group p-value (1/|G|) Σ_g 1{T(gX) ≥ T(X)} (the identity is in G)."""
    return float(np.mean(tg >= t0 - _tie(t0)))


def ar_test(g, SA: np.ndarray, SB: np.ndarray, B: int, rng: np.random.Generator) -> dict:
    """Approximate randomization (Def 3), two-sided on |δ|."""
    d = float(g(SA.sum(0)) - g(SB.sum(0)))
    null = swap_deltas(g, SA, SB, swap_matrix(B, len(SA), rng))
    return {"d": d, "null": null, "p": mc_pvalue(abs(d), np.abs(null))}


# --------------------------------------------------------------------------------------------
# 2. Randomization is exact for any metric (Thm 1, Prop 1).
# --------------------------------------------------------------------------------------------


def mcnemar_exact(n10: int, n01: int) -> float:
    """McNemar's exact test (two-sided): the sign test on the discordant pairs, n₁₀ ~ Bin(n₁₀ + n₀₁, ½) under H₀."""
    nd = n10 + n01
    if nd == 0:
        return 1.0
    k = min(n10, n01)
    return float(min(1.0, 2 * stats.binom.cdf(k, nd, 0.5)))


def discordant(ca: np.ndarray, cb: np.ndarray) -> tuple[int, int]:
    """(n₁₀, n₀₁): segments A gets right and B wrong, and the reverse."""
    ca, cb = np.asarray(ca, bool), np.asarray(cb, bool)
    return int((ca & ~cb).sum()), int((~ca & cb).sum())


AA_STREAMS = (5010, 5011, 5012, 5013)  # Topic 27's trigram sampler, four fresh streams of 64 samples per prompt
AA_SCORED = 2  # the first two streams also carry Topic 27's six segment metrics (Prop 5's 64 experiments)
AA_HALF = om.MBR_SAMPLES // 2  # experiment (stream, j) pairs sample j with sample j + 32: 128 experiments
AA_N = (10, 20, 47, 94, 188)  # block sizes: disjoint prompt blocks are independent experiments
AA_B = 999
AA_METRICS = ("corpus-bleu",) + om.METRICS  # corpus BLEU, then Topic 27's six segment metrics (means)


@lru_cache(maxsize=1)
def aa_data() -> dict:
    """The exact A/A null. For every stream, prompt, and sample: BLEU statistics (and, on the first AA_SCORED streams,
    Topic 27's six segment scores) against the prompt's held-out continuations. Under the trigram, a prompt's samples
    are i.i.d., so sample j and sample j + 32 are exchangeable within every prompt: the sharp null of Thm 1 holds
    exactly."""
    refs = om.references()
    P = len(refs)
    St = np.zeros((len(AA_STREAMS), P, om.MBR_SAMPLES, K))
    Sc = np.full((len(AA_STREAMS), P, om.MBR_SAMPLES, len(om.METRICS)), np.nan)
    for a, stream in enumerate(AA_STREAMS):
        S = om.sample_set(stream)
        for p in range(P):
            for j, y in enumerate(S[p]):
                s = om.bleu_stats(y, refs[p])
                St[a, p, j] = [s.c, s.r, *s.m, *s.t]
                if a < AA_SCORED:
                    sc = om.segment_scores(y, refs[p])
                    Sc[a, p, j] = [sc[m] for m in om.METRICS]
    order = _rng(5020).permutation(P)  # prompts in a fixed random order, so blocks mix prompt types
    return {"stats": St[:, order], "scores": Sc[:, order]}


def aa_experiments(n: int):
    """Every A/A experiment with n prompts: (stream, j, block) → the two systems' rows, A = sample j, B = j + 32."""
    D = aa_data()
    for a in range(len(AA_STREAMS)):
        for j in range(AA_HALF):
            for b in range(D["stats"].shape[1] // n):
                sl = slice(b * n, (b + 1) * n)
                scored = a < AA_SCORED
                yield (D["stats"][a, sl, j], D["stats"][a, sl, j + AA_HALF],
                       D["scores"][a, sl, j] if scored else None, D["scores"][a, sl, j + AA_HALF] if scored else None)


AA_SHOP_N = (10, 20)  # the block sizes at which every metric is tested (Prop 5); elsewhere corpus BLEU only


def _metric_rows(m: str, St: np.ndarray, Sc: np.ndarray):
    if m == "corpus-bleu":
        return bleu_pooled, St
    return mean_pooled, mean_rows(Sc[:, om.METRICS.index(m)])


@lru_cache(maxsize=None)
def aa_pvalues(n: int) -> dict:
    """At block size n, over every A/A experiment: corpus BLEU's two-sided approximate-randomization p-value and its
    one-sided (A > B) bootstrap p-values by Koehn's rule and by the 2δ rule (Prop 2); and, at the AA_SHOP_N sizes on
    the scored streams, every metric's AR p-value (one swap draw per experiment, shared by the metrics: Prop 5)."""
    rng = _rng(5030 + n)
    ar, koehn, shift = [], [], []
    shop = {m: [] for m in AA_METRICS} if n in AA_SHOP_N else None
    for SA, SB, CA, CB in aa_experiments(n):
        E = swap_matrix(AA_B, n, rng)
        for m in (AA_METRICS if shop is not None and CA is not None else AA_METRICS[:1]):
            g, A = _metric_rows(m, SA, CA)
            Bm = _metric_rows(m, SB, CB)[1]
            d = float(g(A.sum(0)) - g(Bm.sum(0)))
            p = mc_pvalue(abs(d), np.abs(swap_deltas(g, A, Bm, E)))
            if m == "corpus-bleu":
                ar.append(p)
            if shop is not None and CA is not None:
                shop[m].append(p)
        W = boot_counts(n, AA_B, rng)
        d = float(bleu_pooled(SA.sum(0)) - bleu_pooled(SB.sum(0)))
        ds = bootstrap_deltas(bleu_pooled, SA, SB, W)
        koehn.append(koehn_p(ds))
        shift.append(shift_p(ds, d))
    return {"ar": np.array(ar), "koehn": np.array(koehn), "shift": np.array(shift),
            "shop": None if shop is None else {m: np.array(v) for m, v in shop.items()}}


def size(p: np.ndarray, alpha: float = ALPHA) -> float:
    return float(np.mean(p <= alpha))


def binomial_band(n: int, alpha: float, z: float = 3.0) -> float:
    """The upper edge of a z-sigma band around α for a rejection rate measured on n independent experiments."""
    return alpha + z * math.sqrt(alpha * (1 - alpha) / n)


# --------------------------------------------------------------------------------------------
# 3. The bootstrap's two p-values (Prop 2), the delta method (Thm 2), and the kink (Prop 3).
# --------------------------------------------------------------------------------------------


def koehn_p(ds: np.ndarray) -> float:
    """Koehn's rule as a one-sided p-value for 'A better than B': the share of resamples where A does not win."""
    return float(np.mean(ds <= 0))


def shift_p(ds: np.ndarray, d: float) -> float:
    """Berg-Kirkpatrick et al.'s rule: P*(δ* − δ̂ ≥ δ̂) = P*(δ* ≥ 2δ̂), the recentered (shifted-null) bootstrap."""
    return float(np.mean(ds >= 2 * d - _tie(d)))


def order_stat(ds: np.ndarray, alpha: float) -> tuple[float, float]:
    """Prop 2's empirical quantiles: q_α = δ*_(⌊αB⌋+1) and q_{1−α} = δ*_(B−⌊αB⌋) (1-based order statistics)."""
    s = np.sort(ds)
    k = int(math.floor(alpha * len(s)))
    return float(s[k]), float(s[len(s) - 1 - k])


def log_bleu_grad(mu: np.ndarray) -> np.ndarray:
    """∇ log BLEU at the mean statistics μ (c ≠ r, every m_n > 0): the brevity term contributes r/c² and −1/c below
    c = r and nothing above it; each precision contributes 1/(N m_n) and −1/(N t_n)."""
    c, r = mu[0], mu[1]
    gr = np.zeros(K)
    if c < r:
        gr[0], gr[1] = r / c ** 2, -1 / c
    gr[2:2 + N] = 1 / (N * mu[2:2 + N])
    gr[2 + N:] = -1 / (N * mu[2 + N:])
    return gr


def influences(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """Thm 2's per-segment influences ψ_i = ∇g(μ̂)·(Z_i − Z̄) of δ = BLEU(A) − BLEU(B), at the sample means."""
    muA, muB = SA.mean(0), SB.mean(0)
    gA = float(bleu_pooled(muA)) * log_bleu_grad(muA)
    gB = float(bleu_pooled(muB)) * log_bleu_grad(muB)
    return (SA - muA) @ gA - (SB - muB) @ gB


def linearized_var(SA: np.ndarray, SB: np.ndarray) -> float:
    """The delta method's variance of δ̂: (1/n²) Σ ψ_i², the bootstrap world's exact linear-term variance (undefined,
    nan, when a system matches no n-gram of some order: log BLEU has no gradient there)."""
    if (SA[:, 2:2 + N].sum(0) == 0).any() or (SB[:, 2:2 + N].sum(0) == 0).any():
        return math.nan
    psi = influences(SA, SB)
    return float(np.sum(psi ** 2) / len(psi) ** 2)


MIN_MATCHES = 50  # pooled matched 4-grams below which Thm 2's linearization is not yet trusted


def remainder_scaling(SA: np.ndarray, SB: np.ndarray, ns, B: int, rng: np.random.Generator) -> list[float]:
    """In the bootstrap world (the empirical distribution of the n segments as the population), the mean of
    √m·|δ*_m − δ̂ − ψ̄*_m| over resamples of size m: Thm 2's o_p(m^{−1/2}) remainder, which shrinks like m^{−1/2}."""
    n = len(SA)
    psi = influences(SA, SB)
    d = float(bleu_pooled(SA.mean(0)) - bleu_pooled(SB.mean(0)))
    out = []
    for m in ns:
        W = rng.multinomial(m, np.full(n, 1.0 / n), size=B).astype(float)
        ds = bootstrap_deltas(bleu_pooled, SA, SB, W)
        lin = W @ psi / m
        out.append(float(np.mean(np.sqrt(m) * np.abs(ds - d - lin))))
    return out


KINK_N, KINK_R, KINK_B = 100, 2000, 499
KINK_H = (-5.0, -1.0, -0.5, 0.0, 0.5, 1.0)  # the local parameter h = √n μ: far on the linear side, near, at, beyond
KINK_LEVEL = 0.90


@lru_cache(maxsize=None)
def kink_coverage(h: float, n: int = KINK_N, R: int = KINK_R) -> dict:
    """Prop 3 in its reduced form: φ(x̄) = min(0, x̄) for x̄ the mean of n i.i.d. N(h/√n, 1) (log BP is min(0, ·) of a
    smooth function of means, linear at c = r). Over R datasets, each bootstrapped KINK_B times (indices resampled):
    the share with φ̂ = 0; the coverage of φ(μ) by the nominal-90% percentile and basic intervals; and the mean and
    across-dataset spread of P*(φ* = φ̂), which a consistent bootstrap would drive to a constant."""
    rng = _rng(5100 + int(round(10 * h)) + 1000 * (n != KINK_N))
    mu = h / math.sqrt(n)
    X = rng.normal(mu, 1.0, (R, n))
    phi = min(0.0, mu)
    a = (1 - KINK_LEVEL) / 2
    zero, pct, basic, atom = 0, 0, 0, []
    for x in X:
        fs = np.minimum(0.0, x[rng.integers(0, n, (KINK_B, n))].mean(1))
        f = min(0.0, float(x.mean()))
        lo, hi = np.quantile(fs, [a, 1 - a])
        zero += f == 0.0
        pct += lo <= phi <= hi
        basic += 2 * f - hi <= phi <= 2 * f - lo
        atom.append(float(np.mean(fs == f)))
    return {"zero": zero / R, "percentile": pct / R, "basic": basic / R, "atom_mean": float(np.mean(atom)),
            "atom_sd": float(np.std(atom))}


# --------------------------------------------------------------------------------------------
# 4. Clustered segments (Def 4, Thm 3).
# --------------------------------------------------------------------------------------------


def design_effect(m: float, rho: float) -> float:
    """Thm 3(a): Var(x̄) = (σ²/n)(1 + (m − 1)ρ)."""
    return 1 + (m - 1) * rho


def equicorrelated(K_: int, m: int, sigma2: float, rho: float) -> np.ndarray:
    """The covariance of K_ independent clusters of m equicorrelated segments."""
    blk = sigma2 * ((1 - rho) * np.eye(m) + rho * np.ones((m, m)))
    return np.kron(np.eye(K_), blk)


def segment_var(x: np.ndarray) -> float:
    """The segment bootstrap's variance of the mean, exactly: σ̂²/n with σ̂² the 1/n variance."""
    return float(np.var(x) / len(x))


def cluster_var(x: np.ndarray, groups: np.ndarray) -> float:
    """Def 4's cluster bootstrap variance of the mean x̄ = Σ T_k/Σ m_k, linearized: Σ_k (T_k − m_k x̄)²/n²."""
    _, inv = np.unique(groups, return_inverse=True)
    T = np.bincount(inv, weights=x)
    m = np.bincount(inv)
    return float(np.sum((T - m * x.mean()) ** 2) / len(x) ** 2)


def cluster_boot_var(x: np.ndarray, groups: np.ndarray, B: int, rng: np.random.Generator) -> float:
    """Def 4 by Monte Carlo: resample whole clusters with replacement, carrying their segments."""
    _, inv = np.unique(groups, return_inverse=True)
    T = np.bincount(inv, weights=x)
    m = np.bincount(inv).astype(float)
    C = rng.multinomial(len(T), np.full(len(T), 1.0 / len(T)), size=B).astype(float)
    return float(np.var(C @ T / (C @ m)))


def icc_anova(x: np.ndarray, groups: np.ndarray) -> float:
    """The one-way ANOVA intraclass correlation for unequal cluster sizes: (MSB − MSW)/(MSB + (m₀ − 1)MSW), with
    m₀ = (n − Σ m_k²/n)/(K − 1)."""
    _, inv = np.unique(groups, return_inverse=True)
    m = np.bincount(inv).astype(float)
    T = np.bincount(inv, weights=x)
    n, Kc = len(x), len(m)
    gm = T / m
    msb = np.sum(m * (gm - x.mean()) ** 2) / (Kc - 1)
    msw = np.sum((x - gm[inv]) ** 2) / (n - Kc)
    m0 = (n - np.sum(m ** 2) / n) / (Kc - 1)
    return float((msb - msw) / (msb + (m0 - 1) * msw))


def contrast_pair_corr(x: np.ndarray, sent: np.ndarray, contrast: np.ndarray) -> float:
    """The Pearson correlation between the first and second rows of every two-aspect (contrast) sentence."""
    idx = np.flatnonzero(contrast)
    idx = idx[np.argsort(sent[idx], kind="stable")]
    a, b = x[idx[0::2]], x[idx[1::2]]
    assert np.array_equal(sent[idx[0::2]], sent[idx[1::2]])
    return float(np.corrcoef(a, b)[0, 1])


@lru_cache(maxsize=1)
def aspect_panel() -> dict:
    """Topic 19's test aspect rows: correctness of the aspect-blind bigram LR and of the stand-in lexicon reader, with
    each row's document, sentence, and contrast flag."""
    rs = ab.rows("test")
    y = ab.labels(rs)
    return {"blind": (ab.blind_predict(rs) == y).astype(float), "lexicon": (ab.lexicon_predict(rs) == y).astype(float),
            "doc": np.array([r.doc for r in rs]), "sent": np.array([r.doc * 1000 + r.sent for r in rs]),
            "contrast": np.array([r.contrast for r in rs])}


@lru_cache(maxsize=1)
def cluster_table() -> dict:
    """Design effects (cluster over segment bootstrap variance) for the two readers' accuracies and their paired
    difference, with documents and with sentences as clusters; with each, a within-cluster correlation: the ANOVA ICC
    for documents, and for sentences (almost all of one row) the correlation between the two rows of the contrast
    sentences."""
    P = aspect_panel()
    out = {}
    for name, x in (("blind", P["blind"]), ("lexicon", P["lexicon"]), ("difference", P["blind"] - P["lexicon"])):
        for unit in ("doc", "sent"):
            rho = icc_anova(x, P[unit]) if unit == "doc" else contrast_pair_corr(x, P["sent"], P["contrast"])
            out[f"{name}/{unit}"] = {"de": cluster_var(x, P[unit]) / segment_var(x), "icc": rho,
                                     "mean": float(x.mean()), "se_segment": math.sqrt(segment_var(x)),
                                     "se_cluster": math.sqrt(cluster_var(x, P[unit]))}
    out["docs"], out["sentences"], out["rows"] = len(set(P["doc"])), len(set(P["sent"])), len(P["doc"])
    return out


# --------------------------------------------------------------------------------------------
# 5. What a test set cannot tell you (Def 5, Thm 4).
# --------------------------------------------------------------------------------------------

TRAIN_WORLD = ltc.WorldConfig(30, True)  # Topic 17's bursty world, V = 101: NB misspecified, LR well specified
TRAIN_GRID = (256, 320, 352, 384, 416, 448, 480, 512, 1024)  # nested prefixes of one 1,024-document training set
TRAIN_DRAWS = 40
TRAIN_POP_N, TRAIN_TEST_N = 20000, 3000
NB_ALPHA, LR_LAMBDA = 1.0, 1e-2  # fixed: the procedure is "fit with these settings", no tuning


def _fit_pair(X, y):
    W, b = ltc.nb_fit(X, y, 2, NB_ALPHA)
    th = ltc.blr_intercept_fit(X, y.astype(float), LR_LAMBDA)
    return (W[1] - W[0], b[1] - b[0]), (th[:-1], th[-1])


@lru_cache(maxsize=1)
def training_draws() -> dict:
    """For each training size: TRAIN_DRAWS fresh training sets from the world (each draw's sizes are nested prefixes of
    one set); for each, the population accuracy gap Δ(D) = acc(LR) − acc(NB) on a 20,000-document draw, the test-set
    gap on a fixed 3,000-document test set, and the test-set McNemar p-value. Training draws come from one stream
    (5200), the evaluation sets from 5201 and 5202."""
    w = ltc.world(TRAIN_WORLD)
    Xp, yp = ltc.sample_world(w, TRAIN_POP_N, _rng(5201))
    Xt, yt = ltc.sample_world(w, TRAIN_TEST_N, _rng(5202))
    rng = _rng(5200)
    out = {n: [] for n in TRAIN_GRID}
    for _ in range(TRAIN_DRAWS):
        Xall, yall = ltc.sample_world(w, TRAIN_GRID[-1], rng)
        for n in TRAIN_GRID:
            rows = out[n]
            X, y = Xall[:n], yall[:n]
            (wn, bn), (wl, bl) = _fit_pair(X, y)
            pl, pn = (Xp @ wl + bl > 0) == yp, (Xp @ wn + bn > 0) == yp
            cl, cn = (Xt @ wl + bl > 0) == yt, (Xt @ wn + bn > 0) == yt
            n10, n01 = discordant(cl, cn)
            rows.append({"pop": float(pl.mean() - pn.mean()), "test": float(cl.mean() - cn.mean()), "n10": n10,
                         "n01": n01, "p": mcnemar_exact(n10, n01), "p_large": mcnemar_exact(*discordant(pl, pn))})
    return out


def crossover_size() -> int:
    """The training size whose across-draw mean population gap is closest to 0 (the procedure-level null)."""
    T = training_draws()
    return min(TRAIN_GRID, key=lambda n: abs(np.mean([r["pop"] for r in T[n]])))


def total_variance(n: int) -> dict:
    """Thm 4(a) at training size n: Var over (D, test) of the test-set gap, against E[Var(· | D)] + Var(Δ(D)), with the
    conditional variance of a paired accuracy difference, (p₁₀ + p₀₁ − (p₁₀ − p₀₁)²)/n_test, from the counts."""
    rows = training_draws()[n]
    t = np.array([r["test"] for r in rows])
    pop = np.array([r["pop"] for r in rows])
    within = np.mean([(r["n10"] + r["n01"]) / TRAIN_TEST_N - ((r["n10"] - r["n01"]) / TRAIN_TEST_N) ** 2
                      for r in rows]) / TRAIN_TEST_N
    return {"total": float(t.var(ddof=1)), "within": float(within), "between": float(pop.var(ddof=1)),
            "mean_pop": float(pop.mean()), "se_mean_pop": float(pop.std(ddof=1) / math.sqrt(len(pop))),
            "reject": float(np.mean([r["p"] <= ALPHA for r in rows])),
            "reject_lr": float(np.mean([r["p"] <= ALPHA and r["n10"] > r["n01"] for r in rows])),
            "reject_nb": float(np.mean([r["p"] <= ALPHA and r["n10"] < r["n01"] for r in rows])),
            "reject_large": float(np.mean([r["p_large"] <= ALPHA for r in rows])),
            "reject_large_lr": float(np.mean([r["p_large"] <= ALPHA and r["pop"] > 0 for r in rows])),
            "reject_large_nb": float(np.mean([r["p_large"] <= ALPHA and r["pop"] < 0 for r in rows]))}


# --------------------------------------------------------------------------------------------
# 6. Power and multiplicity (Prop 4, Prop 5).
# --------------------------------------------------------------------------------------------


def mcnemar_power(n: int, p10: float, p01: float, alpha: float = ALPHA) -> float:
    """Prop 4: the exact two-sided McNemar test's power. n_d ~ Bin(n, p_d); given n_d, n₁₀ ~ Bin(n_d, p₁₀/p_d); the
    test rejects when min(n₁₀, n₀₁) ≤ c(n_d), the largest k with 2·F_{Bin(n_d, ½)}(k) ≤ α."""
    pd = p10 + p01
    sd = math.sqrt(n * pd * (1 - pd))
    nd = np.arange(max(0, math.floor(n * pd - 12 * sd - 1)), min(n, math.ceil(n * pd + 12 * sd + 1)) + 1)  # Bin(n, p_d)
    # mass outside ±12 sd is below 1e-30
    c = stats.binom.ppf(alpha / 2, nd, 0.5)  # smallest k with F(k) ≥ α/2
    c = np.where(stats.binom.cdf(c, nd, 0.5) <= alpha / 2, c, c - 1)  # largest k with F(k) ≤ α/2
    q = p10 / pd
    rej = np.where(c >= 0, stats.binom.cdf(c, nd, q) + stats.binom.sf(nd - c - 1, nd, q), 0.0)
    rej = np.minimum(rej, 1.0)  # c ≥ n_d/2 never happens at α < 1, but keep the sum a probability
    return float(np.sum(stats.binom.pmf(nd, n, pd) * rej))


def connor_n(p10: float, p01: float, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Prop 4's normal approximation (Connor 1987): n = [z_{1−α/2}√p_d + z_{1−β}√(p_d − Δ²)]²/Δ², Δ = p₁₀ − p₀₁."""
    pd, dl = p10 + p01, p10 - p01
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za * math.sqrt(pd) + zb * math.sqrt(pd - dl ** 2)) ** 2 / dl ** 2)


CONNOR_CASES = ((0.12, 0.08), (0.10, 0.05), (0.20, 0.15), (0.06, 0.03))


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjusted p-values: max_{j ≤ i} (m − j + 1) p_(j), capped at 1 (formalStatistics Thm 3)."""
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m)
    out[o] = adj
    return out


def shopping_size(n: int, k: int, alpha: float = ALPHA) -> float:
    """Prop 5: the A/A rejection rate of 'some one of the first k metrics rejects at α'."""
    shop = aa_pvalues(n)["shop"]
    P = np.column_stack([shop[m] for m in AA_METRICS[:k]])
    return float(np.mean(P.min(1) <= alpha))


# --------------------------------------------------------------------------------------------
# 7. Finance case study: tone classifiers, and the summarizers' pair matrix.
# --------------------------------------------------------------------------------------------

FIN_PAIRS = ((("lr", False), ("nb", False)), (("lr", True), ("lr", False)), (("nbw", True), ("lr", True)),
             (("nbw", True), ("nbw", False)))  # a clear gap, a smaller one, a near-tie, and one more step


def _fin_correct(method: str, bigrams: bool) -> np.ndarray:
    r = ltc.fin_fit(ltc.FinConfig(method, bigrams))
    return r["logp"].argmax(1) == ltc.fin_split()[5]


def _fin_name(cfg) -> str:
    return f"{cfg[0]}-{'bi' if cfg[1] else 'uni'}"


@lru_cache(maxsize=1)
def classifier_table() -> list[dict]:
    """Each finance classifier pair: accuracies, discordant counts, exact McNemar p, Connor's n for 80% power at the
    observed discordance rates, and the exact power at the test set's 1,500 filings."""
    out = []
    for a, b in FIN_PAIRS:
        ca, cb = _fin_correct(*a), _fin_correct(*b)
        n10, n01 = discordant(ca, cb)
        n = len(ca)
        p10, p01 = n10 / n, n01 / n
        out.append({"a": _fin_name(a), "b": _fin_name(b), "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                    "n10": n10, "n01": n01, "n": n, "p": mcnemar_exact(n10, n01),
                    "n80": connor_n(p10, p01), "power": mcnemar_power(n, p10, p01)})
    return out


@lru_cache(maxsize=1)
def fin_stats() -> dict:
    """Per Topic 27 system: the 90 filings' BLEU statistics, and fact-F1 rows (decomposable)."""
    fs = om.finance_scores()
    return {s: {"bleu": stat_matrix(fs[s]["stats"]), "f1": mean_rows(fs[s]["f1"])} for s in om.SYSTEMS}


MATRIX_B = 9999


@lru_cache(maxsize=1)
def pair_matrix() -> dict:
    """All 45 summarizer pairs: δ and the two-sided approximate-randomization p-value for corpus BLEU and for fact F1
    (one stream, 5300), with Holm-adjusted p-values and Benjamini–Hochberg decisions at 0.05 for each metric."""
    F = fin_stats()
    rng = _rng(5300)
    pairs = list(itertools.combinations(om.SYSTEMS, 2))
    out = {"pairs": pairs}
    for metric, g in (("bleu", bleu_pooled), ("f1", mean_pooled)):
        res = [ar_test(g, F[a][metric], F[b][metric], MATRIX_B, rng) for a, b in pairs]
        p = np.array([r["p"] for r in res])
        thr = ck.bh_threshold(p, len(p), ALPHA)
        out[metric] = {"d": np.array([r["d"] for r in res]), "p": p, "holm": holm_adjust(p),
                       "bh": (p <= thr) & (thr > 0)}
    return out


def metric_verdicts() -> dict:
    """Where corpus BLEU and fact F1 reach different verdicts at 0.05 (two-sided AR): pairs significant on one metric
    only, and pairs significant on both but in opposite directions."""
    M = pair_matrix()
    b, f = M["bleu"], M["f1"]
    sb, sf = b["p"] <= ALPHA, f["p"] <= ALPHA
    opp = sb & sf & (np.sign(b["d"]) != np.sign(f["d"]))
    return {"one_only": [M["pairs"][i] for i in np.flatnonzero(sb != sf)],
            "opposite": [M["pairs"][i] for i in np.flatnonzero(opp)],
            "both_agree": int((sb & sf & ~opp).sum()), "neither": int((~sb & ~sf).sum())}


CLOUD_PAIRS = (("writer", "noisy-0.1"), ("noisy-0.3", "noisy-0.5"), ("terse", "flipped"),
               ("extract", "textrank-3"), ("textrank-3", "random-3"))
CLOUD_N = (10, 30, 90)
CLOUD_B = 4000
CLOUD_BINS = 40


@lru_cache(maxsize=1)
def clouds() -> dict:
    """The lab's first tab: for each pair and subset size (the first n filings of one fixed permutation, stream 5400),
    the bootstrap and randomization clouds of corpus-BLEU δ, with Koehn's, the 2δ rule's, and AR's p-values and the
    linearized (Thm 2) and bootstrap standard deviations."""
    F = fin_stats()
    order = _rng(5400).permutation(90)
    rng = _rng(5401)
    out = {}
    for a, b in CLOUD_PAIRS:
        for n in CLOUD_N:
            idx = order[:n]
            SA, SB = F[a]["bleu"][idx], F[b]["bleu"][idx]
            d = float(bleu_pooled(SA.sum(0)) - bleu_pooled(SB.sum(0)))
            ds = bootstrap_deltas(bleu_pooled, SA, SB, boot_counts(n, CLOUD_B, rng))
            ar = ar_test(bleu_pooled, SA, SB, CLOUD_B, rng)
            out[(a, b, n)] = {"d": d, "boot": ds, "null": ar["null"], "p_ar": ar["p"], "p_koehn": koehn_p(ds),
                              "p_shift": shift_p(ds, d), "sd_boot": float(ds.std()),
                              "sd_lin": math.sqrt(linearized_var(SA, SB))}
    return out


POWER_PAIRS = (("writer", "noisy-0.1"), ("writer", "noisy-0.3"), ("writer", "noisy-0.5"))
POWER_N = (5, 10, 15, 20, 30, 45)
POWER_REPS, POWER_B = 200, 499
POWER_METRICS = ("bleu", "chrf", "f1")


@lru_cache(maxsize=1)
def power_curves() -> dict:
    """Subsampling power: for each flip-rate pair and metric, the share of POWER_REPS random n-filing subsets (without
    replacement) where two-sided AR rejects at 0.05. The truth is known: every flip lowers fact F1."""
    F = fin_stats()
    fs = om.finance_scores()
    rng = _rng(5500)
    out = {}
    for a, b in POWER_PAIRS:
        for metric in POWER_METRICS:
            if metric == "bleu":
                g, SA, SB = bleu_pooled, F[a]["bleu"], F[b]["bleu"]
            elif metric == "f1":
                g, SA, SB = mean_pooled, F[a]["f1"], F[b]["f1"]
            else:
                g, SA, SB = mean_pooled, mean_rows(fs[a]["segment"]["chrf"]), mean_rows(fs[b]["segment"]["chrf"])
            curve = []
            for n in POWER_N:
                rej = 0
                for _ in range(POWER_REPS):
                    idx = rng.choice(90, n, replace=False)
                    rej += ar_test(g, SA[idx], SB[idx], POWER_B, rng)["p"] <= ALPHA
                curve.append(rej / POWER_REPS)
            out[(a, b, metric)] = curve
    return out


# --------------------------------------------------------------------------------------------
# The lab's payload.
# --------------------------------------------------------------------------------------------


def _hist(x: np.ndarray, lo: float, hi: float, bins: int) -> list[int]:
    return np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))[0].tolist()


def _clouds_payload() -> list[dict]:
    out = []
    for (a, b, n), c in clouds().items():
        lo = float(min(c["boot"].min(), c["null"].min()))
        hi = float(max(c["boot"].max(), c["null"].max()))
        out.append({"a": a, "b": b, "n": n, "d": _r(c["d"], 4), "lo": _r(lo, 4), "hi": _r(hi, 4),
                    "boot": _hist(c["boot"], lo, hi, CLOUD_BINS), "null": _hist(c["null"], lo, hi, CLOUD_BINS),
                    "p_ar": _r(c["p_ar"], 4), "p_koehn": _r(c["p_koehn"], 4), "p_shift": _r(c["p_shift"], 4),
                    "sd_boot": _r(c["sd_boot"], 4), "sd_lin": _r(c["sd_lin"], 4)})
    return out


def _aa_payload() -> dict:
    out = {"n": list(AA_N), "metrics": list(AA_METRICS), "rows": []}
    for n in AA_N:
        P = aa_pvalues(n)
        out["rows"].append({"n": n, "experiments": len(P["koehn"]),
                            "hist_ar": _hist(P["ar"], 0, 1, 20), "hist_koehn": _hist(P["koehn"], 0, 1, 20),
                            "hist_shift": _hist(P["shift"], 0, 1, 20), "size_ar": _r(size(P["ar"])),
                            "size_koehn": _r(size(P["koehn"])), "size_shift": _r(size(P["shift"])),
                            "shopping": ([_r(shopping_size(n, k)) for k in range(1, len(AA_METRICS) + 1)]
                                         if n in AA_SHOP_N else None)})
    return out


def _mcnemar_payload() -> list[dict]:
    grid = [int(round(x)) for x in np.geomspace(100, 200000, 40)]
    out = []
    for row in classifier_table():
        p10, p01 = row["n10"] / row["n"], row["n01"] / row["n"]
        out.append({k: (_r(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                   | {"p": float(f"{row['p']:.3g}"), "n80": math.ceil(row["n80"]), "grid": grid,
                      "power_curve": [_r(mcnemar_power(m, p10, p01)) for m in grid]})
    return out


def _clusters_payload() -> dict:
    T = cluster_table()
    return {"rows": T["rows"], "docs": T["docs"], "sentences": T["sentences"],
            "table": {k: {kk: _r(vv, 4) for kk, vv in v.items()} for k, v in T.items() if isinstance(v, dict)}}


def _draws_payload() -> dict:
    n = crossover_size()
    tv = total_variance(n)
    return {"n": n, "test_n": TRAIN_TEST_N, "large_n": TRAIN_POP_N,
            "draws": [{"pop": _r(r["pop"], 4), "test": _r(r["test"], 4), "p": _r(r["p"], 4),
                       "p_large": _r(r["p_large"], 4), "n10": r["n10"], "n01": r["n01"]} for r in training_draws()[n]],
            "summary": {k: _r(v, 6) for k, v in tv.items()}}


def _matrix_payload() -> dict:
    M = pair_matrix()
    return {"systems": list(om.SYSTEMS), "pairs": [list(p) for p in M["pairs"]],
            **{m: {"d": _r(M[m]["d"], 4), "p": _r(M[m]["p"], 4), "holm": _r(M[m]["holm"], 4),
                   "bh": [bool(x) for x in M[m]["bh"]]} for m in ("bleu", "f1")}}


def _power_payload() -> dict:
    return {"n": list(POWER_N), "metrics": list(POWER_METRICS), "pairs": [list(p) for p in POWER_PAIRS],
            "curves": {f"{a}|{b}|{m}": _r(v) for (a, b, m), v in power_curves().items()}}


def viz_payload() -> dict:
    return {"clouds": _clouds_payload(), "mcnemar": _mcnemar_payload(), "power": _power_payload(),
            "aa": _aa_payload(), "clusters": _clusters_payload(), "draws": _draws_payload(),
            "matrix": _matrix_payload()}


# --------------------------------------------------------------------------------------------
# The harness.
# --------------------------------------------------------------------------------------------


def test_def1_pooled_bleu_matches_topic27():
    """Def 1: the vectorized pooled BLEU equals Topic 27's corpus BLEU for every system, on the full test set and on
    random subsets, and is homogeneous of degree 0 (weights on the simplex give the same value as counts)."""
    F = fin_stats()
    fs = om.finance_scores()
    rng = _rng(5600)
    for s in om.SYSTEMS:
        assert abs(float(bleu_pooled(F[s]["bleu"].sum(0))) - om.system_score(s, "bleu")) < 1e-12
        idx = rng.choice(90, 17, replace=False)
        ref = om.bleu(om.pool(fs[s]["stats"][i] for i in idx))
        assert abs(float(bleu_pooled(F[s]["bleu"][idx].sum(0))) - ref) < 1e-12
        W = boot_counts(90, 5, rng)
        assert np.allclose(weighted(bleu_pooled, F[s]["bleu"], W), weighted(bleu_pooled, F[s]["bleu"], W / 90),
                           atol=1e-13)


def test_thm1_exact_aa_size():
    """Thm 1: under the exact A/A null, approximate randomization's rejection rate at 0.05 and 0.10 stays inside a
    3σ band above α at every block size, for every metric; and its p-values are never below 1/(B + 1)."""
    for n in AA_N:
        P = aa_pvalues(n)
        for metric, p in ({"corpus-bleu": P["ar"]} | (P["shop"] or {})).items():
            assert p.min() >= 1 / (AA_B + 1)
            for alpha in (0.05, 0.10):
                assert size(p, alpha) <= binomial_band(len(p), alpha), (n, metric, alpha, size(p, alpha))


SUPER_ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def test_thm1_aa_pvalues_are_superuniform():
    """Thm 1 says P(p ≤ α) ≤ α, not that p is uniform: at the smallest block size (2,304 independent experiments) the
    corpus-BLEU randomization p-values' empirical CDF stays inside the 3σ band above α at every α on a grid, and sits
    below α at the larger ones (many swaps of a 10-prompt block change nothing, so p has atoms and the test is
    conservative)."""
    p = aa_pvalues(AA_N[0])["ar"]
    for alpha in SUPER_ALPHAS:
        assert size(p, alpha) <= binomial_band(len(p), alpha), (alpha, size(p, alpha))
    assert size(p, 0.5) < 0.5 - 3 * math.sqrt(0.25 / len(p))


def test_prop2_bootstrap_size_is_asymptotic():
    """Thm 1 against the bootstrap at 10 prompts: the 2δ (shifted-null) rule's A/A size at 0.05 is above its 3σ band
    while randomization's is inside it; by 94 prompts both bootstrap rules are back inside the band."""
    P = aa_pvalues(AA_N[0])
    m = len(P["shift"])
    assert size(P["shift"]) > binomial_band(m, ALPHA) and size(P["ar"]) <= binomial_band(m, ALPHA)
    P = aa_pvalues(94)
    for k in ("koehn", "shift"):
        assert size(P[k]) <= binomial_band(len(P[k]), ALPHA), (k, size(P[k]))


def test_thm1_monte_carlo_vs_full_group():
    """Thm 1: on a 12-filing subset, the Monte Carlo p-value of corpus-BLEU δ approaches the full-group p-value over
    all 4,096 swaps (within 3 Monte Carlo standard errors), and both are valid p-values on the same data."""
    F = fin_stats()
    idx = _rng(5601).choice(90, 12, replace=False)
    for a, b in (("textrank-3", "random-3"), ("writer", "noisy-0.3")):
        SA, SB = F[a]["bleu"][idx], F[b]["bleu"][idx]
        d = float(bleu_pooled(SA.sum(0)) - bleu_pooled(SB.sum(0)))
        pg = group_pvalue(abs(d), np.abs(swap_deltas(bleu_pooled, SA, SB, full_group(12))))
        B = 20000
        pm = ar_test(bleu_pooled, SA, SB, B, _rng(5602))["p"]
        assert abs(pm - pg) <= 3 * math.sqrt(pg * (1 - pg) / B) + 1 / B, (a, b, pm, pg)


def test_thm1_matches_scipy_permutation_test():
    """Cross-check: on a decomposable metric (mean fact F1) with 12 filings, the full swap group's one-sided p-value
    equals scipy's exact permutation_test with permutation_type='samples'."""
    fs = om.finance_scores()
    idx = _rng(5603).choice(90, 12, replace=False)
    x, y = fs["noisy-0.3"]["f1"][idx], fs["noisy-0.5"]["f1"][idx]
    E = full_group(12)
    td = swap_deltas(mean_pooled, mean_rows(x), mean_rows(y), E)
    ours = group_pvalue(float(x.mean() - y.mean()), td)
    sp = stats.permutation_test((x, y), lambda u, v: np.mean(u - v), permutation_type="samples",
                                alternative="greater", n_resamples=10 ** 6)
    assert abs(ours - sp.pvalue) < 1e-12, (ours, sp.pvalue)


def test_prop1_randomization_is_mcnemar():
    """Prop 1: on 0/1 correctness, the full-group randomization p-value equals exact McNemar (and scipy's binomtest)
    on small subsets enumerated exactly; on the full 1,500 filings the Monte Carlo AR p-value agrees with exact
    McNemar within Monte Carlo error; and concordant filings never move δ."""
    rng = _rng(5604)
    for a, b in FIN_PAIRS:
        ca, cb = _fin_correct(*a), _fin_correct(*b)
        for _ in range(3):
            idx = rng.choice(len(ca), 14, replace=False)
            A, Bm = mean_rows(ca[idx]), mean_rows(cb[idx])
            d = float(ca[idx].mean() - cb[idx].mean())
            pg = group_pvalue(abs(d), np.abs(swap_deltas(mean_pooled, A, Bm, full_group(14))))
            n10, n01 = discordant(ca[idx], cb[idx])
            assert abs(pg - mcnemar_exact(n10, n01)) < 1e-12
            if n10 + n01:
                assert abs(mcnemar_exact(n10, n01) - stats.binomtest(n10, n10 + n01, 0.5).pvalue) < 1e-12
        n10, n01 = discordant(ca, cb)
        B = 20000
        pm = ar_test(mean_pooled, mean_rows(ca), mean_rows(cb), B, rng)["p"]
        pe = mcnemar_exact(n10, n01)
        assert abs(pm - pe) <= 3 * math.sqrt(pe * (1 - pe) / B) + 2 / B, (a, b, pm, pe)
    conc = _fin_correct("lr", True) == _fin_correct("nbw", True)
    E = swap_matrix(50, len(conc), rng)
    A, Bm = mean_rows(_fin_correct("nbw", True)), mean_rows(_fin_correct("lr", True))
    E2 = E & ~conc  # swapping only discordant filings gives the same δ as swapping everything chosen
    assert np.allclose(swap_deltas(mean_pooled, A, Bm, E), swap_deltas(mean_pooled, A, Bm, E2), atol=1e-14)


def test_prop2_inversions_and_symmetry():
    """Prop 2: Koehn's p ≤ α iff q_α > 0, and the 2δ rule's p ≤ α iff 2δ̂ − q_{1−α} > 0, at every α on a grid, for
    every cloud; on a cloud symmetrized about δ̂ the two p-values are equal; and on writer vs noisy-0.1 (all 90
    filings) they differ."""
    for c in clouds().values():
        ds, d = c["boot"], c["d"]
        for alpha in (0.01, 0.025, 0.05, 0.1, 0.2):
            lo, hi = order_stat(ds, alpha)
            assert (koehn_p(ds) <= alpha) == (lo > 0)
            assert (shift_p(ds, d) <= alpha) == (2 * d - hi > _tie(d))
        sym = np.concatenate([ds, 2 * d - ds])
        assert abs(koehn_p(sym) - shift_p(sym, d)) <= 1 / len(sym)
    c = clouds()[("writer", "noisy-0.1", 90)]
    assert c["p_shift"] > c["p_koehn"] + 0.001


def test_thm2_gradient_finite_differences():
    """Thm 2: the analytic ∇ log BLEU matches central differences at every system's mean statistics (c ≠ r)."""
    F = fin_stats()
    for s in om.SYSTEMS:
        mu = F[s]["bleu"].mean(0)
        g = log_bleu_grad(mu)
        for k in range(K):
            h = 1e-6 * mu[k]
            e = np.zeros(K)
            e[k] = h
            fd = (math.log(bleu_pooled(mu + e)) - math.log(bleu_pooled(mu - e))) / (2 * h)
            assert abs(fd - g[k]) <= 1e-6 * max(1.0, abs(g[k])), (s, k, fd, g[k])


def test_thm2_bootstrap_variance_matches_linearization():
    """Thm 2: at n = 90 the bootstrap standard deviation of corpus-BLEU δ is within 5% of the delta method's for every
    cloud pair whose systems both pool at least 50 matched 4-grams (the extractors pool 3–7, and their linearization is
    off by more), and in the bootstrap world the remainder √m·|δ* − δ̂ − ψ̄*| shrinks about like m^{−1/2}."""
    F = fin_stats()
    for (a, b, n), c in clouds().items():
        if n == 90 and min(F[a]["bleu"][:, 1 + N].sum(), F[b]["bleu"][:, 1 + N].sum()) >= MIN_MATCHES:
            assert abs(c["sd_boot"] / c["sd_lin"] - 1) < 0.05, (a, b, c["sd_boot"], c["sd_lin"])
    rem = remainder_scaling(F["writer"]["bleu"], F["noisy-0.3"]["bleu"], (30, 300, 3000), 2000, _rng(5605))
    assert rem[0] > rem[1] > rem[2] and rem[0] / rem[2] > 5, rem


def test_prop3_kink():
    """Prop 3: at the kink (h = 0) half the datasets have φ̂ = 0 (the limit min(0, aW) has an atom of ½); the bootstrap's
    own atom P*(φ* = φ̂) keeps an across-dataset spread above 0.3 at n = 100 and at n = 400 (its law stays random rather
    than converging); just on the linear side (h = −0.5) the basic interval covers below 0.7 at nominal 0.9, while at
    the kink both intervals over-cover; far from it (h = −5) both cover within 0.03 of 0.9 and the atom is gone."""
    k0, kn, kf = kink_coverage(0.0), kink_coverage(-0.5), kink_coverage(-5.0)
    assert abs(k0["zero"] - 0.5) < 3 * math.sqrt(0.25 / KINK_R), k0
    assert k0["atom_sd"] > 0.3 and kink_coverage(0.0, 400, 500)["atom_sd"] > 0.3
    assert kn["basic"] < 0.7 and k0["percentile"] > KINK_LEVEL and k0["basic"] > KINK_LEVEL, (kn, k0)
    assert abs(kf["percentile"] - KINK_LEVEL) < 0.03 and abs(kf["basic"] - KINK_LEVEL) < 0.03, kf
    assert kf["zero"] == 0.0 and kf["atom_sd"] == 0.0


def test_prop3_flip_family_sits_at_the_kink():
    """Prop 3's remark: every flip-family system's pooled c/r is within 1% of 1, and its brevity penalty is the same
    for both systems of a pair on every resample (they share lengths), so it cancels from their δ."""
    F = fin_stats()
    W = boot_counts(90, 200, _rng(5606))
    lp = None
    for s in om.FLIP_FAMILY:
        P = F[s]["bleu"].sum(0)
        assert abs(P[0] / P[1] - 1) < 0.01
        Q = W @ F[s]["bleu"]
        bp = np.minimum(0, 1 - Q[:, 1] / Q[:, 0])
        assert lp is None or np.array_equal(bp, lp)
        lp = bp


def test_thm3_design_effect_algebra():
    """Thm 3(a, b): with an equicorrelated covariance, Var(x̄) = (σ²/n)(1 + (m − 1)ρ) and the expected segment-bootstrap
    variance is (σ²/n)(1 − (1 + (m − 1)ρ)/n), exactly, including negative ρ ≥ −1/(m − 1)."""
    for Kc, m, rho in ((5, 4, 0.3), (6, 3, -0.4), (4, 2, -1.0), (3, 6, 0.9)):
        n, s2 = Kc * m, 1.7
        S = equicorrelated(Kc, m, s2, rho)
        one = np.ones(n)
        assert abs(one @ S @ one / n ** 2 - s2 / n * design_effect(m, rho)) < 1e-12
        C = np.eye(n) - np.ones((n, n)) / n
        assert abs(np.trace(C @ S) / n ** 2 - s2 / n * (1 - design_effect(m, rho) / n)) < 1e-12


def test_thm3_cluster_bootstrap_is_consistent():
    """Thm 3(c): on simulated clusters (K = 400, m = 8, ρ = 0.3) the cluster bootstrap's Monte Carlo variance matches its
    linearized form and the true design effect, while the segment bootstrap's falls short by 1/(1 + 7ρ)."""
    rng = _rng(5607)
    Kc, m, rho = 400, 8, 0.3
    x = (math.sqrt(rho) * np.repeat(rng.normal(size=Kc), m) + math.sqrt(1 - rho) * rng.normal(size=Kc * m))
    g = np.repeat(np.arange(Kc), m)
    lin, mc = cluster_var(x, g), cluster_boot_var(x, g, 4000, rng)
    assert abs(mc / lin - 1) < 0.08, (mc, lin)
    truth = design_effect(m, rho) / (Kc * m)
    assert abs(lin / truth - 1) < 0.25 and abs(segment_var(x) * design_effect(m, rho) / truth - 1) < 0.1


def test_thm3_aspect_rows():
    """Thm 3 on Topic 19's aspect rows: the aspect-blind reader is right on exactly one aspect of every contrast
    sentence, so with sentences as clusters its design effect is below 1; with documents as clusters the lexicon
    reader's accuracy has a design effect above 1.3, while the readers' paired difference has one below 1.1."""
    P = aspect_panel()
    cs = P["contrast"]
    _, inv = np.unique(P["sent"][cs], return_inverse=True)
    assert np.all(np.bincount(inv, weights=P["blind"][cs]) == 1) and np.all(np.bincount(inv) == 2)
    T = cluster_table()
    assert T["blind/sent"]["de"] < 1
    assert T["lexicon/doc"]["de"] > 1.3 and T["lexicon/doc"]["icc"] > 0
    assert T["difference/doc"]["de"] < 1.1


def test_thm4_total_variance_and_both_directions():
    """Thm 4: near the crossover (the training size whose mean population gap is within 2 across-draw standard errors
    of 0, so the procedure-level null holds to the draws' precision), the between-draw variance is the larger part of
    the total and the law of total variance holds to within 30%; test-set McNemar on 3,000 documents rejects above its
    3σ band (40 draws), in both directions, and on 20,000 documents it rejects more often still, in both directions."""
    n = crossover_size()
    tv = total_variance(n)
    assert abs(tv["mean_pop"]) < 2 * tv["se_mean_pop"], tv
    assert tv["between"] > tv["within"], tv
    assert abs((tv["within"] + tv["between"]) / tv["total"] - 1) < 0.3, tv
    assert tv["reject"] > binomial_band(TRAIN_DRAWS, ALPHA) and tv["reject_lr"] > 0 and tv["reject_nb"] > 0, tv
    assert tv["reject_large"] > tv["reject"] and tv["reject_large_lr"] > 0 and tv["reject_large_nb"] > 0, tv


def test_prop4_connor_n_has_about_80_percent_power():
    """Prop 4: the exact McNemar power at Connor's n (rounded up) is within 0.03 of 0.80 for every case; exact power
    rises with n; and the finance near-tie (NB-weighted bigram vs LR bigram) has power below 0.15 at 1,500 filings and
    needs more than 20,000 for 80%."""
    for p10, p01 in CONNOR_CASES:
        n = math.ceil(connor_n(p10, p01))
        assert abs(mcnemar_power(n, p10, p01) - 0.8) < 0.03, (p10, p01, n, mcnemar_power(n, p10, p01))
        assert mcnemar_power(n // 2, p10, p01) < mcnemar_power(n, p10, p01) < mcnemar_power(2 * n, p10, p01)
    tie = [r for r in classifier_table() if (r["a"], r["b"]) == ("nbw-bi", "lr-bi")][0]
    assert tie["p"] > 0.5 and tie["power"] < 0.15 and tie["n80"] > 20000, tie


def test_prop5_metric_shopping():
    """Prop 5: under the exact A/A null, the size of 'any of k metrics rejects' rises with k, stays at most kα (within
    its band), and is below 1 − (1 − α)^k at k = 7 (the metrics are positively dependent)."""
    for n in (AA_N[0], AA_N[1]):
        m = len(aa_pvalues(n)["shop"]["corpus-bleu"])
        sz = [shopping_size(n, k) for k in range(1, len(AA_METRICS) + 1)]
        assert all(a <= b for a, b in zip(sz, sz[1:]))
        assert all(s <= binomial_band(m, k * ALPHA) for k, s in zip(range(1, 8), sz))
        assert sz[-1] < 1 - (1 - ALPHA) ** len(AA_METRICS)
        assert sz[-1] > binomial_band(m, ALPHA)  # shopping does inflate the size


def test_holm_and_bh():
    """Cited results, computed: Holm's adjusted p-values are monotone in the raw ones and never below them, Holm
    rejects a subset of BH's rejections at 0.05, and the flip family's corpus-BLEU gaps survive Holm."""
    M = pair_matrix()
    for metric in ("bleu", "f1"):
        p, h, bh = M[metric]["p"], M[metric]["holm"], M[metric]["bh"]
        assert np.all(h >= p - 1e-15)
        o = np.argsort(p, kind="stable")
        assert np.all(np.diff(h[o]) >= -1e-15)
        assert np.all(~(h <= ALPHA) | bh)
    for a, b in zip(om.FLIP_FAMILY, om.FLIP_FAMILY[1:]):
        i = M["pairs"].index((a, b))
        assert M["bleu"]["holm"][i] <= ALPHA, (a, b, M["bleu"]["holm"][i])


def test_case_study_the_metric_decides():
    """Finance case study: significance belongs to the metric. Of the 45 summarizer pairs, 9 are significant on one of
    corpus BLEU and fact F1 only (writer vs extract on BLEU alone, terse vs flipped on F1 alone), and 8 more are
    significant on both in opposite directions (flipped vs extract, textrank-3, and random-3 among them; flipped vs
    lead-3 is significant on BLEU alone, both fact F1s being 0)."""
    v = metric_verdicts()
    assert len(v["one_only"]) == 9 and len(v["opposite"]) == 8, v
    assert ("writer", "extract") in v["one_only"] and ("terse", "flipped") in v["one_only"]
    assert {("flipped", "extract"), ("flipped", "textrank-3"), ("flipped", "random-3")} <= set(v["opposite"])
    assert len(v["one_only"]) + len(v["opposite"]) + v["both_agree"] + v["neither"] == 45


def test_power_rises_with_n_and_flip_rate():
    """Subsampling power on the flip family rises with n (up to Monte Carlo noise of 0.08) and with the flip rate at
    every n, for every metric."""
    C = power_curves()
    for metric in POWER_METRICS:
        for a, b in POWER_PAIRS:
            c = C[(a, b, metric)]
            assert all(y >= x - 0.08 for x, y in zip(c, c[1:])), (a, b, metric, c)
        for k in range(len(POWER_N)):
            col = [C[(a, b, metric)][k] for a, b in POWER_PAIRS]
            assert col[0] <= col[1] + 0.08 <= col[2] + 0.16, (metric, k, col)


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def summary() -> dict:
    """The numbers the page quotes, at 6 decimals."""
    out: dict = {}
    for n in AA_N:
        P = aa_pvalues(n)
        out[f"A/A n={n}: experiments, size AR, Koehn, 2δ"] = (f"{len(P['koehn'])}, {size(P['ar']):.6f}, "
                                                             f"{size(P['koehn']):.6f}, {size(P['shift']):.6f}")
        if n in AA_SHOP_N:
            out[f"A/A n={n}: shopping k=1..7"] = ", ".join(f"{shopping_size(n, k):.6f}" for k in range(1, 8))
    for (a, b, n), c in clouds().items():
        out[f"cloud {a} vs {b} n={n}: δ, p AR, Koehn, 2δ, sd boot, sd lin"] = (
            f"{c['d']:.6f}, {c['p_ar']:.6f}, {c['p_koehn']:.6f}, {c['p_shift']:.6f}, {c['sd_boot']:.6f}, "
            f"{c['sd_lin']:.6f}")
    for h in KINK_H:
        k = kink_coverage(h)
        out[f"Prop 3 h={h:g}: φ̂=0, percentile, basic, atom mean, sd"] = (
            f"{k['zero']:.6f}, {k['percentile']:.6f}, {k['basic']:.6f}, {k['atom_mean']:.6f}, {k['atom_sd']:.6f}")
    out["Prop 3 h=0 n=400: atom sd"] = kink_coverage(0.0, 400, 500)["atom_sd"]
    F = fin_stats()
    out["flip family pooled c/r"] = float(F["writer"]["bleu"].sum(0)[0] / F["writer"]["bleu"].sum(0)[1])
    T = cluster_table()
    for k, v in T.items():
        if isinstance(v, dict):
            out[f"clusters {k}: mean, DE, ICC, SE seg, SE clu"] = (f"{v['mean']:.6f}, {v['de']:.6f}, {v['icc']:.6f}, "
                                                                 f"{v['se_segment']:.6f}, {v['se_cluster']:.6f}")
    out["clusters rows / docs / sentences"] = f"{T['rows']}, {T['docs']}, {T['sentences']}"
    out["Thm 4 crossover n"] = crossover_size()
    for n in TRAIN_GRID:
        tv = total_variance(n)
        out[f"Thm 4 n={n}: mean Δ, se, total, within, between, reject, LR, NB, reject 20k"] = ", ".join(
            f"{tv[k]:.6f}" for k in ("mean_pop", "se_mean_pop", "total", "within", "between", "reject", "reject_lr",
                                     "reject_nb", "reject_large"))
    for r in classifier_table():
        out[f"McNemar {r['a']} vs {r['b']}: acc, acc, n10, n01, p, n80, power"] = (
            f"{r['acc_a']:.6f}, {r['acc_b']:.6f}, {r['n10']}, {r['n01']}, {r['p']:.6f}, {r['n80']:.1f}, {r['power']:.6f}")
    for p10, p01 in CONNOR_CASES:
        n = math.ceil(connor_n(p10, p01))
        out[f"Connor ({p10}, {p01}): n, exact power"] = f"{n}, {mcnemar_power(n, p10, p01):.6f}"
    M = pair_matrix()
    for metric in ("bleu", "f1"):
        out[f"matrix {metric}: raw, Holm, BH rejections of 45"] = (
            f"{int((M[metric]['p'] <= ALPHA).sum())}, {int((M[metric]['holm'] <= ALPHA).sum())}, "
            f"{int(M[metric]['bh'].sum())}")
    v = metric_verdicts()
    out["BLEU vs F1 verdicts: one only, opposite, agree, neither"] = (
        f"{len(v['one_only'])}, {len(v['opposite'])}, {v['both_agree']}, {v['neither']}")
    for (a, b, metric), c in power_curves().items():
        out[f"power {a} vs {b} {metric}"] = ", ".join(f"{x:.6f}" for x in c)
    return out


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
    if failed == 0:
        print("\nSummary:")
        for k, v in summary().items():
            print(f"  {k:70} {v:.6f}" if isinstance(v, float) else f"  {k:70} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
