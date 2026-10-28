"""Zipf's and Heaps' laws — the reference implementation for formalNLP's first topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. Rank–frequency is the type-count tail turned sideways (exact identity; exponent map α = 1 + 1/s).
2. Zipf as a probability law: ζ(s) < ∞ iff s > 1, and Zipf maximizes entropy at fixed E[log r].
3. Zipf ⇒ Heaps (Karlin 1967): E V(n) ~ Γ(1 − 1/s)(Cn)^{1/s}, so β = 1/s; the hapax share → 1/s.
4. Pitman–Yor plants both laws: E[K_n] + θ/d = (θ/d)·Γ(θ+n+d)Γ(θ)/(Γ(θ+d)Γ(θ+n)) ~ c·n^d.
5. Estimating the exponent: discrete power-law MLE (Hurwitz zeta) versus log–log OLS.
6. Finance case study: the shared corpus plants d = HEAPS_DISCOUNT, a known answer — which the generic
   asymptotic estimators miss at corpus scale and Theorem 3's exact offset recovers.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/zipf-heaps-vocabulary-statistics/zipf_heaps_vocabulary_statistics.py
"""

from __future__ import annotations

import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finance-corpus"))
import finance_corpus as fc  # noqa: E402

# ONE stream for every Monte-Carlo claim below (STARTER-PROMPT A1).
SEED = 20261022


# --------------------------------------------------------------------------------------------
# 1. Rank–frequency and the frequency spectrum.
# --------------------------------------------------------------------------------------------


def rank_frequency(tokens: list[str]) -> np.ndarray:
    """Type counts sorted descending: f(1) ≥ f(2) ≥ … ≥ f(V)."""
    return np.array(sorted(Counter(tokens).values(), reverse=True), dtype=float)


def frequency_spectrum(counts: np.ndarray) -> dict[int, int]:
    """V_m = number of types seen exactly m times."""
    return dict(sorted(Counter(counts.astype(int).tolist()).items()))


def rank_of_frequency(counts: np.ndarray, f: float) -> int:
    """r(f) = #{types with count ≥ f}: the rank of the last type whose count is at least f."""
    return int(np.sum(counts >= f))


def ols_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of log y on log x — the classical, fragile Zipf/Heaps estimator."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def log_spaced_ranks(max_rank: int, k: int = 40, min_rank: int = 1) -> np.ndarray:
    """Integer ranks at k geometrically spaced points in [min_rank, max_rank] (duplicates dropped)."""
    return np.unique(np.geomspace(min_rank, max_rank, k).astype(int))


def zipf_slope_window(counts: np.ndarray, r_lo: int, r_hi: int, *, dense: bool = True) -> float:
    """OLS rank–frequency slope over ranks [r_lo, r_hi]; dense uses every rank, else 40 log-spaced."""
    r_hi = min(r_hi, len(counts))
    ranks = np.arange(r_lo, r_hi + 1) if dense else log_spaced_ranks(r_hi, min_rank=r_lo)
    return ols_loglog_slope(ranks, counts[ranks - 1])


# --------------------------------------------------------------------------------------------
# 2. Zipf as a probability law.
# --------------------------------------------------------------------------------------------


def zipf_pmf(s: float, R: int) -> np.ndarray:
    """p_r ∝ r^{-s} on {1, …, R} (normalized by the generalized harmonic number H_{R,s})."""
    w = np.arange(1, R + 1, dtype=float) ** -s
    return w / w.sum()


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def maxent_exponent(mu: float, R: int) -> float:
    """The s whose truncated Zipf has E[log r] = mu: the Lagrange multiplier of the max-entropy problem."""
    logr = np.log(np.arange(1, R + 1))
    return float(optimize.brentq(lambda s: zipf_pmf(s, R) @ logr - mu, -20.0, 20.0))


# --------------------------------------------------------------------------------------------
# 3. Zipf ⇒ Heaps (Karlin 1967).
# --------------------------------------------------------------------------------------------

_KARLIN_R = 2_000_000  # exact head; the tail r > R is summed by its Taylor series in n·p_r ≪ 1


@lru_cache(maxsize=8)
def _zipf_head(s: float) -> tuple[np.ndarray, float]:
    C = 1.0 / special.zeta(s)
    return C * np.arange(1, _KARLIN_R + 1, dtype=float) ** -s, C


def _tail_moments(s: float) -> tuple[float, float]:
    """Σ_{r>R} p_r and Σ_{r>R} p_r² for p_r = r^{-s}/ζ(s)."""
    C = 1.0 / special.zeta(s)
    return C * special.zeta(s, _KARLIN_R + 1), C**2 * special.zeta(2 * s, _KARLIN_R + 1)


def expected_vocab(s: float, n: int, *, poissonized: bool = False) -> float:
    """E V(n) = Σ_r [1 − (1 − p_r)^n] for p_r = r^{-s}/ζ(s), the infinite Zipf law; poissonized
    replaces (1 − p)^n by e^{−np}. Tail r > R to second order: np − n(n−1)p²/2 (n²p²/2 Poissonized);
    the dropped third-order term is below (n·p_R)³ per type, negligible for every n used here."""
    p, _ = _zipf_head(s)
    seen = -np.expm1(-n * p) if poissonized else -np.expm1(n * np.log1p(-p))
    m1, m2 = _tail_moments(s)
    return float(np.sum(seen) + n * m1 - (n * n if poissonized else n * (n - 1)) * m2 / 2)


def expected_hapax(s: float, n: int) -> float:
    """E V_1(n) = Σ_r n p_r (1 − p_r)^{n−1}: expected number of types seen exactly once."""
    p, _ = _zipf_head(s)
    m1, m2 = _tail_moments(s)
    return float(np.sum(n * p * np.exp((n - 1) * np.log1p(-p))) + n * m1 - n * (n - 1) * m2)


def expected_vocab_finite(p: np.ndarray, n: int) -> float:
    """E V(n) for a finite vocabulary with probabilities p (bounded by len(p): it must saturate)."""
    return float(np.sum(-np.expm1(n * np.log1p(-p))))


def karlin_integral(s: float, n: int) -> float:
    """I(n) = ∫_0^∞ (1 − e^{−nCx^{−s}}) dx = Γ(1 − 1/s)(Cn)^{1/s}, C = 1/ζ(s)."""
    return float(special.gamma(1 - 1 / s) * (n / special.zeta(s)) ** (1 / s))


# --------------------------------------------------------------------------------------------
# 4. Pitman–Yor.
# --------------------------------------------------------------------------------------------


def pitman_yor_crp(n: int, d: float, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Table index of each of n customers in a PY(d, θ) Chinese restaurant (tables = word types).

    New table with P = (θ + dK)/(θ + n); else table k with P ∝ c_k − d. O(1) per step: split
    c_k − d = (c_k − 1) + (1 − d), i.e. pick a uniformly random non-founding customer's table
    (total weight n − K) or a uniformly random table (total weight (1 − d)K).
    """
    if not (0 <= d < 1 and theta > -d):
        raise ValueError("need 0 <= d < 1 and theta > -d")
    u = rng.random((n, 2))
    labels = np.empty(n, dtype=np.int64)
    joiners: list[int] = []  # table of every non-founding customer
    k = 0
    for i in range(n):
        x = u[i, 0] * (theta + i)
        if x < theta + d * k:
            labels[i] = k
            k += 1
            continue
        x -= theta + d * k  # x ∈ [0, (i − k) + (1 − d)k)
        if x < i - k:
            t = joiners[int(u[i, 1] * len(joiners))]
        else:
            t = int(u[i, 1] * k)
        labels[i] = t
        joiners.append(t)
    return labels


def py_expected_types(n: np.ndarray | int, d: float, theta: float) -> np.ndarray:
    """Exact E[K_n] = (θ/d)·[Γ(θ+n+d)Γ(θ) / (Γ(θ+d)Γ(θ+n)) − 1]; the d → 0 limit is θ(ψ(θ+n) − ψ(θ)).
    θ > 0 keeps every Gamma positive so the log-Gamma form is exact (PY also allows −d < θ ≤ 0)."""
    if theta <= 0:
        raise ValueError("this closed form is written for theta > 0")
    n = np.asarray(n, dtype=float)
    if d == 0:
        return theta * (special.digamma(theta + n) - special.digamma(theta))
    log_ratio = (special.gammaln(theta + n + d) + special.gammaln(theta)
                 - special.gammaln(theta + d) - special.gammaln(theta + n))
    return theta / d * np.expm1(log_ratio)


def py_asymptote(n: np.ndarray | int, d: float, theta: float) -> np.ndarray:
    """E[K_n] ~ Γ(θ+1)/(d·Γ(θ+d)) · n^d."""
    return np.exp(special.gammaln(theta + 1) - special.gammaln(theta + d)) / d * np.asarray(n, float) ** d


def sibuya_pmf(m: np.ndarray | int, d: float) -> np.ndarray:
    """Limit share of PY types seen exactly m times: d·Γ(m − d) / (Γ(1 − d)·m!) ~ m^{−(1+d)}·d/Γ(1−d)."""
    m = np.asarray(m, dtype=float)
    return d * np.exp(special.gammaln(m - d) - special.gammaln(1 - d) - special.gammaln(m + 1))


def heaps_slope_shifted(growth: np.ndarray, d: float, theta: float, start: int = 1000, k: int = 25) -> float:
    """Theorem 3 as an estimator: slope of log(K_n + θ/d) on log(n + θ), which is d up to O(1/n)."""
    ns = log_spaced_ranks(len(growth), k, min_rank=start)
    return ols_loglog_slope(ns + theta, growth[ns - 1] + theta / d)


def types_seen(labels_or_tokens) -> np.ndarray:
    """V(n) for n = 1, …, N: distinct types among the first n draws."""
    seen: set = set()
    out = np.empty(len(labels_or_tokens), dtype=float)
    for i, t in enumerate(labels_or_tokens):
        seen.add(t)
        out[i] = len(seen)
    return out


def heaps_slope(growth: np.ndarray, start: int = 1000, k: int = 40) -> float:
    """OLS slope of log V(n) on log n at k log-spaced n ≥ start (Heaps' β)."""
    ns = log_spaced_ranks(len(growth), k, min_rank=start)
    return ols_loglog_slope(ns, growth[ns - 1])


# --------------------------------------------------------------------------------------------
# 5. Estimating the exponent.
# --------------------------------------------------------------------------------------------


def discrete_powerlaw_mle(counts: np.ndarray, xmin: int) -> float:
    """MLE of α for P(X = x) = x^{−α}/ζ(α, xmin), x ≥ xmin, fit to the type counts ≥ xmin.

    The log-likelihood ℓ(α) = −n·log ζ(α, xmin) − α·Σ log x_i is concave in α (log ζ is a log-sum-exp
    of linear functions of α), so a bounded scalar search finds the unique maximizer.
    """
    x = counts[counts >= xmin]
    n, sum_log = len(x), float(np.sum(np.log(x)))
    res = optimize.minimize_scalar(lambda a: n * np.log(special.zeta(a, xmin)) + a * sum_log,
                                   bounds=(1.0 + 1e-6, 8.0), method="bounded", options={"xatol": 1e-10})
    return float(res.x)


def continuous_powerlaw_mle(counts: np.ndarray, xmin: int) -> float:
    """Clauset–Shalizi–Newman eq. (3.7): α ≈ 1 + n / Σ log(x_i / (xmin − 1/2)), good for xmin ≳ 6."""
    x = counts[counts >= xmin]
    return float(1 + len(x) / np.sum(np.log(x / (xmin - 0.5))))


def mle_stderr(alpha: float, n_tail: int) -> float:
    """Asymptotic standard error (α − 1)/√n of the power-law MLE."""
    return (alpha - 1) / np.sqrt(n_tail)


def ks_distance(counts: np.ndarray, xmin: int, alpha: float) -> float:
    """Kolmogorov–Smirnov distance between the empirical tail CDF and the fitted discrete power law."""
    x = np.sort(counts[counts >= xmin]).astype(int)
    support = np.arange(xmin, x[-1] + 1)
    model_cdf = 1 - special.zeta(alpha, support + 1) / special.zeta(alpha, xmin)
    emp_cdf = np.searchsorted(x, support, side="right") / len(x)
    return float(np.max(np.abs(emp_cdf - model_cdf)))


def select_xmin(counts: np.ndarray, candidates=range(1, 31), min_tail: int = 50) -> tuple[int, float]:
    """Clauset et al.'s rule: the xmin whose MLE fit minimizes the KS distance (tails ≥ min_tail types)."""
    best = None
    for xmin in candidates:
        if np.sum(counts >= xmin) < min_tail:
            break
        a = discrete_powerlaw_mle(counts, xmin)
        dist = ks_distance(counts, xmin, a)
        if best is None or dist < best[2]:
            best = (xmin, a, dist)
    assert best is not None
    return best[0], best[1]


def zipf_exponent_from_alpha(alpha: float) -> float:
    """Corollary 1: spectrum tail exponent α ⇔ rank–frequency exponent s = 1/(α − 1)."""
    return 1.0 / (alpha - 1.0)


# --------------------------------------------------------------------------------------------
# 6. Finance case study: the shared corpus, with its Pitman–Yor content draws recorded.
# --------------------------------------------------------------------------------------------

POWERLAW_XMINS = (2, 3, 5, 8, 10, 15)


@lru_cache(maxsize=1)
def finance_streams() -> dict[str, tuple[str, ...]]:
    """Token streams of generate_corpus(300, 0): "full" text, all Pitman–Yor "content" draws in corpus
    order, and the draws of each restaurant ("common", plus one per sector).

    Rebuilds the corpus with fc._CorpusGen exactly as fc.generate_corpus does, wrapping each
    restaurant's draw to record it; test_finance_rebuild_matches_corpus pins the two together.
    """
    gen = fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)
    streams: dict[str, list[str]] = {"content": []}
    for name, py in (("common", gen.common), *gen.by_sector.items()):
        streams[name] = []

        def recording(rng, _draw=py.draw, _own=streams[name]):
            w = _draw(rng)
            _own.append(w)
            streams["content"].append(w)
            return w

        py.draw = recording
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    docs = [gen.doc(i, int(t)) for i, t in enumerate(tones)]
    streams["full"] = [t for d in docs for t in fc._tokens(d.text)]
    return {k: tuple(v) for k, v in streams.items()}


@lru_cache(maxsize=1)
def finance_summary() -> dict[str, float]:
    st = finance_streams()
    full, content, common = list(st["full"]), list(st["content"]), list(st["common"])
    f_full, f_content = rank_frequency(full), rank_frequency(content)
    xmin_full, alpha_full = select_xmin(f_full)
    alphas = [discrete_powerlaw_mle(f_content, x) for x in POWERLAW_XMINS]
    d, theta = fc.HEAPS_DISCOUNT, fc.PY_STRENGTH
    return {
        # Full text: what a reader of "real" text would measure.
        "N": len(full), "V": len(f_full), "hapax_share": float(np.mean(f_full == 1)),
        "zipf_slope_logspaced": fc.zipf_slope(full), "heaps_full": fc.heaps_exponent(full),
        "xmin_full": xmin_full, "alpha_full": alpha_full, "s_full": zipf_exponent_from_alpha(alpha_full),
        # All Pitman–Yor draws, with the generic (asymptotic) estimators.
        "N_content": len(content), "V_content": len(f_content),
        "hapax_share_content": float(np.mean(f_content == 1)),
        "heaps_content_naive": heaps_slope(types_seen(content)),
        "alpha_content_min": min(alphas), "alpha_content_max": max(alphas),
        # The largest restaurant, with Theorem 3's exact offset as the estimator.
        "N_common": len(common), "V_common": len(set(common)),
        "heaps_common_naive": heaps_slope(types_seen(common)),
        "heaps_common_shifted": heaps_slope_shifted(types_seen(common), d, theta),
        "EK_common": float(py_expected_types(len(common), d, theta)),
    }


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX).
# --------------------------------------------------------------------------------------------

LAB_DISCOUNTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LAB_N = 20_000
LAB_WINDOWS = ((1, 10), (10, 100), (100, 1000), (1000, 3425), (1, 1000), (1, 3425))


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    """Panel A: the corpus rank–frequency curve, OLS slopes per fitting window, and the MLE fit.
    Panel B: one PY(d, θ = PY_STRENGTH) restaurant per lab discount — K_n against exact E[K_n],
    the naive and offset-corrected Heaps slopes, and its rank–frequency curve."""
    f_full = rank_frequency(list(finance_streams()["full"]))
    ranks = log_spaced_ranks(len(f_full), 60)
    s = finance_summary()
    theta = fc.PY_STRENGTH
    rng = np.random.default_rng([SEED, 1])
    growth = {}
    for d in LAB_DISCOUNTS:
        labels = pitman_yor_crp(LAB_N, d, theta, rng)
        g = types_seen(labels)
        ns = log_spaced_ranks(LAB_N, 30, min_rank=10)
        counts = np.sort(np.bincount(labels))[::-1].astype(float)
        rr = log_spaced_ranks(len(counts), 30)
        growth[d] = {
            "n": ns.tolist(), "K": g[ns - 1].astype(int).tolist(),
            "EK": np.round(py_expected_types(ns, d, theta), 1).tolist(),
            "heaps_naive": round(heaps_slope(g), 3), "heaps_shifted": round(heaps_slope_shifted(g, d, theta), 3),
            "rank": rr.tolist(), "freq": counts[rr - 1].astype(int).tolist(),
        }
    return {
        "corpus": {"rank": ranks.tolist(), "freq": f_full[ranks - 1].astype(int).tolist(),
                   "N": s["N"], "V": s["V"]},
        # Dense OLS per window; the intercept (natural log) only places the drawn line.
        "windows": {f"{lo}-{hi}": dict(zip(("slope", "intercept"), np.round(np.polyfit(
            np.log(np.arange(lo, hi + 1)), np.log(f_full[lo - 1 : hi]), 1), 3).tolist())) for lo, hi in LAB_WINDOWS},
        # The MLE as a rank-plot line: f(r) = xmin·(r / r_xmin)^(−s), through (r(xmin), xmin) by Prop 1.
        "mle": {"xmin": s["xmin_full"], "alpha": round(s["alpha_full"], 3), "s": round(s["s_full"], 3),
                "r_xmin": rank_of_frequency(f_full, s["xmin_full"])},
        "theta": theta, "growth": growth,
    }


# --------------------------------------------------------------------------------------------
# Verification harness — each pedagogical claim of the topic.
# --------------------------------------------------------------------------------------------


def _full_counts() -> np.ndarray:
    return rank_frequency(list(finance_streams()["full"]))


def test_rank_is_count_tail():
    """Prop 1 (exact): the rank of a type with count f, ties ranked last, is #{types with count ≥ f}."""
    counts = _full_counts()
    for f in np.unique(counts):
        last_rank_with_f = int(np.max(np.nonzero(counts == f)[0])) + 1
        assert rank_of_frequency(counts, f) == last_rank_with_f, f
    spectrum = frequency_spectrum(counts)
    assert sum(spectrum.values()) == len(counts)
    assert sum(m * v for m, v in spectrum.items()) == int(counts.sum())


def test_spectrum_exponent_maps_to_rank_exponent():
    """Prop 1 corollary on random instances: counts with spectrum tail α give rank slope −1/(α−1)."""
    rng = np.random.default_rng([SEED, 2])
    for alpha in (1.6, 2.0, 2.5):
        counts = np.sort(stats.zipf.rvs(alpha, size=200_000, random_state=rng))[::-1].astype(float)
        # The rank–frequency curve is a step function at small counts; fit where counts are ≥ 10.
        r_hi = rank_of_frequency(counts, 10)
        slope = zipf_slope_window(counts, 10, r_hi, dense=False)
        assert abs(slope + zipf_exponent_from_alpha(alpha)) < 0.12, (alpha, slope)


def test_zeta_converges_iff_s_above_one():
    """Σ r^{-s} converges for s > 1 (to ζ(s)) and diverges like log R at s = 1."""
    for s in (1.1, 1.5, 2.0):
        partial = np.sum(np.arange(1, 10**6 + 1, dtype=float) ** -s)
        tail = special.zeta(s, 10**6 + 1)
        assert abs(partial + tail - special.zeta(s)) < 1e-9, s
    h = np.cumsum(1.0 / np.arange(1, 10**6 + 1))
    for R in (10**2, 10**4, 10**6):
        assert abs(h[R - 1] - np.log(R) - np.euler_gamma) < 1.0 / R, R


def test_zipf_maximizes_entropy_at_fixed_log_rank():
    """Mandelbrot: among p on {1..R} with E[log r] = μ, entropy is maximized by p_r ∝ r^{-s}.
    Cross-checked against a generic constrained optimizer (SLSQP) started from uniform."""
    R, s_true = 40, 1.3
    logr = np.log(np.arange(1, R + 1))
    p_star = zipf_pmf(s_true, R)
    mu = float(p_star @ logr)
    assert abs(maxent_exponent(mu, R) - s_true) < 1e-9
    cons = ({"type": "eq", "fun": lambda p: p.sum() - 1}, {"type": "eq", "fun": lambda p: p @ logr - mu})
    res = optimize.minimize(lambda p: -entropy(np.clip(p, 1e-15, None)), np.full(R, 1 / R),
                            constraints=cons, bounds=[(0, 1)] * R, method="SLSQP",
                            options={"ftol": 1e-14, "maxiter": 1000})
    assert res.success, res.message
    assert np.max(np.abs(res.x - p_star)) < 1e-4
    # Any feasible perturbation lowers the entropy (strict concavity).
    rng = np.random.default_rng([SEED, 3])
    A = np.vstack([np.ones(R), logr])
    for _ in range(50):
        v = rng.standard_normal(R)
        v -= A.T @ np.linalg.lstsq(A.T, v, rcond=None)[0]  # project onto {Σv = 0, Σv log r = 0}
        q = p_star + 1e-3 * v / np.max(np.abs(v)) * p_star.min()
        assert entropy(q) < entropy(p_star)


KARLIN_S = (1.25, 1.5, 2.0, 3.0)
KARLIN_N = (10**2, 10**3, 10**4, 10**5, 10**6)


def test_karlin_poissonized_sandwich():
    """Proof step (exact): g(x) = 1 − e^{−nCx^{−s}} decreases, so I − 1 ≤ Σ_{r≥1} g(r) ≤ I."""
    for s in KARLIN_S:
        for n in KARLIN_N:
            I, S = karlin_integral(s, n), expected_vocab(s, n, poissonized=True)
            assert I - 1 - 1e-6 <= S <= I + 1e-6, (s, n, I, S)


def test_karlin_depoissonization_gap_is_lower_order():
    """Proof step: (1 − p)^n ≤ e^{−np}, so E V(n) ≥ Σ g(r); the gap is O(1)·n^{1/s}/n → negligible."""
    for s in KARLIN_S:
        gaps = []
        for n in KARLIN_N:
            gap = expected_vocab(s, n) - expected_vocab(s, n, poissonized=True)
            assert gap >= -1e-9, (s, n, gap)
            gaps.append(gap / n ** (1 / s))
        assert gaps[-1] < gaps[0] and gaps[-1] < 1e-4, (s, gaps)


def test_karlin_zipf_implies_heaps():
    """Thm 1: E V(n) / Γ(1−1/s)(Cn)^{1/s} → 1 with relative error O(n^{−1/s}), so the Heaps slope → 1/s."""
    for s in KARLIN_S:
        ratios = [expected_vocab(s, n) / karlin_integral(s, n) for n in KARLIN_N]
        assert abs(ratios[-1] - 1) < abs(ratios[0] - 1) and abs(ratios[-1] - 1) < 0.005, (s, ratios)
        n_pair = np.array(KARLIN_N[-2:])
        beta = ols_loglog_slope(n_pair, np.array([expected_vocab(s, n) for n in n_pair]))
        assert abs(beta - 1 / s) < 0.005, (s, beta)
    # Convergence is slowest for LARGE s: the error term is at most 1 type against I ~ n^{1/s} types.
    gap = {s: abs(expected_vocab(s, 10**4) / karlin_integral(s, 10**4) - 1) for s in KARLIN_S}
    assert gap[3.0] > gap[2.0] > gap[1.5], gap


def test_finite_vocabulary_saturates():
    """Heaps needs an open vocabulary: a Zipf law truncated at R types has E V(n) ≤ R, so growth stalls."""
    s, R = 1.25, 10_000
    p = zipf_pmf(s, R)
    v = {n: expected_vocab_finite(p, n) for n in (10**3, 10**4, 10**6, 10**7)}
    assert all(x <= R for x in v.values())
    early = np.log(v[10**4] / v[10**3]) / np.log(10)
    late = np.log(v[10**7] / v[10**6]) / np.log(10)
    assert early > 0.6 / s and late < 0.25 / s, (early, late)


def test_karlin_hapax_share():
    """Cor: E V_1(n) / E V(n) → 1/s — the share of types seen once estimates the Heaps exponent."""
    for s in (1.5, 2.0, 3.0):
        share = expected_hapax(s, 10**6) / expected_vocab(s, 10**6)
        assert abs(share - 1 / s) < 0.02, (s, share)


def test_py_exact_formula_solves_recursion():
    """Thm 2 proof: E[K_{n+1}] = E[K_n] + (θ + d·E[K_n])/(θ + n), E[K_0] = 0."""
    for d, theta in ((0.3, 1.0), (0.6, 20.0), (0.8, 0.5), (0.5, 0.05)):
        ek = py_expected_types(np.arange(0, 1001), d, theta)
        assert abs(ek[0]) < 1e-12
        step = ek[:-1] + (theta + d * ek[:-1]) / (theta + np.arange(0, 1000))
        assert np.max(np.abs(ek[1:] - step) / ek[1:]) < 1e-10, (d, theta)


def test_py_monte_carlo_matches_exact_mean():
    """E[K_n] from 200 restaurants (one stream) lies within 3.5 standard errors of the formula."""
    rng = np.random.default_rng([SEED, 4])
    for d, theta in ((0.3, 5.0), (0.6, 20.0), (0.8, 1.0)):
        ks = [len(np.unique(pitman_yor_crp(2000, d, theta, rng))) for _ in range(200)]
        mean, se = np.mean(ks), np.std(ks, ddof=1) / np.sqrt(len(ks))
        assert abs(mean - py_expected_types(2000, d, theta)) < 3.5 * se, (d, theta, mean)


def test_py_matches_corpus_sampler():
    """Our O(1) sampler and finance_corpus's cumulative-sum sampler draw from the same law."""
    rng = np.random.default_rng([SEED, 5])
    ours = [len(np.unique(pitman_yor_crp(3000, 0.6, 20.0, rng))) for _ in range(60)]
    theirs = []
    for _ in range(60):
        py = fc._PitmanYor(0.6, 20.0, [], fc._PseudoWords(()))
        theirs.append(len({py.draw(rng) for _ in range(3000)}))
    pooled_se = np.sqrt(np.var(ours, ddof=1) / 60 + np.var(theirs, ddof=1) / 60)
    assert abs(np.mean(ours) - np.mean(theirs)) < 3.5 * pooled_se


def test_py_growth_is_n_to_the_d():
    """E[K_n] + θ/d = c·n^d·(1 + O(1/n)) — so the offset, not the power, is the slow part: E[K_n]/(c n^d)
    is off by (θ/d)/(c n^d). d → 0 recovers the Dirichlet process's θ(ψ(θ+n) − ψ(θ)) ~ θ log n."""
    for d, theta in ((0.3, 5.0), (0.6, 20.0), (0.8, 1.0)):
        for n in (10**3, 10**6, 10**9):
            ek, asym = float(py_expected_types(n, d, theta)), float(py_asymptote(n, d, theta))
            # 2dθ/n is the O(1/n) term; 1e-5 covers gammaln's float error at arguments near 1e9.
            assert abs((ek + theta / d) / asym - 1) < 2 * d * theta / n + 1e-5, (d, theta, n)
    for n in (10, 1000, 10**5):
        dp = float(py_expected_types(n, 0.0, 3.0))
        # d = 1e-5: small enough for the O(d) term, large enough that θ/d·expm1(·) keeps its digits.
        assert abs(float(py_expected_types(n, 1e-5, 3.0)) - dp) < 1e-3 * dp, n


def test_py_spectrum_is_sibuya():
    """Cited (Pitman 2006, Lemma 3.11): K_{n,m}/K_n → d·Γ(m−d)/(Γ(1−d)m!), whose tail ~ m^{−(1+d)}:
    the count spectrum has α = 1 + d, so by Prop 1 the rank–frequency exponent is s = 1/d.
    Asserted on 200k-customer restaurants: shares for m = 1..5 tightly, the tail exponent loosely."""
    rng = np.random.default_rng([SEED, 6])
    for d in (0.4, 0.6, 0.8):
        counts = np.bincount(pitman_yor_crp(200_000, d, 10.0, rng)).astype(float)
        m = np.arange(1, 6)
        shares, limit = np.array([np.mean(counts == k) for k in m]), sibuya_pmf(m, d)
        se = np.sqrt(limit * (1 - limit) / len(counts))  # K ≈ 1.3k types at d = 0.4, 30k at d = 0.8
        assert np.all(np.abs(shares - limit) < 4 * se), (d, shares, limit)
        _, alpha = select_xmin(counts, range(1, 41))
        assert abs(alpha - (1 + d)) < 0.1, (d, alpha)


def test_mle_is_consistent_and_matches_scipy():
    """The discrete MLE recovers α on random instances and agrees with scipy.stats.fit at xmin = 1."""
    rng = np.random.default_rng([SEED, 7])
    for alpha in (1.6, 2.2):
        x = stats.zipf.rvs(alpha, size=20_000, random_state=rng).astype(float)
        a_hat = discrete_powerlaw_mle(x, 1)
        assert abs(a_hat - alpha) < 4 * mle_stderr(alpha, len(x)), (alpha, a_hat)
        ref = stats.fit(stats.zipf, x.astype(int), bounds={"a": (1.01, 8.0)},
                        optimizer=lambda fun, bounds, **kw: optimize.minimize(
                            fun, x0=np.mean(np.asarray(bounds, float), axis=1), bounds=bounds,
                            method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12}))
        assert abs(ref.params.a - a_hat) < 1e-4, (ref.params.a, a_hat)


def test_mle_score_equation():
    """At α̂ the score vanishes: mean log x = −∂_α log ζ(α, xmin)."""
    counts = _full_counts()
    for xmin in (1, 5, 10):
        a = discrete_powerlaw_mle(counts, xmin)
        h = 1e-6
        dlogz = (np.log(special.zeta(a + h, xmin)) - np.log(special.zeta(a - h, xmin))) / (2 * h)
        assert abs(np.mean(np.log(counts[counts >= xmin])) + dlogz) < 1e-5, xmin


def test_continuous_approximation_close_for_large_xmin():
    counts = _full_counts()
    for xmin in (6, 10, 20):
        assert abs(continuous_powerlaw_mle(counts, xmin) - discrete_powerlaw_mle(counts, xmin)) < 0.03


def test_ols_is_biased_where_mle_is_not():
    """On 40 random power-law samples (α = 1.6, so s = 5/3): OLS on the log–log frequency HISTOGRAM is
    badly biased (Clauset et al.'s warning); OLS on the rank plot is closer but still at least twice as
    far off as the MLE, and neither OLS comes with a valid standard error."""
    rng = np.random.default_rng([SEED, 8])
    alpha, s_true = 1.6, zipf_exponent_from_alpha(1.6)
    rank_err, hist_err, mle_err = [], [], []
    for _ in range(40):
        counts = np.sort(stats.zipf.rvs(alpha, size=5_000, random_state=rng))[::-1].astype(float)
        rank_err.append(abs(-zipf_slope_window(counts, 1, len(counts)) - s_true))
        spec = frequency_spectrum(counts)
        alpha_hist = -ols_loglog_slope(np.array(list(spec)), np.array(list(spec.values()), float))
        hist_err.append(abs(zipf_exponent_from_alpha(alpha_hist) - s_true))
        mle_err.append(abs(zipf_exponent_from_alpha(discrete_powerlaw_mle(counts, 1)) - s_true))
    assert np.mean(rank_err) > 2 * np.mean(mle_err), (np.mean(rank_err), np.mean(mle_err))
    assert np.mean(hist_err) > 20 * np.mean(mle_err), (np.mean(hist_err), np.mean(mle_err))


def test_ols_slope_depends_on_window():
    """On the finance corpus, dense OLS over different decades of rank disagrees by more than 1."""
    counts = _full_counts()
    slopes = [zipf_slope_window(counts, lo, hi) for lo, hi in ((1, 10), (10, 100), (100, 1000))]
    assert max(slopes) - min(slopes) > 1.0, slopes


def test_finance_rebuild_matches_corpus():
    """The recorded rebuild is byte-identical to fc.generate_corpus(300, 0) (fingerprint-pinned), and
    the restaurant streams partition the content stream."""
    st = finance_streams()
    assert list(st["full"]) == [t for d in fc.generate_corpus(300, 0) for t in fc._tokens(d.text)]
    assert sum(len(st[k]) for k in ("common", *fc.SECTORS)) == len(st["content"]) > 0
    assert set(st["content"]) <= set(st["full"])


def test_ols_matches_corpus_helper():
    """Our log-spaced window slope reproduces finance_corpus.zipf_slope exactly (same estimator)."""
    full = list(finance_streams()["full"])
    assert abs(zipf_slope_window(_full_counts(), 1, 1000, dense=False) - fc.zipf_slope(full)) < 1e-12


def test_full_text_shows_the_real_text_tension():
    """Full text looks like real text — OLS Zipf s ≈ 0.9, Heaps β ≈ 0.5 — so β = 1/s fails by ~2×, although
    one parameter d generated it: a closed-class template head on an open-class Pitman–Yor tail."""
    s = finance_summary()
    s_ols = -s["zipf_slope_logspaced"]
    assert 0.85 <= s_ols <= 1.0 and 0.45 <= s["heaps_full"] <= 0.55, (s_ols, s["heaps_full"])
    assert (1 / s_ols) / s["heaps_full"] > 1.8


def test_templates_saturate_full_text_growth():
    """Closed-class template words stop arriving, so full-text Heaps β sits below the content β."""
    s = finance_summary()
    assert s["heaps_full"] < s["heaps_content_naive"] - 0.1, (s["heaps_full"], s["heaps_content_naive"])
    st = finance_streams()
    template_types = set(st["full"]) - set(st["content"])
    first_half = set(st["full"][: len(st["full"]) // 2]) - set(st["content"])
    assert len(first_half) / len(template_types) > 0.9  # ≥ 90% of template types arrive in the first half


def test_generic_estimators_miss_the_planted_truth():
    """Even on the pure content stream, the asymptotic estimators miss d = 0.6 at corpus scale: the naive
    Heaps slope overshoots (Theorem 3's −θ/d offset), and the power-law MLE's α̂ drifts with xmin far from
    1 + d (the Sibuya spectrum is a power law only for counts ≫ 1, which a 12k-draw stream barely has)."""
    s, d = finance_summary(), fc.HEAPS_DISCOUNT
    assert s["heaps_content_naive"] > d + 0.05, s["heaps_content_naive"]
    assert s["alpha_content_max"] - s["alpha_content_min"] > 0.2
    assert s["alpha_content_min"] > 1 + d + 0.25, s["alpha_content_min"]


def test_theorem_three_recovers_the_planted_discount():
    """On the largest restaurant, the offset-corrected fit log(K + θ/d) ~ d·log(n + θ) recovers d, and K
    sits near the exact E[K_n]. (θ = PY_STRENGTH is known here; d enters only through the θ/d shift.)"""
    s, d = finance_summary(), fc.HEAPS_DISCOUNT
    assert abs(s["heaps_common_shifted"] - d) < 0.02, s["heaps_common_shifted"]
    assert s["heaps_common_naive"] - d > s["heaps_common_shifted"] - d + 0.01
    assert abs(s["V_common"] - s["EK_common"]) < 0.15 * s["EK_common"], (s["V_common"], s["EK_common"])


def test_viz_payload():
    """The lab's slider claim: the offset-corrected Heaps slope tracks d; the naive one overshoots."""
    p = viz_payload()
    assert len(p["corpus"]["rank"]) == len(p["corpus"]["freq"]) >= 40
    assert all(np.diff(p["corpus"]["freq"]) <= 0)
    assert set(p["growth"]) == set(LAB_DISCOUNTS)
    for lo, hi in LAB_WINDOWS:  # the window readout is exactly zipf_slope_window's dense OLS
        assert p["windows"][f"{lo}-{hi}"]["slope"] == round(zipf_slope_window(_full_counts(), lo, hi), 3)
    for d, g in p["growth"].items():
        assert abs(g["heaps_shifted"] - d) < 0.03, (d, g["heaps_shifted"])
        assert g["heaps_naive"] > g["heaps_shifted"], (d, g)


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "vocabulary-laws-lab.json"


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
    if not failed:
        print("\nFinance summary (generate_corpus(300, 0)):")
        for k, v in finance_summary().items():
            print(f"  {k:24} {v:.4f}" if isinstance(v, float) else f"  {k:24} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
