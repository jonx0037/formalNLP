"""Collocations and keyword extraction — the reference implementation for formalNLP's twenty-sixth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. The 2×2 table. A bigram (w₁, w₂) splits the N bigram positions of a corpus into a 2×2 table (Def 1), a point of
   the tetrahedron Δ³; the independent tables form the saddle p₁₁p₂₂ = p₁₂p₂₁ (Def 2). PMI, the t-score, Pearson's
   X², and Dunning's G² (Def 3) measure the table's distance from that surface in different ways: the product of the
   margins is the KL projection onto it, so G² = 2N·KL(p̂ ‖ p̂_r ⊗ p̂_c) = 2N·Î (Thm 1), and X² = N·χ²(p̂ ‖ p̂_r ⊗ p̂_c)
   = Nφ̂² (Prop 1).
2. Rare pairs. Two words seen once each, and only together: PMI = log N (its maximum), X² = N exactly, G² ≈ 2 log N + 2,
   t ≈ 1, and the exact p-value is 1/N (Prop 2). Among perfect pairs, PMI falls with the count while the evidence rises
   (Prop 3).
3. Tests and their sizes. Given its margins, the joint count is hypergeometric under independence (Lemma 1), which
   makes Fisher's test exact (Def 4). X² and G² are asymptotically χ²₁ (Thm 2), but their exact sizes at small expected
   counts are far from nominal (Prop 4). Testing only the pairs one has seen is a selection: when a pair's chance of
   being seen at all is below α, every sighting is "significant" (Prop 5).
4. Keyness. Comparing two corpora is the same 2×2 table with the corpus as the second variable (Def 5, Prop 6).
   Rayson and Garside's LL sums one row of it, so it sits below G² by a term KL ≤ χ² bounds (Rem 3).
5. TextRank. The co-occurrence graph (Def 6) and PageRank with damping d (Def 7): a positive stochastic matrix has one
   stationary distribution, and it is positive (Thm 3); power iteration contracts in ℓ1 by d per step (Thm 4). On an
   undirected graph, d → 1 gives each word's share of the edge weight, which on the weighted window graph is its
   frequency up to boundary terms; teleporting in proportion to strength instead makes π exactly the strength share at
   every d (Prop 7).
6. Finance case study. The shared corpus's generator is the answer key. Words drawn from the Pitman–Yor restaurants
   are independent given the document's sector, so two drawn words side by side (a compound or a list neighbor) are a
   null pair; template text is a fixed phrase. G², the t-score, and Fisher's test rank fixed phrases first, while PMI's
   top is null pairs seen once. Keyness against the other sectors finds planted sector nouns; TextRank, which never sees
   a reference corpus, tracks frequency.

Cross-checks: scipy's ``fisher_exact``, ``chi2_contingency`` (Pearson and log-likelihood), and ``hypergeom``; a direct
convex solve for the KL projection (Thm 1); exact enumeration of every table with given margins (Prop 4, Prop 5); and a
dense eigenvector for PageRank (Thm 3).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/collocations-keyword-extraction/collocations_keyword_extraction.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import xlogy

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "distributional-semantics-ppmi-svd"))
import distributional_semantics_ppmi_svd as ds  # noqa: E402 — module objects: their test_* stay out of this harness

fc, ng, zhv = ds.fc, ds.ng, ds.zhv

# ONE stream per random-instance claim, each from default_rng([SEED, k]); k ≥ 3900 keeps this module's streams apart
# from Topics 1–25.
SEED = ds.SEED
VIZ_JSON = _HERE.parents[1] / "src" / "components" / "viz" / "data" / "collocation-lab.json"
ALPHA = 0.05
CHI2_CRIT = float(stats.chi2.ppf(1 - ALPHA, 1))  # 3.841459…


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


def _r(x, nd: int = 3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


# --------------------------------------------------------------------------------------------
# 1. The 2×2 table (Def 1, Def 2, Def 3, Thm 1, Prop 1).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """Def 1. Over N bigram positions: a = c(w₁ w₂), r = c(w₁ ·), c = c(· w₂). The other cells follow:
    b = r − a, c' = c − a, d = N − r − c + a."""

    a: float | np.ndarray
    r: float | np.ndarray
    c: float | np.ndarray
    n: float | np.ndarray

    def cells(self) -> np.ndarray:
        """[[a, b], [c', d]] as a 2×2 array (vectorizes over array-valued fields: shape (2, 2, …))."""
        a, r, c, n = (np.asarray(x, dtype=float) for x in (self.a, self.r, self.c, self.n))
        return np.array([[a, r - a], [c - a, n - r - c + a]])

    def expected(self) -> np.ndarray:
        """The cells of the product of the margins: E_ij = (row i total)(column j total)/N."""
        a, r, c, n = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (self.a, self.r, self.c, self.n)))
        return np.array([[r * c, r * (n - c)], [(n - r) * c, (n - r) * (n - c)]]) / n


def scores(t: Table) -> dict[str, np.ndarray]:
    """Def 3, vectorized over array-valued tables: pmi = log(aN/(rc)), t = (a − E)/√a with E = rc/N,
    x2 = N(ad − bc')²/(r(N−r)c(N−c)), g2 = 2 Σ O log(O/E) over the four cells (0 log 0 = 0)."""
    o, e = t.cells(), t.expected()
    a, n = o[0, 0], np.asarray(t.n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(a) - np.log(e[0, 0])
        tsc = (a - e[0, 0]) / np.sqrt(a)
    x2 = n * phi(t) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = 2 * xlogy(o, np.where(o > 0, o / e, 1.0)).sum(axis=(0, 1))  # log of the ratio: no N log N cancellation
    return {"pmi": pmi, "t": tsc, "x2": x2, "g2": np.maximum(g2, 0.0)}


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) in nats over matching entries, 0 log 0 = 0."""
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    return float((xlogy(p, p) - xlogy(p, q)).sum())


def chi2_div(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson's χ² divergence Σ (p − q)²/q."""
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    return float(((p - q) ** 2 / q).sum())


def independence_projection(p: np.ndarray) -> np.ndarray:
    """Thm 1: argmin over independent 2×2 tables q of KL(p ‖ q) is the outer product of p's margins."""
    p = np.asarray(p, dtype=float)
    return np.outer(p.sum(1), p.sum(0))


def projection_by_solver(p: np.ndarray) -> np.ndarray:
    """Thm 1's cross-check: minimize KL(p ‖ u vᵀ) over (u, v) in the open simplices numerically."""
    p = np.asarray(p, dtype=float)

    def q_of(z: np.ndarray) -> np.ndarray:
        u, v = 1 / (1 + np.exp(-z[0])), 1 / (1 + np.exp(-z[1]))
        return np.outer([u, 1 - u], [v, 1 - v])

    res = optimize.minimize(lambda z: kl(p, q_of(z)), np.zeros(2), method="BFGS", options={"gtol": 1e-12})
    return q_of(res.x)


def phi(t: Table) -> np.ndarray:
    """The φ coefficient (ad − bc')/√(r(N−r)c(N−c)); Prop 1: X² = Nφ²."""
    o = t.cells()
    r, c, n = (np.asarray(x, dtype=float) for x in (t.r, t.c, t.n))
    return (o[0, 0] * o[1, 1] - o[0, 1] * o[1, 0]) / np.sqrt(r * (n - r) * c * (n - c))


def mutual_information(p: np.ndarray) -> float:
    """Î(X; Y) of a 2×2 joint distribution, in nats (formalML shannon-entropy, Def 5)."""
    return kl(p, independence_projection(p))


def saddle_slice(r: int, c: int, n: int) -> dict:
    """Lab tab *Table geometry*: for fixed margins, a = max(0, r + c − N) … min(r, c) traces a segment of Δ³ that
    crosses the independence surface at a = rc/N; every score along it, its exact one-sided p-value, and the
    segment's points in barycentric 3-D coordinates (vertices p₁₁, p₁₂, p₂₁, p₂₂ of a regular tetrahedron)."""
    a = np.arange(max(0, r + c - n), min(r, c) + 1)
    t = Table(a, r, c, n)
    s = scores(t)
    o = t.cells() / n
    verts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    xyz = np.einsum("ka,kd->ad", o.reshape(4, -1), verts)
    return {"a": a, "expected": r * c / n, "p_exact": stats.hypergeom.sf(a - 1, n, r, c), "xyz": xyz, **s}


def independence_surface(m: int) -> np.ndarray:
    """The saddle q = u vᵀ, (u, v) on an m × m grid, in the tetrahedron coordinates of saddle_slice."""
    verts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    g = np.linspace(0, 1, m)
    u, v = np.meshgrid(g, g, indexing="ij")
    q = np.stack([u * v, u * (1 - v), (1 - u) * v, (1 - u) * (1 - v)])
    return np.einsum("kij,kd->ijd", q, verts)


# --------------------------------------------------------------------------------------------
# 2. Rare pairs (Prop 2, Prop 3).
# --------------------------------------------------------------------------------------------


def hapax_forms(n: float) -> dict[str, float]:
    """Prop 2's closed forms for the table [[1, 0], [0, N − 1]]: pmi = log N, x2 = N, g2 = 2[log N + (N−1) log(N/(N−1))],
    t = 1 − 1/N, p_exact = 1/N."""
    return {"pmi": math.log(n), "x2": float(n), "g2": 2 * (math.log(n) - (n - 1) * math.log1p(-1 / n)),
            "t": 1 - 1 / n, "p_exact": 1 / n}


def binary_entropy(x: np.ndarray) -> np.ndarray:
    """H(x) = −x log x − (1 − x) log(1 − x), in nats."""
    x = np.asarray(x, dtype=float)
    return -xlogy(x, x) - xlogy(1 - x, 1 - x)


def perfect_pair(k: np.ndarray, n: float) -> dict[str, np.ndarray]:
    """Prop 3: the table [[k, 0], [0, N − k]] for k = 1, 2, …: pmi = log(N/k) falls, x2 = N for every k, g2 = 2N·H(k/N)
    and t = √k(1 − k/N) rise (k < N/3), and −log p_exact = log C(N, k) rises (k < N/2)."""
    k = np.asarray(k, dtype=float)
    s = scores(Table(k, k, k, n))
    from scipy.special import gammaln

    logC = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return {"k": k, **s, "g2_entropy": 2 * n * binary_entropy(k / n), "neglog_p": logC}


# --------------------------------------------------------------------------------------------
# 3. Tests and their sizes (Lemma 1, Def 4, Thm 2, Prop 4, Prop 5).
# --------------------------------------------------------------------------------------------


def hypergeom_pmf(r: int, c: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lemma 1: the support a = max(0, r + c − n) … min(r, c) and P(a | margins) = C(r, a) C(n − r, c − a)/C(n, c)."""
    from scipy.special import gammaln

    a = np.arange(max(0, r + c - n), min(r, c) + 1)

    def lc(x, y):
        return gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1)

    return a, np.exp(lc(r, a) + lc(n - r, c - a) - lc(n, c))


_FISHER_RTOL = 1 + 1e-7  # scipy's tolerance for "no more probable than the observed table"


def log_p_greater(a, r, c, n) -> np.ndarray:
    """log P(A ≥ a | margins) by log-sum-exp over the hypergeometric tail (exact where sf() underflows to 0),
    elementwise over arrays of tables."""
    from scipy.special import gammaln, logsumexp

    def lc(x, y):
        return gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1)

    a, r, c, n = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (a, r, c, n)))
    out = np.empty(a.shape)
    for i in np.ndindex(a.shape):
        lo = max(a[i], r[i] + c[i] - n[i], 0.0)
        x = np.arange(lo, min(r[i], c[i]) + 1)
        out[i] = min(0.0, float(logsumexp(lc(r[i], x) + lc(n[i] - r[i], c[i] - x) - lc(n[i], c[i]))))
    return out


def fisher_p(a: int, r: int, c: int, n: int, alternative: str) -> float:
    """Def 4. 'greater': P(A ≥ a); 'two-sided': Σ over tables no more probable than the observed one (scipy's rule)."""
    sup, pmf = hypergeom_pmf(r, c, n)
    if alternative == "greater":
        return float(min(1.0, pmf[sup >= a].sum()))
    obs = pmf[sup == a][0]
    return float(min(1.0, pmf[pmf <= obs * _FISHER_RTOL].sum()))


def exact_size(r: int, c: int, n: int, stat: str, alpha: float) -> float:
    """Prop 4: P(reject | margins) under independence for 'x2' and 'g2' (reject when the statistic exceeds the χ²₁
    quantile) and 'fisher' (two-sided p ≤ α), by summing the hypergeometric pmf over the rejection region."""
    sup, pmf = hypergeom_pmf(r, c, n)
    if stat == "fisher":
        rej = np.array([pmf[pmf <= q * _FISHER_RTOL].sum() <= alpha for q in pmf])
    else:
        rej = scores(Table(sup, r, c, n))[stat] > stats.chi2.ppf(1 - alpha, 1)
    return float(pmf[rej].sum())


SIZE_N = 10_000
SIZE_E = np.round(np.geomspace(0.01, 20, 25), 4)


def size_curve(n: int, expected: np.ndarray) -> dict[str, list[float]]:
    """Lab tab *Test size*: exact sizes against E₁₁ = r²/N with r = c = round(√(E N)) (fixed N); the grid's E values
    are those of the rounded margins."""
    rs = sorted({max(1, int(round(math.sqrt(e * n)))) for e in expected})
    out: dict[str, list[float]] = {"r": rs, "expected": [r * r / n for r in rs]}
    for stat in ("x2", "g2", "fisher"):
        out[stat] = [exact_size(r, r, n, stat, ALPHA) for r in rs]
    return out


def selection(r: int, c: int, n: int, alpha: float) -> dict[str, float]:
    """Prop 5: P(seen) = P(A ≥ 1), P(reject) and P(reject | seen) for the one-sided exact test. When P(A ≥ 1) ≤ α,
    P(reject | seen) = 1 although P(reject) ≤ α."""
    sup, pmf = hypergeom_pmf(r, c, n)
    pval = np.cumsum(pmf[::-1])[::-1]  # P(A ≥ a) at each support point
    rej = pval <= alpha
    seen = float(pmf[sup >= 1].sum())
    p_rej = float(pmf[rej].sum())
    return {"p_seen": seen, "p_reject": p_rej, "p_reject_given_seen": float(pmf[rej & (sup >= 1)].sum()) / seen}


def selection_curve(n: int, rs: tuple[int, ...]) -> dict[str, list[float]]:
    """Prop 5 across margins r = c: P(seen), P(reject), and P(reject | seen) against E = r²/N."""
    rows = [selection(r, r, n, ALPHA) for r in rs]
    return {"r": list(rs), "expected": [r * r / n for r in rs], **{k: [x[k] for x in rows] for k in rows[0]}}


def chi2_limit_draws(n: int, pr: float, pc: float, reps: int, k: int) -> dict[str, np.ndarray]:
    """Thm 2 on random instances: multinomial tables under independence from stream k; returns x2, g2, and √N·φ̂ for
    the KS checks against χ²₁ and N(0, 1), and |g2 − x2| shrinking with N."""
    p = np.outer([pr, 1 - pr], [pc, 1 - pc]).ravel()
    o = _rng(k).multinomial(n, p, size=reps).astype(float)
    t = Table(o[:, 0], o[:, 0] + o[:, 1], o[:, 0] + o[:, 2], float(n))
    s = scores(t)
    return {"x2": s["x2"], "g2": s["g2"], "z": math.sqrt(n) * phi(t)}


# --------------------------------------------------------------------------------------------
# 4. Keyness (Def 5, Prop 6).
# --------------------------------------------------------------------------------------------


def keyness(target: Counter, reference: Counter) -> dict[str, np.ndarray]:
    """Def 5: for every word, the table (count in target, target size, count overall, total size). Returns words,
    signed g2 (positive when over-used in the target), the log₂ ratio with ½-smoothing, and the counts."""
    words = np.array(sorted(set(target) | set(reference)))
    a = np.array([target[w] for w in words], dtype=float)
    b = np.array([reference[w] for w in words], dtype=float)
    nt, nr = float(sum(target.values())), float(sum(reference.values()))
    g2 = scores(Table(a, nt, a + b, nt + nr))["g2"]
    over = a / nt > b / nr
    return {"words": words, "g2": np.where(over, g2, -g2), "log_ratio": np.log2((a + 0.5) / nt) - np.log2((b + 0.5) / nr),
            "target": a, "reference": b}


def rayson_garside_ll(a: np.ndarray, b: np.ndarray, nt: float, nr: float) -> np.ndarray:
    """Rayson and Garside's (2000, §2) keyness LL: 2[a log(a/E₁) + b log(b/E₂)] over the word's row only, with
    E₁ = N_T(a + b)/(N_T + N_R), E₂ = N_R(a + b)/(N_T + N_R). Rem 3: it is G² minus the other-words row's term (≥ 0)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    e1, e2 = nt * (a + b) / (nt + nr), nr * (a + b) / (nt + nr)
    return 2 * (xlogy(a, a / e1) + xlogy(b, b / e2))


# --------------------------------------------------------------------------------------------
# 5. TextRank (Def 6, Def 7, Thm 3, Thm 4, Prop 7).
# --------------------------------------------------------------------------------------------


def cooc_graph(sents, window: int, weighted: bool, stop: frozenset[str]) -> tuple[tuple[str, ...], sparse.csr_matrix]:
    """Def 6: words not in `stop` that co-occur with another word are nodes; two distinct words within `window`
    positions of each other in the stop-filtered sentence are joined (weight = number of such co-occurrences, or 1 if
    not weighted)."""
    pairs: Counter = Counter()
    for s in sents:
        f = [w for w in s if w not in stop]
        for i in range(len(f)):
            for j in range(i + 1, min(i + window, len(f))):
                if f[i] != f[j]:
                    pairs[(f[i], f[j])] += 1
    nodes = tuple(sorted({w for p in pairs for w in p}))
    ix = {w: i for i, w in enumerate(nodes)}
    rows = np.array([ix[p[0]] for p in pairs] + [ix[p[1]] for p in pairs])
    cols = np.array([ix[p[1]] for p in pairs] + [ix[p[0]] for p in pairs])
    vals = np.array(list(pairs.values()) * 2, dtype=float)
    w = sparse.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    if not weighted:
        w.data[:] = 1.0
    return nodes, w


def _transition(w) -> tuple[sparse.csr_matrix, np.ndarray]:
    """P = D⁻¹W with dangling rows left zero, and the dangling indicator."""
    w = sparse.csr_matrix(w, dtype=float)
    s = np.asarray(w.sum(1)).ravel()
    dangling = s == 0
    inv = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, s))
    return sparse.diags(inv) @ w, dangling


def pagerank(w, d: float, tol: float, x0: np.ndarray | None = None,
             teleport: np.ndarray | None = None) -> tuple[np.ndarray, list[float]]:
    """Def 7 by power iteration, x ← (1 − d)v + d Pᵀx with P = D⁻¹W and teleport distribution v (uniform unless given;
    rows without edges teleport uniformly). Returns π and the ℓ1 step sizes ‖x_{t+1} − x_t‖₁ (Thm 4: ≤ dᵗ‖x₁ − x₀‖₁)."""
    p, dangling = _transition(w)
    pt = p.T.tocsr()
    n = p.shape[0]
    v = np.full(n, 1.0 / n) if teleport is None else np.asarray(teleport, dtype=float)
    x = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, dtype=float)
    steps: list[float] = []
    for _ in range(100_000):
        y = (1 - d) * v + d * (pt @ x + x[dangling].sum() / n)
        steps.append(float(np.abs(y - x).sum()))
        x = y
        if steps[-1] < tol:
            break
    return x, steps


def google_matrix(w, d: float) -> np.ndarray:
    """Def 7's G = (1 − d)/n 𝟏𝟏ᵀ + d P̃, where P̃ is P with dangling rows replaced by the uniform row (dense)."""
    p, dangling = _transition(w)
    p = p.toarray()
    n = p.shape[0]
    p[dangling] = 1.0 / n
    return (1 - d) / n + d * p


def pagerank_dense(w, d: float) -> np.ndarray:
    """Thm 3's cross-check: the Perron eigenvector of the dense Google matrix, normalized to sum 1."""
    vals, vecs = np.linalg.eig(google_matrix(w, d).T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def strength_identity(sents, stop: frozenset[str]) -> dict[str, int]:
    """Prop 7's exact form for window 2: strength(w) = 2c(w) − e(w) − 2s(w), where c counts w in the stop-filtered
    stream, e its occurrences at a sentence edge (a one-word sentence counts twice), and s the adjacent (w, w) pairs.
    Returns the per-word mismatch against the graph (empty when the identity holds)."""
    c, e, s_ = Counter(), Counter(), Counter()
    for sent in sents:
        f = [w for w in sent if w not in stop]
        if not f:
            continue
        c.update(f)
        e[f[0]] += 1
        e[f[-1]] += 1
        for x, y in zip(f, f[1:]):
            if x == y:
                s_[x] += 1
    nodes, w = cooc_graph(sents, 2, True, stop)
    strength = dict(zip(nodes, np.asarray(w.sum(1)).ravel()))
    out = {}
    for x in c:
        mism = int(round(strength.get(x, 0.0))) - (2 * c[x] - e[x] - 2 * s_[x])
        if mism:
            out[x] = mism
    return out


def random_graph(n: int, p: float, k: int, directed: bool) -> sparse.csr_matrix:
    """An Erdős–Rényi graph with weights in {1, 2, 3} from stream k (directed: independent arcs; some rows dangle)."""
    rng = _rng(k)
    m = (rng.random((n, n)) < p) * rng.integers(1, 4, size=(n, n))
    np.fill_diagonal(m, 0)
    if not directed:
        m = np.triu(m, 1)
        m = m + m.T
    return sparse.csr_matrix(m.astype(float))


# --------------------------------------------------------------------------------------------
# 6. Finance case study.
# --------------------------------------------------------------------------------------------

_OPEN, _CLOSE, _JOIN = "\x01", "\x02", "\x03"  # around a drawn word; between a compound's two halves


class _TaggedGen(fc._CorpusGen):
    """generate_corpus(300, 0)'s generator, with every restaurant draw wrapped in sentinels. It makes the same random
    calls in the same order, so stripping the sentinels gives the corpus text exactly (asserted)."""

    def word(self, sector: str) -> str:
        return _OPEN + super().word(sector) + _CLOSE

    def phrase(self, sector: str) -> str:
        first = self.word(sector)
        if self.rng.random() < fc.COMPOUND_RATE:
            second = self.word(sector)
            return f"{first}{_JOIN}{second}" if second != first else first
        return first


_TOKEN = re.compile(r"[a-z]+")


def _strip(marked: str) -> tuple[str, np.ndarray]:
    """The clean text and, per clean character, whether it lies inside a restaurant draw."""
    out, drawn, inside = [], [], False
    for ch in marked:
        if ch == _OPEN:
            inside = True
        elif ch == _CLOSE:
            inside = False
        elif ch == _JOIN:
            out.append(" ")
            drawn.append(False)
        else:
            out.append(ch)
            drawn.append(inside)
    return "".join(out), np.array(drawn)


@lru_cache(maxsize=1)
def tagged_documents() -> tuple[tuple[tuple[tuple[str, ...], tuple[str, ...]], ...], ...]:
    """Topic 3's training text as documents of sentences, each token labeled D (a restaurant draw), E (inside an
    entity span), or L (template text). The sentences equal ng.corpus()["train"] (asserted in the harness)."""
    gen = _TaggedGen(0, None, fc.HEAPS_DISCOUNT)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    marked = [gen.doc(i, int(t)) for i, t in enumerate(tones)]
    out = []
    for md, cd in zip(marked, ds._generator_run()[0]):
        text, drawn = _strip(md.text)
        if text != cd.text:
            raise AssertionError(f"tagged run diverged at {cd.doc_id}")
        ent = np.zeros(len(text), dtype=bool)
        for e in cd.entities:
            ent[e.start:e.end] = True
        sents, pos = [], 0
        for piece in ng._SENTENCE_END.split(text):
            start = text.index(piece, pos)
            pos = start + len(piece)
            toks = [(m.group(), start + m.start()) for m in _TOKEN.finditer(piece.lower())]
            if toks:
                sents.append((tuple(w for w, _ in toks),
                              tuple("D" if drawn[i] else "E" if ent[i] else "L" for _, i in toks)))
        out.append(tuple(sents))
    return tuple(out)


def tagged_sentences() -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(s for d in tagged_documents() for s in d)


def doc_sectors() -> tuple[str, ...]:
    return tuple(fc._PUBLIC[d.company][1] for d in ds._generator_run()[0])


@lru_cache(maxsize=1)
def entity_words() -> frozenset[str]:
    """Every word type that occurs inside an entity span somewhere in the training text."""
    return frozenset(w for toks, labs in tagged_sentences() for w, l in zip(toks, labs) if l == "E")


def _pair_class(kinds: Counter, w1: str, w2: str, home: dict[str, str]) -> str:
    """'fixed': only ever template–template or entity–entity; 'topical': only ever draw–draw, both words planted in
    one sector; 'null': only ever draw–draw otherwise (a shared-vocabulary word on either side: independent given the
    sector); 'boundary': everything else."""
    if set(kinds) <= {"LL", "EE"}:
        return "fixed"
    if set(kinds) == {"DD"}:
        h1, h2 = home.get(w1), home.get(w2)
        if h1 is None or h2 is None:
            return "boundary"
        return "topical" if h1 == h2 != "common" else "null"
    return "boundary"


@lru_cache(maxsize=1)
def fin_bigrams() -> dict:
    """Every bigram type in the training sentences (adjacent tokens within a sentence, no padding) with its table,
    Def 3's scores, the one-sided exact p-value, and its provenance class (_pair_class)."""
    counts: Counter = Counter()
    kinds: dict[tuple[str, str], Counter] = {}
    for toks, labs in tagged_sentences():
        for i in range(len(toks) - 1):
            g = (toks[i], toks[i + 1])
            counts[g] += 1
            kinds.setdefault(g, Counter())[labs[i] + labs[i + 1]] += 1
    types = sorted(counts)
    n = float(sum(counts.values()))
    first, second = Counter(), Counter()
    for (x, y), k in counts.items():
        first[x] += k
        second[y] += k
    a = np.array([counts[g] for g in types], dtype=float)
    r = np.array([first[g[0]] for g in types], dtype=float)
    c = np.array([second[g[1]] for g in types], dtype=float)
    home = ds.planted_labels()
    return {"types": types, "n": n, "a": a, "r": r, "c": c, **scores(Table(a, r, c, n)),
            "logp": (logp := log_p_greater(a, r, c, n)), "p": np.exp(logp),
            "cls": np.array([_pair_class(kinds[g], g[0], g[1], home) for g in types])}


MEASURES = ("pmi", "t", "x2", "g2", "fisher", "freq")
RANK_KS = (10, 50, 200, 500)


def rank_key(b: dict, measure: str) -> np.ndarray:
    """The score each measure ranks by (larger is more associated); the exact test ranks by −log p."""
    if measure == "fisher":
        return -b["logp"]
    return b["a"] if measure == "freq" else b[measure]


def tie_blocks(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices sorted by key (descending; alphabetical within a value) and a block id per sorted position. Values equal
    to a relative 10⁻⁹ share a block (X² = N for every perfect pair, computed with rounding error)."""
    order = np.argsort(-key, kind="stable")
    v = key[order]
    gap = np.abs(np.diff(v)) > 1e-9 * np.maximum(1.0, np.abs(v[1:]))
    return order, np.concatenate([[0], np.cumsum(gap)])


def expected_topk(key: np.ndarray, mask: np.ndarray, ks) -> np.ndarray:
    """E[share of `mask` in the top k] when ties are broken uniformly at random, for every k in ks: whole blocks count
    their members, and the block cut at k contributes its mask share for each of the slots it fills."""
    order, block = tie_blocks(key)
    m = mask[order].astype(float)
    sizes = np.bincount(block)
    hits = np.bincount(block, weights=m)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    before = np.concatenate([[0.0], np.cumsum(hits)[:-1]])
    out = []
    for k in ks:
        j = block[k - 1]
        out.append((before[j] + (k - start[j]) * hits[j] / sizes[j]) / k)
    return np.array(out)


def fin_rankings(ks: tuple[int, ...]) -> dict:
    """Per measure, with ties broken at random in expectation: precision@k against 'fixed' and the shares of 'null' and
    'topical' pairs in the top k (formalRAG set-metrics Def 1), the size of the block tied at the top, and the top 15
    (alphabetical within ties) with their classes."""
    b = fin_bigrams()
    out = {}
    for m in MEASURES:
        key = rank_key(b, m)
        order, block = tie_blocks(key)
        out[m] = {name: dict(zip(ks, expected_topk(key, b["cls"] == cl, ks).tolist()))
                  for name, cl in (("precision", "fixed"), ("null", "null"), ("topical", "topical"))}
        out[m]["top_block"] = int((block == 0).sum())
        out[m]["top"] = [(" ".join(b["types"][i]), str(b["cls"][i]), int(b["a"][i])) for i in order[:15]]
    return out


def bh_threshold(p: np.ndarray, m: int, q: float) -> float:
    """Benjamini–Hochberg over m hypotheses of which `p` are the ones not trivially 1: the largest p_(i) ≤ qi/m (0 if
    none). formalStatistics multiple-testing-and-false-discovery, Thm 5."""
    s = np.sort(p)
    ok = np.nonzero(s <= q * np.arange(1, len(s) + 1) / m)[0]
    return float(s[ok[-1]]) if len(ok) else 0.0


@lru_cache(maxsize=1)
def dd_stratum() -> dict:
    """The draw–draw bigram positions alone, as their own 2×2 world: N, the margins over first and second slots, and
    each observed pair's exact one-sided p-value and class. Within it, a 'null' pair is independent by construction."""
    counts: Counter = Counter()
    for toks, labs in tagged_sentences():
        for i in range(len(toks) - 1):
            if labs[i] == "D" and labs[i + 1] == "D":
                counts[(toks[i], toks[i + 1])] += 1
    types = sorted(counts)
    n = sum(counts.values())
    first, second = Counter(), Counter()
    for (x, y), k in counts.items():
        first[x] += k
        second[y] += k
    a = np.array([counts[g] for g in types])
    r = np.array([first[g[0]] for g in types])
    c = np.array([second[g[1]] for g in types])
    home = ds.planted_labels()
    cls = np.array([_pair_class(Counter({"DD": 1}), *g, home) for g in types])
    return {"types": types, "n": n, "a": a, "r": r, "c": c, "p": stats.hypergeom.sf(a - 1, n, r, c), "cls": cls,
            "m_all": len(first) * len(second)}


def fin_selection() -> dict:
    """Prop 5 in the corpus: on the D–D stratum, BH at q = 0.05 over the observed pairs only vs over all |W₁|·|W₂|
    candidate pairs (unobserved pairs have p = 1); rejections per class under each, the nominal-0.05 rejection rate,
    and the share of D–D pairs seen once. The same comparison on the pooled table of every bigram closes it."""
    s = dd_stratum()
    out: dict = {"n": s["n"], "types": len(s["types"]), "m_all": s["m_all"], "seen_once": float((s["a"] == 1).mean())}
    for name, m in (("observed", len(s["p"])), ("all", s["m_all"])):
        thr = bh_threshold(s["p"], m, ALPHA)
        rej = s["p"] <= thr if thr > 0 else np.zeros(len(s["p"]), dtype=bool)
        out[name] = {k: (int((s["cls"] == k).sum()), int(rej[s["cls"] == k].sum())) for k in ("null", "topical")}
    out["nominal"] = {k: float((s["p"][s["cls"] == k] <= ALPHA).mean()) for k in ("null", "topical")}
    b = fin_bigrams()
    thr = bh_threshold(b["p"], len(b["p"]), ALPHA)
    out["pooled_observed"] = {k: (int((b["cls"] == k).sum()), int((b["p"][b["cls"] == k] <= thr).sum()))
                              for k in ("fixed", "null", "topical", "boundary")}
    return out


KEY_K = 25
TOP_STOP = 100  # TextRank's stop list: the corpus's most frequent words (no tagger; Mihalcea and Tarau filter by POS)


@lru_cache(maxsize=1)
def sector_counts() -> dict[str, Counter]:
    """Per sector, the word counts of its documents, entity words excluded (they are keyness's trivial winners)."""
    ent = entity_words()
    out = {s: Counter() for s in fc.SECTORS}
    for sec, doc in zip(doc_sectors(), tagged_documents()):
        out[sec].update(w for toks, _ in doc for w in toks if w not in ent)
    return out


def fin_keyness(k: int) -> dict:
    """Each sector's documents against the rest: precision@k of signed G², the log ratio, and frequency, scored against
    the planted sector nouns (ds.planted_labels()); and the top 10 by G²."""
    home = ds.planted_labels()
    cnt = sector_counts()
    out = {}
    for s in fc.SECTORS:
        rest = Counter()
        for t in fc.SECTORS:
            if t != s:
                rest.update(cnt[t])
        kz = keyness(cnt[s], rest)
        gold = np.array([home.get(w) == s for w in kz["words"]])
        row = {}
        for m, key in (("g2", kz["g2"]), ("log_ratio", kz["log_ratio"]), ("freq", kz["target"])):
            row[m] = float(gold[np.argsort(-key, kind="stable")[:k]].mean())
        row["top"] = [str(w) for w in kz["words"][np.argsort(-kz["g2"], kind="stable")[:10]]]
        row["docs"] = sum(1 for x in doc_sectors() if x == s)
        out[s] = row
    return out


@lru_cache(maxsize=1)
def textrank_stop() -> frozenset[str]:
    """TextRank's candidate filter: the TOP_STOP most frequent training words, and every entity word."""
    c = Counter(w for toks, _ in tagged_sentences() for w in toks)
    return frozenset(w for w, _ in c.most_common(TOP_STOP)) | entity_words()


TR_D = (0.5, 0.85, 0.95, 0.99)


def fin_textrank(k: int, d_grid: tuple[float, ...]) -> dict:
    """Per sector, TextRank (weighted and binary, window 2) on the sector's sentences with textrank_stop() removed:
    precision@k against the planted sector nouns at each d, Spearman with the filtered frequency, iterations to 10⁻¹²;
    and keyness on the same candidates for comparison."""
    home = ds.planted_labels()
    stop = textrank_stop()
    secs = doc_sectors()
    out = {}
    for s in fc.SECTORS:
        sents = [toks for sec, doc in zip(secs, tagged_documents()) if sec == s for toks, _ in doc]
        freq = Counter(w for t in sents for w in t if w not in stop)
        row: dict = {}
        for weighted in (True, False):
            nodes, w = cooc_graph(sents, 2, weighted, stop)
            gold = np.array([home.get(x) == s for x in nodes])
            f = np.array([freq[x] for x in nodes], dtype=float)
            per_d = {}
            for d in d_grid:
                pr, steps = pagerank(w, d, 1e-12)
                per_d[d] = {"precision": float(gold[np.argsort(-pr, kind="stable")[:k]].mean()),
                            "spearman": float(stats.spearmanr(pr, f)[0]), "iterations": len(steps)}
            row["weighted" if weighted else "binary"] = per_d
        nodes, w = cooc_graph(sents, 2, True, stop)
        gold = np.array([home.get(x) == s for x in nodes])
        f = np.array([freq[x] for x in nodes], dtype=float)
        row["freq"] = float(gold[np.argsort(-f, kind="stable")[:k]].mean())
        row["nodes"] = len(nodes)
        out[s] = row
    return out


@lru_cache(maxsize=1)
def fin_edgar_keyness() -> dict:
    """The synthetic corpus against the three EDGAR excerpts: the top 15 by signed G², and the share of the top 50 that
    are template words (types that never occur as a draw or inside an entity). What a reference corpus finds first is
    what makes the target unusual, and here that is its templates."""
    syn = Counter(w for toks, _ in tagged_sentences() for w in toks)
    edgar = Counter(w for s in ng.corpus()["edgar"] for w in s)
    kz = keyness(syn, edgar)
    order = np.argsort(-kz["g2"], kind="stable")
    labels: dict[str, set[str]] = {}
    for toks, labs in tagged_sentences():
        for w, l in zip(toks, labs):
            labels.setdefault(w, set()).add(l)
    top50 = [str(kz["words"][i]) for i in order[:50]]
    return {"top": [(str(kz["words"][i]), float(kz["g2"][i])) for i in order[:15]],
            "template_share": float(np.mean([labels.get(w) == {"L"} for w in top50])),
            "edgar_tokens": int(sum(edgar.values())), "synthetic_tokens": int(sum(syn.values()))}


# --------------------------------------------------------------------------------------------
# 7. The lab's payload.
# --------------------------------------------------------------------------------------------

GEOMETRY_PRESETS = ((100, 40, 50), (1000, 20, 30), (1000, 5, 5))
RARE_N = 57_052  # the training text's bigram positions (fin_bigrams()["n"], asserted)
RARE_K = tuple(range(1, 21))
SIZE_R = tuple(range(1, 451))  # r = c at SIZE_N: E₁₁ = r²/N from 10⁻⁴ to 20.25
SELECTION_R = tuple(range(1, 81))
CURVE_KS = tuple(range(1, 501))


def viz_payload() -> dict:
    """Every number the lab shows. Scores at 3 decimals, probabilities at 4 significant figures (they span decades),
    shares at 3 decimals."""

    def sig(x, n: int = 4):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [sig(v, n) for v in x]
        x = float(x)
        return None if not math.isfinite(x) else float(f"{x:.{n}g}")

    geometry = []
    for n, r, c in GEOMETRY_PRESETS:
        g = saddle_slice(r, c, n)
        geometry.append({"n": n, "r": r, "c": c, "expected": _r(g["expected"]), "a": g["a"].tolist(),
                         "xyz": _r(g["xyz"], 4), "p": sig(g["p_exact"]),
                         **{k: _r(g[k]) for k in ("pmi", "t", "x2", "g2")}})
    surface = independence_surface(11)
    pp = perfect_pair(np.array(RARE_K), RARE_N)
    sc = size_curve(SIZE_N, np.array([r * r / SIZE_N for r in SIZE_R]))
    dd = dd_stratum()
    sel = selection_curve(dd["n"], SELECTION_R)
    b = fin_bigrams()
    rank = fin_rankings(RANK_KS)
    curves = {m: _r(expected_topk(rank_key(b, m), b["cls"] == "fixed", CURVE_KS)) for m in MEASURES}
    nulls = {m: _r(expected_topk(rank_key(b, m), np.isin(b["cls"], ("null", "topical")), CURVE_KS)) for m in MEASURES}
    fs = fin_selection()
    kw = fin_keyness(KEY_K)
    tr = fin_textrank(KEY_K, TR_D)
    nodes, w = cooc_graph([t for s_, d in zip(doc_sectors(), tagged_documents()) if s_ == "industrials" for t, _ in d],
                          2, True, textrank_stop())
    conv = {f"{d:g}": sig(pagerank(w, d, 1e-12)[1][:400:4]) for d in TR_D}
    return {
        "geometry": {"presets": geometry, "surface": _r(surface, 4)},
        "rare": {"n": RARE_N, "k": list(RARE_K), **{key: _r(pp[key]) for key in ("pmi", "t", "x2", "g2", "neglog_p")},
                 "hapax": {k: sig(v, 6) for k, v in hapax_forms(RARE_N).items()}},
        "size": {"n": SIZE_N, "r": sc["r"], "expected": sig(sc["expected"]),
                 **{k: sig(sc[k]) for k in ("x2", "g2", "fisher")}},
        "selection": {"n": dd["n"], "r": sel["r"], "expected": sig(sel["expected"]),
                      **{k: sig(sel[k]) for k in ("p_seen", "p_reject", "p_reject_given_seen")},
                      "corpus": {"types": fs["types"], "m_all": fs["m_all"], "seen_once": _r(fs["seen_once"]),
                                 "observed": fs["observed"], "all": fs["all"],
                                 "nominal": {k: _r(v) for k, v in fs["nominal"].items()}}},
        "bigrams": {"n": int(b["n"]), "types": len(b["types"]),
                    "classes": {k: int((b["cls"] == k).sum()) for k in ("fixed", "null", "topical", "boundary")},
                    "precision": curves, "nulls": nulls,
                    "top": {m: rank[m]["top"] for m in MEASURES}, "top_block": {m: rank[m]["top_block"] for m in MEASURES}},
        "keywords": {s: {"docs": kw[s]["docs"], "keyness_top": kw[s]["top"],
                         "keyness": {m: _r(kw[s][m]) for m in ("g2", "log_ratio", "freq")},
                         "textrank": {kind: {f"{d:g}": {"precision": _r(v["precision"]), "spearman": _r(v["spearman"]),
                                                        "iterations": v["iterations"]} for d, v in tr[s][kind].items()}
                                      for kind in ("weighted", "binary")},
                         "freq_after_stop": _r(tr[s]["freq"]), "nodes": tr[s]["nodes"]} for s in fc.SECTORS},
        "convergence": {"sector": "industrials", "nodes": len(nodes), "every": 4, "steps": conv},
    }


# --------------------------------------------------------------------------------------------
# Verification harness. Every pedagogical claim on the page is a test here.
# --------------------------------------------------------------------------------------------


def _random_tables(k: int, m: int, n: int) -> list[Table]:
    """m integer tables with N = n, cells from a Dirichlet–multinomial draw on stream k (all margins positive)."""
    rng = _rng(k)
    out = []
    while len(out) < m:
        o = rng.multinomial(n, rng.dirichlet(np.ones(4)))
        a, b_, c_, d_ = (int(x) for x in o)
        if min(a + b_, c_ + d_, a + c_, b_ + d_) > 0:
            out.append(Table(a, a + b_, a + c_, n))
    return out


def test_thm1_projection_is_product_of_margins():
    rng = _rng(3900)
    for _ in range(20):
        p = rng.dirichlet(np.ones(4)).reshape(2, 2)
        q = independence_projection(p)
        assert np.allclose(q, projection_by_solver(p), atol=1e-6)
        u, v = rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(2))
        # The Pythagorean identity behind the proof: KL(p ‖ uvᵀ) = KL(p ‖ q*) + KL(p_r ‖ u) + KL(p_c ‖ v).
        assert abs(kl(p, np.outer(u, v)) - (kl(p, q) + kl(p.sum(1), u) + kl(p.sum(0), v))) < 1e-12


def test_thm1_g2_is_2n_kl_and_2n_mutual_information():
    for t in _random_tables(3901, 50, 400):
        p = t.cells() / t.n
        g2 = float(scores(t)["g2"])
        assert abs(g2 - 2 * t.n * kl(p, independence_projection(p))) < 1e-9 * max(1.0, g2)
        assert abs(g2 - 2 * t.n * mutual_information(p)) < 1e-9 * max(1.0, g2)


def test_prop1_x2_is_n_chi2_divergence_and_n_phi_squared():
    for t in _random_tables(3902, 50, 400):
        p = t.cells() / t.n
        x2 = float(scores(t)["x2"])
        assert abs(x2 - t.n * chi2_div(p, independence_projection(p))) < 1e-9 * max(1.0, x2)
        assert abs(x2 - t.n * float(phi(t)) ** 2) < 1e-9 * max(1.0, x2)


def test_scores_match_scipy():
    for t in _random_tables(3903, 50, 300):
        o = t.cells()
        s = scores(t)
        x2 = stats.chi2_contingency(o, correction=False)[0]
        g2 = stats.chi2_contingency(o, correction=False, lambda_="log-likelihood")[0]
        assert abs(float(s["x2"]) - x2) < 1e-9 * max(1.0, x2) and abs(float(s["g2"]) - g2) < 1e-9 * max(1.0, g2)
        e = t.r * t.c / t.n
        if t.a > 0:
            assert abs(float(s["pmi"]) - math.log(t.a / e)) < 1e-12
            assert abs(float(s["t"]) - (t.a - e) / math.sqrt(t.a)) < 1e-12


def test_prop2_hapax_closed_forms():
    for n in (10, 1000, RARE_N, 10**7):
        s, h = scores(Table(1, 1, 1, n)), hapax_forms(n)
        for k in ("pmi", "x2", "g2", "t"):
            assert abs(float(s[k]) - h[k]) < 1e-9 * max(1.0, abs(h[k])), (n, k)
        assert abs(fisher_p(1, 1, 1, n, "greater") * n - 1) < 1e-7  # log-gamma differences lose ~ε·N log N
        # (N−1) log(N/(N−1)) = 1 − 1/(2N) − 1/(6N²) − …, so G² = 2 log N + 2 − 1/N − O(N⁻²).
        gap = 2 * math.log(n) + 2 - h["g2"]
        assert 1 / n - 1e-12 < gap < 1 / n + 1 / n**2 + 1e-12  # 1e-12: the subtraction's rounding
    # PMI ≤ log N and X² ≤ N on every table with a ≥ 1: the hapax pair attains both maxima.
    for t in _random_tables(3904, 200, 500):
        if t.a >= 1:
            s = scores(t)
            assert float(s["pmi"]) <= math.log(t.n) + 1e-12 and float(s["x2"]) <= t.n * (1 + 1e-12)


def test_prop3_perfect_pairs_invert_pmi():
    k = np.arange(1, 200)
    pp = perfect_pair(k, RARE_N)
    assert np.all(np.diff(pp["pmi"]) < 0) and np.allclose(pp["x2"], RARE_N, rtol=1e-12)
    assert np.allclose(pp["g2"], pp["g2_entropy"], rtol=1e-12) and np.all(np.diff(pp["g2"]) > 0)
    assert np.all(np.diff(pp["t"]) > 0) and np.all(np.diff(pp["neglog_p"]) > 0)
    for j in (1, 2, 5, 20):
        assert abs(-log_p_greater(j, j, j, RARE_N) - pp["neglog_p"][j - 1]) < 1e-9 * pp["neglog_p"][j - 1]
    assert abs(pp["neglog_p"][0] - pp["pmi"][0]) < 1e-9  # at k = 1, −log p = PMI = log N


def test_lemma1_hypergeometric_matches_scipy_and_enumeration():
    rng = _rng(3905)
    for _ in range(30):
        n = int(rng.integers(20, 400))
        r, c = (int(x) for x in rng.integers(1, n, size=2))
        sup, pmf = hypergeom_pmf(r, c, n)
        assert np.allclose(pmf, stats.hypergeom.pmf(sup, n, r, c), rtol=1e-10, atol=1e-300)
        assert abs(pmf.sum() - 1) < 1e-12
    # Lemma 1 itself: multinomial tables under independence, conditioned on their margins, by exact enumeration.
    n, pr, pc = 7, 0.3, 0.6
    probs = np.outer([pr, 1 - pr], [pc, 1 - pc]).ravel()
    joint: dict[tuple[int, int], dict[int, float]] = {}
    for o in ((i, j, k, n - i - j - k) for i in range(n + 1) for j in range(n + 1 - i) for k in range(n + 1 - i - j)):
        w = stats.multinomial.pmf(o, n, probs)
        joint.setdefault((o[0] + o[1], o[0] + o[2]), {}).setdefault(o[0], 0.0)
        joint[(o[0] + o[1], o[0] + o[2])][o[0]] += w
    for (r, c), dist in joint.items():
        tot = sum(dist.values())
        sup, pmf = hypergeom_pmf(r, c, n)
        assert np.allclose([dist.get(int(a), 0.0) / tot for a in sup], pmf, atol=1e-12)


def test_def4_fisher_matches_scipy():
    for t in _random_tables(3906, 60, 120):
        o = t.cells().astype(int)
        for alt in ("two-sided", "greater"):
            ref = stats.fisher_exact(o, alternative=alt).pvalue
            assert abs(fisher_p(int(t.a), int(t.r), int(t.c), int(t.n), alt) - ref) < 1e-10


THM2_N = (1000, 10_000, 100_000)


@lru_cache(maxsize=1)
def thm2_draws() -> tuple[dict, ...]:
    return tuple(chi2_limit_draws(n, 0.3, 0.4, 20_000, 3907 + j) for j, n in enumerate(THM2_N))


def test_thm2_asymptotics_on_random_instances():
    med = []
    for n, d in zip(THM2_N, thm2_draws()):
        if n >= 10_000:
            assert stats.kstest(d["z"], "norm").statistic < 0.015
            assert stats.kstest(d["x2"], stats.chi2(1).cdf).statistic < 0.015
            assert stats.kstest(d["g2"], stats.chi2(1).cdf).statistic < 0.015
        med.append(float(np.median(np.abs(d["g2"] - d["x2"]))))
    assert med[0] / med[1] > 2.5 and med[1] / med[2] > 2.5  # G² − X² = O_p(N^{-1/2}): ≈ √10 per decade


@lru_cache(maxsize=1)
def _size_dense() -> dict:
    return size_curve(SIZE_N, np.array([r * r / SIZE_N for r in SIZE_R]))


def test_prop4_exact_sizes_far_from_nominal_at_small_e():
    sc = _size_dense()
    e, x2, g2, fi = (np.array(sc[k]) for k in ("expected", "x2", "g2", "fisher"))
    assert x2.max() > 0.15 and 0.1 < e[x2.argmax()] < 0.3  # X²: anti-conservative near E ≈ 0.2
    assert g2.max() > 0.12 and 1 < e[g2.argmax()] < 4  # G²: anti-conservative near E ≈ 2 (Moore 2004)
    assert fi.max() <= ALPHA + 1e-12  # the exact test never exceeds α
    for r, ei, row in zip(sc["r"], e, zip(x2, g2, fi)):
        seen = selection(r, r, SIZE_N, ALPHA)["p_seen"]
        if seen <= ALPHA:  # every sighting rejects, under every test: the size is P(seen)
            assert np.allclose(row, seen, rtol=1e-9)


def test_prop5_selection_makes_every_sighting_significant():
    rng = _rng(3910)
    for _ in range(200):
        n = int(rng.integers(100, 20_000))
        r, c = (int(x) for x in rng.integers(1, max(2, n // 20), size=2))
        s = selection(r, c, n, ALPHA)
        assert s["p_reject"] <= ALPHA + 1e-12
        if s["p_seen"] <= ALPHA:
            assert s["p_reject_given_seen"] == 1.0 and abs(s["p_reject"] - s["p_seen"]) < 1e-12
        assert s["p_reject_given_seen"] <= min(1.0, ALPHA / s["p_seen"]) + 1e-12
    # At the corpus stratum's N, every sighting rejects exactly for r = c ≤ 14 (E ≤ 0.047); at r = 20 it is 0.0428.
    curve = selection_curve(dd_stratum()["n"], SELECTION_R)
    assert [x == 1.0 for x in curve["p_reject_given_seen"]] == [r <= 14 for r in SELECTION_R]
    assert abs(curve["p_reject_given_seen"][19] - 0.0428) < 5e-5


def test_observed_only_bh_rejects_nulls_all_pairs_bh_does_not():
    fs = fin_selection()
    assert fs["seen_once"] > 0.97
    null_n, null_rej = fs["observed"]["null"]
    assert null_rej / null_n > 0.7 and fs["nominal"]["null"] > 0.7
    assert fs["all"]["null"][1] == 0 and fs["all"]["topical"][1] == 0
    s = dd_stratum()
    assert s["p"].min() > ALPHA / s["m_all"]  # Bonferroni over all candidate pairs: no rejection either
    # The pooled table repeats it: most null pairs pass BH over the observed bigrams.
    pooled_n, pooled_rej = fs["pooled_observed"]["null"]
    assert pooled_rej / pooled_n > 0.7


def test_prop6_keyness_is_the_same_table():
    syn = sector_counts()["energy"]
    rest = Counter()
    for s in fc.SECTORS:
        if s != "energy":
            rest.update(sector_counts()[s])
    kz = keyness(syn, rest)
    nt, nr = sum(syn.values()), sum(rest.values())
    for i in range(0, len(kz["words"]), 97):
        a, b_ = kz["target"][i], kz["reference"][i]
        o = np.array([[a, nt - a], [b_, nr - b_]])  # word × corpus, transposed from Def 5's layout
        g2 = stats.chi2_contingency(o, correction=False, lambda_="log-likelihood")[0] if a + b_ > 0 else 0.0
        assert abs(abs(kz["g2"][i]) - g2) < 1e-8 * max(1.0, g2)
        assert abs(abs(kz["g2"][i]) - 2 * (nt + nr) * mutual_information(o / (nt + nr))) < 1e-8 * max(1.0, g2)


def test_thm3_pagerank_matches_dense_eigenvector():
    for j, d in enumerate((0.5, 0.85, 0.99)):
        w = random_graph(60, 0.05, 3911 + j, directed=True)
        assert (np.asarray(w.sum(1)).ravel() == 0).any()  # dangling rows present
        pi, _ = pagerank(w, d, 1e-15)
        assert np.abs(pi - pagerank_dense(w, d)).max() < 1e-12
        assert abs(pi.sum() - 1) < 1e-12 and pi.min() >= (1 - d) / 60 * (1 - 1e-12)
        lam = np.sort(np.abs(np.linalg.eigvals(google_matrix(w, d))))
        assert abs(lam[-1] - 1) < 1e-10 and lam[-2] <= d + 1e-10  # Thm 3(c): every other |λ| ≤ d


def test_thm4_power_iteration_contracts_by_d():
    rng = _rng(3914)
    w = random_graph(80, 0.06, 3915, directed=True)
    for d in (0.5, 0.85, 0.95):
        pi = pagerank_dense(w, d)
        x0 = rng.dirichlet(np.ones(80))
        x, steps = x0.copy(), []
        g = google_matrix(w, d)
        for t in range(60):
            assert np.abs(x - pi).sum() <= d**t * np.abs(x0 - pi).sum() * (1 + 1e-9) + 1e-13  # 1e-13: rounding floor
            y = x @ g
            steps.append(np.abs(y - x).sum())
            x = y
        assert all(s <= d**t * steps[0] * (1 + 1e-9) + 1e-13 for t, s in enumerate(steps))


def test_rem3_rayson_garside_is_g2_minus_a_nonnegative_term():
    cnt = sector_counts()
    for s in fc.SECTORS:
        rest = Counter()
        for t in fc.SECTORS:
            if t != s:
                rest.update(cnt[t])
        kz = keyness(cnt[s], rest)
        nt, nr = sum(cnt[s].values()), sum(rest.values())
        ll = rayson_garside_ll(kz["target"], kz["reference"], nt, nr)
        g2 = np.abs(kz["g2"])
        gap = g2 - ll
        # The omitted row: 2[(N_T − a) log((N_T − a)/(N_T − E₁)) + (N_R − b) log(…)] ≥ 0, and KL ≤ χ² bounds it by
        # 2(a − E₁)²[1/(N_T − E₁) + 1/(N_R − E₂)].
        a_, b2 = kz["target"], kz["reference"]
        e1, e2 = nt * (a_ + b2) / (nt + nr), nr * (a_ + b2) / (nt + nr)
        assert gap.min() > -1e-9 and np.all(gap <= 2 * (a_ - e1) ** 2 * (1 / (nt - e1) + 1 / (nr - e2)) + 1e-9)
        assert np.max(gap / np.maximum(g2, 1e-12)) < 0.07
        top = np.argsort(-kz["g2"], kind="stable")[:KEY_K]
        top_ll = np.argsort(-np.where(kz["g2"] > 0, ll, -ll), kind="stable")[:KEY_K]
        assert set(top) == set(top_ll), s  # the same top 25 in every sector


def test_prop7_strength_identity_and_d_to_one_limit():
    assert strength_identity(ng.corpus()["train"], textrank_stop()) == {}
    assert strength_identity(ng.corpus()["train"], frozenset()) == {}
    w = random_graph(50, 0.15, 3916, directed=False)
    s = np.asarray(w.sum(1)).ravel()
    assert (s > 0).all()
    # Connected (asserted through the dense spectrum: eigenvalue 1 of D⁻¹W is simple), so π_d → s/Σs.
    p = _transition(w)[0].toarray()
    assert np.sum(np.abs(np.linalg.eigvals(p) - 1) < 1e-9) == 1
    for d, tol in ((1 - 1e-4, 1e-3), (1 - 1e-7, 1e-6)):
        assert np.abs(pagerank_dense(w, d) - s / s.sum()).max() < tol
    # Two components: each keeps its node share |C|/n of the mass, spread by strength inside it.
    w2 = sparse.block_diag([random_graph(30, 0.2, 3917, False), random_graph(20, 0.25, 3918, False)]).tocsr()
    s2 = np.asarray(w2.sum(1)).ravel()
    lim = np.concatenate([0.6 * s2[:30] / s2[:30].sum(), 0.4 * s2[30:] / s2[30:].sum()])
    assert np.abs(pagerank_dense(w2, 1 - 1e-7) - lim).max() < 1e-6
    # (c) Teleporting to s/vol instead of uniformly makes π_d = s/vol at every d (Grolmusz's case); uniform does not.
    for d in (0.5, 0.85):
        pi_s, _ = pagerank(w, d, 1e-15, teleport=s / s.sum())
        pi_u, _ = pagerank(w, d, 1e-15)
        assert np.abs(pi_s - s / s.sum()).max() < 1e-13 and np.abs(pi_u - s / s.sum()).max() > 1e-3


def test_finance_text_is_topic3s():
    assert tuple(t for t, _ in tagged_sentences()) == ng.corpus()["train"]
    assert fin_bigrams()["n"] == RARE_N
    home = ds.planted_labels()
    drawn = {w for toks, labs in tagged_sentences() for w, l in zip(toks, labs) if l == "D"}
    assert len(drawn - set(home)) <= 2  # the shared seed "reserves" (two sectors), dropped by planted_labels
    assert {w for toks, labs in tagged_sentences() for w, l in zip(toks, labs) if l == "D" and w in home} <= set(home)


def test_g2_t_fisher_rank_fixed_phrases_pmi_ranks_nulls():
    r = fin_rankings(RANK_KS)
    for m in ("g2", "t", "fisher"):
        assert r[m]["null"][500] == 0 and r[m]["topical"][500] == 0
    assert r["g2"]["precision"][500] > 0.89 and r["fisher"]["precision"][500] > 0.89
    assert r["pmi"]["top_block"] == 97 and r["pmi"]["null"][50] + r["pmi"]["topical"][50] > 0.9
    assert r["x2"]["top_block"] == 196 and r["x2"]["null"][200] > 0.2
    b = fin_bigrams()
    perfect = (b["a"] == b["r"]) & (b["a"] == b["c"])
    assert perfect.sum() == 196 and np.allclose(b["x2"][perfect], b["n"], rtol=1e-9)
    hapax = (b["a"] == 1) & perfect
    assert {k: int((b["cls"][hapax] == k).sum()) for k in ("fixed", "null", "topical")} == {"fixed": 5, "null": 48, "topical": 44}
    assert int((stats.hypergeom.sf(b["a"] - 1, b["n"], b["r"], b["c"]) == 0).sum()) == 46  # why the tails are in log space
    assert math.ceil(math.log(1e-12 / 2) / math.log(0.85)) + 1 == 176


def test_keyness_finds_planted_sector_nouns():
    kw = fin_keyness(KEY_K)
    for s in fc.SECTORS:
        assert kw[s]["g2"] >= 0.8 and kw[s]["log_ratio"] >= 0.8 and kw[s]["freq"] <= 0.12, s
    ek = fin_edgar_keyness()
    assert ek["template_share"] > 0.5


def test_textrank_tracks_frequency():
    tr, kw = fin_textrank(KEY_K, TR_D), fin_keyness(KEY_K)
    for s in fc.SECTORS:
        for kind in ("weighted", "binary"):
            assert tr[s][kind][0.85]["spearman"] > 0.7, (s, kind)
            assert tr[s][kind][0.85]["precision"] < kw[s]["g2"], (s, kind)
        # Thm 4's iteration count: the first step is at most 2, so ⌈log(10⁻¹²/2)/log d⌉ + 1 steps always suffice.
        for d, v in tr[s]["weighted"].items():
            assert v["iterations"] <= math.ceil(math.log(1e-12 / 2) / math.log(d)) + 1


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def summary() -> dict:
    """The numbers the page quotes, at 6 decimals."""
    out: dict = {}
    h = hapax_forms(RARE_N)
    out.update({f"hapax {k} (N = {RARE_N})": v for k, v in h.items()})
    sc = _size_dense()
    for k in ("x2", "g2", "fisher"):
        i = int(np.argmax(sc[k]))
        out[f"size max {k} (N = {SIZE_N})"] = sc[k][i]
        out[f"size argmax E {k}"] = sc["expected"][i]
    for n, d in zip(THM2_N, thm2_draws()):
        out[f"Thm 2 median |G² − X²| at N = {n}"] = float(np.median(np.abs(d["g2"] - d["x2"])))
    sel = selection_curve(dd_stratum()["n"], SELECTION_R)
    out["selection: largest r with P(reject | seen) = 1"] = max(r for r, v in zip(SELECTION_R, sel["p_reject_given_seen"]) if v == 1.0)
    out["selection: P(reject | seen) at r = 20"] = sel["p_reject_given_seen"][19]
    b = fin_bigrams()
    out["bigram positions N"] = int(b["n"])
    out["bigram types"] = len(b["types"])
    for k in ("fixed", "null", "topical", "boundary"):
        out[f"class {k}"] = int((b["cls"] == k).sum())
    r = fin_rankings(RANK_KS)
    for m in MEASURES:
        out[f"{m} top block"] = r[m]["top_block"]
        for k in RANK_KS:
            out[f"{m} precision@{k}"] = r[m]["precision"][k]
            out[f"{m} null+topical@{k}"] = r[m]["null"][k] + r[m]["topical"][k]
    fs = fin_selection()
    out["DD positions"] = fs["n"]
    out["DD types"] = fs["types"]
    out["DD candidate pairs"] = fs["m_all"]
    out["DD seen once"] = fs["seen_once"]
    for k in ("null", "topical"):
        out[f"BH observed-only {k} (n, rejected)"] = str(fs["observed"][k])
        out[f"BH all-pairs {k} (n, rejected)"] = str(fs["all"][k])
        out[f"nominal 0.05 rejection {k}"] = fs["nominal"][k]
    out["pooled BH observed-only (class: n, rejected)"] = str(fs["pooled_observed"])
    kw = fin_keyness(KEY_K)
    tr = fin_textrank(KEY_K, TR_D)
    for s in fc.SECTORS:
        out[f"{s} docs"] = kw[s]["docs"]
        for m in ("g2", "log_ratio", "freq"):
            out[f"{s} keyness {m} P@{KEY_K}"] = kw[s][m]
        for kind in ("weighted", "binary"):
            v = tr[s][kind][0.85]
            out[f"{s} textrank {kind} d=0.85 P@{KEY_K}"] = v["precision"]
            out[f"{s} textrank {kind} d=0.85 spearman"] = v["spearman"]
        out[f"{s} textrank iterations d=0.85"] = tr[s]["weighted"][0.85]["iterations"]
        out[f"{s} frequency after stop P@{KEY_K}"] = tr[s]["freq"]
    ek = fin_edgar_keyness()
    out["EDGAR keyness template share of top 50"] = ek["template_share"]
    out["EDGAR keyness top 8"] = ", ".join(w for w, _ in ek["top"][:8])
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
            print(f"  {k:72} {v:.6f}" if isinstance(v, float) else f"  {k:72} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
