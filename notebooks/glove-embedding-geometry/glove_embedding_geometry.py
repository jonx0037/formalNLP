"""GloVe and the geometry of word embeddings — the reference implementation for formalNLP's eighth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. A log-bilinear model of counts: GloVe's weighted least squares on log co-occurrence counts (Pennington,
   Socher, and Manning 2014), its ratio derivation with the hypotheses stated (Cauchy's equation forces exp),
   and the log-bilinear form that makes PMI rank ≤ d + 2 (Topic 6 Thm 3 iii).
2. Three weights: the Poisson log-bilinear likelihood's excess per cell is X(e^δ − 1 − δ), the Bregman
   divergence of exp, with curvature X. GloVe is its quadratic with the weight X tempered to f(X) and the zero
   cells dropped; SGNS near its optimum is GloVe with frozen biases and weight ab/(a + b) (Topic 7 Cor 1).
3. What the loss cannot see: with flat weights and every cell observed, GloVe is a truncated SVD of the
   double-centered PMI matrix; with additive log counts on the support (constant counts, say) the biases fit
   everything and the vectors are zero. On the 300-document corpus most observed pairs are seen once, and
   GloVe's vectors do not recover the planted sectors; a longer corpus brings them back.
4. The algorithm: GloVe's AdaGrad as the C code runs it (minibatched here), and alternating least squares as an
   exact block-descent cross-check.
5. Gauge and analogies: biases widen the gauge to affine maps; ridge with unpenalized biases makes every
   stationary point balanced and centered; an exact parallelogram is a fact about count ratios and survives every
   gauge, while 3CosAdd's answer does not.
6. Finance case study: relations from the filings knowledge base (city → region, company → sector) as vector
   offsets, for GloVe, SGNS, and Topic 6's SVD.

Cross-checks: the hand-written gradient against complex-step and central differences (Topic 5's checkers); the
flat-weight closed form against ``scipy.optimize`` L-BFGS and ALS; the Poisson loss against ``scipy.stats``; ALS
against AdaGrad on the corpus.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/glove-embedding-geometry/glove_embedding_geometry.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from scipy.special import xlogy

_HERE = Path(__file__).resolve().parent
for _dep in ("word2vec-implicit-factorization", "distributional-semantics-ppmi-svd", "neural-language-models",
             "finance-corpus", "n-gram-models-smoothing"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import distributional_semantics_ppmi_svd as ds  # noqa: E402
import finance_corpus as fc  # noqa: E402
import n_gram_models_smoothing as ng  # noqa: E402
import neural_language_models as nlm  # noqa: E402
import word2vec_implicit_factorization as w2v  # noqa: E402

# ONE stream per random-instance claim, each from default_rng([SEED, k]); k ≥ 800 keeps this module's streams apart
# from Topics 6 and 7.
SEED = 20261023

X_MAX, GAMMA = 100.0, 0.75  # Pennington et al.'s weight f (their α; γ here, since α is Topic 6's smoothing exponent)
CONTEXT, D = "L8", 25  # Topic 7's controlled matrix and rank (Topic 6's selected SVD)
SCALE_DOCS = (300, 1200, 3000)  # the generator continued; 300 is Topic 6's corpus
RIDGE = 1e-3  # ALS's ridge on W and C (biases unpenalized)


# --------------------------------------------------------------------------------------------
# 1. A log-bilinear model of counts.
# --------------------------------------------------------------------------------------------


def glove_weight(x, x_max: float = X_MAX, gamma: float = GAMMA) -> np.ndarray:
    """Def 1's f(x) = min{1, (x/x_max)^γ}: 0 at x = 0, increasing, and capped at 1 from x_max on."""
    return np.minimum(1.0, (np.asarray(x, dtype=float) / x_max) ** gamma)


def _cells(X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The observed cells Ω of X: rows, columns, and counts, in COO order."""
    X = sparse.coo_matrix(X)
    return X.row, X.col, X.data.astype(float)


def scores_on(W, Cv, b, bt, rows, cols) -> np.ndarray:
    """s_ij = ⟨w_i, c_j⟩ + b_i + b̃_j on the given cells."""
    return np.einsum("nd,nd->n", W[rows], Cv[cols]) + b[rows] + bt[cols]


def glove_loss(W, Cv, b, bt, X, x_max: float = X_MAX, gamma: float = GAMMA):
    """Def 1. J = Σ_Ω f(X_ij)(s_ij − log X_ij)², over the observed cells only (log 0 has no value). Accepts complex
    parameters, for the complex-step check."""
    i, j, x = _cells(X)
    r = scores_on(W, Cv, b, bt, i, j) - np.log(x)
    return np.sum(glove_weight(x, x_max, gamma) * r * r)


def glove_loss_grad(W, Cv, b, bt, X, x_max: float = X_MAX, gamma: float = GAMMA):
    """J and its gradient. With G = 2 f(X)(s − log X) on Ω and 0 off it: ∇_W = GC, ∇_C = GᵀW, ∇_b = G𝟙, ∇_b̃ = Gᵀ𝟙."""
    i, j, x = _cells(X)
    r = scores_on(W, Cv, b, bt, i, j) - np.log(x)
    fw = glove_weight(x, x_max, gamma)
    G = sparse.csr_matrix((2 * fw * r, (i, j)), shape=(W.shape[0], Cv.shape[0]))
    return (float(np.sum(fw * r * r)), G @ Cv, G.T @ W, np.asarray(G.sum(axis=1)).ravel(),
            np.asarray(G.sum(axis=0)).ravel())


def ratio_scores(P_cond: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Prop 1's conclusion: s_ij = log P(j | i) + β_j, for which exp(s_ij − s_kj) = P(j | i)/P(j | k)."""
    return np.log(P_cond) + beta[None, :]


def cauchy_defect(F, u: np.ndarray, v: np.ndarray) -> float:
    """max |F(u − v) − F(u)/F(v)| over a grid: zero for F = exp(λ·), and (Prop 1) for no other continuous F > 0."""
    U, Vv = np.meshgrid(u, v, indexing="ij")
    return float(np.max(np.abs(F(U - Vv) - F(U) / F(Vv))))


def pmi_offsets(b: np.ndarray, bt: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prop 2: for an exact fit s = log X on every cell, PMI − WCᵀ = a𝟙ᵀ + 𝟙βᵀ with a = b − log X_i· + log N and
    β = b̃ − log X_·j. Returns (a, β)."""
    return b - np.log(X.sum(axis=1)) + math.log(X.sum()), bt - np.log(X.sum(axis=0))


# --------------------------------------------------------------------------------------------
# 2. Three weights: Poisson, SGNS, GloVe.
# --------------------------------------------------------------------------------------------


def poisson_excess(s, x) -> np.ndarray:
    """Thm 1. ℓ(s; x) − inf ℓ for the Poisson NLL ℓ(s; x) = e^s − xs + log x!: e^s − x − x(s − log x), which is
    x(e^δ − 1 − δ) with δ = s − log x when x > 0, and e^s (whose infimum 0 is never attained) when x = 0."""
    s, x = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(x, dtype=float))
    pos = x > 0
    delta = s - np.log(np.where(pos, x, 1.0))
    return np.where(pos, x * (np.expm1(delta) - delta), np.exp(s))  # expm1: no cancellation near δ = 0


def bregman_exp(s, s_star) -> np.ndarray:
    """D_A(s, s*) = A(s) − A(s*) − A′(s*)(s − s*) for the Poisson log-partition A = exp."""
    return np.exp(s) - np.exp(s_star) - np.exp(s_star) * (np.asarray(s) - s_star)


def sgns_negative_weight(C, alpha: float = w2v.ALPHA, k: float = w2v.K_NEG) -> np.ndarray:
    """Topic 7's b = k #(w) P_α(c) on every observed cell of C (COO order)."""
    C = sparse.coo_matrix(C)
    _, n_w, pn = ds.marginals(C, alpha)
    return k * n_w[C.row] * pn[C.col]


def frozen_bias_residual(W, Cv, counts: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """Cor 1: SGNS's local coordinate x − x* = w·c − log a + log b, written as GloVe's residual s − log a with the
    biases frozen at b_w = log k#(w) and b̃_c = log P_α(c). Dense, observed cells only (−∞-free when a > 0)."""
    b_w = np.log(k * counts.sum(axis=1))
    bt_c = np.log(w2v.noise_distribution(counts, alpha))
    with np.errstate(divide="ignore"):
        return W @ Cv.T + b_w[:, None] + bt_c[None, :] - np.log(counts)


@lru_cache(maxsize=None)
def weight_shares(benchmark_rows: bool) -> dict[str, list[float]]:
    """Cor 1 on the corpus: on the observed L8 cells (or on the benchmark nouns' rows, as in Topic 7's §5 mechanism),
    the share of each local weight — Poisson's X, SGNS's ab/(a + b) at k = 5 and α = ¾, GloVe's f(X) — in each count
    band, with the share of cells."""
    C = sparse.coo_matrix(ds.cooc(CONTEXT))
    a = C.data.astype(float)
    b = sgns_negative_weight(C)
    keep = np.ones(len(a), dtype=bool)
    if benchmark_rows:
        keep = np.isin(C.row, [ds.index()[w] for w in ds.benchmark_words()])
    band = w2v.band_of(a)
    out = {}
    for name, wt in (("cells", np.ones_like(a)), ("poisson", a), ("sgns", w2v.local_weight(a, b)),
                     ("glove", glove_weight(a))):
        tot = wt[keep].sum()
        out[name] = [float(wt[keep & (band == i)].sum() / tot) for i in range(len(w2v.COUNT_BANDS))]
    return out


# --------------------------------------------------------------------------------------------
# 3. What the loss cannot see.
# --------------------------------------------------------------------------------------------


def double_center(M: np.ndarray) -> np.ndarray:
    """J_n M J_m: M less its row means and column means, plus its grand mean."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def flat_weight_optimum(X: np.ndarray, d: int) -> dict[str, np.ndarray | float]:
    """Thm 2's closed form for f ≡ 1 and X > 0 everywhere: Z = (J_n log X J_m)_d, the score matrix
    S = Z + (log X − J_n log X J_m), the minimum Σ_{i>d} σ_i², and the biases b, b̃ that carry the offsets."""
    L = np.log(X)
    Dc = double_center(L)
    U, s, Vt = np.linalg.svd(Dc, full_matrices=False)
    Z = (U[:, :d] * s[:d]) @ Vt[:d]
    return {"S": Z + (L - Dc), "Z": Z, "loss": float(np.sum(s[d:] ** 2)), "sv": s,
            "W": U[:, :d] * np.sqrt(s[:d]), "C": Vt[:d].T * np.sqrt(s[:d]),
            "b": L.mean(axis=1) - L.mean(), "bt": L.mean(axis=0)}


def binarized(X, kappa: float = 1.0) -> sparse.csr_matrix:
    """X with every observed count set to κ: Prop 3's case of constant counts on the same support."""
    B = sparse.csr_matrix(X, dtype=float, copy=True)
    B.data[:] = kappa
    return B


def impute_zeros(x0: float = 0.5, x_max: float = X_MAX, gamma: float = GAMMA) -> tuple[float, float]:
    """Def 2: an unseen pair counts as seen x₀ times, with GloVe's own weight. Returns (weight f(x₀), target log x₀)."""
    return float(glove_weight(x0, x_max, gamma)), math.log(x0)


# --------------------------------------------------------------------------------------------
# 4. The algorithm: AdaGrad as GloVe runs it, and ALS.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GloVeConfig:
    """Def 3's hyperparameters. docs: the corpus length (300 is Topic 6's text; more continues its generator)."""
    context: str = CONTEXT
    d: int = D
    x_max: float = X_MAX
    gamma: float = GAMMA
    epochs: int = 50
    batch: int = 512
    lr0: float = 0.05
    docs: int = 300
    checkpoints: tuple[int, ...] = (1, 5, 10, 25, 50)


CONTROLLED = GloVeConfig()


@dataclass(frozen=True)
class GloVeModel:
    W: np.ndarray
    C: np.ndarray
    b: np.ndarray
    bt: np.ndarray
    loss: float  # J at the end
    history: tuple[dict, ...]


@lru_cache(maxsize=1)
def _continued_texts() -> tuple[str, ...]:
    """The training corpus's own generator run on to max(SCALE_DOCS) documents, as Topic 3's continuation_run: the
    first 300 are generate_corpus(300, 0) exactly, and the rest continue the same Pitman–Yor restaurants, so Topic 6's
    benchmark nouns keep their sectors. Shorter corpora are prefixes, so each is the same text continued."""
    gen = fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    texts = [gen.doc(i, int(t)).text for i, t in enumerate(tones)]
    more = gen.rng.permutation(np.resize([-1, 0, 1], max(SCALE_DOCS) - 300))
    texts += [gen.doc(300 + i, int(t)).text for i, t in enumerate(more)]
    return tuple(texts)


@lru_cache(maxsize=None)
def continued_sentences(docs: int) -> tuple[tuple[str, ...], ...]:
    """The first `docs` documents of the continued corpus, split and tokenized as Topic 3 does."""
    return tuple(s for t in _continued_texts()[:docs] for s in ng.split_sentences(t))


@lru_cache(maxsize=None)
def counts_for(context: str, docs: int) -> sparse.csr_matrix:
    """The co-occurrence matrix (Topic 6 Def 1) of the first `docs` documents. docs = 300 is Topic 6's matrix itself;
    longer corpora are counted on Topic 6's vocabulary, so ``ds.cooccurrence_of`` drops the new types and a window
    closes over them."""
    return ds.cooc(context) if docs == 300 else ds.cooccurrence_of(continued_sentences(docs), context)


def _segment_sum(idx: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A minibatch's rows summed per distinct index: (the distinct indices, their sums). Repeated rows add, so this is
    a product with the batch's incidence matrix in CSC form (Topic 7's scatter)."""
    u, inv = np.unique(idx, return_inverse=True)
    M = sparse.csc_matrix((np.ones(len(inv)), inv, np.arange(len(inv) + 1)), shape=(len(u), len(inv)))
    return u, M @ vals


@lru_cache(maxsize=None)
def train_glove(cfg: GloVeConfig) -> GloVeModel:
    """Def 3. GloVe's AdaGrad on the observed cells of the counts for cfg: each epoch a fresh permutation of Ω in
    minibatches of cfg.batch, and for each cell the C code's step on ½f(s − log x)² — fdiff = η f (s − log x), the
    update fdiff·(gradient direction)/√gradsq with gradsq starting at 1 and accumulating the squared updates (Duchi,
    Hazan, and Singer 2011). The C code's initialization (U[0, 1] − ½)/d for every vector entry and bias. Minibatch
    summation replaces the C code's per-cell Hogwild updates. One stream (rng [SEED, 800])."""
    X = counts_for(cfg.context, cfg.docs)
    i_all, j_all, x = _cells(X)
    logx, fw = np.log(x), glove_weight(x, cfg.x_max, cfg.gamma)
    V, d, n = X.shape[0], cfg.d, len(x)
    rng = np.random.default_rng([SEED, 800])
    W, Cv = (rng.random((V, d)) - 0.5) / d, (rng.random((V, d)) - 0.5) / d
    b, bt = (rng.random(V) - 0.5) / d, (rng.random(V) - 0.5) / d
    gW, gC, gb, gbt = np.ones((V, d)), np.ones((V, d)), np.ones(V), np.ones(V)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        perm = rng.permutation(n)
        for s in range(0, n, cfg.batch):
            p = perm[s : s + cfg.batch]
            i, j = i_all[p], j_all[p]
            fd = cfg.lr0 * fw[p] * (scores_on(W, Cv, b, bt, i, j) - logx[p])
            ui, sW = _segment_sum(i, fd[:, None] * Cv[j])  # both from the pre-step parameters, as in the C code
            uj, sC = _segment_sum(j, fd[:, None] * W[i])
            _, sb = _segment_sum(i, fd)
            _, sbt = _segment_sum(j, fd)
            W[ui] -= sW / np.sqrt(gW[ui])
            gW[ui] += sW**2
            Cv[uj] -= sC / np.sqrt(gC[uj])
            gC[uj] += sC**2
            b[ui] -= sb / np.sqrt(gb[ui])
            gb[ui] += sb**2
            bt[uj] -= sbt / np.sqrt(gbt[uj])
            gbt[uj] += sbt**2
        if epoch in cfg.checkpoints:
            history.append({"epoch": epoch, "loss": float(glove_loss(W, Cv, b, bt, X, cfg.x_max, cfg.gamma))})
    loss = float(glove_loss(W, Cv, b, bt, X, cfg.x_max, cfg.gamma))
    assert math.isfinite(loss), f"AdaGrad diverged for {cfg}"
    return GloVeModel(W, Cv, b, bt, loss, tuple(history))


def _als_objective(W, Cv, b, bt, i, j, x, fw, zeros, ridge) -> float:
    """ALS's objective: J (plus, with zeros = (w₀, t₀), w₀ Σ_{Ω^c} (s − t₀)², from the Gram matrices of the augmented
    factors so that no V × V matrix is formed) plus ridge (‖W‖² + ‖C‖²)."""
    s = scores_on(W, Cv, b, bt, i, j)
    J = float(np.sum(fw * (s - np.log(x)) ** 2))
    if zeros is not None:
        w0, t0 = zeros
        Wb = np.column_stack([W, b, np.ones(len(b))])
        Cb = np.column_stack([Cv, np.ones(len(bt)), bt])
        total = np.sum((Wb.T @ Wb) * (Cb.T @ Cb)) - 2 * t0 * (Wb.sum(axis=0) @ Cb.sum(axis=0)) + t0**2 * len(b) * len(bt)
        J += w0 * float(total - np.sum((s - t0) ** 2))
    return J + ridge * float(np.sum(W**2) + np.sum(Cv**2))


def als_glove(X, d: int, *, x_max: float = X_MAX, gamma: float = GAMMA, iters: int = 40, ridge: float = RIDGE,
              zeros: tuple[float, float] | None = None, seed: int = 801) -> GloVeModel:
    """GloVe by alternating least squares, an exact block descent. With (C, b̃) fixed, each row's (w_i, b_i) minimizes
    Σ_j ω_ij((w_i, b_i)·(c_j, 1) − (y_ij − b̃_j))² + λ‖w_i‖², a weighted ridge regression solved by its normal
    equations; then symmetrically for the contexts. ω = f(X) and y = log X on Ω. zeros = (w₀, t₀) adds every
    unobserved cell with weight w₀ and target t₀ (Def 2) through Σ_j ω_ij c̄_jc̄_jᵀ = w₀C̄ᵀC̄ +
    Σ_{j∈Ω_i}(f_ij − w₀)c̄_jc̄_jᵀ (Hu, Koren, and Volinsky 2008). Records the objective after every sweep; each
    half-step is an exact minimization, so it never increases."""
    X = sparse.csr_matrix(X)
    i, j, x = _cells(X)
    V, m = X.shape
    w0, t0 = zeros if zeros is not None else (0.0, 0.0)
    fw = glove_weight(x, x_max, gamma)
    F = sparse.csr_matrix((fw - w0, (i, j)), shape=(V, m))
    FY = sparse.csr_matrix((fw * np.log(x) - w0 * t0, (i, j)), shape=(V, m))
    Ft, FYt = F.T.tocsr(), FY.T.tocsr()
    rng = np.random.default_rng([SEED, seed])
    W, Cv = (rng.random((V, d)) - 0.5) / max(d, 1), (rng.random((m, d)) - 0.5) / max(d, 1)
    b, bt = np.zeros(V), np.zeros(m)
    reg = ridge * np.eye(d + 1)
    reg[d, d] = 0.0  # biases unpenalized

    def half(other, off, F_, FY_):
        Ob = np.column_stack([other, np.ones(len(other))])
        G = (F_ @ np.einsum("vi,vj->vij", Ob, Ob).reshape(len(Ob), -1)).reshape(-1, d + 1, d + 1) + reg
        rhs = FY_ @ Ob - F_ @ (off[:, None] * Ob)
        if w0:
            G = G + w0 * (Ob.T @ Ob)
            rhs = rhs + w0 * (t0 * Ob.sum(axis=0) - off @ Ob)
        th = np.linalg.solve(G, rhs[..., None])[..., 0]
        return th[:, :d], th[:, d]

    history = []
    for t in range(1, iters + 1):
        W, b = half(Cv, bt, F, FY)
        Cv, bt = half(W, b, Ft, FYt)
        history.append({"iter": t, "objective": _als_objective(W, Cv, b, bt, i, j, x, fw, zeros, ridge)})
    return GloVeModel(W, Cv, b, bt, float(glove_loss(W, Cv, b, bt, X, x_max, gamma)), tuple(history))


@lru_cache(maxsize=None)
def als_model(docs: int, imputed: bool) -> GloVeModel:
    """The cached ALS fits the page uses: the controlled counts (or a longer corpus's), with or without Def 2."""
    return als_glove(counts_for(CONTEXT, docs), D, zeros=impute_zeros() if imputed else None)


def glove_vectors(m: GloVeModel, rep: str = "W+C") -> np.ndarray:
    """The reported vectors: W + C (Pennington et al.'s convention, primary here) or W."""
    return m.W + m.C if rep == "W+C" else m.W


# --------------------------------------------------------------------------------------------
# 5. Gauge and analogies.
# --------------------------------------------------------------------------------------------


def affine_gauge(W, Cv, b, bt, A, t, s, beta: float) -> tuple[np.ndarray, ...]:
    """Prop 4 (i), in row-vector form: w′ = wA + t, c′ = cA⁻ᵀ + s, b′ = b − ⟨wA, s⟩ − ⟨t, s⟩ + β,
    b̃′ = b̃ − ⟨cA⁻ᵀ, t⟩ − β. Every score s_ij, and so J, is unchanged."""
    WA, CA = W @ A, Cv @ np.linalg.inv(A).T
    return WA + t, CA + s, b - WA @ s - t @ s + beta, bt - CA @ t - beta


def recover_affine_gauge(p1, p2) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Prop 4 (ii): from two parameter sets (W, C, b, b̃) with the same scores and full-column-rank augmented word
    matrices [W, b, 𝟙], the unique G with [W₂, b₂, 𝟙] = [W₁, b₁, 𝟙]G, read as (A, t, s, β)."""
    W1, _, b1, _ = p1
    W2, _, b2, _ = p2
    d = W1.shape[1]
    G = np.linalg.lstsq(np.column_stack([W1, b1, np.ones(len(b1))]),
                        np.column_stack([W2, b2, np.ones(len(b2))]), rcond=None)[0]
    A, u, t = G[:d, :d], G[:d, d], G[d + 1, :d]
    s = -np.linalg.solve(A, u)
    return A, t, s, float(G[d + 1, d] + t @ s)


def balance_and_centering(W, Cv) -> dict[str, float]:
    """Prop 5's diagnostics: Topic 7's imbalance ‖WᵀW − CᵀC‖/‖WᵀW‖, and ‖mean row‖ / mean ‖row‖ for W and for C."""
    def centering(M):
        return float(np.linalg.norm(M.mean(axis=0)) / np.linalg.norm(M, axis=1).mean())
    return {"imbalance": w2v.imbalance(W, Cv), "mean_w": centering(W), "mean_c": centering(Cv)}


RELATIONS = ("city_region", "company_sector")
RELATION_LABELS = {"city_region": "city → region", "company_sector": "company → sector"}


@lru_cache(maxsize=1)
def kb_relations() -> dict[str, tuple[tuple[str, str], ...]]:
    """§6's relations from the filings knowledge base, as (head, tail) word pairs: LOCATED_IN between headquarters
    cities and their regions, and IN_SECTOR between each public company (the first token of its short name) and its
    sector. CEO surnames are too rare to use, and several companies share a city."""
    short = {name: alias for alias, name in fc.WORLD.aliases.items()}
    city = tuple((c.lower(), r.lower()) for c, r in fc.WORLD.facts("LOCATED_IN"))
    sector = tuple((short[n].split()[0].lower(), fc.COMPANIES[n][0]) for n in fc.PUBLIC_COMPANIES)
    return {"city_region": city, "company_sector": sector}


@lru_cache(maxsize=None)
def analogy_questions(relation: str) -> np.ndarray:
    """Every a : b :: c : d with (a, b) ≠ (c, d) in the relation, a ≠ c and b ≠ d, as rows of vocabulary indices."""
    idx = ds.index()
    pairs = kb_relations()[relation]
    return np.array([(idx[a], idx[b], idx[c], idx[d]) for a, b in pairs for c, d in pairs if a != c and b != d])


def three_cos_add(U: np.ndarray, a: int, b: int, c: int, candidates=None, exclude: bool = True, top: int = 1
                  ) -> np.ndarray:
    """Def 4 for one question, on unit rows U: the top candidates x by cos(x̂, b̂ − â + ĉ), over all words or the given
    candidates, with a, b, c excluded unless exclude=False."""
    s = U @ (U[b] - U[a] + U[c])
    if candidates is not None:
        keep = np.full(len(s), -np.inf)
        keep[candidates] = s[candidates]
        s = keep
    if exclude:
        s[[a, b, c]] = -np.inf
    return np.argsort(-s, kind="stable")[:top]


def analogy_scores(X: np.ndarray, relation: str) -> dict[str, float]:
    """Def 4 over every question of a relation: open-vocabulary and within-category 3CosAdd accuracy, the share of
    questions whose top candidate before the exclusion is b itself (Linzen 2016), and within-category chance."""
    Q = analogy_questions(relation)
    U = ds.unit_rows(X, np.arange(X.shape[0]))
    S = (U[Q[:, 1]] - U[Q[:, 0]] + U[Q[:, 2]]) @ U.T
    b_nearest = np.argmax(S, axis=1) == Q[:, 1]
    rows = np.arange(len(Q))
    for col in range(3):
        S[rows, Q[:, col]] = -np.inf
    tails = np.unique([ds.index()[t] for _, t in kb_relations()[relation]])
    St = np.full_like(S, -np.inf)
    St[:, tails] = S[:, tails]
    return {"open": float(np.mean(np.argmax(S, axis=1) == Q[:, 3])),
            "within": float(np.mean(np.argmax(St, axis=1) == Q[:, 3])),
            "b_nearest": float(np.mean(b_nearest)), "n": len(Q), "chance_within": 1 / (len(tails) - 1)}


def parallelogram_defect(S_rows: np.ndarray) -> float:
    """Thm 3's diagnostic on fitted score rows ordered (a, b, c, d): sd_j(Δ_j)/sd_j(s_bj − s_aj), with
    Δ = s_b − s_a − s_d + s_c. It is 0 iff w_b − w_a = w_d − w_c (given Thm 3's rank condition), and it depends on the
    scores only, so on no factorization of them."""
    a, b, c, d = S_rows
    return float(np.std(b - a - d + c) / np.std(b - a))


def score_rows(params, rows) -> np.ndarray:
    """Fitted score rows: s = WCᵀ (+ b𝟙ᵀ + 𝟙b̃ᵀ for GloVe), for (W, C) or (W, C, b, b̃)."""
    W, Cv = params[0], params[1]
    S = W[rows] @ Cv.T
    if len(params) == 4:
        S = S + params[2][rows, None] + params[3][None, :]
    return S


def median_defect(params, relation: str) -> float:
    return float(np.median([parallelogram_defect(score_rows(params, q)) for q in analogy_questions(relation)]))


def gauge_free_scores(params, relation: str) -> dict[str, float]:
    """Thm 3 as a solver: answer a : b :: c : ? with the tail d′ (other than b) whose score rows come closest to a
    parallelogram, argmin sd_j(s_b − s_a − s_d′ + s_c). It reads the scores only, so every factorization of them gives
    the same answers. Returns its accuracy, and the median defect at the right tail and at the wrong ones."""
    Q = analogy_questions(relation)
    tails = np.unique([ds.index()[t] for _, t in kb_relations()[relation]])
    rows = np.unique(np.concatenate([Q[:, :3].ravel(), tails]))
    pos = {r: k for k, r in enumerate(rows)}
    S = score_rows(params, rows)
    St = S[[pos[t] for t in tails]]
    base = S[[pos[r] for r in Q[:, 1]]] - S[[pos[r] for r in Q[:, 0]]] + S[[pos[r] for r in Q[:, 2]]]
    sd = np.std(base[:, None, :] - St[None, :, :], axis=2)  # questions × tails
    sd[Q[:, 1][:, None] == tails[None, :]] = np.inf  # b itself is excluded
    right = Q[:, 3][:, None] == tails[None, :]
    norm = np.std(S[[pos[r] for r in Q[:, 1]]] - S[[pos[r] for r in Q[:, 0]]], axis=1)[:, None]
    wrong = ~right & np.isfinite(sd)
    return {"within": float(np.mean(tails[np.argmin(sd, axis=1)] == Q[:, 3])),
            "defect_right": float(np.median((sd / norm)[right])), "defect_wrong": float(np.median((sd / norm)[wrong]))}


# --------------------------------------------------------------------------------------------
# Corpus summaries: §3's benchmark and scale, §5's gauge table, §6's finance case study.
# --------------------------------------------------------------------------------------------


def _chance_nn(words) -> float:
    lab = ds.planted_labels()
    share = np.unique([lab[w] for w in words], return_counts=True)[1] / len(words)
    return float(np.sum(share**2))


@lru_cache(maxsize=1)
def benchmark_summary() -> dict:
    """§3's controlled comparison on Topic 6's benchmark (L8, d = 25): GloVe (W + C and W, by AdaGrad; W + C by ALS)
    against Topic 7's controlled SGNS and Topic 6's SVD(SPPMI₅), p = 0, with the jackknife standard error of GloVe's
    test AUC and of the differences (Topic 7's paired statistics); the sparsity of the counts and how much of log X
    on Ω the biases alone explain."""
    test = list(ds.benchmark_split()[1])
    lab = ds.planted_labels()
    g = train_glove(CONTROLLED)
    Xg, sgns, svd = glove_vectors(g), w2v.sgns_vectors(w2v.CONTROLLED), ds.vectors((CONTEXT, "sppmi5", D, 0.0))
    out: dict = {}
    for name, X in (("glove", Xg), ("glove_W", g.W), ("glove_als", glove_vectors(als_model(300, False))),
                    ("sgns", sgns), ("svd", svd)):
        out.update({f"{name}_{k}": v for k, v in w2v._scores(X).items()})
    out["glove_auc_se"] = ds.jackknife_se(lambda ws: ds.pair_auc(Xg, ws, lab), test)
    out.update({f"glove_vs_sgns_{k}": v for k, v in w2v._paired(Xg, sgns, test).items()})
    out.update({f"glove_vs_svd_{k}": v for k, v in w2v._paired(Xg, svd, test).items()})
    out["chance_nn"] = _chance_nn(test)
    X = counts_for(CONTEXT, 300)
    out["singleton_share"] = float(np.mean(X.data == 1))
    out["le2_share"] = float(np.mean(X.data <= 2))
    _, _, x = _cells(X)
    fw, y = glove_weight(x), np.log(x)
    J_const = float(np.sum(fw * (y - np.sum(fw * y) / np.sum(fw)) ** 2))
    J_bias = als_glove(X, 0, ridge=0.0, iters=60).loss  # d = 0: the biases alone, a weighted two-way fit
    out["additive_share"] = 1 - J_bias / J_const
    out["vector_share"] = (J_bias - als_model(300, False).loss) / J_const
    return out


def _svd_vectors(docs: int) -> np.ndarray:
    if docs == 300:
        return ds.vectors((CONTEXT, "sppmi5", D, 0.0))
    U, _, _ = ds.truncated_svd(ds.ppmi(counts_for(CONTEXT, docs), w2v.ALPHA, w2v.K_NEG), D,
                               np.random.default_rng([SEED, 814]))
    return U


@lru_cache(maxsize=1)
def scale_summary() -> dict:
    """§3's scale experiment: for each corpus length, the singleton share of the observed cells, and the test AUC and
    1-NN of GloVe and of GloVe with imputed zeros (Def 2), both by ALS (40 sweeps; AdaGrad at 3,000 documents would
    cost the harness's budget), and of SVD(SPPMI₅) at d = 25, p = 0."""
    out: dict = {}
    for docs in SCALE_DOCS:
        X = counts_for(CONTEXT, docs)
        out[f"N_{docs}"], out[f"nnz_{docs}"] = float(X.sum()), X.nnz
        out[f"singleton_share_{docs}"] = float(np.mean(X.data == 1))
        for name, Xv in (("glove", glove_vectors(als_model(docs, False))),
                         ("imputed", glove_vectors(als_model(docs, True))), ("svd", _svd_vectors(docs))):
            sc = w2v._scores(Xv)
            out[f"{name}_test_auc_{docs}"], out[f"{name}_test_nn_{docs}"] = sc["test_auc"], sc["test_nn"]
    return out


GAUGE_KAPPA = 100.0


@lru_cache(maxsize=1)
def gauge_variants() -> dict[str, dict[str, tuple[np.ndarray, ...]]]:
    """Factorizations of one set of scores. SGNS (Topic 7's controlled model, vectors W): as trained, Topic 6's
    p-family of its product, and Topic 7's own random κ = 100 gauge. GloVe (the controlled model, vectors W + C): as
    trained, balanced (p = ½), centered and translated (Prop 4's t, s, with the biases adjusted), and a random
    κ = 100 A. SGNS has no biases, so a translation would change its scores; only GloVe's biases absorb one."""
    m = w2v.train_sgns(w2v.CONTROLLED)
    sg = {"trained": (m.W, m.C)}
    for q, name in ((0.0, "p0"), (0.5, "balanced"), (1.0, "p1")):
        sg[name] = w2v.refactor(m.W, m.C, q)
    sg["random"] = w2v.random_gauge(m.W, m.C, GAUGE_KAPPA, np.random.default_rng([w2v.SEED, 730]))
    g = train_glove(CONTROLLED)
    I = np.eye(D)
    gl = {"trained": (g.W, g.C, g.b, g.bt), "balanced": (*w2v.refactor(g.W, g.C, 0.5), g.b, g.bt),
          "centered": affine_gauge(g.W, g.C, g.b, g.bt, I, -g.W.mean(axis=0), -g.C.mean(axis=0), 0.0)}
    rng = np.random.default_rng([SEED, 815])
    u = rng.normal(size=D)
    u *= np.linalg.norm(g.W, axis=1).mean() / np.linalg.norm(u)
    gl["translated"] = affine_gauge(g.W, g.C, g.b, g.bt, I, u, u, 0.0)
    gl["random"] = (*w2v.random_gauge(g.W, g.C, GAUGE_KAPPA, np.random.default_rng([SEED, 813])), g.b, g.bt)
    return {"sgns": sg, "glove": gl}


def variant_vectors(method: str, params) -> np.ndarray:
    return params[0] if method == "sgns" else params[0] + params[1]


@lru_cache(maxsize=1)
def gauge_table() -> dict:
    """Cor 2 on the corpus: for each variant, the largest change in the scores (relative; zero to rounding), the
    benchmark's test AUC, each relation's 3CosAdd accuracy (open and within), and Thm 3's median defect, which
    depends on the scores alone."""
    X = counts_for(CONTEXT, 300)
    out: dict = {}
    for method, variants in gauge_variants().items():
        base = variants["trained"]
        S0 = base[0] @ base[1].T
        for name, params in variants.items():
            S = params[0] @ params[1].T
            if method == "glove":  # compare full scores, biases included, on Ω and through J
                i, j, _ = _cells(X)
                ds_ = scores_on(*params, i, j) - scores_on(*base, i, j)
                change = float(np.abs(ds_).max() / np.abs(scores_on(*base, i, j)).max())
                out[f"{method}|{name}|loss"] = float(glove_loss(*params, X))
            else:
                change = float(np.linalg.norm(S - S0) / np.linalg.norm(S0))
            vec = variant_vectors(method, params)
            out[f"{method}|{name}|score_change"] = change
            out[f"{method}|{name}|test_auc"] = w2v._scores(vec)["test_auc"]
            for rel in RELATIONS:
                sc = analogy_scores(vec, rel)
                out[f"{method}|{name}|{rel}|open"], out[f"{method}|{name}|{rel}|within"] = sc["open"], sc["within"]
                out[f"{method}|{name}|{rel}|defect"] = median_defect(params, rel)
                out[f"{method}|{name}|{rel}|gauge_free"] = gauge_free_scores(params, rel)["within"]
    return out


@lru_cache(maxsize=1)
def finance_summary() -> dict:
    """§6: each relation's 3CosAdd accuracy (open, within, b nearest) for GloVe (W + C and W), SGNS (W and W + C), and
    Topic 6's SVD(SPPMI₅) at p = 0 and ½; the gauge-free solver on each model's fitted scores, with the median defect
    at the right and the wrong tails; the relation words' smallest count."""
    g = train_glove(CONTROLLED)
    m = w2v.train_sgns(w2v.CONTROLLED)
    U1 = ds.embedding(CONTEXT, "sppmi5", D, 1.0)
    V0 = ds.context_embedding(CONTEXT, "sppmi5", D, 1.0)  # W₁C₁ᵀ = M_d, the SVD's fitted matrix
    methods = {"glove": glove_vectors(g), "glove_W": g.W, "sgns": m.W, "sgns_wc": m.W + m.C,
               "svd_p0": ds.vectors((CONTEXT, "sppmi5", D, 0.0)), "svd_half": ds.vectors((CONTEXT, "sppmi5", D, 0.5))}
    out: dict = {}
    cnt = ds.unigram_counts()
    for rel in RELATIONS:
        words = sorted({w for p in kb_relations()[rel] for w in p})
        out[f"{rel}|min_count"] = float(min(cnt[ds.index()[w]] for w in words))
        for name, X in methods.items():
            for k, v in analogy_scores(X, rel).items():
                out[f"{rel}|{name}|{k}"] = v
        for name, params in (("glove", (g.W, g.C, g.b, g.bt)), ("sgns", (m.W, m.C)), ("svd", (U1, V0))):
            for k, v in gauge_free_scores(params, rel).items():
                out[f"{rel}|{name}|gauge_free_{k}"] = v
    return out


# --------------------------------------------------------------------------------------------
# The lab's data.
# --------------------------------------------------------------------------------------------

LAB_QUESTIONS = {"city_region": (("houston", "texas", "boston"), ("chicago", "illinois", "dublin"),
                                 ("denver", "colorado", "toronto"), ("columbus", "ohio", "frankfurt")),
                 "company_sector": (("halvorsen", "industrials", "pellucid"), ("saltmarsh", "banking", "northgate"),
                                    ("tamsin", "retail", "orrin"), ("brightwater", "technology", "kingsbridge"))}
LAB_VARIANTS = {"glove": ("trained", "balanced", "centered", "random"), "sgns": ("trained", "balanced", "random")}


def _r(x: float, nd: int = 3):
    return ds._r(x, nd)


def _projection(X: np.ndarray, words) -> list[list[float]]:
    """2-D coordinates of the words' unit vectors on their own top two principal axes (the plane 3CosAdd's offsets are
    drawn in)."""
    Y = ds.unit_rows(X, [ds.index()[w] for w in words])
    Y = Y - Y.mean(axis=0)
    _, _, Vt = np.linalg.svd(Y, full_matrices=False)
    P = Y @ Vt[:2].T
    return [[_r(v, 2) for v in row] for row in P]


@lru_cache(maxsize=1)
def lab_cells() -> dict:
    """Panel A's cells: Topic 7's *revenue* row at L8 (its most associated context in each count band, and the most
    frequent context it never meets), with a, SGNS's b at k = 5 and α = ¾, log a, the trained GloVe score s, and the
    trained SGNS score; plus a histogram of the observed counts so the page can recompute GloVe's band shares for any
    x_max ≤ 100 and γ."""
    C = ds.cooc(CONTEXT)
    g = train_glove(CONTROLLED)
    i = ds.index()[w2v.LAB_ROW]
    cells = []
    for c in w2v.lab_pairs():
        j = ds.index()[c["context"]]
        b = w2v.K_NEG * c["n_w"] * c["p_noise"][str(w2v.ALPHA)]
        s = float(g.W[i] @ g.C[j] + g.b[i] + g.bt[j])
        cells.append({"context": c["context"], "a": c["a"], "b": _r(b), "log_a": _r(math.log(c["a"])) if c["a"] else None,
                      "glove_s": _r(s), "sgns_x": c["trained"],
                      "x_star": c["x_star"][f"{w2v.ALPHA}|{w2v.K_NEG}"]})
    data = sparse.coo_matrix(C).data
    hist = [int(np.sum(data == a)) for a in range(1, 100)] + [int(np.sum(data >= 100))]
    sh_all = weight_shares(False)
    return {"row": w2v.LAB_ROW, "cells": cells, "hist": hist, "bands": list(w2v.BAND_NAMES),
            "share": {k: [_r(v) for v in sh_all[k]] for k in ("cells", "poisson", "sgns", "glove")}}


@lru_cache(maxsize=1)
def lab_analogies() -> dict:
    """Panel B's analogy view: for each relation, its words and pairs; for each method and variant, a 2-D projection
    of the relation's words, the relation's accuracies, and the top three open-vocabulary answers to four curated
    questions."""
    vocab, idx = ds.vocab(), ds.index()
    gv = gauge_variants()
    gt = gauge_table()
    out: dict = {}
    for rel in RELATIONS:
        pairs = kb_relations()[rel]
        words = [w for p in pairs for w in p]
        words = list(dict.fromkeys(words))
        cell: dict = {"label": RELATION_LABELS[rel], "words": words, "pairs": [list(p) for p in pairs],
                      "questions": [list(q) for q in LAB_QUESTIONS[rel]], "methods": {}}
        for method, names in LAB_VARIANTS.items():
            cell["methods"][method] = {}
            for name in names:
                X = variant_vectors(method, gv[method][name])
                U = ds.unit_rows(X, np.arange(X.shape[0]))
                sc = analogy_scores(X, rel)
                cell["methods"][method][name] = {
                    "xy": _projection(X, words), "open": _r(sc["open"]), "within": _r(sc["within"]),
                    "b_nearest": _r(sc["b_nearest"]), "defect": _r(gt[f"{method}|{name}|{rel}|defect"]),
                    "gauge_free": _r(gt[f"{method}|{name}|{rel}|gauge_free"]),
                    "answers": [[vocab[k] for k in three_cos_add(U, idx[a], idx[b], idx[c], top=3)]
                                for a, b, c in LAB_QUESTIONS[rel]]}
        out[rel] = cell
    return out


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    """Panel A: lab_cells(). Panel B: lab_analogies() and the scale curve."""
    sc = scale_summary()
    scale = {"docs": list(SCALE_DOCS),
             **{k: [_r(sc[f"{k}_{n}"]) for n in SCALE_DOCS]
                for k in ("singleton_share", "glove_test_auc", "imputed_test_auc", "svd_test_auc")}}
    return {"context": CONTEXT, "d": D, "x_max": X_MAX, "gamma": GAMMA, "k": w2v.K_NEG, "alpha": w2v.ALPHA,
            "weights": lab_cells(), "analogies": lab_analogies(), "scale": scale,
            "chance_auc": 0.5}


# --------------------------------------------------------------------------------------------
# Summary — the numbers the page quotes.
# --------------------------------------------------------------------------------------------


def summary() -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in benchmark_summary().items():
        out[f"bench_{k}"] = v
    for h in train_glove(CONTROLLED).history:
        out[f"adagrad_loss_epoch{h['epoch']}"] = h["loss"]
    out["adagrad_loss"] = train_glove(CONTROLLED).loss
    out["als_loss"] = als_model(300, False).loss
    out["weight_sum_omega"] = float(np.sum(glove_weight(_cells(counts_for(CONTEXT, 300))[2])))
    for rows in (False, True):
        sh = weight_shares(rows)
        tag = "bench" if rows else "all"
        for name in ("cells", "poisson", "sgns", "glove"):
            out[f"share_{tag}_{name}_rare"] = sh[name][0] + sh[name][1]
    for k, v in balance_and_centering(train_glove(CONTROLLED).W, train_glove(CONTROLLED).C).items():
        out[f"adagrad_{k}"] = v
    for k, v in balance_and_centering(als_model(300, False).W, als_model(300, False).C).items():
        out[f"als_{k}"] = v
    out.update({f"scale_{k}": v for k, v in scale_summary().items()})
    out.update({f"gauge_{k}": v for k, v in gauge_table().items()})
    out.update({f"fin_{k}": v for k, v in finance_summary().items()})
    return out


# --------------------------------------------------------------------------------------------
# Verification harness — every claim the page makes.
# --------------------------------------------------------------------------------------------


def _small_positive(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    return w2v._small_instance(rng, n, m) + 1.0


def test_continued_corpus():
    """§3's longer corpora continue Topic 6's text: their first 300 documents are Topic 6's sentences exactly, and the
    300-document counts are Topic 6's L8 matrix."""
    assert continued_sentences(300) == ds.sentences()
    assert (counts_for(CONTEXT, 300) != ds.cooc(CONTEXT)).nnz == 0
    assert counts_for(CONTEXT, 1200).sum() > counts_for(CONTEXT, 300).sum()


def test_ratio_argument():
    """Prop 1: F = exp(λ·) satisfies F(u − v) = F(u)/F(v), continuous non-exponential F do not; vectors whose scores
    are log P(j | i) + β_j satisfy the ratio identity exp⟨w_i − w_k, c_j⟩ = P(j | i)/P(j | k) on random joints; and
    after the symmetrization, scores equal to log X give J = 0."""
    u = np.linspace(-3, 3, 61)
    for lam in (-1.3, 0.4, 1.0):
        assert cauchy_defect(lambda z: np.exp(lam * z), u, u) < 1e-9 * np.exp(6 * abs(lam))
    for F in (lambda z: 1 + z**2, lambda z: np.exp(z) + 0.1, lambda z: 1 / (1 + np.exp(-z))):
        assert cauchy_defect(F, u, u) > 1e-2
    rng = np.random.default_rng([SEED, 802])
    for _ in range(10):
        counts = _small_positive(rng, 6, 7)
        P = counts / counts.sum(axis=1, keepdims=True)
        S = ratio_scores(P, rng.normal(size=7))
        U_, s_, Vt_ = np.linalg.svd(S, full_matrices=False)
        W, Cv = U_ * s_, Vt_.T  # an exact factorization WCᵀ = S
        for i in range(6):
            for k in range(6):
                assert np.abs(np.exp((W[i] - W[k]) @ Cv.T) - P[i] / P[k]).max() < 1e-10 * (P[i] / P[k]).max()
        L = np.log(counts)
        opt = flat_weight_optimum(counts, min(counts.shape) - 1)  # full rank after double-centering
        assert glove_loss(opt["W"], opt["C"], opt["b"], opt["bt"], sparse.csr_matrix(counts), x_max=1e-9) < 1e-18 * np.sum(L**2)


def test_log_bilinear_rank():
    """Prop 2: positive tables X = exp(WCᵀ + b𝟙ᵀ + 𝟙b̃ᵀ) with d ∈ {1, 2, 4} have PMI = WCᵀ + a𝟙ᵀ + 𝟙βᵀ with the stated
    offsets (to 10⁻¹⁰), so rank PMI ≤ d + 2, while exp(PMI) is full rank; α = ¾ moves only the column offset."""
    rng = np.random.default_rng([SEED, 803])
    for d in (1, 2, 4):
        W, Cv = rng.normal(0, 0.6, (14, d)), rng.normal(0, 0.6, (12, d))
        b, bt = rng.normal(size=14), rng.normal(size=12)
        X = np.exp(W @ Cv.T + b[:, None] + bt[None, :])
        P = w2v.pmi_dense(X)
        a, beta = pmi_offsets(b, bt, X)
        assert np.abs(P - (W @ Cv.T + a[:, None] + beta[None, :])).max() < 1e-10
        assert ds.numerical_rank(P) <= d + 2 < ds.numerical_rank(np.exp(P))
        pa = w2v.noise_distribution(X, 0.75)
        Pa = np.log(X * X.sum()) - np.log(X.sum(axis=1))[:, None] - np.log(pa * X.sum())[None, :]
        Dm = Pa - P
        assert np.abs(Dm - Dm[:1]).max() < 1e-10  # a column offset only


def test_glove_gradient():
    """Def 1's hand-written gradient against Topic 5's complex step (to 10⁻¹⁰ relative) and central differences
    (10⁻⁶), on a random instance with unobserved cells."""
    rng = np.random.default_rng([SEED, 808])
    X = sparse.csr_matrix(w2v._small_instance(rng, 7, 6, zero_share=0.4))
    W, Cv = rng.normal(0, 0.5, (7, 3)), rng.normal(0, 0.5, (6, 3))
    b, bt = rng.normal(0, 0.5, 7), rng.normal(0, 0.5, 6)
    _, gW, gC, gb, gbt = glove_loss_grad(W, Cv, b, bt, X, x_max=5.0)
    fns = ((gW, lambda T: glove_loss(T, Cv, b, bt, X, x_max=5.0), W),
           (gC, lambda T: glove_loss(W, T, b, bt, X, x_max=5.0), Cv),
           (gb, lambda T: glove_loss(W, Cv, T, bt, X, x_max=5.0), b),
           (gbt, lambda T: glove_loss(W, Cv, b, T, X, x_max=5.0), bt))
    for g, f, th in fns:
        assert np.abs(nlm.complex_step_grad(f, th) - g).max() <= 1e-10 * np.abs(g).max()
        assert np.abs(nlm.central_diff_grad(f, th) - g).max() <= 1e-6 * np.abs(g).max()


def test_poisson_bregman():
    """Thm 1: (i) the excess Poisson NLL is x(e^δ − 1 − δ) = D_exp(s, log x) to 10⁻¹² relative and matches
    scipy.stats.poisson; it lies between ½xδ²e^{min(δ,0)} and ½xδ²e^{max(δ,0)}; its curvature at log x is x.
    (ii) For x = 0 it is e^s, strictly decreasing. (iii) On random instances near the optimum, the exact excess and
    the quadratic ½Σ X δ² differ by at most Σ X|δ|³."""
    xs = np.array([0.5, 1.0, 3.0, 40.0, 250.0])
    deltas = np.linspace(-3, 3, 121)
    for x in xs:
        s = math.log(x) + deltas
        ex = poisson_excess(s, x)
        direct = np.exp(s) - x - x * s + xlogy(x, x)  # ℓ(s; x) − ℓ(log x; x), term by term
        scale = np.exp(s) + x + x * np.abs(s) + abs(xlogy(x, x))  # its rounding scale
        assert np.all(np.abs(ex - direct) <= 1e-14 * scale)
        assert np.all(np.abs(bregman_exp(s, math.log(x)) - ex) <= 1e-14 * scale)
        lo, hi = 0.5 * x * deltas**2 * np.exp(np.minimum(deltas, 0)), 0.5 * x * deltas**2 * np.exp(np.maximum(deltas, 0))
        assert np.all(lo <= ex * (1 + 1e-12) + 1e-15) and np.all(ex <= hi * (1 + 1e-12) + 1e-15)
        h = 1e-4
        curv = (poisson_excess(math.log(x) + h, x) - 2 * poisson_excess(math.log(x), x) + poisson_excess(math.log(x) - h, x)) / h**2
        assert abs(curv - x) < 1e-5 * max(1.0, x)
    for k in (0, 1, 4, 17):  # the full NLL against scipy, at integer counts
        s = np.linspace(-2, 4, 13)
        nll = np.exp(s) - k * s + math.lgamma(k + 1)
        assert np.abs(nll + stats.poisson.logpmf(k, np.exp(s))).max() < 1e-10 * np.abs(nll).max()
    s = np.linspace(-30, 5, 351)
    e0 = poisson_excess(s, 0.0)
    assert np.all(np.diff(e0) > 0) and np.all(e0 > 0)
    rng = np.random.default_rng([SEED, 804])
    for _ in range(10):
        X = _small_positive(rng, 6, 5)
        dl = rng.uniform(-0.05, 0.05, X.shape)
        exact = np.sum(poisson_excess(np.log(X) + dl, X))
        assert abs(exact - 0.5 * np.sum(X * dl**2)) <= np.sum(X * np.abs(dl) ** 3)


def test_three_weights():
    """Cor 1: SGNS's local coordinate is GloVe's residual with frozen biases (to 10⁻¹²), and near x* the SGNS cell
    loss is ½·ab/(a + b)·(that residual)² to third order; ab/(a + b) ≤ min(a, b) ≤ a (Poisson's weight); f ≤ 1 and
    f(X)/X decreases below x_max, so GloVe's share of weight on rare pairs exceeds Poisson's on any corpus. Corpus: on
    the benchmark nouns' rows, SGNS's share on pairs seen ≤ 4 times is Topic 7's 0.548; GloVe puts the most of its
    weight there, then Poisson, then SGNS, all below the pairs' share of cells."""
    rng = np.random.default_rng([SEED, 805])
    for _ in range(10):
        counts = _small_positive(rng, 6, 5)
        W, Cv = rng.normal(0, 0.5, (6, 2)), rng.normal(0, 0.5, (5, 2))
        a, b = counts, w2v.negative_weights(counts, w2v.ALPHA, w2v.K_NEG)
        r = frozen_bias_residual(W, Cv, counts, w2v.ALPHA, w2v.K_NEG)
        assert np.abs(r - (W @ Cv.T - np.log(a / b))).max() < 1e-12
        x_star = np.log(a / b)
        dl = rng.uniform(-0.02, 0.02, a.shape)
        excess = -w2v.cell_objective(x_star + dl, a, b) + w2v.cell_objective(x_star, a, b)
        assert np.abs(excess - 0.5 * w2v.local_weight(a, b) * dl**2).max() <= 0.2 * np.max((a + b) * np.abs(dl) ** 3)
        assert np.all(w2v.local_weight(a, b) <= np.minimum(a, b) + 1e-12)
    xg = np.arange(1, 100)
    assert np.all(glove_weight(xg) <= 1) and np.all(np.diff(glove_weight(xg) / xg) < 0)
    sh = weight_shares(True)
    rare = {k: v[0] + v[1] for k, v in sh.items()}
    assert abs(rare["sgns"] - 0.548) < 5e-4, rare
    assert rare["sgns"] < rare["poisson"] < rare["glove"] < rare["cells"], rare


def test_flat_weight_closed_form():
    """Thm 2 on random full-support instances: J_n log X J_m = J_n PMI J_m (to 10⁻¹²); L-BFGS on J (f ≡ 1) from a
    random start and ALS without ridge both reach the closed form's minimum (to 10⁻⁸ relative) and its score matrix
    (to 10⁻⁶)."""
    rng = np.random.default_rng([SEED, 806])
    for _ in range(10):
        n, m = int(rng.integers(6, 13)), int(rng.integers(5, 11))
        d = int(rng.integers(1, 4))
        X = _small_positive(rng, n, m)
        L = np.log(X)
        assert np.abs(double_center(L) - double_center(w2v.pmi_dense(X))).max() < 1e-12
        opt = flat_weight_optimum(X, d)
        sizes = (n * d, m * d, n, m)

        def fun(th):
            W, Cv, b, bt = np.split(th, np.cumsum(sizes)[:-1])
            W, Cv = W.reshape(n, d), Cv.reshape(m, d)
            R = W @ Cv.T + b[:, None] + bt[None, :] - L
            return float(np.sum(R**2)), np.concatenate([(2 * R @ Cv).ravel(), (2 * R.T @ W).ravel(),
                                                        2 * R.sum(axis=1), 2 * R.sum(axis=0)])
        th = w2v._lbfgs(fun, rng.normal(0, 0.3, sum(sizes)))
        W, Cv, b, bt = np.split(th, np.cumsum(sizes)[:-1])
        W, Cv = W.reshape(n, d), Cv.reshape(m, d)
        S = W @ Cv.T + b[:, None] + bt[None, :]
        assert abs(fun(th)[0] - opt["loss"]) <= 1e-8 * max(opt["loss"], 1e-12) + 1e-14
        assert np.abs(S - opt["S"]).max() < 1e-6
        als = als_glove(sparse.csr_matrix(X), d, x_max=1e-9, ridge=0.0, iters=400, seed=816)
        S2 = als.W @ als.C.T + als.b[:, None] + als.bt[None, :]
        assert abs(als.loss - opt["loss"]) <= 1e-8 * max(opt["loss"], 1e-12) + 1e-14
        assert np.abs(S2 - opt["S"]).max() < 1e-6


def test_support_blindness():
    """Prop 3: with log X additive on the support, (0, 0, u, v) attains J = 0 for any support; ridge ALS goes there
    from a random start. Corpus: on the L8 support with every count set to e, ALS's vectors are exactly zero and the
    benchmark is exactly at chance."""
    rng = np.random.default_rng([SEED, 807])
    for _ in range(6):
        u, v = rng.normal(size=9), rng.normal(size=8)
        for zero_share in (0.3, 0.6):
            mask = rng.random((9, 8)) > zero_share
            mask[:, 0] = mask[0, :] = True
            X = sparse.csr_matrix(np.where(mask, np.exp(u[:, None] + v[None, :]), 0.0))
            assert glove_loss(np.zeros((9, 3)), np.zeros((8, 3)), u, v, X) < 1e-24
            m = als_glove(X, 3, ridge=1e-2, iters=300, seed=817)
            i, j, _ = _cells(X)
            assert m.loss < 1e-12 and np.abs(np.einsum("nd,nd->n", m.W[i], m.C[j])).max() < 1e-5
    B = binarized(counts_for(CONTEXT, 300), math.e)
    m = als_glove(B, D, iters=3)
    assert max(np.abs(m.W).max(), np.abs(m.C).max()) < 1e-12 and m.loss < 1e-20
    assert np.abs(m.b[:, None] + m.bt[None, :] - 1.0)[B.nonzero()].max() < 1e-12  # the biases carry log e = 1
    sc = w2v._scores(np.zeros((B.shape[0], D)))  # the exact optimum's vectors: every cosine ties
    assert sc["test_auc"] == 0.5 and sc["dev_auc"] == 0.5


def test_training():
    """Def 3 and the cross-check: AdaGrad's loss falls across its checkpoints; ALS's objective never increases (to
    rounding); ALS reaches a lower J than 50 AdaGrad epochs, so AdaGrad is not converged."""
    g = train_glove(CONTROLLED)
    losses = [h["loss"] for h in g.history]
    assert all(a > b for a, b in zip(losses, losses[1:])), losses
    m = als_model(300, False)
    obj = [h["objective"] for h in m.history]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(obj, obj[1:])), obj
    assert m.loss < g.loss


def test_benchmark():
    """§3: on Topic 6's test nouns GloVe's vectors (AdaGrad, W + C) are within 2.5 jackknife standard errors of
    chance AUC, and their 1-NN is below 2 × chance; SGNS and the SVD of the same counts beat GloVe by more than 5
    standard errors. Most observed cells are seen once, and the biases alone explain most of log X on Ω."""
    c = benchmark_summary()
    assert abs(c["glove_test_auc"] - 0.5) < 2.5 * c["glove_auc_se"], c
    assert abs(c["glove_als_test_auc"] - 0.5) < 0.05
    assert c["glove_test_nn"] < 2 * c["chance_nn"]
    assert c["glove_vs_sgns_auc_diff"] < -5 * c["glove_vs_sgns_auc_diff_se"]
    assert c["glove_vs_svd_auc_diff"] < -5 * c["glove_vs_svd_auc_diff_se"]
    assert c["singleton_share"] > 0.65 and c["le2_share"] > 0.75
    assert c["additive_share"] > 0.5


def test_scale_curve():
    """§3: continuing the corpus, the singleton share of the observed cells falls, and the test AUC of GloVe and of
    GloVe with imputed zeros (Def 2) rises at each step; imputation beats plain GloVe at every length, and by 0.25 or
    more once the corpus is four times longer; the SVD of the same counts stays above plain GloVe."""
    s = scale_summary()
    sing = [s[f"singleton_share_{n}"] for n in SCALE_DOCS]
    assert all(a > b for a, b in zip(sing, sing[1:])), sing
    for name in ("glove", "imputed"):
        auc = [s[f"{name}_test_auc_{n}"] for n in SCALE_DOCS]
        assert all(a < b for a, b in zip(auc, auc[1:])), (name, auc)
    for n in SCALE_DOCS:
        assert s[f"imputed_test_auc_{n}"] > s[f"glove_test_auc_{n}"] and s[f"svd_test_auc_{n}"] > s[f"glove_test_auc_{n}"]
    assert all(s[f"imputed_test_auc_{n}"] - s[f"glove_test_auc_{n}"] > 0.25 for n in SCALE_DOCS[1:])


def test_zero_imputation():
    """Def 2: imputing unseen pairs at half a count makes the support full, so with flat weights it is Thm 2 on the
    imputed table; ALS's implicit identity gives the same objective as the explicit dense sum (to 10⁻⁹)."""
    rng = np.random.default_rng([SEED, 809])
    X = sparse.csr_matrix(w2v._small_instance(rng, 8, 7, zero_share=0.4))
    w0, t0 = impute_zeros()
    m = als_glove(X, 2, zeros=(w0, t0), iters=30, seed=818)
    Xd = X.toarray()
    S = m.W @ m.C.T + m.b[:, None] + m.bt[None, :]
    wt = np.where(Xd > 0, glove_weight(Xd), w0)
    tgt = np.where(Xd > 0, np.log(np.where(Xd > 0, Xd, 1.0)), t0)
    dense = float(np.sum(wt * (S - tgt) ** 2) + RIDGE * (np.sum(m.W**2) + np.sum(m.C**2)))
    assert abs(m.history[-1]["objective"] - dense) < 1e-9 * dense
    Xi = np.where(Xd > 0, Xd, 0.5)
    flat = als_glove(X, 2, x_max=1e-9, ridge=0.0, zeros=(1.0, math.log(0.5)), iters=400, seed=819)
    assert abs(flat.history[-1]["objective"] - flat_weight_optimum(Xi, 2)["loss"]) < 1e-8 * flat_weight_optimum(Xi, 2)["loss"]


def test_affine_gauge():
    """Prop 4: a random κ = 50 A with random t, s, β leaves every score unchanged (to 10⁻¹⁰ relative); the substitution
    is recovered from the two parameter sets; word cosines move."""
    rng = np.random.default_rng([SEED, 810])
    X = sparse.csr_matrix(w2v._small_instance(rng, 9, 8, zero_share=0.3))
    for _ in range(10):
        W, Cv, b, bt = rng.normal(size=(9, 3)), rng.normal(size=(8, 3)), rng.normal(size=9), rng.normal(size=8)
        q1, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        A = q1 @ np.diag(50 ** np.linspace(-0.5, 0.5, 3)) @ q2
        t, s, beta = rng.normal(size=3), rng.normal(size=3), float(rng.normal())
        p2 = affine_gauge(W, Cv, b, bt, A, t, s, beta)
        S1 = W @ Cv.T + b[:, None] + bt[None, :]
        S2 = p2[0] @ p2[1].T + p2[2][:, None] + p2[3][None, :]
        assert np.abs(S2 - S1).max() < 1e-10 * np.abs(S1).max()
        assert abs(glove_loss(*p2, X) - glove_loss(W, Cv, b, bt, X)) < 1e-10 * glove_loss(W, Cv, b, bt, X)
        A_, t_, s_, beta_ = recover_affine_gauge((W, Cv, b, bt), p2)
        assert np.abs(A_ - A).max() < 1e-8 and np.abs(t_ - t).max() < 1e-8
        assert np.abs(s_ - s).max() < 1e-8 and abs(beta_ - beta) < 1e-8
        cos = lambda M: ds.unit_rows(M, np.arange(len(M))) @ ds.unit_rows(M, np.arange(len(M))).T  # noqa: E731
        assert np.abs(cos(p2[0]) - cos(W)).max() > 1e-2


def test_ridge_balanced_centered():
    """Prop 5: ridge ALS (biases unpenalized) on random instances converges to a stationary point that is balanced and
    centered (both below 10⁻⁶), and the balanced factors attain ½(‖W‖² + ‖C‖²) = ‖WCᵀ‖_*; an unbalanced
    factorization of the same product costs more. Corpus: Prop 5 is a statement about stationary points, and 40 ALS
    sweeps at λ = 10⁻³ are far from one in the gauge directions — the weak ridge has barely pulled W and C toward
    balance (imbalance above 0.5) — so neither trained model sits in the ridge's gauge."""
    rng = np.random.default_rng([SEED, 811])
    for _ in range(5):
        X = sparse.csr_matrix(w2v._small_instance(rng, 9, 8, zero_share=0.3))
        m = als_glove(X, 3, x_max=5.0, ridge=0.05, iters=3000, seed=820)
        bc = balance_and_centering(m.W, m.C)
        assert bc["imbalance"] < 1e-6 and bc["mean_w"] < 1e-6 and bc["mean_c"] < 1e-6, bc
        Z = m.W @ m.C.T
        nuc = np.linalg.svd(Z, compute_uv=False).sum()
        assert nuc > 1e-3 and abs(0.5 * (np.sum(m.W**2) + np.sum(m.C**2)) - nuc) < 1e-6 * nuc
        Wu, Cu = w2v.random_gauge(m.W, m.C, 10.0, rng)
        assert 0.5 * (np.sum(Wu**2) + np.sum(Cu**2)) > nuc * (1 + 1e-3)
    al = balance_and_centering(als_model(300, False).W, als_model(300, False).C)
    assert al["imbalance"] > 0.5, al


def test_exact_parallelogram():
    """Thm 3: (⇒) in an exact log-bilinear table with a planted parallelogram w_d = w_b − w_a + w_c, Δ = PMI_b − PMI_a
    − PMI_d + PMI_c is constant (to 10⁻¹⁰) and the ratio rows are proportional; (⇐) any exact fit of such a table —
    Thm 2's closed form, in its own gauge — has w_b − w_a = w_d − w_c (to 10⁻⁸); breaking the relation breaks both; the
    criterion survives a random affine gauge. SGNS: with scores WCᵀ and no biases, Δ ≡ 0 exactly when the
    parallelogram holds."""
    rng = np.random.default_rng([SEED, 812])
    for _ in range(10):
        d, n, m = 3, 9, 12
        W, Cv = rng.normal(0, 0.6, (n, d)), rng.normal(0, 0.6, (m, d))
        b, bt = rng.normal(size=n), rng.normal(size=m)
        a_, b_, c_, d_ = 0, 1, 2, 3
        W[d_] = W[b_] - W[a_] + W[c_]
        X = np.exp(W @ Cv.T + b[:, None] + bt[None, :])
        P = w2v.pmi_dense(X)
        delta = P[b_] - P[a_] - P[d_] + P[c_]
        assert np.ptp(delta) < 1e-10
        pc = X / X.sum(axis=1, keepdims=True)
        ratio = (pc[b_] / pc[a_]) / (pc[d_] / pc[c_])
        assert np.ptp(ratio) < 1e-10 * ratio.mean()
        opt = flat_weight_optimum(X, d)
        v = opt["W"][b_] - opt["W"][a_] - opt["W"][d_] + opt["W"][c_]
        assert np.abs(v).max() < 1e-8
        assert parallelogram_defect(opt["S"][[a_, b_, c_, d_]]) < 1e-8
        p2 = affine_gauge(opt["W"], opt["C"], opt["b"], opt["bt"], rng.normal(size=(d, d)) + 3 * np.eye(d),
                          rng.normal(size=d), rng.normal(size=d), 0.7)
        assert np.abs(p2[0][b_] - p2[0][a_] - p2[0][d_] + p2[0][c_]).max() < 1e-8
        W2 = W.copy()
        W2[d_] += rng.normal(0, 0.3, d)
        X2 = np.exp(W2 @ Cv.T + b[:, None] + bt[None, :])
        P2 = w2v.pmi_dense(X2)
        assert np.ptp(P2[b_] - P2[a_] - P2[d_] + P2[c_]) > 1e-3
        o2 = flat_weight_optimum(X2, d)
        assert np.abs(o2["W"][b_] - o2["W"][a_] - o2["W"][d_] + o2["W"][c_]).max() > 1e-3
        Sg = W @ Cv.T  # SGNS's exact scores, no biases
        assert np.abs(Sg[b_] - Sg[a_] - Sg[d_] + Sg[c_]).max() < 1e-12
        assert np.abs((W2 @ Cv.T)[b_] - (W2 @ Cv.T)[a_] - (W2 @ Cv.T)[d_] + (W2 @ Cv.T)[c_]).max() > 1e-3


def test_analogy_gauge_dependence():
    """Cor 2: 3CosAdd's answers are invariant under W ↦ DWQ (D positive diagonal, Q orthogonal) but not under a
    general A or a translation. Corpus: every variant keeps its scores (to 10⁻¹⁰ relative), its median defect (to
    10⁻⁸), and its gauge-free answers exactly, while 3CosAdd's accuracy moves — Topic 7's random gauge costs SGNS over
    0.2 on city → region."""
    rng = np.random.default_rng([SEED, 821])
    X = rng.normal(size=(60, 5))
    Q = np.array([(a, b, c) for a, b, c in rng.integers(0, 60, (200, 3)) if len({a, b, c}) == 3])

    def answers(M):
        U = ds.unit_rows(M, np.arange(len(M)))
        return np.array([three_cos_add(U, a, b, c)[0] for a, b, c in Q])
    base = answers(X)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert np.array_equal(answers(rng.uniform(0.2, 5, (60, 1)) * X @ q), base)
    A = rng.normal(size=(5, 5))
    assert np.mean(answers(X @ A) != base) > 0.1
    assert np.mean(answers(X + 2 * rng.normal(size=5)) != base) > 0.1
    gt = gauge_table()
    for method, names in (("sgns", ("p0", "balanced", "p1", "random")),
                          ("glove", ("balanced", "centered", "translated", "random"))):
        for name in names:
            assert gt[f"{method}|{name}|score_change"] < 1e-10, (method, name)
            for rel in RELATIONS:
                assert abs(gt[f"{method}|{name}|{rel}|defect"] - gt[f"{method}|trained|{rel}|defect"]) < 1e-8
                assert gt[f"{method}|{name}|{rel}|gauge_free"] == gt[f"{method}|trained|{rel}|gauge_free"]
    assert gt["sgns|trained|city_region|open"] - gt["sgns|random|city_region|open"] > 0.2
    moved = [abs(gt[f"glove|{n}|city_region|open"] - gt["glove|trained|city_region|open"])
             for n in ("centered", "translated", "random")]
    assert max(moved) > 0.05, moved


def test_finance_findings():
    """§6: SGNS solves more of both relations than GloVe, open-vocabulary and within the category, and beats within-
    category chance by over 2× on both. No model's scores are near exact parallelograms (median defect above 0.5),
    but for every model the right tail is closer to one than the wrong tails are, so the gauge-free solver beats
    chance; for SGNS it nearly matches 3CosAdd in the trained gauge on city → region."""
    f = finance_summary()
    for rel in RELATIONS:
        ch = f[f"{rel}|sgns|chance_within"]
        assert f[f"{rel}|sgns|open"] > f[f"{rel}|glove|open"]
        assert f[f"{rel}|sgns|within"] > f[f"{rel}|glove|within"]
        assert f[f"{rel}|sgns|within"] > 2 * ch
        for name in ("glove", "sgns", "svd"):
            assert 0.5 < f[f"{rel}|{name}|gauge_free_defect_right"] < f[f"{rel}|{name}|gauge_free_defect_wrong"]
            assert f[f"{rel}|{name}|gauge_free_within"] > ch
    assert abs(f["city_region|sgns|gauge_free_within"] - f["city_region|sgns|within"]) < 0.05


def test_viz_payload():
    """The lab's data: Panel A's cells span the count bands plus one unobserved context; the histogram covers every
    observed cell; Panel B has every variant of both methods for both relations, with a projection of every relation
    word and three answers per curated question; the scale curve has one value per corpus length."""
    p = viz_payload()
    cells = p["weights"]["cells"]
    assert cells[-1]["a"] == 0 and all(c["a"] > 0 for c in cells[:-1])
    assert sum(p["weights"]["hist"]) == counts_for(CONTEXT, 300).nnz
    for rel, cell in p["analogies"].items():
        for method, names in LAB_VARIANTS.items():
            for name in names:
                v = cell["methods"][method][name]
                assert len(v["xy"]) == len(cell["words"]) and all(len(a) == 3 for a in v["answers"])
    assert all(len(v) == len(SCALE_DOCS) for v in p["scale"].values())
    assert len(json.dumps(p, separators=(",", ":"))) < 40_000


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
    print("\nSummary (train generate_corpus(300, 0); GloVe on Topic 6's L8 counts; benchmark = planted nouns):")
    for k, v in summary().items():
        print(f"  {k:52} {v:.4g}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else
              f"  {k:52} {v:.4f}" if isinstance(v, float) else f"  {k:52} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
