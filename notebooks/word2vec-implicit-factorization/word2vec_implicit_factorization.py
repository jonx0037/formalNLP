"""word2vec as implicit matrix factorization — the reference implementation for formalNLP's seventh topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. Three objectives, one matrix: skip-gram with a full softmax, noise-contrastive estimation, and
   negative sampling (SGNS) all fit PMI plus row and column offsets. SGNS's cell objective
   a log σ(x) + b log σ(−x) is strictly concave with maximizer PMI_α − log k on observed pairs and no
   maximizer on unobserved ones (Levy and Goldberg 2014); NCE keeps the log k P_n(c) correction and
   recovers log P(c | w).
2. What SGNS minimizes at low rank: exactly a weighted logistic PCA, Σ (a + b) KL(Bern(y) ‖ Bern(σ(x))),
   whose local weight ab/(a + b) is at most min(a, b). Rare pairs get little weight, so the fit to
   PMI_α − log k improves with the pair's count, and the unobserved majority is pushed down.
3. The algorithm: word2vec's minibatch SGD with sampled negatives, whose sampled loss is unbiased for the
   population objective.
4. Gauge and balance: the loss sees only WCᵀ, so it fixes word similarity only up to W ↦ WA. Gradient flow
   conserves WᵀW − CᵀC; balanced factors are UΣ^½R and VΣ^½R, so balanced word vectors are Topic 6's
   p = ½ for the fitted matrix and, for a symmetric product, W + C keeps its positive part (Topic 6 Cor 1).
5. Recovering the planted sectors: SGNS against Topic 6's SVD on the same matrix, and a small dev grid,
   scored on Topic 6's held-out benchmark nouns.
6. Finance case study: a leave-one-out polarity probe on the stand-in lexicon asks whether a factorization
   finds a polarity direction the window counts do not carry.

Cross-checks: the hand-written SGNS gradient against complex-step and central differences (Topic 5's
checkers); every closed-form optimum against ``scipy.optimize`` L-BFGS at full rank; the Bernoulli KL
against ``scipy.stats.entropy``; the corpus optimum against Topic 6's ``pmi``.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/word2vec-implicit-factorization/word2vec_implicit_factorization.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import optimize, sparse, stats

_HERE = Path(__file__).resolve().parent
for _dep in ("distributional-semantics-ppmi-svd", "neural-language-models"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import distributional_semantics_ppmi_svd as ds  # noqa: E402
import neural_language_models as nlm  # noqa: E402

# ONE stream per random-instance claim, each from default_rng([SEED, k]) (STARTER-PROMPT A1); k ≥ 700 keeps
# this module's streams apart from Topic 6's.
SEED = 20261023

K_NEG = 5  # word2vec's default number of negative samples
ALPHA = ds.ALPHA  # 0.75, word2vec's noise exponent (Topic 6 Def 2)
CONTEXTS = ("L2", "L8")
COUNT_BANDS = ((1, 1), (2, 4), (5, 19), (20, 99), (100, None))
BAND_NAMES = ("1", "2–4", "5–19", "20–99", "100+")
FIT_PER_BAND = 40  # pairs per count band in the lab's fit scatter


# --------------------------------------------------------------------------------------------
# 1. Three objectives, one matrix.
# --------------------------------------------------------------------------------------------


def log_sigmoid(x) -> np.ndarray:
    """log σ(x) = −log(1 + e^{−x}), evaluated on each side of 0 with the branch whose exponent is ≤ 0, so it never
    overflows; both branches are analytic, so it accepts the complex step."""
    x = np.asarray(x)
    flat = x.reshape(-1)
    out = np.empty(flat.shape, dtype=np.result_type(flat, np.float64))
    pos = np.real(flat) >= 0
    out[pos] = -np.log1p(np.exp(-flat[pos]))
    out[~pos] = flat[~pos] - np.log1p(np.exp(flat[~pos]))
    return out.reshape(x.shape)


def sigmoid(x) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x)))


def cell_objective(x, a, b) -> np.ndarray:
    """Thm 1's φ(x) = a log σ(x) + b log σ(−x): one cell of −L_NS, with a = #(w, c) and b = k #(w) P_n(c)."""
    return a * log_sigmoid(x) + b * log_sigmoid(-np.asarray(x))


def noise_distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    """P_α(c) ∝ #(c)^α from the column sums — Topic 6's ``marginals``, for dense count matrices."""
    pc = counts.sum(axis=0) ** alpha
    return pc / pc.sum()


def negative_weights(counts: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """Def 2's b = k #(w) P_α(c): the expected number of times (w, c) is drawn as a negative."""
    return k * counts.sum(axis=1)[:, None] * noise_distribution(counts, alpha)[None, :]


def softmax_loss(X: np.ndarray, counts: np.ndarray) -> float:
    """Def 1. L_soft = −Σ #(w, c) log softmax(x_w)_c (Topic 5's log-softmax)."""
    return -np.sum(counts * nlm.log_softmax(X))


def sgns_loss(X: np.ndarray, counts: np.ndarray, alpha: float, k: float) -> float:
    """Def 2. L_NS = −Σ [a log σ(x) + b log σ(−x)] over every cell (dense; small instances)."""
    return -np.sum(cell_objective(X, counts, negative_weights(counts, alpha, k)))


def nce_loss(X: np.ndarray, counts: np.ndarray, alpha: float, k: float) -> float:
    """Def 3. L_NS with x replaced by the NCE logit s = x − log(k P_n(c))."""
    s = X - np.log(k * noise_distribution(counts, alpha))[None, :]
    return -np.sum(cell_objective(s, counts, negative_weights(counts, alpha, k)))


def softmax_optimum(counts: np.ndarray) -> np.ndarray:
    """Prop 1 with κ = 0: log P(c | w), −∞ off the support."""
    with np.errstate(divide="ignore"):
        return np.log(counts / counts.sum(axis=1, keepdims=True))


def sgns_optimum_dense(counts: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """Thm 1: x* = log(a/b) = PMI_α − log k, −∞ off the support."""
    with np.errstate(divide="ignore"):
        return np.log(counts / negative_weights(counts, alpha, k))


def nce_optimum(counts: np.ndarray, alpha: float, k: float) -> np.ndarray:
    """Prop 2: s* = log(a/b), so x* = log(a/b) + log(k P_n(c)) — which is log P(c | w)."""
    return sgns_optimum_dense(counts, alpha, k) + np.log(k * noise_distribution(counts, alpha))[None, :]


def sgns_optimum(C: sparse.spmatrix, alpha: float, k: float) -> sparse.csr_matrix:
    """Thm 1 on the corpus: PMI_α − log k on the support of C, from Topic 6's ``pmi``."""
    M = sparse.coo_matrix(ds.pmi(C, alpha))
    return sparse.csr_matrix((M.data - math.log(k), (M.row, M.col)), shape=M.shape)


def pmi_dense(counts: np.ndarray) -> np.ndarray:
    """PMI (α = 1) of a dense count matrix, −∞ off the support."""
    N = counts.sum()
    with np.errstate(divide="ignore"):
        return np.log(counts * N / np.outer(counts.sum(axis=1), counts.sum(axis=0)))


# --------------------------------------------------------------------------------------------
# 2. What SGNS minimizes at low rank: weighted logistic PCA.
# --------------------------------------------------------------------------------------------


def bernoulli_entropy(y: np.ndarray) -> np.ndarray:
    from scipy.special import xlogy

    return -(xlogy(y, y) + xlogy(1 - y, 1 - y))


def bernoulli_kl(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """KL(Bern(y) ‖ Bern(σ(x))), with 0 log 0 = 0."""
    from scipy.special import xlogy

    return xlogy(y, y) - y * log_sigmoid(x) + xlogy(1 - y, 1 - y) - (1 - y) * log_sigmoid(-x)


def softplus(x) -> np.ndarray:
    """A(x) = log(1 + eˣ), the Bernoulli log-partition in its natural parameter."""
    return -log_sigmoid(-np.asarray(x))


def bregman_softplus(x: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    """D_A(x, x*) = A(x) − A(x*) − A′(x*)(x − x*), with A′ = σ (Thm 2's Bregman form)."""
    return softplus(x) - softplus(x_star) - sigmoid(x_star) * (x - x_star)


def logistic_pca_gap(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Thm 2's right side: Σ (a + b) KL(Bern(y) ‖ Bern(σ(x))), y = a/(a + b)."""
    return float(np.sum((a + b) * bernoulli_kl(a / (a + b), X)))


def local_weight(a, b) -> np.ndarray:
    """Cor 1: the curvature of the cell loss at its optimum, (a + b)σ(x*)σ(−x*) = ab/(a + b)."""
    return a * b / (a + b)


def pair_multiset(C: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """The N ordered (word, context) pairs whose counts are C (Topic 6 Def 1), as two index arrays."""
    C = sparse.coo_matrix(C)
    reps = C.data.astype(np.int64)
    return np.repeat(C.row, reps), np.repeat(C.col, reps)


def population_loss_grad(W: np.ndarray, Cv: np.ndarray, C: sparse.spmatrix, alpha: float, k: float,
                         block: int = 512) -> tuple[float, np.ndarray, np.ndarray]:
    """L_NS(WCᵀ)/N on the corpus and its gradient in (W, C), in row blocks so no dense V × V matrix is held:
    G = (a + b)σ(X) − a, ∇_W = GC, ∇_C = GᵀW (per pair)."""
    C = sparse.csr_matrix(C)
    N, n_w, pn = ds.marginals(C, alpha)
    loss, gW, gC = 0.0, np.zeros_like(W), np.zeros_like(Cv)
    for s in range(0, W.shape[0], block):
        rows = slice(s, s + block)
        X = W[rows] @ Cv.T
        a = C[rows].toarray()
        b = k * n_w[rows, None] * pn[None, :]
        loss -= float(np.sum(cell_objective(X, a, b)))
        G = (a + b) * sigmoid(X) - a
        gW[rows] = G @ Cv
        gC += G.T @ W[rows]
    return loss / N, gW / N, gC / N


def entropy_floor(C: sparse.spmatrix, alpha: float, k: float) -> float:
    """inf L_NS / N = Σ (a + b) H(y) / N over observed cells (unobserved cells have y = 0 and H = 0): Thm 2's
    floor, approached only at full rank."""
    C = sparse.coo_matrix(C)
    N, n_w, pn = ds.marginals(C, alpha)
    a = C.data
    b = k * n_w[C.row] * pn[C.col]
    return float(np.sum((a + b) * bernoulli_entropy(a / (a + b))) / N)


def band_of(counts: np.ndarray) -> np.ndarray:
    """The COUNT_BANDS index of each pair count."""
    out = np.full(counts.shape, -1)
    for i, (lo, hi) in enumerate(COUNT_BANDS):
        out[(counts >= lo) & (counts <= (hi if hi is not None else np.inf))] = i
    return out


def support_scores(W: np.ndarray, Cv: np.ndarray, C: sparse.spmatrix) -> np.ndarray:
    """w·c on every observed pair of C (COO order)."""
    C = sparse.coo_matrix(C)
    return np.einsum("ij,ij->i", W[C.row], Cv[C.col])


@lru_cache(maxsize=None)
def fit_sample(context: str) -> dict[str, np.ndarray]:
    """The lab's fit scatter: FIT_PER_BAND observed pairs per count band (indices into C's COO order), and as many
    unobserved cells, drawn once (rng [SEED, 720])."""
    C = sparse.coo_matrix(ds.cooc(context))
    rng = np.random.default_rng([SEED, 720])
    bands = band_of(C.data)
    obs = np.concatenate([np.sort(rng.choice(np.flatnonzero(bands == i), FIT_PER_BAND, replace=False))
                          for i in range(len(COUNT_BANDS))])
    V = C.shape[0]
    seen = set(zip(C.row.tolist(), C.col.tolist()))
    unobs = []
    while len(unobs) < FIT_PER_BAND:
        w, c = (int(v) for v in rng.integers(0, V, 2))
        if (w, c) not in seen:
            unobs.append((w, c))
    return {"obs": obs, "unobs": np.array(unobs)}


# --------------------------------------------------------------------------------------------
# 3. The algorithm: word2vec's minibatch SGD.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SGNSConfig:
    """Def 4's hyperparameters. checkpoints: the epochs (0 = initialization) at which training records its state."""
    context: str
    d: int
    k: int = K_NEG
    alpha: float = ALPHA
    epochs: int = 20
    batch: int = 512
    lr0: float = 0.05
    checkpoints: tuple[int, ...] = ()


CONTROLLED = SGNSConfig("L8", 25, epochs=40, checkpoints=(0, 1, 4, 10, 40))  # Topic 6's winning matrix
GRID = tuple(SGNSConfig(c, d) for c in CONTEXTS for d in (25, 100))


@dataclass(frozen=True)
class Trained:
    cfg: SGNSConfig
    W: np.ndarray
    C: np.ndarray
    W0: np.ndarray  # the initialization, for Thm 3's D₀
    loss: float  # L_NS / N at the end
    grad_norm: float  # ‖∇L_NS / N‖ at the end
    history: tuple[dict, ...]


def sampled_loss_grad(W: np.ndarray, Cv: np.ndarray, wi: np.ndarray, ctx: np.ndarray, need_loss: bool = True
                      ) -> tuple[float, np.ndarray, np.ndarray]:
    """One minibatch's summed sampled loss Σ_pairs [−log σ(w·c) − Σ_i log σ(−w·c_i)] and its gradient in (W, C).
    ctx[:, 0] is each pair's context and ctx[:, 1:] its k negatives. Repeated rows add, so the scatter is a product
    with the batch's incidence matrix, built directly in CSC form (one entry per column) — over 10× faster than
    np.add.at. Training skips the loss (need_loss=False)."""
    V, d = W.shape
    Wb = W[wi]
    Cb = Cv[ctx]
    s = np.einsum("bd,bkd->bk", Wb, Cb)
    loss = -np.sum(log_sigmoid(s[:, 0])) - np.sum(log_sigmoid(-s[:, 1:])) if need_loss else math.nan
    g = sigmoid(s)  # d/ds of −log σ(−s); the positive column's is σ(s) − 1
    g[:, 0] -= 1.0
    dtype = np.real(W).dtype

    def scatter(idx, vals):
        return sparse.csc_matrix((np.ones(len(idx), dtype), idx, np.arange(len(idx) + 1)), shape=(V, len(idx))) @ vals

    gW = scatter(wi, np.einsum("bk,bkd->bd", g, Cb))
    gC = scatter(ctx.ravel(), (g[..., None] * Wb[:, None, :]).reshape(-1, d))
    return loss, gW, gC


def _record(cfg: SGNSConfig, epoch: int, W, Cv) -> dict:
    C = ds.cooc(cfg.context)
    W, Cv = W.astype(np.float64), Cv.astype(np.float64)
    fs = fit_sample(cfg.context)
    x = support_scores(W, Cv, C)
    rec = {"epoch": epoch, "x_obs": x[fs["obs"]],
           "x_unobs": np.einsum("ij,ij->i", W[fs["unobs"][:, 0]], Cv[fs["unobs"][:, 1]]),
           "loss": population_loss_grad(W, Cv, C, cfg.alpha, cfg.k)[0], "imbalance": imbalance(W, Cv),
           "D": W.T @ W - Cv.T @ Cv}
    if epoch > 0:
        target = sgns_optimum(C, cfg.alpha, cfg.k)
        rec["band_r"] = band_fit(x, sparse.coo_matrix(C).data, sparse.coo_matrix(target).data)["r"]
        sv = np.linalg.svd(gauge_of(W, Cv), compute_uv=False)
        rec["gauge_sv"] = (float(sv[-1]), float(sv[0]))
    return rec


@lru_cache(maxsize=None)
def train_sgns(cfg: SGNSConfig) -> Trained:
    """Def 4. word2vec's SGD on the pair multiset of Topic 6's C for cfg.context: each epoch a fresh permutation in
    minibatches of cfg.batch pairs, k negatives per pair drawn i.i.d. from P_α (Topic 6's ``marginals``, so Thm 1's
    optimum is exactly ``ds.pmi`` − log k), learning rate η₀ max(10⁻⁴, 1 − t/T) linear in the pairs seen, and the
    word2vec C code's initialization W ~ U[−½/d, ½/d], C = 0. Float32 for speed; one stream (rng [SEED, 710])."""
    C = ds.cooc(cfg.context)
    _, _, pn = ds.marginals(C, cfg.alpha)
    wi_all, ci_all = pair_multiset(C)
    V, n = C.shape[0], len(wi_all)
    rng = np.random.default_rng([SEED, 710])
    W = ((rng.random((V, cfg.d)) - 0.5) / cfg.d).astype(np.float32)
    Cv = np.zeros((V, cfg.d), np.float32)
    W0 = W.astype(np.float64)
    cdf = np.cumsum(pn)
    cdf[-1] = 1.0
    T, t = n * cfg.epochs, 0
    history = [_record(cfg, 0, W, Cv)] if 0 in cfg.checkpoints else []
    for epoch in range(1, cfg.epochs + 1):
        perm = rng.permutation(n)
        ctx_all = np.column_stack([ci_all[perm], np.searchsorted(cdf, rng.random((n, cfg.k)))])  # the epoch's negatives
        wi_ep = wi_all[perm]
        for s in range(0, n, cfg.batch):
            lr = np.float32(cfg.lr0 * max(1e-4, 1.0 - t / T))
            _, gW, gC = sampled_loss_grad(W, Cv, wi_ep[s : s + cfg.batch], ctx_all[s : s + cfg.batch], need_loss=False)
            W -= lr * gW
            Cv -= lr * gC
            t += min(cfg.batch, n - s)
        if epoch in cfg.checkpoints:
            history.append(_record(cfg, epoch, W, Cv))
    W, Cv = W.astype(np.float64), Cv.astype(np.float64)
    loss, gW, gC = population_loss_grad(W, Cv, C, cfg.alpha, cfg.k)
    assert math.isfinite(loss), f"SGD diverged for {cfg}"
    return Trained(cfg, W, Cv, W0, loss, float(math.hypot(np.linalg.norm(gW), np.linalg.norm(gC))), tuple(history))


def band_fit(x: np.ndarray, counts: np.ndarray, target: np.ndarray) -> dict[str, list[float]]:
    """Per COUNT_BANDS band: the correlation of w·c with x* = PMI_α − log k, and the mean absolute error."""
    bands = band_of(counts)
    r, mae, n = [], [], []
    for i in range(len(COUNT_BANDS)):
        m = bands == i
        r.append(float(np.corrcoef(x[m], target[m])[0, 1]))
        mae.append(float(np.abs(x[m] - target[m]).mean()))
        n.append(int(m.sum()))
    return {"r": r, "mae": mae, "n": n}


@lru_cache(maxsize=None)
def fit_summary(cfg: SGNSConfig = CONTROLLED) -> dict:
    """§2's corpus check of Cor 1: fit by count band, the unobserved cells, and the gap to Thm 2's floor."""
    m = train_sgns(cfg)
    C = ds.cooc(cfg.context)
    target = sparse.coo_matrix(sgns_optimum(C, cfg.alpha, cfg.k)).data
    counts = sparse.coo_matrix(C).data
    x = support_scores(m.W, m.C, C)
    out = band_fit(x, counts, target)
    rng = np.random.default_rng([SEED, 721])
    V = C.shape[0]
    w, c = rng.integers(0, V, (2, 20_000))
    unseen = np.asarray(C[w, c]).ravel() == 0
    xu = np.einsum("ij,ij->i", m.W[w[unseen]], m.C[c[unseen]])
    out.update({"x_obs_mean": float(x.mean()), "x_unobs_mean": float(xu.mean()),
                "unobs_below_obs_median": float(np.mean(xu < np.median(x))),
                "observed_share": C.nnz / V**2, "loss": m.loss, "floor": entropy_floor(C, cfg.alpha, cfg.k),
                "grad_norm": m.grad_norm})
    out["kl_gap"] = out["loss"] - out["floor"]
    return out


@lru_cache(maxsize=None)
def bias_direction(cfg: SGNSConfig = CONTROLLED) -> dict[str, float]:
    """The product's top singular pair: the share of words (contexts) on which u₁ (v₁) has its majority sign, whether
    σ₁u₁v₁ᵀ is entrywise negative, its share of ‖WCᵀ‖²_F, and its mean entry."""
    m = train_sgns(cfg)
    U, s, Vv = balanced_factors(m.W, m.C)
    u, v = U[:, 0], Vv[:, 0]
    su, sv = np.sign(u.sum()), np.sign(v.sum())
    return {"u_purity": float(np.mean(np.sign(u) == su)), "v_purity": float(np.mean(np.sign(v) == sv)),
            "negative": float(su * sv < 0), "energy_share": float(s[0] ** 2 / np.sum(s**2)),
            "mean_entry": float(s[0] * u.mean() * v.mean()), "s1_over_s2": float(s[0] / s[1])}


# --------------------------------------------------------------------------------------------
# 4. Gauge and balance.
# --------------------------------------------------------------------------------------------


def balanced_factors(W: np.ndarray, Cv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The thin SVD U, s, V of the product WCᵀ, computed through QR so no V × V matrix is formed."""
    Qw, Rw = np.linalg.qr(W)
    Qc, Rc = np.linalg.qr(Cv)
    u, s, vt = np.linalg.svd(Rw @ Rc.T)
    return Qw @ u, s, Qc @ vt.T


def gauge_of(W: np.ndarray, Cv: np.ndarray) -> np.ndarray:
    """The A with W = UΣ^½A (then C = VΣ^½A⁻ᵀ): orthogonal iff (W, C) is balanced (Lemma 1)."""
    U, s, _ = balanced_factors(W, Cv)
    return (U / np.sqrt(s)).T @ W


def refactor(W: np.ndarray, Cv: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Topic 6's eigenvalue weight applied to the fitted product: (UΣ^q, VΣ^{1−q}), the same WCᵀ for every q."""
    U, s, Vv = balanced_factors(W, Cv)
    return U * s**q, Vv * s ** (1 - q)


def random_gauge(W: np.ndarray, Cv: np.ndarray, kappa: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Prop 3: (WA, CA⁻ᵀ) for A = Q₁ diag(κ^{t}) Q₂ with t evenly spaced in [−½, ½] and Q₁, Q₂ Haar-orthogonal, so
    A has condition number κ and the product, hence every loss, is unchanged."""
    d = W.shape[1]
    q1, _ = np.linalg.qr(rng.normal(size=(d, d)))
    q2, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = q1 @ np.diag(kappa ** np.linspace(-0.5, 0.5, d)) @ q2
    return W @ A, Cv @ np.linalg.inv(A).T


def imbalance(W: np.ndarray, Cv: np.ndarray) -> float:
    """‖WᵀW − CᵀC‖_F / ‖WᵀW‖_F: Thm 3's conserved quantity, relative to the word Gram's scale."""
    return float(np.linalg.norm(W.T @ W - Cv.T @ Cv) / np.linalg.norm(W.T @ W))


def _small_instance(rng: np.random.Generator, n: int, m: int, zero_share: float = 0.0) -> np.ndarray:
    counts = rng.poisson(rng.gamma(1.0, 6.0, (n, m))).astype(float)
    counts[rng.random((n, m)) < zero_share] = 0
    counts[counts.sum(axis=1) == 0, 0] = 1
    counts[0, counts.sum(axis=0) == 0] = 1
    return counts


def _dense_factor_loss_grad(W, Cv, a, b):
    X = W @ Cv.T
    G = (a + b) * sigmoid(X) - a
    return -np.sum(cell_objective(X, a, b)), G @ Cv, G.T @ W


def gd_drift(eta: float, horizon: float, counts: np.ndarray, d: int, rng: np.random.Generator) -> float:
    """Thm 3 on a small dense instance: gradient descent with step η for horizon/η steps from a random
    (unbalanced) start; returns ‖D_T − D_0‖_F, which → 0 like η as the steps approach gradient flow."""
    a, b = counts, negative_weights(counts, ALPHA, K_NEG)
    W = rng.normal(0, 0.3, (counts.shape[0], d))
    Cv = rng.normal(0, 0.3, (counts.shape[1], d))
    D0 = W.T @ W - Cv.T @ Cv
    for _ in range(int(round(horizon / eta))):
        _, gW, gC = _dense_factor_loss_grad(W, Cv, a, b)
        W, Cv = W - eta * gW, Cv - eta * gC
    return float(np.linalg.norm(W.T @ W - Cv.T @ Cv - D0))


DRIFT_ETAS = (1e-3, 5e-4, 2.5e-4)
DRIFT_HORIZON = 2.0


@lru_cache(maxsize=1)
def drift_summary() -> dict[str, float]:
    """Thm 3 at small scale: ‖D_T − D_0‖_F after gradient descent to a fixed horizon, for halving steps η."""
    counts = _small_instance(np.random.default_rng([SEED, 707]), 6, 6) + 1
    drifts = [gd_drift(eta, DRIFT_HORIZON, counts, 3, np.random.default_rng([SEED, 708])) for eta in DRIFT_ETAS]
    return {**{f"drift_{eta:g}": v for eta, v in zip(DRIFT_ETAS, drifts)},
            "ratio_1": drifts[0] / drifts[1], "ratio_2": drifts[1] / drifts[2]}


SYM_ETAS = (0.02, 0.01)


@lru_cache(maxsize=1)
def symmetric_dynamics() -> dict[str, float]:
    """Cor 2 (ii) end to end on a small instance: symmetric counts with α = 1 make the objective symmetric and its
    full-rank optimum X* = PMI − log k symmetric and indefinite. Gradient descent at full rank from a balanced start
    (C₀ = RW₀, R orthogonal) is run to horizon 400 (rescaled step 50η on the per-pair loss) for two η; for each, the
    final imbalance, the fit to X*, and the gap between ¼(W + C)(W + C)ᵀ and [X*]₊."""
    rng = np.random.default_rng([SEED, 711])
    n = 5
    S = _small_instance(rng, n, n) + 2
    counts = S + S.T
    a, b = counts, negative_weights(counts, 1.0, K_NEG)
    X_star = sgns_optimum_dense(counts, 1.0, K_NEG)
    lam, Q = np.linalg.eigh(X_star)
    pos = (Q * np.maximum(lam, 0)) @ Q.T
    W_init = rng.normal(0, 0.1, (n, n))
    R, _ = np.linalg.qr(rng.normal(size=(n, n)))
    out = {"n_negative_eigs": int((lam < 0).sum()), "n_positive_eigs": int((lam > 0).sum())}
    for eta in SYM_ETAS:
        W, Cv = W_init.copy(), R @ W_init  # CᵀC = WᵀRᵀRW = WᵀW: balanced at the start
        for _ in range(int(round(400 / eta))):
            G = ((a + b) * sigmoid(W @ Cv.T) - a) / counts.sum()
            W, Cv = W - eta * 50 * G @ Cv, Cv - eta * 50 * G.T @ W
        out[f"imbalance_{eta:g}"] = imbalance(W, Cv)
        out[f"fit_{eta:g}"] = float(np.abs(W @ Cv.T - X_star).max())
        out[f"wc_gap_{eta:g}"] = float(np.abs((W + Cv) @ (W + Cv).T / 4 - pos).max())
    return out


@lru_cache(maxsize=None)
def balance_summary(cfg: SGNSConfig = CONTROLLED) -> dict:
    """§4's corpus check: the initial and final imbalance, the gauge's singular values, and the benchmark under the
    p-family of re-factorizations of the same product and under a random κ = 100 gauge."""
    m = train_sgns(cfg)
    sv = np.linalg.svd(gauge_of(m.W, m.C), compute_uv=False)
    gram = m.W.T @ m.W
    D0 = m.W0.T @ m.W0
    out = {"imbalance_final": imbalance(m.W, m.C), "D0_rel": float(np.linalg.norm(D0) / np.linalg.norm(gram)),
           "gauge_sv_min": float(sv[-1]), "gauge_sv_max": float(sv[0])}
    A = gauge_of(m.W, m.C)
    Ds = {h["epoch"]: h["D"] for h in m.history}
    if 1 in Ds and cfg.epochs in Ds:  # how much of the final D was set in the first epoch
        out["D_change_after_epoch1"] = float(np.linalg.norm(Ds[cfg.epochs] - Ds[1]) / np.linalg.norm(Ds[cfg.epochs]))
        out["D_growth_epoch1"] = float(np.linalg.norm(Ds[1] - D0) / np.linalg.norm(D0))
    out["gauge_top_row_norm"] = float(np.linalg.norm(A[0]))
    out["gauge_rest_row_norm_median"] = float(np.median(np.linalg.norm(A[1:], axis=1)))
    return out


# --------------------------------------------------------------------------------------------
# 5. Recovering the planted sectors.
# --------------------------------------------------------------------------------------------

REPRESENTATIONS = ("W", "W+C", "p0", "p½", "p1")


def sgns_vectors(cfg: SGNSConfig, rep: str = "W") -> np.ndarray:
    """A trained model's word vectors: W as trained, (W + C)/2, or the balanced re-factorization UΣ^p of its product
    at Topic 6's p ∈ {0, ½, 1}."""
    m = train_sgns(cfg)
    if rep == "W":
        return m.W
    if rep == "W+C":
        return (m.W + m.C) / 2
    return refactor(m.W, m.C, {"p0": 0.0, "p½": 0.5, "p1": 1.0}[rep])[0]


def _scores(X) -> dict[str, float]:
    dev, test = ds.benchmark_split()
    lab = ds.planted_labels()
    return {"dev_auc": ds.pair_auc(X, dev, lab), "dev_nn": ds.nn_accuracy(X, dev, lab),
            "test_auc": ds.pair_auc(X, test, lab), "test_nn": ds.nn_accuracy(X, test, lab)}


@lru_cache(maxsize=None)
def sgns_scores(cfg: SGNSConfig, rep: str = "W") -> dict[str, float]:
    return _scores(sgns_vectors(cfg, rep))


@lru_cache(maxsize=1)
def sgns_selected() -> SGNSConfig:
    """Topic 6's rule on the grid: the highest dev AUC of W as trained, dev 1-NN breaking ties."""
    return max(GRID, key=lambda c: (sgns_scores(c)["dev_auc"], sgns_scores(c)["dev_nn"]))


def _paired(Xa, Xb, words) -> dict[str, float]:
    """The AUC difference with its jackknife standard error, and the paired 1-NN difference with its standard error
    (Topic 6's §5 statistics)."""
    lab = ds.planted_labels()
    words = list(words)

    def diff(ws):
        return ds.pair_auc(Xa, ws, lab) - ds.pair_auc(Xb, ws, lab)

    def hits(X):
        S = ds.similarity(X, words)
        np.fill_diagonal(S, -np.inf)
        la = np.array([lab[w] for w in words])
        return (la[np.argmax(S, axis=1)] == la).astype(float)

    dh = hits(Xa) - hits(Xb)
    return {"auc_diff": diff(words), "auc_diff_se": ds.jackknife_se(diff, words), "nn_diff": float(dh.mean()),
            "nn_diff_se": float(dh.std(ddof=1) / math.sqrt(len(dh)))}


@lru_cache(maxsize=1)
def sgns_comparison() -> dict:
    """§5's headline: the controlled pair (SGNS against SVD of the same SPPMI₅ at L8, d = 25), the dev-selected grid
    model against Topic 6's two selected configurations, the rare band, and Cor 1's mechanism — the share of each
    method's emphasis on the benchmark nouns' rare pairs."""
    test = ds.benchmark_split()[1]
    lab = ds.planted_labels()
    ctx, d = CONTROLLED.context, CONTROLLED.d
    svd_cfg = (ctx, "sppmi5", d, 0.0)
    out = {f"ctl_{k}": v for k, v in sgns_scores(CONTROLLED).items()}
    out.update({f"ctl_svd_{k}": v for k, v in ds.scores(svd_cfg).items()})
    out.update({f"ctl_svd_half_{k}": v for k, v in ds.scores((ctx, "sppmi5", d, 0.5)).items()})
    out.update({f"ctl_vs_svd_{k}": v for k, v in _paired(sgns_vectors(CONTROLLED), ds.vectors(svd_cfg), test).items()})
    sel = sgns_selected()
    out["sel"] = f"{sel.context}|d={sel.d}"
    out.update({f"sel_{k}": v for k, v in sgns_scores(sel).items()})
    t6 = ds.selected()
    out.update({f"sel_vs_t6svd_{k}": v for k, v in _paired(sgns_vectors(sel), ds.vectors(t6["svd"]), test).items()})
    out["t6_svd_test_auc"], out["t6_svd_test_nn"] = ds.scores(t6["svd"])["test_auc"], ds.scores(t6["svd"])["test_nn"]
    out["t6_sparse_test_auc"] = ds.scores(t6["sparse"])["test_auc"]
    rare = list(ds.benchmark_words(*ds.RARE_BAND))
    for name, X in (("sgns", sgns_vectors(CONTROLLED)), ("svd", ds.vectors(svd_cfg))):
        out[f"rare_auc_{name}"] = ds.pair_auc(X, rare, lab)
        out[f"rare_nn_{name}"] = ds.nn_accuracy(X, rare, lab)
    # Cor 1's mechanism on the benchmark nouns' rows: what share of SPPMI₅'s squared mass (which the SVD fits with
    # uniform weight) and of SGNS's local weight ab/(a + b) falls on pairs seen at most 4 times.
    C = ds.cooc(ctx)
    rows = np.array([ds.index()[w] for w in ds.benchmark_words()])
    sub = sparse.coo_matrix(C[rows])
    _, n_w, pn = ds.marginals(C, CONTROLLED.alpha)
    a = sub.data
    b = CONTROLLED.k * n_w[rows][sub.row] * pn[sub.col]
    x_star = np.log(a / b)
    rare_pair = a <= 4
    sppmi = np.maximum(x_star, 0) ** 2
    wt = local_weight(a, b)
    out["mech_rare_share_support"] = float(rare_pair.mean())
    out["mech_rare_share_sppmi_mass"] = float(sppmi[rare_pair].sum() / sppmi.sum())
    out["mech_rare_share_sgns_weight"] = float(wt[rare_pair].sum() / wt.sum())
    for cfg in GRID:
        for k, v in sgns_scores(cfg).items():
            out[f"grid_{cfg.context}_{cfg.d}_{k}"] = v
    return out


# --------------------------------------------------------------------------------------------
# 6. Finance case study: polarity isn't a direction.
# --------------------------------------------------------------------------------------------


def polarity_probe(X, positive: Sequence[str], negative: Sequence[str]) -> float:
    """Leave-one-out accuracy of the difference-of-means polarity direction on unit-length vectors: each lexicon
    word is classified by which side of the midpoint between the other words' two class means it falls on. The
    direction and the midpoint are translation-invariant, so a shared offset (§2's bias direction) cannot help."""
    idx = ds.index()
    words = [*positive, *negative]
    Y = ds.unit_rows(X, [idx[w] for w in words])
    y = np.array([1] * len(positive) + [0] * len(negative), dtype=bool)
    hits = []
    for i in range(len(words)):
        keep = np.arange(len(words)) != i
        mp, mn = Y[keep & y].mean(axis=0), Y[keep & ~y].mean(axis=0)
        hits.append(((Y[i] - (mp + mn) / 2) @ (mp - mn) > 0) == y[i])
    return float(np.mean(hits))


@lru_cache(maxsize=1)
def finance_summary() -> dict:
    """§6: the polarity probe for SGNS (the d = 25 grid models, W and (W + C)/2), Topic 6's selected SVD weighting,
    and sparse PPMI at L2 and L8, against Topic 6's SO accuracy; and *strong*'s SGNS neighbors."""
    pw = ds.polarity_words()
    pos, neg = pw["positive"], pw["negative"]
    out: dict = {"n_pos": len(pos), "n_neg": len(neg),
                 "chance_se": math.sqrt(0.25 / (len(pos) + len(neg)))}
    for ctx in CONTEXTS:
        m = SGNSConfig(ctx, 25)
        out[f"probe_sgns_{ctx}"] = polarity_probe(sgns_vectors(m), pos, neg)
        out[f"probe_sgns_wc_{ctx}"] = polarity_probe(sgns_vectors(m, "W+C"), pos, neg)
        out[f"probe_svd_{ctx}"] = polarity_probe(ds.vectors(ds.lab_config(ctx, "svd")), pos, neg)
        out[f"probe_ppmi_{ctx}"] = polarity_probe(ds.weighted(ctx, "ppmi"), pos, neg)
        out[f"so_acc_{ctx}"] = ds.so_accuracy(ctx)
        out[f"strong_neighbors_{ctx}"] = ds.neighbors(sgns_vectors(m), "strong", top=7)
    probes = [v for k, v in out.items() if k.startswith("probe_")]
    out["probe_max"] = max(probes)
    out["probe_max_z"] = (max(probes) - 0.5) / out["chance_se"]
    return out


# --------------------------------------------------------------------------------------------
# The lab's data.
# --------------------------------------------------------------------------------------------

LAB_ROW = "revenue"  # Panel A's word: a frequent template noun whose L8 row reaches every count band
LAB_KS = (1, 5, 15)
LAB_ALPHAS = (1.0, ALPHA)
GAUGE_WORDS = ("strong", "loans")
GAUGE_TOP = 5
GAUGE_KAPPA = 100.0


def _r(x: float, nd: int = 4):
    return ds._r(x, nd)


@lru_cache(maxsize=1)
def lab_pairs() -> list[dict]:
    """Panel A's cells at L8: for LAB_ROW, its most associated context (highest PMI_α) in each count band, and the
    most frequent context it never meets. Each carries a = #(w, c), #(w), N, and P_α(c) for both lab α, so the page can draw the
    curve; x*, the local weight, and the trained w·c (k = 5, α = 0.75) come from here."""
    C = ds.cooc("L8").tocsr()
    vocab, idx = ds.vocab(), ds.index()
    i = idx[LAB_ROW]
    row = C[i].toarray().ravel()
    n_c = np.asarray(C.sum(axis=0)).ravel()
    m = train_sgns(CONTROLLED)
    prow = ds.pmi(C, ALPHA).tocsr()[i].toarray().ravel()
    picks = []
    for lo, hi in COUNT_BANDS:
        cand = np.flatnonzero((row >= lo) & (row <= (hi if hi is not None else np.inf)))
        picks.append(int(cand[np.argmax(prow[cand])]))
    unseen = np.flatnonzero(row == 0)
    picks.append(int(unseen[np.argmax(n_c[unseen])]))
    out = []
    for j in picks:
        cell = {"context": vocab[j], "a": float(row[j]), "n_w": float(row.sum()), "N": float(C.sum()),
                "p_noise": {}, "x_star": {}, "weight": {}, "trained": _r(float(m.W[i] @ m.C[j]), 3)}
        for alpha in LAB_ALPHAS:
            _, n_w, pn = ds.marginals(C, alpha)
            cell["p_noise"][str(alpha)] = float(f"{pn[j]:.6g}")
            for k in LAB_KS:
                b = k * n_w[i] * pn[j]
                key = f"{alpha}|{k}"
                cell["x_star"][key] = _r(math.log(row[j] / b), 3) if row[j] > 0 else None
                cell["weight"][key] = _r(local_weight(row[j], b), 3) if row[j] > 0 else None
        out.append(cell)
    return out


@lru_cache(maxsize=1)
def gauge_cells() -> dict:
    """Panel B's gauge view: for the controlled model's product, the p-family re-factorizations (p = 0, ½, 1), the
    model as trained, and a random κ = 100 gauge — test AUC, 1-NN, and GAUGE_WORDS' top neighbors. The product, and
    so every loss, is the same in all of them."""
    m = train_sgns(CONTROLLED)
    variants = {"trained": m.W, "p0": sgns_vectors(CONTROLLED, "p0"), "p½": sgns_vectors(CONTROLLED, "p½"),
                "p1": sgns_vectors(CONTROLLED, "p1"),
                "random": random_gauge(m.W, m.C, GAUGE_KAPPA, np.random.default_rng([SEED, 730]))[0]}
    out = {}
    for name, X in variants.items():
        sc = _scores(X)
        out[name] = {"test_auc": _r(sc["test_auc"], 3), "test_nn": _r(sc["test_nn"], 3),
                     "neighbors": {w: [[n, _r(s, 2)] for n, s in ds.neighbors(X, w, GAUGE_TOP)] for w in GAUGE_WORDS}}
    return out


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    """Panel A: lab_pairs(). Panel B, fit view: the controlled model's w·c on the fit sample at each checkpoint, with
    the sample's targets x*, count bands, per-band correlations, losses, and imbalances; gauge view: gauge_cells()."""
    C = ds.cooc(CONTROLLED.context)
    fs = fit_sample(CONTROLLED.context)
    coo = sparse.coo_matrix(C)
    target = sparse.coo_matrix(sgns_optimum(C, CONTROLLED.alpha, CONTROLLED.k)).data[fs["obs"]]
    m = train_sgns(CONTROLLED)
    vocab = ds.vocab()
    fit = {"bands": list(BAND_NAMES), "band": band_of(coo.data[fs["obs"]]).tolist(),
           "x_star": [_r(v, 2) for v in target],
           "pairs": [f"{vocab[coo.row[j]]}·{vocab[coo.col[j]]}" for j in fs["obs"]],
           "checkpoints": [{"epoch": h["epoch"], "x": [_r(v, 2) for v in h["x_obs"]],
                            "x_unobs": [_r(v, 2) for v in h["x_unobs"]], "loss": _r(h["loss"]),
                            "imbalance": _r(h["imbalance"], 3),
                            "band_r": [_r(v, 3) for v in h.get("band_r", [])],
                            "gauge_sv": [_r(v, 3) for v in h.get("gauge_sv", ())]} for h in m.history],
           "floor": _r(fit_summary()["floor"])}
    return {"context": CONTROLLED.context, "d": CONTROLLED.d, "k": CONTROLLED.k, "alpha": CONTROLLED.alpha,
            "row": LAB_ROW, "ks": list(LAB_KS), "alphas": [str(a) for a in LAB_ALPHAS], "pairs": lab_pairs(),
            "fit": fit, "gauge": gauge_cells(), "kappa": GAUGE_KAPPA,
            "svd_test_auc": _r(ds.scores((CONTROLLED.context, "sppmi5", CONTROLLED.d, 0.0))["test_auc"], 3),
            "chance_nn": _r(ds.comparison()["test_chance_nn"], 3)}


# --------------------------------------------------------------------------------------------
# Summary — the numbers the page quotes.
# --------------------------------------------------------------------------------------------


def summary() -> dict[str, float]:
    out: dict[str, float] = {}
    for ctx in CONTEXTS:
        C = ds.cooc(ctx)
        out[f"N_{ctx}"] = float(C.sum())
        out[f"support_{ctx}"] = float(C.nnz)
        out[f"observed_share_{ctx}"] = C.nnz / C.shape[0] ** 2
    fit = fit_summary()
    for i, name in enumerate(BAND_NAMES):
        out[f"fit_r_{name}"], out[f"fit_mae_{name}"], out[f"fit_n_{name}"] = fit["r"][i], fit["mae"][i], fit["n"][i]
    out.update({f"fit_{k}": v for k, v in fit.items() if isinstance(v, float)})
    out.update({f"bias_{k}": v for k, v in bias_direction().items()})
    out.update({f"bal_{k}": v for k, v in balance_summary().items()})
    out.update({f"drift_{k}": v for k, v in drift_summary().items()})
    out.update({f"sym_{k}": v for k, v in symmetric_dynamics().items()})
    for h in train_sgns(CONTROLLED).history:
        e = h["epoch"]
        out[f"ckpt{e}_loss"], out[f"ckpt{e}_imbalance"] = h["loss"], h["imbalance"]
        if "band_r" in h:
            out[f"ckpt{e}_r_1"], out[f"ckpt{e}_r_100+"] = h["band_r"][0], h["band_r"][-1]
            out[f"ckpt{e}_gauge_sv_min"], out[f"ckpt{e}_gauge_sv_max"] = h["gauge_sv"]
    for c in lab_pairs():
        out[f"lab_{c['context']}_a"] = c["a"]
        out[f"lab_{c['context']}_x_star"] = c["x_star"][f"{ALPHA}|{K_NEG}"] if c["a"] else math.nan
        out[f"lab_{c['context']}_weight"] = c["weight"][f"{ALPHA}|{K_NEG}"] if c["a"] else math.nan
        out[f"lab_{c['context']}_trained"] = c["trained"]
    for name, cell in gauge_cells().items():
        out[f"gauge_{name}_test_auc"], out[f"gauge_{name}_test_nn"] = cell["test_auc"], cell["test_nn"]
    for k, v in sgns_comparison().items():
        out[f"cmp_{k}"] = v
    for rep in REPRESENTATIONS:
        out[f"ctl_{rep}_test_auc"] = sgns_scores(CONTROLLED, rep)["test_auc"]
    for cfg in (CONTROLLED, *GRID):
        out[f"loss_{cfg.context}_{cfg.d}_{cfg.epochs}"] = train_sgns(cfg).loss
        out[f"gradnorm_{cfg.context}_{cfg.d}_{cfg.epochs}"] = train_sgns(cfg).grad_norm
    out.update({f"fin_{k}": v for k, v in finance_summary().items() if isinstance(v, (int, float))})
    return out


# --------------------------------------------------------------------------------------------
# Verification harness — every claim the page makes.
# --------------------------------------------------------------------------------------------


def _lbfgs(fun, x0):
    r = optimize.minimize(fun, x0, jac=True, method="L-BFGS-B",
                          options={"maxiter": 20_000, "maxfun": 40_000, "ftol": 1e-15, "gtol": 1e-11})
    return r.x


def test_three_optima():
    """Prop 1, Thm 1, Prop 2 on random fully observed instances: unconstrained L-BFGS on each objective lands on its
    closed form (the softmax's up to a row shift); φ is strictly concave; the three optima differ from PMI by row and
    column constants only (double-centering kills the difference); NCE − NS = log(k P_n(c)). With an unobserved cell,
    φ has no maximizer and L-BFGS drives that cell's score toward −∞."""
    rng = np.random.default_rng([SEED, 701])
    for _ in range(12):
        n, m = rng.integers(3, 7), rng.integers(3, 7)
        counts = _small_instance(rng, n, m) + 1
        k, alpha = float(rng.choice([1, 5, 15])), float(rng.uniform(0.5, 1.0))
        a, b = counts, negative_weights(counts, alpha, k)
        shape = counts.shape

        def f_ns(x):
            X = x.reshape(shape)
            return sgns_loss(X, counts, alpha, k), (((a + b) * sigmoid(X) - a)).ravel()

        def f_nce(x):
            X = x.reshape(shape)
            s = X - np.log(k * noise_distribution(counts, alpha))[None, :]
            return nce_loss(X, counts, alpha, k), (((a + b) * sigmoid(s) - a)).ravel()

        def f_soft(x):
            X = x.reshape(shape)
            p = np.exp(nlm.log_softmax(X))
            return softmax_loss(X, counts), (counts.sum(axis=1, keepdims=True) * p - counts).ravel()

        x_ns = _lbfgs(f_ns, np.zeros(n * m)).reshape(shape)
        x_nce = _lbfgs(f_nce, np.zeros(n * m)).reshape(shape)
        x_soft = _lbfgs(f_soft, np.zeros(n * m)).reshape(shape)
        assert np.abs(x_ns - sgns_optimum_dense(counts, alpha, k)).max() < 1e-6
        assert np.abs(x_nce - nce_optimum(counts, alpha, k)).max() < 1e-6
        assert np.abs(nce_optimum(counts, alpha, k) - softmax_optimum(counts)).max() < 1e-12
        rc = lambda Y: Y - Y.mean(axis=1, keepdims=True)  # noqa: E731
        assert np.abs(rc(x_soft) - rc(softmax_optimum(counts))).max() < 1e-6
        P = pmi_dense(counts)
        for opt in (softmax_optimum(counts), nce_optimum(counts, alpha, k), sgns_optimum_dense(counts, alpha, k)):
            Dm = opt - P
            Dm = Dm - Dm.mean(axis=1, keepdims=True) - Dm.mean(axis=0, keepdims=True) + Dm.mean()
            assert np.abs(Dm).max() < 1e-12
        off = nce_optimum(counts, alpha, k) - sgns_optimum_dense(counts, alpha, k)
        assert np.abs(off - np.log(k * noise_distribution(counts, alpha))[None, :]).max() < 1e-12
    xs = np.linspace(-8, 8, 1601)  # second differences resolve φ″ = −(a + b)σ(x)σ(−x) here, not in the tails
    for a_, b_ in ((3.0, 0.7), (0.2, 40.0), (1.0, 1.0)):
        phi = cell_objective(xs, a_, b_)
        assert np.all(np.diff(phi, 2) < 0)
        assert abs(xs[np.argmax(phi)] - math.log(a_ / b_)) < 0.01
    assert np.all(np.diff(cell_objective(np.linspace(-30, 30, 601), 0.0, 2.5)) < 0)  # a = 0: no maximizer
    counts = _small_instance(rng, 5, 4) + 1
    counts[1, 2] = 0
    a, b = counts, negative_weights(counts, ALPHA, K_NEG)
    x = _lbfgs(lambda x: (sgns_loss(x.reshape(a.shape), counts, ALPHA, K_NEG),
                          (((a + b) * sigmoid(x.reshape(a.shape)) - a)).ravel()), np.zeros(a.size)).reshape(a.shape)
    assert x[1, 2] < -10
    mask = counts > 0
    assert np.abs(x[mask] - sgns_optimum_dense(counts, ALPHA, K_NEG)[mask]).max() < 1e-6


def test_sgns_gradient():
    """The hand-written gradients — the population G = (a + b)σ(X) − a factorized as (GC, GᵀW), and the minibatch
    sampled loss's scatter — against Topic 5's complex-step (to 10⁻¹⁰ relative) and central differences (10⁻⁶)."""
    rng = np.random.default_rng([SEED, 702])
    counts = _small_instance(rng, 7, 7, zero_share=0.3)
    a, b = counts, negative_weights(counts, ALPHA, K_NEG)
    W, Cv = rng.normal(0, 0.5, (7, 3)), rng.normal(0, 0.5, (7, 3))
    _, gW, gC = _dense_factor_loss_grad(W, Cv, a, b)
    fW = lambda T: -np.sum(cell_objective(T @ Cv.T, a, b))  # noqa: E731
    fC = lambda T: -np.sum(cell_objective(W @ T.T, a, b))  # noqa: E731
    for g, f, th in ((gW, fW, W), (gC, fC, Cv)):
        cs = nlm.complex_step_grad(f, th)
        assert np.abs(cs - g).max() <= 1e-10 * np.abs(g).max()
        assert np.abs(nlm.central_diff_grad(f, th) - g).max() <= 1e-6 * np.abs(g).max()
    wi = rng.integers(0, 7, 9)
    ctx = rng.integers(0, 7, (9, K_NEG + 1))
    wi[1] = wi[0]  # a repeated word and a repeated context: the scatter must add
    ctx[2, 3] = ctx[0, 0]
    _, gW, gC = sampled_loss_grad(W, Cv, wi, ctx)
    for g, f, th in ((gW, lambda T: sampled_loss_grad(T, Cv, wi, ctx)[0], W),
                     (gC, lambda T: sampled_loss_grad(W, T, wi, ctx)[0], Cv)):
        cs = nlm.complex_step_grad(f, th)
        assert np.abs(cs - g).max() <= 1e-10 * np.abs(g).max()


def test_sampled_loss_is_unbiased():
    """Def 4: over its k i.i.d. negatives from P_n, the expected sampled loss of the whole pair multiset is L_NS —
    exactly (by linearity), and by Monte Carlo within 4 standard errors."""
    rng = np.random.default_rng([SEED, 703])
    counts = _small_instance(rng, 6, 6, zero_share=0.3)
    W, Cv = rng.normal(0, 0.7, (6, 3)), rng.normal(0, 0.7, (6, 3))
    X = W @ Cv.T
    pn = noise_distribution(counts, ALPHA)
    wi, ci = pair_multiset(sparse.coo_matrix(counts))
    exact = -np.sum(log_sigmoid(X[wi, ci])) - K_NEG * np.sum(log_sigmoid(-X[wi]) @ pn)
    assert abs(exact - sgns_loss(X, counts, ALPHA, K_NEG)) < 1e-10 * abs(exact)
    draws = []
    for _ in range(1500):
        negs = rng.choice(len(pn), size=(len(wi), K_NEG), p=pn)
        draws.append(sampled_loss_grad(W, Cv, wi, np.column_stack([ci, negs]))[0])
    draws = np.array(draws)
    assert abs(draws.mean() - exact) < 4 * draws.std(ddof=1) / math.sqrt(len(draws))


def test_logistic_pca_identity():
    """Thm 2 on random instances with unobserved cells: L_NS − Σ (a + b) H(y) = Σ (a + b) KL to 10⁻¹² relative, the
    KL agrees with scipy.stats.entropy and with the softplus Bregman divergence on observed cells; the curvature at
    x* is ab/(a + b), between ½ min(a, b) and min(a, b)."""
    rng = np.random.default_rng([SEED, 704])
    for _ in range(20):
        counts = _small_instance(rng, 8, 6, zero_share=0.4)
        a, b = counts, negative_weights(counts, ALPHA, K_NEG)
        X = rng.normal(0, 3, a.shape)
        lhs = sgns_loss(X, counts, ALPHA, K_NEG) - np.sum((a + b) * bernoulli_entropy(a / (a + b)))
        assert abs(lhs - logistic_pca_gap(X, a, b)) < 1e-12 * abs(lhs)
        y = a / (a + b)
        ref = np.array([stats.entropy([yi, 1 - yi], [sigmoid(xi), 1 - sigmoid(xi)]) for yi, xi in zip(y.ravel(), X.ravel())])
        assert np.abs(ref - bernoulli_kl(y, X).ravel()).max() < 1e-10
        obs = a > 0
        x_star = np.log(a[obs] / b[obs])
        assert np.abs(bregman_softplus(X[obs], x_star) - bernoulli_kl(y[obs], X[obs])).max() < 1e-10
        h = 1e-4
        f = lambda x: -cell_objective(x, a[obs], b[obs])  # noqa: E731
        curv = (f(x_star + h) - 2 * f(x_star) + f(x_star - h)) / h**2
        wt = local_weight(a[obs], b[obs])
        assert np.abs(curv - wt).max() < 1e-5 * max(1.0, wt.max())
        mn = np.minimum(a[obs], b[obs])
        assert np.all(wt <= mn + 1e-12) and np.all(wt >= mn / 2 - 1e-12)


def test_corpus_optimum():
    """Thm 1 on the corpus, at L2 and L8: log(a/b) equals Topic 6's PMI_α − log k on every observed pair to 10⁻¹²,
    φ′ vanishes there, and the unobserved majority (which has no maximizer) is over 95% of cells."""
    for ctx in CONTEXTS:
        C = sparse.coo_matrix(ds.cooc(ctx))
        _, n_w, pn = ds.marginals(C, ALPHA)
        a = C.data
        b = K_NEG * n_w[C.row] * pn[C.col]
        x = np.log(a / b)
        assert np.abs(x - sparse.coo_matrix(sgns_optimum(C, ALPHA, K_NEG)).data).max() < 1e-12
        assert np.abs(a * sigmoid(-x) - b * sigmoid(x)).max() < 1e-9 * (a + b).max()
        assert 1 - C.nnz / C.shape[0] ** 2 > 0.95


def test_fit_by_count_band():
    """Cor 1 on the corpus (controlled model): the correlation of w·c with PMI_α − log k rises and the mean absolute
    error falls across the count bands; frequent pairs are fitted closely; unobserved cells sit below observed ones;
    SGD stops above Thm 2's floor and short of a stationary point."""
    f = fit_summary()
    assert all(f["r"][i] < f["r"][i + 1] for i in range(len(COUNT_BANDS) - 1))
    assert all(f["mae"][i] > f["mae"][i + 1] for i in range(len(COUNT_BANDS) - 1))
    assert f["r"][-1] > 0.95 and f["r"][0] < 0.7
    assert f["x_unobs_mean"] < f["x_obs_mean"] - 1 and f["unobs_below_obs_median"] > 0.9
    assert f["kl_gap"] > 0 and f["grad_norm"] > 0
    bd = bias_direction()
    assert bd["u_purity"] == 1.0 and bd["v_purity"] == 1.0 and bd["negative"] == 1.0
    assert bd["energy_share"] > 0.8


def test_gauge_invariance():
    """Prop 3: every (WA, CA⁻ᵀ) leaves the product and the loss unchanged (to 10⁻¹²), every factorization of the same
    product arises this way, and the word Gram is preserved only for orthogonal A. On the corpus, the p-family and a
    random κ = 100 gauge keep the controlled model's product to 10⁻¹⁰ relative and move its benchmark scores."""
    rng = np.random.default_rng([SEED, 705])
    counts = _small_instance(rng, 9, 8, zero_share=0.3)
    for _ in range(10):
        W, Cv = rng.normal(0, 0.6, (9, 3)), rng.normal(0, 0.6, (8, 3))
        W2, C2 = random_gauge(W, Cv, 50.0, rng)
        L1, L2 = sgns_loss(W @ Cv.T, counts, ALPHA, K_NEG), sgns_loss(W2 @ C2.T, counts, ALPHA, K_NEG)
        assert abs(L1 - L2) < 1e-12 * abs(L1)
        A = np.linalg.pinv(W) @ W2
        assert np.abs(W @ A - W2).max() < 1e-10 and np.abs(Cv @ np.linalg.inv(A).T - C2).max() < 1e-10
        assert np.abs(W2 @ W2.T - W @ W.T).max() > 1e-3
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        assert np.abs((W @ Q) @ (W @ Q).T - W @ W.T).max() < 1e-12
    m = train_sgns(CONTROLLED)
    X = m.W @ m.C.T
    for q in (0.0, 0.5, 1.0):
        Wq, Cq = refactor(m.W, m.C, q)
        assert np.linalg.norm(Wq @ Cq.T - X) < 1e-10 * np.linalg.norm(X)
    Wr, Cr = random_gauge(m.W, m.C, GAUGE_KAPPA, np.random.default_rng([SEED, 730]))
    assert np.linalg.norm(Wr @ Cr.T - X) < 1e-10 * np.linalg.norm(X)
    g = gauge_cells()
    aucs = [g[v]["test_auc"] for v in ("p0", "p½", "p1", "random")]
    assert max(aucs) - min(aucs) > 0.02
    assert g["random"]["test_auc"] < g["p½"]["test_auc"]


def test_gradient_flow_conservation():
    """Thm 3: (i) one real minibatch step on the corpus changes D = WᵀW − CᵀC by exactly η²(∇_Wᵀ∇_W − ∇_Cᵀ∇_C), to
    10⁻¹⁰ relative in float64 — the first-order terms cancel; (ii) on a small instance, gradient descent's drift at a
    fixed horizon halves with η, so gradient flow (η → 0) conserves D."""
    cfg = CONTROLLED
    C = ds.cooc(cfg.context)
    _, _, pn = ds.marginals(C, cfg.alpha)
    wi_all, ci_all = pair_multiset(C)
    rng = np.random.default_rng([SEED, 706])
    m = train_sgns(cfg)
    W, Cv = m.W.copy(), m.C.copy()
    for _ in range(3):
        b = rng.integers(0, len(wi_all), cfg.batch)
        negs = rng.choice(len(pn), size=(cfg.batch, cfg.k), p=pn)
        _, gW, gC = sampled_loss_grad(W, Cv, wi_all[b], np.column_stack([ci_all[b], negs]))
        eta = cfg.lr0
        D = W.T @ W - Cv.T @ Cv
        W2, C2 = W - eta * gW, Cv - eta * gC
        change = W2.T @ W2 - C2.T @ C2 - D
        pred = eta**2 * (gW.T @ gW - gC.T @ gC)
        assert np.linalg.norm(change - pred) < 1e-10 * np.linalg.norm(W.T @ W)
        assert np.linalg.norm(pred) > 0
        W, Cv = W2, C2
    dr = drift_summary()
    assert dr[f"drift_{DRIFT_ETAS[0]:g}"] > 0
    assert 1.8 < dr["ratio_1"] < 2.2 and 1.8 < dr["ratio_2"] < 2.2, dr


def test_balanced_factorization():
    """Lemma 1 and Cor 2 on random instances: a balanced factorization is (UΣ^½R, VΣ^½R) with R orthogonal, its
    word Gram is UΣUᵀ (Topic 6's p = ½), and for a symmetric product ¼(W + C)(W + C)ᵀ = [X]₊ (Topic 6 Cor 1); an
    unbalanced factorization's gauge is not orthogonal. Dynamically: gradient descent from a balanced start on a
    symmetric, fully observed instance at full rank stays balanced to O(η) and converges to factors whose W + C keeps
    the positive part of the symmetric optimum PMI − log k."""
    rng = np.random.default_rng([SEED, 709])
    for _ in range(10):
        W0, C0 = rng.normal(size=(12, 4)), rng.normal(size=(10, 4))
        U, s, Vv = balanced_factors(W0, C0)
        R, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        W, Cv = U * np.sqrt(s) @ R, Vv * np.sqrt(s) @ R
        assert np.abs(W.T @ W - Cv.T @ Cv).max() < 1e-10
        A = gauge_of(W, Cv)
        assert np.abs(A @ A.T - np.eye(4)).max() < 1e-10
        assert np.abs(W @ W.T - (U * s) @ U.T).max() < 1e-10
        A0 = gauge_of(W0, C0)
        assert np.abs(A0 @ A0.T - np.eye(4)).max() > 1e-3
        M = rng.normal(size=(8, 8))
        Q, _ = np.linalg.qr(M)
        lam = rng.normal(0, 3, 5)
        X = (Q[:, :5] * lam) @ Q[:, :5].T
        Ws = Q[:, :5] * np.sqrt(np.abs(lam))
        Cs = Q[:, :5] * (np.sign(lam) * np.sqrt(np.abs(lam)))
        R5, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        Ws, Cs = Ws @ R5, Cs @ R5
        assert np.abs(Ws @ Cs.T - X).max() < 1e-10 and np.abs(Ws.T @ Ws - Cs.T @ Cs).max() < 1e-10
        pos = (Q[:, :5] * np.maximum(lam, 0)) @ Q[:, :5].T
        assert np.abs((Ws + Cs) @ (Ws + Cs).T / 4 - pos).max() < 1e-10
    sd = symmetric_dynamics()
    hi, lo = SYM_ETAS
    assert sd["n_negative_eigs"] > 0 and sd["n_positive_eigs"] > 0
    assert sd[f"fit_{hi:g}"] < 1e-3 and sd[f"fit_{lo:g}"] < 1e-3
    assert sd[f"imbalance_{lo:g}"] < sd[f"imbalance_{hi:g}"] and sd[f"wc_gap_{lo:g}"] < sd[f"wc_gap_{hi:g}"]
    assert sd[f"wc_gap_{lo:g}"] < 0.05, sd


def test_corpus_balance():
    """Thm 3 and Cor 2 on the corpus. word2vec's initialization is balanced up to a tiny D₀, but minibatch SGD at its
    step sizes is not gradient flow: Thm 3's η² terms make D grow over a thousandfold in the first epoch and keep
    growing. What cosines depend on is the gauge A of W = UΣ^½A, and it is near-orthogonal from the first epoch on —
    singular values within [0.7, 1.3], moving by under 0.05 between checkpoints — so W's cosines are close to the
    exactly balanced UΣ^½'s (test AUC within 0.03)."""
    bs = balance_summary()
    assert bs["D0_rel"] < 1e-3 and bs["D_growth_epoch1"] > 100
    assert 0.7 < bs["gauge_sv_min"] <= bs["gauge_sv_max"] < 1.3
    svs = np.array([h["gauge_sv"] for h in train_sgns(CONTROLLED).history if "gauge_sv" in h])
    assert np.ptp(svs, axis=0).max() < 0.05
    ctl = sgns_scores(CONTROLLED)
    bal = sgns_scores(CONTROLLED, "p½")
    assert abs(ctl["test_auc"] - bal["test_auc"]) < 0.03


def test_recovery_on_test_words():
    """§5: the controlled pair and the dev-selected grid model on Topic 6's test nouns. SGNS beats chance and the
    sparse count rows but, at this budget, not the SVD of the same matrix; Cor 1's mechanism holds — the benchmark
    nouns' rare pairs carry most of SPPMI₅'s squared mass and little of SGNS's local weight."""
    c = sgns_comparison()
    chance = ds.comparison()["test_chance_nn"]
    assert c["ctl_test_nn"] > 2 * chance and c["ctl_test_auc"] > 0.75
    assert c["ctl_vs_svd_auc_diff"] < 0
    assert c["mech_rare_share_sppmi_mass"] > c["mech_rare_share_sgns_weight"]
    longer = sgns_scores(CONTROLLED)["test_auc"]
    same_budget = sgns_scores(SGNSConfig("L8", 25))["test_auc"]
    assert longer > same_budget  # more epochs help: the 20-epoch grid model is less converged
    assert c["sel"] in {f"{g.context}|d={g.d}" for g in GRID}


def test_finance_findings():
    """§6, polarity isn't a direction. (i) The probe works when there is a direction: on random vectors with a planted
    polarity offset it scores ≥ 0.95, and on pure noise its mean over 20 draws stays within 0.06 of ½. (ii) On the
    corpus no window-trained representation — SGNS (W or (W + C)/2), Topic 6's selected SVD, sparse PPMI — beats chance
    by two standard errors at L2 or L8, where Topic 6's SO had no evidence (L2) or backwards evidence (L8).
    (iii) SGNS inherits Topic 6's antonym neighbors: *weak* is among *strong*'s top SGNS neighbors at L2 and L8."""
    rng = np.random.default_rng([SEED, 713])
    pw = ds.polarity_words()
    pos, neg = pw["positive"], pw["negative"]
    idx = ds.index()
    rows_p, rows_n = [idx[w] for w in pos], [idx[w] for w in neg]
    u = rng.normal(size=10)
    u /= np.linalg.norm(u)
    X = rng.normal(size=(len(ds.vocab()), 10))
    X[rows_p] += 2 * u
    X[rows_n] -= 2 * u
    assert polarity_probe(X, pos, neg) >= 0.95
    noise = [polarity_probe(rng.normal(size=(len(ds.vocab()), 10)), pos, neg) for _ in range(20)]
    assert abs(np.mean(noise) - 0.5) < 0.06
    f = finance_summary()
    assert f["probe_max"] < 0.5 + 2 * f["chance_se"]
    assert f["so_acc_L2"] == 0.5 and f["so_acc_L8"] < 0.5
    for ctx in CONTEXTS:
        assert "weak" in [w for w, _ in f[f"strong_neighbors_{ctx}"]]


def test_viz_payload():
    """The lab's data: Panel A's pairs span the count bands plus one unobserved pair, each with x* and weight for every
    (α, k); the fit view has one x per sampled pair per checkpoint; the gauge view's variants share the product."""
    p = viz_payload()
    assert p["pairs"][-1]["a"] == 0 and all(c["a"] > 0 for c in p["pairs"][:-1])
    for cell in p["pairs"][:-1]:
        for alpha in p["alphas"]:
            for k in p["ks"]:
                assert cell["x_star"][f"{alpha}|{k}"] is not None
    n = len(p["fit"]["x_star"])
    assert n == FIT_PER_BAND * len(COUNT_BANDS)
    assert all(len(h["x"]) == n for h in p["fit"]["checkpoints"])
    assert [h["epoch"] for h in p["fit"]["checkpoints"]] == list(CONTROLLED.checkpoints)
    assert len(json.dumps(p, separators=(",", ":"))) < 40_000


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "sgns-lab.json"


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
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
    print("\nSummary (train generate_corpus(300, 0); SGNS on Topic 6's pair multisets; benchmark = planted nouns):")
    for k, v in summary().items():
        print(f"  {k:44} {v:.4g}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else
              f"  {k:44} {v:.4f}" if isinstance(v, float) else f"  {k:44} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
