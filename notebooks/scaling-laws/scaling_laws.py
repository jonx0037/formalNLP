"""Neural scaling laws and compute-optimal training — the reference implementation for formalNLP's thirteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Data scaling from Zipf: a memorizing predictor errs exactly when the next token's type is new. Under the corpus's
   own Pitman–Yor(d, θ) generator that probability is (θ + d·E K_n)/(θ + n) in closed form, which falls as n^{−(1−d)}
   (Thm 1); the corpus itself shows the same exponent.
2. Model-size scaling: a Gaussian linear model with spectrum λ_k = k^{−a} and target power λ_k w_k² = k^{−b}. Keeping
   N features leaves the tail Σ_{k>N} k^{−b}, bracketed by the integral test at ≍ N^{−(b−1)} (Thm 2).
3. Data scaling in the same model: gradient flow for time t leaves Σ_k k^{−b} e^{−2t k^{−a}} ~ Γ((b−1)/a)/a · (2t)^{−(b−1)/a}
   (Thm 3). One-pass SGD's second moments close on the diagonal for Gaussian features (Prop 1), so its expected loss
   is computed exactly, with no Monte Carlo, and checked against Monte Carlo.
4. Counting compute: one training step of the toy decoder costs exactly 6N + 12·L·n·d matmul FLOPs per token, with N
   the weight count including the tied unembedding (Prop 2); counted by instrumenting the forward and backward passes.
5. Compute-optimal allocation: minimizing E + A/N^α + B/D^β on 6ND = C gives N* ∝ C^{β/(α+β)} in closed form (Thm 4);
   the solvable model predicts the exponent 1/(1 + a), and Chinchilla's three approaches recover it.
6. A toy-transformer grid on the finance corpus: IsoFLOP profiles at three budgets, Approaches 2 and 3, and the
   Kaplan–Chinchilla shift that comes from counting only non-embedding parameters.
7. Finance case study, data-constrained: a filing corpus is finite, so a domain model repeats it. Muennighoff et
   al.'s effective data D′ = U + U·R*(1 − e^{−R/R*}) against repeated-epoch runs, and the compute-optimal plan under
   a data cap.

Cross-checks: the decoder's gradient against central differences; the exact SGD recursion against Monte Carlo SGD;
Thm 4's closed form against a numeric argmin and its multiplier against a numeric derivative; Thm 1 against
Topic 1's Chinese-restaurant sampler.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/scaling-laws/scaling_laws.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, sparse, special

_HERE = Path(__file__).resolve().parent
for _dep in ("transformer-block-residual-stream", "contextual-embeddings-masked-lm", "perplexity-cross-entropy"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import contextual_embeddings_masked_lm as cm  # noqa: E402 — module objects: their test_* stay out of this harness
import perplexity_cross_entropy as pc  # noqa: E402
import transformer_block_residual_stream as tb  # noqa: E402

ak, fc, nlm = tb.ak, tb.fc, tb.nlm
zhv = pc.zhv

# ONE stream per random-instance claim and per trained model, each from default_rng([SEED, k]); k ≥ 1600 keeps this
# module's streams apart from Topics 5–12.
SEED = ak.SEED
Params = dict[str, np.ndarray]


def _r(x, nd: int = 3):
    return ak._r(x, nd)


def loglog_slope(x, y) -> float:
    """Ordinary least squares slope of log y on log x."""
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


# --------------------------------------------------------------------------------------------
# 1. Data scaling from Zipf: the missing mass of a memorizer.
# --------------------------------------------------------------------------------------------

PY_D, PY_THETA = fc.HEAPS_DISCOUNT, fc.PY_STRENGTH  # the corpus's content-word restaurants (d = 0.6, θ = 20)


def py_missing_mass(n, d: float = PY_D, theta: float = PY_THETA) -> np.ndarray:
    """Thm 1: the probability that draw n + 1 of a PY(d, θ) sequence is a new type, (θ + d·E K_n)/(θ + n) — linear in
    K_n, so its expectation is exact through Topic 1's closed form for E K_n. Equivalently
    Γ(θ+1)Γ(θ+d+n) / (Γ(θ+d)Γ(θ+n+1))."""
    n = np.asarray(n, float)
    return (theta + d * zhv.py_expected_types(n, d, theta)) / (theta + n)


def py_missing_mass_asymptote(n, d: float = PY_D, theta: float = PY_THETA) -> np.ndarray:
    """Γ(θ+1)/Γ(θ+d) · n^{−(1−d)}, the n → ∞ form of py_missing_mass."""
    return np.exp(special.gammaln(theta + 1) - special.gammaln(theta + d)) * np.asarray(n, float) ** (d - 1)


def zipf_exponent_of_discount(d: float) -> float:
    """The ranked frequencies of a PY(d, θ) measure decay as i^{−1/d}: Zipf's exponent s = 1/d."""
    return 1 / d


def hutter_exponent(s: float) -> float:
    """Hutter (2021, §3): a lookup learner on Zipf(s) data has expected error ≍ n^{−(1 − 1/s)}."""
    return 1 - 1 / s


MM_RESTAURANTS, MM_LENGTH = 1000, 2001
MM_CHECKPOINTS = (10, 30, 100, 300, 1000, 2000)


@lru_cache(maxsize=1)
def py_simulation() -> dict:
    """MM_RESTAURANTS independent PY(d, θ) sequences from Topic 1's sampler, one stream: at each checkpoint n, the
    share of sequences whose draw n + 1 is a new type, and the mean K_n, with standard errors."""
    rng = np.random.default_rng([SEED, 1600])
    new = np.zeros((MM_RESTAURANTS, len(MM_CHECKPOINTS)))
    K = np.zeros_like(new)
    for r in range(MM_RESTAURANTS):
        lab = zhv.pitman_yor_crp(MM_LENGTH, PY_D, PY_THETA, rng)
        k_run = np.maximum.accumulate(lab) + 1  # tables after each draw
        for j, n in enumerate(MM_CHECKPOINTS):
            K[r, j] = k_run[n - 1]
            new[r, j] = lab[n] == k_run[n - 1]
    se = lambda X: X.std(0, ddof=1) / math.sqrt(MM_RESTAURANTS)
    return {"n": np.array(MM_CHECKPOINTS), "new": new.mean(0), "new_se": se(new), "K": K.mean(0), "K_se": se(K)}


SCALE_DOCS = 3500  # the scaling corpus: generate_corpus(SCALE_DOCS, 0), ≈ 736k tokens
HOLDOUT = 65536  # its last HOLDOUT tokens are the test stream


@lru_cache(maxsize=1)
def corpus_tokens() -> tuple[str, ...]:
    return tuple(w for d in fc.generate_corpus(SCALE_DOCS, 0) for w in fc._tokens(d.text))


CURVE_BINS = 24
CURVE_FIT_FROM = 10_000  # fit the corpus exponent past the template words' saturation


@lru_cache(maxsize=1)
def corpus_learning_curve() -> dict:
    """The memorizer on the corpus's own text: the share of tokens in geometric bins of position whose type has not
    occurred before. Its log–log slope past CURVE_FIT_FROM estimates −(1 − d)."""
    toks = corpus_tokens()
    seen, flag = set(), np.zeros(len(toks))
    for i, w in enumerate(toks):
        if w not in seen:
            flag[i] = 1
            seen.add(w)
    edges = np.unique(np.geomspace(100, len(toks), CURVE_BINS + 1).astype(int))
    rate = np.array([flag[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    mid = np.sqrt(edges[:-1] * edges[1:])
    keep = mid >= CURVE_FIT_FROM
    return {"n": mid, "rate": rate, "slope": loglog_slope(mid[keep], rate[keep]), "types": len(seen),
            "tokens": len(toks)}


# --------------------------------------------------------------------------------------------
# 2 and 3. The solvable model: power-law spectrum, power-law target, one-pass SGD.
# --------------------------------------------------------------------------------------------

LIN_A, LIN_B = 1.5, 2.5  # λ_k = k^{−a}; λ_k w_k² = k^{−b}
LIN_M = 8192  # the model's dimension: the target lives in R^M, a size-N model keeps the first N coordinates
LIN_ETA = 0.25  # constant SGD step (η·tr Λ ≈ 0.65 < 2, stable)


def spectrum(a: float = LIN_A, b: float = LIN_B, M: int = LIN_M) -> tuple[np.ndarray, np.ndarray]:
    """(λ, s) with λ_k = k^{−a} and s_k = λ_k w_k² = k^{−b}, k = 1 … M."""
    k = np.arange(1, M + 1, dtype=float)
    return k ** -a, k ** -b


def tail_mass(N: int, b: float = LIN_B, M: int = LIN_M) -> float:
    """Σ_{k=N+1}^{M} k^{−b}: the loss a size-N model cannot remove (its approximation error)."""
    k = np.arange(N + 1, M + 1, dtype=float)
    return float((k ** -b).sum())


def tail_bracket(N: int, b: float, M: int) -> tuple[float, float]:
    """Thm 2's integral-test bracket for tail_mass: ∫_{N+1}^{M+1} x^{−b} dx ≤ Σ_{k=N+1}^{M} k^{−b} ≤ ∫_N^M x^{−b} dx."""
    F = lambda x: x ** (1 - b) / (b - 1)
    return F(N + 1) - F(M + 1), F(N) - F(M)


def gf_residual(t: float, a: float = LIN_A, b: float = LIN_B, M: int = LIN_M) -> float:
    """The loss of gradient flow from w = 0 at time t with every coordinate kept: Σ_k k^{−b} e^{−2t k^{−a}}."""
    k = np.arange(1, M + 1, dtype=float)
    return float((k ** -b * np.exp(-2 * t * k ** -a)).sum())


def gf_asymptote(t: float, a: float = LIN_A, b: float = LIN_B) -> float:
    """Thm 3: Γ((b−1)/a)/a · (2t)^{−(b−1)/a}, the integral ∫_0^∞ x^{−b} e^{−2t x^{−a}} dx."""
    return math.gamma((b - 1) / a) / a * (2 * t) ** (-(b - 1) / a)


def gf_bump(t: float, a: float, b: float) -> float:
    """max_x x^{−b} e^{−2t x^{−a}} = (b/(2ta))^{b/a} e^{−b/a}: the most a sum and its integral can differ by, for a
    unimodal summand (Thm 3's proof)."""
    return (b / (2 * t * a)) ** (b / a) * math.exp(-b / a)


def sgd_losses(Ns, checkpoints, eta: float = LIN_ETA, a: float = LIN_A, b: float = LIN_B,
               M: int = LIN_M) -> np.ndarray:
    """Prop 1: the expected population loss of one-pass SGD (x ~ N(0, Λ), y = ⟨w*, x⟩, w₀ = 0, the size-N model fitting
    the first N coordinates) after each checkpoint's number of samples. The error e = w − w*_{1:N} has diagonal second
    moments S_k that evolve on their own: S_k ← (1 − 2ηλ_k + 2η²λ_k²) S_k + η²λ_k (Σ_j λ_j S_j + tail(N)), and the
    loss is Σ_k λ_k S_k + tail(N). All sizes advance together; returns (len(Ns), len(checkpoints))."""
    lam, s = spectrum(a, b, M)
    Ns = np.asarray(Ns)
    Nm = int(Ns.max())
    keep = (np.arange(Nm)[None] < Ns[:, None]).astype(float)
    lamN = lam[None, :Nm]
    S = (s / lam)[None, :Nm] * keep  # E[e_k²] at w = 0 is w_k²
    noise = np.array([tail_mass(int(n), b, M) for n in Ns])
    shrink = (1 - 2 * eta * lamN + 2 * eta**2 * lamN**2) * keep
    pump = eta**2 * lamN * keep
    out = np.empty((len(Ns), len(checkpoints)))
    want = {int(t): j for j, t in enumerate(checkpoints)}
    for t in range(1, int(max(checkpoints)) + 1):
        L = (lamN * S).sum(1) + noise
        S = shrink * S + pump * L[:, None]
        if t in want:
            out[:, want[t]] = (lamN * S).sum(1) + noise
    return out


def sgd_monte_carlo(rng: np.random.Generator, N: int, checkpoints, reps: int, eta: float = LIN_ETA,
                    a: float = LIN_A, b: float = LIN_B, M: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """reps independent one-pass SGD runs on fresh Gaussian samples (the definition sgd_losses solves): the mean loss
    and its standard error at each checkpoint."""
    lam, s = spectrum(a, b, M)
    ws = np.sqrt(s / lam)
    w = np.zeros((reps, N))
    tail = tail_mass(N, b, M)
    out, se = [], []
    for t in range(1, int(max(checkpoints)) + 1):
        x = rng.normal(size=(reps, M)) * np.sqrt(lam)
        resid = (x[:, :N] * w).sum(1) - x @ ws
        w -= eta * resid[:, None] * x[:, :N]
        if t in checkpoints:
            loss = (lam[:N] * (w - ws[:N]) ** 2).sum(1) + tail
            out.append(loss.mean())
            se.append(loss.std(ddof=1) / math.sqrt(reps))
    return np.array(out), np.array(se)


LIN_NS = tuple(int(round(2 ** (3 + i / 2))) for i in range(15))  # 8 … 1024 in half-octaves
LIN_D_MAX = 2**16
LIN_DS = tuple(int(x) for x in np.unique(np.round(np.geomspace(16, LIN_D_MAX, 49))))


@lru_cache(maxsize=1)
def lin_grid() -> np.ndarray:
    """Expected SGD loss L(N, D) on the LIN_NS × LIN_DS grid."""
    return sgd_losses(LIN_NS, LIN_DS)


def lin_loss(N_index: int, D: float) -> float:
    """L(N, D) between checkpoints: linear interpolation of log L in log D."""
    return float(np.exp(np.interp(np.log(D), np.log(LIN_DS), np.log(lin_grid()[N_index]))))


# --------------------------------------------------------------------------------------------
# 4. Counting compute: the toy decoder and its FLOPs.
# --------------------------------------------------------------------------------------------

TOY_V, TOY_CTX, TOY_BATCH, TOY_LAYERS = 512, 32, 32, 2
TOY_WIDTHS = (6, 8, 12, 16, 24, 32, 48)
LR_SCALE, WARMUP = 1.28, 0.1  # Adam's peak rate is LR_SCALE/d (a 1/width rule fixed on a small sweep); 10% warmup


def head_shape(d: int) -> tuple[int, int]:
    """(heads, head width): one head per 16 dimensions, at least one."""
    H = max(1, d // 16)
    return H, d // H


def matmul_params(d: int, layers: int = TOY_LAYERS, V: int = TOY_V) -> int:
    """N: the weights a token's matmuls touch — 12d² per block (Wq, Wk, Wv, Wo; the MLP's d × 4d and 4d × d) and the
    tied d × V embedding, which is also the unembedding. Biases (7d per block) and the position table are not counted."""
    return layers * 12 * d * d + V * d


def nonembedding_params(d: int, layers: int = TOY_LAYERS) -> int:
    """Kaplan et al.'s N: the same count without the embedding."""
    return layers * 12 * d * d


def step_flops(d: int, B: int = TOY_BATCH, n: int = TOY_CTX, layers: int = TOY_LAYERS, V: int = TOY_V) -> int:
    """Prop 2: one training step's matmul FLOPs, B·n tokens × (6N + 12·L·n·d). The 6 is 2 (forward) + 4 (backward: one
    product for the input gradient and one for the weight gradient per weight); 12·L·n·d is QKᵀ and AV, forward and
    backward, which 6N leaves out."""
    return B * n * (6 * matmul_params(d, layers, V) + 12 * layers * n * d)


def attention_share(d: int, n: int = TOY_CTX, layers: int = TOY_LAYERS, V: int = TOY_V) -> float:
    """The attention FLOPs as a share of 6N: 12·L·n·d / 6N."""
    return 12 * layers * n * d / (6 * matmul_params(d, layers, V))


def init_decoder(rng: np.random.Generator, d: int, dtype=np.float32) -> tuple[Params, list]:
    """Tied token table E (V, d) at N(0, 1), a learned position table P (n, d) at N(0, 0.1²) (Topic 12's), and
    TOY_LAYERS of Topic 11's pre-LN blocks with an MLP of width 4d."""
    H, dh = head_shape(d)
    P = {"E": rng.normal(0, 1, (TOY_V, d)), "P": rng.normal(0, 0.1, (TOY_CTX, d))}
    blocks = [tb.init_block(rng, d, H, dh, 4 * d) for _ in range(TOY_LAYERS)]
    P = {k: v.astype(dtype) for k, v in P.items()}
    blocks = [{s: {k: v.astype(dtype) for k, v in blk[s].items()} for s in blk} for blk in blocks]
    return P, blocks


def decoder_forward(P: Params, blocks: list, tok: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """x_0 = E[tok] + P, then pre-LN blocks with causal attention, then the final LayerNorm: (LN(x_L), its radii,
    the per-block caches)."""
    x = P["E"][tok] + P["P"][None, : tok.shape[1]]
    caches = []
    for blk in blocks:
        y, r1 = tb._ln_forward(x)
        a, ca = tb.mha_forward(blk["attn"], y, causal=True)
        x = x + a
        z, r2 = tb._ln_forward(x)
        m, cmc = tb.mlp_forward(blk["mlp"], z)
        x = x + m
        caches.append((y, r1, ca, z, r2, cmc))
    out, r = tb._ln_forward(x)
    return out, r, caches


def decoder_log_probs(P: Params, blocks: list, tok: np.ndarray) -> np.ndarray:
    """log q(x_{i+1} | x_{≤i}) at every position: the tied unembedding LN(x_L) Eᵀ/√d."""
    out, _, _ = decoder_forward(P, blocks, tok)
    return nlm.log_softmax(out @ P["E"].T / math.sqrt(out.shape[-1]))


def decoder_loss_grad(P: Params, blocks: list, tok: np.ndarray, tgt: np.ndarray) -> tuple[float, Params, list]:
    """The mean next-token cross-entropy over every position, with its gradient (test_decoder_gradients)."""
    out, r, caches = decoder_forward(P, blocks, tok)
    d = out.shape[-1]
    o = out.reshape(-1, d)
    lp = nlm.log_softmax(o @ P["E"].T / math.sqrt(d))
    t = tgt.ravel()
    cnt = len(t)
    loss = float(-lp[np.arange(cnt), t].sum() / cnt)
    g = np.exp(lp)
    g[np.arange(cnt), t] -= 1
    g /= cnt
    G = {"E": g.T @ o / math.sqrt(d)}
    gx = tb.ln_backward((g @ P["E"] / math.sqrt(d)).reshape(out.shape), out, r)
    grads = []
    for blk, (y, r1, ca, z, r2, cmc) in zip(reversed(blocks), reversed(caches)):
        dz, Gm = cm.mlp_backward(blk["mlp"], gx, cmc)
        gx = gx + tb.ln_backward(dz, z, r2)
        dy, Ga = tb.mha_backward(blk["attn"], gx, ca)
        gx = gx + tb.ln_backward(dy, y, r1)
        grads.append({"attn": Ga, "mlp": Gm})
    G["P"] = gx.sum(0)
    N = tok.size  # the token-table scatter as one sparse product (CLAUDE.md's csc pattern)
    G["E"] = G["E"] + sparse.csc_matrix((np.ones(N, out.dtype), tok.ravel(), np.arange(N + 1)),
                                        shape=(TOY_V, N)) @ np.asarray(gx).reshape(-1, d)
    return loss, G, grads[::-1]


class _Counted(np.ndarray):
    """An array that adds 2·k FLOPs per output entry of every matmul it takes part in (k the contracted length)."""

    flops = 0

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = [np.asarray(x) if isinstance(x, _Counted) else x for x in inputs]
        if "out" in kwargs:
            kwargs["out"] = tuple(np.asarray(x) if isinstance(x, _Counted) else x for x in kwargs["out"])
        res = getattr(ufunc, method)(*args, **kwargs)
        if ufunc is np.matmul and method == "__call__":
            _Counted.flops += 2 * np.shape(args[0])[-1] * res.size
        return res.view(_Counted) if isinstance(res, np.ndarray) else res


def counted_step_flops(d: int) -> int:
    """Run one decoder_loss_grad on a real batch with every parameter instrumented; the matmul FLOPs it took."""
    rng = np.random.default_rng([SEED, 1610, d])
    P, blocks = init_decoder(rng, d, np.float64)
    view = lambda D: {k: v.view(_Counted) for k, v in D.items()}
    P = view(P)
    blocks = [{s: view(blk[s]) for s in blk} for blk in blocks]
    tok, tgt = toy_batch(rng, TOY_BATCH)
    _Counted.flops = 0
    decoder_loss_grad(P, blocks, tok, tgt)
    return _Counted.flops


# --------------------------------------------------------------------------------------------
# 5. Compute-optimal allocation.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LossFit:
    """L(N, D) = E + A/N^α + B/D^β (Hoffmann et al.'s Eq. 2)."""
    E: float
    A: float
    B: float
    alpha: float
    beta: float

    def loss(self, N, D):
        return self.E + self.A / np.asarray(N, float) ** self.alpha + self.B / np.asarray(D, float) ** self.beta

    @property
    def a(self) -> float:
        """The exponent of N* ∝ C^a."""
        return self.beta / (self.alpha + self.beta)


def compute_optimal(fit: LossFit, C: float, flops_per: float = 6.0) -> dict:
    """Thm 4: the minimizer of fit.loss on flops_per·N·D = C. With G = (αA/(βB))^{1/(α+β)} and c = C/flops_per,
    N* = G·c^{β/(α+β)}, D* = c^{α/(α+β)}/G, and the frontier L*(C) = E + K·c^{−αβ/(α+β)}; the multiplier of the budget
    constraint is ν = −dL*/dC."""
    al, be = fit.alpha, fit.beta
    G = (al * fit.A / (be * fit.B)) ** (1 / (al + be))
    c = C / flops_per
    N = G * c ** (be / (al + be))
    D = c ** (al / (al + be)) / G
    L = float(fit.loss(N, D))
    nu = al * be / (al + be) * (L - fit.E) / C  # −dL*/dC: L* − E ∝ C^{−αβ/(α+β)}
    return {"N": float(N), "D": float(D), "L": L, "nu": nu, "tokens_per_param": float(D / N)}


# Published fits of Eq. 2 (nats, tokens, parameters). Hoffmann et al. (2022, Eq. 10) print the rounded values;
# Besiroglu et al. (2024, Eq. 4) recover the unrounded ones from the TeX source and refit (their Eq. 3, Table 1).
PUBLISHED_FITS = {
    "hoffmann_rounded": LossFit(1.69, 406.4, 410.7, 0.34, 0.28),
    "hoffmann_unrounded": LossFit(1.6934, 406.4, 410.7, 0.3392, 0.2849),
    "besiroglu": LossFit(1.8172, 482.01, 2085.43, 0.3478, 0.3658),
}
GOPHER_C = 5.76e23  # Gopher's training budget in FLOPs, at which Hoffmann et al. sized Chinchilla


def huber(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


HUBER_DELTA = 1e-3  # Hoffmann et al.'s δ


def fit_loss_law(N, D, L, grid: dict, delta: float = HUBER_DELTA) -> tuple[LossFit, float]:
    """Approach 3: minimize Σ Huber_δ(LSE(a − α log N, b − β log D, e) − log L) over (a, b, e, α, β) by L-BFGS from
    every start in the grid (A = e^a, B = e^b, E = e^e). Returns the best fit and its objective."""
    lN, lD, lL = (np.log(np.asarray(x, float)) for x in (N, D, L))

    def obj(p):
        a, b, e, al, be = p
        pred = np.logaddexp(np.logaddexp(a - al * lN, b - be * lD), e)
        r = pred - lL
        w = np.abs(r) <= delta
        f = huber(r, delta).sum()
        dr = np.where(w, r, delta * np.sign(r))
        ea, eb, ee = np.exp(a - al * lN - pred), np.exp(b - be * lD - pred), np.exp(e - pred)
        g = np.array([(dr * ea).sum(), (dr * eb).sum(), (dr * ee).sum(), -(dr * ea * lN).sum(), -(dr * eb * lD).sum()])
        return f, g

    best = None
    for a0 in grid["a"]:
        for b0 in grid["b"]:
            for e0 in grid["e"]:
                for al0 in grid["alpha"]:
                    for be0 in grid["beta"]:
                        res = optimize.minimize(obj, [a0, b0, e0, al0, be0], jac=True, method="L-BFGS-B")
                        if best is None or res.fun < best.fun:
                            best = res
    a, b, e, al, be = best.x
    return LossFit(math.exp(e), math.exp(a), math.exp(b), al, be), float(best.fun)


def parabola_minimum(logN: np.ndarray, L: np.ndarray) -> float:
    """Approach 2's step: the vertex of the least-squares quadratic of L in log N, as N (nan if it opens downward)."""
    p = np.polyfit(logN, L, 2)
    return math.exp(-p[1] / (2 * p[0])) if p[0] > 0 else float("nan")


# The solvable model's fits.

LIN_C = tuple(float(c) for c in np.geomspace(2**12, 2**24, 25))  # IsoFLOP budgets, C = N·D (an SGD step costs ∝ N)
LIN_FIT_GRID = {"a": (0.0, 2.0), "b": (0.0, 2.0), "e": (-12.0, -6.0), "alpha": (0.5, 1.5), "beta": (0.5, 1.5)}


def lin_isoflop(C: float) -> tuple[np.ndarray, np.ndarray]:
    """The sizes N whose D = C/N lies inside the grid, and their losses."""
    Ns, Ls = [], []
    for i, n in enumerate(LIN_NS):
        D = C / n
        if LIN_DS[0] <= D <= LIN_DS[-1]:
            Ns.append(n)
            Ls.append(lin_loss(i, D))
    return np.array(Ns, float), np.array(Ls)


@lru_cache(maxsize=1)
def lin_fits() -> dict:
    """Chinchilla's three approaches on the solvable model, against Thm 4's prediction a = 1/(1 + a_λ) = 0.4.
    Approach 1: the size with the least loss at each budget (constant-step SGD makes every checkpoint an end of
    training). Approach 2: the vertex of a quadratic in log N through the five sizes nearest it, at each budget.
    Approach 3: the Huber fit of Eq. 2 to the whole grid."""
    star1, star2 = [], []
    for C in LIN_C:
        Ns, Ls = lin_isoflop(C)
        j = int(np.argmin(Ls))
        star1.append(Ns[j])
        lo = max(0, min(j - 2, len(Ns) - 5))
        star2.append(parabola_minimum(np.log(Ns[lo:lo + 5]), np.log(Ls[lo:lo + 5])))
    NN, DD = np.meshgrid(LIN_NS, LIN_DS, indexing="ij")
    fit, obj = fit_loss_law(NN.ravel(), DD.ravel(), lin_grid().ravel(), LIN_FIT_GRID)
    pred = np.log(fit.loss(NN, DD))
    resid = pred - np.log(lin_grid())
    return {"a1": loglog_slope(LIN_C, star1), "a2": loglog_slope(LIN_C, star2), "fit": fit, "a3": fit.a,
            "star1": np.array(star1), "star2": np.array(star2), "alpha_pred": LIN_B - 1,
            "beta_pred": (LIN_B - 1) / LIN_A, "a_pred": 1 / (1 + LIN_A), "resid_rms": float(np.sqrt((resid**2).mean())),
            "resid_max": float(np.abs(resid).max())}


# --------------------------------------------------------------------------------------------
# 6. The toy-transformer grid on the finance corpus.
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def toy_vocab() -> dict[str, int]:
    """The TOY_V − 1 commonest types of the training tokens; everything else is [UNK] (id TOY_V − 1)."""
    toks = corpus_tokens()[:-HOLDOUT]
    return {w: i for i, (w, _) in enumerate(Counter(toks).most_common(TOY_V - 1))}


@lru_cache(maxsize=1)
def toy_stream() -> np.ndarray:
    v = toy_vocab()
    return np.array([v.get(w, TOY_V - 1) for w in corpus_tokens()])


def train_tokens() -> int:
    return len(toy_stream()) - HOLDOUT


def toy_batch(rng: np.random.Generator, batch: int, unique: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """batch windows of TOY_CTX + 1 consecutive training tokens at uniform starts (within the first `unique` tokens,
    if given): (inputs, next-token targets)."""
    lim = train_tokens() if unique is None else unique
    st = rng.integers(0, lim - TOY_CTX, size=batch)
    X = toy_stream()[st[:, None] + np.arange(TOY_CTX + 1)]
    return X[:, :-1], X[:, 1:]


TEST_WINDOWS = 1024


@lru_cache(maxsize=1)
def held_out_windows() -> np.ndarray:
    """TEST_WINDOWS windows of the held-out stream at stride 64, shared by every model (paired comparisons)."""
    st = train_tokens() + np.arange(TEST_WINDOWS) * (HOLDOUT // TEST_WINDOWS)
    return toy_stream()[st[:, None] + np.arange(TOY_CTX + 1)]


@dataclass(frozen=True)
class RunConfig:
    d: int
    steps: int
    unique: int | None = None  # train on the first `unique` training tokens only (repeated epochs)
    seed: int = 0


@lru_cache(maxsize=None)
def train_run(cfg: RunConfig) -> dict:
    """Adam (Topic 5's constants) in float32 at peak rate LR_SCALE/d, linear warmup over WARMUP of the steps and
    linear decay to 0 at the run's own end (the schedule matched to its horizon). The initialization stream depends on
    (seed, d); the batch stream on seed only, so every size sees the same windows in the same order. Returns the
    held-out cross-entropy (nats per token) and the run's size and data."""
    rng_init = np.random.default_rng([SEED, 1620, cfg.seed, cfg.d])
    rng_data = np.random.default_rng([SEED, 1621, cfg.seed])
    P, blocks = init_decoder(rng_init, cfg.d)
    leaves = [(P, k) for k in P] + [(blk[s], k) for blk in blocks for s in blk for k in blk[s]]
    m1 = [np.zeros_like(D[k]) for D, k in leaves]
    m2 = [np.zeros_like(D[k]) for D, k in leaves]
    peak, warm = LR_SCALE / cfg.d, max(1, int(WARMUP * cfg.steps))
    for step in range(1, cfg.steps + 1):
        tok, tgt = toy_batch(rng_data, TOY_BATCH, cfg.unique)
        _, G, Gb = decoder_loss_grad(P, blocks, tok, tgt)
        grads = [G[k] for k in P] + [Gb[i][s][k] for i, blk in enumerate(blocks) for s in blk for k in blk[s]]
        c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
        lr = peak * min(step / warm, (cfg.steps - step + 1) / (cfg.steps - warm + 1))
        for j, ((D, k), g) in enumerate(zip(leaves, grads)):
            m1[j] = nlm.ADAM_B1 * m1[j] + (1 - nlm.ADAM_B1) * g
            m2[j] = nlm.ADAM_B2 * m2[j] + (1 - nlm.ADAM_B2) * g**2
            D[k] -= lr * (m1[j] / c1) / (np.sqrt(m2[j] / c2) + nlm.ADAM_EPS)
    W = held_out_windows()
    lp = decoder_log_probs(P, blocks, W[:, :-1])
    test = float(-np.take_along_axis(lp, W[:, 1:, None], -1).mean())
    D = cfg.steps * TOY_BATCH * TOY_CTX
    return {"d": cfg.d, "N": matmul_params(cfg.d), "N_ne": nonembedding_params(cfg.d), "D": D,
            "C": 6 * matmul_params(cfg.d) * D, "loss": test}


TOY_GRID = {1e10: (6, 8, 12, 16, 24), 2.5e10: (8, 12, 16, 24, 32), 6.3e10: (12, 16, 24, 32, 48)}  # C = 6ND
TOY_BUDGETS = tuple(TOY_GRID)


def budget_steps(C: float, d: int) -> int:
    return int(round(C / (6 * matmul_params(d) * TOY_BATCH * TOY_CTX)))


@lru_cache(maxsize=1)
def isoflop_grid() -> tuple[dict, ...]:
    """Every (budget, width) run, with its budget attached."""
    return tuple({**train_run(RunConfig(d, budget_steps(C, d))), "budget": C} for C, ds in TOY_GRID.items()
                 for d in ds)


SEED_CHECK = (1, 2)  # extra seeds of one run: the noise floor of a single point


@lru_cache(maxsize=1)
def seed_spread() -> dict:
    """The middle budget's run at d = 12, retrained with SEED_CHECK's seeds (new initialization, new batches)."""
    d, s = 12, budget_steps(TOY_BUDGETS[1], 12)
    losses = [train_run(RunConfig(d, s, seed=k))["loss"] for k in (0, *SEED_CHECK)]
    return {"losses": losses, "sd": float(np.std(losses, ddof=1))}


TOY_FIT_GRID = {"a": (0.0, 5.0, 10.0), "b": (0.0, 5.0, 10.0), "e": (-1.0, 0.0, 0.5), "alpha": (0.0, 0.5, 1.0, 2.0),
                "beta": (0.0, 0.5, 1.0, 2.0)}


@lru_cache(maxsize=1)
def toy_fits() -> dict:
    """Approach 2 on each budget, in Chinchilla's count N (weights with the tied embedding) and in Kaplan's N_ne
    (without it, against Kaplan's compute 6·N_ne·D of the optimal run), and Approach 3 on every run."""
    runs = isoflop_grid()
    out = {"budgets": [], "N_star": [], "Nne_star": [], "Cne_star": [], "argmin_d": []}
    for C in TOY_BUDGETS:
        rs = [r for r in runs if r["budget"] == C]
        N = np.array([r["N"] for r in rs], float)
        Nne = np.array([r["N_ne"] for r in rs], float)
        L = np.array([r["loss"] for r in rs])
        n_star = parabola_minimum(np.log(N), L)
        ne_star = parabola_minimum(np.log(Nne), L)
        D_star = C / (6 * n_star)
        out["budgets"].append(C)
        out["N_star"].append(n_star)
        out["Nne_star"].append(ne_star)
        out["Cne_star"].append(6 * ne_star * D_star)
        out["argmin_d"].append(rs[int(np.argmin(L))]["d"])
    out["a_chinchilla"] = loglog_slope(out["budgets"], out["N_star"])
    out["a_kaplan"] = loglog_slope(out["Cne_star"], out["Nne_star"])
    fit, obj = fit_loss_law([r["N"] for r in runs], [r["D"] for r in runs], [r["loss"] for r in runs], TOY_FIT_GRID)
    out.update({"fit": fit, "a3": fit.a})
    return out


def embedding_elasticity(N_ne, c: float, p: float):
    """ρ = d log N_ne / d log N for a family with N = N_ne + c·N_ne^p (embeddings growing as the p-th power of the
    rest): ρ = (1 + c·N_ne^{p−1}) / (1 + p·c·N_ne^{p−1}), between 1 (embeddings negligible) and 1/p (dominant)."""
    x = c * np.asarray(N_ne, float) ** (p - 1)
    return (1 + x) / (1 + p * x)


def kaplan_exponent(a: float, rho):
    """Prop 3: if N* ∝ C^a in total parameters, the non-embedding optimum against Kaplan's compute 6·N_ne·D has local
    exponent d log N_ne* / d log C_ne = ρa / (ρa + 1 − a)."""
    rho = np.asarray(rho, float)
    return rho * a / (rho * a + 1 - a)


# Kaplan et al.'s regime: GPT-2's vocabulary and context, d_model = KAPLAN_ASPECT · n_layer, N_ne = 12·n_layer·d².
KAPLAN_V, KAPLAN_CTX, KAPLAN_ASPECT = 50257, 1024, 64
KAPLAN_C = tuple(float(c) for c in np.geomspace(1e15, 1e23, 33))


def kaplan_family_ne(N: float) -> float:
    """The non-embedding share of a total N in that family: solve N = N_ne + (V + n_ctx)·d with d = (A·N_ne/12)^{1/3}."""
    c = (KAPLAN_V + KAPLAN_CTX) * (KAPLAN_ASPECT / 12) ** (1 / 3)
    return optimize.brentq(lambda x: x + c * x ** (1 / 3) - N, 1e-9, N)


@lru_cache(maxsize=1)
def kaplan_shift() -> dict:
    """Besiroglu et al.'s fit, sized by total parameters, read the way Kaplan et al. measured: N_ne* against
    C_ne = 6·N_ne*·D*. The local exponent from finite differences, against Prop 3's ρa/(ρa + 1 − a)."""
    fit = PUBLISHED_FITS["besiroglu"]
    c = (KAPLAN_V + KAPLAN_CTX) * (KAPLAN_ASPECT / 12) ** (1 / 3)
    rows = []
    for C in KAPLAN_C:
        o = compute_optimal(fit, C)
        ne = kaplan_family_ne(o["N"])
        rows.append((C, o["N"], ne, 6 * ne * o["D"]))
    C, N, Ne, Cne = (np.array(x) for x in zip(*rows))
    local = np.gradient(np.log(Ne), np.log(Cne))
    predicted = kaplan_exponent(fit.a, embedding_elasticity(Ne, c, 1 / 3))
    small = C <= 1e18
    return {"C": C, "N": N, "N_ne": Ne, "C_ne": Cne, "local": local, "predicted": predicted,
            "share": 1 - Ne / N, "a_small": loglog_slope(Cne[small], Ne[small]), "a": fit.a,
            "limit": kaplan_exponent(fit.a, 3.0)}


# --------------------------------------------------------------------------------------------
# 7. Finance case study: a finite filing corpus.
# --------------------------------------------------------------------------------------------

R_STAR = 15.387756  # Muennighoff et al. (2023, App. A): the e-folding number of repeated epochs, R_D*
REPEAT_EPOCHS = (4, 16, 64)
REPEAT_D = 12  # the repeated runs: the middle budget's d = 12 model, its data split into fewer unique tokens


def effective_data(D, U, r_star: float = R_STAR):
    """Muennighoff et al.'s Eq. 5: D tokens drawn from U unique ones are worth D′ = U + U·R*(1 − e^{−R/R*}) fresh
    tokens, with R = D/U − 1 the repeated epochs."""
    D, U = np.asarray(D, float), np.asarray(U, float)
    return U + U * r_star * (1 - np.exp(-(D / U - 1) / r_star))


@lru_cache(maxsize=1)
def repetition_runs() -> dict:
    """The fresh-data run and the same compute spent on D/R unique tokens, R ∈ REPEAT_EPOCHS. Each run's loss beside
    the toy Approach-3 fit's prediction at D (fresh) and at Muennighoff's D′."""
    steps = budget_steps(TOY_BUDGETS[1], REPEAT_D)
    fresh = train_run(RunConfig(REPEAT_D, steps))
    D, N = fresh["D"], fresh["N"]
    fit = toy_fits()["fit"]
    rows = [{"epochs": 1, "unique": D, "loss": fresh["loss"], "D_eff": float(D), "pred": float(fit.loss(N, D))}]
    for R in REPEAT_EPOCHS:
        U = D // R
        r = train_run(RunConfig(REPEAT_D, steps, unique=U))
        De = float(effective_data(D, U))
        rows.append({"epochs": R, "unique": U, "loss": r["loss"], "D_eff": De, "pred": float(fit.loss(N, De))})
    return {"N": N, "D": D, "rows": rows}


def constrained_optimum(fit: LossFit, C: float, U: float, r_star: float = R_STAR) -> dict:
    """The compute-optimal size when only U unique tokens exist: minimize fit.loss(N, D′(C/6N, U)) over N, repeating
    data once D = C/6N exceeds U (Muennighoff's data-side decay; their parameter-side decay is not modeled)."""
    def f(logN):
        N = math.exp(logN)
        D = C / (6 * N)
        return float(fit.loss(N, effective_data(D, U, r_star) if D > U else D))
    free = compute_optimal(fit, C)
    res = optimize.minimize_scalar(f, bounds=(math.log(free["N"]) - 3, math.log(free["N"]) + 6), method="bounded",
                                   options={"xatol": 1e-10})
    N = math.exp(res.x)
    D = C / (6 * N)
    return {"N": N, "D": D, "epochs": D / U, "L": float(res.fun), "free": free}


CAP_U = tuple(float(u) for u in np.geomspace(2e10, 2e12, 21))  # unique-token grid for the lab's cap view
BBG_U = 709e9  # BloombergGPT's available tokens (Wu et al. 2023, §4): 363B financial + 345B general, ≈ 709B
BBG_C = 6 * 50.6e9 * 569e9  # its run: 50.6B parameters on 569B tokens, in 6ND


# --------------------------------------------------------------------------------------------
# 8. The lab's data.
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def summary() -> dict:
    lf, tf, sim = lin_fits(), toy_fits(), py_simulation()
    out = {"mm_new_1000": float(sim["new"][MM_CHECKPOINTS.index(1000)]),
           "mm_exact_1000": float(py_missing_mass(1000)), "corpus_slope": corpus_learning_curve()["slope"],
           "lin_alpha": lf["fit"].alpha, "lin_beta": lf["fit"].beta, "lin_a1": lf["a1"], "lin_a2": lf["a2"],
           "lin_a3": lf["a3"], "lin_resid_rms": lf["resid_rms"], "toy_a_chinchilla": tf["a_chinchilla"],
           "toy_a_kaplan": tf["a_kaplan"], "toy_a3": tf["a3"], "toy_alpha": tf["fit"].alpha, "toy_beta": tf["fit"].beta,
           "toy_E": tf["fit"].E, "seed_sd": seed_spread()["sd"], "kaplan_a_small": kaplan_shift()["a_small"],
           "kaplan_limit": kaplan_shift()["limit"]}
    for r in repetition_runs()["rows"]:
        out[f"repeat_{r['epochs']}_loss"] = r["loss"]
    for name, fit in PUBLISHED_FITS.items():
        o = compute_optimal(fit, GOPHER_C)
        out[f"{name}_a"] = fit.a
        out[f"{name}_tokens_per_param"] = o["tokens_per_param"]
    bbg = constrained_optimum(PUBLISHED_FITS["besiroglu"], BBG_C, BBG_U)
    out.update({"bbg_free_N": bbg["free"]["N"], "bbg_capped_N": bbg["N"], "bbg_capped_epochs": bbg["epochs"]})
    return out


@lru_cache(maxsize=None)
def viz_payload() -> dict:
    lf, tf = lin_fits(), toy_fits()
    lin = {"N": list(LIN_NS), "D": list(LIN_DS), "L": [[ak._sig(x, 4) for x in row] for row in lin_grid()],
           "fit": {k: _r(getattr(lf["fit"], k), 4) for k in ("E", "A", "B", "alpha", "beta")},
           "a": {"pred": _r(lf["a_pred"]), "a1": _r(lf["a1"]), "a2": _r(lf["a2"]), "a3": _r(lf["a3"])},
           "C": [ak._sig(c, 4) for c in LIN_C], "star1": [int(x) for x in lf["star1"]],
           "star2": [ak._sig(x, 4) for x in lf["star2"]]}
    toy = {"runs": [{"budget": r["budget"], "d": r["d"], "N": r["N"], "N_ne": r["N_ne"], "D": r["D"],
                     "loss": _r(r["loss"])} for r in isoflop_grid()],
           "N_star": [ak._sig(x, 4) for x in tf["N_star"]], "Nne_star": [ak._sig(x, 4) for x in tf["Nne_star"]],
           "Cne_star": [ak._sig(x, 4) for x in tf["Cne_star"]], "a_chinchilla": _r(tf["a_chinchilla"]),
           "a_kaplan": _r(tf["a_kaplan"]), "a3": _r(tf["a3"]),
           "fit": {k: _r(getattr(tf["fit"], k), 4) for k in ("E", "A", "B", "alpha", "beta")},
           "seed_sd": _r(seed_spread()["sd"])}
    rep = repetition_runs()
    repeat = {"N": rep["N"], "D": rep["D"], "r_star": R_STAR,
              "rows": [{"epochs": r["epochs"], "unique": int(r["unique"]), "D_eff": int(round(r["D_eff"])),
                        "loss": _r(r["loss"]), "pred": _r(r["pred"])} for r in rep["rows"]]}
    k = kaplan_shift()
    kap = {"C": [ak._sig(x, 4) for x in k["C"]], "local": _r(k["local"]), "predicted": _r(k["predicted"]),
           "share": _r(k["share"]), "a": _r(k["a"]), "limit": _r(k["limit"]), "a_small": _r(k["a_small"])}
    c = cap_sweep()
    cap = {"C": ak._sig(BBG_C, 4), "U": [ak._sig(u, 4) for u in c["U"]], "N": [ak._sig(x, 4) for x in c["N"]],
           "epochs": _r(c["epochs"]), "L": _r(c["L"], 4), "free_N": ak._sig(c["free"]["N"], 4),
           "free_D": ak._sig(c["free"]["D"], 4), "free_L": _r(c["free"]["L"], 4), "bbg_U": BBG_U}
    mc = corpus_learning_curve()
    memo = {"n": [int(x) for x in mc["n"]], "rate": [ak._sig(x, 4) for x in mc["rate"]],
            "theory": [ak._sig(x, 4) for x in py_missing_mass(mc["n"])], "slope": _r(mc["slope"])}
    return {"linear": lin, "toy": toy, "repeat": repeat, "kaplan": kap, "cap": cap, "memorizer": memo}


# --------------------------------------------------------------------------------------------
# Verification harness.
# --------------------------------------------------------------------------------------------

def test_missing_mass_exact():
    """Thm 1: the share of PY(0.6, 20) sequences whose next draw is new, and the mean K_n, match the closed forms
    within three standard errors at every checkpoint."""
    sim = py_simulation()
    assert np.all(np.abs(sim["new"] - py_missing_mass(sim["n"])) <= 3 * sim["new_se"]), (sim["new"],)
    assert np.all(np.abs(sim["K"] - zhv.py_expected_types(sim["n"], PY_D, PY_THETA)) <= 3 * sim["K_se"])


def test_dirichlet_case():
    """d = 0 (the one-parameter CRP, a Dirichlet process): the missing mass is θ/(θ + n), so β = 1 — the case Hutter
    (2021, §6) describes; Thm 1's β = 1 − d departs from it as soon as d > 0."""
    n = np.array([10.0, 1e3, 1e6])
    assert np.allclose(py_missing_mass(n, 0.0, PY_THETA), PY_THETA / (PY_THETA + n), rtol=1e-12)


def test_missing_mass_exponent():
    """Thm 1's rate: the closed form approaches Γ(θ+1)/Γ(θ+d)·n^{−(1−d)}, and the exponent agrees with Hutter's
    1 − 1/s at the PY measure's Zipf exponent s = 1/d, for several (d, θ)."""
    for d, theta in ((PY_D, PY_THETA), (0.3, 1.0), (0.8, 5.0)):
        n = np.array([1e6, 1e8])
        ratio = py_missing_mass(n, d, theta) / py_missing_mass_asymptote(n, d, theta)
        assert np.all(np.abs(ratio - 1) < 1e-4), ratio
        slope = loglog_slope(n, py_missing_mass(n, d, theta))
        assert abs(slope + hutter_exponent(zipf_exponent_of_discount(d))) < 1e-4, (d, slope)


def test_corpus_learning_curve():
    """The memorizer on the corpus's text: its new-type rate falls with slope near −(1 − d) = −0.4 (template words
    saturate early, so the fit starts at 10k tokens)."""
    c = corpus_learning_curve()
    assert abs(c["slope"] + (1 - PY_D)) < 0.06, c["slope"]


def test_tail_bracket():
    """Thm 2: the integral test brackets the tail on random (b, N, M), and the tail scales as N^{−(b−1)}."""
    rng = np.random.default_rng([SEED, 1630])
    for _ in range(50):
        b = rng.uniform(1.1, 4.0)
        M = int(rng.integers(100, 5000))
        N = int(rng.integers(1, M - 1))
        lo, hi = tail_bracket(N, b, M)
        t = tail_mass(N, b, M)
        assert lo <= t <= hi, (b, N, M)
    ns = np.array([16, 64, 256])
    assert abs(loglog_slope(ns, [tail_mass(int(n), LIN_B, 10**6) for n in ns]) + (LIN_B - 1)) < 0.02


def test_gradient_flow_asymptote():
    """Thm 3: on random (a, b, t) the gradient-flow residual Σ_{k≤M} f(k) is within the bump max f of ∫_1^M f, which
    is the asymptote times P(s, 2t) − P(s, 2tM^{−a}) (s = (b−1)/a, P the regularized lower incomplete gamma); and the
    ratio to Γ(s)/a·(2t)^{−s} tends to 1."""
    rng = np.random.default_rng([SEED, 1631])
    M = 10**6
    for _ in range(20):
        a, b, t = rng.uniform(1.1, 2.5), rng.uniform(1.5, 4.0), 10 ** rng.uniform(1, 3)
        s = (b - 1) / a
        integral = gf_asymptote(t, a, b) * (special.gammainc(s, 2 * t) - special.gammainc(s, 2 * t * M**-a))
        assert abs(gf_residual(t, a, b, M) - integral) <= gf_bump(t, a, b), (a, b, t)
    ratios = [gf_residual(t, LIN_A, LIN_B, M) / gf_asymptote(t) for t in (10.0, 100.0, 1000.0)]
    assert all(abs(r - 1) < 0.01 for r in ratios), ratios


MC_CHECKPOINTS = (16, 128, 1024)


def test_sgd_recursion_matches_monte_carlo():
    """Prop 1: 400 Monte Carlo SGD runs (M = 256, N = 32) agree with the exact diagonal recursion within three standard
    errors at every checkpoint."""
    mc, se = sgd_monte_carlo(np.random.default_rng([SEED, 1632]), 32, MC_CHECKPOINTS, 400)
    exact = sgd_losses([32], MC_CHECKPOINTS, M=256)[0]
    assert np.all(np.abs(mc - exact) <= 3 * se), (mc, exact, se)


def test_linear_scaling_exponents():
    """Thms 2–4 on the solvable model: Approach 3's α and β land within 0.05 of b − 1 and (b − 1)/a, and all three
    approaches put N* ∝ C^a within 0.03 of a = 1/(1 + a_λ) = 0.4."""
    f = lin_fits()
    assert abs(f["fit"].alpha - f["alpha_pred"]) < 0.05 and abs(f["fit"].beta - f["beta_pred"]) < 0.05, f["fit"]
    for k in ("a1", "a2", "a3"):
        assert abs(f[k] - f["a_pred"]) < 0.03, (k, f[k])


def test_additive_form_is_approximate():
    """Eq. 2 recovers the exponents but not the surface: its log-error is about 0.13 overall and largest for small N at
    small D, where it overestimates — a size-N model's data term sums only its N kept coordinates, not all of them."""
    f = lin_fits()
    assert 0.05 < f["resid_rms"] < 0.2 and f["resid_max"] > 0.2, (f["resid_rms"], f["resid_max"])
    NN, DD = np.meshgrid(LIN_NS, LIN_DS, indexing="ij")
    resid = np.log(f["fit"].loss(NN, DD)) - np.log(lin_grid())
    i, j = np.unravel_index(np.abs(resid).argmax(), resid.shape)
    assert resid[i, j] > 0 and LIN_NS[i] <= 16 and LIN_DS[j] <= 256, (LIN_NS[i], LIN_DS[j])


def _decoder_loss(P, blocks, tok, tgt) -> float:
    lp = decoder_log_probs(P, blocks, tok)
    return float(-np.take_along_axis(lp, tgt[..., None], -1).mean())


def test_decoder_gradients():
    """decoder_loss_grad against central differences (float64) at 4 random coordinates of every parameter array."""
    rng = np.random.default_rng([SEED, 1633])
    P, blocks = init_decoder(rng, 16, np.float64)
    tok, tgt = toy_batch(rng, 4)
    _, G, Gb = decoder_loss_grad(P, blocks, tok, tgt)
    pairs = [(P, k, G[k]) for k in P] + [(blocks[i][s], k, Gb[i][s][k]) for i in range(TOY_LAYERS)
                                         for s in blocks[i] for k in blocks[i][s]]
    eps = 1e-6
    for D, k, g in pairs:
        for idx in zip(*(rng.integers(0, n, 4) for n in D[k].shape)):
            old = D[k][idx]
            D[k][idx] = old + eps
            up = _decoder_loss(P, blocks, tok, tgt)
            D[k][idx] = old - eps
            dn = _decoder_loss(P, blocks, tok, tgt)
            D[k][idx] = old
            num = (up - dn) / (2 * eps)
            assert abs(num - g[idx]) <= 1e-6 + 1e-5 * abs(num), (k, idx, num, g[idx])


def test_flop_count():
    """Prop 2: the instrumented step takes exactly B·n·(6N + 12·L·n·d) matmul FLOPs at every width, and the attention
    term is a few percent to a quarter of 6N on this grid."""
    for d in TOY_WIDTHS:
        assert counted_step_flops(d) == step_flops(d), (d, counted_step_flops(d), step_flops(d))
    shares = [attention_share(d) for d in TOY_WIDTHS]
    assert all(0.05 < s < 0.3 for s in shares) and shares == sorted(shares, reverse=True), shares


def test_compute_optimal_closed_form():
    """Thm 4: the closed form equals a numeric argmin on random fits and budgets, its multiplier equals −dL*/dC, and
    the frontier's exponent is −αβ/(α+β)."""
    rng = np.random.default_rng([SEED, 1634])
    for _ in range(30):
        fit = LossFit(rng.uniform(0, 2), 10 ** rng.uniform(1, 3), 10 ** rng.uniform(1, 3), rng.uniform(0.1, 1.0),
                      rng.uniform(0.1, 1.0))
        C = 10 ** rng.uniform(15, 25)
        o = compute_optimal(fit, C)
        res = optimize.minimize_scalar(lambda x: float(fit.loss(math.exp(x), C / (6 * math.exp(x)))),
                                       bounds=(math.log(o["N"]) - 5, math.log(o["N"]) + 5), method="bounded",
                                       options={"xatol": 1e-10})
        assert abs(res.x - math.log(o["N"])) < 1e-4 and res.fun >= o["L"] - 1e-12 * o["L"]
        h = 1e-4
        dL = (compute_optimal(fit, C * (1 + h))["L"] - compute_optimal(fit, C * (1 - h))["L"]) / (2 * h * C)
        assert abs(-dL - o["nu"]) <= 1e-5 * abs(o["nu"]), (dL, o["nu"])
        ex = (math.log(compute_optimal(fit, 10 * C)["L"] - fit.E) - math.log(o["L"] - fit.E)) / math.log(10)
        assert abs(ex + fit.alpha * fit.beta / (fit.alpha + fit.beta)) < 1e-9


def test_published_fits():
    """Hoffmann's rounded fit gives a = 0.452, the unrounded one 0.4565, Besiroglu's refit 0.5126 and about 20 tokens
    per parameter at Gopher's budget — Chinchilla's own ratio — where the rounded fit asks for far more."""
    a = {k: f.a for k, f in PUBLISHED_FITS.items()}
    assert abs(a["hoffmann_rounded"] - 0.4516) < 1e-3 and abs(a["hoffmann_unrounded"] - 0.4565) < 1e-3, a
    assert abs(a["besiroglu"] - 0.5126) < 1e-3, a
    tpp = {k: compute_optimal(f, GOPHER_C)["tokens_per_param"] for k, f in PUBLISHED_FITS.items()}
    assert 15 < tpp["besiroglu"] < 30 and tpp["hoffmann_rounded"] > 2 * tpp["besiroglu"], tpp
    N = {k: compute_optimal(f, GOPHER_C)["N"] for k, f in PUBLISHED_FITS.items()}
    # Hoffmann et al. (§3.3) report 40B from Approach 3 at Gopher's budget: only the unrounded parameters give it;
    # the printed ones give 32B. Besiroglu's refit lands beside Approaches 1 and 2 (67B and 63B, their Table A3).
    assert abs(N["hoffmann_unrounded"] / 40e9 - 1) < 0.02 and abs(N["hoffmann_rounded"] / 32.2e9 - 1) < 0.01, N
    assert 60e9 < N["besiroglu"] < 75e9, N


def test_toy_isoflop_profiles():
    """Every budget's profile has an interior minimum, and the parabola's N* grows with the budget."""
    tf = toy_fits()
    for C, d_min in zip(TOY_BUDGETS, tf["argmin_d"]):
        ds = [r["d"] for r in isoflop_grid() if r["budget"] == C]
        assert min(ds) < d_min < max(ds), (C, ds, d_min)
    assert np.all(np.diff(tf["N_star"]) > 0), tf["N_star"]


def test_kaplan_counting_shift():
    """Prop 3: Besiroglu's fit (a = 0.513) read in Kaplan's convention has the local exponent ρa/(ρa + 1 − a) at every
    budget, near Kaplan's 0.73 below 1e18 FLOPs and falling toward a as embeddings stop mattering; the limit is
    3a/(1 + 2a). The toy grid shifts the same way."""
    k = kaplan_shift()
    inner = slice(1, -1)
    assert np.all(np.abs(k["local"][inner] - k["predicted"][inner]) < 2e-3), np.abs(k["local"] - k["predicted"]).max()
    assert 0.68 < k["a_small"] < 0.76 and k["predicted"][-1] < k["a"] + 0.03 and k["limit"] > k["predicted"][0]
    assert abs(k["limit"] - 3 * k["a"] / (1 + 2 * k["a"])) < 1e-12
    tf = toy_fits()
    assert tf["a_kaplan"] > tf["a_chinchilla"] + 0.03, (tf["a_kaplan"], tf["a_chinchilla"])


def test_seed_spread():
    """One run's loss moves by a few hundredths of a nat with the seed: the resolution of every toy point."""
    s = seed_spread()
    assert 0.005 < s["sd"] < 0.1, s


def test_repetition():
    """Muennighoff's regime on the toy: 4 epochs land within two seed-spreads of fresh data, 64 epochs are clearly
    worse; D′ ≤ D, with equality at one epoch, and D′ saturates at U(1 + R*)."""
    rows = repetition_runs()["rows"]
    sd = seed_spread()["sd"]
    fresh = rows[0]["loss"]
    by = {r["epochs"]: r for r in rows}
    assert abs(by[4]["loss"] - fresh) < 2 * sd + 0.02 and by[64]["loss"] > fresh + 0.2, rows
    assert float(effective_data(1000.0, 1000.0)) == 1000.0
    assert all(r["D_eff"] <= repetition_runs()["D"] for r in rows)
    assert abs(float(effective_data(1e12, 1.0)) - (1 + R_STAR)) < 1e-9


@lru_cache(maxsize=1)
def cap_sweep() -> dict:
    """The capped optimum at BloombergGPT's compute for every unique-token budget in CAP_U (Besiroglu's fit)."""
    fit = PUBLISHED_FITS["besiroglu"]
    rows = [constrained_optimum(fit, BBG_C, U) for U in CAP_U]
    return {"U": np.array(CAP_U), "N": np.array([r["N"] for r in rows]), "epochs": np.array([r["epochs"] for r in rows]),
            "L": np.array([r["L"] for r in rows]), "free": rows[0]["free"]}


def test_data_cap_moves_the_optimum():
    """At BloombergGPT's compute, Besiroglu's fit wants more than its 709B tokens, but Muennighoff's decay makes the
    1.04 epochs it would repeat nearly free: the optimum moves by under 1%. Shrink the corpus and the cap binds — the
    optimum grows and the loss rises — monotonically in U."""
    o = constrained_optimum(PUBLISHED_FITS["besiroglu"], BBG_C, BBG_U)
    assert o["free"]["D"] > BBG_U and 1 < o["epochs"] < 1.1 and 0 < o["N"] / o["free"]["N"] - 1 < 0.01, o
    c = cap_sweep()
    bind = c["U"] < c["free"]["D"]
    assert np.all(np.diff(c["N"][bind]) < 0) and np.all(np.diff(c["L"][bind]) < 0) and c["epochs"][0] > 5, c
    assert np.allclose(c["N"][~bind], c["free"]["N"], rtol=1e-6)


def test_viz_payload():
    """The lab's data: the full solvable-model grid, every toy run, the repetition rows; under 60 kB."""
    p = viz_payload()
    assert len(p["linear"]["L"]) == len(LIN_NS) and len(p["linear"]["L"][0]) == len(LIN_DS)
    assert len(p["toy"]["runs"]) == len(isoflop_grid()) and len(p["repeat"]["rows"]) == 1 + len(REPEAT_EPOCHS)
    assert len(json.dumps(p, separators=(",", ":"))) < 60_000


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "scaling-laws-lab.json"


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
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
    print(f"\nSummary (solvable model a = {LIN_A}, b = {LIN_B}, η = {LIN_ETA}, M = {LIN_M}; toy decoders: "
          f"{TOY_LAYERS} pre-LN blocks, V = {TOY_V}, n = {TOY_CTX}, generate_corpus({SCALE_DOCS}, 0)):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.6f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
