"""Attention as kernel smoothing — the reference implementation for formalNLP's ninth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. A weighted lookup: softmax attention is the Nadaraya–Watson smoother with the kernel e^{⟨q,k⟩/τ}, which is a
   Gaussian in ‖q − k‖ (bandwidth √τ) times design weights e^{‖k‖²/2τ}; its weights are the posterior
   responsibilities of a Gaussian mixture centered on the keys; one layer without positions reads its prefix as a
   bag, and two causal layers already do not.
2. Temperature is bandwidth: τ → 0 is a lookup and τ → ∞ the mean, both at exponential rates; on n equispaced
   keys on the circle attention is a Fourier multiplier with aliased Bessel gains, so its bias–variance trade-off
   is a closed form whose optimal concentration grows like n^{2/5}; a trained one-layer denoiser finds it.
3. Why 1/√d: unscaled dot products of random d-dimensional vectors have variance d, so the softmax saturates and
   its Jacobian (Topic 5's covariance) vanishes; the scale holds the kernel's spread fixed.
4. A lookup instead of a state: the gradient from position t reaches position k in one hop, scaled by the
   attention weight, where Topic 5's recurrent state multiplies t − k Jacobians; content addressing solves Topic
   5's long-lag task at every lag.
5. Attention over a document: a kernel cache over the document's earlier tokens (Grave et al.'s continuous cache,
   with Topic 6's vectors as keys), mixed with Topic 3's Kneser–Ney; its inverse bandwidth is chosen on dev, and a
   learned bilinear kernel is trained on the pointer likelihood.
6. Kernels that factor: one head's log-attention matrix has rank ≤ d_k + 1; positive random features estimate
   e^{⟨x,y⟩} without bias but with variance e^{2⟨x,y⟩}(e^{‖x+y‖²} − 1); causal linear attention is a recurrence.
7. Finance case study: each EDGAR filing's own earlier words as the cache, in open-vocabulary bits per character.

Cross-checks: every hand-written gradient against complex-step derivatives and central differences (Topic 5's
checkers); the circle's gains against numpy's FFT; the probabilistic claims by Monte Carlo; the Kneser–Ney
baseline against Topic 4's and Topic 5's own accounting.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/attention-as-kernel-smoothing/attention_as_kernel_smoothing.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, special

_HERE = Path(__file__).resolve().parent
for _dep in ("finance-corpus", "zipf-heaps-vocabulary-statistics", "subword-tokenization-bpe-unigram",
             "n-gram-models-smoothing", "perplexity-cross-entropy", "neural-language-models",
             "distributional-semantics-ppmi-svd"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import distributional_semantics_ppmi_svd as ds  # noqa: E402
import finance_corpus as fc  # noqa: E402
import n_gram_models_smoothing as ng  # noqa: E402
import neural_language_models as nlm  # noqa: E402
import perplexity_cross_entropy as pce  # noqa: E402

# ONE stream per random-instance claim and per trained model, each from default_rng([SEED, k]); k ≥ 900 keeps this
# module's streams apart from Topics 5–8.
SEED = 20261023
LN2 = math.log(2)
Params = dict[str, np.ndarray]


def _softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax through Topic 5's log_softmax (complex inputs pass through, for the complex step)."""
    return np.exp(nlm.log_softmax(z))


# --------------------------------------------------------------------------------------------
# 1. A weighted lookup.
# --------------------------------------------------------------------------------------------


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, tau: float | None = None,
              causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Def 2: softmax(QKᵀ/τ + B)V row-wise, τ = √d_k by default; B masks j > t when causal. Returns (O, A)."""
    tau = math.sqrt(K.shape[-1]) if tau is None else tau
    S = Q @ np.swapaxes(K, -1, -2) / tau
    if causal:
        S = np.where(np.tri(Q.shape[-2], K.shape[-2], dtype=bool), S, -np.inf)
    A = _softmax(S)
    return A @ V, A


def kernel_smoother(q: np.ndarray, K: np.ndarray, V: np.ndarray, kernel) -> np.ndarray:
    """Def 1: Σ_j κ(q, k_j) v_j / Σ_j κ(q, k_j) for a positive kernel κ(q, K) → weights."""
    w = kernel(q, K)
    return w @ V / w.sum()


def gaussian_form(q: np.ndarray, K: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    """Prop 1 (ii): e^{⟨q,k⟩/τ} = e^{‖q‖²/2τ} · e^{‖k‖²/2τ} · e^{−‖q−k‖²/2τ} — the query factor (common to every key,
    so it cancels), the design weights ω_j, and the Gaussian kernel of bandwidth √τ."""
    return {"query": np.exp(q @ q / (2 * tau)), "omega": np.exp((K * K).sum(1) / (2 * tau)),
            "gauss": np.exp(-((K - q) ** 2).sum(1) / (2 * tau))}


def mixture_posterior(q: np.ndarray, K: np.ndarray, tau: float, omega: np.ndarray) -> np.ndarray:
    """Prop 2 by Bayes' rule: P(J = j | Q = q) for J ~ π ∝ ω and Q | J ~ N(k_J, τI)."""
    log_joint = np.log(omega / omega.sum()) - ((K - q) ** 2).sum(1) / (2 * tau)
    return _softmax(log_joint)


def sample_mixture(rng: np.random.Generator, K: np.ndarray, V: np.ndarray, tau: float, omega: np.ndarray,
                   n: int) -> tuple[np.ndarray, np.ndarray]:
    """n draws of (Q, V) from Prop 2's generative model."""
    J = rng.choice(len(K), size=n, p=omega / omega.sum())
    return K[J] + math.sqrt(tau) * rng.standard_normal((n, K.shape[1])), V[J]


def causal_stack(X: np.ndarray, layers: list[Params]) -> np.ndarray:
    """Prop 3 (iii): residual layers h ← h + Attn_causal(h), then h ← h + tanh(h W_m), with no positions."""
    h = X
    for L in layers:
        h = h + attention(h @ L["Wq"], h @ L["Wk"], h @ L["Wv"], causal=True)[0]
        h = h + np.tanh(h @ L["Wm"])
    return h


def bidirectional_stack(X: np.ndarray, layers: list[Params]) -> np.ndarray:
    """Prop 3 (i): the same stack without the mask."""
    h = X
    for L in layers:
        h = h + attention(h @ L["Wq"], h @ L["Wk"], h @ L["Wv"])[0]
        h = h + np.tanh(h @ L["Wm"])
    return h


def random_layers(rng: np.random.Generator, p: int, depth: int, scale: float = 0.5) -> list[Params]:
    return [{k: rng.normal(0, scale / math.sqrt(p), (p, p)) for k in ("Wq", "Wk", "Wv", "Wm")}
            for _ in range(depth)]


# --------------------------------------------------------------------------------------------
# 2. Temperature is bandwidth.
# --------------------------------------------------------------------------------------------


def limit_bounds(s: np.ndarray, V: np.ndarray, tau: float) -> dict[str, float]:
    """Prop 4 for one query with scores s over values V: the distances of o_τ from the argmax mean and from the
    plain mean, and their bounds 2V_max(n − |S*|)e^{−Δ/τ}/|S*| and V_max(e^{R/τ} − 1)."""
    n = len(s)
    top = np.isclose(s, s.max(), rtol=0, atol=1e-12)
    a = _softmax(s / tau)
    o = a @ V
    vmax = float(np.linalg.norm(V, axis=1).max())
    gap = float(s.max() - s[~top].max()) if (~top).any() else math.inf
    R = float(s.max() - s.min())
    return {"hard": float(np.linalg.norm(o - V[top].mean(0))),
            "hard_bound": 2 * vmax * (n - top.sum()) * math.exp(-gap / tau) / top.sum(),
            "mean": float(np.linalg.norm(o - V.mean(0))), "mean_bound": vmax * _expm1(R / tau)}


def _expm1(x: float) -> float:
    with np.errstate(over="ignore"):
        return float(np.expm1(x))


def circle_angles(n: int, theta0: float = 0.0) -> np.ndarray:
    return theta0 + 2 * np.pi * np.arange(n) / n


def circle_attention(n: int, c: float, theta0: float = 0.0) -> np.ndarray:
    """Thm 1's A: n equispaced keys and queries u(θ) = (cos θ, sin θ), scores c·cos(θ_i − θ_j), bidirectional."""
    th = circle_angles(n, theta0)
    return _softmax(c * np.cos(th[:, None] - th[None, :]))


def _alias(n: int, c: float) -> np.ndarray:
    """The aliasing offsets m = j·n that carry non-negligible I_{k+m}(c) (I_m(c) falls off super-exponentially
    once |m| ≫ c)."""
    J = int(math.ceil((c + 60.0) / n)) + 1
    return n * np.arange(-J, J + 1)


def circle_gains(n: int, c: float, ks=None) -> np.ndarray:
    """Thm 1 (i): ρ_k(c) = Σ_{m ≡ k (n)} I_m(c) / Σ_{m ≡ 0 (n)} I_m(c), from the generating function
    e^{c cos φ} = Σ_m I_m(c) e^{imφ} (DLMF 10.35.1). Exponentially scaled Bessels (ive) keep large c finite."""
    ks = np.arange(n) if ks is None else np.asarray(ks)
    m = _alias(n, c)
    return special.ive(ks[:, None] + m[None, :], c).sum(1) / special.ive(m, c).sum()


def circle_variance_factor(n: int, c: float) -> float:
    """Σ_m a_m² = Σ_{m≡0} I_m(2c) / (n (Σ_{m≡0} I_m(c))²): the variance a unit of noise leaves after smoothing."""
    m2, m1 = _alias(n, 2 * c), _alias(n, c)
    return float(special.ive(m2, 2 * c).sum() / (n * special.ive(m1, c).sum() ** 2))


def circle_risk(c: float, n: int, power: np.ndarray, sigma: float) -> dict[str, float]:
    """Thm 1 (ii) for f with mean power power[k−1] = |F_k|² + |F_{−k}|² at frequency k = 1…K (K < n/2):
    R(c) = Σ_k (1 − ρ_k)² power_k + σ² Σ_m a_m². For a random f, power_k = E(a_k² + b_k²)/2 = s_k²."""
    power = np.asarray(power, float)
    rho = circle_gains(n, c, np.arange(1, len(power) + 1))
    bias2 = float(((1 - rho) ** 2 * power).sum())
    var = sigma**2 * circle_variance_factor(n, c)
    return {"bias2": bias2, "var": var, "risk": bias2 + var}


def optimal_c(n: int, power: np.ndarray, sigma: float) -> float:
    """argmin_c R(c), on log c."""
    r = optimize.minimize_scalar(lambda lc: circle_risk(math.exp(lc), n, power, sigma)["risk"],
                                 bounds=(-3.0, 14.0), method="bounded", options={"xatol": 1e-10})
    return float(math.exp(r.x))


def rate_prefactor(power: np.ndarray, sigma: float) -> float:
    """Cor 1: c*(n) ~ (4Bn/(σ²√π))^{2/5} with B = ¼ Σ_k k⁴ power_k; this returns (4B/(σ²√π))^{2/5}."""
    k = np.arange(1, len(power) + 1)
    B = 0.25 * float((k**4 * np.asarray(power)).sum())
    return (4 * B / (sigma**2 * math.sqrt(math.pi))) ** 0.4


# The trained denoiser: in-context smoothing of a random function on the circle.

SPECTRA = {"smooth": np.array([1.0, 0.25]), "rough": np.full(6, 0.25)}  # s_k² for k = 1, 2, …
SIGMA = 0.5


@dataclass(frozen=True)
class DenoiserConfig:
    spectrum: str = "smooth"
    n: int = 32
    steps: int = 1500
    batch: int = 32
    lr: float = 0.05
    stream: int = 930


def denoiser_tasks(rng: np.random.Generator, batch: int, n: int, spectrum: str,
                   sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A batch of tasks: θ_j = θ₀ + 2πj/n with θ₀ uniform; f = Σ_k a_k cos kθ + b_k sin kθ with a_k, b_k ~ N(0, s_k²);
    y = f + σε. Returns (U, y, f) with U the position features (cos θ, sin θ), shape (batch, n, 2)."""
    s = np.sqrt(SPECTRA[spectrum])
    th = rng.uniform(0, 2 * np.pi, (batch, 1)) + 2 * np.pi * np.arange(n) / n
    ab = rng.standard_normal((batch, 2, len(s))) * s
    k = np.arange(1, len(s) + 1)
    f = (ab[:, 0, None, :] * np.cos(th[..., None] * k) + ab[:, 1, None, :] * np.sin(th[..., None] * k)).sum(-1)
    y = f + sigma * rng.standard_normal(f.shape)
    return np.stack([np.cos(th), np.sin(th)], -1), y, f


def denoiser_score_matrix(P: Params) -> np.ndarray:
    """M = W_Q W_Kᵀ / √d_k: the bilinear form on position features, s_ij = u_i M u_jᵀ."""
    return P["Wq"] @ P["Wk"].T / math.sqrt(P["Wq"].shape[1])


def denoiser_loss(P: Params, U, y, f):
    M = denoiser_score_matrix(P)
    A = _softmax(U @ M @ np.swapaxes(U, 1, 2))
    o = (A @ y[..., None])[..., 0]
    return ((o - f) ** 2).mean()


def denoiser_loss_grad(P: Params, U, y, f) -> tuple[float, Params]:
    """The mean squared error to the clean f and its gradient: ∂ℓ/∂s_ij = α_ij (y_j − o_i) ∂ℓ/∂o_i, then
    ∂ℓ/∂M = Σ Uᵀ (∂ℓ/∂S) U, and the factors by the chain rule."""
    M = denoiser_score_matrix(P)
    A = _softmax(U @ M @ np.swapaxes(U, 1, 2))
    o = (A @ y[..., None])[..., 0]
    g_o = 2 * (o - f) / o.size
    dS = A * g_o[..., None] * (y[:, None, :] - o[..., None])
    dM = np.einsum("bia,bij,bjc->ac", U, dS, U)
    r = math.sqrt(P["Wq"].shape[1])
    return float(((o - f) ** 2).mean()), {"Wq": dM @ P["Wk"] / r, "Wk": dM.T @ P["Wq"] / r}


@dataclass(frozen=True)
class TrainedDenoiser:
    M: np.ndarray  # the Polyak average of M over the second half of training
    c: float  # tr M / 2, the learned concentration
    anisotropy: float  # ‖M − cI‖_F / ‖M‖_F
    c_star: float  # Thm 1's population optimum for this spectrum and n
    final_loss: float  # mean loss over the last 100 steps


@lru_cache(maxsize=None)
def train_denoiser(cfg: DenoiserConfig) -> TrainedDenoiser:
    """Adam on fresh tasks every step (one stream), from small random factors (nearly uniform attention)."""
    rng = np.random.default_rng([SEED, cfg.stream])
    P = {"Wq": rng.normal(0, 0.3, (2, 2)), "Wk": rng.normal(0, 0.3, (2, 2))}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    M_sum, n_avg, tail = np.zeros((2, 2)), 0, []
    for step in range(1, cfg.steps + 1):
        loss, G = denoiser_loss_grad(P, *denoiser_tasks(rng, cfg.batch, cfg.n, cfg.spectrum))
        c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
        for k in P:
            m[k] = nlm.ADAM_B1 * m[k] + (1 - nlm.ADAM_B1) * G[k]
            s[k] = nlm.ADAM_B2 * s[k] + (1 - nlm.ADAM_B2) * G[k] ** 2
            P[k] -= cfg.lr * (m[k] / c1) / (np.sqrt(s[k] / c2) + nlm.ADAM_EPS)
        if step > cfg.steps // 2:
            M_sum += denoiser_score_matrix(P)
            n_avg += 1
        if step > cfg.steps - 100:
            tail.append(loss)
    M = M_sum / n_avg
    c = float(np.trace(M) / 2)
    return TrainedDenoiser(M=M, c=c, anisotropy=float(np.linalg.norm(M - c * np.eye(2)) / np.linalg.norm(M)),
                           c_star=optimal_c(cfg.n, SPECTRA[cfg.spectrum], SIGMA), final_loss=float(np.mean(tail)))


DENOISERS = (DenoiserConfig("smooth", 32, stream=930), DenoiserConfig("smooth", 64, stream=931),
             DenoiserConfig("rough", 32, stream=932))


# --------------------------------------------------------------------------------------------
# 3. Why 1/√d.
# --------------------------------------------------------------------------------------------


def softmax_jacobian_bound(alpha: np.ndarray) -> dict[str, float]:
    """Thm 2 (ii): ‖diag α − ααᵀ‖₂ ≤ tr = 1 − ‖α‖² ≤ 2(1 − α_max)."""
    J = np.diag(alpha) - np.outer(alpha, alpha)
    return {"norm": float(np.linalg.eigvalsh(J)[-1]), "trace": float(1 - alpha @ alpha),
            "bound": float(2 * (1 - alpha.max()))}


SAT_DIMS = (4, 16, 64, 256, 1024)
SAT_KEYS, SAT_TRIALS = 16, 600


@lru_cache(maxsize=1)
def saturation_curve() -> dict[str, list[float]]:
    """Thm 2 (iii) by Monte Carlo: q ~ N(0, I_d) and SAT_KEYS keys k_j ~ N(0, I_d), drawn explicitly; the mean of
    1 − α_max and of ‖J‖₂ for the unscaled scores ⟨q, k_j⟩ and the scaled ⟨q, k_j⟩/√d."""
    rng = np.random.default_rng([SEED, 940])
    out = {"dims": list(SAT_DIMS), "unscaled_gap": [], "scaled_gap": [], "unscaled_jac": [], "scaled_jac": []}
    for d in SAT_DIMS:
        gaps = {"unscaled": [], "scaled": []}
        jacs = {"unscaled": [], "scaled": []}
        for chunk in range(0, SAT_TRIALS, 100):
            q = rng.standard_normal((100, 1, d))
            K = rng.standard_normal((100, SAT_KEYS, d))
            s = (K * q).sum(-1)
            for name, z in (("unscaled", s), ("scaled", s / math.sqrt(d))):
                a = _softmax(z)
                gaps[name] += list(1 - a.max(1))
                jacs[name] += [softmax_jacobian_bound(row)["norm"] for row in a]
        for name in ("unscaled", "scaled"):
            out[f"{name}_gap"].append(float(np.mean(gaps[name])))
            out[f"{name}_jac"].append(float(np.mean(jacs[name])))
    return out


# --------------------------------------------------------------------------------------------
# 4. A lookup instead of a state.
# --------------------------------------------------------------------------------------------


def one_hop_jacobian(X: np.ndarray, P: Params, t: int, k: int) -> np.ndarray:
    """Prop 5: for k < t, ∂o_t/∂x_k = α_tk [W_V + τ⁻¹ (W_K q_t) ⊗ (v_k − o_t)], acting on row perturbations
    (do_t = dx_k J)."""
    tau = math.sqrt(P["Wk"].shape[1])
    O, A = attention(X @ P["Wq"], X @ P["Wk"], X @ P["Wv"], causal=True)
    q, v = X[t] @ P["Wq"], X[k] @ P["Wv"]
    return A[t, k] * (P["Wv"] + np.outer(P["Wk"] @ q, v - O[t]) / tau)


def one_hop_bound(X: np.ndarray, P: Params, t: int, k: int) -> float:
    """Prop 5's bound α_tk (‖W_V‖ + τ⁻¹ ‖W_K‖ ‖q_t‖ ‖v_k − o_t‖) (spectral norms)."""
    tau = math.sqrt(P["Wk"].shape[1])
    O, A = attention(X @ P["Wq"], X @ P["Wk"], X @ P["Wv"], causal=True)
    q, v = X[t] @ P["Wq"], X[k] @ P["Wv"]
    return float(A[t, k] * (np.linalg.norm(P["Wv"], 2)
                            + np.linalg.norm(P["Wk"], 2) * np.linalg.norm(q) * np.linalg.norm(v - O[t]) / tau))


def lag_construction(L: int, w: float, beta: float = 0.0) -> dict[str, float]:
    """Prop 6, built from Def 2 itself: one-hot inputs over {0, 1, 2, 3}; the query of a noise symbol is 1 and the
    key of a bit symbol is β (d_k = 1, τ = 1), so the bit gets weight e^β/(e^β + L); the value is +1 for bit 0,
    −1 for bit 1, 0 for noise; the logits are (w·o, −w·o). Scored on both bits with all-2 and all-3 noise."""
    Wq, Wk = np.array([[0.0], [0.0], [1.0], [1.0]]), np.array([[beta], [beta], [0.0], [0.0]])
    Wv = np.array([[1.0], [-1.0], [0.0], [0.0]])
    margin = math.exp(beta) / (math.exp(beta) + L)
    out = {"margin": margin, "closed_form_loss": math.log1p(math.exp(-2 * w * margin))}
    losses, correct = [], []
    eye = np.eye(4)
    for bit in (0, 1):
        for noise in (2, 3):
            X = eye[[bit] + [noise] * L]
            o = attention(X @ Wq, X @ Wk, X @ Wv, tau=1.0, causal=True)[0][-1, 0]
            z = np.array([w * o, -w * o])
            losses.append(-nlm.log_softmax(z)[bit])
            correct.append(int(np.argmax(z)) == bit)
            out[f"o_bit{bit}"] = float(o)
    out["loss"], out["accuracy"] = float(np.mean(losses)), float(np.mean(correct))
    return out


LAG_D = nlm.LAG_H  # d_k = d_v = 16, Topic 5's lag-task width


def init_lag_attention(rng: np.random.Generator) -> Params:
    """Scales follow Topic 5's Elman init: input maps N(0, 1/2), output map N(0, 1/√H)."""
    return {"Wq": rng.normal(0, 1 / 2, (4, LAG_D)), "Wk": rng.normal(0, 1 / 2, (4, LAG_D)),
            "Wv": rng.normal(0, 1 / 2, (4, LAG_D)), "Wo": rng.normal(0, 1 / math.sqrt(LAG_D), (LAG_D, 2)),
            "bo": np.zeros(2)}


def _lag_forward(P: Params, Xh: np.ndarray):
    """One causal attention layer read at the last position. Xh: (batch, T, 4) input vectors."""
    tau = math.sqrt(LAG_D)
    q = Xh[:, -1] @ P["Wq"]
    K, V = Xh @ P["Wk"], Xh @ P["Wv"]
    a = _softmax((K @ q[..., None])[..., 0] / tau)
    o = (a[..., None] * V).sum(1)
    lp = nlm.log_softmax(o @ P["Wo"] + P["bo"])
    return q, K, V, a, o, lp


def lag_attention_loss(P: Params, Xh: np.ndarray, bit: np.ndarray):
    return -_lag_forward(P, Xh)[-1][np.arange(len(bit)), bit].mean()


def lag_attention_loss_grad(P: Params, X: np.ndarray, bit: np.ndarray) -> tuple[float, Params, float, float]:
    """Topic 5's lag_loss_grad, for one attention layer: (loss, gradient, ‖∂ℓ/∂x₀‖ averaged over the batch — the
    gradient of the batch-mean loss reaching the bit's own input vector, Topic 5's 'reach' — and accuracy).
    X is Topic 5's (T, n) id array."""
    tau = math.sqrt(LAG_D)
    Xh = np.eye(4)[X.T]
    q, K, V, a, o, lp = _lag_forward(P, Xh)
    n = len(bit)
    g = np.exp(lp)
    g[np.arange(n), bit] -= 1
    g /= n
    G = {"Wo": o.T @ g, "bo": g.sum(0)}
    g_o = g @ P["Wo"].T
    dV = a[..., None] * g_o[:, None, :]
    da = (V * g_o[:, None, :]).sum(-1)
    ds = a * (da - (a * da).sum(1, keepdims=True))
    dK = ds[..., None] * q[:, None, :] / tau
    dq = (ds[..., None] * K).sum(1) / tau
    G["Wv"] = np.einsum("bti,btj->ij", Xh, dV)
    G["Wk"] = np.einsum("bti,btj->ij", Xh, dK)
    G["Wq"] = Xh[:, -1].T @ dq
    dX0 = dV[:, 0] @ P["Wv"].T + dK[:, 0] @ P["Wk"].T  # position 0 is never the query (T > 1)
    reach = float(np.linalg.norm(dX0, axis=1).mean())
    return float(-lp[np.arange(n), bit].mean()), G, reach, float((lp.argmax(1) == bit).mean())


@lru_cache(maxsize=None)
def lag_attention_census(lag: int = nlm.LAG, n_seeds: int = 12) -> list[float]:
    """‖∂ℓ/∂x₀‖ at initialization for n_seeds initializations, on Topic 5's protocol (batches of 256)."""
    rng = np.random.default_rng([SEED, 950, lag])
    out = []
    for _ in range(n_seeds):
        X, bit = nlm.lag_data(rng, 256, lag)
        out.append(lag_attention_loss_grad(init_lag_attention(rng), X, bit)[2])
    return out


@lru_cache(maxsize=None)
def recurrent_census(lag: int, n_seeds: int = 12) -> dict[str, list[float]]:
    """Topic 5's census at another lag, with its own pure functions (lag_data, init_lag_model, lag_loss_grad)."""
    rng = np.random.default_rng([SEED, 951, lag])
    out: dict[str, list[float]] = {name: [] for name, _, _ in nlm.lag_models()}
    for _ in range(n_seeds):
        X, bit = nlm.lag_data(rng, 256, lag)
        for name, cell, fb in nlm.lag_models():
            out[name].append(nlm.lag_loss_grad(nlm.init_lag_model(rng, cell, fb), X, bit, cell)[2])
    return out


@lru_cache(maxsize=1)
def lag_attention_training() -> dict[str, float]:
    """Topic 5's protocol: LAG_STEPS Adam steps on fresh batches of LAG_BATCH at LAG_LR, then 2,000 fresh sequences."""
    rng = np.random.default_rng([SEED, 952])
    P = init_lag_attention(rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    X, bit = nlm.lag_data(rng, 256)
    reach0 = lag_attention_loss_grad(P, X, bit)[2]
    for step in range(1, nlm.LAG_STEPS + 1):
        X, bit = nlm.lag_data(rng, nlm.LAG_BATCH)
        _, G, _, _ = lag_attention_loss_grad(P, X, bit)
        for k in P:
            m[k] = nlm.ADAM_B1 * m[k] + (1 - nlm.ADAM_B1) * G[k]
            s[k] = nlm.ADAM_B2 * s[k] + (1 - nlm.ADAM_B2) * G[k] ** 2
            P[k] -= nlm.LAG_LR * (m[k] / (1 - nlm.ADAM_B1**step)) / (np.sqrt(s[k] / (1 - nlm.ADAM_B2**step))
                                                                      + nlm.ADAM_EPS)
    X, bit = nlm.lag_data(rng, 2000)
    loss, _, _, acc = lag_attention_loss_grad(P, X, bit)
    long_X, long_bit = nlm.lag_data(rng, 2000, 600)
    return {"reach_init": reach0, "test_loss": loss, "accuracy": acc,
            "accuracy_lag600": lag_attention_loss_grad(P, long_X, long_bit)[3]}


LAG_SWEEP = (10, 30, 60, 120, 240, 480)
TOPIC5_LAG_ACCURACY = {"elman": 0.485, "lstm-1": 0.531, "lstm-3": 1.000}  # quoted from Topic 5's page (its harness)


@lru_cache(maxsize=1)
def lag_sweep() -> dict[str, list[float]]:
    """Median reach at initialization across lags: attention against Topic 5's three recurrent cells."""
    out = {"lags": list(LAG_SWEEP), "attention": []}
    for lag in LAG_SWEEP:
        out["attention"].append(float(np.median(lag_attention_census(lag))))
        rec = nlm.long_lag_census() if lag == nlm.LAG else recurrent_census(lag)  # Topic 5's own numbers at 60
        for name, vals in rec.items():
            out.setdefault(name, []).append(float(np.median(vals)))
    return out


# --------------------------------------------------------------------------------------------
# 5. Attention over a document: the kernel cache.
# --------------------------------------------------------------------------------------------

KEY_CFG = ("L2", "sppmi5", 100, 0.0)  # Def 3's keys: Topic 6's SVD vectors, declared in advance, never selected
BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
LAMBDAS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998)
SPLITS = ("train", "dev", "held", "edgar")
DISTANCE_BANDS = ((1, 12), (13, 50), (51, 10**9))  # tokens back to the previous mention; 12 ≈ Topic 5's RNN horizon


@lru_cache(maxsize=None)
def documents(split: str) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Each split as documents of sentences, split and tokenized as Topic 3 does (asserted to concatenate to
    Topic 5's splits in test_documents_partition_corpus)."""
    if split == "train":
        return ds.documents()
    raw = {"dev": lambda: fc.generate_corpus(nlm.DEV_DOCS, nlm.DEV_SEED), "held": lambda: fc.generate_corpus(100, 1),
           "edgar": fc.load_edgar_excerpts}[split]()
    return tuple(tuple(ng.split_sentences(d.text)) for d in raw)


@lru_cache(maxsize=1)
def key_vectors() -> tuple[dict[str, int], np.ndarray]:
    """Topic 6's word vectors for KEY_CFG, unit-normalized by row."""
    E = ds.embedding(*KEY_CFG)
    return ds.index(), E / np.linalg.norm(E, axis=1, keepdims=True)


@dataclass(frozen=True)
class Doc:
    """One document's predicted positions (every word and each sentence's </s>), in Topic 3's event order."""
    targets: tuple[str, ...]  # raw strings
    prev: tuple[str, ...]  # the preceding token: <s> at a sentence start
    closed: np.ndarray  # target ids with every OOV word as one <unk>
    raw: np.ndarray  # target ids of the raw strings (open vocabulary)
    sent: np.ndarray  # sentence index of each position
    kn: np.ndarray  # Kneser–Ney log p of the closed target (Topic 3's fit); zeros on train
    spell: np.ndarray  # −log S of an OOV target under Topic 4's speller, else 0
    oov: np.ndarray  # target is OOV
    chars: int  # Topic 4's character count of the document
    F: np.ndarray  # n × 102 base features of the preceding token: [unit SVD vector, 1[<s>], 1[OOV]]
    oov_key: np.ndarray  # per position, the index of its preceding OOV string among the document's, else −1
    Kdot: np.ndarray  # ⟨ê(u_t), ê(u_i)⟩: SVD cosine, 1 for <s>·<s>, 1 for the same OOV string


@lru_cache(maxsize=None)
def doc_events(split: str) -> tuple[Doc, ...]:
    V = ng.vocabulary()
    idx, E = key_vectors()
    kn = nlm.kn_logprobs(split) if split != "train" else None
    spell_of: dict[str, float] = {}
    out, pos = [], 0
    for sents in documents(split):
        targets, prev, sent = [], [], []
        for j, s in enumerate(sents):
            targets += [*s, ng.EOS]
            prev += [ng.BOS, *s]
            sent += [j] * (len(s) + 1)
        n = len(targets)
        oov = np.array([w not in V for w in targets])
        for w in set(np.array(targets, dtype=object)[oov]):
            spell_of.setdefault(w, pce.spell_nll(w))
        closed_tok = [w if w in V else ng.UNK for w in targets]
        types = {w: i for i, w in enumerate(sorted(set(targets) | set(closed_tok)))}
        F = np.zeros((n, 102))
        oov_strings = sorted({u for u in prev if u != ng.BOS and u not in idx})
        oov_id = {u: i for i, u in enumerate(oov_strings)}
        oov_key = np.full(n, -1)
        for t, u in enumerate(prev):
            if u == ng.BOS:
                F[t, 100] = 1.0
            elif u in idx:
                F[t, :100] = E[idx[u]]
            else:
                F[t, 101] = 1.0
                oov_key[t] = oov_id[u]
        Kdot = F[:, :101] @ F[:, :101].T + ((oov_key[:, None] == oov_key[None, :]) & (oov_key[:, None] >= 0))
        out.append(Doc(targets=tuple(targets), prev=tuple(prev), closed=np.array([types[w] for w in closed_tok]),
                       raw=np.array([types[w] for w in targets]), sent=np.array(sent),
                       kn=kn[pos:pos + n] if kn is not None else np.zeros(n),
                       spell=np.array([spell_of[w] if o else 0.0 for w, o in zip(targets, oov)]), oov=oov,
                       chars=pce.n_chars(sents), F=F, oov_key=oov_key, Kdot=Kdot))
        pos += n
    if kn is not None:
        assert pos == len(kn)
    return tuple(out)


@lru_cache(maxsize=1)
def key_classes() -> dict[str, int]:
    """How many training words share their unit key with another word (cosine 1 to 10⁻⁹): Cor 2's β → ∞ limit pools
    them. They are words seen in identical contexts — once-seen pseudo-words and first names in fixed templates."""
    _, E = key_vectors()
    G = E @ E.T
    np.fill_diagonal(G, 0.0)
    same = G >= 1 - 1e-9
    return {"words": len(E), "pairs": int(same.sum() // 2), "shared_words": int(same.any(1).sum())}


def _strict_lower(n: int) -> np.ndarray:
    return np.tri(n, k=-1, dtype=bool)


def cache_weights(doc: Doc, beta: float, kernel: Params | None = None) -> np.ndarray:
    """Def 3's α_ti over the document's earlier positions (row 0 is all zeros: no history). β = ∞ is Cor 2's limit:
    uniform weight on the earlier positions whose key is closest. A learned kernel (Def 4) adds ⟨F_t A, F_i B⟩."""
    n, low = len(doc.targets), _strict_lower(len(doc.targets))
    if math.isinf(beta):
        S = np.where(low, doc.Kdot, -np.inf)
        mx = S.max(1, keepdims=True)
        W = (low & (S >= mx - 1e-9)).astype(float)
    else:
        S = beta * doc.Kdot
        if kernel is not None:
            S = S + (doc.F @ kernel["A"]) @ (doc.F @ kernel["B"]).T
        S = np.where(low, S, -np.inf)
        mx = S.max(1, keepdims=True)
        W = np.exp(S - np.where(np.isfinite(mx), mx, 0.0))
    den = W.sum(1, keepdims=True)
    return np.divide(W, den, out=np.zeros_like(W), where=den > 0)


def cache_probs(doc: Doc, beta: float, kernel: Params | None = None, open_vocab: bool = False,
                weights: np.ndarray | None = None) -> np.ndarray:
    """p_β(w_t | t) = Σ_{i<t} α_ti 1[w_i = w_t]: closed (OOV targets as one <unk>) or open (raw strings, so a
    repeated OOV word is copied)."""
    W = cache_weights(doc, beta, kernel) if weights is None else weights
    ids = doc.raw if open_vocab else doc.closed
    return (W * (ids[:, None] == ids[None, :])).sum(1)


def _mix(doc: Doc, pc: np.ndarray, lam: float, open_vocab: bool) -> np.ndarray:
    """log(λ p_KN + (1 − λ) p_cache), with λ = 1 at the first position; open: p_KN(<unk>) S(w) for OOV w."""
    base = doc.kn - (doc.spell if open_vocab else 0.0)
    lam_t = np.full(len(base), lam)
    lam_t[0] = 1.0
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log(lam_t) + base, np.log1p(-lam_t) + np.log(pc))


def mixture_logprobs(split: str, beta: float, lam: float, kernel: Params | None = None,
                     open_vocab: bool = False) -> list[np.ndarray]:
    """Per document, log p of every predicted position under KN ⊕ cache."""
    return [_mix(d, cache_probs(d, beta, kernel, open_vocab), lam, open_vocab) for d in doc_events(split)]


def kn_doc_logprobs(split: str, open_vocab: bool = False) -> list[np.ndarray]:
    return [d.kn - (d.spell if open_vocab else 0.0) for d in doc_events(split)]


def _tune_lambda(split: str, pcs: list[np.ndarray]) -> tuple[float, float]:
    """The λ in LAMBDAS minimizing mean closed log-loss on split, given the cache's probabilities."""
    docs = doc_events(split)
    n = sum(len(d.targets) for d in docs)
    loss = {lam: -sum(_mix(d, pc, lam, False).sum() for d, pc in zip(docs, pcs)) / n for lam in LAMBDAS}
    lam = min(loss, key=loss.get)
    return lam, loss[lam]


@lru_cache(maxsize=None)
def dev_grid() -> dict[float, tuple[float, float]]:
    """For each β, the dev-tuned λ and its dev nats per token (closed vocabulary)."""
    return {beta: _tune_lambda("dev", [cache_probs(d, beta) for d in doc_events("dev")]) for beta in BETAS}


@lru_cache(maxsize=1)
def tune_cache() -> tuple[float, float]:
    """(β, λ) minimizing dev log-loss over BETAS × LAMBDAS: held-out text tunes nothing."""
    grid = dev_grid()
    beta = min(grid, key=lambda b: grid[b][1])
    return beta, grid[beta][0]


def doc_rate(logps: list[np.ndarray], denoms: list[float], scale: float = 1.0) -> tuple[float, float]:
    """−Σ log p / Σ denominator, with its standard error from Topic 4's ratio estimator, documents as the units
    (a cache makes the sentences of one document dependent)."""
    return pce.ratio_estimate([-lp.sum() / scale for lp in logps], denoms)


def paired_rate(a: list[np.ndarray], b: list[np.ndarray], denoms: list[float], scale: float = 1.0):
    """The paired difference (a − b) in −log p per unit, by document."""
    return pce.ratio_estimate([(-x.sum() + y.sum()) / scale for x, y in zip(a, b)], denoms)


def repeat_info(doc: Doc, open_vocab: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per position: whether the target occurred earlier in the document, the distance back to its most recent
    earlier occurrence (0 if none), and whether that occurrence lies in an earlier sentence."""
    ids = doc.raw if open_vocab else doc.closed
    last, dist, cross = {}, np.zeros(len(ids), int), np.zeros(len(ids), bool)
    for t, w in enumerate(ids):
        if w in last:
            dist[t] = t - last[w]
            cross[t] = doc.sent[last[w]] < doc.sent[t]
        last[w] = t
    return dist > 0, dist, cross


@lru_cache(maxsize=1)
def cache_benchmark() -> dict:
    """§5: held-out nats per token for KN and KN ⊕ cache at every β (λ tuned on dev per β), the headline at the
    dev-chosen (β, λ) with clustered standard errors, and the gain by class and by distance to the earlier mention."""
    beta, lam = tune_cache()
    docs = doc_events("held")
    n_tok = [float(len(d.targets)) for d in docs]
    kn = kn_doc_logprobs("held")
    out: dict = {"beta": beta, "lambda": lam, "kn": doc_rate(kn, n_tok)}
    for b, (lb, dev_loss) in dev_grid().items():
        lp = mixture_logprobs("held", b, lb)
        out[f"beta={b}"] = {"lambda": lb, "dev": dev_loss, "held": doc_rate(lp, n_tok)[0]}
    mix = mixture_logprobs("held", beta, lam)
    out["cache"] = doc_rate(mix, n_tok)
    out["gain"] = paired_rate(kn, mix, n_tok)
    rep = [repeat_info(d) for d in docs]
    R = np.concatenate([r[0] for r in rep])
    D = np.concatenate([r[1] for r in rep])
    X = np.concatenate([r[2] for r in rep])
    k_all, m_all = np.concatenate(kn), np.concatenate(mix)
    out["repeat_share"] = float(R.mean())
    out["repeat"] = (float(-k_all[R].mean()), float(-m_all[R].mean()))
    out["first"] = (float(-k_all[~R].mean()), float(-m_all[~R].mean()))
    out["bands"] = []
    for lo, hi in DISTANCE_BANDS:
        sel = R & (D >= lo) & (D <= hi)
        out["bands"].append({"lo": lo, "hi": hi, "share": float(sel.sum() / R.sum()),
                             "cross_sentence": float(X[sel].mean()), "kn": float(-k_all[sel].mean()),
                             "cache": float(-m_all[sel].mean())})
    out["gain_from_cross_sentence"] = float((m_all - k_all)[R & X].sum() / (m_all - k_all).sum())
    chars = [float(d.chars) for d in docs]
    out["kn_bpc"] = doc_rate(kn_doc_logprobs("held", True), chars, LN2)
    out["cache_bpc"] = doc_rate(mixture_logprobs("held", beta, lam, open_vocab=True), chars, LN2)
    return out


# Def 4: the learned kernel, trained on the pointer likelihood.

@dataclass(frozen=True)
class KernelConfig:
    rank: int = 16
    epochs: int = 12
    lr: float = 0.01
    batch: int = 10  # documents per Adam step
    init: float = 0.01  # A, B ~ N(0, init²): training starts at Def 3 to within 10⁻⁴ in every score
    stream: int = 960


KERNEL_CFG = KernelConfig()


def _lse_rows(S):
    """Row-wise log-sum-exp that tolerates −∞ entries and complex inputs (for the complex step)."""
    mx = S.real.max(1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(S - mx).sum(1))


def pointer_loss(P: Params, docs, beta: float):
    """−mean over repeat positions of log Σ_{i<t: w_i = w_t} α_ti (Merity et al.'s pointer likelihood)."""
    total, count = 0.0, 0
    for d in docs:
        low = _strict_lower(len(d.targets))
        same = (d.closed[:, None] == d.closed[None, :]) & low
        rep = same.any(1)
        S = np.where(low, beta * d.Kdot + (d.F @ P["A"]) @ (d.F @ P["B"]).T, -np.inf)[rep]
        total = total + (_lse_rows(S) - _lse_rows(np.where(same[rep], S, -np.inf))).sum()
        count += int(rep.sum())
    return total / count


def pointer_loss_grad(P: Params, docs, beta: float) -> tuple[float, Params]:
    """The pointer loss and its gradient: ∂ℓ_t/∂s_ti = α_ti − α_ti 1[w_i = w_t] / P_t, then ∂/∂A = Fᵀ G (F B) and
    ∂/∂B = Fᵀ Gᵀ (F A)."""
    total, count = 0.0, 0
    dA, dB = np.zeros_like(P["A"]), np.zeros_like(P["B"])
    for d in docs:
        n, low = len(d.targets), _strict_lower(len(d.targets))
        same = (d.closed[:, None] == d.closed[None, :]) & low
        rep = same.any(1)
        FA, FB = d.F @ P["A"], d.F @ P["B"]
        S = np.where(low, beta * d.Kdot + FA @ FB.T, -np.inf)
        S[~rep] = 0.0  # rows without a repeat carry no loss
        a = np.exp(S - S.max(1, keepdims=True))
        a = np.where(low, a, 0.0)
        a[~rep] = 0.0
        a[rep] /= a[rep].sum(1, keepdims=True)
        Pt = np.where(rep, (a * same).sum(1), 1.0)
        total += float(-np.log(Pt[rep]).sum())
        count += int(rep.sum())
        G = a - a * same / Pt[:, None]
        dA += d.F.T @ (G @ FB)
        dB += d.F.T @ (G.T @ FA)
    return total / count, {"A": dA / count, "B": dB / count}


@dataclass(frozen=True)
class TrainedKernel:
    params: Params
    epoch: int  # the dev-selected epoch (1-based)
    history: tuple[dict, ...]  # per epoch: train pointer loss, dev nats per token at its dev-tuned λ, that λ
    beta: float
    lam: float


@lru_cache(maxsize=None)
def train_learned_kernel(cfg: KernelConfig = KERNEL_CFG) -> TrainedKernel:
    beta = tune_cache()[0]
    rng = np.random.default_rng([SEED, cfg.stream])
    P = {"A": rng.normal(0, cfg.init, (102, cfg.rank)), "B": rng.normal(0, cfg.init, (102, cfg.rank))}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    train = doc_events("train")
    step, hist, best = 0, [], None
    for ep in range(1, cfg.epochs + 1):
        order = rng.permutation(len(train))
        losses = []
        for b in range(0, len(order), cfg.batch):
            loss, G = pointer_loss_grad(P, [train[i] for i in order[b:b + cfg.batch]], beta)
            losses.append(loss)
            step += 1
            c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
            for k in P:
                m[k] = nlm.ADAM_B1 * m[k] + (1 - nlm.ADAM_B1) * G[k]
                s[k] = nlm.ADAM_B2 * s[k] + (1 - nlm.ADAM_B2) * G[k] ** 2
                P[k] -= cfg.lr * (m[k] / c1) / (np.sqrt(s[k] / c2) + nlm.ADAM_EPS)
        lam, dev = _tune_lambda("dev", [cache_probs(d, beta, P) for d in doc_events("dev")])
        hist.append({"train_pointer": float(np.mean(losses)), "dev": dev, "lambda": lam})
        if best is None or dev < best[1]:
            best = (ep, dev, lam, {k: v.copy() for k, v in P.items()})
    return TrainedKernel(params=best[3], epoch=best[0], history=tuple(hist), beta=beta, lam=best[2])


@lru_cache(maxsize=1)
def learned_benchmark() -> dict:
    """Def 4 on held out against Def 3, with the paired standard error by document."""
    tk = train_learned_kernel()
    docs = doc_events("held")
    n_tok = [float(len(d.targets)) for d in docs]
    beta, lam = tune_cache()
    fixed = mixture_logprobs("held", beta, lam)
    learned = mixture_logprobs("held", tk.beta, tk.lam, tk.params)
    chars = [float(d.chars) for d in docs]
    pointer0 = pointer_loss({k: np.zeros_like(v) for k, v in tk.params.items()}, doc_events("held"), beta)
    return {"learned": doc_rate(learned, n_tok), "fixed": doc_rate(fixed, n_tok),
            "learned_minus_fixed": paired_rate(learned, fixed, n_tok),
            "learned_bpc": doc_rate(mixture_logprobs("held", tk.beta, tk.lam, tk.params, True), chars, LN2),
            "held_pointer_fixed": float(pointer0), "held_pointer_learned": float(pointer_loss(tk.params, docs, beta))}


# --------------------------------------------------------------------------------------------
# 6. Kernels that factor.
# --------------------------------------------------------------------------------------------


def numerical_rank(M: np.ndarray, rtol: float = 1e-9) -> int:
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rtol * s[0]).sum())


def log_attention_rank(Q: np.ndarray, K: np.ndarray, tau: float) -> int:
    """Prop 7: rank of log A = S − λ𝟙ᵀ for one bidirectional head."""
    return numerical_rank(nlm.log_softmax(Q @ K.T / tau))


def best_head_fit(P: np.ndarray, d_k: int) -> float:
    """Prop 7's realizability gap: min over row shifts c and rank-d_k M of ‖log P − c𝟙ᵀ − M‖_F, relative to
    ‖log P J‖_F. The shift is optimal at the row means, so this is Eckart–Young on log P J (J the centering
    matrix): the tail of its singular values beyond d_k."""
    L = np.log(P)
    s = np.linalg.svd(L - L.mean(1, keepdims=True), compute_uv=False)
    return float(math.sqrt((s[d_k:] ** 2).sum() / (s**2).sum()))


def positive_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Prop 8: φ(x) = exp(Wx − ‖x‖²/2)/√m for the m rows w of W, so φ(x)·φ(y) estimates e^{⟨x,y⟩}."""
    return np.exp(X @ W.T - 0.5 * (X * X).sum(-1, keepdims=True)) / math.sqrt(len(W))


def trig_features(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Random Fourier features for the same kernel: e^{⟨x,y⟩} = e^{‖x‖²/2} e^{‖y‖²/2} e^{−‖x−y‖²/2}, with the Gaussian
    factor estimated by √(2/m) cos(Wx + b) (Rahimi and Recht 2007). They can be negative."""
    return np.exp(0.5 * (X * X).sum(-1, keepdims=True)) * math.sqrt(2 / len(W)) * np.cos(X @ W.T + b)


def rf_moments(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Prop 8: the mean e^{⟨x,y⟩} and variance e^{2⟨x,y⟩}(e^{‖x+y‖²} − 1) of one positive feature product."""
    return math.exp(x @ y), math.exp(2 * x @ y) * math.expm1((x + y) @ (x + y))


def linear_attention_quadratic(phiQ: np.ndarray, phiK: np.ndarray, V: np.ndarray, strict: bool = False):
    """Σ_{j≤t} (φ(q_t)·φ(k_j)) v_j / Σ_{j≤t} φ(q_t)·φ(k_j), formed as an n × n matrix (j < t when strict)."""
    A = np.tril(phiQ @ phiK.T, -1 if strict else 0)
    den = A.sum(1, keepdims=True)
    return np.divide(A @ V, den, out=np.zeros((len(A), V.shape[1])), where=den > 0)


def linear_attention_recurrent(phiQ: np.ndarray, phiK: np.ndarray, V: np.ndarray, strict: bool = False):
    """Prop 9: the same smoother as a recurrence with state S_t = S_{t−1} + φ(k_t)ᵀv_t and z_t = z_{t−1} + φ(k_t)ᵀ,
    of fixed size m(d_v + 1) whatever t is (read before the update when strict)."""
    S, z = np.zeros((phiK.shape[1], V.shape[1])), np.zeros(phiK.shape[1])
    out = np.zeros((len(phiQ), V.shape[1]))
    for t in range(len(phiQ)):
        if not strict:
            S += np.outer(phiK[t], V[t])
            z += phiK[t]
        den = phiQ[t] @ z
        out[t] = phiQ[t] @ S / den if den > 0 else 0.0
        if strict:
            S += np.outer(phiK[t], V[t])
            z += phiK[t]
    return out


def doc_keys(doc: Doc) -> np.ndarray:
    """Def 3's explicit unit keys ê(u_t): [SVD vector, <s>, one coordinate per distinct OOV string]."""
    k = int(doc.oov_key.max()) + 1
    O = np.zeros((len(doc.targets), k))
    has = doc.oov_key >= 0
    O[np.flatnonzero(has), doc.oov_key[has]] = 1.0
    return np.hstack([doc.F[:, :101], O])


RF_MS = (16, 64, 256, 1024)
RF_BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)


def random_feature_weights(doc: Doc, beta: float, W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Def 3's weights with e^{β⟨ê_t, ê_i⟩} replaced by m positive features of √β ê (Prop 8). The first 101 columns
    of W are shared by every document; each OOV coordinate gets a fresh column from rng."""
    E = math.sqrt(beta) * doc_keys(doc)
    Wd = np.hstack([W, rng.standard_normal((len(W), E.shape[1] - 101))])
    phi = positive_features(E, Wd)
    K = np.tril(phi @ phi.T, -1)
    den = K.sum(1, keepdims=True)
    return np.divide(K, den, out=np.zeros_like(K), where=den > 0)


@lru_cache(maxsize=1)
def random_feature_benchmark() -> dict:
    """Prop 8 on held-out documents: the mean total-variation distance between the random-feature cache's weights
    and the exact ones, over every position with a history, for β in RF_BETAS and m in RF_MS; and, at the dev-chosen
    (β, λ), the held-out nats per token of KN ⊕ random-feature cache against KN ⊕ exact cache."""
    docs = doc_events("held")
    n = sum(len(d.targets) for d in docs)
    beta_star, lam = tune_cache()
    out: dict = {"tv": {}, "nats": [], "exact": -sum(lp.sum() for lp in mixture_logprobs("held", beta_star, lam)) / n}
    for beta in RF_BETAS:
        exact = [cache_weights(d, beta) for d in docs]
        row = []
        for m in RF_MS:
            rng = np.random.default_rng([SEED, 970, m, int(beta * 100)])
            W = rng.standard_normal((m, 101))
            R = [random_feature_weights(d, beta, W, rng) for d in docs]
            row.append(float(np.concatenate([0.5 * np.abs(r - e).sum(1)[1:] for r, e in zip(R, exact)]).mean()))
            if beta == beta_star:
                lp = [_mix(d, cache_probs(d, beta, weights=r), lam, False) for d, r in zip(docs, R)]
                out["nats"].append(-sum(x.sum() for x in lp) / n)
        out["tv"][beta] = row
    return out


# --------------------------------------------------------------------------------------------
# 7. Finance case study: the filing as memory.
# --------------------------------------------------------------------------------------------

TOPIC5_RNN_DYN_CACHE_EDGAR_BPC = 3.6501  # Topic 5's dynamic RNN with the PPM-C spelling cache (its page, its harness)
EDGAR_NAMES = ("Apple", "ExxonMobil", "Microsoft")  # fc.load_edgar_excerpts() order (sorted file names)


@lru_cache(maxsize=1)
def finance_summary() -> dict:
    """EDGAR in open-vocabulary bits per character: KN, KN with Topic 5's PPM-C spelling cache, KN ⊕ cache at the
    dev-chosen (β, λ) and with the learned kernel, and lzma given the training text; the share of the cache's gain
    that comes from repeated OOV words; and two curated OOV terms per filing."""
    beta, lam = tune_cache()
    tk = train_learned_kernel()
    docs = doc_events("edgar")
    chars = [float(d.chars) for d in docs]
    nc = sum(chars)
    kn = kn_doc_logprobs("edgar", True)
    mix = mixture_logprobs("edgar", beta, lam, open_vocab=True)
    learned = mixture_logprobs("edgar", tk.beta, tk.lam, tk.params, open_vocab=True)
    closed_kn = -sum(d.kn.sum() for d in docs)
    out = {"kn_bpc": -sum(x.sum() for x in kn) / nc / LN2,
           "kn_ppm_bpc": (closed_kn + nlm.ppm_cache_spelled_nats("edgar")) / nc / LN2,
           "cache_bpc": -sum(x.sum() for x in mix) / nc / LN2,
           "learned_bpc": -sum(x.sum() for x in learned) / nc / LN2,
           "lzma_bpc": nlm.lzma_given_bpc("edgar"), "topic5_best_bpc": TOPIC5_RNN_DYN_CACHE_EDGAR_BPC,
           "oov_share": float(np.concatenate([d.oov for d in docs]).mean())}
    gain = np.concatenate([m - k for m, k in zip(mix, kn)])
    rep = np.concatenate([repeat_info(d, True)[0] for d in docs])
    oov = np.concatenate([d.oov for d in docs])
    out["gain_share_oov_repeat"] = float(gain[rep & oov].sum() / gain.sum())
    out["gain_share_vocab_repeat"] = float(gain[rep & ~oov].sum() / gain.sum())
    out["gain_share_first"] = float(gain[~rep].sum() / gain.sum())
    pcs = [cache_probs(d, beta, open_vocab=True) for d in docs]
    oracle = {l: -sum(_mix(d, pc, l, True).sum() for d, pc in zip(docs, pcs)) / nc / LN2 for l in LAMBDAS}
    out["oracle_lambda"] = min(oracle, key=oracle.get)
    out["oracle_bpc"] = oracle[out["oracle_lambda"]]  # a diagnostic, tuned on EDGAR itself: never a result
    out["per_filing"] = [{"name": name, "kn_bpc": -k.sum() / d.chars / LN2, "cache_bpc": -m.sum() / d.chars / LN2}
                         for name, d, k, m in zip(EDGAR_NAMES, docs, kn, mix)]
    out["curated"] = []
    for name, d, k, m in zip(EDGAR_NAMES, docs, kn, mix):
        counts: dict[str, list[int]] = {}
        for t, (w, o) in enumerate(zip(d.targets, d.oov)):
            if o:
                counts.setdefault(w, []).append(t)
        terms = sorted((w for w, ts in counts.items() if len(ts) >= 2 and len(w) >= 3),
                       key=lambda w: (-len(counts[w]), counts[w][0]))
        for w in terms[:2]:
            t1, t2 = counts[w][:2]
            out["curated"].append({"filing": name, "word": w, "first": t1, "second": t2, "count": len(counts[w]),
                                   "kn_bits": (-k[t1] / LN2, -k[t2] / LN2), "cache_bits": (-m[t1] / LN2, -m[t2] / LN2)})
    return out


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX's JSON in Phase B).
# --------------------------------------------------------------------------------------------

LAB_C_GRID = (0.0, *np.round(np.geomspace(0.25, 400, 40), 4))
LAB_CASES = {"smooth-32": DENOISERS[0], "smooth-64": DENOISERS[1], "rough-32": DENOISERS[2]}
LAB_ATTN_TOP = 8  # earlier positions shown per curated token


def _r(x, nd: int = 3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _circle_case(cfg: DenoiserConfig) -> dict:
    power = SPECTRA[cfg.spectrum]
    risks = [circle_risk(c, cfg.n, power, SIGMA) for c in LAB_C_GRID]
    rng = np.random.default_rng([SEED, 980, cfg.n, len(power)])
    U, y, f = denoiser_tasks(rng, 1, cfg.n, cfg.spectrum)
    th = np.arctan2(U[0, :, 1], U[0, :, 0])
    tr = train_denoiser(cfg)
    return {"n": cfg.n, "spectrum": cfg.spectrum, "risk": _r([r["risk"] for r in risks], 4),
            "bias2": _r([r["bias2"] for r in risks], 4), "var": _r([r["var"] for r in risks], 4),
            "gains": [_r(circle_gains(cfg.n, c, np.arange(1, 9)), 3) for c in LAB_C_GRID],
            "c_star": _r(tr.c_star), "c_learned": _r(tr.c), "risk_star": _r(circle_risk(tr.c_star, cfg.n, power, SIGMA)["risk"], 4),
            "theta": _r(th, 4), "f": _r(f[0], 3), "y": _r(y[0], 3)}


def _lab_doc(name: str, doc: Doc, beta: float, lam: float, tk: TrainedKernel, curated: list[int]) -> dict:
    variants = {"kn": None, "b0": 0.0, "bstar": beta, "binf": math.inf}
    bits = {}
    for key, b in variants.items():
        if b is None:
            lp = doc.kn - doc.spell
        else:
            lp = _mix(doc, cache_probs(doc, b, open_vocab=True), dev_grid()[b][0], True)
        bits[key] = lp
    bits["learned"] = _mix(doc, cache_probs(doc, tk.beta, tk.params, True), tk.lam, True)
    W = cache_weights(doc, beta)
    attn = []
    for t in curated:
        top = np.argsort(-W[t])[:LAB_ATTN_TOP]
        attn.append({"t": int(t), "pos": [int(i) for i in top], "w": _r(W[t, top], 3)})
    return {"name": name, "tokens": list(doc.targets), "sent": doc.sent.tolist(), "oov": doc.oov.astype(int).tolist(),
            "bits": {k: _r(-v / LN2, 2) for k, v in bits.items()},
            "bpc": {k: _r(-v.sum() / doc.chars / LN2, 3) for k, v in bits.items()}, "attn": attn}


@lru_cache(maxsize=1)
def lab_held_doc() -> int:
    """The held-out document the lab shows: the one with the largest open-vocabulary gain per token among documents
    of at most 260 positions."""
    beta, lam = tune_cache()
    best, arg = -math.inf, 0
    for i, d in enumerate(doc_events("held")):
        if len(d.targets) <= 260:
            g = (_mix(d, cache_probs(d, beta, open_vocab=True), lam, True) - (d.kn - d.spell)).mean()
            if g > best:
                best, arg = g, i
    return arg


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    beta, lam = tune_cache()
    tk = train_learned_kernel()
    sweep = lag_sweep()
    lt = lag_attention_training()
    hd = doc_events("held")[lab_held_doc()]
    g = _mix(hd, cache_probs(hd, beta, open_vocab=True), lam, True) - (hd.kn - hd.spell)
    held_curated = sorted(int(t) for t in np.argsort(-g)[:3])
    fin = finance_summary()
    docs = [_lab_doc("held-out filing", hd, beta, lam, tk, held_curated)]
    for name, d in zip(EDGAR_NAMES, doc_events("edgar")):
        docs.append(_lab_doc(name, d, beta, lam, tk, [c["second"] for c in fin["curated"] if c["filing"] == name]))
    return {
        "circle": {"c_grid": [float(c) for c in LAB_C_GRID], "sigma": SIGMA,
                   "cases": {k: _circle_case(cfg) for k, cfg in LAB_CASES.items()}},
        "reach": {"lags": sweep["lags"], **{k: _r(np.log10(sweep[k]), 2) for k in ("attention", "elman", "lstm-1", "lstm-3")},
                  "accuracy": {"attention": _r(lt["accuracy"]), "attention_lag600": _r(lt["accuracy_lag600"]),
                               **TOPIC5_LAG_ACCURACY}},
        "cache": {"beta": None if math.isinf(beta) else beta, "lambda": lam,
                  "betas": [None if math.isinf(b) else b for b in BETAS],
                  "held_by_beta": _r([cache_benchmark()[f"beta={b}"]["held"] for b in BETAS], 4),
                  "dev_by_beta": _r([dev_grid()[b][1] for b in BETAS], 4), "documents": docs},
    }


# --------------------------------------------------------------------------------------------
# Summary.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def summary() -> dict[str, float]:
    cb, lb, fin = cache_benchmark(), learned_benchmark(), finance_summary()
    sweep, lt, rf = lag_sweep(), lag_attention_training(), random_feature_benchmark()
    out: dict[str, float] = {}
    for cfg in DENOISERS:
        tr = train_denoiser(cfg)
        out[f"denoiser_{cfg.spectrum}_{cfg.n}_c"] = tr.c
        out[f"denoiser_{cfg.spectrum}_{cfg.n}_c_star"] = tr.c_star
    out["lag60_attention_reach"] = sweep["attention"][sweep["lags"].index(nlm.LAG)]
    for name in ("elman", "lstm-1", "lstm-3"):
        out[f"lag60_{name}_reach"] = sweep[name][sweep["lags"].index(nlm.LAG)]
    out["lag_attention_accuracy"], out["lag_attention_accuracy_600"] = lt["accuracy"], lt["accuracy_lag600"]
    out["cache_beta"], out["cache_lambda"] = cb["beta"], cb["lambda"]
    out["held_kn_nats"], out["held_cache_nats"] = cb["kn"][0], cb["cache"][0]
    out["held_gain_nats"], out["held_gain_se"] = cb["gain"]
    out["held_kn_bpc"], out["held_cache_bpc"] = cb["kn_bpc"][0], cb["cache_bpc"][0]
    out["held_learned_nats"], out["held_learned_minus_fixed"] = lb["learned"][0], lb["learned_minus_fixed"][0]
    out["held_learned_minus_fixed_se"] = lb["learned_minus_fixed"][1]
    for beta, row in rf["tv"].items():
        out[f"rf_tv_beta{beta:g}_m1024"] = row[-1]
    out["rf_nats_m1024"], out["rf_nats_exact"] = rf["nats"][-1], rf["exact"]
    for k in ("kn_bpc", "kn_ppm_bpc", "cache_bpc", "learned_bpc", "lzma_bpc", "gain_share_oov_repeat"):
        out[f"edgar_{k}"] = fin[k]
    return out


# --------------------------------------------------------------------------------------------
# Verification harness.
# --------------------------------------------------------------------------------------------


def _within(x: np.ndarray, se: np.ndarray, k: float = 4.0) -> bool:
    return bool(np.all(np.abs(x) <= k * se))


def test_documents_partition_corpus():
    """The regrouped documents concatenate to Topic 5's splits, and the per-position KN log-probabilities and
    spellings reproduce Topic 4's closed and open accounting, and Topic 5's KN row, exactly."""
    for split in SPLITS:
        assert tuple(s for d in documents(split) for s in d) == nlm.corpus()[split], split
    for split in ("held", "edgar"):
        a, docs = pce.sentence_nll("ikn", split), doc_events(split)
        assert math.isclose(-sum(d.kn.sum() for d in docs), a[:, 0].sum(), rel_tol=1e-12)
        assert math.isclose(sum((d.spell - d.kn).sum() for d in docs), a[:, 3].sum(), rel_tol=1e-12)
        assert sum(len(d.targets) for d in docs) == a[:, 1].sum() and sum(d.chars for d in docs) == a[:, 2].sum()
        bpc = -sum(x.sum() for x in kn_doc_logprobs(split, True)) / sum(d.chars for d in docs) / LN2
        assert abs(bpc - nlm.comparison_row("kn", split)["bpc"]) < 1e-12


def test_attention_is_nadaraya_watson():
    """Prop 1: attention equals Def 1's smoother with the exponential kernel and the weighted Gaussian
    Nadaraya–Watson estimator (bandwidth √τ, weights e^{‖k‖²/2τ}) to 10⁻¹²; the factorization is exact; equal key norms
    give the unweighted estimator and, on a sphere, the vMF kernel; outputs lie in the values' hull; and of two keys
    equally far from the query, the longer one gets exactly e^{Δ‖k‖²/2τ} times the weight."""
    rng = np.random.default_rng([SEED, 900])
    for _ in range(50):
        n, d, dv = int(rng.integers(3, 30)), int(rng.integers(1, 17)), int(rng.integers(1, 5))
        Q, K, V = rng.normal(0, 0.7, (4, d)), rng.normal(0, 0.7, (n, d)), rng.normal(size=(n, dv))
        tau = math.sqrt(d)
        O = attention(Q, K, V)[0]
        for i, q in enumerate(Q):
            g = gaussian_form(q, K, tau)
            assert np.allclose(g["query"] * g["omega"] * g["gauss"], np.exp(K @ q / tau), rtol=1e-12, atol=0)
            assert np.allclose(O[i], kernel_smoother(q, K, V, lambda q, K: np.exp(K @ q / tau)), rtol=1e-12, atol=1e-13)
            assert np.allclose(O[i], kernel_smoother(q, K, V, lambda q, K: g["omega"] * g["gauss"]),
                               rtol=1e-12, atol=1e-13)
            assert np.all(O[i] <= V.max(0) + 1e-12) and np.all(O[i] >= V.min(0) - 1e-12)
        r = 1.7
        Ks = r * K / np.linalg.norm(K, axis=1, keepdims=True)
        qs = r * Q[0] / np.linalg.norm(Q[0])
        g = gaussian_form(qs, Ks, tau)
        o = attention(qs[None], Ks, V)[0][0]
        assert np.allclose(o, kernel_smoother(qs, Ks, V, lambda q, K: g["gauss"]), rtol=1e-12, atol=1e-13)
        cos = Ks @ qs / r**2
        assert np.allclose(o, kernel_smoother(qs, Ks, V, lambda q, K: np.exp(r**2 / tau * cos)), rtol=1e-12, atol=1e-13)
    q, u = rng.normal(size=4), rng.normal(size=4)
    R = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    K2 = np.stack([q + u, q + R @ u])
    A = attention(q[None], K2, np.eye(2), tau=2.0)[1][0]
    assert math.isclose(A[1] / A[0], math.exp((K2[1] @ K2[1] - K2[0] @ K2[0]) / 4.0), rel_tol=1e-12)


def test_mixture_posterior():
    """Prop 2: Bayes' rule for the key-centered mixture gives the softmax weights to 10⁻¹², and on 10⁵ draws of the
    generative model the residual V − o(Q) is uncorrelated with 1, Q₁, and Q₂ (each within 4 standard errors) — the
    defining property of E[V | Q]. The smoother that ignores the design weights fails the same test."""
    rng = np.random.default_rng([SEED, 901])
    K, V, tau = rng.normal(0, 1.5, (5, 2)), rng.normal(size=(5, 3)), 0.8
    omega = np.exp((K * K).sum(1) / (2 * tau))
    for q in rng.normal(size=(20, 2)):
        assert np.allclose(mixture_posterior(q, K, tau, omega), attention(q[None], K, V, tau)[1][0], rtol=1e-12, atol=0)
    Qs, Vs = sample_mixture(rng, K, V, tau, omega, 100_000)
    resid = Vs - attention(Qs, K, V, tau)[0]
    wrong = Vs - np.stack([mixture_posterior(q, K, tau, np.ones(5)) for q in Qs]) @ V
    failed = False
    for h in (np.ones(len(Qs)), Qs[:, 0], Qs[:, 1]):
        x = resid * h[:, None]
        assert _within(x.mean(0), x.std(0) / math.sqrt(len(x)))
        y = wrong * h[:, None]
        failed |= not _within(y.mean(0), y.std(0) / math.sqrt(len(y)))
    assert failed


def test_order_blindness():
    """Prop 3: (i) bidirectional stacks (depth 1 and 3) are permutation-equivariant to 10⁻¹²; (ii) one causal layer,
    reading a held-out sentence through Topic 6's vectors, gives the last position the same output when the prefix is
    reversed; (iii) two causal layers do not."""
    rng = np.random.default_rng([SEED, 902])
    X = rng.normal(size=(7, 6))
    for depth in (1, 3):
        layers, perm = random_layers(rng, 6, depth), rng.permutation(7)
        assert np.allclose(bidirectional_stack(X[perm], layers), bidirectional_stack(X, layers)[perm],
                           rtol=1e-12, atol=1e-13)
    idx, E = key_vectors()
    s = next(s for s in nlm.corpus()["held"] if len(s) >= 8 and all(w in idx for w in s[:8]))
    X = E[[idx[w] for w in s[:8]]]
    Xr = X[[6, 5, 4, 3, 2, 1, 0, 7]]
    one, two = random_layers(rng, 100, 1, 2.0), random_layers(rng, 100, 2, 2.0)
    L = one[0]
    o1 = attention(X @ L["Wq"], X @ L["Wk"], X @ L["Wv"], causal=True)[0][-1]
    o2 = attention(Xr @ L["Wq"], Xr @ L["Wk"], Xr @ L["Wv"], causal=True)[0][-1]
    assert np.allclose(o1, o2, rtol=1e-12, atol=1e-14)
    assert np.allclose(causal_stack(X, one)[-1], causal_stack(Xr, one)[-1], rtol=1e-12, atol=1e-14)
    assert np.abs(causal_stack(X, two)[-1] - causal_stack(Xr, two)[-1]).max() > 1e-3


def test_temperature_limits():
    """Prop 4: both bounds hold on random instances for τ from 10⁻² to 10², and each is within a factor of 3 of the
    truth at its own end on a two-cluster example."""
    rng = np.random.default_rng([SEED, 903])
    for _ in range(30):
        n = int(rng.integers(2, 20))
        s, V = 2 * rng.normal(size=n), rng.normal(size=(n, 3))
        for tau in np.geomspace(1e-2, 1e2, 9):
            b = limit_bounds(s, V, tau)
            assert b["hard"] <= b["hard_bound"] * (1 + 1e-9) + 1e-15
            assert b["mean"] <= b["mean_bound"] * (1 + 1e-9) + 1e-15
    s, V = np.array([1.0] * 4 + [0.0] * 4), np.array([[1.0]] * 4 + [[-1.0]] * 4)
    lo, hi = limit_bounds(s, V, 0.05), limit_bounds(s, V, 50.0)
    assert 1 - 1e-6 <= lo["hard_bound"] / lo["hard"] < 3 and 1 - 1e-6 <= hi["mean_bound"] / hi["mean"] < 3


def test_circle_fourier_multiplier():
    """Thm 1: on a grid of (n, c) the circulant A has numpy's FFT eigenvalues equal to the aliased Bessel gains to
    10⁻¹², the Fourier vectors are its eigenvectors, 0 ≤ ρ_k < 1 = ρ₀ (ρ_k = 0 at c = 0), and Σ a_m² matches; ρ₁ is
    the mean resultant length I₁(c)/I₀(c) once aliasing is negligible; the exact bias is (1/n)‖(A − I)f‖², and the
    risk matches Monte Carlo over the noise."""
    for n in (8, 31, 64, 200):
        for c in (0.0, 0.3, 2.0, 7.3, 40.0, 300.0):
            A, rho = circle_attention(n, c, 0.37), circle_gains(n, c)
            assert np.abs(np.fft.fft(A[0]).real - rho).max() < 1e-12 and np.abs(np.fft.fft(A[0]).imag).max() < 1e-12
            th = circle_angles(n, 0.37)
            for k in range(n):
                e = np.exp(1j * k * th)
                assert np.abs(A @ e - rho[k] * e).max() < 1e-12
            assert rho[0] == 1.0 and np.all(rho[1:] >= 0) and np.all(rho[1:] < 1)
            assert (rho[1] > 0) if c > 0 else np.all(rho[1:] == 0)
            assert math.isclose(circle_variance_factor(n, c), float((A[0] ** 2).sum()), rel_tol=1e-12)
    assert math.isclose(circle_gains(200, 7.3, [1])[0], special.ive(1, 7.3) / special.ive(0, 7.3), rel_tol=1e-12)
    rng = np.random.default_rng([SEED, 920])
    n, c, sigma = 32, 6.0, SIGMA
    th = circle_angles(n)
    f = np.cos(th) + 0.5 * np.sin(3 * th)
    power = np.array([0.5, 0.0, 0.125])
    A = circle_attention(n, c)
    r = circle_risk(c, n, power, sigma)
    assert math.isclose(r["bias2"], float(((A @ f - f) ** 2).mean()), rel_tol=1e-12)
    losses = (((f + sigma * rng.standard_normal((4000, n))) @ A.T - f) ** 2).mean(1)
    assert abs(losses.mean() - r["risk"]) < 4 * losses.std() / math.sqrt(len(losses))


def test_circle_rate():
    """Cor 1, on the exact formula: c*(n)/(Cn^{2/5}) is within 1% of 1 at n = 10⁸ for both spectra, and the log–log
    slope of c* from 10⁶ to 10⁸ is within 0.01 of 2/5."""
    for power in SPECTRA.values():
        pre = rate_prefactor(power, SIGMA)
        cs = {n: optimal_c(n, power, SIGMA) for n in (10**6, 10**8)}
        assert abs(cs[10**8] / (pre * 1e8**0.4) - 1) < 0.01
        assert abs(math.log(cs[10**8] / cs[10**6]) / math.log(100) - 0.4) < 0.01


def test_trained_denoiser():
    """The trained denoiser: its hand-written gradient matches the complex step to 10⁻¹⁰ relative (and central
    differences to 10⁻⁶); the loss at M = c*I averages to Thm 1's risk over fresh tasks; after training, the learned M
    is within 5% of c*I in Frobenius norm and nearly isotropic, for each configuration; the rough spectrum learns a
    sharper kernel than the smooth one, and a longer context a sharper one too."""
    rng = np.random.default_rng([SEED, 904])
    U, y, f = denoiser_tasks(rng, 3, 8, "rough")
    P = {"Wq": rng.normal(size=(2, 2)), "Wk": rng.normal(size=(2, 2))}
    _, G = denoiser_loss_grad(P, U, y, f)
    for key in P:
        fn = lambda t, key=key: denoiser_loss({**P, key: t}, U, y, f)  # noqa: E731
        assert np.allclose(G[key], nlm.complex_step_grad(fn, P[key]), rtol=1e-10, atol=1e-14)
        assert np.allclose(G[key], nlm.central_diff_grad(fn, P[key]), rtol=1e-6, atol=1e-9)
    cs = optimal_c(32, SPECTRA["smooth"], SIGMA)
    Pstar = {"Wq": math.sqrt(cs) * 2**0.25 * np.eye(2), "Wk": math.sqrt(cs) * 2**0.25 * np.eye(2)}
    assert np.allclose(denoiser_score_matrix(Pstar), cs * np.eye(2), rtol=1e-12)
    loss = [denoiser_loss_grad(Pstar, *denoiser_tasks(rng, 1, 32, "smooth"))[0] for _ in range(3000)]
    assert abs(np.mean(loss) - circle_risk(cs, 32, SPECTRA["smooth"], SIGMA)["risk"]) < 4 * np.std(loss) / math.sqrt(3000)
    tr = {(c.spectrum, c.n): train_denoiser(c) for c in DENOISERS}
    for t in tr.values():
        assert np.linalg.norm(t.M - t.c_star * np.eye(2)) / (t.c_star * math.sqrt(2)) < 0.05
        assert t.anisotropy < 0.05
    assert tr["rough", 32].c > 3 * tr["smooth", 32].c and tr["rough", 32].c_star > tr["smooth", 32].c_star
    assert tr["smooth", 64].c > tr["smooth", 32].c and tr["smooth", 64].c_star > tr["smooth", 32].c_star


def test_scaling_and_saturation():
    """Thm 2: (i) Var⟨q, k⟩ = d for Rademacher coordinates (within 4 standard errors); (ii) ‖J‖₂ ≤ tr J = 1 − ‖α‖² ≤
    2(1 − α_max), and J is Topic 5's softmax Jacobian; (iii) unscaled, the mean of 1 − α_max falls at every d and ends
    below 0.05 and a tenth of the scaled value, while scaled it stays within 5%; the Jacobian follows."""
    rng = np.random.default_rng([SEED, 905])
    for d in (8, 64, 512):
        s = (rng.choice([-1.0, 1.0], (20_000, d)) * rng.choice([-1.0, 1.0], (20_000, d))).sum(1)
        se = math.sqrt((np.mean((s - s.mean()) ** 4) - s.var() ** 2) / len(s))
        assert abs(s.var() - d) < 4 * se
    for conc in (0.1, 1.0, 10.0):
        for a in rng.dirichlet(np.full(10, conc), 50):
            b = softmax_jacobian_bound(a)
            assert b["norm"] <= b["trace"] + 1e-15 and b["trace"] <= b["bound"] + 1e-15
            assert np.allclose(np.diag(a) - np.outer(a, a), nlm.softmax_jacobian(np.log(a)), atol=1e-14)
    sc = saturation_curve()
    u, v = sc["unscaled_gap"], sc["scaled_gap"]
    assert all(x > y for x, y in zip(u, u[1:])) and u[-1] < 0.05 and u[-1] < v[-1] / 10
    assert max(v) / min(v) < 1.05
    ju, js = sc["unscaled_jac"], sc["scaled_jac"]
    assert all(x > y for x, y in zip(ju, ju[1:])) and ju[-1] < js[-1] / 3 and max(js) / min(js) < 1.15


def test_one_hop_jacobian():
    """Prop 5: the closed-form ∂o_t/∂x_k matches the complex step to 10⁻¹⁰ relative at every (t, k < t) on random
    causal layers, its spectral norm sits under the bound, and with equal scores it is exactly W_V/(t + 1) (positions
    0…t)."""
    rng = np.random.default_rng([SEED, 906])
    for _ in range(10):
        X = rng.normal(size=(7, 5))
        P = {"Wq": rng.normal(size=(5, 4)), "Wk": rng.normal(size=(5, 4)), "Wv": rng.normal(size=(5, 3))}
        for t in range(1, 7):
            for k in range(t):
                J = one_hop_jacobian(X, P, t, k)

                def out(xk, j, t=t, k=k):
                    Xc = X.astype(complex)
                    Xc[k] = xk
                    return attention(Xc @ P["Wq"], Xc @ P["Wk"], Xc @ P["Wv"], causal=True)[0][t, j]
                Jcs = np.stack([nlm.complex_step_grad(lambda xk, j=j: out(xk, j), X[k]) for j in range(3)], 1)
                assert np.allclose(J, Jcs, rtol=1e-10, atol=1e-13)
                assert np.linalg.norm(J, 2) <= one_hop_bound(X, P, t, k) + 1e-12
    P["Wq"] = np.zeros((5, 4))
    for t in range(1, 7):
        assert np.allclose(one_hop_jacobian(X, P, t, 0), P["Wv"] / (t + 1), rtol=1e-14, atol=1e-15)


def test_lag_attention():
    """Prop 6 and §4: the construction is right at every lag, its loss equals the closed form, and it falls as w grows;
    the hand-written gradient and the reach match the complex step; Topic 5's census reproduces its page; at lag 60
    attention's reach at initialization is 10⁵ times Elman's and above LSTM-1's, below LSTM-3's; across lags it falls
    like 1/(L + 1) (within a factor of 2), Elman's by 30 orders of magnitude, and attention overtakes LSTM-3 by lag
    480; trained by Topic 5's protocol at lag 60, attention is right on every fresh sequence, at lag 60 and at lag 600."""
    for L in (1, 10, 60, 600, 6000):
        r, rb = lag_construction(L, 50.0), lag_construction(L, 50.0, 3.0)
        assert r["accuracy"] == rb["accuracy"] == 1.0
        assert math.isclose(r["loss"], r["closed_form_loss"], rel_tol=1e-9, abs_tol=1e-15)
        assert math.isclose(r["margin"], 1 / (L + 1), rel_tol=1e-14) and rb["margin"] > r["margin"]
        assert math.isclose(r["o_bit0"], r["margin"], rel_tol=1e-12)
        assert lag_construction(L, 500.0)["loss"] < r["loss"] or r["loss"] == 0.0
    rng = np.random.default_rng([SEED, 907])
    X, bit = nlm.lag_data(rng, 5, 6)
    P = init_lag_attention(rng)
    _, G, reach, _ = lag_attention_loss_grad(P, X, bit)
    Xh = np.eye(4)[X.T]
    for key in P:
        fn = lambda t, key=key: lag_attention_loss({**P, key: t}, Xh, bit)  # noqa: E731
        assert np.allclose(G[key], nlm.complex_step_grad(fn, P[key]), rtol=1e-10, atol=1e-14)
        assert np.allclose(G[key], nlm.central_diff_grad(fn, P[key]), rtol=1e-6, atol=1e-9)
    norms = []
    for b in range(len(bit)):
        def at(x0, b=b):
            Z = Xh.astype(complex)
            Z[b, 0] = x0
            return lag_attention_loss(P, Z, bit)
        norms.append(np.linalg.norm(nlm.complex_step_grad(at, Xh[b, 0])))
    assert math.isclose(reach, float(np.mean(norms)), rel_tol=1e-10)
    cen5 = {k: float(np.median(v)) for k, v in nlm.long_lag_census().items()}
    for k, v in {"elman": 2.45e-10, "lstm-1": 2.42e-7, "lstm-3": 1.42e-3}.items():
        assert abs(cen5[k] / v - 1) < 0.005, (k, cen5[k])
    att = float(np.median(lag_attention_census()))
    assert att > 1e5 * cen5["elman"] and att > cen5["lstm-1"] and att < cen5["lstm-3"]
    sw = lag_sweep()
    scaled = [a * (L + 1) for a, L in zip(sw["attention"], sw["lags"])]
    assert max(scaled) / min(scaled) < 2
    assert all(x > y for x, y in zip(sw["elman"], sw["elman"][1:])) and sw["elman"][-1] / sw["elman"][0] < 1e-30
    i60, i480 = sw["lags"].index(60), sw["lags"].index(480)
    assert sw["lstm-3"][i60] > sw["attention"][i60] and sw["lstm-3"][i480] < sw["attention"][i480]
    lt = lag_attention_training()
    assert lt["accuracy"] == 1.0 and lt["accuracy_lag600"] == 1.0 and lt["test_loss"] < 0.01


def _truncate(doc: Doc, n: int) -> Doc:
    return Doc(targets=doc.targets[:n], prev=doc.prev[:n], closed=doc.closed[:n], raw=doc.raw[:n], sent=doc.sent[:n],
               kn=doc.kn[:n], spell=doc.spell[:n], oov=doc.oov[:n], chars=doc.chars, F=doc.F[:n],
               oov_key=doc.oov_key[:n], Kdot=doc.Kdot[:n, :n])


def test_kernel_cache():
    """§5: Cor 2's limits — β = 0 is the unigram document cache, β = ∞ the exact cache of the preceding token's key
    wherever that key occurred before (keys are equal for the same word, and for the word pairs Topic 6's vectors
    cannot tell apart, counted in key_classes); dev chooses an interior (β, λ); held out, the cache beats
    KN by more than 3 standard errors clustered by document, the dev-chosen bandwidth beats both limits, the gain is on
    repeated targets and grows with the distance back to the previous mention, and nearly every repeat's previous
    mention is in an earlier sentence."""
    kc = key_classes()
    assert kc["pairs"] > 0 and kc["shared_words"] < 0.5 * kc["words"]
    for d in doc_events("held")[:10]:
        p0, pinf = cache_probs(d, 0.0), cache_probs(d, math.inf)
        for t in range(1, len(d.targets)):
            assert math.isclose(p0[t], float((d.closed[:t] == d.closed[t]).mean()), rel_tol=1e-12, abs_tol=1e-15)
            ctx = d.Kdot[t, :t] >= 1 - 1e-9
            if ctx.any():
                assert math.isclose(pinf[t], float((d.closed[:t][ctx] == d.closed[t]).mean()), rel_tol=1e-12,
                                    abs_tol=1e-15)
    beta, lam = tune_cache()
    assert BETAS[0] < beta < BETAS[-1] and LAMBDAS[0] < lam < LAMBDAS[-1]
    cb = cache_benchmark()
    assert cb["cache"][0] < cb["kn"][0] and cb["gain"][0] > 3 * cb["gain"][1]
    assert cb[f"beta={beta}"]["held"] < min(cb["beta=0.0"]["held"], cb["beta=inf"]["held"])
    assert cb["repeat"][0] - cb["repeat"][1] > 0.05
    gains = [b["kn"] - b["cache"] for b in cb["bands"]]
    assert all(g > 0 for g in gains) and gains[0] < gains[1] < gains[2]
    assert all(b["cross_sentence"] > 0.85 for b in cb["bands"])
    assert cb["gain_from_cross_sentence"] > 0.9
    assert cb["cache_bpc"][0] < cb["kn_bpc"][0]


def test_learned_kernel():
    """Def 4: the pointer gradient matches the complex step to 10⁻¹⁰ relative (and central differences to 10⁻⁶) on a
    truncated training document; at A = B = 0 the pointer loss is Def 3's; the train pointer loss falls every epoch;
    the dev-chosen epoch beats Def 3 on dev; held out, the learned kernel beats Def 3 by more than 3 paired standard
    errors, and its own pointer likelihood by far more."""
    rng = np.random.default_rng([SEED, 961])
    d = _truncate(doc_events("train")[0], 40)
    beta = tune_cache()[0]
    P = {"A": rng.normal(0, 0.3, (102, 2)), "B": rng.normal(0, 0.3, (102, 2))}
    _, G = pointer_loss_grad(P, [d], beta)
    for key in P:
        fn = lambda t, key=key: pointer_loss({**P, key: t}, [d], beta)  # noqa: E731
        assert np.allclose(G[key], nlm.complex_step_grad(fn, P[key]), rtol=1e-10, atol=1e-14)
        assert np.allclose(G[key], nlm.central_diff_grad(fn, P[key]), rtol=1e-6, atol=1e-9)
    zero = {k: np.zeros_like(v) for k, v in P.items()}
    low = _strict_lower(len(d.targets))
    same = (d.closed[:, None] == d.closed[None, :]) & low
    rep = same.any(1)
    direct = -np.log((cache_weights(d, beta) * same).sum(1)[rep]).mean()
    assert math.isclose(pointer_loss(zero, [d], beta), direct, rel_tol=1e-12)
    tk = train_learned_kernel()
    h = [e["train_pointer"] for e in tk.history]
    assert all(x > y for x, y in zip(h, h[1:]))
    assert tk.history[tk.epoch - 1]["dev"] < dev_grid()[beta][1]
    lb = learned_benchmark()
    assert lb["learned_minus_fixed"][0] < -3 * lb["learned_minus_fixed"][1]
    assert lb["held_pointer_fixed"] - lb["held_pointer_learned"] > 0.25


def test_log_attention_rank():
    """Prop 7: one head's log-attention matrix has rank d_k + 1; its pattern is realizable at d_k (gap 10⁻¹⁰) and not
    at d_k − 1; a random pattern is realizable only at d_k = n − 1; and d_k = 2 realizes a random permutation at low
    temperature."""
    rng = np.random.default_rng([SEED, 908])
    n = 12
    for dk in (1, 2, 3, 5):
        Q, K = rng.normal(size=(n, dk)), rng.normal(size=(n, dk))
        tau = math.sqrt(dk)
        assert log_attention_rank(Q, K, tau) == dk + 1
        P = attention(Q, K, np.eye(n), tau)[1]
        assert best_head_fit(P, dk) < 1e-10 and best_head_fit(P, dk - 1) > 1e-3
    T = rng.dirichlet(np.ones(n), n)
    assert all(best_head_fit(T, dk) > 1e-3 for dk in range(n - 1)) and best_head_fit(T, n - 1) < 1e-10
    sigma = rng.permutation(n)
    ang = 2 * np.pi * np.arange(n) / n
    K = np.stack([np.cos(ang), np.sin(ang)], 1)
    A = attention(60 * K[sigma], K, np.eye(n), tau=1.0)[1]
    assert np.array_equal(A.argmax(1), sigma) and A[np.arange(n), sigma].min() > 0.99


def test_positive_random_features():
    """Prop 8: over 2·10⁵ features, the mean and variance of φ_w(x)φ_w(y) match e^{⟨x,y⟩} and
    e^{2⟨x,y⟩}(e^{‖x+y‖²} − 1) within 4 standard errors; positive features are positive; trigonometric features give a
    negative attention denominator on some instance, positive features never."""
    rng = np.random.default_rng([SEED, 909])
    for _ in range(5):
        x, y = rng.normal(0, 0.3, 4), rng.normal(0, 0.3, 4)
        W = rng.standard_normal((200_000, 4))
        z = np.exp(W @ x - x @ x / 2) * np.exp(W @ y - y @ y / 2)
        mean, var = rf_moments(x, y)
        assert z.min() > 0 and abs(z.mean() - mean) < 4 * math.sqrt(var / len(z))
        assert abs(z.var() - var) < 4 * math.sqrt((np.mean((z - z.mean()) ** 4) - z.var() ** 2) / len(z))
    negative = False
    for _ in range(200):
        Q, K = rng.normal(0, 0.7, (5, 3)), rng.normal(0, 0.7, (6, 3))
        W, b = rng.standard_normal((4, 3)), rng.uniform(0, 2 * np.pi, 4)
        negative |= bool(((trig_features(Q, W, b) @ trig_features(K, W, b).T).sum(1) < 0).any())
        assert ((positive_features(Q, W) @ positive_features(K, W).T).sum(1) > 0).all()
    assert negative


def test_linear_attention_recurrence():
    """Prop 9: the recurrence equals the masked quadratic form to 10⁻¹² (inclusive and strict), and with many positive
    features linear attention approaches softmax attention."""
    rng = np.random.default_rng([SEED, 910])
    Q, K, V = rng.normal(0, 0.4, (30, 3)), rng.normal(0, 0.4, (30, 3)), rng.normal(size=(30, 4))
    W = rng.standard_normal((16, 3))
    for strict in (False, True):
        pq, pk = positive_features(Q, W), positive_features(K, W)
        assert np.allclose(linear_attention_recurrent(pq, pk, V, strict), linear_attention_quadratic(pq, pk, V, strict),
                           rtol=1e-12, atol=1e-13)
    W = rng.standard_normal((20_000, 3))
    approx = linear_attention_quadratic(positive_features(Q, W), positive_features(K, W), V)
    assert np.abs(approx - attention(Q, K, V, tau=1.0, causal=True)[0]).max() < 0.05


def test_random_feature_cache():
    """Props 8–9 on held-out documents: the random-feature cache's total-variation error falls with m at every β and
    rises with β at every m — below 0.05 at β = ½ and above 0.4 at β = 8 with 1,024 features; computed as Prop 9's
    recurrence it equals the quadratic form; its held-out loss approaches the exact cache's as m grows."""
    rf = random_feature_benchmark()
    tv = rf["tv"]
    for row in tv.values():
        assert all(x > y for x, y in zip(row, row[1:]))
    for j in range(len(RF_MS)):
        col = [tv[b][j] for b in RF_BETAS]
        assert all(x < y for x, y in zip(col, col[1:]))
    assert tv[0.5][-1] < 0.05 and tv[8.0][-1] > 0.4
    assert abs(rf["nats"][-1] - rf["exact"]) < abs(rf["nats"][0] - rf["exact"])
    beta, d = tune_cache()[0], doc_events("held")[0]
    rng = np.random.default_rng([SEED, 971])
    W = rng.standard_normal((64, 101))
    state = rng.bit_generator.state
    w_quad = random_feature_weights(d, beta, W, rng)
    rng.bit_generator.state = state
    E = math.sqrt(beta) * doc_keys(d)
    phi = positive_features(E, np.hstack([W, rng.standard_normal((64, E.shape[1] - 101))]))
    onehot = np.eye(int(d.closed.max()) + 1)[d.closed]
    rec = linear_attention_recurrent(phi, phi, onehot, strict=True)[np.arange(len(d.closed)), d.closed]
    assert np.allclose(rec, cache_probs(d, beta, weights=w_quad), rtol=1e-10, atol=1e-14)


def test_finance_findings():
    """§7: on EDGAR, KN ⊕ cache beats KN with Topic 5's PPM-C spelling cache, which beats KN; the learned kernel helps
    further; lzma and Topic 5's best still code it shorter; nearly all of the cache's gain is on repeated OOV words; the
    in-domain λ under-trusts the cache on real text; each curated term's second mention costs less than half its KN
    bits; and the cache helps every filing."""
    f = finance_summary()
    assert f["cache_bpc"] < f["kn_ppm_bpc"] < f["kn_bpc"] and f["learned_bpc"] < f["cache_bpc"]
    assert f["lzma_bpc"] < f["topic5_best_bpc"] < f["cache_bpc"]
    assert f["gain_share_oov_repeat"] > 0.9
    assert f["oracle_lambda"] < tune_cache()[1] and f["oracle_bpc"] < f["cache_bpc"]
    assert len(f["curated"]) == 6 and all(c["cache_bits"][1] < c["kn_bits"][1] / 2 for c in f["curated"])
    assert all(r["cache_bpc"] < r["kn_bpc"] for r in f["per_filing"])


def test_viz_payload():
    """The lab's data: every circle case has a value per c on the grid and the trained c beside c*; the reach view has
    a value per lag; every lab document has bits per token for each variant, and its curated attention rows; the held-out
    U-curve equals the benchmark's; the file stays under 60 kB."""
    p = viz_payload()
    G = len(p["circle"]["c_grid"])
    for k, cfg in LAB_CASES.items():
        case = p["circle"]["cases"][k]
        assert len(case["risk"]) == len(case["bias2"]) == len(case["gains"]) == G and len(case["theta"]) == cfg.n
        assert case["c_learned"] == _r(train_denoiser(cfg).c)
    assert all(len(p["reach"][k]) == len(LAG_SWEEP) for k in ("attention", "elman", "lstm-1", "lstm-3"))
    for doc in p["cache"]["documents"]:
        assert all(len(v) == len(doc["tokens"]) for v in doc["bits"].values()) and doc["attn"]
    assert p["cache"]["held_by_beta"] == _r([cache_benchmark()[f"beta={b}"]["held"] for b in BETAS], 4)
    assert len(json.dumps(p, separators=(",", ":"))) < 60_000


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
    print("\nSummary (train generate_corpus(300, 0); dev generate_corpus(40, 2); held generate_corpus(100, 1); EDGAR):")
    for k, v in summary().items():
        print(f"  {k:40} {v:.4g}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else
              f"  {k:40} {v:.4f}" if isinstance(v, float) else f"  {k:40} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
