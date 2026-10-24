"""Positional encodings: sinusoids, rotations, and RoPE — the reference implementation for formalNLP's tenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. Order, and how a mask leaks it: one causal layer reads its prefix as a multiset, so it knows how far it has read
   (a BOS token's value comes back as 1/(t + 1)) but not in what order; a single NoPE head on the cue task is
   bounded by the best multiset reader.
2. Position as a set of clocks: the sinusoidal code turns d/2 clock hands, a shift is one rotation, and the inner
   product of two positions is a stationary kernel — the characteristic function of the frequencies — that rules
   distance logarithmically as d grows. Added to the content it does not make attention relative: only a
   d-dimensional commutant of score matrices sees offsets alone.
3. Relative position: ALiBi's weights are Topic 9's Nadaraya–Watson smoother with a content × exponential product
   kernel; the same inequality bounds its reach at every length and puts a horizon on what it can retrieve.
4. Rotations: RoPE's scores depend only on the offset and preserve norms; up to an orthogonal change of basis it is
   the only linear relative encoding; its "long-term decay" is a property of an envelope and of matched pairs.
5. Past the training window: single causal heads with each scheme, trained at 32 tokens and read to 256; the
   wavelength census says which RoPE planes see unseen angles; position interpolation, NTK-aware scaling, and YaRN
   re-read the trained RoPE head zero-shot.
6. Finance case study: which company was named last? Heads trained on 32-token windows of the synthetic filings,
   asked at every position of whole held-out filings.

Cross-checks: every hand-written gradient against central differences (Topic 5's checker); the vectorized filing
reader against the plain forward pass; the sine and cosine integrals against scipy.special; the real normal form
against scipy.linalg.schur; the probabilistic claims by Monte Carlo.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/positional-encodings-geometry/positional_encodings_geometry.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import linalg, special

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "attention-as-kernel-smoothing"))
import attention_as_kernel_smoothing as ak  # noqa: E402 — a module object: its test_* stay out of this harness

fc, nlm = ak.fc, ak.nlm

# ONE stream per random-instance claim and per trained head, each from default_rng([SEED, k]); k ≥ 1000 keeps this
# module's streams apart from Topics 5–9.
SEED = ak.SEED
Params = dict[str, np.ndarray]
BASE = 10_000.0


# --------------------------------------------------------------------------------------------
# 2 (first, for §1's head). Position as a set of clocks.
# --------------------------------------------------------------------------------------------


def frequencies(d: int, base: float = BASE) -> np.ndarray:
    """ω_j = b^{−2j/d}, j = 0, …, d/2 − 1."""
    return base ** (-2 * np.arange(d // 2) / d)


def sinusoidal(positions, freqs: np.ndarray) -> np.ndarray:
    """Def 2: p(m) = (cos mω_0, sin mω_0, cos mω_1, …), one row per position."""
    ang = np.multiply.outer(np.asarray(positions, float), freqs)
    P = np.empty((*ang.shape[:-1], 2 * len(freqs)))
    P[..., 0::2], P[..., 1::2] = np.cos(ang), np.sin(ang)
    return P


def rotation(k: float, freqs: np.ndarray) -> np.ndarray:
    """R_ω(k) = ⊕_j R(kω_j), the d × d block-diagonal rotation by k steps."""
    R = np.zeros((2 * len(freqs), 2 * len(freqs)))
    c, s = np.cos(k * freqs), np.sin(k * freqs)
    R[0::2, 0::2][np.diag_indices(len(freqs))] = c
    R[1::2, 1::2][np.diag_indices(len(freqs))] = c
    R[1::2, 0::2][np.diag_indices(len(freqs))] = s
    R[0::2, 1::2][np.diag_indices(len(freqs))] = -s
    return R


def rotate(X: np.ndarray, positions, freqs: np.ndarray) -> np.ndarray:
    """Each row x of X (shape (…, n, d)) turned to R_ω(position) x — RoPE's map, without forming matrices.
    positions has shape (n,), or (…, n) to give every row its own position."""
    ang = np.multiply.outer(np.asarray(positions, float), freqs)
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = X[..., 0::2], X[..., 1::2]
    out = np.empty(np.broadcast_shapes(X.shape, (*ang.shape[:-1], X.shape[-1])))
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def kernel(delta, freqs: np.ndarray) -> np.ndarray:
    """Thm 1 (i): κ_d(Δ) = Σ_j cos(Δω_j) = ⟨p(m), p(m + Δ)⟩."""
    return np.cos(np.multiply.outer(np.asarray(delta, float), freqs)).sum(-1)


def kernel_limit(delta, base: float = BASE) -> np.ndarray:
    """Thm 1 (ii): lim (2/d)κ_d(Δ) = [Ci(|Δ|) − Ci(|Δ|/b)]/ln b, and 1 at Δ = 0."""
    a = np.abs(np.asarray(delta, float))
    safe = np.where(a > 0, a, 1.0)
    out = (special.sici(safe)[1] - special.sici(safe / base)[1]) / math.log(base)
    return np.where(a > 0, out, 1.0)


def log_ruler(delta, base: float = BASE) -> np.ndarray:
    """Thm 1 (ii)'s asymptote for 1 ≪ |Δ| ≪ b: 1 − (γ + ln|Δ|)/ln b."""
    return 1 - (np.euler_gamma + np.log(np.abs(np.asarray(delta, float)))) / math.log(base)


def commutant_projection(M: np.ndarray) -> np.ndarray:
    """Thm 2 (c): the Frobenius-orthogonal projection Π onto ⊕_j span{I, J} — each diagonal 2 × 2 block B becomes
    (tr B/2) I + ((B₂₁ − B₁₂)/2) J, and every off-diagonal block becomes 0."""
    out = np.zeros_like(M)
    for j in range(M.shape[0] // 2):
        B = M[2 * j:2 * j + 2, 2 * j:2 * j + 2]
        a, b = (B[0, 0] + B[1, 1]) / 2, (B[1, 0] - B[0, 1]) / 2
        out[2 * j:2 * j + 2, 2 * j:2 * j + 2] = [[a, -b], [b, a]]
    return out


def relativity_defect(M: np.ndarray) -> float:
    """δ(M) = ‖M − Π(M)‖_F / ‖M‖_F: 0 exactly when the position–position score sees only offsets (Thm 2)."""
    return float(np.linalg.norm(M - commutant_projection(M)) / np.linalg.norm(M))


def window_conditions(freqs: np.ndarray, deltas) -> np.ndarray:
    """The coefficient matrix of Thm 2 (a′)'s window constraints, p(m + 1) ⊗ p(n + 1) − p(m) ⊗ p(n) for m, n in a
    window of d positions, one row per pair — its null space is the set of relative-only M."""
    P = sinusoidal(np.arange(len(freqs) * 2 + 1), freqs)
    rows = [np.kron(P[m + 1], P[n + 1]) - np.kron(P[m], P[n]) for m in deltas for n in deltas]
    return np.array(rows)


def window_condition(n: int, freqs: np.ndarray) -> float:
    """The 2-norm condition number of the n × d sinusoid matrix over positions 0, …, n − 1: Thm 2 (a′) is exact in
    exact arithmetic, but its constraint on the slow planes is weak when the window is short against them."""
    return float(np.linalg.cond(sinusoidal(np.arange(n), freqs)))


# --------------------------------------------------------------------------------------------
# 3. Relative position: biases and the ALiBi kernel.
# --------------------------------------------------------------------------------------------


def alibi_slopes(H: int) -> np.ndarray:
    """Press, Smith, and Lewis's fixed slopes s_h = 2^{−8h/H}, h = 1, …, H."""
    return 2.0 ** (-8 * np.arange(1, H + 1) / H)


def alibi_bias(n: int, slope: float) -> np.ndarray:
    """Def 3: −s(t − j) on j ≤ t, and −∞ above the diagonal (the causal mask)."""
    t, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return np.where(j <= t, -slope * (t - j), -np.inf)


def t5_bucket(offset: int, num_buckets: int = 32, max_distance: int = 128) -> int:
    """Def 3 (illustration): T5's causal relative-position bucket — exact below num_buckets/2, then log-spaced up to
    max_distance, and one shared bucket beyond (Raffel et al. 2020's relative_position_bucket, unidirectional)."""
    exact = num_buckets // 2
    if offset < exact:
        return int(offset)
    val = exact + int(math.log(offset / exact) / math.log(max_distance / exact) * (num_buckets - exact))
    return min(val, num_buckets - 1)


def alibi_tail_bound(W: float, s: float, D: int) -> float:
    """Thm 3 (ii): with content scores in an interval of width W, the weight on offsets ≥ D is at most
    e^{W} e^{−sD}/(1 − e^{−s}), at every length."""
    return math.exp(W - s * D) / (1 - math.exp(-s))


def alibi_horizon(W: float, s: float) -> float:
    """Thm 3 (iii): a key farther back than W/s weighs less than the offset-0 key, whatever its content."""
    return W / s


# --------------------------------------------------------------------------------------------
# 4. Rotations: RoPE.
# --------------------------------------------------------------------------------------------


def offset_score(q: np.ndarray, k: np.ndarray, deltas, freqs: np.ndarray) -> np.ndarray:
    """Prop 3 (iii): s(Δ) = ⟨q, R_θ(Δ) k⟩ = ⟨R_θ(m) q, R_θ(m + Δ) k⟩ for any m."""
    deltas = np.asarray(deltas, float)
    return rotate(np.broadcast_to(k, (len(deltas), len(k))), deltas, freqs) @ q


def trig_form(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prop 3 (iii): plane amplitudes ρ_j = ‖q_j‖‖k_j‖ and phases φ_j = ∠k_j − ∠q_j, so s(Δ) = Σ_j ρ_j cos(Δθ_j + φ_j)."""
    qa, ka = q[0::2] + 1j * q[1::2], k[0::2] + 1j * k[1::2]
    return np.abs(qa) * np.abs(ka), np.angle(ka) - np.angle(qa)


def orthogonal_family(U: np.ndarray, ms) -> np.ndarray:
    """Thm 4 (b): L_m = U^m, with U^{−m} = (Uᵀ)^m."""
    return np.array([np.linalg.matrix_power(U if m >= 0 else U.T, abs(m)) for m in ms])


def real_normal_form(U: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Thm 4 (c): an orthogonal U as Q (⊕_i R(θ_i) ⊕ I_p ⊕ −I_r) Qᵀ with θ_i ∈ (0, π), from scipy's real Schur form
    (block-diagonal for a normal matrix). Returns (Q, angles, p, r); planes first, then the +1 and −1 axes."""
    T, Z = linalg.schur(U, output="real")
    planes, plus, minus, i, n = [], [], [], 0, U.shape[0]
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > tol:
            theta = math.atan2(T[i + 1, i], T[i, i])
            cols = [i, i + 1] if theta > 0 else [i + 1, i]  # a swap conjugates R(θ) to R(−θ)
            planes.append((abs(theta), cols))
            i += 2
        else:
            (plus if T[i, i] > 0 else minus).append(i)
            i += 1
    order = [c for _, cols in planes for c in cols] + plus + minus
    return Z[:, order], np.array([t for t, _ in planes]), len(plus), len(minus)


def normal_form_power(Q: np.ndarray, angles: np.ndarray, p: int, r: int, m: int) -> np.ndarray:
    """Q (⊕_i R(mθ_i) ⊕ I_p ⊕ (−1)^m I_r) Qᵀ."""
    blocks = [np.array([[math.cos(m * t), -math.sin(m * t)], [math.sin(m * t), math.cos(m * t)]]) for t in angles]
    D = linalg.block_diag(*blocks, np.eye(p), (-1) ** m * np.eye(r)) if (len(blocks) or p or r) else np.zeros((0, 0))
    return Q @ D @ Q.T


def abel_envelope(h: np.ndarray, delta: float, freqs: np.ndarray) -> dict[str, float]:
    """Prop 4 (i): |Σ_j h_j e^{iΔθ_j}| ≤ (max_j |h_{j+1} − h_j|) Σ_{j=1}^{J} |S_j|, S_j = Σ_{l<j} e^{iΔθ_l}, h_J = 0.
    Also returns the envelope (1/J)Σ_j |S_j|."""
    e = np.exp(1j * delta * freqs)
    S = np.cumsum(e)  # S_1, …, S_J
    dh = np.abs(np.diff(np.append(h, 0)))
    return {"value": float(abs((h * e).sum())), "bound": float(dh.max() * np.abs(S).sum()),
            "envelope": float(np.abs(S).mean())}


# --------------------------------------------------------------------------------------------
# 5. Past the training window: the census and the rescalings.
# --------------------------------------------------------------------------------------------


def wavelength_census(L: int, d: int, base: float = BASE) -> dict:
    """Thm 5 (i): plane j turns through a whole circle over training offsets 0, …, L − 1 iff (L − 1)θ_j ≥ 2π; the
    others see only the arc [0, (L − 1)θ_j]."""
    th = frequencies(d, base)
    lam = 2 * np.pi / th
    complete = (L - 1) * th >= 2 * np.pi
    return {"freqs": th, "wavelengths": lam, "complete": complete, "n_complete": int(complete.sum()),
            "arc": np.minimum((L - 1) * th, 2 * np.pi)}


def pi_scale(n: int, L: int) -> float:
    """Def 5, PI (Chen et al. 2023): positions m ↦ m/κ with κ = n/L when n > L."""
    return min(1.0, L / n)


def ntk_frequencies(d: int, kappa: float, base: float = BASE) -> np.ndarray:
    """Def 5, NTK-aware (bloc97 2023): the base b ↦ bκ^{d/(d−2)}."""
    return frequencies(d, base * kappa ** (d / (d - 2)))


def yarn_frequencies(d: int, kappa: float, L: int, alpha: float = 1.0, beta: float = 32.0,
                     base: float = BASE) -> tuple[np.ndarray, float]:
    """Def 5, YaRN (Peng et al. 2024): with r_j = L/λ_j and γ the ramp from 0 at r ≤ α to 1 at r ≥ β, θ′_j =
    (1 − γ)θ_j/κ + γθ_j; logits are scaled by (0.1 ln κ + 1)². Returns (θ′, logit scale)."""
    th = frequencies(d, base)
    gamma = np.clip((L * th / (2 * np.pi) - alpha) / (beta - alpha), 0, 1)
    return (1 - gamma) * th / kappa + gamma * th, (0.1 * math.log(kappa) + 1) ** 2 if kappa > 1 else 1.0


def interpolation_bound(h: np.ndarray, freqs: np.ndarray, d1: float, d2: float, delta: float) -> float:
    """Thm 5 (iv): for a(Δ) = Re Σ_j h_j e^{iΔθ_j}, |a(Δ) − linear interpolation| ≤ (Δ − Δ₁)(Δ₂ − Δ)/2 · Σ_j |h_j|θ_j²."""
    return (delta - d1) * (d2 - delta) / 2 * float((np.abs(h) * freqs**2).sum())


def chen_constant(d: int, base: float = BASE) -> float:
    """d/(4 ln b), the constant in Chen et al.'s Theorem 2.1. Their eq. (14) bounds 1/(1 − b^{−4/d}) above by it using
    c^x ≤ 1 + x ln c for x < 0, but convexity gives c^x ≥ 1 + x ln c: d/(4 ln b) is a lower bound on Σ_j θ_j², and
    their stated bound fails at small d (test_interpolation_bound)."""
    return d / (4 * math.log(base))


def chen_stated_bound(h: np.ndarray, d: int, d1: float, d2: float, delta: float, base: float = BASE) -> float:
    """Chen et al.'s Theorem 2.1 as stated: d·max_j|h_j|·(Δ − Δ₁)(Δ₂ − Δ)/(8 ln b)."""
    return d * float(np.abs(h).max()) * (delta - d1) * (d2 - delta) / (8 * math.log(base))


# --------------------------------------------------------------------------------------------
# 1 and 5. The cue task and one causal head per scheme.
# --------------------------------------------------------------------------------------------

D_TOUR, L_TRAIN, N_SYMBOLS, CUE_RATE = 16, 32, 8, 0.1
F_IN = N_SYMBOLS + 2  # one-hot symbol, cue bit, query bit
CUE, QUERY = N_SYMBOLS, N_SYMBOLS + 1
TEST_LENGTHS = (32, 64, 128, 256)
SCHEMES = ("nope", "sinusoidal", "learned", "rope", "alibi")
RESCUES = ("pi", "ntk", "yarn")
ALIBI_SLOPE = 0.5  # ALiBi's first slope at H = 8
TOUR_SEEDS = (0, 1, 2)
TOUR_FREQS = frequencies(D_TOUR)
TEST_N = 2000


def cue_data(rng: np.random.Generator, batch: int, n: int, rate: float = CUE_RATE):
    """The cue task: n − 1 tokens, each a uniform symbol carrying a cue bit with probability `rate` (at least one cue
    is forced), then a query token. The target is the symbol on the most recent cue. Returns (X, y, gap), X one-hot
    (batch, n, F_IN) and gap = the query's offset to that cue."""
    sym = rng.integers(N_SYMBOLS, size=(batch, n))
    cue = rng.random((batch, n)) < rate
    cue[:, -1] = False
    none = ~cue.any(1)
    cue[none, rng.integers(n - 1, size=int(none.sum()))] = True
    X = np.zeros((batch, n, F_IN))
    X[np.arange(batch)[:, None], np.arange(n)[None, :], sym] = 1
    X[..., CUE] = cue
    X[:, -1] = 0
    X[:, -1, QUERY] = 1
    last = (n - 2) - np.argmax(cue[:, :-1][:, ::-1], axis=1)
    return X, sym[np.arange(batch), last], (n - 1) - last


def bag_bayes_accuracy(n: int, rate: float, trials: int, rng: np.random.Generator) -> tuple[float, float]:
    """Prop 1 (ii): a reader of the multiset of tokens sees each ordering of it as equally likely (the tokens are
    exchangeable), so its best guess is the most frequent cue symbol and its accuracy is E[max_s c_s / Σ_s c_s].
    Returns (mean, standard error)."""
    X, _, _ = cue_data(rng, trials, n, rate)
    counts = (X[:, :, :N_SYMBOLS] * X[:, :, CUE:CUE + 1]).sum(1)
    acc = counts.max(1) / counts.sum(1)
    return float(acc.mean()), float(acc.std(ddof=1) / math.sqrt(trials))


def causal_counter(t_max: int) -> np.ndarray:
    """Prop 1 (i): one causal layer with constant keys (all scores equal), a BOS value e at position 0, and zero
    values elsewhere outputs o_t = e/(t + 1) — through Topic 9's attention()."""
    n = t_max + 1
    Q, K = np.zeros((n, 1)), np.zeros((n, 1))
    V = np.zeros((n, 1))
    V[0] = 1.0
    O, _ = ak.attention(Q, K, V, causal=True)
    return O[:, 0]


def init_head(rng: np.random.Generator, scheme: str, n: int = L_TRAIN) -> Params:
    """Scales follow Topic 9's lag head: input map N(0, 1/2), the other maps N(0, 1/√d)."""
    d = D_TOUR
    P = {"E": rng.normal(0, 1 / 2, (F_IN, d)), "Wq": rng.normal(0, 1 / math.sqrt(d), (d, d)),
         "Wk": rng.normal(0, 1 / math.sqrt(d), (d, d)), "Wv": rng.normal(0, 1 / math.sqrt(d), (d, d)),
         "Wo": rng.normal(0, 1 / math.sqrt(d), (d, N_SYMBOLS)), "bo": np.zeros(N_SYMBOLS)}
    if scheme == "learned":
        P["Pos"] = rng.normal(0, 1 / 2, (n, d))
    return P


def _head_forward(P: Params, X: np.ndarray, scheme: str, positions=None, freqs=None, slope: float = ALIBI_SLOPE,
                  temp: float = 1.0):
    """One causal head read at the last position (the query). Returns the intermediates and the log-probabilities."""
    n = X.shape[1]
    positions = np.arange(n, dtype=float) if positions is None else np.asarray(positions, float)
    freqs = TOUR_FREQS if freqs is None else freqs
    h = X @ P["E"]
    if scheme == "sinusoidal":
        h = h + sinusoidal(positions, freqs)
    elif scheme == "learned":
        h = h + P["Pos"][:n]
    q0, K0, V = h[:, -1] @ P["Wq"], h @ P["Wk"], h @ P["Wv"]
    if scheme == "rope":
        q, K = rotate(q0[:, None, :], positions[-1:], freqs)[:, 0], rotate(K0, positions, freqs)
    else:
        q, K = q0, K0
    tau = math.sqrt(D_TOUR)
    s = (K @ q[..., None])[..., 0] / tau
    if scheme == "alibi":
        s = s - slope * (positions[-1] - positions)
    a = ak._softmax(temp * s)
    o = (a[..., None] * V).sum(1)
    lp = nlm.log_softmax(o @ P["Wo"] + P["bo"])
    return h, q, K, V, a, o, lp, positions, freqs, tau


def head_loss_grad(P: Params, X: np.ndarray, y: np.ndarray, scheme: str, **kw) -> tuple[float, Params]:
    """The mean cross-entropy and its hand-written gradient (checked by test_gradients), through RoPE's rotations,
    the additive codes, and ALiBi's (parameter-free) bias."""
    h, q, K, V, a, o, lp, positions, freqs, tau = _head_forward(P, X, scheme, **kw)
    temp = kw.get("temp", 1.0)
    B = len(y)
    g = np.exp(lp)
    g[np.arange(B), y] -= 1
    g /= B
    G = {"Wo": o.T @ g, "bo": g.sum(0)}
    g_o = g @ P["Wo"].T
    dV = a[..., None] * g_o[:, None, :]
    da = (V * g_o[:, None, :]).sum(-1)
    ds = temp * a * (da - (a * da).sum(1, keepdims=True))
    dK = ds[..., None] * q[:, None, :] / tau
    dq = (ds[..., None] * K).sum(1) / tau
    if scheme == "rope":  # the rotation is orthogonal: pull the gradient back by its transpose
        dK, dq = rotate(dK, -positions, freqs), rotate(dq[:, None, :], -positions[-1:], freqs)[:, 0]
    G["Wq"] = h[:, -1].T @ dq
    G["Wk"] = np.einsum("bti,btj->ij", h, dK)
    G["Wv"] = np.einsum("bti,btj->ij", h, dV)
    dh = dK @ P["Wk"].T + dV @ P["Wv"].T
    dh[:, -1] += dq @ P["Wq"].T
    G["E"] = np.einsum("btf,btd->fd", X, dh)
    if scheme == "learned":
        G["Pos"] = np.zeros_like(P["Pos"])
        G["Pos"][:X.shape[1]] = dh.sum(0)
    return float(-lp[np.arange(B), y].mean()), G


def rescue_kwargs(rescue: str | None, n: int, L: int = L_TRAIN) -> dict:
    """The keyword arguments that read a trained RoPE head at length n under a Def 5 map (none below L)."""
    kappa = max(n / L, 1.0)
    if rescue is None or kappa == 1.0:
        return {}
    if rescue == "pi":
        return {"positions": np.arange(n) * pi_scale(n, L)}
    if rescue == "ntk":
        return {"freqs": ntk_frequencies(D_TOUR, kappa)}
    freqs, temp = yarn_frequencies(D_TOUR, kappa, L)
    return {"freqs": freqs, "temp": temp}


def head_accuracy(P: Params, X: np.ndarray, y: np.ndarray, scheme: str, rescue: str | None = None) -> float:
    """Accuracy of the argmax readout; nan for a learned table read past its length."""
    if scheme == "learned" and X.shape[1] > P["Pos"].shape[0]:
        return math.nan
    lp = _head_forward(P, X, scheme, **rescue_kwargs(rescue, X.shape[1]))[6]
    return float((lp.argmax(1) == y).mean())


# --------------------------------------------------------------------------------------------
# 6 (the token streams, needed by the filing-trained heads). Which company was named last?
# --------------------------------------------------------------------------------------------

GAP_BANDS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9))
FILING_SCHEMES = ("nope", "sinusoidal", "rope", "alibi")


def _segments(doc) -> list[tuple[str, object]]:
    """The document as tokens: each entity mention whole (its surface text and the entity), every other word lowercased
    (with None)."""
    out, i = [], 0
    for e in sorted(doc.entities, key=lambda e: e.start):
        out += [(w, None) for w in re.findall(r"[a-z]+", doc.text[i:e.start].lower())]
        out.append((doc.text[e.start:e.end], e))
        i = e.end
    return out + [(w, None) for w in re.findall(r"[a-z]+", doc.text[i:].lower())]


def org_tokens(doc) -> list[tuple[str, int | None]]:
    """Each entity mention collapses to one token; an ORG mention is a cue, labeled by its company's order of first
    appearance in the document (mod 8); every other token is (word, None)."""
    toks, ids = [], {}
    for surface, e in _segments(doc):
        if e is None:
            toks.append((surface, None))
        elif e.type == "ORG":
            ids.setdefault(e.kb_id, len(ids))
            toks.append(("<org>", ids[e.kb_id] % N_SYMBOLS))
        else:
            toks.append((f"<{e.type.lower()}>", None))
    return toks


@dataclass(frozen=True)
class Filing:
    X: np.ndarray      # (n, F_IN): ORG mentions carry their label and the cue bit; other words a CRC-32 symbol
    last: np.ndarray   # index of the latest ORG mention at or before each position (−1 if none)
    label: np.ndarray  # its label (−1 if none)
    tokens: tuple


def _filing(doc) -> Filing:
    toks = org_tokens(doc)
    X = np.zeros((len(toks), F_IN))
    last, label, cur = np.full(len(toks), -1), np.full(len(toks), -1), -1
    for j, (w, lab) in enumerate(toks):
        if lab is None:
            X[j, zlib.crc32(w.encode()) % N_SYMBOLS] = 1
        else:
            X[j, lab], X[j, CUE], cur = 1, 1, j
        last[j] = cur
        label[j] = toks[cur][1] if cur >= 0 else -1
    return Filing(X, last, label, tuple(toks))


@lru_cache(maxsize=None)
def documents(split: str) -> tuple:
    """train: generate_corpus(300, 0) (Topic 9's training split); held: generate_corpus(100, 1)."""
    return tuple(fc.generate_corpus(300, 0) if split == "train" else fc.generate_corpus(100, 1))


@lru_cache(maxsize=None)
def filings(split: str) -> tuple[Filing, ...]:
    return tuple(_filing(d) for d in documents(split))


@lru_cache(maxsize=1)
def _window_index() -> tuple[np.ndarray, np.ndarray]:
    """Every (document, t) whose 31-token window ending at t holds an ORG mention: the training queries."""
    docs, ts = [], []
    for i, f in enumerate(filings("train")):
        t = np.arange(L_TRAIN - 2, len(f.label))
        ok = (f.last[t] >= 0) & (t - f.last[t] <= L_TRAIN - 2)
        docs += [i] * int(ok.sum())
        ts += list(t[ok])
    return np.array(docs), np.array(ts)


def filing_windows(rng: np.random.Generator, batch: int):
    """A batch of training queries: 31 filing tokens, then the query token. Returns (X, y, gap) like cue_data."""
    di, ti = _window_index()
    pick = rng.integers(len(di), size=batch)
    F = filings("train")
    X = np.zeros((batch, L_TRAIN, F_IN))
    y, gap = np.empty(batch, int), np.empty(batch, int)
    for b, (i, t) in enumerate(zip(di[pick], ti[pick])):
        f = F[i]
        X[b, :-1] = f.X[t - (L_TRAIN - 2):t + 1]
        y[b], gap[b] = f.label[t], t + 1 - f.last[t]
    X[:, -1, QUERY] = 1
    return X, y, gap


# --------------------------------------------------------------------------------------------
# 5. Training.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadConfig:
    scheme: str
    seed: int
    source: str = "synthetic"  # or "filings": 32-token windows of the training filings
    steps: int = 1500
    batch: int = 64
    lr: float = 0.01


@dataclass(frozen=True)
class TrainedHead:
    P: dict
    final_loss: float


@lru_cache(maxsize=None)
def train_head(cfg: HeadConfig) -> TrainedHead:
    """Adam (Topic 5's constants) on fresh batches at L = 32; one stream per (source, scheme, seed)."""
    k = 1100 + (100 if cfg.source == "filings" else 0) + 10 * cfg.seed + SCHEMES.index(cfg.scheme)
    rng = np.random.default_rng([SEED, k])
    P = init_head(rng, cfg.scheme)
    m = {key: np.zeros_like(v) for key, v in P.items()}
    s = {key: np.zeros_like(v) for key, v in P.items()}
    losses = []
    for step in range(1, cfg.steps + 1):
        X, y, _ = cue_data(rng, cfg.batch, L_TRAIN) if cfg.source == "synthetic" else filing_windows(rng, cfg.batch)
        loss, G = head_loss_grad(P, X, y, cfg.scheme)
        losses.append(loss)
        c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
        for key in P:
            m[key] = nlm.ADAM_B1 * m[key] + (1 - nlm.ADAM_B1) * G[key]
            s[key] = nlm.ADAM_B2 * s[key] + (1 - nlm.ADAM_B2) * G[key] ** 2
            P[key] -= cfg.lr * (m[key] / c1) / (np.sqrt(s[key] / c2) + nlm.ADAM_EPS)
    return TrainedHead(P, float(np.mean(losses[-100:])))


@lru_cache(maxsize=None)
def cue_test_set(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TEST_N fresh cue sequences of length n, shared by every scheme and seed (paired comparisons)."""
    return cue_data(np.random.default_rng([SEED, 1300, n]), TEST_N, n)


@lru_cache(maxsize=1)
def tournament() -> dict:
    """Accuracy of every scheme at every test length, per seed, and of the RoPE head under each rescue."""
    out: dict = {"schemes": {}, "rescues": {}}
    for scheme in SCHEMES:
        out["schemes"][scheme] = {n: [head_accuracy(train_head(HeadConfig(scheme, s)).P, *cue_test_set(n)[:2], scheme)
                                      for s in TOUR_SEEDS] for n in TEST_LENGTHS}
    for rescue in RESCUES:
        out["rescues"][rescue] = {n: [head_accuracy(train_head(HeadConfig("rope", s)).P, *cue_test_set(n)[:2], "rope",
                                                    rescue) for s in TOUR_SEEDS] for n in TEST_LENGTHS}
    return out


def _token_vector(P: Params, sym: int | None, cue: bool, query: bool = False) -> np.ndarray:
    x = np.zeros(F_IN)
    if query:
        x[QUERY] = 1
    else:
        x[sym] = 1
        x[CUE] = cue
    return x @ P["E"]


def content_scores(P: Params) -> np.ndarray:
    """The query's content scores ⟨q, k⟩/τ against every token type (8 symbols × cue bit, and the query itself) —
    position-free for NoPE, RoPE at offset 0, and ALiBi."""
    q = _token_vector(P, None, False, True) @ P["Wq"]
    types = [_token_vector(P, s, c) for c in (False, True) for s in range(N_SYMBOLS)]
    types.append(_token_vector(P, None, False, True))
    return np.array(types) @ P["Wk"] @ q / math.sqrt(D_TOUR)


def content_width(P: Params) -> float:
    """W = max − min of content_scores: Thm 3's interval width for a trained head."""
    c = content_scores(P)
    return float(c.max() - c.min())


def cue_query_key(P: Params) -> tuple[np.ndarray, np.ndarray]:
    """The query token's q and the symbol-averaged cue key k, before any position enters."""
    k_in = np.mean([_token_vector(P, s, True) for s in range(N_SYMBOLS)], axis=0)
    return _token_vector(P, None, False, True) @ P["Wq"], k_in @ P["Wk"]


def learned_offset_score(P: Params, scheme: str, offsets, length: int = TEST_LENGTHS[-1]) -> np.ndarray:
    """The score the query assigns a cue key at offset r (symbol-averaged): s(r) = ⟨q, R(−r)k_cue⟩/τ for RoPE, the
    content score − s·r for ALiBi; for the absolute codes, the query at the end of a `length`-token sequence."""
    offsets = np.asarray(offsets)
    k_in = np.mean([_token_vector(P, s, True) for s in range(N_SYMBOLS)], axis=0)
    q_in = _token_vector(P, None, False, True)
    tau = math.sqrt(D_TOUR)
    if scheme in ("sinusoidal", "learned"):
        t = length - 1 if scheme == "sinusoidal" else L_TRAIN - 1
        pos = (lambda m: sinusoidal(m, TOUR_FREQS)) if scheme == "sinusoidal" else (lambda m: P["Pos"][m])
        q = (q_in + pos(t)) @ P["Wq"]
        return (k_in + pos(t - offsets)) @ P["Wk"] @ q / tau
    q, k = cue_query_key(P)
    if scheme == "rope":
        return offset_score(q, k, -offsets, TOUR_FREQS) / tau
    base = float(q @ k / tau)
    return base - ALIBI_SLOPE * offsets if scheme == "alibi" else np.full(len(offsets), base)


# --------------------------------------------------------------------------------------------
# 6. Reading whole filings.
# --------------------------------------------------------------------------------------------


def _row_freqs(rescue: str | None, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per query row: the frequencies, the position scale, and the logit scale of a Def 5 map at that row's length."""
    th, sc, tp = [], [], []
    for n in lengths:
        kw = rescue_kwargs(rescue, int(n))
        th.append(kw.get("freqs", TOUR_FREQS))
        sc.append(pi_scale(int(n), L_TRAIN) if rescue == "pi" else 1.0)
        tp.append(kw.get("temp", 1.0))
    return np.array(th), np.array(sc), np.array(tp)


def filing_logprobs(P: Params, scheme: str, f: Filing, rescue: str | None = None) -> np.ndarray:
    """The head's log-probabilities at every position t of a filing, as if the query token followed it (length
    t + 2): every query of the document in one pass. For RoPE the score in plane j is (a·b) cos φ + (a₁b₀ − a₀b₁)
    sin φ with φ = (j − t − 1)·scale·θ_j, so each row can carry its own rescaling."""
    n, tau = len(f.label), math.sqrt(D_TOUR)
    t = np.arange(n)
    h = f.X @ P["E"]
    hq = np.tile(_token_vector(P, None, False, True), (n, 1))
    if scheme == "sinusoidal":
        h, hq = h + sinusoidal(t, TOUR_FREQS), hq + sinusoidal(t + 1, TOUR_FREQS)
    Q, K, V, Ks, Vs = hq @ P["Wq"], h @ P["Wk"], h @ P["Wv"], hq @ P["Wk"], hq @ P["Wv"]
    temp = np.ones(n)
    if scheme == "rope":
        th, sc, temp = _row_freqs(rescue, t + 2)
        qa, qb = Q[0, 0::2], Q[0, 1::2]  # the query vector is the same at every row before rotation
        dot, cross = K[:, 0::2] * qa + K[:, 1::2] * qb, qb * K[:, 0::2] - qa * K[:, 1::2]
        phi = (t[None, :] - t[:, None] - 1)[..., None] * (sc[:, None, None] * th[:, None, :])
        S = (dot[None] * np.cos(phi) + cross[None] * np.sin(phi)).sum(-1)
    else:
        S = Q @ K.T
    Z = np.column_stack([S, (Q * Ks).sum(1)]) / tau  # the bias is added after the 1/τ scale, as in _head_forward
    if scheme == "alibi":
        Z[:, :n] -= ALIBI_SLOPE * (t[:, None] + 1 - t[None, :])
    Z[:, :n] = np.where(t[None, :] <= t[:, None], Z[:, :n], -np.inf)
    A = ak._softmax(temp[:, None] * Z)
    O = A[:, :n] @ V + A[:, n:] * Vs
    return nlm.log_softmax(O @ P["Wo"] + P["bo"])


def read_filing(P: Params, scheme: str, f: Filing, rescue: str | None = None) -> np.ndarray:
    """The head's answer at every position of a filing."""
    return np.argmax(filing_logprobs(P, scheme, f, rescue), axis=1)


@lru_cache(maxsize=1)
def filing_benchmark() -> dict:
    """Every filing-trained head (and the RoPE head under each rescue) on every held-out position with an earlier
    ORG mention: accuracy overall and by gap band, per seed."""
    held = filings("held")
    gaps = np.concatenate([np.arange(len(f.label)) + 1 - f.last for f in held])
    valid = np.concatenate([f.last >= 0 for f in held])
    truth = np.concatenate([f.label for f in held])
    out: dict = {"n_queries": int(valid.sum()), "gaps": gaps[valid], "variants": {}}
    variants = [(s, None) for s in FILING_SCHEMES] + [("rope", r) for r in RESCUES]
    for scheme, rescue in variants:
        name = scheme if rescue is None else f"rope+{rescue}"
        rows = []
        for seed in TOUR_SEEDS:
            P = train_head(HeadConfig(scheme, seed, "filings")).P
            pred = np.concatenate([read_filing(P, scheme, f, rescue) for f in held])
            ok = (pred == truth)[valid]
            rows.append({"all": float(ok.mean()),
                         "bands": [float(ok[(gaps[valid] >= a) & (gaps[valid] <= b)].mean()) for a, b in GAP_BANDS]})
        out["variants"][name] = rows
    out["band_counts"] = [int(((gaps[valid] >= a) & (gaps[valid] <= b)).sum()) for a, b in GAP_BANDS]
    return out


@lru_cache(maxsize=1)
def alibi_forecast() -> list[dict]:
    """Thm 3 (iii) as a forecast, for every ALiBi head trained on either source: its content width W, its horizon
    W/s, and its accuracy on the held-out filings inside half the horizon and beyond it."""
    held, out = filings("held"), []
    valid = np.concatenate([f.last >= 0 for f in held])
    truth = np.concatenate([f.label for f in held])[valid]
    gaps = filing_benchmark()["gaps"]
    for source in ("filings", "synthetic"):
        for seed in TOUR_SEEDS:
            P = train_head(HeadConfig("alibi", seed, source)).P
            W = content_width(P)
            H = alibi_horizon(W, ALIBI_SLOPE)
            ok = np.concatenate([read_filing(P, "alibi", f) for f in held])[valid] == truth
            out.append({"source": source, "seed": seed, "W": W, "horizon": H,
                        "within": float(ok[gaps <= H / 2].mean()), "n_within": int((gaps <= H / 2).sum()),
                        "beyond": float(ok[gaps > H].mean()), "n_beyond": int((gaps > H).sum())})
    return out


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX's JSON in Phase B).
# --------------------------------------------------------------------------------------------

LAB_BASES = (100.0, BASE, 1e6)
LAB_DIMS = (16, 64)
LAB_KERNEL_DIMS = (16, 64, 512)
LAB_DELTA_MAX = 63  # the clock sliders run over positions 0…63
LAB_KERNEL_GRID = tuple(int(x) for x in np.unique(np.round(np.geomspace(1, 4096, 40))))
LAB_LOCK_DELTAS = (1, 4, 16)
LAB_OFFSETS = 256
LAB_KAPPAS = (2, 4, 8)
LAB_FILING_MAX = 220  # tokens: the lab's token strip


def _r(x, nd: int = 3):
    return ak._r(x, nd)


@lru_cache(maxsize=1)
def lab_filing() -> int:
    """The held-out filing (≤ LAB_FILING_MAX tokens) with the most queries that the seed-0 filing-trained ALiBi head
    answers wrong and RoPE + PI answers right — the case study's contrast on one document."""
    Pa = train_head(HeadConfig("alibi", 0, "filings")).P
    Pr = train_head(HeadConfig("rope", 0, "filings")).P
    best = (-1, -1)
    for i, f in enumerate(filings("held")):
        if len(f.label) <= LAB_FILING_MAX:
            v = f.last >= 0
            n = int((v & (read_filing(Pa, "alibi", f) != f.label) & (read_filing(Pr, "rope", f, "pi") == f.label)).sum())
            best = max(best, (n, i))
    return best[1]


def _mean_r(vals, nd: int = 3):
    v = [x for x in vals if not math.isnan(x)]
    return None if not v else round(float(np.mean(v)), nd)


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    tour, fb = tournament(), filing_benchmark()
    sinP = train_head(HeadConfig("sinusoidal", 0)).P
    M = sinP["Wq"] @ sinP["Wk"].T
    lock = {}
    for dlt in LAB_LOCK_DELTAS:
        m = np.arange(LAB_DELTA_MAX - dlt + 1)
        Pm, Pn = sinusoidal(m, TOUR_FREQS), sinusoidal(m + dlt, TOUR_FREQS)
        lock[str(dlt)] = {"additive": _r(np.einsum("md,de,me->m", Pm, M, Pn) / math.sqrt(D_TOUR)),
                          "projected": _r(float(sinusoidal([0], TOUR_FREQS)[0] @ commutant_projection(M)
                                                @ sinusoidal([dlt], TOUR_FREQS)[0]) / math.sqrt(D_TOUR))}
    offsets = np.arange(LAB_OFFSETS)
    curves = {}
    for scheme in SCHEMES:
        P = train_head(HeadConfig(scheme, 0)).P
        r = offsets[:L_TRAIN] if scheme == "learned" else offsets
        curves[scheme] = _r(learned_offset_score(P, scheme, r), 2)
    census = wavelength_census(L_TRAIN, D_TOUR)
    scales = {}
    for kappa in LAB_KAPPAS:
        scales[str(kappa)] = {"pi": _r(np.full(D_TOUR // 2, 1 / kappa)),
                              "ntk": _r(ntk_frequencies(D_TOUR, kappa) / TOUR_FREQS),
                              "yarn": _r(yarn_frequencies(D_TOUR, kappa, L_TRAIN)[0] / TOUR_FREQS)}
    f = filings("held")[lab_filing()]
    answers = {}
    for scheme, rescue in [(s, None) for s in FILING_SCHEMES] + [("rope", "pi")]:
        P = train_head(HeadConfig(scheme, 0, "filings")).P
        answers[scheme if rescue is None else "rope+pi"] = [int(a) for a in read_filing(P, scheme, f, rescue)]
    return {
        "clocks": {"bases": list(LAB_BASES), "dims": list(LAB_DIMS),
                   "kernel": {f"{int(b)}-{d}": _r(kernel(np.arange(LAB_DELTA_MAX + 1), frequencies(d, b)) / (d / 2))
                              for b in LAB_BASES for d in LAB_DIMS}},
        "ruler": {"deltas": list(LAB_KERNEL_GRID),
                  "kernel": {str(d): _r(kernel(LAB_KERNEL_GRID, frequencies(d)) / (d / 2)) for d in LAB_KERNEL_DIMS},
                  "limit": _r(kernel_limit(LAB_KERNEL_GRID)), "log_ruler": _r(log_ruler(LAB_KERNEL_GRID))},
        "relativity": {"defect": _r(relativity_defect(M)), "lock": lock},
        "tournament": {"lengths": list(TEST_LENGTHS),
                       "accuracy": {s: [_mean_r(tour["schemes"][s][n]) for n in TEST_LENGTHS] for s in SCHEMES},
                       "rescues": {r: [_mean_r(tour["rescues"][r][n]) for n in TEST_LENGTHS] for r in RESCUES},
                       "offset_score": curves},
        "census": {"L": L_TRAIN, "wavelengths": _r(census["wavelengths"], 1),
                   "complete": [bool(c) for c in census["complete"]], "scales": scales},
        "filing": {"index": lab_filing(), "tokens": [w for w, _ in _segments(documents("held")[lab_filing()])], "labels": [-1 if lab is None else lab for _, lab in f.tokens],
                   "truth": [int(v) for v in f.label], "answers": answers,
                   "bands": [list(b) if b[1] < 10**9 else [b[0], None] for b in GAP_BANDS],
                   "band_accuracy": {k: [_mean_r([row["bands"][i] for row in rows]) for i in range(len(GAP_BANDS))]
                                     for k, rows in fb["variants"].items()},
                   "horizon": _r(float(np.mean([a["horizon"] for a in alibi_forecast() if a["source"] == "filings"])),
                                 1)},
    }


# --------------------------------------------------------------------------------------------
# Summary.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def summary() -> dict[str, float]:
    tour, fb = tournament(), filing_benchmark()
    out: dict[str, float] = {}
    for s in SCHEMES:
        for n in TEST_LENGTHS:
            out[f"tour_{s}_{n}"] = _mean_r(tour["schemes"][s][n], 6)
    for r in RESCUES:
        for n in TEST_LENGTHS[1:]:
            out[f"tour_rope+{r}_{n}"] = _mean_r(tour["rescues"][r][n], 6)
    for name, rows in fb["variants"].items():
        out[f"filing_{name}_all"] = float(np.mean([row["all"] for row in rows]))
        out[f"filing_{name}_gap>32"] = float(np.mean([np.average(row["bands"][3:], weights=fb["band_counts"][3:])
                                                      for row in rows]))
    out["filing_queries"] = fb["n_queries"]
    out["filing_queries_gap>32"] = int(sum(fb["band_counts"][3:]))
    for a in alibi_forecast():
        tag = f"alibi_{a['source']}_seed{a['seed']}"
        out[f"{tag}_W"], out[f"{tag}_horizon"] = a["W"], a["horizon"]
        out[f"{tag}_within"], out[f"{tag}_beyond"] = a["within"], a["beyond"]
    sinP = train_head(HeadConfig("sinusoidal", 0)).P
    out["sinusoidal_head_defect"] = relativity_defect(sinP["Wq"] @ sinP["Wk"].T)
    return out


# --------------------------------------------------------------------------------------------
# Verification harness.
# --------------------------------------------------------------------------------------------


def _within(x, se, k: float = 4.0) -> bool:
    return bool(np.all(np.abs(x) <= k * np.asarray(se)))


def test_causal_counter():
    """Prop 1 (i): one causal layer with equal scores returns a BOS value as 1/(t + 1), so t is recovered exactly."""
    o = causal_counter(4096)
    assert np.allclose(o, 1 / (np.arange(4097) + 1), rtol=0, atol=1e-15)
    assert np.array_equal(np.rint(1 / o) - 1, np.arange(4097))


def test_nope_reads_a_bag():
    """Prop 1 (ii) with Topic 9's Prop 3 (ii): a single NoPE head is a multiset reader, so its accuracy is at most the
    best multiset reader's at every length (within Monte Carlo error)."""
    rng = np.random.default_rng([SEED, 1001])
    tour = tournament()["schemes"]["nope"]
    for n in TEST_LENGTHS:
        bayes, se = bag_bayes_accuracy(n, CUE_RATE, 20_000, rng)
        head_se = math.sqrt(0.25 / TEST_N)
        assert max(tour[n]) <= bayes + 4 * math.hypot(se, head_se), (n, tour[n], bayes)


def test_shift_is_rotation():
    """Prop 2: p(m + k) = R(k)p(m); R is a homomorphism into SO(d) with R(k)ᵀ = R(−k); ‖p(m)‖² = d/2; plane 0 alone
    makes the code injective (ω_0 = 1 and π is irrational)."""
    rng = np.random.default_rng([SEED, 1002])
    for d in (16, 64, 512):
        w = frequencies(d)
        for m, k in rng.integers(-10_000, 10_000, size=(20, 2)):
            assert np.allclose(sinusoidal([m + k], w)[0], rotation(k, w) @ sinusoidal([m], w)[0], atol=1e-9)
        a, b = rng.integers(-500, 500, 2)
        assert np.allclose(rotation(a, w) @ rotation(b, w), rotation(a + b, w), atol=1e-12)
        assert np.allclose(rotation(a, w).T, rotation(-a, w), atol=1e-15)
        assert np.allclose(rotation(a, w).T @ rotation(a, w), np.eye(d), atol=1e-12)
        Ra = rotation(a, w)  # det = 1 block by block (LAPACK's det on this sparse matrix raises spurious FP flags)
        blocks = [Ra[2 * j:2 * j + 2, 2 * j:2 * j + 2] for j in range(d // 2)]
        assert np.allclose(linalg.block_diag(*blocks), Ra, atol=0)
        assert np.allclose([B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0] for B in blocks], 1, atol=1e-12)
        assert np.allclose((sinusoidal(np.arange(1000), w) ** 2).sum(1), d / 2)
    angles = np.sort(np.mod(np.arange(10_000), 2 * np.pi))
    assert np.diff(angles).min() > 0


def test_stationary_kernel():
    """Thm 1 (i): ⟨p(m), p(n)⟩ = κ_d(n − m); the Gram matrix on any window is Toeplitz and PSD; (2/d)κ_d(Δ) is the
    characteristic function of the d atoms ±ω_j."""
    rng = np.random.default_rng([SEED, 1003])
    for d in (16, 64):
        w = frequencies(d)
        pos = np.sort(rng.integers(0, 5000, 40))
        G = sinusoidal(pos, w) @ sinusoidal(pos, w).T
        assert np.allclose(G, kernel(pos[None, :] - pos[:, None], w), atol=1e-9)
        assert np.linalg.eigvalsh(G).min() > -1e-9
        atoms = np.concatenate([w, -w])
        for dlt in rng.integers(-2000, 2000, 10):
            cf = np.mean(np.exp(1j * dlt * atoms))
            assert abs(cf.imag) < 1e-12 and abs(cf.real - kernel(dlt, w) / (d / 2)) < 1e-12


def test_logarithmic_ruler():
    """Thm 1 (ii): at d = 8,192 the normalized kernel is the Ci limit to 10⁻³ for |Δ| ≤ 1,024; the largest gap over
    |Δ| ≤ 64 shrinks along d = 64, 512, 8,192; the limit is 1 − (γ + ln|Δ|)/ln b to 0.01 on 16 ≤ |Δ| ≤ 1,024."""
    grid = np.array([x for x in LAB_KERNEL_GRID if x <= 1024])
    assert np.max(np.abs(kernel(grid, frequencies(8192)) / 4096 - kernel_limit(grid))) < 1e-3
    small = np.arange(1, 65)
    gaps = [np.max(np.abs(kernel(small, frequencies(d)) / (d / 2) - kernel_limit(small))) for d in (64, 512, 8192)]
    assert gaps[0] > gaps[1] > gaps[2], gaps
    big = grid[grid >= 16]
    assert np.max(np.abs(kernel_limit(big) - log_ruler(big))) < 0.01
    every = np.arange(1, 5000)  # the explicit error bound of Thm 1 (ii)
    assert np.all(np.abs(kernel_limit(every) - log_ruler(every)) <= (2 / every + every**2 / (4 * BASE**2)) / math.log(BASE))
    assert np.allclose(kernel_limit(-big), kernel_limit(big))


def test_commutant():
    """Thm 2: on a window of d + 1 positions the relative-only M form exactly the commutant (dimension d), at
    well-separated frequencies; every M in it scores offsets only; a Gaussian M has E δ² = 1 − 1/d; with Def 2's
    geometric frequencies the window matrix is badly conditioned until the window outruns the slow planes."""
    rng = np.random.default_rng([SEED, 1004])
    d = 8
    w = np.sort(rng.uniform(0.3, 2.8, d // 2))
    C = window_conditions(w, range(d))
    _, sv, Vt = np.linalg.svd(C)  # C is d² × d², so sv has d² entries
    null = Vt[sv < 1e-9 * sv[0]]
    assert len(null) == d, len(null)
    for v in null:  # every null vector lies in the commutant
        Mv = v.reshape(d, d)
        assert np.allclose(commutant_projection(Mv), Mv, atol=1e-9)
    Mc = commutant_projection(rng.normal(size=(d, d)))
    P = sinusoidal(np.arange(-40, 40), w)
    G = P @ Mc @ P.T
    for dlt in range(-10, 11):
        assert np.ptp(np.diagonal(G, dlt)) < 1e-10
    dd = 16
    defects = np.array([relativity_defect(rng.normal(size=(dd, dd))) ** 2 for _ in range(4000)])
    assert _within(defects.mean() - (1 - 1 / dd), defects.std(ddof=1) / math.sqrt(len(defects)))
    conds = [window_condition(n, TOUR_FREQS) for n in (16, 32, 64, 256, 4096)]
    assert all(a > b for a, b in zip(conds, conds[1:])) and conds[0] > 1e3 * conds[-1], conds


def test_sinusoidal_head_defect():
    """Thm 2's reading: the trained sinusoidal-absolute heads' W_Q W_Kᵀ sit far outside the commutant (δ near a
    random matrix's √(1 − 1/d)) — their position–position scores are not functions of the offset."""
    for s in TOUR_SEEDS:
        P = train_head(HeadConfig("sinusoidal", s)).P
        assert relativity_defect(P["Wq"] @ P["Wk"].T) > 0.8


def test_t5_buckets():
    """Def 3: T5's causal buckets are the identity below 16 and monotone; past 16 each bucket starts about 8^{1/16}
    times later than the one before, and the last bucket holds every offset from 113 on (16·8^{15/16} ≈ 113)."""
    b = [t5_bucket(r) for r in range(1000)]
    assert b[:16] == list(range(16)) and all(x <= y for x, y in zip(b, b[1:]))
    assert b[112] == 30 and set(b[113:]) == {31} and math.ceil(16 * 8 ** (15 / 16)) == 113


def test_alibi_product_kernel():
    """Thm 3: ALiBi's weights are content kernel × e^{−sr} normalized; the bidirectional e^{−s|i−j|} and its product
    with a content Gram matrix are PSD; the weight beyond offset D obeys e^{W}e^{−sD}/(1 − e^{−s}) at every length
    to 8,192; and α_r/α_0 ≤ e^{W − sr}."""
    rng = np.random.default_rng([SEED, 1005])
    for _ in range(30):
        n, s, W = int(rng.integers(8, 8192)), float(rng.uniform(0.05, 1)), float(rng.uniform(0.5, 12))
        c = rng.uniform(-W / 2, W / 2, n)
        r = np.arange(n)[::-1]  # offsets of keys 0…n−1 from the query at n − 1
        a = ak._softmax((c - s * r)[None])[0]
        prod = np.exp(c) * np.exp(-s * r)
        assert np.allclose(a, prod / prod.sum(), atol=1e-12)
        for D in (1, 4, 16, 64):
            assert a[r >= D].sum() <= alibi_tail_bound(W, s, D) * (1 + 1e-12)
        assert np.all(a / a[r == 0] <= np.exp(W - s * r) * (1 + 1e-9))
    pos = np.arange(30)
    lap = np.exp(-0.3 * np.abs(pos[:, None] - pos[None, :]))
    X = rng.normal(size=(30, 4))
    assert np.linalg.eigvalsh(lap).min() > 0 and np.linalg.eigvalsh(lap * np.exp(X @ X.T / 2)).min() > -1e-9


def test_rope_relative_isometric():
    """Prop 3: ⟨R(m)q, R(n)k⟩ = ⟨q, R(n − m)k⟩; norms are kept; the offset score is Σ_j ρ_j cos(Δθ_j + φ_j); and
    Topic 9's attention() on rotated queries and keys equals the explicit R(n − m) form."""
    rng = np.random.default_rng([SEED, 1006])
    w = frequencies(64)
    for _ in range(20):
        q, k = rng.normal(size=64), rng.normal(size=64)
        m, n = rng.integers(-5000, 5000, 2)
        lhs = rotate(q[None], [m], w)[0] @ rotate(k[None], [n], w)[0]
        assert abs(lhs - q @ rotation(n - m, w) @ k) < 1e-9
        assert abs(np.linalg.norm(rotate(q[None], [m], w)) - np.linalg.norm(q)) < 1e-12
        rho, phi = trig_form(q, k)
        dl = rng.integers(-3000, 3000, 30)
        assert np.allclose(offset_score(q, k, dl, w), (rho * np.cos(np.multiply.outer(dl, w) + phi)).sum(1), atol=1e-9)
    Q, K, V = rng.normal(size=(3, 12, 16))
    pos = np.arange(12)
    O, A = ak.attention(rotate(Q, pos, TOUR_FREQS), rotate(K, pos, TOUR_FREQS), V, causal=True)
    S = np.array([[Q[i] @ rotation(j - i, TOUR_FREQS) @ K[j] for j in range(12)] for i in range(12)]) / 4
    assert np.allclose(A, ak._softmax(np.where(np.tri(12, dtype=bool), S, -np.inf)), atol=1e-12)


def test_general_relative_encoding():
    """Thm 4: powers of an orthogonal U score offsets only; scipy's real Schur form recovers U as Q(⊕R(θ_i) ⊕ I_p ⊕
    −I_r)Qᵀ with θ_i ∈ (0, π), reproducing U^m for negative and positive m; a non-orthogonal family fails (a)."""
    rng = np.random.default_rng([SEED, 1007])
    for d in (7, 8, 9):
        Z, _ = np.linalg.qr(rng.normal(size=(d, d)))
        cases = [Z]
        Qm, _ = np.linalg.qr(rng.normal(size=(d, d)))  # a U with both a +1 and a −1 axis
        R11 = np.array([[math.cos(1.1), -math.sin(1.1)], [math.sin(1.1), math.cos(1.1)]])
        cases.append(Qm @ linalg.block_diag(R11, np.diag([1.0, -1.0]), np.eye(d - 4)) @ Qm.T)
        for U in cases:
            ms = range(-3, 6)
            L = orthogonal_family(U, ms)
            for i in range(len(ms) - 1):  # L_mᵀL_n = L_{m+1}ᵀL_{n+1}: the score sees n − m only
                for j in range(len(ms) - 1):
                    assert np.allclose(L[i].T @ L[j], L[i + 1].T @ L[j + 1], atol=1e-9)
            Q, ang, p, r = real_normal_form(U)
            assert np.allclose(Q.T @ Q, np.eye(d), atol=1e-9) and np.all((ang > 0) & (ang < np.pi))
            assert 2 * len(ang) + p + r == d
            for m in ms:
                assert np.allclose(normal_form_power(Q, ang, p, r, m), L[list(ms).index(m)], atol=1e-8)
        A = Z @ np.diag(rng.uniform(0.6, 1.4, d)) @ Z.T
        LA = orthogonal_family(A, [0, 1, 2])
        assert not np.allclose(LA[0].T @ LA[1], LA[1].T @ LA[2], atol=1e-3)


def test_long_term_decay():
    """Prop 4: (i) the summation-by-parts bound holds, and the envelope's octave means fall (d = 128); (ii) a matched
    pair with isotropic planes has E⟨q, R(Δ)q⟩ = σ²κ_d(Δ); (iii) for independent q, k the law of ⟨q, R(Δ)k⟩ does not
    depend on Δ: mean 0, variance dσ⁴/4."""
    rng = np.random.default_rng([SEED, 1008])
    w = frequencies(128)
    for _ in range(200):
        h = rng.normal(size=64) + 1j * rng.normal(size=64)
        e = abel_envelope(h, float(rng.uniform(0, 5000)), w)
        assert e["value"] <= e["bound"] * (1 + 1e-12)
    env = np.array([abel_envelope(np.ones(64), dl, w)["envelope"] for dl in range(1, 4096)])
    octaves = [env[2**k - 1:2 ** (k + 1) - 1].mean() for k in range(12)]
    assert all(a > b for a, b in zip(octaves, octaves[1:])), np.round(octaves, 3)
    d, N, sigma = 16, 200_000, 1.3
    q = rng.normal(0, sigma / math.sqrt(2), (N, d))
    k = rng.normal(0, sigma / math.sqrt(2), (N, d))
    for dl in (0, 1, 7, 50, 400):
        x = (rotate(q, np.full(N, dl), TOUR_FREQS) * q).sum(1)
        assert _within(x.mean() - sigma**2 * kernel(dl, TOUR_FREQS), x.std(ddof=1) / math.sqrt(N))
        y = (rotate(k, np.full(N, dl), TOUR_FREQS) * q).sum(1)
        assert _within(y.mean(), y.std(ddof=1) / math.sqrt(N))
        assert abs(y.var() / (d * sigma**4 / 4) - 1) < 0.02


def test_wavelength_census():
    """Thm 5 (i)–(iii): at d = 16, L = 32, two of eight planes turn a whole circle; PI keeps every test offset within
    one position of the trained range; NTK-aware leaves plane 0 and gives the slowest plane exactly 1/κ; YaRN leaves
    the planes with r_j ≥ β unchanged and gives those with r_j ≤ α exactly PI."""
    c = wavelength_census(L_TRAIN, D_TOUR)
    assert c["n_complete"] == 2 and np.allclose(c["wavelengths"][:2], [2 * np.pi, 2 * np.pi * 10 ** 0.5])
    assert c["n_complete"] == int((BASE ** (2 * np.arange(8) / 16) <= (L_TRAIN - 1) / (2 * np.pi)).sum())
    for n in (64, 128, 256, 4096):
        top = (n - 1) * pi_scale(n, L_TRAIN)
        assert L_TRAIN - 1 < top < L_TRAIN
    for kappa in (2.0, 4.0, 8.0):
        r = ntk_frequencies(D_TOUR, kappa) / TOUR_FREQS
        assert abs(r[0] - 1) < 1e-15 and abs(r[-1] - 1 / kappa) < 1e-12 and np.all(np.diff(r) < 0)
        th, temp = yarn_frequencies(D_TOUR, kappa, L_TRAIN)
        rj = L_TRAIN * TOUR_FREQS / (2 * np.pi)
        assert np.allclose(th[rj >= 32], TOUR_FREQS[rj >= 32]) and np.allclose(th[rj <= 1], TOUR_FREQS[rj <= 1] / kappa)
        assert abs(temp - (0.1 * math.log(kappa) + 1) ** 2) < 1e-15


def test_interpolation_bound():
    """Thm 5 (iv): linear interpolation of a(Δ) = Re Σ h_j e^{iΔθ_j} between neighbors errs by at most
    (Δ − Δ₁)(Δ₂ − Δ)/2 · Σ|h_j|θ_j²; Σ_j θ_j² = (1 − b^{−2})/(1 − b^{−4/d}); and Chen et al.'s d/(4 ln b) falls below
    it, by 2.56× at d = 16 and 1.04× at d = 512."""
    rng = np.random.default_rng([SEED, 1009])
    for d in (16, 128):
        w = frequencies(d)
        assert abs((w**2).sum() - (1 - BASE**-2) / (1 - BASE ** (-4 / d))) < 1e-12
        for _ in range(300):
            h = rng.normal(size=d // 2) + 1j * rng.normal(size=d // 2)
            d1 = float(rng.integers(0, 5000))
            x = d1 + float(rng.uniform())
            a = lambda z: float(np.real((h * np.exp(1j * z * w)).sum()))  # noqa: E731
            lin = a(d1) + (x - d1) * (a(d1 + 1) - a(d1))
            assert abs(a(x) - lin) <= interpolation_bound(h, w, d1, d1 + 1, x) * (1 + 1e-9) + 1e-12
    assert abs((frequencies(16) ** 2).sum() / chen_constant(16) - 2.558) < 1e-3
    assert abs((frequencies(512) ** 2).sum() / chen_constant(512) - 1.036) < 1e-3
    for d in (16, 64, 128):  # h_j = 1 on [0, 1] at ½: the stated bound fails, the sharp one holds (d = 128 is LLaMA-7B's)
        w, h = frequencies(d), np.ones(d // 2)
        err = abs(np.cos(w / 2).sum() - (d // 2 + np.cos(w).sum()) / 2)
        assert chen_stated_bound(h, d, 0, 1, 0.5) < err <= interpolation_bound(h, w, 0, 1, 0.5), d


def test_gradients():
    """Every scheme's hand-written gradient (and YaRN's logit scale) against Topic 5's central differences."""
    rng = np.random.default_rng([SEED, 1010])
    for scheme in SCHEMES:
        P = init_head(rng, scheme, 12)
        X, y, _ = cue_data(rng, 5, 12, 0.2)
        kw = {"temp": 1.7} if scheme == "rope" else {}
        _, G = head_loss_grad(P, X, y, scheme, **kw)
        loss = lambda Q: head_loss_grad(Q, X, y, scheme, **kw)[0]  # noqa: E731
        num = nlm.param_grad(nlm.central_diff_grad, loss, P)
        for key in P:
            assert np.max(np.abs(num[key] - G[key])) < 1e-7, (scheme, key)


def test_filing_reader_matches_forward():
    """The one-pass filing reader's log-probabilities equal the plain forward pass on the explicit prefix + query to
    10⁻⁹, at every position of a held-out filing, for every scheme and rescue."""
    f = max(filings("held")[:10], key=lambda x: len(x.label))
    for scheme, rescue in [(s, None) for s in FILING_SCHEMES] + [("rope", r) for r in RESCUES]:
        P = train_head(HeadConfig(scheme, 0, "filings")).P
        fast = filing_logprobs(P, scheme, f, rescue)
        for t in range(len(f.label)):
            X = np.vstack([f.X[:t + 1], np.eye(F_IN)[QUERY]])[None]
            lp = _head_forward(P, X, scheme, **rescue_kwargs(rescue, t + 2))[6][0]
            assert np.max(np.abs(lp - fast[t])) < 1e-9, (scheme, rescue, t)


def test_tournament():
    """§5: in distribution every scheme but NoPE reads the cue; past the window the sinusoidal-absolute heads
    collapse, the ALiBi heads hold, the RoPE heads degrade, and every rescue lifts RoPE at 256."""
    t = tournament()
    S, R = t["schemes"], t["rescues"]
    for s in ("sinusoidal", "learned", "rope", "alibi"):
        assert min(S[s][32]) >= 0.95, (s, S[s][32])
    assert max(S["nope"][32]) < 0.7
    assert all(math.isnan(x) for n in TEST_LENGTHS[1:] for x in S["learned"][n])
    assert max(max(S["sinusoidal"][n]) for n in TEST_LENGTHS[1:]) < 0.35
    assert min(min(S["alibi"][n]) for n in TEST_LENGTHS) >= 0.9
    for seed in range(len(TOUR_SEEDS)):
        rope = [S["rope"][n][seed] for n in TEST_LENGTHS]
        assert all(a > b for a, b in zip(rope, rope[1:])), rope
        for r in RESCUES:
            assert R[r][256][seed] > S["rope"][256][seed] + 0.2, (r, seed)


def test_rope_rebound():
    """§5 with Prop 3 (iii) and Thm 5 (i): each trained RoPE head's cue score falls monotonically from offset 1 to a
    trough within 3 of λ₂/2 ≈ 31.4 (half a turn of the fastest plane that never completes one in training), then
    rises again; its largest amplitude is in plane 3 (λ₃ ≈ 199); the rebound peaks between 170 and 200, near 3λ₂ ≈ 188
    and λ₃, above the score of every in-window cue 16 or more back."""
    lam = wavelength_census(L_TRAIN, D_TOUR)["wavelengths"]
    for s in TOUR_SEEDS:
        P = train_head(HeadConfig("rope", s)).P
        sc = learned_offset_score(P, "rope", np.arange(LAB_OFFSETS))
        trough = 1 + int(np.argmax(np.diff(sc[1:]) > 0))
        assert abs(trough - lam[2] / 2) <= 3 and np.all(np.diff(sc[1:trough + 1]) < 0), (s, trough)
        peak = L_TRAIN + int(np.argmax(sc[L_TRAIN:]))
        assert 170 <= peak <= 200 and sc[peak] > sc[16], (s, peak)
        assert int(np.argmax(trig_form(*cue_query_key(P))[0])) == 3


def test_filings():
    """§6: on whole held-out filings, with heads trained on 32-token windows, the two interpolating rescues (PI and
    YaRN) read the queries past the window better than every other reader in every seed; and Thm 3 (iii) forecasts
    ALiBi: inside half its horizon every head, from either training source, answers every query, and beyond the
    horizon every head falls below 0.75 — below 0.15 for the heads trained on the synthetic stream."""
    fb = filing_benchmark()
    V = fb["variants"]
    far = lambda rows: [np.average(r["bands"][3:], weights=fb["band_counts"][3:]) for r in rows]  # noqa: E731
    best_other = max(max(far(V[k])) for k in V if k not in ("rope+pi", "rope+yarn"))
    assert min(far(V["rope+pi"])) > best_other and min(far(V["rope+yarn"])) > best_other
    for a in alibi_forecast():
        assert a["n_within"] > 1000 and a["n_beyond"] > 1000, a
        assert a["within"] == 1.0 and a["beyond"] < 0.75, a
        assert a["source"] == "filings" or a["beyond"] < 0.15, a


def test_viz_payload():
    """The lab's data: kernels for every base and dimension; one curve per scheme over 256 offsets (32 for the
    learned table); a scale per plane for every rescue and κ; one answer per filing position for every reader; the
    file stays under 60 kB."""
    p = viz_payload()
    assert all(len(v) == LAB_DELTA_MAX + 1 for v in p["clocks"]["kernel"].values())
    assert all(len(v) == len(LAB_KERNEL_GRID) for v in p["ruler"]["kernel"].values())
    for s, curve in p["tournament"]["offset_score"].items():
        assert len(curve) == (L_TRAIN if s == "learned" else LAB_OFFSETS)
    assert all(len(v) == D_TOUR // 2 for sc in p["census"]["scales"].values() for v in sc.values())
    n = len(p["filing"]["tokens"])
    assert all(len(v) == n for v in p["filing"]["answers"].values()) and len(p["filing"]["truth"]) == n
    assert len(p["filing"]["labels"]) == n == len(filings("held")[lab_filing()].label)
    truth, ans = np.array(p["filing"]["truth"]), {k: np.array(v) for k, v in p["filing"]["answers"].items()}
    assert np.all(truth >= 0) and np.all(ans["rope+pi"] == truth)  # §6: RoPE + PI answers every position of the lab's filing
    assert all((ans[k] != truth).sum() > 100 for k in ("nope", "sinusoidal", "rope", "alibi"))
    assert len(json.dumps(p, separators=(",", ":"))) < 60_000


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "positional-encoding-lab.json"


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
    print("\nSummary (cue task: 8 symbols, cue rate 0.1, heads trained at 32; filings: generate_corpus(300, 0) windows,"
          " held generate_corpus(100, 1)):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.6f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
