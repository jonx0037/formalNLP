"""The transformer block and the residual stream — the reference implementation for formalNLP's eleventh topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. The block, assembled: multi-head attention is a sum of Topic 9's smoothers, each writing into a shared stream
   through its own output map; the MLP is a sum of gated value vectors; pre-LN and post-LN differ only in where
   the normalization sits.
2. Attention alone collapses: a row-stochastic matrix shrinks the diameter of the token cloud by its Dobrushin
   coefficient τ(A) ≤ 1 − e^{−r} (r the score range), so a stack of wide-kernel attention layers drives every
   token to one point; the skip connection keeps a copy of each token, and the diameter can shrink by at most
   1 − τ‖W‖ per layer. A narrow kernel (τ ≈ 1) freezes the cloud into clusters, where the bound is silent.
3. LayerNorm is a sphere in a hyperplane: LN(x) = √d · Px/‖Px‖ with P the centering projection; it is
   scale-invariant, and its Jacobian (√d/‖Px‖)(P − uuᵀ) kills the 1 direction and x itself.
4. The identity path: the Jacobian of a residual stack is ∏(I + J_k), a sum over 2^L paths that always holds I;
   pre-LN keeps every layer's gradient on that path, post-LN multiplies it by LN's Jacobian at every layer.
   Xiong et al.'s gradient profiles at initialization, measured.
5. The stream as a channel: an attention-only model's logits are exactly a sum over paths — the direct path
   W_E W_U, one-head paths A^h X W_OV^h W_U, and composed paths A^{h'} A^h X W_OV^h W_OV^{h'} W_U.
6. Induction heads: a 2-layer attention-only model with RoPE (Topic 10) trained on repeated random sequences
   learns a previous-token head and an induction head that reads it through its keys (K-composition); a
   1-layer model cannot; cutting the K-composition returns the 2-layer model to the 1-layer model's loss. The
   logit lens reads the stream between layers.
7. Finance case study: completing a company's name. The bigram table and a 2-layer model trained on the filings
   complete names they saw in training and none they did not; the induction model, which never saw a filing,
   completes both.

Cross-checks: every hand-written gradient against Topic 5's central differences; LayerNorm's Jacobian against
central differences; the path expansion against the forward pass; the Jacobian expansion against the chain rule.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/transformer-block-residual-stream/transformer_block_residual_stream.py
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "positional-encodings-geometry"))
import positional_encodings_geometry as pe  # noqa: E402 — a module object: its test_* stay out of this harness

ak, fc, nlm = pe.ak, pe.fc, pe.nlm

# ONE stream per random-instance claim and per trained model, each from default_rng([SEED, k]); k ≥ 1400 keeps this
# module's streams apart from Topics 5–10.
SEED = ak.SEED
Params = dict[str, np.ndarray]


def _swap(X: np.ndarray) -> np.ndarray:
    return np.swapaxes(X, -1, -2)


# --------------------------------------------------------------------------------------------
# 1 and 3. The block, assembled; LayerNorm.
# --------------------------------------------------------------------------------------------


def center(X: np.ndarray) -> np.ndarray:
    """P x = x − mean(x)·1 on each row: the orthogonal projection onto the hyperplane 1^⊥."""
    return X - X.mean(-1, keepdims=True)


def layer_norm(X: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Def 3 as Ba et al. write it (gain 1, bias 0): (x − μ)/√(σ² + ε) on each row."""
    C = center(X)
    return C / np.sqrt((C**2).mean(-1, keepdims=True) + eps)


def ln_sphere(X: np.ndarray) -> np.ndarray:
    """Prop 3: √d · Px/‖Px‖ — the projection onto 1^⊥, then onto the sphere of radius √d in it."""
    C = center(X)
    return math.sqrt(X.shape[-1]) * C / np.linalg.norm(C, axis=-1, keepdims=True)


def rms_norm(X: np.ndarray) -> np.ndarray:
    """RMSNorm (Zhang and Sennrich): √d · x/‖x‖ — LayerNorm without the centering."""
    return math.sqrt(X.shape[-1]) * X / np.linalg.norm(X, axis=-1, keepdims=True)


def ln_jacobian(x: np.ndarray) -> np.ndarray:
    """Cor 1: ∂LN/∂x = (√d/‖Px‖)(P − uuᵀ), u = Px/‖Px‖ — symmetric."""
    d = len(x)
    c = center(x)
    r = np.linalg.norm(c)
    u = c / r
    return math.sqrt(d) / r * (np.eye(d) - np.ones((d, d)) / d - np.outer(u, u))


def ln_backward(dY: np.ndarray, Y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Jᵀ dY row by row, from the output Y = LN(X) and r = ‖PX‖ (shape (…, 1)); J is symmetric and u = Y/√d ⊥ 1."""
    d = Y.shape[-1]
    u = Y / math.sqrt(d)
    dc = center(dY)
    return math.sqrt(d) / r * (dc - u * (u * dc).sum(-1, keepdims=True))


def _ln_forward(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(center(X), axis=-1, keepdims=True)
    return ln_sphere(X), r


def init_mha(rng: np.random.Generator, d: int, H: int, dh: int, bias: float = 0.0) -> Params:
    """H heads of width dh: Wq, Wk, Wv (H, d, dh) and Wo (H, dh, d) at N(0, 1/fan-in); query and key biases N(0, bias²)
    (with RoPE, the biases carry the content-free part of the score: Topic 10's kernel over offsets)."""
    return {"Wq": rng.normal(0, 1 / math.sqrt(d), (H, d, dh)), "Wk": rng.normal(0, 1 / math.sqrt(d), (H, d, dh)),
            "Wv": rng.normal(0, 1 / math.sqrt(d), (H, d, dh)), "Wo": rng.normal(0, 1 / math.sqrt(H * dh), (H, dh, d)),
            "bq": rng.normal(0, bias, (H, dh)), "bk": rng.normal(0, bias, (H, dh))}


def mha_forward(L: Params, X: np.ndarray, causal: bool = True, freqs: np.ndarray | None = None,
                X_key: np.ndarray | None = None) -> tuple[np.ndarray, tuple]:
    """Def 1: Σ_h softmax(q_h k_hᵀ/√dh) X W_V^h W_O^h on a batch X (B, n, d); with `freqs`, q and k turn by Topic 10's
    RoPE. X_key, if given, feeds the keys instead of X (the K-composition ablation). Returns (out, cache)."""
    n, dh = X.shape[-2], L["Wq"].shape[-1]
    Xk = X if X_key is None else X_key
    q = X[:, None] @ L["Wq"][None] + L["bq"][None, :, None, :]
    k = Xk[:, None] @ L["Wk"][None] + L["bk"][None, :, None, :]
    v = X[:, None] @ L["Wv"][None]
    pos = np.arange(n, dtype=float)
    if freqs is not None:
        q, k = pe.rotate(q, pos, freqs), pe.rotate(k, pos, freqs)
    S = q @ _swap(k) / math.sqrt(dh)
    if causal:
        S = np.where(np.tri(n, dtype=bool), S, -np.inf)
    A = ak._softmax(S)
    o = A @ v
    out = (o @ L["Wo"][None]).sum(1)
    return out, (X, q, k, v, A, o, freqs)


def mha_backward(L: Params, dOut: np.ndarray, cache: tuple) -> tuple[np.ndarray, Params]:
    """The gradient of mha_forward (keys from X), checked against central differences by test_gradients."""
    X, q, k, v, A, o, freqs = cache
    n, dh = X.shape[-2], L["Wq"].shape[-1]
    G = {"Wo": (_swap(o) @ dOut[:, None]).sum(0)}
    do = dOut[:, None] @ _swap(L["Wo"])[None]
    da = do @ _swap(v)
    dv = _swap(A) @ do
    dS = A * (da - (A * da).sum(-1, keepdims=True)) / math.sqrt(dh)
    dq, dk = dS @ k, _swap(dS) @ q
    if freqs is not None:  # the rotation is orthogonal: pull the gradient back by its transpose
        pos = np.arange(n, dtype=float)
        dq, dk = pe.rotate(dq, -pos, freqs), pe.rotate(dk, -pos, freqs)
    XT = _swap(X)[:, None]
    G["Wq"], G["Wk"], G["Wv"] = (XT @ dq).sum(0), (XT @ dk).sum(0), (XT @ dv).sum(0)
    G["bq"], G["bk"] = dq.sum((0, 2)), dk.sum((0, 2))
    dX = (dq @ _swap(L["Wq"])[None] + dk @ _swap(L["Wk"])[None] + dv @ _swap(L["Wv"])[None]).sum(1)
    return dX, G


def mlp_forward(M: Params, X: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Def 1's MLP: ReLU(X W₁ + b₁) W₂ + b₂."""
    Z = X @ M["W1"] + M["b1"]
    h = np.maximum(Z, 0)
    return h @ M["W2"] + M["b2"], (X, Z, h)


def mlp_backward(M: Params, dOut: np.ndarray, cache: tuple) -> tuple[np.ndarray, Params]:
    X, Z, h = cache
    dZ = (dOut @ M["W2"].T) * (Z > 0)
    G = {"W2": np.einsum("bnw,bnd->wd", h, dOut), "b2": dOut.sum((0, 1)),
         "W1": np.einsum("bnd,bnw->dw", X, dZ), "b1": dZ.sum((0, 1))}
    return dZ @ M["W1"].T, G


def key_value_memory(M: Params, x: np.ndarray) -> np.ndarray:
    """Prop 1 (ii): the MLP as Σ_i ReLU(⟨x, k_i⟩ + b_i) v_i over its key columns k_i of W₁ and value rows v_i of W₂."""
    out = M["b2"].copy()
    for i in range(M["W1"].shape[1]):
        out = out + max(float(x @ M["W1"][:, i] + M["b1"][i]), 0.0) * M["W2"][i]
    return out


# --------------------------------------------------------------------------------------------
# 2. Attention alone collapses.
# --------------------------------------------------------------------------------------------


def dobrushin(A: np.ndarray) -> float:
    """Def 2: τ(A) = ½ max_{i,j} ‖A_i· − A_j·‖₁ = 1 − min_{i,j} Σ_k min(A_ik, A_jk) for a row-stochastic A."""
    return float(1 - np.minimum(A[:, None, :], A[None, :, :]).sum(-1).min())


def diameter(X: np.ndarray, ord=2) -> float:
    """max_{i,j} ‖x_i − x_j‖ over the rows of X."""
    return float(np.linalg.norm(X[:, None, :] - X[None, :, :], ord=ord, axis=-1).max())


def relative_spread(X: np.ndarray) -> float:
    """‖PX‖_F/‖X‖_F with P = I − 11ᵀ/n acting on the tokens: 0 iff every row is the same."""
    return float(np.linalg.norm(X - X.mean(0)) / np.linalg.norm(X))


def smoothing_stack(X: np.ndarray, beta: float, depth: int, residual: bool,
                    W: np.ndarray | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Def 2: X ← A X W (pure attention) or X ← X + A X W (with the skip), A = softmax(β X Xᵀ) by Topic 9's
    attention (queries = keys = X, inverse temperature β). Returns the clouds X_0 … X_depth and the matrices A_l."""
    W = np.eye(X.shape[1]) if W is None else W
    Xs, As = [X], []
    for _ in range(depth):
        O, A = ak.attention(Xs[-1], Xs[-1], Xs[-1] @ W, tau=1 / beta)
        Xs.append(Xs[-1] + O if residual else O)
        As.append(A)
    return Xs, As


COLLAPSE_N, COLLAPSE_D, COLLAPSE_DEPTH = 10, 2, 12
COLLAPSE_BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)


@lru_cache(maxsize=1)
def collapse_cloud() -> np.ndarray:
    """The lab's ten tokens in the plane, N(0, I₂)."""
    return np.random.default_rng([SEED, 1400]).normal(size=(COLLAPSE_N, COLLAPSE_D))


@lru_cache(maxsize=None)
def collapse_run(beta: float, residual: bool) -> dict:
    Xs, As = smoothing_stack(collapse_cloud(), beta, COLLAPSE_DEPTH, residual)
    taus = [dobrushin(A) for A in As]
    return {"X": Xs, "tau": taus, "spread": [relative_spread(X) for X in Xs],
            "diam": [diameter(X) for X in Xs], "bound": list(np.cumprod([1.0] + taus) * diameter(Xs[0]))}


# --------------------------------------------------------------------------------------------
# 4. The identity path: pre-LN and post-LN stacks at initialization.
# --------------------------------------------------------------------------------------------

GRAD_D, GRAD_H, GRAD_DH, GRAD_WIDTH, GRAD_N = 64, 4, 16, 256, 16
GRAD_DEPTHS = (6, 12, 24)
GRAD_SEEDS = 8
SCHEMES = ("pre", "post")


def init_block(rng: np.random.Generator, d: int = GRAD_D, H: int = GRAD_H, dh: int = GRAD_DH,
               width: int = GRAD_WIDTH) -> dict[str, Params]:
    """One block at Xavier-style initialization (Xiong et al.'s setting): attention maps N(0, 1/fan-in), the MLP's
    W₁ N(0, 2/d) (He, for the ReLU) and W₂ N(0, 1/width), zero biases."""
    return {"attn": init_mha(rng, d, H, dh),
            "mlp": {"W1": rng.normal(0, math.sqrt(2 / d), (d, width)), "b1": np.zeros(width),
                    "W2": rng.normal(0, 1 / math.sqrt(width), (width, d)), "b2": np.zeros(d)}}


def stack_forward(blocks: list[dict[str, Params]], X: np.ndarray, scheme: str) -> tuple[list[np.ndarray], list]:
    """Def 1: pre-LN x ← x + Attn(LN x), x ← x + MLP(LN x); post-LN x ← LN(x + Attn x), x ← LN(x + MLP x).
    Bidirectional attention, no positions. Returns the stream before each block (and after the last) and caches."""
    xs, caches = [X], []
    for blk in blocks:
        x = xs[-1]
        if scheme == "pre":
            y, r1 = _ln_forward(x)
            a, ca = mha_forward(blk["attn"], y, causal=False)
            x = x + a
            z, r2 = _ln_forward(x)
            m, cm = mlp_forward(blk["mlp"], z)
            x = x + m
            caches.append((y, r1, ca, z, r2, cm))
        else:
            a, ca = mha_forward(blk["attn"], x, causal=False)
            x, r1 = _ln_forward(x + a)
            y = x
            m, cm = mlp_forward(blk["mlp"], x)
            x, r2 = _ln_forward(x + m)
            caches.append((y, r1, ca, x, r2, cm))
        xs.append(x)
    return xs, caches


def stack_loss_grad(blocks: list[dict[str, Params]], X: np.ndarray, R: np.ndarray,
                    scheme: str) -> tuple[float, list[dict[str, Params]], list[np.ndarray]]:
    """ℒ = ⟨R, LN(x_L)⟩/n — a random readout through a final LayerNorm (idempotent on a post-LN output) — with its
    gradient for every parameter and the gradient ∂ℒ/∂x_l of the stream before every block."""
    xs, caches = stack_forward(blocks, X, scheme)
    out, r = _ln_forward(xs[-1])
    n = X.shape[-2]
    loss = float((R * out).sum() / n)
    g = ln_backward(R / n, out, r)
    grads, dstream = [], [g]
    for blk, (y, r1, ca, z, r2, cm) in zip(reversed(blocks), reversed(caches)):
        if scheme == "pre":
            dz, Gm = mlp_backward(blk["mlp"], g, cm)
            g = g + ln_backward(dz, z, r2)
            dy, Ga = mha_backward(blk["attn"], g, ca)
            g = g + ln_backward(dy, y, r1)
        else:
            g = ln_backward(g, z, r2)
            dx, Gm = mlp_backward(blk["mlp"], g, cm)
            g = ln_backward(g + dx, y, r1)
            dx, Ga = mha_backward(blk["attn"], g, ca)
            g = g + dx
        grads.append({"attn": Ga, "mlp": Gm})
        dstream.append(g)
    return loss, grads[::-1], dstream[::-1]


@lru_cache(maxsize=None)
def gradient_profile(scheme: str, depth: int) -> dict:
    """‖∂ℒ/∂W₂^{(l)}‖_F (each block's MLP output map) and ‖x_l‖ per layer, as root-mean-squares over GRAD_SEEDS draws
    of the blocks, the input, and the readout; one stream per (scheme, depth), all draws from it."""
    rng = np.random.default_rng([SEED, 1410 + 10 * SCHEMES.index(scheme) + GRAD_DEPTHS.index(depth)])
    w2, norms = [], []
    for _ in range(GRAD_SEEDS):
        blocks = [init_block(rng) for _ in range(depth)]
        X = rng.normal(size=(1, GRAD_N, GRAD_D))
        R = rng.normal(size=(1, GRAD_N, GRAD_D))
        _, grads, _ = stack_loss_grad(blocks, X, R, scheme)
        xs, _ = stack_forward(blocks, X, scheme)
        w2.append([np.linalg.norm(G["mlp"]["W2"]) for G in grads])
        norms.append([np.sqrt((x**2).sum(-1).mean()) for x in xs])
    return {"w2": np.sqrt(np.mean(np.square(w2), 0)), "norm": np.sqrt(np.mean(np.square(norms), 0))}


def jacobian_paths(Js: list[np.ndarray]) -> np.ndarray:
    """Thm 2: Σ over subsets S = {k₁ < … < k_m} of the layers of J_{k_m} ⋯ J_{k₁} — the 2^L terms of ∏(I + J_k)."""
    d = Js[0].shape[0]
    total = np.zeros((d, d))
    for m in range(len(Js) + 1):
        for S in itertools.combinations(range(len(Js)), m):
            T = np.eye(d)
            for k in S:
                T = Js[k] @ T
            total += T
    return total


# --------------------------------------------------------------------------------------------
# 5 and 6. The attention-only model: path expansion and induction heads.
# --------------------------------------------------------------------------------------------

V_TOK, N_CTX, D_MODEL, N_HEADS, D_HEAD = 256, 48, 32, 2, 16
ROPE = pe.frequencies(D_HEAD)
PERIODS = (6, 24)  # a repeated sequence's period is uniform on 6 … 24


def init_model(rng: np.random.Generator, layers: int) -> Params:
    """Def 4: embedding E (V × d), unembedding U (d × V), and per layer the heads of init_mha with key and query
    biases N(0, ½²). No LayerNorm, no MLP: an attention-only model, so the path expansion is exact."""
    P = {"E": rng.normal(0, 1, (V_TOK, D_MODEL)), "U": rng.normal(0, 1 / math.sqrt(D_MODEL), (D_MODEL, V_TOK))}
    for l in range(layers):
        P.update({f"{key}{l}": v for key, v in init_mha(rng, D_MODEL, N_HEADS, D_HEAD, bias=0.5).items()})
    return P


def _layer(P: Params, l: int) -> Params:
    return {key: P[f"{key}{l}"] for key in ("Wq", "Wk", "Wv", "Wo", "bq", "bk")}


def n_layers(P: Params) -> int:
    return sum(1 for key in P if key.startswith("Wq"))


def model_forward(P: Params, tok: np.ndarray, ablate_k: bool = False) -> tuple[list[np.ndarray], np.ndarray, list]:
    """The stream x_0 = E[tok], x_{l+1} = x_l + MHA_l(x_l) (causal, RoPE), and the logits x_L U. With ablate_k, every
    layer after the first computes its keys from x_0, so no head can read an earlier head's output through its keys.
    Returns (stream, logits, caches)."""
    xs, caches = [P["E"][tok]], []
    for l in range(n_layers(P)):
        out, cache = mha_forward(_layer(P, l), xs[-1], freqs=ROPE, X_key=xs[0] if ablate_k and l > 0 else None)
        xs.append(xs[-1] + out)
        caches.append(cache)
    return xs, xs[-1] @ P["U"], caches


def masked_xent(logits: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> float:
    lp = nlm.log_softmax(logits)
    return float(-(np.take_along_axis(lp, tgt[..., None], -1)[..., 0] * mask).sum() / mask.sum())


def model_loss_grad(P: Params, tok: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> tuple[float, Params]:
    """The masked next-token cross-entropy and its hand-written gradient (checked by test_gradients)."""
    xs, logits, caches = model_forward(P, tok)
    lp = nlm.log_softmax(logits)
    cnt = mask.sum()
    g = np.exp(lp)
    np.put_along_axis(g, tgt[..., None], np.take_along_axis(g, tgt[..., None], -1) - 1, -1)
    g *= mask[..., None] / cnt
    G = {"U": np.einsum("bnd,bnv->dv", xs[-1], g)}
    dx = g @ P["U"].T
    for l in reversed(range(n_layers(P))):
        d_in, Gl = mha_backward(_layer(P, l), dx, caches[l])
        dx = dx + d_in
        G.update({f"{key}{l}": v for key, v in Gl.items()})
    onehot = np.zeros((tok.size, V_TOK))
    onehot[np.arange(tok.size), tok.ravel()] = 1
    G["E"] = onehot.T @ dx.reshape(-1, D_MODEL)
    loss = -(np.take_along_axis(lp, tgt[..., None], -1)[..., 0] * mask).sum() / cnt
    return float(loss), G


def repeated_sequences(rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """N_CTX + 1 tokens repeating a uniform random block of period p ~ U{6, …, 24}. Position i predicts token i + 1;
    the mask keeps i ≥ p, where the current token has occurred once already, p positions back. Returns
    (tok, tgt, mask, p)."""
    p = rng.integers(PERIODS[0], PERIODS[1] + 1, size=batch)
    base = rng.integers(V_TOK, size=(batch, PERIODS[1]))
    T = np.take_along_axis(base, np.arange(N_CTX + 1)[None, :] % p[:, None], 1)
    return T[:, :-1], T[:, 1:], np.arange(N_CTX)[None, :] >= p[:, None], p


def path_logits(P: Params, tok: np.ndarray) -> dict[tuple, np.ndarray]:
    """Thm 3: the logits as a sum over paths. A path is a chain of heads (l₁, h₁), …, (l_m, h_m) with l₁ < … < l_m;
    its term is A^{l_m h_m} ⋯ A^{l₁ h₁} x_0 W_OV^{l₁ h₁} ⋯ W_OV^{l_m h_m} U, with each A taken from the forward pass and
    W_OV = W_V W_O. The empty chain is the direct path x_0 U."""
    xs, _, caches = model_forward(P, tok)
    heads = [(l, h) for l in range(n_layers(P)) for h in range(N_HEADS)]
    out = {}
    for m in range(n_layers(P) + 1):
        for chain in itertools.combinations(heads, m):
            if any(a[0] == b[0] for a, b in zip(chain, chain[1:])):
                continue
            Y = xs[0]
            for l, h in chain:
                Y = caches[l][4][:, h] @ (Y @ P[f"Wv{l}"][h] @ P[f"Wo{l}"][h])
            out[chain] = Y @ P["U"]
    return out


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    source: str = "synthetic"  # or "filings": windows of the training-roster filings
    steps: int = 1000
    batch: int = 32
    lr: float = 0.01


SYNTH2 = ModelConfig(2)
SYNTH1 = ModelConfig(1)
FILING2 = ModelConfig(2, "filings", steps=800)


@lru_cache(maxsize=None)
def train_model(cfg: ModelConfig) -> Params:
    """Adam (Topic 5's constants) on fresh batches; one stream per (source, layers)."""
    rng = np.random.default_rng([SEED, 1430 + 10 * (cfg.source == "filings") + cfg.layers])
    P = init_model(rng, cfg.layers)
    m = {key: np.zeros_like(v) for key, v in P.items()}
    s = {key: np.zeros_like(v) for key, v in P.items()}
    for step in range(1, cfg.steps + 1):
        tok, tgt, mask = (repeated_sequences(rng, cfg.batch)[:3] if cfg.source == "synthetic"
                          else filing_windows(rng, cfg.batch))
        _, G = model_loss_grad(P, tok, tgt, mask)
        c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
        for key in P:
            m[key] = nlm.ADAM_B1 * m[key] + (1 - nlm.ADAM_B1) * G[key]
            s[key] = nlm.ADAM_B2 * s[key] + (1 - nlm.ADAM_B2) * G[key] ** 2
            P[key] -= cfg.lr * (m[key] / c1) / (np.sqrt(s[key] / c2) + nlm.ADAM_EPS)
    return P


TEST_N = 1000


@lru_cache(maxsize=1)
def repeated_test_set() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TEST_N fresh repeated sequences, shared by every model (paired comparisons)."""
    return repeated_sequences(np.random.default_rng([SEED, 1450]), TEST_N)


def head_scores(P: Params) -> dict[str, np.ndarray]:
    """On the test set: each head's mean attention to the previous position (over i ≥ 1) and to the position after
    the current token's previous occurrence, i − p + 1 (over the masked positions) — the induction target."""
    tok, _, mask, p = repeated_test_set()
    _, _, caches = model_forward(P, tok)
    i = np.arange(N_CTX)
    prev, ind = [], []
    for c in caches:
        A = c[4]
        prev.append(A[:, :, i[1:], i[1:] - 1].mean((0, 2)))
        j = np.clip(i[None, :] - p[:, None] + 1, 0, N_CTX - 1)
        a = np.take_along_axis(A, np.broadcast_to(j[:, None, :, None], (*A.shape[:3], 1)), -1)[..., 0]
        ind.append((a * mask[:, None]).sum((0, 2)) / mask.sum())
    return {"prev": np.array(prev), "induction": np.array(ind)}


def composition_scores(P: Params, kind: str = "k") -> np.ndarray:
    """Elhage et al.'s composition scores between a layer-0 head h and a layer-1 head h': ‖W_OV^{0h} M‖_F /
    (‖W_OV^{0h}‖_F ‖M‖_F) with M = W_K^{1h'} (K-composition), W_Q^{1h'} (Q), or W_V^{1h'} W_O^{1h'} (V)."""
    S = np.zeros((N_HEADS, N_HEADS))
    for h, h2 in itertools.product(range(N_HEADS), repeat=2):
        ov = P["Wv0"][h] @ P["Wo0"][h]
        M = {"k": P["Wk1"][h2], "q": P["Wq1"][h2], "v": P["Wv1"][h2] @ P["Wo1"][h2]}[kind]
        S[h, h2] = np.linalg.norm(ov @ M) / (np.linalg.norm(ov) * np.linalg.norm(M))
    return S


def repeated_metrics(P: Params, ablate_k: bool = False) -> dict[str, float]:
    tok, tgt, mask, _ = repeated_test_set()
    _, logits, _ = model_forward(P, tok, ablate_k)
    acc = ((logits.argmax(-1) == tgt) * mask).sum() / mask.sum()
    return {"loss": masked_xent(logits, tgt, mask), "acc": float(acc)}


def logit_lens(P: Params) -> np.ndarray:
    """The probability of the correct next token read from the stream after each layer, x_l U (no final LayerNorm in
    this model, so the lens is the model's own readout), averaged over the test set's masked positions."""
    tok, tgt, mask, _ = repeated_test_set()
    xs, _, _ = model_forward(P, tok)
    out = []
    for x in xs:
        p = np.exp(np.take_along_axis(nlm.log_softmax(x @ P["U"]), tgt[..., None], -1)[..., 0])
        out.append((p * mask).sum() / mask.sum())
    return np.array(out)


# --------------------------------------------------------------------------------------------
# 7. Finance case study: completing a company's name.
# --------------------------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z]+")


def symbol(word: str) -> int:
    """A word's symbol: CRC-32 of its lowercase form mod V_TOK (Topic 10's hashing)."""
    return zlib.crc32(word.lower().encode()) % V_TOK


@dataclass(frozen=True)
class NameDoc:
    ids: np.ndarray      # the document's words as symbols
    words: tuple         # the words, lowercased
    name: tuple          # for each word, (ORG surface text, index of the word within it), or None


def name_doc(doc) -> NameDoc:
    orgs = [(e.start, e.end, doc.text[e.start:e.end]) for e in doc.entities if e.type == "ORG"]
    words, name = [], []
    for m in _WORD.finditer(doc.text):
        words.append(m.group().lower())
        hit = next(((s, t) for s, e, t in orgs if s <= m.start() < e), None)
        name.append(None if hit is None else (hit[1], len(_WORD.findall(doc.text[hit[0]:m.start()]))))
    return NameDoc(np.array([symbol(w) for w in words]), tuple(words), tuple(name))


@lru_cache(maxsize=None)
def roster_docs(split: str) -> tuple[NameDoc, ...]:
    """train: generate_corpus(300, 0) over split_roster()'s training companies; seen: generate_corpus(100, 1) over the
    same companies; unseen: generate_corpus(100, 1) over the held-out companies, whose names never occur in train."""
    train, held = fc.split_roster()
    docs = {"train": lambda: fc.generate_corpus(300, 0, roster=train),
            "seen": lambda: fc.generate_corpus(100, 1, roster=train),
            "unseen": lambda: fc.generate_corpus(100, 1, roster=held)}[split]()
    return tuple(name_doc(d) for d in docs)


def filing_windows(rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N_CTX + 1 consecutive words of a uniformly chosen training filing; every position is a target."""
    D = roster_docs("train")
    T = np.empty((batch, N_CTX + 1), int)
    for b, i in enumerate(rng.integers(len(D), size=batch)):
        t = int(rng.integers(len(D[i].ids) - N_CTX))
        T[b] = D[i].ids[t:t + N_CTX + 1]
    return T[:, :-1], T[:, 1:], np.ones((batch, N_CTX), bool)


@lru_cache(maxsize=None)
def name_queries(split: str) -> dict:
    """Every word of a multi-word company name, except its last, whose name was mentioned in full earlier in the
    same filing within the N_CTX-word window ending at it. Returns windows (q, N_CTX), targets (the name's next
    word), and the (document, position) of each query."""
    W, Y, where = [], [], []
    for di, d in enumerate(roster_docs(split)):
        last: dict[str, int] = {}
        for i, tag in enumerate(d.name):
            if tag is None:
                continue
            surface, j = tag
            nw = len(_WORD.findall(surface))
            prev = last.get(surface)
            if j + 1 < nw and i >= N_CTX - 1 and prev is not None and prev >= i - N_CTX + 1:
                W.append(d.ids[i - N_CTX + 1:i + 1])
                Y.append(d.ids[i + 1])
                where.append((di, i))
            if j == nw - 1:
                last[surface] = i - nw + 1
    return {"W": np.array(W), "Y": np.array(Y), "where": tuple(where)}


@lru_cache(maxsize=1)
def bigram_table() -> np.ndarray:
    """Counts of symbol pairs in the training filings (Topic 3's bigram counts, over symbols)."""
    C = np.zeros((V_TOK, V_TOK))
    for d in roster_docs("train"):
        np.add.at(C, (d.ids[:-1], d.ids[1:]), 1)
    return C


READERS = ("bigram", "filings-direct", "filings-2", "induction-1", "induction-2")


def readers_predict(W: np.ndarray) -> dict[str, np.ndarray]:
    """Each reader's argmax next symbol after every window: the bigram table on its last symbol, the filing-trained
    model's direct path E[x]U and full model, and the repeated-sequence models with one and two layers."""
    F = train_model(FILING2)
    return {"bigram": bigram_table()[W[:, -1]].argmax(1),
            "filings-direct": (F["E"][W[:, -1]] @ F["U"]).argmax(1),
            "filings-2": model_forward(F, W)[1][:, -1].argmax(1),
            "induction-1": model_forward(train_model(SYNTH1), W)[1][:, -1].argmax(1),
            "induction-2": model_forward(train_model(SYNTH2), W)[1][:, -1].argmax(1)}


@lru_cache(maxsize=1)
def name_accuracy() -> dict[str, dict[str, float]]:
    out = {}
    for split in ("seen", "unseen"):
        Q = name_queries(split)
        out[split] = {k: float((v == Q["Y"]).mean()) for k, v in readers_predict(Q["W"]).items()}
    return out


def bias_offset_score(P: Params, l: int, h: int, deltas) -> np.ndarray:
    """The content-free part of head (l, h)'s score with the key Δ positions behind the query: ⟨R(i)b_q, R(i − Δ)b_k⟩/√dh
    = b_qᵀ R(−Δ) b_k/√dh — Topic 10's kernel over offsets, built from the biases."""
    bq, bk = P[f"bq{l}"][h], P[f"bk{l}"][h]
    return np.array([bq @ pe.rotate(bk[None], [-float(d)], ROPE)[0] for d in deltas]) / math.sqrt(D_HEAD)


# --------------------------------------------------------------------------------------------
# Summary and lab payload.
# --------------------------------------------------------------------------------------------


def _r(x, nd: int = 3):
    return ak._r(x, nd)


def _sig(x, n: int = 4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_sig(v, n) for v in x]
    return ak._sig(float(x), n)


@lru_cache(maxsize=1)
def summary() -> dict:
    s2, s1 = train_model(SYNTH2), train_model(SYNTH1)
    full, one, cut = repeated_metrics(s2), repeated_metrics(s1), repeated_metrics(s2, ablate_k=True)
    hs = head_scores(s2)
    acc = name_accuracy()
    pre = {L: gradient_profile("pre", L) for L in GRAD_DEPTHS}
    post = {L: gradient_profile("post", L) for L in GRAD_DEPTHS}
    out = {
        "collapse_spread_pure_b1": collapse_run(1.0, False)["spread"][-1],
        "collapse_spread_pure_b4": collapse_run(4.0, False)["spread"][-1],
        "collapse_diam_skip_b1": collapse_run(1.0, True)["diam"][-1],
        "induction_loss_2layer": full["loss"], "induction_acc_2layer": full["acc"],
        "induction_loss_1layer": one["loss"], "induction_acc_1layer": one["acc"],
        "induction_loss_k_cut": cut["loss"], "induction_acc_k_cut": cut["acc"],
        "prev_token_score": float(hs["prev"][0].max()), "induction_scores": [float(v) for v in hs["induction"][1]],
        "k_composition_max": float(composition_scores(s2, "k").max()),
        "q_composition_max": float(composition_scores(s2, "q").max()),
        "v_composition_max": float(composition_scores(s2, "v").max()),
        "logit_lens": [float(v) for v in logit_lens(s2)],
    }
    for L in GRAD_DEPTHS:
        out[f"pre_w2_last_L{L}"] = float(pre[L]["w2"][-1])
        out[f"post_w2_last_L{L}"] = float(post[L]["w2"][-1])
    for split in ("seen", "unseen"):
        for k, v in acc[split].items():
            out[f"names_{split}_{k}"] = v
    return out


LN_CLOUD = 24


@lru_cache(maxsize=1)
def ln_cloud() -> dict:
    """The lab's LayerNorm figure at d = 3: 24 points with random scales and offsets along 1, in the orthonormal basis
    e₁ = (1, −1, 0)/√2, e₂ = (1, 1, −2)/√6 of the plane 1^⊥ plus their coordinate along e₃ = 1/√3."""
    rng = np.random.default_rng([SEED, 1460])
    X = rng.normal(size=(LN_CLOUD, 3)) * rng.uniform(0.3, 2.5, (LN_CLOUD, 1)) + rng.normal(0, 2, (LN_CLOUD, 1))
    B = np.array([[1, -1, 0] / np.sqrt(2), [1, 1, -2] / np.sqrt(6), [1, 1, 1] / np.sqrt(3)])
    Y = layer_norm(X)
    return {"plane": X @ B[:2].T, "along": X @ B[2], "ln_plane": Y @ B[:2].T, "ln_along": Y @ B[2]}


@lru_cache(maxsize=1)
def lab_sequence() -> int:
    """The test sequence the lab shows: the first with period 11."""
    return int(np.flatnonzero(repeated_test_set()[3] == 11)[0])


def _sparse_attention(A: np.ndarray, floor: float = 0.02) -> list:
    i, j = np.nonzero(A >= floor)
    return [[int(a), int(b), _r(A[a, b], 2)] for a, b in zip(i, j)]


@lru_cache(maxsize=1)
def lab_filing() -> int:
    """The unseen-roster filing with the most name queries."""
    where = name_queries("unseen")["where"]
    return max(set(d for d, _ in where), key=lambda d: (sum(1 for x, _ in where if x == d), -d))


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    s2 = train_model(SYNTH2)
    hs = head_scores(s2)
    prev_h, ind_h = int(hs["prev"][0].argmax()), int(hs["induction"][1].argmax())
    tok, tgt, mask, p = repeated_test_set()
    e = lab_sequence()
    one = tok[e:e + 1]
    xs, logits, caches = model_forward(s2, one)
    paths = path_logits(s2, one)
    centered = {c: v[0] - v[0].mean(-1, keepdims=True) for c, v in paths.items()}
    contrib = {"+".join(f"{l}.{h}" for l, h in c) or "direct": _r(centered[c][np.arange(N_CTX), tgt[e]], 2)
               for c in paths}
    lens = [_r(np.exp(np.take_along_axis(nlm.log_softmax(x[0] @ s2["U"]), tgt[e][:, None], -1)[:, 0]), 3)
            for x in xs]
    cloud = ln_cloud()
    acc = name_accuracy()
    Q = name_queries("unseen")
    fi = lab_filing()
    doc = roster_docs("unseen")[fi]
    preds = readers_predict(Q["W"])
    queries = [[int(i), *[bool(preds[k][q] == Q["Y"][q]) for k in READERS]]
               for q, (d, i) in enumerate(Q["where"]) if d == fi]
    return {
        "collapse": {
            "betas": list(COLLAPSE_BETAS), "depth": COLLAPSE_DEPTH,
            "runs": {f"{b}|{'skip' if r else 'pure'}": {
                "X": [_sig(X.ravel(), 4) for X in collapse_run(b, r)["X"]],
                "diam": _sig(collapse_run(b, r)["diam"], 4), "spread": _r(collapse_run(b, r)["spread"], 3),
                "tau": _r(collapse_run(b, r)["tau"], 3), "bound": _sig(collapse_run(b, r)["bound"], 4)}
                for b in COLLAPSE_BETAS for r in (False, True)}},
        "layernorm": {k: _r(np.asarray(v).ravel(), 3) for k, v in cloud.items()},
        "gradients": {s: {str(L): {"w2": _sig(gradient_profile(s, L)["w2"], 3),
                                   "norm": _sig(gradient_profile(s, L)["norm"], 3)} for L in GRAD_DEPTHS}
                      for s in SCHEMES},
        "induction": {
            "tokens": tok[e].tolist(), "targets": tgt[e].tolist(), "mask": mask[e].tolist(), "period": int(p[e]),
            "prev_head": [0, prev_h], "induction_head": [1, ind_h],
            "attention": {"prev": _sparse_attention(caches[0][4][0, prev_h]),
                          "induction": _sparse_attention(caches[1][4][0, ind_h])},
            "paths": contrib, "lens": lens,
            "lens_mean": _r(logit_lens(s2), 3),
            "metrics": {k: {m: _r(v, 3) for m, v in repeated_metrics(*a).items()}
                        for k, a in (("two", (s2,)), ("one", (train_model(SYNTH1),)), ("cut", (s2, True)))},
            "composition": {k: _r(composition_scores(s2, k).ravel(), 3) for k in "kqv"},
        },
        "names": {
            "readers": list(READERS), "accuracy": {s: {k: _r(v, 3) for k, v in acc[s].items()} for s in acc},
            "counts": {s: len(name_queries(s)["Y"]) for s in ("seen", "unseen")},
            "words": list(doc.words), "queries": queries,
        },
    }


# --------------------------------------------------------------------------------------------
# The harness: one test per claim on the page.
# --------------------------------------------------------------------------------------------


def _spot_grad_check(loss, P: Params, G: Params, rng: np.random.Generator, k: int = 12, eps: float = 1e-6) -> float:
    """Topic 5's central difference at k random coordinates of every parameter; returns the worst error."""
    worst = 0.0
    for key, v in P.items():
        for flat in rng.choice(v.size, size=min(k, v.size), replace=False):
            idx = np.unravel_index(flat, v.shape)
            num = nlm.central_diff_grad(lambda t: loss({**P, key: v + t.reshape(()) * _unit(v.shape, idx)}),
                                        np.zeros(1), eps)[0]
            worst = max(worst, abs(num - G[key][idx]))
    return worst


def _unit(shape, idx) -> np.ndarray:
    U = np.zeros(shape)
    U[idx] = 1
    return U


def test_multi_head_is_a_sum_of_smoothers():
    """Prop 1: multi-head attention without positions or biases is Σ_h Topic 9's attention(XW_Q^h, XW_K^h, XW_V^h) W_O^h;
    the MLP is the key–value sum of (ii)."""
    rng = np.random.default_rng([SEED, 1401])
    L = init_mha(rng, 12, 3, 4)
    X = rng.normal(size=(1, 7, 12))
    out, _ = mha_forward(L, X)
    ref = sum(ak.attention(X[0] @ L["Wq"][h], X[0] @ L["Wk"][h], X[0] @ L["Wv"][h], causal=True)[0] @ L["Wo"][h]
              for h in range(3))
    assert np.abs(out[0] - ref).max() < 1e-12
    blk = init_block(rng, 12, 3, 4, 20)
    blk["mlp"]["b1"] = rng.normal(size=20)
    m, _ = mlp_forward(blk["mlp"], X)
    assert all(np.abs(m[0, t] - key_value_memory(blk["mlp"], X[0, t])).max() < 1e-12 for t in range(7))


def test_layer_norm_geometry():
    """Prop 3 and Cor 1: LN(x) = √d·Px/‖Px‖ lies on 1^⊥ at radius √d; it ignores scale and shifts along 1, and is
    idempotent; RMSNorm agrees with it on centered inputs; the Jacobian is (√d/‖Px‖)(P − uuᵀ) (central differences),
    kills 1 and x, and has singular values √d/‖Px‖ (d − 2 times) and 0 (twice)."""
    rng = np.random.default_rng([SEED, 1402])
    for d in (3, 8, 64):
        X = rng.normal(size=(50, d)) * 3 + 1
        Y = layer_norm(X)
        assert np.abs(Y - ln_sphere(X)).max() < 1e-12
        assert np.abs(Y.sum(1)).max() < 1e-10 and np.abs(np.linalg.norm(Y, axis=1) - math.sqrt(d)).max() < 1e-12
        c, s = rng.uniform(0.1, 10, (50, 1)), rng.normal(size=(50, 1))
        assert np.abs(layer_norm(c * X + s) - Y).max() < 1e-10 and np.abs(layer_norm(Y) - Y).max() < 1e-12
        assert np.abs(rms_norm(center(X)) - Y).max() < 1e-12
        x = X[0]
        J = ln_jacobian(x)
        num = np.stack([nlm.central_diff_grad(lambda t, i=i: layer_norm(t)[i], x) for i in range(d)])
        assert np.abs(J - num).max() < 1e-7 and np.abs(J - J.T).max() < 1e-15
        assert np.abs(J @ np.ones(d)).max() < 1e-12 and np.abs(J @ x).max() < 1e-10
        sv = np.linalg.svd(J, compute_uv=False)
        r = np.linalg.norm(center(x))
        assert np.allclose(sv[:d - 2], math.sqrt(d) / r) and np.abs(sv[d - 2:]).max() < 1e-12
        dY = rng.normal(size=(1, d))
        assert np.abs(ln_backward(dY, Y[:1], np.array([[r]])) - dY @ J).max() < 1e-12


def test_thin_shell():
    """formalRAG's thin shell: for x ~ N(0, I_d), ‖Px‖ ≈ √d, so LayerNorm barely moves a Gaussian vector at large d
    (median ‖LN(x) − Px‖/‖Px‖ below 0.025 at d = 512) and moves it far more at d = 8."""
    rng = np.random.default_rng([SEED, 1403])
    med = {}
    for d in (8, 512):
        X = rng.normal(size=(2000, d))
        C = center(X)
        med[d] = float(np.median(np.linalg.norm(layer_norm(X) - C, axis=1) / np.linalg.norm(C, axis=1)))
    assert med[512] < 0.025 < 0.1 < med[8], med


def test_dobrushin_contraction():
    """Thm 1 (i)–(ii): for row-stochastic A, diam(AX) ≤ τ(A) diam(X) in every norm, τ's two forms agree, and a softmax
    whose rows have score range ≤ r has τ ≤ 1 − e^{−r}."""
    rng = np.random.default_rng([SEED, 1404])
    for _ in range(300):
        n, d = int(rng.integers(2, 12)), int(rng.integers(1, 5))
        A = rng.dirichlet(np.full(n, rng.uniform(0.1, 3)), size=n)
        X = rng.normal(size=(n, d))
        tau = dobrushin(A)
        half_l1 = 0.5 * np.abs(A[:, None, :] - A[None, :, :]).sum(-1).max()
        assert abs(tau - half_l1) < 1e-12
        for ord_ in (1, 2, np.inf):
            assert diameter(A @ X, ord_) <= tau * diameter(X, ord_) + 1e-12
        S = rng.normal(0, rng.uniform(0.1, 3), (n, n))
        r = float((S.max(1) - S.min(1)).max())
        assert dobrushin(ak._softmax(S)) <= 1 - math.exp(-r) + 1e-12


def test_collapse_and_the_skip():
    """Thm 1 (iii) and Prop 2 on the lab's cloud and on random value maps: pure attention's diameter stays under
    ∏τ_l‖W_l‖·diam(X₀), and reaches ~0 within 12 layers whenever β ≤ 2; with the skip it never falls below
    ∏(1 − τ_l‖W_l‖)·diam(X₀), and in the lab's runs (W = I) it never shrinks at all (to rounding). A narrow kernel (β = 4) keeps the
    pure stack apart (τ > 0.97, spread > 0.5): clusters, where the bound says nothing. And the skip saves differences,
    not proportions: at β = 0.25 the diameter more than triples while the relative spread falls below 0.01."""
    for b in COLLAPSE_BETAS:
        pure, skip = collapse_run(b, False), collapse_run(b, True)
        assert all(d <= bd * (1 + 1e-9) + 1e-12 for d, bd in zip(pure["diam"], pure["bound"]))
        dm = np.array(skip["diam"])
        assert np.all(dm[1:] >= dm[:-1] * (1 - 1e-12))
        assert all(dm[l + 1] >= (1 - skip["tau"][l]) * dm[l] * (1 - 1e-12) for l in range(COLLAPSE_DEPTH))
        if b <= 2:
            assert pure["diam"][-1] < 1e-10, b
    wide = collapse_run(0.25, True)  # the skip keeps the differences, not their share: the consensus grows faster
    assert wide["diam"][-1] > 3 * wide["diam"][0] and wide["spread"][-1] < 0.01
    narrow = collapse_run(4.0, False)
    assert min(narrow["tau"]) > 0.97 and narrow["spread"][-1] > 0.5
    rng = np.random.default_rng([SEED, 1405])
    for _ in range(100):
        X0 = rng.normal(size=(8, 4))
        W = rng.normal(size=(4, 4)) * rng.uniform(0.05, 0.5) / 2
        w = np.linalg.norm(W, 2)
        beta = float(rng.uniform(0.1, 3))
        for residual in (False, True):
            Xs, As = smoothing_stack(X0, beta, 6, residual, W)
            for l, A in enumerate(As):
                tau = dobrushin(A)
                if residual:
                    assert diameter(Xs[l + 1]) >= (1 - tau * w) * diameter(Xs[l]) - 1e-12
                else:
                    assert diameter(Xs[l + 1]) <= tau * w * diameter(Xs[l]) + 1e-12


def test_jacobian_paths():
    """Thm 2: the Jacobian of a residual stack x ← x + f_k(x) is ∏(I + J_k) by the chain rule (central differences of
    the whole stack), and that product is the sum over the 2^L subsets of layers — the empty subset is I."""
    rng = np.random.default_rng([SEED, 1406])
    d, L = 5, 4
    Ws = [rng.normal(0, 0.6, (d, d)) for _ in range(L)]
    f = lambda x, W: np.tanh(x @ W)  # noqa: E731

    def run(x):
        for W in Ws:
            x = x + f(x, W)
        return x

    x0 = rng.normal(size=d)
    x, Js = x0, []
    for W in Ws:
        J = (1 - np.tanh(x @ W) ** 2)[:, None] * W.T
        assert np.abs(J - np.stack([nlm.central_diff_grad(lambda t, i=i: f(t, W)[i], x) for i in range(d)])).max() < 1e-8
        Js.append(J)
        x = x + f(x, W)
    prod = np.eye(d)
    for J in Js:
        prod = (np.eye(d) + J) @ prod
    whole = np.stack([nlm.central_diff_grad(lambda t, i=i: run(t)[i], x0) for i in range(d)])
    assert np.abs(prod - whole).max() < 1e-7
    assert np.abs(jacobian_paths(Js) - prod).max() < 1e-12


def test_gradients():
    """The hand-written gradients against Topic 5's central differences: the attention-only RoPE model (at random
    coordinates of every parameter), and the pre-LN and post-LN stacks (every coordinate)."""
    rng = np.random.default_rng([SEED, 1407])
    P = init_model(rng, 2)
    tok, tgt, mask, _ = repeated_sequences(rng, 2)
    loss = lambda Q: model_loss_grad(Q, tok, tgt, mask)[0]  # noqa: E731
    _, G = model_loss_grad(P, tok, tgt, mask)
    assert _spot_grad_check(loss, P, G, rng) < 1e-7
    for scheme in SCHEMES:
        blocks = [init_block(rng, 8, 2, 4, 12) for _ in range(3)]
        for blk in blocks:
            blk["attn"]["bq"], blk["attn"]["bk"] = rng.normal(size=(2, 4)), rng.normal(size=(2, 4))
            blk["mlp"]["b1"] = rng.normal(0, 0.1, 12)
        X, R = rng.normal(size=(1, 5, 8)), rng.normal(size=(1, 5, 8))
        _, grads, dstream = stack_loss_grad(blocks, X, R, scheme)
        for l, blk in enumerate(blocks):
            for part in ("attn", "mlp"):
                for key, v in blk[part].items():
                    def lossf(t, l=l, part=part, key=key):
                        bs = [{p: dict(q) for p, q in b.items()} for b in blocks]
                        bs[l][part][key] = t
                        return stack_loss_grad(bs, X, R, scheme)[0]
                    num = nlm.central_diff_grad(lossf, v)
                    assert np.abs(num - grads[l][part][key]).max() < 1e-7, (scheme, l, part, key)
        num = nlm.central_diff_grad(lambda t: stack_loss_grad(blocks, t, R, scheme)[0], X)
        assert np.abs(num - dstream[0]).max() < 1e-7, scheme


def test_gradient_scale_at_init():
    """Xiong et al.'s Theorem 1, measured: the last block's MLP-output gradient shrinks like L^{−1/2} under pre-LN
    (L = 6 → 24 divides it by 1.6–2.4, √4 = 2) and does not under post-LN (within 25%), where at L = 24 it is over 3×
    the pre-LN one. The pre-LN stream grows (‖x_L‖² ≈ linear in L, about 3× ‖x₀‖ at L = 12); post-LN pins it at √d."""
    pre = {L: gradient_profile("pre", L) for L in GRAD_DEPTHS}
    post = {L: gradient_profile("post", L) for L in GRAD_DEPTHS}
    ratio = pre[6]["w2"][-1] / pre[24]["w2"][-1]
    assert 1.6 < ratio < 2.4, ratio
    lasts = [post[L]["w2"][-1] for L in GRAD_DEPTHS]
    assert max(lasts) / min(lasts) < 1.25, lasts
    assert post[24]["w2"][-1] > 3 * pre[24]["w2"][-1]
    n2 = pre[24]["norm"] ** 2
    slope = np.polyfit(np.arange(25), n2, 1)
    resid = n2 - np.polyval(slope, np.arange(25))
    assert 1 - (resid**2).sum() / ((n2 - n2.mean()) ** 2).sum() > 0.98 and pre[12]["norm"][-1] > 3 * pre[12]["norm"][0]
    assert all(np.allclose(post[L]["norm"][1:], math.sqrt(GRAD_D)) for L in GRAD_DEPTHS)


def test_path_expansion():
    """Thm 3: the sum of the 1 + 2H + H² path terms equals the forward pass's logits (to 1e-10), for the trained
    induction model and for a random 3-layer model (1 + 3H + 3H² + H³ terms)."""
    tok = repeated_test_set()[0][:40]
    P = train_model(SYNTH2)
    paths = path_logits(P, tok)
    assert len(paths) == 1 + 2 * N_HEADS + N_HEADS**2
    assert np.abs(sum(paths.values()) - model_forward(P, tok)[1]).max() < 1e-10
    R3 = init_model(np.random.default_rng([SEED, 1408]), 3)
    paths = path_logits(R3, tok[:5])
    assert len(paths) == (1 + N_HEADS) ** 3
    assert np.abs(sum(paths.values()) - model_forward(R3, tok[:5])[1]).max() < 1e-10


def test_orthogonal_heads_do_not_talk():
    """Prop 4: if every layer-0 head writes into a subspace that the layer-1 keys do not read (W_OV^{0h} W_K^{1h'} = 0),
    the K-composition score is 0 and computing layer 1's keys from x₀ instead of x₁ changes nothing."""
    rng = np.random.default_rng([SEED, 1409])
    P = init_model(rng, 2)
    half = D_MODEL // 2
    P["Wo0"][:, :, half:] = 0
    P["Wk1"][:, :half, :] = 0
    assert composition_scores(P, "k").max() < 1e-15
    tok = repeated_sequences(rng, 4)[0]
    a = model_forward(P, tok)[2][1][4]
    b = model_forward(P, tok, ablate_k=True)[2][1][4]
    assert np.abs(a - b).max() < 1e-12


def test_induction_heads():
    """§6: the 2-layer model reaches loss below 0.25 nats and accuracy above 0.95 on the repeated half; the 1-layer
    model stays above 4 nats and below 0.15; computing layer 1's keys from x₀ (no K-composition) sends the 2-layer model
    above the 1-layer model's loss. A layer-0 head puts > 0.8 of its weight on the previous token, with its content-free
    RoPE score (Topic 10's kernel over offsets) peaking at offset 1; both layer-1 heads put > 0.5 on the induction
    target. K-composition dominates Q and V for every pair and exceeds every K-score of the same model at its
    initialization (train_model's first draw)."""
    s2, s1 = train_model(SYNTH2), train_model(SYNTH1)
    full, one, cut = repeated_metrics(s2), repeated_metrics(s1), repeated_metrics(s2, ablate_k=True)
    assert full["loss"] < 0.25 and full["acc"] > 0.95, full
    assert one["loss"] > 4 and one["acc"] < 0.15, one
    assert cut["loss"] > one["loss"] and cut["acc"] < 0.15, cut
    hs = head_scores(s2)
    h = int(hs["prev"][0].argmax())
    assert hs["prev"][0, h] > 0.8 and np.all(hs["induction"][1] > 0.5), hs
    assert int(np.argmax(bias_offset_score(s2, 0, h, range(N_CTX)))) == 1
    K, Qs, Vs = (composition_scores(s2, k) for k in "kqv")
    assert np.all(K > Qs) and np.all(K > Vs)
    init_k = composition_scores(init_model(np.random.default_rng([SEED, 1430 + 2]), 2), "k").max()
    assert K.min() > init_k, (K, init_k)


def test_one_layer_keys():
    """Prop 5: in the one-layer model the score s(i, j) is a function of (x_i, i, x_j, j) alone, so changing the token
    at j − 1 leaves it exactly unchanged; in the two-layer model the previous-token head writes x_{j−1} into the stream
    at j, and the same change moves layer 1's score by over 2 nats on average."""
    tok = repeated_test_set()[0][:64].copy()
    i, j = N_CTX - 1, N_CTX - 12
    alt = tok.copy()
    alt[:, j - 1] = (alt[:, j - 1] + 1 + np.arange(64) % (V_TOK - 1)) % V_TOK

    def score(P, T):
        c = model_forward(P, T)[2][-1]
        return (c[1][:, :, i] * c[2][:, :, j]).sum(-1) / math.sqrt(D_HEAD)

    s1 = train_model(SYNTH1)
    assert np.abs(score(s1, tok) - score(s1, alt)).max() < 1e-12
    s2 = train_model(SYNTH2)
    assert np.abs(score(s2, tok) - score(s2, alt)).mean() > 2.0


def test_logit_lens():
    """§6: read through U, the stream after layer 0 and after layer 1 gives the correct token about chance (< 2/V);
    after layer 2, over 0.9."""
    lens = logit_lens(train_model(SYNTH2))
    assert lens[0] < 2 / V_TOK and lens[1] < 2 / V_TOK and lens[2] > 0.9, lens


def test_composed_path_carries_the_copy():
    """Thm 3 on the trained model: at the masked positions, the composed paths (layer-0 head → layer-1 head) and the
    layer-1 heads on the direct path together carry over 90% of the correct token's centered logit; the direct path
    and the layer-0 heads alone rank it at chance."""
    P = train_model(SYNTH2)
    tok, tgt, mask, _ = repeated_test_set()
    tok, tgt, mask = tok[:200], tgt[:200], mask[:200]
    paths = path_logits(P, tok)

    def centered(v):
        c = v - v.mean(-1, keepdims=True)
        return np.take_along_axis(c, tgt[..., None], -1)[..., 0][mask]

    total = sum(centered(v) for v in paths.values())
    layer1 = sum(centered(v) for c, v in paths.items() if c and c[-1][0] == 1)
    assert np.median(layer1 / total) > 0.9
    early = sum(v for c, v in paths.items() if not c or c[-1][0] == 0)
    acc = ((early.argmax(-1) == tgt) & mask).sum() / mask.sum()
    assert acc < 0.05, acc


def test_finance_names():
    """§7: on names never seen in training, the bigram table, the filing-trained model, and its direct path complete
    none (≤ 0.01); the 2-layer induction model, which never saw a filing, completes over 0.7 of both the seen and the
    unseen names; the 1-layer induction model stays under 0.1 on both. On seen
    names the filing-trained model beats the bigram table and its own direct path completes over 0.65."""
    acc = name_accuracy()
    for k in ("bigram", "filings-direct", "filings-2"):
        assert acc["unseen"][k] <= 0.01, (k, acc)
    assert acc["unseen"]["induction-2"] > 0.7 and acc["seen"]["induction-2"] > 0.7
    assert acc["unseen"]["induction-1"] < 0.1 and acc["seen"]["induction-1"] < 0.1
    assert acc["seen"]["filings-2"] > acc["seen"]["bigram"] and acc["seen"]["filings-direct"] > 0.65
    assert min(len(name_queries(s)["Y"]) for s in ("seen", "unseen")) > 200


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"
    train, held = fc.split_roster()
    assert not set(train) & set(held)


def test_viz_payload():
    """The lab's data: 5 temperatures × 2 stacks × 13 clouds of 10 points; 24 LayerNorm points; the gradient
    profiles; the lab sequence's paths sum to its centered logits; the filing's queries; the file stays under 60 kB."""
    p = viz_payload()
    runs = p["collapse"]["runs"]
    assert len(runs) == 10 and all(len(r["X"]) == COLLAPSE_DEPTH + 1 for r in runs.values())
    assert all(len(x) == 2 * COLLAPSE_N for r in runs.values() for x in r["X"])
    assert len(p["layernorm"]["plane"]) == 2 * LN_CLOUD
    assert np.allclose(np.hypot(*np.reshape(p["layernorm"]["ln_plane"], (-1, 2)).T), math.sqrt(3), atol=2e-3)
    ind = p["induction"]
    assert ind["period"] == 11 and len(ind["paths"]) == 1 + 2 * N_HEADS + N_HEADS**2
    assert all(len(v) == N_CTX for v in ind["paths"].values()) and len(ind["lens"]) == 3
    assert {q[0] for q in p["names"]["queries"]} <= set(range(len(p["names"]["words"])))
    assert len(json.dumps(p, separators=(",", ":"))) < 60_000


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "residual-stream-lab.json"


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
    print("\nSummary (attention-only RoPE models: V = 256, n = 48, d = 32, 2 heads of 16; repeated sequences of period"
          " 6–24; filings: generate_corpus(300, 0) over split_roster()'s training companies):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.6f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
