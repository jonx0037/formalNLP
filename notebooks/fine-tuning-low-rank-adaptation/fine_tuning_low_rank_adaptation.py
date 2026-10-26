"""Fine-tuning and low-rank adaptation — the reference implementation for formalNLP's twentieth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The model is Topic 12's filing encoder (V = 879, n = 32, d = 64, two pre-LN blocks, 158,447
parameters, trained from scratch by masked-LM in ~21 s); the task is Topic 19's aspect sentiment (a sentence, a gold
aspect span, a polarity in {−, 0, +}), read from the mean of the encoder's final states over the aspect span. Three
ways to adapt the encoder: read it (a probe on frozen features), rewrite it (full fine-tuning), or nudge each weight
matrix by a rank-r product (LoRA). The sections follow the page:

1. One weight, two factors (Def 1, Prop 1). W = W₀ + s·AB with A random and B = 0: rank ≤ r, r(k + m) parameters,
   merged at inference, and factors identified only up to (AS, S⁻¹B).
2. The first step (Thm 1). ∂L/∂A = sGBᵀ, ∂L/∂B = sAᵀG, so at B = 0 the first SGD step is the full step pushed
   through A₀A₀ᵀ, whose mean is rσ²I: the step scales as s²r (SGD) and as s√r (Adam's sign step), and is
   rank-stable only when s ∝ 1/√r (rsLoRA).
3. What the factors conserve (Thm 2). Gradient flow keeps AᵀA − BBᵀ = A₀ᵀA₀; GD drifts by exactly η²·(…).
4. Is the update low-rank? (Prop 2.) Full fine-tuning's ΔW against a Gaussian's spectrum edge, and truncated merges.
5. What fine-tuning leaves alone (Thm 3, Prop 3). SGD moves a linear layer only along its training inputs; unseen
   tokens' embeddings never move; LP-FT; L2-SP as a Gaussian prior at θ₀; forgetting as held-out masked-LM loss.
6. How many directions a task needs (Prop 4): a count-sketch subspace step is unbiased; d₉₀, pretrained vs random.
7. Fine-tuning as a kernel (Prop 5, Cor 1): linearized GD is kernel GD; LoRA's kernel is the A₀-sketched NTK.
8. Finance case study: reading negation (a shortcut, then counterfactual augmentation), aspect reading aggregated to
   document tone, and document-level fine-tuning as the honest negative.

Runtime: ≈ 2¼ min (the topic's experiment grid; the README's "well under 60 s" is waived for this topic only — the
pretrain alone is ~21 s, each of the 31 fine-tunes ~2.4 s with its evaluation, and the headline methods need three
seeds).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/fine-tuning-low-rank-adaptation/fine_tuning_low_rank_adaptation.py
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "contextual-embeddings-masked-lm"))
sys.path.insert(0, str(_HERE.parent / "aspect-based-sentiment"))
import aspect_based_sentiment as ab  # noqa: E402 — module objects: their test_* stay out of this harness
import contextual_embeddings_masked_lm as cm  # noqa: E402

tb, nlm = cm.tb, cm.nlm
ltc, fc = ab.ltc, ab.fc

# ONE stream per random-instance claim and per run, each from default_rng([SEED, k, …]); k ≥ 3300 keeps this
# module's streams apart from Topics 1–19.
SEED = cm.SEED
_r = ltc._r


def _rng(*k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, *k])


# --------------------------------------------------------------------------------------------
# 0. The encoder, and its backward pass from the final states.
# --------------------------------------------------------------------------------------------

D_MODEL = cm.FILING_MLM.d
N_CTX = cm.FILING_MLM.n
MATRICES = (("attn", "Wq"), ("attn", "Wk"), ("attn", "Wv"), ("attn", "Wo"), ("mlp", "W1"), ("mlp", "W2"))


def pretrained() -> cm.Trained:
    """Topic 12's filing encoder (cached there; ~21 s). Never mutate it: every run deep-copies."""
    return cm.train_mlm(cm.FILING_MLM)


@lru_cache(maxsize=1)
def random_twin() -> tuple[dict, list]:
    """The same architecture at Topic 12's initialization, never pretrained: the control for what pretraining buys."""
    return cm.init_encoder(_rng(3300), cm.filing_vocab()[1] + 1, cm.FILING_MLM)


def flat(key: str, W: np.ndarray) -> np.ndarray:
    """A block weight as the matrix of the map x ↦ xW: per-head Wq/Wk/Wv (H, d, dh) → (d, H·dh) (heads side by side),
    Wo (H, dh, d) → (H·dh, d); the MLP's W1, W2 are already matrices."""
    if key in ("Wq", "Wk", "Wv"):
        return W.transpose(1, 0, 2).reshape(W.shape[1], -1)
    return W.reshape(-1, W.shape[-1]) if key == "Wo" else W


def unflat(key: str, M: np.ndarray, shape: tuple) -> np.ndarray:
    if key in ("Wq", "Wk", "Wv"):
        H, d, dh = shape
        return M.reshape(d, H, dh).transpose(1, 0, 2)
    return M.reshape(shape)


def _heads(Y: np.ndarray, B: int, n: int, H: int) -> np.ndarray:
    return Y.reshape(B, n, H, -1).transpose(0, 2, 1, 3)


def _unheads(Y: np.ndarray) -> np.ndarray:
    return Y.transpose(0, 2, 1, 3).reshape(-1, Y.shape[1] * Y.shape[3])


def mha_forward(L: dict, X: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Topic 11's bidirectional mha_forward with every projection one matmul over the flattened batch (its broadcast
    (B, 1, n, d) @ (1, H, d, dh) products were 40% of a fine-tuning step); equal to it, test_fast_attention."""
    B, n, d = X.shape
    H, _, dh = L["Wq"].shape
    Xf = X.reshape(-1, d)
    q = _heads(Xf @ flat("Wq", L["Wq"]), B, n, H) + L["bq"][None, :, None, :]
    k = _heads(Xf @ flat("Wk", L["Wk"]), B, n, H) + L["bk"][None, :, None, :]
    v = _heads(Xf @ flat("Wv", L["Wv"]), B, n, H)
    A = tb.ak._softmax(q @ np.swapaxes(k, -1, -2) / math.sqrt(dh))
    o = A @ v
    return (_unheads(o) @ flat("Wo", L["Wo"])).reshape(B, n, d), (X, q, k, v, A, o, None)


def mha_backward(L: dict, dOut: np.ndarray, cache: tuple) -> tuple[np.ndarray, dict]:
    """Topic 11's mha_backward (no rotation) as flattened matmuls; equal to it, test_fast_attention."""
    X, q, k, v, A, o, _ = cache
    B, n, d = X.shape
    H, _, dh = L["Wq"].shape
    dOf, Xf = dOut.reshape(-1, d), X.reshape(-1, d)
    G = {"Wo": unflat("Wo", _unheads(o).T @ dOf, L["Wo"].shape)}
    do = _heads(dOf @ flat("Wo", L["Wo"]).T, B, n, H)
    da = do @ np.swapaxes(v, -1, -2)
    dv = np.swapaxes(A, -1, -2) @ do
    dS = A * (da - (A * da).sum(-1, keepdims=True)) / math.sqrt(dh)
    dq, dk = dS @ k, np.swapaxes(dS, -1, -2) @ q
    dX = np.zeros_like(Xf)
    for key, dY in (("Wq", dq), ("Wk", dk), ("Wv", dv)):
        Yf = _unheads(dY)
        G[key] = unflat(key, Xf.T @ Yf, L[key].shape)
        dX += Yf @ flat(key, L[key]).T
    G["bq"], G["bk"] = dq.sum((0, 2)), dk.sum((0, 2))
    return dX.reshape(B, n, d), G


def encode(P: dict, blocks: list, tok: np.ndarray) -> tuple[list[np.ndarray], list]:
    """Topic 12's encode (x₀ = E[tok] + P, Topic 11's pre-LN blocks) with the fast attention; equal to cm.encode."""
    xs, caches = [P["E"][tok] + P["P"][None, : tok.shape[1]]], []
    for blk in blocks:
        x = xs[-1]
        y, r1 = tb._ln_forward(x)
        a, ca = mha_forward(blk["attn"], y)
        x = x + a
        z, r2 = tb._ln_forward(x)
        m, cmm = tb.mlp_forward(blk["mlp"], z)
        caches.append((y, r1, ca, z, r2, cmm))
        xs.append(x + m)
    return xs, caches


def final_states(P: dict, blocks: list, tok: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """LN(x_L) (B, n, d), with the LayerNorm radius and the block caches for the backward pass."""
    xs, caches = encode(P, blocks, tok)
    out, r = tb._ln_forward(xs[-1])
    return out, r, caches


def encoder_backward(P: dict, blocks: list, tok: np.ndarray, out: np.ndarray, r: np.ndarray, caches: list,
                     dout: np.ndarray) -> tuple[dict, list]:
    """The gradient of any loss through the encoder, given dL/d LN(x_L): Topic 12's mlm_loss_grad backward loop
    (lifted, test_encoder_backward_matches_topic12). Returns ({E, P}, per-block {attn, mlp} gradients); G["P"] is
    padded to all n_ctx positions."""
    gx = tb.ln_backward(dout, out, r)
    grads = []
    for blk, (y, r1, ca, z, r2, cmm) in zip(reversed(blocks), reversed(caches)):
        dz, Gm = cm.mlp_backward(blk["mlp"], gx, cmm)
        gx = gx + tb.ln_backward(dz, z, r2)
        dy, Ga = mha_backward(blk["attn"], gx, ca)
        gx = gx + tb.ln_backward(dy, y, r1)
        grads.append({"attn": Ga, "mlp": Gm})
    GP = np.zeros_like(P["P"])
    GP[: tok.shape[1]] = gx.sum(0)
    N, d = tok.size, out.shape[-1]
    GE = sparse.csc_matrix((np.ones(N, out.dtype), tok.ravel(), np.arange(N + 1)),
                           shape=(P["E"].shape[0], N)) @ gx.reshape(-1, d)
    return {"E": np.asarray(GE), "P": GP}, grads[::-1]


# --------------------------------------------------------------------------------------------
# 1–3. Def 1, Prop 1, Thm 1, Thm 2: the algebra of a factored update.
# --------------------------------------------------------------------------------------------

ALPHA = 8.0


def lora_scale(r: int, scale: str, alpha: float = ALPHA) -> float:
    """s = α/r (Hu et al.) or α/√r (rsLoRA, Kalajdzievski)."""
    return alpha / r if scale == "lin" else alpha / math.sqrt(r)


def lora_init(rng: np.random.Generator, k: int, m: int, r: int, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Def 1: A₀ (k, r) i.i.d. N(0, σ²) with σ² = 1/k, B₀ = 0 (r, m), so ΔW₀ = 0."""
    return rng.normal(0, 1 / math.sqrt(k), (k, r)).astype(dtype), np.zeros((r, m), dtype)


def lora_grads(G: np.ndarray, A: np.ndarray, B: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Thm 1(a): with W = W₀ + s·AB and G = ∂L/∂W, ∂L/∂A = s·G Bᵀ and ∂L/∂B = s·Aᵀ G."""
    return s * G @ B.T, s * A.T @ G


def first_step(G: np.ndarray, A0: np.ndarray, s: float, eta: float, optimizer: str) -> np.ndarray:
    """Thm 1(b)/(d): the update ΔW₁ = s·A₀B₁ after one step from B₀ = 0. SGD: B₁ = −η·sAᵀG, so ΔW₁ = −ηs²A₀A₀ᵀG.
    Adam's first step is the sign step B₁ = −η·sgn(sAᵀG) (m̂/√v̂ = g/|g|), so ΔW₁ = −ηs·A₀ sgn(A₀ᵀG)."""
    if optimizer == "sgd":
        return -eta * s * s * A0 @ (A0.T @ G)
    return -eta * s * A0 @ np.sign(A0.T @ G)


def sketch_fluctuation(k: int, r: int, sigma2: float) -> float:
    """Thm 1(c): E‖A₀A₀ᵀ − rσ²I‖_F² = rσ⁴k(k + 1) for A₀ (k × r) i.i.d. N(0, σ²), so the first SGD step's relative
    deviation from its mean is √((k + 1)/r) in this norm: the sketch is noisy until r is comparable to k."""
    return r * sigma2**2 * k * (k + 1)


def balance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Thm 2's conserved quantity AᵀA − BBᵀ (r × r)."""
    return A.T @ A - B @ B.T


def gd_drift(A: np.ndarray, B: np.ndarray, G: np.ndarray, s: float, eta: float) -> np.ndarray:
    """Thm 2(b): one GD step on (A, B) changes AᵀA − BBᵀ by exactly η²s²(BGᵀGBᵀ − AᵀGGᵀA)."""
    return eta**2 * s**2 * (B @ G.T @ G @ B.T - A.T @ G @ G.T @ A)


# The scalar picture (lab view 1): w = s·ab, loss ½(s·ab − w*)², from (a₀, 0).
W_STAR, SCALAR_STEPS = 2.0, 400
SCALAR_A0 = (0.25, 0.5, 1.0, 1.5)
SCALAR_ETA = (0.02, 0.1, 0.3)


def scalar_path(a0: float, eta: float, optimizer: str, steps: int = SCALAR_STEPS, s: float = 1.0) -> np.ndarray:
    """(a_t, b_t) under GD or Adam (Topic 5's constants, no decay) on ½(s·ab − w*)²."""
    a, b = a0, 0.0
    m, v = np.zeros(2), np.zeros(2)
    out = [(a, b)]
    for t in range(1, steps + 1):
        e = s * a * b - W_STAR
        g = np.array([s * b * e, s * a * e])
        if optimizer == "gd":
            a, b = a - eta * g[0], b - eta * g[1]
        else:
            m = nlm.ADAM_B1 * m + (1 - nlm.ADAM_B1) * g
            v = nlm.ADAM_B2 * v + (1 - nlm.ADAM_B2) * g * g
            u = (m / (1 - nlm.ADAM_B1**t)) / (np.sqrt(v / (1 - nlm.ADAM_B2**t)) + nlm.ADAM_EPS)
            a, b = a - eta * u[0], b - eta * u[1]
        out.append((a, b))
    return np.array(out)


# --------------------------------------------------------------------------------------------
# Data: Topic 19's aspect rows in the encoder's vocabulary.
# --------------------------------------------------------------------------------------------

TRAIN_NEUTRAL = 1500  # every polar training row, plus this many neutral ones (all neutral rows are read right)
QUAD_SAMPLE = 2000  # negation quadruples scored per run (of 18,120)


def ids_of(tokens) -> np.ndarray:
    voc, V = cm.filing_vocab()
    return np.array([voc.get(w, V - 2) for w in tokens])


def _window(r) -> tuple[np.ndarray, np.ndarray]:
    """The sentence's ids and the aspect span's positions; a sentence longer than n_ctx keeps the n_ctx tokens that
    start at the span (or end the sentence), so the span is always read."""
    x = ids_of(r.toks)
    st = 0 if len(x) <= N_CTX else min(r.span[0], len(x) - N_CTX)
    return x[st: st + N_CTX], np.arange(r.span[0], r.span[1] + 1) - st


def inputs(rs) -> tuple[list, np.ndarray]:
    """Each row as (ids, the aspect span's token positions); the labels (polarity + 1)."""
    return [_window(r) for r in rs], ab.labels(rs)


@lru_cache(maxsize=1)
def augmented_rows() -> tuple:
    """Counterfactual rows: every non-contrast negated-adjective training sentence ("T was not ADJ this period") with
    "not" deleted and the polarity flipped back to the adjective's own (Topic 19's _delete, on the training split)."""
    out = []
    for r, c in zip(ab.rows("train"), ab.constructions("train")):
        if c != "negated adjective" or r.contrast:
            continue
        j = r.toks.index("not")
        adj = r.toks[j + 1]
        for sgn in (1, -1):
            if adj in ab.ADJ[sgn]:
                out.append(ab._delete(r, j, adj, sgn + 1))
    return tuple(out)


@lru_cache(maxsize=None)
def train_set(augment: bool) -> tuple[list, np.ndarray]:
    rs = ab.rows("train")
    con = np.array(ab.constructions("train"))
    neu = np.flatnonzero(con == "neutral")
    keep = np.sort(np.concatenate([np.flatnonzero(con != "neutral"),
                                   _rng(3301).choice(neu, TRAIN_NEUTRAL, replace=False)]))
    sel = [rs[i] for i in keep] + (list(augmented_rows()) if augment else [])
    return inputs(sel)


@lru_cache(maxsize=None)
def eval_set(name: str) -> tuple[list, np.ndarray]:
    if name == "negation":
        q = ab.negation_quadruples()
        pick = np.sort(_rng(3302).choice(len(q), QUAD_SAMPLE, replace=False))
        return inputs([row for i in pick for row in q[i]])
    if name == "valence":
        return inputs([row for qd in ab.valence_quadruples() for row in qd])
    return inputs(ab.rows(name))


def buckets(enc: list, idx) -> dict[int, list[int]]:
    """Rows grouped by length (the encoder has no padding mask, so a batch holds one length)."""
    by: dict[int, list[int]] = {}
    for j in idx:
        by.setdefault(len(enc[j][0]), []).append(int(j))
    return by


def _readout(enc: list, jj: list[int], n: int) -> np.ndarray:
    """(B, n) weights averaging each row's final states over its aspect span."""
    M = np.zeros((len(jj), n), np.float32)
    for q, j in enumerate(jj):
        M[q, enc[j][1]] = 1 / len(enc[j][1])
    return M


def span_features(P: dict, blocks: list, enc: list, chunk: int = 4096) -> np.ndarray:
    """The aspect reading: the mean of LN(x_L) over the aspect span, for every row."""
    F = np.zeros((len(enc), P["E"].shape[1]), np.float32)
    for _, js in buckets(enc, range(len(enc))).items():
        for i in range(0, len(js), chunk):
            jj = js[i: i + chunk]
            tok = np.stack([enc[j][0] for j in jj])
            out = final_states(P, blocks, tok)[0]
            F[jj] = np.einsum("bn,bnd->bd", _readout(enc, jj, tok.shape[1]), out)
    return F


def aspect_loss_grad(P: dict, blocks: list, head: dict, enc: list, jj: list[int], y: np.ndarray,
                     n_norm: int | None = None) -> tuple[float, list, dict, list]:
    """Cross-entropy of the span reading's softmax head on rows jj (one length), divided by n_norm (default len(jj)),
    with its gradients: (loss, [∂W_head, ∂c], {E, P}, per-block) — test_aspect_gradients."""
    tok = np.stack([enc[j][0] for j in jj])
    out, rr, caches = final_states(P, blocks, tok)
    M = _readout(enc, jj, tok.shape[1]).astype(out.dtype)
    f = np.einsum("bn,bnd->bd", M, out)
    lp = nlm.log_softmax(f @ head["W"] + head["c"])
    n = n_norm or len(jj)
    loss = float(-lp[np.arange(len(jj)), y[jj]].sum() / n)
    g = np.exp(lp)
    g[np.arange(len(jj)), y[jj]] -= 1
    g /= n
    G, grads = encoder_backward(P, blocks, tok, out, rr, caches, M[..., None] * (g @ head["W"].T)[:, None, :])
    return loss, [f.T @ g, g.sum(0)], G, grads


# --------------------------------------------------------------------------------------------
# The fine-tuning runs: one cached trainer per frozen config.
# --------------------------------------------------------------------------------------------

PROBE_LAMBDAS = (1e-2, 1e-3, 1e-4)
LR = {"full": 1e-3, "lpft": 1e-3, "l2sp": 1e-3, "lora": 1e-2, "sketch": 1e-3}
LR_HEAD, BATCH, EPOCHS = 1e-2, 64, 3


@dataclass(frozen=True)
class FTConfig:
    method: str = "full"  # full | lpft | lora | l2sp | sketch (the probe is its own function)
    seed: int = 0
    r: int = 16  # LoRA rank
    scale: str = "sqrt"  # LoRA: s = α/√r ("sqrt") or α/r ("lin")
    targets: tuple = ("Wq", "Wv")  # LoRA's adapted matrices, in every block
    lam: float = 0.0  # L2-SP strength
    D: int = 0  # sketch dimension
    init: str = "pretrained"  # or "random" (the never-pretrained twin)
    augment: bool = False  # add the counterfactual rows
    lr: float = 0.0  # 0: the method's default rate (LR)


def base(init: str) -> tuple[dict, list]:
    if init == "random":
        return random_twin()
    return pretrained().P, pretrained().blocks


@lru_cache(maxsize=None)
def probe(init: str, augment: bool) -> dict:
    """Def 1's probe: Topic 17's lr_fit on the frozen span features, λ chosen on the dev rows."""
    P, blocks = base(init)
    enc, y = train_set(augment)
    X = span_features(P, blocks, enc)
    Xd = span_features(P, blocks, eval_set("dev")[0])
    fits = [ltc.lr_fit(X, y, 3, lam) for lam in PROBE_LAMBDAS]
    dev = [float(((Xd @ W.T + b).argmax(1) == eval_set("dev")[1]).mean()) for W, b, _ in fits]
    W, b, _ = fits[int(np.argmax(dev))]
    return {"W": W.T.astype(np.float32), "c": b.astype(np.float32), "lam": PROBE_LAMBDAS[int(np.argmax(dev))]}


class Adam:
    """Topic 5's Adam constants with bias correction, in place on a list of arrays."""

    def __init__(self, xs: list[np.ndarray]):
        self.m = [np.zeros_like(x) for x in xs]
        self.v = [np.zeros_like(x) for x in xs]
        self.t = 0

    def step(self, xs: list[np.ndarray], gs: list[np.ndarray], lr: float) -> None:
        self.t += 1
        c1, c2 = 1 - nlm.ADAM_B1**self.t, 1 - nlm.ADAM_B2**self.t
        for j, (x, g) in enumerate(zip(xs, gs)):
            self.m[j] = nlm.ADAM_B1 * self.m[j] + (1 - nlm.ADAM_B1) * g
            self.v[j] = nlm.ADAM_B2 * self.v[j] + (1 - nlm.ADAM_B2) * g * g
            x -= (lr * (self.m[j] / c1) / (np.sqrt(self.v[j] / c2) + nlm.ADAM_EPS)).astype(x.dtype)


def encoder_leaves(P: dict, blocks: list) -> list[tuple[dict, str]]:
    """Every encoder parameter the classifier can move: E, P, and each block's (the masked-LM bias b is unused)."""
    return [(P, "E"), (P, "P")] + [(blk[s], k) for blk in blocks for s in ("attn", "mlp") for k in blk[s]]


def leaf_grads(G: dict, grads: list, blocks: list) -> list[np.ndarray]:
    return [G["E"], G["P"]] + [grads[i][s][k] for i, blk in enumerate(blocks) for s in ("attn", "mlp") for k in blk[s]]


def sketch(D: int, sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Prop 4's count sketch over all P encoder parameters: bucket h(p) uniform on [D], sign ε(p) = ±1."""
    rng = _rng(3303, D)
    P = sum(sizes)
    return rng.integers(0, D, P), rng.choice(np.array([-1.0, 1.0], np.float32), P)


@lru_cache(maxsize=None)
def fine_tune(cfg: FTConfig) -> dict:
    """Adam with the rate decaying linearly to 0 over EPOCHS of length-bucketed batches (one stream per run). The
    head is a 3-class softmax on the span reading, zero-initialized (the probe's solution for LP-FT). LoRA merges
    W₀ + s·AB into the blocks every step and routes ∂L/∂W through Thm 1(a); the sketch writes θ₀ + Sz."""
    rng = _rng(3310, cfg.seed, int(cfg.augment))  # the batch order: common random numbers across methods
    P0, B0 = base(cfg.init)
    P, blocks = copy.deepcopy(P0), copy.deepcopy(B0)
    enc, y = train_set(cfg.augment)
    if cfg.method == "lpft":
        pr = probe(cfg.init, cfg.augment)
        head = {"W": pr["W"].copy(), "c": pr["c"].copy()}
    else:
        head = {"W": np.zeros((D_MODEL, 3), np.float32), "c": np.zeros(3, np.float32)}
    hl = [head["W"], head["c"]]
    lora, z = [], None
    if cfg.method == "lora":
        s = lora_scale(cfg.r, cfg.scale)
        arng = _rng(3311, cfg.seed)
        for i, blk in enumerate(blocks):
            for sec, k in MATRICES:
                if k in cfg.targets:
                    W0 = flat(k, blk[sec][k]).copy()
                    A, B = lora_init(arng, *W0.shape, cfg.r)
                    lora.append({"blk": i, "sec": sec, "key": k, "W0": W0, "A": A, "B": B, "A0": A.copy(), "s": s,
                                 "shape": blk[sec][k].shape})
        xs = [l["A"] for l in lora] + [l["B"] for l in lora]
    elif cfg.method == "sketch":
        leaves = encoder_leaves(P, blocks)
        sizes = [D_[k].size for D_, k in leaves]
        theta0 = np.concatenate([D_[k].ravel() for D_, k in leaves])
        h, eps = sketch(cfg.D, sizes)
        z = np.zeros(cfg.D, np.float32)
        xs = [z]
    else:
        leaves = encoder_leaves(P, blocks)
        xs = [D_[k] for D_, k in leaves]
    x0 = [x.copy() for x in xs]
    oh, oe = Adam(hl), Adam(xs)
    order = []
    for _ in range(EPOCHS):
        for js in buckets(enc, rng.permutation(len(enc))).values():
            order += [js[i: i + BATCH] for i in range(0, len(js), BATCH)]
    order = [order[i] for i in rng.permutation(len(order))]
    lr = cfg.lr or LR[cfg.method]

    def write():
        for l in lora:
            blocks[l["blk"]][l["sec"]][l["key"]] = unflat(l["key"], l["W0"] + l["s"] * l["A"] @ l["B"], l["shape"])
        if z is not None:
            th = theta0 + eps * z[h]
            o = 0
            for (D_, k), n in zip(leaves, sizes):
                D_[k] = th[o: o + n].reshape(D_[k].shape).astype(np.float32)
                o += n

    for t, jj in enumerate(order):
        write()
        _, gh, G, grads = aspect_loss_grad(P, blocks, head, enc, jj, y)
        frac = 1 - t / len(order)
        oh.step(hl, gh, LR_HEAD * frac)
        if cfg.method == "lora":
            gA, gB = zip(*(lora_grads(flat(l["key"], grads[l["blk"]][l["sec"]][l["key"]]), l["A"], l["B"], l["s"])
                           for l in lora))
            gs = list(gA) + list(gB)
        elif cfg.method == "sketch":
            gflat = np.concatenate([x.ravel() for x in leaf_grads(G, grads, blocks)])
            gs = [np.bincount(h, weights=eps * gflat, minlength=cfg.D).astype(np.float32)]
        else:
            gs = leaf_grads(G, grads, blocks)
        if cfg.lam:
            gs = [gg + cfg.lam * (x - a) for gg, x, a in zip(gs, xs, x0)]
        oe.step(xs, gs, lr * frac)
    write()
    return {"P": P, "blocks": blocks, "head": head, "lora": lora, "steps": len(order)}


# --------------------------------------------------------------------------------------------
# Evaluation: accuracy, constructions, quadruples, forgetting, update norms.
# --------------------------------------------------------------------------------------------

MLM_DOCS, MLM_WINDOWS = 100, 500


@lru_cache(maxsize=1)
def mlm_heldout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forgetting's yardstick: non-overlapping 32-windows of the first test documents (sentences closed by [SEP], as
    in pretraining), corrupted once by Topic 12's Def 2 channel on a fixed stream: (x̃, x, selected)."""
    _, V = cm.filing_vocab()
    seq = []
    for d in ab.docs("test")[:MLM_DOCS]:
        for s0, s1 in d.sentences:
            seq += list(ids_of(fc._tokens(d.text[s0:s1]))) + [V - 1]
    x = np.array(seq[: MLM_WINDOWS * N_CTX]).reshape(-1, N_CTX)
    xt, sel = cm.corrupt(_rng(3304), x, V, V)
    return xt, x, sel


def mlm_loss(P: dict, blocks: list, chunk: int = 250) -> float:
    """Held-out masked-LM loss: mean −log q(x_i | x̃) over the selected positions (Topic 12's Def 2)."""
    xt, x, sel = mlm_heldout()
    tot, cnt = 0.0, 0
    for i in range(0, len(x), chunk):
        lp = nlm.log_softmax(cm.mlm_logits(P, tb._ln_forward(encode(P, blocks, xt[i: i + chunk])[0][-1])[0]))
        m = sel[i: i + chunk]
        tot += float(-np.take_along_axis(lp, x[i: i + chunk, :, None], -1)[..., 0][m].sum())
        cnt += int(m.sum())
    return tot / cnt


def predict(model: dict, name: str) -> np.ndarray:
    """Class probabilities of a model ({P, blocks, head}) on an evaluation set."""
    F = span_features(model["P"], model["blocks"], eval_set(name)[0])
    return np.exp(nlm.log_softmax(F @ model["head"]["W"] + model["head"]["c"]))


def update_norms(model: dict, init: str = "pretrained") -> dict[str, float]:
    """‖ΔW‖_F per matrix type, summed in quadrature over the blocks, and the embedding table's."""
    P0, B0 = base(init)
    out = {k: math.sqrt(sum(float(np.sum((flat(k, b[s][k]) - flat(k, b0[s][k])) ** 2))
                            for b, b0 in zip(model["blocks"], B0))) for s, k in MATRICES}
    out["E"] = float(np.linalg.norm(model["P"]["E"] - P0["E"]))
    return out


@lru_cache(maxsize=None)
def evaluate(cfg: FTConfig | None = None, init: str = "pretrained") -> dict:
    """Every measurement of one run (cfg None is the probe on `init`'s frozen encoder)."""
    if cfg is None:
        P, blocks = base(init)
        model = {"P": P, "blocks": blocks, "head": probe(init, False)}
    else:
        model = fine_tune(cfg)
        init = cfg.init
    pt = predict(model, "test")
    y = eval_set("test")[1]
    pred = pt.argmax(1)
    con = np.array(ab.constructions("test"))
    out = {"acc": float((pred == y).mean()),
           "by": {c: float((pred[con == c] == y[con == c]).mean()) for c in ab.CONSTRUCTIONS},
           "ece": ltc.ece_hat(pt.max(1), pred == y), "pred": tuple(int(v) for v in pred)}
    for q in ("negation", "valence"):
        ok = (predict(model, q).argmax(1) == eval_set(q)[1]).reshape(-1, 4)
        out[q] = {"all": float(ok.all(1).mean()), "pos": [float(v) for v in ok.mean(0)]}
    out["mlm"] = mlm_loss(model["P"], model["blocks"])
    out["norms"] = update_norms(model, init)
    return out




# --------------------------------------------------------------------------------------------
# The grid.
# --------------------------------------------------------------------------------------------

SEEDS = (0, 1, 2)
HEADLINE = {m: tuple(FTConfig(m, seed=s) for s in SEEDS) for m in ("full", "lpft", "lora")}
RANKS = (1, 4, 16, 64)
RANK_RUNS = {(r, sc): FTConfig("lora", r=r, scale=sc) for r in RANKS for sc in ("lin", "sqrt")}  # r = 1: equal s
L2SP_LAMS = (0.1, 1.0)
L2SP_RUNS = tuple(FTConfig("l2sp", lam=lam) for lam in L2SP_LAMS)
SLOW_RUNS = (FTConfig("full", lr=3e-4), FTConfig("lora", lr=3e-3))  # lower rates: more points on the frontier
RANDOM_RUN = FTConfig("full", init="random")
AUG_RUNS = tuple(FTConfig("full", seed=s, augment=True) for s in SEEDS) + (FTConfig("lpft", augment=True),
                                                                           FTConfig("lora", augment=True))
SKETCH_DIMS = (64, 512, 4096)
SKETCH_RUNS = {(D, i): FTConfig("sketch", D=D, init=i) for D in SKETCH_DIMS for i in ("pretrained", "random")}


def runs() -> dict[str, FTConfig]:
    """Every fine-tuning run the page reports, by label (r = 1 has one LoRA run: α/r = α/√r there)."""
    out = {f"{m} s{c.seed}": c for m, cs in HEADLINE.items() for c in cs}
    out |= {f"lora r{r} {sc}": c for (r, sc), c in RANK_RUNS.items() if not (r == 1 and sc == "lin")}
    out |= {f"l2sp {c.lam:g}": c for c in L2SP_RUNS}
    out |= {"full slow": SLOW_RUNS[0], "lora slow": SLOW_RUNS[1], "random full": RANDOM_RUN}
    out |= {f"aug full s{c.seed}": c for c in AUG_RUNS[:3]} | {"aug lpft": AUG_RUNS[3], "aug lora": AUG_RUNS[4]}
    out |= {f"sketch {D} {i}": c for (D, i), c in SKETCH_RUNS.items()}
    return out


def _canon(cfg: FTConfig) -> FTConfig:
    """r = 1 LoRA under α/r is the α/√r run (same s, same stream): one cache entry."""
    return FTConfig("lora", r=1, scale="sqrt") if cfg == RANK_RUNS[(1, "lin")] else cfg


def spread(method: str, key: str = "acc") -> tuple[float, float, float]:
    """(mean, min, max) of a measurement over the headline seeds."""
    v = [evaluate(c)[key] for c in HEADLINE[method]]
    return float(np.mean(v)), float(min(v)), float(max(v))


# --------------------------------------------------------------------------------------------
# Thm 1 at the encoder: the first step from the LP-FT start, against r.
# --------------------------------------------------------------------------------------------

FIRST_R = (1, 2, 4, 8, 16, 32, 64)
FIRST_DRAWS, FIRST_ETA = 16, 1e-2


@lru_cache(maxsize=1)
def lp_gradients() -> dict[tuple[int, str], np.ndarray]:
    """G = ∂L/∂W for Wq, Wv of both blocks on one fixed training batch, at the pretrained encoder with the probe's
    head (at a zero head every encoder gradient is zero, so the first step is taken from the LP-FT start)."""
    P, blocks = base("pretrained")
    head = probe("pretrained", False)
    enc, y = train_set(False)
    by = buckets(enc, range(len(enc)))
    L = max(by, key=lambda n: len(by[n]))
    _, _, _, grads = aspect_loss_grad(P, blocks, head, enc, by[L][:BATCH], y)
    return {(i, k): flat(k, grads[i]["attn"][k]).astype(np.float64) for i in range(len(blocks)) for k in ("Wq", "Wv")}


@lru_cache(maxsize=1)
def first_step_table() -> dict:
    """‖ΔW₁‖_F (summed in quadrature over the four adapted matrices), averaged over FIRST_DRAWS draws of A₀, for each
    r, scaling, and optimizer; and the norm of the expected SGD step η s² r σ² ‖G‖ (Thm 1(c))."""
    Gs = lp_gradients()
    draws = {r: [[_rng(3320, r, i).normal(0, 1 / math.sqrt(G.shape[0]), (G.shape[0], r)) for G in Gs.values()]
                 for i in range(FIRST_DRAWS)] for r in FIRST_R}  # the same A₀ for every scaling and optimizer
    out = {}
    for opt in ("sgd", "adam"):
        for sc in ("lin", "sqrt"):
            out[(opt, sc)] = [float(np.mean([math.sqrt(sum(float(np.sum(first_step(G, A0, lora_scale(r, sc), FIRST_ETA,
                                                                                    opt) ** 2))
                                                           for G, A0 in zip(Gs.values(), d))) for d in draws[r]]))
                              for r in FIRST_R]
    gnorm = math.sqrt(sum(float(np.sum(G**2)) for G in Gs.values()))
    out["expected_sgd"] = {sc: [FIRST_ETA * lora_scale(r, sc) ** 2 * r * gnorm / D_MODEL  # σ² = 1/k, k = d
                                for r in FIRST_R] for sc in ("lin", "sqrt")}
    out["gnorm"] = gnorm
    return out


def trained_lora_norm(cfg: FTConfig) -> float:
    """‖sAB‖_F over the adapted matrices at the end of a LoRA run."""
    return math.sqrt(sum(float(np.sum((l["s"] * l["A"] @ l["B"]) ** 2)) for l in fine_tune(_canon(cfg))["lora"]))


def balance_drift(cfg: FTConfig) -> float:
    """Thm 2 under Adam: max over adapted matrices of ‖AᵀA − BBᵀ − A₀ᵀA₀‖_F / ‖A₀ᵀA₀‖_F at the end of a run."""
    return max(float(np.linalg.norm(balance(l["A"], l["B"]) - l["A0"].T @ l["A0"]) / np.linalg.norm(l["A0"].T @ l["A0"]))
               for l in fine_tune(_canon(cfg))["lora"])


# --------------------------------------------------------------------------------------------
# Thm 2 on a matrix toy: gradient descent's drift is O(η²) per step.
# --------------------------------------------------------------------------------------------

TOY_K, TOY_M, TOY_R, TOY_N = 12, 8, 3, 40


@lru_cache(maxsize=1)
def toy_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares ½‖X(W₀ + AB) − Y‖²/n with a rank-TOY_R target update (float64)."""
    rng = _rng(3330)
    X = rng.normal(size=(TOY_N, TOY_K))
    W0 = rng.normal(size=(TOY_K, TOY_M)) / math.sqrt(TOY_K)
    dW = rng.normal(size=(TOY_K, TOY_R)) @ rng.normal(size=(TOY_R, TOY_M)) / TOY_K
    return X, W0, X @ (W0 + dW)


def toy_gd(eta: float, steps: int, s: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """GD on (A, B) of the toy from Def 1's init; returns (A, B, A₀, final loss)."""
    X, W0, Y = toy_problem()
    A, B = lora_init(_rng(3331), TOY_K, TOY_M, TOY_R, np.float64)
    A0 = A.copy()
    for _ in range(steps):
        G = X.T @ (X @ (W0 + s * A @ B) - Y) / TOY_N
        dA, dB = lora_grads(G, A, B, s)
        A, B = A - eta * dA, B - eta * dB
    loss = 0.5 * float(np.sum((X @ (W0 + s * A @ B) - Y) ** 2)) / TOY_N
    return A, B, A0, loss


TOY_ETAS = (0.004, 0.008, 0.016, 0.032)
TOY_TIME = 8.0  # a fixed horizon t = η·steps


@lru_cache(maxsize=1)
def toy_drift() -> dict:
    """Drift ‖AᵀA − BBᵀ − A₀ᵀA₀‖_F at a fixed horizon t for each η: Thm 2(b) gives O(η) at fixed t (O(η²) per step,
    t/η steps), so the log-log slope against η is 1."""
    d = []
    for eta in TOY_ETAS:
        A, B, A0, _ = toy_gd(eta, round(TOY_TIME / eta))
        d.append(float(np.linalg.norm(balance(A, B) - A0.T @ A0)))
    slope = float(np.polyfit(np.log(TOY_ETAS), np.log(d), 1)[0])
    return {"eta": list(TOY_ETAS), "drift": d, "slope": slope}


# --------------------------------------------------------------------------------------------
# Prop 2: is full fine-tuning's update low-rank?
# --------------------------------------------------------------------------------------------

TRUNC_R = (0, 1, 2, 4, 8, 16, 32, 64)


@lru_cache(maxsize=1)
def spectra() -> list[dict]:
    """Singular values of each block matrix's ΔW under full fine-tuning (seed 0), against a same-norm Gaussian's edge
    τ(√k + √m), τ = ‖ΔW‖_F/√(km); the stable rank ‖ΔW‖_F²/σ₁² and the energy share of the top 4 and 16."""
    model, (_, B0) = fine_tune(HEADLINE["full"][0]), base("pretrained")
    out = []
    for i, (b, b0) in enumerate(zip(model["blocks"], B0)):
        for sec, k in MATRICES:
            dW = (flat(k, b[sec][k]) - flat(k, b0[sec][k])).astype(np.float64)
            sv = np.linalg.svd(dW, compute_uv=False)
            km = dW.shape
            tau = float(np.linalg.norm(dW)) / math.sqrt(km[0] * km[1])
            e = np.cumsum(sv**2) / np.sum(sv**2)
            out.append({"block": i, "matrix": k, "shape": list(km), "sv": sv.tolist(),
                        "edge": tau * (math.sqrt(km[0]) + math.sqrt(km[1])),
                        "stable_rank": float(np.sum(sv**2) / sv[0] ** 2), "top4": float(e[3]), "top16": float(e[15])})
    return out


@lru_cache(maxsize=1)
def truncations() -> dict:
    """Test accuracy of full fine-tuning (seed 0) with every block ΔW replaced by its best rank-r approximation (the
    fine-tuned embeddings and head kept); r = 0 keeps only those."""
    model, (_, B0) = fine_tune(HEADLINE["full"][0]), base("pretrained")
    y = eval_set("test")[1]
    acc = []
    for r in TRUNC_R:
        blocks = copy.deepcopy(model["blocks"])
        for b, b0 in zip(blocks, B0):
            for sec, k in MATRICES:
                W0 = flat(k, b0[sec][k]).astype(np.float64)
                U, sv, Vt = np.linalg.svd(flat(k, b[sec][k]).astype(np.float64) - W0, full_matrices=False)
                b[sec][k] = unflat(k, (W0 + (U[:, :r] * sv[:r]) @ Vt[:r]).astype(np.float32), b[sec][k].shape)
        m = {"P": model["P"], "blocks": blocks, "head": model["head"]}
        acc.append(float((predict(m, "test").argmax(1) == y).mean()))
    return {"r": list(TRUNC_R), "acc": acc, "full": evaluate(HEADLINE["full"][0])["acc"]}


# --------------------------------------------------------------------------------------------
# Thm 3: what fine-tuning leaves alone.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def unseen_tokens() -> np.ndarray:
    """Token ids that never occur in the fine-tuning inputs (with or without the counterfactual rows)."""
    seen = np.zeros(cm.filing_vocab()[1] + 1, bool)
    for aug in (False, True):
        for x, _ in train_set(aug)[0]:
            seen[x] = True
    return np.flatnonzero(~seen)


def embedding_rows_moved(cfg: FTConfig) -> tuple[int, int]:
    """(rows of E that moved at all, rows that moved among tokens unseen in fine-tuning)."""
    E0 = base(cfg.init)[0]["E"]
    moved = np.any(fine_tune(_canon(cfg))["P"]["E"] != E0, axis=1)
    return int(moved.sum()), int(moved[unseen_tokens()].sum())


SPAN_K, SPAN_DIM, SPAN_N = 10, 4, 30


def span_toy(optimizer: str, steps: int = 200, eta: float = 0.05) -> float:
    """Thm 3(a): one linear layer W (k × m) on inputs confined to a SPAN_DIM-dimensional subspace S, trained by SGD
    or Adam on squared loss; returns ‖(I − Π_S)(W_t − W₀)‖_F (0 under SGD; not under Adam's per-coordinate scaling)."""
    rng = _rng(3340)
    Q = np.linalg.qr(rng.normal(size=(SPAN_K, SPAN_K)))[0][:, :SPAN_DIM]
    X = rng.normal(size=(SPAN_N, SPAN_DIM)) @ Q.T
    Y = rng.normal(size=(SPAN_N, 3))
    W0 = rng.normal(size=(SPAN_K, 3))
    W = W0.copy()
    m, v = np.zeros_like(W), np.zeros_like(W)
    for t in range(1, steps + 1):
        i = rng.integers(SPAN_N, size=5)
        g = X[i].T @ (X[i] @ W - Y[i]) / 5
        if optimizer == "sgd":
            W -= eta * g
        else:
            m = nlm.ADAM_B1 * m + (1 - nlm.ADAM_B1) * g
            v = nlm.ADAM_B2 * v + (1 - nlm.ADAM_B2) * g * g
            W -= eta * (m / (1 - nlm.ADAM_B1**t)) / (np.sqrt(v / (1 - nlm.ADAM_B2**t)) + nlm.ADAM_EPS)
    return float(np.linalg.norm((np.eye(SPAN_K) - Q @ Q.T) @ (W - W0)))


KUMAR_D, KUMAR_K, KUMAR_M, KUMAR_N = 10, 2, 4, 40


def kumar_toy(head: str, eta: float = 0.01, steps: int = 4000) -> dict:
    """Thm 3(c), Kumar et al.'s setting: f(x) = vᵀBx, B ∈ ℝ^{k×d}, squared loss, training inputs spanning an
    m-dimensional S, perfect features B₀ = UB*. GD (small η, a gradient-flow proxy) from v₀ = the probe's head
    ("lp") or a random head ("random"). Returns the OOD loss on inputs from S⊥, max |(B_t − B₀)x| over an S⊥ basis
    (their Lemma A.3), and the drift of vvᵀ − BBᵀ (their Lemma A.4)."""
    rng = _rng(3345)
    Q = np.linalg.qr(rng.normal(size=(KUMAR_D, KUMAR_D)))[0]
    S, Sperp = Q[:, :KUMAR_M], Q[:, KUMAR_M:]
    Bstar = np.linalg.qr(rng.normal(size=(KUMAR_D, KUMAR_K)))[0].T  # orthonormal rows
    vstar = rng.normal(size=KUMAR_K)
    U = rng.normal(size=(KUMAR_K, KUMAR_K)) + 2 * np.eye(KUMAR_K)
    B = U @ Bstar
    X = rng.normal(size=(KUMAR_N, KUMAR_M)) @ S.T
    Y = X @ Bstar.T @ vstar
    F = X @ B.T
    v = np.linalg.lstsq(F, Y, rcond=None)[0] if head == "lp" else rng.normal(size=KUMAR_K)
    B0, inv0 = B.copy(), np.outer(v, v) - B @ B.T
    for _ in range(steps):
        res = X @ B.T @ v - Y
        gv, gB = B @ X.T @ res / KUMAR_N, np.outer(v, res @ X) / KUMAR_N
        v, B = v - eta * gv, B - eta * gB
    Xo = rng.normal(size=(500, KUMAR_D - KUMAR_M)) @ Sperp.T
    ood = float(np.mean((Xo @ B.T @ v - Xo @ Bstar.T @ vstar) ** 2))
    idl = float(np.mean((X @ B.T @ v - Y) ** 2))
    return {"ood": ood, "id": idl, "perp_move": float(np.abs((B - B0) @ Sperp).max()),
            "balance_drift": float(np.linalg.norm(np.outer(v, v) - B @ B.T - inv0))}


# --------------------------------------------------------------------------------------------
# Prop 3: L2-SP is ridge shrinkage of the displacement toward θ₀.
# --------------------------------------------------------------------------------------------


def l2sp_quadratic(H: np.ndarray, theta_star: np.ndarray, theta0: np.ndarray, lam: float) -> np.ndarray:
    """argmin ½(θ − θ*)ᵀH(θ − θ*) + ½λ‖θ − θ₀‖² — by Prop 3, θ₀ + (H + λI)⁻¹H(θ* − θ₀)."""
    return theta0 + np.linalg.solve(H + lam * np.eye(len(H)), H @ (theta_star - theta0))


# --------------------------------------------------------------------------------------------
# Prop 4: a count-sketch subspace; d₉₀.
# --------------------------------------------------------------------------------------------


def sketch_matrix(rng: np.random.Generator, P: int, D: int) -> np.ndarray:
    """Prop 4's S (P × D): one ±1 per row at a uniform column."""
    S = np.zeros((P, D))
    S[np.arange(P), rng.integers(0, D, P)] = rng.choice([-1.0, 1.0], P)
    return S


def d90(init: str) -> dict:
    """The least sketch dimension on the grid reaching 90% of full fine-tuning's accuracy for the same encoder."""
    full = spread("full")[0] if init == "pretrained" else evaluate(RANDOM_RUN)["acc"]
    acc = [evaluate(SKETCH_RUNS[(D, init)])["acc"] for D in SKETCH_DIMS]
    hit = [D for D, a in zip(SKETCH_DIMS, acc) if a >= 0.9 * full]
    return {"full": full, "acc": acc, "d90": hit[0] if hit else None}


# --------------------------------------------------------------------------------------------
# Prop 5, Cor 1: fine-tuning as a kernel.
# --------------------------------------------------------------------------------------------

NTK_TRAIN, NTK_TEST, NTK_R, NTK_STEPS, NTK_ADAM_LR = 120, 240, 16, 300, 1e-2


def kernel_gd(K: np.ndarray, Kx: np.ndarray, f0: np.ndarray, f0x: np.ndarray, grad_f, eta: float,
              steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Prop 5: GD on a linearized model is f ← f − ηK∇_f ℓ(f) on the training outputs and f(x) ← f(x) − ηK(x, X)∇_f ℓ
    at any other point; K over (example, output) pairs."""
    f, fx = f0.copy(), f0x.copy()
    for _ in range(steps):
        g = grad_f(f)
        f, fx = f - eta * K @ g, fx - eta * Kx @ g
    return f, fx


def _lora_jacobian(P, blocks, head, enc, rows_, A, Bs, s) -> np.ndarray:
    """∂f_c(x)/∂(A, B) for LoRA on Wq, Wv of both blocks (Thm 1(a): s·G Bᵀ and s·AᵀG), flattened, for each row and
    class: (len(rows)·3, Σ (k + m)·r). At B = 0 the A-columns are zero."""
    J = []
    for j in rows_:
        tok = enc[j][0][None]
        out, rr, caches = final_states(P, blocks, tok)
        M = _readout(enc, [j], tok.shape[1])
        for c in range(3):
            _, grads = encoder_backward(P, blocks, tok, out, rr, caches, M[..., None] * head["W"][:, c][None, None, :])
            J.append(np.concatenate([x.ravel() for key in A for x in
                                     lora_grads(flat(key[1], grads[key[0]]["attn"][key[1]]), A[key], Bs[key], s)]))
    return np.array(J, np.float64)


def _lora_run(P, blocks, head, enc, y, tr, A0, s, optimizer: str, lr: float, steps: int):
    """Full-batch LoRA on rows tr from B = 0, the head frozen: GD (rate lr) or Adam. Returns (blocks, A, B)."""
    A = {key: a.copy() for key, a in A0.items()}
    Bs = {key: np.zeros((NTK_R, D_MODEL), np.float32) for key in A}
    blk = copy.deepcopy(blocks)
    W0 = {(i, k): flat(k, blocks[i]["attn"][k]).copy() for (i, k) in A}
    by = buckets(enc, tr)
    xs = [A[key] for key in A] + [Bs[key] for key in A]
    opt = Adam(xs)

    def write():
        for (i, k) in A:
            blk[i]["attn"][k] = unflat(k, W0[(i, k)] + s * A[(i, k)] @ Bs[(i, k)], blocks[i]["attn"][k].shape)

    for _ in range(steps):
        write()
        acc_g = {key: [np.zeros_like(A[key]), np.zeros_like(Bs[key])] for key in A}
        for jj in by.values():
            _, _, _, grads = aspect_loss_grad(P, blk, head, enc, jj, y, len(tr))
            for key in A:
                dA, dB = lora_grads(flat(key[1], grads[key[0]]["attn"][key[1]]), A[key], Bs[key], s)
                acc_g[key][0] += dA
                acc_g[key][1] += dB
        gs = [acc_g[key][0] for key in A] + [acc_g[key][1] for key in A]
        if optimizer == "gd":
            for x, g in zip(xs, gs):
                x -= (lr * g).astype(x.dtype)
        else:
            opt.step(xs, gs, lr)
    write()
    return blk, A, Bs


@lru_cache(maxsize=1)
def ntk_experiment() -> dict:
    """Linearize LoRA (r = NTK_R, α/√r, Wq/Wv) at B = 0 around the LP-FT start (the probe's head, frozen; at a zero
    head the Jacobian is zero), on NTK_TRAIN training rows of the commonest length. Run NTK_STEPS full-batch steps of cross-entropy GD for the
    real LoRA model (A and B) and the same steps as kernel GD (Prop 5), and Adam from the same start. Report test
    accuracies on NTK_TEST rows, agreement with the kernel predictor, the logit-change gap, and how far K moved."""
    P, blocks = base("pretrained")
    head = probe("pretrained", False)
    enc, y = train_set(False)
    rng = _rng(3350)
    by = buckets(enc, range(len(enc)))
    common = max(by, key=lambda n: len(by[n]))  # one length, so each full-batch step is one batch
    tr = rng.choice(by[common], NTK_TRAIN, replace=False)
    te_enc, te_y = eval_set("test")
    te = rng.choice(len(te_enc), NTK_TEST, replace=False)
    s = lora_scale(NTK_R, "sqrt")
    A0 = {(i, k): rng.normal(0, 1 / math.sqrt(D_MODEL), (D_MODEL, NTK_R)).astype(np.float32)
          for i in range(len(blocks)) for k in ("Wq", "Wv")}
    B0 = {key: np.zeros((NTK_R, D_MODEL), np.float32) for key in A0}
    Jtr = _lora_jacobian(P, blocks, head, enc, tr, A0, B0, s)
    Jte = _lora_jacobian(P, blocks, head, te_enc, te, A0, B0, s)
    K, Kx = Jtr @ Jtr.T, Jte @ Jtr.T
    f0 = predict_logits(P, blocks, head, enc, tr).reshape(-1)
    f0x = predict_logits(P, blocks, head, te_enc, te).reshape(-1)
    Y = np.eye(3)[y[tr]].reshape(-1)

    def grad_f(f):
        return (np.exp(nlm.log_softmax(f.reshape(-1, 3))).reshape(-1) - Y) / NTK_TRAIN

    def ce(f):
        return float(-np.sum(Y * nlm.log_softmax(f.reshape(-1, 3)).reshape(-1)) / NTK_TRAIN)

    eta = NTK_TRAIN / float(np.linalg.eigvalsh(K)[-1])
    f_lin, fx_lin = kernel_gd(K, Kx, f0, f0x, grad_f, eta, NTK_STEPS)
    yt = te_y[te]
    out = {"acc_start": float((f0x.reshape(-1, 3).argmax(1) == yt).mean()), "loss_start": ce(f0),
           "acc_kernel": float((fx_lin.reshape(-1, 3).argmax(1) == yt).mean()), "loss_kernel": ce(f_lin), "eta": eta}
    for opt, lr in (("gd", eta), ("adam", NTK_ADAM_LR)):
        blk, A, Bs = _lora_run(P, blocks, head, enc, y, tr, A0, s, opt, lr, NTK_STEPS)
        fx = predict_logits(P, blk, head, te_enc, te).reshape(-1)
        JT = _lora_jacobian(P, blk, head, enc, tr, A, Bs, s)
        out[opt] = {"acc": float((fx.reshape(-1, 3).argmax(1) == yt).mean()),
                    "loss": ce(predict_logits(P, blk, head, enc, tr).reshape(-1)),
                    "agree": float((fx.reshape(-1, 3).argmax(1) == fx_lin.reshape(-1, 3).argmax(1)).mean()),
                    "logit_gap": float(np.linalg.norm(fx - fx_lin) / np.linalg.norm(fx - f0x)),
                    "kernel_move": float(np.linalg.norm(JT @ JT.T - K) / np.linalg.norm(K)),
                    "balance_drift": max(float(np.linalg.norm(balance(A[k_], Bs[k_]) - A0[k_].T @ A0[k_])
                                               / np.linalg.norm(A0[k_].T @ A0[k_])) for k_ in A)}
    return out


def predict_logits(P, blocks, head, enc, rows_) -> np.ndarray:
    """Logits (len(rows), 3) in row order, through span_features."""
    sub = [enc[j] for j in rows_]
    return span_features(P, blocks, sub).astype(np.float64) @ head["W"] + head["c"]


# --------------------------------------------------------------------------------------------
# Finance: aspect reading aggregated to tone; document-level fine-tuning as the honest negative.
# --------------------------------------------------------------------------------------------


def tone_from(pred) -> float:
    """Document tone from an aspect reader's test predictions: the first aspect of each tone-bearing sentence, then
    Topic 19's Bayes rule on the (+, 0, −) counts (Prop 2 there)."""
    rs = ab.rows("test")
    first = {(r.doc, r.sent): i for i, r in enumerate(rs) if r.first}
    ok = []
    for di, d in enumerate(ab.docs("test")):
        p = [pred[first[(di, s)]] - 1 for s in ab.tone_sentences(d)]
        ok.append(ab.bayes_tone((p.count(1), p.count(0), p.count(-1))) == d.tone)
    return float(np.mean(ok))


DOC_EPOCHS, DOC_LR, DOC_LR_HEAD, DOC_BATCH = 4, 3e-3, 3e-2, 32


def doc_windows(docs) -> tuple[list[np.ndarray], np.ndarray]:
    """Each document as non-overlapping n_ctx windows of its ids, sentences closed by [SEP] (the last window
    right-aligned so every id is read), and the tones (+1)."""
    _, V = cm.filing_vocab()
    out = []
    for d in docs:
        x = np.concatenate([np.append(ids_of(fc._tokens(d.text[a:b])), V - 1) for a, b in d.sentences])
        st = list(range(0, len(x) - N_CTX + 1, N_CTX))
        if st[-1] + N_CTX < len(x):
            st.append(len(x) - N_CTX)
        out.append(np.stack([x[a: a + N_CTX] for a in st]))
    return out, np.array([d.tone + 1 for d in docs])


@lru_cache(maxsize=None)
def doc_set(split: str):
    return doc_windows(ab.docs(split))


def _doc_pool(W: list[np.ndarray], P: dict, blocks: list) -> np.ndarray:
    return np.stack([final_states(P, blocks, w)[0].mean((0, 1)) for w in W])


@lru_cache(maxsize=None)
def doc_fine_tune(init: str) -> dict:
    """Topic 17's document tone read from the mean final state over a document's windows: full fine-tuning (zero head;
    Adam, rate decaying linearly) and the frozen probe, on the same 1,500 / 1,500 split."""
    P, blocks = copy.deepcopy(base(init)[0]), copy.deepcopy(base(init)[1])
    W, y = doc_set("train")
    Wt, yt = doc_set("test")
    F = _doc_pool(W, *base(init))
    Wp, bp, _ = ltc.lr_fit(F, y, 3, 1e-3)
    probe_acc = float(((_doc_pool(Wt, *base(init)) @ Wp.T + bp).argmax(1) == yt).mean())
    head = {"W": np.zeros((D_MODEL, 3), np.float32), "c": np.zeros(3, np.float32)}
    hl = [head["W"], head["c"]]
    leaves = encoder_leaves(P, blocks)
    xs = [D_[k] for D_, k in leaves]
    oh, oe = Adam(hl), Adam(xs)
    rng = _rng(3360, int(init == "random"))
    order = [p[i: i + DOC_BATCH] for p in (rng.permutation(len(W)) for _ in range(DOC_EPOCHS))
             for i in range(0, len(W), DOC_BATCH)]
    for t, idx in enumerate(order):
        tok = np.concatenate([W[j] for j in idx])
        doc = np.concatenate([np.full(len(W[j]), q) for q, j in enumerate(idx)])
        out, rr, caches = final_states(P, blocks, tok)
        cnt = np.bincount(doc) * N_CTX
        f = np.stack([out[doc == q].sum((0, 1)) for q in range(len(idx))]) / cnt[:, None]
        g = np.exp(nlm.log_softmax(f @ head["W"] + head["c"]))
        g[np.arange(len(idx)), y[idx]] -= 1
        g /= len(idx)
        frac = 1 - t / len(order)
        oh.step(hl, [f.T @ g, g.sum(0)], DOC_LR_HEAD * frac)
        dout = np.broadcast_to(((g @ head["W"].T) / cnt[:, None])[doc][:, None, :], out.shape).astype(np.float32)
        G, grads = encoder_backward(P, blocks, tok, out, rr, caches, dout)
        oe.step(xs, leaf_grads(G, grads, blocks), DOC_LR * frac)
    pred = (_doc_pool(Wt, P, blocks) @ head["W"] + head["c"]).argmax(1)
    trn = (_doc_pool(W, P, blocks) @ head["W"] + head["c"]).argmax(1)
    return {"acc": float((pred == yt).mean()), "train": float((trn == y).mean()), "probe": probe_acc}


@lru_cache(maxsize=1)
def document_readers() -> dict:
    out = dict(ab.document_baselines())
    out["bigram LR (Topic 17)"] = ltc.fin_fit(ltc.FinConfig("lr", True))["acc"]
    return out


# --------------------------------------------------------------------------------------------
# The lab's payload and the summary.
# --------------------------------------------------------------------------------------------

PATH_EVERY = 4  # scalar paths are stored every 4th step


def _scalar_payload() -> dict:
    paths = {opt: {f"{a0:g}": {f"{eta:g}": _r(scalar_path(a0, eta, opt)[::PATH_EVERY].tolist())
                                 for eta in SCALAR_ETA} for a0 in SCALAR_A0} for opt in ("gd", "adam")}
    drift = {opt: {f"{a0:g}": {f"{eta:g}": _r(float(np.max(np.abs(np.diff(scalar_path(a0, eta, opt) ** 2, axis=1)[:, 0]
                                                                   + a0**2))), 4)
                               for eta in SCALAR_ETA} for a0 in SCALAR_A0} for opt in ("gd", "adam")}
    return {"w_star": W_STAR, "a0": list(SCALAR_A0), "eta": list(SCALAR_ETA), "every": PATH_EVERY,
            "steps": SCALAR_STEPS, "paths": paths, "max_drift": drift}


def _spread_payload(method: str, key: str = "acc") -> dict:
    m, lo, hi = spread(method, key)
    return {"mean": _r(m), "min": _r(lo), "max": _r(hi)}


def _rank_payload() -> dict:
    fs = first_step_table()
    trained = {sc: [{"r": r, "acc": _r(evaluate(_canon(RANK_RUNS[(r, sc)]))["acc"]),
                     "mlm": _r(evaluate(_canon(RANK_RUNS[(r, sc)]))["mlm"]),
                     "norm": _r(trained_lora_norm(RANK_RUNS[(r, sc)]))} for r in RANKS] for sc in ("lin", "sqrt")}
    sig = tb.ak._sig  # significant figures: the SGD α/r steps fall to 10⁻⁴
    return {"r": list(FIRST_R), "first_step": {f"{o}_{sc}": [sig(v, 4) for v in fs[(o, sc)]] for o in ("sgd", "adam")
                                               for sc in ("lin", "sqrt")},
            "expected_sgd": {sc: [sig(x, 4) for x in v] for sc, v in fs["expected_sgd"].items()}, "eta": FIRST_ETA,
            "alpha": ALPHA,
            "trained": trained, "probe": _r(evaluate(None)["acc"]),
            "full": _spread_payload("full"), "lpft": _spread_payload("lpft"), "lora": _spread_payload("lora")}


def _spectrum_payload() -> dict:
    return {"matrices": [{"block": x["block"], "matrix": x["matrix"], "shape": x["shape"], "sv": _r(x["sv"], 4),
                          "edge": _r(x["edge"], 4), "stable_rank": _r(x["stable_rank"]), "top4": _r(x["top4"]),
                          "top16": _r(x["top16"])} for x in spectra()],
            "trunc": {"r": truncations()["r"], "acc": _r(truncations()["acc"]), "full": _r(truncations()["full"])}}


def _point(label: str, cfg: FTConfig | None) -> dict:
    e = evaluate(cfg) if cfg is not None else evaluate(None)
    method = "probe" if cfg is None else ("random" if cfg.init == "random" else cfg.method)
    return {"label": label, "method": method, "acc": _r(e["acc"]), "mlm": _r(e["mlm"]),
            "augment": bool(cfg.augment) if cfg else False}


def _frontier_payload() -> dict:
    pts = [_point("probe", None)] + [_point(k, _canon(c)) for k, c in runs().items()]
    return {"points": pts, "mlm_pretrained": _r(mlm_loss(*base("pretrained"))),
            "d90": {i: {"D": list(SKETCH_DIMS), "acc": _r(d90(i)["acc"]), "full": _r(d90(i)["full"]),
                        "threshold": _r(0.9 * d90(i)["full"]), "d90": d90(i)["d90"]} for i in ("pretrained", "random")},
            "ntk": _ntk_payload()}


def _ntk_payload() -> dict:
    """Accuracies, agreement, and kernel movement to the page's 3 decimals; losses to 4 significant figures (the
    page quotes 0.5990 against 0.5989, and Adam's 5·10⁻⁵)."""
    n, sig = ntk_experiment(), tb.ak._sig
    arm = {opt: {"acc": _r(n[opt]["acc"]), "loss": sig(n[opt]["loss"], 4), "agree": _r(n[opt]["agree"]),
                 "kernel_move": _r(n[opt]["kernel_move"]), "balance_drift": sig(n[opt]["balance_drift"], 3)}
           for opt in ("gd", "adam")}
    return {"acc_start": _r(n["acc_start"]), "loss_start": sig(n["loss_start"], 4), "acc_kernel": _r(n["acc_kernel"]),
            "loss_kernel": sig(n["loss_kernel"], 4), **arm}


NEG_READERS = (("probe", None), ("full", HEADLINE["full"][0]), ("LP-FT", HEADLINE["lpft"][0]),
               ("LoRA", HEADLINE["lora"][0]), ("random-init", RANDOM_RUN))


def _negation_payload() -> dict:
    rd = document_readers()
    tone = [{"label": k, "acc": _r(v)} for k, v in rd.items() if "ceiling" not in k]
    tone += [{"label": "document fine-tune", "acc": _r(doc_fine_tune("pretrained")["acc"])},
             {"label": "document fine-tune, random init", "acc": _r(doc_fine_tune("random")["acc"])},
             {"label": "aspect fine-tune, aggregated", "acc": _r(tone_from(evaluate(HEADLINE["full"][0])["pred"]))},
             {"label": "augmented, aggregated", "acc": _r(tone_from(evaluate(AUG_RUNS[0])["pred"]))}]
    quads = {}
    for name, cs in (("full", HEADLINE["full"]), ("augmented", AUG_RUNS[:3])):
        quads[name] = [_r(evaluate(c)["negation"]["pos"]) for c in cs]
    return {"constructions": list(ab.CONSTRUCTIONS),
            "n": [int(np.sum(np.array(ab.constructions("test")) == c)) for c in ab.CONSTRUCTIONS],
            "readers": [{"label": lab, "by": [_r((evaluate(c) if c else evaluate(None))["by"][k])
                                              for k in ab.CONSTRUCTIONS],
                         "acc": _r((evaluate(c) if c else evaluate(None))["acc"])} for lab, c in NEG_READERS],
            "quads": quads, "tone": tone, "ceiling": _r(rd["Bayes ceiling (Topic 17)"])}


def viz_payload() -> dict:
    return {"scalar": _scalar_payload(), "rank": _rank_payload(), "spectrum": _spectrum_payload(),
            "frontier": _frontier_payload(), "negation": _negation_payload()}


# --------------------------------------------------------------------------------------------
# Verification harness: one test per claim on the page.
# --------------------------------------------------------------------------------------------


def _f64(P: dict, blocks: list) -> tuple[dict, list]:
    return ({k: v.astype(np.float64) for k, v in P.items()},
            [{s_: {k: v.astype(np.float64) for k, v in b[s_].items()} for s_ in b} for b in blocks])


def test_encoder_backward_matches_topic12():
    """encoder_backward, fed Topic 12's masked-LM output gradient, returns mlm_loss_grad's gradients exactly
    (float64, a small random encoder): the lifted loop is the same function."""
    rng = _rng(3390)
    cfg = cm._small_config()
    P, blocks = cm.init_encoder(rng, cm.H_VOCAB + 1, cfg, dtype=np.float64)
    tok = rng.integers(0, cm.H_VOCAB + 1, (3, cfg.n))
    tgt = rng.integers(0, cm.H_VOCAB, (3, cfg.n))
    sel = rng.random((3, cfg.n)) < 0.5
    _, Gm, gm = cm.mlm_loss_grad(P, blocks, tok, tgt, sel)
    out, r, caches = final_states(P, blocks, tok)
    d, V = out.shape[-1], P["b"].shape[0]
    rows = np.flatnonzero(sel.ravel())
    o = out.reshape(-1, d)[rows]
    g = np.exp(nlm.log_softmax(cm.mlm_logits(P, o)))
    g[np.arange(len(rows)), tgt.ravel()[rows]] -= 1
    g /= len(rows)
    dout = np.zeros((tok.size, d))
    dout[rows] = g @ P["E"][:V] / math.sqrt(d)
    G, grads = encoder_backward(P, blocks, tok, out, r, caches, dout.reshape(out.shape))
    tied = np.zeros_like(P["E"])
    tied[:V] = g.T @ o / math.sqrt(d)
    assert np.allclose(G["E"] + tied, Gm["E"], rtol=0, atol=1e-13) and np.allclose(G["P"], Gm["P"], atol=1e-13)
    for a, b in zip(grads, gm):
        for s_ in a:
            for k in a[s_]:
                assert np.allclose(a[s_][k], b[s_][k], rtol=0, atol=1e-13), (s_, k)


def test_fast_attention():
    """The flattened-matmul attention and encoder equal Topic 11's mha_forward/mha_backward and Topic 12's encode
    (float64, the pretrained encoder's first block on real rows)."""
    P, blocks = _f64(*base("pretrained"))
    rng = _rng(3389)
    X = rng.normal(size=(3, 11, D_MODEL))
    a, ca = mha_forward(blocks[0]["attn"], X)
    b, cb = tb.mha_forward(blocks[0]["attn"], X, causal=False)
    assert np.allclose(a, b, rtol=0, atol=1e-12)
    dO = rng.normal(size=a.shape)
    (dx, G), (dx2, G2) = mha_backward(blocks[0]["attn"], dO, ca), tb.mha_backward(blocks[0]["attn"], dO, cb)
    assert np.allclose(dx, dx2, rtol=0, atol=1e-11) and all(np.allclose(G[k], G2[k], rtol=0, atol=1e-11) for k in G2)
    tok = np.stack([x for x, _ in train_set(False)[0][:400] if len(x) == 9][:5])
    assert all(np.allclose(u, w, rtol=0, atol=1e-11) for u, w in zip(encode(P, blocks, tok)[0], cm.encode(P, blocks, tok)[0]))


def test_aspect_gradients():
    """The classification loss's gradient through the pretrained encoder (cast to float64) against central
    differences, on a few entries of every kind of parameter."""
    P, blocks = _f64(*base("pretrained"))
    rng = _rng(3391)
    head = {"W": rng.normal(0, 0.3, (D_MODEL, 3)), "c": rng.normal(0, 0.1, 3)}
    enc, y = train_set(False)
    by = buckets(enc, range(len(enc)))
    jj = by[12][:4]
    _, gh, G, grads = aspect_loss_grad(P, blocks, head, enc, jj, y)
    probes = [("E", P["E"], G["E"], (enc[jj[0]][0][3], 5)), ("P", P["P"], G["P"], (2, 7)),
              ("W", head["W"], gh[0], (4, 1))]
    for i in range(2):
        for s_, k in MATRICES:
            idx = tuple(int(rng.integers(0, n)) for n in blocks[i][s_][k].shape)
            probes.append((f"{i}{k}", blocks[i][s_][k], grads[i][s_][k], idx))
    for name, arr, g, idx in probes:
        old = arr[idx]
        arr[idx] = old + 1e-6
        lp_ = aspect_loss_grad(P, blocks, head, enc, jj, y)[0]
        arr[idx] = old - 1e-6
        lm = aspect_loss_grad(P, blocks, head, enc, jj, y)[0]
        arr[idx] = old
        assert abs((lp_ - lm) / 2e-6 - g[idx]) < 1e-8 * max(1.0, abs(g[idx])), name


def _toy_loss(W0, A, B, s, X, Y):
    return 0.5 * float(np.sum((X @ (W0 + s * A @ B) - Y) ** 2))


def test_prop1_factorization():
    """Prop 1: rank sAB ≤ r; r(k + m) parameters; the merged map equals the adapter's two-path forward; (AS, S⁻¹B)
    leaves ΔW unchanged; two rank-r updates can sum to rank 2r."""
    rng = _rng(3392)
    k, m, r = 9, 7, 2
    A, B, X = rng.normal(size=(k, r)), rng.normal(size=(r, m)), rng.normal(size=(5, k))
    W0 = rng.normal(size=(k, m))
    assert np.linalg.matrix_rank(A @ B) == r and A.size + B.size == r * (k + m) < k * m
    assert np.allclose(X @ (W0 + 0.5 * A @ B), X @ W0 + 0.5 * (X @ A) @ B)
    S = rng.normal(size=(r, r)) + 3 * np.eye(r)
    assert np.allclose((A @ S) @ (np.linalg.inv(S) @ B), A @ B)
    A2, B2 = rng.normal(size=(k, r)), rng.normal(size=(r, m))
    assert np.linalg.matrix_rank(A @ B + A2 @ B2) == 2 * r


def test_thm1a_factored_gradient():
    """Thm 1(a): ∂L/∂A = sGBᵀ and ∂L/∂B = sAᵀG against central differences (float64)."""
    rng = _rng(3393)
    k, m, r, s = 6, 5, 3, 0.7
    W0, A, B = rng.normal(size=(k, m)), rng.normal(size=(k, r)), rng.normal(size=(r, m))
    X, Y = rng.normal(size=(8, k)), rng.normal(size=(8, m))
    G = X.T @ (X @ (W0 + s * A @ B) - Y)
    dA, dB = lora_grads(G, A, B, s)
    for M_, dM in ((A, dA), (B, dB)):
        for idx in np.ndindex(M_.shape):
            old = M_[idx]
            M_[idx] = old + 1e-6
            lp_ = _toy_loss(W0, A, B, s, X, Y)
            M_[idx] = old - 1e-6
            lm = _toy_loss(W0, A, B, s, X, Y)
            M_[idx] = old
            assert abs((lp_ - lm) / 2e-6 - dM[idx]) < 1e-6


def test_thm1b_first_step():
    """Thm 1(b): at B = 0 the A-gradient is 0 and one SGD step gives ΔW₁ = −ηs²A₀A₀ᵀG; one Adam step from zero moments
    is the sign step of Thm 1(d) (to Adam's ε)."""
    rng = _rng(3394)
    k, m, r, s, eta = 10, 6, 3, 1.3, 0.05
    A0, B0 = lora_init(rng, k, m, r, np.float64)
    G = rng.normal(size=(k, m))
    dA, dB = lora_grads(G, A0, B0, s)
    assert np.all(dA == 0)
    assert np.allclose(s * A0 @ (B0 - eta * dB), first_step(G, A0, s, eta, "sgd"), atol=1e-14)
    opt = Adam([B0])
    opt.step([B0], [dB], eta)
    assert np.allclose(s * A0 @ B0, first_step(G, A0, s, eta, "adam"), atol=1e-6)


def test_thm1c_expected_first_step():
    """Thm 1(c): E[A₀A₀ᵀ] = rσ²I, so the mean SGD first step is −ηs²rσ²G: over 4,000 draws of A₀ the empirical mean
    of ΔW₁ matches it entrywise within 4 standard errors."""
    rng = _rng(3395)
    k, m, r, s, eta = 8, 5, 4, 1.0, 1.0
    G = rng.normal(size=(k, m))
    draws = np.array([first_step(G, lora_init(rng, k, m, r, np.float64)[0], s, eta, "sgd") for _ in range(4000)])
    target = -eta * s**2 * r * (1 / k) * G
    se = draws.std(0) / math.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(0) - target) < 4 * se + 1e-12)


def test_thm1d_sign_step_bounds():
    """Thm 1(d): under Adam's sign step, ηs·σ_min(A₀)√(rm) ≤ ‖ΔW₁‖_F ≤ ηs·σ_max(A₀)√(rm), on random instances."""
    rng = _rng(3396)
    for _ in range(200):
        k, m = int(rng.integers(4, 40)), int(rng.integers(2, 30))
        r = int(rng.integers(1, k + 1))
        A0 = lora_init(rng, k, m, r, np.float64)[0]
        G = rng.normal(size=(k, m))
        n = np.linalg.norm(first_step(G, A0, 0.9, 0.1, "adam"))
        sv = np.linalg.svd(A0, compute_uv=False)
        c = 0.1 * 0.9 * math.sqrt(r * m)
        assert c * sv[-1] - 1e-12 <= n <= c * sv[0] + 1e-12


def test_thm2_one_step_identity():
    """Thm 2(b): one GD step changes AᵀA − BBᵀ by exactly η²s²(BGᵀGBᵀ − AᵀGGᵀA) (Du, Hu, and Lee's Claim B.1)."""
    rng = _rng(3397)
    for _ in range(20):
        k, m, r = 7, 5, 3
        A, B, G = rng.normal(size=(k, r)), rng.normal(size=(r, m)), rng.normal(size=(k, m))
        s, eta = float(rng.uniform(0.2, 2)), float(rng.uniform(0.01, 0.5))
        dA, dB = lora_grads(G, A, B, s)
        A1, B1 = A - eta * dA, B - eta * dB
        assert np.allclose(balance(A1, B1) - balance(A, B), gd_drift(A, B, G, s, eta), atol=1e-11)


def test_thm2_conservation():
    """Thm 2(a): at a fixed horizon t = η·steps the drift of AᵀA − BBᵀ from A₀ᵀA₀ is O(η) (O(η²) per step), so it
    vanishes in the flow limit: log-log slope 1 against η. The scalar paths keep a² − b² = a₀² to O(η) under GD."""
    td = toy_drift()
    assert abs(td["slope"] - 1) < 0.05 and td["drift"][0] < 1e-3
    for a0 in SCALAR_A0:
        path = scalar_path(a0, SCALAR_ETA[0], "gd")
        assert np.max(np.abs(path[:, 0] ** 2 - path[:, 1] ** 2 - a0**2)) < 0.05 * max(1.0, a0**2)
        assert abs(path[-1, 0] * path[-1, 1] - W_STAR) < 1e-3


def test_prop3_l2sp_shrinkage():
    """Prop 3: the L2-SP optimum of a quadratic is θ₀ + (H + λI)⁻¹H(θ* − θ₀), and along each eigenvector of H the
    displacement is shrunk by h/(h + λ)."""
    rng = _rng(3398)
    n = 6
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    h = rng.uniform(0.1, 5, n)
    H = Q @ np.diag(h) @ Q.T
    ts, t0, lam = rng.normal(size=n), rng.normal(size=n), 0.7
    th = l2sp_quadratic(H, ts, t0, lam)
    assert np.allclose(H @ (th - ts) + lam * (th - t0), 0, atol=1e-12)
    assert np.allclose(Q.T @ (th - t0), h / (h + lam) * (Q.T @ (ts - t0)))


def test_prop4_sketch_unbiased():
    """Prop 4: diag(SSᵀ) = 1 exactly, E[SSᵀ] = I off the diagonal, and E‖Sᵀg‖² = ‖g‖² (Monte Carlo, 4 SE)."""
    rng = _rng(3399)
    P, D = 30, 6
    g = rng.normal(size=P)
    vals, off = [], []
    for _ in range(4000):
        S = sketch_matrix(rng, P, D)
        SS = S @ S.T
        assert np.all(np.diag(SS) == 1)
        off.append(SS[0, 1])
        vals.append(float(np.sum((S.T @ g) ** 2)))
    vals = np.array(vals)
    assert abs(vals.mean() - g @ g) < 4 * vals.std() / math.sqrt(len(vals))
    assert abs(np.mean(off)) < 4 * np.std(off) / math.sqrt(len(off))


def test_prop5_kernel_gd():
    """Prop 5: GD on a linearized model, f_lin(x; θ) = f₀(x) + J(x)(θ − θ₀), is kernel GD with K = JJᵀ on the training
    and test outputs (cross-entropy and squared loss); for squared loss f_t − Y = (I − ηK)^t (f₀ − Y)."""
    rng = _rng(3400)
    n, nt, p, eta, T = 9, 5, 14, 0.05, 30
    J, Jx = rng.normal(size=(n * 3, p)), rng.normal(size=(nt * 3, p))
    f0, f0x = rng.normal(size=n * 3), rng.normal(size=nt * 3)
    Y = np.eye(3)[rng.integers(0, 3, n)].reshape(-1)
    losses = {"ce": lambda f: (np.exp(nlm.log_softmax(f.reshape(-1, 3))).reshape(-1) - Y) / n, "sq": lambda f: f - Y}
    for name, gf in losses.items():
        th = np.zeros(p)
        for _ in range(T):
            th = th - eta * J.T @ gf(f0 + J @ th)
        f, fx = kernel_gd(J @ J.T, Jx @ J.T, f0, f0x, gf, eta, T)
        assert np.allclose(f, f0 + J @ th, atol=1e-10) and np.allclose(fx, f0x + Jx @ th, atol=1e-10), name
    K = J @ J.T
    f, _ = kernel_gd(K, Jx @ J.T, f0, f0x, losses["sq"], eta, T)
    assert np.allclose(f - Y, np.linalg.matrix_power(np.eye(n * 3) - eta * K, T) @ (f0 - Y), atol=1e-10)


def test_cor1_lora_kernel():
    """Cor 1: at B = 0 LoRA's kernel is s²·tr(Gₓᵀ A₀A₀ᵀ G_x′), whose mean over A₀ is s²rσ² times the full kernel
    tr(GₓᵀG_x′) of the adapted matrix (Monte Carlo, 4 SE)."""
    rng = _rng(3401)
    k, m, r, s = 8, 5, 3, 1.5
    Gx, Gy = rng.normal(size=(k, m)), rng.normal(size=(k, m))
    vals = []
    for _ in range(4000):
        A0 = lora_init(rng, k, m, r, np.float64)[0]
        vals.append(float(np.sum((s * A0.T @ Gx) * (s * A0.T @ Gy))))
    vals = np.array(vals)
    target = s**2 * r / k * float(np.sum(Gx * Gy))
    assert abs(vals.mean() - target) < 4 * vals.std() / math.sqrt(len(vals))


def test_thm3a_span_lemma():
    """Thm 3(a): SGD moves a linear layer only along its inputs' span, ‖(I − Π_S)(W_t − W₀)‖ = 0; Adam's
    per-coordinate scaling breaks it."""
    assert span_toy("sgd") < 1e-12 and span_toy("adam") > 0.1


def test_thm3c_kumar():
    """Thm 3(c) (Kumar et al.'s Lemma A.3 and Prop 3.7): fine-tuning never moves B on S⊥; from the probe's head on
    perfect features the OOD loss stays 0, from a random head it does not."""
    lp, rnd = kumar_toy("lp"), kumar_toy("random")
    assert lp["perp_move"] < 1e-12 and rnd["perp_move"] < 1e-12
    assert lp["ood"] < 1e-20 and rnd["ood"] > 1e-2 and rnd["id"] < 1e-3


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "low-rank-adaptation-lab.json"


def test_thm1c_sketch_fluctuation():
    """Thm 1(c): E‖A₀A₀ᵀ − rσ²I‖_F² = rσ⁴k(k + 1) (Monte Carlo, 4 SE, at two ranks)."""
    rng = _rng(3402)
    for k, r in ((6, 2), (10, 7)):
        v = np.array([float(np.sum((A @ A.T - r / k * np.eye(k)) ** 2))
                      for A in (lora_init(rng, k, 1, r, np.float64)[0] for _ in range(6000))])
        assert abs(v.mean() - sketch_fluctuation(k, r, 1 / k)) < 4 * v.std() / math.sqrt(len(v))


def test_thm1_rank_scaling_at_the_encoder():
    """Thm 1 at the encoder (four adapted matrices, the LP-FT start, 16 draws of A₀ per rank): Adam's first step is
    rank-stable under α/√r (within a factor 1.5 over r = 1…64) and shrinks as 1/√r under α/r; SGD's mean step under
    α/√r is r-independent while a single draw sits √((k + 1)/r) away, so its norm falls toward the mean."""
    fs = first_step_table()
    a_sqrt, a_lin = np.array(fs[("adam", "sqrt")]), np.array(fs[("adam", "lin")])
    assert a_sqrt.max() / a_sqrt.min() < 1.5
    assert np.allclose(a_lin / a_sqrt, 1 / np.sqrt(FIRST_R), rtol=1e-12)  # s_lin/s_sqrt = 1/√r, same draws
    e = np.array(fs["expected_sgd"]["sqrt"])
    assert np.allclose(e, e[0]) and np.all(np.diff(fs[("sgd", "sqrt")]) < 0) and fs[("sgd", "sqrt")][-1] > e[0]


def test_thm2_adam_breaks_balance():
    """Thm 2 is a gradient-descent fact: full-batch GD from the LP-FT start keeps AᵀA − BBᵀ within 10⁻⁴ of A₀ᵀA₀
    (relative), while Adam moves it by more than half its size, in the NTK runs and in the headline LoRA run."""
    n = ntk_experiment()
    assert n["gd"]["balance_drift"] < 1e-4 and n["adam"]["balance_drift"] > 0.5
    assert balance_drift(HEADLINE["lora"][0]) > 0.5


def test_pretraining_buys_negation():
    """The ladder on aspect rows: the frozen probe < the never-pretrained encoder fine-tuned < every pretrained seed;
    and pretraining is what reads negated adjectives (a gap of more than 0.1 to the random-init encoder)."""
    probe_, rnd = evaluate(None), evaluate(RANDOM_RUN)
    assert probe_["acc"] < rnd["acc"] < spread("full")[1]
    neg = np.mean([evaluate(c)["by"]["negated adjective"] for c in HEADLINE["full"]])
    assert neg - rnd["by"]["negated adjective"] > 0.1


def test_prop2_update_is_low_rank():
    """Prop 2 measured: every block matrix's full-FT ΔW has σ₁ above a same-norm Gaussian's edge and stable rank
    below 6; the rank-4 truncation keeps ≥ 98% of full fine-tuning's accuracy, while dropping the block updates (r = 0)
    leaves near chance; LoRA trained at r = 4 does not reach the rank-4 truncation."""
    for x in spectra():
        assert x["sv"][0] > x["edge"] and x["stable_rank"] < 6, (x["block"], x["matrix"])
    t = truncations()
    a = dict(zip(t["r"], t["acc"]))
    assert a[4] >= 0.98 * t["full"] and a[0] < 0.6
    assert all(evaluate(_canon(RANK_RUNS[(4, sc)]))["acc"] < a[4] for sc in ("lin", "sqrt"))


def test_thm3b_unseen_embeddings_never_move():
    """Thm 3(b): under Adam (full, LP-FT, augmented) and L2-SP no embedding row of a token absent from the fine-tuning
    inputs moves; full fine-tuning moves every row it saw; LoRA moves none."""
    seen = cm.filing_vocab()[1] + 1 - len(unseen_tokens())
    for c in HEADLINE["full"] + HEADLINE["lpft"] + L2SP_RUNS + AUG_RUNS[:3]:
        assert embedding_rows_moved(c)[1] == 0
    assert embedding_rows_moved(HEADLINE["full"][0])[0] == seen
    assert embedding_rows_moved(HEADLINE["lora"][0]) == (0, 0)


def test_lpft_forgets_less():
    """Kumar et al.'s LP-FT, measured as forgetting: every LP-FT seed ends with lower held-out masked-LM loss than every
    full fine-tuning seed, at a mean accuracy within 0.01."""
    assert max(evaluate(c)["mlm"] for c in HEADLINE["lpft"]) < min(evaluate(c)["mlm"] for c in HEADLINE["full"])
    assert abs(spread("lpft")[0] - spread("full")[0]) < 0.01


def test_lpft_dominates_lora():
    """At this scale LoRA does not "learn less and forget less" (Biderman et al.): every LoRA run (seeds, ranks,
    scalings, the slower rate, augmented) is Pareto-dominated by every LP-FT run, which is both more accurate and
    forgets less."""
    lora = [evaluate(_canon(c)) for c in runs().values() if c.method == "lora"]
    lpft = [evaluate(c) for c in runs().values() if c.method == "lpft"]
    assert max(e["acc"] for e in lora) < min(e["acc"] for e in lpft)
    assert min(e["mlm"] for e in lora) > max(e["mlm"] for e in lpft)


def test_prop3_l2sp_tradeoff():
    """Prop 3 measured: raising λ (0 = full fine-tuning) lowers both forgetting and accuracy."""
    e = [evaluate(HEADLINE["full"][0])] + [evaluate(c) for c in L2SP_RUNS]
    assert all(a["mlm"] > b["mlm"] and a["acc"] > b["acc"] for a, b in zip(e, e[1:]))


def test_prop4_intrinsic_dimension():
    """Prop 4 measured: no sketch on the grid (D ≤ 4,096, 2.6% of the parameters) reaches 90% of full fine-tuning for
    either encoder; the pretrained encoder leads the random one at every D."""
    pre, rnd = d90("pretrained"), d90("random")
    assert pre["d90"] is None and rnd["d90"] is None
    assert all(a > b for a, b in zip(pre["acc"], rnd["acc"]))


def test_prop5_kernel_regime():
    """Prop 5 measured: from the LP-FT start, full-batch GD on LoRA stays in the kernel regime (≥ 98% test agreement
    with kernel GD, training losses within 10⁻³, K moves < 10%); Adam leaves it (K moves > 100%, agreement < 80%)."""
    n = ntk_experiment()
    assert n["gd"]["agree"] >= 0.98 and abs(n["gd"]["loss"] - n["loss_kernel"]) < 1e-3 and n["gd"]["kernel_move"] < 0.1
    assert n["adam"]["kernel_move"] > 1 and n["adam"]["agree"] < 0.8


def test_negation_shortcut_and_augmentation():
    """Topic 19's quadruples: the fine-tuned encoder reads both negated forms (mean ≥ 0.9 over seeds) but not the
    deleted-"not" frame, which training never shows (its ADJ₋ copy ≤ 0.3 on every seed); with the counterfactual rows
    every seed reads all four positions (≥ 0.95)."""
    pos = np.array([evaluate(c)["negation"]["pos"] for c in HEADLINE["full"]])
    assert pos[:, :2].mean() >= 0.9 and np.all(pos[:, 3] <= 0.3)
    aug = np.array([evaluate(c)["negation"]["pos"] for c in AUG_RUNS[:3]])
    assert np.all(aug >= 0.95)


def test_tone_by_aggregation():
    """Finance: the fine-tuned aspect reader, aggregated by Topic 19's Bayes rule, lands within 2 SE of the 0.890
    ceiling and above every document-level reader; document-level fine-tuning lands below unigram LR, and the
    never-pretrained encoder does as well within 0.05."""
    rd = document_readers()
    agg = tone_from(evaluate(HEADLINE["full"][0])["pred"])
    se = math.sqrt(ltc.bayes_ceiling() * (1 - ltc.bayes_ceiling()) / len(ab.docs("test")))
    assert abs(agg - ltc.bayes_ceiling()) < 2 * se and agg > max(v for k, v in rd.items() if "ceiling" not in k)
    dp, dr = doc_fine_tune("pretrained"), doc_fine_tune("random")
    assert dp["acc"] < rd["unigram LR (Topic 17)"] and abs(dp["acc"] - dr["acc"]) < 0.05


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with --write-payload after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def summary() -> dict:
    out = {}
    out["train rows / augmented rows / test rows"] = (f"{len(train_set(False)[0])} / {len(augmented_rows())} / "
                                                      f"{len(eval_set('test')[0])}")
    out["masked-LM loss, pretrained (held-out)"] = mlm_loss(*base("pretrained"))
    out["probe: lambda / test acc"] = f"{probe('pretrained', False)['lam']:g} / {evaluate(None)['acc']:.6f}"
    for label, c in runs().items():
        e = evaluate(_canon(c))
        out[f"{label}: acc / mlm / ece / negadj"] = (f"{e['acc']:.6f} / {e['mlm']:.6f} / {e['ece']:.6f} / "
                                                    f"{e['by']['negated adjective']:.6f}")
        out[f"{label}: negation quad positions / all; valence all"] = (
            f"{np.round(e['negation']['pos'], 6).tolist()} / {e['negation']['all']:.6f}; {e['valence']['all']:.6f}")
    for m in HEADLINE:
        out[f"{m}: acc mean/min/max"] = "/".join(f"{v:.6f}" for v in spread(m))
        out[f"{m}: mlm mean/min/max"] = "/".join(f"{v:.6f}" for v in spread(m, "mlm"))
    fs = first_step_table()
    for k in (("sgd", "lin"), ("sgd", "sqrt"), ("adam", "lin"), ("adam", "sqrt")):
        out[f"Thm 1 first step {k}"] = " ".join(f"{v:.6f}" for v in fs[k])
    out["Thm 1 expected SGD step (sqrt)"] = f"{fs['expected_sgd']['sqrt'][0]:.6f}"
    for (r, sc), c in RANK_RUNS.items():
        out[f"LoRA r={r} {sc}: trained ||sAB||"] = trained_lora_norm(c)
    nf = evaluate(HEADLINE["full"][0])["norms"]
    out["full s0: ||dW|| Wq, Wv (both blocks) / all six types / E"] = (
        f"{math.hypot(nf['Wq'], nf['Wv']):.6f} / {math.sqrt(sum(nf[k] ** 2 for _, k in MATRICES)):.6f} / {nf['E']:.6f}")
    out["Thm 2 toy drift slope"] = toy_drift()["slope"]
    out["Thm 2 Adam drift (headline LoRA s0)"] = balance_drift(HEADLINE["lora"][0])
    for x in spectra():
        out[f"Prop 2 {x['block']}{x['matrix']}: s1/edge, stable rank, top4"] = (
            f"{x['sv'][0] / x['edge']:.6f}, {x['stable_rank']:.6f}, {x['top4']:.6f}")
    t = truncations()
    out["Prop 2 truncation acc by r"] = " ".join(f"{r}:{a:.6f}" for r, a in zip(t["r"], t["acc"]))
    out["Thm 3 unseen tokens / rows moved (full s0)"] = f"{len(unseen_tokens())} / {embedding_rows_moved(HEADLINE['full'][0])}"
    out["Thm 3(c) Kumar OOD: LP-FT / random head"] = f"{kumar_toy('lp')['ood']:.3e} / {kumar_toy('random')['ood']:.6f}"
    for i in ("pretrained", "random"):
        d = d90(i)
        out[f"Prop 4 sketch acc ({i}), full, d90"] = f"{[round(a, 6) for a in d['acc']]}, {d['full']:.6f}, {d['d90']}"
    n = ntk_experiment()
    out["Prop 5 start acc / loss; kernel acc / loss"] = (f"{n['acc_start']:.6f} / {n['loss_start']:.6f}; "
                                                        f"{n['acc_kernel']:.6f} / {n['loss_kernel']:.6f}")
    for opt in ("gd", "adam"):
        out[f"Prop 5 {opt}: acc, loss, agree, logit gap, K move"] = ", ".join(
            f"{n[opt][k]:.6f}" for k in ("acc", "loss", "agree", "logit_gap", "kernel_move"))
    for k, v in document_readers().items():
        out[f"document: {k}"] = v
    for i in ("pretrained", "random"):
        d = doc_fine_tune(i)
        out[f"document fine-tune ({i}): test / train / probe"] = f"{d['acc']:.6f} / {d['train']:.6f} / {d['probe']:.6f}"
    out["tone by aggregation: full s0 / augmented s0"] = (f"{tone_from(evaluate(HEADLINE['full'][0])['pred']):.6f} / "
                                                         f"{tone_from(evaluate(AUG_RUNS[0])['pred']):.6f}")
    return out





def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


def _run_harness() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    print("\nSummary:")
    for k, v in summary().items():
        print(f"  {k:62} {v:.6f}" if isinstance(v, float) else f"  {k:62} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__" and "--quick" in sys.argv:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f) and n in sys.argv]
    for n, f in tests:
        f()
        print("PASS", n)
elif __name__ == "__main__":
    raise SystemExit(_run_harness())
