"""Contextual embeddings and the masked-LM objective — the reference implementation for formalNLP's twelfth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. One vector per type: a polyseme's type-level PMI is the log of a mixture of its senses' exponentiated PMIs, so a
   static vector (Topic 7's SGNS factorizes shifted PMI) sees the mixture; any static map's sense probe is capped at
   the majority sense's share.
2. The masked-LM objective: BERT's corruption channel (select with probability ρ; of the selected, [MASK] with m,
   a uniform random token with r, keep with κ). The population risk is entropy plus KL, so its unique minimizer is
   the posterior of the original token given the corrupted sequence (Thm 1); under the channel that posterior has a
   closed form (Thm 2): the context's prediction at a [MASK], shrunk toward the visible token elsewhere, and ρ drops
   out. A pre-LN bidirectional encoder (Topic 11's blocks, a learned position table, tied embeddings) trained on an
   HMM language whose exact posteriors come from forward–backward, so the theorems are checked against the truth.
3. Pseudo-likelihood and the missing joint: one mask per sequence is Besag's pseudo-likelihood, Fisher-consistent
   (Thm 3, checked on an Ising chain); Brook's lemma rebuilds a positive joint from its full conditionals (Thm 4);
   the four-cycle compatibility gap δ vanishes for any joint's conditionals and does not for the MLM's (Cor 2).
4. Geometry: self-similarity, anisotropy, and Ethayarajh's MEV (Prop 2) by layer.
5. Probing: a linear sense probe against the static cap and the exact Bayes accuracy, with a shuffled-label control.
6. Finance case study: an encoder trained on generate_corpus(300, 0) — Topic 7's corpus. Company aliases that are
   both ORG ("Kestrel Turbine Services") and PRODUCT ("the Kestrel turbine"); direction words whose news depends on
   the aspect they report ("costs rose" is bad news); and Topic 7's strong/weak probe, rerun on contextual vectors.

Cross-checks: the encoder's gradient against Topic 5's central differences; the selected-rows loss against the full
forward pass; the matmul MLP backward against Topic 11's; Thm 2 against brute-force enumeration of the joint;
Brook's lemma against the enumerated joint.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/contextual-embeddings-masked-lm/contextual_embeddings_masked_lm.py
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "transformer-block-residual-stream"))
sys.path.insert(0, str(_HERE.parent / "word2vec-implicit-factorization"))
import transformer_block_residual_stream as tb  # noqa: E402 — module objects: their test_* stay out of this harness
import word2vec_implicit_factorization as w2v  # noqa: E402

pe, ak, fc, nlm = tb.pe, tb.ak, tb.fc, tb.nlm
ds = w2v.ds

# ONE stream per random-instance claim and per trained model, each from default_rng([SEED, k]); k ≥ 1500 keeps this
# module's streams apart from Topics 5–11.
SEED = ak.SEED
Params = dict[str, np.ndarray]

# BERT's corruption channel (Devlin et al. 2019, §3.1): select 15%; of those, 80% [MASK], 10% random, 10% kept.
RHO, M_MASK, R_RAND, KAPPA = 0.15, 0.8, 0.1, 0.1


# --------------------------------------------------------------------------------------------
# 1. One vector per type: the HMM language and the mixture identity.
# --------------------------------------------------------------------------------------------

H_STATES, H_VOCAB, H_N = 8, 25, 24  # hidden states, word types, sequence length
H_SWITCH = 0.05  # probability of leaving a regime per step
H_POLY = 0.4  # the share of the polyseme in its two states' emissions
POLYSEME = 0  # token 0, emitted by one state of each regime (sense_states())


@lru_cache(maxsize=1)
def hmm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, B, π): two regimes of four states; within a regime the rows are Dirichlet(½) draws, and with probability
    H_SWITCH the chain jumps to a uniform state of the other regime. State h emits its own three tokens 3h+1 … 3h+3;
    the heaviest state of each regime under the stationary law π also emits the polyseme, token 0, with probability
    H_POLY, so its two senses are nearly equally common. Every sequence has positive probability (T > 0, and every
    token is emitted by some state)."""
    rng = np.random.default_rng([SEED, 1500])
    T = np.zeros((H_STATES, H_STATES))
    for g in (0, 1):
        own, other = np.arange(4) + 4 * g, np.arange(4) + 4 * (1 - g)
        for h in own:
            T[h, own] = (1 - H_SWITCH) * rng.dirichlet(np.full(4, 0.5))
            T[h, other] = H_SWITCH / 4
    w, vecs = np.linalg.eig(T.T)
    pi = np.real(vecs[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    B = np.zeros((H_STATES, H_VOCAB))
    for h in range(H_STATES):
        B[h, 1 + 3 * h: 4 + 3 * h] = rng.dirichlet(np.ones(3))
    for h in (int(np.argmax(pi[:4])), 4 + int(np.argmax(pi[4:]))):
        B[h] *= 1 - H_POLY
        B[h, POLYSEME] = H_POLY
    return T, B, pi


def sense_states() -> tuple[int, int]:
    """The polyseme's two senses: the states that emit it (one per regime)."""
    return tuple(int(h) for h in np.flatnonzero(hmm()[1][:, POLYSEME]))


H_NAMES = ("w", *(f"{'ab'[h // 4]}{3 * (h % 4) + j + 1}" for h in range(H_STATES) for j in range(3)))
H_MASK = H_VOCAB  # the input-only [MASK] symbol


def sample_hmm(rng: np.random.Generator, batch: int, n: int = H_N) -> tuple[np.ndarray, np.ndarray]:
    """batch sequences of length n from the stationary chain: (tokens, hidden states)."""
    T, B, pi = hmm()
    cT, cB = T.cumsum(1), B.cumsum(1)
    h = np.zeros((batch, n), int)
    x = np.zeros((batch, n), int)
    for i in range(n):
        u = rng.random(batch)[:, None]
        h[:, i] = (u > (pi.cumsum()[None] if i == 0 else cT[h[:, i - 1]])).sum(1)
        x[:, i] = (rng.random(batch)[:, None] > cB[h[:, i]]).sum(1)
    return np.minimum(x, H_VOCAB - 1), np.minimum(h, H_STATES - 1)


def _forward_backward(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward–backward over emission likelihoods E (b, n, K): returns α̂_i = P(h_i | obs_{<i}) and β̂_i ∝
    P(obs_{>i} | h_i), each row normalized; the leave-one-out posterior is ∝ α̂β̂ and the full one ∝ α̂ E β̂."""
    T, _, pi = hmm()
    b, n, K = E.shape
    a = np.empty((b, n, K))
    be = np.ones((b, n, K))
    prior = np.broadcast_to(pi, (b, K))
    for i in range(n):
        a[:, i] = prior / prior.sum(1, keepdims=True)
        f = a[:, i] * E[:, i]
        prior = (f / f.sum(1, keepdims=True)) @ T
    for i in range(n - 2, -1, -1):
        v = (E[:, i + 1] * be[:, i + 1]) @ T.T
        be[:, i] = v / v.sum(1, keepdims=True)
    return a, be


def channel(rho: float = RHO, m: float = M_MASK, r: float = R_RAND, kappa: float = KAPPA) -> np.ndarray:
    """C[x, s] = P(x̃ = s | x) for one position (s ∈ tokens ∪ {[MASK]}): kept unselected with 1 − ρ; selected, [MASK]
    with m, a uniform token with r (which may be x), kept with κ."""
    V = H_VOCAB
    C = np.full((V, V + 1), rho * r / V)
    C[:, V] = rho * m
    C[np.arange(V), np.arange(V)] += 1 - rho + rho * kappa
    return C


def selected_channel(V: int, m: float = M_MASK, r: float = R_RAND, kappa: float = KAPPA) -> np.ndarray:
    """S[y, s] = P(x̃_i = s | x_i = y, i selected): Def 2's corruption of a selected position."""
    S = np.full((V, V + 1), r / V)
    S[:, V] = m
    S[np.arange(V), np.arange(V)] += kappa
    return S


def bayes_mlm(xt: np.ndarray, rho: float = RHO, m: float = M_MASK, r: float = R_RAND,
              kappa: float = KAPPA) -> tuple[np.ndarray, np.ndarray]:
    """Thm 2 on the HMM: p_i = P(x_i | x̃_{−i}) by forward–backward with the channel folded into the emissions, and
    the Bayes-optimal MLM prediction q_i(y) ∝ p_i(y) S[y, x̃_i] at every position (as if selected). (q, p), (b, n, V)."""
    _, B, _ = hmm()
    Bt = B @ channel(rho, m, r, kappa)
    a, be = _forward_backward(np.moveaxis(Bt[:, xt], 0, -1))
    post = a * be
    p = (post / post.sum(-1, keepdims=True)) @ B
    q = p * np.moveaxis(selected_channel(H_VOCAB, m, r, kappa)[:, xt], 0, -1)
    return q / q.sum(-1, keepdims=True), p


def clean_conditionals(x: np.ndarray) -> np.ndarray:
    """P(x_i = y | x_{−i}) of the HMM itself, at every position: (b, n, V)."""
    _, B, _ = hmm()
    a, be = _forward_backward(np.moveaxis(B[:, x], 0, -1))
    post = a * be
    return (post / post.sum(-1, keepdims=True)) @ B


def sense_posterior(x: np.ndarray) -> np.ndarray:
    """P(h_i = 1 | x) / (P(h_i = 1 | x) + P(h_i = 5 | x)) at every position (meaningful where x_i is the polyseme)."""
    _, B, _ = hmm()
    E = np.moveaxis(B[:, x], 0, -1)
    a, be = _forward_backward(E)
    post = a * E * be
    a_, b_ = sense_states()
    tot = post[..., a_] + post[..., b_]
    return np.divide(post[..., a_], tot, out=np.full(tot.shape, np.nan), where=tot > 0)


def neighbor_pmi() -> dict[str, np.ndarray]:
    """Prop 1 on the HMM, exactly: with c the next token, PMI(w, c) for the polyseme as a type, the senses' PMI(s_k, c),
    and the senses' shares π_k (from the stationary law)."""
    T, B, pi = hmm()
    joint_h = pi[:, None] * T  # P(h_i, h_{i+1})
    pc = (pi @ T) @ B  # P(c) (stationary)
    pw_c = np.zeros(H_VOCAB)
    sense_pmi, share = [], []
    for s in sense_states():
        p_c_given_s = (joint_h[s] / pi[s]) @ B  # P(c | h_i = s), which is P(c | s, x_i = w) by the HMM's independence
        sense_pmi.append(np.log(p_c_given_s / pc))
        share.append(pi[s] * B[s, POLYSEME])
        pw_c += pi[s] * B[s, POLYSEME] * p_c_given_s
    share = np.array(share) / sum(share)
    type_pmi = np.log(pw_c / sum(pi[s] * B[s, POLYSEME] for s in sense_states()) / pc)
    return {"type": type_pmi, "senses": np.array(sense_pmi), "share": share}


# --------------------------------------------------------------------------------------------
# 2. The masked-LM objective: the corruption channel and the encoder.
# --------------------------------------------------------------------------------------------


def corrupt(rng: np.random.Generator, x: np.ndarray, V: int, mask_id: int, rho: float = RHO, m: float = M_MASK,
            r: float = R_RAND) -> tuple[np.ndarray, np.ndarray]:
    """Def 2: each position selected independently with probability ρ; a selected one becomes [MASK] with m, a uniform
    token of the V with r, and stays with 1 − m − r. Returns (x̃, selected)."""
    sel = rng.random(x.shape) < rho
    u = rng.random(x.shape)
    xt = x.copy()
    xt[sel & (u < m)] = mask_id
    rnd = sel & (u >= m) & (u < m + r)
    xt[rnd] = rng.integers(V, size=int(rnd.sum()))
    return xt, sel


@dataclass(frozen=True)
class MLMConfig:
    source: str  # "hmm" or "filings"
    n: int
    d: int
    heads: int
    layers: int
    steps: int
    batch: int
    lr: float = 0.01
    stream: int = 0


HMM_MLM = MLMConfig("hmm", H_N, 32, 2, 2, steps=3000, batch=64, stream=1510)
FILING_MLM = MLMConfig("filings", 32, 64, 4, 2, steps=2000, batch=64, stream=1520)
D_HEAD = 16


def init_encoder(rng: np.random.Generator, V_in: int, cfg: MLMConfig, dtype=np.float32) -> tuple[Params, list]:
    """Token table E (V_in, d) at N(0, 1) — tied: its first V rows are also the output map — a learned position table
    P (n, d) at N(0, 0.1²) (BERT's absolute table, Topic 10's Def 1), an output bias, and Topic 11's pre-LN blocks
    with an MLP of width 4d."""
    P = {"E": rng.normal(0, 1, (V_in, cfg.d)), "P": rng.normal(0, 0.1, (cfg.n, cfg.d)), "b": np.zeros(V_in - 1)}
    blocks = [tb.init_block(rng, cfg.d, cfg.heads, D_HEAD, 4 * cfg.d) for _ in range(cfg.layers)]
    P = {k: v.astype(dtype) for k, v in P.items()}
    blocks = [{s: {k: v.astype(dtype) for k, v in blk[s].items()} for s in blk} for blk in blocks]
    return P, blocks


def encode(P: Params, blocks: list, tok: np.ndarray) -> tuple[list[np.ndarray], list]:
    """The stream x_0 = E[tok] + P, then Topic 11's pre-LN blocks with bidirectional attention: (xs, caches)."""
    return tb.stack_forward(blocks, P["E"][tok] + P["P"][None, : tok.shape[1]], "pre")


def mlm_logits(P: Params, out: np.ndarray) -> np.ndarray:
    """Tied unembedding of the final LayerNorm's output: LN(x_L) E[:V]ᵀ/√d + b ([MASK] is never a prediction)."""
    V = P["b"].shape[0]
    return out @ P["E"][:V].T / math.sqrt(out.shape[-1]) + P["b"]


def mlm_log_probs(P: Params, blocks: list, tok: np.ndarray) -> np.ndarray:
    """log q(· | x̃) at every position — the plain forward pass the training loss is checked against."""
    xs, _ = encode(P, blocks, tok)
    return nlm.log_softmax(mlm_logits(P, tb._ln_forward(xs[-1])[0]))


def mlp_backward(M: Params, dOut: np.ndarray, cache: tuple) -> tuple[np.ndarray, Params]:
    """Topic 11's mlp_backward with its weight gradients as matmuls over the flattened batch (its einsum is the
    training loop's bottleneck); equal to it, test_fast_mlp_backward."""
    X, Z, h = cache
    dZ = (dOut @ M["W2"].T) * (Z > 0)
    d, w = X.shape[-1], h.shape[-1]
    G = {"W2": h.reshape(-1, w).T @ dOut.reshape(-1, d), "b2": dOut.sum((0, 1)),
         "W1": X.reshape(-1, d).T @ dZ.reshape(-1, w), "b1": dZ.sum((0, 1))}
    return dZ @ M["W1"].T, G


def mlm_loss_grad(P: Params, blocks: list, tok: np.ndarray, tgt: np.ndarray,
                  sel: np.ndarray) -> tuple[float, Params, list]:
    """Def 2's loss, the mean of −log q(x_i | x̃) over the selected positions, with its gradient. Logits are formed
    only at the selected rows (the rest carry no loss); the backward pass is Topic 11's pre-LN branch of
    stack_loss_grad on the forward pass's own caches."""
    xs, caches = encode(P, blocks, tok)
    out, r = tb._ln_forward(xs[-1])
    d = out.shape[-1]
    rows = np.flatnonzero(sel.ravel())
    o = out.reshape(-1, d)[rows]
    lp = nlm.log_softmax(mlm_logits(P, o))
    t, cnt = tgt.ravel()[rows], len(rows)
    loss = float(-lp[np.arange(cnt), t].sum() / cnt)
    g = np.exp(lp)
    g[np.arange(cnt), t] -= 1
    g /= cnt
    V = P["b"].shape[0]
    G = {"b": g.sum(0), "E": np.zeros_like(P["E"])}
    G["E"][:V] = g.T @ o / math.sqrt(d)
    dout = np.zeros((tok.size, d), out.dtype)
    dout[rows] = g @ P["E"][:V] / math.sqrt(d)
    gx = tb.ln_backward(dout.reshape(out.shape), out, r)
    grads = []
    for blk, (y, r1, ca, z, r2, cm) in zip(reversed(blocks), reversed(caches)):
        dz, Gm = mlp_backward(blk["mlp"], gx, cm)
        gx = gx + tb.ln_backward(dz, z, r2)
        dy, Ga = tb.mha_backward(blk["attn"], gx, ca)
        gx = gx + tb.ln_backward(dy, y, r1)
        grads.append({"attn": Ga, "mlp": Gm})
    G["P"] = gx.sum(0)
    N = tok.size  # the token-table scatter as one sparse product (CLAUDE.md's csc pattern)
    G["E"] += sparse.csc_matrix((np.ones(N, out.dtype), tok.ravel(), np.arange(N + 1)),
                                shape=(P["E"].shape[0], N)) @ gx.reshape(-1, d)
    return loss, G, grads[::-1]


def _hmm_batch(rng: np.random.Generator, batch: int, n: int = H_N):
    x, _ = sample_hmm(rng, batch, n)
    xt, sel = corrupt(rng, x, H_VOCAB, H_MASK)
    return xt, x, sel


@dataclass(frozen=True)
class Trained:
    cfg: MLMConfig
    P: dict
    blocks: list
    loss: tuple


@lru_cache(maxsize=None)
def train_mlm(cfg: MLMConfig) -> Trained:
    """Adam (Topic 5's constants) in float32 on fresh corrupted batches, the learning rate decaying linearly to 0;
    one stream per model. HMM: fresh sequences every step. Filings: 32-token windows of one document."""
    rng = np.random.default_rng([SEED, cfg.stream])
    V_in = H_VOCAB + 1 if cfg.source == "hmm" else filing_vocab()[1] + 1
    P, blocks = init_encoder(rng, V_in, cfg)
    leaves = [(P, k) for k in P] + [(blk[s], k) for blk in blocks for s in blk for k in blk[s]]
    m1 = [np.zeros_like(D[k]) for D, k in leaves]
    m2 = [np.zeros_like(D[k]) for D, k in leaves]
    trace = []
    for step in range(1, cfg.steps + 1):
        tok, tgt, sel = _hmm_batch(rng, cfg.batch) if cfg.source == "hmm" else filing_batch(rng, cfg.batch, cfg.n)
        loss, G, Gb = mlm_loss_grad(P, blocks, tok, tgt, sel)
        grads = [G[k] for k in P] + [Gb[i][s][k] for i, blk in enumerate(blocks) for s in blk for k in blk[s]]
        c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
        lr = cfg.lr * (1 - (step - 1) / cfg.steps)
        for j, ((D, k), g) in enumerate(zip(leaves, grads)):
            m1[j] = nlm.ADAM_B1 * m1[j] + (1 - nlm.ADAM_B1) * g
            m2[j] = nlm.ADAM_B2 * m2[j] + (1 - nlm.ADAM_B2) * g**2
            D[k] -= lr * (m1[j] / c1) / (np.sqrt(m2[j] / c2) + nlm.ADAM_EPS)
        if step % 100 == 0:
            trace.append(loss)
    return Trained(cfg, P, blocks, tuple(trace))


H_TEST = 400


@lru_cache(maxsize=1)
def hmm_test_set() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H_TEST fresh corrupted sequences (x̃, x, selected) for every HMM evaluation."""
    return _hmm_batch(np.random.default_rng([SEED, 1530]), H_TEST)


def _kl_rows(q: np.ndarray, lq: np.ndarray) -> np.ndarray:
    return (q * (np.log(np.maximum(q, 1e-300)) - lq)).sum(-1)


@lru_cache(maxsize=1)
def hmm_fit() -> dict:
    """Thms 1–2 measured: at the test set's selected positions, the exact Bayes predictor's cross-entropy (the risk's
    entropy part), the trained model's, and their mean KL, split by what the position shows; and Thm 2's shrinkage
    read off the model itself at visible tokens."""
    xt, x, sel = hmm_test_set()
    m = train_mlm(HMM_MLM)
    q, _ = bayes_mlm(xt)
    lq = mlm_log_probs(m.P, m.blocks, xt).astype(float)
    shown = np.where(xt == H_MASK, 0, np.where(xt == x, 2, 1))  # 0 [MASK], 1 a different token, 2 the original
    kl = _kl_rows(q, lq)
    ce_exact = -np.log(np.take_along_axis(q, x[..., None], -1)[..., 0])
    ce_model = -np.take_along_axis(lq, x[..., None], -1)[..., 0]
    out = {"ce_exact": float(ce_exact[sel].mean()), "ce_model": float(ce_model[sel].mean()),
           "kl": float(kl[sel].mean()), "entropy": float((-(q * np.log(np.maximum(q, 1e-300))).sum(-1))[sel].mean()),
           "n_selected": int(sel.sum())}
    for k, name in enumerate(("mask", "random", "kept")):
        out[f"kl_{name}"] = float(kl[sel & (shown == k)].mean())
        out[f"n_{name}"] = int((sel & (shown == k)).sum())
    # Thm 2 on the model itself: mask each kept position, read p̂ = q(· | [MASK]), and compare the model's q(v) at the
    # visible token with p̂(v)(κ + r/V)/(κ p̂(v) + r/V).
    b_idx, i_idx = np.nonzero(sel & (shown == 2))
    masked = xt[b_idx].copy()
    masked[np.arange(len(b_idx)), i_idx] = H_MASK
    p_hat = np.exp(mlm_log_probs(m.P, m.blocks, masked).astype(float)[np.arange(len(b_idx)), i_idx])
    v = x[b_idx, i_idx]
    pv = p_hat[np.arange(len(b_idx)), v]
    V = H_VOCAB
    pred = pv * (KAPPA + R_RAND / V) / (KAPPA * pv + R_RAND / V)
    got = np.exp(lq[b_idx, i_idx, v])
    out["shrink_pred_mean"], out["shrink_model_mean"] = float(pred.mean()), float(got.mean())
    out["shrink_mae"] = float(np.abs(pred - got).mean())
    out["shrink_masked_mean"] = float(pv.mean())
    return out


# --------------------------------------------------------------------------------------------
# 3. Pseudo-likelihood and the missing joint.
# --------------------------------------------------------------------------------------------

ISING_N, ISING_H, ISING_J = 10, 0.2, 0.5


@lru_cache(maxsize=1)
def ising_states() -> tuple[np.ndarray, np.ndarray]:
    """Every configuration of the open chain in {−1, 1}^10 and its probability ∝ exp(hΣx_i + JΣx_i x_{i+1})."""
    X = np.array(list(itertools.product((-1, 1), repeat=ISING_N)), dtype=float)
    e = ISING_H * X.sum(1) + ISING_J * (X[:, :-1] * X[:, 1:]).sum(1)
    p = np.exp(e - e.max())
    return X, p / p.sum()


def _pl_terms(X: np.ndarray) -> np.ndarray:
    """Each site's sufficient pair (x_i, x_i·(x_{i−1} + x_{i+1})): log p(x_i | x_{−i}) = log σ(2 ⟨θ, that pair⟩)."""
    nb = np.zeros_like(X)
    nb[:, 1:] += X[:, :-1]
    nb[:, :-1] += X[:, 1:]
    return np.stack([X, X * nb], -1)


def log_pl(theta: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """The mean log pseudo-likelihood Σ_i log p_θ(x_i | x_{−i}) of rows X (weights w), its gradient, and Hessian."""
    F = _pl_terms(X)
    z = 2 * F @ theta
    w = np.full(len(X), 1 / len(X)) if w is None else w
    val = float((w[:, None] * w2v.log_sigmoid(z)).sum())
    s = w2v.sigmoid(-z)  # d/dz log σ(z) = σ(−z)
    grad = 2 * np.einsum("b,bi,bik->k", w, s, F)
    hess = -4 * np.einsum("b,bi,bik,bil->kl", w, s * (1 - s), F, F)
    return val, grad, hess


def fit_pl(X: np.ndarray) -> np.ndarray:
    """Maximize the (concave) log pseudo-likelihood by Newton's method from θ = 0."""
    theta = np.zeros(2)
    for _ in range(30):
        _, g, Hs = log_pl(theta, X)
        theta = theta - np.linalg.solve(Hs, g)
    return theta


@lru_cache(maxsize=1)
def ising_fits() -> dict:
    """Thm 3 on the chain: the population gradient of the expected log-PL at the truth, and the PL estimates from
    N = 200 and N = 20000 exact samples (one stream)."""
    X, p = ising_states()
    truth = np.array([ISING_H, ISING_J])
    out = {"pop_grad": float(np.abs(log_pl(truth, X, p)[1]).max())}
    rng = np.random.default_rng([SEED, 1540])
    for N in (200, 20000):
        theta = fit_pl(X[rng.choice(len(X), size=N, p=p)])
        out[f"h_{N}"], out[f"J_{N}"] = float(theta[0]), float(theta[1])
        out[f"err_{N}"] = float(np.abs(theta - truth).max())
    return out


BROOK_N = 4


@lru_cache(maxsize=1)
def hmm_joint(n: int = BROOK_N) -> np.ndarray:
    """The HMM's exact joint over all V^n sequences of length n, as a tensor (positive everywhere)."""
    T, B, pi = hmm()
    a = pi[None, :] * B.T  # (V, K): P(x_1, h_1)
    for _ in range(n - 1):
        a = (a @ T)[..., None, :] * B.T  # (…, V, K)
    return a.sum(-1)


def brook_rebuild(cond: list[np.ndarray], y: tuple[int, ...]) -> np.ndarray:
    """Thm 4: p(x)/p(y) = Π_i p(x_i | x_{<i}, y_{>i}) / p(y_i | x_{<i}, y_{>i}) from the full conditionals alone
    (cond[i] is the joint normalized along axis i), at every x; normalized to a distribution."""
    n = len(cond)
    V = cond[0].shape[0]
    grids = np.meshgrid(*[np.arange(V)] * n, indexing="ij")
    log_ratio = np.zeros(cond[0].shape)
    for i in range(n):
        num = tuple(grids[k] if k <= i else np.full_like(grids[k], y[k]) for k in range(n))
        den = tuple(grids[k] if k < i else np.full_like(grids[k], y[k]) for k in range(n))
        log_ratio += np.log(cond[i][num]) - np.log(cond[i][den])
    r = np.exp(log_ratio - log_ratio.max())
    return r / r.sum()


def cycle_gap(qi: np.ndarray, qj: np.ndarray, a: int, a2: int, b: int, b2: int) -> float:
    """Cor 2's four-cycle log-ratio for positions i, j, with qi[y, b] = q(x_i = y | x_j = b, rest) and qj[y, a] =
    q(x_j = y | x_i = a, rest): log p(a, b) − log p(a', b') along (a, b) → (a', b) → (a', b') minus along
    (a, b) → (a, b') → (a', b'). Zero for the conditionals of any joint."""
    via_i = math.log(qi[a, b] / qi[a2, b]) + math.log(qj[b, a2] / qj[b2, a2])
    via_j = math.log(qj[b, a] / qj[b2, a]) + math.log(qi[a, b2] / qi[a2, b2])
    return via_i - via_j


COMPAT_PAIRS = 200


def _pair_conditionals(kind: str, x: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """qi[y, b] and qj[y, a] for one sequence x, with every value b of x_j (and a of x_i) substituted:
    "clean" — the HMM's own conditionals; "bayes" — Thm 2's Bayes-optimal MLM prediction at a [MASK] (the other
    positions read through the channel); "model" — the trained encoder with x_i (or x_j) masked."""
    V = H_VOCAB
    out = []
    for at, other in ((i, j), (j, i)):
        X = np.repeat(x[None], V, 0)
        X[:, other] = np.arange(V)
        if kind == "clean":
            Q = clean_conditionals(X)[:, at]
        else:
            X[:, at] = H_MASK
            if kind == "bayes":
                Q = bayes_mlm(X)[0][:, at]
            else:
                m = train_mlm(HMM_MLM)
                Q = np.exp(mlm_log_probs(m.P, m.blocks, X).astype(float)[:, at])
        out.append(Q.T)  # [y, value at the other position]
    return out[0], out[1]


@lru_cache(maxsize=1)
def compatibility() -> dict:
    """Cor 2 measured: |δ| over COMPAT_PAIRS random (sequence, adjacent pair, values) from one stream, for the
    HMM's own conditionals, the Bayes-optimal MLM's, and the trained encoder's. The values a, a' (b, b') are the two
    most probable under the clean conditional, so every ratio is between well-estimated entries."""
    rng = np.random.default_rng([SEED, 1550])
    xs, _ = sample_hmm(rng, COMPAT_PAIRS)
    gaps: dict[str, list[float]] = {"clean": [], "bayes": [], "model": []}
    for x in xs:
        i = int(rng.integers(H_N - 1))
        j = i + 1
        qi, qj = _pair_conditionals("clean", x, i, j)
        a, a2 = np.argsort(-qi[:, x[j]])[:2]
        b, b2 = np.argsort(-qj[:, x[i]])[:2]
        for kind in gaps:
            Qi, Qj = (qi, qj) if kind == "clean" else _pair_conditionals(kind, x, i, j)
            gaps[kind].append(cycle_gap(Qi, Qj, int(a), int(a2), int(b), int(b2)))
    out = {}
    for kind, g in gaps.items():
        g = np.abs(g)
        out[f"{kind}_median"], out[f"{kind}_mean"], out[f"{kind}_max"] = (float(np.median(g)), float(g.mean()),
                                                                          float(g.max()))
    out["gaps"] = {k: v for k, v in gaps.items()}
    return out


# --------------------------------------------------------------------------------------------
# 4–5. Geometry and probing.
# --------------------------------------------------------------------------------------------


def self_similarity(Hm: np.ndarray) -> float:
    """Def 3 (Ethayarajh): the mean cosine between distinct occurrences' vectors (rows of Hm)."""
    U = Hm / np.linalg.norm(Hm, axis=1, keepdims=True)
    S = U @ U.T
    k = len(U)
    return float((S.sum() - np.trace(S)) / (k * (k - 1)))


def mev(Hm: np.ndarray) -> float:
    """Def 3 / Prop 2: σ₁²/‖H‖_F², the share of the occurrence matrix's energy one direction can carry."""
    s = np.linalg.svd(Hm, compute_uv=False)
    return float(s[0] ** 2 / (s**2).sum())


def anisotropy(Hm: np.ndarray, rng: np.random.Generator, pairs: int = 4000) -> float:
    """Def 3's baseline (Ethayarajh's eq. 4): the mean cosine between random pairs of occurrence vectors of any
    words. SelfSim − this baseline is the anisotropy-adjusted self-similarity."""
    U = Hm / np.linalg.norm(Hm, axis=1, keepdims=True)
    i, j = rng.integers(len(U), size=(2, pairs))
    keep = i != j
    return float((U[i[keep]] * U[j[keep]]).sum(1).mean())


PROBE_LAMBDA = 0.3  # ridge strength of every probe, fixed before the final runs (never tuned on a test split)


def probe_fit(X: np.ndarray, y: np.ndarray, lam: float = PROBE_LAMBDA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A linear probe: ridge-penalized logistic regression on standardized features, by Newton's method. Returns
    (weights with intercept last, mean, scale)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(25):
        p = w2v.sigmoid(Z @ w)
        g = Z.T @ (p - y) / len(y) + lam * w
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + lam * np.eye(len(w))
        w = w - np.linalg.solve(Hs, g)
    return w, mu, sd


def probe_predict(fit, X: np.ndarray) -> np.ndarray:
    w, mu, sd = fit
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w > 0


def cv_accuracy(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, folds: int = 5) -> float:
    """k-fold cross-validated probe accuracy (one permutation from rng)."""
    perm = rng.permutation(len(y))
    hits = 0
    for f in np.array_split(perm, folds):
        tr = np.setdiff1d(perm, f)
        hits += int((probe_predict(probe_fit(X[tr], y[tr]), X[f]) == y[f]).sum())
    return hits / len(y)


def binom_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


H_PROBE_TRAIN, H_PROBE_TEST = 1500, 1500


@lru_cache(maxsize=1)
def hmm_occurrences() -> dict:
    """The polyseme's occurrences in clean training and test streams: their vectors at layers 0–2 (the stream before
    each block, and after the last), the true senses, and the exact sense posteriors."""
    m = train_mlm(HMM_MLM)
    out = {}
    for split, stream, size in (("train", 1560, H_PROBE_TRAIN), ("test", 1561, H_PROBE_TEST)):
        x, h = sample_hmm(np.random.default_rng([SEED, stream]), size)
        xs, _ = encode(m.P, m.blocks, x)
        occ = x == POLYSEME
        out[split] = {"layers": [np.asarray(v[occ], float) for v in xs], "sense": h[occ] == sense_states()[0],
                      "posterior": sense_posterior(x)[occ], "x": x, "occ": occ}
    return out


@lru_cache(maxsize=1)
def hmm_probe() -> dict:
    """§5 on the HMM: the probe trained on one stream and tested on another, layer by layer, against the static cap
    max_k π_k (Cor 1) and the exact Bayes accuracy E[max_k P(sense k | x)]; plus a shuffled-label control."""
    occ = hmm_occurrences()
    tr, te = occ["train"], occ["test"]
    share = float(te["sense"].mean())
    out = {"n_test": int(len(te["sense"])), "static_cap": max(share, 1 - share),
           "bayes": float(np.maximum(te["posterior"], 1 - te["posterior"]).mean()),
           "bayes_rule_acc": float(((te["posterior"] > 0.5) == te["sense"]).mean())}
    rng = np.random.default_rng([SEED, 1562])
    for layer in range(len(tr["layers"])):
        fit = probe_fit(tr["layers"][layer], tr["sense"])
        out[f"acc_{layer}"] = float((probe_predict(fit, te["layers"][layer]) == te["sense"]).mean())
        ctl = probe_fit(tr["layers"][layer], rng.permutation(tr["sense"]))  # scored against shuffled test labels too
        out[f"control_{layer}"] = float((probe_predict(ctl, te["layers"][layer]) == rng.permutation(te["sense"])).mean())
    # the token table alone (the static map E[x]): one constant vector, so the probe learns one label for all
    E = np.asarray(train_mlm(HMM_MLM).P["E"][POLYSEME], float)
    fit = probe_fit(np.tile(E, (len(tr["sense"]), 1)), tr["sense"])
    out["acc_static"] = float((probe_predict(fit, np.tile(E, (len(te["sense"]), 1))) == te["sense"]).mean())
    return out


@lru_cache(maxsize=1)
def hmm_geometry() -> dict:
    """§4 on the HMM: the polyseme's self-similarity and MEV by layer (test stream), with Ethayarajh's baselines over
    every token's vectors: the mean cosine of random pairs, and the MEV of 1000 random vectors."""
    m = train_mlm(HMM_MLM)
    te = hmm_occurrences()["test"]
    xs, _ = encode(m.P, m.blocks, te["x"][:300])
    rng = np.random.default_rng([SEED, 1563])
    out = {}
    for layer, Hm in enumerate(te["layers"]):
        allv = np.asarray(xs[layer], float).reshape(-1, xs[layer].shape[-1])
        out[f"self_sim_{layer}"] = self_similarity(Hm)
        out[f"mev_{layer}"] = mev(Hm)
        out[f"aniso_{layer}"] = anisotropy(allv, rng)
        out[f"mev_base_{layer}"] = mev(allv[rng.choice(len(allv), ANISO_SAMPLE // 2, replace=False)])
        out[f"self_sim_adj_{layer}"] = out[f"self_sim_{layer}"] - out[f"aniso_{layer}"]
    return out


# --------------------------------------------------------------------------------------------
# 6. Finance case study: the filing encoder.
# --------------------------------------------------------------------------------------------

FILING_MIN_COUNT = 5


@lru_cache(maxsize=1)
def filing_vocab() -> tuple[dict[str, int], int]:
    """The training types seen at least FILING_MIN_COUNT times (Topic 6's vocabulary of generate_corpus(300, 0)), then
    [UNK] and [SEP] (between sentences): (word → id, V). The input-only [MASK] is id V."""
    words = [w for w, c in zip(ds.vocab(), ds.unigram_counts()) if c >= FILING_MIN_COUNT]
    return {w: i for i, w in enumerate(words)}, len(words) + 2


def _token_offsets(text: str) -> list[re.Match]:
    return list(re.finditer(r"[a-z]+", text.lower()))  # fc._tokens, with character offsets


@lru_cache(maxsize=1)
def filing_docs() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Each training document as ids with [SEP] after every sentence, and each id's word position in the document's
    token list (−1 for [SEP]). Its tokens are Topic 6's sentences, asserted equal to fc._tokens of the text."""
    ids, V = filing_vocab()
    unk, sep = V - 2, V - 1
    out = []
    for d, sents in zip(ds._generator_run()[0], ds.documents()):
        assert [t.group() for t in _token_offsets(d.text)] == [w for s in sents for w in s]
        seq, pos, j = [], [], 0
        for s in sents:
            for w in s:
                seq.append(ids.get(w, unk))
                pos.append(j)
                j += 1
            seq.append(sep)
            pos.append(-1)
        out.append((np.array(seq), np.array(pos)))
    return tuple(out)


@lru_cache(maxsize=1)
def filing_stream() -> tuple[np.ndarray, np.ndarray]:
    """The documents concatenated into one id stream (training windows may cross a document boundary, so every token
    can be read at a window's center), and each word token's index in it."""
    seq = np.concatenate([d[0] for d in filing_docs()])
    return seq, np.flatnonzero(np.concatenate([d[1] for d in filing_docs()]) >= 0)


def filing_batch(rng: np.random.Generator, batch: int, n: int):
    """batch windows of n consecutive ids of the stream at uniform starts, corrupted by Def 2."""
    seq, _ = filing_stream()
    _, V = filing_vocab()
    st = rng.integers(0, len(seq) - n + 1, size=batch)
    x = seq[st[:, None] + np.arange(n)]
    xt, sel = corrupt(rng, x, V, V)
    return xt, x, sel


ANISO_SAMPLE = 2000


@lru_cache(maxsize=1)
def analysis_tokens() -> np.ndarray:
    """The word tokens §§4 and 7 read: every alias, lexicon, and news-labeled occurrence, and ANISO_SAMPLE tokens drawn
    uniformly (one stream) for the anisotropy baseline. Indices into filing_tokens(), sorted."""
    words, _, _ = filing_tokens()
    pw = ds.polarity_words()
    lex = set(pw["positive"]) | set(pw["negative"])
    keep = {k for k, w in enumerate(words) if w in ALIASES or w in lex} | set(news_labels())
    keep |= set(np.random.default_rng([SEED, 1592]).choice(len(words), ANISO_SAMPLE, replace=False).tolist())
    return np.array(sorted(keep))


@lru_cache(maxsize=1)
def filing_vectors() -> dict:
    """The analysis tokens' vectors at layers 0–2 (clean input), each read at the center of its own window of n — so
    the position term is the same for every token, and layer 0, E[x] + P[n/2], is a static map. The stream is padded
    with [SEP] for the few tokens within n/2 of either end. {"index": token → row, "layers": [(rows, d)] × 3}."""
    m = train_mlm(FILING_MLM)
    n, half = FILING_MLM.n, FILING_MLM.n // 2
    seq, word_at = filing_stream()
    _, V = filing_vocab()
    pad = np.concatenate([np.full(half, V - 1), seq, np.full(half, V - 1)])
    toks = analysis_tokens()
    starts = word_at[toks]  # in the padded stream, the window [t, t + n) has stream token t at position n/2
    layers = [np.empty((len(toks), FILING_MLM.d)) for _ in range(FILING_MLM.layers + 1)]
    for lo in range(0, len(toks), 512):
        W = pad[starts[lo: lo + 512, None] + np.arange(n)]
        xs, _ = encode(m.P, m.blocks, W)
        for layer, x in enumerate(xs):
            layers[layer][lo: lo + 512] = x[:, half]
    return {"index": {int(t): i for i, t in enumerate(toks)}, "layers": layers}


def _rows(tokens) -> list[np.ndarray]:
    fv = filing_vectors()
    r = [fv["index"][int(t)] for t in tokens]
    return [L[r] for L in fv["layers"]]


@lru_cache(maxsize=1)
def filing_tokens() -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """Every word token in document order: (word, entity type or "", the document's tone)."""
    words, types, tones = [], [], []
    for d in ds._generator_run()[0]:
        toks = _token_offsets(d.text)
        et = [""] * len(toks)
        for e in d.entities:
            for j, t in enumerate(toks):
                if t.start() >= e.start and t.end() <= e.end:
                    et[j] = e.type
        words += [t.group() for t in toks]
        types += et
        tones += [d.tone] * len(toks)
    return tuple(words), tuple(types), tuple(tones)


DIRECTION_WORDS = ("rose", "increased", "climbed", "fell", "declined", "decreased")
EVAL_VERBS = ("exceeded", "beat", "surpassed", "outperformed", "missed", "trailed", "undershot")


@lru_cache(maxsize=1)
def news_labels() -> dict[int, bool]:
    """Token index → good news?, for every direction word or evaluative verb that reports a gold aspect: the first such
    word within six tokens after a non-neutral aspect term carries that aspect's polarity (fc's aspect gold: "costs
    rose" is −1, "revenue rose" +1)."""
    targets = set(DIRECTION_WORDS) | set(EVAL_VERBS)
    out, base = {}, 0
    for d in ds._generator_run()[0]:
        toks = _token_offsets(d.text)
        for a in d.aspects:
            if a.polarity == 0:
                continue
            j = next(k for k, t in enumerate(toks) if t.start() >= a.end)
            for k in range(j, min(j + 6, len(toks))):
                if toks[k].group() in targets:
                    out[base + k] = a.polarity > 0
                    break
        base += len(toks)
    return out


SGNS_BASELINE = w2v.SGNSConfig("L2", 25)  # Topic 7's d = 25 grid model with the ±2 window: a local static baseline


def _static_rows(X: np.ndarray, words) -> np.ndarray:
    idx = ds.index()
    return np.stack([X[idx[w]] for w in words])


ALIASES = ("kestrel", "marrow", "ostrander", "stratus", "kingsbridge", "ridge", "pellucid", "quillon", "tamsin",
           "saltmarsh", "orrin", "larkspur", "fairhaven", "copperleaf", "juniper", "solenne", "verity", "calloway",
           "meridale", "redfern", "northgate")


@lru_cache(maxsize=1)
def alias_typing() -> dict:
    """§7(a): each company alias that the gold tags both ORG and PRODUCT. Leave-one-alias-out: a probe trained on the
    other aliases' occurrences types every occurrence of the held-out one, from SGNS (Topic 7), the token table E, or
    the stream at layers 0–2 (layer 0 is static: E[x] + P[n/2]); pooled over aliases against the per-alias majority.
    And kestrel alone (57 ORG, 61 PRODUCT) by 5-fold CV."""
    words, types, _ = filing_tokens()
    occ = np.array([k for k, (w, t) in enumerate(zip(words, types)) if w in ALIASES and t in ("ORG", "PRODUCT")])
    L = _rows(occ)
    al = np.array([words[k] for k in occ])
    y = np.array([types[k] == "PRODUCT" for k in occ])
    reps = {"sgns": _static_rows(w2v.sgns_vectors(SGNS_BASELINE), al), **{f"layer{l}": L[l] for l in range(3)}}
    out = {"n": int(len(occ)), "n_product": int(y.sum()),
           "majority": float(sum(max(y[al == a].mean(), 1 - y[al == a].mean()) * (al == a).sum() for a in ALIASES)
                             / len(occ))}
    for name, X in reps.items():
        hits = 0
        for a in ALIASES:
            te = al == a
            hits += int((probe_predict(probe_fit(X[~te], y[~te]), X[te]) == y[te]).sum())
        out[f"loao_{name}"] = hits / len(occ)
    kk = al == "kestrel"
    out["kestrel_n"], out["kestrel_product"] = int(kk.sum()), int(y[kk].sum())
    out["kestrel_majority"] = float(max(y[kk].mean(), 1 - y[kk].mean()))
    rng = np.random.default_rng([SEED, 1570])
    for l in range(3):
        out[f"kestrel_cv_layer{l}"] = cv_accuracy(L[l][kk], y[kk], rng)
    return out


@lru_cache(maxsize=1)
def news_polarity() -> dict:
    """§7(b): the news a direction word or evaluative verb reports. Per word, 5-fold CV of a probe on its occurrences
    at each layer, against Cor 1's static cap (the word's majority share — what SGNS or the token table, one vector per
    word, can reach at best); pooled over words with ≥ 20 labeled occurrences; with a shuffled-label control."""
    lab = news_labels()
    words, _, _ = filing_tokens()
    rng = np.random.default_rng([SEED, 1580])
    per = {}
    for w in (*DIRECTION_WORDS, *EVAL_VERBS):
        k = np.array([t for t in sorted(lab) if words[t] == w])
        if len(k) < 20:
            continue
        L = _rows(k)
        y = np.array([lab[t] for t in k])
        per[w] = {"n": int(len(k)), "good": int(y.sum()), "cap": float(max(y.mean(), 1 - y.mean())),
                  **{f"layer{l}": cv_accuracy(L[l], y, rng) for l in range(3)},
                  "control1": cv_accuracy(L[1], rng.permutation(y), rng)}
    n = sum(v["n"] for v in per.values())
    out = {"words": per, "n": n, "cap": sum(v["cap"] * v["n"] for v in per.values()) / n}
    for key in ("layer0", "layer1", "layer2", "control1"):
        out[key] = sum(v[key] * v["n"] for v in per.values()) / n
    return out


@lru_cache(maxsize=1)
def strong_weak() -> dict:
    """§7(b), Topic 7's question: its leave-one-out polarity probe (w2v.polarity_probe) on each lexicon word's mean
    vector at layers 0–2 (layer 0 is the static table), beside SGNS; and how much the text carries — for each
    lexicon occurrence, the vote of the other lexicon words within ±8 tokens, ±32, and the whole document."""
    pw = ds.polarity_words()
    pos, neg = pw["positive"], pw["negative"]
    words, _, tones = filing_tokens()
    idx = ds.index()
    by_word: dict[str, list[int]] = {}
    for k, w in enumerate(words):
        if w in pos or w in neg:
            by_word.setdefault(w, []).append(k)
    out = {"n_words": len(pos) + len(neg), "chance_se": math.sqrt(0.25 / (len(pos) + len(neg))),
           "sgns": w2v.polarity_probe(w2v.sgns_vectors(SGNS_BASELINE), pos, neg)}
    means = {w: [L.mean(0) for L in _rows(ks)] for w, ks in by_word.items()}
    for l in range(3):
        X = np.zeros((len(idx), FILING_MLM.d))
        for w, mv in means.items():
            X[idx[w]] = mv[l]
        out[f"layer{l}"] = w2v.polarity_probe(X, pos, neg)
    pol = np.array([(w in pos) - (w in neg) for w in words])
    bounds, base = [], 0
    for d in ds._generator_run()[0]:
        nt = len(_token_offsets(d.text))
        bounds.append((base, base + nt))
        base += nt
    for half, name in ((8, "vote8"), (32, "vote32"), (10**9, "vote_doc")):
        hits = []
        for lo, hi in bounds:
            for k in range(lo, hi):
                if pol[k] == 0:
                    continue
                v = pol[max(lo, k - half): k].sum() + pol[k + 1: min(hi, k + 1 + half)].sum()
                hits.append(0.5 if v == 0 else float((v > 0) == (pol[k] > 0)))
        out[name] = float(np.mean(hits))
    out["n_occurrences"] = int((pol != 0).sum())
    toned = [k for k in range(len(words)) if pol[k] != 0 and tones[k] != 0]
    out["word_matches_tone"] = float(np.mean([(pol[k] > 0) == (tones[k] > 0) for k in toned]))
    return out


@lru_cache(maxsize=1)
def filing_geometry() -> dict:
    """§4 on the filings: kestrel's self-similarity and MEV by layer, beside Ethayarajh's baselines over ANISO_SAMPLE
    uniformly drawn word tokens (the MEV baseline from the first half of them)."""
    words, _, _ = filing_tokens()
    K = _rows([k for k, w in enumerate(words) if w == "kestrel"])
    A = _rows(np.random.default_rng([SEED, 1592]).choice(len(words), ANISO_SAMPLE, replace=False))
    rng = np.random.default_rng([SEED, 1590])
    out = {}
    for l in range(3):
        out[f"self_sim_{l}"], out[f"mev_{l}"] = self_similarity(K[l]), mev(K[l])
        out[f"aniso_{l}"] = anisotropy(A[l], rng)
        out[f"mev_base_{l}"] = mev(A[l][: ANISO_SAMPLE // 2])
        out[f"self_sim_adj_{l}"] = out[f"self_sim_{l}"] - out[f"aniso_{l}"]
    return out


@lru_cache(maxsize=1)
def filing_fit() -> dict:
    """The filing encoder's masked-token cross-entropy on 200 fresh corrupted windows (one stream), beside the
    unigram entropy of the same targets."""
    m = train_mlm(FILING_MLM)
    xt, x, sel = filing_batch(np.random.default_rng([SEED, 1591]), 200, FILING_MLM.n)
    lq = mlm_log_probs(m.P, m.blocks, xt).astype(float)
    ce = -np.take_along_axis(lq, x[..., None], -1)[..., 0][sel]
    cnt = Counter(filing_stream()[0].tolist())
    tot = sum(cnt.values())
    uni = -np.array([math.log(cnt[int(t)] / tot) for t in x[sel]])
    return {"ce": float(ce.mean()), "unigram_ce": float(uni.mean()), "n": int(sel.sum())}


# --------------------------------------------------------------------------------------------
# Summary and the lab's data.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def summary() -> dict:
    out = {}
    for prefix, d in (("hmm", hmm_fit()), ("ising", ising_fits()), ("probe", hmm_probe()), ("geom", hmm_geometry()),
                      ("alias", alias_typing()), ("fgeom", filing_geometry()), ("ffit", filing_fit())):
        out.update({f"{prefix}_{k}": v for k, v in d.items() if isinstance(v, (int, float))})
    out.update({f"compat_{k}": v for k, v in compatibility().items() if isinstance(v, float)})
    out.update({f"news_{k}": v for k, v in news_polarity().items() if isinstance(v, (int, float))})
    out.update({f"sw_{k}": v for k, v in strong_weak().items() if isinstance(v, (int, float))})
    return out


def _r(x, nd: int = 3):
    return ak._r(x, nd)


LAB_STREAM = 1595


@lru_cache(maxsize=1)
def lab_hmm_sequence() -> np.ndarray:
    """Figure A's HMM sentence: the first of a fresh stream's sequences holding the polyseme in both senses."""
    rng = np.random.default_rng([SEED, LAB_STREAM])
    while True:
        x, h = sample_hmm(rng, 1)
        senses = set(h[0][x[0] == POLYSEME].tolist())
        if len(senses) == 2:
            return x[0]


def _model_probs(m: Trained, X: np.ndarray) -> np.ndarray:
    return np.exp(mlm_log_probs(m.P, m.blocks, X).astype(float))


@lru_cache(maxsize=1)
def lab_hmm() -> dict:
    """Figure A, HMM: at every position, what the exact Bayes predictor (Thm 2) and the encoder put on each token when
    the position shows [MASK], its own token, or a substitute — the token the exact [MASK] prediction ranks second
    after the true one (or first, if the true one is not first). The rest of the sentence is clean."""
    x = lab_hmm_sequence()
    m = train_mlm(HMM_MLM)
    n = len(x)
    masked = np.repeat(x[None], n, 0)
    masked[np.arange(n), np.arange(n)] = H_MASK
    p_mask = bayes_mlm(masked)[0][np.arange(n), np.arange(n)]
    sub = np.array([int(np.argsort(-p)[1] if np.argmax(p) == x[i] else np.argmax(p)) for i, p in enumerate(p_mask)])
    views = {}
    for mode, shown in (("mask", np.full(n, H_MASK)), ("kept", x), ("random", sub)):
        X = np.repeat(x[None], n, 0)
        X[np.arange(n), np.arange(n)] = shown
        q = bayes_mlm(X)[0][np.arange(n), np.arange(n)]
        qm = _model_probs(m, X)[np.arange(n), np.arange(n)]
        views[mode] = {"exact": [_r(v) for v in q], "model": [_r(v) for v in qm],
                       "kl": _r(_kl_rows(q, np.log(qm)))}
    post = sense_posterior(x[None])[0]
    return {"tokens": x.tolist(), "names": list(H_NAMES), "substitute": sub.tolist(), "views": views,
            "sense_a": [None if x[i] != POLYSEME else _r(post[i]) for i in range(n)]}


LAB_TOP = 5


def _id_names() -> list[str]:
    ids, V = filing_vocab()
    names = [""] * V
    for w, i in ids.items():
        names[i] = w
    names[V - 2], names[V - 1] = "[UNK]", "[SEP]"
    return names


@lru_cache(maxsize=1)
def lab_windows() -> list[dict]:
    """Figure A, filings: four windows of n, each centered (as in filing_vectors) on a chosen token — the first
    PRODUCT kestrel, the first ORG kestrel followed by "turbine", the first "rose" reporting bad news, and the first
    reporting good news. At every position: the encoder's top LAB_TOP words with that position masked, and with it
    visible, the model's probability of the visible word beside Thm 2's prediction from its own [MASK] guess."""
    words, types, _ = filing_tokens()
    lab = news_labels()
    picks = [next(k for k, w in enumerate(words) if w == "kestrel" and types[k] == "PRODUCT"),
             next(k for k, w in enumerate(words) if w == "kestrel" and types[k] == "ORG" and words[k + 1] == "turbine"),
             next(k for k in sorted(lab) if words[k] == "rose" and not lab[k]),
             next(k for k in sorted(lab) if words[k] == "rose" and lab[k])]
    kinds = ("kestrel, PRODUCT", "kestrel, ORG", "rose, bad news", "rose, good news")
    m = train_mlm(FILING_MLM)
    n, half = FILING_MLM.n, FILING_MLM.n // 2
    seq, word_at = filing_stream()
    _, V = filing_vocab()
    names = _id_names()
    out = []
    for k, kind in zip(picks, kinds):
        t = word_at[k]
        x = seq[t - half: t + half]
        masked = np.repeat(x[None], n, 0)
        masked[np.arange(n), np.arange(n)] = V
        pm = _model_probs(m, masked)[np.arange(n), np.arange(n)]
        pk = _model_probs(m, x[None])[0][np.arange(n), x]
        pv = pm[np.arange(n), x]
        formula = pv * (KAPPA + R_RAND / V) / (KAPPA * pv + R_RAND / V)
        top = np.argsort(-pm, 1)[:, :LAB_TOP]
        out.append({"kind": kind, "center": half, "words": [names[i] for i in x],
                    "top": [[names[j] for j in row] for row in top],
                    "top_p": [[_r(pm[i, j]) for j in row] for i, row in enumerate(top)],
                    "masked_p": [_r(v) for v in pv], "kept_p": [_r(v) for v in pk], "formula_p": [_r(v) for v in formula]})
    return out


def _pca2(Hm: np.ndarray, flip_by: np.ndarray) -> np.ndarray:
    """The rows' projections on their top two principal axes, each axis signed so that the rows where flip_by holds
    have a negative mean (a deterministic orientation)."""
    C = Hm - Hm.mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    Y = C @ Vt[:2].T
    return Y * np.where(Y[flip_by].mean(0) > 0, -1, 1)


LAB_POINTS = 150


@lru_cache(maxsize=1)
def lab_clouds() -> dict:
    """Figure B: the polyseme's first LAB_POINTS test occurrences and every kestrel, projected on their own top two
    principal axes at each layer, with their senses and (HMM) exact sense posteriors."""
    te = hmm_occurrences()["test"]
    sense = te["sense"][:LAB_POINTS]
    hmm_xy = [_pca2(L[:LAB_POINTS], sense) for L in te["layers"]]
    words, types, _ = filing_tokens()
    kk = [k for k, w in enumerate(words) if w == "kestrel"]
    prod = np.array([types[k] == "PRODUCT" for k in kk])
    K = _rows(kk)
    return {"hmm": {"xy": [_r(v.ravel()) for v in hmm_xy], "sense_a": sense.astype(int).tolist(),
                    "posterior": _r(te["posterior"][:LAB_POINTS])},
            "kestrel": {"xy": [_r(_pca2(L, prod).ravel()) if np.linalg.matrix_rank(L - L.mean(0)) > 0 else
                               [0.0] * (2 * len(kk)) for L in K], "product": prod.astype(int).tolist()}}


@lru_cache(maxsize=None)
def viz_payload() -> dict:
    """Every number the lab shows, rounded to the page's precision."""
    hp, hg, af, fg = hmm_probe(), hmm_geometry(), alias_typing(), filing_geometry()
    npol, sw, comp, hf = news_polarity(), strong_weak(), compatibility(), hmm_fit()
    layers = range(HMM_MLM.layers + 1)
    return {
        "channel": {"rho": RHO, "mask": M_MASK, "random": R_RAND, "kept": KAPPA, "hmm_vocab": H_VOCAB,
                    "filing_vocab": filing_vocab()[1]},
        "hmm": {**lab_hmm(), "fit": {k: _r(hf[k]) for k in ("kl", "kl_mask", "kl_random", "kl_kept", "entropy",
                                                            "shrink_pred_mean", "shrink_model_mean",
                                                            "shrink_masked_mean")}},
        "filings": lab_windows(),
        "compat": {k: [max(-12.0, _r(math.log10(max(abs(g), 1e-300)), 2)) for g in comp["gaps"][k]]
                   for k in ("clean", "bayes", "model")},
        "compat_median": {k: float(f"{comp[k + '_median']:.3g}") for k in ("clean", "bayes", "model")},
        "clouds": lab_clouds(),
        "hmm_layers": [{"self_sim": _r(hg[f"self_sim_{l}"]), "mev": _r(hg[f"mev_{l}"]), "aniso": _r(hg[f"aniso_{l}"]),
                        "adj": _r(hg[f"self_sim_adj_{l}"]), "mev_base": _r(hg[f"mev_base_{l}"]),
                        "probe": _r(hp[f"acc_{l}"])} for l in layers],
        "hmm_probe": {"static_cap": _r(hp["static_cap"]), "bayes": _r(hp["bayes"]), "n": hp["n_test"]},
        "kestrel_layers": [{"self_sim": _r(fg[f"self_sim_{l}"]), "mev": _r(fg[f"mev_{l}"]),
                            "aniso": _r(fg[f"aniso_{l}"]), "adj": _r(fg[f"self_sim_adj_{l}"]),
                            "mev_base": _r(fg[f"mev_base_{l}"]), "probe": _r(af[f"kestrel_cv_layer{l}"])}
                           for l in layers],
        "alias": {"majority": _r(af["majority"]), "kestrel_majority": _r(af["kestrel_majority"]), "n": af["n"],
                  **{k: _r(af[f"loao_{k}"]) for k in ("sgns", "layer0", "layer1", "layer2")}},
        "news": {"cap": _r(npol["cap"]), "control": _r(npol["control1"]), "n": npol["n"],
                 **{f"layer{l}": _r(npol[f"layer{l}"]) for l in layers},
                 "words": {w: {"n": v["n"], "good": v["good"], "cap": _r(v["cap"]),
                               **{f"layer{l}": _r(v[f"layer{l}"]) for l in layers}}
                           for w, v in npol["words"].items()}},
        "strong_weak": {k: _r(sw[k]) for k in ("sgns", "layer0", "layer1", "layer2", "chance_se", "vote8", "vote32",
                                               "vote_doc")},
    }


# --------------------------------------------------------------------------------------------
# The harness: one test per claim on the page.
# --------------------------------------------------------------------------------------------


def _small_config() -> MLMConfig:
    return MLMConfig("hmm", 6, 16, 2, 2, steps=0, batch=3)


def test_mixture_identity():
    """Prop 1, exactly on the HMM: the polyseme's type-level PMI with the next token is log Σ_k π_k exp PMI(s_k, c),
    between max_k(log π_k + PMI_k) and max_k PMI_k."""
    q = neighbor_pmi()
    mix = np.log((q["share"][:, None] * np.exp(q["senses"])).sum(0))
    assert np.abs(q["type"] - mix).max() < 1e-12
    lo = (np.log(q["share"])[:, None] + q["senses"]).max(0)
    hi = q["senses"].max(0)
    assert np.all(lo <= q["type"] + 1e-12) and np.all(q["type"] <= hi + 1e-12)
    assert np.abs(q["senses"][0] - q["senses"][1]).max() > 2  # the senses really disagree about the neighbors


def test_static_cap():
    """Cor 1: a static map gives every occurrence of a word one vector, so one prediction; the token table's sense
    probe on the HMM sits at a sense's share, never above the majority, and the contextual layers beat it."""
    r = hmm_probe()
    assert r["acc_static"] <= r["static_cap"] + 1e-12
    assert r["static_cap"] < 0.6 and r["acc_2"] > r["static_cap"] + 0.3


def test_proper_scoring():
    """Thm 1: E_p[−log q] = H(p) + KL(p‖q) for random p, q, minimized only at q = p; on the HMM the exact Bayes
    predictor's held-out cross-entropy is below the trained encoder's by its mean KL (Monte Carlo agreement)."""
    rng = np.random.default_rng([SEED, 1501])
    p, q = rng.dirichlet(np.ones(9), size=2)
    ce = -(p * np.log(q)).sum()
    assert abs(ce - (-(p * np.log(p)).sum() + (p * np.log(p / q)).sum())) < 1e-12 and ce > -(p * np.log(p)).sum()
    f = hmm_fit()
    assert f["ce_exact"] < f["ce_model"]
    assert abs((f["ce_model"] - f["ce_exact"]) - f["kl"]) < 0.02


def _enumerated_posterior(xt: np.ndarray, i: int, rho: float) -> np.ndarray:
    """P(x_i = y | x̃, i ∈ M) by brute force over the enumerated joint of a length-3 HMM sequence, with the selection
    of i explicit and the channel at the other positions: no use of Thm 2."""
    p = hmm_joint(3)
    C = channel(rho)
    S = selected_channel(H_VOCAB)
    w = p.copy()
    for j in range(3):
        f = rho * S[:, xt[j]] if j == i else C[:, xt[j]]
        shape = [1, 1, 1]
        shape[j] = H_VOCAB
        w = w * f.reshape(shape)
    post = w.sum(tuple(k for k in range(3) if k != i))
    return post / post.sum()


def test_bayes_80_10_10():
    """Thm 2 against brute force: the closed form q(y) ∝ p(y)(κ1[y = v] + r/V) at a visible token and q = p at a
    [MASK], for two selection rates (ρ drops out of the formula, only p moves); and the corollaries: r = 0 copies
    the visible token, κ = 0 ignores it."""
    rng = np.random.default_rng([SEED, 1502])
    for rho in (RHO, 0.4):
        for _ in range(6):
            xt = rng.integers(H_VOCAB + 1, size=3)
            i = int(rng.integers(3))
            q, _ = bayes_mlm(xt[None], rho=rho)
            assert np.abs(q[0, i] - _enumerated_posterior(xt, i, rho)).max() < 1e-12
    x = np.array([[3, 7, 0, 9]])
    q0, p0 = bayes_mlm(x, r=0.0, kappa=0.2, m=0.8)
    assert np.allclose(q0[0, 2], np.eye(H_VOCAB)[0], atol=1e-12)
    qk, pk = bayes_mlm(x, r=0.2, kappa=0.0, m=0.8)
    assert np.abs(qk - pk).max() < 1e-12


def test_fast_mlp_backward():
    """The matmul MLP backward equals Topic 11's einsum version to rounding."""
    rng = np.random.default_rng([SEED, 1503])
    blk = tb.init_block(rng, 12, 3, 4, 20)
    X = rng.normal(size=(2, 5, 12))
    _, cache = tb.mlp_forward(blk["mlp"], X)
    dO = rng.normal(size=(2, 5, 12))
    a, Ga = mlp_backward(blk["mlp"], dO, cache)
    b, Gb = tb.mlp_backward(blk["mlp"], dO, cache)
    assert np.abs(a - b).max() < 1e-12 and all(np.abs(Ga[k] - Gb[k]).max() < 1e-12 for k in Ga)


def test_mlm_gradients():
    """The encoder's hand-written gradient against Topic 5's central differences (float64, a small encoder), and the
    selected-rows loss against the plain forward pass's full log-probabilities."""
    rng = np.random.default_rng([SEED, 1504])
    cfg = _small_config()
    P, blocks = init_encoder(rng, 12, cfg, dtype=np.float64)
    tok = rng.integers(12, size=(cfg.batch, cfg.n))
    tgt = rng.integers(11, size=tok.shape)
    sel = rng.random(tok.shape) < 0.5
    loss, G, Gb = mlm_loss_grad(P, blocks, tok, tgt, sel)
    lp = mlm_log_probs(P, blocks, tok)
    assert abs(loss - float(-np.take_along_axis(lp, tgt[..., None], -1)[..., 0][sel].mean())) < 1e-12
    worst = 0.0
    leaves = [(P, G, k) for k in P] + [(blocks[i][s], Gb[i][s], k) for i in range(cfg.layers) for s in blocks[i]
                                       for k in blocks[i][s]]
    for D, GD, k in leaves:
        for flat in rng.choice(D[k].size, size=min(6, D[k].size), replace=False):
            ix = np.unravel_index(flat, D[k].shape)
            old = D[k][ix]
            vals = []
            for e in (1e-6, -1e-6):
                D[k][ix] = old + e
                vals.append(mlm_loss_grad(P, blocks, tok, tgt, sel)[0])
            D[k][ix] = old
            worst = max(worst, abs((vals[0] - vals[1]) / 2e-6 - GD[k][ix]))
    assert worst < 1e-7, worst


def test_hmm_mlm_matches_bayes():
    """Thms 1–2 on the trained encoder: its mean KL to the exact Bayes posterior is under 2% of the entropy, and under
    0.1 nats at [MASK]s, at random tokens, and at kept tokens alike; and at a kept token its probability of the
    visible token tracks Thm 2's shrinkage of its own [MASK] prediction, far above that prediction."""
    f = hmm_fit()
    assert f["kl"] < 0.02 * f["entropy"]
    assert all(f[f"kl_{k}"] < 0.1 for k in ("mask", "random", "kept"))
    assert f["shrink_mae"] < 0.05
    assert f["shrink_model_mean"] > f["shrink_masked_mean"] + 0.2


def test_pseudo_likelihood():
    """Thm 3 on the Ising chain: the expected log-PL's gradient vanishes at the true (h, J), and the PL estimate
    approaches it as the sample grows."""
    r = ising_fits()
    assert r["pop_grad"] < 1e-12
    assert r["err_20000"] < r["err_200"] and r["err_20000"] < 0.03


def test_brook():
    """Thm 4: the HMM's joint over all 25⁴ sequences of length 4, rebuilt from its full conditionals alone, to
    rounding."""
    p = hmm_joint(BROOK_N)
    p = p / p.sum()
    cond = [p / p.sum(i, keepdims=True) for i in range(BROOK_N)]
    rebuilt = brook_rebuild(cond, (1, 2, 3, 4))
    assert np.abs(rebuilt - p).max() < 1e-15 and np.abs(np.log(rebuilt / p)).max() < 1e-9


def test_compatibility():
    """Cor 2: the four-cycle gap vanishes for the HMM's own conditionals; the trained encoder's conditionals violate it
    by orders of magnitude more, and so — less — do the Bayes-optimal MLM's (a [MASK]'s target conditions on a noisy
    context, so even the population optimum is not a joint's conditionals)."""
    c = compatibility()
    assert c["clean_max"] < 1e-9
    assert c["model_median"] > 1e3 * c["clean_max"] and c["model_median"] > 0.01
    assert c["bayes_median"] > 1e-6 and c["bayes_median"] < c["model_median"]


def test_mev_geometry():
    """Prop 2: MEV lies in [1/rank, 1], is 1 exactly for collinear rows, and the mean is the least-squares constant
    summary; on the HMM the token table is static (MEV = self-similarity = 1 for one repeated row), and context
    spreads the polyseme's occurrences."""
    rng = np.random.default_rng([SEED, 1505])
    A = rng.normal(size=(30, 6))
    assert 1 / 6 <= mev(A) <= 1 and abs(mev(np.outer(rng.normal(size=30), rng.normal(size=6))) - 1) < 1e-12
    c = A.mean(0)
    assert all(((A - c) ** 2).sum() <= ((A - (c + e)) ** 2).sum() for e in rng.normal(size=(20, 6)) * 0.1)
    E = np.asarray(train_mlm(HMM_MLM).P["E"][POLYSEME], float)
    assert abs(mev(np.tile(E, (5, 1))) - 1) < 1e-12 and abs(self_similarity(np.tile(E, (5, 1))) - 1) < 1e-12
    g = hmm_geometry()
    assert g["mev_2"] < g["mev_0"] and g["self_sim_2"] < g["self_sim_0"]


def test_hmm_probe():
    """§5: the layer-2 probe reaches within 0.05 of the exact Bayes accuracy, far above the static cap; on shuffled
    labels the same probe stays near the cap."""
    r = hmm_probe()
    assert r["acc_2"] > r["bayes"] - 0.05 and r["bayes"] > 0.9
    assert all(abs(r[f"control_{l}"] - r["static_cap"]) < 0.06 for l in range(3))


def test_alias_typing():
    """§7(a): static representations (SGNS; layer 0, read at a fixed position) stay at or below the per-alias
    majority; the contextual layers beat it by many standard errors, and type kestrel well above its 0.517 majority,
    which layer 0 cannot."""
    a = alias_typing()
    se = binom_se(a["majority"], a["n"])
    assert a["loao_sgns"] <= a["majority"] + 1e-12 and a["loao_layer0"] <= a["majority"] + 1e-12
    assert a["loao_layer1"] > a["majority"] + 5 * se and a["loao_layer2"] > a["majority"] + 5 * se
    assert a["kestrel_cv_layer1"] > a["kestrel_majority"] + 0.15 and a["kestrel_cv_layer0"] <= a["kestrel_majority"]
    assert filing_geometry()["mev_0"] > 1 - 1e-6  # every kestrel occurrence has one layer-0 vector


def test_news_polarity():
    """§7(b): the contextual probe reads the news a direction word reports from the aspect beside it, well above the
    static cap; the shuffled-label control does not."""
    r = news_polarity()
    assert r["layer1"] > r["cap"] + 5 * binom_se(r["cap"], r["n"])
    assert r["control1"] < r["cap"] + 0.02


def test_strong_weak():
    """§7(b), Topic 7's question: in this corpus the text barely carries a lexicon word's polarity (a whole document's
    vote reaches under 0.6), and no representation's type-level probe reaches two standard errors above chance."""
    s = strong_weak()
    assert s["vote_doc"] < 0.6 and s["vote8"] < s["vote_doc"]
    assert all(s[k] < 0.5 + 2 * s["chance_se"] for k in ("sgns", "layer0", "layer1", "layer2"))


def test_viz_payload():
    """The lab's data: the HMM sentence holds both senses, its three views agree with Thm 2's exact predictor to the
    reported KL; the filing windows are centered on their tokens; every layer row is present; the file stays under
    60 kB."""
    p = viz_payload()
    h = p["hmm"]
    assert len(h["tokens"]) == H_N and {v is None for v in h["sense_a"]} == {True, False}
    assert all(len(v["exact"]) == H_N and len(v["exact"][0]) == H_VOCAB for v in h["views"].values())
    assert all(abs(sum(row) - 1) < 0.02 for v in h["views"].values() for row in v["exact"])
    for w, word in zip(p["filings"], ("kestrel", "kestrel", "rose", "rose")):
        assert w["words"][w["center"]] == word and len(w["words"]) == FILING_MLM.n
    assert len(p["clouds"]["hmm"]["xy"]) == 3 and len(p["clouds"]["kestrel"]["product"]) == 118
    assert len(p["compat"]["model"]) == COMPAT_PAIRS
    assert len(json.dumps(p, separators=(",", ":"))) < 60_000


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "masked-lm-lab.json"


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
    print("\nSummary (pre-LN bidirectional encoders, 2 blocks, tied embeddings, learned positions; HMM: d = 32, n = 24;"
          " filings: d = 64, n = 32 over generate_corpus(300, 0)):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.6f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
