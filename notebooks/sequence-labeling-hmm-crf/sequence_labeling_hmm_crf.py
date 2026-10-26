"""Sequence labeling, from HMMs to linear-chain CRFs — the reference implementation for formalNLP's twenty-first topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. One recursion, two semirings. On a chain of nonnegative potentials, one forward pass in (+, ×) gives the partition
   function, a backward pass adds every node and pair marginal, and the same pass in (max, ×) gives the mode (Thm 1).
   formalStatistics' forward (Thm 5), forward–backward (Thm 7), and Viterbi (Thm 8) are the HMM case.
2. The HMM tagger. Supervised maximum likelihood is normalized counts, and add-α is a Dirichlet posterior mean
   (Prop 1): an n-gram model over tags times naive Bayes emissions.
3. The linear-chain CRF. Its conditional log-likelihood has gradient "observed minus expected features" and Hessian
   −Σ Cov[F | x], so it is concave, and strictly concave under an L2 penalty (Thm 2).
4. A generative–discriminative pair. Every HMM's conditional is a CRF with indicator features, and every such CRF with
   start and end weights is some HMM's conditional (Thm 3): the two differ only in how they are estimated (Cor 1).
5. Local normalization. An MEMM whose factor at t reads x only through x_{1:t+w} has prefix marginals that ignore
   every later word (Prop 2): label bias.
6. Decoding risk. Viterbi minimizes expected 0–1 sequence loss; posterior-marginal decoding minimizes expected Hamming
   loss (Thm 4).
7. The structured perceptron. On data separable with margin γ, it makes at most R²/γ² mistakes (Thm 5).
8. Simulated worlds, where the Bayes tagger is exact: HMM vs CRF vs perceptron learning curves, well specified and
   misspecified.
9. Finance case study. BIO tags over the corpus's gold entity spans: HMM, MEMM, CRF, averaged perceptron.

Cross-checks: brute-force enumeration of every path for Thms 1–5 and Prop 2; Topic 12's scaled forward–backward on
its HMM; complex-step derivatives for every gradient and Hessian-vector product.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/sequence-labeling-hmm-crf/sequence_labeling_hmm_crf.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import optimize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "n-gram-models-smoothing"))
sys.path.insert(0, str(_HERE.parent / "linear-text-classifiers"))
sys.path.insert(0, str(_HERE.parent / "contextual-embeddings-masked-lm"))
import contextual_embeddings_masked_lm as cm  # noqa: E402 — module objects: their test_* stay out of this harness
import linear_text_classifiers as ltc  # noqa: E402
import n_gram_models_smoothing as ng  # noqa: E402

fc = ltc.fc

# ONE stream per random-instance claim and per sampled grid, each from default_rng([SEED, k]); k ≥ 3400 keeps this
# module's streams apart from Topics 1–20.
SEED = ltc.SEED
_r = ltc._r


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


# --------------------------------------------------------------------------------------------
# 1. Chain inference in two semirings (Thm 1).
# --------------------------------------------------------------------------------------------
#
# A chain of length T over K labels is scored by log-potentials: unary U[t, k] and pairwise P[t, j, k] (the edge
# from label j at t to label k at t + 1). Start and end weights are folded into U[0] and U[T−1]. Everything is
# batched over sequences of one length: U is (n, T, K) and P is (K, K), shared, or (n, T−1, K, K).


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    """log Σ exp along ``axis``, safe for complex inputs (complex step) and for all-−∞ slices. The shift is the real
    part's max, so the result is holomorphic in a's imaginary perturbation."""
    m = np.max(a.real, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def _pair(P: np.ndarray, t: int) -> np.ndarray:
    return P if P.ndim == 2 else P[:, t]


def chain_forward(U: np.ndarray, P: np.ndarray, op: str = "sum") -> tuple[np.ndarray, np.ndarray]:
    """Forward pass of Thm 1 in log space. op = "sum" is the (+, ×) semiring: alpha[:, t, k] is the log of the summed
    potential of every prefix ending in k, and the second return is log Z. op = "max" is (max, ×): alpha is the best
    prefix's log-potential and the second return is the mode's score."""
    red = (lambda a, ax: _lse(a, ax)) if op == "sum" else (lambda a, ax: np.max(a, axis=ax))
    n, T, K = U.shape
    alpha = np.empty(U.shape, dtype=np.result_type(U, P))
    alpha[:, 0] = U[:, 0]
    for t in range(1, T):
        alpha[:, t] = U[:, t] + red(alpha[:, t - 1, :, None] + _pair(P, t - 1), 1)
    return alpha, red(alpha[:, -1], 1)


def chain_backward(U: np.ndarray, P: np.ndarray, op: str = "sum") -> np.ndarray:
    """beta[:, t, j]: log of the summed (op = "sum") or best (op = "max") potential of every suffix after position t,
    given label j at t."""
    red = (lambda a, ax: _lse(a, ax)) if op == "sum" else (lambda a, ax: np.max(a, axis=ax))
    n, T, K = U.shape
    beta = np.zeros(U.shape, dtype=np.result_type(U, P))
    for t in range(T - 2, -1, -1):
        beta[:, t] = red(_pair(P, t) + (U[:, t + 1] + beta[:, t + 1])[:, None, :], 2)
    return beta


def max_marginals(U: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Thm 1 in (max, ×): mu[:, t, k] = max over paths through label k at t of p(z | x), from one forward and one
    backward max pass. Each column's largest entry is the mode's probability, attained at the Viterbi label."""
    alpha, _ = chain_forward(U, P, "max")
    _, logZ = chain_forward(U, P, "sum")
    return np.exp(alpha + chain_backward(U, P, "max") - logZ[:, None, None])


def chain_marginals(U: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(log Z, node marginals (n, T, K), pair marginals (n, T−1, K, K)) of the Gibbs distribution ∝ exp(score)."""
    alpha, logZ = chain_forward(U, P, "sum")
    beta = chain_backward(U, P)
    node = np.exp(alpha + beta - logZ[:, None, None])
    pair = np.exp(alpha[:, :-1, :, None] + (P if P.ndim == 4 else P[None, None])
                  + (U[:, 1:] + beta[:, 1:])[:, :, None, :] - logZ[:, None, None, None])
    return logZ, node, pair


def viterbi(U: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The (max, ×) pass with back-pointers: each sequence's highest-scoring path (n, T) and its score (n,)."""
    n, T, K = U.shape
    delta = U[:, 0].copy()
    back = np.zeros((n, T, K), int)
    for t in range(1, T):
        cand = delta[:, :, None] + _pair(P, t - 1)
        back[:, t] = np.argmax(cand, axis=1)
        delta = U[:, t] + np.take_along_axis(cand, back[:, t][:, None, :], 1)[:, 0]
    path = np.zeros((n, T), int)
    path[:, -1] = np.argmax(delta, axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[np.arange(n), t, path[:, t]]
    return path, delta.max(1)


def path_score(U: np.ndarray, P: np.ndarray, z: np.ndarray) -> np.ndarray:
    """score(z) = Σ_t U[t, z_t] + Σ_t P[t, z_t, z_{t+1}] for each sequence of the batch."""
    n, T, _ = U.shape
    rows = np.arange(n)
    s = U[rows[:, None], np.arange(T)[None], z].sum(1)
    for t in range(T - 1):
        Pt = _pair(P, t)
        s = s + (Pt[z[:, t], z[:, t + 1]] if Pt.ndim == 2 else Pt[rows, z[:, t], z[:, t + 1]])
    return s


def brute_force(U: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every one of the K^T paths of a single sequence (U (T, K)) and its score — the ground truth for Thm 1."""
    T, K = U.shape
    paths = np.array(list(itertools.product(range(K), repeat=T)))
    Ub = np.broadcast_to(U, (len(paths), T, K))
    Pb = P if P.ndim == 2 else np.broadcast_to(P, (len(paths), *P.shape))
    return paths, path_score(Ub, Pb, paths)


# --------------------------------------------------------------------------------------------
# 2. The HMM tagger (Prop 1).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HMM:
    """Log-parameters of an HMM tagger with an explicit end: start ν (K), transitions A (K, K), end e (K), emissions
    B (K, V). With end = 0 (log 1) it is formalStatistics' Def 3, run for a fixed length."""

    start: np.ndarray
    trans: np.ndarray
    end: np.ndarray
    emit: np.ndarray


def hmm_fit(xs: list[np.ndarray], zs: list[np.ndarray], K: int, V: int, alpha: float = 0.0,
            with_end: bool = True) -> HMM:
    """Prop 1: the supervised MLE is normalized counts; add-α to every count is the posterior mean under symmetric
    Dirichlet(α) priors on each row (the MAP under Dirichlet(α + 1)), the n-gram topic's additive estimator."""
    c0, cA, cE, cB = np.full(K, alpha), np.full((K, K), alpha), np.full(K, alpha), np.full((K, V), alpha)
    for x, z in zip(xs, zs):
        c0[z[0]] += 1
        np.add.at(cA, (z[:-1], z[1:]), 1)
        cE[z[-1]] += 1
        np.add.at(cB, (z, x), 1)
    with np.errstate(divide="ignore"):
        if with_end:
            out = cA.sum(1) + cE
            return HMM(np.log(c0 / c0.sum()), np.log(cA / out[:, None]), np.log(cE / out), np.log(cB / cB.sum(1, keepdims=True)))
        return HMM(np.log(c0 / c0.sum()), np.log(cA / cA.sum(1, keepdims=True)), np.zeros(K),
                   np.log(cB / cB.sum(1, keepdims=True)))


def hmm_potentials(m: HMM, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An HMM as a chain: U[t, k] = log b_k(x_t) (plus log ν at t = 0 and log e at T − 1), P = log A. Then the
    (+, ×) pass gives log p(x_{1:T}) and the node marginals are the posterior γ_t (Thm 3a's first step)."""
    U = m.emit[:, x].T[None].copy()
    U[:, 0] += m.start
    U[:, -1] += m.end
    return U, m.trans


def hmm_from_topic12() -> HMM:
    """Topic 12's HMM (cm.hmm(): 8 states, 25 tokens, stationary start, fixed length — no end)."""
    T, B, pi = cm.hmm()
    with np.errstate(divide="ignore"):
        return HMM(np.log(pi), np.log(T), np.zeros(len(pi)), np.log(B))


# --------------------------------------------------------------------------------------------
# 3. Features, and the linear-chain CRF (Def 3, Thm 2).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bucket:
    """Sequences of one length L: token-feature rows X (n·L, F), sequence-major, and gold labels z (n, L)."""

    X: sp.csr_matrix
    z: np.ndarray


@dataclass(frozen=True)
class Data:
    """A labeled set, bucketed by length, with its observed feature totals (the "observed" half of Thm 2)."""

    buckets: tuple[Bucket, ...]
    F: int
    K: int
    n: int
    obs_W: np.ndarray  # (F, K): Σ_i Σ_t x_t ⊗ e_{z_t}
    obs_trans: np.ndarray  # (K, K)
    obs_start: np.ndarray
    obs_end: np.ndarray


def index_features(feats: list[list[list[str]]], vocab: dict[str, int] | None = None,
                   min_count: int = 1) -> dict[str, int]:
    """A feature vocabulary from training sequences (feats[i][t] = the feature strings at token t of sequence i)."""
    if vocab is not None:
        return vocab
    counts: dict[str, int] = {}
    for seq in feats:
        for fs in seq:
            for f in fs:
                counts[f] = counts.get(f, 0) + 1
    return {f: j for j, f in enumerate(sorted(f for f, c in counts.items() if c >= min_count))}


def make_data(feats: list[list[list[str]]], zs: list[np.ndarray], vocab: dict[str, int], K: int) -> Data:
    """Encode sequences as length buckets of binary feature rows (features outside ``vocab`` are dropped)."""
    F = len(vocab)
    by_len: dict[int, list[int]] = {}
    for i, z in enumerate(zs):
        by_len.setdefault(len(z), []).append(i)
    buckets = []
    obs_W = np.zeros((F, K))
    obs_trans, obs_start, obs_end = np.zeros((K, K)), np.zeros(K), np.zeros(K)
    for L in sorted(by_len):
        idx, ptr = [], [0]
        for i in by_len[L]:
            for fs in feats[i]:
                idx.extend(sorted({vocab[f] for f in fs if f in vocab}))
                ptr.append(len(idx))
        X = sp.csr_matrix((np.ones(len(idx)), np.array(idx, int), np.array(ptr)), shape=(len(ptr) - 1, F))
        z = np.array([zs[i] for i in by_len[L]], int).reshape(-1, L)
        buckets.append(Bucket(X, z))
        obs_W += np.asarray((X.T @ np.eye(K)[z.ravel()]))
        np.add.at(obs_trans, (z[:, :-1].ravel(), z[:, 1:].ravel()), 1)
        np.add.at(obs_start, z[:, 0], 1)
        np.add.at(obs_end, z[:, -1], 1)
    return Data(tuple(buckets), F, K, len(zs), obs_W, obs_trans, obs_start, obs_end)


def crf_unpack(theta: np.ndarray, F: int, K: int):
    """theta = [vec W (F×K), vec trans (K×K), start (K), end (K)]."""
    a, b = F * K, F * K + K * K
    return theta[:a].reshape(F, K), theta[a:b].reshape(K, K), theta[b:b + K], theta[b + K:]


def crf_size(F: int, K: int) -> int:
    return F * K + K * K + 2 * K


def crf_potentials(theta: np.ndarray, X, L: int, F: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """The chain of one length bucket: U[t, k] = θ·f(x, t, k) plus start/end weights; P = the transition weights."""
    W, trans, start, end = crf_unpack(theta, F, K)
    U = np.asarray(X @ W).reshape(-1, L, K)
    U[:, 0] += start
    U[:, -1] += end
    return U, trans


def crf_objective(theta: np.ndarray, data: Data, lam: float) -> tuple[float, np.ndarray]:
    """Thm 2's objective: (1/n) Σ_i [log Z(x_i) − θ·F(x_i, z_i)] + (λ/2)‖θ‖², and its gradient
    (1/n) Σ_i (E_θ[F | x_i] − F(x_i, z_i)) + λθ, the expectation taken from Thm 1's node and pair marginals.
    Complex-step safe: every operation is holomorphic in theta."""
    F, K = data.F, data.K
    W, trans, start, end = crf_unpack(theta, F, K)
    dt = theta.dtype
    gW = np.zeros((F, K), dt)
    gT, gS, gE = np.zeros((K, K), dt), np.zeros(K, dt), np.zeros(K, dt)
    logZ = 0.0
    for bk in data.buckets:
        L = bk.z.shape[1]
        U, P = crf_potentials(theta, bk.X, L, F, K)
        lz, node, pair = chain_marginals(U, P)
        logZ = logZ + lz.sum()
        gW += bk.X.T @ node.reshape(-1, K)
        gT += pair.sum((0, 1))
        gS += node[:, 0].sum(0)
        gE += node[:, -1].sum(0)
    gold = (np.sum(data.obs_W * W) + np.sum(data.obs_trans * trans) + data.obs_start @ start + data.obs_end @ end)
    f = (logZ - gold) / data.n + 0.5 * lam * np.sum(theta * theta)
    g = np.concatenate([(gW - data.obs_W).ravel(), (gT - data.obs_trans).ravel(), gS - data.obs_start,
                        gE - data.obs_end]) / data.n + lam * theta
    return f, g


def crf_hessp(theta: np.ndarray, v: np.ndarray, data: Data, lam: float) -> np.ndarray:
    """Thm 2's Hessian, (1/n) Σ_i Cov_θ[F | x_i] + λI, applied to v: exactly, as Im ∇f(θ + ihv)/h (complex step)."""
    h = 1e-20
    return crf_objective(theta + 1j * h * v, data, lam)[1].imag / h


def crf_fit(data: Data, lam: float, x0: np.ndarray | None = None, gtol: float = 1e-8) -> np.ndarray:
    """The unique maximizer of Thm 2's penalized conditional likelihood, by trust-region Newton–CG with exact
    Hessian-vector products (the linear-text-classifiers recipe)."""
    r = optimize.minimize(lambda th: crf_objective(th, data, lam), np.zeros(crf_size(data.F, data.K)) if x0 is None
                          else x0, jac=True, hessp=lambda th, v: crf_hessp(th, v, data, lam), method="trust-ncg",
                          options={"maxiter": 200, "gtol": gtol})
    return r.x


def crf_predict(theta: np.ndarray, data: Data, decoder: str = "viterbi") -> list[tuple[np.ndarray, np.ndarray]]:
    """(gold, predicted) label arrays per bucket, by Viterbi or posterior-marginal (MBR) decoding (Thm 4)."""
    out = []
    for bk in data.buckets:
        U, P = crf_potentials(theta, bk.X, bk.z.shape[1], data.F, data.K)
        pred = viterbi(U, P)[0] if decoder == "viterbi" else chain_marginals(U, P)[1].argmax(2)
        out.append((bk.z, pred))
    return out


def token_accuracy(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    return float(sum((g == p).sum() for g, p in pairs) / sum(g.size for g, _ in pairs))


# --------------------------------------------------------------------------------------------
# 4. The MEMM: a chain of locally normalized logistic regressions (Def 4, Prop 2).
# --------------------------------------------------------------------------------------------


def memm_rows(bk: Bucket, K: int, prev: np.ndarray | None = None) -> sp.csr_matrix:
    """Each token's feature row with the previous label appended as a one-hot over K + 1 columns (column K = START).
    With prev None, the gold previous labels (how an MEMM trains: every factor is fit on gold history)."""
    n, L = bk.z.shape
    if prev is None:
        prev = np.concatenate([np.full((n, 1), K), bk.z[:, :-1]], 1).ravel()
    onehot = sp.csr_matrix((np.ones(n * L), prev, np.arange(n * L + 1)), shape=(n * L, K + 1))
    return sp.hstack([bk.X, onehot], format="csr")


def memm_fit(data: Data, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit p(z_t | z_{t−1}, x) by softmax LR on every token row (Topic 17's lr_fit). Returns (W (K, F + K + 1), b)."""
    X = sp.vstack([memm_rows(bk, data.K) for bk in data.buckets], format="csr")
    y = np.concatenate([bk.z.ravel() for bk in data.buckets])
    W, b, _ = ltc.lr_fit(X, y, data.K, lam)
    return W, b


def memm_potentials(W: np.ndarray, b: np.ndarray, X, L: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """An MEMM as a chain: U[0] = log p(z_1 | START, x), U[t > 0] = 0, P[t, j, k] = log p(z_{t+1} = k | z_t = j, x).
    Every factor is locally normalized, so its log Z is exactly 0."""
    F = X.shape[1]
    a = (np.asarray(X @ W[:, :F].T) + b).reshape(-1, L, K)  # the label-independent part of each logit
    logits = a[:, :, None, :] + W[:, F:].T[None, None]  # (n, L, K + 1 previous, K)
    logp = logits - _lse(logits, 3)[..., None]
    U = np.zeros(a.shape)
    U[:, 0] = logp[:, 0, K]
    return U, logp[:, 1:, :K, :]


# --------------------------------------------------------------------------------------------
# 5. The structured perceptron (Thm 5).
# --------------------------------------------------------------------------------------------


def perceptron(data: Data, epochs: int, average: bool = True, order: np.random.Generator | None = None,
               ) -> tuple[np.ndarray, int]:
    """Collins's structured perceptron on the CRF's feature map: decode each sequence by Viterbi under the current θ
    and, on a mistake, add F(x, z) − F(x, ẑ). Returns (θ, mistakes); with ``average`` the averaged θ (Collins 2002
    §2.5), via the running-sum trick θ̄ = θ − S/c."""
    F, K = data.F, data.K
    theta = np.zeros(crf_size(F, K))
    S = np.zeros_like(theta)
    seqs = [(bk.X[i * bk.z.shape[1]:(i + 1) * bk.z.shape[1]], bk.z[i]) for bk in data.buckets
            for i in range(bk.z.shape[0])]
    W, trans, start, end = crf_unpack(theta, F, K)  # views into theta
    SW, ST, SS, SE = crf_unpack(S, F, K)
    c, mistakes = 1, 0
    for _ in range(epochs):
        for i in (order.permutation(len(seqs)) if order is not None else range(len(seqs))):
            X, z = seqs[i]
            U, P = crf_potentials(theta, X, len(z), F, K)
            zh = viterbi(U, P)[0][0]
            if not np.array_equal(zh, z):
                mistakes += 1
                for lab, sgn in ((z, 1.0), (zh, -1.0)):
                    rows = np.repeat(np.arange(len(lab)), np.diff(X.indptr))
                    np.add.at(W, (X.indices, lab[rows]), sgn)
                    np.add.at(SW, (X.indices, lab[rows]), sgn * c)
                    np.add.at(trans, (lab[:-1], lab[1:]), sgn)
                    np.add.at(ST, (lab[:-1], lab[1:]), sgn * c)
                    start[lab[0]] += sgn
                    SS[lab[0]] += sgn * c
                    end[lab[-1]] += sgn
                    SE[lab[-1]] += sgn * c
            c += 1
    return (theta - S / c if average else theta), mistakes


# --------------------------------------------------------------------------------------------
# 6. The generative–discriminative pair (Thm 3, Cor 1).
# --------------------------------------------------------------------------------------------


def crf_from_hmm(m: HMM) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Thm 3(a): an HMM's conditional is the indicator-feature CRF with θ = the HMM's log-parameters:
    (transition weights, start weights, end weights, emission weights μ (K, V))."""
    return m.trans, m.start, m.end, m.emit


def hmm_from_crf(trans: np.ndarray, start: np.ndarray, end: np.ndarray, mu: np.ndarray) -> HMM:
    """Thm 3(b): an HMM (with an end state) whose conditional p(z | x) equals the indicator-feature CRF's
    p(z | x) ∝ exp(start[z_1] + Σ_t trans[z_{t−1}, z_t] + Σ_t mu[z_t, x_t] + end[z_T]), for every x and every length.

    The h-transform construction (page §5):
      S_k = Σ_w e^{mu[k, w]}            normalizes state k's emissions: b_k(w) = e^{mu[k, w]} / S_k;
      M_jk = e^{trans[j, k]} S_k        what is left of the transition after the emission is normalized;
      pick any c > ρ(M) (spectral radius) and solve (cI − M) h = e^{end}; h > 0 by the Neumann series;
      a_jk = M_jk h_k / (c h_j),  a_{j,stop} = e^{end[j]} / (c h_j)   — each row sums to 1 by the choice of h;
      ν_k ∝ e^{start[k]} S_k h_k.
    The joint then telescopes to exp(score) / (c^T Σ_j e^{start[j]} S_j h_j), a function of T alone.

    Returns an HMM in log space (start, trans, end, emit), with ``end`` holding log a_{j,stop}.
    """
    S = np.exp(mu).sum(1)
    M = np.exp(trans) * S[None, :]
    c = 2.0 * np.abs(np.linalg.eigvals(M)).max()  # any c > ρ(M) works; 2ρ keeps cI − M well conditioned
    h = np.linalg.solve(c * np.eye(len(S)) - M, np.exp(end))
    nu = np.exp(start) * S * h
    return HMM(np.log(nu / nu.sum()), np.log(M * h[None, :] / (c * h[:, None])), end - np.log(c * h),
               mu - np.log(S)[:, None])


# --------------------------------------------------------------------------------------------
# 7. Decoding risk (Thm 4).
# --------------------------------------------------------------------------------------------


def expected_losses(U: np.ndarray, P: np.ndarray, zhat: np.ndarray) -> tuple[float, float]:
    """For one sequence and a decoded path ẑ: (expected Hamming loss Σ_t (1 − γ_t(ẑ_t)), expected 0–1 loss
    1 − p(ẑ | x)) under the chain's Gibbs distribution."""
    logZ, node, _ = chain_marginals(U[None], P)
    ham = float(np.sum(1 - node[0, np.arange(len(zhat)), zhat]))
    zo = float(1 - np.exp(path_score(U[None], P, zhat[None])[0] - logZ[0]))
    return ham, zo



def crf_fit_lbfgs(data: Data, lam: float, x0: np.ndarray | None = None) -> np.ndarray:
    """The same maximizer by L-BFGS run to gradient 10⁻⁶ (ftol 0, so it stops on the gradient alone): within ~10⁻⁵ of
    trust-ncg's θ at a fifth of its cost on the experiment grids, where an exact Hv is a complex forward–backward
    pass. test_lbfgs_reaches_the_trust_region_optimum pins the gap."""
    r = optimize.minimize(lambda th: crf_objective(th, data, lam), np.zeros(crf_size(data.F, data.K)) if x0 is None
                          else x0, jac=True, method="L-BFGS-B",
                          options={"gtol": 1e-6, "ftol": 0.0, "maxiter": 5000, "maxcor": 30})
    return r.x

def window_features(x, w: int, left: int | None = None) -> list[list[str]]:
    """Token-window features: a bias, the word, the words up to ``left`` positions before it (default w) and up to w
    positions after it. The HMM's indicator features are w = 0 without the bias; an MEMM with lookahead w reads
    x_{1:t+w} when left ≥ t."""
    T = len(x)
    left = w if left is None else left
    out = []
    for t in range(T):
        fs = ["bias", f"w0={x[t]}"]
        for d in range(1, left + 1):
            fs.append(f"w-{d}={x[t - d]}" if t - d >= 0 else f"w-{d}=<s>")
        for d in range(1, w + 1):
            fs.append(f"w+{d}={x[t + d]}" if t + d < T else f"w+{d}=</s>")
        out.append(fs)
    return out


# ============================================================================================
# Harness — exactness on small chains (Thms 1–5, Props 1–2).
# ============================================================================================


def _random_chain(rng: np.random.Generator, T: int, K: int, varying: bool = False, holes: bool = False):
    U = rng.normal(size=(T, K))
    P = rng.normal(size=(T - 1, K, K) if varying else (K, K))
    if holes:  # forbidden labels and transitions (log 0), as in an HMM whose states emit disjoint tokens
        U[rng.random(U.shape) < 0.2] = -np.inf
        P[rng.random(P.shape) < 0.2] = -np.inf
        z = rng.integers(0, K, T)  # keep one path feasible, so Z > 0
        U[np.arange(T), z] = rng.normal(size=T)
        if T > 1:
            P[(np.arange(T - 1), z[:-1], z[1:]) if varying else (z[:-1], z[1:])] = rng.normal(size=T - 1)
    return U, P


def test_thm1_semirings_match_brute_force():
    """Thm 1: one (+, ×) pass gives log Z and, with the backward pass, every node and pair marginal; the (max, ×)
    pass gives the mode and its score — on random chains, shared or position-varying, with forbidden cells."""
    rng = _rng(3400)
    for trial in range(60):
        T, K = int(rng.integers(1, 6)), int(rng.integers(1, 4))
        U, P = _random_chain(rng, T, K, varying=trial % 2 == 1 and T > 1, holes=trial % 3 == 0)
        Pb = P if P.ndim == 2 else P[None]
        paths, s = brute_force(U, P)
        with np.errstate(divide="ignore"):
            lz_true = np.log(np.sum(np.exp(s - s.max()))) + s.max()
        lz, node, pair = chain_marginals(U[None], Pb)
        assert abs(lz[0] - lz_true) < 1e-10
        p = np.exp(s - lz_true)
        for t in range(T):
            assert np.allclose(node[0, t], np.bincount(paths[:, t], p, K), atol=1e-12)
        for t in range(T - 1):
            pt = np.zeros((K, K))
            np.add.at(pt, (paths[:, t], paths[:, t + 1]), p)
            assert np.allclose(pair[0, t], pt, atol=1e-12)
        assert abs(chain_forward(U[None], Pb, "max")[1][0] - s.max()) < 1e-12
        path, best = viterbi(U[None], Pb)
        assert abs(best[0] - s.max()) < 1e-12 and abs(path_score(U[None], Pb, path)[0] - s.max()) < 1e-12


def test_thm1_hmm_case_is_topic12_forward_backward():
    """The HMM case of Thm 1 is formalStatistics' forward–backward: on Topic 12's HMM, our node marginals equal the
    posterior α̂ E β̂ from Topic 12's scaled forward–backward, and log Z is log p(x) (checked by enumeration at T = 5)."""
    m = hmm_from_topic12()
    x, _ = cm.sample_hmm(_rng(3401), 20)
    T_, B, _ = cm.hmm()
    E = B[:, x].transpose(1, 2, 0)
    a, be = cm._forward_backward(E)
    post = a * E * be
    post /= post.sum(2, keepdims=True)
    for i in range(len(x)):
        U, P = hmm_potentials(m, x[i])
        assert np.abs(chain_marginals(U, P)[1][0] - post[i]).max() < 1e-10
    U, P = hmm_potentials(m, x[0, :5])
    paths, s = brute_force(U[0], P)
    assert abs(chain_forward(U, P)[1][0] - np.log(np.exp(s).sum())) < 1e-10


def test_prop1_hmm_mle_is_counts():
    """Prop 1: hmm_fit is normalized counts (add-α: add α to every count), with the end state counted in each row."""
    rng = _rng(3402)
    K, V = 3, 4
    zs = [rng.integers(0, K, int(rng.integers(1, 7))) for _ in range(40)]
    xs = [rng.integers(0, V, len(z)) for z in zs]
    m = hmm_fit(xs, zs, K, V, alpha=0.5)
    cA = np.full((K, K), 0.5)
    cE = np.full(K, 0.5)
    for z in zs:
        for a, b in zip(z[:-1], z[1:]):
            cA[a, b] += 1
        cE[z[-1]] += 1
    out = cA.sum(1) + cE
    assert np.allclose(np.exp(m.trans), cA / out[:, None]) and np.allclose(np.exp(m.end), cE / out)
    assert np.allclose(np.exp(m.trans).sum(1) + np.exp(m.end), 1) and np.allclose(np.exp(m.emit).sum(1), 1)
    # With its end state the fitted HMM is a distribution over pairs (x, z) of every length: the joint summed over all
    # x of length T is the length law ν A^{T−1} e, and the lengths' probabilities sum to ν (I − A)⁻¹ e = 1.
    nu, A, e = np.exp(m.start), np.exp(m.trans), np.exp(m.end)
    for T in range(1, 6):
        total = sum(math.exp(chain_forward(*hmm_potentials(m, np.array(x)))[1][0])
                    for x in itertools.product(range(V), repeat=T))
        assert abs(total - nu @ np.linalg.matrix_power(A, T - 1) @ e) < 1e-12
    assert abs(nu @ np.linalg.solve(np.eye(K) - A, e) - 1) < 1e-12


def _toy_data(rng: np.random.Generator, n: int, K: int, V: int, w: int, Tmax: int = 4):
    zs = [rng.integers(0, K, int(rng.integers(1, Tmax + 1))) for _ in range(n)]
    xs = [rng.integers(0, V, len(z)) for z in zs]
    feats = [window_features(x, w) for x in xs]
    vocab = index_features(feats)
    return make_data(feats, zs, vocab, K), xs, zs


def test_thm2_gradient_hessian_and_concavity():
    """Thm 2: the gradient is observed minus expected features (checked by complex step on the objective, coordinate
    by coordinate), the Hessian-vector product is exact (symmetric, ⪰ λI), and the expected features equal brute-force
    path averages."""
    rng = _rng(3403)
    data, xs, zs = _toy_data(rng, 12, 3, 3, 1)
    lam = 0.1
    theta = rng.normal(size=crf_size(data.F, data.K)) * 0.5
    f, g = crf_objective(theta, data, lam)
    h = 1e-20
    gc = np.array([crf_objective(theta + 1j * h * e, data, lam)[0].imag / h for e in np.eye(len(theta))])
    assert np.abs(g - gc).max() < 1e-12
    H = np.array([crf_hessp(theta, e, data, lam) for e in np.eye(len(theta))])
    assert np.abs(H - H.T).max() < 1e-12
    assert np.linalg.eigvalsh(H).min() > lam - 1e-10
    # Expected features by enumeration, for one sequence: E[F] = Σ_z p(z | x) F(x, z).
    bk = data.buckets[-1]
    L = bk.z.shape[1]
    U, P = crf_potentials(theta, bk.X[:L], L, data.F, data.K)
    paths, s = brute_force(U[0], P)
    p = np.exp(s - np.log(np.exp(s).sum()))
    _, node, pair = chain_marginals(U, P)
    for t in range(L):
        assert np.allclose(node[0, t], np.bincount(paths[:, t], p, data.K))


def test_thm2_unique_optimum():
    """Strict concavity under λ > 0: fits from two different starts reach one θ, where the gradient vanishes."""
    rng = _rng(3404)
    data, _, _ = _toy_data(rng, 30, 3, 4, 1)
    a = crf_fit(data, 0.05, gtol=1e-11)
    b = crf_fit(data, 0.05, x0=rng.normal(size=crf_size(data.F, data.K)) * 3, gtol=1e-11)
    assert np.abs(a - b).max() < 1e-8
    assert np.abs(crf_objective(a, data, 0.05)[1]).max() < 1e-10


def test_thm3a_hmm_conditional_is_a_crf():
    """Thm 3(a): with θ = the HMM's log-parameters, the CRF's p(z | x) is the HMM's p(z, x)/p(x), path by path."""
    rng = _rng(3405)
    K, V = 3, 4
    zs = [rng.integers(0, K, 5) for _ in range(30)]
    xs = [rng.integers(0, V, 5) for _ in range(30)]
    m = hmm_fit(xs, zs, K, V, alpha=1.0)
    trans, start, end, mu = crf_from_hmm(m)
    x = rng.integers(0, V, 4)
    U = mu[:, x].T.copy()
    U[0] += start
    U[-1] += end
    paths, s = brute_force(U, trans)
    joint = np.array([m.start[z[0]] + sum(m.trans[a, b] for a, b in zip(z[:-1], z[1:]))
                      + sum(m.emit[k, w] for k, w in zip(z, x)) + m.end[z[-1]] for z in paths])
    assert np.allclose(np.exp(s - np.log(np.exp(s).sum())), np.exp(joint - np.log(np.exp(joint).sum())))


def test_thm3b_every_indicator_crf_is_an_hmm_conditional():
    """Thm 3(b): hmm_from_crf returns a valid HMM (rows of A plus stop sum to 1, emissions and ν are distributions)
    whose conditional equals the CRF's on every path, for random weights, sentences, and lengths."""
    rng = _rng(3406)
    for trial in range(25):
        K, V = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        trans, start, end, mu = (rng.normal(size=(K, K)) * 2, rng.normal(size=K), rng.normal(size=K) * (trial % 2),
                                 rng.normal(size=(K, V)) * 2)
        m = hmm_from_crf(trans, start, end, mu)
        assert np.allclose(np.exp(m.trans).sum(1) + np.exp(m.end), 1)
        assert np.allclose(np.exp(m.emit).sum(1), 1) and np.isclose(np.exp(m.start).sum(), 1)
        for T in (1, 2, 5):
            x = rng.integers(0, V, T)
            U = mu[:, x].T.copy()
            U[0] += start
            U[-1] += end
            _, s = brute_force(U, trans)
            Uh, Ph = hmm_potentials(m, x)
            _, sh = brute_force(Uh[0], Ph)
            assert np.abs(np.exp(s - np.log(np.exp(s).sum())) - np.exp(sh - np.log(np.exp(sh).sum()))).max() < 1e-12


def test_prop2_memm_prefix_ignores_the_future():
    """Prop 2: an MEMM whose factor at t reads x_{1:t+w} has p(z_{1:t} | x) unchanged when any x_s with s > t + w
    changes (node marginals at t and earlier agree to 10⁻¹²), and its log Z is exactly 0. A CRF with the same
    features moves at the last position the MEMM cannot revise: the counterexample."""
    rng = _rng(3407)
    K, V, w, T = 3, 5, 1, 7
    feats0 = [window_features(rng.integers(0, V, T), w) for _ in range(40)]
    zs = [rng.integers(0, K, T) for _ in range(40)]
    vocab = index_features(feats0)
    data = make_data(feats0, zs, vocab, K)
    Wm = rng.normal(size=(K, data.F + K + 1))
    bm = rng.normal(size=K)
    theta = rng.normal(size=crf_size(data.F, K))
    x = rng.integers(0, V, T)
    crf_moved = 0.0
    for trial in range(30):
        y = x.copy()
        y[T - 1] = (x[T - 1] + 1 + trial % (V - 1)) % V  # change the last word only
        t_safe = T - 1 - w - 1  # the last position whose factor cannot see x_T
        nodes = {}
        for name, seq in (("x", x), ("y", y)):
            X = make_data([window_features(seq, w)], [np.zeros(T, int)], vocab, K).buckets[0].X
            U, P = memm_potentials(Wm, bm, X, T, K)
            lz, node, _ = chain_marginals(U, P)
            assert abs(lz[0]) < 1e-12
            Uc, Pc = crf_potentials(theta, X, T, data.F, K)
            nodes[name] = (node[0], chain_marginals(Uc, Pc)[1][0])
        assert np.abs(nodes["x"][0][:t_safe + 1] - nodes["y"][0][:t_safe + 1]).max() < 1e-12
        crf_moved = max(crf_moved, np.abs(nodes["x"][1][t_safe] - nodes["y"][1][t_safe]).max())
    assert crf_moved > 1e-2  # the CRF's marginal at the same position moves (its influence decays with distance)


def test_thm4_decoding_risk():
    """Thm 4: posterior-marginal decoding minimizes expected Hamming loss and Viterbi minimizes expected 0–1 loss —
    each against every one of the K^T paths."""
    rng = _rng(3408)
    for _ in range(40):
        U, P = _random_chain(rng, 5, 3)
        paths, _ = brute_force(U, P)
        mbr = chain_marginals(U[None], P)[1][0].argmax(1)
        vit = viterbi(U[None], P)[0][0]
        losses = np.array([expected_losses(U, P, z) for z in paths])
        hm, zm = expected_losses(U, P, mbr)
        hv, zv = expected_losses(U, P, vit)
        assert hm <= losses[:, 0].min() + 1e-12 and zv <= losses[:, 1].min() + 1e-12
        assert hm <= hv + 1e-12 and zv <= zm + 1e-12


SEP_K, SEP_V, SEP_T = 3, 4, 4


@lru_cache(maxsize=1)
def teacher_table() -> tuple[np.ndarray, dict, list]:
    """A unit-norm teacher θ* on the indicator features (K = 3, V = 4, T = 4) and, for each of the V^T = 256 inputs,
    its feature rows, the teacher's path z*(x), the margin of z*(x) over the runner-up path, and the input's radius
    max_z ‖F(x, z*) − F(x, z)‖ — all by enumeration."""
    K, V, T = SEP_K, SEP_V, SEP_T
    vocab = {f: j for j, f in enumerate(["bias", *(f"w0={v}" for v in range(V))])}
    teacher = _rng(3412).normal(size=crf_size(len(vocab), K))
    teacher /= np.linalg.norm(teacher)
    rows = []
    for x in itertools.product(range(V), repeat=T):
        X = make_data([window_features(np.array(x), 0)], [np.zeros(T, int)], vocab, K).buckets[0].X
        U, P = crf_potentials(teacher, X, T, len(vocab), K)
        paths, sc = brute_force(U[0], P)
        order = np.argsort(sc)[::-1]
        gold = paths[order[0]]
        Phi = np.array([_phi(X, z, len(vocab), K) for z in paths])
        rows.append((window_features(np.array(x), 0), gold, float(sc[order[0]] - sc[order[1]]),
                     float(np.linalg.norm(Phi - _phi(X, gold, len(vocab), K), axis=1).max())))
    return teacher, vocab, rows


def separable_instance(rng: np.random.Generator, n: int, gamma_min: float):
    """n training sequences drawn (with replacement) from the inputs whose teacher margin is at least gamma_min:
    separable by θ* with margin γ ≥ gamma_min by construction. Returns (data, γ, R) for the drawn set."""
    _, vocab, rows = teacher_table()
    ok = [r for r in rows if r[2] >= gamma_min]
    pick = [ok[i] for i in rng.integers(0, len(ok), n)]
    data = make_data([r[0] for r in pick], [r[1] for r in pick], vocab, SEP_K)
    return data, min(r[2] for r in pick), max(r[3] for r in pick)


def _phi(X, z: np.ndarray, F: int, K: int) -> np.ndarray:
    """The CRF feature map F(x, z) in theta's layout."""
    phi = np.zeros(crf_size(F, K))
    W, trans, start, end = crf_unpack(phi, F, K)
    rows = np.repeat(np.arange(len(z)), np.diff(X.indptr))
    np.add.at(W, (X.indices, z[rows]), 1.0)
    np.add.at(trans, (z[:-1], z[1:]), 1.0)
    start[z[0]] += 1
    end[z[-1]] += 1
    return phi


def test_thm5_perceptron_mistake_bound():
    """Thm 5: on data separable with margin γ (by a unit-norm θ*) and ‖F(x, z) − F(x, z')‖ ≤ R, the perceptron makes at
    most R²/γ² mistakes before a full error-free pass — for several margins and example orders."""
    rng = _rng(3409)
    for gmin in (0.2, 0.06, 0.03):
        data, gamma, R = separable_instance(rng, 60, gmin)
        theta, mistakes = perceptron(data, 200, average=False, order=_rng(3410))
        assert mistakes <= R * R / (gamma * gamma)
        assert token_accuracy(crf_predict(theta, data)) == 1.0


# --------------------------------------------------------------------------------------------
# 8. Simulated worlds: the Bayes tagger is exact.
# --------------------------------------------------------------------------------------------
#
# Topic 12's chain (cm.hmm(): 8 states in two regimes, stationary start, length 24) with its emissions blended toward
# uniform, B_ε = (1 − ε)B + ε/V, so tags are not read off the tokens. In the misspecified world each word comes, with
# probability ρ, from the NEXT state's emission row: an HMM (one state per emission) cannot represent it, a CRF with
# the previous word as a feature can, and the Bayes tagger is still a chain (the emission becomes a pair potential).

EPS = 0.3
RHO = 0.5
WORLDS = ("hmm", "next")  # well specified (ρ = 0), misspecified (ρ = RHO)
SIM_N = (10, 30, 100, 300, 1000)
SIM_ALPHAS = (1.0, 0.1, 0.01)  # HMM add-α, tuned per n on a validation draw (a closed-form fit, so tuning is free)
SIM_EPOCHS = 5


def world_emissions() -> np.ndarray:
    _, B, _ = cm.hmm()
    return (1 - EPS) * B + EPS / B.shape[1]


def world_draw(k: int, n: int, world: str) -> tuple[np.ndarray, np.ndarray]:
    """n sequences (tokens, states). Both worlds share the stream's hidden paths (common random numbers): the same
    states, with tokens redrawn per world."""
    rng = _rng(k)
    _, z = cm.sample_hmm(rng, n)
    B = world_emissions()
    N = z.shape[1]
    nxt = np.concatenate([z[:, 1:], z[:, -1:]], 1)
    use_next = (rng.random(z.shape) < (RHO if world == "next" else 0.0)) & (np.arange(N) < N - 1)
    src = np.where(use_next, nxt, z)
    x = (rng.random(z.shape)[..., None] > B.cumsum(1)[src]).sum(-1)
    return np.minimum(x, B.shape[1] - 1), z


def world_chain(x: np.ndarray, world: str, prefix: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The true conditional p(z | x) as a chain: U carries log π at t = 0 (and the last word's emission), P carries
    log T plus each word's emission log((1 − ρ)B[z_t, x_t] + ρ B[z_{t+1}, x_t]). With ``prefix``, x is the start of a
    longer sequence, so the last word may still come from the unseen next state: its emission is
    (1 − ρ)B[z, x] + ρ Σ_k T[z, k] B[k, x]."""
    T, _, pi = cm.hmm()
    B = world_emissions()
    rho = RHO if world == "next" else 0.0
    n, N = x.shape
    U = np.zeros((n, N, len(pi)))
    U[:, 0] += np.log(pi)
    last = (1 - rho) * B + rho * (T @ B) if prefix else B
    U[:, -1] += np.log(last[:, x[:, -1]].T)
    Bx = B[:, x[:, :-1]].transpose(1, 2, 0)
    P = np.log(T)[None, None] + np.log((1 - rho) * Bx[:, :, :, None] + rho * Bx[:, :, None, :])
    return U, P


def lagged_bayes(world: str, w: int, x: np.ndarray, z: np.ndarray) -> float:
    """Token accuracy of the Bayes rule that reads only x_{1:t+w} for z_t (posterior mode of z_t on the prefix chain):
    by Cor 2, no MBR-decoded MEMM with lookahead w beats it."""
    N = x.shape[1]
    hits = 0
    for t in range(N):
        e = min(N, t + w + 1)
        node = chain_marginals(*world_chain(x[:, :e], world, prefix=e < N))[1]
        hits += int((node[:, t].argmax(1) == z[:, t]).sum())
    return hits / z.size


def _features(xs: np.ndarray, w: int) -> list[list[list[str]]]:
    return [window_features(x, w) for x in xs]


def cond_nll(U: np.ndarray, P: np.ndarray, z: np.ndarray) -> float:
    """Conditional log-loss per token, Σ_i [log Z(x_i) − score(z_i)] / Σ_i T_i, in nats."""
    return float((chain_forward(U, P)[1] - path_score(U, P, z)).sum() / z.size)


def crf_eval(theta: np.ndarray, data: Data) -> tuple[float, float]:
    """(token accuracy by Viterbi, conditional log-loss per token) of a CRF on a labeled set."""
    hit = loss = tok = 0.0
    for bk in data.buckets:
        U, P = crf_potentials(theta, bk.X, bk.z.shape[1], data.F, data.K)
        hit += (viterbi(U, P)[0] == bk.z).sum()
        loss += (chain_forward(U, P)[1] - path_score(U, P, bk.z)).sum()
        tok += bk.z.size
    return float(hit / tok), float(loss / tok)


def hmm_potentials_batch(m: HMM, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """hmm_potentials for a batch of equal-length sequences x (n, T)."""
    U = m.emit[:, x].transpose(1, 2, 0).copy()
    U[:, 0] += m.start
    U[:, -1] += m.end
    return U, m.trans


@lru_cache(maxsize=None)
def world_curves(world: str) -> dict:
    """Learning curves in one world: test token accuracy and conditional log-loss of the HMM (add-α tuned), the
    indicator CRF (w = 0, the HMM's own conditional family by Thm 3), the window CRF (w = 1), and the averaged
    perceptron (w = 1, SIM_EPOCHS epochs; accuracy only) at every n of SIM_N, against the Bayes tagger. Each CRF is
    the MAP estimate under a standard normal prior on θ (λ = 1/n on the mean objective); the HMM's add-α is chosen on
    a fresh validation draw by conditional log-loss (a proper score)."""
    K, V = world_emissions().shape
    xtr, ztr = world_draw(3420, max(SIM_N), world)  # training sets are prefixes of one draw
    xva, zva = world_draw(3421, 500, world)
    xte, zte = world_draw(3422, 2000, world)
    Ub, Pb = world_chain(xte, world)
    node = chain_marginals(Ub, Pb)[1]
    out = {"n": list(SIM_N), "bayes": float((viterbi(Ub, Pb)[0] == zte).mean()),
           "bayes_mbr": float((node.argmax(2) == zte).mean()), "bayes_nll": cond_nll(Ub, Pb, zte)}
    for name in ("hmm", "crf0", "crf1", "perceptron", "hmm_nll", "crf0_nll", "crf1_nll"):
        out[name] = []
    for n in SIM_N:
        fits = [hmm_fit(list(xtr[:n]), list(ztr[:n]), K, V, alpha=a, with_end=False) for a in SIM_ALPHAS]
        h = min(fits, key=lambda f: cond_nll(*hmm_potentials_batch(f, xva), zva))
        U, P = hmm_potentials_batch(h, xte)
        out["hmm"].append(float((viterbi(U, P)[0] == zte).mean()))
        out["hmm_nll"].append(cond_nll(U, P, zte))
        for w in (0, 1):
            ftr = _features(xtr[:n], w)
            vocab = index_features(ftr)
            dtr = make_data(ftr, list(ztr[:n]), vocab, K)
            dte = make_data(_features(xte, w), list(zte), vocab, K)
            acc, nll = crf_eval(crf_fit_lbfgs(dtr, 1.0 / n), dte)
            out[f"crf{w}"].append(acc)
            out[f"crf{w}_nll"].append(nll)
            if w == 1:
                th, _ = perceptron(dtr, SIM_EPOCHS, order=_rng(3423))
                out["perceptron"].append(crf_eval(th, dte)[0])
    return out


@lru_cache(maxsize=None)
def memm_table(world: str) -> dict:
    """Cor 2 at n = max(SIM_N): for lookahead w ∈ {0, 1} (two words of left context), the lag-w Bayes ceiling, the
    MEMM decoded by posterior marginals (bounded by the ceiling), and the same MEMM decoded by Viterbi (not bounded:
    the max over futures is not constant even though their sum is)."""
    K = world_emissions().shape[0]
    xtr, ztr = world_draw(3420, max(SIM_N), world)
    xte, zte = world_draw(3422, 2000, world)
    out = {"w": [0, 1], "ceiling": [], "mbr": [], "viterbi": []}
    for w in (0, 1):
        ftr = [window_features(x, w, left=2) for x in xtr]
        vocab = index_features(ftr)
        W, b = memm_fit(make_data(ftr, list(ztr), vocab, K), 1.0 / ztr.size)  # MAP under N(0, I): λ = 1/tokens
        bk = make_data([window_features(x, w, left=2) for x in xte], list(zte), vocab, K).buckets[0]
        U, P = memm_potentials(W, b, bk.X, bk.z.shape[1], K)
        out["ceiling"].append(lagged_bayes(world, w, xte, zte))
        out["mbr"].append(float((chain_marginals(U, P)[1].argmax(2) == bk.z).mean()))
        out["viterbi"].append(float((viterbi(U, P)[0] == bk.z).mean()))
    return out


# --------------------------------------------------------------------------------------------
# 9. Finance case study: BIO tags over the corpus's gold entity spans.
# --------------------------------------------------------------------------------------------

TYPES = ("ORG", "PERSON", "GPE", "PRODUCT", "SECTOR", "DATE", "MONEY", "PERCENT")
TAGS = ("O", *(f"{p}-{t}" for t in TYPES for p in "BI"))
TAG_ID = {t: i for i, t in enumerate(TAGS)}
FIN_TRAIN, FIN_DEV, FIN_TEST = (300, 0), (60, 2), (200, 1)  # generate_corpus(n_docs, seed) runs
FIN_N = (3, 10, 30)  # training documents, prefixes of FIN_TRAIN
FIN_ALPHAS = (1.0, 0.1, 0.01)


def bio_sentences(docs) -> tuple[list[list[str]], list[np.ndarray], list[int]]:
    """Each sentence's tokens (fc._tokens) and BIO tags from the character-offset entity spans; the third list gives
    each sentence's document index. No span crosses a sentence or cuts a token (test_bio_alignment)."""
    xs, zs, doc = [], [], []
    for i, d in enumerate(docs):
        for s, e in d.sentences:
            toks = fc._tokens(d.text[s:e])
            if not toks:
                continue
            z = np.zeros(len(toks), int)
            for en in d.entities:
                if s <= en.start and en.end <= e:
                    a = len(fc._tokens(d.text[s:en.start]))
                    b = a + len(fc._tokens(d.text[en.start:en.end]))
                    z[a] = TAG_ID["B-" + en.type]
                    z[a + 1:b] = TAG_ID["I-" + en.type]
            xs.append(toks)
            zs.append(z)
            doc.append(i)
    return xs, zs, doc


def spans(z: np.ndarray) -> set[tuple[int, int, str]]:
    """Entity spans (start, end, type) read off a BIO sequence, conlleval-style: an I- that does not continue a span
    of its type opens one."""
    out, cur = set(), None
    for t, k in enumerate(list(z) + [0]):
        tag = TAGS[k]
        if cur is not None and not (tag.startswith("I-") and tag[2:] == cur[1]):
            out.add((cur[0], t, cur[1]))
            cur = None
        if tag != "O" and cur is None:
            cur = (t, tag[2:])
    return out


def span_f1(golds: list[np.ndarray], preds: list[np.ndarray]) -> float:
    """Exact-match span F1, micro-averaged over every entity of every sentence."""
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        G, Pd = spans(g), spans(p)
        tp += len(G & Pd)
        fp += len(Pd - G)
        fn += len(G - Pd)
    return 2 * tp / (2 * tp + fp + fn)


def fin_features(toks: list[str], causal: bool) -> list[list[str]]:
    """The word, its 3-letter prefix and suffix, and the words two to the left — and, unless causal, two to the right.
    No gazetteer: names are learned only from training text."""
    T = len(toks)
    out = []
    for t, wd in enumerate(toks):
        fs = ["bias", f"w0={wd}", f"pre3={wd[:3]}", f"suf3={wd[-3:]}"]
        for d in (1, 2):
            fs.append(f"w-{d}={toks[t - d]}" if t >= d else f"w-{d}=<s>")
            if not causal:
                fs.append(f"w+{d}={toks[t + d]}" if t + d < T else f"w+{d}=</s>")
        out.append(fs)
    return out


@lru_cache(maxsize=1)
def fin_splits() -> dict:
    out = {}
    for name, (n, seed) in (("train", FIN_TRAIN), ("dev", FIN_DEV), ("test", FIN_TEST)):
        out[name] = bio_sentences(fc.generate_corpus(n, seed))
    return out


def _sorted_preds(data: Data, preds: list[np.ndarray], order: list[int]) -> list[np.ndarray]:
    """Back from length buckets to the original sentence order."""
    flat = [row for p in preds for row in p]
    out: list[np.ndarray | None] = [None] * len(order)
    for pos, i in enumerate(order):
        out[i] = flat[pos]
    return out


def _bucket_order(zs: list[np.ndarray]) -> list[int]:
    by_len: dict[int, list[int]] = {}
    for i, z in enumerate(zs):
        by_len.setdefault(len(z), []).append(i)
    return [i for L in sorted(by_len) for i in by_len[L]]


def fin_scores(xs, zs, preds: list[np.ndarray], train_vocab) -> dict:
    """Token accuracy, span F1, and accuracy on two slices: the word "the" (in an entity iff the NEXT word starts a
    PRODUCT or DATE) and tokens whose word never occurs in training."""
    g, p = np.concatenate(zs), np.concatenate(preds)
    words = np.array([w for x in xs for w in x])
    the = words == "the"
    oov = ~np.isin(words, list(train_vocab))
    return {"acc": float((g == p).mean()), "f1": span_f1(zs, preds), "the": float((g[the] == p[the]).mean()),
            "oov": float((g[oov] == p[oov]).mean()), "oov_share": float(oov.mean())}


def _enc(wid: dict[str, int], x: list[str]) -> np.ndarray:
    return np.array([wid.get(w, len(wid)) for w in x])


@lru_cache(maxsize=None)
def fin_models(n_docs: int) -> dict:
    """Every finance tagger trained on the first n_docs training documents. CRF and MEMMs are MAP under N(0, I)
    (λ = 1/sentences, 1/tokens); the HMM's add-α is chosen by dev conditional log-loss."""
    sp_ = fin_splits()
    xtr, ztr, dtr_doc = sp_["train"]
    ns = int(np.searchsorted(dtr_doc, n_docs))
    xtr, ztr = xtr[:ns], ztr[:ns]
    xva, zva, _ = sp_["dev"]
    K = len(TAGS)
    # HMM: words seen once in training share an <unk> emission, and so do test words never seen (Prop 1, add-α).
    cnt: dict[str, int] = {}
    for x in xtr:
        for w in x:
            cnt[w] = cnt.get(w, 0) + 1
    wid = {w: i for i, w in enumerate(sorted(w for w, c in cnt.items() if c >= 2))}
    fits = [hmm_fit([_enc(wid, x) for x in xtr], ztr, K, len(wid) + 1, alpha=a) for a in FIN_ALPHAS]
    hmm = min(fits, key=lambda f: sum(cond_nll(*hmm_potentials(f, _enc(wid, x)), z[None]) * len(z)
                                      for x, z in zip(xva, zva)))
    out = {"vocab_words": frozenset(cnt), "hmm": (hmm, wid)}
    for causal in (False, True):
        ftr = [fin_features(x, causal) for x in xtr]
        vocab = index_features(ftr)
        dtr = make_data(ftr, ztr, vocab, K)
        tag = "causal" if causal else "full"
        out[f"memm_{tag}"] = (memm_fit(dtr, 1.0 / sum(len(z) for z in ztr)), vocab)
        if not causal:
            out["crf"] = (crf_fit_lbfgs(dtr, 1.0 / len(ztr)), vocab)
            out["perceptron"] = (perceptron(dtr, SIM_EPOCHS, order=_rng(3430))[0], vocab)
    return out


def fin_chain(models: dict, name: str, x: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One sentence's chain under a fitted finance tagger (the perceptron's scores are not probabilities)."""
    K = len(TAGS)
    if name == "hmm":
        h, wid = models["hmm"]
        return hmm_potentials(h, _enc(wid, x))
    params, vocab = models[name]
    data = make_data([fin_features(x, name == "memm_causal")], [np.zeros(len(x), int)], vocab, K)
    X = data.buckets[0].X
    if name.startswith("memm"):
        return memm_potentials(*params, X, len(x), K)
    return crf_potentials(params, X, len(x), data.F, K)


@lru_cache(maxsize=None)
def fin_run(n_docs: int) -> dict:
    """Test scores of every finance tagger (Viterbi decoding) trained on n_docs documents."""
    models = fin_models(n_docs)
    xte, zte, _ = fin_splits()["test"]
    K = len(TAGS)
    out = {}
    for name in ("hmm", "memm_causal", "memm_full", "crf", "perceptron"):
        if name == "hmm":
            preds = [viterbi(*fin_chain(models, name, x))[0][0] for x in xte]
        else:
            params, vocab = models[name]
            dte = make_data([fin_features(x, name == "memm_causal") for x in xte], zte, vocab, K)
            pred = []
            for bk in dte.buckets:
                L = bk.z.shape[1]
                U, P = (memm_potentials(*params, bk.X, L, K) if name.startswith("memm")
                        else crf_potentials(params, bk.X, L, dte.F, K))
                pred.append(viterbi(U, P)[0])
            preds = _sorted_preds(dte, pred, _bucket_order(zte))
        out[name] = fin_scores(xte, zte, preds, models["vocab_words"])
    return out


# ============================================================================================
# The lab's examples, chosen by rule.
# ============================================================================================

PROP2_PREFIX = "solenne power guided to percent growth in the"  # a test sentence's opening ("... the first quarter")
PROP2_CONTINUATIONS = ("first quarter", "prior year", "broader aerospace", "company plan", "kestrel platform")
PROP2_N_DOCS = 30
PERCEPTRON_GAMMAS = (0.3, 0.2, 0.12, 0.08, 0.06, 0.04)


@lru_cache(maxsize=1)
def trellis_example() -> dict:
    """The first test sequence of the well-specified world whose Viterbi path and posterior-marginal path differ in at
    least two positions and whose mode is unique (in every column the runner-up max-marginal is at most 99% of the
    mode's probability), under the true model: marginals, max-marginals, both paths, and the gold states."""
    x, z = world_draw(3422, 2000, "hmm")
    U, P = world_chain(x, "hmm")
    vit = viterbi(U, P)[0]
    node = chain_marginals(U, P)[1]
    mbr = node.argmax(2)
    for i in np.flatnonzero((vit != mbr).sum(1) >= 2):
        mm = max_marginals(U[i:i + 1], P[i:i + 1])[0]
        if np.all(np.sort(mm, 1)[:, -2] <= 0.99 * mm.max()):
            break
    i = int(i)
    return {"index": i, "x": x[i], "z": z[i], "node": node[i], "maxmarg": mm, "viterbi": vit[i], "mbr": mbr[i]}


@lru_cache(maxsize=1)
def prop2_example() -> dict:
    """Prop 2 on a finance sentence: the opening of a test sentence, completed by in-vocabulary continuations. For
    each, every token's marginals and the Viterbi tags under the CRF, the causal MEMM, and the HMM (30 documents)."""
    models = fin_models(PROP2_N_DOCS)
    out = {}
    for cont in PROP2_CONTINUATIONS:
        x = (PROP2_PREFIX + " " + cont).split()
        out[cont] = {}
        for name in ("crf", "memm_causal", "hmm"):
            U, P = fin_chain(models, name, x)
            out[cont][name] = {"node": chain_marginals(U, P)[1][0], "viterbi": viterbi(U, P)[0][0]}
    return out


@lru_cache(maxsize=1)
def perceptron_sweep() -> list[dict]:
    """Thm 5 across margins: for each threshold, a separable training set (60 sequences), its margin γ and radius R,
    the perceptron's mistakes until an error-free pass, and the bound R²/γ²."""
    rng = _rng(3409)
    out = []
    for g in PERCEPTRON_GAMMAS:
        data, gamma, R = separable_instance(rng, 60, g)
        _, mistakes = perceptron(data, 200, average=False, order=_rng(3410))
        out.append({"threshold": g, "gamma": gamma, "R": R, "mistakes": mistakes, "bound": R * R / gamma ** 2})
    return out


# ============================================================================================
# Harness — the experiments' claims.
# ============================================================================================


def test_lbfgs_reaches_the_trust_region_optimum():
    """crf_fit_lbfgs (the experiments' optimizer) lands within 10⁻⁴ of trust-ncg's exact optimum and decodes the same
    test tags, on a 100-sequence training set of the misspecified world."""
    K = world_emissions().shape[0]
    x, z = world_draw(3420, 100, "next")
    f = _features(x, 1)
    data = make_data(f, list(z), index_features(f), K)
    a, b = crf_fit(data, 0.01, gtol=1e-9), crf_fit_lbfgs(data, 0.01)
    assert np.abs(a - b).max() < 1e-4
    xt, zt = world_draw(3422, 300, "next")
    dt = make_data(_features(xt, 1), list(zt), index_features(f), K)
    assert all(np.array_equal(p, q) for (_, p), (_, q) in zip(crf_predict(a, dt), crf_predict(b, dt)))


def test_bayes_tagger_and_thm4_on_the_worlds():
    """The Bayes tagger is exact in both worlds; decoded by posterior marginals it beats its own Viterbi path on token
    accuracy (Thm 4: MBR is the Hamming-optimal rule), and no learned tagger beats MBR."""
    for w in WORLDS:
        c = world_curves(w)
        assert c["bayes_mbr"] > c["bayes"]
        for name in ("hmm", "crf0", "crf1", "perceptron"):
            assert max(c[name]) < c["bayes_mbr"]
        assert min(c["hmm_nll"] + c["crf0_nll"] + c["crf1_nll"]) > c["bayes_nll"]


def test_well_specified_world_generative_leads_then_ties():
    """HMM world: the HMM (generative, closed form) leads the indicator CRF at n = 10 in accuracy and log-loss, and at
    n = 1000 the two agree to 0.001 in accuracy and log-loss — one family (Thm 3), one limit."""
    c = world_curves("hmm")
    assert c["hmm"][0] > c["crf0"][0] and c["hmm_nll"][0] < c["crf0_nll"][0]
    assert abs(c["hmm"][-1] - c["crf0"][-1]) < 1e-3 and abs(c["hmm_nll"][-1] - c["crf0_nll"][-1]) < 1e-3
    assert c["bayes_mbr"] - c["hmm"][-1] < 0.01


def test_misspecified_world_discriminative_wins():
    """Next-state world: the window CRF leads the HMM at every n, by more than 0.04 at n = 1000. The indicator CRF —
    the HMM's own conditional family — beats it on conditional log-loss at large n (what conditional MLE optimizes),
    while its accuracy does not: Cor 1 is a statement about the estimator's criterion."""
    c = world_curves("next")
    assert all(a > b for a, b in zip(c["crf1"], c["hmm"]))
    assert c["crf1"][-1] - c["hmm"][-1] > 0.04
    assert c["crf0_nll"][-1] < c["hmm_nll"][-1] and c["crf0_nll"][-2] < c["hmm_nll"][-2]
    assert c["crf0"][-1] < c["hmm"][-1]


def test_cor2_memm_is_capped_by_the_lagged_bayes_rule():
    """Cor 2: an MEMM with lookahead w, decoded by posterior marginals, scores below the Bayes rule that reads only
    x_{1:t+w}. Decoded by Viterbi, the causal MEMM beats that ceiling in the HMM world: Prop 2 binds marginals, and
    the max over futures still sees the future."""
    for w in WORLDS:
        t = memm_table(w)
        assert all(m_ < c for m_, c in zip(t["mbr"], t["ceiling"]))
        assert t["ceiling"][0] < t["ceiling"][1] < world_curves(w)["bayes_mbr"]
    t = memm_table("hmm")
    assert t["viterbi"][0] > t["ceiling"][0]


def test_bio_alignment():
    """Every entity span maps onto whole tokens of one sentence, and the BIO tags reproduce the spans exactly."""
    docs = fc.generate_corpus(*FIN_TRAIN)
    n_ent = sum(len(d.entities) for d in docs)
    xs, zs, _ = bio_sentences(docs)
    assert sum(len(spans(z)) for z in zs) == n_ent
    for d in docs:
        for en in d.entities:
            assert any(s <= en.start and en.end <= e for s, e in d.sentences)
            assert fc._tokens(d.text[en.start:en.end])  # no span vanishes under [a-z]+
    words = [w for x in xs for w in x]
    tags = np.concatenate(zs)
    for w in ("million", "percent"):  # MONEY and PERCENT are read off one word after tokenization
        assert all(TAGS[t] != "O" for wd, t in zip(words, tags) if wd == w)


def test_finance_claims():
    """Finance BIO (test documents): the HMM leads the CRF at 3 training documents; by 30 every discriminative tagger
    passes it, and the unpenalized averaged perceptron leads in accuracy and span F1. Unseen words are the hard
    tokens. The causal MEMM still tags "the" almost perfectly: in this corpus the left context settles it."""
    r3, r30 = fin_run(3), fin_run(30)
    assert r3["hmm"]["acc"] > r3["crf"]["acc"] and r3["hmm"]["f1"] > r3["crf"]["f1"]
    for name in ("memm_causal", "memm_full", "crf", "perceptron"):
        assert r30[name]["acc"] > r30["hmm"]["acc"]
    assert max(r30, key=lambda k: r30[k]["acc"]) == "perceptron" == max(r30, key=lambda k: r30[k]["f1"])
    for n in FIN_N:
        r = fin_run(n)
        assert all(r[k]["oov"] < r[k]["acc"] for k in r)
    assert r30["memm_causal"]["the"] > 0.99


def test_prop2_finance_example():
    """The lab's sentence: every continuation word was seen in training (the prefix is a real test sentence's, whose
    company the 30 training documents never name); the causal MEMM's marginals at "the" are the
    same for every continuation (Prop 2, to 10⁻¹²), while the CRF moves "the" from B-DATE ("first quarter") to O
    ("prior year"), and the MEMM's Viterbi path still opens a DATE there."""
    ex = prop2_example()
    seen = fin_models(PROP2_N_DOCS)["vocab_words"]
    t = len(PROP2_PREFIX.split()) - 1
    assert all(w in seen for c in PROP2_CONTINUATIONS for w in c.split())  # the edits stay in vocabulary
    base = ex[PROP2_CONTINUATIONS[0]]["memm_causal"]["node"][:t + 1]
    for c in PROP2_CONTINUATIONS:
        assert np.abs(ex[c]["memm_causal"]["node"][:t + 1] - base).max() < 1e-12
    assert TAGS[ex["first quarter"]["crf"]["viterbi"][t]] == "B-DATE"
    assert TAGS[ex["prior year"]["crf"]["viterbi"][t]] == "O" and ex["prior year"]["crf"]["node"][t, 0] > 0.95
    assert TAGS[ex["prior year"]["memm_causal"]["viterbi"][t]] == "B-DATE"


def test_trellis_example():
    """The lab's trellis: Viterbi and posterior-marginal paths differ; every column's largest max-marginal is the
    mode's probability, on the Viterbi label (Thm 1 in (max, ×)); the marginal path scores higher expected Hamming
    accuracy and the Viterbi path higher probability (Thm 4)."""
    ex = trellis_example()
    mm = ex["maxmarg"]
    assert np.allclose(mm.max(1), mm.max()) and np.array_equal(mm.argmax(1), ex["viterbi"])
    assert ex["node"][np.arange(len(ex["mbr"])), ex["mbr"]].sum() > ex["node"][np.arange(len(ex["viterbi"])), ex["viterbi"]].sum()
    tr = viz_payload()["trellis"]
    assert all(tr["maxmarg"][t][k] == 1.0 for t, k in enumerate(tr["viterbi"]))
    assert tr["correct"]["mbr"] > tr["correct"]["viterbi"] and tr["expected"]["mbr"] > tr["expected"]["viterbi"]


def test_perceptron_sweep():
    """Thm 5 at every margin of the sweep: mistakes ≤ R²/γ², and the bound grows as the margin shrinks."""
    sw = perceptron_sweep()
    assert all(r["mistakes"] <= r["bound"] for r in sw)
    assert all(a["bound"] < b["bound"] for a, b in zip(sw, sw[1:]))


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


# ============================================================================================
# Summary and the lab's payload.
# ============================================================================================

VIZ_JSON = _HERE.parent.parent / "src" / "components" / "viz" / "data" / "sequence-labeling-lab.json"


def summary() -> dict:
    out = {}
    for w in WORLDS:
        c, t = world_curves(w), memm_table(w)
        out[f"{w}: Bayes accuracy (MBR / Viterbi)"] = f"{c['bayes_mbr']:.6f} / {c['bayes']:.6f}"
        for name in ("hmm", "crf0", "crf1", "perceptron"):
            out[f"{w}: {name} acc by n"] = " ".join(f"{v:.6f}" for v in c[name])
        for name in ("hmm", "crf0", "crf1"):
            out[f"{w}: {name} nll by n"] = " ".join(f"{v:.6f}" for v in c[f"{name}_nll"])
        out[f"{w}: Bayes nll"] = c["bayes_nll"]
        out[f"{w}: MEMM w=0,1 ceiling / mbr / viterbi"] = " | ".join(
            f"{a:.6f} {b:.6f} {d:.6f}" for a, b, d in zip(t["ceiling"], t["mbr"], t["viterbi"]))
    for n in FIN_N:
        for name, r in fin_run(n).items():
            out[f"finance {n} docs: {name} acc f1 the oov"] = f"{r['acc']:.6f} {r['f1']:.6f} {r['the']:.6f} {r['oov']:.6f}"
        out[f"finance {n} docs: oov share"] = fin_run(n)["hmm"]["oov_share"]
    for r in perceptron_sweep():
        out[f"perceptron γ≥{r['threshold']}: γ R mistakes bound"] = (
            f"{r['gamma']:.6f} {r['R']:.6f} {r['mistakes']} {r['bound']:.1f}")
    return out


def _sig(x: float, k: int = 3):
    return None if not math.isfinite(x) else float(f"{x:.{k}g}")


def viz_payload() -> dict:
    """Every number the lab shows, rounded to the page's precision."""
    ex = trellis_example()
    st = cm.H_NAMES
    mm = ex["maxmarg"]
    T_ = np.arange(len(ex["z"]))
    trellis = {"tokens": [st[v] for v in ex["x"]], "gold": ex["z"].tolist(), "viterbi": ex["viterbi"].tolist(),
               "mbr": ex["mbr"].tolist(), "node": [_r(r) for r in ex["node"]],
               "maxmarg": [_r(r) for r in mm / mm.max()],  # relative to the mode: 1 exactly on the Viterbi path
               "mode_prob": _sig(float(mm.max())), "K": int(ex["node"].shape[1]),
               "correct": {d: int((ex[d] == ex["z"]).sum()) for d in ("viterbi", "mbr")},
               "expected": {d: _r(float(ex["node"][T_, ex[d]].sum())) for d in ("viterbi", "mbr")}}
    p2 = prop2_example()
    prop2 = {"prefix": PROP2_PREFIX.split(), "tags": list(TAGS), "continuations": []}
    for cont in PROP2_CONTINUATIONS:
        row = {"words": cont.split(), "models": {}}
        for name, v in p2[cont].items():
            row["models"][name] = {"viterbi": v["viterbi"].tolist(), "top": [
                [[int(k), _r(col[k])] for k in np.argsort(col)[::-1][:3]] for col in v["node"]]}
        prop2["continuations"].append(row)
    curves = {}
    for w in WORLDS:
        c, t = world_curves(w), memm_table(w)
        curves[w] = {k: (_r(v) if k != "n" else v) for k, v in c.items()}
        curves[w]["memm"] = {"w": t["w"], "ceiling": _r(t["ceiling"]), "mbr": _r(t["mbr"]), "viterbi": _r(t["viterbi"])}
    finance = {"n_docs": list(FIN_N), "models": {}}
    for name in ("hmm", "memm_causal", "memm_full", "crf", "perceptron"):
        finance["models"][name] = {k: [_r(fin_run(n)[name][k]) for n in FIN_N] for k in ("acc", "f1", "oov")}
    finance["oov_share"] = [_r(fin_run(n)["hmm"]["oov_share"]) for n in FIN_N]
    perc = [{"threshold": r["threshold"], "gamma": _r(r["gamma"]), "R": _r(r["R"]), "mistakes": r["mistakes"],
             "bound": _sig(r["bound"])} for r in perceptron_sweep()]
    return {"trellis": trellis, "prop2": prop2, "curves": curves, "finance": finance, "perceptron": perc}


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
    print("\nSummary:")
    for k, v in summary().items():
        print(f"  {k:62} {v:.6f}" if isinstance(v, float) else f"  {k:62} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__" and "--dev" in sys.argv:
    import time

    t0 = time.time()
    for k, v in summary().items():
        print(f"  {k:62} {v:.6f}" if isinstance(v, float) else f"  {k:62} {v}")
    print("time", round(time.time() - t0, 1))
elif __name__ == "__main__":
    raise SystemExit(_run_harness())
