"""Topic models: latent Dirichlet allocation — the reference implementation for formalNLP's twenty-fifth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. The generative model. LDA draws topics φ_k ~ Dir(β), proportions θ_d ~ Dir(α), and each token's topic and word
   (Def 1). A document's words are i.i.d. from p_d = Φᵀθ_d, a point of the topic polytope conv{φ_k} (Def 2, Prop 1).
   For fixed topics, fitting θ_d by maximum likelihood is the KL projection of the document's word frequencies onto
   that polytope (Prop 2); latent semantic analysis projects in Euclidean norm onto a signed subspace instead (Rem 1).
2. Identifiability. Relabeling topics leaves everything unchanged, and without further conditions the polytope is not
   determined by the documents (Prop 3). A topic that every document carries at one weight cannot be told apart from
   the others at all (Prop 4). Under separability (Def 3), the per-document co-occurrence estimator is unbiased for
   Q = ΦᵀRΦ (Prop 5), every row of the normalized co-occurrence matrix is a convex combination of the anchor words'
   rows, and when R is nonsingular the anchors are exactly the vertices (Thm 1), which Algorithm 1 finds.
3. Collapsed Gibbs. Integrating θ and Φ out (Lemma 1) leaves a full conditional in three counts (Thm 2); the sweep
   (Algorithm 2) leaves the exact posterior invariant and converges to it (Cor 1); one sample's point estimates are
   Dirichlet posterior means (Prop 6).
4. Mean-field VI. E[log θ_k] = ψ(γ_k) − ψ(Σγ) (Lemma 2); LDA's CAVI updates follow (Thm 3), and every block update
   raises the ELBO (Cor 2). CVB0 is the deterministic twin of the collapsed sampler (Rem 2).
5. Evaluation. Held-out log-likelihood and plug-in document completion (Def 4). The harmonic-mean estimator is
   consistent but biased upward, and its variance is infinite whenever a held-out document holds at least α_k tokens
   of topic k's anchor words (Thm 4); importance sampling from the prior is unbiased with finite variance (Prop 7).
   Coherence scores a topic's top words by document co-occurrence (Def 5).
6. Finance case study. The shared corpus's generator plants seven vocabularies, six sectors and one shared, and every
   document draws half its content nouns from its sector and half from the shared one. That is Prop 4's situation:
   the identifiable truth is six topics, and the anchor search finds one anchor per sector and none for the shared
   vocabulary. The right single-membership model beats LDA at document completion, but EM from random starts cannot
   find it. ``topic_mixtures`` is the stage the capstone imports.

Cross-checks: exact enumeration (Thm 2, Cor 1, Thm 4, Prop 5), linear programs for hull membership (Thm 1),
quadrature and central differences (Lemma 2, Thm 4), a direct convex solve (Prop 2), and scikit-learn's batch
variational Bayes for the ELBO (Thm 3).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python notebooks/topic-models-lda/topic_models_lda.py
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, sparse
from scipy.special import digamma, gammaln, logsumexp, xlogy

_HERE = Path(__file__).resolve().parent
for _dep in ("distributional-semantics-ppmi-svd", "linear-text-classifiers"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import distributional_semantics_ppmi_svd as ds  # noqa: E402 — module objects: their test_* stay out of this harness
import linear_text_classifiers as ltc  # noqa: E402

fc, ng = ds.fc, ds.ng

# ONE stream per random-instance claim and per initialization, each from default_rng([SEED, k]); k ≥ 3800 keeps this
# module's streams apart from Topics 1–24. The pure-Python samplers take a fixed integer seed (never hash()).
SEED = ds.SEED
_r = ltc._r


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


def log_beta(a: np.ndarray) -> np.ndarray:
    """log B(a) = Σ log Γ(a_k) − log Γ(Σ a_k), over the last axis."""
    a = np.asarray(a, dtype=float)
    return gammaln(a).sum(-1) - gammaln(a.sum(-1))


# --------------------------------------------------------------------------------------------
# 1. The generative model (Def 1, Def 2, Prop 1, Prop 2, Rem 1).
# --------------------------------------------------------------------------------------------
#
# Topics φ_k ∈ Δ^{V−1} are the rows of Φ (K × V); a document's proportions θ_d ∈ Δ^{K−1}. LDA draws φ_k ~ Dir(β𝟏),
# θ_d ~ Dir(α), z_dn ~ Cat(θ_d), w_dn ~ Cat(φ_{z_dn}) (Def 1). Given θ_d the words are i.i.d. from p_d = Φᵀθ_d, a
# point of the topic polytope T(Φ) = conv{φ_k} (Def 2, Prop 1).


def sample_counts(rng: np.random.Generator, Phi: np.ndarray, alpha: np.ndarray, n_docs: int,
                  length: int) -> tuple[np.ndarray, np.ndarray]:
    """Def 1 with Φ given: (X, Θ), X the n_docs × V count matrix of documents of ``length`` tokens. Given θ_d the
    counts are Multinomial(length, Φᵀθ_d) (Prop 1), so tokens need not be drawn one topic at a time."""
    Theta = rng.dirichlet(alpha, n_docs)
    P = Theta @ Phi
    P /= P.sum(1, keepdims=True)
    return rng.multinomial(length, P), Theta


def tokens_of(X: np.ndarray) -> list[list[int]]:
    """Bag-of-words documents as token lists (word ids in increasing order; LDA is exchangeable within a document)."""
    return [np.repeat(np.arange(X.shape[1]), row).tolist() for row in np.asarray(X, dtype=int)]


def counts_of(docs: list[list[int]], V: int) -> np.ndarray:
    X = np.zeros((len(docs), V))
    for d, doc in enumerate(docs):
        np.add.at(X[d], doc, 1)
    return X


def fit_theta(x: np.ndarray, Phi: np.ndarray, iters: int = 20_000, tol: float = 1e-14) -> np.ndarray:
    """Prop 2: argmax_θ Σ_w x_w log (Φᵀθ)_w over the simplex, by EM (each step raises the likelihood)."""
    K = Phi.shape[0]
    w = np.nonzero(x)[0]
    c, P = x[w], Phi[:, w]
    th = np.full(K, 1.0 / K)
    for _ in range(iters):
        R = th[:, None] * P
        R /= R.sum(0, keepdims=True)
        new = R @ c / c.sum()
        if np.abs(new - th).max() < tol:
            return new
        th = new
    return th


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def entropy(p: np.ndarray) -> float:
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


# The simplex tab: V = 3 words, K = 3 topics, so the topic polytope is a triangle inside the word triangle.
SIMPLEX_PHI = np.array([[0.70, 0.20, 0.10],
                        [0.15, 0.70, 0.15],
                        [0.10, 0.25, 0.65]])
SIMPLEX_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0)
SIMPLEX_LENGTHS = (10, 40, 160)
SIMPLEX_DOCS = 80


def ternary(P: np.ndarray) -> np.ndarray:
    """Barycentric (p₀, p₁, p₂) → plane: vertices at (0, 0), (1, 0), (½, √3/2)."""
    P = np.atleast_2d(P)
    return np.column_stack([P[:, 1] + 0.5 * P[:, 2], P[:, 2] * math.sqrt(3) / 2])


@lru_cache(maxsize=None)
def simplex_world(alpha: float, length: int) -> dict:
    """Documents of one (α, n_d) cell: θ_d, the exact p_d, the observed p̂_d = x_d/n_d, and p̂_d's KL projection onto
    the topic triangle (Prop 2). One stream per α (k = 3800 + index), shared across lengths, so the same θ_d appear
    at every n_d and only the sampling noise changes."""
    a_idx = SIMPLEX_ALPHAS.index(alpha)
    rng = _rng(3800 + a_idx)
    Theta = rng.dirichlet(np.full(3, alpha), SIMPLEX_DOCS)
    P = Theta @ SIMPLEX_PHI
    Xs = {n: _rng(3810 + 10 * a_idx + SIMPLEX_LENGTHS.index(n)).multinomial(n, P / P.sum(1, keepdims=True))
          for n in SIMPLEX_LENGTHS}
    X = Xs[length]
    Phat = X / length
    proj = np.array([fit_theta(x, SIMPLEX_PHI) @ SIMPLEX_PHI for x in X])
    return {"theta": Theta, "p": P, "phat": Phat, "proj": proj,
            "outside": float(np.mean([kl(ph, pr) > 1e-9 for ph, pr in zip(Phat, proj)]))}


# --------------------------------------------------------------------------------------------
# 2. Identifiability and anchor words (Prop 3, Prop 4, Def 3, Prop 5, Thm 1, Algorithm 1).
# --------------------------------------------------------------------------------------------


def prop3_construction() -> dict:
    """Prop 3(b): V = 3, K = 2. Every p_d lies on the segment [φ₂, φ₁]; stretching it to [φ₂′, φ₁′] ⊃ [φ₂, φ₁]
    inside the simplex gives other topics and proportions with exactly the same P = ΘΦ."""
    Phi = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])
    t = 0.1
    Phi2 = np.array([Phi[0] + t * (Phi[0] - Phi[1]), Phi[1] + t * (Phi[1] - Phi[0])])
    Theta = _rng(3820).dirichlet([1.0, 1.0], 25)
    P = Theta @ Phi
    # p = θ₁φ₁ + θ₂φ₂ with φ₁ = (φ₁′(1 + t) + tφ₂′)/(1 + 2t) and φ₂ = (tφ₁′ + (1 + t)φ₂′)/(1 + 2t).
    M = np.array([[1 + t, t], [t, 1 + t]]) / (1 + 2 * t)   # Φ = M Φ′
    Theta2 = Theta @ M
    return {"Phi": Phi, "Phi2": Phi2, "Theta": Theta, "Theta2": Theta2, "P": P, "P2": Theta2 @ Phi2, "M": M}


def prop4_world(c: float = 0.5) -> dict:
    """Prop 4: S = 3 own topics plus a shared φ₀ carried by every document at weight c. Every document's words are
    i.i.d. from ψ_s = (1 − c)φ_s + cφ₀, a single-membership model; R = E[θθᵀ] has rank S."""
    rng = _rng(3821)
    S, V = 3, 9
    Phi0 = rng.dirichlet(np.ones(V))
    Phis = rng.dirichlet(np.ones(V), S)
    pi = np.array([0.5, 0.3, 0.2])
    Psi = (1 - c) * Phis + c * Phi0
    vs = np.hstack([(1 - c) * np.eye(S), np.full((S, 1), c)])   # row s: θ of a sector-s document (S × (S + 1))
    R = (vs.T * pi) @ vs
    return {"S": S, "V": V, "c": c, "Phi0": Phi0, "Phis": Phis, "pi": pi, "Psi": Psi, "thetas": vs, "R": R}


def cooccurrence(X: np.ndarray) -> np.ndarray:
    """Prop 5: Q̂ = average over documents with n_d ≥ 2 of (x_d x_dᵀ − diag x_d)/(n_d(n_d − 1))."""
    X = np.asarray(X, dtype=float)
    n = X.sum(1)
    keep = n >= 2
    Xk, nk = X[keep], n[keep]
    W = Xk / np.sqrt(nk * (nk - 1))[:, None]
    return (W.T @ W - np.diag((Xk / (nk * (nk - 1))[:, None]).sum(0))) / keep.sum()


def normalize_rows(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Q̄, p): p = Q𝟏 the word marginal, Q̄ = diag(p)⁻¹Q (rows of zero-marginal words left at 0)."""
    p = Q.sum(1)
    Qbar = np.divide(Q, p[:, None], out=np.zeros_like(Q), where=p[:, None] > 0)
    return Qbar, p


def dirichlet_second_moment(alpha: np.ndarray) -> np.ndarray:
    """R = E[θθᵀ] for θ ~ Dir(α): (ααᵀ + diag α)/(α₀(α₀ + 1)) (formalStatistics multivariate-distributions Thm 5)."""
    a0 = alpha.sum()
    return (np.outer(alpha, alpha) + np.diag(alpha)) / (a0 * (a0 + 1))


def hull_distance(y: np.ndarray, B: np.ndarray) -> float:
    """min ‖Bᵀλ − y‖₁ over λ in the simplex, by a linear program: 0 iff y ∈ conv(rows of B)."""
    K, V = B.shape
    # variables: λ (K), s (V) with −s ≤ Bᵀλ − y ≤ s
    c = np.concatenate([np.zeros(K), np.ones(V)])
    A_ub = np.block([[B.T, -np.eye(V)], [-B.T, -np.eye(V)]])
    b_ub = np.concatenate([y, -y])
    A_eq = np.concatenate([np.ones(K), np.zeros(V)])[None]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (K + V),
                           method="highs")
    assert res.status == 0, res.message
    return float(res.fun)


def fast_anchor_words(Qbar: np.ndarray, K: int, candidates: np.ndarray) -> list[int]:
    """Algorithm 1, first half (Arora et al. 2013, Alg. 4): the candidate row farthest from the origin, then
    repeatedly the row farthest from the affine span of those chosen (Gram–Schmidt on differences)."""
    R = Qbar[candidates].astype(float).copy()
    first = int(np.argmax((R ** 2).sum(1)))
    chosen = [first]
    R = R - R[first]
    basis: list[np.ndarray] = []
    for _ in range(1, K):
        for b in basis[-1:]:   # earlier directions were already removed when they were added
            R -= np.outer(R @ b, b)
        j = int(np.argmax((R ** 2).sum(1)))
        chosen.append(j)
        basis.append(R[j] / np.linalg.norm(R[j]))
    return [int(candidates[j]) for j in chosen]


def project_simplex_rows(Y: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex (sort-based)."""
    n, K = Y.shape
    U = -np.sort(-Y, axis=1)
    css = np.cumsum(U, axis=1) - 1
    idx = np.arange(1, K + 1)
    rho = ((U - css / idx) > 0).sum(1)
    tau = css[np.arange(n), rho - 1] / rho
    return np.maximum(Y - tau[:, None], 0.0)


def recover_l2(Qbar: np.ndarray, p: np.ndarray, anchors: list[int], iters: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1, second half (RecoverL2, Arora et al. 2013 §3): C_w = argmin over the simplex of
    ‖Q̄_w − C_wᵀQ̄_S‖², all words at once by accelerated projected gradient; then Bayes' rule,
    φ_k(w) ∝ C_wk p(w). Returns (Φ̂ (K × V), C (V × K))."""
    B = Qbar[anchors]
    G = B @ B.T
    L = float(np.linalg.eigvalsh(G).max())
    target = Qbar @ B.T
    K = len(anchors)
    C = np.full((Qbar.shape[0], K), 1.0 / K)
    Y, t = C.copy(), 1.0
    for _ in range(iters):
        Cn = project_simplex_rows(Y - (Y @ G - target) / L)
        tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
        Y = Cn + ((t - 1) / tn) * (Cn - C)
        C, t = Cn, tn
    A = C * p[:, None]
    return (A / A.sum(0)).T, C


def thm1_world(alpha: float = 0.5, singular: bool = False) -> dict:
    """Thm 1 at the population level: a separable Φ (K = 4, V = 12, word k is topic k's anchor) and Q = ΦᵀRΦ
    exactly. With ``singular``, topic 3 is shared by every document at weight ½ (Prop 4's R, rank 3)."""
    rng = _rng(3822)
    K, V = 4, 12
    Phi = np.zeros((K, V))
    Phi[:, K:] = rng.dirichlet(np.ones(V - K), K) * 0.9
    Phi[np.arange(K), np.arange(K)] = 0.1
    if singular:
        w = prop4_world()
        R = np.zeros((K, K))
        for s in range(3):
            v = np.zeros(K)
            v[s], v[3] = 0.5, 0.5
            R += w["pi"][s] * np.outer(v, v)
    else:
        R = dirichlet_second_moment(np.full(K, alpha))
    Q = Phi.T @ R @ Phi
    Qbar, p = normalize_rows(Q)
    C = (Phi * R.sum(1)[:, None]).T / p[:, None]   # C_wk = P(z = k | w) = φ_k(w) π_k / p(w)
    return {"K": K, "V": V, "Phi": Phi, "R": R, "Q": Q, "Qbar": Qbar, "p": p, "C": C, "anchors": list(range(K))}


# The anchors tab and recovery-vs-M curve: separable and non-separable worlds of V = 400 words, K = 5 topics.
ANCHOR_K, ANCHOR_V, ANCHOR_MASS, ANCHOR_ALPHA, ANCHOR_LEN = 5, 400, 0.05, 0.3, 50
ANCHOR_DF = 0.05   # candidates: words in at least this share of the documents
ANCHOR_M = (250, 1000, 4000, 16000)


@lru_cache(maxsize=None)
def anchor_phi(separable: bool) -> np.ndarray:
    """Dense topics over V words (Dir(𝟏) on the rest, so every other word has mass in every topic); in the
    separable world word k is topic k's anchor with mass ANCHOR_MASS, while the non-separable twin spreads each
    anchor's mass over every topic. (Sparse Dir(0.1) topics are already nearly separable: their frequent words carry
    almost all their mass in one topic, and the twin would recover as well as the separable world.)"""
    rng = _rng(3830)
    K, V = ANCHOR_K, ANCHOR_V
    Phi = np.zeros((K, V))
    Phi[:, K:] = rng.dirichlet(np.ones(V - K), K) * (1 - ANCHOR_MASS)
    if separable:
        Phi[np.arange(K), np.arange(K)] = ANCHOR_MASS
    else:
        Phi[:, :K] = ANCHOR_MASS * rng.dirichlet(np.ones(K), K)
    return Phi


@lru_cache(maxsize=None)
def anchor_corpus(separable: bool) -> tuple[np.ndarray, np.ndarray]:
    """The largest corpus once; smaller M are its prefixes (one stream, k = 3831)."""
    return sample_counts(_rng(3831), anchor_phi(separable), np.full(ANCHOR_K, ANCHOR_ALPHA), max(ANCHOR_M),
                         ANCHOR_LEN)


@lru_cache(maxsize=None)
def anchor_recovery(separable: bool, M: int | None) -> dict:
    """Algorithm 1 on the first M documents (M = None: the population Q = ΦᵀRΦ). Candidates are the words in at least
    ANCHOR_DF of the documents: a fixed count lets rare words, whose sampled rows are mostly noise, win the
    farthest-point search."""
    if M is None:
        gold = anchor_phi(separable)
        Q = gold.T @ dirichlet_second_moment(np.full(ANCHOR_K, ANCHOR_ALPHA)) @ gold
        cand = np.arange(ANCHOR_V)
    else:
        X = anchor_corpus(separable)[0][:M]
        Q = cooccurrence(X)
        cand = np.nonzero((X > 0).sum(0) >= ANCHOR_DF * M)[0]
    Qbar, p = normalize_rows(Q)
    anchors = fast_anchor_words(Qbar, ANCHOR_K, cand)
    Phi_hat, _ = recover_l2(Qbar, p, anchors)
    gold = anchor_phi(separable)
    perm = align_topics(Phi_hat, gold)
    return {"anchors": anchors, "l1": np.abs(Phi_hat[perm] - gold).sum(1), "Phi": Phi_hat[perm]}


# --------------------------------------------------------------------------------------------
# Topic alignment (Prop 3(a): topics are identified only up to relabeling).
# --------------------------------------------------------------------------------------------


def align_topics(est: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Match estimated topics to gold topics one-to-one, so that ``est[perm]`` lines up row by row with ``gold``.

    est: K_est × V, gold: K_gold × V, both with rows on the simplex. Returns integer array ``perm`` of length K_gold
    with perm[j] = the estimated topic matched to gold topic j. When K_est > K_gold (the finance K = 7 fits against
    the six identifiable topics), the unmatched estimated topics are left out.
    """
    # Cost: L1 distance (twice the total variation), bounded by 2 and finite under any smoothing. The Hungarian
    # algorithm minimizes the total cost over one-to-one matchings; with K_est > K_gold the rectangular problem
    # leaves the worst-fitting estimated topics unmatched.
    cost = np.abs(gold[:, None, :] - est[None, :, :]).sum(-1)   # K_gold × K_est
    rows, cols = optimize.linear_sum_assignment(cost)
    return cols[np.argsort(rows)]


# --------------------------------------------------------------------------------------------
# 3. Collapsed Gibbs sampling (Lemma 1, Thm 2, Algorithm 2, Cor 1, Prop 6).
# --------------------------------------------------------------------------------------------


def log_joint(ndk: np.ndarray, nkw: np.ndarray, alpha: np.ndarray, beta: float) -> float:
    """log p(w, z) with θ and Φ integrated out, by Lemma 1 twice:
    Σ_d [log B(α + n_d·) − log B(α)] + Σ_k [log B(β𝟏 + n_k·) − log B(β𝟏)]."""
    V = nkw.shape[1]
    return float((log_beta(alpha + ndk) - log_beta(alpha)).sum()
                 + (log_beta(beta + nkw) - log_beta(np.full(V, beta))).sum())


def log_prior_z(ndk: np.ndarray, alpha: np.ndarray) -> float:
    """log p(z) = Σ_d [log B(α + n_d·) − log B(α)]."""
    return float((log_beta(alpha + ndk) - log_beta(alpha)).sum())


def count_tables(docs: list[list[int]], z: list[list[int]], K: int, V: int) -> tuple[np.ndarray, np.ndarray]:
    ndk = np.zeros((len(docs), K))
    nkw = np.zeros((K, V))
    for d, (doc, zd) in enumerate(zip(docs, z)):
        np.add.at(ndk[d], zd, 1)
        np.add.at(nkw, (zd, doc), 1)
    return ndk, nkw


def full_conditional(docs, z, d: int, i: int, K: int, V: int, alpha: np.ndarray, beta: float) -> np.ndarray:
    """Thm 2: P(z_di = k | z_−di, w) ∝ (n_dk⁻ + α_k)(n_kw⁻ + β)/(n_k⁻ + Vβ), counts without token (d, i)."""
    ndk, nkw = count_tables(docs, z, K, V)
    k0, w = z[d][i], docs[d][i]
    ndk[d, k0] -= 1
    nkw[k0, w] -= 1
    q = (ndk[d] + alpha) * (nkw[:, w] + beta) / (nkw.sum(1) + V * beta)
    return q / q.sum()


@dataclass
class GibbsRun:
    z: list[list[int]]
    ndk: np.ndarray
    nkw: np.ndarray
    log_joint: list[float]   # after each sweep
    phi: np.ndarray          # Prop 6's means averaged over the kept sweeps
    theta: np.ndarray


def gibbs(docs: list[list[int]], V: int, K: int, alpha: float, beta: float, sweeps: int, seed: int,
          keep: int = 0, visit=None) -> GibbsRun:
    """Algorithm 2: systematic-scan collapsed Gibbs with Thm 2's conditional, in pure Python (a sweep over the 63k
    finance tokens takes a few hundredths of a second). The last ``keep`` sweeps' Prop 6 means are averaged. With
    ``visit``, the state is handed to it after every sweep instead (no trace)."""
    rnd = random.Random(seed)
    nkw = [[0] * V for _ in range(K)]
    nk = [0] * K
    ndk = [[0] * K for _ in docs]
    Z = []
    for d, doc in enumerate(docs):
        zd = [rnd.randrange(K) for _ in doc]
        Z.append(zd)
        for w, k in zip(doc, zd):
            nkw[k][w] += 1
            nk[k] += 1
            ndk[d][k] += 1
    Vb = V * beta
    a = np.full(K, float(alpha))
    trace = []
    phi_sum = np.zeros((K, V))
    theta_sum = np.zeros((len(docs), K))
    cum = [0.0] * K
    for s in range(sweeps):
        for d, doc in enumerate(docs):
            nd, zd = ndk[d], Z[d]
            for i, w in enumerate(doc):
                k = zd[i]
                nkw[k][w] -= 1
                nk[k] -= 1
                nd[k] -= 1
                tot = 0.0
                for j in range(K):
                    tot += (nd[j] + alpha) * (nkw[j][w] + beta) / (nk[j] + Vb)
                    cum[j] = tot
                u = rnd.random() * tot
                j = 0
                while cum[j] < u:
                    j += 1
                zd[i] = j
                nkw[j][w] += 1
                nk[j] += 1
                nd[j] += 1
        if visit is not None:
            visit(Z)
            continue
        A, B = np.array(ndk, float), np.array(nkw, float)
        trace.append(log_joint(A, B, a, beta))
        if s >= sweeps - keep:
            phi_sum += (B + beta) / (B.sum(1, keepdims=True) + Vb)
            theta_sum += (A + alpha) / (A.sum(1, keepdims=True) + K * alpha)
    A, B = np.array(ndk, float), np.array(nkw, float)
    if keep == 0:
        phi_sum = (B + beta) / (B.sum(1, keepdims=True) + Vb)
        theta_sum = (A + alpha) / (A.sum(1, keepdims=True) + K * alpha)
        keep = 1
    return GibbsRun(Z, A, B, trace, phi_sum / keep, theta_sum / keep)


# The toy corpus every exact claim runs on: two documents, 7 tokens, V = 3, K = 2 (2⁷ = 128 states).
TOY_DOCS = [[0, 1, 0, 2], [2, 2, 1]]
TOY_V, TOY_K, TOY_ALPHA, TOY_BETA = 3, 2, 0.5, 0.5


def toy_states(docs=TOY_DOCS, K: int = TOY_K):
    n = sum(map(len, docs))
    cuts = np.cumsum([0] + [len(d) for d in docs])
    for flat in itertools.product(range(K), repeat=n):
        yield [list(flat[cuts[d]:cuts[d + 1]]) for d in range(len(docs))]


def toy_posterior(docs=TOY_DOCS, V: int = TOY_V, K: int = TOY_K, alpha: float = TOY_ALPHA,
                  beta: float = TOY_BETA) -> tuple[list, np.ndarray, float]:
    """(states, p(z | w) over them, log p(w)) by enumeration."""
    a = np.full(K, alpha)
    states = list(toy_states(docs, K))
    lj = np.array([log_joint(*count_tables(docs, z, K, V), a, beta) for z in states])
    lw = float(logsumexp(lj))
    return states, np.exp(lj - lw), lw


def toy_kernel(site: int, docs=TOY_DOCS, V: int = TOY_V, K: int = TOY_K, alpha: float = TOY_ALPHA,
               beta: float = TOY_BETA) -> np.ndarray:
    """The single-site Gibbs kernel that resamples token ``site`` (flat index) from Thm 2's conditional."""
    states = list(toy_states(docs, K))
    index = {tuple(itertools.chain(*z)): j for j, z in enumerate(states)}
    cuts = np.cumsum([0] + [len(d) for d in docs])
    d = int(np.searchsorted(cuts, site, side="right") - 1)
    i = site - cuts[d]
    a = np.full(K, alpha)
    P = np.zeros((len(states), len(states)))
    for j, z in enumerate(states):
        q = full_conditional(docs, z, d, i, K, V, a, beta)
        for k in range(K):
            z2 = [list(x) for x in z]
            z2[d][i] = k
            P[j, index[tuple(itertools.chain(*z2))]] += q[k]
    return P


# --------------------------------------------------------------------------------------------
# 4. Mean-field variational inference (Lemma 2, Thm 3, Cor 2, Rem 2).
# --------------------------------------------------------------------------------------------


def dirichlet_expectation(g: np.ndarray) -> np.ndarray:
    """Lemma 2: E[log θ_k] = ψ(γ_k) − ψ(Σ_j γ_j) for θ ~ Dir(γ), over the last axis."""
    g = np.asarray(g, dtype=float)
    return digamma(g) - digamma(g.sum(-1, keepdims=True))


@dataclass
class CaviRun:
    lam: np.ndarray    # K × V, q(φ_k) = Dir(λ_k)
    gam: np.ndarray    # D × K, q(θ_d) = Dir(γ_d)
    elbo: list[float]  # after every block update (E-step, then M-step, per iteration)


@dataclass(frozen=True)
class _Entries:
    """The nonzero counts of X as flat arrays (document, word, count), with the scatter matrices that sum per-entry
    rows back to documents (D × nnz) and to words (V × nnz)."""
    di: np.ndarray
    wi: np.ndarray
    cts: np.ndarray
    to_doc: sparse.csr_matrix
    to_word: sparse.csr_matrix


def _entries(X: np.ndarray) -> _Entries:
    X = np.asarray(X, dtype=float)
    di, wi = np.nonzero(X)
    cts = X[di, wi]
    j = np.arange(len(di))
    return _Entries(di, wi, cts, sparse.csr_matrix((cts, (di, j)), shape=(X.shape[0], len(di))),
                    sparse.csr_matrix((cts, (wi, j)), shape=(X.shape[1], len(di))))


def _dirichlet_terms(gam: np.ndarray, lam: np.ndarray, alpha: float, beta: float) -> float:
    """E log p(θ) − E log q(θ) + E log p(φ) − E log q(φ), summed over documents and topics."""
    K, V = lam.shape
    Elt, Elb = dirichlet_expectation(gam), dirichlet_expectation(lam)
    a = np.full(K, alpha)
    th = ((a - 1) * Elt).sum() - len(gam) * log_beta(a) - ((gam - 1) * Elt).sum() + log_beta(gam).sum()
    ph = ((beta - 1) * Elb).sum() - K * log_beta(np.full(V, beta)) - ((lam - 1) * Elb).sum() + log_beta(lam).sum()
    return float(th + ph)


def elbo(E: _Entries, r: np.ndarray, gam: np.ndarray, lam: np.ndarray, alpha: float, beta: float) -> float:
    """The ELBO of smoothed LDA with explicit responsibilities r (one row per nonzero count: tokens of one type in
    one document share their optimal responsibility)."""
    L = dirichlet_expectation(gam)[E.di] + dirichlet_expectation(lam)[:, E.wi].T
    return float((E.cts * (r * L - xlogy(r, r)).sum(1)).sum()) + _dirichlet_terms(gam, lam, alpha, beta)


def elbo_star(X: np.ndarray, gam: np.ndarray, lam: np.ndarray, alpha: float, beta: float) -> float:
    """The ELBO maximized over the responsibilities in closed form (Thm 3's r is a softmax, so the token term is
    Σ x_dw log Σ_k exp(E log θ_dk + E log φ_kw)): the bound scikit-learn's ``_approx_bound`` reports."""
    E = _entries(X)
    L = dirichlet_expectation(gam)[E.di] + dirichlet_expectation(lam)[:, E.wi].T
    return float(E.cts @ logsumexp(L, axis=1)) + _dirichlet_terms(gam, lam, alpha, beta)


def cavi(X: np.ndarray, K: int, alpha: float, beta: float, iters: int, seed: int, local_tol: float = 1e-3,
         local_max: int = 100, lam0: np.ndarray | None = None, track: bool = True) -> CaviRun:
    """Thm 3 as batch variational EM for smoothed LDA, on scikit-learn's schedule. E-step: alternate
    r ← softmax(E log θ + E log φ) and γ ← α + Σ r, each document until its mean |Δγ| falls below ``local_tol``
    (at most ``local_max`` steps). Documents are independent given λ, so updating them together is still an exact
    block maximization, and the schedule affects which optimum is reached, never monotonicity. M-step:
    λ ← β + Σ r. λ starts at Gamma(100, 1/100) noise, as in scikit-learn. With ``track`` the explicit-r ELBO is
    recorded after every E-step and every M-step (Cor 2)."""
    E = _entries(X)
    D, V = np.asarray(X).shape
    lam = (np.random.default_rng([SEED, 3840, seed]).gamma(100.0, 0.01, (K, V)) if lam0 is None
           else np.array(lam0, dtype=float))
    gam = np.ones((D, K))
    trace: list[float] = []

    def resp(g: np.ndarray, Elb: np.ndarray) -> np.ndarray:
        L = dirichlet_expectation(g)[E.di] + Elb[:, E.wi].T
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))

    for _ in range(iters):
        Elb = dirichlet_expectation(lam)
        active = np.ones(D, dtype=bool)
        for _ in range(local_max):
            gn = alpha + E.to_doc @ resp(gam, Elb)
            moved = np.abs(gn - gam).mean(1)
            gam[active] = gn[active]
            active &= moved >= local_tol
            if not active.any():
                break
        r = resp(gam, Elb)
        if track:
            trace.append(elbo(E, r, gam, lam, alpha, beta))
        lam = beta + (E.to_word @ r).T
        if track:
            trace.append(elbo(E, r, gam, lam, alpha, beta))
    return CaviRun(lam, gam, trace)


# --------------------------------------------------------------------------------------------
# 5. Evaluation (Def 4, Thm 4, Prop 7, Def 5).
# --------------------------------------------------------------------------------------------


def fold_in(docs: list[list[int]], Phi: np.ndarray, alpha: float, sweeps: int, burn: int, seed: int) -> np.ndarray:
    """θ̂_d = E[θ_d | w_d, Φ] estimated by Gibbs over z with Φ fixed and θ integrated out (P(z = k | rest) ∝
    (n_dk⁻ + α)φ_k(w)), averaging (n_dk + α)/(n_d + Kα) over the sweeps after ``burn``."""
    rnd = random.Random(seed)
    K = Phi.shape[0]
    cols = [Phi[:, w].tolist() for w in range(Phi.shape[1])]
    out = np.zeros((len(docs), K))
    cum = [0.0] * K
    for d, doc in enumerate(docs):
        zd = [rnd.randrange(K) for _ in doc]
        nd = [0] * K
        for k in zd:
            nd[k] += 1
        acc = [0.0] * K
        for s in range(sweeps):
            for i, w in enumerate(doc):
                nd[zd[i]] -= 1
                col = cols[w]
                tot = 0.0
                for j in range(K):
                    tot += (nd[j] + alpha) * col[j]
                    cum[j] = tot
                u = rnd.random() * tot
                j = 0
                while cum[j] < u:
                    j += 1
                zd[i] = j
                nd[j] += 1
            if s >= burn:
                for j in range(K):
                    acc[j] += nd[j]
        m = sweeps - burn
        out[d] = (np.array(acc) / m + alpha) / (len(doc) + K * alpha)
    return out


def halves(docs: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Def 4's split: even positions estimate, odd positions are scored."""
    return [d[0::2] for d in docs], [d[1::2] for d in docs]


def completion(Phi: np.ndarray, theta_hat: np.ndarray, scored: list[list[int]]) -> float:
    """Def 4: plug-in document completion, the mean over scored tokens of log Σ_k θ̂_dk φ_k(w)."""
    tot, n = 0.0, 0
    for th, doc in zip(theta_hat, scored):
        if doc:
            tot += float(np.log(th @ Phi[:, doc]).sum())
            n += len(doc)
    return tot / n


def mixture_of_unigrams(X: np.ndarray, K: int, beta: float, iters: int, *, init: np.ndarray | None = None,
                        seed: int = 0) -> dict:
    """EM for the single-membership model: π_k, φ_k with add-β smoothing (MAP under Dir(β + 1)), responsibilities
    over whole documents. ``init`` gives starting responsibilities (the gold sectors), else Dirichlet noise. Stops
    when an iteration raises the log-likelihood by less than 10⁻¹⁰ relative."""
    X = np.asarray(X, dtype=float)
    r = (np.random.default_rng([SEED, 3850, seed]).dirichlet(np.ones(K), X.shape[0]) if init is None
         else np.array(init, dtype=float))
    ll = -np.inf
    for _ in range(iters):
        pi = (r.sum(0) + 1e-12) / (r.sum() + K * 1e-12)
        phi = r.T @ X + beta
        phi /= phi.sum(1, keepdims=True)
        L = np.log(pi) + X @ np.log(phi).T
        new = float(logsumexp(L, axis=1).sum())
        r = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        if new - ll < 1e-10 * abs(new):
            ll = new
            break
        ll = new
    return {"pi": pi, "phi": phi, "r": r, "loglik": ll}


def mou_posterior(model: dict, docs: list[list[int]]) -> np.ndarray:
    """P(cluster | the given words) for each document: the single-membership model's θ̂ for Def 4."""
    lp = np.log(model["phi"])
    out = np.zeros((len(docs), len(model["pi"])))
    for d, doc in enumerate(docs):
        s = np.log(model["pi"]) + lp[:, doc].sum(1)
        out[d] = np.exp(s - logsumexp(s))
    return out


def toy_marginals(docs, V: int, K: int, alpha: float, beta: float) -> dict:
    """Exact pieces of Thm 4 and Prop 7 on an enumerable corpus: log p(w), and over the states z the posterior
    p(z | w), the prior p(z), and the likelihood p(w | z) = p(w, z)/p(z)."""
    a = np.full(K, alpha)
    states = list(toy_states(docs, K))
    tabs = [count_tables(docs, z, K, V) for z in states]
    lj = np.array([log_joint(ndk, nkw, a, beta) for ndk, nkw in tabs])
    lpz = np.array([log_prior_z(ndk, a) for ndk, _ in tabs])
    lw = float(logsumexp(lj))
    return {"log_pw": lw, "post": np.exp(lj - lw), "prior": np.exp(lpz), "loglik": lj - lpz}


def hm_relative_variance(m: dict) -> float:
    """Var_{p(z|w)}[1/p(w|z)] · p(w)² — the relative variance of one harmonic-mean term (Thm 4)."""
    inv = np.exp(-m["loglik"] - (-m["log_pw"]))   # p(w)/p(w|z)
    return float(m["post"] @ inv ** 2 - 1.0)


def is_relative_variance(m: dict) -> float:
    """Var_{p(z)}[p(w|z)] / p(w)² — the relative variance of one prior-importance-sampling term (Prop 7)."""
    ratio = np.exp(m["loglik"] - m["log_pw"])
    return float(m["prior"] @ ratio ** 2 - 1.0)


# Thm 4(c): one held-out document, Φ fixed and separable, θ continuous. K = 2, α = (0.5, 0.5); the document holds
# one token of topic 0's anchor (word 0) and otherwise the shared word 2, so c = 1 ≥ α₀ and the harmonic mean has
# infinite variance, from θ₀ → 0 alone (no token of topic 1's anchor, so nothing diverges at θ₀ → 1).
HM_PHI = np.array([[0.4, 0.0, 0.6], [0.0, 0.5, 0.5]])
HM_DOC = [0, 2, 2, 2, 2, 2]
HM_ALPHA = np.array([0.5, 0.5])


def hm_doc_likelihood(t: np.ndarray, doc=HM_DOC) -> np.ndarray:
    """p(w_d | θ = (t, 1 − t), Φ) on a grid of t."""
    t = np.atleast_1d(t)
    return np.prod([t * HM_PHI[0, w] + (1 - t) * HM_PHI[1, w] for w in doc], axis=0)


def hm_marginal(doc=HM_DOC) -> float:
    """p(w_d | Φ, α) by quadrature over θ₀ ~ Beta(α₀, α₁)."""
    from scipy.stats import beta as beta_dist
    val, _ = integrate.quad(lambda t: beta_dist.pdf(t, *HM_ALPHA) * hm_doc_likelihood(t, doc)[0], 0, 1,
                            epsabs=1e-14, epsrel=1e-12, limit=200)
    return float(val)


def hm_posterior_draws(rng: np.random.Generator, n: int, doc=HM_DOC) -> np.ndarray:
    """Exact posterior draws of θ₀ given w_d: the posterior is a finite mixture of Beta densities (expand the
    likelihood's product into monomials t^a (1 − t)^b), so draw a component, then a Beta."""
    coef = {(0, 0): 1.0}
    for w in doc:
        new: dict[tuple[int, int], float] = {}
        for (a, b), c in coef.items():
            if HM_PHI[0, w] > 0:
                new[(a + 1, b)] = new.get((a + 1, b), 0.0) + c * HM_PHI[0, w]
            if HM_PHI[1, w] > 0:
                new[(a, b + 1)] = new.get((a, b + 1), 0.0) + c * HM_PHI[1, w]
        coef = new
    keys = list(coef)
    logw = np.array([math.log(coef[k]) + float(log_beta(np.array([HM_ALPHA[0] + k[0], HM_ALPHA[1] + k[1]])))
                     for k in keys])
    w = np.exp(logw - logsumexp(logw))
    comp = rng.choice(len(keys), size=n, p=w)
    A = np.array([HM_ALPHA[0] + keys[c][0] for c in comp])
    B = np.array([HM_ALPHA[1] + keys[c][1] for c in comp])
    return rng.beta(A, B)


def harmonic_mean(L: np.ndarray) -> float:
    return float(1.0 / np.mean(1.0 / L))


def umass_coherence(top: list[int], Xbin: np.ndarray) -> float:
    """Def 5 (Mimno et al. 2011, Eq. 1): Σ_{m ≥ 2} Σ_{l < m} log (D(v_m, v_l) + 1)/D(v_l), document frequencies D."""
    Df = Xbin[:, top].sum(0)
    co = Xbin[:, top].T @ Xbin[:, top]
    return float(sum(math.log((co[m, l] + 1) / Df[l]) for m in range(1, len(top)) for l in range(m)))


def npmi_coherence(top: list[int], Xbin: np.ndarray) -> float:
    """Def 5 (after Newman et al. 2010, normalized): the mean over pairs of top words of
    log(P(v, v′)/(P(v)P(v′))) / (−log P(v, v′)), with document-level probabilities; a never-co-occurring pair
    scores −1."""
    n = Xbin.shape[0]
    P = Xbin[:, top].sum(0) / n
    J = (Xbin[:, top].T @ Xbin[:, top]) / n
    vals = []
    for m in range(1, len(top)):
        for l in range(m):
            vals.append(-1.0 if J[m, l] == 0 else
                        (math.log(J[m, l] / (P[m] * P[l])) / -math.log(J[m, l]) if J[m, l] < 1 else 1.0))
    return float(np.mean(vals))


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information with the arithmetic-mean normalization (scikit-learn's default)."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    P = C / C.sum()
    pa, pb = P.sum(1), P.sum(0)
    m = P > 0
    mi = float((P[m] * np.log(P[m] / np.outer(pa, pb)[m])).sum())
    ha, hb = entropy(pa), entropy(pb)
    return 1.0 if ha == hb == 0 else mi / ((ha + hb) / 2)


# --------------------------------------------------------------------------------------------
# 6. Finance case study: seven planted vocabularies, six identifiable topics.
# --------------------------------------------------------------------------------------------
#
# The training text is generate_corpus(300, 0) rebuilt from its own generator (as Topic 6 does), run on for 100
# more documents that continue the same Pitman–Yor restaurants: the held-out text. Every content noun is drawn from
# the document's sector restaurant with probability SECTOR_SHARE = ½ and from the shared one otherwise, so every
# document is Prop 4's ½ / ½ mixture. A word's restaurant is read off the generator (words seated in exactly one).

TOPICS = tuple(fc.SECTORS) + ("common",)
FIN_MORE = 100
FIN_ALPHA, FIN_BETA = 0.1, 0.1
FIN_SWEEPS, FIN_KEEP = 300, 50
FIN_SEEDS = (0, 1, 2)
FIN_RESTARTS = 20


@lru_cache(maxsize=1)
def finance_run() -> dict:
    gen = fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    train = tuple(gen.doc(i, int(t)) for i, t in enumerate(tones))
    more_tones = gen.rng.permutation(np.resize([-1, 0, 1], FIN_MORE))
    held = tuple(gen.doc(300 + i, int(t)) for i, t in enumerate(more_tones))
    rest = {s: frozenset(py.types) for s, py in gen.by_sector.items()} | {"common": frozenset(gen.common.types)}
    homes: dict[str, list[str]] = {}
    for name, types in rest.items():
        for w in types:
            homes.setdefault(w, []).append(name)
    return {"train": train, "held": held, "home": {w: h[0] for w, h in homes.items() if len(h) == 1}}


def sector_of(doc: fc.FinanceDoc) -> int:
    return fc.SECTORS.index(fc._PUBLIC[doc.company][1])


@lru_cache(maxsize=None)
def fin_data(vocab_name: str) -> dict:
    """One vocabulary treatment of the finance text: "restaurant" (content nouns with one planted home), "band"
    (words in 3 to 90 of the 300 training documents), or "all". Held-out tokens outside the vocabulary are dropped."""
    run = finance_run()
    tr = [fc._tokens(d.text) for d in run["train"]]
    he = [fc._tokens(d.text) for d in run["held"]]
    if vocab_name == "restaurant":
        words = sorted({w for t in tr for w in t if w in run["home"]})
    else:
        df: dict[str, int] = {}
        for t in tr:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        words = sorted(w for w, c in df.items() if vocab_name == "all" or 3 <= c <= 0.3 * len(tr))
    idx = {w: i for i, w in enumerate(words)}
    docs = [[idx[w] for w in t if w in idx] for t in tr]
    held = [[idx[w] for w in t if w in idx] for t in he]
    V = len(words)
    out = {"vocab": words, "docs": docs, "held": held, "V": V, "X": counts_of(docs, V),
           "sector": np.array([sector_of(d) for d in run["train"]]),
           "held_sector": np.array([sector_of(d) for d in run["held"]]),
           "doc_type": np.array([("earnings-call", "mdna", "8-k").index(d.doc_type) for d in run["train"]]),
           "tone": np.array([d.tone for d in run["train"]])}
    if vocab_name == "restaurant":
        home = np.array([TOPICS.index(run["home"][w]) for w in words])
        S = len(fc.SECTORS)
        psi = np.zeros((S, V))
        phi7 = np.zeros((len(TOPICS), V))
        theta7 = np.zeros((len(docs), len(TOPICS)))
        for d, doc in enumerate(docs):
            np.add.at(psi[out["sector"][d]], doc, 1)
            np.add.at(phi7, (home[doc], doc), 1)
            theta7[d] = np.bincount(home[doc], minlength=len(TOPICS)) / len(doc)
        out |= {"home": home, "psi": psi / psi.sum(1, keepdims=True), "phi7": phi7 / phi7.sum(1, keepdims=True),
                "theta7": theta7, "tokens7": phi7.sum(1)}
    return out


@lru_cache(maxsize=None)
def fin_gibbs(vocab_name: str, K: int, seed: int, alpha: float = FIN_ALPHA) -> GibbsRun:
    data = fin_data(vocab_name)
    return gibbs(data["docs"], data["V"], K, alpha, FIN_BETA, FIN_SWEEPS, seed=3860 + seed, keep=FIN_KEEP)


@lru_cache(maxsize=None)
def fin_cavi(vocab_name: str, K: int, seed: int) -> CaviRun:
    data = fin_data(vocab_name)
    return cavi(data["X"], K, FIN_ALPHA, FIN_BETA, 100, seed, track=False)


@lru_cache(maxsize=None)
def fin_cavi_warm(seed: int) -> CaviRun:
    """CAVI on the restaurant vocabulary started from a Gibbs run's topic counts (λ = β + n_kw) instead of noise:
    the probe that tells a stalled optimizer from a bound that prefers another answer."""
    data = fin_data("restaurant")
    return cavi(data["X"], 6, FIN_ALPHA, FIN_BETA, 100, seed, lam0=fin_gibbs("restaurant", 6, seed).nkw + FIN_BETA,
                track=False)


@lru_cache(maxsize=None)
def fin_mou(vocab_name: str, K: int) -> dict:
    """Mixture of unigrams by EM: the best of FIN_RESTARTS random starts by training log-likelihood, and the run
    started at the gold sectors (the oracle start)."""
    data = fin_data(vocab_name)
    runs = [mixture_of_unigrams(data["X"], K, FIN_BETA, 300, seed=s) for s in range(FIN_RESTARTS)]
    best = max(runs, key=lambda m: m["loglik"])
    oracle = mixture_of_unigrams(data["X"], K, FIN_BETA, 300, init=np.eye(K)[data["sector"]])
    return {"best": best, "oracle": oracle, "restart_logliks": [m["loglik"] for m in runs]}


def lda_completion(Phi: np.ndarray, data: dict, seed: int) -> float:
    est, scored = halves(data["held"])
    return completion(Phi, fold_in(est, Phi, FIN_ALPHA, 200, 50, seed=3870 + seed), scored)


def mou_completion(model: dict, data: dict) -> float:
    est, scored = halves(data["held"])
    return completion(model["phi"], mou_posterior(model, est), scored)


def cavi_phi(run: CaviRun) -> np.ndarray:
    return run.lam / run.lam.sum(1, keepdims=True)


@lru_cache(maxsize=1)
def fin_anchors() -> dict:
    """Algorithm 1 on the restaurant vocabulary: anchors for K = 6 and K = 7, recovery of the six ψ_s, and how far
    each frequent word's Q̄ row sits (Euclidean) from the hull of the six sector anchors' rows."""
    data = fin_data("restaurant")
    X = data["X"]
    Qbar, p = normalize_rows(cooccurrence(X))
    cand = np.nonzero((X > 0).sum(0) >= 10)[0]
    a6 = fast_anchor_words(Qbar, 6, cand)
    a7 = fast_anchor_words(Qbar, 7, cand)
    Phi6, C = recover_l2(Qbar, p, a6)
    perm = align_topics(Phi6, data["psi"])
    # Each row's Euclidean distance from the anchors' hull: RecoverL2 already solved the simplex-constrained fit.
    resid = np.linalg.norm(Qbar - C @ Qbar[a6], axis=1)
    dist = {int(w): float(resid[w]) for w in cand if w not in a6}
    return {"a6": a6, "a7": a7, "Phi6": Phi6[perm], "l1": np.abs(Phi6[perm] - data["psi"]).sum(1),
            "dist": dist, "cand": cand}


@lru_cache(maxsize=1)
def fin_table() -> dict:
    """The case study's numbers on the restaurant vocabulary: per method, recovery L1 against the six ψ_s and the
    plug-in completion on the held-out documents; MoU's training log-likelihoods."""
    data = fin_data("restaurant")
    rows: dict[str, dict] = {}
    for s in FIN_SEEDS:
        g = fin_gibbs("restaurant", 6, s)
        perm = align_topics(g.phi, data["psi"])
        rows.setdefault("gibbs", {"l1": [], "completion": [], "nmi": []})
        rows["gibbs"]["l1"].append(np.abs(g.phi[perm] - data["psi"]).sum(1).mean())
        rows["gibbs"]["completion"].append(lda_completion(g.phi, data, s))
        rows["gibbs"]["nmi"].append(nmi(data["sector"], g.theta.argmax(1)))
        c = fin_cavi("restaurant", 6, s)
        ph = cavi_phi(c)
        perm = align_topics(ph, data["psi"])
        rows.setdefault("cavi", {"l1": [], "completion": [], "nmi": []})
        rows["cavi"]["l1"].append(np.abs(ph[perm] - data["psi"]).sum(1).mean())
        rows["cavi"]["completion"].append(lda_completion(ph, data, s))
        rows["cavi"]["nmi"].append(nmi(data["sector"], c.gam.argmax(1)))
        rows["cavi"].setdefault("elbo", []).append(elbo_star(data["X"], c.gam, c.lam, FIN_ALPHA, FIN_BETA))
        w = fin_cavi_warm(s)
        ph = cavi_phi(w)
        perm = align_topics(ph, data["psi"])
        rows.setdefault("cavi_warm", {"l1": [], "completion": [], "nmi": [], "elbo": []})
        rows["cavi_warm"]["l1"].append(np.abs(ph[perm] - data["psi"]).sum(1).mean())
        rows["cavi_warm"]["completion"].append(lda_completion(ph, data, s))
        rows["cavi_warm"]["nmi"].append(nmi(data["sector"], w.gam.argmax(1)))
        rows["cavi_warm"]["elbo"].append(elbo_star(data["X"], w.gam, w.lam, FIN_ALPHA, FIN_BETA))
    mou = fin_mou("restaurant", 6)
    for name in ("best", "oracle"):
        m = mou[name]
        perm = align_topics(m["phi"], data["psi"])
        rows[f"mou_{name}"] = {"l1": [np.abs(m["phi"][perm] - data["psi"]).sum(1).mean()],
                               "completion": [mou_completion(m, data)], "nmi": [nmi(data["sector"], m["r"].argmax(1))],
                               "loglik": m["loglik"]}
    return rows


@lru_cache(maxsize=1)
def fin_templates() -> dict:
    """What LDA (K = 6, one seed, 150 sweeps) finds when the vocabulary keeps the template text: NMI of each document's
    dominant topic with its sector, document type, and tone, and the median largest proportion."""
    out = {}
    for name in ("all", "band"):
        data = fin_data(name)
        g = gibbs(data["docs"], data["V"], 6, FIN_ALPHA, FIN_BETA, 150, seed=3865, keep=FIN_KEEP)
        top = g.theta.argmax(1)
        out[name] = {"sector": nmi(data["sector"], top), "doc_type": nmi(data["doc_type"], top),
                     "tone": nmi(data["tone"], top), "max_theta": float(np.median(g.theta.max(1))),
                     "top_words": [[data["vocab"][j] for j in np.argsort(-g.phi[k])[:8]] for k in range(6)]}
    return out


@lru_cache(maxsize=1)
def _topic_model() -> tuple[np.ndarray, tuple[str, ...], dict[str, int]]:
    data = fin_data("restaurant")
    g = fin_gibbs("restaurant", 6, 0)
    perm = align_topics(g.phi, data["psi"])
    return g.phi[perm], tuple(fc.SECTORS), {w: i for i, w in enumerate(data["vocab"])}


def topic_mixtures(docs, sweeps: int = 200, burn: int = 50, seed: int = 0) -> tuple[np.ndarray, tuple[str, ...]]:
    """The capstone's topic stage: θ̂_d over the six identifiable finance topics for any FinanceDocs, by folding
    each document's content nouns into the headline Gibbs fit (restaurant vocabulary, K = 6, seed 0), whose topics
    are aligned to the sectors. Words outside the training vocabulary are dropped; a document with none gets the
    prior mean. Returns (D × 6 proportions, the sector names labeling the columns)."""
    Phi, labels, idx = _topic_model()
    toks = [[idx[w] for w in fc._tokens(d.text) if w in idx] for d in docs]
    return fold_in(toks, Phi, FIN_ALPHA, sweeps, burn, seed=3880 + seed), labels


# --------------------------------------------------------------------------------------------
# Lab data.
# --------------------------------------------------------------------------------------------

# The anchors tab's ternary world: K = 3, so the population rows Q̄_w lie in the anchors' triangle exactly (Thm 1),
# and a sample's rows can be drawn in the same triangle by their least-squares barycentric coordinates.
TRI_K, TRI_V = 3, 60


@lru_cache(maxsize=1)
def tri_world() -> tuple[np.ndarray, np.ndarray]:
    rng = _rng(3890)
    Phi = np.zeros((TRI_K, TRI_V))
    Phi[:, TRI_K:] = rng.dirichlet(np.full(TRI_V - TRI_K, 0.3), TRI_K) * 0.9
    Phi[np.arange(TRI_K), np.arange(TRI_K)] = 0.1
    X, _ = sample_counts(_rng(3891), Phi, np.full(TRI_K, ANCHOR_ALPHA), max(ANCHOR_M), ANCHOR_LEN)
    return Phi, X


def barycentric(Qbar: np.ndarray, anchors: list[int]) -> np.ndarray:
    """Least-squares coordinates c with Σc = 1 of each row on the anchors' rows (unconstrained in sign, so a
    sampled row outside the triangle shows as outside)."""
    B = Qbar[anchors]
    D = B[1:] - B[0]
    coef, *_ = np.linalg.lstsq(D.T, (Qbar - B[0]).T, rcond=None)
    return np.column_stack([1 - coef.sum(0), coef.T])


GV_M, GV_SWEEPS, GV_ITERS = 500, 150, 60
GV_BETAS = (1.0, 0.01)   # the world's own topic prior (Dir(𝟏) off the anchors), and a sparse misspecified one


@lru_cache(maxsize=None)
def gibbs_vs_vi(beta: float) -> dict:
    """The separable V = 400, K = 5 world at M = 500: collapsed Gibbs, CAVI, and anchors on the same documents,
    the two engines at topic prior β."""
    X = anchor_corpus(True)[0][:GV_M]
    gold = anchor_phi(True)
    g = gibbs(tokens_of(X), ANCHOR_V, ANCHOR_K, ANCHOR_ALPHA, beta, GV_SWEEPS, seed=3892, keep=50)
    c = cavi(X, ANCHOR_K, ANCHOR_ALPHA, beta, GV_ITERS, 0)
    out = {"gibbs_trace": g.log_joint, "cavi_trace": c.elbo[1::2], "anchors_l1": anchor_recovery(True, GV_M)["l1"]}
    for name, ph in (("gibbs", g.phi), ("cavi", cavi_phi(c))):
        out[f"{name}_l1"] = np.abs(ph[align_topics(ph, gold)] - gold).sum(1)
    return out


VIZ_JSON = _HERE.parent.parent / "src" / "components" / "viz" / "data" / "topic-model-lab.json"


def viz_payload() -> dict:
    """Every number the lab shows, rounded to the page's precision (3 decimals)."""
    simplex = {"phi": _r(ternary(SIMPLEX_PHI).ravel()), "alphas": list(SIMPLEX_ALPHAS),
               "lengths": list(SIMPLEX_LENGTHS), "cells": {}}
    for a in SIMPLEX_ALPHAS:
        for n in SIMPLEX_LENGTHS:
            w = simplex_world(a, n)
            simplex["cells"][f"{a:g}/{n}"] = {"p": _r(ternary(w["p"]).ravel()), "phat": _r(ternary(w["phat"]).ravel()),
                                              "proj": _r(ternary(w["proj"]).ravel()), "outside": _r(w["outside"])}
    Phi3, X3 = tri_world()
    tri = {}
    for M in ANCHOR_M:
        X = X3[:M]
        Qbar, _ = normalize_rows(cooccurrence(X))
        bc = barycentric(Qbar, list(range(TRI_K)))
        tri[f"{M}"] = _r(ternary(bc).ravel())
    pop = Phi3.T @ dirichlet_second_moment(np.full(TRI_K, ANCHOR_ALPHA)) @ Phi3
    tri["population"] = _r(ternary(barycentric(normalize_rows(pop)[0], list(range(TRI_K)))).ravel())
    curve = {("sep" if sep else "nonsep"): [_r(float(anchor_recovery(sep, M)["l1"].mean())) for M in ANCHOR_M]
             for sep in (True, False)}
    curve["floor"] = _r(float(anchor_recovery(False, None)["l1"].mean()))
    fa = fin_anchors()
    fd = fin_data("restaurant")
    home = fd["home"]
    ft = fin_table()
    words = {}
    for name, ph in (("gibbs", fin_gibbs("restaurant", 6, 0).phi), ("cavi", cavi_phi(fin_cavi("restaurant", 6, 0))),
                     ("mou_oracle", fin_mou("restaurant", 6)["oracle"]["phi"]),
                     ("mou_best", fin_mou("restaurant", 6)["best"]["phi"]), ("anchors", fa["Phi6"])):
        ph = ph[align_topics(ph, fd["psi"])] if name != "anchors" else ph
        words[name] = [[[fd["vocab"][j], int(home[j])] for j in np.argsort(-ph[k])[:10]] for k in range(6)]
    g0 = fin_gibbs("restaurant", 6, 0)
    order = np.lexsort((np.arange(len(fd["docs"])), fd["sector"]))
    theta = g0.theta[:, align_topics(g0.phi, fd["psi"])][order]
    return {"simplex": simplex,
            "anchors": {"tri": tri, "M": list(ANCHOR_M), "curve": curve,
                        "finance": {"a6": [[fd["vocab"][a], int(home[a])] for a in fa["a6"]],
                                    "a7": [[fd["vocab"][a], int(home[a])] for a in fa["a7"]]}},
            "gibbs_vi": {f"{b:g}": {"gibbs": _r(gibbs_vs_vi(b)["gibbs_trace"], 1),
                                    "cavi": _r(gibbs_vs_vi(b)["cavi_trace"], 1),
                                    "l1": {k: _r(gibbs_vs_vi(b)[f"{k}_l1"]) for k in ("gibbs", "cavi", "anchors")},
                                    "mean": {k: _r(float(gibbs_vs_vi(b)[f"{k}_l1"].mean()))
                                             for k in ("gibbs", "cavi", "anchors")}}
                         for b in GV_BETAS},
            "finance": {"topics": list(TOPICS), "words": words,
                        "theta": [_r(row, 2) for row in theta], "sector": fd["sector"][order].tolist(),
                        "table": {k: {m: _r(float(np.mean(v[m]))) for m in ("l1", "completion", "nmi")}
                                  for k, v in ft.items()} | {"anchors": {"l1": _r(float(fa["l1"].mean()))}},
                        "templates": {k: {m: _r(v[m]) for m in ("sector", "doc_type", "tone", "max_theta")}
                                      for k, v in fin_templates().items()}}}


# --------------------------------------------------------------------------------------------
# Tests: every claim the page makes.
# --------------------------------------------------------------------------------------------


def test_prop1_documents_live_in_the_polytope():
    rng = _rng(3900)
    Phi = rng.dirichlet(np.ones(6), 3)
    th = rng.dirichlet(np.ones(3))
    n, reps = 40, 20_000
    X = rng.multinomial(n, th @ Phi, size=reps)
    mean, se = X.mean(0) / n, X.std(0) / n / math.sqrt(reps)
    assert np.all(np.abs(mean - th @ Phi) < 4 * se)
    P = rng.dirichlet(np.ones(3), 50) @ Phi
    assert np.allclose(P.sum(1), 1) and (P >= 0).all()


def test_prop2_fitting_theta_is_a_kl_projection():
    rng = _rng(3901)
    Phi = rng.dirichlet(np.ones(6), 3)
    x = rng.multinomial(30, rng.dirichlet(np.ones(6))).astype(float)
    n, ph = x.sum(), x / x.sum()
    for th in rng.dirichlet(np.ones(3), 20):
        q = th @ Phi
        assert abs(x @ np.log(q) + n * (entropy(ph) + kl(ph, q))) < 1e-10
    best = fit_theta(x, Phi)
    kbest = kl(ph, best @ Phi)
    assert all(kl(ph, th @ Phi) >= kbest - 1e-12 for th in rng.dirichlet(np.ones(3), 2000))
    res = optimize.minimize(lambda t: kl(ph, np.abs(t) / np.abs(t).sum() @ Phi), np.full(3, 1 / 3),
                            method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20_000})
    assert abs(res.fun - kbest) < 1e-9


def test_simplex_tab_noise_shrinks_with_length():
    for a in SIMPLEX_ALPHAS:
        out = [simplex_world(a, n)["outside"] for n in SIMPLEX_LENGTHS]
        assert out[0] > out[-1], (a, out)
        w = simplex_world(a, SIMPLEX_LENGTHS[0])
        assert np.allclose(w["p"], w["theta"] @ SIMPLEX_PHI)


def test_rem1_lsa_coordinates_are_signed():
    X = fin_data("restaurant")["X"]
    P = X / X.sum(1, keepdims=True)
    U, sv, Vt = np.linalg.svd(P, full_matrices=False)
    approx = (U[:, :6] * sv[:6]) @ Vt[:6]
    assert (Vt[1] < 0).any() and (Vt[1] > 0).any()
    assert _r((approx < -1e-12).mean()) == 0.283


def test_prop3_relabeling_and_stretching():
    rng = _rng(3902)
    a = np.full(TOY_K, TOY_ALPHA)
    for _ in range(10):
        z = [[int(k) for k in rng.integers(TOY_K, size=len(d))] for d in TOY_DOCS]
        zp = [[1 - k for k in zd] for zd in z]
        assert abs(log_joint(*count_tables(TOY_DOCS, z, TOY_K, TOY_V), a, TOY_BETA)
                   - log_joint(*count_tables(TOY_DOCS, zp, TOY_K, TOY_V), a, TOY_BETA)) < 1e-12
    c = prop3_construction()
    assert np.abs(c["P"] - c["P2"]).max() < 1e-14
    assert (c["Theta2"] >= 0).all() and np.allclose(c["Theta2"].sum(1), 1)
    assert (c["Phi2"] >= 0).all() and np.allclose(c["Phi2"].sum(1), 1)
    assert min(np.abs(c["Phi2"] - c["Phi"]).max(), np.abs(c["Phi2"] - c["Phi"][::-1]).max()) > 0.04


def test_prop4_shared_topic_is_unidentifiable():
    w = prop4_world()
    full = np.vstack([w["Phis"], w["Phi0"]])
    assert np.abs(w["thetas"] @ full - w["Psi"]).max() < 1e-15
    assert np.linalg.matrix_rank(w["R"], tol=1e-12) == w["S"]
    for c2 in (0.0, 0.25):
        alt = (w["Psi"] - c2 * w["Phi0"]) / (1 - c2)
        assert (alt >= 0).all()
        assert np.abs((1 - c2) * alt + c2 * w["Phi0"] - w["Psi"]).max() < 1e-15


def test_prop5_cooccurrence_estimator_is_unbiased():
    rng = _rng(3903)
    V, n = 3, 4
    Phi = rng.dirichlet(np.ones(V), 2)
    th = rng.dirichlet(np.ones(2))
    p = th @ Phi
    E = np.zeros((V, V))
    for seq in itertools.product(range(V), repeat=n):
        x = np.bincount(seq, minlength=V).astype(float)
        E += np.prod(p[list(seq)]) * (np.outer(x, x) - np.diag(x)) / (n * (n - 1))
    assert np.abs(E - Phi.T @ np.outer(th, th) @ Phi).max() < 1e-15


def test_thm1_anchor_rows_are_the_vertices():
    W = thm1_world()
    A = W["anchors"]
    assert np.abs(W["C"] @ W["Qbar"][A] - W["Qbar"]).max() < 1e-14
    assert np.abs(W["C"].sum(1) - 1).max() < 1e-14 and (W["C"] >= 0).all()
    assert all(hull_distance(W["Qbar"][a], W["Qbar"][[b for b in A if b != a]]) > 0.1 for a in A)
    assert all(hull_distance(W["Qbar"][w], W["Qbar"][A]) < 1e-9 for w in range(W["K"], W["V"]))
    found = fast_anchor_words(W["Qbar"], W["K"], np.arange(W["V"]))
    assert sorted(found) == A
    Ph, _ = recover_l2(W["Qbar"], W["p"], found, iters=20_000)
    assert np.abs(Ph[np.argsort(found)] - W["Phi"]).max() < 1e-10


def test_thm1_needs_nonsingular_r():
    W = thm1_world(singular=True)
    assert np.linalg.matrix_rank(W["R"], tol=1e-12) == 3
    assert hull_distance(W["Qbar"][3], W["Qbar"][[0, 1, 2]]) < 1e-9
    assert 3 not in fast_anchor_words(W["Qbar"], W["K"], np.arange(W["V"]))


def test_anchor_recovery_improves_with_documents():
    sep = [float(anchor_recovery(True, M)["l1"].mean()) for M in ANCHOR_M]
    non = [float(anchor_recovery(False, M)["l1"].mean()) for M in ANCHOR_M]
    assert all(a > b for a, b in zip(sep, sep[1:])), sep
    assert non[-1] > 2 * sep[-1], (non, sep)
    assert all(sorted(anchor_recovery(True, M)["anchors"]) == list(range(ANCHOR_K)) for M in ANCHOR_M[2:])
    # With population Q, the separable world is recovered exactly; the twin keeps a floor that no M removes.
    assert float(anchor_recovery(True, None)["l1"].max()) < 1e-9
    floor = float(anchor_recovery(False, None)["l1"].mean())
    assert floor > 0.1 and non[-1] > floor, (floor, non)


def test_lemma1_dirichlet_multinomial_marginal():
    from scipy.stats import beta as beta_dist
    for a, b, n1, n2 in ((0.5, 0.5, 3, 1), (2.0, 0.3, 0, 4), (1.0, 1.0, 2, 2)):
        val, _ = integrate.quad(lambda t: t ** n1 * (1 - t) ** n2 * beta_dist.pdf(t, a, b), 0, 1, epsabs=1e-14)
        assert abs(math.log(val) - (log_beta(np.array([a + n1, b + n2])) - log_beta(np.array([a, b])))) < 1e-9


def test_thm2_full_conditional():
    rng = _rng(3904)
    docs = [[0, 3, 1, 1, 4], [2, 2, 0], [4, 1, 3, 3]]
    K, V, beta = 3, 5, 0.3
    a = np.array([0.2, 0.7, 1.1])
    for _ in range(25):
        z = [[int(k) for k in rng.integers(K, size=len(d))] for d in docs]
        d = int(rng.integers(len(docs)))
        i = int(rng.integers(len(docs[d])))
        lj = []
        for k in range(K):
            z2 = [list(x) for x in z]
            z2[d][i] = k
            lj.append(log_joint(*count_tables(docs, z2, K, V), a, beta))
        assert np.abs(full_conditional(docs, z, d, i, K, V, a, beta) - np.exp(lj - logsumexp(lj))).max() < 1e-12


def test_cor1_the_posterior_is_stationary():
    states, post, _ = toy_posterior()
    n = sum(map(len, TOY_DOCS))
    P = np.eye(len(states))
    for site in range(n):
        Pi = toy_kernel(site)
        assert np.abs(post[:, None] * Pi - (post[:, None] * Pi).T).max() < 1e-15   # detailed balance per site
        P = P @ Pi
    assert np.abs(post @ P - post).max() < 1e-12
    assert (P > 0).all()   # one sweep reaches every state: irreducible and aperiodic
    lam, vec = np.linalg.eig(P.T)
    st = np.real(vec[:, np.argmin(np.abs(lam - 1))])
    assert np.abs(st / st.sum() - post).max() < 1e-12


def test_algorithm2_sampler_visits_the_posterior():
    states, post, _ = toy_posterior()
    index = {tuple(itertools.chain(*z)): j for j, z in enumerate(states)}
    freq = np.zeros(len(states))

    def visit(Z):
        freq[index[tuple(itertools.chain(*Z))]] += 1

    gibbs(TOY_DOCS, TOY_V, TOY_K, TOY_ALPHA, TOY_BETA, 40_000, seed=3905, visit=visit)
    # Per-token marginals P(z_i = 0 | w): 128 states at 40k correlated sweeps leave a Monte Carlo floor of ≈ 0.02 in
    # total variation, so the test reads the seven one-token marginals instead.
    flat = np.array([list(itertools.chain(*z)) for z in states])
    f = freq / freq.sum()
    assert np.abs((f - post) @ (flat == 0)).max() < 0.01


def test_prop6_point_estimates_are_posterior_means():
    rng = _rng(3906)
    counts = np.array([5.0, 0.0, 2.0, 1.0])
    beta = 0.3
    draws = rng.dirichlet(counts + beta, 200_000)
    se = draws.std(0) / math.sqrt(len(draws))
    assert np.all(np.abs(draws.mean(0) - (counts + beta) / (counts.sum() + len(counts) * beta)) < 4 * se)


def test_lemma2_dirichlet_log_moment():
    from scipy.stats import beta as beta_dist
    rng = _rng(3907)
    for g in (np.array([0.3, 1.2, 2.5]), np.array([4.0, 0.7])):
        h = 1e-6
        fd = np.array([(log_beta(g + h * e) - log_beta(g - h * e)) / (2 * h) for e in np.eye(len(g))])
        assert np.abs(fd - dirichlet_expectation(g)).max() < 1e-7
    a, b = 0.7, 1.9
    val, _ = integrate.quad(lambda t: math.log(t) * beta_dist.pdf(t, a, b), 0, 1, epsabs=1e-13, limit=200)
    assert abs(val - dirichlet_expectation(np.array([a, b]))[0]) < 1e-9
    draws = np.log(rng.dirichlet([0.3, 1.2, 2.5], 400_000))
    assert np.all(np.abs(draws.mean(0) - dirichlet_expectation(np.array([0.3, 1.2, 2.5])))
                  < 4 * draws.std(0) / math.sqrt(len(draws)))


def test_cor2_elbo_never_decreases():
    X, _ = sample_counts(_rng(3908), anchor_phi(True), np.full(ANCHOR_K, ANCHOR_ALPHA), 100, ANCHOR_LEN)
    e = np.array(cavi(X, ANCHOR_K, ANCHOR_ALPHA, 0.1, 15, 0).elbo)
    assert (np.diff(e) >= -1e-9 * np.abs(e[1:])).all()


def test_thm3_bound_matches_scikit_learn():
    from sklearn.decomposition import LatentDirichletAllocation
    X, _ = sample_counts(_rng(3909), anchor_phi(True), np.full(ANCHOR_K, ANCHOR_ALPHA), 120, ANCHOR_LEN)
    sk = LatentDirichletAllocation(ANCHOR_K, doc_topic_prior=0.3, topic_word_prior=0.1, learning_method="batch",
                                   max_iter=20, random_state=0).fit(X)
    gam = sk.transform(X, normalize=False)
    ours = elbo_star(X, gam, sk.components_, 0.3, 0.1)
    theirs = float(sk._approx_bound(X, gam, False))
    assert abs(ours - theirs) < 1e-8 * abs(theirs), (ours, theirs)


def test_thm4_harmonic_mean_identity_and_bias():
    m = toy_marginals(TOY_DOCS, TOY_V, TOY_K, TOY_ALPHA, TOY_BETA)
    assert abs(math.log(m["post"] @ np.exp(-m["loglik"])) + m["log_pw"]) < 1e-12
    assert abs(math.log(m["prior"] @ np.exp(m["loglik"])) - m["log_pw"]) < 1e-12     # Prop 7: unbiased
    rng = _rng(3910)
    L = np.exp(m["loglik"])
    S, reps = 20, 4000
    hm = np.array([harmonic_mean(L[rng.choice(len(L), S, p=m["post"])]) for _ in range(reps)])
    pw = math.exp(m["log_pw"])
    assert hm.mean() > pw and np.median(hm) > pw


def test_prop7_variance_table():
    # The page's table: one-document corpora of n tokens (words 0, 1, 2, 0, …), K = 2, α = β = ½.
    rows = {}
    for n in (4, 8, 12):
        m = toy_marginals([[i % 3 for i in range(n)]], 3, 2, 0.5, 0.5)
        rows[n] = (_r(hm_relative_variance(m)), _r(is_relative_variance(m)))
    assert rows == {4: (0.25, 0.387), 8: (0.613, 3.033), 12: (1.015, 17.424)}, rows
    assert all(h < i for h, i in rows.values())


def test_thm4c_anchor_tokens_make_the_variance_infinite():
    # ∫ Beta(t; ½, ½)/p(w_d | t) dt over [ε, 1] grows like ε^{−1/2}: one anchor token, c = 1 ≥ α₀ = ½.
    from scipy.stats import beta as beta_dist
    def tail(eps):
        edges = np.geomspace(eps, 1, int(round(-math.log10(eps))) + 1)
        return sum(integrate.quad(lambda t: beta_dist.pdf(t, *HM_ALPHA) / hm_doc_likelihood(t)[0], a, b,
                                  epsabs=0, epsrel=1e-10, limit=200)[0] for a, b in zip(edges, edges[1:]))
    ratios = [tail(e / 100) / tail(e) for e in (1e-4, 1e-6)]
    assert all(8 < r < 11 for r in ratios), ratios


def test_coherence_definitions():
    Xb = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    assert abs(umass_coherence([0, 1], Xb) - math.log(4 / 3)) < 1e-15
    assert npmi_coherence([0, 1], Xb) > 0.99 and npmi_coherence([0, 2], Xb) < 0


def test_finance_text_is_topic6s_and_continues_it():
    run = finance_run()
    assert [d.text for d in run["train"]] == [d.text for d in ds._generator_run()[0]]
    held = tuple(s for d in run["held"] for s in ng.split_sentences(d.text))
    assert held == ng.continuation_run(FIN_MORE)["continuation"]
    planted = ds.planted_labels()
    both = [w for w in planted if w in run["home"]]
    assert len(both) > 0.95 * len(planted) and all(planted[w] == run["home"][w] for w in both)


def test_seven_restaurants_every_document_half_and_half():
    fd = fin_data("restaurant")
    assert (fd["V"], int(fd["X"].sum()), sum(map(len, fd["held"]))) == (2908, 13382, 3735)
    assert np.bincount(fd["sector"]).tolist() == [142, 52, 40, 20, 27, 19]
    th = fd["theta7"]
    own = th[np.arange(len(th)), fd["sector"]]
    assert 0.45 < np.median(th[:, 6]) < 0.55 and 0.45 < np.median(own) < 0.55
    assert (1 - th[:, 6] - own).mean() < 0.03


def test_finance_anchors_one_per_sector_none_shared():
    fa = fin_anchors()
    home = fin_data("restaurant")["home"]
    assert sorted(int(home[a]) for a in fa["a6"]) == list(range(6))
    assert all(int(home[a]) != 6 for a in fa["a7"])


def test_mou_oracle_beats_lda_but_em_cannot_find_it():
    t = fin_table()
    assert t["mou_oracle"]["completion"][0] > max(t["gibbs"]["completion"])
    mou = fin_mou("restaurant", 6)
    assert mou["oracle"]["loglik"] > max(mou["restart_logliks"])


def test_cavi_stalls_where_its_own_bound_is_lower():
    t = fin_table()
    assert all(a + 1000 < b for a, b in zip(t["cavi"]["elbo"], t["cavi_warm"]["elbo"])), (t["cavi"]["elbo"],
                                                                                         t["cavi_warm"]["elbo"])
    assert max(t["cavi"]["nmi"]) < 0.5 < min(t["cavi_warm"]["nmi"]) and min(t["gibbs"]["nmi"]) > 0.9


@lru_cache(maxsize=1)
def fin_sklearn() -> dict:
    """scikit-learn's batch VB on the restaurant vocabulary (its own initialization and schedule, 100 iterations):
    the stall is not an artifact of this module's CAVI."""
    from sklearn.decomposition import LatentDirichletAllocation
    data = fin_data("restaurant")
    sk = LatentDirichletAllocation(6, doc_topic_prior=FIN_ALPHA, topic_word_prior=FIN_BETA, learning_method="batch",
                                   max_iter=100, random_state=0).fit(data["X"])
    gam = sk.transform(data["X"], normalize=False)
    return {"nmi": nmi(data["sector"], gam.argmax(1)), "elbo": elbo_star(data["X"], gam, sk.components_, FIN_ALPHA,
                                                                          FIN_BETA)}


def test_scikit_learn_also_stops_short():
    sk, warm = fin_sklearn(), fin_table()["cavi_warm"]
    assert sk["elbo"] < min(warm["elbo"]) and sk["nmi"] < min(warm["nmi"]), sk


def test_the_prior_ranks_the_engines():
    # Asuncion et al.'s point on one world: at the world's own β Gibbs recovers the topics best; at a sparse,
    # misspecified β the order reverses.
    right, wrong = (gibbs_vs_vi(b) for b in GV_BETAS)
    assert right["gibbs_l1"].mean() < right["cavi_l1"].mean()
    assert wrong["gibbs_l1"].mean() > wrong["cavi_l1"].mean()
    assert right["gibbs_l1"].mean() < wrong["gibbs_l1"].mean()


def test_templates_hide_tone_and_document_type():
    for name, v in fin_templates().items():
        assert v["doc_type"] < 0.05 and v["tone"] < 0.05, (name, v)


def test_topic_mixtures_reads_held_out_sectors():
    run = finance_run()
    th, labels = topic_mixtures(run["held"])
    assert labels == tuple(fc.SECTORS) and np.allclose(th.sum(1), 1)
    acc = float(np.mean(th.argmax(1) == fin_data("restaurant")["held_sector"]))
    assert acc > 0.8, acc


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def summary() -> dict:
    """The numbers the page quotes, at 6 decimals."""
    out: dict = {}
    fd = fin_data("restaurant")
    out["restaurant V"] = fd["V"]
    out["restaurant tokens"] = int(fd["X"].sum())
    out["median common share"] = float(np.median(fd["theta7"][:, 6]))
    for k, v in fin_table().items():
        for m in ("l1", "completion", "nmi", "elbo"):
            if m in v:
                out[f"{k} {m} mean"] = float(np.mean(v[m]))
                out[f"{k} {m} range"] = f"{min(v[m]):.6f} .. {max(v[m]):.6f}"
    out["sklearn nmi"] = fin_sklearn()["nmi"]
    out["sklearn elbo"] = fin_sklearn()["elbo"]
    mou = fin_mou("restaurant", 6)
    out["mou oracle loglik"] = mou["oracle"]["loglik"]
    out["mou best restart loglik"] = max(mou["restart_logliks"])
    for k, v in fin_templates().items():
        for m in ("sector", "doc_type", "tone", "max_theta"):
            out[f"{k} {m}"] = v[m]
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
