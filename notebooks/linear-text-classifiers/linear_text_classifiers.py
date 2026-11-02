"""Linear text classifiers, generative vs. discriminative — the reference implementation for formalNLP's seventeenth
topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. One hypothesis class. Multinomial and Bernoulli naive Bayes (NB) posteriors are softmax-linear in the document's
   count vector; on documents of one length the NB family and multinomial logistic regression (LR) coincide, and on
   documents of varying length LR's family is strictly larger (Prop 1).
2. Two estimating equations. Penalized softmax LR is convex, its Hessian is Σ_i (diag p_i − p_i p_iᵀ) ⊗ x̃_i x̃_iᵀ plus
   the penalty, and its optimum matches every class × word co-count up to nλW (and every class total exactly). NB
   matches class-conditional word totals instead, which LR's equations need not satisfy (Thm 1).
3. How fast. Ng and Jordan's generative bound: with m = O(ε⁻² log(V/δ)) documents NB's Bernoulli parameters are all
   within ε (Lemma 1), and its error is within G(τ) of its asymptotic error (Thm 2). For a multinomial the parameters
   are at most 1/V, so uniform relative accuracy costs Θ(1/θ_min) ≥ V tokens, and V^s log V under Zipf (Thm 3).
   Burstiness keeps LR well specified while NB's population weights stop matching the Bayes rule when bursts fall
   unevenly on the informative words (Prop 2): the generative–discriminative crossover, measured in worlds whose Bayes
   error is known, and which noise-only bursts produce with equal asymptotes.
4. Separation. With V ≫ n the training documents are linearly separable, the logistic MLE does not exist, and the
   L2 path's direction converges to the max-margin separator (Thm 4); gradient descent's does too (Soudry et al.).
5. Priors on features. NB-weighted LR is LR with a per-feature ridge λ Σ v_j²/r_j², and add-α smoothing bounds a
   word's log-count ratio by log((c_j + α)/α) plus a constant (Prop 3).
6. Finance case study. Tone is an XOR of aspect valence and direction ("costs rose" is bad news): no unigram-linear
   classifier represents it, and bigrams do (Prop 4). NB, LR, and NB-weighted LR on unigrams and bigrams, against the
   full-text Bayes accuracy, with label noise and calibration. Six splits, vocabulary cuts, redrawn words, exact
   readers, and (behind --slow) training sets of 15,000 and 50,000 documents test each explanation the page offers.

Cross-checks: scikit-learn's MultinomialNB, BernoulliNB, and LogisticRegression; finite differences for every
gradient and Hessian; scipy.optimize.linprog for (non)separability; exact enumeration for the Bayes ceiling.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python notebooks/linear-text-classifiers/linear_text_classifiers.py
Add --slow for the checks that fit on 15,000 and 50,000 documents.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import optimize, stats
from scipy.special import expit, logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "zipf-heaps-vocabulary-statistics"))
import zipf_heaps_vocabulary_statistics as zh  # noqa: E402 — a module object: its test_* stay out of this harness

fc = zh.fc

# ONE stream per random-instance claim and per sampled grid, each from default_rng([SEED, k]); k ≥ 2000 keeps this
# module's streams apart from Topics 1–16.
SEED = zh.SEED


def _r(x, nd: int = 3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


# --------------------------------------------------------------------------------------------
# 1. Bag of words, and naive Bayes as a softmax-linear classifier (Def 1, Prop 1).
# --------------------------------------------------------------------------------------------


def features(tokens: list[str], bigrams: bool) -> list[str]:
    """A document's feature occurrences: its tokens, and with ``bigrams`` also each adjacent pair ("costs rose")."""
    return list(tokens) + ([f"{a} {b}" for a, b in zip(tokens, tokens[1:])] if bigrams else [])


def featurize(tok_docs, vocab: dict[str, int] | None = None, *, bigrams: bool = False,
              min_df: int = 1) -> tuple[sp.csr_matrix, dict[str, int]]:
    """Def 1: the count matrix X (documents × features). With no ``vocab``, it is built from these documents, keeping
    features in at least ``min_df`` of them, in sorted order; with one, features outside it are dropped."""
    seqs = [features(t, bigrams) for t in tok_docs]
    if vocab is None:
        df: dict[str, int] = {}
        for f in seqs:
            for w in set(f):
                df[w] = df.get(w, 0) + 1
        vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= min_df))}
    rows, cols = [], []
    for i, f in enumerate(seqs):
        for w in f:
            j = vocab.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(seqs), len(vocab)))
    X.sum_duplicates()
    return X, vocab


def binarize(X: sp.csr_matrix) -> sp.csr_matrix:
    B = X.copy()
    B.data = np.ones_like(B.data)
    return B


def nb_fit(X, y: np.ndarray, K: int, alpha: float = 1.0, *, bernoulli: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Prop 1's (W, b) for naive Bayes with add-α smoothing, so that log P(y = k | x) = softmax(Wx + b)_k.

    Multinomial: θ_kj = (c_kj + α)/(Σ_j c_kj + αV), W_kj = log θ_kj, b_k = log π̂_k.
    Bernoulli (x binarized): θ_kj = (d_kj + α)/(n_k + 2α), W_kj = logit θ_kj, b_k = log π̂_k + Σ_j log(1 − θ_kj).
    """
    Y = np.eye(K)[y]
    nk = Y.sum(0)
    prior = np.log(nk / nk.sum())
    if bernoulli:
        d = np.asarray((binarize(X).T @ Y).T)
        theta = (d + alpha) / (nk[:, None] + 2 * alpha)
        return np.log(theta) - np.log1p(-theta), prior + np.log1p(-theta).sum(1)
    c = np.asarray((X.T @ Y).T)
    theta = (c + alpha) / (c.sum(1, keepdims=True) + alpha * X.shape[1])
    return np.log(theta), prior


def log_posterior(X, W: np.ndarray, b: np.ndarray, *, bernoulli: bool = False) -> np.ndarray:
    """log softmax(Wx + b) for every row of X (binarized first for a Bernoulli model)."""
    Z = np.asarray((binarize(X) if bernoulli else X) @ W.T) + b
    return Z - logsumexp(Z, axis=1, keepdims=True)


def nb_log_posterior_direct(x: np.ndarray, theta: np.ndarray, prior: np.ndarray, *, bernoulli: bool = False):
    """The generative computation, one document at a time: log π_k + log P(x | y = k), normalized. The multinomial
    coefficient is the same for every class and is kept, so this is the literal Bayes rule, not Prop 1's form."""
    if bernoulli:
        on = x > 0
        ll = np.array([np.log(t[on]).sum() + np.log1p(-t[~on]).sum() for t in theta])
    else:
        ll = np.array([stats.multinomial.logpmf(x, n=int(x.sum()), p=t) for t in theta])
    z = prior + ll
    return z - logsumexp(z)


def nb_from_logits(W: np.ndarray, b: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Prop 1(b): the multinomial NB (θ, π) whose posterior equals softmax(Wx + b) on every document of length L.
    θ_k = softmax(W_k), and the row normalizers L·log Z_k are a bias on length-L documents."""
    logZ = logsumexp(W, axis=1)
    theta = np.exp(W - logZ[:, None])
    logpi = b + L * logZ
    return theta, logpi - logsumexp(logpi)


# --------------------------------------------------------------------------------------------
# 2. Softmax logistic regression: convexity and the estimating equations (Thm 1).
# --------------------------------------------------------------------------------------------


def lr_objective(theta: np.ndarray, X, y: np.ndarray, K: int, lam, *, with_grad: bool = True):
    """Mean cross-entropy of softmax(Wx + b) plus (1/2)Σ_kj λ_j W_kj² (λ a scalar or one value per feature); the
    intercepts b are not penalized. theta = [vec W (K×V, row-major), b]."""
    n, V = X.shape
    W, b = theta[:K * V].reshape(K, V), theta[K * V:]
    Z = np.asarray(X @ W.T) + b
    lse = logsumexp(Z, axis=1)
    L = float(lse.mean() - Z[np.arange(n), y].mean() + 0.5 * np.sum(lam * W * W))
    if not with_grad:
        return L
    G = np.exp(Z - lse[:, None])
    G[np.arange(n), y] -= 1.0
    G /= n
    return L, np.concatenate([(np.asarray((X.T @ G).T) + lam * W).ravel(), G.sum(0)])


def lr_hessian(W: np.ndarray, b: np.ndarray, X: np.ndarray, lam) -> np.ndarray:
    """Thm 1's Hessian of lr_objective in theta's order: (1/n)Σ_i (diag p_i − p_i p_iᵀ) ⊗ x̃_i x̃_iᵀ, x̃ = (x, 1),
    regrouped so W's K×V block comes first, plus the penalty on W."""
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    n, V = X.shape
    K = W.shape[0]
    Z = X @ W.T + b
    P = np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
    Xt = np.hstack([X, np.ones((n, 1))])
    H = np.zeros((K * (V + 1), K * (V + 1)))  # blocks in (k, feature-or-intercept) order
    for i in range(n):
        H += np.kron(np.diag(P[i]) - np.outer(P[i], P[i]), np.outer(Xt[i], Xt[i]))
    H /= n
    pen = np.zeros((K, V + 1))
    pen[:, :V] = lam
    H += np.diag(pen.ravel())
    perm = np.concatenate([(np.arange(K)[:, None] * (V + 1) + np.arange(V)[None, :]).ravel(),
                           np.arange(K) * (V + 1) + V])  # → [vec W, b]
    return H[np.ix_(perm, perm)]


def lr_hessp(theta: np.ndarray, v: np.ndarray, X, y: np.ndarray, K: int, lam) -> np.ndarray:
    """Thm 1's Hessian applied to a direction v = [vec U, c], without forming it. With A = XUᵀ + c (each document's
    logit change) and P = softmax(XWᵀ + b), the change in the softmax gradient is M = (P ⊙ A − P ⊙ rowsum(P ⊙ A))/n,
    so H v = [vec(MᵀX) + λU, column sums of M]."""
    n, V = X.shape
    W, b = theta[:K * V].reshape(K, V), theta[K * V:]
    Z = np.asarray(X @ W.T) + b
    P = np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
    U, c = v[:K * V].reshape(K, V), v[K * V:]
    A = np.asarray(X @ U.T) + c
    M = P * A
    M -= P * M.sum(1, keepdims=True)
    M /= n
    return np.concatenate([(np.asarray((X.T @ M).T) + lam * U).ravel(), M.sum(0)])


def lr_fit(X, y: np.ndarray, K: int, lam, x0: np.ndarray | None = None, *,
           gtol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Penalized softmax LR by trust-region Newton–CG (the analytic gradient and lr_hessp). Returns (W, b, theta) —
    theta warm-starts the next fit on a path."""
    n, V = X.shape
    r = optimize.minimize(lr_objective, np.zeros(K * V + K) if x0 is None else x0, args=(X, y, K, lam), jac=True,
                          hessp=lr_hessp, method="trust-ncg", options={"maxiter": 500, "gtol": gtol})
    return r.x[:K * V].reshape(K, V), r.x[K * V:], r.x


def estimating_residuals(X, y: np.ndarray, W: np.ndarray, b: np.ndarray, lam) -> tuple[np.ndarray, np.ndarray]:
    """Thm 1's two equations at (W, b): R_kj = Σ_i (y_ik − p_ik) x_ij − nλ_j W_kj and t_k = n_k − Σ_i p_ik. Both are
    zero at LR's optimum."""
    n, K = X.shape[0], W.shape[0]
    Y = np.eye(K)[y]
    E = Y - np.exp(log_posterior(X, W, b))
    return np.asarray((X.T @ E).T) - n * lam * W, E.sum(0)


def blr_fit(Z, lam: float, x0: np.ndarray | None = None, *, gtol: float = 1e-12) -> np.ndarray:
    """Binary homogeneous logistic regression on signed rows z_i = y_i x_i: argmin (1/n)Σ log(1 + e^{−z_i·w})
    + (λ/2)‖w‖², by trust-region Newton–CG. No intercept, as in Thm 4 and Soudry et al.'s setting."""
    n, V = Z.shape

    def f(w):
        m = Z @ w
        return float(np.logaddexp(0, -m).mean() + 0.5 * lam * w @ w), -(Z.T @ expit(-m)) / n + lam * w

    def hp(w, v):
        m = Z @ w
        return Z.T @ (expit(m) * expit(-m) * (Z @ v)) / n + lam * v

    r = optimize.minimize(f, np.zeros(V) if x0 is None else x0, jac=True, hessp=hp, method="trust-ncg",
                          options={"maxiter": 1000, "gtol": gtol})
    return r.x


def blr_intercept_fit(X, y01: np.ndarray, lam, x0: np.ndarray | None = None) -> np.ndarray:
    """Binary LR with an unpenalized intercept, y ∈ {0, 1}, by trust-region Newton–CG: returns theta = [w, b]. The
    penalty is (1/2)Σ_j λ_j w_j² (λ a scalar or one value per feature). The learning-curve and NB-weighted
    workhorse."""
    n, V = X.shape
    s = 2.0 * y01 - 1.0

    def f(t):
        m = s * (X @ t[:V] + t[V])
        g = -s * expit(-m) / n
        return float(np.logaddexp(0, -m).mean() + 0.5 * np.sum(lam * t[:V] ** 2)), np.concatenate(
            [X.T @ g + lam * t[:V], [g.sum()]])

    def hp(t, v):
        m = s * (X @ t[:V] + t[V])
        u = expit(m) * expit(-m) * (X @ v[:V] + v[V]) / n
        return np.concatenate([X.T @ u + lam * v[:V], [u.sum()]])

    r = optimize.minimize(f, np.zeros(V + 1) if x0 is None else x0, jac=True, hessp=hp, method="trust-ncg",
                          options={"maxiter": 500, "gtol": 1e-10})
    return r.x


# --------------------------------------------------------------------------------------------
# 3. Sample complexity: Ng–Jordan's generative bound (Lemma 1, Thm 2), the multinomial and Zipf version (Thm 3),
#    burstiness (Prop 2), and the learning curves.
# --------------------------------------------------------------------------------------------


def lemma1_m(eps: float, delta: float, V: int) -> int:
    """Lemma 1(a)'s per-class sample size m_0 = ⌈ln(4V/δ)/(2ε²)⌉: given at least m_0 training documents in each class,
    all 2V empirical Bernoulli parameters are within ε of the truth w.p. ≥ 1 − δ (Hoeffding, two-sided, and a union
    bound over 2V events)."""
    return math.ceil(math.log(4 * V / delta) / (2 * eps * eps))


def thm2_tau(eps: float, rho0: float, V: int) -> float:
    """Thm 2's margin width τ: if every parameter (and the prior) is within ε ≤ ρ0/2 of a truth in [ρ0, 1 − ρ0], each
    of the 2V + 2 log terms of the discriminant moves by at most 2ε/ρ0, so |l̂(x) − l_∞(x)| ≤ 4(V + 1)ε/ρ0 =: τV."""
    return 4 * (V + 1) * eps / (rho0 * V)


def thm3_tokens(eps: float, delta: float, theta_min: float, V: int) -> float:
    """Thm 3(a)'s token count: N ≥ 3 log(2V/δ)/(ε² θ_min) puts every θ̂_j within a factor 1 ± ε of θ_j w.p. ≥ 1 − δ
    (multiplicative Chernoff + a union bound)."""
    return 3 * math.log(2 * V / delta) / (eps * eps * theta_min)


def zipf_theta_min(V: int, s: float) -> float:
    """θ_min = V^{−s}/H_{V,s} for Zipf(s) on V words — Topic 1's zipf_pmf, last entry."""
    return float(zh.zipf_pmf(s, V)[-1])


@dataclass(frozen=True)
class WorldConfig:
    noise_words: int  # V − |S| uninformative words, the most frequent ones, sharing one distribution
    bursty: bool  # repeat each occurrence of half the words BURST times (Prop 2)


BURST = 4  # Prop 2's repeat factor r_j for the bursty half of the vocabulary
WORLD_NOISE = (30, 3000)


@lru_cache(maxsize=None)
def corpus_binary() -> tuple[sp.csr_matrix, np.ndarray, dict[str, int]]:
    """The polar documents of the finance training corpus, generate_corpus(1500, 0) (tone −1 → 0, +1 → 1), as
    unigram counts: the source of the simulated worlds' informative words and noise frequencies, and of the funnel."""
    T, y = fin_split()[:2]
    keep = np.flatnonzero(y != 1)
    X, voc = featurize([T[i] for i in keep])
    return X, (y[keep] == 2).astype(int), voc


def informative_words() -> tuple[str, ...]:
    """S: the stand-in lexicon, the direction words, and "not" — the only words whose class-conditional rates the
    simulated worlds let differ."""
    lex = set().union(*fc.STANDIN_LEXICON.values()) | set(fc._UP) | set(fc._DOWN) | {"not"}
    return tuple(sorted(w for w in lex if w in corpus_binary()[2]))


@dataclass(frozen=True)
class World:
    theta: np.ndarray  # 2 × V class-conditional word distributions; the first |S| columns are S
    rep: np.ndarray  # V repeat factors (all 1 unless bursty)
    lengths: np.ndarray  # document lengths, drawn from the corpus's polar documents
    n_informative: int


@lru_cache(maxsize=None)
def world(cfg: WorldConfig) -> World:
    """A world where multinomial NB is the true model of the underlying draws: S's two class-conditional
    distributions from the corpus (add-1/2 smoothed), holding S's corpus share m_S of every document, and the rest
    spread over ``noise_words`` uninformative words in proportion to their corpus frequencies. The Bayes error does
    not depend on noise_words. In the bursty world, a fixed random half of the words (stream 2001) occur in runs of
    BURST copies."""
    X, y, voc = corpus_binary()
    S = np.array([voc[w] for w in informative_words()])
    freq = np.asarray(X.sum(0)).ravel()
    cS = np.asarray((X[:, S].T @ np.eye(2)[y]).T) + 0.5
    mS = freq[S].sum() / freq.sum()
    inS = np.zeros(len(freq), bool)
    inS[S] = True
    noise = np.array([j for j in np.argsort(-freq, kind="stable") if not inS[j]][:cfg.noise_words])
    pn = freq[noise] / freq[noise].sum()
    theta = np.hstack([mS * cS / cS.sum(1, keepdims=True), np.tile((1 - mS) * pn, (2, 1))])
    V = theta.shape[1]
    rep = np.where(_rng(2001).random(V) < 0.5, float(BURST), 1.0) if cfg.bursty else np.ones(V)
    return World(theta, rep, np.asarray(X.sum(1)).ravel().astype(int), len(S))


def sample_world(w: World, n: int, rng: np.random.Generator) -> tuple[sp.csr_matrix, np.ndarray]:
    """n documents: y uniform on {0, 1}, a corpus length L, multinomial counts from θ_y, each count times r_j."""
    y = rng.integers(0, 2, n)
    L = rng.choice(w.lengths, n)
    C = np.vstack([rng.multinomial(L[i], w.theta[y[i]]) for i in range(n)]).astype(float) * w.rep
    return sp.csr_matrix(C), y


def bayes_weights(w: World) -> tuple[np.ndarray, float]:
    """The Bayes discriminant on observed counts: Σ_j (x_j/r_j) log(θ_1j/θ_0j), linear in x (Prop 2(a))."""
    return (np.log(w.theta[1]) - np.log(w.theta[0])) / w.rep, 0.0


def nb_population_weights(w: World) -> tuple[np.ndarray, float]:
    """Prop 2(b): NB fit to infinitely many bursty documents estimates θ̃_k ∝ r ⊙ θ_k, so its weight per observed
    count is log(θ_1j/θ_0j) + log(Z_0/Z_1) with Z_k = Σ_j r_j θ_kj: r_j times the Bayes weight, plus a constant per
    count."""
    t = w.theta * w.rep
    Zk = t.sum(1)
    return np.log(w.theta[1]) - np.log(w.theta[0]) + np.log(Zk[0] / Zk[1]), 0.0


def accuracy(X, y: np.ndarray, wv: np.ndarray, b: float) -> float:
    return float(((X @ wv + b > 0).astype(int) == y).mean())


CURVE_N = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
CURVE_REPS = 4
NB_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
LR_LAMBDAS = (1e-1, 1e-2, 1e-3)
TEST_N, VALID_N = 3000, 1500


@lru_cache(maxsize=None)
def world_eval_sets(cfg: WorldConfig):
    rng = _rng(2002 + 10 * WORLD_NOISE.index(cfg.noise_words))  # common random numbers across bursty
    w = world(cfg)
    return sample_world(w, TEST_N, rng), sample_world(w, VALID_N, rng)


@lru_cache(maxsize=None)
def learning_curve(cfg: WorldConfig) -> dict:
    """NB and LR test accuracy against n, CURVE_REPS nested training draws per world (each rep draws 4096 documents
    and trains on prefixes). Each method's hyperparameter (α for NB, λ for LR) is chosen per fit on an independent
    validation draw — a luxury only a simulated world allows. Also the Bayes and NB-asymptotic accuracies."""
    w = world(cfg)
    (Xt, yt), (Xv, yv) = world_eval_sets(cfg)
    rng = _rng(2100 + 10 * WORLD_NOISE.index(cfg.noise_words))
    nb_acc = np.zeros((CURVE_REPS, len(CURVE_N)))
    lr_acc = np.zeros_like(nb_acc)
    nb_alpha = np.zeros_like(nb_acc)
    for rep in range(CURVE_REPS):
        Xall, yall = sample_world(w, CURVE_N[-1], rng)
        x0 = None
        for k, n in enumerate(CURVE_N):
            Xn, yn = Xall[:n], yall[:n]
            best = (-1.0, 0.0, 0.0)
            for a in NB_ALPHAS:
                W, b = nb_fit(Xn, yn, 2, a) if len(set(yn)) == 2 else (np.zeros((2, Xn.shape[1])), np.zeros(2))
                wv, bv = W[1] - W[0], b[1] - b[0]
                best = max(best, (accuracy(Xv, yv, wv, bv), accuracy(Xt, yt, wv, bv), a))
            nb_acc[rep, k], nb_alpha[rep, k] = best[1], best[2]
            best = (-1.0, 0.0)
            path = x0
            for lam in LR_LAMBDAS:
                path = blr_intercept_fit(Xn, yn, lam, path)
                if lam == LR_LAMBDAS[0]:
                    x0 = path  # the next n starts from this n's most-penalized fit
                best = max(best, (accuracy(Xv, yv, path[:-1], path[-1]), accuracy(Xt, yt, path[:-1], path[-1])))
            lr_acc[rep, k] = best[1]
    return {"V": int(w.theta.shape[1]), "bayes": accuracy(Xt, yt, *bayes_weights(w)),
            "nb_inf": accuracy(Xt, yt, *nb_population_weights(w)),
            "nb": nb_acc, "lr": lr_acc, "nb_alpha": nb_alpha}


def crossover_n(curve: dict) -> int | None:
    """The first n on the grid from which LR's mean accuracy stays at or above NB's (None if it never does)."""
    d = curve["lr"].mean(0) - curve["nb"].mean(0)
    for k in range(len(CURVE_N)):
        if np.all(d[k:] >= 0):
            return CURVE_N[k]
    return None


def rare_token_share(n_docs: int, cfg: WorldConfig, below: int = 5) -> float:
    """The share of test tokens whose word occurs fewer than ``below`` times among n_docs training documents — the
    part of every document Thm 3 says NB has not yet learned to relative accuracy."""
    (Xt, _), _ = world_eval_sets(cfg)
    Xn, _ = sample_world(world(cfg), n_docs, _rng(2200 + n_docs))
    c = np.asarray(Xn.sum(0)).ravel()
    t = np.asarray(Xt.sum(0)).ravel()
    return float(t[c < below].sum() / t.sum())


# --------------------------------------------------------------------------------------------
# 3c. What the worlds' prose rests on: the one Bayes accuracy, who bursts, once-seen words, α's climb, Thm 2's slack.
# --------------------------------------------------------------------------------------------

POP_N = 1_000_000  # documents behind a population accuracy: standard error about 0.00035


def collapsed_world(w: World) -> tuple[np.ndarray, np.ndarray]:
    """θ and r with the noise words merged into one lump per repeat factor. Noise words have one rate in both classes,
    so nothing the population rules read depends on which noise word occurred; V stops mattering."""
    S = w.n_informative
    noise = w.theta[:, S:]
    vals = np.unique(w.rep[S:])
    lumps = np.column_stack([noise[:, w.rep[S:] == r].sum(1) for r in vals])
    return np.hstack([w.theta[:, :S], lumps]), np.concatenate([w.rep[:S], vals])


def population_accuracy(w: World, prior: float = 0.5, n: int = POP_N, stream: int = 2500) -> tuple[float, float]:
    """(Bayes, population NB) accuracy on n fresh documents, y ~ Bernoulli(prior). Bayes reads Σ_j (x_j/r_j) log(θ_1j/θ_0j)
    + logit(prior); NB reads nb_population_weights(w) + logit(prior) (Prop 2)."""
    th, rep = collapsed_world(w)
    wb, wn = (np.log(th[1]) - np.log(th[0])) / rep, nb_population_weights(World(th, rep, w.lengths, w.n_informative))[0]
    logit = math.log(prior / (1 - prior))
    rng = _rng(stream)
    hb = hn = 0
    for s in range(0, n, 50_000):
        m = min(50_000, n - s)
        y = (rng.random(m) < prior).astype(int)
        L = rng.choice(w.lengths, m)
        C = np.zeros((m, th.shape[1]))
        for k in (0, 1):
            i = np.flatnonzero(y == k)
            C[i] = rng.multinomial(L[i], th[k])
        X = C * rep
        hb += int(((X @ wb + logit > 0) == y).sum())
        hn += int(((X @ wn + logit > 0) == y).sum())
    return hb / n, hn / n


def masked_world(nw: int, rep: np.ndarray) -> World:
    b = world(WorldConfig(nw, False))
    return World(b.theta, rep, b.lengths, b.n_informative)


def burst_mask(nw: int, who: str) -> np.ndarray:
    """Which words repeat BURST times: "none", "every word", "all informative", "half informative" (every other word
    of S), or "noise only"."""
    base = world(WorldConfig(nw, False))
    V, S = base.theta.shape[1], base.n_informative
    r = np.ones(V)
    if who == "every word":
        r[:] = BURST
    elif who == "all informative":
        r[:S] = BURST
    elif who == "half informative":
        r[:S:2] = BURST
    elif who == "noise only":
        r[S:] = BURST
    return r


@lru_cache(maxsize=None)
def population_table() -> dict:
    """The population accuracies: one Bayes accuracy for every world, and NB's by who bursts, with the page's two bursty
    worlds under the names of their V."""
    nw = WORLD_NOISE[0]
    out = {}
    for who in ("none", "every word", "all informative", "half informative", "noise only"):
        out[who] = population_accuracy(masked_world(nw, burst_mask(nw, who)))
    out["bayes"] = out["none"][0]
    for nw in WORLD_NOISE:
        w = world(WorldConfig(nw, True))
        out[f"page bursty V={w.theta.shape[1]}"] = population_accuracy(w)
    every = masked_world(WORLD_NOISE[0], burst_mask(WORLD_NOISE[0], "every word"))
    out["every word, prior 0.3"] = population_accuracy(every, 0.3)
    return out


def fast_world_sample(w: World, n: int, rng: np.random.Generator, *, keep: int | None = None):
    """sample_world in chunks of 2,000. With ``keep``, the stream's labels and lengths are still drawn for all n
    documents, but only the chunks covering the first ``keep`` are sampled: those documents match the full draw's."""
    y = rng.integers(0, 2, n)
    L = rng.choice(w.lengths, n)
    out = []
    for s in range(0, keep or n, 2000):
        yy, LL = y[s:s + 2000], L[s:s + 2000]
        C = np.zeros((len(yy), w.theta.shape[1]))
        for k in (0, 1):
            i = np.flatnonzero(yy == k)
            if len(i):
                C[i] = rng.multinomial(LL[i], w.theta[k])
        out.append(sp.csr_matrix(C * w.rep))
    return sp.vstack(out).tocsr(), y


BURST_N = (64, 256, 1024, 4096)
BURST_N_SLOW = (16384, 32768)
BURST_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
BURST_LAMBDAS = (0.1, 0.01, 1e-3, 1e-4)


@lru_cache(maxsize=None)
def burst_curve(nw: int, who: str, ns: tuple[int, ...] = BURST_N) -> dict:
    """Tuned NB and LR against n when only ``who`` bursts: four draws per n, tested on 6,000 documents. Validation and
    test come first in their stream, and every n of a draw is a prefix of the same 32,768-document stream."""
    w = masked_world(nw, burst_mask(nw, who))
    rng = np.random.default_rng(1720)
    Xt, yt = fast_world_sample(w, 6000, rng)
    Xv, yv = fast_world_sample(w, 1500, rng)
    N, L = np.zeros((4, len(ns))), np.zeros((4, len(ns)))
    for k in range(4):
        Xall, yall = fast_world_sample(w, 32768, np.random.default_rng([1721, nw, k]), keep=ns[-1])
        for i, n in enumerate(ns):
            Xn, yn = Xall[:n], yall[:n]
            best = (-1.0, 0.0)
            for a in BURST_ALPHAS:
                W, b = nb_fit(Xn, yn, 2, a)
                wv, bv = W[1] - W[0], b[1] - b[0]
                best = max(best, (accuracy(Xv, yv, wv, bv), accuracy(Xt, yt, wv, bv)))
            N[k, i] = best[1]
            best, p = (-1.0, 0.0), None
            for lam in BURST_LAMBDAS:
                p = blr_intercept_fit(Xn, yn, lam, p)
                best = max(best, (accuracy(Xv, yv, p[:-1], p[-1]), accuracy(Xt, yt, p[:-1], p[-1])))
            L[k, i] = best[1]
    return {"n": ns, "nb": N, "lr": L, "wins": (L > N).sum(0), "bayes": accuracy(Xt, yt, *bayes_weights(w)),
            "nb_inf": accuracy(Xt, yt, *nb_population_weights(w))}


def first_clean_sweep(c: dict) -> int | None:
    """The smallest n from which LR beats NB in all four draws at every larger n of the curve."""
    wins = c["wins"]
    for k in range(len(c["n"])):
        if np.all(wins[k:] == 4):
            return c["n"][k]
    return None


@lru_cache(maxsize=None)
def noise_score_spread(n: int = 256) -> dict:
    """What a ×4 burst on the noise words does to each fit: the SD, over 6,000 test documents, of the part of the score
    the noise words contribute, NB at α = 1 and LR at λ = 0.01, with the same underlying draws, four draws."""
    nw = WORLD_NOISE[0]
    plain = world(WorldConfig(nw, False))
    S = plain.n_informative
    rep = burst_mask(nw, "noise only")
    rng = np.random.default_rng(12)

    def draw(m, g):
        y = (g.random(m) < 0.5).astype(int)
        L = g.choice(plain.lengths, m)
        C = np.zeros((m, plain.theta.shape[1]))
        for k in (0, 1):
            i = np.flatnonzero(y == k)
            C[i] = g.multinomial(L[i], plain.theta[k])
        return sp.csr_matrix(C), y

    Xtu, _ = draw(6000, rng)
    out = {"nb": np.zeros((4, 2)), "lr": np.zeros((4, 2)), "lr_w": np.zeros((4, 2))}
    for d in range(4):
        Xu, y = draw(n, np.random.default_rng([11, n, d]))
        for j, rr in enumerate((np.ones(len(rep)), rep)):
            A, C = Xu @ sp.diags(rr), Xtu @ sp.diags(rr)
            W, _ = nb_fit(A, y, 2, 1.0)
            out["nb"][d, j] = float((C[:, S:] @ (W[1] - W[0])[S:]).std())
            p = blr_intercept_fit(A, y, 0.01)
            out["lr"][d, j] = float((C[:, S:] @ p[:-1][S:]).std())
            out["lr_w"][d, j] = float(np.abs(p[:-1][S:]).mean())
    return {k: v.mean(0) for k, v in out.items()}


def burst_weight_ratio() -> dict[float, float]:
    """Prop 3(a) with r = 4 on the noise words: how much a ×4 burst divides LR's noise weights, by λ, on 1,024 documents.
    It is exactly 4 only at λ = 0."""
    nw = WORLD_NOISE[0]
    wd = world(WorldConfig(nw, False))
    (Xt, yt), _ = world_eval_sets(WorldConfig(nw, False))
    X, y, S = Xt[:1024], yt[:1024], wd.n_informative
    Xb = X @ sp.diags(burst_mask(nw, "noise only"))
    out = {}
    for lam in (0.1, 0.01, 0.001, 1e-5):
        W, _, _ = lr_fit(X, y, 2, lam)
        W2, _, _ = lr_fit(Xb, y, 2, lam)
        out[lam] = float(np.abs((W[1] - W[0])[S:]).mean() / np.abs((W2[1] - W2[0])[S:]).mean())
    return out


@lru_cache(maxsize=None)
def tail_draws() -> dict:
    """Three 4,096-document draws of the long-tailed world, prefixes at each n: words seen exactly once in one class
    (mean of both classes) and in the pooled data, and what α = 10 does to validation accuracy against α = 1 through
    each of its three channels. "Imbalance" is the raw fit against one whose classes have equal token totals; "flatten"
    and "shrink" apply α to the informative words alone and to the noise words alone, classes equalized."""
    cfg = WorldConfig(WORLD_NOISE[-1], False)
    w = world(cfg)
    _, (Xv, yv) = world_eval_sets(cfg)
    S = w.n_informative
    per_class, pooled, effects = {}, {}, {}
    for rep in range(3):
        Xall, yall = fast_world_sample(w, 4096, np.random.default_rng([1705, rep]))
        for n in (16, 64, 256, 1024, 4096):
            Xn, yn = Xall[:n], yall[:n]
            cc = np.asarray((Xn.T @ np.eye(2)[yn]).T)
            per_class.setdefault(n, []).append(float((cc == 1).sum(1).mean()))
            pooled.setdefault(n, []).append(int((cc.sum(0) == 1).sum()))
            if n < 256:
                continue
            N = cc.sum(1)

            def eq_tokens(a):
                return accuracy(Xv, yv, np.log(cc[1] / N[1] * N.mean() + a) - np.log(cc[0] / N[0] * N.mean() + a), 0.0)

            W, b = nb_fit(Xn, yn, 2, 10.0)
            raw = accuracy(Xv, yv, W[1] - W[0], b[1] - b[0])
            base = eq_tokens(np.ones(cc.shape[1]))
            on_noise, on_info = np.full(cc.shape[1], 10.0), np.full(cc.shape[1], 1.0)
            on_noise[:S], on_info[:S] = 1.0, 10.0
            effects.setdefault(n, []).append((raw - eq_tokens(np.full(cc.shape[1], 10.0)), eq_tokens(on_info) - base,
                                              eq_tokens(on_noise) - base))
    return {"per_class": {n: float(np.mean(v)) for n, v in per_class.items()},
            "pooled": {n: v for n, v in pooled.items()},
            "effects": {n: np.mean(v, 0) for n, v in effects.items()}}


@lru_cache(maxsize=None)
def thm2_slack() -> dict:
    """Where Thm 2's 402 nats go, at m_0 documents per class on the Bernoulli world, mean of 20 draws: the largest |l̂ − l|
    on 4,000 test documents, the same document's terms with their real sizes and every sign aligned, the same with each
    term at its own worse case (the worst document anyone could build), and the median and largest single term."""
    p = bernoulli_world()
    m0, V = lemma1_m(BERN_EPS, BERN_DELTA, BERN_V), BERN_V
    rng = np.random.default_rng(99)
    yt = rng.integers(0, 2, 4000)
    Bt = (rng.random((4000, V)) < p[yt]).astype(float)
    rows = []
    for _ in range(20):
        ph = np.vstack([(rng.random((m0, V)) < p[k]).mean(0) for k in (0, 1)])
        e1, e0 = np.log(ph[1]) - np.log(p[1]), np.log(ph[0]) - np.log(p[0])
        f1, f0 = np.log1p(-ph[1]) - np.log1p(-p[1]), np.log1p(-ph[0]) - np.log1p(-p[0])
        T1, T0 = Bt * e1 + (1 - Bt) * f1, Bt * e0 + (1 - Bt) * f0
        worst = np.maximum(np.abs(e1), np.abs(f1)) + np.maximum(np.abs(e0), np.abs(f0))
        rows.append((np.abs((T1 - T0).sum(1)).max(), (np.abs(T1) + np.abs(T0)).sum(1).max(), worst.sum(),
                     np.median(np.abs(T1)), np.maximum.reduce([np.abs(e1), np.abs(f1), np.abs(e0), np.abs(f0)]).max()))
    a = np.mean(rows, 0)
    return {"signed": float(a[0]), "aligned": float(a[1]), "worst_doc": float(a[2]), "median_term": float(a[3]),
            "largest_term": float(a[4]), "per_term_bound": 2 * BERN_EPS / BERN_RHO0}


# --------------------------------------------------------------------------------------------
# 4. Separation (Thm 4).
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def tone_binary(n_docs: int = 300, seed: int = 0) -> tuple[sp.csr_matrix, np.ndarray]:
    """Signed rows z_i = y_i x_i (y = tone ∈ {−1, +1}) of the polar documents in generate_corpus(n_docs, seed)."""
    docs = [d for d in fc.generate_corpus(n_docs, seed) if d.tone != 0]
    X, _ = featurize([fc._tokens(d.text) for d in docs])
    s = np.array([d.tone for d in docs], float)
    return sp.csr_matrix(sp.diags(s) @ X), s


def hard_margin_svm(Z) -> np.ndarray:
    """The homogeneous hard-margin SVM ŵ = argmin ‖w‖² s.t. z_i·w ≥ 1, through its dual max Σa − ½aᵀGa, a ≥ 0,
    G = ZZᵀ; ŵ = Zᵀa."""
    G = np.asarray((Z @ Z.T).todense()) if sp.issparse(Z) else Z @ Z.T

    def f(a):
        Ga = G @ a
        return 0.5 * a @ Ga - a.sum(), Ga - 1.0

    r = optimize.minimize(f, np.zeros(G.shape[0]), jac=True, method="L-BFGS-B", bounds=[(0, None)] * G.shape[0],
                          options={"maxiter": 100000, "gtol": 1e-13, "ftol": 1e-20})
    return np.asarray(Z.T @ r.x).ravel()


SEP_LAMBDAS = tuple(10.0 ** -k for k in range(1, 8))
GD_STEPS = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)


@lru_cache(maxsize=None)
def separation() -> dict:
    """Thm 4 on the tone documents: the SVM direction and margin, the L2 path (‖ŵ_λ‖, cosine, normalized margin),
    and gradient descent at step 1/β (β = ‖Z‖²/(4n), the logistic loss's smoothness) from w = 0."""
    Z, _ = tone_binary()
    n = Z.shape[0]
    w_svm = hard_margin_svm(Z)
    u = w_svm / np.linalg.norm(w_svm)
    path, x0 = [], None
    for lam in SEP_LAMBDAS:
        x0 = blr_fit(Z, lam, x0)
        nw = np.linalg.norm(x0)
        path.append({"lam": lam, "norm": float(nw), "cos": float(x0 @ u / nw), "margin": float((Z @ x0).min() / nw)})
    sigma = sp.linalg.svds(Z, k=1, return_singular_vectors=False)[0]
    eta = 4 * n / sigma ** 2
    w = np.zeros(Z.shape[1])
    gd, t = [], 0
    for T in GD_STEPS:
        while t < T:
            w += eta * (Z.T @ expit(-(Z @ w))) / n
            t += 1
        nw = np.linalg.norm(w)
        gd.append({"t": T, "norm": float(nw), "cos": float(w @ u / nw), "loss": float(np.logaddexp(0, -(Z @ w)).mean())})
    return {"n": n, "V": Z.shape[1], "svm_margin": float((Z @ u).min()), "svm_norm": float(np.linalg.norm(w_svm)),
            "min_constraint": float((Z @ w_svm).min()), "path": path, "gd": gd, "eta": float(eta)}


# --------------------------------------------------------------------------------------------
# 5. Priors on features: NB-weighted LR and the smoothing bound (Prop 3).
# --------------------------------------------------------------------------------------------


def log_count_ratio(X, y01: np.ndarray, alpha: float = 1.0, *, binary: bool = True) -> np.ndarray:
    """Wang and Manning's r = log((p/‖p‖₁)/(q/‖q‖₁)), p = α + Σ_{y=1} f, q = α + Σ_{y=0} f, on binarized (or raw)
    features."""
    F = binarize(X) if binary else X
    p = alpha + np.asarray(F[y01 == 1].sum(0)).ravel()
    q = alpha + np.asarray(F[y01 == 0].sum(0)).ravel()
    return np.log(p / p.sum()) - np.log(q / q.sum())


def smoothing_bound(X, y01: np.ndarray, alpha: float) -> np.ndarray:
    """Prop 3(b)'s per-word bound on |r_j| with raw counts: log((c_j + α)/α) + |log((N_1 + αV)/(N_0 + αV))|, c_j the
    word's total count and N_k class k's token total."""
    c = np.asarray(X.sum(0)).ravel()
    N1, N0 = X[y01 == 1].sum(), X[y01 == 0].sum()
    V = X.shape[1]
    return np.log((c + alpha) / alpha) + abs(math.log((N1 + alpha * V) / (N0 + alpha * V)))


# --------------------------------------------------------------------------------------------
# 6. Finance case study: tone, the aspect XOR, and the full-text ceiling (Prop 4).
# --------------------------------------------------------------------------------------------

TONES = (-1, 0, 1)
FIN_TRAIN, FIN_DEV, FIN_TEST = (1500, 0), (600, 2), (1500, 1)  # (n_docs, seed) for generate_corpus
FIN_NOISE = (0.0, 0.15, 0.3)
FIN_ALPHAS = (0.1, 0.3, 1.0, 3.0)  # the page's table grid: α = 3, its top, wins on every split
FIN_LAMBDAS = (1e-2, 1e-3)  # the page's table grid: λ = 0.01, its top, wins on unigrams
FIN_ALPHAS_WIDE = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)  # past both edges, for the cuts that say "wide grid"
FIN_LAMBDAS_WIDE = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4)
FIN_MIN_DF = 2


def bayes_ceiling() -> float:
    """The full-text Bayes accuracy: each document has k ~ Uniform{5, …, 8} aspect sentences whose polarities are
    i.i.d. from _SENTENCE_TONE_MIX[tone], and nothing else in the text depends on the tone, so the polarity counts are
    sufficient. With a uniform tone prior, accuracy = Σ_k P(k) Σ_counts max_t P(counts | t, k)/3."""
    mix = {t: dict(zip(*fc._SENTENCE_TONE_MIX[t])) for t in TONES}
    acc = 0.0
    for k in range(5, 9):
        for a in range(k + 1):
            for b in range(k + 1 - a):
                cnt = (a, b, k - a - b)  # (+, 0, −) sentences
                acc += 0.25 * max(stats.multinomial.pmf(cnt, k, [mix[t][1], mix[t][0], mix[t][-1]]) for t in TONES) / 3
    return float(acc)


@lru_cache(maxsize=None)
def fin_split():
    """Tokens and clean tones (0, 1, 2 for −1, 0, +1) of the train, dev, and test corpora."""
    out = []
    for n, seed in (FIN_TRAIN, FIN_DEV, FIN_TEST):
        docs = fc.generate_corpus(n, seed)
        out += [[fc._tokens(d.text) for d in docs], np.array([d.tone + 1 for d in docs])]
    return tuple(out)


@lru_cache(maxsize=None)
def fin_train_labels(noise: float) -> np.ndarray:
    """Training tones under label_noise (the text is identical for every noise level, so only labels change)."""
    if noise == 0:
        return fin_split()[1]
    return np.array([d.tone + 1 for d in fc.generate_corpus(*FIN_TRAIN, label_noise=noise)])


@lru_cache(maxsize=None)
def fin_features(bigrams: bool):
    Ttr, _, Tdv, _, Tte, _ = fin_split()
    X, voc = featurize(Ttr, bigrams=bigrams, min_df=FIN_MIN_DF)
    return X, featurize(Tdv, voc, bigrams=bigrams)[0], featurize(Tte, voc, bigrams=bigrams)[0], voc


def nbw_ratios(X, y: np.ndarray, alpha: float = 1.0) -> list[np.ndarray]:
    """One-vs-rest NB weighting for three tones: class k's features are r_k ⊙ x̂ with r_k the log-count ratio of k
    against the rest (Wang and Manning's r, extended one-vs-rest)."""
    return [log_count_ratio(X, (y == k).astype(int), alpha) for k in range(3)]


@dataclass(frozen=True)
class FinConfig:
    method: str  # "nb" (multinomial NB) | "lr" (softmax LR) | "nbw" (one-vs-rest NB-weighted binary LR)
    bigrams: bool
    noise: float = 0.0
    wide: bool = False  # the wide grids instead of the page's table grids


@lru_cache(maxsize=None)
def fin_fit(cfg: FinConfig) -> dict:
    """Fit on the (noisy) training labels, pick the hyperparameter on the clean dev set, score the clean test set:
    accuracy, the confusion matrix, test probabilities (for calibration), and the fitted weights."""
    X, Xd, Xt, voc = fin_features(cfg.bigrams)
    alphas, lambdas = (FIN_ALPHAS_WIDE, FIN_LAMBDAS_WIDE) if cfg.wide else (FIN_ALPHAS, FIN_LAMBDAS)
    _, _, _, ydv, _, yte = fin_split()
    ytr = fin_train_labels(cfg.noise)
    best = None
    if cfg.method == "nb":
        for a in alphas:
            W, b = nb_fit(X, ytr, 3, a)
            dev = float((log_posterior(Xd, W, b).argmax(1) == ydv).mean())
            if best is None or dev > best[0]:
                best = (dev, a, W, b)
        dev, hp, W, b = best
        lp = log_posterior(Xt, W, b)
    elif cfg.method == "lr":
        x0 = None
        for lam in lambdas:
            W, b, x0 = lr_fit(X, ytr, 3, lam, x0)
            dev = float((log_posterior(Xd, W, b).argmax(1) == ydv).mean())
            if best is None or dev > best[0]:
                best = (dev, lam, W, b)
        dev, hp, W, b = best
        lp = log_posterior(Xt, W, b)
    else:
        rs = nbw_ratios(X, ytr)
        Bt, Bd, Bx = binarize(Xt), binarize(Xd), binarize(X)
        thetas = [None] * 3
        for lam in lambdas:
            for k in range(3):
                thetas[k] = blr_intercept_fit(Bx @ sp.diags(rs[k]), (ytr == k).astype(float), lam, thetas[k])
            sd = np.column_stack([(Bd @ sp.diags(rs[k])) @ thetas[k][:-1] + thetas[k][-1] for k in range(3)])
            dev = float((sd.argmax(1) == ydv).mean())
            if best is None or dev > best[0]:
                best = (dev, lam, [t.copy() for t in thetas])
        dev, hp, ths = best
        st_ = np.column_stack([(Bt @ sp.diags(rs[k])) @ ths[k][:-1] + ths[k][-1] for k in range(3)])
        lp = st_ - logsumexp(st_, axis=1, keepdims=True)  # a score, not a calibrated posterior
        W, b = np.vstack([rs[k] * ths[k][:-1] for k in range(3)]), np.array([t[-1] for t in ths])
    pred = lp.argmax(1)
    conf = np.zeros((3, 3), int)
    np.add.at(conf, (yte, pred), 1)
    return {"acc": float((pred == yte).mean()), "dev": dev, "hp": hp, "confusion": conf, "logp": lp,
            "W": W, "b": b, "voc": voc}


def fin_weight(cfg: FinConfig, feature: str) -> float:
    """The fitted positive-minus-negative weight W_{+,j} − W_{−,j} of one feature (nan if it is not in the vocabulary)."""
    f = fin_fit(cfg)
    j = f["voc"].get(feature)
    return float("nan") if j is None else float(f["W"][2, j] - f["W"][0, j])


def bin_index(conf: np.ndarray, B: int) -> np.ndarray:
    """Equal-width bins on [0, 1] (Topic 16's): bin k holds conf ∈ [k/B, (k+1)/B), conf = 1 joins the last bin."""
    return np.minimum((np.asarray(conf) * B).astype(int), B - 1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, B: int):
    """(count, mean confidence, accuracy) of every nonempty bin, in bin order (Topic 16's)."""
    b = bin_index(conf, B)
    n = np.bincount(b, minlength=B)
    s_conf = np.bincount(b, weights=conf, minlength=B)
    s_corr = np.bincount(b, weights=correct.astype(float), minlength=B)
    k = n > 0
    return n[k], s_conf[k] / n[k], s_corr[k] / n[k]


def ece_hat(conf: np.ndarray, correct: np.ndarray, B: int = 15) -> float:
    """The plug-in binned ECE: Σ_b (n_b/n)·|ā_b − c̄_b| (Topic 16's)."""
    n, c, a = reliability_bins(conf, correct, B)
    return float(n @ np.abs(a - c) / n.sum())


def fin_calibration(cfg: FinConfig) -> dict:
    f = fin_fit(cfg)
    yte = fin_split()[5]
    P = np.exp(f["logp"])
    conf, hit = P.max(1), P.argmax(1) == yte
    return {"conf": float(conf.mean()), "acc": float(hit.mean()), "ece": ece_hat(conf, hit, 15),
            "class_total_gap": float(np.abs(P.sum(0) - np.bincount(yte, minlength=3)).max() / len(yte))}


FIN_PREFIXES = (30, 100, 300, 1000, 1500)


@lru_cache(maxsize=None)
def fin_prefix_curve() -> dict:
    """NB and LR on the first n training documents (unigram features of the full training vocabulary, each method at
    the hyperparameter its full-data fit chose): the finance corpus's learning curve, which shows no crossover."""
    X, _, Xt, _ = fin_features(False)
    ytr, yte = fin_split()[1], fin_split()[5]
    a, lam = fin_fit(FinConfig("nb", False))["hp"], fin_fit(FinConfig("lr", False))["hp"]
    nb, lr = [], []
    for n in FIN_PREFIXES:
        W, b = nb_fit(X[:n], ytr[:n], 3, a)
        nb.append(float((log_posterior(Xt, W, b).argmax(1) == yte).mean()))
        W, b, _ = lr_fit(X[:n], ytr[:n], 3, lam)
        lr.append(float((log_posterior(Xt, W, b).argmax(1) == yte).mean()))
    return {"n": FIN_PREFIXES, "nb": nb, "lr": lr}


XOR_DOCS = ("revenue rose", "revenue fell", "costs rose", "costs fell")
XOR_LABELS = (1, -1, -1, 1)


def xor_separable(bigrams: bool) -> bool:
    """Prop 4 by linear programming: is there (w, b) with y(w·x + b) ≥ 1 on the four one-clause documents?"""
    X, _ = featurize([d.split() for d in XOR_DOCS], bigrams=bigrams)
    X = np.asarray(X.todense())
    y = np.array(XOR_LABELS, float)
    A = -y[:, None] * np.hstack([X, np.ones((4, 1))])
    r = optimize.linprog(np.zeros(X.shape[1] + 1), A_ub=A, b_ub=-np.ones(4), bounds=[(None, None)] * (X.shape[1] + 1),
                         method="highs")
    return r.status == 0


XOR_FEATURES = ("rose", "fell", "costs rose", "costs fell", "taxes rose", "sales rose", "not strong", "not weak",
                "strong", "weak")


# --------------------------------------------------------------------------------------------
# 6b. The case study's explanations, tested: six splits, cuts, redrawn words, exact readers.
# --------------------------------------------------------------------------------------------

FIN_SPLITS = ((0, 2, 1), (10, 110, 210), (11, 111, 211), (12, 112, 212), (13, 113, 213), (14, 114, 214))  # (train, dev,
#             test) seeds of generate_corpus; the first is the page's split
FIN_PATH = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4)
CUT_ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CUT_LAMBDAS = (0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4)


def _calib(lp: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(mean confidence, accuracy, 15-bin ECE) of log posteriors lp."""
    P = np.exp(lp)
    conf, hit = P.max(1), P.argmax(1) == y
    return float(conf.mean()), float(hit.mean()), ece_hat(conf, hit, 15)


def _acc(lp: np.ndarray, y: np.ndarray) -> float:
    return float((lp.argmax(1) == y).mean())


def _sub(mats, idx):
    return tuple(M[:, idx].tocsr() for M in mats)


def tune_nb(X, y, Xd, yd, Xt, yt, grid) -> tuple:
    """(α, confidence, accuracy, ECE, test log posteriors) at the α with the best dev accuracy, ties to the first."""
    best = None
    for a in grid:
        W, b = nb_fit(X, y, 3, a)
        dv = _acc(log_posterior(Xd, W, b), yd)
        if best is None or dv > best[0]:
            best = (dv, a, W, b)
    lp = log_posterior(Xt, best[2], best[3])
    return (best[1], *_calib(lp, yt), lp)


def tune_lr(X, y, Xd, yd, Xt, yt, grid) -> tuple:
    """tune_nb for softmax LR, warm-starting down the grid."""
    best, x0 = None, None
    for lam in grid:
        W, b, x0 = lr_fit(X, y, 3, lam, x0, gtol=1e-7)
        dv = _acc(log_posterior(Xd, W, b), yd)
        if best is None or dv > best[0]:
            best = (dv, lam, W, b)
    lp = log_posterior(Xt, best[2], best[3])
    return (best[1], *_calib(lp, yt), lp)


def tune_nbw(X, y, Xd, yd, Xt, yt, grid) -> tuple:
    """tune_nb for one-vs-rest NB-weighted LR (fin_fit's "nbw"); the "posteriors" are normalized scores."""
    rs = nbw_ratios(X, y)
    Bt, Bd, Bx = binarize(Xt), binarize(Xd), binarize(X)
    th, best = [None] * 3, None
    for lam in grid:
        for k in range(3):
            th[k] = blr_intercept_fit(Bx @ sp.diags(rs[k]), (y == k).astype(float), lam, th[k])
        sd = np.column_stack([(Bd @ sp.diags(rs[k])) @ th[k][:-1] + th[k][-1] for k in range(3)])
        dv = float((sd.argmax(1) == yd).mean())
        if best is None or dv > best[0]:
            best = (dv, lam, [t.copy() for t in th])
    st_ = np.column_stack([(Bt @ sp.diags(rs[k])) @ best[2][k][:-1] + best[2][k][-1] for k in range(3)])
    lp = st_ - logsumexp(st_, axis=1, keepdims=True)
    return (best[1], *_calib(lp, yt), lp)


@lru_cache(maxsize=None)
def fin_corpora(i: int) -> tuple:
    """fin_split's six items for split i of FIN_SPLITS."""
    if i == 0:
        return fin_split()
    out = []
    for n, seed in zip((FIN_TRAIN[0], FIN_DEV[0], FIN_TEST[0]), FIN_SPLITS[i]):
        docs = fc.generate_corpus(n, seed)
        out += [[fc._tokens(d.text) for d in docs], np.array([d.tone + 1 for d in docs])]
    return tuple(out)


@lru_cache(maxsize=None)
def fin_split_features(i: int, bigrams: bool) -> tuple:
    Ttr, _, Tdv, _, Tte, _ = fin_corpora(i)
    X, voc = featurize(Ttr, bigrams=bigrams, min_df=FIN_MIN_DF)
    return X, featurize(Tdv, voc, bigrams=bigrams)[0], featurize(Tte, voc, bigrams=bigrams)[0], voc


def lexicon_words(voc: dict[str, int]) -> tuple[list[str], list[str]]:
    """The generator's own tone vocabulary, in this vocabulary: its 42 polar, direction, and "not" words, and those plus
    every other lexicon word and aspect word (99)."""
    polar = set(fc.STANDIN_LEXICON["positive"]) | set(fc.STANDIN_LEXICON["negative"]) | set(fc._UP) | set(fc._DOWN) | {"not"}
    everything = set().union(*fc.STANDIN_LEXICON.values()) | set(fc._UP) | set(fc._DOWN) | {"not"}
    asp = {w for t in fc.ASPECT_TERMS for w in t.split()}
    return sorted(w for w in polar if w in voc), sorted(w for w in everything | asp if w in voc)


NEUTRAL_WORDS = ("line", "prior", "period", "broadly", "unchanged", "held", "steady")  # the neutral sentences' phrases


@lru_cache(maxsize=1)
def template_words() -> frozenset[str]:
    """Every word of the generator's fixed text: its templates, comments, openers, and closers."""
    out: set[str] = set()
    for grp in [*fc._FACT_TEMPLATES.values(), *fc._DISTRACTOR_TEMPLATES.values(), *fc._DOC_OPENERS.values(), fc._COMMENTS,
                fc._FOCUS, fc._CLOSERS]:
        for t in grp:
            out |= set(fc._tokens(re.sub(r"\{\w+\}", " ", t)))
    out |= set(fc._tokens("in line with the prior period broadly unchanged held steady across percent and was were this "
                          "period expectations management pointed to in but reported of for guided growth tax expense "
                          "stood at capital expenditures totaled crude production averaged thousand barrels per day "
                          "estimates change outlook remains given filing discloses a related review is ongoing lease "
                          "liability cost of revenue"))
    return frozenset(out)


def word_partition(voc: dict[str, int]) -> tuple[list[str], list[str], list[str]]:
    """Tone and aspect words, the other template words, and the content words the generator fills in: names, products,
    Pitman–Yor words."""
    A = lexicon_words(voc)[1]
    T = sorted(w for w in voc if w in template_words() and w not in set(A))
    N = sorted(w for w in voc if w not in set(A) and w not in template_words())
    return A, T, N


def chi2_rank(X, y: np.ndarray) -> np.ndarray:
    """Words by summed per-class chi-squared of word presence, on training data only."""
    B = binarize(X)
    n = X.shape[0]
    df = np.asarray(B.sum(0)).ravel()
    s = np.zeros(X.shape[1])
    for k in range(3):
        a = np.asarray(B[y == k].sum(0)).ravel()
        nk = (y == k).sum()
        e = df * nk / n
        s += (a - e) ** 2 / np.maximum(e, 1e-9) + ((df - a) - df * (n - nk) / n) ** 2 / np.maximum(df * (n - nk) / n, 1e-9)
    return np.argsort(-s, kind="stable")


@lru_cache(maxsize=None)
def fin_path(i: int, bigrams: bool) -> tuple[dict, ...]:
    """LR down FIN_PATH on split i: dev accuracy and log loss, test confidence, accuracy, and ECE, training accuracy, and
    the largest training class-total miss."""
    X, Xd, Xt, _ = fin_split_features(i, bigrams)
    _, ytr, _, ydv, _, yte = fin_corpora(i)
    rows, x0 = [], None
    for lam in FIN_PATH:
        W, b, x0 = lr_fit(X, ytr, 3, lam, x0, gtol=1e-7)
        lpd, lpt, lpx = log_posterior(Xd, W, b), log_posterior(Xt, W, b), log_posterior(X, W, b)
        c, a, e = _calib(lpt, yte)
        rows.append({"lam": lam, "dev": _acc(lpd, ydv), "dev_nll": float(-lpd[np.arange(len(ydv)), ydv].mean()), "conf": c,
                     "acc": a, "ece": e, "train_acc": _acc(lpx, ytr),
                     "class_miss": float(np.abs(np.exp(lpx).sum(0) - np.bincount(ytr, minlength=3)).max())})
    return tuple(rows)


@lru_cache(maxsize=None)
def fin_penalty_table() -> dict:
    """Six splits, LR, by feature set: every row of fin_path averaged per λ; the λ FIN_LAMBDAS' accuracy rule chose on
    each split; and the result of choosing λ by dev log loss on FIN_PATH instead."""
    out = {}
    for tag, bg in (("uni", False), ("bi", True)):
        paths = [fin_path(i, bg) for i in range(len(FIN_SPLITS))]
        by = {r["lam"]: {k: float(np.mean([p[j][k] for p in paths])) for k in r if k != "lam"} | {
            "ece_sd": float(np.std([p[j]["ece"] for p in paths], ddof=1))} for j, r in enumerate(paths[0])}
        narrow = [max((r for r in p if r["lam"] in FIN_LAMBDAS), key=lambda r: r["dev"]) for p in paths]
        ll = [min(p, key=lambda r: r["dev_nll"]) for p in paths]
        best = [max(p[1:], key=lambda r: r["dev"]) for p in paths]
        out[tag] = {"by": by, "chosen": [r["lam"] for r in narrow], "chosen_ece": float(np.mean([r["ece"] for r in narrow])),
                    "chosen_acc": float(np.mean([r["acc"] for r in narrow])),
                    "ll_lam": [r["lam"] for r in ll], "ll_ece": float(np.mean([r["ece"] for r in ll])),
                    "ll_acc": float(np.mean([r["acc"] for r in ll])), "best_acc": [r["acc"] for r in best]}
    return out


@lru_cache(maxsize=None)
def fin_grid_edges() -> dict:
    """Where the table grids stop. NB on unigrams over six splits at FIN_ALPHAS and at FIN_ALPHAS_WIDE; and the page's
    split refit on the wide grids, for every method."""
    a, w = [], []
    for i in range(len(FIN_SPLITS)):
        X, Xd, Xt, _ = fin_split_features(i, False)
        _, ytr, _, ydv, _, yte = fin_corpora(i)
        a.append(tune_nb(X, ytr, Xd, ydv, Xt, yte, FIN_ALPHAS)[:4])
        w.append(tune_nb(X, ytr, Xd, ydv, Xt, yte, FIN_ALPHAS_WIDE)[:4])
    wide = {}
    for m in FIN_METHODS:
        for bg in (False, True):
            c = FinConfig(m, bg, wide=True)
            wide[f"{m}-{'bi' if bg else 'uni'}"] = {"hp": fin_fit(c)["hp"], **fin_calibration(c)}
    return {"nb_alphas": [r[0] for r in a], "nb_acc": [r[2] for r in a], "nb_wide_alphas": [r[0] for r in w],
            "nb_wide_acc": [r[2] for r in w], "wide": wide}


@lru_cache(maxsize=None)
def fin_vocab_cuts() -> dict:
    """Six splits, tuned: NB and LR on the generator's 99 tone and aspect words, on the top 50 words by chi-squared, and on
    all words; NB on 42 polar words, on the 99 plus one word from each of three neutral phrases, and on the 99 plus all
    seven neutral-phrase words."""
    rows: dict[str, list] = {}
    for i in range(len(FIN_SPLITS)):
        X, Xd, Xt, voc = fin_split_features(i, False)
        _, ytr, _, ydv, _, yte = fin_corpora(i)
        mats = (X, Xd, Xt)
        V42, V99 = lexicon_words(voc)
        words = {"42": V42, "99": V99, "99+3": V99 + [w for w in ("line", "broadly", "steady") if w in voc and w not in V99],
                 "99+7": V99 + [w for w in NEUTRAL_WORDS if w in voc and w not in V99]}
        for tag, ws in words.items():
            A, B, C = _sub(mats, [voc[w] for w in ws])
            rows.setdefault("nb " + tag, []).append(tune_nb(A, ytr, B, ydv, C, yte, CUT_ALPHAS)[:4])
        A, B, C = _sub(mats, [voc[w] for w in V99])
        rows.setdefault("lr 99", []).append(tune_lr(A, ytr, B, ydv, C, yte, CUT_LAMBDAS)[:4])
        A, B, C = _sub(mats, chi2_rank(X, ytr)[:50])
        rows.setdefault("nb top50", []).append(tune_nb(A, ytr, B, ydv, C, yte, CUT_ALPHAS[1:])[:4])
        rows.setdefault("lr top50", []).append(tune_lr(A, ytr, B, ydv, C, yte, CUT_LAMBDAS)[:4])
        rows.setdefault("nb all", []).append(tune_nb(X, ytr, Xd, ydv, Xt, yte, FIN_ALPHAS_WIDE)[:4])
        best = max(fin_path(i, False), key=lambda r: r["dev"])
        rows.setdefault("lr all", []).append((best["lam"], best["conf"], best["acc"], best["ece"]))
    return {k: {"hp": [r[0] for r in v], "conf": float(np.mean([r[1] for r in v])), "acc": float(np.mean([r[2] for r in v])),
                "acc_sd": float(np.std([r[2] for r in v], ddof=1)), "ece": float(np.mean([r[3] for r in v]))}
            for k, v in rows.items()}


REDRAWS = {"real": (False, False), "content iid": (True, False), "templates iid": (False, True), "both iid": (True, True)}


@lru_cache(maxsize=None)
def fin_redraws() -> dict:
    """Keep each document's token counts in each word group and redraw the words inside the group independently: content
    words from their pooled training rates, tone and template words from their class's rates. Six splits for NB at
    α = 3; on the first four splits also tuned NB and LR, for the accuracy gap."""
    rows: dict[str, list] = {t: [] for t in REDRAWS}
    gaps: dict[str, list] = {t: [] for t in REDRAWS}
    for i in range(len(FIN_SPLITS)):
        X, Xd, Xt, voc = fin_split_features(i, False)
        _, ytr, _, ydv, _, yte = fin_corpora(i)
        A, T, N = word_partition(voc)
        iAT = np.array([voc[w] for w in A + T])
        iN = np.array([voc[w] for w in N])
        rng = np.random.default_rng([99, i])
        pool = np.asarray(X[:, iN].sum(0)).ravel().astype(float)
        pool /= pool.sum()
        th = {k: (lambda c: c / c.sum())(np.asarray(X[ytr == k][:, iAT].sum(0)).ravel().astype(float)) for k in range(3)}

        def rebuild(M, y, iid_n, iid_at):
            C = M.toarray().astype(float)
            if iid_n:
                C[:, iN] = rng.multinomial(C[:, iN].sum(1).astype(int), pool)
            if iid_at:
                L = C[:, iAT].sum(1).astype(int)
                for k in range(3):
                    r = np.flatnonzero(y == k)
                    C[np.ix_(r, iAT)] = rng.multinomial(L[r], th[k])
            return sp.csr_matrix(C)

        for tag, (a, b) in REDRAWS.items():
            Y, Yd, Yt = rebuild(X, ytr, a, b), rebuild(Xd, ydv, a, b), rebuild(Xt, yte, a, b)
            W, bb = nb_fit(Y, ytr, 3, 3.0)
            rows[tag].append(_calib(log_posterior(Yt, W, bb), yte))
            if i < 4:
                gaps[tag].append((tune_nb(Y, ytr, Yd, ydv, Yt, yte, FIN_ALPHAS_WIDE)[2],
                                  tune_lr(Y, ytr, Yd, ydv, Yt, yte, CUT_LAMBDAS[:5])[2]))
    return {t: {**dict(zip(("conf", "acc", "ece"), map(float, np.mean(rows[t], 0)))),
                **dict(zip(("nb", "lr"), map(float, np.mean(gaps[t], 0))))} for t in REDRAWS}


@lru_cache(maxsize=None)
def fin_bigram_gain() -> dict:
    """LR's gain over unigrams, six splits, paired: from the 84 bigrams that pair an aspect word with a direction word (the
    XOR of Prop 4), from those and the 16 "not" bigrams, and from all bigrams."""
    terms = {w for t in fc.ASPECT_TERMS for w in t.split()}
    dirs = set(fc._UP) | set(fc._DOWN)
    is_xor = lambda f: len(f.split()) == 2 and f.split()[0] in terms and f.split()[1] in dirs  # noqa: E731
    is_not = lambda f: len(f.split()) == 2 and f.split()[0] == "not"  # noqa: E731
    rows, counts = [], []
    for i in range(len(FIN_SPLITS)):
        Xb, Xdb, Xtb, vb = fin_split_features(i, True)
        _, ytr, _, ydv, _, yte = fin_corpora(i)
        uni = max(fin_path(i, False)[1:], key=lambda r: r["dev"])["acc"]
        every = max(fin_path(i, True)[1:], key=lambda r: r["dev"])["acc"]
        xor = [j for f, j in vb.items() if " " not in f or is_xor(f)]
        xor_not = [j for f, j in vb.items() if " " not in f or is_xor(f) or is_not(f)]
        a = tune_lr(*_sub((Xb,), xor), ytr, *_sub((Xdb,), xor), ydv, *_sub((Xtb,), xor), yte, FIN_PATH[1:])[2]
        b = tune_lr(*_sub((Xb,), xor_not), ytr, *_sub((Xdb,), xor_not), ydv, *_sub((Xtb,), xor_not), yte, FIN_PATH[1:])[2]
        rows.append((a - uni, b - uni, every - uni))
        counts.append((sum(is_xor(f) for f in vb), sum(is_not(f) for f in vb)))
    m = np.mean(rows, 0)
    return {"xor": float(m[0]), "xor_not": float(m[1]), "all": float(m[2]), "by_split": np.array(rows), "counts": counts}


PREFIX_TUNED = (8, 16, 30, 60, 100, 300)


@lru_cache(maxsize=None)
def fin_prefix_tuned() -> dict:
    """The learning curve, honestly tuned: on the first n documents of each split, with those documents' own vocabulary
    and each hyperparameter chosen at that n on the wide grids; and the same on the generator's tone and aspect words plus
    the neutral phrases. A prefix missing a tone is skipped."""
    rows: dict[tuple, list] = {}
    for i in range(len(FIN_SPLITS)):
        X, Xd, Xt, voc = fin_split_features(i, False)
        Ttr, ytr, Tdv, ydv, Tte, yte = fin_corpora(i)
        V99 = lexicon_words(voc)[1]
        clean = _sub((X, Xd, Xt), [voc[w] for w in V99 + [w for w in NEUTRAL_WORDS if w in voc and w not in V99]])
        for n in PREFIX_TUNED:
            yn = ytr[:n]
            if len(set(yn.tolist())) < 3:
                continue
            Y, vc = featurize(Ttr[:n], min_df=1 if n < 30 else 2)
            Yd, Yt = featurize(Tdv, vc)[0], featurize(Tte, vc)[0]
            rows.setdefault(("own", n), []).append((tune_nb(Y, yn, Yd, ydv, Yt, yte, FIN_ALPHAS_WIDE)[2],
                                                    tune_lr(Y, yn, Yd, ydv, Yt, yte, FIN_LAMBDAS_WIDE)[2]))
            A, B, C = clean
            rows.setdefault(("clean", n), []).append((tune_nb(A[:n], yn, B, ydv, C, yte, FIN_ALPHAS_WIDE)[2],
                                                      tune_lr(A[:n], yn, B, ydv, C, yte, FIN_LAMBDAS_WIDE)[2]))
    return {k: {"nb": float(np.mean(v, 0)[0]), "lr": float(np.mean(v, 0)[1]),
                "wins": int((np.array(v)[:, 1] > np.array(v)[:, 0]).sum()), "splits": len(v)} for k, v in rows.items()}


@lru_cache(maxsize=None)
def fin_noise_streams() -> dict:
    """The noise table over six streams of flips, stream 0 being generate_corpus's own. NB and LR on the table grids, LR
    and NB-weighted on the wide λ grid down to 0.003, below which no stream's tuned λ falls."""
    X, Xd, Xt, _ = fin_features(False)
    _, ytr, _, ydv, _, yte = fin_split()
    res: dict[float, list] = {q: [] for q in FIN_NOISE[1:]}
    lambdas = FIN_LAMBDAS_WIDE[:5]
    for stream in range(6):
        r = np.random.default_rng([170, stream])
        for q in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            flip, sh = r.random(len(ytr)) < q, r.integers(1, 3, len(ytr))
            if q not in res:
                continue
            y = fin_train_labels(q) if stream == 0 else (ytr + flip * sh) % 3
            lw = tune_lr(X, y, Xd, ydv, Xt, yte, lambdas)
            nw = tune_nbw(X, y, Xd, ydv, Xt, yte, lambdas)
            res[q].append((tune_nb(X, y, Xd, ydv, Xt, yte, FIN_ALPHAS)[2], tune_lr(X, y, Xd, ydv, Xt, yte, FIN_LAMBDAS)[2],
                           lw[2], nw[2], lw[0], float((y != ytr).mean())))
    out = {}
    for q, v in res.items():
        a = np.array(v)
        out[q] = {"nb": a[:, 0].mean(), "nb_sd": a[:, 0].std(ddof=1), "lr": a[:, 1].mean(), "lr_sd": a[:, 1].std(ddof=1),
                  "lr_wide": a[:, 2].mean(), "lr_wide_sd": a[:, 2].std(ddof=1), "nbw_wide": a[:, 3].mean(),
                  "nbw_wide_sd": a[:, 3].std(ddof=1), "lams": a[:, 4].tolist(), "flips": a[:, 5].tolist()}
    return out


def moment_gaps() -> dict:
    """Thm 1 on the finance unigrams: NB's class-total and co-count misses at α = 3 and α = 0.001, and LR's at its tuned
    λ, both raw (the co-count gap, which equals nλW) and as the residual of the penalized equations."""
    X = fin_features(False)[0]
    y = fin_split()[1]
    out = {}
    for a in (3.0, 1e-3):
        W, b = nb_fit(X, y, 3, a)
        R, t = estimating_residuals(X, y, W, b, 0.0)
        out[f"nb {a:g}"] = {"class": np.abs(t), "cocount": float(np.abs(R).max())}
    f = fin_fit(FinConfig("lr", False))
    raw, t = estimating_residuals(X, y, f["W"], f["b"], 0.0)
    pen, _ = estimating_residuals(X, y, f["W"], f["b"], f["hp"])
    out["lr"] = {"raw_max": float(np.abs(raw).max()), "raw_mean": float(np.abs(raw).mean()),
                 "residual": float(np.abs(pen).max()), "class": float(np.abs(t).max()),
                 "n_lam_W": float(np.abs(len(y) * f["hp"] * f["W"]).max())}
    return out


def two_word_picture() -> dict:
    """The (strong, weak) plane on the finance unigrams: each tone's share of training documents at the origin and its mean
    counts, and the test documents at the origin."""
    X, _, Xt, voc = fin_features(False)
    y, yte = fin_split()[1], fin_split()[5]
    i = [voc["strong"], voc["weak"]]
    A, C = X[:, i].toarray(), Xt[:, i].toarray()
    at = (A == 0).all(1)
    return {"origin": [float(at[y == k].mean()) for k in range(3)], "mean": [A[y == k].mean(0).tolist() for k in range(3)],
            "test_origin": int((C == 0).all(1).sum()), "n_test": len(yte), "median": np.median(A, 0).tolist()}


def phrase_counts(phrases: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """(occurrences, documents) of each phrase in the finance training corpus, on whole words."""
    text = [" ".join(t) for t in fin_split()[0]]
    out = {}
    for ph in phrases:
        pat = re.compile(r"(?<![a-z])" + ph + r"(?![a-z])")
        hits = [len(pat.findall(s)) for s in text]
        out[ph] = (sum(hits), sum(h > 0 for h in hits))
    return out


CONTRAST_QUOTE = ("Taxes declined 7 percent in quotalines, but lease liabilities rose 14 percent in equityflow, a trend "
                  "executives characterized as weaker.")


def aspect_census(n: int = FIN_TEST[0], seed: int = FIN_TEST[1]) -> dict:
    """Every aspect sentence of generate_corpus(n, seed) by how it carries its polarity. Wraps the generator's aspect
    sentence and document methods to log (polarity, text) without touching its random stream."""
    log: list = []
    per: list = []
    orig, orig_doc = fc._CorpusGen.aspect_sentence, fc._CorpusGen.doc

    def patched(self, b, pol, sector):
        start = b.length
        orig(self, b, pol, sector)
        log.append((pol, "".join(b.parts)[start:b.length]))

    def patched_doc(self, i, tone):
        log.clear()
        d = orig_doc(self, i, tone)
        per.append(list(log))
        return d

    fc._CorpusGen.aspect_sentence, fc._CorpusGen.doc = patched, patched_doc
    try:
        fc.generate_corpus(n, seed)
    finally:
        fc._CorpusGen.aspect_sentence, fc._CorpusGen.doc = orig, orig_doc
    pos, neg = set(fc.STANDIN_LEXICON["positive"]), set(fc.STANDIN_LEXICON["negative"])
    cnt: dict[str, int] = {}
    for doc in per:
        for q, txt in doc:
            toks = fc._tokens(txt.lower())
            sp_, sn = [t for t in toks if t in pos], [t for t in toks if t in neg]
            if q == 0:
                k = "neutral"
            elif ", but " in txt:
                k = "contrast"
            elif not sp_ and not sn:
                k = "direction only" if any(t in fc._UP or t in fc._DOWN for t in toks) else "none"
            elif sp_ and sn:
                k = "both"
            elif (1 if sp_ else -1) == q:
                k = "lexicon agrees"
            else:
                k = "not" if "not" in toks else "verb trap" if "expectations" in toks else "other"
            cnt[k] = cnt.get(k, 0) + 1
    tot = sum(cnt.values())
    return {"n": tot, **{k: v / tot for k, v in cnt.items()}}


def reader_accuracy(*, blind_contrast: bool = True, verb_surface: bool = False, surface_sign: bool = False) -> float:
    """The exact Bayes accuracy of a reader of the aspect sentences, by enumeration as in bayes_ceiling. Every reader sees
    neutral sentences. blind_contrast: a contrast sentence is one symbol, its clause order unread. verb_surface: a
    "beat"/"missed expectations" sentence shows only whether the verb's word is positive or negative, so a cost that
    "exceeded expectations" looks like a sale that did. surface_sign: on top of both, a sentence that carries its polarity
    only on a direction word is unreadable, and "not" + adjective is read as the adjective."""
    cats_ = list(fc.ASPECT_CATEGORIES)
    w = np.array([1 if fc.ASPECT_CATEGORIES[c][0] > 0 else 0.6 for c in cats_])
    pv = w[[fc.ASPECT_CATEGORIES[c][0] > 0 for c in cats_]].sum() / w.sum()
    pos, neg = pv * 0.85, 1 - pv  # non-contrast polar sentences: 15% of positive-valence ones use a "Management" noun
    silent = 1 - fc.DIRECTION_COMMENT_RATE
    dir_only, trap_neg, trap_verb, good_verb = pos * 0.35 * silent + neg * 0.8 * silent, pos * 0.10, neg * 0.2, pos * 0.15
    c_ = fc.CONTRAST_RATE
    mix = {t: dict(zip(*fc._SENTENCE_TONE_MIX[t])) for t in TONES}
    p_cat = {}
    for t in TONES:
        p, z, n = mix[t][1], mix[t][0], mix[t][-1]
        nc = 1 - c_
        blind = dir_only if surface_sign else 0.0
        trap = trap_neg if surface_sign else 0.0
        verb = good_verb + trap_verb if verb_surface else 0.0
        rest = 1 - blind - trap - verb
        cats = {"0": z, "+": (p * rest + n * trap) * nc, "-": (n * rest + p * trap) * nc}
        if blind:
            cats["?"] = (p + n) * nc * blind
        if verb:
            cats["V+"], cats["V-"] = (p * good_verb + n * trap_verb) * nc, (n * good_verb + p * trap_verb) * nc
        if blind_contrast:
            cats["C"] = (p + n) * c_
        else:
            cats["+"] += p * c_
            cats["-"] += n * c_
        p_cat[t] = [cats[k] for k in sorted(cats)]
    K = len(p_cat[1])

    def comps(k, m):
        if m == 1:
            yield (k,)
            return
        for i in range(k + 1):
            for r in comps(k - i, m - 1):
                yield (i, *r)

    acc = 0.0
    for k in range(5, 9):
        C = np.array(list(comps(k, K)))
        acc += 0.25 * np.max([stats.multinomial.pmf(C, k, p_cat[t]) for t in TONES], axis=0).sum() / 3
    return float(acc)


def direction_weights() -> dict:
    """LR's unigram weights, positive minus negative: the six direction words against all words."""
    f = fin_fit(FinConfig("lr", False))
    d = f["W"][2] - f["W"][0]
    a = np.abs(d)
    words = {w: float(d[f["voc"][w]]) for w in (*fc._UP, *fc._DOWN)}
    return {"words": words, "sd": float(d.std()), "median_abs": float(np.median(a)),
            "rose_top_share": float((a > abs(words["rose"])).mean())}


SIZE_N = (1500, 15000, 50000)
SIZE_LAMBDAS = (0.1, 0.03, 0.01, 0.003, 0.001, 3e-4)


def _tune_big(X, y, Xd, yd, Xt, yt) -> tuple[float, float, float]:
    """(λ, test accuracy, test ECE) at the best tuning accuracy on SIZE_LAMBDAS."""
    x0, best = None, None
    for lam in SIZE_LAMBDAS:
        W, b, x0 = lr_fit(X, y, 3, lam, x0, gtol=1e-6)
        d = _acc(log_posterior(Xd, W, b), yd)
        if best is None or d > best[0]:
            best = (d, lam, *_calib(log_posterior(Xt, W, b), yt)[1:])
    return best[1:]


@lru_cache(maxsize=None)
def training_sizes() -> dict:
    """The slow tier: past 1,500 documents, by concatenating independent corpora (each has its own content words), with
    3,000 tuning and 9,000 test documents. LR on all unigrams, on the top 200 by chi-squared, and on bigrams, and tuned
    NB on unigrams."""
    pool = [fc.generate_corpus(2500, 300 + s) for s in range(20)]
    Ttr = [fc._tokens(d.text) for c in pool for d in c]
    ytr = np.array([d.tone + 1 for c in pool for d in c])
    Tdv, ydv, Tte, yte = [], [], [], []
    for s in range(2):
        c = fc.generate_corpus(1500, 410 + s)
        Tdv += [fc._tokens(d.text) for d in c]
        ydv += [d.tone + 1 for d in c]
    for s in range(6):
        c = fc.generate_corpus(1500, 400 + s)
        Tte += [fc._tokens(d.text) for d in c]
        yte += [d.tone + 1 for d in c]
    ydv, yte = np.array(ydv), np.array(yte)
    out = {}
    for n in SIZE_N:
        T, y = Ttr[:n], ytr[:n]
        df = 2 if n < 5000 else 3
        X, voc = featurize(T, min_df=df)
        Xd, Xt = featurize(Tdv, voc)[0], featurize(Tte, voc)[0]
        top = chi2_rank(X, y)[:200]
        Xb, vb = featurize(T, bigrams=True, min_df=df)
        Xdb, Xtb = featurize(Tdv, vb, bigrams=True)[0], featurize(Tte, vb, bigrams=True)[0]
        out[n] = {"all": _tune_big(X, y, Xd, ydv, Xt, yte), "top200": _tune_big(X[:, top].tocsr(), y, Xd[:, top].tocsr(), ydv,
                                                                                Xt[:, top].tocsr(), yte),
                  "bi": _tune_big(Xb, y, Xdb, ydv, Xtb, yte),
                  "nb": tune_nb(X, y, Xd, ydv, Xt, yte, (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0))[:4],
                  "V": X.shape[1], "V_bi": Xb.shape[1]}
    return out


# --------------------------------------------------------------------------------------------
# 3b. A Bernoulli world that meets Ng and Jordan's hypotheses, for Lemma 1 and Thm 2.
# --------------------------------------------------------------------------------------------

BERN_V, BERN_RHO0, BERN_EPS, BERN_DELTA, BERN_REPS = 200, 0.1, 0.05, 0.2, 40


@lru_cache(maxsize=None)
def bernoulli_world() -> np.ndarray:
    """2 × BERN_V presence probabilities p(x_j = 1 | y): the BERN_V words with the highest document frequency among
    the corpus's polar documents, per-class document frequencies clipped to [ρ0, 1 − ρ0]. Bernoulli NB is the true
    model (features independent given y), with P(y = 1) = 1/2."""
    X, y, _ = corpus_binary()
    B = binarize(X)
    top = np.argsort(-np.asarray(B.sum(0)).ravel(), kind="stable")[:BERN_V]
    df = np.vstack([np.asarray(B[y == k][:, top].mean(0)).ravel() for k in (0, 1)])
    return np.clip(df, BERN_RHO0, 1 - BERN_RHO0)


def bernoulli_discriminant(p: np.ndarray, prior1: float, B: np.ndarray) -> np.ndarray:
    """l(x) = Σ_j log[p(x_j | 1)/p(x_j | 0)] + log[P(y = 1)/P(y = 0)], for binary rows B."""
    w = np.log(p[1]) - np.log(p[0]) - np.log1p(-p[1]) + np.log1p(-p[0])
    return B @ w + np.log1p(-p[1]).sum() - np.log1p(-p[0]).sum() + math.log(prior1 / (1 - prior1))


@lru_cache(maxsize=None)
def ng_jordan_check() -> dict:
    """Lemma 1 and Thm 2 on the Bernoulli world. Each of BERN_REPS reps draws m_0 = lemma1_m(ε, δ, V) documents per
    class (so 2m_0 in all, prior exactly 1/2) and fits unsmoothed NB. Records the share of reps whose largest
    parameter deviation exceeds ε (Lemma 1 says ≤ δ), the deviations' 1 − δ quantile, and, on a fresh test sample,
    whether every document ĥ gets wrong and h_∞ gets right has |l_∞| ≤ τV (Thm 2), the largest |l̂ − l_∞| seen, the
    bound τV, and G(τ)."""
    p = bernoulli_world()
    m0 = lemma1_m(BERN_EPS, BERN_DELTA, BERN_V)
    rng = _rng(2400)
    yt = rng.integers(0, 2, 4000)
    Bt = (rng.random((4000, BERN_V)) < p[yt]).astype(float)
    l_inf = bernoulli_discriminant(p, 0.5, Bt)
    right_inf = (l_inf > 0) == (yt == 1)
    devs, inclusion, max_gap, gaps_ok = [], True, 0.0, True
    for _ in range(BERN_REPS):
        phat = np.vstack([(rng.random((m0, BERN_V)) < p[k]).mean(0) for k in (0, 1)])
        eps_hat = float(np.abs(phat - p).max())
        devs.append(eps_hat)
        l_hat = bernoulli_discriminant(np.clip(phat, 1e-12, 1 - 1e-12), 0.5, Bt)
        gap = np.abs(l_hat - l_inf)
        max_gap = max(max_gap, float(gap.max()))
        tauV = thm2_tau(eps_hat, BERN_RHO0, BERN_V) * BERN_V
        gaps_ok &= bool(gap.max() <= tauV) and eps_hat <= BERN_RHO0 / 2
        wrong_hat = (l_hat > 0) != (yt == 1)
        inclusion &= bool(np.all(np.abs(l_inf[wrong_hat & right_inf]) <= tauV))
    tauV = thm2_tau(BERN_EPS, BERN_RHO0, BERN_V) * BERN_V
    devs = np.array(devs)
    return {"m0": m0, "fail_rate": float((devs > BERN_EPS).mean()), "dev_quantile": float(np.quantile(devs, 1 - BERN_DELTA)),
            "dev_max": float(devs.max()), "inclusion": inclusion, "gaps_within_bound": gaps_ok,
            "max_gap": max_gap, "tauV": tauV, "G": float(np.mean(right_inf & (np.abs(l_inf) <= tauV))),
            "median_abs_l": float(np.median(np.abs(l_inf))), "bayes_err": float(1 - right_inf.mean()),
            "eps_for_unit_tau": BERN_RHO0 / (4 * (BERN_V + 1)),
            "m0_for_unit_tau": lemma1_m(BERN_RHO0 / (4 * (BERN_V + 1)), BERN_DELTA, BERN_V)}


def zipf_world_tokens() -> dict:
    """Thm 3 on the long-tailed world (V = 3,071): θ_min, the token count Thm 3(a) asks for at ε = 1/2 and δ = 0.1,
    the tokens per class that 4,096 documents supply, the expected number of words not yet seen at that size, and the
    share of test tokens from words seen fewer than 5 times, by training size. For Zipf(1.1) on 3,071 words, the same
    token count."""
    w = world(WorldConfig(WORLD_NOISE[-1], False))
    th = w.theta[0]
    per_class = float(w.lengths.mean() * CURVE_N[-1] / 2)
    return {"V": int(len(th)), "theta_min": float(th.min()),
            "tokens_needed": thm3_tokens(0.5, 0.1, float(th.min()), len(th)),
            "tokens_supplied": per_class,
            "unseen_words": float(np.sum((1 - th) ** per_class)),
            "zipf_tokens_needed": thm3_tokens(0.5, 0.1, zipf_theta_min(len(th), 1.1), len(th)),
            "uniform_tokens_needed": thm3_tokens(0.5, 0.1, 1 / len(th), len(th)),
            "rare_share": {n: rare_token_share(n, WorldConfig(WORLD_NOISE[-1], False)) for n in (64, 512, 4096)}}


# --------------------------------------------------------------------------------------------
# 5b. The funnel: NB log-count ratios and LR weights against word frequency (Prop 3, lab view 2).
# --------------------------------------------------------------------------------------------

FUNNEL_ALPHAS = (0.1, 1.0, 10.0)
FUNNEL_LAMBDAS = (1e-2, 1e-3, 1e-4)
FUNNEL_SAMPLE = 500


@lru_cache(maxsize=None)
def funnel() -> dict:
    """On the training corpus's 1,000 polar documents (V = 6,307): every word's count, its multinomial log-count ratio r_j at
    each α with Prop 3(b)'s bound, and binary LR's weight at each λ — for S and FUNNEL_SAMPLE other words (stream
    2300)."""
    X, y, voc = corpus_binary()
    inv = {i: w for w, i in voc.items()}
    c = np.asarray(X.sum(0)).ravel()
    S = {voc[w] for w in informative_words()}
    others = np.array(sorted(set(range(X.shape[1])) - S))
    pick = np.array(sorted(S | set(_rng(2300).choice(others, FUNNEL_SAMPLE, replace=False).tolist())))
    r = {a: log_count_ratio(X, y, a, binary=False) for a in FUNNEL_ALPHAS}
    bound = {a: smoothing_bound(X, y, a) for a in FUNNEL_ALPHAS}
    lr, x0 = {}, None
    for lam in FUNNEL_LAMBDAS:
        x0 = blr_intercept_fit(X, y.astype(float), lam, x0)
        lr[lam] = x0[:-1]
    return {"pick": pick, "words": [inv[j] for j in pick], "count": c, "informative": np.isin(pick, list(S)),
            "r": r, "bound": bound, "lr": lr, "all_within_bound": all(bool(np.all(np.abs(r[a]) <= bound[a] + 1e-12))
                                                                      for a in FUNNEL_ALPHAS)}


# --------------------------------------------------------------------------------------------
# 7. The lab payload and the summary.
# --------------------------------------------------------------------------------------------

FIN_METHODS = ("nb", "lr", "nbw")


def _g(x) -> str:
    return f"{x:g}"


def _curve_payload(cfg: WorldConfig) -> dict:
    c = learning_curve(cfg)
    pop = population_table()
    out = {"V": c["V"], "bursty": cfg.bursty, "bayes": _r(pop["bayes"]),
           "nb_inf": _r(pop[f"page bursty V={c['V']}"][1] if cfg.bursty else pop["bayes"]), "crossover": crossover_n(c)}
    for m in ("nb", "lr"):
        out[m] = {"mean": _r(c[m].mean(0)), "lo": _r(c[m].min(0)), "hi": _r(c[m].max(0))}
    return out


def _fin_payload() -> dict:
    yte = fin_split()[5]
    methods = {}
    for m in FIN_METHODS:
        for bg in (False, True):
            f = fin_fit(FinConfig(m, bg))
            cal = fin_calibration(FinConfig(m, bg))
            methods[f"{m}-{'bi' if bg else 'uni'}"] = {
                "acc": _r(f["acc"]), "confusion": f["confusion"].tolist(), "hp": _g(f["hp"]),
                "conf": _r(cal["conf"]), "ece": _r(cal["ece"]), "V": len(f["voc"])}
    noise = {m: _r([fin_fit(FinConfig(m, False, q))["acc"] for q in FIN_NOISE]) for m in FIN_METHODS}
    xor = {f: {"uni": _r(fin_weight(FinConfig("lr", False), f)), "bi": _r(fin_weight(FinConfig("lr", True), f))}
           for f in XOR_FEATURES}
    return {"ceiling": _r(bayes_ceiling()), "n_test": int(len(yte)), "methods": methods,
            "noise": {"levels": list(FIN_NOISE), **noise}, "xor": xor}


def viz_payload() -> dict:
    fun = funnel()
    pick = fun["pick"]
    sep = separation()
    return {
        "curves": {"n": list(CURVE_N), "worlds": [_curve_payload(WorldConfig(nw, b)) for nw in WORLD_NOISE
                                                   for b in (False, True)]},
        "funnel": {"alphas": [_g(a) for a in FUNNEL_ALPHAS], "lambdas": [_g(lam) for lam in FUNNEL_LAMBDAS],
                   "words": fun["words"], "count": [int(fun["count"][j]) for j in pick],
                   "informative": [bool(v) for v in fun["informative"]],
                   "r": {_g(a): _r(fun["r"][a][pick]) for a in FUNNEL_ALPHAS},
                   "bound": {_g(a): _r(fun["bound"][a][pick]) for a in FUNNEL_ALPHAS},
                   "lr": {_g(lam): _r(fun["lr"][lam][pick]) for lam in FUNNEL_LAMBDAS}},
        "separation": {"n": sep["n"], "V": sep["V"], "svm_margin": _r(sep["svm_margin"]),
                       "slope": _r(math.log(10) / sep["svm_margin"]),
                       "path": [{"lam": _g(p["lam"]), "norm": _r(p["norm"]), "cos": _r(p["cos"]),
                                 "margin": _r(p["margin"])} for p in sep["path"]],
                       "gd": [{"t": g["t"], "norm": _r(g["norm"]), "cos": _r(g["cos"]), "loss": _r(g["loss"], 5)}
                              for g in sep["gd"]]},
        "finance": _fin_payload(),
    }


VIZ_JSON = (Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data"
            / "linear-text-classifiers-lab.json")


def summary() -> dict:
    out = {}
    for nw in WORLD_NOISE:
        for b in (False, True):
            c = learning_curve(WorldConfig(nw, b))
            tag = f"world V={c['V']} {'bursty' if b else 'NB-true'}"
            out[f"{tag}: Bayes / NB-asymptotic acc"] = f"{c['bayes']:.6f} / {c['nb_inf']:.6f}"
            out[f"{tag}: NB / LR acc at n=4096"] = f"{c['nb'].mean(0)[-1]:.6f} / {c['lr'].mean(0)[-1]:.6f}"
            out[f"{tag}: crossover n"] = crossover_n(c)
            out[f"{tag}: median validated NB alpha by n"] = " ".join(f"{a:g}" for a in np.median(c["nb_alpha"], 0))
    nj = ng_jordan_check()
    for k in ("m0", "fail_rate", "dev_quantile", "max_gap", "tauV", "G", "median_abs_l", "m0_for_unit_tau"):
        out[f"Ng-Jordan world: {k}"] = nj[k]
    zt = zipf_world_tokens()
    for k in ("theta_min", "tokens_needed", "tokens_supplied", "unseen_words", "zipf_tokens_needed",
              "uniform_tokens_needed"):
        out[f"Thm 3: {k}"] = zt[k]
    for n, v in zt["rare_share"].items():
        out[f"Thm 3: rare-token share at n={n}"] = v
    sep = separation()
    out["Thm 4: SVM normalized margin"] = sep["svm_margin"]
    out["Thm 4: path cos at lam=1e-7"] = sep["path"][-1]["cos"]
    out["Thm 4: path norm at lam=1e-6 / 1e-7"] = f"{sep['path'][-2]['norm']:.6f} / {sep['path'][-1]['norm']:.6f}"
    out["Thm 4: GD cos at t=1e5"] = sep["gd"][-1]["cos"]
    out["finance: full-text Bayes accuracy"] = bayes_ceiling()
    for m in FIN_METHODS:
        for bg in (False, True):
            cfg = FinConfig(m, bg)
            cal = fin_calibration(cfg)
            out[f"finance {m}-{'bi' if bg else 'uni'}: acc / conf / ECE15 (hp)"] = (
                f"{cal['acc']:.6f} / {cal['conf']:.6f} / {cal['ece']:.6f} ({fin_fit(cfg)['hp']:g})")
    for m in FIN_METHODS:
        out[f"finance {m}-uni accuracy by label noise {FIN_NOISE}"] = " / ".join(
            f"{fin_fit(FinConfig(m, False, q))['acc']:.6f}" for q in FIN_NOISE)
    c = fin_prefix_curve()
    out[f"finance unigram prefixes {FIN_PREFIXES}: NB"] = " / ".join(f"{a:.6f}" for a in c["nb"])
    out[f"finance unigram prefixes {FIN_PREFIXES}: LR"] = " / ".join(f"{a:.6f}" for a in c["lr"])
    for f in XOR_FEATURES:
        out[f"LR weight (+ minus −) of '{f}': uni / bi"] = (
            f"{fin_weight(FinConfig('lr', False), f):.6f} / {fin_weight(FinConfig('lr', True), f):.6f}")
    pop = population_table()
    out["worlds: Bayes accuracy (10^6 documents)"] = pop["bayes"]
    for k, v in pop.items():
        if k != "bayes":
            out[f"worlds: Bayes / NB population, {k}"] = f"{v[0]:.6f} / {v[1]:.6f}"
    for nw in WORLD_NOISE:
        for who in ("noise only", "half informative"):
            c = burst_curve(nw, who)
            out[f"bursts V={nw + 71} {who}: NB / LR by n {BURST_N}"] = (
                " ".join(f"{a:.3f}" for a in c["nb"].mean(0)) + " / " + " ".join(f"{a:.3f}" for a in c["lr"].mean(0)))
            out[f"bursts V={nw + 71} {who}: first n LR wins 4/4 from"] = first_clean_sweep(c)
    t = tail_draws()
    out["tail: pooled once-seen words by n"] = str({n: round(float(np.mean(v))) for n, v in t["pooled"].items()})
    out["tail: per-class once-seen words by n"] = str({n: round(v) for n, v in t["per_class"].items()})
    for n, e in t["effects"].items():
        out[f"tail: alpha 10 vs 1 at n={n}: imbalance / flatten / shrink"] = " / ".join(f"{x:+.3f}" for x in e)
    for k, v in thm2_slack().items():
        out[f"Thm 2 slack: {k}"] = v
    g = moment_gaps()
    out["moments: NB class / co-count at alpha 3"] = f"{g['nb 3']['class'].max():.3f} / {g['nb 3']['cocount']:.3f}"
    out["moments: NB class / co-count at alpha 0.001"] = f"{g['nb 0.001']['class'].max():.3f} / {g['nb 0.001']['cocount']:.3f}"
    out["moments: LR raw max / mean / residual / class"] = (
        f"{g['lr']['raw_max']:.3f} / {g['lr']['raw_mean']:.3f} / {g['lr']['residual']:.2e} / {g['lr']['class']:.2e}")
    for kw in (dict(), dict(verb_surface=True), dict(verb_surface=True, surface_sign=True)):
        out[f"readers: {' '.join(kw) or 'contrast blind'}"] = reader_accuracy(**kw)
    for tag, row in fin_penalty_table().items():
        out[f"penalty {tag}: chosen lambdas / ll ECE / ll acc"] = f"{row['chosen']} / {row['ll_ece']:.6f} / {row['ll_acc']:.6f}"
        for lam in (0.03, 0.01, 0.003, 0.001):
            b = row["by"][lam]
            out[f"penalty {tag} lam={lam:g}: ECE / acc / train acc"] = f"{b['ece']:.6f} / {b['acc']:.6f} / {b['train_acc']:.6f}"
    for k, v in fin_vocab_cuts().items():
        out[f"cuts {k}: conf / acc / ECE"] = f"{v['conf']:.6f} / {v['acc']:.6f} / {v['ece']:.6f}"
    for k, v in fin_redraws().items():
        out[f"redraw {k}: conf / acc / ECE; NB / LR"] = (
            f"{v['conf']:.6f} / {v['acc']:.6f} / {v['ece']:.6f}; {v['nb']:.6f} / {v['lr']:.6f}")
    g = fin_bigram_gain()
    out["bigram gain: xor / xor+not / all"] = f"{g['xor']:.6f} / {g['xor_not']:.6f} / {g['all']:.6f}"
    for k, v in fin_prefix_tuned().items():
        out[f"prefix {k[0]} n={k[1]}: NB / LR / LR wins"] = f"{v['nb']:.6f} / {v['lr']:.6f} / {v['wins']}/{v['splits']}"
    for q, v in fin_noise_streams().items():
        out[f"noise {q}: NB / LR / LR wide / NBW wide"] = (
            f"{v['nb']:.6f} / {v['lr']:.6f} / {v['lr_wide']:.6f} / {v['nbw_wide']:.6f}")
    e = fin_grid_edges()
    out["grid edges: six-split NB narrow / wide"] = f"{np.mean(e['nb_acc']):.6f} / {np.mean(e['nb_wide_acc']):.6f}"
    for k, v in e["wide"].items():
        out[f"wide grid {k}: hp, acc / conf / ECE"] = f"{v['hp']:g}, {v['acc']:.6f} / {v['conf']:.6f} / {v['ece']:.6f}"
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: one test per claim on the page.
# --------------------------------------------------------------------------------------------


def _small_problem(stream: int, n: int = 40, V: int = 6, K: int = 3):
    rng = _rng(stream)
    X = sp.csr_matrix(rng.poisson(1.5, (n, V)).astype(float))
    return X, rng.integers(0, K, n), rng


def test_nb_is_softmax_linear():
    """Prop 1(a): Prop 1's (W, b) reproduce the literal Bayes rule, multinomial and Bernoulli, to 1e-9."""
    X, y, _ = _small_problem(3000)
    for bern in (False, True):
        W, b = nb_fit(X, y, 3, 0.7, bernoulli=bern)
        lp = log_posterior(X, W, b, bernoulli=bern)
        if bern:
            theta, prior = expit(W), b - np.log1p(-expit(W)).sum(1)
        else:
            theta, prior = np.exp(W), b
        for i in range(X.shape[0]):
            d = nb_log_posterior_direct(np.asarray(X[i].todense()).ravel(), theta, prior, bernoulli=bern)
            assert np.abs(d - lp[i]).max() < 1e-9


def test_nb_matches_scikit_learn():
    """Prop 1's NB equals scikit-learn's MultinomialNB and BernoulliNB to 1e-8 on the finance unigrams."""
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB

    X, _, Xt, _ = fin_features(False)
    y = fin_split()[1]
    for a in (0.3, 1.0):
        W, b = nb_fit(X, y, 3, a)
        assert np.abs(MultinomialNB(alpha=a).fit(X, y).predict_log_proba(Xt) - log_posterior(Xt, W, b)).max() < 1e-8
        W, b = nb_fit(X, y, 3, a, bernoulli=True)
        sk = BernoulliNB(alpha=a, binarize=0.0).fit(X, y).predict_log_proba(Xt)
        assert np.abs(sk - log_posterior(Xt, W, b, bernoulli=True)).max() < 1e-8


def test_fixed_length_families_coincide():
    """Prop 1(b): any softmax(Wx + b) is an NB posterior on documents of one length L; with lengths varying, LR's
    family is larger (a one-word vocabulary: every NB posterior ignores x, softmax(Wx + b) need not)."""
    rng = _rng(3001)
    W, b = rng.normal(size=(3, 8)), rng.normal(size=3)
    for L in (5, 40):
        theta, logpi = nb_from_logits(W, b, L)
        X = rng.multinomial(L, np.full(8, 1 / 8), size=30).astype(float)
        nb = np.vstack([nb_log_posterior_direct(x, theta, logpi) for x in X])
        assert np.abs(nb - log_posterior(X, W, b)).max() < 1e-10
    lengths = np.array([[3.0], [30.0]])
    assert np.abs(np.diff(log_posterior(lengths, *nb_fit(sp.csr_matrix(lengths), np.array([0, 1]), 2, 1.0)), axis=0)).max() < 1e-12
    assert np.abs(np.diff(log_posterior(lengths, np.array([[0.0], [0.1]]), np.zeros(2)), axis=0)).max() > 1.0


def test_gradient_and_hessian():
    """Thm 1: lr_objective's gradient, lr_hessian, and lr_hessp agree with central differences (1e-6) and each
    other (1e-10); the Hessian is PSD, with a null space of dimension V + 1 at λ = 0 and 1 at λ > 0."""
    X, y, rng = _small_problem(3002)
    n, V = X.shape
    th = rng.normal(scale=0.3, size=3 * V + 3)
    for lam in (0.0, 0.05):
        L, g = lr_objective(th, X, y, 3, lam)
        h = 1e-6
        fd = np.array([(lr_objective(th + h * e, X, y, 3, lam, with_grad=False)
                        - lr_objective(th - h * e, X, y, 3, lam, with_grad=False)) / (2 * h) for e in np.eye(len(th))])
        assert np.abs(fd - g).max() < 1e-6
        H = lr_hessian(th[:3 * V].reshape(3, V), th[3 * V:], X, lam)
        fdH = np.array([(lr_objective(th + h * e, X, y, 3, lam)[1] - lr_objective(th - h * e, X, y, 3, lam)[1]) / (2 * h)
                        for e in np.eye(len(th))])
        assert np.abs(fdH - H).max() < 1e-6
        v = rng.normal(size=len(th))
        assert np.abs(lr_hessp(th, v, X, y, 3, lam) - H @ v).max() < 1e-10
        ev = np.linalg.eigvalsh(H)
        assert ev.min() > -1e-12
        assert int(np.sum(ev < 1e-10)) == (V + 1 if lam == 0 else 1)


def test_lr_matches_scikit_learn():
    """Thm 1's fitter agrees with scikit-learn's multinomial LogisticRegression (λ = 1/(Cn), intercept unpenalized) to
    1e-4 on probabilities, on the finance unigrams' 300 most frequent words, and reaches an objective no higher (the gap
    is scikit-learn's L-BFGS stopping at ~1e-5; ours stops at gradient 1e-10)."""
    from sklearn.linear_model import LogisticRegression

    X, _, Xt, _ = fin_features(False)
    y = fin_split()[1]
    top = np.argsort(-np.asarray(X.sum(0)).ravel(), kind="stable")[:300]
    Xs, Xts = X[:, top], Xt[:, top]
    lam = 1e-2
    W, b, _ = lr_fit(Xs, y, 3, lam)
    sk = LogisticRegression(C=1 / (lam * Xs.shape[0]), tol=1e-12, max_iter=20000).fit(Xs, y)
    assert np.abs(sk.predict_proba(Xts) - np.exp(log_posterior(Xts, W, b))).max() < 1e-4
    ours = lr_objective(np.concatenate([W.ravel(), b]), Xs, y, 3, lam, with_grad=False)
    theirs = lr_objective(np.concatenate([sk.coef_.ravel(), sk.intercept_]), Xs, y, 3, lam, with_grad=False)
    assert ours <= theirs + 1e-12


def test_estimating_equations():
    """Thm 1: at LR's optimum Σ_i (y_ik − p_ik) x_ij = nλW_kj and Σ_i p_ik = n_k. At λ = 0 on a non-separable
    problem (the 10 most frequent finance words) every class × word co-count is matched. NB's fit satisfies neither:
    its class totals miss by more than 1% of n on the finance unigrams."""
    f = fin_fit(FinConfig("lr", False))
    X = fin_features(False)[0]
    y = fin_split()[1]
    R, t = estimating_residuals(X, y, f["W"], f["b"], f["hp"])
    assert np.abs(R).max() < 1e-6 and np.abs(t).max() < 1e-6
    top = np.argsort(-np.asarray(X.sum(0)).ravel(), kind="stable")[:10]
    W0, b0, _ = lr_fit(X[:, top], y, 3, 0.0)
    R0, t0 = estimating_residuals(X[:, top], y, W0, b0, 0.0)
    assert np.abs(R0).max() < 1e-6 and np.abs(t0).max() < 1e-6 and np.abs(W0).max() < 50
    nb = fin_fit(FinConfig("nb", False))
    Rn, tn = estimating_residuals(X, y, nb["W"], nb["b"], 0.0)
    assert np.abs(tn).max() > 0.01 * len(y) and np.abs(Rn).max() > 100 * np.abs(R).max() + 1


def test_lemma1_concentration():
    """Lemma 1: at m_0 = ⌈ln(4V/δ)/(2ε²)⌉ documents per class (1,659 for V = 200, ε = 0.05, δ = 0.2), the share of
    reps with a parameter off by more than ε is at most δ — here 0, and the 1 − δ quantile of the largest deviation
    sits well inside ε (the union bound is loose)."""
    nj = ng_jordan_check()
    assert nj["m0"] == 1659
    assert nj["fail_rate"] <= BERN_DELTA and nj["fail_rate"] == 0.0
    assert nj["dev_quantile"] < 0.85 * BERN_EPS


def test_thm2_margin_bound():
    """Thm 2: on every rep, |l̂ − l_∞| ≤ τV pointwise, and every test document NB gets wrong while h_∞ gets it right
    has |l_∞| ≤ τV. The bound is sufficient, not tight: τV = 402 nats against a largest observed gap of 2.15, so
    G(τ) covers every correct document (vacuous at this m), and τV ≈ 1 would take 2.7·10⁸ documents per class."""
    nj = ng_jordan_check()
    assert nj["gaps_within_bound"] and nj["inclusion"]
    assert abs(nj["tauV"] - 402.0) < 1e-9 and nj["max_gap"] < nj["tauV"] / 100
    assert abs(nj["G"] - (1 - nj["bayes_err"])) < 1e-12
    assert nj["m0_for_unit_tau"] > 10 ** 8


def test_thm3_relative_accuracy_costs_one_over_theta_min():
    """Thm 3: (a) uniform relative accuracy needs N ≥ 3 ln(2V/δ)/(ε² θ_min) tokens per class, and θ_min ≤ 1/V makes
    that at least the uniform-vocabulary count; (b) the long-tailed world asks for 27.7M tokens where 4,096 documents
    supply 0.44M, and leaves ≈ 67 words unseen. (c) A Monte-Carlo check of (a)'s Chernoff step on a 50-word Zipf."""
    zt = zipf_world_tokens()
    assert zt["theta_min"] <= 1 / zt["V"] and zt["tokens_needed"] >= zt["uniform_tokens_needed"]
    assert zt["tokens_needed"] > 50 * zt["tokens_supplied"] and zt["unseen_words"] > 30
    assert zt["zipf_tokens_needed"] > 10 * zt["uniform_tokens_needed"]
    shares = list(zt["rare_share"].values())
    assert shares[0] > shares[1] > shares[2] > 0
    alpha = np.median(learning_curve(WorldConfig(WORLD_NOISE[-1], False))["nb_alpha"], axis=0)
    assert alpha[-1] > alpha[CURVE_N.index(64)]  # validated α ends far above where it starts (test_alpha_channels says why)
    p = zh.zipf_pmf(1.1, 50)
    N = math.ceil(thm3_tokens(0.5, 0.1, p[-1], 50))
    draws = _rng(3003).multinomial(N, p, size=200) / N
    assert np.mean(np.abs(draws / p - 1).max(1) > 0.5) <= 0.1


def test_bursty_world_keeps_lr_well_specified():
    """Prop 2: (a) the Bayes rule on bursty counts is linear, Σ_j (x_j/r_j) log(θ_1j/θ_0j), and has the plain world's
    accuracy on the same draws; (b) NB's population weights are r_j times the Bayes weight per underlying occurrence, plus
    a term in the length, and with bursts falling on half the words that costs ≥ 0.02 accuracy; in the NB-true world
    they are the Bayes weights."""
    for nw in WORLD_NOISE:
        plain, bursty = learning_curve(WorldConfig(nw, False)), learning_curve(WorldConfig(nw, True))
        assert plain["bayes"] == bursty["bayes"] == plain["nb_inf"]
        assert bursty["nb_inf"] < bursty["bayes"] - 0.02
        w = world(WorldConfig(nw, True))
        wb, _ = bayes_weights(w)
        wn, _ = nb_population_weights(w)
        Zk = (w.theta * w.rep).sum(1)
        assert np.abs(wn - math.log(Zk[0] / Zk[1]) - w.rep * wb).max() < 1e-12 and (w.rep > 1).sum() > 0


def test_generative_discriminative_regimes():
    """§3's learning curves. NB-true worlds: tuned NB is at or above tuned LR (within 0.01) at every n, so there is no
    crossover. Bursty worlds: NB leads at the smallest n, LR overtakes it for good at some n ≥ 256 and leads at 4,096,
    and NB never beats its own asymptote by more than 0.01; at V = 101 LR has passed NB's asymptote by n = 4,096."""
    for nw in WORLD_NOISE:
        c = learning_curve(WorldConfig(nw, False))
        assert crossover_n(c) is None and np.all(c["nb"].mean(0) >= c["lr"].mean(0) - 0.01)
        b = learning_curve(WorldConfig(nw, True))
        x = crossover_n(b)
        assert x is not None and x >= 256
        nb, lr = b["nb"].mean(0), b["lr"].mean(0)
        assert nb[0] > lr[0] and lr[-1] > nb[-1] and np.all(nb <= b["nb_inf"] + 0.01)
    small = learning_curve(WorldConfig(WORLD_NOISE[0], True))
    assert small["lr"].mean(0)[-1] > small["nb_inf"]


def test_separable_means_no_mle():
    """Thm 4(a)–(b): the 200 polar tone documents (V = 2,738) have full row rank, so every labeling is separable
    (checked by LP for the real labels and three random ones); along t·ŵ the logistic loss falls to 0 and is never 0,
    and the gradient's component along ŵ is negative everywhere (no stationary point)."""
    Z, s = tone_binary()
    X = sp.diags(s) @ Z
    assert np.linalg.matrix_rank(np.asarray(X.todense())) == Z.shape[0]
    rng = _rng(3004)
    for labels in [s] + [rng.choice([-1.0, 1.0], Z.shape[0]) for _ in range(3)]:
        A = -(sp.diags(labels) @ X)
        r = optimize.linprog(np.zeros(X.shape[1]), A_ub=A, b_ub=-np.ones(len(labels)), bounds=[(None, None)] * X.shape[1],
                             method="highs")
        assert r.status == 0
    sep = separation()
    assert sep["min_constraint"] > 1 - 1e-6
    w = hard_margin_svm(Z)
    losses = [float(np.logaddexp(0, -t * (Z @ w)).mean()) for t in (1, 2, 4, 8, 16)]
    assert all(a > b > 0 for a, b in zip(losses, losses[1:]))
    for p in (np.zeros(Z.shape[1]), w, 10 * w):
        grad = -(Z.T @ expit(-(Z @ p))) / Z.shape[0]
        assert grad @ w < 0


def test_l2_path_turns_toward_max_margin():
    """Thm 4(c): as λ ↓ 0, ‖ŵ_λ‖ grows without bound (≈ ln 10/γ* per decade at the end), and the direction's cosine to
    the SVM and its normalized margin rise monotonically (0.998 at λ = 10⁻⁷). Gradient descent's direction does the
    same, more slowly (Soudry et al.)."""
    sep = separation()
    path = sep["path"]
    assert all(b["norm"] > a["norm"] for a, b in zip(path, path[1:]))
    assert all(b["cos"] > a["cos"] for a, b in zip(path, path[1:])) and path[-1]["cos"] > 0.998
    assert all(b["margin"] > a["margin"] for a, b in zip(path, path[1:])) and path[-1]["margin"] < sep["svm_margin"]
    slope = (path[-1]["norm"] - path[-2]["norm"]) / (math.log(10) / sep["svm_margin"])
    assert 0.8 < slope < 1.2
    gd = sep["gd"]
    assert all(b["cos"] > a["cos"] for a, b in zip(gd, gd[1:])) and gd[-1]["cos"] > 0.98
    assert all(b["loss"] < a["loss"] for a, b in zip(gd, gd[1:]))


def test_nb_weighting_is_a_feature_ridge():
    """Prop 3(a): LR on r ⊙ x with penalty λ‖v‖² is LR on x with penalty λΣ_j w_j²/r_j² (w_j = r_j v_j): the same
    decision values to 1e-7, on the finance unigrams' polar documents. On the summed log loss that penalty is a Gaussian
    prior of variance r_j²/(nλ): the sum's gradient at the fit equals −w_j nλ/r_j²."""
    X = binarize(fin_features(False)[0])
    y = fin_split()[1]
    keep = y != 1
    Xp, yp = X[keep], (y[keep] == 2).astype(float)
    r = log_count_ratio(Xp, yp.astype(int))
    nz = np.abs(r) > 1e-9
    Xp, r = Xp[:, nz], r[nz]
    lam = 1e-2
    t1 = blr_intercept_fit(Xp @ sp.diags(r), yp, lam)
    t2 = blr_intercept_fit(Xp, yp, lam / r ** 2)
    assert np.abs(r * t1[:-1] - t2[:-1]).max() < 1e-6
    assert np.abs((Xp @ (r * t1[:-1]) + t1[-1]) - (Xp @ t2[:-1] + t2[-1])).max() < 1e-7
    s_ = 2 * yp - 1
    m = s_ * (Xp @ t2[:-1] + t2[-1])
    assert np.abs(-(Xp.T @ (s_ * expit(-m))) + t2[:-1] * Xp.shape[0] * lam / r ** 2).max() < 1e-6


def test_smoothing_bounds_log_count_ratios():
    """Prop 3(b): every word's |r_j| ≤ ln((c_j + α)/α) + |ln((N_1 + αV)/(N_0 + αV))|, at every α; a word seen once
    therefore has |r_j| ≤ ln(1 + 1/α) + const, so rare words' extreme weights shrink as α grows."""
    fu = funnel()
    assert fu["all_within_bound"]
    c = fu["count"]
    once = c == 1
    spread = [float(np.abs(fu["r"][a][once]).max()) for a in FUNNEL_ALPHAS]
    assert spread[0] > spread[1] > spread[2]


def test_aspect_xor():
    """Prop 4: the four one-clause documents are not linearly separable in unigram counts and are in bigrams; on the
    corpus, LR leaves the direction words at the noise level (|w| < 0.1: "rose" is good for sales and bad for costs) while
    the cost bigrams carry the sign."""
    assert not xor_separable(False) and xor_separable(True)
    for f in ("rose", "fell"):
        assert abs(fin_weight(FinConfig("lr", False), f)) < 0.1
    assert fin_weight(FinConfig("lr", True), "costs rose") < -0.2 < 0.2 < fin_weight(FinConfig("lr", True), "costs fell")
    assert fin_weight(FinConfig("lr", True), "not weak") > 0 > fin_weight(FinConfig("lr", True), "not strong")


def test_bayes_ceiling():
    """§6: the full-text Bayes accuracy is 0.889919 (enumeration over 5–8 aspect sentences), and a direct Monte Carlo
    of the generator's polarity draws agrees to 0.01."""
    acc = bayes_ceiling()
    assert abs(acc - 0.889919) < 1e-6
    mix = {t: dict(zip(*fc._SENTENCE_TONE_MIX[t])) for t in TONES}
    rule = {}  # the Bayes rule, tabulated once per (k, counts)
    for k in range(5, 9):
        for a in range(k + 1):
            for b in range(k + 1 - a):
                cnt = (a, b, k - a - b)
                rule[cnt] = TONES[int(np.argmax([stats.multinomial.pmf(cnt, k, [mix[t][1], mix[t][0], mix[t][-1]])
                                                 for t in TONES]))]
    rng = _rng(3005)
    hits = 0
    for _ in range(20000):
        t = TONES[rng.integers(3)]
        pols = rng.choice(fc._SENTENCE_TONE_MIX[t][0], size=int(rng.integers(5, 9)), p=fc._SENTENCE_TONE_MIX[t][1])
        hits += rule[(int(np.sum(pols == 1)), int(np.sum(pols == 0)), int(np.sum(pols == -1)))] == t
    assert abs(hits / 20000 - acc) < 0.01


def test_finance_tone():
    """§6: LR beats NB on unigrams and on bigrams; bigrams help every method; nothing here reaches the full-text ceiling.
    NB is overconfident (mean confidence − accuracy > 0.2, ECE₁₅ > 0.2). On this split LR is near-calibrated on unigrams
    (ECE₁₅ < 0.06) and overconfident on bigrams (confidence − accuracy > 0.05), which is the small λ the accuracy grid
    picked there, not the bigrams: test_penalty_sets_calibration."""
    acc = {(m, bg): fin_fit(FinConfig(m, bg))["acc"] for m in FIN_METHODS for bg in (False, True)}
    for bg in (False, True):
        assert acc[("lr", bg)] > acc[("nb", bg)] + 0.05
    for m in FIN_METHODS:
        assert acc[(m, True)] > acc[(m, False)]
    assert max(acc.values()) < bayes_ceiling() - 0.08
    for bg in (False, True):
        nb = fin_calibration(FinConfig("nb", bg))
        assert nb["conf"] - nb["acc"] > 0.2 and nb["ece"] > 0.2
    assert fin_calibration(FinConfig("lr", False))["ece"] < 0.06
    lb = fin_calibration(FinConfig("lr", True))
    assert lb["conf"] - lb["acc"] > 0.05 and len(fin_fit(FinConfig("lr", True))["voc"]) > len(fin_split()[1])


def test_finance_has_no_crossover():
    """§6: LR is ahead of NB at every training prefix from 30 to 1,500 documents, at the hyperparameters of the full-data
    fits and on the full-data vocabulary. test_prefix_tuned retunes at each n."""
    c = fin_prefix_curve()
    assert all(l > n for n, l in zip(c["nb"], c["lr"]))


def test_label_noise():
    """§6: training-label noise costs every method accuracy; LR stays above NB at every level."""
    for m in FIN_METHODS:
        a = [fin_fit(FinConfig(m, False, q))["acc"] for q in FIN_NOISE]
        assert a[0] > a[-1]
    for q in FIN_NOISE:
        assert fin_fit(FinConfig("lr", False, q))["acc"] > fin_fit(FinConfig("nb", False, q))["acc"]


def test_two_word_picture():
    """§Two ways: on (strong, weak) most documents of every tone are at the origin (956 of 1,500 test documents), and the
    tones differ only in their means."""
    p = two_word_picture()
    assert min(p["origin"]) > 0.5 and p["origin"][1] > 0.75 and p["median"] == [0.0, 0.0]
    assert p["test_origin"] == 956 and p["n_test"] == 1500
    assert p["mean"][2][0] > p["mean"][2][1] and p["mean"][0][1] > p["mean"][0][0]


def test_moment_gaps():
    """Thm 1 on the corpus, like against like. NB's raw misses are 22.3 class documents and 268 co-counts at α = 3, and
    smoothing makes most of that: 6.4 and 74.8 at α = 0.001. LR's raw co-count gap is 8.81, which is exactly nλW; its
    4.5e-8 is the residual of the penalized equations."""
    g = moment_gaps()
    assert abs(g["nb 3"]["class"].max() - 22.32) < 0.01 and abs(g["nb 3"]["cocount"] - 268.2) < 0.1
    assert abs(g["nb 0.001"]["class"].max() - 6.38) < 0.01 and abs(g["nb 0.001"]["cocount"] - 74.8) < 0.1
    lr = g["lr"]
    assert abs(lr["raw_max"] - 8.81) < 0.01 and abs(lr["raw_max"] - lr["n_lam_W"]) < 1e-6
    assert lr["residual"] < 1e-7 and lr["class"] < 1e-8


def test_thm2_slack():
    """Thm 2's slack sits mostly in the term sizes. Each log term is charged 2ε/ρ0 = 1 nat; real terms have median 0.013 and
    maximum 0.17. With real sizes and aligned signs a test document reaches 8.0 (×50 below 402), the worst constructible
    document 12.3, and with real signs 1.8 (a further ×4.4)."""
    t = thm2_slack()
    assert abs(t["aligned"] - 8.0) < 0.1 and abs(t["worst_doc"] - 12.3) < 0.1 and abs(t["signed"] - 1.82) < 0.05
    assert t["median_term"] < 0.02 and t["largest_term"] < 0.2 and t["per_term_bound"] == 1.0
    assert 402 / t["aligned"] > 45 and t["aligned"] / t["signed"] > 4


def test_one_bayes_accuracy():
    """§Burstiness: every world has one Bayes accuracy, 0.854 on 10⁶ documents. V doesn't enter (the informative words'
    rates are the same in all four, and noise words read 0), and bursts are a deterministic relabeling."""
    a, b = world(WorldConfig(30, False)), world(WorldConfig(3000, False))
    S = a.n_informative
    assert np.array_equal(a.theta[:, :S], b.theta[:, :S])
    t = population_table()
    assert abs(t["bayes"] - 0.854) < 0.001
    assert all(abs(t[k][0] - t["bayes"]) < 0.002 for k in t if k not in ("bayes", "every word, prior 0.3"))


def test_even_bursts_cost_no_asymptote():
    """Prop 2(c): if every word with θ_1j ≠ θ_0j repeats the same r times, Z_0 = Z_1 and NB's population weights are exactly
    r times the Bayes weights, whatever the other words do, so with π = 1/2 its rule is Bayes's. Every word, all
    informative words, and noise words only all qualify. An unbalanced prior breaks it: the score scales and the prior
    doesn't (0.873 against 0.863 at π = 0.3)."""
    for nw in WORLD_NOISE:
        for who, c in (("none", 1.0), ("every word", float(BURST)), ("all informative", float(BURST)), ("noise only", 1.0)):
            w = masked_world(nw, burst_mask(nw, who))
            assert np.abs(nb_population_weights(w)[0] - c * bayes_weights(w)[0]).max() < 1e-12
    t = population_table()
    assert all(abs(t[k][0] - t[k][1]) < 1e-4 for k in ("none", "every word", "all informative", "noise only"))
    bp, np_ = t["every word, prior 0.3"]
    assert bp - np_ > 0.008


def test_uneven_bursts_cost_the_asymptote():
    """Prop 2(b): half the informative words bursting, no noise words, costs NB 0.045 of asymptote; the page's half of all
    words costs 0.039, 0.815 in both worlds (the 3,000-document 0.820 and 0.815 are one number)."""
    t = population_table()
    assert abs(t["half informative"][0] - t["half informative"][1] - 0.045) < 0.003
    for k in ("page bursty V=101", "page bursty V=3071"):
        assert abs(t[k][1] - 0.815) < 0.002


def test_noise_bursts_cross_with_equal_asymptotes():
    """§Burstiness: the crossover doesn't need unequal asymptotes. With only noise words bursting NB's asymptote is the
    Bayes accuracy, and LR still beats it in all four draws from n = 256 (V = 101) and n = 1,024 (V = 3,071). Half the
    informative words, with the asymptote gap: from n = 1,024 and n = 4,096."""
    for nw, x in zip(WORLD_NOISE, (256, 1024)):
        c = burst_curve(nw, "noise only")
        assert abs(c["bayes"] - c["nb_inf"]) < 1e-9 and first_clean_sweep(c) == x
    for nw, x in zip(WORLD_NOISE, (1024, 4096)):
        c = burst_curve(nw, "half informative")
        assert c["bayes"] - c["nb_inf"] > 0.04 and first_clean_sweep(c) == x


def test_a_burst_divides_lr_weights():
    """Why noise bursts hurt NB and not LR. Prop 3(a) with r = 4: refit, and LR's weight on a word repeated four times
    divides by about 4 (3.73, 3.96, 3.99, 4.00 as λ falls, exactly 4 at 0), so that word's part of the score barely moves;
    NB's weights can't see r, and its noise part grows fourfold."""
    q = burst_weight_ratio()
    assert np.all(np.diff(list(q.values())) > 0) and 3.7 < q[0.1] and abs(q[1e-5] - 4) < 1e-3
    sp_ = noise_score_spread()
    assert sp_["nb"][1] / sp_["nb"][0] > 3.9 and sp_["lr"][1] / sp_["lr"][0] < 1.1
    assert 3.5 < sp_["lr_w"][0] / sp_["lr_w"][1] < 4.1


def test_once_seen_words_peak():
    """§Zipf tail: words seen once don't grow with n. Pooled they peak near n = 256 (about 650) and are down to about 30 at
    4,096; per class they peak near 1,024 (about 600) and are down to 160, where α is still rising."""
    t = tail_draws()
    pooled = {n: float(np.mean(v)) for n, v in t["pooled"].items()}
    assert max(pooled, key=pooled.get) == 256 and pooled[256] > 600 and pooled[4096] < 40
    assert max(t["per_class"], key=t["per_class"].get) == 1024 and t["per_class"][4096] < 170


def test_alpha_channels():
    """§Zipf tail: why validated α rises. α = 10 against α = 1 costs through unequal class token totals (−0.103 at n = 256,
    −0.002 at 4,096) and through flattening the informative words (−0.052, −0.006), and pays through shrinking noise words
    (+0.063, +0.024). The pay falls with n; the costs fall faster, as αV pseudo-tokens shrink beside the class's own."""
    e = tail_draws()["effects"]
    assert e[256][0] < -0.09 and abs(e[4096][0]) < 0.01
    assert e[256][1] < e[4096][1] < 0 and all(e[n][2] > 0.02 for n in e)
    c = learning_curve(WorldConfig(WORLD_NOISE[-1], False))
    alpha = np.median(c["nb_alpha"], axis=0)
    assert alpha[0] < 1 and alpha[-1] == 20 and np.any(np.diff(alpha) < 0)  # rising, not monotone


def test_hessian_needs_no_span():
    """Thm 1(a): at λ > 0 the null space is the one intercept shift whether or not the x̃_i span R^{V+1}: here n < V + 1."""
    X, y, rng = _small_problem(3006, n=5, V=6)
    th = rng.normal(scale=0.3, size=3 * 6 + 3)
    H = lr_hessian(th[:18].reshape(3, 6), th[18:], X, 0.05)
    assert int(np.sum(np.linalg.eigvalsh(H) < 1e-10)) == 1


def test_corpus_quotes():
    """The quotes on the page. "margins improved" is not in the corpus; the contrast sentence is, in the test split.
    "revenue rose" occurs 71 times, 45 of them inside "cost of revenue rose", which is why its weight is 0.001 where "sales
    rose" gets 0.265. "impairment" doesn't burst here: 30 occurrences in 29 documents."""
    c = phrase_counts(("revenue rose", "cost of revenue rose", "margins improved", "impairment"))
    assert c["margins improved"] == (0, 0) and c["impairment"] == (30, 29)
    assert c["revenue rose"][0] == 71 and c["cost of revenue rose"][0] == 45
    assert any(CONTRAST_QUOTE in d.text for d in fc.generate_corpus(*FIN_TEST))
    lr = FinConfig("lr", True)
    assert abs(fin_weight(lr, "revenue rose")) < 0.01 and abs(fin_weight(lr, "sales rose") - 0.265) < 0.001


def test_direction_weights():
    """The direction words sit at the noise level rather than at zero: |"rose"| = 0.060 is in the top 26% of all unigram
    weights (SD 0.092, median magnitude 0.031), and the six words' signs follow no pattern."""
    d = direction_weights()
    assert abs(d["words"]["rose"] - 0.060) < 0.001 and all(abs(v) < 0.07 for v in d["words"].values())
    assert 0.25 < d["rose_top_share"] < 0.27 and abs(d["sd"] - 0.092) < 0.001 and abs(d["median_abs"] - 0.031) < 0.001


def test_exact_readers():
    """§Finance: the ceiling, taken apart exactly. A reader blind only to which clause of a contrast sentence counts has
    0.8733, 0.017 below 0.8899; one who also sees a verb only as its word, 0.8613; one who further can't read a bare
    direction word and takes "not" + adjective at face value, 0.8160. The sentence census on the test corpus matches the
    generator's form probabilities these readers use."""
    assert abs(reader_accuracy(blind_contrast=False) - bayes_ceiling()) < 1e-12
    a, b, c = reader_accuracy(), reader_accuracy(verb_surface=True), reader_accuracy(verb_surface=True, surface_sign=True)
    assert abs(a - 0.8733) < 1e-4 and abs(b - 0.8613) < 1e-4 and abs(c - 0.8160) < 1e-4
    assert abs(bayes_ceiling() - a - 0.0166) < 1e-4
    t = aspect_census()
    assert t["n"] == 9695 and abs(t["contrast"] - 0.130) < 0.001 and abs(t["neutral"] - 0.337) < 0.001
    assert abs(t["lexicon agrees"] - 0.405) < 0.001 and abs(t["direction only"] - 0.061) < 0.001
    assert abs(t["verb trap"] - 0.036) < 0.001 and abs(t["not"] - 0.031) < 0.001


def test_xor_is_under_a_third_of_the_bigram_gain():
    """§Finance: Prop 4's 84 aspect–direction bigrams give LR +0.012 of the +0.039 all bigrams give, six splits, paired;
    with the 16 "not" bigrams, +0.021."""
    g = fin_bigram_gain()
    assert g["counts"][0] == (84, 16) and abs(g["xor"] - 0.012) < 0.001 and abs(g["xor_not"] - 0.021) < 0.001
    assert abs(g["all"] - 0.039) < 0.001 and g["xor"] < g["all"] / 3 and np.all(g["by_split"][:, 2] > 0)


def test_penalty_sets_calibration():
    """§Finance: LR's calibration follows λ, not the features, and not Thm 1's class-total equation. The unigram count
    matrix has rank 1,500, so unigrams are exactly separable too, and training accuracy reaches 1 by λ = 0.001. Class totals
    hold at every λ, including λ = 0.001 where unigram ECE is 0.147. At equal λ ≤ 0.01 bigrams are the better calibrated.
    The accuracy rule picked λ = 0.001 on bigrams in 2 of 6 splits, the page's among them; choosing by dev log loss gives
    ECE 0.033 on both feature sets at no accuracy cost."""
    X = fin_features(False)[0]
    assert np.linalg.matrix_rank(X.toarray()) == 1500
    t = fin_penalty_table()
    u, b = t["uni"], t["bi"]
    assert u["by"][1e-3]["train_acc"] == 1.0 and b["by"][1e-2]["train_acc"] > 0.999
    assert u["by"][1e-3]["class_miss"] < 1e-4 and abs(u["by"][1e-3]["ece"] - 0.147) < 0.001
    assert all(b["by"][lam]["ece"] < u["by"][lam]["ece"] for lam in (0.01, 0.003, 0.001))
    assert b["chosen"].count(1e-3) == 2 and b["chosen"][0] == 1e-3 and set(b["chosen"]) == {1e-3, 1e-2}
    assert abs(u["ll_ece"] - 0.033) < 0.001 and abs(b["ll_ece"] - 0.033) < 0.001
    assert u["ll_acc"] >= u["chosen_acc"] and b["ll_acc"] > b["chosen_acc"] - 0.002


def test_grid_edges():
    """§Finance: the table grids stop at what they select. α = 3 is the top of FIN_ALPHAS and wins on all six splits; past
    it dev picks 10–100, and six-split NB goes from 0.607 to 0.617. λ = 0.01 is the top of FIN_LAMBDAS and LR's choice on
    unigrams, which a wide grid keeps; NB-weighted unigrams move down, to 0.709."""
    e = fin_grid_edges()
    assert set(e["nb_alphas"]) == {FIN_ALPHAS[-1]} and all(a >= 10 for a in e["nb_wide_alphas"])
    assert abs(np.mean(e["nb_acc"]) - 0.607) < 0.001 and abs(np.mean(e["nb_wide_acc"]) - 0.617) < 0.001
    w = e["wide"]
    assert w["nb-uni"]["hp"] == 100 and abs(w["nb-uni"]["acc"] - 0.6147) < 1e-3 and abs(w["nb-uni"]["ece"] - 0.203) < 1e-3
    assert fin_fit(FinConfig("lr", False))["hp"] == FIN_LAMBDAS[0] and w["lr-uni"]["hp"] == FIN_LAMBDAS[0]
    assert w["lr-bi"]["hp"] == 0.003 and abs(w["lr-bi"]["ece"] - 0.073) < 1e-3 and abs(w["nbw-uni"]["acc"] - 0.709) < 1e-3


def test_noise_streams():
    """§Finance: the noise table's stream drew 16.9% and 31.1% flips; over six streams NB is 0.569 and 0.533, not 0.526 and
    0.510. LR's tuned λ rises with noise, past the table grid's 0.01, and on the wide grid LR and NB-weighted tie within
    their spread. LR stays ahead of NB at every level."""
    n = fin_noise_streams()
    assert abs(n[0.15]["flips"][0] - 0.169) < 0.001 and abs(n[0.3]["flips"][0] - 0.311) < 0.001
    assert abs(n[0.15]["nb"] - 0.569) < 0.001 and abs(n[0.3]["nb"] - 0.533) < 0.001
    assert abs(n[0.15]["lr"] - 0.674) < 0.001 and abs(n[0.3]["lr"] - 0.594) < 0.001
    assert abs(n[0.15]["lr_wide"] - 0.690) < 0.001 and abs(n[0.3]["lr_wide"] - 0.637) < 0.001
    for q in n:
        assert min(n[q]["lams"]) >= 0.03 and n[q]["lr"] > n[q]["nb"] and n[q]["lr_wide"] > n[q]["lr"]
        assert abs(n[q]["lr_wide"] - n[q]["nbw_wide"]) < n[q]["lr_wide_sd"] + n[q]["nbw_wide_sd"]


def test_prefix_tuned():
    """§Finance: retuned at each n on that n's own vocabulary, LR wins all six splits from n = 60, and at n = 8 it is a coin
    flip (2 of the 4 splits with all three tones). On 106 tone, aspect, and neutral words NB leads every split through
    n = 300: the early lead theory allows, which the noise words take away."""
    p = fin_prefix_tuned()
    assert all(p[("own", n)]["wins"] == 6 for n in (60, 100, 300))
    assert p[("own", 8)]["wins"] == 2 and p[("own", 8)]["splits"] == 4
    assert all(p[("clean", n)]["wins"] == 0 for n in PREFIX_TUNED)


def test_vocabulary_cuts():
    """§Finance: LR doesn't fully shrink the noise words: on the 50 best words by chi-squared it scores 0.777 against 0.734
    on all 3,246, and the NB–LR gap falls from 0.117 to 0.022. NB on the 99 tone and aspect words is calibrated (ECE
    0.027, 0.250 on all words)."""
    v = fin_vocab_cuts()
    assert abs(v["lr top50"]["acc"] - 0.777) < 0.001 and abs(v["lr all"]["acc"] - 0.734) < 0.001
    assert abs(v["nb top50"]["acc"] - 0.755) < 0.001 and abs(v["nb all"]["acc"] - 0.617) < 0.001
    assert abs(v["nb 99"]["ece"] - 0.027) < 0.001 and abs(v["nb 99"]["acc"] - 0.705) < 0.001
    assert v["nb 99"]["ece"] < v["nb 99+3"]["ece"] < v["nb 99+7"]["ece"] < v["nb all"]["ece"]


def test_noise_words_share_the_overconfidence():
    """§Finance: co-occurrence does explain NB's overconfidence, and about half of what co-occurs is content words. NB at
    α = 3, six splits: ECE 0.297 on real text, 0.191 with content words redrawn independently, 0.191 with template words
    redrawn, 0.058 with both."""
    r = fin_redraws()
    assert abs(r["real"]["ece"] - 0.297) < 0.001 and abs(r["both iid"]["ece"] - 0.058) < 0.001
    assert abs(r["content iid"]["ece"] - 0.191) < 0.001 and abs(r["templates iid"]["ece"] - 0.191) < 0.001


def test_clustered_noise_is_most_of_the_gap():
    """§Finance: the NB–LR gap is mostly estimation among clustered content words, not a far-off asymptote. Four splits,
    tuned: 0.116 on real text, 0.041 with content words redrawn, and reversed (−0.020) with everything redrawn."""
    r = fin_redraws()
    gap = {t: r[t]["lr"] - r[t]["nb"] for t in r}
    assert abs(gap["real"] - 0.116) < 0.001 and abs(gap["content iid"] - 0.041) < 0.001
    assert abs(gap["both iid"] + 0.020) < 0.002


def slowcheck_training_sizes():
    """--slow, §Finance: most of the distance to the ceiling was data. Unigrams level off near 0.80; bigrams reach 0.847 at
    50,000 documents, within 0.043 of the ceiling and 0.026 of the contrast-blind reader; NB on unigrams is still climbing."""
    t = training_sizes()
    assert abs(t[50000]["bi"][1] - 0.847) < 0.001 and abs(t[15000]["bi"][1] - 0.826) < 0.001
    assert abs(t[50000]["all"][1] - 0.792) < 0.002 and abs(t[50000]["top200"][1] - 0.798) < 0.002
    assert t[1500]["top200"][1] > t[1500]["bi"][1]
    assert t[1500]["nb"][2] < t[15000]["nb"][2] < t[50000]["nb"][2] < t[50000]["all"][1]
    assert t[50000]["nb"][2] - t[15000]["nb"][2] > 0.03


def slowcheck_noise_bursts_past_4096():
    """--slow, §Burstiness: past 4,096 with noise-only bursts. At V = 101 the curves meet by n = 32,768, as equal asymptotes
    predict; at V = 3,071 LR still wins all four draws there."""
    a, b = (burst_curve(nw, "noise only", BURST_N + BURST_N_SLOW) for nw in WORLD_NOISE)
    assert a["wins"][-2] == 4 and a["wins"][-1] < 4
    assert b["wins"][-1] == 4


def test_viz_payload():
    """The payload is JSON-clean, rounded, and carries what the page narrates."""
    p = viz_payload()
    s = json.dumps(p)
    assert "NaN" not in s and "Infinity" not in s
    assert p["finance"]["ceiling"] == _r(bayes_ceiling())
    assert len(p["curves"]["worlds"]) == 2 * len(WORLD_NOISE)
    assert all(w["crossover"] is None for w in p["curves"]["worlds"] if not w["bursty"])
    assert all(w["crossover"] is not None for w in p["curves"]["worlds"] if w["bursty"])
    assert p["separation"]["path"][-1]["cos"] == _r(separation()["path"][-1]["cos"])


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


def _run_harness(slow: bool = False) -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    if slow:
        tests += [(name, fn) for name, fn in globals().items() if name.startswith("slowcheck_") and callable(fn)]
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
    if slow:
        for n, row in training_sizes().items():
            print(f"  n={n}: LR all / top 200 / bigrams (λ, acc, ECE) {row['all']} {row['top200']} {row['bi']}; "
                  f"NB (α, conf, acc, ECE) {row['nb']}; V {row['V']}, {row['V_bi']}")
        for nw in WORLD_NOISE:
            c = burst_curve(nw, "noise only", BURST_N + BURST_N_SLOW)
            print(f"  noise-only bursts V={nw + 71}: NB {c['nb'].mean(0).round(3)} LR {c['lr'].mean(0).round(3)} "
                  f"LR wins {c['wins']}")
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
    raise SystemExit(_run_harness("--slow" in sys.argv))
