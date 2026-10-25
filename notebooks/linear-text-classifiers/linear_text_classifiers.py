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
   Burstiness keeps LR well specified while NB's population weights overcount repeated words (Prop 2): the
   generative–discriminative crossover, measured in worlds whose Bayes error is known.
4. Separation. With V ≫ n the training documents are linearly separable, the logistic MLE does not exist, and the
   L2 path's direction converges to the max-margin separator (Thm 4); gradient descent's does too (Soudry et al.).
5. Priors on features. NB-weighted LR is LR with a per-feature ridge λ Σ v_j²/r_j², and add-α smoothing bounds a
   word's log-count ratio by log((c_j + α)/α) plus a constant (Prop 3).
6. Finance case study. Tone is an XOR of aspect valence and direction ("costs rose" is bad news): no unigram-linear
   classifier represents it, and bigrams do (Prop 4). NB, LR, and NB-weighted LR on unigrams and bigrams, against the
   full-text Bayes accuracy, with label noise and calibration.

Cross-checks: scikit-learn's MultinomialNB, BernoulliNB, and LogisticRegression; finite differences for every
gradient and Hessian; scipy.optimize.linprog for (non)separability; exact enumeration for the Bayes ceiling.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python notebooks/linear-text-classifiers/linear_text_classifiers.py
"""

from __future__ import annotations

import json
import math
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
FIN_ALPHAS = (0.1, 0.3, 1.0, 3.0)
FIN_LAMBDAS = (1e-2, 1e-3)
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


@lru_cache(maxsize=None)
def fin_fit(cfg: FinConfig) -> dict:
    """Fit on the (noisy) training labels, pick the hyperparameter on the clean dev set, score the clean test set:
    accuracy, the confusion matrix, test probabilities (for calibration), and the fitted weights."""
    X, Xd, Xt, voc = fin_features(cfg.bigrams)
    _, _, _, ydv, _, yte = fin_split()
    ytr = fin_train_labels(cfg.noise)
    best = None
    if cfg.method == "nb":
        for a in FIN_ALPHAS:
            W, b = nb_fit(X, ytr, 3, a)
            dev = float((log_posterior(Xd, W, b).argmax(1) == ydv).mean())
            if best is None or dev > best[0]:
                best = (dev, a, W, b)
        dev, hp, W, b = best
        lp = log_posterior(Xt, W, b)
    elif cfg.method == "lr":
        x0 = None
        for lam in FIN_LAMBDAS:
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
        for lam in FIN_LAMBDAS:
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
    out = {"V": c["V"], "bursty": cfg.bursty, "bayes": _r(c["bayes"]), "nb_inf": _r(c["nb_inf"]),
           "crossover": crossover_n(c)}
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
    has |l_∞| ≤ τV. The bound is sufficient, not tight: τV = 402 nats against a largest observed gap of 2.36, so
    G(τ) covers every correct document (vacuous at this m), and τV ≈ 1 would take 2.7·10⁸ documents per class."""
    nj = ng_jordan_check()
    assert nj["gaps_within_bound"] and nj["inclusion"]
    assert abs(nj["tauV"] - 402.0) < 1e-9 and nj["max_gap"] < nj["tauV"] / 100
    assert abs(nj["G"] - (1 - nj["bayes_err"])) < 1e-12
    assert nj["m0_for_unit_tau"] > 10 ** 8


def test_thm3_relative_accuracy_costs_one_over_theta_min():
    """Thm 3: (a) uniform relative accuracy needs N ≥ 3 ln(2V/δ)/(ε² θ_min) tokens per class, and θ_min ≤ 1/V makes
    that at least the uniform-vocabulary count; (b) the long-tailed world asks for 27.7M tokens where 4,096 documents
    supply 0.44M, and leaves ≈ 64 words unseen. (c) A Monte-Carlo check of (a)'s Chernoff step on a 50-word Zipf."""
    zt = zipf_world_tokens()
    assert zt["theta_min"] <= 1 / zt["V"] and zt["tokens_needed"] >= zt["uniform_tokens_needed"]
    assert zt["tokens_needed"] > 50 * zt["tokens_supplied"] and zt["unseen_words"] > 30
    assert zt["zipf_tokens_needed"] > 10 * zt["uniform_tokens_needed"]
    shares = list(zt["rare_share"].values())
    assert shares[0] > shares[1] > shares[2] > 0
    alpha = np.median(learning_curve(WorldConfig(WORLD_NOISE[-1], False))["nb_alpha"], axis=0)
    assert alpha[-1] > alpha[CURVE_N.index(64)]  # the long tail wants more smoothing as n grows
    p = zh.zipf_pmf(1.1, 50)
    N = math.ceil(thm3_tokens(0.5, 0.1, p[-1], 50))
    draws = _rng(3003).multinomial(N, p, size=200) / N
    assert np.mean(np.abs(draws / p - 1).max(1) > 0.5) <= 0.1


def test_bursty_world_keeps_lr_well_specified():
    """Prop 2: (a) the Bayes rule on bursty counts is linear, Σ_j (x_j/r_j) log(θ_1j/θ_0j), and has the plain world's
    accuracy on the same draws; (b) NB's population weights are r_j times too large on bursty words, costing ≥ 0.02
    accuracy; in the NB-true world they are the Bayes weights."""
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
    decision values to 1e-7, on the finance unigrams' polar documents."""
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
    corpus, LR gives the direction words near-zero unigram weight (|w| < 0.1, "rose" is good for revenue and bad for
    costs) while the cost bigrams carry the sign."""
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
    """§6: LR beats NB on unigrams and on bigrams; bigrams help every method; nothing reaches the full-text ceiling.
    NB is overconfident (mean confidence − accuracy > 0.2, ECE₁₅ > 0.2). LR on unigrams is near-calibrated (ECE₁₅ <
    0.06) but on bigrams, with more features than documents, it is overconfident (confidence − accuracy > 0.05):
    Thm 1 matches training moments, and a nearly separable training set (Thm 4) makes those moments optimistic."""
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
    """§6: on the finance corpus LR is ahead of NB at every training prefix from 30 to 1,500 documents — NB is
    misspecified here and its asymptote is far below LR's."""
    c = fin_prefix_curve()
    assert all(l > n for n, l in zip(c["nb"], c["lr"]))


def test_label_noise():
    """§6: training-label noise costs every method accuracy; LR stays above NB at every level."""
    for m in FIN_METHODS:
        a = [fin_fit(FinConfig(m, False, q))["acc"] for q in FIN_NOISE]
        assert a[0] > a[-1]
    for q in FIN_NOISE:
        assert fin_fit(FinConfig("lr", False, q))["acc"] > fin_fit(FinConfig("nb", False, q))["acc"]


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
