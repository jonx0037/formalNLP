"""Calibration of language models — the reference implementation for formalNLP's sixteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. What calibration means for a next-token distribution. Canonical calibration (P(Y | f(X)) = f(X)) implies top-label
   calibration, not conversely; a forecaster whose rows are all distinct is canonically calibrated only if it equals
   the true conditional (Prop 1). Every log loss and Brier score splits into an irreducible term, a refinement gap,
   and a reliability term, and the recalibrated forecaster removes exactly the last (Thm 1).
2. Measuring it. Binned ECE under-reads the true top-label error (binning hides, Jensen across a bin) and the plug-in
   estimator over-reads it (sampling inflates, Jensen within a bin); the squared estimator has an exact correction
   (Thm 2). A calibrated twin (correctness redrawn from the model's own confidences) measures the noise floor.
3. Temperature scaling. The held-out NLL is convex in β = 1/τ with derivative E[E_{p_β} z − z_y], so the optimum is
   unique and matches moments; it keeps the argmax but not the ranking of confidences across rows (Prop 2).
4. Why τ = 1 is special. In a family closed under scaling the logits, a stationary point of the penalized training
   loss has d/dβ L_train = −2λΩ(θ) at β = 1: an unpenalized MLE sits at τ*_train = 1 exactly, an L2-penalized one
   below 1 (Thm 3). Topic 14's decoder has no such direction and lands at 0.94.
5. Prediction sets. If f is canonically calibrated, any set chosen from f(X) covers Y with probability equal to its
   expected kept mass, so a nucleus of mass p covers at least p (Prop 3); sampling at τ ≠ 1 decalibrates a calibrated
   model (Cor 2).
6. The corpus. Topic 3's KN trigram and Topic 14's decoder on three test sets: the same generator run on
   (in distribution), Topic 3's held-out split (a fresh run with new names), and real EDGAR filings.
7. Finance case study. On EDGAR both engines are overconfident and the nucleus covers a third of the text; Topic 15's
   fact model is underconfident on facts it has seen and overconfident on facts it has not, so no one temperature
   calibrates it.

Cross-checks: exact enumeration on random finite worlds for every identity; Poisson-binomial distributions for the
estimator's expectation; scipy.optimize for every temperature fit; Topic 15's truncation_set for the nucleus.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/language-model-calibration/language_model_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "sampling-temperature-nucleus"))
import sampling_temperature_nucleus as st  # noqa: E402 — a module object: its test_* stay out of this harness

bs, ng, pce, fc, ak = st.bs, st.ng, st.pce, st.fc, st.ak
sl = bs.sl

# ONE stream per random-instance claim and per sampled grid, each from default_rng([SEED, k]); k ≥ 1900 keeps this
# module's streams apart from Topics 5–15.
SEED = st.SEED
BOS, EOS, UNK = st.BOS, st.EOS, st.UNK


def _r(x, nd: int = 3):
    return ak._r(x, nd)


def _xlogy_ratio(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Σ_v p_v log(p_v/q_v) row-wise, with 0·log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return t.sum(axis=-1)


def _entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)


# --------------------------------------------------------------------------------------------
# 1. Calibration in a finite world: definitions and the decomposition.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """A finite world: contexts x with probabilities w[x], the true next-token conditionals P[x], and a forecaster's
    rows F[x] (all rows on the simplex Δ^V). Every population quantity below is an exact finite sum."""

    w: np.ndarray  # (n,)
    P: np.ndarray  # (n, V)
    F: np.ndarray  # (n, V)


def level_sets(F: np.ndarray) -> np.ndarray:
    """The level-set label of each context: contexts with identical forecasts share one."""
    return np.unique(F, axis=0, return_inverse=True)[1].ravel()


def recalibrated(world: World) -> np.ndarray:
    """C[x] = P(Y | f(X) = F[x]): the true conditional averaged over the level set of F[x]."""
    g = level_sets(world.F)
    C = np.zeros_like(world.P)
    for k in np.unique(g):
        m = g == k
        C[m] = (world.w[m] @ world.P[m]) / world.w[m].sum()
    return C


def log_loss(world: World, F: np.ndarray | None = None) -> float:
    F = world.F if F is None else F
    with np.errstate(divide="ignore"):
        return float(world.w @ -np.where(world.P > 0, world.P * np.log(F), 0.0).sum(axis=1))


def brier(world: World, F: np.ndarray | None = None) -> float:
    """E‖F(X) − e_Y‖² = E[‖F‖² − 2⟨P, F⟩ + 1]."""
    F = world.F if F is None else F
    return float(world.w @ ((F**2).sum(axis=1) - 2 * (world.P * F).sum(axis=1) + 1))


def log_decomposition(world: World) -> dict:
    """Thm 1 (log loss): E[−log F_Y] = E H(P) + E KL(P ‖ C) + E KL(C ‖ F), every term exact."""
    C, w = recalibrated(world), world.w
    return {"loss": log_loss(world), "irreducible": float(w @ _entropy(world.P)),
            "refinement_gap": float(w @ _xlogy_ratio(world.P, C)), "reliability": float(w @ _xlogy_ratio(C, world.F)),
            "recalibrated_loss": log_loss(world, C)}


def brier_decomposition(world: World) -> dict:
    """Thm 1 (Brier): E‖F − e_Y‖² = E‖P − e_Y‖² + E‖C − P‖² + E‖F − C‖² (Pythagoras, twice)."""
    C, w = recalibrated(world), world.w
    return {"loss": brier(world), "irreducible": float(w @ (1 - (world.P**2).sum(axis=1))),
            "refinement_gap": float(w @ ((C - world.P) ** 2).sum(axis=1)),
            "reliability": float(w @ ((world.F - C) ** 2).sum(axis=1)), "recalibrated_loss": brier(world, C)}


def top_label(world: World) -> tuple[np.ndarray, np.ndarray]:
    """(conf(x) = max F[x], r(x) = P(Y = argmax F[x] | x)); ties in the argmax go to the lower token id."""
    a = world.F.argmax(axis=1)
    return world.F.max(axis=1), world.P[np.arange(len(a)), a]


def top_label_ece(world: World) -> float:
    """The true top-label calibration error E|P(correct | conf) − conf|, conditioning on each distinct confidence."""
    conf, r = top_label(world)
    tot = 0.0
    for c in np.unique(conf):
        m = conf == c
        tot += abs(world.w[m] @ (r[m] - c))
    return float(tot)


def binned_population_ece(world: World, B: int) -> float:
    """The population ECE of B equal-width confidence bins: Σ_b P(b)·|E[r − conf | b]|."""
    conf, r = top_label(world)
    b = bin_index(conf, B)
    return float(sum(abs(world.w[b == k] @ (r[b == k] - conf[b == k])) for k in np.unique(b)))


def random_world(rng: np.random.Generator, n: int = 40, V: int = 5, pool: int = 8) -> World:
    """Contexts share forecasts from a pool of `pool` rows, so level sets are nontrivial; the truth is free."""
    rows = rng.dirichlet(np.full(V, 0.6), size=pool)
    return World(rng.dirichlet(np.ones(n)), rng.dirichlet(np.full(V, 0.6), size=n), rows[rng.integers(0, pool, n)])


def calibrated_world(rng: np.random.Generator, **kw) -> World:
    """A random world whose forecaster is replaced by its own recalibration: canonically calibrated by construction."""
    w = random_world(rng, **kw)
    return World(w.w, w.P, recalibrated(w))


def counterexample_world() -> World:
    """Prop 1(b): one context, truth (0.5, 0.2, 0.3), forecast (0.5, 0.3, 0.2) — top-label calibrated (it is right
    with probability 0.5 and says 0.5), canonically miscalibrated."""
    return World(np.array([1.0]), np.array([[0.5, 0.2, 0.3]]), np.array([[0.5, 0.3, 0.2]]))


def temper_rows(F: np.ndarray, tau: float) -> np.ndarray:
    """Topic 15's temper on probability rows (zero stays zero)."""
    with np.errstate(divide="ignore"):
        return np.exp(st.temper(np.log(F), tau))


# --------------------------------------------------------------------------------------------
# 2. Measuring calibration: the binned estimator and its two biases.
# --------------------------------------------------------------------------------------------


def bin_index(conf: np.ndarray, B: int) -> np.ndarray:
    """Equal-width bins on [0, 1]: bin k holds conf ∈ [k/B, (k+1)/B), and conf = 1 joins the last bin."""
    return np.minimum((np.asarray(conf) * B).astype(int), B - 1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(count, mean confidence, accuracy) of every nonempty bin, in bin order."""
    b = bin_index(conf, B)
    n = np.bincount(b, minlength=B)
    s_conf = np.bincount(b, weights=conf, minlength=B)
    s_corr = np.bincount(b, weights=correct.astype(float), minlength=B)
    k = n > 0
    return n[k], s_conf[k] / n[k], s_corr[k] / n[k]


def ece_hat(conf: np.ndarray, correct: np.ndarray, B: int) -> float:
    """The plug-in binned ECE: Σ_b (n_b/n)·|ā_b − c̄_b|."""
    n, c, a = reliability_bins(conf, correct, B)
    return float(n @ np.abs(a - c) / n.sum())


def sq_ece_debiased(conf: np.ndarray, correct: np.ndarray, B: int) -> float:
    """Thm 2(c)'s estimator of the squared binned error: Σ_b (n_b/n)[(ā_b − c̄_b)² − ā_b(1 − ā_b)/(n_b − 1)], bins
    with n_b ≥ 2 (a one-token bin has no variance estimate and is dropped from both sums)."""
    n, c, a = reliability_bins(conf, correct, B)
    k = n >= 2
    return float(n[k] @ ((a[k] - c[k]) ** 2 - a[k] * (1 - a[k]) / (n[k] - 1)) / n.sum())


def poisson_binomial(c: np.ndarray) -> np.ndarray:
    """The exact distribution of S = Σ Bernoulli(c_i), by the convolution recursion."""
    pmf = np.array([1.0])
    for ci in c:
        pmf = np.append(pmf * (1 - ci), 0.0) + np.append(0.0, pmf * ci)
    return pmf


def bin_expectations(c: np.ndarray, p: np.ndarray) -> dict:
    """One bin of n tokens whose correctness is Bernoulli(c_i) and whose confidences are p_i: the exact expectations
    of the plug-in gap |ā − p̄|, the plug-in square (ā − p̄)², and the debiased square, from S's distribution."""
    n, pmf = len(c), poisson_binomial(c)
    a = np.arange(n + 1) / n
    pbar = float(np.mean(p))
    return {"abs": float(pmf @ np.abs(a - pbar)), "sq": float(pmf @ (a - pbar) ** 2),
            "debiased": float(pmf @ ((a - pbar) ** 2 - a * (1 - a) / (n - 1))),
            "true_gap": abs(float(np.mean(c)) - pbar), "spread": float(np.sum((c - c.mean()) ** 2))}


def twin_ece(conf: np.ndarray, Bs, R: int, rng: np.random.Generator) -> np.ndarray:
    """(R, len(Bs)) plug-in ECEs of R calibrated twins: the same confidences, correctness redrawn as Bernoulli(conf).
    A twin is calibrated by construction, so its ECE is pure estimator noise."""
    draws = rng.random((R, len(conf))) < conf[None, :]
    out = np.zeros((R, len(Bs)))
    for j, B in enumerate(Bs):
        b = bin_index(conf, B)
        order = np.argsort(b, kind="stable")
        bs_, starts = np.unique(b[order], return_index=True)
        n = np.diff(np.append(starts, len(b)))
        c = np.add.reduceat(conf[order], starts) / n
        a = np.add.reduceat(draws[:, order].astype(float), starts, axis=1) / n
        out[:, j] = np.abs(a - c) @ n / len(conf)
    return out


# --------------------------------------------------------------------------------------------
# 3–4. Temperature scaling.
# --------------------------------------------------------------------------------------------


def _moments(Z: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise (log Σ e^{βz}, E_{p_β} z, Var_{p_β} z) for logit rows Z (−inf entries are outside the support)."""
    x = beta * Z
    L = st._lse(x, axis=1)
    p = np.exp(x - L[:, None])
    zf = np.where(np.isfinite(Z), Z, 0.0)
    m1 = (p * zf).sum(axis=1)
    return L, m1, (p * (zf - m1[:, None]) ** 2).sum(axis=1)


def temperature_objective(Z: np.ndarray, n_u: np.ndarray, s_y: float, beta: float) -> tuple[float, float, float]:
    """Prop 2: (L(β), L'(β), L''(β)) for the mean NLL of tempered rows, L(β) = Σ_u n_u log Σ_v e^{βz_uv} − β·Σ z_y, over
    N = Σ n_u tokens; row u occurs n_u times and s_y sums the observed tokens' logits."""
    L, m1, v = _moments(Z, beta)
    N = n_u.sum()
    return float((n_u @ L - beta * s_y) / N), float((n_u @ m1 - s_y) / N), float(n_u @ v / N)


def fit_beta(Z: np.ndarray, n_u: np.ndarray, s_y: float, tol: float = 1e-13) -> float:
    """The NLL-optimal β = 1/τ by Newton's method on the convex L(β), halving any step that leaves β ≤ 0 or raises L."""
    beta = 1.0
    f, g, h = temperature_objective(Z, n_u, s_y, beta)
    for _ in range(100):
        step = g / h
        t = 1.0
        while True:
            nb = beta - t * step
            if nb > 0:
                nf, ng_, nh = temperature_objective(Z, n_u, s_y, nb)
                if nf <= f + 1e-15:
                    break
            t /= 2
            if t < 1e-12:
                return beta
        beta, f, g, h = nb, nf, ng_, nh
        if abs(g) < tol:
            break
    return beta


# --------------------------------------------------------------------------------------------
# 5. The corpus engines on three test sets.
# --------------------------------------------------------------------------------------------

TEST_SETS = ("continuation", "held", "edgar")
ENGINES = ("trigram", "decoder")
NUCLEUS_PS = (0.5, 0.8, 0.9, 0.95)


@lru_cache(maxsize=None)
def sentences(name: str) -> tuple[tuple[str, ...], ...]:
    """continuation: the training corpus's generator run on for 100 documents (ng.continuation_run), the same source;
    held: Topic 3's held-out split, generate_corpus(100, 1), a fresh run of the generator whose restaurants restart
    from the seed nouns, so its word frequencies are not the training text's;
    edgar: the public MD&A excerpts, real filings; train_half: every second training sentence (the decoder's
    training-data temperature)."""
    if name == "continuation":
        return tuple(ng.continuation_run(100)["continuation"])
    if name == "train_half":
        return tuple(ng.corpus()["train"][::2])
    return tuple(ng.corpus()[name])


@dataclass(frozen=True)
class Rows:
    """An engine's next-token distributions on a test set. Z holds one log-probability row per distinct context (the
    trigram's last two words; every position for the decoder); token i occurred after context ctx[i]."""

    Z: np.ndarray  # (U, V)
    order: np.ndarray  # (U, V) tokens by decreasing probability, ties to the lower id (τ-invariant)
    ctx: np.ndarray  # (n,)
    y: np.ndarray  # (n,)
    rank: np.ndarray  # (n,) rank of y in its row's order
    sent: np.ndarray  # (n,) sentence index

    @property
    def n_u(self) -> np.ndarray:
        return np.bincount(self.ctx, minlength=len(self.Z)).astype(float)

    def s_y(self, idx: np.ndarray | None = None) -> float:
        i = np.arange(len(self.y)) if idx is None else idx
        return float(self.Z[self.ctx[i], self.y[i]].sum())


def _finish(Z: np.ndarray, ctx: list, y: list, sent: list) -> Rows:
    order = np.argsort(-Z, axis=1, kind="stable").astype(np.int32)
    ctx_a, y_a = np.array(ctx), np.array(y)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(Z.shape[1], dtype=np.int32), Z.shape), axis=1)
    return Rows(Z, order, ctx_a, y_a, rank[ctx_a, y_a].astype(int), np.array(sent))


def _trigram_rows(sents) -> Rows:
    """Topic 3's interpolated KN trigram over its own alphabet with <unk> as an outcome (an unknown word is the event
    <unk>), so every token of any text is scored; contexts pad with two <s>."""
    vec = bs.trigram().vec
    ix = vec.index
    ids: dict = {}
    Zs, ctx, y, sent = [], [], [], []
    for si, s in enumerate(sents):
        p = [BOS, BOS] + [w if w in ix else UNK for w in s] + [EOS]
        for i in range(2, len(p)):
            u = (p[i - 2], p[i - 1])
            k = ids.get(u)
            if k is None:
                k = ids[u] = len(Zs)
                Zs.append(np.log(vec(u)))
            ctx.append(k)
            y.append(ix[p[i]])
            sent.append(si)
    return _finish(np.array(Zs), ctx, y, sent)


def _decoder_rows(sents) -> Rows:
    """Topic 14's decoder over its own 512-token alphabet with <unk> as an outcome and <s> masked (it never follows a
    word inside a sentence), every position of each sentence's first DEC_CTX inputs: words past position 32 are not
    scored, since the position table ends there."""
    dec = bs.decoder()
    ix = dec.index
    seqs = [([ix[BOS]] + [ix.get(w, ix[UNK]) for w in s] + [ix[EOS]])[: bs.DEC_CTX + 1] for s in sents]
    by_len: dict[int, list[int]] = {}
    for i, q in enumerate(seqs):
        by_len.setdefault(len(q), []).append(i)
    out: dict[int, np.ndarray] = {}
    for L, idx in by_len.items():
        X = np.array([seqs[i] for i in idx])
        lp = sl.decoder_log_probs(dec.P, dec.blocks, X[:, :-1])
        lp[..., ix[BOS]] = -np.inf
        lp = lp - st._lse(lp, axis=-1, keepdims=True)
        for j, i in enumerate(idx):
            out[i] = lp[j]
    Zs, y, sent = [], [], []
    for i, q in enumerate(seqs):
        Zs.append(out[i])
        y += q[1:]
        sent += [i] * (len(q) - 1)
    Z = np.concatenate(Zs)
    return _finish(Z, list(range(len(Z))), y, sent)


@lru_cache(maxsize=None)
def rows(engine: str, test: str) -> Rows:
    return (_trigram_rows if engine == "trigram" else _decoder_rows)(sentences(test))


@lru_cache(maxsize=None)
def tau_star(engine: str, test: str) -> float:
    R = rows(engine, test)
    return 1 / fit_beta(R.Z, R.n_u, R.s_y())


@dataclass(frozen=True)
class TauStats:
    conf: np.ndarray  # (n,) confidence of the top token
    correct: np.ndarray  # (n,) the top token occurred
    nll: float  # mean −log q(y)
    covered: dict  # p → share of tokens inside nucleus(p) of the tempered row
    kept: dict  # p → mean kept mass of nucleus(p)


@lru_cache(maxsize=None)
def stats_at(engine: str, test: str, tau: float) -> TauStats:
    return tau_stats(rows(engine, test), tau)


def tau_stats(R: Rows, tau: float) -> TauStats:
    """Every per-token statistic of the tempered rows q = softmax(z/τ), from one pass over the distinct rows."""
    Zt = st.temper(R.Z, tau)
    conf_u, pred_u = np.exp(Zt.max(axis=1)), Zt.argmax(axis=1)
    cs = np.cumsum(np.exp(np.take_along_axis(Zt, R.order, axis=1)), axis=1)
    covered, kept = {}, {}
    for p in NUCLEUS_PS:
        k = (cs < p - 1e-12).sum(axis=1) + 1  # Topic 15's nucleus: the shortest sorted prefix of mass ≥ p
        covered[p] = float(np.mean(R.rank < k[R.ctx]))
        kept[p] = float(np.mean(cs[np.arange(len(k)), k - 1][R.ctx]))
    nll = float(-Zt[R.ctx, R.y].mean())
    return TauStats(conf_u[R.ctx], pred_u[R.ctx] == R.y, nll, covered, kept)


@lru_cache(maxsize=None)
def cross_fit_gain(engine: str, test: str) -> float:
    """The held-out NLL a temperature recovers, cross-fitted: fit τ on the even sentences and score the odd ones, and
    the reverse; the mean drop in NLL from τ = 1. By Cor 1 it estimates a lower bound on the reliability term."""
    R = rows(engine, test)
    gains = []
    for fold in (0, 1):
        fit, ev = np.flatnonzero(R.sent % 2 == fold), np.flatnonzero(R.sent % 2 != fold)
        n_fit = np.bincount(R.ctx[fit], minlength=len(R.Z)).astype(float)
        n_ev = np.bincount(R.ctx[ev], minlength=len(R.Z)).astype(float)
        b = fit_beta(R.Z, n_fit, R.s_y(fit))
        gains.append(temperature_objective(R.Z, n_ev, R.s_y(ev), 1.0)[0]
                     - temperature_objective(R.Z, n_ev, R.s_y(ev), b)[0])
    return float(np.mean(gains))


def confidence_kendall(engine: str, test: str, tau: float) -> float:
    """Kendall's τ_b between the distinct rows' confidences at temperature 1 and at `tau` (1 iff no pair swaps)."""
    Z = rows(engine, test).Z
    return float(stats.kendalltau(Z.max(axis=1), st.temper(Z, tau).max(axis=1)).statistic)


# --------------------------------------------------------------------------------------------
# 4. A family closed under scaling: the log-linear bigram.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogLinearConfig:
    lam: float  # the penalty (λ/2)‖W‖²


LAMBDAS = (1e-3, 1e-4, 1e-5)


@lru_cache(maxsize=None)
def bigram_counts(name: str) -> np.ndarray:
    """C[a, b] = how often b follows a in a test set, over the decoder's 512-token alphabet (<s> opens a sentence)."""
    ix = {w: i for i, w in enumerate(bs.decoder_vocab())}
    V = len(ix)
    C = np.zeros((V, V))
    for s in sentences(name) if name != "train" else ng.corpus()["train"]:
        t = [ix[BOS]] + [ix.get(w, ix[UNK]) for w in s] + [ix[EOS]]
        np.add.at(C, (t[:-1], t[1:]), 1)
    return C


@lru_cache(maxsize=None)
def loglinear_bigram(cfg: LogLinearConfig) -> np.ndarray:
    """W minimizing the mean training NLL of softmax(W[a]) plus (λ/2)‖W‖², by L-BFGS from W = 0. Logits are linear in
    W, so βW is the same family at temperature 1/β."""
    C = bigram_counts("train")
    V, N = len(C), C.sum()
    n = C.sum(axis=1, keepdims=True)

    def f(w):
        W = w.reshape(V, V)
        L = st._lse(W, axis=1, keepdims=True)
        val = float((n[:, 0] @ L[:, 0] - (C * W).sum()) / N + cfg.lam / 2 * (w @ w))
        return val, ((n * np.exp(W - L) - C) / N + cfg.lam * W).ravel()

    res = optimize.minimize(f, np.zeros(V * V), jac=True, method="L-BFGS-B",
                            options={"maxiter": 5000, "gtol": 1e-11, "ftol": 1e-16})
    return res.x.reshape(V, V)


def bigram_temperature(W: np.ndarray, name: str) -> float:
    """The NLL-optimal τ of the bigram's rows on a test set (rows weighted by their prefix counts)."""
    C = bigram_counts(name)
    return 1 / fit_beta(W, C.sum(axis=1), float((C * W).sum()))


def bigram_dbeta(W: np.ndarray) -> float:
    """d/dβ of the mean training NLL of βW at β = 1."""
    C = bigram_counts("train")
    return temperature_objective(W, C.sum(axis=1), float((C * W).sum()), 1.0)[1]


# --------------------------------------------------------------------------------------------
# 7. Finance case study: Topic 15's fact model.
# --------------------------------------------------------------------------------------------

FIN_TAUS = st.FIN_TAUS
REGIMES = st.REGIMES


def _fact_logits(p: st.Probe) -> tuple[np.ndarray, int]:
    """The fact model's log-probabilities restricted to the relation's answers (tempering then conditioning on the
    answer set equals conditioning then tempering), and the KB answer's position."""
    return p.row[p.answers], int(np.flatnonzero(p.answers == p.correct)[0])


def fact_stats(probes, tau: float) -> dict:
    """Mean confidence (top answer's probability among the answers), accuracy (top answer is the KB's), mean
    P(KB answer), and mean −log P(KB answer), at temperature τ."""
    conf, acc, pc, nll = [], [], [], []
    for p in probes:
        z, j = _fact_logits(p)
        q = st.temper(z, tau)
        conf.append(math.exp(q.max()))
        acc.append(q.argmax() == j)
        pc.append(math.exp(q[j]))
        nll.append(-q[j])
    return {"conf": float(np.mean(conf)), "acc": float(np.mean(acc)), "p_correct": float(np.mean(pc)),
            "nll": float(np.mean(nll))}


def fact_regime(g: str) -> tuple[st.Probe, ...]:
    return tuple(p for p in st.probes() if g == "all" or p.regime == g)


@lru_cache(maxsize=None)
def fact_tau_star(g: str) -> float:
    """The NLL-optimal τ for a regime on [FIN_TAUS[0], FIN_TAUS[-1]] (0.040 to 10): the best grid point, refined by a
    bounded search between its neighbors. A regime whose optimum is a limit lands on an edge (the search near a flat
    edge would stop short of it)."""
    sub = fact_regime(g)
    nll = [fact_stats(sub, t)["nll"] for t in FIN_TAUS]
    i = int(np.argmin(nll))
    if i in (0, len(FIN_TAUS) - 1):
        return float(FIN_TAUS[i])
    res = optimize.minimize_scalar(lambda lt: fact_stats(sub, math.exp(lt))["nll"], method="bounded",
                                   bounds=(math.log(FIN_TAUS[i - 1]), math.log(FIN_TAUS[i + 1])), options={"xatol": 1e-9})
    return float(math.exp(res.x))


def count_aware_nll() -> dict:
    """Pooled NLL of the KB answer at τ = 1, at the one pooled τ*, and with each regime at its own τ*."""
    ps = fact_regime("all")
    per = sum(len(fact_regime(g)) * fact_stats(fact_regime(g), fact_tau_star(g))["nll"] for g in REGIMES) / len(ps)
    return {"tau1": fact_stats(ps, 1.0)["nll"], "pooled": fact_stats(ps, fact_tau_star("all"))["nll"], "per_regime": per}


# --------------------------------------------------------------------------------------------
# 8. The lab's payload.
# --------------------------------------------------------------------------------------------

LAB_TAUS = tuple(st.TAUS[k + 28] for k in range(-6, 15, 2))  # 10^{k/20}, k = −6, −4, …, 14: 0.501 to 5.01, τ = 1 on it
LAB_BINS = (1, 5, 10, 15, 30, 60)
TWIN_R = 100
TWIN_STREAM = 1910


def _set_payload(engine: str, test: str) -> dict:
    pts = []
    for tau in LAB_TAUS:
        s = stats_at(engine, test, tau)
        rng = np.random.default_rng([SEED, TWIN_STREAM, ENGINES.index(engine), TEST_SETS.index(test)])
        tw = twin_ece(s.conf, LAB_BINS, TWIN_R, rng)
        bins = {}
        for B in LAB_BINS:
            n, c, a = reliability_bins(s.conf, s.correct, B)
            bins[str(B)] = [[int(k), _r(x), _r(y)] for k, x, y in zip(n, c, a)]
        pts.append({"nll": _r(s.nll), "acc": _r(s.correct.mean()), "conf": _r(s.conf.mean()),
                    "cover": _r(s.covered[0.9]), "kept": _r(s.kept[0.9]),
                    "ece": _r([ece_hat(s.conf, s.correct, B) for B in LAB_BINS]),
                    "twin": _r(tw.mean(axis=0)), "twin_sd": _r(tw.std(axis=0, ddof=1)), "bins": bins})
    t = tau_star(engine, test)
    s = stats_at(engine, test, t)
    return {"n": len(rows(engine, test).y), "tau_star": _r(t), "nll_star": _r(s.nll), "gain": _r(cross_fit_gain(engine, test)),
            "points": pts}


def _facts_payload() -> dict:
    out = {"taus": _r(list(FIN_TAUS), 4)}
    for g in REGIMES + ("all",):
        sub = fact_regime(g)
        curve = [fact_stats(sub, t) for t in FIN_TAUS]
        out[g] = {"n": len(sub), "acc": _r(curve[0]["acc"]), "tau_star": _r(fact_tau_star(g)),
                  **{k: _r([c[k] for c in curve]) for k in ("conf", "p_correct", "nll")}}
    out["count_aware"] = {k: _r(v) for k, v in count_aware_nll().items()}
    return out


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    return {"taus": _r(list(LAB_TAUS), 4), "bins": list(LAB_BINS), "twin_r": TWIN_R,
            "engines": {e: {"V": int(rows(e, "edgar").Z.shape[1]), "sets": {t: _set_payload(e, t) for t in TEST_SETS}}
                        for e in ENGINES},
            "facts": _facts_payload()}


VIZ_JSON = (Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data"
            / "language-model-calibration-lab.json")


def summary() -> dict:
    out = {}
    for e in ENGINES:
        for t in TEST_SETS:
            R, ts = rows(e, t), tau_star(e, t)
            s1, s2 = stats_at(e, t, 1.0), stats_at(e, t, ts)
            out[f"{e}_{t}_n"] = len(R.y)
            out[f"{e}_{t}_unk_share"] = float(np.mean(R.y == (bs.trigram().unk if e == "trigram" else bs.decoder().unk)))
            out[f"{e}_{t}_tau_star"] = ts
            out[f"{e}_{t}_conf"], out[f"{e}_{t}_acc"] = float(s1.conf.mean()), float(s1.correct.mean())
            out[f"{e}_{t}_ece15"] = ece_hat(s1.conf, s1.correct, 15)
            out[f"{e}_{t}_ece15_tau_star"] = ece_hat(s2.conf, s2.correct, 15)
            out[f"{e}_{t}_conf_tau_star"] = float(s2.conf.mean())
            out[f"{e}_{t}_nll"], out[f"{e}_{t}_nll_tau_star"] = s1.nll, s2.nll
            out[f"{e}_{t}_cover90"], out[f"{e}_{t}_kept90"] = s1.covered[0.9], s1.kept[0.9]
            out[f"{e}_{t}_cover90_tau_star"] = s2.covered[0.9]
            out[f"{e}_{t}_gain"] = cross_fit_gain(e, t)
            tw = twin_ece(s1.conf, LAB_BINS, TWIN_R,
                          np.random.default_rng([SEED, TWIN_STREAM, ENGINES.index(e), TEST_SETS.index(t)]))
            j = LAB_BINS.index(15)
            out[f"{e}_{t}_twin15_mean"], out[f"{e}_{t}_twin15_sd"] = float(tw[:, j].mean()), float(tw[:, j].std(ddof=1))
            out[f"{e}_{t}_sq_ece15_plugin"] = float(sum(
                n * (a - c) ** 2 for n, c, a in zip(*reliability_bins(s1.conf, s1.correct, 15))) / len(R.y))
            out[f"{e}_{t}_sq_ece15_debiased"] = sq_ece_debiased(s1.conf, s1.correct, 15)
    out["decoder_train_tau_star"] = tau_star("decoder", "train_half")
    out["trigram_continuation_kendall_tau2"] = confidence_kendall("trigram", "continuation", 2.0)
    for lam in LAMBDAS:
        W = loglinear_bigram(LogLinearConfig(lam))
        out[f"bigram_{lam:g}_dbeta"] = bigram_dbeta(W)
        out[f"bigram_{lam:g}_minus_lam_norm"] = -lam * float((W**2).sum())
        for name in ("train", "continuation", "held"):
            out[f"bigram_{lam:g}_{name}_tau_star"] = bigram_temperature(W, name)
    for g in REGIMES + ("all",):
        s = fact_stats(fact_regime(g), 1.0)
        out[f"fact_{g}_n"] = len(fact_regime(g))
        out[f"fact_{g}_conf"], out[f"fact_{g}_acc"], out[f"fact_{g}_nll"] = s["conf"], s["acc"], s["nll"]
        out[f"fact_{g}_tau_star"] = fact_tau_star(g)
    for k, v in count_aware_nll().items():
        out[f"fact_nll_{k}"] = v
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: every pedagogical claim, as an assert.
# --------------------------------------------------------------------------------------------


def _worlds(n: int, stream: int, calibrated: bool = False) -> list[World]:
    rng = np.random.default_rng([SEED, stream])
    return [(calibrated_world if calibrated else random_world)(rng) for _ in range(n)]


def test_canonical_implies_top_label():
    """Prop 1(a)(b): a canonically calibrated forecaster has zero top-label error (on 50 random calibrated worlds),
    and the counterexample is top-label calibrated but not canonically."""
    for w in _worlds(50, 1901, calibrated=True):
        assert np.abs(recalibrated(w) - w.F).max() < 1e-12
        assert top_label_ece(w) < 1e-12
    cx = counterexample_world()
    assert top_label_ece(cx) < 1e-15 and np.abs(recalibrated(cx) - cx.F).max() > 0.09


def test_distinct_rows_make_calibration_truth():
    """Prop 1(c): when every context has its own forecast row, the recalibration is the truth itself, so canonical
    calibration means F = P and the reliability term is the whole excess loss E KL(P ‖ F)."""
    rng = np.random.default_rng([SEED, 1902])
    for _ in range(30):
        w = World(rng.dirichlet(np.ones(25)), rng.dirichlet(np.ones(5), 25), rng.dirichlet(np.ones(5), 25))
        assert len(np.unique(level_sets(w.F))) == 25
        assert np.abs(recalibrated(w) - w.P).max() < 1e-12
        d = log_decomposition(w)
        assert abs(d["reliability"] - (d["loss"] - d["irreducible"])) < 1e-12 and abs(d["refinement_gap"]) < 1e-12


def test_log_and_brier_decompositions():
    """Thm 1 on 50 random worlds: each loss equals irreducible + refinement gap + reliability to 1e-12; the
    recalibrated forecaster's loss is irreducible + refinement gap; every term is ≥ 0, and the reliability term vanishes
    exactly on calibrated worlds."""
    for w in _worlds(50, 1903):
        for d in (log_decomposition(w), brier_decomposition(w)):
            assert abs(d["loss"] - d["irreducible"] - d["refinement_gap"] - d["reliability"]) < 1e-12
            assert abs(d["recalibrated_loss"] - d["irreducible"] - d["refinement_gap"]) < 1e-12
            assert min(d["refinement_gap"], d["reliability"]) > -1e-15 and d["reliability"] > 1e-6
    for w in _worlds(20, 1904, calibrated=True):
        assert log_decomposition(w)["reliability"] < 1e-12 and brier_decomposition(w)["reliability"] < 1e-12


def test_any_recalibration_is_bounded_by_reliability():
    """Cor 1: any map of the forecast (here every temperature on a 41-point grid) recovers at most the reliability
    term, and on a calibrated world no temperature lowers the loss."""
    taus = np.geomspace(0.2, 5, 41)
    for w in _worlds(30, 1905):
        rel = log_decomposition(w)["reliability"]
        assert all(log_loss(w) - log_loss(w, temper_rows(w.F, t)) <= rel + 1e-12 for t in taus)
    for w in _worlds(20, 1906, calibrated=True):
        assert all(log_loss(w, temper_rows(w.F, t)) >= log_loss(w) - 1e-12 for t in taus)


def test_binning_hides():
    """Thm 2(a) on 50 random worlds: along the nested grid B = 1, 2, 4, …, 64 the population binned ECE never
    decreases, and it never exceeds the true top-label error."""
    for w in _worlds(50, 1907):
        e = [binned_population_ece(w, 2**k) for k in range(7)]
        assert all(b >= a - 1e-12 for a, b in zip(e, e[1:])) and e[-1] <= top_label_ece(w) + 1e-12


def test_sampling_inflates_and_the_square_debiases():
    """Thm 2(b)(c) by exact Poisson-binomial expectations on 200 random bins (2 to 30 tokens): E|ā − p̄| ≥ |c̄ − p̄|,
    strictly; E(ā − p̄)² = (c̄ − p̄)² + Var ā; the debiased square has expectation (c̄ − p̄)² − Σ(c_i − c̄)²/(n(n − 1))."""
    rng = np.random.default_rng([SEED, 1908])
    for _ in range(200):
        n = int(rng.integers(2, 31))
        c, p = rng.uniform(0.05, 0.95, n), rng.uniform(0, 1, n)
        e = bin_expectations(c, p)
        assert e["abs"] > e["true_gap"] + 1e-9
        assert abs(e["sq"] - (e["true_gap"] ** 2 + np.sum(c * (1 - c)) / n**2)) < 1e-12
        assert abs(e["debiased"] - (e["true_gap"] ** 2 - e["spread"] / (n * (n - 1)))) < 1e-12


def test_calibrated_twin_floor():
    """Thm 2(b) on the corpus: the trigram's calibrated twins (in distribution) have a positive ECE that rises with the
    bin count on LAB_BINS, while the trigram's own ECE sits more than four twin standard deviations above the twin
    mean at B = 15 (a small, real miscalibration)."""
    s = stats_at("trigram", "continuation", 1.0)
    tw = twin_ece(s.conf, LAB_BINS, TWIN_R, np.random.default_rng([SEED, TWIN_STREAM, 0, 0]))
    m = tw.mean(axis=0)
    assert np.all(np.diff(m) > 0) and m[0] > 0
    j = LAB_BINS.index(15)
    assert ece_hat(s.conf, s.correct, 15) > m[j] + 4 * tw[:, j].std(ddof=1)


def test_temperature_fit_is_convex_and_matches_moments():
    """Prop 2: on every engine × test set, L'' > 0 and finite differences match L' and L''; at Newton's τ* the
    expected logit equals the observed one (|L'| < 1e-10), which by convexity is the global optimum; on the EDGAR sets
    Newton also agrees with scipy's bounded search."""
    for e in ENGINES:
        for t in TEST_SETS:
            R = rows(e, t)
            n_u, s_y = R.n_u, R.s_y()
            for b in (0.6, 1.5):
                f, g, h = temperature_objective(R.Z, n_u, s_y, b)
                d = 1e-5
                fp, gp, _ = temperature_objective(R.Z, n_u, s_y, b + d)
                fm, gm, _ = temperature_objective(R.Z, n_u, s_y, b - d)
                assert h > 0 and abs((fp - fm) / (2 * d) - g) < 1e-6 and abs((gp - gm) / (2 * d) - h) < 1e-5 * max(1, h)
            b = 1 / tau_star(e, t)
            assert abs(temperature_objective(R.Z, n_u, s_y, b)[1]) < 1e-10
            if t != "edgar":
                continue
            ref = optimize.minimize_scalar(lambda x: temperature_objective(R.Z, n_u, s_y, x)[0], bounds=(0.05, 5),
                                           method="bounded", options={"xatol": 1e-10})
            assert abs(ref.x - b) < 1e-6


def test_temperature_keeps_argmax_not_ranking():
    """Prop 2(c): accuracy is the same at every lab temperature, but the confidences of distinct rows change order
    (Kendall's τ_b < 1 between τ = 1 and τ = 2 on the in-distribution trigram rows), so temperature scaling is not a
    monotone map of the confidence."""
    accs = {round(float(stats_at("trigram", "continuation", t).correct.mean()), 15) for t in LAB_TAUS}
    assert len(accs) == 1
    assert confidence_kendall("trigram", "continuation", 2.0) < 0.99
    f, g = temper_rows(np.array([[0.5, 0.5, 0.0]]), 2.0), temper_rows(np.array([[0.6, 0.2, 0.2]]), 2.0)
    assert abs(f.max() - 0.5) < 1e-12 and round(float(g.max()), 3) == 0.464


def test_kneser_ney_reliability_shape():
    """The in-distribution trigram's 15-bin reliability diagram: every bin with mean confidence in [1/3, 0.6) is
    overconfident and every bin above 2/3 is underconfident (KN's discounts), and the top bin holds the most tokens."""
    s = stats_at("trigram", "continuation", 1.0)
    n, c, a = reliability_bins(s.conf, s.correct, 15)
    mid, top = (c >= 1 / 3) & (c < 0.6), c > 2 / 3
    assert mid.sum() >= 3 and np.all(a[mid] < c[mid]) and top.sum() >= 3 and np.all(a[top] > c[top])
    assert n.argmax() == len(n) - 1


def test_scale_closed_stationarity():
    """Thm 3 on the log-linear bigram: at the penalized optimum, d/dβ L_train(βW) at β = 1 equals −λ‖W‖² (relative
    1e-5); so, by Prop 2's convexity, τ*_train < 1, and it falls as λ grows."""
    prev = 1.0
    for lam in sorted(LAMBDAS):
        W = loglinear_bigram(LogLinearConfig(lam))
        d, target = bigram_dbeta(W), -lam * float((W**2).sum())
        assert abs(d - target) < 1e-5 * abs(target)
        t = bigram_temperature(W, "train")
        assert t < prev and t < 1
        prev = t


def test_decoder_has_no_temperature_direction():
    """Thm 3's contrapositive in practice: Topic 14's decoder (parameter-free final LayerNorm, tied unembedding scaled
    by a fixed 1/√d) has no parameter that scales its logits, and its training-data τ* is not 1."""
    t = tau_star("decoder", "train_half")
    assert 0.9 < t < 0.97


def test_in_distribution_nearly_calibrated():
    """On the continuation (same generator): both engines have τ* within 0.05 of 1, ECE₁₅ below 0.02, and a
    cross-fitted temperature gain below 0.003 nats."""
    for e in ENGINES:
        s = stats_at(e, "continuation", 1.0)
        assert abs(tau_star(e, "continuation") - 1) < 0.05 and ece_hat(s.conf, s.correct, 15) < 0.02
        assert cross_fit_gain(e, "continuation") < 0.003


def test_shift_can_go_either_way():
    """Topic 3's held-out split (a fresh run, new names): the decoder becomes underconfident (accuracy above confidence,
    τ* < 0.9) while the trigram turns slightly overconfident (τ* > 1)."""
    s = stats_at("decoder", "held", 1.0)
    assert s.correct.mean() > s.conf.mean() + 0.03 and tau_star("decoder", "held") < 0.9
    assert tau_star("trigram", "held") > 1


def test_edgar_overconfidence():
    """Real filings: both engines are overconfident (confidence above accuracy, τ* > 1.5, a positive cross-fitted
    gain); the trigram's nucleus(0.9) covers under half of the tokens and still does at its τ*; and the NLL-optimal
    temperature overshoots top-label calibration (confidence falls below accuracy)."""
    for e in ENGINES:
        s = stats_at(e, "edgar", 1.0)
        assert s.conf.mean() > s.correct.mean() + 0.05 and tau_star(e, "edgar") > 1.5 and cross_fit_gain(e, "edgar") > 0
    s1, s2 = stats_at("trigram", "edgar", 1.0), stats_at("trigram", "edgar", tau_star("trigram", "edgar"))
    assert s1.covered[0.9] < 0.5 and s2.covered[0.9] < 0.5 and s1.kept[0.9] > 0.9
    assert s2.conf.mean() < s2.correct.mean()


def test_coverage_equals_kept_mass_when_calibrated():
    """Prop 3: on 30 calibrated worlds, the nucleus of every mass p covers Y with probability exactly its expected kept
    mass (≥ p); on random worlds the two differ. On the trigram's in-distribution rows with labels drawn from the model
    itself, coverage of nucleus(0.9) matches the kept mass within 4 standard errors."""
    for calibrated, stream in ((True, 1911), (False, 1912)):
        gaps = []
        for w in _worlds(30, stream, calibrated):
            for p in NUCLEUS_PS:
                keep = st.truncation_set(np.log(w.F), "nucleus", p)
                cover, mass = float(w.w @ (w.P * keep).sum(axis=1)), float(w.w @ (w.F * keep).sum(axis=1))
                assert mass >= p - 1e-12
                gaps.append(abs(cover - mass))
        assert (max(gaps) < 1e-12) if calibrated else (max(gaps) > 0.05)
    R = rows("trigram", "continuation")
    rng = np.random.default_rng([SEED, 1913])
    P = np.exp(R.Z[R.ctx])
    u = rng.random(len(R.ctx))
    y = np.minimum((np.cumsum(P, axis=1) < u[:, None]).sum(axis=1), P.shape[1] - 1)
    twin = Rows(R.Z, R.order, R.ctx, y, np.argsort(R.order, axis=1)[R.ctx, y], R.sent)
    s = tau_stats(twin, 1.0)
    se = math.sqrt(s.kept[0.9] * (1 - s.kept[0.9]) / len(y))
    assert abs(s.covered[0.9] - s.kept[0.9]) < 4 * se


def test_sampling_temperature_decalibrates():
    """Cor 2: on 20 calibrated worlds, the tempered sampler at any τ ≠ 1 has a positive reliability term."""
    for w in _worlds(20, 1914, calibrated=True):
        for t in (0.5, 0.8, 1.25, 2.0):
            assert log_decomposition(World(w.w, w.P, temper_rows(w.F, t)))["reliability"] > 1e-6


def test_fast_nucleus_matches_topic15():
    """tau_stats' nucleus (one sort per row, reused at every τ) equals Topic 15's truncation_set row by row."""
    R = rows("trigram", "edgar")
    for tau in (0.7, 1.0, 2.0):
        Zt = st.temper(R.Z, tau)
        for p in NUCLEUS_PS:
            keep = st.truncation_set(Zt, "nucleus", p)
            k = (np.cumsum(np.exp(np.take_along_axis(Zt, R.order, axis=1)), axis=1) < p - 1e-12).sum(axis=1) + 1
            mine = np.zeros_like(keep)
            np.put_along_axis(mine, R.order, np.arange(R.Z.shape[1])[None, :] < k[:, None], axis=1)
            assert np.array_equal(keep, mine)


def test_fact_regimes_pull_apart():
    """The finance case study: seen facts are always right but under-confident (τ* at the grid's low edge; NLL falls
    as τ falls, reaching 0 to double precision); unseen facts are over-confident (τ* at the high edge; NLL falls with τ
    to within 0.05 of the mean log answer count, the uniform forecast's loss);
    one pooled temperature recovers under 0.02 nats while a temperature per regime recovers more than 0.4."""
    seen, un = fact_stats(fact_regime("seen"), 1.0), fact_stats(fact_regime("unseen"), 1.0)
    assert seen["acc"] == 1.0 and seen["conf"] < 0.9 and un["conf"] > un["acc"] + 0.2
    assert fact_tau_star("seen") < 1.001 * FIN_TAUS[0] and fact_tau_star("unseen") > 0.999 * FIN_TAUS[-1]
    i1 = FIN_TAUS.index(1.0)
    nll_seen = [fact_stats(fact_regime("seen"), t)["nll"] for t in FIN_TAUS[: i1 + 1]]
    nll_un = [fact_stats(fact_regime("unseen"), t)["nll"] for t in FIN_TAUS[i1:]]
    assert np.all(np.diff(nll_seen) >= 0) and nll_seen[0] < 1e-12 and np.all(np.diff(nll_un) < 0)
    logk = np.mean([math.log(len(p.answers)) for p in fact_regime("unseen")])
    assert 0 < nll_un[-1] - logk < 0.05
    assert FIN_TAUS[0] < fact_tau_star("once") < 1
    c = count_aware_nll()
    assert c["tau1"] - c["pooled"] < 0.02 and c["tau1"] - c["per_regime"] > 0.4


def test_viz_payload():
    p = viz_payload()
    assert p["taus"][LAB_TAUS.index(1.0)] == 1.0 and len(p["engines"]) == 2
    for e in ENGINES:
        for t in TEST_SETS:
            sp = p["engines"][e]["sets"][t]
            assert len(sp["points"]) == len(LAB_TAUS) and sp["n"] == len(rows(e, t).y)
            for pt in sp["points"]:
                assert set(pt["bins"]) == {str(B) for B in LAB_BINS} and sum(b[0] for b in pt["bins"]["15"]) == sp["n"]
    assert set(p["facts"]) >= {*REGIMES, "all", "count_aware", "taus"}


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
    print(f"\nSummary (Topic 3's IKN trigram, V = {len(bs.trigram().alphabet)}; Topic 14's decoder, V = {bs.DEC_V}; "
          f"Topic 15's order-{st.FACT_ORDER} KN fact model, {len(st.probes())} probes):")
    for k, v in summary().items():
        print(f"  {k:44} {v:.6f}" if isinstance(v, float) else f"  {k:44} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__" and "--dev" in sys.argv:
    import time

    t0 = time.time()
    for k, v in summary().items():
        print(f"  {k:44} {v:.6f}" if isinstance(v, float) else f"  {k:44} {v}")
    print("time", round(time.time() - t0, 1))
elif __name__ == "__main__":
    raise SystemExit(_run_harness())
