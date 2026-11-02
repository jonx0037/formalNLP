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
   below 1 (Thm 3). Topic 14's decoder has no such direction, stops short of a stationary point, and lands at 0.94
   across its four training seeds. Theorem 3 says nothing about held-out temperatures; the bigram's cross 1 on some sets
   and not on others, and the trigram's 1.003 is a balance between its known words and <unk>.
5. Prediction sets. If f is canonically calibrated, any set chosen from f(X) covers Y with probability equal to its
   expected kept mass, so a nucleus of mass p covers at least p (Prop 3); sampling at τ ≠ 1 decalibrates a calibrated
   model (Cor 2).
6. The corpus. Topic 3's KN trigram and Topic 14's decoder on three test sets: the same generator run on
   (in distribution), Topic 3's held-out split (a fresh run with new names), and real EDGAR filings.
7. Finance case study. On EDGAR both engines look overconfident, but the trigram's reading is one token: <unk> is
   last in every row, so on its known words the means agree and τ* is 1.10, and coverage cannot pass the known share at
   any temperature. A map that moves mass onto <unk> recovers six times what temperature does. Topic 15's fact model is
   underconfident on facts it has seen and overconfident on facts it has not, so no one temperature calibrates it; on
   the 111 probes no row mixes regimes, so the regime sits on the reliability side of Thm 1 and the refinement gap is
   the company.

Cross-checks: exact enumeration on random finite worlds for every identity; Poisson-binomial distributions for the
estimator's expectation; scipy.optimize for every temperature fit; Topic 15's truncation_set for the nucleus.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/language-model-calibration/language_model_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
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
    with np.errstate(divide="ignore", invalid="ignore"):
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


def _trigram_rows(sents, vec=None) -> Rows:
    """Topic 3's interpolated KN trigram over its own alphabet with <unk> as an outcome (an unknown word is the event
    <unk>), so every token of any text is scored; contexts pad with two <s>. `vec` swaps in a refit model."""
    vec = bs.trigram().vec if vec is None else vec
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


def _decoder_rows(sents, dec=None) -> Rows:
    """Topic 14's decoder over its own 512-token alphabet with <unk> as an outcome and <s> masked (it never follows a
    word inside a sentence), every position of each sentence's first DEC_CTX inputs: words past position 32 are not
    scored, since the position table ends there. `dec` swaps in another training seed."""
    dec = bs.decoder() if dec is None else dec
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


def _nucleus_tokens(cs: np.ndarray, R: Rows, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Per token: is y inside nucleus(p) of its row, and that nucleus's kept mass. cs is the rows' sorted cumulative mass."""
    k = (cs < p - 1e-12).sum(axis=1) + 1  # Topic 15's nucleus: the shortest sorted prefix of mass ≥ p
    return R.rank < k[R.ctx], cs[np.arange(len(k)), k - 1][R.ctx]


def tau_stats(R: Rows, tau: float) -> TauStats:
    """Every per-token statistic of the tempered rows q = softmax(z/τ), from one pass over the distinct rows."""
    Zt = st.temper(R.Z, tau)
    conf_u, pred_u = np.exp(Zt.max(axis=1)), Zt.argmax(axis=1)
    cs = np.cumsum(np.exp(np.take_along_axis(Zt, R.order, axis=1)), axis=1)
    covered, kept = {}, {}
    for p in NUCLEUS_PS:
        cv, km = _nucleus_tokens(cs, R, p)
        covered[p], kept[p] = float(cv.mean()), float(km.mean())
    nll = float(-Zt[R.ctx, R.y].mean())
    return TauStats(conf_u[R.ctx], pred_u[R.ctx] == R.y, nll, covered, kept)


def _unk(R: Rows) -> int:
    """<unk>'s id, by the alphabet size of the rows: the trigram's 3,427 or the decoder's 512."""
    return bs.trigram().unk if R.Z.shape[1] == len(bs.trigram().alphabet) else bs.decoder().unk


def _temperature_gain(R: Rows, fit: np.ndarray, ev: np.ndarray) -> float:
    """Fit τ on the tokens `fit`; the drop in mean NLL from τ = 1 that it gives on the tokens `ev`."""
    n_fit = np.bincount(R.ctx[fit], minlength=len(R.Z)).astype(float)
    n_ev = np.bincount(R.ctx[ev], minlength=len(R.Z)).astype(float)
    b = fit_beta(R.Z, n_fit, R.s_y(fit))
    return (temperature_objective(R.Z, n_ev, R.s_y(ev), 1.0)[0] - temperature_objective(R.Z, n_ev, R.s_y(ev), b)[0])


def _fit_alpha(q: np.ndarray, is_unk: np.ndarray) -> float:
    """The α minimizing the mean NLL of g_α(F) = (1 − α)F + α·e_<unk>, where q is F's probability of each observed token."""
    res = optimize.minimize_scalar(lambda a: -np.mean(np.log((1 - a) * q + a * is_unk)), bounds=(1e-6, 1 - 1e-6),
                                   method="bounded", options={"xatol": 1e-7})
    return float(res.x)


def _alpha_gain(R: Rows, fit: np.ndarray, ev: np.ndarray) -> float:
    """_temperature_gain for g_α: a function of the forecast alone, so Cor 1 bounds it too."""
    q, is_unk = np.exp(R.Z[R.ctx, R.y]), (R.y == _unk(R)).astype(float)
    a = _fit_alpha(q[fit], is_unk[fit])
    return float(-np.mean(np.log(q[ev])) + np.mean(np.log((1 - a) * q[ev] + a * is_unk[ev])))


def _even_odd(R: Rows, gain) -> float:
    """Fit on the even sentences and score the odd ones, and the reverse; the mean per evaluated token, so the two folds
    count by their size (their token counts differ, and on 33 sentences that matters)."""
    folds = [(np.flatnonzero(R.sent % 2 == k), np.flatnonzero(R.sent % 2 != k)) for k in (0, 1)]
    return float(np.average([gain(R, f, e) for f, e in folds], weights=[len(e) for _, e in folds]))


@lru_cache(maxsize=None)
def cross_fit_gain(engine: str, test: str) -> float:
    """The held-out NLL a temperature recovers, cross-fitted on even and odd sentences. By Cor 1 it estimates a lower
    bound on the reliability term, and that bound can be far from tight: see unk_map_gain."""
    return _even_odd(rows(engine, test), _temperature_gain)


@lru_cache(maxsize=None)
def unk_map_gain(engine: str, test: str) -> float:
    """cross_fit_gain for the one-parameter map g_α, which moves mass onto <unk> and nothing else."""
    return _even_odd(rows(engine, test), _alpha_gain)


SPLIT_STREAMS = {"temperature": 7100, "unk": 7400}  # 20 random halves of the sentences, the same halves for every cell
N_SPLITS = 20


@lru_cache(maxsize=None)
def random_split_gains(engine: str, test: str, kind: str = "temperature") -> tuple[float, float]:
    """(mean, SD) of the cross-fitted gain over N_SPLITS random halves of the sentences: fit on one half, score the other."""
    R = rows(engine, test)
    rng = np.random.default_rng([SEED, SPLIT_STREAMS[kind]])
    ns = int(R.sent.max()) + 1
    out = []
    for _ in range(N_SPLITS):
        half = (rng.permutation(ns) < ns // 2)[R.sent]
        gain = _temperature_gain if kind == "temperature" else _alpha_gain
        out.append(gain(R, np.flatnonzero(half), np.flatnonzero(~half)))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def unk_map_stats(engine: str, test: str) -> dict:
    """g_α fit on all of a test set: α, the NLL it reaches, and the nucleus's coverage against its kept mass."""
    R = rows(engine, test)
    u, n = _unk(R), len(R.y)
    a = _fit_alpha(np.exp(R.Z[R.ctx, R.y]), (R.y == u).astype(float))
    P = np.exp(R.Z) * (1 - a)
    P[:, u] += a
    o = np.argsort(-P, axis=1, kind="stable")
    cs = np.cumsum(np.take_along_axis(P, o, axis=1), axis=1)
    k = (cs < 0.9 - 1e-12).sum(axis=1) + 1
    rank = np.argsort(o, axis=1)[R.ctx, R.y]
    return {"alpha": a, "nll": float(-np.log(P[R.ctx, R.y]).mean()), "cover": float(np.mean(rank < k[R.ctx])),
            "kept": float(cs[np.arange(len(k)), k - 1][R.ctx].mean()), "n": n}


def confidence_kendall(engine: str, test: str, tau: float, tokens: bool = False) -> float:
    """Kendall's τ_b between confidences at temperature 1 and at `tau` (1 iff no pair swaps), over the distinct rows, or
    over the tokens, which weights each row by how often it occurs."""
    R = rows(engine, test)
    c1, c2 = R.Z.max(axis=1), st.temper(R.Z, tau).max(axis=1)
    return float(stats.kendalltau(*((c1[R.ctx], c2[R.ctx]) if tokens else (c1, c2))).statistic)


def top_fifth(engine: str, test: str, tau: float) -> dict:
    """A selective rule that keeps the fifth of the tokens with the highest confidence, at τ = 1 and at `tau`: the share
    of its tokens that stay, and the accuracy of each selection."""
    R, s = rows(engine, test), stats_at(engine, test, 1.0)
    c1, c2 = R.Z.max(axis=1)[R.ctx], st.temper(R.Z, tau).max(axis=1)[R.ctx]
    q = len(c1) // 5
    a, b = np.argsort(-c1, kind="stable")[:q], np.argsort(-c2, kind="stable")[:q]
    return {"overlap": len(set(a.tolist()) & set(b.tolist())) / q, "acc1": float(s.correct[a].mean()),
            "acc2": float(s.correct[b].mean())}


# Reading the trigram's reliability diagram.


def top_bin_positions() -> dict:
    """Where the trigram's 15-bin top bin sits in a continuation sentence: the share of its tokens at positions 0–5, the
    same share among all tokens, and the shares of a sentence's first three tokens and of all tokens that land in it."""
    R, s = rows("trigram", "continuation"), stats_at("trigram", "continuation", 1.0)
    top = bin_index(s.conf, 15) == 14
    pos = np.arange(len(R.sent)) - np.searchsorted(R.sent, R.sent)
    return {"top_at_0_5": float(np.mean(pos[top] < 6)), "all_at_0_5": float(np.mean(pos < 6)),
            "first3_in_top": float(np.mean(top[pos < 3])), "all_in_top": float(top.mean()), "n": int(top.sum()),
            "acc": float(s.correct[top].mean())}


DISCOUNT_SCALES = (0.1, 0.25, 0.5, 1.0, 1.4)  # × Ney's discount, capped at 0.99


@lru_cache(maxsize=None)
def rediscounted_rows(f: float, test: str = "continuation") -> Rows:
    """The trigram refit with every order's discount scaled by f (Ney's D₃ is 0.849, so 1.4 hits the cap)."""
    ds = {k: (min(v[0] * f, 0.99),) for k, v in ng.kn_models()["ikn"].discounts.items()}
    return _trigram_rows(sentences(test), pce.KNVector(ng.kn_fit(ng.corpus()["train"], discounts=ds), pce.word_alphabet()))


def discount_shape() -> dict:
    """f → (τ*, gap from 1/3 to 0.6, gap above 2/3), accuracy minus confidence on the continuation. The two regions are
    the tokens Ney's trigram puts there, the same tokens in every column, so only the model moves."""
    c0 = stats_at("trigram", "continuation", 1.0).conf
    mid, top = (c0 >= 1 / 3) & (c0 < 0.6), c0 > 2 / 3
    out = {}
    for f in DISCOUNT_SCALES:
        R = rediscounted_rows(f)
        s = tau_stats(R, 1.0)
        out[f] = (1 / fit_beta(R.Z, R.n_u, R.s_y()), float(s.correct[mid].mean() - s.conf[mid].mean()),
                  float(s.correct[top].mean() - s.conf[top].mean()))
    return out


def by_top_label(engine: str, test: str) -> dict:
    """(tokens, mean confidence, accuracy) split by what the forecast puts on top: <unk> or a word."""
    R, s = rows(engine, test), stats_at(engine, test, 1.0)
    m = R.Z.argmax(axis=1)[R.ctx] == _unk(R)
    return {k: (int(x.sum()), float(s.conf[x].mean()), float(s.correct[x].mean())) for k, x in (("unk", m), ("word", ~m))}


def seen_context_rows(test: str = "continuation") -> dict:
    """The trigram's contexts in a test set, how many occurred in training, and how many distinct rows those have."""
    R = rows("trigram", test)
    ids: dict = {}
    ix = bs.trigram().vec.index
    for s in sentences(test):
        p = [BOS, BOS] + [w if w in ix else UNK for w in s] + [EOS]
        for i in range(2, len(p)):
            ids.setdefault((p[i - 2], p[i - 1]))
    totals = bs.trigram().vec.model.totals
    seen = np.array([totals.get(u, 0) > 0 for u in ids])
    return {"contexts": len(ids), "seen": int(seen.sum()), "seen_rows": len(np.unique(np.round(R.Z[seen], 10), axis=0))}


def repeated_sentences(test: str = "continuation") -> dict:
    """How many of a test set's sentences repeat a training sentence word for word, and their share of the tokens."""
    tr = set(ng.corpus()["train"])
    dup = np.array([s in tr for s in sentences(test)])
    return {"sentences": int(dup.sum()), "of": len(dup), "token_share": float(dup[rows("trigram", test).sent].mean())}


# The known-word cut: EDGAR's numbers are mostly one token.


@lru_cache(maxsize=None)
def known_rows(engine: str, test: str) -> Rows:
    """The engine's rows conditioned on the next token not being <unk>, on the tokens where it isn't. A forecast of the
    known words alone; on the trigram, whose P(<unk>) is about 1e-5, the conditioning changes nothing that shows."""
    R = rows(engine, test)
    u = _unk(R)
    Z = R.Z.copy()
    Z[:, u] = -np.inf
    i = np.flatnonzero(R.y != u)
    return _finish(Z - st._lse(Z, axis=1, keepdims=True), R.ctx[i].tolist(), R.y[i].tolist(), R.sent[i].tolist())


KNOWN_TWIN_STREAM = 7500


@lru_cache(maxsize=None)
def known_summary(engine: str, test: str) -> dict:
    """The table row of the known-word cut, in the same order as the full test set's row."""
    R = known_rows(engine, test)
    s = tau_stats(R, 1.0)
    tw = twin_ece(s.conf, (15,), TWIN_R, np.random.default_rng([SEED, KNOWN_TWIN_STREAM]))
    return {"n": len(R.y), "conf": float(s.conf.mean()), "acc": float(s.correct.mean()),
            "ece15": ece_hat(s.conf, s.correct, 15), "twin15": float(tw.mean()), "tau_star": 1 / fit_beta(R.Z, R.n_u, R.s_y()),
            "cover90": s.covered[0.9], "kept90": s.kept[0.9], "nll": s.nll}


def unk_summary(engine: str, test: str) -> dict:
    """<unk>'s place in the engine's rows: the worst rank of the outcome, its largest and mean probability, and the
    <unk> tokens' hits and nucleus(0.9) coverage."""
    R, s = rows(engine, test), stats_at(engine, test, 1.0)
    u, unk = _unk(R), R.y == _unk(R)
    cv, _ = _nucleus_tokens(np.cumsum(np.exp(np.take_along_axis(R.Z, R.order, axis=1)), axis=1), R, 0.9)
    r = np.argsort(R.order, axis=1)[:, u]
    return {"last_in_every_row": bool(np.all(r == R.Z.shape[1] - 1)), "share": float(unk.mean()), "tokens": int(unk.sum()),
            "max_p": float(np.exp(R.Z[:, u]).max()), "hits": int(s.correct[unk].sum()), "covered": int(cv[unk].sum())}


def nll_by_token_type(engine: str, test: str, tau: float) -> dict:
    """Mean −log q(y) on the known and the <unk> tokens at temperature τ."""
    R = rows(engine, test)
    z, unk = st.temper(R.Z, tau)[R.ctx, R.y], R.y == _unk(R)
    return {"known": float(-z[~unk].mean()), "unk": float(-z[unk].mean())}


def edgar_tau_reading(engine: str = "trigram") -> dict:
    """What τ* does and does not do to top-label calibration on EDGAR: the τ at which mean confidence equals accuracy,
    the 15-bin occupancy at τ*, the ECE there, and the smallest ECE on a grid of τ from 0.5 to 12."""
    R = rows(engine, "edgar")
    acc = float(stats_at(engine, "edgar", 1.0).correct.mean())
    match = optimize.brentq(lambda t: np.exp(st.temper(R.Z, t).max(axis=1))[R.ctx].mean() - acc, 1.0, 3.0, xtol=1e-4)
    s = stats_at(engine, "edgar", tau_star(engine, "edgar"))
    grid = [ece_hat(*(lambda x: (x.conf, x.correct))(tau_stats(R, float(t))), 15) for t in np.geomspace(0.5, 12, 33)]
    return {"tau_conf_eq_acc": float(match), "bins_at_tau_star": reliability_bins(s.conf, s.correct, 15)[0].tolist(),
            "ece_at_tau_star": ece_hat(s.conf, s.correct, 15), "ece_min": float(min(grid))}


def coverage_ceiling(taus=tuple(np.geomspace(0.5, 10, 25))) -> dict:
    """The trigram's nucleus(0.9) on EDGAR across τ: its best coverage against the known share. <unk> is the least likely
    token of its row at every τ, so its mass is at most 1/V, and a prefix of mass ≥ 0.9 never needs it."""
    R = rows("trigram", "edgar")
    return {"best": max(tau_stats(R, float(t)).covered[0.9] for t in taus), "known_share": float(np.mean(R.y != _unk(R)))}


# --------------------------------------------------------------------------------------------
# 4. A family closed under scaling: the log-linear bigram.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogLinearConfig:
    lam: float  # the penalty (λ/2)‖W‖²


LAMBDAS = (1e-3, 1e-4, 1e-5)
LAMBDA_GRID = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7)


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


def bigram_temperatures(lam: float) -> dict:
    """τ* on training and on every test set: Theorem 3 pins only the first."""
    W = loglinear_bigram(LogLinearConfig(lam))
    return {name: bigram_temperature(W, name) for name in ("train",) + TEST_SETS}


DECODER_SEEDS = st.DECODER_SEEDS


def decoder_train_tau(name: str) -> float:
    """τ* of one training seed of the decoder on every second training sentence."""
    R = _decoder_rows(sentences("train_half"), st.engine(name))
    return 1 / fit_beta(R.Z, R.n_u, R.s_y())


def decoder_gradient_norm(name: str = "decoder") -> tuple[float, float]:
    """(‖∇L‖, the drop in L from one gradient step of size 0.5), L the mean NLL over every training window. Neither is
    zero, so 1,200 steps of Adam did not end at a stationary point."""
    lm = st.engine(name)
    stream, starts = bs.sentence_stream()
    X = stream[starts[:, None] + np.arange(bs.DEC_CTX + 1)]
    P32 = {k: v.astype(np.float32) for k, v in lm.P.items()}
    b32 = [{s: {k: v.astype(np.float32) for k, v in blk[s].items()} for s in blk} for blk in lm.blocks]
    loss, G, Gb = sl.decoder_loss_grad(P32, b32, X[:, :-1], X[:, 1:])
    grads = [G[k] for k in P32] + [Gb[i][s][k] for i, blk in enumerate(b32) for s in blk for k in blk[s]]
    norm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    for k in P32:
        P32[k] = P32[k] - 0.5 * G[k]
    for i, blk in enumerate(b32):
        for s in blk:
            for k in blk[s]:
                blk[s][k] = blk[s][k] - 0.5 * Gb[i][s][k]
    return norm, float(loss) - float(sl.decoder_loss_grad(P32, b32, X[:, :-1], X[:, 1:])[0])


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


def fact_conf(p: st.Probe) -> float:
    return math.exp(st.temper(p.row[p.answers], 1.0).max())


def fact_confidence_by_count() -> dict:
    """Training count → mean confidence, plus the lowest confidence among probes seen twice or more, the highest among
    the rest, and the probability that a once-seen probe outranks a never-seen one."""
    ps = st.probes()
    conf = {c: [fact_conf(p) for p in ps if p.count == c] for c in sorted({p.count for p in ps})}
    a, b = conf[1], conf[0]
    return {"mean": {c: float(np.mean(v)) for c, v in conf.items()},
            "seen_min": min(fact_conf(p) for p in ps if p.regime == "seen"),
            "rest_max": max(fact_conf(p) for p in ps if p.regime != "seen"),
            "once_over_never": float(stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)))}


@lru_cache(maxsize=1)
def fact_row_labels() -> np.ndarray:
    """A label per probe: probes share one iff the model gives them the same row."""
    keys = [np.round(p.row, 9).tobytes() for p in st.probes()]
    return np.unique(keys, return_inverse=True)[1].ravel()


def fact_world() -> World:
    """Theorem 1's world on the probes: each probe a context of equal weight, the knowledge base's answer as the truth
    (a point mass, so the irreducible term is 0), and the forecast the row restricted to the relation's answers. The
    columns are the union of the two relations' answers."""
    ps = st.probes()
    cols = np.unique(np.concatenate([p.answers for p in ps]))
    F, P = np.zeros((len(ps), len(cols))), np.zeros((len(ps), len(cols)))
    labels = fact_row_labels()
    first = {}
    for i, p in enumerate(ps):
        F[i, np.searchsorted(cols, p.answers)] = np.exp(first.setdefault(labels[i], st.temper(p.row[p.answers], 1.0)))
        P[i, np.searchsorted(cols, p.correct)] = 1.0
    return World(np.full(len(ps), 1 / len(ps)), P, F)


def fact_rows_summary() -> dict:
    """How the probes sit on rows: the row count, the rows that mix regimes, the never-seen probes and their rows."""
    ps, labels = st.probes(), fact_row_labels()
    reg = np.array([p.regime for p in ps])
    w = fact_world()
    d = log_decomposition(w)
    gap = -np.log(recalibrated(w)[w.P > 0])  # with a point-mass truth, KL(P ‖ C) is −log C(the right answer)
    return {"rows": int(labels.max()) + 1, "mixed": sum(len(set(reg[labels == k])) > 1 for k in np.unique(labels)),
            "never": int((reg == "unseen").sum()), "never_rows": len(set(labels[reg == "unseen"])),
            "refinement_never": float(gap[reg == "unseen"].sum() / len(ps)), **d}


@lru_cache(maxsize=1)
def _fact_nll_grid() -> np.ndarray:
    """(probes, len(FIN_TAUS)) −log P(KB answer) at every grid temperature."""
    return np.array([[fact_stats((p,), t)["nll"] for t in FIN_TAUS] for p in st.probes()])


def fact_leave_one_row_out() -> dict:
    """The grid τ fit on every probe off a row, scored on that row's probes: pooled, and within the row's regime."""
    G, labels = _fact_nll_grid(), fact_row_labels()
    reg = np.array([p.regime for p in st.probes()])
    pooled, regime = np.zeros(len(G)), np.zeros(len(G))
    for i in range(len(G)):
        off = labels != labels[i]
        pooled[i] = G[i, G[off].mean(axis=0).argmin()]
        same = off & (reg == reg[i])
        regime[i] = G[i, G[same].mean(axis=0).argmin()] if same.any() else G[i, FIN_TAUS.index(1.0)]
    return {"pooled": float(pooled.mean()), "regime": float(regime.mean())}


def fact_by_relation(g: str, k: int) -> tuple[st.Probe, ...]:
    """One regime's probes for one relation, named by its answer count: 6 sectors or 13 cities."""
    return tuple(p for p in fact_regime(g) if len(p.answers) == k)


def fact_grid_tau(probes) -> float:
    """The grid τ of least mean NLL on a set of probes."""
    return float(FIN_TAUS[int(np.argmin([fact_stats(probes, t)["nll"] for t in FIN_TAUS]))])


def never_seen_past_the_dial() -> dict:
    """Never-seen probes at τ = 10, the top of the grid, against the uniform forecast and chance; and the sector probes'
    L(β) on the whole line: these logits are finite, so L is convex there and its minimizer can be negative."""
    ps = fact_regime("unseen")
    s = fact_stats(ps, 10.0)
    ks = np.array([len(p.answers) for p in ps])
    sec = fact_by_relation("unseen", 6)
    Z = np.array([_fact_logits(p)[0] for p in sec])
    n_u, s_y = np.ones(len(sec)), float(sum(z[j] for z, j in map(_fact_logits, sec)))
    r = optimize.minimize_scalar(lambda b: temperature_objective(Z, n_u, s_y, b)[0], bounds=(-3, 8), method="bounded",
                                 options={"xatol": 1e-6})
    names = st.fact_model().alphabet
    regimes_of = {a: Counter(p.regime for p in st.probes() if len(p.answers) == 6 and p.answer == a)
                  for a in ("technology", "industrials")}
    return {"favorites": Counter(names[p.answers[p.row[p.answers].argmax()]] for p in sec), "regimes_of": regimes_of,
            "conf": s["conf"], "acc": s["acc"], "nll": s["nll"], "uniform": float(np.log(ks).mean()),
            "chance": float(np.mean(1 / ks)), "sectors": len(sec), "sector_rows": len({z.tobytes() for z in Z}),
            "sector_hits": int(fact_stats(sec, 1.0)["acc"] * len(sec) + 0.5), "beta": float(r.x), "beta_nll": float(r.fun),
            "dL0": temperature_objective(Z, n_u, s_y, 0.0)[1], "log6": math.log(6)}


def count_aware_nll() -> dict:
    """Pooled NLL of the KB answer at τ = 1, at the one pooled τ*, and with each regime at its own τ*: all in sample."""
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
    for e, t in (("trigram", "continuation"), ("trigram", "edgar"), ("decoder", "edgar")):
        out[f"{e}_{t}_gain_random_mean"], out[f"{e}_{t}_gain_random_sd"] = random_split_gains(e, t)
        out[f"{e}_{t}_unk_map_gain"] = unk_map_gain(e, t)
        out[f"{e}_{t}_unk_map_random_mean"], out[f"{e}_{t}_unk_map_random_sd"] = random_split_gains(e, t, "unk")
        for k, v in known_summary(e, t).items():
            out[f"{e}_{t}_known_{k}"] = v
    out["good_turing_unseen_mass"] = ng.good_turing_unseen_mass()
    for name in DECODER_SEEDS:
        out[f"{name}_train_tau_star"] = decoder_train_tau(name)
    out["decoder_train_tau_star"] = tau_star("decoder", "train_half")
    out["trigram_continuation_kendall_tau2"] = confidence_kendall("trigram", "continuation", 2.0)
    out["trigram_continuation_kendall_tau2_tokens"] = confidence_kendall("trigram", "continuation", 2.0, tokens=True)
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
    for k, v in fact_leave_one_row_out().items():
        out[f"fact_nll_leave_one_row_{k}"] = v
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
    expected logit equals the observed one (|L'| < 1e-10), which by convexity is the global optimum. On the two EDGAR sets
    alone, Newton is also checked against scipy's bounded search (each evaluation on a large set is expensive, and
    convexity already settles the rest)."""
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
    overconfident and every bin above 2/3 is underconfident, and the top bin holds the most tokens."""
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
    """Topic 14's decoder (parameter-free final LayerNorm, tied unembedding scaled by a fixed 1/√d) has no parameter
    that scales its logits, and 1,200 steps of Adam leave it off a stationary point: the full training gradient is not
    zero, and one step along it lowers the training loss. Both of Thm 3's hypotheses fail, and on every second training
    sentence its τ* is 0.937–0.943 across #14's four seeds."""
    t = tau_star("decoder", "train_half")
    assert 0.9 < t < 0.97
    taus = [decoder_train_tau(n) for n in DECODER_SEEDS]
    assert abs(taus[0] - t) < 1e-12 and 0.936 < min(taus) and max(taus) < 0.944
    assert len(sentences("train_half")) == (len(ng.corpus()["train"]) + 1) // 2
    norm, drop = decoder_gradient_norm()
    assert norm > 0.01 and drop > 1e-4


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
    """Real filings, all tokens: both engines read overconfident (confidence above accuracy, τ* > 1.5, a positive
    cross-fitted gain); the trigram's nucleus(0.9) covers under half of the tokens and still does at its τ*; and at its
    NLL-optimal temperature the trigram's mean confidence falls below its accuracy. See test_ece_barely_moves_past_tau_star
    for what that does and doesn't mean, and test_edgar_is_mostly_one_token for where the reading comes from."""
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
    to within 0.05 of the mean log answer count, the uniform forecast's loss, and stays above it);
    in sample, one pooled temperature recovers under 0.02 nats while a temperature per regime recovers more than 0.4."""
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


def test_twin_floor_grows_like_root_b():
    """Thm 2(b) on the corpus: the twin's mean is about 0.25·√(B/n) at every lab bin count, so it grows like √B. The
    trigram's plug-in ECE rises 0.019 from one bin to 60, and the twin's floor accounts for 0.010 of that: part (a) is
    about the population ECE, so it does not predict the plug-in's rise, and bin counts 10 and 15 aren't nested anyway."""
    s = stats_at("trigram", "continuation", 1.0)
    m = twin_ece(s.conf, LAB_BINS, TWIN_R, np.random.default_rng([SEED, TWIN_STREAM, 0, 0])).mean(axis=0)
    c = m / np.sqrt(np.array(LAB_BINS) / len(s.conf))
    assert c.min() > 0.2 and c.max() < 0.3
    assert 0.018 < ece_hat(s.conf, s.correct, 60) - ece_hat(s.conf, s.correct, 1) < 0.02 and 0.0095 < m[-1] - m[0] < 0.0105
    assert 15 % 10 != 0


def test_top_bin_is_not_sentence_openers():
    """The top bin of the in-distribution trigram holds the generator's fixed phrases, wherever they fall: 55% of its
    tokens are at positions 0–5, against 53% of all tokens, and only 24% of a sentence's first three tokens land in it,
    against 33% overall."""
    t = top_bin_positions()
    assert t["n"] == 7652 and t["acc"] > 0.999
    assert 0.54 < t["top_at_0_5"] < 0.56 and 0.52 < t["all_at_0_5"] < 0.54
    assert 0.23 < t["first3_in_top"] < 0.25 and 0.32 < t["all_in_top"] < 0.34


def test_discounts_explain_half_the_shape():
    """Refit with scaled discounts, scored on the same tokens per region. The gap above 2/3 flips sign with D (−0.006 at a
    tenth of Ney's, +0.012 at Ney's). The overconfidence between 1/3 and 0.6 is there at every D and shrinks as D grows
    (−0.070 to −0.023), so the discounts relieve it. Scale 1 is the page's own trigram, and τ* falls as D grows."""
    d = discount_shape()
    fs = sorted(d)
    assert abs(d[1.0][0] - tau_star("trigram", "continuation")) < 1e-9
    assert all(d[a][0] > d[b][0] for a, b in zip(fs, fs[1:]))
    assert all(d[f][1] < 0 for f in fs) and all(d[a][1] < d[b][1] for a, b in zip(fs, fs[1:]))
    assert d[0.1][2] < -0.005 and d[1.0][2] > 0.01 and round(d[0.1][1], 3) == -0.07 and round(d[1.4][1], 3) == -0.023


def test_trigram_temperature_is_a_balance():
    """Theorem 3 doesn't apply to the trigram, which is no likelihood optimum and has no scaling direction, and 1.003 is a
    balance: on its known words alone τ* is 0.96, and <unk>, which it almost never forecasts, pulls the fit back up."""
    assert abs(tau_star("trigram", "continuation") - 1) < 0.005
    assert 0.955 < known_summary("trigram", "continuation")["tau_star"] < 0.965


def test_edgar_is_mostly_one_token():
    """The known-word cut. 478 of the trigram's 830 EDGAR tokens are <unk>, which is last in every row, with P under 1 in
    20,000, 0 hits and 0 of 478 covered. On the other 352: confidence 0.146, accuracy 0.139, τ* 1.10, coverage 0.841 against
    kept 0.904. Not calibrated: ECE 0.118 against a twin's 0.037, since the 50 tokens above 0.2 hit 7 times. The decoder's
    trouble is real: given a known word, confidence 0.383, accuracy 0.095, τ* 3.23, coverage 0.321."""
    u = unk_summary("trigram", "edgar")
    assert u["tokens"] == 478 and len(rows("trigram", "edgar").y) == 830 and u["max_p"] < 1 / 20000
    assert u["hits"] == 0 and u["covered"] == 0
    assert all(unk_summary("trigram", t)["last_in_every_row"] for t in TEST_SETS)
    k = known_summary("trigram", "edgar")
    assert k["n"] == 352 and abs(k["conf"] - k["acc"]) < 0.01 and 1.09 < k["tau_star"] < 1.11
    assert 0.84 < k["cover90"] < 0.842 and 0.903 < k["kept90"] < 0.905
    assert k["ece15"] > 3 * k["twin15"] and 0.117 < k["ece15"] < 0.119 and 0.035 < k["twin15"] < 0.039
    R = known_rows("trigram", "edgar")
    s = tau_stats(R, 1.0)
    high = s.conf >= 0.2
    assert high.sum() == 50 and s.correct[high].sum() == 7
    d = known_summary("decoder", "edgar")
    assert d["n"] == 296 and 0.38 < d["conf"] < 0.385 and 0.09 < d["acc"] < 0.1 and 3.2 < d["tau_star"] < 3.25
    a, b = nll_by_token_type("trigram", "edgar", 1.0), nll_by_token_type("trigram", "edgar", tau_star("trigram", "edgar"))
    assert b["known"] > a["known"] + 0.6 and b["unk"] < a["unk"] - 2


def test_coverage_cannot_pass_the_known_share():
    """Temperature keeps each row's order (Prop 2), a nucleus keeps a sorted prefix, and <unk> is last, so its tempered
    mass is at most 1/V and a prefix of mass ≥ 0.9 never needs it. Coverage stays at or below the known share, 0.424."""
    c = coverage_ceiling()
    assert c["best"] <= c["known_share"] and 0.42 < c["best"] and round(c["known_share"], 3) == 0.424
    assert 1 / len(bs.trigram().alphabet) < 1 - 0.9


def test_a_map_of_the_forecast_beats_temperature_by_six():
    """Cor 1's bound from temperature is far from tight. g_α(F) = (1 − α)F + α·e_<unk> is one parameter and a function of
    the forecast alone, and it recovers 5.5 nats on trigram EDGAR (temperature 0.89), 1.75 on decoder EDGAR (0.71), and
    0.173 on the continuation (temperature about 0), where α = 0.0275 is the <unk> share and Good–Turing's N₁/N = 0.0271
    forecasts it from training alone. On the decoder's in-distribution text it recovers nothing."""
    for e, t, lo, hi in (("trigram", "edgar", 5.3, 5.7), ("decoder", "edgar", 1.6, 1.9), ("trigram", "continuation", 0.16, 0.18)):
        m, sd = random_split_gains(e, t, "unk")
        assert lo < m < hi and sd < 0.2 and lo < unk_map_gain(e, t) < hi
        assert m > random_split_gains(e, t)[0]
    g = unk_map_stats("trigram", "continuation")
    assert 0.027 < g["alpha"] < 0.028 and abs(ng.good_turing_unseen_mass() - 0.0271) < 5e-5
    assert unk_map_gain("decoder", "continuation") < 1e-3
    h = unk_map_stats("trigram", "edgar")
    assert 0.57 < h["alpha"] < 0.58 and 3.3 < h["nll"] < 3.32 and abs(h["cover"] - h["kept"]) < 0.02


def test_cross_fit_counts_tokens():
    """The even and odd folds of EDGAR differ in size, so their unweighted mean (0.886) would sit above the in-sample
    gain (0.885), which a held-out fit shouldn't manage. Weighted by tokens it is 0.883; 20 random halves give
    0.890 ± 0.025, trigram, and 0.712 ± 0.062, decoder; on the continuation the trigram's random halves average zero."""
    R = rows("trigram", "edgar")
    inside = stats_at("trigram", "edgar", 1.0).nll - stats_at("trigram", "edgar", tau_star("trigram", "edgar")).nll
    assert 0.88 < cross_fit_gain("trigram", "edgar") < inside < 0.886
    folds = [(np.flatnonzero(R.sent % 2 == k), np.flatnonzero(R.sent % 2 != k)) for k in (0, 1)]
    assert len(folds[0][1]) != len(folds[1][1]) and np.mean([_temperature_gain(R, f, e) for f, e in folds]) > inside
    m, sd = random_split_gains("trigram", "edgar")
    assert round(m, 2) == 0.89 and 0.02 < sd < 0.03
    m, sd = random_split_gains("decoder", "edgar")
    assert round(m, 2) == 0.71 and 0.05 < sd < 0.07
    assert abs(random_split_gains("trigram", "continuation")[0]) < 1e-4


def test_ece_barely_moves_past_tau_star():
    """On trigram EDGAR the NLL-optimal 2.366 overshoots mean confidence only: mean confidence equals accuracy near
    τ = 1.40. ECE is within 0.001 of its smallest value on the grid at τ*, because 820 of 830 tokens sit in the bottom
    bin, so ECE there is the mean gap."""
    r = edgar_tau_reading()
    assert 1.39 < r["tau_conf_eq_acc"] < 1.41 and r["bins_at_tau_star"][0] == 820
    assert r["ece_at_tau_star"] - r["ece_min"] < 0.001


def test_decoder_held_gap_is_unk_on_top():
    """Nearly all of the decoder's underconfidence on held sits on the rows whose top label is <unk> (3,649 tokens,
    confidence 0.456, accuracy 0.785); rows with a word on top read 0.686 and 0.693. <unk> rises from 10.3% of the
    decoder's continuation tokens to 16.0% of held."""
    b = by_top_label("decoder", "held")
    assert b["unk"][0] == 3649 and b["unk"][2] - b["unk"][1] > 0.3 and abs(b["word"][2] - b["word"][1]) < 0.01
    share = {t: float(np.mean(rows("decoder", t).y == bs.decoder().unk)) for t in ("continuation", "held")}
    assert round(share["continuation"], 3) == 0.103 and round(share["held"], 3) == 0.16


def test_confidence_tracks_the_count():
    """The fact model's confidence does record how often it saw a fact, down to once. Mean confidence rises with the
    count, every probe seen twice or more has confidence ≥ 0.677, and every other ≤ 0.505. Only once against never can't
    be separated: a once-seen probe outranks a never-seen one with probability 0.516."""
    c = fact_confidence_by_count()
    m = list(c["mean"].values())
    assert all(a < b for a, b in zip(m, m[1:])) and [round(x, 3) for x in m] == [0.303, 0.332, 0.705, 0.807, 0.870, 0.895]
    assert c["seen_min"] > 0.677 and c["rest_max"] < 0.505 and 0.5 < c["once_over_never"] < 0.53


def test_fact_regime_is_a_function_of_the_row():
    """Thm 1 on the probes, each a context, the knowledge base as truth: 1.623 = 0 + 0.736 refinement + 0.887 reliability.
    111 probes sit on 42 rows and no row mixes regimes, so the regime is a function of the row, a temperature per regime
    is a map of the forecast, and its gain falls under Cor 1. The refinement gap is the company: all of it comes from
    the 48 never-seen probes, which share 3 rows with different right answers."""
    r = fact_rows_summary()
    assert r["rows"] == 42 and r["mixed"] == 0 and r["never"] == 48 and r["never_rows"] == 3
    assert r["irreducible"] == 0 and abs(r["loss"] - r["refinement_gap"] - r["reliability"]) < 1e-12
    assert abs(r["refinement_never"] - r["refinement_gap"]) < 1e-12
    assert (round(r["loss"], 3), round(r["refinement_gap"], 3), round(r["reliability"], 3)) == (1.623, 0.736, 0.887)
    c = count_aware_nll()
    assert abs(r["loss"] - c["tau1"]) < 1e-12 and c["tau1"] - c["per_regime"] < r["reliability"]


def test_one_temperature_fails_off_its_rows():
    """Leave one row out and the pooled temperature does worse than τ = 1 (1.769 against 1.623), while a temperature per
    regime holds up (1.130): the pooled in-sample 1.613 was flattery."""
    o, c = fact_leave_one_row_out(), count_aware_nll()
    assert o["pooled"] > c["tau1"] > c["pooled"] and round(o["pooled"], 3) == 1.769 and round(o["regime"], 3) == 1.13


def test_never_seen_is_past_the_end_of_the_dial():
    """τ = 10 is the grid's edge, not a calibration. Never-seen probes there still lose to uniform (2.056 against 2.017),
    and accuracy (0.0625) is below chance (0.14), so no positive temperature calibrates them: mean confidence is at least
    chance at every τ. On their 34 sector probes, on 2 rows, the top answer is right once, and the convex L has its
    minimum at β = −1.14 (1.555 against log 6 = 1.792). That is a selection effect: the rows' favorites are industrials
    and technology, the sectors the model has seen most, so a company it hasn't seen is rarely in one (every technology
    probe's context occurred in training, and 12 of the 14 industrials probes')."""
    n = never_seen_past_the_dial()
    assert n["nll"] > n["uniform"] and n["acc"] < n["chance"] < n["conf"]
    assert n["sectors"] == 34 and n["sector_rows"] == 2 and n["sector_hits"] == 1 and n["dL0"] > 0.4
    assert -1.2 < n["beta"] < -1.1 and 1.55 < n["beta_nll"] < 1.56 < n["log6"]
    assert n["favorites"] == {"industrials": 20, "technology": 14}
    assert n["regimes_of"]["technology"]["unseen"] == 0 and n["regimes_of"]["industrials"]["unseen"] == 2


def test_once_hides_two_relations():
    """"Seen once" pools sectors (21 probes, accuracy 0.524, grid τ* 0.447) and cities (12, accuracy 1.0, grid τ* at the
    edge), so its best τ, 0.260, fits neither."""
    sec, cit = fact_by_relation("once", 6), fact_by_relation("once", 13)
    assert len(sec) == 21 and len(cit) == 12 and fact_stats(cit, 1.0)["acc"] == 1.0
    assert round(fact_stats(sec, 1.0)["acc"], 3) == 0.524
    assert round(fact_grid_tau(sec), 3) == 0.447 and fact_grid_tau(cit) == FIN_TAUS[0]
    assert fact_tau_star("once") < 0.27 < fact_grid_tau(sec)


def test_reordering_by_tokens():
    """Kendall's 0.860 is over distinct rows; over tokens it is about 0.81. A rule that keeps the top fifth by confidence
    keeps 91.5% of its tokens when τ goes from 1 to 2, and its accuracy goes from 1.000 to 0.997."""
    assert round(confidence_kendall("trigram", "continuation", 2.0), 3) == 0.860
    assert round(confidence_kendall("trigram", "continuation", 2.0, tokens=True), 2) == 0.81
    t = top_fifth("trigram", "continuation", 2.0)
    assert round(t["overlap"], 3) == 0.915 and t["acc1"] > t["acc2"] and round(t["acc2"], 3) == 0.997


def test_held_out_temperatures_are_not_pinned():
    """Thm 3 at all ten penalties, and what it doesn't say. Training τ* stays below 1 and climbs as λ falls. On the
    continuation τ* crosses 1 near λ = 1e-6, on EDGAR between 1e-4 and 3e-5, and on held it never crosses on this grid."""
    prev = 0.0
    for lam in LAMBDA_GRID:
        W = loglinear_bigram(LogLinearConfig(lam))
        d, target = bigram_dbeta(W), -lam * float((W**2).sum())
        assert abs(d - target) < 1e-5 * abs(target)
        t = bigram_temperatures(lam)
        assert prev < t["train"] < 1 and t["held"] < 1
        prev = t["train"]
    t = {lam: bigram_temperatures(lam) for lam in (1e-4, 3e-5, 3e-6, 1e-6)}
    assert t[1e-4]["edgar"] < 1 < t[3e-5]["edgar"] and t[3e-6]["continuation"] < 1 < t[1e-6]["continuation"] < 1.01


def test_vocabularies():
    """The event spaces: the trigram's 3,427 outcomes are 3,425 word types plus </s> and <unk>; the decoder's 512 are 509
    words plus <s>, </s>, and <unk>, and <s> is never scored."""
    a, d = bs.trigram().alphabet, bs.decoder_vocab()
    assert len(a) == 3427 and len([w for w in a if w not in (EOS, UNK)]) == 3425 and BOS not in a
    assert len(d) == 512 and tuple(d[-3:]) == (BOS, EOS, UNK)
    assert bs.decoder().bos not in rows("decoder", "held").y and np.isneginf(rows("decoder", "held").Z[:, bs.decoder().bos]).all()


def test_test_set_anatomy():
    """What the two ends of the table are. 486 of the continuation's 2,043 sentences repeat a training sentence word for
    word (20% of its tokens), and EDGAR is 33 sentences. The trigram meets 5,462 contexts on the continuation; 2,950 were
    seen in training, on 2,480 distinct rows, so most seen contexts have a row of their own, and not all."""
    r = repeated_sentences()
    assert r["sentences"] == 486 and r["of"] == 2043 and 0.2 < r["token_share"] < 0.21 and len(sentences("edgar")) == 33
    assert seen_context_rows() == {"contexts": 5462, "seen": 2950, "seen_rows": 2480}


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
