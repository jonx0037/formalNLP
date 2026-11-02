"""Sampling: temperature, top-k, and nucleus — the reference implementation for formalNLP's fifteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Sampling is exact; search is not. Ancestral sampling returns each finite string with exactly its model probability,
   and halts almost surely iff the model is consistent (Thm 1). The Gumbel-max trick makes a draw the argmax of
   perturbed log-probabilities, so greedy decoding is that loop with the noise off (Prop 1).
2. Temperature as a Gibbs distribution: softmax(z/τ) maximizes ⟨q, z⟩ + τH(q); its entropy rises with τ at rate
   Var(z)/τ³; τ → 0 and τ → ∞ give the uniform distributions on the argmax set and on the support (Thm 2). Tempering
   divides every least-squares Zipf slope by τ exactly, and moves the intercept too (Prop 2).
3. Local against global temperature: the per-step tempered sampler is p(y)^{1/τ} divided by the product of the steps'
   normalizers, so it equals the tempered sequence distribution only when that product is constant. It upweights
   strings in proportion to exp((1/τ − 1)·Σ_t H_{1/τ}(p(·|h_t))), the accumulated Rényi entropy of their prefixes, and
   as τ → 0 on a finite tree with a unique mode and unique argmaxes along greedy's path it lands on greedy's string,
   not the mode. For τ > 1 the tempered sequence distribution can fail to exist (Thm 3).
4. Truncation: the nucleus is a minimum-cardinality set of mass ≥ p. Keeping a set of mass m costs TV = 1 − m and
   KL = −log m per step. Over a sequence, KL(q ‖ p) = E_q Σ_t −log m(h_t) exactly, TV(q, p) ≤ E_q Σ_t (1 − m(h_t)), and
   the probability that pure sampling never leaves the kept sets is E_q Π_t m(h_t) (Thm 4).
5. Truncation as desmoothing: one threshold recovers the support of a uniform mixture with λ < 1 (Prop 3). In
   interpolated Kneser–Ney the seen set is recovered exactly from the model's own decomposition at a context with
   c(h) > 0, but no single setting of any rule recovers it on every row (Prop 4).
6. Typicality: E NLL = E Σ_t H(h_t), and the gap is a martingale, so a sample's NLL tracks its own path's entropy
   (Prop 5). Prop 5 places the samples and says nothing about the text: the mode, sample, and text averages are three
   different pools, and on the text's weights the comparison depends on the statistic (pools).
7. Quality against diversity on the corpus: Topic 3's trigram and Topic 14's decoder under every rule, with the
   per-word NLL under an oracle KN model fit to five times more text from the same generator. Sampled claims that
   move with the stream are read over STREAMS, and the decoder's over its training seeds.
8. Finance case study, boilerplate against fabrication: an order-5 Kneser–Ney model as a fact model, probed with
   "<company> operates in the" and similar frames and scored exactly against the knowledge base. Contexts seen twice or
   more, once, and never respond to temperature and truncation in different ways, and a never-seen context is not a
   never-seen fact: "seen" counts the four-word context.

Cross-checks: exact enumeration on random trees for every identity; the Gibbs variational principle against
scipy.optimize; the Kneser–Ney decomposition against the model's own probabilities.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/sampling-temperature-nucleus/sampling_temperature_nucleus.py
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
for _dep in ("beam-search-decoding", "zipf-heaps-vocabulary-statistics"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import beam_search_decoding as bs  # noqa: E402 — module objects: their test_* stay out of this harness
import zipf_heaps_vocabulary_statistics as zh  # noqa: E402

ng, pce, fc, ak = bs.ng, bs.pce, bs.fc, bs.ak

# ONE stream per random-instance claim and per sampled grid, each from default_rng([SEED, k]); k ≥ 1800 keeps this
# module's streams apart from Topics 5–14.
SEED = bs.SEED
BOS, EOS, UNK = bs.BOS, bs.EOS, bs.UNK
Prefix = tuple[int, ...]


def _r(x, nd: int = 3):
    return ak._r(x, nd)


def _lse(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log Σ exp along an axis, shifted by the max (rows always hold a finite entry here)."""
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _log(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(p)


def entropy_rows(lp: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each row of log-probabilities (−inf allowed)."""
    lp = np.atleast_2d(lp)
    p = np.exp(lp)
    return -np.sum(np.where(p > 0, p * np.where(np.isfinite(lp), lp, 0.0), 0.0), axis=1)


# --------------------------------------------------------------------------------------------
# 1. Sampling is exact: ancestral sampling and the Gumbel-max trick.
# --------------------------------------------------------------------------------------------


def tree_dist(lm: bs.TableLM, step=None) -> dict[Prefix, float]:
    """Every finished string of a finite tree with its probability under a per-step sampler: step(row) maps a
    next-token probability vector to the vector the sampler draws from (default: the model itself)."""
    out, stack = {}, [(lm.root, 1.0)]
    while stack:
        h, P = stack.pop()
        row = lm.rows[h] if step is None else step(lm.rows[h])
        for j in np.flatnonzero(row > 0):
            if j == lm.eos:
                out[h + (j,)] = out.get(h + (j,), 0.0) + P * row[j]
            else:
                stack.append((h + (j,), P * row[j]))
    return out


def sample_tree(lm: bs.TableLM, rng: np.random.Generator, n: int) -> Counter:
    """n ancestral draws from a finite tree, level by level: every draw at prefix h uses p(· | h)."""
    out: Counter = Counter()
    groups = {lm.root: n}
    while groups:
        nxt: dict[Prefix, int] = {}
        for h, m in groups.items():
            counts = rng.multinomial(m, lm.rows[h])
            for j in np.flatnonzero(counts):
                if j == lm.eos:
                    out[h + (j,)] += int(counts[j])
                else:
                    nxt[h + (j,)] = nxt.get(h + (j,), 0) + int(counts[j])
        groups = nxt
    return out


def gumbel_max(z: np.ndarray, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n draws of (argmax_w (z_w + G_w), max_w (z_w + G_w)) with G i.i.d. Gumbel(0, 1)."""
    s = z[None, :] + rng.gumbel(size=(n, len(z)))
    return s.argmax(axis=1), s.max(axis=1)


def stationary_lengths(lm: bs.StationaryLM, rng: np.random.Generator, n: int) -> np.ndarray:
    """|Y| (words before </s>) for n ancestral draws from a stationary model: the step at which </s> first comes up."""
    p_stop = math.exp(lm.row[lm.eos])
    return rng.geometric(p_stop, size=n) - 1


# --------------------------------------------------------------------------------------------
# 2. Temperature as a Gibbs distribution.
# --------------------------------------------------------------------------------------------


def temper(lp: np.ndarray, tau: float) -> np.ndarray:
    """softmax(z/τ) in log space, row-wise; −inf entries stay −inf (they are outside the support)."""
    x = np.asarray(lp, dtype=float) / tau
    return x - _lse(x, axis=-1, keepdims=True)


def gibbs_objective(q: np.ndarray, z: np.ndarray, tau: float) -> float:
    """⟨q, z⟩ + τH(q) — the negative free energy that softmax(z/τ) maximizes over the simplex."""
    nz = q > 0
    return float(q @ z + tau * -np.sum(q[nz] * np.log(q[nz])))


def entropy_derivative(z: np.ndarray, tau: float) -> float:
    """Thm 2(b): dH(p_τ)/dτ = Var_{p_τ}(z)/τ³."""
    p = np.exp(temper(z, tau))
    fin = np.isfinite(z)
    m = p[fin] @ z[fin]
    return float(p[fin] @ (z[fin] - m) ** 2 / tau**3)


def zipf_slope(lp: np.ndarray) -> float:
    """Least-squares slope of log p against log rank over a row's support (Topic 1's estimator)."""
    p = np.sort(np.exp(lp[np.isfinite(lp)]))[::-1]
    return zh.ols_loglog_slope(np.arange(1, len(p) + 1), p)


def zipf_crossing(s: float, n: int, tau: float) -> dict:
    """What tempering does to a Zipf(s) line besides its slope: log p at rank 1 before and after, and the rank where
    the two lines cross, which is where (1/τ − 1)·log p(r) = log Z_τ."""
    lp = np.log(zh.zipf_pmf(s, n))
    log_Z = float(_lse(lp / tau))
    return {"before": float(lp[0]), "after": float(temper(lp, tau)[0]),
            "cross": math.exp((float(lp[0]) - log_Z / (1 / tau - 1)) / s)}


# --------------------------------------------------------------------------------------------
# 3. Local against global temperature.
# --------------------------------------------------------------------------------------------


def local_tempered(lm: bs.TableLM, tau: float) -> dict[Prefix, float]:
    """q_τ: temper every next-token distribution, then sample ancestrally."""
    return tree_dist(lm, lambda row: np.exp(temper(_log(row), tau)))


def global_tempered(lm: bs.TableLM, tau: float) -> dict[Prefix, float]:
    """π_τ ∝ p(y)^{1/τ} over the tree's finished strings."""
    base = tree_dist(lm)
    w = {y: P ** (1 / tau) for y, P in base.items()}
    Z = sum(w.values())
    return {y: v / Z for y, v in w.items()}


def tv(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(y, 0.0) - b.get(y, 0.0)) for y in set(a) | set(b))


def argmax_ties(lm: bs.TableLM, h: Prefix) -> int:
    """How many tokens tie for the largest probability at prefix h (1 = a unique argmax)."""
    return int(np.sum(lm.rows[h] >= lm.rows[h].max() - 1e-12))


def renyi_path(lm: bs.TableLM, y: Prefix, tau: float) -> float:
    """Σ_t H_α(p(· | h_t)) along y's prefixes, α = 1/τ (Rényi entropy; H_α = log Σ p^α/(1 − α))."""
    a, tot = 1 / tau, 0.0
    for t in range(len(y)):
        row = lm.rows[lm.root + y[:t]]
        row = row[row > 0]
        tot += math.log(np.sum(row**a)) / (1 - a)
    return tot


def permutation_tree(rng: np.random.Generator, m: int, depth: int) -> bs.TableLM:
    """Every open prefix shorter than `depth` draws a permutation of one fixed vector over the m words (no </s>), and
    </s> is forced at depth. So every string multiplies `depth` copies of one Z_τ and the forced row's 1: the condition
    of Thm 3(a) holds, and local = global. The forced row is why "every row is a permutation of one vector" needs its
    exception: the last row of a same-length string is all </s>."""
    v = rng.dirichlet(np.full(m, 0.7))
    alphabet = (EOS,) + tuple(f"w{i}" for i in range(m))
    rows: dict[Prefix, np.ndarray] = {}
    frontier: list[Prefix] = [()]
    while frontier:
        h = frontier.pop()
        if len(h) == depth:
            rows[h] = np.eye(m + 1)[0]
            continue
        rows[h] = np.concatenate([[0.0], rng.permutation(v)])
        frontier += [h + (j,) for j in range(1, m + 1)]
    return bs.TableLM(alphabet, rows)


def welleck_ratio(k: int, tau: float, eps: float = 0.05) -> float:
    """The global tempered partition function of #14's stationary model (bs.welleck_model) is Σ_t r^t·ε^{1/τ} with
    r = k^{1 − 1/τ}(1 − ε)^{1/τ}: finite iff r < 1."""
    return k ** (1 - 1 / tau) * (1 - eps) ** (1 / tau)


def welleck_critical_tau(k: int, eps: float = 0.05) -> float:
    """The τ at which r = 1: above it the tempered sequence distribution does not exist."""
    return (math.log(k) - math.log(1 - eps)) / math.log(k)


# --------------------------------------------------------------------------------------------
# 4. Truncation.
# --------------------------------------------------------------------------------------------

RULES = ("top-k", "nucleus", "min-p", "eta", "typical")


def truncation_set(lp: np.ndarray, rule: str, param: float) -> np.ndarray:
    """The kept set S(h) of each row of log-probabilities (−inf = outside the support), as a boolean mask.

    top-k       the k most probable tokens (Fan et al. 2018)
    nucleus     the shortest prefix of the sorted order whose mass reaches p (Holtzman et al. 2020)
    min-p       {w : p_w ≥ ρ·max p} (Nguyen et al. 2025)
    eta         {w : p_w > min(η, √η·e^{−H})}, never empty (Hewitt et al. 2022)
    typical     tokens sorted by |−log p_w − H|, kept until their mass reaches the typical mass (Meister et al. 2023,
                in its greedy form)

    Ties in the sort order go to the lower token id (a stable sort), so top-k keeps exactly k tokens and the nucleus is
    one of the minimum-cardinality sets Thm 4(a) describes. A mass target counts as reached within 1e-12, so float
    round-off in a cumulative sum never adds a token."""
    lp = np.atleast_2d(np.asarray(lp, dtype=float))
    n, V = lp.shape
    fin = np.isfinite(lp)
    p = np.exp(lp)
    rows = np.arange(n)[:, None]
    keep = np.zeros((n, V), dtype=bool)

    def by_order(order: np.ndarray, count: np.ndarray) -> np.ndarray:
        out = np.zeros((n, V), dtype=bool)
        out[rows, order] = np.arange(V)[None, :] < count[:, None]
        return out

    if rule == "top-k":
        keep = by_order(np.argsort(-lp, axis=1, kind="stable"), np.full(n, int(param)))
    elif rule in ("nucleus", "typical"):
        if rule == "nucleus":
            order = np.argsort(-lp, axis=1, kind="stable")
        else:
            H = entropy_rows(lp)[:, None]
            order = np.argsort(np.where(fin, np.abs(-lp - H), np.inf), axis=1, kind="stable")
        c = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
        keep = by_order(order, (c < param - 1e-12).sum(axis=1) + 1)
    elif rule == "min-p":
        keep = p >= param * p.max(axis=1, keepdims=True)
    elif rule == "eta":
        H = entropy_rows(lp)[:, None]
        keep = p > np.minimum(param, math.sqrt(param) * np.exp(-H))
        keep[np.arange(n), lp.argmax(axis=1)] = True
    else:
        raise ValueError(rule)
    return keep & fin


def truncate(lp: np.ndarray, rule: str, param: float) -> np.ndarray:
    """p_S: the model restricted to the kept set and renormalized (log space)."""
    lp = np.atleast_2d(np.asarray(lp, dtype=float))
    q = np.where(truncation_set(lp, rule, param), lp, -np.inf)
    return q - _lse(q, axis=1, keepdims=True)


def decode_rows(lp: np.ndarray, setting: tuple[str, float]) -> np.ndarray:
    """The distribution a decoding setting samples from, row-wise: ("temperature", τ) or (rule, parameter)."""
    rule, x = setting
    return temper(np.atleast_2d(lp), x) if rule == "temperature" else truncate(lp, rule, x)


def kept_mass(lp: np.ndarray, rule: str, param: float) -> np.ndarray:
    lp = np.atleast_2d(lp)
    return np.where(truncation_set(lp, rule, param), np.exp(lp), 0.0).sum(axis=1)


def best_mass_of_size(p: np.ndarray, size: int) -> float:
    """Brute force: the largest mass of any `size`-subset (Thm 4(a)'s comparison)."""
    import itertools
    return max(sum(p[list(c)]) for c in itertools.combinations(range(len(p)), size))


def min_size_reaching(p: np.ndarray, target: float) -> int:
    """Brute force: the fewest tokens of any subset with mass ≥ target."""
    import itertools
    for size in range(1, len(p) + 1):
        if any(sum(p[list(c)]) >= target - 1e-12 for c in itertools.combinations(range(len(p)), size)):
            return size
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class SeqCost:
    kl: float  # KL(q ‖ p) by enumeration
    kl_identity: float  # E_q Σ_t −log m(h_t)
    tv: float  # TV(q, p) by enumeration
    tv_bound: float  # E_q Σ_t (1 − m(h_t))
    avoid: float  # P_p(the string never leaves the kept sets), by enumeration
    avoid_identity: float  # E_q Π_t m(h_t)


def sequence_cost(lm: bs.TableLM, rule: str, param: float) -> SeqCost:
    """Thm 4(c) on a finite tree, every quantity by exact enumeration."""
    p = tree_dist(lm)
    q: dict[Prefix, float] = {}
    e_log = e_one = e_prod = avoid = 0.0
    stack = [(lm.root, 1.0, 1.0, 0.0, 0.0, 1.0)]  # prefix, q(h), p(h), Σ −log m, Σ (1 − m), Π m
    while stack:
        h, Q, P, sl, so, sp = stack.pop()
        lp = _log(lm.rows[h])
        keep = truncation_set(lp, rule, param)[0]
        m = float(lm.rows[h][keep].sum())
        for j in np.flatnonzero(keep):
            qj = lm.rows[h][j] / m
            args = (h + (j,), Q * qj, P * lm.rows[h][j], sl - math.log(m), so + 1 - m, sp * m)
            if j == lm.eos:
                y, Qy, Py = args[0], args[1], args[2]
                q[y] = Qy
                e_log += Qy * args[3]
                e_one += Qy * args[4]
                e_prod += Qy * args[5]
                avoid += Py
            else:
                stack.append(args)
    kl = sum(Q * math.log(Q / p[y]) for y, Q in q.items())
    return SeqCost(kl, e_log, tv(q, p), e_one, avoid, e_prod)


# --------------------------------------------------------------------------------------------
# 5. Truncation as desmoothing.
# --------------------------------------------------------------------------------------------


def uniform_mixture(rng: np.random.Generator, V: int, support: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """(p*, p = (1 − λ)p* + λ/|V|) with p* a Dirichlet draw on `support` random tokens."""
    star = np.zeros(V)
    idx = rng.choice(V, size=support, replace=False)
    star[idx] = rng.dirichlet(np.ones(support))
    return star, (1 - lam) * star + lam / V


def kn_decomposition(vec: pce.KNVector, u: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P(· | u), its backoff part γ(u)·P(· | u⁻), the seen mask c(u w) > 0) for an interpolated KN context u, read
    from the model's own counts: P(w | u) = [max(c(uw) − D, 0) + freed(u)·P(w | u⁻)]/c(u·)."""
    m = vec.model
    full = vec(u)
    total = m.totals.get(u, 0)
    seen = np.zeros(len(vec.alphabet), dtype=bool)
    if not total:
        return full, full.copy(), seen
    backoff = m.freed[u] / total * vec(u[1:])
    seen[vec._rows[u][0]] = True
    return full, backoff, seen


def _trigram_context(lm: bs.TrigramLM, h: Prefix) -> tuple[str, ...]:
    return tuple(BOS if i == lm.bos else lm.alphabet[i] for i in h[-2:])


@lru_cache(maxsize=1)
def trigram_rows() -> tuple[np.ndarray, np.ndarray]:
    """(the trigram's sampling rows, its seen masks) at every prompt's root: 188 rows over V, <unk> excluded."""
    lm = bs.trigram()
    rows, seen = [], []
    for p in bs.prompts():
        h = lm.root(p)
        rows.append(lm.next_logp([h])[0])
        s = kn_decomposition(lm.vec, _trigram_context(lm, h))[2]
        s[lm.unk] = False
        seen.append(s)
    return np.array(rows), np.array(seen)


def precision_recall(keep: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise precision (kept tokens that were seen) and recall (seen tokens that were kept)."""
    tp = (keep & seen).sum(axis=1)
    return tp / np.maximum(keep.sum(axis=1), 1), tp / np.maximum(seen.sum(axis=1), 1)


# Lab and claim grids for the single-row rules.
ROW_GRID = {"top-k": (1, 2, 3, 5, 8, 12, 20, 40, 80, 160), "nucleus": (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
            "min-p": (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01), "eta": (0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4),
            "typical": (0.2, 0.5, 0.7, 0.9, 0.95)}


def desmoothing_table() -> dict:
    """Prop 4 on the trigram's 188 prompt rows: each rule's mean precision and recall against the seen set at every
    grid value, whether any value is perfect on every row, and the share of rows whose seen tokens all outrank their
    unseen ones (rows a per-row threshold could separate). Rows whose context never occurred have nothing to recover
    (the whole row is backoff) and are left out."""
    rows, seen = trigram_rows()
    has = seen.any(axis=1)
    rows, seen = rows[has], seen[has]
    out = {"n_rows": int(has.sum())}
    for rule, grid in ROW_GRID.items():
        pr = [precision_recall(truncation_set(rows, rule, x), seen) for x in grid]
        out[rule] = {"precision": [float(a.mean()) for a, _ in pr], "recall": [float(b.mean()) for _, b in pr],
                     "perfect": [bool(np.all(a == 1) and np.all(b == 1)) for a, b in pr]}
    p = np.exp(rows)
    lo = np.where(seen, p, np.inf).min(axis=1)
    hi = np.where(~seen & np.isfinite(rows), p, -np.inf).max(axis=1)
    out["separable"] = float(np.mean(lo > hi))
    out["seen_size"] = float(seen.sum(axis=1).mean())
    return out


# --------------------------------------------------------------------------------------------
# 6–7. Sampling the corpus engines.
# --------------------------------------------------------------------------------------------

N_PER_PROMPT = 8
ENGINE_STREAM = {"trigram": 1820, "decoder": 1821}
STREAMS = (0, 1, 2, 3, 4)  # stream s is default_rng([SEED, ENGINE_STREAM[engine] + 10·s]); the lab shows stream 0
DECODER_SEEDS = ("decoder", "decoder1", "decoder2", "decoder3")  # #14's SEEDS: "decoder" is seed 0
SETTINGS: tuple[tuple[str, float], ...] = (
    *(("temperature", t) for t in (0.4, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)),
    *(("top-k", k) for k in (5, 20, 80)),
    *(("nucleus", p) for p in (0.5, 0.8, 0.9, 0.95)),
    *(("min-p", r) for r in (0.2, 0.1, 0.05, 0.02)),
    *(("eta", e) for e in (3e-3, 1e-3, 3e-4)),
    *(("typical", m) for m in (0.2, 0.5, 0.9)),
)
PURE = ("temperature", 1.0)


@dataclass(frozen=True)
class Draws:
    prompt: np.ndarray  # prompt index of every draw
    tokens: tuple[Prefix, ...]  # generated tokens (ending in </s> when finished)
    finished: np.ndarray
    logp: np.ndarray  # Σ log p(y_t | h_t) under the model
    logq: np.ndarray  # Σ log q(y_t | h_t) under the sampler
    entropy: np.ndarray  # Σ H(p(· | h_t)) along the path


def _max_len(name: str, lm, root: Prefix) -> int:
    return bs.TRIGRAM_MAX_LEN if name == "trigram" else lm.max_len(root)


@lru_cache(maxsize=None)
def engine(name: str):
    """#14's two engines, plus "decoder1" to "decoder3": the same decoder trained from #14's other seeds."""
    return bs.DecoderLM(bs.DecoderConfig(seed=int(name[-1]))) if name[-1] in "123" else bs.engine(name)


@lru_cache(maxsize=None)
def draws(name: str, setting: tuple[str, float], stream: int = 0) -> Draws:
    """N_PER_PROMPT draws per prompt from one engine under one setting. Every setting restarts the engine's stream
    (common random numbers: settings differ by their rule, not by their luck), and a decoder's other training seeds
    share its stream. Each step draws one uniform per open string and inverts the sampler's CDF; the trigram transforms
    each Markov state once."""
    lm = engine(name)
    rng = np.random.default_rng([SEED, ENGINE_STREAM[name.rstrip("123")] + 10 * stream])
    roots = [lm.root(p) for p in bs.prompts() for _ in range(N_PER_PROMPT)]
    n = len(roots)
    seqs = [list(r) for r in roots]
    caps = np.array([_max_len(name, lm, r) for r in roots])
    done = np.zeros(n, dtype=bool)
    logp, logq, ent = np.zeros(n), np.zeros(n), np.zeros(n)
    cache: dict = {}
    open_ = np.arange(n)
    while len(open_):
        if name == "trigram":
            keys = [tuple(seqs[i][-2:]) for i in open_]
            for k in set(keys) - cache.keys():
                lp = lm.next_logp([k])[0]
                q = decode_rows(lp, setting)[0]
                cache[k] = (lp, q, np.cumsum(np.exp(q)), float(entropy_rows(lp)[0]))
            got = [cache[k] for k in keys]
        else:  # the decoder has no Markov state, but draws that share a prefix share a row
            keys = [tuple(seqs[i]) for i in open_]
            uniq = list(dict.fromkeys(keys))
            lp = lm.next_logp(uniq)
            q = decode_rows(lp, setting)
            H = entropy_rows(lp)
            cdf = np.cumsum(np.exp(q), axis=1)
            at = {k: r for r, k in enumerate(uniq)}
            got = [(lp[at[k]], q[at[k]], cdf[at[k]], float(H[at[k]])) for k in keys]
        u = rng.random(len(open_))
        for i, x, (lp_row, q_row, cdf, H) in zip(open_, u, got):
            j = min(int(np.searchsorted(cdf, x * cdf[-1], side="right")), len(cdf) - 1)
            seqs[i].append(j)
            logp[i] += lp_row[j]
            logq[i] += q_row[j]
            ent[i] += H
            if j == lm.eos:
                done[i] = True
        open_ = np.array([i for i in open_ if not done[i] and len(seqs[i]) - len(roots[i]) < caps[i]], dtype=int)
    toks = tuple(tuple(s[len(r):]) for s, r in zip(seqs, roots))
    return Draws(np.repeat(np.arange(len(bs.prompts())), N_PER_PROMPT), toks, done, logp, logq, ent)


# The oracle: Kneser–Ney on five times the training text, drawn from the training corpus's own generator.
ORACLE_DOCS = 1500
ORACLE_ORDER = 3


@lru_cache(maxsize=1)
def oracle() -> tuple[ng.KNModel, frozenset[str]]:
    """Interpolated KN (Topic 3's recursion, Ney's discounts) on ng.continuation_run(1500): the generator that wrote
    the training corpus, run on for five times as many documents, so its restaurants and vocabulary carry over. Words
    seen once in that text become <unk>, so an unknown word costs what a rare one does."""
    sents = ng.continuation_run(ORACLE_DOCS)["continuation"]
    c = Counter(w for s in sents for w in s)
    vocab = frozenset(w for w, k in c.items() if k >= 2)
    data = ng.with_unk(sents, vocab)
    alphabet = tuple(sorted(vocab | {EOS, UNK}))
    return pce.kn_fit_any(data, ORACLE_ORDER, alphabet), vocab


@lru_cache(maxsize=None)
def _oracle_prob(word: str, context: tuple[str, ...]) -> float:
    return ng.interpolated_kn_prob(word, context, oracle()[0])


def oracle_logp(prompt: tuple[str, ...], words: tuple[str, ...]) -> float:
    """log P_oracle(words + </s> | <s> <s> prompt), the continuation only."""
    vocab = oracle()[1]
    seq = [BOS] * (ORACLE_ORDER - 1) + [w if w in vocab else UNK for w in prompt]
    tot = 0.0
    for w in [w if w in vocab else UNK for w in words] + [EOS]:
        tot += math.log(_oracle_prob(w, tuple(seq[-(ORACLE_ORDER - 1):])))
        seq.append(w)
    return tot


EIGHT_K_TAIL = ("to", "disclose", "a", "material", "event")


def _loops(words: tuple[str, ...]) -> bool:
    return any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2))


@lru_cache(maxsize=None)
def metrics(name: str, setting: tuple[str, float], stream: int = 0) -> dict:
    """One point of the frontier: finished share; mean words of finished continuations; per-word NLL under the model
    and under the oracle (pooled over finished continuations, </s> included); distinct-2 over all continuations;
    distinct sentences; copy rate over finished strings (prompt + continuation is a training sentence); loop rate (a
    word three times in a row); share ending on the 8-K tail; per-word KL(q ‖ p), estimated as E_q Σ (log q − log p)
    over all draws. The NLLs, the distinct sentences, and the copy rate leave unfinished draws out, which matters at
    τ = 1.5."""
    lm, d = engine(name), draws(name, setting, stream)
    P, train = bs.prompts(), bs.training_sentences()
    fin = np.flatnonzero(d.finished)
    words = [tuple(lm.words(t[:-1] if d.finished[i] else t)) for i, t in enumerate(d.tokens)]
    n_tok = np.array([len(t) for t in d.tokens])
    bigrams = [b for w in words for b in zip(w, w[1:])]
    sents = [P[d.prompt[i]] + words[i] for i in fin]
    orc = sum(oracle_logp(P[d.prompt[i]], words[i]) for i in fin)
    return {
        "finished": float(d.finished.mean()),
        "length": float(np.mean([len(words[i]) for i in fin])),
        "nll": float(-d.logp[fin].sum() / n_tok[fin].sum()),
        "oracle_nll": float(-orc / n_tok[fin].sum()),
        "distinct2": len(set(bigrams)) / max(len(bigrams), 1),
        "distinct": len(set(sents)) / len(sents),
        "copied": float(np.mean([s in train for s in sents])),
        "loops": float(np.mean([_loops(w) for w in words])),
        "tail": float(np.mean([words[i][-5:] == EIGHT_K_TAIL for i in fin])),
        "kl": float((d.logq - d.logp).sum() / n_tok.sum()),
    }


def temperature_curve(name: str, stream: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The temperature settings' (model NLL, oracle NLL), coldest first: one curve on the frontier's axes."""
    temps = [metrics(name, s, stream) for s in SETTINGS if s[0] == "temperature"]
    return np.array([m["nll"] for m in temps]), np.array([m["oracle_nll"] for m in temps])


def frontier_offset(name: str, setting: tuple[str, float]) -> float | None:
    """How far a setting's oracle NLL sits above the temperature curve at the same model NLL (linear interpolation
    between the temperature settings); None outside the temperature curve's range."""
    x, y = temperature_curve(name)
    m = metrics(name, setting)
    if not x[0] <= m["nll"] <= x[-1]:
        return None
    return float(m["oracle_nll"] - np.interp(m["nll"], x, y))


def typicality(name: str) -> dict:
    """Prop 5 on pure samples: the mean of NLL − Σ_t H(h_t) over draws and its standard error (finished and
    unfinished draws alike: the martingale needs no stopping rule beyond the cap)."""
    d = draws(name, PURE)
    gap = -d.logp - d.entropy
    return {"gap": float(gap.mean()), "se": float(gap.std(ddof=1) / math.sqrt(len(gap)))}


@lru_cache(maxsize=1)
def mode_reference() -> dict:
    """#14's exact trigram modes (Dijkstra) on the frontier's axes: pooled per-word NLL under the trigram and under
    the oracle (</s> included). The decoder's exact modes cost ~6 s and are left to #14."""
    lm, P = bs.trigram(), bs.prompts()
    modes = bs.exact_modes("trigram")
    n = sum(len(m.tokens) for m in modes)
    orc = sum(oracle_logp(p, tuple(lm.words(m.tokens[:-1]))) for p, m in zip(P, modes))
    return {"nll": float(-sum(m.logp for m in modes) / n), "oracle_nll": float(-orc / n)}


@lru_cache(maxsize=None)
def heldout_reference(name: str) -> dict:
    """The held-out MD&A sentences the prompts open, on the frontier's axes: pooled per-word NLL of their
    continuations (</s> included) under the engine and under the oracle. Only sentences the engine can score count:
    every word in its vocabulary and, for the decoder, inside its 32 positions. On the trigram this is #14's
    reference_nll."""
    lm = engine(name)
    want, tot, orc, n = set(bs.prompts()), 0.0, 0.0, 0
    for d in fc.generate_corpus(100, 1):
        if d.doc_type != "mdna":
            continue
        for s in ng.split_sentences(d.text):
            if s[:3] not in want or not all(w in lm.index for w in s):
                continue
            root, toks = lm.root(s[:3]), tuple(lm.index[w] for w in s[3:]) + (lm.eos,)
            if name == "decoder":
                if len(toks) > lm.max_len(root):
                    continue
                tot += lm.string_logp(root, toks)
            else:
                h = root
                for j in toks:
                    tot += lm.next_logp([h])[0][j]
                    h += (j,)
            orc += oracle_logp(s[:3], s[3:])
            n += len(toks)
    return {"nll": -tot / n, "oracle_nll": -orc / n, "n_tokens": n}


def avoid_tail(name: str, p: float = 0.95) -> dict:
    """Thm 4(c)'s last identity on the corpus: P(pure sampling never leaves nucleus(p)) = E_q Π_t m(h_t) under the
    nucleus sampler q, estimated from q's draws (unfinished draws count up to the cap), against the direct estimate
    from pure samples."""
    dq, dp = draws(name, ("nucleus", p)), draws(name, PURE)
    lm = bs.engine(name)
    est = float(np.mean(np.exp(dq.logp - dq.logq)))
    hits = []
    for i, t in enumerate(dp.tokens):
        h = list(lm.root(bs.prompts()[dp.prompt[i]]))
        ok = True
        for j in t:
            lp = lm.next_logp([tuple(h)])[0] if name == "decoder" else lm.ranked(tuple(h))[0]
            if not truncation_set(lp, "nucleus", p)[0][j]:
                ok = False
                break
            h.append(j)
        hits.append(ok)
    return {"identity": est, "direct": float(np.mean(hits)), "se": float(np.std(hits, ddof=1) / math.sqrt(len(hits)))}


def head_is_typical(stream: int = 0) -> dict:
    """How often the likeliest token is also the most typical one (least |−log p − H|), on the trigram: over the 188
    prompt rows, and over the steps of pure samples. The rest of the time typical sampling at a small mass can drop
    the head of a distribution."""

    def hit(lp: np.ndarray) -> bool:
        return int(np.argmin(np.where(np.isfinite(lp), np.abs(-lp - entropy_rows(lp)[0]), np.inf))) == int(lp.argmax())

    lm, P, d = bs.trigram(), bs.prompts(), draws("trigram", PURE, stream)
    states: Counter = Counter()
    for i, t in enumerate(d.tokens):
        h = lm.root(P[d.prompt[i]])
        for j in t:
            states[h[-2:]] += 1
            h = h[-2:] + (j,)
    w = {k: hit(lm.next_logp([k])[0]) for k in states}
    steps = sum(states.values())
    return {"rows": float(np.mean([hit(lp) for lp in trigram_rows()[0]])), "n_steps": steps,
            "steps": sum(states[k] * w[k] for k in states) / steps}


def typical_nll(name: str, stream: int = 0) -> dict[float, float]:
    """Per-word model NLL of typical sampling at each mass of SETTINGS."""
    return {x: metrics(name, (r, x), stream)["nll"] for r, x in SETTINGS if r == "typical"}


def token_costs(prompt: tuple[str, ...], words: tuple[str, ...]) -> np.ndarray:
    """One row per generated token, </s> included: (cost under the trigram, cost under the oracle, outside the oracle's
    vocabulary). The oracle's columns sum to −oracle_logp."""
    lm, vocab = bs.trigram(), oracle()[1]
    seq = [BOS] * (ORACLE_ORDER - 1) + [w if w in vocab else UNK for w in prompt]
    h, out = lm.root(prompt), []
    for w in (*words, EOS):
        j, o = lm.index[w], w if w == EOS or w in vocab else UNK
        out.append((-lm.next_logp([h])[0][j], -math.log(_oracle_prob(o, tuple(seq[-(ORACLE_ORDER - 1):]))), o == UNK))
        h += (j,)
        seq.append(o)
    return np.array(out)


@lru_cache(maxsize=1)
def held_text() -> dict:
    """The trigram's held-out text, sentence by sentence: how many held-out MD&A sentences open a prompt, and for each
    one it can score, the prompt's index and its token_costs."""
    at, lm = {p: i for i, p in enumerate(bs.prompts())}, bs.trigram()
    n_open, rows = 0, []
    for d in fc.generate_corpus(100, 1):
        if d.doc_type != "mdna":
            continue
        for s in ng.split_sentences(d.text):
            n_open += s[:3] in at
            if s[:3] in at and all(w in lm.index for w in s):
                rows.append((at[s[:3]], token_costs(s[:3], s[3:])))
    return {"n_open": n_open, "rows": rows}


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


BAND = (6, 12)  # a string length, in tokens, that both the samples and the text have plenty of


def pools(stream: int = 0) -> dict:
    """The trigram's samples on the text's weights. The mode, sample, and text averages come from three pools: 188
    prompts, 8 draws per prompt, and the scorable sentences, which cover fewer prompts and weigh them unequally. So each
    finished sample gets its prompt's sentence count as a weight, and both sides report nats per word, mean and median
    nats per string, and nats per word on strings of BAND tokens."""
    text = held_text()["rows"]
    per_prompt = Counter(i for i, _ in text)
    d, lm = draws("trigram", PURE, stream), bs.trigram()
    idx = np.array([i for i in np.flatnonzero(d.finished) if d.prompt[i] in per_prompt])
    N, L = -d.logp[idx], np.array([len(d.tokens[i]) for i in idx], dtype=float)
    w = np.array([per_prompt[d.prompt[i]] for i in idx], dtype=float)
    tN = np.array([c[:, 0].sum() for _, c in text])
    tL = np.array([len(c) for _, c in text], dtype=float)
    band, tband = (L >= BAND[0]) & (L <= BAND[1]), (tL >= BAND[0]) & (tL <= BAND[1])
    return {"n_text_prompts": len(per_prompt), "per_prompt": (min(per_prompt.values()), max(per_prompt.values())),
            "samples": {"per_word": float((w * N).sum() / (w * L).sum()), "mean": float(np.average(N, weights=w)),
                        "median": _weighted_median(N, w),
                        "band": float((w[band] * N[band]).sum() / (w[band] * L[band]).sum())},
            "text": {"per_word": float(tN.sum() / tL.sum()), "mean": float(tN.mean()), "median": float(np.median(tN)),
                     "band": float(tN[tband].sum() / tL[tband].sum())}}


def text_distance(stream: int = 0) -> dict:
    """Where the held-out text sits against the trigram's temperature curve: the curve's oracle NLL at the text's own
    model NLL, and the model NLL at which the curve reaches the text's oracle NLL."""
    held, (x, y) = heldout_reference("trigram"), temperature_curve("trigram", stream)
    return {"curve_at_text": float(np.interp(held["nll"], x, y)),
            "model_at_text_oracle": float(np.interp(held["oracle_nll"], y, x))}


def text_margin(stream: int = 0, settings: tuple[tuple[str, float], ...] = SETTINGS) -> float:
    """The least, over settings, by which a setting's oracle NLL exceeds its model NLL by more than the text's does."""
    held = heldout_reference("trigram")
    ms = [metrics("trigram", s, stream) for s in settings]
    return min(m["oracle_nll"] - m["nll"] for m in ms) - (held["oracle_nll"] - held["nll"])


HOT_TAUS = (1.75, 2.0)  # past the frontier's grid: the margin above has crossed zero by here


def rare_words(stream: int = 0) -> dict:
    """The text's distance from the curve, taken apart. Tokens outside the oracle's vocabulary all score as <unk>, so
    the oracle charges them far less than the trigram does, and the text has more of them than samples do. The
    temperature curve is redrawn on the other tokens, and read at the text's own cost on them."""
    P, lm = bs.prompts(), bs.trigram()
    text = np.vstack([c for _, c in held_text()["rows"]])
    rare = text[:, 2] > 0
    xs, ys, share = [], [], None
    for s in SETTINGS:
        if s[0] != "temperature":
            continue
        d = draws("trigram", s, stream)
        fin = np.flatnonzero(d.finished)
        A = np.vstack([token_costs(P[d.prompt[i]], tuple(lm.words(d.tokens[i][:-1]))) for i in fin])
        known = A[:, 2] == 0
        xs.append(A[known, 0].mean())
        ys.append(A[known, 1].mean())
        if s == PURE:
            share = float((~known).mean())
    return {"text_share": float(rare.mean()), "sample_share": share,  # at τ = 1
            "rare_model": float(text[rare, 0].mean()), "rare_oracle": float(text[rare, 1].mean()),
            "known_model": float(text[~rare, 0].mean()), "known_oracle": float(text[~rare, 1].mean()),
            "known_gap": float(np.interp(text[~rare, 0].mean(), xs, ys) - text[~rare, 1].mean())}


# --------------------------------------------------------------------------------------------
# 8. Finance case study: boilerplate against fabrication.
# --------------------------------------------------------------------------------------------

FACT_ORDER = 5
FRAMES = (("IN_SECTOR", ("operates", "in", "the")), ("IN_SECTOR", ("is", "classified", "under")),
          ("HQ_IN", ("is", "headquartered", "in")))


@lru_cache(maxsize=1)
def fact_model() -> pce.KNVector:
    """Topic 3's interpolated Kneser–Ney at order 5 on the training sentences: a trigram with a longer memory, long
    enough to see the company in "<company> operates in the"."""
    alphabet = pce.word_alphabet()
    return pce.KNVector(pce.kn_fit_any(ng.corpus()["train"], FACT_ORDER, alphabet), alphabet)


@dataclass(frozen=True)
class Probe:
    entity: str
    relation: str
    frame: tuple[str, ...]
    answer: str
    count: int  # training count of the 4-word context
    row: np.ndarray  # log P(· | context), <unk> excluded and renormalized
    answers: np.ndarray  # alphabet ids of every answer of the relation
    correct: int  # alphabet id of the KB's answer

    @property
    def regime(self) -> str:
        return "unseen" if self.count == 0 else "once" if self.count == 1 else "seen"


def _fact_context(entity: str, frame: tuple[str, ...]) -> tuple[str, ...]:
    return ((BOS,) * (FACT_ORDER - 1) + tuple(fc._tokens(entity)) + frame)[-(FACT_ORDER - 1):]


@lru_cache(maxsize=1)
def probes() -> tuple[Probe, ...]:
    """Every (entity, frame) of FRAMES whose KB answer is a single in-vocabulary word: public companies and
    subsidiaries for IN_SECTOR (answers: the six sectors), and for HQ_IN (answers: the headquarters cities)."""
    vec, kb = fact_model(), fc.build_world()
    unk = vec.index[UNK]
    out = []
    for rel, frame in FRAMES:
        facts = kb.facts(rel)
        answers = np.array(sorted({vec.index[t.lower()] for _, t in facts}))
        for h, t in facts:
            u = _fact_context(h, frame)
            p = vec(u).copy()
            p[unk] = 0.0
            with np.errstate(divide="ignore"):
                row = np.log(p / p.sum())
            out.append(Probe(h, rel, frame, t.lower(), int(vec.model.totals.get(u, 0)), row, answers,
                             vec.index[t.lower()]))
    return tuple(out)


def answer_split(row_q: np.ndarray, pr: Probe) -> tuple[float, float]:
    """(P(the KB's answer), P(any other answer of the relation)) under a sampling distribution."""
    q = np.exp(row_q)
    return float(q[pr.correct]), float(q[pr.answers].sum() - q[pr.correct])


# τ = 10^{k/20}, k = −28 … 20: twenty steps per decade from 0.040 to 10, with τ = 1 on the grid exactly.
TAUS = tuple(float(10 ** (k / 20)) for k in range(-28, 21))
FIN_TAUS = TAUS
FIN_GRID = {"nucleus": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
            "top-k": (1, 2, 3, 4, 5, 6, 8, 10, 20, 50),
            "min-p": (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01),
            "eta": ROW_GRID["eta"], "typical": ROW_GRID["typical"]}
REGIMES = ("seen", "once", "unseen")


@lru_cache(maxsize=1)
def fabrication_curves() -> dict:
    """Exact P(correct) and P(wrong answer) averaged within each regime, against τ and against each truncation rule's
    parameter (no sampling: every number is a sum over one next-token distribution)."""
    ps = probes()
    out: dict = {"taus": FIN_TAUS, "n": {g: sum(p.regime == g for p in ps) for g in REGIMES}}
    for g in REGIMES:
        sub = [p for p in ps if p.regime == g]
        curves = {"temperature": [[answer_split(temper(p.row, t), p) for p in sub] for t in FIN_TAUS]}
        for rule, grid in FIN_GRID.items():
            curves[rule] = [[answer_split(truncate(p.row, rule, x)[0], p) for p in sub] for x in grid]
        out[g] = {k: {"correct": [float(np.mean([a for a, _ in pts])) for pts in v],
                      "wrong": [float(np.mean([b for _, b in pts])) for pts in v]} for k, v in curves.items()}
        out[g]["argmax_right"] = float(np.mean([p.row[p.answers].argmax() == list(p.answers).index(p.correct)
                                                for p in sub]))
    return out


def _argmax_wrong(p: Probe) -> bool:
    return p.answers[p.row[p.answers].argmax()] != p.correct


def _fabrication_at(rule: str, x: float, regime: str) -> tuple[float, float]:
    """Mean (P(the KB's answer), P(any other answer)) of a regime's probes under one setting."""
    sub = [p for p in probes() if p.regime == regime]
    return tuple(np.mean([answer_split(truncate(p.row, rule, x)[0], p) for p in sub], axis=0))


def favorite_dropped(rule: str, x: float, regime: str) -> float:
    """The share of a regime's probes whose likeliest answer the rule's kept set leaves out."""
    return float(np.mean([not truncation_set(p.row, rule, x)[0][p.answers[p.row[p.answers].argmax()]]
                          for p in probes() if p.regime == regime]))


def seen_context_anatomy() -> dict:
    """What "seen once" means, relation by relation. `sighting` is the one discounted count's share of P(answer | u),
    the first term of Prop 4, on the fact model's renormalized row; `top_prior` is the backoff's largest answer, its
    favorite before that count, averaged over the probes; `wrong` counts the probes whose argmax is wrong."""
    vec = fact_model()
    out = {}
    for rel in ("IN_SECTOR", "HQ_IN"):
        sub = [p for p in probes() if p.regime == "once" and p.relation == rel]
        gain, top = [], []
        for p in sub:
            full, back, _ = kn_decomposition(vec, _fact_context(p.entity, p.frame))
            z = 1 - full[vec.index[UNK]]
            gain.append((full[p.correct] - back[p.correct]) / z)
            top.append(back[p.answers].max() / z)
        out[rel] = {"n": len(sub), "wrong": sum(_argmax_wrong(p) for p in sub), "sighting": float(np.mean(gain)),
                    "top_prior": float(np.mean(top))}
    return out


def fact_in_the_other_frame() -> tuple[int, int]:
    """"Seen" counts the four-word context, not the fact: of the never-seen sector probes, how many have the same fact
    in training under the sibling frame ("<company> operates in the <sector>" for "<company> is classified under", and
    the reverse), and how many there are. An n-gram cannot carry a fact across frames."""
    text = " " + " | ".join(" ".join(s) for s in ng.corpus()["train"]) + " "
    frames = [f for r, f in FRAMES if r == "IN_SECTOR"]
    sub = [p for p in probes() if p.regime == "unseen" and p.relation == "IN_SECTOR"]
    hits = 0
    for p in sub:
        other = frames[1] if p.frame == frames[0] else frames[0]
        hits += f" {' '.join((*fc._tokens(p.entity), *other, p.answer))} " in text
    return hits, len(sub)


@lru_cache(maxsize=1)
def decoder_on_facts() -> dict:
    """#14's decoder on the same probes, wherever the frame and the answer are in its 509 words (company words outside
    them become <unk>): how often the answer it ranks first among those the relation allows is the KB's, against the
    chance of a uniform pick."""
    lm, vec = bs.decoder(), fact_model()
    out = {}
    for rel in ("IN_SECTOR", "HQ_IN"):
        right, chance = [], []
        for p in probes():
            if p.relation != rel or p.answer not in lm.index or not all(w in lm.index for w in p.frame):
                continue
            prompt = [w if w in lm.index else UNK for w in (*fc._tokens(p.entity), *p.frame)]
            q = lm.next_logp([lm.root(prompt)])[0]
            ans = [lm.index[vec.alphabet[j]] for j in p.answers if vec.alphabet[j] in lm.index]
            right.append(ans[int(np.argmax(q[ans]))] == lm.index[p.answer])
            chance.append(1 / len(ans))
        out[rel] = {"n": len(right), "right": float(np.mean(right)), "chance": float(np.mean(chance))}
    return out


def fact_examples() -> list[dict]:
    """One probe per regime for the lab, chosen by rule: "operates in the" probes only; in "seen", the most-seen
    context; in "once", the first (alphabetical) whose argmax answer is wrong; in "unseen", the first alphabetical."""
    ps = [p for p in probes() if p.frame == FRAMES[0][1]]
    vec = fact_model()
    pick = [max((p for p in ps if p.regime == "seen"), key=lambda p: (p.count, p.entity)),
            min((p for p in ps if p.regime == "once" and _argmax_wrong(p)), key=lambda p: p.entity),
            min((p for p in ps if p.regime == "unseen"), key=lambda p: p.entity)]
    out = []
    for p in pick:
        q = np.exp(p.row[p.answers])
        out.append({"entity": p.entity, "regime": p.regime, "count": p.count, "answer": p.answer,
                    "answers": [vec.alphabet[j] for j in p.answers], "p": _r(q, 3),
                    "correct": [_r(answer_split(temper(p.row, t), p)[0]) for t in FIN_TAUS]})
    return out


# --------------------------------------------------------------------------------------------
# 9. The lab's payload.
# --------------------------------------------------------------------------------------------

LAB_TAUS = TAUS
LAB_TOP = 12


@lru_cache(maxsize=1)
def lab_prompts() -> tuple[int, ...]:
    """Five trigram prompt rows for the dial, chosen by rule: the prompts at the 10th, 30th, 50th, 70th, and 90th
    percentiles of next-token entropy (ties to the earlier prompt in sorted order)."""
    H = entropy_rows(trigram_rows()[0])
    order = np.argsort(H, kind="stable")
    return tuple(int(order[int(round(q * (len(order) - 1)))]) for q in (0.1, 0.3, 0.5, 0.7, 0.9))


ZIPF_TOY = (1.1, 60)  # (s, n): the dial's Zipf row


def _row_payload(lp: np.ndarray, words, seen: np.ndarray | None) -> dict:
    order = np.argsort(-lp, kind="stable")[:LAB_TOP]
    dial = []
    for t in LAB_TAUS:
        q = temper(lp, t)
        p = np.exp(q[np.isfinite(q)])
        z = lp[np.isfinite(lp)]
        dial.append({"p": _r(np.exp(q[order]), 4), "H": _r(entropy_rows(q)[0]), "slope": _r(zipf_slope(q)),
                     "Ez": _r(p @ z)})
    cuts = {}
    for rule, grid in ROW_GRID.items():
        pts = []
        for x in grid:
            keep = truncation_set(lp, rule, x)[0]
            m = float(np.exp(lp[keep]).sum())
            pt = {"x": x, "n": int(keep.sum()), "m": _r(m), "tv": _r(1 - m), "kl": _r(-math.log(m)),
                  "top": [bool(keep[j]) for j in order]}
            if seen is not None:
                pr, rc = precision_recall(keep[None, :], seen[None, :])
                pt.update({"precision": _r(pr[0]), "recall": _r(rc[0])})
            pts.append(pt)
        cuts[rule] = pts
    out = {"words": [words[j] for j in order], "support": int(np.isfinite(lp).sum()), "slope": _r(zipf_slope(lp)),
           "dial": dial, "cuts": cuts}
    if seen is not None:
        out["seen"] = [bool(seen[j]) for j in order]
        out["n_seen"] = int(seen.sum())
    return out


def _tree_payload() -> dict:
    lab = bs.lab_tree()
    base = tree_dist(lab)
    ys = sorted(base, key=lambda y: (-base[y], y))
    pts = []
    for t in LAB_TAUS:
        loc, glo = local_tempered(lab, t), global_tempered(lab, t)
        pts.append({"local": _r([loc[y] for y in ys], 4), "global": _r([glo[y] for y in ys], 4),
                    "tv": _r(tv(loc, glo))})
    greedy = bs.greedy(lab, (), max_len=10).tokens
    mode = bs.brute_force_mode(lab)[1]
    return {"strings": [" ".join(lab.words(y[:-1])) for y in ys], "p": _r([base[y] for y in ys], 4),
            "greedy": ys.index(greedy), "mode": ys.index(mode), "points": pts}


def _frontier_payload(name: str) -> dict:
    out = {}
    for rule in ("temperature",) + RULES:
        pts = [(x, metrics(name, (r, x))) for r, x in SETTINGS if r == rule]
        out[rule] = [{"x": x, **{k: _r(v) for k, v in m.items()}} for x, m in pts]
    held = heldout_reference(name)
    out["heldout"] = {"nll": _r(held["nll"]), "oracle_nll": _r(held["oracle_nll"])}
    if name == "trigram":
        out["mode"] = {k: _r(v) for k, v in mode_reference().items()}
    return out


def _fabrication_payload() -> dict:
    c = fabrication_curves()
    out = {"taus": _r(list(FIN_TAUS), 4), "grids": {k: list(v) for k, v in FIN_GRID.items()}, "n": c["n"],
           "examples": fact_examples()}
    for g in REGIMES:
        out[g] = {k: {s: _r(v[s]) for s in ("correct", "wrong")} for k, v in c[g].items() if k != "argmax_right"}
        out[g]["argmax_right"] = _r(c[g]["argmax_right"])
    return out


def viz_payload() -> dict:
    lm = bs.trigram()
    rows, seen = trigram_rows()
    dial = [{"prompt": " ".join(bs.prompts()[i]), **_row_payload(rows[i], lm.alphabet, seen[i])} for i in lab_prompts()]
    s, n = ZIPF_TOY
    zipf = {"prompt": f"Zipf(s = {s:g}) over {n} ranks",
            **_row_payload(np.log(zh.zipf_pmf(s, n)), [f"r{r}" for r in range(1, n + 1)], None)}
    return {"taus": _r(list(LAB_TAUS), 4), "rows": dial + [zipf], "grids": {k: list(v) for k, v in ROW_GRID.items()},
            "tree": _tree_payload(),
            "frontier": {"trigram": _frontier_payload("trigram"), "decoder": _frontier_payload("decoder")},
            "fabrication": _fabrication_payload(), "n_prompts": len(bs.prompts()), "n_draws": N_PER_PROMPT}


VIZ_JSON = (Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data"
            / "sampling-temperature-nucleus-lab.json")


def summary() -> dict:
    out = {"n_prompts": len(bs.prompts()), "draws_per_setting": len(bs.prompts()) * N_PER_PROMPT,
           "welleck_critical_tau_k5": welleck_critical_tau(5)}
    dt = desmoothing_table()
    out["kn_separable_rows"], out["kn_seen_size"] = dt["separable"], dt["seen_size"]
    for k, v in mode_reference().items():
        out[f"trigram_mode_{k}"] = v
    for name in ("trigram", "decoder"):
        for k, v in heldout_reference(name).items():
            out[f"{name}_heldout_{k}"] = v
        for k, v in typicality(name).items():
            out[f"{name}_typ_{k}"] = v
        for setting in SETTINGS:
            m = metrics(name, setting)
            for k in ("nll", "oracle_nll", "distinct2", "copied", "length", "finished", "kl", "loops", "tail"):
                out[f"{name}_{setting[0]}{setting[1]:g}_{k}"] = m[k]
    for k, v in head_is_typical().items():
        out[f"head_typical_{k}"] = v
    for k, v in {**text_distance(), **rare_words()}.items():
        out[f"text_{k}"] = v
    out["text_margin"] = text_margin()
    for k, v in pools()["samples"].items():
        out[f"pool_samples_{k}"] = v
    for k, v in pools()["text"].items():
        out[f"pool_text_{k}"] = v
    for rel, v in decoder_on_facts().items():
        out[f"decoder_{rel}_right"], out[f"decoder_{rel}_chance"] = v["right"], v["chance"]
    c = fabrication_curves()
    for g in REGIMES:
        out[f"fact_{g}_n"] = c["n"][g]
        out[f"fact_{g}_argmax_right"] = c[g]["argmax_right"]
        out[f"fact_{g}_correct_tau1"] = c[g]["temperature"]["correct"][FIN_TAUS.index(1.0)]
        out[f"fact_{g}_correct_tau_min"] = c[g]["temperature"]["correct"][0]
        out[f"fact_{g}_wrong_tau1"] = c[g]["temperature"]["wrong"][FIN_TAUS.index(1.0)]
        out[f"fact_{g}_wrong_tau_min"] = c[g]["temperature"]["wrong"][0]
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: every pedagogical claim, as an assert.
# --------------------------------------------------------------------------------------------


def _random_trees(n: int, m: int = 3, depth: int = 4, stream: int = 1800) -> list[bs.TableLM]:
    rng = np.random.default_rng([SEED, stream])
    return [bs.random_tree(rng, m, depth) for _ in range(n)]


def test_ancestral_sampling_is_exact():
    """Thm 1(i): on the lab tree and 30 random trees, 20,000 ancestral draws match p(y): χ² goodness of fit, and every
    leaf with an expected count of at least 5 within 4.5 standard errors. The other leaves are pooled into one χ² cell,
    and on the random trees they are 1,849 of 3,630."""
    rng = np.random.default_rng([SEED, 1801])
    n = 20_000
    pooled = leaves = 0
    for k, lm in enumerate([bs.lab_tree()] + _random_trees(30)):
        p = tree_dist(lm)
        assert abs(sum(p.values()) - 1) < 1e-12
        got = sample_tree(lm, rng, n)
        ys = sorted(p)
        obs, exp_ = np.array([got[y] for y in ys]), np.array([p[y] * n for y in ys])
        assert sum(got.values()) == n and set(got) <= set(p)
        big = exp_ >= 5  # χ² on cells with enough expected counts; the rest pooled
        o, e = np.append(obs[big], obs[~big].sum()), np.append(exp_[big], exp_[~big].sum())
        o, e = o[e > 0], e[e > 0]
        assert stats.chisquare(o, e).pvalue > 1e-3
        assert np.all(np.abs(obs - exp_)[big] <= 4.5 * np.sqrt(exp_ * (1 - exp_ / n))[big])
        if k:
            pooled, leaves = pooled + int((~big).sum()), leaves + len(ys)
    assert (pooled, leaves) == (1849, 3630)


def test_halting_is_consistency():
    """Thm 1(ii) on #14's stationary model (k = 3, ε = 0.05): every draw halts and |Y| is Geometric(ε), with mean
    (1 − ε)/ε = 19 — the model is consistent, so ancestral sampling halts almost surely."""
    lm = bs.welleck_model(3)
    L = stationary_lengths(lm, np.random.default_rng([SEED, 1802]), 50_000)
    assert abs(L.mean() - 19.0) < 4 * L.std(ddof=1) / math.sqrt(len(L))
    assert abs(np.mean(L == 0) - 0.05) < 4 * math.sqrt(0.05 * 0.95 / len(L))


def test_gumbel_max():
    """Prop 1: argmax(z + G) ~ softmax(z); max(z + G) ~ Gumbel(log Σ e^z); the max is independent of the argmax."""
    rng = np.random.default_rng([SEED, 1803])
    z = np.log(rng.dirichlet(np.ones(6)))
    n = 200_000
    arg, mx = gumbel_max(z, rng, n)
    p = np.exp(z - _lse(z))
    freq = np.bincount(arg, minlength=6) / n
    assert np.all(np.abs(freq - p) < 4 * np.sqrt(p * (1 - p) / n))
    assert stats.kstest(mx, "gumbel_r", args=(float(_lse(z)), 1.0)).pvalue > 1e-3
    sd = mx.std()
    for i in range(6):
        sel = mx[arg == i]
        assert abs(sel.mean() - mx.mean()) < 4.5 * sd / math.sqrt(len(sel))


def test_temperature_is_gibbs():
    """Thm 2(a): softmax(z/τ) maximizes ⟨q, z⟩ + τH(q) with maximum τ·log Σ e^{z/τ}: no scipy optimum and no
    Dirichlet-random q beats it."""
    rng = np.random.default_rng([SEED, 1804])
    for _ in range(50):
        V, tau = int(rng.integers(3, 9)), float(np.exp(rng.uniform(-1.5, 1.5)))
        z = rng.normal(0, 2, V)
        p = np.exp(temper(z, tau))
        best = gibbs_objective(p, z, tau)
        assert abs(best - tau * _lse(z / tau)) < 1e-10
        res = optimize.minimize(lambda a: -gibbs_objective(np.exp(a - _lse(a)), z, tau), np.zeros(V), method="BFGS")
        assert -res.fun <= best + 1e-9 and np.abs(np.exp(res.x - _lse(res.x)) - p).max() < 1e-4
        Q = rng.dirichlet(np.ones(V), size=200)
        assert all(gibbs_objective(q, z, tau) <= best + 1e-12 for q in Q)


def test_entropy_rises_with_temperature():
    """Thm 2(b)(c): dH/dτ = Var(z)/τ³ ≥ 0 (against a central difference); entropy is nondecreasing on a 200-point τ grid
    for all 188 trigram prompt rows; τ → 0 spreads the mass uniformly over the argmax set (KN rows can tie) and τ → ∞ spreads it uniformly on the support."""
    rng = np.random.default_rng([SEED, 1805])
    for _ in range(50):
        z = rng.normal(0, 2, 7)
        tau, h = float(np.exp(rng.uniform(-1, 1))), 1e-5
        fd = (entropy_rows(temper(z, tau + h))[0] - entropy_rows(temper(z, tau - h))[0]) / (2 * h)
        assert abs(fd - entropy_derivative(z, tau)) < 1e-6 * max(1.0, abs(fd))
    rows = trigram_rows()[0]
    grid = np.geomspace(0.05, 20, 200)
    H = np.array([entropy_rows(temper(rows, t)) for t in grid])
    assert np.all(np.diff(H, axis=0) >= -1e-9)
    support = np.isfinite(rows).sum(axis=1)
    ties = rows >= rows.max(axis=1, keepdims=True) - 1e-12  # KN rows can tie at the top: the limit is uniform on ties
    cold = np.exp(temper(rows, 1e-6))
    assert np.all(np.where(ties, cold, 0).sum(axis=1) > 0.99)
    assert np.all(np.abs(cold[ties] - np.repeat(1 / ties.sum(axis=1), ties.sum(axis=1))) < 1e-2)
    assert np.all(np.abs(entropy_rows(temper(rows, 1e4)) - np.log(support)) < 1e-2)


def test_temperature_divides_the_zipf_slope():
    """Prop 2: the least-squares Zipf slope of any tempered row is exactly the row's slope over τ (every trigram prompt
    row at τ = 0.5 and 2); a Zipf(s) row has slope −s."""
    for lp in trigram_rows()[0]:
        s = zipf_slope(lp)
        for tau in (0.5, 2.0):
            assert abs(zipf_slope(temper(lp, tau)) - s / tau) < 1e-9
    s, n = ZIPF_TOY
    assert abs(zipf_slope(np.log(zh.zipf_pmf(s, n))) + s) < 1e-9


def test_tempering_moves_the_intercept():
    """Prop 2's prose: tempering is no rotation about the intercept. On Zipf(1.1) over 60 ranks at τ = 0.5, log p at
    rank 1 goes from −1.374 to −0.395, and the two lines cross near rank 2.4."""
    s, n = ZIPF_TOY
    c = zipf_crossing(s, n, 0.5)
    assert abs(c["before"] + 1.374) < 5e-4 and abs(c["after"] + 0.395) < 5e-4 and abs(c["cross"] - 2.4) < 0.05
    lp, q = np.log(zh.zipf_pmf(s, n)), temper(np.log(zh.zipf_pmf(s, n)), 0.5)
    r = np.arange(1, n + 1)
    assert np.all((q > lp) == (r < c["cross"]))


def test_local_temperature_is_not_global():
    """Thm 3: q_τ(y) = p(y)^{1/τ}/Π_t Z_τ(h_t), so log(q_τ/π_τ) = log 𝒵_τ + (1/τ − 1)·Σ_t H_{1/τ}(h_t) (30 random
    trees, τ ∈ {0.5, 0.8, 2}); on permutation trees q_τ = π_τ; on the lab tree they differ, and as τ → 0 the local
    sampler collapses on greedy's "rose sharply" while the global one collapses on the mode "was flat". The lab tree
    has ties at "rose on" and "rose in", off greedy's path, so (c) must ask only for unique argmaxes along that path."""
    for lm in _random_trees(30, stream=1806):
        base = tree_dist(lm)
        for tau in (0.5, 0.8, 2.0):
            loc, glo = local_tempered(lm, tau), global_tempered(lm, tau)
            logZ = math.log(sum(P ** (1 / tau) for P in base.values()))
            for y in base:
                want = logZ + (1 / tau - 1) * renyi_path(lm, y, tau)
                assert abs(math.log(loc[y] / glo[y]) - want) < 1e-9
    rng = np.random.default_rng([SEED, 1807])
    for _ in range(10):
        lm = permutation_tree(rng, 3, 3)
        for tau in (0.3, 0.7, 2.5):
            assert tv(local_tempered(lm, tau), global_tempered(lm, tau)) < 1e-12
    lab = bs.lab_tree()
    assert tv(local_tempered(lab, 0.5), global_tempered(lab, 0.5)) > 0.05
    g, m = bs.greedy(lab, (), max_len=10).tokens, bs.brute_force_mode(lab)[1]
    assert lab.words(g[:-1]) == ["rose", "sharply"] and lab.words(m[:-1]) == ["was", "flat"]
    assert local_tempered(lab, 0.02)[g] > 0.99 and global_tempered(lab, 0.02)[m] > 0.99
    ix = lab.alphabet.index
    assert argmax_ties(lab, (ix("rose"), ix("on"))) == 2 and argmax_ties(lab, (ix("rose"), ix("in"))) == 2
    assert all(argmax_ties(lab, g[:t]) == 1 for t in range(len(g)))
    assert sorted(tree_dist(lab).values())[-1] > sorted(tree_dist(lab).values())[-2]


def test_tempered_sequence_distribution_can_diverge():
    """Thm 3's remark: on #14's stationary model (k = 5, ε = 0.05) Σ_y p(y)^{1/τ} is finite iff τ < 1.0319; above it
    the partial sums grow without bound, while the per-step sampler is a distribution at every τ."""
    tc = welleck_critical_tau(5)
    assert abs(welleck_ratio(5, tc) - 1) < 1e-12 and 1.03 < tc < 1.04
    assert welleck_ratio(5, 1.0) < 1 < welleck_ratio(5, 1.1)
    r = welleck_ratio(5, 1.1)
    assert sum(r**t for t in range(2000)) > 50 * sum(r**t for t in range(20))
    lm = bs.welleck_model(5)
    assert abs(np.exp(temper(lm.row, 1.5)).sum() - 1) < 1e-12


def test_nucleus_is_minimal():
    """Thm 4(a), brute force over every subset of 200 random 8-token distributions: the nucleus has the fewest tokens
    of any set reaching mass p, and the most mass among sets of its size; top-k has the most mass among k-sets."""
    rng = np.random.default_rng([SEED, 1808])
    for _ in range(200):
        p = rng.dirichlet(np.full(8, 0.6))
        lp = np.log(p)
        for target in (0.3, 0.6, 0.9, 0.97):
            keep = truncation_set(lp, "nucleus", target)[0]
            assert keep.sum() == min_size_reaching(p, target)
            assert abs(p[keep].sum() - best_mass_of_size(p, int(keep.sum()))) < 1e-12
        for k in (1, 3, 5):
            assert abs(p[truncation_set(lp, "top-k", k)[0]].sum() - best_mass_of_size(p, k)) < 1e-12


def test_truncation_costs_per_step():
    """Thm 4(b): keeping mass m costs TV = 1 − m and KL(p_S ‖ p) = −log m exactly, and KL(p ‖ p_S) = ∞."""
    rng = np.random.default_rng([SEED, 1809])
    for _ in range(100):
        lp = np.log(rng.dirichlet(np.full(30, 0.3)))
        for rule, grid in ROW_GRID.items():
            for x in grid[1:4]:
                q = truncate(lp, rule, x)[0]
                m = kept_mass(lp, rule, x)[0]
                P, Q = np.exp(lp), np.exp(q)
                assert abs(0.5 * np.abs(Q - P).sum() - (1 - m)) < 1e-12
                nz = Q > 0
                assert abs(np.sum(Q[nz] * (q[nz] - lp[nz])) + math.log(m)) < 1e-12
                assert m == 1 or not np.all(nz)


def test_truncation_costs_per_sequence():
    """Thm 4(c) on 30 random trees under every rule, by exact enumeration: KL(q ‖ p) = E_q Σ −log m(h_t),
    TV(q, p) ≤ E_q Σ (1 − m(h_t)), and P_p(never leave the kept sets) = E_q Π m(h_t)."""
    for lm in _random_trees(30, m=4, stream=1810):
        for rule, grid in ROW_GRID.items():
            for x in grid[1:4]:
                c = sequence_cost(lm, rule, x)
                assert abs(c.kl - c.kl_identity) < 1e-12 and c.tv <= c.tv_bound + 1e-12
                assert abs(c.avoid - c.avoid_identity) < 1e-12


def test_pure_sampling_leaves_the_nucleus():
    """Thm 4(c)–(d) on the trigram: P(a pure sample never leaves nucleus(0.95)) estimated two ways (E_q Π m under
    nucleus sampling, and directly from pure samples) agrees within 4 standard errors, and is well below 1."""
    a = avoid_tail("trigram")
    assert abs(a["identity"] - a["direct"]) < 4 * a["se"] and a["direct"] < 0.85


def test_the_tail_bound_counts_strings_that_reach_step_n():
    """Thm 4(d): with kept masses at most 1 − ε along the way, P(the first n draws happen and are all kept) ≤ (1 − ε)^n
    (exactly, on 30 random trees under the nucleus). "Stays inside for n steps" is false if a string may halt early:
    with </s> at 0.9 at the root and only </s> kept, a draw never leaves the kept sets with probability 0.9 > 0.9²."""

    def reach(lm: bs.TableLM, n: int, keep_rule: tuple[str, float]) -> tuple[float, float]:
        total, worst, stack = 0.0, 0.0, [(lm.root, 1.0)]
        while stack:
            h, P = stack.pop()
            if len(h) == n:
                total += P
                continue
            keep = truncation_set(_log(lm.rows[h]), *keep_rule)[0]
            worst = max(worst, float(lm.rows[h][keep].sum()))
            stack += [(h + (j,), P * lm.rows[h][j]) for j in np.flatnonzero(keep) if j != lm.eos]
        return total, worst

    for lm in _random_trees(30, m=4, stream=1812):
        for n in (1, 2, 3):
            total, worst = reach(lm, n, ("nucleus", 0.7))
            assert total <= worst**n + 1e-12
    halt = bs.table_lm({"": {EOS: 0.9, "w": 0.1}, "w": {EOS: 1.0}})
    assert reach(halt, 2, ("nucleus", 0.9))[0] == 0.0
    assert sequence_cost(halt, "nucleus", 0.9).avoid == 0.9 > 0.9**2


def test_uniform_desmoothing():
    """Prop 3: for p = (1 − λ)p* + λ/|V|, the threshold λ/|V| recovers supp p* exactly (50 random mixtures)."""
    rng = np.random.default_rng([SEED, 1811])
    for _ in range(50):
        V, lam = int(rng.integers(20, 200)), float(rng.uniform(0.05, 0.6))
        star, p = uniform_mixture(rng, V, int(rng.integers(2, 15)), lam)
        assert np.array_equal(p > lam / V + 1e-15, star > 0)


def test_kn_seen_set_is_recoverable_but_no_rule_finds_it():
    """Prop 4 on the trigram's prompt rows: P(w | u) − γ(u)·P(w | u⁻) > 0 exactly on the seen set; no grid value of any
    rule has precision and recall 1 on every row, though most rows (0.85) are separable by a threshold of their own."""
    lm = bs.trigram()
    for p in bs.prompts():
        full, back, seen = kn_decomposition(lm.vec, _trigram_context(lm, lm.root(p)))
        if seen.any():
            assert np.array_equal(full - back > 1e-12, seen)
    d = desmoothing_table()
    assert not any(any(d[r]["perfect"]) for r in ROW_GRID)
    assert 0.7 < d["separable"] < 1


def test_samples_are_typical_of_the_model():
    """Prop 5: E[NLL − Σ_t H(h_t)] = 0 within 4 standard errors on both engines. As measured, not as predicted: on the
    trigram the pooled per-word NLL of pure samples lies between the mode's and the held-out text's."""
    for name in ("trigram", "decoder"):
        t = typicality(name)
        assert abs(t["gap"]) < 4 * t["se"]
    mode, pure, held = mode_reference()["nll"], metrics("trigram", PURE)["nll"], heldout_reference("trigram")["nll"]
    assert mode < pure < held


def test_the_three_averages_are_three_pools():
    """Prop 5 places the samples; the text's place is a measurement, and the order depends on the statistic. Modes
    cover 188 prompts; the text 417 of 443 scorable sentences on 182 prompts, 1 to 16 per prompt; samples 8 per prompt.
    On the text's weights, over five streams, samples sit below the text per word (2.26–2.39 against 2.576), level
    with it in mean per string (20.1–22.7 against 22.17), and far below it in median (12.2–14.2 against 20.0) and at
    6–12 tokens (1.93–2.07 against 2.54)."""
    assert len(bs.prompts()) == 188 == len(bs.exact_modes("trigram"))
    assert held_text()["n_open"] == 443 and len(held_text()["rows"]) == 417
    P = [pools(s) for s in STREAMS]
    assert all(p["n_text_prompts"] == 182 and p["per_prompt"] == (1, 16) for p in P)
    t = P[0]["text"]
    assert abs(t["per_word"] - heldout_reference("trigram")["nll"]) < 1e-9
    assert (round(t["per_word"], 3), round(t["mean"], 2)) == (2.576, 22.17)
    assert (round(t["median"], 1), round(t["band"], 2)) == (20.0, 2.54)
    S = {k: [p["samples"][k] for p in P] for k in t}
    assert (round(min(S["per_word"]), 2), round(max(S["per_word"]), 2)) == (2.26, 2.39)
    assert max(S["per_word"]) < t["per_word"]
    assert (round(min(S["mean"]), 1), round(max(S["mean"]), 1)) == (20.1, 22.7)
    assert min(S["mean"]) < t["mean"] < max(S["mean"])
    assert (round(min(S["median"]), 1), round(max(S["median"]), 1)) == (12.2, 14.2)
    assert (round(min(S["band"]), 2), round(max(S["band"]), 2)) == (1.93, 2.07)


def test_the_head_is_often_typical():
    """Def 4's prose: the likeliest token is also the most typical one on 0.617 of the prompt rows and on about 0.48 of
    the steps along pure samples. The rest of the time a small typical mass can drop the head."""
    h = [head_is_typical(s) for s in STREAMS]
    assert all(round(x["rows"], 3) == 0.617 for x in h)
    assert all(0.47 < x["steps"] < 0.50 for x in h) and round(h[0]["steps"], 2) == 0.48


def test_typical_cost_depends_on_the_engine():
    """Tightening typical sampling does not steadily change its cost. On the trigram, on every stream, cost is U-shaped
    in the typical mass (five-stream means 1.65, 1.50, 1.74 at 0.2, 0.5, 0.9). On the decoder 0.2 and 0.5 tie on seed
    0, and 0.2 is cheaper on seeds 1–3."""
    tri = [typical_nll("trigram", s) for s in STREAMS]
    assert all(t[0.5] < t[0.2] and t[0.5] < t[0.9] for t in tri)
    means = [round(float(np.mean([t[m] for t in tri])), 2) for m in (0.2, 0.5, 0.9)]
    assert means == [1.65, 1.50, 1.74]
    tied = [typical_nll("decoder", s) for s in STREAMS]
    assert all(abs(t[0.2] - t[0.5]) < 0.012 for t in tied)
    assert all(typical_nll(name)[0.2] < typical_nll(name)[0.5] - 0.03 for name in DECODER_SEEDS[1:])


def test_temperature_trades_quality_for_diversity():
    """§7: on both engines, as τ rises the model NLL, the oracle NLL, and distinct-2 rise and the copy rate falls. Every
    truncation rule's KL falls as its parameter loosens, and so does its model NLL rise, except typical sampling: at a
    small typical mass it can drop the head of a distribution, so its samples are less predictable than at a larger
    mass."""
    for name in ("trigram", "decoder"):
        temps = [metrics(name, s) for s in SETTINGS if s[0] == "temperature"]
        for k in ("nll", "oracle_nll", "distinct2"):
            assert all(a[k] < b[k] for a, b in zip(temps, temps[1:])), (name, k)
        copied = {t: metrics(name, ("temperature", t))["copied"] for t in (0.5, 1.0, 1.5)}
        assert copied[0.5] > copied[1.0] > copied[1.5]
        for rule in RULES:
            pts = [metrics(name, s) for s in SETTINGS if s[0] == rule]
            assert all(a["kl"] > b["kl"] for a, b in zip(pts, pts[1:])), (name, rule)
            if rule != "typical":
                assert all(a["nll"] < b["nll"] for a, b in zip(pts, pts[1:])), (name, rule)
    assert metrics("trigram", ("typical", 0.2))["nll"] > metrics("trigram", ("typical", 0.5))["nll"]


def test_copying_falls_on_the_mean():
    """§7: five-stream means of the copy rate fall at every step from τ = 0.4 on both engines. One stream need not: on
    stream 0 the trigram rises from 0.4 to 0.5, and the decoder from 0.7 to 0.85."""
    temps = [s for s in SETTINGS if s[0] == "temperature"]
    for name in ("trigram", "decoder"):
        means = np.mean([[metrics(name, t, s)["copied"] for t in temps] for s in STREAMS], axis=0)
        assert np.all(np.diff(means) < 0), (name, means)
    assert metrics("trigram", temps[0])["copied"] < metrics("trigram", temps[1])["copied"]
    assert metrics("decoder", temps[2])["copied"] < metrics("decoder", temps[3])["copied"]


def test_hot_nlls_count_finished_strings():
    """§7: at τ = 1.5 the 40-token cap stops many draws, and the NLLs and copy rates leave them out: 0.76–0.78 of the
    trigram's draws finish on every stream, and about 0.94 of the decoder's."""
    hot = ("temperature", 1.5)
    tri = [metrics("trigram", hot, s)["finished"] for s in STREAMS]
    assert 0.755 < min(tri) and max(tri) < 0.785
    assert all(0.93 < metrics("decoder", hot, s)["finished"] < 0.955 for s in STREAMS)


def test_rules_share_one_frontier():
    """§7: every truncation setting lies within 0.1 nats of oracle NLL of the temperature curve at its own model NLL,
    on both engines: how much a rule cuts matters more than which rule it is."""
    for name in ("trigram", "decoder"):
        offs = [frontier_offset(name, s) for s in SETTINGS if s[0] != "temperature"]
        assert all(o is not None and abs(o) < 0.1 for o in offs), (name, offs)


def test_the_text_is_off_every_curve():
    """§7: no setting run reaches the held-out text (2.576 under the trigram, 1.968 under the oracle). "Below every
    curve" holds only at the text's own model NLL, where the curve reads about 2.7. The curve's cold settings sit far
    below 1.968, and it reaches that oracle cost at model NLL about 1.77. Every setting's oracle NLL exceeds its model
    NLL by more than the text's does, by at least 0.25 on all five streams, and that margin is a statement about the
    settings: it reaches zero at about τ = 1.75 and is negative on every stream at τ = 2, where about a third of the
    trigram's draws finish."""
    held = heldout_reference("trigram")
    assert round(held["nll"], 3) == 2.576 and round(held["oracle_nll"], 3) == 1.968
    for s in STREAMS:
        d = text_distance(s)
        assert 2.69 < d["curve_at_text"] < 2.74 and 1.74 < d["model_at_text_oracle"] < 1.79
        assert temperature_curve("trigram", s)[1][0] < held["oracle_nll"] - 0.5
        assert 0.25 < text_margin(s) < 0.3
        assert abs(text_margin(s, (("temperature", HOT_TAUS[0]),))) < 0.03
        assert text_margin(s, (("temperature", HOT_TAUS[1]),)) < -0.05
        assert 0.3 < metrics("trigram", ("temperature", HOT_TAUS[1]), s)["finished"] < 0.34


def test_rare_words_carry_part_of_the_gap():
    """§7: the text's 0.75 nats from the curve, a five-stream mean, is not all model error. 6.4% of its tokens are
    outside the oracle's vocabulary, costing 10.31 under the trigram and 3.04 under the oracle, against about 2% of the
    samples' tokens. On the other tokens the text sits about 0.44 from a curve redrawn on those tokens (0.43–0.45)."""
    oracle_cost = heldout_reference("trigram")["oracle_nll"]
    assert round(float(np.mean([text_distance(s)["curve_at_text"] for s in STREAMS])) - oracle_cost, 2) == 0.75
    assert round(float(np.mean([rare_words(s)["known_gap"] for s in STREAMS])), 2) == 0.44
    for s in STREAMS:
        r = rare_words(s)
        assert round(r["text_share"], 3) == 0.064 and 0.02 < r["sample_share"] < 0.03
        assert round(r["rare_model"], 2) == 10.31 and round(r["rare_oracle"], 2) == 3.04
        assert round(r["known_model"], 2) == 2.05 and round(r["known_oracle"], 2) == 1.90
        assert 0.42 < r["known_gap"] < 0.45


def test_fabrication_regimes():
    """§8, exact on the order-5 fact model's probes.
    Seen ≥ 2 times: the argmax is always right; as τ falls to 0, P(wrong) falls monotonically to 0.
    Seen once: lowering τ from 1 raises P(correct) on average, yet at τ → 0 the probes whose argmax is wrong (0.30 of
    them) fabricate with probability → 1.
    Never seen: no τ and no setting of top-k, nucleus, min-p, or η lifts P(correct) above 0.1."""
    c, i1 = fabrication_curves(), FIN_TAUS.index(1.0)
    seen = c["seen"]["temperature"]
    assert c["seen"]["argmax_right"] == 1.0 and seen["wrong"][0] < 1e-6
    assert np.all(np.diff(seen["wrong"][: i1 + 1]) >= -1e-12)
    once = c["once"]
    assert once["temperature"]["correct"][0] > once["temperature"]["correct"][i1]
    assert abs(once["temperature"]["wrong"][0] - (1 - once["argmax_right"])) < 5e-3 and 0.2 < 1 - once["argmax_right"]
    for p in probes():
        if p.regime == "once" and p.answers[p.row[p.answers].argmax()] != p.correct:
            assert answer_split(temper(p.row, 1e-6), p)[1] > 0.99 and answer_split(temper(p.row, FIN_TAUS[0]), p)[1] > 0.98
    un = c["unseen"]
    assert max(max(v["correct"]) for k, v in un.items() if isinstance(v, dict) and k != "typical") < 0.1


def test_a_little_cooling_raises_once_seen_fabrication():
    """§8: on once-seen contexts P(wrong answer) is not monotone in τ. It rises from 0.544 at τ = 1 to a peak of 0.607
    near τ = 0.75, as the mass on non-answers moves onto answers, is back to 0.551 at τ = 0.5, and falls to 0.302 at
    τ = 0.04. Copperleaf's right answer peaks at 0.178 near τ = 0.79 before it collapses to 0."""
    c, T = fabrication_curves()["once"]["temperature"], np.array(FIN_TAUS)
    w, r = np.array(c["wrong"]), np.array(c["correct"])
    i1, i5, top = FIN_TAUS.index(1.0), int(np.argmin(np.abs(T - 0.5))), int(w.argmax())
    assert [round(float(w[i]), 3) for i in (i1, top, i5, 0)] == [0.544, 0.607, 0.551, 0.302]
    assert 0.7 < T[top] < 0.8 and abs(w[top - 1] - w[top]) < 1e-3
    assert 1 - w[top] - r[top] < 1 - w[i1] - r[i1]
    e = next(x for x in fact_examples() if x["regime"] == "once")
    assert e["entity"] == "Copperleaf Markets" and round(max(e["correct"]), 3) == 0.178
    assert 0.75 < T[int(np.argmax(e["correct"]))] < 0.8 and e["correct"][0] < 1e-3


def test_typical_lifts_the_unseen_by_dropping_the_favorite():
    """§8: the fifth rule is the exception. Typical at mass 0.2 takes never-seen contexts to right 0.191, wrong 0.692,
    by dropping the likeliest answer on every such probe, and the same move cuts once-seen contexts from right 0.240 to
    0.064. On finer grids than FIN_GRID the other rules stay under 0.1 too."""
    right, wrong = _fabrication_at("typical", 0.2, "unseen")
    assert round(right, 3) == 0.191 and round(wrong, 3) == 0.692
    assert favorite_dropped("typical", 0.2, "unseen") == 1.0 == favorite_dropped("typical", 0.2, "once")
    i1 = FIN_TAUS.index(1.0)
    assert round(fabrication_curves()["once"]["temperature"]["correct"][i1], 3) == 0.240
    assert round(_fabrication_at("typical", 0.2, "once")[0], 3) == 0.064
    fine = {"top-k": range(1, 60), "nucleus": np.arange(0.05, 1.0, 0.05), "min-p": np.arange(0.01, 1, 0.03),
            "eta": 10 ** np.linspace(-4, -0.5, 36)}
    for rule, grid in fine.items():
        assert max(_fabrication_at(rule, x, "unseen")[0] for x in grid) < 0.1, rule


def test_cooling_raises_fabrication_on_unseen_contexts():
    """§8: sharpening does something where nothing was learned. On never-seen contexts P(wrong answer) rises from 0.699
    at τ = 1 to 0.951 at τ = 0.04, because the 0.238 on words that are no answer at all moves onto answers."""
    c, i1 = fabrication_curves()["unseen"]["temperature"], FIN_TAUS.index(1.0)
    assert round(c["wrong"][i1], 3) == 0.699 and round(c["wrong"][0], 3) == 0.951
    other = [1 - a - b for a, b in zip(c["correct"], c["wrong"])]
    assert round(other[i1], 3) == 0.238 and other[0] < 1e-3


def test_seen_counts_the_context():
    """§8: the regimes count a four-word context, not a fact. 14 of the 34 never-seen sector probes have the same fact
    in training under the sibling frame. All 10 once-seen failures are sectors (10 of 21; cities 0 of 12), and one
    sighting adds about 0.124, enough to beat the cities' flat prior but not the sectors' lopsided one."""
    assert fact_in_the_other_frame() == (14, 34)
    a = seen_context_anatomy()
    assert (a["IN_SECTOR"]["n"], a["IN_SECTOR"]["wrong"]) == (21, 10)
    assert (a["HQ_IN"]["n"], a["HQ_IN"]["wrong"]) == (12, 0)
    assert all(round(v["sighting"], 3) == 0.124 for v in a.values())
    assert a["HQ_IN"]["top_prior"] < 0.124 < a["IN_SECTOR"]["top_prior"]
    assert sum(_argmax_wrong(p) for p in probes() if p.regime == "once") == 10


def test_the_decoder_answers_near_chance():
    """§8: on the probes it can score, #14's decoder ranks the right answer first on 0.189 of 74 sector probes against
    chance 0.167, and 0.115 of 26 headquarters probes against chance 0.125. Both sit within two standard errors of
    chance."""
    d = decoder_on_facts()
    s, h = d["IN_SECTOR"], d["HQ_IN"]
    assert (s["n"], round(s["right"], 3), round(s["chance"], 3)) == (74, 0.189, 0.167)
    assert (h["n"], round(h["right"], 3), round(h["chance"], 3)) == (26, 0.115, 0.125)
    for v in d.values():
        assert abs(v["right"] - v["chance"]) < 2 * math.sqrt(v["chance"] * (1 - v["chance"]) / v["n"])


def test_draw_scores_match_the_model():
    """Every draw's accumulated log p equals the engine's own score of the drawn string: one forward pass for the
    decoder (≤ 1e-9), the step rows for the trigram."""
    for name in ("trigram", "decoder"):
        lm, d = bs.engine(name), draws(name, ("nucleus", 0.9))
        for i in range(0, len(d.tokens), 97):
            root = lm.root(bs.prompts()[d.prompt[i]])
            if name == "decoder":
                if d.finished[i]:
                    assert abs(lm.string_logp(root, d.tokens[i]) - d.logp[i]) < 1e-9
            else:
                h, s = root, 0.0
                for j in d.tokens[i]:
                    s += lm.next_logp([h])[0][j]
                    h += (j,)
                assert abs(s - d.logp[i]) < 1e-9


def test_viz_payload():
    p = viz_payload()
    assert len(p["rows"]) == 6 and len(p["rows"][0]["dial"]) == len(LAB_TAUS)
    assert p["tree"]["strings"][p["tree"]["greedy"]] == "rose sharply"
    assert p["tree"]["strings"][p["tree"]["mode"]] == "was flat"
    assert [e["regime"] for e in p["fabrication"]["examples"]] == list(REGIMES)
    assert set(p["fabrication"]["grids"]) == set(RULES)
    assert set(p["frontier"]["trigram"]) >= {"temperature", *RULES, "mode", "heldout"}
    for row in p["rows"][:5]:
        assert row["dial"][LAB_TAUS.index(1.0)]["slope"] == row["slope"]


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
    print(f"\nSummary ({len(bs.prompts())} held-out MD&A prompts × {N_PER_PROMPT} draws; Topic 3's IKN trigram, "
          f"V = {len(bs.trigram().alphabet)}; Topic 14's decoder, V = {bs.DEC_V}; order-{FACT_ORDER} KN fact model):")
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
