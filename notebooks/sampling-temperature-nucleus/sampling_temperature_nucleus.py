"""Sampling: temperature, top-k, and nucleus — the reference implementation for formalNLP's fifteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Sampling is exact; search is not. Ancestral sampling returns each finite string with exactly its model probability,
   and halts almost surely iff the model is consistent (Thm 1). The Gumbel-max trick makes a draw the argmax of
   perturbed log-probabilities, so sampling and search differ only by noise (Prop 1).
2. Temperature as a Gibbs distribution: softmax(z/τ) maximizes ⟨q, z⟩ + τH(q); its entropy rises with τ at rate
   Var(z)/τ³; τ → 0 and τ → ∞ give the argmax and the uniform distribution (Thm 2). Tempering divides every
   least-squares Zipf slope by τ exactly (Prop 2).
3. Local against global temperature: the per-step tempered sampler is p(y)^{1/τ} divided by the product of the steps'
   normalizers, so it equals the tempered sequence distribution only when that product is constant. It upweights
   strings in proportion to exp((1/τ − 1)·Σ_t H_{1/τ}(p(·|h_t))), the accumulated Rényi entropy of their prefixes, and
   as τ → 0 it lands on greedy's string, not the mode. For τ > 1 the tempered sequence distribution can fail to exist
   (Thm 3).
4. Truncation: the nucleus is a minimum-cardinality set of mass ≥ p. Keeping a set of mass m costs TV = 1 − m and
   KL = −log m per step. Over a sequence, KL(q ‖ p) = E_q Σ_t −log m(h_t) exactly, TV(q, p) ≤ E_q Σ_t (1 − m(h_t)), and
   the probability that pure sampling never leaves the kept sets is E_q Π_t m(h_t) (Thm 4).
5. Truncation as desmoothing: a uniform mixture's support is recovered by one threshold (Prop 3). In interpolated
   Kneser–Ney the seen set is recovered exactly from the model's own decomposition, but no single setting of any rule
   recovers it on every row (Prop 4).
6. Typicality: E NLL = E Σ_t H(h_t), and the gap is a martingale, so a sample's NLL tracks its own path's entropy. It
   sits between the mode and the held-out text (Prop 5).
7. Quality against diversity on the corpus: Topic 3's trigram and Topic 14's decoder under every rule, with the
   per-word NLL under an oracle KN model fit to five times more text from the same generator.
8. Finance case study, boilerplate against fabrication: an order-5 Kneser–Ney model as a fact model, probed with
   "<company> operates in the" and similar frames and scored exactly against the knowledge base. Facts seen twice or
   more, facts seen once, and facts never seen respond to temperature and truncation in opposite ways.

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
    </s> is forced at depth: every string has the same normalizer product, so local = global (Thm 3(a))."""
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
    """The global tempered partition function of Welleck's stationary model is Σ_t r^t·ε^{1/τ} with
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
def draws(name: str, setting: tuple[str, float]) -> Draws:
    """N_PER_PROMPT draws per prompt from one engine under one setting. Every setting restarts the engine's one stream
    (common random numbers: settings differ by their rule, not by their luck). Each step draws one uniform per open
    string and inverts the sampler's CDF; the trigram transforms each Markov state once."""
    lm = bs.engine(name)
    rng = np.random.default_rng([SEED, ENGINE_STREAM[name]])
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
def metrics(name: str, setting: tuple[str, float]) -> dict:
    """One point of the frontier: finished share; mean words of finished continuations; per-word NLL under the model
    and under the oracle (pooled over finished continuations, </s> included); distinct-2 over all continuations;
    distinct sentences; copy rate (prompt + continuation is a training sentence); loop rate (a word three times in a
    row); share ending on the 8-K tail; per-word KL(q ‖ p), estimated as E_q Σ (log q − log p) over all draws."""
    lm, d = bs.engine(name), draws(name, setting)
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


def frontier_offset(name: str, setting: tuple[str, float]) -> float | None:
    """How far a setting's oracle NLL sits above the temperature curve at the same model NLL (linear interpolation
    between the temperature settings); None outside the temperature curve's range."""
    temps = [metrics(name, s) for s in SETTINGS if s[0] == "temperature"]
    x, y = np.array([m["nll"] for m in temps]), np.array([m["oracle_nll"] for m in temps])
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
    lm = bs.engine(name)
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
            u = ((BOS,) * (FACT_ORDER - 1) + tuple(fc._tokens(h)) + frame)[-(FACT_ORDER - 1):]
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
            "min-p": (0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)}
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


def fact_examples() -> list[dict]:
    """One probe per regime for the lab, chosen by rule: "operates in the" probes only; in "seen", the most-seen
    context; in "once", the first (alphabetical) whose argmax answer is wrong; in "unseen", the first alphabetical."""
    ps = [p for p in probes() if p.frame == FRAMES[0][1]]
    vec = fact_model()

    def wrong(p):
        return p.answers[p.row[p.answers].argmax()] != p.correct

    pick = [max((p for p in ps if p.regime == "seen"), key=lambda p: (p.count, p.entity)),
            min((p for p in ps if p.regime == "once" and wrong(p)), key=lambda p: p.entity),
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
    c = fabrication_curves()
    for g in REGIMES:
        out[f"fact_{g}_n"] = c["n"][g]
        out[f"fact_{g}_argmax_right"] = c[g]["argmax_right"]
        out[f"fact_{g}_correct_tau1"] = c[g]["temperature"]["correct"][FIN_TAUS.index(1.0)]
        out[f"fact_{g}_correct_tau_min"] = c[g]["temperature"]["correct"][0]
        out[f"fact_{g}_wrong_tau_min"] = c[g]["temperature"]["wrong"][0]
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: every pedagogical claim, as an assert.
# --------------------------------------------------------------------------------------------


def _random_trees(n: int, m: int = 3, depth: int = 4, stream: int = 1800) -> list[bs.TableLM]:
    rng = np.random.default_rng([SEED, stream])
    return [bs.random_tree(rng, m, depth) for _ in range(n)]


def test_ancestral_sampling_is_exact():
    """Thm 1(i): on the lab tree and 30 random trees, 20,000 ancestral draws match p(y) leaf by leaf (χ² goodness of
    fit, and every leaf within 4.5 standard errors)."""
    rng = np.random.default_rng([SEED, 1801])
    n = 20_000
    for lm in [bs.lab_tree()] + _random_trees(30):
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


def test_halting_is_consistency():
    """Thm 1(ii) on Welleck's stationary model (k = 3, ε = 0.05): every draw halts and |Y| is Geometric(ε), with mean
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


def test_local_temperature_is_not_global():
    """Thm 3: q_τ(y) = p(y)^{1/τ}/Π_t Z_τ(h_t), so log(q_τ/π_τ) = log 𝒵_τ + (1/τ − 1)·Σ_t H_{1/τ}(h_t) (30 random
    trees, τ ∈ {0.5, 0.8, 2}); on permutation trees q_τ = π_τ; on the lab tree they differ, and as τ → 0 the local
    sampler collapses on greedy's "rose sharply" while the global one collapses on the mode "was flat"."""
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


def test_tempered_sequence_distribution_can_diverge():
    """Thm 3's remark: on Welleck's stationary model (k = 5, ε = 0.05) Σ_y p(y)^{1/τ} is finite iff τ < 1.0319; above it
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
    """Prop 5: E[NLL − Σ_t H(h_t)] = 0 within 4 standard errors on both engines; on the trigram the per-word NLL of
    pure samples lies between the mode's and the held-out text's (pooled, </s> included)."""
    for name in ("trigram", "decoder"):
        t = typicality(name)
        assert abs(t["gap"]) < 4 * t["se"]
    mode, pure, held = mode_reference()["nll"], metrics("trigram", PURE)["nll"], heldout_reference("trigram")["nll"]
    assert mode < pure < held


def test_temperature_trades_quality_for_diversity():
    """§7: on both engines, as τ rises the model NLL, the oracle NLL, and distinct-2 rise and the copy rate falls. Every
    truncation rule's KL falls as its parameter loosens, and so does its model NLL rise, except typical sampling: at a
    small typical mass it drops the head of each distribution (the likeliest token is rarely the one whose surprisal is
    closest to the entropy), so its samples are less predictable than at a larger mass."""
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


def test_rules_share_one_frontier():
    """§7: every truncation setting lies within 0.1 nats of oracle NLL of the temperature curve at its own model NLL,
    on both engines: how much a rule cuts matters more than which rule it is."""
    for name in ("trigram", "decoder"):
        offs = [frontier_offset(name, s) for s in SETTINGS if s[0] != "temperature"]
        assert all(o is not None and abs(o) < 0.1 for o in offs), (name, offs)


def test_the_text_is_off_every_curve():
    """§7: no sampler reaches the held-out text. On the trigram every setting's oracle NLL exceeds its model NLL by
    more than the text's does (−0.607), by at least 0.25 nats: the gap is model error, which no decoding rule removes."""
    held = heldout_reference("trigram")
    gap = held["oracle_nll"] - held["nll"]
    assert all(metrics("trigram", s)["oracle_nll"] - metrics("trigram", s)["nll"] > gap + 0.25 for s in SETTINGS)


def test_fabrication_regimes():
    """§8, exact on the order-5 fact model's probes.
    Seen ≥ 2 times: the argmax is always right; as τ falls to 0, P(wrong) falls monotonically to 0.
    Seen once: lowering τ from 1 raises P(correct) on average, yet at τ → 0 the probes whose argmax is wrong (0.30 of
    them) fabricate with probability → 1.
    Never seen: no τ and no rule setting lifts P(correct) above 0.1."""
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
    assert max(max(v["correct"]) for v in un.values() if isinstance(v, dict)) < 0.1


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
