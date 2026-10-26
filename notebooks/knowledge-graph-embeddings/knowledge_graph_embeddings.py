"""Knowledge-graph embeddings: TransE and RotatE — the reference implementation for formalNLP's twenty-fourth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Link prediction. A knowledge graph is a set of triples (h, r, t); a scoring function ranks every entity as the
   answer to a query (h, r, ?) or (?, r, t) (Def 1). The filtered rank of the true answer discards the other known
   answers and splits ties evenly; MRR and Hits@k average it over queries (Def 2).
2. Translation. TransE embeds entities as points and relations as translations, scoring −‖e_h + w_r − e_t‖ (Def 3).
   Translations model inverse (w′ = −w) and composed (w₃ = w₁ + w₂) relations exactly, and a symmetric relation
   only at w = 0 with its pairs collapsed (Prop 1). Without the unit-norm constraint on entities, scaling alone
   drives Bordes et al.'s margin loss to zero on any embedding that separates its triples (Prop 2).
3. Rotation. Multiplying a complex number by e^{iθ} rotates the plane by θ (Lemma 1). RotatE embeds entities in ℂᵏ
   and relations as unit-modulus vectors, scoring −‖h ∘ r − t‖ (Def 4). A relation is symmetric iff every phase is
   0 or π, two are inverses iff their phases are negatives, and r₃ = r₁ ∘ r₂ iff the phases add (Thm 1).
4. Relations as group actions. Both models act on entity space by a group of isometries, ℝᵏ by translation and the
   torus U(1)ᵏ by coordinatewise rotation (Def 5). Under any isometric action, facts that hold within ε pull an
   N-to-1 relation's heads within 2ε, a symmetric pair's double action within 2ε of the identity, and the
   neighbors on an odd cycle of a symmetric relation within 2Lε (Thm 2). Both groups are abelian, so on the Cayley
   graph of S₃ no corruption can sit farther than 5ε from its query (Thm 3); the dihedral group D₃ ⊂ O(2) fits that
   graph exactly (Cor 1).
5. The bilinear foil. DistMult scores ⟨h, r, t⟩ and ComplEx Re⟨h, r, t̄⟩ (Def 6). DistMult scores every relation
   symmetrically (Prop 3); ComplEx's antisymmetric part is −2 Σ Im rᵢ Im(hᵢ t̄ᵢ) (Prop 4); and RotatE's squared
   distance is ComplEx's score with unit-modulus relations, plus the two entity norms (Prop 5).
6. Finance case study. The finance KB (``fc.WORLD``) carries each pattern: COMPETES_WITH is symmetric,
   PARENT_OF/SUBSIDIARY_OF are inverses, BASED_IN = HQ_IN ∘ LOCATED_IN. Pattern-wise held-out facts measure what
   each model infers; Topic 23's extracted KB about the held-out companies is then completed by embedding it beside
   the training roster's facts.

Training uses every corruption of every triple (the exact expectation of uniform negative sampling) under Sun et
al.'s logistic loss, full-batch Adam, float64. Cross-checks: central differences for every gradient; brute-force
ranks; exact constructions for Prop 1, Thm 1, and Cor 1; random draws for the identities.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/knowledge-graph-embeddings/knowledge_graph_embeddings.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit, log_expit

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "relation-extraction-distant-supervision"))
import relation_extraction_distant_supervision as rex  # noqa: E402 — module objects: their test_* stay out of this harness

fc, ltc, nlm = rex.fc, rex.ltc, rex.nlm

# ONE stream per random-instance claim and per initialization, each from default_rng([SEED, k]); k ≥ 3700 keeps this
# module's streams apart from Topics 1–23.
SEED = rex.SEED
_r = ltc._r
_sig = rex._sig
WORLD = fc.WORLD


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


# --------------------------------------------------------------------------------------------
# 1. Link prediction (Def 1, Def 2).
# --------------------------------------------------------------------------------------------
#
# A knowledge graph over entities 𝓔 and relations 𝓡 is a set of triples (h, r, t) (Def 1). A scoring function f
# assigns every triple a real score; link prediction answers (h, r, ?) by ranking every t ∈ 𝓔 by f(h, r, t), and
# (?, r, t) likewise. The filtered rank of the true answer counts only candidates that are not themselves known
# answers to the same query, and splits ties evenly (the realistic rank: the mean of the optimistic rank, ties
# below, and the pessimistic one, ties above). MRR is the mean reciprocal filtered rank and Hits@k the share of
# ranks ≤ k (Def 2). Both directions of every test triple are queries.

RELATIONS = rex.RELATIONS
TIE = 1e-9  # scores within TIE · (1 + |s|) of the answer's are ties (Prop 3's exact ties differ in the last bits)


@dataclass(frozen=True)
class KG:
    """A knowledge graph's entity and relation vocabularies and its training triples (sorted, so hashable)."""
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]

    @property
    def eid(self) -> dict:
        return _index(self.entities)

    @property
    def rid(self) -> dict:
        return _index(self.relations)


@lru_cache(maxsize=None)
def _index(names: tuple) -> dict:
    return {x: i for i, x in enumerate(names)}


ENTITIES = tuple(sorted(WORLD.types))


def realistic_rank(s: np.ndarray, ans: int, others) -> float:
    """Def 2: 1 + #{candidates scoring above the answer} + #{tied candidates}/2, over every candidate but the other
    known answers."""
    keep = np.ones(len(s), bool)
    keep[list(others)] = False
    keep[ans] = False
    tol = TIE * (1 + abs(s[ans]))
    above = (s > s[ans] + tol) & keep
    tied = (np.abs(s - s[ans]) <= tol) & keep
    return 1 + float(above.sum()) + float(tied.sum()) / 2


def brute_force_rank(s, ans: int, others) -> float:
    """Def 2 by enumeration: the mean rank of the answer over every ordering of the candidates that sorts them by
    score, breaking ties uniformly (its average over the tied block's positions)."""
    cand = [j for j in range(len(s)) if j == ans or j not in set(others)]
    tol = TIE * (1 + abs(s[ans]))
    above = sum(1 for j in cand if j != ans and s[j] > s[ans] + tol)
    block = [j for j in cand if abs(s[j] - s[ans]) <= tol]  # the answer and its ties
    return float(np.mean([above + 1 + p for p in range(len(block))]))


def known_answers(triples) -> tuple[dict, dict]:
    """(h, r) → its known tails, (t, r) → its known heads."""
    tails, heads = defaultdict(set), defaultdict(set)
    for h, r, t in triples:
        tails[(h, r)].add(t)
        heads[(t, r)].add(h)
    return tails, heads


def metrics(ranks) -> dict:
    ranks = np.asarray(ranks, float)
    return {"mrr": float(np.mean(1 / ranks)), "h1": float(np.mean(ranks <= 1)), "h3": float(np.mean(ranks <= 3)),
            "h10": float(np.mean(ranks <= 10)), "n": int(len(ranks))}


# --------------------------------------------------------------------------------------------
# 2–5. The models (Defs 3, 4, 6) as query points against every entity.
# --------------------------------------------------------------------------------------------
#
# Every model here scores a tail query (h, r, ?) through one query point q = φ_r(e_h) against every entity: a
# distance model by γ − ‖q − e_t‖ (TransE: φ = e_h + w_r; RotatE: φ = h ∘ r), a bilinear model by the inner product
# q · e_t (DistMult: q = h ∘ w_r; ComplEx: q = h ∘ r in ℂᵏ read as ℝ²ᵏ, since Re(z t̄) = z·t). Head queries use the
# map that solves for the head: TransE e_t − w_r, RotatE t ∘ r̄ (rotation is an isometry, so ‖e ∘ r − t‖ = ‖e − t ∘ r̄‖),
# DistMult w_r ∘ t, ComplEx r̄ ∘ t. One backward pass through the score matrix then serves all four. Complex vectors
# are stored as [real | imaginary] halves; RotatE stores its relations as phases θ, so |rᵢ| = 1 by construction.

MODELS = ("TransE", "RotatE", "DistMult", "ComplEx")
DISTANCE = ("TransE", "RotatE")
EPS = 1e-12  # inside the square root: d = √(‖q − e‖² + EPS), smooth at q = e


def cmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise complex product of [re | im] arrays."""
    k = a.shape[-1] // 2
    ar, ai, br, bi = a[..., :k], a[..., k:], b[..., :k], b[..., k:]
    return np.concatenate([ar * br - ai * bi, ar * bi + ai * br], -1)


def conj(a: np.ndarray) -> np.ndarray:
    k = a.shape[-1] // 2
    return np.concatenate([a[..., :k], -a[..., k:]], -1)


def unit(theta: np.ndarray) -> np.ndarray:
    """e^{iθ} as [cos θ | sin θ]."""
    return np.concatenate([np.cos(theta), np.sin(theta)], -1)


def as_complex(a: np.ndarray) -> np.ndarray:
    k = a.shape[-1] // 2
    return a[..., :k] + 1j * a[..., k:]


def queries(model: str, E: np.ndarray, R: np.ndarray, x: np.ndarray, r: np.ndarray, side: str) -> np.ndarray:
    """Query points for tail queries (x = heads) or head queries (x = tails)."""
    tail = side == "tail"
    if model == "TransE":
        return E[x] + R[r] if tail else E[x] - R[r]
    if model == "RotatE":
        u = unit(R[r])
        return cmul(E[x], u if tail else conj(u))
    if model == "DistMult":
        return E[x] * R[r]
    if model == "ComplEx":
        return cmul(E[x], R[r] if tail else conj(R[r]))
    raise ValueError(model)


def query_vjp(model: str, E, R, x, r, side: str, dQ: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(∂L/∂e_x, ∂L/∂(relation row)) per query, from ∂L/∂q."""
    tail = side == "tail"
    if model == "TransE":
        return dQ, (dQ if tail else -dQ)
    if model == "DistMult":
        return dQ * R[r], dQ * E[x]
    if model == "RotatE":
        u = unit(R[r])
        k = Q.shape[-1] // 2
        dth = dQ[:, :k] * -Q[:, k:] + dQ[:, k:] * Q[:, :k]  # ∂q/∂θ = i q
        return cmul(dQ, conj(u) if tail else u), (dth if tail else -dth)
    if model == "ComplEx":
        if tail:
            return cmul(dQ, conj(R[r])), cmul(dQ, conj(E[x]))
        return cmul(dQ, R[r]), conj(cmul(dQ, conj(E[x])))
    raise ValueError(model)


def score_matrix(model: str, Q: np.ndarray, E: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Scores of every query against every entity, and the distances for a distance model."""
    if model in DISTANCE:
        sq = (Q * Q).sum(1)[:, None] + (E * E).sum(1)[None] - 2 * Q @ E.T
        D = np.sqrt(np.maximum(sq, 0) + EPS)
        return gamma - D, D
    return Q @ E.T, None


@dataclass(frozen=True)
class Config:
    """One training run. k is the complex dimension: every model gets 2k real numbers per entity (TransE and
    DistMult embed in ℝ²ᵏ). RotatE keeps k phases per relation, the others 2k reals. gamma is the distance models'
    margin, lam the bilinear models' L2 penalty."""
    model: str
    k: int = 16
    gamma: float = 0.0
    lam: float = 0.0
    steps: int = 1500
    lr: float = 0.02
    seed: int = 0
    checkpoints: tuple[int, ...] = ()


@dataclass(frozen=True)
class Problem:
    """A KG's training triples as index arrays, with each query's known answers (excluded from its negatives) and
    the one-hot scatter matrices that sum per-query gradients into entity and relation rows."""
    H: np.ndarray
    Rr: np.ndarray
    T: np.ndarray
    n: int
    m: int
    pos: np.ndarray  # (2B, n) one-hot answers: tail queries first, then head queries
    ans: np.ndarray  # (2B,) the answers' columns
    negw: np.ndarray  # (2B, n) weight 1/|negatives| on each negative
    scatter_x: sparse.csr_matrix  # (n, 2B): row of the query's own entity
    scatter_r: sparse.csr_matrix  # (m, 2B)


@lru_cache(maxsize=None)
def problem(kg: KG) -> Problem:
    eid, rid = kg.eid, kg.rid
    H = np.array([eid[h] for h, _, _ in kg.triples])
    Rr = np.array([rid[r] for _, r, _ in kg.triples])
    T = np.array([eid[t] for _, _, t in kg.triples])
    n, m, B = len(kg.entities), len(kg.relations), len(H)
    tails, heads = known_answers(kg.triples)
    known = np.zeros((2 * B, n), bool)
    for b, (h, r, t) in enumerate(kg.triples):
        known[b, [eid[x] for x in tails[(h, r)]]] = True
        known[B + b, [eid[x] for x in heads[(t, r)]]] = True
    pos = np.zeros((2 * B, n))
    pos[np.arange(B), T] = 1
    pos[B + np.arange(B), H] = 1
    neg = ~known
    negw = neg / neg.sum(1, keepdims=True)
    x = np.concatenate([H, T])
    one = np.ones(2 * B)
    return Problem(H, Rr, T, n, m, pos, np.concatenate([T, H]), negw,
                   sparse.csr_matrix((one, (x, np.arange(2 * B))), shape=(n, 2 * B)),
                   sparse.csr_matrix((one, (np.concatenate([Rr, Rr]), np.arange(2 * B))), shape=(m, 2 * B)))


def loss_grad(model: str, E: np.ndarray, R: np.ndarray, P: Problem, gamma: float, lam: float,
              need_loss: bool = True):
    """Sun et al.'s logistic loss with every corruption: the mean over the 2B queries of
    −log σ(z_answer) − Σ_negatives (1/|negatives|) log σ(−z), plus lam (‖E‖²/n + ‖R‖²/m); z = γ − d or the score.
    The loss is NaN when need_loss is False (training skips it)."""
    B = len(P.H)
    Qt = queries(model, E, R, P.H, P.Rr, "tail")
    Qh = queries(model, E, R, P.T, P.Rr, "head")
    Q = np.concatenate([Qt, Qh])
    Z, D = score_matrix(model, Q, E, gamma)
    rows = np.arange(2 * B)
    L = (float(-log_expit(Z[rows, P.ans]).sum() - (P.negw * log_expit(-Z)).sum()) / (2 * B) if need_loss
         else math.nan)
    S = expit(Z)
    G = P.negw * S  # ∂L/∂Z, times 2B
    G[rows, P.ans] -= 1 - S[rows, P.ans]
    G /= 2 * B
    if D is not None:
        GD = -G / D
        dQ = Q * GD.sum(1, keepdims=True) - GD @ E
        dE = E * GD.sum(0)[:, None] - GD.T @ Q
    else:
        dQ = G @ E
        dE = G.T @ Q
    et, rt = query_vjp(model, E, R, P.H, P.Rr, "tail", dQ[:B], Qt)
    eh, rh = query_vjp(model, E, R, P.T, P.Rr, "head", dQ[B:], Qh)
    dE = dE + P.scatter_x @ np.concatenate([et, eh])
    dR = P.scatter_r @ np.concatenate([rt, rh])
    if lam:
        if need_loss:
            L += lam * (float((E * E).sum()) / P.n + float((R * R).sum()) / P.m)
        dE = dE + 2 * lam * E / P.n
        dR = dR + 2 * lam * R / P.m
    return L, dE, dR


def init(cfg: Config, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian rows of norm ≈ 1 (TransE projected onto the unit sphere); RotatE's phases uniform on [−π, π)."""
    rng = _rng(3700 + cfg.seed)
    E = rng.normal(0, 1, (n, 2 * cfg.k)) / math.sqrt(2 * cfg.k)
    if cfg.model == "RotatE":
        R = rng.uniform(-np.pi, np.pi, (m, cfg.k))
    else:
        R = rng.normal(0, 1, (m, 2 * cfg.k)) / math.sqrt(2 * cfg.k)
    if cfg.model == "TransE":
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E, R


@dataclass(frozen=True)
class Fit:
    E: np.ndarray
    R: np.ndarray
    loss: float
    snapshots: tuple  # (step, E, R) at cfg.checkpoints


@lru_cache(maxsize=None)
def fit(cfg: Config, kg: KG) -> Fit:
    """Full-batch Adam (β = 0.9, 0.999); TransE's entities are projected back onto the unit sphere after every step
    (Bordes et al.'s constraint)."""
    P = problem(kg)
    E, R = init(cfg, P.n, P.m)
    mo = [np.zeros_like(E), np.zeros_like(R)]
    ve = [np.zeros_like(E), np.zeros_like(R)]
    snaps = []
    L = math.nan
    for s in range(1, cfg.steps + 1):
        L, dE, dR = loss_grad(cfg.model, E, R, P, cfg.gamma, cfg.lam, s == cfg.steps)
        for i, (p, g) in enumerate(((E, dE), (R, dR))):
            mo[i] = 0.9 * mo[i] + 0.1 * g
            ve[i] = 0.999 * ve[i] + 0.001 * g * g
            p -= cfg.lr * (mo[i] / (1 - 0.9 ** s)) / (np.sqrt(ve[i] / (1 - 0.999 ** s)) + 1e-8)
        if cfg.model == "TransE":
            E /= np.linalg.norm(E, axis=1, keepdims=True)
        if s in cfg.checkpoints:
            snaps.append((s, E.copy(), R.copy()))
    return Fit(E, R, L, tuple(snaps))


def scores(model: str, E, R, gamma: float, x: int, r: int, side: str) -> np.ndarray:
    """f(x, r, ·) for side "tail", f(·, r, x) for "head", over every entity."""
    Q = queries(model, E, R, np.array([x]), np.array([r]), side)
    return score_matrix(model, Q, E, gamma)[0][0]


def triple_score(model: str, E, R, gamma: float, h: int, r: int, t: int) -> float:
    return float(scores(model, E, R, gamma, h, r, "tail")[t])


def evaluate(cfg: Config, kg: KG, E, R, tests, truth=WORLD.triples) -> list[float]:
    """Filtered realistic ranks of both directions of every test triple, filtered against ``truth``."""
    eid, rid = kg.eid, kg.rid
    tails, heads = known_answers(truth)
    H = np.array([eid[h] for h, _, _ in tests])
    Rr = np.array([rid[r] for _, r, _ in tests])
    T = np.array([eid[t] for _, _, t in tests])
    St = score_matrix(cfg.model, queries(cfg.model, E, R, H, Rr, "tail"), E, cfg.gamma)[0]
    Sh = score_matrix(cfg.model, queries(cfg.model, E, R, T, Rr, "head"), E, cfg.gamma)[0]
    out = []
    for b, (h, r, t) in enumerate(tests):
        out.append(realistic_rank(St[b], eid[t], [eid[x] for x in tails[(h, r)]]))
        out.append(realistic_rank(Sh[b], eid[h], [eid[x] for x in heads[(t, r)]]))
    return out


ANTISYMMETRIC = tuple(r for r in RELATIONS if r != "COMPETES_WITH")


def direction_accuracy(cfg: Config, kg: KG, E, R, facts) -> float:
    """The share of antisymmetric facts the model scores above their reversal, ties counting 1/2."""
    eid, rid = kg.eid, kg.rid
    acc = []
    for h, r, t in facts:
        if r in ANTISYMMETRIC:
            a = triple_score(cfg.model, E, R, cfg.gamma, eid[h], rid[r], eid[t])
            b = triple_score(cfg.model, E, R, cfg.gamma, eid[t], rid[r], eid[h])
            tol = TIE * (1 + abs(a))
            acc.append(0.5 if abs(a - b) <= tol else float(a > b))
    return float(np.mean(acc))


# --------------------------------------------------------------------------------------------
# 2. Translation: TransE (Def 3, Prop 1, Prop 2).
# --------------------------------------------------------------------------------------------
#
# Prop 1. If r₁'s facts hold exactly, e_t = e_h + w₁, then w₂ = −w₁ makes every reversed pair a fact of r₂ exactly,
# and if r₁ and r₂ hold exactly along a two-step path, w₃ = w₁ + w₂ holds exactly across it. A symmetric pair holds
# exactly only if e_t = e_h + w and e_h = e_t + w, so 2w = 0: w = 0 and e_h = e_t.
#
# Prop 2. Bordes et al.'s margin loss Σ [γ + d(h, r, t) − d(h′, r, t′)]₊ over triples and their corruptions. Scaling
# every entity and relation by s > 0 scales every distance by s, so each term becomes [γ − s Δ]₊ with Δ = d′ − d. If
# every Δ > 0 the loss is 0 for s ≥ γ/min Δ while every rank is unchanged: the margin measures nothing until the
# entities are pinned to the unit sphere.


def margin_loss(E, R, P: Problem, gamma: float) -> float:
    """Bordes et al.'s loss for TransE, summed over every (query, negative) pair."""
    B = len(P.H)
    Q = np.concatenate([queries("TransE", E, R, P.H, P.Rr, "tail"), queries("TransE", E, R, P.T, P.Rr, "head")])
    _, D = score_matrix("TransE", Q, E, 0.0)
    d_pos = (D * P.pos).sum(1, keepdims=True)
    return float((np.maximum(0, gamma + d_pos - D) * (P.negw > 0)).sum() / (2 * B))


def planted_transe(n: int = 30, m: int = 3, k: int = 4, stream: int = 3790) -> tuple[KG, np.ndarray, np.ndarray]:
    """A KG read off a random TransE embedding: (h, r, t) is a fact when t is the nearest entity to e_h + w_r and h
    the nearest to e_t − w_r, so the planted embedding separates every fact from its head and tail corruptions."""
    rng = _rng(stream)
    E = rng.normal(0, 1, (n, 2 * k))
    R = rng.normal(0, 1, (m, 2 * k))
    ents, rels = tuple(f"e{i:02d}" for i in range(n)), tuple(f"r{j}" for j in range(m))
    triples = []
    for h in range(n):
        for r in range(m):
            t = int(np.argmin(np.linalg.norm(E[h] + R[r] - E, axis=1)))
            if t != h and int(np.argmin(np.linalg.norm(E[t] - R[r] - E, axis=1))) == h:
                triples.append((ents[h], rels[r], ents[t]))
    return KG(ents, rels, tuple(sorted(set(triples)))), E, R


# --------------------------------------------------------------------------------------------
# 3. Rotation: RotatE (Lemma 1, Def 4, Thm 1).
# --------------------------------------------------------------------------------------------
#
# Lemma 1. (a + ib)(cos θ + i sin θ) = (a cos θ − b sin θ) + i(a sin θ + b cos θ): multiplying by e^{iθ} applies the
# rotation matrix [[cos θ, −sin θ], [sin θ, cos θ]] to (a, b). It preserves length, and e^{iα}e^{iβ} = e^{i(α+β)}.
#
# Thm 1 (Sun et al. 2019's Lemmas 1–3 say RotatE "can infer" each pattern; this makes them exact, both directions).
# With rᵢ = e^{iθᵢ}: (a) a symmetric relation's facts h ∘ r = t and t ∘ r = h hold for an entity with every hᵢ ≠ 0 iff r ∘ r = 1, i.e. θᵢ ∈ {0, π}; (b) r₂ holds exactly on r₁'s reversed pairs
# iff r₂ = r̄₁, θ₂ = −θ₁; (c) r₃ holds exactly across every r₁-then-r₂ path iff r₃ = r₁ ∘ r₂, θ₃ ≡ θ₁ + θ₂ (mod 2π).
# Unlike TransE, a symmetric relation with some θᵢ = π keeps its pairs apart (h = −t on those coordinates).


def wrap(x: np.ndarray) -> np.ndarray:
    """An angle in [−π, π)."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def phase_residuals(R: np.ndarray, rid: dict) -> dict:
    """Thm 1's three phase equations on a trained RotatE, per coordinate: distance of COMPETES_WITH's phases from
    {0, π}; |θ_PARENT + θ_SUB|; |θ_BASED − θ_HQ − θ_LOC|; and |θ_ACQUIRED − θ_PARENT| (ACQUIRED ⇒ PARENT_OF on every
    acquired pair)."""
    th = {r: R[rid[r]] for r in rid}
    c = th["COMPETES_WITH"]
    return {"symmetry": np.minimum(np.abs(wrap(c)), np.abs(wrap(c - np.pi))),
            "inverse": np.abs(wrap(th["PARENT_OF"] + th["SUBSIDIARY_OF"])),
            "composition": np.abs(wrap(th["BASED_IN"] - th["HQ_IN"] - th["LOCATED_IN"])),
            "implication": np.abs(wrap(th["ACQUIRED"] - th["PARENT_OF"]))}


# Under uniform random phases the four residuals have medians π/4 (symmetry: the distance to the nearer of two
# points, uniform on [0, π/2]) and π/2 (the others: |uniform on [−π, π)|).
UNIFORM_MEDIAN = {"symmetry": math.pi / 4, "inverse": math.pi / 2, "composition": math.pi / 2,
                  "implication": math.pi / 2}


# --------------------------------------------------------------------------------------------
# 4. Relations as group actions (Def 5, Thm 2, Thm 3, Cor 1).
# --------------------------------------------------------------------------------------------
#
# Def 5. A group G acts on entity space X by isometries; relation r is an element g_r and f(h, r, t) = −d(g_r e_h,
# e_t). TransE: G = (ℝᵏ, +) acting by translation. RotatE: G = U(1)ᵏ acting by coordinatewise multiplication on ℂᵏ.
#
# Thm 2. Let every fact of r hold within ε: d(g_r e_h, e_t) ≤ ε. (a) Two heads with one tail sit within 2ε (two
# tails of one head likewise): d(e_h, e_h′) = d(g e_h, g e_h′) ≤ d(g e_h, e_t) + d(e_t, g e_h′). (b) A symmetric pair
# moves e_h within 2ε under g²: d(g² e_h, e_h) ≤ d(g² e_h, g e_t) + d(g e_t, e_h) ≤ 2ε; for TransE that is ‖2w‖ ≤ 2ε, so
# ‖w‖ ≤ ε and ‖e_h − e_t‖ ≤ 2ε. (c) On an odd cycle x₀ … x_{L−1} of a symmetric relation, adjacent entities sit within
# 2Lε, for every isometric action.
#
# Thm 3. If G is abelian and every generator fact of S₃'s Cayley graph holds within ε (relations A: x ↦ xa and
# B: x ↦ xb, a a transposition, b a 3-cycle), then for every x the corruption (xa, B, xba) sits within 5ε of its
# query: d(g_B e_{xa}, e_{xba}) ≤ 5ε. A model that keeps every corruption at distance ≥ γ therefore has γ ≤ 5ε.
#
# Cor 1. The dihedral group D₃ ⊂ O(2), the symmetries of an equilateral triangle, is isomorphic to S₃. Acting on the
# orbit of a generic point it satisfies every fact of the Cayley graph exactly, with distinct entities.


def abelian_ratio(model: str, E, R, kg: KG) -> dict:
    """ε = the largest distance of a fact, γ_min = the smallest distance of a corruption; Thm 3 says γ_min ≤ 5ε."""
    P = problem(kg)
    Q = np.concatenate([queries(model, E, R, P.H, P.Rr, "tail"), queries(model, E, R, P.T, P.Rr, "head")])
    D = np.sqrt(((Q[:, None, :] - E[None]) ** 2).sum(-1))
    eps, gmin = float(D[P.pos > 0].max()), float(D[P.negw > 0].min())
    return {"eps": eps, "gamma_min": gmin, "ratio": gmin / eps}


S3 = tuple(sorted(itertools.permutations(range(3))))
S3_A, S3_B = (1, 0, 2), (1, 2, 0)  # a transposition and a 3-cycle


def s3_mul(x: tuple, g: tuple) -> tuple:
    """The permutation x ∘ g (apply g first)."""
    return tuple(x[g[i]] for i in range(3))


def _name(x: tuple) -> str:
    return "".join(map(str, x))


@lru_cache(maxsize=1)
def s3_kg() -> KG:
    """The Cayley graph of S₃ with generators a and b as a KG: 6 entities, 12 facts."""
    ents = tuple(_name(x) for x in S3)
    trip = [(_name(x), "A", _name(s3_mul(x, S3_A))) for x in S3] + [(_name(x), "B", _name(s3_mul(x, S3_B)))
                                                                    for x in S3]
    return KG(ents, ("A", "B"), tuple(sorted(trip)))


def _perm_matrix(g: tuple) -> np.ndarray:
    """g's action on the plane through the triangle with vertices e^{2πik/3}: vertex k ↦ vertex g(k)."""
    V = np.array([[math.cos(2 * math.pi * k / 3), math.sin(2 * math.pi * k / 3)] for k in range(3)]).T  # 2 × 3
    P = np.zeros((3, 3))
    P[list(g), range(3)] = 1
    return V @ P @ np.linalg.pinv(V)


def _inverse(x: tuple) -> tuple:
    return tuple(sorted(range(3), key=lambda i: x[i]))


DIHEDRAL_POINT = np.array([0.9, 0.3])


@lru_cache(maxsize=1)
def dihedral_fit() -> dict:
    """Cor 1's construction: e_x = M(x⁻¹) p for a generic point p, so e_{xg} = M(g)⁻¹ e_x, and relation g acts by the
    orthogonal matrix M(g)⁻¹ (a reflection for a, a rotation by ±2π/3 for b)."""
    E = {x: _perm_matrix(_inverse(x)) @ DIHEDRAL_POINT for x in S3}
    G = {"A": np.linalg.inv(_perm_matrix(S3_A)), "B": np.linalg.inv(_perm_matrix(S3_B))}
    kg = s3_kg()
    err = max(float(np.linalg.norm(G[r] @ E[tuple(map(int, h))] - E[tuple(map(int, t))])) for h, r, t in kg.triples)
    sep = min(float(np.linalg.norm(E[x] - E[y])) for x in S3 for y in S3 if x != y)
    gmin = min(float(np.linalg.norm(G[r] @ E[tuple(map(int, h))] - E[x])) for h, r, t in kg.triples for x in S3
               if _name(x) != t)
    return {"E": E, "G": G, "max_fact_error": err, "separation": sep, "gamma_min": gmin,
            "commute_gap": float(np.linalg.norm(G["A"] @ G["B"] - G["B"] @ G["A"])),
            "det": {r: float(np.linalg.det(g)) for r, g in G.items()}}


S3_K = (1, 2, 8)  # complex dimensions for the abelian runs
S3_GAMMAS = (1.0, 2.0, 4.0)
S3_SEEDS = (0, 1)


@lru_cache(maxsize=None)
def s3_best(model: str, k: int) -> dict:
    """The largest γ_min/ε over S3_GAMMAS × S3_SEEDS for an abelian model on the Cayley graph (2,000 steps each)."""
    kg = s3_kg()
    best = None
    for g in S3_GAMMAS:
        for seed in S3_SEEDS:
            cfg = Config(model, k=k, gamma=g, steps=2000, seed=10 + seed)
            f = fit(cfg, kg)
            res = abelian_ratio(model, f.E, f.R, kg) | {"gamma": g, "seed": seed, "loss": f.loss, "fit": f}
            if best is None or res["ratio"] > best["ratio"]:
                best = res
    return best


# --------------------------------------------------------------------------------------------
# 5. The bilinear foil (Def 6, Props 3–5).
# --------------------------------------------------------------------------------------------
#
# Prop 3. ⟨h, r, t⟩ = Σ hᵢ rᵢ tᵢ is unchanged by swapping h and t, so DistMult scores (t, r, h) exactly as (h, r, t)
# for every relation: on an antisymmetric fact it ties its reversal.
# Prop 4. With z = h ∘ t̄, f(h, r, t) − f(t, r, h) = Re Σ rᵢ zᵢ − Re Σ rᵢ z̄ᵢ = −2 Σ Im rᵢ Im zᵢ: ComplEx is symmetric in
# r iff Im r = 0 (DistMult), and antisymmetric iff Re r = 0.
# Prop 5. If |rᵢ| = 1, ‖h ∘ r − t‖² = ‖h‖² + ‖t‖² − 2 Re⟨h, r, t̄⟩.


def complex_score(h: np.ndarray, r: np.ndarray, t: np.ndarray) -> float:
    """ComplEx on complex vectors: Re Σ hᵢ rᵢ t̄ᵢ."""
    return float(np.real(np.sum(h * r * np.conj(t))))


# --------------------------------------------------------------------------------------------
# 6. Finance case study: patterns held out, and completing Topic 23's extracted KB.
# --------------------------------------------------------------------------------------------
#
# Pattern tests. From the full WORLD KB we hide facts each pattern entails from what stays: one direction of every
# COMPETES_WITH pair (symmetry), one fact of every PARENT_OF/SUBSIDIARY_OF pair (inverse), half the BASED_IN facts
# (composition through HQ_IN and LOCATED_IN), and every subsidiary's IN_SECTOR (inheritance: subsidiaries share their
# parent's sector). Hyperparameters are chosen on split 1's held-out facts; the page reports split 0, three seeds.

PATTERNS = ("symmetry", "inverse", "composition", "inheritance")


@lru_cache(maxsize=None)
def pattern_split(split: int) -> tuple[KG, dict]:
    rng = _rng(3710 + split)
    T = sorted(WORLD.triples)
    test = {}
    pairs = sorted({tuple(sorted((h, t))) for h, r, t in T if r == "COMPETES_WITH"})
    test["symmetry"] = tuple((a, "COMPETES_WITH", b) if rng.random() < 0.5 else (b, "COMPETES_WITH", a)
                             for a, b in pairs)
    par = sorted((h, t) for h, r, t in T if r == "PARENT_OF")
    test["inverse"] = tuple((p, "PARENT_OF", s) if rng.random() < 0.5 else (s, "SUBSIDIARY_OF", p) for p, s in par)
    based = [x for x in T if x[1] == "BASED_IN"]
    test["composition"] = tuple(based[i] for i in sorted(rng.permutation(len(based))[: len(based) // 2]))
    subs = {s for _, s in par}
    test["inheritance"] = tuple(x for x in T if x[1] == "IN_SECTOR" and x[0] in subs)
    held = set().union(*map(set, test.values()))
    return KG(ENTITIES, RELATIONS, tuple(x for x in T if x not in held)), test


FULL_KG = KG(ENTITIES, RELATIONS, tuple(sorted(WORLD.triples)))

# The validation grid: split 1, averaged over SEEDS. Steps are read off each 1,500-step run's checkpoints.
GRID = {"TransE": ("gamma", (0.5, 1.0, 1.5)), "RotatE": ("gamma", (3.0, 6.0, 9.0)),
        "DistMult": ("lam", (0.0003, 0.001, 0.003)), "ComplEx": ("lam", (0.003, 0.01, 0.03))}
STEP_GRID = (250, 500, 1000, 1500)
SEEDS = (0, 1, 2)


def _mean_pattern_mrr(cfg: Config, kg: KG, E, R, test: dict) -> float:
    return float(np.mean([metrics(evaluate(cfg, kg, E, R, test[p]))["mrr"] for p in PATTERNS]))


@lru_cache(maxsize=None)
def tuning(model: str) -> dict:
    """Mean pattern MRR on split 1, averaged over SEEDS, for every grid value and step count; the chosen config
    maximizes it."""
    kg, test = pattern_split(1)
    key, values = GRID[model]
    table = defaultdict(list)
    for v in values:
        for seed in SEEDS:
            cfg = Config(model, steps=max(STEP_GRID), seed=seed, checkpoints=STEP_GRID, **{key: v})
            for s, E, R in fit(cfg, kg).snapshots:
                table[(v, s)].append(_mean_pattern_mrr(cfg, kg, E, R, test))
    table = {vs: float(np.mean(x)) for vs, x in table.items()}
    v, s = max(table, key=table.get)
    return {"table": table, "chosen": Config(model, steps=s, **{key: v})}


def chosen(model: str, seed: int = 0) -> Config:
    return replace(tuning(model)["chosen"], seed=seed)


@lru_cache(maxsize=None)
def pattern_results(model: str) -> dict:
    """Split 0, per seed: filtered MRR and Hits@{1,3,10} per pattern, and direction accuracy on all WORLD facts."""
    kg, test = pattern_split(0)
    out = {p: [] for p in PATTERNS} | {"direction": [], "train_mrr": []}
    for seed in SEEDS:
        cfg = chosen(model, seed)
        f = fit(cfg, kg)
        for p in PATTERNS:
            out[p].append(metrics(evaluate(cfg, kg, f.E, f.R, test[p])))
        out["direction"].append(direction_accuracy(cfg, kg, f.E, f.R, sorted(WORLD.triples)))
        out["train_mrr"].append(metrics(evaluate(cfg, kg, f.E, f.R, kg.triples))["mrr"])
    return out


def mean_sd(xs) -> tuple[float, float]:
    return float(np.mean(xs)), float(np.std(xs))


@lru_cache(maxsize=1)
def rotate_full() -> Fit:
    """RotatE (its chosen config, seed 0) on the whole WORLD KB: the model whose phases the page reads."""
    return fit(chosen("RotatE"), FULL_KG)


@lru_cache(maxsize=1)
def competitor_graph() -> dict:
    """COMPETES_WITH as an undirected graph: degrees and triangles (Thm 2(c)'s odd cycles)."""
    adj = defaultdict(set)
    for h, r, t in WORLD.triples:
        if r == "COMPETES_WITH":
            adj[h].add(t)
    tri = sorted({tuple(sorted((a, b, c))) for a in adj for b in adj[a] for c in adj[b] if c in adj[a] and c != a})
    return {"adj": {k: tuple(sorted(v)) for k, v in adj.items()}, "triangles": tri,
            "degrees": Counter(len(v) for v in adj.values())}


# Completion. The KB knows every WORLD fact among the training roster's entities (and the shared GPE and SECTOR
# entities); Topic 23's pipeline read the held-out companies' filings and extracted facts with scores. We train on
# the known facts plus the extracted ones above a threshold, then rank the WORLD facts about held-out entities that
# extraction missed.

THRESHOLDS = (0.5, 0.1)


@lru_cache(maxsize=1)
def extracted() -> dict:
    """Topic 23's ``extracted_kb()``: MIL on Topic 22's spans over the held-out roster's filings, noisy-OR facts."""
    return rex.extracted_kb()


@lru_cache(maxsize=1)
def held_entities() -> frozenset:
    return frozenset(fc._roster_closure(rex.HELD_ROSTER)[1])


@lru_cache(maxsize=None)
def completion_problem(tau: float) -> dict:
    held = held_entities()
    known = [x for x in sorted(WORLD.triples) if x[0] not in held and x[2] not in held]
    about = [x for x in sorted(WORLD.triples) if x[0] in held or x[2] in held]
    ext = sorted(f for f, v in extracted().items() if v > tau)
    missing = tuple(x for x in about if x not in set(ext))
    covered = {e for f in ext for e in (f[0], f[2])}
    return {"kg": KG(ENTITIES, RELATIONS, tuple(sorted(set(known) | set(ext)))), "known": len(known),
            "about": len(about), "extracted": tuple(ext), "true": sum(f in WORLD.triples for f in ext),
            "missing": missing, "uncovered": tuple(sorted(held - covered))}


@lru_cache(maxsize=None)
def completion_results(model: str, tau: float) -> dict:
    """Per seed and per missing relation: filtered ranks of both directions of the missing facts."""
    cp = completion_problem(tau)
    rels = sorted({r for _, r, _ in cp["missing"]})
    out = {r: [] for r in rels}
    for seed in SEEDS:
        cfg = chosen(model, seed)
        f = fit(cfg, cp["kg"])
        for r in rels:
            out[r].append(metrics(evaluate(cfg, cp["kg"], f.E, f.R, [x for x in cp["missing"] if x[1] == r])))
    return out


@lru_cache(maxsize=None)
def completion_fit(model: str, tau: float) -> tuple[Config, Fit]:
    cfg = chosen(model)
    return cfg, fit(cfg, completion_problem(tau)["kg"])


# --------------------------------------------------------------------------------------------
# The lab's toy (tab 1) and its example.
# --------------------------------------------------------------------------------------------
#
# In the plane (k = 1), a symmetric relation on a pair a, b = −a and on a triangle a, ωa, ω²a (ω = e^{2πi/3}), each
# pair stated both ways. A rotation by θ and a translation by s along a → b; the lab shows every fact's distance on a
# grid. Rotation by π satisfies the pair exactly (Thm 1(a)); no translation brings the pair's two facts below |b − a|
# (d₁ + d₂ ≥ 2|b − a|, so max(d₁, d₂) ≥ |b − a|); on the triangle no rotation satisfies the six facts (Thm 2(c)).

TOY_A = complex(0.8, 0.6)
TOY_THETA = tuple(range(0, 361, 5))  # degrees
TOY_SHIFT = tuple(round(-2 + 0.1 * i, 1) for i in range(41))


def _toy_entities(case: str) -> list[complex]:
    if case == "pair":
        return [TOY_A, -TOY_A]
    w = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    return [TOY_A, TOY_A * w, TOY_A * w * w]


def _toy_facts(case: str) -> list[tuple[int, int]]:
    n = len(_toy_entities(case))
    return [(i, j) for i in range(n) for j in range(n) if i != j]


@lru_cache(maxsize=1)
def toy_grid() -> dict:
    out = {}
    for case in ("pair", "triangle"):
        P = _toy_entities(case)
        F = _toy_facts(case)
        a, b = P[0], P[1]
        axis = (b - a) / abs(b - a)
        rot = [[abs(P[i] * complex(math.cos(math.radians(th)), math.sin(math.radians(th))) - P[j]) for i, j in F]
               for th in TOY_THETA]
        tr = [[abs(P[i] + s * axis - P[j]) for i, j in F] for s in TOY_SHIFT]
        out[case] = {"points": [[p.real, p.imag] for p in P], "facts": F, "rotation": rot, "translation": tr}
    return out


@lru_cache(maxsize=1)
def lab_example() -> tuple:
    """The overview's query: the missing SUBSIDIARY_OF fact (at τ = 0.5) that extraction scored highest below the
    threshold."""
    cp = completion_problem(0.5)
    miss = [x for x in cp["missing"] if x[1] == "SUBSIDIARY_OF"]
    return max(miss, key=lambda x: extracted().get(x, 0.0))


# --------------------------------------------------------------------------------------------
# The harness: every claim the page makes.
# --------------------------------------------------------------------------------------------


def test_kb_as_the_page_describes_it():
    """§6's KB: sizes, the pattern tests' sizes, which test facts the KB's rules entail from what stays, and the
    competitor graph's triangles."""
    assert (len(ENTITIES), len(RELATIONS), len(WORLD.triples)) == (114, 10, 254)
    assert Counter(WORLD.types.values()) == {"PERSON": 48, "ORG": 37, "GPE": 23, "SECTOR": 6}
    kg, test = pattern_split(0)
    assert {p: len(v) for p, v in test.items()} == {"symmetry": 25, "inverse": 13, "composition": 18,
                                                    "inheritance": 13}
    assert len(kg.triples) == 254 - 69
    entailed = rex.closure(set(kg.triples))
    for p in ("symmetry", "inverse", "composition"):
        assert set(test[p]) <= entailed, p
    assert not set(test["inheritance"]) & entailed  # inheritance is a regularity of the world, not a KB rule
    cg = competitor_graph()
    assert len(cg["triangles"]) == 8 and cg["degrees"] == {1: 6, 2: 10, 3: 8}


def test_def2_realistic_rank_is_the_mean_over_tie_breaks():
    rng = _rng(3720)
    for _ in range(300):
        n = int(rng.integers(2, 12))
        s = rng.integers(0, 4, n).astype(float)  # many ties
        ans = int(rng.integers(n))
        others = [j for j in range(n) if j != ans and rng.random() < 0.3]
        assert realistic_rank(s, ans, others) == brute_force_rank(s, ans, others)
    assert metrics([1, 2, 4, 10, 11]) == {"mrr": (1 + 1 / 2 + 1 / 4 + 1 / 10 + 1 / 11) / 5, "h1": 0.2, "h3": 0.4,
                                          "h10": 0.8, "n": 5}


def test_gradients_match_central_differences():
    kg, _ = pattern_split(0)
    sub = KG(ENTITIES, RELATIONS, kg.triples[::7])
    P = problem(sub)
    for model in MODELS:
        E, R = init(Config(model, k=3, seed=99), P.n, P.m)
        _, dE, dR = loss_grad(model, E, R, P, 1.5, 0.03)
        rng = _rng(3721)
        for A, G in ((E, dE), (R, dR)):
            for _ in range(20):
                i = tuple(int(rng.integers(s)) for s in A.shape)
                h = 1e-6
                A[i] += h
                up = loss_grad(model, E, R, P, 1.5, 0.03)[0]
                A[i] -= 2 * h
                dn = loss_grad(model, E, R, P, 1.5, 0.03)[0]
                A[i] += h
                assert abs((up - dn) / (2 * h) - G[i]) < 1e-7 * (1 + abs(G[i])), (model, i)


def test_prop1_translations_invert_and_compose_exactly():
    rng = _rng(3722)
    e, w1, w2 = (rng.normal(size=(5, 8)) for _ in range(3))
    t1 = e + w1  # r₁'s facts, exact
    assert np.allclose(t1 + (-w1), e, atol=1e-14)  # w = −w₁ holds on every reversed pair
    t2 = t1 + w2
    assert np.allclose(e + (w1 + w2), t2, atol=1e-14)  # w₃ = w₁ + w₂ across every path
    for w in rng.normal(size=(200, 8)):  # a symmetric pair: max(‖e_h + w − e_t‖, ‖e_t + w − e_h‖) ≥ ‖w‖
        h, t = rng.normal(size=(2, 8))
        assert max(np.linalg.norm(h + w - t), np.linalg.norm(t + w - h)) >= np.linalg.norm(w) - 1e-12


def test_prop2_scaling_alone_drives_the_margin_loss_to_zero():
    kg, E, R = planted_transe()
    P = problem(kg)
    Q = np.concatenate([queries("TransE", E, R, P.H, P.Rr, "tail"), queries("TransE", E, R, P.T, P.Rr, "head")])
    D = score_matrix("TransE", Q, E, 0.0)[1]
    d_pos = D[np.arange(len(Q)), P.ans]
    gap = (D - d_pos[:, None])[P.negw > 0]
    assert gap.min() > 0  # the planted embedding separates every fact from its corruptions
    s_star = 1.0 / gap.min()
    losses = [margin_loss(s * E, s * R, P, 1.0) for s in (0.25, 0.5, 1.0, 2.0, s_star, 2 * s_star)]
    assert all(a >= b for a, b in zip(losses, losses[1:])) and losses[0] > 0
    assert losses[-2] == 0.0 and losses[-1] == 0.0
    for s in (1.0, 2 * s_star):  # ranks are unchanged by scaling
        Ds = score_matrix("TransE", s * Q, s * E, 0.0)[1]
        assert np.array_equal(np.argsort(Ds, 1, kind="stable"), np.argsort(D, 1, kind="stable"))


def test_lemma1_multiplying_by_a_unit_complex_number_rotates():
    rng = _rng(3723)
    for th, z in zip(rng.uniform(-np.pi, np.pi, 100), rng.normal(size=(100, 2))):
        M = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        w = complex(*z) * complex(math.cos(th), math.sin(th))
        assert np.allclose([w.real, w.imag], M @ z, atol=1e-14)
        assert abs(abs(w) - np.linalg.norm(z)) < 1e-14
    a, b = rng.uniform(-np.pi, np.pi, 2)
    assert np.allclose(cmul(unit(np.array([a])), unit(np.array([b]))), unit(np.array([a + b])), atol=1e-14)


def test_thm1_rotation_patterns_are_phase_equations():
    rng = _rng(3724)
    k = 6
    h = as_complex(rng.normal(size=2 * k))
    sym = rng.choice([0.0, np.pi], k)
    r = np.exp(1j * sym)
    t = h * r
    assert np.allclose(t * r, h, atol=1e-14)  # (a): θ ∈ {0, π} satisfies both facts
    for i in range(k):  # one phase off {0, π}: the reversed fact fails by |hᵢ| |rᵢ² − 1|
        th = sym.copy()
        th[i] += 0.3
        r2 = np.exp(1j * th)
        assert np.linalg.norm(h * r2 * r2 - h) >= abs(h[i]) * abs(np.exp(0.6j) - 1) - 1e-12
    assert np.linalg.norm(t - h) > 0.1 or np.all(sym == 0)  # θ = π keeps the pair apart, unlike TransE
    th1, th2 = rng.uniform(-np.pi, np.pi, (2, k))
    t1 = h * np.exp(1j * th1)
    assert np.allclose(t1 * np.exp(-1j * th1), h, atol=1e-14)  # (b): inverse phases −θ₁
    t2 = t1 * np.exp(1j * th2)
    assert np.allclose(h * np.exp(1j * (th1 + th2)), t2, atol=1e-13)  # (c): composed phases θ₁ + θ₂
    assert np.linalg.norm(h * np.exp(1j * (th1 + th2 + 0.2)) - t2) > 0.01


def test_thm1_trained_phases_obey_the_equations_and_incompleteness_breaks_them():
    """On the full KB the trained phases sit on Thm 1's solutions; on split 0, whose hidden facts are negatives
    during training, the symmetric and inverse phases move off them."""
    full = {k: float(np.median(v)) for k, v in phase_residuals(rotate_full().R, FULL_KG.rid).items()}
    assert full["symmetry"] < 0.001 and full["inverse"] < 0.001 and full["composition"] < 0.05
    assert full["implication"] < 0.1
    split = {k: float(np.median(v)) for k, v in phase_residuals(fit(chosen("RotatE"), pattern_split(0)[0]).R,
                                                                FULL_KG.rid).items()}
    assert split["symmetry"] > 0.3 and split["inverse"] > 0.3
    assert all(full[k] < 0.1 * UNIFORM_MEDIAN[k] for k in full)


def act(model: str, R: np.ndarray, X: np.ndarray, r: int) -> np.ndarray:
    """Def 5's action of relation r on arbitrary points X (rows), for the two isometric models."""
    return queries(model, X, R, np.arange(len(X)), np.full(len(X), r), "tail")


def _isometric_fits():
    yield "TransE", fit(chosen("TransE"), FULL_KG)
    yield "RotatE", rotate_full()


def _fact_distance(model, f: Fit, h, r, t) -> float:
    eid, rid = FULL_KG.eid, FULL_KG.rid
    q = queries(model, f.E, f.R, np.array([eid[h]]), np.array([rid[r]]), "tail")[0]
    return float(np.linalg.norm(q - f.E[eid[t]]))


def test_thm2_isometries_collapse_n_to_1_heads_symmetric_pairs_and_odd_cycles():
    eid, rid = FULL_KG.eid, FULL_KG.rid
    for model, f in _isometric_fits():
        for r in ("IN_SECTOR", "HQ_IN", "BASED_IN"):  # (a)
            by_tail = defaultdict(list)
            for h, rr, t in WORLD.triples:
                if rr == r:
                    by_tail[t].append(h)
            eps = max(_fact_distance(model, f, h, r, t) for t, hs in by_tail.items() for h in hs)
            for hs in by_tail.values():
                for a, b in itertools.combinations(hs, 2):
                    assert np.linalg.norm(f.E[eid[a]] - f.E[eid[b]]) <= 2 * eps + 1e-9
        comp = [(h, t) for h, r, t in WORLD.triples if r == "COMPETES_WITH"]
        eps = max(_fact_distance(model, f, h, "COMPETES_WITH", t) for h, t in comp)
        for h, _ in comp:  # (b): g² moves e_h at most 2ε
            g2 = act(model, f.R, act(model, f.R, f.E[[eid[h]]], rid["COMPETES_WITH"]), rid["COMPETES_WITH"])
            assert np.linalg.norm(g2[0] - f.E[eid[h]]) <= 2 * eps + 1e-9
        for tri in competitor_graph()["triangles"]:  # (c), L = 3
            e3 = max(_fact_distance(model, f, a, "COMPETES_WITH", b) for a in tri for b in tri if a != b)
            for a, b in itertools.combinations(tri, 2):
                assert np.linalg.norm(f.E[eid[a]] - f.E[eid[b]]) <= 6 * e3 + 1e-9
    rng = _rng(3725)  # (c) on random cycles of every odd length, random isometries
    for L in (3, 5, 7):
        for _ in range(50):
            k = 4
            X = rng.normal(size=(L, 2 * k))
            th = rng.uniform(-np.pi, np.pi, (1, k))
            gx = lambda x: cmul(x, unit(th))  # noqa: E731
            eps = max(np.linalg.norm(gx(X[[i]]) - X[[(i + 1) % L]]) for i in range(L))
            eps = max(eps, max(np.linalg.norm(gx(X[[(i + 1) % L]]) - X[[i]]) for i in range(L)))
            for i in range(L):
                assert np.linalg.norm(X[i] - X[(i + 1) % L]) <= 2 * L * eps + 1e-9


def test_thm3_abelian_models_cannot_separate_s3s_corruptions():
    kg = s3_kg()
    rng = _rng(3726)
    for model in DISTANCE:
        for _ in range(300):
            E, R = init(Config(model, k=int(rng.integers(1, 5)), seed=int(rng.integers(10 ** 6))), 6, 2)
            res = abelian_ratio(model, E, R, kg)
            assert res["gamma_min"] <= 5 * res["eps"] + 1e-9
        for k in S3_K:
            b = s3_best(model, k)
            assert b["ratio"] <= 5 and b["gamma_min"] <= 5 * b["eps"]
    t1 = s3_best("TransE", 1)  # its best plane run collapses all six entities: every distance is ‖w‖ = 2
    assert np.ptp(t1["fit"].E, axis=0).max() < 1e-3 and abs(t1["eps"] - 2) < 1e-3 and abs(t1["ratio"] - 1) < 1e-3
    # The proof's chain, per x, on a trained model: d(g_B e_{xa}, e_{xba}) ≤ 5ε.
    b = s3_best("RotatE", 8)
    f, eid = b["fit"], kg.eid
    for x in S3:
        xa, xba = _name(s3_mul(x, S3_A)), _name(s3_mul(s3_mul(x, S3_B), S3_A))
        q = queries("RotatE", f.E, f.R, np.array([eid[xa]]), np.array([1]), "tail")[0]
        assert np.linalg.norm(q - f.E[eid[xba]]) <= 5 * b["eps"] + 1e-9


def test_cor1_the_dihedral_group_fits_s3_exactly():
    d = dihedral_fit()
    assert d["max_fact_error"] < 1e-12 and d["separation"] > 0.5 and d["gamma_min"] > 0.5
    assert abs(d["det"]["A"] + 1) < 1e-12 and abs(d["det"]["B"] - 1) < 1e-12  # a reflection and a rotation
    assert d["commute_gap"] > 1  # D₃ is not abelian
    for g in d["G"].values():
        assert np.allclose(g @ g.T, np.eye(2), atol=1e-12)
    assert s3_mul(S3_A, S3_B) != s3_mul(S3_B, S3_A)


def test_props3to5_bilinear_identities():
    rng = _rng(3727)
    for _ in range(200):
        k = 5
        h, r, t = (rng.normal(size=k) for _ in range(3))
        assert abs(np.sum(h * r * t) - np.sum(t * r * h)) < 1e-12  # Prop 3
        hc, rc, tc = (as_complex(rng.normal(size=2 * k)) for _ in range(3))
        z = hc * np.conj(tc)
        lhs = complex_score(hc, rc, tc) - complex_score(tc, rc, hc)
        assert abs(lhs + 2 * np.sum(rc.imag * z.imag)) < 1e-12  # Prop 4
        assert abs(complex_score(hc, rc.real + 0j, tc) - complex_score(tc, rc.real + 0j, hc)) < 1e-12
        ri = 1j * rc.imag
        assert abs(complex_score(hc, ri, tc) + complex_score(tc, ri, hc)) < 1e-12
        u = np.exp(1j * rng.uniform(-np.pi, np.pi, k))  # Prop 5
        lhs = np.linalg.norm(hc * u - tc) ** 2
        assert abs(lhs - (np.linalg.norm(hc) ** 2 + np.linalg.norm(tc) ** 2 - 2 * complex_score(hc, u, tc))) < 1e-10


def test_tuning_picks_the_validation_argmax():
    for model in MODELS:
        tu = tuning(model)
        key, values = GRID[model]
        best = max(tu["table"].values())
        c = tu["chosen"]
        assert tu["table"][(getattr(c, key), c.steps)] == best
        assert len(tu["table"]) == len(values) * len(STEP_GRID)


def test_pattern_results_the_page_reports():
    res = {m: pattern_results(m) for m in MODELS}
    mrr = {m: {p: mean_sd([x["mrr"] for x in res[m][p]])[0] for p in PATTERNS} for m in MODELS}
    direction = {m: mean_sd(res[m]["direction"])[0] for m in MODELS}
    assert all(x["mrr"] == 1.0 for x in res["DistMult"]["symmetry"])  # symmetric by construction (Prop 3)
    assert direction["DistMult"] == 0.5  # and it ties every reversal
    assert min(direction[m] for m in ("TransE", "RotatE", "ComplEx")) > 0.95
    assert max(MODELS, key=lambda m: mrr[m]["composition"]) == "RotatE"
    assert max(MODELS, key=lambda m: mrr[m]["inheritance"]) == "TransE"
    assert min(MODELS, key=lambda m: mrr[m]["symmetry"]) == "RotatE"
    assert max(mrr["DistMult"]["inheritance"], mrr["ComplEx"]["inheritance"]) < 0.1
    assert mrr["TransE"]["composition"] < mrr["RotatE"]["composition"] - 0.3  # Prop 1's exactness is not learned


def test_completing_the_extracted_kb():
    cp = completion_problem(0.5)
    assert (len(cp["extracted"]), cp["true"], cp["known"], cp["about"]) == (45, 45, 188, 66)
    assert Counter(r for _, r, _ in cp["missing"]) == {"COMPETES_WITH": 26, "SUBSIDIARY_OF": 2}
    assert cp["uncovered"] == ()  # every held-out entity has at least one extracted fact
    for m in ("RotatE", "DistMult", "ComplEx"):  # the inverse of an extracted PARENT_OF, ranked first
        assert all(x["mrr"] == 1.0 for x in completion_results(m, 0.5)["SUBSIDIARY_OF"])
    h10 = {m: mean_sd([x["h10"] for x in completion_results(m, 0.5)["COMPETES_WITH"]])[0] for m in MODELS}
    assert max(h10, key=h10.get) == "TransE"
    noisy = completion_problem(0.1)
    assert (len(noisy["extracted"]), noisy["true"]) == (64, 47)
    assert Counter(r for _, r, _ in noisy["missing"]) == {"COMPETES_WITH": 26}
    h10n = {m: mean_sd([x["h10"] for x in completion_results(m, 0.1)["COMPETES_WITH"]])[0] for m in MODELS}
    assert h10n["TransE"] < h10["TransE"] - 0.1  # false triples cost TransE most
    assert h10n["TransE"] - h10["TransE"] == min(h10n[m] - h10[m] for m in MODELS)
    h, r, t = lab_example()
    assert r == "SUBSIDIARY_OF" and 0.3 < extracted()[(h, r, t)] < 0.5


# --------------------------------------------------------------------------------------------
# Summary, the lab's payload, and the runner.
# --------------------------------------------------------------------------------------------

VIZ_JSON = _HERE.parent.parent / "src" / "components" / "viz" / "data" / "knowledge-graph-embedding-lab.json"
TOP = 5  # candidates the link-prediction tab lists per query


def _fmt(*v) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def summary() -> dict:
    out = {}
    for m in MODELS:
        c = chosen(m)
        out[f"{m}: chosen gamma, lam, steps | validation mean MRR"] = (
            f"{c.gamma:g} {c.lam:g} {c.steps} | {max(tuning(m)['table'].values()):.6f}")
        res = pattern_results(m)
        for p in PATTERNS:
            ms = [mean_sd([x[key] for x in res[p]]) for key in ("mrr", "h1", "h3", "h10")]
            out[f"{m} [{p}]: MRR (sd) H@1 H@3 H@10"] = f"{ms[0][0]:.6f} ({ms[0][1]:.6f}) " + _fmt(*(x[0] for x in ms[1:]))
        out[f"{m}: direction accuracy, train MRR"] = _fmt(mean_sd(res["direction"])[0],
                                                           mean_sd(res["train_mrr"])[0])
    for name, R in (("full", rotate_full().R), ("split 0", fit(chosen("RotatE"), pattern_split(0)[0]).R)):
        res = phase_residuals(R, FULL_KG.rid)
        out[f"RotatE phases [{name}]: median residual sym inv comp impl"] = _fmt(
            *(float(np.median(res[k])) for k in ("symmetry", "inverse", "composition", "implication")))
    for m in DISTANCE:
        for k in S3_K:
            b = s3_best(m, k)
            out[f"S3 {m} k={k}: eps, gamma_min, ratio (gamma, seed)"] = (
                f"{_fmt(b['eps'], b['gamma_min'], b['ratio'])} ({b['gamma']:g}, {b['seed']})")
    d = dihedral_fit()
    out["D3: max fact error, separation, gamma_min, commute gap"] = (
        f"{d['max_fact_error']:.3e} {_fmt(d['separation'], d['gamma_min'], d['commute_gap'])}")
    for tau in THRESHOLDS:
        cp = completion_problem(tau)
        out[f"completion tau={tau:g}: train, extracted, true, missing"] = (
            f"{len(cp['kg'].triples)} {len(cp['extracted'])} {cp['true']} {len(cp['missing'])}")
        for m in MODELS:
            for r, v in completion_results(m, tau).items():
                out[f"completion tau={tau:g} {m} [{r}]: MRR (sd) H@10"] = (
                    f"{mean_sd([x['mrr'] for x in v])[0]:.6f} ({mean_sd([x['mrr'] for x in v])[1]:.6f}) "
                    f"{mean_sd([x['h10'] for x in v])[0]:.6f}")
    h, r, t = lab_example()
    out["lab example: fact, extracted score"] = f"({h}, {r}, {t}) {extracted()[(h, r, t)]:.6f}"
    return out


def _ms(xs) -> list:
    m, s = mean_sd(xs)
    return [_r(m), _r(s)]


def _queries_tab() -> dict:
    """Seed 0 on split 0: every test triple's tail query, each model's top candidates (filtered) and the answer's
    filtered rank."""
    kg, test = pattern_split(0)
    eid, rid = kg.eid, kg.rid
    tails, _ = known_answers(WORLD.triples)
    rows = []
    for p in PATTERNS:
        for h, r, t in test[p]:
            row = {"pattern": p, "h": eid[h], "r": r, "t": eid[t], "models": {}}
            others = [eid[x] for x in tails[(h, r)] if x != t]
            for m in MODELS:
                cfg = chosen(m)
                f = fit(cfg, kg)
                s = scores(m, f.E, f.R, cfg.gamma, eid[h], rid[r], "tail")
                s2 = s.copy()
                s2[others] = -np.inf
                top = [int(j) for j in np.argsort(-s2, kind="stable")[:TOP]]
                row["models"][m] = {"top": top, "rank": _r(realistic_rank(s, eid[t], others), 1)}
            rows.append(row)
    return {"rows": rows}


def viz_payload() -> dict:
    """Every number the lab shows, rounded to the page's precision (3 decimals)."""
    toy = {case: {"points": [_r(p) for p in v["points"]], "facts": v["facts"],
                  "rotation": [_r(x) for x in v["rotation"]], "translation": [_r(x) for x in v["translation"]]}
           for case, v in toy_grid().items()}
    toy |= {"theta": list(TOY_THETA), "shift": list(TOY_SHIFT)}
    phases = {}
    for name, f in (("full", rotate_full()), ("split", fit(chosen("RotatE"), pattern_split(0)[0]))):
        k = f.R.shape[1]
        mod = np.sqrt(f.E[:, :k] ** 2 + f.E[:, k:] ** 2).mean(0)
        res = phase_residuals(f.R, FULL_KG.rid)
        phases[name] = {"theta": {r: _r(wrap(f.R[FULL_KG.rid[r]])) for r in RELATIONS}, "weight": _r(mod),
                        "residual": {key: _r(v) for key, v in res.items()},
                        "median": {key: _r(float(np.median(v))) for key, v in res.items()}}
    phases["uniform_median"] = {key: _r(v) for key, v in UNIFORM_MEDIAN.items()}
    d = dihedral_fit()
    kg3 = s3_kg()
    s3 = {"entities": list(kg3.entities), "facts": [[kg3.eid[h], r, kg3.eid[t]] for h, r, t in kg3.triples],
          "dihedral": {"points": [_r(d["E"][tuple(map(int, e))]) for e in kg3.entities], "eps": _r(d["max_fact_error"]),
                       "separation": _r(d["separation"]), "gamma_min": _r(d["gamma_min"])},
          "abelian": {}}
    for m in DISTANCE:
        s3["abelian"][m] = {}
        for k in S3_K:
            b = s3_best(m, k)
            row = {"eps": _r(b["eps"]), "gamma_min": _r(b["gamma_min"]), "ratio": _r(b["ratio"])}
            if k == 1:
                row["points"] = [_r(x) for x in b["fit"].E]
            s3["abelian"][m][f"{k}"] = row
    patterns = {m: {p: {key: _ms([x[key] for x in pattern_results(m)[p]]) for key in ("mrr", "h1", "h3", "h10")}
                    for p in PATTERNS} | {"direction": _ms(pattern_results(m)["direction"]),
                                          "train_mrr": _ms(pattern_results(m)["train_mrr"])}
                for m in MODELS}
    configs = {m: {"gamma": chosen(m).gamma, "lam": chosen(m).lam, "steps": chosen(m).steps} for m in MODELS}
    completion = {}
    for tau in THRESHOLDS:
        cp = completion_problem(tau)
        eid = cp["kg"].eid
        tails, _ = known_answers(WORLD.triples)
        missing = []
        for h, r, t in cp["missing"]:
            ranks = {}
            for m in MODELS:
                cfg, f = completion_fit(m, tau)
                s = scores(m, f.E, f.R, cfg.gamma, eid[h], cp["kg"].rid[r], "tail")
                ranks[m] = _r(realistic_rank(s, eid[t], [eid[x] for x in tails[(h, r)] if x != t]), 1)
            missing.append({"h": eid[h], "r": r, "t": eid[t], "score": _r(extracted().get((h, r, t), 0.0)),
                            "ranks": ranks})
        completion[f"{tau:g}"] = {
            "extracted": [{"h": eid[h], "r": r, "t": eid[t], "score": _r(extracted()[(h, r, t)]),
                           "true": (h, r, t) in WORLD.triples} for h, r, t in cp["extracted"]],
            "missing": missing, "known": cp["known"], "true": cp["true"],
            "metrics": {m: {r: {"mrr": _ms([x["mrr"] for x in v]), "h10": _ms([x["h10"] for x in v])}
                            for r, v in completion_results(m, tau).items()} for m in MODELS}}
    h, r, t = lab_example()
    return {"entities": list(ENTITIES), "types": [WORLD.types[e] for e in ENTITIES], "relations": list(RELATIONS),
            "held": sorted(FULL_KG.eid[e] for e in held_entities()), "toy": toy, "phases": phases, "s3": s3,
            "patterns": patterns, "configs": configs, "queries": _queries_tab(), "completion": completion,
            "example": {"h": FULL_KG.eid[h], "r": r, "t": FULL_KG.eid[t], "score": _r(extracted()[(h, r, t)])},
            "sizes": {"entities": len(ENTITIES), "triples": len(WORLD.triples),
                      "tests": {p: len(v) for p, v in pattern_split(0)[1].items()},
                      "triangles": len(competitor_graph()["triangles"])}}


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
        print(f"  {k:72} {v:.6f}" if isinstance(v, float) else f"  {k:72} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
