"""Aspect-based sentiment analysis — the reference implementation for formalNLP's nineteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The task: given a sentence and one of its gold aspect spans (category c, valence v(c) = ±1), read the
aspect's polarity in {−, 0, +}. On the finance corpus the polarity is a deterministic function of the clause and the
category, so the aspect-level Bayes accuracy is 1, and the question is which hypothesis class can represent it. The
sections follow the page:

1. The aspect-blind ceiling. Every sentence with two aspects has opposite polarities ("revenue rose, but costs climbed"),
   so any classifier of the sentence alone gets at most one aspect of each such pair right (Prop 1). Topic 18's lexicon
   and an aspect-blind bigram LR trained on aspect labels sit below the bound; the LR hits it exactly on contrasts.
2. One aspect-query attention layer is a signed vote (Def 2, Thm 1). With query from the aspect, separable
   content × position scores, and values in logit space, the sign of every pairwise logit difference is the sign of
   Σᵢ e^{sᵢ} uᵢ, where uᵢ is item i's vote. So for a fixed aspect the layer is a linear classifier on the bag of items
   whose weights are positive rescalings of fixed signed votes. (b) When the aspect enters only through the query, a
   vote's sign is the same for every aspect, so valence × direction is fitted only if "rose" and "fell" vote the SAME
   way and the query amplifies one of them against the aspect's context: the direction words cannot vote their
   meaning. Aspect-conditioned values lift that. (c) Swapping one word for another moves the vote by an amount whose
   sign no context can change, and negation is exactly a context that reverses an adjective's effect: no such layer,
   with any values, reads all four of {T was ADJ₊, T was ADJ₋, T was not ADJ₊, T was not ADJ₋}. (d) Bigram items
   restore it.
3. Attention is not identified by decisions (Thm 2). Adding log r_t to every score of item type t and mapping its
   value z ↦ (z + b)/r_t − b multiplies every logit vector by one positive number, so every decision is unchanged while
   the attention can be moved onto any chosen words. The invariant is the vote αᵢ(zᵢ + b) up to that factor.
4. From aspects to tone (Prop 2). Document tone given the polarities of the tone-bearing sentences is Topic 17's
   full-text Bayes ceiling; an aspect reader with sentence error ε loses at most E[k]·ε of it. Aspect-level reading
   reaches the ceiling that no document-level method on this site came near.
5. Finance case study: contrast sets, the aggregation, and readings of the EDGAR excerpts' aspect mentions (no gold).

Cross-checks: the layer against Topic 9's ``attention`` (a one-row query); every gradient against complex-step
derivatives (Topic 5's checker); Thm 1(b)/(c) by linear-programming feasibility over every sign pattern; Thm 2 on the
trained model, on the whole test split.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with scikit-learn==1.9.1 python notebooks/aspect-based-sentiment/aspect_based_sentiment.py
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "lexicon-sentiment-loughran-mcdonald"))
sys.path.insert(0, str(_HERE.parent / "attention-as-kernel-smoothing"))
import attention_as_kernel_smoothing as ak  # noqa: E402 — module objects: their test_* stay out of this harness
import lexicon_sentiment_loughran_mcdonald as ls  # noqa: E402

ltc, fc, nlm = ls.ltc, ls.fc, ak.nlm

# ONE stream per random-instance claim and per initialization, each from default_rng([SEED, k]); k ≥ 3200 keeps this
# module's streams apart from Topics 1–18.
SEED = ls.SEED
_r = ltc._r


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


CATS = tuple(fc.ASPECT_CATEGORIES)
VALENCE = np.array([fc.ASPECT_CATEGORIES[c][0] for c in CATS])
LABELS = ("negative", "neutral", "positive")  # class index = polarity + 1
R = 12  # relative offsets are clipped to [−R, R]
SPLITS = {"train": ltc.FIN_TRAIN, "dev": ltc.FIN_DEV, "test": ltc.FIN_TEST}


# --------------------------------------------------------------------------------------------
# 0. Data: one row per gold aspect (Def 1).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AspectRow:
    """A gold aspect in its sentence: the sentence's tokens, each token's offset from the aspect span (0 inside it),
    the category index, the label (polarity + 1), and where it sits."""
    doc: int
    sent: int
    toks: tuple[str, ...]
    rel: tuple[int, ...]
    cat: int
    y: int
    text: str
    contrast: bool  # the sentence holds two aspects (always of opposite polarity)
    first: bool  # the sentence's first aspect
    span: tuple[int, int]  # first and last token of the aspect term


def _offsets(n: int, lo: int, hi: int) -> tuple[int, ...]:
    return tuple(i - lo if i < lo else 0 if i <= hi else i - hi for i in range(n))


def aspect_rows(docs) -> tuple[AspectRow, ...]:
    rows = []
    for di, d in enumerate(docs):
        for si, (s0, s1) in enumerate(d.sentences):
            asp = [a for a in d.aspects if s0 <= a.start < s1]
            toks = tuple(fc._tokens(d.text[s0:s1]))
            for a in asp:
                lo = len(fc._tokens(d.text[s0:a.start]))
                hi = lo + len(fc._tokens(d.text[a.start:a.end])) - 1
                rows.append(AspectRow(di, si, toks, _offsets(len(toks), lo, hi), CATS.index(a.category),
                                      a.polarity + 1, d.text[s0:s1], len(asp) > 1, a is asp[0], (lo, hi)))
    return tuple(rows)


@lru_cache(maxsize=None)
def docs(split: str) -> tuple:
    return ls.fin_test_docs() if split == "test" else tuple(fc.generate_corpus(*SPLITS[split]))


@lru_cache(maxsize=None)
def rows(split: str) -> tuple[AspectRow, ...]:
    return aspect_rows(docs(split))


@dataclass(frozen=True)
class _Asp:
    category: str


CONSTRUCTIONS = ls.CONSTRUCTIONS + ("neutral",)


def construction(row: AspectRow) -> str:
    """Topic 18's construction of the aspect's own clause (a contrast sentence is split at ", but "); neutral aspects
    are "neutral"."""
    if row.y == 1:
        return "neutral"
    text = row.text.split(", but ")[0 if row.first else 1] if row.contrast else row.text
    return ls.construction(text, [_Asp(CATS[row.cat])])


@lru_cache(maxsize=None)
def constructions(split: str) -> tuple[str, ...]:
    return tuple(construction(r) for r in rows(split))


def labels(rs) -> np.ndarray:
    return np.array([r.y for r in rs])


# --------------------------------------------------------------------------------------------
# 1. The aspect-blind ceiling (Prop 1), and two aspect-blind readers.
# --------------------------------------------------------------------------------------------


def contrast_bound(rs) -> float:
    """Prop 1: 1 − m/(2N), with m the aspects in two-aspect (opposite-polarity) sentences."""
    return 1 - sum(r.contrast for r in rs) / (2 * len(rs))


def lexicon_predict(rs) -> np.ndarray:
    """Topic 18's stand-in lexicon on the whole sentence, negation scope k = 1: sign(pos − neg), a tie read neutral."""
    return np.array([int(np.sign(p - n)) + 1 for p, n, _, _ in (ls.negated_counts(list(r.toks), 1) for r in rs)])


BLIND_LAMBDAS = (1e-2, 1e-3, 1e-4)


@lru_cache(maxsize=None)
def blind_lr() -> dict:
    """An aspect-blind softmax LR on the sentence's unigrams and bigrams (Topic 17's ``lr_fit``), trained on aspect
    labels; λ chosen on dev."""
    X, voc = ltc.featurize([r.toks for r in rows("train")], bigrams=True, min_df=2)
    Xd, _ = ltc.featurize([r.toks for r in rows("dev")], voc, bigrams=True)
    best = None
    for lam in BLIND_LAMBDAS:
        W, b, _ = ltc.lr_fit(X, labels(rows("train")), 3, lam)
        acc = float(((Xd @ W.T + b).argmax(1) == labels(rows("dev"))).mean())
        if best is None or acc > best["dev"]:
            best = {"lam": lam, "dev": acc, "W": W, "b": b}
    best["voc"] = voc
    return best


def blind_predict(rs) -> np.ndarray:
    m = blind_lr()
    X, _ = ltc.featurize([r.toks for r in rs], m["voc"], bigrams=True)
    return np.asarray(X @ m["W"].T + m["b"]).argmax(1)


# --------------------------------------------------------------------------------------------
# 2. One aspect-query attention layer (Def 2) and its vote (Thm 1).
# --------------------------------------------------------------------------------------------
# Items are (type, offset) pairs: every token, and with bigrams also every adjacent pair, placed at its right token.
# Scores are separable: s = S[c, type] + P[c, offset]. Values live in logit space: z = Z[type] (+ Zc[c, type] when the
# values are aspect-conditioned); logits = Σ α z + b (b per category with aspect-conditioned values, else shared).


def items(row: AspectRow, bigrams: bool) -> list[tuple[str, int]]:
    out = list(zip(row.toks, row.rel))
    if bigrams:
        out += [(f"{a} {b}", r) for a, b, r in zip(row.toks, row.toks[1:], row.rel[1:])]
    return out


@lru_cache(maxsize=None)
def vocab(bigrams: bool) -> dict[str, int]:
    """Item types seen at least twice in training; id 0 is <unk>."""
    count: dict[str, int] = {}
    for r in rows("train"):
        for t, _ in items(r, bigrams):
            count[t] = count.get(t, 0) + 1
    voc = {"<unk>": 0}
    for t in sorted(w for w, n in count.items() if n >= 2):
        voc[t] = len(voc)
    return voc


def batch(rs, voc: dict[str, int], bigrams: bool) -> dict[str, np.ndarray]:
    """Ragged arrays: item types, clipped offsets (shifted to 0 … 2R), the row of each item, row starts."""
    its = [items(r, bigrams) for r in rs]
    lens = np.array([len(x) for x in its])
    B = {"tok": np.array([voc.get(t, 0) for x in its for t, _ in x], dtype=int),
         "rel": np.array([min(R, max(-R, o)) + R for x in its for _, o in x], dtype=int),
         "row": np.repeat(np.arange(len(rs)), lens), "starts": np.concatenate([[0], np.cumsum(lens)[:-1]]),
         "cat": np.array([r.cat for r in rs], dtype=int), "y": np.array([r.y for r in rs], dtype=int)}
    B["ci"] = B["cat"][B["row"]]
    return B


def init_layer(V: int, aspect_values: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    th = {"S": np.zeros((len(CATS), V)), "P": np.zeros((len(CATS), 2 * R + 1)),
          "Z": 0.01 * rng.standard_normal((V, 3)), "b": np.zeros((len(CATS) if aspect_values else 1, 3))}
    if aspect_values:
        th["Zc"] = np.zeros((len(CATS), V, 3))
    return th


def _bias(th, cat: np.ndarray) -> np.ndarray:
    return th["b"][cat] if th["b"].shape[0] > 1 else np.broadcast_to(th["b"][0], (len(cat), 3))


def layer_forward(th, B) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scores s, attention α, item values z, logits). Complex parameters pass through (the complex step)."""
    s = th["S"][B["ci"], B["tok"]] + th["P"][B["ci"], B["rel"]]
    e = np.exp(s - np.maximum.reduceat(np.real(s), B["starts"])[B["row"]])
    a = e / np.add.reduceat(e, B["starts"])[B["row"]]
    z = th["Z"][B["tok"]] + (th["Zc"][B["ci"], B["tok"]] if "Zc" in th else 0)
    return s, a, z, np.add.reduceat(a[:, None] * z, B["starts"]) + _bias(th, B["cat"])


def layer_loss(th, B, lam: float):
    lg = layer_forward(th, B)[3]
    lp = nlm.log_softmax(lg)
    return -lp[np.arange(len(B["y"])), B["y"]].mean() + 0.5 * lam * sum(
        (v ** 2).sum() for k, v in th.items() if k != "b")


def layer_loss_grad(th, B, lam: float) -> tuple[float, dict[str, np.ndarray]]:
    _, a, z, lg = layer_forward(th, B)
    N, V, C = len(B["y"]), th["Z"].shape[0], len(CATS)
    p = np.exp(nlm.log_softmax(lg))
    L = -np.log(p[np.arange(N), B["y"]]).mean() + 0.5 * lam * sum((v ** 2).sum() for k, v in th.items() if k != "b")
    g = p.copy()
    g[np.arange(N), B["y"]] -= 1
    g /= N
    gz = a[:, None] * g[B["row"]]
    G = {"Z": np.stack([np.bincount(B["tok"], gz[:, k], V) for k in range(3)], 1) + lam * th["Z"]}
    if "Zc" in th:
        G["Zc"] = (np.stack([np.bincount(B["ci"] * V + B["tok"], gz[:, k], C * V) for k in range(3)], 1)
                   .reshape(C, V, 3) + lam * th["Zc"])
    # ∂L/∂sᵢ = αᵢ (gᵀzᵢ − gᵀ(ℓ − b)): the softmax covariance, row by row
    ds = a * ((z * g[B["row"]]).sum(1) - ((lg - _bias(th, B["cat"])) * g).sum(1)[B["row"]])
    G["S"] = np.bincount(B["ci"] * V + B["tok"], ds, C * V).reshape(C, V) + lam * th["S"]
    G["P"] = np.bincount(B["ci"] * (2 * R + 1) + B["rel"], ds, C * (2 * R + 1)).reshape(C, -1) + lam * th["P"]
    G["b"] = (np.stack([np.bincount(B["cat"], g[:, k], C) for k in range(3)], 1) if th["b"].shape[0] > 1
              else g.sum(0, keepdims=True))
    return float(L), G


@dataclass(frozen=True)
class RungConfig:
    bigrams: bool
    aspect_values: bool
    stream: int
    steps: int = 300
    lr: float = 0.05
    lam: float = 1e-5


RUNGS = {"shared": RungConfig(False, False, 3201), "aspect": RungConfig(False, True, 3202),
         "bigram": RungConfig(True, True, 3203)}
RUNG_NAMES = ("lexicon", "blind", "shared", "aspect", "bigram")


@lru_cache(maxsize=None)
def train_rung(cfg: RungConfig) -> dict[str, np.ndarray]:
    """Full-batch Adam on the training aspects (no tuning: one fixed schedule for every rung)."""
    voc = vocab(cfg.bigrams)
    B = batch(rows("train"), voc, cfg.bigrams)
    th = init_layer(len(voc), cfg.aspect_values, _rng(cfg.stream))
    m = {k: np.zeros_like(v) for k, v in th.items()}
    v2 = {k: np.zeros_like(v) for k, v in th.items()}
    for t in range(1, cfg.steps + 1):
        _, G = layer_loss_grad(th, B, cfg.lam)
        for k in th:
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            th[k] -= cfg.lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
    return th


def rung_forward(name: str, rs, th=None):
    cfg = RUNGS[name]
    B = batch(rs, vocab(cfg.bigrams), cfg.bigrams)
    return B, layer_forward(train_rung(cfg) if th is None else th, B)


def predict(name: str, rs) -> np.ndarray:
    if name == "lexicon":
        return lexicon_predict(rs)
    if name == "blind":
        return blind_predict(rs)
    return rung_forward(name, rs)[1][3].argmax(1)


@lru_cache(maxsize=None)
def split_predictions(name: str) -> tuple[int, ...]:
    return tuple(int(v) for v in predict(name, rows("test")))


def votes(th, B, a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Item i's vote αᵢ(zᵢ + b) (3 columns); the votes of a row sum to its logits."""
    return a[:, None] * (z + _bias(th, B["cat"])[B["row"]])


def accuracy_table() -> dict:
    """Test accuracy per rung: all, contrast aspects, and per construction."""
    y = labels(rows("test"))
    con = np.array(constructions("test"))
    contrast = np.array([r.contrast for r in rows("test")])
    out = {}
    for name in RUNG_NAMES:
        ok = np.array(split_predictions(name)) == y
        out[name] = {"all": float(ok.mean()), "contrast": float(ok[contrast].mean()),
                     "by": {c: float(ok[con == c].mean()) for c in CONSTRUCTIONS}}
    out["n"] = {c: int((con == c).sum()) for c in CONSTRUCTIONS}
    return out


# Thm 1(b)/(c) as linear programs: the vote weights βₐ(t) = e^{sₐ(t)} u(t) of a fixed aspect, and the context's summed
# vote F, are free reals except for the sign constraints the layer imposes; strict inequalities get margin 1 (the
# conditions are invariant under positive scaling).


def _feasible(A_ub: np.ndarray, b_ub: np.ndarray, bounds) -> bool:
    r = optimize.linprog(np.zeros(A_ub.shape[1]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return r.status == 0


def valence_patterns(shared_votes: bool) -> set[tuple[int, int]]:
    """Thm 1(b): the vote-sign patterns (sign u(up), sign u(down)) under which a layer fits the valence quadruple
    {A up: +, A down: −, B up: −, B down: +} (A of valence +1, B of −1; context sums F_A, F_B free). Variables
    [β_A(up), β_A(dn), β_B(up), β_B(dn), F_A, F_B]. With shared votes a pattern fixes the sign of both aspects' β for a
    word; with aspect-conditioned votes only aspect A's pattern is fixed and B's are free."""
    ok = set()
    for su, sd in itertools.product((-1, 0, 1), repeat=2):
        sign = lambda s: (0, 0) if s == 0 else (1, None) if s > 0 else (None, -1)  # noqa: E731 — |β| ≥ 1 when s ≠ 0
        bounds = [sign(su), sign(sd)] + ([sign(su), sign(sd)] if shared_votes else [(None, None)] * 2) + [(None, None)] * 2
        # rows: −(F_A + β_A(up)) ≤ −1;  F_A + β_A(dn) ≤ −1;  F_B + β_B(up) ≤ −1;  −(F_B + β_B(dn)) ≤ −1
        A = np.array([[-1, 0, 0, 0, -1, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, -1, 0, -1]], float)
        if _feasible(A, -np.ones(4), bounds):
            ok.add((su, sd))
    return ok


def negation_feasible(bigram_items: bool, w: tuple[float, float] = (1.0, 1.0)) -> bool:
    """Thm 1(c): one aspect, the adjective slot ∈ {ADJ₊, ADJ₋}, with and without "not" before it. Labels: not ADJ₊ −,
    not ADJ₋ +, ADJ₊ +, ADJ₋ −. Unigram items give the adjective weights w₁a₊, w₁a₋ in the negated context and w₂a₊,
    w₂a₋ without it (w = the position factors, > 0); the two contexts' other votes F₁, F₂ are free. Variables
    [a₊, a₋, F₁, F₂] (+ the bigrams "not ADJ₊", "not ADJ₋", "was ADJ₊", "was ADJ₋" with bigram items)."""
    n = 8 if bigram_items else 4
    A = np.zeros((4, n))
    A[0, [0, 2]] = w[0], 1  # F₁ + w₁a₊ ≤ −1
    A[1, [1, 2]] = -w[0], -1  # F₁ + w₁a₋ ≥ 1
    A[2, [0, 3]] = -w[1], -1  # F₂ + w₂a₊ ≥ 1
    A[3, [1, 3]] = w[1], 1  # F₂ + w₂a₋ ≤ −1
    if bigram_items:
        A[0, 4], A[1, 5], A[2, 6], A[3, 7] = 1, -1, -1, 1
    return _feasible(A, -np.ones(4), [(None, None)] * n)


UP, DOWN = ("rose", "increased", "climbed"), ("fell", "declined", "decreased")
VALENCE_TERMS = ("revenue", "costs")  # single-token terms of categories with valence +1 and −1


def _swap(row: AspectRow, changes: dict[int, str], cat: int, y: int) -> AspectRow:
    toks = tuple(changes.get(i, t) for i, t in enumerate(row.toks))
    return AspectRow(row.doc, row.sent, toks, row.rel, cat, y, " ".join(toks), False, True, row.span)


@lru_cache(maxsize=None)
def valence_quadruples() -> tuple[tuple[AspectRow, ...], ...]:
    """Contrast sets for Thm 1(b): every non-contrast test "direction only" sentence with a one-token term, its term
    set to revenue / costs and its direction word to each (up, down) pair. Gold is v(c) × direction."""
    out = []
    for r, c in zip(rows("test"), constructions("test")):
        if c != "direction only" or r.contrast or r.span[0] != r.span[1]:
            continue
        j = next(i for i in range(r.span[1] + 1, len(r.toks)) if r.toks[i] in UP + DOWN)
        for u, d in itertools.product(UP, DOWN):
            quad = []
            for term, w in itertools.product(VALENCE_TERMS, (u, d)):
                cat = CATS.index("revenue" if term == "revenue" else "costs")
                sign = VALENCE[cat] * (1 if w in UP else -1)
                quad.append(_swap(r, {r.span[0]: term, j: w}, cat, sign + 1))
            out.append(tuple(quad))
    return tuple(out)


ADJ = {s: tuple(sorted(w for w in ls.LEX["positive" if s > 0 else "negative"] if fc.LEXICON_POS[w] == "ADJ"))
       for s in (1, -1)}


def _delete(row: AspectRow, j: int, adj: str, y: int) -> AspectRow:
    toks = row.toks[:j] + (adj,) + row.toks[j + 2:]
    return AspectRow(row.doc, row.sent, toks, _offsets(len(toks), *row.span), row.cat, y, " ".join(toks), False, True,
                     row.span)


@lru_cache(maxsize=None)
def negation_quadruples() -> tuple[tuple[AspectRow, ...], ...]:
    """Contrast sets for Thm 1(c): every non-contrast negated test sentence ("T was not ADJ this period") with the
    adjective set to each positive/negative pair, once as written and once with "not" deleted ("T was ADJ this
    period"). Order: not ADJ₊ (−), not ADJ₋ (+), ADJ₊ (+), ADJ₋ (−)."""
    out = []
    for r, c in zip(rows("test"), constructions("test")):
        if c != "negated adjective" or r.contrast:
            continue
        j = r.toks.index("not")
        for ap, an in itertools.product(ADJ[1], ADJ[-1]):
            out.append((_swap(r, {j + 1: ap}, r.cat, 0), _swap(r, {j + 1: an}, r.cat, 2),
                        _delete(r, j, ap, 2), _delete(r, j, an, 0)))
    return tuple(out)


def quadruple_scores(quads, name: str) -> np.ndarray:
    """1 where the reader gets all four of a quadruple right."""
    flat_ = [r for q in quads for r in q]
    ok = predict(name, flat_) == labels(flat_)
    return ok.reshape(-1, 4).all(1)


def shared_direction_votes() -> dict[str, float]:
    """The shared rung's ± vote u(w) = (z₊ − z₋)(w) of each direction word (aspect-independent, Thm 1(b))."""
    th, voc = train_rung(RUNGS["shared"]), vocab(False)
    return {w: float(th["Z"][voc[w], 2] - th["Z"][voc[w], 0]) for w in UP + DOWN + ("exceeded", "missed")
            if w in voc}


# --------------------------------------------------------------------------------------------
# 3. Attention is not identified by decisions (Thm 2).
# --------------------------------------------------------------------------------------------


def as_general(th) -> dict[str, np.ndarray]:
    """The same layer with a per-category value table Zc and per-category bias (Z = 0)."""
    C = len(CATS)
    Zc = th["Z"][None] + (th["Zc"] if "Zc" in th else 0)
    return {"S": th["S"].copy(), "P": th["P"].copy(), "Z": np.zeros_like(th["Z"]),
            "Zc": np.broadcast_to(Zc, (C, *th["Z"].shape)).copy(), "b": np.broadcast_to(th["b"], (C, 3)).copy()}


def reweighted(th, log_r: np.ndarray) -> dict[str, np.ndarray]:
    """Thm 2's map: S' = S + log r, z' = (z + b)/r − b, for a positive weight r_t per item type."""
    g = as_general(th)
    r = np.exp(log_r)
    g["S"] = g["S"] + log_r[None, :]
    g["Zc"] = (g["Zc"] + g["b"][:, None, :]) / r[None, :, None] - g["b"][:, None, :]
    return g


TERM_WORDS = frozenset(w for t in fc.ASPECT_TERMS for w in t.split())
LOG10_R = tuple(float(v) for v in np.round(np.arange(0, 6.01, 0.5), 2))


def term_log_r(voc: dict[str, int], log10_r: float) -> np.ndarray:
    """log r: log10_r·ln 10 on the aspect-term words (as unigram items), 0 elsewhere."""
    v = np.zeros(len(voc))
    v[[voc[w] for w in TERM_WORDS if w in voc]] = log10_r * np.log(10)
    return v


@lru_cache(maxsize=None)
def identifiability() -> dict:
    """Thm 2 on the trained bigram rung, the whole test split, r = 10⁶ on the aspect-term words."""
    cfg = RUNGS["bigram"]
    voc = vocab(True)
    th = train_rung(cfg)
    B = batch(rows("test"), voc, True)
    _, a, z, lg = layer_forward(th, B)
    th2 = reweighted(th, term_log_r(voc, LOG10_R[-1]))
    _, a2, z2, lg2 = layer_forward(th2, B)
    ratio = lg2 / lg
    lam = ratio.mean(1)
    is_term = np.isin(B["tok"], [voc[w] for w in TERM_WORDS if w in voc])
    mass = np.add.reduceat(np.where(is_term, a, 0), B["starts"])
    mass2 = np.add.reduceat(np.where(is_term, a2, 0), B["starts"])
    v1, v2 = votes(th, B, a, z), votes(th2, B, a2, z2)
    return {"same_argmax": bool((lg.argmax(1) == lg2.argmax(1)).all()),
            "ratio_spread": float(np.abs(ratio - lam[:, None]).max() / lam.min()), "lam_min": float(lam.min()),
            "lam_max": float(lam.max()), "vote_dev": float(np.abs(v2 - lam[B["row"], None] * v1).max()),
            "mass_before": float(mass.mean()), "mass_after_mean": float(mass2.mean()),
            "mass_after_min": float(mass2.min())}


# --------------------------------------------------------------------------------------------
# 4. From aspects to tone (Prop 2).
# --------------------------------------------------------------------------------------------


def tone_sentences(d) -> list[int]:
    """The tone-bearing (aspect) sentences: they open on an aspect, or on "Management pointed to", and name no amount,
    product, or company (which excludes the template-neutral tax, lease, cost-of-revenue, reported, and closing
    sentences). Their first aspect's polarity is the sentence's draw from the tone's mixture."""
    out = []
    for si, (s0, s1) in enumerate(d.sentences):
        asp = [a for a in d.aspects if s0 <= a.start < s1]
        named = any(s0 <= e.start < s1 and e.type in ("MONEY", "PRODUCT", "ORG") for e in d.entities)
        if asp and not named and (asp[0].start == s0 or d.text[s0:s1].startswith("Management pointed to")):
            out.append(si)
    return out


_MIX = {t: dict(zip(*fc._SENTENCE_TONE_MIX[t])) for t in (-1, 0, 1)}


def bayes_tone(counts: tuple[int, int, int]) -> int:
    """The Bayes tone given (+, 0, −) sentence counts, uniform prior (Topic 17's ``bayes_ceiling`` rule)."""
    k = sum(counts)
    return max((-1, 0, 1), key=lambda t: stats.multinomial.pmf(counts, k, [_MIX[t][1], _MIX[t][0], _MIX[t][-1]]))


@lru_cache(maxsize=None)
def aggregation(name: str = "bigram") -> dict:
    """Document tone from the first aspect of each tone-bearing test sentence: gold polarities, and a reader's."""
    first = {(r.doc, r.sent): i for i, r in enumerate(rows("test")) if r.first}
    pred = np.array(split_predictions(name)) - 1
    gold_ok, pred_ok, ks, wrong_docs, errors = [], [], [], 0, 0
    for di, d in enumerate(docs("test")):
        idx = [first[(di, s)] for s in tone_sentences(d)]
        g = [rows("test")[i].y - 1 for i in idx]
        p = [int(pred[i]) for i in idx]
        ks.append(len(idx))
        e = sum(a != b for a, b in zip(g, p))
        errors += e
        wrong_docs += e > 0
        gold_ok.append(bayes_tone((g.count(1), g.count(0), g.count(-1))) == d.tone)
        pred_ok.append(bayes_tone((p.count(1), p.count(0), p.count(-1))) == d.tone)
    n = len(ks)
    return {"gold": float(np.mean(gold_ok)), "pred": float(np.mean(pred_ok)), "ks": ks, "errors": errors,
            "sentences": int(sum(ks)), "wrong_docs": wrong_docs, "eps": errors / sum(ks), "mean_k": float(np.mean(ks)),
            "se": float(np.sqrt(ltc.bayes_ceiling() * (1 - ltc.bayes_ceiling()) / n)), "n": n}


@lru_cache(maxsize=None)
def document_baselines() -> dict:
    """The document-level readers of Topics 17 and 18 on the same test split."""
    return {"net tone (Topic 18)": ls.lexicon_classifier("finance")["acc"],
            "category counts (Topic 18)": ls.category_classifier()["acc"],
            "unigram LR (Topic 17)": ltc.fin_fit(ltc.FinConfig("lr", False))["acc"],
            "Bayes ceiling (Topic 17)": ltc.bayes_ceiling()}


# --------------------------------------------------------------------------------------------
# 5. EDGAR: aspect mentions in real MD&A (no gold).
# --------------------------------------------------------------------------------------------

_TERM_CAT = {t: c for c, (_, ts) in fc.ASPECT_CATEGORIES.items() for t in ts} | {"lease liability": "liabilities"}
_TERM_RE = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in sorted(_TERM_CAT, key=len, reverse=True)), re.I)


@lru_cache(maxsize=None)
def edgar_readings() -> tuple[dict, ...]:
    """Every aspect-term mention in the EDGAR excerpts, read by the bigram rung (and the lexicon) in its sentence."""
    out = []
    for d in fc.load_edgar_excerpts():
        for sent in re.split(r"(?<=[.])\s+", d.text):
            for m in _TERM_RE.finditer(sent):
                lo = len(fc._tokens(sent[:m.start()]))
                toks = tuple(fc._tokens(sent))
                hi = lo + len(fc._tokens(m.group())) - 1
                row = AspectRow(-1, -1, toks, _offsets(len(toks), lo, hi), CATS.index(_TERM_CAT[m.group().lower()]),
                                1, sent, False, True, (lo, hi))
                B, (_, a, z, lg) = rung_forward("bigram", [row])
                p = np.exp(nlm.log_softmax(lg))[0]
                out.append({"company": d.company, "term": m.group(), "category": CATS[row.cat], "tokens": len(toks),
                            "sentence": sent, "pred": LABELS[int(p.argmax())], "p": p.tolist(),
                            "lexicon": LABELS[int(lexicon_predict([row])[0])]})
    return tuple(out)


# --------------------------------------------------------------------------------------------
# Lab payload and summary.
# --------------------------------------------------------------------------------------------


def _positions(rs, name: str, th=None) -> list[dict]:
    """Per-token attention and ± vote (summing a token's unigram and bigram items), prediction, probabilities."""
    B, (_, a, z, lg) = rung_forward(name, rs, th)
    th = train_rung(RUNGS[name]) if th is None else th
    v = votes(th, B, a, z)
    # batch() lists each row's unigrams then its bigrams: rebuild the token position of every item in that order
    pos = np.concatenate([np.concatenate([np.arange(len(r.toks)), np.arange(1, len(r.toks))]) if RUNGS[name].bigrams
                          else np.arange(len(r.toks)) for r in rs])
    p = np.exp(nlm.log_softmax(lg))
    out = []
    for n, r in enumerate(rs):
        sel = B["row"] == n
        att = np.bincount(pos[sel], a[sel], len(r.toks))
        vote = np.bincount(pos[sel], v[sel, 2] - v[sel, 0], len(r.toks))
        out.append({"att": _r(att), "vote": _r(vote), "pred": int(lg[n].argmax()), "p": _r(p[n]),
                    "margin": _r(float(lg[n, 2] - lg[n, 0]))})
    return out


@lru_cache(maxsize=None)
def curated() -> tuple[int, ...]:
    """Test rows the lab opens on, chosen by rule: the first contrast aspect whose two clauses are direction + comment
    and adjective; the first direction-only aspect the shared rung misreads; the first cost-like verb aspect; the first
    negated aspect the aspect rung misreads; the first noun-phrase aspect; the first neutral aspect sentence."""
    rs, con = rows("test"), constructions("test")
    y = labels(rs)
    sh, asp = np.array(split_predictions("shared")), np.array(split_predictions("aspect"))
    picks = []
    pair = next(i for i in range(len(rs) - 1) if rs[i].contrast and rs[i].first
                and {con[i], con[i + 1]} == {"direction + comment", "adjective"})
    picks.append(pair)
    picks.append(next(i for i in range(len(rs)) if con[i] == "direction only" and not rs[i].contrast and sh[i] != y[i]))
    picks.append(next(i for i in range(len(rs)) if con[i] == "verb, cost-like" and not rs[i].contrast))
    picks.append(next(i for i in range(len(rs)) if con[i] == "negated adjective" and not rs[i].contrast
                      and asp[i] != y[i]))
    picks.append(next(i for i in range(len(rs)) if con[i] == "noun phrase"))
    picks.append(next(i for i in range(len(rs)) if con[i] == "neutral" and "prior period" in rs[i].text))
    return tuple(picks)


def _read_payload() -> dict:
    rs = rows("test")
    idx = list(curated())
    idx.insert(1, idx[0] + 1)  # the contrast sentence's second aspect
    sel = [rs[i] for i in idx]
    readers = {name: _positions(sel, name) for name in ("shared", "aspect", "bigram")}
    others = {name: [int(v) for v in predict(name, sel)] for name in ("lexicon", "blind")}
    return {"rows": [{"tokens": list(r.toks), "span": list(r.span), "category": CATS[r.cat],
                      "valence": int(VALENCE[r.cat]), "gold": r.y, "construction": construction(r),
                      "contrast": r.contrast, "lexicon": others["lexicon"][n], "blind": others["blind"][n],
                      **{name: readers[name][n] for name in readers}} for n, r in enumerate(sel)]}


def _constructions_payload() -> dict:
    t = accuracy_table()
    vq, nq = valence_quadruples(), negation_quadruples()
    return {"constructions": list(CONSTRUCTIONS), "n": [t["n"][c] for c in CONSTRUCTIONS],
            "rungs": [{"id": name, "all": _r(t[name]["all"]), "contrast": _r(t[name]["contrast"]),
                       "errors": int((np.array(split_predictions(name)) != labels(rows("test"))).sum()),
                       "by": [_r(t[name]["by"][c]) for c in CONSTRUCTIONS],
                       "valence_quads": _r(float(quadruple_scores(vq, name).mean())),
                       "negation_quads": _r(float(quadruple_scores(nq, name).mean()))} for name in RUNG_NAMES],
            "n_valence_quads": len(vq), "n_negation_quads": len(nq), "bound": _r(contrast_bound(rows("test"))),
            "direction_votes": {w: _r(v) for w, v in shared_direction_votes().items()}}


def _identifiability_payload() -> dict:
    """The contrast sentence's second aspect read by the bigram rung as r grows on the aspect-term words."""
    row = rows("test")[curated()[0] + 1]
    voc, th = vocab(True), train_rung(RUNGS["bigram"])
    B = batch([row], voc, True)
    base = layer_forward(th, B)[3][0]
    steps = []
    for lr10 in LOG10_R:
        th2 = reweighted(th, term_log_r(voc, lr10))
        pos = _positions([row], "bigram", th2)[0]
        lg = layer_forward(th2, B)[3][0]
        v = np.array(pos["vote"])
        mass = sum(a for a, w in zip(pos["att"], row.toks) if w in TERM_WORDS)
        steps.append({"log10_r": lr10, "att": pos["att"], "share": _r(v / np.abs(v).sum()), "pred": pos["pred"],
                      "term_mass": _r(mass),
                      "p": pos["p"], "scale": _r(float(lg[2] - lg[0]) / float(base[2] - base[0]), 4)})
    return {"tokens": list(row.toks), "span": list(row.span), "category": CATS[row.cat], "gold": row.y,
            "term_words": [w in TERM_WORDS for w in row.toks], "steps": steps}


def _tone_payload() -> dict:
    ag = aggregation()
    base = document_baselines()
    return {"bars": [{"label": k, "acc": _r(v)} for k, v in base.items() if "ceiling" not in k]
            + [{"label": "aspect reader (bigram layer)", "acc": _r(ag["pred"])},
               {"label": "gold aspect polarities", "acc": _r(ag["gold"])}],
            "ceiling": _r(base["Bayes ceiling (Topic 17)"]), "se": _r(ag["se"]), "mean_k": _r(ag["mean_k"]),
            "errors": ag["errors"], "sentences": ag["sentences"]}


def _edgar_payload() -> list[dict]:
    return [{"company": e["company"], "term": e["term"], "category": e["category"], "sentence": e["sentence"],
             "pred": e["pred"], "p": _r(e["p"]), "lexicon": e["lexicon"]} for e in edgar_readings()]


def viz_payload() -> dict:
    return {"read": _read_payload(), "constructions": _constructions_payload(),
            "identifiability": _identifiability_payload(), "tone": _tone_payload(), "edgar": _edgar_payload()}


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "aspect-sentiment-lab.json"


def summary() -> dict:
    out = {}
    te = rows("test")
    y = labels(te)
    out["test aspects / per document"] = f"{len(te)} / {len(te) / len(docs('test')):.6f}"
    out["test polarity counts (−, 0, +)"] = " / ".join(str(int((y == k).sum())) for k in range(3))
    out["test aspects in contrast sentences (m)"] = sum(r.contrast for r in te)
    out["Prop 1: aspect-blind bound 1 − m/2N"] = contrast_bound(te)
    out["blind LR: lambda / dev acc"] = f"{blind_lr()['lam']:g} / {blind_lr()['dev']:.6f}"
    t = accuracy_table()
    for name in RUNG_NAMES:
        out[f"{name}: test acc / contrast acc"] = f"{t[name]['all']:.6f} / {t[name]['contrast']:.6f}"
        out[f"{name}: by construction"] = " ".join(f"{c}={t[name]['by'][c]:.6f}" for c in CONSTRUCTIONS)
    out["construction counts"] = " ".join(f"{c}={t['n'][c]}" for c in CONSTRUCTIONS)
    out["Thm 1(b): feasible (u_up, u_dn) sign patterns, shared / aspect votes"] = (
        f"{sorted(valence_patterns(True))} / {sorted(valence_patterns(False))}")
    out["Thm 1(c): negation feasible, unigram / bigram items"] = (
        f"{negation_feasible(False)} / {negation_feasible(True)}")
    out["shared rung ± votes"] = " ".join(f"{w}={v:+.6f}" for w, v in shared_direction_votes().items())
    vq, nq = valence_quadruples(), negation_quadruples()
    out["contrast sets: valence / negation quadruples"] = f"{len(vq)} / {len(nq)}"
    for name in RUNG_NAMES:
        out[f"{name}: all-four-right share, valence / negation"] = (
            f"{quadruple_scores(vq, name).mean():.6f} / {quadruple_scores(nq, name).mean():.6f}")
    idf = identifiability()
    out["Thm 2: same argmax / max logit-ratio spread / vote deviation"] = (
        f"{idf['same_argmax']} / {idf['ratio_spread']:.2e} / {idf['vote_dev']:.2e}")
    out["Thm 2: logit scale range"] = f"{idf['lam_min']:.6f} / {idf['lam_max']:.6f}"
    out["Thm 2: attention on term words before / after (mean, min)"] = (
        f"{idf['mass_before']:.6f} / {idf['mass_after_mean']:.6f}, {idf['mass_after_min']:.6f}")
    ag = aggregation()
    out["Prop 2: tone sentences / mean k / k counts 5..8"] = (
        f"{ag['sentences']} / {ag['mean_k']:.6f} / {[ag['ks'].count(k) for k in range(5, 9)]}")
    out["Prop 2: gold aggregation / bigram-rung aggregation"] = f"{ag['gold']:.6f} / {ag['pred']:.6f}"
    out["Prop 2: sentence errors / eps / docs with an error"] = f"{ag['errors']} / {ag['eps']:.6f} / {ag['wrong_docs']}"
    out["Prop 2: ceiling / SE at n"] = f"{ltc.bayes_ceiling():.6f} / {ag['se']:.6f}"
    for k, v in document_baselines().items():
        out[f"document: {k}"] = v
    for i, e in enumerate(edgar_readings()):
        out[f"EDGAR {i}: {e['company']} '{e['term']}' ({e['tokens']} tokens)"] = f"{e['pred']} ({max(e['p']):.3f}); lexicon {e['lexicon']}"
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: one test per claim on the page.
# --------------------------------------------------------------------------------------------


def test_rows_match_the_gold():
    """Def 1: every gold aspect is a row, its span tokens are an aspect term, and polarity is relative to valence
    (a two-aspect sentence always has opposite polarities)."""
    for split in ("train", "test"):
        assert len(rows(split)) == sum(len(d.aspects) for d in docs(split))
    for r in rows("test")[:3000]:
        term = " ".join(r.toks[r.span[0]:r.span[1] + 1])
        assert term in fc.ASPECT_TERMS, (term, r.text)
        assert r.rel[r.span[0]] == 0 == r.rel[r.span[1]]
    te = rows("test")
    for i, r in enumerate(te):
        if r.contrast and r.first:
            assert te[i + 1].contrast and te[i + 1].y == 2 - r.y and r.y != 1


def test_prop1_aspect_blind_ceiling():
    """Prop 1: any sentence-only reader gets at most one aspect of each contrast pair; both aspect-blind readers sit
    at or below 1 − m/2N, and the blind LR's contrast accuracy is exactly 1/2."""
    te = rows("test")
    bound = contrast_bound(te)
    assert abs(bound - 0.911277) < 1e-6
    t = accuracy_table()
    for name in ("lexicon", "blind"):
        ok = np.array(split_predictions(name)) == labels(te)
        pairs = [ok[i] + ok[i + 1] for i, r in enumerate(te) if r.contrast and r.first]
        assert max(pairs) <= 1, name
        assert t[name]["all"] <= bound + 1e-12
    assert t["blind"]["contrast"] == 0.5
    assert t["blind"]["by"]["verb, cost-like"] < 0.5  # term and verb are not adjacent: no bigram carries the trap


def test_layer_matches_topic9_attention():
    """Def 2 is Topic 9's attention with a one-row query: scores as QKᵀ (τ = 1), logit-space values as V."""
    rs = rows("test")[:20]
    th = train_rung(RUNGS["aspect"])
    B, (s, a, z, lg) = rung_forward("aspect", rs)
    for n in range(len(rs)):
        sel = B["row"] == n
        O, A = ak.attention(np.ones((1, 1)), np.real(s[sel])[:, None], z[sel], tau=1.0)
        assert np.allclose(A[0], a[sel], atol=1e-13)
        assert np.allclose(O[0] + th["b"][rs[n].cat], lg[n], atol=1e-12)


def test_gradients_complex_step():
    """Every rung's analytic gradient matches the complex-step derivative (Topic 5's checker) on a small batch."""
    rs = rows("test")[:8]
    for bigrams, aspect_values in ((False, False), (False, True), (True, True)):
        its = sorted({t for r in rs for t, _ in items(r, bigrams)})
        voc = {"<unk>": 0} | {t: i + 1 for i, t in enumerate(its)}
        B = batch(rs, voc, bigrams)
        rng = _rng(3210 + 2 * bigrams + aspect_values)
        th = {k: rng.standard_normal(v.shape) * 0.5 for k, v in init_layer(len(voc), aspect_values, rng).items()}
        _, G = layer_loss_grad(th, B, 1e-3)
        keys = list(th)
        sizes = [th[k].size for k in keys]
        flat = np.concatenate([th[k].ravel() for k in keys])

        def f(theta):
            parts = np.split(theta, np.cumsum(sizes)[:-1])
            return layer_loss({k: p.reshape(th[k].shape) for k, p in zip(keys, parts)}, B, 1e-3)

        cs = nlm.complex_step_grad(f, flat)
        an = np.concatenate([G[k].ravel() for k in keys])
        assert np.abs(cs - an).max() < 1e-10, np.abs(cs - an).max()


def test_thm1a_vote_identity():
    """Thm 1(a): sign(ℓ_k − ℓ_l) = sign(Σᵢ e^{sᵢ} uᵢ^{kl}), and ℓ_k − ℓ_l = Σᵢ αᵢ uᵢ^{kl}, on random layers."""
    rng = _rng(3220)
    rs = rows("test")[:400]
    for aspect_values in (False, True):
        voc = vocab(False)
        th = {k: rng.standard_normal(v.shape) for k, v in init_layer(len(voc), aspect_values, rng).items()}
        B = batch(rs, voc, False)
        s, a, z, lg = layer_forward(th, B)
        zb = z + _bias(th, B["cat"])[B["row"]]
        for k, l in ((2, 0), (2, 1), (1, 0)):
            u = zb[:, k] - zb[:, l]
            num = np.add.reduceat(np.exp(s) * u, B["starts"])
            assert np.array_equal(np.sign(num), np.sign(lg[:, k] - lg[:, l]))
            assert np.allclose(np.add.reduceat(a * u, B["starts"]), lg[:, k] - lg[:, l], atol=1e-12)


def test_thm1b_direction_words_cannot_vote_their_meaning():
    """Thm 1(b): with votes shared across aspects, the valence quadruple is fitted exactly when u(up) and u(down) are
    nonzero with the SAME sign; with aspect-conditioned votes the meaning pattern (+, −) also works. On the contrast
    sets, every quadruple the shared rung gets fully right uses a same-sign pair of direction words."""
    assert valence_patterns(True) == {(1, 1), (-1, -1)}
    assert (1, -1) in valence_patterns(False)
    u = shared_direction_votes()
    quads = valence_quadruples()
    ok = quadruple_scores(quads, "shared")
    for q, good in zip(quads, ok):
        if good:
            up, dn = q[0].toks[[i for i, t in enumerate(q[0].toks) if t in UP][0]], q[1].toks[
                [i for i, t in enumerate(q[1].toks) if t in DOWN][0]]
            assert u[up] * u[dn] > 0, (up, dn)
    assert quadruple_scores(quads, "aspect").mean() > 0.99


def test_thm1c_negation_is_not_representable():
    """Thm 1(c): swapping one item for another at an offset moves the vote by a fixed-sign amount whatever the context,
    so no layer with a separable kernel and token values reads both "T was ADJ" and "T was not ADJ" for both
    polarities (LP infeasible for random position factors); bigram items make it feasible. Both unigram rungs get NO
    quadruple fully right; the bigram rung gets most."""
    rng = _rng(3225)
    for w in [(1.0, 1.0)] + [tuple(rng.uniform(0.01, 10, 2)) for _ in range(20)]:
        assert not negation_feasible(False, w)
        assert negation_feasible(True, w)
    quads = negation_quadruples()
    assert quadruple_scores(quads, "shared").sum() == 0
    assert quadruple_scores(quads, "aspect").sum() == 0
    assert quadruple_scores(quads, "bigram").mean() > 0.9


def test_the_ladder():
    """Each rung fixes what the theorems say it fixes: aspect values read valence (direction and the valence trap) and
    contrast; only bigram items read negation; the bigram rung is right on all but a handful of test aspects."""
    t = accuracy_table()
    for name in ("shared", "aspect"):
        assert t[name]["by"]["negated adjective"] < 0.6
    assert t["aspect"]["by"]["direction only"] > 0.99 > t["aspect"]["by"]["direction only"] - 0.1 > t["shared"]["by"][
        "direction only"]
    assert min(v for c, v in t["aspect"]["by"].items() if c != "negated adjective") > 0.99
    assert t["aspect"]["contrast"] > t["blind"]["contrast"] + 0.3
    assert t["bigram"]["all"] > 0.999 and t["bigram"]["by"]["negated adjective"] > 0.99


def test_thm2_reweighting_preserves_decisions():
    """Thm 2 on random layers and random r: every logit vector is multiplied by one positive number, so argmaxes are
    equal, and each item's vote is multiplied by the same number."""
    rng = _rng(3230)
    rs = rows("test")[:400]
    voc = vocab(True)
    B = batch(rs, voc, True)
    for aspect_values in (False, True):
        th = {k: rng.standard_normal(v.shape) for k, v in init_layer(len(voc), aspect_values, rng).items()}
        _, a, z, lg = layer_forward(th, B)
        th2 = reweighted(th, rng.normal(0, 2, len(voc)))
        _, a2, z2, lg2 = layer_forward(th2, B)
        lam = np.add.reduceat(np.exp(layer_forward(th, B)[0]), B["starts"]) / np.add.reduceat(
            np.exp(layer_forward(th2, B)[0]), B["starts"])
        assert np.allclose(lg2, lam[:, None] * lg, rtol=1e-9, atol=1e-12)
        assert np.array_equal(lg.argmax(1), lg2.argmax(1))
        assert np.allclose(votes(th2, B, a2, z2), lam[B["row"], None] * votes(th, B, a, z), atol=1e-12)
        assert np.abs(a - a2).max() > 0.5  # the attention itself moved


def test_thm2_on_the_trained_model():
    """Thm 2 on the trained bigram rung: r = 10⁶ on the aspect-term words leaves every test decision unchanged and
    puts almost all attention on those words."""
    idf = identifiability()
    assert idf["same_argmax"]
    assert idf["ratio_spread"] < 1e-6 and idf["vote_dev"] < 1e-10
    assert idf["mass_before"] < 0.5 < 0.99 < idf["mass_after_mean"]


def test_tone_sentences_are_the_generators():
    """Prop 2's input: every document has 5–8 tone-bearing sentences (the generator's k), each opening on an aspect,
    and their polarities follow the tone's mixture (a χ² test per tone)."""
    for split in ("train", "test"):
        for d in docs(split):
            assert 5 <= len(tone_sentences(d)) <= 8, d.doc_id
    first = {(r.doc, r.sent): r for r in rows("train") if r.first}
    for t in (-1, 0, 1):
        pol = [first[(di, s)].y - 1 for di, d in enumerate(docs("train")) if d.tone == t for s in tone_sentences(d)]
        obs = [pol.count(p) for p in (-1, 0, 1)]
        exp = [len(pol) * _MIX[t][p] for p in (-1, 0, 1)]
        assert stats.chisquare(obs, exp).pvalue > 1e-3, (t, obs, exp)


def test_prop2_aggregation_reaches_the_ceiling():
    """Prop 2: gold aspect polarities through the Bayes rule land within 2 SE of Topic 17's ceiling; the reader's
    aggregation differs from it by at most the share of documents holding a reader error (≤ E[k]·ε), far above
    every document-level reader."""
    ag = aggregation()
    assert abs(ag["gold"] - ltc.bayes_ceiling()) < 2 * ag["se"]
    assert abs(ag["pred"] - ag["gold"]) <= ag["wrong_docs"] / ag["n"] <= ag["mean_k"] * ag["eps"] + 1e-12
    base = document_baselines()
    assert ag["pred"] > max(v for k, v in base.items() if "ceiling" not in k) + 0.1


def test_edgar_readings():
    """The case study reads every aspect-term mention in the EDGAR excerpts (no gold: shapes only)."""
    e = edgar_readings()
    assert len(e) >= 4 and any("Microsoft" in x["company"] for x in e)
    for x in e:
        assert abs(sum(x["p"]) - 1) < 1e-12 and x["pred"] in LABELS


def test_viz_payload():
    """The lab narrates: the curated rows are what their rule says, and the identifiability view keeps its decision."""
    p = viz_payload()
    rd = p["read"]["rows"]
    assert rd[0]["contrast"] and rd[1]["contrast"] and rd[0]["gold"] == 2 - rd[1]["gold"]
    assert rd[2]["construction"] == "direction only" and rd[2]["shared"]["pred"] != rd[2]["gold"]
    assert rd[2]["aspect"]["pred"] == rd[2]["gold"]
    assert rd[4]["construction"] == "negated adjective" and rd[4]["aspect"]["pred"] != rd[4]["gold"]
    assert rd[4]["bigram"]["pred"] == rd[4]["gold"]
    steps = p["identifiability"]["steps"]
    assert len({s["pred"] for s in steps}) == 1 and steps[0]["scale"] == 1
    assert len(p["constructions"]["rungs"]) == len(RUNG_NAMES)


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with --write-payload after changing any number here."""
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
