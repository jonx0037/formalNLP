"""Relation extraction and distant supervision — the reference implementation for formalNLP's twenty-third topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Relation mentions and distant supervision. A relation mention is an ordered pair of linked entity mentions in one
   sentence, labeled with a relation or NA (Def 1). Distant supervision labels it with every relation the knowledge
   base holds between the two entities (Def 2). Its errors split exactly into three sources: co-mentions that state
   nothing, facts the sentence entails without stating, and a second KB relation beside the stated one (Prop 1).
2. What distant supervision learns. Under log-loss against the DS labels, the population optimum is the expected DS
   label given the features: the chance the sentence states r plus the chance the KB holds r unstated (Thm 1). On a
   context that states nothing it is the KB-related rate of that context's pairs, at every sample size (Cor 1).
3. At least one witness. Multi-instance learning labels bags (every mention of a pair), not sentences (Def 3). The
   noisy-OR bag likelihood's gradient is the expected instance gradient under each instance's posterior of being the
   witness (Thm 2), so once one instance is confident the others stop learning (Cor 2). Which instance labels the
   at-least-one constraints determine (Prop 2).
   Facts aggregate mentions by max or by noisy-OR (Def 4); noisy-OR can only exceed max and tends to 1 as weak
   mentions pile up, so it suits a model trained under it, MIL, and overcounts for sentence-trained ones (Prop 3).
4. An incomplete KB is positive-unlabeled. If the KB knows each entity pair independently with probability c (Def 5),
   the DS optimum is c times the complete-KB optimum on every relation (Thm 3; Elkan and Noto 2008), argmax keeps a
   stating template only while c > 1/2, and scoring predictions against such a KB reports c times their precision
   (Prop 4).
5. Finance case study: names held out. Topic 22's headline tagger finds the spans, exact string matching links them
   to the KB, an entity-blind classifier reads the relation, and the facts aggregate (Def 6). A triple can be right
   only if both its arguments were found (Prop 5). The extracted KB is what Topic 24 embeds.

Cross-checks: closed-form optima and brute-force hypothesis enumeration on random toys for Thm 1 and Prop 2;
complex-step derivatives for Thm 2; the corpus's own ``distant_supervision_precision`` for Prop 1; Topic 17's
``lr_objective`` for the soft-label objective at one-hot targets.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/relation-extraction-distant-supervision/relation_extraction_distant_supervision.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "named-entity-recognition"))
import named_entity_recognition as ner  # noqa: E402 — module objects: their test_* stay out of this harness

sl, ltc, fc, nlm = ner.sl, ner.ltc, ner.fc, ner.ft.nlm

# ONE stream per random-instance claim and per sampled quantity, each from default_rng([SEED, k]); k ≥ 3600 keeps
# this module's streams apart from Topics 1–22.
SEED = ner.SEED
_r = ltc._r
_sig = ner.ft.tb.ak._sig  # significant figures, for values spanning decades (Cor 2's vanishing coefficients)


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


def _lse(Z: np.ndarray, axis: int) -> np.ndarray:
    return sl._lse(Z, axis)


# --------------------------------------------------------------------------------------------
# 1. Relation mentions and distant supervision (Def 1, Def 2, Prop 1).
# --------------------------------------------------------------------------------------------
#
# A sentence's mentions are its entity spans (token offsets), each linked to a WORLD name when it has one. A relation
# mention is an ordered pair (i, j) of distinct linked mentions in one sentence; its gold label is the relation the
# sentence states from i to j, or NA (Def 1). REPORTED and GUIDED are document events with a MONEY or PERCENT tail,
# never KB facts, so they are outside this topic's candidates. Distant supervision labels (i, j) with KB(h, t), the
# set of relations the KB holds from i's entity to j's (Def 2).

WORLD = fc.WORLD
RELATIONS = ("CEO_OF", "CFO_OF", "IN_SECTOR", "COMPETES_WITH", "SUBSIDIARY_OF", "PARENT_OF", "ACQUIRED", "HQ_IN",
             "LOCATED_IN", "BASED_IN")
LABELS = ("NA", *RELATIONS)
LABEL_ID = {r: k for k, r in enumerate(LABELS)}
K = len(LABELS)
LINKED = ("ORG", "PERSON", "GPE", "SECTOR")  # the types WORLD names


@dataclass(frozen=True)
class Mention:
    a: int  # tokens a, …, b − 1
    b: int
    type: str
    kb: str | None  # the WORLD name it links to


@dataclass(frozen=True)
class RESentence:
    toks: tuple[str, ...]
    mentions: tuple[Mention, ...]
    gold: frozenset  # (i, relation, j) over mention indices: what the sentence states
    doc: int


def re_sentences(docs) -> tuple[RESentence, ...]:
    """Every non-empty sentence (in Topic 21's bio_sentences order) with its gold mentions and stated relations."""
    out = []
    for di, d in enumerate(docs):
        for s, e in d.sentences:
            toks = fc._tokens(d.text[s:e])
            if not toks:
                continue
            idx, ms = {}, []
            for k, en in enumerate(d.entities):
                if s <= en.start and en.end <= e:
                    a = len(fc._tokens(d.text[s:en.start]))
                    idx[k] = len(ms)
                    ms.append(Mention(a, a + len(fc._tokens(d.text[en.start:en.end])), en.type, en.kb_id))
            gold = frozenset((idx[r.head], r.rel, idx[r.tail]) for r in d.relations
                             if r.head in idx and r.rel in LABEL_ID)
            out.append(RESentence(tuple(toks), tuple(ms), gold, di))
    return tuple(out)


TRAIN_ROSTER, HELD_ROSTER = ner.TRAIN_ROSTER, ner.HELD_ROSTER
CORPORA = {"train": ((300, 0), TRAIN_ROSTER), "dev": ((60, 2), TRAIN_ROSTER),
           "seen": ((200, 1), TRAIN_ROSTER), "unseen": ((200, 1), HELD_ROSTER)}
TESTS = ("seen", "unseen")


@lru_cache(maxsize=None)
def docs(name: str) -> tuple:
    """"train": all of generate_corpus(300, 0) on Topic 22's training roster (Topic 22 tagged with the first 30 of its
    documents); "dev": generate_corpus(60, 2) on that roster; "seen"/"unseen": Topic 22's two test runs, (200, 1) on
    the training roster and on the held-out one."""
    (n, seed), roster = CORPORA[name]
    return tuple(fc.generate_corpus(n, seed, roster=roster))


@lru_cache(maxsize=None)
def corpus(name: str) -> tuple[RESentence, ...]:
    return re_sentences(docs(name))


@dataclass(frozen=True)
class Instance:
    sent: int  # index into the corpus's sentences
    i: int  # head mention
    j: int  # tail mention
    h: str  # head entity (WORLD name)
    t: str


def candidates(sents) -> tuple[Instance, ...]:
    """Every ordered pair of distinct linked mentions in one sentence (Def 1's candidates)."""
    out = []
    for si, s in enumerate(sents):
        linked = [(k, m) for k, m in enumerate(s.mentions) if m.kb is not None]
        for (i, a), (j, b) in itertools.permutations(linked, 2):
            out.append(Instance(si, i, j, a.kb, b.kb))
    return tuple(out)


@lru_cache(maxsize=None)
def instances(name: str) -> tuple[Instance, ...]:
    return candidates(corpus(name))


def gold_label(sents, x: Instance) -> str:
    """The relation the sentence states from x.i to x.j, or NA (gold is single-label: test_gold_is_single_label)."""
    rs = [r for i, r, j in sents[x.sent].gold if (i, j) == (x.i, x.j)]
    return rs[0] if rs else "NA"


def kb_between(kb: frozenset, h: str, t: str) -> tuple[str, ...]:
    return tuple(r for r in RELATIONS if (h, r, t) in kb)


def ds_labels(x: Instance, kb: frozenset = WORLD.triples) -> tuple[str, ...]:
    """Def 2: every relation the KB holds from x's head entity to its tail, or NA when it holds none."""
    return kb_between(kb, x.h, x.t) or ("NA",)


# Prop 1's three error sources. A DS label r on (i, j) that the sentence does not state is ENTAILED when r(h, t)
# follows from what the sentence does state by the KB's own rules (BASED_IN = HQ_IN ∘ LOCATED_IN, ACQUIRED ⇒
# PARENT_OF, PARENT_OF and SUBSIDIARY_OF inverse, COMPETES_WITH symmetric); OVERLAP when the sentence states a
# different relation for the same ordered pair; CO-MENTION when it states nothing that yields r. Distractor sentences
# are the generator's co-mentions; the rest of that bucket is pairs that happen to share a fact sentence.
SOURCES = ("co-mention", "entailed", "overlap")


def closure(facts: set) -> set:
    """The facts a sentence's stated facts entail under the KB's rules (a fixpoint)."""
    out = set(facts)
    while True:
        new = set(out)
        for h, r, t in out:
            if r == "COMPETES_WITH":
                new.add((t, r, h))
            if r == "PARENT_OF":
                new.add((t, "SUBSIDIARY_OF", h))
            if r == "SUBSIDIARY_OF":
                new.add((t, "PARENT_OF", h))
            if r == "ACQUIRED":
                new.add((h, "PARENT_OF", t))
            if r == "HQ_IN":
                new |= {(h, "BASED_IN", t2) for h2, r2, t2 in out if r2 == "LOCATED_IN" and h2 == t}
        if new == out:
            return out
        out = new


def stated_facts(s: RESentence) -> set:
    return {(s.mentions[i].kb, r, s.mentions[j].kb) for i, r, j in s.gold}


def ds_error_sources(name: str) -> dict:
    """Prop 1 on a corpus: every DS label (instance, r ≠ NA) is correct (stated) or has exactly one of the three error
    sources; also every stated relation carries its DS label (DS never misses one), and per-relation precision."""
    sents = corpus(name)
    count = Counter()
    per_rel = {r: [0, 0] for r in RELATIONS}
    missed = 0
    ent_cache: dict[int, set] = {}
    for x in instances(name):
        g = gold_label(sents, x)
        ds = ds_labels(x)
        missed += g != "NA" and g not in ds
        for r in ds:
            if r == "NA":
                continue
            per_rel[r][0] += 1
            if r == g:
                count["correct"] += 1
                per_rel[r][1] += 1
                continue
            if x.sent not in ent_cache:
                ent_cache[x.sent] = closure(stated_facts(sents[x.sent]))
            if (x.h, r, x.t) in ent_cache[x.sent]:
                count["entailed"] += 1
            elif g != "NA":
                count["overlap"] += 1
            else:
                count["co-mention"] += 1
    labels = sum(v[0] for v in per_rel.values())
    return {"labels": labels, "correct": count["correct"], **{k: count[k] for k in SOURCES}, "missed": missed,
            "precision": count["correct"] / labels,
            "per_relation": {r: {"labels": n, "precision": c / n if n else None} for r, (n, c) in per_rel.items()}}


# --------------------------------------------------------------------------------------------
# Features: what a relation classifier sees (Mintz et al. 2009's lexical features, entity-blind).
# --------------------------------------------------------------------------------------------
#
# The type pair, which mention comes first, the token string between the two mentions conjoined with that order, its
# words and bigrams, its length (capped at 6), and one word of context on each side. The conjunction matters: which
# argument is the subsidiary depends on the template ("X, a subsidiary of Y" against "Y owns X") as well as on the
# order, an interaction no sum of separate order and string weights can represent (Topic 17's XOR). No feature names
# an argument's own words, so a model read on held-out names sees what it saw in training; the "argument words"
# ablation adds them back (h=…, t=…).

FEATURE_SETS = ("lexical", "lexical-unconjoined", "lexical+args", "lexical+encoder")


def lexical_features(s: RESentence, i: int, j: int, args: bool = False, conjoin: bool = True) -> list[str]:
    a, b = s.mentions[i], s.mentions[j]
    first, second = (a, b) if a.a < b.a else (b, a)
    btw = s.toks[first.b:second.a]
    order = "ht" if a.a < b.a else "th"
    fs = [f"ty={a.type}>{b.type}", f"ord={order}", f"btw={order + ':' if conjoin else ''}" + "_".join(btw),
          f"len={min(len(btw), 6)}"]
    fs += ["bw=" + w for w in btw] + [f"bw2={u}_{v}" for u, v in zip(btw, btw[1:])]
    fs.append("L=" + (s.toks[first.a - 1] if first.a > 0 else "<s>"))
    fs.append("R=" + (s.toks[second.b] if second.b < len(s.toks) else "</s>"))
    if args:
        fs += ["h=" + w for w in s.toks[a.a:a.b]] + ["t=" + w for w in s.toks[b.a:b.b]]
    return fs


def context_class(s: RESentence, i: int, j: int) -> tuple:
    """A template's fingerprint: the type pair, order, between string, and right-hand word (content words occur in
    neither for the generator's co-mention templates). Thm 1's corpus check and Prop 2 group instances by it."""
    f = lexical_features(s, i, j)
    return (f[0], f[1], f[2], f[-1])


def index(rows: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for fs in rows:
        for f in fs:
            vocab.setdefault(f, len(vocab))
    return vocab


def design(rows: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    X = np.zeros((len(rows), len(vocab)))
    for n, fs in enumerate(rows):
        for f in fs:
            if f in vocab:
                X[n, vocab[f]] = 1.0
    return X


# --------------------------------------------------------------------------------------------
# 2. What distant supervision learns (Thm 1, Cor 1).
# --------------------------------------------------------------------------------------------
#
# The DS label distribution of an instance is ℓ(r) = 1[r ∈ L]/|L| with L = KB(h, t) (or {NA}). Its log-loss risk
# R(q) = E[−Σ_r ℓ(r) log q(r | φ)] is minimized over all conditionals by q*(r | φ) = E[ℓ(r) | φ] (Gibbs' inequality,
# pointwise in φ). Gold is a function of the sentence and always a KB fact, so E[ℓ(r) | φ] is the chance the sentence
# states r (with weight 1/|L|) plus the chance the KB holds r unstated. A context that never states a relation gets
# q*(NA | φ) = 1 − ρ(φ), where ρ is the KB-related share of its pairs; no amount of data moves it (Cor 1).


def soft_objective(theta: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float):
    """Mean cross-entropy of softmax(XWᵀ + b) against row-stochastic targets Y, plus (λ/2)‖W‖²; intercepts free.
    At one-hot Y it is Topic 17's lr_objective (test_soft_objective_is_topic17s)."""
    n, V = X.shape
    W, b = theta[:K * V].reshape(K, V), theta[K * V:]
    Z = X @ W.T + b
    lse = _lse(Z, 1)
    L = float(np.real(np.sum(Y * (lse[:, None] - Z)) / n + 0.5 * lam * np.sum(W * W)))
    G = (np.exp(Z - lse[:, None]) - Y) / n
    return L, np.concatenate([(G.T @ X + lam * W).ravel(), G.sum(0)])


def fit_soft(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """L-BFGS on the strictly convex soft-label objective (λ = 1/n: MAP under N(0, I) on the summed objective)."""
    r = optimize.minimize(soft_objective, np.zeros(K * X.shape[1] + K), args=(X, Y, lam), jac=True,
                          method="L-BFGS-B", options={"gtol": 1e-7, "ftol": 0.0, "maxiter": 5000, "maxcor": 30})
    assert r.success, r.message
    return r.x


def probs(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    V = X.shape[1]
    Z = X @ theta[:K * V].reshape(K, V).T + theta[K * V:]
    return np.exp(Z - _lse(Z, 1)[:, None])


def target(labels: tuple[str, ...]) -> np.ndarray:
    y = np.zeros(K)
    for r in labels:
        y[LABEL_ID[r]] += 1.0 / len(labels)
    return y


# The relation classifiers. Every one is a softmax over NA and the ten relations on one feature set, trained on the
# "train" corpus's candidates against one labeling: gold (what each sentence states), DS (Def 2), or the noisy-OR bag
# likelihood (Def 3). keep < 1 trains against Def 5's incomplete KB; kappa < 1 keeps only that share of the related
# co-mention instances (Cor 1's slider: it moves each co-mention template's KB-related rate ρ).
@dataclass(frozen=True)
class Spec:
    feats: str  # one of FEATURE_SETS
    labels: str  # "gold", "ds", or "mil"
    keep: float = 1.0  # Def 5's c
    kappa: float = 1.0  # share of related co-mention training instances kept


@lru_cache(maxsize=None)
def instance_kind(name: str) -> tuple[str, ...]:
    """Each candidate's kind: "stated" (gold ≠ NA), "entailed" (the KB relation follows from what the sentence
    states), "related" (a KB-related pair the sentence says nothing about), or "unrelated" (KB(h, t) is empty)."""
    sents = corpus(name)
    out = []
    for x in instances(name):
        if gold_label(sents, x) != "NA":
            out.append("stated")
        elif not kb_between(WORLD.triples, x.h, x.t):
            out.append("unrelated")
        elif any((x.h, r, x.t) in closure(stated_facts(sents[x.sent])) for r in kb_between(WORLD.triples, x.h, x.t)):
            out.append("entailed")
        else:
            out.append("related")
    return tuple(out)


KINDS = ("stated", "entailed", "related", "unrelated")


@lru_cache(maxsize=1)
def _kappa_uniforms() -> np.ndarray:
    """One uniform per training candidate (stream 3601): a related co-mention is kept at rate κ iff its uniform is
    below κ, so the kept sets are nested in κ."""
    return _rng(3601).random(len(instances("train")))


@lru_cache(maxsize=None)
def train_rows(kappa: float) -> tuple[int, ...]:
    kind, u = instance_kind("train"), _kappa_uniforms()
    return tuple(n for n in range(len(kind)) if kind[n] != "related" or u[n] < kappa)


@lru_cache(maxsize=1)
def _pair_uniforms() -> dict:
    """One uniform per unordered entity pair the world relates (stream 3600): the KB of Def 5 at keep rate c knows a
    pair's facts, all of them, iff its uniform is below c, so the KBs are nested in c."""
    pairs = sorted({tuple(sorted((h, t))) for h, _, t in WORLD.triples})
    return dict(zip(pairs, _rng(3600).random(len(pairs))))


@lru_cache(maxsize=None)
def kept_kb(c: float, stream: int = 3600) -> frozenset:
    """Def 5: the world's facts on the entity pairs the KB knows (stream 3600; stream 3602 draws an independent KB
    of the same rate, for Prop 4's evaluation)."""
    if c >= 1:
        return WORLD.triples
    u = _pair_uniforms() if stream == 3600 else dict(zip(sorted(_pair_uniforms()),
                                                        _rng(stream).random(len(_pair_uniforms()))))
    return frozenset(f for f in WORLD.triples if u[tuple(sorted((f[0], f[2])))] < c)


def feature_rows(name: str, sents, insts, feats: str) -> list[list[str]]:
    return [lexical_features(sents[x.sent], x.i, x.j, feats == "lexical+args", feats != "lexical-unconjoined")
            for x in insts]


@lru_cache(maxsize=None)
def known_words() -> frozenset[str]:
    return frozenset(w for s in corpus("train") for w in s.toks)


def encoder_block(sents, insts, unk_unseen: bool) -> np.ndarray:
    """Baldini Soares et al.'s entity-start representation on Topic 12's frozen encoder: the final states at the
    head's and the tail's first tokens, side by side (2 × 64). With unk_unseen, every word the training corpus never
    shows enters as [UNK], Topic 22's reading of names never seen anywhere."""
    toks = [s.toks for s in sents]
    unk = tuple(np.array([w not in known_words() for w in x]) for x in toks) if unk_unseen else None
    H = ner.states(tuple(toks), unk)
    return np.array([np.concatenate([H[x.sent][sents[x.sent].mentions[x.i].a],
                                     H[x.sent][sents[x.sent].mentions[x.j].a]]) for x in insts])


def matrix(spec: Spec, vocab: dict, sents, insts, unk_unseen: bool) -> np.ndarray:
    X = design(feature_rows("", sents, insts, spec.feats), vocab)
    if spec.feats == "lexical+encoder":
        X = np.hstack([X, encoder_block(sents, insts, unk_unseen)])
    return X


@lru_cache(maxsize=None)
def fit(spec: Spec) -> dict:
    sents, insts = corpus("train"), instances("train")
    rows = train_rows(spec.kappa)
    sub = [insts[n] for n in rows]
    vocab = index(feature_rows("", sents, sub, spec.feats))
    X = matrix(spec, vocab, sents, sub, False)
    kb = kept_kb(spec.keep)
    if spec.labels == "mil":
        return {"theta": mil_fit(X, sub, kb), "vocab": vocab}
    Y = np.array([target((gold_label(sents, x),) if spec.labels == "gold" else ds_labels(x, kb)) for x in sub])
    return {"theta": fit_soft(X, Y, 1.0 / len(sub)), "vocab": vocab}


@lru_cache(maxsize=None)
def predict(spec: Spec, name: str) -> np.ndarray:
    """P(label | candidate) on a corpus's gold-span candidates (names unseen in training read as [UNK] by the
    encoder)."""
    f = fit(spec)
    return probs(f["theta"], matrix(spec, f["vocab"], corpus(name), instances(name), True))


def sentence_scores(name: str, P: np.ndarray) -> dict:
    """Micro P/R/F1 of the argmax labels against gold over the non-NA labels, and the share of each kind of candidate
    labeled with some relation (for "stated", the share labeled with the stated one)."""
    sents, insts, kinds = corpus(name), instances(name), instance_kind(name)
    pred = [LABELS[k] for k in P.argmax(1)]
    tp = fp = fn = 0
    by_kind = {k: [0, 0] for k in KINDS}
    for x, p, kd in zip(insts, pred, kinds):
        g = gold_label(sents, x)
        tp += p == g != "NA"
        fp += p != "NA" and p != g
        fn += g != "NA" and p != g
        by_kind[kd][0] += 1
        by_kind[kd][1] += (p == g) if kd == "stated" else (p != "NA")
    P_, R_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"p": P_, "r": R_, "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "kinds": {k: (v[1] / v[0] if v[0] else None) for k, v in by_kind.items()},
            "n_kinds": {k: v[0] for k, v in by_kind.items()}}


AGGREGATORS = ("max", "noisy-or")


def facts(insts, P: np.ndarray, agg: str) -> dict:
    """Def 4's aggregation of mention probabilities into fact scores: "max" scores (h, r, t) by its most confident
    mention (Mintz et al.'s and Hoffmann et al.'s deterministic OR); "noisy-or" by 1 − Π(1 − P(r | mention)), the
    probability that some mention states r if the mentions were independent (Prop 3: it can only exceed max, and it
    tends to 1 as mentions with P(r) ≥ ε accumulate)."""
    out: dict = defaultdict(float)
    for x, p in zip(insts, P):
        for k, r in enumerate(LABELS[1:], 1):
            f = (x.h, r, x.t)
            out[f] = max(out[f], float(p[k])) if agg == "max" else out[f] + math.log1p(-min(float(p[k]), 1 - 1e-16))
    return dict(out) if agg == "max" else {f: -math.expm1(v) for f, v in out.items()}


def fact_scores(F: dict, insts, truth: frozenset = WORLD.triples, tau: float = 0.5) -> dict:
    """Fact-level P/R/F1 at threshold τ and average precision, against the facts of `truth` whose ordered pair is
    co-mentioned in the corpus's gold candidates `insts` (the facts a reader of this text could at best recover)."""
    pairs = {(x.h, x.t) for x in insts}
    gold = {f for f in truth if (f[0], f[2]) in pairs}
    pred = {f for f, v in F.items() if v > tau}
    tp = len(pred & gold)
    order = sorted(F, key=lambda f: (-F[f], f))
    hits = np.array([f in gold for f in order], float)
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    ap = float((prec * hits).sum() / max(1, len(gold)))
    return {"p": tp / max(1, len(pred)), "r": tp / max(1, len(gold)), "f1": 2 * tp / max(1, len(pred) + len(gold)),
            "ap": ap, "n_pred": len(pred), "n_gold": len(gold)}


# --------------------------------------------------------------------------------------------
# 3. At least one witness (Def 3, Thm 2, Cor 2, Prop 2).
# --------------------------------------------------------------------------------------------
#
# A bag is every training mention of one ordered entity pair. The at-least-one assumption (Riedel, Yao, and McCallum
# 2010) says a bag whose pair the KB relates by r holds at least one mention stating r, and a bag the KB does not
# relate by r holds none. With instance probabilities p_i(r) from the softmax, treated as independent, the bag's
# likelihood for r is the noisy-OR P(∃) = 1 − Π_i (1 − p_i(r)) when r ∈ KB(h, t) and Π_i (1 − p_i(r)) otherwise.
# Thm 2: ∇ log P(∃) = Σ_i E[∇ log P(y_i) | ∃], the posterior w_i = p_i/P(∃) weighting each instance (Fisher's
# identity); Cor 2: instance i's coefficient Π_{j≠i}(1 − p_j)/P(∃) vanishes once another instance is confident.
# Prop 2: over a finite set of context classes, the at-least-one constraints admit h_max = 1[c ∉ N] (N the classes
# of negative bags) and admit nothing else iff every class outside N is the only such class of some positive bag.


def bags(insts) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Instance order sorted by bag, and each bag's (h, t), for np.add.reduceat."""
    order = sorted(range(len(insts)), key=lambda n: (insts[n].h, insts[n].t, n))
    keys = [(insts[n].h, insts[n].t) for n in order]
    return np.array(order), keys


def mil_objective(theta: np.ndarray, X: np.ndarray, starts: np.ndarray, pos: np.ndarray, lam: float):
    """Mean over bags of −Σ_r log(bag likelihood for r), plus (λ/2)‖W‖²; X's rows sorted by bag, `starts` each bag's
    first row, pos[B, r] = r ∈ KB(bag). log(1 − p_i(r)) is computed as a log-sum-exp over the other labels, so the
    objective is complex-step safe and never forms 1 − p."""
    n, V = X.shape
    W, b = theta[:K * V].reshape(K, V), theta[K * V:]
    Z = X @ W.T + b
    lse = _lse(Z, 1)
    P = np.exp(Z - lse[:, None])
    A = np.stack([_lse(np.delete(Z, k, 1), 1) - lse for k in range(1, K)], 1)  # log(1 − p_i(r)), r = 1..K−1
    S = np.add.reduceat(A, starts, axis=0)  # Σ_{i∈B} log(1 − p_i(r))
    nb = len(starts)
    c = np.where(pos, -np.log(-np.expm1(np.where(pos, S, -1.0))), -S)
    L = np.sum(c) / nb + 0.5 * lam * np.sum(W * W)
    dS = np.where(pos, 1.0 / np.expm1(-np.where(pos, S, -1.0)), -1.0) / nb
    GA = np.repeat(dS, np.diff(np.append(starts, n)), axis=0)  # dL/dA, row by row
    M = np.hstack([np.zeros((n, 1)), GA * np.exp(-A)])
    dZ = P * (M.sum(1, keepdims=True) - M - GA.sum(1, keepdims=True))
    return (float(np.real(L)) if np.isrealobj(theta) else L), np.concatenate([(dZ.T @ X + lam * W).ravel(),
                                                                            dZ.sum(0)])


def mil_problem(X: np.ndarray, insts, kb: frozenset):
    order, keys = bags(insts)
    starts = np.flatnonzero([k == 0 or keys[k] != keys[k - 1] for k in range(len(keys))])
    pos = np.array([[r in kb_between(kb, *keys[s]) for r in RELATIONS] for s in starts])
    return X[order], starts, pos


def mil_fit(X: np.ndarray, insts, kb: frozenset) -> np.ndarray:
    """L-BFGS on the noisy-OR bag objective (nonconvex), from zero weights, λ = 1/bags."""
    Xs, starts, pos = mil_problem(X, insts, kb)
    r = optimize.minimize(mil_objective, np.zeros(K * X.shape[1] + K), args=(Xs, starts, pos, 1.0 / len(starts)),
                          jac=True, method="L-BFGS-B",
                          options={"gtol": 1e-7, "ftol": 0.0, "maxiter": 5000, "maxcor": 30})
    assert r.success, r.message
    return r.x


def witness_weights(p: np.ndarray) -> dict:
    """Thm 2 and Cor 2 for one bag and one relation, from its instances' p_i(r): the posterior w_i = p_i/P(∃) that
    instance i states r given that some instance does, and the coefficient Π_{j≠i}(1 − p_j)/P(∃) by which the bag's
    gradient moves instance i's p_i."""
    P_any = -math.expm1(float(np.sum(np.log1p(-p))))
    rest = np.array([math.exp(float(np.sum(np.log1p(-np.delete(p, i))))) for i in range(len(p))])
    return {"P_any": P_any, "w": p / P_any, "push": rest / P_any}


def at_least_one_violations(name: str = "train", kb: frozenset = WORLD.triples) -> dict:
    """Positive (bag, r) pairs with no stating mention (the at-least-one assumption fails), against DS's instance-
    level label errors."""
    sents, insts = corpus(name), instances(name)
    by_bag = defaultdict(list)
    for x in insts:
        by_bag[(x.h, x.t)].append(gold_label(sents, x))
    pos = viol = 0
    for (h, t), gs in by_bag.items():
        for r in kb_between(kb, h, t):
            pos += 1
            viol += r not in gs
    e = ds_error_sources(name)
    return {"bags": len(by_bag), "positive": pos, "violated": viol, "rate": viol / pos,
            "ds_error": 1 - e["precision"]}


def consistent_hypotheses(n_classes: int, bag_list: list[tuple[frozenset, int]]) -> list[tuple[int, ...]]:
    """Every h ∈ {0, 1}^classes with max_{c∈B} h(c) = label(B) for every bag (brute force)."""
    return [h for h in itertools.product((0, 1), repeat=n_classes)
            if all(max(h[c] for c in B) == y for B, y in bag_list)]


def identifiability(n_classes: int, bag_list: list[tuple[frozenset, int]]) -> dict:
    """Prop 2's quantities: N (classes in some negative bag), the always-consistent h_max = 1[c ∉ N], and the classes
    forced to 1 (the only non-N class of some positive bag). Unique iff every class is in N or forced."""
    N = set().union(*[B for B, y in bag_list if y == 0]) if any(y == 0 for _, y in bag_list) else set()
    forced = {next(iter(B - N)) for B, y in bag_list if y == 1 and len(B - N) == 1}
    h_max = tuple(int(c not in N) for c in range(n_classes))
    return {"N": N, "forced": forced, "h_max": h_max, "unique": all(c in N or c in forced for c in range(n_classes))}


@lru_cache(maxsize=1)
def corpus_identifiability() -> dict:
    """Prop 2 on the training corpus, per relation r: bags are entity pairs, classes are context_class templates,
    a bag is positive iff r ∈ KB(h, t). Counts the stating templates the constraints force to 1 and the non-stating
    templates of positive bags that some negative bag pins to 0, plus how many stating mentions each covers."""
    sents, insts = corpus("train"), instances("train")
    cls = [context_class(sents[x.sent], x.i, x.j) for x in insts]
    out = {}
    for r in RELATIONS:
        bag_cls = defaultdict(set)
        for x, c in zip(insts, cls):
            bag_cls[(x.h, x.t)].add(c)
        neg = set().union(*[cs for (h, t), cs in bag_cls.items() if (h, r, t) not in WORLD.triples])
        forced = {next(iter(cs - neg)) for (h, t), cs in bag_cls.items()
                  if (h, r, t) in WORLD.triples and len(cs - neg) == 1}
        stating = {c for x, c in zip(insts, cls) if gold_label(sents, x) == r}
        in_pos = set().union(*[cs for (h, t), cs in bag_cls.items() if (h, r, t) in WORLD.triples])
        out[r] = {"stating": stating, "forced": forced, "neg": neg, "in_pos": in_pos,
                  "nonstating_in_pos": in_pos - stating}
    return out


# --------------------------------------------------------------------------------------------
# 4. An incomplete KB is positive-unlabeled (Def 5, Thm 3, Prop 4).
# --------------------------------------------------------------------------------------------
#
# Def 5: the KB knows each unordered entity pair the world relates, with all of its facts, independently with
# probability c, and independently of the text (selected completely at random, at the pair level). A mention of a pair
# the KB does not know is labeled NA, positive-unlabeled data. Thm 3 (Elkan and Noto 2008's Lemma 1, at the pair
# level): the DS target becomes ℓ_c(r) = s·ℓ(r) for r ≠ NA, with s ~ Bernoulli(c) independent of φ given the pair, so
# q*_c(r | φ) = c·q*(r | φ) on every relation and dividing by c recovers the complete-KB optimum. Their estimator of c
# averages g = 1 − q_c(NA | x) over labeled positives of a validation set, which is exact only when the complete-KB
# target is 1 on those positives; Thm 1 says it is not (co-mentions), so the estimate runs low.

KEEP_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


def corrected(P: np.ndarray, c: float) -> np.ndarray:
    """Thm 3's correction: relation probabilities divided by c, NA the remainder, renormalized if they exceed 1."""
    Q = P.copy()
    Q[:, 1:] = P[:, 1:] / c
    over = Q[:, 1:].sum(1) > 1
    Q[over, 1:] /= Q[over, 1:].sum(1, keepdims=True)
    Q[:, 0] = 1 - Q[:, 1:].sum(1)
    return Q


@lru_cache(maxsize=None)
def c_estimate(c: float) -> float:
    """Elkan and Noto's e1 on the dev corpus: the mean of g(x) = 1 − q_c(NA | x) over the dev candidates the KB of
    keep rate c labels with a relation."""
    P = predict(Spec("lexical", "ds", keep=c), "dev")
    lab = np.array([ds_labels(x, kept_kb(c)) != ("NA",) for x in instances("dev")])
    return float((1 - P[lab, 0]).mean())


@lru_cache(maxsize=None)
def incomplete_kb(c: float) -> dict:
    """The DS classifier trained against the KB of keep rate c, on the seen test: its relation mass relative to the
    complete-KB model's (Thm 3 predicts c), sentence scores raw and corrected (by the true c and by e1), and its max-
    aggregated facts scored against the world, against the training KB itself, and against an independent KB of the
    same rate (stream 3602; Prop 4 predicts c times the world precision)."""
    ins = instances("seen")
    P = predict(Spec("lexical", "ds", keep=c), "seen")
    P1 = predict(Spec("lexical", "ds"), "seen")
    e1 = c_estimate(c)
    F = facts(ins, P, "max")
    return {"kb_facts": len(kept_kb(c)), "mass_ratio": float((1 - P[:, 0]).sum() / (1 - P1[:, 0]).sum()), "e1": e1,
            "raw": sentence_scores("seen", P), "by_c": sentence_scores("seen", corrected(P, c)),
            "by_e1": sentence_scores("seen", corrected(P, e1)),
            "facts_world": fact_scores(F, ins), "facts_kb": fact_scores(F, ins, kept_kb(c)),
            "facts_indep": fact_scores(F, ins, kept_kb(c, 3602))}


# --------------------------------------------------------------------------------------------
# 5. Finance case study: names held out (Def 6, Prop 5).
# --------------------------------------------------------------------------------------------
#
# Def 6, the pipeline: Topic 22's headline tagger (word deletion dropout plus frozen Topic 12 states, unseen words read
# as [UNK]) finds the spans in each test sentence; a span of a linked type is linked to the WORLD name whose tokens it
# spells exactly (canonical names and aliases, type checked); every ordered pair of linked predicted mentions is a
# candidate for the relation classifier, trained on gold spans; facts aggregate as in Def 4. Topic 22 trained on 30
# documents about the 18 training companies; the relation classifiers here train on all 300 of that run. The unseen
# test is about the 6 held-out companies, whose names neither model saw.


@lru_cache(maxsize=1)
def link_table() -> dict:
    """(tokens, type) → WORLD name, for every canonical name and alias."""
    out = {}
    for name, ty in WORLD.types.items():
        out[(tuple(fc._tokens(name)), ty)] = name
    for alias, name in WORLD.aliases.items():
        out[(tuple(fc._tokens(alias)), WORLD.types[name])] = name
    return out


def link(toks: tuple, a: int, b: int, ty: str) -> str | None:
    return link_table().get((toks[a:b], ty)) if ty in LINKED else None


SPAN_SOURCES = ("gold", "predicted")


@lru_cache(maxsize=None)
def predicted_corpus(name: str) -> tuple[RESentence, ...]:
    """The test corpus with Topic 22's headline spans (constrained Viterbi) in place of the gold mentions; gold
    triples are kept where both arguments were predicted exactly (span and type), for Def 1's candidates."""
    gold = corpus(name)
    s22 = ner.split(name)
    pred = ner.predictions(ner.GRID[ner.HEADLINE], name, True)["constrained"]
    assert [g.toks for g in gold] == list(s22.toks)
    out = []
    for g, S in zip(gold, pred):
        ms = tuple(Mention(a, b, ty, link(g.toks, a, b, ty)) for a, b, ty in sorted(S))
        where = {(m.a, m.b, m.type): k for k, m in enumerate(ms)}
        key = [(m.a, m.b, m.type) for m in g.mentions]
        tr = frozenset((where[key[i]], r, where[key[j]]) for i, r, j in g.gold
                       if key[i] in where and key[j] in where)
        out.append(RESentence(g.toks, ms, tr, g.doc))
    return tuple(out)


def triples_of(sents, insts, P: np.ndarray | None) -> set:
    """(sentence, head span, relation, tail span) for every gold triple (P None) or every non-NA prediction."""
    if P is None:
        return {(k, (s.mentions[i].a, s.mentions[i].b, s.mentions[i].type), r,
                 (s.mentions[j].a, s.mentions[j].b, s.mentions[j].type))
                for k, s in enumerate(sents) for i, r, j in s.gold}
    out = set()
    for x, p in zip(insts, P):
        r = LABELS[int(p.argmax())]
        if r != "NA":
            mi, mj = sents[x.sent].mentions[x.i], sents[x.sent].mentions[x.j]
            out.add((x.sent, (mi.a, mi.b, mi.type), r, (mj.a, mj.b, mj.type)))
    return out


@lru_cache(maxsize=None)
def pipeline(labels: str, name: str, spans: str) -> dict:
    """End-to-end strict triple P/R/F1 (both argument spans and types exact, and the relation), Prop 5's bound (the
    share of gold triples whose two arguments were both predicted and linked), and max-aggregated facts against the
    world on the gold candidates' pairs."""
    spec = Spec("lexical", labels)
    sents = corpus(name) if spans == "gold" else predicted_corpus(name)
    ins = candidates(sents)
    f = fit(spec)
    P = probs(f["theta"], matrix(spec, f["vocab"], sents, ins, True))
    G, Pr = triples_of(corpus(name), None, None), triples_of(sents, ins, P)
    tp = len(G & Pr)
    found = triples_of(sents, None, None) if spans == "predicted" else G
    agg = "noisy-or" if labels == "mil" else "max"
    have = {(k, (m.a, m.b, m.type)) for k, s in enumerate(sents) for m in s.mentions}
    miss, tot = Counter(), Counter()
    for k, hs, _, ts in G:
        for a in (hs, ts):
            tot[a[2]] += 1
            miss[a[2]] += (k, a) not in have
    return {"p": tp / max(1, len(Pr)), "r": tp / len(G), "f1": 2 * tp / (len(G) + len(Pr)),
            "bound": len(found) / len(G), "n_gold": len(G), "arg_miss": {ty: miss[ty] / tot[ty] for ty in LINKED},
            "facts": fact_scores(facts(ins, P, agg), instances(name))}


def extracted_kb(name: str = "unseen", labels: str = "mil", spans: str = "predicted") -> dict:
    """The knowledge base this topic hands to Topic 24: fact → aggregated score from the pipeline on one test corpus
    (by default MIL on Topic 22's spans over the held-out companies, aggregated by its own noisy-OR)."""
    spec = Spec("lexical", labels)
    sents = corpus(name) if spans == "gold" else predicted_corpus(name)
    ins = candidates(sents)
    f = fit(spec)
    return facts(ins, probs(f["theta"], matrix(spec, f["vocab"], sents, ins, True)),
                 "noisy-or" if labels == "mil" else "max")


# --------------------------------------------------------------------------------------------
# The lab's example and the grids it browses.
# --------------------------------------------------------------------------------------------

LAB_SPECS = {"gold": Spec("lexical", "gold"), "DS": Spec("lexical", "ds"), "MIL": Spec("lexical", "mil"),
             "DS + argument words": Spec("lexical+args", "ds"), "DS + encoder": Spec("lexical+encoder", "ds"),
             "gold + encoder": Spec("lexical+encoder", "gold"),
             "gold, unconjoined": Spec("lexical-unconjoined", "gold")}
KAPPA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MIN_TEMPLATE = 30  # Cor 1's corpus check: co-mention templates with at least this many training candidates


@lru_cache(maxsize=1)
def example_instance() -> int:
    """The lab's candidate: the first seen-test "P has spoken publicly about X" co-mention of a KB-related pair (P an
    officer of X) that the DS classifier labels with a relation and the MIL classifier labels NA."""
    sents, ins, kinds = corpus("seen"), instances("seen"), instance_kind("seen")
    ds, mil = predict(LAB_SPECS["DS"], "seen"), predict(LAB_SPECS["MIL"], "seen")
    for n, x in enumerate(ins):
        if (kinds[n] == "related" and context_class(sents[x.sent], x.i, x.j)[2] == "btw=ht:has_spoken_publicly_about"
                and ds[n].argmax() != 0 and mil[n].argmax() == 0):
            return n
    raise AssertionError("no example candidate")


@lru_cache(maxsize=1)
def comention_templates() -> list[dict]:
    """Cor 1 on the corpus: each co-mention template (every training candidate of it is a related or unrelated
    co-mention, and some are related) with at least MIN_TEMPLATE training candidates, its training KB-related rate ρ
    and top relation's share π of that mass, and the DS classifier's mean probabilities on its seen-test candidates."""
    sents, ins, kinds = corpus("train"), instances("train"), instance_kind("train")
    by: dict = defaultdict(list)
    bad = set()
    for x, kd in zip(ins, kinds):
        c = context_class(sents[x.sent], x.i, x.j)
        by[c].append(x)
        if kd in ("stated", "entailed"):
            bad.add(c)
    tsents, tins = corpus("seen"), instances("seen")
    P = predict(LAB_SPECS["DS"], "seen")
    fitted: dict = defaultdict(list)
    for x, p in zip(tins, P):
        fitted[context_class(tsents[x.sent], x.i, x.j)].append(p)
    out = []
    for c, xs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        T = np.array([target(ds_labels(x)) for x in xs]).mean(0)
        if len(xs) < MIN_TEMPLATE or c in bad or c not in fitted or T[0] == 1:
            continue
        Pc = np.array(fitted[c]).mean(0)
        out.append({"template": c, "n": len(xs), "rho": float(1 - T[0]), "top": LABELS[int(T[1:].argmax()) + 1],
                    "pi": float(T[1:].max() / (1 - T[0])), "fitted": float(1 - Pc[0]),
                    "fitted_top": float(Pc[1:].max()), "fitted_na": float(Pc[0]), "n_test": len(fitted[c])})
    return out


@lru_cache(maxsize=1)
def kappa_sweep() -> dict:
    """Cor 1's slider: the DS classifier trained with only a share κ of the related co-mention candidates, on the seen
    test: sentence P/R/F1, the share of related and unrelated co-mentions labeled with a relation, and the training
    co-mention base rate ρ(κ) over every template that states nothing."""
    kinds = instance_kind("train")
    sents, ins = corpus("train"), instances("train")
    tsents, tins = corpus("seen"), instances("seen")
    tpl = [t["template"] for t in comention_templates()]
    cls = [context_class(sents[x.sent], x.i, x.j) for x in ins]
    tcls = [context_class(tsents[x.sent], x.i, x.j) for x in tins]
    out: dict = {"kappa": list(KAPPA_GRID), "rho": [], "p": [], "r": [], "f1": [], "related": [], "unrelated": [],
                 "templates": [{"rho": [], "labeled": []} for _ in tpl]}
    for k in KAPPA_GRID:
        rows = train_rows(k)
        rel = sum(kinds[n] == "related" for n in rows)
        out["rho"].append(rel / (rel + sum(kinds[n] == "unrelated" for n in rows)))
        P = predict(Spec("lexical", "ds", kappa=k), "seen")
        s = sentence_scores("seen", P)
        for key in ("p", "r", "f1"):
            out[key].append(s[key])
        out["related"].append(s["kinds"]["related"])
        out["unrelated"].append(s["kinds"]["unrelated"])
        for t, d in zip(tpl, out["templates"]):
            mine = [n for n in rows if cls[n] == t]
            d["rho"].append(sum(kinds[n] == "related" for n in mine) / len(mine))
            lab = [P[n].argmax() != 0 for n in range(len(tins)) if tcls[n] == t]
            d["labeled"].append(sum(lab) / len(lab))
    return out


# ============================================================================================
# Verification harness — every claim the page makes.
# ============================================================================================


def test_gold_is_single_label_and_within_the_kb():
    """Each candidate states at most one relation, and every stated relation is a KB fact (so DS never misses one)."""
    for name in ("train", *TESTS):
        sents = corpus(name)
        for s in sents:
            pairs = Counter((i, j) for i, _, j in s.gold)
            assert max(pairs.values(), default=1) == 1
            assert stated_facts(s) <= WORLD.triples
        assert ds_error_sources(name)["missed"] == 0


def test_alignment_with_topic22():
    """This topic's test sentences are Topic 22's, token for token and span for span, and string linking reproduces
    every gold mention's KB name."""
    for name in TESTS:
        s22 = ner.split(name)
        assert [s.toks for s in corpus(name)] == list(s22.toks)
        assert [frozenset((m.a, m.b, m.type) for m in s.mentions) for s in corpus(name)] == list(s22.spans)
        for s in corpus(name):
            for m in s.mentions:
                assert link(s.toks, m.a, m.b, m.type) == m.kb


def test_prop1_ds_errors_split_into_three_sources():
    """Prop 1: every DS label is stated or has exactly one error source, and the precision is the corpus's own."""
    for name in ("train", "seen"):
        e = ds_error_sources(name)
        assert e["correct"] + sum(e[k] for k in SOURCES) == e["labels"]
        assert abs(e["precision"] - fc.distant_supervision_precision(docs(name))) < 1e-12
        assert all(e[k] > 0 for k in SOURCES) and e["co-mention"] > e["entailed"] > e["overlap"]
    e = ds_error_sources("train")
    assert e["per_relation"]["LOCATED_IN"]["precision"] == 1.0  # stated only in HQ appositives, never co-mentioned
    assert min(v["precision"] for v in e["per_relation"].values()) == e["per_relation"]["ACQUIRED"]["precision"]


def test_soft_objective_is_topic17s():
    """At one-hot targets the soft-label objective and its gradient are Topic 17's lr_objective."""
    rng = _rng(3603)
    X = rng.normal(size=(40, 7))
    y = rng.integers(K, size=40)
    th = rng.normal(size=K * 7 + K)
    a, ga = soft_objective(th, X, np.eye(K)[y], 0.3)
    b, gb = ltc.lr_objective(th, X, y, K, 0.3)
    assert abs(a - b) < 1e-12 and np.abs(ga - gb).max() < 1e-12


def _toy_risk(Z, Pphi, PL, menu):
    """Thm 1's population risk on a finite world: Σ_φ P(φ) Σ_L P(L | φ) (−Σ_r ℓ_L(r) log softmax(Z_φ)_r)."""
    logq = Z - _lse(Z, 1)[:, None]
    ell = np.array([[1.0 / len(L) if r in L else 0.0 for r in range(Z.shape[1])] for L in menu])
    return np.sum(Pphi[:, None] * PL * -(logq @ ell.T))


def test_thm1_ds_optimum_is_the_expected_ds_label():
    """Thm 1 on random finite worlds: the risk's gradient vanishes at log q* with q* = E[ℓ | φ], and every other
    conditional has strictly higher risk. Cor 1: a context whose label sets never include a stated relation has
    q*(NA) = 1 − ρ."""
    rng = _rng(3604)
    Kt = 4
    menu = [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3}), frozenset({1, 2}), frozenset({2, 3})]
    for _ in range(20):
        M = 5
        Pphi = rng.dirichlet(np.ones(M))
        PL = rng.dirichlet(np.ones(len(menu)), size=M)
        ell = np.array([[1.0 / len(L) if r in L else 0.0 for r in range(Kt)] for L in menu])
        qstar = PL @ ell
        Z0 = np.log(qstar)
        g = nlm.complex_step_grad(lambda Z: _toy_risk(Z, Pphi, PL, menu), Z0)
        assert np.abs(g).max() < 1e-12
        base = _toy_risk(Z0, Pphi, PL, menu)
        for _ in range(20):
            D = rng.normal(size=Z0.shape)
            D -= D.mean(1, keepdims=True)  # a shift common to a row changes nothing
            assert _toy_risk(Z0 + 0.1 * D, Pphi, PL, menu) > base
        rho = 1 - PL[:, 0]  # menu[0] is {NA}
        assert np.allclose(qstar[:, 0], 1 - rho)


def test_cor1_ds_learns_each_templates_base_rate():
    """Cor 1 on the corpus: on every frequent co-mention template the fitted DS classifier's P(not NA) on seen-test
    candidates is within 0.05 of the template's training KB-related rate, and argmax picks a relation exactly where
    ρ·π > 1 − ρ (up to the fitted values)."""
    ts = comention_templates()
    assert len(ts) >= 8
    for t in ts:
        assert abs(t["fitted"] - t["rho"]) < 0.05, t
        if abs(t["rho"] * t["pi"] - (1 - t["rho"])) > 0.1:
            assert (t["fitted_top"] > t["fitted_na"]) == (t["rho"] * t["pi"] > 1 - t["rho"]), t


def test_gold_and_ds_at_the_sentence_level():
    """Order-conjoined lexical features read this corpus's templates perfectly under gold; DS keeps the stated
    relations but labels related co-mentions with a relation and entailed pairs with their KB fact."""
    for name in TESTS:
        g = sentence_scores(name, predict(LAB_SPECS["gold"], name))
        d = sentence_scores(name, predict(LAB_SPECS["DS"], name))
        assert g["f1"] == 1.0
        assert sentence_scores(name, predict(LAB_SPECS["gold, unconjoined"], name))["f1"] < 0.98  # the XOR
        assert d["f1"] < 0.9 and d["kinds"]["stated"] >= 0.97 and d["kinds"]["entailed"] == 1.0
        assert d["kinds"]["related"] > 0.25 and d["kinds"]["unrelated"] < 0.06


def test_cor1_slider_moves_comention_false_positives():
    """Cor 1's slider: removing related co-mentions lowers ρ, and the share of related co-mentions labeled with a
    relation falls from its κ = 1 value to 0 at κ = 0, where DS's sentence F1 is highest."""
    k = kappa_sweep()
    assert all(a > b for a, b in zip(k["rho"][1:], k["rho"]))
    assert k["related"][0] == 0.0 and k["related"][-1] > 0.25
    assert k["f1"][0] == max(k["f1"])


def test_thm2_noisy_or_gradient_is_the_expected_instance_gradient():
    """Thm 2 (Fisher's identity): ∂ log P(∃)/∂Z equals Σ_i ∂/∂Z [w_i log p_i + (1 − w_i) log(1 − p_i)] with the
    posteriors w_i = p_i/P(∃) held fixed, by complex step on random bags of softmax instances."""
    rng = _rng(3605)
    for _ in range(20):
        n, Kt, r = int(rng.integers(1, 6)), 4, 1
        Z = rng.normal(size=(n, Kt)) * 2

        def logp(Zc):
            return Zc[:, r] - _lse(Zc, 1)

        def log1mp(Zc):
            return _lse(np.delete(Zc, r, 1), 1) - _lse(Zc, 1)

        f = lambda Zc: np.log(-np.expm1(np.sum(log1mp(Zc))))  # noqa: E731
        p = np.exp(logp(Z))
        w = witness_weights(p)["w"]
        e = lambda Zc: np.sum(w * logp(Zc) + (1 - w) * log1mp(Zc))  # noqa: E731
        assert np.allclose(nlm.complex_step_grad(f, Z), nlm.complex_step_grad(e, Z), atol=1e-13)


def test_cor2_one_witness_silences_the_rest():
    """Cor 2: instance i's coefficient Π_{j≠i}(1 − p_j)/P(∃) equals ∂ log P(∃)/∂p_i, and falls to 0 as another
    instance's p_k → 1, while the confident instance's own coefficient tends to 1."""
    p = np.array([0.3, 0.2, 0.6])
    w = witness_weights(p)
    f = lambda q: np.log(-np.expm1(np.sum(np.log1p(-q))))  # noqa: E731
    assert np.allclose(w["push"], nlm.complex_step_grad(f, p), atol=1e-14)
    pushes = []
    for m in range(1, 8):
        q = np.array([0.3, 0.2, 1 - 10.0 ** -m])
        pushes.append(witness_weights(q)["push"])
    assert all(a[0] > b[0] for a, b in zip(pushes, pushes[1:])) and pushes[-1][0] < 1e-6
    assert abs(pushes[-1][2] - 0.56) < 1e-5  # (1 − 0.3)(1 − 0.2)/P(∃) → 0.56


def test_mil_gradient_matches_complex_step():
    """The noisy-OR bag objective's analytic gradient equals its complex-step gradient on a small random problem."""
    rng = _rng(3606)
    n, V = 18, 4
    X = rng.normal(size=(n, V))
    starts = np.array([0, 3, 4, 9, 12, 13])
    pos = rng.random((len(starts), K - 1)) < 0.3
    th = rng.normal(size=K * V + K) * 0.5
    _, g = mil_objective(th, X, starts, pos, 0.2)
    cs = nlm.complex_step_grad(lambda t: mil_objective(t, X.astype(complex), starts, pos, 0.2)[0], th)
    assert np.abs(g - cs).max() < 1e-12


def test_prop2_identifiability_by_brute_force():
    """Prop 2 on random bag worlds labeled by a true h* under at-least-one: (a) h_max = 1[c ∉ N] is always consistent;
    (b) the consistent set is a singleton iff every class is in N or the only non-N class of some positive bag; (c) a
    unique consistent hypothesis is h*."""
    rng = _rng(3607)
    uniques = 0
    for _ in range(300):
        C = int(rng.integers(2, 7))
        h = tuple(int(v) for v in rng.random(C) < 0.4)
        bl = []
        for _ in range(int(rng.integers(1, 9))):
            B = frozenset(int(c) for c in rng.choice(C, size=int(rng.integers(1, 4))))
            bl.append((B, max(h[c] for c in B)))
        cons = consistent_hypotheses(C, bl)
        idf = identifiability(C, bl)
        assert idf["h_max"] in cons and h in cons
        assert (len(cons) == 1) == idf["unique"]
        if idf["unique"]:
            uniques += 1
            assert cons == [h]
    assert 30 < uniques < 270


def test_mil_trades_sentence_recall_for_precision():
    """At least one holds far better than DS's sentence labels; MIL labels no co-mention with a relation, keeps
    precision ≥ 0.97, and loses sentence recall; its recall is much higher on the stating templates the at-least-one
    constraints force (Prop 2) than on those they leave free."""
    v = at_least_one_violations()
    assert v["rate"] < 0.05 < 0.25 < v["ds_error"]
    for name in TESTS:
        m = sentence_scores(name, predict(LAB_SPECS["MIL"], name))
        d = sentence_scores(name, predict(LAB_SPECS["DS"], name))
        assert m["kinds"]["related"] == 0.0 and m["kinds"]["unrelated"] == 0.0
        assert m["p"] >= 0.97 and m["r"] < d["r"] - 0.15
        f = mil_recall_by_identifiability(name)
        assert f["forced"] > f["free"] + 0.25


@lru_cache(maxsize=None)
def mil_recall_by_identifiability(name: str) -> dict:
    """MIL's recall on a test corpus's stated candidates, split by whether their (template, relation) is forced to 1
    by the training bags' at-least-one constraints (Prop 2) or left free."""
    ci = corpus_identifiability()
    sents, ins = corpus(name), instances(name)
    P = predict(LAB_SPECS["MIL"], name)
    hit, tot = Counter(), Counter()
    for x, p in zip(ins, P):
        g = gold_label(sents, x)
        if g == "NA":
            continue
        c = context_class(sents[x.sent], x.i, x.j)
        side = "forced" if c in ci[g]["forced"] else "free"
        tot[side] += 1
        hit[side] += LABELS[int(p.argmax())] == g
    return {k: hit[k] / tot[k] for k in ("forced", "free")} | {f"n_{k}": tot[k] for k in ("forced", "free")}


def test_prop3_noisy_or_dominates_max():
    """Prop 3: noisy-OR ≥ max for every fact, and 1 − (1 − ε)^m → 1: sentence-trained DS loses fact precision under
    noisy-OR (repeated co-mentions of a pair accumulate), while MIL, trained under noisy-OR, does not."""
    for name in TESTS:
        ins = instances(name)
        for spec in LAB_SPECS.values():
            P = predict(spec, name)
            mx, no = facts(ins, P, "max"), facts(ins, P, "noisy-or")
            assert all(no[f] >= mx[f] - 1e-12 for f in mx)
        P = predict(LAB_SPECS["DS"], name)
        assert fact_scores(facts(ins, P, "noisy-or"), ins)["p"] < fact_scores(facts(ins, P, "max"), ins)["p"] - 0.15
        P = predict(LAB_SPECS["MIL"], name)
        assert fact_scores(facts(ins, P, "noisy-or"), ins)["p"] >= 0.99


def test_mil_leads_at_the_fact_level():
    """MIL trails DS on sentences of the seen test but leads it on facts on both tests (each by its own aggregator)."""
    for name in TESTS:
        ins = instances(name)
        d = fact_scores(facts(ins, predict(LAB_SPECS["DS"], name), "max"), ins)
        m = fact_scores(facts(ins, predict(LAB_SPECS["MIL"], name), "noisy-or"), ins)
        assert m["f1"] > d["f1"] and m["p"] > d["p"]
    assert (sentence_scores("seen", predict(LAB_SPECS["MIL"], "seen"))["f1"]
            < sentence_scores("seen", predict(LAB_SPECS["DS"], "seen"))["f1"])


def test_argument_words_let_ds_memorize_the_kb():
    """With the arguments' own words as features (or Topic 12 states at the entity starts), DS labels most related
    co-mentions of seen pairs with their KB relation, and far fewer once the names are new."""
    for spec in ("DS + argument words", "DS + encoder"):
        s = sentence_scores("seen", predict(LAB_SPECS[spec], "seen"))
        u = sentence_scores("unseen", predict(LAB_SPECS[spec], "unseen"))
        assert s["kinds"]["related"] > 0.75 > 0.35 > u["kinds"]["related"]
    assert sentence_scores("unseen", predict(LAB_SPECS["gold + encoder"], "unseen"))["f1"] < 0.98


def test_thm3_pair_level_scar_scales_the_ds_optimum_by_c():
    """Thm 3 exactly on a finite world: averaging the DS target over every subset of known pairs (weights
    c^k (1 − c)^(n−k)) gives c times the complete-KB target on every relation; and on the corpus the incomplete-KB
    classifier's total relation mass is within 0.05 of c times the complete-KB classifier's."""
    rng = _rng(3608)
    for c in (0.3, 0.7):
        n_pairs, Kt = 4, 3
        labels = [frozenset(int(v) for v in rng.choice(np.arange(1, Kt), size=int(rng.integers(0, 3)), replace=False))
                  for _ in range(n_pairs)]
        weights = rng.dirichlet(np.ones(n_pairs))  # P(pair | φ) for one context
        full = sum(w * (np.eye(Kt)[sorted(L)].mean(0) if L else np.eye(Kt)[0]) for w, L in zip(weights, labels))
        avg = np.zeros(Kt)
        for keep in itertools.product((0, 1), repeat=n_pairs):
            pr = math.prod(c if k else 1 - c for k in keep)
            avg += pr * sum(w * (np.eye(Kt)[sorted(L)].mean(0) if L and k else np.eye(Kt)[0])
                            for w, L, k in zip(weights, labels, keep))
        assert np.allclose(avg[1:], c * full[1:]) and abs(avg.sum() - 1) < 1e-12
    for c in KEEP_GRID[1:]:
        ratio = incomplete_kb(c)["mass_ratio"]
        assert ratio < c and (c < 0.4 or c - ratio < 0.05), (c, ratio)  # the L2 prior shrinks the rarer signal


def test_thm3_correction_and_the_e1_estimate():
    """Thm 3(b): argmax keeps a stating template's relation only while c > 1/2 (q_c(r) = c against NA's 1 − c), so the
    uncorrected classifier's recall holds above 0.9 down to c = 0.7 and collapses below; dividing by the true c restores
    the complete-KB recall within 0.03 down to c = 0.6. A smaller c first helps: co-mention templates fall below the
    argmax threshold before stating ones do, so raw sentence F1 at c = 0.8 beats c = 1. Elkan and Noto's e1 runs below
    c at c = 1, because DS's own target is below 1 on co-mention positives (Thm 1)."""
    r1 = incomplete_kb(1.0)["raw"]["r"]
    for c in KEEP_GRID:
        res = incomplete_kb(c)
        assert (res["raw"]["r"] > 0.9) == (c >= 0.7), c
        if c >= 0.6:
            assert res["by_c"]["r"] > r1 - 0.03, c
        assert res["by_c"]["r"] > res["raw"]["r"] - 1e-12
    assert incomplete_kb(0.8)["raw"]["f1"] > incomplete_kb(1.0)["raw"]["f1"] + 0.05
    assert c_estimate(1.0) < 0.95


def test_prop4_heldout_evaluation_reports_c_times_precision():
    """Prop 4: for a fixed prediction set, the expected precision measured against a KB of keep rate c is c times
    the world precision (exact, by enumeration on a toy); on the corpus, scoring against an independent KB of rate c
    lands within 0.1 of c times the world precision."""
    rng = _rng(3609)
    S = [(int(rng.integers(5)), bool(rng.random() < 0.7)) for _ in range(12)]  # (pair, true?) for each predicted fact
    for c in (0.4, 0.8):
        e = 0.0
        for keep in itertools.product((0, 1), repeat=5):
            pr = math.prod(c if k else 1 - c for k in keep)
            e += pr * sum(t and keep[q] for q, t in S) / len(S)
        assert abs(e - c * sum(t for _, t in S) / len(S)) < 1e-12
    for c in KEEP_GRID[1:-1]:  # at c = 0.3 the classifier predicts only a handful of facts
        res = incomplete_kb(c)
        assert abs(res["facts_indep"]["p"] - c * res["facts_world"]["p"]) < 0.1, c


def test_prop5_triple_recall_is_bounded_by_argument_recall():
    """Prop 5: end-to-end triple recall never exceeds the share of gold triples whose two arguments were found, the
    gold-trained classifier reaches the bound exactly, and the bound is below 1 on both tests."""
    for name in TESTS:
        for labels in ("gold", "ds", "mil"):
            res = pipeline(labels, name, "predicted")
            assert res["r"] <= res["bound"] + 1e-12
        assert pipeline("gold", name, "predicted")["r"] == pipeline("gold", name, "predicted")["bound"] < 1
        assert pipeline("gold", name, "gold")["f1"] == 1.0


def test_held_out_names_cost_the_entity_blind_classifier_nothing():
    """On gold spans the entity-blind classifiers score at least as well on the held-out companies as on the seen
    ones (MIL's sentence F1, everyone's fact precision), so the pipeline's loss on held-out names is Topic 22's."""
    for labels in ("gold", "mil"):
        assert pipeline(labels, "unseen", "gold")["f1"] >= pipeline(labels, "seen", "gold")["f1"]
    assert pipeline("mil", "unseen", "predicted")["facts"]["f1"] > 0.95
    # the seen test's lower bound: its arguments include training-roster names the 30-document tagger rarely saw
    seen, held = pipeline("gold", "seen", "predicted"), pipeline("gold", "unseen", "predicted")
    assert seen["bound"] < held["bound"] and seen["arg_miss"]["ORG"] > 2 * held["arg_miss"]["ORG"]


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


# ============================================================================================
# Summary and the lab's payload.
# ============================================================================================

VIZ_JSON = _HERE.parent.parent / "src" / "components" / "viz" / "data" / "relation-extraction-lab.json"


def _fmt(*v) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def summary() -> dict:
    out = {}
    for name in ("train", *TESTS):
        e = ds_error_sources(name)
        out[f"Prop 1 [{name}]: labels, correct, co-mention, entailed, overlap, precision"] = (
            f"{e['labels']} {e['correct']} {e['co-mention']} {e['entailed']} {e['overlap']} {e['precision']:.6f}")
        out[f"[{name}] candidates by kind"] = " ".join(f"{k}={v}" for k, v in Counter(instance_kind(name)).items())
    v = at_least_one_violations()
    out["at least one: bags, positive (bag, r), violated, rate, DS label error"] = (
        f"{v['bags']} {v['positive']} {v['violated']} {v['rate']:.6f} {v['ds_error']:.6f}")
    for t in TESTS:
        ins = instances(t)
        for lab, spec in LAB_SPECS.items():
            s = sentence_scores(t, predict(spec, t))
            k = s["kinds"]
            out[f"{lab} [{t}]: sentence P R F1 | related unrelated entailed"] = (
                f"{_fmt(s['p'], s['r'], s['f1'])} | {_fmt(k['related'], k['unrelated'], k['entailed'])}")
            for agg in AGGREGATORS:
                f = fact_scores(facts(ins, predict(spec, t), agg), ins)
                out[f"{lab} [{t}] facts ({agg}): P R F1 AP"] = _fmt(f["p"], f["r"], f["f1"], f["ap"])
        m = mil_recall_by_identifiability(t)
        out[f"MIL [{t}] recall on forced / free templates (n)"] = (
            f"{m['forced']:.6f} {m['free']:.6f} ({m['n_forced']} {m['n_free']})")
    for tpl in comention_templates():
        out[f"Cor 1 {tpl['template'][0]} {tpl['template'][2]}: n rho pi fitted"] = (
            f"{tpl['n']} {_fmt(tpl['rho'], tpl['pi'], tpl['fitted'])}")
    k = kappa_sweep()
    out["kappa sweep: rho"] = _fmt(*k["rho"])
    out["kappa sweep: F1"] = _fmt(*k["f1"])
    out["kappa sweep: related labeled"] = _fmt(*k["related"])
    for c in KEEP_GRID:
        res = incomplete_kb(c)
        out[f"c = {c:g}: facts, mass ratio, e1 | raw R F1 | by c R F1 | by e1 F1"] = (
            f"{res['kb_facts']} {_fmt(res['mass_ratio'], res['e1'])} | {_fmt(res['raw']['r'], res['raw']['f1'])} | "
            f"{_fmt(res['by_c']['r'], res['by_c']['f1'])} | {res['by_e1']['f1']:.6f}")
        out[f"c = {c:g}: fact P world, vs KB, vs independent KB"] = _fmt(
            res["facts_world"]["p"], res["facts_kb"]["p"], res["facts_indep"]["p"])
    for t in TESTS:
        for sp in SPAN_SOURCES:
            for lab in ("gold", "ds", "mil"):
                r = pipeline(lab, t, sp)
                if lab == "gold" and sp == "predicted":
                    out[f"pipeline [{t}]: argument miss rate by type"] = " ".join(
                        f"{ty}={v:.6f}" for ty, v in r["arg_miss"].items())
                out[f"pipeline {lab} [{t}, {sp} spans]: P R F1 bound | facts P R F1"] = (
                    f"{_fmt(r['p'], r['r'], r['f1'], r['bound'])} | "
                    f"{_fmt(r['facts']['p'], r['facts']['r'], r['facts']['f1'])}")
    kb = extracted_kb()
    out["extracted KB (MIL, unseen, predicted spans): facts > 0.5, of them true"] = (
        f"{sum(v > 0.5 for v in kb.values())} {sum(v > 0.5 and f in WORLD.triples for f, v in kb.items())}")
    for name in ("train", *TESTS):
        out[f"{name}: sentences, candidates, gold triples"] = (
            f"{len(corpus(name))} {len(instances(name))} {sum(len(s.gold) for s in corpus(name))}")
    return out


def _sentence_json(s: RESentence) -> dict:
    return {"toks": list(s.toks), "mentions": [[m.a, m.b, m.type, m.kb] for m in s.mentions],
            "gold": [[i, r, j] for i, r, j in sorted(s.gold)]}


def viz_payload() -> dict:
    """Every number the lab shows, rounded to the page's precision (3 decimals)."""
    sents, ins, kinds = corpus("seen"), instances("seen"), instance_kind("seen")
    n = example_instance()
    x = ins[n]
    example = {"sentence": _sentence_json(sents[x.sent]), "i": x.i, "j": x.j, "h": x.h, "t": x.t,
               "kb": list(kb_between(WORLD.triples, x.h, x.t)), "kind": kinds[n],
               "probs": {lab: {LABELS[k]: _r(float(v)) for k, v in enumerate(predict(spec, "seen")[n]) if v >= 0.0005}
                         for lab, spec in LAB_SPECS.items() if spec.feats == "lexical"}}
    e = {name: ds_error_sources(name) for name in ("train", "seen")}
    ds_labels_tab = {name: {"labels": v["labels"], "correct": v["correct"], **{k: v[k] for k in SOURCES},
                            "precision": _r(v["precision"]),
                            "per_relation": {r: {"labels": d["labels"], "precision": _r(d["precision"])}
                                             for r, d in v["per_relation"].items()}} for name, v in e.items()}
    templates = [{"types": t["template"][0][3:], "order": t["template"][1][4:], "between": t["template"][2][7:],
                  "right": t["template"][3][2:], "n": t["n"], "rho": _r(t["rho"]), "top": t["top"], "pi": _r(t["pi"]),
                  "fitted": _r(t["fitted"]), "fitted_top": _r(t["fitted_top"]), "fitted_na": _r(t["fitted_na"])}
                 for t in comention_templates()]
    k = kappa_sweep()
    kappa = {key: (v if key == "kappa" else [{kk: _r(vv) for kk, vv in d.items()} for d in v] if key == "templates"
                   else _r(v)) for key, v in k.items()}
    models = {}
    for lab, spec in LAB_SPECS.items():
        models[lab] = {}
        for t in TESTS:
            s = sentence_scores(t, predict(spec, t))
            models[lab][t] = {"p": _r(s["p"]), "r": _r(s["r"]), "f1": _r(s["f1"]),
                              "kinds": {kd: _r(v) for kd, v in s["kinds"].items()},
                              "facts": {agg: {kk: _r(vv) for kk, vv in fact_scores(
                                  facts(instances(t), predict(spec, t), agg), instances(t)).items()
                                  if kk in ("p", "r", "f1", "ap")} for agg in AGGREGATORS}}
    v = at_least_one_violations()
    witness = mil_witness_example()
    mil = {"violations": {"bags": v["bags"], "positive": v["positive"], "violated": v["violated"],
                          "rate": _r(v["rate"]), "ds_error": _r(v["ds_error"])},
           "identifiability": {t: {kk: (_r(vv) if isinstance(vv, float) else vv)
                                   for kk, vv in mil_recall_by_identifiability(t).items()} for t in TESTS},
           "witness": witness}
    incomplete = {f"{c:g}": {"kb_facts": r["kb_facts"], "mass_ratio": _r(r["mass_ratio"]), "e1": _r(r["e1"]),
                             **{key: {kk: _r(r[key][kk]) for kk in ("p", "r", "f1")}
                                for key in ("raw", "by_c", "by_e1")},
                             **{key: _r(r[key]["p"]) for key in ("facts_world", "facts_kb", "facts_indep")},
                             "facts_world_r": _r(r["facts_world"]["r"])}
                  for c in KEEP_GRID for r in [incomplete_kb(c)]}
    pipe = {t: {sp: {lab: {kk: (_r(vv) if kk not in ("facts", "arg_miss") else {k2: _r(v2) for k2, v2 in vv.items()
                                                                            if k2 in ("p", "r", "f1", *LINKED)})
                           for kk, vv in pipeline(lab, t, sp).items() if kk != "n_gold"}
                     for lab in ("gold", "ds", "mil")} for sp in SPAN_SOURCES} for t in TESTS}
    return {"example": example, "ds_labels": ds_labels_tab, "templates": templates, "kappa": kappa,
            "models": models, "mil": mil, "keep_grid": list(KEEP_GRID), "incomplete": incomplete,
            "pipeline": pipe, "held_out": list(HELD_ROSTER),
            "sizes": {name: {"sentences": len(corpus(name)), "candidates": len(instances(name)),
                             "triples": sum(len(s.gold) for s in corpus(name))} for name in ("train", *TESTS)}}


@lru_cache(maxsize=1)
def mil_witness_example() -> dict:
    """The lab's bag for Thm 2 and Cor 2: the seen test's bag of the example's pair under its KB relation (the one
    the DS classifier picked), each mention's text, MIL and DS probabilities of that relation, the posterior w_i, and
    the coefficient Π_{j≠i}(1 − p_j)/P(∃)."""
    sents, ins = corpus("seen"), instances("seen")
    n0 = example_instance()
    x0 = ins[n0]
    r = LABELS[int(predict(LAB_SPECS["DS"], "seen")[n0].argmax())]
    k = LABEL_ID[r]
    rows = [n for n, x in enumerate(ins) if (x.h, x.t) == (x0.h, x0.t)]
    pm, pd = predict(LAB_SPECS["MIL"], "seen")[rows, k], predict(LAB_SPECS["DS"], "seen")[rows, k]
    w = witness_weights(pm)
    return {"relation": r, "h": x0.h, "t": x0.t,
            "mentions": [{"toks": list(sents[ins[n].sent].toks), "i": ins[n].i, "j": ins[n].j,
                          "head": [sents[ins[n].sent].mentions[ins[n].i].a, sents[ins[n].sent].mentions[ins[n].i].b],
                          "tail": [sents[ins[n].sent].mentions[ins[n].j].a, sents[ins[n].sent].mentions[ins[n].j].b],
                          "stated": gold_label(sents, ins[n]) == r, "mil": _r(float(a)), "ds": _r(float(b)),
                          "w": _r(float(c)), "push": _sig(float(d), 3)}
                         for n, a, b, c, d in zip(rows, pm, pd, w["w"], w["push"])],
            "P_any": _r(w["P_any"])}


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
        print(f"  {k:84} {v:.6f}" if isinstance(v, float) else f"  {k:84} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
