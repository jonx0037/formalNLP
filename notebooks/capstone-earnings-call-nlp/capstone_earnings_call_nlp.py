"""Capstone: an earnings-call NLP pipeline — the reference implementation for formalNLP's twenty-ninth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. The pipeline and its oracles. Four stages read an earnings call: entities (Topic 22's headline tagger, linked by
   Topic 23's exact string match), relations (Topic 23's MIL reader, aggregated by noisy-OR), tone (Topic 18's
   dictionary, Topic 17's unigram LR, or Topic 19's aspect reader through its Bayes rule), and aspects (a term matcher
   and Topic 19's bigram reader). Their output is the call graph: the facts the calls state, and per call its subject,
   tone, and aspect polarities. Replacing a stage's prediction by its gold output is that stage's oracle (Def 1).
2. Bounds. End-to-end recall lies between Bonferroni's lower bound and the smallest stage recall, with the product of
   formalRAG's cascade law in between under independence (Prop 1).
3. The oracle game. v(S) scores the pipeline with the stages in S replaced by their oracles (Def 2). A sequential
   ceiling analysis credits each stage with the Harsanyi dividends of the coalitions it completes, so two orders
   disagree exactly when some dividend of two or more stages is nonzero; the Shapley value splits every dividend
   equally and is the only attribution that is efficient, symmetric, null on null players, and additive (Thm 1). A
   stage whose oracle changes nothing gets zero in every order (Cor 1). The error count is additive over stages that
   touch disjoint triples; micro F1 is not, because its denominator couples them (Prop 2).
4. Handoffs. A Bayes downstream can only gain when it receives the posterior instead of the argmax (Thm 2a); a fixed
   downstream rule can lose (Thm 2b). Topic 22's span marginals and samples, fed to Topic 23's noisy-OR, raise fact
   AP and lower F1 at the hard pipeline's threshold; Topic 22's F*/2 threshold, computed from the soft scores alone,
   recovers it (Def 3).
5. Joint inference over the KB. With inverse and symmetric relations merged, the functional constraints give a
   totally unimodular constraint matrix, so the LP relaxation solves the MAP exactly (Def 4, Thm 3); composition rules
   break total unimodularity (Prop 3).
6. How sure are the shares? The Shapley value is linear in v, and v is a ratio of pooled counts, so one cluster
   bootstrap over companies gives every share's sampling distribution (Prop 4). The tone readers, compared on the
   same calls by Topic 28's tests.

Cross-checks: brute-force determinants for Thm 3, exhaustive integer programs for Prop 3, the Shapley formula against
the permutation average and the dividend split, Topic 19's own ``tone_sentences`` and Topic 27's ``doc_facts`` for the
gold side of the tone and aspect stages, Topic 23's ``predicted_corpus`` for the entity stage.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/capstone-earnings-call-nlp/capstone_earnings_call_nlp.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

_HERE = Path(__file__).resolve().parent
for _d in ("relation-extraction-distant-supervision", "aspect-based-sentiment", "topic-models-lda",
           "knowledge-graph-embeddings", "significance-testing-for-nlp"):
    sys.path.insert(0, str(_HERE.parent / _d))
import relation_extraction_distant_supervision as rex  # noqa: E402 — module objects: their test_* stay out of this harness
import aspect_based_sentiment as ab  # noqa: E402
import topic_models_lda as lda  # noqa: E402
import knowledge_graph_embeddings as kge  # noqa: E402
import significance_testing_for_nlp as sig  # noqa: E402

ner, sl, fc = rex.ner, rex.sl, rex.fc
ls, ltc = ab.ls, ab.ltc
WORLD = fc.WORLD

# ONE stream per random-instance claim and per sampled quantity, each from default_rng([SEED, k]); k ≥ 6000 keeps
# this module's streams apart from Topics 1–28.
SEED = rex.SEED
_r = ltc._r


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


# --------------------------------------------------------------------------------------------
# 1. The pipeline and its oracles (Def 1).
# --------------------------------------------------------------------------------------------
#
# The test documents are Topic 22's two test runs, generate_corpus(200, 1) on the training roster ("seen") and on the
# held-out roster ("unseen"); the pipeline reads their earnings calls (66 and 60). The filings of the same runs (MD&A
# and 8-K) never enter the graph: Thm 2's tuned-threshold check uses them as held-out tuning data.

TESTS = ("seen", "unseen")
VIEWS = ("seen", "unseen", "both")
STAGES = ("E", "R", "T", "A")
STAGE_NAMES = {"E": "entities", "R": "relations", "T": "tone", "A": "aspects"}
READERS = ("dictionary", "LR", "aspects")
MIL = rex.Spec("lexical", "mil")
TAU = 0.5  # Topic 23's fact threshold
CALL = "earnings-call"


def docs(name: str) -> tuple:
    return rex.docs(name)


@lru_cache(maxsize=None)
def doc_ids(name: str, part: str) -> frozenset[int]:
    """The documents of a test run: its earnings calls ("calls") or its MD&A and 8-K filings ("filings")."""
    return frozenset(i for i, d in enumerate(docs(name)) if (d.doc_type == CALL) == (part == "calls"))


def calls(name: str) -> tuple[int, ...]:
    return tuple(sorted(doc_ids(name, "calls")))


@lru_cache(maxsize=None)
def sentence_index(name: str) -> tuple[tuple[int, int], ...]:
    """(document, sentence) of every sentence of Topic 23's corpus, in its order (it skips token-free sentences)."""
    out = tuple((di, si) for di, d in enumerate(docs(name)) for si, (s, e) in enumerate(d.sentences)
                if fc._tokens(d.text[s:e]))
    assert len(out) == len(rex.corpus(name))
    return out


def mentions(name: str, gold_E: bool) -> tuple:
    """Stage E: the gold mentions, or Topic 22's headline spans (constrained Viterbi) linked by exact string match."""
    return rex.corpus(name) if gold_E else rex.predicted_corpus(name)


@lru_cache(maxsize=None)
def reader_probs(name: str, gold_E: bool) -> tuple:
    """Stage R's input and output: every candidate pair of linked mentions and Topic 23's MIL label probabilities."""
    sents = mentions(name, gold_E)
    ins = rex.candidates(sents)
    f = rex.fit(MIL)
    return ins, rex.probs(f["theta"], rex.matrix(MIL, f["vocab"], sents, ins, True))


@lru_cache(maxsize=None)
def fact_scores(name: str, gold_E: bool, part: str = "calls") -> dict:
    """Topic 23's noisy-OR fact scores over the candidates in one part of a test run."""
    ins, P = reader_probs(name, gold_E)
    keep = doc_ids(name, part)
    rows = [k for k, x in enumerate(ins) if mentions(name, gold_E)[x.sent].doc in keep]
    return rex.facts([ins[k] for k in rows], P[rows], "noisy-or")


@lru_cache(maxsize=None)
def stated(name: str, gold_E: bool, part: str = "calls") -> frozenset:
    """Stage R's oracle: the facts the text states between the stage-E mentions (all of them with gold mentions; with
    predicted ones, those whose two arguments were found, typed, and linked)."""
    keep = doc_ids(name, part)
    return frozenset((s.mentions[i].kb, r, s.mentions[j].kb) for s in mentions(name, gold_E) if s.doc in keep
                     for i, r, j in s.gold)


def gold_facts(name: str) -> frozenset:
    return stated(name, True)


def facts_out(name: str, gold_E: bool, gold_R: bool, tau: float = TAU) -> frozenset:
    if gold_R:
        return stated(name, gold_E)
    return frozenset(f for f, v in fact_scores(name, gold_E).items() if v > tau)


@lru_cache(maxsize=None)
def pair_owner(name: str) -> dict:
    """For every ordered entity pair a call co-mentions (gold or predicted mentions), the first such call: the call
    a fact about the pair is charged to when the graph is split by company (Prop 4)."""
    out: dict = {}
    for gold_E in (True, False):
        sents = mentions(name, gold_E)
        for x in rex.candidates(sents):
            d = sents[x.sent].doc
            if d in doc_ids(name, "calls"):
                out[(x.h, x.t)] = min(out.get((x.h, x.t), d), d)
    return out


@lru_cache(maxsize=None)
def subjects(name: str, gold_E: bool) -> dict:
    """The call's subject: its first linked ORG mention (the opener's "welcome to the {org} earnings call")."""
    first: dict = {}
    for s in mentions(name, gold_E):
        if s.doc in doc_ids(name, "calls") and s.doc not in first:
            first[s.doc] = s
    out = {}
    for d in calls(name):
        orgs = [m for m in sorted(first[d].mentions, key=lambda m: m.a) if m.type == "ORG" and m.kb]
        out[d] = orgs[0].kb if orgs else None
    return out


# Stage A: a term matcher over the closed aspect vocabulary (Topic 19's EDGAR regex), and Topic 19's bigram reader
# on every match. Its oracle is the gold aspect spans with their polarities.


def _sentence_of(d, start: int) -> int:
    return next(si for si, (s0, s1) in enumerate(d.sentences) if s0 <= start < s1)


@lru_cache(maxsize=None)
def aspects(name: str, gold_A: bool) -> dict:
    """Per call, (sentence, start, end, category, polarity) for every aspect mention, in text order."""
    out = {}
    if gold_A:
        for di in calls(name):
            d = docs(name)[di]
            out[di] = tuple((_sentence_of(d, a.start), a.start, a.end, a.category, a.polarity) for a in d.aspects)
        return out
    found, rows = {}, []
    for di in calls(name):
        d = docs(name)[di]
        found[di] = []
        for si, (s0, s1) in enumerate(d.sentences):
            ms = list(ab._TERM_RE.finditer(d.text, s0, s1))
            toks = tuple(fc._tokens(d.text[s0:s1]))
            for k, m in enumerate(ms):
                lo = len(fc._tokens(d.text[s0:m.start()]))
                hi = lo + len(fc._tokens(m.group())) - 1
                cat = ab._TERM_CAT[m.group().lower()]
                rows.append(ab.AspectRow(di, si, toks, ab._offsets(len(toks), lo, hi), ab.CATS.index(cat), 1,
                                         d.text[s0:s1], len(ms) > 1, k == 0, (lo, hi)))
                found[di].append((si, m.start(), m.end(), cat))
    pol = iter(int(p) - 1 for p in ab.predict("bigram", rows))
    for di in calls(name):
        out[di] = tuple((*a[:4], next(pol)) for a in found[di])
    return out


def aspect_facts(asp) -> frozenset:
    """Per category, the sign of the summed polar polarities (Topic 27's ``doc_facts`` rule); cancelling categories
    state nothing."""
    tot: Counter = Counter()
    for *_, cat, p in asp:
        tot[cat] += p
    return frozenset((c, int(np.sign(v))) for c, v in tot.items() if v)


# Stage T. The dictionary is Topic 18's net tone with its tuned cutpoints, the LR Topic 17's unigram model; both read
# the whole call. The aspect reader is Topic 19's Prop 2: the first aspect of every tone-bearing sentence, through the
# Bayes rule for the generator's sentence mixture. A tone-bearing sentence opens on an aspect (or "Management pointed
# to") and names no amount, product, or company, so this reader depends on stage E's types and stage A's aspects.


@lru_cache(maxsize=None)
def document_tones(name: str, reader: str) -> tuple[int, ...]:
    ds = [docs(name)[i] for i in calls(name)]
    if reader == "dictionary":
        lex = ls.lexicon_classifier("finance")
        C = np.array([ls.category_counts(fc._tokens(d.text)) for d in ds])
        return tuple(int(t) - 1 for t in ls.apply_thresholds(ls.net_tone(C), lex["lo"], lex["hi"]))
    f = ltc.fin_fit(ltc.FinConfig("lr", False))
    X, _ = ltc.featurize([fc._tokens(d.text) for d in ds], f["voc"])
    return tuple(int(t) - 1 for t in ltc.log_posterior(X, f["W"], f["b"]).argmax(1))


@lru_cache(maxsize=None)
def sentence_types(name: str, gold_E: bool) -> dict:
    """(document, sentence) → the entity types stage E finds in it."""
    out: dict = defaultdict(set)
    for (di, si), s in zip(sentence_index(name), mentions(name, gold_E)):
        out[(di, si)] |= {m.type for m in s.mentions}
    return out


def tone_bearing(name: str, di: int, gold_E: bool, asp) -> list[int]:
    """Topic 19's ``tone_sentences`` with stage E's types and stage A's aspects in place of the gold ones."""
    d, types = docs(name)[di], sentence_types(name, gold_E)
    out = []
    for si, (s0, s1) in enumerate(d.sentences):
        first = [a for a in asp if a[0] == si]
        named = bool(types.get((di, si), set()) & {"MONEY", "PRODUCT", "ORG"})
        if first and not named and (first[0][1] == s0 or d.text[s0:s1].startswith("Management pointed to")):
            out.append(si)
    return out


@lru_cache(maxsize=None)
def aspect_tones(name: str, gold_E: bool, gold_A: bool) -> tuple[int, ...]:
    asp = aspects(name, gold_A)
    out = []
    for di in calls(name):
        first = {}
        for a in asp[di]:
            first.setdefault(a[0], a[4])
        p = [first[si] for si in tone_bearing(name, di, gold_E, asp[di])]
        out.append(ab.bayes_tone((p.count(1), p.count(0), p.count(-1))))
    return tuple(out)


def tones(name: str, reader: str, gold_E: bool, gold_A: bool, gold_T: bool) -> tuple[int, ...]:
    if gold_T:
        return tuple(docs(name)[i].tone for i in calls(name))
    if reader == "aspects":
        return aspect_tones(name, gold_E, gold_A)
    return document_tones(name, reader)


def call_id(name: str, di: int) -> str:
    return f"{name}:{docs(name)[di].doc_id}"


def call_graph(name: str, S: frozenset, reader: str) -> frozenset:
    """Def 1's output with the stages in S replaced by their oracles: the stated facts, and per call (id, ABOUT,
    subject), (id, TONE, tone), and (id, ASPECT:category, sign)."""
    gE, gR, gT, gA = ("E" in S), ("R" in S), ("T" in S), ("A" in S)
    out = set(facts_out(name, gE, gR))
    subj, asp = subjects(name, gE), aspects(name, gA)
    for di, t in zip(calls(name), tones(name, reader, gE, gA, gT)):
        c = call_id(name, di)
        if subj[di]:
            out.add((c, "ABOUT", subj[di]))
        out.add((c, "TONE", t))
        out |= {(c, "ASPECT:" + cat, s) for cat, s in aspect_facts(asp[di])}
    return frozenset(out)


def gold_graph(name: str) -> frozenset:
    return call_graph(name, frozenset(STAGES), "LR")


@lru_cache(maxsize=None)
def _call_of(name: str) -> dict:
    return {call_id(name, i): i for i in calls(name)}


def owner(name: str, triple: tuple) -> str:
    """The company a triple is charged to: a call triple's covered company, a fact's first co-mentioning call's."""
    h, r, t = triple
    di = _call_of(name)[h] if r in ("ABOUT", "TONE") or r.startswith("ASPECT:") else pair_owner(name)[(h, t)]
    return f"{name}:{docs(name)[di].company}"


# --------------------------------------------------------------------------------------------
# 2–3. The oracle game (Def 2, Prop 1, Thm 1, Cor 1, Prop 2).
# --------------------------------------------------------------------------------------------

COALITIONS = tuple(frozenset(c) for k in range(len(STAGES) + 1) for c in itertools.combinations(STAGES, k))


@lru_cache(maxsize=None)
def companies(view: str) -> tuple[str, ...]:
    names = TESTS if view == "both" else (view,)
    return tuple(sorted({f"{n}:{docs(n)[i].company}" for n in names for i in calls(n)}))


@lru_cache(maxsize=None)
def game_counts(view: str, reader: str) -> np.ndarray:
    """(coalition, company, [true positives, predicted, gold]) over the call graph: the pooled statistics every v(S),
    and every bootstrap replicate of it, is a ratio of (Prop 4)."""
    comp = {c: k for k, c in enumerate(companies(view))}
    names = TESTS if view == "both" else (view,)
    C = np.zeros((len(COALITIONS), len(comp), 3))
    for n in names:
        G = gold_graph(n)
        for a, S in enumerate(COALITIONS):
            P = call_graph(n, S, reader)
            for x in P & G:
                C[a, comp[owner(n, x)], 0] += 1
            for x in P:
                C[a, comp[owner(n, x)], 1] += 1
            for x in G:
                C[a, comp[owner(n, x)], 2] += 1
    return C


def f1_of(C: np.ndarray) -> np.ndarray:
    """Micro F1 of pooled counts [..., (tp, predicted, gold)]: homogeneous of degree 0, so W @ C resamples it."""
    return 2 * C[..., 0] / (C[..., 1] + C[..., 2])


def errors_of(C: np.ndarray) -> np.ndarray:
    """False positives plus false negatives."""
    return C[..., 1] + C[..., 2] - 2 * C[..., 0]


def game(view: str, reader: str, metric: str = "f1") -> dict:
    """Def 2: v(S) for every coalition. The F1 game counts gains; the error game counts errors removed, v(S) =
    errors(∅) − errors(S), so both are 0 at the empty coalition and rise with oracles."""
    C = game_counts(view, reader).sum(1)
    if metric == "f1":
        vals = f1_of(C) - f1_of(C[0])
    else:
        vals = errors_of(C[0]) - errors_of(C)
    return {S: float(v) for S, v in zip(COALITIONS, vals)}


def shapley(v: dict, players: tuple = STAGES) -> dict:
    """φᵢ = Σ_{S ∌ i} |S|!(n − |S| − 1)!/n! · (v(S ∪ i) − v(S))."""
    n = len(players)
    out = {}
    for i in players:
        rest = [j for j in players if j != i]
        out[i] = sum(math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                     * (v[frozenset(S) | {i}] - v[frozenset(S)])
                     for k in range(n) for S in itertools.combinations(rest, k))
    return out


def dividends(v: dict, players: tuple = STAGES) -> dict:
    """Harsanyi dividends (the Möbius transform): d(T) = Σ_{S ⊆ T} (−1)^{|T| − |S|} v(S), so v(S) = Σ_{T ⊆ S} d(T)."""
    return {T: sum((-1) ** (len(T) - k) * v[frozenset(S)] for k in range(len(T) + 1)
                   for S in itertools.combinations(sorted(T), k)) for T in v}


def sequential(v: dict, order: tuple) -> dict:
    """A ceiling analysis in one order: each stage's gain when its oracle joins those before it."""
    S, out = frozenset(), {}
    for i in order:
        out[i] = v[S | {i}] - v[S]
        S = S | {i}
    return out


def shapley_matrix(players: tuple = STAGES) -> np.ndarray:
    """The linear map v ↦ φ as an (n, 2ⁿ) matrix over COALITIONS (Prop 4)."""
    M = np.zeros((len(players), len(COALITIONS)))
    for a, S in enumerate(COALITIONS):
        e = {T: float(T == S) for T in COALITIONS}
        for k, i in enumerate(players):
            M[k, a] = shapley(e, players)[i]
    return M


def recall_bounds(view: str) -> dict:
    """Prop 1 on the facts: r_E = the share of gold facts with both arguments found and linked (Topic 23's Prop 5),
    r_R = the share the reader recovers from gold mentions, and the end-to-end recall with both stages predicted."""
    names = TESTS if view == "both" else (view,)

    def rec(gE, gR):
        num = sum(len(facts_out(n, gE, gR) & gold_facts(n)) for n in names)
        return num / sum(len(gold_facts(n)) for n in names)

    rE, rR, r = rec(False, True), rec(True, False), rec(False, False)
    both = sum(len(stated(n, False) & facts_out(n, True, False)) for n in names) / sum(len(gold_facts(n)) for n in names)
    return {"r_E": rE, "r_R": rR, "end": r, "product": rE * rR, "bonferroni": max(0.0, rE + rR - 1),
            "min": min(rE, rR), "intersection": both}


# --------------------------------------------------------------------------------------------
# 4. Handoffs (Def 3, Thm 2).
# --------------------------------------------------------------------------------------------
#
# Hard: Topic 22's constrained Viterbi spans. Marginal: every span with marginal q > 10⁻³ (Topic 22's Prop 4) enters
# as a mention, and a candidate pair's relation probability is scaled by q_head · q_tail, the chance both arguments
# are real if spans were independent. Sampled (Finkel, Manning, and Ng 2006): K span sets per sentence from the
# constrained chain by forward filtering, backward sampling; the fact score is the mean over the K pipeline runs of
# each run's noisy-OR. Every variant reads with the same MIL weights and aggregates with the same noisy-OR.

HANDOFFS = ("hard", "marginal", "sampled")
SAMPLES = 64
Q_MIN = 1e-3
HEADLINE = ner.GRID[ner.HEADLINE]
TAU_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))


def ffbs(U: np.ndarray, P: np.ndarray, rng: np.random.Generator, K: int) -> np.ndarray:
    """K exact joint samples per chain (U (n, T, L) unaries, P (L, L) transitions, log scale): z_T from the forward
    message, then z_t ∝ α_t(z) P(z, z_{t+1}) backward. Returns (K, n, T)."""
    alpha, _ = sl.chain_forward(U, P)
    n, T, _ = U.shape
    Z = np.empty((K, n, T), dtype=int)
    u = rng.random((K, T, n))
    for k in range(K):
        w = alpha[:, T - 1]
        for t in range(T - 1, -1, -1):
            if t < T - 1:
                w = alpha[:, t] + P[:, Z[k, :, t + 1]].T
            p = np.exp(w - w.max(1, keepdims=True))
            c = np.cumsum(p / p.sum(1, keepdims=True), 1)
            Z[k, :, t] = np.minimum((c < u[k, t][:, None]).sum(1), c.shape[1] - 1)
    return Z


@lru_cache(maxsize=None)
def span_views(name: str) -> dict:
    """Per sentence (Topic 22's split order = Topic 23's corpus order): the hard span set, the marginals above Q_MIN,
    and SAMPLES sampled span sets."""
    rng = _rng(6001 + TESTS.index(name))
    out = {}
    for idx, U, P in ner.chains(HEADLINE, name, True):
        Uc, Pc = ner.constrain(U, P, HEADLINE.scheme)
        z = sl.viterbi(Uc, Pc)[0]
        q = ner.span_marginals(Uc, Pc, HEADLINE.scheme)
        Z = ffbs(Uc, Pc, rng, SAMPLES)
        for j, i in enumerate(idx):
            out[i] = (frozenset(ner.decode(z[j], HEADLINE.scheme)),
                      {sp: float(v[j]) for sp, v in q.items() if v[j] > Q_MIN},
                      tuple(frozenset(ner.decode(Z[k, j], HEADLINE.scheme)) for k in range(SAMPLES)))
    return out


@lru_cache(maxsize=None)
def _soft_instances(name: str) -> tuple:
    """Every ordered pair of linked spans any handoff proposes, with the MIL probabilities of its relation."""
    toks = rex.corpus(name)
    key, sents, ins = {}, [], []
    for i, (hard, marg, samp) in span_views(name).items():
        pool = sorted(hard | set(marg) | set().union(*samp))
        linked = [sp for sp in pool if rex.link(toks[i].toks, *sp) is not None]
        for h, t in itertools.permutations(linked, 2):
            ms = (rex.Mention(*h, rex.link(toks[i].toks, *h)), rex.Mention(*t, rex.link(toks[i].toks, *t)))
            key[(i, h, t)] = len(sents)
            sents.append(rex.RESentence(toks[i].toks, ms, frozenset(), toks[i].doc))
            ins.append(rex.Instance(len(ins), 0, 1, ms[0].kb, ms[1].kb))
    f = rex.fit(MIL)
    return key, ins, rex.probs(f["theta"], rex.matrix(MIL, f["vocab"], sents, ins, True))


def _noisy_or(pairs, ins, P) -> dict:
    acc: dict = defaultdict(float)
    for n, w in pairs:
        x = ins[n]
        for k, r in enumerate(rex.LABELS[1:], 1):
            acc[(x.h, r, x.t)] += math.log1p(-min(w * float(P[n, k]), 1 - 1e-16))
    return {f: -math.expm1(v) for f, v in acc.items()}


@lru_cache(maxsize=None)
def soft_scores(name: str, how: str, part: str) -> dict:
    key, ins, P = _soft_instances(name)
    views = span_views(name)
    sents = [i for i, s in enumerate(rex.corpus(name)) if s.doc in doc_ids(name, part)]

    def pairs(i, S, w=None):
        return [(key[(i, h, t)], 1.0 if w is None else w[h] * w[t]) for h, t in itertools.permutations(sorted(S), 2)
                if (i, h, t) in key]

    if how == "hard":
        return _noisy_or([p for i in sents for p in pairs(i, views[i][0])], ins, P)
    if how == "marginal":
        return _noisy_or([p for i in sents for p in pairs(i, views[i][1], views[i][1])], ins, P)
    out: dict = defaultdict(float)
    for k in range(SAMPLES):
        for f, v in _noisy_or([p for i in sents for p in pairs(i, views[i][2][k])], ins, P).items():
            out[f] += v / SAMPLES
    return dict(out)


def gold_of(name: str, part: str) -> frozenset:
    return stated(name, True, part)


def prf_at(F: dict, gold: frozenset, tau: float) -> dict:
    pred = {f for f, v in F.items() if v > tau}
    tp = len(pred & gold)
    return {"p": tp / max(1, len(pred)), "r": tp / len(gold), "f1": 2 * tp / (len(pred) + len(gold)), "n": len(pred)}


def average_precision(F: dict, gold: frozenset) -> float:
    order = sorted(F, key=lambda f: (-F[f], f))
    hits = np.array([f in gold for f in order], float)
    return float((np.cumsum(hits) / np.arange(1, len(hits) + 1) * hits).sum() / len(gold))


def f_star(F: dict) -> tuple[float, float]:
    """Topic 22's Thm 3 for independent facts with calibrated scores q: the expected-count F1 of the top k,
    2 Σ_{top k} q / (k + Σ q), is maximized by thresholding at F*/2 (Lipton, Elkan, and Naryanaswamy 2014, Cor 1).
    Returns (F*, F*/2)."""
    q = np.sort(np.array(list(F.values())))[::-1]
    F_k = 2 * np.cumsum(q) / (np.arange(1, len(q) + 1) + q.sum())
    return float(F_k.max()), float(F_k.max() / 2)


@lru_cache(maxsize=None)
def handoff_table(name: str) -> dict:
    """Per handoff on the calls: P/R/F1 at τ = 0.5 and on TAU_GRID, AP, the F*/2 threshold and its F1, and the τ that
    maximizes F1 on the same run's filings, applied to the calls."""
    gold = gold_of(name, "calls")
    out = {}
    for how in HANDOFFS:
        F, Ff = soft_scores(name, how, "calls"), soft_scores(name, how, "filings")
        Fs, half = f_star(F)
        gf = gold_of(name, "filings")
        tuned = max(TAU_GRID, key=lambda t: (prf_at(Ff, gf, t)["f1"], -abs(t - TAU)))
        out[how] = {"at": prf_at(F, gold, TAU), "grid": [prf_at(F, gold, t) for t in TAU_GRID],
                    "ap": average_precision(F, gold), "f_star": Fs, "half": half, "at_half": prf_at(F, gold, half),
                    "tuned": tuned, "at_tuned": prf_at(F, gold, tuned)}
    return out


def bayes_risks(joint: np.ndarray, like: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
    """Thm 2a on a finite world: joint (X, Y) law, observation likelihood p(o | x) (columns o), loss[a, y]. The
    downstream sees the posterior q(· | o) (soft) or argmax_x q(x | o) (hard) and acts Bayes-optimally on what it sees.
    Returns (soft risk, hard risk)."""
    px = joint.sum(1)
    po_x = like  # (X, O)
    joint_o_y = np.einsum("xo,xy->oy", po_x, joint)  # p(o, y)
    soft = sum(min(loss[a] @ joint_o_y[o] for a in range(loss.shape[0])) for o in range(po_x.shape[1]))
    post = po_x * px[:, None]  # p(x, o)
    hard_of = post.argmax(0)  # the argmax x for each o (first on ties)
    joint_h_y = np.zeros((joint.shape[0], joint.shape[1]))
    for o, h in enumerate(hard_of):
        joint_h_y[h] += joint_o_y[o]
    hard = sum(min(loss[a] @ joint_h_y[h] for a in range(loss.shape[0])) for h in range(joint.shape[0]))
    return float(soft), float(hard)


# Thm 2b's counterexample: a mention whose span is right with probability q = 0.6 and whose reader, given the right
# span, says the fact with probability p = 0.8 (and nothing otherwise), so P(fact) = qp = 0.48. A missed fact costs 3,
# a false one 1, so the Bayes rule asserts it when P(fact) > 1/4. The downstream rule is the hard pipeline's: assert
# when the score exceeds 1/2. Hard, the span is kept (q > 1/2) and the score is p = 0.8: asserted, the Bayes act. Soft,
# the score is qp = 0.48: withheld. Expected cost 0.52 · 1 = 0.52 hard, 0.48 · 3 = 1.44 soft.
COUNTER = {"q": 0.6, "p": 0.8, "c_fn": 3.0, "c_fp": 1.0, "tau": 0.5}


def counterexample() -> dict:
    q, p, cfn, cfp, tau = (COUNTER[k] for k in ("q", "p", "c_fn", "c_fp", "tau"))
    prob = q * p
    hard_score, soft_score = (p if q > 0.5 else 0.0), q * p
    cost = lambda assert_: (1 - prob) * cfp if assert_ else prob * cfn  # noqa: E731
    return {"prob": prob, "hard": cost(hard_score > tau), "soft": cost(soft_score > tau),
            "bayes": min(cost(True), cost(False)), "bayes_tau": cfp / (cfp + cfn), "retuned": cost(soft_score > cfp / (cfp + cfn))}


# --------------------------------------------------------------------------------------------
# 5. Joint inference over the KB (Def 4, Thm 3, Prop 3).
# --------------------------------------------------------------------------------------------
#
# Variables are fact indicators x_f with weight w_f = logit(s_f) − logit(τ), so with no rows the MAP is the threshold
# rule. PARENT_OF(a, b) and SUBSIDIARY_OF(b, a) are one variable, as are COMPETES_WITH(a, b) and (b, a): every mention
# of either direction is evidence for the one fact, so the merged score is their noisy-OR (Topic 23's Def 4), never
# below either. (Adding the two directions' log-odds instead lets a reader that states only one direction veto it:
# Brightwater Analytics PARENT_OF Larch Data, 0.685, was dropped at τ = 0.55 for its 0.398 inverse.) Functional rows (from WORLD, every relation's maximum degree):
# at most one tail per (head, r) for the head-functional relations, and at most one head per (r, tail) for the
# tail-functional ones. Rules (Prop 3): BASED_IN = HQ_IN ∘ LOCATED_IN in both directions the KB satisfies, and
# ACQUIRED ⇒ PARENT_OF.

HEAD_FUNCTIONAL = ("HQ_IN", "IN_SECTOR", "BASED_IN", "LOCATED_IN", "SUBSIDIARY_OF", "CEO_OF", "CFO_OF", "ACQUIRED")
TAIL_FUNCTIONAL = ("CEO_OF", "CFO_OF", "ACQUIRED")
INVERSE = {"PARENT_OF": "SUBSIDIARY_OF"}
SYMMETRIC = ("COMPETES_WITH",)


def canonical(f: tuple) -> tuple:
    h, r, t = f
    if r in INVERSE:
        return (t, INVERSE[r], h)
    if r in SYMMETRIC and h > t:
        return (t, r, h)
    return f


def expand(c: tuple) -> set:
    h, r, t = c
    if r == "SUBSIDIARY_OF":
        return {c, (t, "PARENT_OF", h)}
    if r in SYMMETRIC:
        return {c, (t, r, h)}
    return {c}


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def kb_problem(F: dict, tau: float, rules: bool, functional: bool = True) -> dict:
    """Def 4 as matrices: variables (merged facts), weights, the functional rows A_f x ≤ 1, and the rule rows."""
    acc: dict = defaultdict(float)
    for f, s in F.items():
        acc[canonical(f)] += math.log1p(-min(s, 1 - 1e-16))
    V = {k: _logit(-math.expm1(a)) - _logit(tau) for k, a in acc.items()}
    keys = sorted(V)
    ix = {k: n for n, k in enumerate(keys)}
    slots: dict = defaultdict(list)
    for k in keys:
        h, r, t = k
        if r in HEAD_FUNCTIONAL:
            slots[("head", h, r)].append(ix[k])
        if r in TAIL_FUNCTIONAL:
            slots[("tail", t, r)].append(ix[k])
    func = [({j: 1.0 for j in v}, 1.0, key[0]) for key, v in sorted(slots.items()) if len(v) > 1 and functional]
    rule = []
    if rules:
        by = defaultdict(list)
        for k in keys:
            by[k[1]].append(k)
        for h, _, o in by["HQ_IN"]:
            for o2, _, c in by["LOCATED_IN"]:
                kb = (h, "BASED_IN", c)
                if o2 == o and kb in ix:
                    a, l, b = ix[(h, "HQ_IN", o)], ix[(o, "LOCATED_IN", c)], ix[kb]
                    rule.append(({a: 1.0, l: 1.0, b: -1.0}, 1.0, "rule"))
                    rule.append(({a: 1.0, b: 1.0, l: -1.0}, 1.0, "rule"))
        for a_, _, b_ in by["ACQUIRED"]:
            kp = (b_, "SUBSIDIARY_OF", a_)
            if kp in ix:
                rule.append(({ix[(a_, "ACQUIRED", b_)]: 1.0, ix[kp]: -1.0}, 0.0, "rule"))
    rows = func + rule
    A = sparse.lil_matrix((len(rows), len(keys)))
    for i, (row, _, _) in enumerate(rows):
        for j, a in row.items():
            A[i, j] = a
    return {"keys": keys, "w": np.array([V[k] for k in keys]), "A": A.tocsr(), "b": np.array([b for _, b, _ in rows]),
            "family": [fam for _, _, fam in rows], "n_func": len(func), "n_rule": len(rule)}


def solve_kb(prob: dict, integral: bool) -> tuple[np.ndarray, float]:
    """max wᵀx subject to A x ≤ b, 0 ≤ x ≤ 1, as an integer program or its LP relaxation (HiGHS)."""
    n = len(prob["w"])
    cons = (optimize.LinearConstraint(prob["A"], -np.inf, prob["b"]),) if prob["A"].shape[0] else ()
    res = optimize.milp(-prob["w"], constraints=cons, bounds=optimize.Bounds(0, 1),
                        integrality=np.ones(n) if integral else np.zeros(n))
    assert res.success, res.message
    return res.x, float(-res.fun)


def map_facts(F: dict, tau: float, rules: bool, functional: bool = True) -> dict:
    prob = kb_problem(F, tau, rules, functional)
    x, val = solve_kb(prob, True)
    xl, vl = solve_kb(prob, False)
    chosen = set()
    for k, v in zip(prob["keys"], x):
        if v > 0.5:
            chosen |= expand(k)
    frac = int(np.sum((xl > 1e-7) & (xl < 1 - 1e-7)))
    return {"facts": frozenset(chosen), "value": val, "lp_value": vl, "fractional": frac,
            "n_func": prob["n_func"], "n_rule": prob["n_rule"], "n_vars": len(prob["keys"])}


def slot_argmax(F: dict, tau: float) -> frozenset:
    """Thm 3(b): with one functional family per relation, the MAP keeps each slot's best positive-weight fact."""
    prob = kb_problem(F, tau, False)
    best: dict = {}
    keep = set()
    for k, w in zip(prob["keys"], prob["w"]):
        h, r, t = k
        if w <= 0:
            continue
        if r in HEAD_FUNCTIONAL:
            s = ("head", h, r)
            if s not in best or w > best[s][1]:
                best[s] = (k, w)
        else:
            keep |= expand(k)
    return frozenset(keep | {f for k, _ in best.values() for f in expand(k)})


@lru_cache(maxsize=None)
def kb_table(name: str, gold_E: bool) -> dict:
    """Per τ on TAU_GRID: the threshold rule, the merged threshold (inverse and symmetric facts as one variable), the
    functional MAP, and the MAP with rules, against the gold facts."""
    F, gold = fact_scores(name, gold_E), gold_facts(name)
    out = {"tau": list(TAU_GRID), "threshold": [], "merged": [], "functional": [], "rules": [], "fractional": [],
           "flips": []}
    for tau in TAU_GRID:
        thr = frozenset(f for f, v in F.items() if v > tau)
        mm, mf, mr = map_facts(F, tau, False, False), map_facts(F, tau, False), map_facts(F, tau, True)
        for k, s in (("threshold", thr), ("merged", mm["facts"]), ("functional", mf["facts"]), ("rules", mr["facts"])):
            tp = len(s & gold)
            out[k].append({"p": tp / max(1, len(s)), "r": tp / len(gold), "f1": 2 * tp / (len(s) + len(gold)),
                           "n": len(s)})
        out["fractional"].append((mf["fractional"], mr["fractional"]))
        out["flips"].append({"merge_added": sorted(mm["facts"] - thr), "removed": sorted(mm["facts"] - mf["facts"]),
                             "rule_removed": sorted(mf["facts"] - mr["facts"]),
                             "rule_added": sorted(mr["facts"] - mf["facts"])})
    return out


# Prop 3's instance: one company, two candidate HQ cities o₁, o₂, each with a LOCATED_IN fact into country c, and
# BASED_IN(company, c). Variables (hq₁, hq₂, loc₁, loc₂, based); rows hq₁ + hq₂ ≤ 1 and, per i, the two rules
# hqᵢ + locᵢ − based ≤ 1 and hqᵢ + based − locᵢ ≤ 1.
PROP3_A = np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, -1], [1, 0, -1, 0, 1], [0, 1, 0, 1, -1], [0, 1, 0, -1, 1]], float)
PROP3_B = np.ones(5)
PROP3_W = np.array([1.0, 1.0, -2.0, -2.0, 1.0])


def brute_ilp(w: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple[float, tuple]:
    best = (-np.inf, ())
    for z in itertools.product((0, 1), repeat=len(w)):
        z = np.array(z, float)
        if np.all(A @ z <= b + 1e-12) and w @ z > best[0]:
            best = (float(w @ z), tuple(int(v) for v in z))
    return best


def max_abs_minor(A: np.ndarray) -> int:
    """The largest |det| over every square submatrix (TU iff ≤ 1)."""
    m, n = A.shape
    best = 0
    for k in range(1, min(m, n) + 1):
        for R in itertools.combinations(range(m), k):
            for C in itertools.combinations(range(n), k):
                best = max(best, abs(round(float(np.linalg.det(A[np.ix_(R, C)])))))
    return best


def random_functional_matrix(rng: np.random.Generator, n_vars: int) -> np.ndarray:
    """A random KB fragment under Def 4's functional rows: each variable gets a relation and a head and tail from a
    few entities; rows are the head slots of head-functional relations and the tail slots of tail-functional ones."""
    rels = ("HQ_IN", "CEO_OF", "COMPETES_WITH", "SUBSIDIARY_OF", "ACQUIRED")
    facts = set()
    while len(facts) < n_vars:
        facts.add((int(rng.integers(3)), rels[rng.integers(len(rels))], int(rng.integers(3))))
    facts = sorted(facts)
    slots: dict = defaultdict(list)
    for j, (h, r, t) in enumerate(facts):
        if r in HEAD_FUNCTIONAL:
            slots[("head", h, r)].append(j)
        if r in TAIL_FUNCTIONAL:
            slots[("tail", t, r)].append(j)
    rows = [v for v in slots.values() if len(v) > 1] or [[0]]
    A = np.zeros((len(rows), len(facts)))
    for i, v in enumerate(rows):
        A[i, v] = 1
    return A


# --------------------------------------------------------------------------------------------
# 6. How sure are the shares (Prop 4), and the readers compared.
# --------------------------------------------------------------------------------------------

BOOT = 2000


@lru_cache(maxsize=None)
def share_bootstrap(view: str, reader: str) -> dict:
    """Resample companies with replacement (their calls and the facts charged to them), recompute every v(S) from the
    pooled counts, and apply the Shapley matrix: φ* = M f1(W @ C) − baseline."""
    C = game_counts(view, reader)  # (coalitions, companies, 3)
    n = C.shape[1]
    W = _rng(6010 + VIEWS.index(view) * 3 + READERS.index(reader)).multinomial(n, np.full(n, 1.0 / n), size=BOOT)
    V = f1_of(np.einsum("bc,acs->bas", W.astype(float), C))  # (B, coalitions)
    V = V - V[:, :1]
    Phi = V @ shapley_matrix().T  # (B, stages)
    return {"phi": Phi, "lo": np.percentile(Phi, 5, 0), "hi": np.percentile(Phi, 95, 0),
            "p_E_gt_R": float(np.mean(Phi[:, 0] > Phi[:, 1]))}


@lru_cache(maxsize=None)
def reader_correct(reader: str) -> tuple[np.ndarray, np.ndarray]:
    """Per call of both tests: whether the reader's tone is right (with predicted entities and aspects), and the
    call's company (the cluster)."""
    ok, grp = [], []
    for n in TESTS:
        t = tones(n, reader, False, False, False)
        for di, x in zip(calls(n), t):
            ok.append(x == docs(n)[di].tone)
            grp.append(f"{n}:{docs(n)[di].company}")
    return np.array(ok), np.array(grp)


READER_PAIRS = (("LR", "dictionary"), ("aspects", "LR"), ("aspects", "dictionary"))


@lru_cache(maxsize=None)
def reader_tests() -> dict:
    """Topic 28's McNemar exact test per pair of readers, and the approximate-randomization test with whole companies
    swapped (cluster AR), on accuracy."""
    out = {}
    for k, (a, b) in enumerate(READER_PAIRS):
        ca, g = reader_correct(a)
        cb, _ = reader_correct(b)
        n10, n01 = sig.discordant(ca, cb)
        groups = sorted(set(g))
        SA = np.array([[ca[g == c].sum(), (g == c).sum()] for c in groups], float)
        SB = np.array([[cb[g == c].sum(), (g == c).sum()] for c in groups], float)
        ar = sig.ar_test(sig.mean_pooled, SA, SB, 9999, _rng(6030 + k))
        out[(a, b)] = {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()), "n10": n10, "n01": n01,
                       "mcnemar": sig.mcnemar_exact(n10, n01), "cluster_ar": ar["p"], "d": ar["d"]}
    return out


FIXED_NOUNS = frozenset(w for ws in fc.SECTOR_NOUNS.values() for w in ws) | frozenset(fc.COMMON_NOUNS)


@lru_cache(maxsize=None)
def topic_stage(name: str, fixed_only: bool) -> float:
    """Topic 25's ``topic_mixtures`` on a run's calls, its dominant topic scored against the call's sector; fixed_only
    folds in only the 80 seeded sector and common nouns, the words whose sector every generator run shares."""
    ds = [docs(name)[i] for i in calls(name)]
    if not fixed_only:
        theta, labels = lda.topic_mixtures(ds)
    else:
        Phi, labels, idx = lda._topic_model()
        toks = [[idx[w] for w in fc._tokens(d.text) if w in idx and w in FIXED_NOUNS] for d in ds]
        theta = lda.fold_in(toks, Phi, lda.FIN_ALPHA, 200, 50, seed=3880)
    return float(np.mean([labels[int(t.argmax())] == fc._PUBLIC[d.company][1] for t, d in zip(theta, ds)]))


@lru_cache(maxsize=1)
def completion_ranks() -> dict:
    """Topic 24's RotatE completion (fit on the held-out roster's extracted facts at τ = 0.5) on the unseen calls'
    missed facts: filtered realistic ranks of both query directions."""
    missed = sorted(gold_facts("unseen") - facts_out("unseen", False, False))
    cfg, fit = kge.completion_fit("RotatE", 0.5)
    kg = kge.completion_problem(0.5)["kg"]
    ranks = kge.evaluate(cfg, kg, fit.E, fit.R, missed)
    extracted = kge.completion_problem(0.5)["extracted"]
    return {"missed": missed, "ranks": ranks, "in_filings": [f in set(extracted) for f in missed]}


# --------------------------------------------------------------------------------------------
# Verification harness: one test per claim.
# --------------------------------------------------------------------------------------------


def test_stage_e_is_topic23s_and_subjects_are_the_covered_company():
    for n in TESTS:
        assert len(sentence_index(n)) == len(rex.corpus(n)) == len(ner.split(n).toks)
        assert all(s.doc == di for (di, _), s in zip(sentence_index(n), rex.corpus(n)))
        assert all(subjects(n, True)[i] == docs(n)[i].company for i in calls(n))
    assert (len(calls("seen")), len(calls("unseen"))) == (66, 60)


def test_gold_side_is_topic19s_and_topic27s():
    """Every oracle reproduces its prerequisite's own gold reading: aspect facts are Topic 27's ``doc_facts``, the
    tone-bearing sentences Topic 19's ``tone_sentences``."""
    for n in TESTS:
        g = aspects(n, True)
        for di in calls(n):
            d = docs(n)[di]
            assert aspect_facts(g[di]) == frozenset(sig.om.doc_facts(d))
            assert tone_bearing(n, di, True, g[di]) == ab.tone_sentences(d)


def test_cor1_the_aspect_stage_is_a_null_player():
    """The matcher finds exactly the gold aspect spans (the vocabulary is closed and exhaustive) and Topic 19's bigram
    reader reads every one right, so the stage's oracle changes nothing: v(S ∪ A) = v(S) for every S, and A gets 0 in
    every order and under Shapley, for every reader and view."""
    for n in TESTS:
        assert aspects(n, False) == aspects(n, True)
    for view in VIEWS:
        for reader in READERS:
            for metric in ("f1", "errors"):
                v = game(view, reader, metric)
                assert all(v[S | {"A"}] == v[S] for S in COALITIONS)
                assert shapley(v)["A"] == 0
                assert all(sequential(v, o)["A"] == 0 for o in itertools.permutations(STAGES))


def _random_game(rng: np.random.Generator, players: tuple, max_order: int) -> dict:
    d = {T: (rng.normal() if 1 <= len(T) <= max_order else 0.0)
         for T in (frozenset(c) for k in range(len(players) + 1) for c in itertools.combinations(players, k))}
    return {S: sum(x for T, x in d.items() if T <= S) for S in d}


def test_thm1_ceiling_analysis_and_the_shapley_value():
    """(a) The sequential share of i in order π is the sum of the dividends of the coalitions whose last member (in π)
    is i. (b) Every order agrees iff every dividend of two or more players is 0. (c) Shapley = the average over orders
    = each dividend split equally among its members; it is efficient, symmetric, null on null players, additive, and
    gives a unanimity game u_T exactly 1/|T| on T."""
    rng = _rng(6000)
    P = ("a", "b", "c", "d")
    for trial in range(30):
        v = _random_game(rng, P, 4 if trial % 3 else 1)
        d = dividends(v, P)
        phi = shapley(v, P)
        orders = list(itertools.permutations(P))
        avg = {i: np.mean([sequential(v, o)[i] for o in orders]) for i in P}
        for i in P:
            assert abs(phi[i] - avg[i]) < 1e-12
            assert abs(phi[i] - sum(x / len(T) for T, x in d.items() if i in T)) < 1e-12
        for o in orders:
            seq = sequential(v, o)
            for i in P:
                last = [T for T in d if T and max(T, key=o.index) == i]
                assert abs(seq[i] - sum(d[T] for T in last)) < 1e-12
        assert abs(sum(phi.values()) - v[frozenset(P)]) < 1e-12  # efficiency
        agree = all(max(abs(sequential(v, o)[i] - sequential(v, orders[0])[i]) for i in P) < 1e-12 for o in orders)
        assert agree == all(abs(x) < 1e-12 for T, x in d.items() if len(T) >= 2)
    for T in (frozenset("a"), frozenset("ab"), frozenset("bcd")):
        u = {S: float(T <= S) for S in _random_game(rng, P, 1)}
        assert all(abs(shapley(u, P)[i] - (1 / len(T) if i in T else 0)) < 1e-12 for i in P)
    v1, v2 = _random_game(rng, P, 4), _random_game(rng, P, 4)
    s = shapley({S: v1[S] + v2[S] for S in v1}, P)
    assert all(abs(s[i] - shapley(v1, P)[i] - shapley(v2, P)[i]) < 1e-12 for i in P)  # additivity


def test_thm1_on_the_corpus_entities_and_relations_are_complements():
    """The corpus game's one sizable dividend is d({E, R}) > 0: a fact needs both its arguments and its relation, so
    each oracle is worth more once the other has joined, and a ceiling analysis that fixes E first credits it less
    than one that fixes R first."""
    for view in VIEWS:
        for reader in ("dictionary", "LR"):
            v = game(view, reader)
            d = dividends(v)
            assert d[frozenset("ER")] > 0.0028
            assert max(abs(x) for T, x in d.items() if len(T) >= 2 and T != frozenset("ER")) < d[frozenset("ER")]
            first, last = sequential(v, ("E", "R", "T", "A"))["E"], sequential(v, ("R", "E", "T", "A"))["E"]
            assert last > first
    v = game("both", "LR")
    assert abs(sum(shapley(v).values()) - v[frozenset(STAGES)]) < 1e-12


def test_prop2_errors_add_over_disjoint_triples_f1_does_not():
    """The dictionary and LR read tone from the whole call, so stage T touches only TONE triples, which no other
    stage writes: in the error game every dividend containing T and another stage is exactly 0. Micro F1 couples
    them through its denominator, so the same dividends are nonzero there. The aspect reader's tone depends on
    stage E's types (tone-bearing sentences name no company), but on these calls stage E's errors never move a
    sentence in or out, so its error game is additive too."""
    for view in VIEWS:
        for reader in ("dictionary", "LR"):
            de, df = dividends(game(view, reader, "errors")), dividends(game(view, reader))
            for T in de:
                if "T" in T and len(T) >= 2:
                    assert de[T] == 0, (view, reader, T)
            assert abs(df[frozenset("ET")]) > 1e-4 and abs(df[frozenset("RT")]) > 1e-4
            assert de[frozenset("ER")] > 0
    for n in TESTS:  # the aspect reader's tone-bearing sentences survive stage E's errors: its tone never moves
        assert aspect_tones(n, False, False) == aspect_tones(n, True, False)
    for view in VIEWS:
        de = dividends(game(view, "aspects", "errors"))
        assert all(x == 0 for T, x in de.items() if len(T) >= 2 and T != frozenset("ER"))


def test_prop1_two_sided_bounds_and_modularity():
    """Bonferroni ≤ P(A ∩ B) ≤ min(r_E, r_R), with A = both arguments found and B = the reader recovers the fact from
    gold mentions. End to end equals P(A ∩ B) only if stage R reads found arguments as it reads gold ones: true on the
    unseen calls, false on the seen calls, where predicted spans change the reader's context (0.696 < 0.739). The
    product sits below P(A ∩ B) in every view: success at the two stages is positively associated."""
    for view in VIEWS:
        b = recall_bounds(view)
        assert b["bonferroni"] <= b["intersection"] <= b["min"]
        assert b["end"] <= b["intersection"]
        assert b["product"] <= b["intersection"]
    names = {n: recall_bounds(n) for n in TESTS}
    assert names["unseen"]["end"] == names["unseen"]["intersection"]
    assert names["seen"]["intersection"] - names["seen"]["end"] > 0.04
    for n in TESTS:  # end-to-end successes are always inside A ∩ B
        win = facts_out(n, False, False) & gold_facts(n)
        assert win <= stated(n, False) & facts_out(n, True, False)


def test_thm2a_the_argmax_is_a_garbling():
    """On random finite worlds, a Bayes downstream that sees the posterior never does worse than one that sees its
    argmax, and on some worlds strictly better."""
    rng = _rng(6002)
    strict = 0
    for _ in range(300):
        nx, ny, no = rng.integers(2, 5), rng.integers(2, 4), rng.integers(2, 7)
        joint = rng.dirichlet(np.ones(nx * ny)).reshape(nx, ny)
        like = rng.dirichlet(np.ones(no), size=nx)
        loss = rng.random((rng.integers(2, 4), ny))
        soft, hard = bayes_risks(joint, like, loss)
        assert soft <= hard + 1e-12
        strict += soft < hard - 1e-9
    assert strict == 22


def test_thm2b_a_fixed_rule_can_lose():
    c = counterexample()
    assert abs(c["prob"] - 0.48) < 1e-12 and c["bayes_tau"] == 0.25
    assert abs(c["hard"] - 0.52) < 1e-12 and abs(c["soft"] - 1.44) < 1e-12
    assert c["hard"] == c["bayes"] == c["retuned"] < c["soft"]


def test_ffbs_samples_the_constrained_chain():
    """Forward filtering, backward sampling on random chains: every joint path's frequency against its exact
    probability (brute force), per-cell z-scores where the expected count is at least 5."""
    rng = _rng(6003)
    for _ in range(3):
        U = rng.normal(size=(2, 4, 3))
        P = rng.normal(size=(3, 3))
        Z = ffbs(U, P, rng, 4000)
        for j in range(2):
            paths = list(itertools.product(range(3), repeat=4))
            sc = np.array([U[j, np.arange(4), list(z)].sum() + sum(P[a, b] for a, b in zip(z, z[1:])) for z in paths])
            p = np.exp(sc - sc.max())
            p /= p.sum()
            cnt = Counter(tuple(int(x) for x in Z[k, j]) for k in range(4000))
            for z, pz in zip(paths, p):
                if 4000 * pz >= 5:
                    assert abs(cnt[z] - 4000 * pz) / math.sqrt(4000 * pz * (1 - pz)) < 4.5


def test_handoffs_on_the_corpus():
    """Soft handoffs carry information (AP rises on the seen calls, where entity errors are common) but lower F1 at
    the hard pipeline's τ = 0.5, since q-scaled scores fall below it. Topic 22's F*/2 rule, from the scores alone,
    sits below 0.5 and raises every handoff's F1, but noisy-OR's soft scores are underconfident and it stays short;
    a threshold tuned on the same run's filings gives the soft handoffs the best F1 on the seen calls and ties the
    hard one on the unseen calls."""
    for n in TESTS:
        t = handoff_table(n)
        for how in ("marginal", "sampled"):
            assert t[how]["at"]["f1"] < t["hard"]["at"]["f1"]
            assert t[how]["at_tuned"]["f1"] >= t["hard"]["at_tuned"]["f1"] - 0.02
        for how in HANDOFFS:
            assert t[how]["half"] < TAU and t[how]["at_half"]["f1"] >= t[how]["at"]["f1"]
            assert t[how]["at"]["p"] == 1.0
    s = handoff_table("seen")
    assert s["marginal"]["ap"] - s["hard"]["ap"] > 0.1 and s["sampled"]["ap"] - s["hard"]["ap"] > 0.09
    assert s["marginal"]["at_tuned"]["f1"] - s["hard"]["at_tuned"]["f1"] > 0.04
    assert s["marginal"]["at_half"]["f1"] < s["hard"]["at"]["f1"]


def test_soft_handoffs_keep_mass_on_missed_spans():
    """Of the 92 gold spans the hard tagger misses on the seen calls, 61 keep marginal mass above 0.1 (30 of 38 on the
    unseen calls): the information a soft handoff passes on."""
    for n, (miss, kept) in zip(TESTS, ((92, 61), (38, 30))):
        v, s22 = span_views(n), ner.split(n)
        rows = [i for i, s in enumerate(rex.corpus(n)) if s.doc in doc_ids(n, "calls")]
        lost = [(i, g) for i in rows for g in s22.spans[i] if g not in v[i][0]]
        assert len(lost) == miss and sum(v[i][1].get(g, 0.0) > 0.1 for i, g in lost) == kept


def test_merge_by_noisy_or_not_by_log_odds():
    """Adding the two directions' log-odds drops Brightwater Analytics PARENT_OF Larch Data (0.685) at τ = 0.55 for its
    0.398 inverse; the noisy-OR merge keeps both directions. And the aspect stage reads 1,209 gold aspects."""
    F = fact_scores("seen", True)
    f, inv = ("Brightwater Analytics", "PARENT_OF", "Larch Data"), ("Larch Data", "SUBSIDIARY_OF", "Brightwater Analytics")
    assert abs(F[f] - 0.685) < 5e-4 and abs(F[inv] - 0.398) < 5e-4
    assert _logit(F[f]) + _logit(F[inv]) - 2 * _logit(0.55) < 0
    assert {f, inv} <= map_facts(F, 0.55, False)["facts"]
    assert sum(len(aspects(n, True)[i]) for n in TESTS for i in calls(n)) == 1209


def test_thm3_functional_rows_are_totally_unimodular():
    """(a) Random KB fragments under the functional rows: every square minor is 0 or ±1. On the corpus the LP
    relaxation is integral at every τ, and for random weights on the same constraint matrices. (b) Where no
    tail-functional row binds, the MAP is each slot's best positive fact; at low τ the one-CEO-per-company rows bind
    on the unseen calls, and there it is a matching."""
    rng = _rng(6004)
    for _ in range(40):
        A = random_functional_matrix(rng, int(rng.integers(3, 8)))
        assert max_abs_minor(A) <= 1
    for n in TESTS:
        for gE in (True, False):
            t = kb_table(n, gE)
            assert all(fr[0] == 0 for fr in t["fractional"])
            F = fact_scores(n, gE)
            prob = kb_problem(F, TAU, False)
            for _ in range(3):
                prob["w"] = rng.normal(size=len(prob["w"]))
                x, _ = solve_kb(prob, False)
                assert np.all((x < 1e-7) | (x > 1 - 1e-7))
            for tau in TAU_GRID:  # they differ only where a one-CEO-per-company row binds (a bipartite matching)
                diff = slot_argmax(F, tau) ^ map_facts(F, tau, False)["facts"]
                assert all(r in TAIL_FUNCTIONAL for _, r, _ in diff)
    for n in TESTS:
        Fn = fact_scores(n, False)
        binds = [t for t in TAU_GRID if slot_argmax(Fn, t) != map_facts(Fn, t, False)["facts"]]
        assert binds == ([0.05] if n == "unseen" else [])


def test_prop3_composition_rules_break_total_unimodularity():
    """Two rules on one (HQ, LOCATED, BASED) triangle give a minor of −2, and with weights (1, 1, −2, −2, 1) the LP
    puts half of the HQ on each city (value 3/2) where the best integer point is worth 1. On the corpus the LP with
    rules is integral anyway, at every τ."""
    assert max_abs_minor(PROP3_A) >= 2
    assert round(float(np.linalg.det(PROP3_A[np.ix_((1, 2), (0, 2))]))) == -2
    prob = {"w": PROP3_W, "A": sparse.csr_matrix(PROP3_A), "b": PROP3_B}
    x, val = solve_kb(prob, False)
    assert abs(val - 1.5) < 1e-9
    assert np.allclose(x, [0.5, 0.5, 0, 0, 0.5]) or abs(PROP3_W @ np.array([0.5, 0.5, 0, 0, 0.5]) - val) < 1e-9
    assert brute_ilp(PROP3_W, PROP3_A, PROP3_B)[0] == 1.0
    assert abs(solve_kb(prob, True)[1] - 1.0) < 1e-9
    for n in TESTS:
        for gE in (True, False):
            assert all(fr[1] == 0 for fr in kb_table(n, gE)["fractional"])


def test_joint_inference_on_the_corpus():
    """Merging inverse and symmetric facts (noisy-OR over both directions' mentions) completes the direction a reader
    missed, raising F1 at every τ ≥ 0.4, but at low τ it also completes the inverses of weak false facts. The
    functional rows never lower F1 against the merged threshold and raise it at τ = 0.05 on every corpus (precision
    repair: a second HQ city, a region read as a city, a second CFO). The rules lower F1 at τ = 0.5 on the seen calls,
    where they delete true LOCATED_IN facts whose BASED_IN leg no call states (the closed world again)."""
    for n in TESTS:
        for gE in (True, False):
            t = kb_table(n, gE)
            assert all(f["f1"] >= m["f1"] for f, m in zip(t["functional"], t["merged"]))
            assert all(m["f1"] >= h["f1"] for tau, m, h in zip(t["tau"], t["merged"], t["threshold"]) if tau >= 0.4)
            k = t["tau"].index(0.05)
            assert t["functional"][k]["f1"] > t["merged"][k]["f1"]
            assert all(f not in gold_facts(n) for f in t["flips"][k]["removed"])  # every repair removes a false fact
    assert any(m["f1"] < h["f1"] for m, h in zip(kb_table("seen", True)["merged"], kb_table("seen", True)["threshold"]))
    for gE in (True, False):
        t = kb_table("seen", gE)
        k = t["tau"].index(TAU)
        assert t["rules"][k]["f1"] < t["functional"][k]["f1"]
        assert any(f[1] == "LOCATED_IN" and f in WORLD.triples for f in t["flips"][k]["rule_removed"])


def test_filings_fill_what_the_calls_missed():
    """Every fact the unseen calls state but the pipeline misses is in Topic 24's completion KB (extracted from the
    same companies' filings) or is the inverse of one that is, and RotatE ranks all of them first."""
    c = completion_ranks()
    assert len(c["missed"]) == 6 and all(r == 1 for r in c["ranks"])
    ext = set(kge.completion_problem(0.5)["extracted"])
    for f, inside in zip(c["missed"], c["in_filings"]):
        assert inside or any(e in ext for e in expand(canonical(f)))


def test_prop4_shapley_is_linear_so_one_bootstrap_serves_every_share():
    """φ = M v with M the Shapley matrix, and v a ratio of pooled counts, so φ* = M f1(W @ C) resamples every share.
    On the full sample it reproduces the dict Shapley exactly. Tone's share excludes 0 in every view; entities' and
    relations' cannot be ordered (P(φ_E* > φ_R*) between 0.2 and 0.9), and on the unseen calls entities' share
    touches 0."""
    M = shapley_matrix()
    for view in VIEWS:
        for reader in READERS:
            v = game(view, reader)
            phi = M @ np.array([v[S] for S in COALITIONS])
            assert np.allclose(phi, [shapley(v)[i] for i in STAGES], atol=1e-14)
            b = share_bootstrap(view, reader)
            assert b["lo"][2] > 0 and abs(b["lo"][3]) < 1e-12 and abs(b["hi"][3]) < 1e-12
        assert 0.2 < share_bootstrap(view, "LR")["p_E_gt_R"] < 0.9
    assert share_bootstrap("unseen", "LR")["lo"][0] <= 1e-12


def test_tone_readers_compared():
    """On the 126 calls: aspects 0.857 > LR 0.738 > dictionary 0.587. Every McNemar p < 0.01; swapping whole
    companies (calls of one company share a subject and a coverage rate) gives the larger p for LR vs dictionary."""
    t = reader_tests()
    acc = {r: float(reader_correct(r)[0].mean()) for r in READERS}
    assert acc["aspects"] > acc["LR"] > acc["dictionary"]
    assert all(x["mcnemar"] < 0.01 for x in t.values())
    assert t[("LR", "dictionary")]["cluster_ar"] > t[("LR", "dictionary")]["mcnemar"]
    assert abs(acc["aspects"] - 108 / 126) < 1e-12


def test_tone_share_is_mostly_the_bayes_floor():
    """The Bayes tone (Topic 19's Bayes rule on the gold polarities of the gold tone-bearing sentences) is what the
    aspect reader outputs on every call, so the aspect reader's tone share is error no reader can remove: Topic 17's
    ceiling (0.890) bounds every reader of this text. Only the part of a reader's share above it is fixable."""
    for n in TESTS:
        assert aspect_tones(n, True, True) == aspect_tones(n, False, False)
    assert abs(float(reader_correct("aspects")[0].mean()) - 108 / 126) < 1e-12
    v_lr, v_asp = shapley(game("both", "LR"))["T"], shapley(game("both", "aspects"))["T"]
    assert v_asp > 0.02 and v_lr - v_asp > 0.018


def test_topic_stage_needs_the_fixed_nouns():
    """On seed-1 calls Topic 25's fold-in reads each generator run's own pseudo-words with seed 0's sectors; folding
    in only the 80 fixed nouns raises dominant-topic sector accuracy on both rosters."""
    for n in TESTS:
        assert topic_stage(n, True) > topic_stage(n, False) + 0.15


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


# --------------------------------------------------------------------------------------------
# Lab payload and summary.
# --------------------------------------------------------------------------------------------

VIZ_JSON = _HERE.parent.parent / "src" / "components" / "viz" / "data" / "capstone-pipeline-lab.json"
ORDERS = tuple(itertools.permutations(STAGES))


def _key(S) -> str:
    return "".join(i for i in STAGES if i in S)


def _cased(text: str) -> list[str]:
    """The raw-cased tokens, aligned one for one with fc._tokens."""
    import re
    return re.findall(r"[A-Za-z]+", text)


def call_errors(name: str, di: int) -> dict:
    """Which stages err on one call (tone by the LR reader), and how many of its sentences have a wrong span set."""
    pos = [k for k, (d, _) in enumerate(sentence_index(name)) if d == di]
    G, P = rex.corpus(name), rex.predicted_corpus(name)
    e_bad = sum({(m.a, m.b, m.type) for m in G[k].mentions} != {(m.a, m.b, m.type) for m in P[k].mentions} for k in pos)
    own = {f for f in gold_facts(name) | facts_out(name, False, False) if pair_owner(name)[(f[0], f[2])] == di}
    r_bad = any((f in gold_facts(name)) != (f in facts_out(name, False, False)) for f in own)
    t_bad = tones(name, "LR", False, False, False)[calls(name).index(di)] != docs(name)[di].tone
    return {"E": e_bad > 0, "R": r_bad, "T": t_bad, "e_sentences": e_bad}


@lru_cache(maxsize=None)
def trace_call(name: str) -> int:
    """The lab's example: the call erring at the most stages, then with the most mis-tagged sentences, then first."""
    return max(calls(name), key=lambda di: (sum(call_errors(name, di)[k] for k in "ERT"),
                                            call_errors(name, di)["e_sentences"], -di))


def _trace_payload(name: str) -> dict:
    di = trace_call(name)
    d = docs(name)[di]
    G, P = rex.corpus(name), rex.predicted_corpus(name)
    pa, ga = aspects(name, False)[di], aspects(name, True)[di]
    sents = []
    for k, (dd, si) in enumerate(sentence_index(name)):
        if dd != di:
            continue
        s0, s1 = d.sentences[si]
        toks = _cased(d.text[s0:s1])
        assert [w.lower() for w in toks] == list(G[k].toks)

        def asp(lst):
            out = []
            for a_si, a0, a1, cat, pol in lst:
                if a_si == si:
                    lo = len(fc._tokens(d.text[s0:a0]))
                    out.append([lo, lo + len(fc._tokens(d.text[a0:a1])), cat, pol])
            return out

        sents.append({"tokens": toks,
                      "gold": [[m.a, m.b, m.type, m.kb] for m in G[k].mentions],
                      "pred": [[m.a, m.b, m.type, m.kb] for m in P[k].mentions],
                      "aspects": asp(pa), "gold_aspects": asp(ga),
                      "tone_bearing": si in tone_bearing(name, di, False, pa)})
    own = sorted(f for f in gold_facts(name) | facts_out(name, False, False) if pair_owner(name)[(f[0], f[2])] == di)
    F = fact_scores(name, False)
    j = calls(name).index(di)
    return {"test": name, "doc_id": d.doc_id, "company": d.company, "subject": subjects(name, False)[di],
            "tone": d.tone, "tones": {r: tones(name, r, False, False, False)[j] for r in READERS},
            "aspect_facts": sorted([c, s_] for c, s_ in aspect_facts(pa)), "sentences": sents,
            "facts": [{"h": h, "r": r, "t": t, "gold": (h, r, t) in gold_facts(name),
                       "pred": (h, r, t) in facts_out(name, False, False), "score": _r(F.get((h, r, t), 0.0), 3)}
                      for h, r, t in own],
            "errors": {k: bool(v) for k, v in call_errors(name, di).items() if k in "ERT"}}


def _game_payload() -> dict:
    out = {}
    for view in VIEWS:
        out[view] = {}
        for reader in READERS:
            C = game_counts(view, reader).sum(1)
            v, e = game(view, reader), game(view, reader, "errors")
            d, b = dividends(v), share_bootstrap(view, reader)
            out[view][reader] = {
                "f1": {_key(S): _r(float(x), 3) for S, x in zip(COALITIONS, f1_of(C))},
                "errors": {_key(S): int(x) for S, x in zip(COALITIONS, errors_of(C))},
                "shapley": [_r(shapley(v)[i], 4) for i in STAGES],
                "shapley_errors": [_r(shapley(e)[i], 2) for i in STAGES],
                "dividends": {_key(T): _r(x, 4) for T, x in d.items() if len(T) >= 2 and abs(x) > 1e-12},
                "orders": {"".join(o): [_r(sequential(v, o)[i], 4) for i in STAGES] for o in ORDERS},
                "orders_errors": {"".join(o): [int(sequential(e, o)[i]) for i in STAGES] for o in ORDERS},
                "lo": [_r(float(x), 4) + 0.0 for x in b["lo"]], "hi": [_r(float(x), 4) + 0.0 for x in b["hi"]],
                "p_E_gt_R": _r(b["p_E_gt_R"], 3)}
    return out


def _prf(x: dict) -> dict:
    return {"p": _r(x["p"], 3), "r": _r(x["r"], 3), "f1": _r(x["f1"], 3), "n": x["n"]}


def _fact(f: tuple) -> str:
    return f"{f[0]} {f[1]} {f[2]}"


def viz_payload() -> dict:
    handoffs = {}
    for n in TESTS:
        t = handoff_table(n)
        handoffs[n] = {how: {"grid": [_prf(x) for x in t[how]["grid"]], "at": _prf(t[how]["at"]),
                             "ap": _r(t[how]["ap"], 3), "half": _r(t[how]["half"], 3),
                             "at_half": _prf(t[how]["at_half"]), "tuned": t[how]["tuned"],
                             "at_tuned": _prf(t[how]["at_tuned"])} for how in HANDOFFS}
    kb = {}
    for n in TESTS:
        kb[n] = {}
        for gE in (True, False):
            t = kb_table(n, gE)
            kb[n]["gold" if gE else "predicted"] = {
                m: [_prf(x) for x in t[m]] for m in ("threshold", "merged", "functional", "rules")} | {
                "fractional": [list(x) for x in t["fractional"]],
                "flips": [{k: [_fact(f) for f in v] for k, v in fl.items()} for fl in t["flips"]]}
    x, val = solve_kb({"w": PROP3_W, "A": sparse.csr_matrix(PROP3_A), "b": PROP3_B}, False)
    ilp = brute_ilp(PROP3_W, PROP3_A, PROP3_B)
    rt = reader_tests()
    comp = completion_ranks()
    return {
        "stages": [{"key": k, "name": STAGE_NAMES[k]} for k in STAGES], "readers": list(READERS),
        "views": list(VIEWS), "coalitions": [_key(S) for S in COALITIONS], "orders": ["".join(o) for o in ORDERS],
        "counts": {n: {"calls": len(calls(n)), "companies": len(companies(n)), "facts": len(gold_facts(n)),
                       "triples": len(gold_graph(n))} for n in TESTS},
        "trace": {n: _trace_payload(n) for n in TESTS},
        "games": _game_payload(),
        "bounds": {view: {k: _r(v, 3) for k, v in recall_bounds(view).items()} for view in VIEWS},
        "tau_grid": list(TAU_GRID), "tau": TAU,
        "handoffs": handoffs,
        "kb": kb,
        "prop3": {"names": ["hq1", "hq2", "loc1", "loc2", "based"], "A": PROP3_A.astype(int).tolist(),
                  "w": PROP3_W.tolist(), "lp_x": [_r(float(v), 3) + 0.0 for v in x], "lp": _r(val, 3),
                  "ilp": ilp[0], "ilp_x": list(ilp[1])},
        "counter": {k: _r(v, 3) for k, v in counterexample().items()} | {k: v for k, v in COUNTER.items()},
        "readers_compared": {"accuracy": {r: _r(float(reader_correct(r)[0].mean()), 3) for r in READERS},
                             "pairs": [{"a": a, "b": b, "n10": rt[(a, b)]["n10"], "n01": rt[(a, b)]["n01"],
                                        "mcnemar": _r(rt[(a, b)]["mcnemar"], 4), "cluster_ar": _r(rt[(a, b)]["cluster_ar"], 4)}
                                       for a, b in READER_PAIRS],
                             "by_test": {n: {r: _r(float(np.mean(np.array(tones(n, r, False, False, False))
                                                                  == np.array([docs(n)[i].tone for i in calls(n)]))), 3)
                                             for r in READERS} for n in TESTS}},
        "topic": {n: {"all": _r(topic_stage(n, False), 3), "fixed": _r(topic_stage(n, True), 3)} for n in TESTS},
        "completion": [{"fact": _fact(f), "in_filings": inside, "ranks": comp["ranks"][2 * k:2 * k + 2]}
                       for k, (f, inside) in enumerate(zip(comp["missed"], comp["in_filings"]))],
    }


def _fmt(*v) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def summary() -> dict:
    out = {}
    for n in TESTS:
        out[f"{n}: calls, companies, gold facts, gold triples"] = (
            f"{len(calls(n))} {len(companies(n))} {len(gold_facts(n))} {len(gold_graph(n))}")
    for view in VIEWS:
        for reader in READERS:
            v = game(view, reader)
            C = game_counts(view, reader).sum(1)
            phi, b = shapley(v), share_bootstrap(view, reader)
            out[f"[{view}, {reader}] F1(∅) F1(N) | Shapley E R T A"] = (
                _fmt(f1_of(C[0]), f1_of(C[-1])) + " | " + _fmt(*(phi[i] for i in STAGES)))
            out[f"[{view}, {reader}] d(ER) d(ET) d(RT) | 90% lo/hi E, R, T | P(E>R)"] = (
                _fmt(*(dividends(v)[frozenset(k)] for k in ("ER", "ET", "RT"))) + " | "
                + _fmt(b["lo"][0], b["hi"][0], b["lo"][1], b["hi"][1], b["lo"][2], b["hi"][2]) + f" | {b['p_E_gt_R']:.6f}")
            out[f"[{view}, {reader}] error game: Shapley E R T A, d(ER)"] = (
                _fmt(*(shapley(game(view, reader, 'errors'))[i] for i in STAGES))
                + f" {dividends(game(view, reader, 'errors'))[frozenset('ER')]:.6f}")
        out[f"[{view}, LR] E first vs R first: E's share"] = _fmt(sequential(game(view, "LR"), ("E", "R", "T", "A"))["E"],
                                                                   sequential(game(view, "LR"), ("R", "E", "T", "A"))["E"])
        b = recall_bounds(view)
        out[f"[{view}] r_E r_R | Bonferroni product P(A∩B) end min"] = (
            _fmt(b["r_E"], b["r_R"]) + " | " + _fmt(b["bonferroni"], b["product"], b["intersection"], b["end"], b["min"]))
    for n in TESTS:
        t = handoff_table(n)
        for how in HANDOFFS:
            h = t[how]
            out[f"[{n}, {how}] P R F1 @0.5 | AP | F*/2, F1 there | tuned τ, F1 there"] = (
                _fmt(h["at"]["p"], h["at"]["r"], h["at"]["f1"]) + f" | {h['ap']:.6f} | "
                + _fmt(h["half"], h["at_half"]["f1"]) + f" | {h['tuned']:g} {h['at_tuned']['f1']:.6f}")
    for n in TESTS:
        for gE in (True, False):
            t = kb_table(n, gE)
            for tau in (0.05, 0.1, 0.5, 0.8):
                k = t["tau"].index(tau)
                out[f"[{n}, {'gold' if gE else 'predicted'} spans, τ={tau:g}] F1 threshold merged functional rules"] = (
                    _fmt(*(t[m][k]["f1"] for m in ("threshold", "merged", "functional", "rules"))))
    for (a, b_), x in reader_tests().items():
        out[f"tone {a} vs {b_}: acc a, acc b, n10 n01, McNemar p, cluster-AR p"] = (
            _fmt(x["acc_a"], x["acc_b"]) + f" {x['n10']} {x['n01']} " + _fmt(x["mcnemar"], x["cluster_ar"]))
    for n in TESTS:
        out[f"[{n}] Bayes tone accuracy (= aspect reader)"] = _fmt(float(np.mean(
            np.array(aspect_tones(n, True, True)) == np.array([docs(n)[i].tone for i in calls(n)]))))
        out[f"[{n}] topic stage: sector accuracy all words, fixed nouns"] = _fmt(topic_stage(n, False), topic_stage(n, True))
    c = completion_ranks()
    out["unseen calls: missed facts, in filings' KB, RotatE ranks"] = (
        f"{len(c['missed'])} {sum(c['in_filings'])} {sorted(set(c['ranks']))}")
    for n in TESTS:
        out[f"[{n}] trace call, errors"] = f"{docs(n)[trace_call(n)].doc_id} {call_errors(n, trace_call(n))}"
    return out


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
    if failed == 0:
        print("\nSummary:")
        for k, v in summary().items():
            print(f"  {k:84} {v:.6f}" if isinstance(v, float) else f"  {k:84} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
