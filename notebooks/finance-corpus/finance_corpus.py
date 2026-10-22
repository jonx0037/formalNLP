"""Shared synthetic finance corpus for formalNLP's finance case studies.

Not a topic: the data layer every ``financeCaseStudy`` notebook and the capstone import, so the
whole site argues from ONE corpus. Provides

* ``generate_corpus`` — a seeded synthetic corpus of earnings-call, MD&A, and 8-K style documents
  about a fictional company roster, with gold tone labels, entity spans, and relations;
* ``STANDIN_LEXICON`` / ``lexicon_score`` — a small hand-written sentiment lexicon. It is NOT
  derived from the Loughran–McDonald Master Dictionary (non-commercial license; never committed
  here). Topics that teach the real dictionary fetch it at runtime;
* ``load_edgar_excerpts`` — short public MD&A excerpts from SEC EDGAR, listed in
  ``edgar/SOURCES.md``.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/finance-corpus/finance_corpus.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

EntityType = Literal["ORG", "PERSON", "MONEY", "PERCENT", "DATE", "PRODUCT"]
RelationType = Literal["CEO_OF", "REPORTED", "ACQUIRED", "GUIDED"]
DocType = Literal["earnings-call", "mdna", "8-k"]
Tone = Literal[-1, 0, 1]
LexiconCategory = Literal["positive", "negative", "uncertainty", "litigious"]

EDGAR_DIR = Path(__file__).resolve().parent / "edgar"


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    type: EntityType


@dataclass(frozen=True)
class Relation:
    head: int  # index into FinanceDoc.entities
    rel: RelationType
    tail: int


@dataclass(frozen=True)
class FinanceDoc:
    doc_id: str
    company: str
    doc_type: DocType
    text: str
    tone: Tone
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]


# Hand-written for this site; deliberately NOT drawn from the Loughran–McDonald dictionary.
# Every word carries a part of speech so each template slot gets a grammatical filler.
LEXICON_POS: dict[str, str] = {
    # positive
    "strong": "ADJ", "improved": "ADJ", "robust": "ADJ", "favorable": "ADJ", "resilient": "ADJ",
    "healthy": "ADJ", "solid": "ADJ", "stronger": "ADJ", "efficient": "ADJ", "confident": "ADJ",
    "gains": "NOUN", "momentum": "NOUN", "upside": "NOUN", "strength": "NOUN",
    "exceeded": "VERB", "outperformed": "VERB", "beat": "VERB", "surpassed": "VERB",
    # negative
    "weak": "ADJ", "adverse": "ADJ", "weaker": "ADJ", "unfavorable": "ADJ", "disappointing": "ADJ",
    "soft": "ADJ", "loss": "NOUN", "impairment": "NOUN", "shortfall": "NOUN", "headwinds": "NOUN",
    "softness": "NOUN", "contraction": "NOUN", "writedown": "NOUN", "downturn": "NOUN",
    "missed": "VERB", "trailed": "VERB", "undershot": "VERB",
    # uncertainty
    "uncertain": "ADJ", "unpredictable": "ADJ", "contingent": "ADJ", "preliminary": "ADJ",
    "tentative": "ADJ", "unclear": "ADJ", "variability": "NOUN", "fluctuations": "NOUN",
    "volatility": "NOUN", "assumptions": "NOUN", "estimates": "NOUN", "risk": "NOUN",
    "may": "MODAL", "could": "MODAL", "approximately": "ADV", "possibly": "ADV",
    # litigious
    "litigation": "NOUN", "plaintiff": "NOUN", "lawsuit": "NOUN", "settlement": "NOUN",
    "indemnification": "NOUN", "subpoena": "NOUN", "arbitration": "NOUN", "claimant": "NOUN",
    "counsel": "NOUN", "allegation": "NOUN", "injunction": "NOUN", "statute": "NOUN",
    "jurisdiction": "NOUN", "defendant": "NOUN", "regulatory": "ADJ",
}
_CATEGORY_WORDS = {
    "positive": "strong improved robust favorable resilient healthy solid stronger efficient confident "
                "gains momentum upside strength exceeded outperformed beat surpassed",
    "negative": "weak adverse weaker unfavorable disappointing soft loss impairment shortfall headwinds "
                "softness contraction writedown downturn missed trailed undershot",
    "uncertainty": "uncertain unpredictable contingent preliminary tentative unclear variability "
                   "fluctuations volatility assumptions estimates risk may could approximately possibly",
    "litigious": "litigation plaintiff lawsuit settlement indemnification subpoena arbitration claimant "
                 "counsel allegation injunction statute jurisdiction defendant regulatory",
}
STANDIN_LEXICON: dict[LexiconCategory, frozenset[str]] = {
    cat: frozenset(words.split()) for cat, words in _CATEGORY_WORDS.items()  # type: ignore[misc]
}

# sha256 prefix of "\n".join(texts of generate_corpus(300, 0)) under the pinned numpy.
CORPUS_FINGERPRINT = "176db960b22b8d75"

# Fictional roster: company → (sector, CEO, flagship product). Sectors give topic models signal.
COMPANIES: dict[str, tuple[str, str, str]] = {
    "Halvorsen Industrial": ("industrials", "Marta Halvorsen", "the Kestrel turbine"),
    "Brightwater Analytics": ("technology", "Daniel Okafor", "the Lumen platform"),
    "Cedar Ridge Bancorp": ("banking", "Priya Raman", "Ridge Business Checking"),
    "Northgate Energy": ("energy", "Samuel Whitlock", "the Northgate pipeline"),
    "Tamsin Retail Group": ("retail", "Elena Vasquez", "the Tamsin loyalty program"),
    "Orrin Biosciences": ("healthcare", "Hiroshi Tanaka", "the Orrin assay"),
    "Pellucid Software": ("technology", "Grace Adeyemi", "Pellucid Cloud"),
    "Marrow Freight": ("industrials", "Tomas Lindqvist", "the Marrow logistics network"),
    "Saltmarsh Capital": ("banking", "Nadia Petrova", "the Saltmarsh wealth desk"),
    "Juniper Grid": ("energy", "Owen Castillo", "the Juniper storage array"),
    "Verity Home": ("retail", "Amara Nwosu", "Verity Essentials"),
    "Calloway Health": ("healthcare", "Rachel Brennan", "the Calloway clinic network"),
}

SECTOR_NOUNS: dict[str, list[str]] = {
    "industrials": "orders backlog shipments machinery plants aerospace freight capacity components contracts".split(),
    "technology": "subscriptions software cloud bookings licenses customers platform developers seats renewals".split(),
    "banking": "deposits loans credit reserves margin lending branches mortgages portfolio capital".split(),
    "energy": "production barrels pipeline throughput drilling reserves output power generation wells".split(),
    "retail": "stores traffic inventory merchandise basket shoppers ecommerce assortment comparable promotions".split(),
    "healthcare": "patients trials clinics therapies prescriptions reimbursement enrollment devices procedures approvals".split(),
}

COMMON_NOUNS = """
    quarter year revenue demand costs operations business segment results performance market
    growth expenses cash investment pricing guidance volume earnings balance outlook strategy
    productivity efficiency headcount leverage liquidity dividends shareholders footprint
""".split()

QUARTERS = ["the first quarter", "the second quarter", "the third quarter", "the fourth quarter",
            "fiscal 2025", "fiscal 2026"]

_DOC_OPENERS: dict[DocType, str] = {
    "earnings-call": "Good morning, and welcome to the {org} earnings call.",
    "mdna": "The following discussion of {org} should be read with the consolidated statements.",
    "8-k": "{org} is filing this current report to disclose a material event.",
}


class _DocBuilder:
    """Concatenates text pieces while recording entity spans and relations."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []

    def text(self, s: str) -> None:
        self.parts.append(s)
        self.length += len(s)

    def ent(self, s: str, etype: EntityType) -> int:
        self.entities.append(Entity(self.length, self.length + len(s), etype))
        self.text(s)
        return len(self.entities) - 1

    def rel(self, head: int, rel: RelationType, tail: int) -> None:
        self.relations.append(Relation(head, rel, tail))


def _zipf_pick(rng: np.random.Generator, words: list[str], s: float = 1.1) -> str:
    weights = 1.0 / np.arange(1, len(words) + 1) ** s
    return words[rng.choice(len(words), p=weights / weights.sum())]


def _money(rng: np.random.Generator) -> str:
    return f"${rng.integers(12, 980)}.{rng.integers(0, 10)} million"


def _lex(rng: np.random.Generator, category: str, pos: str) -> str:
    words = sorted(w for w in STANDIN_LEXICON[category] if LEXICON_POS[w] == pos)  # type: ignore[index]
    return words[rng.integers(len(words))]


def _tone_sentence(rng: np.random.Generator, tone: int, nouns: list[str]) -> str:
    """One sentence expressing `tone`; polar sentences are sometimes negated (a lexicon trap)."""
    n1, n2 = _zipf_pick(rng, nouns), _zipf_pick(rng, nouns)
    if tone == 0:
        return [f"Results in {n1} were in line with the prior period. ",
                f"The {n1} trend was broadly unchanged, and {n2} held steady. "][rng.integers(2)]
    cat, opp = ("positive", "negative") if tone > 0 else ("negative", "positive")
    form = rng.integers(4)
    if form == 0:
        return f"Results in {n1} were {_lex(rng, cat, 'ADJ')}, and the {n2} trend was {_lex(rng, cat, 'ADJ')}. "
    if form == 1:
        return f"The {n1} business {_lex(rng, cat, 'VERB')} expectations this period. "
    if form == 2:
        return f"Management pointed to {_lex(rng, cat, 'NOUN')} in {n1} and {_lex(rng, cat, 'NOUN')} in {n2}. "
    # Negation: the polarity is carried by "not" + an opposite-polarity adjective.
    return f"Results in {n1} were not {_lex(rng, opp, 'ADJ')} this period. "


def _hedge_sentence(rng: np.random.Generator, nouns: list[str]) -> str:
    n1, n2 = _zipf_pick(rng, nouns), _zipf_pick(rng, nouns)
    return (f"Estimates of {_lex(rng, 'uncertainty', 'NOUN')} {_lex(rng, 'uncertainty', 'MODAL')} change, and the "
            f"{n1} outlook remains {_lex(rng, 'uncertainty', 'ADJ')} given {_lex(rng, 'uncertainty', 'NOUN')} "
            f"in {n2}. ")


def _legal_sentence(rng: np.random.Generator) -> str:
    nouns = sorted(w for w in STANDIN_LEXICON["litigious"] if LEXICON_POS[w] == "NOUN")
    a, b2, c = (nouns[i] for i in rng.choice(len(nouns), size=3, replace=False))
    return (f"The filing discloses a {a} and a related {b2}; {_lex(rng, 'litigious', 'ADJ')} review of the "
            f"{c} is ongoing. ")


# Sentence-level tone given the document tone: polar documents mix in neutral and
# opposite-polarity sentences, so document tone is a majority, not a unanimity.
_SENTENCE_TONE_MIX = {1: ([1, 0, -1], [0.6, 0.2, 0.2]), -1: ([-1, 0, 1], [0.6, 0.2, 0.2]),
                      0: ([0, 1, -1], [0.5, 0.25, 0.25])}


def _generate_doc(rng: np.random.Generator, i: int, tone: int) -> FinanceDoc:
    company = list(COMPANIES)[rng.integers(len(COMPANIES))]
    sector, ceo, product = COMPANIES[company]
    doc_type: DocType = ("earnings-call", "mdna", "8-k")[rng.integers(3)]
    nouns = SECTOR_NOUNS[sector] + COMMON_NOUNS
    b = _DocBuilder()

    opener = _DOC_OPENERS[doc_type]
    pre, post = opener.split("{org}")
    b.text(pre)
    org = b.ent(company, "ORG")
    b.text(post + " ")

    # Relation-bearing sentences: each document carries a random subset of the four relations.
    for kind in rng.permutation(["CEO_OF", "REPORTED", "ACQUIRED", "GUIDED"])[: rng.integers(2, 5)]:
        if kind == "CEO_OF":
            p = b.ent(ceo, "PERSON")
            b.text(", chief executive officer of ")
            o = b.ent(company, "ORG")
            b.text(", said the team remained focused on execution. ")
            b.rel(p, "CEO_OF", o)
        elif kind == "REPORTED":
            o = b.ent(company, "ORG")
            b.text(" reported revenue of ")
            m = b.ent(_money(rng), "MONEY")
            b.text(" for ")
            b.ent(QUARTERS[rng.integers(len(QUARTERS))], "DATE")
            b.text(". ")
            b.rel(o, "REPORTED", m)
        elif kind == "ACQUIRED":
            target = list(COMPANIES)[rng.integers(len(COMPANIES))]
            if target == company:
                continue
            o = b.ent(company, "ORG")
            b.text(" agreed to acquire ")
            t = b.ent(target, "ORG")
            b.text(" for ")
            b.ent(_money(rng), "MONEY")
            b.text(". ")
            b.rel(o, "ACQUIRED", t)
        else:
            o = b.ent(company, "ORG")
            b.text(" guided to ")
            pct = b.ent(f"{rng.integers(1, 15)} percent", "PERCENT")
            b.text(" growth in ")
            b.ent(QUARTERS[rng.integers(len(QUARTERS))], "DATE")
            b.text(". ")
            b.rel(o, "GUIDED", pct)

    # Tone-bearing sentences, hedges (all tones), and legal language (8-Ks).
    values, probs = _SENTENCE_TONE_MIX[tone]
    for _ in range(rng.integers(4, 8)):
        b.text(_tone_sentence(rng, int(rng.choice(values, p=probs)), nouns))
    if rng.random() < 0.7:
        b.text(_hedge_sentence(rng, nouns))
    if doc_type == "8-k" and rng.random() < 0.75:
        b.text(_legal_sentence(rng))
    b.text("Demand for ")
    b.ent(product, "PRODUCT")
    b.text(f" shaped the {_zipf_pick(rng, nouns)} and {_zipf_pick(rng, nouns)} outlook.")

    return FinanceDoc(
        doc_id=f"synth-{i:04d}",
        company=company,
        doc_type=doc_type,
        text="".join(b.parts),
        tone=tone,  # type: ignore[arg-type]
        entities=tuple(b.entities),
        relations=tuple(b.relations),
    )


def generate_corpus(n_docs: int = 300, seed: int = 0) -> list[FinanceDoc]:
    """A deterministic synthetic corpus; tones are exactly balanced up to n_docs mod 3."""
    rng = np.random.default_rng(seed)
    tones = rng.permutation(np.resize([-1, 0, 1], n_docs))
    return [_generate_doc(rng, i, int(t)) for i, t in enumerate(tones)]


def lexicon_score(text: str, lexicon: dict[str, frozenset[str]] = STANDIN_LEXICON) -> float:
    """Net tone: (positive hits − negative hits) / max(1, token count)."""
    toks = _tokens(text)
    pos = sum(t in lexicon["positive"] for t in toks)
    neg = sum(t in lexicon["negative"] for t in toks)
    return (pos - neg) / max(1, len(toks))


def load_edgar_excerpts() -> list[FinanceDoc]:
    """Public MD&A excerpts listed in edgar/SOURCES.md (tone 0, no gold annotations)."""
    import re

    manifest = (EDGAR_DIR / "SOURCES.md").read_text() if (EDGAR_DIR / "SOURCES.md").exists() else ""
    companies = {m.group(1): m.group(2).strip()
                 for m in re.finditer(r"^\|\s*(\S+\.txt)\s*\|\s*([^|]+)\|", manifest, re.M)}
    return [
        FinanceDoc(doc_id=f"edgar-{f.stem}", company=companies.get(f.name, ""), doc_type="mdna",
                   text=f.read_text().strip(), tone=0, entities=(), relations=())
        for f in sorted(EDGAR_DIR.glob("*.txt"))
    ]


# --------------------------------------------------------------------------------------------
# Verification harness — every claim downstream topics rely on.
# --------------------------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+", text.lower())


def test_deterministic():
    assert generate_corpus(50, seed=7) == generate_corpus(50, seed=7)


def test_seed_changes_output():
    assert generate_corpus(50, seed=7) != generate_corpus(50, seed=8)


def test_size():
    assert len(generate_corpus(300)) == 300


def test_tone_balanced():
    tones = [d.tone for d in generate_corpus(300)]
    for t in (-1, 0, 1):
        share = tones.count(t) / len(tones)
        assert abs(share - 1 / 3) <= 0.10, (t, share)


def test_doc_types_present():
    assert {d.doc_type for d in generate_corpus(300)} == {"earnings-call", "mdna", "8-k"}


def test_entity_spans_valid():
    for d in generate_corpus(300):
        assert d.entities, d.doc_id
        for e in d.entities:
            span = d.text[e.start : e.end]
            assert 0 <= e.start < e.end <= len(d.text), (d.doc_id, e)
            assert span and span.strip() == span, (d.doc_id, e, span)


def test_relations_index_entities():
    for d in generate_corpus(300):
        for r in d.relations:
            assert 0 <= r.head < len(d.entities) and 0 <= r.tail < len(d.entities), (d.doc_id, r)
            assert r.head != r.tail, (d.doc_id, r)


def test_all_relation_types_present():
    rels = {r.rel for d in generate_corpus(300) for r in d.relations}
    assert rels == {"CEO_OF", "REPORTED", "ACQUIRED", "GUIDED"}, rels


def test_zipfian():
    from collections import Counter

    counts = Counter(t for d in generate_corpus(300) for t in _tokens(d.text))
    freqs = np.array(sorted(counts.values(), reverse=True)[:200], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    slope = np.polyfit(np.log(ranks), np.log(freqs), 1)[0]
    assert len(freqs) == 200, len(freqs)
    assert -1.3 <= slope <= -0.7, slope


def test_lexicon_shape_and_disjoint():
    assert set(STANDIN_LEXICON) == {"positive", "negative", "uncertainty", "litigious"}
    for cat, words in STANDIN_LEXICON.items():
        assert 15 <= len(words) <= 30, (cat, len(words))
    assert not STANDIN_LEXICON["positive"] & STANDIN_LEXICON["negative"]


def test_lexicon_signal():
    from scipy.stats import spearmanr

    docs = generate_corpus(300)
    rho = spearmanr([lexicon_score(d.text) for d in docs], [d.tone for d in docs]).statistic
    assert rho >= 0.5, rho


def test_fast():
    import time

    t0 = time.perf_counter()
    generate_corpus(300)
    assert time.perf_counter() - t0 < 5.0


def test_edgar_manifest():
    import re

    manifest = (EDGAR_DIR / "SOURCES.md").read_text()
    files = sorted(EDGAR_DIR.glob("*.txt"))
    assert len(files) >= 3, len(files)
    for f in files:
        assert re.search(rf"^\|\s*{re.escape(f.name)}\s*\|", manifest, re.M), f.name
        assert len(f.read_text().split()) <= 400, f.name
    docs = load_edgar_excerpts()
    assert len(docs) == len(files) and all(d.doc_type == "mdna" and d.tone == 0 for d in docs)


def test_lexicon_words_have_parts_of_speech():
    assert set(LEXICON_POS) == set().union(*STANDIN_LEXICON.values())
    assert set(LEXICON_POS.values()) <= {"ADJ", "NOUN", "VERB", "MODAL", "ADV"}


def test_predicate_slots_are_grammatical():
    import re

    for d in generate_corpus(300):
        for w in re.findall(r"\b(?:were|was|remains|not) ([a-z]+)", d.text):
            assert LEXICON_POS.get(w, "ADJ") == "ADJ", (d.doc_id, w)


def test_uncertainty_appears_in_every_tone():
    docs = generate_corpus(300)
    for t in (-1, 0, 1):
        tone_docs = [d for d in docs if d.tone == t]
        mean = np.mean([sum(w in STANDIN_LEXICON["uncertainty"] for w in _tokens(d.text)) for d in tone_docs])
        assert mean >= 0.5, (t, mean)


def test_litigious_language_in_8k():
    docs = [d for d in generate_corpus(300) if d.doc_type == "8-k"]
    hits = [sum(w in STANDIN_LEXICON["litigious"] for w in _tokens(d.text)) for d in docs]
    assert np.mean([h > 0 for h in hits]) >= 0.5, np.mean([h > 0 for h in hits])


def test_tone_not_trivially_separable():
    """A three-line lexicon rule must NOT recover the gold tone (it did at 96.8%)."""
    docs = generate_corpus(300)

    def rule(text):
        toks = _tokens(text)
        if any(w in STANDIN_LEXICON["uncertainty"] for w in toks):
            return 0
        return int(np.sign(sum(w in STANDIN_LEXICON["positive"] for w in toks)
                           - sum(w in STANDIN_LEXICON["negative"] for w in toks)))

    acc = np.mean([rule(d.text) == d.tone for d in docs])
    assert acc <= 0.85, acc


def test_fingerprint():
    """Pins the corpus bytes: downstream viz mirror these numbers to the decimal."""
    import hashlib

    digest = hashlib.sha256("\n".join(d.text for d in generate_corpus(300, 0)).encode()).hexdigest()
    assert digest.startswith(CORPUS_FINGERPRINT), digest[:16]


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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
