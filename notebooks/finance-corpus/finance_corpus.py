"""Shared synthetic finance corpus for formalNLP's finance case studies.

Not a topic: the data layer every ``financeCaseStudy`` notebook and the capstone import, so the
whole site argues from ONE corpus. Provides

* ``WORLD`` / ``build_world`` — a fixed knowledge base over a fictional roster (24 public companies,
  13 subsidiaries, their officers, cities, regions, sectors). It is built from ``WORLD_SEED`` and
  does NOT depend on the corpus seed, so every corpus describes the same world. Its relations carry
  the patterns KG-embedding topics need: ``COMPETES_WITH`` is symmetric, ``SUBSIDIARY_OF`` and
  ``PARENT_OF`` are inverses, ``BASED_IN = HQ_IN ∘ LOCATED_IN`` (composition), and every
  ``ACQUIRED`` target is a subsidiary acquired exactly once (``ACQUIRED ⇒ PARENT_OF``).
* ``generate_corpus`` — seeded earnings-call, MD&A, and 8-K style documents rendered FROM that world:
  fact sentences (several paraphrases per relation), distractor co-mentions, reported and guided
  figures, aspect-level sentiment, hedges, and legal language. Gold: document tone, entity spans
  linked to the KB, relations, aspect polarities, and sentence spans. ``roster=`` restricts the
  companies (and their subsidiaries and officers) a corpus may mention, for held-out-name NER tests
  (see ``split_roster``); ``label_noise=`` flips tone labels without touching the text.
* An open vocabulary: content words come from Pitman–Yor processes (one per sector plus one shared)
  with discount ``HEAPS_DISCOUNT``, so the content vocabulary grows as ``n ** discount`` (Heaps' law)
  and held-out text has unseen words and singleton bigrams. New types beyond the seeded real words
  are pseudo-words built from English-like morphemes (``subledgers``, ``crossfleet``), so subword
  topics see real morpheme structure.
* ``STANDIN_LEXICON`` / ``lexicon_score`` — a small hand-written sentiment lexicon. It is NOT derived
  from the Loughran–McDonald Master Dictionary (non-commercial license; never committed here). Topics
  that teach the real dictionary fetch it at runtime. ``GENERAL_STANDIN_LEXICON`` adds the
  ``FINANCE_POLYSEMES`` (liability, tax, cost, crude, capital) a general-purpose lexicon reads as
  negative, the misreading Loughran and McDonald documented.
* ``load_edgar_excerpts`` — short public MD&A excerpts from SEC EDGAR, listed in ``edgar/SOURCES.md``.
  They are UNLABELED (``tone is None``), which is not the same as labeled neutral (``tone == 0``).

Annotation conventions (the gold every extraction topic evaluates against):

* Spans are ``[start, end)`` character offsets into ``text`` with no surrounding whitespace.
* PRODUCT and DATE spans include a leading "the" when the name takes one ("the Kestrel turbine",
  "the third quarter"); "Pellucid Cloud" and "fiscal 2026" have none. ORG, PERSON, GPE, and SECTOR
  spans never include "the", and a possessive "'s" is outside the span. Entity annotation is
  exhaustive for every roster name and alias, every "$… million", and every "N percent".
* ``Entity.kb_id`` links ORG/PERSON/GPE/SECTOR mentions (including aliases such as "Halvorsen") to
  the canonical ``WORLD`` name; it is None for MONEY, PERCENT, DATE, and PRODUCT.
* A sentence's relations are the facts it STATES plus their logical equivalents (the symmetric
  ``COMPETES_WITH`` twin; the ``SUBSIDIARY_OF``/``PARENT_OF`` inverse). Facts a reader could infer
  but the text does not state, such as ``BASED_IN`` from "headquartered in Toronto, Canada", are not
  annotated; the KB has them. Distractor sentences co-mention a pair without stating a relation, so
  their gold is empty even when the KB relates the pair. Distant supervision mislabels exactly these.
  ``REPORTED`` and ``GUIDED`` are document events (ORG → MONEY / PERCENT), not KB facts.
* ``Aspect`` spans cover the aspect term ("operating margin"); polarity is the evaluation for the
  company, so "costs rose" is -1 and "costs fell" is +1. Annotation is exhaustive for
  ``ASPECT_CATEGORIES`` terms; neutral mentions are polarity 0.
* ``sentences`` are gold sentence spans (numbers like "$12.5 million" defeat naive splitting).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/finance-corpus/finance_corpus.py
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

EntityType = Literal["ORG", "PERSON", "MONEY", "PERCENT", "DATE", "PRODUCT", "GPE", "SECTOR"]
KBRelation = Literal["CEO_OF", "CFO_OF", "IN_SECTOR", "COMPETES_WITH", "SUBSIDIARY_OF", "PARENT_OF",
                     "ACQUIRED", "HQ_IN", "LOCATED_IN", "BASED_IN"]
RelationType = Literal["CEO_OF", "CFO_OF", "IN_SECTOR", "COMPETES_WITH", "SUBSIDIARY_OF", "PARENT_OF",
                       "ACQUIRED", "HQ_IN", "LOCATED_IN", "BASED_IN", "REPORTED", "GUIDED"]
DocType = Literal["earnings-call", "mdna", "8-k"]
Tone = Literal[-1, 0, 1]
LexiconCategory = Literal["positive", "negative", "uncertainty", "litigious"]

EDGAR_DIR = Path(__file__).resolve().parent / "edgar"


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    type: EntityType
    kb_id: str | None = None  # canonical WORLD name; None for MONEY, PERCENT, DATE, PRODUCT


@dataclass(frozen=True)
class Relation:
    head: int  # index into FinanceDoc.entities
    rel: RelationType
    tail: int


@dataclass(frozen=True)
class Aspect:
    start: int
    end: int
    category: str  # a key of ASPECT_CATEGORIES
    polarity: Tone


@dataclass(frozen=True)
class FinanceDoc:
    doc_id: str
    company: str
    doc_type: DocType
    text: str
    tone: Tone | None  # None = unlabeled (EDGAR), distinct from 0 = labeled neutral
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    aspects: tuple[Aspect, ...] = ()
    sentences: tuple[tuple[int, int], ...] = ()
    clean_tone: Tone | None = None  # tone before label_noise; equals tone when no flip happened


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

# Neutral in finance, negative to a general-purpose lexicon (Loughran and McDonald 2011 found most
# Harvard-IV "negative" words in 10-Ks are of this kind). Inflections included.
FINANCE_POLYSEMES = frozenset("liability liabilities tax taxes cost costs crude capital".split())
GENERAL_STANDIN_LEXICON: dict[LexiconCategory, frozenset[str]] = {
    **STANDIN_LEXICON, "negative": STANDIN_LEXICON["negative"] | FINANCE_POLYSEMES,
}

# sha256 prefix of "\n".join(texts of generate_corpus(300, 0)) under the pinned numpy.
CORPUS_FINGERPRINT = "c1a05ac33d67a902"

# --------------------------------------------------------------------------------------------
# The world: a fixed knowledge base, built before any text.
# --------------------------------------------------------------------------------------------

WORLD_SEED = 0
SECTORS = ("industrials", "technology", "banking", "energy", "retail", "healthcare")

# City → region (US state or country). BASED_IN(org, region) = HQ_IN(org, city) ∘ LOCATED_IN(city, region).
CITIES: dict[str, str] = {
    "Houston": "Texas", "Dallas": "Texas", "Chicago": "Illinois", "Boston": "Massachusetts",
    "Denver": "Colorado", "Columbus": "Ohio", "Toronto": "Canada", "Calgary": "Canada",
    "Frankfurt": "Germany", "Munich": "Germany", "Zurich": "Switzerland", "Dublin": "Ireland",
    "Osaka": "Japan",
}

# Public roster: name → (alias, sector, CEO, CFO, flagship product, headquarters city).
_PUBLIC: dict[str, tuple[str, str, str, str, str, str]] = {
    "Halvorsen Industrial": ("Halvorsen", "industrials", "Marta Halvorsen", "Felix Brandt", "the Kestrel turbine", "Columbus"),
    "Marrow Freight": ("Marrow", "industrials", "Tomas Lindqvist", "Ingrid Sato", "the Marrow logistics network", "Chicago"),
    "Dunmore Aerospace": ("Dunmore", "industrials", "Colin Mercer", "Leah Okonkwo", "the Wren actuator", "Munich"),
    "Ostrander Machine Works": ("Ostrander", "industrials", "Benedikt Kraus", "Yuki Morita", "the Ostrander press line", "Frankfurt"),
    "Brightwater Analytics": ("Brightwater", "technology", "Daniel Okafor", "Sofia Lindgren", "the Lumen platform", "Boston"),
    "Pellucid Software": ("Pellucid", "technology", "Grace Adeyemi", "Martin Hollis", "Pellucid Cloud", "Dublin"),
    "Quillon Systems": ("Quillon", "technology", "Arjun Mehta", "Clara Weiss", "the Quillon ledger suite", "Toronto"),
    "Stratus Point Networks": ("Stratus Point", "technology", "Lena Fischer", "David Mensah", "the Stratus edge router", "Denver"),
    "Cedar Ridge Bancorp": ("Cedar Ridge", "banking", "Priya Raman", "Thomas Avery", "Ridge Business Checking", "Chicago"),
    "Saltmarsh Capital": ("Saltmarsh", "banking", "Nadia Petrova", "Julian Reyes", "the Saltmarsh wealth desk", "Boston"),
    "Fairhaven Trust": ("Fairhaven", "banking", "Robert Ashby", "Mei Lin Zhou", "Fairhaven Premier Savings", "Zurich"),
    "Kingsbridge Financial": ("Kingsbridge", "banking", "Aisha Rahman", "Patrick Doyle", "the Kingsbridge card", "Toronto"),
    "Northgate Energy": ("Northgate", "energy", "Samuel Whitlock", "Hannah Moreau", "the Northgate pipeline", "Houston"),
    "Juniper Grid": ("Juniper", "energy", "Owen Castillo", "Erik Johansson", "the Juniper storage array", "Denver"),
    "Redfern Petroleum": ("Redfern", "energy", "Victor Alvarez", "Naomi Clarke", "the Redfern offshore field", "Calgary"),
    "Solenne Power": ("Solenne", "energy", "Kenji Watanabe", "Isabel Duarte", "the Solenne wind farm", "Dallas"),
    "Tamsin Retail Group": ("Tamsin", "retail", "Elena Vasquez", "George Whitaker", "the Tamsin loyalty program", "Dallas"),
    "Verity Home": ("Verity", "retail", "Amara Nwosu", "Lucas Bergman", "Verity Essentials", "Chicago"),
    "Larkspur Outfitters": ("Larkspur", "retail", "Fiona Gallagher", "Mateo Rossi", "the Larkspur trail line", "Denver"),
    "Copperleaf Markets": ("Copperleaf", "retail", "Dmitri Volkov", "Ama Boateng", "Copperleaf Fresh", "Columbus"),
    "Orrin Biosciences": ("Orrin", "healthcare", "Hiroshi Tanaka", "Laura Kim", "the Orrin assay", "Boston"),
    "Calloway Health": ("Calloway", "healthcare", "Rachel Brennan", "Ahmed Haddad", "the Calloway clinic network", "Houston"),
    "Meridale Medical": ("Meridale", "healthcare", "Simone Laurent", "Noah Feldman", "the Meridale infusion pump", "Osaka"),
    "Thessaly Therapeutics": ("Thessaly", "healthcare", "Andrea Costa", "Ravi Iyer", "the Thessaly oncology program", "Frankfurt"),
}

# Subsidiary → parent. Which were acquired (and for how much, when) is drawn from WORLD_SEED.
_SUBSIDIARIES: dict[str, str] = {
    "Ashcombe Controls": "Halvorsen Industrial", "Kestrel Turbine Services": "Halvorsen Industrial",
    "Fenwick Logistics": "Marrow Freight", "Larch Data": "Brightwater Analytics",
    "Tidewell Labs": "Pellucid Software", "Brackenbury Lending": "Cedar Ridge Bancorp",
    "Moorfield Advisors": "Saltmarsh Capital", "Granite Basin Midstream": "Northgate Energy",
    "Heliocell Storage": "Juniper Grid", "Pennant Home Goods": "Verity Home",
    "Siskin Apparel": "Larkspur Outfitters", "Corvane Diagnostics": "Orrin Biosciences",
    "Westbury Clinics": "Calloway Health",
}

# Backward-compatible view: public company → (sector, CEO, flagship product).
COMPANIES: dict[str, tuple[str, str, str]] = {n: (v[1], v[2], v[4]) for n, v in _PUBLIC.items()}
PUBLIC_COMPANIES: tuple[str, ...] = tuple(_PUBLIC)

QUARTERS = ["the first quarter", "the second quarter", "the third quarter", "the fourth quarter",
            "fiscal 2025", "fiscal 2026"]

Triple = tuple[str, KBRelation, str]


@dataclass(frozen=True)
class WorldKB:
    types: dict[str, EntityType]  # canonical name → ORG, PERSON, GPE, or SECTOR
    aliases: dict[str, str]  # short name → canonical name
    triples: frozenset[Triple]
    deals: dict[tuple[str, str], tuple[str, str]]  # (acquirer, target) → (price, closing date)

    def between(self, a: str, b: str) -> set[str]:
        """KB relations r with r(a, b)."""
        return {r for h, r, t in self.triples if h == a and t == b}

    def facts(self, rel: str) -> list[tuple[str, str]]:
        return sorted((h, t) for h, r, t in self.triples if r == rel)


def _money(rng: np.random.Generator) -> str:
    return f"${rng.integers(12, 980)}.{rng.integers(0, 10)} million"


def build_world(seed: int = WORLD_SEED) -> WorldKB:
    rng = np.random.default_rng(seed)
    types: dict[str, EntityType] = {s: "SECTOR" for s in SECTORS}
    aliases: dict[str, str] = {}
    triples: set[Triple] = set()
    deals: dict[tuple[str, str], tuple[str, str]] = {}
    for city, region in CITIES.items():
        types[city] = types[region] = "GPE"
        triples.add((city, "LOCATED_IN", region))
    for name, (alias, sector, ceo, cfo, _product, hq) in _PUBLIC.items():
        types[name], types[ceo], types[cfo] = "ORG", "PERSON", "PERSON"
        aliases[alias] = name
        triples |= {(name, "IN_SECTOR", sector), (ceo, "CEO_OF", name), (cfo, "CFO_OF", name),
                    (name, "HQ_IN", hq), (name, "BASED_IN", CITIES[hq])}
    cities = list(CITIES)
    for sub, parent in _SUBSIDIARIES.items():
        hq = cities[rng.integers(len(cities))]
        types[sub] = "ORG"
        triples |= {(sub, "IN_SECTOR", _PUBLIC[parent][1]), (sub, "HQ_IN", hq), (sub, "BASED_IN", CITIES[hq]),
                    (sub, "SUBSIDIARY_OF", parent), (parent, "PARENT_OF", sub)}
        if rng.random() < 0.6:
            triples.add((parent, "ACQUIRED", sub))
            deals[(parent, sub)] = (_money(rng), QUARTERS[rng.integers(len(QUARTERS))])
    for sector in SECTORS:
        members = [n for n, v in _PUBLIC.items() if v[1] == sector]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if rng.random() < 0.6:
                    triples |= {(a, "COMPETES_WITH", b), (b, "COMPETES_WITH", a)}
        for a in members:  # nobody competes with no one
            if not any(h == a and r == "COMPETES_WITH" for h, r, _ in triples):
                b = [m for m in members if m != a][rng.integers(len(members) - 1)]
                triples |= {(a, "COMPETES_WITH", b), (b, "COMPETES_WITH", a)}
    return WorldKB(types, aliases, frozenset(triples), deals)


WORLD = build_world()

# Relation → the equivalent relation on the reversed pair; a sentence stating one states both.
_EQUIVALENT: dict[str, str] = {"COMPETES_WITH": "COMPETES_WITH", "SUBSIDIARY_OF": "PARENT_OF",
                               "PARENT_OF": "SUBSIDIARY_OF"}


def split_roster(holdout: float = 0.25, seed: int = 0) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(train, held-out) public companies, stratified by sector, for NER-generalization splits."""
    rng = np.random.default_rng(seed)
    train, held = [], []
    for sector in SECTORS:
        members = [n for n, v in _PUBLIC.items() if v[1] == sector]
        k = int(round(holdout * len(members)))
        order = rng.permutation(len(members))
        held += [members[i] for i in order[:k]]
        train += [members[i] for i in order[k:]]
    return tuple(sorted(train)), tuple(sorted(held))


def _roster_closure(roster: Iterable[str] | None) -> tuple[list[str], frozenset[str]]:
    """Public companies a corpus covers, and every ORG/PERSON it may mention (subsidiaries, officers)."""
    companies = list(PUBLIC_COMPANIES) if roster is None else [c for c in PUBLIC_COMPANIES if c in set(roster)]
    unknown = set(roster or ()) - set(PUBLIC_COMPANIES)
    if unknown or not companies:
        raise ValueError(f"roster must be a non-empty subset of PUBLIC_COMPANIES; unknown: {sorted(unknown)}")
    keep = set(companies)
    for h, r, t in WORLD.triples:
        if r == "PARENT_OF" and h in companies:
            keep.add(t)
        if r in ("CEO_OF", "CFO_OF") and t in companies:
            keep.add(h)
    return companies, frozenset(keep)


# --------------------------------------------------------------------------------------------
# Open vocabulary: Pitman–Yor content words.
# --------------------------------------------------------------------------------------------

HEAPS_DISCOUNT = 0.6  # Pitman–Yor discount d: the content vocabulary grows as n**d
PY_STRENGTH = 20.0  # concentration θ > 0
SECTOR_SHARE = 0.5  # share of content words drawn from the document's sector restaurant
COMPOUND_RATE = 0.4  # share of content slots filled by a two-noun compound
COVERAGE_SKEW = 0.8  # Zipf exponent of how often each roster company is the subject of a document

# Seeded real words: each restaurant's first new types, before pseudo-words. Aspect terms and the
# polysemes are kept out so their counts stay under template control.
SECTOR_NOUNS: dict[str, list[str]] = {
    "industrials": "orders backlog shipments machinery plants aerospace freight capacity components contracts".split(),
    "technology": "subscriptions software cloud bookings licenses customers platform developers seats renewals".split(),
    "banking": "deposits loans credit reserves fees lending branches mortgages portfolio underwriting".split(),
    "energy": "production barrels pipeline throughput drilling reserves output power generation wells".split(),
    "retail": "stores traffic inventory merchandise basket shoppers ecommerce assortment markdowns promotions".split(),
    "healthcare": "patients trials clinics therapies prescriptions reimbursement enrollment devices procedures approvals".split(),
}
COMMON_NOUNS = """
    operations markets segments performance growth expenses cash investments pricing earnings services
    productivity efficiency headcount leverage dividends channels partnerships commodities logistics
""".split()

_PREFIXES = ("re", "sub", "co", "inter", "multi", "micro", "cross", "over", "under", "non", "pre", "post",
             "trans", "mid", "out")
_ROOTS = """margin ledger freight tariff yield asset credit vendor contract bond fund lease stock trade
    channel network market port rate price grid fleet plant store clinic trial license seat loan deposit
    well barrel pipe cargo route brand shelf patent device claim premium policy broker index share
    equity coupon hedge swap quota cycle merchant dealer supplier tender levy rebate royalty accrual
    payable receivable warrant escrow""".split()
_SUFFIXES = ("ing", "ings", "age", "ment", "er", "ers", "ware", "line", "lines", "book", "books", "work",
             "works", "point", "side", "flow", "flows", "base")


class _PseudoWords:
    """Fresh, never-repeated English-like nouns of two morphemes, prefix + root (+ plural "s") or
    root + suffix; three-morpheme words appear only once those crowd, so rarer words run longer."""

    def __init__(self, reserved: Iterable[str]) -> None:
        self.used = set(reserved)

    def __call__(self, rng: np.random.Generator) -> str:
        for attempt in range(1000):
            root = _ROOTS[rng.integers(len(_ROOTS))]
            prefix = _PREFIXES[rng.integers(len(_PREFIXES))] if attempt > 50 or rng.random() < 0.5 else ""
            if prefix and attempt <= 50:
                stem, suffix = prefix + root, ("s", "")[rng.integers(2)]
            else:
                stem, suffix = prefix + root, _SUFFIXES[rng.integers(len(_SUFFIXES))]
            if stem.endswith("e") and suffix[:1] in ("a", "e", "i"):  # lease + ing → leasing
                stem = stem[:-1]
            w = stem + suffix
            if w not in self.used and len(w) > 4:
                self.used.add(w)
                return w
        raise RuntimeError("pseudo-word space exhausted")


class _PitmanYor:
    """Chinese-restaurant sampler for PY(d, θ). Each new table draws a fresh type (seeded real words
    first, then pseudo-words), so tables = types and E[K_n] + θ/d ∝ Γ(θ+n+d)/Γ(θ+n) ≈ (θ+n)**d."""

    def __init__(self, discount: float, strength: float, seeds: list[str], fresh: _PseudoWords) -> None:
        if not (0 <= discount < 1 and strength > 0):
            raise ValueError("need 0 <= discount < 1 and strength > 0")
        self.d, self.theta, self.fresh = discount, strength, fresh
        self.seeds = list(reversed(seeds))
        self.types: list[str] = []
        self.counts = np.zeros(64)
        self.n = 0

    def draw(self, rng: np.random.Generator) -> str:
        k_types = len(self.types)
        u = rng.random() * (self.theta + self.n)
        if u < self.theta + self.d * k_types:  # new table: P = (θ + dK) / (θ + n)
            w = self.seeds.pop() if self.seeds else self.fresh(rng)
            if k_types == len(self.counts):
                self.counts = np.concatenate([self.counts, np.zeros(k_types)])
            self.types.append(w)
            self.counts[k_types] = 1
        else:  # existing table k: P = (c_k − d) / (θ + n)
            cum = np.cumsum(self.counts[:k_types] - self.d)
            k = min(int(np.searchsorted(cum, u - self.theta - self.d * k_types, side="right")), k_types - 1)
            self.counts[k] += 1
            w = self.types[k]
        self.n += 1
        return w


# --------------------------------------------------------------------------------------------
# Sentiment: aspects with valence, so direction words carry polarity a lexicon cannot see.
# --------------------------------------------------------------------------------------------

# category → (valence, terms). Valence +1: up is good for the company; -1: up is bad ("costs rose").
ASPECT_CATEGORIES: dict[str, tuple[int, tuple[str, ...]]] = {
    "revenue": (1, ("revenue", "net sales", "sales")),
    "margin": (1, ("gross margin", "operating margin", "margins")),
    "demand": (1, ("demand", "order intake", "volume")),
    "liquidity": (1, ("liquidity", "free cash flow", "capital levels")),
    "costs": (-1, ("costs", "operating costs", "cost of revenue")),
    "tax": (-1, ("tax expense", "taxes", "income taxes")),
    "liabilities": (-1, ("lease liabilities", "pension liabilities", "liabilities")),
}
# Every term the gold annotates (template-only extras included), for exhaustiveness checks.
ASPECT_TERMS: frozenset[str] = frozenset(
    {t for _, terms in ASPECT_CATEGORIES.values() for t in terms} | {"lease liability"})
_UP, _DOWN = ("rose", "increased", "climbed"), ("fell", "declined", "decreased")


class _DocBuilder:
    """Concatenates text pieces while recording entity spans, relations, aspects, and sentences."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.entities: list[Entity] = []
        self.relations: list[Relation] = []
        self.aspects: list[Aspect] = []
        self.sentences: list[tuple[int, int]] = []
        self.mentioned: set[str] = set()
        self._sent_start = 0

    def text(self, s: str) -> None:
        self.parts.append(s)
        self.length += len(s)

    def ent(self, s: str, etype: EntityType, kb_id: str | None = None) -> int:
        self.entities.append(Entity(self.length, self.length + len(s), etype, kb_id))
        self.text(s)
        return len(self.entities) - 1

    def rel(self, head: int, rel: RelationType, tail: int) -> None:
        self.relations.append(Relation(head, rel, tail))

    def aspect(self, s: str, category: str, polarity: int) -> None:
        self.aspects.append(Aspect(self.length, self.length + len(s), category, polarity))  # type: ignore[arg-type]
        self.text(s)

    def end_sentence(self) -> None:
        self.sentences.append((self._sent_start, self.length))
        self.text(" ")
        self._sent_start = self.length

    def joined(self) -> str:
        return "".join(self.parts).rstrip()


def _lex(rng: np.random.Generator, category: str, pos: str) -> str:
    words = sorted(w for w in STANDIN_LEXICON[category] if LEXICON_POS[w] == pos)  # type: ignore[index]
    return words[rng.integers(len(words))]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


# Fact templates. {h}/{t} are the fact's head and tail; {region} adds a stated LOCATED_IN;
# {n} is a content word; {money}/{date} come from the KB deal for ACQUIRED.
_FACT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "CEO_OF": ("{h}, chief executive officer of {t}, said the team remained focused on {n} and {n}.",
               "{t} chief executive {h} opened the call with a review of {n}.",
               "Speaking for {t}, chief executive {h} discussed priorities in {n}."),
    "CFO_OF": ("{h}, chief financial officer of {t}, reviewed {n} and {n}.",
               "{t} finance chief {h} walked through results in {n}.",
               "{h}, who serves as chief financial officer at {t}, took questions on {n}."),
    "IN_SECTOR": ("{h} operates in the {t} sector, chiefly in {n}.",
                  "Within {t}, {h} remains a mid-sized participant.",
                  "{h} is classified under {t}."),
    "COMPETES_WITH": ("{h} competes with {t} for {n} business.",
                      "{h} and {t} remain direct competitors.",
                      "Management named {t} as the main rival of {h}."),
    "SUBSIDIARY_OF": ("{h}, a subsidiary of {t}, contributed to {n} results.",
                      "{t} owns {h} outright.",
                      "{t}, the parent company of {h}, consolidates its {n}."),
    "ACQUIRED": ("{h} agreed to acquire {t} for {money}.",
                 "{h} completed its acquisition of {t} in {date}.",
                 "The purchase of {t} by {h} closed for {money}."),
    "HQ_IN": ("{h} is headquartered in {t}.",
              "{h} is headquartered in {t}, {region}.",
              "From its head office in {t}, {region}, {h} manages {n}.",
              "{h}, based in {t}, expanded its {n} footprint."),
    "BASED_IN": ("{h} is based in {t}.",
                 "{h} keeps its corporate base in {t}.",
                 "{h} has its roots in {t}."),
}
# Distractors co-mention a pair without stating any relation: (head type, tail type) → templates.
_DISTRACTOR_TEMPLATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("ORG", "ORG"): ("{a} and {b} both presented at an industry conference.",
                     "Analysts who cover {a} also follow {b}.",
                     "Questions about {b} came up during the {a} discussion."),
    ("PERSON", "ORG"): ("{a} has spoken publicly about {b}.",
                        "{a} attended an investor event hosted by {b}."),
    ("ORG", "GPE"): ("{a} hosted investors in {b}.",
                     "{a} held its annual meeting in {b}."),
    ("ORG", "SECTOR"): ("{a} commented on conditions across {b}.",),
}

_DOC_OPENERS: dict[DocType, tuple[str, ...]] = {
    "earnings-call": ("Good morning, and welcome to the {org} earnings call.",
                      "Thank you for joining the {org} quarterly results call."),
    "mdna": ("The following discussion of {org} should be read with the consolidated statements.",
             "This section reviews the financial condition of {org} for the period."),
    "8-k": ("{org} is filing this current report to disclose a material event.",
            "On this date {org} furnished a current report under the securities rules."),
}
_COMMENTS = (", which management described as {adj}", ", a result the company called {adj}",
             ", which analysts viewed as {adj}", ", a trend executives characterized as {adj}")
# Neutral, content-dense sentences: earnings prose is mostly nouns, and the open vocabulary lives here.
_FOCUS = ("Management highlighted {n}, {n}, and {n} as areas of focus.",
          "The team continued to invest in {n}, {n}, and {n}.",
          "Activity in {n} and {n} tracked the broader {n} cycle.",
          "Priorities for the year include {n}, {n}, {n}, and {n}.",
          "The plan reallocates resources from {n} and {n} toward {n} and {n}.",
          "Customers added {n} alongside {n} and {n}.",
          "Results reflected {n}, {n}, {n}, and {n}.")
_CLOSERS = ("{demand} for {product} shaped the {n} and {n} outlook.",
            "Management expects {demand} for {product} to support {n} and {n}.",
            "{demand} for {product} remained a priority across {n} and {n}.")

# Aspect-level polarity given the document tone: polar documents mix in neutral and
# opposite-polarity aspects, so document tone is a majority, not a unanimity.
_SENTENCE_TONE_MIX = {1: ([1, 0, -1], [0.65, 0.2, 0.15]), -1: ([-1, 0, 1], [0.65, 0.2, 0.15]),
                      0: ([0, 1, -1], [0.6, 0.2, 0.2])}
CONTRAST_RATE = 0.2  # share of polar aspect sentences that pair two aspects of opposite polarity
DIRECTION_COMMENT_RATE = 0.75  # share of "rose/fell" clauses that add an evaluative comment


class _CorpusGen:
    """One corpus: a seeded RNG, the Pitman–Yor restaurants it shares across documents, and the
    slice of WORLD its roster may mention."""

    def __init__(self, seed: int, roster: Iterable[str] | None, discount: float) -> None:
        self.rng = np.random.default_rng(seed)
        self.companies, self.keep = _roster_closure(roster)
        # Coverage is skewed, as in real filings and calls: the k-th roster company is covered ∝ k**-s.
        weights = 1.0 / np.arange(1, len(self.companies) + 1) ** COVERAGE_SKEW
        self.coverage = weights / weights.sum()
        self.facts = sorted(t for t in WORLD.triples if all(
            WORLD.types[x] not in ("ORG", "PERSON") or x in self.keep for x in (t[0], t[2])))
        reserved = (set(LEXICON_POS) | FINANCE_POLYSEMES | set(SECTORS) | set(COMMON_NOUNS)
                    | {w for ws in SECTOR_NOUNS.values() for w in ws}
                    | {w for term in ASPECT_TERMS for w in term.split()})
        fresh = _PseudoWords(reserved)
        self.common = _PitmanYor(discount, PY_STRENGTH, COMMON_NOUNS, fresh)
        self.by_sector = {s: _PitmanYor(discount, PY_STRENGTH, SECTOR_NOUNS[s], fresh) for s in SECTORS}

    # -- primitives -------------------------------------------------------------------------
    def word(self, sector: str) -> str:
        py = self.by_sector[sector] if self.rng.random() < SECTOR_SHARE else self.common
        return py.draw(self.rng)

    def phrase(self, sector: str) -> str:
        """A content noun, or a noun–noun compound of two independent draws ("freight contracts")."""
        first = self.word(sector)
        if self.rng.random() < COMPOUND_RATE:
            second = self.word(sector)
            return f"{first} {second}" if second != first else first
        return first

    def pick(self, seq):
        return seq[self.rng.integers(len(seq))]

    def mention(self, b: _DocBuilder, name: str) -> int:
        """A KB entity mention; repeat ORG mentions sometimes use the alias."""
        alias = next((a for a, n in WORLD.aliases.items() if n == name), None)
        surface = alias if alias and name in b.mentioned and self.rng.random() < 0.3 else name
        b.mentioned.add(name)
        return b.ent(surface, WORLD.types[name], name)

    def render(self, b: _DocBuilder, template: str, slots: dict[str, Callable[[], int | None]]) -> dict[str, int]:
        import re

        idx: dict[str, int] = {}
        for piece in re.split(r"(\{\w+\})", template):
            if piece.startswith("{") and piece.endswith("}"):
                got = slots[piece[1:-1]]()
                if got is not None:
                    idx[piece[1:-1]] = got
            elif piece:
                b.text(piece)
        b.end_sentence()
        return idx

    # -- sentence kinds ---------------------------------------------------------------------
    def fact_sentence(self, b: _DocBuilder, fact: Triple, sector: str) -> None:
        h, rel, t = fact
        template = self.pick(_FACT_TEMPLATES[rel])
        money, date = WORLD.deals.get((h, t), (None, None))
        idx = self.render(b, template, {
            "h": lambda: self.mention(b, h), "t": lambda: self.mention(b, t),
            "region": lambda: self.mention(b, CITIES[t]),
            "n": lambda: b.text(self.phrase(sector)),
            "money": lambda: b.ent(money, "MONEY"), "date": lambda: b.ent(date, "DATE"),
        })
        b.rel(idx["h"], rel, idx["t"])  # type: ignore[arg-type]
        if rel in _EQUIVALENT:
            b.rel(idx["t"], _EQUIVALENT[rel], idx["h"])  # type: ignore[arg-type]
        if "region" in idx:
            b.rel(idx["t"], "LOCATED_IN", idx["region"])

    def distractor_sentence(self, b: _DocBuilder, company: str) -> None:
        """Co-mention a pair, KB-related 60% of the time, without stating a relation."""
        kinds = list(_DISTRACTOR_TEMPLATES)
        a_type, b_type = kinds[self.rng.integers(len(kinds))]
        if a_type == "PERSON":
            pool_a = sorted(x for x in self.keep if WORLD.types[x] == "PERSON")
            related = [p for p in pool_a if WORLD.between(p, company)]
            unrelated = [p for p in pool_a if not WORLD.between(p, company)]
            use = related if self.rng.random() < 0.6 or not unrelated else unrelated
            pair = (self.pick(use), company)
        else:
            pool_b = sorted(x for x in (self.keep if b_type == "ORG" else WORLD.types)
                            if WORLD.types[x] == b_type and x != company)
            related = [x for x in pool_b if WORLD.between(company, x)]
            unrelated = [x for x in pool_b if not WORLD.between(company, x)]
            use = related if (self.rng.random() < 0.6 and related) or not unrelated else unrelated
            pair = (company, self.pick(use))
        self.render(b, self.pick(_DISTRACTOR_TEMPLATES[(a_type, b_type)]),
                    {"a": lambda: self.mention(b, pair[0]), "b": lambda: self.mention(b, pair[1])})

    def place(self, sector: str) -> str:
        """The content-word complement of an aspect clause: one noun, sometimes two."""
        if self.rng.random() < 0.5:
            return f"{self.phrase(sector)} and {self.phrase(sector)}"
        return self.phrase(sector)

    def aspect_clause(self, b: _DocBuilder, category: str, pol: int, sector: str, sentence_start: bool) -> None:
        rng = self.rng
        valence, terms = ASPECT_CATEGORIES[category]
        term = self.pick(terms)
        be = "were" if term.endswith("s") else "was"
        surface = _cap(term) if sentence_start else term
        if pol == 0:
            b.aspect(surface, category, 0)
            form = rng.integers(3)
            if form == 0:
                b.text(f" {be} in line with the prior period")
            elif form == 1:
                b.text(f" {be} broadly unchanged in {self.place(sector)}")
            else:
                b.text(f" held steady across {self.word(sector)} and {self.word(sector)}")
            return
        signed = pol * valence  # direction of the move / polarity of the evaluative word
        forms, probs = (["dir", "adj", "neg", "verb"], [0.35, 0.40, 0.10, 0.15]) if valence > 0 else (
            ["dir", "verb"], [0.8, 0.2])
        form = forms[rng.choice(len(forms), p=probs)]
        b.aspect(surface, category, pol)
        if form == "dir":  # "rose"/"fell" carry no listed polarity; only the comment, if any, does
            b.text(f" {self.pick(_UP if signed > 0 else _DOWN)} ")
            b.ent(f"{rng.integers(1, 15)} percent", "PERCENT")
            b.text(f" in {self.place(sector)}")
            if rng.random() < DIRECTION_COMMENT_RATE:
                adj = _lex(rng, "positive" if pol > 0 else "negative", "ADJ")
                b.text(self.pick(_COMMENTS).format(adj=adj))
        elif form == "adj":
            b.text(f" in {self.place(sector)} {be} {_lex(rng, 'positive' if pol > 0 else 'negative', 'ADJ')}")
        elif form == "neg":  # polarity carried by "not" + an opposite-polarity adjective
            b.text(f" {be} not {_lex(rng, 'negative' if pol > 0 else 'positive', 'ADJ')} this period")
        else:  # for a cost-like aspect "exceeded expectations" is bad news: a valence trap
            b.text(f" in {self.place(sector)} {_lex(rng, 'positive' if signed > 0 else 'negative', 'VERB')} "
                   "expectations")

    def aspect_sentence(self, b: _DocBuilder, pol: int, sector: str) -> None:
        rng = self.rng
        cats = list(ASPECT_CATEGORIES)
        weights = np.array([1 if ASPECT_CATEGORIES[c][0] > 0 else 0.6 for c in cats])
        first = cats[rng.choice(len(cats), p=weights / weights.sum())]
        if pol != 0 and rng.random() < CONTRAST_RATE:
            second = self.pick([c for c in cats if c != first])
            self.aspect_clause(b, first, pol, sector, True)
            b.text(", but ")
            self.aspect_clause(b, second, -pol, sector, False)
        elif pol != 0 and ASPECT_CATEGORIES[first][0] > 0 and rng.random() < 0.15:
            b.text("Management pointed to ")
            b.text(_lex(rng, "positive" if pol > 0 else "negative", "NOUN") + " in ")
            term = self.pick(ASPECT_CATEGORIES[first][1])
            b.aspect(term, first, pol)
        else:
            self.aspect_clause(b, first, pol, sector, True)
        b.text(".")
        b.end_sentence()

    def focus_sentence(self, b: _DocBuilder, sector: str) -> None:
        used: set[str] = set()

        def fresh_noun() -> None:  # list items stay distinct ("orders, orders, and orders" is not English)
            w = self.phrase(sector)
            for _ in range(20):
                if w not in used:
                    break
                w = self.phrase(sector)
            used.add(w)
            b.text(w)

        self.render(b, self.pick(_FOCUS), {"n": fresh_noun})

    def polyseme_sentence(self, b: _DocBuilder, sector: str) -> None:
        """Neutral, finance-sense uses of words a general lexicon reads as negative."""
        rng = self.rng
        options = ["tax", "lease", "capex", "cor"] + (["crude"] * 3 if sector == "energy" else [])
        kind = self.pick(options)
        if kind == "tax":
            b.aspect("Tax expense", "tax", 0)
            b.text(" was ")
            b.ent(_money(rng), "MONEY")
            b.text(" for ")
            b.ent(self.pick(QUARTERS), "DATE")
        elif kind == "lease":
            b.text("The ")
            b.aspect("lease liability", "liabilities", 0)
            b.text(" stood at ")
            b.ent(_money(rng), "MONEY")
        elif kind == "capex":
            b.text("Capital expenditures totaled ")
            b.ent(_money(rng), "MONEY")
            b.text(f" across {self.word(sector)}")
        elif kind == "cor":
            b.aspect("Cost of revenue", "costs", 0)
            b.text(" was ")
            b.ent(_money(rng), "MONEY")
        else:
            b.text(f"Crude production averaged {rng.integers(40, 900)} thousand barrels per day")
        b.text(".")
        b.end_sentence()

    def hedge_sentence(self, b: _DocBuilder, sector: str) -> None:
        rng = self.rng
        risk_nouns = sorted(w for w in STANDIN_LEXICON["uncertainty"] if LEXICON_POS[w] == "NOUN" and w != "estimates")
        b.text(f"Estimates of {self.pick(risk_nouns)} {_lex(rng, 'uncertainty', 'MODAL')} change, "
               f"and the {self.word(sector)} outlook remains {_lex(rng, 'uncertainty', 'ADJ')} given "
               f"{_lex(rng, 'uncertainty', 'NOUN')} in {self.word(sector)}.")
        b.end_sentence()

    def legal_sentence(self, b: _DocBuilder) -> None:
        rng = self.rng
        nouns = sorted(w for w in STANDIN_LEXICON["litigious"] if LEXICON_POS[w] == "NOUN")
        x, y, z = (nouns[i] for i in rng.choice(len(nouns), size=3, replace=False))
        b.text(f"The filing discloses a {x} and a related {y}; {_lex(rng, 'litigious', 'ADJ')} review of the "
               f"{z} is ongoing.")
        b.end_sentence()

    # -- documents --------------------------------------------------------------------------
    def fact_pool(self, company: str) -> list[Triple]:
        """Facts a document about `company` may state: its own, and its subsidiaries' place and sector."""
        subs = {t for h, r, t in self.facts if h == company and r == "PARENT_OF"}
        pool = []
        for h, r, t in self.facts:
            if r in ("PARENT_OF", "LOCATED_IN"):  # stated via SUBSIDIARY_OF / an HQ appositive
                continue
            if r == "COMPETES_WITH" and h != company:  # one orientation of the symmetric pair
                continue
            if company in (h, t) or (h in subs and r in ("HQ_IN", "BASED_IN", "IN_SECTOR")):
                pool.append((h, r, t))
        return pool

    def doc(self, i: int, tone: int) -> FinanceDoc:
        rng = self.rng
        company = self.companies[rng.choice(len(self.companies), p=self.coverage)]
        _alias, sector, _ceo, _cfo, product, _hq = _PUBLIC[company]
        doc_type: DocType = ("earnings-call", "mdna", "8-k")[rng.integers(3)]
        b = _DocBuilder()

        pre, post = self.pick(_DOC_OPENERS[doc_type]).split("{org}")
        b.text(pre)
        self.mention(b, company)
        b.text(post)
        b.end_sentence()

        pool = self.fact_pool(company)
        facts = [pool[j] for j in rng.choice(len(pool), size=min(len(pool), int(rng.integers(2, 5))), replace=False)]
        middle: list[Callable[[], None]] = [lambda f=f: self.fact_sentence(b, f, sector) for f in facts]
        middle += [lambda: self.distractor_sentence(b, company)] * int(rng.integers(1, 3))
        if rng.random() < 0.6:
            middle.append(lambda: self.reported_sentence(b, company))
        if rng.random() < 0.6:
            middle.append(lambda: self.guided_sentence(b, company))
        if rng.random() < 0.8:
            middle.append(lambda: self.polyseme_sentence(b, sector))
        middle += [lambda: self.focus_sentence(b, sector)] * int(rng.integers(3, 7))
        for j in rng.permutation(len(middle)):
            middle[j]()

        values, probs = _SENTENCE_TONE_MIX[tone]
        for _ in range(rng.integers(5, 9)):
            self.aspect_sentence(b, int(rng.choice(values, p=probs)), sector)
        if rng.random() < 0.7:
            self.hedge_sentence(b, sector)
        if doc_type == "8-k" and rng.random() < 0.75:
            self.legal_sentence(b)
        closer = self.pick(_CLOSERS)
        self.render(b, closer, {
            "demand": lambda: b.aspect("Demand" if closer.startswith("{demand}") else "demand", "demand", 0),
            "product": lambda: b.ent(product, "PRODUCT"), "n": lambda: b.text(self.word(sector))})

        return FinanceDoc(
            doc_id=f"synth-{i:04d}", company=company, doc_type=doc_type, text=b.joined(),
            tone=tone, entities=tuple(b.entities), relations=tuple(b.relations),  # type: ignore[arg-type]
            aspects=tuple(b.aspects), sentences=tuple(b.sentences), clean_tone=tone,  # type: ignore[arg-type]
        )

    def reported_sentence(self, b: _DocBuilder, company: str) -> None:
        o = self.mention(b, company)
        b.text(" reported ")
        b.aspect("revenue", "revenue", 0)
        b.text(" of ")
        m = b.ent(_money(self.rng), "MONEY")
        b.text(" for ")
        b.ent(self.pick(QUARTERS), "DATE")
        b.text(".")
        b.end_sentence()
        b.rel(o, "REPORTED", m)

    def guided_sentence(self, b: _DocBuilder, company: str) -> None:
        o = self.mention(b, company)
        b.text(" guided to ")
        pct = b.ent(f"{self.rng.integers(1, 15)} percent", "PERCENT")
        b.text(" growth in ")
        b.ent(self.pick(QUARTERS), "DATE")
        b.text(".")
        b.end_sentence()
        b.rel(o, "GUIDED", pct)


def generate_corpus(n_docs: int = 300, seed: int = 0, *, roster: Iterable[str] | None = None,
                    discount: float = HEAPS_DISCOUNT, label_noise: float = 0.0) -> list[FinanceDoc]:
    """A deterministic synthetic corpus; tones are exactly balanced up to n_docs mod 3.

    roster: public companies the corpus may cover (default all); their subsidiaries and officers
        come with them, and nothing outside that closure is ever mentioned.
    discount: Pitman–Yor discount of the content vocabulary (its Heaps exponent).
    label_noise: probability each tone label is replaced by one of the other two tones. Drawn from a
        separate stream, so the text is identical for every value; ``clean_tone`` keeps the truth.
    """
    gen = _CorpusGen(seed, roster, discount)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], n_docs))
    docs = [gen.doc(i, int(t)) for i, t in enumerate(tones)]
    if label_noise > 0:
        noise = np.random.default_rng([seed, 1])
        flipped = []
        for d in docs:
            flip, shift = noise.random() < label_noise, int(noise.integers(1, 3))
            tone = (d.tone + 1 + shift) % 3 - 1 if flip else d.tone  # type: ignore[operator]
            flipped.append(replace(d, tone=tone))
        docs = flipped
    return docs


def lexicon_score(text: str, lexicon: dict[str, frozenset[str]] = STANDIN_LEXICON) -> float:
    """Net tone: (positive hits − negative hits) / max(1, token count)."""
    toks = _tokens(text)
    pos = sum(t in lexicon["positive"] for t in toks)
    neg = sum(t in lexicon["negative"] for t in toks)
    return (pos - neg) / max(1, len(toks))


def load_edgar_excerpts() -> list[FinanceDoc]:
    """Public MD&A excerpts listed in edgar/SOURCES.md (unlabeled: tone None, no gold annotations)."""
    import re

    manifest = (EDGAR_DIR / "SOURCES.md").read_text() if (EDGAR_DIR / "SOURCES.md").exists() else ""
    companies = {m.group(1): m.group(2).strip()
                 for m in re.finditer(r"^\|\s*(\S+\.txt)\s*\|\s*([^|]+)\|", manifest, re.M)}
    return [
        FinanceDoc(doc_id=f"edgar-{f.stem}", company=companies.get(f.name, ""), doc_type="mdna",
                   text=f.read_text().strip(), tone=None, entities=(), relations=())
        for f in sorted(EDGAR_DIR.glob("*.txt"))
    ]


def heaps_exponent(tokens: list[str], start: int = 1000) -> float:
    """OLS slope of log V(n) on log n over geometrically spaced n ≥ start (Heaps' β)."""
    seen: set[str] = set()
    growth = np.empty(len(tokens))
    for k, tok in enumerate(tokens):
        seen.add(tok)
        growth[k] = len(seen)
    ns = np.unique(np.geomspace(start, len(tokens), 40).astype(int))
    return float(np.polyfit(np.log(ns), np.log(growth[ns - 1]), 1)[0])


# --------------------------------------------------------------------------------------------
# Verification harness — every claim downstream topics rely on.
# --------------------------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+", text.lower())


@lru_cache(maxsize=1)
def _default_corpus() -> tuple[FinanceDoc, ...]:
    return tuple(generate_corpus(300, 0))


def _span(d: FinanceDoc, x) -> str:
    return d.text[x.start : x.end]


def test_deterministic():
    assert generate_corpus(50, seed=7) == generate_corpus(50, seed=7)


def test_seed_changes_output():
    assert generate_corpus(50, seed=7) != generate_corpus(50, seed=8)


def test_size():
    assert len(generate_corpus(300)) == 300


def test_tone_balanced():
    tones = [d.tone for d in _default_corpus()]
    for t in (-1, 0, 1):
        share = tones.count(t) / len(tones)
        assert abs(share - 1 / 3) <= 0.10, (t, share)


def test_doc_types_present():
    assert {d.doc_type for d in _default_corpus()} == {"earnings-call", "mdna", "8-k"}


def test_entity_spans_valid():
    for d in _default_corpus():
        assert d.entities, d.doc_id
        for e in d.entities:
            span = d.text[e.start : e.end]
            assert 0 <= e.start < e.end <= len(d.text), (d.doc_id, e)
            assert span and span.strip() == span, (d.doc_id, e, span)
            assert (e.kb_id is None) == (e.type in ("MONEY", "PERCENT", "DATE", "PRODUCT")), (d.doc_id, e)
            if e.kb_id:
                assert span in (e.kb_id, *[a for a, n in WORLD.aliases.items() if n == e.kb_id]), (d.doc_id, span)


def test_relations_index_entities():
    for d in _default_corpus():
        for r in d.relations:
            assert 0 <= r.head < len(d.entities) and 0 <= r.tail < len(d.entities), (d.doc_id, r)
            assert r.head != r.tail, (d.doc_id, r)


def test_all_relation_types_present():
    # Updated deliberately in v2: the world KB adds eight relation types to the original four.
    rels = {r.rel for d in _default_corpus() for r in d.relations}
    assert rels == {"CEO_OF", "CFO_OF", "IN_SECTOR", "COMPETES_WITH", "SUBSIDIARY_OF", "PARENT_OF",
                    "ACQUIRED", "HQ_IN", "LOCATED_IN", "BASED_IN", "REPORTED", "GUIDED"}, rels


def zipf_slope(tokens: list[str], max_rank: int = 1000) -> float:
    """OLS slope of log frequency on log rank at 40 log-spaced ranks in [1, max_rank]."""
    from collections import Counter

    freqs = np.array(sorted(Counter(tokens).values(), reverse=True), dtype=float)
    ranks = np.unique(np.geomspace(1, min(max_rank, len(freqs)), 40).astype(int))
    return float(np.polyfit(np.log(ranks), np.log(freqs[ranks - 1]), 1)[0])


def test_zipfian():
    """Updated deliberately in v2. The old test fit ranks 1–200 densely, which covered nearly all of the
    closed ~300-type vocabulary. With an open vocabulary, the top 200 is the closed-class template head
    (slope ≈ −0.7); the law is a claim about the whole range, so fit log-spaced ranks 1–1000, giving
    each decade equal weight instead of letting ranks 100–200 dominate."""
    tokens = [t for d in _default_corpus() for t in _tokens(d.text)]
    assert len(set(tokens)) >= 1000, len(set(tokens))
    slope = zipf_slope(tokens)
    assert -1.3 <= slope <= -0.8, slope


def test_pitman_yor_sampler_heaps_exponent():
    """E[K_n] + θ/d ∝ (θ+n)**d exactly, so the slope of log(K_n + θ/d) on log(n + θ) recovers d."""
    for d in (0.4, 0.6, 0.8):
        rng = np.random.default_rng(1)
        py = _PitmanYor(d, PY_STRENGTH, [], _PseudoWords(()))
        draws = [py.draw(rng) for _ in range(20000)]
        ns = np.unique(np.geomspace(1000, len(draws), 25).astype(int))
        k = np.array([len(set(draws[:n])) for n in ns])
        beta = np.polyfit(np.log(ns + PY_STRENGTH), np.log(k + PY_STRENGTH / d), 1)[0]
        assert abs(beta - d) <= 0.03, (d, beta)


def test_corpus_vocabulary_grows():
    """Full text: closed-class template words saturate, so β sits below d but well above 0."""
    beta = heaps_exponent([t for d in _default_corpus() for t in _tokens(d.text)])
    assert 0.3 <= beta <= HEAPS_DISCOUNT, beta


def test_discount_controls_growth():
    def n_types(d):
        return len({t for doc in generate_corpus(150, 0, discount=d) for t in _tokens(doc.text)})

    assert n_types(0.3) < n_types(0.6) < n_types(0.8)


def test_singleton_bigrams():
    """Good–Turing and Kneser–Ney need a real mass of once-seen events."""
    from collections import Counter

    bigrams = Counter(pair for d in _default_corpus() for s, e in d.sentences
                      for pair in zip(_tokens(d.text[s:e]), _tokens(d.text[s:e])[1:]))
    share = sum(c == 1 for c in bigrams.values()) / len(bigrams)
    assert share >= 0.3, share


def test_heldout_has_oov_words():
    docs = list(_default_corpus())
    order = np.random.default_rng(0).permutation(len(docs))
    train = {t for i in order[:240] for t in _tokens(docs[i].text)}
    held = [t for i in order[240:] for t in _tokens(docs[i].text)]
    oov = sum(t not in train for t in held) / len(held)
    assert 0.005 <= oov <= 0.10, oov


def test_kb_relation_patterns():
    kb = set(WORLD.triples)
    for h, r, t in kb:
        assert h != t, (h, r, t)
        if r == "COMPETES_WITH":
            assert (t, r, h) in kb  # symmetric
        if r == "SUBSIDIARY_OF":
            assert (t, "PARENT_OF", h) in kb  # inverse
        if r == "PARENT_OF":
            assert (t, "SUBSIDIARY_OF", h) in kb
        if r == "ACQUIRED":
            assert (h, "PARENT_OF", t) in kb and (h, t) in WORLD.deals
    composed = {(o, reg) for o, c in WORLD.facts("HQ_IN") for c2, reg in WORLD.facts("LOCATED_IN") if c == c2}
    assert composed == set(WORLD.facts("BASED_IN"))  # BASED_IN = HQ_IN ∘ LOCATED_IN
    targets = [t for _, t in WORLD.facts("ACQUIRED")]
    assert len(targets) == len(set(targets)) and set(targets) <= set(_SUBSIDIARIES)
    for org in (n for n, ty in WORLD.types.items() if ty == "ORG"):  # functional relations
        assert [h for h, _ in WORLD.facts("HQ_IN")].count(org) == 1, org
        assert [h for h, _ in WORLD.facts("IN_SECTOR")].count(org) == 1, org
    for c in PUBLIC_COMPANIES:
        assert [h for h, t in WORLD.facts("CEO_OF") if t == c] and [h for h, t in WORLD.facts("CFO_OF") if t == c]
        assert any(h == c for h, _ in WORLD.facts("COMPETES_WITH")), c


def test_world_is_independent_of_corpus_seed():
    assert build_world() == WORLD and all(d.entities for d in generate_corpus(5, seed=99))


def test_gold_relations_are_kb_facts_closed_under_equivalence():
    for d in _default_corpus():
        gold = {(r.head, r.rel, r.tail) for r in d.relations}
        for h, rel, t in gold:
            if rel in ("REPORTED", "GUIDED"):
                assert d.entities[t].type == ("MONEY" if rel == "REPORTED" else "PERCENT"), d.doc_id
                continue
            triple = (d.entities[h].kb_id, rel, d.entities[t].kb_id)
            assert triple in WORLD.triples, (d.doc_id, triple)
            if rel in _EQUIVALENT:
                assert (t, _EQUIVALENT[rel], h) in gold, (d.doc_id, triple)


def _sentence_of(d: FinanceDoc, pos: int) -> int:
    return next(k for k, (s, e) in enumerate(d.sentences) if s <= pos < e)


def distant_supervision_precision(docs: Iterable[FinanceDoc]) -> float:
    """Share of (sentence, ordered KB-entity pair, KB relation) labels that the sentence states."""
    labeled = correct = 0
    for d in docs:
        gold = {(r.head, r.rel, r.tail) for r in d.relations}
        linked = [(k, e) for k, e in enumerate(d.entities) if e.kb_id]
        for i, a in linked:
            for j, b in linked:
                if i != j and _sentence_of(d, a.start) == _sentence_of(d, b.start):
                    for rel in WORLD.between(a.kb_id, b.kb_id):  # type: ignore[arg-type]
                        labeled += 1
                        correct += (i, rel, j) in gold
    return correct / labeled


def test_distractors_make_distant_supervision_noisy():
    p = distant_supervision_precision(_default_corpus())
    assert 0.55 <= p <= 0.9, p


def test_kb_coverage():
    """Most KB facts are stated somewhere, so relation extraction can recover the world."""
    stated = {(d.entities[r.head].kb_id, r.rel, d.entities[r.tail].kb_id)
              for d in _default_corpus() for r in d.relations}
    cover = len(stated & WORLD.triples) / len(WORLD.triples)
    assert cover >= 0.85, cover


def test_roster_holdout():
    train, held = split_roster()
    assert len(held) == len(SECTORS) and not set(train) & set(held)
    _, held_closure = _roster_closure(held)
    held_names = held_closure | {a for a, n in WORLD.aliases.items() if n in held_closure}
    for d in generate_corpus(200, 3, roster=train):
        assert d.company in train
        for e in d.entities:
            assert e.kb_id not in held_closure, (d.doc_id, e.kb_id)
        for name in held_names:
            assert name not in d.text, (d.doc_id, name)
    assert {d.company for d in generate_corpus(60, 3, roster=held)} == set(held)


def test_names_and_numbers_exhaustively_annotated():
    import re

    names = [n for n, ty in WORLD.types.items() if ty in ("ORG", "PERSON", "GPE")] + list(WORLD.aliases)
    pattern = re.compile(r"\$\d+\.\d million|\b\d+ percent\b|" + "|".join(
        rf"\b{re.escape(n)}\b" for n in sorted(names, key=len, reverse=True)))
    for d in _default_corpus():
        for m in pattern.finditer(d.text):
            assert any(e.start <= m.start() and m.end() <= e.end for e in d.entities), (d.doc_id, m.group())


def test_sentence_spans():
    for d in _default_corpus():
        assert d.sentences[0][0] == 0 and d.sentences[-1][1] == len(d.text), d.doc_id
        for (s, e), (s2, _) in zip(d.sentences, d.sentences[1:]):
            assert d.text[e:s2] == " " and d.text[e - 1] == ".", (d.doc_id, d.text[s:e])
        for x in (*d.entities, *d.aspects):
            assert _sentence_of(d, x.start) == _sentence_of(d, x.end - 1), (d.doc_id, x)


def test_aspects_valid_and_exhaustive():
    import re

    pattern = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in sorted(ASPECT_TERMS, key=len, reverse=True)),
                         re.I)
    for d in _default_corpus():
        assert d.aspects, d.doc_id
        for a in d.aspects:
            assert a.category in ASPECT_CATEGORIES and a.polarity in (-1, 0, 1), (d.doc_id, a)
            assert _span(d, a).lower() in ASPECT_TERMS, (d.doc_id, _span(d, a))
        for m in pattern.finditer(d.text):
            assert any(a.start <= m.start() and m.end() <= a.end for a in d.aspects), (d.doc_id, m.group())


def test_contrastive_sentences():
    """Sentence-level labels cannot express "revenue rose, but costs climbed"; aspect gold can."""
    mixed = total = 0
    for d in _default_corpus():
        for s, e in d.sentences:
            pols = {a.polarity for a in d.aspects if s <= a.start < e}
            total += bool(pols - {0})
            mixed += {-1, 1} <= pols
    assert mixed / total >= 0.08, mixed / total


def test_tone_agrees_with_aspect_majority():
    docs = _default_corpus()
    agree = np.mean([np.sign(sum(a.polarity for a in d.aspects)) == d.tone for d in docs])
    assert 0.6 <= agree <= 0.95, agree


def test_lexicon_shape_and_disjoint():
    assert set(STANDIN_LEXICON) == {"positive", "negative", "uncertainty", "litigious"}
    for cat, words in STANDIN_LEXICON.items():
        assert 15 <= len(words) <= 30, (cat, len(words))
    assert not STANDIN_LEXICON["positive"] & STANDIN_LEXICON["negative"]
    assert not FINANCE_POLYSEMES & set().union(*STANDIN_LEXICON.values())


def test_lexicon_signal():
    from scipy.stats import spearmanr

    docs = _default_corpus()
    rho = spearmanr([lexicon_score(d.text) for d in docs], [d.tone for d in docs]).statistic
    assert rho >= 0.5, rho


def test_polysemes_mislead_a_general_lexicon():
    from scipy.stats import spearmanr

    docs = _default_corpus()
    tones = [d.tone for d in docs]
    rho_fin = spearmanr([lexicon_score(d.text) for d in docs], tones).statistic
    rho_gen = spearmanr([lexicon_score(d.text, GENERAL_STANDIN_LEXICON) for d in docs], tones).statistic
    assert rho_fin - rho_gen >= 0.05, (rho_fin, rho_gen)
    for t in (-1, 0, 1):
        hits = [sum(w in FINANCE_POLYSEMES for w in _tokens(d.text)) for d in docs if d.tone == t]
        assert np.mean(hits) >= 0.5, (t, np.mean(hits))


def test_label_noise():
    clean, noisy = generate_corpus(300, 0), generate_corpus(300, 0, label_noise=0.2)
    assert [d.text for d in clean] == [d.text for d in noisy]
    assert all(d.tone == d.clean_tone for d in clean)
    assert all(n.clean_tone == c.tone and n.tone in (-1, 0, 1) for c, n in zip(clean, noisy))
    rate = np.mean([n.tone != n.clean_tone for n in noisy])
    assert abs(rate - 0.2) <= 0.07, rate


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
    # Updated deliberately in v2: unlabeled excerpts carry tone None, not a fake neutral 0.
    assert len(docs) == len(files) and all(d.doc_type == "mdna" and d.tone is None for d in docs)
    assert all(d.tone is not None for d in _default_corpus())


def test_lexicon_words_have_parts_of_speech():
    assert set(LEXICON_POS) == set().union(*STANDIN_LEXICON.values())
    assert set(LEXICON_POS.values()) <= {"ADJ", "NOUN", "VERB", "MODAL", "ADV"}


def test_predicate_slots_are_grammatical():
    import re

    for d in _default_corpus():
        for w in re.findall(r"\b(?:were|was|remains|not) ([a-z]+)", d.text):
            assert LEXICON_POS.get(w, "ADJ") == "ADJ", (d.doc_id, w)


def test_uncertainty_appears_in_every_tone():
    docs = _default_corpus()
    for t in (-1, 0, 1):
        tone_docs = [d for d in docs if d.tone == t]
        mean = np.mean([sum(w in STANDIN_LEXICON["uncertainty"] for w in _tokens(d.text)) for d in tone_docs])
        assert mean >= 0.5, (t, mean)


def test_litigious_language_in_8k():
    docs = [d for d in _default_corpus() if d.doc_type == "8-k"]
    hits = [sum(w in STANDIN_LEXICON["litigious"] for w in _tokens(d.text)) for d in docs]
    assert np.mean([h > 0 for h in hits]) >= 0.5, np.mean([h > 0 for h in hits])


def lexicon_rule(text: str) -> int:
    """The three-line baseline: any uncertainty word → neutral, else the sign of net polarity."""
    toks = _tokens(text)
    if any(w in STANDIN_LEXICON["uncertainty"] for w in toks):
        return 0
    return int(np.sign(sum(w in STANDIN_LEXICON["positive"] for w in toks)
                       - sum(w in STANDIN_LEXICON["negative"] for w in toks)))


def test_tone_not_trivially_separable():
    """A three-line lexicon rule must NOT recover the gold tone (it did at 96.8%)."""
    acc = np.mean([lexicon_rule(d.text) == d.tone for d in _default_corpus()])
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
