"""Perplexity, cross-entropy, and compression — the reference implementation for formalNLP's fourth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. Cross-entropy and the log score: H(P, Q) = H(P) + KL(P ‖ Q) (formalML's decomposition, cited), so the
   log score is strictly proper; perplexity is the geometric mean of 1/Q, and a model can do worse than
   the uniform distribution.
2. Perplexity as a branching factor: the AEP — the typical set holds almost all the probability and has
   about e^{nH} members, so e^H counts equally likely choices per token.
3. From held-out text to cross-entropy: per-token held-out log-loss is a ratio estimator over i.i.d.
   sentences, consistent for E[NLL(S)]/E[|S|], with a delta-method standard error; Shannon–McMillan–
   Breiman (cited) for stationary ergodic sources; every model's cross-entropy rate bounds the entropy rate.
4. Compression is prediction: McMillan — uniquely decodable codes obey Kraft; the Shannon code under the
   wrong model costs KL extra; an arithmetic code of a whole text under Q is within 2 bits of −log₂ Q(text),
   run here exactly, with big integers, on the held-out corpus.
5. Choosing the unit: a string's total log-likelihood does not depend on how it is tokenized, so bits per
   character compares models across tokenizers and per-token perplexity does not; mapping words to <unk>
   can only lower the reported loss.
6. Finance case study: on EDGAR MD&A text the trigram models do worse than uniform per token, the <unk>
   convention hides most of their cost, and lzma beats every static model.

Cross-checks: entropy and KL against ``scipy.stats.entropy``; unique decodability by Sardinas–Patterson,
exhaustively over small codes; the arithmetic decoder reproduces every coded text; zlib, bz2, and lzma as
reference coders.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/perplexity-cross-entropy/perplexity_cross_entropy.py
"""

from __future__ import annotations

import bz2
import dataclasses
import heapq
import itertools
import lzma
import math
import sys
import zlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import special, stats

_HERE = Path(__file__).resolve().parent
for _dep in ("finance-corpus", "zipf-heaps-vocabulary-statistics", "subword-tokenization-bpe-unigram",
             "n-gram-models-smoothing"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import n_gram_models_smoothing as ng  # noqa: E402
import subword_tokenization_bpe_unigram as sw  # noqa: E402
import zipf_heaps_vocabulary_statistics as zhv  # noqa: E402

# ONE stream per random-instance claim below, spawned from one seed (STARTER-PROMPT A1).
SEED = 20261023

BOS, EOS, UNK = ng.BOS, ng.EOS, ng.UNK
ORDER = ng.ORDER  # the word models are Topic 3's trigram models
LN2 = math.log(2)
TESTS = ("held", "edgar")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
NGram = tuple[str, ...]


# --------------------------------------------------------------------------------------------
# 1. Cross-entropy and the log score.
# --------------------------------------------------------------------------------------------


def cross_entropy(p, q) -> float:
    """H(p, q) = −Σ p log q in nats; ∞ when q is 0 where p is not."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    if np.any(q[m] <= 0):
        return math.inf
    return float(-np.sum(p[m] * np.log(q[m])))


def kl(p, q) -> float:
    """KL(p ‖ q) = Σ p log(p/q) in nats, computed directly (not as H(p, q) − H(p))."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    if np.any(q[m] <= 0):
        return math.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def perplexity(nats_per_unit: float) -> float:
    """exp of a log-loss in nats per unit (= 2 ** bits per unit)."""
    return math.exp(nats_per_unit)


def n_chars(sents) -> int:
    """Characters of the normalized text " ".join(words) + "\\n" per sentence: each word's letters plus one
    separator after it (a space, or the sentence end after the last word)."""
    return sum(sum(map(len, s)) + len(s) for s in sents)


def n_word_tokens(sents) -> int:
    """Topic 3's predicted tokens: every word and each sentence's </s>."""
    return sum(len(s) + 1 for s in sents)


def text_of(sents) -> str:
    return "".join(" ".join(s) + "\n" for s in sents)


def uniform_log_loss() -> float:
    """The uniform distribution over Topic 3's vocabulary V: log |V| nats on every token of any text."""
    return math.log(len(ng.vocabulary()))


# --------------------------------------------------------------------------------------------
# Models as full distributions. Coding needs P(· | u) over the whole alphabet, not one probability,
# so these evaluate the prerequisite's recursions for every symbol at once (asserted equal to the scalar
# ng.interpolated_kn_prob and ng.additive_prob in test_vector_models_match_scalar).
# --------------------------------------------------------------------------------------------

FALLBACK_D = 0.5  # a discount for orders whose count-of-counts lack n₁ or n₂ (Ney's estimate is undefined)


def kn_fit_any(sents, order: int, alphabet) -> ng.KNModel:
    """ng.kn_fit with Ney's per-order discount n₁/(n₁ + 2n₂) where it is defined and FALLBACK_D where it is
    not (character models' lower orders can lack singletons), and a uniform base over `alphabet`."""
    raw = ng.ngram_counts(sents, order)
    used = {order: raw[order], **{k: ng.continuation_counts(raw[k + 1]) for k in range(1, order)}}
    ds = {}
    for k, c in used.items():
        N = ng.count_of_counts(c)
        ds[k] = (ng.ney_discount(N),) if N.get(1) and N.get(2) else (FALLBACK_D,)
    return dataclasses.replace(ng.kn_fit(sents, order, discounts=ds), vocab_size=len(alphabet))


class KNVector:
    """P(· | u) of an ng.KNModel as a vector over a sorted alphabet — ng.interpolated_kn_prob's recursion,
    evaluated for every symbol at once. Cached for small alphabets; for the word alphabet only the unigram
    level is cached (one vector per context would be ~27 kB × thousands of contexts)."""

    def __init__(self, model: ng.KNModel, alphabet):
        assert model.vocab_size == len(alphabet)
        self.model, self.alphabet = model, tuple(alphabet)
        self.index = {a: i for i, a in enumerate(self.alphabet)}
        rows: dict[NGram, tuple[list, list]] = defaultdict(lambda: ([], []))
        for k, cnt in model.counts.items():
            for g, c in cnt.items():
                if c > 0:
                    ix, kept = rows[g[:-1]]
                    ix.append(self.index[g[-1]])
                    kept.append(max(c - model.discount(k, c), 0.0))
        self._rows = {u: (np.array(ix), np.array(kept)) for u, (ix, kept) in rows.items()}
        self._cache: dict[NGram, np.ndarray] = {}
        self._depth = model.order if len(self.alphabet) <= 64 else 1

    def __call__(self, u) -> np.ndarray:
        u = tuple(u)
        hit = self._cache.get(u)
        if hit is not None:
            return hit
        k = len(u) + 1
        lower = np.full(len(self.alphabet), 1.0 / len(self.alphabet)) if k == 1 else self(u[1:])
        total = self.model.totals.get(u, 0)
        if total:
            out = self.model.freed[u] * lower
            if u in self._rows:
                ix, kept = self._rows[u]
                out[ix] += kept
            out /= total
        else:
            out = lower
        if k <= self._depth:
            out.flags.writeable = False
            self._cache[u] = out
        return out


class LaplaceVector:
    """ng.additive_prob (add-one on the trigram counts) as a vector over the word alphabet."""

    def __init__(self):
        self.alphabet = word_alphabet()
        self.index = {a: i for i, a in enumerate(self.alphabet)}
        rows: dict[NGram, tuple[list, list]] = defaultdict(lambda: ([], []))
        for g, c in ng.train_counts()[ORDER].items():
            ix, cs = rows[g[:-1]]
            ix.append(self.index[g[-1]])
            cs.append(c)
        self._rows = {u: (np.array(ix), np.array(cs, float)) for u, (ix, cs) in rows.items()}
        self._tot = ng.train_totals()[ORDER]

    def __call__(self, u) -> np.ndarray:
        u = tuple(u)
        out = np.ones(len(self.alphabet))
        if u in self._rows:
            ix, cs = self._rows[u]
            out[ix] += cs
        return out / (self._tot.get(u, 0) + len(self.alphabet))


@lru_cache(maxsize=1)
def word_alphabet() -> tuple[str, ...]:
    """Topic 3's V, sorted. "</s>" and "<unk>" sort before every lowercase word."""
    return tuple(sorted(ng.vocabulary()))


SPELL_ORDER = 3  # the speller is a character trigram Kneser–Ney model
SPELL_ALPHABET = tuple(LETTERS) + (EOS,)  # EOS ends a spelling


def spell_corpus(k: int) -> tuple[NGram, ...]:
    """The speller's training data for a vocabulary that sends every word with training count ≤ k to <unk>:
    the spellings of the training TOKENS of words with count ≤ max(k, 1). At k = 0 no training word is
    unknown, so the hapaxes stand in for the unseen (the Good–Turing logic of Topic 3)."""
    c1 = ng.train_counts()[1]
    cut = max(k, 1)
    return tuple(tuple(g[0]) for g, n in sorted(c1.items()) if g[0] != EOS and n <= cut for _ in range(n))


@lru_cache(maxsize=None)
def speller(k: float = 1, order: int = SPELL_ORDER) -> KNVector:
    return KNVector(kn_fit_any(spell_corpus(k), order, SPELL_ALPHABET), SPELL_ALPHABET)


@lru_cache(maxsize=None)
def spell_nll(word: str, k: float = 1, order: int = SPELL_ORDER) -> float:
    """−log S(word): its letters and the end mark under the speller (scalar, ng.interpolated_kn_prob)."""
    m = speller(k, order).model
    return -sum(math.log(ng.interpolated_kn_prob(g[-1], g[:-1], m)) for g in ng.events([tuple(word)], order))


@lru_cache(maxsize=None)
def word_vector(name: str):
    """The closed-vocabulary word model by name: Laplace, interpolated KN, or modified KN (Topic 3's fits)."""
    if name == "laplace":
        return LaplaceVector()
    return KNVector(ng.kn_models()[name], word_alphabet())


class OpenVocabModel:
    """A word model made into a code for ANY text: an in-vocabulary word is one symbol; an out-of-vocabulary
    word is <unk> followed by its letters and an end mark, spelled by `spell`. The word history sees <unk>,
    as in Topic 3. Both ends of the arithmetic code run this same state machine. The code is slightly
    deficient — the speller can also spell in-vocabulary words — which wastes code space but never fails."""

    def __init__(self, words, spell: KNVector, vocab):
        self.words, self.spell, self.vocab = words, spell, frozenset(vocab)

    def start(self):
        return ("w", (BOS,) * (ORDER - 1))

    def probs(self, state):
        if state[0] == "w":
            return self.words.alphabet, self.words.index, self.words(state[1])
        return self.spell.alphabet, self.spell.index, self.spell(state[2])

    def step(self, state, sym: str):
        if state[0] == "w":
            if sym == EOS:
                return self.start()
            u = state[1][1:] + (sym,)
            return ("c", u, (BOS,) * (SPELL_ORDER - 1)) if sym == UNK else ("w", u)
        _, u, cctx = state
        return ("w", u) if sym == EOS else ("c", u, cctx[1:] + (sym,))

    def symbols(self, sents) -> list[str]:
        out: list[str] = []
        for s in sents:
            for w in s:
                out += [w] if w in self.vocab else [UNK, *w, EOS]
            out.append(EOS)
        return out


def sents_from_symbols(syms) -> list[NGram]:
    """Invert OpenVocabModel.symbols."""
    sents, cur, i = [], [], 0
    while i < len(syms):
        s = syms[i]
        if s == EOS:
            sents.append(tuple(cur))
            cur = []
        elif s == UNK:
            j = syms.index(EOS, i + 1)
            cur.append("".join(syms[i + 1 : j]))
            i = j
        else:
            cur.append(s)
        i += 1
    return sents


@lru_cache(maxsize=None)
def open_model(name: str) -> OpenVocabModel:
    return OpenVocabModel(word_vector(name), speller(1), ng.vocabulary())


# --------------------------------------------------------------------------------------------
# 2. Perplexity as a branching factor: the AEP, exactly, by type classes.
# --------------------------------------------------------------------------------------------


def compositions3(n: int) -> np.ndarray:
    """Every (n₁, n₂, n₃) with n₁ + n₂ + n₃ = n — the type classes of length-n sequences over 3 symbols."""
    a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = a + b <= n
    return np.stack([a[keep], b[keep], n - a[keep] - b[keep]], axis=1)


def typical_set(p, n: int, eps: float, q=None) -> dict[str, float]:
    """The set E = {x ∈ Σⁿ : |−(1/n) log q(x) − H(p, q)| ≤ eps} for i.i.d. p on 3 symbols (q = p: the typical
    set A). Exact, by summing over type classes: log|E|, P_p(E), the target H(p, q), and Chebyshev's bound
    1 − Var_p(−log q(X))/(n eps²) on P_p(E)."""
    p = np.asarray(p, float)
    q = p if q is None else np.asarray(q, float)
    comp = compositions3(n)
    log_mult = special.gammaln(n + 1) - special.gammaln(comp + 1).sum(1)
    rate = -(comp @ np.log(q)) / n
    target = cross_entropy(p, q)
    inside = np.abs(rate - target) <= eps
    var = float(p @ np.log(q) ** 2 - (p @ np.log(q)) ** 2)
    if not inside.any():
        return {"log_size": -math.inf, "prob": 0.0, "target": target, "chebyshev": 1 - var / (n * eps**2)}
    return {"log_size": float(special.logsumexp(log_mult[inside])),
            "prob": float(np.exp(special.logsumexp(log_mult[inside] + comp[inside] @ np.log(p)))),
            "target": target, "chebyshev": 1 - var / (n * eps**2)}


TYPICAL_EXAMPLE = ((0.6, 0.3, 0.1), 300, 0.1)  # p, n, ε for the page's worked example


def typical_example() -> dict[str, float]:
    p, n, eps = TYPICAL_EXAMPLE
    t = typical_set(p, n, eps)
    H = zhv.entropy(np.array(p))
    return {"H": H, "branching": math.exp(H), "n": n, "eps": eps, "log_size": t["log_size"], "prob": t["prob"],
            "share_of_sequences": math.exp(t["log_size"] - n * math.log(3))}


# --------------------------------------------------------------------------------------------
# 3. From held-out text to cross-entropy.
# --------------------------------------------------------------------------------------------


def ratio_estimate(y, n) -> tuple[float, float]:
    """Σy/Σn and its delta-method standard error with the sentences as the i.i.d. units (Thm 2):
    se = sqrt(Σ_j (y_j − r̂ n_j)² / (m(m − 1))) / n̄."""
    y, n = np.asarray(y, float), np.asarray(n, float)
    m, r = len(y), y.sum() / n.sum()
    return float(r), float(math.sqrt(np.sum((y - r * n) ** 2) / (m * (m - 1))) / n.mean())


@lru_cache(maxsize=None)
def sentence_nll(name: str, test: str = "held") -> np.ndarray:
    """Per held-out sentence: [closed-vocabulary NLL (nats, Topic 3's events), word tokens, characters, open
    NLL (closed + the spelling of each OOV word)]. name ∈ {laplace, ikn, mkn}."""
    f, V = ng.prob_fn(name), ng.vocabulary()
    rows = []
    for s in ng.corpus()[test]:
        closed = -sum(math.log(f(g[-1], g[:-1])) for g in ng.events(ng.with_unk([s]), ORDER))
        spelled = sum(spell_nll(w) for w in s if w not in V)
        rows.append((closed, len(s) + 1, sum(map(len, s)) + len(s), closed + spelled))
    return np.array(rows)


def token_level_se(name: str, test: str = "held") -> float:
    """The naive standard error that treats every token as an independent draw."""
    f = ng.prob_fn(name)
    x = np.array([-math.log(f(g[-1], g[:-1])) for g in ng.heldout_events(test)])
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def random_sentence_source(rng: np.random.Generator, m: int = 5, stop: float = 0.2) -> np.ndarray:
    """A bigram sentence source on m words: row u ∈ {<s>, w₁, …, w_m} holds P(· | u) over (w₁, …, w_m, </s>),
    with P(</s> | u) ≥ stop so sentences end."""
    P = (1 - stop) * rng.dirichlet(np.ones(m + 1), size=m + 1)
    P[:, -1] += stop
    return P


def exact_ratio(P: np.ndarray, Q: np.ndarray) -> tuple[float, float]:
    """E[NLL_Q(S)] and E[|S|] for S ~ P, exactly: the expected cost and length accumulated before absorption
    in </s>, solved from (I − T) v = c with T the word-to-word transitions (context of w_j is row j + 1)."""
    m = P.shape[0] - 1
    T = np.zeros((m + 1, m + 1))
    T[:, 1:] = P[:, :m]
    A = np.eye(m + 1) - T
    return float(np.linalg.solve(A, (P * -np.log(Q)).sum(1))[0]), float(np.linalg.solve(A, np.ones(m + 1))[0])


def sample_sentences(P: np.ndarray, n: int, rng: np.random.Generator):
    """n i.i.d. sentences from the bigram source P, all simulated in parallel. Returns flat arrays
    (sentence id, context row, symbol) — one entry per predicted token, </s> included."""
    m = P.shape[0] - 1
    cum = np.cumsum(P, axis=1)
    ctx, alive = np.zeros(n, int), np.ones(n, bool)
    ids, ctxs, syms = [], [], []
    while alive.any():
        idx = np.flatnonzero(alive)
        sym = np.minimum((rng.random(idx.size)[:, None] > cum[ctx[idx]]).sum(1), m)
        ids.append(idx), ctxs.append(ctx[idx].copy()), syms.append(sym)
        ended = sym == m
        ctx[idx[~ended]] = sym[~ended] + 1
        alive[idx[ended]] = False
    return np.concatenate(ids), np.concatenate(ctxs), np.concatenate(syms)


def sentence_ratio(P: np.ndarray, Q: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    ids, ctx, sym = sample_sentences(P, n, rng)
    y = np.bincount(ids, weights=-np.log(Q[ctx, sym]), minlength=n)
    return ratio_estimate(y, np.bincount(ids, minlength=n))


def random_chain(rng: np.random.Generator, m: int = 4) -> np.ndarray:
    return rng.dirichlet(np.ones(m), size=m)


def stationary(P: np.ndarray) -> np.ndarray:
    m = P.shape[0]
    A = np.vstack([P.T - np.eye(m), np.ones(m)])
    return np.linalg.lstsq(A, np.r_[np.zeros(m), 1.0], rcond=None)[0]


def entropy_rate(P: np.ndarray) -> float:
    """Σ_u π_u H(P(· | u)) — formalML shannon-entropy, Prop 7."""
    return float(stationary(P) @ (-(P * np.log(P)).sum(1)))


def cross_entropy_rate(P: np.ndarray, Q: np.ndarray) -> float:
    """Σ_u π_u Σ_w P(w | u)(−log Q(w | u)) — the limit of −(1/n) log Q(Xⁿ) for X ~ P (the ergodic theorem)."""
    return float(stationary(P) @ (-(P * np.log(Q)).sum(1)))


def sample_path(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """One path of length n of the chain P, started from its stationary law."""
    cum = [list(np.cumsum(r)) for r in P]
    u = rng.random(n)
    x = np.empty(n, int)
    x[0] = min(bisect_right(list(np.cumsum(stationary(P))), u[0]), len(P) - 1)
    for t in range(1, n):
        x[t] = min(bisect_right(cum[x[t - 1]], u[t]), len(P) - 1)
    return x


def batch_means_se(x: np.ndarray, batches: int = 100) -> float:
    b = x[: len(x) // batches * batches].reshape(batches, -1).mean(1)
    return float(b.std(ddof=1) / math.sqrt(batches))


# --------------------------------------------------------------------------------------------
# 4. Compression is prediction: Kraft–McMillan, Shannon codes, arithmetic coding.
# --------------------------------------------------------------------------------------------


def kraft_sum(lengths) -> Fraction:
    return sum((Fraction(1, 2**int(n)) for n in lengths), Fraction(0))


def is_prefix_free(code) -> bool:
    words = sorted(code)
    return len(set(words)) == len(words) and not any(b.startswith(a) for a, b in zip(words, words[1:]))


def canonical_code(lengths) -> list[str] | None:
    """Kraft's converse, constructively: sort the lengths, give codeword i the first ℓᵢ bits of
    Σ_{j < i} 2^{−ℓⱼ} — the left end of the next free dyadic interval. None if the lengths violate Kraft."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: list[str | None] = [None] * len(lengths)
    acc = Fraction(0)
    for i in order:
        n = int(lengths[i])
        v = acc * 2**n
        if v.denominator != 1 or v >= 2**n:
            return None
        out[i] = format(int(v), f"0{n}b") if n else ""
        acc += Fraction(1, 2**n)
    return out


def is_uniquely_decodable(code) -> bool:
    """Sardinas–Patterson: a code is uniquely decodable iff no set of dangling suffixes contains a codeword.
    S₁ = {w : a w = b, a ≠ b codewords}; S_{i+1} = {w : a w ∈ Sᵢ} ∪ {w : s w = a, s ∈ Sᵢ}."""
    C = set(code)
    if len(C) < len(code) or "" in C:
        return False

    def dangling(A, B):
        return {b[len(a):] for a in A for b in B if len(b) > len(a) and b.startswith(a)}

    S, seen = dangling(C, C), set()
    while S:
        if S & C:
            return False
        key = frozenset(S)
        if key in seen:
            return True
        seen.add(key)
        S = dangling(C, S) | dangling(S, C)
    return True


def ambiguous_within(code, max_len: int = 10) -> bool:
    """Brute force, independent of Sardinas–Patterson: does some bit-string of length ≤ max_len have two
    parsings into codewords? count[n][s] = number of parsings of the n-bit string with value s."""
    count = [np.ones(1, np.int64)]
    for n in range(1, max_len + 1):
        s, c = np.arange(2**n), np.zeros(2**n, np.int64)
        for w in code:
            k = len(w)
            if k <= n:
                hit = (s & (2**k - 1)) == int(w, 2)
                c[hit] += count[n - k][s[hit] >> k]
        if np.any(c > 1):
            return True
        count.append(c)
    return False


@lru_cache(maxsize=1)
def mcmillan_census() -> dict[str, int]:
    """Every binary code of 3 or 4 distinct codewords of length ≤ 3: how many are uniquely decodable
    (Sardinas–Patterson), how many of those are not prefix-free, and how many break Kraft."""
    words = [format(i, f"0{n}b") for n in (1, 2, 3) for i in range(2**n)]
    out = {"codes": 0, "ud": 0, "ud_not_prefix": 0, "kraft_violations": 0, "ud_kraft_violations": 0,
           "brute_force_disagreements": 0}
    for r in (3, 4):
        for code in itertools.combinations(words, r):
            ud, over = is_uniquely_decodable(code), kraft_sum(map(len, code)) > 1
            out["codes"] += 1
            out["ud"] += ud
            out["ud_not_prefix"] += ud and not is_prefix_free(code)
            out["kraft_violations"] += over
            out["ud_kraft_violations"] += ud and over
            out["brute_force_disagreements"] += ud == ambiguous_within(code)
    return out


def huffman_lengths(p) -> list[int]:
    heap = [(float(x), i, [i]) for i, x in enumerate(p)]
    heapq.heapify(heap)
    lengths = [0] * len(p)
    tie = len(p)
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap)
        b, _, sb = heapq.heappop(heap)
        for i in sa + sb:
            lengths[i] += 1
        heapq.heappush(heap, (a + b, tie, sa + sb))
        tie += 1
    return lengths


def shannon_lengths(q) -> list[int]:
    """⌈−log₂ q(x)⌉: the Shannon code for the model q."""
    return [math.ceil(-math.log2(x)) for x in q]


CODE_BITS = 40  # integer frequencies f = ⌊q·(2^40 − |A|)⌋ + 1: every symbol keeps a nonzero interval


def frequencies(q: np.ndarray) -> np.ndarray:
    """The integer model the coder actually uses: f ≥ 1 with f/Σf ≈ q (quantization costs ~1e-8 bits/symbol)."""
    return np.floor(q * float((1 << CODE_BITS) - len(q))).astype(np.int64) + 1


@dataclass(frozen=True)
class Code:
    bits: int  # the codeword, read as the binary fraction bits / 2^length
    length: int
    n_symbols: int
    model_bits: float  # −log₂ Q(x) under the floating-point model
    quantized_bits: float  # −log₂ Q̃(x) under the integer frequencies actually coded
    shannon_bits: int  # Σ ⌈−log₂ Q(xᵢ | x<ᵢ)⌉: coding symbol by symbol with Shannon codes instead
    width: int  # the final interval [A/D, (A + W)/D) has width W/D = Q̃(x)
    denom: int

    def bitstring(self) -> str:
        return format(self.bits, f"0{self.length}b") if self.length else ""


def shortest_dyadic(A: int, W: int, D: int) -> tuple[int, int]:
    """The shortest codeword: least L with an integer b such that [b/2^L, (b+1)/2^L) ⊆ [A/D, (A+W)/D). If a
    length-L dyadic interval fits, its left half fits at L + 1, so the first L that fits is the least."""
    L = max(0, D.bit_length() - W.bit_length() - 1)
    while True:
        b = -((-A << L) // D)  # ⌈A·2^L / D⌉: the leftmost length-L grid point at or after A/D
        if (b + 1) * D <= (A + W) << L:
            return b, L
        L += 1


def arithmetic_encode(model, symbols) -> Code:
    """Elias's arithmetic code with exact big-integer arithmetic (Thm 4). The interval starts as [0, 1); each
    symbol s keeps the slice [cum(s), cum(s) + f(s)) / T of the current interval, so after the whole message
    its width is Q̃(x) = Π f/T. The codeword names the shortest dyadic interval inside it.
    In exact rationals the interval is [A/D, (A + W)/D): A ← A·T + W·cum(s), W ← W·f(s), D ← D·T."""
    A, W, D = 0, 1, 1
    state = model.start()
    mb = qb = 0.0
    sb = n = 0
    for s in symbols:
        _, index, q = model.probs(state)
        i = index[s]
        f = frequencies(q)
        lo, fi, T = int(f[:i].sum()), int(f[i]), int(f.sum())
        A, W, D = A * T + W * lo, W * fi, D * T
        mb -= math.log2(q[i])
        qb -= math.log2(fi / T)
        sb += math.ceil(-math.log2(q[i]))
        state = model.step(state, s)
        n += 1
    b, L = shortest_dyadic(A, W, D)
    return Code(b, L, n, mb, qb, sb, W, D)


def arithmetic_decode(model, code: Code) -> list[str]:
    """Run the same model and read the symbols back. r = P/Qd is the codeword's position inside the current
    interval; the symbol is the slice containing r·T, and r is then rescaled to that slice."""
    P, Qd = code.bits, 1 << code.length
    state = model.start()
    out = []
    for _ in range(code.n_symbols):
        alphabet, _, q = model.probs(state)
        f = frequencies(q)
        cum = np.concatenate(([0], np.cumsum(f)))
        s = P * int(cum[-1])
        i = int(np.searchsorted(cum, s // Qd, side="right")) - 1
        P, Qd = s - int(cum[i]) * Qd, Qd * int(f[i])
        out.append(alphabet[i])
        state = model.step(state, alphabet[i])
    return out


def code_bounds_hold(c: Code) -> bool:
    """Thm 4 in exact integers: −log₂ Q̃(x) ≤ L ≤ ⌈−log₂ Q̃(x)⌉ + 1, i.e. 2^{−L} ≤ W/D and (L ≥ 2 ⇒)
    2^{L−2} < D/W."""
    return (c.width << c.length) >= c.denom and (c.length < 2 or (c.width << (c.length - 2)) < c.denom)


@lru_cache(maxsize=None)
def corpus_code(name: str = "ikn", test: str = "held") -> Code:
    """The whole held-out (or EDGAR) text, losslessly, as ONE arithmetic codeword under the open model."""
    m = open_model(name)
    return arithmetic_encode(m, m.symbols(ng.corpus()[test]))


def settled_bits(A: int, W: int, D: int, cap: int = 4096) -> int:
    """How many leading bits every number in [A/D, (A+W)/D) shares — the bits an encoder could already emit."""
    k = 0
    while k < cap and (A << (k + 1)) // D == -((-(A + W) << (k + 1)) // D) - 1:
        k += 1
    return k


class BigramModel:
    """A small random sequential model for the coder's strict tests: P(· | previous symbol)."""

    def __init__(self, P: np.ndarray, start: np.ndarray):
        self.P, self.p0 = P, start
        self.alphabet = tuple(str(i) for i in range(len(start)))
        self.index = {a: i for i, a in enumerate(self.alphabet)}

    def start(self):
        return None

    def probs(self, state):
        return self.alphabet, self.index, self.p0 if state is None else self.P[int(state)]

    def step(self, state, sym):
        return sym


COMPRESSORS = ("zlib", "bz2", "lzma")


def compress(name: str, data: bytes) -> bytes:
    if name == "zlib":
        return zlib.compress(data, 9)
    if name == "bz2":
        return bz2.compress(data, 9)
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def zlib_given(data: bytes, prime: bytes) -> bytes:
    """A genuine conditional code: zlib with the last 32 KiB of the training text as a preset dictionary
    (the decoder holds the same dictionary)."""
    c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY, prime[-32768:])
    return c.compress(data) + c.flush()


@lru_cache(maxsize=1)
def compressor_table() -> dict[str, dict[str, dict[str, float]]]:
    """Bits per character of each test text: compressed alone, and given the training text — for zlib a true
    conditional code (preset dictionary); for bz2 and lzma the extra bytes |C(train + test)| − |C(train)|,
    an approximation to one."""
    T = text_of(ng.corpus()["train"]).encode()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name in COMPRESSORS:
        out[name] = {}
        base = len(compress(name, T))
        for test in TESTS:
            X = text_of(ng.corpus()[test]).encode()
            given = len(zlib_given(X, T)) if name == "zlib" else len(compress(name, T + X)) - base
            out[name][test] = {"alone": len(compress(name, X)) * 8 / len(X), "given_train": given * 8 / len(X)}
    return out


# --------------------------------------------------------------------------------------------
# 5. Choosing the unit.
# --------------------------------------------------------------------------------------------


def renormalized(vec: KNVector, u) -> np.ndarray:
    """Q'(· | u): the closed word model with <unk> removed and renormalized — a proper distribution over
    sentences of in-vocabulary words, hence over their strings."""
    q = np.array(vec(u))
    q[vec.index[UNK]] = 0.0
    return q / q.sum()


def word_logprob(sent: NGram, vec: KNVector) -> float:
    """log Q'(sentence) = Σᵢ log Q'(wᵢ | uᵢ) + log Q'(</s> | u_{n+1}): the word-level factorization."""
    u, total = (BOS,) * (ORDER - 1), 0.0
    for w in (*sent, EOS):
        total += math.log(renormalized(vec, u)[vec.index[w]])
        u = u[1:] + (w,)
    return total


def induced_char_logprobs(sent: NGram, vec: KNVector) -> list[float]:
    """The SAME model Q', factored by characters (Thm 5): the log-probability of each character of
    " ".join(sent) + "\\n" given the characters before it.

    A partial word π after context u has prefix mass M(π) = Σ_{w ⊒ π} Q'(w | u) — a contiguous slice of the
    sorted alphabet. Letter c then has probability M(πc)/M(π). The character after a complete word w is its
    separator: Q'(w | u)/M(w) × P(</s> next) for "\\n", × (1 − P(</s> next)) for " ". After a space the next
    word cannot be </s>, so the first letter is normalized by 1 − Q'(</s> | u). Each word's factors
    telescope to Q'(w | u), and the separators carry the </s> decisions."""
    alpha, eos = vec.alphabet, vec.index[EOS]
    u = (BOS,) * (ORDER - 1)
    q = renormalized(vec, u)
    cs = np.concatenate(([0.0], np.cumsum(q)))
    norm, out = 1.0, []
    for i, w in enumerate(sent):
        prev = norm
        for j in range(1, len(w) + 1):
            m = cs[bisect_left(alpha, w[:j] + "{")] - cs[bisect_left(alpha, w[:j])]  # "{" sorts after "z"
            out.append(math.log(m / prev))
            prev = m
        nxt = u[1:] + (w,)
        q2 = renormalized(vec, nxt)
        end = q2[eos] if i == len(sent) - 1 else 1.0 - q2[eos]
        out.append(math.log(q[vec.index[w]] / prev * end))
        u, q = nxt, q2
        cs = np.concatenate(([0.0], np.cumsum(q)))
        norm = 1.0 - q[eos]
    return out


CHAR_ORDERS = (3, 5, 7)
CHAR_ALPHABET = tuple(sorted(" " + LETTERS)) + (EOS,)  # letters, the space, and the sentence end
BPE_ORDER = 5
BPE_SEP = "▁"  # the word separator in a piece sequence (the space, in character terms)
BPE_SIZES = (100, 400)  # plus the two-part-MDL-optimal vocabulary of Topic 2


def char_sequences(sents) -> tuple[NGram, ...]:
    return tuple(tuple(" ".join(s)) for s in sents)


@lru_cache(maxsize=None)
def char_model(order: int) -> ng.KNModel:
    return kn_fit_any(char_sequences(ng.corpus()["train"]), order, CHAR_ALPHABET)


@lru_cache(maxsize=1)
def bpe_full() -> sw.MergeModel:
    """Topic 2's BPE on the same training words, run to BPE_MERGES_MAX merges for the MDL curve."""
    return sw.bpe_train(sw.finance_words()["train"], sw.BPE_MERGES_MAX)


@lru_cache(maxsize=1)
def bpe_mdl_size() -> int:
    """|V| at the minimum of Topic 2's two-part code along BPE's merge path (its page: 1,870 pieces)."""
    m = bpe_full()
    return len(m.alphabet) + int(np.argmin(sw.mdl_curve(m)))


def bpe_sizes() -> tuple[int, ...]:
    return (*BPE_SIZES, bpe_mdl_size())


@lru_cache(maxsize=None)
def bpe_at(size: int) -> sw.MergeModel:
    return sw.bpe_train(sw.finance_words()["train"], size - len(bpe_full().alphabet))


@lru_cache(maxsize=None)
def bpe_segment(word: str, size: int) -> tuple[str, ...]:
    m = bpe_at(size)
    return tuple(m.segs[word]) if word in m.segs else tuple(sw.replay_encode(word, m.merges))


def bpe_sequences(sents, size: int) -> tuple[NGram, ...]:
    """Each sentence as BPE pieces with BPE_SEP between words — a bijection with its string."""
    return tuple(tuple(itertools.chain.from_iterable(
        (BPE_SEP, *bpe_segment(w, size)) if i else bpe_segment(w, size) for i, w in enumerate(s))) for s in sents)


def bpe_alphabet(size: int) -> tuple[str, ...]:
    m = bpe_at(size)
    return tuple(sorted(set(m.alphabet) | {a + b for a, b in m.merges} | {BPE_SEP, EOS}))


@lru_cache(maxsize=None)
def bpe_model(size: int) -> ng.KNModel:
    return kn_fit_any(bpe_sequences(ng.corpus()["train"], size), BPE_ORDER, bpe_alphabet(size))


def unit_models() -> tuple[str, ...]:
    return (*(f"char-{n}" for n in CHAR_ORDERS), *(f"bpe-{s}" for s in bpe_sizes()), "word", "word-closed")


def _kn_total(model: ng.KNModel, seqs) -> tuple[float, int]:
    evs = tuple(ng.events(seqs, model.order))
    return ng.log_loss(lambda w, u: ng.interpolated_kn_prob(w, u, model), evs) * len(evs), len(evs)


@lru_cache(maxsize=None)
def unit_row(name: str, test: str = "held") -> dict:
    """One model under every unit: total NLL (nats) over the test text, its token count under the model's own
    tokenizer, and the per-token, per-character (bpc), and per-word-token normalizations of the same total.
    "word" is the open-vocabulary code; "word-closed" counts <unk> as one event and is not a code."""
    sents = ng.corpus()[test]
    if name.startswith("char-"):
        total, n_tok = _kn_total(char_model(int(name[5:])), char_sequences(sents))
    elif name.startswith("bpe-"):
        size = int(name[4:])
        total, n_tok = _kn_total(bpe_model(size), bpe_sequences(sents, size))
    else:
        col = 3 if name == "word" else 0
        total, n_tok = float(sentence_nll("ikn", test)[:, col].sum()), n_word_tokens(sents)
    nc, nw = n_chars(sents), n_word_tokens(sents)
    return {"model": name, "is_code": name != "word-closed", "tokens": n_tok, "tokens_per_char": n_tok / nc,
            "nats_per_token": total / n_tok, "pp_token": perplexity(total / n_tok), "bpc": total / nc / LN2,
            "pp_word": perplexity(total / nw), "total_nats": total}


UNK_CUTOFFS = (0, 1, 4, 16, 64, math.inf)  # words with training count ≤ k become <unk>


@lru_cache(maxsize=None)
def cutoff_vocab(k: float) -> frozenset[str]:
    return frozenset(g[0] for g, n in ng.train_counts()[1].items() if n > k) | {EOS, UNK}


@lru_cache(maxsize=None)
def cutoff_model(k: float) -> ng.KNModel:
    V = cutoff_vocab(k)
    return kn_fit_any(ng.with_unk(ng.corpus()["train"], V), ORDER, sorted(V))


@lru_cache(maxsize=None)
def cutoff_row(k: float, test: str = "held") -> dict:
    """Interpolated KN with vocabulary V_k: the closed per-token loss (<unk> as one event) and the open bits
    per character (each <unk> then spelled by the speller trained for this cutoff)."""
    V, m, sents = cutoff_vocab(k), cutoff_model(k), ng.corpus()[test]
    evs = tuple(ng.events(ng.with_unk(sents, V), ORDER))
    closed = ng.log_loss(lambda w, u: ng.interpolated_kn_prob(w, u, m), evs) * len(evs)
    sk = 1 if k == 0 else k
    spelled = sum(spell_nll(w, sk) for s in sents for w in s if w not in V)
    oov = sum(w not in V for s in sents for w in s) / n_word_tokens(sents)  # Topic 3's share: over words and </s>
    return {"k": None if math.isinf(k) else int(k), "V": len(V), "oov_share": oov,
            "closed_nats_per_token": closed / len(evs), "closed_pp": perplexity(closed / len(evs)),
            "closed_bpc": closed / n_chars(sents) / LN2, "open_bpc": (closed + spelled) / n_chars(sents) / LN2}


@lru_cache(maxsize=1)
def invariance_demo() -> dict[str, float]:
    """Thm 5 on the held-out sentences made only of in-vocabulary words: Q' (interpolated KN without <unk>)
    scored once, reported per word token and per character."""
    vec, V = word_vector("ikn"), ng.vocabulary()
    sents = [s for s in ng.corpus()["held"] if all(w in V for w in s)]
    total = -sum(word_logprob(s, vec) for s in sents)
    nc, nt = n_chars(sents), n_word_tokens(sents)
    return {"sentences": len(sents), "chars": nc, "word_tokens": nt, "nats": total,
            "pp_word": perplexity(total / nt), "pp_char": perplexity(total / nc), "bpc": total / nc / LN2}


# --------------------------------------------------------------------------------------------
# 6. Finance case study: EDGAR.
# --------------------------------------------------------------------------------------------

SPELLER_ORDERS = (1, SPELL_ORDER)  # a letter-frequency speller against the character trigram


@lru_cache(maxsize=1)
def speller_comparison() -> dict[str, dict[int, float]]:
    """Open-vocabulary bpc of interpolated KN with the OOV words spelled by a letter unigram (order 1) or by
    the character trigram, both fit to the training hapaxes."""
    out: dict[str, dict[int, float]] = {}
    for test in TESTS:
        a, sents = sentence_nll("ikn", test), ng.corpus()[test]
        out[test] = {o: (a[:, 0].sum() + sum(spell_nll(w, 1, o) for s in sents for w in s
                                             if w not in ng.vocabulary())) / n_chars(sents) / LN2
                     for o in SPELLER_ORDERS}
    return out


@lru_cache(maxsize=1)
def finance_summary() -> dict[str, float]:
    held, ed = sentence_nll("ikn", "held"), sentence_nll("ikn", "edgar")
    out = {"uniform_nats": uniform_log_loss(), "V": len(ng.vocabulary()),
           "oov_share_edgar": ng.oov_share(ng.corpus()["edgar"]), "oov_share_held": ng.oov_share(ng.corpus()["held"]),
           "spelled_share_edgar": float(1 - ed[:, 0].sum() / ed[:, 3].sum()),
           "spelled_share_held": float(1 - held[:, 0].sum() / held[:, 3].sum())}
    for name in ("laplace", "ikn", "mkn"):
        for test in TESTS:
            a = sentence_nll(name, test)
            out[f"{name}_{test}_closed_nats"] = float(a[:, 0].sum() / a[:, 1].sum())
            out[f"{name}_{test}_closed_bpc"] = float(a[:, 0].sum() / a[:, 2].sum() / LN2)
            out[f"{name}_{test}_open_bpc"] = float(a[:, 3].sum() / a[:, 2].sum() / LN2)
    return out


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX's JSON in Phase B).
# --------------------------------------------------------------------------------------------

LAB_SENTENCES = (
    ("held", ("halvorsen", "reported", "revenue", "of", "million", "for", "the", "second", "quarter")),
    ("held", ("liabilities", "were", "broadly", "unchanged", "in", "coquotas", "nonroyaltys")),  # Topic 2's word
    ("edgar", ("book", "reserves", "are", "established", "for", "these", "plans")),  # 5 of 7 words OOV
)
LAB_SMOOTHERS = ("laplace", "ikn", "mkn")
LAB_TOP = 5


def _r(x: float, nd: int = 4):
    return None if not math.isfinite(x) else round(float(x), nd)


def interval_trace(model, symbols, top: int = LAB_TOP) -> dict:
    """Panel A's data for one sentence: per symbol [symbol, mode, Q, slice start, bits so far, settled bits]
    and the top alternatives [symbol, Q, slice start]; then the codeword of the sentence coded alone."""
    A, W, D = 0, 1, 1
    state, steps, mb = model.start(), [], 0.0
    for s in symbols:
        alphabet, index, q = model.probs(state)
        f = frequencies(q)
        cum = np.concatenate(([0], np.cumsum(f)))
        T, i = int(cum[-1]), index[s]
        A, W, D = A * T + W * int(cum[i]), W * int(f[i]), D * T
        mb -= math.log2(q[i])
        alts = np.argsort(-q, kind="stable")[:top]
        steps.append({"s": [s, state[0], _r(q[i], 6), _r(cum[i] / T, 6), _r(mb, 3), settled_bits(A, W, D)],
                      "top": [[alphabet[j], _r(q[j], 6), _r(cum[j] / T, 6)] for j in alts]})
        state = model.step(state, s)
    c = arithmetic_encode(model, symbols)
    return {"steps": steps, "length": c.length, "model_bits": _r(c.model_bits, 3), "codeword": c.bitstring(),
            "shannon_bits": c.shannon_bits}


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    sentences = []
    for test, words in LAB_SENTENCES:
        by = {}
        for name in LAB_SMOOTHERS:
            m = open_model(name)
            by[name] = interval_trace(m, m.symbols([words]))
        sentences.append({"test": test, "words": list(words), "chars": n_chars([words]),
                          "oov": [w for w in words if w not in ng.vocabulary()], "by_smoother": by})
    units = {t: [{k: (_r(v) if isinstance(v, float) else v) for k, v in unit_row(name, t).items()
                  if k != "total_nats"} for name in unit_models()] for t in TESTS}
    cutoffs = {t: [{k: (_r(v) if isinstance(v, float) else v) for k, v in cutoff_row(k_, t).items()}
                   for k_ in UNK_CUTOFFS] for t in TESTS}
    fs = finance_summary()
    corpus = {t: {name: {"closed_nats": _r(fs[f"{name}_{t}_closed_nats"]), "open_bpc": _r(fs[f"{name}_{t}_open_bpc"]),
                         "closed_bpc": _r(fs[f"{name}_{t}_closed_bpc"])} for name in LAB_SMOOTHERS} for t in TESTS}
    codes = {t: {"length": corpus_code("ikn", t).length, "model_bits": _r(corpus_code("ikn", t).model_bits, 3),
                 "chars": n_chars(ng.corpus()[t]),
                 "shannon_bits": corpus_code("ikn", t).shannon_bits} for t in TESTS}
    comp = {name: {t: {k: _r(v) for k, v in row.items()} for t, row in d.items()}
            for name, d in compressor_table().items()}
    return {"sentences": sentences, "units": units, "cutoffs": cutoffs, "corpus": corpus, "codes": codes,
            "compressors": comp, "uniform_nats": _r(uniform_log_loss()), "V": len(ng.vocabulary())}


# --------------------------------------------------------------------------------------------
# Summary — the numbers the page quotes.
# --------------------------------------------------------------------------------------------


def summary() -> dict[str, float]:
    out: dict[str, float] = {}
    for t in TESTS:
        sents = ng.corpus()[t]
        out[f"{t}_sentences"], out[f"{t}_word_tokens"] = len(sents), n_word_tokens(sents)
        out[f"{t}_chars"] = n_chars(sents)
    a = sentence_nll("ikn", "held")
    out["ikn_held_nats"], out["ikn_held_se_sentence"] = ratio_estimate(a[:, 0], a[:, 1])
    out["ikn_held_se_token"] = token_level_se("ikn")
    out["ikn_held_pp"] = perplexity(out["ikn_held_nats"])
    out["ikn_held_open_bpc"], out["ikn_held_open_bpc_se"] = (x / LN2 for x in ratio_estimate(a[:, 3], a[:, 2]))
    d = sentence_nll("mkn", "held")[:, 0] - a[:, 0]
    out["mkn_minus_ikn"], out["mkn_minus_ikn_se"] = ratio_estimate(d, a[:, 1])
    for t in TESTS:
        c = corpus_code("ikn", t)
        out[f"code_{t}_bits"], out[f"code_{t}_model_bits"] = c.length, c.model_bits
        out[f"code_{t}_quantized_bits"], out[f"code_{t}_shannon_bits"] = c.quantized_bits, c.shannon_bits
        out[f"code_{t}_symbols"] = c.n_symbols
        out[f"code_{t}_bpc"] = c.length / n_chars(ng.corpus()[t])
        for name, row in compressor_table().items():
            out[f"{name}_{t}_alone_bpc"], out[f"{name}_{t}_given_bpc"] = row[t]["alone"], row[t]["given_train"]
        for name in unit_models():
            r = unit_row(name, t)
            out[f"unit_{t}_{name}_pp_token"], out[f"unit_{t}_{name}_bpc"] = r["pp_token"], r["bpc"]
        for k in UNK_CUTOFFS:
            r, lab = cutoff_row(k, t), "all" if math.isinf(k) else int(k)
            out[f"cutoff_{t}_{lab}_closed_pp"], out[f"cutoff_{t}_{lab}_open_bpc"] = r["closed_pp"], r["open_bpc"]
    out["bpe_mdl_size"] = bpe_mdl_size()
    for k, v in typical_example().items():
        out[f"typical_{k}"] = v
    for k, v in mcmillan_census().items():
        out[f"mcmillan_{k}"] = v
    for k, v in invariance_demo().items():
        out[f"invariance_{k}"] = v
    for t, row in speller_comparison().items():
        for o, v in row.items():
            out[f"speller_{t}_order{o}_open_bpc"] = v
    out.update({f"fin_{k}": v for k, v in finance_summary().items()})
    return out


# --------------------------------------------------------------------------------------------
# Verification harness — every claim the page makes.
# --------------------------------------------------------------------------------------------


def test_gibbs_decomposition():
    """Prop 1 (formalML kl-divergence, Prop 1, cited): H(p, q) = H(p) + KL(p ‖ q) ≥ H(p), equality iff q = p.
    Cross-check: scipy.stats.entropy for H and KL."""
    rng = np.random.default_rng([SEED, 1])
    for _ in range(300):
        k = int(rng.integers(2, 40))
        p, q = rng.dirichlet(np.full(k, 0.5)), rng.dirichlet(np.full(k, 0.5))
        H, X, D = zhv.entropy(p), cross_entropy(p, q), kl(p, q)
        assert abs(X - (H + D)) < 1e-9 * max(1, X) and D > 0
        assert abs(H - stats.entropy(p)) < 1e-10 and abs(D - stats.entropy(p, q)) < 1e-9 * max(1, D)
        assert abs(kl(p, p)) < 1e-12


def test_log_score_strictly_proper():
    """Cor 1: the expected log score E_p[−log q(X)] has the unique minimizer q = p; the linear score E_p[q(X)]
    is not proper — it is maximized by a point mass, not by p."""
    rng = np.random.default_rng([SEED, 2])
    for _ in range(200):
        k = int(rng.integers(2, 12))
        p = rng.dirichlet(np.ones(k))
        for _ in range(5):
            q = rng.dirichlet(np.ones(k))
            assert cross_entropy(p, q) > cross_entropy(p, p)
        assert np.max(p) > p @ p  # the linear score prefers the point mass at the mode to the truth


def test_perplexity_facts():
    """Prop 2: perplexity is the geometric mean of 1/Q over the tokens, at most their arithmetic mean; the
    uniform model on k symbols has perplexity exactly k on any text; 1 ≤ e^{H(p)} ≤ |supp p|."""
    rng = np.random.default_rng([SEED, 3])
    for _ in range(100):
        q = rng.uniform(1e-4, 1, size=int(rng.integers(1, 200)))
        pp = perplexity(float(np.mean(-np.log(q))))
        assert abs(pp - float(np.prod(q ** (-1 / len(q))))) < 1e-8 * pp and pp <= np.mean(1 / q) * (1 + 1e-12)
        k = int(rng.integers(1, 5000))
        assert abs(perplexity(math.log(k)) - k) < 1e-8 * k
        p = rng.dirichlet(np.full(int(rng.integers(1, 30)), 0.3))
        assert 1 - 1e-12 <= math.exp(zhv.entropy(p)) <= np.count_nonzero(p) * (1 + 1e-12)
    a = sentence_nll("ikn", "held")
    assert abs(a[:, 0].sum() / a[:, 1].sum() - 2.8551) < 5e-5  # Topic 3's interpolated KN, reproduced


def test_worse_than_uniform_out_of_domain():
    """Prop 2 on EDGAR: perplexity above |V| means worse than the uniform guess. Interpolated and modified KN
    cross that line out of domain; Laplace, which smooths toward uniform hardest, barely stays under it."""
    fs, u = finance_summary(), uniform_log_loss()
    assert fs["ikn_edgar_closed_nats"] > fs["mkn_edgar_closed_nats"] > u > fs["laplace_edgar_closed_nats"]
    assert perplexity(fs["ikn_edgar_closed_nats"]) > len(ng.vocabulary())
    assert fs["ikn_held_closed_nats"] < fs["laplace_held_closed_nats"] < u  # in domain all three beat uniform


def test_typical_set_aep():
    """Thm 1 (AEP), exactly by type classes, on random 3-symbol sources: Chebyshev's P(A) ≥ 1 − σ²/(nε²), the
    sandwich |A| e^{−n(H+ε)} ≤ P(A) ≤ |A| e^{−n(H−ε)} with |A| ≤ e^{n(H+ε)}; and the model version
    P(|−(1/n) log q(Xⁿ) − H(p, q)| ≤ ε) ≥ 1 − Var(−log q)/(nε²)."""
    rng = np.random.default_rng([SEED, 4])
    for _ in range(12):
        p = 0.05 + 0.85 * rng.dirichlet(np.ones(3))
        q = rng.dirichlet(np.ones(3))
        H = zhv.entropy(p)
        for n in (40, 120, 400):
            for eps in (0.1, 0.25):
                t = typical_set(p, n, eps)
                assert t["prob"] >= t["chebyshev"] - 1e-12 and t["log_size"] <= n * (H + eps) + 1e-9
                if t["prob"] > 0:
                    lp = math.log(t["prob"])
                    assert t["log_size"] - n * (H + eps) <= lp + 1e-9 <= t["log_size"] - n * (H - eps) + 1e-9
                tm = typical_set(p, n, eps, q)
                assert tm["prob"] >= tm["chebyshev"] - 1e-12
    ex = typical_example()
    assert ex["prob"] > 0.9 and ex["share_of_sequences"] < 1e-15  # almost all the probability, almost none of Σⁿ


def test_ratio_estimator_target():
    """Thm 2: on random bigram sentence sources (exact targets by absorbing-chain algebra), the per-token
    log-loss of many i.i.d. sentences sits within 4 standard errors of E[NLL_Q(S)]/E[|S|]."""
    rng = np.random.default_rng([SEED, 5])
    for _ in range(20):
        P, Q = random_sentence_source(rng), random_sentence_source(rng)
        e_nll, e_len = exact_ratio(P, Q)
        r, se = sentence_ratio(P, Q, 20_000, rng)
        assert abs(r - e_nll / e_len) < 4 * se and se < 0.05


def test_ratio_se_coverage():
    """Thm 2's delta-method interval r̂ ± 1.96 se covers the target at its nominal rate (400 replications of
    300 sentences; binomial ±3 s.d. band)."""
    rng = np.random.default_rng([SEED, 6])
    P, Q = random_sentence_source(rng), random_sentence_source(rng)
    e_nll, e_len = exact_ratio(P, Q)
    hits = sum(abs((rs := sentence_ratio(P, Q, 300, rng))[0] - e_nll / e_len) <= 1.96 * rs[1] for _ in range(400))
    assert 0.917 <= hits / 400 <= 0.983, hits


def test_corpus_standard_errors():
    """On the held-out corpus the sentence-level standard error is ~1.4× the naive token-level one, and the
    paired MKN − IKN difference is many standard errors from 0 although the marginal intervals overlap."""
    a, b = sentence_nll("ikn", "held"), sentence_nll("mkn", "held")
    r, se = ratio_estimate(a[:, 0], a[:, 1])
    assert se > 1.25 * token_level_se("ikn")
    d, se_d = ratio_estimate(b[:, 0] - a[:, 0], a[:, 1])
    assert d < 0 and abs(d) > 10 * se_d and abs(d) < se


def test_smb_on_a_markov_chain():
    """Shannon–McMillan–Breiman (cited) and the ergodic theorem for a Markov model, on random ergodic chains:
    along one path of length 10⁵, −(1/n) log P(Xⁿ) → entropy rate and −(1/n) log Q(Xⁿ) → cross-entropy rate
    (within 4 batch-means standard errors)."""
    rng = np.random.default_rng([SEED, 7])
    for _ in range(3):
        P, Q = random_chain(rng), random_chain(rng)
        x = sample_path(P, 100_000, rng)
        for M, target in ((P, entropy_rate(P)), (Q, cross_entropy_rate(P, Q))):
            per = -np.log(M[x[:-1], x[1:]])
            assert abs(per.mean() - target) < 4 * batch_means_se(per) + 1e-3


def test_cross_entropy_rate_bounds_entropy_rate():
    """Cor 2: E[−log Q(Xⁿ)] − H(Xⁿ) = KL(Pⁿ ‖ Qⁿ) ≥ 0 for every block length (exact enumeration), so the rates
    obey H(P, Q) ≥ H(P), with equality at Q = P."""
    rng = np.random.default_rng([SEED, 8])
    for _ in range(25):
        P, Q = random_chain(rng, 3), random_chain(rng, 3)
        pi, piq = stationary(P), stationary(Q)
        for n in (1, 2, 4):
            seqs = np.array(list(itertools.product(range(3), repeat=n)))
            lp = np.log(pi[seqs[:, 0]]) + np.log(P[seqs[:, :-1], seqs[:, 1:]]).sum(1)
            lq = np.log(piq[seqs[:, 0]]) + np.log(Q[seqs[:, :-1], seqs[:, 1:]]).sum(1)
            assert np.exp(lp) @ (lp - lq) >= -1e-12
        assert cross_entropy_rate(P, Q) > entropy_rate(P) and abs(cross_entropy_rate(P, P) - entropy_rate(P)) < 1e-12


def test_kraft_prefix_and_converse():
    """Kraft both ways (formalML shannon-entropy Thm 6, cited): Huffman codes are prefix-free with Kraft sum
    exactly 1; any lengths with Σ 2^{−ℓ} ≤ 1 get a prefix code by the dyadic-interval construction; lengths
    with Σ > 1 get none."""
    rng = np.random.default_rng([SEED, 9])
    for _ in range(200):
        p = rng.dirichlet(np.full(int(rng.integers(2, 25)), 0.5))
        L = huffman_lengths(p)
        code = canonical_code(L)
        assert code is not None and is_prefix_free(code) and kraft_sum(L) == 1
        lengths = [int(x) for x in rng.integers(1, 9, size=int(rng.integers(1, 12)))]
        code = canonical_code(lengths)
        assert (code is not None) == (kraft_sum(lengths) <= 1)
        if code is not None:
            assert is_prefix_free(code) and [len(c) for c in code] == lengths


def test_mcmillan_exhaustive():
    """Thm 3 (McMillan): every uniquely decodable code satisfies Kraft — checked by Sardinas–Patterson on ALL
    binary codes of 3 or 4 codewords of length ≤ 3, which include uniquely decodable codes that are not
    prefix-free (e.g. {0, 01, 11}). Codes with Kraft sum > 1 are never uniquely decodable. Cross-check:
    Sardinas–Patterson agrees with brute-force parse counting on every one of these codes."""
    assert is_uniquely_decodable(["0", "01", "11"]) and not is_prefix_free(["0", "01", "11"])
    assert not is_uniquely_decodable(["0", "01", "10"])  # 010 = 0·10 = 01·0
    assert not is_uniquely_decodable(["0", "1", "01"]) and is_uniquely_decodable(["00", "01", "10", "11"])
    c = mcmillan_census()
    assert c["ud_kraft_violations"] == 0 and c["ud_not_prefix"] > 0 and c["kraft_violations"] > 0
    assert c["brute_force_disagreements"] == 0  # Sardinas–Patterson agrees with parse counting to 10 bits


def test_source_coding_with_the_wrong_model():
    """Prop 3: the Shannon code for q is a prefix code whose expected length under p lies in
    [H₂(p, q), H₂(p, q) + 1) — the true entropy plus KL(p ‖ q) plus less than a bit; Huffman for p achieves
    [H₂(p), H₂(p) + 1) (formalML shannon-entropy Thm 7, cited)."""
    rng = np.random.default_rng([SEED, 10])
    for _ in range(300):
        k = int(rng.integers(2, 40))
        p, q = rng.dirichlet(np.full(k, 0.7)), rng.dirichlet(np.full(k, 0.7))
        Lq = shannon_lengths(q)
        assert kraft_sum(Lq) <= 1
        X2 = cross_entropy(p, q) / LN2
        assert X2 - 1e-12 <= p @ np.array(Lq) < X2 + 1
        H2 = zhv.entropy(p) / LN2
        assert H2 - 1e-12 <= p @ np.array(huffman_lengths(p)) < H2 + 1


def test_arithmetic_code_random_instances():
    """Thm 4 on random sequential models: decoding recovers every message; −log₂ Q̃ ≤ L ≤ ⌈−log₂ Q̃⌉ + 1 in exact
    integers; over ALL messages of a fixed length the codewords are prefix-free with Kraft sum ≤ 1."""
    rng = np.random.default_rng([SEED, 11])
    for _ in range(200):
        k = int(rng.integers(2, 6))
        m = BigramModel(rng.dirichlet(np.full(k, 0.5), size=k), rng.dirichlet(np.ones(k)))
        msg = [str(s) for s in rng.integers(0, k, size=int(rng.integers(1, 60)))]
        c = arithmetic_encode(m, msg)
        assert arithmetic_decode(m, c) == msg and code_bounds_hold(c)
    m = BigramModel(rng.dirichlet(np.ones(3), size=3), rng.dirichlet(np.ones(3)))
    codes = [arithmetic_encode(m, list(x)).bitstring() for x in itertools.product(m.alphabet, repeat=5)]
    assert is_prefix_free(codes) and kraft_sum(map(len, codes)) <= 1


def test_arithmetic_code_corpus():
    """Thm 4 on the corpus: the whole held-out text (and EDGAR), coded as ONE codeword under interpolated KN
    with OOV words spelled, decodes to the exact sentences; its length is within 2 bits of −log₂ Q̃(text);
    quantizing to integer frequencies costs < 0.01 bits in all; symbol-by-symbol Shannon coding would cost
    over 0.3 bits per symbol more; and the code length equals the open-vocabulary NLL in sentence_nll."""
    for test in TESTS:
        m, c = open_model("ikn"), corpus_code("ikn", test)
        assert sents_from_symbols(arithmetic_decode(m, c)) == list(ng.corpus()[test])
        assert code_bounds_hold(c) and c.length - c.quantized_bits < 2
        assert abs(c.quantized_bits - c.model_bits) < 0.01
        assert c.shannon_bits - c.length > 0.3 * c.n_symbols
        assert abs(c.model_bits * LN2 - sentence_nll("ikn", test)[:, 3].sum()) < 1e-6 * c.model_bits


def test_compressors_are_models():
    """Prop 4: a compressor is a code, so its lengths are a model's NLL. In domain the KN code beats every
    general-purpose compressor given the same training text; on EDGAR lzma beats every static model."""
    comp = compressor_table()
    T = text_of(ng.corpus()["train"]).encode()
    X = text_of(ng.corpus()["held"]).encode()
    d = zlib.decompressobj(zdict=T[-32768:])
    assert d.decompress(zlib_given(X, T)) == X  # the zlib conditional code decodes with the dictionary
    held_bpc = corpus_code("ikn", "held").length / n_chars(ng.corpus()["held"])
    assert all(held_bpc < comp[c]["held"]["given_train"] for c in COMPRESSORS)
    lz = comp["lzma"]["edgar"]["given_train"]
    assert all(lz < unit_row(name, "edgar")["bpc"] for name in unit_models() if unit_row(name, "edgar")["is_code"])
    assert len(text_of(ng.corpus()["held"])) == n_chars(ng.corpus()["held"])


def test_vector_models_match_scalar():
    """The vectorized models used for coding equal the prerequisite's scalar recursions (ng.interpolated_kn_prob,
    ng.additive_prob, and the speller), and every vector sums to 1."""
    rng = np.random.default_rng([SEED, 12])
    evs = ng.heldout_events()
    for name in LAB_SMOOTHERS:
        vec, f = word_vector(name), ng.prob_fn(name)
        for i in rng.choice(len(evs), 300, replace=False):
            g = evs[i]
            q = vec(g[:-1])
            assert abs(q.sum() - 1) < 1e-9 and abs(q[vec.index[g[-1]]] - f(g[-1], g[:-1])) < 1e-12
    sp = speller()
    for w in ("nonroyaltys", "coquotas", "reserves"):
        for g in ng.events([tuple(w)], SPELL_ORDER):
            q = sp(g[:-1])
            scalar = ng.interpolated_kn_prob(g[-1], g[:-1], sp.model)
            assert abs(q.sum() - 1) < 1e-12 and abs(q[sp.index[g[-1]]] - scalar) < 1e-14


def test_tokenization_invariance():
    """Thm 5: on every held-out sentence of in-vocabulary words, the word model Q' factored by characters
    (induced_char_logprobs) gives the same total log-probability as its word factorization; so per-character
    and per-word-token perplexities of one model satisfy n_char · log PP_char = n_tok · log PP_tok. Each
    character step is a distribution: M(π) = Σ_c M(πc) + Q'(π)."""
    vec, V = word_vector("ikn"), ng.vocabulary()
    sents = [s for s in ng.corpus()["held"] if all(w in V for w in s)]
    tot_c = 0.0
    for s in sents:
        lc, lw = induced_char_logprobs(s, vec), word_logprob(s, vec)
        assert len(lc) == n_chars([s]) and abs(sum(lc) - lw) < 1e-9 * max(1, abs(lw))
        tot_c += sum(lc)
    d = invariance_demo()
    assert abs(-tot_c - d["nats"]) < 1e-8 * d["nats"] and d["pp_char"] < 2 < 15 < d["pp_word"]
    per_char = d["chars"] * math.log(perplexity(-tot_c / d["chars"]))
    assert abs(per_char - d["word_tokens"] * math.log(d["pp_word"])) < 1e-6 * d["nats"]
    alpha = vec.alphabet
    q = renormalized(vec, ("the", "second"))
    cs = np.concatenate(([0.0], np.cumsum(q)))
    mass = lambda pre: cs[bisect_left(alpha, pre + "{")] - cs[bisect_left(alpha, pre)]  # noqa: E731
    for pre in ("q", "qu", "quar", "quarter", "re", "rev"):
        here = q[vec.index[pre]] if pre in vec.index else 0.0
        assert abs(mass(pre) - sum(mass(pre + c) for c in LETTERS) - here) < 1e-14


def test_per_token_perplexity_misranks():
    """Thm 5's consequence: ranking models by per-token perplexity is not ranking them by code length. The
    character 7-gram has a far lower per-token perplexity than the word trigram and a higher bpc; every
    tokenizer-free unit (per character, per word token) gives one ranking."""
    rows = {name: unit_row(name, "held") for name in unit_models() if name != "word-closed"}
    assert rows["char-7"]["pp_token"] < rows["word"]["pp_token"] / 5 and rows["char-7"]["bpc"] > rows["word"]["bpc"]
    by_pp = sorted(rows, key=lambda n: rows[n]["pp_token"])
    by_bpc = sorted(rows, key=lambda n: rows[n]["bpc"])
    assert by_pp != by_bpc and by_bpc == sorted(rows, key=lambda n: rows[n]["pp_word"])
    assert bpe_mdl_size() == 1870  # Topic 2's MDL optimum (1,844 merges)


def test_unk_flatters():
    """Prop 5: the open-vocabulary loss is the closed loss plus the spelling of every OOV word, so closed ≤
    open for every model and test set; and as the vocabulary shrinks the closed perplexity falls toward 1
    while, in domain, the open code gets longer at every step (on EDGAR it gets shorter overall)."""
    for test in TESTS:
        for name in LAB_SMOOTHERS:
            a = sentence_nll(name, test)
            assert np.all(a[:, 0] <= a[:, 3] + 1e-12)
        rows = [cutoff_row(k, test) for k in UNK_CUTOFFS]
        assert all(r["closed_bpc"] <= r["open_bpc"] for r in rows)
        assert rows[0]["V"] == len(ng.vocabulary())
        assert abs(rows[0]["oov_share"] - ng.oov_share(ng.corpus()[test])) < 1e-12
    held = [cutoff_row(k, "held") for k in UNK_CUTOFFS]
    assert abs(held[0]["closed_nats_per_token"] - 2.8551) < 5e-5  # k = 0 is Topic 3's model
    assert all(a["closed_pp"] > b["closed_pp"] and a["open_bpc"] < b["open_bpc"] for a, b in zip(held, held[1:]))
    assert held[-1]["closed_pp"] < 2 < held[-1]["open_bpc"]
    edgar = [cutoff_row(k, "edgar") for k in UNK_CUTOFFS]  # out of domain the honest code improves instead
    assert all(a["closed_pp"] > b["closed_pp"] for a, b in zip(edgar, edgar[1:]))
    assert edgar[-1]["open_bpc"] < edgar[0]["open_bpc"]


def test_finance_findings():
    """§6: on EDGAR most of the honest code is spelling — the closed-vocabulary bpc is under half the open
    one — while in domain spelling is a small share; and the speller fit to the corpus's pseudo-words, better
    in domain, spells real English worse than plain letter frequencies."""
    fs = finance_summary()
    assert fs["spelled_share_edgar"] > 0.5 > 0.1 > fs["spelled_share_held"]
    assert fs["ikn_edgar_open_bpc"] > 2 * fs["ikn_edgar_closed_bpc"]
    sp = speller_comparison()  # the trigram speller wins in domain and loses to letter frequencies on EDGAR
    assert sp["held"][SPELL_ORDER] < sp["held"][1] and sp["edgar"][1] < sp["edgar"][SPELL_ORDER]
    assert abs(sp["held"][SPELL_ORDER] - fs["ikn_held_open_bpc"]) < 1e-12
    assert abs(fs["oov_share_edgar"] - 0.5759) < 5e-5  # Topic 3's figure


def test_viz_payload():
    p = viz_payload()
    for sent in p["sentences"]:
        for name, tr in sent["by_smoother"].items():
            assert tr["steps"] and abs(tr["steps"][-1]["s"][4] - tr["model_bits"]) < 2e-3
            assert tr["model_bits"] - 1e-3 <= tr["length"] < tr["model_bits"] + 2 + 1e-3
            for st in tr["steps"]:
                assert sum(q for _, q, _ in st["top"]) <= 1 + 1e-5 and st["s"][5] <= tr["length"]
    assert p["sentences"][1]["oov"] == ["nonroyaltys"] and len(p["sentences"][2]["oov"]) == 5
    assert not p["sentences"][0]["oov"]
    for t in TESTS:
        closed = [r for r in p["units"][t] if not r["is_code"]]
        assert len(closed) == 1 and closed[0]["model"] == "word-closed"


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
    print("\nSummary (train generate_corpus(300, 0), held out generate_corpus(100, 1), EDGAR excerpts):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.4g}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else
              f"  {k:36} {v:.4f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
