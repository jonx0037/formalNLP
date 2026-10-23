"""Subword tokenization as compression — the reference implementation for formalNLP's second topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. The open-vocabulary problem: a vocabulary encodes every string iff it contains the alphabet;
   a length-n string has 2^{n−1} segmentations (L-step Fibonacci when pieces are capped at L).
2. BPE as greedy compression: a merge saves exactly its replacement count; replaying merges in rank
   order reproduces training segmentations; greedy is not optimal (optimal pair encoding is
   APX-complete, Kozma & Voderholzer — cited, checked by brute force on tiny strings).
3. WordPiece's likelihood criterion: the exact change in unigram log-likelihood from a merge, and its
   count-weighted-PMI leading term with an explicit remainder bound.
4. Unigram LM: Viterbi and forward–backward over the segmentation lattice; EM never lowers the
   marginal likelihood; vocabulary pruning is the (heuristic) SentencePiece loss.
5. Tokenization as compression: the bits-back sandwich between Viterbi and marginal code lengths, and
   a two-part MDL code length with an interior optimum in the vocabulary size.
6. Finance case study: the shared corpus builds rare words from planted prefix/root/suffix morphemes —
   a known answer for how well each tokenizer's boundaries track morphology.

Cross-checks: our BPE merges equal Hugging Face ``tokenizers``' BpeTrainer exactly, and our Viterbi
under SentencePiece's trained piece scores reproduces ``sentencepiece``'s own encodings.

Run (executes the verification harness; sentencepiece ships wheels through CPython 3.13):
    uv run --python 3.13 --with numpy==2.5.4 --with scipy==1.18.1 --with tokenizers==0.23.3 --with sentencepiece==0.2.2 python notebooks/subword-tokenization-bpe-unigram/subword_tokenization_bpe_unigram.py
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "finance-corpus"))
sys.path.insert(0, str(_HERE.parent / "zipf-heaps-vocabulary-statistics"))
import finance_corpus as fc  # noqa: E402
import zipf_heaps_vocabulary_statistics as zhv  # noqa: E402

# ONE stream for every random-instance claim below (STARTER-PROMPT A1).
SEED = 20261023
NEG_INF = -math.inf


# --------------------------------------------------------------------------------------------
# 1. Segmentations: what a subword vocabulary can and cannot encode.
# --------------------------------------------------------------------------------------------


def segmentations(x: str, vocab, max_len: int | None = None):
    """Every segmentation of x into pieces of vocab (brute force; small inputs only)."""
    if not x:
        yield ()
        return
    for j in range(1, len(x) + 1 if max_len is None else min(len(x), max_len) + 1):
        if x[:j] in vocab:
            for rest in segmentations(x[j:], vocab, max_len):
                yield (x[:j], *rest)


def count_segmentations(x: str, vocab=None, max_len: int | None = None) -> int:
    """|Seg_V(x)| by dynamic programming; vocab=None means every substring is a piece."""
    n, L = len(x), len(x) if max_len is None else max_len
    S = [1] + [0] * n
    for j in range(1, n + 1):
        S[j] = sum(S[i] for i in range(max(0, j - L), j) if vocab is None or x[i:j] in vocab)
    return S[n]


def lbonacci(n: int, L: int) -> int:
    """S(0) = 1, S(m) = S(m−1) + … + S(m−L): compositions of n with parts ≤ L."""
    S = [1]
    for m in range(1, n + 1):
        S.append(sum(S[max(0, m - L) : m]))
    return S[n]


def encodable(x: str, vocab) -> bool:
    return count_segmentations(x, vocab, max(map(len, vocab))) > 0


# --------------------------------------------------------------------------------------------
# 2. BPE: greedy merges.
# --------------------------------------------------------------------------------------------


def apply_merge(tokens: list[str], pair: tuple[str, str]) -> tuple[list[str], int]:
    """Replace adjacent (a, b) by the single token a + b, scanning left to right, never overlapping.

    Returns the new token list and r, the number of replacements made. "Never overlapping": once
    tokens i and i+1 are merged, the scan resumes at i+2, so ["a", "a", "a"] with pair ("a", "a")
    becomes ["aa", "a"] with r = 1. Prop 3 says len(new) == len(tokens) − r.
    """
    a, b = pair
    out, i, r = [], 0, 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            out.append(a + b)
            i, r = i + 2, r + 1
        else:
            out.append(tokens[i])
            i += 1
    return out, r


def window_pair_counts(tokens: list[str]) -> Counter:
    """Every adjacent window (t_i, t_{i+1}) — the count BPE ranks by (overlapping runs included)."""
    return Counter(zip(tokens, tokens[1:]))


@dataclass
class MergeModel:
    """A trained merge sequence: alphabet, merges in rank order, and each training word's segmentation."""

    alphabet: tuple[str, ...]
    merges: list[tuple[str, str]]
    segs: dict[str, tuple[str, ...]]
    gains: list[float]  # the score each merge won with (pair count for BPE, ΔL for WordPiece)
    loglik: list[float]  # unigram MLE log-likelihood of the training tokens after k merges, k = 0, 1, …


def _train_merges(word_counts: Counter, n_merges: int, criterion: str) -> MergeModel:
    """Shared incremental trainer. criterion "bpe": highest window count, ties to the smaller pair of
    token ids (characters numbered in codepoint order, then merged tokens in creation order) —
    Hugging Face's BpeTrainer rule. criterion "likelihood": highest exact ΔL (Prop 5)."""
    words = [list(w) for w in word_counts]
    counts = list(word_counts.values())
    alphabet = tuple(sorted({c for w in word_counts for c in w}))
    tid = {c: k for k, c in enumerate(alphabet)}
    pairs: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    tok_counts: Counter = Counter()
    for k, (w, c) in enumerate(zip(words, counts)):
        for p in zip(w, w[1:]):
            pairs[p] += c
            where[p].add(k)
        for t in w:
            tok_counts[t] += c
    n_tokens = sum(tok_counts.values())
    merges, gains = [], []
    sum_xlogx = sum(_xlogx(c) for c in tok_counts.values())
    loglik = [sum_xlogx - _xlogx(n_tokens)]
    for _ in range(n_merges):
        live = [p for p, c in pairs.items() if c > 0]
        if not live:
            break
        if criterion == "bpe":
            best = max(live, key=lambda p: (pairs[p], -tid[p[0]], -tid[p[1]]))
            gain = float(pairs[best])
        else:
            def delta(p):
                m = pairs[p] if p[0] != p[1] else sum(counts[k] * apply_merge(words[k], p)[1] for k in where[p])
                return merge_delta_loglik(tok_counts[p[0]], tok_counts[p[1]], m, n_tokens, p[0] == p[1])

            gain, best = max(((delta(p), p) for p in live), key=lambda dp: (dp[0], -tid[dp[1][0]], -tid[dp[1][1]]))
        a, b = best
        tid[a + b] = len(tid)
        merges.append(best)
        gains.append(gain)
        touched = {a, b, a + b}
        sum_xlogx -= sum(_xlogx(tok_counts[t]) for t in touched)
        for k in list(where.pop(best, ())):
            old = words[k]
            if len(old) < 2:
                continue
            new, r = apply_merge(old, best)
            if r == 0:
                continue
            c = counts[k]
            for p in zip(old, old[1:]):
                pairs[p] -= c
            for p in zip(new, new[1:]):
                pairs[p] += c
                where[p].add(k)
            tok_counts[a] -= r * c
            tok_counts[b] -= r * c
            tok_counts[a + b] += r * c
            n_tokens -= r * c
            words[k] = new
        pairs.pop(best, None)
        sum_xlogx += sum(_xlogx(tok_counts[t]) for t in touched)
        loglik.append(sum_xlogx - _xlogx(n_tokens))
    return MergeModel(alphabet, merges, {w: tuple(s) for w, s in zip(word_counts, words)}, gains, loglik)


def bpe_train(word_counts: Counter, n_merges: int) -> MergeModel:
    return _train_merges(word_counts, n_merges, "bpe")


def wordpiece_train(word_counts: Counter, n_merges: int) -> MergeModel:
    """Schuster & Nakajima's criterion: merge the pair whose unit most increases the training
    likelihood under a unigram model of the current tokens (exact ΔL, Prop 5)."""
    return _train_merges(word_counts, n_merges, "likelihood")


def replay_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Rank-replay encoding: apply merge 1 everywhere, then merge 2, … (Prop 4)."""
    tokens, present = list(word), set(word)
    for a, b in merges:
        if a in present and b in present:
            tokens, r = apply_merge(tokens, (a, b))
            if r:
                present = set(tokens)
    return tokens


def maxmatch_encode(word: str, vocab) -> list[str]:
    """WordPiece's encoder: repeatedly take the longest vocabulary prefix of what remains."""
    out, i, L = [], 0, max(map(len, vocab))
    while i < len(word):
        j = next(j for j in range(min(len(word), i + L), i, -1) if word[i:j] in vocab)
        out.append(word[i:j])
        i = j
    return out


def merge_vocab(model: MergeModel, k: int) -> set[str]:
    return set(model.alphabet) | {a + b for a, b in model.merges[:k]}


# Kozma–Voderholzer's setting: one string, merges counted by actual (non-overlapping) replacements.


def pair_encoding_length(s: str, merges) -> int:
    tokens = list(s)
    for p in merges:
        tokens, _ = apply_merge(tokens, p)
    return len(tokens)


def greedy_utilities(s: str, k: int) -> set[int]:
    """Greedy BPE on one string: each step merges a pair with the most replacements. Returns the
    utility (tokens saved after k merges) of EVERY greedy run — one per way of breaking ties."""
    def go(tokens, k):
        cands = set(zip(tokens, tokens[1:]))
        if k == 0 or not cands:
            return {len(s) - len(tokens)}
        rs = {p: apply_merge(tokens, p)[1] for p in cands}
        top = max(rs.values())
        return set().union(*(go(apply_merge(tokens, p)[0], k - 1) for p in cands if rs[p] == top))

    return go(list(s), k)


def optimal_pair_encoding(s: str, k: int) -> tuple[list[tuple[str, str]], int]:
    """Exhaustive search over all merge sequences of length ≤ k (exponential; tiny strings only)."""
    best: tuple[int, list] = (0, [])

    def go(tokens, seq):
        nonlocal best
        if len(s) - len(tokens) > best[0]:
            best = (len(s) - len(tokens), list(seq))
        if len(seq) == k:
            return
        for p in sorted(set(zip(tokens, tokens[1:]))):
            go(apply_merge(tokens, p)[0], seq + [p])

    go(list(s), [])
    return best[1], best[0]


# --------------------------------------------------------------------------------------------
# 3. WordPiece: the likelihood gain of a merge.
# --------------------------------------------------------------------------------------------


def _xlogx(x: float) -> float:
    return x * math.log(x) if x > 0 else 0.0


def unigram_loglik(tok_counts) -> float:
    """Unigram MLE log-likelihood Σ_t c_t log(c_t / N) = Σ c log c − N log N."""
    cs = [c for c in tok_counts.values() if c > 0] if hasattr(tok_counts, "values") else tok_counts
    return sum(_xlogx(c) for c in cs) - _xlogx(sum(cs))


def merge_delta_loglik(c_a: float, c_b: float, m: float, N: float, same: bool = False) -> float:
    """Exact ΔL when m occurrences of (a, b) become the new token ab (Prop 5). If a = b, each
    replacement consumes two a's: c_a → c_a − 2m."""
    if same:
        return _xlogx(m) + _xlogx(c_a - 2 * m) - _xlogx(c_a) - (_xlogx(N - m) - _xlogx(N))
    return (_xlogx(m) + _xlogx(c_a - m) + _xlogx(c_b - m) - _xlogx(c_a) - _xlogx(c_b)
            - (_xlogx(N - m) - _xlogx(N)))


def merge_delta_leading(c_a: float, c_b: float, m: float, N: float) -> float:
    """The leading term m·[log(mN / (c_a c_b)) − 1]: count-weighted PMI of the pair."""
    return m * (math.log(m * N / (c_a * c_b)) - 1)


def merge_delta_remainder_bound(c_a: float, c_b: float, m: float, N: float) -> tuple[float, float]:
    """ΔL − leading = R_a + R_b − R_N with each R_c ∈ [0, m² / (2(c − m))] (Prop 5's remainder)."""
    def r(c):
        return math.inf if c <= m else m * m / (2 * (c - m))
    return -r(N), r(c_a) + r(c_b)


# --------------------------------------------------------------------------------------------
# 4. Unigram LM: the segmentation lattice.
# --------------------------------------------------------------------------------------------


def viterbi_segment(word: str, logp: dict[str, float], max_len: int) -> tuple[list[str], float]:
    """The maximum-probability segmentation of word under a unigram model (Theorem 2).

    logp maps each vocabulary piece to its log-probability; pieces are at most max_len characters.
    Returns (pieces, total log-probability). If no segmentation exists, returns ([], -inf).
    best[j] = max over i of best[i] + logp[word[i:j]], with best[0] = 0; recover the pieces by
    remembering which i won at each j.
    """
    n = len(word)
    best, back = [0.0] + [NEG_INF] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_len), j):
            t = word[i:j]
            if t in logp and best[i] + logp[t] > best[j]:
                best[j], back[j] = best[i] + logp[t], i
    if best[n] == NEG_INF:
        return [], NEG_INF
    pieces, j = [], n
    while j > 0:
        pieces.append(word[back[j] : j])
        j = back[j]
    return pieces[::-1], best[n]


def _lse(xs: list[float]) -> float:
    m = max(xs)
    return m if m == NEG_INF else m + math.log(sum(math.exp(x - m) for x in xs))


def forward_backward(word: str, logp: dict[str, float], max_len: int):
    """log P(x) = log Σ_seg Π p(t) (Prop 6) and each lattice edge's posterior probability.
    Returns (log Z, [(i, j, piece, posterior), …])."""
    n = len(word)
    alpha = [0.0] + [NEG_INF] * n
    for j in range(1, n + 1):
        terms = [alpha[i] + logp[word[i:j]] for i in range(max(0, j - max_len), j) if word[i:j] in logp]
        alpha[j] = _lse(terms) if terms else NEG_INF
    beta = [NEG_INF] * n + [0.0]
    for i in range(n - 1, -1, -1):
        terms = [logp[word[i:j]] + beta[j] for j in range(i + 1, min(n, i + max_len) + 1) if word[i:j] in logp]
        beta[i] = _lse(terms) if terms else NEG_INF
    logz = alpha[n]
    edges = []
    for i in range(n):
        for j in range(i + 1, min(n, i + max_len) + 1):
            t = word[i:j]
            if t in logp and alpha[i] > NEG_INF and beta[j] > NEG_INF:
                edges.append((i, j, t, math.exp(alpha[i] + logp[t] + beta[j] - logz)))
    return logz, edges


def em_step(word_counts: Counter, logp: dict[str, float], max_len: int) -> tuple[dict[str, float], float]:
    """One EM iteration. E: expected piece counts from forward–backward. M: p(t) ∝ E[c_t], the exact
    maximizer of Q (Theorem 3). Returns the new model and the log-likelihood of the OLD model."""
    exp_counts: Counter = Counter()
    ll = 0.0
    for w, c in word_counts.items():
        logz, edges = forward_backward(w, logp, max_len)
        ll += c * logz
        for _, _, t, q in edges:
            exp_counts[t] += c * q
    total = sum(exp_counts.values())
    return {t: math.log(v / total) for t, v in exp_counts.items() if v > 0}, ll


def corpus_loglik(word_counts: Counter, logp: dict[str, float], max_len: int) -> float:
    return sum(c * forward_backward(w, logp, max_len)[0] for w, c in word_counts.items())


def piece_losses(word_counts: Counter, logp: dict[str, float], max_len: int, keep: set[str]) -> dict[str, float]:
    """SentencePiece's pruning loss (a heuristic): the change in Viterbi log-likelihood if piece t were
    removed and its Viterbi frequency reassigned to t's best alternative segmentation."""
    freq: Counter = Counter()
    for w, c in word_counts.items():
        for t in viterbi_segment(w, logp, max_len)[0]:
            freq[t] += c
    total = sum(freq.values())
    losses = {}
    for t in logp:
        if t in keep:
            continue
        if freq[t] == 0:
            losses[t] = 0.0
            continue
        alt, _ = viterbi_segment(t, {u: v for u, v in logp.items() if u != t}, max_len)
        log_alt_sum = math.log(total + freq[t] * (len(alt) - 1))
        lp_alt = sum(math.log(freq[u] + freq[t]) - log_alt_sum for u in alt)
        losses[t] = freq[t] * (math.log(freq[t] / total) - lp_alt)
    return losses


@dataclass
class UnigramModel:
    logp: dict[str, float]
    max_len: int
    loglik: float  # corpus log-likelihood (nats) after training


def unigram_train(word_counts: Counter, sizes, max_len: int = 10, seed_size: int = 2000,
                  shrink: float = 0.75, em_iters: int = 2) -> dict[int, UnigramModel]:
    """Kudo's Unigram LM, top-down: seed with frequent substrings, alternate EM and pruning, and
    snapshot the model each time the vocabulary reaches one of `sizes` (characters are never pruned)."""
    chars: Counter = Counter()
    subs: Counter = Counter()
    for w, c in word_counts.items():
        for ch in w:
            chars[ch] += c
        for i in range(len(w)):
            for j in range(i + 2, min(len(w), i + max_len) + 1):
                subs[w[i:j]] += c
    seeds = sorted(subs, key=lambda t: (-subs[t] * len(t), t))[:seed_size]
    init = {**{t: subs[t] for t in seeds}, **chars}
    total = sum(init.values())
    logp = {t: math.log(v / total) for t, v in init.items()}
    keep = set(chars)
    out = {}
    for target in sorted(sizes, reverse=True):
        while True:
            for _ in range(em_iters):
                logp, _ = em_step(word_counts, logp, max_len)
            if len(logp) <= target:
                break
            losses = piece_losses(word_counts, logp, max_len, keep)
            n_remove = min(len(logp) - target, max(1, int(len(logp) * (1 - shrink))))
            for t in sorted(losses, key=lambda t: (losses[t], t))[:n_remove]:
                del logp[t]
        out[target] = UnigramModel(dict(logp), max_len, corpus_loglik(word_counts, logp, max_len))
    return out


# --------------------------------------------------------------------------------------------
# 5. Tokenization as compression.
# --------------------------------------------------------------------------------------------

LETTER_BITS = math.log2(27)  # spell a piece: 26 letters + an end marker, uniformly


def vocab_bits(vocab) -> float:
    """L(V): each piece spelled out letter by letter, plus its end marker."""
    return sum((len(t) + 1) * LETTER_BITS for t in vocab)


def data_bits(segs: dict[str, list[str]], word_counts: Counter) -> float:
    """L(D | V): the corpus's tokens under the unigram MLE code of those same tokens."""
    tc: Counter = Counter()
    for w, c in word_counts.items():
        for t in segs[w]:
            tc[t] += c
    return -unigram_loglik(tc) / math.log(2)


def mdl_curve(model: MergeModel) -> np.ndarray:
    """Two-part code length (bits) after k = 0, 1, … merges: the spelled-out vocabulary L(V) plus the
    training tokens under their unigram MLE code, L(D | V) = −log-likelihood / ln 2."""
    lex = np.cumsum([vocab_bits(model.alphabet)] + [vocab_bits([a + b]) for a, b in model.merges])
    return lex - np.array(model.loglik) / math.log(2)


def word_lexicon_bits(word_counts: Counter) -> float:
    """The same two-part code with whole words as the vocabulary (no subwords at all)."""
    return vocab_bits(word_counts) - unigram_loglik(word_counts) / math.log(2)


def tokens_per_word(segs: dict[str, list[str]], word_counts: Counter) -> float:
    return sum(c * len(segs[w]) for w, c in word_counts.items()) / sum(word_counts.values())


# --------------------------------------------------------------------------------------------
# 6. Finance case study: planted morphemes.
# --------------------------------------------------------------------------------------------

VOCAB_SIZES = (50, 100, 200, 400, 800)  # matched |V| for every tokenizer comparison
BPE_MERGES_MAX = 4000  # BPE runs on to near-whole words for the MDL curve (§5)


@lru_cache(maxsize=1)
def finance_words() -> dict:
    """Training words of generate_corpus(300, 0) (Topic 1's corpus) and held-out words of
    generate_corpus(100, 1); the training corpus's planted pseudo-words, recorded by rebuilding it
    with fc._CorpusGen exactly as generate_corpus does."""
    gen = fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)
    fresh = gen.common.fresh
    reserved = set(fresh.used)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    docs = [gen.doc(i, int(t)) for i, t in enumerate(tones)]
    train = Counter(t for d in docs for t in fc._tokens(d.text))
    held = Counter(t for d in fc.generate_corpus(100, 1) for t in fc._tokens(d.text))
    pseudo = frozenset(fresh.used - reserved)
    return {"train": train, "held": held, "pseudo": pseudo, "docs": [d.text for d in docs],
            "lines": [" ".join(fc._tokens(d.text)) for d in docs],
            "oov": Counter({w: c for w, c in held.items() if w not in train})}


def planted_parses(word: str) -> set[tuple[int, ...]]:
    """Every boundary set (character offsets) of a (prefix, root, suffix) decomposition that
    fc._PseudoWords could have produced: prefix + root + ("s" | ""), or [prefix] + root + suffix,
    with the generator's e-drop (lease + ing → leasing)."""
    out = set()
    for prefix in ("",) + fc._PREFIXES:
        if not word.startswith(prefix):
            continue
        for root in fc._ROOTS:
            stem = prefix + root
            for suffix in (("s", "") if prefix else ()) + fc._SUFFIXES:
                s = stem[:-1] if stem.endswith("e") and suffix[:1] in ("a", "e", "i") else stem
                if s + suffix == word:
                    out.add(tuple(b for b in (len(prefix), len(s)) if 0 < b < len(word)))
    return out


def boundaries(pieces) -> set[int]:
    return set(itertools.accumulate(len(p) for p in pieces[:-1]))


def boundary_prf(segs: dict[str, list[str]], gold: dict[str, set[int]]) -> dict[str, float]:
    """Micro precision/recall/F1 of predicted internal boundaries against the planted ones."""
    tp = fp = fn = 0
    for w, g in gold.items():
        pred = boundaries(segs[w])
        tp, fp, fn = tp + len(pred & g), fp + len(pred - g), fn + len(g - pred)
    p, r = tp / (tp + fp), tp / (tp + fn)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}


@lru_cache(maxsize=1)
def trained() -> dict:
    """Every tokenizer, trained once on the training word counts. BPE and WordPiece are trained to
    the largest size; greedy merge sequences are prefix-closed, so the first k merges ARE the k-merge
    model. Unigram is snapshotted on its way down through each size."""
    fw = finance_words()
    n_max = max(VOCAB_SIZES) - len({c for w in fw["train"] for c in w})
    return {"bpe": bpe_train(fw["train"], BPE_MERGES_MAX), "wordpiece": wordpiece_train(fw["train"], n_max),
            "unigram": unigram_train(fw["train"], VOCAB_SIZES)}


def segment_all(kind: str, size: int, words) -> dict[str, list[str]]:
    t = trained()
    if kind == "unigram":
        m = t["unigram"][size]
        return {w: viterbi_segment(w, m.logp, m.max_len)[0] for w in words}
    model = t[kind]
    k = size - len(model.alphabet)
    return {w: replay_encode(w, model.merges[:k]) for w in words}


@lru_cache(maxsize=1)
def finance_summary() -> dict:
    fw = finance_words()
    train, held = fw["train"], fw["held"]
    gold_train = {w: set(next(iter(ps))) for w in fw["pseudo"] if len(ps := planted_parses(w)) == 1}
    gold_oov = {w: set(next(iter(ps))) for w in fw["oov"] if len(ps := planted_parses(w)) == 1}
    out = {"N": sum(train.values()), "V": len(train), "pseudo_types": len(fw["pseudo"]),
           "pseudo_unambiguous": len(gold_train), "held_N": sum(held.values()),
           "oov_types": len(fw["oov"]), "oov_token_rate": sum(fw["oov"].values()) / sum(held.values()),
           "oov_unambiguous": len(gold_oov)}
    for kind in ("bpe", "wordpiece", "unigram"):
        for size in VOCAB_SIZES:
            segs = segment_all(kind, size, set(train) | set(held))
            out[f"{kind}_{size}_tpw_held"] = tokens_per_word(segs, held)
            prf = boundary_prf(segs, gold_train)
            out[f"{kind}_{size}_p_train"], out[f"{kind}_{size}_r_train"] = prf["precision"], prf["recall"]
            out[f"{kind}_{size}_f1_train"] = prf["f1"]
            out[f"{kind}_{size}_f1_oov"] = boundary_prf(segs, gold_oov)["f1"]
    curve = mdl_curve(trained()["bpe"])
    k = int(np.argmin(curve))
    out.update({"mdl_bits_chars": float(curve[0]), "mdl_argmin_merges": k, "mdl_bits_min": float(curve[k]),
                "mdl_vocab_at_min": k + len(trained()["bpe"].alphabet), "mdl_bits_words": word_lexicon_bits(train)})
    return out


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX).
# --------------------------------------------------------------------------------------------

LAB_WORDS = ("subscriptions", "underwriting", "crossfleet", "colevys", "nonroyaltys", "supplierers")
LAB_LATTICE_WORDS = ("colevys", "nonroyaltys", "underwriting")
LAB_POSTERIOR_FLOOR = 0.005  # lattice edges below this posterior are not drawn


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    """Panel A: tokens per held-out word, |V|, and live segmentations, BPE vs. WordPiece, per merge
    count. Panel B: each lattice word's Unigram lattice per vocabulary size — edge posteriors, the
    Viterbi path, and the bits-back sandwich."""
    fw = finance_words()
    t = trained()
    alphabet = len(t["bpe"].alphabet)
    merges_panel = {}
    for size in (alphabet, *VOCAB_SIZES):
        k = size - alphabet
        row = {"merges": k, "vocab": size}
        for kind in ("bpe", "wordpiece"):
            segs = {w: replay_encode(w, t[kind].merges[:k]) for w in set(fw["held"]) | set(LAB_WORDS)}
            row[kind] = {"tpw": round(tokens_per_word(segs, fw["held"]), 3),
                         "segs": {w: segs[w] for w in LAB_WORDS}}
        merges_panel[k] = row
    lattice, unigram_segs = {}, {}
    for size in VOCAB_SIZES:
        m = t["unigram"][size]
        unigram_segs[size] = {w: viterbi_segment(w, m.logp, m.max_len)[0] for w in LAB_WORDS}
        lattice[size] = {}
        for w in LAB_LATTICE_WORDS:
            logz, edges = forward_backward(w, m.logp, m.max_len)
            pieces, best = viterbi_segment(w, m.logp, m.max_len)
            lattice[size][w] = {
                "edges": [[i, j, round(q, 3)] for i, j, _, q in edges if q >= LAB_POSTERIOR_FLOOR],
                "viterbi": pieces, "bits_viterbi": round(-best / math.log(2), 2),
                "bits_marginal": round(-logz / math.log(2), 2),
                "log2_segs": round(math.log2(count_segmentations(w, m.logp, m.max_len)), 2)}
    s = finance_summary()
    # Planted boundaries (character offsets) for the pseudo-words; real words have none.
    planted = {w: sorted(next(iter(ps))) if w in fw["pseudo"] | set(fw["oov"]) and len(ps := planted_parses(w)) == 1
               else None for w in LAB_WORDS}
    f1 = {size: {k: round(s[f"{k}_{size}_f1_train"], 3) for k in ("bpe", "wordpiece", "unigram")} for size in VOCAB_SIZES}
    return {"alphabet": alphabet, "merges": merges_panel, "lattice": lattice, "unigram_segs": unigram_segs,
            "planted": planted, "f1": f1, "words": list(LAB_WORDS), "lattice_words": list(LAB_LATTICE_WORDS)}


# --------------------------------------------------------------------------------------------
# Verification harness — each pedagogical claim of the topic.
# --------------------------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _random_string(rng, n: int, alphabet: str) -> str:
    return "".join(alphabet[i] for i in rng.integers(len(alphabet), size=n))


def test_completeness_iff_alphabet_in_vocab():
    """Prop 1: every string over Σ is encodable iff Σ ⊆ V; decode∘encode is the identity."""
    rng = _rng()
    vocab = {"a", "b", "c", "ab", "bca", "cc"}
    for _ in range(200):
        x = _random_string(rng, int(rng.integers(1, 12)), "abc")
        assert encodable(x, vocab)
        assert all("".join(s) == x for s in segmentations(x, vocab))
    assert not encodable("abc", vocab - {"c"} | {"ab"}) and not encodable("c", vocab - {"c"})
    # On the corpus: every held-out word, OOV included, is encodable by every trained tokenizer.
    fw = finance_words()
    for kind in ("bpe", "unigram"):
        segs = segment_all(kind, 50, fw["oov"])
        assert all("".join(s) == w for w, s in segs.items())


def test_segmentation_count_is_lbonacci():
    """Prop 2: with every substring a piece, |Seg(x)| = 2^{n−1}; with pieces ≤ L it is the L-step
    Fibonacci number — checked against brute-force enumeration."""
    for n in range(1, 13):
        x = "x" * n
        allsubs = {x[i:j] for i in range(n) for j in range(i + 1, n + 1)}
        assert count_segmentations(x) == 2 ** (n - 1) == sum(1 for _ in segmentations(x, allsubs))
        for L in (1, 2, 3, 5):
            assert count_segmentations(x, None, L) == lbonacci(n, L) == sum(1 for _ in segmentations(x, allsubs, L))
    assert count_segmentations("x" * 16) == 2 ** 15


def test_merge_accounting():
    """Prop 3: a merge shortens the sequence by exactly its replacement count r; for a ≠ b, r equals
    the window count (occurrences cannot overlap); for a = b, a run of length ℓ gives ⌊ℓ/2⌋."""
    rng = _rng()
    for _ in range(500):
        toks = list(_random_string(rng, int(rng.integers(2, 20)), "abc"))
        for p in (("a", "b"), ("b", "a"), ("a", "a"), ("c", "c")):
            new, r = apply_merge(toks, p)
            assert len(new) == len(toks) - r and "".join(new) == "".join(toks)
            if p[0] != p[1]:
                assert r == window_pair_counts(toks)[p]
    for ell in range(1, 12):
        new, r = apply_merge(["a"] * ell, ("a", "a"))
        assert r == ell // 2 and window_pair_counts(["a"] * ell)[("a", "a")] == max(0, ell - 1)
    assert apply_merge(["a", "a", "a"], ("a", "a")) == (["aa", "a"], 1)


def test_replay_reproduces_training_segmentation():
    """Prop 4: replaying the first k merges in rank order reproduces every training word's
    segmentation after k training merges — and greedy merge sequences are prefix-closed."""
    fw = finance_words()
    bpe = trained()["bpe"]
    assert all(replay_encode(w, bpe.merges) == list(s) for w, s in bpe.segs.items())
    small = bpe_train(fw["train"], 74)
    assert small.merges == bpe.merges[:74]
    assert all(replay_encode(w, bpe.merges[:74]) == list(s) for w, s in small.segs.items())


def _hf_bpe(lines: list[str], vocab_size: int):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train_from_iterator(lines, trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=0,
                                                       show_progress=False, special_tokens=[]))
    return tok


def test_bpe_matches_hf_tokenizers():
    """Cross-check: our BPE's merge list equals Hugging Face BpeTrainer's exactly (same tie rule),
    and both encoders agree on every held-out word, OOV included."""
    import json

    fw = finance_words()
    bpe = trained()["bpe"]
    tok = _hf_bpe(fw["lines"], len(bpe.alphabet) + len(bpe.merges))
    hf = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
          for m in json.loads(tok.to_str())["model"]["merges"]]
    assert hf == bpe.merges, next((k, a, b) for k, (a, b) in enumerate(zip(hf, bpe.merges)) if a != b)
    for w in fw["held"]:
        assert tok.encode(w).tokens == replay_encode(w, bpe.merges), w


GREEDY_GAP = ("aaaabaaab", 3, 5, 6)  # smallest string (exhaustive over |Σ| ≤ 3, n ≤ 9) where EVERY greedy run loses


def test_greedy_is_not_optimal():
    """Greedy BPE is a heuristic for an APX-complete objective (Kozma & Voderholzer — cited). On
    "aaaabaaab" with 3 merges every greedy run saves 5 tokens; the best merge sequence (ab, a·ab, a·aab)
    saves 6 — greedy's forced first merge aa (3 replacements) shatters the repeated unit "aaab".
    On random strings, every greedy run keeps at least the proved worst-case factor 1/3."""
    s, k, ug, uo = GREEDY_GAP
    assert greedy_utilities(s, k) == {ug} and optimal_pair_encoding(s, k)[1] == uo
    assert pair_encoding_length(s, [("a", "b"), ("a", "ab"), ("a", "aab")]) == len(s) - uo
    assert apply_merge(list(s), ("a", "a"))[1] == 3 == max(apply_merge(list(s), p)[1] for p in set(zip(s, s[1:])))
    rng = _rng()
    worst, strict = 1.0, 0
    for _ in range(300):
        x = _random_string(rng, int(rng.integers(6, 13)), "abc"[: int(rng.integers(2, 4))])
        for kk in (2, 3):
            us, uo_ = greedy_utilities(x, kk), optimal_pair_encoding(x, kk)[1]
            assert max(us) <= uo_ and 3 * min(us) >= uo_
            worst, strict = min(worst, min(us) / uo_), strict + (max(us) < uo_)
    assert worst < 1 and strict > 0


def test_wordpiece_delta_exact():
    """Prop 5: merge_delta_loglik equals the log-likelihood recomputed from the token counts before
    and after each of the first WordPiece merges (including any a = a merge)."""
    fw = finance_words()
    wp = trained()["wordpiece"]
    words = {w: list(w) for w in fw["train"]}
    for p, gain in list(zip(wp.merges, wp.gains))[:40]:
        before: Counter = Counter()
        after: Counter = Counter()
        for w, c in fw["train"].items():
            for t in words[w]:
                before[t] += c
            words[w] = apply_merge(words[w], p)[0]
            for t in words[w]:
                after[t] += c
        assert abs((unigram_loglik(after) - unigram_loglik(before)) - gain) < 1e-6 * max(1, abs(gain)), p


def test_wordpiece_leading_term_bound():
    """Prop 5's remainder: ΔL − m[log(mN/(c_a c_b)) − 1] lies in [−m²/(2(N−m)), m²/(2(c_a−m)) + m²/(2(c_b−m))];
    random instances and the corpus's candidate pairs."""
    rng = _rng()
    for _ in range(2000):
        c_a, c_b = (int(v) for v in rng.integers(2, 5000, size=2))
        m = int(rng.integers(1, min(c_a, c_b)))
        N = c_a + c_b + int(rng.integers(0, 10**6))
        lo, hi = merge_delta_remainder_bound(c_a, c_b, m, N)
        rem = merge_delta_loglik(c_a, c_b, m, N) - merge_delta_leading(c_a, c_b, m, N)
        assert lo - 1e-9 <= rem <= hi + 1e-9, (c_a, c_b, m, N, rem, lo, hi)


def test_wordpiece_and_bpe_disagree():
    """Same corpus, different criterion: the likelihood criterion's merge order departs from BPE's at
    the third merge — BPE takes the frequent (e, r); ΔL prefers the cohesive (an, d), whose parts
    rarely occur apart (higher PMI)."""
    t = trained()
    k = next(k for k, (a, b) in enumerate(zip(t["bpe"].merges, t["wordpiece"].merges)) if a != b)
    assert k == 2 and t["bpe"].merges[2] == ("e", "r") and t["wordpiece"].merges[2] == ("an", "d")


def test_maxmatch_differs_from_replay_and_viterbi():
    """The three encoders are different functions of the same vocabulary: on the corpus vocabularies,
    MaxMatch (longest prefix first) disagrees with rank replay and with Viterbi on some words."""
    fw = finance_words()
    t = trained()
    vocab = merge_vocab(t["wordpiece"], 200 - 26)
    diff = [w for w in fw["held"] if maxmatch_encode(w, vocab) != replay_encode(w, t["wordpiece"].merges[: 200 - 26])]
    assert diff
    um = t["unigram"][200]
    diff_u = [w for w in fw["held"] if maxmatch_encode(w, set(um.logp)) != viterbi_segment(w, um.logp, um.max_len)[0]]
    assert diff_u


def _random_unigram(rng, alphabet="abc", n_pieces=10, max_len=3):
    pieces = set(alphabet)
    while len(pieces) < n_pieces:
        pieces.add(_random_string(rng, int(rng.integers(2, max_len + 1)), alphabet))
    w = rng.dirichlet(np.ones(len(pieces)))
    return {t: math.log(v) for t, v in zip(sorted(pieces), w)}


def test_viterbi_matches_brute_force():
    """Theorem 2: the DP's score equals the best of all enumerated segmentations, and its pieces
    concatenate to the word and achieve that score."""
    rng = _rng()
    for _ in range(300):
        logp = _random_unigram(rng)
        x = _random_string(rng, int(rng.integers(1, 11)), "abc")
        pieces, score = viterbi_segment(x, logp, 3)
        brute = max(sum(logp[t] for t in s) for s in segmentations(x, logp, 3))
        assert abs(score - brute) < 1e-9 and "".join(pieces) == x
        assert abs(sum(logp[t] for t in pieces) - score) < 1e-9
    assert viterbi_segment("abd", {"a": -1.0, "b": -1.0}, 1) == ([], NEG_INF)


def test_forward_matches_brute_force():
    """Prop 6: the forward recursion equals log Σ over all enumerated segmentations."""
    rng = _rng()
    for _ in range(300):
        logp = _random_unigram(rng)
        x = _random_string(rng, int(rng.integers(1, 11)), "abc")
        logz, _ = forward_backward(x, logp, 3)
        brute = math.log(sum(math.exp(sum(logp[t] for t in s)) for s in segmentations(x, logp, 3)))
        assert abs(logz - brute) < 1e-9


def test_edge_posteriors_cover_each_char():
    """Every segmentation covers each character exactly once, so the posteriors of the edges
    spanning any position sum to 1 (random instances and the lab's lattices)."""
    rng = _rng()
    cases = [(_random_string(rng, 9, "abc"), _random_unigram(rng), 3) for _ in range(100)]
    um = trained()["unigram"]
    cases += [(w, um[s].logp, um[s].max_len) for s in VOCAB_SIZES for w in LAB_LATTICE_WORDS]
    for x, logp, L in cases:
        _, edges = forward_backward(x, logp, L)
        for pos in range(len(x)):
            assert abs(sum(q for i, j, _, q in edges if i <= pos < j) - 1) < 1e-9


def test_em_is_monotone():
    """Theorem 3: with a fixed vocabulary, EM never lowers the marginal log-likelihood — random
    instances, and the corpus at |V| = 200."""
    rng = _rng()
    for _ in range(30):
        logp = _random_unigram(rng, n_pieces=12)
        data = Counter(_random_string(rng, int(rng.integers(2, 9)), "abc") for _ in range(40))
        prev = NEG_INF
        for _ in range(8):
            logp, ll = em_step(data, logp, 3)
            assert ll >= prev - 1e-9, (ll, prev)
            prev = ll
    m = trained()["unigram"][200]
    logp, prev = m.logp, NEG_INF
    for _ in range(4):
        logp, ll = em_step(finance_words()["train"], logp, m.max_len)
        assert ll >= prev - 1e-6
        prev = ll


SP_VOCAB = 400


@lru_cache(maxsize=1)
def sentencepiece_model():
    import io

    import sentencepiece as spm

    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(finance_words()["lines"]), model_writer=buf,
                                   model_type="unigram", vocab_size=SP_VOCAB, character_coverage=1.0,
                                   normalization_rule_name="identity", minloglevel=2)
    return spm.SentencePieceProcessor(model_proto=buf.getvalue())


def test_viterbi_matches_sentencepiece():
    """Cross-check: given SentencePiece's trained piece scores, our Viterbi reproduces sentencepiece's
    own segmentation of every held-out word (OOV included)."""
    sp = sentencepiece_model()
    scores = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())
              if not (sp.is_control(i) or sp.is_unknown(i))}
    L = max(map(len, scores))
    for w in finance_words()["held"]:
        assert viterbi_segment("▁" + w, scores, L)[0] == sp.encode(w, out_type=str), w


def test_bits_back_sandwich():
    """Prop 7: −log P(x) ≤ −log P(seg*) ≤ −log P(x) + log|Seg(x)| for every held-out word at every
    Unigram size; the Viterbi code wastes at most log|Seg(x)| over the marginal code."""
    fw = finance_words()
    for size, m in trained()["unigram"].items():
        for w in fw["held"]:
            logz, _ = forward_backward(w, m.logp, m.max_len)
            _, best = viterbi_segment(w, m.logp, m.max_len)
            n_seg = count_segmentations(w, m.logp, m.max_len)
            assert -logz <= -best + 1e-9 <= -logz + math.log(n_seg) + 2e-9, (size, w)


def test_mdl_curve_is_exact():
    """The incrementally tracked log-likelihood equals a fresh recount at sampled merge counts."""
    bpe, fw = trained()["bpe"], finance_words()
    for k in (0, 1, 50, 777, 2000):
        segs = {w: replay_encode(w, bpe.merges[:k]) for w in fw["train"]}
        assert abs(data_bits(segs, fw["train"]) + bpe.loglik[k] / math.log(2)) < 1e-6, k


def test_mdl_has_interior_optimum():
    """The two-part code L(V) + L(D|V) along BPE's merge path is minimized strictly between characters
    and whole words: too few pieces make the data long, too many make the vocabulary long."""
    s = finance_summary()
    assert 0 < s["mdl_argmin_merges"] < BPE_MERGES_MAX - 1
    assert s["mdl_bits_min"] < s["mdl_bits_words"] < s["mdl_bits_chars"]


def test_subwords_close_the_vocabulary():
    """Topic 1's open-vocabulary remark, answered: the word vocabulary misses held-out words; every
    subword tokenizer encodes all of them, and its token distribution has almost no hapaxes."""
    fw = finance_words()
    s = finance_summary()
    assert s["oov_types"] > 0 and s["oov_token_rate"] > 0
    words_hapax = float(np.mean(zhv.rank_frequency([w for w, c in fw["train"].items() for _ in range(c)]) == 1))
    segs = segment_all("bpe", 400, fw["train"])
    tok_hapax = float(np.mean(zhv.rank_frequency([t for w, c in fw["train"].items() for _ in range(c) for t in segs[w]]) == 1))
    assert words_hapax > 0.5 > 0.05 > tok_hapax, (words_hapax, tok_hapax)


def _f1_at_tpw(s: dict, kind: str, tpw: float) -> float:
    """kind's training boundary F1 linearly interpolated to a given held-out tokens-per-word."""
    xs = [s[f"{kind}_{v}_tpw_held"] for v in VOCAB_SIZES][::-1]  # increasing
    return float(np.interp(tpw, xs, [s[f"{kind}_{v}_f1_train"] for v in VOCAB_SIZES][::-1]))


def test_unigram_tracks_planted_morphemes():
    """The finance case study's claim. At |V| = 200 and 400, Unigram's boundaries match the planted
    prefix/root/suffix boundaries far better than BPE's — at matched |V|, and also at matched
    compression (BPE's F1 interpolated to Unigram's tokens per word), on training pseudo-words and on
    held-out OOV words alike. By |V| = 800 the three tokenizers are within 0.08 of one another."""
    s = finance_summary()
    for v in (200, 400):
        assert s[f"unigram_{v}_f1_train"] - s[f"bpe_{v}_f1_train"] >= 0.15
        assert s[f"unigram_{v}_f1_oov"] - s[f"bpe_{v}_f1_oov"] >= 0.15
        u_tpw = s[f"unigram_{v}_tpw_held"]
        assert s[f"unigram_{v}_f1_train"] - _f1_at_tpw(s, "bpe", u_tpw) >= 0.1, (v, _f1_at_tpw(s, "bpe", u_tpw))
    top = [s[f"{k}_800_f1_train"] for k in ("bpe", "wordpiece", "unigram")]
    assert max(top) - min(top) < 0.08, top


def test_planted_morphemes_parse_uniquely():
    """Every recorded pseudo-word has at least one generator parse, and almost all have exactly one
    boundary set — so the planted boundaries are well defined."""
    fw = finance_words()
    parses = {w: planted_parses(w) for w in fw["pseudo"]}
    assert all(parses.values())
    assert sum(len(p) == 1 for p in parses.values()) >= 0.95 * len(parses)
    assert {w for w in fw["oov"] if planted_parses(w)} == set(fw["oov"])


def test_finance_rebuild_matches_corpus():
    """The rebuild that records pseudo-words is token-identical to generate_corpus(300, 0)."""
    assert finance_words()["docs"] == [d.text for d in fc.generate_corpus(300, 0)]


def test_viz_payload():
    p = viz_payload()
    rows = [p["merges"][k] for k in sorted(p["merges"])]
    for kind in ("bpe", "wordpiece"):
        tpw = [r[kind]["tpw"] for r in rows]
        assert all(np.diff(tpw) < 0), (kind, tpw)  # more merges, fewer tokens per word
        for r in rows:
            assert all("".join(r[kind]["segs"][w]) == w for w in p["words"])
    for size, words in p["lattice"].items():
        for w, d in words.items():
            assert d["bits_marginal"] <= d["bits_viterbi"] <= d["bits_marginal"] + d["log2_segs"] + 0.01
            assert "".join(d["viterbi"]) == w
    assert p["planted"]["nonroyaltys"] == [3, 10] and p["planted"]["subscriptions"] is None
    assert all("".join(seg) == w for segs in p["unigram_segs"].values() for w, seg in segs.items())


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "subword-tokenization-lab.json"


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    import json

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
    if not failed:
        print("\nFinance summary (train generate_corpus(300, 0), held out generate_corpus(100, 1)):")
        for k, v in finance_summary().items():
            print(f"  {k:28} {v:.4f}" if isinstance(v, float) else f"  {k:28} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
