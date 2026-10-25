"""Decoding as search: greedy and beam search — the reference implementation for formalNLP's fourteenth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Decoding as a shortest path: with edge weights −log p ≥ 0 the mode of a language model is the cheapest
   root-to-</s> path in its prefix tree. Uniform-cost search pops finished strings in order of probability, so the
   first one it pops is the mode (Thm 1), after at most (2 + E|Y|)/p(y*) pops. An order-n Markov model's prefixes
   recombine onto its states, so its mode is a shortest path in a finite graph (Dijkstra, Viterbi).
2. Greedy and beam search are uniform-cost search with a pruned frontier. Neither has an approximation guarantee:
   greedy's string can be ε times as probable as the mode, beam(k) misses a mode that beam(k + 1) finds, and beam(k + 1)
   can return a strictly worse string than beam(k) (Thm 2).
3. A certificate: if beam search's best finished string scores at least every prefix it pruned and every prefix
   still open, it is the global mode (Thm 3). Under length normalization the same holds with a length cap.
4. Length bias: no completion of a prefix beats stopping once </s> is the likeliest next token, and p(y) ≤ q^|y| when
   every step's probability is at most q (Thm 4). Length normalization breaks the monotonicity Thms 1 and 3 rest on,
   and without a length cap its optimum need not exist: a cycle of low mean cost wins in the limit. Greedy decoding is
   the exact maximizer of log p − λ·R_greedy for large λ (Meister et al.'s regularizer at k = 1).
5. Search error against model error on the corpus: Topic 3's trigram (exact by Dijkstra) and a small causal decoder
   (Topic 13's, exact by best-first search with no recombination). As k grows, search error falls and outputs shrink
   toward the mode, which is shorter than the text the prompts came from.
6. Non-termination: greedy decoding on an order-n model is eventually periodic, so it reaches </s> within |V|^(n−1)
   steps or never (Thm 5). A consistent model (p(</s>) > 0 everywhere) can still make beam(k) run forever.
7. Finance case study, the mode is boilerplate: on MD&A prompts the unnormalized mode stops early, and length
   normalization pulls outputs toward the corpus's most predictable template sentences.

Cross-checks: uniform-cost search against brute-force enumeration on random trees, and graph search against tree
search on the trigram; every decoder score against one full forward pass of the decoded string (float64, ≤ 1e-9).

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/beam-search-decoding/beam_search_decoding.py
"""

from __future__ import annotations

import heapq
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _dep in ("n-gram-models-smoothing", "perplexity-cross-entropy", "scaling-laws"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import n_gram_models_smoothing as ng  # noqa: E402 — module objects: their test_* stay out of this harness
import perplexity_cross_entropy as pce  # noqa: E402
import scaling_laws as sl  # noqa: E402

ak, fc, nlm = sl.ak, sl.fc, sl.nlm

# ONE stream per random-instance claim and per trained model, each from default_rng([SEED, k]); k ≥ 1700 keeps this
# module's streams apart from Topics 5–13.
SEED = sl.SEED
BOS, EOS, UNK = ng.BOS, ng.EOS, ng.UNK
Prefix = tuple[int, ...]


def _r(x, nd: int = 3):
    return ak._r(x, nd)


# --------------------------------------------------------------------------------------------
# 0. Models as next-token oracles.
# --------------------------------------------------------------------------------------------
# Every search below sees a model only through three calls: next_logp(prefixes) → (n, A) log-probabilities over the
# model's alphabet (−inf allowed), ranked(prefix) → (that row, its finite entries best-first), and state(prefix), a
# hashable Markov state when prefixes with equal states have equal futures (None otherwise). Prefixes are tuples of
# token ids that start with the model's root (the prompt); the alphabet includes </s> at index `eos`.


class TableLM:
    """A finite language model given by its next-token distribution at every open prefix. Strings end at </s>;
    the table must give every reachable open prefix a row."""

    def __init__(self, alphabet: tuple[str, ...], rows: dict[Prefix, np.ndarray], root: Prefix = ()):
        self.alphabet, self.rows, self.root = tuple(alphabet), rows, root
        self.eos = self.alphabet.index(EOS)
        self._ranked: dict[Prefix, tuple[np.ndarray, np.ndarray]] = {}

    def logp(self, h: Prefix) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self.rows[h])

    def next_logp(self, prefixes) -> np.ndarray:
        return np.stack([self.logp(h) for h in prefixes])

    def ranked(self, h: Prefix):
        hit = self._ranked.get(h)
        if hit is None:
            lp = self.logp(h)
            order = np.argsort(-lp, kind="stable")
            hit = self._ranked[h] = (lp, order[np.isfinite(lp[order])])
        return hit

    def state(self, h: Prefix):
        return None

    def words(self, toks) -> list[str]:
        return [self.alphabet[i] for i in toks]


def table_lm(spec: dict[str, dict[str, float]], root: str = "") -> TableLM:
    """A TableLM from readable rows: {"revenue rose": {"sharply": 0.32, …, "</s>": …}, …}. Prefix keys are space-joined
    words after the root; every row must sum to 1."""
    alphabet = tuple(sorted({w for row in spec.values() for w in row} | {EOS}, key=lambda w: (w != EOS, w)))
    ix = {w: i for i, w in enumerate(alphabet)}
    rows = {}
    for key, row in spec.items():
        assert abs(sum(row.values()) - 1) < 1e-12, key
        p = np.zeros(len(alphabet))
        for w, q in row.items():
            p[ix[w]] = q
        rows[tuple(ix[w] for w in key.split())] = p
    lm = TableLM(alphabet, rows)
    lm.label = root
    return lm


def random_tree(rng: np.random.Generator, m: int, depth: int, conc: float = 0.5) -> TableLM:
    """m words plus </s>; Dirichlet(conc) next-token distributions at every open prefix shorter than `depth`, and a
    forced </s> at depth (so the model is finite and tie-free with probability 1)."""
    alphabet = (EOS,) + tuple(f"w{i}" for i in range(m))
    rows: dict[Prefix, np.ndarray] = {}
    frontier: list[Prefix] = [()]
    while frontier:
        h = frontier.pop()
        if len(h) == depth:
            rows[h] = np.eye(m + 1)[0]
            continue
        rows[h] = rng.dirichlet(np.full(m + 1, conc))
        frontier += [h + (j,) for j in range(1, m + 1)]
    return TableLM(alphabet, rows)


class StationaryLM:
    """The same next-token distribution after every prefix: |Y| is geometric with stop probability p(</s>)."""

    def __init__(self, probs: dict[str, float]):
        self.alphabet = tuple([EOS] + sorted(w for w in probs if w != EOS))
        self.eos, self.root = 0, ()
        with np.errstate(divide="ignore"):
            self.row = np.log(np.array([probs.get(w, 0.0) for w in self.alphabet]))
        order = np.argsort(-self.row, kind="stable")
        self._order = order[np.isfinite(self.row[order])]

    def next_logp(self, prefixes) -> np.ndarray:
        return np.tile(self.row, (len(prefixes), 1))

    def ranked(self, h):
        return self.row, self._order

    def state(self, h):
        return ()  # one state: every prefix has the same future


def enumerate_strings(lm: TableLM, h: Prefix | None = None) -> list[tuple[float, Prefix]]:
    """Every finished string below h (default the root) with its log-probability, by depth-first enumeration."""
    h = lm.root if h is None else h
    out, stack = [], [(h, 0.0)]
    while stack:
        g, s = stack.pop()
        lp = lm.logp(g)
        for j in np.flatnonzero(np.isfinite(lp)):
            if j == lm.eos:
                out.append((s + lp[j], g[len(h):] + (j,)))
            else:
                stack.append((g + (j,), s + lp[j]))
    return out


def brute_force_mode(lm: TableLM, h: Prefix | None = None) -> tuple[float, Prefix]:
    return max(enumerate_strings(lm, h), key=lambda t: (t[0], [-x for x in t[1]]))


def expected_length(lm: TableLM) -> float:
    """E|Y| = Σ_{t ≥ 1} P(|Y| ≥ t) = the total probability of the open prefixes below the root (|Y| counts the words
    before </s>)."""
    tot, stack = 0.0, [(lm.root, 1.0)]
    while stack:
        g, p = stack.pop()
        row = lm.rows[g]
        for j in np.flatnonzero(row):
            if j != lm.eos:
                tot += p * row[j]
                stack.append((g + (j,), p * row[j]))
    return tot


# --------------------------------------------------------------------------------------------
# 1. Decoding as a shortest path: uniform-cost search.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Found:
    tokens: Prefix | None  # the generated tokens, ending in </s> (None: no finished string within the limits)
    logp: float  # its log-probability given the root
    pops: int  # heap pops, the search's work


def uniform_cost(lm, root: Prefix, *, max_len: int = 10**9, recombine: bool = False, cap: int = 10**7) -> Found:
    """Thm 1: best-first search on cost −log p. Children are generated lazily, best first: popping a node pushes its
    best child and its next sibling, so the heap holds O(pops) entries however large the alphabet. With recombine, a
    second prefix reaching a Markov state already expanded is not expanded again (Dijkstra on the state graph)."""
    heap: list = [(0.0, 0, root, None)]
    seen, pops, tick = set(), 0, 0
    while heap:
        c, _, h, sib = heapq.heappop(heap)
        pops += 1
        if sib is not None:  # the next sibling of h is never cheaper than h
            pc, parent, r = sib
            lp, order = lm.ranked(parent)
            if r + 1 < len(order):
                j = order[r + 1]
                tick += 1
                heapq.heappush(heap, (pc - lp[j], tick, parent + (j,), (pc, parent, r + 1)))
        if len(h) > len(root) and h[-1] == lm.eos:
            return Found(h[len(root):], -c, pops)
        if len(h) - len(root) >= max_len or pops > cap:
            continue
        if recombine:
            key = lm.state(h)
            if key in seen:
                continue
            seen.add(key)
        lp, order = lm.ranked(h)
        if len(order):
            tick += 1
            heapq.heappush(heap, (c - lp[order[0]], tick, h + (order[0],), (c, h, 0)))
    return Found(None, -math.inf, pops)


def pop_bound(lm: TableLM) -> float:
    """Thm 1's work bound (2 + E|Y|)/p(y*): every pop before the mode has cost ≤ −log p(y*), and at most
    P(|Y| ≥ t)/p(y*) prefixes of length t (and 1/p(y*) finished strings) are that probable."""
    return (2 + expected_length(lm)) / math.exp(brute_force_mode(lm)[0])


# --------------------------------------------------------------------------------------------
# 2–3. Greedy and beam search, and the certificate.
# --------------------------------------------------------------------------------------------


def length_penalty(n: int, alpha: float) -> float:
    """Wu et al. (2016)'s lp(Y) = ((5 + |Y|)/6)^α, with |Y| the generated tokens including </s>; α = 0 is plain MAP."""
    return ((5 + n) / 6) ** alpha


@dataclass(frozen=True)
class Beam:
    tokens: Prefix | None  # the returned string, ending in </s> (None: nothing finished)
    logp: float  # its log-probability
    score: float  # its normalized score logp / lp(|y|)
    steps: int
    certified: bool  # Thm 3 (α = 0) or its length-capped form (α > 0) holds: the string is the exact optimum


def beam_search_multi(lm, root: Prefix, k: int, *, max_len: int, alphas: tuple[float, ...]) -> tuple[Beam, ...]:
    """Beam(k) for several normalizations at once. At each step every child of every open prefix is a candidate; the k
    most probable are kept, and those ending in </s> move to the finished set. All open prefixes have the same length,
    so normalization never changes which are kept: the runs for different α share one search and differ only in when
    they stop and which finished string they return. k = 1 is greedy decoding.

    Stopping. At α = 0 an extension never raises a score, so search stops once the best finished string scores at
    least the best open prefix. At α > 0 a completion of an open prefix h scores at most logp(h)/lp(max_len) (the
    log-probability can only fall, and lp is largest at the cap), so search stops once the best finished score beats
    that bound.

    Certificate (Thm 3). best_pruned is the best candidate ever dropped. Every string is a finished member of the beam,
    or extends a pruned candidate, or extends a prefix open at exit, so the result is the exact optimum (over strings of
    at most max_len tokens) when its score is at least the bound for the other two."""
    caps = [length_penalty(max_len, a) for a in alphas]
    open_: list[tuple[float, Prefix]] = [(0.0, root)]
    best: list = [None] * len(alphas)  # per α: (score, logp, tokens)
    done: list[Beam | None] = [None] * len(alphas)
    best_pruned, steps = -math.inf, 0

    def result(i: int) -> Beam:
        if best[i] is None:
            return Beam(None, -math.inf, -math.inf, steps, False)
        others = max([best_pruned] + [s for s, _ in open_])
        return Beam(best[i][2], best[i][1], best[i][0], steps, best[i][0] >= others / caps[i])

    while open_ and steps < max_len and any(d is None for d in done):
        steps += 1
        S = np.array([s for s, _ in open_])
        tot = (S[:, None] + lm.next_logp([h for _, h in open_])).ravel()
        A = tot.size // len(open_)
        m = min(k + 1, int(np.isfinite(tot).sum()))
        if m == 0:
            open_ = []
            break
        idx = np.argpartition(-tot, m - 1)[:m] if m < tot.size else np.arange(tot.size)
        idx = idx[np.lexsort((idx, -tot[idx]))][:m]
        if len(idx) > k:
            best_pruned = max(best_pruned, float(tot[idx[k]]))
        nxt = []
        for i in idx[:k]:
            s, h = float(tot[i]), open_[i // A][1] + (int(i % A),)
            if h[-1] != lm.eos:
                nxt.append((s, h))
                continue
            n = len(h) - len(root)
            for a, alpha in enumerate(alphas):
                cand = (s / length_penalty(n, alpha), s, h[len(root):])
                if done[a] is None and (best[a] is None or cand[0] > best[a][0]):
                    best[a] = cand
        open_ = nxt
        for a in range(len(alphas)):
            if done[a] is None and best[a] is not None and (not open_ or best[a][0] >= open_[0][0] / caps[a]):
                done[a] = result(a)
    return tuple(d if d is not None else result(a) for a, d in enumerate(done))


def beam_search(lm, root: Prefix, k: int, *, max_len: int, alpha: float = 0.0) -> Beam:
    return beam_search_multi(lm, root, k, max_len=max_len, alphas=(alpha,))[0]


def greedy(lm, root: Prefix, *, max_len: int) -> Beam:
    return beam_search(lm, root, 1, max_len=max_len)


def greedy_trap(eps: float) -> TableLM:
    """Thm 2(a): "a" (0.6) leads to m equally likely words, "b" (0.4) ends at once. Greedy takes "a" and returns a
    string of probability 0.6/m; the mode "b </s>" has 0.4. The ratio 1.5/m is below ε once m > 1.5/ε."""
    m = int(math.floor(1.5 / eps)) + 1
    spec = {"": {"a": 0.6, "b": 0.4}, "a": {f"x{i:03d}": 1 / m for i in range(m)}, "b": {EOS: 1.0}}
    spec.update({f"a x{i:03d}": {EOS: 1.0} for i in range(m)})
    return table_lm(spec)


def beam_trap(k: int) -> TableLM:
    """Thm 2(b): k decoys, each a little likelier than "a", fan out into two words each; "a" ends at once. Beam(k)
    keeps the k decoys and prunes "a"; beam(k + 1) keeps "a" and returns the mode "a </s>"."""
    pa = 0.9 / (k + 1)
    spec = {"": {"a": pa, **{f"d{i}": (1 - pa) / k for i in range(k)}}, "a": {EOS: 1.0}}
    for i in range(k):
        spec[f"d{i}"] = {"u": 0.5, "v": 0.5}
        spec[f"d{i} u"] = spec[f"d{i} v"] = {EOS: 1.0}
    return table_lm(spec)


def nonmonotone_beam() -> TableLM:
    """Thm 2(c): greedy takes "a" (0.6) and ends with probability 0.06. Beam(2) keeps "a" and "b" at step 1, and
    "b"'s two children (0.2 each) push out all of "a"'s (0.06 each) at step 2; "b"'s line then fans out into 50 words,
    and beam(2) returns 0.004. Beam(3) keeps one of "a"'s children and returns 0.06 again."""
    spec = {"": {"a": 0.6, "b": 0.4}, "a": {f"x{i}": 0.1 for i in range(10)}, "b": {"c": 0.5, "d": 0.5}}
    spec.update({f"a x{i}": {EOS: 1.0} for i in range(10)})
    for c in "cd":
        spec[f"b {c}"] = {f"z{i:02d}": 0.02 for i in range(50)}
        spec.update({f"b {c} z{i:02d}": {EOS: 1.0} for i in range(50)})
    return table_lm(spec)


def lab_tree() -> TableLM:
    """The lab's search tree: after "revenue", greedy takes "rose" (0.45) into a fan of hedges; the mode goes through
    "was" (0.35) to "flat"."""
    return table_lm({
        "": {"rose": 0.45, "was": 0.35, "fell": 0.20},
        "rose": {"sharply": 0.32, "slightly": 0.28, "on": 0.22, "in": 0.18},
        "was": {"flat": 0.80, "lower": 0.20},
        "fell": {"sharply": 0.55, EOS: 0.45},
        "rose sharply": {EOS: 0.60, "on": 0.40},
        "rose slightly": {EOS: 0.70, "on": 0.30},
        "rose on": {"demand": 0.50, "pricing": 0.50},
        "rose in": {"europe": 0.50, "asia": 0.50},
        "was flat": {EOS: 0.85, "on": 0.15},
        "was lower": {EOS: 1.00},
        "fell sharply": {EOS: 1.00},
        "rose sharply on": {"demand": 1.00},
        "rose slightly on": {"pricing": 1.00},
        "was flat on": {"pricing": 1.00},
        **{f"rose on {w}": {EOS: 1.0} for w in ("demand", "pricing")},
        **{f"rose in {w}": {EOS: 1.0} for w in ("europe", "asia")},
        "rose sharply on demand": {EOS: 1.0}, "rose slightly on pricing": {EOS: 1.0}, "was flat on pricing": {EOS: 1.0},
    }, root="revenue")


# --------------------------------------------------------------------------------------------
# 4. Length bias, normalization, and the greedy regularizer.
# --------------------------------------------------------------------------------------------


def stop_dominates(lm: TableLM) -> tuple[int, int]:
    """Thm 4(a) on every open prefix h of a finite model: when </s> is h's likeliest next token, h·</s> is h's best
    completion. Returns (prefixes where </s> is the argmax, prefixes among them where the claim holds)."""
    n = ok = 0
    for h, row in lm.rows.items():
        if row.argmax() == lm.eos:
            n += 1
            ok += brute_force_mode(lm, h)[1] == (lm.eos,)
    return n, ok


def step_ceiling(lm: TableLM) -> float:
    """q: the largest probability any non-</s> step has anywhere in the model (Thm 4(b)'s constant)."""
    return max(float(np.delete(row, lm.eos).max()) for row in lm.rows.values())


def normalized_chain(p_word: float, n: int, alpha: float = 1.0) -> float:
    """The normalized score of w^n </s> under the stationary model p(w) = p_word, p(</s>) = 1 − p_word."""
    return (n * math.log(p_word) + math.log(1 - p_word)) / length_penalty(n + 1, alpha)


def greedy_regret(lm: TableLM, toks: Prefix, h: Prefix | None = None) -> float:
    """R_greedy(y) = Σ_t (u_t(y_t) − min_w u_t(w))², u_t the surprisal −log p(· | y_<t): Meister et al. (2020)'s Eq. 11,
    the k = 1 case of their beam-search regularizer. It is ≥ 0, and zero exactly on the greedy string (tie-free
    models)."""
    h = lm.root if h is None else h
    tot = 0.0
    for j in toks:
        lp = lm.logp(h)
        tot += float(lp.max() - lp[j]) ** 2
        h = h + (j,)
    return tot


def regularized_mode(lm: TableLM, lam: float) -> tuple[float, Prefix]:
    return max(((s - lam * greedy_regret(lm, y), y) for s, y in enumerate_strings(lm)), key=lambda t: t[0])


def greedy_threshold(lm: TableLM) -> float:
    """λ₀ = max over strings y ≠ greedy of (log p(y) − log p(greedy))/R(y), floored at 0: above it the regularized
    objective's unique maximizer is the greedy string."""
    g = greedy(lm, lm.root, max_len=10**6)
    out = 0.0
    for s, y in enumerate_strings(lm):
        R = greedy_regret(lm, y)
        if y != g.tokens:
            assert R > 0
            out = max(out, (s - g.logp) / R)
    return out


# --------------------------------------------------------------------------------------------
# 5. The corpus engines: Topic 3's trigram and a small causal decoder.
# --------------------------------------------------------------------------------------------


class TrigramLM:
    """Topic 3's interpolated Kneser–Ney trigram (Topic 4's vectorized form), conditioned at each step on not
    emitting <unk>: q(w | u) = p(w | u)/(1 − p(<unk> | u)). The Markov state is the last two words."""

    def __init__(self):
        self.vec = pce.word_vector("ikn")
        self.alphabet = pce.word_alphabet()
        self.index = {w: i for i, w in enumerate(self.alphabet)}
        self.eos, self.unk, self.bos = self.index[EOS], self.index[UNK], len(self.alphabet)
        self._rows: dict = {}

    def root(self, prompt) -> Prefix:
        return (self.bos, self.bos) + tuple(self.index[w] for w in prompt)

    def state(self, h: Prefix):
        return h[-2:]

    def _row(self, u):
        hit = self._rows.get(u)
        if hit is None:
            p = self.vec(tuple(BOS if i == self.bos else self.alphabet[i] for i in u)).copy()
            p[self.unk] = 0.0
            with np.errstate(divide="ignore"):
                lp = np.log(p / p.sum())
            order = np.argsort(-lp, kind="stable")
            hit = self._rows[u] = (lp, order[np.isfinite(lp[order])])
        return hit

    def next_logp(self, prefixes) -> np.ndarray:
        return np.stack([self._row(h[-2:])[0] for h in prefixes])

    def ranked(self, h: Prefix):
        return self._row(h[-2:])

    def words(self, toks) -> list[str]:
        return [self.alphabet[i] for i in toks]


@lru_cache(maxsize=1)
def trigram() -> TrigramLM:
    return TrigramLM()


DEC_V, DEC_CTX = sl.TOY_V, sl.TOY_CTX  # Topic 13's decoder shapes: 512 tokens, 32 positions


@lru_cache(maxsize=1)
def decoder_vocab() -> tuple[str, ...]:
    """The DEC_V − 3 commonest words of Topic 3's training sentences, then <s>, </s>, <unk> (ids DEC_V − 3 … − 1)."""
    c = Counter(w for s in ng.corpus()["train"] for w in s)
    return tuple(w for w, _ in c.most_common(DEC_V - 3)) + (BOS, EOS, UNK)


@lru_cache(maxsize=1)
def sentence_stream() -> tuple[np.ndarray, np.ndarray]:
    """The training sentences as one stream <s> w₁ … w_n </s> <s> …, and the offset of every sentence's <s>."""
    ix = {w: i for i, w in enumerate(decoder_vocab())}
    stream, starts = [], []
    for s in ng.corpus()["train"]:
        starts.append(len(stream))
        stream += [ix[BOS]] + [ix.get(w, ix[UNK]) for w in s] + [ix[EOS]]
    return np.array(stream + [ix[BOS]] * (DEC_CTX + 1)), np.array(starts)


@dataclass(frozen=True)
class DecoderConfig:
    d: int = 32
    steps: int = 1200
    seed: int = 0


DECODER = DecoderConfig()


@lru_cache(maxsize=None)
def train_decoder(cfg: DecoderConfig) -> tuple[dict, list, float]:
    """Topic 13's causal pre-LN decoder (2 blocks, learned positions, tied embeddings) trained by Topic 13's recipe:
    Adam in float32 at peak rate 1.28/d, 10% warmup, linear decay. Each window of DEC_CTX + 1 tokens starts at a
    sentence's <s>, so position 0 always holds <s>, as it does when we decode. Returns float64 copies of the weights
    (decoding runs in float64) and the final batch loss."""
    rng_init = np.random.default_rng([SEED, 1700, cfg.seed, cfg.d])
    rng = np.random.default_rng([SEED, 1701, cfg.seed])
    stream, starts = sentence_stream()
    P, blocks = sl.init_decoder(rng_init, cfg.d)
    leaves = [(P, k) for k in P] + [(blk[s], k) for blk in blocks for s in blk for k in blk[s]]
    m1 = [np.zeros_like(D[k]) for D, k in leaves]
    m2 = [np.zeros_like(D[k]) for D, k in leaves]
    peak, warm = sl.LR_SCALE / cfg.d, max(1, int(sl.WARMUP * cfg.steps))
    loss = math.nan
    for step in range(1, cfg.steps + 1):
        X = stream[starts[rng.integers(0, len(starts), size=sl.TOY_BATCH)][:, None] + np.arange(DEC_CTX + 1)]
        loss, G, Gb = sl.decoder_loss_grad(P, blocks, X[:, :-1], X[:, 1:])
        grads = [G[k] for k in P] + [Gb[i][s][k] for i, blk in enumerate(blocks) for s in blk for k in blk[s]]
        c1, c2 = 1 - nlm.ADAM_B1**step, 1 - nlm.ADAM_B2**step
        lr = peak * min(step / warm, (cfg.steps - step + 1) / (cfg.steps - warm + 1))
        for j, ((D, k), g) in enumerate(zip(leaves, grads)):
            m1[j] = nlm.ADAM_B1 * m1[j] + (1 - nlm.ADAM_B1) * g
            m2[j] = nlm.ADAM_B2 * m2[j] + (1 - nlm.ADAM_B2) * g**2
            D[k] -= lr * (m1[j] / c1) / (np.sqrt(m2[j] / c2) + nlm.ADAM_EPS)
    P64 = {k: v.astype(np.float64) for k, v in P.items()}
    b64 = [{s: {k: v.astype(np.float64) for k, v in blk[s].items()} for s in blk} for blk in blocks]
    return P64, b64, float(loss)


class DecoderLM:
    """The trained decoder as a next-token oracle, conditioned at each step on emitting neither <unk> nor <s>
    (renormalized). No Markov state: two prefixes never share a future, so nothing recombines."""

    def __init__(self, cfg: DecoderConfig = DECODER):
        self.P, self.blocks, self.train_loss = train_decoder(cfg)
        self.alphabet = decoder_vocab()
        self.index = {w: i for i, w in enumerate(self.alphabet)}
        self.bos, self.eos, self.unk = (self.index[w] for w in (BOS, EOS, UNK))
        self._ranked: dict[Prefix, tuple[np.ndarray, np.ndarray]] = {}

    def root(self, prompt) -> Prefix:
        return (self.bos,) + tuple(self.index[w] for w in prompt)

    def max_len(self, root: Prefix) -> int:
        """Tokens a string after `root` may have: the position table ends at DEC_CTX inputs."""
        return DEC_CTX + 1 - len(root)

    def _mask(self, lp: np.ndarray) -> np.ndarray:
        lp = lp.copy()
        lp[..., [self.unk, self.bos]] = -np.inf
        return lp - np.logaddexp.reduce(lp, axis=-1, keepdims=True)

    def next_logp(self, prefixes) -> np.ndarray:
        groups: dict[int, list[int]] = {}
        for i, h in enumerate(prefixes):
            groups.setdefault(len(h), []).append(i)
        out = np.empty((len(prefixes), DEC_V))
        for idx in groups.values():
            X = np.array([prefixes[i] for i in idx])
            out[idx] = self._mask(sl.decoder_log_probs(self.P, self.blocks, X)[:, -1])
        return out

    def ranked(self, h: Prefix):
        hit = self._ranked.get(h)
        if hit is None:
            lp = self.next_logp([h])[0]
            order = np.argsort(-lp, kind="stable")
            hit = self._ranked[h] = (lp, order[np.isfinite(lp[order])])
        return hit

    def state(self, h: Prefix):
        return None

    def string_logp(self, root: Prefix, toks: Prefix) -> float:
        """log q(toks | root) from ONE forward pass over the whole string (the cross-check on every search score)."""
        seq = root + toks
        lp = self._mask(sl.decoder_log_probs(self.P, self.blocks, np.array([seq[:-1]]))[0])
        return float(sum(lp[t, seq[t + 1]] for t in range(len(root) - 1, len(seq) - 1)))

    def words(self, toks) -> list[str]:
        return [self.alphabet[i] for i in toks]


@lru_cache(maxsize=1)
def decoder() -> DecoderLM:
    return DecoderLM()


@lru_cache(maxsize=1)
def prompts() -> tuple[tuple[str, ...], ...]:
    """Every distinct three-word opening of a held-out MD&A sentence longer than six words (generate_corpus(100, 1),
    Topic 3's held-out split) whose words are all in the decoder's vocabulary, sorted."""
    voc = set(decoder_vocab()[:-3])
    docs = fc.generate_corpus(100, 1)
    return tuple(sorted({s[:3] for d in docs if d.doc_type == "mdna" for s in ng.split_sentences(d.text)
                         if len(s) > 6 and all(w in voc for w in s[:3])}))


@lru_cache(maxsize=1)
def reference_lengths() -> np.ndarray:
    """For each prompt, the mean number of words that follow it in the held-out MD&A sentences it opens."""
    docs = fc.generate_corpus(100, 1)
    rest: dict[tuple[str, ...], list[int]] = {}
    for d in docs:
        if d.doc_type == "mdna":
            for s in ng.split_sentences(d.text):
                rest.setdefault(s[:3], []).append(len(s) - 3)
    return np.array([np.mean(rest[p]) for p in prompts()])


TRIGRAM_MAX_LEN = 40  # generated tokens, </s> included
KS = (1, 2, 4, 8, 16, 32, 64)
ALPHAS = (0.0, 0.5, 1.0)


def engine(name: str):
    return trigram() if name == "trigram" else decoder()


def _max_len(name: str, root: Prefix) -> int:
    return TRIGRAM_MAX_LEN if name == "trigram" else decoder().max_len(root)


@lru_cache(maxsize=None)
def exact_modes(name: str) -> tuple[Found, ...]:
    """Every prompt's exact mode: Dijkstra on the trigram's state graph; best-first tree search on the decoder."""
    lm = engine(name)
    out = []
    for p in prompts():
        root = lm.root(p)
        out.append(uniform_cost(lm, root, max_len=_max_len(name, root), recombine=name == "trigram"))
    return tuple(out)


@lru_cache(maxsize=None)
def _beam_table(name: str) -> dict[int, tuple[tuple[Beam, ...], ...]]:
    lm = engine(name)
    return {k: tuple(beam_search_multi(lm, lm.root(p), k, max_len=_max_len(name, lm.root(p)), alphas=ALPHAS)
                     for p in prompts()) for k in KS}


def beam_runs(name: str, alpha: float) -> dict[int, tuple[Beam, ...]]:
    """Beam(k) on every prompt, for every k in KS, at normalization α."""
    a = ALPHAS.index(alpha)
    return {k: tuple(r[a] for r in runs) for k, runs in _beam_table(name).items()}


@lru_cache(maxsize=1)
def training_sentences() -> frozenset[tuple[str, ...]]:
    return frozenset(ng.corpus()["train"])


def _sentence(lm, p, toks) -> tuple[str, ...]:
    return tuple(p) + tuple(lm.words(toks[:-1]))


@lru_cache(maxsize=None)
def curve(name: str, alpha: float) -> dict:
    """Beam(k) against the exact mode, for every k in KS, on every prompt:
    search_error (α = 0: the beam's string is less probable than the mode), exact (it is the mode), certified (Thm 3
    fired), finished, mean length in words (finished strings), copied (prompt + output is a training sentence verbatim),
    distinct outputs, and per-word NLL (nats) of the outputs."""
    lm, modes = engine(name), exact_modes(name)
    train = training_sentences()
    rows = []
    for k, runs in beam_runs(name, alpha).items():
        fin = [(p, b) for p, b in zip(prompts(), runs) if b.tokens is not None]
        err = [b.tokens is None or b.logp < m.logp - 1e-9 for b, m in zip(runs, modes)]
        rows.append({
            "k": k,  # the exact optimum is known only for α = 0 (the mode); at α > 0 only the certificate speaks
            "search_error": float(np.mean(err)) if alpha == 0 else None,
            "exact": float(np.mean([b.tokens == m.tokens for b, m in zip(runs, modes)])) if alpha == 0 else None,
            "certified": float(np.mean([b.certified for b in runs])),
            "finished": len(fin) / len(runs),
            "length": float(np.mean([len(b.tokens) - 1 for _, b in fin])),
            "copied": float(np.mean([_sentence(lm, p, b.tokens) in train for p, b in fin])),
            "distinct": len({b.tokens for _, b in fin}),
            "nll": float(np.mean([-b.logp / len(b.tokens) for _, b in fin])),
        })
    return {"rows": rows}


def mode_stats(name: str) -> dict:
    lm, modes = engine(name), exact_modes(name)
    L = np.array([len(m.tokens) - 1 for m in modes])
    train = training_sentences()
    return {"length": float(L.mean()), "empty": float((L == 0).mean()),
            "copied": float(np.mean([_sentence(lm, p, m.tokens) in train for p, m in zip(prompts(), modes)])),
            "pops_max": max(m.pops for m in modes), "pops_median": float(np.median([m.pops for m in modes])),
            "nll": float(np.mean([-m.logp / len(m.tokens) for m in modes]))}


@lru_cache(maxsize=1)
def eos_argmax_prompts() -> tuple[bool, ...]:
    """For each prompt, whether </s> is the trigram's likeliest next token (Thm 4(a): then the mode is empty)."""
    lm = trigram()
    return tuple(int(lm.ranked(lm.root(p))[1][0]) == lm.eos for p in prompts())


def nonmonotone_share(name: str) -> float:
    """The share of prompts on which some larger beam returns a strictly less probable string than a smaller one."""
    runs = beam_runs(name, 0.0)
    bad = 0
    for i in range(len(prompts())):
        best = -math.inf
        for k in KS:
            s = runs[k][i].logp
            bad_here = s < best - 1e-9
            best = max(best, s)
            if bad_here:
                bad += 1
                break
    return bad / len(prompts())


# --------------------------------------------------------------------------------------------
# 6. Non-termination: greedy orbits on a finite-state model.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Orbit:
    tokens: Prefix  # greedy's tokens up to </s>, or through one full period of the cycle
    preperiod: int  # steps before the cycle starts (or the length to </s>)
    period: int  # 0 when greedy reaches </s>


def greedy_orbit(lm, root: Prefix) -> Orbit:
    """Thm 5: on a Markov model greedy's next token is a function of the state, so the state sequence is a
    deterministic walk on a finite set and becomes periodic as soon as a state repeats."""
    h, first = root, {}
    while True:
        s = lm.state(h)
        if s in first:
            t0 = first[s]
            return Orbit(h[len(root):], t0, len(h) - len(root) - t0)
        first[s] = len(h) - len(root)
        j = int(lm.ranked(h)[1][0])
        h = h + (j,)
        if j == lm.eos:
            return Orbit(h[len(root):], len(h) - len(root), 0)


@lru_cache(maxsize=1)
def trigram_orbits() -> tuple[Orbit, ...]:
    lm = trigram()
    return tuple(greedy_orbit(lm, lm.root(p)) for p in prompts())


def welleck_model(k: int, eps: float = 0.05) -> StationaryLM:
    """k words, each (1 − ε)/k, and </s> with ε < (1 − ε)/k after every prefix: |Y| is geometric, so the model is
    consistent, yet </s> never ranks among any step's top k candidates."""
    assert eps < (1 - eps) / k
    return StationaryLM({EOS: eps, **{f"w{i}": (1 - eps) / k for i in range(k)}})


# --------------------------------------------------------------------------------------------
# 7. Finance case study: the mode is boilerplate.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def attractor() -> dict:
    """The trigram's length-normalized beam(16) outputs (α = 1): their commonest final five words, and how many of
    the prompts' outputs end in them."""
    lm = trigram()
    runs = beam_runs("trigram", 1.0)[16]
    tails = Counter(tuple(lm.words(b.tokens[:-1]))[-5:] for b in runs if b.tokens is not None and len(b.tokens) > 5)
    tail, n = tails.most_common(1)[0]
    return {"tail": " ".join(tail), "share": n / len(runs)}


@lru_cache(maxsize=1)
def reference_nll() -> float:
    """Per-word NLL (nats, </s> included) of the held-out MD&A sentences the prompts open, continuation only, under
    the trigram: the text the model is asked to continue, for comparison with what decoding returns."""
    lm = trigram()
    docs = fc.generate_corpus(100, 1)
    want, tot, n = set(prompts()), 0.0, 0
    for d in docs:
        if d.doc_type != "mdna":
            continue
        for s in ng.split_sentences(d.text):
            if s[:3] in want and all(w in lm.index for w in s):
                h = lm.root(s[:3])
                for w in s[3:] + (EOS,):
                    j = lm.index[w]
                    tot -= lm.next_logp([h])[0][j]
                    h, n = h + (j,), n + 1
    return tot / n


# --------------------------------------------------------------------------------------------
# 8. The lab's payload.
# --------------------------------------------------------------------------------------------


def _trace_tree(lm: TableLM) -> list[dict]:
    """Every node of a finite tree: id (space-joined words), parent, word, conditional and cumulative probability."""
    nodes, stack = [{"id": "", "parent": None, "w": lm.label, "p": 1.0, "P": 1.0}], [((), 1.0)]
    while stack:
        h, P = stack.pop()
        row = lm.rows[h]
        for j in np.flatnonzero(row):
            g = h + (j,)
            nodes.append({"id": " ".join(lm.words(g)), "parent": " ".join(lm.words(h)), "w": lm.alphabet[j],
                          "p": _r(row[j]), "P": _r(P * row[j], 4)})
            if j != lm.eos:
                stack.append((g, P * row[j]))
    return sorted(nodes, key=lambda n: (n["id"].count(" ") + (n["id"] != ""), n["id"]))


def beam_trace(lm: TableLM, k: int) -> dict:
    """Beam(k) step by step on the lab tree: at each step the kept, finished, and pruned candidates (node ids)."""
    open_, steps, best, best_pruned = [(0.0, ())], [], None, -math.inf
    while open_:
        cand = sorted(((s + lm.logp(h)[j], h + (j,)) for s, h in open_ for j in np.flatnonzero(lm.rows[h])),
                      key=lambda t: (-t[0], t[1]))
        keep, drop = cand[:k], cand[k:]
        if drop:
            best_pruned = max(best_pruned, drop[0][0])
        fin = [c for c in keep if c[1][-1] == lm.eos]
        open_ = [c for c in keep if c[1][-1] != lm.eos]
        for c in fin:
            if best is None or c[0] > best[0]:
                best = c
        steps.append({"kept": [" ".join(lm.words(h)) for _, h in open_],
                      "finished": [" ".join(lm.words(h)) for _, h in fin],
                      "pruned": [" ".join(lm.words(h)) for _, h in drop]})
        if best is not None and (not open_ or best[0] >= open_[0][0]):
            break
    b = beam_search(lm, (), k, max_len=10)
    assert b.tokens == best[1] and abs(b.logp - best[0]) < 1e-12
    return {"k": k, "steps": steps, "result": " ".join(lm.words(best[1])), "P": _r(math.exp(best[0]), 4),
            "certified": b.certified}


def ucs_trace(lm: TableLM) -> list[dict]:
    """Uniform-cost search's pops on the lab tree, in order (node id and cumulative probability)."""
    heap, out, tick = [(0.0, 0, ())], [], 0
    while heap:
        c, _, h = heapq.heappop(heap)
        out.append({"id": " ".join(lm.words(h)), "P": _r(math.exp(-c), 4)})
        if h and h[-1] == lm.eos:
            return out
        for j in np.flatnonzero(lm.rows[h]):
            tick += 1
            heapq.heappush(heap, (c - math.log(lm.rows[h][j]), tick, h + (j,)))
    raise AssertionError("unreachable")


def _example(lm, p, toks) -> str:
    return " ".join(lm.words(toks))


def loop_examples(n: int = 4) -> list[dict]:
    """Greedy cycles on the trigram for the lab's strip: the first n prompts (sorted) whose greedy run cycles, with the
    words before the cycle and the cycle itself. Every cycle on these prompts is the same one."""
    lm, out = trigram(), []
    for p, o in zip(prompts(), trigram_orbits()):
        if o.period:
            out.append({"prompt": " ".join(p), "pre": lm.words(o.tokens[: o.preperiod]),
                        "cycle": lm.words(o.tokens[o.preperiod:]), "period": o.period})
        if len(out) == n:
            break
    return out


@lru_cache(maxsize=1)
def example_prompts(n: int = 6) -> tuple[int, ...]:
    """The lab's table: the first n prompts (sorted) on which the trigram's greedy, beam(16), and normalized beam(16)
    outputs are not all the exact mode, so each row shows at least one decoder departing from the mode."""
    modes, runs = exact_modes("trigram"), [beam_runs("trigram", a)[k] for a, k in ((0.0, 1), (0.0, 16), (1.0, 16))]
    picks = [i for i in range(len(prompts())) if any(r[i].tokens != modes[i].tokens for r in runs)]
    return tuple(picks[:n])


def examples(name: str) -> list[dict]:
    """example_prompts()'s greedy, beam(16), normalized beam(16) (α = 1), and exact outputs on one engine."""
    lm, modes = engine(name), exact_modes(name)
    g, b, bn = beam_runs(name, 0.0)[1], beam_runs(name, 0.0)[16], beam_runs(name, 1.0)[16]
    out = []
    for i in example_prompts():
        row = {"prompt": " ".join(prompts()[i]), "mode": _example(lm, prompts()[i], modes[i].tokens)}
        for key, r in (("greedy", g[i]), ("beam", b[i]), ("norm", bn[i])):
            row[key] = None if r.tokens is None else _example(lm, prompts()[i], r.tokens)
        out.append(row)
    return out


def _curve_payload(name: str) -> dict:
    out = {}
    for a in ALPHAS:
        rows = curve(name, a)["rows"]
        out[f"{a:g}"] = {key: [_r(r[key]) if isinstance(r[key], float) else r[key] for r in rows]
                         for key in ("search_error", "exact", "certified", "finished", "length", "copied", "distinct",
                                     "nll")}
    ms = mode_stats(name)
    out["mode"] = {"length": _r(ms["length"]), "empty": _r(ms["empty"]), "copied": _r(ms["copied"]),
                   "nll": _r(ms["nll"])}
    out["reference_length"] = _r(float(reference_lengths().mean()))
    return out


def viz_payload() -> dict:
    lab = lab_tree()
    tree = {"nodes": _trace_tree(lab), "beams": [beam_trace(lab, k) for k in (1, 2, 3)], "ucs": ucs_trace(lab),
            "mode": " ".join(lab.words(brute_force_mode(lab)[1])), "P_mode": _r(math.exp(brute_force_mode(lab)[0]), 4)}
    orbits = trigram_orbits()
    loops = {"examples": loop_examples(), "share": _r(np.mean([o.period > 0 for o in orbits])),
             "periods": sorted({o.period for o in orbits if o.period})}
    return {"tree": tree, "ks": list(KS), "alphas": [f"{a:g}" for a in ALPHAS], "n_prompts": len(prompts()),
            "trigram": _curve_payload("trigram"), "decoder": _curve_payload("decoder"), "loops": loops,
            "examples": {"trigram": examples("trigram"), "decoder": examples("decoder")},
            "attractor": {"tail": attractor()["tail"], "share": _r(attractor()["share"])}}


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "beam-search-decoding-lab.json"


def summary() -> dict:
    out = {"n_prompts": len(prompts()), "reference_length": float(reference_lengths().mean()),
           "reference_nll": reference_nll(), "decoder_train_loss": decoder().train_loss,
           "eos_argmax_share": float(np.mean(eos_argmax_prompts())),
           "greedy_loop_share": float(np.mean([o.period > 0 for o in trigram_orbits()])),
           "attractor_share": attractor()["share"]}
    for name in ("trigram", "decoder"):
        for key, v in mode_stats(name).items():
            out[f"{name}_mode_{key}"] = v
        out[f"{name}_nonmonotone"] = nonmonotone_share(name)
        for a in ALPHAS:
            for r in curve(name, a)["rows"]:
                if r["k"] in (1, 4, 16, 64):
                    for key in ("search_error", "certified", "length", "copied", "distinct"):
                        if r[key] is not None:
                            out[f"{name}_a{a:g}_k{r['k']}_{key}"] = r[key]
    return out


# --------------------------------------------------------------------------------------------
# Verification harness: every pedagogical claim, as an assert.
# --------------------------------------------------------------------------------------------


def _random_trees(n: int, m: int = 3, depth: int = 5, stream: int = 1710) -> list[TableLM]:
    rng = np.random.default_rng([SEED, stream])
    return [random_tree(rng, m, depth) for _ in range(n)]


def test_uniform_cost_is_exact():
    """Thm 1: on 200 random trees uniform-cost search returns the brute-force mode, within (2 + E|Y|)/p(y*) pops."""
    for lm in _random_trees(200):
        f, (s, y) = uniform_cost(lm, ()), brute_force_mode(lm)
        assert f.tokens == y and abs(f.logp - s) < 1e-12
        assert f.pops <= pop_bound(lm), (f.pops, pop_bound(lm))


def test_trigram_graph_search_equals_tree_search():
    """Corollary to Thm 1: recombining the trigram's prefixes on their state (Dijkstra) changes nothing but the work."""
    lm = trigram()
    for p, g in zip(prompts(), exact_modes("trigram")):
        t = uniform_cost(lm, lm.root(p), max_len=TRIGRAM_MAX_LEN)
        assert t.tokens == g.tokens and abs(t.logp - g.logp) < 1e-12 and t.pops >= g.pops


def test_greedy_has_no_approximation_guarantee():
    """Thm 2(a): greedy's string is less than ε times as probable as the mode, for ε down to 1e-3; no certificate."""
    for eps in (0.1, 0.01, 0.001):
        lm = greedy_trap(eps)
        g, (s, _) = greedy(lm, (), max_len=5), brute_force_mode(lm)
        assert math.exp(g.logp - s) < eps and not g.certified


def test_beam_k_misses_what_beam_k_plus_one_finds():
    """Thm 2(b): for k = 1, …, 8, beam(k) misses the trap's mode and beam(k + 1) returns it, certified."""
    for k in range(1, 9):
        lm = beam_trap(k)
        s, y = brute_force_mode(lm)
        b, b1 = beam_search(lm, (), k, max_len=5), beam_search(lm, (), k + 1, max_len=5)
        assert b.logp < s - 1e-9 and not b.certified
        assert b1.tokens == y and b1.certified


def test_beam_is_not_monotone_in_k():
    """Thm 2(c): beam(2) returns 0.004 where beam(1) and beam(3) return 0.06; it also happens on the corpus."""
    lm = nonmonotone_beam()
    P = [math.exp(beam_search(lm, (), k, max_len=5).logp) for k in (1, 2, 3)]
    assert np.allclose(P, [0.06, 0.004, 0.06])
    assert nonmonotone_share("trigram") > 0 and nonmonotone_share("decoder") > 0


def test_certificate_is_sound():
    """Thm 3: whenever the certificate fires, the beam's string is the mode — on random trees, both corpus engines,
    and every k; at α > 0 the capped form never fires on a string that a certified larger beam beats."""
    for lm in _random_trees(100, stream=1711):
        s, y = brute_force_mode(lm)
        for k in (1, 2, 3, 5):
            b = beam_search(lm, (), k, max_len=6)
            assert not b.certified or (b.tokens == y and abs(b.logp - s) < 1e-12)
    for name in ("trigram", "decoder"):
        modes = exact_modes(name)
        for k, runs in beam_runs(name, 0.0).items():
            for b, m in zip(runs, modes):
                assert not b.certified or (b.tokens == m.tokens and abs(b.logp - m.logp) < 1e-9), (name, k)
        for a in ALPHAS[1:]:
            runs = beam_runs(name, a)
            for i in range(len(prompts())):
                cert = [runs[k][i] for k in KS if runs[k][i].certified]
                for b in (runs[k][i] for k in KS):
                    assert all(b.score <= c.score + 1e-9 for c in cert), (name, a)
    assert curve("trigram", 0.0)["rows"][-1]["certified"] == 1.0


def test_lab_tree():
    """The lab's walk-through: greedy takes "rose sharply" (0.0864); beam(2) returns the mode "was flat" (0.238) and
    certifies it, since nothing it pruned beat 0.238; greedy's pruned "was" (0.35) blocks its certificate."""
    lm = lab_tree()
    g, b2 = greedy(lm, (), max_len=6), beam_search(lm, (), 2, max_len=6)
    assert lm.words(g.tokens) == ["rose", "sharply", EOS] and abs(math.exp(g.logp) - 0.0864) < 1e-12
    assert lm.words(b2.tokens) == ["was", "flat", EOS] and abs(math.exp(b2.logp) - 0.238) < 1e-12
    assert b2.certified and not g.certified
    assert ucs_trace(lm)[-1]["id"] == "was flat </s>"


def test_stopping_dominates():
    """Thm 4(a): wherever </s> is the likeliest next token, stopping is the best completion — on every prefix of 100
    random trees; on the trigram, every prompt whose next-token argmax is </s> has the empty mode."""
    for lm in _random_trees(100, stream=1712):
        n, ok = stop_dominates(lm)
        assert n == ok
    L = [len(m.tokens) for m in exact_modes("trigram")]
    assert all(l == 1 for e, l in zip(eos_argmax_prompts(), L) if e)
    assert np.mean(eos_argmax_prompts()) < np.mean([l == 1 for l in L])  # sufficient, far from necessary


def test_step_ceiling_bounds_the_mode_length():
    """Thm 4(b): p(y) ≤ q^|y| for every string, so |y*| ≤ log p(y′)/log q for any finished y′ (here greedy's)."""
    for lm in _random_trees(100, stream=1713):
        q = step_ceiling(lm)
        for s, y in enumerate_strings(lm):
            assert s <= (len(y) - 1) * math.log(q) + 1e-12
        s, y = brute_force_mode(lm)
        assert len(y) - 1 <= greedy(lm, (), max_len=10).logp / math.log(q) + 1e-12


def test_normalization_has_no_optimum():
    """Length normalization without a cap: under p(w) = 0.9, the normalized score of w^n </s> rises with n toward
    6·log 0.9, so no string attains the supremum (α = 1, Wu et al.'s lp)."""
    v = np.array([normalized_chain(0.9, n) for n in range(2001)])
    assert np.all(np.diff(v) > 0) and abs(v[-1] - 6 * math.log(0.9)) < 0.01 and v[-1] < 6 * math.log(0.9)


def test_greedy_is_the_regularized_optimum():
    """Meister et al.'s Thm 3.1 on finite trees: for λ above a finite threshold the maximizer of log p − λ·R_greedy is
    the greedy string; at λ = 0 it is the mode."""
    for lm in _random_trees(50, stream=1714):
        lam0, g = greedy_threshold(lm), greedy(lm, (), max_len=10)
        assert regularized_mode(lm, lam0 + 1.0)[1] == g.tokens
        assert regularized_mode(lm, 0.0)[1] == brute_force_mode(lm)[1]


def test_search_error_against_model_error():
    """§5: the trigram's search error falls from greedy to 0 by k = 64 while its strings shrink to the mode's length;
    the decoder, with nothing to recombine, keeps search errors at k = 64. Both modes are shorter than the text."""
    tri, dec = curve("trigram", 0.0)["rows"], curve("decoder", 0.0)["rows"]
    assert tri[0]["search_error"] > 0.5 and tri[-1]["search_error"] == 0.0
    assert abs(tri[-1]["length"] - mode_stats("trigram")["length"]) < 1e-12 and tri[0]["length"] > tri[-1]["length"]
    assert dec[-1]["search_error"] > 0.1 and dec[0]["length"] > dec[-1]["length"] > mode_stats("decoder")["length"]
    ref = float(reference_lengths().mean())
    assert mode_stats("trigram")["length"] < ref / 2 and mode_stats("decoder")["length"] < ref


def test_decoder_scores_match_forward():
    """Every decoder score the searches report equals one full forward pass over the decoded string (float64)."""
    lm = decoder()
    for p, m in zip(prompts(), exact_modes("decoder")):
        assert abs(lm.string_logp(lm.root(p), m.tokens) - m.logp) < 1e-9
    for k in (1, 8, 64):
        for p, b in zip(prompts(), beam_runs("decoder", 0.0)[k]):
            assert abs(lm.string_logp(lm.root(p), b.tokens) - b.logp) < 1e-9


def test_greedy_is_eventually_periodic():
    """Thm 5 on the trigram: greedy either ends or enters a cycle, and a cycling prompt never finishes (beam(1) runs
    to the cap); the orbit's tokens are greedy's tokens."""
    lm = trigram()
    g = beam_runs("trigram", 0.0)[1]
    for p, o, b in zip(prompts(), trigram_orbits(), g):
        if o.period:
            assert b.tokens is None and o.preperiod + o.period <= len(o.tokens)
        else:
            assert b.tokens == o.tokens
    assert 0.1 < np.mean([o.period > 0 for o in trigram_orbits()]) < 0.5


def test_consistent_model_beam_never_ends():
    """Welleck et al.'s Thm 3.4 in a stationary model: P(|Y| > t) = (1 − ε)^t → 0, yet beam(k) never finishes while
    beam(k + 1) returns the mode (the empty string). Thm 3's certificate is sufficient, not necessary: for 2 ≤ k ≤ 4
    beam(k + 1) prunes words more probable than ε before stopping, and the certificate stays silent though the answer is
    right."""
    for k in range(1, 6):
        lm = welleck_model(k)
        assert beam_search(lm, (), k, max_len=300).tokens is None and (1 - 0.05) ** 300 < 1e-6
        b = beam_search(lm, (), k + 1, max_len=300)
        assert b.tokens == (lm.eos,) and abs(b.logp - math.log(0.05)) < 1e-12
        assert b.certified == (k in (1, 5))


def test_the_mode_is_boilerplate():
    """§7: the trigram's mode is the empty continuation on 0.2 of the prompts and less predictable text is never
    chosen (its per-word NLL is half the held-out text's); normalization lengthens the output, and pulls 0.13 of the
    prompts onto one 8-K template."""
    ms = mode_stats("trigram")
    assert ms["empty"] > 0.15 and ms["nll"] < reference_nll() / 1.5
    rows = curve("trigram", 1.0)["rows"]
    assert rows[-1]["length"] > 2 * ms["length"]
    assert attractor()["tail"] == "to disclose a material event" and attractor()["share"] > 0.1


def test_viz_payload():
    p = viz_payload()
    assert p["tree"]["mode"] == "was flat </s>" and [b["certified"] for b in p["tree"]["beams"]] == [False, True, True]
    assert len(p["trigram"]["0"]["search_error"]) == len(KS) and p["trigram"]["1"]["search_error"] == [None] * len(KS)
    assert len(p["loops"]["examples"]) == 4 and {tuple(e["cycle"]) for e in p["loops"]["examples"]} == {("and",)}
    assert p["n_prompts"] == len(prompts()) and len(p["examples"]["trigram"]) == 6


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
    print(f"\nSummary ({len(prompts())} held-out MD&A prompts; Topic 3's IKN trigram, V = {len(trigram().alphabet)}; "
          f"Topic 13's decoder, d = {DECODER.d}, {DECODER.steps} steps, V = {DEC_V}):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.6f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--dev" in sys.argv:
    import time

    t0 = time.time()
    for k, v in summary().items():
        print(f"  {k:36} {v:.6f}" if isinstance(v, float) else f"  {k:36} {v}")
    print("attractor:", attractor()["tail"])
    for name in ("trigram", "decoder"):
        for e in examples(name):
            print(name, e)
    print(loop_examples())
    print("time", round(time.time() - t0, 1))
elif __name__ == "__main__":
    raise SystemExit(_run_harness())
