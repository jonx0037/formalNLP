"""Overlap metrics: BLEU and ROUGE — the reference implementation for formalNLP's twenty-seventh topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. The match grid and clipping. A candidate y and a reference r define a grid whose marked cells are the pairs of equal
   tokens (or n-grams). BLEU's clipped count (Def 1) caps each n-gram's credit at its largest count in any reference;
   it is the size of a maximum matching between candidate occurrences and reference slots (Prop 1), so repeating one
   matched word earns credit only as often as the references hold it.
2. BLEU (Def 2): the geometric mean of the clipped precisions times the brevity penalty, with counts pooled over the
   corpus. With one reference per segment, BLEU ≤ g(c/r) where g(x) = e^{1 − 1/x} below x = 1 and 1/x above it (one
   segment), or x^{−1/N} above it (a corpus): the envelope peaks only at the reference length (Thm 1). Corpus precisions
   are mediants, so one system can win every sentence and lose the corpus (Prop 2). A sentence with no matched 4-gram
   scores 0 whatever else it gets right; add-k smoothing interpolates each precision toward 1, Topic 3's shrinkage with
   a prior at perfect precision (Prop 3). Clipped counts can only rise when a reference is added, but BLEU can fall,
   because the brevity penalty measures against the closest reference (Prop 4).
3. ROUGE (Def 3). ROUGE-N is n-gram recall; ROUGE-L scores the longest common subsequence, which is the longest chain of
   marked cells increasing in both coordinates, a longest path in the grid's DAG, found by the dynamic program; a chain
   is a matching, so the LCS never exceeds the clipped unigram matches and ROUGE-L ≤ ROUGE-1 (Thm 2).
4. What overlap cannot see. k substitutions move a clipped count by at most nk and the LCS by at most k, so ROUGE-N
   recall moves by at most nk/|R_n| (Thm 3); a summary whose m facts each hang on one word loses at most that much when
   every fact is reversed (Cor 1). Cutting a candidate wherever its bigram appears in no reference and permuting the
   pieces never lowers a clipped count, and leaves BLEU unchanged unless a new junction happens to match (Prop 5,
   Callison-Burch et al.'s §3.1 argument made precise).
5. chrF (Def 4) and greedy soft matching (Def 5, BERTScore's form on Topic 7's SGNS vectors). One substitution u → u'
   moves soft precision by at most ‖u − u'‖/|y| and soft recall by at most ‖u − u'‖ (Prop 6); in these vectors
   cos(rose, fell) = 0.995, so ‖rose − fell‖ = 0.10.
6. The metric as a decision rule (Thm 4). Minimum Bayes risk picks argmax_y E_{Y∼p} U(y, Y); exact-match utility gives
   the mode (Topic 14's search target), and BLEU does not. On Topic 14's 188 MD&A prompts the trigram's exact mode is
   short (Topic 14's Thm 4), the brevity penalty punishes it, and BLEU-MBR over 64 samples scores highest. With |H|
   hypotheses and utilities in [0, 1], S samples pick the exact MBR choice with probability ≥ 1 − 2|H|e^{−SΔ²/2}, Δ the
   gap between the best two expected utilities.
7. Metrics and judgments (Prop 7). Averaging over segments removes per-segment noise but not a system's bias: system
   scores converge to their noise-free means, whose ranking reverses every pair whose bias gap exceeds its quality gap.
8. Finance case study, the answer key. The shared corpus's aspect gold gives each test document its facts (per aspect
   category, the sign of the summed polarities); four writers phrase them with the generator's own words, so every fact
   hangs on one direction verb or one polarity adjective. A writer's summary with every fact reversed outscores the
   document's own fact-bearing sentences on every metric, while its fact F1 is 0 against their 1.

Cross-checks: sacrebleu's corpus and sentence BLEU (no smoothing, add-k), sentence chrF; rouge-score's ROUGE-1, -2, -L;
scipy's maximum bipartite matching for Prop 1; brute-force subsequence enumeration for Thm 2.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 --with sacrebleu==2.6.0 --with rouge-score==0.1.2 python notebooks/overlap-metrics-bleu-rouge/overlap_metrics_bleu_rouge.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import maximum_bipartite_matching

_HERE = Path(__file__).resolve().parent
for _dep in ("beam-search-decoding", "collocations-keyword-extraction", "word2vec-implicit-factorization"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import beam_search_decoding as bs  # noqa: E402 — module objects: their test_* stay out of this harness
import collocations_keyword_extraction as ck  # noqa: E402
import word2vec_implicit_factorization as w2v  # noqa: E402

fc, ng, ds = bs.fc, bs.ng, ck.ds

# ONE stream per random-instance claim, each from default_rng([SEED, k]); k ≥ 4000 keeps this module's streams apart
# from Topics 1–26.
SEED = bs.SEED
VIZ_JSON = _HERE.parents[1] / "src" / "components" / "viz" / "data" / "overlap-metrics-lab.json"
N_MAX = 4  # BLEU's maximum n-gram order, with uniform weights 1/N


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


def _r(x, nd: int = 3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


Tokens = tuple[str, ...]


# --------------------------------------------------------------------------------------------
# 1. The match grid and clipping (Def 1, Prop 1).
# --------------------------------------------------------------------------------------------


def ngrams(toks, n: int) -> Counter:
    """c_x(g): the count of every n-gram g of x (no padding)."""
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def ref_max_counts(refs, n: int) -> Counter:
    """max_k c_{r_k}(g): each n-gram's largest count in any reference (the clipping ceiling)."""
    out: Counter = Counter()
    for r in refs:
        for g, k in ngrams(r, n).items():
            if k > out[g]:
                out[g] = k
    return out


def clipped(cand, refs, n: int) -> int:
    """Def 1's m_n(y, R) = Σ_g min(c_y(g), max_k c_{r_k}(g))."""
    ceil = ref_max_counts(refs, n)
    return sum(min(k, ceil[g]) for g, k in ngrams(cand, n).items())


def matching_size(cand, refs, n: int) -> int:
    """Prop 1's cross-check: a maximum matching between the candidate's n-gram occurrences and reference slots (each
    n-gram g owns max_k c_{r_k}(g) slots), by scipy's Hopcroft–Karp."""
    occ = [tuple(cand[i:i + n]) for i in range(len(cand) - n + 1)]
    slots = [g for g, k in ref_max_counts(refs, n).items() for _ in range(k)]
    if not occ or not slots:
        return 0
    edges = [(i, j) for i, g in enumerate(occ) for j, h in enumerate(slots) if g == h]
    if not edges:
        return 0
    rows, cols = zip(*edges)
    a = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(occ), len(slots)))
    return int((maximum_bipartite_matching(a, perm_type="column") >= 0).sum())


def match_grid(cand, ref) -> list[tuple[int, int]]:
    """The marked cells (i, j) with y_i = r_j."""
    return [(i, j) for i, a in enumerate(cand) for j, b in enumerate(ref) if a == b]


def clipped_pairs(cand, ref, n: int) -> list[tuple[int, int]]:
    """One maximum matching for Prop 1's picture: the k-th occurrence of each n-gram in y is paired with its k-th
    occurrence in r (start positions), while both last."""
    seen_r: dict[tuple, list[int]] = {}
    for j in range(len(ref) - n + 1):
        seen_r.setdefault(tuple(ref[j:j + n]), []).append(j)
    used: Counter = Counter()
    out = []
    for i in range(len(cand) - n + 1):
        g = tuple(cand[i:i + n])
        js = seen_r.get(g, [])
        if used[g] < len(js):
            out.append((i, js[used[g]]))
            used[g] += 1
    return out


# --------------------------------------------------------------------------------------------
# 2. BLEU (Def 2, Thm 1, Prop 2, Prop 3, Prop 4).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stats:
    """One segment's (or a corpus's pooled) BLEU statistics: candidate length c, effective reference length r, clipped
    matches m_n and candidate n-gram totals t_n for n = 1..N, and ρ = Σ_g max_k c_{r_k}(g) (Thm 1's unigram ceiling)."""
    c: int
    r: int
    m: tuple[int, ...]
    t: tuple[int, ...]
    rho: int


def closest_length(c: int, refs) -> int:
    """The reference length closest to c, the shorter on a tie (Papineni et al.'s best match length, sacrebleu's rule)."""
    return min((abs(len(x) - c), len(x)) for x in refs)[1]


def bleu_stats(cand, refs, N: int = N_MAX) -> Stats:
    c = len(cand)
    m = tuple(clipped(cand, refs, n) for n in range(1, N + 1))
    t = tuple(max(c - n + 1, 0) for n in range(1, N + 1))
    return Stats(c, closest_length(c, refs), m, t, sum(ref_max_counts(refs, 1).values()))


def pool(stats) -> Stats:
    stats = list(stats)
    N = len(stats[0].m)
    return Stats(sum(s.c for s in stats), sum(s.r for s in stats),
                 tuple(sum(s.m[n] for s in stats) for n in range(N)),
                 tuple(sum(s.t[n] for s in stats) for n in range(N)), sum(s.rho for s in stats))


def brevity_penalty(c: float, r: float) -> float:
    """BP = 1 if c > r, else e^{1 − r/c} (0 for an empty candidate)."""
    if c > r:
        return 1.0
    return 0.0 if c == 0 else math.exp(1 - r / c)


def precisions(s: Stats, smooth: str = "none", k: float = 1.0) -> list[float]:
    """The modified precisions p_n = m_n/t_n. smooth="add-k": (m_n + k)/(t_n + k) for n ≥ 2 (Lin and Och's BLEU+1 at
    k = 1; sacrebleu's add-k). smooth="exp": Chen and Cherry's method 3 (sacrebleu's default), the j-th zero-match order
    gets 1/(2^j t_n)."""
    out, j = [], 0
    for n, (m, t) in enumerate(zip(s.m, s.t)):
        if smooth == "add-k" and n > 0:
            out.append((m + k) / (t + k))
        elif smooth == "exp" and m == 0 and t > 0:
            j += 1
            out.append(1.0 / (2 ** j * t))
        else:
            out.append(m / t if t else 0.0)
    return out


def bleu(s: Stats, smooth: str = "none", k: float = 1.0) -> float:
    """Def 2: BLEU = BP · exp((1/N) Σ_n log p_n), and 0 once any p_n = 0."""
    p = precisions(s, smooth, k)
    if s.c == 0 or min(p) <= 0:
        return 0.0
    return brevity_penalty(s.c, s.r) * math.exp(sum(math.log(x) for x in p) / len(p))


def corpus_bleu(cands, refsets, N: int = N_MAX) -> float:
    return bleu(pool(bleu_stats(y, R, N) for y, R in zip(cands, refsets)))


def sentence_bleu(cand, refs, smooth: str = "none", N: int = N_MAX) -> float:
    return bleu(bleu_stats(cand, refs, N), smooth)


def envelope(x, N: int = N_MAX, corpus: bool = False):
    """Thm 1's g(x) at x = c/r: e^{1 − 1/x} for x ≤ 1, and 1/x (one segment) or x^{−1/N} (a corpus) above."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        left = np.exp(1 - 1 / x)
        right = x ** (-1.0 / N) if corpus else 1 / x
    return np.where(x <= 1, left, right)


def envelope_bound(s: Stats, N: int = N_MAX) -> float:
    """Thm 1 with several references: BLEU ≤ BP · min(1, ρ/c)^{1/N} for a pooled corpus, and BP · min(1, ρ/c) for one
    segment whose references are a single sentence. ρ = r when each segment has one reference."""
    return brevity_penalty(s.c, s.r) * min(1.0, s.rho / s.c) ** (1.0 / N) if s.c else 0.0


# Prop 2's instance: two segments, references of ten words each. A substitutes one middle word in each; B appends five
# unmatched words to the first and stops halfway through the second. A wins both sentences, B wins the corpus (its two
# length errors cancel in the pooled brevity penalty).
SIMPSON_REFS = (tuple("a b c d e f g h i j".split()), tuple("k l m n o p q s t u".split()))
SIMPSON_A = (tuple("a b c d x f g h i j".split()), tuple("k l m n x p q s t u".split()))
SIMPSON_B = (SIMPSON_REFS[0] + ("x",) * 5, SIMPSON_REFS[1][:5])


def simpson() -> dict:
    out = {}
    for name, sysm in (("A", SIMPSON_A), ("B", SIMPSON_B)):
        out[name] = {"sentence": [sentence_bleu(y, [r]) for y, r in zip(sysm, SIMPSON_REFS)],
                     "corpus": corpus_bleu(sysm, [[r] for r in SIMPSON_REFS])}
    return out


def mediant_bounds(stats_list, n: int) -> tuple[float, float, float]:
    """Prop 2(a): the pooled p_n = Σ m_n/Σ t_n lies between the smallest and largest segment ratio (segments with t_n > 0)."""
    rs = [s.m[n] / s.t[n] for s in stats_list if s.t[n] > 0]
    p = pool(stats_list)
    return min(rs), p.m[n] / p.t[n], max(rs)


# Prop 4's instance: the candidate has eight words; the first reference is its first six (r = 6 < c, BP = 1). A second
# reference of nine unseen words is closer in length (|9 − 8| < |6 − 8|), so r becomes 9 > c: precision is unchanged
# and BLEU falls by the factor e^{1 − 9/8}.
REF_CAND = tuple("a b c d e f g h".split())
REF_ONE = REF_CAND[:6]
REF_TWO = tuple("p q r s t u v w x".split())


def more_references() -> dict:
    s1, s2 = bleu_stats(REF_CAND, [REF_ONE]), bleu_stats(REF_CAND, [REF_ONE, REF_TWO])
    return {"m_one": s1.m, "m_two": s2.m, "r_one": s1.r, "r_two": s2.r, "bleu_one": bleu(s1), "bleu_two": bleu(s2)}


# --------------------------------------------------------------------------------------------
# 3. ROUGE (Def 3, Thm 2).
# --------------------------------------------------------------------------------------------


def prf(hit: float, n_cand: float, n_ref: float) -> tuple[float, float, float]:
    p = hit / n_cand if n_cand else 0.0
    r = hit / n_ref if n_ref else 0.0
    return p, r, (0.0 if p + r == 0 else 2 * p * r / (p + r))


def rouge_n(cand, ref, n: int) -> tuple[float, float, float]:
    """Def 3: ROUGE-N precision, recall, and F against one reference; the hit count is the clipped match."""
    c, r = ngrams(cand, n), ngrams(ref, n)
    return prf(sum((c & r).values()), sum(c.values()), sum(r.values()))


def lcs_table(a, b) -> np.ndarray:
    """Thm 2's dynamic program: L[i, j] = L[i−1, j−1] + 1 if a_i = b_j, else max(L[i−1, j], L[i, j−1])."""
    L = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            L[i, j] = L[i - 1, j - 1] + 1 if x == y else max(L[i - 1, j], L[i, j - 1])
    return L


def lcs(a, b) -> int:
    return int(lcs_table(a, b)[-1, -1])


def lcs_path(a, b) -> list[tuple[int, int]]:
    """One longest chain of marked cells, by backtracking the table (diagonal steps first)."""
    L, i, j, out = lcs_table(a, b), len(a), len(b), []
    while i and j:
        if a[i - 1] == b[j - 1] and L[i, j] == L[i - 1, j - 1] + 1:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif L[i - 1, j] >= L[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def lcs_brute(a, b) -> int:
    """Thm 2's cross-check: the longest subsequence of a (by enumeration) that is a subsequence of b."""

    def is_sub(s, t):
        it = iter(t)
        return all(x in it for x in s)

    for k in range(len(a), 0, -1):
        if any(is_sub(s, b) for s in itertools.combinations(a, k)):
            return k
    return 0


def rouge_l(cand, ref) -> tuple[float, float, float]:
    """Def 3: ROUGE-L from the LCS (rouge-score's β = 1 F)."""
    return prf(lcs(cand, ref), len(cand), len(ref))


def best_f(fn, cand, refs) -> float:
    """Several references: the best F over them (rouge-score's score_multi)."""
    return max(fn(cand, r)[2] for r in refs)


# --------------------------------------------------------------------------------------------
# 4. What overlap cannot see (Thm 3, Cor 1, Prop 5).
# --------------------------------------------------------------------------------------------


def substitute(toks, positions, rng: np.random.Generator, alphabet) -> Tokens:
    out = list(toks)
    for i in positions:
        out[i] = alphabet[int(rng.integers(len(alphabet)))]
    return tuple(out)


def chunks(cand, refs) -> list[Tokens]:
    """Prop 5: cut y between y_i and y_{i+1} wherever that bigram occurs in no reference."""
    ref_bigrams = set().union(*(ngrams(r, 2) for r in refs))
    out, cur = [], [cand[0]] if cand else []
    for a, b in zip(cand, cand[1:]):
        if (a, b) in ref_bigrams:
            cur.append(b)
        else:
            out.append(tuple(cur))
            cur = [b]
    if cur:
        out.append(tuple(cur))
    return out


def permutation_census(cand, refs, perms=None) -> dict:
    """Permutations of y's chunks (all of them, or the given index orders): how many keep every m_n (so BLEU) equal,
    that none lowers any m_n, and how many distinct best-reference ROUGE-L values they produce."""
    ch = chunks(cand, refs)
    perms = itertools.permutations(range(len(ch))) if perms is None else perms
    base = bleu_stats(cand, refs).m
    equal = lowered = total = 0
    rl = set()
    for perm in perms:
        y = tuple(w for i in perm for w in ch[i])
        m = bleu_stats(y, refs).m
        total += 1
        equal += m == base
        lowered += any(a < b for a, b in zip(m, base))
        rl.add(round(best_f(rouge_l, y, refs), 12))
    return {"chunks": len(ch), "perms": total, "equal": equal, "lowered": lowered, "rouge_l_values": len(rl)}


# --------------------------------------------------------------------------------------------
# 5. chrF and soft matching (Def 4, Def 5, Prop 6).
# --------------------------------------------------------------------------------------------


def chrf(cand: str, ref: str, N: int = 6, beta: float = 2.0) -> float:
    """Def 4 (sacrebleu's chrF: character n-grams n = 1..6 with whitespace removed, precision and recall averaged over
    the orders both strings reach, then F_β with β = 2)."""
    cs, rs = "".join(cand.split()), "".join(ref.split())
    ps, rs_ = [], []
    for n in range(1, N + 1):
        c = Counter(cs[i:i + n] for i in range(len(cs) - n + 1))
        r = Counter(rs[i:i + n] for i in range(len(rs) - n + 1))
        if c and r:
            hit = sum((c & r).values())
            ps.append(hit / sum(c.values()))
            rs_.append(hit / sum(r.values()))
    if not ps:
        return 0.0
    p, r = float(np.mean(ps)), float(np.mean(rs_))
    b2 = beta * beta
    return 0.0 if p + r == 0 else (1 + b2) * p * r / (b2 * p + r)


def chrf_best(cand, refs) -> float:
    return max(chrf(" ".join(cand), " ".join(r)) for r in refs)


SOFT_CFG = w2v.SGNSConfig("L2", 25)  # Topic 7's smallest grid model (~3 s); context ±2, d = 25


@lru_cache(maxsize=1)
def soft_vectors() -> tuple[dict[str, int], np.ndarray]:
    """Topic 7's trained word vectors W for SOFT_CFG, each row scaled to unit length, keyed by Topic 6's vocabulary."""
    X = np.asarray(w2v.sgns_vectors(SOFT_CFG, "W"), dtype=float)
    return {w: i for i, w in enumerate(ds.vocab())}, X / np.linalg.norm(X, axis=1, keepdims=True)


def similarity(a, b) -> np.ndarray:
    """S[i, j] = cos(a_i, b_j) for in-vocabulary pairs; any other pair scores 1 if the tokens are equal, else 0."""
    ix, U = soft_vectors()
    S = np.array([[1.0 if x == y else 0.0 for y in b] for x in a]).reshape(len(a), len(b))
    ia = [i for i, x in enumerate(a) if x in ix]
    jb = [j for j, y in enumerate(b) if y in ix]
    if ia and jb:
        S[np.ix_(ia, jb)] = U[[ix[a[i]] for i in ia]] @ U[[ix[b[j]] for j in jb]].T
    return S


def soft_match(cand, ref) -> tuple[float, float, float]:
    """Def 5 (greedy matching, BERTScore's form without idf weights or rescaling): P = mean over y of the best cosine
    in r, R = mean over r of the best cosine in y, F their harmonic mean."""
    if not cand or not ref:
        return 0.0, 0.0, 0.0
    S = similarity(cand, ref)
    p, r = float(S.max(1).mean()), float(S.max(0).mean())
    return p, r, (0.0 if p + r <= 0 else 2 * p * r / (p + r))


def cosine(a: str, b: str) -> float:
    ix, U = soft_vectors()
    return float(U[ix[a]] @ U[ix[b]])


# --------------------------------------------------------------------------------------------
# 6. The metric as a decision rule: minimum Bayes risk (Thm 4).
# --------------------------------------------------------------------------------------------


def mbr_choice(hyps, support, probs, utility) -> int:
    """Thm 4: argmax over hypotheses of Σ_s p(s) U(h, s) (the first maximizer)."""
    gains = [sum(p * utility(h, s) for s, p in zip(support, probs)) for h in hyps]
    return int(np.argmax(gains))


def bleu_plus_one(h, s) -> float:
    """Sentence BLEU with Lin and Och's add-one smoothing (orders n ≥ 2), against one reference: Thm 4's utility."""
    return sentence_bleu(h, [s], smooth="add-k")


# The toy of Thm 4(b): a two-word string is the mode (0.2), and eight one-word edits of a seven-word sentence share the
# rest (0.1 each). Each edit agrees with the others on most n-grams, so its expected BLEU beats the mode's.
TOY_CENTER = tuple("the quarter closed with stronger margins overall".split())


def toy_distribution() -> tuple[list[Tokens], np.ndarray]:
    edits = [TOY_CENTER[:i] + ("x" + str(i),) + TOY_CENTER[i + 1:] for i in range(len(TOY_CENTER))]
    edits.append(TOY_CENTER[:-1])
    return [("margins", "held")] + edits, np.array([0.2] + [0.1] * len(edits))


def toy_mbr() -> dict:
    sup, p = toy_distribution()
    exact = [sum(q * (h == s) for s, q in zip(sup, p)) for h in sup]
    gains = [sum(q * bleu_plus_one(h, s) for s, q in zip(sup, p)) for h in sup]
    order = np.argsort(gains)[::-1]
    return {"mode": int(np.argmax(p)), "mbr_exact_match": int(np.argmax(exact)), "mbr_bleu": int(order[0]),
            "gains": gains, "gap": float(gains[order[0]] - gains[order[1]])}


def hoeffding_samples(n_hyp: int, gap: float, delta: float) -> int:
    """Thm 4(c): S ≥ 2 log(2|H|/δ)/Δ² samples make every estimated expected utility lie within Δ/2 of its mean with
    probability ≥ 1 − δ (Hoeffding plus a union bound), so the estimated MBR choice is the exact one."""
    return int(math.ceil(2 * math.log(2 * n_hyp / delta) / gap ** 2))


TOY_TRIALS = 400


def toy_monte_carlo(S: int) -> float:
    """The share of TOY_TRIALS sample sets of size S on which Monte Carlo MBR picks the exact MBR string."""
    sup, p = toy_distribution()
    U = np.array([[bleu_plus_one(h, s) for s in sup] for h in sup])
    target = toy_mbr()["mbr_bleu"]
    rng = _rng(4000)
    hits = 0
    for _ in range(TOY_TRIALS):
        counts = np.bincount(rng.choice(len(sup), S, p=p), minlength=len(sup))
        hits += int(np.argmax(U @ counts)) == target
    return hits / TOY_TRIALS


# The corpus engines: Topic 14's prompts and its trigram (Topic 3's Kneser–Ney, <unk> masked).
MBR_SAMPLES = 64


@lru_cache(maxsize=1)
def references() -> tuple[tuple[Tokens, ...], ...]:
    """For each of Topic 14's prompts, the continuations of every held-out MD&A sentence it opens (generate_corpus(100, 1)):
    Topic 14's reference_lengths dictionary, keeping the words."""
    rest: dict[Tokens, list[Tokens]] = {}
    for d in fc.generate_corpus(100, 1):
        if d.doc_type == "mdna":
            for s in ng.split_sentences(d.text):
                rest.setdefault(tuple(s[:3]), []).append(tuple(s[3:]))
    return tuple(tuple(rest[p]) for p in bs.prompts())


def _words(lm, toks) -> Tokens:
    if toks is None:
        return ()
    return tuple(lm.words(toks[:-1] if toks and toks[-1] == lm.eos else toks))


@lru_cache(maxsize=1)
def decoded() -> dict[str, tuple[Tokens, ...]]:
    """Topic 14's trigram outputs per prompt: the exact mode (Dijkstra), beam(16) at α = 0 and α = 1 (one shared search)."""
    lm = bs.engine("trigram")
    modes = bs.exact_modes("trigram")
    beams = [bs.beam_search_multi(lm, lm.root(p), 16, max_len=bs.TRIGRAM_MAX_LEN, alphas=(0.0, 1.0))
             for p in bs.prompts()]
    return {"mode": tuple(_words(lm, m.tokens) for m in modes),
            "beam-16": tuple(_words(lm, b[0].tokens) for b in beams),
            "norm-16": tuple(_words(lm, b[1].tokens) for b in beams)}


@lru_cache(maxsize=None)
def sample_set(stream: int) -> tuple[tuple[Tokens, ...], ...]:
    """MBR_SAMPLES ancestral samples per prompt from the trigram (temperature 1, at most TRIGRAM_MAX_LEN tokens), from
    one stream; each step draws one uniform per open string and inverts its CDF, transforming each Markov state once."""
    lm = bs.engine("trigram")
    rng = _rng(stream)
    roots = [lm.root(p) for p in bs.prompts() for _ in range(MBR_SAMPLES)]
    seqs = [list(r) for r in roots]
    cache: dict = {}
    open_ = list(range(len(seqs)))
    for _ in range(bs.TRIGRAM_MAX_LEN):
        if not open_:
            break
        u = rng.random(len(open_))
        nxt = []
        for i, x in zip(open_, u):
            key = tuple(seqs[i][-2:])
            cdf = cache.get(key)
            if cdf is None:
                cdf = cache[key] = np.cumsum(np.exp(lm.next_logp([key])[0]))
            j = min(int(np.searchsorted(cdf, x * cdf[-1], side="right")), len(cdf) - 1)
            seqs[i].append(j)
            if j != lm.eos:
                nxt.append(i)
        open_ = nxt
    words = [_words(lm, tuple(s[len(r):])) for s, r in zip(seqs, roots)]
    return tuple(tuple(words[k * MBR_SAMPLES:(k + 1) * MBR_SAMPLES]) for k in range(len(bs.prompts())))


def samples() -> tuple[tuple[Tokens, ...], ...]:
    return sample_set(4010)


def utility_matrix(hyps, refs, N: int = N_MAX) -> np.ndarray:
    """U[h, s] = BLEU+1(h, s) for every hypothesis and pseudo-reference, vectorized over n-gram count matrices."""
    H, S = len(hyps), len(refs)
    cl = np.array([len(h) for h in hyps], dtype=float)
    rl = np.array([len(s) for s in refs], dtype=float)
    logp = np.zeros((H, S))
    for n in range(1, N + 1):
        index: dict = {}
        ch, cr = [ngrams(h, n) for h in hyps], [ngrams(s, n) for s in refs]
        for c in ch + cr:
            for g in c:
                index.setdefault(g, len(index))
        A, B = np.zeros((H, max(len(index), 1))), np.zeros((S, max(len(index), 1)))
        for i, c in enumerate(ch):
            for g, k in c.items():
                A[i, index[g]] = k
        for i, c in enumerate(cr):
            for g, k in c.items():
                B[i, index[g]] = k
        m = np.minimum(A[:, None, :], B[None, :, :]).sum(-1)
        t = np.maximum(cl - n + 1, 0)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = m / t if n == 1 else (m + 1) / (t + 1)
            logp += np.log(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        bp = np.where(cl[:, None] > rl[None, :], 1.0, np.exp(1 - rl[None, :] / cl[:, None]))
        U = bp * np.exp(logp / N)
    U[~np.isfinite(U)] = 0.0
    U[cl == 0] = 0.0
    return U


@lru_cache(maxsize=1)
def mbr_outputs() -> dict[str, tuple[Tokens, ...]]:
    """Per prompt: BLEU-MBR over the distinct samples and Topic 14's three outputs, with the 64 samples as pseudo-
    references; and exact-match MBR over the samples, which is the most frequent sample (Thm 4(a)'s Monte Carlo form)."""
    dec = decoded()
    mbr, freq = [], []
    for i, S in enumerate(samples()):
        hyps = list(dict.fromkeys(list(S) + [dec[k][i] for k in ("mode", "beam-16", "norm-16")]))
        U = utility_matrix(hyps, S)
        mbr.append(hyps[int(np.argmax(U.mean(1)))])
        c = Counter(S)
        freq.append(max(c, key=lambda y: (c[y], -S.index(y))))
    return {"mbr": tuple(mbr), "sample-mode": tuple(freq), "sample": tuple(S[0] for S in samples())}


DECODING_SYSTEMS = ("mode", "beam-16", "norm-16", "sample", "mbr")


@lru_cache(maxsize=1)
def decoding_table() -> dict:
    """Corpus BLEU against all held-out references, and Thm 1's single-reference coordinates (the first reference)."""
    outs = {**decoded(), **mbr_outputs()}
    refs = references()
    first = [[R[0]] for R in refs]
    ref_len = float(np.mean([np.mean([len(x) for x in R]) for R in refs]))
    out = {}
    for k in DECODING_SYSTEMS:
        ys = outs[k]
        s_all = pool(bleu_stats(y, R) for y, R in zip(ys, refs))
        s_one = pool(bleu_stats(y, R) for y, R in zip(ys, first))
        out[k] = {"bleu": bleu(s_all), "length": float(np.mean([len(y) for y in ys])),
                  "empty": float(np.mean([len(y) == 0 for y in ys])), "bp": brevity_penalty(s_all.c, s_all.r),
                  "p": precisions(s_all), "x_one": s_one.c / s_one.r, "bleu_one": bleu(s_one),
                  "bound_one": float(envelope(s_one.c / s_one.r, corpus=True))}
    out["reference_length"] = ref_len
    out["refs_per_prompt"] = float(np.mean([len(R) for R in refs]))
    return out


SWEEP_X = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)
FILLER = "zzz"  # in no reference


def length_sweep() -> list[dict]:
    """Thm 1's tightness: each prompt's first reference cut to ⌈x|r|⌉ words (x ≤ 1), or extended by FILLER to
    ⌊x|r|⌉ words (x > 1), scored as a corpus against that reference."""
    first = [R[0] for R in references()]
    out = []
    for x in SWEEP_X:
        ys = [r[:max(1, math.ceil(x * len(r)))] if x <= 1 else r + (FILLER,) * (round(x * len(r)) - len(r))
              for r in first]
        s = pool(bleu_stats(y, [r]) for y, r in zip(ys, first))
        out.append({"x": s.c / s.r, "bleu": bleu(s), "bound": float(envelope(s.c / s.r, corpus=True)),
                    "segment_bound": float(envelope(s.c / s.r))})
    return out


def mbr_stability() -> float:
    """The share of prompts on which BLEU-MBR's choice is also drawn in a second, independent sample set."""
    return float(np.mean([y in set(T) for y, T in zip(mbr_outputs()["mbr"], sample_set(4011))]))


# --------------------------------------------------------------------------------------------
# 7–8. The finance answer key: facts, writers, systems (Prop 7, Cor 1, case study).
# --------------------------------------------------------------------------------------------

Fact = tuple[str, int]  # (aspect category, net polarity ±1)
ADJ_PAIRS = (("strong", "weak"), ("stronger", "weaker"), ("favorable", "unfavorable"))  # stand-in lexicon adjectives
OPENERS = ("{co} reported that", "for {co}", "at {co}")
K_REFS = 4  # reference writers per document


def doc_facts(doc, sentence_ids=None) -> list[Fact]:
    """Per aspect category, the sign of the summed polarities of its polar aspects (inside the given sentences, if
    any), in order of first polar mention; categories whose polarities cancel state no fact."""
    tot: Counter = Counter()
    order: list[str] = []
    for a in doc.aspects:
        if not a.polarity:
            continue
        if sentence_ids is not None and not any(doc.sentences[i][0] <= a.start < doc.sentences[i][1]
                                                for i in sentence_ids):
            continue
        tot[a.category] += a.polarity
        if a.category not in order:
            order.append(a.category)
    return [(c, int(np.sign(tot[c]))) for c in order if tot[c]]


@dataclass(frozen=True)
class Writer:
    """One writer's choices for a document's facts: per fact a term (of the category's three), a form (0: direction verb,
    1: polarity adjective), and a word index; an opener; and an order. Flipping a fact changes only its one verb or
    adjective, to the same index in the opposite list."""
    terms: tuple[int, ...]
    forms: tuple[int, ...]
    words: tuple[int, ...]
    opener: int
    order: tuple[int, ...]


def draw_writer(m: int, rng: np.random.Generator) -> Writer:
    terms, forms, words = (tuple(int(x) for x in rng.integers(k, size=m)) for k in (3, 2, 3))
    opener = int(rng.integers(len(OPENERS)))
    order = tuple(range(m)) if rng.random() < 0.5 else tuple(int(i) for i in rng.permutation(m))
    return Writer(terms, forms, words, opener, order)


def clause(fact: Fact, term: int, form: int, word: int) -> str:
    cat, pol = fact
    valence, terms = fc.ASPECT_CATEGORIES[cat]
    t = terms[term]
    if form == 0:  # the direction of the move: "costs fell" is good news
        return f"{t} {(fc._UP if pol * valence > 0 else fc._DOWN)[word]}"
    be = "were" if t.endswith("s") else "was"
    return f"{t} {be} {ADJ_PAIRS[word][0 if pol > 0 else 1]}"


def render(company: str, facts: list[Fact], w: Writer, keep=None, flip=()) -> tuple[Tokens, list[Fact]]:
    """A summary's tokens and the facts it states. keep: indices kept (all by default); flip: indices reversed."""
    stated, parts = [], [OPENERS[w.opener].format(co=company)]
    order = [i for i in w.order if keep is None or i in keep]
    for k, i in enumerate(order):
        f = (facts[i][0], -facts[i][1]) if i in flip else facts[i]
        stated.append(f)
        parts.append(("and " if k == len(order) - 1 and k > 0 else "") + clause(f, w.terms[i], w.forms[i], w.words[i]))
    return tuple(fc._tokens(" ".join(parts))), stated


def fact_f1(stated, gold) -> float:
    """F1 between the stated and the gold fact sets (a fact is a category with its sign)."""
    s, g = set(stated), set(gold)
    hit = len(s & g)
    return 0.0 if hit == 0 else 2 * hit / (len(s) + len(g))


def sentence_similarity(a, b) -> float:
    """Mihalcea and Tarau's TextRank sentence similarity: shared word types over log|a| + log|b|."""
    den = math.log(len(a)) + math.log(len(b)) if len(a) > 0 and len(b) > 0 else 0.0
    return len(set(a) & set(b)) / den if den > 0 else 0.0


def textrank_sentences(sents, k: int) -> list[int]:
    """Topic 26's PageRank (d = 0.85) on the sentence-similarity graph; the top k sentences, in document order."""
    n = len(sents)
    W = np.array([[sentence_similarity(sents[i], sents[j]) if i != j else 0.0 for j in range(n)] for i in range(n)])
    pi, _ = ck.pagerank(W, 0.85, 1e-12)
    top = sorted(np.argsort(-pi, kind="stable")[:k])
    return [int(i) for i in top]


NOISE_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # the lab's flip slider
SYSTEMS = ("writer", "noisy-0.1", "noisy-0.3", "noisy-0.5", "terse", "flipped", "extract", "textrank-3", "lead-3",
           "random-3")
FLIP_FAMILY = ("writer", "noisy-0.1", "noisy-0.3", "noisy-0.5", "flipped")  # one surface, quality set by flips only


@dataclass(frozen=True)
class Item:
    doc_id: str
    company: str
    facts: tuple[Fact, ...]
    refs: tuple[Tokens, ...]
    outputs: dict  # system → (tokens, stated facts)
    sentences: tuple[Tokens, ...]


@lru_cache(maxsize=1)
def finance_items() -> tuple[Item, ...]:
    """The test documents (generate_corpus(100, 1), every type) that state at least two facts. Per document: K_REFS
    reference writers and one candidate writer, all independent; the candidate writer's summary flipped at every rate in
    NOISE_RATES with common uniforms (fact i flips when u_i < e, so "noisy-1.0" is "flipped"); "terse" keeps each fact
    with probability ½ (at least one); and four extractors that copy document sentences."""
    rng_w, rng_u, rng_x = _rng(4020), _rng(4021), _rng(4022)
    items = []
    for d in fc.generate_corpus(100, 1):
        facts = doc_facts(d)
        if len(facts) < 2:
            continue
        m = len(facts)
        writers = [draw_writer(m, rng_w) for _ in range(K_REFS + 1)]
        refs = tuple(render(d.company, facts, w)[0] for w in writers[:K_REFS])
        cw = writers[K_REFS]
        u, v = rng_u.random(m), rng_u.random(m)
        sents = tuple(tuple(fc._tokens(d.text[s:e])) for s, e in d.sentences)
        out = {f"noisy-{e:g}": render(d.company, facts, cw, flip={i for i in range(m) if u[i] < e}) for e in NOISE_RATES}
        out["writer"], out["flipped"] = out["noisy-0"], out["noisy-1"]
        keep = {i for i in range(m) if v[i] < 0.5} or {int(np.argmin(v))}
        out["terse"] = render(d.company, facts, cw, keep=keep)

        def extract(ids):
            ids = sorted(ids)
            return tuple(w for i in ids for w in sents[i]), doc_facts(d, ids)

        polar = [i for i, (s, e) in enumerate(d.sentences) if any(a.polarity and s <= a.start < e for a in d.aspects)]
        out["extract"] = extract(polar)
        out["textrank-3"] = extract(textrank_sentences(sents, 3))
        out["lead-3"] = extract(range(3))
        out["random-3"] = extract(rng_x.choice(len(sents), 3, replace=False))
        items.append(Item(d.doc_id, d.company, tuple(facts), refs, out, sents))
    return tuple(items)


METRICS = ("bleu", "rouge-1", "rouge-2", "rouge-l", "chrf", "soft")


def segment_scores(y, refs) -> dict[str, float]:
    """One candidate's segment scores against its references: BLEU+1 (all references), and the best F over references for
    ROUGE-1/2/L, chrF, and the soft match."""
    return {"bleu": sentence_bleu(y, refs, smooth="add-k"),
            "rouge-1": best_f(lambda a, b: rouge_n(a, b, 1), y, refs),
            "rouge-2": best_f(lambda a, b: rouge_n(a, b, 2), y, refs),
            "rouge-l": best_f(rouge_l, y, refs), "chrf": chrf_best(y, refs), "soft": best_f(soft_match, y, refs)}


@lru_cache(maxsize=1)
def finance_scores() -> dict:
    """Per system: per-document segment scores, fact F1, and BLEU statistics (for corpus BLEU over any subset)."""
    items = finance_items()
    names = SYSTEMS + tuple(f"noisy-{e:g}" for e in NOISE_RATES if f"noisy-{e:g}" not in SYSTEMS)
    out = {}
    for s in dict.fromkeys(names):
        seg = [segment_scores(it.outputs[s][0], it.refs) for it in items]
        out[s] = {"segment": {k: np.array([x[k] for x in seg]) for k in METRICS},
                  "f1": np.array([fact_f1(it.outputs[s][1], it.facts) for it in items]),
                  "stats": [bleu_stats(it.outputs[s][0], it.refs) for it in items]}
    return out


def system_score(s: str, metric: str, idx=None) -> float:
    """A system's corpus-level score: pooled corpus BLEU, or the mean segment score for the others."""
    sc = finance_scores()[s]
    idx = range(len(sc["f1"])) if idx is None else idx
    if metric == "bleu":
        return bleu(pool(sc["stats"][i] for i in idx))
    return float(np.mean(sc["segment"][metric][list(idx)]))


def system_table() -> dict:
    return {s: {**{m: system_score(s, m) for m in METRICS}, "f1": float(finance_scores()[s]["f1"].mean()),
                "length": float(np.mean([len(it.outputs[s][0]) for it in finance_items()]))} for s in SYSTEMS}


def kendall(x, y) -> float:
    t = stats.kendalltau(x, y).statistic
    return float(t)


def segment_tau(metric: str, systems=SYSTEMS) -> float:
    """Segment level: per document, Kendall's τ-b between the systems' segment scores and their fact F1, averaged over the
    documents where both vary."""
    sc = finance_scores()
    vals = []
    for i in range(len(finance_items())):
        x = [sc[s]["segment"][metric][i] for s in systems]
        y = [sc[s]["f1"][i] for s in systems]
        if len(set(x)) > 1 and len(set(y)) > 1:
            vals.append(kendall(x, y))
    return float(np.mean(vals))


def system_tau(metric: str, systems=SYSTEMS, idx=None) -> float:
    """System level: Kendall's τ-b between the systems' corpus scores and mean fact F1 (over the documents idx)."""
    sc = finance_scores()
    idx = list(range(len(finance_items()))) if idx is None else list(idx)
    return kendall([system_score(s, metric, idx) for s in systems], [sc[s]["f1"][idx].mean() for s in systems])


TAU_N = (1, 2, 5, 10, 20, 45, 90)
TAU_DRAWS = 100


@lru_cache(maxsize=1)
def tau_curve() -> dict[str, list[float]]:
    """Prop 7 on the corpus: mean system-level τ when every system is scored on the same n random documents, for each n
    in TAU_N (TAU_DRAWS subsets each; n = 90 is the whole set)."""
    rng = _rng(4030)
    n_docs = len(finance_items())
    subsets = {n: [rng.choice(n_docs, n, replace=False) for _ in range(TAU_DRAWS if n < n_docs else 1)] for n in TAU_N}
    out = {}
    for m in METRICS:
        row = []
        for n in TAU_N:
            ts = [system_tau(m, idx=idx) for idx in subsets[n]]
            row.append(float(np.nanmean(ts)))
        out[m] = row
    return out


def reversed_pairs(metric: str) -> list[tuple[str, str]]:
    """Pairs (a, b) where a has the higher mean fact F1 but the lower corpus score."""
    tab = system_table()
    return [(a, b) for a in SYSTEMS for b in SYSTEMS
            if tab[a]["f1"] > tab[b]["f1"] and tab[a][metric] < tab[b][metric]]


def bias_model_tau(n: int, rng: np.random.Generator, q, beta, b: float, sigma: float) -> float:
    """Prop 7's model: segment score b·q_j + β_j + ε (ε ~ N(0, σ²)), averaged over n segments per system."""
    q, beta = np.asarray(q), np.asarray(beta)
    s = b * q + beta + rng.normal(0, sigma, (n, len(q))).mean(0)
    return kendall(s, q)


def flip_curve() -> dict[str, list[float]]:
    """The lab's slider: the candidate writer with each fact flipped when u_i < e, per metric and fact F1."""
    out = {m: [system_score(f"noisy-{e:g}", m) for e in NOISE_RATES] for m in METRICS}
    out["f1"] = [float(finance_scores()[f"noisy-{e:g}"]["f1"].mean()) for e in NOISE_RATES]
    return out


def flip_bound() -> dict:
    """Cor 1 per document: flipping all m facts is m substitutions, so best-reference ROUGE-1 recall falls by at most
    m/|r| and ROUGE-2 recall by at most 2m/(|r| − 1) against each reference."""
    worst = {1: 0.0, 2: 0.0}
    for it in finance_items():
        y, z = it.outputs["writer"][0], it.outputs["flipped"][0]
        m = len(it.facts)
        for r in it.refs:
            for n in (1, 2):
                drop = rouge_n(y, r, n)[1] - rouge_n(z, r, n)[1]
                bound = n * m / (len(r) - n + 1)
                worst[n] = max(worst[n], drop - bound)
    return worst


def flipped_beats_extract() -> dict[str, float]:
    """Per metric, the share of documents on which the all-flipped summary outscores the fact-bearing extract."""
    sc = finance_scores()
    return {m: float(np.mean(sc["flipped"]["segment"][m] > sc["extract"]["segment"][m])) for m in METRICS}


CENSUS_PERMS = 200


@lru_cache(maxsize=1)
def chunk_census() -> dict:
    """Prop 5 over the candidate writers' summaries: CENSUS_PERMS uniformly random chunk orders per summary (one stream);
    log10_orders is the mean log10 of the number of orders, k!."""
    rng = _rng(4040)
    rows = []
    for it in finance_items():
        y = it.outputs["writer"][0]
        k = len(chunks(y, it.refs))
        rows.append(permutation_census(y, it.refs, [rng.permutation(k) for _ in range(CENSUS_PERMS)]))
    return {"docs": len(rows), "chunks": float(np.mean([r["chunks"] for r in rows])),
            "log10_orders": float(np.mean([math.log10(math.factorial(r["chunks"])) for r in rows])),
            "equal_share": float(np.mean([r["equal"] / r["perms"] for r in rows])),
            "lowered": int(sum(r["lowered"] for r in rows)),
            "multi_rouge_l": float(np.mean([r["rouge_l_values"] > 1 for r in rows if r["chunks"] > 1])),
            "multi_chunk": float(np.mean([r["chunks"] > 1 for r in rows]))}


def zero_four_gram() -> dict[str, float]:
    """Prop 3: the share of segments with no matched 4-gram (unsmoothed sentence BLEU = 0), per system."""
    sc = finance_scores()
    return {s: float(np.mean([st.m[3] == 0 for st in sc[s]["stats"]])) for s in SYSTEMS}


def corpus_vs_sentence() -> dict:
    """Prop 2 on the finance systems: pairs ordered differently by corpus BLEU and by mean sentence BLEU+1."""
    tab = {s: (system_score(s, "bleu"), float(finance_scores()[s]["segment"]["bleu"].mean())) for s in SYSTEMS}
    disc = [(a, b) for a, b in itertools.combinations(SYSTEMS, 2)
            if (tab[a][0] - tab[b][0]) * (tab[a][1] - tab[b][1]) < 0]
    return {"discordant": disc, "table": tab}


def soft_flip_shift() -> dict:
    """Prop 6 on the corpus: the largest per-substitution change of soft P and R over all writer/flipped pairs, against
    the bound ‖u − u'‖ (per document, the largest distance among its flipped word pairs)."""
    ix, U = soft_vectors()
    worst_p = worst_r = 0.0
    for it in finance_items():
        y, z = it.outputs["writer"][0], it.outputs["flipped"][0]
        diffs = [(a, b) for a, b in zip(y, z) if a != b]
        dist = max(float(np.linalg.norm(U[ix[a]] - U[ix[b]])) for a, b in diffs)
        k = len(diffs)
        for r in it.refs:
            p0, r0, _ = soft_match(y, r)
            p1, r1, _ = soft_match(z, r)
            worst_p = max(worst_p, abs(p1 - p0) - k * dist / len(y))
            worst_r = max(worst_r, abs(r1 - r0) - k * dist)
    return {"excess_p": worst_p, "excess_r": worst_r}


# --------------------------------------------------------------------------------------------
# The lab's payload.
# --------------------------------------------------------------------------------------------

GRID_SYSTEMS = ("writer", "flipped", "extract")


def grid_example() -> int:
    """The lab's example document: the first with three or four facts on which the flipped summary's BLEU+1 beats the
    extract's and the extract has at most 40 words."""
    sc = finance_scores()
    for i, it in enumerate(finance_items()):
        if (len(it.facts) in (3, 4) and sc["flipped"]["segment"]["bleu"][i] > sc["extract"]["segment"]["bleu"][i]
                and len(it.outputs["extract"][0]) <= 40):
            return i
    raise AssertionError("no grid example")


def _grid_payload() -> dict:
    i = grid_example()
    it = finance_items()[i]
    ref = it.refs[0]
    out = {"doc": it.doc_id, "company": it.company, "facts": [list(f) for f in it.facts], "ref": list(ref),
           "refs": [" ".join(r) for r in it.refs], "systems": {}}
    for s in GRID_SYSTEMS:
        y, stated = it.outputs[s]
        sc = segment_scores(y, [ref])
        out["systems"][s] = {"tokens": list(y), "matches": [list(p) for p in match_grid(y, ref)],
                             "clip1": [list(p) for p in clipped_pairs(y, ref, 1)],
                             "clip2": [list(p) for p in clipped_pairs(y, ref, 2)],
                             "lcs": [list(p) for p in lcs_path(y, ref)],
                             "scores": {k: _r(v) for k, v in sc.items()}, "f1": _r(fact_f1(stated, it.facts))}
    return out


def mbr_examples(n: int = 5) -> list[int]:
    """The lab's MBR table: the first n prompts (sorted) whose exact mode is empty and whose MBR choice is not."""
    dec, mb = decoded(), mbr_outputs()
    return [i for i in range(len(bs.prompts())) if not dec["mode"][i] and mb["mbr"][i]][:n]


def viz_payload() -> dict:
    """Every number the lab shows, at 3 decimals."""
    tab, dec = system_table(), decoding_table()
    outs = {**decoded(), **mbr_outputs()}
    corr = {m: {"segment": _r(segment_tau(m)), "system": _r(system_tau(m)),
                "family": _r(system_tau(m, FLIP_FAMILY))} for m in METRICS}
    return {
        "grid": _grid_payload(),
        "envelope": {"sweep": [{k: _r(v) for k, v in row.items()} for row in length_sweep()],
                     "systems": {k: {"x": _r(dec[k]["x_one"]), "bleu": _r(dec[k]["bleu_one"])}
                                 for k in DECODING_SYSTEMS}},
        "flip": {"rates": list(NOISE_RATES), **{k: _r(v) for k, v in flip_curve().items()},
                 "extract": {m: _r(tab["extract"][m]) for m in (*METRICS, "f1")}},
        "systems": {s: {k: _r(v) for k, v in tab[s].items()} for s in SYSTEMS},
        "correlation": {"metrics": corr, "n": list(TAU_N), "curve": {k: _r(v) for k, v in tau_curve().items()}},
        "mbr": {"systems": {k: {"bleu": _r(dec[k]["bleu"]), "length": _r(dec[k]["length"]),
                                "empty": _r(dec[k]["empty"]), "bp": _r(dec[k]["bp"])} for k in DECODING_SYSTEMS},
                "reference_length": _r(dec["reference_length"]),
                "examples": [{"prompt": " ".join(bs.prompts()[i]),
                              "refs": [" ".join(r) for r in references()[i]],
                              **{k: " ".join(outs[k][i]) for k in DECODING_SYSTEMS},
                              "scores": {k: _r(sentence_bleu(outs[k][i], references()[i], smooth="add-k"))
                                         for k in DECODING_SYSTEMS}}
                             for i in mbr_examples()]},
    }


# --------------------------------------------------------------------------------------------
# Tests: every claim on the page.
# --------------------------------------------------------------------------------------------

_ALPH = tuple("abcdefgh")


def _random_pairs(k: int, n: int, max_refs: int = 3):
    rng = _rng(k)
    out = []
    for _ in range(n):
        refs = [tuple(rng.choice(_ALPH, int(rng.integers(3, 14)))) for _ in range(int(rng.integers(1, max_refs + 1)))]
        cand = tuple(rng.choice(_ALPH, int(rng.integers(2, 16))))
        out.append((cand, refs))
    return out


def test_prop1_clipping_is_a_maximum_matching():
    """Prop 1: m_n equals the maximum matching between candidate occurrences and reference slots, on random instances."""
    for cand, refs in _random_pairs(4100, 300):
        for n in (1, 2, 3):
            assert clipped(cand, refs, n) == matching_size(cand, refs, n)
    # repetition earns credit only as often as a reference holds the word
    s = bleu_stats(("the",) * 7, [tuple("the cat sat on the mat".split())])
    assert s.m[0] == 2 and precisions(s)[0] == 2 / 7


def test_prop1_matching_picture():
    """The lab's clipped pairs form a matching of the stated size against one reference."""
    for cand, refs in _random_pairs(4101, 200, 1):
        for n in (1, 2):
            pairs = clipped_pairs(cand, refs[0], n)
            assert len(pairs) == clipped(cand, refs, n)
            assert len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
            assert all(cand[i:i + n] == refs[0][j:j + n] for i, j in pairs)


def test_bleu_matches_sacrebleu():
    """Def 2 equals sacrebleu's corpus BLEU (tokenize="none"), and sentence BLEU without and with add-k smoothing."""
    import sacrebleu
    from sacrebleu.metrics import BLEU

    pairs = _random_pairs(4102, 300)
    K = max(len(r) for _, r in pairs)
    # sacrebleu takes reference streams of equal count: pad by repeating the first reference (clipping takes the max,
    # closest length is unchanged by a duplicate)
    streams = [[" ".join(r[k] if k < len(r) else r[0]) for _, r in pairs] for k in range(K)]
    ours = corpus_bleu([c for c, _ in pairs], [r for _, r in pairs])
    theirs = sacrebleu.corpus_bleu([" ".join(c) for c, _ in pairs], streams, tokenize="none").score / 100
    assert abs(ours - theirs) < 1e-12, (ours, theirs)
    for sm, ours_sm in (("none", "none"), ("add-k", "add-k"), ("exp", "exp")):
        m = BLEU(tokenize="none", smooth_method=sm, effective_order=False, smooth_value=1 if sm == "add-k" else None)
        for c, r in pairs:
            if len(c) < N_MAX:
                continue
            a = sentence_bleu(c, r, smooth=ours_sm)
            b = m.sentence_score(" ".join(c), [" ".join(x) for x in r]).score / 100
            assert abs(a - b) < 1e-12, (sm, a, b)


def test_rouge_matches_rouge_score():
    """Def 3 equals rouge-score's ROUGE-1, ROUGE-2, and ROUGE-L (no stemming), precision, recall, and F."""
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    for cand, refs in _random_pairs(4103, 300, 1):
        got = sc.score(" ".join(refs[0]), " ".join(cand))
        for key, ours in (("rouge1", rouge_n(cand, refs[0], 1)), ("rouge2", rouge_n(cand, refs[0], 2)),
                          ("rougeL", rouge_l(cand, refs[0]))):
            assert np.allclose(ours, (got[key].precision, got[key].recall, got[key].fmeasure), atol=1e-12)
    # and on the finance summaries (lowercase words only, so rouge-score's tokenizer agrees with fc._tokens)
    for it in finance_items()[:20]:
        y = it.outputs["writer"][0]
        got = sc.score(" ".join(it.refs[0]), " ".join(y))
        assert abs(got["rougeL"].fmeasure - rouge_l(y, it.refs[0])[2]) < 1e-12


def test_chrf_matches_sacrebleu():
    """Def 4 equals sacrebleu's sentence chrF (one reference)."""
    import sacrebleu

    for cand, refs in _random_pairs(4104, 200, 1):
        a, b = " ".join(cand), " ".join(refs[0])
        assert abs(chrf(a, b) - sacrebleu.sentence_chrf(a, [b]).score / 100) < 1e-12


def test_thm1_length_envelope():
    """Thm 1: with one reference per segment, BLEU ≤ g(c/r) for a segment and the corpus form for a corpus; the general
    bound BP·min(1, ρ/c)^{1/N} holds with several references; the left branch is attained by reference prefixes."""
    for cand, refs in _random_pairs(4105, 400, 1):
        s = bleu_stats(cand, refs)
        assert bleu(s) <= envelope(s.c / s.r) + 1e-12
        assert bleu(s, "add-k") <= envelope(s.c / s.r) + 1e-12 or s.c < s.r  # smoothing can exceed it only when short
    pairs = _random_pairs(4106, 400, 3)
    for k in range(0, 400, 20):
        chunk = pairs[k:k + 20]
        s = pool(bleu_stats(c, r) for c, r in chunk)
        assert bleu(s) <= envelope_bound(s) + 1e-12
        one = pool(bleu_stats(c, r[:1]) for c, r in chunk)
        assert bleu(one) <= envelope(one.c / one.r, corpus=True) + 1e-12
    for row in length_sweep():
        assert row["bleu"] <= row["bound"] + 1e-12
        if row["x"] <= 1:
            assert abs(row["bleu"] - row["bound"]) < 1e-12  # prefixes: every precision is 1, BLEU = BP
    # the envelope's only maximum is x = 1
    xs = np.append(np.linspace(0.05, 4, 4000), 1.0)
    for corpus in (False, True):
        g = envelope(xs, corpus=corpus)
        assert abs(xs[np.argmax(g)] - 1) < 2e-3 and abs(g.max() - 1) < 1e-5


def test_thm1_decoding_points_under_envelope():
    """The decoding systems sit under the corpus envelope (single reference), and the short mode is penalized."""
    tab = decoding_table()
    for k in DECODING_SYSTEMS:
        assert tab[k]["bleu_one"] <= tab[k]["bound_one"] + 1e-12
    assert tab["mode"]["x_one"] < 0.5 and tab["mode"]["bp"] < 0.2


def test_prop2_corpus_is_not_mean_sentence():
    """Prop 2: A wins both sentences, B wins the corpus; pooled precisions are mediants of the segment ratios."""
    s = simpson()
    assert all(a > b for a, b in zip(s["A"]["sentence"], s["B"]["sentence"]))
    assert s["A"]["corpus"] < s["B"]["corpus"]
    pb = pool(bleu_stats(y, [r]) for y, r in zip(SIMPSON_B, SIMPSON_REFS))
    assert brevity_penalty(pb.c, pb.r) == 1.0
    rng = _rng(4108)
    pairs = _random_pairs(4109, 400)
    for _ in range(100):
        idx = rng.choice(400, 6, replace=False)
        st = [bleu_stats(*pairs[i]) for i in idx]
        for n in range(N_MAX):
            if any(x.t[n] for x in st):
                lo, mid, hi = mediant_bounds(st, n)
                assert lo - 1e-12 <= mid <= hi + 1e-12


def test_prop3_zero_frequency_and_shrinkage():
    """Prop 3: with no matched 4-gram, unsmoothed sentence BLEU is 0 whatever the lower orders; add-k precision is the
    interpolation t/(t + k)·p̂ + k/(t + k)·1 (Topic 3's shrinkage toward perfect precision), so smoothed BLEU is positive."""
    y = tuple("costs fell and revenue rose".split())
    r = tuple("revenue rose and costs fell".split())
    s = bleu_stats(y, [r])
    assert s.m[0] == 5 and s.m[3] == 0 and bleu(s) == 0.0 and bleu(s, "add-k") > 0
    for cand, refs in _random_pairs(4110, 200):
        st = bleu_stats(cand, refs)
        for k in (0.5, 1.0, 3.0):
            p = precisions(st, "add-k", k)
            for n in range(1, N_MAX):
                if st.t[n]:
                    lam = st.t[n] / (st.t[n] + k)
                    assert abs(p[n] - (lam * st.m[n] / st.t[n] + (1 - lam))) < 1e-12
        if st.m[0]:
            assert bleu(st, "add-k") > 0
    z = zero_four_gram()
    assert z["extract"] > 0.9 and z["writer"] < z["extract"]


def test_prop4_more_references():
    """Prop 4: adding a reference never lowers a clipped count, but it can lower BLEU through the brevity penalty."""
    for cand, refs in _random_pairs(4111, 300, 3):
        for n in range(1, N_MAX + 1):
            assert clipped(cand, refs[:1], n) <= clipped(cand, refs, n)
    mr = more_references()
    assert mr["m_one"] == mr["m_two"] and mr["r_one"] == 6 and mr["r_two"] == 9
    assert abs(mr["bleu_two"] - mr["bleu_one"] * math.exp(1 - 9 / 8)) < 1e-12 and mr["bleu_two"] < mr["bleu_one"]


def test_thm2_lcs_is_longest_path():
    """Thm 2: the DP equals brute-force LCS; the backtracked path is a chain of marked cells increasing in both
    coordinates; LCS ≤ clipped unigram matches, so every ROUGE-L component is at most ROUGE-1's."""
    rng = _rng(4112)
    for _ in range(300):
        a = tuple(rng.choice(_ALPH[:4], int(rng.integers(1, 10))))
        b = tuple(rng.choice(_ALPH[:4], int(rng.integers(1, 10))))
        L = lcs(a, b)
        assert L == lcs_brute(a, b)
        path = lcs_path(a, b)
        assert len(path) == L and all(a[i] == b[j] for i, j in path)
        assert all(i1 < i2 and j1 < j2 for (i1, j1), (i2, j2) in zip(path, path[1:]))
        assert L <= clipped(a, [b], 1)
        assert all(x <= y + 1e-12 for x, y in zip(rouge_l(a, b), rouge_n(a, b, 1)))


def test_thm3_lipschitz_in_substitutions():
    """Thm 3: k substitutions move every clipped count by at most nk and the LCS by at most k."""
    rng = _rng(4113)
    for cand, refs in _random_pairs(4114, 300):
        k = int(rng.integers(1, min(4, len(cand)) + 1))
        z = substitute(cand, rng.choice(len(cand), k, replace=False), rng, _ALPH + ("q",))
        for n in range(1, N_MAX + 1):
            assert abs(clipped(z, refs, n) - clipped(cand, refs, n)) <= n * k
        for r in refs:
            assert abs(lcs(z, r) - lcs(cand, r)) <= k


def test_cor1_flips_within_the_bound():
    """Cor 1 on the finance summaries: reversing all m facts is m one-word substitutions, ROUGE-N recall falls by at most
    nm/(|r| − n + 1) against every reference, and fact F1 falls from 1 to 0."""
    for it in finance_items():
        y, z = it.outputs["writer"][0], it.outputs["flipped"][0]
        assert len(y) == len(z) and sum(a != b for a, b in zip(y, z)) == len(it.facts)
    worst = flip_bound()
    assert worst[1] <= 1e-12 and worst[2] <= 1e-12
    sc = finance_scores()
    assert sc["writer"]["f1"].min() == 1.0 and sc["flipped"]["f1"].max() == 0.0


def test_prop5_chunk_permutations():
    """Prop 5: permuting chunks cut at unmatched bigrams never lowers a clipped count (random instances and every finance
    summary); most permutations keep BLEU exactly, while ROUGE-L moves."""
    for cand, refs in _random_pairs(4115, 150):
        if len(chunks(cand, refs)) <= 5:
            assert permutation_census(cand, refs)["lowered"] == 0
    c = chunk_census()
    assert c["lowered"] == 0 and c["equal_share"] > 0.5 and c["multi_chunk"] > 0.5


def test_prop6_soft_match_is_blind_to_flips():
    """Prop 6: the soft scores move within the ‖u − u'‖ bounds; rose/fell are each other's nearest neighbors."""
    assert cosine("rose", "fell") > 0.99
    ix, U = soft_vectors()
    sims = U @ U[ix["rose"]]
    order = np.argsort(-sims)
    assert ds.vocab()[order[1]] == "fell"
    sh = soft_flip_shift()
    assert sh["excess_p"] <= 1e-12 and sh["excess_r"] <= 1e-12
    tab = system_table()
    assert tab["writer"]["soft"] - tab["flipped"]["soft"] < tab["writer"]["rouge-1"] - tab["flipped"]["rouge-1"]


def test_case_study_flipped_beats_extract():
    """The case study's headline: the all-flipped summary (fact F1 0) outscores the fact-bearing extract (fact F1 1) on
    every metric at the corpus level."""
    tab = system_table()
    assert tab["extract"]["f1"] == 1.0 and tab["flipped"]["f1"] == 0.0
    for m in METRICS:
        assert tab["flipped"][m] > tab["extract"][m], m
    fb = flipped_beats_extract()
    assert min(fb.values()) > 0.5


def test_thm4_mbr_toy():
    """Thm 4: exact-match MBR is the mode; BLEU-MBR is not; Hoeffding's sample size makes Monte Carlo MBR exact at least
    1 − δ of the time."""
    t = toy_mbr()
    sup, p = toy_distribution()
    assert t["mbr_exact_match"] == t["mode"] == 0 and t["mbr_bleu"] != 0
    S = hoeffding_samples(len(sup), t["gap"], 0.05)
    assert toy_monte_carlo(S) >= 0.95


def test_thm4_mbr_on_the_corpus():
    """Thm 4 on the trigram: BLEU-MBR scores the highest corpus BLEU of the five systems; the mode is the shortest; the
    most frequent sample is the exact mode on most prompts (exact-match MBR estimates the mode)."""
    tab = decoding_table()
    assert max(DECODING_SYSTEMS, key=lambda k: tab[k]["bleu"]) == "mbr"
    assert min(DECODING_SYSTEMS, key=lambda k: tab[k]["length"]) == "mode"
    assert tab["mode"]["length"] < tab["reference_length"] / 2
    agree = np.mean([a == b for a, b in zip(mbr_outputs()["sample-mode"], decoded()["mode"])])
    assert agree > 0.5
    # the vectorized utility equals the scalar one
    S = samples()[0]
    hyps = list(dict.fromkeys(S))[:8]
    U = utility_matrix(hyps, S[:8])
    for i, h in enumerate(hyps):
        for j, s in enumerate(S[:8]):
            assert abs(U[i, j] - bleu_plus_one(h, s)) < 1e-12


def test_prop7_bias_survives_averaging():
    """Prop 7: in the model, system τ rises with n toward τ(b·q + β, q), which is below 1 when a bias gap exceeds a
    quality gap; on the corpus, systems that share one surface and differ only by flips are ranked perfectly while the
    full system set is not, and every metric puts the flipped summary above the extract."""
    rng = _rng(4120)
    q, beta = np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.array([0.0, 0.0, 0.6, 0.0, 0.0])
    lim = kendall(q + beta, q)
    assert lim < 1
    means = [np.mean([bias_model_tau(n, rng, q, beta, 1.0, 1.0) for _ in range(400)]) for n in (1, 10, 1000)]
    assert means[0] < means[1] < means[2] and abs(means[2] - lim) < 0.02
    for m in METRICS:
        assert abs(system_tau(m, FLIP_FAMILY) - 1) < 1e-12
        assert system_tau(m) < 0.6
        assert ("extract", "flipped") in reversed_pairs(m)
    for m in METRICS:
        curve = tau_curve()[m]
        assert curve[-1] < 1


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def summary() -> dict:
    """The numbers the page quotes, at 6 decimals."""
    out: dict = {}
    s = simpson()
    out.update({f"Prop 2 {k} sentence {i + 1}": v for k in "AB" for i, v in enumerate(s[k]["sentence"])})
    out.update({f"Prop 2 {k} corpus": s[k]["corpus"] for k in "AB"})
    mr = more_references()
    out["Prop 4 BLEU one ref"], out["Prop 4 BLEU two refs"] = mr["bleu_one"], mr["bleu_two"]
    t = toy_mbr()
    out["Thm 4 toy gains (mode, best)"] = f"{t['gains'][0]:.6f}, {max(t['gains']):.6f}"
    out["Thm 4 toy gap"] = t["gap"]
    out["Thm 4 Hoeffding S (δ = 0.05)"] = hoeffding_samples(len(toy_distribution()[0]), t["gap"], 0.05)
    tab = decoding_table()
    for k in DECODING_SYSTEMS:
        out[f"decoding {k}: BLEU, length, empty, BP"] = (f"{tab[k]['bleu']:.6f}, {tab[k]['length']:.6f}, "
                                                         f"{tab[k]['empty']:.6f}, {tab[k]['bp']:.6f}")
    out["decoding reference length / refs per prompt"] = f"{tab['reference_length']:.6f}, {tab['refs_per_prompt']:.6f}"
    out["decoding prompts"] = len(bs.prompts())
    out["MBR stability"] = mbr_stability()
    out["sample mode = exact mode"] = float(np.mean([a == b for a, b in zip(mbr_outputs()["sample-mode"],
                                                                        decoded()["mode"])]))
    items = finance_items()
    out["finance docs / facts per doc / ref length"] = (f"{len(items)}, {np.mean([len(i.facts) for i in items]):.6f}, "
                                                       f"{np.mean([len(r) for i in items for r in i.refs]):.6f}")
    for sname, row in system_table().items():
        out[f"system {sname}"] = ", ".join(f"{k} {v:.6f}" for k, v in row.items())
    for m in METRICS:
        out[f"τ {m}: segment, system, family"] = (f"{segment_tau(m):.6f}, {system_tau(m):.6f}, "
                                                   f"{system_tau(m, FLIP_FAMILY):.6f}")
        out[f"reversed pairs {m}"] = len(reversed_pairs(m))
    out["flipped beats extract (share of docs)"] = ", ".join(f"{k} {v:.6f}" for k, v in flipped_beats_extract().items())
    out["cos(rose, fell)"] = cosine("rose", "fell")
    out["‖rose − fell‖"] = math.sqrt(2 - 2 * cosine("rose", "fell"))
    for k, v in chunk_census().items():
        out[f"Prop 5 {k}"] = v
    out["Prop 3 zero 4-gram share"] = ", ".join(f"{k} {v:.6f}" for k, v in zero_four_gram().items())
    out["Prop 2 discordant finance pairs"] = str(corpus_vs_sentence()["discordant"])
    return out


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
            print(f"  {k:60} {v:.6f}" if isinstance(v, float) else f"  {k:60} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
