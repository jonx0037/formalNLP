"""Distributional semantics: PPMI and truncated SVD — the reference implementation for formalNLP's sixth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. Words as points: a word is the row of a word × context co-occurrence matrix. PMI rows are log-ratio
   coordinates of the context distribution (their P-average is a KL divergence), and PMI removes the
   background that raw counts share: as a word's contexts approach the corpus marginal, count rows all
   become parallel while PMI rows keep only the word-specific part.
2. What counts cannot tell: PMI ≤ min(−log P(w), −log P(c)); a pair seen once has plug-in PMI
   log N − log n_w − log n_c, a function of frequency alone; under independence PPMI is positive with
   probability P(X > μ), and a pair with μ < 1 seen once scores at least log(1/μ). Context-distribution
   smoothing lowers PMI exactly for contexts rarer than t* = (N/Z_α)^{1/(1−α)}.
3. Truncated SVD: W_p = U_d Σ_d^p. Word · context reproduces M_d for every p (Eckart–Young); word · word is
   the best rank-d approximation of (MMᵀ)^p, which for a symmetric PMI matrix with negative eigenvalues is
   |M|_d, not M_d. Adding word and context vectors keeps exactly the positive part.
4. Two kinds of low rank: a latent-class (mixture) model makes the ratio matrix R = e^{PMI} low rank and
   rank(R − 𝟙𝟙ᵀ) ≤ K − 1, while its PMI leaves rank K − 1 at second order in the strength of association (a
   Hadamard series); a log-bilinear model makes PMI low rank and R not.
5. Recovering planted structure: the shared corpus draws content nouns from one Pitman–Yor restaurant per
   sector, so each noun has a known sector. Hyperparameters are chosen on half the nouns (dev), over local
   contexts only, and every reported score is on the other half (test).
6. Finance case study: polarity words are distributional neighbors of their antonyms in a window, and
   Turney's semantic orientation recovers polarity only from document-level contexts.

Cross-checks: ``scipy.sparse.linalg.svds`` against ``numpy.linalg.svd``; the pair AUC against
``scipy.stats.mannwhitneyu``; the KL/MI identities against ``scipy.stats.entropy``; the Poisson null against
``scipy.stats.poisson``.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/distributional-semantics-ppmi-svd/distributional_semantics_ppmi_svd.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh, svds

_HERE = Path(__file__).resolve().parent
for _dep in ("finance-corpus", "zipf-heaps-vocabulary-statistics", "n-gram-models-smoothing"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import finance_corpus as fc  # noqa: E402
import n_gram_models_smoothing as ng  # noqa: E402
import zipf_heaps_vocabulary_statistics as zhv  # noqa: E402

# ONE stream per random-instance claim, each from default_rng([SEED, k]) (STARTER-PROMPT A1).
SEED = 20261023

NGram = tuple[str, ...]
Config = tuple[str, str, int | None, float | None]  # (context, weighting, d, p); d = None is the sparse matrix

CONTEXTS = ("L1", "L2", "L4", "L8", "sentence", "document")
WINDOWS = {"L1": 1, "L2": 2, "L4": 4, "L8": 8}  # symmetric windows inside a sentence
# Selection runs over local contexts only. The generator plants each noun's sector at the document level, so
# document co-occurrence reads the label directly (raw counts reach test 1-NN 1.000); it is reported as an oracle.
LOCAL_CONTEXTS = CONTEXTS[:-1]
ALPHA = 0.75  # context-distribution smoothing exponent (Levy, Goldberg, and Dagan 2015)
WEIGHTINGS: dict[str, tuple[float, float] | None] = {  # name → (α, k); None is the raw count matrix
    "counts": None, "ppmi": (1.0, 1.0), "ppmi_a": (ALPHA, 1.0), "sppmi5": (ALPHA, 5.0), "sppmi15": (ALPHA, 15.0)}
SVD_BASES = ("ppmi", "ppmi_a", "sppmi5")
SVD_DIMS = (10, 25, 50, 100, 200, 300)
SVD_POWERS = (0.0, 0.5, 1.0)
D_MAX = max(SVD_DIMS)
MIN_COUNT = 5  # benchmark nouns are seen at least this often
RARE_BAND = (2, 4)  # rarer nouns: never used for selection, so a second test set
SECTORS = fc.SECTORS
K_SECTORS = len(SECTORS)
LOWRANK_MIN_COUNT = 20  # Thm 3's corpus check: content nouns seen this often


# --------------------------------------------------------------------------------------------
# Data: the training text rebuilt from its own generator, so the planted restaurants are visible.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _generator_run() -> tuple[tuple[fc.FinanceDoc, ...], dict[str, frozenset[str]]]:
    """generate_corpus(300, 0), run from its own generator so the Pitman–Yor restaurants it drew content nouns
    from stay inspectable: (documents, restaurant name → the types it seated)."""
    gen = fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)
    tones = gen.rng.permutation(np.resize([-1, 0, 1], 300))
    docs = tuple(gen.doc(i, int(t)) for i, t in enumerate(tones))
    rest = {s: frozenset(py.types) for s, py in gen.by_sector.items()}
    rest["common"] = frozenset(gen.common.types)
    return docs, rest


@lru_cache(maxsize=1)
def documents() -> tuple[tuple[NGram, ...], ...]:
    """The training text as documents of sentences, split and tokenized as Topic 3 does (ng.split_sentences)."""
    return tuple(tuple(ng.split_sentences(d.text)) for d in _generator_run()[0])


@lru_cache(maxsize=1)
def sentences() -> tuple[NGram, ...]:
    """Topic 3's training sentences, ng.corpus()["train"] (asserted equal)."""
    return tuple(s for d in documents() for s in d)


@lru_cache(maxsize=1)
def vocab() -> tuple[str, ...]:
    """Training word types, sorted. Rows and columns of every matrix here (no </s>, no <unk>)."""
    return tuple(sorted({w for s in sentences() for w in s}))


@lru_cache(maxsize=1)
def index() -> dict[str, int]:
    return {w: i for i, w in enumerate(vocab())}


@lru_cache(maxsize=1)
def unigram_counts() -> np.ndarray:
    c = Counter(w for s in sentences() for w in s)
    return np.array([c[w] for w in vocab()], dtype=float)


def count_of(w: str) -> int:
    return int(unigram_counts()[index()[w]])


@lru_cache(maxsize=1)
def planted_labels() -> dict[str, str]:
    """Every training noun the generator drew from exactly one restaurant → that restaurant (a sector, or
    "common" for the shared one). Seeded nouns belong to one sector by construction, and pseudo-words are never
    reused across restaurants; the one seed shared by two sectors (banking and energy "reserves") is dropped."""
    rest = _generator_run()[1]
    homes: dict[str, list[str]] = {}
    for name, types in rest.items():
        for w in types:
            homes.setdefault(w, []).append(name)
    idx = index()
    return {w: h[0] for w, h in sorted(homes.items()) if len(h) == 1 and w in idx}


def benchmark_words(lo: int = MIN_COUNT, hi: int | None = None) -> tuple[str, ...]:
    """Sector nouns (the common restaurant excluded) seen between lo and hi times in training, sorted."""
    lab = planted_labels()
    return tuple(w for w in sorted(lab) if lab[w] != "common" and lo <= count_of(w) <= (hi or 10**9))


@lru_cache(maxsize=1)
def benchmark_split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(dev, test): within each sector, a seeded permutation of its benchmark nouns, the first half to dev.
    Dev chooses every hyperparameter; test reports. The two halves never share a noun."""
    rng = np.random.default_rng([SEED, 0])
    lab = planted_labels()
    dev, test = [], []
    for s in SECTORS:
        ws = [w for w in benchmark_words() if lab[w] == s]
        perm = rng.permutation(len(ws))
        dev += [ws[i] for i in perm[: len(ws) // 2]]
        test += [ws[i] for i in perm[len(ws) // 2 :]]
    return tuple(sorted(dev)), tuple(sorted(test))


@lru_cache(maxsize=1)
def shuffled_documents() -> tuple[tuple[NGram, ...], ...]:
    """The permutation null: every training token shuffled across the whole corpus (rng [SEED, 2]), with every
    sentence and document length kept. Each word's frequency is unchanged; every association is destroyed."""
    rng = np.random.default_rng([SEED, 2])
    flat = [w for s in sentences() for w in s]
    perm = rng.permutation(len(flat))
    out, pos = [], 0
    for d in documents():
        doc = []
        for s in d:
            doc.append(tuple(flat[i] for i in perm[pos : pos + len(s)]))
            pos += len(s)
        out.append(tuple(doc))
    return tuple(out)


# --------------------------------------------------------------------------------------------
# 1. Def 1–2: co-occurrence matrices and PMI.
# --------------------------------------------------------------------------------------------


def _flat_ids(units) -> tuple[np.ndarray, np.ndarray]:
    """Token ids of the training types in each unit (a sentence or a document) and the unit each came from."""
    idx = index()
    ids, uid = [], []
    for u, toks in enumerate(units):
        for w in toks:
            j = idx.get(w)
            if j is not None:
                ids.append(j)
                uid.append(u)
    return np.array(ids, dtype=np.int64), np.array(uid, dtype=np.int64)


def cooccurrence_of(units, context: str) -> sparse.csr_matrix:
    """Def 1. C[w, c] = the number of ordered token pairs (i, j), i ≠ j, inside one unit, with w at i and c at j,
    and |i − j| ≤ L for a window context "L1" … "L8"; any i ≠ j for "sentence" and "document". Units are sentences
    except for "document", whose units are documents (lists of sentences, concatenated). C is symmetric, and a word
    co-occurs with its own other tokens."""
    n = len(vocab())
    if context == "document":
        units = [tuple(w for s in d for w in s) for d in units]
    ids, uid = _flat_ids(units)
    if context in WINDOWS:
        rows, cols = [], []
        for delta in range(1, WINDOWS[context] + 1):
            same = uid[:-delta] == uid[delta:]
            a, b = ids[:-delta][same], ids[delta:][same]
            rows += [a, b]
            cols += [b, a]
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    S = sparse.csr_matrix((np.ones(len(ids)), (ids, uid)), shape=(n, int(uid.max()) + 1))
    C = (S @ S.T - sparse.diags(np.asarray(S.sum(axis=1)).ravel())).tocsr()  # n(n − 1) ordered self-pairs
    C.eliminate_zeros()
    return C


def _units(context: str, null: bool = False):
    docs = shuffled_documents() if null else documents()
    return docs if context == "document" else tuple(s for d in docs for s in d)


@lru_cache(maxsize=None)
def cooc(context: str, null: bool = False) -> sparse.csr_matrix:
    """The training text's co-occurrence matrix for one context (null=True: the permutation null's)."""
    return cooccurrence_of(_units(context, null), context)


def marginals(C: sparse.spmatrix, alpha: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """(N, n_w, P_α): the total, the row sums, and the smoothed context distribution P_α(c) = n_c^α / Σ n_c'^α."""
    N = float(C.sum())
    n_w = np.asarray(C.sum(axis=1)).ravel()
    pc = np.asarray(C.sum(axis=0)).ravel() ** alpha
    return N, n_w, pc / pc.sum()


def pmi(C: sparse.spmatrix, alpha: float = 1.0) -> sparse.csr_matrix:
    """Def 2. PMI_α(w, c) = log [P(w, c) / (P(w) P_α(c))] on the support of C, with P(w, c) = C/N and P(w) = n_w/N
    (Church and Hanks 1990; α < 1 is Levy et al.'s smoothing). Off the support the plug-in PMI is −∞, and the
    sparse matrix stores only the support (zeros on it are kept explicitly)."""
    C = sparse.coo_matrix(C)
    N, n_w, pc = marginals(C, alpha)
    vals = np.log(C.data / N) - np.log(n_w[C.row] / N) - np.log(pc[C.col])
    return sparse.csr_matrix((vals, (C.row, C.col)), shape=C.shape)


def ppmi(C: sparse.spmatrix, alpha: float = 1.0, k: float = 1.0) -> sparse.csr_matrix:
    """Def 2. max(PMI_α − log k, 0): k = 1 is PPMI, k > 1 the shifted SPPMI_k of Levy and Goldberg (2014), whose
    unclipped matrix PMI − log k is what skip-gram with k negative samples factorizes (Topic 7)."""
    M = sparse.coo_matrix(pmi(C, alpha))
    v = M.data - math.log(k)
    keep = v > 0
    return sparse.csr_matrix((v[keep], (M.row[keep], M.col[keep])), shape=M.shape)


@lru_cache(maxsize=None)
def weighted(context: str, weighting: str, null: bool = False) -> sparse.csr_matrix:
    """The matrix a weighting puts on one context: raw counts, PPMI, PPMI_α, or SPPMI_k (WEIGHTINGS)."""
    C = cooc(context, null)
    spec = WEIGHTINGS[weighting]
    return C if spec is None else ppmi(C, *spec)


def pmi_dense(P: np.ndarray) -> np.ndarray:
    """PMI of a dense joint distribution P[w, c] against its own marginals (−inf where P = 0)."""
    with np.errstate(divide="ignore"):
        return np.log(P) - np.log(P.sum(1))[:, None] - np.log(P.sum(0))[None, :]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------------------------
# 2. Thm 1 and Props 1–4: background, ceilings, singletons, the null, smoothing.
# --------------------------------------------------------------------------------------------


def background_family(P: np.ndarray, q: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Thm 1's family: the context distribution (1 − ε)P + εq of a word near the background P, and its PMI row
    log(1 − ε + εq/P) against P."""
    row = (1 - eps) * P + eps * q
    return row, np.log(row / P)


@lru_cache(maxsize=1)
def background_summary() -> dict[str, float]:
    """Thm 1 on the corpus, at L2 on the test nouns: how closely each raw count row aligns with the context
    marginal (the background every row shares), and the mean cosine of cross-sector pairs under counts and PPMI."""
    C = cooc("L2")
    _, _, P = marginals(C)
    test, lab = list(benchmark_split()[1]), planted_labels()
    ids = [index()[w] for w in test]
    Y = unit_rows(C, ids)
    align = Y @ (P / np.linalg.norm(P))
    out = {"count_marginal_cos_median": float(np.median(align)), "count_marginal_cos_min": float(align.min())}
    cross = np.array([[lab[a] != lab[b] for b in test] for a in test])
    for name in ("counts", "ppmi"):
        S = similarity(weighted("L2", name), test)
        out[f"{name}_cross_cos_mean"] = float(S[np.triu(cross, 1)].mean())
        out[f"{name}_same_cos_mean"] = float(S[np.triu(~cross, 1)].mean())
    return out


@lru_cache(maxsize=None)
def singleton_summary(context: str = "L2") -> dict[str, float]:
    """Prop 2 on the corpus: Topic 1's frequency spectrum of the pair counts (V₁ = pairs seen once), the share of
    the PPMI support they make up, and the largest PPMI on rows of each word-frequency band."""
    C = cooc(context)
    spec = zhv.frequency_spectrum(C.data)
    M = weighted(context, "ppmi")
    on = sparse.coo_matrix(C.multiply(M > 0))
    f = unigram_counts()
    bands = {"hapax": (1, 1), "2-4": (2, 4), "5-19": (5, 19), "20-99": (20, 99), "100+": (100, 10**9)}
    out = {"pairs": float(C.nnz), "singletons": float(spec[1]), "singleton_share": spec[1] / C.nnz,
           "ppmi_support": float(M.nnz), "ppmi_singleton_share": float(np.mean(on.data == 1)),
           "N": float(C.sum())}
    for name, (lo, hi) in bands.items():
        rows = np.flatnonzero((f >= lo) & (f <= hi))
        out[f"max_ppmi_{name}"] = float(M[rows].max())
    return out


def poisson_null(mu: float) -> dict[str, float]:
    """Prop 3 for one pair with expected count μ under independence: P(PPMI > 0) = P(X > μ), E[PPMI], and the
    smallest positive PPMI, for X ~ Poisson(μ) and PPMI = max(log(X/μ), 0) — by exact summation."""
    hi = int(mu + 40 * math.sqrt(mu) + 60)
    x = np.arange(0, hi + 1)
    pmf = stats.poisson.pmf(x, mu)
    pos = x > mu
    first = int(x[pos][0])
    return {"p_positive": float(pmf[pos].sum()), "e_ppmi": float(np.sum(pmf[pos] * np.log(x[pos] / mu))),
            "min_positive": math.log(first / mu)}


@lru_cache(maxsize=None)
def null_summary(context: str = "L2") -> dict[str, float]:
    """Prop 3 on the corpus: the real and the permutation-null PPMI matrices (every true PMI of the null is 0), and
    the share of the real PPMI support whose independence-expected count μ = n_w n_c / N is below 1."""
    out = {}
    for name, null in (("real", False), ("null", True)):
        for wt in ("ppmi", "sppmi5"):
            M = weighted(context, wt, null)
            out[f"{name}_{wt}_nnz"] = float(M.nnz)
            out[f"{name}_{wt}_fro2"] = float(M.multiply(M).sum())
    C = cooc(context)
    N, n_w, _ = marginals(C)
    M = sparse.coo_matrix(weighted(context, "ppmi"))
    mu = n_w[M.row] * n_w[M.col] / N  # C is symmetric: n_c = n_w
    out["real_ppmi_mu_below_1_share"] = float(np.mean(mu < 1))
    # Prop 3's expected false-positive count under Poisson independence with the training margins.
    mu_all = np.outer(n_w, n_w) / N
    out["poisson_expected_positives"] = float(stats.poisson.sf(np.floor(mu_all), mu_all).sum())
    return out


def smoothing_threshold(n_c: np.ndarray, alpha: float) -> float:
    """Prop 4. P_α(c)/P(c) = (N/Z_α) n_c^{α−1} crosses 1 at t* = (N/Z_α)^{1/(1−α)}, with N = Σ n_c, Z_α = Σ n_c^α."""
    n_c = n_c[n_c > 0]
    return float((n_c.sum() / np.sum(n_c**alpha)) ** (1 / (1 - alpha)))


@lru_cache(maxsize=None)
def smoothing_summary(context: str = "L2") -> dict[str, float]:
    C = cooc(context)
    n_c = np.asarray(C.sum(axis=0)).ravel()
    t = smoothing_threshold(n_c, ALPHA)
    below = n_c < t
    return {"t_star": t, "types_below": float(below.mean()), "tokens_below": float(n_c[below].sum() / n_c.sum()),
            **{f"nnz_{w}": float(weighted(context, w).nnz) for w in ("ppmi", "ppmi_a", "sppmi5", "sppmi15")}}


# --------------------------------------------------------------------------------------------
# 3. Def 3 and Thm 2: truncated SVD and what the eigenvalue weight p measures.
# --------------------------------------------------------------------------------------------


def truncated_svd(M, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The top-d singular triplets of a sparse matrix (ARPACK, via scipy.sparse.linalg.svds), sorted descending.
    The starting vector comes from rng, so the result is deterministic."""
    U, s, Vt = svds(M, k=d, v0=rng.standard_normal(min(M.shape)))
    o = np.argsort(-s)
    return U[:, o], s[o], Vt[o]


@lru_cache(maxsize=None)
def svd_of(context: str, weighting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return truncated_svd(weighted(context, weighting), D_MAX, np.random.default_rng([SEED, 1]))


def embedding(context: str, weighting: str, d: int, p: float) -> np.ndarray:
    """Def 3. W_p = U_d Σ_d^p, the word vectors of the rank-d SVD (p = 1 classical, ½ symmetric, 0 whitened)."""
    U, s, _ = svd_of(context, weighting)
    return U[:, :d] * s[:d] ** p


def context_embedding(context: str, weighting: str, d: int, p: float) -> np.ndarray:
    """Def 3. C_p = V_d Σ_d^{1−p}, the context vectors, so that W_p C_pᵀ = M_d for every p."""
    _, s, Vt = svd_of(context, weighting)
    return Vt[:d].T * s[:d] ** (1 - p)


def psd_power_truncation(G: np.ndarray, p: float, d: int) -> np.ndarray:
    """The best rank-d approximation of G^p for symmetric PSD G (G⁰ is the projector onto range G)."""
    lam, Q = np.linalg.eigh(G)
    lam = np.clip(lam, 0, None)
    tol = 1e-10 * lam.max()
    lp = np.where(lam > tol, lam, 0.0) ** p if p > 0 else (lam > tol).astype(float)
    o = np.argsort(-lam)[:d]  # order by λ, not λ^p: at p = 0 every nonzero λ^p ties at 1
    return (Q[:, o] * lp[o]) @ Q[:, o].T


def eig_truncation(M: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """The top-d eigenpairs of a symmetric matrix by |λ|: (λ, Q)."""
    lam, Q = np.linalg.eigh(M)
    o = np.argsort(-np.abs(lam))[:d]
    return lam[o], Q[:, o]


@lru_cache(maxsize=1)
def spectrum_summary() -> dict[str, float]:
    """Thm 2 on the corpus: the L2 PPMI matrix (α = 1, so symmetric) has many large negative eigenvalues, so the
    p = ½ word-word Gram |M|_d is far from M_d; the leading negative eigenvector splits the vocabulary nearly
    bipartitely."""
    M = weighted("L2", "ppmi")
    lam, Q = eigsh(M, k=100, which="LM", v0=np.random.default_rng([SEED, 7]).standard_normal(M.shape[0]))
    o = np.argsort(-np.abs(lam))
    lam, Q = lam[o], Q[:, o]
    neg = lam < 0
    gap = 2 * np.sqrt(np.sum(lam[neg] ** 2)) / np.sqrt(np.sum(lam**2))
    j = int(np.flatnonzero(neg)[0])
    side = np.sign(Q[:, j])
    Mc = sparse.coo_matrix(M)
    cut = float(Mc.data[side[Mc.row] != side[Mc.col]].sum() / Mc.data.sum())
    order = np.argsort(Q[:, j])
    return {"symmetric_err": float(abs(M - M.T).max()), "n_negative_top100": int(neg.sum()),
            "lambda1": float(lam[0]), "first_negative_rank": j + 1, "first_negative": float(lam[j]),
            "gram_gap_rel": float(gap), "cut_mass": cut, "eig": lam,
            "neg_side": tuple(vocab()[i] for i in order[:8]), "pos_side": tuple(vocab()[i] for i in order[::-1][:8])}


@lru_cache(maxsize=1)
def word_plus_context_summary() -> dict[str, float]:
    """Cor 1 on the corpus, at the configuration where Thm 2 is checked (L2, PPMI, d = 100, p = ½; not tuned):
    test-noun AUC and 1-NN of the word vectors W_½ and of (W_½ + C_½)/2, which drops the negative eigen-directions."""
    test, lab = benchmark_split()[1], planted_labels()
    W, Cv = embedding("L2", "ppmi", 100, 0.5), context_embedding("L2", "ppmi", 100, 0.5)
    return {"w_auc": pair_auc(W, test, lab), "w_nn": nn_accuracy(W, test, lab),
            "wc_auc": pair_auc((W + Cv) / 2, test, lab), "wc_nn": nn_accuracy((W + Cv) / 2, test, lab)}


# --------------------------------------------------------------------------------------------
# 4. Thm 3: two kinds of low rank.
# --------------------------------------------------------------------------------------------


def ratio_matrix(P: np.ndarray) -> np.ndarray:
    """R[w, c] = P(w, c) / (P(w) P(c)) = e^{PMI(w, c)}."""
    return P / np.outer(P.sum(1), P.sum(0))


def numerical_rank(A: np.ndarray, rtol: float = 1e-9) -> int:
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > rtol * s[0]))


def mixture_joint(rng: np.random.Generator, n_w: int, n_c: int, K: int) -> np.ndarray:
    """A K-class latent model: P = Σ_z π_z φ_z ψ_zᵀ, with W ⊥ C given Z (Dirichlet components, all entries > 0)."""
    pi = rng.dirichlet(np.ones(K))
    Phi = rng.dirichlet(np.full(n_w, 0.5), size=K).T
    Psi = rng.dirichlet(np.full(n_c, 0.5), size=K).T
    return (Phi * pi) @ Psi.T


def log_bilinear_joint(rng: np.random.Generator, n_w: int, n_c: int, d: int) -> np.ndarray:
    """P ∝ exp(XYᵀ + a𝟙ᵀ + 𝟙bᵀ), X, Y of width d: the form skip-gram and GloVe fit (Topics 7–8)."""
    X, Y = rng.normal(size=(n_w, d)), rng.normal(size=(n_c, d))
    L = X @ Y.T + rng.normal(size=(n_w, 1)) + rng.normal(size=(1, n_c))
    P = np.exp(L - L.max())
    return P / P.sum()


def residual_energy(s: np.ndarray, r: int) -> float:
    """The share of Σ σ² beyond the first r singular values: Eckart–Young's relative error of the best rank-r fit."""
    return float(np.sum(s[r:] ** 2) / np.sum(s**2))


@lru_cache(maxsize=None)
def lowrank_summary(null: bool = False) -> dict:
    """Thm 3 on the corpus. Document-level co-occurrence of the content nouns (every restaurant, the common one
    included) seen at least LOWRANK_MIN_COUNT times, with the submatrix's own margins: the singular values of
    R − 𝟙𝟙ᵀ and of PPMI. Within a document the generator draws every content noun from that document's sector
    restaurant or the common one, so a six-sector mixture predicts rank(R − 𝟙𝟙ᵀ) ≤ 5."""
    lab = planted_labels()
    nouns = [w for w in sorted(lab) if count_of(w) >= LOWRANK_MIN_COUNT]
    ids = [index()[w] for w in nouns]
    sub = cooc("document", null)[ids][:, ids].toarray()
    P = sub / sub.sum()
    R = ratio_matrix(P)
    s_r = np.linalg.svd(R - 1, compute_uv=False)
    with np.errstate(divide="ignore"):
        L = np.log(R)
    s_p = np.linalg.svd(np.where(L > 0, L, 0.0), compute_uv=False)
    r = K_SECTORS - 1
    return {"nouns": len(nouns), "classes": len({lab[w] for w in nouns}), "zero_share": float(np.mean(sub == 0)),
            "ratio_sv": s_r / s_r[0], "ppmi_sv": s_p / s_p[0], "ratio_sv_raw": s_r,
            "ratio_resid": residual_energy(s_r, r), "ppmi_resid": residual_energy(s_p, r)}


# --------------------------------------------------------------------------------------------
# 5. Def 4: the planted-sector benchmark, and selection on dev nouns.
# --------------------------------------------------------------------------------------------


def unit_rows(X, ids) -> np.ndarray:
    """The rows ids of X, each scaled to unit length (zero rows stay zero)."""
    Y = X[ids]
    Y = Y.toarray() if sparse.issparse(Y) else np.asarray(Y, dtype=float)
    nr = np.linalg.norm(Y, axis=1, keepdims=True)
    return np.divide(Y, nr, out=np.zeros_like(Y), where=nr > 0)


def similarity(X, words) -> np.ndarray:
    """Cosine similarities among the rows of X for the given words."""
    Y = unit_rows(X, [index()[w] for w in words])
    return Y @ Y.T


def nn_accuracy(X, words, labels) -> float:
    """Def 4. The share of words whose nearest other word (by cosine, among `words`) has the same label."""
    S = similarity(X, words)
    np.fill_diagonal(S, -np.inf)
    lab = np.array([labels[w] for w in words])
    return float(np.mean(lab[np.argmax(S, axis=1)] == lab))


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """The Mann–Whitney probability that a random positive outscores a random negative (ties count ½)."""
    r = stats.rankdata(scores)
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_auc(X, words, labels) -> float:
    """Def 4. P(cos of a random same-sector pair > cos of a random cross-sector pair), over all pairs of `words`."""
    S = similarity(X, words)
    lab = np.array([labels[w] for w in words])
    iu = np.triu_indices(len(words), 1)
    return auc(S[iu], (lab[:, None] == lab[None, :])[iu])


def jackknife_se(stat, words) -> float:
    """Leave-one-word-out jackknife standard error of a statistic of a word list (pairs share words, so pairs are
    not independent; words are the sampling unit)."""
    words = list(words)
    n = len(words)
    vals = np.array([stat(words[:i] + words[i + 1 :]) for i in range(n)])
    return float(np.sqrt((n - 1) / n * np.sum((vals - vals.mean()) ** 2)))


SPARSE_CONFIGS: tuple[Config, ...] = tuple((c, w, None, None) for c in CONTEXTS for w in WEIGHTINGS)
SVD_CONFIGS: tuple[Config, ...] = tuple(
    (c, w, d, p) for c in CONTEXTS for w in SVD_BASES for d in SVD_DIMS for p in SVD_POWERS)


def vectors(cfg: Config):
    c, w, d, p = cfg
    return weighted(c, w) if d is None else embedding(c, w, d, p)


@lru_cache(maxsize=None)
def scores(cfg: Config) -> dict[str, float]:
    dev, test = benchmark_split()
    lab, X = planted_labels(), vectors(cfg)
    return {"dev_auc": pair_auc(X, dev, lab), "dev_nn": nn_accuracy(X, dev, lab),
            "test_auc": pair_auc(X, test, lab), "test_nn": nn_accuracy(X, test, lab)}


@lru_cache(maxsize=1)
def selected() -> dict[str, Config]:
    """The pre-declared rule: within each family (sparse matrices, SVD embeddings) and over LOCAL_CONTEXTS, the
    configuration with the highest dev AUC, dev 1-NN breaking ties. Test nouns play no part."""
    def best(cfgs):
        return max((c for c in cfgs if c[0] in LOCAL_CONTEXTS), key=lambda c: (scores(c)["dev_auc"], scores(c)["dev_nn"]))
    return {"sparse": best(SPARSE_CONFIGS), "svd": best(SVD_CONFIGS)}


@lru_cache(maxsize=1)
def comparison() -> dict[str, float]:
    """§5's headline: the two dev-selected configurations on the test nouns, with the jackknife standard error of
    the AUC difference and the paired standard error of the 1-NN difference; the rare band (2–4 occurrences), which
    no selection saw; and the document-context oracle."""
    lab = planted_labels()
    test = list(benchmark_split()[1])
    Xa, Xb = vectors(selected()["sparse"]), vectors(selected()["svd"])

    def diff(ws):
        return pair_auc(Xa, ws, lab) - pair_auc(Xb, ws, lab)

    def nn_hits(X, ws):
        S = similarity(X, ws)
        np.fill_diagonal(S, -np.inf)
        la = np.array([lab[w] for w in ws])
        return (la[np.argmax(S, axis=1)] == la).astype(float)

    dh = nn_hits(Xa, test) - nn_hits(Xb, test)
    out = {"auc_sparse": pair_auc(Xa, test, lab), "auc_svd": pair_auc(Xb, test, lab), "auc_diff": diff(test),
           "auc_diff_se": jackknife_se(diff, test), "nn_sparse": nn_accuracy(Xa, test, lab),
           "nn_svd": nn_accuracy(Xb, test, lab), "nn_diff": float(dh.mean()),
           "nn_diff_se": float(dh.std(ddof=1) / math.sqrt(len(dh)))}
    rare = list(benchmark_words(*RARE_BAND))
    ctx = selected()["sparse"][0]
    for name, X in (("sparse", Xa), ("svd", Xb), ("counts", weighted(ctx, "counts"))):
        out[f"rare_auc_{name}"] = pair_auc(X, rare, lab)
        out[f"rare_nn_{name}"] = nn_accuracy(X, rare, lab)
    labs = Counter(lab[w] for w in rare)
    out["rare_n"] = len(rare)
    out["rare_chance_nn"] = sum((v / len(rare)) ** 2 for v in labs.values())
    tl = Counter(lab[w] for w in test)
    out["test_chance_nn"] = sum((v / len(test)) ** 2 for v in tl.values())
    for wt in ("counts", "ppmi"):
        out.update({f"oracle_{wt}_{k}": v for k, v in scores(("document", wt, None, None)).items()})
    # How narrow the SVD's win is: the local SVD configurations that beat the selected sparse matrix, and where
    # the selected (context, base, p)'s AUC peaks in d on dev and on test.
    local = [c for c in SVD_CONFIGS if c[0] in LOCAL_CONTEXTS]
    sp = scores(selected()["sparse"])
    for split in ("dev", "test"):
        out[f"svd_beating_sparse_{split}"] = sum(scores(c)[f"{split}_auc"] > sp[f"{split}_auc"] for c in local)
        ctx, base, _, p = selected()["svd"]
        out[f"svd_peak_d_{split}"] = max(SVD_DIMS, key=lambda d: scores((ctx, base, d, p))[f"{split}_auc"])
    out["svd_configs_local"] = len(local)
    return out


# --------------------------------------------------------------------------------------------
# 6. Finance case study: antonyms, polarity from context, and the polysemes.
# --------------------------------------------------------------------------------------------


def polarity_words() -> dict[str, tuple[str, ...]]:
    """The in-repo stand-in lexicon's positive and negative words that occur in training (never the real
    Loughran–McDonald dictionary)."""
    idx = index()
    return {cat: tuple(sorted(w for w in fc.STANDIN_LEXICON[cat] if w in idx)) for cat in ("positive", "negative")}


def polarity_adjectives() -> dict[str, tuple[str, ...]]:
    return {cat: tuple(w for w in ws if fc.LEXICON_POS[w] == "ADJ") for cat, ws in polarity_words().items()}


def semantic_orientation(C: sparse.spmatrix, word: str, pos, neg) -> float:
    """Turney's (2002) semantic orientation, SO(w) = PMI(w, POS) − PMI(w, NEG), with each seed set merged into one
    paradigm context (Turney and Littman 2003) and ½ added to both co-occurrence counts (no zero-count
    log): SO = log[(C(w, POS) + ½) / (C(w, NEG) + ½)] − log[n(POS) / n(NEG)]. The word is removed from its own
    seed set first, so the score never sees its own label. A word that never co-occurs with either seed set has no
    evidence, and SO = 0 (otherwise the marginal term alone would assign it the polarity of the rarer seed set)."""
    idx = index()
    pos = [idx[p] for p in pos if p != word]
    neg = [idx[q] for q in neg if q != word]
    row = C[idx[word]]
    n_c = np.asarray(C.sum(axis=0)).ravel()
    cp, cn = row[:, pos].sum(), row[:, neg].sum()
    if cp + cn == 0:
        return 0.0
    return float(math.log((cp + 0.5) / (cn + 0.5)) - math.log(n_c[pos].sum() / n_c[neg].sum()))


@lru_cache(maxsize=None)
def so_scores(context: str) -> dict[str, tuple[str, float]]:
    """Leave-one-out SO of every stand-in polarity word from one context's counts: word → (label, SO)."""
    C, pw = cooc(context), polarity_words()
    return {w: (cat, semantic_orientation(C, w, pw["positive"], pw["negative"])) for cat in pw for w in pw[cat]}


@lru_cache(maxsize=None)
def so_accuracy(context: str) -> float:
    """Leave-one-out polarity accuracy of SO's sign: 1 for the right sign, 0 for the wrong one, ½ for SO = 0
    (no evidence: a coin flip)."""
    return float(np.mean([0.5 if so == 0 else float((so > 0) == (cat == "positive"))
                          for cat, so in so_scores(context).values()]))


def so_coverage(context: str) -> float:
    """The share of polarity words with any seed co-occurrence in this context (SO ≠ 0)."""
    return float(np.mean([so != 0 for _, so in so_scores(context).values()]))


def neighbors(X, word: str, top: int = 6, min_count: int = MIN_COUNT) -> list[tuple[str, float]]:
    """The top cosine neighbors of a word among training words seen at least min_count times."""
    cand = np.flatnonzero(unigram_counts() >= min_count)
    cand = cand[cand != index()[word]]
    q = unit_rows(X, [index()[word]])[0]
    sims = unit_rows(X, cand) @ q
    o = np.argsort(-sims, kind="stable")[:top]
    return [(vocab()[cand[i]], float(sims[i])) for i in o]


@lru_cache(maxsize=1)
def finance_summary() -> dict:
    """§6: (1) at L2 the nearest neighbors of "strong" include its antonyms, and polarity adjectives are as close
    across polarity as within it; (2) SO's polarity accuracy by context; (3) the polysemes' SO from document
    contexts; (4) EDGAR's coverage by training vectors."""
    pa = polarity_adjectives()
    out: dict = {"strong_neighbors_L2": neighbors(weighted("L2", "ppmi"), "strong", top=7)}
    for ctx in CONTEXTS:
        S = similarity(weighted(ctx, "ppmi"), pa["positive"] + pa["negative"])
        a = len(pa["positive"])
        within = np.r_[S[:a, :a][np.triu_indices(a, 1)], S[a:, a:][np.triu_indices(len(S) - a, 1)]]
        out[f"adj_within_{ctx}"] = float(within.mean())
        out[f"adj_cross_{ctx}"] = float(S[:a, a:].mean())
        out[f"so_acc_{ctx}"] = so_accuracy(ctx)
        out[f"so_coverage_{ctx}"] = so_coverage(ctx)
        for cat in ("positive", "negative"):
            out[f"so_mean_{cat}_{ctx}"] = float(np.mean([so for c, so in so_scores(ctx).values() if c == cat]))
    pw, C = polarity_words(), cooc("document")
    so = {w: semantic_orientation(C, w, pw["positive"], pw["negative"]) for w in sorted(fc.FINANCE_POLYSEMES)
          if w in index()}
    out["polyseme_so"] = so
    out["polyseme_so_mean"] = float(np.mean(list(so.values())))
    out["polyseme_abs_so_median"] = float(np.median(np.abs(list(so.values()))))
    lex_so = [abs(semantic_orientation(C, w, pw["positive"], pw["negative"])) for cat in pw for w in pw[cat]]
    out["lexicon_abs_so_median"] = float(np.median(lex_so))
    edgar = [w for s in ng.corpus()["edgar"] for w in s]
    out["edgar_tokens"] = len(edgar)
    out["edgar_covered"] = float(np.mean([w in index() for w in edgar]))
    out["edgar_covered_min_count"] = float(np.mean([w in index() and count_of(w) >= MIN_COUNT for w in edgar]))
    return out


# --------------------------------------------------------------------------------------------
# The lab's data.
# --------------------------------------------------------------------------------------------

# One frequent seeded noun per sector (production: its energy-only template "Crude production averaged …"), the
# polarity pair, a polyseme, the background word "and", and a litigious word.
LAB_WORDS = ("strong", "weak", "liability", "orders", "software", "loans", "production", "stores", "trials",
             "and", "litigation")
LAB_WEIGHTINGS = ("counts", "ppmi", "sppmi5", "svd")
LAB_TOP = 5
SPECTRUM_K = 100  # singular values the lab plots, real against shuffled


def _r(x: float, nd: int = 4):
    return None if not math.isfinite(x) else round(float(x), nd)


def _sig(x: float, n: int = 4):
    """Round to n significant figures (for quantities spanning decades)."""
    return None if not math.isfinite(x) else float(f"{x:.{n}g}")


def word_tag(w: str) -> str:
    """The chip a word gets in the lab: its planted sector, a stand-in lexicon category, "polyseme", "common"
    (the shared restaurant), or "other"."""
    lab = planted_labels()
    if w in lab:
        return lab[w]
    for cat in ("positive", "negative", "uncertainty", "litigious"):
        if w in fc.STANDIN_LEXICON[cat]:
            return cat
    return "polyseme" if w in fc.FINANCE_POLYSEMES else "other"


def lab_config(context: str, weighting: str) -> Config:
    """A lab cell: a sparse weighting at a context, or "svd" — the dev-selected SVD's (base, d, p) at that context."""
    if weighting != "svd":
        return (context, weighting, None, None)
    _, base, d, p = selected()["svd"]
    return (context, base, d, p)


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    """Panel A: for every (context, lab weighting) cell, each lab word's top LAB_TOP neighbors [word, cosine] and
    the cell's test-noun AUC and 1-NN; `tags` gives each word shown its tag (sector, lexicon category, …), and a word
    absent from it is "other".
    Panel B: the dev-selected SVD's (d, p) sweep on test nouns; the L2 spectra, real against shuffled; the L2 PPMI's
    signed eigenvalues; and Thm 3's two normalized spectra."""
    sel = selected()
    cells, shown = {}, set(LAB_WORDS)
    for ctx in CONTEXTS:
        for wt in LAB_WEIGHTINGS:
            cfg = lab_config(ctx, wt)
            X, sc = vectors(cfg), scores(cfg)
            nb = {w: [[n, _r(s, 2)] for n, s in neighbors(X, w, LAB_TOP)] for w in LAB_WORDS}
            shown |= {n for row in nb.values() for n, _ in row}
            cells[f"{ctx}|{wt}"] = {"test_auc": _r(sc["test_auc"]), "test_nn": _r(sc["test_nn"]), "neighbors": nb}
    sweep_ctx, sweep_base = sel["svd"][0], sel["svd"][1]
    sweep = {str(p): {"auc": [_r(scores((sweep_ctx, sweep_base, d, p))["test_auc"]) for d in SVD_DIMS],
                      "nn": [_r(scores((sweep_ctx, sweep_base, d, p))["test_nn"]) for d in SVD_DIMS]}
             for p in SVD_POWERS}
    spec = {}
    for wt in ("ppmi", "sppmi5"):
        null = truncated_svd(weighted("L2", wt, True), SPECTRUM_K, np.random.default_rng([SEED, 1]))[1]
        spec[wt] = {"real": [_sig(x) for x in svd_of("L2", wt)[1][:SPECTRUM_K]], "null": [_sig(x) for x in null]}
    ss, lr, fin = spectrum_summary(), lowrank_summary(), finance_summary()
    return {
        "contexts": list(CONTEXTS), "local_contexts": list(LOCAL_CONTEXTS), "weightings": list(LAB_WEIGHTINGS),
        "words": list(LAB_WORDS), "tags": {w: t for w in sorted(shown) if (t := word_tag(w)) != "other"}, "cells": cells,
        "selected": {k: list(v) for k, v in sel.items()}, "chance_nn": _r(comparison()["test_chance_nn"]),
        "so_accuracy": {ctx: _r(fin[f"so_acc_{ctx}"]) for ctx in CONTEXTS},
        "so_coverage": {ctx: _r(fin[f"so_coverage_{ctx}"]) for ctx in CONTEXTS},
        "sweep": {"context": sweep_ctx, "base": sweep_base, "dims": list(SVD_DIMS), "by_p": sweep,
                  "sparse_auc": _r(scores(sel["sparse"])["test_auc"]), "sparse_nn": _r(scores(sel["sparse"])["test_nn"])},
        "spectrum": spec, "eig_l2": [_sig(x) for x in ss["eig"]],
        "lowrank": {"ratio": [_r(x) for x in lr["ratio_sv"][:15]], "ppmi": [_r(x) for x in lr["ppmi_sv"][:15]],
                    "rank": K_SECTORS - 1, "ratio_resid": _r(lr["ratio_resid"]), "ppmi_resid": _r(lr["ppmi_resid"]),
                    "noise": _r(float(lowrank_summary(True)["ratio_sv_raw"][0] / lr["ratio_sv_raw"][0]))},
    }


# --------------------------------------------------------------------------------------------
# Summary — the numbers the page quotes.
# --------------------------------------------------------------------------------------------


def summary() -> dict[str, float]:
    dev, test = benchmark_split()
    out: dict[str, float] = {"V": len(vocab()), "tokens": float(unigram_counts().sum()),
                             "benchmark": len(benchmark_words()), "dev": len(dev), "test": len(test),
                             "sector_nouns": len(benchmark_words(1))}
    for ctx in CONTEXTS:
        out[f"N_{ctx}"] = float(cooc(ctx).sum())
        out[f"pairs_{ctx}"] = float(cooc(ctx).nnz)
    out.update({f"bg_{k}": v for k, v in background_summary().items()})
    out.update({f"single_{k}": v for k, v in singleton_summary().items()})
    out.update({f"perm_{k}": v for k, v in null_summary().items()})
    out.update({f"wc_{k}": v for k, v in word_plus_context_summary().items()})
    out.update({f"smooth_{k}": v for k, v in smoothing_summary().items()})
    out.update({f"spec_{k}": v for k, v in spectrum_summary().items() if isinstance(v, (int, float))})
    lr = lowrank_summary()
    out.update({"lowrank_nouns": lr["nouns"], "lowrank_classes": lr["classes"], "lowrank_zero_share": lr["zero_share"],
                "lowrank_ratio_resid": lr["ratio_resid"], "lowrank_ppmi_resid": lr["ppmi_resid"],
                "lowrank_ratio_sv6": float(lr["ratio_sv"][5]),
                "lowrank_null_sv1_over_real_sv1": float(lowrank_summary(True)["ratio_sv_raw"][0] / lr["ratio_sv_raw"][0])})
    for fam, cfg in selected().items():
        out[f"selected_{fam}"] = "|".join(str(x) for x in cfg)  # type: ignore[assignment]
        out.update({f"{fam}_{k}": v for k, v in scores(cfg).items()})
    out.update({f"cmp_{k}": v for k, v in comparison().items()})
    for wt in ("ppmi", "ppmi_a"):
        out[f"smoothing_sentence_{wt}_dev_auc"] = scores(("sentence", wt, None, None))["dev_auc"]
    for ctx in CONTEXTS:
        for wt in ("counts", "ppmi"):
            out[f"grid_{ctx}_{wt}_test_auc"] = scores((ctx, wt, None, None))["test_auc"]
            out[f"grid_{ctx}_{wt}_test_nn"] = scores((ctx, wt, None, None))["test_nn"]
    fin = finance_summary()
    out.update({f"fin_{k}": v for k, v in fin.items() if isinstance(v, (int, float))})
    return out


# --------------------------------------------------------------------------------------------
# Verification harness — every claim the page makes.
# --------------------------------------------------------------------------------------------


def test_planted_labels():
    """The rebuilt generator reproduces generate_corpus(300, 0) and Topic 3's training sentences exactly; every
    benchmark noun has one restaurant; the dev/test split is stratified and disjoint."""
    docs = _generator_run()[0]
    assert [d.text for d in docs] == [d.text for d in fc.generate_corpus(300, 0)]
    assert sentences() == ng.corpus()["train"]
    assert len(vocab()) == len(ng.vocabulary()) - 2  # Topic 3's V adds </s> and <unk>
    lab = planted_labels()
    assert "reserves" not in lab and all(v in (*SECTORS, "common") for v in lab.values())
    dev, test = benchmark_split()
    assert not set(dev) & set(test) and set(dev) | set(test) == set(benchmark_words())
    for s in SECTORS:
        nd, nt = sum(lab[w] == s for w in dev), sum(lab[w] == s for w in test)
        assert nt - nd in (0, 1)
    assert not set(benchmark_words(*RARE_BAND)) & set(benchmark_words())


def test_cooccurrence_definition():
    """Def 1: the vectorized window counts equal a direct double loop on random sentences; every C is symmetric;
    whole-unit counts are SSᵀ − diag(S𝟙)."""
    rng = np.random.default_rng([SEED, 3])
    words = vocab()[:40]
    sents = [tuple(words[i] for i in rng.integers(0, 40, rng.integers(1, 12))) for _ in range(60)]
    for ctx in ("L1", "L2", "L4", "sentence"):
        C = cooccurrence_of(sents, ctx).toarray()
        D = np.zeros_like(C)
        L = WINDOWS.get(ctx, 10**9)
        for s in sents:
            for i, a in enumerate(s):
                for j, b in enumerate(s):
                    if i != j and abs(i - j) <= L:
                        D[index()[a], index()[b]] += 1
        assert np.array_equal(C, D)
    for ctx in CONTEXTS:
        C = cooc(ctx)
        assert abs(C - C.T).max() == 0


def test_pmi_log_ratio_coordinates():
    """Prop 1 on random joint distributions: the PMI row determines P(·|w) = P(c) e^{PMI}; its P(·|w)-average is
    KL(P(·|w) ‖ P_C) ≥ 0, zero exactly under independence; and Σ_w P(w) KL = I(W; C) (scipy.stats.entropy)."""
    rng = np.random.default_rng([SEED, 4])
    for _ in range(50):
        P = rng.dirichlet(np.ones(12 * 9)).reshape(12, 9)
        X = pmi_dense(P)
        pw, pc = P.sum(1), P.sum(0)
        cond = P / pw[:, None]
        assert np.allclose(pc * np.exp(X), cond)
        kl = np.sum(cond * X, axis=1)
        assert np.allclose(kl, [stats.entropy(cond[i], pc) for i in range(12)]) and np.all(kl > 0)
        mi = stats.entropy(P.ravel(), np.outer(pw, pc).ravel())
        assert abs(pw @ kl - mi) < 1e-12
        Q = np.outer(pw, pc)
        assert np.allclose(pmi_dense(Q), 0, atol=1e-12)


def test_pmi_removes_background():
    """Thm 1. Near the background (ε → 0) every count row is parallel to P (1 − cos = O(ε²)), while
    cos(PMI rows) → cos(q/P − 𝟙, q′/P − 𝟙) at rate O(ε); f = q/P − 𝟙 is P-centered. On the corpus (L2, test
    nouns) count rows align with the context marginal and cross-sector count cosines are high; PPMI's are low."""
    rng = np.random.default_rng([SEED, 5])
    for _ in range(30):
        P = rng.dirichlet(np.ones(50))
        q1, q2 = rng.dirichlet(np.ones(50)), rng.dirichlet(np.ones(50))
        f1, f2 = q1 / P - 1, q2 / P - 1
        assert abs(P @ f1) < 1e-12
        target = cosine(f1, f2)
        F = max(np.abs(f1).max(), np.abs(f2).max())  # the asymptotic regime is ε‖f‖∞ ≪ 1
        errs, gaps = [], []
        for j in (2, 3, 4, 5):
            a1, x1 = background_family(P, q1, 10.0**-j / F)
            a2, x2 = background_family(P, q2, 10.0**-j / F)
            gaps.append(1 - cosine(a1, a2))
            errs.append(abs(cosine(x1, x2) - target))
        assert all(errs[i + 1] < 0.2 * errs[i] for i in range(3)) and errs[-1] < 1e-4  # O(ε)
        assert all(gaps[i + 1] < 0.02 * gaps[i] for i in range(3))  # O(ε²)
    b = background_summary()
    assert b["counts_cross_cos_mean"] > 0.5 and b["counts_same_cos_mean"] < 1.1 * b["counts_cross_cos_mean"]
    assert b["ppmi_cross_cos_mean"] < 0.1 and b["ppmi_same_cos_mean"] > 1.5 * b["ppmi_cross_cos_mean"]


def test_pmi_ceiling_and_singletons():
    """Prop 2. PMI(w, c) ≤ min(−log P(w), −log P(c)), with equality when c occurs only with w; on every pair seen
    once, the plug-in PMI is exactly log N − log n_w − log n_c (and −log n_w − log P_α(c) under smoothing)."""
    rng = np.random.default_rng([SEED, 6])
    for _ in range(50):
        P = rng.dirichlet(np.ones(15 * 10)).reshape(15, 10)
        P[3] = 0
        P[3, 7] = rng.random() * 0.1
        P[:, 7] = 0
        P[3, 7] = 0.05
        P /= P.sum()
        X = pmi_dense(P)
        pw, pc = P.sum(1), P.sum(0)
        fin = np.isfinite(X)
        ceil = np.minimum(-np.log(pw)[:, None], -np.log(pc)[None, :])
        assert np.all(X[fin] <= ceil[fin] + 1e-12)
        assert abs(X[3, 7] + math.log(pw[3])) < 1e-12  # c₇ occurs only with w₃ (and vice versa)
    for alpha in (1.0, ALPHA):
        C = sparse.coo_matrix(cooc("L2"))
        N, n_w, pc = marginals(C, alpha)
        M = sparse.coo_matrix(pmi(C, alpha))
        one = C.data == 1
        assert np.array_equal(M.row, C.row) and np.array_equal(M.col, C.col)
        pred = -np.log(n_w[C.row[one]]) - np.log(pc[C.col[one]])
        assert np.max(np.abs(M.data[one] - pred)) < 1e-12
        if alpha == 1.0:
            assert np.max(np.abs(M.data[one] - (math.log(N) - np.log(n_w[C.row[one]]) - np.log(n_w[C.col[one]])))) < 1e-12
            assert np.all(M.data <= np.minimum(-np.log(n_w[M.row] / N), -np.log(n_w[M.col] / N)) + 1e-12)
    s = singleton_summary()
    assert s["singleton_share"] > 0.6 and s["ppmi_singleton_share"] > 0.6
    assert s["max_ppmi_hapax"] > s["max_ppmi_100+"]


def test_ppmi_under_independence():
    """Prop 3. For X ~ Poisson(μ): P(PPMI > 0) = P(X > μ); for μ < 1 that is 1 − e^{−μ}, every positive value is at
    least log(1/μ), and E[PPMI] ≥ (1 − e^{−μ}) log(1/μ); P(PPMI > 0) → ½ and E[PPMI]·√(2πμ) → 1 as μ grows. A Monte
    Carlo independence matrix with Zipf margins (Topic 1) puts most PPMI mass on pairs with μ < 1. On the corpus the
    shuffled text has more PPMI energy than the real one, and Poisson independence predicts its support size."""
    for mu in (1e-3, 0.01, 0.1, 0.5, 0.9):
        r = poisson_null(mu)
        assert abs(r["p_positive"] - (1 - math.exp(-mu))) < 1e-12
        assert r["min_positive"] >= math.log(1 / mu) - 1e-12
        assert r["e_ppmi"] >= (1 - math.exp(-mu)) * math.log(1 / mu)
    big = [poisson_null(mu) for mu in (100.0, 1000.0, 10000.0)]
    assert all(abs(r["p_positive"] - 0.5) < 0.03 for r in big)
    assert abs(big[-1]["e_ppmi"] * math.sqrt(2 * math.pi * 10000.0) - 1) < 0.02
    rng = np.random.default_rng([SEED, 8])
    p = zhv.zipf_pmf(1.1, 400)
    mu = 2e5 * np.outer(p, p)
    X = rng.poisson(mu)
    with np.errstate(divide="ignore"):
        pp = np.where(X > 0, np.log(np.maximum(X, 1) / mu), 0.0).clip(min=0)
    assert np.sum(pp[mu < 1] ** 2) / np.sum(pp**2) > 0.5
    n = null_summary()
    assert n["null_ppmi_fro2"] > n["real_ppmi_fro2"] and n["null_ppmi_nnz"] > n["real_ppmi_nnz"]
    assert n["real_sppmi5_fro2"] > n["null_sppmi5_fro2"]  # the shift by log 5 reverses the comparison
    assert abs(n["poisson_expected_positives"] / n["null_ppmi_nnz"] - 1) < 0.05
    assert n["real_ppmi_mu_below_1_share"] > 0.8


def test_context_smoothing_threshold():
    """Prop 4. P_α/P is strictly decreasing in n_c and equals 1 at t*; so on the support PMI_α < PMI exactly for
    contexts rarer than t*. SPPMI supports are nested: supp SPPMI₁₅ ⊆ supp SPPMI₅ ⊆ supp PPMI_α."""
    rng = np.random.default_rng([SEED, 9])
    for _ in range(30):
        n_c = rng.integers(1, 5000, 200).astype(float)
        a = rng.uniform(0.3, 0.95)
        ratio = (n_c**a / np.sum(n_c**a)) / (n_c / n_c.sum())
        o = np.argsort(n_c)
        assert np.all(np.diff(ratio[o])[np.diff(n_c[o]) > 0] < 0)
        t = smoothing_threshold(n_c, a)
        assert np.all((ratio > 1) == (n_c < t))
    C = cooc("L2")
    n_c = np.asarray(C.sum(axis=0)).ravel()
    t = smoothing_threshold(n_c, ALPHA)
    A, B = sparse.coo_matrix(pmi(C, ALPHA)), sparse.coo_matrix(pmi(C))
    lower = A.data < B.data
    assert np.array_equal(lower, n_c[A.col] < t)
    s5, s15, s1 = (weighted("L2", w) != 0 for w in ("sppmi5", "sppmi15", "ppmi_a"))
    assert (s15 > s5).nnz == 0 and (s5 > s1).nnz == 0


def test_svds_matches_dense_svd():
    """Cross-check: ARPACK's top-50 singular triplets of a 700-word PPMI block match numpy.linalg.svd."""
    f = unigram_counts()
    top = np.argsort(-f, kind="stable")[:700]
    M = weighted("L2", "ppmi_a")[top][:, top]
    U, s, Vt = truncated_svd(M, 50, np.random.default_rng([SEED, 10]))
    U0, s0, Vt0 = np.linalg.svd(M.toarray())
    assert np.max(np.abs(s - s0[:50]) / s0[:50]) < 1e-8
    Md, Md0 = (U * s) @ Vt, (U0[:, :50] * s0[:50]) @ Vt0[:50]
    assert np.linalg.norm(Md - Md0) / np.linalg.norm(Md0) < 1e-6


def test_eigenvalue_weighting():
    """Thm 2. W_p C_pᵀ = M_d for every p (and M_d beats random rank-d competitors); W_p W_pᵀ = [(MMᵀ)^p]_d; for
    symmetric M, W_½W_½ᵀ = |M|_d, which equals M_d iff the top-d eigenvalues are ≥ 0, with
    ‖|M|_d − M_d‖_F = 2(Σ_{λ<0} λ²)^½. On the corpus the L2 PPMI is symmetric and strongly indefinite."""
    rng = np.random.default_rng([SEED, 11])
    for shape, d in (((30, 20), 5), ((25, 25), 8)):
        for _ in range(10):
            M = rng.normal(size=shape)
            U, s, Vt = np.linalg.svd(M)
            Md = (U[:, :d] * s[:d]) @ Vt[:d]
            for _ in range(20):
                B = rng.normal(size=(shape[0], d)) @ rng.normal(size=(d, shape[1]))
                assert np.linalg.norm(M - Md) <= np.linalg.norm(M - B)
            for p in (0.0, 0.25, 0.5, 1.0):
                W, Cp = U[:, :d] * s[:d] ** p, Vt[:d].T * s[:d] ** (1 - p)
                assert np.allclose(W @ Cp.T, Md, atol=1e-10)
                assert np.allclose(W @ W.T, psd_power_truncation(M @ M.T, p, d), atol=1e-8)
    for psd in (True, False):
        for _ in range(10):
            A = rng.normal(size=(20, 20))
            M = A @ A.T if psd else (A + A.T) / 2
            U, s, Vt = np.linalg.svd(M)
            W = U[:, :6] * np.sqrt(s[:6])
            lam, Q = eig_truncation(M, 6)
            Md, Abs = (Q * lam) @ Q.T, (Q * np.abs(lam)) @ Q.T
            assert np.allclose(W @ W.T, Abs, atol=1e-9)
            assert np.allclose(Abs, Md, atol=1e-9) == bool(np.all(lam >= 0))
            assert abs(np.linalg.norm(Abs - Md) - 2 * np.sqrt(np.sum(lam[lam < 0] ** 2))) < 1e-9
    s = spectrum_summary()
    assert s["symmetric_err"] < 1e-12 and s["n_negative_top100"] >= 30
    assert s["first_negative_rank"] <= 3 and s["gram_gap_rel"] > 1
    assert s["cut_mass"] > 0.5  # most PPMI mass joins the two sides of λ₂'s eigenvector: near-bipartite


def test_word_plus_context():
    """Cor 1. For symmetric M, W_½ + C_½ = 2Q_dΛ₊^½ (eigen-directions with λ < 0 cancel), so
    ((W + C)/2)((W + C)/2)ᵀ = [M_d]₊, the positive part of M_d — exactly, whatever signs the SVD picks. Checked on
    random symmetric matrices and on the corpus's L2 PPMI with ARPACK's vectors."""
    rng = np.random.default_rng([SEED, 12])
    for _ in range(20):
        A = rng.normal(size=(18, 18))
        M = (A + A.T) / 2
        U, s, Vt = np.linalg.svd(M)
        signs = rng.choice([-1.0, 1.0], 18)  # any joint sign flip of (u_i, v_i) is another SVD
        U, Vt = U * signs, Vt * signs[:, None]
        d = 7
        H = (U[:, :d] * np.sqrt(s[:d]) + Vt[:d].T * np.sqrt(s[:d])) / 2
        lam, Q = eig_truncation(M, d)
        assert np.allclose(H @ H.T, (Q * np.clip(lam, 0, None)) @ Q.T, atol=1e-9)
    W, Cv = embedding("L2", "ppmi", 100, 0.5), context_embedding("L2", "ppmi", 100, 0.5)
    scale = math.sqrt(svd_of("L2", "ppmi")[1][0])
    cols = np.arange(100)
    cosv = np.sum(unit_rows(W.T, cols) * unit_rows(Cv.T, cols), axis=1)  # cos(u_i, v_i) = sign λ_i
    assert np.all(np.abs(np.abs(cosv) - 1) < 1e-6)
    neg = cosv < 0
    assert np.max(np.abs(W[:, neg] + Cv[:, neg])) < 1e-6 * scale  # these directions cancel in W + C
    assert np.max(np.abs(W[:, ~neg] - Cv[:, ~neg])) < 1e-6 * scale  # these double
    assert int(neg.sum()) == spectrum_summary()["n_negative_top100"]
    wc = word_plus_context_summary()
    assert wc["wc_auc"] > wc["w_auc"] and wc["wc_nn"] > wc["w_nn"]


def test_mixture_ratio_low_rank():
    """Thm 3. (i) A K-class mixture has rank R = K and rank(R − 𝟙𝟙ᵀ) = K − 1. (ii) Its PMI = log R is the
    Hadamard series Σ (−1)^{n+1}(R − 𝟙𝟙ᵀ)^{∘n}/n, whose n-th term has rank ≤ C(K + n − 2, n); so PMI departs from
    rank K − 1 only at second order in the association: |PMI − (R − 𝟙)| ≤ δ²/(2(1 − δ)) for δ = max|R − 1| < 1,
    and strongly associated mixtures have PMI well beyond rank K − 1. (iii) A rank-d log-bilinear model has
    rank PMI ≤ d + 2 while R = e^{PMI} is far beyond it. On the corpus, document-level noun co-occurrence is far
    closer to rank K − 1 = 5 in R − 𝟙𝟙ᵀ than in PPMI."""
    rng = np.random.default_rng([SEED, 13])
    for K in (2, 3, 5):
        for _ in range(10):
            P = mixture_joint(rng, 40, 30, K)
            R = ratio_matrix(P)
            assert numerical_rank(R) == K and numerical_rank(R - 1) == K - 1
            for n in (2, 3):
                assert numerical_rank((R - 1) ** n) <= math.comb(K + n - 2, n)
            L = np.log(R)
            assert residual_energy(np.linalg.svd(L, compute_uv=False), K - 1) > 1e-3  # strong: PMI is not rank K − 1
            # weaken the association: P_t = (1 − t) p pᵀ + t P is a mixture with R_t − 𝟙𝟙ᵀ = t(R − 𝟙𝟙ᵀ)
            res = []
            for t in (0.1, 0.01, 0.001):
                delta = t * np.abs(R - 1).max()
                Lt = np.log(1 + t * (R - 1))
                if delta < 1:
                    assert np.abs(Lt - t * (R - 1)).max() <= delta**2 / (2 * (1 - delta)) + 1e-15
                res.append(residual_energy(np.linalg.svd(Lt, compute_uv=False), K - 1))
            assert res[0] > res[1] > res[2] and 0.005 < res[2] / res[1] < 0.02  # the excess energy is O(t²)
    for d in (1, 2, 4):
        for _ in range(10):
            P = log_bilinear_joint(rng, 40, 30, d)
            assert numerical_rank(pmi_dense(P)) <= d + 2
            assert numerical_rank(ratio_matrix(P), 1e-9) > d + 2
    lr, ln = lowrank_summary(), lowrank_summary(True)
    assert lr["classes"] == K_SECTORS + 1
    assert lr["ratio_resid"] < 0.5 * lr["ppmi_resid"]
    assert lr["ratio_sv"][K_SECTORS - 2] > 2 * lr["ratio_sv"][K_SECTORS - 1]  # the gap after rank 5
    noise = ln["ratio_sv_raw"][0] / lr["ratio_sv_raw"][0]  # the shuffled corpus's largest, on the real scale
    assert lr["ratio_sv"][K_SECTORS - 2] > 2 * noise and lr["ratio_sv"][K_SECTORS - 1] < 1.25 * noise


def test_auc_matches_mannwhitney():
    """Def 4's AUC equals scipy's Mann–Whitney U / (n₁n₀), ties included, on random scores and on a corpus config."""
    rng = np.random.default_rng([SEED, 14])
    for _ in range(20):
        x = rng.integers(0, 6, 300).astype(float)
        y = rng.random(300) < 0.3
        u = stats.mannwhitneyu(x[y], x[~y]).statistic
        assert abs(auc(x, y) - u / (y.sum() * (~y).sum())) < 1e-12
    test, lab = benchmark_split()[1], planted_labels()
    S = similarity(weighted("L2", "ppmi"), test)
    la = np.array([lab[w] for w in test])
    iu = np.triu_indices(len(test), 1)
    same = (la[:, None] == la[None, :])[iu]
    u = stats.mannwhitneyu(S[iu][same], S[iu][~same]).statistic
    assert abs(pair_auc(weighted("L2", "ppmi"), test, lab) - u / (same.sum() * (~same).sum())) < 1e-12


def test_recovery_on_test_words():
    """§5. On the test nouns PPMI beats raw counts at every local context, in AUC and 1-NN, and wider contexts
    recover more; the dev-selected SVD beats the dev-selected sparse matrix in AUC by many standard errors at an
    equal 1-NN; on the rare band, which no selection saw, both beat counts; the document context is an oracle."""
    for ctx in LOCAL_CONTEXTS:
        a, b = scores((ctx, "counts", None, None)), scores((ctx, "ppmi", None, None))
        assert b["test_auc"] > a["test_auc"] and b["test_nn"] > a["test_nn"]
    aucs = [scores((ctx, "ppmi", None, None))["test_auc"] for ctx in LOCAL_CONTEXTS]
    assert all(x < y for x, y in zip(aucs[:3], aucs[1:4]))  # L1 < L2 < L4 < L8
    c = comparison()
    assert min(c["nn_sparse"], c["nn_svd"]) > 3 * c["test_chance_nn"]
    assert c["auc_svd"] - c["auc_sparse"] > 5 * c["auc_diff_se"]
    assert abs(c["nn_diff"]) <= 2 * c["nn_diff_se"]
    # ... but only in a narrow band of rank: most SVD configurations lose, and dev and test peak at the same d
    assert c["svd_beating_sparse_test"] < 0.3 * c["svd_configs_local"] and c["svd_beating_sparse_dev"] < 0.3 * c["svd_configs_local"]
    assert c["svd_peak_d_dev"] == c["svd_peak_d_test"] == selected()["svd"][2]
    assert min(c["rare_auc_sparse"], c["rare_auc_svd"]) > c["rare_auc_counts"]
    assert min(c["rare_nn_sparse"], c["rare_nn_svd"]) > c["rare_nn_counts"] > c["rare_chance_nn"]
    assert c["oracle_counts_test_nn"] == 1.0 and c["oracle_counts_test_auc"] > 0.99
    # Prop 4's smoothing does not help here: the dev-selected sparse matrix is unsmoothed, and beats its smoothed twin
    assert selected()["sparse"][1] == "ppmi"
    ctx = selected()["sparse"][0]
    assert scores((ctx, "ppmi", None, None))["dev_auc"] > scores((ctx, "ppmi_a", None, None))["dev_auc"]


def test_finance_findings():
    """§6. Antonyms are window neighbors; window polarity is near chance; SO recovers polarity from documents only;
    the polysemes score near zero; EDGAR coverage is partial."""
    f = finance_summary()
    nbrs = [w for w, _ in f["strong_neighbors_L2"]]
    assert len(set(nbrs) & set(polarity_words()["negative"])) >= 2
    for ctx in ("L1", "L2", "L4"):
        assert abs(f[f"adj_within_{ctx}"] - f[f"adj_cross_{ctx}"]) < 0.05
        assert f[f"so_coverage_{ctx}"] == 0 and f[f"so_acc_{ctx}"] == 0.5  # no evidence at all
    for ctx in ("L8", "sentence"):  # evidence pointing the wrong way
        assert f[f"so_acc_{ctx}"] < 0.4 and f[f"so_mean_positive_{ctx}"] < 0 < f[f"so_mean_negative_{ctx}"]
    assert f["so_coverage_document"] == 1 and f["so_acc_document"] > 0.8
    assert f["so_mean_positive_document"] > 0 > f["so_mean_negative_document"]
    assert abs(f["polyseme_so_mean"]) < 0.25 * f["lexicon_abs_so_median"]
    assert f["polyseme_abs_so_median"] < 0.5 * f["lexicon_abs_so_median"]
    assert 0.2 < f["edgar_covered"] < 0.8


def test_viz_payload():
    """The lab's data: every cell has neighbors for every lab word, sorted by cosine; the sweep has one value per
    d; spectra have D_MAX values; the low-rank inset shows the gap."""
    p = viz_payload()
    assert len(p["cells"]) == len(CONTEXTS) * len(LAB_WEIGHTINGS)
    for key, cell in p["cells"].items():
        ctx, wt = key.split("|")
        assert cell["test_auc"] == _r(scores(lab_config(ctx, wt))["test_auc"])
        for w in LAB_WORDS:
            nb = cell["neighbors"][w]
            assert len(nb) == LAB_TOP and all(nb[i][1] >= nb[i + 1][1] for i in range(LAB_TOP - 1))
            assert w not in [x[0] for x in nb] and all(p["tags"].get(x[0], "other") == word_tag(x[0]) for x in nb)
    for row in p["sweep"]["by_p"].values():
        assert len(row["auc"]) == len(row["nn"]) == len(SVD_DIMS)
    sel = p["selected"]["svd"]
    assert p["sweep"]["by_p"][str(sel[3])]["auc"][SVD_DIMS.index(sel[2])] == p["cells"][f"{sel[0]}|svd"]["test_auc"]
    assert all(len(v["real"]) == len(v["null"]) == SPECTRUM_K for v in p["spectrum"].values())
    assert p["spectrum"]["ppmi"]["null"][0] > p["spectrum"]["ppmi"]["real"][0]  # Prop 3, seen in the spectrum
    lo = p["lowrank"]
    assert lo["ratio"][lo["rank"] - 1] > 2 * lo["ratio"][lo["rank"]] and lo["ratio"][lo["rank"]] < 1.25 * lo["noise"]
    assert sum(x < 0 for x in p["eig_l2"]) == spectrum_summary()["n_negative_top100"]
    assert len(json.dumps(p, separators=(",", ":"))) < 40_000


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
    print("\nSummary (train generate_corpus(300, 0); benchmark = planted sector nouns, dev/test split):")
    for k, v in summary().items():
        print(f"  {k:40} {v:.4g}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else
              f"  {k:40} {v:.4f}" if isinstance(v, float) else f"  {k:40} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
