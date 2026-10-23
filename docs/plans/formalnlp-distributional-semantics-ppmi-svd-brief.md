# Brief — `distributional-semantics-ppmi-svd`

**Distributional Semantics: PPMI and Truncated SVD.** Roadmap #6; prerequisite:
`zipf-heaps-vocabulary-statistics` (published). The first `representations` topic. Prerequisite of
`word2vec-implicit-factorization` (#7) and `topic-models-lda` (#25).

- Phase A (this brief and the verified notebook): 2026-10-23. Harness 15/15 under Python 3.15.0, numpy==2.5.4, and
  scipy==1.18.1 in ≈ 9 s. The narrative notebook (23 cells) executes (exit 0, ≈ 12.5 s) and is committed without
  outputs.
- Scaffold decisions (Jonathan, 2026-10-23):
  - the spine is Thms 1–3 and Props 1–4, with Thm 3 (two kinds of low rank) and Prop 3 (PPMI under independence)
    both in. Cor 1 (word + context) fell out of Thm 2 during authoring. Parallel analysis for choosing d and a
    bipartite-spectrum proposition are out;
  - hyperparameters are chosen on a **dev/test split of the benchmark nouns**, not of the text;
  - selection runs over the **local contexts only** (L1–L8, sentence). Document context is an oracle row, because the
    generator plants each noun's sector at the document level;
  - **one lab, two linked panels**;
  - every function body was written by Claude at Jonathan's request, for his review.
- Source of truth: `notebooks/distributional-semantics-ppmi-svd/distributional_semantics_ppmi_svd.py`. Every number
  below is printed or asserted by it. Nothing here may be changed without changing it.
- Binding forward promises:
  - Topic 5, Thm 6 and §"Where this goes next": *"A low-rank matrix of log-probabilities, or of association scores
    like pointwise mutual information, is also what word embeddings factorize."* Def 2 sets up the shifted PMI matrix
    PMI − log k that #7 will show skip-gram factorizes, without proving #7's result. Thm 2 is its truncated-SVD
    factorization. Thm 3 is the dual of Topic 5's Thm 6: a mixture makes the ratio matrix low rank, a log-bilinear
    model makes the PMI matrix low rank.
  - Topic 1 (the prerequisite): its frequency spectrum, applied to pair counts, measures the singleton artifact
    (Prop 2), and its Zipf law supplies the margins of Prop 3's Monte Carlo. Both import Topic 1's `.py`.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Distributional Semantics: PPMI and Truncated SVD |
| `{{SUBTITLE}}` | Know a word by the company it keeps: pointwise mutual information subtracts the background every count row shares, a truncated SVD keeps its best low-rank picture, and the context, not the method, decides whether that picture shows topic or polarity. |
| `{{SLUG}}` | `distributional-semantics-ppmi-svd` (= graph node id) |
| `{{DOMAIN}}` | `representations` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[zipf-heaps-vocabulary-statistics]` |
| `{{FORMALML_PREREQS}}` | `svd`, `spectral-theorem`, `kl-divergence`; connections `density-ratio-estimation`, `pca-low-rank` |
| `{{FORMALRAG_PREREQS}}` | none; connections `pmi-retrieval-value`, `vector-space-model-tfidf`, `pca-dimensionality-reduction` |
| `{{STATS_CALC_PREREQS}}` | Stats `discrete-distributions`; connection `multiple-testing-and-false-discovery`; Calc `power-taylor-series` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/zipf-heaps-vocabulary-statistics/`, `notebooks/n-gram-models-smoothing/`, `notebooks/finance-corpus/` |

Tags: `distributional-semantics`, `pointwise-mutual-information`, `ppmi`, `truncated-svd`, `word-embeddings`,
`co-occurrence`, `low-rank`, `semantic-orientation`. `estimatedReadTime` ≈ 40. In-site `connections`: none, since both
dependents are still planned. The edge `zipf-heaps-vocabulary-statistics → distributional-semantics-ppmi-svd` already
exists in the graph.

**Scope (watch-list: formalRAG `pmi-retrieval-value`, design doc §6).**
- **formalRAG `pmi-retrieval-value`** owns PMI as *retrieval value*:
  - pmi(a; d | q) between a generator's answer and a retrieved document;
  - the conditional MI I(A; D | Q) as entropy reduction (its Thm 1);
  - the sign of pmi at the true answer (Prop 1);
  - saturation by the chain rule (Prop 2);
  - InfoNCE as a lower bound (Cor 1).

  It never builds a word × context matrix, never estimates PMI from counts, and never factorizes. **This topic** owns
  PMI as a *word-association matrix*: its estimation from co-occurrence counts, its biases (the ceiling, singletons,
  the independence null, smoothing), its reweightings (PPMI, SPPMI_k), and its factorization (truncated SVD, the
  eigenvalue weight, two kinds of low rank). The shared algebra, E_{P(c|w)}[PMI] = KL and Σ_w P(w) KL = I(W; C), is
  Prop 1 (ii)–(iii), *cited* to that page's Thm 1 and not re-proved. The page links there from Prop 1.
- **formalML `svd`** owns the SVD and Eckart–Young–Mirsky (its Thm 2), and its §7 covers LSA, the SVD of a
  term-document matrix. **formalML `spectral-theorem`** owns the spectral theorem (its Thm 1). **formalRAG
  `pca-dimensionality-reduction`** owns PCA for embeddings, its own Eckart–Young statement, and LSA as the truncated
  SVD of TF-IDF. We cite Eckart–Young and the spectral theorem and prove only what is new here: what W_p W_pᵀ
  approximates (Thm 2), the w + c identity (Cor 1), and Thm 3. LSA is pointed to, never re-derived. Our document
  context is DDᵀ, LSA's Gram matrix, and the page says so once.
- **formalRAG `vector-space-model-tfidf`** owns cosine similarity (its Def 5), the magnitude invariance (Prop 2), and
  IDF as self-information. We use cosine as defined there. **TF-IDF, BM25, and every retrieval model stay in
  formalRAG** (CLAUDE.md): cross-link only.
- Out of scope apart from pointers:
  - skip-gram/word2vec as implicit factorization (#7). Def 2's SPPMI_k is the matrix it factorizes, and Cor 1 is the
    w + c trick it inherits;
  - GloVe (#8);
  - contextual embeddings (#12);
  - lexicon sentiment (#18). Turney's SO is used here only as a distributional probe;
  - LDA (#25). Thm 3 (i) is its geometry, with no Dirichlet priors and no inference;
  - Dunning's G² and collocations (#26), the cure for Prop 3's false positives.

## 1. Section outline (geometric first; each section opens with a picture or a corpus number)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | at L2 (PPMI), the nearest neighbors of *strong*: improved, stronger, healthy, **weak**, **soft**, confident, **weaker** | — | cells 0–1 |
| 1 | Words as points | raw count rows: mean cosine 0.648 within a sector, 0.638 across; PPMI 0.056 and 0.029 | Def 1 (co-occurrence matrix), Def 2 (PMI_α, PPMI, SPPMI_k); **Prop 1** (log-ratio coordinates); **Thm 1** (PMI removes the background) | cells 2–3 |
| 2 | What counts cannot tell | 72.6% of L2's nonzero pairs are seen once; a shuffled corpus has more PPMI energy than the real one | **Prop 2** (ceiling; the singleton artifact); **Prop 3** (PPMI under independence) + remark; **Prop 4** (smoothing threshold t*) | cells 4–9 |
| 3 | Truncated SVD | L2's PPMI is symmetric with λ₂ = −83.9: 44 of its top 100 eigenvalues are negative | Def 3 (W_p, C_p); **Thm 2** (what p measures); **Cor 1** (word + context keeps the positive part) | cells 10–12 |
| 4 | Two kinds of low rank | σ(R − 𝟙𝟙ᵀ) for 115 nouns, normalized: 1, .963, .924, .617, .421, then **.184** | **Thm 3** (mixture ⇒ ratio low rank; Hadamard series; log-bilinear ⇒ PMI low rank) | cells 13–14 |
| 5 | Recovering the planted sectors | test AUC 0.813 (sparse) against 0.973 (SVD) at equal 1-NN 0.965 | Def 4 (benchmark, 1-NN, pair AUC); the protocol | cells 15–17 |
| — | The lab | `<DistributionalLab client:visible />` | — | cells 20–21 |
| 6 | Finance case study: what distribution cannot see | windows give no polarity evidence, sentences give it backwards, documents give it right | Turney's SO as a probe | cells 18–19 |
| 7 | Where this goes next | — | #7 (SPPMI_k is skip-gram's target; Cor 1 is its w + c), #8 (GloVe; Thm 3 iii), #12 (one vector per type), #18 (SO needs documents), #25 (Thm 3 i), #26 (Dunning's test cures Prop 3) | cell 22 |

Lab embed: one `<DistributionalLab client:visible />` after §5, because its readouts are §5's test scores. Panel A
needs §§1, 2, 5, and 6. Panel B needs §§2–5. Forward pointers go from Def 2 (to #7), Thm 3 (to #7, #8, #25), and
Prop 3 (to #26).

## 2. Definitions

- **Text.**
  - Topic 3's training split, `generate_corpus(300, 0)`: 6,130 sentences, 63,182 word tokens, and 3,425 word types.
    The types are Topic 3's V without `</s>` and `<unk>`, and they are the rows and columns of every matrix.
  - `_generator_run()` rebuilds the corpus from its own generator. `test_planted_labels` asserts the text is
    identical to `generate_corpus(300, 0)` and to `ng.corpus()["train"]`.
  - EDGAR appears only in §6's coverage figure. There is no held-out text in this topic: the held-out unit is the
    benchmark noun (Def 4).
- **Planted labels.** The generator draws each content noun from one of six sector Pitman–Yor restaurants or from a
  shared one.
  - Seeded nouns belong to one sector by construction, and pseudo-words are never reused across restaurants. The one
    seed in two sectors (*reserves*, banking and energy) is dropped.
  - 1,800 training nouns come from a sector restaurant. 227 of them are seen ≥ 5 times: industrials 101,
    technology 40, banking 35, energy 18, retail 17, healthcare 16.
  - Industrials dominates because the corpus's coverage is Zipf-skewed toward its first company.
- **Split.** Within each sector, a seeded permutation (`rng [SEED, 0]`) sends half the nouns to **dev (112)** and the
  rest to **test (115)**. The **rare band** is the 396 sector nouns seen 2–4 times. No selection ever sees it, so it
  is a second test set.

- **Def 1 (co-occurrence matrix).**
  - For a context type κ (a window L ∈ {1, 2, 4, 8} inside a sentence, the whole sentence, or the whole document),
    C[w, c] counts the ordered token pairs (i, j), i ≠ j, in one unit, with w at i, c at j, and |i − j| ≤ L for a
    window.
  - C is symmetric, and a word co-occurs with its own other tokens. Write N = Σ C and n_w = Σ_c C[w, c].
  - Whole-unit counts are SSᵀ − diag(S𝟙) for the word × unit incidence matrix S. For documents this is LSA's Gram
    matrix DDᵀ, less its diagonal correction.
  - L2: N = 215,948 over 53,293 nonzero pairs. Sentence: 661,904 over 158,054. Document: 13,442,460 over 1,285,899.
  - `test_cooccurrence_definition` checks the vectorized counts against a double loop on random sentences.
- **Def 2 (PMI, PPMI, SPPMI; Church and Hanks 1990; Levy and Goldberg 2014; Levy, Goldberg, and Dagan 2015).**
  - P(w, c) = C/N, P(w) = n_w/N, and the smoothed context distribution is P_α(c) = n_c^α / Σ n_c'^α.
  - PMI_α(w, c) = log P(w, c)/(P(w)P_α(c)), which is −∞ off the support.
  - PPMI_α = max(PMI_α, 0) and SPPMI_{α,k} = max(PMI_α − log k, 0).
  - The weightings used: counts; PPMI (α = 1, k = 1); PPMI_α (α = 0.75); SPPMI₅ and SPPMI₁₅ (α = 0.75).
  - The *unclipped* PMI − log k is the matrix #7 shows skip-gram with k negative samples factorizes. This page names
    it and does not prove it.
- **Def 3 (truncated SVD).** M = UΣVᵀ; word vectors W_p = U_dΣ_d^p and context vectors C_p = V_dΣ_d^{1−p}, with
  p ∈ {0, ½, 1} and d ∈ {10, 25, 50, 100, 200, 300}. ARPACK (`scipy.sparse.linalg.svds`, rng-seeded start) computes the
  top 300 once per (context, base), and every smaller d truncates it. `test_svds_matches_dense_svd` checks the top
  50 on a 700-word block against `numpy.linalg.svd`: singular values to 10⁻⁸ relative, and M_d to 10⁻⁶.
- **Def 4 (benchmark).**
  - **1-NN accuracy** is the share of nouns whose nearest other noun (by cosine, among the same split's nouns) has
    the same sector. Chance is Σ share² = 0.269 on test and 0.216 on the rare band.
  - **Pair AUC** is the Mann–Whitney probability that a random same-sector pair has a higher cosine than a random
    cross-sector pair (ties count ½). `test_auc_matches_mannwhitney` checks it against `scipy.stats.mannwhitneyu`.
  - Standard errors: jackknife over nouns for the AUC difference (pairs share nouns), and paired over nouns for the
    1-NN difference.
- **Protocol.**
  - Within each family (24 sparse configurations; 270 SVD configurations over three bases {PPMI, PPMI_α, SPPMI₅},
    six d, and three p), select the configuration with the highest **dev** AUC over LOCAL_CONTEXTS, dev 1-NN
    breaking ties.
  - The metric was declared before selection. Test nouns play no part.
  - Document context is excluded from selection and reported as an oracle (§4 below).

## 3. Statements and proofs

### Proposition 1 (log-ratio coordinates)
For a joint distribution with full-support marginals:
- (i) P(c | w) = P(c) e^{PMI(w,c)}, so a PMI row and the marginal determine the context distribution;
- (ii) Σ_c P(c | w) PMI(w, c) = KL(P(·|w) ‖ P_C) ≥ 0, with equality iff w is independent of its contexts;
- (iii) Σ_w P(w) KL(P(·|w) ‖ P_C) = I(W; C).

*Proof.* (i) is the definition. (ii) follows by substituting (i), with Gibbs' inequality for the sign. (iii) is the
definition of MI as KL from the joint to the product, the identity formalRAG `pmi-retrieval-value` proves (its Thm 1)
for answers and documents. ∎

**Asserted** (`test_pmi_log_ratio_coordinates`), on 50 random 12 × 9 joints:
- (i) holds to rounding;
- the KL matches `scipy.stats.entropy` and is positive;
- Σ P(w) KL equals the MI to 10⁻¹²;
- the PMI of a product distribution is 0.

### Theorem 1 (PMI removes the background)
Let P be a context distribution with full support, and let P(·|w) = (1 − ε_w)P + ε_w q_w with q_w ≠ P. Write
f_w = q_w/P − 𝟙. As ε_w, ε_w′ → 0:
- (a) the cosine of the two words' count rows tends to 1, with 1 − cos = O(ε²);
- (b) PMI_w/ε_w → f_w, so cos(PMI_w, PMI_w′) → cos(f_w, f_w′) with error O(ε);
- (c) Σ_c P(c) f_w(c) = 0.

*Proof.*
- (a) A count row is n_w(P + ε_w(q_w − P)), so each row's angle to P is O(ε), the angle between the two rows is
  O(ε), and 1 − cos θ ≤ θ²/2.
- (b) PMI_w(c) = log(1 + ε_w f_w(c)) = ε_w f_w(c) + O(ε_w²‖f_w‖∞²), uniformly once ε_w‖f_w‖∞ ≤ ½, since
  |log(1 + x) − x| ≤ x² there (formalCalculus `power-taylor-series`). Cosine is scale-free and continuous away from 0.
- (c) Σ_c P(c)(q_w(c)/P(c) − 1) = 1 − 1. ∎

*Reading.* Every count row shares the corpus background P. Near the background all count rows point the same way, so
counts measure how corpus-like a word is. PMI subtracts the background in log coordinates (Prop 1), and only the
word-specific part f_w, which is P-centered, is left. The theorem is a limit statement about words near the
background. It is not a claim that real rows have small ε.

- **Asserted** on 30 random instances (`test_pmi_removes_background`), with ε = 10⁻ʲ/‖f‖∞, j = 2 … 5 (the
  asymptotic regime starts at ε‖f‖∞ ≪ 1; a fixed ε grid was pre-asymptotic when ‖f‖∞ reached 2,800):
  - the PMI cosine error falls ≥ 5× per decade and is below 10⁻⁴ at j = 5;
  - the count gap 1 − cos falls ≥ 50× per decade;
  - (c) holds to 10⁻¹².
- **Corpus** (L2, test nouns):
  - raw count rows: mean cosine **0.648** within a sector and **0.638** across. Counts barely tell sectors apart;
  - PPMI: **0.056** within and **0.029** across. The absolute level is low, and same-sector pairs are twice as close.

### Proposition 2 (the ceiling, and the singleton artifact)
- (i) PMI(w, c) ≤ min(−log P(w), −log P(c)), with equality in the first iff c occurs only with w. For smoothed
  contexts, PMI_α(w, c) ≤ −log P_α(c).
- (ii) On a pair seen exactly once, the plug-in PMI_α(w, c) = −log n_w − log P_α(c). For α = 1 this is
  log N − log n_w − log n_c, a function of the two frequencies alone and additive in w and c.

*Proof.* (i) follows from P(w, c) ≤ P(c) and P(w, c) ≤ P(w). (ii) follows by substituting C[w, c] = 1. ∎

**Asserted** (`test_pmi_ceiling_and_singletons`):
- the ceiling, and its equality case, on 50 random joints;
- on the corpus, (ii) to 10⁻¹² on **every** singleton pair at L2, for both α = 1 and α = 0.75;
- the ceiling on every entry.

Corpus, through Topic 1's frequency spectrum applied to the pair counts:
- **38,674 of L2's 53,293 nonzero pairs (72.6%) are seen once**. They make up 73.3% of the PPMI support.
- The largest PPMI is 10.49 on rows of once-seen words, 9.80 for words seen 2–4 times, 8.88 for 5–19, 7.38 for
  20–99, and 5.73 for 100+.

Most of the matrix is frequency, not association.

### Proposition 3 (PPMI under independence)
Let X ~ Poisson(μ) be a pair's count when W ⊥ C with known margins, μ = n_w n_c/N, so its true PMI is 0. Let
PPMI = max(log(X/μ), 0). Then:
- (i) P(PPMI > 0) = P(X > μ);
- (ii) if μ < 1, P(PPMI > 0) = 1 − e^{−μ}, every positive value is at least log(1/μ), and
  E[PPMI] ≥ (1 − e^{−μ}) log(1/μ).

*Proof.* (i) log(X/μ) > 0 iff X > μ. (ii) For μ < 1, X > μ iff X ≥ 1, and then log(X/μ) ≥ log(1/μ). ∎

*Remark (asserted, not proved).* As μ → ∞, P(PPMI > 0) → ½ and √(2πμ)·E[PPMI] → 1, by the central limit theorem for
(X − μ)/√μ (E[Z₊] = 1/√(2π)). Rare pairs give rare but large false positives, and frequent pairs give frequent but
tiny ones. PPMI tests every pair at once, which is a multiple-testing problem (formalStatistics
`multiple-testing-and-false-discovery`). The cure, a likelihood-ratio test (Dunning 1993), is the collocations
topic's.

**Asserted** (`test_ppmi_under_independence`):
- (i)–(ii) by exact Poisson summation (`scipy.stats.poisson`) at μ ∈ {10⁻³, …, 0.9};
- the remark at μ ∈ {10², 10³, 10⁴}, to 0.03 and 0.02;
- a Monte Carlo independence matrix with Zipf margins (`zhv.zipf_pmf(1.1, 400)`, N = 2·10⁵) puts most PPMI energy
  on pairs with μ < 1.

Corpus, L2, with the **permutation null**: every training token shuffled corpus-wide, sentence and document lengths
kept, so every margin is fixed and every association destroyed:
- **90.5%** of the real PPMI's 50,648 positive entries have μ < 1. A single co-occurrence could have produced each of
  them under independence.
- The shuffled text has **94,860** positive entries, where Poisson independence with the training margins predicts
  93,603 (1.3% off). Its ‖PPMI‖²_F is **729,130**, more than the real **691,005**. At this scale, PPMI's magnitude is
  mostly sampling noise; the signal is in *which* entries are positive.
- Shifting by log 5 reverses the comparison: SPPMI₅ has energy 234,580 for the real text against 206,808 shuffled.

### Proposition 4 (context-distribution smoothing)
For 0 < α < 1, P_α(c)/P(c) = (N/Z_α) n_c^{α−1}, with Z_α = Σ n_c^α. This ratio is strictly decreasing in n_c and equals
1 at t* = (N/Z_α)^{1/(1−α)}. So PMI_α < PMI exactly on contexts rarer than t*. For k ≥ k′,
supp SPPMI_{α,k} ⊆ supp SPPMI_{α,k′}.

*Proof.* n ↦ n^{α−1} is strictly decreasing, and PMI_α − PMI = −log(P_α(c)/P(c)). The shift subtracts log k from
every entry. ∎

**Asserted** (`test_context_smoothing_threshold`):
- on 30 random count vectors with random α, the ratio is decreasing and crosses 1 at t*;
- on the corpus, PMI_α < PMI ⟺ n_c < t* on every support entry, and the SPPMI supports are nested.

Corpus, L2:
- **t* = 338.4**. 95.8% of context types, carrying 34.8% of context tokens, lie below it.
- Support sizes are 50,648 (PPMI), 50,618 (PPMI_α), 36,382 (SPPMI₅), and 22,795 (SPPMI₁₅).

Smoothing tempers rare contexts, but a singleton's PMI_α is still −log n_w − log P_α(c) (Prop 2). On the benchmark,
α = 0.75 does not help. The dev-selected sparse matrix is the unsmoothed one: sentence PPMI has dev AUC 0.798 at
α = 1 against 0.793 at α = 0.75 (asserted in `test_recovery_on_test_words`).

### Theorem 2 (what the eigenvalue weight measures)
- (i) W_pC_pᵀ = M_d for every p. This is the best rank-d approximation of M in Frobenius and spectral norms
  (Eckart and Young 1936; Mirsky 1960; formalML `svd`, Thm 2).
- (ii) W_pW_pᵀ = U_dΣ_d^{2p}U_dᵀ = [(MMᵀ)^p]_d, the best rank-d approximation of the PSD matrix (MMᵀ)^p. At p = 1 it
  approximates second-order co-occurrence (shared contexts; Schütze 1998), and at p = 0 it is the projector U_dU_dᵀ.
- (iii) Let M be symmetric, with eigenvalues ordered by modulus. Then W_½W_½ᵀ = |M|_d = Q_d|Λ_d|Q_dᵀ. This equals M_d
  iff λ₁, …, λ_d ≥ 0, and ‖|M|_d − M_d‖_F = 2(Σ_{i≤d, λ_i<0} λ_i²)^{½}.

*Proof.*
- (i) Substitute.
- (ii) MMᵀ = UΣ²Uᵀ, so (MMᵀ)^p = UΣ^{2p}Uᵀ is a spectral decomposition of a PSD matrix, whose best rank-d
  approximation keeps the top d eigenpairs.
- (iii) By the spectral theorem (formalML `spectral-theorem`, Thm 1), M = QΛQᵀ, and σ_i = |λ_i|, u_i = q_i,
  v_i = sign(λ_i) q_i is an SVD. Then |M|_d − M_d = Σ_{i≤d, λ_i<0} 2|λ_i| q_iq_iᵀ, whose Frobenius norm follows from
  orthonormality. Joint sign flips of (u_i, v_i) change nothing. ∎

**Asserted** (`test_eigenvalue_weighting`):
- on random 30 × 20 and 25 × 25 matrices, (i) for p ∈ {0, ¼, ½, 1}, and M_d beats 20 random rank-d competitors each
  time;
- (ii) against an independent eigendecomposition of (MMᵀ)^p;
- (iii) on random symmetric PSD and indefinite matrices: |M|_d = M_d exactly when the top-d eigenvalues are ≥ 0, and
  the gap formula holds to 10⁻⁹.

Corpus (L2 PPMI with α = 1, which is symmetric to 9·10⁻¹⁶):
- Of its 100 eigenvalues of largest modulus, **44 are negative**, including **λ₂ = −83.9** (λ₁ = 137.4).
- At d = 100, ‖|M|_d − M_d‖_F/‖M_d‖_F = **1.248**: the p = ½ word–word Gram is not the PMI approximation at all.
- λ₂'s eigenvector splits template words (*and, alongside, toward, in, customers, added, highlighted, markets*) from
  content pseudo-nouns (*channelline, nonreceivables, royaltyside, wellflow, couponflow, …*). **58.7%** of PPMI mass
  crosses the cut, so the matrix is close to bipartite: in a window, a word keeps company with the other side, not
  its own.

### Corollary 1 (adding context vectors keeps the positive part)
For symmetric M, W_½ + C_½ = 2Q_d[Λ_d]₊^{½}: the eigen-directions with λ < 0 cancel. Hence
¼(W_½ + C_½)(W_½ + C_½)ᵀ = [M_d]₊ = Σ_{i≤d, λ_i>0} λ_i q_iq_iᵀ.

*Proof.* With u_i = s_iq_i and v_i = s_i sign(λ_i) q_i for any signs s_i, √σ_i(u_i + v_i) = s_i√|λ_i|(1 + sign λ_i)q_i. ∎

Levy, Goldberg, and Dagan's (2015) "w + c" heuristic is, for a symmetric window, exactly the projection onto the
positive eigenspace.

**Asserted** (`test_word_plus_context`):
- on 20 random symmetric matrices, under random joint sign flips;
- on the corpus, with ARPACK's vectors: cos(u_i, v_i) = ±1 for all 100 directions, the negative ones cancel in W + C
  to 10⁻⁶, and their count (44) matches Thm 2's eigensolver;
- the reading, at the configuration where Thm 2 is checked (L2, PPMI, d = 100, p = ½, not tuned): on test nouns,
  W gives AUC 0.712 and 1-NN 0.809, while (W + C)/2 gives **0.755** and **0.878**.

### Theorem 3 (two kinds of low rank)
Let R = [P(w, c)/(P(w)P(c))] = e^{∘PMI}.
- (i) If W ⊥ C | Z for a latent Z ∈ {1, …, K}, so that P = Σ_z π_z φ_zψ_zᵀ (pLSA's aspect model; Hofmann 1999), then
  R = AΠBᵀ with A = D_w⁻¹Φ and B = D_c⁻¹Ψ nonnegative and Π = diag π. Hence rank R ≤ K, and
  R − 𝟙𝟙ᵀ = A(Π − ππᵀ)Bᵀ has rank ≤ K − 1.
- (ii) When δ = max|R − 1| < 1, PMI = Σ_{n≥1} ((−1)^{n+1}/n)(R − 𝟙𝟙ᵀ)^{∘n}. The n-th term has rank ≤ C(K + n − 2, n),
  and ‖PMI − (R − 𝟙𝟙ᵀ)‖_max ≤ δ²/(2(1 − δ)). The PMI of a weakly associated mixture is nearly rank K − 1; that of a
  strongly associated one need not be.
- (iii) If PMI = XYᵀ + a𝟙ᵀ + 𝟙bᵀ with X, Y of width d (a log-bilinear model, the form skip-gram and GloVe fit), then
  rank PMI ≤ d + 2, while R = e^{∘PMI} need not be low rank.

*Proof.*
- (i) P(w) = (Φπ)_w, so Aπ = 𝟙, and likewise Bπ = 𝟙. Hence 𝟙𝟙ᵀ = Aππᵀ Bᵀ, and Π − ππᵀ annihilates 𝟙.
- (ii) Apply the series of log(1 + x) entrywise (formalCalculus `power-taylor-series`). A rank-r matrix Σ_{i≤r} a_ib_iᵀ
  has n-th Hadamard power Σ_{i₁≤…≤i_n} c(a_{i₁}∘…∘a_{i_n})(b_{i₁}∘…∘b_{i_n})ᵀ, a sum of C(r + n − 1, n) rank-one terms;
  take r = K − 1. The bound is |log(1 + x) − x| ≤ Σ_{n≥2}|x|ⁿ/2 = x²/(2(1 − |x|)).
- (iii) Rank is subadditive. ∎

*Reading.* Topic models (#25) assume (i); embeddings (#7–8) assume (iii). The two hypotheses about the same matrix
agree to first order in the strength of association and part ways as it grows. Topic 5's Thm 6 is (iii) for
log-probabilities, and its mixture of softmaxes escapes (iii) the way (i)'s mixtures do. That a mixture's PMI is "full
rank" was this brief's first draft. The harness showed the claim false for weakly associated mixtures, which is how
(ii) entered.

**Asserted** (`test_mixture_ratio_low_rank`):
- on random K-mixtures (K ∈ {2, 3, 5}, 40 × 30):
  - rank R = K and rank(R − 𝟙𝟙ᵀ) = K − 1;
  - the Hadamard-power rank bound for n = 2, 3;
  - strong association leaves > 10⁻³ of log R's energy beyond rank K − 1;
- along the weakened family P_t = (1 − t)ppᵀ + tP, where R_t − 𝟙𝟙ᵀ = t(R − 𝟙𝟙ᵀ):
  - the max-norm bound;
  - the excess energy falls as O(t²), with a ratio in (0.005, 0.02) per decade;
- on random log-bilinear models (d ∈ {1, 2, 4}), rank PMI ≤ d + 2 while R exceeds it.

**Corpus.** Document context, the 115 content nouns (from all seven restaurants) seen ≥ 20 times, with the
submatrix's own margins; 15.2% of pairs are unseen. Within a document the generator draws every content noun from
that document's sector restaurant or the common one, so a six-sector mixture predicts rank(R − 𝟙𝟙ᵀ) ≤ 5.

| normalized σᵢ | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| R − 𝟙𝟙ᵀ | 1 | .963 | .924 | .617 | .421 | **.184** | .171 | .157 |
| PPMI | 1 | .749 | .588 | .547 | .492 | .344 | .240 | .213 |

The gap comes after rank 5 = K − 1, and σ₆ sits at the shuffled corpus's noise level (its largest σ of R − 𝟙𝟙ᵀ is
0.161 of the real largest). Energy beyond rank 5 is **11.6%** for R − 𝟙𝟙ᵀ and **27.8%** for PPMI.

## 4. Recovering the planted sectors (§5) and the finance case study (§6)

**The local table** (`test_recovery_on_test_words`; test nouns; AUC / 1-NN):

| context | counts | PPMI |
|---|---|---|
| L1 | 0.491 / 0.304 | 0.581 / 0.609 |
| L2 | 0.526 / 0.574 | 0.712 / 0.826 |
| L4 | 0.548 / 0.652 | 0.796 / 0.957 |
| L8 | 0.555 / 0.652 | 0.813 / 0.957 |
| sentence | 0.560 / 0.635 | 0.813 / 0.965 |
| *document (oracle)* | *0.992 / 1.000* | *1.000 / 1.000* |

- PPMI beats counts at every local context, in both measures (Thm 1).
- AUC rises with the window, L1 < L2 < L4 < L8. Sector is a topical, document-level property, so wider contexts see
  more of it.
- Document co-occurrence reads the planted label itself: even raw counts reach test 1-NN 1.000. That is why it is
  excluded from selection.

**The selected pair** (dev-selected, scored on test):

| family | selected on dev | dev AUC / 1-NN | test AUC / 1-NN |
|---|---|---|---|
| sparse | sentence, PPMI | 0.798 / 0.866 | **0.813 / 0.965** |
| SVD | L8, SPPMI₅, d = 25, p = 0 | 0.951 / 0.946 | **0.973 / 0.965** |

- SVD − sparse = **+0.160 test AUC** (jackknife se 0.019, 8.3 se), at an **equal** 1-NN (0.965 both; paired se of
  the difference 0.021). Truncation buys global ranking, not nearest neighbors: sparse PPMI already puts each noun's
  nearest neighbor in its sector 96.5% of the time. Chance is 0.269.
- **Narrow band.** The win holds only in a narrow band of rank, and the harness asserts this:
  - 59 of 270 local SVD configurations beat the sparse matrix on test AUC (62 on dev);
  - at the selected (L8, SPPMI₅, p = 0), test AUC by d ∈ {10, 25, 50, 100, 200, 300} is 0.524, **0.973**, 0.948,
    0.766, 0.631, 0.601;
  - dev and test peak at the same d = 25;
  - at d = 10, the top directions are frequency and template structure (Prop 2, Thm 2), not sectors;
  - p = 0 (whitening) wins at d = 25, while p = 1 decays most slowly in d (0.774 at d = 300).
- **Rare band** (396 nouns seen 2–4 times, never used for selection):

  | | counts | sparse | SVD |
  |---|---|---|---|
  | AUC | 0.517 | 0.575 | **0.728** |
  | 1-NN | 0.394 | **0.717** | 0.657 |

  Chance 1-NN is 0.216. SVD again buys global ranking and gives up a little at the nearest neighbor.

### Finance case study (§6)

`test_finance_findings`. The stand-in lexicon's polarity words, 18 positive and 17 negative present in training,
never the real Loughran–McDonald dictionary. In the generator's templates both polarities fill the same slots
(*…, which management described as* **strong** / **weak**).

1. **Antonyms are window neighbors.**
   - At L2 (PPMI), the seven nearest neighbors of *strong* are improved (0.408), stronger, healthy, **weak** (0.359),
     **soft**, confident, and **weaker**. The lab's L2|PPMI cell shows the first five.
   - Polarity adjectives are as close across polarity as within it: 0.986 / 0.984 at L1, 0.265 / 0.248 at L2, and
     0.214 / 0.212 at L4.
   - Distributional similarity is substitutability, and an antonym is the best substitute.
2. **Polarity from context: three regimes.** Turney's (2002) semantic orientation:
   - Definition: SO(w) = PMI(w, POS) − PMI(w, NEG), leave-one-out, with each seed set merged into one paradigm context
     (Turney and Littman 2003) and ½ added to both counts.
   - A word with no seed co-occurrence scores 0, a coin flip worth ½. Without that rule the marginal term alone gave
     those words the rarer seed set's polarity, which produced a spurious 0.771 at L1–L4 in a draft.

   | context | coverage | accuracy | mean SO, positive / negative |
   |---|---|---|---|
   | L1, L2, L4 | 0 | 0.500 | 0 / 0 |
   | L8 | 0.657 | **0.343** | −0.326 / +0.310 |
   | sentence | 0.657 | **0.286** | −0.728 / +0.384 |
   | document | 1.000 | **0.829** | +0.270 / −0.363 |

   - In a window of four or fewer tokens, no polarity word ever meets another, so there is no evidence.
   - Within a sentence the evidence points *backwards*. Contrastive sentences (*…, but costs rose …*) and *not* +
     opposite-polarity constructions pair opposite polarities.
   - Across a document the tone majority shows. Polarity is a document-level property here, as sector is, and only a
     context of that size can see it.
3. **The polysemes** (*liability, liabilities, tax, taxes, cost, costs, crude, capital*):
   - document-level SO puts them near zero, with median |SO| 0.059 against the lexicon words' 0.259 and a mean of
     +0.019;
   - in this domain their distribution reads as neutral, which is Loughran and McDonald's (2011) point, reached by
     distribution instead of a hand list;
   - two are not near zero, *liability* (+0.344) and *crude* (−0.401). SO from 300 documents is noisy word by word.
4. **EDGAR.** 40.0% of EDGAR's 797 word tokens have a training vector (39.0% have one from ≥ 5 occurrences). A
   type-level embedding has nothing to say about the rest, and the subword topic (#2) and the contextual-embedding
   topic (#12) are the two responses.

## 5. `rigorFlag`

> The corpus is synthetic and its ground truth is planted: a noun's "sector" is the Pitman–Yor restaurant the
> generator drew it from, so recovery here measures how well a method reads structure we put there, not how well it
> captures meaning in English. Document-level co-occurrence reads that label almost directly and is excluded from
> selection; every reported score is on held-out benchmark nouns, with hyperparameters chosen on the other half, but
> the benchmark is 227 nouns (115 for testing), 44% from one sector. The grid of contexts, weightings, ranks, and
> weights, and the decision to exclude document contexts, were shaped by exploratory runs that scored all 227 nouns
> before the split was fixed. Truncated SVD beats sparse PPMI in global ranking, not in nearest neighbors, and only
> in a narrow band of rank: 59 of 270 SVD configurations beat the sparse matrix on test nouns, and the selected rank
> is a sharp peak. Theorem 3's mixture structure holds for the generator only approximately (Pitman–Yor draws are
> exchangeable, not independent, given the sector) and is checked on 115 frequent nouns. Under independence most
> positive PPMI entries are noise at this scale: a shuffled corpus has more PPMI energy than the real one. The
> polarity results use a 35-word hand-written stand-in lexicon, not the Loughran–McDonald dictionary, and the three
> regimes they show are properties of the generator's templates. The large-μ limit of Proposition 3 is asserted
> numerically, not proved.

## 6. Viz design intent — `DistributionalLab` (one component, two linked panels)

All data comes from `viz_payload()`. Mirror it to the decimal in `src/components/viz/data/distributional-lab.json` and
add a Phase B `test_viz_mirror_matches_payload` (the Topic 1–5 pattern). The payload is 37.6 kB; `test_viz_payload`
caps it at 40 kB. Use Marginalia tokens only: one `--color-*` per sector and per lexicon tag, mapped in the
component.

**Panel A, "The company a word keeps."**
- Controls:
  - a word picker over `LAB_WORDS`: *strong, weak, liability, orders, software, loans, production, stores, trials,
    and, litigation*;
  - a context selector over the six contexts, with *document* badged "oracle";
  - a weighting toggle: counts, PPMI, SPPMI₅, and SVD (the dev-selected SPPMI₅, d = 25, p = 0, at the chosen context).
- The view is the word's top five neighbors as horizontal cosine bars, each with a tag chip from `tags` (sector,
  lexicon category, polyseme, common; absent = other).
- Readouts are the cell's test AUC and 1-NN, against chance 0.269. When a polarity word is picked, also show the
  context's SO accuracy and coverage (`so_accuracy`, `so_coverage`).
- *Lessons:*
  - counts return template and frequency neighbors, and PPMI returns sector neighbors once the window widens;
  - *strong*'s neighbors at L2 include *weak, soft, weaker*;
  - the SVD's neighbors of *loans, stores, trials, software, orders* are sector-pure at L8;
  - *production*'s neighbors are the words of the template *Crude production averaged … thousand barrels per day*,
    which is syntagmatic company rather than paradigmatic;
  - polarity evidence is absent in windows, reversed in sentences, and right in documents.

**Panel B, "Rank and spectrum."** Two views behind a toggle:
- *Rank:* test AUC and 1-NN against d ∈ {10, …, 300} for p ∈ {0, ½, 1} at (L8, SPPMI₅) (`sweep`). Horizontal lines
  mark the sparse winner (0.813 / 0.965). A marker at the selected (25, 0) lets the reader see the sharp peak.
- *Spectrum:*
  - the L2 singular values (log y), real against shuffled, for PPMI and SPPMI₅ (`spectrum`, 100 values, 4 significant
    figures). For PPMI the shuffled curve starts *above* the real one (155.3 against 137.4: Prop 3, seen in a
    spectrum). For SPPMI₅ it starts below.
  - Under it, the 100 signed eigenvalues of L2's PPMI as up and down bars (`eig_l2`, 44 negative).
  - Inset: Thm 3's two normalized spectra (`lowrank`), with the rank-5 marker and the noise line at 0.161.
- *Lessons:*
  - SVD's gain lives in a narrow band of d;
  - unshifted PPMI's top singular value is no larger than a shuffled corpus's, and shifting changes that;
  - half the large eigenvalues are negative, which is what p = ½ word–word dot products misread (Thm 2, Cor 1);
  - the ratio matrix shows the planted rank, and PPMI smears it.

**Link.** Choosing the SVD weighting in Panel A moves Panel B's marker to the selected (d, p). Choosing a context in
Panel A shows its 1-NN and AUC in Panel B's rank view whenever that context is L8 (the sweep's).

Mobile: neighbor bars stack full-width, slider rows get `flexWrap: 'wrap'`, inputs get `minWidth: 0`, and long
pseudo-words in chips are truncated with an ellipsis (Topic 5 lesson).

## 7. Notebook cell map (`01_distributional_semantics_ppmi_svd.ipynb`, 23 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; corpus, planted labels, split | `test_planted_labels` |
| 2–3 | §1 Defs 1–2, Prop 1, Thm 1; corpus cosines | `test_cooccurrence_definition`, `test_pmi_log_ratio_coordinates`, `test_pmi_removes_background` |
| 4–5 | §2 Prop 2; Topic 1's spectrum of pair counts | `test_pmi_ceiling_and_singletons` |
| 6–7 | Prop 3 and remark; the permutation null | `test_ppmi_under_independence` |
| 8–9 | Prop 4; t*, supports | `test_context_smoothing_threshold` |
| 10–12 | §3 Def 3, Thm 2, Cor 1; spectrum, w + c | `test_svds_matches_dense_svd`, `test_eigenvalue_weighting`, `test_word_plus_context` |
| 13–14 | §4 Thm 3; the noun spectra | `test_mixture_ratio_low_rank` |
| 15–17 | §5 Def 4, protocol; the table, selection, sweep, comparison, rare band | `test_auc_matches_mannwhitney`, `test_recovery_on_test_words` |
| 18–19 | §6 finance | `test_finance_findings` |
| 20–22 | viz payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Run:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/distributional-semantics-ppmi-svd/distributional_semantics_ppmi_svd.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/distributional-semantics-ppmi-svd/01_distributional_semantics_ppmi_svd.ipynb
```

## 8. Cross-site links (all verified `status: "published"` on GitHub `main` 2026-10-23; re-verify at the B3 gate)

| field | topic | relationship |
|---|---|---|
| `formalmlPrereqs` | `svd` | Truncated SVD gives the word vectors, and Eckart–Young says the word-by-context product is the best rank-d approximation of the PMI matrix; this page proves what word-by-word products approximate. |
| `formalmlPrereqs` | `spectral-theorem` | A symmetric window makes the PMI matrix symmetric, and its eigendecomposition shows why half-power word vectors misread its negative eigenvalues and why adding context vectors removes them. |
| `formalmlPrereqs` | `kl-divergence` | The average PMI of a word over its own contexts is the KL divergence from the corpus's context distribution, and its frequency-weighted sum is the mutual information of words and contexts. |
| `formalmlConnections` | `density-ratio-estimation` | The exponentiated PMI matrix is the density ratio of the joint word–context distribution to the product of its marginals; mixtures make this ratio, not its logarithm, low rank. |
| `formalmlConnections` | `pca-low-rank` | Word vectors are an uncentered truncated SVD of a sparse PMI matrix, not PCA: centering would destroy sparsity, and the eigenvalue weight replaces PCA's variance scaling. |
| `formalragConnections` | `pmi-retrieval-value` | Both pages use pointwise mutual information: there it scores what a retrieved document adds to an answer, here it is a word-association matrix estimated from counts and factorized. |
| `formalragConnections` | `vector-space-model-tfidf` | Word vectors are compared by the cosine defined there; PMI reweights a word's co-occurrence row as IDF reweights a document's term vector, against a background rate. |
| `formalragConnections` | `pca-dimensionality-reduction` | Latent semantic analysis is the truncated SVD of a term-document matrix; document-level co-occurrence here is that matrix's Gram matrix, reweighted by PMI. |
| `formalstatisticsPrereqs` | `discrete-distributions` | Under independence a pair's co-occurrence count is Poisson, and that model shows why positive PMI on rare pairs is mostly a false positive. |
| `formalstatisticsConnections` | `multiple-testing-and-false-discovery` | Positive PMI tests every word pair at once against independence; its false positives concentrate on rare pairs, a multiple-testing problem that a likelihood-ratio test addresses. |
| `formalcalculusPrereqs` | `power-taylor-series` | The series of log(1 + x) shows that PMI removes the background near independence and that a mixture's PMI departs from low rank only at second order. |

Not cited:
- formalML `shannon-entropy` (KL carries the MI);
- formalML `representation-learning` (embeddings proper are #7's subject);
- formalStatistics `central-limit-theorem` (Prop 3's remark names the CLT in prose; add it as a connection in Phase B
  if the page links it);
- formalCalculus `eigenvalues-eigenvectors` (`spectral-theorem` carries it).

## 9. References (Chicago 17th, notes-and-bibliography; every URL verified 2026-10-23 — DOI titles via Crossref)

1. Harris, Zellig S. "Distributional Structure." *Word* 10, no. 2–3 (1954): 146–62.
   https://doi.org/10.1080/00437956.1954.11659520. — paper; the distributional hypothesis.
2. Firth, J. R. "A Synopsis of Linguistic Theory, 1930–1955." In *Studies in Linguistic Analysis*, 1–32. Oxford: Basil
   Blackwell, 1957. https://archive.org/details/studiesinlinguis0000vari. — book; "the company it keeps."
3. Church, Kenneth Ward, and Patrick Hanks. "Word Association Norms, Mutual Information, and Lexicography."
   *Computational Linguistics* 16, no. 1 (1990): 22–29. https://aclanthology.org/J90-1003/. — paper; Def 2.
4. Deerwester, Scott, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. "Indexing by Latent
   Semantic Analysis." *Journal of the American Society for Information Science* 41, no. 6 (1990): 391–407.
   https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9. — paper; LSA (pointer).
5. Dunning, Ted. "Accurate Methods for the Statistics of Surprise and Coincidence." *Computational Linguistics* 19,
   no. 1 (1993): 61–74. https://aclanthology.org/J93-1003/. — paper; Prop 3's cure.
6. Schütze, Hinrich. "Automatic Word Sense Discrimination." *Computational Linguistics* 24, no. 1 (1998): 97–123.
   https://aclanthology.org/J98-1004/. — paper; second-order co-occurrence (Thm 2 ii).
7. Hofmann, Thomas. "Probabilistic Latent Semantic Analysis." In *Proceedings of the 15th Conference on Uncertainty
   in Artificial Intelligence*, 289–96. Stockholm, 1999. https://arxiv.org/abs/1301.6705. — paper; Thm 3 (i).
8. Turney, Peter D. "Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of
   Reviews." In *Proceedings of the 40th Annual Meeting of the ACL*, 417–24. Philadelphia: ACL, 2002.
   https://aclanthology.org/P02-1053/. — paper; SO (§6).
9. Turney, Peter D., and Michael L. Littman. "Measuring Praise and Criticism: Inference of Semantic Orientation from
   Association." *ACM Transactions on Information Systems* 21, no. 4 (2003): 315–46.
   https://doi.org/10.1145/944012.944013. — paper; merged seed sets.
10. Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet Allocation." *Journal of Machine Learning
    Research* 3 (2003): 993–1022. https://www.jmlr.org/papers/v3/blei03a.html. — paper; pointer to #25.
11. Bullinaria, John A., and Joseph P. Levy. "Extracting Semantic Representations from Word Co-occurrence Statistics:
    A Computational Study." *Behavior Research Methods* 39, no. 3 (2007): 510–26. https://doi.org/10.3758/BF03193020.
    — paper; PPMI with cosine.
12. Turney, Peter D., and Patrick Pantel. "From Frequency to Meaning: Vector Space Models of Semantics." *Journal of
    Artificial Intelligence Research* 37 (2010): 141–88. https://doi.org/10.1613/jair.2934. — paper; survey.
13. Loughran, Tim, and Bill McDonald. "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and
    10-Ks." *Journal of Finance* 66, no. 1 (2011): 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x. — paper;
    the polysemes (§6).
14. Bullinaria, John A., and Joseph P. Levy. "Extracting Semantic Representations from Word Co-occurrence Statistics:
    Stop-Lists, Stemming, and SVD." *Behavior Research Methods* 44, no. 3 (2012): 890–907.
    https://doi.org/10.3758/s13428-011-0183-8. — paper; SVD and its eigenvalue weighting.
15. Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. "Distributed Representations of Words
    and Phrases and Their Compositionality." In *Advances in Neural Information Processing Systems* 26 (2013).
    https://arxiv.org/abs/1310.4546. — paper; skip-gram with negative sampling (pointer).
16. Levy, Omer, and Yoav Goldberg. "Neural Word Embedding as Implicit Matrix Factorization." In *Advances in Neural
    Information Processing Systems* 27 (2014).
    https://proceedings.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html. —
    paper; SPPMI_k (Def 2), proved in #7.
17. Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation."
    In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1532–43. Doha: ACL,
    2014. https://aclanthology.org/D14-1162/. — paper; pointer to #8.
18. Levy, Omer, Yoav Goldberg, and Ido Dagan. "Improving Distributional Similarity with Lessons Learned from Word
    Embeddings." *Transactions of the Association for Computational Linguistics* 3 (2015): 211–25.
    https://doi.org/10.1162/tacl_a_00134. — paper; α = 0.75, the eigenvalue weight, w + c (Prop 4, Thm 2, Cor 1).
19. Eckart, Carl, and Gale Young. "The Approximation of One Matrix by Another of Lower Rank." *Psychometrika* 1, no. 3
    (1936): 211–18. https://doi.org/10.1007/BF02288367. — paper; Thm 2 (i).
20. Mirsky, L. "Symmetric Gauge Functions and Unitarily Invariant Norms." *Quarterly Journal of Mathematics* 11, no. 1
    (1960): 50–59. https://doi.org/10.1093/qmath/11.1.50. — paper; Thm 2 (i), all unitarily invariant norms.
21. Mann, H. B., and D. R. Whitney. "On a Test of Whether One of Two Random Variables Is Stochastically Larger than
    the Other." *Annals of Mathematical Statistics* 18, no. 1 (1947): 50–60. https://doi.org/10.1214/aoms/1177730491.
    — paper; pair AUC (Def 4).
22. Efron, Bradley. *The Jackknife, the Bootstrap and Other Resampling Plans*. Philadelphia: SIAM, 1982.
    https://doi.org/10.1137/1.9781611970319. — book; jackknife standard errors (§5).
23. Horn, Roger A., and Charles R. Johnson. *Topics in Matrix Analysis*. Cambridge: Cambridge University Press, 1991.
    https://doi.org/10.1017/CBO9780511840371. — book; Hadamard products (Thm 3 ii).
24. Lehoucq, R. B., D. C. Sorensen, and C. Yang. *ARPACK Users' Guide: Solution of Large-Scale Eigenvalue Problems
    with Implicitly Restarted Arnoldi Methods*. Philadelphia: SIAM, 1998. https://doi.org/10.1137/1.9780898719628. —
    book; the sparse SVD (Def 3).

## 10. Phase B notes

- MDX at `src/content/topics/distributional-semantics-ppmi-svd.mdx`. Page numbering follows page order: Defs 1–4,
  Props 1–4, Thms 1–3, Cor 1. Prop 3's large-μ limit is an unnumbered remark. `references[].type`: 2, 22, 23, and 24
  are `book`, and the rest are `paper`.
- Graph: flip the node's `status` to `published` and remove the title from `planned` in `src/data/curriculum.ts`.
  The edge from `zipf-heaps-vocabulary-statistics` already exists. Frontmatter: `prerequisites:
  [zipf-heaps-vocabulary-statistics]`, and no in-site `connections`.
- The lab reads `src/components/viz/data/distributional-lab.json`
  (`json.dumps(viz_payload(), separators=(',', ':'))`). Add `test_viz_mirror_matches_payload`. Payload rounding is
  deliberate: neighbor cosines to 2 decimals, scores to 4, and spectra and eigenvalues to 4 significant figures (they
  span decades on a log axis). `_r` and `_sig` map non-finite values to `null`, and none occur.
- Keep matrix notation safe for MDX: write 𝟙𝟙ᵀ as `\mathbb{1}\mathbb{1}^\top`, and Hadamard powers as `^{\circ n}`.
  No bare `<` or `{` in prose. Use `\lt` for "μ < 1" inside math. Lexicon and template words go in italics, not
  KaTeX.
- Long displays go to display math: Thm 2 (iii)'s gap, Thm 3 (i)'s factorization, (ii)'s series, Prop 4's t*. At
  375 px the prose column is about 343 px. The §5 table and the SO table may need horizontal scroll wrappers.
- Overview hook: *strong*'s L2 neighbors under PPMI (α = 1). These are the same matrix and the same order as the lab's
  L2|PPMI cell, so the reader can reproduce the hook in the lab.
- Link back to Topic 1 from Prop 2 (the frequency spectrum of pair counts) and from the Zipf margins in Prop 3. Link
  Topic 5's Thm 6 from Thm 3's reading. Topic 5's §"Where this goes next" promised exactly this matrix.
- Re-run the harness at the B3 gate (≈ 9 s, far under budget). If the lab needs more neighbor cells, the 40 kB
  payload cap in `test_viz_payload` is the constraint to revisit, not the time.

## 11. Phase B record (2026-10-23)

- **Numbering and placement.**
  - Page numbering matches the brief: Defs 1–4, Props 1–4, Thms 1–3, Cor 1, in page order (verified in the built
    DOM). Prop 3's large-μ limit is an unnumbered remark.
  - 𝟙 is written `\mathbf{1}`, as in Topic 5, rather than `\mathbb{1}`.
  - The lab, `DistributionalLab`, sits after §5 under "The company a word keeps, and the shape of the spectrum."
- **Payload and component.**
  - `test_viz_mirror_matches_payload` requires `src/components/viz/data/distributional-lab.json` (37.6 kB) to equal
    `viz_payload()` exactly. The harness is 16/16 (≈ 9 s), and the notebook exits 0.
  - Tag chips carry their tag as text: sectors are filled from the badge, definition, and theorem tokens, and lexicon
    categories are outlined. Color is never the only cue, and no hex is used.
  - The link between panels: choosing PPMI or SPPMI₅ in Panel A opens Panel B's spectrum for that matrix, and
    choosing SVD opens the rank view with the selected point enlarged.
- **Overview hook.** It uses *strong*'s L2 PPMI neighbors (α = 1), the lab's default cell. The finance summary was
  switched from PPMI_α to PPMI before Phase B so the two agree, and §6's cosines are the PPMI ones (L2: 0.265 / 0.248).
- **Cross-site.** All eleven targets were re-verified `status: "published"` on GitHub `main`.
- **Gates.**
  - `pnpm test` passes 39/39, and `astro sync` and `pnpm validate` are clean.
  - `pnpm audit:cross-site` exits 0. The eleven outward links appear as missing reciprocals, as expected until
    launch, and formalNLP has 0 deferred and 0 drifted targets.
  - `pnpm build` exits 0.
- **DOM checks** on the built page (preview pane shown, desktop width):
  - 0 `.katex-error`, 433 `.katex`, and `noindex, nofollow` intact.
  - The lab hydrates once `[data-lab="distributional"]` is scrolled into view.
  - Readouts equal the payload:
    - the default L2 | PPMI | *strong* cell gives improved, stronger, healthy, weak, soft, with AUC 0.712, 1-NN
      0.826, and SO 0.500 at coverage 0;
    - document context gives *robust* first and SO 0.829, and sentence context gives SO 0.286;
    - SVD at L8 shows *loans, stores, trials, software, orders* sector-pure, and *production*'s neighbors are
      template words;
    - the rank view shows 0.973 against 0.813, with d = 10 at 0.524 and d = 300 at 0.601; its 1-NN view shows 0.965
      for both;
    - the spectra are PPMI 137.4 real against 155.3 shuffled and SPPMI₅ 56.4 against 42.7, with 44 negative
      eigenvalues and 11.6% residual energy for the ratio matrix.
  - The console shows no errors.
- **Mobile.** `innerWidth === 375` before and after hydration, `scrollWidth` 375, no unclipped overflow, and the lab
  is 343 px wide.
