# Brief — `glove-embedding-geometry`

**GloVe and the Geometry of Word Embeddings.** Roadmap #8. Prerequisite: `word2vec-implicit-factorization` (#7,
PR #13), published. This is the third `representations` topic and a prerequisite of `knowledge-graph-embeddings`
(#24), which makes "a relation is a vector offset" its training objective.

**Status: Phase A scaffold** (2026-10-23). This draft holds the outline, statements with proof sketches, function
signatures, and the cell map, for Jonathan's reaction before any function body is written. Numbers marked *spike*
come from throwaway runs in the planning session. Each will be re-derived and asserted by the `.py`, or dropped.

**Scope decisions** (Jonathan, 2026-10-23):
- **In:** every optional result — the flat-weight theorem, the scale experiment and the zeros fix, the ratio
  argument, and ridge fixing the gauge.
- **Checks:** internal only — Topic 5's complex step, closed forms against L-BFGS on small instances, numpy's SVD,
  and ALS against AdaGrad. No new data.
- **Benchmark:** controlled, with GloVe's published settings — Topic 6's dev/test nouns, the same L8 counts,
  d = 25, x_max = 100, γ = ¾, 50 AdaGrad epochs, against Topic 7's 40-epoch SGNS and SVD(SPPMI₅). No dev grid.
- **Finance:** relations from the filings knowledge base as vector offsets.
- **Difficulty:** `intermediate`.
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–7.

**Binding forward promises, and where they are kept:**
- **Topic 6, §"Where this goes next":** "GloVe fits a log-bilinear model to log co-occurrence counts, the form of
  Theorem 3's third part." Kept by Prop 2.
- **Topic 7, §"Where this goes next":** GloVe fits log #(w, c) by weighted least squares with explicit biases and a
  weight that grows with the count. Analogies are word–word geometry and so, by its Prop 3, about the gauge.
  - Thm 1 and Cor 1 compare f(X) with SGNS's ab/(a + b), and with the Poisson weight X that both temper.
  - Prop 4 widens the gauge to affine maps once biases are present.
  - Thm 3 and Cor 2 split analogies into the exact, gauge-free part (a statement about count ratios) and the
    nearest-neighbor answer, which depends on the gauge.
  - The rigorFlag treats analogy arithmetic as approximate.

**Thesis.** GloVe is a tempered, support-restricted Poisson log-bilinear fit.
- **The weight** decides how far each count is trusted (§2).
- **The support restriction** decides what the vectors cannot see. On a small corpus, where most observed pairs
  are seen once, that is nearly everything (§3).
- **The biases** widen the gauge to affine maps, so the objective fixes neither similarity nor the
  nearest-neighbor answer to an analogy. It fixes only the exact parallelograms, and those are facts about count
  ratios (§5).

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | GloVe and the Geometry of Word Embeddings |
| `{{SUBTITLE}}` | *(draft)* GloVe is a tempered Poisson fit to log counts that ignores every pair it never saw; its biases widen the gauge to affine maps, so exact analogies are facts about count ratios while the answers a nearest-neighbor search returns depend on the factorization. |
| `{{SLUG}}` | `glove-embedding-geometry` (= graph node id) |
| `{{DOMAIN}}` | `representations` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[word2vec-implicit-factorization]` |
| `{{FORMALML_PREREQS}}` | *candidates* `svd`, `information-geometry`, `pca-low-rank`; connections `gradient-descent`, `convex-analysis` (nuclear norm), `representation-learning` |
| `{{FORMALRAG_PREREQS}}` | none; *candidate* connections `hypersphere-vmf-geometry`, `dense-retrieval-dual-encoders` |
| `{{STATS_CALC_PREREQS}}` | *candidates* Stats `generalized-linear-models`, `linear-regression`; connections `regularization-and-penalized-estimation`, `exponential-families`; Calc `mean-value-taylor` |
| `{{RIGOR_FLAG}}` | §5 below (draft) |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/word2vec-implicit-factorization/`, `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/finance-corpus/` |

- **Tags:** `glove`, `word-embeddings`, `weighted-least-squares`, `poisson-regression`, `log-bilinear-model`,
  `word-analogies`, `gauge-freedom`, `matrix-factorization`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`:** none, since #24 is still planned.
- **Graph:** the incoming edge from `word2vec-implicit-factorization` and the outgoing edge to
  `knowledge-graph-embeddings` already exist.

**Scope.** This topic is not on the design doc's §6 watch-list, but it adjoins:
- **formalRAG `hypersphere-vmf-geometry`**, which owns cosine geometry on the sphere: its Prop 1 (cosine and
  distance give the same ranking), the vMF distribution, and anisotropy of embedding clouds. We use cosine as
  defined there and do not re-treat anisotropy or concentration. The "all-but-the-top" isotropy correction (Mu and
  Viswanath 2018) is at most a pointer.
- **formalRAG `dense-retrieval-dual-encoders`**: inner-product scores and the rank ceiling (its Thm 2). Cited for
  the gauge reading of any two-tower inner product.
- **formalML `svd`** (Eckart–Young, its Thm 2) and **`pca-low-rank`** (PCA via SVD of the centered matrix). Thm 2
  cites both and proves only the double-centering step.
- **formalStatistics `generalized-linear-models`** (canonical-link concavity, IRLS = Fisher scoring, its Thm 2).
  Thm 1's reading cites it; the Poisson GLM is not re-derived.
- **This topic owns:**
  - the GloVe objective and its ratio derivation;
  - GloVe as a tempered, support-restricted Poisson fit;
  - the flat-weight theorem;
  - support blindness;
  - the affine gauge, and ridge's choice within it;
  - exact parallelograms and their count-ratio criterion;
  - analogy evaluation and its gauge dependence.
- **Out of scope apart from pointers:**
  - contextual embeddings (#12);
  - TransE and relation learning (#24): §6 only previews them;
  - isotropy post-processing;
  - fastText;
  - debiasing;
  - retrieval and ANN, which stay in formalRAG (CLAUDE.md).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | the same L8 counts give SGNS test AUC 0.886 and GloVe ≈ 0.5 (*spike*); and "same loss, different analogies" | — | cells 0–1 |
| 1 | A log-bilinear model of counts | *revenue*'s log-count row against two context words' ratio rows | Def 1; **Prop 1** (ratio argument); **Prop 2** (rank ≤ d + 2) | cells 2–3 |
| 2 | Three weights | one pair's loss curve under Poisson, SGNS, and GloVe | **Thm 1** (tempered Poisson); **Cor 1** (three local weights); corpus: weight mass by count band | cells 4–5 |
| 3 | What the loss cannot see | the binarized matrix: biases fit it exactly, and the vectors are zero | **Thm 2** (flat weights + full support); **Prop 3** (support blindness); Def 2 (imputed zeros); corpus: benchmark, scale curve, imputation | cells 6–9 |
| 4 | The algorithm | — | Def 3 (GloVe's AdaGrad, as run); ALS as exact block descent | cells 10–11 |
| 5 | Gauge and analogies | one city → region parallelogram under three factorizations of the same scores | **Prop 4** (affine gauge); **Prop 5** (ridge: balanced and centered); Def 4 (3CosAdd); **Thm 3** (exact parallelograms); **Cor 2** (3CosAdd and the gauge); corpus gauge table | cells 12–14 |
| — | The lab | `<GloVeLab client:visible />` | — | cells 17–18 |
| 6 | Finance case study: a filings knowledge base as offsets | *Houston : Texas :: Boston : ?* | Def 4 applied; Thm 3's defect | cells 15–16 |
| 7 | Where this goes next | — | #12, #24 (TransE: w_h + r ≈ w_t as the objective), formalRAG | — |

- **Lab placement.** One `<GloVeLab client:visible />` after §5. Panel A needs §2. Panel B needs §§3 and 5.
- **Back-links:**
  - Topic 6's Thm 3 (iii) from Prop 2;
  - Topic 6's Def 3 from Thm 2;
  - Topic 7's Thm 2 and Cor 1 from Thm 1 and Cor 1;
  - Topic 7's Prop 3 from Prop 4;
  - Topic 7's Lemma 1 and Cor 2 from Prop 5.

## 2. Definitions

**Notation.**
- X is Topic 6's L8 count matrix #(w, c), symmetric, V = 3,425. Ω = {(i, j) : X_ij > 0} is its support:
  143,345 cells, 1.22% of V².
- Scores are s_ij = ⟨w_i, c_j⟩ + b_i + b̃_j, with W, C ∈ ℝ^{V×d} and b, b̃ ∈ ℝ^V.
- PMI is Topic 6's Def 2 (α = 1 unless stated).
- J_n = I − 𝟙𝟙ᵀ/n is the centering matrix.

**Definitions.**
- **Def 1 (GloVe; Pennington, Socher, and Manning 2014).**
  J(W, C, b, b̃) = Σ_{(i,j)∈Ω} f(X_ij)(s_ij − log X_ij)², with f(x) = min{1, (x/x_max)^γ}, x_max = 100, γ = ¾.
  - The reported vectors are W + C (Pennington et al.). W alone is also scored.
  - γ is used rather than the paper's α, which Topics 6–7 use for context smoothing. The page says so once.
- **Def 2 (GloVe with imputed zeros).**
  - Set X̃ = X on Ω and X̃ = ½ off Ω, and minimize J on X̃ over all V² cells. Unseen pairs count as seen half a
    time, with GloVe's own weight f(½) ≈ 0.019.
  - This parameter-free variant is ours. Swivel (Shazeer et al. 2016) is the published model that "notices what's
    missing", with a soft hinge on unobserved cells. The page cites it and does not implement it.
  - Training is by ALS, with the uniform part handled by the implicit-feedback identity (Hu, Koren, and Volinsky
    2008).
- **Def 3 (GloVe's AdaGrad, as run).**
  - Each epoch is a fresh permutation of Ω in minibatches of 512, with an AdaGrad step on each parameter row
    (Duchi, Hazan, and Singer 2011), η = 0.05, and accumulators started at 1.
  - Initialization follows the GloVe C code: (U[0, 1] − ½)/d for every vector entry and bias.
  - 50 epochs (the paper's setting for d < 300). Float64.
  - **Departure, stated on the page:** minibatch summation replaces the C code's per-cell Hogwild updates. The
    counts are Topic 6's unweighted window counts, not GloVe's 1/distance weighting, so that all three methods
    factor the same matrix.
- **Def 4 (3CosAdd; Mikolov, Yih, and Zweig 2013).** For a : b :: c : ?, answer
  argmax_{x∉{a,b,c}} cos(x̂, b̂ − â + ĉ), where hats are unit vectors.
  - "Open" means over the whole vocabulary; "within" restricts the candidates to the relation's tail set.
  - Also reported: the share of questions whose top candidate *without* the exclusion is b itself (Linzen 2016).
  - 3CosMul (Levy and Goldberg 2014b) is a possible secondary row.

**Configurations.**
- **Controlled GloVe:** L8, d = 25, Def 1's defaults, 50 epochs. This is the same matrix and rank as Topic 7's
  controlled SGNS and Topic 6's selected SVD.
- **Cross-check:** ALS on the same objective with ridge 10⁻³ (reported separately, with its own loss).
- **Scale:** the generator continued to 1,200 and 3,000 documents (same restaurants, so the same benchmark nouns),
  with counts on Topic 6's 300-document vocabulary.

## 3. Statements and proof sketches

Full proofs go in the brief once the harness passes. Each statement lists the asserts planned for it.

### Proposition 1 (the ratio argument, with its hypotheses)
Let F: ℝ → (0, ∞) be continuous with F(u − v) = F(u)/F(v) for all u, v. Then F(u) = e^{λu} for some λ. If, in
addition, F(⟨w_i − w_k, c_j⟩) = P(j | i)/P(j | k) for all words i, k and contexts j, and λ ≠ 0, then
⟨w_i, c_j⟩ = λ⁻¹(log X_ij − log X_i + β_j) for constants β_j. Absorbing λ into W and replacing the word term −log X_i
with a free b_i, which restores the word–context symmetry the model lacks, gives Def 1's s_ij.

*Sketch.*
1. G = log F satisfies G(0) = 0, G(−v) = −G(v), and G(u + v) = G(u) + G(v), which is Cauchy's equation.
2. Continuity gives G linear (on the rationals first, then by density).
3. The ratio identity makes F(⟨w_i, c_j⟩)/P(j | i) independent of i.

*Reading.* The derivation's hypotheses are modeling choices, not facts about data: ratios as the relevant
statistic, a function of a dot product of differences, and symmetry restored by hand. The page says this, and the
rigorFlag repeats it.

**Asserted** (`test_ratio_argument`):
- scores built as log P(j | i) + β_j satisfy the ratio identity with F = exp to 10⁻¹² on random joints;
- a continuous non-exponential F (for example 1 + u²) violates F(u − v) = F(u)/F(v) on a grid;
- after the symmetrization, scores equal to log X on a full-support instance give J = 0.

### Proposition 2 (log-bilinear form; Topic 6 Thm 3 iii)
If s_ij = log X_ij on every cell (full support, exact fit), then
PMI = WCᵀ + a𝟙ᵀ + 𝟙βᵀ with a_i = b_i − log X_i· + log N and β_j = b̃_j − log X_·j. So rank PMI ≤ d + 2, and the
same holds for PMI_α, since α moves only the column offset.

*Sketch.* Substitute log X_ij = PMI_ij + log X_i· + log X_·j − log N, then use subadditivity of rank.

**Asserted** (`test_log_bilinear_rank`): random log-bilinear count tables (d ∈ {1, 2, 4}). The numerical rank of PMI
is ≤ d + 2 and the stated offsets are recovered to 10⁻¹⁰.

### Theorem 1 (GloVe is a tempered, support-restricted Poisson fit)
Let X_ij ~ Poisson(e^{s_ij}) independently, with per-cell negative log-likelihood ℓ(s; x) = e^s − xs + log x!.
- **(i)** For x > 0, ℓ has the unique minimizer s* = log x. With δ = s − log x,
  ℓ(s; x) − ℓ(s*; x) = x(e^δ − 1 − δ) = D_A(s, s*),
  the Bregman divergence of the Poisson log-partition A(s) = e^s. Its curvature at s* is x, and
  ½xδ²e^{min(δ,0)} ≤ x(e^δ − 1 − δ) ≤ ½xδ²e^{max(δ,0)}.
- **(ii)** For x = 0, ℓ(s; 0) = e^s decreases strictly toward 0 as s → −∞ and has no minimizer. This is the
  a = 0 case of Topic 7's Thm 1.
- **(iii)** Hence ½Σ_Ω X_ij(s_ij − log X_ij)² is the second-order Poisson fit on the support. Def 1 is that sum
  with the Fisher weight X replaced by f(X), and with the zero cells' term Σ_{Ω^c} e^{s_ij} dropped.

*Sketch.*
1. e^s = xe^δ.
2. The Bregman form is the exponential-family identity of Topic 7's Thm 2 with A = exp.
3. The sandwich is Taylor's theorem with the Lagrange remainder (formalCalculus `mean-value-taylor`):
   e^δ − 1 − δ = ½δ²e^ξ, with ξ between 0 and δ.

*Reading.* One Fisher-scoring (IRLS) step of the Poisson GLM at the saturated fit has working response log X and
weight X (formalStatistics `generalized-linear-models`, Thm 2). Pennington et al. (2014, §3.1) reach weighted least
squares on logs from skip-gram's global cross-entropy and then replace the weight X_i by f(X_ij). Thm 1 says which
likelihood that is the quadratic of.

**Asserted** (`test_poisson_bregman`):
- (i) the identity and the Bregman form to 10⁻¹² on a grid of (x, δ);
- the sandwich holds;
- the numerical curvature equals x;
- (ii) strict decrease for x = 0;
- (iii) on random instances, the quadratic and the exact excess NLL agree to O(δ³).

### Corollary 1 (three local weights)
Near their observed-cell optima, all three objectives are weighted least squares in a log-count coordinate:

| objective | local loss | weight | unobserved cell |
|---|---|---|---|
| Poisson | ½X(s − log X)² | X | e^s, pushed toward −∞ |
| SGNS (Topic 7 Thm 2, Cor 1) | ½ ab/(a + b) (w·c + log k#(w) + log P_α(c) − log a)² | ab/(a + b) ≤ min(a, b) | b log(1 + e^{w·c}), pushed toward −∞ |
| GloVe | f(X)(s − log X)² | f(X) ≤ 1 | no term |

- SGNS's local loss is GloVe's residual with the biases **frozen** at b_w = log k#(w) and b̃_c = log P_α(c) (Shi
  and Liu 2014).
- For X ≤ x_max, f(X)/X = X^{γ−1}x_max^{−γ} decreases in X. Relative to Poisson, GloVe upweights rare pairs by
  X^{−1/4}. SGNS's weight is ≈ a when a ≪ b, so on rare pairs it is Poisson's.

*Sketch.* Thm 1 (i). Topic 7's x* = log a − log b with b = k#(w)P_α(c).

**Asserted** (`test_three_weights`):
- the frozen-bias identity to 10⁻¹² on random instances;
- the weight inequalities;
- **corpus:** each weight's share of its total on pairs seen ≤ 4 times, on all of Ω and on the benchmark nouns'
  rows. Topic 7 reported 54.8% for SGNS on the benchmark rows. GloVe's and Poisson's shares are to be measured.

### Theorem 2 (flat weights and full support: GloVe is an SVD of double-centered PMI)
Let X > 0 entrywise, L = log X, and f ≡ 1. Then:
- min J = Σ_{i>d} σ_i², where the σ_i are the singular values of J_n L J_m = J_n PMI J_m;
- at every minimizer the score matrix is S = (J_n PMI J_m)_d + (L − J_n L J_m), unique when σ_d > σ_{d+1};
- one minimizer has WCᵀ = (J_n PMI J_m)_d, so the word vectors are Topic 6's Def 3, applied to the double-centered
  PMI matrix instead of PPMI, up to the gauge.

*Sketch.*
1. ℝ^{n×m} = 𝒟 ⊕ 𝒪 orthogonally, where 𝒟 is the doubly centered matrices, 𝒪 = {a𝟙ᵀ + 𝟙βᵀ}, and M ↦ J_nMJ_m is
   the projector onto 𝒟.
2. The biases range over all of 𝒪. By Pythagoras, J = ‖J_n(L − WCᵀ)J_m‖² + (an 𝒪 term the biases zero).
3. J_nWCᵀJ_m ranges over the rank-≤ d elements of 𝒟. Eckart–Young (formalML `svd`, Thm 2) then applies, and the
   best rank-d approximation of an element of 𝒟 lies in 𝒟.
4. L − PMI ∈ 𝒪.

*Reading.* Thm 2 is the full-information limit: with every cell observed and flat weights, GloVe sees the whole
association matrix, as Topic 6's SVD did. It is the two-sided-centered version of PCA (formalML `pca-low-rank`).
Prop 3 is the opposite limit.

**Asserted** (`test_flat_weight_closed_form`): on 10 random full-support instances (6–12 × 5–10, d ∈ {1, 2, 3}):
- L-BFGS on J from random starts and ALS (ridge 0) reach the closed form's loss to 10⁻⁸ relative;
- their score matrices match the closed form to 10⁻⁶;
- J_n L J_m = J_n PMI J_m to 10⁻¹².

### Proposition 3 (support blindness)
J depends on X only through the observed triples {(i, j, X_ij) : (i, j) ∈ Ω}. If log X is additive on Ω, that is
log X_ij = u_i + v_j for every (i, j) ∈ Ω (in particular, if every observed count is the same κ), then
(W, C, b, b̃) = (0, 0, u, v) attains J = 0, the global minimum, whatever Ω is. Any W, C with ⟨w_i, c_j⟩ = 0 on Ω
are also optimal. The vectors are not determined by which pairs co-occur.

*Sketch.* J ≥ 0, and direct substitution gives J = 0.

**Asserted** (`test_support_blindness`):
- random instances: two supports with the same additive values give J = 0 at the zero-vector solution;
- **corpus:** on the binarized L8 matrix, ALS converges to W = C = 0 (max |w·c| on Ω below 10⁻⁸). The benchmark is
  then exactly at chance (0.500).

**Corpus findings for §3** (`test_benchmark`, `test_scale_curve`, `test_zero_imputation`):
- **Sparsity.** 69% of observed L8 cells are seen once, and 80% at most twice (*spike*).
- **The controlled comparison** (AUC / 1-NN, dev and test, with the jackknife standard error of GloVe − SGNS and
  GloVe − SVD). *Spike:* GloVe W + C test AUC ≈ 0.48 (AdaGrad) and 0.51 (ALS), against 0.886 and 0.973. The page
  states the chance-level result with its standard error.
- **Scale** (the generator continued; Def 2 and the SVD at each size):
  - *Spike* for GloVe by ALS: 0.51 (300 docs), 0.59 (1,200), 0.68 (3,000), with singleton share 0.69, 0.59, 0.54;
  - SVD: 0.973, 0.985, 0.987.
  - Asserted shape: GloVe's AUC rises monotonically with corpus size, and the singleton share falls.
- **Imputed zeros** (Def 2) at 300 docs. To be measured. A cruder spike variant (target −1, weight 0.01) reached
  0.70. If Def 2 does not beat GloVe, the page says so.

### Definition 3 and the cross-check
- AdaGrad (Def 3) is the shipped algorithm.
- ALS is exact block coordinate descent. With (C, b̃) fixed, each row's (w_i, b_i) solves a weighted ridge
  regression in closed form (formalStatistics `linear-regression`, its normal equations), and symmetrically for
  the contexts. J + ridge is non-increasing.

**Asserted** (`test_glove_gradient`, `test_training`):
- the hand-written gradient against complex step and central differences (Topic 5's `complex_step_grad`,
  `central_diff_grad`) to 10⁻⁸;
- ALS's objective is non-increasing;
- AdaGrad's loss decreases across checkpoints;
- both runs' final losses are reported. *Spike:* ALS reaches a much lower loss than 50 AdaGrad epochs, so AdaGrad
  is not converged, and the page says so.

### Proposition 4 (GloVe's gauge is affine)
Write W̄ = [W, b, 𝟙] and C̄ = [C, 𝟙, b̃], so S = W̄C̄ᵀ.
- **(i)** For A ∈ GL(d), t, s ∈ ℝ^d, and β ∈ ℝ, the substitution below leaves every s_ij, and so J, unchanged:
  - w′_i = Aᵀw_i + t and c′_j = A⁻¹c_j + s (column-vector form);
  - b′_i = b_i − ⟨Aᵀw_i, s⟩ − ⟨t, s⟩ + β;
  - b̃′_j = b̃_j − ⟨A⁻¹c_j, t⟩ − β.
- **(ii)** Conversely, if two parameter sets give the same S and both augmented matrices have full column rank
  d + 2, they are related by such a substitution, uniquely.
- **(iii)** Cosines among word vectors are not determined by J. Differences w_i − w_k transform by Aᵀ alone, since
  translations cancel.

*Sketch.*
- (i) Expand ⟨w′_i, c′_j⟩.
- (ii) Topic 7's Prop 3 on (W̄, C̄) gives W̄′ = W̄G for a unique G ∈ GL(d + 2). The two pinned columns of ones
  force G's last column to be e_{d+2} and its (d + 1)-th row to be e_{d+1}ᵀ. That block form is (A, t, s, β).

**Asserted** (`test_affine_gauge`):
- random instances: a κ = 50 gauge with random t, s, β leaves S unchanged to 10⁻¹⁰ relative;
- (A, t, s, β) is recovered from the two parameter sets;
- word cosines move.

### Proposition 5 (ridge chooses the balanced, centered factorization)
Let J_λ = J + λ(‖W‖²_F + ‖C‖²_F) with λ > 0 and the biases unpenalized. At every stationary point of J_λ:
- WᵀW = CᵀC (balanced);
- 𝟙ᵀW = 𝟙ᵀC = 0 (centered).

Hence W = UΣ^½R for the SVD of the fitted WCᵀ (Topic 7 Lemma 1): the word vectors are Topic 6's p = ½ for the matrix
GloVe fitted, centered. Globally, min over factorizations of ½(‖W‖² + ‖C‖²) is the nuclear norm ‖WCᵀ‖_*, with
equality exactly at balanced factors (Srebro, Rennie, and Jaakkola 2005; formalML `convex-analysis`).

*Sketch.*
1. Let G = ∂J/∂S, which is 2f(X)(s − log X) on Ω and 0 off it.
2. Bias stationarity gives G𝟙 = 0 and Gᵀ𝟙 = 0.
3. ∇_W = GC + 2λW = 0 and ∇_C = GᵀW + 2λC = 0. Then WᵀGC = −2λWᵀW, and its transpose CᵀGᵀW = −2λCᵀC, give
   balance.
4. 𝟙ᵀ∇_W = 𝟙ᵀGC + 2λ𝟙ᵀW = 2λ𝟙ᵀW = 0 gives centering.

*Reading.* Unregularized GloVe (Def 1) has no such pin. Its trained gauge is whatever AdaGrad's path leaves, which
is Topic 7's lesson again.

**Asserted** (`test_ridge_balanced_centered`):
- random instances: ALS with ridge converges to imbalance and column means below 10⁻⁶;
- the nuclear-norm identity holds at the balanced factors;
- **corpus:** ridge ALS (λ = 10⁻³) at L8 is balanced and centered to tolerance, while AdaGrad's trained factors are
  not. Report the imbalance and ‖mean‖/mean ‖w‖ for each.

### Theorem 3 (exact parallelograms are facts about count ratios)
Let a, b, c, d be words, and K a set of contexts on which all four rows are observed and fitted exactly
(s_ij = log X_ij). Suppose [C_K, 𝟙] has full column rank d + 1. Then
w_b − w_a = w_d − w_c ⟺ Δ_j := PMI_bj − PMI_aj − PMI_dj + PMI_cj is constant over j ∈ K
⟺ P(j | b)/P(j | a) ∝ P(j | d)/P(j | c) on K.
The condition is invariant under Prop 4's gauge. For SGNS at its exact optimum (scores PMI_α − log k, no biases) the
same holds with "constant" replaced by "zero", given C_K of full column rank d.

*Sketch.*
1. Let v = w_b − w_a − w_d + w_c. Then ⟨v, c_j⟩ + (b_b − b_a − b_d + b_c) = Δ log X_j = Δ_j + (row terms), because
   the column offsets and log N cancel: the coefficients sum to zero.
2. If v = 0, Δ is constant.
3. Conversely, if Δ is constant, then [C_K, 𝟙](v; κ) = 0 for some κ, so v = 0 by the rank condition.

*Attribution.* Pennington et al. motivate GloVe by ratios. Ethayarajh, Duvenaud, and Hirst (2019) prove that
linear analogies correspond to ratio conditions for SGNS and GloVe at zero reconstruction error. Allen and
Hospedales (2019) derive the PMI-vector form with explicit error terms. Thm 3 is the elementary linear-algebra form
for the biased model, with the gauge clause added. Both papers are to be read in Phase A, and the statement aligned
with or credited to theirs.

**Asserted** (`test_exact_parallelogram`):
- random exact log-bilinear tables in which a planted ratio relation holds: v = 0 to 10⁻¹⁰, and Δ is constant;
- breaking the relation makes v ≠ 0;
- the criterion survives a random affine gauge;
- the SGNS variant on random exact PMI tables.

### Corollary 2 (3CosAdd and the gauge)
Def 4's answer is invariant under W ↦ DWQ, with D positive diagonal (per-word scaling) and Q orthogonal. It is
not invariant under a general A or a translation t, both of which leave J unchanged (Prop 4) and leave Thm 3's
exact parallelograms intact.

*Sketch.* Unit-normalizing undoes D, and Q preserves inner products. One small explicit instance shows that
translation changes the argmax.

**Asserted** (`test_analogy_gauge_dependence`):
- the invariances to 10⁻¹² on random instances;
- **corpus gauge table:** the same scores (to 10⁻¹⁰), different analogy accuracy. *Spike* for SGNS, city → region
  open: trained 0.633, p = 0 0.620, p = ½ 0.620, p = 1 0.573, random κ = 100 0.253, translated by 3 × mean 0.633.
  The same rows are planned for GloVe (W + C, centered, random A).

## 4. The finance case study (§6): a filings knowledge base as offsets

The knowledge base is `fc.CITIES` and `fc._PUBLIC`, as written by the generator's fact templates. Single-token
names only.
- **city → region** (LOCATED_IN, stated through the HQ template's "{city}, {region}"): 13 cities, 10 regions, 150
  questions with distinct regions. City counts are 3–26 and region counts 3–60.
- **company → sector** (IN_SECTOR): 24 companies by the first token of their short name, 6 sectors, 480 questions
  with distinct sectors. Company counts are 13–371 and sector counts 19–110.
- Dropped: CEO surnames (1–21 occurrences, and *halvorsen* is both a company and a surname) and company → city
  (several companies share a city).

`test_finance_findings`, per method (GloVe W + C, GloVe W, SGNS W and W + C, SVD(SPPMI₅) p = 0 and ½):
- open and within-category 3CosAdd accuracy, and the "b is nearest" rate;
- Thm 3's gauge-invariant **parallelogram defect**: sd_j(ΔS_j)/sd_j(S_bj − S_aj) on the fitted score rows S, which
  are WCᵀ, plus the biases for GloVe. This is zero iff v = 0, given the rank condition, and it is a property of the
  fit, not of the factorization;
- chance levels for within-category answers: 1/9 for regions (one excluded), 1/5 for sectors;
- *spike*: SGNS 0.633 / 0.375 open; GloVe (AdaGrad) 0.41 / 0.14. SVD rows are to be measured (the spike's zero
  rows broke its cosine).

Reading, to be confirmed by the numbers: region − city is close to a constant offset because both words fill fixed
template slots ("headquartered in {city}" against "based in {region}"), which is Thm 3's ratio condition holding
approximately. TransE (#24) trains w_h + r ≈ w_t directly instead of hoping a co-occurrence objective yields it.

## 5. `rigorFlag` (draft points)

1. GloVe's ratio derivation (Prop 1) rests on modeling choices: ratios as the statistic, a dot product of
   differences, and symmetry restored by hand. It is not a theorem about language.
2. Thm 1 makes GloVe the *quadratic* of a Poisson fit, with a tempered weight and the zero cells dropped. GloVe is
   not a likelihood, and its f is a heuristic choice.
3. GloVe's chance-level score on the planted-sector benchmark is a small-corpus effect. GloVe's published regime is
   billions of tokens. Our scale experiment continues the synthetic generator and drops tokens outside the
   300-document vocabulary.
4. Exact parallelograms need exact fits. Corpus analogies hold only approximately, the open-vocabulary protocol
   excludes the query words (Linzen 2016), and the answer depends on the factorization (Cor 2).
5. Fifty AdaGrad epochs are not converged. Minibatch summation departs from the C code's Hogwild updates. The counts
   are Topic 6's unweighted windows. One seed.
6. The knowledge base is synthetic, with 13 cities and 24 companies. Question sets are small and share words.
7. Def 2 is our parameter-free variant, not Swivel.

## 6. Viz design intent — `GloVeLab` (one component, two linked panels)

**Data.** All data comes from `viz_payload()`, mirrored in `src/components/viz/data/glove-lab.json`.
- `test_viz_payload` caps it at 40 kB, and `test_viz_mirror_matches_payload` checks the mirror.
- Every displayed number is rounded inside `viz_payload()` to the page's precision.
- Marginalia `--color-*` tokens only.

**Panel A, "Three weights, one count."**
- *Revenue*'s L8 row (Topic 7's `LAB_ROW`): one context per count band, plus the most frequent context it never
  meets.
- **View:** the per-cell loss against the score, centered at each objective's optimum. There are three curves:
  GloVe f(a)(s − log a)², Poisson a(e^δ − 1 − δ), and SGNS's cell loss in its own coordinate.
- **Controls:** x_max ∈ {10, 100} and γ ∈ {½, ¾, 1}.
- **Readouts:** the three weights, and bars of each weight's mass by count band, recomputed in the browser from a
  count histogram in the payload. SGNS's shares are precomputed at k = 5 and α = ¾.
- For the unobserved context, GloVe's curve is flat ("no term: the score is free"), and Poisson and SGNS slope
  downward.
- *Lessons:* curvature is weight; GloVe caps its weight at x_max; the zero cell is invisible to GloVe.

**Panel B, "What the vectors know."** Two views behind a toggle.
- *Analogies:*
  - controls: a relation, a method (GloVe, SGNS), and a factorization (as trained, p = ½, random κ = 100,
    translated);
  - a 2-D projection of the relation's words under that factorization, with offset arrows;
  - precomputed accuracies, "b is nearest" rates, and top-3 answers for a short list of curated questions;
  - a fixed readout: "same scores, same loss."
  - *Lesson:* a random gauge visibly breaks the parallelograms while the loss is unchanged.
- *Scale:* test AUC against corpus size (300, 1,200, 3,000 documents) for GloVe, Def 2, and SVD, with the singleton
  share. *Lesson:* GloVe recovers as counts stop being binary.
- **Link:** choosing GloVe in Panel B highlights its curve in Panel A.

**Mobile:** follow Topics 6–7. Rows wrap, inputs get `minWidth: 0`, and `scrollWidth === 375` must hold in every
view and toggle.

## 7. Function signatures (`glove_embedding_geometry.py`)

Bodies are written after Jonathan's reaction. The module imports Topic 7 as `w2v` and Topic 6 as `ds`, plus `fc`,
`ng` (for `split_sentences`), and `nlm` (complex-step checkers). It reimplements none of their functions.

```python
SEED = 20261023          # streams default_rng([SEED, k]) with k ≥ 800, apart from Topics 6–7
X_MAX, GAMMA = 100.0, 0.75
CONTEXT, D = "L8", 25    # Topic 7's controlled matrix and rank
SCALE_DOCS = (300, 1200, 3000)

# §1 the model
def glove_weight(x, x_max: float = X_MAX, gamma: float = GAMMA) -> np.ndarray
def glove_loss(W, Cv, b, bt, X: sparse.spmatrix, x_max=X_MAX, gamma=GAMMA) -> float          # Def 1, Ω only
def glove_loss_grad(W, Cv, b, bt, X, x_max=X_MAX, gamma=GAMMA) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def ratio_scores(P_cond: np.ndarray, beta: np.ndarray) -> np.ndarray                        # Prop 1's model
def pmi_offsets(W, Cv, b, bt, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]               # Prop 2's a, β

# §2 three weights
def poisson_excess(s, x) -> np.ndarray                     # ℓ(s; x) − ℓ(log x; x); e^s for x = 0
def bregman_exp(s, s_star) -> np.ndarray                   # D_A with A = exp
def frozen_bias_residual(W, Cv, X: np.ndarray, alpha: float, k: float) -> np.ndarray   # Cor 1: SGNS as GloVe
def weight_shares(rows: np.ndarray | None = None) -> dict[str, float]                   # Cor 1 corpus, by band

# §3 what the loss cannot see
def double_center(M: np.ndarray) -> np.ndarray
def flat_weight_optimum(X: np.ndarray, d: int) -> dict[str, np.ndarray]                # Thm 2: S, WCᵀ, loss
def binarized(X: sparse.spmatrix) -> sparse.csr_matrix                                  # Prop 3's corpus check
def impute_zeros(X: sparse.spmatrix, x0: float = 0.5) -> tuple[float, float]            # Def 2: (weight, target)

# §4 training
@dataclass(frozen=True)
class GloVeConfig:
    context: str = CONTEXT; d: int = D; x_max: float = X_MAX; gamma: float = GAMMA
    epochs: int = 50; batch: int = 512; lr0: float = 0.05; docs: int = 300; checkpoints: tuple[int, ...] = ()

@dataclass(frozen=True)
class GloVeModel:
    W: np.ndarray; C: np.ndarray; b: np.ndarray; bt: np.ndarray; loss: float; history: tuple[dict, ...]

def counts_for(context: str, docs: int = 300) -> sparse.csr_matrix                     # ds.cooc, or continued text
def continued_sentences(docs: int) -> tuple[tuple[str, ...], ...]                       # fc._CorpusGen run on
def train_glove(cfg: GloVeConfig) -> GloVeModel                                         # Def 3, cached
def als_glove(X: sparse.spmatrix, d: int, *, x_max=X_MAX, gamma=GAMMA, iters: int = 40, ridge: float = 1e-3,
              zeros: tuple[float, float] | None = None, seed: int = 801) -> GloVeModel   # cross-check; Def 2 via zeros
def glove_vectors(m: GloVeModel, rep: str = "W+C") -> np.ndarray

# §5 gauge and analogies
def affine_gauge(W, Cv, b, bt, A, t, s, beta) -> tuple[np.ndarray, ...]                 # Prop 4 (i)
def recover_affine_gauge(params_1, params_2) -> tuple[np.ndarray, ...]                  # Prop 4 (ii)
def score_rows(m: GloVeModel | tuple[np.ndarray, np.ndarray], rows) -> np.ndarray       # S restricted to rows
def balance_and_centering(W, Cv) -> dict[str, float]                                    # Prop 5 diagnostics
def three_cos_add(X: np.ndarray, a: int, b: int, c: int, candidates=None, exclude: bool = True) -> int
def kb_relations() -> dict[str, tuple[tuple[str, str], ...]]                            # city→region, company→sector
def analogy_scores(X: np.ndarray, relation: str) -> dict[str, float]                    # open, within, b_nearest, n
def parallelogram_defect(S_rows: np.ndarray) -> float                                   # Thm 3, gauge-free
def gauge_table() -> dict                                                               # Cor 2 corpus

# summaries, lab, harness
def benchmark_summary() -> dict          # controlled GloVe vs SGNS vs SVD, with jackknife/paired se (w2v._paired)
def scale_summary() -> dict
def finance_summary() -> dict
def viz_payload() -> dict
def summary() -> dict[str, float]
# tests: test_ratio_argument, test_log_bilinear_rank, test_poisson_bregman, test_three_weights,
#   test_flat_weight_closed_form, test_support_blindness, test_glove_gradient, test_training, test_benchmark,
#   test_scale_curve, test_zero_imputation, test_affine_gauge, test_ridge_balanced_centered,
#   test_exact_parallelogram, test_analogy_gauge_dependence, test_finance_findings, test_viz_payload,
#   test_viz_mirror_matches_payload (Phase B)
```

**Budget.**
- *Spike timings:* SGNS controlled 13.9 s, controlled GloVe ≈ 3 s, ALS ≈ 2 s per fit, continued generation 1.6 s,
  and counts and SVDs ≈ 1 s each.
- The estimate is ≈ 45 s for the harness.
- If AdaGrad at 3,000 documents (nnz ≈ 431k) pushes the harness past ~50 s, the scale run trains by ALS, and the
  page says so.

## 8. Notebook cell map (`01_glove_embedding_geometry.ipynb`, ≈ 20 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; the L8 counts and their sparsity | — |
| 2–3 | §1 Def 1, Prop 1, Prop 2 | `test_ratio_argument`, `test_log_bilinear_rank`, `test_glove_gradient` |
| 4–5 | §2 Thm 1, Cor 1; weight shares | `test_poisson_bregman`, `test_three_weights` |
| 6–9 | §3 Thm 2, Prop 3, Def 2; benchmark, scale, imputation | `test_flat_weight_closed_form`, `test_support_blindness`, `test_benchmark`, `test_scale_curve`, `test_zero_imputation` |
| 10–11 | §4 Def 3; AdaGrad against ALS | `test_training` |
| 12–14 | §5 Prop 4, Prop 5, Def 4, Thm 3, Cor 2; gauge table | `test_affine_gauge`, `test_ridge_balanced_centered`, `test_exact_parallelogram`, `test_analogy_gauge_dependence` |
| 15–16 | §6 finance | `test_finance_findings` |
| 17–19 | lab payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

## 9. Cross-site candidates (to be verified `published` on GitHub `main` in Phase A)

All of these are `status: "published"` in the local sibling checkouts as of 2026-10-23. GitHub `main` is the
authority.

| field | topic | role |
|---|---|---|
| `formalmlPrereqs` | `svd` | Eckart–Young in Thm 2 |
| `formalmlPrereqs` | `information-geometry` | Bregman divergence of the Poisson log-partition (Thm 1) |
| `formalmlPrereqs` | `pca-low-rank` | Thm 2 as two-sided-centered PCA |
| `formalmlConnections` | `convex-analysis` | the nuclear norm's variational form (Prop 5) |
| `formalmlConnections` | `gradient-descent` | adaptive steps; AdaGrad (check whether the page treats it) |
| `formalmlConnections` | `representation-learning` | embeddings as learned representations; analogies |
| `formalragConnections` | `hypersphere-vmf-geometry` | cosine geometry and anisotropy of embedding clouds |
| `formalragConnections` | `dense-retrieval-dual-encoders` | inner-product scores share the gauge freedom |
| `formalstatisticsPrereqs` | `generalized-linear-models` | the Poisson log link; IRLS = Fisher scoring (Thm 1's reading) |
| `formalstatisticsPrereqs` | `linear-regression` | weighted least squares and the normal equations (ALS) |
| `formalstatisticsConnections` | `regularization-and-penalized-estimation` | ridge, which pins the gauge (Prop 5) |
| `formalstatisticsConnections` | `exponential-families` | the Poisson log-partition |
| `formalcalculusPrereqs` | `mean-value-taylor` | the Lagrange remainder in Thm 1's sandwich |

## 10. References (Chicago 17th; every URL and title verified in Phase A)

1. Pennington, Socher, and Manning, "GloVe: Global Vectors for Word Representation," EMNLP 2014.
   https://aclanthology.org/D14-1162/ — Def 1, Prop 1, Thm 1's reading.
2. Mikolov, Yih, and Zweig, "Linguistic Regularities in Continuous Space Word Representations," NAACL 2013.
   https://aclanthology.org/N13-1090/ — Def 4.
3. Levy and Goldberg, "Linguistic Regularities in Sparse and Explicit Word Representations," CoNLL 2014.
   https://aclanthology.org/W14-1618/ — 3CosMul.
4. Levy, Goldberg, and Dagan, "Improving Distributional Similarity with Lessons Learned from Word Embeddings,"
   TACL 3 (2015). https://doi.org/10.1162/tacl_a_00134 — GloVe against SGNS, W + C.
5. Shi and Liu, "Linking GloVe with word2vec," arXiv:1411.5595 (2014) — Cor 1 (title verified).
6. Arora, Li, Liang, Ma, and Risteski, "A Latent Variable Model Approach to PMI-based Word Embeddings," TACL 4
   (2016). https://doi.org/10.1162/tacl_a_00106 — analogies under a generative model (pointer).
7. Allen and Hospedales, "Analogies Explained: Towards Understanding Word Embeddings," ICML 2019.
   arXiv:1901.09813 — Thm 3 (title verified).
8. Ethayarajh, Duvenaud, and Hirst, "Towards Understanding Linear Word Analogies," ACL 2019. arXiv:1810.04882 —
   Thm 3 (title verified).
9. Linzen, "Issues in Evaluating Semantic Spaces Using Word Analogies," RepEval 2016.
   https://aclanthology.org/W16-2503/ — the exclusion caveat.
10. Rogers, Drozd, and Li, "The (Too Many) Problems of Analogical Reasoning with Word Vectors," *SEM 2017.
    https://aclanthology.org/S17-1017/ — analogy evaluation.
11. Duchi, Hazan, and Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization," JMLR
    12 (2011) — Def 3.
12. Hu, Koren, and Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM 2008 — Def 2's ALS
    identity.
13. Shazeer, Doherty, Evans, and Waterson, "Swivel: Improving Embeddings by Noticing What's Missing,"
    arXiv:1602.02215 (2016) — Def 2's published counterpart (title verified).
14. Srebro and Jaakkola, "Weighted Low-Rank Approximations," ICML 2003 — weighted low rank has no closed form.
15. Srebro, Rennie, and Jaakkola, "Maximum-Margin Matrix Factorization," NeurIPS 17 (2005) — Prop 5.
16. Collins, Dasgupta, and Schapire, "A Generalization of Principal Components Analysis to the Exponential
    Family," NeurIPS 14 (2001) — Thm 1.
17. Cotterell, Poliak, Van Durme, and Eisner, "Explaining and Generalizing Skip-Gram through Exponential Family
    Principal Component Analysis," EACL 2017 — Thm 1.
18. McCullagh and Nelder, *Generalized Linear Models*, 2nd ed. (1989) — Thm 1's IRLS reading.
19. Aczél, *Lectures on Functional Equations and Their Applications* (1966) — Prop 1.
20. Eckart and Young (1936) — Thm 2.
21. Bordes et al., "Translating Embeddings for Modeling Multi-relational Data," NeurIPS 2013 — pointer to #24.

## 11. Open points for Jonathan's reaction

1. **Primary vectors.** W + C, GloVe's own convention, is the plan, with W also scored.
2. **Def 2's form.** Half-count imputation (x₀ = ½, weight f(½)) is parameter-free. If it does not recover signal,
   should the page show the crude variant's 0.70, which was tuned by eye, or report Def 2 as it falls?
   Recommendation: report Def 2 as it falls, and mention Swivel.
3. **Panel B's scale view**, or a single analogy view with the scale curve moved to a static figure in §3.
4. **Authorship.** Claude writes every body, as in Topics 5–7. If you would rather write one yourself, the natural
   candidates are `parallelogram_defect` (a design choice about what "approximately a parallelogram" means) or
   `impute_zeros`.
