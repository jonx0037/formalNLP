# Brief — `glove-embedding-geometry`

**GloVe and the Geometry of Word Embeddings.** Roadmap #8. Prerequisite: `word2vec-implicit-factorization` (#7,
PR #13), published. This is the third `representations` topic and a prerequisite of `knowledge-graph-embeddings`
(#24), which makes "a relation is a vector offset" its training objective.

**Phase A** (this brief and the verified notebook), 2026-10-23:
- The harness passes 18/18 under Python 3.15.0, numpy==2.5.4, and scipy==1.18.1, in ≈ 46 s.
- The narrative notebook (21 cells) executes with exit 0 in ≈ 51 s, including environment and kernel start, and is
  committed without outputs.

**Scope decisions** (Jonathan, 2026-10-23):
- **In:** every optional result — the flat-weight theorem, the scale experiment and the zeros fix, the ratio
  argument, and ridge fixing the gauge.
- **Checks:** internal only — Topic 5's complex step, closed forms against L-BFGS on small instances, numpy's SVD,
  `scipy.stats.poisson`, and ALS against AdaGrad.
- **Benchmark:** controlled, with GloVe's published settings — Topic 6's dev/test nouns, the same L8 counts,
  d = 25, x_max = 100, γ = ¾, 50 AdaGrad epochs, against Topic 7's 40-epoch SGNS and SVD(SPPMI₅). No dev grid.
- **Finance:** relations from the filings knowledge base as vector offsets.
- **Difficulty:** `intermediate`.
- **§11 of the scaffold, resolved on Claude's recommendations:**
  - W + C is the primary vector;
  - Def 2 is reported as it falls (it works);
  - the scale curve gets its own view in the lab;
  - every function body was written by Claude for Jonathan's review, as in Topics 5–7.

**Changes from the scaffold, each forced by the harness:**
1. **Weight shares.** GloVe puts the *most* weight on rare pairs (72% on the benchmark rows), then Poisson (59%),
   then SGNS (55%). The scaffold guessed GloVe below SGNS. GloVe above Poisson is a theorem, since f(X)/X decreases;
   SGNS below Poisson is a corpus fact.
2. **The parallelogram defect** is ≈ 1 for every model (0.80–1.26 median). No fit is near an exact parallelogram,
   and the defect alone does not rank the models. The scaffold's assert "GloVe's defect exceeds SGNS's" fails on
   company → sector and was dropped. Added in its place: the **gauge-free solver**, Thm 3 turned into an analogy
   method. It reads the scores only, so every factorization gives it the same answers, and the right tail beats the
   wrong ones for every model.
3. **Prop 5 on the corpus.** Forty ALS sweeps at λ = 10⁻³ are far from a stationary point in the gauge directions
   (imbalance 0.999). Prop 5 is asserted on small instances only, and the page says the trained models do not sit in
   the ridge's gauge.
4. **The scale curve is trained by ALS.** AdaGrad at 3,000 documents cost ≈ 15 s of a 60 s budget. The controlled
   comparison stays AdaGrad.
5. **Def 2 works:** 0.602, 0.855, and 0.946 test AUC at 300, 1,200, and 3,000 documents, against plain GloVe's
   0.507, 0.583, and 0.695.
6. **SGNS's gauge table has no translation.** Without biases a translation changes SGNS's scores, so only GloVe's
   biases absorb one.
7. **Attribution** (papers read): Thm 3 is the exact case of Allen and Hospedales's generalized paraphrase theorem,
   adapted to biases. Cor 1 extends Shi and Liu's optimum-level observation to the local loss. Prop 1 cites Aczél
   and Dhombres (1989), because Aczél (1966) has no DOI or stable URL.

**Source of truth:** `notebooks/glove-embedding-geometry/glove_embedding_geometry.py`. Every number below is printed
or asserted by it, and none may change without it.

**Binding forward promises, and where they are kept:**
- **Topic 6, §"Where this goes next":** "GloVe fits a log-bilinear model to log co-occurrence counts, the form of
  Theorem 3's third part." Kept by Prop 2.
- **Topic 7, §"Where this goes next":** GloVe fits log #(w, c) by weighted least squares with explicit biases and a
  count-increasing weight, compared with SGNS's ab/(a + b). Analogies are word–word geometry and so, by its Prop 3,
  about the gauge.
  - Thm 1 and Cor 1 compare f(X), ab/(a + b), and the Poisson weight X that both temper.
  - Prop 4 widens the gauge to affine maps.
  - Thm 3 and Cor 2 split analogies into the gauge-free part (count ratios) and the nearest-neighbor answer, which
    depends on the gauge.
  - The rigorFlag treats analogy arithmetic as approximate.

**Thesis.** GloVe is a tempered, support-restricted Poisson log-bilinear fit.
- **The weight** decides how far each count is trusted (§2).
- **The support restriction** decides what the vectors cannot see. On this corpus, where most observed pairs are
  seen once, that is the planted structure itself (§3).
- **The biases** widen the gauge to affine maps, so the objective fixes the exact parallelograms (facts about count
  ratios) and not the answers a nearest-neighbor search returns (§5).

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | GloVe and the Geometry of Word Embeddings |
| `{{SUBTITLE}}` | GloVe fits log co-occurrence counts by a tempered Poisson least squares that never sees the pairs that did not occur; its biases widen the gauge to affine maps, so an exact analogy is a fact about count ratios while the answer a nearest-neighbor search returns depends on the factorization. |
| `{{SLUG}}` | `glove-embedding-geometry` (= graph node id) |
| `{{DOMAIN}}` | `representations` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[word2vec-implicit-factorization]` |
| `{{FORMALML_PREREQS}}` | `svd`, `information-geometry`, `pca-low-rank`; connections `convex-analysis`, `gradient-descent`, `representation-learning` |
| `{{FORMALRAG_PREREQS}}` | none; connections `hypersphere-vmf-geometry`, `dense-retrieval-dual-encoders` |
| `{{STATS_CALC_PREREQS}}` | Stats `generalized-linear-models`, `linear-regression`; connections `regularization-and-penalized-estimation`, `exponential-families`; Calc `mean-value-taylor` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/word2vec-implicit-factorization/`, `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/finance-corpus/` |

- **Tags:** `glove`, `word-embeddings`, `weighted-least-squares`, `poisson-regression`, `log-bilinear-model`,
  `word-analogies`, `gauge-freedom`, `matrix-factorization`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`:** none, since #24 is still planned.
- **Graph:** both edges already exist.

**Scope.** This topic is not on the design doc's §6 watch-list, but it adjoins:
- **formalRAG `hypersphere-vmf-geometry`**, which owns cosine geometry on the sphere: its Prop 1, the vMF
  distribution, and anisotropy. We use cosine as defined there and do not re-treat anisotropy. The isotropy
  post-processing literature is out of scope.
- **formalRAG `dense-retrieval-dual-encoders`**: inner-product scores and the rank ceiling (its Thm 2), cited for
  the gauge reading of two-tower scores.
- **formalML `svd`** (Eckart–Young, its Thm 2) and **`pca-low-rank`** (its Thm 1, PCA via SVD). Thm 2 cites both and
  proves only the double-centering step.
- **formalStatistics `generalized-linear-models`** (its Thm 2, IRLS = Fisher scoring), cited in Thm 1's reading. The
  Poisson GLM is not re-derived.
- **This topic owns:**
  - the GloVe objective and its ratio derivation;
  - GloVe as a tempered, support-restricted Poisson fit;
  - the flat-weight theorem;
  - support blindness and the imputed-zeros variant;
  - the affine gauge and ridge's choice within it;
  - exact parallelograms, their count-ratio criterion, and the gauge-free solver;
  - analogy evaluation and its gauge dependence.
- **Out of scope apart from pointers:**
  - contextual embeddings (#12);
  - TransE (#24), which §6 previews;
  - isotropy post-processing;
  - fastText;
  - debiasing;
  - retrieval and ANN, which stay in formalRAG.

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | the same L8 counts: GloVe 0.476, SGNS 0.886, SVD 0.973 test AUC; and "same loss, different analogies" (SGNS city → region 0.633 → 0.253 under a random basis change) | — | cells 0–1 |
| 1 | A log-bilinear model of counts | 69% of observed pairs seen once; 178,100 parameters against 143,345 observed cells | Def 1; **Prop 1** (ratio argument); **Prop 2** (rank ≤ d + 2) | cells 2–3 |
| 2 | Three weights | *revenue*'s five cells, from *plantline* (seen once) to *cost* (129) | **Thm 1** (tempered Poisson); **Cor 1** (three local weights); weight shares by band | cells 4–5 |
| 3 | What the loss cannot see | set every count to e: the biases fit it exactly and the vectors are zero | **Thm 2** (flat weights + full support); **Prop 3** (support blindness); Def 2 (imputed zeros); benchmark, additive share, scale curve | cells 6–11 |
| 4 | The algorithm | — | Def 3 (GloVe's AdaGrad, as run); ALS as exact block descent | cells 12–13 |
| 5 | Gauge and analogies | one set of scores, five factorizations, five analogy accuracies | **Prop 4** (affine gauge); **Prop 5** (ridge: balanced and centered); Def 4 (3CosAdd); **Thm 3** (exact parallelograms); **Cor 2** (3CosAdd and the gauge); the gauge-free solver | cells 14–15 |
| — | The lab | `<GloVeLab client:visible />` | — | cells 18–19 |
| 6 | Finance case study: a filings knowledge base as offsets | *Houston : Texas :: Boston : ?* — GloVe answers *massachusetts*, SGNS *ireland* | Def 4 and the gauge-free solver on two KB relations | cells 16–17 |
| 7 | Where this goes next | — | #12, #24 (TransE: w_h + r ≈ w_t as the objective), formalRAG | — |

- **Lab placement.** One `<GloVeLab client:visible />` after §5. Panel A needs §2, Panel B needs §§3 and 5.
- **Back-links:**
  - Topic 6's Thm 3 (iii) from Prop 2, and its Def 3 from Thm 2;
  - Topic 7's Thm 2 and Cor 1 from Thm 1 and Cor 1;
  - Topic 7's Prop 3 from Prop 4;
  - Topic 7's Lemma 1 and Cor 2 from Prop 5;
  - Topic 7's §5 numbers (0.886, the 54.8% mechanism share) from §§2–3.

## 2. Definitions

**Notation.**
- X is Topic 6's L8 count matrix #(w, c), symmetric, V = 3,425. Ω = {(i, j) : X_ij > 0}: 143,345 cells, 1.22%
  of V².
- Scores are s_ij = ⟨w_i, c_j⟩ + b_i + b̃_j.
- PMI is Topic 6's Def 2 (α = 1 unless stated).
- J_n = I − 𝟙𝟙ᵀ/n.
- γ names GloVe's weight exponent, because α is Topic 6's context-smoothing exponent. The page says so once.

**Definitions.**
- **Def 1 (GloVe; Pennington, Socher, and Manning 2014).**
  J(W, C, b, b̃) = Σ_{(i,j)∈Ω} f(X_ij)(s_ij − log X_ij)², with f(x) = min{1, (x/x_max)^γ}, x_max = 100, γ = ¾.
  The reported vectors are W + C. W is also scored.
- **Def 2 (GloVe with imputed zeros).** X̃ = X on Ω and ½ off it; minimize J on X̃ over all V² cells. Every unseen
  pair counts as seen half a time, with GloVe's own weight f(½) = 0.0188.
  - It is parameter-free and ours. Swivel (Shazeer et al. 2016) is the published model that uses the unobserved
    cells, with a different loss.
  - It is trained by ALS, using Σ_j ω_ij c̄_jc̄_jᵀ = w₀C̄ᵀC̄ + Σ_{j∈Ω_i}(f_ij − w₀)c̄_jc̄_jᵀ (Hu, Koren, and Volinsky
    2008), so no V × V matrix is formed.
- **Def 3 (GloVe's AdaGrad, as run).**
  - Each epoch is a fresh permutation of Ω in minibatches of 512. Each cell takes the C code's step on
    ½f(s − log x)²: fdiff = ηf(s − log x), and each update is fdiff × direction / √gradsq, with gradsq starting at 1
    and accumulating the squared updates (Duchi, Hazan, and Singer 2011).
  - η = 0.05. Initialization (U[0, 1] − ½)/d for every vector entry and bias. 50 epochs. Float64. One stream.
  - **Departures, stated on the page:** minibatch summation replaces the C code's Hogwild updates, and the counts are
    Topic 6's unweighted windows, not GloVe's 1/distance weighting, so that all three methods factor one matrix.
- **Def 4 (3CosAdd; Mikolov, Yih, and Zweig 2013).** a : b :: c : ? is answered by
  argmax_{x∉{a,b,c}} cos(x̂, b̂ − â + ĉ).
  - "Open" means the whole vocabulary; "within" restricts the candidates to the relation's tails.
  - "b nearest" is the share of questions whose top candidate before the exclusion is b itself (Linzen 2016).
- **The gauge-free solver** (unnumbered; §5 and §6). Answer with the tail d′ ≠ b that minimizes
  sd_j(s_bj − s_aj − s_d′j + s_cj) over every context j. It reads the scores only.

**Configurations.**
- **Controlled:** L8, d = 25, Def 1's defaults, 50 AdaGrad epochs.
- **Cross-check:** ALS with ridge λ = 10⁻³ on W and C (biases free), 40 sweeps.
- **Scale:** the generator continued to 1,200 and 3,000 documents. Shorter corpora are prefixes, so each is the same
  text continued. Counts are on Topic 6's vocabulary (new types dropped, so a window closes over them). Both GloVe
  fits are by ALS.

## 3. Statements and proofs

### Proposition 1 (the ratio argument, with its hypotheses)
Let F: ℝ → (0, ∞) be continuous with F(u − v) = F(u)/F(v) for all u, v. Then F(u) = e^{λu} for some λ ∈ ℝ. If in
addition F(⟨w_i − w_k, c_j⟩) = P(j | i)/P(j | k) for all words i, k and contexts j, and λ ≠ 0, then
⟨w_i, c_j⟩ = λ⁻¹(log X_ij − log X_i + β_j) for constants β_j.

*Proof.*
1. Let G = log F. Then G(u − v) = G(u) − G(v). With u = v, G(0) = 0. With u = 0, G(−v) = −G(v). So
   G(u + v) = G(u − (−v)) = G(u) + G(v), which is Cauchy's equation.
2. By induction G(nu) = nG(u), so G(p/q) = (p/q)G(1) for rationals. Continuity extends this to G(u) = λu with
   λ = G(1) (Aczél and Dhombres 1989).
3. The ratio identity gives F(⟨w_i, c_j⟩)/P(j | i) = F(⟨w_k, c_j⟩)/P(j | k) for all i, k, so it equals some g_j > 0.
   Hence λ⟨w_i, c_j⟩ = log P(j | i) + log g_j, with P(j | i) = X_ij/X_i. ∎

*Reading.* Absorbing λ into W and replacing the word term −log X_i with a free b_i, which restores the word–context
symmetry the model lacks, gives Def 1's s_ij. The hypotheses are modeling choices: ratios as the statistic, a
function of a dot product of differences, and symmetry restored by hand. They are not facts about data.

**Asserted** (`test_ratio_argument`):
- exp(λu) has a Cauchy defect below 10⁻⁹ × e^{6|λ|} on a grid, while 1 + u², eᵘ + 0.1, and the logistic function
  exceed 10⁻²;
- vectors factoring log P(j | i) + β_j satisfy exp⟨w_i − w_k, c_j⟩ = P(j | i)/P(j | k) to 10⁻¹⁰ relative on random
  joints;
- Thm 2's closed form at full rank gives J = 0 on a full-support table.

### Proposition 2 (log-bilinear form; Topic 6 Thm 3 iii)
If s_ij = log X_ij on every cell, then PMI = WCᵀ + a𝟙ᵀ + 𝟙βᵀ with a_i = b_i − log X_i· + log N and
β_j = b̃_j − log X_·j. So rank PMI ≤ d + 2, and α ≠ 1 moves only the column offset.

*Proof.* PMI_ij = log X_ij − log X_i· − log X_·j + log N. Substitute s_ij, then use subadditivity of rank. For
PMI_α, log P_α(c) replaces log(X_·j/N). ∎

**Asserted** (`test_log_bilinear_rank`): on random positive tables exp(WCᵀ + b𝟙ᵀ + 𝟙b̃ᵀ) with d ∈ {1, 2, 4}:
- the offsets hold to 10⁻¹⁰;
- rank PMI ≤ d + 2 < rank exp(PMI);
- PMI_α − PMI is a column offset to 10⁻¹⁰.

### Theorem 1 (GloVe is a tempered, support-restricted Poisson fit)
Let X_ij ~ Poisson(e^{s_ij}) independently, with ℓ(s; x) = e^s − xs + log x!.
- **(i)** For x > 0, ℓ has the unique minimizer s* = log x. With δ = s − log x,
  ℓ(s; x) − ℓ(s*; x) = x(e^δ − 1 − δ) = D_A(s, s*), the Bregman divergence of A(s) = e^s. Its curvature at s* is x,
  and ½xδ²e^{min(δ,0)} ≤ x(e^δ − 1 − δ) ≤ ½xδ²e^{max(δ,0)}.
- **(ii)** For x = 0, ℓ(s; 0) = e^s decreases strictly toward 0 as s → −∞, and has no minimizer.
- **(iii)** Hence ½Σ_Ω X(s − log X)² is the second-order Poisson fit on the support. Def 1 is that sum with the
  Fisher weight X replaced by f(X), and with the zero cells' Σ_{Ω^c} e^s dropped.

*Proof.*
1. ℓ′ = e^s − x and ℓ″ = e^s > 0, so s* = log x is the unique minimizer, with curvature x.
2. With e^s = xe^δ, ℓ(s) − ℓ(s*) = x(e^δ − 1 − δ). This is A(s) − A(s*) − A′(s*)(s − s*), since A′(s*) = x.
3. Taylor's theorem with the Lagrange remainder (formalCalculus `mean-value-taylor`) gives e^δ − 1 − δ = ½δ²e^ξ,
   with ξ between 0 and δ. That gives the sandwich.
4. (ii) is immediate. ∎

*Reading.* One Fisher-scoring (IRLS) step of the Poisson GLM at the saturated fit has working response log X and
weight X (formalStatistics `generalized-linear-models`, Thm 2). Pennington et al. (2014, §3.1) reach weighted least
squares on logs from skip-gram's global cross-entropy and replace the weight X_i by f(X_ij). Thm 1 names the
likelihood this is the quadratic of. It is Collins, Dasgupta, and Schapire's (2001) exponential-family PCA with the
Poisson family (cf. Cotterell et al. 2017 for skip-gram). Its zero cells keep a push toward −∞ that GloVe drops.

**Asserted** (`test_poisson_bregman`):
- (i) the stable form x(expm1 δ − δ) equals the term-by-term excess and D_A to 10⁻¹⁴ of their rounding scale, for
  x ∈ {0.5, 1, 3, 40, 250} and δ ∈ [−3, 3];
- the sandwich holds;
- the numerical curvature is x to 10⁻⁵;
- the full NLL matches `scipy.stats.poisson.logpmf`;
- (ii) strict decrease for x = 0;
- (iii) |exact − ½ΣXδ²| ≤ ΣX|δ|³ on random instances.

### Corollary 1 (three local weights)

| objective | local loss | weight | unobserved cell |
|---|---|---|---|
| Poisson | ½X(s − log X)² | X | e^s, pushed toward −∞ |
| SGNS (Topic 7 Thm 2, Cor 1) | ½ ab/(a + b) (w·c + log k#(w) + log P_α(c) − log a)² | ab/(a + b) ≤ min(a, b) | b log(1 + e^{w·c}), pushed toward −∞ |
| GloVe | f(X)(s − log X)² | f(X) ≤ 1 | no term |

- SGNS's local loss is GloVe's residual with the biases **frozen** at b_w = log k#(w) and b̃_c = log P_α(c). Shi and
  Liu (2014) observe this at the optimum; here it holds for the local loss, weight included.
- f(X)/X = X^{γ−1}x_max^{−γ} decreases for X ≤ x_max. So on every corpus GloVe's share of weight on rare pairs
  exceeds Poisson's.

*Proof.* Thm 1 (i), and Topic 7's x* = log a − log b with b = k#(w)P_α(c). If w₁/w₂ decreases in X, then w₁'s share
on {X ≤ t} exceeds w₂'s for every t. ∎

**Asserted** (`test_three_weights`):
- the frozen-bias identity to 10⁻¹²;
- the SGNS cell loss equals ½ab/(a + b)δ² to third order;
- the weight inequalities, and f(X)/X decreasing;
- **corpus**, the share of each weight on pairs seen ≤ 4 times:

| rows | cells | GloVe f(X) | Poisson X | SGNS ab/(a + b) |
|---|---|---|---|---|
| benchmark nouns (Topic 7's §5) | 0.923 | **0.721** | 0.585 | 0.548 (= Topic 7's 54.8%) |
| all of Ω | 0.885 | 0.523 | 0.296 | 0.300 |

  GloVe gives its weight to exactly the cells whose values (mostly log 1 = 0) carry the least.

### Theorem 2 (flat weights and full support: GloVe is an SVD of double-centered PMI)
Let X > 0 entrywise, L = log X, and f ≡ 1. Then:
- min J = Σ_{i>d} σ_i², where the σ_i are the singular values of J_n L J_m = J_n PMI J_m;
- at every minimizer, S = (J_n PMI J_m)_d + (L − J_n L J_m), unique when σ_d > σ_{d+1};
- one minimizer is W = U_dΣ_d^{½}, C = V_dΣ_d^{½}, b = row means of L − its grand mean, and b̃ = column means of L.

*Proof.*
1. Let 𝒟 = {M : M𝟙 = 0, 𝟙ᵀM = 0} and 𝒪 = {a𝟙ᵀ + 𝟙βᵀ}. Then ⟨M, a𝟙ᵀ + 𝟙βᵀ⟩_F = aᵀM𝟙 + 𝟙ᵀMβ = 0 for M ∈ 𝒟, and
   dim 𝒟 + dim 𝒪 = (n − 1)(m − 1) + (n + m − 1) = nm. So ℝ^{n×m} = 𝒟 ⊕ 𝒪 orthogonally, and M ↦ J_nMJ_m is the
   projector onto 𝒟.
2. The biases range over 𝒪, so J = ‖J_n(L − WCᵀ)J_m‖²_F + ‖Π_𝒪(L − WCᵀ) − b𝟙ᵀ − 𝟙b̃ᵀ‖²_F, and the biases zero the
   second term.
3. J_nWCᵀJ_m ranges over the elements of 𝒟 of rank ≤ d, each attained with WCᵀ equal to itself.
4. By Eckart–Young (formalML `svd`, Thm 2) the best is (J_nLJ_m)_d, which lies in 𝒟, because its singular vectors
   lie in the row and column spaces of J_nLJ_m.
5. Finally L − PMI = (log X_i·)𝟙ᵀ + 𝟙(log X_·j)ᵀ − log N 𝟙𝟙ᵀ ∈ 𝒪. ∎

*Reading.* This is the full-information limit. With every cell observed, GloVe sees the whole association matrix,
as Topic 6's SVD does: it is Topic 6's Def 3 applied to double-centered PMI, two-sided PCA (formalML
`pca-low-rank`). Prop 3 is the opposite limit, and Def 2 moves toward this one.

**Asserted** (`test_flat_weight_closed_form`): on 10 random full-support instances (6–12 × 5–10, d ∈ {1, 2, 3}):
- J_nLJ_m = J_n PMI J_m to 10⁻¹²;
- L-BFGS from a random start and ALS (no ridge, 400 sweeps) reach the minimum to 10⁻⁸ relative and the score
  matrix to 10⁻⁶.

### Proposition 3 (support blindness)
J depends on X only through {(i, j, X_ij) : (i, j) ∈ Ω}. If log X_ij = u_i + v_j on Ω (in particular, if every
observed count is the same), then (0, 0, u, v) attains J = 0, the global minimum, whatever Ω is. Any W, C with
⟨w_i, c_j⟩ = 0 on Ω are also optimal.

*Proof.* J ≥ 0, and direct substitution gives J = 0. ∎

**Asserted** (`test_support_blindness`):
- random instances: two supports with the same additive values both give J = 0 at (0, 0, u, v), and ridge ALS
  reaches it (loss below 10⁻¹², and |w·c| on Ω below 10⁻⁵);
- **corpus:** with every L8 count set to e, ALS's vectors are 0 to 10⁻¹², the biases carry b + b̃ = 1, and the
  exact optimum's vectors score exactly 0.500 dev and test AUC.

**Corpus findings for §3** (`test_benchmark`, `test_scale_curve`, `test_zero_imputation`).
- **Sparsity.** 69.0% of observed L8 cells are seen once and 80.4% at most twice. At d = 25, GloVe has 178,100
  parameters against 143,345 observed cells.
- **Additive share.** On Ω, the biases alone (a weighted two-way fit, d = 0) explain 71.9% of f-weighted variance
  of log X. 25-dimensional vectors add 27.3%, and that part carries no sector information.
- **The controlled comparison** (L8, d = 25; Topic 6's nouns):

| method | dev AUC / 1-NN | test AUC / 1-NN |
|---|---|---|
| GloVe, AdaGrad, W + C | 0.480 / 0.330 | **0.476** / 0.270 |
| GloVe, AdaGrad, W | 0.485 / 0.339 | 0.469 / 0.322 |
| GloVe, ALS, W + C | 0.517 / 0.330 | 0.507 / 0.348 |
| SGNS (Topic 7, 40 epochs) | 0.871 / 0.866 | **0.886** / 0.852 |
| SVD(SPPMI₅), p = 0 (Topic 6) | 0.951 / 0.946 | **0.973** / 0.965 |

  - GloVe's test AUC − ½ = −0.024 (jackknife se 0.022), so it is at chance. Chance 1-NN is 0.269.
  - GloVe − SGNS: −0.410 AUC (se 0.027) and −0.583 1-NN (se 0.048).
  - GloVe − SVD: −0.497 (se 0.021) and −0.696 (se 0.045).
- **Scale and imputed zeros** (both GloVe fits by ALS):

| documents | seen once | GloVe | Def 2 (imputed) | SVD(SPPMI₅) |
|---|---|---|---|---|
| 300 | 0.690 | 0.507 | 0.602 | 0.973 |
| 1,200 | 0.590 | 0.583 | 0.855 | 0.991 |
| 3,000 | 0.538 | 0.695 | **0.946** | 0.986 |

  - Asserted: the singleton share falls, both GloVe curves rise at each step, Def 2 beats plain GloVe everywhere (by
    over 0.25 from 1,200 documents on), and the SVD stays above plain GloVe.
- **Def 2's identity.** Its implicit ALS objective equals the explicit dense sum to 10⁻⁹. With flat weights it is
  Thm 2 on X̃: 400 sweeps reach the closed form to 10⁻⁸ relative.

### Definition 3 and the cross-check
ALS is exact block coordinate descent. With (C, b̃) fixed, each row's (w_i, b_i) solves a weighted ridge regression
by its normal equations (formalStatistics `linear-regression`), and symmetrically for the contexts.

**Asserted** (`test_glove_gradient`, `test_training`):
- the hand-written gradient against complex step (10⁻¹⁰ relative) and central differences (10⁻⁶);
- AdaGrad's J falls across its checkpoints: 13,247 (epoch 1), 7,482, 4,216, 1,306, 693.4 (epoch 50);
- ALS's objective never increases;
- ALS reaches J = 228.3 in 40 sweeps. Σ_Ω f = 9,387.8, so the f-weighted mean squared residual is 0.074 for AdaGrad
  and 0.024 for ALS. Fifty AdaGrad epochs are not converged.

### Proposition 4 (GloVe's gauge is affine)
Write W̄ = [W, b, 𝟙] and C̄ = [C, 𝟙, b̃], so S = W̄C̄ᵀ.
- **(i)** For A ∈ GL(d), t, s ∈ ℝ^d, and β ∈ ℝ, the substitution below leaves every s_ij unchanged:
  - w′_i = Aᵀw_i + t and c′_j = A⁻¹c_j + s;
  - b′_i = b_i − ⟨Aᵀw_i, s⟩ − ⟨t, s⟩ + β;
  - b̃′_j = b̃_j − ⟨A⁻¹c_j, t⟩ − β.
- **(ii)** If two parameter sets give the same S and both W̄ have full column rank d + 2, they are related by a
  unique such substitution.
- **(iii)** Cosines among word vectors are not determined by J. Differences w_i − w_k transform by Aᵀ alone.

*Proof.*
- (i) ⟨Aᵀw, A⁻¹c⟩ = ⟨w, c⟩, and the cross terms ⟨Aᵀw_i, s⟩ + ⟨t, A⁻¹c_j⟩ + ⟨t, s⟩ are absorbed by the bias changes.
- (ii) Topic 7's Prop 3 on (W̄, C̄) gives W̄′ = W̄G and C̄′ = C̄G⁻ᵀ for a unique G ∈ GL(d + 2).
  - W̄'s last column is 𝟙 on both sides, and W̄ has full column rank, so Ge_{d+2} = e_{d+2}.
  - C̄'s (d + 1)-th column is 𝟙 on both sides, so G⁻ᵀe_{d+1} = e_{d+1}, that is, e_{d+1}ᵀG = e_{d+1}ᵀ.
  - So G = [[A, u, 0], [0ᵀ, 1, 0], [tᵀ, β′, 1]]. Reading W̄G = [WA + 𝟙tᵀ, Wu + b + β′𝟙, 𝟙] gives (i) with u = −As
    and β′ = β − ⟨t, s⟩.
- (iii) Translations cancel in differences. ∎

**Asserted** (`test_affine_gauge`): random κ = 50 substitutions with random t, s, β leave S unchanged to 10⁻¹⁰
relative. (A, t, s, β) is recovered to 10⁻⁸, and word cosines move.

### Proposition 5 (ridge chooses the balanced, centered factorization)
Let J_λ = J + λ(‖W‖²_F + ‖C‖²_F) with λ > 0 and the biases unpenalized. At every stationary point:
- WᵀW = CᵀC (balanced);
- 𝟙ᵀW = 𝟙ᵀC = 0 (centered).

So W = UΣ^{½}R for the SVD of WCᵀ (Topic 7 Lemma 1): Topic 6's p = ½ for the matrix GloVe fitted, centered. Globally,
min over factorizations of ½(‖W‖² + ‖C‖²) is ‖WCᵀ‖_*, attained exactly at balanced factors (Srebro, Rennie, and
Jaakkola 2005; formalML `convex-analysis`).

*Proof.*
1. Let G = ∂J/∂S, which is 2f(X)(s − log X) on Ω and 0 off it.
2. Bias stationarity gives G𝟙 = 0 and Gᵀ𝟙 = 0.
3. ∇_W = GC + 2λW = 0 and ∇_C = GᵀW + 2λC = 0. Then WᵀGC = −2λWᵀW. Its transpose is CᵀGᵀW = −2λCᵀC, and the
   left sides are transposes of each other, so WᵀW = CᵀC.
4. 𝟙ᵀ∇_W = (Gᵀ𝟙)ᵀC + 2λ𝟙ᵀW = 2λ𝟙ᵀW = 0, and likewise for C.
5. If WᵀW = CᵀC, then W = PSQᵀ and C = P′SQᵀ share S and Q, so WCᵀ = PS²P′ᵀ and
   ‖WCᵀ‖_* = tr S² = ½(‖W‖² + ‖C‖²). ∎

*Reading.* Unregularized GloVe has no such pin. Its trained gauge is whatever its optimizer's path leaves, which is
Topic 7's lesson again.

**Asserted** (`test_ridge_balanced_centered`):
- on 5 random instances (x_max = 5, λ = 0.05, 3,000 sweeps) ALS's fixed point has imbalance and column means below
  10⁻⁶, and ½(‖W‖² + ‖C‖²) = ‖WCᵀ‖_* to 10⁻⁶;
- a κ = 10 regauging of the same product costs more;
- **corpus:** 40 ALS sweeps at λ = 10⁻³ are not near a stationary point in the gauge directions: imbalance 0.999,
  centering 0.355 / 0.386 (AdaGrad's unregularized factors: 0.093, 0.139 / 0.142). ALS's first sweep fits W through
  the small initial C, and the weak ridge has barely pulled the two toward balance. Neither trained model sits in
  the ridge's gauge.

### Theorem 3 (exact parallelograms are facts about count ratios)
Let a, b, c, d be words, and K a set of contexts on which all four rows are observed and fitted exactly
(s_ij = log X_ij). Suppose [C_K, 𝟙] has full column rank d + 1. Then
w_b − w_a = w_d − w_c ⟺ Δ_j := PMI_bj − PMI_aj − PMI_dj + PMI_cj is constant on K
⟺ P(j | b)/P(j | a) ∝ P(j | d)/P(j | c) on K.
The condition survives every substitution of Prop 4. For SGNS at its exact optimum (scores PMI_α − log k, no biases)
the same holds with "constant" replaced by "zero", given C_K of full column rank d.

*Proof.*
1. Let v = w_b − w_a − w_d + w_c and κ = b_b − b_a − b_d + b_c. Then ⟨v, c_j⟩ + κ = Δ log X_j.
2. log X_ij = PMI_ij + log X_i· + log X_·j − log N, and the coefficients (1, −1, −1, 1) sum to zero. So the column
   terms cancel and Δ log X_j = Δ_j + κ′, with κ′ = Δ log X_i· constant.
3. If v = 0, then Δ_j = κ − κ′ is constant.
4. Conversely, if Δ_j = δ₀ on K, then [C_K, 𝟙](v; κ − κ′ − δ₀) = 0, so v = 0.
5. The ratio form is exp(Δ_j) = (P(j | b)P(j | c))/(P(j | a)P(j | d)).
6. A substitution maps v to Aᵀv, because the coefficients sum to zero.
7. Without biases, ⟨v, c_j⟩ = Δ_j exactly. ∎

*Attribution.* Pennington et al. conjecture that relations are ratios.
- Allen and Hospedales (2019) prove the general relation w_* = w_W + C†(ρ + σ − τ𝟙), with paraphrase and dependence
  error terms, by multiplying a PMI-row identity by the context pseudo-inverse (their Thm 1 and Thm 2,
  Cor 2.1). Thm 3 is its exact, zero-error case, adapted to GloVe's biases, and the gauge clause is ours.
- Ethayarajh, Duvenaud, and Hirst (2019) give a csPMI criterion under exact reconstruction, stated through word–word
  inner products and norms, which depend on the gauge. Thm 3 states the criterion through the context side, which is
  why it is gauge-free.

**Asserted** (`test_exact_parallelogram`): on 10 random exact log-bilinear tables with a planted parallelogram:
- Δ is constant to 10⁻¹⁰, and the ratio rows are proportional to 10⁻¹⁰;
- Thm 2's closed form, which fits exactly in its own gauge, has v = 0 to 10⁻⁸ and defect below 10⁻⁸;
- a random affine substitution keeps v = 0;
- breaking the relation breaks Δ and v (> 10⁻³);
- the SGNS form, without biases, holds exactly.

### Corollary 2 (3CosAdd and the gauge)
Def 4's answer is invariant under W ↦ DWQ, with D positive diagonal and Q orthogonal. It is not invariant under a
general A or a translation, both of which leave J unchanged (Prop 4) and leave Thm 3's parallelograms intact. The
gauge-free solver's answers are invariant under all of Prop 4.

*Proof.* Unit-normalizing undoes D, and Q preserves inner products. The solver reads S, which Prop 4 leaves
unchanged. The harness exhibits the failures for A and t. ∎

**Asserted** (`test_analogy_gauge_dependence`):
- on random instances the invariance under DWQ is exact, while a general A or a translation changes over 10% of the
  answers;
- **corpus:** every variant keeps its scores (relative change below 1.4·10⁻¹⁴), its median defect (to 10⁻⁸), and its
  gauge-free accuracy exactly. 3CosAdd moves:

| model | variant | test AUC | city → region open / within | gauge-free | company → sector open / within | gauge-free |
|---|---|---|---|---|---|---|
| SGNS | as trained | 0.886 | 0.633 / 0.707 | 0.693 | 0.375 / 0.515 | 0.446 |
| SGNS | p = 0 | 0.849 | 0.620 / 0.687 | 0.693 | 0.369 / 0.475 | 0.446 |
| SGNS | p = ½ (balanced) | 0.901 | 0.620 / 0.707 | 0.693 | 0.373 / 0.510 | 0.446 |
| SGNS | p = 1 | 0.890 | 0.573 / 0.707 | 0.693 | 0.342 / 0.531 | 0.446 |
| SGNS | random κ = 100 | 0.737 | **0.253** / 0.440 | 0.693 | **0.104** / 0.313 | 0.446 |
| GloVe | as trained (W + C) | 0.476 | 0.367 / 0.580 | 0.360 | 0.138 / 0.433 | 0.423 |
| GloVe | balanced | 0.476 | 0.360 / 0.580 | 0.360 | 0.138 / 0.435 | 0.423 |
| GloVe | centered | 0.485 | 0.373 / 0.573 | 0.360 | 0.133 / 0.440 | 0.423 |
| GloVe | translated | 0.466 | 0.380 / 0.580 | 0.360 | 0.115 / 0.444 | 0.423 |
| GloVe | random κ = 100 | 0.470 | 0.213 / 0.500 | 0.360 | 0.087 / 0.367 | 0.423 |

  The SGNS rows are Topic 7's gauge cells, with the same random gauge (rng [w2v.SEED, 730]). The "translated" row
  moves both W and C by a random vector as long as the mean word vector.

## 4. Finance case study (§6): a filings knowledge base as offsets

The relations come from `fc.WORLD` and `fc.COMPANIES` through public accessors.
- **city → region** (LOCATED_IN): 13 cities, 10 regions, 150 questions with distinct regions. The rarest words occur
  3 times (*calgary*, *japan*).
- **company → sector** (IN_SECTOR): 24 companies by the first token of their short name, 6 sectors, 480 questions.
  The rarest occurs 13 times.
- CEO surnames (1–21 occurrences) and company → city (shared cities) are not used.

`test_finance_findings`:

| relation | model | open | within | b nearest | gauge-free (defect: right / wrong) |
|---|---|---|---|---|---|
| city → region (chance within 0.111) | GloVe W + C | 0.367 | 0.580 | 0.040 | 0.360 (1.262 / 1.790) |
| | GloVe W | 0.413 | 0.600 | 0.060 | — |
| | SGNS W | **0.633** | 0.707 | 0.027 | 0.693 (1.162 / 1.809) |
| | SGNS W + C | 0.680 | 0.767 | 0.013 | — |
| | SVD p = 0 | 0.540 | 0.707 | 0.040 | 0.713 (0.802 / 1.294) |
| | SVD p = ½ | 0.513 | 0.700 | 0.040 | — |
| company → sector (chance within 0.200) | GloVe W + C | 0.138 | 0.433 | **0.415** | 0.423 (0.895 / 1.120) |
| | GloVe W | 0.140 | 0.423 | 0.438 | — |
| | SGNS W | **0.375** | 0.515 | 0.044 | 0.446 (0.910 / 1.088) |
| | SGNS W + C | 0.338 | 0.677 | 0.008 | — |
| | SVD p = 0 | 0.190 | 0.715 | 0.008 | 0.500 (0.907 / 1.049) |
| | SVD p = ½ | 0.194 | 0.690 | 0.010 | — |

Asserted:
- SGNS beats GloVe open and within on both relations, and beats within-category chance by over 2× on both;
- for every model the median defect exceeds 0.5 (no fit is near an exact parallelogram), yet the right tail's defect
  is below the wrong tails', and the gauge-free solver beats chance;
- for SGNS on city → region the gauge-free solver (0.693) is within 0.05 of 3CosAdd in the trained gauge (0.707).

Readings:
- The scores carry the relations, but no model's offsets are constant. The nearest-neighbor answer needs only
  relative proximity, and it depends on the factorization.
- GloVe's company → sector answers would be the given sector itself 41.5% of the time without the exclusion, which
  is Linzen's caveat at full strength.
- Curated questions for the lab (top-3 open answers):

  | question | GloVe | SGNS as trained | SGNS, random gauge |
  |---|---|---|---|
  | *Houston : Texas :: Boston : ?* | *massachusetts* | *ireland* | *downturn* |
  | *Chicago : Illinois :: Dublin : ?* | *canada* | *ireland* | — |
  | *Halvorsen : industrials :: Pellucid : ?* | *software* | *technology* | — |

- TransE (#24) makes w_h + r ≈ w_t the training objective instead of hoping a co-occurrence objective yields it.

## 5. `rigorFlag`

> GloVe's derivation from co-occurrence ratios rests on modeling choices — ratios as the statistic, a function of a
> dot product of differences, symmetry restored by hand — and Theorem 1 makes GloVe the quadratic of a Poisson fit
> with a tempered weight and the unobserved pairs dropped, not a likelihood. On this 300-document synthetic corpus,
> where 69% of observed pairs are seen once, GloVe's vectors do not recover the planted sectors (test AUC 0.476,
> within 1.1 standard errors of chance) while SGNS and the SVD of the same counts do. The effect fades as the
> generator's text is continued (0.695 at ten times the text) and imputing unseen pairs at half a count — our own
> parameter-free variant, not the published Swivel — reaches 0.946 there; GloVe's published regime is billions of
> tokens, and the scale experiment counts the continued text on the 300-document vocabulary, by ALS. Exact
> parallelograms need exact fits: no model's scores come close to one (median defect 0.80–1.26), 3CosAdd's answers
> depend on the factorization (a random basis change at identical loss cuts SGNS's city → region accuracy from 0.633
> to 0.253), and the open-vocabulary protocol excludes the query words, without which GloVe would answer 41.5% of
> the sector questions with the given sector. The knowledge base is synthetic, with 13 cities and 24 companies, and
> its questions share words. Fifty AdaGrad epochs are not converged (ALS reaches a third of the loss), minibatch
> summation departs from the C code's updates, the counts are Topic 6's unweighted windows rather than GloVe's
> distance-weighted ones, and every result is one seed. Proposition 5 describes stationary points that neither
> trained model reaches.

## 6. Viz design intent — `GloVeLab` (one component, two linked panels)

**Data.** All data comes from `viz_payload()`, mirrored in `src/components/viz/data/glove-lab.json` (11.6 kB;
`test_viz_payload` caps it at 40 kB).
- Add `test_viz_mirror_matches_payload` in Phase B.
- Scores and accuracies are rounded to 3 decimals inside the payload, and projections to 2.
- Marginalia tokens only.

**Panel A, "Three weights, one count"** (`weights`).
- *Revenue*'s L8 cells (Topic 7's `lab_pairs`), each with a, SGNS's b (k = 5, α = ¾), log a, the trained GloVe score
  s, the trained SGNS score, and x*:

  | context | a | b | log a | GloVe s | SGNS x (x*) |
  |---|---|---|---|---|---|
  | plantline | 1 | 0.516 | 0 | 0.069 | −0.913 (0.662) |
  | subswap | 2 | 2.172 | 0.693 | 0.563 | −0.818 (−0.083) |
  | outperformed | 10 | 10.391 | 2.303 | 1.559 | −0.743 (−0.038) |
  | fiscal | 53 | 24.738 | 3.970 | 3.900 | 0.751 (0.762) |
  | cost | 129 | 25.311 | 4.860 | 4.800 | 1.383 (1.629) |
  | outlook | 0 | 71.47 | — | 3.558 (free) | −6.282 (none) |

- **View:** three loss curves against each objective's own coordinate:
  - GloVe f(a)(s − log a)²;
  - Poisson a(e^δ − 1 − δ);
  - SGNS −φ, computed from a and b.

  Dots mark the trained scores. For *outlook*, GloVe has no term ("the score is free: 3.558"), while Poisson's e^s
  and SGNS's b·softplus slope downward.
- **Controls:** x_max ∈ {10, 100} and γ ∈ {½, ¾, 1}.
- **Bars:** each weight's share by count band. GloVe's shares are recomputed in the browser from `hist` (cells with
  count 1…99, plus a ≥ 100 bucket, where f = 1 for either x_max). The Poisson, SGNS, and cell shares are fixed.
- *Lessons:* curvature is weight; GloVe caps its weight and spends most of it on singletons; GloVe never sees the
  zero cell.

**Panel B, "What the vectors know."** Two views behind a toggle.
- *Analogies* (`analogies`):
  - controls: a relation, a method (GloVe: as trained, balanced, centered, random; SGNS: as trained, balanced,
    random), and a curated question;
  - the relation's words in their own 2-D principal plane, with head → tail arrows;
  - readouts: open, within, and b nearest, plus the gauge-free accuracy and median defect, which stay fixed across
    variants ("same scores, same loss, same gauge-free answers");
  - the curated question's top-3 answers.
  - *Lesson:* a random basis change scrambles the arrows and the answers, while the scores, and everything computed
    from them alone, stay put.
- *Scale* (`scale`): test AUC against documents (300, 1,200, 3,000) for GloVe, Def 2, and SVD, with the singleton
  share, and chance at 0.5.
- **Link:** choosing GloVe in Panel B highlights GloVe's curve in Panel A.

**Mobile:** follow Topics 6–7. Rows wrap, inputs get `minWidth: 0`, and `scrollWidth === 375` must hold in every view
and toggle. Long pseudo-words in answer lists are truncated with an ellipsis.

## 7. Notebook cell map (`01_glove_embedding_geometry.ipynb`, 21 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; L8 sparsity, parameters against cells | — |
| 2–3 | §1 Def 1, Prop 1, Prop 2 | `test_ratio_argument`, `test_log_bilinear_rank`, `test_glove_gradient` |
| 4–5 | §2 Thm 1, Cor 1; weight shares | `test_poisson_bregman`, `test_three_weights` |
| 6–7 | §3 Thm 2, Prop 3 | `test_flat_weight_closed_form`, `test_support_blindness` |
| 8–9 | the controlled comparison; additive share | `test_benchmark` |
| 10–11 | scale and imputed zeros | `test_scale_curve`, `test_zero_imputation` |
| 12–13 | §4 Def 3; AdaGrad against ALS | `test_training` |
| 14–15 | §5 Props 4–5, Def 4, Thm 3, Cor 2; gauge table | `test_affine_gauge`, `test_ridge_balanced_centered`, `test_exact_parallelogram`, `test_analogy_gauge_dependence` |
| 16–17 | §6 finance | `test_finance_findings` |
| 18–20 | lab payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

`test_continued_corpus` runs only in the harness. Commands:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/glove-embedding-geometry/glove_embedding_geometry.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/glove-embedding-geometry/01_glove_embedding_geometry.ipynb
```

## 8. Cross-site links (all `status: "published"` on GitHub `main`, verified 2026-10-23; re-verify at the B3 gate)

| field | topic | relationship |
|---|---|---|
| `formalmlPrereqs` | `svd` | Eckart–Young gives the flat-weight theorem: with every co-occurrence observed and uniform weights, GloVe's optimum is the truncated SVD of the double-centered PMI matrix. |
| `formalmlPrereqs` | `information-geometry` | Each observed cell of the Poisson log-bilinear likelihood contributes the Bregman divergence of the exponential, whose quadratic with a tempered weight is GloVe's loss. |
| `formalmlPrereqs` | `pca-low-rank` | Centering rows and columns before a truncated SVD is two-sided PCA, which is what GloVe computes when every pair is observed and the weights are flat. |
| `formalmlConnections` | `convex-analysis` | The nuclear norm's variational form, the least half sum of squared factor norms, is why weight decay pins GloVe's factorization at the balanced one. |
| `formalmlConnections` | `gradient-descent` | GloVe trains by AdaGrad, the per-coordinate adaptive step behind RMSProp and Adam; the page shows its trained factorization is whatever that path leaves. |
| `formalmlConnections` | `representation-learning` | Word analogies are the classic probe of learned representations; the page shows exact parallelograms are facts about count ratios while nearest-neighbor answers depend on the basis. |
| `formalragConnections` | `hypersphere-vmf-geometry` | Analogy answers and benchmark scores here are cosine comparisons of unit vectors, the geometry that page develops on the sphere, anisotropy included. |
| `formalragConnections` | `dense-retrieval-dual-encoders` | A dual encoder scores by an inner product of two embeddings, so it shares GloVe's gauge freedom: the loss fixes the scores, not the geometry of either tower. |
| `formalstatisticsPrereqs` | `generalized-linear-models` | GloVe's weighted least squares on log counts is one Fisher-scoring step of a Poisson regression with log link at the saturated fit, with the Fisher weight tempered. |
| `formalstatisticsPrereqs` | `linear-regression` | Alternating least squares solves GloVe one block at a time, each word's vector and bias by the normal equations of a weighted ridge regression. |
| `formalstatisticsConnections` | `regularization-and-penalized-estimation` | A ridge penalty on word and context vectors with free biases makes every stationary point balanced and centered: the penalty chooses the geometry the loss leaves free. |
| `formalstatisticsConnections` | `exponential-families` | The Poisson log-partition supplies the Bregman divergence whose quadratic is GloVe's loss, with curvature equal to the count. |
| `formalcalculusPrereqs` | `mean-value-taylor` | Taylor's theorem with the Lagrange remainder sandwiches the Poisson cell loss between two quadratics, which is how GloVe's least squares approximates it. |

**Not cited:**
- formalML `kl-divergence` (Topic 7 carries it);
- formalRAG `infonce-contrastive-objective` (not used here);
- formalCalculus `convex-optimization` (a draft stub);
- formalStatistics `maximum-likelihood` (`generalized-linear-models` carries the Poisson fit).

formalML `gradient-descent` treats RMSProp and Adam but not AdaGrad, and the relationship says so.

## 9. References (Chicago 17th, notes-and-bibliography; every URL verified 2026-10-23 — ACL and arXiv by title, DOIs via Crossref)

1. Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation."
   In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1532–43. Doha: ACL,
   2014. https://aclanthology.org/D14-1162/. — paper; Def 1, Prop 1, Thm 1's reading.
2. Mikolov, Tomas, Wen-tau Yih, and Geoffrey Zweig. "Linguistic Regularities in Continuous Space Word
   Representations." In *Proceedings of NAACL-HLT 2013*, 746–51. Atlanta: ACL, 2013.
   https://aclanthology.org/N13-1090/. — paper; Def 4.
3. Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. "Efficient Estimation of Word Representations in Vector
   Space." arXiv:1301.3781, 2013. https://arxiv.org/abs/1301.3781. — paper; the analogy test set (pointer).
4. Levy, Omer, and Yoav Goldberg. "Linguistic Regularities in Sparse and Explicit Word Representations." In
   *Proceedings of the Eighteenth Conference on Computational Natural Language Learning*, 171–80. Ann Arbor: ACL,
   2014. https://aclanthology.org/W14-1618/. — paper; 3CosMul and the additive reading of 3CosAdd.
5. Levy, Omer, Yoav Goldberg, and Ido Dagan. "Improving Distributional Similarity with Lessons Learned from Word
   Embeddings." *Transactions of the Association for Computational Linguistics* 3 (2015): 211–25.
   https://doi.org/10.1162/tacl_a_00134. — paper; GloVe against SGNS, W + C.
6. Shi, Tianze, and Zhiyuan Liu. "Linking GloVe with word2vec." arXiv:1411.5595, 2014.
   https://arxiv.org/abs/1411.5595. — paper; Cor 1.
7. Arora, Sanjeev, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. "A Latent Variable Model Approach to
   PMI-based Word Embeddings." *Transactions of the Association for Computational Linguistics* 4 (2016): 385–99.
   https://doi.org/10.1162/tacl_a_00106. — paper; analogies under a generative model (pointer).
8. Allen, Carl, and Timothy Hospedales. "Analogies Explained: Towards Understanding Word Embeddings." In *Proceedings
   of the 36th International Conference on Machine Learning*, PMLR 97 (2019). https://arxiv.org/abs/1901.09813. —
   paper; Thm 3.
9. Ethayarajh, Kawin, David Duvenaud, and Graeme Hirst. "Towards Understanding Linear Word Analogies." In
   *Proceedings of the 57th Annual Meeting of the ACL*, 3253–62. Florence: ACL, 2019.
   https://arxiv.org/abs/1810.04882. — paper; Thm 3.
10. Linzen, Tal. "Issues in Evaluating Semantic Spaces Using Word Analogies." In *Proceedings of the 1st Workshop on
    Evaluating Vector-Space Representations for NLP*, 13–18. Berlin: ACL, 2016. https://aclanthology.org/W16-2503/.
    — paper; the exclusion caveat (Def 4).
11. Rogers, Anna, Aleksandr Drozd, and Bofang Li. "The (Too Many) Problems of Analogical Reasoning with Word
    Vectors." In *Proceedings of the 6th Joint Conference on Lexical and Computational Semantics*, 135–48.
    Vancouver: ACL, 2017. https://aclanthology.org/S17-1017/. — paper; analogy evaluation (§6).
12. Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization." *Journal of Machine Learning Research* 12 (2011): 2121–59.
    https://www.jmlr.org/papers/v12/duchi11a.html. — paper; Def 3.
13. Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative Filtering for Implicit Feedback Datasets." In *2008
    Eighth IEEE International Conference on Data Mining*, 263–72. Pisa: IEEE, 2008.
    https://doi.org/10.1109/ICDM.2008.22. — paper; Def 2's ALS identity.
14. Shazeer, Noam, Ryan Doherty, Colin Evans, and Chris Waterson. "Swivel: Improving Embeddings by Noticing What's
    Missing." arXiv:1602.02215, 2016. https://arxiv.org/abs/1602.02215. — paper; Def 2's published counterpart.
15. Srebro, Nathan, and Tommi Jaakkola. "Weighted Low-Rank Approximations." In *Proceedings of the Twentieth
    International Conference on Machine Learning*, 720–27. Washington, DC: AAAI Press, 2003.
    https://cdn.aaai.org/ICML/2003/ICML03-094.pdf. — paper; weighted low rank has no closed form (§2 remark).
16. Srebro, Nathan, Jason D. M. Rennie, and Tommi S. Jaakkola. "Maximum-Margin Matrix Factorization." In *Advances in
    Neural Information Processing Systems* 17 (2005).
    https://papers.nips.cc/paper_files/paper/2004/hash/e0688d13958a19e087e123148555e4b4-Abstract.html. — paper;
    Prop 5.
17. Collins, Michael, Sanjoy Dasgupta, and Robert E. Schapire. "A Generalization of Principal Components Analysis to
    the Exponential Family." In *Advances in Neural Information Processing Systems* 14 (2001).
    https://papers.nips.cc/paper_files/paper/2001/hash/f410588e48dc83f2822a880a68f78923-Abstract.html. — paper;
    Thm 1.
18. Cotterell, Ryan, Adam Poliak, Benjamin Van Durme, and Jason Eisner. "Explaining and Generalizing Skip-Gram
    through Exponential Family Principal Component Analysis." In *Proceedings of the 15th Conference of the European
    Chapter of the ACL*, 175–81. Valencia: ACL, 2017. https://aclanthology.org/E17-2028/. — paper; Thm 1.
19. McCullagh, Peter, and John A. Nelder. *Generalized Linear Models*. 2nd ed. London: Chapman and Hall, 1989.
    https://doi.org/10.1201/9780203753736. — book; Thm 1's IRLS reading.
20. Aczél, János, and Jean Dhombres. *Functional Equations in Several Variables*. Cambridge: Cambridge University
    Press, 1989. https://doi.org/10.1017/CBO9781139086578. — book; Cauchy's equation (Prop 1).
21. Eckart, Carl, and Gale Young. "The Approximation of One Matrix by Another of Lower Rank." *Psychometrika* 1, no. 3
    (1936): 211–18. https://doi.org/10.1007/BF02288367. — paper; Thm 2.
22. Bordes, Antoine, Nicolas Usunier, Alberto Garcia-Durán, Jason Weston, and Oksana Yakhnenko. "Translating
    Embeddings for Modeling Multi-relational Data." In *Advances in Neural Information Processing Systems* 26
    (2013). https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html. —
    paper; pointer to #24.

## 10. Phase B notes

- **Numbering.** MDX at `src/content/topics/glove-embedding-geometry.mdx`, in page order: Defs 1–4, Props 1–5,
  Thms 1–3, Cors 1–2. The gauge-free solver is unnumbered prose after Cor 2. No citations in `TheoremBlock` titles.
- **References.** `references[].type`: 19 and 20 are `book`, the rest `paper`.
- **Graph.** Flip the node's `status` to `published` and remove the title from `planned` in `src/data/curriculum.ts`.
  Frontmatter: `prerequisites: [word2vec-implicit-factorization]`, `connections: []`.
- **Lab data.** `src/components/viz/data/glove-lab.json` = `json.dumps(viz_payload(), separators=(',', ':'))`. Add
  `test_viz_mirror_matches_payload`.
- **Notation.**
  - Write 𝟙 as `\mathbf{1}`.
  - Use γ for GloVe's exponent and keep α for P_α.
  - Write f(X) with X the count. Write "d + 2" in math.
  - No bare `<`; use `\lt`. Lexicon, city, and pseudo-words go in italics, not KaTeX.
- **Display math:** Def 1, Thm 1's identity and sandwich, Cor 1's table, Thm 2's decomposition, Prop 4's
  substitution, and Thm 3's chain of equivalences.
- **Mobile.** At 375 px the prose column is about 343 px. The benchmark, scale, gauge, and finance tables need
  horizontal-scroll wrappers.
- **Overview hook:** GloVe 0.476 against SGNS 0.886 and SVD 0.973 on the same counts, and SGNS's city → region
  0.633 → 0.253 under a basis change at identical loss.
- **Page precision.** Quote numbers from the harness at full precision, not from `summary()`'s 4-decimal print.
  The payload already rounds to 3 decimals.
- **Gate.** Re-run the harness at the B3 gate (≈ 46 s). The notebook runs ≈ 51 s: under the 60 s limit, with little
  room for more training.
