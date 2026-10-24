# Brief — `word2vec-implicit-factorization`

**word2vec as Implicit Matrix Factorization.** Roadmap #7. Prerequisites: `distributional-semantics-ppmi-svd` (#6,
PR #12) and `neural-language-models` (#5, PR #11), both published. This is the second `representations` topic and a
prerequisite of `glove-embedding-geometry` (#8) and `contextual-embeddings-masked-lm` (#12).

**Phase A** (this brief and the verified notebook), 2026-10-23:
- The harness passes 13/13 under Python 3.15.0, numpy==2.5.4, and scipy==1.18.1, in ≈ 44 s.
- The narrative notebook (18 cells) executes with exit 0 in ≈ 47 s and is committed without outputs.

**Scaffold decisions** (Jonathan, 2026-10-23):
- **In:** gauge + balancedness.
- **Out:** word2vec's sampling details (dynamic window, subsampling) and hierarchical softmax, kept only as a remark
  and pointers.
- **Checks:** internal only — complex step, L-BFGS, Topic 6's SVD.
- **Training budget:** the controlled run trains 40 epochs and the dev grid 20.
- **Bias direction:** stays a corpus observation, not a numbered result.
- **Difficulty:** `intermediate`.
- **Authorship:** every function body was written by Claude at Jonathan's request, for his review.

**Source of truth:** `notebooks/word2vec-implicit-factorization/word2vec_implicit_factorization.py`. Every number below
is printed or asserted by it, and none may change without it.

**Binding forward promises, and where they are kept:**
- **Topic 6, §"Where this goes next":** "Skip-gram with negative sampling learns word and context vectors whose dot
  products, at the optimum, equal PMI − log k … The word2vec topic proves it, and its customary sum of word and
  context vectors is Corollary 1's projection."
  - Thm 1 proves the first half.
  - Lemma 1 and Cor 2 prove the second, for balanced factors and up to a rotation.
  - The page must say plainly that both hold only under conditions SGD meets approximately: full rank for the
    first, balance for the second.
- **Topic 5, §5 and §8:** the embedding table becomes the product, and the training objective "factorizes a matrix of
  association scores, a low-rank matrix of the kind Theorem 6 is about." Prop 2 is the bridge: all three objectives
  fit PMI up to row and column offsets, which is Topic 5's rank ≤ d + 2.

**Thesis.**
- **The objective chooses the matrix:** shifted PMI, weighted by counts and softly clipped (§§1–2).
- **The optimizer chooses the factorization.** The loss sees only WCᵀ (§4), so it cannot fix word similarity.
  Balance is what fixes it, and balanced factors are Topic 6's p = ½.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | word2vec as Implicit Matrix Factorization |
| `{{SUBTITLE}}` | Skip-gram's three objectives share one optimum up to offsets, shifted PMI; negative sampling fits it by a count-weighted logistic PCA; and since the loss sees only the product of word and context vectors, the optimizer's balance, not the objective, decides which similarities we get. |
| `{{SLUG}}` | `word2vec-implicit-factorization` (= graph node id) |
| `{{DOMAIN}}` | `representations` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[distributional-semantics-ppmi-svd, neural-language-models]` |
| `{{FORMALML_PREREQS}}` | `density-ratio-estimation`, `kl-divergence`, `information-geometry`, `svd`; connections `gradient-descent`, `representation-learning`, `random-walks` |
| `{{FORMALRAG_PREREQS}}` | none; connections `infonce-contrastive-objective`, `negative-sampling-hard-negatives`, `dense-retrieval-dual-encoders` |
| `{{STATS_CALC_PREREQS}}` | Stats `generalized-linear-models`; Calc `first-order-odes`, `hessian` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/neural-language-models/`, `notebooks/finance-corpus/` |

Tags: `word2vec`, `skip-gram`, `negative-sampling`, `noise-contrastive-estimation`, `pointwise-mutual-information`,
`matrix-factorization`, `logistic-pca`, `implicit-bias`. `estimatedReadTime` ≈ 40. In-site `connections`: none, since
both dependents are still planned. The graph's two incoming edges already exist.

**Scope.** This topic is not on the design doc's §6 watch-list, but it adjoins formalRAG.
- **formalRAG `infonce-contrastive-objective`** owns the (N+1)-way softmax contrastive loss:
  - the CPC bound, its Thm 1. Its proof sketch notes that the optimal critic is PMI up to a per-query constant;
  - the alignment–uniformity decomposition (Thm 2);
  - the gradient (Thm 3).
- **formalRAG `negative-sampling-hard-negatives`** owns hard-negative mining, false negatives, and debiasing (its
  Props 1–2, Thm 1, Cor 1).
- Neither page has the *binary* SGNS objective, NCE in its data-vs-noise form, P_α noise, or word2vec.
- **This topic owns:**
  - SGNS and its population optimum;
  - the softmax / NCE / negative-sampling relation;
  - weighted logistic PCA;
  - gauge and balance.

  The page states the contrast once: InfoNCE's critic is PMI only up to a per-query constant, while SGNS's logit is
  pinned to PMI_α − log k.
- **formalML `density-ratio-estimation`** Thm 4 ("Classification-DRE identity") is cited for the reading logit = log
  density ratio + log prior odds. That page asserts the logistic population minimizer without proving it, so Thm 1
  proves it per cell.
- **formalML `svd`** Thm 2 (Eckart–Young–Mirsky) and **formalRAG `dense-retrieval-dual-encoders`** Thm 2 (the rank
  ceiling) are cited, not re-proved.
- **Out of scope apart from pointers:**
  - GloVe, analogies, and linear structure (#8);
  - contextual embeddings (#12);
  - KG embeddings (#24, which also use negative sampling);
  - fastText (subword vectors; Topic 2);
  - hierarchical softmax;
  - the dynamic window and subsampling (one remark; Levy, Goldberg, and Dagan 2015 treat them as count reweightings).
  - **Retrieval** stays in formalRAG (CLAUDE.md).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | the fit-by-count picture: SGNS's w·c tracks PMI_α − log 5 with r = 0.98 on pairs seen ≥ 100 times and r = 0.64 on pairs seen once | — | cells 0–1 |
| 1 | Three objectives, one matrix | one word's scores over contexts as a row of a matrix | Defs 1–3; **Prop 1**; **Thm 1**; **Prop 2** | cells 2–3 |
| 2 | What SGNS minimizes at low rank | the five count bands | **Thm 2**; **Cor 1**; corpus: fit by band, unobserved cells, the bias direction | cells 4–5 |
| 3 | The algorithm | — | Def 4; the unbiased sampled loss; remark on the dynamic window and subsampling | cells 6–7 |
| 4 | Gauge and balance | "same loss, different neighbors": *strong* and *loans* under five factorizations of one product | **Prop 3**; **Thm 3**; **Lemma 1**; **Cor 2** | cells 8–10 |
| 5 | Recovering the planted sectors | the controlled pair: SGNS 0.886 against SVD 0.973 on the same matrix | Topic 6's protocol; Cor 1's mechanism | cells 11–12 |
| — | The lab | `<SGNSLab client:visible />` | — | cells 15–16 |
| 6 | Finance case study: polarity isn't a direction | *strong*'s nearest SGNS neighbor at L2 is *disappointing* | the leave-one-out probe | cells 13–14 |
| 7 | Where this goes next | — | #8 (GloVe's f(X) weight against ab/(a + b); analogies), #12, #24, formalRAG InfoNCE | — |

- **Lab placement.** One `<SGNSLab client:visible />` after §5. Panel A needs §§1–2, and Panel B needs §§2 and 4.
- **Forward pointers:**
  - from Cor 1 to #8 (GloVe is also a weighted least squares, on log counts);
  - from Prop 3 to #8 (analogies are a statement about the gauge-dependent word Gram);
  - from §6 to #12.

## 2. Definitions

**Notation.**
- Counts are Topic 6's Def 1, for one window context (L2 or L8), written #(w, c). Then #(w) = Σ_c #(w, c),
  N = Σ #, and P_α(c) = #(c)^α / Σ #(c′)^α.
- P_α is computed from the column sums (`ds.marginals`). This is what makes Thm 1 exact against Topic 6's `pmi`.
- Word and context vectors are the rows of W, C ∈ ℝ^{V×d}. Scores are x_{wc} = ⟨**w**_w, **c**_c⟩, so X = WCᵀ.
  In this topic C is the context matrix, and the page says so once.
- **Corpus.** L2 has N = 215,948 over 53,293 observed cells (0.45% of 3,425²). L8 has N = 576,098 over 143,345 (1.22%).

**Definitions.**
- **Def 1 (skip-gram, full softmax; Mikolov et al. 2013a).** L_soft(X) = −Σ_{w,c} #(w, c) log softmax(x_{w·})_c.
- **Def 2 (skip-gram with negative sampling; Mikolov et al. 2013b).**
  - L_NS(X) = −Σ_{w,c} [a log σ(x) + b log σ(−x)], with a = #(w, c) and b = k #(w) P_α(c).
  - b is the expected number of times (w, c) is drawn as a negative, because each pair draws k negatives i.i.d. from
    P_n = P_α.
- **Def 3 (noise-contrastive estimation; Gutmann and Hyvärinen 2010, 2012; Mnih and Teh 2012).** The classifier
  logit is s = x − log(k P_n(c)), where x models an unnormalized log P(c | w). L_NCE is L_NS with x replaced by s.
- **Def 4 (word2vec's SGD, as run here).**
  - Each epoch is a fresh permutation of the N pairs in minibatches of 512.
  - Each pair gets k = 5 negatives from P_α.
  - The step is on the summed sampled loss, with η_t = 0.05 · max(10⁻⁴, 1 − t/T).
  - Initialization follows the word2vec C code: W ~ U[−1/(2d), 1/(2d)] and C = 0. Training is in float32.
  - **Departure, stated on the page:** minibatch summation replaces the C code's sequential (Hogwild) updates.
  - The scatter is a product with the batch's CSC incidence matrix, more than 10× faster than `np.add.at`.

**Configurations.**
- **Controlled:** L8, d = 25, k = 5, α = 0.75, 40 epochs. This is Topic 6's winning matrix and rank.
- **Grid:** {L2, L8} × d ∈ {25, 100}, 20 epochs each.

## 3. Statements and proofs

### Proposition 1 (the softmax optimum)
For a row with #(w) > 0, the row loss −Σ_c #(w, c) log p_c over p = softmax(x_w) attains its infimum iff
#(w, c) > 0 for all c. Its minimizers are then x_{wc} = log P(c | w) + κ_w, that is, x = PMI(w, c) + log P(c) + κ_w.
Otherwise the infimum is approached as x_{wc} → −∞ and never attained. This is Topic 5's Cor 1 (a never-seen target
has no maximum-likelihood fit), one context at a time.

*Proof.* The loss is #(w) H(P(·|w), p) ≥ #(w) H(P(·|w)), with equality iff p = P(·|w) (Gibbs; formalML
`kl-divergence`, Props 1–2). The softmax has the open simplex as its range and is invariant to adding a constant to
x_w. ∎

### Theorem 1 (Levy and Goldberg 2014)
For a, b > 0, φ(x) = a log σ(x) + b log σ(−x) is strictly concave, with unique maximizer x* = log(a/b). For a = 0 < b,
φ increases strictly toward 0 as x → −∞ and has no maximizer.

Hence, for unconstrained X:
- L_NS decouples by cell, and a minimizer exists iff every pair is observed;
- the minimizer is x*_{wc} = log[#(w, c) / (k #(w) P_α(c))] = PMI_α(w, c) − log k. This is Topic 6's Def 2, the
  unclipped shifted matrix.

*Proof.*
1. φ′ = aσ(−x) − bσ(x) and φ″ = −(a + b)σ(x)σ(−x) < 0.
2. φ′ = 0 iff e^{−x} = b/a.
3. For a = 0, φ′ = −bσ(x) < 0.
4. Substitute P(w, c) = #/N and P(w) = #(w)/N. ∎

*Reading.* σ(x*) = a/(a + b) is the posterior probability that a pair is data rather than noise, at prior odds 1/k.
So x* is the log density ratio plus the log prior odds (formalML `density-ratio-estimation`, Thm 4).

**Asserted.**
- **Closed forms against L-BFGS** (`test_three_optima`). On 12 random fully observed instances (3–6 × 3–6, k ∈ {1, 5,
  15}, α ∈ [0.5, 1]), unconstrained L-BFGS on each objective lands on its closed form to 10⁻⁶. The softmax's is
  compared after row-centering.
- **Shape of φ.** Second differences of φ are negative on [−8, 8], and its argmax is within 0.01 of log(a/b). With
  a = 0, φ is strictly increasing toward −∞.
- **An unobserved cell.** In an instance with one zero count, L-BFGS drives that cell below −10 while every other cell
  converges.
- **On the corpus** (`test_corpus_optimum`). At L2 and L8, log(a/b) equals Topic 6's `pmi(C, 0.75)` − log 5 on every
  observed pair to 10⁻¹², and φ′ vanishes there. Over 95% of cells are unobserved, which leaves them with no maximizer
  (98.8% at L8).

### Proposition 2 (NCE and negative sampling; one matrix up to offsets)
Under Thm 1's hypotheses:
- NCE's optimum is s* = log(a/b), so x*_NCE = log P(c | w), self-normalized;
- x*_NCE − x*_NS = log(k P_n(c)), a column offset. Negative sampling is NCE with the noise correction dropped;
- every optimum is PMI (α = 1) plus a row constant plus a column constant β_c:

| objective | row constant | column constant β_c |
|---|---|---|
| softmax | κ_w | log P(c) |
| NCE | 0 | log P(c) |
| NS | 0 | log P(c) − log P_α(c) − log k |

So any optimum has rank at most rank(PMI) + 2. With bias terms, a d-dimensional model expresses all three alike
(Topic 5 Thm 6; Topic 6 Thm 3 iii).

*Proof.* Apply Thm 1 to s. Then use log P(c | w) = PMI + log P(c) and PMI_α = PMI + log P(c) − log P_α(c), and
subadditivity of rank. ∎

*Contrast.* InfoNCE's optimal critic is PMI plus a per-query constant (formalRAG `infonce-contrastive-objective`,
Thm 1's proof). SGNS pins that constant.

**Asserted** (`test_three_optima`), on all 12 instances:
- the NCE optimum equals log P(c | w) to 10⁻¹²;
- NCE − NS equals log(k P_n(c)) to 10⁻¹²;
- double-centering (optimum − PMI) leaves 0 to 10⁻¹² for all three objectives.

### Theorem 2 (SGNS is exactly a weighted logistic PCA; Collins, Dasgupta, and Schapire 2001; Landgraf and Bellay 2017)
Let y = a/(a + b) ∈ [0, 1], and let H be the Bernoulli entropy. For every X:

L_NS(X) − Σ (a + b) H(y) = Σ_{w,c} (a + b) KL(Bern(y) ‖ Bern(σ(x_{wc}))).

So inf L_NS = Σ (a + b) H(y), and fitting a rank-d WCᵀ is a weighted KL projection of the natural-parameter target
X* = logit(y). On observed cells each term is (a + b) D_A(x, x*), the Bregman divergence of the softplus
A(x) = log(1 + eˣ) (formalML `information-geometry`, Def 8 and Thm 5). The curvature at x* is
(a + b)σ(x*)σ(−x*) = ab/(a + b).

*Proof.*
1. −a log σ(x) − b log σ(−x) = (a + b)[−y log σ(x) − (1 − y) log(1 − σ(x))], a Bernoulli cross-entropy, which equals
   H(y) + KL (formalML `kl-divergence`, Prop 1).
2. The Bregman form is the exponential-family identity KL(p_{θ*} ‖ p_θ) = A(θ) − A(θ*) − A′(θ*)(θ − θ*), with A′ = σ.
3. Differentiate twice. ∎

### Corollary 1 (against Topic 6's SVD)
Truncated SVD of SPPMI_k minimizes Σ (max(x*, 0) − x)², with uniform weight on a hard-clipped target (Eckart–Young;
formalML `svd`, Thm 2). SGNS minimizes Σ (a + b) D_A(x, x*). Three differences follow:
1. **Weights.** Near x*, SGNS ≈ ½ Σ ab/(a + b) (x − x*)², with ½ min(a, b) ≤ ab/(a + b) ≤ min(a, b), so rare pairs
   get little weight.
2. **Unobserved cells.** They have y = 0 and pay b log(1 + eˣ), a soft one-sided push toward −∞ where the SVD clips
   to 0.
3. **Negative targets.** Observed x* < 0 are kept, not clipped.

*Proof.* The weights follow from Thm 2's curvature and a + b ≤ 2 max(a, b). The unobserved term is direct
substitution. ∎

**Asserted** (`test_logistic_pca_identity`), on 20 random instances with 40% unobserved cells:
- Thm 2's identity holds to 10⁻¹² relative;
- the KL matches `scipy.stats.entropy` and the softplus Bregman form to 10⁻¹⁰;
- the numerical curvature at x* equals ab/(a + b), within ½ min(a, b) and min(a, b).

**Corpus** (`test_fit_by_count_band`). The controlled model: the correlation of w·c with PMI_α − log 5, and the mean
absolute error, by pair count.

| seen | pairs | r | mean abs. error |
|---|---|---|---|
| once | 98,926 | 0.641 | 1.297 |
| 2–4 | 27,994 | 0.818 | 0.537 |
| 5–19 | 12,200 | 0.940 | 0.291 |
| 20–99 | 3,409 | 0.979 | 0.198 |
| 100+ | 816 | 0.985 | 0.139 |

- r rises and the error falls monotonically, as Cor 1 predicts. 69% of the observed pairs are seen once.
- **Unobserved cells.** The mean w·c is −5.16 on unobserved cells against −0.63 on observed ones, and 97.9% of
  unobserved cells score below the observed median.
- **Not converged.** The loss is 1.6514 per pair against Thm 2's floor of 1.4752 (a KL gap of 0.176), with gradient
  norm 0.0037. SGD is not at a stationary point, and the rigorFlag says so.
- **The bias direction** (a corpus observation, not a numbered result). The product's top singular pair has u₁ > 0
  on every word and v₁ < 0 on every context, so σ₁u₁v₁ᵀ is entrywise negative.
  - It carries 87.4% of ‖WCᵀ‖²_F, with σ₁/σ₂ = 6.3.
  - Its mean entry, −5.15, is the unobserved cells' level.
  - Reading: word2vec has no bias terms. Prop 2 says the optimum wants offsets, and Thm 1 says the unobserved 98.8%
    want −∞, so SGNS spends one dimension on a negative offset.

### Proposition 3 (every factorization; the gauge)
If W′C′ᵀ = WCᵀ with all four matrices of full column rank d, then W′ = WA and C′ = CA⁻ᵀ for a unique A ∈ GL(d).
Every objective of Defs 1–3 therefore fixes the word Gram matrix only up to WMWᵀ, with M = AAᵀ ≻ 0 arbitrary. Dot
products among word vectors are preserved iff A is orthogonal.

*Proof.*
1. col W′ = col X = col W, so W′ = WA.
2. W's left inverse turns WAC′ᵀ = WCᵀ into AC′ᵀ = Cᵀ.
3. WMWᵀ = WWᵀ with W of full column rank forces M = I. ∎

**Asserted** (`test_gauge_invariance`):
- **Random instances.** A κ = 50 gauge leaves the loss unchanged to 10⁻¹² relative; A is recovered as W⁺W′; the
  word Gram moves; orthogonal A keep it.
- **Corpus, "same loss, different neighbors."** One product WCᵀ, five factorizations, all with the same product to
  10⁻¹⁰ relative. Here U, Σ, V is the product's SVD, and p is Topic 6's eigenvalue weight.

| factorization | test AUC | test 1-NN | *strong*'s top 5 |
|---|---|---|---|
| as trained | 0.886 | 0.852 | weak, unfavorable, solid, soft, healthy |
| UΣ^p, p = 0 | 0.849 | 0.878 | unfavorable, weak, soft, resilient, disappointing |
| p = ½ (balanced) | **0.901** | 0.852 | weak, unfavorable, solid, resilient, soft |
| p = 1 | 0.890 | 0.843 | weak, unfavorable, solid, healthy, efficient |
| random A, κ = 100 | 0.737 | 0.670 | adverse, healthy, unfavorable, solid, confident |

  Under the random gauge, *loans*'s nearest neighbor becomes *trialpoint*; it is *fees* as trained.

### Theorem 3 (gradient flow conserves the imbalance; Du, Hu, and Lee 2018; Arora, Cohen, and Hazan 2018)
Let L(W, C) = f(WCᵀ) with f differentiable and G = ∇f(X).
- Along gradient flow Ẇ = −GC, Ċ = −GᵀW, the matrix D = WᵀW − CᵀC is constant.
- For a step with any Ĝ, including a minibatch's sparse sampled gradient,
  D_{t+1} − D_t = η²(∇_Wᵀ∇_W − ∇_Cᵀ∇_C), where ∇_W = ĜC and ∇_C = ĜᵀW.

*Proof.*
1. d/dt WᵀW = −CᵀGᵀW − WᵀGC = d/dt CᵀC.
2. Expanding the discrete step, the first-order terms cancel. Gradient flow is formalCalculus `first-order-odes`'s
   ODE. ∎

**Asserted** (`test_gradient_flow_conservation`):
- **Real minibatch steps.** Three real L8 minibatch steps from the trained model, in float64, match the η² formula to
  10⁻¹⁰ relative, and the change itself is nonzero.
- **Small instance.** Gradient descent to a fixed horizon has drift 0.0284, 0.0146, and 0.0074 for
  η = 10⁻³, 5·10⁻⁴, 2.5·10⁻⁴. The ratios are 1.95 and 1.98, so the drift → 0 like η.

### Lemma 1 (balanced factorizations)
If X = UΣVᵀ has rank d with distinct singular values, WCᵀ = X, and WᵀW = CᵀC, then W = UΣ^½R and C = VΣ^½R for an
orthogonal R.

*Proof.*
1. Let B = WᵀW = CᵀC ≻ 0, so XXᵀ = WBWᵀ.
2. With the thin SVD W = PSQᵀ, WBWᵀ = PS⁴Pᵀ = UΣ²Uᵀ.
3. Distinct eigenvalues give P = UE, with E diagonal ±1, and S = Σ^½. So W = UΣ^½R_W, and likewise C = VΣ^½R_C.
4. Σ^½(R_W R_Cᵀ)Σ^½ = Σ forces R_W = R_C. ∎

### Corollary 2 (balanced word2vec is Topic 6's p = ½, and w + c is the positive part)
For balanced factors of X:
- (i) WWᵀ = UΣUᵀ, Topic 6's Def 3 at p = ½ for the matrix SGNS fitted;
- (ii) if X = QΛQᵀ is symmetric, ¼(W + C)(W + C)ᵀ = [X]₊, Topic 6's Cor 1, up to R.

Gradient flow from D₀ = 0 stays balanced (Thm 3), so this holds wherever the flow converges.

*Proof.* Substitute Lemma 1. For (ii), use Topic 6 Cor 1's computation, with u_i = s_iq_i and
v_i = s_i sign(λ_i)q_i. ∎

**Asserted** (`test_balanced_factorization`):
- **Random instances.** Balanced factors have gauge AAᵀ = I to 10⁻¹⁰, and WWᵀ = UΣUᵀ. Unbalanced factors have a
  non-orthogonal gauge. For symmetric indefinite products, ¼(W + C)(W + C)ᵀ = [X]₊ to 10⁻¹⁰.
- **End to end** (`symmetric_dynamics`). A symmetric 5 × 5 instance at α = 1 has a symmetric, indefinite optimum
  (2 negative eigenvalues, 3 positive). Gradient descent at full rank from a balanced start fits X* to 2·10⁻¹⁴.

| η | final imbalance | ¼(W + C)(W + C)ᵀ − [X*]₊, max entry |
|---|---|---|
| 0.02 | 0.0065 | 0.0042 |
| 0.01 | 0.0037 | 0.0024 |

  Both shrink with η.

**Corpus** (`test_corpus_balance`):
- **D is not conserved.** Word2vec's initialization is nearly balanced (‖D₀‖ = 1.3·10⁻⁴ of the final ‖WᵀW‖), but
  minibatch SGD at its step sizes is not gradient flow. D grows 1,819× in the first epoch, keeps growing (‖D‖ =
  4,162, 6,823, 9,101, 13,597 at epochs 1, 4, 10, 40), and ends at 0.80 of ‖WᵀW‖.
- **The gauge is near-orthogonal.** Cosines depend on the gauge A in W = UΣ^½A, and A is near-orthogonal from the
  first epoch on:

| epoch | 1 | 4 | 10 | 40 |
|---|---|---|---|---|
| A's singular values | [0.801, 1.206] | [0.786, 1.207] | [0.786, 1.215] | [0.788, 1.204] |

- **Where the imbalance sits.** It is concentrated in the bias direction (A's top row norm 0.907; median row norm
  0.997). D looks large because σ₁ dominates.
- **Effect on the benchmark.** Trained W scores test AUC 0.886 against 0.901 for the exactly balanced UΣ^½.
  Topic 6's promise holds approximately, and the page says how approximately.
- **Honest note.** With α = 0.75 the target is not symmetric, so Cor 2 (ii) is checked on small instances only.

## 4. Recovering the planted sectors (§5) and the finance case study (§6)

Protocol: Topic 6's Def 4, with the same 112 dev and 115 test nouns, pair AUC and 1-NN, jackknife and paired
standard errors, and selection on dev only (`test_recovery_on_test_words`).

**The controlled pair** (L8, SPPMI₅ / α = 0.75, k = 5, d = 25):

| method | dev AUC / 1-NN | test AUC / 1-NN |
|---|---|---|
| SGNS, 40 epochs (W) | 0.870 / 0.866 | **0.886** / 0.852 |
| SVD(SPPMI₅), p = 0 (Topic 6's selection) | 0.951 / 0.946 | **0.973** / 0.965 |
| SVD(SPPMI₅), p = ½ | 0.944 / 0.929 | 0.966 / 0.965 |

- **SGNS − SVD:** −0.087 test AUC (jackknife se 0.017, 5.1 se) and −0.113 in 1-NN (paired se 0.034). Chance 1-NN is
  0.269.
- **Representation:** W and (W + C)/2 score alike (0.886 and 0.886).
- **Mechanism (Cor 1, 1), on the benchmark nouns' rows.** Pairs seen at most 4 times are 92.3% of the support. They
  carry 98.7% of SPPMI₅'s squared mass, which the SVD fits uniformly, but only 54.8% of SGNS's local weight.
- **The rare band** (396 nouns seen 2–4 times, never selected on). SGNS scores AUC 0.652 / 1-NN 0.593, against SVD's
  0.728 / 0.657.
- **More epochs help.** The 20-epoch L8, d = 25 run scores 0.859.

**The dev grid** (20 epochs):

| config | dev AUC / 1-NN | test AUC / 1-NN | loss / pair |
|---|---|---|---|
| L2, d = 25 | 0.720 / 0.634 | 0.709 / 0.591 | 1.1797 |
| L2, d = 100 | 0.697 / 0.643 | 0.723 / 0.652 | 1.0968 |
| **L8, d = 25** (selected) | **0.863** / 0.848 | 0.859 / 0.765 | 1.6723 |
| L8, d = 100 | 0.830 / 0.920 | 0.833 / 0.887 | 1.5684 |

- Against Topic 6's selected SVD: −0.114 AUC (se 0.021) and −0.200 1-NN (se 0.039).
- Topic 6's selected sparse PPMI (sentence context) scores 0.813, so SGNS at L8 beats sparse counts in AUC but not
  the SVD.
- As in Topic 6, wider windows see more of a document-level label.

### Finance case study (§6): polarity isn't a direction

`test_finance_findings` uses the stand-in lexicon's 18 positive and 17 negative training words, never the real
Loughran–McDonald dictionary.

1. **Antonyms are SGNS neighbors too.** At L2, *strong*'s seven nearest SGNS neighbors are disappointing (0.974),
   improved, stronger, weak (0.956), robust, unfavorable, and favorable. At L8 *weak* is first (0.960). SGNS
   factorizes the same counts that made antonyms substitutes in Topic 6.
2. **The probe.** A leave-one-out difference-of-means direction on unit vectors classifies each held-out lexicon word
   by its side of the midpoint. It is translation-invariant, so the bias direction cannot help it.
   - **Sanity checks.** It scores ≥ 0.95 on random vectors with a planted polarity offset. On pure noise its mean
     over 20 draws is within 0.06 of ½.

   | context | SGNS W | SGNS (W + C)/2 | Topic 6 SVD | sparse PPMI | Topic 6 SO |
   |---|---|---|---|---|---|
   | L2 | 0.571 | 0.600 | 0.543 | 0.629 | 0.500 |
   | L8 | 0.571 | 0.600 | 0.514 | 0.629 | 0.343 |

   - Chance is 0.5 ± 0.085. The best probe is 1.52 standard errors above chance, and none reaches two.
   - Topic 6 found that windows give no polarity evidence (L2) or backwards evidence (L8). A factorization of those
     counts finds no polarity direction either. Polarity in this corpus is a document-level property, which Topic 6's
     document SO read (0.829) and which no window model sees.
3. **Polysemes dropped.** The scaffold's polyseme projections were dropped: a direction no better than chance cannot
   place them.

## 5. `rigorFlag`

> The theorem that skip-gram with negative sampling factorizes PMI_α − log k is exact only for an unconstrained score
> matrix and only on observed pairs: unobserved pairs (98.8% of cells at L8) have no optimum, and at word2vec's rank the
> objective is a count-weighted logistic PCA (Theorem 2), not a factorization of shifted PMI, so rare pairs are fitted
> loosely (r = 0.64 for pairs seen once against 0.98 for pairs seen 100 or more times). The trained models are finite-
> step minibatch SGD with sampled negatives, not converged (the loss stays 0.176 per pair above its floor) and not
> gradient flow: the conserved quantity of Theorem 3 grows over a thousandfold, so the balanced, p = ½ reading of
> Corollary 2 and the w + c identity hold only approximately (the gauge's singular values lie in [0.79, 1.20]), and with
> α = 0.75 the target is not symmetric, so the w + c identity is checked on small instances only. Minibatch summation
> departs from word2vec's sequential updates, and the dynamic window and subsampling are omitted. The corpus is
> synthetic with planted sectors (Topic 6's caveats carry over), the comparison with the SVD is at one budget (40 and
> 20 epochs) on a four-configuration grid, and the polarity probe uses a 35-word hand-written stand-in lexicon.

## 6. Viz design intent — `SGNSLab` (one component, two linked panels)

**Data.**
- All data comes from `viz_payload()`, mirrored to the decimal in `src/components/viz/data/sgns-lab.json` (16.4 kB;
  `test_viz_payload` caps it at 40 kB).
- Add `test_viz_mirror_matches_payload` in Phase B.
- Marginalia tokens only; one `--color-*` per count band, mapped in the component.

**Panel A, "One cell of the objective"** (`pairs`).
- *Revenue*'s most associated L8 context (highest PMI_α) in each count band, plus the most frequent context it never
  meets:

  | context | a | x* (α = 0.75, k = 5) | weight ab/(a + b) | trained w·c |
  |---|---|---|---|---|
  | plantline | 1 | 0.662 | 0.34 | −0.913 |
  | subswap | 2 | −0.083 | 1.04 | −0.818 |
  | outperformed | 10 | −0.038 | 5.10 | −0.743 |
  | fiscal | 53 | 0.762 | 16.87 | 0.751 |
  | cost | 129 | 1.629 | 21.16 | 1.383 |
  | outlook | 0 | none | — | −6.282 |

- **Controls:** the pair, k ∈ {1, 5, 15}, and α ∈ {1, 0.75}.
- **View:** the curve −φ(x) = −[a log σ(x) + b log σ(−x)], drawn from the payload's a, #(w), N, and P_α(c). A vertical
  rule marks x*, and a dot marks the trained w·c.
- **Readouts:** x* and the weight, from the payload's per-(α, k) tables. For *outlook*, the curve decreases without
  bound and the readout says "no minimum: the score is pushed toward −∞".
- *Lessons:*
  - x* moves by −log k as k changes;
  - the curve's width is 1/weight;
  - the singleton is fitted worst (trained −0.91 against x* 0.66), *fiscal* almost exactly.

**Panel B, "Training and the gauge."** Two views behind a toggle.
- *Fit* (`fit`):
  - w·c against x* for 200 sampled pairs (40 per band), colored by band, plus a strip of 40 unobserved cells at the
    bottom;
  - an epoch slider over checkpoints 0, 1, 4, 10, 40, with readouts for the loss, the per-band r, and the gauge's
    singular-value range;
  - at epoch 0 every score is 0 (C = 0).
  - *Lessons:* frequent pairs converge to the diagonal first and rare pairs stay loose; unobserved cells sink.
- *Gauge* (`gauge`):
  - five factorizations of one product, with test AUC/1-NN and the top 5 neighbors of *strong* and *loans*;
  - a fixed readout, "same product, same loss".
  - *Lesson:* the objective does not decide similarity. Balance (p = ½) is the best of the five, and an arbitrary
    gauge is far worse.
- **Link:** picking a pair in Panel A highlights it in Panel B's fit view when the pair is in the sample. Otherwise it
  shows its band.

**Mobile:** follow Topic 6. Readouts and the scatter stack full-width, slider rows get `flexWrap: 'wrap'`, and inputs
get `minWidth: 0`. Long pseudo-words in neighbor lists are truncated with an ellipsis.

## 7. Notebook cell map (`01_word2vec_implicit_factorization.ipynb`, 18 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; the two co-occurrence matrices | — |
| 2–3 | §1 Defs 1–3, Prop 1, Thm 1, Prop 2 | `test_three_optima`, `test_sgns_gradient`, `test_sampled_loss_is_unbiased`, `test_corpus_optimum` |
| 4–5 | §2 Thm 2, Cor 1; fit by band, bias direction | `test_logistic_pca_identity`, `test_fit_by_count_band` |
| 6–7 | §3 Def 4; the training history | — |
| 8–10 | §4 Prop 3, Thm 3, Lemma 1, Cor 2 | `test_gauge_invariance`, `test_gradient_flow_conservation`, `test_balanced_factorization`, `test_corpus_balance` |
| 11–12 | §5 the controlled pair, the grid, the mechanism | `test_recovery_on_test_words` |
| 13–14 | §6 finance | `test_finance_findings` |
| 15–17 | lab payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Run:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/word2vec-implicit-factorization/word2vec_implicit_factorization.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/word2vec-implicit-factorization/01_word2vec_implicit_factorization.ipynb
```

## 8. Cross-site links (all `status: "published"` on GitHub `main`, verified 2026-10-23; re-verify at the B3 gate)

| field | topic | relationship |
|---|---|---|
| `formalmlPrereqs` | `density-ratio-estimation` | Negative sampling is a classifier between observed pairs and noise pairs, so its optimal logit is the log density ratio plus the log prior odds of that page's classification identity, here PMI minus log k. |
| `formalmlPrereqs` | `kl-divergence` | The cross-entropy decomposition turns the negative-sampling loss into a count-weighted sum of Bernoulli KL divergences, which is what skip-gram minimizes at low rank. |
| `formalmlPrereqs` | `information-geometry` | Each cell's KL term is a Bregman divergence of the Bernoulli log-partition, so skip-gram with negative sampling is an exponential-family (logistic) PCA in natural parameters. |
| `formalmlPrereqs` | `svd` | Truncated SVD of shifted PPMI is the uniform-weight, Eckart–Young counterpart that the page compares with skip-gram's count-weighted factorization of the same matrix. |
| `formalmlConnections` | `gradient-descent` | Word2vec is stochastic gradient descent on a factorized loss; the page shows that its gradient flow conserves the difference of the word and context Gram matrices, which fixes the factorization. |
| `formalmlConnections` | `representation-learning` | Word2vec's embeddings are an early learned representation; the page shows that its objective fixes only the product of word and context vectors, so similarity depends on the optimizer. |
| `formalmlConnections` | `random-walks` | DeepWalk applies skip-gram to random-walk co-occurrences, so the implicit-factorization result proved here is the one behind its matrix-factorization reading. |
| `formalragConnections` | `infonce-contrastive-objective` | InfoNCE is the softmax counterpart of negative sampling: its optimal critic is PMI up to a per-query constant, while skip-gram's binary loss pins the logit to PMI minus log k. |
| `formalragConnections` | `negative-sampling-hard-negatives` | Both pages draw negatives from a noise distribution; there the question is which negatives to mine for retrieval, here what the unigram-to-the-0.75 noise makes the optimum factorize. |
| `formalragConnections` | `dense-retrieval-dual-encoders` | A dual encoder's score is an inner product of two embeddings, the same factorized form as word and context vectors, with the same rank ceiling and the same gauge freedom. |
| `formalstatisticsPrereqs` | `generalized-linear-models` | Each cell of the negative-sampling objective is a logistic regression likelihood with a canonical link, strictly concave in the score, which is why its optimum is unique. |
| `formalcalculusPrereqs` | `first-order-odes` | Gradient descent is the Euler discretization of gradient flow, the ODE along which the page proves that the word and context Gram matrices' difference is conserved. |
| `formalcalculusPrereqs` | `hessian` | The second derivative of the cell loss at its optimum is the local weight that skip-gram puts on each word–context pair, which makes it a weighted least squares near the optimum. |

**Not cited:**
- formalML `shannon-entropy` (hierarchical softmax is out of scope);
- formalStatistics `exponential-families` (`information-geometry` carries the Bregman form);
- formalStatistics `law-of-large-numbers` (the unbiased sampled loss is shown by linearity, with a Monte Carlo check
  only);
- formalCalculus `convex-optimization` (a draft stub).

## 9. References (Chicago 17th, notes-and-bibliography; every URL verified 2026-10-23 — arXiv and NeurIPS pages by title, DOIs via Crossref)

1. Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. "Efficient Estimation of Word Representations in Vector
   Space." arXiv:1301.3781, 2013. https://arxiv.org/abs/1301.3781. — paper; skip-gram (Def 1).
2. Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. "Distributed Representations of Words
   and Phrases and Their Compositionality." In *Advances in Neural Information Processing Systems* 26 (2013).
   https://arxiv.org/abs/1310.4546. — paper; negative sampling, P^{0.75} noise (Def 2).
3. Gutmann, Michael, and Aapo Hyvärinen. "Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized
   Statistical Models." In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and
   Statistics*, PMLR 9 (2010): 297–304. https://proceedings.mlr.press/v9/gutmann10a.html. — paper; NCE (Def 3).
4. Gutmann, Michael U., and Aapo Hyvärinen. "Noise-Contrastive Estimation of Unnormalized Statistical Models, with
   Applications to Natural Image Statistics." *Journal of Machine Learning Research* 13 (2012): 307–61.
   https://www.jmlr.org/papers/v13/gutmann12a.html. — paper; NCE (Def 3).
5. Mnih, Andriy, and Yee Whye Teh. "A Fast and Simple Algorithm for Training Neural Probabilistic Language Models."
   In *Proceedings of the 29th International Conference on Machine Learning* (2012). https://arxiv.org/abs/1206.6426.
   — paper; NCE for language models, self-normalization (Prop 2).
6. Mnih, Andriy, and Koray Kavukcuoglu. "Learning Word Embeddings Efficiently with Noise-Contrastive Estimation." In
   *Advances in Neural Information Processing Systems* 26 (2013).
   https://papers.nips.cc/paper_files/paper/2013/hash/db2b4182156b2f1f817860ac9f409ad7-Abstract.html. — paper; NCE
   embeddings (Prop 2).
7. Goldberg, Yoav, and Omer Levy. "word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding
   Method." arXiv:1402.3722, 2014. https://arxiv.org/abs/1402.3722. — paper; the objective (Def 2).
8. Levy, Omer, and Yoav Goldberg. "Neural Word Embedding as Implicit Matrix Factorization." In *Advances in Neural
   Information Processing Systems* 27 (2014).
   https://proceedings.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html. —
   paper; Thm 1.
9. Dyer, Chris. "Notes on Noise Contrastive Estimation and Negative Sampling." arXiv:1410.8251, 2014.
   https://arxiv.org/abs/1410.8251. — paper; NCE vs. negative sampling (Prop 2).
10. Levy, Omer, Yoav Goldberg, and Ido Dagan. "Improving Distributional Similarity with Lessons Learned from Word
    Embeddings." *Transactions of the Association for Computational Linguistics* 3 (2015): 211–25.
    https://doi.org/10.1162/tacl_a_00134. — paper; w + c, the eigenvalue weight, the dynamic window, and subsampling
    as count reweighting (Cor 2, §3 remark).
11. Collins, Michael, Sanjoy Dasgupta, and Robert E. Schapire. "A Generalization of Principal Components Analysis to
    the Exponential Family." In *Advances in Neural Information Processing Systems* 14 (2001).
    https://papers.nips.cc/paper_files/paper/2001/hash/f410588e48dc83f2822a880a68f78923-Abstract.html. — paper;
    exponential-family PCA (Thm 2).
12. Landgraf, Andrew J., and Jeremy Bellay. "word2vec Skip-Gram with Negative Sampling Is a Weighted Logistic PCA."
    arXiv:1705.09755, 2017. https://arxiv.org/abs/1705.09755. — paper; Thm 2.
13. Du, Simon S., Wei Hu, and Jason D. Lee. "Algorithmic Regularization in Learning Deep Homogeneous Models: Layers
    Are Automatically Balanced." In *Advances in Neural Information Processing Systems* 31 (2018).
    https://arxiv.org/abs/1806.00900. — paper; Thm 3.
14. Arora, Sanjeev, Nadav Cohen, and Elad Hazan. "On the Optimization of Deep Networks: Implicit Acceleration by
    Overparameterization." In *Proceedings of the 35th International Conference on Machine Learning* (2018).
    https://arxiv.org/abs/1802.06509. — paper; Thm 3 (balancedness under gradient flow).
15. Eckart, Carl, and Gale Young. "The Approximation of One Matrix by Another of Lower Rank." *Psychometrika* 1, no. 3
    (1936): 211–18. https://doi.org/10.1007/BF02288367. — paper; Cor 1.
16. Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "DeepWalk: Online Learning of Social Representations." In
    *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 701–10.
    New York: ACM, 2014. https://doi.org/10.1145/2623330.2623732. — paper; skip-gram on graphs (pointer).
17. Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation."
    In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1532–43. Doha: ACL,
    2014. https://aclanthology.org/D14-1162/. — paper; pointer to #8.
18. Bojanowski, Piotr, Edouard Grave, Armand Joulin, and Tomas Mikolov. "Enriching Word Vectors with Subword
    Information." *Transactions of the Association for Computational Linguistics* 5 (2017): 135–46.
    https://doi.org/10.1162/tacl_a_00051. — paper; fastText (pointer).
19. Turney, Peter D. "Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of
    Reviews." In *Proceedings of the 40th Annual Meeting of the ACL*, 417–24. Philadelphia: ACL, 2002.
    https://aclanthology.org/P02-1053/. — paper; SO (§6, via Topic 6).

## 10. Phase B notes

- **Numbering.** MDX at `src/content/topics/word2vec-implicit-factorization.mdx`, numbered in page order: Defs 1–4,
  Props 1–3, Thms 1–3, Lemma 1, Cors 1–2. The bias direction is an unnumbered corpus observation.
- **References.** `references[].type`: all `paper`.
- **Graph.** Flip the node's `status` to `published`, and remove the title from `planned` in `src/data/curriculum.ts`.
  Frontmatter: `prerequisites: [distributional-semantics-ppmi-svd, neural-language-models]`, `connections: []`.
- **Lab data.**
  - The lab reads `src/components/viz/data/sgns-lab.json`, written as
    `json.dumps(viz_payload(), separators=(',', ':'))`. Add `test_viz_mirror_matches_payload`.
  - Payload rounding: scores to 2 decimals in the fit scatter, 3 in Panel A, AUCs to 4.
  - `p_noise` keeps 6 significant figures so the curve can be redrawn exactly.
- **Notation.** Write 𝟙 as `\mathbf{1}` (Topic 5–6 convention), Hadamard powers as `^{\circ n}`, and "d + 2" in math.
  No bare `<` in prose; use `\lt` in math. Lexicon and pseudo-words go in italics, not KaTeX.
- **Display math** for Thm 2's identity, Prop 2's table of offsets, Thm 3's discrete step, and Lemma 1's
  factorization.
- **Mobile.** At 375 px the prose column is about 343 px, so the gauge table and the grid table need
  horizontal-scroll wrappers.
- **Overview hook:** the fit-by-band numbers (0.98 against 0.64) and "same loss, different neighbors" (0.901 balanced
  against 0.737 for a random gauge).
- **Back-links:**
  - Topic 6's Def 2 (SPPMI) from Thm 1;
  - Topic 6's Def 3 and Thm 2 (p) from Cor 2 (i), and its Cor 1 from Cor 2 (ii);
  - Topic 5's Cor 1 from Prop 1, and its Thm 6 from Prop 2.
- **Gate.** Re-run the harness at the B3 gate (≈ 44 s). The budget is the constraint: the notebook runs ≈ 47 s, under
  the 60 s limit with little room for more training.

## 11. Phase B record (2026-10-23)

- **Numbering and placement.**
  - Page numbering matches the brief: Defs 1–4, Props 1–3, Thms 1–3, Lemma 1, Cors 1–2, in page order (verified in
    the built DOM). The bias direction is unnumbered prose.
  - Citations moved out of theorem titles into the surrounding prose, as in Topics 5–6.
  - The lab, `SGNSLab`, sits after §5 under "One cell, and the factorization training picks".
- **Payload and component.**
  - `test_viz_mirror_matches_payload` requires `src/components/viz/data/sgns-lab.json` (16.4 kB) to equal
    `viz_payload()` exactly. The harness is 14/14 (≈ 44 s), and the notebook exits 0 (≈ 48 s).
  - The gauge view's scores and the chance and SVD readouts are rounded to 3 decimals in the payload, so the lab shows
    the page's numbers exactly (0.901, not a `toFixed` of 0.9005).
  - Count bands differ by point radius as well as color.
- **Cross-site.** All thirteen targets were re-verified `status: "published"` on GitHub `main`.
- **Gates.**
  - `pnpm test` passes 39/39, and `astro sync` and `pnpm validate` are clean.
  - `pnpm audit:cross-site` exits 0. The thirteen outward links appear as missing reciprocals, as expected until
    launch, and formalNLP has 0 deferred and 0 drifted targets.
  - `pnpm build` exits 0.
- **DOM checks** on the built page (preview pane shown, desktop width):
  - 0 `.katex-error`, 460 `.katex`, and `noindex, nofollow` intact.
  - The lab hydrates once `[data-lab="sgns"]` is scrolled into view.
  - Readouts equal the payload:
    - *plantline*: x* 0.662, weight 0.340, trained −0.913, b 0.52;
    - *fiscal*: 0.762 / 16.866 / 0.751, and at k = 15 x* −0.337 (= 0.762 − log 3);
    - *outlook*: "no minimum";
    - the epoch slider (driven by real key presses): epoch 0 has every score at 0 and loss 4.1589; epoch 40 has
      loss 1.6514 against the floor 1.4752, r 0.641 / 0.985, and gauge σ 0.788–1.204;
    - the gauge view: 0.886, 0.849, 0.901, 0.890, 0.737, against SVD 0.973.
  - The console shows no errors.
- **Mobile.**
  - At 375 px the gauge rows first overflowed (`scrollWidth` 435), because the AUC readout could not wrap in a grid
    cell. They are now flex rows that wrap.
  - `innerWidth === 375` and `scrollWidth` is 375 in every view; the lab is 343 px wide.
