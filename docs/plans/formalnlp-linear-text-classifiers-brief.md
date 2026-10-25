# Brief — `linear-text-classifiers`

**Linear Text Classifiers: Generative vs. Discriminative.** Roadmap #17, the first topic of the Tasks layer (domain
`classification-sentiment`). Its prerequisite `zipf-heaps-vocabulary-statistics` (#1) is published. Four topics depend on
it: `lexicon-sentiment-loughran-mcdonald`, `fine-tuning-low-rank-adaptation`, `sequence-labeling-hmm-crf`, and
`significance-testing-for-nlp`. It is not on the §6 watch-list, but Phase A found an overlap with formalML
`naive-bayes`; §7 states the scope against it.

**Phase A** (2026-10-25; approved by Jonathan with the recommended answers to §9: `intermediate`; keep every result;
lead with "no crossover when NB is the true model", then the bursty world; NB-weighted LR, with NBSVM cited). Results are numbered per type, as on the page: Definitions 1–2,
Propositions 1–4, Lemma 1, Theorems 1–4.
- **Harness:** 23/23 pass under numpy==2.5.4, scipy==1.18.1, and scikit-learn==1.9.1, also with warnings promoted to
  errors. The run takes ≈ 42 s, inside the 60 s budget. The four learning curves (≈ 22 s) and the finance bigram LR
  (≈ 5 s) dominate.
- **Notebook:** 21 cells; `jupyter execute` exits 0 in ≈ 57 s, including kernel start, and is committed without outputs.
- **Lab data:** `src/components/viz/data/linear-text-classifiers-lab.json`, 49 kB, pinned by
  `test_viz_mirror_matches_payload`.
- **Scope:** the agreed plan ("cite basics, prove gaps"; NB side of Ng–Jordan proved; separation with the λ → 0 path
  proved; tone with the XOR and bigrams), with four changes the prototypes forced:
  1. **There is no crossover where NB is the true model.** With both methods tuned, NB leads at every n, in both
     vocabulary sizes. The classic crossover needs NB to be misspecified while LR is well specified. **Prop 2
     (burstiness) is new:** it builds exactly that world, and there the crossover appears (n = 512 at V = 101, 2,048
     at V = 3,071).
  2. **Thm 3 is new.** Ng and Jordan's O(log V) is a statement about Bernoulli features with rates in [ρ₀, 1 − ρ₀]. A
     multinomial's rates sum to 1, so uniform relative accuracy costs Θ(1/θ_min) ≥ V tokens, and V^s log V under
     Zipf. This ties the page to its prerequisite.
  3. **The finance corpus never shows a crossover either.** LR beats NB at every training prefix from 30 to 1,500
     documents (`test_finance_has_no_crossover`). NB's trouble is thousands of uninformative Pitman–Yor words, which
     it cannot down-weight.
  4. **"NBSVM" became NB-weighted LR** (one-vs-rest log-count ratios, logistic loss, no interpolation). Prop 3(a) is
     the exact statement of what the weighting does. Wang and Manning's squared-hinge SVM and β-interpolation are
     cited, not reproduced.

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

**Simulated worlds** (binary tone; |S| = 71 informative words holding a share m_S = 0.062166 of every document; noise
words in corpus proportions; lengths from the corpus; 4 nested training draws; test 3,000 and validation 1,500 documents,
common random numbers across bursty/plain). NB's α ∈ {0.1, 0.3, 1, 3, 10, 30} and LR's λ ∈ {10⁻¹, 10⁻², 10⁻³} are chosen
per fit on the validation draw.

| World | V | Bayes acc. | NB asymptote | NB / LR at n = 4,096 | crossover n | Test |
|---|---|---|---|---|---|---|
| NB-true | 101 | 0.852667 | 0.852667 | 0.852167 / 0.847667 | none | `test_generative_discriminative_regimes` |
| bursty (r = 4 on half) | 101 | 0.852667 | 0.820000 | 0.816667 / 0.849167 | 512 | same |
| NB-true | 3,071 | 0.855000 | 0.855000 | 0.831000 / 0.811167 | none | same |
| bursty | 3,071 | 0.855000 | 0.815000 | 0.771750 / 0.790250 | 2,048 | same |

Mean accuracy by n (8, 16, …, 4,096), from the payload (3 decimals):
- V = 101 NB-true: NB 0.561 … 0.852; LR 0.495 … 0.848.
- V = 101 bursty: NB 0.548 … 0.817; LR 0.503 … 0.849.
- V = 3,071 NB-true: NB … 0.831; LR … 0.811.
- V = 3,071 bursty: NB … 0.772; LR … 0.790.

The validated NB α (median over reps) at V = 3,071 NB-true grows with n: 0.55, 0.1, 0.2, 0.65, 1, 2, 2, 6.5, 3, 20. At
V = 101 it stays at 0.1–0.65. This is asserted by `test_thm3_…`: α at 4,096 exceeds α at 64.

**Ng–Jordan world** (Bernoulli NB true, V = 200, rates clipped to [0.1, 0.9], ε = 0.05, δ = 0.2, 40 reps; tests
`test_lemma1_concentration`, `test_thm2_margin_bound`):

| Quantity | Value |
|---|---|
| m₀ = ⌈ln(4V/δ)/(2ε²)⌉ per class | 1,659 |
| share of reps with a deviation > ε | 0.000000 |
| 0.8-quantile of the largest deviation | 0.039549 |
| largest observed \|l̂ − l_∞\| | 2.151747 |
| Thm 2's τV = 4(V + 1)ε/ρ₀ | 402.000000 |
| G(τ) (vacuous: equals the whole correct mass, 1 − Bayes error) | 0.800500 |
| median \|l_∞\| | 1.629790 |
| m₀ needed for τV = 1 | 268,070,320 |

**Thm 3, long-tailed world** (V = 3,071; ε = 1/2, δ = 0.1; test `test_thm3_relative_accuracy_costs_one_over_theta_min`):

| Quantity | Value |
|---|---|
| θ_min | 4.76898·10⁻⁶ |
| tokens per class Thm 3(a) asks for | 27,743,023 |
| tokens per class 4,096 documents supply | 436,509 |
| the same count if θ_min were 1/V | 406,311 |
| the same count for Zipf(1.1) on 3,071 words | 5,536,598 |
| expected words unseen at 4,096 documents | 66.560808 |
| share of test tokens from words seen < 5 times, n = 64 / 512 / 4,096 | 0.104256 / 0.028054 / 0.001822 |

**Separation** (the 200 polar documents of `generate_corpus(300, 0)`, V = 2,738, homogeneous model; tests
`test_separable_means_no_mle` and `test_l2_path_turns_toward_max_margin`):
- The rows have full rank 200. The real labels and three random labelings are all LP-separable.
- SVM normalized margin γ* = 0.692322, so ln 10/γ* = 3.326.
- L2 path, λ = 10⁻¹ … 10⁻⁷:
  - ‖ŵ_λ‖: 1.541, 3.714, 6.291, 9.066, 11.969, 14.958, 18.005. The last decade adds 3.047, which is 0.92 of ln 10/γ*.
  - cos(ŵ_λ, ŵ_SVM): 0.930, 0.979, 0.990, 0.994, 0.996, 0.997, 0.998146.
  - normalized margin: 0.032 … 0.645, below γ* = 0.692.
- Gradient descent at η = 1/β from 0, t = 10 … 10⁵:
  - cos: 0.591, 0.629, 0.715, 0.816, 0.906, 0.951, 0.973, 0.982, 0.987307.
  - loss: 0.661 … 0.0025.

**Funnel** (the training corpus's 1,000 polar documents, V = 6,307): the largest |r_j| among once-seen words at
α = 0.1 / 1 / 10 is 2.407905 / 0.702653 / 0.101635. Each equals Prop 3(b)'s bound, which is attained.

**Finance tone** (train `generate_corpus(1500, 0)`, dev `(600, 2)`, test `(1500, 1)`, three classes, min_df = 2;
unigram V = 3,246, bigram V = 17,706; tests `test_finance_tone`, `test_label_noise`, `test_aspect_xor`,
`test_bayes_ceiling`):

| Method | features | acc. | mean conf. | ECE₁₅ | chosen hp |
|---|---|---|---|---|---|
| multinomial NB | unigram | 0.600667 | 0.898592 | 0.301125 | α = 3 |
| multinomial NB | bigram | 0.694000 | 0.962488 | 0.269633 | α = 3 |
| softmax LR | unigram | 0.738667 | 0.779815 | 0.046209 | λ = 10⁻² |
| softmax LR | bigram | 0.771333 | 0.868700 | 0.099164 | λ = 10⁻³ |
| NB-weighted LR | unigram | 0.718667 | 0.728153 | 0.028503 | λ = 10⁻² |
| NB-weighted LR | bigram | 0.776000 | 0.795517 | 0.026709 | λ = 10⁻² |

- Full-text Bayes accuracy: **0.889919**, by exact enumeration, with Monte Carlo agreeing to 0.01.
- Training prefixes n = 30 / 100 / 300 / 1,000 / 1,500 (unigrams, each method at its full-data hyperparameter):

  | Method | 30 | 100 | 300 | 1,000 | 1,500 |
  |---|---|---|---|---|---|
  | NB | 0.324000 | 0.437333 | 0.516667 | 0.602000 | 0.600667 |
  | LR | 0.460000 | 0.514667 | 0.620000 | 0.723333 | 0.738667 |
- Label noise 0 / 0.15 / 0.3 (unigram features):

  | Method | 0 | 0.15 | 0.3 |
  |---|---|---|---|
  | NB | 0.600667 | 0.526000 | 0.510000 |
  | LR | 0.738667 | 0.660000 | 0.584000 |
  | NB-weighted LR | 0.718667 | 0.688667 | 0.598000 |

- Moment matching on the unigram fit:
  - LR's largest co-count residual is 4.5·10⁻⁸ and its largest class-total residual 8.3·10⁻¹⁰.
  - NB's class totals miss by (0.42, −22.32, +21.89) documents of 1,500, and its co-counts by up to 268.
- LR weight, positive minus negative, as unigram / bigram fit:

  | Feature | unigram | bigram |
  |---|---|---|
  | "rose" | 0.060303 | 0.083205 |
  | "fell" | 0.024196 | −0.014213 |
  | "costs rose" | — | −0.354523 |
  | "costs fell" | — | 0.526164 |
  | "taxes rose" | — | −0.351073 |
  | "sales rose" | — | 0.265152 |
  | "not strong" | — | −0.198898 |
  | "not weak" | — | 0.335904 |
  | "strong" | 0.664745 | 0.687090 |
  | "weak" | −1.081776 | −1.060478 |

Notes for the prose:
- The bigram model's largest weights are "⟨adjective⟩ but" ("disappointing but" −0.59 in the spike). In a contrast
  sentence the first clause carries the sentence's polarity, so the bigram marks which adjective counts. This is
  structure beyond the XOR, worth one sentence.
- LR on bigrams has more features than documents (17,706 > 1,500), so its training set is nearly separable (Thm 4).
  Thm 1's training moments are then optimistic, and on the test set it is overconfident (0.869 vs 0.771). This is the
  bridge from Thm 1 to Thm 4.
- NB-weighted LR's one-vs-rest scores are pushed through a softmax for the ECE. Read its calibration as a score's, not
  a posterior's.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Linear Text Classifiers: Generative vs. Discriminative |
| `{{SUBTITLE}}` | Naive Bayes and logistic regression draw the same kind of boundary through word-count space and disagree only about where to put it. We prove they share a hypothesis class and match different moments, derive how fast each learns (and why Zipf's long tail breaks the generative shortcut), watch the logistic MLE run off to infinity on separable text while its direction settles on the maximum-margin line, and find the one thing no unigram model of financial tone can see: that "costs rose" is bad news. |
| `{{SLUG}}` | `linear-text-classifiers` |
| `{{DOMAIN}}` | `classification-sentiment` |
| `{{DIFFICULTY}}` | `intermediate` (see §9) |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[zipf-heaps-vocabulary-statistics]` (graph edge). Connections: `n-gram-models-smoothing` (additive smoothing), `language-model-calibration` (ECE, overconfidence) |
| `{{FORMALML_PREREQS}}` | `naive-bayes`, `vc-dimension`, `convex-analysis` |
| `{{FORMALML_CONNECTIONS}}` | `high-dimensional-regression`, `generalization-bounds`, `gradient-descent` |
| `{{STATS_PREREQS}}` | `generalized-linear-models`, `maximum-likelihood`, `multivariate-distributions` |
| `{{STATS_CONNECTIONS}}` | `regularization-and-penalized-estimation` |
| `{{FORMALRAG_CONNECTIONS}}` | `bm25-binary-independence-model`, `query-likelihood-language-models` |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/zipf-heaps-vocabulary-statistics/` (imported as `zh`), `notebooks/finance-corpus/` |

## 1. Section outline (geometric first)

1. **Two ways to draw one line.** Start with a picture: 2-D count space for two words ("strong", "weak"), the NB
   boundary and the LR boundary as two lines. Then the claim of the page: same family of lines, different rules for
   placing them.
2. **One hypothesis class.** Def 1 (the count vector) and Prop 1: NB is softmax-linear, and LR's family is NB's
   plus one length feature. Def 2: the generative–discriminative pair.
3. **Two estimating equations.** Thm 1: convexity, the Kronecker Hessian with its null space, and moment matching. LR
   matches co-counts and class totals; NB does not, and it misses its own class totals by 22 documents.
4. **How fast.** Lemma 1 and Thm 2 (Ng–Jordan's generative side, proved), with the honest table: τV = 402 nats
   against an observed 2.15. The discriminative Ω(V) is cited, through formalML's VC dimension of half-spaces.
5. **Multinomials and the Zipf tail.** Thm 3 says relative accuracy costs 1/θ_min, which ties back to Topic 1. Then the
   validated α growing with n, and the rare-token share.
6. **Burstiness and the crossover.** Prop 2, then lab view 1: NB-true worlds show no crossover; bursty worlds show
   one. Ng and Jordan's two regimes appear exactly where their asymptotic errors differ.
7. **Separation.** Thm 4 (no MLE; full row rank ⇒ every labeling separable; the λ → 0 direction is the max margin,
   proved), with GD's implicit bias cited (Soudry et al.). Lab view 3.
8. **NB features as a prior.** Prop 3: NB weighting is a per-word ridge, and smoothing bounds rare words' ratios.
   Lab view 2, the funnel.
9. **Finance case study** (`<FinanceCaseStudy>`):
   - (a) The aspect XOR (Prop 4) and the weights that show it.
   - (b) The method × feature table against the 0.890 ceiling, NB's overconfidence, bigram LR's overconfidence,
     and label noise. Lab view 4.
10. **Where this sits.** Complement NB and tf transforms (Rennie et al.), lexicons next (Topic 18), CRFs (Topic 21),
    significance of accuracy gaps (Topic 28), fine-tuned encoders (Topic 20).

## 2. Statements and proofs

Notation: documents i = 1…n with count vectors x_i ∈ ℕ^V, lengths L_i = Σ_j x_ij, labels y_i ∈ {1, …, K} (one-hot
y_ik). θ_kj is P(word j | class k), π_k = P(y = k). Logs are natural.

**Definition 1 (bag of words).** A document's count vector x has x_j = the number of occurrences of feature j (a word,
or with bigrams also an adjacent pair). Its binarization is x̂_j = 1{x_j > 0}.

**Definition 2 (generative–discriminative pair; Ng and Jordan 2002).**
- A generative classifier fits p(x, y) and predicts argmax_y p(y | x).
- Its discriminative partner fits p(y | x) directly, in the parametric family the generative model induces.
- Multinomial NB and softmax LR on counts are such a pair (Prop 1).

**Proposition 1 (NB is softmax-linear).**
- (a) Multinomial NB gives log P(y = k | x) = log softmax_k(Wx + b), with W_kj = log θ_kj and b_k = log π_k.
  Bernoulli NB gives the same on x̂, with W_kj = logit θ_kj and b_k = log π_k + Σ_j log(1 − θ_kj).
- (b) For every (W, b) and every length L, there is a multinomial NB whose posterior equals softmax(Wx + b) on all
  documents of length L. Across lengths,

  softmax(Wx + b) = softmax(W̃x + b + L(x) λ̄), with W̃_k = W_k − λ̄_k and λ̄_k = log Σ_j e^{W_kj}.

  So LR's family is NB's posteriors with one extra feature, the length L(x). It is strictly larger: with V = 1,
  every NB posterior is constant in x, and softmax(Wx + b) need not be.

*Proof.*
- (a) By Bayes, log P(y = k | x) = log π_k + log P(x | y = k) − log P(x).
  - Multinomial: log P(x | k) = log(L!/∏x_j!) + Σ_j x_j log θ_kj. The coefficient does not depend on k, so it cancels
    in the normalization.
  - Bernoulli: log P(x̂ | k) = Σ_j [x̂_j log θ_kj + (1 − x̂_j) log(1 − θ_kj)] = Σ_j x̂_j logit θ_kj + Σ_j log(1 − θ_kj).
- (b) Let θ_k = softmax(W_k), so W_k = log θ_k + λ̄_k 1, and W_k·x = Σ_j x_j log θ_kj + L(x) λ̄_k. On length-L
  documents, put log π_k ∝ b_k + L λ̄_k. Then the NB posterior is softmax(Wx + b).
- When lengths vary, the term L(x) λ̄_k is a feature, not a bias. With V = 1, θ_k1 = 1 for every class, so NB's
  posterior is π. ∎

Tests: `test_nb_is_softmax_linear` (the literal Bayes rule, to 10⁻⁹), `test_nb_matches_scikit_learn` (10⁻⁸), and
`test_fixed_length_families_coincide`. The Bernoulli weight is the RSJ relevance weight (formalRAG
`bm25-binary-independence-model`, Thm 1), which is cited.

**Theorem 1 (two estimating equations).** Let F(W, b) = (1/n)Σ_i [log Σ_k e^{z_ik} − z_{i,y_i}] + (λ/2)‖W‖², with
z_i = Wx_i + b, p_i = softmax(z_i), and x̃_i = (x_i, 1).
- (a) F is convex. Its Hessian in (W, b) is

  (1/n)Σ_i (diag p_i − p_i p_iᵀ) ⊗ x̃_i x̃_iᵀ + λ(I_K ⊗ diag(1, …, 1, 0)).

  At λ = 0 the null space contains every 1_K ⊗ v, v ∈ ℝ^{V+1}, which has dimension V + 1. At λ > 0 it contains
  exactly the common intercept shift 1_K ⊗ e_{V+1}, when the p_i are interior and the x̃_i span ℝ^{V+1}.
- (b) At any minimizer,

  Σ_i (y_ik − p_ik) x_ij = nλ W_kj for all k and j, and Σ_i p_ik = n_k for all k.

  At λ = 0, LR's fitted model reproduces every class × word co-count and every class total.
- (c) Multinomial NB's MLE instead solves θ̂_kj = Σ_{i: y_i = k} x_ij / Σ_{i: y_i = k} L_i. It matches each class's
  word totals, and in general it satisfies neither equation in (b).

*Proof.*
- (a)
  - log-sum-exp is convex with Hessian diag p − ppᵀ, and z_i is linear in θ = (W, b) with z_i = (I_K ⊗ x̃_iᵀ)θ, so the
    chain rule gives the Kronecker form. Each term is PSD.
  - Null space at λ = 0: (diag p − ppᵀ)1 = 0, so 1 ⊗ v is in every term's null space.
  - Null space at λ > 0: for a direction (U, c) the quadratic form is ≥ λ‖U‖², which is zero only if U = 0. It then
    reduces to (1/n)Σ_i cᵀ(diag p_i − p_i p_iᵀ)c, zero iff c ∝ 1 when the p_i are interior.
- (b) ∂F/∂W_kj = (1/n)Σ_i (p_ik − y_ik) x_ij + λW_kj and ∂F/∂b_k = (1/n)Σ_i (p_ik − y_ik). Set both to zero.
- (c) NB maximizes the joint likelihood Σ_i [log π_{y_i} + Σ_j x_ij log θ_{y_i j}] under Σ_j θ_kj = 1. The
  Lagrangian's stationarity is the stated ratio.
- On the finance unigrams NB's class totals miss by 22.3 documents. ∎

Tests:
- `test_gradient_and_hessian`: the gradient, Hessian, and Hessian-vector product checked by finite differences, PSD,
  and the nullity at V + 1 and at 1.
- `test_lr_matches_scikit_learn`: our objective is ≤ scikit-learn's, with probabilities within 10⁻⁴.
- `test_estimating_equations`: residuals ≤ 10⁻⁶. At λ = 0 on the 10 most frequent words (not separable) every
  co-count is matched.

Calibration-in-the-large is the k-equation of (b), a first-moment cousin of Topic 16's Thm 3. formalStatistics
`generalized-linear-models` Thm 1 is the binary canonical-link case, and is cited.

**Lemma 1 (uniform concentration; Ng and Jordan 2002, Lemma 3, made explicit).** Take V binary features and a class
b with m_b ≥ m₀ = ⌈ln(4V/δ)/(2ε²)⌉ training documents, for every b ∈ {0, 1}.
- (a) Conditional on the labels, with probability ≥ 1 − δ every empirical rate satisfies
  |p̂(x_j = 1 | b) − p(x_j = 1 | b)| ≤ ε.
- (b) If m ≥ ln(4/δ)/(2ε²), then |π̂ − π| ≤ ε with probability ≥ 1 − δ.
- (c) If moreover ρ₀ ≤ π ≤ 1 − ρ₀, then m ≥ m₀/(ρ₀ − ε) documents give every class at least m₀ on that event.
- (d) Add-l smoothing moves each rate by at most l/m_b.

So m = O(ε⁻² log(V/δ)) documents suffice for fixed ρ₀.

*Proof.*
- (a) p̂_jb is a mean of m_b i.i.d. Bernoulli(p_jb). Hoeffding gives P(|p̂_jb − p_jb| > ε) ≤ 2e^{−2ε²m_b}, and the
  union bound over the 2V pairs (j, b) gives ≤ 4V e^{−2ε²m₀} ≤ δ.
- (b) Hoeffding on the labels.
- (c) On (b)'s event, m_b ≥ (π_b − ε)m ≥ (ρ₀ − ε)m.
- (d) |(c + l)/(m_b + 2l) − c/m_b| ≤ l/m_b for 0 ≤ c ≤ m_b. ∎

**Theorem 2 (NB's error; Ng and Jordan 2002, Thm 4, with explicit constants).** Suppose every true rate p(x_j = 1 | b)
and π lie in [ρ₀, 1 − ρ₀], and every estimate is within ε ≤ ρ₀/2. Let l(x) = Σ_j log[p(x_j | 1)/p(x_j | 0)] +
log[π/(1 − π)] be the population discriminant, with sign rule h_∞, and l̂, ĥ the plug-in versions.
- (a) |l̂(x) − l(x)| ≤ τV with τV = 4(V + 1)ε/ρ₀, for every x.
- (b) ε(ĥ) ≤ ε(h_∞) + G(τ), where G(τ) = P[(l(x) ∈ [0, τV] ∧ y = 1) ∨ (l(x) ∈ [−τV, 0] ∧ y = 0)].

With Lemma 1, ε = O(√(log(V/δ)/m)) and so τ = O(√(log V/m)), Ng and Jordan's form. If G(τ) ≤ ε₀/2 + F(τ) with
F(τ) → 0, then m = O(log V) documents bring NB within ε₀ of its asymptotic error (their Cor 6, which we cite).

*Proof.*
- (a) Each true probability is ≥ ρ₀ (for x_j = 1 and for x_j = 0) and each estimate is within ε ≤ ρ₀/2. By the mean
  value theorem, |log â − log a| ≤ |â − a|/min(â, a) ≤ ε/(ρ₀/2).
- The discriminant has 2V feature log terms and 2 prior log terms, so |l̂ − l| ≤ (2V + 2)(2ε/ρ₀).
- (b) If ĥ errs where h_∞ is right, then sign l̂ ≠ sign l, so |l(x)| ≤ |l̂(x) − l(x)| ≤ τV. Also h_∞ was right,
  which is G's event. ∎

**Cited, not proved:**
- The discriminative side (Ng and Jordan's Prop 2): with VC dimension V + 1 for half-spaces (formalML `vc-dimension`),
  ε(h_Dis) ≤ ε(h_Dis,∞) + O(√((V/m) log(m/V))), so m = Ω(V) in the worst case.
- Their Prop 1: ε(h_Dis,∞) ≤ ε(h_Gen,∞).
- Both concern h_Dis as the **0-1 training-error minimizer** over half-spaces. Their own paper calls the logistic MLE
  "logistic regression" by abuse of terminology. The rigorFlag says so.

Tests:
- `test_lemma1_concentration`: m₀ = 1,659, and failures 0 ≤ δ = 0.2.
- `test_thm2_margin_bound`: the pointwise bound (a) and the inclusion (b) on every rep.
- The same test asserts the looseness: the largest observed gap is 2.15 against 402, G(τ) is vacuous, and τV = 1 needs
  2.7·10⁸ documents per class.
- The bound adds V errors as if aligned. For independent errors the sum grows like √V; that is a page remark, not a
  claim.

**Theorem 3 (multinomial rates and the Zipf tail).** Let a class's training tokens be N i.i.d. draws from θ ∈ Δ^V, with
counts C_j, and θ̂ = C/N. Let ε ∈ (0, 1/2].
- (a) If N ≥ 3 ln(2V/δ)/(ε²θ_min), then with probability ≥ 1 − δ every |θ̂_j/θ_j − 1| ≤ ε. Then every
  |log θ̂_j − log θ_j| ≤ 2ε, and for two such classes a length-L document's discriminant moves by at most 4εL.
- (b) If N ≤ 1/(2θ_min), the rarest word is unseen with probability ≥ 1/2, so its relative error is 1. Uniform relative
  accuracy needs N > 1/(2θ_min) ≥ V/2 tokens.
- (c) Under Zipf(s), θ_min = V^{−s}/H_{V,s}, so (a)'s count grows like V^s H_{V,s} log V: polynomial in V, not
  logarithmic.

*Proof.*
- (a) Multiplicative Chernoff gives P(|C_j − Nθ_j| ≥ εNθ_j) ≤ 2e^{−ε²Nθ_j/3}. A union bound over V words, each with
  θ_j ≥ θ_min, gives ≤ 2V e^{−ε²Nθ_min/3} ≤ δ.
- Then −log(1 − ε) ≤ ε/(1 − ε) ≤ 2ε and log(1 + ε) ≤ ε. The discriminant Σ_j x_j(log θ̂_1j − log θ̂_0j) moves by at
  most Σ_j x_j · 4ε = 4εL.
- (b) P(C_min = 0) = (1 − θ_min)^N ≥ 1 − Nθ_min ≥ 1/2.
- (c) θ_min = V^{−s}/H_{V,s}, which is Topic 1's `zipf_pmf`. ∎

The rarest words occur rarely in test documents too, so (a) is sufficient and far from necessary.
- The share of test tokens whose word was seen fewer than 5 times falls from 0.104 at n = 64 to 0.0018 at 4,096.
- What (c) does predict is that the long tail keeps NB's smoothing from settling: the validated α climbs from ≈ 0.5 to
  20 with n at V = 3,071 and stays near 0.2 at V = 101.

Test: `test_thm3_relative_accuracy_costs_one_over_theta_min`, including a Monte Carlo check of (a) on a 50-word
Zipf.

**Proposition 2 (burstiness).** Draw u ~ Multinomial(L, θ_y) with L independent of y, and observe x = r ⊙ u for fixed
r_j ≥ 1.
- (a) P(y | x) = softmax over y of Σ_j (x_j/r_j) log θ_yj + log π_y. This is linear in x, so softmax LR contains the
  truth, and its population 0-1 error is the Bayes error.
- (b) Multinomial NB fit to infinitely many such documents converges to θ̃_k = r ⊙ θ_k/Z_k, with Z_k = Σ_j r_jθ_kj.
  Its weight per observed count is log(θ_1j/θ_0j) + log(Z_0/Z_1). Per underlying occurrence that is r_j times the
  Bayes weight, plus a length term. So NB's asymptotic error generally exceeds the Bayes error.

*Proof.*
- (a) u = x/r is a function of x, so P(y | x) = P(y | u), which Prop 1(a) writes as stated.
- (b) By the law of large numbers, θ̂_kj → E[x_j | k]/E[Σ_j x_j | k] = r_jθ_kj E[L]/(E[L] Z_k). ∎

In the simulated worlds, half the words have r_j = 4. The NB asymptote is 0.820 against a Bayes accuracy of 0.853 at
V = 101, and 0.815 against 0.855 at V = 3,071.

Tests: `test_bursty_world_keeps_lr_well_specified` (the identity to 10⁻¹² and the gap ≥ 0.02) and
`test_generative_discriminative_regimes`.

**Theorem 4 (separation).** Let z_i = s_i x_i with s_i ∈ {±1}, ℓ(w) = (1/n)Σ_i log(1 + e^{−z_i·w}), and
F_λ = ℓ + (λ/2)‖w‖².
- (a) If some w̄ has z_i·w̄ > 0 for all i, then inf ℓ = 0 is not attained. ∇ℓ(w)·w̄ < 0 at every w, so ℓ has no
  stationary point. For λ > 0, F_λ has a unique minimizer ŵ_λ.
- (b) If the x_i are linearly independent (so n ≤ V), every labeling s is separable.
- (c) Let ŵ = argmin ‖w‖² s.t. z_i·w ≥ 1, the hard-margin SVM; it is unique. Let u* = ŵ/‖ŵ‖ and γ* = 1/‖ŵ‖. Then, as
  λ → 0:
  - ‖ŵ_λ‖ → ∞;
  - min_i z_i·ŵ_λ/‖ŵ_λ‖ → γ*;
  - ŵ_λ/‖ŵ_λ‖ → u*.

  This is Rosset, Zhu, and Hastie (2004, NIPS 16), Thm 2.1, at p = 2 for the logistic loss.

*Proof.*
- (a)
  - ℓ(tw̄) → 0 as t → ∞, and ℓ > 0 everywhere.
  - ∇ℓ(w)·w̄ = −(1/n)Σ_i σ(−z_i·w) z_i·w̄ < 0.
  - F_λ is strictly convex and coercive.
- (b) X: ℝ^V → ℝ^n is onto, so Xw = s has a solution, and then z_i·w = s_i² = 1.
- (c) Write ŵ_λ = ρ_λu_λ with ‖u_λ‖ = 1 and γ_λ = min_i z_i·u_λ.
  - *ρ_λ → ∞.* Fix R. Then F_λ ≥ c_R := min_{‖w‖ ≤ R} ℓ(w) > 0 on the ball. Outside it,
    F_λ(R′u*) ≤ e^{−R′γ*} + λR′²/2, which is < c_R for R′ large and λ small.
  - *Compare at equal norm.* F_λ(ŵ_λ) ≤ F_λ(ρ_λu*) and the penalties are equal, so ℓ(ρ_λu_λ) ≤ ℓ(ρ_λu*) ≤
    e^{−ρ_λγ*}.
  - *The margin is eventually positive.* If γ_λ < 0, some term of ℓ(ρ_λu_λ) is ≥ log 2/n, which exceeds e^{−ρ_λγ*}
    once ρ_λ is large. So γ_λ ≥ 0 eventually.
  - *Lower bound on the margin.* log(1 + e^{−m}) ≥ e^{−m}/2 for m ≥ 0, so (1/2n)e^{−ρ_λγ_λ} ≤ e^{−ρ_λγ*}. Hence
    γ_λ ≥ γ* − ln(2n)/ρ_λ → γ*.
  - *Direction.* Every limit point ū of u_λ, on the compact sphere, has min_i z_i·ū ≥ γ*. So ū is a max-margin
    direction, and ū = u* by uniqueness. ∎

**Cited, not proved:**
- GD from 0 with step η < 2/β has w(t) = ŵ log t + ρ(t) with ‖ρ(t)‖ = O(log log t). Its direction converges at rate
  O(log log t/log t), and O(1/log t) for almost every dataset (Soudry et al. 2018, Thms 3 and 5; homogeneous model;
  with an intercept GD finds argmin ‖w‖² + b² instead).
- The empirical ‖ŵ_λ‖ ≈ ln(1/λ)/γ* (the last decade adds 0.92·ln 10/γ*) is an observation, not a claim. The
  constrained-path version is RZH (JMLR 2004, §5 Thm 3).

Tests: `test_separable_means_no_mle` and `test_l2_path_turns_toward_max_margin`.

**Proposition 3 (NB features as a prior).**
- (a) Let r_j ≠ 0. Binary LR on features r ⊙ x with penalty (λ/2)‖v‖² has the same decision function as LR on x with
  penalty (λ/2)Σ_j w_j²/r_j². The map is w = r ⊙ v.
- (b) For r_j = log[(c_1j + α)/(N_1 + αV)] − log[(c_0j + α)/(N_0 + αV)], with c_j = c_0j + c_1j:

  |r_j| ≤ ln((c_j + α)/α) + |ln((N_1 + αV)/(N_0 + αV))|.

  The bound is attained by words seen once in one class.

*Proof.*
- (a) The change of variables is a bijection that leaves every margin x·w unchanged and maps ‖v‖² to Σ w_j²/r_j².
- (b) α/(c_j + α) ≤ (c_1j + α)/(c_0j + α) ≤ (c_j + α)/α, since 0 ≤ c_kj ≤ c_j. ∎

So weak NB evidence (small |r_j|) becomes a strong prior toward 0, and add-α smoothing caps what a rare word can say.
On the funnel the once-seen maximum is 2.408, 0.703, and 0.102 at α = 0.1, 1, and 10, each equal to its bound.
Wang and Manning's NBSVM (squared hinge, β = 0.25 interpolation toward the scalar mean magnitude) is the source and is
cited.

Tests: `test_nb_weighting_is_a_feature_ridge` (10⁻⁷) and `test_smoothing_bounds_log_count_ratios`.

**Proposition 4 (the aspect XOR).** On the four one-clause documents "revenue rose" (+), "revenue fell" (−), "costs
rose" (−), and "costs fell" (+), no (w, b) separates the labels in unigram counts. Bigram indicators separate them.

*Proof.*
- Suppose (w, b) separates them.
- The two positive documents give w_revenue + w_rose + b > 0 and w_costs + w_fell + b > 0. Their sum is
  S = w_revenue + w_costs + w_rose + w_fell + 2b > 0.
- The two negative documents give w_revenue + w_fell + b < 0 and w_costs + w_rose + b < 0, whose sum is the same S < 0.
  This is a contradiction.
- With bigram indicators, put weight ±1 on each document's own bigram. ∎

On the corpus, LR's unigram weights for "rose" and "fell" stay near 0 (0.060 and 0.024), while the bigram fit puts
−0.355 on "costs rose" and +0.526 on "costs fell".

Tests: `test_aspect_xor` (LP infeasible and feasible; the corpus signs).

## 3. Corpus and finance measurements (§9)

- **Full-text ceiling.** Each document has k ~ Uniform{5, …, 8} aspect sentences, with polarities i.i.d. from
  `_SENTENCE_TONE_MIX[tone]`. Nothing else in a document depends on its tone: the facts, distractors, hedges, and
  closers are drawn independently, and each sentence's form depends only on its own polarity. So the counts of
  (+, 0, −) sentences are sufficient, and the Bayes accuracy is Σ_k ¼ Σ_counts max_t P(counts | t, k)/3 = 0.889919.
  This assumes a reader who recovers each sentence's polarity, which the templates make deterministic.
- **The table** (§3a). Bigrams help every method. LR beats NB by 0.138 on unigrams and 0.077 on bigrams. NB-weighted
  LR on bigrams is best (0.776) and best calibrated (ECE 0.027). Nothing comes within 0.11 of the ceiling: no
  bag-of-n-grams model parses a contrast sentence or a "not ⟨adj⟩ this period".
- **Calibration.**
  - NB is overconfident by 0.30 on unigrams, because it double-counts correlated template words (Niculescu-Mizil and
    Caruana 2005).
  - LR is nearly calibrated on unigrams (ECE 0.046), and overconfident on bigrams, where V > n.
- **Label noise.** At noise 0.3, NB loses 0.091 (to 0.510) and LR loses 0.155 (to 0.584). LR stays ahead at every
  level, and NB-weighted LR degrades least at 0.15.
- **EDGAR** has no tone labels (`tone=None`), so it enters no accuracy claim.

## 4. The lab (`LinearTextClassifiersLab`, `data/linear-text-classifiers-lab.json`)

Payload:
- `curves.n` (10 sizes) and `curves.worlds` (4: V ∈ {101, 3071} × {NB-true, bursty}). Each world has `bayes`,
  `nb_inf`, `crossover`, and per method `mean`/`lo`/`hi` (min–max over 4 reps).
- `funnel`: 571 words (S plus 500 sampled) with `words`, `count`, `informative`, `r[α]`, `bound[α]` for
  α ∈ {0.1, 1, 10}, and `lr[λ]` for λ ∈ {10⁻², 10⁻³, 10⁻⁴}.
- `separation`: `svm_margin`, `slope`, `path[7]`, `gd[9]`.
- `finance`: `ceiling`, `methods[6]` (acc, confusion, conf, ece, hp, V), `noise`, and `xor[10]`.

Every printed number is a payload value, rounded in `viz_payload()`.

- **View 1, Crossover.**
  - Controls: a V toggle (101 / 3,071) and a world toggle (NB true / bursty).
  - Chart: accuracy against n on a log axis, NB and LR mean lines with min–max bands, a dashed Bayes line, a dotted
    NB-asymptote line (it coincides with Bayes in NB-true worlds), and a crossover marker.
  - Readouts: Bayes, the NB asymptote, the crossover n (or "none"), and both accuracies at a hovered n.
  - Lesson: no crossover when NB is right; with bursts, NB wins early and LR passes it, then passes NB's asymptote.
- **View 2, the funnel.**
  - Controls: a toggle between NB's log-count ratio r_j and LR's weight, with an α or λ selector.
  - Chart: a scatter of the weight against word count on a log axis, informative words in oxblood and the rest muted,
    with the curve ±(ln((c + α)/α) + const) when r is shown; hover shows the word.
  - Lesson: rare words get NB's most extreme weights unless α pulls them in; LR's penalty flattens the whole noise
    cloud.
- **View 3, Separation.**
  - Charts: the path's ‖ŵ_λ‖ against λ (log, with a reference line of slope ln 10/γ*), cos to the SVM for both the path
    (against λ) and GD (against t, log), and the normalized margin approaching γ*.
  - Lesson: the norm runs away, while the direction settles on the max-margin direction.
- **View 4, Finance tone.**
  - A 3 × 2 grid of accuracy (method × features) with the ceiling marked; clicking a cell shows its confusion matrix,
    confidence, and ECE.
  - A small line chart of accuracy against label noise.
  - A table of the XOR weights (unigram vs bigram).
  - Lesson: the bigram fixes the XOR, and LR is not NB's equal at any noise level.

## 5. rigorFlag (draft)

> Ng and Jordan's generative bound is proved here as they state it, for binary features with every rate in
> [ρ₀, 1 − ρ₀]; with explicit constants it is vacuous at any realistic sample size (τV = 402 nats against observed gaps
> near 2). Their discriminative results concern the 0-1 training-error minimizer over half-spaces, not the logistic
> MLE, and we cite rather than prove them. Neither regime is universal: Xue and Titterington (2008) find the two-regime
> picture unreliable on real data, and in our finance corpus there is no crossover at all — logistic regression wins
> at every size we tried. The crossover we show lives in simulated worlds built for it: one where naive Bayes is the
> true model (no crossover) and one with planted burstiness (a crossover). Every hyperparameter in those worlds is
> tuned on a fresh validation draw, a luxury real data does not offer. Theorem 3's token count is sufficient, not
> necessary. Theorem 4's λ → 0 limit is Rosset, Zhu, and Hastie's; gradient descent's implicit bias is Soudry et al.'s,
> checked here only numerically. The finance numbers come from a synthetic corpus with planted structure and a known
> Bayes accuracy (0.890) that assumes a reader who parses every aspect sentence; the NB-weighted model is a logistic
> one-vs-rest variant of NBSVM, and its probabilities are softmaxed scores, not posteriors.

## 6. Notebook cell map (21 cells)

| Cell | Type | Content |
|---|---|---|
| 1 | md | Title and the five claims |
| 2 | code | Setup: `import linear_text_classifiers as ltc`, corpus sizes |
| 3 | md | §1: Def 1 and Prop 1 |
| 4 | code | Prop 1 tests and scikit-learn |
| 5 | md | §2: Thm 1 |
| 6 | code | Gradient, Hessian, and residuals for LR vs NB |
| 7 | md | §3: Lemma 1 and Thm 2 |
| 8 | code | Ng–Jordan world table |
| 9 | md | §4: Thm 3 |
| 10 | code | Zipf tokens, rare share, α by n |
| 11 | md | §5: Prop 2 |
| 12 | code | The four learning curves |
| 13 | md | §6: Thm 4 |
| 14 | code | Path and GD tables |
| 15 | md | §7: Prop 3 |
| 16 | code | Feature-ridge test and funnel maxima |
| 17 | md | §8: finance |
| 18 | code | XOR, ceiling, prefix curve, table, and weights |
| 19 | md | Payload |
| 20 | code | Payload check |
| 21 | md | Where this sits |

## 7. Cross-site and scope

**Scope statement.**
- **formalML `naive-bayes`** (published) owns:
  - NB as a classifier and the generative/discriminative distinction, both in prose;
  - Gaussian, Multinomial, Bernoulli, and Complement NB;
  - Laplace smoothing as a Dirichlet posterior mean, and empirical-Bayes α;
  - Ng–Jordan for Gaussian NB vs LR, as prose rate sketches.

  This page **cites all of that** and proves what it leaves open:
  - NB as an exact softmax-linear classifier and LR's family as NB's plus length (Prop 1);
  - the multiclass Hessian and the two estimating equations (Thm 1);
  - the discrete Ng–Jordan generative bound with explicit constants (Lemma 1, Thm 2);
  - the multinomial and Zipf sample cost (Thm 3);
  - burstiness (Prop 2);
  - separation and the λ → 0 path (Thm 4);
  - NB features as a prior (Prop 3);
  - the aspect XOR (Prop 4).

  Their conjugacy proof block is malformed (it closes after one sentence), so the formal Dirichlet–multinomial
  conjugacy is cited from formalStatistics `multivariate-distributions` instead.
- **formalStatistics `generalized-linear-models`** owns binary logistic regression's concavity, IRLS = Newton, and
  asymptotic normality (Thms 1–3), and it treats separation only as Remark 8. Thm 1 here is the multiclass extension
  that its Remark 27 defers, and Thm 4 turns its remark into theorems.
- **formalRAG**:
  - `bm25-binary-independence-model` Thm 1 (the RSJ weight) is Bernoulli NB's weight, and gets a Connection.
  - `query-likelihood-language-models` (Dirichlet smoothing, the zero-frequency catastrophe) is multinomial NB's
    scoring rule applied to one document, and gets a Connection.
  - No retrieval content appears here.

| Field | Entries (all verified `published` on GitHub `main`, 2026-10-25) |
|---|---|
| `formalmlPrereqs` | `naive-bayes` (NB variants, Laplace, Gaussian Ng–Jordan), `vc-dimension` (half-spaces: VC dim d + 1, behind Ng–Jordan's Prop 2), `convex-analysis` (convexity, first- and second-order conditions) |
| `formalmlConnections` | `high-dimensional-regression` (logistic lasso, §11.1), `generalization-bounds` (margin-based bound: why the max-margin direction generalizes), `gradient-descent` (rates for the convex objective in Thm 1) |
| `formalstatisticsPrereqs` | `generalized-linear-models` (Thms 1–2: binary score, Hessian, IRLS), `maximum-likelihood` (score equation, consistency), `multivariate-distributions` (Dirichlet–multinomial conjugacy) |
| `formalstatisticsConnections` | `regularization-and-penalized-estimation` (MAP ↔ ridge, which Prop 3(a) specializes to a per-word prior) |
| `formalragConnections` | `bm25-binary-independence-model` (Thm 1, RSJ weight = Bernoulli NB weight), `query-likelihood-language-models` (Dirichlet smoothing; NB scoring of one document) |

Do not cite formalCalculus `convex-optimization`: it is still `draft`.

## 8. References

All of these were checked against Crossref, NeurIPS/JMLR/ACL Anthology metadata, or the PDF. The theorem statements of
Ng–Jordan, RZH (2003 and 2004), Soudry et al., Wang–Manning, and Cover were read in the PDFs.

- Albert, A., and J. A. Anderson. 1984. "On the Existence of Maximum Likelihood Estimates in Logistic Regression
  Models." *Biometrika* 71 (1): 1–10. https://doi.org/10.1093/biomet/71.1.1 (the separation trichotomy, cited from
  the abstract; Thm 4(a) is proved here).
- Cover, Thomas M. 1965. "Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications
  in Pattern Recognition." *IEEE Transactions on Electronic Computers* EC-14 (3): 326–34.
  https://doi.org/10.1109/PGEC.1965.264137 (Thm 1, the function-counting theorem).
- Domingos, Pedro, and Michael Pazzani. 1997. "On the Optimality of the Simple Bayesian Classifier under Zero-One
  Loss." *Machine Learning* 29 (2–3): 103–30. https://doi.org/10.1023/A:1007413511361
- Efron, Bradley. 1975. "The Efficiency of Logistic Regression Compared to Normal Discriminant Analysis." *Journal of
  the American Statistical Association* 70 (352): 892–98. https://doi.org/10.1080/01621459.1975.10480319
- Hoeffding, Wassily. 1963. "Probability Inequalities for Sums of Bounded Random Variables." *Journal of the American
  Statistical Association* 58 (301): 13–30. https://doi.org/10.1080/01621459.1963.10500830
- Joachims, Thorsten. 1998. "Text Categorization with Support Vector Machines: Learning with Many Relevant Features."
  In *Machine Learning: ECML-98*, edited by Claire Nédellec and Céline Rouveirol, 137–42. Berlin: Springer.
  https://doi.org/10.1007/BFb0026683
- Lewis, David D. 1998. "Naive (Bayes) at Forty: The Independence Assumption in Information Retrieval." In *Machine
  Learning: ECML-98*, edited by Claire Nédellec and Céline Rouveirol, 4–15. Berlin: Springer.
  https://doi.org/10.1007/BFb0026666
- Loughran, Tim, and Bill McDonald. 2011. "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and
  10-Ks." *Journal of Finance* 66 (1): 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x
- McCallum, Andrew, and Kamal Nigam. 1998. "A Comparison of Event Models for Naive Bayes Text Classification." In
  *Learning for Text Categorization: Papers from the 1998 AAAI Workshop*, 41–48. Technical Report WS-98-05. Menlo Park,
  CA: AAAI Press. https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf
- Ng, Andrew Y., and Michael I. Jordan. 2002. "On Discriminative vs. Generative Classifiers: A Comparison of Logistic
  Regression and Naive Bayes." In *Advances in Neural Information Processing Systems 14*, edited by Thomas G.
  Dietterich, Suzanna Becker, and Zoubin Ghahramani, 841–48. Cambridge, MA: MIT Press.
  https://proceedings.neurips.cc/paper_files/paper/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf
  - One numbering sequence: Prop 1, Prop 2 (VC side, m = Ω(n)), Lemma 3, Thm 4, Prop 5, Cor 6.
  - The formal h_Dis minimizes 0-1 training error.
- Niculescu-Mizil, Alexandru, and Rich Caruana. 2005. "Predicting Good Probabilities with Supervised Learning." In
  *Proceedings of the 22nd International Conference on Machine Learning*, 625–32. New York: ACM.
  https://doi.org/10.1145/1102351.1102430 (NB pushes predictions toward 0 and 1).
- Rennie, Jason D. M., Lawrence Shih, Jaime Teevan, and David R. Karger. 2003. "Tackling the Poor Assumptions of Naive
  Bayes Text Classifiers." In *Proceedings of the Twentieth International Conference on Machine Learning*, 616–23.
  Menlo Park, CA: AAAI Press. https://cdn.aaai.org/ICML/2003/ICML03-081.pdf
- Rosset, Saharon, Ji Zhu, and Trevor Hastie. 2004. "Boosting as a Regularized Path to a Maximum Margin Classifier."
  *Journal of Machine Learning Research* 5: 941–73. https://jmlr.org/papers/v5/rosset04a.html (§5, Thm 3: the
  constrained path).
- Rosset, Saharon, Ji Zhu, and Trevor Hastie. 2004. "Margin Maximizing Loss Functions." In *Advances in Neural
  Information Processing Systems 16*, edited by Sebastian Thrun, Lawrence K. Saul, and Bernhard Schölkopf, 1237–44.
  Cambridge, MA: MIT Press.
  https://proceedings.neurips.cc/paper_files/paper/2003/file/0fe473396242072e84af286632d3f0ff-Paper.pdf (Thm 2.1:
  the penalized path, λ → 0).
- Rubinstein, Y. Dan, and Trevor Hastie. 1997. "Discriminative vs Informative Learning." In *Proceedings of the Third
  International Conference on Knowledge Discovery and Data Mining*, 49–53. Menlo Park, CA: AAAI Press.
  https://cdn.aaai.org/KDD/1997/KDD97-008.pdf
- Soudry, Daniel, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. 2018. "The Implicit Bias of
  Gradient Descent on Separable Data." *Journal of Machine Learning Research* 19 (70): 1–57.
  https://jmlr.org/papers/v19/18-188.html
  - Thm 3: w(t) = ŵ log t + ρ(t).
  - Thm 5: rates, O(log log t/log t) in general.
  - Thm 7: multiclass.
- Wang, Sida, and Christopher D. Manning. 2012. "Baselines and Bigrams: Simple, Good Sentiment and Topic
  Classification." In *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume
  2: Short Papers)*, 90–94. Jeju Island, Korea: Association for Computational Linguistics.
  https://aclanthology.org/P12-2018/
  - Eq. 2: the log-count ratio.
  - Eq. 4: the interpolation, with w̄ = ‖w‖₁/|V| a scalar.
  - Settings: α = 1, C = 1, β = 0.25.
- Xue, Jing-Hao, and D. Michael Titterington. 2008. "Comment on 'On Discriminative vs. Generative Classifiers: A
  Comparison of Logistic Regression and Naive Bayes.'" *Neural Processing Letters* 28 (3): 169–87.
  https://doi.org/10.1007/s11063-008-9088-7 (cited from the abstract; the full text is paywalled).

## 9. Open questions for the checkpoint

1. **Difficulty.** `intermediate` is proposed: the tools are Hoeffding/Chernoff, convexity, and a compactness argument.
   But Thm 4(c) and the Kronecker Hessian are as hard as Topic 16, which is `advanced`. Which?
2. **Size.** Four theorems, a lemma, and four propositions. Thm 3 could shrink to a corollary of Lemma 1 plus a
   remark, and Prop 3(a) could become a remark. Keep everything, or trim?
3. **The crossover's framing.** The page leads with "no crossover when NB is the true model". The textbook picture
   appears only in the planted-burstiness world, and the real corpus shows none. Is that the right emphasis, or
   should the bursty world lead?
4. **NB-weighted LR vs NBSVM.** We use a logistic, one-vs-rest variant without Wang and Manning's β-interpolation, and
   cite theirs. Acceptable, or reproduce the squared-hinge SVM with β = 0.25 (another ≈ 1 s)?
