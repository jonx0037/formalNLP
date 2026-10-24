# Brief — `transformer-block-residual-stream`

**The Transformer Block and the Residual Stream.** Roadmap #11. Prerequisites: `attention-as-kernel-smoothing` (#9,
PR #17) and `positional-encodings-geometry` (#10, PR #18), both published. This topic is a prerequisite of
`contextual-embeddings-masked-lm` (#12) and `scaling-laws` (#13).

**Phase A** (this brief and the verified notebook), 2026-10-24:
- The harness passes 18/18 under numpy==2.5.4 and scipy==1.18.1 in ≈ 23 s, also with warnings promoted to errors.
- The narrative notebook (35 cells) executes with exit 0 in ≈ 27 s, including kernel start, and is committed
  without outputs.

**Scope decisions** (Claude's, 2026-10-24; Jonathan skipped the scoping questions and approved the plan):
- **The spine.** Topic 9 showed that attention smooths, and iterated smoothing is diffusion, so a pure-attention
  stack collapses to consensus. The skip connection keeps each token itself, which makes the stream a shared channel
  that every head reads with a linear map and writes into by addition. That additivity gives us three things: an
  exact path expansion of the logits, the identity path for the gradient, and the two-layer composition behind
  induction heads.
- **The computational pillar.** Attention-only RoPE models in NumPy, with hand-written backprop and Adam (Topic 10's
  pattern):
  - V = 256, n = 48, d = 32, two heads of width 16 per layer;
  - a 2-layer and a 1-layer model, trained on repeated random sequences;
  - a 2-layer model trained on hashed filings.
  - Plus pre-LN and post-LN stacks at initialization (d = 64, four heads, ReLU MLP of width 256), for the gradient
    profiles.
- **No LayerNorm in the trained models.** Without it the path expansion (Thm 3) is exact, with no frozen scales. LN
  gets its own section (geometry, Jacobian), and the init-stack experiment exercises it.
- **The MLP** is one proposition (key–value memory), not a section. MLP interpretability belongs to later topics.
- **Difficulty:** `advanced`. Topics 9 and 10 are `intermediate`, and this one assumes both.
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–10.

**Changes from the approved plan, each forced by a spike or the harness:**
1. **The finance case study inverted.** The plan trained induction on the filings. The spike showed a 2-layer model
   trained on the training-roster filings completes 0.000 of the held-out companies' names: memorizing names in the
   direct path is cheaper than learning to copy them. The bigram table also scores 0.000. A 2-layer model trained only
   on repeated random sequences, which never saw a filing, completes 0.768. The case study's thesis is now that memory
   in the weights is a lookup table, while the composed path is a general copying algorithm. The plan's fallback
   became the main line, and the filing-trained model is kept as the contrast.
2. **The collapse figure reports the absolute diameter.** The skip preserves differences, not proportions. At
   β = 0.25, A ≈ 11ᵀ/n, so X + AX adds the mean to every token: the diameter grows 3.58×, yet the relative spread falls
   to 0.0016. Prop 2 is stated for the diameter, and the relative effect is an asserted caveat.
3. **The narrow-kernel regime is in scope.** At β = 4, τ stays at 0.977 or above, and the pure stack freezes into
   clusters (spread 0.614 at depth 12) instead of collapsing. Thm 1 is silent there; we say so, and cite Geshkovski
   et al.'s clustering.
4. **The direct path ≈ bigram claim was dropped.** The filing model's direct-path argmax matches the bigram table's on
   only 46% of the 50 most frequent symbols, so the page claims only what §7 measures.
5. **The 1-layer control trains for 1000 steps**, the same as the 2-layer model. At 400 steps it was visibly
   unconverged.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | The Transformer Block and the Residual Stream |
| `{{SUBTITLE}}` | Attention alone averages tokens into one another; the residual connection keeps each token itself, and turns the block into a shared channel that heads read with linear maps and write into by addition — which makes the logits an exact sum of paths, keeps an identity path open for the gradient, and lets two heads compose into an induction head. |
| `{{SLUG}}` | `transformer-block-residual-stream` (= graph node id) |
| `{{DOMAIN}}` | `attention-transformers` |
| `{{DIFFICULTY}}` | `advanced` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[attention-as-kernel-smoothing, positional-encodings-geometry]` |
| `{{FORMALML_PREREQS}}` | `random-walks`, `svd`; connections `gradient-descent`, `pca-low-rank` |
| `{{FORMALRAG_PREREQS}}` | none; connections `high-dimensional-geometry`, `johnson-lindenstrauss` |
| `{{STATS_CALC_PREREQS}}` | Stats `linear-regression`; Calc `jacobian`; Calc connection `numerical-odes` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/positional-encodings-geometry/`, `notebooks/attention-as-kernel-smoothing/`, `notebooks/finance-corpus/` |

- **Tags:** `transformer`, `residual-stream`, `layer-norm`, `rank-collapse`, `pre-ln`, `path-expansion`,
  `induction-heads`, `logit-lens`, `interpretability`.
- `estimatedReadTime` ≈ 45.
- **In-site `connections`:** none, since #12 and #13 are still planned.
- **Graph:** both incoming edges (#9 → #11, #10 → #11) and both outgoing edges (#11 → #12, #11 → #13) exist.

**Scope.** This topic is not on the §6 watch-list. These neighbors adjoin it:
- **Topic 9** owns attention as a smoother, temperature as bandwidth, and one head's rank bound (its Prop 7). We
  import `attention`, and Prop 1 here is stated through it.
- **Topic 10** owns RoPE and the kernel over offsets. We import `rotate`/`frequencies`. The previous-token head's
  content-free score is Topic 10's kernel, built from the head's biases (§6).
- **formalML `random-walks`** owns mixing and the convergence theorem (its Thm 3) and the lazy walk (its Prop 1). Thm 1
  here is the contraction step behind them, in the Dobrushin form. Attention plus the skip, X + AX = 2·((I + A)/2)X,
  is a scaled lazy walk.
- **formalCalculus `jacobian`** owns the multivariate chain rule (its Thm 2), which Thm 2 here applies.
- **This topic owns:**
  - the block's definitions;
  - collapse and the skip (Thm 1, Prop 2);
  - LN's geometry (Prop 3, Cor 1);
  - the identity path (Thm 2) and the init gradient profiles;
  - the path expansion and composition (Thm 3, Prop 4);
  - induction heads, the 1-layer limit (Prop 5), and the logit lens;
  - the finance case study.
- **Out of scope:**
  - masked LM and bidirectional pretraining (#12);
  - scaling (#13);
  - MLP interpretability, superposition, and sparse autoencoders (mentioned, not covered);
  - tuned lens training;
  - mixture-of-experts and normalization variants beyond RMSNorm;
  - mean-field and NTK analyses beyond citing Xiong et al.

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | ten tokens in the plane under twelve layers of attention, at β = 1. Without the skip the diameter goes from 2.574 to 0. With it, the cloud grows 2862× and stays apart. Then the headline: the 2-layer model completes 0.768 of company names it never saw, and the filing-trained model 0.000 | — | cells 0–1 |
| 1 | The block, assembled | Topic 9's smoother, H times, each head writing through its own W_O into one vector | Def 1; **Prop 1** | cells 2–3 |
| 2 | Attention alone collapses | averaging twice is averaging more; the lab's cloud at β = 0.25 … 4 | Def 2; **Thm 1**; **Prop 2** | cells 4–8 |
| 3 | LayerNorm is a sphere in a hyperplane | a point in R³, its shadow on the plane x + y + z = 0, pushed out to the circle of radius √3 | Def 3; **Prop 3**; **Cor 1** | cells 9–10 |
| 4 | The identity path | the product (I + J₁)(I + J₂)(I + J₃) multiplied out: eight terms, one of them I | **Thm 2**; Xiong et al., measured | cells 11–14 |
| 5 | The stream as a channel | the logits of the trained model split into nine path terms that sum back to the forward pass (error 1.8e-14) | Def 4; **Thm 3**; **Prop 4** | cells 15–18 |
| 6 | Induction heads | `… [A][B] … [A] → [B]`: a previous-token head, then a head that looks for its own token in the previous-token head's output | **Prop 5**; experiments; logit lens | cells 19–27 |
| — | The lab | `<ResidualStreamLab client:visible />` | — | cells 32–33 |
| 7 | Finance case study: completing a company's name | held-out filing 31: "Tamsin Retail Group", a company that never occurs in training | — | cells 28–31 |
| 8 | Where this goes next | — | #12 (bidirectional attention on the same block; the masked-LM objective); #13 (depth and width as the scaling axes; the pre-LN stream's √L growth); formalRAG `high-dimensional-geometry` | — |

- **Lab placement:** after §6. Figure A needs §§2–3, and Figure B needs §§5–6. Figure B's filing view previews §7.
- **Back-links to Topic 9:**
  - Def 2 (scaled dot-product attention) from Def 1;
  - Prop 2 (temperature limits) and Thm 1 (temperature is bandwidth) from Thm 1 (ii);
  - Prop 7 (one head's rank bound) from Def 4;
  - Prop 5 (one hop) from Thm 2.
- **Back-links to Topic 10:**
  - Def 4 (RoPE) and Thm 1 (the kernel over offsets) from §6's previous-token head;
  - Prop 1 (a causal layer counts) from §1's residual stream.

## 2. Definitions

**Notation.**
- A sequence of n tokens is a matrix X ∈ R^{n×d}, one row x_i per position. Maps act on the right, x ↦ xW.
- **1** is the all-ones vector.
- On features, P = I_d − 11ᵀ/d centers each row. On tokens, P_n = I_n − 11ᵀ/n subtracts the mean row.
- diam(X) = max_{i,j} ‖x_i − x_j‖, for any norm on R^d. ‖W‖ is the operator norm of x ↦ xW for that norm.

**Def 1 (the transformer block).**
- A **head** h, with maps W_Q^h, W_K^h, W_V^h ∈ R^{d×d_h} and W_O^h ∈ R^{d_h×d}, sends X to
  softmax(X W_Q^h (X W_K^h)ᵀ/√d_h + B) X W_V^h W_O^h. Here B is the causal mask (−∞ above the diagonal) or 0, and
  RoPE (Topic 10, Def 4) may rotate the queries and keys.
- **Multi-head attention** is MHA(X) = Σ_h head_h(X).
- An **MLP** is MLP(x) = σ(xW₁ + b₁)W₂ + b₂, applied to each row, with σ = ReLU here.
- The **pre-LN block** is X ← X + MHA(LN X), then X ← X + MLP(LN X).
- The **post-LN block** (Vaswani et al.'s) is X ← LN(X + MHA X), then X ← LN(X + MLP X).
- The **residual stream** is the sequence X₀, X₁, … of these matrices. Each sublayer reads it and writes back by
  addition.

**Def 2 (the Dobrushin coefficient; the smoothing stack).**
- For a row-stochastic A ∈ R^{n×n},
  τ(A) = ½ max_{i,j} ‖A_{i·} − A_{j·}‖₁ = 1 − min_{i,j} Σ_k min(A_ik, A_jk) ∈ [0, 1].
- The two forms agree because, for probability vectors a and b, |a_k − b_k| = a_k + b_k − 2 min(a_k, b_k).
- τ(A) = 0 exactly when all rows are equal, and τ(A) = 1 when two rows have disjoint support.
- The **pure attention stack** is X_{l+1} = A_l X_l W_l. The **stack with the skip** is X_{l+1} = X_l + A_l X_l W_l.
- In the lab, A_l = softmax(β X_l X_lᵀ), so queries = keys = X (Topic 9's attention at temperature 1/β), and W = I.

**Def 3 (LayerNorm, RMSNorm).**
- LN(x) = (x − μ1)/σ with μ = mean(x) and σ² = mean((x − μ)²), with gain 1 and bias 0, as at initialization. Ba et
  al. add ε to σ²; we take ε = 0.
- RMSNorm(x) = √d · x/‖x‖ (Zhang and Sennrich).

**Def 4 (the attention-only model; circuits; composition).**
- The model is x₀ = E[t] (a row of the embedding), x_{l+1} = x_l + MHA_l(x_l) (causal, RoPE, with query and key
  biases), and logits = x_L U. There is no MLP and no LN.
- Head (l, h) has an **OV circuit** W_OV^{lh} = W_V^{lh} W_O^{lh} ∈ R^{d×d}, which is what it writes, and a
  **QK circuit**, its score (x_i W_Q + b_q)·R(j − i)(x_j W_K + b_k)/√d_h.
- A path is a chain (l₁, h₁), …, (l_m, h_m) with l₁ < … < l_m.
- **Composition scores** (Elhage et al.): ‖W_OV^{0h} M‖_F/(‖W_OV^{0h}‖_F‖M‖_F), where M is W_K^{1h'} for K-composition,
  W_Q^{1h'} for Q-composition, and W_OV^{1h'} for V-composition. They measure how much of what head (0, h) writes
  head (1, h') reads.

## 3. Statements and proofs

**Prop 1 (heads are smoothers that write; the MLP is a memory).**
- (i) Without RoPE or biases, MHA(X) = Σ_h Attn(XW_Q^h, XW_K^h, XW_V^h) W_O^h, with Attn Topic 9's Def 2. Each head is a
  Nadaraya–Watson smoother (Topic 9, Prop 1), and the block adds H smoothed copies, each mapped into the stream by its
  own W_O^h.
- (ii) MLP(x) = b₂ + Σ_{i=1}^{w} σ(⟨x, k_i⟩ + b_{1i}) v_i, with key k_i the i-th column of W₁ and value v_i the i-th row
  of W₂.
- *Proof.* (i) Concatenating heads and multiplying by the stacked W_O is the sum of the per-head products; this is
  block matrix multiplication.
- (ii) xW₁ has entries ⟨x, k_i⟩, and σ(·)W₂ = Σ_i σ(·)_i v_i. ∎
- *Reading* (Geva et al.). An MLP is attention over w stored keys, with an unnormalized gate σ in place of softmax:
  its keys and values are parameters, where attention's come from the context.
- *Harness:* `test_multi_head_is_a_sum_of_smoothers`. Head by head against `ak.attention` to 1e-12, and row by row
  against `key_value_memory`.

**Thm 1 (attention contracts toward consensus).** Let A be row-stochastic.
- (i) diam(AX) ≤ τ(A) diam(X), in every norm.
- (ii) If each row of the score matrix S has range max_k S_ik − min_k S_ik ≤ r, then τ(softmax S) ≤ 1 − e^{−r}.
- (iii) The pure stack satisfies diam(X_L) ≤ (∏_{l<L} τ(A_l)‖W_l‖) diam(X₀).
- *Proof of (i).*
  - Fix rows i and j, and let c = A_{i·} − A_{j·}. Then Σ_k c_k = 0 and ‖c‖₁ = 2τ_ij, with τ_ij ≤ τ(A).
  - If τ_ij = 0, rows i and j are equal and there is nothing to prove.
  - Otherwise split c = τ_ij(p − q), with p = c⁺/τ_ij and q = c⁻/τ_ij probability vectors. Each part has mass τ_ij,
    because the positive and negative parts of a zero-sum vector have equal mass.
  - Then (AX)_i − (AX)_j = Σ_k c_k x_k = τ_ij(Σ_k p_k x_k − Σ_m q_m x_m) = τ_ij Σ_{k,m} p_k q_m (x_k − x_m).
  - By the triangle inequality its norm is ≤ τ_ij Σ p_k q_m ‖x_k − x_m‖ ≤ τ(A) diam(X). Maximize over i and j.
- *Proof of (ii).* Each entry satisfies A_ik = e^{S_ik}/Σ_m e^{S_im} ≥ e^{min_m S_im}/(n e^{max_m S_im}) ≥ e^{−r}/n.
  So Σ_k min(A_ik, A_jk) ≥ e^{−r}, and Def 2 gives τ ≤ 1 − e^{−r}.
- *Proof of (iii).* First, ‖x_i W − x_j W‖ ≤ ‖W‖ ‖x_i − x_j‖, so diam(XW) ≤ ‖W‖ diam(X). Then (i) gives
  diam(A_l X_l W_l) ≤ τ(A_l) diam(X_l W_l) ≤ τ(A_l)‖W_l‖ diam(X_l), and we induct on l. ∎
- *Reading.*
  - With A_l = softmax(βXXᵀ), r grows with β times the score range. A wide kernel (small β, Topic 9's high
    temperature) has τ well below 1, and collapse is geometric.
  - Dong, Cordonnier, and Loukas prove the faster, doubly exponential rate for the relative residual of pure
    self-attention with value maps, by a path decomposition like Thm 3's. We cite it and do not reprove it.
  - (i) is the contraction step of formalML `random-walks`' convergence theorem in the Dobrushin form (Seneta, §4.3).
- *Where the bound is silent.* A narrow kernel (β = 4) makes each token attend mostly to itself. τ stays at 0.977 or
  above, the bound allows anything, and the pure stack freezes into clusters (spread 0.614 at depth 12). Geshkovski
  et al. show that cluster formation is the long-time behavior of attention dynamics.
- *Harness:* `test_dobrushin_contraction`, with 300 random instances in the ℓ₁, ℓ₂, and ℓ∞ norms and both forms of τ;
  `test_collapse_and_the_skip`.

**Prop 2 (the skip keeps the differences).** For X' = X + AXW, diam(X') ≥ (1 − τ(A)‖W‖) diam(X). Over L layers this
gives diam(X_L) ≥ ∏(1 − τ_l‖W_l‖) diam(X₀) ≥ (1 − Σ τ_l‖W_l‖) diam(X₀) whenever every τ_l‖W_l‖ ≤ 1.
- *Proof.*
  - Pick i and j with ‖x_i − x_j‖ = diam(X).
  - x'_i − x'_j = (x_i − x_j) + ((AXW)_i − (AXW)_j).
  - By Thm 1 (i) applied to XW, together with diam(XW) ≤ ‖W‖ diam(X), the second term has norm ≤ τ(A)‖W‖ diam(X).
    The reverse triangle inequality then gives the claim.
  - The product bound is Weierstrass's inequality ∏(1 − a_l) ≥ 1 − Σ a_l, for a_l ∈ [0, 1]. ∎
- *Reading.* Pure attention multiplies the differences between tokens by at most τ‖W‖. The skip adds the differences
  back unchanged, so a stack of small updates (Σ τ‖W‖ < 1) cannot erase them.
- *Caveat (asserted).* The skip keeps differences, not proportions. With A = 11ᵀ/n and W = I, X + AX adds the mean row
  to every token, so the deviations stay while the common part doubles.
  - In the lab at β = 0.25, the diameter grows 3.58× and the relative spread ‖P_n X‖_F/‖X‖_F falls to 0.0016.
  - As the common part grows, the scores sharpen or flatten with it, and τ tracks them.
  - In all ten lab runs with the skip, the diameter never decreases.
- *Harness:* `test_collapse_and_the_skip`. It covers the lab's five temperatures, plus 100 random instances with random
  W and both bounds, layer by layer.

**Prop 3 (LayerNorm is a sphere in a hyperplane).** LN(x) = √d · Px/‖Px‖. In words, LN projects onto 1^⊥ and then
radially onto the sphere of radius √d inside that hyperplane. Consequences:
- LN(cx + s1) = LN(x) for c > 0;
- LN∘LN = LN;
- LN(x) = RMSNorm(Px).
- *Proof.*
  - σ² = ‖Px‖²/d, so (x − μ1)/σ = Px/(‖Px‖/√d).
  - P(cx + s1) = cPx, and normalizing removes c > 0.
  - On y ∈ 1^⊥ with ‖y‖ = √d, Py = y, so LN(y) = y.
  - The last identity is the definition of RMSNorm applied to Px. ∎
- Brody, Alon, and Yahav make this geometric reading and argue that both components, the projection and the scaling,
  matter for the attention that follows. The centering is formalStatistics `linear-regression`'s orthogonal
  projection, onto the complement of the intercept column (its Thm 1).
- *Thin shell.* For x ~ N(0, I_d), ‖Px‖ concentrates at √(d − 1) (formalRAG `high-dimensional-geometry`, Thm 1). So
  LN barely moves a Gaussian vector at large d. The median of ‖LN(x) − Px‖/‖Px‖ is below 0.025 at d = 512 and above
  0.1 at d = 8. LN acts when the stream's norm drifts away from √d, which is exactly what happens with depth in pre-LN
  (§4).
- *Harness:* `test_layer_norm_geometry` (d = 3, 8, 64) and `test_thin_shell`.

**Cor 1 (LayerNorm's Jacobian).** Let u = Px/‖Px‖. Then ∂LN/∂x = (√d/‖Px‖)(P − uuᵀ). This matrix is symmetric, and it
kills both 1 and x itself. Its singular values are √d/‖Px‖, with multiplicity d − 2, and 0, twice.
- *Proof.*
  - With c = Px, d(√d c/‖c‖) = (√d/‖c‖)(I − uuᵀ)dc, and dc = P dx.
  - Since u ⊥ 1, uuᵀP = uuᵀ.
  - P1 = 0 and uᵀ1 = 0, which kills 1.
  - Px = c and uᵀx = ‖c‖, so (P − uuᵀ)x = c − u‖c‖ = 0, which kills x.
  - P − uuᵀ is the orthogonal projection onto the (d − 2)-dimensional complement of span{1, u}. ∎
- *Reading.* The gradient through LN is scaled by 1/‖Px‖ and loses its radial component: LN does not care how long
  its input is.
- *Harness:* `test_layer_norm_geometry` checks it against Topic 5's central differences to 1e-7, and checks
  `ln_backward` against Jᵀ.

**Thm 2 (the identity path).**
- (i) For a residual stack x_{k+1} = x_k + f_k(x_k) with Jacobians J_k = ∂f_k/∂x at the visited points,
  ∂x_L/∂x_l = (I + J_{L−1}) ⋯ (I + J_l) = Σ over subsets {k₁ < … < k_m} ⊆ {l, …, L − 1} of J_{k_m} ⋯ J_{k₁}.
  - That is 2^{L−l} terms, and the empty subset contributes I.
  - So ∂ℒ/∂x_l = ∂ℒ/∂x_L · (I + Σ_{S ≠ ∅} …): every layer receives the top gradient unchanged, plus the paths
    through the sublayers.
- (ii) Pre-LN: f_k = F_k ∘ LN, so J_k = J_{F_k} J_LN(x_k), and by Cor 1 ‖J_LN(x_k)‖ = √d/‖Px_k‖. As the stream grows,
  every branch term shrinks, but the I term does not.
- (iii) Post-LN: x_{k+1} = LN(x_k + F_k(x_k)), so ∂x_{k+1}/∂x_k = J_LN(x_k + F_k(x_k))(I + J_{F_k}). Every path,
  including the skip-only path, passes through L − l LayerNorm Jacobians.
- *Proof.*
  - (i) The chain rule (formalCalculus `jacobian`, Thm 2) gives ∂x_{k+1}/∂x_k = I + J_k and composes them.
    Distributing the product of L − l binomials gives one term per subset, with factors in layer order.
  - (ii) and (iii) apply the chain rule to the two block forms and use Cor 1. ∎
- Veit, Wilber, and Belongie read (i) as an ensemble of paths of varying lengths.
- *What we measure (Xiong et al.).* Xiong et al.'s Theorem 1 bounds the last FFN layer's gradient at initialization
  by O(d√ln d) for post-LN and O(d√(ln d/L)) for pre-LN. It holds with high probability under a concentration
  assumption on the hidden states' norms, and is proved by mean-field arguments (statement checked against the arXiv
  v2 PDF). The `rigorFlag` records that
  we do not prove this: we measure the scaling it predicts, at random initialization.
  - Pre-LN: the last block's ‖∂ℒ/∂W₂‖_F falls 10.168718 → 7.673006 → 5.217570 at L = 6, 12, 24. The 6 → 24 ratio is
    1.95, against √4 = 2.
  - Post-LN: it stays at 22.019151, 23.416205, 22.717818, which is 4.35× pre-LN at L = 24.
  - The pre-LN stream grows: ‖x_l‖² is linear in l (R² > 0.98), and ‖x_L‖ goes 8.105715 → 49.281404 at L = 24.
    That growth is (ii)'s mechanism: J_LN shrinks like 1/√L. Post-LN pins ‖x_l‖ at √d = 8.
  - The *shape* of the per-layer profile depends on the initialization, and ours (He-initialized ReLU MLP) differs from
    Xiong et al.'s Figure 3. We assert the scaling, not the shape.
- *Harness:* `test_jacobian_paths` checks the chain rule against central differences and the subset sum against the
  product to 1e-12. `test_gradient_scale_at_init` checks the scaling with 8 draws per (scheme, depth).
  `test_gradients` covers both stacks, every coordinate.

**Thm 3 (the path expansion).** In the attention-only model, take each head's attention matrix A^{lh} from the forward
pass. Then
  logits = Σ over paths (l₁,h₁) … (l_m,h_m) of A^{l_m h_m} ⋯ A^{l₁h₁} x₀ W_OV^{l₁h₁} ⋯ W_OV^{l_m h_m} U.
- The empty path is the **direct path** x₀U = E[t]U, a function of the current token alone: a bigram model.
- With L = 2 and H heads there are 1 + 2H + H² paths. With L layers there are (1 + H)^L, at most one head per layer.
- *Proof.*
  - Each layer is x_{l+1} = x_l(I) + Σ_h A^{lh} x_l W_OV^{lh}. Here A acts on the token index (left) and W_OV on the
    feature index (right), and the two commute in the sense that A(XW) = (AX)W.
  - Substituting x_l recursively and distributing gives one term per choice, at each layer, of either the skip or
    one head. Collecting the left factors and the right factors gives the formula. ∎
- *Honest scope.* The A^{lh} depend on x_l, so the expansion is exact *given the patterns*. It is linear in the values,
  not in the attention. K- and Q-composition live inside the patterns; V-composition lives in the products of
  W_OV's.
- *Harness:* `test_path_expansion` checks both the trained model (9 paths) and a random 3-layer model (27 paths)
  against the forward pass, to 1e-10. Measured error on the trained model: 1.8e-14.

**Prop 4 (orthogonal subspaces do not talk).** If W_OV^{0h} W_K^{1h'} = 0 for every layer-0 head h, then the keys of
head (1, h') computed from x₁ and from x₀ coincide. Its attention pattern is then the same with or without layer 0's
output, and the K-composition score is 0.
- *Proof.* x₁W_K^{1h'} = x₀W_K^{1h'} + Σ_h A^{0h} x₀ W_OV^{0h} W_K^{1h'} = x₀W_K^{1h'}. The bias b_k and the rotation
  are added and applied after this product. ∎
- *Reading.* The stream is a d-dimensional channel shared by every head. Heads communicate only through the subspaces
  where one writes and another reads.
- *Harness:* `test_orthogonal_heads_do_not_talk`. It builds a model whose layer-0 heads write into the first 16
  coordinates while the layer-1 keys read only the last 16, and checks that the two patterns agree to 1e-12.

**Prop 5 (what one layer's keys see).** In a one-layer model, the score s_h(i, j) is a function of (x_i, i, x_j, j)
alone. In particular it does not change when token j − 1 changes.
- In two layers, the key at j reads x₁[j] = x₀[j] + Σ_h Σ_k A^{0h}_{jk} x₀[k]W_OV^{0h}. A previous-token head
  (A^{0h}_{j,j−1} ≈ 1) writes x₀[j − 1] there.
- *Proof.* In one layer, q_i = R(i)(x_i W_Q + b_q) and k_j = R(j)(x_j W_K + b_k), where x_i and x_j are the embeddings
  of tokens i and j. ∎
- *Reading.* Induction, `[A][B] … [A] → [B]`, must attend to the position whose *predecessor* equals the current
  token. A one-layer score cannot see predecessors, so it cannot single that position out by content. Two layers can,
  through K-composition. This is a statement about information, not a lower bound on loss: a one-layer model can
  still raise every token it has seen. The quantitative gap is measured.
- *Harness:* `test_one_layer_keys`. In the 1-layer model, changing token j − 1 changes s(i, j) by less than 1e-12. In
  the 2-layer model, layer 1's score moves by 4.45 nats on average (asserted > 2).

**§6 experiments (asserted in `test_induction_heads`, `test_logit_lens`, `test_composed_path_carries_the_copy`).**
- **The task.** Repeated random sequences: a uniform block of V = 256 tokens with period p ~ U{6, …, 24}, repeated to
  n = 48. Loss is scored at positions i ≥ p. Both models train for 1000 Adam steps at batch 32.
- **Results** on 1000 test sequences:

  | model | loss (nats) | accuracy |
  |---|---|---|
  | 2-layer | 0.156970 | 0.974159 |
  | 1-layer | 4.279829 | 0.107541 |
  | 2-layer, keys from x₀ (no K-composition) | 6.134730 | 0.075163 |

  Uniform guessing gives ln 256 = 5.545.
- **The heads.**
  - Head (0, 0) puts 0.887284 of its weight on the previous token. Its content-free score
    b_qᵀR(−Δ)b_k/√d_h, which is Topic 10's kernel over offsets, peaks at Δ = 1 (7.78 against 6.71 at Δ = 0).
  - Heads (1, 0) and (1, 1) put 0.592626 and 0.594006 on the induction target i − p + 1.
- **Composition scores.**
  - K-composition is 0.245–0.272 for every pair. Q-composition is 0.046–0.075 and V-composition 0.036–0.076.
  - Every K-score exceeds the largest K-score of the same model at its initialization (0.188).
- **The paths.** At the masked positions, the paths that end in a layer-1 head carry a median of more than 90% of the
  correct token's centered logit. The direct path plus the layer-0 heads alone rank the correct token first under 5%
  of the time.
- **The logit lens.** We read x_lU. This model has no final LN, so the lens is the model's own readout. The probability
  of the correct token after layer 0, layer 1, and layer 2 is 0.003872, 0.003930, and 0.958364; chance is 1/256 =
  0.003906. The answer exists only after the second layer has composed with the first. On models with a final LN,
  the lens reads W_U·LN(x_l), a heuristic that Belrose et al. refine with a trained affine probe per layer.

## 4. Finance case study: completing a company's name

**Setup.**
- `fc.split_roster()` splits the public companies 75/25, stratified by sector.
- Three corpora:
  - train: `generate_corpus(300, 0, roster=train)`;
  - seen: `generate_corpus(100, 1, roster=train)`, new filings about the same companies;
  - unseen: `generate_corpus(100, 1, roster=held)`, companies whose names never occur in training.
- Words are hashed to CRC-32 mod 256, Topic 10's convention.
- **Queries:** every word of a multi-word ORG name except the last, where the same name was mentioned in full earlier
  in the filing, inside the 48-word window ending at the query. The target is the next word of the name. There are 269
  seen queries and 336 unseen.

**Readers** (argmax next symbol):
- the bigram table counted on the training filings;
- the filing-trained 2-layer model's direct path, E[x]U;
- the filing-trained 2-layer model itself (800 steps on 49-word windows);
- the 1-layer induction model;
- the 2-layer induction model.

| reader | seen | unseen |
|---|---|---|
| bigram table | 0.576208 | 0.000000 |
| filing-trained direct path | 0.717472 | 0.000000 |
| filing-trained 2-layer | 0.814126 | 0.000000 |
| induction, 1 layer | 0.055762 | 0.071429 |
| induction, 2 layers | 0.710037 | 0.767857 |

**The punchline.** Everything the filing-trained model knows about names lives in its weights. It completes 81% of
the names it was trained on and none of the others. The bigram table does the same, so the model's skill on seen names
is a better lookup table. The induction model never saw a word of finance. It completes 71% of the seen names and 77%
of the unseen ones, because it reads the name off the filing's own earlier text.
- This is Topic 9's kernel cache made into a circuit: memory in the context.
- The filing-trained model, with the same architecture and two layers, never learned to copy. Copying is needed only
  when the names are new, and in training they never were.

**Lab filing:** held-out filing 31 ("Tamsin Retail Group"), the unseen filing with the most queries (12). It shows
each reader's hit or miss at every query.

**Honest notes.**
- The corpus is synthetic, and its names repeat often.
- Hash collisions put an unseen word's symbol on some training word's bigrams, so the table answers with that word's
  continuation.
- The induction model's misses (0.23 of the unseen queries) are not analyzed here.
- Nothing here is real filing text or a real dictionary.

## 5. `rigorFlag`

Proved in full:
- Thm 1, Prop 2, Prop 3, Cor 1, Thm 2 (i)–(iii), Thm 3, Prop 4, and Prop 5;
- Prop 1, which is algebra.

Not proved here, and flagged:
1. **Doubly exponential collapse** (Dong et al.) is cited. We prove the geometric rate (Thm 1 (iii)), which is weaker.
2. **Xiong et al.'s gradient bounds** are mean-field results at initialization, under their assumptions. We measure
   the predicted L^{−1/2} scaling on one toy configuration; we do not prove it. The per-layer profile shape is not
   reproduced and not claimed.
3. **The path expansion is exact given the attention patterns** and is linear in the values, not in the patterns.
   With LayerNorm it is exact only if each token's LN scale is frozen at its forward-pass value. Our trained models
   omit LN for that reason.
4. **"Channel," "key–value memory," and "logit lens" are readings, not theorems.** The lens is exact for our LN-free
   model and a heuristic elsewhere (Belrose et al.).
5. **Prop 5 limits what one layer can *see*, not what loss it can reach.** The 1-layer control's loss (4.28) is a
   1000-step measurement, not a lower bound.
6. **The induction mechanism is shown on toy models.** Its role in large models' in-context learning is Olsson et al.'s
   empirical claim: causal evidence for small attention-only models, correlational for large ones.
7. **The finance result is on a synthetic corpus with hashed words.** It shows the mechanism, not a production name
   tagger.
8. **The skip keeps differences, not proportions** (Prop 2's caveat). Relative collapse can still happen, and the
   lab shows it at β = 0.25.

## 6. Viz design intent — `ResidualStreamLab.tsx` (one component, two figures; data `residual-stream-lab.json`, 33.6 kB)

**A. "Smoothing without a self."**
- **The cloud.** Ten tokens in the plane through 12 layers.
  - A depth slider (0–12), a β selector (0.25, 0.5, 1, 2, 4), and a pure/skip toggle.
  - Each layer is drawn about its centroid, in the stack's own coordinates, with a scale readout, because the skip
    stack grows to ×4000.
  - Beside it, diam(X_l) on a log axis for both stacks, with the pure stack's bound ∏τ_l · diam(X₀) dashed. A τ_l
    readout per layer.
  - What the reader learns: wide kernels collapse, narrow kernels cluster, and the skip never shrinks the cloud.
    At β = 0.25 the relative-spread readout shows proportions shrinking while differences survive.
- **LayerNorm.** 24 points in R³, drawn in the plane 1^⊥ (coordinates e₁, e₂), with their offset along 1 as a
  vertical tick.
  - A two-step toggle, "center", then "normalize": the points drop onto the plane, then onto the circle of radius √3.
  - A scale slider multiplies the cloud, and the normalized points do not move (Prop 3's invariance).

**B. "Two heads that compose."**
- **The patterns.** Test sequence 42, with period 11: the previous-token head's and the induction head's attention as
  48 × 48 heatmaps, stored sparse (weights ≥ 0.02). Hovering a query row highlights its target.
- **The paths.** At a selected position, the correct token's centered logit split into the 9 path terms. They are
  grouped as direct, layer-0 heads, layer-1 heads, and composed, and they sum to the total.
- **The ablations and the lens.**
  - The metrics table, with the K-composition cut as a toggle.
  - The logit lens as three bars (0.004, 0.004, 0.958) and per position.
  - The composition scores as three 2 × 2 grids (K, Q, V).
- **The filing.** Filing 31's words, with the 12 queries marked and each reader's hit or miss, beside the
  seen/unseen accuracy table.
- All numbers come from `viz_payload()`. Displayed values are rounded inside the payload, and
  `test_viz_mirror_matches_payload` pins the file.

## 7. The module, as built (`transformer_block_residual_stream.py`)

**Imports.** Topic 10 is imported as a module object (`pe`), and Topic 9 and Topic 5 are reached through it: `ak =
pe.ak` and `nlm = pe.nlm`. This keeps their `test_*` functions out of this harness.

**Reuses.**
- From Topic 9: `ak.attention` (Def 2's stack, Prop 1), `ak._softmax`, `ak._r`, `ak._sig`, `ak.SEED`.
- From Topic 10: `pe.rotate` and `pe.frequencies` (RoPE in every trained model).
- From Topic 5: `nlm.log_softmax`, `nlm.central_diff_grad`, and `nlm.ADAM_*`.
- From the finance corpus: `fc.generate_corpus(…, roster=…)`, `fc.split_roster`, `fc.CORPUS_FINGERPRINT`.

**Sections.**
- §1/§3, the block and LayerNorm:
  - `center`, `layer_norm`, `ln_sphere`, `rms_norm`, `ln_jacobian`, `ln_backward`;
  - `init_mha`, `mha_forward`, `mha_backward`;
  - `mlp_forward`, `mlp_backward`, `key_value_memory`.
- §2: `dobrushin`, `diameter`, `relative_spread`, `smoothing_stack`, `collapse_cloud`, `collapse_run`.
- §4: `init_block`, `stack_forward`, `stack_loss_grad`, `gradient_profile`, `jacobian_paths`.
- §5/§6, the attention-only model:
  - building and running it: `init_model`, `model_forward` (with `ablate_k`), `model_loss_grad`;
  - data and decomposition: `repeated_sequences`, `path_logits`;
  - training: `ModelConfig` and `train_model` (lru_cache; streams 1431, 1432, 1442);
  - analysis: `head_scores`, `composition_scores`, `repeated_metrics`, `logit_lens`, `bias_offset_score`.
- §7, finance: `symbol`, `NameDoc`, `roster_docs`, `filing_windows`, `name_queries`, `bigram_table`,
  `readers_predict`, `name_accuracy`.
- Payload: `ln_cloud`, `lab_sequence`, `lab_filing`, `viz_payload`, `summary`, and 18 tests.

**Random streams:** `default_rng([SEED, k])` with k = 1400–1460, which keeps this module apart from Topics 5–10's
900–1399.

**Timing.** The three trainings take about 7, 5, and 8 s. The full harness takes ≈ 23 s.

## 8. Notebook cell map (`01_transformer_block_residual_stream.ipynb`)

| cells | content |
|---|---|
| 0–1 | title; imports, the headline numbers |
| 2–3 | §1: Def 1, Prop 1, checked head by head |
| 4–5 | §2: Def 2, Thm 1, the Dobrushin coefficient on a random instance |
| 6–8 | the lab's cloud: τ, diameters, and spreads by β, pure vs skip; Prop 2 and its caveat |
| 9–10 | §3: Prop 3, Cor 1, the thin shell; LN at d = 3 |
| 11–14 | §4: Thm 2 by subsets; Xiong et al.'s bounds; the gradient profiles and the stream norm |
| 15–18 | §5: Def 4, Thm 3 on the trained model (9 paths); Prop 4 |
| 19–27 | §6: the task and the three models; the heads, the offset kernel, the composition scores; Prop 5; the paths; the logit lens |
| 28–31 | §7: the accuracy table; the punchline; filing 31 query by query |
| 32–33 | the lab payload |
| 34 | `assert tb._run_harness() == 0` |

**Commands** (both exit 0):
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/transformer-block-residual-stream/transformer_block_residual_stream.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/transformer-block-residual-stream/01_transformer_block_residual_stream.ipynb
```

## 9. Cross-site links (all `status: "published"` on GitHub `main`, checked 2026-10-24; re-verify at the B3 gate)

| field | slug | bridge |
|---|---|---|
| `formalmlPrereqs` | `random-walks` | Thm 3 (convergence) and Prop 1 (the lazy walk's spectrum): Thm 1 is the contraction step, and X + AX is a scaled lazy walk |
| `formalmlPrereqs` | `svd` | Thm 1 (SVD), for the rank of W_OV = W_V W_O ≤ d_h, and the composition scores as Frobenius ratios |
| `formalcalculusPrereqs` | `jacobian` | Thm 2 (the multivariate chain rule): ∂(x + f(x))/∂x = I + J, composed over layers |
| `formalstatisticsPrereqs` | `linear-regression` | Thm 1 (OLS as orthogonal projection): LN's centering is the projection off the intercept column |
| `formalmlConnections` | `gradient-descent` | the descent lemma's step-size condition, beside post-LN's large output-layer gradients and warmup |
| `formalmlConnections` | `pca-low-rank` | the stream as a shared subspace: heads read and write low-rank projections of it |
| `formalcalculusConnections` | `numerical-odes` | Thm 1 (Euler's global error): a residual block x + f(x) is one Euler step of x' = f(x) |
| `formalragConnections` | `high-dimensional-geometry` | Thm 1 (thin-shell concentration): ‖Px‖ ≈ √d, so LN barely moves a Gaussian vector |
| `formalragConnections` | `johnson-lindenstrauss` | Thm 3 (JL): many nearly orthogonal directions fit in d = 32, which is why random 32-dimensional embeddings of 256 tokens can still be copied |

**Not cited:**
- `spectral-theorem`, `representation-learning`, `concentration-inequalities`, `stability-dynamics`, `double-descent`:
  weaker fits;
- formalStatistics `multivariate-distributions`.

## 10. References (Chicago 17th; titles verified against arXiv `citation_title` and Crossref on 2026-10-24)

1. Vaswani et al., "Attention Is All You Need," NeurIPS 30 (2017). arXiv:1706.03762 — Def 1 (post-LN).
2. He, Zhang, Ren, and Sun, "Deep Residual Learning for Image Recognition," CVPR 2016. arXiv:1512.03385 — the skip.
3. Ba, Kiros, and Hinton, "Layer Normalization," 2016. arXiv:1607.06450 — Def 3.
4. Zhang and Sennrich, "Root Mean Square Layer Normalization," NeurIPS 32 (2019). arXiv:1910.07467 — Def 3.
5. Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, and Liu, "On Layer Normalization in the Transformer
   Architecture," ICML 2020. arXiv:2002.04745 — Thm 2's measurement.
6. Dong, Cordonnier, and Loukas, "Attention Is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with
   Depth," ICML 2021. arXiv:2103.03404 — Thm 1's reading, Thm 3.
7. Veit, Wilber, and Belongie, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks," NeurIPS 29
   (2016). arXiv:1605.06431 — Thm 2.
8. Elhage et al., "A Mathematical Framework for Transformer Circuits," *Transformer Circuits Thread* (2021).
   https://transformer-circuits.pub/2021/framework/index.html — Def 4, Thm 3, Prop 4 (blog).
9. Olsson et al., "In-context Learning and Induction Heads," *Transformer Circuits Thread* (2022). arXiv:2209.11895 —
   §6.
10. Geva, Schuster, Berant, and Levy, "Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021.
    arXiv:2012.14913 — Prop 1 (ii).
11. nostalgebraist, "interpreting GPT: the logit lens," LessWrong (2020).
    https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens — §6 (blog).
12. Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman, and Steinhardt, "Eliciting Latent Predictions from
    Transformers with the Tuned Lens," 2023. arXiv:2303.08112 — §6.
13. Brody, Alon, and Yahav, "On the Expressivity Role of LayerNorm in Transformers' Attention," Findings of ACL 2023.
    arXiv:2305.02582 — Prop 3.
14. Geshkovski, Letrouit, Polyanskiy, and Rigollet, "A Mathematical Perspective on Transformers," 2023.
    arXiv:2312.10794 — Thm 1's silent regime (clusters).
15. Seneta, *Non-negative Matrices and Markov Chains*, 2nd ed. (Springer, 1981). https://doi.org/10.1007/0-387-32792-4
    — Def 2, Thm 1 (i).
16. Dobrushin, "Central Limit Theorem for Nonstationary Markov Chains. I," *Theory of Probability and Its
    Applications* 1, no. 1 (1956). https://doi.org/10.1137/1101006 — Def 2.

## 11. Phase B notes

- **Numbering in page order:**
  D1, P1, D2, T1, P2, D3, P3, C1, T2, D4, T3, P4, P5.
  - Proofs go in separate `<TheoremBlock type="proof">` blocks.
  - `TheoremBlock` titles carry no citations; attribute in the prose.
- **Precision.** Page numbers are quoted to 3 decimals from the 6-decimal values above.
- **MDX:** backtick `[A][B] … [A] → [B]`, since it contains brackets inside JSX prose; use `\lt`/`\gt` in math.
- **Lab data:** regenerate the JSON from `viz_payload()`, never hand-edit it.
- **Graph:** flip the node to `published`, and remove its title from `curriculum.ts`'s `planned`.
- **Re-verify** reference 5's author list and reference 12's venue at the B3 gate.
