# Brief — `fine-tuning-low-rank-adaptation`

**Fine-Tuning and Low-Rank Adaptation.** This is roadmap #20 (domain `classification-sentiment`). Its prerequisites are
published: `contextual-embeddings-masked-lm` (#12, the encoder) and `linear-text-classifiers` (#17, the LR head). It is a
leaf node. Topic 19 ends by promising that "the next topic, fine-tuning, trains such an encoder", meaning one whose
every value already depends on its neighbors and on the aspect. It is not on the §6 watch-list. A grep of every
sibling's `main` found no LoRA, adapter, or PEFT content (2026-10-25).

**Status: Phase A harness built (2026-10-25).**
- The scaffold checkpoint was approved by Jonathan with the recommended answers to §5: `advanced`; §7 and §8 folded
  into one section; the LoRA forgetting finding stated plainly; the span readout.
- The cited statements were read in the papers themselves (§7).
- `notebooks/fine-tuning-low-rank-adaptation/fine_tuning_low_rank_adaptation.py` runs its harness in ≈ 2¼ min under
  `-W error`.
- The notebook (25 cells) exits 0 under `jupyter execute` in ≈ 2¼ min. The Phase A checkpoint was approved by Jonathan
  (2026-10-25).
- **Phase B** shipped the MDX, `LowRankAdaptationLab` (5 tabs), and the graph and curriculum flip.
  - Gates: `pnpm test` 39/39, `astro sync`, `validate`, `audit:cross-site` (7 deferred reciprocals, no drift), and
    `build`.
  - Page: 0 `.katex-error`, 469 KaTeX spans, no `‘`, every tab hydrated, and `scrollWidth` 375 at the mobile preset
    in every tab and toggle.
- Numbers below are the harness's. Every *(proto)* figure in the findings list is superseded by §3a.

**Phase A decisions** (plan mode and the prototype checkpoint, 2026-10-25):
- All four extras: forgetting + L2-SP, the span lemma + LP-FT, intrinsic dimension, and NTK linearization.
- Claude writes every body.
- The main testbed is **Topic 19's aspect rows**. Document tone is reached by aggregation, with one document-level
  fine-tune as the honest negative.
- **The runtime is relaxed to ≈ 2 min for this topic only** (the README asks for well under 60 s). The reason: the
  ~21 s pretrain and ~3 s per fine-tune need ~30 runs for 3-seed spreads, and a 5k-row subsample loses negation.
- Negation: report the shortcut, then run counterfactual augmentation.

**What the prototypes found (*proto*, the filing encoder `FILING_MLM`: V = 879, n = 32, d = 64, 2 layers, 158,447
parameters, float32):**

1. **The encoder backward pass.** `encoder_backward(P, blocks, tok, out, r, caches, dout)` is lifted from
   `cm.mlm_loss_grad` (Topic 12, L305–317). The document-tone loss through it matches central differences to ≈ 10⁻¹⁰
   (float64).
2. **Document tone is a weak testbed.**
   - Bag-of-words LR on the encoder's 879 ids scores 0.745.
   - The frozen mean-pooled probe scores 0.553–0.580.
   - Full fine-tuning scores 0.51–0.68 and overfits (train 0.94, test 0.68).
   - LP-FT scores 0.585, L2-SP 0.51–0.52.
   - A random-init encoder fine-tuned the same way scores 0.570, against 0.579 pretrained.
   - Pretraining on 69k tokens buys nothing at document level, where 1,500 labels have to teach a per-sentence XOR
     through a mean over ~230 tokens.
3. **Aspect rows separate the methods** (span readout: the mean of the final-LN states over the aspect term's tokens; one
   sentence per row, bucketed by length; 1.5 s per epoch on 14,267 rows).

   | Reader | Test acc. | Negated adj. | Masked-LM loss after (from 1.429) |
   |---|---|---|---|
   | Frozen probe | 0.762 | 0.515 | 1.429 |
   | Random-init, fine-tuned | 0.908 | 0.447 | 6.7 |
   | Full fine-tuning, 2 epochs (3 seeds) | 0.961–0.974 | 0.74–0.98 | 2.29–2.48 |
   | Full fine-tuning, 4 epochs | 0.995 | 0.996 | 2.90 |
   | LP-FT | 0.949–0.975 | 0.73–0.85 | **1.80–1.89** |
   | L2-SP, λ = 1 / 0.1 | 0.747 / 0.862 | ~0.5 | 1.46 / 1.60 |
   | LoRA on Wq, Wv, r = 16, α/√r, rate 10⁻² | 0.955 | 0.80 | **5.17** |

   **LoRA forgets more than full fine-tuning here.** That contradicts Biderman et al. (2024) at our scale. It might
   partly be the rate, so it gets a matched-accuracy comparison in the harness before the page says it.
4. **The negation shortcut.**
   - On Topic 19's quadruples, the encoder reads both negated forms perfectly (1.000 / 1.000). It reads the deleted-"not"
     forms ("sales were adverse this period") as neutral, so the quadruple score is 0.000.
   - That frame never occurs without "not" in training, so the encoder learned the frame, not the adjective.
   - Adding the 310 deleted-"not" rows (counterfactual augmentation) gives one seed ≈ 0.98 on all four positions, but
     not another seed.
5. **Costs.**
   - Larger batches buy no throughput: ≈ 10k rows/s at batch sizes 64–256.
   - A 5k random subsample drops negated-adjective accuracy to ≈ 0.5 for every method.
   - Keeping every polar row plus 1,500 neutral ones (9,283 rows, 3 epochs, 2.8 s) keeps quality: 0.959–0.980, with
     negated adjectives 0.71–0.996.

---

## 0. Placeholder table

| | |
|---|---|
| TITLE | Fine-Tuning and Low-Rank Adaptation |
| SUBTITLE (draft) | A pretrained encoder can be read (a probe), rewritten (full fine-tuning), or nudged inside a low-rank subspace (LoRA). The factored update's first step is a random sketch of the full one, its two factors conserve a hyperbola, and what fine-tuning never touches is exactly what it forgets least. |
| SLUG / DOMAIN | `fine-tuning-low-rank-adaptation` / `classification-sentiment` |
| DIFFICULTY | `advanced` (proposed) |
| FINANCE_CASE_STUDY | `true` |
| PREREQUISITES | `[contextual-embeddings-masked-lm, linear-text-classifiers]`; `aspect-based-sentiment` as a frontmatter connection (no new graph edge) |
| formalMLPrereqs | `svd` (Thm 2, Eckart–Young–Mirsky), `representation-learning` (Def 9.1, the linear probe) |
| formalMLConnections | `double-descent` (Thm 1, Marchenko–Pastur; the lazy/NTK section), `information-geometry` (Fisher metric, for EWC) |
| formalStatisticsPrereqs | `regularization-and-penalized-estimation` (Thm 5, MAP ↔ penalty; Remark 4, ridge's SVD shrinkage) |
| formalCalculusPrereqs | `jacobian` (Thm 2, the multivariate chain rule) |
| formalRAGConnections | `dense-retrieval-dual-encoders` (Thm 2: the rank ceiling of *scores* against the rank of *updates*) |
| REFERENCE_NOTEBOOKS | `notebooks/contextual-embeddings-masked-lm/`, `notebooks/aspect-based-sentiment/` |

All seven sibling slugs are `status: "published"` on GitHub `main` (rechecked 2026-10-25), and their theorem numbers
are confirmed by grepping `<TheoremBlock … number=`:
- formalML `svd` Thm 2 (Eckart–Young–Mirsky);
- formalML `double-descent` Thm 1 (Marchenko–Pastur);
- formalML `representation-learning` Def 9.1 (linear probe);
- formalStatistics `regularization-and-penalized-estimation` Thm 5 (MAP–penalization) and Remark 4 (ridge's SVD);
- formalCalculus `jacobian` Thm 2 (the multivariate chain rule).

**Scope.**
- No quantization or QLoRA, and no retriever or embedding fine-tuning (both are formalRAG's).
- No instruction tuning, RLHF, or agentic content.
- LoRA here is a rank constraint on the *weight update* ΔW of a generic encoder.

## 1. Section outline (geometric first)

1. **Overview: read, rewrite, or nudge.**
   - Topic 19's handoff.
   - The three adaptations, side by side on one aspect sentence.
   - The probe reads 0.762, full fine-tuning 0.97, and a random-init encoder 0.908. Pretraining is worth the gap
     between the last two, and negation lives in that gap (0.976 vs 0.447).
2. **One weight, two factors (Def 1, Prop 1).**
   - The scalar picture first: w = s·ab, the (a, b) plane, and the level sets ab = const.
   - Rank-r updates, the parameter count r(k + m) against km, merging at inference, and the gauge orbit (AS, S⁻¹B).
   - The rank-≤ r set is not a linear space.
   - Lab view 1.
3. **The first step of a factored update (Thm 1).**
   - The factored gradient, and why A gets none at B = 0.
   - The first step is the full step pushed through A₀A₀ᵀ (a random sketch).
   - Its size scales as s²r under SGD and s√r under Adam, so the step is rank-stable only when s ∝ 1/√r (rsLoRA).
   - Lab view 2.
4. **What two factors conserve (Thm 2).**
   - The hyperbolas a² − b² = a₀²; the matrix AᵀA − BBᵀ = A₀ᵀA₀.
   - GD drifts by O(η²), and Adam breaks the invariant (measured).
   - Lab view 1, again.
5. **Is the update low-rank? (Prop 2.)**
   - The full fine-tuning ΔW's singular spectrum per matrix, against a same-norm Gaussian's Marchenko–Pastur edge.
   - The accuracy of truncated merges, citing formalML `svd` Thm 2.
   - Lab view 3.
6. **What fine-tuning leaves alone (Thm 3, Prop 3).**
   - SGD's update to a linear layer lies in the span of that layer's training inputs. Embedding rows of unseen tokens
     never move.
   - Kumar et al.'s feature distortion, and LP-FT.
   - L2-SP as a Gaussian prior at θ₀.
   - Forgetting is the rise of held-out masked-LM loss.
   - The negation shortcut is the in-distribution trap seen from the other side.
   - Lab view 4.
7. **Two lenses on why a small update suffices (Prop 4, Prop 5, Cor 1).** This merges the scaffold's §7 and §8.
   - A count-sketch subspace gives an unbiased random step. No D ≤ 4,096 reaches d₉₀, but the pretrained encoder leads
     at every D.
   - Linearize around the LP-FT start, since a zero head has a zero Jacobian; kernel GD follows.
   - The LoRA kernel is the A₀-sketched NTK (by Thm 1).
   - GD stays in the kernel regime (agreement 0.992, K moves 6.5%), while Adam leaves it (K moves 169%, test 0.621).
8. (folded into 7)
9. **Reading negation in filings, inside `<FinanceCaseStudy>`.**
   - Per-construction accuracy.
   - The shortcut, then augmentation, scored on the quadruples (3 seeds).
   - Aspect reader → document tone via Topic 19's Prop 2, against 0.637 / 0.724 / 0.739 / 0.771 and the 0.890
     ceiling.
   - The document-level fine-tune as the honest negative.
   - Lab view 5.
10. **Where this sits / Honest caveats (`RigorFlag`) / Implementation.**

## 2. Statements (proofs sketched; full in the Phase A brief)

**Convention.** We use the codebase's row-vector convention: a layer maps x ∈ ℝ^k to xW, with W ∈ ℝ^{k×m}. Hu et al.
write the column-vector transpose (their B zero, A random). We keep the *roles*: A is random, B starts at zero, and
ΔW = s·AB.

**Def 1 (low-rank adaptation).** W = W₀ + s·AB with A ∈ ℝ^{k×r} (A₀ i.i.d. N(0, σ²), σ² = 1/k), B ∈ ℝ^{r×m} (B₀ = 0),
and scale s (α/r in Hu et al.; α/√r in rsLoRA). W₀ is frozen. A probe freezes everything but the head; full fine-tuning
frees every parameter.

**Prop 1 (what the factorization buys).**
- (a) rank ΔW ≤ r, with r(k + m) parameters instead of km.
- (b) x(W₀ + sAB) is one matrix after merging, so inference cost is unchanged.
- (c) (AS, S⁻¹B) gives the same ΔW for every invertible S, so the factors are not identifiable.
- (d) The sum of two rank-r updates can have rank 2r.

*Proof:* direct.

**Thm 1 (the factored gradient and the first step).** Let G = ∂L/∂W at W₀ + sAB.
- (a) ∂L/∂A = s·G Bᵀ and ∂L/∂B = s·Aᵀ G.
- (b) At B = 0, ∂L/∂A = 0. One SGD step gives ΔW₁ = −η s² A₀A₀ᵀ G: the full step −ηG pushed through the rank-r map
  A₀A₀ᵀ.
- (c) E[A₀A₀ᵀ] = rσ² I_k, so E ΔW₁ = −η s² r σ² G. Under s = α/r this vanishes as 1/r, and under s = α/√r it is
  independent of r.
  - **Added:** E‖A₀A₀ᵀ − rσ²I‖_F² = rσ⁴k(k + 1), from Gaussian fourth moments.
  - So a single draw sits a relative √((k + 1)/r) from the mean step, which is 8× at r = 1, k = 64. The sketch is
    noisy until r is comparable to k.
  - At the encoder this shows directly. The SGD α/√r norm falls 0.0428 → 0.0081 from r = 1 to r = 64, toward the
    mean step 0.0057.
- (d) Under Adam (or sign descent), the first step is B₁ = −η·sgn(Aᵀ₀G), so ΔW₁ = −η s A₀ sgn(A₀ᵀG), and
  η s σ_min(A₀)√(rm) ≤ ‖ΔW₁‖_F ≤ η s σ_max(A₀)√(rm).
  - For Gaussian A₀ with r ≤ k, σ(A₀) ∈ σ(√k ± √r) with high probability (Davidson and Szarek; cite).
  - So ‖ΔW₁‖_F ≍ η s √(rkm)·σ, which is rank-stable iff s ∝ 1/√r again.

*Proof:*
- (a) is the chain rule, citing formalCalculus `jacobian` Thm 2.
- (b) and (c) are substitution plus E[A A ᵀ]_{ij} = Σ_l E[A_il A_jl].
- (d): ‖A₀S‖_F² = Σ_j ‖A₀ s_j‖² with ‖s_j‖² = r, bounded by the extreme singular values.

*Cite* Kalajdzievski (2023), Thm 3.2: "rank-stabilized iff γ_r ∈ Θ_r(1/√r)", in expectation over the initialization.
- Its setting: **SGD** (the Appendix A proof), any fixed number of updates n, as r → ∞ with d₁ and d₂ fixed, at
  leading order in γ_r.
- It says nothing specific about the first step, and nothing about Adam.
- Thm 1(c) is its exact n = 1 case. Thm 1(d), the Adam/sign step, is **ours**; the page says so.
- Hu et al.'s text uses Gaussian A, but their loralib code uses Kaiming-uniform A. Malladi et al. swap the roles of
  A and B.

*Harness:*
- (b) exactly, on random instances.
- (c) over an A₀ ensemble.
- (d) the bounds, on random instances.
- The trained ‖ΔW‖ against r under both scalings.

**Thm 2 (balancedness).** Cited as Du, Hu, and Lee (2018), Thm 2.2 (linear layers, gradient flow) and the exact
one-step identity, their Appendix B Claim B.1; and as Arora, Cohen, and Hazan (2018), Eq. 9 (the invariant for t ≥ t₀,
proved in App. A.1). Under gradient flow on (A, B),
d/dt (AᵀA − BBᵀ) = 0.
- With B₀ = 0, AᵀA − BBᵀ = A₀ᵀA₀ for all t. In particular ‖A‖_F² − ‖B‖_F² = ‖A₀‖_F², so A must grow as B grows.
- One GD step of size η changes the invariant by exactly η²s²(BGᵀGBᵀ − AᵀGGᵀA).
- Scalar case (k = m = r = 1): a² − b² = a₀², a hyperbola, which meets the minimizer curve s·ab = w* at a single point.

*Proof:* differentiate both Gram matrices and compare. Both derivatives equal −s(AᵀGBᵀ + BGᵀA).

*Harness:*
- The conservation under small-η GD, with a drift slope of 2 in log η.
- Adam's drift, measured.

**Prop 2 (truncating the full update).**
- For any ΔW with singular values σ₁ ≥ …, the truncated SVD ΔW_r minimizes ‖ΔW − M‖_F over rank-≤ r M, with error
  Σ_{i>r} σ_i². This is cited (formalML `svd` Thm 2).
- We report each matrix's spectrum against a same-Frobenius-norm Gaussian matrix's Marchenko–Pastur edge (formalML
  `double-descent` Thm 1), and the accuracy of W₀ + ΔW_r merges.
- Empirical: the page will say whether full fine-tuning's update *is* low-rank here, whichever way it comes out.

**Thm 3 (the span lemma).**
- (a) For a linear map applied to fixed inputs x₁, …, x_n (rows of X) with output gradients δ_i, every SGD step is
  ΔW = −η Σ x_iᵀδ_i. So the columns of W_t − W₀ lie in span{x_i}, and v W_t = v W₀ for every v ⊥ span{x_i}.
- (b) In the encoder: an embedding row whose token never occurs in the fine-tuning data has zero gradient at every step,
  so it is unchanged. That holds under SGD, under Adam (zero gradients leave both moments at 0), and under L2-SP (at
  θ₀), but **not** under decoupled weight decay.
- (c) **Corollary.** For a two-layer linear network f(x) = (xB)v trained from (B₀, v₀) on inputs spanning S, B changes
  only on S. Off S, the features stay B₀'s while v moves, so out-of-distribution error grows with the head's starting
  error (Kumar et al. 2022). LP-FT (v₀ = the probe) shrinks that error.

*Proof:* (a) and (b) direct. For (c), the S-restriction is Kumar et al.'s Lemma A.3 (B₀x = B_ft(t)x for x ∈ S⊥), and
their Lemma A.4 is Thm 2's invariant for (v, B).
- Cite their Thm 3.3 (the OOD lower bound, with cos θ_max(R₀, S⊥) > 0) without reproving it.
- We prove and test their **Prop 3.7**: with perfect features B₀ = UB*, the OOD loss is 0 for all t from v₀ = the
  probe's head, and > 0 from a Gaussian v₀.

*Harness:*
- (a) on a linear toy.
- (b) as a count of embedding rows changed, which equals the number of distinct fine-tuning token ids, under every
  method.
- (c) on the toy.

**Prop 3 (L2-SP is a prior at θ₀; Li, Grandvalet, and Davoine 2018).**
- (λ/2)‖θ − θ₀‖² = −log N(θ; θ₀, λ⁻¹I) + const, so the penalized optimum is MAP under that prior (formalStatistics
  Thm 5).
- For a locally quadratic loss with Hessian H and full-FT optimum θ*, θ̂ − θ₀ = (H + λI)⁻¹H(θ* − θ₀): ridge shrinkage
  of the displacement along H's eigenvectors (formalStatistics Remark 4).
- EWC replaces λI by the Fisher (cited, not run).

*Proof:* direct.

*Harness:* the quadratic identity on random instances.

**Prop 4 (a count-sketch subspace is an unbiased step).**
- Let θ = θ₀ + Sz, where S ∈ ℝ^{P×D} has one entry ε_p = ±1 per row, at a uniform column h(p).
- Li et al. (2018) normalize P's columns to unit length, and Aghajanyan, Gupta, and Zettlemoyer (2021) use Fastfood
  with learned per-layer scales (SAID). Our unnormalized ±1 sketch is the one with E[SSᵀ] = I. Adam's per-coordinate
  scaling makes the column scale irrelevant to training.
- Then ∇_z L = Sᵀg with E‖Sᵀg‖² = ‖g‖², and the z-step Δθ = −ηSSᵀg has E[SSᵀ] = I_P. The step is unbiased, with
  collision variance ∝ P/D.
- d₉₀ = the least D whose accuracy is ≥ 0.9× full fine-tuning's, which is empirical.

*Proof:* signs are independent with mean zero.

*Harness:*
- Both identities, over a sketch ensemble.
- d₉₀ for the pretrained encoder against the random-init one.

**Prop 5 (linearized fine-tuning is kernel gradient descent).**
- Define f_lin(x; θ) = f(x; θ₀) + J(x)(θ − θ₀), with squared loss on one-hot targets.
- Under gradient flow, ḟ(X) = −K(f(X) − Y) with K = JJᵀ. So f_t(X) = Y + e^{−tK}(f₀(X) − Y), and on a test point
  f_t(x) = f₀(x) − K(x, X)K⁻¹(I − e^{−tK})(f₀(X) − Y).
- **Corollary 1:** at B₀ = 0, the LoRA kernel is K_LoRA(x, x′) = s² Σ_W tr(G_xᵀ A₀A₀ᵀ G_{x′}). Its expectation is s²rσ²
  times the full kernel of the adapted matrices (by Thm 1(c)).

*Proof:* the chain rule, then a linear ODE.

*Cite* Jacot, Gabriel, and Hongler (2018), and Malladi et al. (2023), whose setting is prompt-based and uses SignGD
kernels for Adam.

*Harness:*
- Kernel-GD equivalence on a small instance.
- The LoRA-kernel expectation.
- On a ~600-row subsample around the LP-FT start: the kernel predictor's accuracy and its agreement with real LoRA
  fine-tuning, and ‖K_T − K₀‖/‖K₀‖.

## 3. Experiment grid (≈ 2 min; one `@lru_cache`d `fine_tune(cfg)` per frozen config)

**Data.** Topic 19's `rows("train")`, with every polar row plus 1,500 neutral ones (9,283 rows), evaluated on all 14,179
test rows.
- The input is one sentence; the readout is the mean of the final-LN states over the aspect span.
- A 3-class softmax head (zero init; the probe's solution for LP-FT).
- Adam with the rate decaying linearly to 0.

**Per run:** test accuracy, per-construction accuracy, the quadruple positions, held-out masked-LM loss (fixed corruption
stream, on test-document windows), ‖ΔW‖ per matrix, and ECE₁₅.

| Block | Runs | ≈ s |
|---|---|---|
| probe (`ltc.lr_fit` on frozen features) | 1 | 1 |
| full / LP-FT / LoRA (r = 16, α/√r) × 3 seeds | 9 | 27 |
| LoRA r ∈ {1, 4, 64} × {α/r, α/√r}, + r = 16 at α/r | 6 | 17 |
| L2-SP λ ∈ {0.1, 1} | 2 | 6 |
| random-init full fine-tuning | 1 | 3 |
| augmentation: full × 3 seeds, LP-FT, LoRA | 5 | 15 |
| intrinsic-D D ∈ {64, 512, 4096} × {pretrained, random} | 6 | 17 |
| NTK: kernel + LoRA fine-tuning on 600 rows | 1 | 3 |
| document-level fine-tune (the honest negative) | 1 | 7 |
| pretrain + corpora + theory tests + payload | | 30 |

If the harness exceeds ~2.5 min, the first lever is the intrinsic-D grid (two values of D).

## 4. The lab (`LowRankAdaptationLab`, `data/low-rank-adaptation-lab.json`, 5 tabs)

1. **One weight, two factors.**
   - The (a, b) plane, the loss contours of (s·ab − w*)², and the conserved hyperbolas.
   - GD and Adam paths from (a₀, 0), precomputed for a grid of a₀ and η.
   - The readouts: a² − b², and the drift.
2. **Rank and scaling.**
   - ‖ΔW₁‖ (Thm 1, exact) and the trained ‖ΔW‖ against r, for α/r and α/√r.
   - Accuracy against r, with probe, full, and LP-FT reference bands (seed min–max).
3. **What full fine-tuning changed.**
   - The per-matrix singular spectrum against the MP edge.
   - A truncation-rank slider, with the merged accuracy from the payload.
4. **Learning against forgetting.**
   - A scatter of accuracy against Δ masked-LM loss: every method, every seed, with L2-SP, intrinsic-D, and
     random-init marked.
5. **Reading negation.**
   - Per-construction bars for probe, full, LP-FT, LoRA, and random-init.
   - The quadruple positions before and after augmentation.
   - The tone bars (0.637 / 0.724 / 0.739 / 0.771 / document-level fine-tune / aspect-aggregated / 0.890).

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

**Data.** 9,283 training rows: every polar row plus 1,500 neutral ones. 310 counterfactual rows. 14,179 test rows. The
pretrained encoder's held-out masked-LM loss is 1.523078 (500 windows of the first 100 test documents).

**Headline runs** (3 seeds; test accuracy and masked-LM loss after, as mean / min / max):

| Method | Acc. | Masked-LM loss |
|---|---|---|
| Probe (λ = 10⁻⁴) | 0.760773 | 1.523078 |
| Full fine-tuning | 0.982791 / 0.970449 / 0.989492 | 2.681748 / 2.553161 / 2.758515 |
| LP-FT | 0.976750 / 0.974681 / 0.980535 | 1.928428 / 1.889458 / 1.965172 |
| LoRA (r = 16, α/√r, Wq/Wv, rate 10⁻²) | 0.928345 / 0.911348 / 0.955568 | 5.467829 / 5.207720 / 5.704776 |
| Random-init, full fine-tuning | 0.950913 | 6.900702 |

- Negated-adjective accuracy: 0.988938 / 0.920354 / 0.991150 for full fine-tuning against 0.716814 for random-init.
- **Every LP-FT seed Pareto-dominates every LoRA run.** LoRA's best accuracy is 0.962832 (r = 64, α/r), below LP-FT's
  0.974681. LoRA's lowest forgetting is 2.728947 (the slower rate), above LP-FT's 1.965172.
- The plain claim "LoRA forgets more than every full fine-tuning run" fails by a hair: LoRA at the slower rate scores
  0.935327 / 2.728947, against full seed 0 at 0.989492 / 2.758515. So the page states the dominance.
- Update sizes: trained ‖sAB‖ is 28.4 / 37.5 / 45.1 / 61.8 at r = 1 / 4 / 16 / 64 under α/√r, and 28.4 / 27.2 / 23.8 /
  21.4 under α/r. Full fine-tuning (seed 0) moves the same four matrices by only ‖ΔW‖ = 2.686668. Across all six types it moves
7.572545, and E 3.870588.
- So LoRA's update to Wq/Wv is ~17× larger than full fine-tuning's.
- That is the mechanism behind its forgetting here. The update is confined to two matrix types, at Adam's 10× rate,
  so it must be big.

**L2-SP and the slower rates (acc / masked-LM loss).**
- λ = 0.1: 0.905282 / 1.711512.
- λ = 1: 0.757035 / 1.542878.
- Full fine-tuning at rate 3·10⁻⁴: 0.920234 / 1.815103.
- LoRA at rate 3·10⁻³: 0.935327 / 2.728947.

**Thm 1 at the encoder** (four adapted matrices, the LP-FT start, 16 draws, η = 10⁻², r = 1, 2, 4, …, 64):

| Optimizer, scale | ‖ΔW₁‖_F as r runs 1 → 64 |
|---|---|
| Adam, α/√r | 1.284752 1.257667 1.292256 1.323962 1.357638 1.451462 1.625643 |
| Adam, α/r | 1.284752 0.889305 0.646128 0.468091 0.339409 0.256585 0.203205 |
| SGD, α/√r | 0.042792 0.029974 0.021696 0.017224 0.012404 0.009671 0.008070 (mean step 0.005676) |

**Thm 2.**
- On the toy, the drift slope against η at a fixed horizon is 1.003781.
- Under Adam, the headline LoRA run's relative drift is 1.420033.
- GD from the LP-FT start drifts by 2.4·10⁻⁶; Adam from the same start drifts by 1.43.

**Prop 2** (full fine-tuning, seed 0):
- σ₁/edge is 1.76–3.72 over the 12 matrices, and the stable rank is 2.06–5.55.
- Truncated-merge accuracy by rank r:

  | r | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|---|---|
  | Acc. | 0.479935 | 0.808167 | 0.929050 | 0.976585 | 0.986741 | 0.988998 | 0.989421 | 0.989492 |

- Rank 4 keeps 98.7% of the accuracy.
- LoRA trained at r = 4 reaches 0.954158 (α/r) and 0.946752 (α/√r): a rank-4 solution exists, and LoRA does not find
  it in 3 epochs.

**Thm 3.**
- 419 vocabulary rows are never seen in fine-tuning. Full fine-tuning moves exactly the 461 rows it saw and none of the
  419.
- Kumar toy: OOD loss 1.6·10⁻³¹ (LP-FT) against 6.848272 (random head).
- SGD's span residual is 2·10⁻¹⁵; Adam's is 1.01.

**Prop 4.** Sketch accuracy at D = 64 / 512 / 4,096:
- pretrained 0.753509 / 0.819522 / 0.870372 (90% of full fine-tuning is 0.884512);
- random-init 0.722265 / 0.786233 / 0.822978 (90% of 0.950913 is 0.855822).

d₉₀ > 4,096 for both encoders, and the pretrained one leads at every D.

**Prop 5** (120 training rows of one length, 240 test rows, 300 full-batch steps). Start: acc 0.766667, loss 0.619552.

| Arm | Test acc. | Train loss | Agrees with kernel | Logit gap | Kernel moved |
|---|---|---|---|---|---|
| Kernel GD | 0.745833 | 0.599021 | — | — | — |
| LoRA, GD | 0.745833 | 0.598913 | 0.991667 | 0.179997 | 0.065491 |
| LoRA, Adam | 0.620833 | 0.000052 | 0.658333 | 0.994525 | 1.689079 |

Adam memorizes the 120 rows and leaves the kernel regime.

**Negation.**
- Full fine-tuning quadruple positions per seed (not ADJ₊, not ADJ₋, ADJ₊, ADJ₋):

  | Seed | not ADJ₊ | not ADJ₋ | ADJ₊ | ADJ₋ |
  |---|---|---|---|---|
  | 0 | 1.0 | 1.0 | 0.0075 | 0.0 |
  | 1 | 0.944 | 0.926 | 0.4085 | 0.1715 |
  | 2 | 1.0 | 1.0 | 0.128 | 0.0 |

- With the 310 counterfactual rows, every full seed scores 1.000 at all four positions.
- Augmented LP-FT scores 0.890 on whole quadruples; augmented LoRA 0.010.

**Finance.**
- Document readers: net tone 0.636667, categories 0.724000, unigram LR 0.738667, bigram LR 0.771333, ceiling
  0.889919.
- Document-level fine-tuning: 0.558667 (train 0.664000; probe 0.529333). Random-init: 0.536000 (probe 0.514000).
- Aspect fine-tuning aggregated to tone: 0.880000 (full seed 0) and 0.878667 (augmented), within 2 SE (0.0081) of the
  ceiling.

## 5. Checkpoint answers (approved)

1. `advanced`.
2. §7 (intrinsic dimension) and §8 (NTK) become one section, "Two lenses on why a small update suffices".
3. The LoRA finding is stated as the Pareto dominance in §3a, against Biderman et al. (2024).
   - Their forgetting measure is HellaSwag / WinoGrande / ARC-C on Llama-2-7B.
   - LoRA's best rate is about 10× full fine-tuning's (their App. B), as here.
   - Their α = 2r against peft's α/r is the same scaling issue as Thm 1.
4. The span readout, with one sentence on Sun, Huang, and Qiu's pair input.

## 5b. rigorFlag (draft)

1. **The model is a toy.** The encoder is a 158k-parameter toy pretrained from scratch on 69k tokens of synthetic
   filings. Nothing here is measured on a pretrained language model, and the LoRA findings are about this scale.
2. **Forgetting means one thing here.** It is held-out masked-LM loss on the same synthetic filings, not downstream
   capability (Biderman et al. use HellaSwag, WinoGrande, and ARC-C on Llama-2-7B).
   - Each method runs at one rate for three epochs, with LoRA at 10× full fine-tuning's rate.
   - "Every LP-FT seed dominates every LoRA run" is this grid's result, not a general law.
   - LoRA's large updates (‖sAB‖ ≈ 45 on Wq/Wv against full fine-tuning's 2.7) are the measured mechanism.
3. **Thm 1(d), Adam's sign step, is ours.** Kalajdzievski's rank-stabilization theorem is proved for SGD as r → ∞.
   Our r ≤ 64 = d never reaches that limit, and at a fixed rate neither scaling wins on accuracy.
4. **Thm 2 and Thm 3(a) are gradient-descent facts.** Every encoder run uses Adam, which breaks both; the harness
   measures by how much. What survives under Adam is Thm 3(b): rows with zero gradient never move.
5. **d₉₀ and the kernel comparison are empirical.**
   - d₉₀ comes from a three-point grid with an unnormalized ±1 count sketch (not Fastfood or SAID), and neither
     encoder reaches it by D = 4,096.
   - The kernel comparison runs on 120 rows. There, GD at a stable rate barely trains (loss 0.620 → 0.599 in 300 steps),
     so "GD follows its kernel" holds in a regime that learns little.
6. **The training set is a subsample.** It keeps every polar row and 1,500 of the 6,484 neutral ones, and three seeds
   spread full fine-tuning over 0.970–0.989.
   - The quadruples are a 2,000-of-18,120 sample.
   - The counterfactual rows add exactly the deleted-"not" frame the quadruples test. So augmentation shows the encoder
     *can* read the frame, not that it generalizes to frames it never saw.
7. **Tone uses gold structure.** Tone by aggregation takes Topic 19's gold aspect spans and its tone-sentence rule.
   Document-level fine-tuning (0.559) is one configuration: mean-pooled, three epochs, no tuning beyond the rate.
8. **Runtime.** The README's 60 s rule is waived for this topic: the harness runs in ≈ 2¼ min.

## 6. Notebook cell map (25 cells)

| # | Kind | Content |
|---|---|---|
| 1 | md | Title, the seven points, how to run |
| 2 | code | Imports, the encoder's size, row counts, the pretrained masked-LM loss |
| 3–4 | md / code | §1: probe vs random-init vs full fine-tuning (3 seeds) |
| 5–6 | md / code | §2 Def 1, Prop 1: the scalar paths and their invariant |
| 7–10 | md / code / md / code | §3 Thm 1: first-step norms against r, the fluctuation law, the trained rank sweep |
| 11–12 | md / code | §4 Thm 2: toy drift and its slope, Adam's drift |
| 13–14 | md / code | §5 Prop 2: spectra, truncated merges, LoRA at r = 4 |
| 15–18 | md / code / md / code | §6 Thm 3, Prop 3: span residuals, embedding rows, Kumar toy, spreads, L2-SP and the slow rates |
| 19–20 | md / code | §7 Prop 4, Prop 5: d₉₀ grid, the kernel experiment |
| 21–22 | md / code | §8: negation quadruples before and after augmentation; tone |
| 23–24 | md / code | The whole harness |
| 25 | md | Where this sits |

## 7. References (verified 2026-10-25 against the papers)

- Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
  "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9
  (arXiv:2106.09685).
  - §4.1: A Gaussian, B zero, scale α/r.
  - §7.1, Table 5: {Wq, Wv} best under a fixed budget.
  - §7.2: the low "intrinsic rank".
- Kalajdzievski, Damjan. "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA." arXiv:2312.03732, 2023.
  https://arxiv.org/abs/2312.03732. Def 3.1 and Thm 3.2 (SGD).
- Du, Simon S., Wei Hu, and Jason D. Lee. "Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are
  Automatically Balanced." NeurIPS 31 (2018).
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract.html.
  Thm 2.2; Claim B.1.
- Arora, Sanjeev, Nadav Cohen, and Elad Hazan. "On the Optimization of Deep Networks: Implicit Acceleration by
  Overparameterization." ICML 2018, PMLR 80:244–253. https://proceedings.mlr.press/v80/arora18a.html. Eqs. 7 and 9.
- Kumar, Ananya, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. "Fine-Tuning can Distort Pretrained
  Features and Underperform Out-of-Distribution." ICLR 2022. https://arxiv.org/abs/2202.10054.
  Thm 3.3; Lemmas A.3–A.4; Prop 3.7.
- Aghajanyan, Armen, Sonal Gupta, and Luke Zettlemoyer. "Intrinsic Dimensionality Explains the Effectiveness of
  Language Model Fine-Tuning." ACL-IJCNLP 2021, 7319–7328. https://doi.org/10.18653/v1/2021.acl-long.568.
  - Anthology author order; arXiv lists Zettlemoyer second.
  - d₉₀ is measured against the fine-tuned model's evaluation metric.
- Li, Chunyuan, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. "Measuring the Intrinsic Dimension of Objective
  Landscapes." ICLR 2018. https://arxiv.org/abs/1804.08838.
- Malladi, Sadhika, Alexander Wettig, Dingli Yu, Danqi Chen, and Sanjeev Arora. "A Kernel-Based View of Language Model
  Fine-Tuning." ICML 2023, PMLR 202:23610–23641. https://proceedings.mlr.press/v202/malladi23a.html.
  - Def 3.2: kernel behavior. Def 4.2: the A-SignGD kernel.
  - Thm 5.5: the prompt-based setting.
  - Thm 7.2: the SGD-LoRA kernel ≈ the eNTK w.h.p.
- Jacot, Arthur, Franck Gabriel, and Clément Hongler. "Neural Tangent Kernel: Convergence and Generalization in Neural
  Networks." NeurIPS 31 (2018), 8571–8580.
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html.
- Biderman, Dan, et al. (12 authors). "LoRA Learns Less and Forgets Less." TMLR, August 2024.
  https://openreview.net/forum?id=aloEru2qCG.
- Li, Xuhong, Yves Grandvalet, and Franck Davoine. "Explicit Inductive Bias for Transfer Learning with Convolutional
  Networks." ICML 2018, PMLR 80:2825–2834. https://proceedings.mlr.press/v80/li18a.html.
  - Eq. 3 penalizes the new head with β; ours uses β = 0, which the page says.
- Kirkpatrick, James, et al. "Overcoming Catastrophic Forgetting in Neural Networks." PNAS 114, no. 13 (2017):
  3521–3526. https://doi.org/10.1073/pnas.1611835114.
- Sun, Chi, Luyao Huang, and Xipeng Qiu. "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary
  Sentence." NAACL-HLT 2019, 380–385. https://doi.org/10.18653/v1/N19-1035.
- Kaushik, Divyansh, Eduard Hovy, and Zachary C. Lipton. "Learning the Difference that Makes a Difference with
  Counterfactually-Augmented Data." ICLR 2020. https://arxiv.org/abs/1909.12434.
- Gardner, Matt, et al. "Evaluating Models' Local Decision Boundaries via Contrast Sets." Findings of EMNLP 2020,
  1307–1323. https://doi.org/10.18653/v1/2020.findings-emnlp.117.
- Vershynin, Roman. "Introduction to the Non-Asymptotic Analysis of Random Matrices." In *Compressed Sensing: Theory and
  Applications*, ed. Y. Eldar and G. Kutyniok, 210–268. Cambridge University Press, 2012.
  https://doi.org/10.1017/CBO9780511794308.006.
  - Thm 5.32 (Gordon) and Cor 5.35, for σ(A₀) ∈ √k ± √r.
  - Davidson and Szarek's theorem number could not be verified, so they are not cited.
- Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding." NAACL-HLT 2019, 4171–4186. https://doi.org/10.18653/v1/N19-1423.
- Houlsby, Neil, et al. "Parameter-Efficient Transfer Learning for NLP." ICML 2019, PMLR 97:2790–2799.
  https://proceedings.mlr.press/v97/houlsby19a.html.
