# Brief — `attention-as-kernel-smoothing`

**Attention as Kernel Smoothing.** Roadmap #9. Prerequisite: `neural-language-models` (#5, PR #11), published. This
is the first `attention-transformers` topic and a prerequisite of `positional-encodings-geometry` (#10),
`transformer-block-residual-stream` (#11), and `aspect-based-sentiment` (#19).

**Phase A** (this brief and the verified notebook), 2026-10-23:
- The harness passes 19/19 under Python 3.15.0, numpy==2.5.4, and scipy==1.18.1, in ≈ 17 s.
- The narrative notebook (29 cells) executes with exit 0 in ≈ 21 s, including environment and kernel start, and is
  committed without outputs.

**Scope decisions** (Jonathan, 2026-10-23):
- **In:** every optional result:
  - the circle theorem and the trained denoiser;
  - positive random features, with causal linear attention as a recurrence;
  - a learned kernel for the document cache;
  - one head's rank bound.
- **Checks:** internal only:
  - Topic 5's complex step and central differences for every hand-written gradient;
  - `scipy.special`;
  - numpy's FFT;
  - Monte Carlo.
- **Protocol:**
  - two toys trained from scratch in NumPy (the circle denoiser, Topic 5's long-lag task);
  - the non-parametric kernel cache over each document's history, mixed with Topic 3's KN;
  - β, λ, and the learned kernel's epoch are chosen on Topic 5's dev documents;
  - one seed, and no attention language model.
- **Finance:** each EDGAR filing's own history as the memory.
- **Difficulty:** `intermediate`.
- **The scaffold's five open points, all resolved on Claude's recommendations:**
  - the small in-domain gain is reported as it falls;
  - L2 keys;
  - the lab after §5;
  - Prop 3 (iii) kept;
  - every function body written by Claude for Jonathan's review, as in Topics 5–8.

**Changes from the scaffold, each forced by the harness:**
1. **The lag-60 reach is 3.76·10⁻⁵, not ≈ 10⁻².** Topic 5's census measures the gradient of the batch-mean loss
   (1/256 per sequence), and one attention layer also dilutes by 1/61. At lag 60, attention sits five orders above
   Elman and below the forget-biased LSTM (1.42·10⁻³). The sweep was extended to lag 480, where attention overtakes
   that LSTM: a forget gate postpones geometric decay, while a lookup has none.
2. **The second denoiser is n = 64, not 128.** n = 128 cost 9.6 s and landed 6.8% below c*. At n = 64 it costs
   2.6 s and lands 0.5% away.
3. **Cor 1's asymptote is approached faster than the scaffold said.** At n = 32, c*/(Cn^{2/5}) is 0.949 for the smooth
   spectrum and 0.816 for the rough one. The scaffold's "far from the asymptote" assert was dropped.
4. **The λ grid was extended.** On the scaffold's grid, dev chose its edge (0.99) at every β. On the extended grid
   (to 0.998), dev chooses β = 2, λ = 0.98, an interior pair. The spike had β = 4, λ = 0.95 on a grid ending at 0.95.
5. **Cor 2's β → ∞ limit is the exact cache of the key, not of the word.** 1,225 pairs of training words (358 words)
   have identical unit vectors in Topic 6's SVD. They are once-seen pseudo-words and first names in fixed templates.
6. **Prop 8 on text is measured by the total-variation error of the cache's weights.** At λ = 0.98 the mixture's
   loss barely registers the approximation, and is not monotone in m.
7. **EDGAR at the dev setting is 4.4982 bpc, not ≈ 4.43**, because the extended grid gives the cache less weight. lzma
   (2.9080) and Topic 5's dynamic RNN with the PPM-C cache (3.6501) still code EDGAR shorter.
8. **The curated EDGAR terms** are the most repeated OOV types of at least three letters. Many are common English
   words the synthetic generator never writes: *other*, *are*, *our*, *that*.

**Source of truth:** `notebooks/attention-as-kernel-smoothing/attention_as_kernel_smoothing.py`. Every number below is
printed or asserted by it, and none may change without it.

**Binding forward promise, and where it is kept.**
- **Topic 5, §"Where this goes next":** "The recurrent state compresses the whole prefix into 128 numbers, and
  Theorem 5 says how fast it forgets. Attention replaces the compression with a weighted lookup over every earlier
  position, and the attention topic reads that lookup as kernel smoothing."
  - The lookup read as kernel smoothing: Props 1–2, Def 3.
  - How fast it forgets: Prop 5 (one hop, diluted by the weight), and Prop 6 with the trained lag model against Topic
    5's own census.
  - The compression returns: Prop 9, priced by Prop 8.

**Thesis.** Attention is a Nadaraya–Watson smoother whose kernel is learned.
- **The kernel** is a Gaussian in the query–key distance, tilted by the keys' norms (§1).
- **The temperature** is the bandwidth. On the circle the trade-off is exact: a Fourier low-pass filter with Bessel
  gains, and a trained layer finds its optimum (§2).
- **The 1/√d scale** holds the kernel's spread fixed as d grows (§3).
- **A lookup's gradient takes one hop**, diluted by competitors, where a recurrent state's is a product of Jacobians
  (§4).
- **Without positions** one layer reads its prefix as a bag (§1). On text, a kernel over a document's earlier tokens
  reads across sentences that no model of Topics 3–5 could see. It helps a little on the synthetic corpus, which
  repeats itself little within a document, and a lot on real filings, by copying words the training text never
  contained (§§5, 7).

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Attention as Kernel Smoothing |
| `{{SUBTITLE}}` | Softmax attention is the Nadaraya–Watson estimator with an exponential kernel: the temperature is a bandwidth whose trade-off is exact on the circle, the 1/√d scale keeps the kernel from collapsing into a lookup, and a weighted lookup over every earlier position reaches in one step what a recurrent state forgets. |
| `{{SLUG}}` | `attention-as-kernel-smoothing` (= graph node id) |
| `{{DOMAIN}}` | `attention-transformers` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[neural-language-models]` |
| `{{FORMALML_PREREQS}}` | `kernel-regression`; connections `gaussian-processes`, `message-passing` |
| `{{FORMALRAG_PREREQS}}` | `hypersphere-vmf-geometry`; connections `retrieval-vs-long-context`, `late-interaction-learned-sparse` |
| `{{STATS_CALC_PREREQS}}` | Stats `kernel-density-estimation`; Calc `fourier-series` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/neural-language-models/`, `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/perplexity-cross-entropy/`, `notebooks/finance-corpus/` |

- **Tags:** `attention`, `kernel-smoothing`, `nadaraya-watson`, `softmax`, `temperature`, `transformers`,
  `random-features`, `linear-attention`, `cache-language-model`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`:** none, since #10, #11, and #19 are still planned.
- **Graph:** all four edges already exist.

**Scope.** This topic is not on the design doc's §6 watch-list, but these sibling topics adjoin it:
- **formalML `kernel-regression`** owns Nadaraya–Watson asymptotics: bias (its Thm 1), variance (Thm 2), AMISE,
  Stone's rate (Thm 4), and the local-linear fix. We cite them, and prove only the exact case on the circle (Thm 1).
- **formalStatistics `kernel-density-estimation`** owns the KDE. Prop 2 uses its Thm 1.
- **formalRAG `retrieval-vs-long-context`** owns attention's Θ(n²) cost, FlashAttention's caveat, and lost-in-the-middle.
  §6 treats linear attention only as kernel factorization.
- **formalRAG `hypersphere-vmf-geometry`** owns the vMF density, its Bessel normalizer (Thm 2), and the mean
  resultant length (Prop 2). Thm 1's ρ₁ is that page's mean resultant length on the circle.
- **formalRAG `late-interaction-learned-sparse` and `cross-encoders-reranking`** own query–document interaction
  scoring. MaxSim is Prop 4 (i)'s limit, as a pointer only.
- **This topic owns:**
  - the attention–Nadaraya–Watson identity, the mixture-posterior reading, and order blindness;
  - the two temperature limits;
  - the circle theorem and its rate;
  - the 1/√d saturation theorem;
  - the one-hop gradient;
  - the kernel cache and its learned kernel;
  - one head's rank bound;
  - random-feature factorization and linear attention as a recurrence.
- **Out of scope:**
  - positions (#10);
  - the block (#11);
  - masked LMs (#12);
  - scaling (#13);
  - FlashAttention and long-context engineering (formalRAG);
  - attention as explanation (#19).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | trained at lag 60, one attention layer reads the bit at lag 600 with accuracy 1.000; at initialization its gradient falls like 1/(L + 1) (2.2·10⁻⁴ at lag 10, 4.9·10⁻⁶ at 480) while Elman's falls to 9.4·10⁻⁴⁶; on EDGAR, a cache over each filing's own words cuts KN from 5.6605 to 4.4982 bpc | — | cells 0–1 |
| 1 | A weighted lookup | two keys equally far from the query, of different lengths: weight ratio 0.701003 = e^{Δ‖k‖²/2τ} | Def 1; Def 2; **Prop 1**; **Prop 2**; **Prop 3** | cells 2–5 |
| 2 | Temperature is bandwidth | the gains ρ₁…ρ₆ at c = 7.3, n = 64: 0.9287, 0.7456, 0.5202, 0.3180, 0.1717, 0.0827 | **Prop 4**; **Thm 1**; **Cor 1**; the trained denoiser | cells 6–9 |
| 3 | Why 1/√d | mean 1 − α_max at d = 4 → 1,024: unscaled 0.571 → 0.041, scaled 0.766 → 0.759 | **Thm 2** | cells 10–11 |
| 4 | A lookup instead of a state | Topic 5's census at lag 60: Elman 2.45·10⁻¹⁰, LSTM (b_f = 3) 1.42·10⁻³, attention 3.76·10⁻⁵ | **Prop 5**; **Prop 6**; the sweep and training | cells 12–15 |
| 5 | Attention over a document | the held-out filing's *multibroker*, *crosspremium*, *outreceivable*: OOV pseudo-words copied from earlier in the document | Def 3; **Cor 2**; the U-curve; the benchmark; Def 4 | cells 16–21 |
| — | The lab | `<AttentionKernelLab client:visible />` | — | cells 26–28 |
| 6 | Kernels that factor | one head's logits are a rank-d_k matrix | **Prop 7**; **Prop 8**; **Prop 9**; the random-feature cache | cells 22–23 |
| 7 | Finance case study: the filing as memory | Apple's *other*: 48.9 bits at its second mention under KN, 12.1 with the cache | Def 3 on EDGAR, open vocabulary | cells 24–25 |
| 8 | Where this goes next | — | #10 (Prop 3; §2's u(θ) is a sinusoidal encoding of frequency one), #11 (several heads, several kernels; the residual stream), #12, #19, formalRAG | — |

- **Lab placement:** after §5. Panel A needs §2, and Panel B needs §§4–5. §§6–7 follow the lab, and Panel B's EDGAR
  documents preview §7.
- **Back-links:**
  - Topic 5's Prop 2 from Thm 2 (ii);
  - its Thm 5, the lag task, and the census from Props 5–6;
  - its Thm 6 from Prop 7;
  - its Thm 1 (smoothing by geometry) from Cor 2;
  - its Prop 6 (the spelling floor) and Topic 4's open-vocabulary code from §7;
  - Topic 6's SVD vectors as the keys.

## 2. Definitions

**Notation.** Rows are tokens.
- Q = XW_Q, K = XW_K, and V = XW_V.
- Scores are s_tj = ⟨q_t, k_j⟩/τ, with τ = √d_k unless stated.
- 𝒥_t is the set of readable positions: all j (bidirectional) or j ≤ t (causal).
- α_t = softmax(s_t) over 𝒥_t, and o_t = Σ_j α_tj v_j.
- V_max = max_j ‖v_j‖.

**Definitions.**
- **Def 1 (kernel smoother; Nadaraya 1964, Watson 1964).** m̂(q) = Σ_j κ(q, k_j)v_j / Σ_j κ(q, k_j), for a positive
  kernel κ. formalML `kernel-regression`'s Def 2 is the case κ = K_h(q − k).
- **Def 2 (scaled dot-product attention; Bahdanau, Cho, and Bengio 2015; Vaswani et al. 2017).**
  softmax(QKᵀ/τ + B)V, where the mask B is 0 on 𝒥_t and −∞ off it.
- **Def 3 (kernel cache).**
  - **Setting.** A document's predicted positions run across its sentences in Topic 3's event order: each word, then
    each sentence's `</s>`. Position t has target w_t and preceding token u_t, which is `<s>` at a sentence start.
  - **Keys.** ê(u) is u's unit row in Topic 6's SVD (`KEY_CFG` = L2, SPPMI₅, d = 100, p = 0) for training words.
    `<s>` gets its own coordinate, and each distinct OOV string gets its own coordinate.
  - **The cache.** p_β(w | t) = Σ_{i<t} α_ti 1[w_i = w], with α_ti ∝ exp(β⟨ê(u_t), ê(u_i)⟩).
  - **The mixture.** λ p_KN + (1 − λ) p_β, and p_KN alone at a document's first position.
  - **Open vocabulary.** p_KN(`<unk>`)S(w) for an OOV w (Topic 4's speller), while the cache's values are raw strings.
  - **Attribution.** Grave, Joulin, and Usunier's (2017) continuous cache, with fixed word vectors in place of RNN
    states. At β = 0 it is Kuhn and De Mori's (1990) unigram cache.
- **Def 4 (learned kernel).**
  - **The scores** become β⟨ê(u_t), ê(u_i)⟩ + ⟨F_tA, F_iB⟩. F is the 102-dimensional feature [SVD vector, 1[`<s>`],
    1[OOV]], and A, B ∈ ℝ^{102×16}, with entries N(0, 0.01²) at the start.
  - **Training.** Adam (lr 0.01, 10 documents per step) on the 300 training documents, minimizing the pointer loss
    −mean_t log Σ_{i<t: w_i = w_t} α_ti over positions whose target occurred earlier (cf. Merity et al. 2017). It is
    KN-free, because in-sample KN probabilities are overfit.
  - **Selection.** Twelve epochs, with the epoch and λ chosen on dev at β = 2.

**Configurations.**
- **Splits:**

  | split | source | documents | predicted tokens | median per document |
  |---|---|---|---|---|
  | train | `generate_corpus(300, 0)` | 300 | 69,312 | 231 |
  | dev | Topic 5's `generate_corpus(40, 2)` | 40 | 9,558 | 238 |
  | held | `generate_corpus(100, 1)` | 100 | 23,055 | 231 |
  | EDGAR | three filings | 3 | 830 | 287 |

  Each split is regrouped with `ng.split_sentences`, and the regrouping is asserted to concatenate to
  `nlm.corpus()[split]`.
- **Grids:**
  - β ∈ {0, ½, 1, 2, 4, 8, 16, ∞};
  - λ ∈ {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998};
  - chosen jointly on dev by mean closed-vocabulary log-loss.
- **Toys:**
  - **denoiser:** σ = ½, smooth spectrum s² = (1, ¼) at k = 1, 2, rough spectrum s² = ¼ at k = 1…6. Configurations
    (smooth, 32), (smooth, 64), (rough, 32), each 1,500 Adam steps on 32 fresh tasks (lr 0.05), averaging M over the
    second half;
  - **lag:** Topic 5's `lag_data`, d_k = d_v = 16, init scales as Topic 5's Elman, 600 Adam steps at lag 60 (batch 64,
    lr 0.01).
- **Streams:** default_rng([SEED, k]) with k ≥ 900, apart from Topics 5–8.

## 3. Statements and proofs

### Proposition 1 (attention is a Nadaraya–Watson smoother)
- **(i)** o_t is Def 1's smoother with κ_τ(q, k) = e^{⟨q,k⟩/τ}.
- **(ii)** κ_τ(q, k) = e^{‖q‖²/2τ} · e^{‖k‖²/2τ} · e^{−‖q−k‖²/2τ}, so o_t is the Gaussian Nadaraya–Watson estimator with
  bandwidth h = √τ (d_k^{1/4} at the default scale) and design weights ω_j = e^{‖k_j‖²/2τ}.
- **(iii)** Equal key norms give the unweighted estimator. On a sphere of radius r, κ_τ = e^{(r²/τ) cos∠(q,k)}, the vMF
  kernel with concentration r²/τ (formalRAG `hypersphere-vmf-geometry`, Def 1).
- **(iv)** o_t ∈ conv{v_j}.

*Proof.* ‖q − k‖² = ‖q‖² + ‖k‖² − 2⟨q, k⟩. The query factor is common to the numerator and denominator. (iv): the
weights are positive and sum to 1. ∎

*Reading.* Tsai et al. (2019) read attention as a kernel smoother. The factorization makes the design weights
explicit: a longer key is a design point of exponentially greater weight.

**Asserted** (`test_attention_is_nadaraya_watson`):
- 50 random instances: (i)–(ii) to 10⁻¹², (iii) on spheres, and (iv);
- two keys at equal distance 1.736396 from the query, with squared norms 5.1082 and 3.6872, get the weight ratio
  0.701003 = e^{Δ‖k‖²/2τ} at τ = 2.

### Proposition 2 (the mixture posterior)
If J ~ π ∝ ω, Q | J ~ N(k_J, τI), and V = v_J, then P(J = j | Q = q) = α_j(q) and E[V | Q = q] = o(q). Equivalently, o
is the conditional mean under p̂(q, v) = Σ_j π_jφ_τ(q − k_j)δ_{v_j}(v), a KDE in q with point masses in v
(formalStatistics `kernel-density-estimation`, Thm 1).

*Proof.* By Bayes' rule, P(J = j | q) ∝ π_jφ_τ(q − k_j). By Prop 1 (ii), π_jφ_τ(q − k_j) ∝ ω_j e^{−‖q−k_j‖²/2τ} ∝ e^{⟨q,k_j⟩/τ}.
Normalize. ∎

*Reading.* The weights are the responsibilities of an isotropic Gaussian mixture with the keys as means: one soft
assignment, then the mean of the assigned values. Gabbur, Bilkhu, and Movellan (2021) give a probabilistic reading of
dot-product attention as inference in such a mixture.

**Asserted** (`test_mixture_posterior`):
- Bayes equals the softmax to 10⁻¹²;
- on 10⁵ draws of the model, V − o(Q) is uncorrelated with 1, Q₁, and Q₂ (each coordinate within 4 standard errors),
  which is the defining property of E[V | Q];
- the smoother that drops the design weights fails the same test.

### Proposition 3 (order blindness)
- **(i)** Bidirectional attention, and any stack of it with position-wise maps, is permutation-equivariant.
- **(ii)** One causal layer's o_t depends on x₁, …, x_{t−1} only through their multiset.
- **(iii)** Two causal layers do not have this property.

*Proof.* (i) The softmax and the sum over j commute with relabeling. (ii) The sum over j < t is symmetric in its terms.
(iii) Earlier positions see different prefixes, so the multiset the second layer reads changes; a counterexample is
asserted. ∎

*Reading.*
- One layer without positions reads *the dog bit the man* as *the man bit the dog*. That is #10's motivation.
- The causal mask itself leaks order, which is why causal language models without positional encodings still learn
  some (Haviv et al. 2022).

**Asserted** (`test_order_blindness`):
- (i) to 10⁻¹² at depths 1 and 3;
- (ii) on the held-out prefix *good morning and welcome to the brightwater analytics* and its reversal (the last word
  kept), through Topic 6's vectors: |Δ| = 2.2·10⁻¹⁶;
- (iii) with two layers, |Δ| = 2.415.

### Proposition 4 (the two temperature limits)
With S* = argmax s, Δ the gap to the runner-up, and R the score range:
- **(i)** ‖o_τ − v̄_{S*}‖ ≤ 2V_max(|𝒥| − |S*|)e^{−Δ/τ}/|S*|;
- **(ii)** ‖o_τ − v̄_𝒥‖ ≤ V_max(e^{R/τ} − 1).

*Proof.*
- (i) The mass off the argmax is ε ≤ (|𝒥| − |S*|)e^{−Δ/τ}/|S*|. Every weight on S* equals (1 − ε)/|S*|, so
  o − v̄_{S*} = Σ_{j∉S*} α_j(v_j − v̄_{S*}), of norm ≤ 2V_maxε.
- (ii) nα_j ∈ [e^{−R/τ}, e^{R/τ}], so |α_j − 1/n| ≤ (e^{R/τ} − 1)/n. Sum against ‖v_j‖ ≤ V_max. ∎

*Reading.*
- (i) is one-step retrieval: separated keys make attention an associative memory, exact up to e^{−Δ/τ}, as in
  Ramsauer et al.'s (2021) modern Hopfield networks.
- MaxSim (formalRAG) is the τ → 0 limit.

**Asserted** (`test_temperature_limits`):
- both bounds on 30 random instances, for τ from 10⁻² to 10²;
- on a two-cluster example each is within a factor of 3 of the truth at its own end: (i) ≈ 1.0 at τ = 0.05, and
  (ii) ≈ 2.02 at τ = 50.

### Theorem 1 (attention on the circle is a Fourier multiplier)
Let θ_j = θ₀ + 2πj/n, with features u(θ) = (cos θ, sin θ), and scores s_ij = c cos(θ_i − θ_j) (M = cI). Let the
values be y_j = f(θ_j) + ε_j, with ε i.i.d. of mean 0 and variance σ². Let A be the bidirectional attention matrix,
and F_k = (1/n)Σ_j f(θ_j)e^{−ikθ_j}.
- **(i)** A is circulant. The Fourier vectors (e^{ikθ_j})_j are its eigenvectors, with eigenvalues
  ρ_k(c) = Σ_{m≡k (n)} I_m(c) / Σ_{m≡0 (n)} I_m(c).
  For c > 0, 0 < ρ_k < 1 = ρ₀ for k ≢ 0. At c = 0, ρ_k = 0.
- **(ii)** R(c) = (1/n)Σ_i E(o_i − f(θ_i))² = Σ_k (1 − ρ_k)²|F_k|² + σ² Σ_{m≡0} I_m(2c) / (n(Σ_{m≡0} I_m(c))²).

*Proof.*
1. A_ij = a_{(j−i) mod n} with a_m = e^{c cos(2πm/n)}/Z, so A is circulant.
2. (Ae^{(k)})_i = e^{ikθ_i}Σ_m a_m e^{2πikm/n}. By the generating function e^{c cos φ} = Σ_{l∈ℤ} I_l(c)e^{ilφ} (DLMF
   10.35.1), Σ_m e^{c cos(2πm/n)}e^{2πikm/n} = nΣ_{l≡−k} I_l(c) = nΣ_{l≡k} I_l(c), since I_{−l} = I_l. The normalizer
   Z is the k = 0 case.
3. **Bounds.** I_l(c) > 0 for c > 0, so ρ_k > 0. |ρ_k| = |Σ_m a_m cos(2πkm/n)| ≤ Σ_m a_m = 1, with equality iff
   cos(2πkm/n) = 1 for every m, that is, k ≡ 0 (mod n). At c = 0, I_l(0) = 0 for l ≠ 0.
4. **Bias.** E o − f = (A − I)f = Σ_k (ρ_k − 1)F_k e^{(k)}. By discrete Parseval (the e^{(k)} are orthogonal with
   squared norm n), (1/n)‖(A − I)f‖² = Σ_k (1 − ρ_k)²|F_k|².
5. **Variance.** (1/n)E‖Aε‖² = σ²tr(AᵀA)/n = σ²Σ_m a_m², since every row is a rotation of a. Σ_m a_m² is step 2 at
   2c, divided by Z². ∎

*Reading.*
- One frequency of position makes attention a low-pass filter. Mode k keeps ρ_k ≈ I_k(c)/I₀(c) of its amplitude.
- ρ₁ is formalRAG's mean resultant length on the circle.
- Raising c, which lowers the temperature, opens the filter.
- This is formalML `kernel-regression`'s trade-off as a closed form. The features u(θ) are #10's sinusoidal encoding
  at frequency one.

**Asserted** (`test_circle_fourier_multiplier`):
- for n ∈ {8, 31, 64, 200} and c ∈ {0, 0.3, 2, 7.3, 40, 300}, numpy's FFT eigenvalues equal the formula to 10⁻¹² (3.3·10⁻¹⁶
  at n = 64, c = 7.3), every Fourier vector is an eigenvector, the bounds hold, and Σa_m² matches;
- ρ₁ = I₁/I₀ to 10⁻¹² at n = 200, c = 7.3;
- the bias equals (1/n)‖(A − I)f‖² to 10⁻¹² for f = cos θ + ½ sin 3θ;
- the risk matches 4,000 noise draws within 4 standard errors.

### Corollary 1 (the bandwidth rate)
For a fixed nonconstant trigonometric polynomial f with power_k = |F_k|² + |F_{−k}|²,
c*(n) ~ (4Bn/(σ²√π))^{2/5}, with B = ¼Σ_k k⁴ power_k. So the angular bandwidth 1/√c* is of order n^{−1/5}, and
R(c*) = O(n^{−4/5}): formalML `kernel-regression`'s AMISE rate in one dimension.

*Proof.*
- **Expansions.** I_k(c)/I₀(c) = 1 − k²/(2c) + O(c⁻²) and I₀(2c)/I₀(c)² = √(πc)(1 + O(c⁻¹)) (DLMF 10.40.1), and the
  aliased terms are super-exponentially small once c = o(n²). So R(c) = B/c² + σ²√(πc)/n + o(c⁻² + √c/n).
- **c* → ∞.** For bounded c the bias term is bounded below, while R(n^{2/5}) → 0, so c* → ∞.
- **Calculus.** Minimizing B/c² + σ²√(πc)/n gives c^{5/2} = 4Bn/(σ²√π). ∎

**Asserted** (`test_circle_rate`): c*(n)/(Cn^{2/5}), with the log–log slope of c* between 10⁶ and 10⁸ (asserted within
0.01 of 2/5) and c*(10⁸) (asserted within 1% of the asymptote):

| spectrum | C | n = 32 | 64 | 128 | 1,024 | 10⁶ | 10⁸ | slope | c*(10⁸) |
|---|---|---|---|---|---|---|---|---|---|
| smooth | 2.636222 | 0.949368 | 0.962383 | 0.971897 | 0.988058 | 0.999252 | 0.999882 | 0.400137 | 4,177.64 |
| rough | 17.513082 | 0.815531 | 0.905726 | 0.929438 | 0.969857 | 0.998104 | 0.999700 | 0.400347 | 27,748.03 |

**The trained denoiser** (unnumbered; Collins et al. 2024).
- **Model.** One attention layer reads tokens (u(θ_j), y_j). Queries and keys see the position only, so the scores are
  u_iMu_jᵀ with M = W_QW_Kᵀ/√2, and the values are y_j. The loss is the mean squared error to the clean f.
- **Population optimum.** Rotation and reflection invariance of the task distribution make it M = c*I, with
  c* = argmin E_f R(c) from Thm 1 (ii) at E power_k = s_k².
- **Collins et al.** prove that a softmax attention unit learns a window that narrows as the tasks' Lipschitz
  constant grows and widens with label noise. Here the optimum is exact, and the trained layer is compared to it.

**Asserted** (`test_trained_denoiser`):
- the gradient against the complex step (10⁻¹⁰ relative) and central differences (10⁻⁶);
- at M = c*I the mean loss over 3,000 fresh tasks equals R(c*) within 4 standard errors;
- after training, each configuration lands within 5% of c*I, with anisotropy below 5%:

| spectrum, n | learned c | Thm 1's c* | ‖M − c*I‖/‖c*I‖ | anisotropy | R(c*) (bias², variance) |
|---|---|---|---|---|---|
| smooth, 32 | 10.005582 | 10.010980 | 0.001214 | 0.001088 | 0.054563 (0.011616, 0.042947) |
| smooth, 64 | 13.320306 | 13.390658 | 0.005285 | 0.000578 | 0.031578 (0.006611, 0.024967) |
| rough, 32 | 57.201356 | 57.129880 | 0.001259 | 0.000138 | 0.140472 (0.033833, 0.106639) |

- the rough spectrum learns a kernel over 3× sharper, and a longer context a sharper one.

### Theorem 2 (why 1/√d)
- **(i)** For independent q and k with i.i.d. coordinates of mean 0 and variance 1, E⟨q, k⟩ = 0 and Var⟨q, k⟩ = d.
- **(ii)** J = diag α − ααᵀ (Topic 5, Prop 2) satisfies ‖J‖₂ ≤ tr J = 1 − ‖α‖² ≤ 2(1 − α_max).
- **(iii)** Let q ~ N(0, I_d) and k₁, …, k_n be i.i.d. N(0, I_d), with n fixed. Unscaled, 1 − α_max → 0 a.s. as
  d → ∞. Scaled by 1/√d, α → softmax(g) in law, with g i.i.d. N(0, 1).

*Proof.*
- (i) The cross terms have mean 0, and E(q_ik_i)² = 1.
- (ii) J is PSD, so ‖J‖₂ ≤ tr J. Then Σα_j(1 − α_j) = 1 − ‖α‖² ≤ 1 − α²_max = (1 − α_max)(1 + α_max).
- (iii) Given q, the scores ⟨q, k_j⟩ are i.i.d. N(0, ‖q‖²), equal in law to ‖q‖g_j. So
  1 − α_max ≤ (n − 1)e^{−‖q‖G}, with G > 0 the gap between the top two g's, a.s. By the SLLN, ‖q‖/√d → 1 a.s., so
  ‖q‖ → ∞ a.s. Scaled, the scores are (‖q‖/√d)g, which converges a.s. to g, and the softmax is continuous. ∎

*Reading.* This is Vaswani et al.'s footnote made exact. Unscaled, the kernel's bandwidth shrinks like d^{−1/2} in
score units, attention collapses to a lookup at initialization, and the gradient through the scores vanishes.

**Asserted** (`test_scaling_and_saturation`):
- (i) with Rademacher coordinates at d ∈ {8, 64, 512};
- (ii) on Dirichlet α, and J equals Topic 5's `softmax_jacobian`;
- (iii) with 600 trials of 16 explicit keys at each d (the unscaled curves fall at every d, the scaled ones stay within
  5% and 15%):

| d | 4 | 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|---|
| mean 1 − α_max, unscaled | 0.570819 | 0.309345 | 0.138696 | 0.072528 | 0.040752 |
| mean 1 − α_max, scaled | 0.766209 | 0.754966 | 0.748249 | 0.751409 | 0.759285 |
| mean ‖J‖₂, unscaled | 0.257100 | 0.249960 | 0.156871 | 0.093075 | 0.055911 |
| mean ‖J‖₂, scaled | 0.191765 | 0.202596 | 0.207893 | 0.206537 | 0.202783 |

### Proposition 5 (one hop)
For one causal layer and k < t, ∂o_t/∂x_k = α_tk[W_V + τ⁻¹(W_Kq_t) ⊗ (v_k − o_t)], on row perturbations. So
‖∂o_t/∂x_k‖ ≤ α_tk(‖W_V‖ + τ⁻¹‖W_K‖‖q_t‖‖v_k − o_t‖). There is no product over the positions in between. With
equal scores, α_tk = 1/(t + 1), counting positions 0…t.

*Proof.*
- x_k enters o_t through v_k = x_kW_V and through s_tk = ⟨q_t, x_kW_K⟩/τ. For k < t it does not enter q_t.
- The softmax's Jacobian column is ∂α_tj/∂s_tk = α_tj(δ_jk − α_tk), so ∂o_t/∂s_tk = α_tk(v_k − o_t). Chain. ∎

*Reading.*
- Topic 5's Thm 5 bounds the RNN's path by a product of t − k step Jacobians.
- The lookup's path has length one, and its only loss is dilution by competitors, which the scores can undo.

**Asserted** (`test_one_hop_jacobian`):
- the closed form against the complex step to 10⁻¹⁰ relative at every (t, k < t) on 10 random layers;
- the bound;
- W_V/(t + 1) exactly at W_Q = 0.

### Proposition 6 (content addressing solves the lag task)
In Topic 5's task (x₀ ∈ {0, 1}, L noise symbols from {2, 3}, then report x₀), take one causal layer read at the last
position, with one-hot inputs, constant scores, values +1, −1, 0, 0, and logits (wo, −wo). Then
o = (1 − 2x₀)/(L + 1), the prediction is right for every L, and the cross-entropy log(1 + e^{−2w/(L+1)}) → 0 as w → ∞.
Scoring the bit symbols β above the noise raises the margin to e^β/(e^β + L).

*Proof.* Direct computation. ∎

**Asserted** (`test_lag_attention`):
- **The construction**, built from Def 2, at L ∈ {1, 10, 60, 600, 6,000}: accuracy 1, the loss equal to the closed form
  (0.177399 at L = 60, w = 50), and the margin 1/(L + 1).
- **The gradients:** the hand-written gradient and the reach against the complex step.
- **Topic 5's census reproduces its page:** 2.45·10⁻¹⁰, 2.42·10⁻⁷, 1.42·10⁻³, each within 0.5%.
- **The sweep** (12 initializations per lag; the lag-60 recurrent column is Topic 5's own `long_lag_census()`):

| lag L | 10 | 30 | 60 | 120 | 240 | 480 |
|---|---|---|---|---|---|---|
| attention | 2.216·10⁻⁴ | 5.709·10⁻⁵ | 3.763·10⁻⁵ | 2.766·10⁻⁵ | 9.805·10⁻⁶ | 4.884·10⁻⁶ |
| attention × (L + 1) | 2.44·10⁻³ | 1.77·10⁻³ | 2.30·10⁻³ | 3.35·10⁻³ | 2.36·10⁻³ | 2.35·10⁻³ |
| Elman | 1.882·10⁻⁴ | 2.784·10⁻⁷ | 2.450·10⁻¹⁰ | 1.373·10⁻¹⁵ | 1.872·10⁻³³ | 9.413·10⁻⁴⁶ |
| LSTM, b_f = 1 | 1.978·10⁻⁴ | 3.285·10⁻⁵ | 2.420·10⁻⁷ | 3.349·10⁻¹¹ | 1.732·10⁻¹⁶ | 1.833·10⁻³¹ |
| LSTM, b_f = 3 | 1.124·10⁻³ | 8.731·10⁻⁴ | 1.425·10⁻³ | 2.359·10⁻⁴ | 1.513·10⁻⁴ | 1.288·10⁻⁷ |

  - attention × (L + 1) stays within a factor of 2 (1.77–3.35·10⁻³);
  - Elman falls at every lag, by 41 orders of magnitude;
  - LSTM (b_f = 3) is above attention at 60 and below it at 480;
  - at lag 60 attention is 10⁵ times Elman and above LSTM-1.
- **Training** by Topic 5's protocol at lag 60: accuracy 1.000 on 2,000 fresh sequences at lag 60, and 1.000 at
  lag 600, with test loss 3.0·10⁻⁶. Topic 5's 0.485 (Elman), 0.531 (b_f = 1), and 1.000 (b_f = 3) are quoted. The
  lag-600 generalization is ours: a lookup never counts steps.

### Def 3, Corollary 2 (the cache's limits), and the benchmark
**Cor 2.** By Prop 4:
- at β = 0, Def 3 is the unigram document cache;
- as β → ∞, it is the exact cache of the preceding token's key: the earlier positions whose key equals ê(u_t), when
  any exist.

Keys are equal for the same word, and also for word pairs that Topic 6's vectors cannot tell apart: 1,225 pairs
(358 of 3,425 words), seen in identical contexts. When u_t's key has not occurred, the cache falls back to the nearest
keys: Topic 5's smoothing by geometry, over a document.

**Asserted** (`test_kernel_cache`):
- **The limits:** both, on 10 held-out documents, position by position.
- **The dev grid** (closed vocabulary; λ tuned on dev per β), with the held-out loss at the same (β, λ):

| β | 0 | ½ | 1 | **2** | 4 | 8 | 16 | ∞ |
|---|---|---|---|---|---|---|---|---|
| dev λ | 0.99 | 0.99 | 0.98 | **0.98** | 0.98 | 0.99 | 0.99 | 0.99 |
| dev nats | 2.789824 | 2.789375 | 2.788679 | **2.787046** | 2.787093 | 2.791762 | 2.792470 | 2.793682 |
| held nats | 2.836120 | 2.835278 | 2.830595 | **2.828196** | 2.828354 | 2.838605 | 2.840762 | 2.842359 |

  The bandwidth U-curve: dev picks an interior β, and on held out it beats both limits.
- **The benchmark** (held out; standard errors clustered by document):
  - KN 2.855053 nats per token (se 0.025747);
  - KN ⊕ cache 2.828196 (se 0.024641);
  - the paired gain is 0.026857 (se 0.002536), 10.6 standard errors.
- **By class:**
  - repeated targets (41.8694% of positions): 1.797762 → 1.705777;
  - first mentions: 3.616584 → 3.636636 (the cache's weight costs them a little).
- **By distance back to the previous mention:**

| band | share of repeats | in an earlier sentence | KN | KN ⊕ cache |
|---|---|---|---|---|
| 1–12 | 0.362789 | 0.890919 | 1.426940 | 1.380223 |
| 13–50 | 0.454884 | 0.991801 | 1.763580 | 1.681649 |
| > 50 | 0.182327 | 1.000000 | 2.620894 | 2.413753 |

  The gain grows with distance. On repeats whose previous mention lies in an earlier sentence (where Topic 5's RNN,
  reset per sentence, had nothing), the gain is 1.306699 times the net gain.
- **Open vocabulary:** KN 0.637102 bpc (se 0.005198) → 0.632167 (se 0.005020).

**The learned kernel** (`test_learned_kernel`):
- **Checks:**
  - the pointer gradient against the complex step (10⁻¹⁰ relative) and central differences (10⁻⁶), on a 40-position
    training document at rank 2;
  - at A = B = 0 the pointer loss is Def 3's.
- **Training:**
  - the train pointer loss falls every epoch (3.001310 → 2.073484);
  - dev chooses epoch 11 of 12 (2.781427, against Def 3's 2.787046), with λ = 0.97.
- **Held out:**
  - 2.821058 (se 0.024667), learned − fixed −0.007139 (se 0.001080), 6.6 standard errors;
  - 0.630752 bpc;
  - the cache's own held-out pointer loss falls from 3.255830 to 2.383208.

### Proposition 7 (one head's rank bound)
For one bidirectional head, log A = QKᵀ/τ − λ𝟙ᵀ, with λ_i the row log-sum-exp, so rank log A ≤ d_k + 1. A positive
row-stochastic pattern P is realizable by one head iff some row shift of log P has rank ≤ d_k. The least
Frobenius distance from log P to {S + c𝟙ᵀ : rank S ≤ d_k} is the tail beyond d_k of the singular values of
log P − (row means)𝟙ᵀ.

*Proof.*
- **The bound.** Subadditivity of rank.
- **The iff.** The softmax is invariant to row shifts, and a rank-≤ d_k matrix τS factors as QKᵀ.
- **The distance.** For fixed S the best shift is the row means of log P − S, which leaves (log P − S)Jₙ. Since SJₙ
  ranges over the rank-≤ d_k matrices with rows orthogonal to 𝟙, Eckart–Young (formalML `svd`) applies to log P Jₙ. ∎

*Reading.*
- This is Topic 5's Thm 6 with keys in place of the vocabulary (Bhojanapalli et al. 2020).
- Hard patterns escape the bound in the τ → 0 limit: d_k = 2 realizes any permutation.
- Several heads give several kernels (#11).

**Asserted** (`test_log_attention_rank`):
- rank d_k + 1 for d_k ∈ {1, 2, 3, 5} at n = 12;
- a head's own pattern is realizable at d_k (gap below 10⁻¹⁰) and not at d_k − 1;
- a Dirichlet pattern only at d_k = 11;
- d_k = 2 realizes a random permutation with weights above 0.99.

### Proposition 8 (positive random features; Choromanski et al. 2021)
For w ~ N(0, I), φ_w(x) = e^{⟨w,x⟩−‖x‖²/2} gives E[φ_w(x)φ_w(y)] = e^{⟨x,y⟩} and
Var = e^{2⟨x,y⟩}(e^{‖x+y‖²} − 1). For e^{β⟨x̂,ŷ⟩} on unit vectors, m features have relative variance
(e^{β‖x̂+ŷ‖²} − 1)/m ≤ (e^{4β} − 1)/m. Positive features never give a nonpositive denominator, while trigonometric
features (Rahimi and Recht 2007) can.

*Proof.* E e^{⟨w,a⟩} = e^{‖a‖²/2}, at a = x + y and at a = 2(x + y). ∎

*Reading.* The variance is largest where the kernel is largest. A sharp lookup does not compress.

**Asserted** (`test_positive_random_features`):
- the mean and variance over 2·10⁵ features within 4 standard errors, on 5 random pairs;
- positivity;
- a trigonometric-feature instance with a negative denominator among 200 tries.

### Proposition 9 (causal linear attention is a recurrence; Katharopoulos et al. 2020)
With φ ≥ 0, ô_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t, with S_t = S_{t−1} + φ(k_t)ᵀv_t ∈ ℝ^{m×d_v} and z_t = z_{t−1} + φ(k_t)ᵀ. The
state has fixed size m(d_v + 1).

*Proof.* Associativity of the sums over j ≤ t. ∎

*Reading.* The lookup is compressed back into a recurrent state, Topic 5's design, and Prop 8 prices it.

**Asserted:**
- `test_linear_attention_recurrence`:
  - the recurrence equals the masked quadratic form to 10⁻¹² (inclusive and strict);
  - 20,000 positive features approach softmax attention within 0.05.
- `test_random_feature_cache`:
  - the random-feature cache, computed as the recurrence, equals the quadratic form;
  - its mean total-variation error against the exact weights falls with m at every β, and rises with β at every m:

| β | m = 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|
| ½ | 0.082228 | 0.047278 | 0.023037 | 0.012555 |
| 1 | 0.168573 | 0.095980 | 0.049460 | 0.028520 |
| **2** (dev) | 0.263782 | 0.220637 | 0.117096 | 0.096262 |
| 4 | 0.421869 | 0.394055 | 0.319481 | 0.265629 |
| 8 | 0.734222 | 0.588936 | 0.494874 | 0.433747 |

  - KN ⊕ random-feature cache at the dev (β, λ), held out: 2.832728, 2.828224, 2.829693, and 2.830394 for
    m = 16, 64, 256, 1,024, against the exact 2.828196. That is closer at 1,024 than at 16, but not monotone,
    because λ = 0.98 hides most of the cache's error.

## 4. The finance case study (§7): the filing as memory

EDGAR's three MD&A excerpts (830 tokens, 57.5904% OOV), each with its own earlier words as the cache, at the dev
setting (β = 2, λ = 0.98) and never re-tuned. Open-vocabulary bits per character (`test_finance_findings`):

| model | EDGAR bpc |
|---|---|
| KN (= Topic 5's KN row) | 5.660499 |
| KN with Topic 5's PPM-C spelling cache (re-run) | 4.767425 |
| **KN ⊕ kernel cache** | **4.498235** |
| KN ⊕ learned kernel | 4.479819 |
| Topic 5's dynamic RNN with PPM-C (quoted) | 3.6501 |
| lzma given the training text (re-run) | 2.907982 |

- **Per filing:** Apple 5.650394 → 4.153066, ExxonMobil 5.646507 → 4.575183, Microsoft 5.682117 → 4.763556.
- **Where the gain comes from:** 97.7369% from repeated OOV words, 2.4724% from repeated vocabulary words, and
  −0.2093% from first mentions.
- **Diagnostic only, never a result:** λ tuned on EDGAR itself would be 0.6, giving 4.352594. The in-domain λ
  under-trusts the cache on real text, which repeats itself more than the generator does.
- **Curated terms** (the two most repeated OOV types of ≥ 3 letters per filing; second mention, KN → cache bits):

| filing | term | count | KN | cache |
|---|---|---|---|---|
| Apple | *other* | 9 | 48.94 | 12.14 |
| Apple | *tariffs* | 8 | 30.64 | 9.55 |
| ExxonMobil | *plans* | 8 | 28.22 | 8.55 |
| ExxonMobil | *are* | 7 | 33.07 | 11.17 |
| Microsoft | *our* | 8 | 43.19 | 12.54 |
| Microsoft | *that* | 4 | 52.09 | 11.26 |

  First mentions cost the cache about 0.03 bits more than KN. Common English words are OOV here because the
  synthetic generator never writes them, so a cache that copies earns most where the tokenizer failed: Topic 5's
  spelling floor, lowered by copying, not by spelling better.
- **Asserted:**
  - the cache beats KN with PPM-C, which beats KN, and the learned kernel beats the cache;
  - lzma and Topic 5's best still win;
  - over 90% of the gain is on repeated OOV words;
  - the oracle λ is below the dev λ;
  - each curated second mention costs under half its KN bits;
  - the cache helps every filing.

## 5. `rigorFlag`

> Propositions 1 and 2 are exact for one head with given keys, but a trained transformer learns its keys, so its
> "design points" are fitted and the asymptotics of kernel regression do not describe it. Theorem 1 is exact only for
> equispaced keys and one frequency of position; the trained denoiser is a one-layer toy from one seed, and Collins
> et al.'s theorem concerns their setting, not this one. Theorem 2's saturation result is about Gaussian queries and
> keys at initialization. The long-lag task is built for content addressing — its bit symbols differ from its noise
> symbols — and Topic 5's trained recurrent accuracies are quoted, not re-run. The document cache adds information,
> not only a method: its gain comes from memory across sentences that Kneser–Ney and Topic 5's per-sentence RNN never
> had, so it does not show that attention beats recurrence. On the synthetic corpus, which repeats itself little
> within a document, the gain is 0.027 nats per token (0.005 bits per character); on three EDGAR filings it is 1.16
> bits per character, almost all from copying words the synthetic training text never contained, with λ tuned in
> domain, and lzma and Topic 5's adaptive RNN still code the filings shorter. As β → ∞ the cache pools words Topic 6's
> vectors cannot tell apart, not only identical words. Positive random features are least accurate exactly where the
> kernel is sharp, and every trained result is one seed.

## 6. Viz design intent — `AttentionKernelLab` (one component, two panels)

**Data.** All data comes from `viz_payload()` (55.1 kB now; `test_viz_payload` caps it at 60 kB), mirrored in
`src/components/viz/data/attention-kernel-lab.json`.
- Add `test_viz_mirror_matches_payload` in Phase B.
- Losses and gains are rounded to 3–4 decimals in the payload, and per-token bits to 2.
- Marginalia tokens only.

**Panel A, "Temperature is bandwidth"** (`circle`).
- **Data:** for each case (smooth-32, smooth-64, rough-32), on a 41-value c grid (0 and 40 log-spaced values from ¼
  to 400):
  - the exact risk, bias², and variance;
  - ρ₁…ρ₈;
  - c* and the learned c;
  - one sample (θ, f, y).
- **Views:**
  - *Smoother:* the keys on a circle, sized by their weight for a clicked query, and the unrolled strip of f, y, and
    the smoothed output, computed live from (θ, y) and c (exact arithmetic of Def 2; no displayed readout depends on
    it);
  - *Risk:* R(c) with its bias² and variance, with c* and the learned c marked.
- *Lessons:* the filter opens as c rises; bias and variance cross at c*; the trained layer finds it.

**Panel B, "A lookup instead of a state."** Two views behind a toggle.
- *Document* (`cache`):
  - **documents:** the held-out document (index 83, 198 positions) and the three filings;
  - **per token:** bits under KN, β = 0, β = 2, β = ∞, and the learned kernel;
  - **per document:** bpc for each variant (held-out: 0.742, 0.716, 0.713, 0.738, 0.713; Apple 5.650, 4.235, 4.153,
    4.642, 4.112);
  - **curated tokens'** top-8 attention weights at β = 2.
  - **Controls:** the β variant, and kernel (fixed / learned).
  - **Readouts:** the document's bpc, and the clicked token's bits.
  - The U-curve on the held-out split (`held_by_beta`, `dev_by_beta`) as an inset.
- *Reach* (`reach`): log₁₀ reach against lag for attention, Elman, and the two LSTMs, with 1/(L + 1) as a guide, and
  the trained accuracies.
- *Lessons:* attention reads across sentences, bandwidth matters on text, copying pays where the vocabulary fails, and
  a lookup's gradient is diluted, not decayed.

**Mobile.** Rows wrap, inputs get `minWidth: 0`, the token strip wraps, and long OOV words are truncated with an
ellipsis. `scrollWidth === 375` in every view, toggle, and selection.

## 7. The module (`attention_as_kernel_smoothing.py`, as built)

- **Imports:**
  - Topic 5 as `nlm`: `complex_step_grad`, `central_diff_grad`, `log_softmax`, `softmax_jacobian`, `lag_data`,
    `lag_loss_grad`, `init_lag_model`, `lag_models`, `long_lag_census`, `kn_logprobs`, `corpus`,
    `ppm_cache_spelled_nats`, `lzma_given_bpc`, `comparison_row`, the Adam and lag constants, `DEV_DOCS`, `DEV_SEED`;
  - Topic 6 as `ds`: `embedding`, `index`, `documents`;
  - Topic 4 as `pce`: `spell_nll`, `n_chars`, `ratio_estimate`, `sentence_nll`;
  - Topic 3 as `ng`, and `fc`.
- It never calls Topic 5's trained models or `ds.selected()`.
- **Section map:**
  - §1: `attention`, `kernel_smoother`, `gaussian_form`, `mixture_posterior`, `sample_mixture`, `causal_stack`,
    `bidirectional_stack`;
  - §2: `limit_bounds`, `circle_attention`, `circle_gains`, `circle_variance_factor`, `circle_risk`, `optimal_c`,
    `rate_prefactor`, `denoiser_*`, `train_denoiser(DenoiserConfig)`;
  - §3: `softmax_jacobian_bound`, `saturation_curve`;
  - §4: `one_hop_jacobian`, `one_hop_bound`, `lag_construction`, `lag_attention_*`, `recurrent_census`, `lag_sweep`;
  - §5: `documents`, `key_vectors`, `key_classes`, `doc_events` (the `Doc` record), `cache_weights`, `cache_probs`,
    `mixture_logprobs`, `dev_grid`, `tune_cache`, `cache_benchmark`, `pointer_loss[_grad]`,
    `train_learned_kernel(KernelConfig)`, `learned_benchmark`;
  - §6: `log_attention_rank`, `best_head_fit`, `positive_features`, `trig_features`, `rf_moments`,
    `linear_attention_quadratic`, `linear_attention_recurrent`, `random_feature_weights`, `random_feature_benchmark`;
  - §7: `finance_summary`; then `viz_payload`, `summary`, and 19 tests.
- **Timing:**
  - ≈ 17 s in all;
  - the three denoisers ≈ 4.3 s;
  - the lag sweep ≈ 3 s;
  - the learned kernel ≈ 1.4 s;
  - the random-feature grid ≈ 3 s;
  - everything on text ≈ 2 s.

## 8. Notebook cell map (`01_attention_as_kernel_smoothing.ipynb`, 29 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; the splits as documents | `test_documents_partition_corpus` |
| 2–5 | §1 Defs 1–2, Props 1–3; the tilt; a reversed prefix | `test_attention_is_nadaraya_watson`, `test_mixture_posterior`, `test_order_blindness` |
| 6–9 | §2 Prop 4, Thm 1, Cor 1; the trained denoiser | `test_temperature_limits`, `test_circle_fourier_multiplier`, `test_circle_rate`, `test_trained_denoiser` |
| 10–11 | §3 Thm 2 | `test_scaling_and_saturation` |
| 12–15 | §4 Props 5–6; the census, sweep, and training | `test_one_hop_jacobian`, `test_lag_attention` |
| 16–21 | §5 Def 3, Cor 2, the dev grid, the benchmark, Def 4 | `test_kernel_cache`, `test_learned_kernel` |
| 22–23 | §6 Props 7–9; the random-feature cache | `test_log_attention_rank`, `test_positive_random_features`, `test_linear_attention_recurrence`, `test_random_feature_cache` |
| 24–25 | §7 finance | `test_finance_findings` |
| 26–28 | lab payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Commands:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/attention-as-kernel-smoothing/attention_as_kernel_smoothing.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/attention-as-kernel-smoothing/01_attention_as_kernel_smoothing.ipynb
```

## 9. Cross-site links (all `status: "published"` on GitHub `main`, verified 2026-10-23; re-verify at the B3 gate)

| field | topic | relationship |
|---|---|---|
| `formalmlPrereqs` | `kernel-regression` | Softmax attention is its Nadaraya–Watson estimator with a Gaussian kernel and key-norm design weights; on the circle its bias–variance trade-off and AMISE rate become an exact closed form. |
| `formalmlConnections` | `gaussian-processes` | A Gaussian-process posterior mean is also a linear smoother of the responses, but it inverts the kernel matrix where Nadaraya–Watson and attention only normalize each row. |
| `formalmlConnections` | `message-passing` | Graph attention aggregates a node's neighbors with softmax weights, which is this page's kernel smoother restricted to a neighborhood of the graph. |
| `formalstatisticsPrereqs` | `kernel-density-estimation` | Attention's output is the conditional mean of a kernel density estimate whose components sit on the keys, so its weights are the mixture's posterior responsibilities. |
| `formalcalculusPrereqs` | `fourier-series` | On equispaced keys around a circle, attention is diagonalized by the Fourier basis, and its exact risk is computed by Parseval's identity from the Bessel gains. |
| `formalragPrereqs` | `hypersphere-vmf-geometry` | Attention between unit vectors is a von Mises–Fisher kernel; its Bessel normalizer and mean resultant length are the gains of the circle theorem. |
| `formalragConnections` | `retrieval-vs-long-context` | That page prices attention's quadratic cost in long contexts; this page factors the kernel with random features to make it linear, and measures what the factorization loses. |
| `formalragConnections` | `late-interaction-learned-sparse` | ColBERT's MaxSim keeps each query token's best-matching document token, the zero-temperature limit of the attention lookup analyzed here. |

**Not cited:**
- formalML `double-descent` (left for #13);
- formalML `local-regression` (no leave-one-out bandwidth here);
- formalML `svd` (Prop 7 uses Eckart–Young through Topic 5's citation; add it in Phase B if the page names it);
- formalRAG `cross-encoders-reranking` (its page does not treat attention);
- formalCalculus `jacobian` (Topic 5 carries it).

## 10. References (Chicago 17th, notes-and-bibliography; verified 2026-10-23 — arXiv titles and authors from the abstract pages, DOIs via Crossref, ACL Anthology bib)

1. Nadaraya, E. A. "On Estimating Regression." *Theory of Probability and Its Applications* 9, no. 1 (1964): 141–42.
   https://doi.org/10.1137/1109020. — paper; Def 1.
2. Watson, Geoffrey S. "Smooth Regression Analysis." *Sankhyā: The Indian Journal of Statistics, Series A* 26, no. 4
   (1964): 359–72. https://www.jstor.org/stable/25049340. — paper; Def 1.
   - The volume and pages were confirmed through the Bielefeld record and a *Statistical Science* reference list.
   - The JSTOR id is formalML `kernel-regression`'s. JSTOR's page rejects automated fetches, so check it by hand at B3.
3. Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural Machine Translation by Jointly Learning to Align and
   Translate." Paper presented at the 3rd International Conference on Learning Representations, San Diego, 2015.
   https://arxiv.org/abs/1409.0473. — paper; Def 2.
4. Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia
   Polosukhin. "Attention Is All You Need." In *Advances in Neural Information Processing Systems* 30 (2017).
   https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html. — paper; Def 2,
   Thm 2.
5. Tsai, Yao-Hung Hubert, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. "Transformer
   Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel." In *Proceedings of the
   2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on
   Natural Language Processing*, 4344–53. Hong Kong: ACL, 2019. https://aclanthology.org/D19-1443/. — paper; Prop 1.
6. Gabbur, Prasad, Manjot Bilkhu, and Javier Movellan. "Probabilistic Attention for Interactive Segmentation." In
   *Advances in Neural Information Processing Systems* 34 (2021). https://arxiv.org/abs/2106.15338. — paper; Prop 2.
7. Haviv, Adi, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. "Transformer Language Models without Positional
   Encodings Still Learn Positional Information." In *Findings of the Association for Computational Linguistics:
   EMNLP 2022*, 1382–90. Abu Dhabi: ACL, 2022. https://aclanthology.org/2022.findings-emnlp.99/. — paper; Prop 3 (iii).
8. Ramsauer, Hubert, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber,
   et al. "Hopfield Networks is All You Need." Paper presented at the 9th International Conference on Learning
   Representations, 2021. https://arxiv.org/abs/2008.02217. — paper; Prop 4.
9. Collins, Liam, Advait Parulekar, Aryan Mokhtari, Sujay Sanghavi, and Sanjay Shakkottai. "In-Context Learning with
   Transformers: Softmax Attention Adapts to Function Lipschitzness." In *Advances in Neural Information Processing
   Systems* 37 (2024). https://arxiv.org/abs/2402.11639. — paper; the trained denoiser.
10. Olver, F. W. J., et al., eds. *NIST Digital Library of Mathematical Functions*. §10.35, "Generating Function and
    Associated Series," and §10.40, "Asymptotic Expansions for Large Argument." https://dlmf.nist.gov/10.35. —
    documentation; Thm 1, Cor 1.
11. Bhojanapalli, Srinadh, Chulhee Yun, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. "Low-Rank Bottleneck
    in Multi-head Attention Models." In *Proceedings of the 37th International Conference on Machine Learning*, PMLR
    119 (2020). https://arxiv.org/abs/2002.07028. — paper; Prop 7.
12. Rahimi, Ali, and Benjamin Recht. "Random Features for Large-Scale Kernel Machines." In *Advances in Neural
    Information Processing Systems* 20 (2007).
    https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html. — paper;
    Prop 8.
13. Choromanski, Krzysztof, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter
    Hawkins, et al. "Rethinking Attention with Performers." Paper presented at the 9th International Conference on
    Learning Representations, 2021. https://arxiv.org/abs/2009.14794. — paper; Prop 8.
14. Katharopoulos, Angelos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. "Transformers are RNNs: Fast
    Autoregressive Transformers with Linear Attention." In *Proceedings of the 37th International Conference on
    Machine Learning*, PMLR 119 (2020). https://arxiv.org/abs/2006.16236. — paper; Prop 9.
15. Kuhn, Roland, and Renato De Mori. "A Cache-Based Natural Language Model for Speech Recognition." *IEEE
    Transactions on Pattern Analysis and Machine Intelligence* 12, no. 6 (1990): 570–83.
    https://doi.org/10.1109/34.56193. — paper; Def 3 at β = 0.
16. Grave, Edouard, Armand Joulin, and Nicolas Usunier. "Improving Neural Language Models with a Continuous Cache."
    Paper presented at the 5th International Conference on Learning Representations, Toulon, 2017.
    https://arxiv.org/abs/1612.04426. — paper; Def 3.
17. Merity, Stephen, Caiming Xiong, James Bradbury, and Richard Socher. "Pointer Sentinel Mixture Models." Paper
    presented at the 5th International Conference on Learning Representations, Toulon, 2017.
    https://arxiv.org/abs/1609.07843. — paper; Def 4.
18. Yang, Zhilin, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. "Breaking the Softmax Bottleneck: A
    High-Rank RNN Language Model." Paper presented at the 6th International Conference on Learning Representations,
    Vancouver, 2018. https://arxiv.org/abs/1711.03953. — paper; Prop 7's callback.
19. Church, Kenneth W., and William A. Gale. "Poisson Mixtures." *Natural Language Engineering* 1, no. 2 (1995):
    163–90. https://doi.org/10.1017/S1351324900000139. — paper; burstiness (§§5, 7).

Wand and Jones's *Kernel Smoothing* (https://doi.org/10.1201/b14876, verified) is not cited, because formalML
carries it.

## 11. Phase B notes

- **Numbering.** MDX at `src/content/topics/attention-as-kernel-smoothing.mdx`, in page order: Defs 1–4, Props 1–9,
  Thms 1–2, Cors 1–2. The trained denoiser is unnumbered prose after Cor 1. No citations in `TheoremBlock` titles.
- **References.** `references[].type`: 10 is `documentation`, the rest `paper`.
- **Graph.** Flip the node's `status` to `published`, and remove the title from `planned` in `src/data/curriculum.ts`.
  Frontmatter: `prerequisites: [neural-language-models]`, `connections: []`.
- **Lab data.** `src/components/viz/data/attention-kernel-lab.json` = `json.dumps(viz_payload(), separators=(',',
  ':'))`. Add `test_viz_mirror_matches_payload`. Trim the payload if the TSX needs less, and keep it under 60 kB.
- **Notation.**
  - Write 𝟙 as `\mathbf{1}`.
  - Write ⊗ for the outer product in Prop 5.
  - No bare `<`: use `\lt`.
  - Special tokens (`<s>`, `</s>`, `<unk>`) and the curated words go in backticks or italics, never in KaTeX.
  - Long number lists get one `$…$` per number.
- **Display math:** Prop 1 (ii); Thm 1's ρ_k and R(c); Cor 1's c*; Prop 5's Jacobian; Prop 8's moments; Prop 9's
  recurrence. Thm 1 and Prop 1 state their parts as bold labeled paragraphs, not list items (the Topic 8 lesson).
- **Mobile.** At 375 px the prose column is ≈ 343 px. The sweep, dev-grid, rate, denoiser, band, random-feature, and
  EDGAR tables need horizontal-scroll wrappers.
- **Page precision.** Quote from this brief's 6-decimal values, never from `summary()`'s 4-decimal print. The
  per-token bits in the lab are 2 decimals, and document bpc 3.
- **Overview hook:** the lag-600 accuracy and the 1/(L + 1) against 10⁻⁴⁶ comparison; EDGAR 5.6605 → 4.4982 bpc, with
  lzma's 2.9080 stated in the same sentence.
- **Gate.** Re-run the harness (≈ 17 s) and the notebook (≈ 21 s) at B3.
