# Brief — `attention-as-kernel-smoothing`

**Attention as Kernel Smoothing.** Roadmap #9. Prerequisite: `neural-language-models` (#5, PR #11), published. This
is the first `attention-transformers` topic and a prerequisite of `positional-encodings-geometry` (#10),
`transformer-block-residual-stream` (#11), and `aspect-based-sentiment` (#19).

**Status: Phase A scaffold** (2026-10-23). This draft holds the outline, statements with proof sketches, function
signatures, and the cell map, for Jonathan's reaction before any function body is written. Numbers marked *spike*
come from throwaway runs in the planning session. Each will be re-derived and asserted by the `.py`, or dropped.

**Scope decisions** (Jonathan, 2026-10-23):
- **In:** every optional result:
  - the circle theorem and the trained in-context denoiser;
  - positive random features, with causal linear attention as a recurrence;
  - a learned kernel (trained W_Q, W_K) for the document cache;
  - the one-head rank bound.
- **Checks:** internal only:
  - Topic 5's complex step and central differences for every hand-written gradient;
  - `scipy.special` (`iv`, `ive`, `softmax`, `logsumexp`);
  - numpy's FFT for circulant eigenvalues;
  - Monte Carlo for the probabilistic claims.
- **Protocol:**
  - two toys trained from scratch in NumPy: the circle denoiser and Topic 5's long-lag task;
  - on text, the non-parametric kernel cache over each document's history, mixed with Topic 3's KN;
  - β, λ, and the learned kernel's epochs are chosen on Topic 5's dev documents;
  - one seed, and no attention language model (that is #11's).
- **Finance:** each EDGAR filing's own history as the attention memory.
- **Difficulty:** `intermediate`.
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–8.

**Binding forward promise, and where it is kept.**
- **Topic 5, §"Where this goes next":** "The recurrent state compresses the whole prefix into 128 numbers, and
  Theorem 5 says how fast it forgets. Attention replaces the compression with a weighted lookup over every earlier
  position, and the attention topic reads that lookup as kernel smoothing."
  - The lookup read as kernel smoothing: Prop 1, Prop 2, Def 3.
  - How fast it forgets: Prop 5 (one hop, diluted by 1/t, not a product of Jacobians), and Prop 6 with the trained
    lag model against Topic 5's own census.
  - Compression, revisited: Prop 9 (factored kernels turn the lookup back into a fixed-size state) and Prop 8 (what
    that costs).

**Thesis.** Attention is a Nadaraya–Watson smoother whose kernel is learned.
- **The kernel** is a Gaussian in the query–key distance, tilted by the keys' norms (§1).
- **The temperature** is the bandwidth. On the circle the bias–variance trade-off is exact: a Fourier low-pass
  filter with Bessel gains (§2).
- **The 1/√d scale** holds the kernel's spread fixed as d grows. Without it the softmax saturates and its Jacobian
  vanishes (§3).
- **A lookup has a one-hop gradient path.** It is diluted by the number of competitors, and the scores can undo that
  dilution. A recurrent state has a product of Jacobians (§4).
- **Without positions** one attention layer sees its prefix as a bag (§1). On text, a kernel over a document's
  earlier tokens reaches across sentences that no model of Topics 3–5 could see (§5).

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Attention as Kernel Smoothing |
| `{{SUBTITLE}}` | *(draft)* Softmax attention is the Nadaraya–Watson estimator with an exponential kernel: the temperature is a bandwidth, the 1/√d scale keeps the kernel from collapsing into a lookup, and a weighted lookup over every earlier position reaches in one step what a recurrent state forgets. |
| `{{SLUG}}` | `attention-as-kernel-smoothing` (= graph node id) |
| `{{DOMAIN}}` | `attention-transformers` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[neural-language-models]` |
| `{{FORMALML_PREREQS}}` | *candidates* `kernel-regression`; connections `gaussian-processes`, `message-passing` |
| `{{FORMALRAG_PREREQS}}` | *candidate* `hypersphere-vmf-geometry`; connections `retrieval-vs-long-context`, `late-interaction-learned-sparse` |
| `{{STATS_CALC_PREREQS}}` | *candidates* Stats `kernel-density-estimation`; Calc `fourier-series` |
| `{{RIGOR_FLAG}}` | §5 below (draft) |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/neural-language-models/`, `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/perplexity-cross-entropy/`, `notebooks/finance-corpus/` |

- **Tags:** `attention`, `kernel-smoothing`, `nadaraya-watson`, `softmax`, `temperature`, `transformers`,
  `random-features`, `linear-attention`, `cache-language-model`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`:** none, since #10, #11, and #19 are still planned.
- **Graph:** the edge `neural-language-models → attention-as-kernel-smoothing` and the three outgoing edges already
  exist.

**Scope.** This topic is not on the design doc's §6 watch-list, but four sibling topics adjoin it:
- **formalML `kernel-regression`** owns Nadaraya–Watson asymptotics: interior bias O(h²) (its Thm 1), variance
  (Thm 2), AMISE bandwidth, Stone's minimax rate, and local-linear boundary correction. We cite them, and prove only
  the exact finite-n case on the circle (Thm 1), where they become a closed form.
- **formalStatistics `kernel-density-estimation`** owns the KDE. Prop 2 uses its validity (Thm 1) for the joint
  whose conditional mean attention computes.
- **formalRAG `retrieval-vs-long-context`** owns attention's Θ(n²) arithmetic, FlashAttention's memory caveat, and
  lost-in-the-middle. We cite the cost, and treat linear attention only as kernel factorization (§6), not as a
  long-context method.
- **formalRAG `hypersphere-vmf-geometry`** owns the vMF density, its Bessel normalizer (its Thm 2), and the mean
  resultant length (its Prop 2). Prop 1 (iii) names the vMF kernel. Thm 1's first gain ρ₁ is that page's mean
  resultant length on the circle, I₁(c)/I₀(c).
- **formalRAG `late-interaction-learned-sparse` and `cross-encoders-reranking`** own query–document interaction
  scoring. ColBERT's MaxSim is Prop 4 (i)'s τ → 0 limit, as a pointer only.
- **This topic owns:**
  - the attention–Nadaraya–Watson identity and its Gaussian decomposition;
  - the mixture-posterior reading;
  - order blindness of one attention layer;
  - the two temperature limits with rates;
  - the exact circle theorem and its rate;
  - the 1/√d saturation theorem;
  - the one-hop gradient;
  - the kernel cache over a document's history and its learned kernel;
  - one head's rank bound;
  - random-feature factorization and linear attention as a recurrence.
- **Out of scope apart from pointers:**
  - positional encodings, including RoPE (#10);
  - the block, the residual stream, and the MLP (#11);
  - masked language models (#12);
  - scaling (#13);
  - FlashAttention and long-context engineering (formalRAG);
  - attention as explanation (a #19 question).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | the gradient reaching the bit at lag 60: one attention layer ≈ 10⁻² against Elman's 2.45·10⁻¹⁰ (Topic 5); EDGAR, KN 5.66 → ≈ 4.4 bpc with attention over the filing's own words (*spike*) | — | cells 0–1 |
| 1 | A weighted lookup | five keys, one query, the weights as Gaussian bumps scaled by key length | Def 1 (kernel smoother); Def 2 (scaled dot-product attention); **Prop 1** (attention is Nadaraya–Watson); **Prop 2** (the mixture posterior); **Prop 3** (order blindness) | cells 2–5 |
| 2 | Temperature is bandwidth | the same query at τ → 0 (one key), τ = √d, and τ → ∞ (the mean) | **Prop 4** (two limits, with rates); **Thm 1** (attention on the circle is a Fourier multiplier); **Cor 1** (c* ∝ n^{2/5}); the trained denoiser | cells 6–9 |
| 3 | Why 1/√d | p_max against d, scaled and unscaled | **Thm 2** (variance d; the Jacobian bound; saturation) | cells 10–11 |
| 4 | A lookup instead of a state | Topic 5's lag task: Elman 2.45·10⁻¹⁰, LSTM (b_f = 3) 1.42·10⁻³, attention ≈ 10⁻² | **Prop 5** (one hop); **Prop 6** (content addressing solves the lag task); census, training, lag sweep | cells 12–14 |
| — | The lab | `<AttentionKernelLab client:visible />` | — | cells 24–26 |
| 5 | Attention over a document | a held-out filing: a company name, its second mention 150 tokens later, KN's and the cache's bits for it | Def 3 (kernel cache); **Cor 2** (unigram cache to exact-context cache); the bandwidth U-curve on text; Def 4 (learned kernel); the benchmark | cells 15–19 |
| 6 | Kernels that factor | one head's logits are a rank-d_k matrix | **Prop 7** (rank ≤ d_k + 1); **Prop 8** (positive random features); **Prop 9** (causal linear attention is a recurrence); the random-feature cache against m | cells 20–21 |
| 7 | Finance case study: the filing as memory | Apple's MD&A: the second mention of an OOV term copied for a few bits instead of spelled for dozens | Def 3 on EDGAR, open vocabulary; against KN, Topic 4's spelling floor, and lzma | cells 22–23 |
| 8 | Where this goes next | — | #10 (Prop 3, and §2's circle is a sinusoidal encoding of frequency one), #11 (heads as several smoothers, the residual stream), #12, #19 (attention to aspect words), formalRAG | — |

- **Lab placement.** One `<AttentionKernelLab client:visible />` after §4. Panel A needs §2. Panel B needs §§4–5,
  so it previews §5. The alternative is to place the lab after §5; see open point 3.
- **Back-links:**
  - Topic 5's Prop 2 (softmax Jacobian = covariance) from Thm 2 (ii);
  - Topic 5's Thm 5 and the long-lag task from Prop 5 and Prop 6;
  - Topic 5's Thm 6 (softmax bottleneck) from Prop 7;
  - Topic 5's Thm 1 (smoothing by geometry) from Cor 2's nearest-context case;
  - Topic 4's open-vocabulary code and Topic 5's Prop 6 (the spelling floor) from §7;
  - Topic 6's SVD vectors as the cache's keys.

## 2. Definitions

**Notation.** Rows are tokens, as in Topic 5.
- X ∈ ℝ^{n×p} has rows x_t.
- Q = XW_Q and K = XW_K are in ℝ^{n×d_k}, and V = XW_V is in ℝ^{n×d_v}. Their rows are q_t, k_j, v_j.
- Scores are s_tj = ⟨q_t, k_j⟩/τ, with τ = √d_k unless stated.
- 𝒥_t is the set of positions t may read: all j (bidirectional) or j ≤ t (causal).
- α_tj = e^{s_tj}/Σ_{i∈𝒥_t} e^{s_ti}, and o_t = Σ_j α_tj v_j.
- V_max = max_j ‖v_j‖.

**Definitions.**
- **Def 1 (kernel smoother; Nadaraya 1964, Watson 1964).** Take design points k_j, responses v_j, and a positive
  kernel κ. Then m̂(q) = Σ_j κ(q, k_j)v_j / Σ_j κ(q, k_j).
  - With κ(q, k) = K_h(q − k) this is formalML `kernel-regression`'s Def 2.
  - With design weights ω_j > 0, κ(q, k_j) = ω_jK_h(q − k_j).
- **Def 2 (scaled dot-product attention; Bahdanau, Cho, and Bengio 2015; Vaswani et al. 2017).**
  Attn(Q, K, V) = softmax(QKᵀ/τ + B)V, row-wise. The mask B is 0 on 𝒥_t and −∞ off it.
- **Def 3 (kernel cache).**
  - **Setting.** A document's tokens are concatenated across sentences, as Topic 3 tokenizes them, each sentence
    closed by `</s>`. Position t has target w_t and preceding token u_t, which is `<s>` at a sentence start.
  - **Keys.** ê(u) is the unit vector of u's row in Topic 6's SVD (SPPMI₅, L2, d = 100, p = 0) for training words.
    `<s>` gets its own coordinate. Each distinct OOV string gets its own coordinate, so OOV keys match only
    themselves.
  - **The cache.** p_β(w | t) = Σ_{i<t} α_ti 1[w_i = w], with α_ti ∝ exp(β⟨ê(u_t), ê(u_i)⟩) over the document's
    earlier positions.
  - **The mixture.** p(w | t) = λ p_KN(w | t) + (1 − λ) p_β(w | t), and p = p_KN at a document's first token.
  - **Open vocabulary.** p_KN(w) is p_KN(`<unk>`)S(w) for OOV w (Topic 4's speller S), while the cache's values are
    raw strings. So a repeated OOV word is copied, not re-spelled.
  - **Attribution.** This is Grave, Joulin, and Usunier's (2017) continuous cache, with fixed word vectors in place
    of RNN states. At β = 0 it is Kuhn and De Mori's (1990) unigram cache.
- **Def 4 (learned kernel).** The scores are β⟨ê(u_t), ê(u_i)⟩ + ⟨ê(u_t)A, ê(u_i)B⟩, with A, B ∈ ℝ^{102×r} learned
  on the training documents.
  - The objective is the pointer log-likelihood Σ_t log Σ_{i<t: w_i = w_t} α_ti, over positions whose target occurred
    earlier in the document (cf. Merity et al. 2017). It is KN-free, because in-sample KN probabilities are overfit.
  - A = B = 0 at the start, so training starts from Def 3.
  - The epoch count and λ are chosen on dev.

**Configurations.**
- **Splits:**
  - train = `generate_corpus(300, 0)`, 300 documents, median 231 tokens;
  - dev = Topic 5's `generate_corpus(40, 2)`;
  - held = `generate_corpus(100, 1)`;
  - EDGAR = 3 filings.

  Each split is regrouped into documents with `ng.split_sentences`, and the harness asserts that the regrouping
  concatenates to `nlm.corpus()[split]`.
- **Grids:**
  - β ∈ {0, ½, 1, 2, 4, 8, 16, ∞};
  - λ ∈ {0.50, 0.55, …, 0.95, 0.97, 0.99};
  - chosen jointly on dev by mean log-loss.
- **Toys:**
  - circle: n ∈ {32, 128}, σ = ½, two spectra (smooth: energy at k ≤ 2; rough: up to k = 6), Adam;
  - lag: Topic 5's `lag_data`, LAG = 60, 600 Adam steps, batch 64, lr 0.01 (its protocol).
- **Streams:** default_rng([SEED, k]) with k ≥ 900, apart from Topics 5–8.

## 3. Statements and proof sketches

### Proposition 1 (attention is a Nadaraya–Watson smoother)
- **(i)** o_t is Def 1's smoother with κ_τ(q, k) = e^{⟨q,k⟩/τ}, design points k_j, and responses v_j.
- **(ii)** κ_τ(q, k) = e^{‖q‖²/2τ} · e^{‖k‖²/2τ} · e^{−‖q−k‖²/2τ}. The query factor cancels in the ratio, so o_t is
  the Gaussian Nadaraya–Watson estimator with bandwidth h = √τ (h = d_k^{1/4} at the default scale) and design
  weights ω_j = e^{‖k_j‖²/2τ}.
- **(iii)** If the keys share one norm, it is the unweighted Gaussian estimator. If q and k lie on a sphere of radius
  r, κ_τ = e^{(r²/τ) cos∠(q,k)}, the von Mises–Fisher kernel with concentration r²/τ (formalRAG
  `hypersphere-vmf-geometry`, Def 1).
- **(iv)** o_t lies in the convex hull of {v_j : j ∈ 𝒥_t}, so ‖o_t‖ ≤ V_max.

*Proof sketch.* Polarization, ‖q − k‖² = ‖q‖² + ‖k‖² − 2⟨q, k⟩. Factors common to the numerator and denominator
cancel. (iv): the weights are positive and sum to 1. ∎

*Reading.*
- The scale τ is a squared bandwidth, so the default bandwidth grows like d_k^{1/4}.
- A long key is a design point of exponentially greater weight: attention's "design density" is tilted by ‖k‖².
- The kernel reading is due to Tsai et al. (2019). The Gaussian factorization with explicit design weights is the
  form this page uses throughout.

**Asserted** (`test_attention_is_nadaraya_watson`):
- on random instances, (i)–(ii) hold to 10⁻¹², and (iii) on unit spheres;
- (iv) holds;
- rescaling one key's norm moves weight onto it, by exactly the factor e^{Δ‖k‖²/2τ}.

### Proposition 2 (the mixture posterior)
Let J ~ π with π_j ∝ ω_j, Q | J ~ N(k_J, τI), and V = v_J. Then P(J = j | Q = q) = α_j(q) and E[V | Q = q] = o(q).
Equivalently, o is the conditional mean under the joint p̂(q, v) = Σ_j π_j φ_τ(q − k_j)δ_{v_j}(v): a kernel density
estimate in q, with point masses in v (formalStatistics `kernel-density-estimation`, Thm 1).

*Proof sketch.* Bayes' rule, then Prop 1 (ii): π_jφ_τ(q − k_j) ∝ e^{⟨q,k_j⟩/τ}. ∎

*Reading.* The weights are the E-step responsibilities of an isotropic Gaussian mixture whose means are the keys.
Attention is one soft assignment, followed by the mean of the assigned values (cf. Gabbur, Bilkhu, and Movellan
2021, to be verified).

**Asserted** (`test_mixture_posterior`):
- closed-form Bayes equals the softmax to 10⁻¹²;
- a Monte Carlo draw of the generative model (10⁵ samples, one stream) matches E[V | Q ∈ ball(q)] within 4 standard
  errors as the ball shrinks.

### Proposition 3 (order blindness)
- **(i)** Bidirectional attention is permutation-equivariant: Attn(PX) = P Attn(X) for every permutation matrix P,
  and so is any stack of such layers with position-wise maps.
- **(ii)** One causal layer's o_t depends on the prefix x₁, …, x_{t−1} only through its multiset. Two prefixes that
  rearrange the same tokens, followed by the same x_t, get the same output.
- **(iii)** (ii) fails for two causal layers. Earlier positions see different prefixes, so their outputs, and the
  multiset the second layer reads, differ.

*Proof sketch.* (i) The softmax and the sum over j commute with relabeling. (ii) The sum over j < t is a function of
the multiset. (iii) A two-token counterexample, asserted numerically. ∎

*Reading.*
- Without positions, one layer reads "the dog bit the man" and "the man bit the dog" as the same bag. This is the
  case for #10.
- (iii) is why causal transformers without positional encodings still learn some order (Haviv et al. 2022): the
  causal mask itself leaks position.

**Asserted** (`test_order_blindness`):
- (i) under random permutations to 10⁻¹², for depth 1 and depth 3;
- (ii) exactly, on a corpus prefix and its rearrangement;
- (iii) a two-layer counterexample with outputs differing by more than 10⁻³.

### Proposition 4 (the two temperature limits)
Fix scores s_j over 𝒥. Let S* = argmax s, Δ = max s − max_{j∉S*} s_j, R = max s − min s, and v̄_A be the mean of v
over A.
- **(i)** ‖o_τ − v̄_{S*}‖ ≤ 2V_max(|𝒥| − |S*|)e^{−Δ/τ}/|S*|.
- **(ii)** ‖o_τ − v̄_𝒥‖ ≤ V_max(e^{R/τ} − 1).

*Proof sketch.*
- (i) The off-argmax mass ε = Σ_{j∉S*} α_j ≤ (|𝒥| − |S*|)e^{−Δ/τ}/|S*|. Then o − v̄_{S*} = Σ_{j∉S*} α_j(v_j − v̄_{S*}).
- (ii) nα_j ∈ [e^{−R/τ}, e^{R/τ}], so |α_j − 1/n| ≤ (e^{R/τ} − 1)/n. ∎

*Reading.*
- (i) is retrieval: separated keys make attention an associative memory, exact up to e^{−Δ/τ}. This is the one-step
  convergence of Ramsauer et al.'s (2021) modern Hopfield networks. ColBERT's MaxSim (formalRAG) is the τ → 0 limit.
- (ii) is bag-of-values averaging.
- The bandwidth runs between a nearest-neighbor lookup and a global mean.

**Asserted** (`test_temperature_limits`): both bounds hold on random instances over τ ∈ [10⁻², 10²]. Each is tight
to within a factor of 3 at small or large τ on a two-cluster example.

### Theorem 1 (attention on the circle is a Fourier multiplier)
Let θ_j = θ₀ + 2πj/n, with u(θ) = (cos θ, sin θ) as each token's key and query features, and M = cI with c ≥ 0. So
s_ij = c cos(θ_i − θ_j). Let the values be y_j = f(θ_j) + ε_j, with ε i.i.d. of mean 0 and variance σ², and let A be
the n × n (bidirectional) attention matrix. Write F_k = (1/n)Σ_j f(θ_j)e^{−ikθ_j}.
- **(i)** A is circulant. The Fourier vectors (e^{ikθ_j})_j are its eigenvectors, with eigenvalues
  ρ_k(c) = Σ_{m ≡ k (mod n)} I_m(c) / Σ_{m ≡ 0 (mod n)} I_m(c).
  For c > 0, 0 < ρ_k < 1 = ρ₀ for every k ≢ 0. For c = 0, ρ_k = 0.
- **(ii)** The exact risk at the design points is
  R(c) = (1/n)Σ_i E(o_i − f(θ_i))² = Σ_k (1 − ρ_k)²|F_k|² + σ² Σ_{m≡0} I_m(2c) / (n(Σ_{m≡0} I_m(c))²).

*Proof sketch.*
1. The generating function e^{c cos φ} = Σ_{m∈ℤ} I_m(c)e^{imφ} (DLMF 10.35.1), summed over the n grid angles, keeps
   the terms with m ≡ k (mod n). The normalizer is the k = 0 case.
2. ρ_k is real and positive (I_m = I_{−m} > 0 for c > 0). |ρ_k| = |Σ_j a_j cos(kθ_j)| ≤ Σ_j a_j = 1, with equality
   iff cos(kθ_j) = 1 for all j, that is, k ≡ 0.
3. Bias: (A − I)f in the Fourier basis, then discrete Parseval. Variance: σ²tr(AᵀA)/n = σ²Σ_m a_m², with Σ_m a_m²
   from step 1 at 2c. ∎

*Reading.*
- Attention with one frequency of position is a low-pass filter. Mode k keeps the fraction ρ_k ≈ I_k(c)/I₀(c) of its
  amplitude, ρ₁ is formalRAG's mean resultant length on the circle, and raising c (lowering the temperature) opens
  the filter.
- formalML `kernel-regression`'s bias–variance trade-off, here as a closed form with no asymptotics.
- u(θ) is a sinusoidal positional encoding of frequency one, which is #10's subject.

**Asserted** (`test_circle_fourier_multiplier`):
- the formula against numpy's FFT eigenvalues (*spike:* 4.4·10⁻¹⁶ at n = 64, c = 7.3) on a grid of (n, c);
- the variance factor exactly (*spike:* 0.072749 both ways);
- the risk formula against Monte Carlo of noisy y (within 4 standard errors);
- 0 < ρ_k < 1.

### Corollary 1 (the bandwidth rate)
For a fixed nonconstant trigonometric polynomial f, the risk-minimizing c*(n) satisfies
c*(n) ~ (4Bn/(σ²√π))^{2/5}, with B = ¼Σ_k k⁴|F_k|². So the angular bandwidth 1/√c* is of order n^{−1/5}, and
R(c*) is of order n^{−4/5}: formalML `kernel-regression`'s AMISE rate in one dimension.

*Proof sketch.*
- I_k(c)/I₀(c) = 1 − k²/(2c) + O(c⁻²) and I₀(2c)/I₀(c)² = √(πc)(1 + O(c⁻¹)) (DLMF 10.40.1), and aliasing is
  super-exponentially small once c = o(n²). So R(c) = B/c² + σ²√(πc)/n + lower order.
- The bias term bounds R away from 0 for bounded c, so c* → ∞.
- Then calculus. ∎

**Asserted** (`test_circle_rate`), on the exact formula:
- c*(n)/n^{2/5} converges to the constant (within 1% at n = 10⁸);
- the log–log slope between 10⁶ and 10⁸ is within 0.01 of 0.4.

The spike shows the approach is slow (c* ≈ 1.6 → 3.5 for n = 32 → 1,024 at σ = ½), and the page says so.

**The trained denoiser** (unnumbered; Collins et al. 2024).
- **Model.** One attention layer. Tokens are (u(θ_j), y_j). W_Q, W_K ∈ ℝ^{2×2} read the position features only, so
  M = W_QW_Kᵀ/√2. The values are y_j, fixed.
- **Training.** Adam on fresh random tasks: f with random Fourier coefficients from the smooth or the rough spectrum,
  θ₀ uniform, σ = ½. The loss is the mean squared error to the clean f.
- **The population optimum** is M = c*I, with c* = argmin E_f R(c) computed from Thm 1 (ii) with E|F_k|² = s_k²/2.
  Rotation and reflection invariance of the task distribution rule out the other components.
- **Asserted** (`test_trained_denoiser`):
  - the learned M is within 5% of c*I in Frobenius norm, for both spectra and both n;
  - the rough spectrum learns a larger c than the smooth one (a sharper kernel for a less smooth function);
  - c grows from n = 32 to n = 128;
  - the hand-written gradient matches the complex step to 10⁻¹⁰ relative.

### Theorem 2 (why 1/√d)
- **(i)** If q and k are independent with i.i.d. coordinates of mean 0 and variance 1, then E⟨q, k⟩ = 0 and
  Var⟨q, k⟩ = d.
- **(ii)** The softmax Jacobian J = diag α − ααᵀ (Topic 5, Prop 2) satisfies ‖J‖₂ ≤ tr J = 1 − ‖α‖² ≤ 2(1 − α_max).
  So ‖∂o/∂s‖ → 0 as the softmax saturates.
- **(iii)** Let q ~ N(0, I_d) and keys k₁, …, k_n be i.i.d. N(0, I_d). Unscaled scores give 1 − α_max → 0 almost
  surely as d → ∞, at fixed n. Scaled by 1/√d, α converges in law to softmax(g) with g i.i.d. N(0, 1), so
  E(1 − α_max) stays bounded away from 0.

*Proof sketch.*
- (i) Expand the sum.
- (ii) J is PSD, so its spectral norm is at most its trace, and Σα_j(1 − α_j) = 1 − ‖α‖² ≤ (1 − α_max)(1 + α_max).
- (iii) Given q, the scores ⟨q, k_j⟩ are i.i.d. N(0, ‖q‖²), that is, ‖q‖g_j. Then 1 − α_max ≤ (n − 1)e^{−‖q‖G},
  with G the gap between the top two g's, which is positive a.s. ‖q‖/√d → 1 a.s. (the SLLN), which gives both
  limits. ∎

*Reading.*
- This is Vaswani et al.'s footnote made exact.
- Without the scale, the kernel's bandwidth shrinks like d^{−1/2} in score units, so attention collapses to a lookup
  at initialization and the gradient through the scores vanishes (Topic 5's Prop 2 covariance, now as a cost).

**Asserted** (`test_scaling_and_saturation`):
- (i) by Monte Carlo;
- (ii) on random α;
- (iii): mean 1 − α_max over d ∈ {4, …, 1,024} at n = 16, falling toward 0 unscaled and flat scaled;
- ‖J‖ tracks it.

### Proposition 5 (one hop)
For one causal layer and k < t, ∂o_t/∂x_k = α_tk[W_V + τ⁻¹W_Kq_tᵀ(v_k − o_t)], acting on row perturbations. Hence
‖∂o_t/∂x_k‖ ≤ α_tk(‖W_V‖ + τ⁻¹‖W_K‖‖q_t‖‖v_k − o_t‖). There is no product over the positions between k and t. With
equal scores, α_tk = 1/t.

*Proof sketch.* Differentiate o_t = Σ_j α_tj v_j: v_k depends on x_k through W_V, and s_tk through k_k = x_kW_K. The
softmax Jacobian's column k is α_tk(e_k − α_t). ∎

*Reading.*
- Topic 5's Thm 5 bounds the RNN's path by Π_{j=k+1}^{t} ‖diag(1 − h_j²)W_hᵀ‖.
- The lookup's path has length one, and its only loss is dilution by competing positions, which the scores
  themselves can undo.
- That is "replaces the compression with a weighted lookup", stated as a derivative.

**Asserted** (`test_one_hop_jacobian`): the closed form against the complex step to 10⁻¹⁰ relative on random
instances, and the bound at every (t, k).

### Proposition 6 (content addressing solves the lag task)
In Topic 5's task (x₀ ∈ {0, 1}, then L noise symbols from {2, 3}, then report x₀), take one causal attention layer
read at the last position, with one-hot inputs, W_Q = 0, W_V = (1, −1, 0, 0)ᵀ, and logits (wo, −wo). Then
o = (1 − 2x₀)/(L + 1), the prediction is correct for every L, and the cross-entropy log(1 + e^{−2w/(L+1)}) → 0 as
w → ∞. A key that scores the two bit symbols β above the noise raises the margin to e^β/(e^β + L).

*Proof sketch.* Direct computation. ∎

**Corpus-free findings** (`test_lag_attention`):
- **Census at initialization**, 12 inits, the median ‖∂ℓ/∂x₀‖ on Topic 5's batch protocol. Elman and the two LSTMs
  come from Topic 5's own `long_lag_census()`, re-run here (*spike:* 2.4499·10⁻¹⁰, 2.4200·10⁻⁷, 1.4246·10⁻³, in
  0.24 s). Attention is expected near 10⁻².
- **Training** by Topic 5's protocol (600 Adam steps, lr 0.01, batch 64). Expected accuracy 1.000 on 2,000 fresh
  sequences, against Topic 5's 0.485 (Elman), 0.531 (b_f = 1), and 1.000 (b_f = 3), which are quoted, not re-run.
- **Lag sweep** at initialization, L ∈ {10, 20, 40, 60, 120}: attention's reach falls like 1/(L + 1), and Elman's
  geometrically (Topic 5's `init_lag_model` and `lag_loss_grad`, pure).

### Definition 3, Corollary 2 (the cache's two limits), and the benchmark
**Cor 2.** By Prop 4:
- at β = 0, Def 3 is the unigram document cache;
- as β → ∞, the cache's mass goes to the earlier positions whose preceding token is closest to u_t in cosine. If u_t
  occurred before, these are exactly its occurrences (distinct training words have cosine < 1), giving the
  document's bigram counts #(u_t w)/#(u_t).

Otherwise the cache falls back to the nearest earlier contexts: Topic 5's smoothing by geometry, over a document.

**Benchmark** (`test_kernel_cache`; closed-vocabulary nats per token, as in Topic 5's table; standard errors
clustered by document, by Topic 4's ratio estimator with documents as units).
- *Spike* (L2 keys; dev picked β = 4, λ = 0.95):
  - held out, KN 2.8551 → 2.8317 nats per token;
  - on targets that occurred earlier in the document (41.9% of tokens): 1.798 → 1.671;
  - on first mentions: 3.617 → 3.667.
- *Spike*, the bandwidth U-curve on held-out text at the dev λ: β = 0 2.8399, β = 4 2.8317, β → ∞ 2.8559.
- *Spike*, open-vocabulary bpc on held out: KN 0.6371 → ≈ 0.632, a small gain. The synthetic generator draws content
  words from corpus-wide Pitman–Yor restaurants, so a document repeats itself little more than the corpus does.
- **To assert:**
  - the regrouped documents concatenate to Topic 5's splits;
  - the dev-chosen β is interior to the grid (a bandwidth, neither the unigram nor the exact cache);
  - the cache beats KN on held out by more than 3 clustered standard errors;
  - the gain lives on repeat targets;
  - the gain by distance to the previous mention, in bands ≤ 12 tokens, 13–50, and > 50, with the share across
    sentence boundaries (where Topic 5's RNN, reset per sentence, had nothing).
- **The learned kernel** (Def 4):
  - complex-step gradient check;
  - the pointer loss falls on train;
  - dev picks the epoch;
  - held-out result against Def 3, reported as it falls.

### Proposition 7 (one head's rank bound)
For one bidirectional head, log A = S − λ𝟙ᵀ, with S = QKᵀ/τ and λ_i the row log-sum-exp. So rank log A ≤ d_k + 1. A
positive row-stochastic pattern P is realizable by one head iff some row shift of log P has rank ≤ d_k.

*Proof sketch.* Subadditivity of rank. The softmax is invariant to row shifts, and a rank-≤ d_k matrix τS′ factors as
QKᵀ. ∎

*Reading.*
- This is Topic 5's Thm 6 (the softmax bottleneck) with keys in place of the vocabulary (Bhojanapalli et al. 2020).
- Hard patterns escape it in the τ → 0 limit (Prop 4), since d_k = 2 suffices for any permutation.
- Several heads give several kernels, which is #11's subject.

**Asserted** (`test_log_attention_rank`):
- rank = d_k + 1 on random heads;
- the best rank-d_k row-shifted fit to a random target pattern has positive error until d_k = n − 1, and 0 there;
- the permutation construction at d_k = 2.

### Proposition 8 (positive random features; Choromanski et al. 2021)
For w ~ N(0, I_p), φ_w(x) = exp(⟨w, x⟩ − ‖x‖²/2) satisfies E[φ_w(x)φ_w(y)] = e^{⟨x,y⟩} and
Var[φ_w(x)φ_w(y)] = e^{2⟨x,y⟩}(e^{‖x+y‖²} − 1). For the kernel e^{β⟨x̂,ŷ⟩} on unit vectors, m features have
relative variance (e^{β‖x̂+ŷ‖²} − 1)/m ≤ (e^{4β} − 1)/m. Positive features never give a nonpositive denominator,
while trigonometric features (Rahimi and Recht 2007) can.

*Proof sketch.* The Gaussian moment-generating function, E e^{⟨w,a⟩} = e^{‖a‖²/2}, at a = x + y and at a = 2(x + y). ∎

*Reading.* The variance is largest where the kernel is largest. A sharp kernel (large β), which is what makes a
lookup useful, is what does not compress.

**Asserted** (`test_positive_random_features`):
- unbiasedness and the variance formula by Monte Carlo (within 4 standard errors);
- positivity;
- a trigonometric-feature instance with a negative denominator.

### Proposition 9 (causal linear attention is a recurrence; Katharopoulos et al. 2020)
With a kernel φ(q)ᵀφ(k), φ ≥ 0, the causal smoother is ô_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t, where
S_t = S_{t−1} + φ(k_t)ᵀv_t ∈ ℝ^{m×d_v} and z_t = z_{t−1} + φ(k_t)ᵀ. Its state has fixed size m(d_v + 1), whatever t
is.

*Proof sketch.* Associativity: Σ_{j≤t}(φ(q_t)ᵀφ(k_j))v_j = φ(q_t)ᵀ(Σ_{j≤t}φ(k_j)ᵀv_j). ∎

*Reading.* The lookup has been compressed back into a recurrent state, Topic 5's design, and Prop 8 prices it.

**Asserted** (`test_linear_attention_recurrence`, `test_random_feature_cache`):
- the recurrence equals the masked quadratic form to 10⁻¹²;
- the random-feature cache on held out at the dev β, for m ∈ {16, 64, 256, 1,024}. Its loss approaches Def 3's as m
  grows, at the rate Prop 8 allows. If at β = 4 it stays far from Def 3 at m = 1,024, the page says so: that is the
  point.

## 4. The finance case study (§7): the filing as memory

EDGAR's three MD&A excerpts (Apple, Microsoft, ExxonMobil; 33 sentences, 830 tokens, 57.6% OOV). Each filing's
earlier tokens are its cache, with β and λ chosen on synthetic dev and not re-tuned.
- **Open-vocabulary bits per character** (*spike*):
  - KN 5.6605;
  - KN ⊕ cache at the dev-chosen setting ≈ 4.43 (out of domain the best λ is lower: ≈ 4.34 at λ = 0.6, a
    diagnostic only, never a result).
- **Baselines** re-run here:
  - Topic 5's PPM-C spelling cache on KN (`nlm.ppm_cache_spelled_nats`);
  - lzma given the training text, 2.9080 (`nlm.lzma_given_bpc`).
- **Quoted from Topic 5:** dynamic RNN + PPM-C 3.6501.
- **Readings:**
  - repeated OOV terms are copied, not spelled, and the share of the gain from OOV repeats is asserted;
  - the in-domain λ under-trusts the cache on real text, which is burstier than the generator;
  - lzma still wins, expected and to be asserted as it falls.
- **Curated tokens for the lab:** the first and second mentions of two OOV terms per filing, with their KN and cache
  bits.

## 5. `rigorFlag` (draft points)

- **What is a theorem.** Props 1–2 are exact identities for one head with given keys. A trained transformer learns
  its keys, so the "design points" are themselves fitted, and the n → ∞, h → 0 asymptotics of kernel regression do
  not describe a trained network.
- **Thm 1** is exact for equispaced keys and one frequency of position. Cor 1's rate is asymptotic and approached
  slowly. The trained denoiser is a one-layer toy trained from one seed, and Collins et al.'s theorem is for their
  setting, not this one.
- **Thm 2 (iii)** is about Gaussian queries and keys at initialization. Trained scores are not Gaussian.
- **The lag task is built for content addressing.** Its bit symbols differ from its noise symbols. With identical
  symbols, finding position 0 needs positional information (#10). Topic 5's trained LSTM and Elman numbers are
  quoted, not re-run.
- **The cache adds information, not just a method.** Its gain comes from document memory that Topic 3's KN and
  Topic 5's per-sentence RNN never had, so it is not "attention beats recurrence".
- **The synthetic corpus repeats itself little within a document**, so the in-domain gain is small. EDGAR is three
  filings, with λ tuned in domain, and lzma still codes it shorter.
- **Positive random features fail exactly where the kernel is sharp** (Prop 8).
- **Every trained result is one seed.**

## 6. Viz design intent — `AttentionKernelLab` (one component, two panels)

**Data.** All data comes from `viz_payload()`, mirrored in `src/components/viz/data/attention-kernel-lab.json`.
- `test_viz_payload` caps the file at 60 kB.
- Add `test_viz_mirror_matches_payload` in Phase B.
- Displayed numbers are rounded in the payload: 3 decimals for losses and gains, 2 for weights shown as bars. Derived
  sums are rounded from exact values (the Topic 8 lesson).
- Marginalia tokens only.

**Panel A, "Temperature is bandwidth"** (the circle).
- **Controls:**
  - c, on a payload grid of about 41 log-spaced values from 0 to past c*;
  - the spectrum (smooth / rough);
  - n (32 / 128).
- **View:**
  - the n keys on a circle, each a dot sized by its weight for a chosen query (click a dot);
  - an unrolled strip with f, the noisy y, and the smoothed output;
  - the gains ρ_k as bars.
- **A second view, "Risk":**
  - R(c) with its bias² and variance parts, from Thm 1 (ii);
  - c* marked, and the trained denoiser's learned c beside it;
  - the τ → 0 and τ → ∞ ends labeled "one key" and "the mean".
- *Lessons:* raising c opens a low-pass filter; bias and variance cross at c*; the trained layer finds c*.

**Panel B, "A lookup instead of a state."** Two views behind a toggle.
- *Document*:
  - a held-out document or an EDGAR filing, as a wrapped token strip;
  - clicking a token shows its attention weights over the earlier tokens (shaded), the cache's top 3, and KN's and
    the mixture's bits for the true token;
  - controls: β on the dev grid, and kernel (fixed / learned);
  - readouts: the document's bpc under KN and under the mixture;
  - curated tokens are pre-selected.
- *Reach*: log₁₀ of the gradient reaching the bit against the lag, for attention, Elman, and LSTM (b_f = 3), with
  1/(L + 1) as a guide.
- *Lessons:* attention reads across sentences, the bandwidth matters on text too, and a lookup's gradient is diluted,
  not decayed.

**Mobile.** Rows wrap, inputs get `minWidth: 0`, the token strip wraps, and long OOV words are truncated with an
ellipsis. `scrollWidth === 375` in every view, toggle, and selection.

## 7. Function signatures (`attention_as_kernel_smoothing.py`)

Bodies are written after Jonathan's reaction. The module imports:
- Topic 5 as `nlm`: `complex_step_grad`, `central_diff_grad`, `log_softmax`, `softmax`, `softmax_jacobian`,
  `lag_data`, `lag_loss_grad`, `init_lag_model`, `long_lag_census`, `kn_logprobs`, `corpus`, `spelled_nats`,
  `ppm_cache_spelled_nats`, `lzma_given_bpc`, `DEV_DOCS`, `DEV_SEED`;
- Topic 6 as `ds`: `embedding`, `index`, `documents`;
- Topic 4 as `pce`: `spell_nll`, `n_chars`, `ratio_estimate`;
- Topic 3 as `ng` (`split_sentences`, `vocabulary`), and `fc`.

It never calls Topic 5's trained models or `ds.selected()` (4.6 s), and it reimplements none of these functions.

```python
SEED = 20261023                      # streams default_rng([SEED, k]) with k ≥ 900
KEY_CFG = ("L2", "sppmi5", 100, 0.0) # Def 3's keys: Topic 6's SVD, declared, not selected on held-out data
BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
LAMBDAS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)

# §1 a weighted lookup
def attention(Q, K, V, tau: float | None = None, causal: bool = False) -> tuple[np.ndarray, np.ndarray]  # (O, A)
def kernel_smoother(q, K, V, kernel) -> np.ndarray                                    # Def 1
def gaussian_form(q, K, tau: float) -> dict[str, np.ndarray]                          # Prop 1 (ii): query factor, ω, Gaussian
def mixture_posterior(q, K, tau: float, omega: np.ndarray) -> np.ndarray              # Prop 2, by Bayes
def sample_mixture(rng, K, V, tau, omega, n: int) -> tuple[np.ndarray, np.ndarray]    # Prop 2's Monte Carlo
def causal_stack(X, layers: list[dict]) -> np.ndarray                                  # Prop 3 (iii)

# §2 temperature is bandwidth
def limit_bounds(s: np.ndarray, V: np.ndarray, tau: float) -> dict[str, float]         # Prop 4: bounds and actual gaps
def circle_attention(n: int, c: float, theta0: float = 0.0) -> np.ndarray              # Thm 1's A
def circle_gains(n: int, c: float) -> np.ndarray                                       # ρ_k by aliased Bessel sums (ive)
def circle_variance_factor(n: int, c: float) -> float                                  # Σ a_m²
def circle_risk(c: float, n: int, power: np.ndarray, sigma: float) -> dict[str, float] # Thm 1 (ii): bias², var, risk
def optimal_c(n: int, power: np.ndarray, sigma: float) -> float
def rate_prefactor(power: np.ndarray, sigma: float) -> float                           # Cor 1's (4B/σ²√π)^{2/5}
SPECTRA: dict[str, np.ndarray]                                                         # smooth / rough s_k²
def denoiser_tasks(rng, batch: int, n: int, spectrum: str, sigma: float) -> tuple[np.ndarray, ...]
def denoiser_loss_grad(P: dict, theta, y, f) -> tuple[float, dict]
def train_denoiser(cfg: "DenoiserConfig") -> "Trained"                                 # lru_cache, frozen cfg

# §3 why 1/√d
def softmax_jacobian_bound(alpha: np.ndarray) -> dict[str, float]                      # ‖J‖, tr J, 2(1 − α_max)
def saturation_curve(dims, n: int, trials: int, rng) -> dict[str, list[float]]         # Thm 2 (iii)

# §4 a lookup instead of a state
def one_hop_jacobian(X, P: dict, t: int, k: int) -> np.ndarray                         # Prop 5's closed form
def lag_construction(L: int, w: float, beta: float = 0.0) -> dict[str, float]          # Prop 6
def init_lag_attention(rng) -> dict
def lag_attention_loss_grad(P: dict, X, bit) -> tuple[float, dict, float, float]       # (loss, G, reach, acc)
def lag_attention_census(n_seeds: int = 12) -> dict[str, list[float]]
def lag_attention_training() -> dict[str, float]
def lag_sweep(lags=(10, 20, 40, 60, 120)) -> dict[str, list[float]]                   # attention vs Elman at init

# §5 attention over a document
def documents(split: str) -> tuple[tuple[tuple[str, ...], ...], ...]                   # per-doc sentences, asserted
def doc_events(split: str) -> list[dict]          # per doc: targets, preceding tokens, KN log p, spelling nats
def key_matrix(prev: list[str]) -> np.ndarray                                          # ê(u), Def 3
def cache_probs(doc: dict, beta: float, kernel: dict | None = None, open_vocab: bool = False) -> np.ndarray
def mixture_logprobs(split: str, beta: float, lam: float, kernel=None, open_vocab=False) -> np.ndarray
def tune_cache(kernel=None) -> tuple[float, float]                                     # (β, λ) on dev
def cache_benchmark() -> dict                     # held out: overall, repeat/first, distance bands, clustered se
def pointer_loss_grad(P: dict, docs: list[dict]) -> tuple[float, dict]                 # Def 4
def train_learned_kernel(cfg: "KernelConfig") -> "Trained"

# §6 kernels that factor
def log_attention_rank(Q, K, tau: float) -> int
def best_head_fit(P: np.ndarray, d_k: int) -> float                                    # Prop 7: row-shifted rank-d_k fit
def positive_features(X, W) -> np.ndarray                                              # Prop 8
def trig_features(X, W, b) -> np.ndarray
def rf_moments(x, y) -> tuple[float, float]                                            # closed-form mean and variance
def linear_attention_recurrent(phiQ, phiK, V) -> np.ndarray                            # Prop 9
def linear_attention_quadratic(phiQ, phiK, V) -> np.ndarray
def random_feature_cache(split: str, beta: float, lam: float, m: int, rng) -> np.ndarray

# summaries, lab, harness
def finance_summary() -> dict
def viz_payload() -> dict
def summary() -> dict[str, float]
# tests: test_attention_is_nadaraya_watson, test_mixture_posterior, test_order_blindness, test_temperature_limits,
#   test_circle_fourier_multiplier, test_circle_rate, test_trained_denoiser, test_scaling_and_saturation,
#   test_one_hop_jacobian, test_lag_attention, test_documents_partition_corpus, test_kernel_cache,
#   test_learned_kernel, test_log_attention_rank, test_positive_random_features,
#   test_linear_attention_recurrence, test_random_feature_cache, test_finance_findings, test_viz_payload,
#   test_viz_mirror_matches_payload (Phase B)
```

**Budget.**
- *Spike timings:*
  - importing Topic 5, 0.53 s;
  - KN on dev, held, and EDGAR, 0.17 s;
  - spelling, 0.07 s;
  - lzma, 0.13 s;
  - Topic 5's census, 0.24 s;
  - one Topic 6 SVD, ≈ 0.4 s;
  - the whole cache grid on dev and held, ≈ 1.3 s.
- To time first: the learned kernel's epoch, the random-feature cache at m = 1,024, and the denoiser's steps.
- The estimate is well under 60 s. If it nears 60 s, trim the learned-kernel epochs first, then the random-feature
  grid, then the denoiser steps, and never the Topic 5 checks.

## 8. Notebook cell map (`01_attention_as_kernel_smoothing.ipynb`, ≈ 27 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; the splits and documents | `test_documents_partition_corpus` |
| 2–5 | §1 Defs 1–2, Props 1–3 | `test_attention_is_nadaraya_watson`, `test_mixture_posterior`, `test_order_blindness` |
| 6–9 | §2 Prop 4, Thm 1, Cor 1; the trained denoiser | `test_temperature_limits`, `test_circle_fourier_multiplier`, `test_circle_rate`, `test_trained_denoiser` |
| 10–11 | §3 Thm 2 | `test_scaling_and_saturation` |
| 12–14 | §4 Props 5–6; census, training, sweep | `test_one_hop_jacobian`, `test_lag_attention` |
| 15–19 | §5 Def 3, Cor 2, benchmark, Def 4 | `test_kernel_cache`, `test_learned_kernel` |
| 20–21 | §6 Props 7–9; the random-feature cache | `test_log_attention_rank`, `test_positive_random_features`, `test_linear_attention_recurrence`, `test_random_feature_cache` |
| 22–23 | §7 finance | `test_finance_findings` |
| 24–26 | lab payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Commands:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/attention-as-kernel-smoothing/attention_as_kernel_smoothing.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/attention-as-kernel-smoothing/01_attention_as_kernel_smoothing.ipynb
```

## 9. Cross-site candidates (all `status: "published"` on GitHub `main`, checked 2026-10-23; re-verify at the B3 gate)

| field | topic | role |
|---|---|---|
| `formalmlPrereqs` | `kernel-regression` | Def 1 is its Def 2; Thm 1 and Cor 1 are its bias–variance trade-off and AMISE rate in closed form on the circle |
| `formalmlConnections` | `gaussian-processes` | the GP posterior mean is a linear smoother that inverts its kernel matrix (its Thm 1), where Nadaraya–Watson normalizes it |
| `formalmlConnections` | `message-passing` | graph attention (its Def 6) is Def 2 restricted to a neighborhood: kernel smoothing over a graph |
| `formalstatisticsPrereqs` | `kernel-density-estimation` | Prop 2's joint is a kernel density estimate, valid by its Thm 1, and attention is its conditional mean |
| `formalcalculusPrereqs` | `fourier-series` | Thm 1 diagonalizes attention on the circle in the Fourier basis and measures its risk by Parseval |
| `formalragPrereqs` | `hypersphere-vmf-geometry` | the vMF kernel (its Def 1), its Bessel normalizer (Thm 2), and the mean resultant length (Prop 2), which is Thm 1's first gain |
| `formalragConnections` | `retrieval-vs-long-context` | owns attention's Θ(n²) cost and positional bias; §6's factored kernels are the kernel-side view of that cost |
| `formalragConnections` | `late-interaction-learned-sparse` | MaxSim is Prop 4's zero-temperature limit of attention between query and document tokens |

**Not cited:**
- formalML `double-descent` (left for #13 `scaling-laws`);
- formalML `local-regression` (unless Def 3's bandwidth were chosen by leave-one-out);
- formalRAG `cross-encoders-reranking` (its page does not treat attention);
- formalCalculus `jacobian` (Topic 5 carries it).

## 10. References (Chicago 17th; every URL and title verified in Phase A)

1. Nadaraya, "On Estimating Regression," *Theory of Probability and Its Applications* 9 (1964).
   https://doi.org/10.1137/1109020 — Def 1.
2. Watson, "Smooth Regression Analysis," *Sankhyā A* 26 (1964). JSTOR — Def 1.
3. Bahdanau, Cho, and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," ICLR 2015.
   arXiv:1409.0473 — Def 2.
4. Vaswani et al., "Attention Is All You Need," NeurIPS 30 (2017). arXiv:1706.03762 — Def 2, Thm 2.
5. Tsai, Bai, Yamada, Morency, and Salakhutdinov, "Transformer Dissection: An Unified Understanding for
   Transformer's Attention via the Lens of Kernel," EMNLP-IJCNLP 2019. ACL Anthology D19-1443 — Prop 1.
6. Gabbur, Bilkhu, and Movellan, "Probabilistic Attention for Interactive Segmentation," NeurIPS 34 (2021) — Prop 2
   (to verify).
7. Haviv, Ram, Press, Izsak, and Levy, "Transformer Language Models without Positional Encodings Still Learn
   Positional Information," Findings of EMNLP 2022 — Prop 3 (iii).
8. Ramsauer et al., "Hopfield Networks is All You Need," ICLR 2021. arXiv:2008.02217 — Prop 4 (i).
9. Collins, Parulekar, Mokhtari, Sanghavi, and Shakkottai, "In-Context Learning with Transformers: Softmax Attention
   Adapts to Function Lipschitzness," NeurIPS 2024. arXiv:2402.11639 — the trained denoiser.
10. Olver et al., *NIST Digital Library of Mathematical Functions*, §§10.35, 10.40. https://dlmf.nist.gov/10.35 —
    Thm 1, Cor 1 (documentation).
11. Bhojanapalli, Yun, Rawat, Reddi, and Kumar, "Low-Rank Bottleneck in Multi-head Attention Models," ICML 2020.
    arXiv:2002.07028 — Prop 7.
12. Rahimi and Recht, "Random Features for Large-Scale Kernel Machines," NeurIPS 20 (2007) — Prop 8.
13. Choromanski et al., "Rethinking Attention with Performers," ICLR 2021. arXiv:2009.14794 — Prop 8.
14. Katharopoulos, Vyas, Pappas, and Fleuret, "Transformers are RNNs: Fast Autoregressive Transformers with Linear
    Attention," ICML 2020. arXiv:2006.16236 — Prop 9.
15. Kuhn and De Mori, "A Cache-Based Natural Language Model for Speech Recognition," *IEEE TPAMI* 12 (1990).
    https://doi.org/10.1109/34.56193 — Def 3 at β = 0.
16. Grave, Joulin, and Usunier, "Improving Neural Language Models with a Continuous Cache," ICLR 2017.
    arXiv:1612.04426 — Def 3.
17. Merity, Xiong, Bradbury, and Socher, "Pointer Sentinel Mixture Models," ICLR 2017. arXiv:1609.07843 — Def 4.
18. Yang, Dai, Salakhutdinov, and Cohen, "Breaking the Softmax Bottleneck," ICLR 2018. arXiv:1711.03953 — Prop 7's
    callback (Topic 5).
19. Church and Gale, "Poisson Mixtures," *Natural Language Engineering* 1 (1995) — burstiness, §5's reading (to
    verify).
20. Wand and Jones, *Kernel Smoothing* (1995) — background, through formalML (pointer, if cited).

## 11. Open points for Jonathan's reaction

1. **The in-domain cache gain is small** (*spike:* held-out bpc 0.6371 → ≈ 0.632), because the generator's documents
   are not bursty. The plan reports it as it falls, makes the bandwidth U-curve on text the in-domain lesson, and
   carries the large effect on EDGAR. The alternative, a burstier corpus variant via `fc._CorpusGen`, would fork the
   shared generator's behavior, so it is not recommended.
2. **Cache keys:** L2 SVD vectors of the preceding word, declared in advance. L1 and L8 gave the same held-out
   numbers to within 0.0002 bpc in the spike. The learned kernel can adapt any of them.
3. **Lab placement:** after §4 (Panel B previews §5), or after §5 (one embed that sees everything, later in the
   page). Recommendation: after §5.
4. **Prop 3 (iii)** corrects a tempting overstatement: a stack of causal layers without positions is not order-blind.
   That is honest and points at #10. Keep it, or drop it to a remark?
5. **Authorship.** Claude writes every body, as in Topics 5–8. If you would rather write one yourself, the natural
   candidates are `cache_probs` (the design of Def 3's open-vocabulary mixture) and `lag_construction`.
