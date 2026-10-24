# Brief — `positional-encodings-geometry`

**Positional Encodings: Sinusoids, Rotations, and RoPE.** Roadmap #10. Prerequisite: `attention-as-kernel-smoothing`
(#9, PR #17), published. This topic is a prerequisite of `transformer-block-residual-stream` (#11).

**Status: Phase A scaffold** (2026-10-24). This draft holds the outline, statements with proof sketches, function
signatures, and the cell map, for Jonathan's reaction before any function body is written. Numbers marked *spike*
come from throwaway runs in this session's planning scratchpad: a NumPy prototype of the tournament, with every
hand-written gradient checked against central differences to ≤ 3.4·10⁻¹⁰, three seeds, and about 1 s per training
run. Each number will be re-derived and asserted by the `.py`, or dropped.

**Scope decisions** (Jonathan, 2026-10-24):
- **Computational pillar:** a trained length-generalization tournament in NumPy.
  - Single causal heads, with hand-written gradients and Adam.
  - Schemes: NoPE, sinusoidal-absolute, learned-absolute, RoPE, and ALiBi.
  - Trained at L = 32 and tested to 8× that length.
  - The trained RoPE head is re-read under three context-extension rescalings.
- **Context extension:** a full section, flagged.
  - The wavelength census is a theorem.
  - Position interpolation (PI) is a theorem.
  - The NTK-aware endpoints are a theorem.
  - The interpolation bound is proved in its sharp form.
  - YaRN is described and labeled a heuristic.
- **Relative schemes:** ALiBi in full (a product kernel), with Shaw et al. and T5 in one definitional paragraph.
- **Difficulty:** `intermediate`, as for #9 (open point 5).
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–9.

**Binding forward promise, and where it is kept.**
- **Topic 9, §"Where this goes next":** "Proposition 3 says one layer of attention cannot tell *the dog bit the man*
  from *the man bit the dog*. The positional-encodings topic adds position, and Theorem 1's features
  $(\cos\theta, \sin\theta)$ are already a sinusoidal encoding of one frequency: several frequencies, and rotations of
  the queries and keys, are its subject."
  - *Adds position:* §§2–4.
  - *Several frequencies:* Def 2, Prop 2, Thm 1.
  - *Rotations of the queries and keys:* Def 4, Prop 3, Thm 4.
  - Topic 9's Prop 3 (iii), "the causal mask itself leaks order," is made exact by Prop 1.
- **A callback Topic 9 earned without naming it.** Its trained circle denoiser landed on M ≈ cI because its task
  distribution is rotation-invariant (brief §3 of #9). cI is the one-plane case of this topic's commutant (Thm 2):
  the denoiser learned a relative-only score because nothing else was rewarded.

**Thesis.** A positional encoding is a choice of kernel over offsets, and length generalization asks what that
kernel does past the training window.
- **Sinusoids** give a fixed stationary kernel. With geometric frequencies it is a *logarithmic ruler*: similarity
  falls by ln 2/ln b per doubling of distance (§2).
- **Added to the content, a stationary code does not make attention relative.** Only a d-dimensional commutant of
  score matrices sees offsets alone, and a learned W_Q W_Kᵀ is generically outside it (§2).
- **ALiBi** multiplies Topic 9's content kernel by an exponential kernel in time. It extrapolates in length because
  it cannot look far, and the same inequality gives both facts (§3).
- **RoPE** rotates queries and keys so that every score depends only on the offset. Up to the learned projections it
  is the only linear way to do so (§4).
- **The offset score is a trigonometric polynomial.** Planes whose wavelength exceeds the training length see unseen
  angles past it, and position interpolation keeps every angle inside the trained arc (§5).

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Positional Encodings: Sinusoids, Rotations, and RoPE |
| `{{SUBTITLE}}` | *(draft)* A positional encoding is a kernel over offsets: sinusoids are a fixed stationary kernel that rules distance logarithmically, RoPE rotates queries and keys so every learned score sees only the offset, and ALiBi multiplies in an exponential kernel that extrapolates in length because it cannot look far. |
| `{{SLUG}}` | `positional-encodings-geometry` (= graph node id) |
| `{{DOMAIN}}` | `attention-transformers` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[attention-as-kernel-smoothing]` |
| `{{FORMALML_PREREQS}}` | `gaussian-processes`; connections `kernel-regression`, `expander-graphs` |
| `{{FORMALRAG_PREREQS}}` | none; connections `retrieval-vs-long-context`, `optimized-product-quantization` |
| `{{STATS_CALC_PREREQS}}` | Stats `central-limit-theorem`; Calc `fourier-series`, `linear-algebra`, `linear-systems`; Calc connection `eigenvalues-eigenvectors` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/attention-as-kernel-smoothing/`, `notebooks/finance-corpus/` |

- **Tags:** `positional-encoding`, `sinusoidal`, `rope`, `rotations`, `alibi`, `relative-position`,
  `length-generalization`, `context-extension`, `transformers`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`:** none, since #11 is still planned.
- **Graph:** both edges (#9 → #10 → #11) already exist.

**Scope.** This topic is not on the design doc's §6 watch-list, but these sibling topics adjoin it:
- **formalRAG `retrieval-vs-long-context`** owns attention's Θ(n²) cost and lost-in-the-middle, which it models as an
  assumed U-shaped weight. This topic offers one partial mechanism, a recency kernel (Thm 3, Prop 4). It does not
  re-model the U-curve or the cost, and it does not claim RoPE explains the U-curve.
- **formalML `gaussian-processes`** owns positive-definite kernels, the Matérn family, stationarity, and Bochner (all
  in prose). We cite its Def 2, Def 3 (ν = ½), and Prop 2 (products).
- **formalCalculus `fourier-series`** owns the orthogonality of the trigonometric system (its Thm 1). Its §12 also
  says sinusoidal encodings "generalize to unseen lengths" and that the columns are orthogonal.
  - Thm 1 there covers integer harmonics over a full period, and 10000^{−2j/d} frequencies are not harmonics.
  - The tournament contradicts the extrapolation claim (sinusoidal-absolute falls from 0.998 to 0.19 at 2×, *spike*).
  - We cite Thm 1 as the contrast, and log the claim for the launch reciprocal PR (§5). The sibling is not touched.
- **formalCalculus `linear-systems`** owns the matrix exponential. R(k) = exp(kA) with A block-skew is its Ex 5.
- **formalStatistics `central-limit-theorem`** owns the characteristic function (its Def 1) and Lévy's continuity
  theorem (its Thm 3). Thm 1 (ii) is a weak limit of characteristic functions.
- **This topic owns:**
  - the causal counter (Prop 1);
  - the shift-as-rotation identity (Prop 2);
  - the sinusoidal kernel and its logarithmic ruler (Thm 1);
  - the commutant of relative-only scores (Thm 2);
  - ALiBi as a product kernel, with its locality bound and horizon (Thm 3);
  - RoPE's relativity and its characterization (Prop 3, Thm 4);
  - what long-term decay means (Prop 4);
  - the wavelength census and the rescalings (Thm 5);
  - the tournament and the finance case.
- **Out of scope:**
  - multi-head and multi-layer position, and the residual stream (#11);
  - BERT's learned table in practice (#12);
  - scaling (#13);
  - fine-tuning after rescaling (#20 and the literature; we evaluate zero-shot only);
  - FlashAttention and long-context engineering (formalRAG);
  - 2-D and graph positional encodings (a pointer to formalML `graph-laplacians` at most).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | *spike:* trained at 32 tokens and read at 256, sinusoidal-absolute 0.15, RoPE 0.41, RoPE + PI 0.95, ALiBi 0.97. On filings, when the last company was named > 32 tokens back, ALiBi 0.09 and RoPE + PI 0.98 | — | cells 0–1 |
| 1 | Order, and how a mask leaks it | Topic 9's Prop 3 by link; a BOS token's value through a causal layer reads 1, ½, ⅓, … | **Prop 1** | cells 2–3 |
| 2 | Position as a set of clocks | the hands of d/2 clocks, turning at rates 1, b^{−2/d}, …; two positions, one rotation apart | Def 1; Def 2; **Prop 2**; **Thm 1**; **Thm 2** | cells 4–9 |
| 3 | Relative position: biases and the ALiBi kernel | Shaw's clipped offsets and T5's log buckets; ALiBi's weights as a product of two kernels | Def 3; **Thm 3** | cells 10–11 |
| 4 | Rotations: RoPE | rotate the query by m and the key by n; only n − m survives the dot product | Def 4; **Prop 3**; **Thm 4**; **Prop 4** | cells 12–16 |
| 5 | Past the training window | the tournament table; the RoPE head's learned cue score, falling over [0, 32) and *rising again* (*spike:* 94.3 at Δ = 1, 3.8 at 31, 91.9 at 182) | the tournament; Def 5; **Thm 5** | cells 17–23 |
| — | The lab | `<PositionalEncodingLab client:visible />` | — | cells 27–28 |
| 6 | Finance case study: which company was named last? | a held-out MD&A where the parent was named 60 tokens back and the subsidiary since | Thm 3 (iii) as a forecast | cells 24–26 |
| 7 | Where this goes next | — | #11 (Prop 1's counter, written into the residual stream; heads with several slopes and frequencies), #12 (BERT's learned table), formalRAG `retrieval-vs-long-context` | — |

- **Lab placement:** after §5. Figure A needs §§2–4, and Figure B needs §5. Figure B's filing view previews §6.
- **Back-links to Topic 9:**
  - Prop 3 (order blindness) from §1;
  - Prop 1 (iii) (the vMF reading, equal norms) from Prop 3;
  - Def 1 (the kernel smoother) from Thm 3 (i);
  - Thm 1 (the circle, one frequency) from Def 2 and Thm 2;
  - Prop 8 (random Fourier features) from Thm 1 (i)'s spectral reading;
  - `attention()` and the lag-task training pattern from §5.

## 2. Definitions

**Notation.**
- d is even, and the planes are j = 0, …, d/2 − 1.
- The frequencies are ω_j = b^{−2j/d}, with base b = 10⁴ unless stated.
- R(φ) = [[cos φ, −sin φ], [sin φ, cos φ]] and J = R(π/2).
- R_ω(k) = ⊕_j R(kω_j) is the block-diagonal rotation by k steps.
- Rows are tokens. Queries and keys are q_t = h_t W_Q and k_j = h_j W_K, so the content score is h_t M h_jᵀ with
  M = W_Q W_Kᵀ, over τ = √d_k.
- Offsets are r = t − j ≥ 0 under the causal mask, and Δ = n − m in general.

**Definitions.**
- **Def 1 (absolute additive encoding).** h_m = x_m E + p(m), for a code p: {0, 1, …} → ℝ^d. Two kinds:
  - *learned*: a table P ∈ ℝ^{L×d}, undefined past L;
  - *fixed*: Def 2.
- **Def 2 (sinusoidal code; Vaswani et al. 2017).** p(m) = (cos mω_0, sin mω_0, …, cos mω_{d/2−1},
  sin mω_{d/2−1}).
  - Vaswani's interleaving puts sin before cos in each plane. That is a fixed reflection per plane, and it leaves every
    inner product unchanged.
  - Topic 9's u(θ) is the one-plane case.
- **Def 3 (relative schemes).**
  - *Shaw, Uszkoreit, and Vaswani (2018):* s_tj = q_t(k_j + a_{clip(j−t, K)})ᵀ/τ, with learned vectors for each
    clipped offset.
  - *T5 (Raffel et al. 2020):* a learned scalar b_{bucket(t−j)}. The buckets are exact for small offsets,
    log-spaced up to 128, and shared across layers.
  - *ALiBi (Press, Smith, and Lewis 2022):* s_tj = q_t k_jᵀ/τ − s_h (t − j) for j ≤ t, with fixed slopes
    s_h = 2^{−8h/H} for h = 1, …, H.
  - Our single head uses s = ½, ALiBi's first slope at H = 8.
- **Def 4 (RoPE; Su et al. 2024).** q̃_m = R_θ(m) q_m and k̃_n = R_θ(n) k_n, with θ_j = b^{−2j/d}.
  - The frequencies are Def 2's. RoPE is the sinusoidal code applied multiplicatively.
  - Complex form: each plane z_j ↦ z_j e^{imθ_j}.
- **Def 5 (context-extension maps)**, for target length L' = κL:
  - *PI (Chen et al. 2023):* m ↦ m/κ.
  - *NTK-aware (bloc97 2023):* b ↦ bκ^{d/(d−2)}.
  - *YaRN (Peng et al. 2024):* a per-plane ramp. With r_j = L/λ_j, the plane gets PI when r_j < α = 1, is unchanged
    when r_j > β = 32, and is linear in between. Logits are scaled by (0.1 ln κ + 1)².

**Configurations** (to be fixed by the harness; *spike* values shown).
- **The cue task.**
  - The alphabet has V = 8 symbols, and each token carries a cue bit with probability 0.1. At least one cue is
    forced, and a query token ends the sequence.
  - The target is the symbol on the most recent cue, so it needs content (*is this a cue?*) and order (*which is
    latest?*).
  - Inputs are one-hot: [symbol, cue, query] ∈ ℝ^{10}.
- **The head.**
  - E ∈ ℝ^{10×16}, and W_Q, W_K, W_V ∈ ℝ^{16×16}. The readout is W_o ∈ ℝ^{16×8} with a bias. Causal, read at the last
    position.
  - Training: 1,500 Adam steps on fresh batches of 64 at L = 32, lr 0.01.
  - Initialization follows Topic 9's lag head: N(0, ½) inputs and N(0, 1/√d) maps.
  - Learned-absolute adds P ∈ ℝ^{32×16}.
- **Test lengths:** 32, 64, 128, 256, with 2,000 fresh sequences each. The rescues are evaluated zero-shot at
  κ = L'/32.
- **Seeds:** three, on streams default_rng([SEED, k]) with k ≥ 1000, apart from Topics 5–9. SEED is Topic 9's
  `ak.SEED`.
- **Finance:**
  - *Documents.* Held-out documents `fc.generate_corpus(100, 1)`; *spike:* median 198 tokens, 2 companies and 8
    ORG mentions per document.
  - *Tokens.* Each entity mention collapses to one token. An ORG mention is a cue, labeled by its company's order of
    first appearance in the document (mod 8). Other tokens carry a hashed symbol (CRC-32 mod 8).
  - *Training (open point 1).* Training heads for this case on 32-token windows of `generate_corpus(300, 0)` is
    recommended.
  - EDGAR excerpts carry no entity spans, so they are excluded and the page says so.

## 3. Statements and proof sketches

### Proposition 1 (a causal layer counts but does not order)
- **(i)** Suppose all scores are equal (constant keys), x_0 is a BOS token with value v_0 = e, and v_j = 0 for j ≥ 1.
  Then one causal layer outputs o_t = e/(t + 1), so t = 1/⟨o_t, e⟩ − 1 is recovered exactly.
- **(ii)** With Topic 9's Prop 3 (ii), one causal layer reads its prefix as a multiset. The multiset's size is the
  position, but its order is invisible.

*Proof sketch.* (i) Uniform weights over t + 1 positions. (ii) The size of a multiset is a function of the multiset. ∎

*Reading.*
- A causal NoPE stack can write absolute position into its residual stream in layer 1 and use it in layer 2 (Haviv
  et al. 2022; Kazemnejad et al. 2023 give a construction).
- The tournament's single NoPE head cannot: it reads a bag of cues.

**Asserted:**
- `test_causal_counter`: (i) exactly, for t ≤ 4,096.
- `test_nope_reads_a_bag`: the trained NoPE head's accuracy is at most the Bayes accuracy of the best multiset reader
  (predict the most frequent cue symbol, splitting ties), within Monte Carlo error, at every length.
  - *Spike* accuracies: NoPE 0.548 / 0.359 / 0.284 / 0.205 at 32 / 64 / 128 / 256.

### Proposition 2 (shift is a rotation; Vaswani et al.'s "linear function", proved)
- **(i)** p(m + k) = R_ω(k) p(m) for all integers m and k.
- **(ii)** k ↦ R_ω(k) is a group homomorphism ℤ → SO(d): R_ω(a)R_ω(b) = R_ω(a + b) and R_ω(k)ᵀ = R_ω(−k).
  - R_ω(k) = exp(kA), with A = ⊕ ω_j J (formalCalculus `linear-systems`, Def 2, Thm 2 (c)–(d), Ex 5).
- **(iii)** ‖p(m)‖² = d/2. Every position lies on a winding line on the torus T^{d/2}, inside the sphere of radius
  √(d/2).
- **(iv)** The code is injective: p(m) = p(n) forces (m − n)ω_0 ∈ 2πℤ with ω_0 = 1, and π is irrational.

*Proof sketch.* The angle-addition formulas, plane by plane. ∎

*Reading.* Shifting every position by k turns each clock hand by k of its own steps, which is one fixed rotation.
The model can read "k later" without knowing "where".

**Asserted** (`test_shift_is_rotation`): (i)–(iii) to 10⁻¹² on random m, k at d = 16, 64, 512; and (iv) on
m, n < 10⁵.

### Theorem 1 (the stationary kernel, and the logarithmic ruler)
- **(i) Stationary and positive definite.**
  - ⟨p(m), p(n)⟩ = κ_d(n − m), with κ_d(Δ) = Σ_j cos(Δω_j).
  - The Gram matrix [κ_d(m − n)] is Toeplitz and positive semidefinite, because it is a Gram matrix.
  - (2/d)κ_d(Δ) = E cos(ΔΩ_d), with Ω_d uniform on the d atoms ±ω_j. This is the characteristic function of Ω_d
    (formalStatistics `central-limit-theorem`, Def 1).
  - Herglotz's theorem gives the converse (*cite* Brockwell and Davis): every positive-definite function on ℤ is the
    characteristic function of a measure on (−π, π]. A stationary position code is a choice of spectral measure; the
    sinusoids pick d atoms, and random Fourier features (Topic 9's Prop 8) sample them.
- **(ii) The logarithmic ruler.** As d → ∞ with b fixed, Ω_d converges in distribution to the symmetrized
  log-uniform law on [1/b, 1], whose density is 1/(2|ω| ln b). Hence for every fixed Δ ≠ 0:

  (2/d)κ_d(Δ) → I_b(Δ) = [Ci(|Δ|) − Ci(|Δ|/b)] / ln b,

  and I_b(Δ) = 1 − (γ + ln|Δ|)/ln b + [Ci(|Δ|) + O(Δ²/b²)]/ln b, so I_b(Δ) ≈ 1 − (γ + ln|Δ|)/ln b for 1 ≪ |Δ| ≪ b.

*Proof sketch.*
- (i) Angle subtraction, then Σ c_m c_n κ(m − n) = ‖Σ c_m p(m)‖² ≥ 0.
- (ii) u_j = 2j/d is a left Riemann grid on [0, 1), so Ω_d ⇒ ±b^{−U} with U uniform; cos(Δ·) is bounded and
  continuous. Substituting ω = b^{−u} gives ∫_{1/b}^{1} cos(Δω)/(ω ln b) dω. Then use Ci's integral form and
  asymptotics (DLMF 6.2.11, 6.12).
- ∎

*Reading.*
- Similarity is a ruler that is logarithmic in distance: each doubling of |Δ| costs ln 2/ln b ≈ 0.075 at b = 10⁴.
- Geometric frequency spacing gives every octave of distance equal resolution.
- Checked in planning: at d = 8,192, (2/d)κ_d(Δ) = 0.9739 / 0.7713 / 0.6345 / 0.4873 / 0.3347 / 0.1850 at
  Δ = 1 / 4 / 16 / 64 / 256 / 1,024. The limit gives 0.9740 / 0.7715 / 0.6348 / 0.4873 / 0.3349 / 0.1850.
- At d = 64 the finite-d deviation reaches 0.019 by Δ = 256. That is the Riemann error, which grows with |Δ|ln b/d.

**Asserted:**
- `test_stationary_kernel`: (i) to 10⁻¹², with PSD Gram matrices on random windows.
- `test_logarithmic_ruler`:
  - (ii) at d = 8,192 to 10⁻³ for |Δ| ≤ 1,024;
  - the gap shrinks monotonically along d = 64, 512, 8,192;
  - the asymptotic form to 0.01 on 16 ≤ |Δ| ≤ 1,024.

### Theorem 2 (which score matrices see only offsets)
Let the ω_j be distinct in (0, π). For M ∈ ℝ^{d×d}, the following are equivalent:
- **(a)** p(m)ᵀ M p(n) depends only on n − m, for all integers m, n;
- **(a′)** the same holds on one window of d + 1 consecutive positions;
- **(b)** M R_ω(1) = R_ω(1) M;
- **(c)** M = ⊕_j (a_j I + b_j J), block-diagonal with one rotation-scaling block per plane.

The matrices in (c) form a d-dimensional subspace of ℝ^{d×d}. Its orthogonal projection Π keeps each diagonal block's
(tr B/2) I + ((B₂₁ − B₁₂)/2) J and zeroes the off-diagonal blocks. On it, the score is the trigonometric polynomial
Σ_j a_j cos(Δω_j) + b_j sin(Δω_j).

*Proof sketch.*
- (a) ⇔ (b): g(m + 1, n + 1) = g(m, n) for all m, n ⟺ p(m)ᵀ(RᵀMR − M)p(n) = 0 for all m, n ⟺ RᵀMR = M, since the
  p(m) span ℝ^d.
- (a′): any d consecutive p(m) span ℝ^d. The matrix [e^{imλ}] over the d distinct nodes e^{±iω_j} is Vandermonde.
- (b) ⇔ (c): R has distinct eigenvalue pairs e^{±iω_j}, so its commutant is block-diagonal, and a real 2 × 2 block
  commuting with a rotation by an angle other than 0 or π is aI + bJ.
- ∎

*Reading.*
- **Adding a stationary code to the content does not make attention relative.**
  - Score: (x_t E + p(t)) M (x_j E + p(j))ᵀ.
  - Content–position cross terms enter (Ke, He, and Liu 2021), and the position–position term is relative only if M
    lies in the commutant.
  - A Gaussian M has relativity defect δ(M) = ‖M − Π(M)‖_F/‖M‖_F with E δ² = 1 − 1/d.
- **Topic 9's denoiser** learned M = cI, which is the commutant with one plane.
- **The exactness is fragile in practice.** Over a window short compared with the slow planes' wavelengths, the
  sinusoid matrix is badly conditioned (*spike:* numerical rank 14 of 16 over 64 positions). The constraint that
  (a′) puts on the slow planes is weak (§5).

**Asserted** (`test_commutant`):
- (a′) ⇒ (c) by solving the window's linear constraints;
- (c) ⇒ (a) to 10⁻¹²;
- E δ² = 1 − 1/d by Monte Carlo;
- the window's condition number grows as the slow planes' wavelengths outrun it.

`test_sinusoidal_head_defect`:
- the trained sinusoidal head's δ(M);
- its cross-term share;
- whether projecting M onto the commutant, and dropping the cross terms, changes its accuracy past L (reported as it
  falls).

### Theorem 3 (ALiBi is a product kernel: locality, and its horizon)
For one causal ALiBi head with slope s > 0, suppose the content scores satisfy |c_tj| = |q_t k_jᵀ/τ| ≤ B.
- **(i)** The weights are α_tj ∝ exp(c_tj) · e^{−s(t−j)}: Topic 9's Nadaraya–Watson smoother (its Def 1) with a
  product kernel, content × position.
  - In bidirectional form e^{−s|t−j|} is the Matérn-½ kernel (formalML `gaussian-processes` Def 3). It is positive
    definite, and so is the product (that topic's Prop 2).
- **(ii) Locality at every length.** The mass on offsets ≥ D is at most e^{2B} e^{−sD} / (1 − e^{−s}), for every t.
- **(iii) The horizon.** α at offset r is at most e^{2B − sr} times α at offset 0. A key farther back than
  2B/s gets less weight than the query's own position, whatever its content.

*Proof sketch.* The denominator is at least its offset-0 term, e^{−B}. Each numerator term is at most e^{B − sr}.
Sum the geometric tail. ∎

*Reading.*
- (ii) and (iii) are one inequality read two ways. ALiBi extrapolates in length *because* it cannot look far.
- Press, Smith, and Lewis's perplexity results are about local prediction. Chi et al. (2023) trace them to a bounded
  receptive field (*to verify*).
- §6 uses (iii) as a forecast: measure B on the trained head, and read off the gap at which retrieval must fail.

**Asserted:**
- `test_alibi_product_kernel`: (i)–(iii) on random instances.
- `test_alibi_horizon`: the trained head's B and 2B/s, against its accuracy by gap band on filings.

### Proposition 3 (RoPE is relative by construction, and isometric)
- **(i)** ⟨R_θ(m)q, R_θ(n)k⟩ = q R_θ(n − m) kᵀ (rows).
- **(ii)** ‖R_θ(m)q‖ = ‖q‖. In Topic 9's Prop 1 factorization, the design weights e^{‖k‖²/2τ} carry no position:
  RoPE moves only angles. This is the vMF reading of Topic 9's Prop 1 (iii).
- **(iii)** For fixed q and k, the offset score s(Δ) = q R_θ(Δ) kᵀ = Σ_j ρ_j cos(Δθ_j + φ_j) is a trigonometric
  polynomial. Here ρ_j = ‖q_j‖‖k_j‖ (plane norms), and φ_j is the angle between the two plane components.

*Proof sketch.* Prop 2 (ii), then u(α)ᵀ R(φ) u(β) in polar form. ∎

**Asserted** (`test_rope_relative_isometric`): (i)–(iii) to 10⁻¹² on random instances. Also, `ak.attention` on
rotated Q and K agrees with the explicit R(n − m) form.

### Theorem 4 (RoPE is the general linear relative encoding)
Let L_m ∈ ℝ^{d×d}, for m ∈ ℤ, with L_0 = I. The following are equivalent:
- **(a)** ⟨L_m q, L_n k⟩ depends only on (q, k, n − m), for all q, k, m, n;
- **(b)** L_m = U^m for some orthogonal U;
- **(c)** L_m = Q (⊕_i R(mθ_i) ⊕ I_p ⊕ (−1)^m I_r) Qᵀ, for some orthogonal Q, angles θ_i ∈ (0, π), and p, r ≥ 0.

So, up to an orthogonal change of basis (which W_Q and W_K absorb), the linear relative encodings are RoPE:
- with free frequencies;
- plus unrotated planes, which is partial rotary;
- plus Nyquist planes at θ = π.

*Proof sketch.*
- (a) ⇒ (b): (a) says L_mᵀL_n = G(n − m).
  - m = n gives L_mᵀL_m = G(0) = I, so each L_m is orthogonal.
  - G(k) = L_0ᵀL_k = L_k.
  - L_mᵀL_{m+k} = L_k gives L_{m+k} = L_m L_k, so L_m = L_1^m.
- (b) ⇒ (c): the real normal form of an orthogonal matrix (*cite* Horn and Johnson 2013, §2.5).
- (c) ⇒ (a): Prop 3 (i).
- ∎

*Reading.*
- RoPE fixes U's spectrum to Def 2's geometric frequencies.
- The theorem says no linear multiplicative scheme can do something RoPE cannot, apart from choosing the
  frequencies.
- Su et al. derive the two-dimensional case under smoothness assumptions. (b) needs none.

**Asserted** (`test_general_relative_encoding`):
- (b) ⇒ (a) for random orthogonal U;
- the real Schur form (scipy) reconstructs U in shape (c);
- a non-orthogonal family (L_m = A^m with A not orthogonal) violates (a).

### Proposition 4 (what "long-term decay" means)
- **(i) Su et al.'s envelope.** In complex form, with h_j = q_j k̄_j and S_j = Σ_{l<j} e^{iΔθ_l}:
  |Σ_j h_j e^{iΔθ_j}| ≤ (max_j |h_{j+1} − h_j|) Σ_{j=1}^{d/2} |S_j|, with h_{d/2} = 0.
  - Proved in full by summation by parts.
  - The envelope (2/d)Σ|S_j| is observed to fall with Δ on average, and not monotonically. That part is a
    measurement, not a theorem.
- **(ii) Matched pairs.** If q = k and the plane energies have E‖q_j‖² = σ² for each j, then
  E[q R_θ(Δ) qᵀ] = σ² κ_d(Δ). This is Thm 1's ruler, and it decays logarithmically.
- **(iii) Unrelated pairs.** If q and k are independent with mean zero, then E[q R_θ(Δ) kᵀ] = 0 for every Δ, and the
  variance does not depend on Δ.

*Proof sketch.* (i) Abel summation. (ii) u R(φ) uᵀ = ‖u‖² cos φ. (iii) Linearity and rotation invariance of the
second moment. ∎

*Reading.*
- "Long-term decay" is a property of an envelope and of matched query–key pairs.
- A trained head decays only if it learns to: the RoPE head's s(Δ) falls over [0, 32) because the cue task rewards
  recency, and it rebounds past 32 (§5).

**Asserted** (`test_long_term_decay`):
- (i) on random instances;
- the envelope's octave means decrease;
- (ii) and (iii) by Monte Carlo within 4 standard errors.

### The tournament (§5) and Theorem 5 (wavelength census and rescalings)
*Spike* table, from seed 1 (seeds 2 and 3 agree in pattern):

| scheme | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| NoPE | 0.548 | 0.359 | 0.284 | 0.205 |
| sinusoidal-absolute | 0.998 | 0.193 | 0.152 | 0.150 |
| learned-absolute | 0.988 | n/a | n/a | n/a |
| RoPE | 1.000 | 0.915 | 0.785 | 0.412 |
| RoPE + PI | — | 1.000 | 0.991 | 0.952 |
| RoPE + NTK | — | 0.957 | 0.952 | 0.965 |
| ALiBi | 0.997 | 0.977 | 0.977 | 0.969 |

The RoPE head's cue score s(Δ) (Prop 3 (iii), symbol-averaged):
- At Δ = 1, 5, 10, 20, 31: 94.3, 83.0, 59.6, 27.7, 3.8.
- Then it rises again: 15.1 at 40, 23.7 at 63, −21.8 at 80, 51.4 at 200.
- Its maximum on [32, 256) is 91.9 at Δ = 182.

**Theorem 5.**
- **(i) Census.** Over training offsets 0, …, L − 1, plane j turns through a whole circle iff (L − 1)θ_j ≥ 2π.
  Approximately, that is λ_j = 2π/θ_j ≤ L.
  - The number of complete planes is n_c(L) = #{j : b^{2j/d} ≤ (L − 1)/2π}.
  - The other planes see only the arc [0, (L − 1)θ_j]. At larger offsets they produce angles never seen in training.
  - At d = 16, L = 32 only 2 of 8 planes complete a turn (λ = 6.3, 19.9; then 62.8, 198.7, …).
- **(ii) PI** maps test offsets [0, κL − 1] into [0, L), so every angle lies inside the trained arc. Adjacent tokens
  then differ by θ_j/κ, a κ-fold loss of resolution in every plane.
- **(iii) NTK-aware.** θ′_j = θ_j κ^{−2j/(d−2)}: plane 0 is unchanged, and the slowest plane gets exactly PI's 1/κ,
  with the planes in between geometric. Checked: the ratios are 1 and 1/κ.
- **(iv) Interpolation versus extrapolation** (after Chen et al. 2023). For s(Δ) = Re Σ_j h_j e^{iΔθ_j}, linear
  interpolation between grid points Δ₁ < Δ₂ errs by at most ((Δ − Δ₁)(Δ₂ − Δ)/2) Σ_j |h_j| θ_j².
  - So the sharp constant is Σ_j θ_j² = (1 − b^{−2})/(1 − b^{−4/d}) (for planes j < d/2).
  - Outside the trained range no such bound exists: |s| can reach Σ_j |h_j|.
  - **To verify:** Chen et al.'s stated constant. If it is d/(4 ln b), that is the large-d approximation of ours, and it
    undercounts by 2.56× at d = 16 and by 1.04× at d = 512 (checked numerically in planning). If so, it goes in the
    rigorFlag.

*Proof sketch.* (i) An arc is complete iff its length reaches 2π. (ii) Arithmetic. (iii) Substitute the new base.
(iv) The interpolation remainder f(Δ) − ℓ(Δ) = f″(ξ)(Δ − Δ₁)(Δ − Δ₂)/2, with |f″| ≤ Σ|h_j|θ_j². ∎

**Asserted:**
- `test_wavelength_census`: (i)–(iii).
- `test_interpolation_bound`: (iv) on random instances, with the bound's tightness reported.
- `test_tournament`:
  - every scheme except NoPE is ≥ 0.95 at 32;
  - sinusoidal-absolute is < 0.3 at 64 and beyond;
  - ALiBi is ≥ 0.95 through 256;
  - RoPE falls monotonically and is < 0.6 at 256;
  - each rescue lifts RoPE at 256.
  - Bounds are set from the three seeds' minimum and maximum, as reported.
- `test_rope_rebound`: the learned s(Δ) has a maximum on [L, 256) above its value at the median training gap. The
  offset where s first rises again lies in a plane-wavelength band that (i) predicts.
- `test_gradients`: every scheme's hand-written gradient against central differences, ≤ 10⁻⁷.

## 4. The finance case study (§6): which company was named last?
- **Setting.**
  - An MD&A names a parent and its subsidiaries, and then refers back to "the company".
  - The question, asked at every position of a held-out document: which company was named most recently?
  - Answering it is the cue task on real document geometry, where mentions are bursty and the gaps are long. *Spike:*
    3,123 of the queries in 40 held-out documents have their answer more than 32 tokens back.
- ***Spike*** (seed 1, heads trained on the synthetic cue task; open point 1 would retrain on filing windows):

  | scheme | all queries | gap > 32 |
  |---|---|---|
  | NoPE | 0.744 | 0.721 |
  | sinusoidal-absolute | 0.667 | 0.579 |
  | RoPE | 0.790 | 0.619 |
  | RoPE + PI | 0.984 | — |
  | RoPE + NTK | 0.831 | — |
  | ALiBi | 0.615 | **0.085** |

  - Seeds 2 and 3 give ALiBi 0.093 and 0.301 for gap > 32, and RoPE + PI 0.984 and 0.974.
- **Lesson.** The scheme that won the length tournament loses the filing.
  - Thm 3 (iii) forecasts the gap at which ALiBi must fail.
  - NoPE does well here only because a document names about two companies: the bag of cues usually has a majority.
- **Data rules.** The data is the synthetic corpus only: no transcripts, no LM dictionary. EDGAR is excluded (no
  entity spans), and the page says so.

## 5. `rigorFlag` (draft points)
1. **Long-term decay** (Prop 4) is a property of an envelope and of matched pairs. RoPE does not make unrelated keys
   fade.
2. **The commutant theorem** (Thm 2) is exact, but its window version is badly conditioned when the window is shorter
   than the slow planes' wavelengths.
3. **Cited, not proved:**
   - Herglotz's theorem (the converse in Thm 1 (i));
   - the real normal form of orthogonal matrices (Thm 4);
   - Ci's asymptotics (DLMF).
4. **NTK-aware scaling and YaRN are empirical heuristics.**
   - YaRN's (0.1 ln κ + 1) temperature is a fit.
   - We evaluate every rescue zero-shot. Published results fine-tune after rescaling.
5. **The tournament is toy-scale:** one head, d = 16, 8 symbols, three seeds. It demonstrates mechanisms and
   measures nothing about language models at scale.
6. **The finance case** uses synthetic documents with hashed distractor symbols. The labels are first-appearance
   order mod 8.
7. **Folklore corrected:**
   - Sinusoidal encodings do not by themselves extrapolate (§5), and the PE columns are not orthogonal: Thm 1 (i) puts
     the Gram matrix at κ_d(m − n), not δ_mn.
   - This contradicts formalCalculus `fourier-series` §12. It is logged for the launch reciprocal PR, and the sibling
     is not touched here.
8. **Chen et al.'s interpolation constant** is a large-d approximation, if it is d/(4 ln b) (to verify; Thm 5 (iv)).

## 6. Viz design intent — `PositionalEncodingLab` (one component, two figures)

**Data.**
- All data comes from `viz_payload()`, mirrored in `src/components/viz/data/positional-encoding-lab.json`.
- `test_viz_payload` caps the file at 60 kB, and `test_viz_mirror_matches_payload` pins it.
- Displayed numbers are rounded in the payload: 3 decimals for accuracies and kernel values. Derived numbers are
  rounded from exact values.
- Marginalia tokens only. Wrapper: `<div data-lab="positional">`.

**Figure A, "Clocks and kernels."**
- *Clocks:* d/2 dials, one per plane, each a hand at angle mω_j.
  - Controls: m and n sliders; base (10², 10⁴, 10⁶); d (16 / 64).
  - The difference hand R_ω(n − m) is overlaid.
  - Readouts: ⟨p(m), p(n)⟩ and κ_d(n − m).
  - *Lesson:* shifting both positions turns every hand together, and the inner product does not move.
- *Kernel:* (2/d)κ_d(Δ) on a log-Δ axis, with the Ci limit and the ruler 1 − (γ + ln Δ)/ln b overlaid.
  - Toggle: d = 16 / 64 / 512.
  - A small Toeplitz heatmap of the Gram matrix sits beside it.
- *Rotation:* one RoPE plane, with q and k as arrows rotated by mθ and nθ.
  - Controls: m and n sliders, and a "lock the offset" toggle that slides m and n together.
  - Readouts: the RoPE score (constant under the lock), and the additive score under the trained sinusoidal head's M
    (it drifts under the lock: the relativity defect).

**Figure B, "Past the training window."**
- *Tournament:* accuracy against test length (32 → 256, log₂ axis), one line per scheme plus the three rescues.
  - Learned-absolute ends at 32 with a marker.
  - Clicking a scheme shows its learned offset score s(Δ) over [0, 256), with [0, 32) shaded as "seen in training".
    The RoPE rebound is visible.
- *Census:* one bar per plane, showing λ_j against an L slider.
  - Bars are shaded complete or partial.
  - Toggle: none / PI / NTK / YaRN, showing each plane's frequency scale factor.
- *Filings:* a held-out document as a token strip.
  - ORG mentions are chips colored by company.
  - Clicking a position shows each scheme's answer and the true one.
  - Each scheme's accuracy is shown by gap band, with ALiBi's 2B/s horizon marked.

*Lessons:*
- the encoding is a prior over offsets;
- RoPE's prior is a trigonometric polynomial that misbehaves where training never looked;
- ALiBi's prior cannot look far;
- PI keeps the angles in range.

**Mobile.** Rows wrap, dials reflow into a grid, inputs get `minWidth: 0`, and the token strip wraps.
`scrollWidth === 375` in every view, toggle, and selection.

## 7. Function signatures (`positional_encodings_geometry.py`)

Bodies are written after Jonathan's reaction.
- **Topic 9** is imported as a module object (`import attention_as_kernel_smoothing as ak`), never `from … import`,
  because its `_run_harness` collects `test_*` globals.
- **Reused:**
  - `ak.attention`, `ak._softmax`, `ak._r`, `ak._sig`, `ak.SEED`;
  - Topic 5 via `ak.nlm`: `complex_step_grad`, `central_diff_grad`, `log_softmax`, `ADAM_*`;
  - `ak.fc` (`generate_corpus`).
- **Run line:** `uv run --with numpy==2.5.4 --with scipy==1.18.1 python
  notebooks/positional-encodings-geometry/positional_encodings_geometry.py`.

```python
BASE = 10_000.0
D_TOUR, L_TRAIN, N_SYMBOLS, CUE_RATE = 16, 32, 8, 0.1
TEST_LENGTHS = (32, 64, 128, 256)
SCHEMES = ("nope", "sinusoidal", "learned", "rope", "alibi")
RESCUES = ("pi", "ntk", "yarn")
ALIBI_SLOPE = 0.5                     # ALiBi's first slope at H = 8
TOUR_SEEDS = (0, 1, 2)                # streams default_rng([ak.SEED, 1000 + 10*seed + k])

# §1 order, and how a mask leaks it
def causal_counter(t_max: int) -> np.ndarray                                   # Prop 1 (i): o_t = e/(t+1), via ak.attention
def bag_bayes_accuracy(n: int, rate: float, trials: int, rng) -> float         # the best multiset reader (Prop 1 (ii))

# §2 position as a set of clocks
def frequencies(d: int, base: float = BASE) -> np.ndarray                      # ω_j = b^{−2j/d}
def sinusoidal(positions, freqs) -> np.ndarray                                 # Def 2, (cos, sin) per plane
def rotation(k: float, freqs) -> np.ndarray                                    # R_ω(k), d × d
def rotate(X, positions, freqs) -> np.ndarray                                  # rows of X turned by R_ω(position); RoPE's kernel
def kernel(delta, freqs) -> np.ndarray                                         # κ_d(Δ)
def kernel_limit(delta, base: float = BASE) -> np.ndarray                      # Thm 1 (ii), scipy.special.sici
def log_ruler(delta, base: float = BASE) -> np.ndarray                         # 1 − (γ + ln|Δ|)/ln b
def commutant_projection(M) -> np.ndarray                                      # Π(M), Thm 2 (c)
def relativity_defect(M) -> float                                              # ‖M − Π(M)‖_F / ‖M‖_F
def window_condition(n: int, freqs) -> float                                   # the caveat to Thm 2 (a′)

# §3 relative biases
def alibi_slopes(H: int) -> np.ndarray                                         # 2^{−8h/H}
def alibi_bias(n: int, slope: float) -> np.ndarray                             # −s(t − j) on j ≤ t
def t5_bucket(offset: int, num_buckets: int = 32, max_distance: int = 128) -> int  # Def 3, illustration
def alibi_tail_bound(B: float, s: float, D: int) -> float                      # Thm 3 (ii)
def alibi_horizon(B: float, s: float) -> float                                 # Thm 3 (iii): 2B/s

# §4 rotations
def offset_score(q, k, deltas, freqs) -> np.ndarray                            # Prop 3 (iii)
def orthogonal_family(U, ms) -> np.ndarray                                     # Thm 4 (b)
def real_normal_form(U) -> tuple[np.ndarray, np.ndarray, int, int]             # Thm 4 (c): Q, angles, p, r (scipy.linalg.schur)
def abel_envelope(h, delta, freqs) -> dict[str, float]                         # Prop 4 (i): |sum|, bound, (2/d)Σ|S_j|

# §5 past the training window
def wavelength_census(L: int, d: int, base: float = BASE) -> dict              # Thm 5 (i)
def pi_positions(n: int, L: int = L_TRAIN) -> np.ndarray                       # Def 5 / Thm 5 (ii)
def ntk_frequencies(d: int, kappa: float, base: float = BASE) -> np.ndarray    # Thm 5 (iii)
def yarn_frequencies(d: int, kappa: float, L: int = L_TRAIN, alpha=1.0, beta=32.0, base=BASE) -> tuple[np.ndarray, float]
def interpolation_bound(h, freqs, d1: float, d2: float, delta: float) -> float # Thm 5 (iv)

@dataclass(frozen=True)
class HeadConfig:                     # one trained head
    scheme: str; seed: int; source: str = "synthetic"   # or "filings" (open point 1)
    n: int = L_TRAIN; rate: float = CUE_RATE; steps: int = 1500; batch: int = 64; lr: float = 0.01

def cue_data(rng, batch: int, n: int, rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # (X, y, gap)
def init_head(rng, scheme: str, n: int) -> dict
def head_loss_grad(P, X, y, scheme, *, positions=None, freqs=None, slope=ALIBI_SLOPE, temp=1.0)
    -> tuple[float, dict]                                                      # hand-written, checked by test_gradients
def head_accuracy(P, X, y, scheme, *, rescue: str | None = None) -> float
def train_head(cfg: HeadConfig) -> "TrainedHead"                               # lru_cache; ≈ 1 s
def tournament() -> dict                                                       # accuracies [scheme][length][seed] + rescues
def learned_offset_score(P, deltas) -> np.ndarray                              # the cue key's s(Δ), symbol-averaged
def content_bound(P) -> float                                                  # B for Thm 3 over the finite token types

# §6 finance
def org_tokens(doc) -> list[tuple[str, int | None]]                            # mentions collapsed; ORG cues labelled
def filing_queries(docs) -> list                                               # every position with an earlier ORG mention
def filing_accuracy(P, scheme, docs, rescue=None) -> dict                      # overall and by gap band (vectorized per doc)
def finance_summary() -> dict

# payload, summary, harness
def viz_payload() -> dict             # lru_cache
def summary() -> dict
```

**Timing budget:** under 60 s for the harness.
- The tournament is 15 trainings (≈ 15 s), plus 5 more for filing windows if open point 1 is accepted.
- Filing evaluation is vectorized per document (the prototype's per-position loop took ≈ 2 s per scheme).
- Time one training before fixing the step count.

## 8. Notebook cell map (`01_positional_encodings_geometry.ipynb`, ≈ 29 cells)

| cells | content |
|---|---|
| 0–1 | title, imports (`pe`, `ak`), the overview table |
| 2–3 | §1: Prop 1's counter; the NoPE head against the bag reader |
| 4–5 | §2: Def 2; Prop 2 asserted; the clocks (matplotlib-free, numbers only) |
| 6–7 | Thm 1: the kernel, its Ci limit, the ruler table |
| 8–9 | Thm 2: the commutant, a random M's defect, the trained sinusoidal head's defect |
| 10–11 | §3: Def 3, T5 buckets, Thm 3 (i)–(iii) |
| 12–16 | §4: Def 4, Prop 3, Thm 4 (Schur form), Prop 4 (envelope, matched/unrelated pairs) |
| 17–20 | §5: training the five heads; the tournament table; the learned s(Δ) and the rebound |
| 21–23 | Def 5, Thm 5: census, PI, NTK, YaRN, the interpolation bound; the rescues' table |
| 24–26 | §6: the filings, the gap bands, Thm 3 (iii)'s forecast against the measured collapse |
| 27–28 | the lab payload; `summary()` |

**Commands:**
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/positional-encodings-geometry/positional_encodings_geometry.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/positional-encodings-geometry/01_positional_encodings_geometry.ipynb
```

## 9. Cross-site candidates (all `status: "published"` on GitHub `main`, checked 2026-10-24; re-verify at the B3 gate)

| field | slug | bridge |
|---|---|---|
| `formalcalculusPrereqs` | `fourier-series` | Thm 1 (orthogonality of the trigonometric system), cited as the contrast: PE frequencies are not harmonics, and the Gram matrix is κ(m − n); Remark 7 (complex exponential form) for Def 4 |
| `formalcalculusPrereqs` | `linear-algebra` | Ex 9 (the standard matrix of a rotation) and Thm 3 (products compose), for Prop 2 |
| `formalcalculusPrereqs` | `linear-systems` | Def 2, Thm 2 (c)–(d), Ex 5: R_ω(k) = exp(kA) with A block-skew |
| `formalmlPrereqs` | `gaussian-processes` | Def 2 (PD kernel), Def 3 (Matérn, ν = ½ is ALiBi's kernel), Prop 2 (products of kernels) |
| `formalstatisticsPrereqs` | `central-limit-theorem` | Def 1 (characteristic function), Thm 3 (Lévy's continuity): Thm 1 is a CF and its limit |
| `formalcalculusConnections` | `eigenvalues-eigenvectors` | Ex 3 and Remark 2: complex eigenvalues act as a rotation-scaling on a real plane, the blocks of Thms 2 and 4 |
| `formalmlConnections` | `kernel-regression` | Thm 3 (i): ALiBi is Nadaraya–Watson with a product kernel in content and time |
| `formalmlConnections` | `expander-graphs` | Prop 2 (spectrum of abelian Cayley graphs): on cyclic positions a stationary Gram matrix is circulant and the DFT diagonalizes it |
| `formalragConnections` | `retrieval-vs-long-context` | Prop 3 there (a buried passage is a soft erasure) beside Thm 3 (iii) here: a recency kernel is one mechanism for burial; we do not model the U-curve |
| `formalragConnections` | `optimized-product-quantization` | Thm 1 there (a shared rotation preserves inner products); RoPE's different rotations leave exactly R(n − m) |

**Not cited:**
- `hilbert-spaces`, `approximation-theory`, `graph-laplacians`, `spectral-theorem` (optional, weak);
- `smooth-manifolds` (prose only);
- formalStatistics `hidden-markov-models` (draft).

## 10. References (Chicago 17th; every URL and title verified in Phase A)

1. Vaswani et al., "Attention Is All You Need," NeurIPS 30 (2017). arXiv:1706.03762 — Def 2, Prop 2.
2. Shaw, Uszkoreit, and Vaswani, "Self-Attention with Relative Position Representations," NAACL-HLT 2018. ACL Anthology
   N18-2074 — Def 3.
3. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," *JMLR* 21
   (2020) — Def 3 (T5 buckets).
4. Dai et al., "Transformer-XL: Attentive Language Models beyond a Fixed-Length Context," ACL 2019. P19-1285 —
   relative scores (pointer).
5. Ke, He, and Liu, "Rethinking Positional Encoding in Language Pre-training," ICLR 2021. arXiv:2006.15595 — Thm 2's
   cross terms.
6. Su, Ahmed, Lu, Pan, Bo, and Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding,"
   *Neurocomputing* 568 (2024). arXiv:2104.09864 — Def 4, Prop 4 (i).
7. Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length
   Extrapolation," ICLR 2022. arXiv:2108.12409 — Def 3, Thm 3.
8. Chi, Fan, Rudnicky, and Ramadge, "Dissecting Transformer Length Extrapolation via the Lens of Receptive Field
   Analysis," ACL 2023 — Thm 3's reading (to verify).
9. Haviv, Ram, Press, Izsak, and Levy, "Transformer Language Models without Positional Encodings Still Learn
   Positional Information," Findings of EMNLP 2022 — Prop 1.
10. Kazemnejad, Padhi, Natesan Ramamurthy, Das, and Reddy, "The Impact of Positional Encoding on Length
    Generalization in Transformers," NeurIPS 36 (2023). arXiv:2305.19466 — Prop 1.
11. Chen, Wong, Chen, and Tian, "Extending Context Window of Large Language Models via Positional Interpolation,"
    2023. arXiv:2306.15595 — Def 5, Thm 5 (iv).
12. bloc97, "NTK-Aware Scaled RoPE …," Reddit r/LocalLLaMA (2023) — Def 5 (blog).
13. Peng, Quesnelle, Fan, and Shippole, "YaRN: Efficient Context Window Extension of Large Language Models,"
    ICLR 2024. arXiv:2309.00071 — Def 5.
14. Brockwell and Davis, *Time Series: Theory and Methods*, 2nd ed. (Springer, 1991) — Herglotz's theorem (§4.3),
    Thm 1 (i).
15. Horn and Johnson, *Matrix Analysis*, 2nd ed. (Cambridge, 2013) — the real normal form, Thm 4.
16. Olver et al., *NIST Digital Library of Mathematical Functions*, §6 (sine and cosine integrals) — Thm 1 (ii)
    (documentation).
17. Rahimi and Recht, "Random Features for Large-Scale Kernel Machines," NeurIPS 20 (2007) — Thm 1 (i)'s spectral
    reading (via Topic 9).

## 11. Open points for Jonathan's reaction

1. **Finance training source.** Retrain each scheme on 32-token windows of the training filings (cue = ORG mention)
   and test on whole held-out filings. That is "trained short, read long" with no change of distribution besides
   length. The alternative reuses the synthetic-trained heads, as the *spike* did, which mixes a shift in cue
   statistics into the length effect. Recommendation: retrain (≈ 5 s more).
2. **ALiBi's reversal as the case study's punchline**, with Thm 3 (iii) as a forecast checked against the gap bands.
   If the retrained filing heads do not reproduce it, the page reports what happens instead.
3. **Seeds.** Three seeds. The page quotes the mean and the min–max, and the lab shows seed 0's learned curves with
   the seed-mean accuracies. Recommendation as stated.
4. **YaRN evaluated zero-shot** as a third rescue. It costs nothing to evaluate, and it is labeled a heuristic.
   Recommendation: include.
5. **Difficulty** stays `intermediate`, as for #9. Thm 4 cites the orthogonal normal form instead of proving it. The
   alternative is `advanced`.
6. **The `fourier-series` folklore** goes into rigorFlag item 7 and the launch reciprocal PR's notes, phrased as a
   correction of a general claim, not of the sibling.
