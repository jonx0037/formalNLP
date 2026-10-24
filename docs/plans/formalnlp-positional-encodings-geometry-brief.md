# Brief — `positional-encodings-geometry`

**Positional Encodings: Sinusoids, Rotations, and RoPE.** Roadmap #10. Prerequisite: `attention-as-kernel-smoothing`
(#9, PR #17), published. This topic is a prerequisite of `transformer-block-residual-stream` (#11).

**Phase A** (this brief and the verified notebook), 2026-10-24:
- The harness passes 20/20 under Python 3.15.0, numpy==2.5.4, and scipy==1.18.1, in ≈ 33 s. It also passes with
  warnings promoted to errors.
- The narrative notebook (30 cells) executes with exit 0 in ≈ 48 s, including environment and kernel start, and is
  committed without outputs.

**Scope decisions** (Jonathan, 2026-10-24):
- **Computational pillar:** a trained length-generalization tournament in NumPy.
  - Single causal heads, with hand-written gradients and Adam.
  - Schemes: NoPE, sinusoidal-absolute, learned-absolute, RoPE, and ALiBi.
  - Trained at L = 32 and read to 256.
  - The RoPE head is re-read zero-shot under PI, NTK-aware scaling, and YaRN.
- **Context extension:** a full section, flagged.
- **Relative schemes:** ALiBi in full, with Shaw et al. and T5 in one definitional paragraph.
- **The scaffold's six open points, all resolved on Claude's recommendations:**
  1. The finance heads are retrained on 32-token filing windows.
  2. ALiBi's horizon is the case study's punchline, checked as a forecast.
  3. Three seeds; the page quotes means and ranges.
  4. YaRN is evaluated zero-shot.
  5. Difficulty is `intermediate`.
  6. The `fourier-series` folklore goes into the rigorFlag and the launch reciprocal-PR notes.
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–9.

**Changes from the scaffold, each forced by the harness:**
1. **The filing-trained ALiBi heads do not collapse as the scaffold's spike did.** Past 32 tokens they read
   0.580 (0.378–0.759 across seeds), not 0.085. The spike's heads were trained on the synthetic stream, and those heads
   still collapse when read on filings: 0.087–0.112 past their horizon. The case study now carries both, because
   Thm 3 (iii) forecasts both:
   - inside half the horizon, every one of the six heads answers every query (1.000);
   - past the horizon, filing-trained heads fall to 0.319–0.723, and synthetic-trained heads to 0.087–0.112.
2. **Thm 3 is stated with the content scores' width W = max − min,** not a bound |c| ≤ B. The horizon is W/s, which is
   at most 2B/s, and the bound is tighter.
3. **A bug was caught, and the test that should have caught it was strengthened.**
   - The one-pass filing reader first subtracted ALiBi's bias before the 1/τ scale, so it read every ALiBi head at
     slope 1/8 instead of 1/2.
   - The reader test compared only argmaxes at seven positions, and passed. It now matches the plain forward pass's
     log-probabilities to 10⁻⁹ at every position of a held-out filing, for every scheme and rescue.
   - Every filing number below is from the corrected reader.
4. **Chen et al.'s Theorem 2.1 is false as stated, not merely loose.**
   - Their eq. (14) uses c^x ≤ 1 + x ln c for x < 0. Convexity gives the reverse, so their d/(4 ln c) is a lower bound
     on Σ_j θ_j².
   - With h_j = 1 on [0, 1] at s = ½, the interpolation error exceeds their stated bound at d = 16, 64, and 128. At
     d = 128, LLaMA-7B's head dimension and their own setting, it is 0.459282 against 0.434294.
   - The sharp constant Σ_j θ_j² = (1 − c^{−2})/(1 − c^{−4/d}) holds. arXiv v2 (2023) is the latest version.
5. **T5's last bucket starts at offset 113, not 128.** The log map reaches bucket 31 at 16 · 8^{15/16} ≈ 113.
6. **Thm 1 (ii)'s finite-d convergence is not monotone pointwise at large |Δ|.** At Δ = 256 the gap is 0.018819 at
   d = 64 and 0.021398 at d = 512. The assert is therefore on the largest gap over |Δ| ≤ 64.
7. **The RoPE rebound is sharper than the scaffold's claim.** The scaffold's "rises above the score at the median
   gap" failed for seed 0 (84.137 < 91.984). As measured, every seed:
   - falls monotonically to a trough at 32–34, within 3 of λ₂/2 = 31.4;
   - has its largest amplitude in plane 3 (λ₃ = 198.7);
   - rebounds to a peak at 181–182, near 3λ₂ ≈ 188 and λ₃;
   - and the peak is above the score of every in-window cue 16 or more back.
8. **The rescues split by task.** On the synthetic stream at 256, NTK-aware scaling leads (0.977000; PI 0.917000). On
   filings past 32, PI leads (0.985645; NTK 0.788389). At d = 16, YaRN is PI with mild corrections on planes 0 and 1
   (γ = 0.132 and 0.021), plus its logit scale, and it tracks PI on both tasks.
9. **One NoPE filing seed answered "the first-named company" at every position.** Its 0.766 equals the share of
   queries whose answer is that company. The other two seeds learned the same multiset rule (0.536 each).

**Source of truth:** `notebooks/positional-encodings-geometry/positional_encodings_geometry.py`. Every number below is
printed or asserted by it, and none may change without it.

**Binding forward promise, and where it is kept.**
- **Topic 9, §"Where this goes next":** "Proposition 3 says one layer of attention cannot tell *the dog bit the man*
  from *the man bit the dog*. The positional-encodings topic adds position, and Theorem 1's features
  $(\cos\theta, \sin\theta)$ are already a sinusoidal encoding of one frequency: several frequencies, and rotations of
  the queries and keys, are its subject."
  - *Adds position:* §§2–4.
  - *Several frequencies:* Def 2, Prop 2, Thm 1.
  - *Rotations of the queries and keys:* Def 4, Prop 3, Thm 4.
  - Topic 9's Prop 3 (iii), "the causal mask itself leaks order," is made exact by Prop 1.
- **A callback.** Topic 9's trained circle denoiser landed on M ≈ cI because its task distribution is
  rotation-invariant. cI is the one-plane case of Thm 2's commutant.

**Thesis.** A positional encoding is a choice of kernel over offsets, and length generalization asks what that kernel
does past the training window.
- **Sinusoids** are a fixed stationary kernel that rules distance logarithmically (§2).
- **Added to the content, they do not make attention relative.** Trained sinusoidal heads sit as far from the
  commutant as a random matrix, and they collapse at 2× length (§§2, 5).
- **ALiBi** is Topic 9's smoother with a content × exponential product kernel. One inequality bounds its reach at every
  length and puts a horizon on what it can retrieve (§§3, 6).
- **RoPE** makes every score a function of the offset. Up to the learned projections, it is the only linear way to do so
  (§4).
- **Its offset score is a trigonometric polynomial.** Training pins it on [0, 32) only; past that, the planes that never
  completed a turn rebound. Interpolating positions keeps every angle inside the trained arc (§5).

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Positional Encodings: Sinusoids, Rotations, and RoPE |
| `{{SUBTITLE}}` | A positional encoding is a kernel over offsets: sinusoids are a fixed stationary kernel that rules distance logarithmically, RoPE rotates queries and keys so every learned score sees only the offset, and ALiBi multiplies in an exponential kernel that extrapolates in length because it cannot look far. |
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
  assumed U-shaped weight. This topic offers one partial mechanism, a recency kernel with a horizon (Thm 3). It does
  not re-model the U-curve or the cost, and it does not claim RoPE explains the U-curve.
- **formalML `gaussian-processes`** owns positive-definite kernels, Matérn, stationarity, and Bochner (in prose). We cite
  its Def 2, Def 3 (ν = ½), and Prop 2.
- **formalCalculus `fourier-series`** owns the orthogonality of the trigonometric system (its Thm 1). Its §12 says
  sinusoidal encodings generalize to unseen lengths and that the columns are orthogonal. We cite its Thm 1 as the
  contrast, and the claim is logged for the launch reciprocal PR (§5, item 7). The sibling is not touched.
- **formalCalculus `linear-systems`** owns the matrix exponential (R(k) = exp(kA), its Ex 5).
- **formalStatistics `central-limit-theorem`** owns the characteristic function (its Def 1) and Lévy's continuity
  theorem (its Thm 3).
- **This topic owns:**
  - the causal counter;
  - shift-as-rotation;
  - the sinusoidal kernel and its logarithmic ruler;
  - the commutant;
  - ALiBi's product kernel, locality, and horizon;
  - RoPE's relativity and its characterization;
  - what long-term decay means;
  - the census and the rescalings, including the corrected interpolation bound;
  - the tournament and the filings.
- **Out of scope:**
  - multi-head and multi-layer position, and the residual stream (#11);
  - BERT's learned table (#12);
  - scaling (#13);
  - fine-tuning after rescaling;
  - long-context engineering (formalRAG);
  - 2-D and graph encodings.

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | trained at 32 tokens and read at 256: sinusoidal-absolute 0.142500, RoPE 0.465667, RoPE + PI 0.917000, ALiBi 0.977833. On whole filings, past 32 tokens: ALiBi 0.580072, RoPE + PI 0.985645 | — | cells 0–1 |
| 1 | Order, and how a mask leaks it | Topic 9's Prop 3 by link; a BOS value through one causal layer reads 1, ½, ⅓, … | **Prop 1** | cells 2–3 |
| 2 | Position as a set of clocks | the hands of d/2 clocks turning at rates 1, b^{−2/d}, …; two positions one rotation apart | Def 1; Def 2; **Prop 2**; **Thm 1**; **Thm 2** | cells 4–9 |
| 3 | Relative position: biases and the ALiBi kernel | Shaw's clipped offsets; T5's buckets, the last of which holds every offset from 113 on; ALiBi's weights as a product of two kernels | Def 3; **Thm 3** | cells 10–11 |
| 4 | Rotations: RoPE | rotate the query by m and the key by n; only n − m survives the dot product | Def 4; **Prop 3**; **Thm 4**; **Prop 4** | cells 12–17 |
| 5 | Past the training window | the tournament table; the RoPE head's cue score falling over [0, 32) and rebounding to a peak at 181 | the tournament; Def 5; **Thm 5** | cells 18–23 |
| — | The lab | `<PositionalEncodingLab client:visible />` | — | cells 27–28 |
| 6 | Finance case study: which company was named last? | held-out filing 92: the parent named early, a subsidiary since | Thm 3 (iii) as a forecast | cells 24–26 |
| 7 | Where this goes next | — | #11 (Prop 1's counter written into the residual stream; many heads, many slopes and frequencies); #12 (BERT's learned table); formalRAG `retrieval-vs-long-context` | — |

- **Lab placement:** after §5. Figure A needs §§2–4, and Figure B needs §5. Figure B's filing view previews §6.
- **Back-links to Topic 9:**
  - Prop 3 from §1;
  - Prop 1 (iii) from Prop 3 (ii);
  - Def 1 from Thm 3 (i);
  - Thm 1 (the circle) from Def 2 and Thm 2;
  - Prop 8 (random Fourier features) from Thm 1 (i);
  - `attention()` and the lag-task training pattern from §5.

## 2. Definitions

**Notation.**
- d is even, and the planes are j = 0, …, d/2 − 1.
- The frequencies are ω_j = θ_j = b^{−2j/d}, with b = 10⁴ unless stated.
- R(φ) is the plane rotation, J = R(π/2), and R_ω(k) = ⊕_j R(kω_j).
- Rows are tokens: q_t = h_t W_Q and k_j = h_j W_K, so the content score is h_t M h_jᵀ/τ with M = W_Q W_Kᵀ and
  τ = √d_k.
- Under the causal mask the offset is r = t − j ≥ 0. In general Δ = n − m.

**Definitions.**
- **Def 1 (absolute additive encoding).** h_m = x_m E + p(m), with a learned table (undefined past its length) or a
  fixed code.
- **Def 2 (sinusoidal code; Vaswani et al. 2017).** p(m) = (cos mω_0, sin mω_0, …). Vaswani's sin-before-cos order is
  a fixed reflection per plane, which leaves every inner product unchanged. Topic 9's u(θ) is the one-plane case.
- **Def 3 (relative schemes).**
  - *Shaw, Uszkoreit, and Vaswani (2018):* q_t(k_j + a_{clip(j−t, K)})ᵀ/τ.
  - *T5 (Raffel et al. 2020):* a learned scalar per bucket of t − j. The causal buckets are exact below 16,
    log-spaced to 128, and shared from 113 on.
  - *ALiBi (Press, Smith, and Lewis 2022):* q_t k_jᵀ/τ − s_h(t − j), with fixed s_h = 2^{−8h/H}.
  - Our single head uses s = ½, ALiBi's first slope at H = 8.
- **Def 4 (RoPE; Su et al. 2024).** q̃_m = R_θ(m)q_m and k̃_n = R_θ(n)k_n: Def 2's frequencies, applied
  multiplicatively. In complex form, z_j ↦ z_j e^{imθ_j}.
- **Def 5 (context-extension maps)**, for target length L′ = κL:
  - *PI (Chen et al. 2023):* m ↦ m/κ.
  - *NTK-aware (bloc97 2023):* b ↦ bκ^{d/(d−2)}.
  - *YaRN (Peng et al. 2024):* with r_j = L/λ_j and γ ramping linearly from 0 at r ≤ α = 1 to 1 at r ≥ β = 32,
    θ′_j = (1 − γ)θ_j/κ + γθ_j. Logits are scaled by (0.1 ln κ + 1)².

**Configurations.**
- **The cue task.**
  - Each sequence is n − 1 tokens, then a query token. Each token has a uniform symbol from V = 8 and carries a cue bit
    with probability 0.1. At least one cue is forced.
  - The target is the symbol on the most recent cue.
  - Inputs are one-hot: [symbol, cue, query] ∈ ℝ^{10}.
- **The head.**
  - E ∈ ℝ^{10×16}, and W_Q, W_K, W_V ∈ ℝ^{16×16}. The readout is W_o ∈ ℝ^{16×8} with a bias. Causal, read at the query.
  - Initialization follows Topic 9's lag head: N(0, ½) for E (and for the learned table P ∈ ℝ^{32×16}), and
    N(0, 1/√d) for the maps.
  - Training: 1,500 Adam steps (Topic 5's constants) on fresh batches of 64 at L = 32, lr 0.01.
- **Test sets.** 2,000 fresh sequences per length (32, 64, 128, 256), shared by every head. The rescues are evaluated
  zero-shot at κ = n/32.
- **Seeds.** Three per configuration, on streams default_rng([SEED, k]) with SEED = Topic 9's 20261023:
  - k = 1100 + 10·seed + scheme index for synthetic heads;
  - k = 1200 + … for filing heads;
  - k = 1300 for the test sets;
  - k = 1001–1010 for the claims.
- **Filings.**
  - *Documents.* Train: `generate_corpus(300, 0)`. Held out: `generate_corpus(100, 1)`, median 198 tokens and at most
    264, with a median of 8 ORG mentions per document.
  - *Tokens.* Each entity mention collapses to one token. An ORG mention is a cue, labeled by its company's order of
    first appearance (mod 8; at most 5 companies occur). Other tokens carry their word's CRC-32 mod 8.
  - *Training queries.* Every (document, t) whose 31-token window ending at t holds an ORG mention.
  - *Test queries.* Every held-out position with an earlier ORG mention: 19,285 in all, 7,895 (40.9%) of them more than
    32 tokens back.
  - *Gap bands* (counts): 1–8 (5,106), 9–16 (2,988), 17–32 (3,296), 33–64 (3,734), 65–128 (3,932), 129+ (229).
  - The answers' shares by label: 0.766 (first-named), 0.133, 0.077, 0.024.

## 3. Statements and proofs

### Proposition 1 (a causal layer counts but does not order)
- **(i)** With equal scores, a BOS value e at position 0, and zero values elsewhere, one causal layer outputs
  o_t = e/(t + 1).
- **(ii)** One causal layer reads its prefix as a multiset (Topic 9, Prop 3 (ii)). The multiset's size is the position,
  but its order is invisible.

*Proof.* (i) The weights are uniform over t + 1 positions. (ii) The size of a multiset is a function of it. ∎

*Reading.*
- A causal NoPE stack can write position in layer 1 and use it in layer 2 (Haviv et al. 2022; Kazemnejad et al. 2023).
- A single NoPE head on the cue task is a multiset reader. The tokens are exchangeable, so the best multiset reader
  guesses the most frequent cue symbol, with accuracy E[max_s c_s/Σ_s c_s].

**Asserted:**
- `test_causal_counter`: (i) exactly for t ≤ 4,096, with t recovered as round(1/o_t) − 1.
- `test_nope_reads_a_bag`: the NoPE heads never beat the best multiset reader, within 4 standard errors.

| n | best multiset reader (20,000 draws) | NoPE heads (3 seeds) |
|---|---|---|
| 32 | 0.544666 ± 0.001797 | 0.5380, 0.5410, 0.5455 |
| 64 | 0.372293 ± 0.000995 | 0.3590, 0.3690, 0.3730 |
| 128 | 0.286284 ± 0.000528 | 0.2855, 0.2875, 0.2865 |
| 256 | 0.234103 ± 0.000318 | 0.2375, 0.2315, 0.2435 |

### Proposition 2 (shift is a rotation; Vaswani et al.'s "linear function", proved)
- **(i)** p(m + k) = R_ω(k)p(m).
- **(ii)** k ↦ R_ω(k) is a homomorphism ℤ → SO(d), with R_ω(k)ᵀ = R_ω(−k) and R_ω(k) = exp(kA) for A = ⊕ω_jJ
  (formalCalculus `linear-systems` Ex 5).
- **(iii)** ‖p(m)‖² = d/2.
- **(iv)** The code is injective, because plane 0 has ω_0 = 1 and π is irrational.

*Proof.* (i) is the angle-addition formulas, plane by plane. (ii) follows from (i) and the block structure.
(iii) cos² + sin² = 1. (iv) p(m) = p(n) forces m − n ∈ 2πℤ. ∎

**Asserted** (`test_shift_is_rotation`):
- (i) to 10⁻⁹ on random m, k ∈ [−10⁴, 10⁴] at d = 16, 64, 512;
- (ii), including det = 1 block by block (LAPACK's det on this sparse matrix raises spurious floating-point flags under
  Accelerate);
- (iii);
- (iv) on m < 10⁴.

### Theorem 1 (the stationary kernel, and the logarithmic ruler)
- **(i)** ⟨p(m), p(n)⟩ = κ_d(n − m), with κ_d(Δ) = Σ_j cos(Δω_j).
  - Every Gram matrix is Toeplitz and positive semidefinite.
  - (2/d)κ_d(Δ) = E cos(ΔΩ_d) is the characteristic function of Ω_d, uniform on the d atoms ±ω_j (formalStatistics
    `central-limit-theorem` Def 1).
  - By Herglotz's theorem (*cited*: Brockwell and Davis 1991, §4.3), every positive-definite function on ℤ is the
    characteristic function of a measure on (−π, π]. A stationary code is a choice of spectral measure.
- **(ii) The logarithmic ruler.** As d → ∞ with b fixed, Ω_d ⇒ ±b^{−U}, with U uniform on [0, 1] (density
  1/(2|ω| ln b) on 1/b ≤ |ω| ≤ 1). For fixed Δ ≠ 0,

  (2/d)κ_d(Δ) → I_b(Δ) = [Ci(|Δ|) − Ci(|Δ|/b)]/ln b = 1 − (γ + ln|Δ|)/ln b + [Ci(|Δ|) + O(Δ²/b²)]/ln b.

*Proof.*
- (i) Angle subtraction, then Σ c_m c_n κ(m − n) = ‖Σ c_m p(m)‖² ≥ 0.
- (ii) u_j = 2j/d is a left Riemann grid on [0, 1), so the empirical law of u_j converges to uniform, and Ω_d converges
  in distribution by the continuous mapping u ↦ b^{−u}. Since cos(Δ·) is bounded and continuous, the expectations
  converge (formalStatistics `central-limit-theorem` Thm 3).
- Substituting ω = b^{−u}, du = −dω/(ω ln b), gives ∫_{1/b}^1 cos(Δω)/(ω ln b) dω = [Ci(|Δ|) − Ci(|Δ|/b)]/ln b.
- The expansion uses Ci(x) = γ + ln x − ∫_0^x (1 − cos t)/t dt (DLMF 6.2.11) at x = |Δ|/b, where the integral is
  O(x²).
- ∎

*Reading.*
- Each doubling of |Δ| costs ln 2/ln b = 0.075257 of similarity.
- Geometric frequencies give every octave of distance equal resolution.

| Δ | 1 | 4 | 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|---|---|
| d = 64 | 0.966151 | 0.747949 | 0.605401 | 0.434565 | 0.353683 | 0.262925 |
| d = 512 | 0.973055 | 0.768313 | 0.630988 | 0.485390 | 0.313466 | 0.186206 |
| d = 8,192 | 0.973907 | 0.771306 | 0.634519 | 0.487258 | 0.334749 | 0.185029 |
| Ci limit | 0.973963 | 0.771508 | 0.634758 | 0.487335 | 0.334864 | 0.185022 |
| ruler 1 − (γ + ln Δ)/ln b | 0.937330 | 0.786815 | 0.636300 | 0.485785 | 0.335270 | 0.184755 |

**Asserted:**
- `test_stationary_kernel`: (i) to 10⁻⁹, PSD on random windows, and the characteristic-function identity to 10⁻¹².
- `test_logarithmic_ruler`:
  - d = 8,192 is within 10⁻³ of the limit for |Δ| ≤ 1,024;
  - the largest gap over |Δ| ≤ 64 shrinks along d = 64, 512, 8,192;
  - the ruler is within 0.01 of the limit for 16 ≤ |Δ| ≤ 1,024.

### Theorem 2 (which score matrices see only offsets)
Let the ω_j be distinct in (0, π). For M ∈ ℝ^{d×d}, the following are equivalent:
- **(a)** p(m)ᵀMp(n) depends only on n − m, for all m, n ∈ ℤ;
- **(a′)** the same holds on one window of d + 1 consecutive positions;
- **(b)** M R_ω(1) = R_ω(1) M;
- **(c)** M = ⊕_j (a_j I + b_j J).

The matrices in (c) form a d-dimensional subspace. Its Frobenius-orthogonal projection Π keeps each diagonal block's
(tr B/2)I + ((B₂₁ − B₁₂)/2)J and zeroes the rest. On it the score is Σ_j a_j cos(Δω_j) + b_j sin(Δω_j).

*Proof.*
- (a) ⇔ (b): g(m + 1, n + 1) = g(m, n) for all m, n ⟺ p(m)ᵀ(RᵀMR − M)p(n) = 0 for all m, n ⟺ RᵀMR = M, since the
  p(m) span ℝ^d.
- (a′) ⇒ (b): any d consecutive p(m) already span ℝ^d, because [e^{imλ}] over the d distinct nodes e^{±iω_j} is
  Vandermonde.
- (b) ⇔ (c): R has distinct eigenvalue pairs, so its commutant is block-diagonal, and a real 2 × 2 block commuting with
  a rotation by an angle in (0, π) is aI + bJ.
- ∎

*Reading.*
- **A stationary code added to the content does not make attention relative.**
  - Score: (x_tE + p(t))M(x_jE + p(j))ᵀ.
  - Content–position cross terms enter (Ke, He, and Liu 2021), and the position–position term is relative only inside
    the commutant.
  - For a Gaussian M, ‖ΠM‖²/‖M‖² ~ Beta(d/2, (d² − d)/2), so E δ² = 1 − 1/d.
- **The trained sinusoidal heads** have δ(W_Q W_Kᵀ) = 0.970950, 0.955155, 0.973256. A random matrix has
  √(1 − 1/16) = 0.968246: the training never moved toward relativity.
- **Exactness is fragile.** The d = 16 sinusoid matrix's condition number over n positions:

  | n | 16 | 32 | 64 | 256 | 4,096 |
  |---|---|---|---|---|---|
  | cond | 1.897·10¹⁹ | 2.242·10¹⁸ | 8.957·10¹⁷ | 1.243·10¹¹ | 17.66 |

  Its numerical rank over 64 positions is 14 of 16.

**Asserted:**
- `test_commutant`:
  - at random well-separated frequencies (d = 8), the null space of (a′)'s window constraints has dimension exactly d,
    and all of it lies in the commutant;
  - (c) ⇒ (a) to 10⁻¹⁰;
  - E δ² = 1 − 1/d by 4,000 draws, within 4 standard errors;
  - the condition numbers fall monotonically, by more than 10³.
- `test_sinusoidal_head_defect`: every trained head has δ > 0.8.

### Theorem 3 (ALiBi is a product kernel: locality, and its horizon)
For one causal ALiBi head with slope s, let the content scores c_tj = q_t k_jᵀ/τ lie in an interval of width W.
- **(i)** α_tj ∝ e^{c_tj} · e^{−s(t−j)}: Topic 9's Nadaraya–Watson smoother (its Def 1) with a content × time product
  kernel.
  - In bidirectional form e^{−s|t−j|} is the Matérn-½ kernel (formalML `gaussian-processes` Def 3). It is positive
    definite, and so is the product (that topic's Prop 2).
- **(ii) Locality at every length.** The weight on offsets ≥ D is at most e^{W}e^{−sD}/(1 − e^{−s}).
- **(iii) The horizon.** α at offset r is at most e^{W − sr} times α at offset 0. Past r = W/s a key weighs less than the
  query's own position, whatever its content.

*Proof.* α_r/α_0 = e^{c_r − c_0 − sr} ≤ e^{W − sr}, which is (iii). Summing over r ≥ D, and using α_0 ≤ 1, gives
(ii). ∎

*Reading.*
- (ii) and (iii) are one inequality read two ways: ALiBi extrapolates in length because it cannot look far. Chi et al.
  (2023) trace ALiBi's extrapolation to a bounded receptive field.
- (ii) is vacuous inside the horizon. With W = 20 and s = ½: 2.54 at D = 40, 1.15·10⁻⁴ at 60, 5.24·10⁻⁹ at 80.
- (iii) bounds a weight, not an answer. Whether a cue past the horizon can still be read depends on what the nearer
  tokens' values carry, and §6 measures it.

**Asserted** (`test_alibi_product_kernel`), on 30 random instances up to length 8,192:
- (i) to 10⁻¹²;
- (ii) at D = 1, 4, 16, 64;
- (iii);
- PSD of the Matérn-½ Gram matrix and of its Schur product with a content Gram matrix.

### Proposition 3 (RoPE is relative by construction, and isometric)
- **(i)** ⟨R_θ(m)q, R_θ(n)k⟩ = ⟨q, R_θ(n − m)k⟩.
- **(ii)** ‖R_θ(m)q‖ = ‖q‖. In Topic 9's Prop 1 factorization, the design weights e^{‖k‖²/2τ} carry no position: RoPE
  moves only angles.
- **(iii)** For fixed q and k, s(Δ) = Σ_j ρ_j cos(Δθ_j + φ_j), with ρ_j = ‖q_j‖‖k_j‖ and φ_j = ∠k_j − ∠q_j.

*Proof.* Prop 2 (ii). In polar form, u(α)ᵀR(φ)u(β) = cos(β + φ − α). ∎

**Asserted** (`test_rope_relative_isometric`):
- (i)–(iii) to 10⁻⁹ on random instances;
- Topic 9's `attention()` on rotated Q and K equals the explicit R(n − m) form to 10⁻¹².

### Theorem 4 (RoPE is the general linear relative encoding)
Let L_m ∈ ℝ^{d×d} (m ∈ ℤ) with L_0 = I. The following are equivalent:
- **(a)** ⟨L_m q, L_n k⟩ depends only on (q, k, n − m);
- **(b)** L_m = U^m for an orthogonal U;
- **(c)** L_m = Q(⊕_i R(mθ_i) ⊕ I_p ⊕ (−1)^m I_r)Qᵀ, with θ_i ∈ (0, π).

W_Q and W_K absorb Q. So, up to the learned projections, the linear relative encodings are RoPE with free frequencies,
plus unrotated planes (partial rotary) and Nyquist planes.

*Proof.*
- (a) ⇒ (b): (a) says L_mᵀL_n = G(n − m).
  - m = n gives L_mᵀL_m = I.
  - G(k) = L_0ᵀL_k = L_k.
  - L_mᵀL_{m+k} = L_k gives L_{m+k} = L_m L_k, so L_m = L_1^m.
- (b) ⇒ (c): the real normal form of an orthogonal matrix (*cited*: Horn and Johnson 2013, §2.5).
- (c) ⇒ (a): Prop 3 (i).
- ∎

*Reading.* RoPE fixes U's spectrum to Def 2's frequencies. Su et al. derive the 2-D case under smoothness assumptions,
and (b) needs none.

**Asserted** (`test_general_relative_encoding`), at d = 7, 8, 9, for a random orthogonal U and for one with explicit
±1 axes:
- (a) via L_mᵀL_n = L_{m+1}ᵀL_{n+1};
- scipy's real Schur form reconstructs U^m for m = −3, …, 5 to 10⁻⁸, with angles in (0, π) and 2·#angles + p + r = d;
- a symmetric non-orthogonal A violates (a).

### Proposition 4 (what "long-term decay" means)
- **(i)** With h_j = q_j k̄_j and S_j = Σ_{l<j} e^{iΔθ_l}: |Σ_j h_j e^{iΔθ_j}| ≤ (max_j |h_{j+1} − h_j|) Σ_j |S_j|, with
  h_{d/2} = 0 (Su et al.).
- **(ii)** If q = k with E‖q_j‖² = σ² for each plane, then E⟨q, R(Δ)q⟩ = σ²κ_d(Δ). This is Thm 1's ruler.
- **(iii)** If q and k are independent and k is isotropic in each plane, then the law of ⟨q, R(Δ)k⟩ does not depend on
  Δ. For the Gaussian case: mean 0, variance dσ⁴/4.

*Proof.* (i) Summation by parts: Σ_j h_j(S_{j+1} − S_j) = −Σ_j S_{j+1}(h_{j+1} − h_j). (ii) uᵀR(φ)u = ‖u‖² cos φ.
(iii) R(Δ)k has k's law. ∎

*Reading.*
- The envelope (2/d)Σ|S_j| at d = 128 falls on average. Its octave means, for octaves [2^k, 2^{k+1}) with k = 0…11:
  31.538, 27.277, 20.995, 17.695, 14.367, 12.020, 10.020, 8.268, 6.973, 5.960, 5.298, 4.880.
- It is not monotone within octaves. That part is a measurement, not a theorem.
- "Decay" is a property of an envelope and of matched pairs. A trained head decays only where it learned to (§5).

**Asserted** (`test_long_term_decay`):
- (i) on 200 random instances;
- the octave means fall;
- (ii) and (iii) by 200,000 draws at Δ = 0, 1, 7, 50, 400, within 4 standard errors, with the variance within 2%.

### The tournament (§5)
**Accuracy:** mean over 3 seeds, with the range.

| scheme | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| NoPE | 0.541500 (0.538–0.546) | 0.367000 (0.359–0.373) | 0.286500 (0.286–0.288) | 0.237500 (0.232–0.244) |
| sinusoidal-absolute | 0.995833 (0.994–0.997) | 0.200667 (0.160–0.225) | 0.133667 (0.125–0.146) | 0.142500 (0.126–0.171) |
| learned-absolute | 0.988500 (0.985–0.993) | n/a | n/a | n/a |
| RoPE | 0.999833 (1.000–1.000) | 0.947167 (0.917–0.965) | 0.872167 (0.797–0.915) | 0.465667 (0.310–0.677) |
| ALiBi | 0.998667 (0.998–0.999) | 0.979000 (0.975–0.986) | 0.978000 (0.971–0.984) | 0.977833 (0.971–0.985) |
| RoPE + PI | — | 0.998000 | 0.979833 | 0.917000 |
| RoPE + NTK | — | 0.969500 | 0.966000 | 0.977000 |
| RoPE + YaRN | — | 0.998167 | 0.978667 | 0.914000 |

**The RoPE head's cue score** s(r) = ⟨q, R(−r)k_cue⟩/τ (symbol-averaged), by seed:

| seed | s(1) | s(16) | trough | rebound peak | plane amplitudes ρ_j/τ, j = 0…7 |
|---|---|---|---|---|---|
| 0 | 114.100 | 53.528 | r = 34 (5.909) | r = 181 (84.137) | 0.23, 2.57, 24.97, 59.09, 30.09, 4.86, 9.32, 6.60 |
| 1 | 136.705 | 73.192 | r = 32 (31.697) | r = 181 (131.166) | 0.13, 3.74, 26.17, 65.15, 15.49, 10.58, 26.56, 8.51 |
| 2 | 122.109 | 64.276 | r = 32 (24.593) | r = 182 (123.120) | 0.25, 3.83, 28.38, 45.08, 5.74, 13.53, 29.31, 10.44 |

*Reading.* Training pins s on [0, 32) only. The head carries its recency in planes 2 (λ = 62.8) and 3 (λ = 198.7),
whose half-turns span the window. Past it, plane 2 turns back first: the trough at λ₂/2 is the window's edge. The two
planes nearly realign near 3λ₂ ≈ 188 and λ₃, where a far cue outscores any true cue 16 or more back.

**Asserted:**
- `test_gradients`: every scheme, and YaRN's logit scale, against central differences (Topic 5's
  `param_grad(central_diff_grad, …)`), ≤ 10⁻⁷.
- `test_tournament`:
  - every scheme except NoPE is ≥ 0.95 at 32, and NoPE is < 0.7;
  - learned-absolute is n/a past 32;
  - sinusoidal-absolute is < 0.35 past 32;
  - ALiBi is ≥ 0.9 at every length;
  - RoPE falls strictly with length in every seed;
  - every rescue lifts each seed's RoPE head at 256 by more than 0.2.
- `test_rope_rebound`: the trough, the monotone fall, plane 3's amplitude, a peak in [170, 200], and peak > s(16), in
  every seed.

### Theorem 5 (wavelength census and rescalings)
- **(i) Census.** Plane j turns a whole circle over training offsets 0, …, L − 1 iff (L − 1)θ_j ≥ 2π.
  - At d = 16, L = 32 the wavelengths are 6.3, 19.9, 62.8, 198.7, 628.3, 1,986.9, 6,283.2, 19,869.2.
  - Two planes complete a turn. The rest see angles past L that training never produced.
- **(ii) PI** maps test offsets [0, κL − 1] into [0, L − 1/κ], within one position of the trained range, at a κ-fold
  loss of resolution in every plane.
  - The loss is visible on filings: RoPE + PI reads 0.995 on gaps ≤ 8, where RoPE reads 1.000.
- **(iii) NTK-aware.** θ′_j = θ_j κ^{−2j/(d−2)}: plane 0 is unchanged, and the slowest plane gets exactly 1/κ.
  - At κ = 4 the per-plane ratios are 1, 0.820, 0.673, 0.552, 0.453, 0.371, 0.305, 0.250.
  - YaRN's ratios at κ = 4 are 0.349, 0.265, then 0.25 for every other plane, with logit scale 1.296.
- **(iv) Interpolation versus extrapolation.**
  - Linear interpolation of a(Δ) = Re Σ_j h_j e^{iΔθ_j} between Δ₁ < Δ₂ errs by at most
    ½(Δ − Δ₁)(Δ₂ − Δ)Σ_j |h_j|θ_j², with Σ_j θ_j² = (1 − b^{−2})/(1 − b^{−4/d}).
  - Past the trained range no such bound holds, and |a| can reach Σ|h_j|.
  - **Chen et al.'s Theorem 2.1 states the constant d/(4 ln b).** Their eq. (14) derives it from c^x ≤ 1 + x ln c for
    x < 0, the reverse of convexity. Their constant undercounts Σθ_j² by 2.558× at d = 16, 1.315× at 64, 1.151× at
    128, and 1.036× at 512.
  - The stated bound fails for h_j = 1 on [0, 1] at ½:

    | d | error | stated bound | sharp bound |
    |---|---|---|---|
    | 16 | 0.121137 | 0.054287 | 0.138889 |
    | 64 | 0.259756 | 0.217147 | 0.285611 |
    | 128 | 0.459282 | 0.434294 | 0.499789 |
    | 512 | 1.667841 | 1.737178 (holds) | 1.800427 |

  - Their qualitative conclusion, that interpolation is far better controlled than extrapolation, survives with the
    corrected constant.

*Proof.* (i) An arc is complete iff its length reaches 2π. (ii) (κL − 1)/κ = L − 1/κ. (iii) Substitute the new base.
(iv) The interpolation remainder is a(Δ) − ℓ(Δ) = a″(ξ)(Δ − Δ₁)(Δ − Δ₂)/2, with |a″| ≤ Σ|h_j|θ_j². ∎

**Asserted:**
- `test_wavelength_census`: (i)–(iii), and YaRN's ramp endpoints.
- `test_interpolation_bound`:
  - (iv) on 600 random instances at d = 16 and 128;
  - the closed form for Σθ_j²;
  - the 2.558× and 1.036× ratios;
  - the counterexample at d = 16, 64, 128.

## 4. The finance case study (§6): which company was named last?
- **Setting.**
  - A filing names a parent and its subsidiaries, then refers back to "the company".
  - At every held-out position with an earlier ORG mention we ask which company was named most recently.
  - The heads are retrained on 32-token windows of the training filings, so at test time the only shift is length.
- **Results:** accuracy, mean of 3 seeds (`filing_benchmark`).

  | reader | all (19,285) | past 32 (7,895) | 1–8 | 9–16 | 17–32 | 33–64 | 65–128 | 129+ |
  |---|---|---|---|---|---|---|---|---|
  | NoPE | 0.612963 | 0.528436 | 0.693 | 0.677 | 0.633 | 0.533 | 0.512 | 0.738 |
  | sinusoidal | 0.761818 | 0.711590 | 0.800 | 0.795 | 0.793 | 0.733 | 0.693 | 0.668 |
  | RoPE | 0.878040 | 0.720583 | 1.000 | 1.000 | 0.956 | 0.771 | 0.680 | 0.591 |
  | ALiBi | 0.828087 | 0.580072 | 1.000 | 1.000 | 1.000 | 0.656 | 0.513 | 0.504 |
  | RoPE + PI | 0.990787 | 0.985645 | 0.995 | 0.994 | 0.994 | 0.990 | 0.982 | 0.980 |
  | RoPE + NTK | 0.913128 | 0.788389 | 1.000 | 1.000 | 0.999 | 0.752 | 0.830 | 0.661 |
  | RoPE + YaRN | 0.990303 | 0.984843 | 0.994 | 0.993 | 0.994 | 0.991 | 0.979 | 0.980 |

- **Thm 3 (iii) as a forecast** (`alibi_forecast`): every ALiBi head, from either training source, read on the held-out
  filings.

  | trained on | seed | W | horizon W/s | inside H/2 | beyond H |
  |---|---|---|---|---|---|
  | filings | 0 | 25.553 | 51.1 | 1.000 (n = 10,204) | 0.723 (n = 5,502) |
  | filings | 1 | 26.479 | 53.0 | 1.000 (n = 10,391) | 0.487 (n = 5,395) |
  | filings | 2 | 27.898 | 55.8 | 1.000 (n = 10,572) | 0.319 (n = 5,076) |
  | synthetic | 0 | 19.349 | 38.7 | 1.000 (n = 8,929) | 0.112 (n = 7,048) |
  | synthetic | 1 | 17.875 | 35.8 | 1.000 (n = 8,388) | 0.087 (n = 7,459) |
  | synthetic | 2 | 18.523 | 37.0 | 1.000 (n = 8,666) | 0.108 (n = 7,182) |

- **Lessons.**
  - The tournament's winner loses 42% of the filing's far queries, and its own theorem says where.
  - Inside half its horizon every ALiBi head is perfect.
  - Past the horizon, how much survives depends on how quiet the nearer tokens' values are. Filing-trained heads, which
    saw long cue-free spans, keep more than the synthetic-trained ones.
  - Interpolating RoPE's positions (PI, YaRN) reads the whole filing at ≥ 0.98 in every band.
  - NoPE's 0.613 is mostly the base rate: one seed always names the first company (0.766 of answers are that company).
- **The lab's filing** is held-out document 92 (209 tokens), chosen by `lab_filing()`. It is the filing of at most 220
  tokens with the most queries that the seed-0 ALiBi head misses and RoPE + PI answers.
- **Data rules.** The data is the synthetic corpus only: no transcripts, no LM dictionary. EDGAR is excluded because its
  excerpts carry no entity spans, and the page says so.

## 5. `rigorFlag`
1. **Long-term decay** is a property of an envelope and of matched pairs (Prop 4). RoPE does not make unrelated keys
   fade.
2. **The commutant theorem** (Thm 2) is exact, but over a window shorter than the slow planes' wavelengths its
   constraint is nearly void (condition numbers 10¹⁷–10¹⁹).
3. **Cited, not proved:** Herglotz's theorem (Thm 1 (i)'s converse), the real normal form of orthogonal matrices
   (Thm 4), and Ci's expansion (DLMF).
4. **NTK-aware scaling and YaRN are heuristics.**
   - YaRN's (0.1 ln κ + 1)² logit scale is a fit.
   - Every rescue here is zero-shot, while published results fine-tune after rescaling.
   - Which rescue wins depends on the task: NTK on the synthetic stream, PI on filings.
5. **The tournament is toy-scale:** one head, d = 16, 8 symbols, three seeds. It demonstrates mechanisms, and says
   nothing quantitative about language models at scale.
6. **The finance case uses synthetic filings** with hashed distractor symbols and first-appearance labels. EDGAR is
   excluded.
7. **Folklore corrected:**
   - Sinusoidal encodings do not by themselves extrapolate (0.996 at 32, 0.201 at 64).
   - Their columns are not orthogonal: the Gram matrix is κ_d(m − n), not δ_mn.
   - This contradicts formalCalculus `fourier-series` §12, whose Thm 1 covers integer harmonics over a full period. It
     is logged for the launch reciprocal PR, and the sibling is not touched here.
8. **Chen et al.'s Theorem 2.1 is false as stated for d ≤ 128** (§3, Thm 5 (iv)). The corrected constant restores it,
   and their conclusion stands.
9. **Thm 3 (iii) bounds a weight, not an answer.** Past the horizon the filing-trained heads keep 0.32–0.72, because a
   small weight can still carry the answer when the nearer values are quiet.

## 6. Viz design intent — `PositionalEncodingLab` (one component, two figures)

**Data.**
- All data comes from `viz_payload()` (17,250 bytes).
- Phase B writes `src/components/viz/data/positional-encoding-lab.json` as
  `json.dumps(viz_payload(), separators=(',', ':'))`, and adds `test_viz_mirror_matches_payload`.
- Every displayed number is rounded in the payload: 3 decimals for kernels and accuracies, 2 for offset scores, 1 for
  wavelengths and the horizon.
- Marginalia tokens only. Wrapper: `<div data-lab="positional">`.

**Figure A, "Clocks and kernels."**
- *Clocks:* one dial per plane (d = 16 or 64), each with a hand at angle mω_j.
  - Controls: m and n sliders (0–63), and the base (10², 10⁴, 10⁶).
  - The readout (2/d)κ_d(n − m) comes from `clocks.kernel["<base>-<d>"]`, indexed by |n − m|.
  - *Lesson:* sliding m and n together turns every hand by the same angle, and the readout does not move.
- *Ruler:* (2/d)κ_d(Δ) for d = 16, 64, 512 on a log-Δ axis (40 grid points from 1 to 4,096), with the Ci limit and
  the ruler overlaid (`ruler`).
- *Relativity:* for a fixed offset Δ ∈ {1, 4, 16}, the trained sinusoidal head's position–position score
  p(m)ᵀMp(m + Δ)/τ as m slides (`relativity.lock[Δ].additive`), against the commutant's constant (`.projected`).
  - δ = 0.971 is shown.
  - *Lesson:* the learned score drifts with absolute position, while a relative score could not.

**Figure B, "Past the training window."**
- *Tournament:* accuracy against length (log₂ axis), one line per scheme plus the three rescues (`tournament.accuracy`,
  `.rescues`).
  - Learned-absolute ends at 32.
  - Clicking a scheme shows its seed-0 offset score over 0–255 (`tournament.offset_score`), with [0, 32) shaded and
    RoPE's rebound visible.
- *Census:* one bar per plane for λ_j, against the training length, shaded complete or partial (`census`).
  - Toggle: none / PI / NTK / YaRN at κ = 2, 4, 8, showing each plane's scale factor (`census.scales`).
- *Filing:* held-out document 92 as a token strip, with ORG mentions as chips colored by company label.
  - Clicking a position shows the truth and each reader's answer (`filing.answers`: NoPE, sinusoidal, RoPE, ALiBi,
    RoPE + PI; seed 0, filing-trained).
  - Each reader's accuracy is shown by gap band (`filing.band_accuracy`, seed means), with ALiBi's mean horizon (53.3)
    marked.

*Lessons:*
- the encoding is a prior over offsets;
- RoPE's prior is a trigonometric polynomial that misbehaves where training never looked;
- ALiBi's prior cannot look far;
- interpolation keeps the angles in range.

**Mobile.** Dials reflow into a grid, rows wrap, inputs get `minWidth: 0`, and the token strip wraps.
`scrollWidth === 375` in every view, toggle, and selection.

## 7. The module, as built (`positional_encodings_geometry.py`)
- **Imports Topic 9 as a module object** (`import attention_as_kernel_smoothing as ak`). Topic 9's `_run_harness`
  collects `test_*` globals, so this keeps them out of this harness.
- **Reuses:**
  - `ak.attention` (Prop 1, Prop 3), `ak._softmax`, `ak._r`, `ak.SEED`;
  - Topic 5 via `ak.nlm`: `log_softmax`, `param_grad`, `central_diff_grad`, `ADAM_*`;
  - `ak.fc.generate_corpus`.
- **Sections:**
  - §2 clocks: `frequencies`, `sinusoidal`, `rotation`, `rotate`, `kernel`, `kernel_limit`, `log_ruler`,
    `commutant_projection`, `relativity_defect`, `window_conditions`, `window_condition`.
  - §3: `alibi_slopes`, `alibi_bias`, `t5_bucket`, `alibi_tail_bound`, `alibi_horizon`.
  - §4: `offset_score`, `trig_form`, `orthogonal_family`, `real_normal_form`, `normal_form_power`, `abel_envelope`.
  - §5 maps: `wavelength_census`, `pi_scale`, `ntk_frequencies`, `yarn_frequencies`, `interpolation_bound`,
    `chen_constant`, `chen_stated_bound`.
  - The cue task and head:
    - task and baselines: `cue_data`, `bag_bayes_accuracy`, `causal_counter`;
    - the head: `init_head`, `_head_forward`, `head_loss_grad`, `rescue_kwargs`, `head_accuracy`;
    - training and the tournament: `HeadConfig`, `train_head` (lru_cache, ≈ 1 s each), `cue_test_set`, `tournament`;
    - reading a trained head: `content_scores`, `content_width`, `cue_query_key`, `learned_offset_score`.
  - Filings: `org_tokens`, `Filing`, `filings`, `filing_windows`, `filing_logprobs` (one pass per document; RoPE's
    per-row rescaling through the polar form), `read_filing`, `filing_benchmark`, `alibi_forecast`.
  - Payload: `lab_filing`, `viz_payload`, `summary`, and 20 tests.
- **Timing:** 27 trainings (15 synthetic, 12 filing) ≈ 27 s, and the full harness ≈ 33 s.

## 8. Notebook cell map (`01_positional_encodings_geometry.ipynb`, 30 cells)

| cells | content |
|---|---|
| 0–1 | title; imports, the tournament table |
| 2–3 | §1: Prop 1; the NoPE heads against the best multiset reader |
| 4–5 | §2: Defs 1–2, Prop 2 |
| 6–7 | Thm 1: the kernel, its Ci limit, the ruler table |
| 8–9 | Thm 2: the commutant, the trained heads' defects, the condition numbers |
| 10–11 | §3: Def 3, T5's buckets, Thm 3 |
| 12–13 | §4: Def 4, Prop 3 |
| 14–15 | Thm 4: a random orthogonal 7 × 7 in normal form |
| 16–17 | Prop 4: the envelope's octave means |
| 18–19 | §5: the tournament per seed |
| 20–21 | the RoPE heads' cue scores, troughs, and rebounds |
| 22–23 | Def 5, Thm 5: the census, Chen et al.'s counterexample, the rescues |
| 24–26 | §6: the filings by gap band; Thm 3 (iii)'s forecast |
| 27–28 | the lab payload |
| 29 | `assert pe._run_harness() == 0` |

**Commands** (both exit 0):
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/positional-encodings-geometry/positional_encodings_geometry.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/positional-encodings-geometry/01_positional_encodings_geometry.ipynb
```

## 9. Cross-site links (all `status: "published"` on GitHub `main`, checked 2026-10-24; re-verify at the B3 gate)

| field | slug | bridge |
|---|---|---|
| `formalcalculusPrereqs` | `fourier-series` | Thm 1 (orthogonality of the trigonometric system), cited as the contrast: PE frequencies are not harmonics, and the Gram matrix is κ(m − n); Remark 7 (complex exponential form) for Def 4 |
| `formalcalculusPrereqs` | `linear-algebra` | Ex 9 (the standard matrix of a rotation) and Thm 3 (products compose), for Prop 2 |
| `formalcalculusPrereqs` | `linear-systems` | Def 2, Thm 2 (c)–(d), Ex 5: R_ω(k) = exp(kA) with A block-skew |
| `formalmlPrereqs` | `gaussian-processes` | Def 2 (PD kernel), Def 3 (Matérn; ν = ½ is ALiBi's kernel), Prop 2 (products of kernels) |
| `formalstatisticsPrereqs` | `central-limit-theorem` | Def 1 (characteristic function), Thm 3 (Lévy's continuity): Thm 1 is a characteristic function and its limit |
| `formalcalculusConnections` | `eigenvalues-eigenvectors` | Ex 3 and Remark 2: complex eigenvalues act as rotation-scaling on a real plane, the blocks of Thms 2 and 4 |
| `formalmlConnections` | `kernel-regression` | Thm 3 (i): ALiBi is Nadaraya–Watson with a product kernel in content and time |
| `formalmlConnections` | `expander-graphs` | Prop 2 there (spectrum of abelian Cayley graphs): on cyclic positions a stationary Gram matrix is circulant, diagonalized by the DFT |
| `formalragConnections` | `retrieval-vs-long-context` | Prop 3 there (a buried passage is a soft erasure) beside Thm 3 (iii) here: a recency kernel with a horizon is one mechanism for burial; we do not model the U-curve |
| `formalragConnections` | `optimized-product-quantization` | Thm 1 there (a shared rotation preserves inner products); RoPE's different rotations leave exactly R(n − m) |

**Not cited:**
- `hilbert-spaces`, `approximation-theory`, `graph-laplacians`, `spectral-theorem` (weak);
- `smooth-manifolds` (prose only);
- formalStatistics `hidden-markov-models` (draft).

## 10. References (Chicago 17th; titles verified against arXiv `citation_title`, ACL Anthology, JMLR, and Crossref on 2026-10-24)

1. Vaswani et al., "Attention Is All You Need," NeurIPS 30 (2017). arXiv:1706.03762 — Def 2, Prop 2.
2. Shaw, Uszkoreit, and Vaswani, "Self-Attention with Relative Position Representations," NAACL-HLT 2018.
   https://aclanthology.org/N18-2074/ — Def 3.
3. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," *JMLR* 21 (2020).
   https://jmlr.org/papers/v21/20-074.html — Def 3.
4. Dai, Yang, Yang, Carbonell, Le, and Salakhutdinov, "Transformer-XL: Attentive Language Models beyond a Fixed-Length
   Context," ACL 2019. https://aclanthology.org/P19-1285/ — relative scores (pointer).
5. Ke, He, and Liu, "Rethinking Positional Encoding in Language Pre-training," ICLR 2021. arXiv:2006.15595 — Thm 2's
   cross terms.
6. Su, Ahmed, Lu, Pan, Bo, and Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding,"
   *Neurocomputing* 568 (2024). https://doi.org/10.1016/j.neucom.2023.127063 — Def 4, Prop 4 (i).
7. Press, Smith, and Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation,"
   ICLR 2022. arXiv:2108.12409 — Def 3, Thm 3.
8. Chi, Fan, Rudnicky, and Ramadge, "Dissecting Transformer Length Extrapolation via the Lens of Receptive Field
   Analysis," ACL 2023. https://aclanthology.org/2023.acl-long.756/ — Thm 3's reading.
9. Haviv, Ram, Press, Izsak, and Levy, "Transformer Language Models without Positional Encodings Still Learn Positional
   Information," Findings of EMNLP 2022. https://aclanthology.org/2022.findings-emnlp.99/ — Prop 1.
10. Kazemnejad, Padhi, Natesan Ramamurthy, Das, and Reddy, "The Impact of Positional Encoding on Length Generalization in
    Transformers," NeurIPS 36 (2023). arXiv:2305.19466 — Prop 1.
11. Chen, Wong, Chen, and Tian, "Extending Context Window of Large Language Models via Positional Interpolation,"
    2023. arXiv:2306.15595 — Def 5, Thm 5 (iv) and its correction.
12. bloc97, "NTK-Aware Scaled RoPE Allows LLaMA Models to Have Extended (8k+) Context Size without Any Fine-Tuning and
    Minimal Perplexity Degradation," Reddit r/LocalLLaMA (2023).
    https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/ — Def 5
    (blog). The URL is as cited in Peng et al.'s bibliography, because Reddit blocks direct fetches.
13. Peng, Quesnelle, Fan, and Shippole, "YaRN: Efficient Context Window Extension of Large Language Models,"
    ICLR 2024. arXiv:2309.00071 — Def 5. (Re-check the venue at the B3 gate; OpenReview blocked the fetch.)
14. Brockwell and Davis, *Time Series: Theory and Methods*, 2nd ed. (Springer, 1991). https://doi.org/10.1007/978-1-4419-0320-4
    — Herglotz's theorem (§4.3), Thm 1 (i).
15. Horn and Johnson, *Matrix Analysis*, 2nd ed. (Cambridge University Press, 2013; Crossref dates the online edition
    2012). https://doi.org/10.1017/CBO9781139020411 — the real normal form (§2.5), Thm 4.
16. Olver et al., *NIST Digital Library of Mathematical Functions*, §6. https://dlmf.nist.gov/6 — Thm 1 (ii)
    (documentation).
17. Rahimi and Recht, "Random Features for Large-Scale Kernel Machines," NeurIPS 20 (2007) — Thm 1 (i)'s spectral
    reading, via Topic 9.

## 11. Phase B notes
- **Numbering in page order:**
  P1, D1, D2, P2, T1, T2, D3, T3, D4, P3, T4, P4, D5, T5.
  - Proofs go in separate `<TheoremBlock type="proof">` blocks.
  - `TheoremBlock` titles carry no citations; attribute in the prose.
- **Precision.**
  - Page accuracies are quoted to 3 decimals from the 6-decimal values above, never from `summary()`'s print.
  - Ranges are quoted to 3 decimals.
- **The Chen correction** is stated plainly and once (§5, with a RigorFlag item), with the counterexample's d = 128 row.
  It is not editorialized.
- **MDX:** `\lt`/`\gt` in math; one `$…$` per number in long inline lists.
- **Lab data:** regenerate the JSON from `viz_payload()`, never hand-edit it, then add `test_viz_mirror_matches_payload`.
- **Graph:** flip the node to `published`, and remove its title from `curriculum.ts`'s `planned`.

## 12. Phase B record
*(to be filled at the B3 gate)*
