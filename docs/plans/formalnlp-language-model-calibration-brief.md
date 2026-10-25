# Brief — `language-model-calibration`

**Calibration of Language Models.** Roadmap #16. Its prerequisite is `sampling-temperature-nucleus` (#15), which is
published (PR #28). It closes the Generation layer, and nothing in-site depends on it yet. It **is** on the §6 overlap
watch-list, against formalRAG `selective-generation-abstention`. Phase A also found a second overlap, with formalML
`uncertainty-quantification`. §7 states the scope against both.

**Phase A** (2026-10-25; approved by Jonathan with the recommended answers to §9). Results are numbered per type, as on the page:
- **Harness:** 22/22 pass under numpy==2.5.4 and scipy==1.18.1, also with warnings promoted to errors.
  - The run takes ≈ 47 s, inside the 60 s budget. The ≈ 6 s decoder retrain and the ≈ 5 s log-linear fits dominate.
- **Notebook:** 19 cells; executes with exit 0 in ≈ 49 s, including kernel start, and is committed without outputs.
- **Lab data:** `src/components/viz/data/language-model-calibration-lab.json`, 102 kB, pinned by
  `test_viz_mirror_matches_payload`.
- **Scope:** the agreed "token + fact calibration," with three changes the prototypes forced:
  1. **The oracle is dropped.** The plan scored engines against Topic 15's `oracle()`. Top-label calibration needs
     only held-out text, and canonical calibration is shown exactly on finite worlds instead.
  2. **EDGAR joins the finance case study.** On real filings the in-distribution near-calibration collapses, which is
     the strongest finance finding (§3).
  3. **Thm 3 is new.** It explains why τ = 1 is special, using a log-linear bigram fit to stationarity. This was not
     in the plan.

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

Engines: Topic 3's IKN trigram (V = 3,427, `<unk>` an outcome) and Topic 14's decoder (V = 512, `<unk>` an outcome,
`<s>` masked, first 32 inputs of each sentence). Test sets:

- **continuation:** `ng.continuation_run(100)`, the training generator run on (in distribution). 23,219 tokens.
- **held:** Topic 3's split, `generate_corpus(100, 1)`, a fresh run of the generator. 23,055 tokens.
- **edgar:** the public MD&A excerpts. 830 trigram tokens and 747 decoder tokens.

| Quantity | trigram cont. | trigram held | trigram EDGAR | decoder cont. | decoder held | decoder EDGAR |
|---|---|---|---|---|---|---|
| mean confidence / accuracy | 0.522337 / 0.527111 | 0.511732 / 0.499892 | 0.144343 / 0.059036 | 0.652922 / 0.660967 | 0.649088 / 0.707395 | 0.430239 / 0.195448 |
| ECE₁₅ (plug-in) | 0.017797 | 0.022998 | 0.085307 | 0.015919 | 0.064192 | 0.246355 |
| calibrated-twin floor at B = 15 (mean ± sd, 100 twins) | 0.006124 ± 0.001223 | 0.006125 ± 0.001144 | 0.023486 ± 0.007278 | 0.006342 ± 0.001383 | 0.006368 ± 0.001454 | 0.044464 ± 0.010162 |
| squared ECE₁₅: plug-in / debiased | 0.000430 / 0.000326 | 0.001158 / 0.001060 | 0.042127 / 0.041306 | 0.000434 / 0.000330 | 0.013350 / 0.013251 | 0.114711 / 0.111839 |
| NLL at τ = 1 | 2.537868 | 2.855053 | 8.796272 | 1.407406 | 1.160864 | 5.547841 |
| NLL-optimal τ* | 1.002926 | 1.055108 | 2.365815 | 0.957256 | 0.824420 | 1.726377 |
| NLL at τ* | 2.537852 | 2.849397 | 7.911123 | 1.405293 | 1.126613 | 4.847466 |
| cross-fitted temperature gain (nats) | 0.000016 | 0.005591 | 0.885663 | 0.002096 | 0.034188 | 0.696004 |
| ECE₁₅ at τ* | 0.019030 | 0.022370 | 0.050912 | 0.019263 | 0.041931 | 0.127285 |
| confidence at τ* | 0.521380 | 0.493895 | 0.008474 | 0.669067 | 0.716228 | 0.139203 |
| nucleus(0.9): coverage / mean kept mass | 0.926440 / 0.938600 | 0.875949 / 0.938267 | 0.356627 / 0.904416 | 0.950988 / 0.944538 | 0.966732 / 0.944014 | 0.603748 / 0.906708 |
| nucleus(0.9) coverage at τ* | 0.927301 | 0.891954 | 0.415663 | 0.943495 | 0.940143 | 0.784471 |
| share of tokens that are `<unk>` | 0.027521 | 0.006680 | 0.575904 | 0.102804 | 0.160095 | 0.603748 |

| Quantity | Value | Test |
|---|---|---|
| decoder τ* on every second training sentence | 0.942725 | `test_decoder_has_no_temperature_direction` |
| Kendall's τ_b, trigram row confidences at τ = 1 vs τ = 2 (continuation) | 0.860376 | `test_temperature_keeps_argmax_not_ranking` |
| log-linear bigram, λ = 10⁻⁵ / 10⁻⁴ / 10⁻³: d/dβ L_train at β = 1 | −0.581107 / −1.456424 / −1.091581 | `test_scale_closed_stationarity` |
| … −λ‖W‖² (equal to the line above to 6 decimals) | −0.581107 / −1.456424 / −1.091581 | same |
| … τ*_train | 0.870719 / 0.743596 / 0.520289 | same |
| … τ* on continuation / held | 0.888912, 0.828161 / 0.744253, 0.710561 / 0.515895, 0.472388 | — |
| fact probes seen / once / unseen / all: n | 30 / 33 / 48 / 111 | `test_fact_regimes_pull_apart` |
| … confidence among answers | 0.784609 / 0.331922 / 0.303461 / 0.441962 | same |
| … accuracy (top answer is the KB's) | 1.000000 / 0.696970 / 0.062500 / 0.504505 | same |
| … NLL of the KB answer at τ = 1 | 0.248023 / 1.247546 / 2.739886 / 1.622741 | same |
| … τ* on [0.040, 10] | 0.039811 (edge) / 0.260322 / 10.000000 (edge) / 1.179415 | same |
| pooled NLL: τ = 1 / one pooled τ* / τ* per regime | 1.622741 / 1.612789 / 1.118871 | same |

Notes for the prose:
- The unregularized log-linear bigram fit (λ = 0) reached τ*_train = 1.00000 in a spike, with dL/dβ = −7·10⁻⁸. It
  needed 30 s, and its held-out τ* sits at the search edge (≥ 5), because unseen bigrams get logits near −∞. The
  harness keeps λ > 0.
- In distribution the NLL-optimal τ makes ECE₁₅ slightly *worse* on both engines (0.017797 → 0.019030 and
  0.015919 → 0.019263). Minimizing NLL is not minimizing ECE.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Calibration of Language Models |
| `{{SUBTITLE}}` | When a model says 0.9, is it right nine times in ten? For a next-token distribution over thousands of words the question splits in two: a strong form no corpus can test and a weak form any corpus can. We prove that every log loss separates into what the text cannot tell us, what the model's context throws away, and what a recalibration could recover. Then we measure the weak form honestly, fit the one-parameter fix, explain why τ = 1 is where maximum likelihood leaves it, and watch it all break on real filings. |
| `{{SLUG}}` | `language-model-calibration` |
| `{{DOMAIN}}` | `decoding-generation` |
| `{{DIFFICULTY}}` | `advanced` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[sampling-temperature-nucleus]` (graph edge). Connection: `perplexity-cross-entropy` (log loss) |
| `{{FORMALML_PREREQS}}` | `uncertainty-quantification`, `kl-divergence` |
| `{{FORMALML_CONNECTIONS}}` | `conformal-prediction` |
| `{{FORMALRAG_CONNECTIONS}}` | `significance-testing-calibration`, `selective-generation-abstention` |
| `{{STATS_PREREQS}}` | `exponential-families`, `maximum-likelihood` |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/sampling-temperature-nucleus/` (imported as `st`), `notebooks/beam-search-decoding/` |

## 1. Section outline (geometric first)

1. **A forecast you can check.** The trigram's reliability diagram on the continuation comes first, as a picture: it
   hugs the diagonal. The same diagram on EDGAR leaves it. Lab view 1 opens here.
2. **What calibration means for a 3,427-way forecast.** Def 1 and Prop 1. Canonical calibration asks for all 3,427
   coordinates at once. When every context has its own row, it is the same as being the true conditional, so the
   corpus can test only top-label calibration.
3. **What miscalibration costs.** Thm 1 is geometric first: the Brier score as Pythagoras, twice, then the log loss
   as the same picture with KL in place of squared distance. Cor 1 says no recalibration can recover more than the
   reliability term.
4. **Measuring it honestly.** Thm 2: binning hides and sampling inflates. The calibrated twin gives the noise floor,
   then a table of ECE against the floor for the six engine × test-set cells.
5. **One dial.** Prop 2: temperature scaling is a convex one-parameter exponential family that matches one moment.
   It keeps the argmax but not the ranking, the contrast with formalRAG's Thm 2.
6. **Why τ = 1 at all.** Thm 3 with the log-linear table, and the decoder's 0.94 as the contrapositive.
   - Topic 15's τ, revisited (Cor 2): the same dial set for opposite reasons. Sampling temperature miscalibrates on
     purpose.
7. **Calibrated sets.** Prop 3: a nucleus is a prediction set, and calibration makes its coverage equal its kept
   mass. A coverage table follows, with conformal prediction as the guarantee that does not need calibration
   (formalML, cross-link only).
8. **Finance case study** (`<FinanceCaseStudy>`):
   - (a) Synthetic MD&A to real filings. Overconfidence, a cross-fitted gain of 0.89 nats, coverage of 0.357, and a
     τ* that overshoots.
   - (b) The fact model by regime. Seen facts are underconfident and unseen facts overconfident, so one temperature
     cannot serve both. Lab view 2.
9. **Where this sits.** Sequence-level calibration (Braverman et al. 2020), verbalized and multiple-choice confidence
   in LLMs (Kadavath et al. 2022), post-training hurting calibration (OpenAI 2023, Fig. 8), abstention (formalRAG).

## 2. Statements and proofs

Notation: a context X (the trigram's last two words, or the decoder's prefix), its next token Y ∈ 𝒱, and the
forecast F = f(X) ∈ Δ^𝒱. The truth is P = P(Y | X), so the forecaster's context is all the truth may condition on.
The recalibration is C = P(Y | F) = E[P | F]. KL and H are in nats.

**Definition 1 (calibration).**
- f is **canonically calibrated** if C = F almost surely: among contexts where f says q, the next token is
  distributed as q.
- Let conf = max_v F_v, and let â be the lowest-id argmax. f is **top-label calibrated** if P(Y = â | conf) = conf.
- The **top-label calibration error** is E|P(Y = â | conf) − conf|.

**Proposition 1.**
- (a) Canonical implies top-label.
- (b) The converse fails. Take one context with truth (0.5, 0.2, 0.3) and forecast (0.5, 0.3, 0.2).
- (c) If f takes distinct values on distinct contexts, then C = P. So canonical calibration holds iff F = P, and
  the reliability term of Thm 1 is the whole excess loss E KL(P ‖ F).

*Proof.*
- (a) C = F gives P(Y = â | F) = F_â = conf. Since conf is a function of F, the tower property gives
  P(Y = â | conf) = E[conf | conf] = conf.
- (b) The model says 0.5 and is right with probability 0.5, but C = P ≠ F.
- (c) Each level set {F = q} is one context, so C = E[P | F] = P. ∎

Tests: `test_canonical_implies_top_label` and `test_distinct_rows_make_calibration_truth`.

**Theorem 1 (three-term decomposition).**

E[−log F_Y] = E H(P) + E KL(P ‖ C) + E KL(C ‖ F), and

E‖F − e_Y‖² = E(1 − ‖P‖²) + E‖C − P‖² + E‖F − C‖².

The three terms are, in order:
- the **irreducible** term (what the text cannot tell any forecaster that sees X);
- the **refinement gap** (what F discards of X);
- the **reliability** term (what recalibrating F recovers).

Each is ≥ 0. The recalibrated forecaster C has the first two terms only, and reliability = 0 iff f is calibrated.
Bröcker (2009, Eq. 10) splits E[score] into E e(C) + E d(F, C). His Section 3 (sufficiency) splits E e(C) further.
For the Brier score with binned binary forecasts this is Murphy (1973), proved on formalML (`uncertainty-quantification`,
Thm 3).

*Proof (log).*
- Conditioned on F, Y has law C, so E[−log F_Y | F] = Σ_v C_v(−log F_v) = H(C) + KL(C ‖ F).
- −log C_v is F-measurable and E[P | F] = C, so E H(C) = E Σ_v P_v(−log C_v) = E[H(P) + KL(P ‖ C)].

*Proof (Brier).* This is Pythagoras twice.
- ‖F − e_Y‖² = ‖F − C‖² + ‖C − e_Y‖² + 2⟨F − C, C − e_Y⟩. The cross term has conditional mean
  ⟨F − C, C − E[e_Y | F]⟩ = 0.
- Do the same with C, P, and conditioning on X, where E[e_Y | X] = P.
- Finally E[‖P − e_Y‖² | X] = 1 − ‖P‖². ∎

**Corollary 1.** For any map g, loss(g ∘ F) ≥ loss(F) − E KL(C ‖ F). On a calibrated forecaster no temperature lowers
the loss.

*Proof.* Let C_g = E[P | g(F)]. Theorem 1 applied to g ∘ F gives loss(g ∘ F) ≥ E H(P) + E KL(P ‖ C_g). Then

E KL(P ‖ C_g) − E KL(P ‖ C) = E Σ_v C_v log(C_v/C_{g,v}) = E KL(C ‖ C_g) ≥ 0,

because log(C/C_g) is F-measurable. ∎

Corpus use: the cross-fitted temperature gain therefore estimates a lower bound on the reliability term (0.89 nats on
EDGAR). Test: `test_any_recalibration_is_bounded_by_reliability`.

**Theorem 2 (two biases of the binned estimator).**

Let ρ(c) = P(Y = â | conf = c). For a partition 𝔅 of [0, 1] into bins, let ECE_𝔅 = Σ_b P(b)·|E[ρ(conf) − conf | b]|
be the population binned error, and let ÊCE be the plug-in estimate on n tokens.
- **(a) Binning hides.** ECE_𝔅 ≤ ECE_𝔅′ ≤ ECE for every refinement 𝔅′ of 𝔅. So ECE₁ ≤ ECE₂ ≤ ECE₄ ≤ … ≤ the true error.
- **(b) Sampling inflates.** Assume the tokens' correctness indicators are independent Bernoulli(ρ_i) given their
  confidences. Then E[ÊCE | confidences] ≥ Σ_b (n_b/n)|ρ̄_b − c̄_b|, the sample's binned error. The inequality is
  strict in every bin where ā_b − c̄_b can take both signs. Under perfect calibration ÊCE is pure noise, of order
  Σ_b (n_b/n)·√(2v_b/(π n_b)) ≈ √(B/n).
- **(c) The square can be debiased exactly.** Per bin,
  E[(ā_b − c̄_b)² − ā_b(1 − ā_b)/(n_b − 1)] = (ρ̄_b − c̄_b)² − Σ_{i∈b}(ρ_i − ρ̄_b)²/(n_b(n_b − 1)).
  For a binned model, every ρ_i in a bin is equal and the estimator is unbiased.

*Proof.*
- (a) The bin b is a function of conf, so E[ρ − conf | b] = E[E[ρ − conf | conf] | b]. Jensen gives
  |E[ρ − conf | b]| ≤ E[|ρ − conf| | b]. The same argument applies within each cell of 𝔅′ ⊂ 𝔅.
- (b) Jensen within the bin: E|ā − c̄| ≥ |Eā − c̄| = |ρ̄ − c̄|.
- (c) Write S = Σ y_i, so that ā = S/n_b, E S = n_b ρ̄, and Var S = Σρ_i(1 − ρ_i) = n_b ρ̄(1 − ρ̄) − D with
  D = Σ(ρ_i − ρ̄)². Then:
  - E(ā − c̄)² = (ρ̄ − c̄)² + [n_b ρ̄(1 − ρ̄) − D]/n_b².
  - Since ā(1 − ā)·n_b/(n_b − 1) is the sample variance s², E s² = ρ̄(1 − ρ̄) + D/(n_b(n_b − 1)).
  - Subtracting E s²/n_b leaves (ρ̄ − c̄)² − D/(n_b(n_b − 1)). ∎

Attribution: the debiased form is due to Bröcker (2012) and Ferro & Fricker (2012), with sample complexity from
Kumar, Liang & Ma (2019, Def. 5.2 and Thm 5.4). Roelofs et al. (2022) document the bias's severity. Part (c)'s
heterogeneity term is our exact statement for an unbinned model.

Tests: `test_binning_hides` (nested B = 1, 2, …, 64 on 50 worlds),
`test_sampling_inflates_and_the_square_debiases` (exact Poisson-binomial expectations on 200 random bins), and
`test_calibrated_twin_floor` (corpus).

**Proposition 2 (temperature scaling).** Let L(β) = (1/N)Σ_tokens[log Σ_v e^{βz_v} − βz_y] with β = 1/τ.
- **(a)** L′(β) = (1/N)Σ(E_{p_β}z − z_y) and L″(β) = (1/N)Σ Var_{p_β}(z) ≥ 0. L is convex, and strictly so unless
  every row is constant on its support. These are the log-partition derivatives of formalStatistics
  `exponential-families` Thm 7.1, with β as the natural parameter and z_y as the sufficient statistic.
- **(b)** A minimizer, if it exists, is unique. It is the β at which the expected logit equals the observed one:
  Σ E_{p_β} z = Σ z_y (the moment matching of formalStatistics Thm 7.4, and the constraint of Guo et al.'s Claim 1,
  2017, Supplement S2).
- **(c)** Every row keeps its argmax, so accuracy is unchanged. But for τ ≠ 1 the order of confidences across rows
  can change. Example: f = (0.5, 0.5, 0) and g = (0.6, 0.2, 0.2) have confidences 0.5 < 0.6. At τ = 2 they become
  0.5 > 0.464.
  - So temperature is not a monotone map of the confidence, and formalRAG `significance-testing-calibration` Thm 2's
    ranking invariance does not apply.
  - On the trigram's rows, Kendall's τ_b between the confidences at τ = 1 and at τ = 2 is 0.860.

*Proof.* (a) Differentiate log Σ_v e^{βz_v}. (b) Strict convexity gives uniqueness, and the first-order condition is
the moment equation. (c) softmax(z/τ) is increasing in each z_v. For the example, √0.6/(√0.6 + 2√0.2) = 0.464. ∎

Fit: Newton's method from β = 1 (formalStatistics `maximum-likelihood` Thm 6) agrees with scipy's bounded search to
10⁻⁶, and |L′(β*)| < 10⁻¹⁰.

**Theorem 3 (where maximum likelihood leaves the temperature).** Suppose a model family has a reparametrization
θ ↦ s_β(θ), differentiable at β = 1, with z_{s_β(θ)} = β·z_θ and Ω(s_β(θ)) = β²Ω(θ). Let θ̂ be a stationary point
of L_train(θ) + λΩ(θ). Then

d/dβ L_train(β z_θ̂) |_{β=1} = −2λΩ(θ̂).

So an unpenalized stationary point has τ*_train = 1. With λΩ(θ̂) > 0, Prop 2's convexity puts τ*_train < 1: the
model is underconfident on its own training data.

*Proof.* φ(β) = L_train(s_β(θ̂)) + λβ²Ω(θ̂) has φ′(1) = ∇(L + λΩ)(θ̂)·∂_β s_β(θ̂) = 0, and the λ-term contributes
2λΩ(θ̂). ∎

Instances:
- **Log-linear bigram, s_β(W) = βW and Ω = ½‖W‖².** The identity holds to 6 decimals at λ = 10⁻⁵, 10⁻⁴, and 10⁻³,
  with τ*_train = 0.871, 0.744, and 0.520.
- **The decoder has no such s_β.** Its final LayerNorm has no gain, its unembedding is tied and scaled by a fixed
  1/√d, and scaling E moves its inputs too. Its training-data τ* is 0.943, so Thm 3 constrains nothing there.

Tests: `test_scale_closed_stationarity` and `test_decoder_has_no_temperature_direction`.

**Proposition 3 (calibrated sets).** If f is canonically calibrated, then every set S(F) chosen from the forecast
satisfies P(Y ∈ S(F)) = E F(S(F)), the expected kept mass. A nucleus of mass p therefore covers Y with probability
≥ p. Contrapositively, coverage ≠ kept mass certifies miscalibration.

*Proof.* P(Y ∈ S | F) = C(S) = F(S). ∎

**Corollary 2 (sampling decalibrates).** If f is calibrated and τ ≠ 1, the sampler F_τ = softmax(log F/τ) is
miscalibrated, with reliability E KL(F ‖ F_τ) > 0 unless F is uniform on its support.

*Proof.* F ↦ F_τ is injective, so σ(F_τ) = σ(F) and C(F_τ) = C(F) = F. ∎

Tests: `test_coverage_equals_kept_mass_when_calibrated` (exact on 30 worlds, plus a label-from-model twin on trigram
rows within 4 SE) and `test_sampling_temperature_decalibrates`.

## 3. Corpus and finance measurements (§§4–8)

- **In distribution, both engines are close to calibrated, but not perfectly.**
  - τ* is 1.003 for the trigram and 0.957 for the decoder.
  - The cross-fitted gains are 0.000016 and 0.0021 nats.
  - ECE₁₅ is 0.0178 against a twin floor of 0.0061 ± 0.0012, about 10 SD above the floor. The miscalibration is
    real but small.
  - The trigram's reliability diagram is overconfident in the middle bins (0.33–0.6) and underconfident above 0.67.
    These are KN's discounts at work. The 7,652 tokens above 0.93 are the templated MD&A openers.
- **Shift can go either way.** On `held` the trigram is slightly overconfident (τ* = 1.055). The decoder turns
  underconfident (accuracy 0.707 against confidence 0.649, τ* = 0.824).
- **EDGAR (finance a).**
  - Both engines are badly overconfident: confidence 0.144 against accuracy 0.059, and 0.430 against 0.195.
  - Most of the EDGAR text is outside the vocabulary: 58% of trigram tokens are `<unk>`.
  - **The trigram's nucleus(0.9) covers 0.357 of the tokens while keeping 0.904 of the mass**, a large Prop 3
    violation. At τ* = 2.37 coverage rises only to 0.416, and confidence collapses to 0.0085, below the accuracy.
  - So the NLL-optimal temperature fixes the log loss (8.796 → 7.911) by flattening every row, and overshoots
    top-label calibration.
  - The twin floor at n = 830 is 0.023, so the EDGAR ECE of 0.085 is 8.5 SD above it.
- **Facts (finance b).**
  - Topic 15's order-5 KN, restricted to the relation's answers (6 sectors or 13 cities), on 111 probes.
  - Seen facts are always right but say 0.785 (τ* → 0).
  - Unseen facts are right 0.0625 of the time but say 0.303 (τ* → ∞: the uniform forecast's loss, 2.017, is the
    floor).
  - The once-seen regime has an interior τ* = 0.260.
  - One pooled τ* = 1.18 recovers 0.010 nats. A temperature per regime recovers 0.504 (1.623 → 1.119).
  - The fix is to condition on the training count, which a single temperature cannot see.

## 4. The lab (`CalibrationLab`, `data/language-model-calibration-lab.json`)

Payload: `taus` (11 points, 10^{k/20} for k = −6 … 14 step 2, so 0.501 … 5.01 with τ = 1 included), `bins`
(1, 5, 10, 15, 30, 60), and `engines.{trigram,decoder}.sets.{continuation,held,edgar}`. Each set has `n`, `tau_star`,
`nll_star`, `gain`, and `points[τ]`, where each point holds:
- `nll`, `acc`, and `conf`;
- `cover` and `kept` for nucleus(0.9);
- `ece[B]`, `twin[B]`, and `twin_sd[B]`;
- `bins[B]`, a list of [count, mean confidence, accuracy] per nonempty bin.

`facts` holds `taus` (Topic 15's 49 points) and, per regime and pooled, `n`, `acc`, `tau_star`, and per-τ `conf`,
`p_correct`, and `nll`, plus `count_aware`.

- **View 1, Reliability.**
  - Controls: an engine toggle, a test-set toggle, a τ slider over the 11 points (τ* shown as a readout, not on the
    slider), and a bin-count selector.
  - Left panel: a reliability diagram with circles sized by count, the diagonal, and gap ticks.
  - Right panel: ECE against B on a log axis, the engine's curve against the twin's mean ± 2 SD band, with the
    current B marked.
  - Readouts: NLL, accuracy, confidence, ECE and twin floor, nucleus(0.9) coverage against kept mass, τ*, and the
    cross-fitted gain.
  - The lessons: binning hides and sampling inflates (watch the twin curve rise with B); temperature moves the dots
    but never the accuracy; on EDGAR no τ brings coverage near 0.9.
- **View 2, Facts by regime.**
  - Controls: a τ slider over the 49 fact temperatures.
  - Charts: per regime, mean confidence against accuracy (a flat line), and NLL against τ on a log axis with each
    regime's τ* marked.
  - Readout: the pooled, per-regime, and τ = 1 NLLs.
  - The lesson: the seen curve falls to the left and the unseen curve falls to the right, so no single τ is best
    for both.

Every printed number is a payload value, already rounded to 3 decimals in `viz_payload()`.

## 5. rigorFlag (draft)

> Every corpus number here is **top-label** calibration estimated by equal-width binning. Theorem 2 shows that
> estimator is biased in both directions. Canonical calibration over a 3,427-word vocabulary cannot be estimated from
> 23,000 tokens; we demonstrate it only on finite worlds where every expectation is an exact sum. Theorem 2(b) and the
> calibrated twin treat a test set's tokens as independent given their confidences, which consecutive tokens of real
> text are not, so the twin floor is a floor under that assumption. The in-distribution test set is the training
> generator run on, a synthetic best case. The near-calibration of Kneser–Ney there is an empirical finding, not a
> theorem about its discounts. EDGAR is 830 tokens, most of them unknown words to both engines. The decoder is scored
> on its own 512-token event space, with `<unk>` as an outcome and only through 32 positions, so its accuracy is not
> comparable to the trigram's. Theorem 3 constrains only the training-data temperature; nothing here explains where
> held-out temperatures land.

## 6. Notebook cell map (planned, ~19 cells)

1. Title and three claims (md).
2. Setup: import `language_model_calibration as lc`, with `S = lc.summary()` (code).
3. §1–2: Def 1 and Prop 1 (md).
4. Prop 1 tests and the counterexample (code).
5. Thm 1 and Cor 1 (md).
6. Decomposition of one random world, printed term by term (code).
7. Thm 2 (md).
8. Exact bin expectations, plus twin against actual ECE vs B for the trigram (code).
9. Prop 2 (md).
10. τ* table with Newton vs scipy (code).
11. Thm 3 (md).
12. Log-linear table and the decoder's τ* (code).
13. Prop 3 and Cor 2 (md).
14. Coverage table (code).
15. Finance (md).
16. EDGAR and fact regimes (code).
17. The lab payload (md).
18. Payload check (code).
19. Where this sits (md).

## 7. Cross-site and scope

**Scope statement.**
- formalML `uncertainty-quantification` (published) owns the binary apparatus:
  - Dawid's definition, binned and conditional calibration;
  - ECE and MCE;
  - strict properness of Brier and log loss (its Thm 2) and Murphy's binned Brier decomposition (its Thm 3);
  - the remark that temperature preserves accuracy.

  This page **cites all of those** and proves what they leave open:
  - multiclass, population-level, log-loss calibration (Thm 1's three terms and Cor 1's bound);
  - both directions of ECE's bias, and the exact debiased square (Thm 2);
  - temperature's convexity, moment matching, and failure to preserve ranking (Prop 2);
  - where maximum likelihood leaves τ (Thm 3);
  - what calibration buys a truncation set (Prop 3).
- formalRAG `significance-testing-calibration` defines the reliability diagram, ECE, and MCE for retrieval scores and
  proves monotone recalibration preserves ranking. It gets a Connection, and Prop 2(c) contrasts with it.
- formalRAG `selective-generation-abstention` owns Chow's rule, risk–coverage, and conformal selective risk. It gets
  a Connection, with no abstention content here.

| Field | Entries (all verified `published` on GitHub `main`, 2026-10-25) |
|---|---|
| `formalmlPrereqs` | `uncertainty-quantification` (calibration defs, strict properness, Murphy), `kl-divergence` (KL, Gibbs) |
| `formalmlConnections` | `conformal-prediction` (marginal coverage without calibration; Prop 3's contrast) |
| `formalstatisticsPrereqs` | `exponential-families` (Thm 7.1 log-partition moments, Thm 7.4 moment matching), `maximum-likelihood` (score equation, Thm 6 Newton) |
| `formalragConnections` | `significance-testing-calibration` (Thm 2 ranking invariance vs Prop 2(c)), `selective-generation-abstention` (what to do with a calibrated confidence) |

## 8. References

All titles were checked against arXiv meta tags or Crossref.

- Bröcker, Jochen. 2009. "Reliability, Sufficiency, and the Decomposition of Proper Scores." *Quarterly Journal of
  the Royal Meteorological Society* 135 (643): 1512–19. https://doi.org/10.1002/qj.456 (read on arXiv 0806.0813:
  Eqs. 6–13, Section 3).
- Bröcker, Jochen. 2012. "Estimating Reliability and Resolution of Probability Forecasts through Decomposition of the
  Empirical Score." *Climate Dynamics* 39: 655–67. https://doi.org/10.1007/s00382-011-1191-1
- Braverman, Mark, Xinyi Chen, Sham Kakade, Karthik Narasimhan, Cyril Zhang, and Yi Zhang. 2020. "Calibration,
  Entropy Rates, and Memory in Language Models." *ICML*, PMLR 119. https://proceedings.mlr.press/v119/braverman20a.html
- DeGroot, Morris H., and Stephen E. Fienberg. 1983. "The Comparison and Evaluation of Forecasters." *The
  Statistician* 32 (1/2): 12–22. https://doi.org/10.2307/2987588
- Desai, Shrey, and Greg Durrett. 2020. "Calibration of Pre-trained Transformers." *EMNLP*, 295–302.
  https://doi.org/10.18653/v1/2020.emnlp-main.21
- Ferro, Christopher A. T., and Thomas E. Fricker. 2012. "A Bias-Corrected Decomposition of the Brier Score."
  *QJRMS* 138 (668): 1954–60. https://doi.org/10.1002/qj.1924
- Gneiting, Tilmann, and Adrian E. Raftery. 2007. "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA*
  102 (477): 359–78. https://doi.org/10.1198/016214506000001437
- Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. "On Calibration of Modern Neural Networks."
  *ICML*. https://arxiv.org/abs/1706.04599 (ECE with M = 15; temperature scaling; Supplement S2, Claim 1).
- Kadavath, Saurav, et al. 2022. "Language Models (Mostly) Know What They Know." arXiv:2207.05221.
- Kumar, Ananya, Percy Liang, and Tengyu Ma. 2019. "Verified Uncertainty Calibration." *NeurIPS*.
  https://arxiv.org/abs/1909.10155 (Defs. 5.1–5.2, Thms 5.3–5.4).
- Murphy, Allan H. 1973. "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology* 12 (4):
  595–600. https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2
- Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht. 2015. "Obtaining Well Calibrated Probabilities
  Using Bayesian Binning." *AAAI* 29 (1). https://doi.org/10.1609/aaai.v29i1.9602
- OpenAI. 2023. "GPT-4 Technical Report." arXiv:2303.08774 (Fig. 8: post-training reduces calibration on MMLU).
- Ovadia, Yaniv, et al. 2019. "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty under Dataset
  Shift." *NeurIPS*. https://arxiv.org/abs/1906.02530
- Roelofs, Rebecca, Nicholas Cain, Jonathon Shlens, and Michael C. Mozer. 2022. "Mitigating Bias in Calibration Error
  Estimation." *AISTATS*. https://arxiv.org/abs/2012.08668 (bias most severe for perfectly calibrated models).
- Vaicenavicius, Juozas, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas B. Schön. 2019.
  "Evaluating Model Calibration in Classification." *AISTATS*. https://arxiv.org/abs/1902.06977 (Eq. 1: canonical
  calibration).

## 9. Open questions for the checkpoint

1. **Difficulty.** `advanced` is proposed: Thm 3 and Thm 2(c) are the hardest arguments on the site so far. Or
   `intermediate`, matching Topic 15?
2. **Theorem count.** Six results plus two corollaries. Is §6's Thm 3 worth its length, or should it shrink to a
   proposition with the log-linear table?
3. **EDGAR's `<unk>` share** (58%) dominates its numbers. Keep it as the honest real-text test, or also report EDGAR
   restricted to in-vocabulary tokens? That would add a test set to the lab.
