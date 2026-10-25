# Brief — `scaling-laws`

**Neural Scaling Laws and Compute-Optimal Training.** Roadmap #13. Prerequisites: `transformer-block-residual-stream`
(#11, PR #20) and `perplexity-cross-entropy` (#4), both published. This topic is a prerequisite of nothing in the
graph. It is not on the §6 overlap watch-list.

**Phase A** (this brief and the verified notebook), 2026-10-24:
- The harness passes 21/21 under numpy==2.5.4 and scipy==1.18.1 in ≈ 34 s, also with warnings promoted to errors.
- The narrative notebook (29 cells) executes with exit 0 in ≈ 39 s, including kernel start, and is committed without
  outputs.
- The lab's data file `src/components/viz/data/scaling-laws-lab.json` is 11.6 kB, pinned by
  `test_viz_mirror_matches_payload`.

**Scope decisions** (Jonathan, 2026-10-24, on Claude's recommendations):
- **Engine: both.** A solvable power-law model whose exponents are proved and then recovered by fitting, plus a toy-transformer
  IsoFLOP grid on the finance corpus as the "does language follow it" check.
- **The full theory spine:**
  - data scaling from Zipf;
  - model-size scaling;
  - 6ND and compute-optimal allocation;
  - Kaplan versus Chinchilla.
- **Finance thread: data-constrained.** A filing corpus is finite, so the case study uses Muennighoff's repetition law and the
  data-capped optimum.
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–12.
- **Difficulty:** `advanced`, domain `attention-transformers`, `financeCaseStudy: true`.

**The spine.** On log–log axes loss against N, D, or C is a line bending to a floor. Where do the exponents come from,
and how should a budget be split?
- **Data.** A memorizer errs when the next type is new. Under the corpus's own Pitman–Yor generator that probability
  has a closed form, which falls as n^{−(1−d)} = n^{−0.4} (Thm 1). The corpus's text shows slope −0.406526.
- **Model size.** In a linear model with spectrum k^{−a} and target power k^{−b}, a size-N model keeps a tail
  ≍ N^{−(b−1)} (Thm 2).
- **Data in the same model.** Gradient flow leaves ≍ t^{−(b−1)/a} (Thm 3). SGD's expected loss is exact for Gaussian
  features (Prop 1).
- **Compute.** One training step costs exactly 6N + 12Lnd FLOPs per token (Prop 2), counted by instrumenting the arrays.
- **Allocation.** Minimizing E + A/N^α + B/D^β on 6ND = C gives N* ∝ C^{β/(α+β)} (Thm 4). In the solvable model that is
  C^{1/(1+a)} = C^{0.4}. Chinchilla's three approaches recover 0.402771, 0.408160, and 0.404644.
- **Units.** Read in Kaplan's non-embedding units, the same law has local exponent ρa/(ρa + 1 − a) (Prop 3). This gives
  0.705 below 1e18 FLOPs from Besiroglu's a = 0.513: Kaplan's 0.73, explained.
- **Finance.** The toy model shrugs off 4 epochs and is hurt by 64. At the compute of BloombergGPT's run, a 709B-token cap
  barely binds.

**Changes from the approved plan, each forced by a spike or the harness:**
1. **Learned absolute positions, not RoPE.** Topic 10's `rotate` writes into a fresh `np.empty`, which drops float32 to
   float64. It also detaches the arrays from the FLOP instrument, so Prop 2's count would have missed QKᵀ. Learned
   positions (Topic 12's table) keep training float32 and every matmul counted. The step got faster: the grid runs in
   ≈ 31 s.
2. **Peak learning rate 1.28/d with 10% warmup.** Undertraining dominated at a fixed rate: at lr 0.01, d = 32 for 400
   steps reached 2.14 nats, against 1.70 at 0.04. A sweep put the best rate at about 1.28/d (0.16 at d = 8, 0.04 at
   d = 32). That is a 1/width rule, the μP direction for Adam. Warmup halved the seed spread, and clipping changed
   nothing. The residual seed spread of one point is 0.010399 nats (`test_seed_spread`).
3. **An explicit width list per budget.** With a step-range rule, the lowest budget's minimum sat at its smallest width
   (d = 8). Adding d = 6 made it interior: 2.982 at d = 6, 2.726 at d = 8, 2.889 at d = 12.
4. **Prop 3 is new.** The plan said "toggle embedding counting". On the toy grid that toggle moves Approach 2's a only
   from 0.629944 to 0.706395, because Kaplan's compute moves with the count. The closed form explains why. Feeding a
   *published* fit through a GPT-2-sized family reproduces Pearce and Song's reconciliation exactly: the finite-difference
   local exponent equals ρa/(ρa + 1 − a) to 2e-3 at every budget.
5. **Hutter's §6 caveat.** Hutter (2021, §6) says the CRP gives β = 1 and the two-parameter process "also only leads to
   β = 1". That covers his continuous-feature extension. For the predictive missing mass, Thm 1 gives β = 1 − d exactly.
   At d = 0 (the Dirichlet case, `test_dirichlet_case`) the formula is θ/(θ + n), so β = 1 there, as he says. The page
   states the difference and does not lean on his §6.
6. **The printed Chinchilla fit is not the one that produced the paper's 40B.** At Gopher's 5.76e23 FLOPs, the rounded
   parameters (Eq. 10) give N* = 32.2B, and the unrounded ones (Besiroglu's Eq. 4) give 40.3B, the paper's §3.3 figure.
   Besiroglu's refit gives 72.2B, beside Approaches 1–2 (67B and 63B). This is pinned in `test_published_fits`.
7. **Eq. 2 recovers the exponents but misses the surface.** The fit's log-residual is 0.128376 RMS and 0.400361 at its
   worst (N = 8, D = 64). The fit overestimates there: a size-N model's data term sums only its N kept coordinates.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Neural Scaling Laws and Compute-Optimal Training |
| `{{SUBTITLE}}` | Loss falls as a power of parameters, data, and compute: in two solvable models the exponents are theorems — Zipf's tail for data, a spectrum's tail for size — and a Lagrange multiplier turns the fitted law into the compute-optimal split, whose exponent depends on what you count as a parameter |
| `{{SLUG}}` | `scaling-laws` (= graph node id) |
| `{{DOMAIN}}` | `attention-transformers` |
| `{{DIFFICULTY}}` | `advanced` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[transformer-block-residual-stream, perplexity-cross-entropy]` |
| `{{FORMALML_PREREQS}}` | `lagrangian-duality`, `gradient-descent`; connections `double-descent`, `bayesian-nonparametrics` |
| `{{FORMALRAG_PREREQS}}` | none |
| `{{STATS_CALC_PREREQS}}` | Stats `linear-regression`, `method-of-moments`; Stats connection `bootstrap`; Calc `series-convergence`, `improper-integrals` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/transformer-block-residual-stream/`, `notebooks/contextual-embeddings-masked-lm/`, `notebooks/zipf-heaps-vocabulary-statistics/`, `notebooks/finance-corpus/` |

- **Tags:** `scaling-laws`, `compute-optimal`, `chinchilla`, `power-laws`, `pitman-yor`, `missing-mass`, `sgd`,
  `flops`, `data-constrained`.
- `estimatedReadTime` ≈ 45.
- **In-site `connections`:** none, since no published topic depends on this one. Topic 1 (Heaps, the PY sampler) and
  Topic 3 (Good–Turing) are referenced in the prose. Their edges reach #13 through #4.
- **Graph:** both incoming edges (#11 → #13, #4 → #13) exist; there is no outgoing edge.

**Scope.**
- **No sibling owns scaling laws.** formalML `double-descent` owns the interpolation threshold and sample-wise and model-wise
  risk curves (Hastie et al.'s Thm 2) in least squares. It says nothing about power-law exponents in N or D, or about
  compute. We link it as the regime our monotone curves leave out.
- **formalML `lagrangian-duality`** owns equality multipliers and the sensitivity theorem (its Thm 7, ∂p*/∂v = −ν*). Thm 4's
  multiplier is that shadow price.
- **formalStatistics `method-of-moments`** owns Huber M-estimation (§15.11, its Thm 11), which Approach 3 uses.
- **Topic 1** owns Heaps' law, the PY sampler (`zhv.pitman_yor_crp`), and the closed form for E K_n
  (`zhv.py_expected_types`). Thm 1 is one line on top of it.
- **Topic 3** owns Good–Turing, the frequentist estimator of the same missing mass. Thm 1 is its Bayesian twin under the
  generator.
- **Topic 11** owns the pre-LN block. The toy decoder is that block, made causal.
- **Out of scope:**
  - emergence and downstream-task scaling;
  - scaling of inference compute;
  - mixture-of-experts;
  - μP and hyperparameter transfer (one sentence, as the reason for the 1/d rate);
  - data pruning;
  - broken power laws;
  - the variance-limited regimes of Bahri et al. (named, not covered).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | the toy grid's three IsoFLOP valleys on one log–log plot, beside the solvable model's lines | — | cells 0–1 |
| 1 | Data scaling from Zipf | the corpus's new-type rate on log–log axes, slope −0.406526 past 10k tokens; the memorizer as the simplest learner | Def 1 (memorizer, missing mass); **Thm 1**; remark (Hutter, d = 0) | cells 2–5 |
| 2 | Model-size scaling | the spectrum k^{−1.5} and the target power k^{−2.5} as bars; a size-N model keeps the first N | Def 2 (solvable model); **Thm 2** | cells 6–7 |
| 3 | Data scaling and SGD | gradient flow learns the big-eigenvalue directions first: the frontier k ≈ t^{1/a} moving right | **Thm 3**; **Prop 1** (exact SGD) | cells 8–9 |
| 4 | Counting compute | one block's matmuls as boxes, forward ×1 and backward ×2 | **Prop 2** (6N + 12Lnd); Kaplan's Eq. 2.2 beside it | cells 10–11 |
| 5 | Compute-optimal allocation | an IsoFLOP valley: big models starve for data, small ones saturate | **Thm 4**; the multiplier; Approaches 1–3; Eq. 2 misfit; published fits | cells 12–17 |
| 6 | Toy transformers on filings | the three valleys and their parabolas | Approach 2 and 3 on the grid; the seed spread | cells 18–19 |
| 7 | Kaplan against Chinchilla | the embedding share falling from 0.993 to 0.009 as C grows | **Prop 3**; its limit 3a/(1+2a) | cells 20–21 |
| 8 | Finance: a finite corpus | the loss after 1, 4, 16, 64 epochs at fixed compute | Muennighoff's D′; the capped optimum | `<FinanceCaseStudy>`, cells 22–25 |
| 9 | What the laws are not | — | rigorFlag in prose | — |

## 2. Statements and proofs

The notebook's markdown cells carry the full statements and proofs; the MDX copies them, changing only formatting.

- **Def 1.** A memorizer's error after n tokens: the probability that token n + 1 is a type it has not seen.
- **Thm 1** (missing mass under PY(d, θ), 0 ≤ d < 1, θ > 0). The probability is
  (θ + d·E K_n)/(θ + n) = Γ(θ+1)Γ(θ+d+n)/(Γ(θ+d)Γ(θ+n+1)) ∼ Γ(θ+1)/Γ(θ+d)·n^{−(1−d)}.
  - **Proof:** linearity of the seating rule, the telescoping recursion E K_{n+1} + θ/d = (1 + d/(θ+n))(E K_n + θ/d),
    and Gamma ratios.
  - **Remark:** the PY measure's ranked frequencies decay as i^{−1/d}, so Zipf's s = 1/d. Hutter's §3 rate n^{−(1−1/s)} is
    the same exponent. At d = 0 it is θ/(θ + n).
  - **Sources:** Pitman (2006, §3.2, Eq. 3.9) for the rule; §3.3 for E K_n's asymptote.
- **Def 2** (solvable model).
  - x ~ N(0, Λ) on R^M, with λ_k = k^{−a}, and y = ⟨w*, x⟩ with λ_k w*_k² = k^{−b}.
  - A size-N model fits the first N coordinates.
  - Loss: Σ_{k≤N} λ_k(w_k − w*_k)² + Σ_{k>N} k^{−b}.
  - Constants: a = 1.5, b = 2.5, M = 8192, η = 0.25 (η·tr Λ ≈ 0.65).
- **Thm 2** (integral test). ∫_{N+1}^{M+1} x^{−b} ≤ Σ_{N+1}^{M} k^{−b} ≤ ∫_N^M x^{−b}. Proof: monotone summand.
- **Thm 3** (gradient flow). With s = (b − 1)/a, Σ_k k^{−b} e^{−2tk^{−a}} = Γ(s)/a·(2t)^{−s}(1 + o(1)).
  - **Proof:** substitute u = 2tx^{−a}. The sum is within the summand's maximum, (b/(2ta))^{b/a}e^{−b/a} = O(t^{−b/a}), of
    the integral over [1, ∞). The piece over (0, 1) is O(e^{−2t}).
  - **Harness:** the bracket against the exact incomplete-gamma integral on 20 random (a, b, t).
- **Prop 1** (exact one-pass SGD). The update is
  S_k ← (1 − 2ηλ_k + 2η²λ_k²)S_k + η²λ_k(Σ_j λ_j S_j + tail(N)).
  - **Proof:** Isserlis, E[xxᵀAxxᵀ] = 2ΛAΛ + tr(ΛA)Λ. The tail coordinates are independent noise. Only the diagonal enters.
  - **Harness:** against 400 Monte Carlo runs within 3 SE.
  - **Cited, not proved:** the SGD-noise phases of Paquette et al. (2024) and Bordelon et al. (2024). Our constant step sits
    in their regime where SGD noise only rescales the N-term (η small against 1/tr Λ). The harness's agreement of
    α̂ = 1.523121 with b − 1 = 1.5 is the backing.
- **Prop 2** (FLOPs). Per token, a training step costs 6N + 12Lnd, where N = 12Ld² + Vd is the weight count with the tied
  embedding.
  - **Proof:** forward 2 per weight; backward 4, one product for the input gradient and one for the weight gradient.
    QKᵀ and AV cost 2nd each, times 3.
  - **Harness:** `_Counted`, an ndarray subclass, counts 2k per output entry of every matmul in the real
    forward/backward. The counts are equal to the integer at all seven widths.
  - **Note:** Kaplan's Table 1 / Eq. 2.2 count attention as 2·n_layer·n_ctx·d_attn forward (QKᵀ only) and leave the
    de-embedding out of N. Porian's "last-layer FLOPs" is the second of these.
- **Thm 4** (compute-optimal). For L = E + AN^{−α} + BD^{−β} on 6ND = C:
  - N* = G(C/6)^{β/(α+β)} and D* = G^{−1}(C/6)^{α/(α+β)}, with G = (αA/(βB))^{1/(α+β)};
  - L* = E + K·C^{−αβ/(α+β)} and ν = −dL*/dC.
  - **Proof:** substitute D = C/6N. The function is strictly convex in log N, and the first-order condition is αAN^{−α} =
    βBD^{−β}.
  - **Harness:** against a numeric argmin, with the multiplier checked by finite difference on 30 random fits.
  - **Corollary for Def 2:** α = b − 1 and β = (b − 1)/a, so a* = 1/(1 + a), independent of b.
- **Prop 3** (Kaplan's units). If N* ∝ C^a and ρ = d log N_ne / d log N, then d log N_ne* / d log C_ne = ρa/(ρa + 1 − a).
  - **Proof:** chain rule on C_ne = 6N_ne*D*.
  - **Fixed aspect ratio:** embeddings grow as N_ne^{1/3}, so ρ = (1 + x)/(1 + x/3) ∈ [1, 3) and the limit is 3a/(1 + 2a)
    (0.759341 at Besiroglu's a).
  - **Family:** V = 50257, n_ctx = 1024, d_model = 64·n_layer.

## 3. Numbers the page quotes (harness, ≥ 6 decimals)

| Quantity | Value | Test |
|---|---|---|
| PY missing mass at n = 1000: exact / simulated ± SE | 0.208700 / 0.194000 ± 0.0125 | `test_missing_mass_exact` |
| corpus new-type slope past 10k tokens (735,618 tokens, 13,765 types) | −0.406526 | `test_corpus_learning_curve` |
| solvable model α̂, β̂ (predicted 1.5, 1.0) | 1.523121, 1.035215 | `test_linear_scaling_exponents` |
| a: Approaches 1, 2, 3 (predicted 0.4) | 0.402771, 0.408160, 0.404644 | same |
| Eq. 2 log-residual RMS / max | 0.128376 / 0.400361 | `test_additive_form_is_approximate` |
| attention / 6N, d = 6 … 48 | 0.195 … 0.077 | `test_flop_count` |
| published a: Hoffmann rounded / unrounded / Besiroglu | 0.451613 / 0.456497 / 0.512612 | `test_published_fits` |
| N* at Gopher's 5.76e23: same three | 32.2B / 40.3B / 72.2B | same |
| tokens per parameter: same three | 92.647 / 59.079 / 18.391 | same |
| toy N* per budget (1e10, 2.5e10, 6.3e10) | 7,128 / 10,831 / 22,716 | `test_toy_isoflop_profiles` |
| toy a: Approach 2 (Chinchilla count) / Kaplan count / Approach 3 | 0.629944 / 0.706395 / 0.693986 | `test_kaplan_counting_shift` |
| toy Approach 3 fit E, α, β | 0.772357, 0.402919, 0.913749 | — |
| one point's seed spread (sd, 3 seeds) | 0.010399 | `test_seed_spread` |
| Kaplan-units exponent below 1e18 FLOPs / limit | 0.705216 / 0.759341 | `test_kaplan_counting_shift` |
| repetition: loss at 1, 4, 16, 64 epochs | 2.297965 / 2.314821 / 2.357148 / 2.826828 | `test_repetition` |
| toy fit's prediction at Muennighoff's D′ | 2.3010 / 2.3259 / 2.4712 / 3.2362 | — |
| BloombergGPT compute, U = 709B: free N* / capped N* / epochs | 38.97B / 39.10B / 1.038 | `test_data_cap_moves_the_optimum` |

Page precision: quote losses to 3 decimals from the payload (`_r`). The toy repetition is gentler than Muennighoff's
law predicts (2.357 observed at 16 epochs against 2.471 at D′). The synthetic "unique" text is template-generated and
already redundant, so a repeated window costs less. That is a finding for the page, not a contradiction.

## 4. The lab (`ScalingLawsLab`, `data/scaling-laws-lab.json`)

Three views and a strip. All numbers come from the payload.
- **(a) Log–log curves.** The solvable model's L(N, D) for the 15 sizes against the 49 checkpoints, a toggle between
  "vs D at fixed N" and "vs N at fixed D", and the Approach-3 fit overlaid as dashed lines. The residual
  is visible at small N. *Learn:* each size's curve bends into its own tail(N) floor, and the bend's onset moves right
  with N.
- **(b) IsoFLOP explorer.** A slider over the 25 budgets in `linear.C` shows the valley L(N, C/N), its minimum (`star1`),
  the parabola vertex (`star2`), and the frontier line with slope a. A second panel shows the toy grid's three valleys
  and their N*, with a toggle **Chinchilla count ↔ Kaplan count** that redraws the toy's N* in N_ne and the
  `kaplan.local` curve against C (the published-fit family), with Prop 3's prediction dotted. *Learn:* the same
  optimum reads as exponent 0.51 or 0.73 depending on what you count.
- **(c) Data-constrained.** The repetition rows as bars (observed against the toy fit at D′), and a slider over
  `cap.U` that shows the capped N* and epochs at BloombergGPT's compute against the free optimum. *Learn:* a few epochs
  are nearly free, and the cap binds only when U is far below D*.
- **Strip (in §1).** The memorizer: `memorizer.rate` against Thm 1's `theory` on log–log axes.

Mobile: every view stacks; the toggles are flex-wrap rows (CLAUDE.md's 375 px rules).

## 5. rigorFlag (draft)

> The exponents here are theorems only inside two toy models: a memorizer on Pitman–Yor text (Thm 1) and a linear
> model with a power-law spectrum (Thms 2–3). For real language models, the power-law form E + A/N^α + B/D^β is a fit:
> it misses even our solvable model's loss surface by 13% RMS while recovering its exponents.
>
> The fits disagree with one another:
> - Hoffmann et al.'s three approaches give a = 0.46 to 0.50.
> - Their printed parameters give 32B at Gopher's budget, where their text says 40B.
> - Besiroglu et al.'s refit moves a to 0.51.
> - Kaplan et al.'s 0.73 comes from counting non-embedding parameters at small scale.
>
> Our toy grid spans less than a decade of compute with one seed per point (spread ≈ 0.01 nats). Its a = 0.63 cannot
> separate Kaplan from Chinchilla. 6ND drops attention's 12Lnd, 8–20% of compute at our widths. Muennighoff's
> repetition law is itself a fit, at a scale and on text unlike ours. Extrapolation beyond the fitted range is a bet,
> not a theorem.

## 6. Notebook cell map

29 cells: 0 title · 1 headline · 2–5 §1 (Thm 1, Dirichlet case, corpus curve) · 6–7 §2 (Thm 2) · 8–9 §3 (Thm 3,
Prop 1) · 10–11 §4 (Prop 2) · 12–15 §5 (Thm 4, Approaches 1–3, Eq. 2 misfit) · 16–17 published fits · 18–19 §6 toy grid ·
20–21 §7 Prop 3 · 22–25 §8 finance · 26–28 payload and harness.

## 7. Cross-site (verified `status: published` on each sibling's GitHub `main`, 2026-10-24)

- `formalmlPrereqs`:
  - `lagrangian-duality`: Thm 4's multiplier is its sensitivity theorem's shadow price (∂p*/∂v = −ν*) for the budget
    6ND = C.
  - `gradient-descent`: Thm 3 and Prop 1 analyze gradient flow and one-pass SGD on a quadratic, whose convergence
    rates that topic proves.
- `formalmlConnections`:
  - `double-descent`: the sample-wise and model-wise risk curves of least squares, including the interpolation peak our
    monotone power laws leave out.
  - `bayesian-nonparametrics`: the Dirichlet process, whose E K_n ≈ θ log n and missing mass θ/(θ + n) are Thm 1's
    d = 0 case.
- `formalstatisticsPrereqs`:
  - `linear-regression`: every exponent on the page is an OLS slope on log–log axes.
  - `method-of-moments`: Approach 3 is a Huber M-estimator (its §15.11 and Thm 11).
- `formalstatisticsConnections`:
  - `bootstrap`: Hoffmann et al.'s exponent intervals are bootstrap percentiles, and Besiroglu et al.'s are bootstrap SEs.
- `formalcalculusPrereqs`:
  - `series-convergence`: Thm 2 is its integral test (Thm 8) with the remainder written out.
  - `improper-integrals`: Thms 1 and 3 evaluate Gamma-function integrals (its Def 4, Thms 4–5).
- **Not cited:** formalCalculus `convex-optimization` (draft).

## 8. References (Chicago 17 N-B; all verified; full entries in the MDX frontmatter)

- **Core:** Kaplan et al. 2020 (arXiv 2001.08361); Hoffmann et al. 2022 (NeurIPS 35: 30016–30, printed as "An Empirical
  Analysis of Compute-Optimal Large Language Model Training"; note the arXiv title); Hestness et al. 2017
  (arXiv 1712.00409); Hutter 2021 (arXiv 2102.04074); Bahri et al. 2024 (PNAS 121(27), doi 10.1073/pnas.2311878121);
  Maloney, Roberts, and Sully 2022 (arXiv 2210.16859).
- **Corrections and replications:** Paquette et al. 2024 (NeurIPS 37: 16459–537); Bordelon, Atanasov, and Pehlevan 2024
  (PMLR 235: 4345–82); Besiroglu et al. 2024 (arXiv 2404.10102); Pearce and Song 2024 (TMLR); Porian et al. 2024
  (NeurIPS 37: 100535–70).
- **Data-constrained and finance:** Muennighoff et al. 2023 (NeurIPS 36: 50358–76; NeurIPS author order: Tazi before
  Piktus); Wu et al. 2023, BloombergGPT (arXiv 2303.17564; "İrsoy").
- **Classical:** Pitman and Yor 1997 (Ann. Probab. 25(2), doi 10.1214/aop/1024404422); Pitman 2006 (LNM 1875,
  doi 10.1007/b11601500); Good 1953 (Biometrika 40(3–4), doi 10.1093/biomet/40.3-4.237); Huber 1964 (Ann. Math. Statist.
  35(1), doi 10.1214/aoms/1177703732).

## 9. Open questions for the checkpoint

1. **Kaplan view's family.** The family is GPT-2's V and n_ctx with aspect 64. Pearce and Song simulate Kaplan's actual
   sizes (their 0.78). Our 0.705 below 1e18 FLOPs is a different, stated family. Should we keep ours or match theirs?
2. **BloombergGPT framing.** The case study uses the published counts only (709B available, 50.6B on 569B). It names
   the model as the motivating example. Is that within "framing only, not product news"?
