# Brief — `sampling-temperature-nucleus`

**Sampling: Temperature, Top-k, and Nucleus.** Roadmap #15. Its prerequisite is `beam-search-decoding` (#14), which
is published. It is a prerequisite of `language-model-calibration` (#16). It is not on the §6 overlap watch-list.

**Phase A** (2026-10-24; outline and statements approved by Jonathan):
- **Harness:** 23/23 pass under numpy==2.5.4 and scipy==1.18.1 in ≈ 34 s, also with warnings promoted to errors.
- **Notebook:** 21 cells; executes with exit 0 in ≈ 40 s, including kernel start, and is committed without outputs.
- **Lab data:** `src/components/viz/data/sampling-temperature-nucleus-lab.json` is 90 kB, pinned by
  `test_viz_mirror_matches_payload`.
- Numbers marked *(spike)* below are superseded by §3a.

**Changes from the approved statements, each forced by the harness:**
1. **Thm 3(b) is now an identity:** log(q_τ/π_τ) = log 𝒵_τ + (1/τ − 1)·Σ_t H_{1/τ}(p(· | h_t)). Since
   log Z_τ(h) = (1 − α)H_α, local tempering reweights by the accumulated **Rényi** entropy of order 1/τ. This replaces
   the qualitative "upweights high-entropy prefixes."
2. **Typical sampling is not monotone.** At typical mass 0.2 it drops each distribution's head, so the trigram's samples
   cost 1.617 nats per word against 1.464 at mass 0.5. This is pinned as a claim. Every other rule's NLL rises, and every
   rule's KL falls, as its parameter loosens.
3. **All rules share one frontier.** Every truncation setting lies within 0.1 nats of the temperature curve's oracle NLL
   at its own model NLL, on both engines (`test_rules_share_one_frontier`). τ = 0.4 was added to the grid so the hardest
   cuts fall inside the curve's range, giving 25 settings.
4. **Frontier axes.** x is model NLL and y is oracle NLL, both pooled per word with `</s>` included. Distinct-2 depends
   on sample size, so it is a readout only.
   - The **decoder gets no held-out marker**: the held-out sentences it can score (all words in its 509-word vocabulary,
     within 32 positions) are a templated subset with NLL 0.445.
   - The decoder's exact modes (≈ 6 s) are left to #14, so only the trigram's mode is marked.
5. **Prop 4's statistics exclude the 11 prompt rows whose context the trigram never saw**, since those rows are all
   backoff. 177 rows remain.
6. **The τ grid is 10^{k/20}, k = −28 … 20** (0.040 to 10), so τ = 1 lies on it exactly. KN rows can tie at the top,
   so Thm 2(c)'s limit is tested as uniform on the argmax set.

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

| Quantity | Value | Test |
|---|---|---|
| trigram per-word NLL: mode / pure samples / held-out text | 0.834742 / 2.347971 / 2.575659 | `test_samples_are_typical_of_the_model` |
| trigram oracle NLL: mode / pure / held-out text | 1.073178 / 2.500323 / 1.968218 | `test_the_text_is_off_every_curve` |
| decoder pure samples: NLL / oracle NLL | 1.383573 / 2.163343 | — |
| E[NLL − Σ H]: trigram / decoder | 0.216 ± 0.159 / 0.037 ± 0.097 | same |
| lab tree TV(q_τ, π_τ) at τ = 0.5 | 0.302539 | `test_local_temperature_is_not_global` |
| Welleck (k = 5, ε = 0.05) critical τ | 1.031870 | `test_tempered_sequence_distribution_can_diverge` |
| P(pure sampling never leaves nucleus(0.95)): direct / identity | 0.782580 ± 0.010640 / 0.774669 | `test_pure_sampling_leaves_the_nucleus` |
| KN prompt rows with a seen context / separable / mean seen size | 177 / 0.847458 / 5.344633 | `test_kn_seen_set_is_recoverable_but_no_rule_finds_it` |
| copy rate τ = 0.5 / 1.5: trigram; decoder | 0.147606 / 0.024497; 0.225549 / 0.058907 | `test_temperature_trades_quality_for_diversity` |
| fact probes: seen / once / unseen | 30 / 33 / 48 | `test_fabrication_regimes` |
| seen: P(correct), P(wrong) at τ = 1 | 0.727256 / 0.197862 | same |
| once: argmax right; τ = 1 correct / wrong; τ = 0.040 correct / wrong | 0.696970; 0.239832 / 0.544213; 0.697726 / 0.302274 | same |
| unseen: argmax right; τ = 1 correct; max over every setting | 0.0625; 0.063653; < 0.1 (0.082) | same |

Check each value against the harness before quoting it on the page.

**Scope decisions** (Jonathan, 2026-10-24, on Claude's recommendations):
- **Spine: exact against biased.**
  - Ancestral sampling draws exactly from the model, and Gumbel-max makes sampling a search on perturbed scores.
  - Temperature and truncation bias the draw in ways we can state exactly: a free-energy optimum, a local/global gap,
    and TV/KL identities.
  - Typicality explains why #14's mode is nothing like a sample.
- **Engines: three, as in #14.**
  - Hand-built toy trees, exact by enumeration.
  - Topic 3's KN trigram, with exact per-step distributions.
  - #14's trained Topic-13 decoder.
- **Finance thread: boilerplate ↔ fabrication.**
  - The fact model is an **order-5 interpolated KN** (Topic 3's `kn_fit_any`, fit in 0.3 s). This was chosen after the
    spike showed #14's decoder and the trigram at chance on KB facts.
  - Fabrication is scored exactly against `WorldKB`, with no Monte Carlo.
- **Authorship:** Claude writes every body for Jonathan's review, as in Topics 5–14.
- **Difficulty:** `intermediate`, domain `decoding-generation`, `financeCaseStudy: true`.

**What the spike found (and changed):**
1. **The toy decoder knows no facts.** P(correct sector | "<company> operates in the") averages 0.205 over 29
   companies, about chance for 6 sectors (1/6). The trigram's two-word context cannot see the company at all.
   *(spike)*
2. **An order-5 KN has three regimes, set by how often the fact's 4-word context occurred in training:**
   - **seen ≥ 2 times:** the argmax is right and the remaining mass is smoothing (e.g. Halvorsen: 0.802 on
     "industrials");
   - **seen once:** the discounted count (≈ 0.12) loses to the backoff's sector prior (industrials 0.283, technology
     0.248), so the argmax is wrong for 4 of the 17 seen sector contexts;
   - **never seen:** the argmax is right 0.05 of the time.

   *(spike)* This is the finance thread's mechanism: truncation and low τ remove fabrication in the first regime,
   entrench it in the second, and cannot touch it in the third.
3. **Sampling is cheap.** The decoder draws 1,504 strings in 0.7 s. The trigram costs ~2–8 s per setting at N = 8 per
   prompt, so the harness transforms each distinct Markov state once and samples by inverse CDF.
4. **Samples are not the text.**
   - At τ = 1 the trigram's samples have per-word NLL 2.163 under the trigram. That is its entropy rate along its own
     paths, below the held-out text's 2.576 (a cross-entropy) and far above the mode's 1.367 (#14).
   - The decoder's τ = 1 samples are at 1.331. *(spike)*
5. **The oracle for "quality."** The corpus generator itself is the true distribution, so it serves as Caccia et al.'s
   oracle. We fit a KN model to a 5× continuation of the training corpus's own generator (`ng.continuation_run`, ~1 s),
   which continues the same Pitman–Yor restaurants so the vocabulary carries over. Oracle NLL is then the quality axis.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Sampling: Temperature, Top-k, and Nucleus |
| `{{SUBTITLE}}` | Ancestral sampling draws exactly from the model; temperature tilts it toward a Gibbs distribution and truncation cuts its tail, each at a cost we can compute exactly — and neither can add knowledge the model lacks |
| `{{SLUG}}` | `sampling-temperature-nucleus` (= graph node id) |
| `{{DOMAIN}}` | `decoding-generation` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[beam-search-decoding]` |
| `{{FORMALML_PREREQS}}` | `shannon-entropy`, `kl-divergence`; connections `information-geometry`, `sequential-monte-carlo` |
| `{{FORMALRAG_PREREQS}}` | none |
| `{{STATS_CALC_PREREQS}}` | Stats `exponential-families` (prereq), `large-deviations` (connection) |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/beam-search-decoding/`, `notebooks/n-gram-models-smoothing/`, `notebooks/zipf-heaps-vocabulary-statistics/`, `notebooks/finance-corpus/` |

- **Tags:** `sampling`, `temperature`, `top-k`, `nucleus-sampling`, `min-p`, `typical-sampling`, `gumbel-max`,
  `truncation`, `desmoothing`, `hallucination`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`** (prose links; only the #14 edge is in the graph):
  - Topic 1, for Zipf's law and the s/τ proposition;
  - Topic 3, for KN backoff as the smoothing that truncation removes;
  - Topic 5, for consistency: ancestral sampling halts with probability 1 iff the model is consistent;
  - Topic 14, for greedy, the mode, and #14's NLL numbers.
- **Graph:** the incoming edge #14 → #15 exists, and so does the outgoing #15 → #16.

**Scope.**
- **No sibling owns decoding.**
- **#16 owns** temperature scaling as calibration (Guo et al. 2017): fitting τ to held-out log loss. #15 uses τ as a
  decoding knob only, with one forward-pointing sentence.
- **#27 owns** BLEU. No self-BLEU here; diversity is distinct-n.
- **Out of scope:**
  - MBR decoding, contrastive search, speculative decoding, and classifier-free guidance;
  - mirostat, given one citation as a perplexity-targeting controller;
  - stochastic beam search beyond its statement (Kool et al., cited).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | #14's lab tree: the mode "was flat" (0.238), greedy "rose sharply" (0.0864), and a cloud of 1,000 samples | — | cells 0–1 |
| 1 | Sampling is exact; search is not | the tree with each leaf's probability as its sampling frequency | Def 1; **Thm 1** (exactness, halting ⟺ consistency); **Prop 1** (Gumbel-max); Kool et al. cited | cells 2–3 |
| 2 | Temperature as a Gibbs distribution | one next-token distribution as τ slides from 0.1 to 5 | Def 2; **Thm 2** (free energy; entropy monotone; limits); **Prop 2** (Zipf s → s/τ) | cells 4–5 |
| 3 | Local against global temperature | the lab tree: per-step τ → 0 lands on greedy, global τ → 0 on the mode | **Thm 3** (the normalizer product; condition; limits); Remark (global tempering can diverge for τ > 1) | cells 6–7 |
| 4 | Truncation: top-k, nucleus, min-p, η | sorted bars with the four cutoffs | Def 3; **Thm 4** (a)–(d) | cells 8–9 |
| 5 | Truncation as desmoothing | a KN row split into its discounted counts and its backoff mass | **Prop 3** (uniform mixture: one threshold recovers the support); **Prop 4** (KN: the seen set is recoverable exactly; no single threshold does it); Hewitt et al. | cells 10–11 |
| 6 | Typicality | sample NLLs as a histogram, with the mode and the held-out text marked | **Prop 5** (E NLL = E Σ H(h_t); martingale concentration); AEP cited; Def 4 (locally typical set) | cells 12–13 |
| 7 | Quality against diversity | frontier curves over τ, k, p, min-p, typical on both engines | measurements | cells 14–15 |
| 8 | Finance: boilerplate ↔ fabrication | three companies, three regimes, and P(correct) vs τ | exact probabilities from Thm 4(b) and Thm 2 | `<FinanceCaseStudy>`, cells 16–17 |
| 9 | What sampling is not | — | rigorFlag in prose | — |

## 2. Statements and proofs

The notebook's markdown cells carry the full statements and proofs; the MDX copies them, changing only formatting.
Notation: V is the alphabet with `</s>` in it, h a prefix, p(· | h) the next-token distribution, and z(h) = log p(· | h)
its log-probabilities. Strings y end in `</s>`.

- **Def 1** (ancestral sampling). Draw Y₁ ~ p(· | x), then Y_t ~ p(· | x, Y_<t) until `</s>`.
- **Thm 1** (sampling is exact).
  - **(i)** For every finite string y, P(ancestral sampling returns y) = p(y | x).
  - **(ii)** It halts with probability Σ_y p(y | x). So it halts almost surely iff the model is consistent (Topic 5).
  - **Proof:**
    - (i) is the chain rule, since the draws are conditionally independent given the prefix.
    - For (ii), the halting events {return y} are disjoint, so P(halt) = Σ_y p(y).
  - **Harness:**
    - exact on the lab tree and on 50 random trees: a χ² goodness-of-fit at 20,000 draws, with every leaf's empirical
      frequency within 4 standard errors;
    - halting: on Welleck's stationary model the length is Geometric(ε).
- **Prop 1** (Gumbel-max; Yellott 1977, Maddison et al. 2014). Let G_w be i.i.d. Gumbel(0, 1). Then
  argmax_w (z_w + G_w) ~ softmax(z), and max_w (z_w + G_w) ~ Gumbel(log Σ e^{z_w}), independent of the argmax.
  - **Proof:** with the CDF exp(−e^{−g}), P(argmax = i, max ≤ m) = ∫_{−∞}^{m} e^{z_i − g} exp(−e^{z_i − g}) ·
    Π_{j≠i} exp(−e^{z_j − g}) dg = (e^{z_i}/Z) exp(−Z e^{−m}).
  - **Corollary:** sampling is the argmax of perturbed scores, so search and sampling differ only by the noise.
  - **Cited:** Kool et al. 2019 (the top-down Gumbel process and Gumbel-top-k) turn beam search on perturbed sequence
    scores into exact sampling without replacement. One paragraph, no proof.
  - **Harness:** the argmax frequencies against softmax, and a KS test of the max against Gumbel(log Z).
- **Def 2** (temperature). p_τ(w | h) = p(w | h)^{1/τ}/Z_τ(h) = softmax(z(h)/τ), with τ > 0.
- **Thm 2** (temperature is a Gibbs distribution). For any z ∈ ℝ^V and τ > 0:
  - **(a)** p_τ is the unique maximizer over the simplex of ⟨q, z⟩ + τ H(q), with maximum τ log Σ e^{z/τ}.
  - **(b)** With β = 1/τ, dH(p_τ)/dτ = Var_{p_τ}(z)/τ³ ≥ 0, with equality iff z is constant on the support.
    E_{p_τ}[z] is nonincreasing in τ.
  - **(c)** As τ → 0, p_τ tends to uniform on argmax z. As τ → ∞, it tends to uniform on the support.
  - **Proof:**
    - (a): ⟨q, z⟩ + τH(q) = τ log Z_τ − τ KL(q ‖ p_τ), and Gibbs' inequality finishes it.
    - (b): log Z(β) is the cumulant generating function of z under the base measure, so d log Z/dβ = E_β z and
      d²/dβ² = Var_β z (formalStatistics `exponential-families` Thm 7.1). Then H = log Z − β E_β z gives
      dH/dβ = −β Var_β z.
    - (c): direct.
  - **Remark:** the tempered family is an exponential family with sufficient statistic log p and natural parameter β.
    Annealing walks along it (formalML `information-geometry`, its exponential-family Thm 5). Ackley, Hinton, and
    Sejnowski 1985 introduced the term in neural networks; Kirkpatrick et al. 1983 introduced annealing.
  - **Harness:**
    - (a) against `scipy.optimize` on 50 random z, plus 10⁴ Dirichlet-random q never beating p_τ;
    - (b) the entropy finite-difference derivative against Var/τ³, and monotone on a 200-point τ grid for every
      trigram prompt row.
- **Prop 2** (temperature is the Zipf slope). If p_r ∝ r^{−s} over ranks 1..n, then p_τ ∝ r^{−s/τ}. More generally,
  the least-squares slope of log p_τ against log rank is exactly (slope of log p)/τ for **any** distribution: ranks do
  not change, and log p_τ = (1/τ) log p − log Z_τ.
  - **Proof:** linearity of OLS in the response, plus invariance to adding a constant.
  - **Harness:**
    - exact on every trigram prompt row;
    - **measured, not exact:** the Topic-1 Zipf exponent of the *sampled text* at τ ∈ {0.7, 1, 1.4} departs from s/τ,
      because text mixes many contexts. The page says so.
- **Thm 3** (local temperature is not global temperature). Write the per-step tempered sampler as
  q_τ(y) = Π_t p_τ(y_t | h_t). The global tempered distribution is π_τ(y) = p(y)^{1/τ}/𝒵_τ, defined when
  𝒵_τ = Σ_y p(y)^{1/τ} < ∞. Then:
  - **(a)** q_τ(y) = p(y)^{1/τ}/Π_t Z_τ(h_t). So q_τ = π_τ iff Π_t Z_τ(h_t) is the same for every y in the support
    (sufficient: Z_τ(h) constant in h, e.g. every row a permutation of one vector).
  - **(b)** For τ < 1, Z_τ(h) ≤ 1, with equality iff p(· | h) is a point mass. So relative to π_τ, q_τ upweights strings
    that pass through high-entropy prefixes.
  - **(c)** If argmaxes are unique, q_τ → δ(greedy string) and π_τ → δ(mode) as τ → 0. On #14's lab tree these are
    "rose sharply" and "was flat".
  - **Remark:** for τ > 1, 𝒵_τ can be infinite on an infinite-support consistent model (Welleck's stationary model:
    Σ_t ((1 − ε)/k)^{t/τ} k^t ε^{1/τ} diverges once k^{1 − 1/τ}(1 − ε)^{1/τ} ≥ 1). So "the model at temperature τ" may
    not exist as a sequence distribution, while the per-step sampler always does.
  - **Proof:**
    - (a) substitutes Def 2 and collects;
    - (b): Σ_w p_w^{1/τ} ≤ Σ_w p_w = 1 for 1/τ > 1;
    - (c) takes term-wise limits on a finite tree;
    - the Remark is a geometric series.
  - **Harness:** TV(q_τ, π_τ) on the lab tree and 50 random trees over a τ grid, zero exactly on permutation-row trees,
    with both limits checked.
- **Def 3** (truncation). For a kept set S(h) ⊆ V, p_S(w | h) = p(w | h) 1[w ∈ S]/m(h), where m(h) = p(S(h) | h). The
  rules, with p sorted p₍₁₎ ≥ p₍₂₎ ≥ …:
  - **top-k:** the k largest (Fan et al. 2018);
  - **nucleus(p):** the smallest prefix of the sorted order with mass ≥ p (Holtzman et al. 2020);
  - **min-p(ρ):** {w : p_w ≥ ρ p₍₁₎} (Nguyen et al. 2024);
  - **η:** {w : p_w > min(η, √η e^{−H})} (Hewitt et al. 2022);
  - **locally typical(τ):** Def 4.

  Ties are broken by token id. (*Contribution point:* the body of `truncation_set`.)
- **Thm 4** (what truncation costs).
  - **(a)** The nucleus is a minimum-cardinality set with mass ≥ p, and among sets of its size it has maximum mass.
    Top-k has maximum mass among k-sets.
    - **Proof:** an exchange argument. The k largest entries dominate any k entries.
  - **(b)** TV(p_S, p) = 1 − m, KL(p_S ‖ p) = −log m, and KL(p ‖ p_S) = ∞ when m < 1.
    - **Proof:** direct.
  - **(c)** Sequence level: for the truncated sampler q,
    - KL(q ‖ p) = E_q[Σ_t −log m(h_t)] **exactly** (chain rule for KL);
    - TV(q, p) ≤ E_q[Σ_t (1 − m(h_t))] (step-wise maximal coupling: the paths agree until the first decoupling, which
      happens at step t with probability 1 − m(h_t) given agreement so far).
  - **(d)** Pure sampling's tail risk: if the "tail" T(h) has mass ≥ ε at every step, P(no tail token in n steps) ≤
    (1 − ε)^n. This is Holtzman et al.'s argument made exact.
  - **Harness:**
    - (a) brute force over all subsets on 200 random 8-word distributions;
    - (b) exact;
    - (c) exact enumeration on random trees for every rule (KL equality to 10⁻¹², TV bound);
    - on the corpus, the per-token KL(q ‖ p) estimate for each rule enters the frontier.
- **Prop 3** (uniform desmoothing; Hewitt et al. 2022's model). If p = (1 − λ)p* + λ/|V|, then
  {w : p_w > λ/|V|} = supp p* exactly. If p* is bounded below by δ on its support, every threshold in
  (λ/|V|, (1 − λ)δ + λ/|V|) recovers it.
  - **Proof:** p_w = λ/|V| iff p*_w = 0.
- **Prop 4** (KN desmoothing). For interpolated KN,
  p(w | h) = max(c(hw) − D, 0)/c(h) + γ(h) p_lower(w | h⁻).
  - The seen set {w : c(hw) > 0} = {w : p(w | h) > γ(h) p_lower(w | h⁻)} is recoverable exactly from the model's own
    decomposition, since c(hw) ≥ 1 > D.
  - No single threshold on p(w | h) recovers it in general, because the backoff mass γ(h) p_lower(w) varies by word.
  - **Proof:** read off the recursion.
  - **Harness:** on the trigram, the precision and recall of top-k, nucleus, min-p, and η against the exact seen set,
    over all 188 prompt rows and along sampled paths.
  - **Page claim (to pin):** no setting of any rule reaches both precision and recall 1 on every row.
- **Prop 5** (samples are typical of the model, not of the text).
  - E_p[−log p(Y | x)] = E_p[Σ_t H(p(· | h_t))], the chain rule for entropy.
  - M_n = Σ_{t≤n} (−log p(Y_t | h_t) − H(h_t)) is a martingale. So a sample's NLL concentrates around its path's
    accumulated entropy, not around the mode's NLL or the text's.
  - **Cited:** the AEP for i.i.d. sources (Cover and Thomas, §3.1) and Shannon–McMillan–Breiman for stationary ergodic
    ones (§16.8).
  - **Harness:**
    - the mean of NLL − Σ H along sampled paths is ≈ 0 within 4 standard errors on both engines;
    - the ordering mode NLL < τ = 1 sample NLL < held-out text NLL on the trigram (1.367 < 2.163 < 2.576 *(spike)*).
- **Def 4** (locally typical; Meister et al. 2023, Alg. 5.3 in its greedy form per their erratum). Sort tokens by
  |−log p_w − H(h)| and keep them until their mass reaches the typical mass.

## 3. Corpus and finance measurements (§7–8)

- **Grid** (per engine, N samples per prompt × 188 prompts, one RNG stream per engine):
  - τ ∈ {0.5, 0.7, 0.85, 1, 1.15, 1.3, 1.5};
  - k ∈ {5, 20, 80};
  - nucleus p ∈ {0.5, 0.8, 0.9, 0.95};
  - min-p ρ ∈ {0.02, 0.05, 0.1, 0.2};
  - η ∈ {3·10⁻⁴, 10⁻³, 3·10⁻³};
  - typical τ ∈ {0.2, 0.5, 0.9}.
- **Metrics:**
  - finished share and mean length (vs. the held-out 7.298);
  - per-word NLL under the generating model;
  - per-word **oracle NLL** under the KN fit to the 5× continuation (the quality axis; lower is more like the
    generator);
  - distinct-2 and distinct sentences (the diversity axis);
  - copy rate, i.e. verbatim training sentences (the boilerplate axis);
  - loop rate (a period-1 repeat of ≥ 3 tokens);
  - per-token KL(q ‖ p) from Thm 4(c), or the importance log-ratio for τ.
- **Finance probes.**
  - **Probes:** every (entity, frame) with entities from `WorldKB.facts("IN_SECTOR")` and `facts("HQ_IN")`, and frames
    "operates in the", "is classified under", and "is headquartered in".
  - **Model:** the order-5 KN. Each probe's regime is the training count of its 4-word context: 0, 1, or ≥ 2.
  - **Scoring:** for each decoding rule and parameter we take the exact P(correct answer), P(wrong answer), and
    P(non-answer token), averaged within each regime. These are curves, with no sampling.
  - **Boilerplate side:** the copy rate of free samples from the 188 MD&A prompts (trigram and decoder), and the share of
    samples that end on #14's 8-K tail.
  - **Page claims (to pin):**
    - **(i)** in regime ≥ 2, P(wrong) falls monotonically as τ ↓ and as nucleus p ↓;
    - **(ii)** in regime 1, P(wrong) → 1 as τ → 0 for the companies whose argmax is wrong, so lowering τ makes
      fabrication confident;
    - **(iii)** in regime 0, no rule moves P(correct) above the backoff prior's share.

## 4. The lab (`SamplingLab`, `data/sampling-temperature-nucleus-lab.json`)

Four views. Every number comes from the payload.

- **(a) Temperature dial.**
  - One next-token row: a prompt picker over 6 trigram rows chosen by rule, plus a Zipf toy.
  - A τ slider on the payload grid of 41 log-spaced values.
  - Readouts: sorted bars, entropy, the Zipf slope (exactly slope/τ), and E[z].
  - *Learn:* τ rotates the log-log line; entropy only rises with τ.
- **(b) Truncation cutoffs.** The same row with a rule selector (top-k, nucleus, min-p, η, typical) and a parameter
  slider.
  - Kept bars are highlighted, plus the TV = 1 − m and KL = −log m readouts.
  - On trigram rows, a "seen in training" overlay from Prop 4.
  - *Learn:* the four rules cut in different places, and none cuts exactly at the seen set.
- **(c) Local vs. global.** #14's lab tree with leaf probabilities under q_τ and π_τ as τ slides, and the TV readout.
  - *Learn:* as τ → 0 the local sampler collapses on greedy's "rose sharply" and the global one on the mode "was flat".
- **(d) Frontier.**
  - Engine toggle; x = distinct-2 (diversity), y = oracle NLL (quality).
  - One curve per rule family across its parameter grid; the τ = 1 point, #14's mode, and the held-out text are marked.
  - **Finance sub-panel:** P(correct) / P(wrong) vs τ for the three regimes, with one example company each.

Mobile: views stack, controls are flex-wrap rows, and `scrollWidth === 375` holds in every view and toggle.

## 5. rigorFlag (draft)

> **What is proved.** Thms 1–4 and Props 1–5 are proved here, with three exceptions that are cited instead:
> Kool et al.'s stochastic beam search, the AEP / Shannon–McMillan–Breiman theorems, and Hewitt et al.'s and
> Meister et al.'s motivations for η and typical sampling.
>
> **What is heuristic.** Every truncation threshold (k, p, ρ, η, the typical mass) is a heuristic: none is derived from
> the model's actual error. Desmoothing assumes a mixture form that a neural model need not have. Per-step temperature
> is not the tempered sequence distribution (Thm 3), and that distribution may not exist for τ > 1.
>
> **What is measured, and on what.** The corpus engines are tiny (a 3,427-word trigram and a width-32 decoder over 509
> words), both renormalized each step without `<unk>`. "Quality" is oracle NLL under a KN fit to more text from the same
> synthetic generator, so it is a proxy with its own smoothing. Fabrication is scored only for three relation frames
> against a synthetic knowledge base.

## 6. Notebook cell map (planned, ~21 cells)

| Cells | Content |
|---|---|
| 0–1 | title, headline |
| 2–3 | §1: Def 1, Thm 1, Prop 1, Kool |
| 4–5 | §2: Def 2, Thm 2, Prop 2 |
| 6–7 | §3: Thm 3, the Remark |
| 8–9 | §4: Def 3, Thm 4 |
| 10–11 | §5: Props 3–4 |
| 12–13 | §6: Prop 5, Def 4 |
| 14–15 | §7: grid and frontier |
| 16–17 | §8: finance |
| 18–20 | payload and harness |

## 7. Cross-site

All verified `status: published` on each sibling's GitHub `main`, 2026-10-24.

- `formalmlPrereqs`:
  - `shannon-entropy`: H(p_τ), conditional entropies along paths, and the chain rule for entropy (its Thm 1) drive
    Thm 2 and Prop 5.
  - `kl-divergence`: truncation's cost is KL(p_S ‖ p) = −log m, and the reverse KL is infinite. Its forward/reverse
    section is the right frame.
- `formalmlConnections`:
  - `information-geometry`: the tempered family is an exponential family in β = 1/τ, and annealing walks a curve in its
    dually flat geometry (its Thm 5).
  - `sequential-monte-carlo`: ancestral sampling is the proposal of SMC with no resampling, and tempering sequences
    are SMC's standard bridge between distributions.
- `formalstatisticsPrereqs`:
  - `exponential-families`: d log Z/dβ = E z and d² log Z/dβ² = Var z (its Thm 7.1) give Thm 2(b).
- `formalstatisticsConnections`:
  - `large-deviations`: a sample's NLL concentrates around its accumulated entropy at an exponential rate (its Azuma–
    Hoeffding Thm 7, for bounded surprisal increments).
- **Not cited:** formalRAG `selective-generation-abstention` belongs to #16's watch-list scope.

## 8. References

Verified 2026-10-24 against arXiv, ACL Anthology, PMLR, Crossref, and the PDFs (pdftotext). Chicago 17 N-B in the MDX.

- **Truncation:**
  - Fan, Lewis, and Dauphin 2018 (ACL, P18-1082, pp. 889–898). They sample "from the k = 10 most likely candidates"
    (§5.4) without claiming the method as theirs, so the page says **popularized**, not introduced.
  - Holtzman et al. 2020 (ICLR, arXiv 1904.09751). V^(p) is Eq. 2, the rescaled distribution Eq. 3, and temperature
    Eq. 4, credited to Ackley et al.
  - Hewitt, Manning, and Liang 2022 (Findings of EMNLP, 2022.findings-emnlp.249, pp. 3414–3427).
    - The smoothing model is Eq. 3: P_θ = λP* + (1 − λ)Q, with Q within δ of uniform. Note that their λ weights P*,
      the reverse of our Prop 3.
    - The "Absolute" and "Relative probability" principles are §3.4, Eqs. 5–6.
    - η = min(ε, √ε e^{−h}) is §4.4.
    - They give support recovery in prose (§4.4 Eq. 8, App. A.2 Eq. 16), with no numbered proposition, so we cite the
      sections.
  - Meister et al. 2023 (TACL 11: 102–121, doi 10.1162/tacl_a_00536).
    - The typical set is Def 3.5 (Eq. 7); the sampler is Alg. 5.3 with mass parameter τ.
    - Per their erratum (arXiv fn. 18), Def 4 uses the **greedy** form: sort by |H + log p| and take words until the
      mass reaches τ.
    - The page calls that mass parameter "the typical mass", to avoid a clash with temperature τ.
  - Nguyen et al. **2025** (ICLR 2025 oral, arXiv 2407.01082). §3.1: p_scaled = p_base · p_max, keeping P(v) ≥ p_scaled
    (≥ here, against η's >). In common implementations temperature is applied before the threshold.
  - Basu et al. 2021 (ICLR, arXiv 2007.14966; mirostat, mention only).
- **Gumbel:**
  - Yellott 1977 (*J. Math. Psych.* 15(2): 109–144, doi 10.1016/0022-2496(77)90026-8): the argmax result, via Luce's
    choice axiom.
  - Maddison, Tarlow, and Minka 2014 (NeurIPS 27, arXiv 1411.0030; Eqs. 1–2; no page range confirmed, so cite without
    pages).
  - Kool, van Hoof, and Welling 2019 (ICML, PMLR 97: 3499–3508). Gumbel-max is stated unnumbered in §2.3; Gumbel-top-k
    is **Thm 1** in §2.4.
- **Temperature:**
  - Jaynes 1957 (*Phys. Rev.* 106(4): 620–630, doi 10.1103/PhysRev.106.620);
  - Kirkpatrick, Gelatt, and Vecchi 1983 (*Science* 220(4598): 671–680, doi 10.1126/science.220.4598.671);
  - Ackley, Hinton, and Sejnowski 1985 (*Cogn. Sci.* 9(1): 147–169, doi 10.1207/s15516709cog0901_7).
- **Local vs. global (Thm 3's prior statements, attributed in prose):**
  - Caccia et al. 2020 (ICLR, arXiv 1811.02549, §3). Tempering the conditionals "is different from changing the
    temperature of the joint". The bias is toward lower entropy early in the sentence, and tempering the joint is
    intractable. They also sweep temperature to trace quality–diversity curves and use oracle NLL.
  - Shih, Sadigh, and Ermon 2023, "Long Horizon Temperature Scaling" (ICML, PMLR 202: 31422–31434, arXiv 2302.03686;
    §2.1 Eq. 3). Myopic T → 0 "will not necessarily produce the argmax sample".
  - Thm 3's exact normalizer-product form, the iff condition, the entropy-direction statement (b), and the divergence
    remark are, as far as we found, not stated in either source. We prove them.
  - Andor et al. 2016 (ACL, P16-1231, pp. 2442–2452; Thm 3.1, P_L ⊊ P_G, after Smith and Johnson 2007): label-bias
    intuition only, not a theorem about temperature.
- **Typicality:** Cover and Thomas 2006 (Wiley, 2nd ed., doi 10.1002/047174882X). Cite by **section**: §3.1 (AEP) and
  §16.8 (Shannon–McMillan–Breiman). The theorem numbers 3.1.1 and 16.8.1 were seen only in an unofficial scan.
- **Quality–diversity:** Zhang et al. 2021 (HumEval at EACL, 2021.humeval-1.3, pp. 25–33).
- **Forward pointer:** Guo et al. 2017 (ICML, PMLR 70: 1321–1330).

Full verification notes are in the session scratchpad (`refs15/refs.md`) and are not committed.

## 9. Open questions for the checkpoint

1. **The fact model.** An order-5 KN is a fourth model on the page, used only in §8. The page frames it as "Topic 3's
   model with a longer memory," which is honest. Is that acceptable, or should the finance probes stay on the decoder
   at chance?
2. **The quality axis.** Oracle NLL under a KN fit to more text from the same generator, as in Caccia et al., is the
   cleanest quality proxy available without human judgment. Keep it, or report only diversity, copy rate, and
   fabrication?
3. **Grid size.** 30 settings × 2 engines × 188 prompts × N = 8 fits the 60 s budget only with per-state caching. If it
   doesn't, drop η and typical from the corpus grid and keep them in the lab's single-row views.
