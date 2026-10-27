# Brief — `significance-testing-for-nlp`

Topic 28 of 29, domain `nlp-evaluation`. Phase A is complete: the harness
`notebooks/significance-testing-for-nlp/significance_testing_for_nlp.py` passes 22/22 in about 46 s (and under
`python -W error`), and `01_significance_testing_for_nlp.ipynb` (28 cells) executes cleanly. Every number below is
quoted from the harness's `summary()`.

**Decisions (Jonathan, 2026-10-27).** Plan mode:
- Full gap-fill scope against formalRAG `significance-testing-calibration`.
- The training-variance testbed is Topic 17's simulated worlds.
- The brevity-penalty kink gets a proposition plus a harness test.

At the scaffold checkpoint Jonathan approved the four recommendations:
- 128 A/A experiments;
- one-sided p-values for Prop 2 and two-sided for the matrix;
- a Gaussian toy for Prop 3 plus the flip-family remark;
- almost stochastic order cited only.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Significance Testing for NLP |
| `{{SUBTITLE}}` | Resampled corpora are points on a simplex, and swapped outputs are vertices of a cube. A corpus metric is a surface over both. Randomization is exact on any surface, the bootstrap only where the surface is smooth, and neither sees variance that the resampled unit hides. |
| `{{SLUG}}` | `significance-testing-for-nlp` |
| `{{DOMAIN}}` / `{{DIFFICULTY}}` / `{{FINANCE_CASE_STUDY}}` | `nlp-evaluation` / `advanced` / `true` |
| `{{PREREQUISITES}}` | `[overlap-metrics-bleu-rouge, linear-text-classifiers]` (the graph's incoming edges) |
| `{{FORMALML_PREREQS}}` | `rank-tests` (Thm 3: permutation-test exactness under a finite group; Thm 1: sign-test null; Def 9 and Thm 12: Fisher's sharp null); `concentration-inequalities` (Thm 4: Hoeffding, for the Monte Carlo error of a p-value) |
| `{{FORMALRAG_PREREQS}}` | `significance-testing-calibration` (Def 1: paired t and d_z; Thm 1: pairing reduces variance; Prop 1: power-based n). Cited, never re-proved. |
| `{{STATS_CALC_PREREQS}}` | formalStatistics `bootstrap` (Thm 3: Efron–Bickel–Freedman, scalar mean); `central-limit-theorem` (Thm 7: multivariate CLT; Thm 8: delta method, scalar); `multiple-testing-and-false-discovery` (Thm 1: Bonferroni; Thm 3: Holm; Thm 5: BH); `hypothesis-testing` (Thm 3: p-value uniformity under a continuous null). formalCalculus `probability-and-union-bound` (Cor 1: Bonferroni's inequality). |
| connections (in-site) | `language-model-calibration`, `named-entity-recognition` (span F1 is another non-decomposable metric) |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/overlap-metrics-bleu-rouge/`, `notebooks/linear-text-classifiers/`, `notebooks/aspect-based-sentiment/` |

**Sibling check (2026-10-27, GitHub `main`, all `published`).** Every theorem number in the table was read from the
fetched MDX. Three gaps shape what this page proves itself:
- **Scalar results only.** formalStatistics' delta method (CLT Thm 8) and its Efron–Bickel–Freedman theorem (bootstrap
  Thm 3) are both scalar. The page proves the multivariate delta step itself and cites the multivariate bootstrap mean
  to Bickel and Freedman (1981), without a theorem number (see §7).
- **Uncovered methods.** No sibling covers McNemar's test, approximate randomization, corpus-level (non-decomposable)
  metrics, bootstrap p-value conventions, clustered resampling, training-run variance, or metric shopping.
- **Full group only.** formalML `rank-tests` Thm 3 proves exactness for the full-group average. The Monte Carlo version
  (Thm 1 here) is not on any sibling.

**Scope against formalRAG `significance-testing-calibration`.** That topic tests per-query **means** (AP, NDCG) with a
paired t-test, a sign-flip permutation test and a recentered bootstrap. It also shows pairing's variance reduction and
gives a power-based query count. This topic starts where those tools stop:
- the metric is not a mean (corpus BLEU, F1);
- the outcome is 0/1 (McNemar);
- the bootstrap's p-value convention matters;
- the metric has a kink (the brevity penalty, BP);
- segments are not independent (clusters);
- the question concerns a training procedure, not one trained model.

The paired t-test, Cohen's d_z and the normal-approximation power n are cited to formalRAG. Calibration and drift are
not touched.

## 1. Section outline (geometric first)

1. **Two clouds.** For a pair of systems on Topic 27's 90 filings:
   - each bootstrap resample is a weight vector on the simplex Δ⁸⁹, centered at the uniform point;
   - each randomization is a vertex of the swap cube {0, 1}⁹⁰;
   - corpus BLEU's difference is a surface over both.

   The lab's first tab shows both clouds. The hook: writer vs. noisy-0.1 differ by δ = 0.006828 BLEU with
   randomization p = 0.0005, while extract vs. textrank-3 differ by δ = 0.006112 with p = 0.2017.
2. **The unit of resampling** (Defs 1–3).
3. **Randomization is exact for any metric** (Thm 1, Prop 1).
4. **The bootstrap's two p-values, and where it breaks** (Prop 2, Thm 2, Prop 3).
5. **Clustered segments** (Def 4, Thm 3).
6. **What a test set cannot tell you** (Def 5, Thm 4).
7. **Power and multiplicity** (Prop 4, Prop 5; Holm and BH cited).
8. **Finance case study.** Tone classifiers on 1,500 filings; the 45 summarizer pairs on 90 filings.
9. **Where this sits / Honest caveats / Implementation.**

## 2. Statements (numbered per type, in page order; the same numbers in the `.py` and the notebook)

**Def 1 (paired evaluation, corpus metric).**
- **Setting.** n segments. Segment i carries both systems' outputs, the references, and per-system sufficient
  statistics s_i^A, s_i^B ∈ ℝ^K. For BLEU these are (c, r, m₁..₄, t₁..₄), so K = 10.
- **Corpus metric.** M = g(Σ_i s_i), and the weighted form is M_w = g(Σ_i w_i s_i).
- **Decomposable vs. not.** A metric is decomposable when g(P) = P₀/P₁ on rows (x_i, 1), i.e. a mean; accuracy is an
  example. Corpus BLEU and F1 are not decomposable.
- **Homogeneity.** Corpus BLEU is homogeneous of degree 0, so weights on the simplex and counts give the same value.
  Harness: `test_def1_pooled_bleu_matches_topic27`.

**Def 2 (paired bootstrap; Koehn 2004, §6).**
- Draw multinomial(n, 1/n) counts, i.e. a random point of the simplex. Recompute δ* with the **same** weights for both
  systems.
- Koehn's words: if one system outperforms the other in 95% of 1,000 resampled test sets, "we draw the conclusion that
  it is better with 95% statistical significance."

**Def 3 (approximate randomization; Noreen 1989, Riezler and Maxwell 2005, Fig. 1).**
- Swap each segment's two statistic tuples with probability ½.
- The p-value is p = (1 + #{b : |δ_b| ≥ |δ̂|})/(B + 1).

**Thm 1 (Monte Carlo randomization p-values are valid for every statistic).**
- **Statement.** Assume the sharp null: g·X =d X for every g in the swap group G = (ℤ/2)ⁿ. Then P(p ≤ α) ≤ α for every
  statistic T, every B ≥ 1 and every α.
- **Proof (full).**
  1. Take U₀ uniform on G, independent of the data. Then X′ = U₀X =d X.
  2. (U₀, ε₁U₀, …, ε_BU₀) is the image of i.i.d. uniform variables under a bijection of G^{B+1}. So it is i.i.d.
     uniform and independent of X, and the vector (T(X′), T(ε_bX′)) is exchangeable.
  3. The rank argument, with ties counted against the observed value, finishes the proof.
- **Remarks.**
  - The full-group case is formalML `rank-tests` Thm 3.
  - "+1" follows Phipson and Smyth (2010, §4): b/m inflates the type I error (§3).
  - B ≥ log(2/η)/(2ε²) gives Monte Carlo error ≤ ε with probability 1 − η (formalML Thm 4).
  - **Superuniform, not uniform.** With 10 prompts many swaps change nothing, so p has atoms and the test is
    conservative.
- **Harness.**
  - **The exact A/A null.** Four trigram sample streams (5010–5013) give 64 i.i.d. samples per prompt. Sample j and
    sample j + 32 are exchangeable within every prompt. Disjoint prompt blocks are independent experiments: 2,304 at
    n = 10, 1,152 at 20, 512 at 47, 256 at 94, 128 at 188.
  - **AR size at 0.05:** 0.048611, 0.040799, 0.039062, 0.050781, 0.062500, each inside its 3σ band. The empirical CDF
    stays inside the band on a grid of α values, and size(0.5) = below 0.5 − 3σ.
  - **Monte Carlo vs. full group.** The Monte Carlo p matches the full-group p over 4,096 swaps.
  - **scipy cross-check.** The full group matches scipy's `permutation_test(permutation_type="samples")` to 10⁻¹².

**Prop 1 (on 0/1 losses, randomization is exact McNemar).**
- **Statement.** For accuracy, the full-group randomization test is McNemar's exact test: conditional on n_d, n₁₀ is
  Bin(n_d, ½).
- **Proof (full).** Concordant segments are swap-invariant. Each discordant segment's contribution changes sign under a
  swap, so nδ = Σ_{discordant} ε_i with Rademacher ε_i.
- **Harness.**
  - Exact equality on enumerated 14-filing subsets, and with `scipy.stats.binomtest`.
  - On the full 1,500 filings, Monte Carlo AR agrees within Monte Carlo error.
  - Swapping only the discordant filings gives the same δ.
- **Note.** Dietterich (1998, §3.1) uses the continuity-corrected χ² form, and Card et al. (2020, §3.1) the uncorrected
  one. We use the exact binomial form.

**Prop 2 (Koehn's rule inverts the percentile interval; the 2δ rule inverts the basic interval).**
- **Statement.** Let q_α = δ*₍⌊αB⌋+1₎ and q_{1−α} = δ*₍B−⌊αB⌋₎.
  - Koehn's p = P*(δ* ≤ 0) satisfies p ≤ α iff q_α > 0.
  - Berg-Kirkpatrick, Burkett and Klein (2012, Fig. 1, §2.2) define p = P*(δ* ≥ 2δ̂); it satisfies p ≤ α iff
    2δ̂ − q_{1−α} > 0.
  - If the bootstrap law is symmetric about δ̂, the two p-values are equal.
  - BBK count strictly (>) with p ≈ s/b, and their fn. 2 notes the symmetric equivalence. We count ≥ (conservative).
- **Proof (full).** Order-statistic counting. Symmetry follows by reflection: δ* − δ̂ =d δ̂ − δ*.
- **Remark: a third convention.** sacreBLEU 2.6.0's `paired_bs` follows Riezler and Maxwell (Fig. 2, Noreen's shift).
  It compares |δ*| − mean|δ*| with |δ̂| and returns (c + 1)/(n + 1). Its AR uses a strict > (from multeval), while R&M
  use ≥. The page names all three conventions.
- **Harness.**
  - The inversions hold exactly on every cloud at five α values, and the symmetrized cloud gives equal p-values.
  - Writer vs. noisy-0.1 at 90 filings: Koehn 0.000000, 2δ 0.004000.
  - **A/A sizes at 0.05.**

    | Prompts | Koehn | 2δ |
    |---|---|---|
    | 10 | 0.037326 | **0.068576** |
    | 20 | 0.059896 | 0.059896 |
    | 47 | 0.064453 | 0.060547 |
    | 94 | 0.046875 | 0.058594 |

    At 10 prompts the 2δ rule's size is outside its band (0.0636), while AR's is inside.
  - **A degenerate cloud.** Extract vs. textrank-3 at 10 filings has a pooled 4-gram count of 0, so BLEU is 0 on most
    resamples. The 2δ rule then gives p = 0.000500 where AR gives 0.854536.

**Thm 2 (corpus BLEU is asymptotically decomposable away from the kink).**
- **Statement.** Let Z_i = (s_i^A, s_i^B) be i.i.d. with finite second moments, every expected match count positive,
  and E c ≠ E r. Then:
  - √n(δ̂ − δ) → N(0, ∇gᵀΣ∇g);
  - δ̂ = δ + n⁻¹Σ ∇g(μ)·(Z_i − μ) + o_p(n^{−1/2});
  - the paired bootstrap's √n(δ* − δ̂) has the same normal limit almost surely.
- **Proof.**
  1. Apply the multivariate CLT (formalStatistics CLT Thm 7).
  2. Take a first-order Taylor step, written out. The sibling's Thm 8 is scalar, so this step is proved here.
  3. Use the bootstrap CLT for the mean in ℝ^K, from Bickel and Freedman (1981), and the same Taylor step
     conditionally.
- **Gradient of log BLEU.** The brevity term contributes r/c² and −1/c below c = r (nothing above). Each precision
  contributes 1/(N m_n) and −1/(N t_n).
- **Harness.**
  - The gradient matches central differences.
  - At 90 filings, the bootstrap sd is within 5% of the linearization for every pair whose systems pool ≥ 50 matched
    4-grams. For example, writer vs. noisy-0.1: 0.002166 vs. 0.002158.
  - The extractors pool only 3–7 matched 4-grams. Extract vs. textrank-3 gives 0.004976 vs. 0.004476.
  - In the bootstrap world, √m·E|remainder| shrinks about like m^{−1/2} across m = 30, 300, 3,000.

**Prop 3 (the brevity-penalty kink).**
- **Statement.** log BP = min(0, 1 − r̄/c̄). If E c = E r, then √n log BP(Z̄) → min(0, aW) with W ~ N(0, 1): an atom of
  ½ at 0, so the limit is not Gaussian. The bootstrap is not consistent there.
- **Proof.**
  - The limit follows from the continuous mapping theorem, since min(0, ·) is continuous and positively homogeneous.
  - For inconsistency, cite Fang and Santos (2019): Thm 3.1 says the standard bootstrap is consistent iff φ′ is linear
    (with G₀ Gaussian), and Cor 3.1 says a non-Gaussian limit makes the bootstrap inconsistent. Andrews (2000) is the
    classical boundary example, max(X̄, 0). Fang and Santos are read from arXiv 1404.3763 v2; check the numbering
    against the journal version.
  - Proved here: the bootstrap's conditional atom P*(φ* = φ̂) is a function of √n x̄, which converges in law but not
    in probability.
- **Harness** (reduced form φ = min(0, x̄), with x̄ the mean of n = 100 N(h/√n, 1) draws; 2,000 datasets, 499
  resamples each).

  | h | φ̂ = 0 | Percentile | Basic | Atom mean | Atom sd |
  |---|---|---|---|---|---|
  | −5 | 0.000 | 0.8865 | 0.8870 | 0.000 | 0.000 |
  | −1 | 0.169 | 0.8855 | **0.6325** | 0.117 | 0.266 |
  | −0.5 | 0.298 | 0.8935 | **0.5440** | 0.211 | 0.332 |
  | 0 | 0.5215 | 0.9445 | 0.9435 | 0.393 | 0.391 |
  | 0.5 | 0.7005 | 0.9825 | 0.9825 | 0.557 | 0.385 |
  | 1 | 0.8335 | 0.9955 | 0.9960 | 0.704 | 0.340 |

  - The atom's sd at h = 0 is still 0.380998 at n = 400. It does not converge to a constant.
  - The basic interval, which the 2δ rule inverts, fails worst just on the linear side of the kink.
- **Remark: the kink is live.** Every flip-family system sits at pooled c/r = 1.007741. Within a flip-family pair both
  systems share lengths, so BP cancels on every resample (asserted). The kink bites when the two systems' lengths differ
  near r.

**Def 4 (cluster bootstrap).** Resample whole clusters (documents or sentences) with replacement.

**Thm 3 (design effect).**
- **Statement.** Take K clusters of m segments, with variance σ², within-cluster correlation ρ, and n = Km.
  - (a) Var(x̄) = (σ²/n)(1 + (m − 1)ρ).
  - (b) E[segment-bootstrap variance] = (σ²/n)(1 − (1 + (m − 1)ρ)/n).
  - (c) The cluster bootstrap is consistent as K → ∞ (cite Davison and Hinkley 1997, §3.8; Field and Welsh 2007).
- **Proof.** (a) and (b) in full, by bilinearity, using tr(CΣ) for the centering matrix C. (c) is cited.
- **Harness.**
  - (a) and (b) exactly, including ρ = −1 at m = 2.
  - (c) on simulated clusters with K = 400, m = 8, ρ = 0.3.
  - Topic 19's 14,179 test aspect rows (1,500 documents, 12,921 sentences):

    | Reader | Unit | Mean | Design effect | Within-cluster ρ |
    |---|---|---|---|---|
    | blind | sentence | 0.887792 | 0.822554 | −1 (contrast pairs) |
    | lexicon | document | 0.784752 | **1.555407** | ICC 0.057302 |
    | difference | document | 0.103040 | 0.930738 | ICC −0.011707 |

  - The aspect-blind reader is right on exactly one aspect of every contrast sentence.
  - A single system's accuracy interval needs a document-cluster bootstrap: its standard error moves from 0.003452 to
    0.004305. The paired difference does not, because pairing absorbs the document effect.

**Def 5 (procedure-level null).** H₀^proc: E_D[acc(P_A, D) − acc(P_B, D)] = 0 over training draws D. Contrast
H₀^model(D): acc(P_A, D) = acc(P_B, D) for the one trained pair.

**Thm 4 (a test set cannot test a procedure).**
- **Statement.**
  - (a) Var(δ̂) = E_D[Var(δ̂ | D)] + Var_D(Δ(D)).
  - (b) Suppose H₀^proc holds, P(Δ(D) = 0) = π₀, and the test is consistent against every fixed Δ ≠ 0. Then
    P(reject) → (1 − π₀) + π₀α′ with α′ ≤ α.
- **Proof (full).** (a) is the law of total variance. For (b), condition on D and apply dominated convergence.
- **Harness.** Topic 17's bursty world (V = 101); NB with α = 1 vs. LR with λ = 10⁻², both fixed; 40 nested training
  draws.
  - Crossover rule: the n with |mean Δ| smallest. That is n = 384, where mean Δ = −0.000564 ± 0.001278.
  - At the crossover, between-draw variance is 0.000065 and within-draw variance is 0.000048; the total is 0.000113.
  - McNemar rejects in 0.175 of draws on 3,000 test documents (LR better in 0.025, NB in 0.150) and in **0.625** on
    20,000.
  - Further from the crossover, both rejection rates rise toward 1: at n = 1,024 they are 0.85 and 1.0.

**Prop 4 (exact McNemar power).**
- **Statement.** n_d ~ Bin(n, p_d), and given n_d, n₁₀ ~ Bin(n_d, p₁₀/p_d). Exact power is the resulting double sum, over
  a ±12σ window in n_d. Connor's normal-approximation n is [z√p_d + z_β√(p_d − Δ²)]²/Δ².
- **Note on Connor (1987, Eq. 3, via SAS PROC POWER).** The original is one-sided z_{1−α}, and SAS inverts the summed
  one-sided powers for two-sided tests. Our z_{1−α/2} is the usual two-sided approximation; say so.
- **Harness.**
  - Exact power at Connor's n is 0.7756–0.7870 across four cases (Connor is "slightly conservative" per SAS, and here
    slightly anti-conservative).
  - Finance near-tie, NB-weighted bigram vs. LR bigram: 0.776000 vs. 0.771333 accuracy; 108 vs. 101 discordant;
    p = 0.678222; power at 1,500 filings 0.067070; 80% power needs **50,215 filings**.
  - Card et al. (2020) estimate McNemar power by simulation (Fig. 2) and do not cite Connor.

**Prop 5 (metric shopping).**
- **Statement.** "Any of k tests rejects at α" has size ≤ kα (Bonferroni, cited from formalCalculus Cor 1 and
  formalStatistics Thm 1). The size is 1 − (1 − α)^k if the tests are independent, and α if the metrics are comonotone.
- **Proof (full).** The union bound, plus the two extreme couplings.
- **Harness.** On the A/A null's two scored streams (1,152 experiments at n = 10, 576 at 20), testing corpus BLEU plus
  Topic 27's six segment metrics:
  - at n = 10, the size rises from 0.050347 (k = 1) to 0.160590 (k = 7);
  - at n = 20, from 0.031250 to 0.149306;
  - both stay below 1 − 0.95⁷ = 0.302 because the metrics are positively dependent.
- **Also: Holm and BH** over the 45 summarizer pairs (`ck.bh_threshold`, Topic 26).

## 3. Findings (harness, 6 decimals)

- **Two clouds (90 filings).** δ, AR p, Koehn, 2δ, sd boot, sd linearized:

  | Pair | δ | AR p | Koehn | 2δ | sd boot | sd lin |
  |---|---|---|---|---|---|---|
  | writer / noisy-0.1 | 0.006828 | 0.000500 | 0.000000 | 0.004000 | 0.002166 | 0.002158 |
  | noisy-0.3 / noisy-0.5 | 0.017685 | 0.000250 | 0.000000 | 0.003000 | 0.005689 | 0.005697 |
  | terse / flipped | −0.013671 | 0.290177 | | | | |
  | extract / textrank-3 | 0.006112 | 0.201700 | | | | |
  | textrank-3 / random-3 | 0.001948 | 0.675331 | | | | |

  At 10 filings, writer vs. noisy-0.1 has AR p = 1.000000: only one filing differs, so every swap gives the same |δ|.
- **Summarizer matrix (45 pairs).** Randomization rejects 38 corpus-BLEU gaps and 41 fact-F1 gaps, the same under Holm
  and BH: the p-values are either at the 10⁻⁴ floor or large. **The metric decides which differences are
  significant** (`test_case_study_the_metric_decides`):
  - 9 pairs are significant on one metric only.
  - 8 more are significant on both, in opposite directions: three of the four flipped-vs.-extractor pairs (flipped vs.
    lead-3 is BLEU-only, both fact F1s being 0).
  - 27 agree, and 1 pair (textrank-3 vs. random-3) is significant on neither.

  So BLEU and fact F1 reach different verdicts on 17 of 45 pairs. Examples:
  - Writer vs. extract: BLEU +0.359800, p = 0.0001, yet fact F1 is identical (δ 0, p = 1).
  - Terse vs. flipped: BLEU −0.0137 (p 0.3029), but F1 +0.6672 (p 0.0001).
  - Lead-3 vs. random-3: BLEU −0.0004 (p 0.8937), but F1 −0.2771 (p 0.0001).
- **Subsampling power** (rejection rate over 200 random subsets of n filings):

  | Comparison | Metric | n = 5 | 10 | 15 | 20 | 30 | 45 |
  |---|---|---|---|---|---|---|---|
  | writer vs. noisy-0.1 | BLEU | 0 | 0 | 0.025 | 0.085 | 0.41 | 0.84 |
  | writer vs. noisy-0.1 | chrF | 0 | 0 | 0.01 | 0.045 | 0.11 | 0.205 |
  | writer vs. noisy-0.1 | F1 | 0 | 0.02 | 0.205 | 0.52 | 0.955 | 1.0 |

  chrF averages character n-grams, and a flipped verb moves few of them, so it is the least powerful at detecting fact
  flips.
- **Classifier pairs (McNemar).**

  | Pair | n₁₀ / n₀₁ | p | Filings for 80% power |
  |---|---|---|---|
  | LR-uni vs. NB-uni | 295 / 88 | < 10⁻⁶ | 103 |
  | LR-bi vs. LR-uni | 82 / 33 | 0.000006 | 562 |
  | NBW-bi vs. LR-bi | 108 / 101 | 0.678222 | 50,215 |
  | NBW-bi vs. NBW-uni | 119 / 33 | < 10⁻⁶ | 240 |

## 4. The lab: `SignificanceTestingLab` (6 tabs; every number from `src/components/viz/data/significance-testing-lab.json`, 18 KB)

1. **Two clouds** (`clouds`): 5 pairs × n ∈ {10, 30, 90}. The bootstrap and AR histograms share 40 bins on a common
   range, with δ̂ marked. Readouts: p_AR, p_Koehn, p_2δ, and the bootstrap vs. linearized sd. Lesson: the two clouds
   have different centers and the same spread to first order. The degenerate extractor cloud is where the 2δ rule
   breaks.
2. **McNemar** (`mcnemar`): the four classifier pairs' discordance counts, the exact p, and the power curve over a
   40-point log grid from 100 to 200,000 filings. Mark n = 1,500 and Connor's n.
3. **Power** (`power`): rejection rate vs. n for the three flip rates × three metrics.
4. **A/A** (`aa`): p-value histograms (20 bins) for AR, Koehn and 2δ at each n; size at 0.05 with the 3σ band; the
   metric-shopping curve over k = 1..7 at n ∈ {10, 20}.
5. **Clusters and training draws** (`clusters`, `draws`):
   - a design-effect slider for m and ρ, with the aspect-row markers;
   - Thm 4's 40 draws as a scatter of population Δ against test-set p, on 3,000 and on 20,000 documents, with the
     α line.
6. **Multiplicity** (`matrix`): the 10 × 10 system matrix with a BLEU/F1 toggle and raw/Holm/BH views. Cells where
   the two metrics disagree are outlined.

## 5. Checkpoint answers (Jonathan, 2026-10-27)

At the scaffold checkpoint Jonathan accepted every recommendation:
- **A/A null.** 128 experiments at full size. The block design turned these into 2,304 independent experiments at 10
  prompts.
- **Sidedness.** One-sided for Prop 2, two-sided for the matrix.
- **Prop 3 evidence.** A Gaussian toy, plus the flip-family remark.
- **ASO.** Cited only.

**Open for Phase B.** sacreBLEU's paired tests seed their own stream (`SACREBLEU_SEED`, default 12345) and use the
R&M/Noreen shift bootstrap, so no cross-check can be exact. Phase A therefore has none. A Monte Carlo-tolerance check
of its AR on terse vs. flipped would be the only option; Jonathan to decide.

## 5b. rigorFlag (draft)

> Every test on this page conditions on one trained model per system and one test set, and Theorem 4 shows that a test
> set cannot answer whether a training procedure is better. Randomization's exactness (Theorem 1) is for the sharp null
> that each segment's two outputs are exchangeable, a stronger hypothesis than "equal expected BLEU." The bootstrap's
> validity is asymptotic: at 10 prompts the 2δ rule's A/A size is 0.069, at the brevity penalty's kink the bootstrap is
> inconsistent, and with a handful of matched 4-grams the delta method is off by 11%. Connor's sample size is a normal
> approximation (one-sided in the original). Topic 27's systems, Topic 17's classifiers, and Topic 19's readers are
> synthetic, so the numbers illustrate the theorems, not field practice. A significant difference of 0.0068 BLEU is not
> an important one. Significance belongs to the metric: corpus BLEU and fact F1 reach different verdicts on 17 of the
> 45 summarizer pairs. Report effect sizes beside p-values.

## 6. Notebook cell map (28 cells, `01_significance_testing_for_nlp.ipynb`)

| Cells | Content |
|---|---|
| 1–2 | Title and framing; imports |
| 3–4 | Def 1 (pooled BLEU per system, c/r, matched 4-grams) |
| 5–6 | The two clouds |
| 7–9 | Thm 1 (A/A sizes, superuniformity, full group, scipy) |
| 10–11 | Prop 1 (classifier pairs) |
| 12–13 | Prop 2 (inversions, A/A bootstrap sizes) |
| 14–15 | Thm 2 (gradient, linearization, remainder) |
| 16–17 | Prop 3 (kink table) |
| 18–19 | Thm 3 (design effects) |
| 20–21 | Thm 4 (training draws) |
| 22–24 | Props 4–5 (power, shopping, subsampling power) |
| 25–26 | The summarizer matrix |
| 27–28 | The whole harness |

## 7. References (verified 2026-10-27 from the Anthology PDFs, Crossref, and publisher records; Chicago notes-bibliography)

**Verification caveats:**
- **Bickel–Freedman.** Unreadable (Project Euclid bot check), so cite it without a theorem number.
- **Noreen.** The (nc + 1)/(nt + 1) formula is verified only through Yeh (2000) and Riezler and Maxwell (2005).
- **Connor.** Paywalled; the formula is verified via SAS PROC POWER documentation.
- **Andrews.** Pages are 399–405 per Crossref (RePEc says 399–406); the example comes from Cowles DP 1157.
- **Fang and Santos.** Numbers are from arXiv v2.
- **Kish.** Only p. 162, via secondary sources.
- **Efron and Tibshirani.** Their book does not say "basic interval"; the term is Davison and Hinkley's (§5.2).
- **Søgaard et al.** The PDF byline reads "Hector Martinez"; the Anthology reads "Héctor Martínez Alonso".

**Bibliography:**
- Andrews, Donald W. K. "Inconsistency of the Bootstrap When a Parameter Is on the Boundary of the Parameter Space." *Econometrica* 68, no. 2 (2000): 399–405. https://doi.org/10.1111/1468-0262.00114.
- Benjamini, Yoav, and Yosef Hochberg. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *JRSS B* 57, no. 1 (1995): 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x.
- Berg-Kirkpatrick, Taylor, David Burkett, and Dan Klein. "An Empirical Investigation of Statistical Significance in NLP." In *EMNLP-CoNLL 2012*, 995–1005. https://aclanthology.org/D12-1091/.
- Bickel, Peter J., and David A. Freedman. "Some Asymptotic Theory for the Bootstrap." *Annals of Statistics* 9, no. 6 (1981): 1196–1217. https://doi.org/10.1214/aos/1176345637.
- Card, Dallas, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, and Dan Jurafsky. "With Little Power Comes Great Responsibility." In *EMNLP 2020*, 9263–74. https://doi.org/10.18653/v1/2020.emnlp-main.745.
- Connor, Robert J. "Sample Size for Testing Differences in Proportions for the Paired-Sample Design." *Biometrics* 43, no. 1 (1987): 207–11. https://doi.org/10.2307/2531961.
- Davison, A. C., and D. V. Hinkley. *Bootstrap Methods and Their Application*. Cambridge University Press, 1997. https://doi.org/10.1017/CBO9780511802843.
- Dietterich, Thomas G. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." *Neural Computation* 10, no. 7 (1998): 1895–1923. https://doi.org/10.1162/089976698300017197.
- Dror, Rotem, Gili Baumer, Segev Shlomov, and Roi Reichart. "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing." In *ACL 2018*, 1383–92. https://doi.org/10.18653/v1/P18-1128.
- Dror, Rotem, Segev Shlomov, and Roi Reichart. "Deep Dominance - How to Properly Compare Deep Neural Models." In *ACL 2019*, 2773–85. https://doi.org/10.18653/v1/P19-1266.
- Efron, Bradley, and Robert J. Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall, 1993. https://doi.org/10.1007/978-1-4899-4541-9.
- Fang, Zheng, and Andres Santos. "Inference on Directionally Differentiable Functions." *Review of Economic Studies* 86, no. 1 (2019): 377–412. https://doi.org/10.1093/restud/rdy049.
- Field, C. A., and A. H. Welsh. "Bootstrapping Clustered Data." *JRSS B* 69, no. 3 (2007): 369–90. https://doi.org/10.1111/j.1467-9868.2007.00593.x.
- Holm, Sture. "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian Journal of Statistics* 6, no. 2 (1979): 65–70. https://www.jstor.org/stable/4615733.
- Koehn, Philipp. "Statistical Significance Tests for Machine Translation Evaluation." In *EMNLP 2004*, 388–95. https://aclanthology.org/W04-3250/.
- McNemar, Quinn. "Note on the Sampling Error of the Difference between Correlated Proportions or Percentages." *Psychometrika* 12, no. 2 (1947): 153–57. https://doi.org/10.1007/BF02295996.
- Noreen, Eric W. *Computer-Intensive Methods for Testing Hypotheses: An Introduction*. Wiley, 1989. (ISBN 0-471-61136-0; needs a catalog URL.)
- Phipson, Belinda, and Gordon K. Smyth. "Permutation P-values Should Never Be Zero." *SAGMB* 9, no. 1 (2010): article 39. https://doi.org/10.2202/1544-6115.1585.
- Reimers, Nils, and Iryna Gurevych. "Reporting Score Distributions Makes a Difference." In *EMNLP 2017*, 338–48. https://doi.org/10.18653/v1/D17-1035.
- Riezler, Stefan, and John T. Maxwell III. "On Some Pitfalls in Automatic Evaluation and Significance Testing for MT." In *ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization*, 57–64. 2005. https://aclanthology.org/W05-0908/.
- Søgaard, Anders, Anders Johannsen, Barbara Plank, Dirk Hovy, and Héctor Martínez Alonso. "What's in a p-value in NLP?" In *CoNLL 2014*, 1–10. https://doi.org/10.3115/v1/W14-1601.
- Yeh, Alexander. "More Accurate Tests for the Statistical Significance of Result Differences." In *COLING 2000*. https://arxiv.org/abs/cs/0008005.
