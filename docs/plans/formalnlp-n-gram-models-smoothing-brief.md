# Brief — `n-gram-models-smoothing`

**n-Gram Language Models and Smoothing: From Laplace to Kneser–Ney.** Roadmap #3; prerequisite:
`zipf-heaps-vocabulary-statistics` (published). The first `language-modeling` topic and the prerequisite
of `perplexity-cross-entropy`, `sequence-labeling-hmm-crf`, `collocations-keyword-extraction`, and
`overlap-metrics-bleu-rouge`.

- Phase A (this brief + verified notebook): 2026-10-23. Harness 27/27 under Python 3.15, numpy==2.5.4,
  scipy==1.18.1, nltk==3.10.4 in ≈ 15 s; narrative notebook executes (exit 0, ≈ 28 s, committed without
  outputs).
- Scaffold decisions (Jonathan, 2026-10-23): the theory arc runs through HPYLM — smoothing read three
  ways as empirical Bayes; **one lab, two linked panels**; the KN core (`continuation_counts`,
  `interpolated_kn_prob`) was to be Jonathan's, handed back ("write both KN bodies yourself");
  formalStatistics `empirical-bayes` is citable (Topic 33, jonx0037/formalStatistics#53, merged
  2026-10-23 04:52 UTC, `status: "published"`).
- Source of truth: `notebooks/n-gram-models-smoothing/n_gram_models_smoothing.py`. Every number below is
  printed or asserted by it; nothing here may be changed without changing it.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | n-Gram Language Models and Smoothing: From Laplace to Kneser–Ney |
| `{{SUBTITLE}}` | Every smoother is a guess about the words we have not seen — read three ways as empirical Bayes: a Dirichlet prior fit to the data, Robbins' formula on word counts, and Kneser–Ney as a hierarchical Pitman–Yor posterior. |
| `{{SLUG}}` | `n-gram-models-smoothing` (= graph node id) |
| `{{DOMAIN}}` | `language-modeling` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[zipf-heaps-vocabulary-statistics]` |
| `{{FORMALML_PREREQS}}` | `kl-divergence`, `bayesian-nonparametrics` |
| `{{FORMALRAG_PREREQS}}` | `query-likelihood-language-models` |
| `{{STATS_CALC_PREREQS}}` | Stats `maximum-likelihood`, `hierarchical-bayes-and-partial-pooling`, `empirical-bayes`; Calc none (no Lagrange-multiplier topic; `convex-optimization` is still `draft`) |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/zipf-heaps-vocabulary-statistics/`, `notebooks/subword-tokenization-bpe-unigram/`, `notebooks/finance-corpus/` |

Tags: `n-gram`, `smoothing`, `good-turing`, `kneser-ney`, `empirical-bayes`, `pitman-yor`,
`language-model`. `estimatedReadTime` ≈ 45. In-site `connections`: none (dependents still planned). The
graph edge `zipf-heaps-vocabulary-statistics → n-gram-models-smoothing` already exists.

**Scope against formalRAG `query-likelihood-language-models` (overlap watch-list).** RAG owns document
unigram language models for ranking: the zero-frequency catastrophe in retrieval, Jelinek–Mercer and
Dirichlet smoothing of P(t | d), *Dirichlet smoothing is a posterior mean* (its Theorem 1), the KL ranking
view, the IDF-like effect, and length adaptivity. We **cite** its Theorem 1 and do not re-prove it, and
cite it for Jelinek–Mercer. Our object is the *conditional* next-word model P(w | u), recursive over
orders, and what RAG does not cover: Type-II fitting of the Dirichlet strength, Good–Turing, backoff,
discounting, Kneser–Ney, and the hierarchical Pitman–Yor reading. Held-out quality is reported as
log-loss in nats per token; perplexity is foreshadowed only (`perplexity-cross-entropy` owns it).

**Scope against formalStatistics `empirical-bayes` (Topic 33).** Robbins' formula and its proof (§33.3,
Theorem 2), the compound decision frame (§33.2), and Robbins' instability where counts are small
(Remark 5) are Topic 33's; we cite them and spend our proofs on the word-count instance (Theorem 2's
exact binomial identity) and on Kneser–Ney.

## 1. Section outline (geometric first; each section opens with a picture or a corpus number)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 1 | The zero-count problem | 32.47% of held-out trigrams (22.17% of bigrams) never occur in training | Def 1; **Prop 1** (MLE, KL gap), **Prop 2** (catastrophe) | cells 0–3 |
| 2 | Smoothing as shrinkage | the MLE on a face of the simplex, pulled inward along a line | Def 2; RAG Thm 1 (cited); **Prop 3** (interpolation); **Thm 1** (evidence) → Type-II α̂ vs. oracle | cells 4–6 |
| 3 | Good–Turing: Robbins' formula on word counts | the trigram spectrum N₁…N₅ = 19,521 / 1,736 / 748 / 444 / 309 | **Thm 2** (exact identity), **Cor 1** (missing-mass bias); Robbins (cited, T33); SGT (heuristic) | cells 6–9 |
| 4 | Backoff | `in osaka ___` (7 tokens, always `</s>`) | Def 3 (Katz); **Prop 4** (normalization); the cutoff's zeros | cells 10–11 |
| 5 | Absolute discounting | SGT's r − r* ≈ 0.65–0.76 for r ≥ 2 | Def 4; **Prop 5** (normalization); **Thm 3** (leave-one-out) | cells 12–13 |
| 6 | Kneser–Ney | `industrial`: 284 tokens, one context | Def 5 (continuation counts); **Thm 4** (marginal constraint, general D(c)); IKN; MKN (heuristic) | cells 14–15 |
| 7 | Kneser–Ney as a hierarchical Pitman–Yor posterior | Topic 1's Pitman–Yor restaurant, one per context | Def 6 (HPYLM); **Prop 6** (one table per type, θ = 0 ⇒ IKN); Gibbs sampler (cited MCMC) | cells 16–19 |
| — | The smoothers side by side | the held-out table (§3 below) | — | cells 20–21 |
| 8 | Finance case study | `<FinanceCaseStudy>`: the Halvorsen effect; what Good–Turing forecasts | — | cells 22–23 |
| 9 | Where this goes next | — | log-loss → `perplexity-cross-entropy`; Markov chains → `sequence-labeling-hmm-crf`; continuation counts → `collocations-keyword-extraction`; n-gram overlap → `overlap-metrics-bleu-rouge`; n-grams over subwords (Topic 2 §7) | cells 24–26 |

Lab embed: one `<NGramSmoothingLab client:visible />`. Topic 1 and Topic 2 embed in the Overview; here
Panel A needs §§2–6 to read fully and Panel B needs §§3, 5, 7, so embed after §7 (before the side-by-side
table), with forward pointers from §§2 and 4.

## 2. Definitions

Throughout: a sentence is padded with n − 1 copies of `<s>` in front and one `</s>` behind; every word and
the final `</s>` is a predicted token, `<s>` never is. V = training word types ∪ {`</s>`, `<unk>`}
(|V| = 3,427); held-out tokens outside V become `<unk>`. A context u is an (n − 1)-gram, u⁻ drops its
oldest word, c(u w) counts the n-gram u w, c(u ·) = Σ_w c(u w), and N₁₊(u •) = |{w : c(u w) > 0}|. All
lower-order counts are marginals of the top-order events (`test_counts_are_marginals`). n = 3.

- **Def 1 (order-n model, MLE).** P(w_i | w_{i−n+1:i−1}); P̂(w | u) = c(u w)/c(u ·), 0 for unseen u.
- **Def 2 (hierarchical Dirichlet; MacKay–Peto).** P_k(w | u) = (c(u w) + α_k P_{k−1}(w | u⁻))/(c(u ·) + α_k),
  P_1(w) = (c(w) + α₁/|V|)/(N + α₁). Uniform base ⇒ add-δ (Lidstone), δ = α/|V|; Laplace at δ = 1.
- **Def 3 (Katz backoff).** Seen: d_r · c(u w)/c(u ·) with d_r = (r*/r − (k+1)N_{k+1}/N₁)/(1 − (k+1)N_{k+1}/N₁)
  for r ≤ k = 5, d_r = 1 above; unseen: α(u) P_katz(w | u⁻); Laplace at the unigram level.
- **Def 4 (interpolated absolute discounting).** P(w | u) = [max(c(u w) − D, 0) + D N₁₊(u •) P(w | u⁻)]/c(u ·),
  raw counts at every order; uniform 1/|V| under the unigram level; unseen u ⇒ P(w | u⁻).
- **Def 5 (continuation counts; interpolated KN).** N₁₊(• v) = |{x : c(x v) > 0}|. Interpolated KN is Def 4
  with raw counts at the top order and continuation counts at every lower order; discounts D_k(c) per
  order — one (Ney: n₁/(n₁ + 2n₂) of that order's counts) or three (modified KN, Chen–Goodman:
  D₁ = 1 − 2Y n₂/n₁, D₂ = 2 − 3Y n₃/n₂, D₃₊ = 3 − 4Y n₄/n₃, Y = n₁/(n₁ + 2n₂)). Implementation form:
  P(w | u) = [max(c − D_k(c), 0) + freed(u) P(w | u⁻)]/c(u ·), freed(u) = Σ_{seen w} D_k(c(u w)).
- **Def 6 (HPYLM; Teh 2006).** One Pitman–Yor Chinese restaurant per context u, base distribution the
  restaurant of u⁻; each table in u sends one customer to u⁻. Given a seating (c_uw customers and t_uw
  tables for w in u), P(w | u) = (c_uw − d_k t_uw)/(θ_k + c_u·) + (θ_k + d_k t_u·)/(θ_k + c_u·) · P(w | u⁻),
  uniform base.

## 3. Statements and proofs

### Proposition 1 (the MLE is the relative frequency)
For every distribution q on the followers of u, Σ_w c(u w) log P̂(w | u) − Σ_w c(u w) log q(w) =
c(u ·) KL(P̂ ‖ q) ≥ 0.

*Proof.* Substitute c(u w) = c(u ·) P̂(w | u): the difference is c(u ·) Σ_w P̂ log(P̂/q). Gibbs' inequality
(formalML `kl-divergence`) makes it ≥ 0 with equality iff q = P̂. ∎ **Asserted** on 50 random corpus
contexts × 5 random q (`test_mle_maximizes_likelihood`).

### Proposition 2 (the zero-frequency catastrophe)
The MLE's held-out log-loss is ∞ iff some held-out n-gram is unseen in training. *Proof:* a log-loss is a
sum of −log P̂, finite iff every P̂ > 0, i.e. every held-out n-gram has c > 0. ∎ Held out: 0.67% of
unigrams (OOV), 22.17% of bigrams, 32.47% of trigrams are unseen; on the seen trigrams alone the MLE is
finite (`test_zero_frequency_catastrophe`).

### Proposition 3 (shrinkage is interpolation)
Def 2 equals (1 − λ_u) P̂(w | u) + λ_u P_{k−1}(w | u⁻) with λ_u = α_k/(c(u ·) + α_k); λ_u = 1 for unseen u.

*Proof.* Divide numerator and denominator: c(u w)/(c(u ·) + α) = (1 − λ_u) P̂ and α P_{k−1}/(c(u ·) + α) =
λ_u P_{k−1}. ∎ *Reading:* the posterior mean (RAG Thm 1, cited) moves the MLE along the segment to the prior
mean, harder where the context has less evidence. **Asserted** on 3,000 held-out events
(`test_dirichlet_is_interpolation`).

### Theorem 1 (the Dirichlet–multinomial evidence)
With independent Dirichlet(α m) priors per context, m a distribution on V,
log p(data | α) = Σ_u [log Γ(α) − log Γ(α + c(u ·))] + Σ_{(u,w) seen} [log Γ(α m_w + c(u w)) − log Γ(α m_w)].

*Proof.* For one context, p(sequence | θ) = Π_w θ_w^{c(u w)}; integrate against the Dirichlet density:
∫ Π θ_w^{α m_w + c − 1} dθ / B(α m) = B(α m + c)/B(α m), with B(β) = Π Γ(β_w)/Γ(Σ β_w) and Σ_w α m_w = α.
Contexts are independent; unseen (u, w) contribute Γ(α m_w)/Γ(α m_w) = 1. ∎ **Cross-check:** plus the
multinomial coefficient, equals `scipy.stats.dirichlet_multinomial.logpmf` on 30 random instances
(`test_evidence_matches_scipy`).

**Type-II MLE (empirical Bayes, parametric).** Maximize Thm 1 order by order (m = the fitted lower
order): α̂ = (1,105.50, 9.589, 25.015). Each is an interior maximum, and the model beats Laplace
(`test_type2_alpha_is_a_maximum`). The held-out oracle (coordinate descent on held-out log-loss) is
α = (68,911.6, 591.44, 6.974). Log-loss: Laplace 5.7859, Dirichlet Type-II 3.5335, Dirichlet oracle 3.2427.
*Reading (Remark):* the evidence is computed on training text, which contains no `<unk>` events, so it
cannot pay for unseen words; the oracle smooths the lower orders far harder because 0.67% of held-out
tokens are new. Link T28 §28.7 (Type-II MLE; EB intervals too narrow — the same blind spot).

### Theorem 2 (Good–Turing, exact)
For n i.i.d. draws from any p and every r ≥ 0, E[M_r(n)] = (r + 1)/(n + 1) · E[N_{r+1}(n + 1)], where N_r(n)
counts types seen exactly r times and M_r(n) is their total probability.

*Proof.* E[M_r(n)] = Σ_w p_w C(n, r) p_w^r (1 − p_w)^{n−r} = Σ_w C(n, r) p_w^{r+1}(1 − p_w)^{(n+1)−(r+1)}; and
(r + 1)/(n + 1) · C(n + 1, r + 1) = C(n, r), so this is (r + 1)/(n + 1) Σ_w C(n + 1, r + 1) p_w^{r+1}(1 − p_w)^{n−r} =
(r + 1)/(n + 1) E[N_{r+1}(n + 1)]. ∎ (Mixtures of i.i.d. — exchangeable sequences — inherit it: both
sides are linear in the law.) **Asserted** exactly (≤ 1e-12 relative) on 40 random p, n ≤ 300, r ≤ 5
(`test_good_turing_identity_exact`).

### Corollary 1 (missing-mass bias)
0 ≤ E[N₁(n)]/n − E[M₀(n)] = Σ_w p_w²(1 − p_w)^{n−1} ≤ 1/n.

*Proof.* Expand both expectations; the difference is Σ_w p_w(1 − p_w)^{n−1} p_w. Each p(1 − p)^{n−1} ≤
max_p p(1 − p)^{n−1} = (1/n)(1 − 1/n)^{n−1} ≤ 1/n, and Σ p_w = 1. ∎ **Asserted** exactly on 40 random p and
by simulation (400 samples of n = 2,000 from Zipf(1.1) on 5,000 types, within 4 standard errors)
(`test_missing_mass_bias_bound`). Concentration of the missing mass (McAllester–Ortiz 2003) is cited.

**Robbins' formula (cited, T33 §33.3 Thm 2).** Poissonize, c_w ~ Poisson(λ_w), and read the spread of λ_w
across types as g: E[λ | c = r] = (r + 1) f(r + 1)/f(r), f(r) ∝ E[N_r]. So r* = (r + 1)N_{r+1}/N_r *is*
Robbins' estimator, with word types for policy holders (T33 §33.2's compound-decision frame). **Asserted**:
our `turing_rstar` on T33's CASI Table 6.1 claims reproduces T33's printed Robbins row 0.168, 0.363, 0.527,
1.33, 1.43, 6.00, 1.75; and the Poissonized identity holds exactly on 20 random λ vectors
(`test_good_turing_is_robbins`).

**Simple Good–Turing (heuristic; Gale–Sampson 1995).** Turing's r* is erratic where N_{r+1} is small
(T33 Rem 5): trigram r − r* = 0.822, 0.707, 0.626, 0.520, 1.039, −0.382, 2.484, −0.914 (r = 1…8). SGT averages
Z_r = 2N_r/(t − q), fits log Z_r on log r (slope −1.846), and switches from Turing's r* to the fit at the
first r where they agree within 1.96 s.e. (r = 3), renormalizing the seen mass to 1 − N₁/N. Its trigram
r − r*: 0.822, 0.707, 0.648, 0.688, 0.714, 0.733, 0.747, 0.758 — nearly constant (Church–Gale 1991), the
observation behind absolute discounting. The log-linear form is made plausible by Topic 1's power-law
spectrum; not a theorem. **Cross-check:** equals `nltk.probability.SimpleGoodTuringProbDist` on the bigram
spectrum to 1e-15, seen probabilities and unseen mass (`test_sgt_matches_nltk`).

### Proposition 4 (Katz normalizes)
α(u) = [1 − Σ_{seen w} d c(u w)/c(u ·)] / [1 − Σ_{seen w} P_katz(w | u⁻)] makes P_katz(· | u) sum to 1 whenever
the denominator is positive.

*Proof.* Seen words carry Σ_seen d c/c(u ·); unseen words carry α(u) Σ_unseen P_katz(w | u⁻) =
α(u)[1 − Σ_seen P_katz(w | u⁻)] by induction on the order (the unigram Laplace level sums to 1). Substitute α. ∎
**Asserted** for 16 contexts incl. unseen and padded ones; all d_r ∈ (0, 1): bigram d = 0.158, 0.616, 0.765,
0.823, 0.944, trigram 0.123, 0.623, 0.777, 0.861, 0.778 (`test_katz_normalizes`).

*The cutoff's zeros (finding, asserted).* 2,170 of 17,269 contexts on this templated corpus reserve no
mass (α = 0): in 564 every follower's count exceeds k, so nothing is discounted; in 1,606 the denominator
is 0 as well — every word the lower order can predict was already seen after u — where Katz is undefined
and we keep the MLE. Three held-out events — after `across
aerospace`, `in osaka`, `of fairhaven` — get probability 0, so **Katz's held-out log-loss is ∞**; on the other
23,052 events it is 3.1103 (`test_katz_cutoff_reintroduces_zeros`). On EDGAR, 87 events are zeroed.

### Proposition 5 (absolute discounting normalizes)
Σ_w max(c(u w) − D, 0) = c(u ·) − D N₁₊(u •) for D ≤ 1, so the numerator sums to c(u ·) when P(· | u⁻) does;
induct down to the uniform base. ∎ **Asserted** with interpolated KN in `test_ikn_normalizes` (absolute
discounting is `kn_fit(..., continuation=False)`).

### Theorem 3 (the leave-one-out discount)
Backing-off absolute discounting with a fixed lower-order β: P(w | u) = (c − D)/c(u ·) if c = c(u w) > 0, else
D N₁₊(u •)/c(u ·) · β(w)/Σ_{unseen w'} β(w'). Its leave-one-out log-likelihood is
F(D) = n₁′ log D + Σ_{r ≥ 2} r n_r log(r − 1 − D) + C, where n₁′ counts singletons whose context has
c(u ·) ≥ 2 and C does not depend on D. F is strictly concave on (0, 1); if n₁′, n₂ > 0 it has a unique
maximizer D*, and D* ≤ n₁′/(n₁′ + 2n₂) ≤ n₁/(n₁ + 2n₂).

*Proof.* Removing one occurrence of (u, w) with count r ≥ 2 leaves (r − 1 − D)/(c(u ·) − 1); with r = 1 and
c(u ·) ≥ 2 it leaves D(N₁₊(u •) − 1)/(c(u ·) − 1) · β(w)/(1 − S_u + β(w)), S_u = Σ_{seen w'} β(w'), D-free but for
the factor D; with r = 1 and c(u ·) = 1 the context vanishes and the probability is β(w). Summing logs gives F.
Each term is concave and n₁′ log D is strictly so. F′(D) = n₁′/D − Σ_{r≥2} r n_r/(r − 1 − D) → +∞ as D → 0
and → −∞ as D → 1 (the r = 2 term), so its root D* is unique. At D*, n₁′/D* = Σ_{r≥2} r n_r/(r − 1 − D*) ≥
2n₂/(1 − D*), i.e. D* ≤ n₁′/(n₁′ + 2n₂); and x/(x + 2n₂) increases in x with n₁′ ≤ n₁. ∎

Ney, Essen, and Kneser's estimate n₁/(n₁ + 2n₂) is this upper bound, not the optimum. Corpus: bigram
D* = 0.7019 (n₁′ = 9,040 of n₁ = 10,917; bound 0.7734; Ney 0.8048); trigram D* = 0.6443 (n₁′ = 9,215 of
19,521 — over half the trigram singletons are alone in their context; bound 0.7263; Ney 0.8490).
**Asserted**: concavity, uniqueness, and the bound on both corpus orders and 200 random spectra
(`test_loo_concave_and_bound`); F differs from the exact leave-one-out log-likelihood (Laplace β) by a
constant across D ∈ {0.1, …, 0.9} (`test_loo_objective_is_exact`).

### Theorem 4 (the marginal constraint)
Take the top order interpolated as in Def 5 with discounts D(c), c − D(c) ≥ 0 for c ≥ 1, and a lower-order
distribution β(· | v). The model preserves the empirical marginals, Σ_x c(x v ·) P(w | x v) = c(v w) for all
v, w, iff β(w | v) = Σ_x D(c(x v w)) / Σ_x freed(x v). With one discount per order this is
N₁₊(• v w)/N₁₊(• v •), the continuation distribution.

*Proof.* Σ_x c(x v ·) P(w | x v) = Σ_x [c(x v w) − D(c(x v w))] + β(w | v) Σ_x freed(x v), and Σ_x c(x v w) = c(v w)
(marginals). The constraint holds iff β(w | v) Σ_x freed(x v) = Σ_x D(c(x v w)). With D(c) = D·[c > 0],
Σ_x D(c(x v w)) = D N₁₊(• v w) and Σ_x freed(x v) = D N₁₊(• v •). ∎ (Kneser and Ney 1995 derive the
backing-off version approximately; in the interpolated form it is exact.)

**Asserted**: with the lower orders undiscounted, interpolated KN preserves bigram and trigram marginals to
1e-8 relative on 25 + 25 samples (`test_kn_marginal_constraint`); with modified-KN discounts, the
discount-weighted β restores the bigram marginals exactly while continuation counts miss by > 1%
(`test_marginals_need_discount_weighted_lower_order`) — so modified KN's lower orders are a heuristic.
Continuation counts agree with brute force on 30 random corpora with planted zero entries, and
`industrial` has N₁₊(• w) = 1 against a raw count of 284 (`test_continuation_counts_bruteforce`).
Interpolated KN, modified KN, and absolute discounting sum to 1 over V for 15 contexts × 3 orders
(`test_ikn_normalizes`). **Cross-check:** with one D = 0.75 at every order and D₁ = 0 (nltk's
configuration: an undiscounted continuation unigram, no uniform base), fed our own n-grams, our
recursion equals `nltk.lm.KneserNeyInterpolated` to 1e-12 on 60 held-out queries (`test_ikn_matches_nltk`).

### Proposition 6 (Kneser–Ney is the one-table HPYLM)
Seat every word at exactly one table in every restaurant, t_uw = min(1, c_uw), with the bottom restaurants
holding the raw counts, and set θ_k = 0. Then (i) the customers of restaurant u at order k < n are the
continuation counts N₁₊(• u w), and (ii) the HPYLM predictive equals interpolated KN with D_k = d_k.

*Proof.* (i) Induct downward. Restaurant u receives one customer per table of each child x u, i.e. per x with
c_{xu,w} > 0. At the level below the top that is N₁₊(• u w) from raw counts; below it, c_{xu,w} > 0 iff
N₁₊(• x u w) > 0 iff c(x u w) > 0 (marginals), so the count is again N₁₊(• u w). (ii) Substitute θ = 0 and
t = min(1, c) in Def 6: (c_uw − d[c_uw > 0] + d N₁₊(u •) P(w | u⁻))/c_u· — Def 5 with D = d, freed(u) = d N₁₊(u •),
the same uniform base. ∎ **Asserted** to 1e-13 on 300 held-out queries with d = (0.6, 0.7, 0.8), and the
seating's customer counts equal KN's counts at orders 1 and 2 (`test_hpylm_one_table_is_ikn`).

**The Gibbs sampler (cited MCMC; Teh 2006).** Re-seat every training token in turn (remove; re-add with
weights c_k − d for an existing table, (θ + d t_u·) P(w | u⁻) for a new one, recursing to the parent when a table
opens or closes), then draw (d_k, θ_k) per order from a 39 × 34 grid posterior (Beta(1, 1) on d, Gamma(1, 1) on
θ) given the seating's Pitman–Yor EPPF. Start from the one-table (KN) seating; 40 sweeps, predictive
averaged over sweeps 21–40. Posterior means d = (0.825, 0.875, 0.775), θ = (0.962, 0.0026, 0.0017): the
higher orders settle at θ ≈ 0, KN's pure-discounting regime. Effective trigram discount d·E[t | c = r],
r = 1…8: 0.775, 1.167, 1.503, 1.787, 1.971, 2.004, 2.260, 2.657 — exactly d at r = 1 (a singleton has one
table), rising with r and past modified KN's D₃₊ = 0.984 by r = 3. Held-out log-loss 2.8220 (three other
seeds: 2.8222–2.8236). **Asserted**: seating invariants after sampling (`test_hpylm_sampler_invariants`);
the discount's growth, its value at r = 1, and θ₂, θ₃ < 0.05 (`test_hpylm_discount_grows_with_count`).
*Correction to the plan:* the sampled discount does not "grow slowly" — it grows far beyond modified KN's
steps, while Good–Turing's measured discount stays ≈ 0.7. The two are not the same quantity: in the HPYLM
the subtracted mass returns through the parent, which in a deterministic context predicts the same word.

### The smoothers side by side (held-out log-loss, nats/token)

| smoother | in-domain (23,055 events) | EDGAR (830 events) |
|---|---|---|
| MLE | ∞ | ∞ |
| Laplace (add-1) | 5.7859 | **8.1326** |
| hierarchical Dirichlet, Type-II α̂ | 3.5335 | 10.7870 |
| hierarchical Dirichlet, oracle α | 3.2427 | — |
| Katz | ∞ (3.1103 on 23,052 events) | ∞ |
| absolute discounting (Ney D_k) | 3.1668 | 9.8499 |
| interpolated KN (Ney D_k) | 2.8551 | 8.7963 |
| modified KN | 2.8407 | 8.6022 |
| **HPYLM (sampled)** | **2.8220** | 8.6540 |

In domain the textbook ordering holds strictly; out of domain — 57.59% of EDGAR tokens are OOV — Laplace's
heavy uniform smoothing wins (`test_heldout_ordering`). One D at every order (the lab's slider): interpolated
KN bottoms out near D = 0.9 (2.8500; 2.8834 at D = 1), absolute discounting likewise (3.1591; 3.2236 at 1).

## 4. Finance case study (§8)

**The Halvorsen effect.** Company names are fixed phrases in the synthetic earnings-call corpus. The
frequent words with the fewest distinct predecessors are `management` (616 tokens, 2 contexts),
`industrial` (284, 1), `team` (234, 1), `unchanged` (232, 1), `steady` (210, 1), `plan` (209, 1). At the
unigram level, `industrial` is rank 27 with P = 0.0041 under absolute discounting and rank 1,259 with
P = 0.00007 under interpolated KN (a factor of 59; ties share a rank) (`test_halvorsen_effect`). In the lab's
`the ⟨unk⟩ ___` context the two models' top words differ the same way: absolute discounting `</s> and in
the of`; KN `and </s> a outlook which` — the words that follow many different contexts.

**What Good–Turing forecasts.** N₁/N = 0.02708 on the training tokens. Three test sets:

| test text | OOV share | vs. N₁/N |
|---|---|---|
| the training generator run on for 100 more documents (`continuation_run`) | 0.02752 | within 1.6% |
| a fresh run, `generate_corpus(100, 1)` | 0.00668 | 4× lower |
| EDGAR MD&A excerpts (public, out of domain) | 0.57590 | 21× higher |

Good–Turing estimates the probability that the *next* token from the *same* process is a new type. The
continuation is that process (the first 300 documents of the rebuild are `generate_corpus(300, 0)`
exactly, `test_continuation_rebuild_matches_corpus`); a fresh run restarts every Pitman–Yor restaurant
from the same seed nouns, so its early types were mostly seen; EDGAR is another source
(`test_good_turing_forecasts_the_continuation`). The estimate is exact about the right question.

## 5. `rigorFlag`

> The n-gram Markov assumption is false for language; these models are estimators, not truths. Katz's
> Good–Turing cutoff, Simple Good–Turing's log-linear fit, absolute discounting's constant D, and modified
> Kneser–Ney's three discounts are heuristics — only the leave-one-out bound D* ≤ n₁/(n₁ + 2n₂) is proved,
> and Ney's familiar estimate is that bound, not the optimum. Kneser–Ney's marginal matching is exact only
> for interpolated models with one discount per order; modified KN's lower orders do not preserve marginals.
> The HPYLM identity holds for one particular seating; the sampled HPYLM is MCMC run for 40 sweeps with no
> convergence guarantee. Good–Turing's unseen mass forecasts the continuation of the same process, not new
> text from another source. The corpus is synthetic and templated, so the ranking of smoothers is evidence
> about this corpus — out of domain it reverses.

## 6. Viz design intent — `NGramSmoothingLab` (one component, two linked panels)

All data from `viz_payload()` (≈ 37 kB); mirror to the decimal in
`src/components/viz/data/n-gram-smoothing-lab.json` with a Phase B `test_viz_mirror_matches_payload`
(the Topic 1/2 pattern). Marginalia tokens only.

**Panel A — "Smooth a context."** A context picker over `LAB_CONTEXTS`: `within industrials ___` (26 tokens,
7 follower types, all company names), `machinery as ___` (seen once), `in osaka ___` (7 tokens, always
`</s>`), `and components ___` (unseen trigram context), `the ⟨unk⟩ ___` (only the unigram level is left). A
smoother toggle: MLE (shows "undefined" for unseen contexts), Laplace, hierarchical Dirichlet (Type-II),
Katz, absolute discounting, interpolated KN, modified KN. The shared **D slider** {0.1, …, 1.0} drives
absolute discounting and interpolated KN (one D at every order). Main view: the top 8 next words as
horizontal bars, filled if seen after the context and hollow if not, then "other seen" and "never seen
here" bars. Readouts: unseen mass in this context, the watch word `industrial` (probability and rank), and
the whole model's held-out log-loss at this setting (Katz: "∞ — 3 events get probability 0; 3.110 on the
rest"; HPYLM 2.822 shown as a reference line). *Lessons:* each smoother is a different mass budget —
Katz gives `in osaka` nothing to spare and singletons lose most of their mass (`machinery as`: Katz keeps
0.123, interpolated KN at D = 0.7 keeps 0.776); KN starves `industrial` once it backs off; log-loss is
U-shaped in D.

**Panel B — "Discount versus count."** x = r = 1…8, y = the discount taken from a count-r trigram: Turing's raw
r − r* (points, visibly erratic from r = 5), Simple Good–Turing's (a curve near 0.7), interpolated KN's
constant D (a horizontal line that follows the shared slider; Ney's 0.849 marked), modified KN's three steps
(0.849, 0.903, 0.984), and the sampled HPYLM's d·E[t | r] (rising from 0.775 to 2.657). A toggle swaps in
the trigram spectrum N_r, r = 1…30, on log–log axes. *Lessons:* Good–Turing measures a nearly constant
discount; KN takes it constant, modified KN lets it climb a little; the Pitman–Yor posterior subtracts far
more from frequent n-grams and returns it through the parent.

**Link.** The shared D: moving it moves Panel B's KN line and every absolute-discounting / KN view in
Panel A; hovering a count r in Panel B highlights Panel A's bars whose training count after the context is r.
(Phase B adds per-word counts to the payload if the highlight ships.)

## 7. Notebook cell map (`01_n_gram_models_smoothing.ipynb`, 27 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; corpus, |V|, spectra | — |
| 2–3 | §1 Props 1–2 | `test_counts_are_marginals`, `test_mle_*`, `test_zero_frequency_*` |
| 4–5 | §2 Prop 3, Thm 1, Type-II vs. oracle | `test_dirichlet_*`, `test_evidence_*`, `test_type2_*` |
| 6–9 | §3 Thm 2, Cor 1, Robbins/CASI; SGT, r − r* | `test_good_turing_*`, `test_missing_mass_*`, `test_sgt_matches_nltk` |
| 10–11 | §4 Katz, Prop 4, the cutoff's zeros | `test_katz_*` |
| 12–13 | §5 Thm 3 | `test_loo_*` |
| 14–15 | §6 Thm 4, IKN, MKN, nltk | `test_continuation_*`, `test_ikn_*`, `test_kn_marginal_*`, `test_marginals_*` |
| 16–19 | §7 Prop 6; the Gibbs sampler | `test_hpylm_*` |
| 20–21 | the side-by-side table | `test_heldout_ordering` |
| 22–23 | §8 finance | `test_continuation_rebuild_*`, `test_good_turing_forecasts_*`, `test_halvorsen_effect` |
| 24–26 | §9 viz payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Run:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with nltk==3.10.4 python notebooks/n-gram-models-smoothing/n_gram_models_smoothing.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with nltk==3.10.4 --with jupyter --with ipykernel jupyter execute notebooks/n-gram-models-smoothing/01_n_gram_models_smoothing.ipynb
```

## 8. Cross-site links (all verified `status: "published"` on 2026-10-23)

| field | topic | relationship |
|---|---|---|
| `formalstatisticsPrereqs` | `maximum-likelihood` | The unsmoothed n-gram model is the maximum-likelihood estimate of each context's next-word multinomial, and its zero-frequency failure is what every smoother repairs. |
| `formalstatisticsPrereqs` | `hierarchical-bayes-and-partial-pooling` | The hierarchical Dirichlet language model shrinks each context toward the lower order and fits its prior strength by Type-II maximum likelihood, the parametric empirical Bayes of §28.7. |
| `formalstatisticsPrereqs` | `empirical-bayes` | Good–Turing's adjusted count (r+1)N_{r+1}/N_r is Robbins' formula (§33.3, Theorem 2) with word types as policy holders, and inherits its instability where counts are sparse. |
| `formalmlPrereqs` | `kl-divergence` | Relative frequencies maximize likelihood by Gibbs' inequality (the gap is the context count times a KL divergence), and smoothers are compared by held-out log-loss. |
| `formalmlPrereqs` | `bayesian-nonparametrics` | Kneser–Ney is read as an approximation to a hierarchical Pitman–Yor process — the two-parameter generalization of the Dirichlet process and its Chinese-restaurant representation. |
| `formalragPrereqs` | `query-likelihood-language-models` | Dirichlet smoothing as a posterior mean (its Theorem 1) and Jelinek–Mercer interpolation are proved there for document models; here they become the conditional next-word model. |

## 9. References (Chicago 17th, notes-and-bibliography; every entry has a URL)

1. Good, I. J. "The Population Frequencies of Species and the Estimation of Population Parameters."
   *Biometrika* 40, no. 3/4 (1953): 237–64. https://doi.org/10.1093/biomet/40.3-4.237. — paper; Thm 2.
2. Robbins, Herbert. "An Empirical Bayes Approach to Statistics." In *Proceedings of the Third Berkeley
   Symposium on Mathematical Statistics and Probability*, vol. 1, 157–63. Berkeley: University of California
   Press, 1956. https://projecteuclid.org/euclid.bsmsp/1200501653. — paper; Robbins' formula.
3. Efron, Bradley, and Ronald Thisted. "Estimating the Number of Unseen Species: How Many Words Did
   Shakespeare Know?" *Biometrika* 63, no. 3 (1976): 435–47. https://doi.org/10.1093/biomet/63.3.435. — paper;
   Good–Turing on words as empirical Bayes.
4. Efron, Bradley, and Trevor Hastie. *Computer Age Statistical Inference*. Cambridge: Cambridge University
   Press, 2016. https://doi.org/10.1017/CBO9781316576533. — book; Table 6.1 (via T33).
5. Katz, Slava M. "Estimation of Probabilities from Sparse Data for the Language Model Component of a
   Speech Recognizer." *IEEE Transactions on Acoustics, Speech, and Signal Processing* 35, no. 3 (1987):
   400–401. https://doi.org/10.1109/TASSP.1987.1165125. — paper; Def 3.
6. Church, Kenneth W., and William A. Gale. "A Comparison of the Enhanced Good–Turing and Deleted Estimation
   Methods for Estimating Probabilities of English Bigrams." *Computer Speech & Language* 5, no. 1 (1991):
   19–54. https://doi.org/10.1016/0885-2308(91)90016-J. — paper; nearly constant discounts.
7. Ney, Hermann, Ute Essen, and Reinhard Kneser. "On Structuring Probabilistic Dependences in Stochastic
   Language Modelling." *Computer Speech & Language* 8, no. 1 (1994): 1–38.
   https://doi.org/10.1006/csla.1994.1001. — paper; absolute discounting, Thm 3.
8. Kneser, Reinhard, and Hermann Ney. "Improved Backing-Off for M-Gram Language Modeling." In *Proceedings of
   ICASSP 1995*, vol. 1, 181–84. https://doi.org/10.1109/ICASSP.1995.479394. — paper; Thm 4.
9. Gale, William A., and Geoffrey Sampson. "Good–Turing Frequency Estimation without Tears." *Journal of
   Quantitative Linguistics* 2, no. 3 (1995): 217–37. https://doi.org/10.1080/09296179508590051. — paper; SGT.
10. MacKay, David J. C., and Linda C. Bauman Peto. "A Hierarchical Dirichlet Language Model." *Natural
    Language Engineering* 1, no. 3 (1995): 289–308. https://doi.org/10.1017/S1351324900000218. — paper; Def 2.
11. Pitman, Jim, and Marc Yor. "The Two-Parameter Poisson–Dirichlet Distribution Derived from a Stable
    Subordinator." *Annals of Probability* 25, no. 2 (1997): 855–900. https://doi.org/10.1214/aop/1024404422.
    — paper; the Pitman–Yor process.
12. Chen, Stanley F., and Joshua Goodman. "An Empirical Study of Smoothing Techniques for Language
    Modeling." *Computer Speech & Language* 13, no. 4 (1999): 359–94. https://doi.org/10.1006/csla.1999.0128.
    — paper; interpolated and modified KN.
13. McAllester, David, and Luis Ortiz. "Concentration Inequalities for the Missing Mass and for Histogram
    Rule Error." *Journal of Machine Learning Research* 4 (2003): 895–911.
    https://www.jmlr.org/papers/v4/mcallester03a.html. — paper; missing-mass concentration (cited).
14. Teh, Yee Whye. "A Hierarchical Bayesian Language Model Based on Pitman–Yor Processes." In *Proceedings of
    COLING/ACL 2006*, 985–92. https://aclanthology.org/P06-1124/. — paper; Def 6, Prop 6, the sampler.
15. Bird, Steven, Ewan Klein, and Edward Loper. "nltk.lm — Language Models." NLTK 3.10 documentation.
    https://www.nltk.org/api/nltk.lm.html. — documentation; cross-check.

## 10. Phase B notes

- MDX at `src/content/topics/n-gram-models-smoothing.mdx`; page numbering follows page order (Defs 1–6,
  Props 1–6, Thms 1–4, Cor 1; keep the brief's proofs). `references[].type`: 4 → `book`, 15 →
  `documentation`, the rest `paper`.
- Graph: flip node `status` to `published`; remove the title from `planned` in `curriculum.ts`. The edge from
  `zipf-heaps-vocabulary-statistics` already exists; frontmatter `prerequisites: [zipf-heaps-vocabulary-statistics]`,
  no new `connections`.
- Lab reads `src/components/viz/data/n-gram-smoothing-lab.json` (`json.dumps(viz_payload(), separators=(',', ':'))`);
  add `test_viz_mirror_matches_payload`. Python `inf` is emitted as `null` already (`_r`).
- `pnpm audit:cross-site` reads the sibling checkouts in place: the local formalStatistics checkout must
  contain merged Topic 33 (`git -C ../formalStatistics log -1 -- src/content/topics/empirical-bayes.mdx`), or
  point `FORMAL_STATISTICS_PATH` at an export of `origin/main` — never pull or switch branches in a sibling.
  The six outward links will report as missing reciprocals — expected until launch.
- Verify reference URLs resolve before shipping (Robbins' Project Euclid link; the JMLR v4 page).
- Doc hygiene: STARTER-PROMPT.md A4's "Known drafts at scaffold time" and the scaffold plan's "Blockers to
  watch" line — `empirical-bayes` is published.

## 11. Phase B record (2026-10-23)

- Page numbering matches the brief (Defs 1–6, Props 1–6, Thms 1–4, Cor 1); remarks unnumbered. The lab
  is embedded after §7 as planned, opening "The smoothers side by side"; the Overview points forward to it.
- The payload's top-word entries became `[word, probability, training count c(u w)]` (was a seen flag) so
  Panel B's hover highlight ships: hovering r lights the words seen exactly r times after the context.
  `test_viz_mirror_matches_payload` requires `src/components/viz/data/n-gram-smoothing-lab.json` to equal
  `viz_payload()` exactly, and `test_viz_payload` now asserts the slider's interior minimum (harness 28/28).
- Long inline number lists were split into one inline formula per number: a single KaTeX span cannot wrap
  and overflowed a 375-px viewport.
- Gates: `pnpm test` 39/39; `astro sync` and `pnpm validate` clean; `pnpm audit:cross-site` exits 0 with
  the six outward links as missing reciprocals (expected) and no deferred or drifted targets; `pnpm build`
  exits 0. The local formalStatistics checkout already held published Topic 33, so no export was needed.
- DOM checks on the built page (foreground pane, desktop width): 0 `.katex-error`, 414 `.katex`; the island
  hydrates (10 bars, 5 Panel B series, 8 hover targets). Readouts equal the payload for interpolated KN at
  D = 0.7, 0.9, 1.0 (log-loss 2.889, 2.850, 2.883), Katz on `in osaka` (unseen mass 0, "∞ — 3 events"),
  `the ⟨unk⟩` (`industrial` #27 under absolute discounting, #1,259 under KN), the MLE's undefined state, and
  the r = 2 hover (lights `halvorsen`, `marrow`, `ostrander`). At 375 px no visible element overflows.
- Known, pre-existing: KaTeX's visually hidden MathML is absolutely positioned and can widen the layout
  viewport on phones (Topic 2 shows it too); a site-wide CSS fix, not part of this topic.
