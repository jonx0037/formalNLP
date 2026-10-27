# Brief — `topic-models-lda`

Topic 25, the first `text-mining` topic. Status: **Phase A complete** (2026-10-26). Harness: 31/31 under `python -W error`, 45 s (no waiver needed). Every
number in §3 is the harness's, at 6 decimals. 

Decisions (Jonathan, 2026-10-26): simulated LDA worlds carry the recovery and identifiability claims, and the
finance case study runs on the untouched `fc` corpus; both inference engines (collapsed Gibbs and mean-field VI) are
proved and run; the geometric anchor is the topic polytope plus anchor words.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Topic Models: Latent Dirichlet Allocation |
| `{{SUBTITLE}}` | Documents as points in a polytope of topics: when the topics can be recovered, the two ways to infer them, and what a topic model finds in filings whose generator plants seven vocabularies but only six identifiable topics. |
| `{{SLUG}}` | `topic-models-lda` |
| `{{DOMAIN}}` | `text-mining` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[distributional-semantics-ppmi-svd]` (Def 1 co-occurrence, Def 3 truncated SVD, Thm 3(1) mixtures make the ratio low rank, Def 4 planted-sector benchmark) |
| `{{FORMALML_PREREQS}}` | `variational-inference` (Def 1 ELBO, Thm 2 CAVI optimal coordinate update, Prop 1 conjugate CAVI), `kl-divergence` (Prop 2, Gibbs' inequality); connection `bayesian-nonparametrics` (Def 1 Dirichlet, Prop 2 Dirichlet–multinomial conjugacy) |
| `{{FORMALRAG_PREREQS}}` | connections `query-likelihood-language-models` (Thm 1, Dirichlet smoothing as posterior mean: the same posterior mean as Prop 6's φ̂), `pca-dimensionality-reduction` (LSA, the Euclidean foil of Rem 1) |
| `{{STATS_CALC_PREREQS}}` | formalStatistics `multivariate-distributions` (Thm 5 Dirichlet moments, Thm 6 Dirichlet–multinomial conjugacy), `bayesian-computation-and-mcmc` (Thm 2 Gibbs is MH with acceptance 1; Rem 10 names the collapsed-Gibbs LDA pattern without deriving it), `exponential-families` (Thm 1, log-partition and moments, one-parameter: Lemma 2 is its K-parameter Dirichlet case, proved directly) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/linear-text-classifiers/`, `notebooks/finance-corpus/` |

All sibling slugs above are `status: published` on GitHub `main`, and their block numbers match (checked
2026-10-26). formalStatistics `exponential-families` Thm 1 is one-parameter only, so Lemma 2 is proved, not cited.
No sibling covers pLSI, NMF, anchor words, LDA's own updates, or topic-model evaluation (grep of every sibling's
`main`, 2026-10-26).

**Scope against siblings.** The siblings prove the ingredients (Dirichlet conjugacy, Gibbs = MH, generic CAVI);
this page proves what they leave open: the collapsed full conditional, LDA's CAVI updates, the polytope geometry,
anchor-word identifiability, and the evaluation identities.

## 1. Section outline (geometric first)

1. **Overview.** One filing's content words as a point in the word simplex; K topics as points whose convex hull
   holds every document's expected word distribution. Topic 6 ended on "a topic model would find this structure
   more easily": we take that up.
2. **The generative model.** Def 1, Def 2, Prop 1, Prop 2, Rem 1 (LSA vs. the polytope). Lab tab *Topic simplex*.
3. **Identifiability and anchor words.** Prop 3, Prop 4, Def 3, Prop 5, Thm 1, Algorithm 1. Lab tab *Anchors*.
4. **Collapsed Gibbs sampling.** Lemma 1, Thm 2, Algorithm 2, Cor 1, Prop 6.
5. **Mean-field variational inference.** Lemma 2, Thm 3, Cor 2, Rem 2 (CVB0). Lab tab *Gibbs vs. VI*.
6. **Evaluating a topic model.** Def 4, Thm 4, Prop 7, Def 5.
7. **Finance case study** (`<FinanceCaseStudy>`): (a) seven restaurants, six topics; (b) recovering the six;
   (c) the optimizers, not the model: CAVI and MoU's EM stall where their own objectives are lower; (d) all words:
   what the templates do; (e) the capstone handoff (`topic_mixtures`).
   Lab tab *Finance topics*.
8. **Where this sits** / honest caveats / implementation.

## 2. Statements (numbered per type, in page order; same numbers in the `.py` and notebook)

Notation: V words, K topics, D documents. Topics φ_k ∈ Δ^{V−1} are the rows of Φ (K × V); a document's topic
proportions θ_d ∈ Δ^{K−1}; z_dn its n-th token's topic; x_d ∈ ℕ^V its counts, n_d = |x_d|. Responsibilities in VI
are r_dnk (not φ, to keep φ for topics).

**Def 1 (LDA).** Given α ∈ ℝ^K_{>0} and β > 0: φ_k ~ Dir(β𝟏_V) for each k; for each document, θ_d ~ Dir(α), and
for each position z_dn ~ Cat(θ_d), w_dn ~ Cat(φ_{z_dn}). pLSI is the same model with θ_d and φ_k as free
parameters (no priors).

**Def 2 (Topic polytope).** T(Φ) = conv{φ_1, …, φ_K} ⊂ Δ^{V−1}.

**Prop 1 (Documents live in the polytope).** Given θ_d and Φ, the words of document d are i.i.d. with distribution
p_d = Φᵀθ_d ∈ T(Φ), and E[x_d | θ_d, n_d] = n_d Φᵀθ_d. Hence the matrix of expected word distributions
P = ΘΦ (D × V) is a product of nonnegative matrices with inner dimension K (nonnegative rank ≤ K).
*Proof:* marginalize z_dn: P(w_dn = w | θ_d) = Σ_k θ_dk φ_k(w).

**Prop 2 (Fitting θ is a KL projection).** For fixed Φ, with p̂_d = x_d/n_d,
Σ_w x_dw log (Φᵀθ)_w = −n_d [H(p̂_d) + KL(p̂_d ‖ Φᵀθ)],
so the θ maximizing the likelihood puts Φᵀθ at the KL projection of p̂_d onto T(Φ); the projection exists, and
the projected point is unique (θ itself is unique when the φ_k are affinely independent). pLSI's MLE minimizes
Σ_d n_d KL(p̂_d ‖ T(Φ)) jointly over Φ; Gaussier and Goutte (2005) show this is NMF under the generalized KL
divergence (stated, cited from its Crossref record: ACM returns 403, so no section is cited). Hofmann (1999,
UAI, §3.3 and Fig. 2) gives the sub-simplex picture and the KL reading for the joint co-occurrence table; the
per-document form here is direct. *Proof:* algebra, then cite formalML `kl-divergence` Prop 2; existence by compactness of
T, uniqueness of the point by strict convexity of KL(p̂ ‖ ·) on the support of p̂ … (handle zero entries carefully;
harness checks the projection against a direct convex solve).

**Rem 1 (LSA's projection is Euclidean and signed).** Topic 6's truncated SVD (Def 3) projects the rows of the
count matrix in Euclidean norm onto a d-dimensional subspace whose coordinates take both signs; LDA projects in KL
onto a K-vertex polytope inside the simplex, so a document's coordinates are proportions.

**Prop 3 (The factorization is not unique).** (a) For every permutation matrix P, (PΦ, ΘPᵀ) gives the same
distribution and, under symmetric priors, the same posterior density: the posterior has K! symmetric modes, so
posterior means of φ_k are meaningless without alignment. (b) Without further conditions the polytope itself is
not identified: an explicit (Φ, Θ) and invertible M ≠ permutation with MΦ ≥ 0, ΘM⁻¹ ≥ 0, rows on simplices, giving
the same P. *Proof:* (a) relabeling is a bijection of z that preserves the joint; (b) construction.

**Prop 4 (A topic every document shares at one weight is unidentifiable).** Suppose every document's θ_d puts
weight c on a shared topic φ_0 and 1 − c on one topic φ_{s(d)} of its own (s(d) ∈ {1, …, S}). Then each
document's words are i.i.d. from ψ_s = (1 − c)φ_s + cφ_0, the same distribution as a single-membership model
(mixture of unigrams) with S topics; every c′ ∈ [0, c_max] with a matching φ′_0 and φ′_s fits equally well, and the
topic–topic matrix R = E[θθᵀ] has rank S < S + 1. *Proof:* direct.
*This is the finance generator, with c = 1 − SECTOR_SHARE = 0.5: see §7(a).*

**Def 3 (Separability).** Φ is separable if each topic k has an anchor word a_k with φ_k(a_k) > 0 and φ_j(a_k) = 0
for j ≠ k.

**Prop 5 (Unbiased co-occurrence).** For a document with n_d ≥ 2,
Q̂_d = (x_d x_dᵀ − diag x_d)/(n_d(n_d − 1)) satisfies E[Q̂_d | θ_d] = Φᵀθ_dθ_dᵀΦ. Averaging over documents,
E[Q̂] = ΦᵀRΦ with R = E[θθᵀ]. *Proof:* x_d x_dᵀ − diag x_d counts ordered pairs of distinct positions, and distinct
positions are i.i.d. from p_d given θ_d. (Arora et al. 2013, full version arXiv 1212.4777, App. D.1.)

**Thm 1 (Anchor rows are the vertices).** Attribution in prose: Arora, Ge, and Moitra (2012) define p-separability
(Def. 1.1; "anchor word" is §1 prose) and prove polynomial-time recovery (Thm 1.4) under a lower bound on the ℓ1
condition number of R (Def. 2.1), which implies nonsingularity; Arora et al. (2013, §3, Eqs. 1–2 and the unnumbered
text after them) give the convex-combination form with C_ik = p(z1 = k | w1 = i). Our statement is the exact,
population-level identifiability core; the finite-sample rates are cited. Let Q = ΦᵀRΦ, p = Q𝟏 the
word marginal, and Q̄ = diag(p)⁻¹Q (each row a distribution: the next word given this one). If Φ is separable with
anchors a_1, …, a_K, then for every word w with p(w) > 0,
Q̄_w = Σ_k C_wk Q̄_{a_k}, with C_wk = P(z = k | w) = φ_k(w)·(R𝟏)_k / p(w) … (exact form fixed in Phase A),
so every row of Q̄ lies in conv{Q̄_{a_1}, …, Q̄_{a_K}}. If R is nonsingular, the anchor rows are affinely
independent, they are exactly the vertices of conv{Q̄_w : w}, and Φ is recovered from C and p by Bayes' rule:
φ_k(w) ∝ C_wk p(w). *Proof:* full (Bayes on the first word of a pair; distinct positions independent given the
topics; vertices of a simplex via affine independence).

**Algorithm 1 (FastAnchorWords + RecoverL2).** Candidates are words in at least 5% of the documents (a fixed count
lets rare words, whose sampled rows are mostly noise, win the farthest-point search). Arora et al. (2013): FastAnchorWords is their Alg. 4; RecoverL2 has
no algorithm number (§3 prose; RecoverKL is Alg. 3). Greedy farthest-point with Gram–Schmidt projection over candidate
rows with document frequency ≥ m; then per word a simplex-constrained least-squares fit of Q̄_w on the anchor rows.

**Lemma 1 (Dirichlet–multinomial marginal).** ∫ Π_k θ_k^{n_k} Dir(θ; α) dθ = B(α + n)/B(α), with
B(a) = Π Γ(a_k)/Γ(Σ a_k). Cite formalStatistics `multivariate-distributions` Thm 6; one-line proof.

**Thm 2 (Collapsed full conditional; Griffiths and Steyvers 2004).** With θ and Φ integrated out,
P(z_i = k | z_{−i}, w) ∝ (n_{dk}^{−i} + α_k) · (n_{k w_i}^{−i} + β)/(n_k^{−i} + Vβ),
where the counts exclude token i. *Proof:* Lemma 1 twice gives p(w, z) in closed form; the ratio for z_i = k vs.
the rest uses Γ(x + 1) = xΓ(x) on exactly the three counts token i touches. Griffiths and Steyvers's Eq. 5 keeps the
document denominator n_d − 1 + Kα (their T topics, W words); it does not depend on k, so ∝ drops it.

**Algorithm 2 (Collapsed Gibbs sweep).**

**Cor 1 (The posterior is the stationary law).** The systematic- or random-scan collapsed Gibbs chain leaves
p(z | w) invariant (formalStatistics `bayesian-computation-and-mcmc` Thm 2), and with α, β > 0 every conditional is
positive, so the chain is irreducible and aperiodic on the finite state space and converges to p(z | w).
*Harness:* on a 7-token, K = 2 corpus (128 states) the exact transition matrix's stationary vector equals the
enumerated posterior to ≤ 10⁻¹².

**Prop 6 (Point estimates from one sample).** E[φ_k(w) | z, w] = (n_kw + β)/(n_k + Vβ) and
E[θ_dk | z] = (n_dk + α_k)/(n_d + Σα): Dirichlet posterior means (cf. formalRAG `query-likelihood-language-models`
Thm 1). Averaging these over samples is meaningful only within one labeling mode (Prop 3(a)).

**Lemma 2 (Dirichlet log-moment).** If θ ~ Dir(γ), E[log θ_k] = ψ(γ_k) − ψ(Σ_j γ_j). *Proof:* differentiate
log B(γ) under the integral (Blei, Ng, and Jordan 2003, App. A.1, an unnumbered display; the Dirichlet is an exponential family with sufficient statistic log θ; formalStatistics
`exponential-families` Thm 1 is the one-parameter case).

**Thm 3 (CAVI for LDA; Blei, Ng, and Jordan 2003).** Under the mean-field family
q = Π_k Dir(φ_k; λ_k) Π_d [Dir(θ_d; γ_d) Π_n Cat(z_dn; r_dn)], the optimal coordinate updates are
r_dnk ∝ exp(E_q log θ_dk + E_q log φ_k(w_dn)), γ_d = α + Σ_n r_dn, λ_kw = β + Σ_{d,n} r_dnk 1[w_dn = w],
with both expectations from Lemma 2. Blei, Ng, and Jordan's Eqs. (6)–(8), derived in their App. A.3; the λ factor is their smoothed LDA (§5.4, Fig. 7).
*Proof:* formalML `variational-inference` Thm 2 (the optimal factor is the
exponentiated expected log complete conditional) and Prop 1 (conjugate case: the expected natural parameter).

**Cor 2 (Monotone ELBO).** Each update maximizes the ELBO in its block, so the ELBO is non-decreasing; the harness
asserts monotonicity to 10⁻⁹ relative and matches scikit-learn's batch VB (same α, β) on the bound.

**Rem 2 (CVB0).** Replacing Thm 2's counts with expected counts gives the CVB0 update (Asuncion et al. 2009, Eq. 13), whose
hyperparameters, not the engine, explain most differences between inference algorithms (their claim; cited).

**Def 4 (Held-out evaluation).** Held-out log-likelihood per token, and document completion: estimate θ from even
positions, score odd positions.

**Thm 4 (The harmonic-mean estimator).** Attribution in prose: Newton and Raftery (1994, pp. 3–26; §7 per Neal's
discussion, the body is paywalled) propose the estimator; Neal's discussion objects that it cannot work when the
posterior barely depends on the prior, and his 2008 blog post names the infinite variance; Wallach et al. (2009,
§4.2 Eq. 14; §6.2) find it "wildly overestimated" topic-model likelihoods. Statement (ours, proved):
(a) E_{p(z | w)}[1/p(w | z)] = 1/p(w), so p̂_HM = (S⁻¹ Σ_s 1/p(w | z⁽ˢ⁾))⁻¹ is consistent;
(b) E[p̂_HM] ≥ p(w) (Jensen: 1/x is convex), so it overestimates in expectation;
(c) with Φ fixed and θ continuous, Var_{p(θ | w)}[1/p(w | θ)] is finite iff ∫ p(θ)/p(w | θ) dθ < ∞; if Φ is separable and
the document holds c ≥ α_k tokens of topic k's anchor words, the integral diverges (the Beta(α_k, α₀ − α_k) marginal
of θ_k against θ_k^{−c}), so the variance is infinite. With α_k = 0.1, one anchor token suffices.
*Harness:* (a) exact on the toy (10⁻¹²); (b) 4,000 replicates of S = 20 exact posterior draws: mean and median above
p(w); (c) the tail integral over [ε, 1] grows tenfold per ε/100 (ε^{−1/2}, c = 1, α₀ = ½). On a finite collapsed state
space the variance is finite, and on the toy it is *smaller* than prior IS's (relative variance 0.649 vs 1.721; at
n = 12 tokens 1.015 vs 17.424): the harmonic mean's defect is bias and, with continuous θ, infinite variance.

**Prop 7 (Importance sampling from the prior is unbiased).** (1/S) Σ_s p(w | z^{(s)}), z^{(s)} ~ p(z), is unbiased
with finite variance (p(w | z) ≤ 1), and by Jensen its logarithm underestimates log p(w) in expectation. Its relative
variance grows quickly with the number of tokens (toy, one document: 0.387, 3.033, 17.424 at n = 4, 8, 12): measured,
not proved. (Left-to-right (Wallach et al. 2009, Alg. 3) and
Chib-style (Alg. 2) estimators are cited, not implemented; the page's comparisons use document completion, which needs no
marginal likelihood.)

**Def 5 (Topic coherence).** UMass coherence (Mimno et al. 2011) and NPMI coherence (Newman et al. 2010, normalized
as in Topic 6 Def 2's PMI).

## 3. Findings (harness, 6 decimals; scratchpad spike superseded)

**Exact checks.** Cor 1: the 7-token toy's systematic sweep has stationary vector = enumerated posterior (≤ 10⁻¹²),
every single-site kernel is reversible, and one sweep reaches every state. Thm 1: population recovery exact
(< 10⁻¹⁰); with Prop 4's singular R the shared topic's anchor row lies in the other anchors' hull (LP distance 0)
and FastAnchorWords picks a non-anchor. Thm 3: our closed-form bound equals scikit-learn's `_approx_bound` to 10⁻⁸
relative. Cor 2: the explicit-r ELBO never decreases across E- and M-steps. toy log p(w) = −9.015442.

**Simplex worlds** (tab 1). Share of documents whose p̂_d lies outside the topic triangle, n_d = 10 / 40 / 160:
α = 0.1: 0.775 / 0.700 / 0.662 (0.6625, which the payload rounds to 0.662); α = 1: 0.613 / 0.313 / 0.150; α = 10: 0.488 / 0.050 / 0.000.

**Anchors** (V = 400, K = 5, dense Dir(𝟏) topics, anchor mass 0.05, α = 0.3, 50 tokens). Mean L1 per topic at
M = 250 / 1k / 4k / 16k: separable 0.541643 / 0.312416 / 0.126474 / 0.064663 (planted anchors found from 4k on;
population 0); without anchors 0.558271 / 0.454478 / 0.211679 / 0.190079 against a population floor of 0.168579.
Sparse Dir(0.1) topics are already nearly separable (their frequent words sit in one topic), so the contrast needs
dense topics: recorded in the `anchor_phi` docstring.

**Gibbs vs. VI** (same world, M = 500; mean L1): at the world's own β = 1, Gibbs 0.293259, CAVI 0.451541, anchors
0.438596; at β = 0.01, Gibbs 0.786969, CAVI 0.460766. The prior ranks the engines (Asuncion et al.'s point).

**Finance, restaurant vocabulary** (V = 2,908, 13,382 tokens in 300 documents; 100 continuation documents, 3,735
tokens, held out; α = β = 0.1, K = 6, 3 seeds; L1 against the six ψ_s; completion in nats per token):

| method | L1 | completion | sector NMI |
|---|---|---|---|
| collapsed Gibbs | 0.547987 (0.367–0.641) | −5.794332 (−5.807 to −5.778) | 0.966746 (0.944–1.000) |
| CAVI from noise | 1.032990 (1.010–1.062) | −6.184908 | 0.186384 (0.134–0.224) |
| CAVI from Gibbs's counts | 0.575959 | −5.839774 | 0.963446 |
| anchors (Algorithm 1) | 0.590171 | — | — |
| MoU, best of 20 EM restarts | 0.985009 | −6.157372 | 0.269964 |
| MoU started at the gold sectors | 0.278737 | −5.733688 | 1.000000 |

- Seven restaurants, six topics: median shared-vocabulary share per document 0.488889; FastAnchorWords' six anchors
  are outlevy (industrials), power (energy), merchandise (retail), cloud (technology), royaltyside (banking),
  patients (healthcare); its 7th pick is freight (industrials), never a shared-vocabulary word.
- CAVI's ELBO* from noise averages −93,556.194 vs −88,821.589 from Gibbs's counts: the bound prefers the sectors, and
  coordinate ascent stalls short of them (scikit-learn's own batch fit, `fin_sklearn`, lands between: NMI 0.652007, ELBO* −90,352.528). Same lesson as MoU: EM's
  best restart has training log-likelihood −80,675.273 vs −77,168.390 at the gold start.
- LSA's rank-6 reconstruction of the documents' word frequencies has 0.283043 of its entries negative (Rem 1).
- `topic_mixtures` reads the held-out documents' sectors at 0.930000 (argmax θ̂).

**Templates.** K = 6, one seed, 150 sweeps. All words: one topic takes the function words and the rest key on company
names; sector NMI 0.942512, doc_type 0.011797, tone 0.010568, median max θ 0.897053. The df band (3–90 documents):
0.708987, 0.023328, 0.014573. LDA never sees tone or document type.

## 3a. Experiment grid (as run)

Toy (7 tokens, V = 3, K = 2, α = β = 0.5) for Thm 2, Cor 1, Thm 4(a–b), Prop 7; one-document toys of n = 2–12 tokens
for the variance table; `HM_DOC` (one anchor token + five shared, α = (½, ½)) for Thm 4(c). Simplex worlds 5 α × 3
lengths × 80 documents. Anchor worlds as above; ternary display world K = 3, V = 60. Finance: restaurant vocabulary
(3 seeds Gibbs/CAVI/warm CAVI, 20 MoU restarts + gold start), all words and df band (one Gibbs run each).

## 4. The lab: `TopicModelLab` (4 tabs; every number from the payload)

1. **Topic simplex.** V = 3 word simplex with K = 3 topic points and their triangle; documents' p_d (exact) and
   p̂_d (from n_d words). Sliders: α (θ toward vertices or center), n_d (noise). Learn: documents live in the
   polytope only in expectation; small α piles them at the vertices.
2. **Anchors.** Q̄ rows of the separable world projected to 2-D (onto the anchors' affine hull), the hull and the
   anchors marked; a second panel of recovery L1 vs. M (separable vs. non-separable). Toggle: the finance rows,
   where the common restaurant sits inside the hull (Prop 4).
3. **Gibbs vs. VI.** Per-sweep log joint (Gibbs) and ELBO (CAVI) traces; recovery L1 per topic for Gibbs, CAVI, and
   anchors; a β toggle (1 = the world's prior, 0.01 = misspecified) that reverses the ranking.
4. **Finance topics.** Top words per topic for a chosen treatment (restaurant words / all words / df band),
   colored by gold restaurant; θ_d on the polytope by gold sector; the LDA vs. MoU completion table.

## 5. Checkpoint answers (Jonathan, 2026-10-26)

1. Finance §7 leads with "seven restaurants, six identifiable topics" (Prop 4 + Thm 1's failed hypothesis), then
   the recoveries against ψ_s, then LDA vs. the correctly specified mixture of unigrams that EM cannot find.
2. Evaluation: Thm 4 and Prop 7 proved and simulated exactly on the toy; left-to-right and Chib-style cited only;
   comparisons by document completion.
3. Harness budget: waived up to ~90 s (as Topics 20, 22, 24).
4. Difficulty: `intermediate`.

## 5b. rigorFlag (draft)

Anchor recovery's guarantees assume separability and a nonsingular topic–topic matrix; the finance generator
satisfies the first exactly and fails the second, so one of its seven planted vocabularies cannot be recovered from
any amount of data. LDA's posterior is intractable: the collapsed sampler is exact only in the limit, and mean-field
VI optimizes a bound by coordinate ascent that stalls in local optima. On the finance corpus CAVI from scikit-learn's
noise start stops where its own bound is about 4,700 nats lower than at the sectors, and EM for the correctly
specified single-membership model stalls the same way, so engine comparisons here measure optimizers as much as
models, and their ranking depends on the topic prior. Held-out likelihood must be estimated, and the harmonic-mean
estimator is biased upward with infinite variance under continuous θ; the page compares models by plug-in document
completion instead. Perplexity and human interpretability can disagree (Chang et al. 2009). The synthetic corpus has
a known answer; real filings do not.

## 6. Notebook cell map (24 cells, `01_topic_models_lda.ipynb`)

1–2 title, imports and the finance vocabulary's size · 3–4 §1 Prop 1, Prop 2, the simplex table · 5–6 Rem 1 (LSA's
negative entries) · 7–8 §2 Props 3–5, Thm 1 and its singular-R failure · 9–10 Algorithm 1's recovery vs. M and the
floor · 11–12 §3 Lemma 1, Thm 2, Cor 1, the sampler's marginals, Prop 6 · 13–14 §4 Lemma 2, Cor 2, Thm 3 vs.
scikit-learn, the β toggle · 15–16 §5 Thm 4, Prop 7's variance table, Def 5 · 17–18 §6 seven restaurants, six
anchors · 19–20 the engines table, CAVI's stall, scikit-learn · 21–22 templates and `topic_mixtures` · 23–24 the full
harness.

## 7. References (verified 2026-10-26: titles, venues, URLs, statement numbers)

- Blei, Ng, and Jordan (2003), JMLR 3: 993–1022, jmlr.org/papers/v3/blei03a.html (no DOI). Eqs. (6)–(8) §5.2,
  App. A.3; App. A.1 unnumbered E[log θ]; §5.4 smoothing (Fig. 7); Fig. 4 (§4.4) topic simplex in word simplex; §4.3
  pLSI.
- Griffiths and Steyvers (2004), PNAS 101 suppl. 1: 5228–35, doi 10.1073/pnas.0307752101. Eq. 5 (includes the
  document denominator).
- Hofmann (1999a), "Probabilistic Latent Semantic Indexing," SIGIR '99, 50–57, doi 10.1145/312624.312649 (metadata
  only). Hofmann (1999b), "Probabilistic Latent Semantic Analysis," UAI 1999, 289–96, arXiv 1301.6705: §3.3, Fig. 2.
- Arora, Ge, and Moitra (2012), "Learning Topic Models—Going beyond SVD," FOCS 2012, 1–10, doi 10.1109/FOCS.2012.49
  (arXiv 1204.1956): Def. 1.1, Def. 1.3, Def. 2.1, Thm 1.4.
- Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu, and Zhu (2013), ICML, PMLR 28(2): 280–88,
  proceedings.mlr.press/v28/arora13.html (arXiv 1212.4777): §3 Eqs. (1)–(2); App. D.1; Algs. 3–4; RecoverL2 unnumbered.
- Donoho and Stodden (2003), NIPS 16 (MIT Press 2004), papers.nips.cc: [R2] Separability (Eq. 5), Thm 1. Pages
  unverified (bibtex blank).
- Gaussier and Goutte (2005), SIGIR 2005, 601–2, doi 10.1145/1076034.1076148: metadata only; no section cited.
- Wallach, Murray, Salakhutdinov, and Mimno (2009), ICML 2009, 1105–12, doi 10.1145/1553374.1553515: §4.2 Eq. 14,
  §6.2, Alg. 2 (Chib-style), Alg. 3 (left-to-right).
- Newton and Raftery (1994), JRSS-B 56 (1): 3–26, doi 10.1111/j.2517-6161.1994.tb01956.x (paywalled; §7 per Neal's
  discussion). Neal (1994), discussion, author copy at radfordneal.wordpress.com. Neal (2008), "The Harmonic Mean of the
  Likelihood: Worst Monte Carlo Method Ever," blog post (type `blog`).
- Chang, Boyd-Graber, Gerrish, Wang, and Blei (2009), NIPS 22, papers.nips.cc (author order from the PDF; site metadata
  differs; pages unverified). Abstract: held-out likelihood can favor less semantically meaningful topics.
- Mimno, Wallach, Talley, Leenders, and McCallum (2011), EMNLP, 262–72, D11-1024: §4.2 Eq. (1).
- Newman, Lau, Grieser, and Baldwin (2010), NAACL-HLT, 100–108, N10-1012: PMI over a 10-word window; median of 45
  pairs.
- Wallach, Mimno, and McCallum (2009), "Rethinking LDA: Why Priors Matter," NIPS 22 (pages unverified): asymmetric α
  helps, asymmetric β does not.
- Teh, Newman, and Welling (2006), NIPS 19 (MIT Press 2007; pages unverified).
- Asuncion, Welling, Smyth, and Teh (2009), UAI 2009, 27–34, arXiv 1205.2662: CVB0 Eq. (13); "amount of smoothing".
- Deerwester et al. (1990), LSA (already cited by Topic 6; reuse its entry).
- Bybee, Kelly, Manela, and Xiu (2024), "Business News and Business Cycles," Journal of Finance 79 (5): 3105–47,
  doi 10.1111/jofi.13377 ("a topic model" on 800,000 WSJ articles); LDA is confirmed only in NBER w26648 (2020, earlier
  title "The Structure of Economic News"), so cite the working paper for the LDA claim.
