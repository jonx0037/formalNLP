# Brief — `sequence-labeling-hmm-crf`

**Sequence Labeling: From HMMs to Linear-Chain CRFs.** This is roadmap #21 (domain `extraction-knowledge-graphs`). Its
prerequisites are published: `n-gram-models-smoothing` (#3: the HMM's transition model is a tag n-gram, and add-α is its
smoother) and `linear-text-classifiers` (#17: the MEMM's factors are Topic 17's softmax LR, and Thm 3 is Topic 17's
generative–discriminative pair at sequence level). It feeds `named-entity-recognition` (#22). The scaffold's blocker is
cleared: formalStatistics `hidden-markov-models` is `status: published` on GitHub `main` (datePublished 2026-10-25).

**Status: shipped (Phase A and Phase B, 2026-10-26).**
- `notebooks/sequence-labeling-hmm-crf/sequence_labeling_hmm_crf.py`: 22 tests, ≈ 46 s under `-W error`.
- `01_sequence_labeling_hmm_crf.ipynb` (26 cells) exits 0 under `jupyter execute`.
- Lab data `src/components/viz/data/sequence-labeling-lab.json` is written by `--write-payload` and pinned by
  `test_viz_mirror_matches_payload`.
- The Phase A checkpoint was approved by Jonathan (2026-10-26) with the recommended answers to §5: `intermediate`, all
  five tabs, and finding 4 in the rigorFlag with a short note in the finance case study.
- **Phase B** shipped the MDX, `SequenceLabelingLab` (5 tabs), and the graph and curriculum flip.
  - Page theorem numbering follows order of appearance and matches the `.py`: Thm 2 = CRF likelihood, Thm 3 = the
    HMM–CRF pair, Thm 4 = decoding risk (the page presents decoding risk after the pair, just before the MEMM).
  - The trellis payload stores max-marginals relative to the mode (the mode's probability is 1.12·10⁻⁴, which rounds to
    0 at three decimals), plus the correct-tag counts (17 vs 19 of 24) and expected counts (16.384 vs 16.928).
  - Gates: `pnpm test` 39/39, `astro sync`, `validate`, `audit:cross-site` (formalNLP 170 edges, all deferred
    reciprocals, 0 mismatches), and `build`.
  - Page: 0 `.katex-error`, 400 KaTeX spans, no `‘`, every tab hydrated and every readout equal to the payload, and
    `scrollWidth` 375 at the mobile preset in all 21 tab and toggle states; no console errors.

**Phase A decisions** (plan mode, 2026-10-26):
- **Scope against #22 (NER), decided with Jonathan.** #21's finance case study tags the gold entity spans as BIO with
  hand-built features (HMM, MEMM, CRF, structured perceptron; token accuracy and a first span F1). #22 owns contextual
  encoders + CRF, span-level evaluation in depth, BIO/BIOES constrained decoding, and held-out names (roster split).
- **What formalStatistics already proves**, so this page cites rather than re-proves: the forward recursion (Thm 5),
  scaling = filtering (Thm 6), forward–backward (Thm 7), Viterbi (Thm 8), EM ascent and Baum–Welch (Thms 9–10). It names
  the CRF only in Rem 23. `exponential-families` Thm 1 is the *one-parameter* A′ = E T, A″ = Var T, so the multivariate
  CRF gradient and Hessian are proved here. No sibling covers CRFs, MEMMs, label bias, or the structured perceptron
  (grepped formalML, formalStatistics, formalRAG `main`, 2026-10-26).
- **Claude wrote `hmm_from_crf`** (Jonathan's call; the plan had offered it as his contribution), with c = 2ρ(M).

**What building the harness found** (each now an assert):
1. **Prop 2 binds marginals, not Viterbi.** The future factors of an MEMM *sum* to one, but their *max* is not constant,
   so a Viterbi-decoded MEMM still sees later words. In the HMM world the causal MEMM decoded by Viterbi scores 0.758375,
   above the filtering ceiling 0.743104 that binds it decoded by marginals (0.735021). This is Lafferty et al.'s label
   bias in its original (Viterbi) form, and it is why Cor 2 is stated for posterior-marginal decoding.
2. **A CRF's revision decays with distance.** Changing the last word of a 7-token chain moved z₁'s marginal by 3·10⁻⁵ and
   z₅'s by 0.03 (random weights). The CRF *can* revise; how far depends on how fast the chain mixes.
3. **Same family, different criterion.** In the misspecified world the indicator CRF — the HMM's own conditional family
   (Thm 3) — beats the HMM on conditional log-loss at n ≥ 300 (1.036193 vs 1.039914 at n = 1000) while trailing it on
   accuracy (0.534250 vs 0.540063). Cor 1 is about the estimator's criterion, not about accuracy.
4. **The finance corpus barely exercises label bias.** PRODUCT and DATE spans open with "the", whose tag depends on the
   *next* word, but the templates make the left context sufficient: the causal MEMM tags "the" at 0.997299 with 30
   training documents. The lab shows Prop 2 on counterfactual continuations instead (§4).
5. **Finance errors are unseen names.** At 3 training documents, 0.369 of test tokens are words never seen in training;
   nearly every error is an unseen name tagged O. That is #22's held-out-names problem, arising here naturally at small n.
6. **Regularization.** Validation log-loss put the CRF's best λ near 1/n at every n (0.1 at n = 10, 0.01 at 100, 0.001 at
   1000). On the mean objective, λ = 1/n is exactly MAP under θ ~ N(0, I), so every discriminative tagger is fit once at
   that prior, untuned. The HMM's add-α is tuned on validation log-loss: add-1 cost the finance HMM 2–5 points (a
   uniform Dirichlet over ~500 words swamps rare tags). The page states this asymmetry.
7. **Optimizer.** trust-ncg with complex-step Hessian-vector products reaches gradient 10⁻¹⁰ in ~13 Newton steps but
   needs ~150 Hv products (8 s at n = 1000, near-separable data). L-BFGS with `ftol = 0, gtol = 10⁻⁶` lands within
   2.4·10⁻⁵ of that θ, with identical decodes, in 2 s. Experiments use L-BFGS; `test_lbfgs_reaches_the_trust_region_optimum`
   pins the gap and `test_thm2_unique_optimum` uses trust-ncg.

---

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Sequence Labeling: From HMMs to Linear-Chain CRFs |
| `{{SUBTITLE}}` | One recursion in two semirings tags a sequence; what changes from HMM to CRF is not the family of conditionals but what the estimator optimizes and what the features can see. |
| `{{SLUG}}` | `sequence-labeling-hmm-crf` |
| `{{DOMAIN}}` | `extraction-knowledge-graphs` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[n-gram-models-smoothing, linear-text-classifiers]` |
| `{{FORMALML_PREREQS}}` | `naive-bayes` (generative–discriminative pair; check §7 wording), `convex-analysis` (concavity of Thm 2) — verify on `main` at Phase B |
| `{{FORMALRAG_PREREQS}}` | none |
| `{{STATS_CALC_PREREQS}}` | formalStatistics `hidden-markov-models` (Thms 4–8), `exponential-families` (Thm 1); connection: `maximum-likelihood` |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/linear-text-classifiers/`, `notebooks/contextual-embeddings-masked-lm/` |

## 1. Section outline (geometric first)

1. **Overview.** A tag per token; the trellis as the picture (K labels × T positions, a path = a tagging). BIO encoding:
   why B is needed ("boston massachusetts" is B-GPE B-GPE).
2. **The HMM tagger.** Def (cite formalStatistics Def 3), Prop 1 (counts; add-α as Dirichlet posterior mean): a tag
   n-gram times naive Bayes emissions.
3. **One recursion, two semirings.** Thm 1 with one proof by distributivity; formalStatistics Thms 5/7/8 as the
   probability case; max-marginals. Lab tab *Trellis*.
4. **Decoding risk.** Thm 4 (Viterbi = Bayes for 0–1 loss; MBR = Bayes for Hamming). Bayes MBR 0.782083 vs Viterbi
   0.776271 in the HMM world.
5. **The linear-chain CRF.** Def; Thm 2 (gradient = observed − expected; Hessian = −Σ Cov; concave; strictly under L2).
6. **A generative–discriminative pair.** Thm 3 (a) and (b) with the h-transform proof; Cor 1. Geometric picture: the
   same set of conditionals, two estimators landing at different points of it.
7. **Local normalization.** MEMM def; Prop 2 (prefix marginals ignore the future); Cor 2 (lagged-Bayes ceiling); the
   Viterbi caveat. Lab tab *CRF vs MEMM*.
8. **The structured perceptron.** Thm 5 with Collins's proof; averaging cited. Lab tab *Perceptron*.
9. **Where the Bayes tagger is exact.** Learning curves in two worlds. Lab tab *Learning curves*.
10. **Finance case study** (`<FinanceCaseStudy>`): BIO over gold spans, the five taggers at 3/10/30 documents, unseen
    names, "the", and the Prop 2 sentence.
11. **Where this sits / honest caveats / implementation.**

## 2. Statements (numbered per type; same numbers in the `.py` and notebook)

**Def 1 (sequence labeling, BIO).** Labels Y; a tagger maps x_{1:T} to z_{1:T} ∈ Y^T. BIO over entity types E:
Y = {O} ∪ {B-e, I-e : e ∈ E}; a span opens at B-e (or at an I-e not continuing a span of type e, conlleval).

**Def 2 (HMM tagger).** formalStatistics Def 3 plus an end state: p(x, z) = ν_{z₁} Π_t a_{z_{t−1}z_t} Π_t b_{z_t}(x_t)
· a_{z_T,stop}, with Σ_k a_{jk} + a_{j,stop} = 1.

**Prop 1.** The supervised MLE is ν̂, â, b̂ = normalized counts (each row's transitions and stop share one
denominator). Adding α to every count is the posterior mean under independent symmetric Dirichlet(α) priors on the
rows. *Proof:* the log-likelihood separates into one multinomial log-likelihood per row; each is maximized by its
frequencies (Lagrange, or Gibbs' inequality); Dirichlet–multinomial conjugacy gives the mean (cite formalStatistics
`multivariate-distributions` Thm 6 as Topic 17 did). With the end state, Σ_T ν A^{T−1} e = ν(I − A)⁻¹e = 1 (asserted).

**Def 3 (chain, potentials).** score(z) = Σ_t U_t(z_t) + Σ_t P_t(z_t, z_{t+1}); Gibbs p(z) = e^{score(z)}/Z.

**Thm 1 (chain distributive law).** In any commutative semiring (⊕, ⊗) with ⊗ distributing over ⊕, define
α₁ = ψ₁(·), α_t(k) = ψ_t(k) ⊗ ⊕_j α_{t−1}(j) ⊗ ψ_{t−1,t}(j, k), and β symmetrically. Then ⊕_k α_T(k) = ⊕_z ⊗ potentials
(Z in (+, ×), the mode's score in (max, ×)); α_t(k) ⊗ β_t(k) aggregates every path through (t, k) (marginal ·Z, or
max-marginal); and α_t(j) ⊗ ψ(j, k) ⊗ ψ_{t+1}(k) ⊗ β_{t+1}(k) every path through the edge. O(TK²). *Proof:* induction
on t, pulling the factors that do not involve z_{1:t−1} out of the inner aggregate by distributivity; one proof for both
semirings. formalStatistics Thms 5, 7, 8 are the HMM potentials in (+, ×) and (max, ×).

**Thm 2 (CRF likelihood).** For p_θ(z | x) = exp(θ·F(x, z))/Z_θ(x) and ℓ(θ) = (1/n) Σ_i log p_θ(z_i | x_i):
∇ℓ = (1/n) Σ_i (F(x_i, z_i) − E_θ[F(x_i, Z) | x_i]); ∇²ℓ = −(1/n) Σ_i Cov_θ[F(x_i, Z) | x_i] ⪯ 0. So ℓ is concave, and
ℓ − (λ/2)‖θ‖² is strictly concave for λ > 0 with a unique maximizer. E_θ F decomposes over Thm 1's node and edge
marginals. *Proof:* differentiate log Z twice (the multivariate version of formalStatistics `exponential-families`
Thm 1); a covariance matrix is PSD.

**Thm 3 (generative–discriminative pair).** Let C_HMM be the conditionals p(z | x) of HMMs with an end state, and C_CRF
those of CRFs with indicator features (transition θ_jk, emission μ_kw, start, end). Then C_HMM = C_CRF.
(a) ⊆: set θ = log a, μ = log b, start = log ν, end = log a_{·,stop}. (b) ⊇: given (θ, μ, start, end), let
S_k = Σ_w e^{μ_kw}, M_jk = e^{θ_jk} S_k, choose c > ρ(M), h = (cI − M)⁻¹ e^{end} (h > 0 by the Neumann series, since
M > 0), and set b_k(w) = e^{μ_kw}/S_k, a_jk = M_jk h_k/(c h_j), a_{j,stop} = e^{end_j}/(c h_j),
ν_k ∝ e^{start_k} S_k h_k. Rows sum to one because (cI − M)h = e^{end}. The joint telescopes:
p(x, z) = e^{score(x, z)}/(c^T Σ_j e^{start_j} S_j h_j), a function of T alone, so p(z | x) is the CRF's.
Smith and Johnson (2007, Cor 2) prove the Mealy-feature version via Chi's method; Sutton and McCallum (2012, §2.3) state
(a) and call the converse "more complicated." **Cor 1.** On indicator features, joint MLE (HMM) and conditional MLE
(CRF) search the same set of conditionals; asymptotically the CRF minimizes conditional KL over it, so its limiting
conditional log-loss is ≤ the HMM's, with equality when the HMM is well specified.

**Def 4 (MEMM).** p(z | x) = Π_t p(z_t | z_{t−1}, x), each factor a softmax LR (McCallum, Freitag, and Pereira 2000).
Lookahead w: factor t depends on x only through x_{1:t+w}.

**Prop 2 (MEMM causality).** With lookahead w, p(z_{1:t} | x) = p(z_{1:t} | x_{1:t+w}) for every t. *Proof:* sum out
z_T, then z_{T−1}, …, z_{t+1}: each summed factor is a distribution, so it sums to one. Consequence (label bias): a state
whose successor distribution ignores x passes its mass on regardless of later words (Lafferty et al. 2001; Sutton and
McCallum 2012, §5.1.3; Smith and Johnson 2007, Thm 2 show C_MEMM ⊊ C_HMM).
**Cor 2.** An MEMM with lookahead w decoded by posterior marginals has expected token accuracy at most that of the Bayes
rule argmax_k p(z_t = k | x_{1:t+w}). *Proof:* by Prop 2 its decision at t is a function of x_{1:t+w}; the Bayes rule
maximizes accuracy among such functions. **Remark:** Viterbi decoding is not covered (finding 1).

**Thm 4 (decoding risk).** Under p(z | x): the Viterbi path minimizes expected 0–1 sequence loss, and the per-position
argmax of the marginals minimizes expected Hamming loss Σ_t 1[ẑ_t ≠ z_t]. *Proof:* expected loss = 1 − p(ẑ | x), and
= Σ_t (1 − γ_t(ẑ_t)), which separates over t.

**Thm 5 (structured perceptron; Collins 2002, Thm 1).** If some unit u has u·(F(x_i, z_i) − F(x_i, z)) ≥ γ for all i and
z ≠ z_i, and ‖F(x_i, z_i) − F(x_i, z)‖ ≤ R, the perceptron makes at most R²/γ² mistakes. *Proof:* on the k-th mistake,
u·θ_{k+1} ≥ u·θ_k + γ (margin) and ‖θ_{k+1}‖² ≤ ‖θ_k‖² + R² (the decoded path scored at least the gold one under θ_k, so
the cross term is ≤ 0); so kγ ≤ ‖θ_{k+1}‖ ≤ √k R. Collins's Thm 2 (non-separable) and averaging (§2.5) are cited.

## 3. Experiment grid (≈ 46 s)

| Block | What | Cost |
|---|---|---|
| Exactness | brute force over K^T paths: Thms 1–5, Props 1–2; Topic 12's forward–backward pin | ~3 s |
| Worlds | Topic 12's chain, B_ε = 0.7B + 0.3/V; "hmm" (ρ = 0) and "next" (ρ = 0.5: a word comes from the next state's row). Train prefixes of one draw (stream 3420), validation 500 (3421), test 2000 (3422). n ∈ {10, 30, 100, 300, 1000}. HMM (α ∈ {1, 0.1, 0.01} by validation log-loss), CRF w = 0 and w = 1 (λ = 1/n), averaged perceptron (w = 1, 5 epochs) | ~21 s |
| Cor 2 | MEMMs (two words left, lookahead 0/1, λ = 1/tokens) at n = 1000 vs lag-w Bayes rules on prefix chains | ~7 s |
| Finance | `generate_corpus` (300, 0) train prefixes of 3/10/30 documents, (60, 2) dev, (200, 1) test; 17 BIO tags; HMM (singletons → `<unk>`), MEMM causal/full, CRF, perceptron; features: word, pre3, suf3, ±2 words (causal: left only) | ~13 s |

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

**Worlds (test token accuracy; Bayes = MBR on the true chain).**

| world | Bayes MBR / Viterbi | HMM n=10 → 1000 | CRF w=0 | CRF w=1 | perceptron |
|---|---|---|---|---|---|
| hmm | 0.782083 / 0.776271 | 0.737563 → 0.775917 | 0.725417 → 0.775792 | 0.708375 → 0.773479 | 0.634292 → 0.771958 |
| next | 0.619979 / 0.598063 | 0.486292 → 0.540063 | 0.479208 → 0.534250 | 0.495854 → 0.591063 | 0.430083 → 0.576854 |

Log-loss (nats/token) at n = 1000: hmm world HMM 0.582663, CRF₀ 0.583281, CRF₁ 0.589113, Bayes 0.579824; next world HMM
1.039914, CRF₀ 1.036193 (also below at n = 300: 1.046133 vs 1.048140), CRF₁ 0.882104, Bayes 0.846403.

**Cor 2 (n = 1000).** hmm world: lag-0 ceiling 0.743104, MEMM-MBR 0.735021, MEMM-Viterbi 0.758375; lag-1 0.769250,
0.762771, 0.765104. next world: 0.588271, 0.574042, 0.562833; 0.609771, 0.593187, 0.579875.

**Finance (test; acc / span F1).**

| docs | HMM | MEMM causal | MEMM full | CRF | perceptron | unseen share |
|---|---|---|---|---|---|---|
| 3 | 0.908604 / 0.617895 | 0.889668 / 0.517316 | 0.894132 / 0.549544 | 0.895140 / 0.555728 | 0.909132 / 0.571527 | 0.368991 |
| 10 | 0.936205 / 0.744590 | 0.941846 / 0.755903 | 0.941126 / 0.771372 | 0.942758 / 0.776305 | 0.960158 / 0.840455 | 0.200912 |
| 30 | 0.977223 / 0.923210 | 0.981567 / 0.936321 | 0.982671 / 0.940593 | 0.983463 / 0.942942 | 0.990976 / 0.963908 | 0.138462 |

"the" accuracy at 30 documents: causal MEMM 0.997299, CRF 0.996759.

**Prop 2 sentence** ("solenne power guided to percent growth in the …", 30 documents): the causal MEMM gives "the"
B-DATE 0.850 under every continuation (identical to 10⁻¹⁶); the CRF gives B-DATE 0.963 for "first quarter" and O 0.977
for "prior year" (0.967, 0.971, 0.897 for "broader aerospace", "company plan", "kestrel platform"), where the MEMM's
Viterbi path still opens a DATE ("the prior" = B-DATE I-DATE). The HMM follows the CRF, except that it reads "the kestrel
platform" as a PRODUCT (B-PRODUCT 0.962): the polyseme's emission outweighs its neighbors, whose words reach a tag only
through the transitions, never as features of it.

**Perceptron sweep** (teacher stream 3412; K = 3, V = 4, T = 4; 60 sequences): mistakes 3, 11, 13, 11, 18, 21 at
γ = 0.314899 … 0.041188, against bounds 867.3 … 50,695.1 (R ≈ 9.27).

## 4. The lab (`SequenceLabelingLab`, `data/sequence-labeling-lab.json`, 5 tabs)

1. **Trellis.** Test sequence 8 of the HMM world (rule: first with Viterbi ≠ MBR in ≥ 2 positions and a unique mode):
   8 states × 24 positions, cell shade = marginal (sum) or max-marginal (max) — a semiring toggle; both paths overlaid;
   gold marks. Lesson: one recursion, two answers; the paths differ at positions 1, 8, 9.
2. **CRF vs MEMM.** The finance sentence with a continuation selector; per token, each model's top tags and Viterbi tag.
   Lesson: the MEMM's bars before the edit never move.
3. **Learning curves.** World toggle, accuracy/log-loss toggle; Bayes line; the Cor 2 table beside it.
4. **Finance.** Acc / span F1 / unseen-word accuracy for the five taggers at 3/10/30 documents.
5. **Perceptron.** Mistakes vs the bound on a log axis, by margin.

## 5. Checkpoint questions (for Jonathan)

1. Difficulty `intermediate` (Thms 1–5 are short proofs; the h-transform is the hardest step)?
2. Keep all five lab tabs, or fold *Finance* into *Learning curves*?
3. State finding 4 (the corpus does not exercise label bias) in the page body, or only in the rigorFlag?

## 5b. rigorFlag (draft)

The finance tagging task is synthetic and templated: with 30 training documents every model passes 0.977 token accuracy,
nearly all remaining errors are names never seen in training (held-out names are the next topic's subject), and the left
context alone settles the one tag ("the") whose gold label depends on the next word, so the corpus barely exercises label
bias; the Prop 2 demonstration uses counterfactual continuations. Thm 3's equivalence is for indicator features; with
overlapping features the CRF's family is larger and no HMM matches it. Prop 2 and Cor 2 bind posterior-marginal decoding
only — a Viterbi-decoded MEMM still sees later words. Discriminative taggers are fit at λ = 1/n (MAP under a standard
normal prior), untuned, while the HMM's add-α is tuned on validation data. Thm 5 assumes separability; its bound is loose
by two to three orders of magnitude here.

## 6. Notebook cell map (26 cells)

1 intro · 2 imports · 3–4 Thm 1 (brute force) · 5–6 Topic 12 pin · 7–8 Prop 1 · 9–10 Thm 2 · 11–12 Thm 3 (h-transform)
· 13–14 Prop 2/Cor 2 table · 15–16 Thm 4 + trellis example · 17–18 Thm 5 sweep · 19–20 world curves · 21–22 finance ·
23–24 Prop 2 sentence · 25–26 harness.

## 7. References (verified 2026-10-26 against the papers)

- Collins, Michael. 2002. "Discriminative Training Methods for Hidden Markov Models: Theory and Experiments with Perceptron
  Algorithms." *EMNLP 2002*, 1–8. ACL Anthology W02-1001. Thm 1 (separable, R²/δ²), Thm 2 (non-separable), §2.5 averaging.
- Lafferty, John, Andrew McCallum, and Fernando C. N. Pereira. 2001. "Conditional Random Fields: Probabilistic Models for
  Segmenting and Labeling Sequence Data." *ICML 2001*, 282–89. https://repository.upenn.edu/cis_papers/159/
- McCallum, Andrew, Dayne Freitag, and Fernando Pereira. 2000. "Maximum Entropy Markov Models for Information
  Extraction and Segmentation." *ICML 2000*, 591–98 (per Sutton and McCallum's bibliography, ref. 74).
- Smith, Noah A., and Mark Johnson. 2007. "Weighted and Probabilistic Context-Free Grammars Are Equally Expressive."
  *Computational Linguistics* 33 (4): 477–91. https://doi.org/10.1162/coli.2007.33.4.477 — Cor 2 (C_HMM = C_MRF, Mealy
  features), Lemma 1 and Thm 2 (C_MEMM ⊊ C_HMM).
- Sutton, Charles, and Andrew McCallum. 2012. "An Introduction to Conditional Random Fields." *Foundations and Trends in
  Machine Learning* 4 (4): 267–373. https://doi.org/10.1561/2200000013 — §2.2 (pairs), §2.3 (HMM → CRF; converse
  "more complicated"), §5.1.3 (MEMMs, label bias, and the caveat about copying future words into features).
- Klein, Dan, and Christopher D. Manning. 2002. "Conditional Structure versus Conditional Estimation in NLP Models."
  *EMNLP 2002* (observation bias). Anthology ID to confirm at Phase B.
- Rabiner (1989) and Viterbi (1967) via formalStatistics `hidden-markov-models`.
