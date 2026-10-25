# Brief — `lexicon-sentiment-loughran-mcdonald`

**Lexicon Sentiment and the Loughran–McDonald Dictionary.** This is roadmap #18 (domain `classification-sentiment`).
Its one prerequisite, `linear-text-classifiers` (#17), is published (PR #32). Two nodes depend on it:
`aspect-based-sentiment` and `capstone-earnings-call-nlp`. It is not on the §6 watch-list. A grep of every sibling's
`main` found no coverage of Loughran–McDonald or lexicon sentiment.

**Phase A** (2026-10-25; approved by Jonathan with the recommended answers to §9: `intermediate`; open on the category simplex; induction as a qualified negative; six lab tabs). Plan-mode answers: an opt-in loader for the real dictionary, with no numbers from it on the
page; lexicon induction included; Claude writes every body. Results are numbered per type, as on the page:
Definitions 1–2, Propositions 1–2, Theorems 1–3.

- **Harness:** 22/22 pass under numpy==2.5.4, scipy==1.18.1, and scikit-learn==1.9.1, also with `-W error`. The run
  takes ≈ 30 s: the six learning curves take ≈ 15 s and Topic 17's split ≈ 3 s.
- **Notebook:** 28 cells. `jupyter execute` exits 0 in ≈ 33 s, and the notebook is committed without outputs.
- **Lab data:** `src/components/viz/data/lexicon-sentiment-lab.json`, 17 kB, pinned by `test_viz_mirror_matches_payload`.
- **Licensing:** `load_lm_dictionary()` reads a reader-supplied copy (`$LM_MASTER_DICTIONARY` or
  `~/.cache/formalnlp/lm/`). It downloads only on `fetch=True` / `FORMALNLP_FETCH_LM=1`, from the Google Drive file
  that the SRAF page links (verified 2026-10-25). The download path was never run, because no download was approved.
  `test_lm_dictionary_not_committed` greps `git ls-files`, and the parser is tested on a synthetic three-row CSV.

**What the prototypes changed against the plan:**

1. **Prop 1 grew.** The plan had only the symmetric-tilt world, in which net tone is the Bayes rule. On the corpus,
   the plug-in Bayes rule of a *category world* (rates of positive, negative, and other tokens per tone) reaches
   0.724. Net tone with tuned cutpoints reaches 0.636667, so the projection pos − neg discards a coordinate.
   - The reason: neutral filings have few polar words of either kind (u₊ = 0.967, u₋ = 0.191 for + vs 0).
   - Prop 1 is now (a) a category world's Bayes rule cuts the simplex by lines; (b) it is net tone iff u₊ = −u₋;
     (c) the tilt world is that case, with cutpoints ±τ.
2. **Thm 2 is stated for a lexicon subspace span(A).** The two cases are A = w_L (net tone) and A = [1_P, 1_N] (the
   categories). The dictionary's two counts beat unigram LR until n = 1,000.
3. **Sentence-level scoring is not worse than document-level.** Single-polarity aspect sentences score 0.758, about
   the same as the 0.748 sign accuracy on positive documents, because those sentences contain a lexicon word by
   construction. Thm 1's corpus check is instead that the i.i.d. model's *exact* error at the mean length (0.252959)
   matches the measured one (0.252000).
4. **Induction did not neutralize the polysemes.**
   - It recovers the signs of 25 of 28 non-seed polar words (0.892857).
   - The polysemes get |z| 0.907, close to the polar words' 1.06–1.08, and *tax*, *taxes*, *liability*, and *crude*
     land negative.
   - The ten most extreme words at each pole include one lexicon word.

   The page's claim is "induction scores a candidate list; it does not select one."
5. **The general list flips a − b.** In positive documents the general list's negative rate (0.026534) exceeds the
   positive rate (0.017917). Thm 1 then gives no bound, and the raw sign error rises toward 1 with length (0.989 at
   L = 3,000). That is Prop 2's mean shift at its extreme, and re-tuned cutpoints recover part of it (0.572).

---

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

All accuracies are on Topic 17's test split (`generate_corpus(1500, 1)`). Training is `(1500, 0)`, and dev is
`(600, 2)`, used only for λ selection.

| claim | value |
|---|---|
| finance stand-in hit rates, positive docs (a, b) | 0.017917 / 0.008160 |
| finance hit rates, neutral docs (pos, neg) | 0.006897 / 0.006828 |
| finance hit rates, negative docs (pos, neg) | 0.007573 / 0.017777 |
| net tone, cutpoints tuned on train: 3-class accuracy | 0.636667 |
| net tone cutpoints (lo, hi) | −0.006192 / 0.007505 |
| polar sign accuracy / tie share | 0.759000 / 0.128000 |
| category-world plug-in Bayes (Prop 1) | 0.724000 |
| slopes (u₊, u₋): + vs 0 / 0 vs − / + vs − | (0.967293, 0.190821) / (−0.105411, −0.968747) / (0.861882, −0.777926) |
| Topic 17 unigram LR | 0.738667 |
| full-text Bayes ceiling (Topic 17) | 0.889919 |
| Thm 1 ρ (finance, positive docs) | 0.998106 |
| mean length of positive docs / expected hits | 210.48 / 5.488681 |
| bound ρ^L̄ / i.i.d. exact / measured sign error | 0.670971 / 0.252959 / 0.252000 |
| length terciles (mean length → sign accuracy) | 182.34 → 0.728125, 210.07 → 0.776812, 242.13 → 0.770149 |
| general list, positive docs (a, b) | 0.017917 / 0.026534 |
| general list: 3-class / polar sign / ties | 0.572000 / 0.600000 / 0.053000 |
| Fisher ratio (+ vs −): finance / general | 3.653405 / 2.456848 |
| test means by tone (−, 0, +): finance | −0.009418 / −0.000484 / 0.009152 |
| test means by tone (−, 0, +): general | −0.027926 / −0.018599 / −0.008687 |
| polyseme rate c per tone (−, 0, +) | 0.018155 / 0.017698 / 0.018374 |
| EDGAR (tokens; stand-in negative; polysemes): Apple | 275; 2; 0 |
| EDGAR: ExxonMobil | 242; 0; 3 (cost, liabilities, tax) |
| EDGAR: Microsoft | 280; 0; 1 (taxes) |
| sentences: adjective / dir + comment / dir only / negated / verb up-good / verb cost-like / noun (n) | 1,205 / 1,746 / 595 / 302 / 446 / 352 / 525 |
| sentence accuracy k = 0: those rows | 1 / 1 / 0 (abstains) / 0 / 1 / 0 / 1; all 0.758461 (abstain 0.115065) |
| sentence accuracy k = 1 | negated → 1; all 0.816863 |
| document accuracy by scope k = 0, 1, 2, 3, 5, 8 | 0.636667, 0.696000 ×4, 0.694000 |
| false flips by k | 0, 0, 0, 0, 0, 36 (of 488 flips at k = 8; 452 at k = 1) |
| uncertainty per 100 tokens by tone (−, 0, +) | 1.618154 / 1.618265 / 1.564762 |
| litigious per 100 tokens: earnings call / MD&A / 8-K | 0 / 0 / 1.330306 |
| category-rate LR: polarity only / all four | 0.724667 / 0.724667 |
| "any uncertainty word → neutral" | 0.414000 |
| Thm 2 G (mean ‖x‖) | 26.363937 |
| Thm 2 net: λ·max‖Δ_k‖ at λ = 1, 10, 100, 1000 | 0.272590, 0.585478, 0.708987, 0.725470 |
| Thm 2 net: max probability gap to the limit | 0.467354, 0.159109, 0.021476, 0.002218 |
| Thm 2 categories: λ·max‖Δ_k‖ | 0.179352, 0.257888, 0.274739, 0.276675 |
| Thm 2 categories: gap | 0.304368, 0.052733, 0.005891, 0.000596 |
| curves n | 10, 30, 100, 300, 1000, 1500 |
| net tone (λ = ∞) | 0.546867, 0.590267, 0.620067, 0.630444, 0.639333, 0.639333 |
| categories (λ = ∞) | 0.622667, 0.692933, 0.712400, 0.718667, 0.716667, 0.716667 |
| support-only LR (word power) | 0.470667, 0.564667, 0.638533, 0.706000, 0.729333, 0.739333 |
| unigram LR | 0.399067, 0.465000, 0.531733, 0.614222, 0.719333, 0.738667 |
| prior, net | 0.549400, 0.646267, 0.700733, 0.722667, 0.750667, 0.766000 |
| prior, categories | 0.616400, 0.697467, 0.712467, 0.725333, 0.748000, 0.756667 |
| crossover (first n with LR ≥ dictionary): net / categories | 1000 / 1000 |
| λ path, net (λ = 10⁴ … 10⁻³) | 0.639333, 0.639333, 0.661333, 0.692000, 0.744000, 0.766000, 0.752667, 0.731333 |
| λ path, categories | 0.716667, 0.716667, 0.716667, 0.724000, 0.736667, 0.756667, 0.758667, 0.735333 |
| Thm 3 graph: vertices in the seeds' component / V (df ≥ 30) | 694 / 699 |
| sign accuracy on 28 non-seed polar words | 0.892857 |
| mean \|z\|: positive / negative / uncertainty / litigious / polysemes | 1.084542 / 1.055352 / 0.616591 / 0.818276 / 0.907476 |
| polyseme z | capital +0.918, cost +0.620, costs +0.171, crude −1.143, liabilities +0.767, liability −1.462, tax −0.911, taxes −1.267 |
| precision of the top / bottom 10 | 0.1 / 0.0 |
| 200 random seed sets: median / p10 / min sign accuracy | 0.821429 / 0.714286 / 0.428571 |

The page copies the payload's rounded values (3 decimals). Quote "almost three-fourths" for LM 2011's Harvard-IV
finding: it is the abstract's wording, verified on Crossref, and the 73.8% in the body was not checked.

## 0. Placeholder table

| | |
|---|---|
| TITLE | Lexicon Sentiment and the Loughran–McDonald Dictionary |
| SUBTITLE (draft) | A dictionary is a linear classifier whose weights were fixed before any label was seen. Net tone projects a document onto one direction of the category simplex and throws away the other. Length, contamination, and what no word list can read set its ceiling, and as a prior for logistic regression the dictionary is worth about a thousand labeled filings. |
| SLUG / DOMAIN | `lexicon-sentiment-loughran-mcdonald` / `classification-sentiment` |
| DIFFICULTY | `intermediate` (proposed; §9) |
| FINANCE_CASE_STUDY | `true` |
| PREREQUISITES | `[linear-text-classifiers]` |
| formalStatisticsPrereqs | `large-deviations` (Thm 6, Cramér's theorem: Thm 1's exponent); `regularization-and-penalized-estimation` (ridge, Thm 2's ancestor) |
| formalMLPrereqs | `concentration-inequalities` (Thm 3, the Chernoff bound); `graph-laplacians` (the Dirichlet form behind Thm 3) |
| formalMLConnections | `naive-bayes` (Prop 1 is NB on collapsed category counts); `random-walks` (absorption and hitting, Thm 3) |
| formalRAGConnections | `vector-space-model-tfidf` (LM 2011's tf–idf term weighting, cited, never derived: formalRAG owns TF-IDF) |
| REFERENCE_NOTEBOOKS | `notebooks/linear-text-classifiers/`, `notebooks/finance-corpus/` |

All seven sibling slugs were checked `status: "published"` on GitHub `main` (2026-10-25).

## 1. Section outline (geometric first)

1. **A dictionary reads a filing.** Open on an MD&A sentence and the stand-in lists. Net tone (pos − neg)/L is
   Loughran and McDonald's proportional measure.
2. **The category simplex (Def 1, Prop 1).**
   - The picture: every document is a point (p̂₊, p̂₋), and net tone is the anti-diagonal coordinate.
   - Prop 1(a)/(b): a category world's Bayes rule cuts by lines, and those lines are anti-diagonal iff u₊ = −u₋.
   - Prop 1(c): the tilt world gives the cutpoints ±τ.
   - The corpus: neutral filings sit near the origin, and net tone mixes them with balanced-but-dense ones (0.637
     against 0.724). Lab view 1.
3. **Length (Thm 1).**
   - The Chernoff bound with the Hellinger-like gap √a − √b, and Cramér tightness.
   - On the corpus: about 5.5 expected hits per filing, a bound of 0.671 against a true error of 0.252. The i.i.d.
     model predicts the measured error to three decimals.
   - The dense preset shows what a richer lexicon buys. Lab view 2.
4. **When is a liability not a liability (Prop 2).** Harvard-IV in 10-Ks, and the stand-in polysemes. Exact
   variance increments; Fisher's ratio drops from 3.65 to 2.46. The general list makes positive filings net-negative
   (b > a), and re-tuned cutpoints rescue only part of it (0.572). EDGAR counts. Lab view 3.
5. **What no lexicon sees (Def 2).**
   - The construction table: abstentions on bare directions, the valence trap, negation.
   - The negation scope sweep: k = 1 is exact on this corpus, and false flips appear at k = 8.
   - Topic 17's XOR, recalled.
   - Uncertainty ≠ neutrality: the rule scores 0.414, and the extra categories add 0.000 to LR on rates.

   Lab view 4.
6. **The dictionary as a prior (Thm 2).** Ridge orthogonal to span(A), ‖Δ_k‖ ≤ G/λ, and the λ = ∞ limit. Learning
   curves: the dictionary is worth about a thousand labels, and the prior is never more than 0.01 below the better end
   while beating both from n = 30. The λ path. Lab view 5.
7. **Inducing a lexicon (Thm 3).** The harmonic function, uniqueness, the maximum principle, and random walks. On the
   corpus: signs 0.893 and random seeds a median of 0.821, but the polysemes are not neutralized and the extremes are
   topical words. LM's curation is the step induction cannot automate. Lab view 6.
8. **The finance case study wrapper.** Sections 4 and 6 sit inside `<FinanceCaseStudy>`, along with the optional
   real-dictionary note (how to run the loader; no numbers).

## 2. Statements and proofs

**Def 1 (lexicon).** A lexicon is a family of disjoint word sets C₁, …, C_r. Here the categories are positive P,
negative N, uncertainty, and litigious. For a count vector x ∈ ℕ^V of length L = Σ_j x_j:
- the counts are n_c = Σ_{j ∈ C_c} x_j and the rates p̂_c = n_c/L;
- the weight vector is w_L = 1_P − 1_N;
- the raw score is s = w_L · x = n₊ − n₋, and the net tone is s/L = p̂₊ − p̂₋.

**Prop 1 (the category world).** Each tone t ∈ {−, 0, +} draws every token's category i.i.d. from
q_t = (q_{t+}, q_{t−}, q_{t0}), where q_{t0} covers every non-polar token. Within a category, the word is drawn from a
distribution r_c that does not depend on t. Priors are equal.

(a) The log posterior is log P(t | x) = n₊ log q_{t+} + n₋ log q_{t−} + n₀ log q_{t0} + c(x), with c(x) free of t.
Dividing by L, each class's score is affine in (p̂₊, p̂₋), so the Bayes regions are polygons cut by lines.

(b) For tones t, t′, set u_c = log(q_{tc}/q_{t′c}) − log(q_{t0}/q_{t′0}) and v = log(q_{t0}/q_{t′0}). The pair is
decided by a threshold on p̂₊ − p̂₋ iff u₊ = −u₋.

(c) Symmetric tilt. Let q₀ = (m, m, 1 − 2m), q₊ = (m e^β, m e^{−β}, 1 − 2m)/Z, and q₋ the mirror image, with
Z = 1 + 2m(cosh β − 1). Then the Bayes rule is: + iff s/L > τ, − iff s/L < −τ, neutral otherwise, with
τ = log Z/β > 0. The ± log-odds is 2βs.

*Proof.* (a) A word j in category c has probability θ_{tj} = q_{tc} r_c(j). So
log P(x | t) = log(L!/Π x_j!) + Σ_c n_c log q_{tc} + Σ_j x_j log r_{c(j)}(j). Only the middle sum depends on t, and
with equal priors the posterior is the softmax of n · log q_t. Since n₀/L = 1 − p̂₊ − p̂₋, each score divided by L is
affine in (p̂₊, p̂₋). The pairwise boundary is {u₊p̂₊ + u₋p̂₋ + v = 0}, a line.

(b) The pair's log-odds divided by L is u₊p̂₊ + u₋p̂₋ + v. It is a function of p̂₊ − p̂₋ alone on the open simplex iff
its gradient (u₊, u₋) is parallel to (1, −1), that is, iff u₊ = −u₋.

(c) For + against 0: u₊ = log(e^β/Z) − log(1/Z) = β, u₋ = −β, and v = −log Z. So + beats 0 iff βs/L > log Z. The
case − against 0 is symmetric. For + against −, u₊ = 2β, u₋ = −2β, and v = 0, so the log-odds is 2βs and + beats − iff
s > 0. Since cosh β > 1, Z > 1 and τ > 0, and the three pairwise rules agree on the stated regions. ∎

Harness: `test_prop1_category_world_is_linear` checks the direct multinomial likelihood (20 random worlds × 10 docs,
10⁻¹⁰) and antisymmetry in the tilt world. `test_prop1_tilt_world_bayes_rule` checks the closed form, 2βs, and the ±τ
decisions. `test_net_tone_discards_a_coordinate` checks u₊ > 0.5 with |u₋| < u₊/2 for + vs 0, and a gain of more than
0.07 on the corpus.

*Remark (unequal priors).* With priors π_t, add log π_t to each score. Divided by L the prior becomes a
length-dependent offset, so a normalized-score threshold is Bayes only when the priors are equal.

**Thm 1 (length).** Let X₁, …, X_L be i.i.d. with P(X = 1) = a, P(X = −1) = b, P(X = 0) = 1 − a − b, where a > b > 0,
and let S_L = Σ X_i. Then (a) P(S_L ≤ 0) ≤ ρ^L with ρ = 1 − (√a − √b)²; (b) (1/L) log P(S_L ≤ 0) → log ρ.

*Proof.* (a) For t ≥ 0, Markov's inequality gives P(S_L ≤ 0) = P(e^{−tS_L} ≥ 1) ≤ E e^{−tS_L} = M(t)^L, with
M(t) = a e^{−t} + b e^t + 1 − a − b. M is strictly convex and M′(t) = 0 at e^{2t} = a/b, that is, at
t* = ½ log(a/b) > 0. There M(t*) = 2√(ab) + 1 − a − b = 1 − (√a − √b)².

(b) Cramér's theorem (formalStatistics `large-deviations` Thm 6; Dembo and Zeitouni, Thm 2.2.3) applies to the mean
S_L/L, with Λ(λ) = log M(−λ) and the rate I(x) = sup_λ [λx − Λ(λ)]. Since E X = a − b > 0, I is convex, vanishes at
a − b, and is finite and continuous on (−1, 1). So inf over (−∞, 0) of I equals inf over (−∞, 0] of I, which is
I(0) = −inf_λ Λ(λ) = −log ρ. The upper and lower bounds therefore meet, and lim (1/L) log P(S_L ≤ 0) = −I(0) = log ρ. ∎

Harness: `test_thm1_bound_holds` checks the bound against the exact trinomial law (50 grid cases plus 200 random
ones) and the minimizer through `minimize_scalar`. `test_thm1_exponent_is_tight` checks that
(log P(S_{2L} ≤ 0) − log P(S_L ≤ 0))/L → log ρ, with a decreasing error below 10⁻³ at L = 1,000.
`test_length_on_the_corpus` checks the bound and that the i.i.d. exact error is within 0.02 of the measured one.

*Remark.* ρ = 2√(ab) + (1 − a − b) is the Bhattacharyya coefficient between the hit distributions of the two polar
tones, so 1 − ρ = (√a − √b)² is their squared Hellinger distance: a lexicon's power is how far apart its hit rates sit
on that scale. A positive document is called correctly once
L(√a − √b)² is large: on the corpus that means L in the thousands, against a mean of 210.

**Prop 2 (contamination).** Add to the negative list a word set Z that is disjoint from P and N and has rate c in
every tone, its tokens taken from the "other" mass. Per token, for positive documents (rates a, b) and negative ones
(b, a):
- (i) the gap between the two tones' means stays 2(a − b);
- (ii) the variances rise by exactly c(1 − c + 2(a − b)) and c(1 − c − 2(a − b));
- (iii) if a ≠ b and c + 2|a − b| < 1, Fisher's ratio (μ₊ − μ₋)²/((σ₊² + σ₋²)/2) falls strictly.

The same holds per document, with means and variances multiplied by L.

*Proof.* The contaminated token score is +1 with probability a and −1 with probability b + c. Its mean is a − b − c
and its second moment a + b + c, so its variance is a + b + c − (a − b − c)². Subtracting the clean variance
a + b − (a − b)² leaves c − c² + 2c(a − b). The negative tone has mean b − a − c, and the same computation gives
c − c² − 2c(a − b). Both means shift by −c, so the gap is unchanged. Under the stated condition both increments are
positive, so the denominator of the ratio grows while its numerator is fixed. Over L i.i.d. tokens, means and
variances scale by L and the ratio by L. ∎

Re-tuning the cutpoints moves them by −c, absorbing the mean shift. The variance stays, and with it the overlap.
Harness: `test_prop2_contamination_moments` (500 random (a, b, c), 10⁻¹⁴) and `test_general_lexicon_misreads_finance`
(c tone-independent within 20%, every mean shifted by about −c, every variance up, the ratio down, 3-class accuracy
down by more than 0.03 and sign accuracy by more than 0.1).

**Def 2 (negation scope k).** A polar hit at position i counts for the other side if a negator (*not*, *no*, *never*)
occurs at some position in [i − k, i − 1]. Das and Chen (2007) mark negated words, and VADER (Hutto and Gilbert 2014)
looks three tokens back.

**Thm 2 (the dictionary as a prior).** Let A ∈ ℝ^{V×r} have full column rank, and fit softmax LR with
logits_k(x) = x·Δ_k + (Aᵀx)·β_k + b_k by minimizing J = CE(Δ, β, b) + (λ/2) Σ_k ‖Δ_k‖², with no penalty on β or b.

(a) In terms of W_k = Δ_k + Aβ_k, J is minimized exactly when W minimizes CE(W, b) + (λ/2) Σ_k ‖P⊥W_k‖², where P⊥ is
the orthogonal projector onto span(A)^⊥, and Δ_k = P⊥W_k. In particular AᵀΔ_k = 0.

(b) At the optimum, ‖Δ_k‖ ≤ G/λ, where G = (1/n) Σ_i ‖x_i‖.

(c) Suppose the λ = ∞ problem (the softmax on Aᵀx with intercepts) has a minimizer that is nondegenerate modulo the
softmax's common shift. Then the fitted probabilities converge to its probabilities, and max_x |p_λ(x) − p_∞(x)| =
O(1/λ) on any bounded set of x.

*Proof.* (a) The loss depends on (Δ, β) only through W. For a fixed W_k, minimizing ‖Δ_k‖² over the decompositions
W_k = Δ_k + Aβ_k is least squares in β_k, with value ‖P⊥W_k‖², attained at Δ_k = P⊥W_k.

(b) Stationarity in Δ_k gives ∇_{W_k}CE + λΔ_k = 0. (Stationarity in β_k gives Aᵀ∇_{W_k}CE = 0, which is (a)'s
orthogonality again.) Since ∇_{W_k}CE = (1/n) Σ_i (p_{ik} − y_{ik}) x_i and |p_{ik} − y_{ik}| ≤ 1, we get
‖Δ_k‖ ≤ G/λ.

(c) With Δ fixed, (β, b) minimizes the convex function CE(Δ, ·, ·), whose stationarity equations at Δ = 0 are the
λ = ∞ problem's. Nondegeneracy (a positive-definite Hessian once the common shift is fixed) and the implicit function
theorem give a smooth branch (β(Δ), b(Δ)) of minimizers near Δ = 0. By convexity the minimizer at Δ_λ lies on that
branch, so ‖(β_λ, b_λ) − (β_∞, b_∞)‖ = O(‖Δ_λ‖). Logits change by at most ‖x‖‖Δ_λ‖ + O(‖Δ_λ‖), and softmax is
1-Lipschitz in sup norm, so by (b) the probability gap is O(1/λ). ∎

Harness: `test_thm2_prior_limit`, for both A:
- orthogonality below 10⁻⁶ at every λ (measured 10⁻¹⁰–10⁻¹⁴);
- λ‖Δ_k‖ ≤ G (measured 0.73 ≤ 26.4: G is loose);
- the gap shrinking by more than 5× per decade (measured ≈ 7–10×);
- β at λ = 1,000 within 5·10⁻³ of the limit.

`test_dictionary_versus_labels` covers the curves. The λ → 0 end is not in the theorem: unpenalized LR does not exist
on separable data (Topic 17, Thm 4), and the path's λ = 10⁻³ end is just ordinary small-ridge LR.

**Thm 3 (harmonic induction).** Let W be a symmetric nonnegative weight matrix on vertices S ∪ U, with seed values
f_S ∈ {−1, +1}^S, where every vertex of U is joined to S by a path of positive weight, and D = diag(W1). Then:
- (a) L_UU = (D − W)_UU is positive definite, so (D − W)_UU f_U = W_US f_S has exactly one solution;
- (b) f_v = Σ_j (W_vj/d_v) f_j for v ∈ U, and min f_S ≤ f ≤ max f_S;
- (c) f_v = 2h_v − 1, where h_v is the probability that the random walk with steps P = D⁻¹W, started at v, reaches a
  +1 seed before a −1 seed;
- (d) clamped propagation f ← D⁻¹Wf (seeds reset) converges to f.

*Proof.* (a) For g ∈ ℝ^U, gᵀL_UU g = ½ Σ_{i,j∈U} W_ij (g_i − g_j)² + Σ_{i∈U} g_i² Σ_{s∈S} W_is ≥ 0. If this is 0,
then g is constant on each connected component of U, and g_i = 0 at every i adjacent to S. Every component of U
contains such a vertex, so g = 0.

(b) The averaging identity is the v-th row of the system. Let M = max_U f, and suppose M > max f_S. At a maximizer v,
f_v is an average of neighbors' values that are all ≤ M, so every neighbor also equals M. Following a path from v to S
reaches a seed of value M > max f_S, a contradiction. The minimum is symmetric.

(c) The walk is absorbed in S almost surely (a finite chain from which S is reachable everywhere). Conditioning on the
first step gives h_v = Σ_j P_vj h_j on U, with h = 1 on S₊ and h = 0 on S₋. The function 2h − 1 satisfies the same
averaging and takes the boundary values ±1, so by (a) it equals f.

(d) The iteration is Jacobi for (a)'s system. Its matrix D_UU⁻¹W_UU is substochastic, and the mass leaks to S from
every component, so its spectral radius is below 1. ∎

Harness: `test_thm3_harmonic_properties` uses 10 random weighted graphs to check positive definiteness, the maximum
principle, and that propagation after 20,000 sweeps equals the solve to 10⁻⁸. On a 5-vertex graph, 20,000 Monte Carlo
walks agree with 2h − 1 to 4 standard errors. `test_induction_on_the_corpus` checks:
- sign accuracy ≥ 0.85, and a random-seed median ≥ 0.75;
- polysemes' mean |z| > 0.7 × the polar words';
- uncertainty below polar;
- precision of the top and bottom 10 ≤ 0.2.

## 3. Corpus and finance measurements

- **Split:** Topic 17's (`ltc.fin_split()`, cached), tokens `[a-z]+`.
- **Cutpoints:** exhaustive search over midpoints of the training scores (`tune_thresholds`), with no dev set needed.
- **Constructions:** classified from the generator's own templates (`construction`). Sentences with mixed or zero
  aspect polarity, or with ", but ", are excluded (5,171 kept).
- **Learning curves:** n ≤ 100 average 10 random subsets, n = 300 averages 3, larger n use one (stream
  `[SEED, 3102]`). λ grids: support {0.1, 0.01, 0.001}, LR Topic 17's {0.01, 0.001}, prior {1, 0.1, 0.01}, all chosen on
  dev. A ridge η = 10⁻⁴ on the dictionary slopes guards against small samples that separate on one count.
- **Graph:** document-level PPMI over types with df ≥ 30 (V = 699; 5 vertices with no positive edge drop out of the
  seeds' component). Seeds are *strong*, *robust* / *weak*, *adverse*, and z centers by the median and scales by the
  sd over determined vertices.
- **EDGAR:** stand-in lists only, never the real dictionary.

## 4. The lab (`LexiconSentimentLab`, `data/lexicon-sentiment-lab.json`)

Six tabs, with every printed number from the payload:

1. **Simplex.** A scatter of 600 test filings in (p̂₊, p̂₋) per 100 tokens, colored by tone. Toggle:
   - net-tone cutpoints (two anti-diagonal lines);
   - the category-world Bayes regions (shaded from `logQ`).

   Readouts: 0.637 / 0.724 / 0.739 (LR). *Learn:* neutral sits at the origin, and lines parallel to the anti-diagonal
   cannot isolate it.
2. **Length.**
   - Exact error and Thm 1's bound against L (log axis), for three presets: finance, general, dense.
   - A slider over the L grid with readouts.
   - The corpus marker at L̄ = 210.5 (measured 0.252, i.i.d. 0.253, bound 0.671).

   The general preset shows "no bound: b > a". *Learn:* the bound is loose but has the right exponent, and density
   matters more than length.
3. **Contamination.** Net-tone histograms per tone, finance ⇄ general, with the cutpoints. Readouts: accuracy, sign
   accuracy, Fisher ratio, and c per tone. *Learn:* everything slides left, and the overlap grows.
4. **Constructions.** Bars of sentence accuracy per construction, with a k = 0/1 toggle, plus the negation sweep
   (document accuracy and false flips against k). *Learn:* which failures a scope rule fixes (negation) and which no
   word list can (direction, the valence trap).
5. **Prior.** Learning curves for the six methods on a log n axis, with the Bayes ceiling (0.890) and Topic 17's LR as
   references. A toggle shows the λ path for either subspace. *Learn:* the dictionary is worth about 1,000 labels, and
   the prior gets the best of both.
6. **Induction.** A strip plot of z per word by category, with seeds ringed and polysemes in their own row. Readouts:
   sign accuracy, mean |z| per category, and the top/bottom-10 lists. *Learn:* the signs are mostly right, but the
   magnitudes don't separate polar from topical words.

## 5. rigorFlag (draft)

> Prop 1 and Thm 1 assume tokens whose categories are i.i.d. given the tone, which no real text satisfies; the
> synthetic corpus breaks it too (its tone lives in five to eight aspect sentences), though the i.i.d. model happens to
> predict the measured sign error almost exactly. Thm 1's tightness is Cramér's theorem, cited rather than proved.
> Prop 2's "general-purpose list" is the corpus stand-in plus eight planted finance polysemes, not the Harvard-IV
> dictionary; Loughran and McDonald's finding that almost three-fourths of Harvard negative words in 10-Ks are not
> negative in context is quoted from their paper, not reproduced, and no number on this page comes from their
> dictionary, which is licensed for academic research and never committed here. Negation scope k = 1 is exact on this
> corpus only because its one negation template is "not ADJ". Thm 2(c) assumes a nondegenerate limit, and its
> constant G is loose by a factor of about 36 on this corpus. The learning curves choose every λ on a 600-document dev
> set, a luxury when labels are the scarce resource. Thm 3's induction runs on a synthetic corpus whose templates fix
> which words share documents; the tf–idf weighting Loughran and McDonald also propose is cited, not tested.

## 6. Notebook cell map (28 cells)

| # | kind | content |
|---|---|---|
| 1 | md | title, six points, licensing note |
| 2 | code | imports, lexicon sizes, split sizes |
| 3–4 | md/code | Def 1, hit rates by tone |
| 5–7 | md/code/md | Prop 1, slopes, accuracies, why neutral breaks net tone |
| 8–10 | md/code/md | Thm 1, checks, corpus numbers |
| 11–12 | md/code | Prop 2, finance vs general, EDGAR |
| 13–16 | md/code/md/code | constructions, negation sweep, uncertainty and litigious |
| 17–20 | md/code/code/md | Thm 2, limit table, learning curves, reading |
| 21–23 | md/code/md | Thm 3, induction, interpretation |
| 24–25 | md/code | optional real dictionary (skips cleanly) |
| 26–27 | md/code | the whole harness |
| 28 | md | where this sits |

## 7. Cross-site and scope

- **Prereqs:**
  - formalStatistics `large-deviations`: Thm 6, Cramér's theorem for i.i.d. means (Thm 1(b)).
  - formalML `concentration-inequalities`: Thm 3, the Chernoff bound (Thm 1(a)).
  - formalML `graph-laplacians`: the Laplacian quadratic form (Thm 3(a)).
  - formalStatistics `regularization-and-penalized-estimation`: Thm 1, ridge (Thm 2 is ridge toward a subspace).
- **Connections:**
  - formalML `naive-bayes`: Prop 1 is multinomial NB on three collapsed categories.
  - formalML `random-walks`: hitting and absorption (Thm 3(c)).
  - formalRAG `vector-space-model-tfidf`: LM's tf–idf weighting, cited only.
- **Scope:** no TF-IDF derivation (formalRAG), no retrieval, no agentic material. Aspect-level sentiment is Topic 19,
  which we foreshadow but do not cover.

## 8. References (verified 2026-10-25)

Journal articles:
- Tim Loughran and Bill McDonald. 2011. "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and
  10-Ks." *Journal of Finance* 66 (1): 35–65. https://doi.org/10.1111/j.1540-6261.2010.01625.x (Crossref; abstract's
  "almost three-fourths").
- Tim Loughran and Bill McDonald. 2016. "Textual Analysis in Accounting and Finance: A Survey." *Journal of Accounting
  Research* 54 (4): 1187–1230. https://doi.org/10.1111/1475-679X.12123
- Paul C. Tetlock. 2007. "Giving Content to Investor Sentiment: The Role of Media in the Stock Market." *Journal of
  Finance* 62 (3): 1139–68. https://doi.org/10.1111/j.1540-6261.2007.01232.x
- Narasimhan Jegadeesh and Di Wu. 2013. "Word Power: A New Approach for Content Analysis." *Journal of Financial
  Economics* 110 (3): 712–29. https://doi.org/10.1016/j.jfineco.2013.08.018
- Sanjiv R. Das and Mike Y. Chen. 2007. "Yahoo! for Amazon: Sentiment Extraction from Small Talk on the Web."
  *Management Science* 53 (9): 1375–88. https://doi.org/10.1287/mnsc.1070.0704
- Maite Taboada, Julian Brooke, Milan Tofiloski, Kimberly Voll, and Manfred Stede. 2011. "Lexicon-Based Methods for
  Sentiment Analysis." *Computational Linguistics* 37 (2): 267–307. https://doi.org/10.1162/COLI_a_00049

Conference papers:
- C. J. Hutto and Eric Gilbert. 2014. "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media
  Text." *ICWSM* 8 (1): 216–25. https://doi.org/10.1609/icwsm.v8i1.14550
- Peter D. Turney. 2002. "Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of
  Reviews." ACL 2002. https://aclanthology.org/P02-1053/
- Vasileios Hatzivassiloglou and Kathleen R. McKeown. 1997. "Predicting the Semantic Orientation of Adjectives." ACL
  1997. https://aclanthology.org/P97-1023/
- William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. 2016. "Inducing Domain-Specific Sentiment
  Lexicons from Unlabeled Corpora." EMNLP 2016. https://aclanthology.org/D16-1057/
- Xiaojin Zhu, Zoubin Ghahramani, and John Lafferty. 2003. "Semi-Supervised Learning Using Gaussian Fields and
  Harmonic Functions." ICML 2003, 912–19.
  https://aaai.org/papers/icml03-118-semi-supervised-learning-using-gaussian-fields-and-harmonic-functions/

Books and documentation:
- Peter G. Doyle and J. Laurie Snell. 1984. *Random Walks and Electric Networks*. MAA. https://arxiv.org/abs/math/0001057
  (harmonic functions, uniqueness, the maximum principle).
- Amir Dembo and Ofer Zeitouni. 2010. *Large Deviations Techniques and Applications*. 2nd ed. Springer.
  https://doi.org/10.1007/978-3-642-03311-7 (Thm 2.2.3, Cramér).
- Loughran–McDonald Master Dictionary, documentation. https://sraf.nd.edu/loughranmcdonald-master-dictionary/
  (type `documentation`; free for academic research).

## 9. Open questions for the checkpoint

1. **Difficulty.** `intermediate` is proposed: Chernoff/Cramér, least squares, and a discrete Dirichlet problem.
2. **Lead.** Should §2 open on the category simplex, where net tone loses 0.087 to the two counts? That reframes LM's
   proportional score as a projection. The alternative is to open on length (Thm 1), as the plan had it.
3. **Induction's verdict.** Report it as a qualified negative ("scores candidates, does not select them"), or drop Thm
   3's corpus section to a remark and keep only the theorem?
4. **Lab size.** Six tabs. Merge Constructions into Contamination ("what a word list misreads") for five?
