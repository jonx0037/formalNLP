# Brief — `aspect-based-sentiment`

**Aspect-Based Sentiment Analysis.** This is roadmap #19 (domain `classification-sentiment`). Its two prerequisites
are published: `lexicon-sentiment-loughran-mcdonald` (#18, PR #34) and `attention-as-kernel-smoothing` (#9). No node
depends on it before the capstone. It is not on the §6 watch-list. A grep of every sibling's `main` found no aspect-based
or targeted sentiment.

**Phase A** (2026-10-25). Plan-mode answers:
- Theorem spine: the signed vote plus identifiability.
- A one-layer ladder built from scratch.
- Gold aspect spans, with term extraction deferred to Topics 21–22.
- EDGAR as qualitative readings.

Checkpoint (2026-10-25): approved by Jonathan with the recommended answers to §9 (`intermediate`; Thm 1 kept as one theorem; EDGAR in lab tab 4; the scope-rule remark stated in §4).

Results are numbered per type, as on the page: Definitions 1–2, Propositions 1–2, Theorems 1–2.

- **Harness:** 16/16 pass under numpy==2.5.4, scipy==1.18.1, and scikit-learn==1.9.1, also with `-W error`. The run
  takes ≈ 27 s:
  - three corpora ≈ 4 s;
  - the three attention rungs ≈ 12 s (300 full-batch Adam steps each);
  - the blind LR ≈ 2 s;
  - Topics 17/18's document baselines ≈ 5 s.
- **Notebook:** 25 cells. `jupyter execute` exits 0 in ≈ 32 s, and the notebook is committed without outputs.
- **Lab data:** `src/components/viz/data/aspect-sentiment-lab.json`, 13 kB, pinned by `test_viz_mirror_matches_payload`.

**What the prototypes changed against the plan:**

1. **Thm 1(b) became a necessity theorem.** The plan said valence × direction is representable because the query is
   the aspect. It is, but only one way. With votes shared across aspects, a vote's sign is aspect-independent, so the
   matched quadruple {A rose +, A fell −, B rose −, B fell +} is fitted only if *rose* and *fell* vote the **same** sign
   and the query amplifies one of them against the context. The direction words cannot vote their meaning.
   - The trained shared rung confirms it. Its ± votes are rose −7.22, fell −7.65, declined −7.71, decreased −7.62, but
     increased +5.88 and climbed +5.80.
   - It gets exactly the quadruples built from same-sign pairs fully right: 0.309 ≈ 3/9.
   - The ladder gained a rung, *aspect-conditioned values* (ATAE-LSTM's move of putting the aspect in the input).
2. **Thm 1(c) is stated as monotone substitution.** Swapping item t for t′ at one offset moves the vote by
   e^{π(d)}(β(t′) − β(t)). Its sign depends on neither the context nor the offset, and negation is a context that
   reverses an adjective's effect.
   - The first contrast set used an unseen filler ("also"). It made the bigram rung fail too, because "also" is
     `<unk>`.
   - The contrast set now deletes "not" ("T was ADJ this period"), which stays in distribution.
3. **Thm 2 preserves decisions, not confidence.** Every logit vector is multiplied by λ(x) = Σc/Σcr. Moving the
   attention onto the aspect words takes r = 10⁶, since the trained model gives them only 0.0055, and at that r the
   logits shrink 10⁻²–10⁻⁶ and the probabilities flatten toward 1/3. This is Topic 16's temperature in disguise.
4. **The aggregation lands on the ceiling.**
   - Gold aspect polarities through the Bayes rule give 0.882667, within one SE (0.0081) of 0.889919.
   - The bigram rung's two sentence errors change no decision.
   - Every document-level reader on the site sat at 0.637–0.766.

---

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

The splits are Topic 17's:
- train `generate_corpus(1500, 0)`: 14,267 aspects;
- dev `(600, 2)`: used only for the blind LR's λ;
- test `(1500, 1)`: 14,179 aspects.

| claim | value |
|---|---|
| test aspects / per document | 14,179 / 9.452667 |
| polarity counts (−, 0, +) | 3,890 / 6,492 / 3,797 |
| aspects in two-aspect sentences m (all opposite polarity) | 2,516 (1,258 sentences) |
| Prop 1 bound 1 − m/2N | 0.911277 |
| construction counts | adjective 1,824; dir + comment 2,775; dir only 908; negated 452; verb up-good 690; verb cost-like 513; noun phrase 525; neutral 6,492 |
| lexicon (Topic 18, k = 1, sentence): all / contrast | 0.784752 / 0.163355 |
| lexicon by construction | adj 0.732456, dir + comment 0.692252, dir only 0.027533, negated 0.727876, verb up 0.723188, verb cost 0.000000, noun 1, neutral 1 |
| blind bigram LR (λ = 10⁻³, dev 0.886783): all / contrast | 0.887792 / 0.500000 |
| blind by construction | adj 0.861842, dir + comment 0.896577, dir only 0.700441, negated 0.752212, verb up 0.643478, verb cost 0.177388, noun 1, neutral 1 |
| shared rung: all / contrast | 0.958953 / 0.897059 |
| shared by construction | adj 0.995614, dir + comment 0.989550, dir only 0.763216, negated 0.508850, verb up 0.853623, verb cost 0.986355, noun 1, neutral 1 |
| aspect-values rung: all / contrast | 0.984131 / 0.968998 |
| aspect by construction | 1 everywhere except negated 0.506637, verb up 0.997101 |
| bigram rung: all / contrast | 0.999859 / 0.999205 (negated 0.995575; 2 errors in 14,179) |
| shared ± votes | rose −7.223, increased +5.884, climbed +5.796, fell −7.655, declined −7.714, decreased −7.621, exceeded −2.480, missed +7.508 |
| Thm 1(b) LP: feasible (sign u_up, sign u_dn), shared / aspect votes | {(+,+), (−,−)} / adds (+,−), (+,0), (0,−) |
| contrast sets | 2,205 valence quadruples; 18,120 negation quadruples |
| all-four-right share, valence: lexicon / blind / shared / aspect / bigram | 0 / 0 / 0.308844 / 1 / 1 |
| all-four-right share, negation | 1 / 1 / 0 / 0 / 1 |
| Thm 2 (bigram rung, test, r = 10⁶ on term words): same argmax | true; logit-ratio spread 2.8·10⁻⁸; vote deviation 1.7·10⁻¹³ |
| Thm 2: attention on term words before / after (mean, min) | 0.005496 / 0.998204, 0.989871 |
| Thm 2: logit scale λ range at r = 10⁶ | 0.000006 – 0.010130 |
| Prop 2: tone sentences / mean k / k = 5, 6, 7, 8 | 9,695 / 6.463333 / 393, 377, 372, 358 |
| Prop 2: gold aggregation / bigram-rung aggregation | 0.882667 / 0.882667 |
| Prop 2: sentence errors / ε / documents with an error | 2 / 0.000206 / 2 |
| Prop 2: ceiling (Topic 17) / SE at n = 1,500 | 0.889919 / 0.008081 |
| document readers: net tone / categories (Topic 18) / unigram LR (Topic 17) | 0.636667 / 0.724000 / 0.738667 |
| EDGAR readings (bigram rung; no gold) | Apple "gross margin" (54 tokens) negative 0.743; ExxonMobil "liabilities" positive 0.495; Microsoft "revenue" positive 0.559, "income taxes" negative 0.484, "revenue" positive 0.497; the lexicon calls 4 of 5 neutral |

The page copies the payload's rounded values (3 decimals).

## 0. Placeholder table

| | |
|---|---|
| TITLE | Aspect-Based Sentiment Analysis |
| SUBTITLE (draft) | "Revenue rose, but costs climbed" holds two opinions, and no reader of the sentence alone gets both. One attention layer whose query is the aspect is a signed vote over the words. It can stretch a word's vote but never flip it, which decides what it can read (valence, with help) and what it cannot (negation). Its attention weights are not an explanation. Read aspect by aspect, the corpus gives up the document tone that every document-level reader missed. |
| SLUG / DOMAIN | `aspect-based-sentiment` / `classification-sentiment` |
| DIFFICULTY | `intermediate` (proposed; §9) |
| FINANCE_CASE_STUDY | `true` |
| PREREQUISITES | `[lexicon-sentiment-loughran-mcdonald, attention-as-kernel-smoothing]` |
| formalMLPrereqs | `kernel-regression` (Nadaraya–Watson: Def 2's position kernel and Topic 9's reading) |
| formalMLConnections | `naive-bayes` (Prop 2's aggregation is NB on sentence-polarity counts) |
| formalRAGConnections | `cross-encoders-reranking` (aspect-blind vs aspect-conditioned is the dual- vs cross-encoder interaction question) |
| REFERENCE_NOTEBOOKS | `notebooks/lexicon-sentiment-loughran-mcdonald/`, `notebooks/attention-as-kernel-smoothing/` |

All four sibling slugs were checked `status: "published"` on GitHub `main` (2026-10-25).

## 1. Section outline (geometric first)

1. **An opinion has a target.**
   - Open on the curated contrast sentence: "Liquidity declined 11 percent in rebateware, a trend executives
     characterized as disappointing, but net sales in clinicing midroutes were healthy."
   - Def 1: aspect, category, valence, polarity. ASC on gold spans; term extraction is Topics 21–22.
   - Prop 1: the aspect-blind ceiling, 0.911. The lexicon scores 0.785 and the blind LR 0.888, with the LR at exactly
     0.500 on contrast aspects.
2. **One layer, one vote (Def 2, Thm 1(a)).**
   - The picture: each word is a signed bar, and the aspect's attention stretches every bar by a positive factor.
     Topic 9's Prop 1 (attention is Nadaraya–Watson) is recalled.
   - The identity sign(ℓ_k − ℓ_l) = sign Σ e^{s_i} u_i^{kl}.
   - Lab view 1.
3. **Valence: the words that cannot mean what they say (Thm 1(b)).**
   - The matched quadruple, and the same-sign necessity.
   - The trained shared rung's votes, with the contrast sets: 0.309 against 1.000.
   - Aspect-conditioned values (ATAE) let the vote's sign depend on the aspect.
   - Lab view 2.
4. **Negation: what a vote cannot do (Thm 1(c)–(d)).**
   - Monotone substitution, and the four-sentence contradiction.
   - Both unigram rungs score 0 on 18,120 quadruples.
   - Bigram items fix it. Topic 18's k = 1 scope rule is a hand-written bigram, and the blind LR has bigrams, so both
     read them too.
   - Topic 17's XOR, now inside the context.
5. **Attention is not the explanation (Thm 2).**
   - The reweighting map.
   - The trained model gives the aspect word 0.0055 of its attention. An equivalent model gives it 0.998, with every
     decision unchanged.
   - The confidence flattens, which is a temperature (Topic 16).
   - The vote αᵢ(zᵢ + b) is the invariant (Kobayashi et al.'s ‖αf(x)‖).
   - Jain and Wallace; Wiegreffe and Pinter.
   - Lab view 3.
6. **From aspects to tone (Prop 2), inside `<FinanceCaseStudy>`.**
   - Tone-bearing sentences, the Bayes rule on counts, and the E[k]·ε bound.
   - 0.883 against the ceiling of 0.890 and the document readers' 0.637–0.739.
   - EDGAR: four of five mentions read polar at p ≤ 0.56. Out of its templates the reader has no concept of "neutral".
   - FiQA Task 1 is the real benchmark.
   - Lab view 4.

## 2. Statements and proofs

**Def 1 (aspect-level sentiment).** A sentence x = (x₁, …, x_n) and an aspect a: a span of x with a category c(a) ∈ C
and a valence v(c) ∈ {−1, +1}. Valence +1 means "up is good for the company" (revenue, margin, demand, liquidity), and
−1 means "up is bad" (costs, tax, liabilities). The aspect's polarity is y ∈ {−, 0, +}. On the corpus:
- a direction clause has y = v(c)·δ, where δ = ±1 is the move;
- an evaluative adjective or noun carries y itself;
- "not ADJ" carries −polarity(ADJ);
- a verb of expectation ("exceeded", "missed") carries v(c)·sign(verb).

Aspect sentiment classification (ASC) predicts y from (x, a). Aspect term extraction is span labeling, deferred to
Topics 21–22.

**Prop 1 (the aspect-blind ceiling).** Let a test set of N aspects contain m aspects that sit two to a sentence with
opposite polarities. Any classifier f(x) of the sentence alone has accuracy ≤ 1 − m/(2N).

*Proof.* Both aspects of such a sentence present f with the same x and have different labels, so f(x) equals at most
one of them. At least m/2 aspects are therefore misread. ∎

Harness: `test_prop1_aspect_blind_ceiling` checks the bound (0.911277), at most one right per pair for both blind
readers, and the blind LR's contrast accuracy of exactly 1/2.

**Def 2 (an aspect-query attention layer).**
- The items of x are its tokens, plus, with bigram items, each adjacent pair (placed at its right token). Item i has a
  type tᵢ and an offset dᵢ from the aspect span (0 inside it).
- Scores are separable in content and position: sᵢ = κ(c, tᵢ) + π(c, dᵢ), with α = softmax(s).
- Values live in logit space, zᵢ ∈ ℝ^K, as zᵢ = Wv(tᵢ) or, with aspect-conditioned values, Wv(c, tᵢ). The logits are
  ℓ = Σᵢ αᵢzᵢ + b_c.
- κ(c, t) = ⟨q(c), k(t)⟩ is a query-key score, so this is Topic 9's attention with the aspect as the query, plus a
  position kernel e^{π(c,d)}.
- *Shared votes:* z and b do not depend on c, so the aspect enters only through the query.

**Thm 1 (one layer is a signed vote).** Write uᵢ^{kl} = (zᵢ + b_c)_k − (zᵢ + b_c)_l for item i's vote for class k over
class l.

(a) ℓ_k − ℓ_l = Σᵢ αᵢuᵢ^{kl}, so sign(ℓ_k − ℓ_l) = sign(Σᵢ e^{sᵢ}uᵢ^{kl}). For a fixed aspect, each decision is
therefore a linear threshold function of the bag of (type, offset) items. Its weights e^{κ(c,t)+π(c,d)}u^{kl}(c, t) are
positive rescalings of signed votes.

(b) *Valence needs aspect-dependent votes.* Let A and B be aspects of valence +1 and −1, let w↑, w↓ be direction
words, and let the four sentences {A w↑, A w↓, B w↑, B w↓} share a context and the direction slot's offset, labeled
+, −, −, +. Suppose the votes are shared. Then a layer reads all four (±-pair) only if u(w↑) and u(w↓) are nonzero
with the same sign. Then the context's summed vote under each aspect has the opposite sign, and the query must weight
w↑ above w↓ (relative to |u|) for A and below it for B. Conversely, every such configuration fits the four. With
aspect-conditioned votes, (u_A(w↑), u_A(w↓)) = (+, −) is also allowed.

(c) *Substitution is monotone, so negation is out of reach.* For a fixed aspect, replacing the type t by t′ at offset
d changes Σᵢ e^{sᵢ}uᵢ^{±} by e^{π(c,d)}(β(t′) − β(t)), where β(t) = e^{κ(c,t)}u^{±}(c, t). Its sign does not depend on
the rest of the sentence or on d. Hence no layer reads all four of {T was ADJ₊ (+), T was ADJ₋ (−), T was not ADJ₊ (−),
T was not ADJ₋ (+)}.

(d) With bigram items, the four sentences are read correctly by some layer.

*Proof.*
- (a) Since Σᵢαᵢ = 1, ℓ_k − ℓ_l = Σᵢ αᵢ(zᵢ,k − zᵢ,l) + (b_{c,k} − b_{c,l}) = Σᵢ αᵢuᵢ^{kl}. Multiplying by
  Σⱼ e^{sⱼ} > 0 gives the sign identity.
- (b) Fix the ± pair. For aspect A, write F_A for the shared context's summed vote Σ e^{s}u and ω_A(w) = e^{s_A(w)} > 0
  for the direction slot's weight. Reading A correctly means F_A + ω_A(w↑)u(w↑) > 0 > F_A + ω_A(w↓)u(w↓), so
  ω_A(w↑)u(w↑) > ω_A(w↓)u(w↓). Reading B means ω_B(w↑)u(w↑) < ω_B(w↓)u(w↓).
  - If u(w↑) ≥ 0 ≥ u(w↓), the second inequality fails, since its left side is ≥ 0 and its right side ≤ 0.
  - If u(w↑) ≤ 0 ≤ u(w↓), the first fails.

  So both are nonzero with one sign. Say it is +, so that F_A < −ω_A(w↓)u(w↓) < 0 and F_B < −ω_B(w↑)u(w↑) < 0, and the
  order conditions are the ones stated. Conversely, given such ω, u, and F, the four strict inequalities hold. With
  aspect-dependent votes, u_A and u_B are separate unknowns and A's pair can be (+, −).
- (c) The sum is additive over items, and only item i changes, from e^{π(c,d)}β(t) to e^{π(c,d)}β(t′). Let x be "T
  was ADJ₊" and x′ the same with ADJ₋, so D(x) > 0 > D(x′) and β(ADJ₋) < β(ADJ₊). Let y be "T was not ADJ₊" and y′
  the same with ADJ₋, so D(y) < 0 < D(y′) and β(ADJ₋) > β(ADJ₊). That is a contradiction. The adjective's offset differs
  between x and y (2 against 3), and the position factor e^{π} > 0 does not change the sign.
- (d) Give the bigram types "not ADJ₊" and "not ADJ₋" votes of −2 and +2, every other item vote 0 except ADJ₊ +1 and
  ADJ₋ −1, and every score 0. With n items, D/(Σe^{s}) is then (+1)/n, (−1)/n, (−1)/n, and (+1)/n. ∎

Harness:
- `test_thm1a_vote_identity`: random layers, both value kinds, three class pairs, to 10⁻¹².
- `test_layer_matches_topic9_attention`: `ak.attention` with a one-row query, to 10⁻¹³.
- `test_gradients_complex_step`: three rungs, to 10⁻¹⁰.
- `test_thm1b_direction_words_cannot_vote_their_meaning`:
  - an LP over all nine sign patterns: shared → {(+,+), (−,−)};
  - every fully-right valence quadruple of the shared rung uses a same-sign pair;
  - the aspect rung is right on > 99% of quadruples.
- `test_thm1c_negation_is_not_representable`:
  - LP infeasible at 21 random position-factor pairs, feasible with bigrams;
  - both unigram rungs score 0 quadruples;
  - the bigram rung scores > 0.9.
- `test_the_ladder`: the accuracies by construction.

*Remark.* Both aspect-blind readers read every negation quadruple:
- the Topic 18 lexicon, because its k = 1 negation scope ("a negator within one token flips the sign") is a
  hand-written bigram feature;
- the blind LR, because "not strong" and the like are among its bigrams.

Neither reads a valence quadruple. The lexicon has no direction words, and the blind LR has no aspect.

**Thm 2 (attention weights are not identified by decisions).** Take a layer as in Def 2 and positive weights r_t > 0
on item types. Define a second layer by κ′(c, t) = κ(c, t) + log r_t, with values z′(c, t) = (z(c, t) + b_c)/r_t − b_c
and everything else unchanged. Then for every input, ℓ′(x) = λ(x)ℓ(x), where λ(x) = Σᵢ e^{sᵢ}/Σᵢ e^{sᵢ}r_{tᵢ} > 0. So:
- (a) every decision (argmax, and every pairwise sign) is unchanged;
- (b) each item's vote αᵢ(zᵢ + b_c) is multiplied by the same λ(x), so the vote profile is invariant;
- (c) for any set T of types present in every sentence, r_t = ρ on T moves the attention mass on T to 1 as ρ → ∞;
- (d) the probabilities are softmax(λ(x)ℓ(x)), a temperature 1/λ(x) that depends on the input, so they are not
  preserved.

If the values are W v with W of full row rank, the same map is realized in v.

*Proof.*
ℓ′ = Σᵢ α′ᵢ(z′ᵢ + b_c) = Σᵢ e^{sᵢ}r_{tᵢ}(zᵢ + b_c)/r_{tᵢ} / Σⱼ e^{sⱼ}r_{tⱼ} = λ(x) Σᵢ αᵢ(zᵢ + b_c) = λ(x)ℓ(x).
- (a) holds because λ > 0. (b) is the same computation term by term.
- (c) The mass on T is ρΣ_T e^{s}/(ρΣ_T e^{s} + Σ_{∉T} e^{s}) → 1.
- (d) is immediate.
- For W v: solve W v′ = (W v + b)/r − b, which is possible when W is onto. ∎

Harness: `test_thm2_reweighting_preserves_decisions` checks random layers and random log r ~ N(0, 4), with
ℓ′ = λℓ to 10⁻⁹ relative, equal argmaxes, votes scaled by λ, and an attention change of > 0.5.
`test_thm2_on_the_trained_model` checks the bigram rung at r = 10⁶ on the aspect-term words: the same argmax on all
14,179 test aspects, a ratio spread below 10⁻⁶, and mean mass 0.005 → 0.998. The spread of 2.8·10⁻⁸ is cancellation
in z′ = (z + b)/r − b at a 10⁻⁶ logit scale.

**Prop 2 (from aspects to tone).** Let a document of tone t have k tone-bearing sentences. Their polarities g₁, …, g_k
are i.i.d. from the tone's mixture, and nothing else in the text depends on t. Then:
- (a) the Bayes rule given the text is the Bayes rule δ* given the counts n = (n₊, n₀, n₋), and its accuracy is Topic
  17's full-text ceiling A*;
- (b) if a reader ĝ replaces gᵢ, then |P(δ*(n̂) = t) − A*| ≤ P(n̂ ≠ n) ≤ E[k]·ε, where ε = E[#{i : ĝᵢ ≠ gᵢ}]/E[k]
  is the reader's sentence error rate.

*Proof.* (a) The likelihood factors as P(g | t)·P(rest), with P(g | t) depending on g only through n, a multinomial;
Topic 17's `bayes_ceiling` evaluates that rule exactly. (b) The two decisions differ only when n̂ ≠ n, which needs at
least one wrong sentence; the union bound gives P ≤ E[#wrong] = E[k]ε. ∎

δ* is Topic 18's category-world rule (Prop 1 there) on the sentence-polarity simplex: linear cuts in n for each k.

Harness:
- `test_tone_sentences_are_the_generators`: k ∈ 5…8 in every training and test document, and χ² on the mixture per
  tone.
- `test_prop2_aggregation_reaches_the_ceiling`:
  - the gold aggregation within 2 SE of A*;
  - the reader within wrong_docs/n ≤ E[k]ε of it;
  - more than 0.1 above every document reader.

## 3. Corpus and finance measurements

- **Rows:** every gold aspect, with its sentence tokens (`fc._tokens`) and offsets from the span. The clause
  construction is Topic 18's `construction` on the aspect's own clause (split at ", but ").
- **Readers:**
  - lexicon: Topic 18's stand-in lexicon with k = 1, on the sentence;
  - blind LR: `ltc.lr_fit` on sentence unigrams + bigrams (min_df 2, λ ∈ {10⁻², 10⁻³, 10⁻⁴} on dev);
  - three attention rungs:
    - one schedule each: 300 full-batch Adam steps, lr 0.05, L2 10⁻⁵, R = 12;
    - vocabulary of item types seen at least twice in training, `<unk>` otherwise;
    - nothing tuned.
- **Contrast sets (Gardner et al. 2020):**
  - valence: 2,205 = 245 direction-only test contexts × 9 (up, down) pairs, with term ∈ {revenue, costs};
  - negation: 18,120 = 302 negated contexts × 60 (ADJ₊, ADJ₋) pairs (10 positive × 6 negative adjectives), with "not"
    kept or deleted.
- **Tone-bearing sentences:** those that open on an aspect (or on "Management pointed to") and name no amount,
  product, or company. That is the generator's k exactly, in every document. A real filing has no such marker (§5).
- **EDGAR:** every aspect-term mention, read by the bigram rung in its sentence, with no gold.

## 4. The lab (`AspectSentimentLab`, `data/aspect-sentiment-lab.json`)

Four tabs, with every printed number from the payload:

1. **Read a sentence.** Seven curated aspects, chosen by rule (`curated()`):
   - the contrast pair (liquidity −, net sales +);
   - a direction-only clause the shared rung misreads (lease liabilities climbed);
   - a cost-like verb (taxes … missed expectations);
   - a negated adjective the aspect rung misreads (sales were not improved);
   - a noun phrase;
   - a neutral clause.

   Per token: attention and ± vote, for the shared / aspect / bigram rungs, with the predictions of all five readers
   against gold. *Learn:* bars stretch but never flip; the contrast sentence's two aspects read the same words
   differently.
2. **What one layer can read.**
   - Accuracy by construction × reader (heatmap), with the Prop 1 bound marked.
   - The two contrast-set rates per reader.
   - The shared rung's direction-word votes.

   *Learn:* the theorems' fingerprints: same-sign direction words, and 0 on negation for every unigram rung.
3. **Attention isn't the explanation.** A slider log₁₀ r ∈ 0…6 on the aspect-term words, on the contrast sentence's
   second aspect. It shows the attention per token (moves), the vote shares (fixed), the decision (fixed), the logit
   scale, and p (flattens). *Learn:* Thm 2, including (d).
4. **From aspects to tone.**
   - Document accuracy bars: net tone 0.637, categories 0.724, LR 0.739, aspect reader 0.883, gold 0.883, with the
     ceiling 0.890 ± SE.
   - The EDGAR readings table.

## 5. rigorFlag (draft)

> On the synthetic corpus an aspect's polarity is a deterministic function of its clause and category, so every result
> here is about representation, not estimation. The near-perfect bigram rung reflects a templated corpus, not real
> filings, where the same reader calls four of five neutral, descriptive EDGAR mentions polar at about even odds.
> Aspect spans and categories are gold. Term extraction is sequence labeling and is left to the extraction topics.
> Thm 1 covers one attention layer with separable content × position scores and values that depend on a single item;
> a second layer, an MLP readout, or bigram items escape it. Thm 2 preserves decisions, not probabilities. The
> tone-bearing sentences are found by the generator's templates (they name no amount or company), a marker real text
> lacks. The rung hyperparameters were fixed in advance, not tuned, and the contrast sets edit test sentences in ways
> the generator never writes ("revenue" in a costs clause; "not" deleted).

## 6. Notebook cell map (25 cells)

| # | kind | content |
|---|---|---|
| 1 | md | title, five points, how to run |
| 2 | code | imports, splits, row counts, polarity counts |
| 3–4 | md/code | Def 1, the curated contrast sentence, construction counts |
| 5–6 | md/code | Prop 1, the bound, the lexicon and the blind LR, contrast accuracy |
| 7–8 | md/code | Def 2, Thm 1(a), the identity check, Topic 9 cross-check |
| 9–11 | md/code/md | Thm 1(b), the LP patterns, the shared rung's votes, the valence contrast sets |
| 12–15 | md/code/md/code | Thm 1(c)–(d), the LP, the negation contrast sets, the ladder table |
| 16–18 | md/code/md | Thm 2, the reweighting on the trained model, the confidence remark |
| 19–20 | md/code | Prop 2, the aggregation, the baselines |
| 21–22 | md/code | EDGAR readings |
| 23–24 | md/code | the whole harness |
| 25 | md | where this sits |

## 7. Cross-site and scope

- **Prereqs:** formalML `kernel-regression`: the Nadaraya–Watson estimator, whose kernel Def 2's position factor
  e^{π(d)} is (Topic 9's Prop 1 makes the content part one too).
- **Connections:**
  - formalML `naive-bayes`: Prop 2's δ* is multinomial NB on sentence-polarity counts with known parameters.
  - formalRAG `cross-encoders-reranking`: the aspect-blind reader is a dual encoder of (sentence, aspect) that never
    lets them interact, and the attention rungs are a minimal cross-encoder.
- **In-site:**
  - Topic 9, Prop 1 (attention is Nadaraya–Watson);
  - Topic 17, Prop 4 (the XOR);
  - Topic 18, Prop 1 (the category world) and Def 2 (negation scope);
  - Topic 16 (temperature, for Thm 2(d));
  - Topics 21–22 (span extraction, forward links as plain text until they ship).
- **Scope:** no retrieval, no agentic material, no term extraction, and no pretrained models.

## 8. References (verified 2026-10-25)

Conference papers:
- Minqing Hu and Bing Liu. 2004. "Mining and Summarizing Customer Reviews." *KDD '04*, 168–77.
  https://doi.org/10.1145/1014052.1014073
- Maria Pontiki, Dimitrios Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh
  Manandhar. 2014. "SemEval-2014 Task 4: Aspect Based Sentiment Analysis." *SemEval 2014*, 27–35.
  https://aclanthology.org/S14-2004/
- Yequan Wang, Minlie Huang, Xiaoyan Zhu, and Li Zhao. 2016. "Attention-based LSTM for Aspect-level Sentiment
  Classification." *EMNLP 2016*, 606–15. https://aclanthology.org/D16-1058/
- Duyu Tang, Bing Qin, and Ting Liu. 2016. "Aspect Level Sentiment Classification with Deep Memory Network." *EMNLP
  2016*, 214–24. https://aclanthology.org/D16-1021/
- Dehong Ma, Sujian Li, Xiaodong Zhang, and Houfeng Wang. 2017. "Interactive Attention Networks for Aspect-Level
  Sentiment Classification." *IJCAI 2017*, 4068–74. https://doi.org/10.24963/ijcai.2017/568
- Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao, and Min Yang. 2019. "A Challenge Dataset and Effective Models for
  Aspect-Based Sentiment Analysis." *EMNLP-IJCNLP 2019*, 6280–85. https://aclanthology.org/D19-1654/ (MAMS: every
  sentence "contains at least two different aspects with different sentiment polarities").
- Chi Sun, Luyao Huang, and Xipeng Qiu. 2019. "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing
  Auxiliary Sentence." *NAACL-HLT 2019*, 380–85. https://aclanthology.org/N19-1035/
- Sarthak Jain and Byron C. Wallace. 2019. "Attention is not Explanation." *NAACL-HLT 2019*, 3543–56.
  https://aclanthology.org/N19-1357/
- Sarah Wiegreffe and Yuval Pinter. 2019. "Attention is not not Explanation." *EMNLP-IJCNLP 2019*, 11–20.
  https://aclanthology.org/D19-1002/
- Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. 2020. "Attention is Not Only a Weight: Analyzing
  Transformers with Vector Norms." *EMNLP 2020*, 7057–75. https://aclanthology.org/2020.emnlp-main.574/ (their f(x) is
  the value transform followed by W^O: say "transformed vectors", not "value vectors").
- Matt Gardner et al. (26 authors). 2020. "Evaluating Models' Local Decision Boundaries via Contrast Sets." *Findings
  of EMNLP 2020*, 1307–23. https://aclanthology.org/2020.findings-emnlp.117/
- Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur.
  2018. "WWW'18 Open Challenge: Financial Opinion Mining and Question Answering." *Companion Proceedings of The Web
  Conference 2018*, 1941–42. https://doi.org/10.1145/3184558.3192301 (Task 1, aspect-based financial sentiment,
  confirmed on the challenge site because the ACM PDF returned 403).
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. "Neural Machine Translation by Jointly Learning to Align
  and Translate." *ICLR 2015*. https://arxiv.org/abs/1409.0473

## 9. Open questions for the checkpoint

1. **Difficulty.** `intermediate` is proposed: the proofs are sign arguments, one union bound, and LP certificates.
2. **Thm 1's shape.** Keep (a)–(d) as one theorem, or split (b) into its own proposition ("Valence needs
   aspect-dependent votes"), giving Thm 1 = identity + monotone substitution, Prop = valence, Thm 2 = identifiability?
3. **EDGAR in the lab.** Fold the readings table into tab 4 (as planned), or keep it as a static MDX table inside the
   case study?
4. **The lexicon's negation remark.** Topic 18's k = 1 rule reads every negation quadruple. Say so in §4 as "a scope
   rule is a hand-written bigram", or leave it to the lab?
