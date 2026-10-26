# Brief — `named-entity-recognition`

**Named Entity Recognition.** This is roadmap #22 (domain `extraction-knowledge-graphs`). Its prerequisites are
published: `sequence-labeling-hmm-crf` (#21: the chain recursions, the CRF and its likelihood, the finance BIO data)
and `contextual-embeddings-masked-lm` (#12: the filing encoder whose frozen states feed the CRF). It feeds
`relation-extraction-distant-supervision` (#23), which consumes entity spans.

**Status: Phase A done and approved by Jonathan (2026-10-26) with the recommended answers to §5: the ≈ 81 s
runtime is accepted (as for Topic 20), the page leads with the dropout lesson, `intermediate`, all five lab tabs.**
- **Phase B** shipped the MDX, `NamedEntityLab` (5 tabs), and the graph and curriculum flip.
  - The reference check (2026-10-26) changed three claims: Lipton et al.'s F*/2 result is their Corollary 1; SemEval-2013
    defines no partial credit (it defers to MUC); Dinkelbach is cited from his abstract (INFORMS returned 403), as
    restating Jagannathan's theorem. Lample et al. found IOBES no better than IOB, so the page cites only Ratinov and
    Roth for BIOES helping.
  - Building the F1 tab added one assert: the realized F1 peaks above τ = 1/2 for the substitution tagger and below
    F*/2 for the headline (0.38 vs 0.458), the calibration story the page tells.
  - Gates: `pnpm test` 39/39, `astro sync`, `validate`, `audit:cross-site` (formalNLP 173 edges, 0 mismatches, all
    missing reciprocals deferred), `build`. Harness 24/24 under `-W error` in 82 s; notebook exits 0.
  - Page: 0 `.katex-error`, 341 KaTeX spans, no `‘`. Every Held-out-names readout (104 reading × match states) equals
    the payload, as do the sliders (driven by real key presses). `scrollWidth` is 375 at the mobile preset in all 82 tab
    and toggle states, after moving Theorem 1's proof formula to display math (inline, it widened the page to 483 px).
    No console errors.
- `notebooks/named-entity-recognition/named_entity_recognition.py`: 24 tests, ≈ 81 s under `-W error` (Topic 12's
  encoder pretraining is 21 s of it; see §5, question 1).
- `01_named_entity_recognition.ipynb` (26 cells) exits 0 under `jupyter execute`.
- Lab data `src/components/viz/data/named-entity-lab.json` (17.8 kB) is written by `--write-payload` and pinned by
  `test_viz_mirror_matches_payload`.

**Phase A decisions** (plan mode, 2026-10-26, with Jonathan):
- **Scope against #21**, agreed while building #21: #22 owns contextual encoders + CRF, span evaluation in depth,
  BIO vs BIOES with constrained decoding, and names held out by design (`fc.split_roster()`).
- **Frozen encoder only** (no end-to-end fine-tuning). Thm 2's chain rule into the encoder is stated and checked by
  central differences, not used for training. The page says fine-tuning might close the CRF-vs-softmax gap and that
  #22 does not test it.
- **Never-seen names through the [UNK] reading** of Topic 12's encoder, which saw the held-out names in unlabeled
  pretraining (35 of the 49 held-out name words are in its vocabulary).
- **Lowercase throughout + one case ablation** (a `cap` feature from the raw text: capitalized, not sentence-initial).
- **Claude writes all code.**
- **No sibling covers NER, BIO/BIOES, span-level F1, or F1-optimal decoding** (grepped formalML, formalStatistics,
  formalRAG `main` for named entity, BIOES, NER, F-measure, optimal threshold, 2026-10-26).

**What building the harness found** (each now an assert):
1. **The word CRF memorizes names.** Trained on 30 documents, Topic 21's feature CRF tags seen-roster ORGs at F1
   0.929150 and predicts *no* ORG span on the held-out test (ORG F1 0): every unseen ORG is tagged O, even in
   "northgate energy reported revenue of". Word identity explains every training label, so context earns no weight.
2. **Dropout must delete, not substitute.** The plan's recipe (rewrite dropped entity tokens as `<unk>`) teaches a
   shortcut: `<unk>` fires only on entity tokens in training, so its O weight goes negative (Prop 5), and every
   unseen word at test, name or not, is pushed toward an entity. Precision falls to 0.481180 (seen) and 0.502575
   (unseen). Deleting the dropped token's word, affix, and neighbor features, adding none, raises strict F1 on
   *both* tests: 0.944515 → 0.975207 (seen), 0.641594 → 0.889543 (unseen). This replaced the plan's single recipe.
3. **Train as you test.** The encoder has no "delete": a dropped token enters as [UNK]. The headline tagger (word
   deletion + encoder [UNK] dropout, both feature sets) scores 0.884015 on the unseen test when the encoder reads
   held-out names by their pretraining ids, and 0.979129 when every word unseen in training enters as [UNK], the
   input its dropout trained (seen test 0.951326). Without dropout the [UNK] reading *hurts* (encoder 0.823188 →
   0.783125), so pretraining exposure to the names is worth something, but less than matching train to test.
4. **Capitalization is most of the signal.** One `cap` feature, no dropout: unseen F1 0.919931 (ORG 0.873494).
5. **The CRF's transitions matter on frozen states.** A softmax head on the same states decodes invalid BIO in 9.9%
   (seen) and 14.9% (unseen) of sentences; Thm 1's constraint lifts it (0.763475 → 0.809466; 0.699371 → 0.710112),
   but the CRF head leads by 0.092 and 0.113.
6. **BIOES buys nothing here** (0.970356 vs 0.975207 seen; 0.887702 vs 0.889543 unseen): the features see the
   neighbors, Prop 2's equality case.
7. **Thm 3 is optimal for the model's beliefs.** F1-optimal decoding lifts the substitution tagger by 0.114 and 0.098
   and moves the headline by +0.001 (seen) and −0.006 (unseen). Neither model's F* is its realized F1: on the unseen
   test the headline expects 0.916043 and its own F1-optimal sets realize 0.973424 (underconfident under the shift).
8. **Above τ = 1/2 no interval program is needed**: two overlapping spans are exclusive events, so their marginals
   sum to at most 1 (a remark under Thm 3, and the lab's fast threshold sweep).
9. **The seen-roster test is not all seen.** At 30 training documents, 214 of its 3,747 spans contain a word training
   never shows (rarely covered companies, cities), so the page also reports F1 on unseen *mentions* (Augenstein et
   al.'s split), alongside the two test sets.

---

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Named Entity Recognition |
| `{{SUBTITLE}}` | A tagger that learns names by their spelling cannot find a name it never saw; what generalizes is context, provided training removes the name the way test-time novelty does. |
| `{{SLUG}}` | `named-entity-recognition` |
| `{{DOMAIN}}` | `extraction-knowledge-graphs` |
| `{{DIFFICULTY}}` | `intermediate` (§5, question 3) |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[sequence-labeling-hmm-crf, contextual-embeddings-masked-lm]` |
| `{{FORMALML_PREREQS}}` | `convex-analysis` (Cor 1's strict concavity); connection `uncertainty-quantification` (calibration behind Thm 3's F*) |
| `{{FORMALRAG_PREREQS}}` | `set-metrics-precision-recall-map-mrr` (precision and recall as set metrics, Def 4) |
| `{{STATS_CALC_PREREQS}}` | none new (Topic 21 carries formalStatistics `hidden-markov-models`) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/sequence-labeling-hmm-crf/`, `notebooks/fine-tuning-low-rank-adaptation/` |

All three sibling slugs are `status: "published"` on GitHub `main` (checked 2026-10-26).

## 1. Section outline (geometric first)

1. **Overview.** NER predicts a set of typed spans; tags are one encoding of that set. The held-out-names problem
   in one picture: the lab sentence ("northgate energy guided to percent growth in fiscal") under the word CRF and
   under the headline.
2. **Encodings.** Def 1, Def 2, Prop 1 (bijection, first-order validity, transfer-matrix count), Prop 2 (BIO vs
   BIOES). Lab tab *Encodings*.
3. **Constrained decoding.** Thm 1.
4. **A CRF on a contextual encoder.** Def 3, Thm 2, Cor 1; the softmax head as P = 0. Lab tab *Heads*.
5. **Span evaluation in depth.** Def 4 (four match schemes), Prop 3; per-type and seen/unseen-mention F1.
6. **F1-optimal span decoding.** Prop 4, Thm 3, the remark on τ ≥ 1/2. Lab tab *F1 decoding*.
7. **Finance case study: names held out by design** (`<FinanceCaseStudy>`). The split; memorization; Def 5 (name
   dropout, two recipes), Prop 5; the encoder readings; case; BIOES; heads. Lab tabs *Held-out names*, *Dropout*.
8. **Where this sits** (hands spans to #23) / honest caveats / implementation.

## 2. Statements (numbered per type, in page order; same numbers in the `.py` and notebook)

**Def 1 (span set).** A span (s, e, τ) covers tokens s, …, e − 1 with type τ ∈ E; a span set is a set of pairwise
disjoint spans.

**Def 2 (BIO, BIOES).** BIO tags a span's first token B-τ and the rest I-τ; BIOES tags a one-token span S-τ and a
longer one B-τ, I-τ, …, E-τ; every other token is O. K = 2|E| + 1 and 4|E| + 1 tags.

**Prop 1 (encodings are bijections).** Each scheme's encoder is a bijection from span sets on T tokens onto the
*valid* sequences, which are exactly those passing a start mask, a transition mask, and an end mask (BIO: I-τ only
after B-τ or I-τ; BIOES: after B-τ or I-τ only I-τ or E-τ, elsewhere only O, B, S; start on O, B, S; end on O, E,
S). So |valid| = startᵀ M^{T−1} end = N(T), where N(T) = N(T−1) + |E| Σ_{l=1}^{T} N(T−l). The conlleval repair
(an I or E that continues nothing opens a span) maps all K^T sequences onto the span sets, many-to-one. *Proof:*
encode then decode reads back each span (B opens, I continues, the next non-continuing tag closes); the masks forbid
exactly the patterns decode must repair; counting by the last token gives the recursion. Asserted exhaustively for
|E| ≤ 2, T ≤ 5.

**Prop 2 (BIO against BIOES).** (a) For every BIO chain (U, P), the BIOES chain U′_t(y) = U_t(c(y)),
P′(y, y′) = P(c(y), c(y′)), with c: S ↦ B, E ↦ I, has the same constrained law over span sets. (b) With token-local
features (U_t depends on x only through x_t) the inclusion is strict: on x = "a d" with gold {(0, 1)} and x = "c d"
with gold {(0, 2)}, every BIO chain gives the two gold sets total probability ≤ 1, while BIOES chains approach 1 on
each. (c) With the left word as a feature of each token, BIO approaches 1 too. *Proof:* (a) c is a bijection between
the valid sets commuting with encode, and score′(y) = score(c(y)). (b) Given z₁ = B, the chain's conditional law of
z₂ is ∝ exp(P(B, z₂) + U₂(z₂)) β₂(z₂), a function of x₂ alone, the same "d" in both sentences; the first gold set
needs z₂ = O, the second z₂ = I, so their probabilities are bounded by q(O) and q(I), with q(O) + q(I) ≤ 1. In BIOES,
S-τ and B-τ are different first tags, and after B the mask forces E. Ours; Ratinov and Roth (2009) report BILOU
(= BIOES) beating BIO empirically. Fitted values: BIO 0.499128 + 0.499128; BIOES 0.994428, 0.995068; BIO + left
0.993592, 0.994257.

**Thm 1 (constrained decoding is conditioning).** Let C be the valid set of Prop 1. Setting U₁(k) = −∞ for invalid
starts, U_T(k) = −∞ for invalid ends, and P(j, k) = −∞ for invalid transitions multiplies the chain's potential by
1[z ∈ C], which factorizes over its nodes and edges. Hence (Topic 21's Thm 1, unchanged) Viterbi returns
argmax_{z∈C} p(z | x), forward–backward returns p(z_t | x, Z ∈ C), and log Z_C − log Z = log p(Z ∈ C | x).

**Def 3 (CRF on an encoder).** Unaries U_t(k) = (W h_t + b)_k from encoder states h_t = LN(x_L)_t, transitions P,
start and end weights. The softmax head is the case P = 0, start = end = 0.

**Thm 2 (the gradient at the potentials).** ∂ log p(z | x)/∂U_t(k) = 1[z_t = k] − γ_t(k) and
∂ log p(z | x)/∂P(j, k) = Σ_t (1[z_t = j, z_{t+1} = k] − ξ_t(j, k)), with γ, ξ Thm 1's node and pair marginals. By
the chain rule the encoder receives dL/dh_t = (γ_t − e_{z_t}) Wᵀ for L = −log p(z | x); with P = 0, γ_t is the
per-token softmax and this is the softmax head's cross-entropy gradient. *Proof:* log p = score(z) − log Z;
∂ score/∂U_t(k) = 1[z_t = k] and ∂ log Z/∂U_t(k) = γ_t(k) by differentiating the forward recursion (the
exponential-family identity of Topic 21's Thm 2 with the potentials as parameters).

**Cor 1 (frozen states).** With the encoder fixed, the features h_t are constants and the objective is Topic 21's
Thm 2 on dense features: concave, and strictly concave with a unique maximizer under λ > 0 (formalML
`convex-analysis`).

**Def 4 (match schemes).** A predicted span is correct, and a gold span found, if some span on the other side
*matches* it: strict (same s, e, τ), boundary (same s, e), type (overlap, same τ), partial (overlap). Precision and
recall are the two shares; F1 their harmonic mean. Segura-Bedmar et al. (2013) for the four criteria (their partial
scheme gives half credit; ours counts any overlap; say so).

**Prop 3.** (a) Under strict matching, micro F1 = Σ_τ w_τ F1_τ with w_τ = (|G_τ| + |P̂_τ|)/Σ_σ (|G_σ| + |P̂_σ|):
the weights move with the predictions, and macro F1 is the uniform mixture. (b) If match relation R ⊆ R′, precision,
recall, and F1 under R′ are each at least those under R; so strict ≤ boundary ≤ partial and strict ≤ type ≤ partial.
*Proof:* (a) F1_τ = 2TP_τ/(|G_τ| + |P̂_τ|) and micro F1 = 2 Σ TP_τ / Σ (|G_τ| + |P̂_τ|). (b) Each span matched under
R is matched under R′; F1 increases in both P and R.

**Prop 4 (span marginals).** In BIO, p((s, e, τ) ∈ S | x) = α_s(B-τ) · Π_{t=s+1}^{e−1} e^{P(·, I-τ) + U_t(I-τ)} ·
Σ_{k ≠ I-τ} e^{P(I-τ or B-τ, k) + U_e(k)} β_e(k) / Z (the last factor is 1 at e = T); in BIOES the path is S-τ, or
B-τ, I-τ, …, E-τ, closed by β_{e−1}. One forward and one backward pass give every span's marginal; Σ_spans q = E|S|
= Σ_t p(z_t ∈ B ∪ S).

**Thm 3 (F1-optimal span decoding).** Let q be the span marginals on a test corpus and m = Σ q = E|S|. Among span
sets Ŝ (one per sentence), maximize Φ(Ŝ) = 2 Σ_{s∈Ŝ} q_s / (|Ŝ| + m), the F1 of expected counts (E TP = Σ_Ŝ q,
E|G| = m). (a) F* = max Φ iff max_Ŝ Σ_{s∈Ŝ} (2q_s − F*) = F* m (Dinkelbach 1967). (b) The iteration
F_{k+1} = Φ(Ŝ_k), Ŝ_k = argmax Σ (2q − F_k), increases strictly until it stops at F*, after finitely many steps. (c)
The inner problem is weighted interval scheduling with weights q_s − F/2, solved by a DP over token positions; the
optimal set thresholds the marginals at F*/2 subject to disjointness, and with disjoint candidates it is exactly
{q > F*/2} (Lipton, Elkan, and Naryanaswamy 2014). **Remark:** overlapping spans are exclusive events, so
q_a + q_b ≤ 1 and the candidates above any τ ≥ 1/2 are already disjoint. Φ is not E[F1]; the two agree
asymptotically on large corpora (Ye et al. 2012).

**Def 5 (name dropout).** Each training token inside a gold span is dropped independently with probability p.
*Substitution* rewrites it as `<unk>` (every feature then names `<unk>`; at test every word unseen in training is
`<unk>`); *deletion* removes its word, prefix, and suffix features and every neighbor's feature naming it, adding
none. Either way it enters the encoder as [UNK].

**Prop 5 (a feature seen only on entities argues against O).** At the optimum of the λ-penalized CRF objective,
Σ_k W[f, k] = 0 for every feature f; and if f fires in training only on tokens whose gold tag lies in A, then
W[f, k] < 0 for every k ∉ A, so Σ_{k∈A} W[f, k] > 0. *Proof:* stationarity gives
nλ W[f, k] = Σ_{t: f fires} (1[z_t = k] − γ_t(k)); summing over k gives 0; for k ∉ A the indicator vanishes and
γ_t(k) > 0 (finite weights). Substitution makes `<unk>` such a feature with A = the entity tags, so its O weight is
negative (asserted on the fitted tagger: the most negative of its 17 weights).

## 3. Experiment grid (≈ 81 s)

| Block | What | Cost |
|---|---|---|
| Exactness | brute force: Props 1–4, Thms 1–3, Cor 1, Prop 5's toy; Thm 2 by complex step and central differences | ~4 s |
| Encoder | Topic 12 `train_mlm(FILING_MLM)` via `ft.pretrained()`, float64 copies; Topic 20's `final_states` per sentence + [SEP] | ~21 s |
| Data | train: first 30 docs of `generate_corpus(300, 0, roster=train)` (598 sentences, 552 spans); seen test `(200, 1, train)` (4,115 / 3,747); unseen test `(200, 1, held)` (4,068 / 3,648) | ~1 s |
| Grid | 9 taggers (below), λ = 1/sentences (MAP under N(0, I)), L-BFGS; softmax head by L-BFGS on Topic 17's objective, λ = 1/tokens | ~25 s |
| Sweep | word CRF, both recipes, p ∈ {0, .25, .5, .75, 1} (dropout uniforms: stream 3500, nested in p) | ~17 s |
| Thm 3 | span marginals, Dinkelbach, 49 thresholds × 2 models × 2 tests | ~8 s |

Grid: words; words + `<unk>` dropout; words + deletion dropout (BIO and BIOES); words + case; encoder; both; both +
deletion dropout (the headline); softmax on encoder. Decoding: constrained Viterbi (Thm 1) unless stated.

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

**Strict F1 (P, R), constrained Viterbi.** Encoder taggers: "ids" = the encoder's vocabulary; "unk" = every word
unseen in training enters as [UNK].

| tagger | seen | unseen | seen-unk | unseen-unk |
|---|---|---|---|---|
| words | 0.944515 (0.997, 0.897) | 0.641594 (1.000, 0.472) | | |
| words, `<unk>` dropout | 0.610764 (0.481, 0.836) | 0.654027 (0.503, 0.936) | | |
| words, deletion dropout | 0.975207 | 0.889543 | | |
| … BIOES | 0.970356 | 0.887702 | | |
| words + case | 0.978343 | 0.919931 | | |
| encoder | 0.901508 | 0.823188 | 0.891840 | 0.783125 |
| both | 0.973032 | 0.856720 | 0.966871 | 0.790523 |
| both, deletion dropout | 0.948567 | 0.884015 | 0.951326 | **0.979129** |
| softmax on encoder | 0.809466 (raw 0.763475) | 0.710112 (raw 0.699371) | 0.812618 | 0.676073 |

ORG F1, unseen test: words 0.000000, words deletion 0.762590, words + case 0.873494, encoder 0.571201, headline
(unk) 0.951114. Unseen-mention F1 (seen test / unseen test): words 0.334630 / 0.202886; headline (unk) 0.891775 /
0.963713. Invalid unconstrained decodes: softmax 0.098663 / 0.148968; every CRF ≤ 0.006146.

**Sweep (strict F1 by p = 0, .25, .5, .75, 1).** Substitution seen 0.944515 0.718638 0.610764 0.445677 0.040547
(P 0.997 … 0.037); unseen 0.641594 0.736259 0.654027 0.584656 0.218085. Deletion seen 0.944515 0.963766 0.975207
0.969656 0.837280; unseen 0.641594 0.794872 0.889543 0.920311 0.822793 (at p = 1 even million, percent, and
quarter never show their identity).

**Thm 3.** Headline (unk): F* 0.909616 / 0.916043, Viterbi 0.951326 / 0.979129, F1-decoded 0.951961 / 0.973424
(seen / unseen). Substitution: F* 0.652544 / 0.648632, Viterbi 0.610764 / 0.654027, F1-decoded 0.724685 /
0.751581.

**Split.** Held out: Dunmore Aerospace, Meridale Medical, Northgate Energy, Saltmarsh Capital, Stratus Point
Networks, Tamsin Retail Group. 49 held-out name words, 35 in the encoder's vocabulary.

## 4. The lab (`NamedEntityLab`, `data/named-entity-lab.json`, 5 tabs)

The example sentence (rule: the first unseen-test sentence with ≥ 3 gold spans including a multi-token ORG that the
word CRF misses entirely and the headline finds exactly) is sentence 1: "northgate energy guided to percent growth
in fiscal" (ORG, PERCENT, DATE).

1. **Encodings.** The sentence in BIO and BIOES; the one-type masks as small matrices (start, transitions, end);
   valid vs all sequences by T; Prop 2's two-sentence counterexample with the three fitted probability pairs.
2. **Held-out names.** Tagger selector × reading (seen / unseen, ids / unk) × match scheme: F1, P, R, per-type bars,
   seen/unseen-mention F1, and the example sentence's predicted spans. Lesson: the word CRF finds no unseen ORG.
3. **Dropout.** Slider p over the sweep; two recipes; seen and unseen F1 and precision. Lesson: substitution's
   precision collapses while deletion improves both tests.
4. **Heads & constraints.** Softmax vs CRF on the encoder: raw vs constrained F1 and the invalid share; BIO vs BIOES.
5. **F1 decoding.** Threshold slider τ for the headline (unk) and the substitution tagger: realized F1, P, R and the
   model's expected F1 against τ, with F*/2 marked and Viterbi's F1 as a reference line; the example sentence's span
   marginals, with the spans above τ highlighted.

## 5. Checkpoint questions (for Jonathan)

1. **Runtime.** The harness takes ≈ 81 s (21 s of it Topic 12's pretraining, which no topic caches across runs;
   18 CRF fits ≈ 42 s). Accept, as for Topic 20, or trim to ≈ 65 s by cutting the substitution sweep to p ∈ {0, 0.5, 1}
   and the threshold grid to 25 points?
2. **Headline and framing.** "Train as you test": the headline is "both, deletion dropout" read with [UNK]
   (0.979129 unseen, 0.951326 seen), with the word-deletion CRF (0.975207 / 0.889543) and words + case (0.978343 /
   0.919931) beside it. OK to lead with the dropout lesson (finding 2) rather than with the encoder?
3. **Difficulty** `intermediate` (proofs are short; Thm 3's Dinkelbach step is the hardest)?
4. **Five lab tabs**, or fold *Heads & constraints* into *Held-out names*?

## 5b. rigorFlag (draft)

The corpus is synthetic and templated, and `fc._tokens` lowercases it, removing the capitalization cue real NER
leans on (one `cap` feature recovers most of the gap). Topic 12's encoder saw the held-out names in unlabeled
pretraining; reading unseen words as [UNK] only approximates names never seen anywhere. The encoder is frozen
(d = 64, two layers): fine-tuning it might close the CRF-vs-softmax gap, and this page does not test that. Name
dropout's rate is fixed a priori at 0.5 and every model is fit once at λ = 1/n, untuned. Thm 3 maximizes F1 of
expected counts, which is expected F1 only asymptotically, and is optimal for the model's own marginals, which the
name shift miscalibrates. Prop 2's strictness holds for token-local features only; the taggers here see their
neighbors, so their BIO/BIOES difference is estimation, not expressiveness.

## 6. Notebook cell map (26 cells)

1 intro · 2 imports · 3–4 Prop 1 (example sentence, counts) · 5–6 Prop 2 · 7–8 Thm 1 · 9–10 Thm 2, Cor 1 · 11–12
Prop 3 · 13–14 Prop 4, Thm 3 · 15–16 the split · 17–18 the grid · 19–20 Prop 5 and the sweep · 21–22 heads, BIOES ·
23–24 Thm 3 on the finance taggers · 25–26 harness.

## 7. References (verified 2026-10-26 against the papers' full text, except Dinkelbach)

- Augenstein, Isabelle, Leon Derczynski, and Kalina Bontcheva. 2017. "Generalisation in Named Entity Recognition: A
  Quantitative Analysis." *Computer Speech & Language* 44: 61–83. https://doi.org/10.1016/j.csl.2017.01.012. The
  seen/unseen split is in the body (arXiv 1701.02877, §3.3, Table 13: unseen-entity F1 far lower, mostly via recall);
  the abstract says only that leading systems rely on surface forms seen in training.
- Collobert, Ronan, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. 2011. "Natural
  Language Processing (Almost) from Scratch." *JMLR* 12: 2493–537. §3.4.2, sentence-level log-likelihood (Eqs. 12–14).
- Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. "BERT." *NAACL-HLT 2019*, 4171–86. N19-1423.
  §5.3: CoNLL-2003 NER with the first sub-token's representation into a token classifier, no CRF.
- Dinkelbach, Werner. 1967. "On Nonlinear Fractional Programming." *Management Science* 13 (7): 492–98.
  https://doi.org/10.1287/mnsc.13.7.492. INFORMS returned 403: cited from the Crossref abstract, which says the paper
  restates and proves Jagannathan's (1966) theorem relating fractional and parametric programs. The page says so.
- Lample, Guillaume, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. "Neural
  Architectures for Named Entity Recognition." *NAACL-HLT 2016*, 260–70. N16-1030. §4.1: singletons → UNK with
  probability 0.5; §2.4: IOBES gave no significant improvement over IOB (do not cite them for BIOES helping).
- Lipton, Zachary C., Charles Elkan, and Balakrishnan Naryanaswamy. 2014. "Optimal Thresholding of Classifiers to
  Maximize F1 Measure." *ECML PKDD 2014, Part II*, LNCS 8725, 225–39. https://doi.org/10.1007/978-3-662-44851-9_15.
  F*/2 is **Corollary 1** of Thm 1 (calibrated scores; F1 as a ratio of population rates). arXiv 1402.1892v2 has a
  different title ("Thresholding Classifiers to Maximize F1 Score") and spells the third author "Narayanaswamy".
- Ratinov, Lev, and Dan Roth. 2009. "Design Challenges and Misconceptions in Named Entity Recognition." *CoNLL 2009*,
  147–55. W09-1119. §4, Table 2: BILOU beats BIO on three of four sets (BIO wins CoNLL03 dev).
- Segura-Bedmar, Isabel, Paloma Martínez, and María Herrero-Zazo. 2013. "SemEval-2013 Task 9: Extraction of Drug-Drug
  Interactions from Biomedical Texts (DDIExtraction 2013)." *SemEval 2013*, 341–50. S13-2056. §3.1: strict, exact
  boundary, partial boundary, type. It does **not** define partial credit (defers to MUC, Chinchor and Sundheim 1993).
- Tjong Kim Sang, Erik F., and Fien De Meulder. 2003. "Introduction to the CoNLL-2003 Shared Task." *CoNLL 2003*,
  142–47. W03-0419. §2.4: exact-match span F1.
- Ye, Nan, Kian Ming A. Chai, Wee Sun Lee, and Hai Leong Chieu. 2012. "Optimizing F-Measures: A Tale of Two
  Approaches." *ICML 2012*. arXiv 1206.4625. Theorem 8 (§2.3): EUM and DTA asymptotically equivalent; pages unverified.
