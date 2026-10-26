# Brief — `relation-extraction-distant-supervision`

**Relation Extraction and Distant Supervision.** This is roadmap #23 (domain `extraction-knowledge-graphs`). Its one
prerequisite is published: `named-entity-recognition` (#22: span sets, the headline tagger, names held out by
design). It feeds `knowledge-graph-embeddings` (#24), which embeds the KB this topic extracts (`extracted_kb()`).

**Status: Phase A done and approved by Jonathan (2026-10-26) with the recommended answers to §5: lead with Thm 1/Cor 1;
keep the encoder rows as one paragraph and one lab row; feature Thm 3(b) in the page and the *Incomplete KB* tab; state
the gold ceiling plainly.**
- **Phase B** shipped the MDX, `RelationExtractionLab` (5 tabs), and the graph and curriculum flip.
  - Building the page added two harness-owned numbers: the unconjoined gold classifier (0.951697 seen, 0.978328
    unseen) and the pipeline's argument miss rates by type, each now asserted.
  - Gates: `pnpm test` 39/39, `astro sync`, `validate`, `audit:cross-site` (formalNLP 178 edges, 0 wrong direction,
    0 slug drift, all missing reciprocals deferred), `build`. Harness 23/23 under `-W error`; notebook exits 0.
  - Page: 0 `.katex-error`, 280 KaTeX spans, no `‘`. Every lab readout across the five tabs, both sliders (driven by
    real key presses), and all toggles equals the payload. At the mobile preset `innerWidth` and `scrollWidth` are 375
    in all 21 tab, toggle, and template states and at both slider ends, after rewording one unbreakable inline code
    filename in Implementation (it widened the page to 395 px). No console errors.
- `notebooks/relation-extraction-distant-supervision/relation_extraction_distant_supervision.py`: 23 tests, 33 s under
  `-W error` (Topic 12's encoder pretraining, inside Topic 22's tagger, is 21 s of it). No waiver needed.
- `01_relation_extraction_distant_supervision.ipynb` (23 cells) exits 0 under `jupyter execute`.
- Lab data `src/components/viz/data/relation-extraction-lab.json` (14.1 kB) is written by `--write-payload` and pinned
  by `test_viz_mirror_matches_payload`.

**Phase A decisions** (plan mode, 2026-10-26, with Jonathan):
- **Four parts:** what DS learns; at-least-one MIL; KB incompleteness as PU learning; the held-out-names pipeline
  and the extracted KB for #24.
- **Topic 22's headline tagger** feeds the pipeline.
- **Follow the corpus gold on entailed facts.** BASED_IN read off "headquartered in Toronto, Canada" counts as a DS
  error at the sentence level (a separate source, Prop 1), true at the fact level; the rigorFlag says the count
  depends on the guideline.
- **`intermediate`, 5 lab tabs.** Claude writes all code.
- **No sibling covers** distant supervision, multi-instance learning, positive-unlabeled learning, or relation
  extraction (grepped formalML, formalStatistics, formalRAG `main`, 2026-10-26).

**What building the harness found** (each now an assert):
1. **With gold sentence labels the task is solved.** Entity-blind lexical features, with the between string conjoined
   with the argument order, reach sentence F1 1.000000 on both tests. The conjunction matters: which argument is the
   subsidiary depends on the template ("X, a subsidiary of Y" vs "Y owns X") *and* the order, an interaction a sum of
   separate weights cannot represent (Topic 17's XOR). Unconjoined, the gold model scored 0.952 and its noisy-OR facts
   picked up reversed SUBSIDIARY_OF/PARENT_OF. So everything the page shows is about the *labels*.
2. **DS learns the KB's opinion of the pair** (Thm 1, Cor 1). On each of the 11 frequent co-mention templates the
   fitted DS classifier's P(not NA) is within 0.05 of the template's KB-related rate ρ (0.529–0.685). It labels a
   relation exactly where ρπ > 1 − ρ: "X commented on conditions across banking" (ρ 0.685, π 1: IN_SECTOR) and
   "P has spoken publicly about X" (ρ 0.646, π 0.595: CFO_OF), but not "Analysts who cover X also follow Y" (ρ 0.600,
   π 0.625: 0.375 < 0.400). DS sentence F1 0.866589 (seen) and 0.887363 (unseen).
3. **At least one buys precision with sentence recall** (Thm 2, Cor 2, Prop 2). At-least-one fails for 4 of 177
   positive (bag, relation) pairs (0.022599), against DS's 0.309392 label error. MIL labels no co-mention with a
   relation (P 0.976703), but recall falls to 0.711488 (seen). Its recall is 0.892105 on stating templates the bag
   constraints force (Prop 2) and 0.533679 on those they leave free. At the fact level MIL leads (noisy-OR F1
   0.934524 vs DS max 0.927614 seen; 0.978261 vs 0.824561 unseen).
4. **Aggregate as you trained** (Prop 3). Noisy-OR overcounts for the sentence-trained DS classifier: fact precision
   0.718876 against max's 0.891753 (seen) and 0.330986 vs 0.701493 (unseen, where few companies repeat many
   co-mentions). MIL, trained under noisy-OR, keeps 1.000000.
5. **An incomplete KB first helps, then collapses DS** (Thm 3, Prop 4). The relation mass scales by c (0.867 at
   c = 0.9 down to 0.374 at 0.4, slightly below c: L2 shrinks the rarer signal). Argmax keeps a stating template only
   while c > 1/2, and co-mention templates fall below threshold first, so raw sentence F1 *rises* to 0.931977 at
   c = 0.8 before recall collapses (0.664491 at 0.6, 0.249347 at 0.4). Dividing by the true c restores recall
   (≥ 0.958 down to c = 0.6). Elkan and Noto's e1 reads 0.909820 at c = 1, low, because DS's own target is below 1 on
   co-mention positives. Against an independent KB of rate c, fact precision reads ≈ c × the world precision.
6. **Names leak through the arguments.** With the arguments' own words as features, DS labels 0.812766 of seen related
   co-mentions with their KB relation (it memorizes the KB pair by pair), 0.304054 once the names are new. Topic 12
   states at the entity starts (Baldini Soares et al.'s representation) do the same: 0.795745 vs 0.263514.
7. **The pipeline loses only what NER loses** (Prop 5). On gold spans the entity-blind classifiers do as well on the
   held-out companies as on the seen ones. With Topic 22's spans, triple recall is capped by argument recall (0.729765
   seen, 0.919505 unseen) and the gold-trained classifier meets the cap. The seen test's cap is lower because its ORG
   and PERSON arguments include rarely covered training-roster companies the 30-document tagger never saw (Topic 22's
   finding 9). The extracted KB (MIL, unseen, predicted spans): 45 facts above 0.5, all true.

---

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Relation Extraction and Distant Supervision |
| `{{SUBTITLE}}` | Labeling sentences by what a knowledge base already knows teaches a classifier the KB's opinion of each entity pair; bags, aggregation, and positive-unlabeled corrections recover facts, not sentences. |
| `{{SLUG}}` | `relation-extraction-distant-supervision` |
| `{{DOMAIN}}` | `extraction-knowledge-graphs` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[named-entity-recognition]` |
| `{{FORMALML_PREREQS}}` | `kl-divergence` (Prop 2, Gibbs' inequality: Thm 1's proof) |
| `{{FORMALRAG_PREREQS}}` | `set-metrics-precision-recall-map-mrr` (Def 1, precision and recall of a set: fact-level P/R); connections `negative-sampling-hard-negatives` (Prop 2 and Thm 1: false negatives and an unknown class prior in contrastive training, the retrieval cousin of Def 5; the page never says "positive-unlabeled") and `graphrag-community-detection` (its entity graph is built from co-occurrence and cosine weights; this topic types the edges) |
| `{{STATS_CALC_PREREQS}}` | formalStatistics `hidden-markov-models` (Rem 16, "Gradients without the M-step": Fisher's identity stated without proof, which Thm 2 proves for the noisy-OR; Thm 9, EM ascent) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/named-entity-recognition/`, `notebooks/linear-text-classifiers/` |

All four sibling slugs are `status: "published"` on GitHub `main` (checked 2026-10-26). Not cited: formalML
`convex-analysis` has no strict-convexity or unique-minimizer theorem (Thm 5 is subgradient optimality), and
formalStatistics `maximum-likelihood` mentions EM only in Rem 10 (a preview) and states no Fisher's identity.
`model-selection-and-information-criteria`'s "Fisher identity" is the information equality J = K, a different result.

## 1. Section outline (geometric first)

1. **Overview.** A sentence, two mentions, a relation or NA; facts aggregate mentions into a KB. The lab candidate:
   "martin hollis has spoken publicly about pellucid software", a co-mention of a pair the KB relates (CFO_OF), which
   DS labels CFO_OF (0.380 against NA 0.351), the gold-trained classifier NA (0.995), MIL NA.
2. **Relation mentions and distant supervision.** Def 1, Def 2, Prop 1. Lab tab *DS labels*.
3. **What a DS-trained classifier learns.** Thm 1, Cor 1. Lab tab *What DS learns*.
4. **At least one witness.** Def 3, Thm 2, Cor 2, Prop 2; then facts: Def 4, Prop 3. Lab tab *At least one*.
5. **An incomplete KB is positive-unlabeled.** Def 5, Thm 3, Prop 4. Lab tab *Incomplete KB*.
6. **Finance case study: names held out** (`<FinanceCaseStudy>`). Def 6, Prop 5, the argument-word ablation, the
   extracted KB. Lab tab *Pipeline & KB*.
7. **Where this sits** (hands the KB to #24) / honest caveats / implementation.

## 2. Statements (numbered per type, in page order; same numbers in the `.py` and notebook)

**Def 1 (relation mentions and facts).** A sentence's mentions are its entity spans, each linked to a KB entity when
it names one. A *candidate* is an ordered pair (i, j) of distinct linked mentions in one sentence; its gold label is
the relation r ∈ R the sentence states from i to j, or NA. A *fact* is a triple (h, r, t) of KB entities. Sentence-
level extraction labels candidates; fact-level extraction (KB population) outputs facts.

**Def 2 (distant supervision; Mintz et al. 2009).** Label each candidate with L = KB(h, t), the set of relations the
KB holds from h to t, or {NA} when it holds none; as a target distribution, ℓ(r) = 1[r ∈ L]/|L|.

**Prop 1 (DS errors have three sources).** Every gold relation is in L (gold is always a KB fact), so DS misses none.
A DS label r ∈ L that the sentence does not state falls in exactly one class: *entailed* (r(h, t) follows from the
sentence's stated facts by the KB's rules: BASED_IN = HQ_IN ∘ LOCATED_IN, ACQUIRED ⇒ PARENT_OF, the PARENT_OF /
SUBSIDIARY_OF inverse, COMPETES_WITH's symmetry), *overlap* (not entailed, and the sentence states another relation
for the same ordered pair), or *co-mention* (the sentence states nothing that yields r). *Proof:* the classes are
tested in that order, so each label lands in one. Train: 1,629 labels, 1,125 correct, 338 co-mention, 137 entailed,
29 overlap; precision 0.690608 (= the corpus's `distant_supervision_precision`). LOCATED_IN is 1.000 precise (stated
only in HQ appositives, never co-mentioned); ACQUIRED is the least (0.493).

**Thm 1 (what DS learns).** Let φ be the classifier's features and R(q) = E[−Σ_r ℓ(r) log q(r | φ)] the DS risk.
Over all conditionals, R is minimized uniquely (φ-almost surely) by q*(r | φ) = E[ℓ(r) | φ]. Because gold ⊆ L,
q*(r | φ) = E[ℓ(r) 1[r stated] | φ] + E[ℓ(r) 1[r unstated] | φ]: the chance the sentence states r (weighted 1/|L|)
plus the chance the KB holds r unstated. *Proof:* conditioning on φ, R = E[Σ_r m_φ(r)(−log q(r | φ))] with
m_φ = E[ℓ | φ] a distribution; Σ_r m(r) log(m(r)/q(r)) ≥ 0 with equality iff q = m (Gibbs). Asserted on random finite
worlds: the risk's complex-step gradient vanishes at log q*, and 400 random perturbations all raise it.

**Cor 1 (a co-mention's label is its base rate).** If a context φ never states a relation, q*(NA | φ) = P(L = {NA} |
φ) = 1 − ρ(φ), ρ the KB-related share of its pairs, and argmax labels its top relation r̂ iff ρπ > 1 − ρ, where
π = q*(r̂ | φ)/ρ. Neither depends on the sample size. Corpus: 11 templates, fitted vs ρ within 0.05 (table in §3a).

**Def 3 (bags and the at-least-one assumption; Dietterich et al. 1997; Riedel, Yao, and McCallum 2010).** (Dietterich
et al. say "examples" made of "instances", not "bags"; the page uses the later word and says so.) A bag is
every training mention of one ordered entity pair. At least one: if r ∈ KB(h, t), some mention in the bag states r;
if not, none does. With instance probabilities p_i(r) from the softmax treated as independent, the bag likelihood is
the noisy-OR P(∃) = 1 − Π_i (1 − p_i(r)) for r ∈ KB(h, t) and Π_i (1 − p_i(r)) otherwise. MIL maximizes the sum of
their logs over bags and relations (λ = 1/bags).

**Thm 2 (the bag gradient is an E-step).** Let y_i ~ Bernoulli(p_i(θ)) independently, E = {∃ i: y_i = 1}. Then
∇ log P(E) = Σ_i [w_i ∇ log p_i + (1 − w_i) ∇ log(1 − p_i)], w_i = P(y_i = 1 | E) = p_i/P(E). *Proof:* Fisher's
identity: ∇ log P(E) = Σ_{y∈E} P(y | E) ∇ log P(y), and log P(y) = Σ_i y_i log p_i + (1 − y_i) log(1 − p_i); take the
conditional expectation term by term (y_i = 1 implies E, so P(y_i = 1 | E) = p_i/P(E)). Complex-step asserted on
random softmax bags. Fisher's identity is cited to Dempster, Laird, and Rubin (1977, §3, Lemma 2 with Eq. 3.2:
DL(φ) = D¹⁰Q(φ | φ)); they do not use the name. Zeng et al. (2015) train on each bag's single most probable instance
instead (the max, whose gradient is all on one witness: Cor 2's limit).

**Cor 2 (one witness silences the rest).** ∂ log P(E)/∂p_i = Π_{j≠i}(1 − p_j)/P(E). As any other p_k → 1 this tends to
0, while the confident instance's own coefficient tends to Π_{j≠k}(1 − p_j). *Proof:* substitute w_i into Thm 2:
w_i/p_i − (1 − w_i)/(1 − p_i) = (1 − P(E))/(P(E)(1 − p_i)). Lab bag (Hollis → Pellucid, CFO_OF): the four stating
mentions carry MIL 0.714–0.845, each co-mention 0.001 with coefficient 0.00252.

**Prop 2 (what the bag labels determine).** Classes c ∈ C (here: templates), a hypothesis h: C → {0, 1}, bags
labeled by max_{c∈B} h*(c) (at least one holds). Let N be the classes of negative bags. (a) h_max = 1[c ∉ N] is
consistent. (b) The consistent set is {h_max} iff every c ∉ N is the only non-N class of some positive bag. (c) If so,
h_max = h*. *Proof:* (a) negative bags hold only N classes; a positive bag holds a class with h* = 1, which is not in N.
(b) Every consistent h is 0 on N and covers each positive bag's non-N classes; h_max − 1[c] stays consistent iff no
positive bag has c as its only non-N class; and any consistent h ≠ h_max is h_max minus some such c's. (c) h* is
consistent. Brute-forced on 300 random worlds. Corpus: MIL recall 0.892105 on forced (template, relation) pairs vs
0.533679 on free ones (seen; 0.939525 vs 0.486339 unseen).

**Def 4 (from mentions to facts).** Score (h, r, t) by the max of P(r | mention) over its mentions (Mintz et al.'s and
Hoffmann et al.'s deterministic OR) or by the noisy-OR 1 − Π(1 − P(r | mention)); predict facts above 1/2.

**Prop 3 (noisy-OR ≥ max, and it saturates).** 1 − Π_i (1 − p_i) ≥ max_i p_i, and ≥ 1 − (1 − ε)^m when m mentions have
p_i ≥ ε. *Proof:* Π(1 − p_i) ≤ 1 − p_max, and ≤ (1 − ε)^m. So noisy-OR fits a model whose instance probabilities
were trained under it (MIL), and overcounts correlated mentions for sentence-trained models.

**Def 5 (an incomplete KB).** Each unordered entity pair the world relates is known, with all its facts,
independently with probability c and independently of the text (selected completely at random, at the pair level);
an unknown pair's mentions are labeled NA. One draw per c, nested in c (stream 3600).

**Thm 3 (Elkan and Noto 2008's Lemma 1, at the pair level).** (a) q*_c(r | φ) = c·q*(r | φ) for every r ∈ R, and
q*_c(NA | φ) = 1 − c(1 − q*(NA | φ)). (b) A template with q*(r | φ) = 1 keeps r under argmax iff c > 1/2. (c) e1, the
mean of g = 1 − q_c(NA) over labeled positives, equals c iff q* = 1 on them. *Proof:* (a) ℓ_c = s·ℓ off NA with
s ~ Bernoulli(c) independent of φ given the pair; condition on the pair. (b) c against 1 − c. (c)
E[g | s = 1] = c·E[1 − q*(NA) | s = 1]. Elkan and Noto recommend e1 ("the first estimator is still the best one to
use") and call it exact "if g(x) = p(s = 1|x) precisely for all x"; their proof that g = c on P sets p(y = 1 | x) = 1
for x ∈ P, which (c) makes explicit, and DS's own target violates it (Thm 1). Exact on a toy by enumerating all keep sets; corpus mass ratio within 0.05 of c
for c ≥ 0.4.

**Prop 4 (held-out evaluation reports c × precision).** For a prediction set S chosen independently of which pairs the
KB knows, E[|S ∩ KB_c|/|S|] = c·|S ∩ W|/|S|. *Proof:* linearity; each true fact is in KB_c with probability c. Exact on
a toy; on the corpus, against an independent KB (stream 3602), within 0.1 of c × the world precision for c ≥ 0.4.

**Def 6 (the pipeline).** Topic 22's headline tagger (constrained Viterbi) gives each test sentence's spans; a span of
a linked type links to the KB entity whose tokens it spells exactly (names and aliases, type checked; on gold spans
this reproduces every gold link); candidates and facts as in Defs 1 and 4, with the classifier trained on gold spans.

**Prop 5 (error propagation).** Under strict matching (both argument spans and types exact, and the relation),
end-to-end triple recall ≤ the share of gold triples whose two arguments were both predicted. *Proof:* a correct
predicted triple is built on two predicted mentions equal to the gold ones. The gold-trained classifier meets it.

## 3. Experiment grid (33 s)

| Block | What | Cost |
|---|---|---|
| Data | train `generate_corpus(300, 0, roster=train)` (6,113 sentences, 2,924 candidates, 1,125 triples); dev `(60, 2, train)`; tests `(200, 1)` on each roster (2,038 / 1,964 candidates) | ~2 s |
| Classifiers | softmax over NA + 10 relations by L-BFGS, λ = 1/n (soft-label objective; = Topic 17's at one-hot); MIL by L-BFGS on the noisy-OR objective (gradient complex-step checked) | ~3 s |
| κ sweep | 11 DS fits with a share κ of related co-mentions kept (stream 3601) | ~1 s |
| c grid | 8 DS fits against KB_c (stream 3600), dev for e1, an independent KB (stream 3602) | ~1 s |
| Pipeline | Topic 22's headline tagger (21 s pretraining + 4 s fit) on both tests; 3 classifiers × 2 span sources | ~27 s |

## 3a. Numbers the page quotes (harness, ≥ 6 decimals)

**Sentence level (seen / unseen), P R F1; share of related co-mentions labeled with a relation.**

| classifier | seen | unseen | related (seen / unseen) |
|---|---|---|---|
| gold | 1.000000 1.000000 1.000000 | 1.000000 1.000000 1.000000 | 0 / 0 |
| DS | 0.779749 0.975196 0.866589 | 0.797531 1.000000 0.887363 | 0.272340 / 0.439189 |
| MIL | 0.976703 0.711488 0.823263 | 1.000000 0.811146 0.895726 | 0 / 0 |
| DS + argument words | 0.682971 0.984334 0.806417 | 0.814013 0.989164 0.893082 | 0.812766 / 0.304054 |
| DS + encoder | 0.679125 0.972585 0.799785 | 0.813586 0.797214 0.805317 | 0.795745 / 0.263514 |
| gold + encoder | 0.997389 (F1) | 0.969650 (F1) | 0 / 0 |

DS labels every entailed candidate (1.000000) and 0.055094 / 0.038600 of unrelated co-mentions.

**Facts (P R F1), max / noisy-OR.** Seen: gold 1.000000 0.983240 0.991549 both; DS max 0.891753 0.966480 0.927614,
noisy-OR 0.718876 1.000000 0.836449; MIL max 1.000000 0.860335 0.924925, noisy-OR 1.000000 0.877095 0.934524.
Unseen: DS max 0.701493 1.000000 0.824561, noisy-OR 0.330986 1.000000 0.497354; MIL 1.000000 0.957447 0.978261 both.

**Cor 1 templates (train n, ρ, π, fitted P(not NA)).** commented on conditions across (ORG→SECTOR) 108 0.685185
1.000000 0.687843; has spoken publicly about (PERSON→ORG) 65 0.646154 0.595238 0.648634; attended an investor event
hosted by 58 0.586207 0.500000 0.588339; hosted investors in (ORG→GPE) 53 0.566038 0.533333 0.573632; held its annual
meeting in 52 0.596154 0.516129 0.601421; also follow (both orders) 40 0.600000 0.625000 0.604707 / 0.599676; and …
both 34 0.529412 0.611111 0.551149 / 0.543278; came up during the … discussion 34 0.529412 0.500000 0.539065 /
0.525596.

**κ sweep (κ = 0, 0.1, …, 1).** DS F1 0.929683 for κ ≤ 0.4, 0.893541 for 0.5–0.9, 0.866589 at 1; related co-mentions
labeled 0, then 0.165957 (the IN_SECTOR template crosses), then 0.272340 (the officer template crosses).

**c grid (c, KB facts, mass ratio, e1, raw R, raw F1, R and F1 after dividing by c; fact P vs world / KB / indep).**
1: 254 1.000000 0.909820 0.975196 0.866589 0.975196 0.866589; 0.891753 0.891753 0.891753.
0.9: 221 0.867326 0.794965 0.972585 0.865776 0.972585 0.865776; 0.891192 0.766839 0.813472.
0.8: 197 0.756483 0.701201 0.956919 0.931977 0.959530 0.868794; 1.000000 0.751479 0.798817.
0.7: 171 0.673675 0.630934 0.925587 0.928618 0.960836 0.872555; 1.000000 0.641509 0.710692.
0.6: 142 0.571237 0.575324 0.664491 0.780077 0.958225 0.866588; 1.000000 0.598214 0.616071.
0.5: 118 0.467865 0.507161 0.567885 0.707893 0.842037 0.803238; 1.000000 0.509434 0.537736.
0.4: 93 0.374489 0.414955 0.249347 0.393004 0.853786 0.823678; 1.000000 0.475410 0.442623.
0.3: 64 0.216862 0.290541 0.033943 0.064596 0.579634 0.679419; 1.000000 0.687500 0.500000.

**Pipeline (triple P R F1, Prop 5's bound; facts P R).** Predicted spans, seen: gold 0.994662 0.729765 0.841867
bound 0.729765; DS 0.763231 0.715405 0.738544; MIL 0.975610 0.522193 0.680272. Unseen: gold 0.983444 0.919505 0.950400
bound 0.919505; DS 0.777487 0.919505 0.842553; MIL 1.000000 0.753870 0.859665 (facts 1.000000 0.957447). Argument miss
rates (seen / unseen): ORG 0.189 / 0.076, PERSON 0.201 / 0.014, GPE 0.045 / 0.006, SECTOR 0.021 / 0.

## 4. The lab (`RelationExtractionLab`, `data/relation-extraction-lab.json`, 5 tabs)

The candidate (rule: the first seen-test "has spoken publicly about" co-mention of a related pair that DS labels with
a relation and MIL labels NA) is "martin hollis has spoken publicly about pellucid software" (KB: CFO_OF).

1. **DS labels.** The candidate with its DS label set and each classifier's probabilities; Prop 1's decomposition
   as a stacked bar (correct / co-mention / entailed / overlap) for train and seen; per-relation precision.
2. **What DS learns.** The 11 templates: ρ, π, fitted P(not NA), and the decision line ρπ = 1 − ρ in the (ρ, π) plane.
   Slider κ moves each template's ρ (payload per κ) and shows which cross; sentence F1 and related co-mentions
   labeled.
3. **At least one.** The Hollis → Pellucid bag: each mention, stated or not, MIL and DS probabilities, w_i, and Cor 2's
   coefficient; the violation rates; MIL's forced/free recall; sentence vs fact F1 for DS and MIL, max vs noisy-OR.
4. **Incomplete KB.** Slider c over the grid: KB facts, mass ratio against the line c, raw and corrected recall/F1, e1
   against c, and fact precision against the world, the KB, and an independent KB (with the line c × world).
5. **Pipeline & KB.** Test (seen/unseen) × spans (gold/predicted) × classifier: triple P/R/F1 with Prop 5's bound
   marked, fact P/R; the argument-word and encoder rows for memorization.

## 5. Checkpoint questions (for Jonathan)

1. **Headline.** Lead with Thm 1/Cor 1 ("DS learns the KB's opinion of the pair, at every sample size") and frame
   MIL as trading sentence recall for fact precision, with Prop 2 locating its misses?
2. **Encoder rows.** Keep "DS + encoder" / "gold + encoder" (entity-start states memorize the KB like argument words)
   as one paragraph and one lab row, or drop them?
3. **Thm 3(b)'s surprise.** Feature "an incomplete KB raises DS's sentence F1 at c = 0.8 before collapsing it" in the
   page and the *Incomplete KB* tab (recommended), or keep it to a remark?
4. **The gold ceiling.** The page says plainly that gold-trained extraction is perfect on this templated corpus, so
   the topic is about supervision, not reading. OK?

## 5b. rigorFlag (draft)

The corpus is synthetic and templated: with sentence-level gold the extraction is perfect, so every number here
measures supervision, not reading. Co-mention sentences relate their pair 60% of the time by the generator's design,
which sets every ρ in Cor 1. Whether "headquartered in Toronto, Canada" states BASED_IN(X, Canada) is a guideline
choice: the corpus says no, so those labels count as DS errors at the sentence level though they are true facts. Thm 1
is a population statement; the fitted classifier is L2-regularized and linear, and matches it within 0.05 per
template. Noisy-OR treats mentions as independent, which repeated templates violate. Def 5's incompleteness is
completely at random by pair, unlike real KBs (well-known entities are better covered); the corrected rows divide by
the true c, and Elkan and Noto's e1 estimate is biased here; each c is one random KB. Linking is exact string matching
against a KB that names every entity, with no unlinkable mentions and no ambiguity. λ = 1/n throughout, untuned.

## 6. Notebook cell map (23 cells)

1 intro · 2 imports and corpus sizes · 3–4 Prop 1 · 5–6 Thm 1, Cor 1 templates · 7 gold vs DS · 8–9 κ sweep · 10–11
Thm 2, Cor 2, the witness bag · 12 violations and MIL · 13–14 Prop 2 · 15–16 Def 4, Prop 3 · 17–18 Thm 3, Prop 4 ·
19–20 pipeline, Prop 5 · 21 memorization, extracted KB · 22–23 harness.

## 7. References (verified 2026-10-26 against full text, except where noted)

- Baldini Soares, Livio, Nicholas FitzGerald, Jeffrey Ling, and Tom Kwiatkowski. 2019. "Matching the Blanks:
  Distributional Similarity for Relation Learning." *ACL 2019*, 2895–905. P19-1279.
  https://doi.org/10.18653/v1/P19-1279. §3.3, "entity start state": the final hidden states at the entity-start
  marker tokens, concatenated (with entity markers in the input; ours has none, so we read the mentions' first
  tokens: say so). The Anthology spells the second author "Fitzgerald"; the PDF has "FitzGerald".
- Bekker, Jessa, and Jesse Davis. 2020. "Learning from Positive and Unlabeled Data: A Survey." *Machine Learning* 109
  (4): 719–60. https://doi.org/10.1007/s10994-020-05877-5. Names the assumption SCAR (§3.1.1, Def 1 in arXiv
  1811.04820).
- Craven, Mark, and Johan Kumlien. 1999. "Constructing Biological Knowledge Bases by Extracting Information from Text
  Sources." *ISMB-99*, 77–86. https://www.aaai.org/Papers/ISMB/1999/ISMB99-010.pdf. "Weakly labeled" examples from a
  database paired with the abstracts it cites; Mintz et al. cite it as the precedent.
- Dempster, A. P., N. M. Laird, and D. B. Rubin. 1977. "Maximum Likelihood from Incomplete Data via the EM
  Algorithm." *JRSS B* 39 (1): 1–22 (1–38 with discussion). https://doi.org/10.1111/j.2517-6161.1977.tb01600.x. §3,
  Lemma 2 with Eq. 3.2 (Fisher's identity, unnamed; the scan's equation numbers need a visual check in Phase B).
- Dietterich, Thomas G., Richard H. Lathrop, and Tomás Lozano-Pérez. 1997. "Solving the Multiple Instance Problem with
  Axis-Parallel Rectangles." *Artificial Intelligence* 89 (1–2): 31–71.
  https://doi.org/10.1016/S0004-3702(96)00034-3. §1: an object is positive iff at least one of its instances is.
- Elkan, Charles, and Keith Noto. 2008. "Learning Classifiers from Only Positive and Unlabeled Data." *KDD 2008*,
  213–20. https://doi.org/10.1145/1401890.1401920. §2: "selected completely at random"; Lemma 1,
  p(y = 1 | x) = p(s = 1 | x)/c; estimators e1, e2, e3, e1 recommended.
- Hoffmann, Raphael, Congle Zhang, Xiao Ling, Luke Zettlemoyer, and Daniel S. Weld. 2011. "Knowledge-Based Weak
  Supervision for Information Extraction of Overlapping Relations." *ACL-HLT 2011*, 541–50. P11-1055. §1: 18.3% of
  the Freebase facts matched to NYT 2007 sentences have overlapping relations; §3.2: Φ_join is a deterministic OR.
- Lin, Yankai, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. 2016. "Neural Relation Extraction with Selective
  Attention over Instances." *ACL 2016*, 2124–33. P16-1200. https://doi.org/10.18653/v1/P16-1200.
- Mintz, Mike, Steven Bills, Rion Snow, and Dan Jurafsky. 2009. "Distant Supervision for Relation Extraction without
  Labeled Data." *ACL-IJCNLP 2009*, 1003–11. P09-1113. §4: any sentence containing both entities "might express that
  relation"; §5.1: each lexical feature is the conjunction of the between words, their order, and a window, with the
  entity types (§5.3); features are aggregated per entity pair (a pair-level classifier, unlike ours, which reads
  sentences); §6: negatives are a random 1% of unrelated pairs; §7.1: held-out evaluation "suffers from false
  negatives" and gives "a rough measure of precision" (Prop 4 quantifies it; they do not say "underestimates").
- Riedel, Sebastian, Limin Yao, and Andrew McCallum. 2010. "Modeling Relations and Their Mentions without Labeled
  Text." *ECML PKDD 2010*, LNCS 6323, 148–63. https://doi.org/10.1007/978-3-642-15939-8_10. §1: at least one
  sentence "might express that relation"; the DS assumption is violated about 13% of the time on Wikipedia and 31% on
  NYT. Their Table 1 (100 candidates each for three relations) gives NYT 38/35/20% and Wikipedia 20/20/10%, which
  averages 16.7%, not 13%: quote the 31% and the table, not the 13%.
- Surdeanu, Mihai, Julie Tibshirani, Ramesh Nallapati, and Christopher D. Manning. 2012. "Multi-instance Multi-label
  Learning for Relation Extraction." *EMNLP-CoNLL 2012*, 455–65. D12-1042.
- Takamatsu, Shingo, Issei Sato, and Hiroshi Nakagawa. 2012. "Reducing Wrong Labels in Distant Supervision for Relation
  Extraction." *ACL 2012*, 721–29. P12-1076.
- Zeng, Daojian, Kang Liu, Yubo Chen, and Jun Zhao. 2015. "Distant Supervision for Relation Extraction via Piecewise
  Convolutional Neural Networks." *EMNLP 2015*, 1753–62. https://doi.org/10.18653/v1/D15-1203. §3.5, Eqs. 8–9: each
  bag trains on its most probable instance.
- Not cited: Louis (1982), whose full text returned 403 (OUP) and is gated on JSTOR.
