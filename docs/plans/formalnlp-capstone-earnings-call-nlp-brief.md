# Brief — `capstone-earnings-call-nlp`

**Capstone: An Earnings-Call NLP Pipeline.** This is roadmap #29, the last node (domain `extraction-knowledge-graphs`).
Its three graph prerequisites are published:

- `lexicon-sentiment-loughran-mcdonald` (#18), the dictionary tone stage;
- `knowledge-graph-embeddings` (#24), which completes the extracted KB;
- `topic-models-lda` (#25), whose `topic_mixtures` is the topic stage.

The pipeline also runs four other published topics' models: Topic 17's LR, Topic 19's aspect reader, Topic 22's
tagger, and Topic 23's MIL reader. It compares the readers with Topic 28's tests.

**Status: Phase A done and approved by Jonathan (2026-10-27) with the recommended answers to §5:**

- the Bayes-tone remark plus one assert;
- no EDGAR pipeline;
- the lab reshaped as needed;
- the proposed LDA sentence;
- no waiver.

**Phase B shipped:**

- **The MDX, `CapstonePipelineLab` (5 tabs), and the graph and curriculum flip.** Every track's `planned` list is now
  empty; the graph keeps 42 edges.
- **The LDA page's qualification** (`topic-models-lda.mdx`, the capstone's-stage paragraph).
- **Phase B corrections.**
  - Three new asserts back prose claims:
    - `test_tone_share_is_mostly_the_bayes_floor`;
    - `test_soft_handoffs_keep_mass_on_missed_spans` (92 missed gold spans, 61 with marginal mass above 0.1);
    - `test_merge_by_noisy_or_not_by_log_odds`. The Brightwater inverse is 0.398, not 0.08, and the additive merge
      drops the fact at τ = 0.55.
  - The functional rows' removals at τ = 0.05 are all false facts (asserted).
  - The per-slot argmax differs from the MAP only at τ = 0.05 on the unseen calls (asserted).
  - Eight "$i$'s"-style possessives became smart quotes in the build and were rephrased.
  - Unlinkable types (MONEY, PERCENT, DATE) were drawn struck through in the trace; only failed links are struck
    through now.
- **Gates.**
  - `pnpm test` 39/39, `astro sync`, `validate`, and `build` all pass.
  - `audit:cross-site`: formalNLP 226 edges, 0 wrong direction, 0 slug drift; every missing reciprocal is deferred.
  - Page: 0 `.katex-error`, 383 KaTeX spans, no `‘`.
  - Lab, checked by script against the payload: 54 lattice states, every handoff, KB, and readers state, and both
    sliders driven by real key presses at both ends.
  - Mobile preset: `innerWidth` and `scrollWidth` are 375 in all 24 tab and toggle states.
  - No console errors; `/paths` shows all 9 tracks complete.
- `notebooks/capstone-earnings-call-nlp/capstone_earnings_call_nlp.py`: 23 tests, 23/23 under `-W error` in ≈ 55 s.
  - Topic 22's tagger and Topic 12's pretraining, inside `predicted_corpus` and `chains`, are ≈ 27 s of that.
  - No waiver was needed.
- `01_capstone_earnings_call_nlp.ipynb` (23 cells) exits 0 under `jupyter execute`.
- Lab data `src/components/viz/data/capstone-pipeline-lab.json` (64.2 kB) is written by `--write-payload` and pinned by
  `test_viz_mirror_matches_payload`.

**Phase A decisions (plan mode, 2026-10-27, with Jonathan):**

- **Math core:** three results no sibling has.
  - Shapley attribution over oracle swaps.
  - Soft vs. hard handoffs.
  - Joint KB inference with a total-unimodularity theorem.
  - formalRAG `capstone-multimodal-financial-rag` already owns the cascade product law (Thm 1) and its FKG lower
    bound (Prop 1), so the page cites them.
- **Aspects:** a matcher stage is added, and its error enters the attribution. It has none, so it becomes Cor 1's
  example: a null player.
- **Topic stage:** fold in only the 80 fixed nouns; report the cross-seed drop; qualify the LDA page's "any filing"
  sentence in this PR.
- **Runtime:** 60 s waived to ≈ 2 min. The harness does not use it.
- **Defaults:** `advanced`; Claude writes all code.

**What building the harness found** (each now an assert):

1. **Tone takes the largest share, and most of it is the Bayes floor.** Both calls, LR reader (Shapley shares of
   F1):

   | Stage | Share |
   |---|---|
   | entities | 0.021536 |
   | relations | 0.015181 |
   | tone | 0.040302 |
   | aspects | 0.000000 |

   - The aspect reader is the Bayes rule on these calls: its tone equals the Bayes tone of the gold polarities on
     every call.
   - Even so, tone keeps a 0.021983 share. That is error no reader can remove (Topic 17's ceiling is 0.890).
   - Oracle gain is not fixable error.

2. **The aspect stage is a null player, exactly.** The term matcher finds every gold aspect span (the vocabulary is
   closed and exhaustive, as `fc`'s exhaustiveness test already asserts), and Topic 19's bigram reader reads all of
   them right (1,209 of 1,209). So v(S ∪ A) = v(S) for every S: A gets 0 in every order and in every view (Cor 1).

3. **The one sizable interaction is entities × relations, and it is complementary.**
   - d({E, R}) = 0.003237 (both views, LR).
   - A ceiling analysis that fixes entities first credits them 0.020346; fixing relations first gives entities
     0.023583.
   - In the error game, the ER dividend is the only nonzero one: 6 errors, against shares of 34 / 24 / 66 / 0
     (entities / relations / tone / aspects).
   - The F1 game adds small negative ET and RT dividends (−0.000774, −0.000528). They come from F1's denominator, not
     from any stage interaction (Prop 2).

4. **The held-out roster is easier for stage E.** Entities' share is 0.033780 on the seen calls against 0.005783 on
   the unseen ones.
   - Topic 22's [UNK] reading tags the six held-out companies almost perfectly.
   - The seen roster's rarely covered companies and cities are the hard names (Topic 22's finding, now visible
     end to end).

5. **Prop 1 needs modularity.**

   | Calls | r_E | r_R | Bonferroni | product | P(A∩B) | end-to-end |
   |---|---|---|---|---|---|---|
   | seen | 0.843478 | 0.817391 | 0.660870 | 0.689452 | 0.739130 | 0.695652 |
   | unseen | 0.955556 | 0.866667 | 0.822222 | 0.828148 | 0.866667 | 0.866667 |

   - Unseen: end-to-end recall equals P(A ∩ B) exactly.
   - Seen: end-to-end is 0.043 below P(A ∩ B). Predicted spans change the reader's context even when both arguments
     are found.
   - Every end-to-end success lies inside A ∩ B (asserted).

6. **Soft handoffs carry information but need a retuned downstream.**

   | Handoff | F1 at τ = 0.5 | AP | F1, τ tuned on filings |
   |---|---|---|---|
   | hard | 0.820513 | 0.743813 | 0.838384 (τ = 0.45) |
   | marginal | 0.756757 | 0.851481 | 0.880383 (τ = 0.1) |
   | sampled | 0.763441 | 0.838490 | 0.880383 (τ = 0.1) |

   - Seen calls: soft handoffs lose at the hard pipeline's threshold, win AP, and win F1 once τ is re-fit on the same
     run's filings.
   - Topic 22's F*/2 rule, computed from the scores alone, gives 0.808290. It is short of that, because noisy-OR
     scores built from q-scaled mention probabilities are underconfident.
   - Unseen calls (entity errors rare): AP is flat (0.920164 / 0.917869 / 0.918946). The tuned soft handoff ties hard
     (0.941176) for marginal and is 0.012605 behind for sampled.

7. **Inverse facts must merge by noisy-OR, not by adding log-odds.** The additive merge let a reader that states only
   one direction veto it: Brightwater Analytics PARENT_OF Larch Data (0.685) was dropped for its weak inverse.
   With the noisy-OR merge (every mention of either direction is evidence for the one fact):
   - merging raises F1 at every τ ≥ 0.4;
   - the functional rows never lower F1 against the merged threshold, and raise it at τ = 0.05 everywhere (unseen
     0.863158 → 0.921348; seen predicted 0.805825 → 0.825871);
   - the repairs are a second HQ city, a region read as a city, and a second CFO.

8. **The rules impose the closed world.** BASED_IN = HQ_IN ∘ LOCATED_IN, both directions, and ACQUIRED ⇒ PARENT_OF.
   - At τ = 0.5 on the seen calls they lower F1 (gold spans 0.904762 → 0.888889).
   - They do it by deleting true LOCATED_IN facts (Boston–Massachusetts, Denver–Colorado) whose BASED_IN leg no call
     states. Topic 24's finding 2 shows up again: unstated facts become negatives.
   - They help on the unseen calls at τ = 0.1 (0.941176 → 0.953488).

9. **The LP is integral even where Thm 3 does not apply.**
   - With the rules, the constraint matrix is not TU (Prop 3's 5-variable instance has a fractional optimum: value
     3/2 against the integer 1).
   - Yet every corpus LP (both runs, both span sources, 19 thresholds, with and without rules) is integral. Roth and
     Yih (2004, §3.3) report the same.
   - The per-slot argmax differs from the MAP only at τ ≤ 0.05 on the unseen calls, where the one-CEO-per-company
     row binds (a bipartite matching, Thm 3(b)'s boundary).

10. **The filings fill what the calls missed.**
    - The unseen calls state 6 facts the pipeline misses: the "X, the parent company of Y" template in both
      directions, plus BASED_IN Japan and LOCATED_IN Osaka.
    - 4 of them are in Topic 24's completion KB, extracted from the same companies' filings, and the other 2 are
      inverses of filed facts.
    - RotatE ranks all 12 queries first. Joint inference over all of a company's documents recovers them; the calls
      alone do not.

11. **The topic stage needs the fixed nouns.**
    - Dominant-topic sector accuracy on the calls: 0.272727 (seen) and 0.583333 (unseen) with all words, against
      0.681818 and 0.750000 with the 80 fixed nouns.
    - The cause: each generator run invents its own pseudo-words, and on seed-1 text they carry seed 0's sectors.

12. **The tone readers, compared on 126 calls.**

    | Reader | Accuracy |
    |---|---|
    | aspects | 0.857143 |
    | LR | 0.738095 |
    | dictionary | 0.587302 |

    - Every McNemar p < 0.01.
    - The cluster AR (whole companies swapped) widens LR vs. dictionary from 0.003378 to 0.035500.
    - It does not widen aspects vs. LR (0.001490 → 0.001400): calls of one company share errors for the
      document-level readers.

13. **The shares are uncertain at 22 companies.** 90% cluster-bootstrap intervals, both views, LR:

    | Stage | Interval |
    |---|---|
    | entities | [0.009682, 0.038871] |
    | relations | [0.009603, 0.022881] |
    | tone | [0.032186, 0.050034] |

    - P(φ*_E > φ*_R) = 0.744, so entities and relations cannot be ordered.
    - On the unseen calls, entities' interval touches 0.

---

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Capstone: An Earnings-Call NLP Pipeline |
| `{{SUBTITLE}}` | Four published stages read earnings calls into a knowledge graph: which stage costs what (Shapley over oracle swaps), what one stage should hand the next (posteriors, if the next is refit), and when the KB's own constraints repair the output exactly (total unimodularity). |
| `{{SLUG}}` | `capstone-earnings-call-nlp` |
| `{{DOMAIN}}` | `extraction-knowledge-graphs` |
| `{{DIFFICULTY}}` | `advanced` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[lexicon-sentiment-loughran-mcdonald, knowledge-graph-embeddings, topic-models-lda]` (the graph's 3 edges; the graph stays at 42 edges) |
| connections (in-site) | `named-entity-recognition`, `relation-extraction-distant-supervision`, `aspect-based-sentiment`, `linear-text-classifiers`, `significance-testing-for-nlp` |
| `{{FORMALRAG_PREREQS}}` | `capstone-multimodal-financial-rag` (Thm 1, the product law exact under independence; Prop 1, positive dependence makes the product a lower bound); connection `cross-encoders-reranking` (Thm 2, the recall pinch: an oracle reranker's ceiling is the pool's recall, the one-stage ceiling analysis) |
| `{{FORMALML_PREREQS}}` | `shannon-entropy` (Thm 4, the data processing inequality: Thm 2a is its decision-theoretic cousin); `bayesian-inference` (Thm 7.1, the Bayes-optimal decision rule, which Thm 2a compares across inputs) |
| `{{STATS_CALC_PREREQS}}` | formalCalculus `probability-and-union-bound` (Cor 1, Bonferroni's inequality: Prop 1's lower bound); formalStatistics `bootstrap` (Thm 3, Efron–Bickel–Freedman consistency, behind Prop 4's resampling); formalStatistics `sufficient-statistics` (Thm 4, Rao–Blackwell: conditioning on more never hurts, Thm 2a's other cousin) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/relation-extraction-distant-supervision/`, `notebooks/significance-testing-for-nlp/` |

**Sibling check (GitHub `main`, 2026-10-27).** Every slug above is `status: published`, with the block numbers
read from the fetched MDX:

- formalRAG capstone: Def 1, Thm 1, Prop 1, Prop 2, …
- `cross-encoders-reranking`: Thm 2 "The recall pinch", Cor 1 "The cascade ceiling".
- formalCalculus `probability-and-union-bound`: Cor 1 "Bonferroni's inequality".
- formalML `shannon-entropy`: Thm 4 "Data Processing Inequality".
- formalML `bayesian-inference`: Thm 7.1 "Bayes-optimal decision rule".
- formalStatistics `sufficient-statistics`: Thm 4 Rao–Blackwell.
- formalStatistics `bootstrap`: Thm 3 Efron–Bickel–Freedman.

**No sibling has** Shapley attribution, oracle-substitution ceiling analysis as a theorem, Blackwell's comparison of
experiments, Finkel–Manning–Ng sampling pipelines, joint vs. pipeline inference, total unimodularity, or two-sided
Fréchet bounds (grepped all four siblings' `main`, 2026-10-27).

**Scope against formalRAG's capstone.** Its product law and FKG bound are cited, not re-proved. Prop 1 adds only:

- the upper bound;
- the modularity condition under which end-to-end recall is P(A ∩ B) at all;
- the corpus's measured gap from modularity.

## 1. Section outline (geometric first)

1. **Overview.** One unseen-roster call traced through every stage (lab tab *Trace a call*), ending in the call
   graph: facts, subject, tone, and aspect polarities. Three questions follow: blame, handoffs, and repair.
2. **The pipeline and its oracles.** Def 1. A table of the four stages, each with the published topic it comes from.
3. **How errors compound.** Prop 1, citing formalRAG Thm 1 and Prop 1; then modularity on the corpus.
4. **Who gets the blame.**
   - Geometric first: the Boolean lattice of 16 oracle configurations, where a ceiling analysis is one path up it.
   - Def 2, Thm 1, Cor 1, Prop 2.
   - Lab tab *Oracle lattice*.
5. **What one stage should hand the next.** Def 3, Thm 2 (the garbling picture: the argmax maps the posterior
   simplex onto its vertices). Lab tab *Hard vs. soft*.
6. **Letting the KB check itself.** Def 4, Thm 3, Prop 3 (the half-HQ vertex drawn as a polytope corner). Lab tab
   *Joint KB*.
7. **How sure are the shares.** Prop 4.
8. **Finance case study** (`<FinanceCaseStudy>`):
   - (a) the calls end to end;
   - (b) three tone readers side by side, tested (the lexicon page's promise);
   - (c) the topic stage across generator runs;
   - (d) the filings fill what the calls missed.
   - Lab tab *Readers*.
9. **Where this sits** (the curriculum's end: each earlier topic's model, composed) / honest caveats /
   implementation.

## 2. Statements (numbered per type, in page order; the same numbers in the `.py` and the notebook)

**Def 1 (pipeline, call graph, oracle).**

- **Stages and the pipeline.** Stages s₁, …, s_k map a call to intermediate outputs (spans with links; facts;
  tone; aspect polarities). The pipeline's output is the call graph 𝒢: the facts the calls state, and per call c the
  triples (c, ABOUT, subject), (c, TONE, t), and (c, ASPECT:a, ±1).
- **Oracle.** Stage i's oracle replaces its output by the gold one, with later stages reading it.
- **Gold.** The call graph with every oracle in place.
- **Stage R's oracle under predicted mentions.** It is the set of facts stated between the found, typed, and linked
  mentions.

**Prop 1 (two-sided bounds).**

- **Setting.** For a gold fact f, let A_f be the event that both arguments are found and linked, and B_f the event
  that the reader recovers f from gold mentions; r_E = P(A) and r_R = P(B) over gold facts.
- **Bounds.** max(0, r_E + r_R − 1) ≤ P(A ∩ B) ≤ min(r_E, r_R). The product law P(A ∩ B) = r_E r_R holds under
  independence (formalRAG Thm 1), and P(A ∩ B) ≥ r_E r_R under positive association (formalRAG Prop 1).
- **Modularity.** If the reader recovers f from predicted mentions whenever A_f ∩ B_f holds, and never otherwise,
  then end-to-end recall equals P(A ∩ B).
- **Proof.**
  - Lower bound: P(A ∩ B) = 1 − P(Aᶜ ∪ Bᶜ) ≥ 1 − P(Aᶜ) − P(Bᶜ) (formalCalculus Cor 1).
  - Upper bound: A ∩ B ⊆ A and A ∩ B ⊆ B.
  - Independence: by definition.
  - Modularity: then success = A ∩ B.
- **Harness.** `test_prop1_two_sided_bounds_and_modularity` checks the bounds in every view, checks success ⊆ A ∩ B,
  and confirms that modularity holds on the unseen calls and fails on the seen ones.

**Def 2 (the oracle game; dividends).**

- **The game.** N = {E, R, T, A}. v(S) = F1(𝒢_S) − F1(𝒢_∅), with 𝒢_S the call graph with the stages in S replaced
  by their oracles. The error game is v_err(S) = err(∅) − err(S), where err counts false plus missing triples.
- **Harsanyi dividends.** d(T) = Σ_{S ⊆ T} (−1)^{|T|−|S|} v(S), so v(S) = Σ_{T ⊆ S} d(T) (Möbius inversion on the
  Boolean lattice).
- **Ceiling analysis.** In order π, stage i's sequential share is v(P_π(i) ∪ {i}) − v(P_π(i)), where P_π(i) is the
  set of stages before i.

**Thm 1 (ceiling analysis and the Shapley value).**

- (a) The sequential share of i in order π is Σ d(T) over the coalitions T whose last member under π is i.
- (b) All k! orders give every stage the same share iff d(T) = 0 for every |T| ≥ 2.
- (c) The average share over orders is φᵢ = Σ_{T ∋ i} d(T)/|T|, which equals Shapley's formula
  Σ_{S ∌ i} |S|!(k − |S| − 1)!/k! · (v(S ∪ i) − v(S)). It is the unique map satisfying efficiency, symmetry, null
  player, and additivity.

**Proof.**
- (a) v(P ∪ {i}) − v(P) = Σ_{T ⊆ P∪i} d(T) − Σ_{T ⊆ P} d(T) = Σ_{T ⊆ P∪i, T ∋ i} d(T). With P = P_π(i), these are
  exactly the T whose members other than i all precede i.
- (b) If every higher dividend is 0, the share is d({i}) in every order. Conversely, suppose all orders agree. For
  i ∈ T, compare an order that lists T ∖ {i} first and then i with an order that starts with i: by (a) the difference
  in i's share is Σ_{T′ ⊆ T, T′ ∋ i, |T′| ≥ 2} d(T′). Induct on |T| from 2 up: every smaller T′ has d(T′) = 0, so
  d(T) = 0.
- (c) Under a uniform random order each member of T is equally likely to be last, so i's expected share is
  Σ_{T ∋ i} d(T)/|T|. Shapley's weight |S|!(k−|S|−1)!/k! is the probability that the set before i is S. For
  uniqueness:
  - write v = Σ_T d(T) u_T, with u_T(S) = 1[T ⊆ S];
  - for each T and each real c, null player gives φᵢ(c u_T) = 0 for i ∉ T, symmetry gives equal values on T, and
    efficiency gives c/|T|;
  - additivity sums the 2^k − 1 games.
- **Harness.** `test_thm1_ceiling_analysis_and_the_shapley_value` checks (a)–(c) on 30 random 4-player games (one in
  three with only singleton dividends), plus unanimity games and additivity.
  `test_thm1_on_the_corpus_entities_and_relations_are_complements` covers the corpus.

**Cor 1 (null players).** If v(S ∪ {i}) = v(S) for every S, then every dividend containing i is 0, and i's share is 0
in every order and under Shapley.
- **Proof.** Pair each S ∌ i with S ∪ {i} in d(T)'s alternating sum, for T ∋ i; each pair cancels.
- **Harness.** `test_cor1_the_aspect_stage_is_a_null_player` (aspects, every reader, every view, both games).

**Prop 2 (additive and coupled metrics).**

- **Statement.** Suppose each triple x's correctness depends only on S ∩ W_x, where W_x is the set of stages that
  write x. Then v_err = Σ_x g_x(S ∩ W_x), so d(T) ≠ 0 only if T ⊆ W_x for some x. Micro F1 is a ratio of the same
  sums, so its dividends are not confined that way.
- **Proof.** Möbius inversion is linear. The dividends of a function of S ∩ W vanish off the subsets of W: for i ∉ W,
  the function ignores i (Cor 1).
- **On the corpus.**
  - W = {E, R} for facts, {E} for ABOUT, {T} for TONE with the dictionary or LR, and {A} for ASPECT.
  - So the error game's only possible higher dividend is d({E, R}), and the corpus has d({E, R}) = 6.
  - The F1 game has d(ET), d(RT) ≠ 0.
  - The aspect reader's W for TONE is {E, A, T} in principle (tone-bearing sentences name no company), but stage E's
    errors never move a sentence on these calls, so it is additive too.
- **Harness.** `test_prop2_errors_add_over_disjoint_triples_f1_does_not`.

**Def 3 (handoffs).**

- **Hard.** Stage i passes ŷ = argmax q(y | x), where q is its posterior.
- **Soft.** Stage i passes q itself: as span marginals (Topic 22's Prop 4), or as K samples (Finkel, Manning, and Ng
  2006; forward filtering, backward sampling, Carter and Kohn 1994).
- **Mechanics.**
  - A marginal handoff scales each candidate pair's relation probability by q_head · q_tail.
  - A sampled handoff averages the K runs' noisy-OR fact scores (Finkel et al. take a majority vote of last-stage
    samples; the harness keeps scores, for AP).

**Thm 2 (garbling).**

- (a) For any loss and any downstream decision, a Bayes downstream that sees q has risk ≤ one that sees argmax q.
- (b) A fixed downstream rule can lose: there is a loss and a rule δ with R(δ ∘ q) > R(δ ∘ argmax q).
- **Proof.**
  - (a) h = argmax is a function of q, so every rule δ(h(q)) is a rule of q, and the Bayes risk over the larger class
    is no larger. This is the easy direction of Blackwell's comparison (1951; 1953); cf. formalML Thm 4 (DPI) and
    formalStatistics Thm 4 (Rao–Blackwell).
  - (b) Take q = 0.6 (span right) and p = 0.8 (reader states the fact given the right span), so P(fact) = 0.48. Missed
    facts cost 3 and false ones 1, so the Bayes act asserts when P > 1/4. Let δ assert when the score is > 1/2.
    - Hard: the span is kept, the score is 0.8, and the fact is asserted, with cost 0.52.
    - Soft: the score is 0.48 and the fact is withheld, with cost 1.44.
    - Re-fitting δ to the soft score (τ = 1/4) restores 0.52.
- **Harness.** `test_thm2a_the_argmax_is_a_garbling` (300 random finite worlds by exact enumeration, 22 strict),
  `test_thm2b_a_fixed_rule_can_lose`, `test_ffbs_samples_the_constrained_chain`, and `test_handoffs_on_the_corpus`.
- **Link to Topic 22.** Its Thm 3 (F*/2) is the F1 analog of re-fitting δ.

**Def 4 (constrained KB MAP).**

- **Variables and weights.** Variables x_f ∈ {0, 1} over scored facts, with weights w_f = logit s_f − logit τ. With
  no constraints the MAP is the threshold rule.
- **Merging.** Inverse pairs (PARENT_OF, SUBSIDIARY_OF) and symmetric pairs (COMPETES_WITH) are one variable, with
  noisy-OR score 1 − Π(1 − s) over both directions.
- **Functional rows.**
  - Σ x ≤ 1 per head slot (h, r) of the head-functional relations: HQ_IN, IN_SECTOR, BASED_IN, LOCATED_IN,
    SUBSIDIARY_OF, CEO_OF, CFO_OF, ACQUIRED.
  - Σ x ≤ 1 per tail slot (r, t) of the tail-functional ones: CEO_OF, CFO_OF, ACQUIRED.
  - Both are read off WORLD's maximum degrees.
- **Rules** (Prop 3):
  - hq + loc − based ≤ 1 and hq + based − loc ≤ 1 for every (HQ_IN(h, o), LOCATED_IN(o, c), BASED_IN(h, c));
  - acq − sub ≤ 0.

**Thm 3 (functional constraints are totally unimodular).**

- (a) After merging, the functional rows' matrix is TU, so the LP relaxation max wᵀx, Ax ≤ 1, 0 ≤ x ≤ 1 has an
  integral optimum (Hoffman and Kruskal 1956), which is the MAP.
- (b) If no tail-functional row binds, the MAP is each head slot's best positive-weight fact, together with every
  positive-weight fact outside the functional relations. When both families bind it is a maximum-weight bipartite
  matching.
- **Proof.**
  - (a), Heller and Tompkins' criterion, proved directly:
    - Every column has at most one 1 in the head family (a fact has one head slot) and at most one in the tail family.
    - Take a square submatrix B and induct on its size.
    - A zero column gives det 0. A column with a single 1 lets us expand to a smaller case.
    - Otherwise every column has exactly one 1 in each family. Then the sum of the head-family rows minus the sum of
      the tail-family rows is 0, so det B = 0.
    - Appending the identity rows for x ≤ 1 keeps TU (expand along them). TU with integral right-hand sides gives an
      integral polytope.
  - (b) The rows partition the variables into disjoint slots, so the LP separates into one problem per slot:
    max Σ w_j x_j subject to Σ x_j ≤ 1, whose optimum is the largest positive w_j or nothing.
- **Harness.** `test_thm3_functional_rows_are_totally_unimodular`: brute-force minors on 40 random fragments; every
  corpus LP integral; random weights on the corpus matrices; argmax = MAP except at binding CEO rows.

**Prop 3 (composition breaks TU).**

- **The instance.** Variables (hq₁, hq₂, loc₁, loc₂, based), with the rows hq₁ + hq₂ ≤ 1 and the two rules per i.
  The minor on rows (hq₁ + loc₁ − based, hq₁ + based − loc₁) and columns (hq₁, loc₁) is −2.
- **Fractional optimum.** With w = (1, 1, −2, −2, 1), x = (½, ½, 0, 0, ½) is feasible with value 3/2. The integer
  optimum is 1: hq_i = 1 forces based = loc_i, and loc costs 2.
- **On the corpus.** The LP with rules is integral at every τ (Roth and Yih 2004, §3.3, saw the same in thousands of
  cases).
- **Harness.** `test_prop3_composition_rules_break_total_unimodularity` and `test_joint_inference_on_the_corpus`.

**Prop 4 (one bootstrap for every share).**

- **Statement.** φ = M v, with M the k × 2^k Shapley matrix, and v(S) = f1(Σ_c C_c(S)) − f1(Σ_c C_c(∅)), with C_c the
  (tp, predicted, gold) counts charged to company c.
- **Consequence.** Resampling companies with multiplicities W gives φ* = M (f1(W C(S)) − f1(W C(∅))). F1 is
  homogeneous of degree 0, so weights and counts agree (Topic 28, Def 1).
- **Proof.** Shapley's formula is linear in v, and v* is v computed on the resampled pooled counts.
- **Charging.** A call triple goes to its covered company; a fact goes to the company of the first call that
  co-mentions its pair.
- **Harness.** `test_prop4_shapley_is_linear_so_one_bootstrap_serves_every_share` (2,000 resamples per view and
  reader).

## 3. Findings (harness, 6 decimals)

See the header list; every number there is printed by `summary()`. Additional:

- **v(∅) in F1, both views:**

  | Reader | v(∅) |
  |---|---|
  | dictionary | 0.899379 |
  | LR | 0.922981 |
  | aspects | 0.941615 |

- **Per-test tone accuracy:**

  | Reader | Seen calls | Unseen calls |
  |---|---|---|
  | dictionary | 0.485 | 0.700 |
  | LR | 0.697 | 0.783 |
  | aspects | 0.848 | 0.867 |

- **Trace calls:**
  - seen synth-0058: errs at E, R, and T, with 4 mis-tagged sentences;
  - unseen synth-0076: errs at E and T, with 4 mis-tagged sentences and no fact error.

## 4. Lab — `CapstonePipelineLab.tsx` (5 tabs; never `*Laboratory.tsx`; no "pipelineStage", "modality", or "architectures" under `src`)

1. **Trace a call.**
   - A test toggle (seen synth-0058 / unseen synth-0076).
   - The call's sentences, with predicted vs. gold spans (types colored) and aspect terms underlined with their
     polarity.
   - Tone-bearing sentences marked.
   - The call's facts (tp / fp / fn), its subject, and the three readers' tones against gold.
   - Lesson: errors are local, and the graph is where they meet.
2. **Oracle lattice.**
   - The Hasse diagram of the 16 coalitions, with F1 (or errors) at each node.
   - A view toggle (seen / unseen / both), a reader toggle, and a metric toggle.
   - Pick an order (drag to reorder, or a select) to highlight its chain, with the sequential shares beside the
     Shapley shares.
   - Nonzero dividends listed, and the bootstrap intervals as whiskers.
   - Lesson: a path is an attribution; only interactions make paths disagree.
3. **Hard vs. soft.**
   - τ slider over TAU_GRID, with a test toggle.
   - P, R, and F1 curves for the hard, marginal, and sampled handoffs.
   - Markers at 0.5, F*/2, and the filings-tuned τ; AP per handoff.
   - Lesson: soft wins only with a re-fit downstream.
4. **Joint KB.**
   - τ slider; test and span-source toggles.
   - F1 for threshold, merged, functional, and rules.
   - The facts each step adds or removes at that τ.
   - Prop 3's polytope picture: the 5-variable instance, with LP point ½ vs. integer 1.
   - Fractional-variable count (always 0 on the corpus).
5. **Readers.** Accuracy per reader and test, McNemar vs. cluster-AR p per pair, topic-stage accuracies (all words
   vs. fixed nouns), and the six missed facts with their filing status and RotatE ranks.

## 5. Checkpoint questions (Phase A → B)

1. **The tone share is mostly the Bayes floor.**
   - Recommendation: say so with Topic 17's ceiling.
   - Also add a second tone oracle, "Bayes tone" (the Bayes rule on gold aspect polarities), to split fixable from
     irreducible error. On these calls it equals the aspect reader's output exactly, so the split is
     reader-dependent: LR's fixable part is 0.040302 − 0.021983.
   - Accept as a prose remark plus one assert, or skip?
2. **EDGAR.** Running the whole pipeline on the three EDGAR excerpts needs new-text NER wiring: the excerpts carry no
   sentence spans, and Topic 22's API takes only its named splits. Recommendation: don't. Quote Topic 19's EDGAR
   aspect readings in prose only.
3. **Lab reshaping.** The payload already has every number listed in §4. Phase B may drop the "both" view if the lab
   gets crowded.
4. **The LDA page's qualification.** Proposed sentence for `topic-models-lda.mdx:640`, after "…returns its six
   proportions": "On a fresh generator run its pseudo-words carry this run's sectors, so the capstone folds in only
   the 80 seeded nouns."
5. **The 60 s rule.** The harness runs ≈ 55 s, so the waiver can stay unused.

## 5b. rigorFlag (draft)

The stages are published topics' models, scored on the synthetic corpus they were built for. So:

- The aspect stage's zero share reflects a closed, exhaustive aspect vocabulary, not a solved problem.
- The calls are generated text, not earnings-call transcripts.

Shapley shares are defined by the oracle game the page chooses: the metric, how coarse each oracle is, and the
call graph's mix of triples. They are not causal error budgets. Tone's share includes error that no reader can
remove: the aspect reader is already the Bayes rule here.

With 22 companies the cluster-bootstrap intervals are wide, and entities cannot be ranked against relations.

Soft handoffs win only after the downstream threshold is re-fit, here on the same run's filings. Topic 22's F*/2
rule assumes calibrated scores and falls short.

The functional constraints come from the synthetic world's own maximum degrees; real KBs have exceptions. That the
LP with composition rules comes out integral is an observation on this corpus, not a theorem: Prop 3 shows a
fractional vertex.

The fixed-noun topic stage repairs a generator artifact, not a property of real filings.

## 6. Notebook cell map (`01_capstone_earnings_call_nlp.ipynb`, 23 cells)

| Cells | Content |
|---|---|
| 1–2 | Intro; imports (`cap`) |
| 3–4 | Def 1, data counts, gold-side checks |
| 5–6 | Prop 1 bounds table |
| 7–8 | Def 2 / Thm 1: Shapley, dividends, three orders |
| 9–11 | Cor 1, Prop 2 (error game); Shapley across views and readers |
| 12–13 | Thm 2a/b, FFBS check, handoff table |
| 14–15 | Thm 3, Prop 3, KB table |
| 16–17 | Filings fill the calls (completion ranks) |
| 18–19 | Prop 4 intervals; readers compared |
| 20–21 | Topic stage |
| 22–23 | Full harness |

## 7. References (verified 2026-10-27)

- **Shapley, Lloyd S. 1953.** "A Value for n-Person Games." In *Contributions to the Theory of Games II*, Annals of
  Mathematics Studies 28, 307–17 (Crossref lists 307–318). doi:10.1515/9781400881970-018.
- **Harsanyi, John C. 1963.** "A Simplified Bargaining Model for the n-Person Cooperative Game." *International
  Economic Review* 4 (2): 194–220. doi:10.2307/2525487.
- **Hoffman, A. J., and J. B. Kruskal. 1956.** "Integral Boundary Points of Convex Polyhedra." In *Linear Inequalities
  and Related Systems*, Annals of Mathematics Studies 38, 223–46. doi:10.1515/9781400881987-014. Crossref dates the
  digitized volume 1957; the print volume is 1956.
- **Heller, I., and C. B. Tompkins. 1956.** "An Extension of a Theorem of Dantzig's." Same volume, 247–54.
  doi:10.1515/9781400881987-015.
- **Blackwell, David. 1953.** "Equivalent Comparisons of Experiments." *Annals of Mathematical Statistics* 24 (2):
  265–72. doi:10.1214/aoms/1177729032. The 1951 Berkeley Symposium paper "Comparison of Experiments" is cited from
  this one.
- **Finkel, Jenny Rose, Christopher D. Manning, and Andrew Y. Ng. 2006.** "Solving the Problem of Cascading Errors:
  Approximate Bayesian Inference for Linguistic Annotation Pipelines." EMNLP, 618–26. ACL Anthology W06-1673.
  - Sampling at each stage conditioned on the previous stages' samples; a majority vote of last-stage samples.
  - k-best lists as the prior practice.
- **Roth, Dan, and Wen-tau Yih. 2004.** "A Linear Programming Formulation for Global Inference in Natural Language
  Tasks." CoNLL. ACL Anthology W04-2401.
  - §3.3: the matrix "is not unimodular", yet the LP relaxation "still produces integer solutions" in thousands of
    cases.
- **Carter, C. K., and R. Kohn. 1994.** "On Gibbs Sampling for State Space Models." *Biometrika* 81 (3): 541–53.
  doi:10.1093/biomet/81.3.541. Forward filtering, backward sampling.
- **Lipton, Elkan, and Naryanaswamy (2014)**, Cor 1 (F*/2), as Topic 22 cites it.
- In-site: Topics 17, 18, 19, 22, 23, 24, 25, 28. Sibling: formalRAG capstone Thm 1 and Prop 1.
- Not cited: Ng's "ceiling analysis" lecture (course video only, no stable URL). The page names the practice without
  attributing it.
