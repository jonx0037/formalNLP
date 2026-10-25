# Brief — `beam-search-decoding`

**Decoding as Search: Greedy and Beam Search.** Roadmap #14. Its prerequisite is `neural-language-models` (#5), which is
published. It is a prerequisite of `sampling-temperature-nucleus` (#15) and `overlap-metrics-bleu-rouge` (#27), and the
first topic in `decoding-generation`. It is not on the §6 overlap watch-list.

**Phase A** (this brief and the verified notebook), 2026-10-24:
- **Harness:** 19/19 pass under numpy==2.5.4 and scipy==1.18.1 in ≈ 31 s, also with warnings promoted to errors.
- **Notebook:** 19 cells; executes with exit 0 in ≈ 35 s, including kernel start, and is committed without outputs.
- **Lab data:** `src/components/viz/data/beam-search-decoding-lab.json` is 9.4 kB, pinned by
  `test_viz_mirror_matches_payload`.

**Scope decisions** (Jonathan, 2026-10-24, on Claude's recommendations):
- **Spine: search error against model error.**
  - Decoding as a shortest path, and exact search.
  - Greedy and beam search as pruned search, with counterexamples.
  - A certificate.
  - Length bias and its corrections.
  - Non-termination.
- **Engine: three.**
  - Hand-built toy trees, exact by enumeration.
  - Topic 3's KN trigram, exact by Dijkstra on its state graph.
  - A Topic-13 causal decoder, exact by best-first tree search.
- **Finance thread:** the mode is boilerplate.
- **Authorship:** every body is written by Claude for Jonathan's review, as in Topics 5–13.
- **Difficulty:** `intermediate`, domain `decoding-generation`, `financeCaseStudy: true`.

**The spine.** A prefix tree with edge weight −log p ≥ 0; the mode is its cheapest root-to-`</s>` path.
- **Exact search.** Uniform-cost search is exact and pops at most (2 + E|Y|)/p(y*) prefixes (Thm 1).
- **Recombination.** A Markov model's prefixes recombine on their states, so Dijkstra finds its mode in polynomial time.
  For RNNs the problem is undecidable, or NP-complete at polynomial length (Chen et al., cited).
- **Pruned search.** Greedy and beam search prune that search and carry no guarantee (Thm 2).
- **Certificate.** One extra number, the best pruned score, certifies when beam search found the global mode (Thm 3).
- **Length bias.** Stopping beats every continuation once `</s>` is the argmax (Thm 4).
  - Normalization breaks monotonicity, and without a cap it can have no optimum.
  - Greedy decoding is Meister et al.'s regularized optimum.
- **Non-termination.** Greedy on a Markov model is eventually periodic (Thm 5), and a consistent model can make beam(k)
  run forever.
- **On the corpus.** As k grows, search error falls and the strings shrink to a mode shorter than the text.
- **Finance.** The mode stops early, and normalization drags outputs onto an 8-K template.

**Changes from the approved plan, each forced by a spike or the harness:**
1. **One prompt set for both engines.** The 188 distinct three-word openings of held-out MD&A sentences
   (`generate_corpus(100, 1)`) whose words are all in the decoder's 509-word vocabulary.
   - The MD&A *document* openers are templated: only two distinct prompts came from first sentences.
2. **Decoding conditions on no `<unk>`.** Both engines renormalize each step without `<unk>` (and the decoder without
   `<s>`).
   - Otherwise `<unk>` is a frequent, uninformative mode token.
   - This is per-step renormalization, not the model conditioned globally, and the rigorFlag says so.
3. **One beam run serves every α.** All open prefixes have the same length, so normalization never changes which are
   kept. The runs for α ∈ {0, 0.5, 1} share one search and differ only in when they stop. This took the harness from
   48 s to 31 s.
4. **The normalized beam needs its own stopping rule.** It stops once the best finished score beats
   logp(h)/lp(T) for every open h. That bound is also what makes Thm 3 hold under normalization, so the
   certificate's α > 0 form is new and proved.
5. **No exact normalized mode on the corpus.** A per-length uniform-cost search for the trigram's α = 1 optimum took
   88 s for 60 prompts and was incomplete. At α > 0 the page reports beam results with the certificate rate, and
   search-error rates only at α = 0.
6. **Meister et al.'s R_greedy squares the surprisal gap** (their Eq. 11), as the reference check found. The harness
   uses the squared form.
7. **Thm 3 is sufficient, not necessary.** On Welleck's stationary model with 2 ≤ k ≤ 4, beam(k + 1) returns the mode
   uncertified, because it pruned words likelier than ε. This is pinned in `test_consistent_model_beam_never_ends`.
8. **The decoder is never empty; the trigram often is.** The trigram's empty modes (0.196809) come from backoff, which
   gives `</s>` a unigram share after rare contexts. Thm 4(a)'s sufficient condition covers only 0.037234 of the
   prompts.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Decoding as Search: Greedy and Beam Search |
| `{{SUBTITLE}}` | The most probable string is a shortest path: uniform-cost search finds it, greedy and beam search prune the search with no guarantee, one pruned score certifies success — and the string they converge to is shorter and blander than the text the model learned from |
| `{{SLUG}}` | `beam-search-decoding` (= graph node id) |
| `{{DOMAIN}}` | `decoding-generation` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[neural-language-models]` |
| `{{FORMALML_PREREQS}}` | `shannon-entropy`; connections `kl-divergence`, `sequential-monte-carlo` |
| `{{FORMALRAG_PREREQS}}` | none |
| `{{STATS_CALC_PREREQS}}` | Calc `sequences-limits` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/n-gram-models-smoothing/`, `notebooks/perplexity-cross-entropy/`, `notebooks/scaling-laws/`, `notebooks/finance-corpus/` |

- **Tags:** `decoding`, `beam-search`, `greedy-decoding`, `dijkstra`, `search-error`, `length-bias`,
  `length-normalization`, `uniform-information-density`, `non-termination`.
- `estimatedReadTime` ≈ 40.
- **In-site `connections`:**
  - Topic 3 (the trigram engine, its backoff);
  - Topic 4 (`KNVector`);
  - Topic 5 (consistency: its bounded-state RNN is tight, and Welleck's Prop contrasts with it);
  - Topic 13 (the decoder).

  Only the #5 edge is in the graph. The others are prose links.
- **Graph:** the incoming edge #5 → #14 exists, and so do the outgoing #14 → #15 and #14 → #27.

**Scope.**
- **No sibling owns decoding.**
- **#15 owns** sampling, temperature, top-k, nucleus, and typicality. The page's "the mode is atypical" remark
  (NLL 1.367 against the text's 2.576) is one sentence that points there.
- **#27 owns** BLEU and ROUGE. Our "model error" is measured by length and predictability only.
- **Out of scope:**
  - minimum Bayes risk decoding (named in §5 as the alternative Eikema and Aziz argue for);
  - constrained decoding, diverse beam search, and speculative decoding;
  - A* with learned heuristics (Hart et al. named for admissibility);
  - KV caching (the decoder recomputes prefixes; a performance detail).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | the lab tree after "revenue": greedy's path, beam(2)'s, the mode | — | cells 0–1 |
| 1 | Decoding as a shortest path | the tree with −log p on the edges | Def 1; **Prop 1**; **Thm 1** (exact + work bound); Cor (recombination); Chen et al. | cells 2–3 |
| 2 | Greedy and beam as pruned search | the lab tree's frontier, cut to k | Def 2; **Thm 2** (a)(b)(c) | cells 4–5 |
| 3 | A certificate | the pruned candidates as greyed nodes, the best one marked | **Thm 3**; the length-capped form; Huang et al. | cells 6–7 |
| 4 | Length bias | `</s>` at 0.1 against 0.9 spread over many words | **Thm 4** (a)(b); no optimum under normalization; Meister's Thm 3.1 at k = 1 | cells 8–9 |
| 5 | Search error against model error | curves vs k on both engines | — (measurements; Koehn and Knowles, Stahlberg and Byrne) | cells 10–11 |
| 6 | Non-termination | the trigram's greedy loop "and and and" | **Thm 5**; **Prop** (Welleck, stationary instance) | cells 12–13 |
| 7 | Finance: the mode is boilerplate | prompt → mode → normalized output, three rows | — | `<FinanceCaseStudy>`, cells 14–15 |
| 8 | What decoding is not | — | rigorFlag in prose | — |

## 2. Statements and proofs

The notebook's markdown cells carry the full statements and proofs; the MDX copies them, changing only formatting.

- **Def 1.** p(y | x) = Π p(y_t | x, y_<t) over strings ending in `</s>`; the mode y* = argmax. A consistent model has
  a mode, because only finitely many strings exceed any ε.
- **Prop 1.** With w(h, w) = −log p(w | x, h) ≥ 0, −log p(y | x) is the path weight. Proof: take −log of the product.
- **Thm 1** (uniform-cost search).
  - **(i) Exact:** the first finished string popped is a mode.
  - **(ii) Work:** if E|Y| < ∞, the search makes at most (2 + E|Y|)/p(y*) pops.
  - **Proof:**
    - the frontier invariant (every unpopped string has a frontier prefix of no greater cost);
    - pops are non-decreasing in cost;
    - Σ_t P(|Y| ≥ t)/p* = (1 + E|Y|)/p* counts the open prefixes, and 1/p* bounds the finished strings.
  - **Lazy siblings:** pop pushes the best child and the next sibling. The invariant survives because an unpushed sibling
    is never cheaper than its stand-in.
  - **Harness:** brute force on 200 random trees (3 words, depth 5, Dirichlet(0.5)), with the pop count under the bound.
- **Cor** (recombination). For an order-n Markov model, prefixes with equal last n − 1 tokens share every completion's
  cost difference. So Dijkstra on ≤ (|V| + 1)^{n−1} states is exact; Viterbi is the length-layered form.
  - **Harness:** graph search equals tree search on all 188 trigram prompts, with no fewer pops for tree search.
- **Cited, not proved** (Chen et al. 2018, single-layer ReLU RNNs with rational weights):
  - best string undecidable (Thm 8);
  - decidable for consistent RNNs, but the length is unbounded by any computable function (Thms 9–10);
  - NP-complete at polynomial length (Thm 11).
- **Def 2** (beam(k)). Full expansion: keep the top k candidates; `</s>`-candidates move to the finished set; stop when
  the best finished string ≥ the best open prefix. Greedy is beam(1).
- **Thm 2** (no guarantee). The three constructions are `greedy_trap`, `beam_trap`, and `nonmonotone_beam`.
  - **(a)** Greedy/mode = 1.5/m < ε, for ε = 0.1, 0.01, 0.001.
  - **(b)** p_a = 0.9/(k + 1), for k = 1, …, 8.
  - **(c)** Beam(1), beam(2), beam(3) return 0.06, 0.004, 0.06.
  - **Proofs:** by direct computation.
- **Thm 3** (certificate). If log p(f) ≥ max(best pruned, best open at exit), then f is a mode.
  - **Proof:** every string finished in the beam, extends a pruned candidate, or extends an open prefix.
  - **Length-capped form:** with lp nondecreasing and log p ≤ 0, a completion of h scores ≤ logp(h)/lp(T).
  - **Remark:** sufficient, not necessary (change 7).
  - **Harness:** soundness on 100 random trees × k ∈ {1, 2, 3, 5} and on both engines × all k. At α > 0, no certified
    result is beaten by any k's result.
- **Thm 4.**
  - **(a)** If `</s>` is the argmax after h, then max_z p(hz) = p(h`</s>`). Proof:
    p(hwz) ≤ p(h)p(w | h) ≤ p(h)p(`</s>` | h).
  - **(b)** If every step is ≤ q, then p(y) ≤ q^{|y|}, so |y*| ≤ log p(y′)/log q.
  - **Harness:** every prefix of 100 trees for (a), with the trigram's `</s>`-argmax prompts ⊆ its empty-mode prompts;
    every string of 100 trees for (b).
- **No optimum under normalization.** For p(w) = 0.9 and α = 1, f(n) = 6(n log 0.9 + log 0.1)/(n + 6) is strictly
  increasing to 6 log 0.9. Proof: the sign of f′ is the sign of 6 log 0.9 − log 0.1 > 0.
- **Meister et al.'s Thm 3.1, proved on finite tie-free trees.** R_greedy = Σ_t (u_t(y_t) − min_w u_t(w))² ≥ 0, with
  equality only on greedy's string. So greedy is the unique maximizer for λ above a finite threshold.
  - **Harness:** 50 random trees, at λ₀ + 1 and at 0.
  - **Cited:** their Thm 3.2 (beam with set regularizer) and the UID regularizers of their §5 (variance, local
    consistency, max, squared). One sentence.
- **Thm 5** (eventually periodic). σ_{t+1} = F(σ_t) on a finite set S; by pigeonhole a state repeats by step |S|.
  - **Harness:** every trigram orbit either ends (and equals beam(1)'s output) or cycles (and beam(1) is unfinished at
    the cap).
- **Prop** (Welleck et al. 2020, stationary instance). k words, each (1 − ε)/k, and `</s>` ε < (1 − ε)/k.
  - The model is consistent, since P(|Y| ≥ t) = (1 − ε)^t, yet beam(k) never terminates.
  - Proof: the best open prefix's k word-children outrank every `</s>`-child.
  - **Cited:** their Thm 3.4 (an RNN for every incomplete decoder) and Thms 4.2–4.3 (the self-terminating fix).

## 3. Numbers the page quotes (harness, ≥ 6 decimals)

| Quantity | Value | Test |
|---|---|---|
| prompts / held-out continuation length (words) | 188 / 7.298203 | — |
| trigram mode: length / empty share / copied verbatim | 2.367021 / 0.196809 / 0.223404 | `test_search_error_against_model_error` |
| trigram pops to the mode: median / max | 8 / 114 | — |
| `</s>`-argmax prompts (Thm 4(a)'s condition) | 0.037234 | `test_stopping_dominates` |
| trigram search error, k = 1, 4, 16, 64 | 0.803191 / 0.281915 / 0.026596 / 0 | same |
| trigram length, k = 1, 4, 16, 64 | 4.503704 / 2.792553 / 2.446809 / 2.367021 | same |
| trigram certified, k = 1, 4, 16, 64 | 0.159574 / 0.686170 / 0.957447 / 1 | `test_certificate_is_sound` |
| decoder mode: length / empty / copied | 4.861702 / 0 / 0.303191 | — |
| decoder pops to the mode: median / max | 25.5 / 4,292 | — |
| decoder search error, k = 1, 4, 16, 64 | 0.670213 / 0.324468 / 0.271277 / 0.207447 | `test_search_error_against_model_error` |
| decoder length, k = 1, 4, 16, 64 | 6.664894 / 6.404255 / 5.920213 / 5.510638 | same |
| decoder certified, k = 1, 4, 16, 64 | 0.180851 / 0.484043 / 0.601064 / 0.670213 | — |
| prompts where a wider beam did worse: trigram / decoder | 0.015957 / 0.015957 | `test_beam_is_not_monotone_in_k` |
| greedy cycles on the trigram (all into "and and …", period 1) | 0.281915 | `test_greedy_is_eventually_periodic` |
| per-word NLL: held-out text / trigram mode | 2.575659 / 1.366751 | `test_the_mode_is_boilerplate` |
| trigram α = 1 length, k = 4, 16, 64 | 4.792553 / 5.127660 / 7.053191 | same |
| trigram α = 1 certified, k = 64 | 0.260638 | — |
| α = 1 beam(16) on the 8-K tail "to disclose a material event" | 0.132979 | same |
| lab tree: greedy / mode / beam(2) certified | 0.0864 / 0.238 / yes | `test_lab_tree` |
| decoder final batch loss (nats/token) | 1.343470 | — |

Page precision: 3 decimals from the payload (`_r`).

## 4. The lab (`BeamSearchLab`, `data/beam-search-decoding-lab.json`)

Three views and a table. All numbers come from the payload.
- **(a) Search-tree stepper.** The lab tree after "revenue": 21 open prefixes and 34 nodes in all, edge labels p, node
  labels cumulative P.
  - **Controls:** the algorithm (greedy, beam(2), beam(3), uniform-cost) and a step button.
  - **Beam display:** the kept, finished, and pruned nodes at each step (`tree.beams[k].steps`), with the best pruned
    score and the certificate's verdict at the end.
  - **Uniform-cost display:** the pops in order (`tree.ucs`).
  - *Learn:*
    - greedy commits to "rose" and never sees "was flat";
    - beam(2) finds it and can prove it, because nothing it pruned beat 0.238;
    - uniform-cost search pops in probability order and stops at the first finished string.
- **(b) Curves vs k.**
  - **Controls:** an engine toggle (trigram ↔ decoder) and an α toggle (0, 0.5, 1).
  - **Lines:** search error (α = 0 only), certified share, mean length with the mode's length and the held-out length
    as reference lines, and copied share, over log₂ k.
  - *Learn:*
    - the trigram's search error reaches 0 while length collapses to the mode's;
    - the decoder keeps searching;
    - normalization buys length and loses the certificate.
- **(c) Greedy-loop strip.** `loops.examples`: prompt → pre-period → the cycle, drawn as a loop arrow, with the loop
  share. *Learn:* a finite-state model's greedy run is a walk on a finite graph.
- **Examples table (in §7).** For six prompts, the outputs of greedy, beam(16), normalized beam(16), and the exact
  mode, for each engine.

Mobile: every view stacks; the toggles are flex-wrap rows (CLAUDE.md's 375 px rules).

## 5. rigorFlag (draft)

> **What is proved.** Thms 1–5 and the propositions are proved, with one exception: the hardness of exact decoding for
> RNNs is cited from Chen et al. (2018), and so are Meister et al.'s beam regularizer for k > 1 and Welleck et al.'s RNN
> construction.
>
> **What is measured, and on what.** The corpus numbers come from two small models: a trigram over 3,427 words and a
> two-block, width-32 decoder over 509 words. The prompts are 188 openings of synthetic MD&A sentences. Search-error
> rates at a given k depend on the model, the prompts, and the length cap (40 tokens for the trigram, the decoder's
> 32 positions).
>
> **Three further limits.**
> - Both engines are decoded conditioned, step by step, on never emitting `<unk>`. That is a renormalized model, not
>   the original one.
> - "Model error" is measured by length and per-word predictability against held-out text. It is not measured by
>   human judgment or by BLEU (Topic 27).
> - At α > 0 we have no exact optimum on the corpus, only the certificate's lower bound on how often beam search
>   found it.

## 6. Notebook cell map

19 cells:

| Cells | Content |
|---|---|
| 0–1 | title, headline |
| 2–3 | §1: Def 1, Prop 1, Thm 1, Cor, Chen |
| 4–5 | §2: Def 2, Thm 2 |
| 6–7 | §3: Thm 3, capped form |
| 8–9 | §4: Thm 4, no optimum, Meister |
| 10–11 | §5: curves |
| 12–13 | §6: Thm 5, Welleck |
| 14–15 | §7: finance |
| 16–18 | payload and harness |

## 7. Cross-site (verified `status: published` on each sibling's GitHub `main`, 2026-10-24)

- `formalmlPrereqs`:
  - `shannon-entropy`: the edge weights are surprisals (its Def 1, information content), and R_greedy and uniform
    information density are statements about surprisal.
- `formalmlConnections`:
  - `kl-divergence`: decoding the mode is the extreme of mode-seeking; its "Forward vs Reverse KL" section contrasts
    mode-covering and mode-seeking fits.
  - `sequential-monte-carlo`: beam search is a deterministic, top-k version of particle resampling, and it suffers the
    same weight degeneracy (its §4 resampling schemes, Def 1 effective sample size).
- `formalcalculusPrereqs`:
  - `sequences-limits`: prefix log-probabilities form a monotone non-increasing sequence. The normalized chain of §4 is
    monotone and bounded, so it converges to a supremum it never attains (its Thm 1, Monotone Convergence).
- **Not cited:**
  - formalML `message-passing` covers GNNs, with no max-product or Viterbi.
  - formalStatistics `hidden-markov-models` and formalCalculus `convex-optimization` are `draft`.
  - formalRAG `selective-generation-abstention` (Chow's reject rule) seemed too loose a link for the empty mode; see Q3.

## 8. References (Chicago 17 N-B; verified against ACL Anthology, PMLR, Crossref, arXiv; full entries in the MDX)

- **Search:**
  - Dijkstra 1959 (*Numer. Math.* 1: 269–71, doi 10.1007/BF01386390);
  - Hart, Nilsson, and Raphael 1968 (IEEE TSSC 4(2): 100–107, doi 10.1109/TSSC.1968.300136);
  - Viterbi 1967 (IEEE TIT 13(2): 260–69, doi 10.1109/TIT.1967.1054010);
  - Karp 1978 (*Discrete Math.* 23(3): 309–11, doi 10.1016/0012-365X(78)90011-0);
  - Lowerre 1976 (CMU PhD thesis, CSD record URL; see Q2).
- **Search and model error:**
  - Koehn and Knowles 2017 (WNMT, W17-3204);
  - Stahlberg and Byrne 2019 (EMNLP-IJCNLP, D19-1331: 51.8% empty modes);
  - Eikema and Aziz 2020 (COLING, 2020.coling-main.398);
  - Cohen and Beck 2019 (PMLR 97: 1290–99; byline "J. Christopher Beck").
- **Length:**
  - Wu et al. 2016 (arXiv 1609.08144, Eq. 14: lp(Y) = (5 + |Y|)^α/(5 + 1)^α);
  - Murray and Chiang 2018 (WMT, W18-6322);
  - Huang, Zhao, and Ma 2017 (EMNLP, D17-1227; Thm 1, "modulo beam size").
- **Theory:**
  - Meister, Cotterell, and Vieira 2020 (EMNLP, 2020.emnlp-main.170; Thms 3.1–3.2, Eq. 11);
  - Chen et al. 2018 (NAACL, N18-1205; Thms 8–11);
  - Welleck et al. 2020 (EMNLP, 2020.emnlp-main.448; Def 2.9, Thm 3.4, Thms 4.2–4.3 in App. C).
- **Repetition:** Holtzman et al. 2020 (ICLR, openreview rygGQyrFvH).

## 9. Open questions for the checkpoint

1. **The decoder's search error at k = 64 (0.207) is the page's case for "no recombination".** A larger decoder (d = 48,
   more steps) would be a better model and might search differently. Keep d = 32 for the time budget (training ≈ 5 s of
   the ≈ 31 s), or spend ≈ 10 s more?
2. **Lowerre's thesis.** The CMU degree record titles it "HARPY Speech Understanding System"; it is usually cited as "The
   HARPY Speech Recognition System". Neither title is linked to a full text. Cite it via the CMU record with the common
   title, or drop it and credit beam search's origin in prose only?
3. **A formalRAG link.** The empty mode is a decoder "abstaining". Is a connection to `selective-generation-abstention`
   (its Thm 2, Chow's rule) worth a sentence, or too loose?
4. **The trigram's one cycle.** Every greedy loop falls into "and and and …" (period 1). The strip would show many prompts
   draining into one cycle. Is that the right picture, or should Thm 5's lab illustration use a toy with longer periods?
