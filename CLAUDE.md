# CLAUDE.md — formalNLP

The mathematics of language. Fifth site in the **formal** series (formalCalculus,
formalStatistics, formalML, formalRAG). Deploys to https://formalnlp.com — **live but `noindex`
until launch**. General preferences (pnpm, uv, Chicago citations, American English,
geometric-first, git workflow) live in `~/CLAUDE.md` — not repeated here.

- Site design (approved): [`docs/plans/formalnlp-site-design.md`](docs/plans/formalnlp-site-design.md)
- Scaffold plan + Topic Roadmap: [`docs/plans/2026-10-22-formalnlp-scaffold-plan.md`](docs/plans/2026-10-22-formalnlp-scaffold-plan.md)
- Per-topic workflow: [`STARTER-PROMPT.md`](STARTER-PROMPT.md)

## Positioning

Rigor-first and gap-filling: tokenization and the statistics of text, language modeling,
representations, attention and transformers, decoding, classification and sentiment, information
extraction and knowledge graphs, text mining, and evaluation — 28 topics + a capstone, lighter
than formalML/formalRAG but held to the same three-pillar standard. A finance case-study thread
(*Applied NLP for Finance*) runs through it. Visual identity: **Marginalia** (oxblood + manuscript
gold on warm paper, Crimson Pro).

## Tech Stack

Scaffolded from formalRAG: Astro 6 + MDX, React 19 + D3.js, Tailwind 4, KaTeX (`remark-math` +
`rehype-katex`), Pagefind, self-hosted fonts via `@fontsource-variable`, pnpm, Vercel. Node ≥ 22.12.

## Commands

```bash
pnpm dev                  # Dev server at localhost:4321
pnpm test                 # tsx --test tests/*.test.ts — domains, curriculum DAG, branding, WCAG
                          #   contrast, robots/noindex, cross-site parsing, publish invariants
pnpm build                # Production build + Pagefind (8 GB heap baked in; minify off on Vercel)
pnpm preview              # Serves dist/ for DOM checks (preview_start "formalnlp-preview"); rebuild
                          #   after every edit; the port moves if another session holds 4329
pnpm exec astro sync      # Validate schema + ALL frontmatter fast
pnpm validate             # validateConnections.ts — roadmap nodes without MDX are NOTICES
pnpm audit:cross-site     # Five-site reciprocity validator (siblings adjacent or FORMAL_*_PATH);
                          #   totals span all sites — read formalNLP's row in the report's Summary
# Python pillar (per-topic, no shared venv; keep the pins — fc.CORPUS_FINGERPRINT assumes this numpy):
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/<slug>/<slug_underscored>.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/<slug>/01_<slug_underscored>.ipynb
```

## Single sources of truth

- `src/data/domains.ts` — the 9 domains, their labels, and the 4 sibling sites. The content
  schema, `curriculum.ts`, the graph color scale, and the tests all import it.
- `src/data/curriculum-graph.json` — all 29 nodes (planned → published) and the 42-edge
  prerequisite DAG. `src/data/curriculum.ts` mirrors the still-planned titles per track.
- A topic touches: `src/content/topics/<slug>.mdx`, `src/components/viz/<Name>Lab.tsx` and its
  `data/<name>-lab.json`, `notebooks/<slug>/` (`.py` + `.ipynb`), `docs/plans/formalnlp-<slug>-brief.md`,
  the node's `status` in `curriculum-graph.json`, and its title in `curriculum.ts`'s `planned`.
- `src/lib/siteConfig.ts` — `PRELAUNCH` (the launch switch) and `SITE_URL`.
- `scripts/lib/crossSite.mjs` — the five sites, their paths/env vars, and cross-site field parsing.
- `src/styles/global.css` — Marginalia tokens (light fixed by the design; dark checked for WCAG AA
  in `tests/contrast.test.ts`).

## Content schema (`src/content.config.ts`)

- `domain`: the 9 keys in `domains.ts`. No RAG pipeline-stage or input-type fields (those are formalRAG-only).
- `financeCaseStudy` (bool), `rigorFlag` (honest caveat), `references.type` includes `documentation`.
- Cross-site arrays for **four** siblings: `formalml`, `formalcalculus`, `formalstatistics`,
  `formalrag` × `Prereqs`/`Connections`. `relationship` ≥ 40 chars.

## Content & verification conventions

- Start each topic with `git checkout main && git pull`: the checkout is often left on the last
  (merged) docs branch, which lacks the previous topic's files.
- Three pillars per topic: rigorous math (KaTeX) + interactive D3 viz + working Python. The
  notebook contract is in `notebooks/README.md`; the `.py` owns every number the viz mirrors.
- A dependent topic's `.py` **imports** its prereq's `.py`, never reimplements it: add the prereq's
  hyphenated dir to `sys.path`, import its underscored module as a module object (`import … as ak`),
  never `from … import`: harnesses collect `test_*` globals, so no helper may be named `test_*` either.
  Finance case studies import `notebooks/finance-corpus/finance_corpus.py`.
- `fc.generate_corpus(100, 1)` is a *fresh* generator run (Pitman–Yor restaurants restart from the
  same seed nouns), not a continuation; for "the same text continued", rebuild with `fc._CorpusGen`
  (see `continuation_run` in `notebooks/n-gram-models-smoothing/`). For a scale run, generate the
  longest continuation once and take prefixes; `ds.cooccurrence_of` counts new text on Topic 6's
  vocabulary (new types drop out, and windows close over them). Synthetic docs carry entity spans
  (`doc.entities`: ORG, GPE, DATE, MONEY, PERCENT, PERSON, PRODUCT, SECTOR; 1–5 companies/doc); EDGAR
  excerpts carry none; `fc._tokens` keeps `[a-z]+` only. For names a model never saw, train on
  `generate_corpus(…, roster=fc.split_roster()[0])` and test on `roster=…[1]`. A model trained on filings
  memorizes names and copies none it did not see (Topic 11: 0.000 on held-out names); in-context
  copying has to be trained on repeated random sequences. Polysemy: 21 company aliases double as
  products (gold `Entity.type`; `kestrel` 57 ORG / 61 PRODUCT), while `reserves` (banking/energy) occurs
  only 8 times. The aspect gold gives each direction word the news it reports ("costs rose" −1), but a
  lexicon word's own polarity is barely in the text (a whole document's other lexicon words vote it
  right 0.571 of the time), so no window or context model recovers it. MD&A document openers are
  templated (two distinct first sentences in `generate_corpus(100, 1)`): draw prompts from every
  sentence's distinct opening instead (Topic 14's 188).
- Lab data is `src/components/viz/data/<lab>.json` = `json.dumps(viz_payload(), separators=(',', ':'))`,
  pinned by the `.py`'s `test_viz_mirror_matches_payload` — regenerate it, never hand-edit.
- `rigorFlag` is load-bearing. Honesty is the differentiator.
- **`pnpm build` passing ≠ math correct.** KaTeX is non-strict: parse errors render as
  `.katex-error` spans and the build still exits 0. Open the page and assert zero `.katex-error`.
- MDX: a bare `<` is JSX — use `\lt`/`\gt` in math, backtick `<s>`/`</s>`/`<unk>`. Long inline
  number lists get one `$…$` per number (one KaTeX span can't wrap); at the mobile preset assert
  `innerWidth === 375`. Smart quotes turn an apostrophe right after `)` or `$…$` into an opening
  quote, a link's closing `)` included ("[Zipf topic](/topics/…)'s"): write "the score of head $h$" or
  "the event that $G$ measures", not "$h$'s score", and assert the built page's text has no `‘`. A shell
  grep for `\$'s` silently misses it (quoting): search the MDX with Python `re`.
- **Verify with DOM checks, not screenshots.** Viz uses `client:visible`: `scrollIntoView` and wait
  ~1–1.5 s before interacting. A hidden browser pane never hydrates any island (server-rendered
  readouts still look right) — show the preview pane first; `/paths`' graph is also
  `hidden md:block`, so use desktop width. Hydrated ⟺ the `astro-island` has lost its `ssr`
  attribute (`[data-lab]` is in the server-rendered HTML too). Scope DOM assertions to
  the lab container (pages carry other SVGs). React-controlled range sliders ignore synthesized
  `input` events — drive them with a real key press after `focus()`. At the mobile preset a scripted
  `scrollIntoView` may never trigger `client:visible`; a real `computer` scroll does. On a long page
  (Topic 16: 336 KaTeX spans) a scripted `scrollIntoView` can leave the page where it was, even at
  desktop width: `find` the lab's caption and `computer` `scroll_to` its ref, then check `ssr` is gone.
- Round every number a lab displays to the page's precision inside `viz_payload()`: `toFixed(3)` on a
  stored `0.9005` renders `0.900` while the page says `0.901`. Derived values too: a lab that summed
  two rounded shares showed 0.297 where the page's rounded sum said 0.296 — put the rounded sum in
  the payload. When the page quotes a number the lab shows, copy the payload's value
  (`round(0.2865, 3)` is `0.286`). When the page narrates a lab example, assert it in
  `test_viz_payload` and pick the lab's default by a rule that finds it, not a hard-coded index.
  `ak._sig(x, 3)` keeps significant figures (10.169 → 10.2): use `ak._r(x, 3)` for anything the page
  quotes to 3 decimals. Key payload dicts by `f"{beta:g}"`: `f"{1.0}"` is `"1.0"`, but JS's
  `` `${1}` `` is `"1"`. Whatever a lab plots that the `.py` computes goes in the payload too (Topic 13's
  IsoFLOP valleys and parabola coefficients): a lab may draw a curve from payload parameters, but every
  number it prints is a payload value.
- Score a mechanism by every position that satisfies its definition, not one exemplar. Topic 11's
  induction heads put 0.593 on the most recent copy (i − p + 1) but 0.943 on all earlier copies,
  since they split weight across copies one period apart. The lab's heatmap caught it, so build
  the views of a mechanism before freezing the page's numbers. Topic 12's filing windows likewise
  showed the encoder keeping 0.734 on a visible word where Theorem 2's optimum keeps 0.950.
- Probes find any cue. Read every probed token at one fixed window position (train on the
  concatenated stream so centered windows are in distribution): from document-internal windows, Topic
  12's context-free layer 0 typed `kestrel` at 0.907 through the position table, because product
  mentions close a document. Score a shuffled-label control against the shuffled labels too; scored on
  the real labels, a random-label probe on bimodal features lands anywhere (0.84, 0.33). A constant
  feature under k-fold CV scores below the majority, since each fold predicts its training majority.
  Where a page claims an optimum, check the trained model against a toy with exact posteriors (an HMM
  by forward–backward, with the corruption channel folded into the emissions).
- Check a one-pass/vectorized evaluator against the plain forward pass on full log-probs at every
  position (≤ 10⁻⁹), not argmaxes at a few: Topic 10's filing reader applied ALiBi's bias before 1/τ.
- Quote page numbers from the harness at ≥ 6 decimals, never from `summary()`'s 4-decimal print
  (0.9405 → 0.940, 0.7285 → 0.728).
- Mobile: assert `scrollWidth === 375` in *every* lab view/toggle. A `nowrap` readout inside a grid
  `minmax(0, 1fr)` cell overflows — use flex-wrap rows. To find the culprit, list elements with no
  overflow-clipped ancestor whose `getBoundingClientRect().right > 375`.
- D3 trees: `d3.stratify` silently drops a node whose id is `""` (falsy), so prefix every id
  (`n:${id}`); `d3.tree`'s default separation gives siblings half the gap of cousins, so labeled trees
  need `.separation(() => 1)` and about 26 px of height per leaf (Topic 14's 34-node tree takes 600).
- `TheoremBlock` titles carry no citations; attribute the result in the surrounding prose. Number each
  type separately (Theorems 1–3, Propositions 1–3, Corollaries 1–2), and use the same numbers in the
  brief, the `.py` docstrings, and the notebook: Topic 16's brief numbered in one sequence and had to be
  renumbered everywhere at Phase B.
- Markdown tables render `display: block; overflow-x: auto`, and `$…$` cells never wrap, so a wide table
  scrolls even at desktop width (the prose box is 704 px). Keep headers short and define them in the
  sentence above; put the long axis down the rows.
- Clip a lab chart whose curves blow up at a grid end (`clipPath`, and say "clipped at …" in its title):
  Topic 16's never-seen facts cost ~35 nats at τ = 0.04 and flattened the 0–3 nat range that matters.
- NumPy SGD: scatter-add minibatch gradients as `csc_matrix((ones, idx, arange(n+1)), shape=(V, n)) @ g`,
  >10× faster than `np.add.at`. Draw an epoch's random numbers in one call (same stream, same results).
  Time one epoch before fixing a grid. Profile a transformer step: `tb.mlp_backward`'s `einsum` was
  ~40% of it (Topic 12's matmul `mlp_backward` is pinned equal); a masked-LM loss should form logits
  only at the selected rows, and float32 training is ~2.5× faster (together, 50 → 11 ms/step at V = 879).
  Topic 10's `pe.rotate` writes into a fresh float64 `np.empty`, so RoPE silently upcasts float32
  attention and drops ndarray subclasses (Topic 13's FLOP counter missed QKᵀ); toy decoders use learned
  positions. Tiny decoders (d ≤ 48) are optimization-limited at Adam's lr 0.01: a peak of 1.28/d
  with 10% warmup took d = 32 from 2.14 to 1.70 nats and halved the seed spread (0.010 nats per run).
  Sweep the rate before fixing a scaling grid.
- Decoding on corpus models (Topic 14). Topic 3's KN trigram (`pce.word_vector("ikn")`) is Markov, so
  Dijkstra on its last-two-words states finds the exact mode in milliseconds; Topic 13's decoder
  recombines nothing (best-first tree search: median 25 pops, max 4,292). Mask `<unk>` (and the decoder's
  `<s>`) and renormalize per step, or `<unk>` becomes a frequent output, and say so in the rigorFlag.
  KN backoff hands `</s>` its unigram share after rare contexts, so 0.197 of the trigram's modes are
  empty. Push only a node's best child and next sibling (lazy successors), so the heap stays O(pops) at
  V = 3,427. Beam runs for several length penalties share one search, since all open prefixes have one
  length (48 → 31 s). Exact search under length normalization does not scale (per-length uniform-cost:
  88 s for 60 prompts, incomplete); report the length-capped certificate instead. For a
  sufficient-not-necessary check like Thm 3's certificate, pin where it stays silent; never assert it fires.
- Sampling on corpus models (Topic 15). Draw one uniform per open string per step and invert the CDF;
  restart the engine's stream for every setting (common random numbers), transform each trigram state
  once, and dedupe the decoder's prefixes within a step. `np.logaddexp.reduce` is slow on V = 3,427 rows:
  use a max-shifted logsumexp (with the dedupe and skipping `exact_modes("decoder")`, 45 → 29 s).
  Neither Topic 14 engine knows a KB fact (decoder P(correct sector) 0.205, chance; the trigram cannot
  see the company). An order-5 KN (`pce.kn_fit_any(train, 5, pce.word_alphabet())`, 0.3 s) does, but a
  fact seen once loses to the backoff's sector prior (discounted count ≈ 0.12 vs 0.28): split probes
  by the context's training count. A KN row's seen set is `vec._rows[u][0]`, exactly where
  P > freed(u)/c(u·)·P(·|u⁻); drop rows whose context is unseen (empty set, recall 0). For an oracle,
  fit KN to `ng.continuation_run(n)["continuation"]` (a fresh `generate_corpus` seed has other
  pseudo-words) with its singletons as `<unk>`; oracle NLL rewards blandness (the mode scores best). The
  decoder can score only held-out sentences inside its 509 words and 32 positions, a templated subset
  (NLL 0.445): give it no held-out reference. Grid τ as 10^{k/20} so τ = 1 is on it exactly. KN rows
  tie at the top, so test τ → 0 as uniform on the argmax set (near-ties need τ ≈ 10⁻⁶). Typical
  sampling at a small mass drops the head and its NLL rises: never assert it monotone. In leaf-frequency
  tests, take per-cell z-scores only where the expected count is ≥ 5.
- Calibration on corpus models (Topic 16). Top-label calibration needs only held-out text, so no oracle;
  canonical calibration over V = 3,427 can't be estimated (a forecaster with distinct rows is
  canonically calibrated iff it is the truth), so show it on finite worlds with exact level-set
  averages. Score every token with `<unk>` as an outcome: the trigram from `bs.trigram().vec` (not
  `TrigramLM`, which renormalizes `<unk>` away), the decoder from `sl.decoder_log_probs` with only `<s>`
  masked, sentences grouped by length (first 32 inputs). Dedupe trigram rows by context (23k tokens →
  5,462 rows) and sort each row once: the nucleus order is τ-invariant, so per-τ coverage is a cumsum.
  Three test sets: `ng.continuation_run(100)["continuation"]` is in distribution; `ng.corpus()["held"]`
  restarts the restaurants from the seed nouns (trigram `<unk>` share 0.007, *below* the continuation's
  0.028, and the decoder turns underconfident there); EDGAR is 830 trigram tokens, 58% `<unk>`. KN is
  nearly calibrated in distribution (τ* 1.003, ECE₁₅ 0.018 vs a calibrated twin's 0.006 ± 0.001):
  report ECE against the twin (same confidences, hits redrawn as Bernoulli(conf)), never alone. The
  NLL-optimal τ is not the ECE-optimal one (it raised in-distribution ECE on both engines, and on EDGAR
  flattened confidence to 0.008 below accuracy 0.059). Cross-fit any temperature gain (even/odd
  sentences). Topic 14's decoder has no temperature direction (parameter-free final LN, tied E at a fixed
  1/√d), so its training τ* is 0.943; to show MLE leaves τ*_train = 1, use a log-linear bigram, where
  d/dβ L_train = −λ‖W‖² exactly at an L2 optimum. Fit it with λ > 0: unpenalized, unseen pairs run to
  −∞ and L-BFGS takes ~30 s. Fact probes restricted to a relation's answers: seen facts are
  underconfident and unseen ones overconfident, so per-regime τ* sit at opposite grid edges; seed a
  bounded search from the grid argmin (a flat edge stops it short, 0.047 vs the 0.040 edge).
- Classification on corpus models (Topic 17). Fit logistic models by trust-region Newton–CG with exact
  Hessian-vector products (`lr_hessp`; the `hp` closures in `blr_fit`/`blr_intercept_fit`): 11–12
  iterations to gradient 10⁻¹¹ on the finance unigrams, where L-BFGS takes 1,000–1,500 and stops ABNORMAL
  at 10⁻⁸ (count features are ill-conditioned by their scale). Never refit through the representer dual
  (w = Xᵀa): it is worse conditioned. Cross-check with `--with scikit-learn==1.9.1`: `LogisticRegression`'s
  C is λ = 1/(Cn) with the intercept unpenalized, and its lbfgs stops near 10⁻⁵, so compare probabilities
  to 10⁻⁴ and assert our objective ≤ theirs. Tone: unigram NB 0.601 vs LR 0.739, and LR leads at every
  training prefix (30 to 1,500), so the corpus has no generative–discriminative crossover. Tone is an XOR
  of aspect valence × direction: unigram LR gives "rose"/"fell" ≈ 0 (0.060, 0.024) while the bigrams carry
  the sign ("costs rose" −0.355, "costs fell" +0.526). The full-text Bayes accuracy is 0.889919
  (`bayes_ceiling`, exact enumeration over k ∈ 5..8 aspect-sentence polarities).
  `generate_corpus(…, label_noise=q)` regenerates the identical text (~1.2 s per 1,500 documents): cache
  the tokens once and regenerate only the training labels (`fin_train_labels`). Bigram LR (V = 17,706 >
  n = 1,500) matches its training moments yet is overconfident on test (confidence 0.869 vs accuracy
  0.771): a nearly separable training set makes those moments optimistic. Simulated worlds: where NB is
  the true model, tuned NB stays at or above tuned LR (within 0.01) at every n; the textbook crossover
  needs NB misspecified and LR well specified, which burstiness gives (x = r ⊙ u; crossover n = 512 at
  V = 101, 2,048 at V = 3,071). The bursty world is a deterministic transform of the plain one's draws,
  so share their random streams (without common random numbers their Bayes accuracies differed). Tune
  each fit's hyperparameter on a fresh validation draw and say so in the rigorFlag, and fix the
  informative words' mass (m_S = 0.062) so the Bayes error does not move with V. Under a long Zipf tail,
  uniform add-α shrinks the informative words too, so the validated α grows with n (≈ 0.5 → 20 at
  V = 3,071).
- Prereq calls cost time: Topic 6's `ds.selected()` takes ~5.5 s and Topic 7's
  `train_sgns(CONTROLLED)` ~14 s; never call Topic 5's trained models (~50 s). Import its pure
  helpers (`complex_step_grad`, `log_softmax`) only. Topic 10's `train_head` takes ~1 s per head
  (`tournament()` trains 15, `filing_benchmark()` 12); import its pure helpers (`frequencies`,
  `sinusoidal`, `rotate`, `kernel`) instead. Topic 11's `train_model` takes ~7 s (`SYNTH2`), ~5 s
  (`SYNTH1`), and ~8 s (`FILING2`); its `mha_forward`/`mha_backward` (batched heads, RoPE, causal) are
  reusable, gradient-checked attention layers. Topic 12's `train_mlm` takes ~12 s (`HMM_MLM`) and ~22 s
  (`FILING_MLM`); its `encode`/`mlm_loss_grad` are a gradient-checked bidirectional pre-LN encoder
  (learned positions, tied embeddings). Topic 7's `SGNSConfig("L2", 25)` takes ~2.8 s. Topic 13's
  `train_run` takes 0.2–5 s per run, and its 20 runs (`isoflop_grid`, `seed_spread`, `repetition_runs`)
  ~31 s; its `decoder_loss_grad` is a gradient-checked causal pre-LN decoder (learned positions, tied
  embeddings), `_Counted` an ndarray subclass that counts matmul FLOPs, `sgd_losses` exact one-pass SGD
  on a Gaussian power-law model (no Monte Carlo), and `compute_optimal`/`fit_loss_law` Chinchilla's
  closed form and Approach 3. Its `init_decoder`/`decoder_loss_grad` hard-code V = `sl.TOY_V` (512) and
  n = `sl.TOY_CTX` (32), so a new vocabulary needs exactly 512 ids. Topic 14 trains it (d = 32, 1,200
  steps) in ~5 s on windows that start at a sentence's `<s>` (position 0 holds `<s>`, as when decoding)
  and decodes with float64 copies, so every search score matches one full forward pass to 10⁻⁹. Topic
  14's `uniform_cost`, `beam_search_multi` (with the pruned-score certificate), and `greedy_orbit` are
  reusable; its `_beam_table("decoder")` takes ~12 s and `exact_modes("decoder")` ~6 s. Topic 15's
  `draws(name, setting)` (1,504 strings) averages ~0.7 s, its 25-setting grid on both engines ~25 s,
  `oracle()` ~1.5 s, `fact_model()` 0.3 s; `truncation_set`, `temper`, `tree_dist`, and
  `local_tempered`/`global_tempered` are pure and reusable. Topic 16's `rows(engine, test)` takes ~2 s
  per trigram set (≈150 MB of float64 rows each; the decoder adds its ~5 s training), `stats_at` one
  temperature ~0.5 s on a trigram set, `loglinear_bigram` 0.5–3.3 s per λ (smaller λ is slower); `recalibrated`/`log_decomposition`,
  `reliability_bins`, `ece_hat`, `sq_ece_debiased`, `twin_ece`, `fit_beta` (Newton on the convex β
  objective), and `cross_fit_gain` are reusable. Topic 17's `learning_curve` takes ~22 s for the four
  worlds and its finance bigram softmax LR ~5 s (the whole harness ~42 s); `featurize`, `nb_fit`
  (multinomial/Bernoulli, Prop 1's closed form), `log_posterior`, `lr_fit`, `estimating_residuals`,
  `log_count_ratio`, `hard_margin_svm` (homogeneous dual), and `bayes_ceiling` are reusable.
- Cached trainers (`@lru_cache`) take an explicit frozen config: `f()` and `f(CFG)` are separate
  cache keys and train twice.
- Run the harness once with `python -W error`. Under macOS Accelerate, `np.linalg.det`/`slogdet` on
  large sparse block matrices raise spurious FP flags — check such determinants block by block.
- Test a published bound's stated constant on adversarial instances before citing it (Chen et al.
  2023's PI bound fails as stated for d ≤ 128). Recompute a paper's headline from its printed parameters:
  Hoffmann et al.'s rounded Eq. 10 gives 32B at Gopher's budget, not their 40B (the unrounded values,
  Besiroglu et al.'s Eq. 4, do). When a cited remark seems to contradict the harness, read its setting:
  Hutter (2021, §6)'s "β = 1" for Pitman–Yor is the d = 0 case, not the missing mass's 1 − d. Prototype
  trained experiments in the scratchpad before the scaffold, but treat spike numbers as provisional; never
  name a scratch script after a stdlib module (an `inspect.py` broke numpy's import). Read a cited
  theorem's own statement, not the prose around it: Meister et al. (2020)'s R_greedy squares the
  surprisal gap (Eq. 11), their text calls its k = 1 case "beam search", and their Appendix A proves Thm
  3.1 under Thm 3.2's name; Welleck et al. (2020)'s Thms 4.2–4.3 appear only in Appendix C. Huang, Zhao,
  and Ma (2017) is arXiv 1809.00069, not 1808.09582 (another paper); Karp (1978) has two Crossref DOIs
  (use 0012-365X(78)90011-0); Lowerre's 1976 thesis has no full text online (cite CMU CSD's degree record).
  Ng and Jordan (NIPS 14, MIT Press 2002, 841–48) number in one sequence (Prop 1, Prop 2 (VC side,
  m = Ω(n)), Lemma 3, Thm 4, Prop 5, Cor 6), and their formal h_Dis minimizes training 0-1 error, not the
  logistic MLE. Rosset, Zhu, and Hastie's NIPS 16 Thm 2.1 is the penalized λ → 0 path; their JMLR 2004
  §5 Thm 3 is the constrained path, and its appendix restarts the theorem numbers. Soudry et al. (JMLR
  2018)'s Thm 5 direction rate is O(log log t/log t), and O(1/log t) only for almost every dataset; with
  an intercept, GD finds argmin ‖w‖² + b². Wang and Manning (2012)'s w̄ = ‖w‖₁/|V| is a scalar (β = 0.25,
  α = 1, C = 1). Albert and Anderson (1984) sits behind a Cloudflare challenge and Xue and Titterington
  (2008) is paywalled: cite both from the abstract, and say so.
- Write excess losses near an optimum in `expm1`/`log1p` form: computed term by term, the Poisson
  excess at x = 250 gives −5·10⁻¹² where the exact value is 0.
- `jupyter execute` the `.ipynb` in place from the repo root (it writes no outputs); a copy elsewhere
  breaks cell 1's import. Check titles: arXiv's meta tags share one line, so extract
  `name="citation_title" content="([^"]*)"` with a regex (a plain grep prints the whole line);
  DOIs `api.crossref.org/works/<doi>`, sibling theorem numbers by grepping `<TheoremBlock … number=`.
  ACL Anthology pages write `<meta content="…" name=citation_title>` (content first, name unquoted);
  find a paper's anthology ID by grepping `aclanthology.org/volumes/<volume>/` for its title.
  `jupyter execute … | tail` hides the exit code; to read outputs run `jupyter nbconvert --to notebook
  --execute <nb> --output-dir <scratchpad>` from the repo root. Reddit, OpenReview, Semantic Scholar, and
  DBLP refuse fetches, and ar5iv times out. Read a theorem's exact statement with `pdftotext` on the
  arXiv PDF, downloaded into its own scratchpad directory. Verify venues via the citing paper's
  bibliography (`pypdf` on its arXiv PDF),
  `proceedings.iclr.cc/paper_files/paper/<year>` (lists 2024, not 2022), or arXiv's Comments field.
- Deeper, topic-specific lessons from 50 shipped topics (numerical traps, nbformat generation,
  hydration quirks) live in `../formalRAG/CLAUDE.md` — read the relevant section rather than
  re-learning it. formalRAG is **read-only** from this repo.

## Do NOT

- Flip `PRELAUNCH` to `false` before all 29 nodes are published. Launch is its own plan.
- Touch sibling repos (formalML, formalStatistics, formalCalculus, formalRAG) before launch —
  reciprocal links are batched into one PR per sibling at launch.
- Commit the Loughran–McDonald dictionary, earnings-call transcripts, or any non-public text.
- Cover ANN indexes, vector databases, quantization, or retrieval models (TF-IDF, BM25, dense
  retrieval, rerankers) — formalRAG owns them; cross-link instead.
- Cover agentic topics (MCP, tool calling, workflows, agent security) — reserved for formalAgentic.
- Cite a sibling slug without checking it exists **and** is `status: published` on the sibling's
  GitHub `main` — a local sibling checkout can lag or sit on another branch. Check what it covers, too:
  formalML `message-passing` is GNNs, with no max-product or Viterbi, and as of 2026-10-24 no published
  sibling covers Viterbi (formalStatistics `hidden-markov-models` is still draft). The design's
  watch-list is not complete: grep each sibling's `main` for the topic's key terms (`git grep -l -i`
  over `src/content/topics`). Topic 16 found formalML `uncertainty-quantification` already covering
  binary calibration, ECE, strict properness, Murphy's decomposition, and temperature scaling, so the
  page cites those and proves only what they leave open. Topic 17 found formalML `naive-bayes` covering
  the NB variants, Laplace smoothing as a Dirichlet posterior, and Ng–Jordan for Gaussian NB only (§7);
  its §6.2 conjugacy proof block closes after one sentence, so cite formalStatistics
  `multivariate-distributions` Thm 6 for the proof.
- Hardcode hex colors in components — use the `--color-*` tokens.
- Commit `docs/plans/audit-output/`, `cross-site-audit-report.md`, or `deferred-reciprocals.md`.
