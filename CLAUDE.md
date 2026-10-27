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
  number lists get one `$…$` per number (one KaTeX span can't wrap), and a long inline formula goes to display math
  (Topic 22's proof of Thm 1 widened the page to 483 px); at the mobile preset assert `innerWidth === 375`. Smart quotes turn an apostrophe right after `)` or `$…$` into an opening
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
  At the mobile preset a stale `find` ref can `scroll_to` thousands of pixels off (Topic 20: the lab ended at
  top = −7,119): `window.scrollBy(0, lab.getBoundingClientRect().top − 300)` by script, then one real `computer`
  scroll fires `client:visible`.
- Round every number a lab displays to the page's precision inside `viz_payload()`: `toFixed(3)` on a
  stored `0.9005` renders `0.900` while the page says `0.901`. Derived values too: a lab that summed
  two rounded shares showed 0.297 where the page's rounded sum said 0.296 — put the rounded sum in
  the payload. When the page quotes a number the lab shows, copy the payload's value
  (`round(0.2865, 3)` is `0.286`). When the page narrates a lab example, assert it in
  `test_viz_payload` and pick the lab's default by a rule that finds it, not a hard-coded index.
  `ak._sig(x, 3)` keeps significant figures (10.169 → 10.2): use `ak._r(x, 3)` for anything the page
  quotes to 3 decimals. Values spanning decades go through `_sig`: `_r(1.26e-4, 4)` is `0.0001`, which a lab's
  `toFixed(3)` shows as `0.000` (Topic 20's SGD first steps, Adam's 5.2·10⁻⁵ NTK loss). `ltc._r` recurses into lists
  but raises on a dict: round nested payload dicts field by field. Key payload dicts by `f"{beta:g}"`: `f"{1.0}"` is `"1.0"`, but JS's
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
  renumbered everywhere at Phase B. Fix the brief's numbers in the order the page will present them: Topic 21's draft
  page put decoding risk (Thm 4) second, and the section was moved rather than renumbering three files.
  `type="algorithm"` is a numbered block too.
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
- Lexicons on corpus models (Topic 18). Net tone with tuned cutpoints scores 0.637, but the two category counts
  (plug-in category-world Bayes, or LR on the two rates) score 0.724: neutral filings have few polar words of either
  kind, and positive filings still carry negative hits (u₊ 0.967, u₋ 0.191), so the pos − neg projection, not the word
  list, costs most of the gap to unigram LR (0.739). Uncertainty words have the same rate in every tone, and litigious
  words occur only in 8-Ks (adding both to a rate model changes nothing). The i.i.d.-token model's exact sign error
  (0.253 at L̄ = 210) matches the measured 0.252. With the general list, b > a even in positive filings (no Chernoff
  bound). A dictionary-centered ridge prior is `ltc.lr_fit` on features with the dictionary counts Aᵀx appended and a
  per-feature λ vector (λ on words, η = 10⁻⁴ on the count columns); AᵀΔ = 0 at the optimum. Harmonic induction: use
  the full document-PPMI graph (kNN sparsification drops sign accuracy to 0.48–0.74), restrict the solve to the seeds'
  connected components (PPMI-isolated words make L_UU singular), and compute z without the seeds, whose ±1 dwarfs every
  propagated value (seed z ≈ 246). Signs come out 0.893 right, but polysemes are not neutralized and the extremes are
  sector words. A dollar amount in MDX prose (`$\$12.5$`) desyncs every later `$…$` pairing check: avoid it.
- Aspect sentiment on corpus models (Topic 19). One row per gold aspect (14,179 in the test split); every two-aspect
  sentence (", but") pairs opposite polarities, so any aspect-blind reader is capped at 1 − m/2N = 0.911 (a blind
  bigram LR lands on exactly 0.500 of contrast aspects), and polarity is a deterministic function of clause and
  category (aspect-level Bayes accuracy 1): results are about representation. Keep a "shared votes" attention rung's
  bias shared too: a per-category bias shifts every vote by an aspect-dependent constant and lets signs flip. Write
  the layer ragged (`np.maximum.reduceat`/`np.add.reduceat` per row, `np.bincount` scatters) with values in logit space
  (z ∈ ℝ³): 300 full-batch Adam steps on 14k aspects take 2–6 s, where a padded d = 16 bilinear version took 25–165 s.
  Contrast sets: never fill a slot with an out-of-vocabulary word ("also" became `<unk>` and broke even the bigram
  rung); edit within the vocabulary or delete a word (removing "not" keeps the quadruple, since a substitution's sign
  does not depend on its offset under a separable kernel). Thm 2's reweighting z′ = (z + b)/r − b at r = 10⁶ leaves a
  logit-ratio spread of ~3·10⁻⁸ from cancellation (test at 10⁻⁶); the trained layer gives aspect words only 0.005 of its
  attention, so moving it there needs r ≈ 10⁶, and the probabilities flatten to 1/3 (a temperature). Tone-bearing
  sentences are those opening on an aspect (or "Management pointed to") that name no MONEY, PRODUCT, or ORG entity:
  exactly the generator's k ∈ 5..8, and their first-aspect polarities through the Bayes rule give 0.883 against the
  0.890 ceiling. The EDGAR excerpts hold five aspect-term mentions; split them at "." only (";" leaves fragments).
- Fine-tuning on corpus models (Topic 20). Topic 12's filing encoder has V = 879 (`[MASK]` = 879 is input-only, so E
  has 880 rows); 512 is Topic 13's `sl.TOY_V`. Topic 12 has no backward pass from dL/d LN(x_L): use Topic 20's
  `encoder_backward` (lifted from `mlm_loss_grad`, pinned equal), and deep-copy `train_mlm`'s cached P/blocks before
  training them. At a zero-initialized head every encoder gradient is 0, so first-step analyses and linearizations start
  from the probe's head (LP-FT). Document tone is a poor fine-tuning testbed: mean-pooled ~7 windows per filing, 1,500
  labels, and full fine-tuning reaches 0.559 against bag-of-words LR 0.745, with a never-pretrained encoder at 0.536.
  Topic 19's aspect rows separate the methods (probe 0.761, random-init 0.951, pretrained 0.970–0.989); keep every polar
  row and 1,500 neutral ones (a 5k random subsample drops negated adjectives to 0.5), and read the mean final state over
  the aspect span. The fine-tuned encoder reads "not ADJ" but calls Topic 19's deleted-"not" quadruple copies neutral
  (training shows "T was ADJ this period" only with "not"); 310 counterfactual rows fix every seed. Seed spread is large
  (full fine-tuning 0.970–0.989), so headline comparisons need 3 seeds. LoRA on Wq/Wv at Adam's 10⁻² forgets more than
  full fine-tuning (‖sAB‖ ≈ 45 against full's 2.7 on the same matrices); every LP-FT run Pareto-dominates every LoRA
  run, while the weaker "LoRA forgets more than every full run" fails by 0.03. Full fine-tuning's ΔW is nearly low-rank
  (rank-4 truncation keeps 0.977 of 0.989) where LoRA trained at r = 4 reaches only 0.95. Under Adam, α/√r holds the
  first step level in r (sign step, bounded by σ(A₀)); under SGD only the mean is level, and a single draw sits
  √((k+1)/r) away. No count sketch with D ≤ 4,096 reaches d₉₀. For the NTK, per-example Jacobians come from single-row
  backward passes; take the training rows at one length, so each full-batch step is one batch. At the stable rate
  η = n/λ_max, GD barely trains in 300 steps (loss 0.620 → 0.599), while Adam memorizes 120 rows and leaves the
  kernel (K moves 169%). Topic 20's harness runs ≈ 2¼ min: the README's 60 s rule was waived for that topic, by
  Jonathan (and later for Topic 22).
- Sequence labeling on corpus models (Topic 21). Topic 12's HMM is nearly deterministic (each state owns three tokens:
  Bayes tagging accuracy 0.995, flat learning curves); blend its emissions toward uniform (ε = 0.3 → Bayes 0.782). A
  world whose word comes from the *next* state (ρ = 0.5) misspecifies the HMM but keeps the Bayes tagger a chain: fold
  each word's emission into the edge it straddles. A prefix chain must marginalize the unseen next state out of its
  last word ((1 − ρ)B + ρ·TB); truncating the full chain silently drops x_{e−1} (lag-0 Bayes read 0.425, truly 0.743).
  CRF fitting: exact Hessian-vector products by complex step need a log-sum-exp shifted by the *real* part's max; at
  n = 1,000 near-separable sequences trust-ncg takes ~150 Hv (8 s), while L-BFGS with `ftol = 0, gtol = 10⁻⁶` lands
  within 2.4·10⁻⁵ in 2 s with identical decodes (pin the gap in a test). Validation-best λ tracked 1/n at every n, and
  λ = 1/n on the mean objective is MAP under N(0, I): fit once, untuned. Add-1 cost the finance HMM 2–5 points (a
  uniform Dirichlet over ~500 words), so tune α (closed form, free). The indicator CRF beat the HMM on conditional
  log-loss but not accuracy in the misspecified world: report log-loss whenever a claim is about conditional MLE.
  Prop 2 (an MEMM's marginals ignore words past its lookahead) binds posterior decoding only: a causal MEMM decoded by
  Viterbi beat the lag-0 Bayes ceiling (0.758 vs 0.743), since future factors sum to one but their max varies. A CRF's
  revision decays with distance (the last word moved z₁ by 3·10⁻⁵). Finance BIO: entity spans are character offsets,
  token index = `len(fc._tokens(text[s:start]))`; no span crosses a sentence or cuts a token, sentences are ≤ 31
  tokens, `million`/`percent` occur only inside entities (MONEY/PERCENT trivial), and at 300 training documents the
  perceptron tags test text perfectly, so the informative regime is 3–30 documents, where nearly every error is an
  unseen name (#22's subject). The templates let the left context settle "the" (causal MEMM 0.997), so show label bias
  on in-vocabulary counterfactual continuations instead. A unit-norm teacher over ~30 weights gives margins of
  0.03–0.3, and rejection sampling for a larger margin hangs (one teacher stream had none above 0.06): enumerate all
  V^T inputs once and draw from those above the threshold. Pick a lab's example path with a unique mode: the first
  candidate had two Viterbi paths tied to 2.5·10⁻¹⁷. Store max-marginals relative to the mode (the mode's probability,
  1.1·10⁻⁴, rounds to 0.000).
- Named entities on corpus models (Topic 22). `fc.split_roster()` holds out one company per sector (with subsidiaries,
  officers, flagship products): train on the first 30 docs of `generate_corpus(300, 0, roster=train)`, test on
  `(200, 1)` with each roster. Every held-out ORG/PERSON/PRODUCT mention has an unseen word, but so do 214 of 3,747
  seen-roster spans at 30 docs (rarely covered companies, cities), so report F1 on unseen *mentions* (filter gold and
  predicted spans by their own words) beside the two tests. The word-feature CRF memorizes names: held-out ORG F1 is
  exactly 0, even in "X reported revenue of". Name dropout must *delete* a dropped token's word/affix features and
  every neighbor's feature naming it: substituting a shared `<unk>` makes it a feature seen only on entities, whose O
  weight is negative at the L2 optimum (Prop 5: nλW[f,k] = Σ(1[z=k] − γ(k)), and Σ_k W[f,k] = 0), so precision
  collapses (0.48) on every unseen content word; deletion raised both tests (seen 0.945 → 0.975, unseen 0.642 →
  0.890). At p = 1 deletion also hides closed-class entity words (million, percent, quarter) and falls back. Topic 12's
  encoder was pretrained on the full roster (35 of 49 held-out name words are in its vocabulary); the encoder cannot
  delete, so dropped tokens enter as [UNK], and that pays only if unseen words are read as [UNK] at test too ("train as
  you test": 0.884 by id, 0.979 as [UNK]); without dropout the [UNK] reading lowers the score. Read each sentence alone
  with a trailing [SEP] (sentences are ≤ 31 tokens). `fc._tokens` lowercases: take case from the raw text with
  `re.findall(r"[A-Za-z]+", text[s:e])`, which aligns with it token for token; one "cap" feature reaches 0.920 on the
  held-out test. On frozen states a softmax head decodes invalid BIO in 10–15% of sentences and trails the CRF by
  0.09–0.11 even constrained; BIOES gained nothing (features see the neighbors). Overlapping spans are exclusive
  events (q_a + q_b ≤ 1), so a threshold τ ≥ 1/2 needs no interval program; skipping it for disjoint candidates cut
  the threshold sweeps from 19.5 to 6 s. Theorem 3's F*/2 is optimal for the model's beliefs: where the realized F1
  peaks (above 1/2 for the overconfident `<unk>` tagger, 0.38 vs F*/2 = 0.458 for the headline) measures
  miscalibration. A softmax head by Topic 17's `lr_fit` (trust-ncg, gtol 10⁻¹⁰) on 6k dense rows took 37 s; L-BFGS
  on `ltc.lr_objective` with dense X takes ~1 s. Topic 22's harness runs ≈ 82 s: the 60 s rule was waived for it by
  Jonathan, as for Topic 20. In a TSX block comment, "F*/2" closes the comment (`*/`) and tsc reports a parse error.
- Relation extraction on corpus models (Topic 23). Candidates are ordered pairs of linked mentions (kb_id set) in one
  sentence; REPORTED/GUIDED (MONEY/PERCENT tails) are document events, not KB facts. Gold is single-label per ordered
  pair and always a KB fact, so distant supervision (KB(h, t)) never misses a stated relation; its precision is
  `fc.distant_supervision_precision` (0.690608 on `generate_corpus(300, 0, roster=train)`), split into co-mention 338,
  entailed 137 (BASED_IN from "headquartered in Toronto, Canada", ACQUIRED ⇒ PARENT_OF: close the sentence's stated
  facts under the KB's rules), and overlap 29. Conjoin the between string with the argument order: unconjoined, gold
  F1 is 0.952 (which argument is the subsidiary is a template × order XOR); conjoined, gold-trained entity-blind LR is
  perfect (1.000 on both tests), so RE results on this corpus measure supervision, not reading. DS learns each
  co-mention template's KB-related rate (0.529–0.685; the generator's 0.6 is split by template, and the reversed order
  of a stating template has ρ = 0, so keep templates whose candidates are all related/unrelated co-mentions with ρ > 0)
  and labels a relation only where ρπ > 1 − ρ (2 of 11). Noisy-OR MIL: compute log(1 − p_i(r)) as a log-sum-exp over
  the other labels minus the full one (complex-step safe; never form 1 − p) and sum per bag with `np.add.reduceat` on
  instances sorted by (h, t); L-BFGS from zero converges in ~80 iterations (2 s). It zeroes co-mention FPs but recall
  falls (0.711) on templates no bag forces (context class = types, order, between string, right word). Aggregate
  facts by max for sentence-trained models: noisy-OR overcounts a pair's repeated co-mentions (DS fact P 0.719 vs
  0.892; held-out 0.331 vs 0.701). Hide KB facts by unordered entity pair (all of a pair's facts together), or the
  c-scaling of the DS target fails on multi-relation pairs (PARENT_OF + ACQUIRED); the fitted mass lands just below c
  (L2), raw F1 *rises* at c = 0.8 (co-mention templates cross the threshold first), and Elkan–Noto's e1 runs low
  (0.910 at c = 1). Topic 22's `split(test)` sentences align token for token with `fc` sentences in `sl.bio_sentences`
  order, and exact string linking (`fc._tokens` of names and aliases, type checked) reproduces every gold link; with
  the headline tagger's spans the triple-recall bound is 0.730 seen (rarely covered training-roster names) vs 0.920
  held out. Argument words or encoder entity-start states let DS memorize the KB (related co-mentions 0.813 seen vs
  0.304 unseen). An unbreakable inline-code module name in Implementation widened the mobile page to 395 px: name long
  modules in prose.
- Knowledge-graph embeddings on corpus models (Topic 24). `fc.WORLD` is 114 entities, 10 relations, 254 triples; every
  model scores a query as one point against every entity (distance γ − d for TransE/RotatE, inner product for
  DistMult/ComplEx; head queries use the inverse map), so one backward pass serves all four, and training uses every
  corruption (the exact expectation of uniform negative sampling, deterministic given the init). Skip the loss value
  during training steps (two `log_expit` passes over the score matrix were most of a step: 1.0 → 0.3 s per 1,000).
  Hidden facts are training negatives: on the full KB RotatE's phases solve Thm 1 to 3 decimals (COMPETES_WITH at
  {0, π}, median 0.000), but with one direction of each competitor pair hidden they move off (0.485) and held-out
  symmetry MRR is the worst of four (0.272), while DistMult, symmetric by construction, scores 1.000. An exact rotation
  scores a pair and its reversal identically, so only phases off {0, π} can push the hidden twin down. DistMult's
  f(h,r,t) and f(t,r,h) differ in the last bits (computed in different orders): direction accuracy read 0.514 until a
  10⁻⁹ relative tie tolerance made it 0.5. Tune on another hiding averaged over 3 seeds (one seed picked RotatE γ = 3
  at 500 steps, composition 0.731; three picked γ = 6 at 1,500, 0.912), read step counts off one run's checkpoints,
  and expect picks at a grid edge (every step pick sat at an end of 250/500/1,000/1,500). COMPETES_WITH is N-to-N with
  8 triangles (degrees 1 × 6, 2 × 10, 3 × 8), so Thm 2(c)'s odd-cycle collapse binds. "Inheritance" (subsidiaries share
  the parent's sector) is no `closure` rule; TransE wins it (0.815) by N-to-1 collapse. TransE's best S₃ run in the plane
  put all six entities at one point (ratio exactly 1). Halvorsen Industrial is the only parent with two subsidiaries
  (Northgate Energy has one). At τ = 0.5, `extracted_kb()` leaves 28 WORLD facts about the 20 held-out entities missing
  (26 COMPETES_WITH, 2 SUBSIDIARY_OF whose PARENT_OF was extracted at 0.688/0.685); every held-out entity has an extracted
  fact. `tests/scaffold.test.ts` rejects the word "architectures" anywhere in a topic, references included. Topic 24's
  harness runs 60–63 s (Topic 23's `extracted_kb()` is 25 s, training over 108 runs ~30 s): the 60 s rule was waived
  for it by Jonathan, as for Topics 20 and 22.
- Topic models on corpus models (Topic 25). Rebuild `fc._CorpusGen(0, None, fc.HEAPS_DISCOUNT)` for 300 + 100 documents
  (training = `ds._generator_run()`'s text, held-out = `ng.continuation_run(100)`, both asserted) and read each word's
  restaurant off the generator after all 400; words with one home agree with `ds.planted_labels()`. The restaurant
  vocabulary is 2,908 words, 13,382 training tokens (3,735 held out); industrials covers 142 of 300 documents. Every
  document draws its nouns ½ from its sector and ½ from the shared restaurant (`SECTOR_SHARE`), so R = E[θθᵀ] is rank 6
  and the shared vocabulary is unidentifiable (Prop 4): score against the six ψ_s (each sector's documents' word
  distribution), never the seven restaurants. Planted words have one home, so the vocabulary is separable, and
  FastAnchorWords finds one anchor per sector and never a shared word. Deterministic optimizers stall here: CAVI from
  scikit-learn's Gamma(100, 0.01) noise reaches sector NMI 0.186 (ELBO* −93,556) where the same updates warm-started from
  Gibbs's counts reach 0.963 (−88,822), and MoU EM's best of 20 random starts has log-likelihood −80,675 vs −77,168 from
  the gold sectors; diagnose a stall by warm-starting from another engine's answer and comparing the objective. The
  E-step schedule changes which optimum CAVI reaches (a cap of 30 local steps moved finance NMI 0.16 → 0.40): use
  scikit-learn's (per document, mean |Δγ| < 10⁻³, ≤ 100 steps) and vectorize it across documents with sparse scatter
  matrices (exact block steps, so Cor 2 holds; it reproduces the per-document loop exactly). Check the closed-form bound
  against `sk._approx_bound(X, sk.transform(X, normalize=False), False)` with `sk.components_` as λ (10⁻⁸ relative).
  Collapsed Gibbs in pure Python costs 0.03–0.07 s per sweep on all 63k tokens: no vectorized substitute needed. The
  prior ranks the engines (dense Dir(𝟏) world: Gibbs best at β = 1, CAVI at β = 0.01). Anchor-recovery worlds need dense
  topics (sparse Dir(0.1) topics are already nearly separable, so a "no anchors" twin recovers as well) and an M-scaled
  candidate threshold (words in ≥ 5% of documents; a fixed df ≥ 10 lets noisy rare rows win the farthest-point
  search). An LP hull distance in V = 2,908 dimensions costs ~55 ms: read distances off RecoverL2's residuals. Over 128
  enumerated states a sampler's total variation has a Monte Carlo floor ≈ 0.02 at 40k sweeps: test per-token
  marginals. Thm 4(c)'s divergence test needs a document with one topic's anchors only (anchors of both make the
  integral over [ε, 1] itself infinite), integrated over geometric segments (`quad` warns otherwise, fatal under
  `-W error`). On all words LDA keys on company names (sector NMI 0.943) and never finds tone or document type (≤ 0.023).
- Collocations on corpus models (Topic 26). Provenance comes free: subclass `fc._CorpusGen`, override `word`/`phrase` to
  wrap each restaurant draw in sentinels (`\x01…\x02`, `\x03` between a compound's halves) and make the same RNG calls;
  the stripped text equals `generate_corpus(300, 0)` byte for byte (assert it), so every token is a draw (D), entity (E),
  or template (L). `fc._tokens` drops commas, so list neighbors ("freight, contracts") are adjacent draws too, independent
  given the sector like compounds. Within-sentence bigrams: N = 57,052 positions, 13,724 types (877 fixed, 3,054 null,
  1,000 topical, 8,793 boundary); "percent in" is boundary because "percent" closes a PERCENT entity. Score rankings with
  tie-averaged precision (expected under random tie order): PMI ties its top 97 at log N and X² ties every perfect pair
  (196) at N, so a stable sort's alphabetical order gave X² a fake P@10 of 1.0. `hypergeom.sf` underflows to 0 for 46
  pairs: rank Fisher by a log-sum-exp tail. Compute G² as `xlogy(o, o/e)`: `xlogy(o, o) − xlogy(o, e)` cancels N log N
  terms (~10⁻⁸ off at N = 10⁷). Test selection inside the draw–draw stratum (N = 4,182, 2,128,632 candidate pairs, 97.5%
  seen once): the pooled table mixes position heterogeneity with the selection. BH over observed pairs rejects 2,218 of
  3,055 true nulls; over all candidate pairs (unseen p = 1) none, and no topical pair either (no power at one sighting).
  Exact sizes need the dense integer grid (r = c = 1…450 at N = 10⁴): a 25-point geomspace put G²'s peak at 0.131, the
  dense grid at 0.157 (E = 1.90); X² peaks at 0.162 (E = 0.176). Boundaries like "P(reject | seen) = 1 up to r = 14" come
  from the harness, never from a prototype's coarse grid (the brief first said r = 10). The notebook must use the
  harness's random streams (`thm2_draws()`), or its printed medians differ from the page. Keyness with entity words
  removed scores P@25 0.80–1.00 (names would win trivially); energy's misses are its own template ("thousand barrels per
  day"). TextRank's Spearman with frequency is 0.80–0.85 and rises with d; a geometry preset with small margins puts every
  table in one corner of the tetrahedron, so lead with balanced ones ((100, 40, 50)). Contraction tests need a 10⁻¹³
  additive floor (‖xₜ − π‖ hits rounding at t ≈ 50, d = 0.5).
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
  `log_count_ratio`, `hard_margin_svm` (homogeneous dual), and `bayes_ceiling` are reusable. Topic 18's
  `learning_curves` takes ~15 s (the harness ~30 s); `tune_thresholds` (exhaustive cutpoints), `category_counts`,
  `harmonic`, `prior_design`/`prior_fit`, `sign_error` (exact trinomial tail), and `load_lm_dictionary` are reusable.
  Topic 19's harness takes ~27 s (three attention rungs ~12 s); `aspect_rows`, `batch`/`layer_forward`/`layer_loss_grad`
  (complex-step safe), `valence_patterns`/`negation_feasible` (LP certificates), `reweighted`, `tone_sentences`, and
  `bayes_tone` are reusable. Topic 11's `mha_forward`'s broadcast `(B, 1, n, d) @ (1, H, d, dh)` products were ~40% of a
  fine-tuning step: Topic 20's `mha_forward`/`mha_backward`/`encode` do each projection as one flattened matmul, are pinned
  equal to Topics 11–12 at 10⁻¹¹, and cut a run from 4.2 to 2.4 s. Topic 20's `fine_tune(cfg)` takes ~1.8 s, plus ~0.6 s
  for `evaluate`; its 31 runs, `ntk_experiment` (~10 s), and `doc_fine_tune` (~5 s each) make up the ≈ 2¼ min harness.
  `encoder_backward`, `aspect_loss_grad`, `lora_grads`, `first_step`, `balance`/`gd_drift`, `sketch_matrix`,
  `kernel_gd`, `tone_from`, and `doc_windows` are reusable. Topic 21's harness takes ~45 s (`world_curves` ~10 s per
  world, `memm_table` ~4 s, `fin_models` ~13 s for 3/10/30 documents); `chain_forward`/`chain_backward` (either
  semiring), `chain_marginals`, `max_marginals`, `viterbi`, `path_score`, `brute_force`, `hmm_fit` (with an end state),
  `make_data` (length buckets), `crf_objective`/`crf_hessp` (complex-step safe), `crf_fit_lbfgs`, `memm_fit` (Topic
  17's `lr_fit` on token rows), `perceptron` (averaged), `hmm_from_crf`, `bio_sentences`, `spans`/`span_f1`, and
  `world_chain`/`lagged_bayes` are reusable. Topic 22's harness takes ≈ 82 s (Topic 12's pretraining, 21 s, runs
  inside `ner.encoder()` and is not cached across processes; 18 CRF fits ≈ 42 s); `tagset`/`encode`/`decode`/`masks`/
  `is_valid` (BIO and BIOES), `constrain` (Thm 1's −∞ masks), `span_marginals`, `interval_dp`, `f1_decode`
  (Dinkelbach), `prf` (four match schemes)/`micro_mixture`, `build` (Topic 21's `Data` from dense or sparse rows),
  `states` (encoder states per sentence), `ner_sentences` (BIO plus case), and `word_features` (substitute or delete
  dropout) are reusable. Topic 23's harness takes ≈ 33 s (Topic 22's headline tagger, pretraining included, is 27 s of
  it, inside `predicted_corpus`); `re_sentences`/`corpus`/`candidates`, `ds_labels`, `closure`, `ds_error_sources`,
  `lexical_features`, `soft_objective`/`fit_soft` (soft-label softmax LR), `mil_objective`/`mil_fit` (noisy-OR,
  complex-step safe), `witness_weights`, `identifiability`, `facts` (max or noisy-OR)/`fact_scores`, `kept_kb`
  (pair-level SCAR), `link`, `predicted_corpus`, `pipeline`, and `extracted_kb` (Topic 24's input) are reusable. Topic 24's
  `fit(cfg, kg)` takes 0.2–0.4 s per 1,000 full-batch steps on the finance KB; `queries`/`query_vjp`/`score_matrix`/
  `loss_grad` (four models, one backward pass), `realistic_rank` (Def 2), `evaluate` (batched filtered ranks),
  `direction_accuracy`, `phase_residuals`, `abelian_ratio`, `s3_kg`/`dihedral_fit`, `pattern_split`, and
  `completion_problem` are reusable.
  Topic 25's harness takes ≈ 45 s (one restaurant-vocabulary Gibbs run ~2.3 s, the all-words run at 150 sweeps ~5 s,
  each CAVI run ~1 s on finance and ~10 s on the V = 400 world); `gibbs` (pure-Python collapsed sampler, `visit` hook),
  `cavi`/`elbo`/`elbo_star`, `fold_in`, `completion` (plug-in document completion), `mixture_of_unigrams`/`mou_posterior`,
  `cooccurrence`/`normalize_rows`/`fast_anchor_words`/`recover_l2`, `hull_distance` (LP), `align_topics` (Hungarian
  on L1), `toy_marginals`, `nmi`, `umass_coherence`/`npmi_coherence`, and `fin_data` are reusable; `topic_mixtures(docs)`
  is the capstone's topic stage (six sector-aligned proportions, held-out sector accuracy 0.930).
  Topic 26's harness takes ≈ 3 s; `Table`/`scores` (vectorized PMI, t, X², G²), `log_p_greater` (exact log tails),
  `hypergeom_pmf`, `fisher_p` (scipy's two-sided rule), `exact_size`, `selection`/`selection_curve`, `keyness`,
  `rayson_garside_ll`, `cooc_graph`, `pagerank` (any teleport vector), `google_matrix`/`pagerank_dense`,
  `strength_identity`, `_TaggedGen`/`tagged_documents` (provenance-labeled corpus), `fin_bigrams`, and
  `tie_blocks`/`expected_topk` (tie-averaged precision) are reusable.
- Cached trainers (`@lru_cache`) take an explicit frozen config: `f()` and `f(CFG)` are separate
  cache keys and train twice, and so are `probe("pretrained")` and `probe("pretrained", False)` (Topic 20 refit a
  probe for 4 s): give cached helpers no defaults and call them positionally. Never seed a stream from `hash()` of a
  tuple with strings: `PYTHONHASHSEED` randomizes it per process, so runs stop being reproducible. Derive streams from
  fixed integers, and share the batch-order stream across methods per seed (common random numbers).
- Run the harness once with `python -W error`. Under macOS Accelerate, `np.linalg.det`/`slogdet` on
  large sparse block matrices raise spurious FP flags — check such determinants block by block.
  Keep `_run_harness` and the `__main__` block last in the `.py`: Topic 21 appended sections after them, and an edit
  that truncated the file at `def _run_harness` deleted those sections silently (the next run failed on a NameError).
  Scripted edits: assert the replaced text is non-empty and occurs once. In Topic 24 an `old = s[s.index(a):s.index(b)]`
  came out empty because `b` also occurred earlier (inside `S3_SEEDS`), and `s.replace("", new)` inserted `new` between
  every character (a 42 MB `.py`; stripping every copy of `new` recovered it).
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
  Loughran and McDonald (2011)'s Crossref abstract has "almost three-fourths" (Harvard-IV negatives in 10-Ks); the
  73.8% in the body is unverified. Their Master Dictionary CSV is a Google Drive file linked from
  sraf.nd.edu/loughranmcdonald-master-dictionary (free for academic research; a category column holds the year added,
  negative if later removed). formalStatistics `large-deviations` Thm 6 is Cramér; formalML `concentration-inequalities`
  Thm 3 is Chernoff.
  Kobayashi et al. (2020)'s f(x) is the value transform followed by W^O: say "transformed vectors", not "value
  vectors". MAMS (Jiang et al. 2019) defines its sentences as having at least two aspects of different polarity. FiQA's
  ACM PDF (Maia et al. 2018) returns 403; confirm Task 1 (aspect-based financial sentiment) on sites.google.com/view/fiqa.
  Topic 20 citation facts:
  - **Kalajdzievski (2023), rsLoRA.** Thm 3.2 is proved for **SGD** only (App. A), at any fixed step count as r → ∞
    with d fixed, and singles out no first step. An Adam statement is ours to prove.
  - **Hu et al. (2022), LoRA.** §4.1 says A is Gaussian, but the loralib code uses Kaiming-uniform A. Malladi et al.
    (2023) swap the A/B roles; their Thm 7.2 is the w.h.p. LoRA-kernel result.
  - **Du, Hu, and Lee (2018).** Thm 2.2 is the linear-layer balancedness. Claim B.1 (App. B) is the exact one-step
    change: η²(VᵀRᵀRV − UᵀRᵀRU).
  - **Kumar et al. (2022).** Lemma A.3 says features never move on S⊥. Lemma A.4 is balancedness for (v, B). Prop 3.7
    is LP-FT on perfect features, and Thm 3.3 the OOD bound.
  - **Aghajanyan et al. (2021).** The ACL Anthology author order is Aghajanyan, Gupta, Zettlemoyer; arXiv's differs.
  - **Li, Grandvalet, and Davoine (2018), L2-SP.** Eq. 3 also penalizes the new head (β).
  - **Davidson and Szarek (2001).** The handbook chapter is paywalled, so its theorem number cannot be verified. Cite
    Vershynin (2012), Thm 5.32 and Cor 5.35, for a Gaussian matrix's √N ± √n singular values.
  - **Jacot et al. (2018).** The NTK title carries the subtitle "Convergence and Generalization in Neural Networks".
  Topic 21 citation facts:
  - **Smith and Johnson (2007).** Cor 2 (HMM conditionals = chain-MRF conditionals) is for Mealy features, by Chi's
    method; Lemma 1 and Thm 2 give C_MEMM ⊊ C_HMM. The Moore-feature converse (an indicator CRF is some HMM's
    conditional) is Topic 21's own h-transform proof: h = (cI − M)⁻¹e^η for c > ρ(M), and it needs an end state.
  - **Sutton and McCallum (2012).** §2.3 proves HMM → CRF and calls the converse "more complicated"; §5.1.3 is MEMMs and
    label bias, with the caveat that copying later words into the features buys them back.
  - **Collins (2002), W02-1001.** Thm 1 is the separable bound R²/δ²; Thm 2 the non-separable one; §2.5 averaging.
  - **McCallum, Freitag, and Pereira (2000).** No DOI, and ACM's page returns 403: cite the author's copy,
    people.cs.umass.edu/~mccallum/papers/memm-icml2000.ps (pages 591–98 from Sutton and McCallum's bibliography).
  - **Klein and Manning (2002)** is W02-1002. formalStatistics `exponential-families` Thm 1 is one-parameter only.
  Topic 22 citation facts:
  - **Lipton, Elkan, and Naryanaswamy (2014).** The F*/2 threshold is **Corollary 1** of Thm 1 (calibrated scores, F1 as a
    ratio of population rates); expected F1 is their Thm 2. LNCS 8725 (ECML PKDD Part II), 225–39. arXiv 1402.1892v2 has
    another title ("Thresholding Classifiers to Maximize F1 Score") and spells the third author "Narayanaswamy".
  - **Dinkelbach (1967).** INFORMS returns 403: cite from the Crossref abstract, which says the paper restates and proves
    Jagannathan's (1966) fractional–parametric theorem and gives the iteration.
  - **Ye et al. (2012).** EUM–DTA asymptotic equivalence is Thm 8 (§2.3); the PDF header says "Optimizing F-Measures",
    arXiv's metadata "F-measure".
  - **Lample et al. (2016), N16-1030.** §4.1 replaces singletons by UNK with probability 0.5; §2.4 found IOBES *no*
    significant gain over IOB. Ratinov and Roth (2009, W09-1119, §4, Table 2) found BILOU better on 3 of 4 sets.
  - **SemEval-2013 Task 9 (S13-2056), §3.1** lists strict, exact-boundary, partial, and type matching but defines no
    partial credit (it defers to MUC, Chinchor and Sundheim 1993).
  - **Augenstein et al. (2017).** The seen/unseen F1 gap is in the body (arXiv 1701.02877, §3.3, Table 13), not the
    abstract. BERT's NER setup (no CRF, first sub-token) is §5.3; Collobert et al.'s sentence-level likelihood §3.4.2.
  Topic 23 citation facts:
  - **Mintz et al. (2009), P09-1113.** §5.1's lexical feature is one conjunction (between words, order, window, types);
    features are aggregated per entity pair (a pair-level classifier); §6 negatives are a random 1% of unrelated pairs;
    §7.1 says held-out evaluation "suffers from false negatives", not that it underestimates precision.
  - **Riedel, Yao, and McCallum (2010).** §1 says DS is violated ~13% (Wikipedia) and 31% (NYT), but Table 1's Wikipedia
    rows average 16.7%: quote the 31% and the table. Author copy at riedelcastro.github.io.
  - **Elkan and Noto (2008).** Lemma 1 is p(y = 1 | x) = p(s = 1 | x)/c; e1 is recommended and called exact "if
    g(x) = p(s = 1|x)"; the proof that g = c on P quietly sets p(y = 1 | x) = 1 there (never says "separable").
  - **Fisher's identity.** Cite Dempster, Laird, and Rubin (1977, pp. 1–22, §3, Lemma 2 with Eq. 3.2; unnamed there);
    Louis (1982) is unreadable (OUP 403, JSTOR gated). Dietterich et al. (1997) say examples and instances, never bags.
  - **Hoffmann et al. (2011), P11-1055.** 18.3% of matched Freebase facts overlap (§1); Φ_join is a deterministic OR
    (§3.2). Zeng et al. (2015, D15-1203, §3.5) train each bag on its most probable instance. Baldini Soares et al.'s
    entity-start state (P19-1279, §3.3) needs entity markers; the Anthology spells "Fitzgerald", the PDF "FitzGerald".
  Topic 24 citation facts:
  - **Sun et al. (2019), RotatE.** Lemmas 1–3 only say RotatE "can infer" each pattern (one-direction proofs, App. B–D);
    the iff forms (rᵢ = ±1, r̄, r₁ ∘ r₂) are claims in §1. Footnote 2: the distance is L1 over complex moduli. Footnote 6
    swaps the 1-to-N/N-to-1 labels relative to Bordes and Wang: do not copy it. No 1-to-N limitation is stated.
  - **Bordes et al. (2013).** No DOI (cite papers.nips.cc); pages 2787–95 come from citing bibliographies. §2: entities
    renormalized every iteration because otherwise the loss is minimized by inflating norms; relations once at init.
    §4.2 introduces "raw" vs "filtered"; §4.3 the 1.5 threshold. Wang et al. (2014, TransH) say "reflexive" for symmetric.
  - **Yang et al. (2015), DistMult** never mention symmetry; Kazemi and Poole (2018, §2, §5.1) assert it in one sentence
    and credit TransE's non-full-expressiveness to Wang, Gemulla, and Li (2018). Trouillon et al. (2017, JMLR 18 (130))
    add Dance as second author; expressiveness is Thm 2 + Cor 1, Cor 3 (sign-rank), Thm 4 (multi-relational).
  - **Cai (2019)** is arXiv 1909.05100 v1, withdrawn in v2 (no venue): cite v1 and say so. Rotate3D's authors are Gao,
    Sun, Shan, Lin, Wang (ACM 403: cite the abstract). Berrendorf et al.'s WI-IAT 2020 paper (371–74) is retitled on arXiv
    (2002.06914, realistic rank §2.2 Eq. 3). Sun et al. (2020, ACL) has five authors ("Partha" is Talukdar's first name).
  Topic 25 citation facts:
  - **Blei, Ng, and Jordan (2003).** The variational updates are Eqs. (6)–(7), with the digamma form a separate Eq. (8),
    derived in App. A.3; App. A.1's E[log θ] is an unnumbered display. §5.4 (Fig. 7) is smoothed LDA; Fig. 4 (§4.4) the
    topic simplex in the word simplex; §4.3 pLSI. JMLR, no DOI.
  - **Griffiths and Steyvers (2004).** Eq. 5 keeps the document denominator n_d − 1 + Tα (T topics, W words).
  - **Arora, Ge, and Moitra (2012).** "Anchor word" is §1 prose; the numbered notion is Def. 1.1 (p-separable); Thm 1.4
    needs a lower bound on R's ℓ1 condition number (Def. 2.1), never the word "nonsingular". Arora et al. (2013):
    FastAnchorWords is Alg. 4, RecoverKL Alg. 3, RecoverL2 unnumbered (§3 prose); the convex-combination form is the
    unnumbered text after Eqs. (1)–(2); the unbiased Q estimator is App. D.1 of arXiv 1212.4777.
  - **Newton and Raftery (1994)** are pp. 3–26 (3–48 includes the discussion); the body is paywalled. Neal's discussion
    never mentions infinite variance; his 2008 blog post ("Worst Monte Carlo Method Ever", radfordneal.wordpress.com)
    does. Wallach et al. (2009): HM §4.2 Eq. 14, "wildly overestimated" §6.2, Chib-style Alg. 2, left-to-right Alg. 3.
  - **Gaussier and Goutte (2005)** and Hofmann's SIGIR paper: ACM returns 403 (metadata only). Hofmann's UAI paper
    (arXiv 1301.6705) §3.3, Fig. 2 has the sub-simplex and the KL reading for the joint table. Chang et al. (2009): use
    the PDF's author order (the NeurIPS site differs). Bybee et al.'s JF 2024 article says "a topic model"; LDA is
    confirmed only in NBER w26648 (2020, "The Structure of Economic News").
  Topic 26 citation facts:
  - **Dunning (1993), J93-1003.** Equations are unnumbered (cite sections): §5.2 λ and its χ² limit, without citing
    Wilks; §5.3 binomial/multinomial forms; §6.1 the 31,777-word bigram study (Tables 2–4) scores only observed bigrams and
    uses the statistic "as a measure"; no multiple comparisons; §7 suggests "Fischer's exact method" (sic).
  - **Moore (2004), W04-3243.** Bilingual Hansard pairs, not bigrams. §2.1 counts singleton pairs against 391,236,048
    possible ones (782.5 expected by chance): credit him for testing over all pairs. §3 Formula 5: G² = 2N·MI. §6 Table 1:
    LLR overstates significance for rare events; §7 recommends Fisher.
  - **Rayson and Garside (2000), W00-0901** (doi 10.3115/1117729.1117730): LL sums the word's row only (2 cells, ≤ G²),
    and they compare no critical value (3.84/6.63 come from elsewhere). Church and Hanks (1990, §4): ordered pairs in a
    window of 5, f(x, y) ≤ 5 set aside. Evert's thesis (elib handle 11682/2573) is behind bot protection; §4.3.3 excludes
    f < 3. Manning and Schütze ch. 5 has no DOI (cite nlp.stanford.edu/fsnlp).
  - **Fisher's exact test** is Fisher (1935, JRSS 98 (1), Example 1, pp. 48–50, doi 10.2307/2342435); Fisher (1922) is
    only degrees of freedom. SMRW 5th ed. §21.02 is unverified.
  - **Mihalcea and Tarau (2004), W04-3252.** The Anthology title says "Text", the PDF "Texts". §2's update is
    unnormalized, (1 − d) + d Σ (n·π); keywords: undirected, unweighted, nouns and adjectives, window 2 best (Table 1),
    threshold 10⁻⁴. **Boudin (2013) is I13-1102** (I13-1062 is TopicRank); his §5 reading of Grolmusz is too broad:
    PageRank ∝ degree only with degree-proportional teleportation (Grolmusz 2015, IPL 115: 633–34,
    doi 10.1016/j.ipl.2015.02.015). Haveliwala and Kamvar (2003): Thm 1 |λ₂| ≤ c, Thm 2 equality; ilpubs.stanford.edu's
    database is down, so read it and Page et al. (1999) via the Wayback Machine.
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
  formalML `message-passing` is GNNs, with no max-product or Viterbi. formalStatistics `hidden-markov-models`
  (published 2026-10-25) proves the forward recursion, forward–backward, Viterbi, and Baum–Welch (Thms 5–10) and names
  CRFs only in Rem 23; no sibling covers CRFs, MEMMs, or the structured perceptron (2026-10-26). No sibling covers NER, BIO/BIOES, span F1, or F1-optimal
  thresholding either; formalRAG `set-metrics-precision-recall-map-mrr` (set precision and recall, Def 1) and formalML
  `uncertainty-quantification` (perfect calibration, Def 3) are the nearest (2026-10-26). No sibling covers distant
  supervision, multiple-instance learning, positive-unlabeled learning, noisy-OR, or relation extraction (2026-10-26):
  formalML `kl-divergence` Prop 2 is Gibbs' inequality, formalStatistics `hidden-markov-models` Rem 16 states Fisher's
  identity (its `maximum-likelihood` only previews EM, Rem 10, and `model-selection-and-information-criteria`'s
  "Fisher identity" is J = K), formalML `convex-analysis` has no unique-minimizer theorem, and formalRAG
  `graphrag-community-detection` builds its graph from co-occurrence and cosine, not extracted relations. No sibling
  covers TransE, RotatE, DistMult, ComplEx, link prediction, Hits@k, or filtered ranking (2026-10-26): formalRAG
  `set-metrics-precision-recall-map-mrr` Def 3 is MRR, formalML `tensor-decompositions` Def 4 is CP, formalCalculus
  `eigenvalues-eigenvectors` Rem 2 reads complex eigenvalues as rotations, and no sibling states Euler's formula as a
  result; formalML `smooth-manifolds` covers matrix Lie groups only in prose (no numbered block). No sibling covers
  pLSI, NMF, anchor words, LDA's own updates, or topic-model evaluation (2026-10-26): formalStatistics
  `bayesian-computation-and-mcmc` Rem 10 names collapsed Gibbs for LDA without deriving it, formalML
  `variational-inference` cites LDA only as a reference, and `multivariate-distributions` Thms 5–6 are the Dirichlet
  moments and conjugacy. No sibling covers collocation scores, keyness, or TextRank (2026-10-26):
  formalML `random-walks` has PageRank only as a remark (Perron–Frobenius cited, no rate proved); formalStatistics
  `likelihood-ratio-tests-and-np` proves scalar Wilks (Thm 4) and only states the vector case (Rem 13),
  `hypothesis-testing` Def 12 states Pearson's limit, and Fisher's test is only `discrete-distributions` Example 6.
  The design's watch-list is not complete: grep each sibling's `main` for the topic's key terms (`git grep -l -i`
  over `src/content/topics`). Topic 16 found formalML `uncertainty-quantification` already covering
  binary calibration, ECE, strict properness, Murphy's decomposition, and temperature scaling, so the
  page cites those and proves only what they leave open. Topic 17 found formalML `naive-bayes` covering
  the NB variants, Laplace smoothing as a Dirichlet posterior, and Ng–Jordan for Gaussian NB only (§7);
  its §6.2 conjugacy proof block closes after one sentence, so cite formalStatistics
  `multivariate-distributions` Thm 6 for the proof.
- Hardcode hex colors in components — use the `--color-*` tokens.
- Commit `docs/plans/audit-output/`, `cross-site-audit-report.md`, or `deferred-reciprocals.md`.
