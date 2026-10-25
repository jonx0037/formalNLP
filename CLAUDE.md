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
  right 0.571 of the time), so no window or context model recovers it.
- Lab data is `src/components/viz/data/<lab>.json` = `json.dumps(viz_payload(), separators=(',', ':'))`,
  pinned by the `.py`'s `test_viz_mirror_matches_payload` — regenerate it, never hand-edit.
- `rigorFlag` is load-bearing. Honesty is the differentiator.
- **`pnpm build` passing ≠ math correct.** KaTeX is non-strict: parse errors render as
  `.katex-error` spans and the build still exits 0. Open the page and assert zero `.katex-error`.
- MDX: a bare `<` is JSX — use `\lt`/`\gt` in math, backtick `<s>`/`</s>`/`<unk>`. Long inline
  number lists get one `$…$` per number (one KaTeX span can't wrap); at the mobile preset assert
  `innerWidth === 375`. Smart quotes turn an apostrophe right after `)` or `$…$` into an opening
  quote: write "the score of head $h$", not "$h$'s score", and assert the built page's text has no `‘`.
- **Verify with DOM checks, not screenshots.** Viz uses `client:visible`: `scrollIntoView` and wait
  ~1–1.5 s before interacting. A hidden browser pane never hydrates any island (server-rendered
  readouts still look right) — show the preview pane first; `/paths`' graph is also
  `hidden md:block`, so use desktop width. Hydrated ⟺ the `astro-island` has lost its `ssr`
  attribute (`[data-lab]` is in the server-rendered HTML too). Scope DOM assertions to
  the lab container (pages carry other SVGs). React-controlled range sliders ignore synthesized
  `input` events — drive them with a real key press after `focus()`.
- Round every number a lab displays to the page's precision inside `viz_payload()`: `toFixed(3)` on a
  stored `0.9005` renders `0.900` while the page says `0.901`. Derived values too: a lab that summed
  two rounded shares showed 0.297 where the page's rounded sum said 0.296 — put the rounded sum in
  the payload. When the page quotes a number the lab shows, copy the payload's value
  (`round(0.2865, 3)` is `0.286`). When the page narrates a lab example, assert it in
  `test_viz_payload` and pick the lab's default by a rule that finds it, not a hard-coded index.
  `ak._sig(x, 3)` keeps significant figures (10.169 → 10.2): use `ak._r(x, 3)` for anything the page
  quotes to 3 decimals. Key payload dicts by `f"{beta:g}"`: `f"{1.0}"` is `"1.0"`, but JS's
  `` `${1}` `` is `"1"`.
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
- `TheoremBlock` titles carry no citations; attribute the result in the surrounding prose.
- NumPy SGD: scatter-add minibatch gradients as `csc_matrix((ones, idx, arange(n+1)), shape=(V, n)) @ g`,
  >10× faster than `np.add.at`. Draw an epoch's random numbers in one call (same stream, same results).
  Time one epoch before fixing a grid. Profile a transformer step: `tb.mlp_backward`'s `einsum` was
  ~40% of it (Topic 12's matmul `mlp_backward` is pinned equal); a masked-LM loss should form logits
  only at the selected rows, and float32 training is ~2.5× faster (together, 50 → 11 ms/step at V = 879).
- Prereq calls cost time: Topic 6's `ds.selected()` takes ~5.5 s and Topic 7's
  `train_sgns(CONTROLLED)` ~14 s; never call Topic 5's trained models (~50 s). Import its pure
  helpers (`complex_step_grad`, `log_softmax`) only. Topic 10's `train_head` takes ~1 s per head
  (`tournament()` trains 15, `filing_benchmark()` 12); import its pure helpers (`frequencies`,
  `sinusoidal`, `rotate`, `kernel`) instead. Topic 11's `train_model` takes ~7 s (`SYNTH2`), ~5 s
  (`SYNTH1`), and ~8 s (`FILING2`); its `mha_forward`/`mha_backward` (batched heads, RoPE, causal) are
  reusable, gradient-checked attention layers. Topic 12's `train_mlm` takes ~12 s (`HMM_MLM`) and ~22 s
  (`FILING_MLM`); its `encode`/`mlm_loss_grad` are a gradient-checked bidirectional pre-LN encoder
  (learned positions, tied embeddings). Topic 7's `SGNSConfig("L2", 25)` takes ~2.8 s.
- Cached trainers (`@lru_cache`) take an explicit frozen config: `f()` and `f(CFG)` are separate
  cache keys and train twice.
- Run the harness once with `python -W error`. Under macOS Accelerate, `np.linalg.det`/`slogdet` on
  large sparse block matrices raise spurious FP flags — check such determinants block by block.
- Test a published bound's stated constant on adversarial instances before citing it (Chen et al.
  2023's PI bound fails as stated for d ≤ 128). Prototype trained experiments in the scratchpad before
  the scaffold, but treat spike numbers as provisional.
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
  GitHub `main` — a local sibling checkout can lag or sit on another branch.
- Hardcode hex colors in components — use the `--color-*` tokens.
- Commit `docs/plans/audit-output/`, `cross-site-audit-report.md`, or `deferred-reciprocals.md`.
