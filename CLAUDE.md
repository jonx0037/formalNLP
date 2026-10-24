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
pnpm test                 # tsx --test tests/*.test.ts — domains, curriculum DAG, branding,
                          #   WCAG contrast, robots/noindex, cross-site parsing
pnpm build                # Production build + Pagefind (8 GB heap baked in; minify off on Vercel)
pnpm exec astro sync      # Validate schema + ALL frontmatter fast
pnpm validate             # validateConnections.ts — roadmap nodes without MDX are NOTICES
pnpm audit:cross-site     # Five-site reciprocity validator (siblings adjacent or FORMAL_*_PATH)
# Python pillar (per-topic, no shared venv):
uv run --with numpy --with scipy python notebooks/<slug>/<slug_underscored>.py
uv run --with numpy --with scipy --with jupyter --with ipykernel jupyter execute notebooks/<slug>/01_<slug_underscored>.ipynb
```

## Single sources of truth

- `src/data/domains.ts` — the 9 domains, their labels, and the 4 sibling sites. The content
  schema, `curriculum.ts`, the graph color scale, and the tests all import it.
- `src/data/curriculum-graph.json` — all 29 nodes (planned → published) and the 42-edge
  prerequisite DAG. `src/data/curriculum.ts` mirrors the still-planned titles per track.
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
  hyphenated dir to `sys.path`, import its underscored module. Finance case studies import
  `notebooks/finance-corpus/finance_corpus.py`.
- `fc.generate_corpus(100, 1)` is a *fresh* generator run (Pitman–Yor restaurants restart from the
  same seed nouns), not a continuation; for "the same text continued", rebuild with `fc._CorpusGen`
  (see `continuation_run` in `notebooks/n-gram-models-smoothing/`).
- Lab data is `src/components/viz/data/<lab>.json` = `json.dumps(viz_payload(), separators=(',', ':'))`,
  pinned by the `.py`'s `test_viz_mirror_matches_payload` — regenerate it, never hand-edit.
- `rigorFlag` is load-bearing. Honesty is the differentiator.
- **`pnpm build` passing ≠ math correct.** KaTeX is non-strict: parse errors render as
  `.katex-error` spans and the build still exits 0. Open the page and assert zero `.katex-error`.
- MDX: a bare `<` is JSX — use `\lt`/`\gt` in math, backtick `<s>`/`</s>`/`<unk>`. Long inline
  number lists get one `$…$` per number (one KaTeX span can't wrap); at the mobile preset assert
  `innerWidth === 375`.
- **Verify with DOM checks, not screenshots.** Viz uses `client:visible`: `scrollIntoView` and wait
  ~1–1.5 s before interacting. A hidden browser pane never hydrates any island (server-rendered
  readouts still look right) — show the preview pane first; `/paths`' graph is also
  `hidden md:block`, so use desktop width. Scope DOM assertions to
  the lab container (pages carry other SVGs). React-controlled range sliders ignore synthesized
  `input` events — drive them with a real key press after `focus()`.
- Round every number a lab displays to the page's precision inside `viz_payload()`: `toFixed(3)` on a
  stored `0.9005` renders `0.900` while the page says `0.901`.
- Quote page numbers from the harness at ≥ 6 decimals, never from `summary()`'s 4-decimal print
  (0.9405 → 0.940, 0.7285 → 0.728).
- Mobile: assert `scrollWidth === 375` in *every* lab view/toggle. A `nowrap` readout inside a grid
  `minmax(0, 1fr)` cell overflows — use flex-wrap rows.
- `TheoremBlock` titles carry no citations; attribute the result in the surrounding prose.
- NumPy SGD: scatter-add minibatch gradients as `csc_matrix((ones, idx, arange(n+1)), shape=(V, n)) @ g`,
  >10× faster than `np.add.at`. Draw an epoch's random numbers in one call (same stream, same results).
  Time one epoch before fixing a grid.
- Prereq calls cost time: Topic 6's `ds.selected()` takes ~5.5 s; never call Topic 5's trained models
  (~50 s). Import its pure helpers (`complex_step_grad`, `log_softmax`) only.
- `jupyter execute` the `.ipynb` in place from the repo root (it writes no outputs); a copy elsewhere
  breaks cell 1's import. Check titles: arXiv `curl -sL arxiv.org/abs/<id> | grep citation_title`,
  DOIs `api.crossref.org/works/<doi>`, sibling theorem numbers by grepping `<TheoremBlock … number=`.
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
