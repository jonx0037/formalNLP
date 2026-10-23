# formalNLP Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up formalnlp.com as a live, `noindex`, zero-topic site that is scaffolded from formalRAG, rebranded as "Marginalia," carries the full 29-node planned curriculum graph, and ships a shared synthetic finance corpus ready for the first topic.

**Architecture:** Copy formalRAG's Astro site shell (UI components, layouts, pages, graph libs, scripts), strip everything RAG-specific, and re-key it on a single `DOMAINS` constant that the schema, curriculum, color scale, and tests all import. Pure logic (domains, robots, cross-site field parsing, color contrast) is extracted into small modules with `tsx --test` / `node --test` unit tests; site-level behavior is verified with `astro sync` → `validate` → `build` → DOM checks. The Python finance corpus follows formalRAG's notebook contract (an importable `.py` whose `assert` harness is the test).

**Tech Stack:** Astro 6, React 19, Tailwind 4, MDX, KaTeX, D3 7, Pagefind, `@fontsource-variable` fonts, `tsx --test`, Python via `uv`, Vercel.

**Spec:** [`docs/plans/formalnlp-site-design.md`](formalnlp-site-design.md) (approved 2026-10-22). Read it before any task.

**Scope:** This plan covers *scaffold → first noindex deploy*. The 28 topics + capstone are each authored afterward through `STARTER-PROMPT.md` (Phase A → Phase B); their order is fixed in the **Topic Roadmap** at the end, not planned task-by-task here.

## Global Constraints

- `pnpm` only (never npm); Python via `uv run --with …`; Node `>=22.12.0`.
- American English in all copy and comments; Chicago 17th (Notes & Bibliography) for citations.
- Git: work on feature branches → PR → merge to `main`; Vercel deploys from `main`. No worktrees.
- Repo `jonx0037/formalNLP` is **public**. Never commit the Loughran–McDonald dictionary or earnings-call transcripts.
- The site stays `noindex` (meta robots + `robots.txt` `Disallow: /`) until all 28 topics + capstone are published.
- No changes to sibling repos (formalML, formalStatistics, formalCalculus, formalRAG) in this plan.
- The `formalRAG` source tree at `../formalRAG` is **read-only**; copy from it, never edit it.
- Notebooks: CPU-only, deterministic, < 60 s, no network at run time except explicitly documented fetches.
- Light-theme Marginalia tokens are fixed by the spec (§5); dark-theme values are this plan's call (Task 4).

## Review Focus

1. **Zero published topics.** Every page (`/`, `/topics`, `/paths`, `/about`) must build and render with an empty `topics` collection. A fresh site with only planned nodes is the launch state for weeks. → test in Task 1, Step 6 and Task 3, Step 6.
2. **Leftover formalRAG identity.** A stray "formalRAG," "retrieval-augmented," GitHub link, or `formalrag.com` canonical in shipped HTML would mislead visitors. → `tests/branding.test.ts` in Task 4.
3. **Noindex leaking off.** If the prelaunch flag is false or the meta tag misses a layout, search engines index a half-built site. → `tests/robots.test.ts` + `dist/` grep in Task 5.
4. **Dark mode illegible.** The oxblood/gold accents on a dark background can fail contrast. → WCAG AA contrast test in Task 4.
5. **Cross-site parsing of the fifth site.** Sibling repos will later carry `formalnlp*` fields and formalNLP carries `formalrag*` fields; an audit that silently ignores either looks green while missing edges. → `tests/crossSite.test.ts` in Task 6.

---

## File Structure

```
formalNLP/
├── .gitignore                      # formalRAG's + .superpowers/
├── package.json  astro.config.mjs  tsconfig.json  vercel.json
├── CLAUDE.md  STARTER-PROMPT.md  README.md
├── docs/plans/                     # design doc, this plan, per-topic briefs later
├── public/favicon.svg
├── scripts/
│   ├── audit-cross-site-links.mjs  # 5-site rewrite (Task 6)
│   └── lib/crossSite.mjs           # pure site/field config + parsing (Task 6)
├── src/
│   ├── content.config.ts           # schema imports DOMAINS (Task 2)
│   ├── content/topics/.gitkeep     # empty at launch
│   ├── data/
│   │   ├── domains.ts              # single source of truth for the 9 domains (Task 2)
│   │   ├── curriculum.ts           # tracks/layers (Task 3)
│   │   └── curriculum-graph.json   # 29 planned nodes + edges (Task 3)
│   ├── lib/
│   │   ├── prerequisiteGraph.ts  validateConnections.ts  synthesizeCrossSiteNodes.ts
│   │   ├── robots.ts               # robotsTxt() (Task 5)
│   │   ├── siteConfig.ts           # SITE constants incl. PRELAUNCH (Task 5)
│   │   └── contrast.ts             # WCAG contrast ratio (Task 4)
│   ├── components/ui/*             # formalRAG's minus PipelineDiagram
│   ├── components/viz/{CurriculumGraph,DAGGraph}.tsx + shared/
│   ├── layouts/{BaseLayout,TopicLayout}.astro
│   ├── pages/{index,about,paths}.astro, topics/{index,[...slug]}.astro, robots.txt.ts
│   └── styles/global.css           # Marginalia tokens (Task 4)
├── tests/*.test.ts                 # tsx --test
└── notebooks/
    ├── README.md
    └── finance-corpus/             # shared generator (Task 8)
        ├── finance_corpus.py
        ├── 01_finance_corpus.ipynb
        └── edgar/{SOURCES.md, *.txt}
```

---

### Task 1: Repository and stripped scaffold

**Files:**
- Create: `.gitignore`, `README.md`, `package.json`, `astro.config.mjs`, `tsconfig.json`, `vercel.json`, `src/content/topics/.gitkeep`
- Copy from `../formalRAG`: `src/components/ui/*` (except `PipelineDiagram.astro`), `src/components/viz/CurriculumGraph.tsx`, `src/components/viz/DAGGraph.tsx`, `src/components/viz/shared/*`, `src/layouts/*`, `src/pages/{index,about,paths}.astro`, `src/pages/topics/*`, `src/lib/*`, `src/styles/global.css`, `src/content.config.ts`, `src/data/curriculum.ts`, `public/favicon.svg`, `scripts/audit-cross-site-links.mjs`
- Test: `tests/scaffold.test.ts`

**Interfaces:**
- Produces: a building Astro site with an empty `topics` collection; `pnpm test` script (`tsx --test tests/*.test.ts`) used by every later task.

- [ ] **Step 1: Initialize git on `main` with the docs and push**

```bash
cd ~/Developer/Sites/formalNLP
git init -b main
# .gitignore = formalRAG's .gitignore + a line ".superpowers/"
git add .gitignore docs/
git commit -m "docs: formalNLP site design and scaffold plan"
gh repo create jonx0037/formalNLP --public --source . --push
git checkout -b feat/site-scaffold
```

Expected: `gh repo view jonx0037/formalNLP --json visibility` → `PUBLIC`. (Creating the public repo is outward-facing: confirm with Jonathan in chat before running `gh repo create`.)

- [ ] **Step 2: Copy the files listed above; do not copy topics, notebooks, `architectures.ts`, `pipeline.ts`, `/architectures`, or any `*Laboratory.tsx`**

- [ ] **Step 3: Adapt config**
  - `package.json`: `"name": "formalnlp"`; keep formalRAG's dependencies/devDependencies; add script `"test": "tsx --test tests/*.test.ts"`.
  - `astro.config.mjs`: `site: 'https://formalnlp.com'`; otherwise identical (keep the Vercel `minify` guard).
  - Delete every reference to `PipelineDiagram`, `pipelineStage`, `modality`, and `architectures` from the copied pages, layouts, and components.

- [ ] **Step 4: Write the failing test `tests/scaffold.test.ts`**

```ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { execGrep } from './helpers';

test('no RAG-only artifacts were copied', () => {
  for (const p of ['src/components/ui/PipelineDiagram.astro', 'src/pages/architectures.astro',
                   'src/data/architectures.ts', 'src/data/pipeline.ts']) {
    assert.equal(existsSync(p), false, p);
  }
  const labs = readdirSync('src/components/viz').filter((f) => f.endsWith('Laboratory.tsx'));
  assert.deepEqual(labs, []);
});

test('no copied source references pipelineStage or modality', () => {
  const hits = execGrep(/pipelineStage|modality|PipelineDiagram/, 'src');
  assert.deepEqual(hits, []);
});

test('astro site points at formalnlp.com', () => {
  assert.match(readFileSync('astro.config.mjs', 'utf8'), /site: 'https:\/\/formalnlp\.com'/);
});
```

`execGrep(re: RegExp, dir: string): string[]` is a small helper in `tests/helpers.ts` that walks `dir` recursively and returns `path:line` for each match. Later tasks reuse it.

- [ ] **Step 5: Run `pnpm install && pnpm test`.** Expected: PASS once Steps 2–3 are complete. Fix any failures by editing the copied files, not the test.

- [ ] **Step 6: Verify the zero-topic build**

Run: `pnpm exec astro sync && pnpm build`
Expected: exit 0; `dist/index.html`, `dist/topics/index.html`, `dist/paths/index.html`, `dist/about/index.html` all exist. (At this point copy still says formalRAG and the schema still has RAG domains. Tasks 2–4 fix that.)

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: scaffold formalNLP from formalRAG shell"
```

---

### Task 2: Domain taxonomy and content schema

**Files:**
- Create: `src/data/domains.ts`
- Modify: `src/content.config.ts`
- Test: `tests/domains.test.ts`

**Interfaces:**
- Produces: `export const DOMAINS = [...] as const; export type Domain = typeof DOMAINS[number]; export const DOMAIN_LABELS: Record<Domain, string>;` in `src/data/domains.ts`. Also `export const SIBLING_SITES = ['formalml', 'formalcalculus', 'formalstatistics', 'formalrag'] as const;` in the same file.

- [ ] **Step 1: Write the failing test**

```ts
import { DOMAINS, DOMAIN_LABELS, SIBLING_SITES } from '../src/data/domains';
test('nine domains in spec order', () => {
  assert.deepEqual([...DOMAINS], [
    'text-as-data', 'language-modeling', 'representations', 'attention-transformers',
    'decoding-generation', 'classification-sentiment', 'extraction-knowledge-graphs',
    'text-mining', 'nlp-evaluation',
  ]);
  assert.equal(DOMAIN_LABELS['attention-transformers'], 'Attention & Transformers');
  assert.equal(DOMAIN_LABELS['extraction-knowledge-graphs'], 'Information Extraction & Knowledge Graphs');
});
test('four sibling sites, formalrag included', () => {
  assert.deepEqual([...SIBLING_SITES], ['formalml', 'formalcalculus', 'formalstatistics', 'formalrag']);
});
```

Labels for the remaining domains: Text as Data · Language Modeling · Representations · Decoding & Generation · Classification & Sentiment · Text Mining · Evaluation.

- [ ] **Step 2: Run `pnpm test`.** Expected: FAIL, module not found.

- [ ] **Step 3: Implement `src/data/domains.ts`, then rewrite `src/content.config.ts`**
  - `domain: z.enum(DOMAINS)`; `crossSiteRef.site: z.enum(SIBLING_SITES)`.
  - Remove `pipelineStage` and `modality`. Keep `financeCaseStudy`, `rigorFlag`, `references` (including `documentation`), and `connections`.
  - Cross-site arrays: `formalml`, `formalcalculus`, `formalstatistics`, and `formalrag`, each with `Prereqs` and `Connections` (8 optional arrays).
  - Update the header comment to explain that formalNLP links up into formalML and formalRAG and down into calculus and statistics.

- [ ] **Step 4: Run `pnpm test && pnpm exec astro sync`.** Expected: PASS; sync exits 0.

- [ ] **Step 5: Commit** — `feat: NLP domain taxonomy and content schema`

---

### Task 3: Curriculum tracks and the 29-node planned graph

**Files:**
- Modify: `src/data/curriculum.ts`, `src/lib/validateConnections.ts` (header comment and node-count wording only)
- Create: `src/data/curriculum-graph.json`
- Test: `tests/curriculum.test.ts`

**Interfaces:**
- Consumes: `DOMAINS`, `DOMAIN_LABELS` (Task 2).
- Produces: `tracks` keeps formalRAG's shape `{domain, label, layer, description, planned: string[], dependsOnFormalML: boolean}`, with `label` taken from `DOMAIN_LABELS`. The graph keeps formalRAG's shape `{nodes: {id,label,status,domain}[], edges: {source,target}[]}`, and every node has `status: "planned"`.

**Layers** (order = `/paths` order): Foundations = text-as-data, language-modeling · Representations = representations, attention-transformers · Generation = decoding-generation · Tasks = classification-sentiment, extraction-knowledge-graphs, text-mining · Evaluation = nlp-evaluation. `dependsOnFormalML` is `true` for representations, attention-transformers, decoding-generation, and text-mining. Track descriptions: one or two sentences each, drawn from the spec §6 topics.

**Nodes** (id → label, domain). The capstone sits in `extraction-knowledge-graphs` (plan's call; the spec left it open):

| id | label | domain |
|---|---|---|
| zipf-heaps-vocabulary-statistics | Zipf's and Heaps' Laws: The Statistics of Vocabulary | text-as-data |
| subword-tokenization-bpe-unigram | Subword Tokenization as Compression: BPE, WordPiece, and Unigram LM | text-as-data |
| n-gram-models-smoothing | n-Gram Language Models and Smoothing: From Laplace to Kneser–Ney | language-modeling |
| perplexity-cross-entropy | Perplexity, Cross-Entropy, and Compression | language-modeling |
| neural-language-models | Neural Language Models: From Bengio's NPLM to Recurrent Networks | language-modeling |
| distributional-semantics-ppmi-svd | Distributional Semantics: PPMI and Truncated SVD | representations |
| word2vec-implicit-factorization | word2vec as Implicit Matrix Factorization | representations |
| glove-embedding-geometry | GloVe and the Geometry of Word Embeddings | representations |
| contextual-embeddings-masked-lm | Contextual Embeddings and the Masked-LM Objective | representations |
| attention-as-kernel-smoothing | Attention as Kernel Smoothing | attention-transformers |
| positional-encodings-geometry | Positional Encodings: Sinusoids, Rotations, and RoPE | attention-transformers |
| transformer-block-residual-stream | The Transformer Block and the Residual Stream | attention-transformers |
| scaling-laws | Neural Scaling Laws and Compute-Optimal Training | attention-transformers |
| beam-search-decoding | Decoding as Search: Greedy and Beam Search | decoding-generation |
| sampling-temperature-nucleus | Sampling: Temperature, Top-k, and Nucleus | decoding-generation |
| language-model-calibration | Calibration of Language Models | decoding-generation |
| linear-text-classifiers | Linear Text Classifiers: Generative vs. Discriminative | classification-sentiment |
| lexicon-sentiment-loughran-mcdonald | Lexicon Sentiment and the Loughran–McDonald Dictionary | classification-sentiment |
| aspect-based-sentiment | Aspect-Based Sentiment Analysis | classification-sentiment |
| fine-tuning-low-rank-adaptation | Fine-Tuning and Low-Rank Adaptation | classification-sentiment |
| sequence-labeling-hmm-crf | Sequence Labeling: From HMMs to Linear-Chain CRFs | extraction-knowledge-graphs |
| named-entity-recognition | Named Entity Recognition | extraction-knowledge-graphs |
| relation-extraction-distant-supervision | Relation Extraction and Distant Supervision | extraction-knowledge-graphs |
| knowledge-graph-embeddings | Knowledge-Graph Embeddings: TransE and RotatE | extraction-knowledge-graphs |
| topic-models-lda | Topic Models: Latent Dirichlet Allocation | text-mining |
| collocations-keyword-extraction | Collocations and Keyword Extraction | text-mining |
| overlap-metrics-bleu-rouge | Overlap Metrics: BLEU and ROUGE | nlp-evaluation |
| significance-testing-for-nlp | Significance Testing for NLP | nlp-evaluation |
| capstone-earnings-call-nlp | Capstone: An Earnings-Call NLP Pipeline | extraction-knowledge-graphs |

**Edges** (`target ← sources`, meaning each source is a prerequisite of target):

```
subword-tokenization-bpe-unigram      ← zipf-heaps-vocabulary-statistics
n-gram-models-smoothing               ← zipf-heaps-vocabulary-statistics
perplexity-cross-entropy              ← n-gram-models-smoothing
neural-language-models                ← perplexity-cross-entropy
distributional-semantics-ppmi-svd     ← zipf-heaps-vocabulary-statistics
word2vec-implicit-factorization       ← distributional-semantics-ppmi-svd, neural-language-models
glove-embedding-geometry              ← word2vec-implicit-factorization
attention-as-kernel-smoothing         ← neural-language-models
positional-encodings-geometry         ← attention-as-kernel-smoothing
transformer-block-residual-stream     ← attention-as-kernel-smoothing, positional-encodings-geometry
contextual-embeddings-masked-lm       ← transformer-block-residual-stream, word2vec-implicit-factorization
scaling-laws                          ← transformer-block-residual-stream, perplexity-cross-entropy
beam-search-decoding                  ← neural-language-models
sampling-temperature-nucleus          ← beam-search-decoding
language-model-calibration            ← sampling-temperature-nucleus
linear-text-classifiers               ← zipf-heaps-vocabulary-statistics
lexicon-sentiment-loughran-mcdonald   ← linear-text-classifiers
aspect-based-sentiment                ← lexicon-sentiment-loughran-mcdonald, attention-as-kernel-smoothing
fine-tuning-low-rank-adaptation       ← contextual-embeddings-masked-lm, linear-text-classifiers
sequence-labeling-hmm-crf             ← n-gram-models-smoothing, linear-text-classifiers
named-entity-recognition              ← sequence-labeling-hmm-crf, contextual-embeddings-masked-lm
relation-extraction-distant-supervision ← named-entity-recognition
knowledge-graph-embeddings            ← relation-extraction-distant-supervision, glove-embedding-geometry
topic-models-lda                      ← distributional-semantics-ppmi-svd
collocations-keyword-extraction       ← zipf-heaps-vocabulary-statistics, n-gram-models-smoothing
overlap-metrics-bleu-rouge            ← n-gram-models-smoothing, beam-search-decoding
significance-testing-for-nlp          ← overlap-metrics-bleu-rouge, linear-text-classifiers
capstone-earnings-call-nlp            ← lexicon-sentiment-loughran-mcdonald, knowledge-graph-embeddings, topic-models-lda
```

In the JSON, each edge is `{source: prereq, target: topic}` (formalRAG's orientation).

- [ ] **Step 1: Write the failing test `tests/curriculum.test.ts`**

```ts
test('29 planned nodes, one per domain at least, all domains valid', () => {
  assert.equal(graph.nodes.length, 29);
  assert.ok(graph.nodes.every((n) => n.status === 'planned' && DOMAINS.includes(n.domain)));
  for (const d of DOMAINS) assert.ok(graph.nodes.some((n) => n.domain === d), d);
  assert.equal(new Set(graph.nodes.map((n) => n.id)).size, 29);
});
test('edges reference real nodes and the graph is acyclic', () => {
  // every source/target ∈ node ids; Kahn's algorithm consumes all 29 nodes
});
test('the DAG fixes the build order the spec calls out', () => {
  assert.ok(hasEdge('transformer-block-residual-stream', 'contextual-embeddings-masked-lm'));
  assert.equal(graph.edges.length, 42);
});
test('tracks cover exactly DOMAINS, in layer order', () => {
  assert.deepEqual(tracks.map((t) => t.domain), [...DOMAINS]);
  assert.deepEqual([...new Set(tracks.map((t) => t.layer))],
    ['Foundations', 'Representations', 'Generation', 'Tasks', 'Evaluation']);
});
test('planned titles match graph labels per domain', () => {
  for (const t of tracks)
    assert.deepEqual(t.planned, graph.nodes.filter((n) => n.domain === t.domain).map((n) => n.label));
});
```

- [ ] **Step 2: Run `pnpm test`.** Expected: FAIL.

- [ ] **Step 3: Write `curriculum-graph.json` (nodes in table order) and `curriculum.ts`.**

- [ ] **Step 4: Run `pnpm test`.** Expected: PASS.

- [ ] **Step 5: Run `pnpm validate`.** Expected: exit 0, 0 errors, 29 roadmap notices.

- [ ] **Step 6: Run `pnpm build && pnpm preview`, then open `/paths` and `/` in the browser.** Expected: `/paths` shows 5 layer headers and 9 tracks with planned titles; the home DAG renders 29 dashed (planned) nodes; the browser console shows no errors.

- [ ] **Step 7: Commit** — `feat: planned 29-node NLP curriculum graph`

---

### Task 4: Marginalia theme and formalNLP identity

**Files:**
- Modify: `src/styles/global.css`, `src/components/viz/shared/colorScales.ts`, `src/layouts/BaseLayout.astro`, `src/components/ui/{Nav,Footer,RigorFlag}.astro`, `src/pages/{index,about}.astro`, `src/pages/topics/index.astro`, `public/favicon.svg`, `package.json`
- Create: `src/lib/contrast.ts`
- Test: `tests/branding.test.ts`, `tests/contrast.test.ts`

**Interfaces:**
- Produces: `contrastRatio(fgHex: string, bgHex: string): number` (WCAG 2.x relative-luminance formula) in `src/lib/contrast.ts`.

**Values:**
- Light tokens are exactly the spec §5 table.
- Dark tokens (plan's call, verified by the test): `--color-bg #1A1614`, `--color-surface #231E1B`, `--color-text #EDE6DA`, `--color-text-secondary #B3A99B`, `--color-accent #E0697C`, `--color-accent-secondary #D9A441`, `--color-border #3A322B`, `--color-definition-bg #3A1E24`, `--color-theorem-bg #3A2F1A`.
- Fonts are self-hosted (local-first). Add `@fontsource-variable/crimson-pro` and `@fontsource-variable/inter`, and import both in `global.css`. Set `--font-serif: 'Crimson Pro Variable', 'Georgia', ui-serif, serif` and `--font-sans: 'Inter Variable', ui-sans-serif, system-ui, sans-serif`.
- `domainColorScale` gets explicit ranges in `DOMAINS` order: `#9F1D35 #A16207 #2F5D62 #5B4B8A #C2410C #4D7C0F #1E4E79 #8C5A3C #57534E`, then `formalml #9ca3af`, `formalcalculus #6b7280`, `formalstatistics #4b5563`, `formalrag #374151`.
- Copy: wordmark `formal<span accent>NLP</span>`; default description and home tagline "The mathematics of language"; home blurb lists the nine domains in prose. Footer GitHub → `https://github.com/jonx0037/formalNLP`. The `/about` page explains the five-site family and the finance thread. Favicon: oxblood "N" on paper.

- [ ] **Step 1: Write the failing tests**

```ts
// tests/contrast.test.ts
import { contrastRatio } from '../src/lib/contrast';
test('reference values', () => {
  assert.equal(contrastRatio('#000000', '#FFFFFF').toFixed(1), '21.0');
  assert.equal(contrastRatio('#FFFFFF', '#FFFFFF'), 1);
});
test('Marginalia accents meet WCAG AA (4.5:1) on both backgrounds', () => {
  for (const [fg, bg] of [['#9F1D35', '#FBF8F1'], ['#A16207', '#FBF8F1'], ['#211A17', '#FBF8F1'],
                          ['#E0697C', '#1A1614'], ['#D9A441', '#1A1614'], ['#EDE6DA', '#1A1614']])
    assert.ok(contrastRatio(fg, bg) >= 4.5, `${fg} on ${bg}`);
});
test('global.css carries exactly these tokens', () => { /* regex each light + dark value above */ });

// tests/branding.test.ts
test('no formalRAG identity remains in src/', () => {
  assert.deepEqual(execGrep(/formalRAG|retrieval-augmented|Source Serif 4/i, 'src')
    .filter((h) => !/formalrag(Prereqs|Connections)|'formalrag'|formalrag\.com/.test(h)), []);
});
test('domainColorScale covers every domain plus four siblings', () => {
  assert.deepEqual(domainColorScale.domain(), [...DOMAINS, ...SIBLING_SITES]);
});
```

- [ ] **Step 2: Run `pnpm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `contrastRatio` and apply the tokens, fonts, color scale, copy, and favicon.** If a dark value fails AA, adjust that dark value, not the light ones.

- [ ] **Step 4: Run `pnpm test && pnpm build`.** Expected: PASS; exit 0. Also run `grep -ri "formalrag" dist/*.html dist/**/*.html`, which should return only `formalrag.com` cross-site links (none exist yet, so empty).

- [ ] **Step 5: Visual check in the browser** at `/` and `/paths`, in light and dark (ThemeToggle). Compare against the selected mockup at `.superpowers/brainstorm/*/content/visual-identity.html` (card A).

- [ ] **Step 6: Commit** — `feat: Marginalia theme and formalNLP identity`

---

### Task 5: Prelaunch noindex mode

**Files:**
- Create: `src/lib/siteConfig.ts`, `src/lib/robots.ts`, `src/pages/robots.txt.ts`
- Modify: `src/layouts/BaseLayout.astro`
- Test: `tests/robots.test.ts`

**Interfaces:**
- Produces: `export const PRELAUNCH = true;` and `export const SITE_URL = 'https://formalnlp.com';` in `siteConfig.ts`. Also `robotsTxt(prelaunch: boolean, siteUrl: string): string` in `robots.ts`. Flipping `PRELAUNCH` to `false` is the launch switch.

- [ ] **Step 1: Write the failing test**

```ts
test('prelaunch disallows everything', () => {
  assert.equal(robotsTxt(true, SITE_URL), 'User-agent: *\nDisallow: /\n');
});
test('launch allows crawling', () => {
  assert.equal(robotsTxt(false, SITE_URL), 'User-agent: *\nAllow: /\n');
});
test('the shipped flag is prelaunch', () => assert.equal(PRELAUNCH, true));
```

- [ ] **Step 2: Run `pnpm test`.** Expected: FAIL.

- [ ] **Step 3: Implement.** `robots.txt.ts` is a static `GET` endpoint returning `robotsTxt(PRELAUNCH, SITE_URL)` with `Content-Type: text/plain`. `BaseLayout` emits `<meta name="robots" content="noindex, nofollow" />` when `PRELAUNCH`.

- [ ] **Step 4: Verify**

Run: `pnpm test && pnpm build && cat dist/robots.txt && grep -L 'noindex, nofollow' $(find dist -name '*.html' -not -path '*/pagefind/*')`
Expected: tests PASS; `robots.txt` shows `Disallow: /`; the `grep -L` lists **no** files (every page carries the tag).

- [ ] **Step 5: Commit** — `feat: prelaunch noindex mode`

---

### Task 6: Five-site cross-site linking

**Files:**
- Create: `scripts/lib/crossSite.mjs`
- Modify: `scripts/audit-cross-site-links.mjs`, `src/lib/synthesizeCrossSiteNodes.ts`, `src/lib/validateConnections.ts` (comment only)
- Test: `tests/crossSite.test.ts`

**Interfaces:**
- Produces, from `scripts/lib/crossSite.mjs`:
  - `SELF = 'formalnlp'`
  - `SITES: Record<'formalnlp'|'formalml'|'formalcalculus'|'formalstatistics'|'formalrag', {label: string, url: string, envVar: string | null, defaultPath: string}>`. Env vars are `FORMAL_ML_PATH`, `FORMAL_CALCULUS_PATH`, `FORMAL_STATISTICS_PATH`, `FORMAL_RAG_PATH`; defaults are `../formalML` etc.; self has `envVar: null` and `defaultPath: '.'`.
  - `CROSS_SITE_FIELDS: string[]` (all 10 `<site>{Prereqs,Connections}` for the 5 sites)
  - `parseField(field: string): {targetSite: string, direction: 'Prereqs'|'Connections'} | null`
  - `reciprocalField(sourceSite: string, direction: string): string`
- The audit script imports these instead of defining its own `REPOS`, `SITE_LABELS`, `CROSS_SITE_FIELDS`, and `FIELD_RE`. The rest of its logic (edge extraction, checks, report rendering) stays as in formalRAG, generalized over `SITES`.

- [ ] **Step 1: Write the failing test**

```ts
import { SITES, SELF, CROSS_SITE_FIELDS, parseField, reciprocalField } from '../scripts/lib/crossSite.mjs';
test('five sites, formalnlp is self', () => {
  assert.deepEqual(Object.keys(SITES).sort(),
    ['formalcalculus', 'formalml', 'formalnlp', 'formalrag', 'formalstatistics']);
  assert.equal(SELF, 'formalnlp');
  assert.equal(SITES.formalrag.url, 'https://formalrag.com');
  assert.equal(CROSS_SITE_FIELDS.length, 10);
});
test('parses fields for the new sites', () => {
  assert.deepEqual(parseField('formalragPrereqs'), { targetSite: 'formalrag', direction: 'Prereqs' });
  assert.deepEqual(parseField('formalnlpConnections'), { targetSite: 'formalnlp', direction: 'Connections' });
  assert.equal(parseField('formalxyzPrereqs'), null);
  assert.equal(parseField('formalragprereqs'), null);
});
test('reciprocal flips direction and names the source site', () => {
  assert.equal(reciprocalField('formalnlp', 'Prereqs'), 'formalnlpConnections');
});
test('synthesizeCrossSiteNodes resolves formalrag URLs', () => { /* a topic with formalragPrereqs → external node url https://formalrag.com/topics/<slug> */ });
```

- [ ] **Step 2: Run `pnpm test`.** Expected: FAIL.

- [ ] **Step 3: Implement `crossSite.mjs`, rewire the audit script, and extend `synthesizeCrossSiteNodes.ts`.** That file needs `CrossSiteSite` to include `'formalrag'`, `SITE_DOMAINS.formalrag`, `formalragPrereqs` read alongside the others, and NLP acronyms (`bpe`, `lda`, `crf`, `hmm`, `ner`, `lm`, `rope`, `bleu`, `rouge`, `lora`, `ppmi`, `svd`) added to `ACRONYMS`.

- [ ] **Step 4: Verify**

Run: `pnpm test && pnpm audit:cross-site`
Expected: tests PASS; the audit exits 0 with the four siblings adjacent, reports 0 formalNLP edges, and writes only gitignored outputs. Confirm with `git status --short`, which should show no `docs/plans/audit-output` entries.

- [ ] **Step 5: Commit** — `feat: generalize cross-site audit to five sites`

---

### Task 7: Authoring docs and the first PR

**Files:**
- Create: `CLAUDE.md`, `STARTER-PROMPT.md`, `notebooks/README.md`; rewrite `README.md`

**Content decisions:**
- `STARTER-PROMPT.md` is formalRAG's with these changes:
  - Brief path `docs/plans/formalnlp-<slug>-brief.md`.
  - The placeholder table drops `{{PIPELINE_STAGE}}` and `{{MODALITIES}}` and gains `{{FORMALRAG_PREREQS}}`.
  - The nine domains replace the ten.
  - Structural exemplar: formalRAG's `bm25-binary-independence-model` (read from `../formalRAG`) until formalNLP's first topic ships, then that topic.
  - New Phase A rule: finance case studies import `notebooks/finance-corpus/finance_corpus.py`, and the real LM dictionary is fetched at runtime, never committed.
  - New stop-and-ask item: any topic on the spec's §6 overlap watch-list must state its scope against the named sibling in its brief.
  - Validation gates add `pnpm test`.
- `CLAUDE.md`: formalRAG's general conventions (KaTeX non-strict gotcha, `browser_evaluate` DOM checks, viz↔Python invariant, prereq `.py` import rule) plus a formalNLP do-NOT list:
  - Don't flip `PRELAUNCH` before all 29 nodes are published.
  - Don't touch sibling repos before launch.
  - Don't commit the LM dictionary or transcripts.
  - Don't cover ANN/vector DBs/retrieval models.
- `notebooks/README.md`: formalRAG's contract, with exemplar `notebooks/finance-corpus/` for import pattern and harness layout.

- [ ] **Step 1: Write the four files.**

- [ ] **Step 2: Verify that no RAG-only instructions remain**

Run: `grep -nE "pipelineStage|modality|PIPELINE_STAGE|MODALITIES|ten domains" CLAUDE.md STARTER-PROMPT.md notebooks/README.md`
Expected: no output.

- [ ] **Step 3: Run the full gate set**

Run: `pnpm test && pnpm exec astro sync && pnpm validate && pnpm build`
Expected: all exit 0.

- [ ] **Step 4: Commit, push, and open the PR**

```bash
git add -A && git commit -m "docs: formalNLP authoring playbook and conventions"
git push -u origin feat/site-scaffold
gh pr create --title "Scaffold formalNLP (Marginalia, 29-node roadmap, noindex)" --body "<what shipped, gate results, prelaunch note>"
```

Review the PR before merging. (Merge is not part of this task.)

---

### Task 8: Shared synthetic finance corpus

**Files:**
- Create: `notebooks/finance-corpus/finance_corpus.py`, `notebooks/finance-corpus/01_finance_corpus.ipynb`, `notebooks/finance-corpus/edgar/SOURCES.md`, `notebooks/finance-corpus/edgar/*.txt`

Branch: `feat/finance-corpus`, cut from `main` after Task 7's PR merges.

**Interfaces** (every `financeCaseStudy` notebook and the capstone import these):
```python
@dataclass(frozen=True)
class Entity:   start: int; end: int; type: Literal["ORG", "PERSON", "MONEY", "PERCENT", "DATE", "PRODUCT"]
@dataclass(frozen=True)
class Relation: head: int; rel: Literal["CEO_OF", "REPORTED", "ACQUIRED", "GUIDED"]; tail: int   # indices into entities
@dataclass(frozen=True)
class FinanceDoc:
    doc_id: str; company: str
    doc_type: Literal["earnings-call", "mdna", "8-k"]
    text: str; tone: Literal[-1, 0, 1]
    entities: tuple[Entity, ...]; relations: tuple[Relation, ...]

STANDIN_LEXICON: dict[Literal["positive", "negative", "uncertainty", "litigious"], frozenset[str]]
def generate_corpus(n_docs: int = 300, seed: int = 0) -> list[FinanceDoc]
def lexicon_score(text: str, lexicon: dict[str, frozenset[str]] = STANDIN_LEXICON) -> float  # (pos − neg) / max(1, tokens)
def load_edgar_excerpts() -> list[FinanceDoc]   # tone=0, entities/relations empty; text from edgar/*.txt
```

`STANDIN_LEXICON` is hand-written (15–30 words per category) and documented in the module docstring as **not** derived from the Loughran–McDonald dictionary. Text is generated from templates over a fictional company roster, with filler vocabulary sampled from a Zipf(s ≈ 1.1) distribution so the corpus has realistic rank-frequency structure. Deps in the docstring: `uv run --with numpy --with scipy python notebooks/finance-corpus/finance_corpus.py`.

- [ ] **Step 1: Write the harness first (`if __name__ == "__main__":` runs every `test_*`).** It should fail because the generators raise `NotImplementedError`.

```python
def test_deterministic():         assert generate_corpus(50, seed=7) == generate_corpus(50, seed=7)
def test_seed_changes_output():   assert generate_corpus(50, seed=7) != generate_corpus(50, seed=8)
def test_size():                  assert len(generate_corpus(300)) == 300
def test_tone_balanced():         # each of −1/0/1 is within 33% ± 10 points of 300
def test_doc_types_present():     # all three doc_type values occur
def test_entity_spans_valid():    # 0 ≤ start < end ≤ len(text) and text[start:end].strip() == text[start:end] != ""
def test_relations_index_entities(): # 0 ≤ head, tail < len(entities), head != tail
def test_zipfian():               # OLS slope of log freq vs log rank over ranks 1–200 ∈ [−1.3, −0.7]
def test_lexicon_disjoint():      # positive ∩ negative == ∅
def test_lexicon_signal():        # Spearman(lexicon_score(text), tone) ≥ 0.5 across the corpus
def test_fast():                  # generate_corpus(300) < 5 s
def test_edgar_manifest():        # every edgar/*.txt has a SOURCES.md row; ≥ 3 excerpts; each ≤ 400 words
```

- [ ] **Step 2: Run** `uv run --with numpy --with scipy python notebooks/finance-corpus/finance_corpus.py`. Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement the generator, `lexicon_score`, and the loader.**

- [ ] **Step 4: Add at least three EDGAR excerpts.** These are MD&A paragraphs of ≤ 400 words each from public 10-K filings. `SOURCES.md` columns: file, company, form, filing date, accession number, URL. SEC fair-access rules require a declared `User-Agent` with contact details. **Ask Jonathan for the User-Agent string or have him fetch the excerpts; never invent or reuse his email without asking.**

- [ ] **Step 5: Run the harness again.** Expected: every test passes; prints the corpus summary; exit 0.

- [ ] **Step 6: Generate and execute the narrative notebook** with formalRAG's nbformat-generator method (see its CLAUDE.md), with no stored outputs.
Run: `uv run --with numpy --with scipy --with jupyter jupyter execute notebooks/finance-corpus/01_finance_corpus.ipynb`
Expected: exit 0.

- [ ] **Step 7: Commit, push, and open the PR** — `feat: shared synthetic finance corpus`

---

### Task 9: Vercel project, domain, and live noindex verification

Runs after both PRs merge. Every step here is outward-facing, so confirm each with Jonathan before running it.

- [ ] **Step 1:** Create the Vercel project from `jonx0037/formalNLP` (framework Astro, `pnpm build`, output `dist`). Production branch is `main`.
- [ ] **Step 2:** Add `formalnlp.com` and `www.formalnlp.com` to the project. Configure DNS at the current registrar per Vercel's instructions; `www` redirects to the apex, matching the siblings.
- [ ] **Step 3: Verify the live prelaunch state**

Run: `curl -s https://formalnlp.com/robots.txt && curl -s https://formalnlp.com/ | grep -c 'noindex, nofollow'`
Expected: `Disallow: /` and a count ≥ 1. Then open `/`, `/paths`, and `/topics` in the browser and confirm they render in Marginalia, light and dark.

---

## Topic Roadmap (after this plan)

Each topic is one run of `STARTER-PROMPT.md` (Phase A brief + verified notebook → Phase B MDX + viz + graph + PR), in this topological order:

1. zipf-heaps-vocabulary-statistics · 2. subword-tokenization-bpe-unigram · 3. n-gram-models-smoothing · 4. perplexity-cross-entropy · 5. neural-language-models · 6. distributional-semantics-ppmi-svd · 7. word2vec-implicit-factorization · 8. glove-embedding-geometry · 9. attention-as-kernel-smoothing · 10. positional-encodings-geometry · 11. transformer-block-residual-stream · 12. contextual-embeddings-masked-lm · 13. scaling-laws · 14. beam-search-decoding · 15. sampling-temperature-nucleus · 16. language-model-calibration · 17. linear-text-classifiers · 18. lexicon-sentiment-loughran-mcdonald · 19. aspect-based-sentiment · 20. fine-tuning-low-rank-adaptation · 21. sequence-labeling-hmm-crf · 22. named-entity-recognition · 23. relation-extraction-distant-supervision · 24. knowledge-graph-embeddings · 25. topic-models-lda · 26. collocations-keyword-extraction · 27. overlap-metrics-bleu-rouge · 28. significance-testing-for-nlp · 29. capstone-earnings-call-nlp

**Blockers to watch:** #3 cites formalStatistics `empirical-bayes` (published 2026-10-23 — cleared) and #21 cites `hidden-markov-models`, still `draft` on that site. It must publish first, or the brief must cite an alternative.

**Launch** (a separate plan, once #29 ships): flip `PRELAUNCH` to `false`; open one reciprocal PR per sibling (`'formalnlp'` in the `crossSiteRef` enum, `formalnlp{Prereqs,Connections}` arrays, audit coverage, nav/footer link); announce.
