# STARTER-PROMPT.md — authoring a formalNLP topic

This is the **running playbook** for taking one curriculum topic from nothing to a shipped,
verified page. It is a living document: when a convention changes, update *this file* — do not
fork it per topic. Each topic copies the placeholder table, fills it in, and works the two phases
in order.

formalNLP is a rigor-first site on the mathematics of language (https://formalnlp.com — live but
`noindex` until launch). Every topic gets **three pillars**: rigorous KaTeX math, an interactive D3
viz, and a working Python notebook pillar. We author end-to-end in Claude Code; project conventions
and the do-NOT list live in [`CLAUDE.md`](CLAUDE.md), the site design in
[`docs/plans/formalnlp-site-design.md`](docs/plans/formalnlp-site-design.md), and general
preferences (pnpm, uv, Chicago citations, American English, geometric-first, git workflow) in
`~/CLAUDE.md`. Prose is self-approved per standing authorization — flag only genuine objections.

---

## How to use this file

1. **Pick the next topic** from the Topic Roadmap in
   `docs/plans/2026-10-22-formalnlp-scaffold-plan.md` (topological order) — it matches
   `src/data/curriculum-graph.json`. Confirm its prerequisites are published (or backfill first).
2. **Fill the placeholder table** below. Everything topic-specific lives there; the two phases are
   invariant across topics.
3. **Run Phase A** (draft & verify the math + notebook). Deliverables: a brief at
   `docs/plans/formalnlp-<slug>-brief.md` and the verified notebook pair in `notebooks/<slug>/`.
4. **Run Phase B** (implement & ship the MDX + viz + graph + PR). Each pillar is verified against
   the notebook's numbers before it is considered done.
5. **Stop and ask** at any of the checkpoints flagged below.

### Placeholder table (fill per topic)

| Placeholder | Example | Where it goes |
|---|---|---|
| `{{TITLE}}` | `Attention as Kernel Smoothing` | frontmatter `title` (start from the graph node label) |
| `{{SUBTITLE}}` | one sentence on the core idea | frontmatter `subtitle` |
| `{{SLUG}}` | `attention-as-kernel-smoothing` (must equal the graph node id) | filename, `notebooks/<slug>/` |
| `{{DOMAIN}}` | `attention-transformers` | frontmatter `domain` (one of the 9 below) |
| `{{DIFFICULTY}}` | `intermediate` | `foundational` \| `intermediate` \| `advanced` |
| `{{FINANCE_CASE_STUDY}}` | `true` | `financeCaseStudy` (bool) |
| `{{PREREQUISITES}}` | `[neural-language-models]` | in-site slugs — the graph's incoming edges |
| `{{FORMALML_PREREQS}}` | `[kernel-regression]` | `formalmlPrereqs` |
| `{{FORMALRAG_PREREQS}}` | `[pmi-retrieval-value]` | `formalragPrereqs` |
| `{{STATS_CALC_PREREQS}}` | `[maximum-likelihood]` | `formalstatisticsPrereqs` / `formalcalculusPrereqs` |
| `{{RIGOR_FLAG}}` | the honest caveat (see Phase A) | frontmatter `rigorFlag` |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/finance-corpus/` | shipped notebooks to model voice/layout on |

The nine domains (`src/data/domains.ts`): `text-as-data`, `language-modeling`, `representations`,
`attention-transformers`, `decoding-generation`, `classification-sentiment`,
`extraction-knowledge-graphs`, `text-mining`, `nlp-evaluation`.

---

## Phase A — Draft & verify (math + notebook)

Produce the mathematics and the notebook pillar *before* touching MDX or viz. The notebook is the
source of truth that the later pillars are checked against.

### A1. The three-pillar contract

- **Math (KaTeX):** definitions, theorems with *full* proofs (never "it can be shown"), and a
  concrete motivating example before each definition. When a full proof is genuinely hard, prove the
  *load-bearing* direction and **cite** the rest to its source, backed by a harness assert — honest
  attribution beats a hand-waved proof.
- **Viz (D3):** one or more interactive components; each shows a parameter the reader can
  manipulate and a thing they should learn from manipulating it.
- **Python (notebook pillar):** the `notebooks/README.md` contract —
  - `notebooks/<slug>/<slug_underscored>.py` — canonical, tested, importable, CPU-only, < 60 s. Deps
    declared in the module docstring as the `uv run --with … python …` line. A harness of
    `assert`-based tests encoding **every pedagogical claim the topic makes**, plus a cross-check
    against a reference library where one exists. This file **owns every number the viz mirrors**.
    Test limit theorems on **strict/random instances**, not only the worked example.
  - `notebooks/<slug>/01_<slug_underscored>.ipynb` — narrative notebook that imports the `.py`.
    **Generate it programmatically** with `nbformat.v4.new_markdown_cell`/`new_code_cell` (raw
    triple-quoted strings so KaTeX backslashes need no escaping), `nbformat.validator.normalize`,
    clear outputs, `nbformat.write`. Commit without stored outputs.
  - Both exit 0:
    ```bash
    uv run --with <deps> python notebooks/<slug>/<slug_underscored>.py
    uv run --with <deps> --with jupyter --with ipykernel jupyter execute notebooks/<slug>/01_<slug_underscored>.ipynb
    ```
- **Data rules (formalNLP-specific):**
  - Corpora are small, license-clean, and shippable in the repo or generated. No model downloads
    beyond a small tokenizer or embedding table; transformer topics use toy-scale models written
    from scratch in NumPy (or PyTorch on CPU).
  - **Finance case studies import `notebooks/finance-corpus/finance_corpus.py`** (the shared
    synthetic corpus, stand-in lexicon, and EDGAR excerpts) — never re-generate a private corpus.
  - **The real Loughran–McDonald dictionary is fetched at runtime and never committed** (its
    non-commercial license vs. this public, DataSalt-owned repo). Asserts run against the in-repo
    stand-in lexicon so the harness stays offline.
  - Monte-Carlo loops draw every sample from ONE `np.random.default_rng()` stream (or
    `SeedSequence(base).spawn(N)`); per-seed `default_rng(s)` over consecutive seeds correlates draws.

### A2. Editorial voice (non-negotiable)

Informed peer, not lecturer. Collaborative "we" (we define, we observe); "you" only for direct
reader instructions ("try dragging the slider"). Introduce **all** notation on first use. No
"simply", "obviously", "it's easy to see". American English. Match the register of formalML's
`svd.mdx`. **Geometric-first:** introduce visually and concretely, then the algebra; foundational
topics stop at geometric intuition. The transformer core is mathematics (kernels, geometry,
probability), never product news.

### A3. The schema fields, and what they mean

Required: `title`, `subtitle`, `status` (`draft` while authoring → `published` at ship),
`difficulty`, `tags`, `domain`, `abstract`, `prerequisites`. formalNLP-specific:

- `financeCaseStudy` — `true` if the topic carries the recurring finance thread (earnings calls,
  10-K MD&A, the finance sentiment vocabulary). Wrap the case study in `<FinanceCaseStudy>`.
- **`rigorFlag` is load-bearing.** Name the celebrated-but-heuristic parts plainly: Kneser–Ney's
  discounts, scaling-law fits that are empirical, BLEU's weak correlation with human judgment,
  word-analogy arithmetic that holds only approximately. Never launder a heuristic as a theorem.
- `notebookPath` points at the `.ipynb`; `githubUrl` at the MDX on GitHub
  (`https://github.com/jonx0037/formalNLP/blob/main/src/content/topics/<slug>.mdx`).
- `references[].type` ∈ `paper | book | course | blog | video | documentation`. **Every reference
  needs a `url`** — DOI for papers/books, ACL Anthology / proceedings / arXiv otherwise.

### A4. Cross-site links — four siblings

formalNLP links **UP into formalML and formalRAG** and **DOWN into formalStatistics and
formalCalculus**. Each entry is `{ topic, site, relationship }` where
`site ∈ {formalml, formalcalculus, formalstatistics, formalrag}` and `relationship` is ≥ 40 chars of
real prose:

- `formalmlPrereqs` / `formalragPrereqs` / `formalstatisticsPrereqs` / `formalcalculusPrereqs` —
  what this topic *needs*.
- `…Connections` — what this topic *informs* or sits beside (rarer).

**Verify the target slug exists AND is `status: published` on the sibling before citing it — do not
invent.** Fast check: `ls ../<sibling>/src/content/topics/` and
`grep '^status:' ../<sibling>/src/content/topics/<slug>.mdx`. A sibling checkout can lag its GitHub
`main` (or sit on another branch), so the authority is `main` itself:
`gh api "repos/jonx0037/<sibling>/contents/src/content/topics/<slug>.mdx?ref=main" --jq .content | base64 -d | grep '^status:'`.
Known drafts at scaffold time: formalStatistics `empirical-bayes` (published 2026-10-23) and
`hidden-markov-models`.
**Reverse links on the siblings are NOT added per topic** — they are batched into one PR per sibling
at launch (design doc §10.4). Do not touch sibling repos.

### A5. Phase A deliverables

- `docs/plans/formalnlp-<slug>-brief.md` — the implementation spec: section outline, theorem
  statements + proofs, viz design intent (component-level, not React/D3 code), notebook-cell map,
  cross-site prereq list, references, and — for watch-list topics — a scope statement against the
  overlapping sibling topic. Per the scaffold-don't-draft preference, build the outline and the math
  collaboratively before drafting full prose.
- `notebooks/<slug>/` — the verified `.py` + `.ipynb`, both exiting 0.

---

## Phase B — Implement & ship (MDX + viz + graph + PR)

The notebook is now immutable and is the source of truth. Build the page against it.

### B1. Required reading (in order)

`CLAUDE.md` → `docs/plans/formalnlp-<slug>-brief.md` → `notebooks/<slug>/01_<slug_underscored>.ipynb`.
Then study a structural exemplar: **until formalNLP's first topic ships, use formalRAG's**
`../formalRAG/src/content/topics/bm25-binary-independence-model.mdx` (frontmatter anatomy,
`TheoremBlock`/`NamedSection`/`RigorFlag`/`FinanceCaseStudy` usage, the `<Viz client:visible />`
embed) and `../formalRAG/src/components/viz/BM25ScoringLaboratory.tsx`. After that, use formalNLP's
own first published topic. (formalRAG is read-only — never edit it.)

### B2. Build order

1. Author the MDX from the brief — pull definitions/theorems/proofs from the notebook, adapt only
   formatting for `TheoremBlock`. Set the full frontmatter (A3) with `status: draft`.
2. Build each viz component in the Marginalia palette (use the `--color-*` tokens, never hex).
   **Viz ↔ Python invariant:** any corpus/weights/numbers the viz shows are mirrored *to the
   decimal* from the topic's `.py`, with a comment citing the asserting test. Change one → change both.
3. In `src/data/curriculum-graph.json` flip the node's `status` to `published`, and remove its title
   from `planned` in `src/data/curriculum.ts`. Wire `prerequisites` + `connections`; both endpoints
   must resolve to real MDX. (The node id and its edges already exist — don't invent new ones
   without updating the design.)
4. Add the outward cross-site arrays (A4).
5. Flip `status: published`, set `datePublished`, `estimatedReadTime`, `githubUrl`, `notebookPath`.

### B3. Validation gates (run in order)

```bash
pnpm test                  # unit tests: domains, curriculum DAG, branding, robots, cross-site
pnpm exec astro sync       # schema + ALL frontmatter (fast, no full render)
pnpm validate              # validateConnections.ts — prereq/connection/graph integrity
pnpm audit:cross-site      # five-site reciprocity (siblings adjacent / FORMAL_*_PATH)
pnpm build && pnpm preview # then open the topic page
```

`pnpm test`'s curriculum test checks that `planned` titles match graph labels for *planned* nodes —
update both files together when a topic ships.

**`pnpm build` / exit 0 ≠ math correct.** KaTeX is non-strict: parse errors render as
`.katex-error` spans and the build still exits 0. **Open the page and verify the DOM with
`javascript_exec`/`browser_evaluate`, not screenshots.** Assert: zero `.katex-error` spans, the
expected `.katex` count, and that each viz mounted. The viz embeds with `client:visible`: to exercise
interactivity, `scrollIntoView` the component and wait ~1–1.5 s for hydration (a background or
narrow browser pane may never hydrate a `hidden md:block` island — use a desktop-width, foreground
pane). Re-run the notebook one final time; confirm the page's numbers equal the notebook's output.

**`audit:cross-site` writes generated files — never commit them** (`docs/plans/audit-output/`,
`docs/plans/cross-site-audit-report.md`, `docs/plans/deferred-reciprocals.md`, all gitignored). It
exits 0 with warnings; a new topic's outward links show up as **missing reciprocals** until launch.
That is the expected deferred state.

### B4. Ship

Feature branch → commit → PR for review before `main` (deploys trigger from `main`). No worktrees
for MDX/content. PR body: what shipped (viz count, prereq additions), the validation-gate results,
and any honest caveats the `rigorFlag` encodes. **Never flip `PRELAUNCH`** in a topic PR.

---

## Stop and ask if

- A proof needs a lemma/prerequisite not on a published in-site or sibling topic the reader can be
  assumed to have read.
- A viz needs a dataset or parameter regime not yet specified.
- A cross-site target slug can't be verified to exist and be published — never fabricate one.
- A verification number disagrees with the notebook, or a `.katex-error` appears and the fix is not
  obvious.
- You're tempted to change a verified number in the harness *or* the viz without updating the other.
- The topic is on the **overlap watch-list** (design doc §6) and its brief has no scope statement
  against the named sibling topic — or the topic would duplicate or contradict any published topic.
- A finance example would need real transcripts, the real LM dictionary committed, or any
  non-public data.

---

*Living document. When a convention, command, or schema field changes, update this file and
`CLAUDE.md` together. The real test that this prompt is complete is authoring the next topic with it
and hitting no missing step.*
