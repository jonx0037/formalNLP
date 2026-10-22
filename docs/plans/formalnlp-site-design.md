# formalNLP.com — Site Design

*Status: draft for approval · 2026-10-22 · Owner: Jonathan Rocha (DataSalt LLC)*

## 1. Intent

formalNLP is the fifth site in the formal\* series (formalCalculus → formalStatistics → formalML →
formalRAG). It fills the gap the series currently leaves open: **the mathematics of turning text into
structure and meaning, and of modeling language itself.** None of the four live sites covers
tokenization, language modeling, word/contextual embeddings as a subject, attention and transformers,
decoding, sequence labeling, information extraction, knowledge-graph embeddings, topic models, or
text-generation evaluation.

It is **lighter** than formalML (~80 topics) and formalRAG (~55 topics) — roughly **27–30 topics** —
but held to the same three-pillar rigor. It leans hard on its siblings: formalML and formalStatistics
for foundations and case-study machinery, formalRAG for retrieval, vector search, and evaluation.

It will eventually become a YouTube playlist on the Jonathan-Rocha-AI channel, like its siblings.

## 2. Goals, non-goals, anti-goals

**Goals**
- Own the **language-modeling and transformer core** (decided) plus the applied-task layer:
  classification/sentiment, information extraction/knowledge graphs, text mining.
- Three pillars per topic, identical to formalRAG: rigorous KaTeX math with full proofs, ≥1
  interactive D3 lab, and a verified Python notebook pair.
- **Finance case-study thread**, flagged per topic exactly as formalRAG does (`financeCaseStudy`),
  drawing on *Applied NLP for Finance* (e.g., Loughran–McDonald lexicon, earnings-call tone, SEC
  filings).
- Dense, real cross-site edges into formalML, formalStatistics, formalCalculus, and formalRAG.
- Launch with a **fuller curriculum before announcing** (see §8), not a soft launch of a few topics.

**Non-goals**
- **Vector databases / ANN indexing / quantization.** formalRAG already owns HNSW, IVF, PQ/OPQ, LSH,
  filtered ANN, MIPS hardness. formalNLP cross-links; it does not re-cover.
- **Retrieval models** (TF-IDF, BM25, query likelihood, dense retrieval, rerankers) — formalRAG.
- **General data mining / clustering** — formalML. formalNLP covers only *text* mining.
- **Agentic AI** (MCP, tool calling, deterministic workflows, agent security) — reserved for
  formalAgentic.
- formalSportsAnalytics is parked; no dependency in either direction.

**Anti-goals** (failure even if it ships)
- A topic that duplicates or quietly contradicts a published sibling topic.
- "It can be shown" hand-waving; heuristic results presented as guarantees (use `rigorFlag`).
- An LLM-hype site: the transformer core is treated as mathematics (kernels, geometry, probability),
  not as product news.
- Visually indistinguishable from formalML/formalRAG.

## 3. Constraints (from standing preferences)

- `pnpm` (never npm) · `uv` for Python · Chicago 17th (Notes & Bibliography) · American English ·
  geometric-first exposition.
- Git: feature branch → commit → push → PR → merge to `main`; Vercel deploys from `main`. No worktrees
  for MDX/content.
- Site repo `jonx0037/formalNLP` is **public**, matching formalRAG. Any `formalnlp-animations` repo is
  **private**; YouTube scripts/slides live under `~/Developer/YouTube-Projects/formalNLP`, local-only.
- Local-first: notebooks CPU-only, < 60 s, no paid APIs in the code pillar.

## 4. Decided: how it gets built

**Scaffold from formalRAG**, not from scratch and not from formalML. formalRAG is the only sibling with
the notebook pillar, `FinanceCaseStudy`/`RigorFlag` components, and the per-topic `STARTER-PROMPT.md`
playbook. Same stack: Astro 6, React 19, Tailwind 4, MDX, KaTeX, D3 7, Pagefind, Vercel.

Carry over unchanged (adapting names only):
- `src/components/ui/*` (TheoremBlock, NamedSection, RigorFlag, FinanceCaseStudy, PrerequisiteChip,
  ThemeToggle, TopicCard, Nav, Footer, Figure, DifficultyBadge, ExternalLink).
- `TopicLayout`, `BaseLayout`, `/topics`, `/topics/[...slug]`, `/paths`, `/about`, curriculum graph.
- `src/lib/validateConnections.ts`, `scripts/audit-cross-site-links.mjs`, the `notebooks/README.md`
  contract (importable `.py` with assert harness owning every number the viz mirrors + narrative
  `.ipynb`; dependent topics import their prereq's `.py`).
- `STARTER-PROMPT.md` adapted for formalNLP (Phase A brief + verified notebook → Phase B MDX, viz,
  graph, PR), and a `CLAUDE.md` with the do-NOT list.

Drop: `PipelineDiagram`, `/architectures`, `src/data/architectures.ts`, `src/data/pipeline.ts`, and the
RAG-specific frontmatter (`pipelineStage`, `modality`).

### Schema departures from formalRAG

- `domain` enum = the nine domains in §6.
- Keep: `financeCaseStudy`, `rigorFlag`, `references` (incl. `documentation`), `connections`.
- Cross-site arrays: `formalml{Prereqs,Connections}`, `formalstatistics{…}`, `formalcalculus{…}`, and
  **new** `formalrag{Prereqs,Connections}`. `crossSiteRef.site` enum gains `'formalrag'`.
- Reciprocals: each sibling must eventually add `'formalnlp'` to its own `crossSiteRef` enum and audit
  script. Per formalRAG's convention, those reciprocal links are **deferred** and added per-sibling as
  each formalNLP topic ships; the audit's missing-reciprocal warnings are the expected state.
  **Superseded by §10.4:** sibling-side reciprocals are batched into one PR per sibling at launch.

## 5. Decided: visual identity — "Marginalia"

Selected from three mockups (A · Marginalia). Philological, humanities-forward: language as a
scholarly object. Distinct from formalML (teal/indigo), formalRAG (indigo/violet), and
formalCalculus/formalStatistics (blue/emerald).

| Token | Light value |
|---|---|
| `--color-bg` | `#FBF8F1` (warm paper) |
| `--color-text` | `#211A17` |
| `--color-accent` | `#9F1D35` (oxblood) — wordmark "NLP", domain labels, definitions |
| `--color-accent-secondary` | `#A16207` (manuscript gold) — theorems, finance badge |
| `--color-border` | `#E6DCCB` |
| `--color-definition-bg` / border | `#F6E7E8` / `#9F1D35` |
| `--color-theorem-bg` / border | `#F7EFDC` / `#A16207` |
| `--font-serif` | **Crimson Pro** (body, headings, theorem text) |
| `--font-sans` | Inter (nav, labels, UI) — shared with siblings |

Wordmark: "formal" in text color + "**NLP**" in oxblood. A subword-token strip motif
(`earn | ings | ▁call | ▁tone`) is available as a recurring visual signature.

**Left to builder:** the dark-mode palette (formalRAG ships a ThemeToggle, so one is required — keep
oxblood/gold recognizable on a warm near-black, checked for WCAG AA), the favicon, and the KaTeX/code
color pairing.

## 6. Decided: curriculum shape (draft topic list)

Nine domains, ~3 topics each. Slugs and titles are **draft** — each is finalized in its Phase A brief.
Every cross-site slug below was verified as `status: published` on its sibling as of 2026-10-22,
**except** formalStatistics `empirical-bayes` and `hidden-markov-models` (both `draft`). Topics citing
those must wait for them to publish or cite an alternative.

| # | Layer | Domain (`domain`) | Draft topics (slug) | Key cross-site prereqs |
|---|---|---|---|---|
| 1 | Foundations | Text as Data (`text-as-data`) | `zipf-heaps-vocabulary-statistics`; `subword-tokenization-bpe-unigram` | ML `shannon-entropy`, `minimum-description-length`; Stats `discrete-distributions` |
| 2 | Foundations | Language Modeling (`language-modeling`) | `n-gram-models-smoothing` (Laplace → Good–Turing → Kneser–Ney); `perplexity-cross-entropy`; `neural-language-models` | Stats `maximum-likelihood`, `empirical-bayes`; ML `kl-divergence`; Calc `jacobian` |
| 3 | Representations | Representations (`representations`) | `distributional-semantics-ppmi-svd`; `word2vec-implicit-factorization`; `glove-embedding-geometry`; `contextual-embeddings-masked-lm` | ML `svd`; RAG `pmi-retrieval-value`, `vector-space-model-tfidf`, `negative-sampling-hard-negatives`, `hypersphere-vmf-geometry`, `dense-retrieval-dual-encoders` |
| 4 | Representations | Attention & Transformers (`attention-transformers`) | `attention-as-kernel-smoothing`; `positional-encodings-geometry` (sinusoidal, RoPE); `transformer-block-residual-stream`; `scaling-laws` | ML `kernel-regression`, `double-descent`; Calc `fourier-series` |
| 5 | Generation | Decoding & Generation (`decoding-generation`) | `beam-search-decoding`; `sampling-temperature-nucleus`; `language-model-calibration` | ML `shannon-entropy`, `conformal-prediction`; RAG `selective-generation-abstention` |
| 6 | Tasks | Classification & Sentiment (`classification-sentiment`) | `linear-text-classifiers` (generative vs. discriminative); `lexicon-sentiment-loughran-mcdonald`; `aspect-based-sentiment`; `fine-tuning-low-rank-adaptation` | ML `naive-bayes`, `pca-low-rank`; Stats `generalized-linear-models` |
| 7 | Tasks | Information Extraction & KGs (`extraction-knowledge-graphs`) | `sequence-labeling-hmm-crf`; `named-entity-recognition`; `relation-extraction-distant-supervision`; `knowledge-graph-embeddings` (TransE/RotatE) | Stats `hidden-markov-models`; ML `message-passing`; RAG `graphrag-community-detection` |
| 8 | Tasks | Text Mining (`text-mining`) | `topic-models-lda`; `collocations-keyword-extraction` (Dunning LR, TextRank) | ML `variational-inference`, `random-walks`; Stats `likelihood-ratio-tests-and-np` |
| 9 | Evaluation | Evaluation (`nlp-evaluation`) | `overlap-metrics-bleu-rouge`; `significance-testing-for-nlp` | Stats `bootstrap`; RAG `significance-testing-calibration` |
| — | Capstone | (domain TBD by builder) | `capstone-earnings-call-nlp` — tokenize → classify tone → extract entities/relations → KG over a filing/call corpus | ties the finance thread together |

Count: 28 topics + capstone. The `/paths` page groups tracks by **layer** (Foundations →
Representations → Generation → Tasks → Evaluation), as formalRAG's `curriculum.ts` does.

**Overlap watch-list** (each Phase A brief must state scope against the sibling):
`significance-testing-for-nlp` vs. RAG `significance-testing-calibration`;
`distributional-semantics-ppmi-svd` vs. RAG `pmi-retrieval-value`;
`contextual-embeddings-masked-lm` vs. RAG `dense-retrieval-dual-encoders`;
`language-model-calibration` vs. RAG `selective-generation-abstention`;
`n-gram-models-smoothing` vs. RAG `query-likelihood-language-models`.

**Signature geometric hooks** the curriculum is built around:
softmax attention = Nadaraya–Watson with an exponential kernel; SGNS = implicit factorization of
shifted PMI (Levy & Goldberg 2014); RoPE = rotations in 2-D planes; temperature = Gibbs/Boltzmann
annealing; TransE = translation in embedding space; Kneser–Ney ≈ an approximation to a hierarchical
Pitman–Yor prior.

## 7. Per-topic workflow

Identical to formalRAG's `STARTER-PROMPT.md`: Phase A produces `docs/plans/formalnlp-<slug>-brief.md`
and `notebooks/<slug>/` (verified, exit 0) before any MDX; Phase B builds MDX + viz against the
notebook (viz ↔ Python numbers mirrored to the decimal), wires the curriculum graph and cross-site
arrays, runs `astro sync` → `validate` → `audit:cross-site` → DOM check (zero `.katex-error`), then
PR. Prose is self-approved per standing authorization.

**Notebook data constraint (new for NLP):** corpora must be small, license-clean, and shippable in the
repo or generated synthetically (CPU < 60 s). No model downloads larger than a small tokenizer or
embedding table; transformer topics use toy-scale models implemented from scratch in NumPy/PyTorch-CPU.

## 8. Launch and sequencing

- **Build order:** site scaffold → Foundations → Representations → Attention & Transformers →
  Decoding → Tasks → Evaluation → capstone, but the **prerequisite DAG wins** where it disagrees
  (e.g., `contextual-embeddings-masked-lm` ships after `transformer-block-residual-stream`). *(Builder's
  call.)*
- **Announce** only when all 28 topics + capstone are published (§10.2); until then the site is live
  but `noindex` (§10.3).
- **Shared finance corpus generator** is built right after the scaffold, before the first
  `financeCaseStudy` topic (§10.1).
- **YouTube:** formalNLP becomes a fifth playlist, queued **after** the four current series per the
  2026-10-22 recording order in `CHANNEL-PLAN.md`. No scripting/animation work until queued.

## 9. Left to the builder

Exact slugs/titles; per-topic viz designs; the dark palette and favicon; capstone domain assignment;
whether to factor shared formal\* components into a package (default: copy, as siblings do); Vercel
project setup and DNS for formalnlp.com; adding `.superpowers/` to `.gitignore`. Note that
`audit-cross-site-links.mjs` assumes its own repo occupies the `formalml` slot and covers three sites —
it must be generalized to five sites with formalNLP as self, not copied verbatim.

## 10. Decisions from the builder check (2026-10-22)

1. **Finance data — synthetic + public SEC.** A shared, seeded synthetic finance-text generator
   (built once, early, imported by every `financeCaseStudy` notebook and the capstone) plus short
   public EDGAR filing excerpts committed to the repo. The Loughran–McDonald Master Dictionary is
   **cited and fetched at runtime, never committed** (non-commercial license vs. a DataSalt-owned public
   repo); harness asserts run against a small in-repo stand-in lexicon so notebooks stay offline and
   < 60 s. No earnings-call transcripts are committed.
2. **Announcement bar — all 28 topics + capstone published.**
3. **Pre-launch visibility — live, noindex.** formalnlp.com deploys from `main` from the first merge
   with site-wide `noindex` (meta robots + `robots.txt` disallow) and no links from sibling sites. The
   noindex flag is removed as part of the announcement.
4. **Sibling reciprocals — batched at launch.** formalNLP topics carry outward cross-site arrays as they
   ship. At announcement, one PR per sibling (formalML, formalStatistics, formalCalculus, formalRAG)
   adds `'formalnlp'` to its `crossSiteRef` enum, the `formalnlp{Prereqs,Connections}` arrays and
   reciprocal entries, audit-script coverage, and a nav/footer link. No sibling-repo changes before
   launch.
