# Notebooks — the Python pillar

Every formalNLP topic ships a **notebook pillar** alongside its math and its interactive
viz. The pillar is two coupled artifacts per topic, living in `notebooks/<slug>/`:

| File | Role | Rule |
|---|---|---|
| `<slug_underscored>.py` | **Canonical reference implementation.** Importable, deterministic, CPU-only, runs end-to-end in well under 60 s. | Owns every number the topic or the viz depends on. Carries a verification harness of `assert`-based tests that encode **each pedagogical claim the topic makes** (limit theorems, monotonicity, the worked-example flip, cross-checks against a reference library). Running it as a script *is* the regression test for the topic. |
| `01_<slug_underscored>.ipynb` | **Narrative notebook.** Imports the `.py` and walks the topic section by section in markdown cells, running the harness so each claim renders as executed output. | Never redefines the math — it calls into `<slug_underscored>.py`. The topic's `notebookPath` frontmatter points here. Commit it **without** stored outputs (we execute to verify, not to snapshot). |

The directory name uses hyphens (`notebooks/attention-as-kernel-smoothing/`); the notebook filename uses
underscores (`01_attention_as_kernel_smoothing.ipynb`). The `.py` file uses the underscored slug
(`attention_as_kernel_smoothing.py`) so it imports cleanly.

## Why two artifacts

The `.py` is machine-verifiable and importable; the `.ipynb` is readable and teaches. Splitting
them lets the tested numbers live in one place while the prose narrative imports them — so prose
can never silently drift from verified output. `notebooks/finance-corpus/` is the reference exemplar
for the import pattern and harness structure (formalRAG's `notebooks/bm25/` for a full topic's cell
layout, until formalNLP's first topic ships).

## The shared finance corpus

`notebooks/finance-corpus/finance_corpus.py` is not a topic — it is the shared data layer for every
`financeCaseStudy` notebook and the capstone: a seeded synthetic corpus of finance documents with
tone labels, entity spans, and relations; a hand-written stand-in sentiment lexicon; and short
public EDGAR excerpts (see `edgar/SOURCES.md`). Import it; never fork it. The real
Loughran–McDonald dictionary is fetched at runtime by the topic that needs it and is never committed.

## The Viz ↔ Python invariant

Any corpus, weights, or worked-example numbers a topic's D3 viz displays are **mirrored to the
decimal** from the topic's `.py`, and the topic claims they match. The `.py` is the source of
truth: change a number there, then update the viz (and re-run the notebook) — never the reverse.

## Running

No shared venv — dependencies are declared per topic in the `.py` module docstring and supplied
ad hoc with `uv run --with …`:

```bash
# Run the reference implementation (asserts every claim, prints the demos):
uv run --with numpy --with scipy python notebooks/finance-corpus/finance_corpus.py

# Execute the narrative notebook top to bottom to verify it still runs (no outputs saved):
uv run --with numpy --with scipy --with jupyter --with ipykernel \
    jupyter execute notebooks/finance-corpus/01_finance_corpus.ipynb
```

Both must exit 0 before a topic ships. See `../STARTER-PROMPT.md` for where the notebook pillar
sits in the full topic-authoring workflow.
