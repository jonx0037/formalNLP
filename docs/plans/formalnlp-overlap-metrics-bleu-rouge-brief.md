# Brief — `overlap-metrics-bleu-rouge`

Topic 27 of 29, domain `nlp-evaluation`. Phase A is complete: the harness `notebooks/overlap-metrics-bleu-rouge/overlap_metrics_bleu_rouge.py`
passes 20/20 in about 15 s (and under `python -W error`), and `01_overlap_metrics_bleu_rouge.ipynb` executes cleanly.
Every number below is quoted to 6 decimals from the harness's `summary()`.

**Decisions (Jonathan, 2026-10-26).** Plan mode:
- Both testbeds: finance fact summaries graded exactly, plus Topic 14's decoding prompts.
- Soft matching on Topic 7's SGNS vectors.
- Correlation with judgments here; intervals and tests in Topic 28.
- Claude writes every body.

Scaffold checkpoint:
- Writers use only in-vocabulary forms (direction verbs, or the stand-in lexicon's adjective pairs).
- A company opener in every summary.
- Prop 7 reframed as noise versus bias.
- 64 MBR samples with a vectorized BLEU+1 utility.
- Beam search at k = 16 only.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Overlap Metrics: BLEU and ROUGE |
| `{{SUBTITLE}}` | A candidate and a reference make a grid of matches. Clipping is a matching on it, ROUGE-L is the longest chain through it, and both move by at most a few n-grams per edited word. So they cannot tell a paraphrase from a fact flip, and the brevity penalty is what makes the mode lose to minimum Bayes risk. |
| `{{SLUG}}` | `overlap-metrics-bleu-rouge` |
| `{{DOMAIN}}` | `nlp-evaluation` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[n-gram-models-smoothing, beam-search-decoding]` (the graph's incoming edges) |
| `{{FORMALML_PREREQS}}` | none |
| `{{FORMALRAG_PREREQS}}` | `rank-fusion-rrf` (Def 2: Kendall-τ distance, used to read τ) |
| `{{STATS_CALC_PREREQS}}` | formalStatistics `expectation-moments` (Thm 13 Jensen, Example 7 AM–GM: BLEU's geometric mean is at most the arithmetic mean of its precisions); formalCalculus `probability-and-union-bound` (Thm 8, uniform Hoeffding over a finite class: Thm 4(c)) |
| connections (formalRAG) | `faithfulness-groundedness` and `llm-as-judge-ragas` (judge-based faithfulness, the alternative to reference overlap); `significance-testing-calibration` (paired tests, Topic 28's ground) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/beam-search-decoding/`, `notebooks/collocations-keyword-extraction/`, `notebooks/word2vec-implicit-factorization/` |

**Sibling check (2026-10-26, every sibling's `origin/main`).** No sibling covers any of these:
- BLEU, ROUGE, chrF, or BERTScore;
- LCS or edit distance;
- MBR decoding;
- metric-versus-judgment correlation.

Every slug cited above is `status: published` on main. formalStatistics' only "rouge|bleu" hits are base64 image data.

**Scope against siblings.**
- formalRAG `faithfulness-groundedness` and `llm-as-judge-ragas` grade generated text against the *source*, with a judge.
  This topic grades it against *references*, by surface overlap, and shows what that cannot see. The page names judges
  as the response in one paragraph and proves nothing about them.
- formalRAG `significance-testing-calibration` and Topic 28 own confidence intervals and paired tests. Here, τ values
  are descriptive.

## 1. Section outline (geometric first)

1. **The match grid.** Candidate tokens down the rows, reference tokens across the columns, a dot where they are equal.
   The lab's first tab shows it. Unigram matches, clipped bigram pairs, and the LCS chain are all features of this one
   picture.
2. **Clipping** (Def 1, Prop 1).
3. **BLEU** (Def 2, Thm 1, Prop 2, Prop 3, Prop 4): the length envelope, corpus versus sentence, zero frequency and
   smoothing, more references.
4. **ROUGE** (Def 3, Thm 2): the LCS as the longest path.
5. **What overlap cannot see** (Thm 3, Cor 1, Prop 5).
6. **Character and soft matching** (Def 4, Def 5, Prop 6).
7. **The metric as a decision rule** (Thm 4): MBR versus the mode on Topic 14's prompts.
8. **Metrics and judgments** (Prop 7): noise averages away, bias does not.
9. **Finance case study** (`<FinanceCaseStudy>`): the answer key. Sections 5, 6, and 8 quote it, but the system table
   and the flipped-versus-extract headline live here.
10. Implementation note, then references.

## 2. Statements (numbered per type, in page order; the same numbers in the `.py` and the notebook)

**Notation.**
- y is the candidate, R = {r₁, …, r_K} the references.
- c_x(g) is the count of n-gram g in x.
- |y| = c, and r is the effective reference length.
- N = 4, with uniform weights 1/N.

**Def 1 (clipped count, modified precision).**
- m_n(y, R) = Σ_g min(c_y(g), max_k c_{r_k}(g)).
- p_n = m_n/t_n, where t_n = max(c − n + 1, 0).
- Example first: "the" seven times against "the cat sat on the mat" has m₁ = 2, so p₁ = 2/7.

**Prop 1 (clipping is a maximum matching).**
- Statement:
  - Give each n-gram g max_k c_{r_k}(g) reference slots, and join each candidate occurrence of g to g's slots.
  - The maximum matching has size m_n.
- Proof:
  - The graph is a disjoint union of complete bipartite graphs K_{c_y(g), M(g)}.
  - Each component's maximum matching has size min(c_y(g), M(g)).
- Harness: scipy Hopcroft–Karp on 300 random instances; the lab's pairing is a valid matching of that size.

**Def 2 (BLEU).**
- BLEU = BP · exp((1/N) Σ log P_n).
- P_n = Σ_i m_n(i)/Σ_i t_n(i) is pooled over the corpus.
- BP = 1 if c > r, else e^{1 − r/c}.
- r = Σ_i r_i, where r_i is the reference length closest to c_i (the shorter one on a tie, sacrebleu's rule).
- BLEU is 0 if any P_n = 0.
- Cross-checks against sacrebleu 2.6.0 (`tokenize="none"`), all to 10⁻¹²:
  - corpus BLEU;
  - sentence BLEU with smoothing none, add-k, and exp.

**Thm 1 (length envelope).**
- Statement:
  - (a) One segment with one reference: BLEU ≤ g(c/r), where g(x) = e^{1−1/x} for x ≤ 1 and g(x) = 1/x for x ≥ 1.
  - (b) A corpus with one reference per segment: the same left branch, and x^{−1/N} on the right.
  - (c) With several references: BLEU ≤ BP · min(1, ρ/c)^{1/N}, where ρ = Σ_i Σ_g max_k c_{r_{ik}}(g).
  - Both envelopes have a unique maximum, 1, at x = 1.
- Proof:
  - p_n ≤ 1 always, and m₁ ≤ ρ (≤ r with one reference).
  - Hence P₁ ≤ ρ/c.
  - For one segment with one reference, m_n ≤ r − n + 1, and (r − n + 1)/(c − n + 1) ≤ r/c iff c ≥ r.
  - Combine with BP.
- Tightness: every reference prefix has all P_n = 1, so it sits exactly on the left branch.
- Harness sweep: the first reference per prompt, cut or extended with an unmatched filler. At x = 0.249260 the sweep
  gives BLEU 0.049200, equal to the bound; at x = 2 it gives 0.434984 against the corpus bound 0.840896.
- Decoding points (single reference, x and BLEU):

  | System | x = c/r | BLEU |
  |---|---|---|
  | mode | 0.329142 | 0.087164 |
  | beam-16 | 0.340237 | 0.093313 |
  | norm-16 | 0.713018 | 0.191189 |
  | sample | 1.079882 | 0.131487 |
  | MBR | 0.654586 | 0.220242 |

- Attribution: the brevity penalty and best-match length are Papineni et al. (2002). The envelope is ours.

**Prop 2 (corpus BLEU is not the mean of sentence BLEU).**
- Statement:
  - (a) Each pooled P_n is a mediant: it lies between the smallest and largest segment ratio.
  - (b) A system can win every sentence and lose the corpus.
- Instance, with references of ten words each:

  | System | Segment 1 | Segment 2 | Sentence BLEU | Corpus BLEU |
  |---|---|---|---|---|
  | A | one middle word substituted | one middle word substituted | 0.658037, 0.658037 | 0.658037 |
  | B | five unmatched words appended | stops halfway | 0.626284, 0.367879 | 0.699487 |

  B's two length errors cancel in the pooled BP (c = r = 20).
- On the finance systems, corpus BLEU and mean sentence BLEU+1 disagree on two pairs: (textrank-3, lead-3) and
  (textrank-3, random-3).

**Prop 3 (zero frequency; smoothing is shrinkage).**
- Statement:
  - (a) If m_N = 0, unsmoothed BLEU = 0, whatever the lower orders.
  - (b) Add-k smoothing (n ≥ 2): (m + k)/(t + k) = λ p̂ + (1 − λ)·1, with λ = t/(t + k). This is Topic 3's Prop 3
    (shrinkage is interpolation) with a prior at perfect precision, so short candidates drift toward precision 1.
- Example: "costs fell and revenue rose" against "revenue rose and costs fell" has five unigram matches, no 4-gram, and
  BLEU 0.
- Share of finance segments with no matched 4-gram:

  | System | Share |
  |---|---|
  | writer | 0.244444 |
  | flipped | 0.344444 |
  | terse | 0.366667 |
  | extract | 0.933333 |
  | lead-3 | 0.966667 |

- Attribution: add-one for n ≥ 2 is Lin and Och (2004, §4), Chen and Cherry's (2014, §2.2) Smoothing 2. "exp" is
  their Smoothing 3 (NIST): the k-th zero-match order gets match count 1/2^k.

**Prop 4 (more references).**
- Statement:
  - (a) m_n is monotone in R.
  - (b) BLEU is not.
- Instance:
  - The candidate has 8 words; reference 1 is its first 6 words, so BLEU = 0.680375.
  - Add reference 2, nine unseen words. It is closer in length, so r goes from 6 to 9 and BP falls by e^{1−9/8}.
  - BLEU becomes 0.600429.

**Def 3 (ROUGE-N, ROUGE-L).**
- ROUGE-N: precision, recall, and F of the clipped match against one reference.
- ROUGE-L: P = LCS/|y|, R = LCS/|r|, and F (rouge-score uses β = 1).
- Several references: the best F over references (rouge-score's `score_multi`). Lin's Eq. (1) sums over references,
  but §2.1 scores multiple references by the maximum pairwise score and jackknifes.
- Cross-check: rouge-score 0.1.2, P/R/F to 10⁻¹², on random instances and on the finance summaries.

**Thm 2 (the LCS is a longest path).**
- Statement:
  - (a) Common subsequences are exactly the chains of marked cells increasing in both coordinates. The LCS is the
    longest path in the grid's DAG, and the dynamic program L[i, j] computes it.
  - (b) LCS ≤ m₁ (clipped unigram matches), so ROUGE-L's P, R, and F are each ≤ ROUGE-1's.
- Proof of (a): optimal substructure. Case split on y_i = r_j: if they are equal, some LCS uses both; otherwise one of
  the two can be dropped.
- Proof of (b): a chain is a matching between equal tokens, and Prop 1 bounds every such matching by m₁.
- Harness: the DP equals brute-force enumeration (300 instances), and the backtracked path is a valid chain.

**Thm 3 (Lipschitz in substitutions).**
- Statement: if z differs from y in k positions,
  - |m_n(z) − m_n(y)| ≤ nk;
  - |LCS(z, r) − LCS(y, r)| ≤ k.
  - Hence |Δp_n| ≤ nk/t_n and |ΔROUGE-N recall| ≤ nk/(|r| − n + 1).
- Proof (clipped count):
  - Each position lies in at most n windows.
  - Changing one window removes one occurrence of g and adds one of g', which moves Σ min(c(g), M(g)) by at most 1.
- Proof (LCS): delete the k positions from the chain.
- BLEU itself is not Lipschitz near P_n = 0, because of the log. State the bound on the precisions.

**Cor 1 (fact flips).**
- Statement: a summary whose m facts each hang on one word loses at most nm/(|r| − n + 1) of ROUGE-N recall when every
  fact is reversed, while fact F1 goes from 1 to 0.
- Harness, all 90 documents × 4 references:
  - every flipped pair differs in exactly m words;
  - no realized drop exceeds the bound;
  - writer F1 is 1 and flipped F1 is 0 everywhere.

**Prop 5 (chunk permutations; Callison-Burch et al. 2006, §3.1, made precise).**
- Statement:
  - Cut y wherever its bigram occurs in no reference, and permute the chunks.
  - (a) No m_n decreases. Within-chunk n-grams are preserved, and every lost n-gram contained an unmatched bigram.
  - (b) BLEU is unchanged unless a new junction bigram happens to match, in which case it rises.
  - (c) ROUGE-L generally changes.
- Callison-Burch et al. argue (a)–(b) in prose: reordering "will not reduce" the matches, and their footnote 2 concedes a
  permutation may raise them. Their Eq. (1) is a lower bound on the variant count, (k − b)!.
- Harness, 200 random orders per summary:
  - 7.322222 chunks per writer summary (mean log₁₀ of the order count: 4.175277, about 15,000 orders);
  - 0.785167 of sampled orders keep every m_n exactly;
  - none lowers any m_n;
  - ROUGE-L takes more than one value for 0.988889 of summaries.

**Def 4 (chrF).**
- Character n-grams for n = 1..6, with whitespace removed.
- P and R averaged over the orders both strings reach (sacrebleu's effective order), then F_β with β = 2.
- Cross-check: sacrebleu's `sentence_chrf`, to 10⁻¹².
- Popović (2015, Eq. 1) proposed β = 3 as a preliminary value; β = 2 comes from Popović (2016, §3), which sacreBLEU's source cites.

**Def 5 (greedy soft matching).**
- Unit vectors from Topic 7's `SGNSConfig("L2", 25)` W.
- P = mean over y of the best cosine in r; R = mean over r of the best cosine in y; F is their harmonic mean.
- Out-of-vocabulary pairs score 1 if equal, else 0.
- This is BERTScore's form (Zhang et al. 2020), without contextual states, idf, or rescaling.

**Prop 6 (soft matching moves less than exact matching).**
- Statement: one substitution u → u' changes P by at most ‖u − u'‖/|y| and R by at most ‖u − u'‖.
- Proof: |⟨u − u', v⟩| ≤ ‖u − u'‖ for unit v, and each max moves by at most that much.
- Numbers:
  - cos(rose, fell) = 0.994626, and "fell" is "rose"'s nearest neighbor;
  - ‖rose − fell‖ = 0.103670;
  - flipping every fact moves corpus soft F from 0.933868 to 0.932539, while ROUGE-1 moves from 0.667339 to 0.638430;
  - no realized change exceeds the bound (excess 0).

**Thm 4 (MBR).**
- Statement: for a distribution p and a utility U ∈ [0, 1], the Bayes decision over a hypothesis set H is
  argmax_h Σ_y p(y) U(h, y).
  - (a) With U = 1[h = y], it is the mode, Topic 14's search target.
  - (b) With U = BLEU+1, it need not be the mode.
  - (c) If S samples estimate every expected utility, S ≥ 2 log(2|H|/δ)/Δ² makes the Monte Carlo choice exact with
    probability ≥ 1 − δ, where Δ is the gap between the two best. This is Hoeffding plus a union bound
    (formalCalculus Thm 8).
- Toy distribution:
  - "margins held" has probability 0.2 and is the mode;
  - eight edits of a seven-word sentence have 0.1 each.
  - Expected BLEU+1: the mode 0.244395, the best edit 0.549638. The best edit is the six-word truncation.
  - Gap Δ = 0.011121, so S = 95,180; Monte Carlo at that S picks the exact choice in 400/400 trials.
- Corpus: 188 prompts with 2.356383 references each, of mean length 7.298203 words.

  | System | BLEU | Mean length | Share empty | BP |
  |---|---|---|---|---|
  | mode | 0.110458 | 2.367021 | 0.196809 | 0.147400 |
  | beam-16 | 0.117668 | 2.446809 | 0.186170 | 0.162096 |
  | norm-16 | 0.212829 | 5.127660 | 0.106383 | 0.668653 |
  | one sample | 0.148739 | 7.765957 | 0.005319 | 1.000000 |
  | BLEU-MBR | 0.254129 | 4.707447 | 0.000000 | 0.608934 |

  - MBR uses 64 samples, with Topic 14's three outputs added as hypotheses.
  - The most frequent sample (exact-match MBR) equals the exact mode on 0.542553 of prompts.
- Link: Topic 14's Thm 4 (the end mark's pull) is why the mode is short; the BP is what punishes it.

**Prop 7 (averaging removes noise, not bias).**
- Statement:
  - Segment score s_ij = b·q_ij + β_j + ε_ij, with independent noise and per-system bias β_j.
  - System means converge almost surely to μ_j = b·q̄_j + β_j.
  - So τ(s̄(n), q̄) → τ(μ, q̄). This equals 1 iff, for every pair with q̄_j < q̄_k, b(q̄_k − q̄_j) > β_j − β_k.
- Model check: q = (0, ¼, ½, ¾, 1), with bias 0.6 on the middle system. τ_∞ = 0.6, and the Monte Carlo means rise
  through n = 1, 10, 1000 to within 0.02 of it.
- Corpus, 10 systems × 90 documents. Systems differing only by flips (writer, noisy-0.1, noisy-0.3, noisy-0.5, flipped)
  get τ = 1 on every metric.

  | Metric | Segment τ (mean over documents) | System τ (all 10 systems) |
  |---|---|---|
  | BLEU | 0.395415 | 0.522862 |
  | ROUGE-1 | 0.378688 | 0.568329 |
  | ROUGE-2 | 0.360149 | 0.522862 |
  | ROUGE-L | 0.383068 | 0.568329 |
  | chrF | 0.342678 | 0.522862 |
  | soft F | 0.385884 | 0.568329 |

- System τ against documents per system, n ∈ (1, 2, 5, 10, 20, 45, 90):

  | Metric | n = 1 | n = 90 |
  |---|---|---|
  | BLEU | 0.375486 | 0.522862 |
  | ROUGE-1 | 0.338074 | 0.568329 |
  | soft F | 0.361619 | 0.568329 |

- Every metric reverses (extract, flipped); BLEU reverses 10 pairs.

## 3. Findings (harness, 6 decimals)

(a) **The headline.** The all-flipped writer summary (fact F1 0) outscores the fact-bearing extract (fact F1 1) on every
metric.

| Metric | Flipped | Extract | Share of documents where flipped wins |
|---|---|---|---|
| BLEU | 0.288952 | 0.024596 | 1.000000 |
| ROUGE-1 | 0.638430 | 0.244612 | 1.000000 |
| ROUGE-2 | 0.341398 | 0.078313 | 0.988889 |
| ROUGE-L | 0.511657 | 0.190666 | 1.000000 |
| chrF | 0.596679 | 0.349003 | 1.000000 |
| soft F | 0.932539 | 0.747930 | 1.000000 |

(b) **The flip slider** (rates 0, 0.1, …, 1, with common uniforms). Fact F1 falls from 1.000000 to 0.000000 while the
metrics barely move:

| Metric | Rate 0 | Rate 1 |
|---|---|---|
| BLEU | 0.384366 | 0.288952 |
| ROUGE-1 | 0.667339 | 0.638430 |
| ROUGE-L | 0.555662 | 0.511657 |
| soft F | 0.933868 | 0.932539 |

(c) **The testbed.** 90 documents (`generate_corpus(100, 1)`, every type) with at least two facts.
- 3.622222 facts per document.
- 4 references each, 15.975000 words on average.
- A fact is a category with the sign of its summed aspect polarities. Categories whose polarities cancel state no fact.

(d) **The decoding thread** (Thm 4 table). BLEU-MBR is best (0.254129) at 4.707447 words. The mode is the worst and
shortest (0.110458 at 2.367021 words, with 0.196809 of prompts empty).

(e) **Correlation** (Prop 7 table). Segment τ is 0.34–0.40. System τ rises to 0.52–0.57 at 90 documents and stops
there, because bias, not noise, sets the ceiling.

## 4. The lab: `OverlapMetricsLab` (5 tabs; every number from the payload, `src/components/viz/data/overlap-metrics-lab.json`)

1. **Match grid.**
   - Document `synth-0017` (Marrow Freight), with facts revenue −, liquidity +, liabilities −.
   - Chosen by rule: the first document with 3–4 facts where the flipped summary's BLEU+1 beats the extract's and the
     extract has at most 40 words.
   - A toggle picks writer, flipped, or extract. Layers: unigram dots, clipped unigram and bigram pairs, and the LCS
     chain.
   - A score readout: BLEU+1, R-1, R-2, R-L, chrF, soft, and fact F1.
2. **Length envelope.** Both envelope branches (drawn from the formula), the sweep points, and the five decoding systems
   at single-reference coordinates.
3. **Flip vs. paraphrase.** A slider over the flip rate shows six metric bars plus fact F1, with the extract as a dashed
   reference line.
4. **Correlation.**
   - Per metric: segment τ, system τ, and flip-family τ.
   - The τ-versus-n curve.
   - The 10-system table (scores and fact F1).
5. **MBR vs. mode.**
   - Corpus BLEU, length, and BP per decoding system.
   - Five example prompts, the first five (sorted) whose exact mode is empty and whose MBR choice is not: indices
     0, 1, 2, 11, 13. Each shows its references and outputs with sentence BLEU+1.

Mobile: `scrollWidth === 375` in every tab. The grid scrolls inside a clipped container. Readouts use flex-wrap rows.

## 5. Checkpoint answers (Jonathan)

- 2026-10-26, scaffold: approved with all four recommendations (writer vocabulary, company opener, Prop 7 as noise
  versus bias, MBR with 64 samples and beam at k = 16 only).
- 2026-10-27, Phase A: approved; Phase B started.

## 5b. rigorFlag (draft)

> BLEU's agreement with human judgment is weak outside system-level machine translation (Reiter 2018), and nothing here
> measures human judgment. The "judgments" are exact fact F1 against a synthetic generator's aspect gold, and the
> references are templated paraphrases written from the same facts, so every fact hangs on one word by construction.
> Real references differ in content, not only in wording. The soft match uses static SGNS vectors, not BERTScore's
> contextual states, idf weights, or rescaling. The τ values are descriptive: no interval or test is claimed (Topic 28).
> MBR is estimated from 64 samples of a trigram with smoothed sentence BLEU as the utility, which Eikema and Aziz (2020,
> n. 9) avoid because smoothing biases BLEU's statistics; Thm 4(c)'s Hoeffding count is a loose sufficient bound. Thm
> 1's envelope needs one reference per segment; with several, the bound uses the reference union ρ.

## 6. Notebook cell map (24 cells, `01_overlap_metrics_bleu_rouge.ipynb`)

| Cells | Content |
|---|---|
| 1–2 | Title and import |
| 3–4 | Clipping, Prop 1 |
| 5–6 | Library cross-checks |
| 7–8 | Thm 1 sweep |
| 9–10 | Prop 2 |
| 11–12 | Props 3–4 |
| 13–14 | Thm 2 |
| 15–16 | Thm 3, Cor 1, Prop 5, grid example |
| 17–18 | Prop 6 |
| 19–20 | Thm 4 toy and corpus table |
| 21–22 | Prop 7 |
| 23–24 | Case-study table, full harness |

## 7. References (verified 2026-10-26 from the PDFs and Anthology records; Chicago notes-bibliography)

- **Papineni, Roukos, Ward, Zhu (2002).** "Bleu: A Method for Automatic Evaluation of Machine Translation." ACL 2002,
  311–18, P02-1040, DOI 10.3115/1073083.1073135.
  - The paper numbers no equations.
  - §2.1: modified precision, with Example 2 ("the" ×7 = 2/7) and footnote 2 (clipping at the max count in any single
    reference).
  - §2.2.2: best match length, with BP computed over the corpus because sentence-level BP punishes short sentences
    harshly.
  - §2.3: BP, and N = 4 with uniform weights.
  - §5, Figs. 5–6: correlations 0.99 and 0.96 over five systems. Footnote 4: BLEU need match humans only over a corpus.
- **Lin (2004).** "ROUGE: A Package for Automatic Evaluation of Summaries." *Text Summarization Branches Out*, 74–81,
  W04-1013.
  - Eq. (1): ROUGE-N.
  - §2.1: multiple references by max, then jackknifing.
  - §3.1, Eqs. (2)–(4): ROUGE-L, with X the reference.
  - §5, Eqs. (16)–(18): ROUGE-S.
- **Lin and Och (2004).** "ORANGE: A Method for Evaluating Automatic Evaluation Metrics for Machine Translation." COLING
  2004, 501–7, C04-1072. §4: add one to the n-gram hit and total counts for n > 1 (BLEUS).
- **Callison-Burch, Osborne, Koehn (2006).** "Re-evaluating the Role of Bleu in Machine Translation Research." EACL 2006,
  249–56, E06-1032.
  - §3.1 and Eq. (1): (k − b)! permutations, a lower bound. Footnote 2: a permutation can raise the score.
  - §4, Figs. 2–4: human-judgment counterexamples.
- **Chen and Cherry (2014).** "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU." WMT 2014, 362–67,
  W14-3346, DOI 10.3115/v1/W14-3346. §2.2: Smoothing 2 is Lin–Och (Eqs. 6–7); Smoothing 3 is NIST.
- **Popović (2015).** "chrF: Character n-gram F-score for Automatic MT Evaluation." WMT 2015, 392–95, W15-3049,
  DOI 10.18653/v1/W15-3049. Eq. (1); §3.1: n = 6, spaces ignored, β = 3 preliminary.
- **Popović (2016).** "chrF Deconstructed: Beta Parameters and n-gram Weights." WMT 2016, 499–504, W16-2341. §3: β = 2,
  uniform weights.
- **Zhang, Kishore, Wu, Weinberger, Artzi (2020).** "BERTScore: Evaluating Text Generation with BERT." ICLR 2020,
  arXiv 1904.09675. §3 has no numbered equations: "BERTScore" (greedy R, P, F), "Importance Weighting" (idf), and
  "Baseline Rescaling".
- **Post (2018).** "A Call for Clarity in Reporting BLEU Scores." WMT 2018, 186–91, W18-6319, DOI 10.18653/v1/W18-6319.
  §2.2, Table 1: reference preprocessing moves scores by up to 1.8, about 1.0 on average.
- **Kumar and Byrne (2004).** "Minimum Bayes-Risk Decoding for Statistical Machine Translation." HLT-NAACL 2004, 169–76,
  N04-1022.
  - Eq. (3): the MBR rule. Eqs. (5)–(6): 0/1 loss gives MAP.
  - §2.1: loss 1 − BLEU, sentence-level and unsmoothed.
- **Eikema and Aziz (2020).** "Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation."
  COLING 2020, 4506–20, 2020.coling-main.398, DOI 10.18653/v1/2020.coling-main.398.
  - §7.4, Eq. (1): sampling-based MBR, with METEOR as the utility.
  - Footnote 9 rejects smoothed BLEU.
- **Reiter (2018).** "A Structured Review of the Validity of BLEU." *Computational Linguistics* 44 (3): 393–401,
  DOI 10.1162/coli_a_00322.
  - 284 correlations from 34 papers.
  - BLEU is supported only for diagnostic, system-level MT evaluation (§3, Figs. 4–5; §5).
- **Mihalcea and Tarau (2004).** "TextRank: Bringing Order into Text." EMNLP 2004, 404–11, W04-3252. §4.1: the sentence
  similarity, unnumbered.
- **Freitag, Grangier, Tan, Liang (2022).** "High Quality Rather than High Model Probability: Minimum Bayes Risk Decoding
  with Neural Metrics." *TACL* 10: 811–25, DOI 10.1162/tacl_a_00491.
