# Brief — `contextual-embeddings-masked-lm`

**Contextual Embeddings and the Masked-LM Objective.** Roadmap #12. Prerequisites: `transformer-block-residual-stream`
(#11, PR #20) and `word2vec-implicit-factorization` (#7, PR #13), both published. This topic is a prerequisite of
`fine-tuning-low-rank-adaptation` (#15) and `named-entity-recognition` (#14). It is on the §6 overlap watch-list against
formalRAG `dense-retrieval-dual-encoders`.

**Phase A** (this brief and the verified notebook), 2026-10-24:
- The harness passed 18/18 at Phase A (20/20 after Phase B, §12) under numpy==2.5.4 and scipy==1.18.1 in ≈ 38 s, also with warnings promoted to errors.
- The narrative notebook (31 cells) executes with exit 0 in ≈ 41 s, including kernel start, and is committed without
  outputs.
- The lab's data file `src/components/viz/data/masked-lm-lab.json` is 52.3 kB, pinned by
  `test_viz_mirror_matches_payload`.

**Scope decisions** (Jonathan, 2026-10-24, on Claude's recommendations):
- **Both finance threads:** alias typing (ORG vs PRODUCT) and Topic 7's strong/weak promise.
- **The full theory spine:**
  - proper scoring;
  - Thm 2, the closed form under 80/10/10;
  - pseudo-likelihood;
  - Brook's lemma and compatibility.
- **A controlled HMM language with exact conditionals,** beside the finance corpus.
- **Authorship:** every function body is written by Claude for Jonathan's review, as in Topics 5–11.
- **Difficulty:** `advanced`, domain `representations`, `financeCaseStudy: true`.

**The spine.** Topic 7 gave each word type one vector. A polyseme's type-level PMI is a log-mixture of its senses'
(Prop 1), so a static map is capped at the majority sense (Cor 1). BERT replaces the type's vector with a function of the
sentence, and trains it by corruption:
- **What it converges to.** The MLM risk is entropy plus KL (Thm 1), and its minimizer has a closed form under the
  80/10/10 channel (Thm 2).
- **It is not a language model.** With one mask per sequence the objective is Besag's pseudo-likelihood (Thm 3). The
  conditionals of any joint pass Brook's four-cycle test (Thm 4, Cor 2), and the MLM's fail it.
- **What the vectors look like.** Occurrences spread with depth, inside an anisotropic cloud (Prop 2, Ethayarajh's
  measures).
- **What they can do that a static vector cannot.** Read the neighbors: a company versus its product, and the news a
  direction word reports. They do no more than the context determines, which is why Topic 7's strong/weak question still
  comes back at chance.

**Changes from the approved plan, each forced by a spike or the harness:**
1. **The polarity thread is reframed.** Topic 7's per-type probe on contextual means stays below two standard errors over
   chance: SGNS 0.571, layer 1 0.629, layer 2 0.571, with chance 0.5 ± 0.085. The spike explained why. Voting the other
   lexicon words near each occurrence gets 0.514 within ±8 tokens, 0.546 within ±32, and 0.571 over the whole document:
   the corpus does not write a lexicon word's polarity into its text. What the text *does* determine is the news an
   occurrence reports ("costs rose" is bad news, "revenue rose" good). A contextual probe reads that at 0.868 against a
   static cap of 0.559, so the case study is now about that, and Topic 7's question gets an honest answer rather than a
   fake win.
2. **The filing encoder trains on one concatenated stream, and every analyzed token is read at its window's center.** In
   the spike, with document-internal windows, layer 0 (E[x] + P_pos, context-free) typed *kestrel* at 0.924 by 5-fold CV.
   It did so through the position table alone: all 61 PRODUCT occurrences sat at window positions 22–24, because product
   mentions close a document and the last window was end-aligned. Centered reads make layer 0 exactly static (kestrel's
   self-similarity and MEV are 1.000). The leak is kept as a rigorFlag lesson: a probe uses any cue it is given.
3. **Windows of 32, not 64.** The doc-level polarity thread no longer needs long windows, and the training budget does.
4. **The polyseme's senses are chosen by a rule:** the heaviest state of each regime under π. The first draw (states 1
   and 5) gave sense shares 0.715/0.285. The rule gives 0.491/0.509.
5. **Float32 training and a matmul MLP backward.** Profiling showed Topic 11's `einsum` MLP backward at 4 ms of a 10 ms
   step, and the full filing loss at 50 ms/step before logits were restricted to selected rows. `mlp_backward` here
   equals `tb.mlp_backward` to 1e-12 (`test_fast_mlp_backward`). Gradient checks run in float64.
6. **The SGNS baseline is Topic 7's `SGNSConfig("L2", 25)`,** a ±2 window from its grid, taking 2.8 s. CONTROLLED is L8
   and takes ≈ 14 s, and alias typing and news polarity are local.
7. **The Bayes-optimal MLM is itself slightly incompatible** (median |δ| = 3.55e-4), so Cor 2's reading gains a
   sentence. At a `[MASK]`, the target conditions on a context that may be corrupted, so the set of targets is not the
   conditionals of one joint.
8. **The control is a shuffled-label control,** trained and scored on permuted labels. It replaces Hewitt and Liang's
   per-type control task, which is degenerate for the HMM's single polyseme. Their paper is cited for the idea.
9. **formalStatistics `multivariate-distributions` → `conditional-probability`.** The former's conditional result is
   Gaussian (its Thm 3), while Thm 2 here is Bayes' theorem (`conditional-probability` Thm 4) on a discrete channel.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Contextual Embeddings and the Masked-LM Objective |
| `{{SUBTITLE}}` | Hide a token and ask a bidirectional encoder to put it back: the objective's optimum is the posterior of the hidden token given its corrupted context, with a closed form under BERT's 80/10/10 corruption; its conditionals belong to no joint distribution; and the vectors it learns give each occurrence its own point, which reads what the neighbors determine and nothing more |
| `{{SLUG}}` | `contextual-embeddings-masked-lm` (= graph node id) |
| `{{DOMAIN}}` | `representations` |
| `{{DIFFICULTY}}` | `advanced` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[transformer-block-residual-stream, word2vec-implicit-factorization]` |
| `{{FORMALML_PREREQS}}` | `representation-learning`, `kl-divergence`; connection `pca-low-rank` |
| `{{FORMALRAG_PREREQS}}` | none; connections `dense-retrieval-dual-encoders`, `hypersphere-vmf-geometry`, `late-interaction-learned-sparse` |
| `{{STATS_CALC_PREREQS}}` | Stats `conditional-probability`, `maximum-likelihood`; Stats connection `bayesian-computation-and-mcmc`; Calc connection `jacobian` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/transformer-block-residual-stream/`, `notebooks/word2vec-implicit-factorization/`, `notebooks/finance-corpus/` |

- **Tags:** `contextual-embeddings`, `masked-language-model`, `bert`, `pseudo-likelihood`, `brooks-lemma`,
  `compatibility`, `anisotropy`, `probing`, `polysemy`.
- `estimatedReadTime` ≈ 45.
- **In-site `connections`:** none, since #14 and #15 are still planned.
- **Graph:** both incoming edges (#11 → #12, #7 → #12) and both outgoing edges (#12 → #14, #12 → #15) exist.

**Scope (watch-list).**
- **formalRAG `dense-retrieval-dual-encoders`** owns:
  - the dual encoder as an architecture: its Thm 1 (factorization reduces retrieval to MIPS);
  - its Thm 2 (the rank ceiling);
  - its Thm 3 (in-batch negatives).

  It treats E_Q and E_P as abstract maps into R^d, trained with InfoNCE, and it says nothing about MLM pretraining,
  contextualization, anisotropy or probing. **This topic owns where an encoder's token vectors come from and what geometry
  they have; RAG takes a pretrained encoder as given.** We do not cover pooling, contrastive fine-tuning or retrieval.
- **formalRAG `late-interaction-learned-sparse`** uses MLM logits for SPLADE's term weights. We cite it as the place where
  the MLM head is used downstream.
- **formalRAG `hypersphere-vmf-geometry`** cites Ethayarajh's cone as the way real embeddings violate vMF isotropy. Prop 2
  and §4 measure it here.

**Other neighbors:**
- **Topic 7** owns SGNS = shifted PMI (its Thm 1) and the polarity probe (`w2v.polarity_probe`). Prop 1 here is stated
  through its Thm 1, and §7(b) reruns its probe.
- **Topic 11** owns the pre-LN block (its Def 1), `stack_forward`, and the identity path. The encoder is that block with
  `causal=False`.
- **Topic 10** owns the learned absolute table (its Def 1), which the encoder trains, keeping Topic 10's forward link.
- **formalML `representation-learning`** owns denoising autoencoders (its Thm 3.2, Tweedie / Vincent 2008), masked
  autoencoding (§8.2) and the linear probe (its Def 9.1). This topic adds BERT's specific channel and its closed-form
  optimum, which that topic states only informally.
- **formalML `kl-divergence`** owns the cross-entropy decomposition (its Prop 1) and Gibbs' inequality (its Prop 2), used
  by Thm 1.

**This topic owns:**
- static versus contextual (Def 1, Prop 1, Cor 1);
- the corruption channel and its optimum (Def 2, Thms 1–2);
- MLM as pseudo-likelihood (Thm 3), Brook's lemma (Thm 4) and the compatibility gap (Cor 2);
- Ethayarajh's measures (Def 3, Prop 2);
- probing against exact Bayes accuracy;
- the finance case study.

**Out of scope:**
- fine-tuning and LoRA (#15);
- NER as sequence labeling (#14);
- ELECTRA's replaced-token detection, span masking, whole-word masking and next-sentence prediction (named, not
  covered);
- generation from MLMs beyond the compatibility result;
- tokenizer effects (Topic 2);
- sentence embeddings and pooling (formalRAG).

## 1. Section outline (geometric first)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | 118 occurrences of *kestrel*, one point under SGNS and under the encoder's layer 0, two clouds by layer 1. The headline: 57 ORG / 61 PRODUCT, static 0.339 (5-fold), layer 1 1.000; "rose" after *costs* versus after *revenue*, static cap 0.559, contextual 0.868 | — | cells 0–1 |
| 1 | One vector per type | the HMM polyseme *w*: its two senses' PMI with the next token disagree by up to 3.65 nats; the type sees log Σ π e^{PMI} | Def 1; **Prop 1**; **Cor 1**; remark (CBOW) | cells 2–5 |
| 2 | The masked-LM objective | a sentence with one token hidden, shown three ways: `[MASK]`, a substitute, itself | Def 2; **Thm 1**; **Thm 2**; the encoder | cells 6–11 |
| 3 | Pseudo-likelihood and the missing joint | two paths from (a, b) to (a′, b′) around a square | **Thm 3**; **Thm 4**; **Cor 2** | cells 12–16 |
| 4 | The geometry of contextual vectors | the polyseme's occurrence cloud at layers 0–2 | Def 3; **Prop 2** | cells 17–18 |
| 5 | Probing | a line through the cloud versus the Bayes boundary | experiments; control | cells 19–20 |
| — | The lab | `<MaskedLMLab client:visible />` | — | cell 28 |
| 6 | Finance case study: a company or its product, and the news a word reports | "the Kestrel turbine" / "Kestrel Turbine Services"; "costs rose" / "revenue rose" | — | cells 21–27 |
| 7 | Where this goes next | — | #15 (a linear head on these vectors, and low-rank updates to them); #14 (NER: alias typing, done per token with a CRF); formalRAG `dense-retrieval-dual-encoders` (pool the vectors, train contrastively); `late-interaction-learned-sparse` (SPLADE reads the MLM head) | — |

- **Lab placement:** after §5. Figure A needs §2, and Figure B needs §§4–5. Figure B's kestrel view previews §6.
- **Back-links:**
  - Topic 7 Thm 1 (SGNS = shifted PMI) from Prop 1; Topic 7's polarity probe from §6(b).
  - Topic 11 Def 1 (the pre-LN block) and Thm 2 (identity path) from §2's encoder.
  - Topic 10 Def 1 (the learned table) from §2.
  - Topic 4 (cross-entropy as code length) from Thm 1.
  - Topic 5's central differences from the gradient check.

## 2. Definitions

**Notation.**
- A sentence is x = (x_1, …, x_n) over a vocabulary of V tokens. x_{−i} is x without position i. x_{<i} and x_{>i} are the
  prefix and suffix.
- A corrupted sentence is x̃, with `[MASK]` an extra input-only symbol. M is the random set of selected positions.
- q(· | x̃) is the encoder's prediction at a position; p denotes a true distribution.

**Def 1 (static and contextual embeddings).**
- A **static** embedding is a map e: x_i ↦ e(x_i) ∈ R^d of the token alone.
- A **contextual** embedding is a map h_i(x_1, …, x_n) ∈ R^d of the sentence and the position.
- An encoder's stream at layer 0, E[x_i] + P_i, is static at a fixed position i.

**Def 2 (the corruption channel; the MLM loss).**
- Each position is selected independently with probability ρ.
- A selected position is replaced by `[MASK]` with probability m, by a token drawn uniformly from the V with probability
  r (possibly itself), and kept with probability κ = 1 − m − r.
- Unselected positions are kept.
- The MLM loss is the mean over selected positions of −log q(x_i | x̃).
- BERT: ρ = 0.15, (m, r, κ) = (0.8, 0.1, 0.1). BERT selects a fixed 15% of positions per sequence. We select independently,
  and §5 flags the difference.
- Write S(s | y) for the selected-position channel: S(`[MASK]` | y) = m, and S(v | y) = κ·1[v = y] + r/V.

**Def 3 (Ethayarajh's measures).** For a word w with occurrence vectors h_1, …, h_k at a layer (the rows of its occurrence
matrix H):
- **SelfSim(w)** is the mean of cos(h_j, h_l) over j ≠ l.
- **Baseline** is the mean cosine between vectors of uniformly drawn tokens of any word. SelfSim − Baseline is the
  anisotropy-adjusted self-similarity.
- **MEV(w)** = σ₁²/Σ_i σ_i² of H, uncentered as in Ethayarajh's eq. 3. Its baseline is the MEV of a random sample of
  vectors.

**The encoder.**
- Topic 11's pre-LN block (its Def 1) with bidirectional attention (`causal=False`), stacked twice.
- The input is x_0 = E[x̃_i] + P_i with a learned table P (Topic 10's Def 1).
- The output is LN(x_L) E[:V]ᵀ/√d + b. E is tied, and `[MASK]` is never a prediction.
- The MLP has width 4d. LayerNorm has no gain or bias (Topic 11's `ln_sphere`).

## 3. Statements and proofs

**Prop 1 (a static vector sees the mixture).** Let w be a type with senses s_1, …, s_K and shares π_k = P(s_k | w).
Suppose a context c depends on an occurrence of w only through its sense: P(c | s_k, w) = P(c | s_k), as in the HMM, where
the next token depends on the current one only through the hidden state.
- (i) PMI(w, c) = log Σ_k π_k exp PMI(s_k, c).
- (ii) max_k (log π_k + PMI(s_k, c)) ≤ PMI(w, c) ≤ max_k PMI(s_k, c).
- *Proof.* P(c | w) = Σ_k P(s_k | w) P(c | s_k, w) = Σ_k π_k P(c | s_k). Divide by P(c) and take logs: P(c | s_k)/P(c) =
  exp PMI(s_k, c), which gives (i). For (ii), Σ_k π_k e^{a_k} ≤ (Σ_k π_k) max_k e^{a_k} = max_k e^{a_k}, and the sum is at
  least each of its terms π_k e^{a_k}. ∎
- *Reading.* Topic 7's Thm 1 says SGNS's optimum is PMI_α − log k. A polyseme's row is therefore the log-mixture (i):
  near the sense that dominates each context, never both at once.
- *Harness:* `test_mixture_identity` (exact on the HMM to 1e-12; the senses' PMIs differ by more than 2 nats somewhere;
  measured max 3.654250).

**Cor 1 (the static cap).** A probe that sees only a static embedding (any classifier g ∘ e) has sense accuracy at most
max_k π_k on the occurrences of w.
- *Proof.* e gives every occurrence of w the same vector e(w), so g(e(w)) is one label ℓ for all of them. The accuracy is
  then P(sense = ℓ | w) = π_ℓ ≤ max_k π_k. ∎
- *Harness:* `test_static_cap`. The token table's probe gets 0.509230 = the cap, the Bayes accuracy is 0.970030, and
  layer 2 gets 0.962479.
- **Remark (CBOW).** CBOW (Mikolov et al. 2013) predicts w_t from the mean of the input vectors of its ±k neighbors: a
  single-mask MLM whose encoder is an average of static vectors, with no positions and no attention.

**Thm 1 (MLM is a proper scoring rule).** At a selected position i, let P_i(· | x̃) = P(x_i = · | x̃, i ∈ M).
- For any predictor q, E[−log q(x_i | x̃) | i ∈ M] = E H(P_i(· | x̃)) + E KL(P_i(· | x̃) ‖ q(· | x̃)).
- So the risk is minimized exactly when q = P_i, almost surely on the law of x̃ given i ∈ M.
- *Proof.* Condition on x̃. Then E[−log q(x_i | x̃) | x̃, i ∈ M] = −Σ_y P_i(y | x̃) log q(y | x̃). Adding and subtracting
  Σ_y P_i log P_i gives H(P_i) + KL(P_i ‖ q), the cross-entropy decomposition (formalML `kl-divergence`, Prop 1). By
  Gibbs' inequality (ibid., Prop 2) KL ≥ 0, with equality iff q(· | x̃) = P_i(· | x̃). Take the expectation over x̃. The
  entropy term does not involve q, and the KL term is minimized pointwise. ∎
- *What we measure.* On the HMM test set (1439 selected positions):
  - the exact predictor's cross-entropy is 1.895124 and the encoder's 1.916888;
  - their mean KL is 0.031756, under 2% of the entropy 1.874197;
  - by what the position shows: 0.021188 at a `[MASK]`, 0.097738 at a random token, 0.039529 at a kept one.
- *Harness:* `test_proper_scoring` (the identity on random p, q to 1e-12; the gap matches the mean KL within 0.02), and
  `test_hmm_mlm_matches_bayes`.

**Thm 2 (the Bayes-optimal predictor under the channel).** Let p(y) = P(x_i = y | x̃_{−i}).
- (i) The minimizer of Thm 1 is q(y) = p(y) S(x̃_i | y) / Σ_{y′} p(y′) S(x̃_i | y′).
- (ii) At a `[MASK]`, q = p.
- (iii) At a visible token v, q(y) = p(y)(κ·1[y = v] + r/V)/(κ p(v) + r/V). In particular
  q(v) = p(v)(κ + r/V)/(κ p(v) + r/V).
- (iv) ρ does not appear.
- (v) If r = 0, then q(v) = 1. If κ = 0, then q = p.
- *Proof.*
  - The selection of i and the draws that corrupt it are independent of x and of every other position's selection and
    draws. So P(i ∈ M | x, x̃_{−i}) = ρ.
  - Given i ∈ M and x, the shown symbol has law S(· | x_i). Hence
    P(x_i = y, x̃_i = s | x̃_{−i}, i ∈ M) = P(x_i = y | x̃_{−i}, i ∈ M) S(s | y) = p(y) S(s | y). The last step holds
    because i ∈ M is independent of (x, x̃_{−i}).
  - Bayes' theorem in x_i gives (i). ρ entered only through the event i ∈ M, which we conditioned on, which gives (iv).
  - (ii): S(`[MASK]` | y) = m for every y, so it cancels.
  - (iii): Σ_{y′} p(y′)(κ·1[y′ = v] + r/V) = κ p(v) + r/V.
  - (v): with r = 0, q(y) = p(y)κ·1[y = v]/(κ p(v)), which is 1[y = v]. With κ = 0, S(v | y) = r/V for every y and
    cancels. ∎
- *Reading.* The 10% of random replacements are what force the encoder to keep a contextual prediction at visible
  positions. Without them it would learn the identity at every visible token. With V = 879 (the filings), r/V ≈ 1.1·10⁻⁴,
  so q(v) ≈ 1 unless p(v) ≲ 10⁻³: in a large vocabulary the visible token is nearly always copied.
- *What we measure.* At 153 kept test positions:
  - the encoder's q(v) averages 0.746530;
  - Thm 2 applied to its own `[MASK]` prediction p̂ gives 0.760706, a mean absolute error of 0.049021;
  - p̂(v) alone averages 0.176315.
- *Harness:* `test_bayes_80_10_10`. The formula equals brute-force enumeration of a length-3 joint, with i's selection
  explicit and the channel at the other positions, to 1e-12 at ρ = 0.15 and 0.4. The corollaries hold to 1e-12. Also
  `test_hmm_mlm_matches_bayes`.

**Thm 3 (single-mask MLM is pseudo-likelihood; Fisher consistency).**
- Choose I uniformly from {1, …, n} and mask it, corrupting nothing else. The MLM risk is
  −(1/n) E Σ_i log q(x_i | x_{−i}) = −(1/n) E log PL_q(x), Besag's pseudo-likelihood.
- If a family {q_θ} contains the true conditionals at θ*, then θ* maximizes E log PL_θ. It is the unique maximizer when
  the conditionals identify θ.
- *Proof.* E_I E_x[−log q(x_I | x_{−I})] = (1/n) Σ_i E[−log q(x_i | x_{−i})], which is the first claim. For each i,
  conditioning on x_{−i}, Thm 1 (with the trivial channel) gives E[−log q_θ(x_i | x_{−i})] ≥ E[−log p(x_i | x_{−i})], with
  equality iff q_θ(· | x_{−i}) = p(· | x_{−i}) a.s. Summing over i, the bound is attained at θ*. Any other maximizer has the
  same conditionals, and therefore equals θ* under identifiability. ∎
- *Honest scope.* We prove the population statement. That the PL *estimator* converges is Besag's program, and we check
  it on one chain rather than cite a general theorem. BERT's multi-mask, 80/10/10 objective is not exactly a PL.
- *What we measure.* For the open Ising chain in {±1}^10 with (h, J) = (0.2, 0.5):
  - the population gradient at the truth is 0 to rounding;
  - the PL estimate from N = 200 exact samples is (0.189346, 0.506314), an error of 0.010654;
  - from N = 20000 it is (0.198625, 0.504240), an error of 0.004240.
- *Harness:* `test_pseudo_likelihood`.

**Thm 4 (Brook's lemma).** Let p be a positive joint on a product space. Then for all x and y,
p(x)/p(y) = Π_{i=1}^{n} p(x_i | x_{<i}, y_{>i}) / p(y_i | x_{<i}, y_{>i}).
So the full conditionals determine the joint, up to normalization.
- *Proof.* Telescope through the sequences z^{(i)} = (x_1, …, x_i, y_{i+1}, …, y_n), with z^{(0)} = y and z^{(n)} = x:
  p(x)/p(y) = Π_i p(z^{(i)})/p(z^{(i−1)}). The sequences z^{(i)} and z^{(i−1)} agree off position i, where they hold x_i
  and y_i. Dividing each by the marginal of their shared coordinates gives
  p(z^{(i)})/p(z^{(i−1)}) = p(x_i | x_{<i}, y_{>i})/p(y_i | x_{<i}, y_{>i}). Positivity makes every term defined. ∎
- *Harness:* `test_brook`. The HMM's joint over all 25⁴ = 390,625 sequences of length 4, rebuilt from its four
  conditionals, matches to 1e-15 absolute and 1e-9 in log.

**Cor 2 (the four-cycle test).** For positions i ≠ j, values a, a′ at i and b, b′ at j, and the rest fixed, define δ:
- δ = [log q_i(a | b)/q_i(a′ | b) + log q_j(b | a′)/q_j(b′ | a′)] − [log q_j(b | a)/q_j(b′ | a) + log q_i(a | b′)/q_i(a′ | b′)].
- If the q's are the conditionals of a positive joint, then δ = 0. So δ ≠ 0 at any (i, j, a, a′, b, b′) proves that no
  joint has them as conditionals.
- *Proof.* Under a joint, the first bracket is log p(a, b)/p(a′, b) + log p(a′, b)/p(a′, b′) = log p(a, b)/p(a′, b′). The
  second bracket is log p(a, b)/p(a, b′) + log p(a, b′)/p(a′, b′), the same quantity. ∎
- Arnold and Press characterize compatibility in full: the converse, for two variables, is that the ratio of the
  conditionals factors. We cite it and prove only the necessity we use.
- *What we measure.* Over 200 random (sequence, adjacent pair, top-two values), the median |δ| is:
  - 2.2·10⁻¹⁶ for the HMM's own conditionals (max 2.2·10⁻¹⁶ to rounding);
  - 3.55·10⁻⁴ for the Bayes-optimal MLM (max 0.915);
  - 0.130 for the trained encoder (mean 0.193, max 1.62).
- *Reading.* The trained encoder is not the conditionals of any joint. Neither, slightly, is the population optimum: at a
  `[MASK]` it predicts the clean token given a context read through the channel, and the set of such targets across
  positions does not come from one joint. Wang and Cho's Gibbs-style generation and Salazar et al.'s PLL scoring use
  these conditionals as if they did. Hennigen and Kim, and Young et al., measure the incompatibility in real MLMs.
- *Harness:* `test_compatibility` (clean max < 1e-9; model median > 10³ × clean and > 0.01; 10⁻⁶ < Bayes median < model
  median).

**Prop 2 (MEV as the best static summary).** Let H ∈ R^{k×d} have singular values σ_1 ≥ … ≥ σ_r > 0.
- (i) The constant c minimizing Σ_j ‖h_j − c‖² is the mean.
- (ii) MEV = σ₁²/‖H‖_F² = max_{‖u‖=1} Σ_j ⟨h_j, u⟩²/Σ_j ‖h_j‖².
- (iii) 1/r ≤ MEV ≤ 1, with MEV = 1 iff r = 1, that is, iff the occurrences are collinear.
- *Proof.*
  - (i): the gradient of Σ_j ‖h_j − c‖² is −2 Σ_j (h_j − c), which vanishes at the mean, and the function is strictly
    convex.
  - (ii): Σ_j ⟨h_j, u⟩² = ‖Hu‖², whose maximum over unit u is σ₁² (the Rayleigh quotient of HᵀH). Also ‖H‖_F² = Σ σ_i².
    By Eckart–Young, the best rank-1 approximation leaves Σ_{i≥2} σ_i².
  - (iii): σ₁² is the largest of r positive terms whose sum is ‖H‖_F², so it is at least their mean. Equality with the
    sum holds iff σ_2 = … = 0. ∎
- Ethayarajh reads MEV as an upper bound on how well one static vector can replace a word's contextual vectors.
- *What we measure.* For the HMM polyseme (test stream) and *kestrel* (filings), by layer:

| layer | HMM SelfSim | HMM MEV | HMM baseline | HMM adj. | kestrel SelfSim | kestrel MEV | filing baseline | kestrel adj. |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.762966 | 0.762782 | 0.067726 | 0.695240 | 1.000000 | 1.000000 | 0.182354 | 0.817646 |
| 1 | 0.619627 | 0.638040 | 0.152142 | 0.467485 | 0.861974 | 0.868410 | 0.320039 | 0.541935 |
| 2 | 0.676130 | 0.690842 | 0.203146 | 0.472984 | 0.845817 | 0.861716 | 0.048264 | 0.797553 |

  - The HMM's layer 0 is not static, because its positions vary.
  - The filing reads put every token at P[16], so kestrel's layer 0 is exactly static.
  - Context lowers self-similarity at both layers. The baseline rises from layer 0 to layer 1 in both models, but it is
    not monotone: the filing encoder's layer 2 is nearly isotropic (0.048). "Upper layers are more anisotropic" is
    Ethayarajh's finding for large models, not a law, and the page says so.
  - MEV baselines are 0.166 / 0.373 / 0.290 (HMM) and 0.195 / 0.288 / 0.093 (filings).
- *Harness:* `test_mev_geometry`.

**§5 probing (asserted in `test_hmm_probe`).**
- The probe is ridge logistic regression (λ = 0.3, fixed before the final runs) on standardized vectors.
- It is trained on 1500 HMM sequences and tested on 1500 others, over 6663 test occurrences:

| layer | probe | shuffled control |
|---|---|---|
| 0 | 0.500976 | 0.503977 |
| 1 | 0.962930 | 0.499325 |
| 2 | 0.962479 | 0.505478 |
| static cap (Cor 1) | 0.509230 | — |
| exact Bayes E max_k P(k \| x) | 0.970030 | — |

## 4. Finance case study: a company or its product, and the news a word reports

**Setup.**
- The encoder has d = 64, 4 heads, 2 blocks and windows of 32. It is trained for 2000 Adam steps at lr 0.01 (linear decay)
  on generate_corpus(300, 0): the same documents as Topics 6–7.
- The vocabulary is the 877 types seen ≥ 5 times (94.3% of tokens), plus `[UNK]` and `[SEP]`.
- Its held-out masked-token cross-entropy is 1.842865 nats against the unigram's 5.364357.
- Every analyzed token is read at the center of its own window, so its layer-0 vector is E[x] + P[16].
- The static baseline is Topic 7's SGNS `SGNSConfig("L2", 25)`.

**(a) A company or its product.**
- The corpus uses 21 company aliases both as ORG and as PRODUCT (gold `Entity.type`): 1720 occurrences, 254 of them
  PRODUCT.
- The probe is leave-one-alias-out: trained on 20 aliases, it types the held-out alias's occurrences. Pooled:

| representation | accuracy |
|---|---|
| per-alias majority (Cor 1's cap) | 0.854651 |
| SGNS (L2, d = 25) | 0.852326 |
| layer 0 (static) | 0.852326 |
| layer 1 | 0.948256 |
| layer 2 | 0.966279 |

- *Kestrel* alone (57 ORG, 61 PRODUCT; majority 0.516949), by 5-fold CV:
  - layer 0: 0.338983 (one vector, so each fold predicts its training majority, which is the test fold's minority);
  - layer 1: 1.000000;
  - layer 2: 1.000000.
- The contexts are templates, "the Kestrel turbine" versus "Kestrel Turbine Services", so 1.000 says the cue is in the
  window, not that the encoder is a tagger. This is the lead-in to NER (#14).

**(b) The news a word reports.**
- The aspect gold has a polarity per clause, and the direction word that follows inherits it ("costs rose" −1, "revenue
  rose" +1). So does the evaluative verb ("costs … exceeded expectations" −1).
- A static vector is capped at each word's majority share.
- Per word, the probe is 5-fold CV, pooled over the 13 words with ≥ 20 labeled occurrences (973 occurrences):
  - cap 0.559096;
  - layer 0: 0.517986;
  - layer 1: 0.868448;
  - layer 2: 0.863309;
  - shuffled control (layer 1): 0.536485.
- The six direction words (101–133 occurrences each) are read at 0.945–0.991. The seven evaluative verbs (26–45 each) are
  not, ranging from 0.357 to 0.853. They have few examples for a 64-dimensional probe, and the aspect term sits four or
  more tokens back. We cannot separate the two causes on this corpus, and the page says so.

**(c) Topic 7's question.**
- Topic 7's leave-one-out polarity probe (`w2v.polarity_probe`) on each lexicon word's mean vector, over 35 words (chance
  0.5 ± 0.085):
  - SGNS 0.571429;
  - layer 0 0.342857;
  - layer 1 0.628571;
  - layer 2 0.571429.
- None is two standard errors above chance.
- The text barely carries this polarity. Voting the other lexicon words near each of the 1414 occurrences gets 0.513791
  within ±8 tokens, 0.546322 within ±32, and 0.570721 over the whole document. A lexicon word matches its document's tone
  0.688928 of the time.
- **The punchline.** A word's polarity is a property of the dictionary; what an occurrence *reports* is a property of its
  context. Per-occurrence vectors recover the second and not the first. Topic 7's sentence "the contextual-embedding topic
  gives each occurrence its own vector" stands, and this page says what that buys.

**Lab windows** (chosen by rule, in `lab_windows`):
- the first PRODUCT *kestrel*;
- the first ORG *kestrel* followed by "turbine";
- the first "rose" reporting bad news;
- the first reporting good news.

## 5. `rigorFlag`

Proved in full: Prop 1, Cor 1, Thm 1, Thm 2 (under independent selection), Thm 3's population statement, Thm 4, Cor 2's
necessity, Prop 2.

Not proved here, and flagged:
1. **BERT selects a fixed 15%,** not each position independently. Thm 2 is exact for independent selection. With a fixed
   count, position i's selection is weakly dependent on the others', and the closed form is an approximation.
2. **Pseudo-likelihood's estimator consistency** is checked on one Ising chain, not proved. The general theory is Besag's
   program, and we cite no theorem from it.
3. **Compatibility's converse** (Arnold and Press) is cited. δ is measured on 200 pairs of one toy encoder, and the
   incompatibility of real MLMs is Hennigen and Kim's and Young et al.'s measurement.
4. **Anisotropy is descriptive.** Gao et al.'s degeneration account (tied embeddings push rare tokens along a common
   direction) is an explanation, not a theorem. In our 2-block filing encoder the anisotropy baseline falls to 0.048 at
   layer 2, so the cone is not universal.
5. **A probe shows that information is linearly decodable, not that the model uses it** (Hewitt and Liang; Tenney et al.).
   The spike's position leak is the cautionary example: the context-free layer 0 typed kestrel at 0.924 through the
   position table alone, until every token was read at one position.
6. **The finance contexts are templates.** Kestrel's 1.000 means the cue is in the window. The evaluative-verb probes fail
   with 26–45 examples each, and sample size and the aspect's distance cannot be separated here.
7. **Topic 7's strong/weak question is not won.** The corpus does not write polarity into windows, and we say so rather
   than tune a probe until it wins.
8. **Toy scale.** These are 2-block encoders with d ≤ 64 on synthetic text. Claims about BERT's layers are cited, not
   reproduced.

## 6. Viz design intent — `MaskedLMLab.tsx` (one component, two figures; data `masked-lm-lab.json`, 52.3 kB)

**A. "Fill in the blank."**
- **The HMM sentence.**
  - 24 tokens chosen by rule (`lab_hmm_sequence`: the first fresh sequence holding *w* in both senses, at positions 3
    and 18). Clicking a token selects position i. A pill row picks what i shows: `[MASK]`, its own token ("kept"), or a
    substitute (the exact `[MASK]` prediction's most probable other token).
  - Bars for all 25 tokens show the exact Bayes q (Thm 2) and the encoder's q side by side, with the KL as a readout.
  - Under "kept" the visible token's bar towers over the `[MASK]` view, which is the shrinkage. Under "substitute" the
    true token keeps mass. The sense posterior is shown on each *w*.
  - *What the reader learns:* q at a visible token is not "copy". It is Thm 2's shrinkage, and the trained encoder
    tracks the exact posterior.
- **The filing windows.**
  - Four windows of 32 words. Clicking a word shows the encoder's top 5 with that word masked.
  - A second readout gives the encoder's probability of the visible word beside Thm 2's prediction from its own
    `[MASK]` guess. These are nearly 1 everywhere, because r/V ≈ 10⁻⁴.
  - *What the reader learns:* in a large vocabulary the channel nearly always copies, and the `[MASK]` view is where the
    context shows.
- **Compatibility.** A strip plot of log₁₀|δ| for the 200 pairs, for the HMM's own conditionals (floored at −12), the
  Bayes MLM's, and the encoder's, with the medians. *What the reader learns:* a joint's conditionals sit at 0, and BERT's
  do not.

**B. "One word, many vectors."**
- **The cloud.**
  - A word selector offers the HMM polyseme (150 test occurrences) or *kestrel* (all 118), with a layer slider over 0, 1
    and 2.
  - Each layer's occurrences are projected on their own top two principal axes, oriented by a rule, and colored by sense
    (A/B; ORG/PRODUCT). The HMM points are shaded by the exact sense posterior.
  - Readouts: SelfSim, MEV, baseline, adjusted SelfSim, MEV baseline, and probe accuracy against the static cap and (HMM)
    the Bayes accuracy.
  - Kestrel's layer 0 is one point (SelfSim = MEV = 1).
  - *What the reader learns:* static is one point; context splits it by sense, as far as the context allows.
- **Probes.** Grouped bars:
  - the alias LOAO (majority, SGNS, layers 0–2);
  - news polarity (cap, layers 0–2, control), with a per-word breakdown on hover;
  - strong/weak (SGNS, layers 0–2, against the chance band 0.5 ± 2 SE), with the vote oracle as three reference ticks.

  *What the reader learns:* where context carries the label, contextual vectors find it, and where it does not, nothing
  does.

Every number comes from `viz_payload()`, pinned by `test_viz_mirror_matches_payload`. Never hand-edit the JSON.

## 7. The module, as built (`contextual_embeddings_masked_lm.py`)

**Imports.** `transformer_block_residual_stream as tb` (with `pe, ak, fc, nlm` from it) and
`word2vec_implicit_factorization as w2v` (with `ds = w2v.ds`), as module objects.

**Reuses:**
- From Topic 11: `init_block`, `stack_forward` (pre-LN, `causal=False`), `mha_backward`, `mlp_forward`, `_ln_forward`,
  `ln_backward`. Its pre-LN backward branch is replayed on the forward pass's caches.
- From Topic 5: `nlm.log_softmax` and `nlm.ADAM_*`.
- From Topic 9: `ak._r`.
- From Topic 7: `w2v.sgns_vectors`, `w2v.polarity_probe`, `w2v.sigmoid`, `w2v.log_sigmoid`.
- From Topic 6: `ds._generator_run`, `ds.documents`, `ds.vocab`, `ds.unigram_counts`, `ds.polarity_words`.
- New, pinned by a test: `mlp_backward`, a matmul version of Topic 11's, the only duplicated function.

**Sections:**
- **§1:** `hmm`, `sense_states`, `sample_hmm`, `_forward_backward`, `clean_conditionals`, `sense_posterior`,
  `neighbor_pmi`.
- **§2:**
  - the channel: `channel`, `selected_channel`, `bayes_mlm`, `corrupt`;
  - the encoder: `MLMConfig`, `init_encoder`, `encode`, `mlm_logits`, `mlm_log_probs`, `mlm_loss_grad`, `train_mlm`;
  - evaluation: `hmm_test_set`, `hmm_fit`.
- **§3:** `ising_states`, `log_pl`, `fit_pl`, `ising_fits`, `hmm_joint`, `brook_rebuild`, `cycle_gap`,
  `_pair_conditionals`, `compatibility`.
- **§§4–5:** `self_similarity`, `mev`, `anisotropy`, `probe_fit`, `probe_predict`, `cv_accuracy`, `hmm_occurrences`,
  `hmm_probe`, `hmm_geometry`.
- **§6:**
  - data and vectors: `filing_vocab`, `filing_docs`, `filing_stream`, `filing_batch`, `analysis_tokens`,
    `filing_vectors`, `filing_tokens`;
  - labels and probes: `news_labels`, `alias_typing`, `news_polarity`, `strong_weak`;
  - fit and geometry: `filing_geometry`, `filing_fit`.
- **Lab:** `lab_hmm_sequence`, `lab_hmm`, `lab_windows`, `lab_clouds`, `viz_payload`.

**Random streams:** `default_rng([SEED, k])` with k = 1500–1595, one per claim or model.

**Timing** (≈ 38 s total):
- the HMM encoder trains in ≈ 12 s (3000 steps, batch 64);
- the filing encoder in ≈ 22 s (2000 steps, batch 64);
- SGNS L2 in 2.8 s;
- the rest in ≈ 2 s.

## 8. Notebook cell map (`01_contextual_embeddings_masked_lm.ipynb`)

| cells | content |
|---|---|
| 0–1 | title; imports, the headline numbers |
| 2–5 | §1: the HMM, Def 1, Prop 1, Cor 1 |
| 6–11 | §2: Def 2, Thm 1, Thm 2, the encoder and its gradient checks |
| 12–16 | §3: Thm 3 (Ising), Thm 4, Cor 2 |
| 17–18 | §4: Def 3, Prop 2, the geometry table |
| 19–22 | §5: probing; the position leak |
| 23–29 | §6: alias typing, news polarity, strong/weak |
| 30–31 | the filing encoder at visible words (Thm 2's shortfall) |
| 32–33 | the lab's data |
| 34 | `assert cm._run_harness() == 0` |

**Commands**
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/contextual-embeddings-masked-lm/contextual_embeddings_masked_lm.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/contextual-embeddings-masked-lm/01_contextual_embeddings_masked_lm.ipynb
```

## 9. Cross-site links (all `status: "published"` on GitHub `main`, checked 2026-10-24; re-verify at the B3 gate)

| field | slug | bridge |
|---|---|---|
| formalmlPrereqs | `representation-learning` | denoising autoencoders (its Thm 3.2), masked autoencoding (§8.2), and the linear probe (its Def 9.1): BERT is the masked autoencoder for text, and §5 probes it |
| formalmlPrereqs | `kl-divergence` | the cross-entropy decomposition (its Prop 1) and Gibbs' inequality (its Prop 2) are Thm 1's proof |
| formalmlConnections | `pca-low-rank` | PCA as optimal projection (its Thm 2): MEV is the share one principal direction carries (Prop 2) |
| formalstatisticsPrereqs | `conditional-probability` | Bayes' theorem (its Thm 4) on the corruption channel is Thm 2 |
| formalstatisticsPrereqs | `maximum-likelihood` | consistency of the MLE (its Thm 3), against which pseudo-likelihood trades efficiency for tractable conditionals (Thm 3) |
| formalstatisticsConnections | `bayesian-computation-and-mcmc` | Gibbs sampling (its Thm 2) needs compatible conditionals; Cor 2 shows an MLM's are not |
| formalragConnections | `dense-retrieval-dual-encoders` | its encoders are taken as given; this topic is where their token vectors come from (watch-list scope) |
| formalragConnections | `hypersphere-vmf-geometry` | it names the anisotropic cone as the way real embeddings break vMF isotropy; §4 measures it |
| formalragConnections | `late-interaction-learned-sparse` | SPLADE's term weights are log(1 + ReLU) of MLM logits, the head trained here |
| formalcalculusConnections | `jacobian` | the multivariate chain rule (its Thm 2) behind the encoder's hand-written backward pass through LayerNorm and attention |

**Not cited:**
- formalStatistics `hidden-markov-models` (draft) and `multivariate-distributions` (its conditional result is Gaussian);
- formalRAG `infonce-contrastive-objective` (contrastive fine-tuning is out of scope);
- formalML `information-bottleneck`.

## 10. References (Chicago 17th; titles verified against ACL Anthology `citation_title`, arXiv, and Crossref on 2026-10-24)

1. Devlin, Chang, Lee, and Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,"
   NAACL-HLT 2019. https://aclanthology.org/N19-1423/ — Def 2 (the 80/10/10 channel), the encoder.
2. Peters et al., "Deep Contextualized Word Representations," NAACL-HLT 2018. https://aclanthology.org/N18-1202/ — Def 1
   (contextual embeddings, ELMo).
3. Mikolov, Chen, Corrado, and Dean, "Efficient Estimation of Word Representations in Vector Space" (2013).
   arXiv:1301.3781 — the CBOW remark.
4. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019). arXiv:1907.11692 — dynamic masking;
   MLM without next-sentence prediction.
5. Vincent, Larochelle, Bengio, and Manzagol, "Extracting and Composing Robust Features with Denoising Autoencoders,"
   ICML 2008. https://doi.org/10.1145/1390156.1390294 — denoising as the objective's ancestor.
6. Besag, "Spatial Interaction and the Statistical Analysis of Lattice Systems," *JRSS B* 36 (1974).
   https://doi.org/10.1111/j.2517-6161.1974.tb00999.x — conditional specification, Thm 4.
7. Besag, "Statistical Analysis of Non-Lattice Data," *The Statistician* 24 (1975). https://doi.org/10.2307/2987782 —
   pseudo-likelihood, Thm 3.
8. Brook, "On the Distinction between the Conditional Probability and the Joint Probability Approaches in the
   Specification of Nearest-Neighbour Systems," *Biometrika* 51 (1964). https://doi.org/10.1093/biomet/51.3-4.481 —
   Thm 4.
9. Arnold and Press, "Compatible Conditional Distributions," *JASA* 84 (1989): 152–156.
   https://doi.org/10.1080/01621459.1989.10478750 — Cor 2's converse.
10. Wang and Cho, "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model," NeuralGen Workshop
    2019. https://aclanthology.org/W19-2304/ — Gibbs-style generation, Cor 2's reading.
11. Salazar, Liang, Nguyen, and Kirchhoff, "Masked Language Model Scoring," ACL 2020.
    https://aclanthology.org/2020.acl-main.240/ — PLL scoring, Thm 3 and Cor 2's reading.
12. Torroba Hennigen and Kim, "Deriving Language Models from Masked Language Models," ACL 2023 (short), 1149–1159.
    https://aclanthology.org/2023.acl-short.99/ — incompatibility of MLM conditionals, measured.
13. Young, Chen, and You, "Inconsistencies in Masked Language Models" (2022). arXiv:2301.00068 — the same, measured.
14. Ethayarajh, "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2
    Embeddings," EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1006/ — Def 3, Prop 2's reading (definitions read from
    the PDF: SelfSim, the baseline (eq. 4), MEV (eq. 3), the adjusted measures).
15. Gao, He, Tan, Qin, Wang, and Liu, "Representation Degeneration Problem in Training Natural Language Generation
    Models," ICLR 2019. arXiv:1907.12009 — the degeneration account of anisotropy (rigorFlag 4).
16. Hewitt and Liang, "Designing and Interpreting Probes with Control Tasks," EMNLP-IJCNLP 2019.
    https://aclanthology.org/D19-1275/ — controls, §5.
17. Tenney, Das, and Pavlick, "BERT Rediscovers the Classical NLP Pipeline," ACL 2019. https://aclanthology.org/P19-1452/
    — layerwise probing.
18. Arora, Li, Liang, Ma, and Risteski, "Linear Algebraic Structure of Word Senses, with Applications to Polysemy," *TACL*
    6 (2018). https://aclanthology.org/Q18-1034/ — a static vector as a superposition of senses, beside Prop 1.
19. Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition," *Proc. IEEE* 77 (1989).
    https://doi.org/10.1109/5.18626 — forward–backward, §1.

## 11. Phase B notes

- **Numbering in page order:** D1, P1, C1, D2, T1, T2, T3, T4, C2, D3, P2.
- **TheoremBlock rules:** proofs go in separate `<TheoremBlock type="proof">` blocks, and titles carry no citations.
  Attribute Brook, Besag, Arnold–Press and Ethayarajh in the prose.
- **Precision:** 3 decimals on the page, copied from the payload, or from the harness's ≥ 6-decimal summary rounded once.
- **MDX hazards:**
  - backtick `[MASK]`, `[SEP]`, `[UNK]`;
  - write δ as `$\delta$`, and use `\lt`/`\gt` in math;
  - no "$w$'s": write "the senses of $w$";
  - one `$…$` per number in long inline number lists.
- **Graph:** flip the node to `published`, and remove "Contextual Embeddings and the Masked-LM Objective" from the
  `representations` track's `planned` (leaving `[]`).
- **Forward links to keep:** Topic 7's "the contextual-embedding topic gives each occurrence its own vector", Topic 10's
  "trains with it" (the learned table), and Topic 11's "the same block without the causal mask".

## 12. Phase B record (2026-10-24)

**Harness changes** (the module grew; no verified number changed):
- **`test_filing_visible_shortfall`** (new). Building the lab exposed that the filing encoder gives a visible word only
  0.50–0.75, where Thm 2 predicts ≈ 1. Over 512 words read at the centers of clean windows:
  - the encoder keeps 0.734263 on the visible word;
  - Thm 2's optimum from its own `[MASK]` guess keeps 0.950336;
  - the guess alone keeps 0.591037.

  The page states the shortfall (rigorFlag 7). Visible positions carry under 3% of the training signal.
- **`test_position_leak`** (new). It reproduces the Phase A spike's leak on the shipped encoder: layer 0, E[x] + P[k] at
  document-internal window positions, types kestrel at 0.906780 by 5-fold CV. All 61 PRODUCT occurrences sit at
  positions 22–24. The page quotes this number in place of the spike's 0.924.
- **Totals:** 20/20 pass in ≈ 38 s (also under `-W error`). The notebook is regenerated with 35 cells and executes with
  exit 0 in ≈ 41 s.
- **Payload:** `viz_payload()` and `masked-lm-lab.json` are unchanged by these additions; the mirror test passes.

**Gates:**
- `pnpm test` 39/39;
- `pnpm exec astro sync` clean;
- `pnpm validate`: all connections valid, 17 planned nodes without MDX;
- `pnpm audit:cross-site`: formalNLP's row shows 12 topics, 110 outgoing edges, 0 slug drift; missing reciprocals
  deferred to launch, generated files not committed;
- `pnpm build` exit 0 (16 pages).

**DOM checks** (`pnpm preview` on port 4330, pane foregrounded):
- **Desktop:**
  - 378 `.katex` spans, 0 `.katex-error`, no `‘` in the text;
  - the island hydrated (`ssr` attribute gone);
  - every Figure A readout (three HMM positions × three modes, four filing windows, the compatibility medians) and every
    Figure B readout and bar equals the payload;
  - the layer slider was driven by real key presses.
- **Mobile preset:** `innerWidth === 375` and `scrollWidth === 375` in all 16 views and toggles. Kestrel at layer 0 is
  one point for 118 occurrences (MEV 1.000, probe 0.339).

**MDX fixes:**
- Three link-then-apostrophe constructions (`)'s`) were reworded, since smart quotes would turn them into an opening
  `‘`.
- The chance band in the polarity bars uses `--color-badge-amber-bg`, because `--color-muted-bg` was invisible in dark
  mode.
