# Brief — `subword-tokenization-bpe-unigram`

**Subword Tokenization as Compression: BPE, WordPiece, and Unigram LM.** Roadmap #2; prerequisite:
`zipf-heaps-vocabulary-statistics` (published). Answers Topic 1's closing remark that a closed word list
cannot keep pace with Heaps growth.

- Phase A (this brief + verified notebook): 2026-10-22. Harness 23/23 under Python 3.13,
  numpy==2.5.4, scipy==1.18.1, tokenizers==0.23.3, sentencepiece==0.2.2 in ≈ 9 s; narrative notebook
  executes (exit 0, committed without outputs). Python 3.13 is pinned because sentencepiece ships no
  CPython 3.15 wheel (a source build needs cmake).
- Scaffold decisions (Jonathan, 2026-10-22): full BPE optimality theory — prove what is provable, cite
  the hardness results, back them with a brute-force counterexample; finance case study = planted
  morpheme recovery; **one lab, two panels**; harness cross-checks against Hugging Face `tokenizers`
  and `sentencepiece`; difficulty `intermediate`; Jonathan writes `apply_merge` and `viterbi_segment`.
- Source of truth: `notebooks/subword-tokenization-bpe-unigram/subword_tokenization_bpe_unigram.py`.
  Every number below is printed or asserted by it; nothing here may be changed without changing it.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Subword Tokenization as Compression: BPE, WordPiece, and Unigram LM |
| `{{SUBTITLE}}` | Three ways to spell an open vocabulary with a closed one — greedy merges, likelihood merges, and a probabilistic lattice — compared as codes, and tested against a morphology we planted. |
| `{{SLUG}}` | `subword-tokenization-bpe-unigram` (= graph node id) |
| `{{DOMAIN}}` | `text-as-data` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[zipf-heaps-vocabulary-statistics]` |
| `{{FORMALML_PREREQS}}` | `shannon-entropy`, `minimum-description-length`, `kl-divergence` |
| `{{FORMALRAG_PREREQS}}` | none (no formalRAG topic discusses tokenization; no connection either) |
| `{{STATS_CALC_PREREQS}}` | Stats `maximum-likelihood`; Calc none (`convex-optimization` is still `draft`) |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/zipf-heaps-vocabulary-statistics/`, `notebooks/finance-corpus/` |

Tags: `tokenization`, `bpe`, `wordpiece`, `unigram-lm`, `em-algorithm`, `mdl`, `morphology`.
`estimatedReadTime` ≈ 40. In-site `connections`: none yet (dependents still planned). The graph edge
`zipf-heaps-vocabulary-statistics → subword-tokenization-bpe-unigram` already exists.

**Scope against siblings.** Not on the overlap watch-list; no sibling topic covers subword tokenization.
formalML `shannon-entropy` / `minimum-description-length` own source coding and MDL in general; we
apply them to one coding problem. formalRAG's retrieval models consume tokens and are out of scope.
Perplexity normalization (per token vs. per character) is foreshadowed only — it belongs to
`perplexity-cross-entropy`.

## 1. Section outline (geometric first; each section opens with a picture or a corpus number)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 1 | The open-vocabulary problem | Held-out corpus: 79 word types never seen in training (0.73% of 21,006 tokens), all pseudo-words (`colevys`, `nonroyaltys`) | Def 1; **Prop 1** (completeness), **Prop 2** (2^{n−1}; L-step Fibonacci) | cells 0–3 |
| 2 | BPE: greedy compression | Merge table: `in, an, er, re, en, ed, es, or, an·d, th` with counts 10,843 … 4,102 | Def 2–3; **Prop 3** (merge accounting), **Prop 4** (replay, prefix-closure); **Thm 1** (cited hardness); the `aaaabaaab` counterexample | cells 4–9 |
| 3 | WordPiece: a likelihood criterion | Same corpus, merge 3 differs: BPE takes `(e, r)`, ΔL takes `(an, d)` | **Prop 5** (exact ΔL, PMI leading term, remainder bound); three encoders | cells 10–13 |
| 4 | The Unigram LM | The lattice of `nonroyaltys` | Def 4; **Thm 2** (Viterbi), **Prop 6** (forward–backward, coverage), **Thm 3** (EM); pruning heuristic | cells 14–17 |
| 5 | Tokenization as compression | MDL curve: chars 1,648,296 bits → min 636,622 at |V| = 1,870 → words 712,162 | **Prop 7** (segmentation sandwich); two-part code (empirical); Topic 1 loop closed | cells 18–21 |
| 6 | Finance case study: planted morphemes | `cross·fleet`, `non·royalty·s`, `supplier·ers` and how each tokenizer splits them | Boundary P/R/F1 at matched |V| and matched compression | cells 22–23 |
| 7 | Where this goes next | — | Per-token vs. per-character perplexity → `perplexity-cross-entropy`; n-grams over subwords → `n-gram-models-smoothing` | cells 24–26 |

Lab embed: one `<SubwordTokenizationLab client:visible />` at the end of §4 (both panels' theory is on
the page); §5 and §6 point back to Panel B's bits readout and Panel A's segmentations.

## 2. Definitions

Throughout: $\Sigma$ is a finite alphabet (here the 26 lowercase letters), and training data is a
multiset of words $x \in \Sigma^+$ with counts $c_x$ (pre-tokenized on whitespace; merges never cross
words).

- **Def 1 (vocabulary, segmentation).** A vocabulary is a finite $V \subset \Sigma^+$. A segmentation of
  $x$ is $(t_1, \dots, t_k) \in V^k$ with $t_1 \cdots t_k = x$; $\mathrm{Seg}_V(x)$ is the set of them.
  Decoding is concatenation.
- **Def 2 (merge, merge sequence).** For tokens $a, b$, `apply_merge` scans a token sequence left to
  right and replaces each adjacent $(a, b)$ by the single token $ab$, resuming two positions later
  (no overlaps); $r$ is the number of replacements. A merge sequence $\mu = ((a_1, b_1), \dots, (a_M, b_M))$
  defines $V_0 = \Sigma$, $V_j = V_{j-1} \cup \{a_j b_j\}$; rank-replay encoding applies merges $1, \dots, M$
  in order.
- **Def 3 (greedy BPE, utility).** BPE training picks, at each step, the pair with the largest *window
  count* $c(a, b) = \sum_x c_x \cdot \#\{i : (t_i, t_{i+1}) = (a, b)\}$; ties go to the smaller pair of
  token ids (characters numbered in codepoint order, merged tokens in creation order — Hugging Face's
  rule). The *utility* of $\mu$ on a string is the number of tokens it saves.
- **Def 4 (Unigram LM).** A probability vector $p$ on $V$; $P(t_1, \dots, t_k) = \prod_i p(t_i)$ and
  $P(x) = \sum_{\mathbf t \in \mathrm{Seg}_V(x)} P(\mathbf t)$. Encoding is the Viterbi segmentation
  $\mathbf t^\ast = \arg\max P(\mathbf t)$.

## 3. Statements and proofs

### Proposition 1 (completeness)
$\mathrm{Seg}_V(x) \ne \varnothing$ for every $x \in \Sigma^+$ iff $\Sigma \subseteq V$; then decoding inverts
every encoding.

*Proof.* If $\Sigma \subseteq V$, $(x_1, \dots, x_n)$ is a segmentation. If $c \in \Sigma \setminus V$, every piece
of a segmentation of the one-letter string $c$ would be a nonempty substring of $c$, i.e. $c$ itself. ∎
**Asserted** on random strings and on every held-out OOV word at |V| = 50 (`test_completeness_*`).

### Proposition 2 (counting segmentations)
If every substring of $x$, $|x| = n$, is a piece, then $|\mathrm{Seg}(x)| = 2^{n-1}$. If pieces have length
$\le L$, $|\mathrm{Seg}(x)| = S_L(n)$ with $S_L(0) = 1$, $S_L(m) = \sum_{k=1}^{\min(L, m)} S_L(m - k)$.

*Proof.* A segmentation is determined by which of the $n - 1$ gaps between letters are boundaries, and
every subset occurs. With the cap, classify segmentations of $x_{1:m}$ by the length $k \le L$ of the last
piece; the rest is a segmentation of $x_{1:m-k}$. ∎ **Asserted** against brute-force enumeration
for $n \le 12$, $L \in \{1, 2, 3, 5\}$; $|\mathrm{Seg}| = 32{,}768$ at $n = 16$ (`test_segmentation_count_is_lbonacci`).

### Proposition 3 (merge accounting)
`apply_merge` with pair $(a, b)$ shortens a sequence by exactly $r$. If $a \ne b$, $r = $ the window count.
If $a = b$, a maximal run of $\ell$ copies of $a$ contributes $\ell - 1$ windows but $\lfloor \ell/2 \rfloor$
replacements.

*Proof.* Each replacement consumes two tokens and emits one; every other token is copied, so the
length drops by $r$. For $a \ne b$, windows at $i$ and $i + 1$ would force $t_{i+1} = b$ and $t_{i+1} = a$,
so occurrences are disjoint; the scan skips only the second token of a matched window, which cannot
begin another occurrence, so every occurrence is replaced. For $a = b$ the scan pairs the run's tokens
$(1,2), (3,4), \dots$; runs are separated by non-$a$ tokens and do not interact. ∎

*Reading.* BPE ranks by windows, so a run like `aaaa` (3 windows, 2 replacements) is over-valued; the
compression it actually buys is $r$. **Asserted** on 500 random sequences × 4 pairs and runs
$\ell \le 11$ (`test_merge_accounting`).

### Proposition 4 (rank replay; prefix-closure)
Replaying merges $1, \dots, k$ on a training word reproduces its segmentation after $k$ training merges,
and the first $k$ merges of a greedy run of any length $M \ge k$ are the $k$-merge model.

*Proof.* Training applies merge $j$ to every word's current sequence with the same deterministic map as
replay, starting from the same characters; induct on $j$. Greedy's choice at step $j$ depends only on the
state after $j - 1$ merges (and a deterministic tie rule), not on $M$. ∎ **Asserted** on all 3,425
training words over 4,000 merges, and a separate 74-merge run equals the long run's prefix
(`test_replay_reproduces_training_segmentation`).

**Cross-check.** Our trainer's 4,000 merges equal Hugging Face `BpeTrainer`'s exactly, and both encoders
agree on all 1,774 held-out word types (`test_bpe_matches_hf_tokenizers`).

### Theorem 1 (cited — the hardness of optimal merges)
(i) Finding a merge sequence of maximum utility on a string is APX-complete, and greedy BPE always
achieves at least $1/3$ of the optimal utility (some inputs force a factor no better than $0.625$)
(Kozma and Voderholzer 2024/2026). (ii) The decision versions of bottom-up (merge) and direct
(vocabulary) tokenization are NP-complete (Whittington, Bachmann, and Pimentel 2025). (iii) Greedy
achieves $\frac{1}{\sigma}(1 - e^{-\sigma})$ of optimal, $\sigma$ a curvature of the utility (Zouhar et al. 2023).

Not proved here. **Asserted** consequences, in Kozma–Voderholzer's one-string setting with greedy
ranking by replacements $r$:

- **The counterexample.** `aaaabaaab`, 3 merges: every greedy run (all tie-breaks enumerated) saves
  exactly 5 tokens; the optimum saves 6 via `ab`, `a·ab`, `a·aab` → `a aaab aaab`. Greedy's first merge is
  forced (`aa` makes 3 replacements, more than any other pair) and breaks both copies of the repeated
  unit `aaab`. Smallest such string by exhaustive search over $|\Sigma| \le 3$, $n \le 9$.
- **The bound holds.** On 300 random strings ($n \in [6, 12]$, $|\Sigma| \in \{2, 3\}$, $k \in \{2, 3\}$),
  every greedy run reaches at least $1/3$ of the optimum, and some are strictly suboptimal
  (`test_greedy_is_not_optimal`).

### Proposition 5 (WordPiece's likelihood gain)
Let current token counts be $c_t$, $N = \sum c_t$, and $\mathcal L = \sum_t c_t\log c_t - N\log N$ (the unigram
MLE log-likelihood). If a merge of $(a, b)$, $a \ne b$, makes $m$ replacements, then
$$\Delta\mathcal L = m\log m + \sum_{c \in \{c_a, c_b\}}\big[(c - m)\log(c - m) - c\log c\big] - \big[(N - m)\log(N - m) - N\log N\big]$$
$$= m\Big[\log\frac{mN}{c_a c_b} - 1\Big] + R_{c_a} + R_{c_b} - R_N, \qquad 0 \le R_c \le \frac{m^2}{2(c - m)}.$$
(For $a = b$: $c_a \to c_a - 2m$ and there is one $c$-term.)

*Proof.* Only $c_a, c_b, c_{ab}$ (from 0 to $m$) and $N$ (to $N - m$) change; the first line is the
difference of $\mathcal L$. For the second, put $x = m/c$ and $g(x) = (1 - x)\log(1 - x) + x$; then
$(c - m)\log(c - m) - c\log c = -m\log c - m + c\,g(m/c)$. Now $g(0) = 0$, $g'(x) = -\log(1 - x) \ge 0$,
$g''(x) = 1/(1 - x)$, so $0 \le g(x) = \int_0^x\!\int_0^y \frac{dz}{1 - z}\,dy \le \frac{x^2}{2(1 - x)}$; set
$R_c = c\,g(m/c)$. Collecting terms, $m\log m - m\log c_a - m\log c_b + m\log N - m$. ∎

*Reading.* The leading term is the pair's pointwise mutual information $\log\frac{c_{ab}N}{c_a c_b}$
weighted by its count: WordPiece prefers *cohesive* pairs whose parts rarely occur apart; BPE prefers
*frequent* pairs. On the corpus they agree on `in`, `an`, then split: BPE takes `(e, r)` (count 7,663),
WordPiece takes `(an, d)` (ΔL = 7,355.4 nats). **Asserted**: the exact formula equals a recount for the
first 40 WordPiece merges; the remainder bound on 2,000 random instances; the merge-3 split
(`test_wordpiece_*`).

**Three encoders, one vocabulary.** Rank replay (BPE), MaxMatch (WordPiece: longest vocabulary prefix
first), and Viterbi (Unigram) are different functions. At |V| = 200, WordPiece's vocabulary gives
`acquisition` → MaxMatch `a c qu i s it ion` vs. replay `a c qu i s i tion`; Unigram's gives `boston` →
MaxMatch `b o s to n` vs. Viterbi `b o s t on` (`test_maxmatch_differs_from_replay_and_viterbi`).

### Theorem 2 (Viterbi)
With $\delta(0) = 0$ and $\delta(j) = \max_{i:\ x_{i:j} \in V,\ j - i \le L}\big[\delta(i) + \log p(x_{i:j})\big]$,
$\delta(n) = \max_{\mathbf t \in \mathrm{Seg}_V(x)}\log P(\mathbf t)$ (and $-\infty$ iff $\mathrm{Seg}_V(x) = \varnothing$),
computed with $O(nL)$ vocabulary lookups; back-pointers recover $\mathbf t^\ast$.

*Proof.* Induct on $j$. Every segmentation of $x_{0:j}$ is a segmentation of some $x_{0:i}$ followed by the
piece $x_{i:j}$, and its log-probability is the sum; the maximum over that union is
$\max_i[\max_{\text{seg of } x_{0:i}} + \log p(x_{i:j})] = \max_i[\delta(i) + \log p(x_{i:j})]$. ∎
**Asserted** against brute force on 300 random models/strings; pieces concatenate to the word and
achieve the score (`test_viterbi_matches_brute_force`). **Cross-check**: given SentencePiece's trained
piece scores (|V| = 400), our Viterbi reproduces `sentencepiece`'s segmentation of all 1,774 held-out
word types (`test_viterbi_matches_sentencepiece`).

### Proposition 6 (forward–backward; coverage)
With $\alpha(0) = 0$, $\alpha(j) = \log\sum_i \exp[\alpha(i) + \log p(x_{i:j})]$ and $\beta$ symmetric from the
right, $\alpha(n) = \log P(x)$. The posterior of edge $(i, j)$ is
$q_{ij} = \exp[\alpha(i) + \log p(x_{i:j}) + \beta(j) - \alpha(n)]$, and for every position $k$,
$\sum_{i \le k < j} q_{ij} = 1$.

*Proof.* The induction of Theorem 2 with sum for max (distributivity). Segmentations through edge
$(i, j)$ factor as (segmentation of $x_{0:i}$) × piece × (segmentation of $x_{j:n}$), so their total mass is
$e^{\alpha(i)}p(x_{i:j})e^{\beta(j)}$. Each segmentation has exactly one piece covering position $k$, so
summing $q$ over covering edges sums $P(\mathbf t)/P(x)$ over all $\mathbf t$. ∎ **Asserted** against brute force
and on the lab's lattices (`test_forward_matches_brute_force`, `test_edge_posteriors_cover_each_char`).

### Theorem 3 (EM never lowers the likelihood)
Fix $V$. Given $p$, let $q(\mathbf t \mid x)$ be the posterior over $\mathrm{Seg}_V(x)$ and
$\bar c_u = \sum_x c_x\,\mathbb E_q[\#u \text{ in } \mathbf t]$. Then $p'(u) = \bar c_u / \sum_v \bar c_v$ satisfies
$\ell(p') \ge \ell(p)$, where $\ell(p) = \sum_x c_x \log P_p(x)$.

*Proof.* For any $p'$, Jensen gives $\log P_{p'}(x) = \log\sum_{\mathbf t} q(\mathbf t|x)\frac{P_{p'}(\mathbf t)}{q(\mathbf t|x)} \ge
\sum_{\mathbf t} q\log P_{p'}(\mathbf t) - \sum_{\mathbf t} q\log q$, with equality at $p' = p$ (the ratio is then the
constant $P_p(x)$). Weighting by $c_x$ and subtracting the equality case,
$\ell(p') - \ell(p) \ge Q(p') - Q(p)$ with $Q(p') = \sum_u \bar c_u\log p'(u)$. Writing $\pi_u = \bar c_u/\sum\bar c$,
$Q(p') = (\sum\bar c)\,[-H(\pi) - \mathrm{KL}(\pi\,\|\,p')]$, maximized exactly at $p' = \pi$ by Gibbs'
inequality (formalML `kl-divergence`). So $\ell(p') - \ell(p) \ge Q(\pi) - Q(p) \ge 0$. ∎

Monotone, not globally optimal. **Asserted** on 30 random models × 8 iterations and 4 corpus iterations
at |V| = 200 (`test_em_is_monotone`). Our M-step is the MLE; SentencePiece uses a digamma-smoothed
(Bayesian) variant — a stated difference, not a claim.

**Pruning (heuristic).** From 2,000 seed substrings (top by count × length, length ≤ 10) plus the 26
letters, alternate 2 EM iterations with removing the 25% of non-letter pieces whose removal costs least,
by SentencePiece's loss: piece $t$'s Viterbi frequency is reassigned to its best alternative segmentation
and the change in Viterbi log-likelihood is $\mathrm{freq}(t)\,[\log p(t) - \log p_{\text{alt}}(t)]$.
Snapshots at |V| ∈ {800, 400, 200, 100, 50}. Corpus log-likelihoods: −584,584.6, −591,170.2,
−671,668.5, −836,287.1, −990,339.3 nats.

### Proposition 7 (the segmentation sandwich)
$-\log P(x) \le -\log P(\mathbf t^\ast) \le -\log P(x) + \log|\mathrm{Seg}_V(x)|$.

*Proof.* The gap is $-\log q(\mathbf t^\ast \mid x) \ge 0$; and since $\mathbf t^\ast$ has the largest posterior among
$|\mathrm{Seg}_V(x)|$ outcomes, $q(\mathbf t^\ast|x) \ge 1/|\mathrm{Seg}_V(x)|$. ∎

*Reading.* The Viterbi code wastes exactly the posterior surprisal of its own choice; marginal
("bits-back") coding recovers it (Townsend, Bird, and Barber 2019). In practice the gap is tiny:
`nonroyaltys` at |V| = 400 costs 21.66 bits (Viterbi) vs. 21.65 (marginal), with $\log_2|\mathrm{Seg}| = 4.32$.
**Asserted** for every held-out word at every size (`test_bits_back_sandwich`).

### The two-part code (empirical, asserted)
$L(V) = \sum_{t \in V}(|t| + 1)\log_2 27$ bits (spell each piece, then an end marker);
$L(D \mid V) = -\mathcal L/\ln 2$ for the training tokens under their unigram MLE. Along BPE's 4,000-merge
path (tracked incrementally and checked against recounts, `test_mdl_curve_is_exact`):

| vocabulary | total bits |
|---|---|
| letters only (k = 0) | 1,648,296 |
| **BPE optimum, k = 1,844 (|V| = 1,870)** | **636,622** |
| whole words (3,425 types) | 712,162 |

Too few pieces make the data long; too many make the lexicon long (`test_mdl_has_interior_optimum`). The
minimum depends on the lexicon code chosen — say so; it is the Morfessor-style trade-off
(Creutz and Lagus 2007), not a law.

**Topic 1's loop closed.** 54.8% of word types are hapaxes; at |V| = 400, 0.57% of BPE's 353 used pieces
are (`test_subwords_close_the_vocabulary`).

## 4. Finance case study (§6) — planted morphemes

`finance_corpus._PseudoWords` builds every rare content noun as prefix + root (+ plural `s`) or
[prefix +] root + suffix, with an e-drop (`lease + ing → leasing`). We rebuild `generate_corpus(300, 0)`
recording which words the generator invented (2,830 of 3,425 types; token-identical rebuild,
`test_finance_rebuild_matches_corpus`) and parse each against the morpheme lists: **every parse is
unique**, and all 79 held-out OOV words parse too (`test_planted_morphemes_parse_uniquely`). Score: micro
precision/recall/F1 of each tokenizer's internal boundaries against the planted ones.

| boundary F1 (train) | |V| = 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| BPE | 0.298 | 0.363 | 0.408 | 0.512 | 0.736 |
| WordPiece | 0.281 | 0.307 | 0.413 | 0.528 | 0.813 |
| Unigram | 0.293 | 0.363 | **0.582** | **0.780** | 0.782 |

| held-out OOV F1 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| BPE | 0.323 | 0.393 | 0.407 | 0.514 | 0.697 |
| WordPiece | 0.313 | 0.335 | 0.455 | 0.569 | 0.812 |
| Unigram | 0.343 | 0.411 | 0.643 | 0.807 | 0.794 |

| tokens per held-out word | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| BPE | 4.751 | 3.832 | 3.036 | 2.257 | 1.539 |
| WordPiece | 5.384 | 4.799 | 3.567 | 2.560 | 1.692 |
| Unigram | 4.917 | 3.887 | 2.575 | 1.847 | 1.746 |

The story, in three beats (all asserted, `test_unigram_tracks_planted_morphemes`):

1. **At small |V| nobody finds morphology.** Recall is near 1 because pieces are near-letters; precision is
   ≈ 0.17–0.23. F1 tracks how far each tokenizer has compressed.
2. **At |V| = 200–400 Unigram pulls away** (+0.17, +0.27 F1 over BPE on training words; +0.24, +0.29 on OOV).
   Not just because it compresses more: interpolating BPE's F1 to Unigram's tokens per word gives ≈ 0.47
   (vs. 0.582) and ≈ 0.64 (vs. 0.780) — margins ≥ 0.1 at matched compression. At |V| = 400 Unigram's
   precision is 0.658 vs. BPE's 0.375 (recall 0.958 vs. 0.809). This reproduces Bostrom and Durrett
   (2020) on a corpus where the right answer is known.
3. **By |V| = 800 the gap closes** (all three within 0.08); WordPiece is highest (0.813). Compression and
   morphology agree once the vocabulary holds most roots.

Showcase segmentations at |V| = 400: `nonroyaltys` → BPE `non ro yalty s`, WordPiece
`non ro y al t y s`, Unigram `non royalty s` (planted `non|royalty|s`); `supplierers` → BPE
`sup p li er ers`, Unigram `supplier ers` (planted `supplier|ers`). OOV F1 ≈ training F1 throughout:
pieces learned on training words carry to unseen words. EDGAR excerpts are not used.

## 5. `rigorFlag`

> BPE is a greedy heuristic for an objective — token count, itself only a proxy for what a model needs —
> whose optimum is NP-hard and APX-complete to approximate; its 1/3 approximation guarantee and the
> hardness results are cited, not proved, and checked only on tiny strings. WordPiece is described
> differently across papers and implementations; the criterion here is the exact unigram likelihood gain.
> The Unigram LM's EM only guarantees that likelihood never decreases, not a global optimum, and its
> vocabulary pruning is SentencePiece's heuristic loss. The two-part code's optimum depends on the lexicon
> code we chose. Nothing here shows that any tokenizer improves a downstream model; and compression is
> not morphology — on a corpus with planted morphemes, the tokenizer that compresses best is not always
> the one whose boundaries are right.

## 6. Viz design intent — `SubwordTokenizationLab` (one component, two linked panels)

All data from `viz_payload()` (8.5 kB); mirror to the decimal in
`src/components/viz/data/subword-tokenization-lab.json` with a Phase B `test_viz_mirror_matches_payload`
(the Topic 1 pattern). Marginalia tokens only.

**Panel A — "Run the merges."** A slider over merge counts {0, 24, 74, 174, 374, 774} (|V| = 26, 50, 100,
200, 400, 800) and a BPE ↔ WordPiece toggle. Readouts: |V| and tokens per held-out word (BPE 6.348 →
1.539; WordPiece 6.348 → 1.692). Main view: the six `LAB_WORDS` (`subscriptions`, `underwriting`,
`crossfleet`, `colevys`, `nonroyaltys`, `supplierers`) drawn as piece strips, with planted morpheme
boundaries (where they exist) as tick marks under each word. *Lessons:* each merge buys compression;
the two criteria merge different pairs; boundaries snap to morphemes only sometimes.

**Panel B — "Read the lattice."** A word picker (`colevys`, `nonroyaltys`, `underwriting`) and a |V|
slider over {50, 100, 200, 400, 800}. The word's lattice: characters on a line, edges (i, j) as arcs with
width ∝ posterior (edges below 0.005 omitted), the Viterbi path highlighted. Readouts: Viterbi bits,
marginal bits, $\log_2|\mathrm{Seg}|$ — Prop 7's sandwich made visible (`nonroyaltys` at |V| = 100: Viterbi
`n o n royalty s`; at 400: `non royalty s`). *Lessons:* the lattice holds many segmentations, but the
posterior concentrates; Viterbi wastes almost nothing over the marginal code; pruning reshapes the path.

**Link.** The |V| control is shared: moving Panel B's slider moves Panel A to the matched merge count, so
the reader compares BPE's strip and Unigram's path for the same vocabulary size.

## 7. Notebook cell map (`01_subword_tokenization_bpe_unigram.ipynb`, 27 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; corpus, held-out, OOV words | — |
| 2–3 | §1 Props 1–2 | `test_completeness_*`, `test_segmentation_count_*` |
| 4–7 | §2 merge accounting, replay, HF cross-check | `test_merge_accounting`, `test_replay_*`, `test_bpe_matches_hf_tokenizers` |
| 8–9 | §2 Thm 1 counterexample | `test_greedy_is_not_optimal` |
| 10–13 | §3 Prop 5; three encoders | `test_wordpiece_*`, `test_maxmatch_*` |
| 14–17 | §4 Viterbi, forward–backward, EM; SentencePiece cross-check | `test_viterbi_*`, `test_forward_*`, `test_edge_*`, `test_em_*` |
| 18–21 | §5 sandwich, MDL, hapax loop | `test_bits_back_sandwich`, `test_mdl_*`, `test_subwords_*` |
| 22–23 | §6 finance | `test_finance_rebuild_*`, `test_planted_*`, `test_unigram_tracks_*` |
| 24–26 | §7 viz payload; full harness | `_run_harness() == 0` |

Run:
```bash
uv run --python 3.13 --with numpy==2.5.4 --with scipy==1.18.1 --with tokenizers==0.23.3 --with sentencepiece==0.2.2 python notebooks/subword-tokenization-bpe-unigram/subword_tokenization_bpe_unigram.py
uv run --python 3.13 --with numpy==2.5.4 --with scipy==1.18.1 --with tokenizers==0.23.3 --with sentencepiece==0.2.2 --with jupyter --with ipykernel jupyter execute notebooks/subword-tokenization-bpe-unigram/01_subword_tokenization_bpe_unigram.ipynb
```

## 8. Cross-site links (all verified `status: "published"` on 2026-10-22)

| field | topic | relationship |
|---|---|---|
| `formalmlPrereqs` | `shannon-entropy` | Tokenizers are compared as codes: a piece costs its Shannon code length −log p, and the Viterbi segmentation is the shortest encoding under the Unigram LM's code. |
| `formalmlPrereqs` | `minimum-description-length` | The vocabulary size is read off a two-part code — bits to spell the vocabulary plus bits to code the corpus with it — whose total has an interior minimum. |
| `formalmlPrereqs` | `kl-divergence` | EM's monotonicity for the Unigram LM rests on Jensen's inequality and on Gibbs' inequality (KL ≥ 0), which makes normalized expected counts the exact M-step. |
| `formalstatisticsPrereqs` | `maximum-likelihood` | Unigram LM piece probabilities are maximum-likelihood estimates over latent segmentations, and WordPiece merges by the gain in unigram log-likelihood. |

## 9. References (Chicago 17th, notes-and-bibliography; every entry has a URL)

1. Gage, Philip. "A New Algorithm for Data Compression." *C Users Journal* 12, no. 2 (1994): 23–38.
   https://www.drdobbs.com/a-new-algorithm-for-data-compression/184402829. — paper; BPE's origin.
2. Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Neural Machine Translation of Rare Words with
   Subword Units." In *Proceedings of ACL 2016*, 1715–25. https://aclanthology.org/P16-1162/. — paper; BPE for NLP.
3. Schuster, Mike, and Kaisuke Nakajima. "Japanese and Korean Voice Search." In *Proceedings of ICASSP
   2012*, 5149–52. https://doi.org/10.1109/ICASSP.2012.6289079. — paper; WordPiece.
4. Wu, Yonghui, et al. "Google's Neural Machine Translation System: Bridging the Gap between Human and
   Machine Translation." arXiv:1609.08144, 2016. https://arxiv.org/abs/1609.08144. — paper; WordPiece in NMT.
5. Kudo, Taku. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword
   Candidates." In *Proceedings of ACL 2018*, 66–75. https://aclanthology.org/P18-1007/. — paper; Unigram LM.
6. Kudo, Taku, and John Richardson. "SentencePiece: A Simple and Language Independent Subword Tokenizer
   and Detokenizer for Neural Text Processing." In *Proceedings of EMNLP 2018: System Demonstrations*,
   66–71. https://aclanthology.org/D18-2012/. — documentation/paper; pruning loss.
7. Zouhar, Vilém, Clara Meister, Juan Luis Gastaldi, Li Du, Tim Vieira, Mrinmaya Sachan, and Ryan
   Cotterell. "A Formal Perspective on Byte-Pair Encoding." In *Findings of ACL 2023*, 598–614.
   https://doi.org/10.18653/v1/2023.findings-acl.38. — paper; Thm 1(iii).
8. Kozma, László, and Johannes Voderholzer. "Theoretical Analysis of Byte-Pair Encoding." In *Proceedings
   of ESA 2026*, LIPIcs 388, 80. https://doi.org/10.4230/LIPIcs.ESA.2026.80 (arXiv:2411.08671). — paper; Thm 1(i).
9. Whittington, Philip, Gregor Bachmann, and Tiago Pimentel. "Tokenisation Is NP-Complete." In
   *Proceedings of ACL 2025*, 28133–53. https://aclanthology.org/2025.acl-long.1365/. — paper; Thm 1(ii).
10. Zouhar, Vilém, Clara Meister, Juan Luis Gastaldi, Li Du, Mrinmaya Sachan, and Ryan Cotterell.
    "Tokenization and the Noiseless Channel." In *Proceedings of ACL 2023*, 5184–207.
    https://aclanthology.org/2023.acl-long.284/. — paper; §5 aside (Rényi efficiency, empirical).
11. Bostrom, Kaj, and Greg Durrett. "Byte Pair Encoding Is Suboptimal for Language Model Pretraining." In
    *Findings of EMNLP 2020*, 4617–24. https://aclanthology.org/2020.findings-emnlp.414/. — paper; §6.
12. Dempster, A. P., N. M. Laird, and D. B. Rubin. "Maximum Likelihood from Incomplete Data via the EM
    Algorithm." *Journal of the Royal Statistical Society, Series B* 39, no. 1 (1977): 1–38.
    https://doi.org/10.1111/j.2517-6161.1977.tb01600.x. — paper; Thm 3.
13. Viterbi, Andrew J. "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding
    Algorithm." *IEEE Transactions on Information Theory* 13, no. 2 (1967): 260–69.
    https://doi.org/10.1109/TIT.1967.1054010. — paper; Thm 2.
14. Rissanen, Jorma. "Modeling by Shortest Data Description." *Automatica* 14, no. 5 (1978): 465–71.
    https://doi.org/10.1016/0005-1098(78)90005-5. — paper; two-part code.
15. Creutz, Mathias, and Krista Lagus. "Unsupervised Models for Morpheme Segmentation and Morphology
    Learning." *ACM Transactions on Speech and Language Processing* 4, no. 1 (2007): 3.
    https://doi.org/10.1145/1187415.1187418. — paper; Morfessor, MDL morphology.
16. Townsend, James, Tom Bird, and David Barber. "Practical Lossless Compression with Latent Variables
    Using Bits Back Coding." In *Proceedings of ICLR 2019*. https://arxiv.org/abs/1901.04866. — paper; Prop 7 reading.

## 10. Phase B notes

- MDX at `src/content/topics/subword-tokenization-bpe-unigram.mdx`; page numbering follows page order
  (keep the brief's proofs). `references[].type`: 6 → `documentation`; the rest `paper`.
- Graph: flip node `status` to `published`; remove the title from `planned` in `curriculum.ts`. The
  edge from `zipf-heaps-vocabulary-statistics` already exists; frontmatter
  `prerequisites: [zipf-heaps-vocabulary-statistics]`, no new `connections`.
- Lab reads `src/components/viz/data/subword-tokenization-lab.json`; add `test_viz_mirror_matches_payload`.
- `pnpm audit:cross-site` will report the four outward links as missing reciprocals — expected until launch.
- Verify reference URLs resolve before shipping (Gage's is a Dr. Dobb's reprint; Kozma–Voderholzer's DOI
  is the ESA 2026 version of arXiv:2411.08671).

## 11. Phase B record (2026-10-22)

- Page numbering matches the brief (Defs 1–4, Props 1–7, Thms 1–3) plus **Example 1** for the
  `aaaabaaab` counterexample. The lab is embedded in the Overview (Topic 1's placement), not at the end
  of §4; the prose says Panel B reads fully after the Unigram section.
- The lab reads `src/components/viz/data/subword-tokenization-lab.json`. The payload gained
  `unigram_segs` (Panel A's lower strip), `planted` (boundary ticks), and `f1` (the strip readout);
  `test_viz_mirror_matches_payload` requires the JSON to equal `viz_payload()` exactly (harness now 24/24).
- One shared |V| slider drives both panels over {26, 50, 100, 200, 400, 800}; Panel B clamps to 50,
  since Unigram has no 26-piece model.
- DOM checks on the built page: 0 `.katex-error`, 316 `.katex`; the island hydrates (6 word strips;
  `nonroyaltys` lattice at |V| = 400 has 4 arcs, 3 on the Viterbi path); every readout equals the
  payload at |V| = 26, 400, and 800 for both tokenizers.
