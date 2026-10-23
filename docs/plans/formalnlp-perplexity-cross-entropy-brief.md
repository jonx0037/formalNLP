# Brief — `perplexity-cross-entropy`

**Perplexity, Cross-Entropy, and Compression.** Roadmap #4; prerequisite: `n-gram-models-smoothing`
(published, PR #6). The second `language-modeling` topic and the prerequisite of `neural-language-models`
and `scaling-laws`.

- Phase A (this brief + verified notebook): 2026-10-23. Harness 22/22 under Python 3.15, numpy==2.5.4,
  scipy==1.18.1 in ≈ 18 s; narrative notebook (28 cells) executes (exit 0, ≈ 38 s, committed without outputs).
- Scaffold decisions (Jonathan, 2026-10-23): the ratio-estimator standard error is in scope (Thm 2 and the
  paired MKN − IKN comparison), while hypothesis tests and the bootstrap stay with #28 `significance-testing-for-nlp`; **one
  lab, two linked panels**; OOV words are spelled by a character-trigram Kneser–Ney model fit to the training
  hapaxes; every function body was handed back ("hand all back to me").
- Source of truth: `notebooks/perplexity-cross-entropy/perplexity_cross_entropy.py`. Every number below is
  printed or asserted by it; nothing here may be changed without changing it.
- Binding forward promises: Topic 3 §"Where this goes next" (connect perplexity to cross-entropy and to
  compression; handle subword-versus-word normalization) and Topic 2 §"Where this goes next" (normalize per
  character or per word; make Topic 2's compression reading exact). §§4–5 discharge both: Thm 4 is the
  exact compression reading, and Thm 5 is the normalization.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Perplexity, Cross-Entropy, and Compression |
| `{{SUBTITLE}}` | Held-out log-loss is a file size: an arithmetic coder turns any language model into a compressor within two bits of it — which makes perplexity a branching factor, bits per character its honest unit, and the `<unk>` convention a flattering lower bound. |
| `{{SLUG}}` | `perplexity-cross-entropy` (= graph node id) |
| `{{DOMAIN}}` | `language-modeling` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[n-gram-models-smoothing]` |
| `{{FORMALML_PREREQS}}` | `shannon-entropy`, `kl-divergence`; connection `minimum-description-length` |
| `{{FORMALRAG_PREREQS}}` | none |
| `{{STATS_CALC_PREREQS}}` | Stats `law-of-large-numbers`, `central-limit-theorem`, `maximum-likelihood`; connections `large-deviations`, `model-selection-and-information-criteria`; Calc none |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/n-gram-models-smoothing/`, `notebooks/subword-tokenization-bpe-unigram/`, `notebooks/finance-corpus/` |

Tags: `perplexity`, `cross-entropy`, `arithmetic-coding`, `kraft-mcmillan`, `aep`, `bits-per-character`,
`tokenization`, `language-model`. `estimatedReadTime` ≈ 40. In-site `connections`: none (both dependents are still
planned). The edge `n-gram-models-smoothing → perplexity-cross-entropy` already exists in the graph.

**Scope against formalML (not a watch-list topic, but three siblings cover pieces of it).**
- `kl-divergence` owns Gibbs' inequality and the decomposition H(p, q) = H(p) + KL(p ‖ q) (its Prop 1). We cite
  it as our Prop 1 and spend no proof on it.
- `shannon-entropy` owns Kraft's inequality for prefix codes (its Thm 6, both directions), Shannon's
  source-coding theorem H ≤ E[ℓ] < H + 1 (its Thm 7), Huffman coding, and the entropy rate of a Markov chain
  (its Prop 7). We cite all four. Its Thm 7 converse is stated for uniquely decodable codes but invokes Kraft,
  which it proves only for prefix codes. **McMillan's theorem** (our Thm 3) is the missing step, and we prove it.
- `minimum-description-length` owns two-part and prequential codes (its Def 6). We cite prequential coding in
  the remark on adaptive coders (lzma's EDGAR win). Topic 2 already used the two-part code.

No sibling proves the AEP (formalML uses it in `minimum-description-length` and `rate-distortion` without
proof). We prove it here (Thm 1) with the typical-set size bounds. Calibration (#16) and scaling laws (#13)
appear only as pointers. Topic 3 already proved its Prop 1 (the MLE minimizes empirical cross-entropy, a gap
of c(u·)·KL), and we cite it.

## 1. Section outline (geometric first; each section opens with a picture or a corpus number)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | "Is a perplexity of 1.66 better than 17.4?" — the character 7-gram against the word trigram | — | cells 0–1 |
| 1 | Cross-entropy and the log score | Topic 3's held-out log-loss, 2.8551 nats = 4.119 bits per token; PP 17.38 | Def 1; **Prop 1** (cited); **Cor 1** (strictly proper); **Prop 2** (perplexity facts; worse than uniform) | cells 2–3 |
| 2 | Perplexity as a branching factor | three symbols with p = (0.6, 0.3, 0.1): e^H = 2.455 "effective choices" out of 3 | **Thm 1** (AEP, typical-set bounds) | cells 4–5 |
| 3 | From held-out text to cross-entropy | 2,049 sentences, 23,055 tokens; one number, two standard errors (0.0320 vs 0.0227) | Def 2; **Thm 2** (ratio estimator: LLN + delta-method SE); SMB and Algoet–Cover (cited); **Cor 2** (cross-entropy rate ≥ entropy rate) | cells 6–8 |
| 4 | Compression is prediction | codewords as disjoint dyadic subintervals of [0, 1) | Kraft (cited); **Thm 3** (McMillan); **Prop 3** (Shannon code under the wrong model); **Thm 4** (arithmetic coding within 2 bits); **Prop 4** (a compressor is a model) | cells 9–17 |
| 5 | Choosing the unit | one model, two perplexities: 15.71 per word token, 1.513 per character | **Thm 5** (tokenization invariance; induced character conditionals); the unit table; **Prop 5** (`<unk>` flatters) | cells 18–22 |
| — | The lab | `<PerplexityCompressionLab client:visible />` | — | — |
| 6 | Finance case study | `<FinanceCaseStudy>`: on EDGAR the trigram is worse than uniform | — | cells 23–24 |
| 7 | Where this goes next | — | cross-entropy training → `neural-language-models`; per-token losses in `scaling-laws` inherit Thm 5's unit problem; adaptive coding → in-context prediction; pointers to #16, #28 | cells 25–27 |

Lab embed: one `<PerplexityCompressionLab client:visible />` after §5 (Panel A needs §4, Panel B needs §5),
with forward pointers from Thm 4 and Thm 5.

## 2. Definitions

Notation follows Topic 3. A sentence is a word sequence padded with two `<s>` in front, and every word and
the final `</s>` is a predicted token. V = training word types ∪ {`</s>`, `<unk>`}, |V| = 3,427. The
**normalized string** of a sentence is `" ".join(words) + "\n"`, so its **characters** are each word's
letters plus one separator after each word (a space, or the sentence end after the last word). Held out:
2,049 sentences, 23,055 word tokens, 154,362 characters. EDGAR: 33 sentences, 830 word tokens, 5,249
characters. Logs are natural (nats) unless marked log₂ (bits).

- **Def 1 (cross-entropy, log-loss, perplexity).** For distributions p, q on a finite set,
  H(p, q) = −Σ_x p(x) log q(x). A sequential model Q scores a text by its total negative log-likelihood
  NLL = −log Q(text) = −Σ_i log Q(x_i | x_<i). Given a **unit** with count n_unit in the text (tokens of the model's
  own tokenizer, characters, or word tokens), the log-loss per unit is NLL/n_unit, and perplexity per unit is
  PP_unit = exp(NLL/n_unit) = 2^(bits per unit). **Bits per character** (bpc) = NLL/(n_chars ln 2).
- **Def 2 (the sentence source).** Sentences S₁, S₂, … are i.i.d. finite token sequences ending in `</s>`.
  Write Y_j = −log Q(S_j) and N_j = |S_j| (tokens, `</s>` included). The **per-token cross-entropy** of Q
  against the source is r = E[Y]/E[N].
- **Def 3 (codes).** A binary code assigns codewords c(x) ∈ {0,1}*. It is **prefix-free** if no codeword is a
  prefix of another, and **uniquely decodable** if its extension to sequences, c(x₁…x_m) = c(x₁)⋯c(x_m), is
  injective.
- **Def 4 (open-vocabulary word model).** Our word model codes an in-vocabulary word as one symbol. An OOV word
  becomes `<unk>` followed by its letters and an end mark, spelled by the **speller** S, a character-trigram
  Kneser–Ney model fit to the 1,877 training hapax types (Good–Turing's stand-in for unseen words, as in
  Topic 3). The word history sees `<unk>`, as in Topic 3. The code is slightly deficient, since S can also
  spell in-vocabulary words, which wastes a little code space but never fails.

## 3. Statements and proofs

### Proposition 1 (cross-entropy = entropy + KL; cited)
H(p, q) = H(p) + KL(p ‖ q) ≥ H(p), with equality iff q = p. This is formalML `kl-divergence`, Prop 1, with
Gibbs' inequality. **Asserted** on 300 random pairs, with H and KL cross-checked against `scipy.stats.entropy`
(`test_gibbs_decomposition`).

### Corollary 1 (the log score is strictly proper)
For every p, E_p[−log q(X)] is uniquely minimized at q = p. *Proof:* E_p[−log q] = H(p, q) = H(p) +
KL(p ‖ q), and only the KL term depends on q. ∎ So a forecaster scored by log-loss does best by reporting its
true belief. Training a language model by cross-entropy is maximum likelihood (Topic 3's Prop 1; Gneiting
and Raftery 2007). Contrast: the linear score E_p[q(X)] is maximized by the point mass at the mode of p, not by
p (`test_log_score_strictly_proper`). Bernardo (1979) shows that, for more than two outcomes, the log score is the only
proper score that depends only on the probability of the outcome; we cite this and use it only as a remark.

### Proposition 2 (what a perplexity is)
(i) For a text of N tokens, PP = exp(−(1/N) Σ log q_i) = (Π 1/q_i)^{1/N}, the geometric mean of the inverse
probabilities, which is at most their arithmetic mean. (ii) A model uniform on k symbols has PP = k on every
text. (iii) 1 ≤ e^{H(p)} ≤ |supp p|. (iv) So PP > |V| means the model did worse on this text than the
uniform guess over V.

*Proof.* (i) is the definition plus AM–GM. (ii) Every q_i = 1/k. (iii) H ≥ 0, and H(p) ≤ log |supp p| is
Prop 1 with q uniform on the support. (iv) follows from (ii). ∎ **Asserted** on random instances
(`test_perplexity_facts`). Corpus: interpolated KN's 2.8551 nats per token, Topic 3's number reproduced, gives
PP = 17.38. On EDGAR, interpolated KN's 8.7963 nats is PP 6,610, and modified KN's is 8.6022. Both exceed
log |V| = 8.1394, so both are worse than uniform. Laplace's 8.1326 barely stays below it
(`test_worse_than_uniform_out_of_domain`).

### Theorem 1 (the AEP and the typical set)
Let X₁, …, X_n be i.i.d. from p on a finite alphabet, with H = H(p) and σ² = Var(−log p(X)). For ε > 0, let
A = {xⁿ : |−(1/n) log p(xⁿ) − H| ≤ ε}. Then (i) P(A) ≥ 1 − σ²/(nε²); (ii) |A| ≤ e^{n(H+ε)}; (iii) |A| ≥
P(A) e^{n(H−ε)}. For any model q, P(|−(1/n) log q(Xⁿ) − H(p, q)| ≤ ε) ≥ 1 − Var(−log q(X))/(nε²).

*Proof.* −log p(Xⁿ) = Σ_i −log p(X_i) is a sum of n i.i.d. terms with mean H and variance σ², so (i) is
Chebyshev (the weak LLN). Each xⁿ ∈ A has e^{−n(H+ε)} ≤ p(xⁿ) ≤ e^{−n(H−ε)}. Summing the left inequality over A
gives 1 ≥ P(A) ≥ |A| e^{−n(H+ε)}, which is (ii). Summing the right one gives P(A) ≤ |A| e^{−n(H−ε)}, which is (iii).
The model version is Chebyshev applied to −log q(X_i). ∎

*Reading.* Almost all the probability sits on between e^{n(H−ε)} and e^{n(H+ε)} sequences of nearly equal
probability. Generating text is then like choosing uniformly among e^H options per token, and a model scoring
−(1/n) log q(Xⁿ) ≈ H(p, q) behaves like a uniform guess among e^{H(p,q)}. That is the precise sense in which
perplexity is a branching factor. **Asserted** exactly, by summing over type classes, on 12 random 3-symbol
sources for n ∈ {40, 120, 400} and ε ∈ {0.1, 0.25}, together with the model version (`test_typical_set_aep`).
Worked example: p = (0.6, 0.3, 0.1) has H = 0.8979 nats and e^H = 2.455. At n = 300 and ε = 0.1 the typical set
has P(A) = 0.9979 and |A| = e^{292.0}, which is 4.8 × 10⁻¹⁷ of all 3³⁰⁰ sequences. P(A) climbs with n (0.750,
0.948, 0.998, 1.000 at n = 40, 120, 300, 1,000), and (1/n) log|A| stays within ε of H (0.934, 0.962, 0.973, 0.980).

### Theorem 2 (held-out log-loss is a ratio estimator)
Under Def 2 with E[N], E[Y] < ∞, the per-token held-out log-loss r̂_m = Σ_{j≤m} Y_j / Σ_{j≤m} N_j → r = E[Y]/E[N]
almost surely. If also E[Y²], E[N²] < ∞, then √m (r̂_m − r) ⇒ N(0, Var(Y − rN)/E[N]²), and
se² = Σ_j (Y_j − r̂ N_j)² / (m(m − 1) N̄²) is a consistent estimate of the variance of r̂_m.

*Proof.* By the strong law (formalStatistics `law-of-large-numbers`), Ȳ → E[Y] and N̄ → E[N] > 0 a.s., and
r̂_m = Ȳ/N̄ is a continuous function of the pair. For the limit law, r̂_m − r = (Ȳ − r N̄)/N̄, and Ȳ − r N̄ is the
mean of the i.i.d. terms Y_j − r N_j, which have mean 0. The CLT (formalStatistics `central-limit-theorem`) and
Slutsky give the stated normal limit. Replacing r by r̂ in the sample variance is consistent by the same laws. ∎

*Remarks.* The target is a ratio of means, not the mean of per-sentence averages, so long sentences weigh more.
The i.i.d. unit is the sentence, not the token: tokens within a sentence share a history, so the naive
token-level standard error is too small. **Asserted**: on 20 random bigram sentence sources, with exact targets
from absorbing-chain algebra ((I − T)v = c), r̂ over 20,000 sentences lies within 4 se of E[Y]/E[N]
(`test_ratio_estimator_target`). The interval r̂ ± 1.96 se covers the target in 400 replications of 300
sentences at a rate inside the binomial ±3 s.d. band around 0.95 (`test_ratio_se_coverage`). Corpus:
interpolated KN scores 2.8551 ± 0.0320 nats per token (sentence-level), against a naive token-level se of 0.0227,
1.41× smaller. Modified KN − interpolated KN = −0.0143 ± 0.0006 (paired by sentence), which is 24 se from 0,
although the two marginal intervals overlap (`test_corpus_standard_errors`). The open-vocabulary code scores
0.6371 ± 0.0065 bpc.

**Shannon–McMillan–Breiman and Algoet–Cover (cited).** For a stationary ergodic source P on a finite alphabet,
−(1/n) log P(Xⁿ) → H̄ (the entropy rate) almost surely (McMillan 1953, in L¹; Breiman 1957, a.s.). For a
finite-order Markov model Q with positive conditionals, −(1/n) log Q(Xⁿ) is an ergodic average of a function of
finitely many coordinates, so Birkhoff's ergodic theorem gives convergence to the cross-entropy rate; Algoet and
Cover (1988) prove SMB by sandwiching between exactly such Markov approximations. **Asserted** on random ergodic Markov chains: along one path of length 10⁵, both
averages land within 4 batch-means standard errors of the analytic rates (formalML `shannon-entropy` Prop 7)
(`test_smb_on_a_markov_chain`).

### Corollary 2 (every model gives an upper bound on the entropy rate)
For a stationary source and any sequential model Q, E[−log Q(Xⁿ)] = H(Xⁿ) + KL(Pⁿ ‖ Qⁿ) ≥ H(Xⁿ). Divided by n,
when the limits exist, this gives cross-entropy rate ≥ entropy rate, with equality at Q = P. *Proof:* Prop 1
applied to the n-block marginals. ∎ **Asserted** exactly for block lengths 1, 2, and 4 on 25 random chain pairs,
by enumeration (`test_cross_entropy_rate_bounds_entropy_rate`). So any model's held-out cross-entropy
estimates an upper bound on the source's entropy rate. Brown et al. (1992) bounded English at 1.75 bpc this way.
For our generator, the best code below gives 0.637 bpc. This is a single realization, so it estimates the
bound rather than certifying it.

### Kraft's inequality (cited, formalML `shannon-entropy` Thm 6)
Lengths ℓ₁, …, ℓ_k admit a prefix code iff Σ 2^{−ℓ_i} ≤ 1. *Picture:* codeword c names the dyadic interval
[0.c, 0.c + 2^{−|c|}) of [0, 1). The prefix property says these intervals are disjoint, so their lengths sum to
at most 1. Conversely, lay the intervals left to right in order of increasing length. **Asserted**: Huffman codes
are prefix-free with Kraft sum exactly 1, and the dyadic-interval construction succeeds iff Σ ≤ 1, on 200 random
instances (`test_kraft_prefix_and_converse`).

### Theorem 3 (McMillan)
Every uniquely decodable code satisfies Σ_x 2^{−ℓ(x)} ≤ 1.

*Proof.* Let ℓ_max be the longest codeword length and K = Σ_x 2^{−ℓ(x)}. For each m ≥ 1,
K^m = Σ_{x₁,…,x_m} 2^{−(ℓ(x₁)+⋯+ℓ(x_m))} = Σ_{j=1}^{m ℓ_max} A_j 2^{−j}, where A_j counts the m-symbol messages
whose concatenated codeword has length j. Unique decodability makes those concatenations distinct binary
strings of length j, so A_j ≤ 2^j and K^m ≤ m ℓ_max. If K > 1, the left side grows exponentially in m and the
right side linearly, a contradiction. ∎

So the converse of the source-coding theorem holds for every uniquely decodable code, not only prefix codes:
prefix codes lose nothing. **Asserted exhaustively**: among all 1,365 binary codes of 3 or 4 codewords of length
≤ 3, Sardinas–Patterson finds 587 uniquely decodable, 214 of them not prefix-free (e.g. {0, 01, 11}). None
violates Kraft, and none of the 182 codes with Kraft sum > 1 is uniquely decodable. **Cross-check:**
Sardinas–Patterson agrees with brute-force parse counting (every bit-string up to 10 bits) on all 1,365 codes
(`test_mcmillan_exhaustive`).

### Proposition 3 (coding with the wrong model costs KL)
The Shannon code for q, ℓ_q(x) = ⌈−log₂ q(x)⌉, is a prefix code, and under p its expected length satisfies
H₂(p, q) ≤ E_p[ℓ_q] < H₂(p, q) + 1, i.e. H₂(p) + KL₂(p ‖ q) + [0, 1) bits.

*Proof.* Σ 2^{−ℓ_q} ≤ Σ q = 1, so a prefix code exists (Kraft). −log₂ q ≤ ℓ_q < −log₂ q + 1, so take E_p and
apply Prop 1. ∎ **Asserted** on 300 random pairs, alongside Huffman's [H₂, H₂ + 1) (formalML Thm 7, cited)
(`test_source_coding_with_the_wrong_model`). *Reading:* the bits a model wastes per symbol, beyond the source's
entropy, are its KL divergence. Coding symbol by symbol adds up to one more bit each, which is the waste Thm 4
removes.

### Theorem 4 (arithmetic coding: the whole text in one codeword)
Let Q be any sequential model. Code a message x = x₁…x_n by nested intervals: start from [0, 1), and at step i
keep the slice of the current interval assigned to x_i, of relative width Q(x_i | x_<i). The final interval
I(x) = [a, a + Q(x)) has width Q(x). Let the codeword be the binary expansion of b = ⌈a 2^L⌉ to L = ⌈−log₂ Q(x)⌉ + 1
bits. Then (i) [b/2^L, (b+1)/2^L) ⊆ I(x); (ii) over any prefix-free set of messages (e.g. all messages with a
fixed number of symbols) the codewords form a prefix code; (iii) −log₂ Q(x) ≤ L_min ≤ ⌈−log₂ Q(x)⌉ + 1 for the
shortest such codeword.

*Proof.* (i) 2^{−L} ≤ Q(x)/2, and b/2^L < a + 2^{−L}, so (b+1)/2^L < a + 2·2^{−L} ≤ a + Q(x). (ii) Slices of one
interval are disjoint, so by induction the intervals I(x) of distinct messages in a prefix-free set are
disjoint, and so are the dyadic intervals inside them. If one codeword were a prefix of another, its dyadic
interval would contain the other's. (iii) The upper bound is (i). For the lower bound, a dyadic interval of
length 2^{−L} inside I(x) needs 2^{−L} ≤ Q(x). ∎

The coder runs with integer frequencies f = ⌊q (2⁴⁰ − |A|)⌋ + 1, so every symbol keeps a nonzero slice. The
bounds hold for the quantized model Q̃, and the arithmetic is exact big-integer rationals
(A ← A·T + W·cum, W ← W·f, D ← D·T). **Asserted** on 200 random bigram models and messages: decoding recovers
the message, and the two-sided bound holds in exact integers. Over all 243 length-5 messages of a 3-symbol model,
the codewords are prefix-free with Kraft sum ≤ 1 (`test_arithmetic_code_random_instances`).

**Corpus** (`test_arithmetic_code_corpus`). The entire held-out text, 24,768 symbols including the spelled OOV
letters, codes as **one codeword of 98,345 bits** under interpolated KN. −log₂ Q = 98,344.31, so the gap is
0.69 bits. Quantization costs < 0.01 bits in all, and decoding reproduces the 2,049 sentences exactly. That
is 0.6371 bpc, and it equals the open-vocabulary NLL to 10⁻⁶ relative. Coding symbol by symbol with Shannon
codes would cost 114,900 bits (+16.8%). EDGAR: 29,713 bits (−log₂ Q = 29,711.96), 5.66 bpc. This makes
Topic 2's "spend −log₂ p(t) bits on piece t" exact: to within two bits for a whole file, not one bit per symbol.

### Proposition 4 (a compressor is a model)
If C is a uniquely decodable code on a set of texts, then q_C(x) = 2^{−ℓ(x)} has Σ q_C ≤ 1 (Thm 3), so ℓ(x) =
−log₂ q_C(x): a file size is the NLL of a sub-probability model. Conversely, Thm 4 turns every model into a code.
So bits per character and cross-entropy are the same quantity. ∎ (`test_compressors_are_models`)

| bits per character | held out, alone | held out, given training text | EDGAR, alone | EDGAR, given training text |
|---|---|---|---|---|
| zlib (level 9; given = 32 KiB preset dictionary, a true conditional code) | 1.4763 | 1.3763 | 3.2661 | 3.0269 |
| bz2 (given = \|C(train + test)\| − \|C(train)\|) | 1.0404 | 0.6771 | 3.1244 | 3.5542 |
| lzma (preset 9e; same difference) | 1.2617 | 0.9559 | 3.4201 | **2.9080** |
| interpolated KN, arithmetic-coded | — | **0.6371** | — | 5.6607 |

In domain, the KN code beats every compressor given the same training text. bz2's block sort comes within 6%,
because it sees the corpus's long templated contexts. On EDGAR, lzma beats every static model we have. The best
of those is BPE-400 KN5 at 3.646 bpc. lzma wins by adapting to the text it is coding, and Thm 4 allows that: the
model may condition on everything already coded (prequential coding, Dawid 1984; formalML
`minimum-description-length` Def 6; PPM, Cleary and Witten 1984).

### Theorem 5 (tokenization invariance)
Let P be a distribution on strings (with an end marker), and τ an injective tokenizer from strings to token
sequences. The token model Q_τ(t) = P(τ⁻¹(t)) satisfies log Q_τ(τ(s)) = log P(s) for every s. Consequently, for
a text T and any two units A and B, n_A log PP_A = n_B log PP_B = −log P(T). Per-unit perplexities of one model
differ only by the exponent n_B/n_A, and every tokenizer-free unit (characters, bytes, word tokens) ranks any set
of models identically.

*Proof.* The chain rule over tokens multiplies the conditionals of the one distribution P, pushed forward
through τ, and injectivity makes the pushforward of s the single sequence τ(s). The unit identity is
NLL = n_unit × (NLL/n_unit). A tokenizer-free unit has the same count for every model, so it rescales all totals
by one constant. ∎

*Constructive form: a word model is a character model.* For Q′ (interpolated KN with `<unk>` removed and
renormalized, a proper distribution on in-vocabulary sentences), the prefix mass after context u is
M(π) = Σ_{w ⊒ π} Q′(w | u), a contiguous slice of the sorted vocabulary. Letter c has probability M(πc)/M(π). The
separator after a complete word w has probability Q′(w | u)/M(w) × Q′(`</s>` next) for the sentence end and
× (1 − Q′(`</s>` next)) for a space, and after a space the first letter is normalized by 1 − Q′(`</s>` | u). The
factors telescope to Q′(w | u). **Asserted** on all 1,903 held-out sentences made of in-vocabulary words:
character and word factorizations agree to 10⁻⁹ relative, and every character step is a distribution
(`test_tokenization_invariance`). The same Q′ on the same text is **PP 15.71 per word token and 1.513 per
character** (0.598 bpc).

**The unit table** (held out; KN at the orders shown; "word" is the open-vocabulary code; "word, closed" treats
`<unk>` as one event and is not a code) (`test_per_token_perplexity_misranks`):

| model | tokens / char | nats / token | PP / token | **bpc** | PP / word token |
|---|---|---|---|---|---|
| character 3-gram | 1.0000 | 1.7146 | 5.555 | 2.4737 | 96,772 |
| character 5-gram | 1.0000 | 0.6913 | 1.996 | 0.9974 | 102.4 |
| character 7-gram | 1.0000 | 0.5095 | **1.665** | 0.7351 | 30.31 |
| BPE \|V\| = 100, 5-gram | 0.6575 | 0.7628 | 2.144 | 0.7236 | 28.73 |
| BPE \|V\| = 400, 5-gram | 0.4433 | 1.0398 | 2.829 | 0.6650 | 21.89 |
| BPE \|V\| = 1,870 (Topic 2's MDL optimum), 5-gram | 0.2950 | 1.5235 | 4.588 | 0.6483 | 20.26 |
| word trigram (interpolated KN), open | 0.1494 | 2.9567 | 19.23 | **0.6371** | 19.23 |
| word trigram, closed (`<unk>`) — not a code | 0.1494 | 2.8551 | 17.38 | (0.6152) | 17.38 |

Per-token perplexity calls the character 7-gram the best model (1.665). In bits per character it ranks fifth of
seven, and the word trigram is best. The two orderings differ, and bpc and per-word perplexity order the models
identically, as Thm 5 says they must. Per-token perplexity rewards short tokens: with fewer bits to spend per
token, every token looks easy. The context each model sees also differs (6 characters for the 7-gram, about
13 for both the word trigram and BPE-1870's 5-gram, at 6.70 and 3.39 characters per token), so this table
compares models as built, not tokenizers in isolation.

### Proposition 5 (`<unk>` can only flatter)
Let a closed-vocabulary model give `<unk>` probability Q(`<unk>` | u), and let any open completion spell the OOV
word w with S(w | ·), where Σ_w S ≤ 1. Then −log Q_open(text) = −log Q_closed(text) − Σ_OOV log S ≥
−log Q_closed(text). The closed score is not a code length at all, because it assigns one probability to
infinitely many strings. *Proof:* every spelling factor is at most 1. ∎ **Asserted** for every sentence, model,
and test set. The cutoff experiment maps every word with training count ≤ k to `<unk>`, refits interpolated KN,
and trains the speller on the training tokens of those words (`test_unk_flatters`):

| k | \|V_k\| | OOV share (held out) | closed PP / token | open bpc | EDGAR closed PP | EDGAR open bpc |
|---|---|---|---|---|---|---|
| 0 (Topic 3's V) | 3,427 | 0.67% | 17.38 | 0.6371 | 6,610 | 5.6605 |
| 1 | 1,550 | 9.13% | 7.99 | 0.7483 | 176.2 | 4.8598 |
| 4 | 879 | 13.14% | 5.36 | 0.7954 | 138.1 | 4.8670 |
| 16 | 483 | 16.37% | 4.35 | 0.8827 | 93.24 | 4.4244 |
| 64 | 217 | 27.35% | 3.69 | 1.2594 | 29.42 | 3.9103 |
| all (V = {`</s>`, `<unk>`}) | 2 | 91.11% | **1.33** | 2.5339 | 1.21 | 3.6760 |

In domain, every step of the cutoff lowers the closed perplexity and lengthens the honest code. A model that
knows no words reports perplexity 1.33. On EDGAR the open code falls from 5.66 to 3.68 bpc as the cutoff
rises (not monotonically: 4.860 at k = 1, 4.867 at k = 4), because the word model's in-domain vocabulary is
worth less there than a speller trained on more words.

## 4. Finance case study (§6)

**Worse than guessing.** On the public EDGAR MD&A excerpts (57.59% of tokens OOV, Topic 3's figure), interpolated
KN's per-token perplexity is 6,610, above |V| = 3,427: it predicts worse than the uniform distribution over its
own vocabulary (8.7963 vs 8.1394 nats; modified KN 8.6022). Laplace's heavy smoothing toward uniform keeps it just
under (8.1326). This is Topic 3's out-of-domain reversal, read as a branching factor (Prop 2).

**What `<unk>` hides.** Closed, interpolated KN on EDGAR is 2.0067 "bpc", which would beat lzma's 2.9080. Honestly
coded, with every OOV word spelled, it is **5.6605 bpc**, and 64.5% of those bits are spelling (3.4% in domain).
The ranking of the three smoothers in honest units is Laplace 5.509 < modified KN 5.616 < interpolated KN 5.661
bpc. The one EDGAR sentence in the lab shows the same order: *book reserves are established for these plans*
costs Laplace 225 bits, modified KN 231, and interpolated KN 233.

**The speller overfits too.** The trigram speller, fit to the corpus's planted pseudo-words, beats a letter
unigram in domain (0.6371 vs 0.6612 bpc) and loses to it on real English (5.6605 vs 4.9351). Every component fit
to the training distribution pays for the shift (`test_finance_findings`).

**The zip file wins.** lzma, given the training text, codes EDGAR at 2.908 bpc. The best static model is BPE-400
KN5 at 3.646, and the character 3-gram (3.720) beats the 7-gram (3.748) and the 5-gram (4.105). lzma adapts
within the document it is coding. That is the case for models that condition on the text so far: cache
models, and eventually in-context prediction.

## 5. `rigorFlag`

> Perplexity and bits per character are estimates on one finite text, not properties of a language. The law of
> large numbers and the standard errors treat held-out sentences as i.i.d., but ours come from a Pitman–Yor
> generator whose restaurants share state across a document — neither i.i.d. nor stationary — so the intervals
> are idealizations; Shannon–McMillan–Breiman is cited, not proved, and checked only on Markov chains. Bits per
> character depend on the character inventory (here lowercase letters, spaces, and sentence ends; digits and
> punctuation are dropped). The arithmetic code is exact only in unbounded precision. The open-vocabulary code is
> deficient, and its speller is a choice — the one that wins in domain loses out of domain. The bz2 and lzma
> "given the training text" figures are differences of file sizes, not conditional codes. A closed-vocabulary
> `<unk>` perplexity is not a code length at all. Every ranking is evidence about one synthetic corpus, and
> several reverse out of domain.

## 6. Viz design intent — `PerplexityCompressionLab` (one component, two linked panels)

All data from `viz_payload()` (≈ 45 kB). Mirror it to the decimal in
`src/components/viz/data/perplexity-compression-lab.json`, with a Phase B `test_viz_mirror_matches_payload`
(the Topic 1–3 pattern). Marginalia tokens only.

**Panel A, "Arithmetic-code a sentence."** A sentence picker over `LAB_SENTENCES`:
1. *halvorsen reported revenue of million for the second quarter* (in domain, 61 characters)
2. *liabilities were broadly unchanged in coquotas nonroyaltys* (one OOV word, Topic 2's *nonroyaltys*, spelled letter by letter)
3. *book reserves are established for these plans* (EDGAR, 5 of 7 words OOV)

A smoother toggle (Laplace, interpolated KN, modified KN) and a step control (play/step through the symbols).
The main view is the current interval drawn full width, partitioned into the top-5 next-symbol slices (labeled,
positioned by their slice starts) plus "other". The chosen symbol's slice is highlighted, then zoomed to full
width on the next step. When `<unk>` is chosen, the alphabet switches to letters (a visible mode change).
Beneath it: a strip of nested intervals on a log-width scale, and the **bit ticker**, showing the settled bits
as a prefix of the final codeword (`settled ≤ length` is asserted). Readouts: −log₂ Q so far, settled bits, the
final codeword length against −log₂ Q and against symbol-by-symbol Shannon bits.

| sentence | Laplace | interpolated KN | modified KN |
|---|---|---|---|
| 1 | 60 bits (59.28) | 16 (14.71) | 16 (14.76) |
| 2 | 88 (86.68) | 65 (64.50) | 65 (63.66) |
| 3 | **225** (224.06) | 233 (231.76) | 231 (229.61) |

*Lessons:* a good prediction is a wide slice and costs few bits, so interpolated KN codes sentence 1 in 16 bits
where Laplace needs 60. Spelling an OOV word costs over half a sentence's bits (*nonroyaltys* is 35.9 of the 65 under
interpolated KN: 14.6 for `<unk>`, 21.3 for its letters). Out of domain, the order of the smoothers reverses. The codeword is never more than 2 bits
longer than −log₂ Q.

**Panel B, "Pick a unit."** A test-set toggle (in domain / EDGAR). Bars for the unit table's eight models, with a
**"rank by"** control: per token (PP, log scale), per character (bpc), or per word token. Switching reorders the
bars, and the closed row is hatched and labeled "not a code". A **cutoff slider** over `UNK_CUTOFFS` drives two
lines, closed PP per token (falling to 1.33) and open bpc (rising to 2.53), on twin axes. *Lessons:* per-token
perplexity rewards short tokens; every tokenizer-free unit gives one ranking; `<unk>` buys perplexity by refusing
to code words.

**Link.** The corpus code (98,345 bits, 0.6371 bpc) is Panel B's "word" row: hovering that bar shows the
codeword length, and Panel A's readout cites it. Choosing EDGAR in Panel B selects sentence 3 in Panel A.

## 7. Notebook cell map (`01_perplexity_cross_entropy.ipynb`, 28 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; corpus sizes, units | — |
| 2–3 | §1 Prop 1, Cor 1, Prop 2; worse than uniform | `test_gibbs_*`, `test_log_score_*`, `test_perplexity_facts`, `test_worse_than_uniform_*` |
| 4–5 | §2 Thm 1; the worked example and its n-sweep | `test_typical_set_aep` |
| 6–8 | §3 Thm 2; corpus SEs; SMB; Cor 2 | `test_ratio_*`, `test_corpus_standard_errors`, `test_smb_*`, `test_cross_entropy_rate_*` |
| 9–12 | §4 Kraft, Thm 3 (census), Prop 3 | `test_kraft_*`, `test_mcmillan_exhaustive`, `test_source_coding_*` |
| 13–15 | Thm 4: a traced sentence; the corpus codeword | `test_arithmetic_code_*` |
| 16–17 | Prop 4: the compressor table | `test_compressors_are_models` |
| 18–20 | §5 Thm 5: induced characters; the unit table | `test_vector_models_*`, `test_tokenization_invariance`, `test_per_token_*` |
| 21–22 | Prop 5: the cutoff table | `test_unk_flatters` |
| 23–24 | §6 finance | `test_finance_findings` |
| 25–27 | §7 viz payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Run:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/perplexity-cross-entropy/perplexity_cross_entropy.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/perplexity-cross-entropy/01_perplexity_cross_entropy.ipynb
```

## 8. Cross-site links (verified `status: "published"` on GitHub `main` 2026-10-23 by the session that started this one; re-verify at the B3 gate)

| field | topic | relationship |
|---|---|---|
| `formalmlPrereqs` | `shannon-entropy` | Kraft's inequality and the source-coding theorem are cited from there; this page adds McMillan's theorem, which its converse needs for uniquely decodable codes, and makes the code exact with arithmetic coding. |
| `formalmlPrereqs` | `kl-divergence` | Cross-entropy is entropy plus KL divergence (its Prop 1); that decomposition makes the log score strictly proper and prices coding with the wrong model at KL bits per symbol. |
| `formalmlConnections` | `minimum-description-length` | An arithmetic code under a sequential model is a prequential code; lzma's out-of-domain win is adaptive (prequential) coding beating every static model. |
| `formalstatisticsPrereqs` | `law-of-large-numbers` | Held-out log-loss converges to the per-token cross-entropy because it is a ratio of two sample means of i.i.d. sentences, each governed by the strong law. |
| `formalstatisticsPrereqs` | `central-limit-theorem` | The standard error of held-out perplexity comes from the CLT and the delta method applied to the ratio estimator, with sentences rather than tokens as the independent units. |
| `formalstatisticsPrereqs` | `maximum-likelihood` | Minimizing empirical cross-entropy is maximizing likelihood, and the log score's strict propriety is why maximum-likelihood language models report calibrated beliefs on average. |
| `formalstatisticsConnections` | `large-deviations` | The AEP is proved here with Chebyshev's polynomial bound; Cramér's theorem sharpens the typical set's concentration to an exponential rate. |
| `formalstatisticsConnections` | `model-selection-and-information-criteria` | Held-out log-loss and AIC estimate the same quantity — expected out-of-sample cross-entropy — one by holding data out, the other by correcting the in-sample fit. |

Not cited: formalRAG `significance-testing-calibration` (calibration is #16's, testing #28's). formalStatistics
`hidden-markov-models` and formalCalculus `convex-optimization` are still `draft`.

## 9. References (Chicago 17th, notes-and-bibliography; every entry has a URL)

1. Shannon, Claude E. "A Mathematical Theory of Communication." *Bell System Technical Journal* 27, no. 3 (1948):
   379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x. — paper; entropy, source coding.
2. Shannon, Claude E. "Prediction and Entropy of Printed English." *Bell System Technical Journal* 30, no. 1
   (1951): 50–64. https://doi.org/10.1002/j.1538-7305.1951.tb01366.x. — paper; entropy rate by prediction.
3. McMillan, Brockway. "The Basic Theorems of Information Theory." *Annals of Mathematical Statistics* 24, no. 2
   (1953): 196–219. https://doi.org/10.1214/aoms/1177729028. — paper; AEP for ergodic sources (cited).
4. McMillan, Brockway. "Two Inequalities Implied by Unique Decipherability." *IRE Transactions on Information
   Theory* 2, no. 4 (1956): 115–16. https://doi.org/10.1109/TIT.1956.1056818. — paper; Thm 3.
5. Breiman, Leo. "The Individual Ergodic Theorem of Information Theory." *Annals of Mathematical Statistics* 28,
   no. 3 (1957): 809–11. https://doi.org/10.1214/aoms/1177706899. — paper; SMB almost surely (cited).
6. Rissanen, Jorma. "Generalized Kraft Inequality and Arithmetic Coding." *IBM Journal of Research and
   Development* 20, no. 3 (1976): 198–203. https://doi.org/10.1147/rd.203.0198. — paper; arithmetic coding.
7. Jelinek, Frederick, Robert L. Mercer, Lalit R. Bahl, and James K. Baker. "Perplexity — a Measure of the
   Difficulty of Speech Recognition Tasks." *Journal of the Acoustical Society of America* 62, no. S1 (1977): S63.
   https://doi.org/10.1121/1.2016299. — paper; perplexity named.
8. Bernardo, José M. "Expected Information as Expected Utility." *Annals of Statistics* 7, no. 3 (1979): 686–90.
   https://doi.org/10.1214/aos/1176344689. — paper; the log score's uniqueness (remark).
9. Cleary, John G., and Ian H. Witten. "Data Compression Using Adaptive Coding and Partial String Matching." *IEEE
   Transactions on Communications* 32, no. 4 (1984): 396–402. https://doi.org/10.1109/TCOM.1984.1096090. — paper;
   PPM, adaptive coding.
10. Dawid, A. Philip. "Present Position and Potential Developments: Some Personal Views. Statistical Theory: The
    Prequential Approach." *Journal of the Royal Statistical Society, Series A* 147, no. 2 (1984): 278–92.
    https://doi.org/10.2307/2981683. — paper; prequential coding.
11. Witten, Ian H., Radford M. Neal, and John G. Cleary. "Arithmetic Coding for Data Compression." *Communications
    of the ACM* 30, no. 6 (1987): 520–40. https://doi.org/10.1145/214762.214771. — paper; practical coders.
12. Algoet, Paul H., and Thomas M. Cover. "A Sandwich Proof of the Shannon–McMillan–Breiman Theorem." *Annals of
    Probability* 16, no. 2 (1988): 899–909. https://doi.org/10.1214/aop/1176991794. — paper; SMB and
    cross-entropy rates (cited).
13. Brown, Peter F., Stephen A. Della Pietra, Vincent J. Della Pietra, Jennifer C. Lai, and Robert L. Mercer. "An
    Estimate of an Upper Bound for the Entropy of English." *Computational Linguistics* 18, no. 1 (1992): 31–40.
    https://aclanthology.org/J92-1002/. — paper; Cor 2 in practice.
14. Cover, Thomas M., and Joy A. Thomas. *Elements of Information Theory*. 2nd ed. Hoboken, NJ: Wiley, 2006.
    https://doi.org/10.1002/047174882X. — book; AEP, Kraft–McMillan, arithmetic coding.
15. Gneiting, Tilmann, and Adrian E. Raftery. "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal
    of the American Statistical Association* 102, no. 477 (2007): 359–78.
    https://doi.org/10.1198/016214506000001437. — paper; Cor 1.
16. Mielke, Sabrina J., Ryan Cotterell, Kyle Gorman, Brian Roark, and Jason Eisner. "What Kind of Language Is Hard
    to Language-Model?" In *Proceedings of the 57th Annual Meeting of the Association for Computational
    Linguistics*, 4975–89. Florence: ACL, 2019. https://aclanthology.org/P19-1491/. — paper; comparing models by
    total surprisal across units (Thm 5).
17. Python Software Foundation. "Data Compression and Archiving." Python 3 documentation.
    https://docs.python.org/3/library/archiving.html. — documentation; zlib, bz2, lzma.

## 10. Phase B notes

- MDX at `src/content/topics/perplexity-cross-entropy.mdx`. Page numbering follows page order (Defs 1–4,
  Props 1–5, Thms 1–5, Cors 1–2). Kraft, SMB/Algoet–Cover, and the Bernardo uniqueness result are cited
  paragraphs, not numbered. `references[].type`: 14 → `book`, 17 → `documentation`, the rest `paper`.
- Graph: flip the node's `status` to `published` and remove the title from `planned` in `curriculum.ts`. The edge
  from `n-gram-models-smoothing` already exists. Frontmatter: `prerequisites: [n-gram-models-smoothing]`, no new
  `connections`.
- Lab reads `src/components/viz/data/perplexity-compression-lab.json`
  (`json.dumps(viz_payload(), separators=(',', ':'))`). Add `test_viz_mirror_matches_payload`. Python `inf`
  is already emitted as `null` (`_r`), and the cutoff "all" row has `k: null`.
- Overview hook: the 1.665-vs-17.38 comparison is per-token PP of two different models. Its resolution, Thm 5,
  needs both the unit table and the 15.71-vs-1.513 same-model pair. Say which is which.
- Topic 2's page (§"Tokenization as compression") says the Shannon code spends −log₂ p(t) bits per piece. Link
  back to it from Thm 4's corpus paragraph as the promised exact version.
- Long inline formulas (Thm 2's se², Thm 4's update rules, the separator probability in Thm 5) should be display
  math. At 375 px the prose column is ~343 px (sibling-session note on KaTeX MathML overflow).
- Verify reference URLs resolve before shipping (Jelinek et al. 1977's JASA supplement DOI; Mielke et al. 2019's
  Anthology ID). Sardinas–Patterson (1953) is named in prose as the decodability test. Its *IRE Convention
  Record* paper has no stable URL, so it is not in the references list. Add it only if one is found.
- The scope paragraph (above) about formalML Thm 7's converse is a gap observation, not a correction of the
  sibling. Phrase it neutrally on the page ("McMillan's theorem extends Kraft to every uniquely decodable code").

## 11. Phase B record (2026-10-23)

- Page numbering matches the brief: Defs 1–4, Props 1–5, Thms 1–5, Cors 1–2, in page order (verified in the
  built DOM). Kraft is a cited paragraph; Shannon–McMillan–Breiman is an unnumbered remark block; Bernardo is
  prose. Special tokens stay out of math: the separator formula writes s(u) for Q′(`</s>` | u), and Prop 5 writes
  q_unk(u).
- The lab, `PerplexityCompressionLab`, is embedded after §5 under "Coding a sentence, choosing a unit". It reads
  `src/components/viz/data/perplexity-compression-lab.json`, and `test_viz_mirror_matches_payload` requires that
  file to equal `viz_payload()` exactly (harness 23/23). Panel B ranks the closed `<unk>` row by its pseudo-bpc,
  hatched and labeled "not a code". The test-set toggle and the sentence picker are linked both ways.
- Cross-site slugs were re-verified `status: "published"` on GitHub `main` for all eight targets. All 17 reference
  URLs resolve, and every DOI's Crossref title matches the cited work.
- Gates: the harness passes 23/23; the notebook exits 0; `pnpm test` passes 39/39; `astro sync` and
  `pnpm validate` are clean. `pnpm audit:cross-site` exits 0 with the eight outward links as missing reciprocals
  (expected until launch) and 0 deferred or drifted targets for formalNLP. `pnpm build` exits 0.
- DOM checks on the built page (foreground pane, desktop width): 0 `.katex-error`, 502 `.katex`, and the island
  hydrates. The readouts equal the payload:
  - sentence 1 under interpolated KN: 16 bits (−log₂ Q = 14.713), 14 settled bits, 23 Shannon bits; Laplace 60;
  - sentence 2: 65 bits (64.502), with the spelling rows in letter mode;
  - per-character and per-word-token rankings are identical, and the per-token ranking differs;
  - the EDGAR toggle selects sentence 3;
  - the cutoff slider reads 7.99 / 0.7483 at k = 1 and 1.33 / 2.5339 at "all".
- Mobile: at 375 px a range input first widened the layout viewport to 414 px. After letting both slider rows
  wrap and the inputs shrink, `innerWidth === 375` with no unclipped overflow, before and after hydration.
