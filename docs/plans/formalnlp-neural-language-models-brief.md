# Brief — `neural-language-models`

**Neural Language Models: From Bengio's NPLM to Recurrent Networks.** Roadmap #5; prerequisite:
`perplexity-cross-entropy` (published, PR #10). The third `language-modeling` topic. Prerequisite of
`word2vec-implicit-factorization`, `attention-as-kernel-smoothing`, and `beam-search-decoding`.

- Phase A (this brief and the verified notebook): 2026-10-23. Harness 21/21 under Python 3.15, numpy==2.5.4, and
  scipy==1.18.1 in ≈ 52 s. The narrative notebook (31 cells) executes (exit 0, ≈ 58 s) and is committed without outputs.
- Scaffold decisions (Jonathan, 2026-10-23):
  - all four optional results are in: RNN tightness, the LSTM cell path with a long-lag task, the softmax
    bottleneck, and non-convexity;
  - the harness trains an NPLM and an Elman RNN, both word-level, and tests the LSTM claims on a synthetic task only;
  - hand-written gradients are cross-checked by complex-step derivatives and central differences (no torch);
  - **one lab, two linked panels**;
  - every function body was written by Claude at Jonathan's request, for his review.
- Source of truth: `notebooks/neural-language-models/neural_language_models.py`. Every number below is printed or
  asserted by it. Nothing here may be changed without changing it.
- Binding forward promises (Topic 4 §"Where this goes next"):
  - *"trained by minimizing exactly the cross-entropy of Corollary 1, and reported in perplexity per token of
    their own tokenizer, which Theorem 5 says to convert."* §2 is the training objective. §6 reports every model
    in bits per character of an exact open-vocabulary code.
  - *"Adaptive prediction … neural models with long contexts."* §4 is the recurrent state. §6–7 are dynamic
    evaluation, an adaptive code tested against lzma on EDGAR.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Neural Language Models: From Bengio's NPLM to Recurrent Networks |
| `{{SUBTITLE}}` | Replace counts with coordinates: a softmax over learned embeddings smooths by geometry, backpropagation is one reverse sweep of the chain rule, and a recurrent state lifts the trigram's horizon — as far as products of Jacobians let gradients reach. |
| `{{SLUG}}` | `neural-language-models` (= graph node id) |
| `{{DOMAIN}}` | `language-modeling` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[perplexity-cross-entropy]` |
| `{{FORMALML_PREREQS}}` | `gradient-descent`, `svd`; connection `minimum-description-length` |
| `{{FORMALRAG_PREREQS}}` | none |
| `{{STATS_CALC_PREREQS}}` | Stats `exponential-families`, `generalized-linear-models`, `maximum-likelihood`; Calc `jacobian`, `gradient`, `mean-value-taylor` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/perplexity-cross-entropy/`, `notebooks/n-gram-models-smoothing/`, `notebooks/finance-corpus/` |

Tags: `neural-language-model`, `embeddings`, `softmax`, `backpropagation`, `bptt`, `vanishing-gradients`, `lstm`,
`dynamic-evaluation`. `estimatedReadTime` ≈ 45. In-site `connections`: none, since all three dependents are still
planned. The edge `perplexity-cross-entropy → neural-language-models` already exists in the graph.

**Scope (not a watch-list topic; four siblings own pieces of it).**
- formalCalculus `jacobian` owns the multivariate chain rule as a product of Jacobians (its Thm 2). It describes
  reverse mode only in prose, for chains. We cite Thm 2 and prove reverse accumulation on a general DAG (Thm 3),
  where fan-out makes adjoints sum.
- formalStatistics owns two results we cite:
  - `exponential-families`: the log-partition identities ∇A = E[T] and ∇²A = Cov[T] (its Thm 7.1). This is our Prop 2.
  - `generalized-linear-models`: the canonical-link score identity Xᵀ(y − μ) (its Thm 1). That page defers the
    multinomial case to formalML, so we prove the K-class version (Thm 2) and its moment-matching corollary.
- formalML owns two more:
  - `gradient-descent`: convex convergence rates. We cite them only to say they do not apply (Prop 3).
  - `svd`: Eckart–Young, which we cite in §5.
- Out of scope apart from pointers:
  - embeddings as implicit matrix factorization (#7, `word2vec-implicit-factorization`; Thm 6's low-rank
    log-probability matrix is the bridge);
  - attention (#9);
  - scaling laws (#13);
  - decoding and beam search (#14);
  - calibration (#16).

  Character-level neural models are not trained here, which is a time-budget choice (§5 rigorFlag).

## 1. Section outline (geometric first; each section opens with a picture or a corpus number)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 0 | Overview | one sentence, *cedar ridge bancorp owns brackenbury lending outright*: KN 45.1 bits, NPLM 28.3 | — | cells 0–1 |
| 1 | From counts to coordinates | the held-out tokens whose bigram training never saw: KN 7.53 nats, NPLM 6.73, RNN 6.25 | Def 1; **Prop 1** (positive, proper); **Thm 1** (smoothing by geometry) | cells 2–5 |
| 2 | Training is maximum likelihood | the trained NPLM's mean p(`<unk>`) = 1.34% against a target rate of 1.40% | **Prop 2** (the softmax is the categorical exponential family); **Thm 2** (score identity, moment matching); **Cor 1** (a never-seen target has no MLE); **Prop 3** (not convex) | cells 6–11 |
| 3 | Backpropagation is reverse accumulation | the NPLM as a computational graph | Def 2; **Thm 3** (reverse accumulation; cost ≤ 2× for matrix products); **Cor 2** (the NPLM's gradient) | cells 12–13 |
| 4 | Recurrent networks and gradients through time | *… expanded its headcount clinicbase footprint*: RNN 41.3 bits, NPLM 57.8 | Def 3; **Thm 4** (tightness); **Cor 3** (BPTT); **Thm 5** (Jacobian products); Def 4; **Prop 4** (the LSTM's cell path) | cells 14–19 |
| 5 | The softmax bottleneck | rank 130 = h + 2 for the NPLM, 611 for KN, over 1,000 contexts | **Thm 6** (rank ≤ d + 2) | cells 20–21 |
| — | The lab | `<NeuralLMLab client:visible />` | — | — |
| 6 | The honest comparison | the held-out text as one codeword: 79,614 bits against KN's 98,345 | the bpc table; **Prop 5** (dynamic evaluation is a code) | cells 22–25 |
| 7 | Finance case study | `<FinanceCaseStudy>`: the networks beat uniform on EDGAR, and in-domain training makes them worse | **Prop 6** (the spelling floor) | cells 26–27 |
| 8 | Where this goes next | — | pointers to #7 (Thm 6 → PMI factorization), #9 (attention replaces the fixed state with a lookup over all of it), #13 (the bpc unit for scaling curves), #14 (decoding a tight model), #16 (calibration) | cells 28–30 |

Lab embed: one `<NeuralLMLab client:visible />` after §5. Panel A needs §§1 and 4; Panel B needs Thm 5 and
Prop 4. Forward pointers go from Thm 1 and Thm 5.

## 2. Definitions

Notation follows Topics 3–4.
- **Splits.**
  - *train*: 6,130 sentences, 69,312 predicted tokens, 460,336 characters.
  - *dev* = `generate_corpus(40, 2)`: 844 sentences, 9,558 tokens. It is a fresh generator run, like held's
    `generate_corpus(100, 1)`.
  - *held*: 2,049 sentences, 23,055 tokens, 154,362 characters.
  - *EDGAR*: 33 sentences, 830 tokens, 5,249 characters.
- **Tuning.** Dev chooses the epoch, the mixture weight, and the dynamic-evaluation step. Held-out text tunes nothing.
- **Vocabulary.** V is Topic 3's vocabulary (|V| = 3,427), and OOV words map to `<unk>`. Logs are natural
  unless marked log₂.

- **Def 1 (NPLM; Bengio et al. 2003).**
  - E ∈ ℝ^{(|V|+1)×d}, with one extra row for `<s>`, maps the context u = (u₁, u₂) to the point
    x(u) = [E_{u₁}; E_{u₂}] ∈ ℝ^{2d}.
  - a = tanh(xW₁ + b₁), z = aW₂ + b₂, and p(· | u) = softmax(z).
  - Here d = 64 and h = 128, with 677,987 parameters. Bengio et al.'s direct input-to-output connections are
    omitted, so Thm 6 applies with state dimension h.
- **Def 2 (computational graph, adjoint).**
  - Nodes v₁, …, v_N are in topological order. Each non-input node is v_i = φ_i(v_{pa(i)}) for a differentiable
    primitive φ_i, and v_N is a scalar.
  - Let F_i(v₁, …, v_i) be v_N computed by running the program from node i + 1 on.
  - The **adjoint** of v_i is v̄_i = ∂F_i/∂v_i: the derivative of the output with the earlier nodes held fixed
    and the later ones recomputed.
- **Def 3 (Elman RNN language model; Elman 1990, Mikolov et al. 2010).**
  - h₀ = 0 and h_t = tanh(x_tW_x + h_{t−1}W_h + b), with x_t the embedding of the previous word (`<s>` first).
  - p(next | prefix) = softmax(h_tW_o + b_o).
  - Here 64/128, with 686,179 parameters. The state is reset at each sentence, matching Topic 4's
    i.i.d.-sentence framework.
- **Def 4 (LSTM; Hochreiter and Schmidhuber 1997, Gers et al. 2000).**
  - Gates i, f, o (sigmoid) and g (tanh) are functions of (x_t, h_{t−1}).
  - c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t, and h_t = o_t ⊙ tanh(c_t).
- **Word dropout (training only).** In every epoch, each training token of a once-seen word (1,877 types)
  becomes `<unk>` with probability ½, both in the context and as a target. This is Good–Turing's stand-in for the
  unseen (Topic 3), implemented as word dropout (cf. Kiperwasser and Goldberg 2016).
- **Training.** Adam (Kingma and Ba 2015), with L2 weight decay 10⁻⁴ on the weight matrices and none on the biases.
  - NPLM: batch 256, learning rate 2·10⁻³.
  - RNN: batches of 32 sentences, learning rate 3·10⁻³, global-norm clipping at 5.
  - Four epochs each, keeping the epoch with the lowest dev loss. Both chose the 4th: dev was still improving.

## 3. Statements and proofs

### Proposition 1 (positive and proper)
Every NPLM and RNN gives every word positive probability in every context, and each next-word distribution sums
to 1. *Proof:* exp > 0, and normalization. ∎

So the zero-count problem of Topic 3 cannot occur, by construction. This is checked on random instances and on
every held-out and EDGAR context of both trained models (`test_softmax_is_positive_and_proper`). Cor 1 shows why
positivity is not yet smoothing.

### Theorem 1 (smoothing by geometry)
For all context points x, x′ ∈ ℝ^{2d} and every word w,
|log p(w | x) − log p(w | x′)| ≤ √2 ‖W₂‖₂ ‖W₁‖₂ ‖x − x′‖₂.

*Proof.* Let f(x) = log p(w | x) = z_w − log Σ_j e^{z_j}. In column form, ∇f(x) = W₁ diag(1 − a²) W₂ (e_w − p).
Here ‖e_w − p‖₂² = (1 − p_w)² + Σ_{j≠w} p_j² ≤ (1 − p_w)² + (Σ_{j≠w} p_j)² = 2(1 − p_w)² ≤ 2, and
0 < 1 − a_k² ≤ 1. So ‖∇f‖ ≤ √2 ‖W₁‖‖W₂‖.

Apply the mean value theorem (formalCalculus `mean-value-taylor`) to g(t) = f(x′ + t(x − x′)) on [0, 1]. This
gives f(x) − f(x′) = ∇f(ξ) · (x − x′) for some ξ on the segment, and Cauchy–Schwarz finishes. ∎

*Reading.* Kneser–Ney's contexts are atoms: two contexts share information only through a common suffix (backoff).
The NPLM's contexts are points, and its predictions move continuously with them. A context never seen in training
is predicted like the seen contexts near it, and "near" is learned from every context at once (Cor 2).

- **Asserted** on 100 random networks, at every pair and word (`test_lipschitz_smoothing_random`).
- **Corpus** (`test_lipschitz_smoothing_corpus`): the trained constant is L = 207.2. Over 2,000 random pairs of
  held-out contexts, the observed ratio max_w |Δ log p| / (L ‖Δx‖) has median 0.0193 and max 0.0403. The theorem
  holds, and its constant is a worst case.
- **Where it pays** (`test_sharing_wins_where_counts_are_absent`). Held-out tokens split by what training saw:

| event class (held-out tokens) | count | KN | NPLM | RNN |
|---|---|---|---|---|
| 0: bigram unseen | 5,112 | 7.525 | 6.735 | 6.247 |
| 1: bigram seen, trigram unseen | 2,374 | 3.563 | 3.293 | 3.066 |
| 2: trigram seen | 15,569 | **1.214** | 1.260 | 1.142 |

  (nats per token.) Geometry wins where counts are absent, and counts beat the NPLM where they exist. The RNN
  wins everywhere, on its longer context. With the same two-word context as KN, the NPLM scores 0.79 nats per token
  better on class 0 and 0.046 worse on class 2.

### Proposition 2 (the softmax is the categorical exponential family)
Let A(z) = log Σ_j e^{z_j}. Then ∇A = p and ∇²A = ∂p/∂z = diag(p) − ppᵀ = Cov_p(e_Y). This is formalStatistics
`exponential-families`, Thm 7.1, for the categorical family. The matrix is symmetric PSD with null space span(𝟙),
so its rank is K − 1. Hence the per-example loss ℓ(z) = −log p_y = A(z) − z_y is convex in z, with gradient
p − e_y and Hessian diag(p) − ppᵀ.

*Proof.* ∂p_i/∂z_j = p_i(δ_ij − p_j). For v ∈ ℝ^K, vᵀ(diag p − ppᵀ)v = Σ p_i v_i² − (Σ p_i v_i)² = Var_p(v_Y) ≥ 0.
This is zero iff v_Y is p-a.s. constant, which means v ∈ span(𝟙) since every p_i > 0. ∎

**Asserted** on 100 random z (`test_softmax_exponential_family`):
- the complex-step Jacobian equals diag p − ppᵀ, which equals the one-hot covariance;
- the matrix is PSD, with 𝟙 in its null space and rank K − 1;
- the complex-step gradient of −log p_y is p − e_y;
- softmax and logsumexp match `scipy.special`.

### Theorem 2 (score identity and moment matching)
For fixed features a_i ∈ ℝ^h (any network's last hidden layer), let z_i = a_iW + b and L(W, b) = (1/n) Σ ℓ(z_i; y_i).
Then:
- ∇_W L = (1/n) Σ a_iᵀ(p_i − e_{y_i}) and ∇_b L = (1/n) Σ (p_i − e_{y_i});
- L is convex in (W, b);
- at any stationary point, (1/n) Σ p_i = (1/n) Σ e_{y_i} and Σ a_i ⊗ p_i = Σ a_i ⊗ e_{y_i}. The model's average
  predicted distribution equals the empirical distribution of the targets, feature by feature.

The first identity holds at every stationary point of the whole network too, since b₂ is among its parameters.

*Proof.* Prop 2 and the chain rule through the affine map. ℓ is convex in z, and z is affine in (W, b). A stationary
point zeroes both gradients. ∎ This is formalStatistics `generalized-linear-models`' canonical-link score
identity Xᵀ(y − μ) (its Thm 1), for K classes.

**Asserted** on 30 random softmax regressions (`test_score_identity_and_moment_matching`):
- the gradient matches the complex step;
- at an L-BFGS stationary point (grad < 10⁻¹¹), both moment conditions hold to 10⁻⁸;
- convexity holds along random segments.

### Corollary 1 (a never-seen target has no maximum-likelihood fit)
If word w never occurs as a target, then ∂L/∂b_w = (1/n) Σ_i p_i(w) > 0 at every parameter value (Prop 1). So L
has no stationary point. Gradient descent lowers b_w at every step, and inf L is approached only as
p(w | ·) → 0.

*Proof.* Thm 2's b-gradient, whose w-component has no −e_y term. ∎ Topic 3's zero-count problem returns one
level up: the softmax is positive, but fitting it by maximum likelihood is not smoothing.

- **Asserted** (`test_zero_frequency_target_has_no_mle`): on 20 random instances, along 300 gradient-descent
  steps, the absent class's bias falls and its mean probability falls at every step, ending below 5% of its start.
- **Corpus**: the raw training text has no `<unk>`.
  - Trained one epoch without dropout, the NPLM's mean p(`<unk>`) on held-out contexts is 2.85·10⁻⁵, below
    1/|V| = 2.92·10⁻⁴. Each held-out OOV token costs 9.76 nats.
  - On EDGAR, where 57.6% of tokens are `<unk>`, it scores 8.2058 nats per token, worse than the uniform guess
    (8.1394).
  - With word dropout, Thm 2 holds on the corpus. The mean p(`<unk>`) over training contexts is 0.0134, against
    the target rate 0.0140 in one dropout draw and the expected rate ½ · n₁/N = 0.0135.
  - With dropout, each held-out OOV token costs 3.13 nats, and EDGAR scores 6.4165 nats per token, better than
    uniform.

### Proposition 3 (the network is not convex)
Relabeling hidden units (W₁ ↦ W₁P, b₁ ↦ Pb₁, W₂ ↦ PᵀW₂) leaves the function, and so L, unchanged. A convex L
would satisfy L((θ + πθ)/2) ≤ (L(θ) + L(πθ))/2 = L(θ). On 4,096 dev events the trained NPLM has
L(θ) = L(πθ) = 2.6955, with π reversing the hidden units, but L at the midpoint is 3.5839. So L is not convex. ∎

The symmetry itself is also checked on random networks (`test_network_is_not_convex`). Consequence: formalML
`gradient-descent`'s convex rates do not apply. Training finds a stationary point at best, and nothing certifies
it is a global optimum.

### Theorem 3 (reverse accumulation)
- (i) v̄_N = 1, and for i < N, v̄_i = Σ_{c : i ∈ pa(c)} v̄_c ∂φ_c/∂v_i. So one sweep in reverse topological
  order computes every adjoint.
- (ii) If each primitive's vector–Jacobian products (one per parent) cost at most ω times evaluating the
  primitive, the sweep costs at most ω × (the forward pass) plus one addition per edge and coordinate.
- For Y = XW, the products ȲWᵀ and XᵀȲ each cost exactly as many multiply–adds as XW, so ω = 2 for matrix
  products. The gradient of a matmul-dominated network costs about twice its forward pass, whatever the number of
  parameters.

*Proof.* (i) F_{i−1}(v₁, …, v_{i−1}) = F_i(v₁, …, v_{i−1}, φ_i(v_{pa(i)})). By the multivariate chain rule
(formalCalculus `jacobian`, Thm 2), for j < i:
∂F_{i−1}/∂v_j = ∂F_i/∂v_j + (∂F_i/∂v_i)(∂φ_i/∂v_j) [j ∈ pa(i)].

The sweep keeps one accumulator per node and, when it reaches node i, adds v̄_i ∂φ_i/∂v_j to each parent's.
Induction on the number of processed nodes shows that after processing N, …, i + 1, the accumulator of v_i
equals ∂F_i/∂v_i = v̄_i. That is exact, because no unprocessed node depends on v_i. (ii) Each node's products are
formed once, and each edge adds once. ∎

Linnainmaa (1976) gave reverse mode; Baur and Strassen (1983) proved the cost bound for arithmetic circuits;
Griewank and Walther (2008) give general ω; Rumelhart et al. (1986) is its use in networks. **Asserted**
(`test_reverse_accumulation_random_dags`): on 200 random DAGs (matmul, tanh, ⊙, +, log-sum-exp; 150+ with
fan-out), every input's adjoint equals its complex-step derivative to 10⁻¹¹ relative. The cost bound (ii) is
proved by operation count, not asserted.

**The complex step** (Squire and Trapp 1998). For real-analytic f, f(θ + ih e_j) = f(θ) + ih ∂_j f + O(h²), so
Im f / h is the derivative with no subtractive cancellation. With h = 10⁻³⁰ it is exact to rounding. It is the
reference that every hand-written gradient here is checked against, with central differences (ε = 10⁻⁶) as a
second check.

### Corollary 2 (the NPLM's gradient)
Per event, with g_z = p − e_y (divided by n for the mean):
- ∂W₂ = aᵀg_z and ∂b₂ = g_z;
- g_a = g_zW₂ᵀ and g_s = g_a ⊙ (1 − a²);
- ∂W₁ = xᵀg_s and ∂b₁ = g_s;
- g_x = g_sW₁ᵀ, and the embedding row of each context word accumulates its slice of g_x (rows of words absent
  from the batch get nothing).

So a word's embedding moves whenever the word occurs, in any context. That is the mechanism by which Thm 1's
geometry is learned. That distributionally similar words end up close is #7's subject, not proved here.
**Asserted** on random instances (`test_nplm_gradient`): complex step to 10⁻¹⁰ relative, central differences to
10⁻⁶, and unused embedding rows exactly zero.

### Theorem 4 (a bounded-state softmax RNN is tight)
Let ε = (Σ_j exp(‖W_o[:, j] − W_o[:, eos]‖₁ + b_{o,j} − b_{o,eos}))⁻¹. Then:
- p(`</s>` | h) ≥ ε on the whole cube ‖h‖_∞ ≤ 1, which contains every tanh state;
- for the length N of a sentence sampled from the model, P(N > n) ≤ (1 − ε)ⁿ;
- the probabilities of all finite sentences sum to 1, and E[N^k] < ∞ for every k.

*Proof.* By Hölder, z_j − z_eos = h · (W_o[:, j] − W_o[:, eos]) + b_{o,j} − b_{o,eos} ≤ ‖h‖_∞‖W_o[:, j] − W_o[:, eos]‖₁ +
b_{o,j} − b_{o,eos}. And p_eos = 1/Σ_j e^{z_j − z_eos}.

P(N > n) is the probability that none of the first n tokens is `</s>`, namely E Π_{t≤n}(1 − p(`</s>` | h_t)) ≤ (1 − ε)ⁿ.
Then P(N = ∞) = lim P(N > n) = 0, and Σ_n n^k(1 − ε)ⁿ < ∞. ∎

So a softmax RNN is a proper distribution over sentences, and Topic 4's Def 2 premises E[N], E[N²] < ∞ hold for
text it generates. Du et al. (2023) characterize tightness in general. Chen et al. (2018) show consistency is
undecidable for unbounded (ReLU) RNNs; bounded states make it easy.

**Asserted** (`test_rnn_tight`):
- on 10 random RNNs with ε ≈ 0.1, p(`</s>`) ≥ ε on 5,000 random cube points and along every sampled path;
- the sampled tails sit under (1 − ε)ⁿ.

Trained RNN:
- ε = 7.1·10⁻¹⁷. That is positive, so the model is tight, but as a numerical tail bound it is vacuous.
- The minimum p(`</s>`) over 500 held-out sentences' states is ≥ ε, and sampled sentences have median length
  below 40.

### Corollary 3 (backpropagation through time)
This is Thm 3 on the unrolled graph (Werbos 1990):
- δ_t = (g_out,t + δ_{t+1}W_hᵀ) ⊙ (1 − h_t²);
- tied weights receive sums over time: ∂W_h = Σ_t h_{t−1}ᵀδ_t, and similarly for W_x, b, and E.

The gradient reaching h_k from the loss at t passes through ∂h_t/∂h_k. **Asserted** (`test_rnn_gradient`):
- complex step and central differences agree on random padded batches;
- batching does not change the summed gradient, since padding carries no loss.

### Theorem 5 (products of Jacobians: vanishing and exploding)
For the Elman recurrence, ∂h_t/∂h_k = Π_{j=t}^{k+1} diag(1 − h_j²)W_hᵀ. Hence
‖∂h_t/∂h_k‖₂ ≤ Π_{j=k+1}^{t} ‖diag(1 − h_j²)W_hᵀ‖₂ ≤ σ_max(W_h)^{t−k}.
- (a) If σ_max(W_h) < 1, every gradient contribution from step t to state k shrinks at least geometrically in
  t − k. Vanishing is guaranteed.
- (b) Growth requires σ_max(W_h) > 1. This is necessary, not sufficient.
- (c) For the linear recurrence, ∂h_t/∂h_k = (W_hᵀ)^{t−k}, and ρ(W_h)^m ≤ ‖W_h^m‖₂, with ‖W_h^m‖₂^{1/m} → ρ(W_h)
  (Gelfand; Horn and Johnson 2012, Cor 5.6.14). So linear gradients explode once ρ > 1 and vanish once ρ < 1.

*Proof.* The chain rule (formalCalculus `jacobian` Thm 2), submultiplicativity of the operator norm, and
‖diag(1 − h²)‖₂ ≤ 1. For (c), an eigenpair W v = λv gives ‖W^m v‖ = |λ|^m‖v‖. ∎ This follows Bengio, Simard, and
Frasconi (1994) and Pascanu, Mikolov, and Bengio (2013, §2).

**Asserted** (`test_jacobian_bound_random`):
- on 40 random tanh RNNs with σ_max ∈ {0.5, 0.9, 1.5, 3}, measured ≤ trajectory bound ≤ σ_max^lag at every lag;
- on 20 random linear maps, ‖W⁴⁰⁰‖ ≥ ρ⁴⁰⁰ and ‖W⁴⁰⁰‖^{1/400} is within 3% of ρ.

**Corpus** (`test_jacobian_bound_corpus`, from each of 200 held-out sentences' last state). The trained RNN has
σ_max(W_h) = 3.350 and ρ(W_h) = 2.131. Neither (a) nor the linear picture applies, and **every** per-step factor
‖diag(1 − h_j²)W_hᵀ‖ has norm above 1 (min 1.62, median 2.06; 37.9% of units saturated at |h| > 0.9). Yet:

| lag m | 1 | 2 | 4 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|
| median ‖∂h_T/∂h_{T−m}‖ | 2.024 | 2.223 | 1.899 | 1.026 | 0.704 | 0.449 | 0.219 |
| median trajectory bound | 2.0 | 4.1 | 17.4 | 296 | 1,242 | 4,786 | 17,657 |

The decay lives in the products: successive factors' expanding directions do not line up, so the product's growth
rate per step, (1/m) log ‖Π‖, turns negative after about eight steps. No per-step norm can see this. It is a
Lyapunov-exponent property of the trajectory. No measured norm exceeds its bound. *Reading:* the recurrent state
lifts the trigram's two-word horizon, softly, to roughly a dozen words. That is enough to predict *footprint*
from *expanded its …* (§4's opener).

### Proposition 4 (the LSTM's cell path)
∂c_t/∂c_k = Π_{j=k+1}^{t} diag(f_j) + R_{t,k}. Here h_k = o_k ⊙ tanh(c_k), with o_k fixed, and every term of R
contains at least one factor ∂c_j/∂h_{j−1}, which is linear in the recurrent weights of gates i, f, g. In particular:
- with those three blocks of W_h at zero, ∂c_t/∂c_k = Π diag(f_j) exactly: a product with no weight matrix;
- with f ≡ 1 (the 1997 LSTM had no forget gate) it is the identity, Hochreiter and Schmidhuber's constant error
  carousel;
- with forget gates, the forget bias b_f sets the cell path at initialization to about σ(b_f)^{t−k}.

*Proof.* Write the state as s_j = (c_j, h_j) and expand ∂c_t/∂c_k as a sum over paths through the one-step block
Jacobians. ∂c_j/∂c_{j−1} = diag(f_j), because no gate reads c. A path that never leaves the c-block contributes
Π diag(f_j). Any other path enters the h-block and must return to c, since the output is c_t. The only block from
h back to c is G_j = ∂c_j/∂h_{j−1} = diag(c_{j−1} ⊙ f_j(1 − f_j))W_{h,f}ᵀ + diag(g_j ⊙ i_j(1 − i_j))W_{h,i}ᵀ +
diag(i_j ⊙ (1 − g_j²))W_{h,g}ᵀ. ∎ The full Jacobian, with its W_h factors, is subject to Thm 5-type bounds.

**Asserted** (`test_lstm_cell_path`): on 10 random LSTMs, with the three blocks zeroed, the complex-step
∂c_T/∂c₀ equals Π diag(f) to 10⁻¹⁴, with W_{h,o} left on. With them scaled by s, the residual is O(s) (its ratio at
s = 10⁻², 10⁻³ is in (5, 20)).

**The long-lag task** (`test_long_lag_memory`). Read a bit, then 60 noise symbols, then report the bit; H = 16.

| model | median gradient reaching the bit at init (12 inits) | test accuracy after 600 Adam steps |
|---|---|---|
| Elman | 2.45·10⁻¹⁰ | 0.485 |
| LSTM, b_f = 1 (Gers et al.; Jozefowicz et al. 2015) | 2.42·10⁻⁷ | 0.531 |
| LSTM, b_f = 3 | **1.42·10⁻³** | **1.000** |

The cell path predicts σ(1)⁶⁰ = 6.9·10⁻⁹ and σ(3)⁶⁰ = 0.054. The measured ratio (≈ 5,900×) is smaller than the
predicted one (≈ 8·10⁶×), because the gradient reaching the bit also passes the input gate and other paths. The
direction and the order of magnitude follow Prop 4. In scratch probes, bias 3 learned the task in 3 of 4 seeds
(the fourth reached 0.86), and bias 1 and Elman in none of 4. The harness asserts one seed. Chrono initialization
(Tallec and Ollivier 2018) sets b_f from the target lag.

### Theorem 6 (the softmax bottleneck; Yang et al. 2018)
If the logits are z(u) = s(u)W + b with a state s(u) ∈ ℝ^d, then on any N contexts the log-probability matrix is
L = [log p(w | u_i)] = SW + 𝟙bᵀ − λ𝟙ᵀ, where λ_i is the log-sum-exp of row i. So rank L ≤ d + 2, and d + 2
generically. Row-centering removes the λ term, so its row-shift class contains a matrix of rank ≤ d + 1. A
target matrix none of whose row shifts has rank ≤ d + 1 cannot be represented exactly by any such model, however
it is trained. *Proof:* the rank of a sum is at most the sum of the ranks. ∎

A mixture of K softmaxes, log Σ_k π_k softmax(s_kW_k), is not of this form, and its rank is unbounded.
**Asserted** (`test_softmax_bottleneck`):
- on random instances, rank = d + 2 exactly, while a two-component mixture exceeds it;
- **corpus**: over the 1,000 most frequent training contexts, rank 130 = h + 2 for the NPLM and 130 for the RNN
  (over 1,000 held-out prefixes), against 611 for interpolated KN.

The best rank-129 approximation of KN's row-centered matrix leaves only 1.48% of its squared Frobenius norm
(Eckart–Young, formalML `svd` Thm 2). So the bottleneck is real in rank and small in energy here. Frobenius
distance in log space is not cross-entropy, so this does not bound the loss the bottleneck costs. The bridge to #7:
a low-rank log-probability (PMI-like) matrix is what word2vec factorizes.

## 4. The honest comparison (§6) and the finance case study (§7)

**The table** (`test_neural_beats_kn_in_domain`). Closed nats per token and open bits per character, each with its
sentence-level standard error (Topic 4 Thm 2). The last column is the paired difference from interpolated KN in bpc.
All models share Topic 3's vocabulary and Topic 4's trigram speller, so the spelling bits are identical across rows
(0.0219 bpc held out, 3.6538 on EDGAR).
- λ (dev) is 0.8 for NPLM ⊕ KN and 0.9 for RNN ⊕ KN.
- The dynamic step (dev, on 400 sentences, from {0.001, 0.003, 0.01, 0.03}) is η = 0.003, on the summed loss.

| held out | nats / token (se) | PP | **bpc** (se) | bpc − KN (paired se) |
|---|---|---|---|---|
| interpolated KN (Topic 4) | 2.8551 (0.0320) | 17.38 | 0.6371 (0.0065) | — |
| NPLM | 2.6831 (0.0301) | 14.63 | 0.6001 (0.0059) | −0.0371 (0.0014) |
| NPLM ⊕ KN | 2.6688 (0.0300) | 14.42 | 0.5970 (0.0059) | −0.0401 (0.0013) |
| Elman RNN | 2.4717 (0.0296) | 11.84 | 0.5545 (0.0059) | −0.0826 (0.0017) |
| RNN ⊕ KN | 2.4593 (0.0294) | 11.70 | 0.5518 (0.0059) | −0.0853 (0.0016) |
| RNN, dynamic evaluation | 2.2919 (0.0265) | 9.89 | **0.5158** (0.0054) | −0.1213 (0.0020) |

Every network beats KN by more than 20 paired standard errors. Mixing with KN helps both networks a little. The
ranking is the same per token and per character, since all models share one tokenizer (Topic 4 Thm 5). Bits per
character is the unit that will survive comparison with subword and character models.

Parameters: NPLM 677,987, RNN 686,179, against 42,458 stored n-gram counts for KN. The networks win with 16×
more numbers. For reference, Topic 3's HPYLM scored 2.822 nats per token.

### Proposition 5 (dynamic evaluation is a code)
Topic 4's Thm 4 needs only a sequential model Q(x_i | x_{<i}). Parameters that are a deterministic function of the
text already coded qualify, because the decoder recomputes them from what it has decoded. This is prequential
coding (Dawid 1984; formalML `minimum-description-length` Def 6). Dynamic evaluation (Mikolov et al. 2010;
Krause et al. 2018) takes one SGD step on each sentence right after coding it. ∎

**Asserted** (`test_dynamic_evaluation_is_a_code`):
- the decoder replays the updates and recovers EDGAR (static and dynamic) and the first 150 held-out sentences
  exactly;
- every code is within 2 bits of −log₂ Q̃ and equals the table's open NLL to 10⁻⁶ relative.

The **whole held-out text is one codeword of 79,614 bits** (−log₂ Q = 79,612.86; 24,768 symbols; 0.5158 bpc),
against KN's 98,345 bits (19.0% shorter). EDGAR: 23,848 bits, 4.5433 bpc.

### Finance case study (§7)

`test_finance_findings`. EDGAR MD&A excerpts, 57.6% of tokens OOV:

| EDGAR | nats / token | PP | bpc |
|---|---|---|---|
| uniform over V | 8.1394 | 3,427 | — |
| interpolated KN | 8.7963 | 6,610 | 5.6605 |
| NPLM | 6.4165 | 612 | 5.1176 |
| NPLM ⊕ KN | 6.1047 | 448 | 5.0465 |
| Elman RNN | 7.2049 | 1,346 | 5.2975 |
| RNN ⊕ KN | 6.8996 | 992 | 5.2278 |
| RNN, dynamic evaluation | **3.8986** | 49.3 | **4.5432** |
| lzma, given the training text (Topic 4) | — | — | **2.9080** |

1. **The networks beat uniform where KN cannot.** KN's p(`<unk>`) is its uniform floor. The networks learned a
   `<unk>` rate (Cor 1 and word dropout), and on a 57.6%-`<unk>` text that rate dominates the per-token loss.
   Without dropout the NPLM is worse than uniform (8.2058), exactly as Cor 1 predicts.
2. **In-domain training hurts out of domain, and the ranking reverses.** At the dev-selected epoch, EDGAR is worse
   than after the first epoch while dev is better:
   - NPLM: EDGAR 5.475, 6.200, 6.533, 6.416; dev 2.929, 2.705, 2.643, 2.642;
   - RNN: EDGAR 5.868, 6.531, 6.807, 7.205; dev 2.848, 2.504, 2.443, 2.427.

   Choosing the epoch on in-domain dev picks a worse EDGAR model. The NPLM beats the RNN on EDGAR (6.42 vs
   7.20 nats; 5.12 vs 5.30 bpc), reversing the in-domain order. The RNN's longer context is longer in-domain
   memorization.
3. **Adaptivity pays.** Dynamic evaluation cuts the RNN's EDGAR loss from 7.20 to 3.90 nats per token, more
   than any static change: it learns EDGAR's own `<unk>` rate and vocabulary of contexts as it codes. This is
   Topic 4's "model that learns from the text it is coding", now a neural model.

### Proposition 6 (the spelling floor)
In any code that sends an OOV word as `<unk>` followed by its spelling under S, the bits per character are at least
Σ_OOV −log₂ S(w) / n_char. *Proof:* the word-level part costs ≥ 0 bits. ∎

On EDGAR the floor is **3.6538 bpc, above lzma's 2.9080**, so no model over this vocabulary and speller can beat
lzma there, however good its word predictions. The gap is the closed vocabulary, which is a tokenizer problem
(Topic 2) and not a modeling one. A PPM-C cache (Moffat 1990), with escape u/(n + u) in front of the speller, lowers
the spelling to 2.7608 bpc, below lzma. The dynamic RNN with the cache codes EDGAR at 3.6501 bpc, so **lzma still
wins** (held out: 0.5081 bpc with the cache). Adaptive coding is the right idea, and lzma applies it to every byte,
while our models apply it only above the word.

## 5. `rigorFlag`

> The neural models here are small networks trained once, with one seed, on a synthetic corpus: their wins over
> Kneser–Ney are evidence about this generator, not about English, and seed-to-seed variation is not measured.
> Training is non-convex (Prop 3), so nothing certifies the trained weights are optimal; the network sizes, learning
> rates, epoch counts, and the word-dropout treatment of `<unk>` were set in scratch probes, some scored on held-out
> text before the dev split existed (everything the harness itself tunes is tuned on dev). Word dropout at rate ½
> is a heuristic stand-in for Good–Turing, not an estimator with a guarantee. Three bounds here are true and
> vacuous on the trained models — the Lipschitz constant (observed ratios ≤ 0.04 of it), the tightness ε (10⁻¹⁶),
> and the spectral bound on gradients (past 10⁴ where the measured norm is 0.2): what the networks actually do is
> invisible to per-step norms. The LSTM result is proved for the cell path only, and the long-lag success is asserted
> for one seed (3 of 4 in probes). The softmax bottleneck is a rank statement; that language needs high rank is a
> hypothesis. Arithmetic codes under floating-point networks are lossless only when encoder and decoder compute
> identical floats — true in one process, not guaranteed across machines. EDGAR is 33 sentences, and no
> character-level neural model was trained.

## 6. Viz design intent — `NeuralLMLab` (one component, two linked panels)

All data comes from `viz_payload()`. Mirror it to the decimal in `src/components/viz/data/neural-lm-lab.json` and
add a Phase B `test_viz_mirror_matches_payload` (the Topic 1–4 pattern). Use Marginalia tokens only.

**Panel A, "One sentence, four models."** A sentence picker over `LAB_SENTENCES`:
1. *cedar ridge bancorp owns brackenbury lending outright* (in domain; three tokens whose bigram training never saw);
2. *pellucid software based in dublin expanded its headcount clinicbase footprint* (in domain; *footprint*
   depends on *expanded its*, five words back);
3. *book reserves are established for these plans* (EDGAR; Topic 4's Panel A sentence 3; 5 of 7 words OOV).

The main view is per-token surprisal bars (bits) for the four models (KN, NPLM, RNN, dynamic RNN), grouped per
token, with the models toggleable. Token chips are colored by event class (trigram seen, bigram only, neither).
An OOV token's spelling bits are a stacked segment shared by every model, which shows the spelling floor. Readouts
are totals per model, with and without spelling, and the sentence's bpc. Before spelling:

| sentence | KN | NPLM | RNN | dynamic RNN |
|---|---|---|---|---|
| 1 | 45.1 | **28.3** | 28.3 | 29.6 |
| 2 | 65.7 | 57.8 | **41.3** | 39.3 |
| 3 | 103.1 | 72.7 | 98.3 | **55.0** |

*Lessons:* geometry beats counts where the counts are absent (1); the recurrent state reaches past the trigram (2);
out of domain the static RNN is the overconfident one and adaptation fixes it (3).

**Panel B, "Gradients through time."** Two views behind a toggle:
- *This sentence:* for the token picked in Panel A, the trained RNN's ‖∂ℓ_t/∂h_k‖ for every earlier state k
  (bars running back in time, `sensitivity`). Under them, every per-step factor norm (`step_norms`, all above 1)
  and the line σ_max = 3.350.
- *Synthetic:* a σ slider over `THROUGH_SIGMAS` (0.5 … 3) and a tanh/linear toggle. The plot shows the median
  log₁₀ ‖∂h_T/∂h_{T−m}‖ for m = 1 … 50, overlaid with the bound σ^m (dashed) and the trained RNN's corpus curve
  (`corpus_decay`). Readouts are σ and the median ρ (ρ ≈ σ/2 for Gaussian W_h, so the linear view explodes only
  past σ ≈ 2).
- An inset shows the long-lag results: three bars of log₁₀ (gradient reaching the bit), with the accuracy labeled.

*Lessons:* σ < 1 forces vanishing; σ > 1 permits growth but tanh rarely delivers it; the trained network's factors
all expand while their product decays; the forget bias decides how far an LSTM's gradient reaches.

**Link.** Clicking a token in Panel A sets the token in Panel B's sentence view. Picking sentence 3 turns the
dynamic-RNN bars on.

Mobile: slider rows get `flexWrap: 'wrap'` and inputs `minWidth: 0` (the Topic 4 lesson).

## 7. Notebook cell map (`01_neural_language_models.ipynb`, 31 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; splits, units | — |
| 2–5 | §1 Def 1, Prop 1, Thm 1; trained NPLM; event classes | `test_softmax_is_positive_*`, `test_lipschitz_*`, `test_sharing_*` |
| 6–9 | §2 Prop 2, Thm 2, Cor 1; the `<unk>` collapse and word dropout | `test_softmax_exponential_family`, `test_score_identity_*`, `test_zero_frequency_*` |
| 10–11 | Prop 3 | `test_network_is_not_convex` |
| 12–13 | §3 Def 2, Thm 3, complex step, Cor 2 | `test_reverse_accumulation_*`, `test_nplm_gradient` |
| 14–15 | §4 Def 3, Thm 4, Cor 3; trained RNN | `test_rnn_gradient`, `test_rnn_tight` |
| 16–17 | Thm 5; the corpus decay table | `test_jacobian_bound_*` |
| 18–19 | Def 4, Prop 4; the long-lag task | `test_lstm_cell_path`, `test_long_lag_memory` |
| 20–21 | §5 Thm 6 | `test_softmax_bottleneck` |
| 22–25 | §6 the table; Prop 5, the codes | `test_neural_beats_kn_in_domain`, `test_dynamic_evaluation_is_a_code` |
| 26–27 | §7 finance; Prop 6 | `test_finance_findings` |
| 28–30 | viz payload; full harness | `test_viz_payload`, `_run_harness() == 0` |

Run:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/neural-language-models/neural_language_models.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/neural-language-models/01_neural_language_models.ipynb
```

## 8. Cross-site links (all verified `status: "published"` on GitHub `main` 2026-10-23; re-verify at the B3 gate)

| field | topic | relationship |
|---|---|---|
| `formalcalculusPrereqs` | `jacobian` | Backpropagation is the multivariate chain rule run in reverse; this page proves reverse accumulation on a general graph with fan-out and bounds products of recurrent Jacobians, both built on its Thm 2. |
| `formalcalculusPrereqs` | `gradient` | Every training step follows the gradient of the cross-entropy, and the softmax's gradient p − y is the partial-derivative calculation this page builds its backpropagation on. |
| `formalcalculusPrereqs` | `mean-value-taylor` | The smoothing-by-geometry theorem is the mean value theorem on a segment between two context embeddings, with the softmax's gradient bounded along it. |
| `formalmlPrereqs` | `gradient-descent` | The networks are trained by (adaptive) stochastic gradient descent; hidden-unit symmetry makes the objective non-convex, so its convex convergence rates do not apply here. |
| `formalmlPrereqs` | `svd` | Spectral norms bound the gradients through time, and Eckart–Young measures how much of Kneser–Ney's log-probability matrix lies beyond the softmax bottleneck's rank. |
| `formalmlConnections` | `minimum-description-length` | Dynamic evaluation is prequential coding: the network updates on each sentence after coding it, and the decoder replays the updates, so its bits per character are a code length. |
| `formalstatisticsPrereqs` | `exponential-families` | The softmax is the categorical exponential family: its Jacobian is the one-hot covariance, which makes the cross-entropy convex in the logits. |
| `formalstatisticsPrereqs` | `generalized-linear-models` | The output layer is a multinomial logit; its score identity, proved here for K classes, extends the canonical-link identity and forces moment matching at every stationary point. |
| `formalstatisticsPrereqs` | `maximum-likelihood` | Training a language model by cross-entropy is maximum likelihood, and its failure for a target never seen in training is why the <unk> token needs word dropout. |

Not cited:
- formalML `kl-divergence` (Topic 4 already carries it);
- formalML `representation-learning` (embeddings are #7's subject);
- formalCalculus `stability-dynamics` (continuous-time; Thm 5's discrete analogue is self-contained);
- formalRAG (no overlap).

## 9. References (Chicago 17th, notes-and-bibliography; every URL verified 2026-10-23 — DOI titles via Crossref)

1. Bengio, Yoshua, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. "A Neural Probabilistic Language
   Model." *Journal of Machine Learning Research* 3 (2003): 1137–55. https://www.jmlr.org/papers/v3/bengio03a.html.
   — paper; Def 1.
2. Linnainmaa, Seppo. "Taylor Expansion of the Accumulated Rounding Error." *BIT Numerical Mathematics* 16, no. 2
   (1976): 146–60. https://doi.org/10.1007/BF01931367. — paper; reverse mode.
3. Baur, Walter, and Volker Strassen. "The Complexity of Partial Derivatives." *Theoretical Computer Science* 22,
   no. 3 (1983): 317–30. https://doi.org/10.1016/0304-3975(83)90110-X. — paper; Thm 3 (ii).
4. Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning Representations by Back-Propagating
   Errors." *Nature* 323, no. 6088 (1986): 533–36. https://doi.org/10.1038/323533a0. — paper; backpropagation.
5. Elman, Jeffrey L. "Finding Structure in Time." *Cognitive Science* 14, no. 2 (1990): 179–211.
   https://doi.org/10.1207/s15516709cog1402_1. — paper; Def 3.
6. Werbos, Paul J. "Backpropagation Through Time: What It Does and How to Do It." *Proceedings of the IEEE* 78,
   no. 10 (1990): 1550–60. https://doi.org/10.1109/5.58337. — paper; Cor 3.
7. Kuhn, Roland, and Renato De Mori. "A Cache-Based Natural Language Model for Speech Recognition." *IEEE
   Transactions on Pattern Analysis and Machine Intelligence* 12, no. 6 (1990): 570–83.
   https://doi.org/10.1109/34.56193. — paper; cache models.
8. Moffat, Alistair. "Implementing the PPM Data Compression Scheme." *IEEE Transactions on Communications* 38,
   no. 11 (1990): 1917–21. https://doi.org/10.1109/26.61469. — paper; PPM-C escape (§7).
9. Bengio, Yoshua, Patrice Simard, and Paolo Frasconi. "Learning Long-Term Dependencies with Gradient Descent Is
   Difficult." *IEEE Transactions on Neural Networks* 5, no. 2 (1994): 157–66. https://doi.org/10.1109/72.279181.
   — paper; Thm 5.
10. Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-Term Memory." *Neural Computation* 9, no. 8 (1997):
    1735–80. https://doi.org/10.1162/neco.1997.9.8.1735. — paper; Def 4, the carousel.
11. Squire, William, and George Trapp. "Using Complex Variables to Estimate Derivatives of Real Functions." *SIAM
    Review* 40, no. 1 (1998): 110–12. https://doi.org/10.1137/S003614459631241X. — paper; the complex step.
12. Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to Forget: Continual Prediction with LSTM."
    *Neural Computation* 12, no. 10 (2000): 2451–71. https://doi.org/10.1162/089976600300015015. — paper; forget gate.
13. Griewank, Andreas, and Andrea Walther. *Evaluating Derivatives: Principles and Techniques of Algorithmic
    Differentiation*. 2nd ed. Philadelphia: SIAM, 2008. https://doi.org/10.1137/1.9780898717761. — book; Thm 3.
14. Mikolov, Tomáš, Martin Karafiát, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. "Recurrent Neural Network
    Based Language Model." In *Proceedings of Interspeech 2010*, 1045–48. Makuhari: ISCA, 2010.
    https://doi.org/10.21437/Interspeech.2010-343. — paper; RNN LMs, dynamic evaluation.
15. Horn, Roger A., and Charles R. Johnson. *Matrix Analysis*. 2nd ed. Cambridge: Cambridge University Press, 2012.
    https://doi.org/10.1017/CBO9781139020411. — book; Gelfand's formula (Cor 5.6.14).
16. Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the Difficulty of Training Recurrent Neural Networks."
    In *Proceedings of the 30th International Conference on Machine Learning*, PMLR 28, no. 3 (2013): 1310–18.
    https://proceedings.mlr.press/v28/pascanu13.html. — paper; Thm 5, clipping.
17. Kingma, Diederik P., and Jimmy Ba. "Adam: A Method for Stochastic Optimization." Paper presented at the 3rd
    International Conference on Learning Representations, San Diego, 2015. https://arxiv.org/abs/1412.6980. — paper;
    training.
18. Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An Empirical Exploration of Recurrent Network
    Architectures." In *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37 (2015):
    2342–50. https://proceedings.mlr.press/v37/jozefowicz15.html. — paper; forget bias 1.
19. Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. *Deep Learning*. Cambridge, MA: MIT Press, 2016.
    https://www.deeplearningbook.org/. — book; background.
20. Kiperwasser, Eliyahu, and Yoav Goldberg. "Simple and Accurate Dependency Parsing Using Bidirectional LSTM
    Feature Representations." *Transactions of the Association for Computational Linguistics* 4 (2016): 313–27.
    https://doi.org/10.1162/tacl_a_00101. — paper; word dropout to `<unk>`.
21. Chen, Yining, Sorcha Gilroy, Andreas Maletti, Jonathan May, and Kevin Knight. "Recurrent Neural Networks as
    Weighted Language Recognizers." In *Proceedings of the 2018 Conference of the North American Chapter of the ACL*,
    2261–71. New Orleans: ACL, 2018. https://aclanthology.org/N18-1205/. — paper; consistency (Thm 4 remark).
22. Krause, Ben, Emmanuel Kahembwe, Iain Murray, and Steve Renals. "Dynamic Evaluation of Neural Sequence Models."
    In *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80 (2018): 2766–75.
    https://proceedings.mlr.press/v80/krause18a.html. — paper; Prop 5.
23. Tallec, Corentin, and Yann Ollivier. "Can Recurrent Neural Networks Warp Time?" Paper presented at the 6th
    International Conference on Learning Representations, Vancouver, 2018. https://arxiv.org/abs/1804.11188. —
    paper; chrono initialization.
24. Yang, Zhilin, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. "Breaking the Softmax Bottleneck: A
    High-Rank RNN Language Model." Paper presented at the 6th International Conference on Learning
    Representations, Vancouver, 2018. https://arxiv.org/abs/1711.03953. — paper; Thm 6.
25. Du, Li, Lucas Torroba Hennigen, Tiago Pimentel, Clara Meister, Jason Eisner, and Ryan Cotterell. "A
    Measure-Theoretic Characterization of Tight Language Models." In *Proceedings of the 61st Annual Meeting of the
    ACL*, 9744–70. Toronto: ACL, 2023. https://aclanthology.org/2023.acl-long.543/. — paper; Thm 4.

Dawid (1984) is cited through Topic 4's reference list. Add it here in Phase B if the page cites it directly.

## 10. Phase B notes

- MDX at `src/content/topics/neural-language-models.mdx`. Page numbering follows page order: Defs 1–4, Props 1–6,
  Thms 1–6, Cors 1–3. Reverse-mode history, the complex step, and Gelfand's formula are prose or cited remarks.
  `references[].type`: 13, 15, and 19 are `book`; the rest are `paper`.
- Graph: flip the node's `status` to `published` and remove the title from `planned` in `curriculum.ts`. The edge
  from `perplexity-cross-entropy` already exists. Frontmatter: `prerequisites: [perplexity-cross-entropy]`, and no
  in-site `connections`.
- The lab reads `src/components/viz/data/neural-lm-lab.json` (`json.dumps(viz_payload(), separators=(',', ':'))`).
  Add `test_viz_mirror_matches_payload`. Python `inf` and `nan` never occur in this payload (`_r` maps them to `null`).
- Keep `<unk>`, `<s>`, and `</s>` out of KaTeX: name them in prose, or write ε's index as "eos" in `\text{}`. MDX
  reads a bare `<` as JSX.
- Long displays should be display math: Thm 3's update, Thm 4's ε, Thm 5's product, and Prop 4's G_j. At 375 px
  the prose column is about 343 px.
- Overview hook: sentence 1's two numbers (45.1 vs 28.3 bits) are closed-vocabulary bits on one sentence, while the
  table's bpc includes spelling. Say which is which.
- Link back to Topic 4's Thm 4 and Thm 5 from §6, and to Topic 3's Good–Turing section from Cor 1's cure. Topic 4's
  §"Where this goes next" promised both bpc reporting and adaptive prediction, and §§6–7 discharge them.
- The lab's EDGAR sentence 3 is Topic 4's Panel A sentence 3. Cross-reference it ("the sentence that cost
  interpolated KN 233 bits in Topic 4's lab").
- Re-run the harness at the B3 gate. The harness is ≈ 52 s; if a lab addition pushes it past 60 s, trim
  `DECAY_SENTENCES` or `LAG_STEPS` before anything else, and re-verify the lag-60 assertion.
