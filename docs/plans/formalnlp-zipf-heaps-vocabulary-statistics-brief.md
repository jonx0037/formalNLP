# Brief — `zipf-heaps-vocabulary-statistics`

**Zipf's and Heaps' Laws: The Statistics of Vocabulary.** formalNLP's first topic, the root of the
prerequisite DAG, and the in-repo structural exemplar for every later topic.

- Phase A (this brief + verified notebook): done 2026-10-22. Harness 26/26 under numpy==2.5.4 /
  scipy==1.18.1 in ≈ 2.5 s; narrative notebook executes (exit 0, committed without outputs).
- Scaffold decisions (Jonathan, 2026-10-22): finance case study uses the shared corpus as a
  **planted-truth** test bed; keep the Mandelbrot max-entropy section (and the formalML
  `shannon-entropy` prereq); **one lab, two linked panels**; Claude writes the MLE.
- Source of truth: `notebooks/zipf-heaps-vocabulary-statistics/zipf_heaps_vocabulary_statistics.py`.
  Every number below is printed by it; nothing here may be changed without changing it.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Zipf's and Heaps' Laws: The Statistics of Vocabulary |
| `{{SUBTITLE}}` | Why a vocabulary's rank–frequency curve and its growth curve are one power law seen twice — and why a straight line through a log–log plot is the wrong way to measure it. |
| `{{SLUG}}` | `zipf-heaps-vocabulary-statistics` (= graph node id) |
| `{{DOMAIN}}` | `text-as-data` |
| `{{DIFFICULTY}}` | `foundational` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[]` (root of the DAG) |
| `{{FORMALML_PREREQS}}` | `shannon-entropy`, `bayesian-nonparametrics` |
| `{{FORMALRAG_PREREQS}}` | none (connection: `vector-space-model-tfidf`) |
| `{{STATS_CALC_PREREQS}}` | Stats `discrete-distributions`, `maximum-likelihood`; Calc `series-convergence`, `improper-integrals` |
| `{{RIGOR_FLAG}}` | §5 below |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/finance-corpus/` |

Tags: `zipf`, `heaps`, `power-laws`, `vocabulary-growth`, `pitman-yor`, `maximum-likelihood`.
`estimatedReadTime` ≈ 35. In-site `connections`: none yet — all five dependents
(`subword-tokenization-bpe-unigram`, `n-gram-models-smoothing`, `distributional-semantics-ppmi-svd`,
`linear-text-classifiers`, `collocations-keyword-extraction`) are still planned, and `pnpm validate`
requires both endpoints to resolve to MDX. The graph edges already exist; nothing to add there.

**Scope against siblings.** Not on the overlap watch-list. formalRAG's `vector-space-model-tfidf` and
`query-likelihood-language-models` *use* term statistics (IDF, smoothing); this topic owns the
*distribution* of those statistics and stops there — no weighting schemes, no smoothing estimators
(Good–Turing/Kneser–Ney belong to `n-gram-models-smoothing`; we only foreshadow the hapax mass).
formalML's `bayesian-nonparametrics` defines the Dirichlet process/CRP but not Pitman–Yor; we
introduce the discount here and cite back for the CRP.

## 1. Section outline (geometric first; each section opens with a picture or a corpus number)

| § | Title | Opens with | Math | Notebook |
|---|---|---|---|---|
| 1 | The shape of a vocabulary | Corpus: 63,182 tokens, 3,425 types, 54.8% seen once; top types `and in the of for percent …`; log-spaced rank/count table | Notation: tokens $n$, types $V(n)$, counts $f(r)$, spectrum $V_m$ | cells 1–3 |
| 2 | Rank–frequency is a tail turned sideways | The rank plot and the count CCDF drawn as the same staircase, axes swapped | Def 1–2, **Prop 1**, Cor 1 | cells 4–5 |
| 3 | Zipf as a probability law | Partial sums of $\sum r^{-s}$ for $s = 1$ vs $1.5$ | Def 3, **Prop 2** ($\zeta$), **Thm 3** (max-entropy) | cells 6–7 |
| 4 | Zipf implies Heaps | $\mathbb{E}V(n)$ against $\Gamma(1-1/s)(Cn)^{1/s}$ on log–log axes | **Lemma A**, **Thm 1** (Karlin), Cor 2 (hapax share), Rem (finite vocabulary saturates) | cells 8–11 |
| 5 | Pitman–Yor plants both laws | A restaurant filling up; $K_n$ path vs. exact mean | Def 4, **Thm 2** (exact $\mathbb{E}K_n$), **Lemma B** (Wendel), **Thm 4** (cited: Pitman 2006 Lemma 3.11) | cells 12–15 |
| 6 | Estimating the exponent | OLS slopes by rank window on the corpus: −0.871, −0.554, −1.807, −0.821 | Def 5, **Prop 3** (MLE: concavity, score equation, uniqueness); OLS critique (empirical, asserted) | cells 16–19 |
| 7 | Finance case study: a planted answer | Full text looks like real text ($s_{\text{OLS}} = 0.917$, $\beta = 0.516$) | Applies Thm 2 as an estimator | cells 20–23 |
| 8 | Where this goes next | — | Forward links: open vocabulary → subwords; hapax mass → Good–Turing; head → stopwords/IDF; Kneser–Ney ≈ hierarchical PY (Teh 2006) | cells 24–26 |

Lab embed: after §2's picture (Panel A is the rank plot) and referenced again in §5 (Panel B).
One `<VocabularyLawsLab client:visible />`, placed at the end of §4 so both panels have their theory
on the page before the reader plays; §6 and §7 point back to Panel A's window readout.

## 2. Definitions

Throughout: a corpus is a token sequence $w_1, \dots, w_N$ over a vocabulary of types.

- **Def 1 (counts, ranks).** $c(w)$ is the number of tokens of type $w$; the $V$ observed types sorted
  by decreasing count give $f(1) \ge f(2) \ge \dots \ge f(V)$, ties ordered arbitrarily. $f(r)$ is
  the rank–frequency curve.
- **Def 2 (spectrum, tail count).** $V_m = \#\{w : c(w) = m\}$; $N_{\ge}(f) = \#\{w : c(w) \ge f\} =
  \sum_{m \ge f} V_m$. $V_1$ counts the *hapax legomena*.
- **Def 3 (Zipf law).** For $s > 1$, $p_r = C r^{-s}$ on $\{1, 2, \dots\}$ with $C = 1/\zeta(s)$; the
  truncated law on $\{1, \dots, R\}$ is normalized by $H_{R,s} = \sum_{r \le R} r^{-s}$, any $s \in \mathbb{R}$.
- **Def 4 (Pitman–Yor restaurant).** $0 \le d < 1$, $\theta > 0$. Customer 1 opens table 1. Given
  $K_n$ tables with occupancies $c_1, \dots, c_{K_n}$, customer $n+1$ opens a new table with probability
  $(\theta + dK_n)/(\theta + n)$ and joins table $k$ with probability $(c_k - d)/(\theta + n)$. Tables are
  word types; $d = 0$ is the Chinese restaurant process of formalML's `bayesian-nonparametrics`.
  (PY allows $-d < \theta \le 0$ too; we need only $\theta > 0$.)
- **Def 5 (discrete power law).** For integer $x_{\min} \ge 1$ and $\alpha > 1$,
  $P(X = x) = x^{-\alpha}/\zeta(\alpha, x_{\min})$, $x \ge x_{\min}$, with the Hurwitz zeta
  $\zeta(\alpha, x_{\min}) = \sum_{x \ge x_{\min}} x^{-\alpha}$.

## 3. Statements and proofs

### Proposition 1 (rank is the tail count)
If ties are ranked last, the rank of any type with count $f$ is $N_{\ge}(f)$; in particular
$r \mapsto f(r)$ and $f \mapsto N_{\ge}(f)$ are generalized inverses.

*Proof.* The types ranked at or above a type $w$ with $c(w) = f$ are exactly those with count $> f$
together with all types with count $f$ (ties last puts $w$ at the end of its tie block). Their number is
$\#\{c > f\} + \#\{c = f\} = N_{\ge}(f)$. ∎

*Geometric reading.* Plot $N_{\ge}(f)$ against $f$ (the unnormalized complementary CDF of the type
counts) and reflect across the diagonal: that is the rank–frequency staircase. **Asserted exactly** on
all 3,425 corpus types (`test_rank_is_count_tail`).

### Corollary 1 (exponent map $\alpha = 1 + 1/s$)
If $N_{\ge}(f) = A f^{-(\alpha - 1)}$ on a range of $f$, then on the corresponding ranks
$f(r) = (A/r)^{1/(\alpha - 1)}$: a rank exponent $s = 1/(\alpha - 1)$. If the spectrum itself has
$V_m = B m^{-\alpha}$, then $N_{\ge}(f) = B\,\zeta(\alpha, f) = \frac{B}{\alpha - 1} f^{-(\alpha-1)}(1 + O(1/f))$,
the same exponent.

*Proof.* Invert Prop 1's identity. For the second claim, $\zeta(\alpha, f)$ is a decreasing sum, so
$\int_f^\infty x^{-\alpha}dx \le \zeta(\alpha, f) \le f^{-\alpha} + \int_f^\infty x^{-\alpha}dx$, i.e.
$\zeta(\alpha, f) = \frac{f^{1-\alpha}}{\alpha - 1}(1 + O(1/f))$. ∎

**Asserted** on 200k-draw power-law samples, $\alpha \in \{1.6, 2.0, 2.5\}$: rank slope within 0.12
of $-1/(\alpha - 1)$ (`test_spectrum_exponent_maps_to_rank_exponent`).

### Proposition 2 ($\zeta(s) < \infty$ iff $s > 1$)
*Proof.* $x \mapsto x^{-s}$ is positive and decreasing for $s > 0$, so by the integral test
(formalCalculus `series-convergence`) $\sum_{r=1}^{R} r^{-s}$ is within 1 of $\int_1^R x^{-s}dx$, which stays
bounded iff $s > 1$; for $s \le 0$ the terms do not tend to 0. At $s = 1$, $H_R = \log R + \gamma + O(1/R)$. ∎

Consequence: an infinite-vocabulary Zipf law needs $s > 1$; real text's fitted $s \approx 1$ sits on
the boundary (rigorFlag). **Asserted**: partial sum + Hurwitz tail $= \zeta(s)$ to $10^{-9}$;
$|H_R - \log R - \gamma| < 1/R$ (`test_zeta_converges_iff_s_above_one`).

### Theorem 3 (Zipf maximizes entropy at fixed mean log-rank — Mandelbrot; Visser 2013)
Fix $R \ge 2$ and $\mu \in (0, \log R)$. Among distributions $q$ on $\{1, \dots, R\}$ with
$\sum_r q_r \log r = \mu$, the Shannon entropy $H(q) = -\sum q_r \log q_r$ has the unique maximizer
$p_r = r^{-s}/H_{R,s}$, where $s \in \mathbb{R}$ is the unique solution of $\mathbb{E}_{p}[\log r] = \mu$.

*Proof.* *Existence of $s$.* $m(s) = \mathbb{E}_{p^{(s)}}[\log r]$ is continuous with
$m'(s) = -\mathrm{Var}_{p^{(s)}}(\log r) < 0$ (differentiate $\log H_{R,s}$ twice; the variance is
positive because $\log r$ is non-constant for $R \ge 2$), and $m(s) \to \log R$ as $s \to -\infty$,
$m(s) \to 0$ as $s \to \infty$. So $m$ is a bijection onto $(0, \log R)$.
*Optimality.* For any feasible $q$, Gibbs' inequality ($\mathrm{KL}(q\,\|\,p) \ge 0$, formalML
`shannon-entropy`) gives
$H(q) \le -\sum q_r \log p_r = \sum q_r (s \log r + \log H_{R,s}) = s\mu + \log H_{R,s}$, and the right
side equals $H(p)$ by the same computation with $q = p$. Equality iff $\mathrm{KL}(q\|p) = 0$ iff $q = p$. ∎

*Reading.* $s$ is the Lagrange multiplier on the log-rank constraint: the "price" of spreading mass to
high ranks. **Asserted**: $R = 40$, $s = 1.3$, $\mu = 1.1703$: root-finding recovers $s$ to $10^{-9}$,
SLSQP from uniform lands on Zipf to $10^{-4}$, and 50 random feasible perturbations all lower $H$
(`test_zipf_maximizes_entropy_at_fixed_log_rank`).

### Lemma A (sums of unimodal functions)
Let $\varphi \ge 0$ be integrable on $[0, \infty)$, nondecreasing on $[0, x^\ast]$ and nonincreasing on
$[x^\ast, \infty)$, with $M = \sup \varphi$. Then
$\int_0^\infty \varphi - M \le \sum_{r \ge 1} \varphi(r) \le \int_0^\infty \varphi + 2M$.
If $\varphi$ is nonincreasing ($x^\ast = 0$): $\int_0^\infty \varphi - \int_0^1 \varphi \le \sum_{r\ge1}\varphi(r) \le \int_0^\infty \varphi$.

*Proof.* Let $m = \lfloor x^\ast \rfloor$. Lower: for $1 \le r \le m$, $\varphi(r) \ge \int_{r-1}^r \varphi$
($\varphi$ rises on $[r-1, r]$); for $r \ge m + 1$, $\varphi(r) \ge \int_r^{r+1}\varphi$. These intervals
cover $[0, \infty)$ except $[m, m+1]$, whose integral is $\le M$. Upper: for $r \le m - 1$,
$\varphi(r) \le \int_r^{r+1}\varphi$; for $r \ge m + 2$, $\varphi(r) \le \int_{r-1}^r \varphi$; the
intervals are disjoint, and the two remaining terms $\varphi(m), \varphi(m+1)$ are each $\le M$. The
monotone case is the same argument with $m = 0$. ∎

### Theorem 1 (Zipf ⇒ Heaps; Karlin 1967)
Let tokens be i.i.d. with $p_r = Cr^{-s}$, $s > 1$, $C = 1/\zeta(s)$, and
$I(n) = \Gamma(1 - 1/s)\,(Cn)^{1/s}$. Then
$$I(n) - 1 \;\le\; \mathbb{E}\,V(n) \;\le\; I(n) + \varepsilon_n, \qquad
\varepsilon_n \le \frac{1}{(1 - C)\,n}\Big[\tfrac{1}{s}\Gamma\big(2 - \tfrac1s\big)(Cn)^{1/s} + 8e^{-2}\Big] = O(n^{1/s - 1}).$$
In particular $\mathbb{E}V(n) \sim I(n)$, $\log \mathbb{E}V(n) = \frac1s \log n + \text{const} + o(1)$:
Heaps' law with $\beta = 1/s$, with **absolute** error at most about one type.

*Proof.* Type $r$ is unseen after $n$ draws with probability $(1-p_r)^n$, so
$\mathbb{E}V(n) = \sum_r [1 - (1-p_r)^n]$.
*Step 0 (the integral).* With $g(x) = 1 - e^{-nCx^{-s}}$ and $u = nCx^{-s}$,
$I := \int_0^\infty g = \frac{(nC)^{1/s}}{s}\int_0^\infty (1 - e^{-u})u^{-1-1/s}du$. Integrate by parts
with $v = -s\,u^{-1/s}$: the boundary term $-s u^{-1/s}(1 - e^{-u})$ vanishes at $\infty$ and, since
$1 - e^{-u} \le u$ and $s > 1$, at $0$; what remains is $s\int_0^\infty u^{-1/s}e^{-u}du = s\,\Gamma(1 - 1/s)$
(formalCalculus `improper-integrals`). So $I = \Gamma(1-1/s)(Cn)^{1/s}$.
*Step 1 (Poissonize).* $g$ is decreasing with $0 \le g \le 1$, so Lemma A gives
$I - 1 \le \sum_{r \ge 1} g(r) \le I$.
*Step 2 (de-Poissonize).* $1 - p \le e^{-p}$ gives $(1-p)^n \le e^{-np}$, so
$\mathbb{E}V(n) \ge \sum_r g(r) \ge I - 1$. For the other side, $-\log(1-p) = \sum_k p^k/k \le p + p^2/(1-p)$,
so $(1-p)^n \ge e^{-np}e^{-np^2/(1-p)} \ge e^{-np}(1 - np^2/(1-p))$ and
$0 \le e^{-np_r} - (1-p_r)^n \le \frac{np_r^2 e^{-np_r}}{1 - p_r} \le \frac{h(np_r)}{(1-C)\,n}$, with
$h(u) = u^2 e^{-u}$ and $p_r \le p_1 = C < 1$. As a function of $x$, $h(nCx^{-s})$ is unimodal with
maximum $h(2) = 4e^{-2}$, and the substitution of Step 0 gives
$\int_0^\infty h(nCx^{-s})dx = \frac{1}{s}\Gamma(2 - \frac1s)(Cn)^{1/s}$; Lemma A bounds the sum by that
plus $8e^{-2}$. Summing over $r$ gives $\mathbb{E}V(n) - \sum_r g(r) \le \varepsilon_n$. ∎

**Asserted** for $s \in \{1.25, 1.5, 2, 3\}$, $n \in \{10^2, \dots, 10^6\}$ (infinite support: exact head
$r \le 2\cdot10^6$, tail by its second-order Taylor series): the sandwich (Step 1), the sign and
$o(n^{1/s})$ size of the Step 2 gap, ratio $\to 1$ within 0.005 at $n = 10^6$, and the $10^5$–$10^6$ log–log
slope within 0.005 of $1/s$ (`test_karlin_*`). Ratios $\mathbb{E}V(n)/I(n)$:

| $s$ | $n=10^2$ | $10^3$ | $10^4$ | $10^5$ | $10^6$ |
|---|---|---|---|---|---|
| 1.25 | 0.9915 | 0.9986 | 0.9998 | 1.0000 | 1.0000 |
| 1.5 | 0.9847 | 0.9966 | 0.9992 | 0.9998 | 1.0000 |
| 2.0 | 0.9651 | 0.9887 | 0.9964 | 0.9989 | 0.9996 |
| 3.0 | 0.9165 | 0.9609 | 0.9818 | 0.9915 | 0.9961 |

Convergence is slowest for **large** $s$: the error is $O(1)$ types against $I \sim n^{1/s}$ types.

**Remark (Heaps needs an open vocabulary).** A truncated Zipf law on $R$ types has $V(n) \le R$. For
$R = 10^4$, $s = 1.25$: $\mathbb{E}V = 270.4, 1358.4, 9808.2, 10000.0$ at $n = 10^3, 10^4, 10^6, 10^7$ — the
local log–log slope falls from above $0.6/s$ to below $0.25/s$ (`test_finite_vocabulary_saturates`).
This is the bridge to subword tokenization: a closed word list cannot keep pace with an open vocabulary.

### Corollary 2 (the hapax share estimates $\beta$)
Under Theorem 1's hypotheses, $\mathbb{E}V_1(n) = \frac1s\Gamma(1 - \frac1s)(Cn)^{1/s} + O(1)$, so
$\mathbb{E}V_1(n)/\mathbb{E}V(n) \to 1/s = \beta$.

*Proof.* $\mathbb{E}V_1(n) = \sum_r np_r(1-p_r)^{n-1}$. Poissonized, $\varphi(x) = ue^{-u}$ with
$u = nCx^{-s}$ is unimodal in $x$ with maximum $e^{-1}$, and Step 0's substitution gives
$\int_0^\infty\varphi = \frac1s\Gamma(1-\frac1s)(Cn)^{1/s}$; Lemma A gives the $O(1)$. De-Poissonizing,
$|(1-p)^{n-1} - e^{-np}| \le p(1-p)^{n-1} + [e^{-np} - (1-p)^n] \le e\,p\,e^{-np} + \frac{np^2e^{-np}}{1-p}$,
so the total error is at most $\frac{e}{n}\sum_r h(np_r) + \frac{1}{(1-C)n}\sum_r u_r^3e^{-u_r}$, which is
$O(n^{1/s-1})$ by the same unimodal bound. Divide by Theorem 1. ∎

**Asserted** at $n = 10^6$: hapax share 0.6667, 0.5002, 0.3346 for $s = 1.5, 2, 3$ (`test_karlin_hapax_share`).

### Theorem 2 (exact mean table count of a Pitman–Yor restaurant)
For $0 < d < 1$, $\theta > 0$:
$$\mathbb{E}K_n + \frac{\theta}{d} \;=\; \frac{\theta}{d}\,\frac{\Gamma(\theta + n + d)\,\Gamma(\theta)}{\Gamma(\theta + d)\,\Gamma(\theta + n)}
\;\sim\; \frac{\Gamma(\theta + 1)}{d\,\Gamma(\theta + d)}\,n^{d},$$
and as $d \to 0$, $\mathbb{E}K_n \to \sum_{i=0}^{n-1}\frac{\theta}{\theta + i} = \theta\,[\psi(\theta + n) - \psi(\theta)] \sim \theta\log n$
(the Dirichlet process).

*Proof.* By Def 4, $\mathbb{E}[K_{n+1} \mid K_n] = K_n + \frac{\theta + dK_n}{\theta + n}$; take expectations
(the map is affine in $K_n$): $\mathbb{E}K_{n+1} = \mathbb{E}K_n + \frac{\theta + d\,\mathbb{E}K_n}{\theta+n}$.
Put $M_n = \mathbb{E}K_n + \theta/d$. Then $\theta + d\,\mathbb{E}K_n = dM_n$, so
$M_{n+1} = M_n\,\frac{\theta + n + d}{\theta + n}$, $M_0 = \theta/d$, and telescoping with
$\Gamma(x + 1) = x\Gamma(x)$ gives the closed form. The asymptotic follows from Lemma B with
$x = \theta + n$: $\Gamma(\theta+n+d)/\Gamma(\theta+n) \sim (\theta + n)^d \sim n^d$, and
$\theta\Gamma(\theta) = \Gamma(\theta+1)$. The $d \to 0$ case is the same recursion with $d = 0$:
$\mathbb{E}K_{n+1} - \mathbb{E}K_n = \theta/(\theta + n)$. ∎

*Reading.* It is $\mathbb{E}K_n + \theta/d$, not $\mathbb{E}K_n$, that is a clean power of $n$. The
offset is invisible asymptotically but large at corpus scale ($\theta/d = 33.3$ for the corpus's
restaurants), and it is exactly why a naive log–log Heaps fit overshoots $d$ (§7).

**Asserted**: the closed form satisfies the recursion to $10^{-10}$ for four $(d, \theta)$; Monte-Carlo
means over 200 restaurants within 3.5 SE for three $(d, \theta)$; our O(1) sampler matches
`finance_corpus._PitmanYor`; $(\mathbb{E}K_n + \theta/d)/(cn^d) - 1 = O(1/n)$ to $n = 10^9$; the
$d \to 0$ limit (`test_py_*`). For the corpus parameters ($d = 0.6$, $\theta = 20$):
$\mathbb{E}K_n = 64.8, 321.5, 1364.2, 22090.8$ at $n = 10^2, 10^3, 10^4, 10^6$, and
$\mathbb{E}K_n/(cn^d) = 0.7360, 0.9168, 0.9773, 0.9985$.

### Lemma B (Wendel 1948)
For $x > 0$, $0 < d < 1$: $\left(\frac{x}{x+d}\right)^{1-d} \le \frac{\Gamma(x+d)}{x^d\,\Gamma(x)} \le 1$.

*Proof.* Hölder with exponents $\frac{1}{1-d}, \frac1d$ on $t^{x+d-1}e^{-t} = (t^{x-1}e^{-t})^{1-d}(t^{x}e^{-t})^{d}$:
$\Gamma(x+d) \le \Gamma(x)^{1-d}\Gamma(x+1)^d = x^d\Gamma(x)$. Applying this at $x + d$ with exponent
$1 - d$: $\Gamma(x+1) \le (x+d)^{1-d}\Gamma(x+d)$, i.e. $\Gamma(x+d) \ge x\Gamma(x)/(x+d)^{1-d}$. ∎

### Theorem 4 (cited — Pitman 2006, Lemma 3.11, with Theorem 3.8 / Corollary 3.9)
For a PY$(d, \theta)$ restaurant, $0 < d < 1$: almost surely (i) $K_n/n^d \to S_d \in (0, \infty)$, a
random limit (the *$d$-diversity*); (ii) the ranked table frequencies satisfy $P^\downarrow_i \sim Z\,i^{-1/d}$
for a random $0 < Z < \infty$ — a Zipf law with $s = 1/d$; (iii) the share of tables of size $m$
converges, $K_{n,m}/K_n \to p_d(m) = \frac{d\,(1-d)_{m-1\uparrow}}{m!} = \frac{d\,\Gamma(m-d)}{\Gamma(1-d)\,m!}$
(the Sibuya law), whose tail is $p_d(m) \sim \frac{d}{\Gamma(1-d)}m^{-(1+d)}$.

Not proved here (the proof runs through Kingman's representation and regular variation); stated, cited,
and **asserted** on $2\times10^5$-customer restaurants, $d \in \{0.4, 0.6, 0.8\}$: shares for $m = 1..5$
within 4 binomial SE of $p_d(m)$, KS-selected power-law $\hat\alpha$ within 0.1 of $1 + d$
(`test_py_spectrum_is_sibuya`). Consistency with Theorem 1 and Prop 1: $\alpha = 1 + d \Rightarrow s = 1/d$
and $\beta = d = 1/s$; and $p_d(1) = d$ is Cor 2's hapax share. One parameter sets both laws.

Note for Panel B: (i) says a single restaurant's $K_n$ tracks $S_d\,n^d$ with a **random** constant, so a
single path sits visibly off $\mathbb{E}K_n$ (e.g. $d = 0.6$: $K_{20000} = 1527$ vs.
$\mathbb{E}K = 2083.7$) while its *slope* is right. The lab should say so.

### Proposition 3 (the discrete power-law MLE)
Let $x_1, \dots, x_n \ge x_{\min}$ be type counts (Def 5) and
$\ell(\alpha) = -n\log\zeta(\alpha, x_{\min}) - \alpha\sum_i\log x_i$. Then $\ell$ is strictly concave on
$(1, \infty)$, $\ell'(\alpha) = n\,(\mathbb{E}_\alpha[\log X] - \overline{\log x})$, and if
$\overline{\log x} > \log x_{\min}$ (not every count equals $x_{\min}$) the MLE $\hat\alpha$ exists, is
unique, and solves the score equation $\overline{\log x} = -\partial_\alpha\log\zeta(\alpha, x_{\min})$.

*Proof.* $\log\zeta(\alpha, x_{\min}) = \log\sum_{x \ge x_{\min}} e^{-\alpha\log x}$ is a log-partition
function: its first derivative is $-\mathbb{E}_\alpha[\log X]$ and its second is
$\mathrm{Var}_\alpha(\log X) > 0$. Hence $\ell'' = -n\,\mathrm{Var}_\alpha(\log X) < 0$ and the score has the
stated form. $\alpha \mapsto \mathbb{E}_\alpha[\log X]$ is continuous and strictly decreasing; it tends to
$\log x_{\min}$ as $\alpha \to \infty$ (mass concentrates on $x_{\min}$) and to $+\infty$ as $\alpha \downarrow 1$
(for any $L$, the finitely many $x < e^L$ carry bounded weight while $\zeta(\alpha, x_{\min}) \to \infty$).
So $\ell'$ has exactly one root when $\overline{\log x} > \log x_{\min}$. ∎

This is a one-parameter exponential family in $\alpha$ with sufficient statistic $\sum\log x_i$, so
consistency and asymptotic normality with Fisher information $\mathrm{Var}_\alpha(\log X)$ follow from
formalStatistics `maximum-likelihood`; Clauset, Shalizi, and Newman's working standard error is
$(\hat\alpha - 1)/\sqrt n$ (continuous approximation). $x_{\min}$ is chosen by minimizing the
Kolmogorov–Smirnov distance between the empirical and fitted tail CDFs (Clauset et al. 2009) — a
heuristic, not a theorem.

**Asserted**: MLE recovers $\alpha \in \{1.6, 2.2\}$ within 4 SE on $2\times10^4$ draws and matches
`scipy.stats.fit` to $10^{-4}$ at $x_{\min} = 1$; the score equation holds to $10^{-5}$ on the corpus for
$x_{\min} \in \{1, 5, 10\}$; the continuous approximation is within 0.03 for $x_{\min} \ge 6$.

**The OLS critique (empirical, asserted — not a theorem).** On 40 samples with $\alpha = 1.6$
($s = 5/3$): OLS on the log–log frequency *histogram* is badly biased (mean error > 20× the MLE's);
OLS on the rank plot is closer but still more than 2× the MLE's error, and neither OLS fit has a valid
standard error (`test_ols_is_biased_where_mle_is_not`). On the corpus, dense OLS slopes by window:

| ranks | 1–10 | 10–100 | 100–1000 | 1000–3425 | 1–1000 | 1–3425 |
|---|---|---|---|---|---|---|
| slope | −0.871 | −0.554 | −1.807 | −0.821 | −1.300 | −1.464 |

Full-text MLE: KS-selected $x_{\min} = 2$, $\hat\alpha = 1.5608$, $\hat s = 1.7833$ (line anchored at
$r(2) = 1548$ by Prop 1).

## 4. Finance case study (§7) — the planted answer

`finance_corpus.py` draws content words from seven PY$(d = 0.6, \theta = 20)$ restaurants (one shared,
one per sector) inside closed-class templates. We rebuild `generate_corpus(300, 0)` with each
restaurant's draws recorded; the rebuild is token-identical to the fingerprinted corpus
(`test_finance_rebuild_matches_corpus`). Truth: $\beta = 0.6$, $s = 1/0.6 \approx 1.667$.

The story, in three beats (all asserted):

1. **Full text looks like real text.** Log-spaced OLS $s = 0.917$, Heaps $\beta = 0.516$ — the familiar
   real-text pairing that violates $\beta = 1/s$ ($1/s = 1.091$) by about 2×, although one parameter
   generated everything. A closed-class head on an open-class tail is enough to produce the tension
   Ferrer i Cancho and Solé (2001) describe as two regimes. Template types saturate (≥ 90% appear in the
   first half of the corpus), pulling full-text $\beta$ below the content $\beta$.
2. **Generic estimators miss the planted truth.** Content stream ($N = 12{,}107$, $V = 2{,}909$): naive
   Heaps slope 0.681 (overshoots: ignores Theorem 2's offset), hapax share 0.644 (→ $d$ only
   asymptotically), power-law MLE $\hat\alpha$ ranges 1.902–2.163 over $x_{\min} \in \{2,3,5,8,10,15\}$ — far
   above $1 + d = 1.6$, because the Sibuya spectrum is a power law only for counts $\gg 1$.
3. **Theorem 2 recovers it.** On the shared restaurant ($n = 6{,}007$, $V = 1{,}108$ vs.
   $\mathbb{E}K = 996.8$): naive Heaps 0.628, offset-corrected fit of $\log(K_n + \theta/d)$ on
   $\log(n + \theta)$ gives **0.6009**. (The fit uses the known $\theta$; $d$ enters only through the
   shift, so this is a consistency check of the model, not a blind estimate — say so on the page.)

EDGAR excerpts (782 words) are not used: too small to fit any tail.

## 5. `rigorFlag`

> Zipf's and Heaps' laws are empirical regularities, not theorems about language. What is proved here is
> conditional: if tokens are i.i.d. from a Zipf law with s > 1, the expected vocabulary grows like
> n^{1/s} (Karlin); if types come from a Pitman–Yor process, both laws hold with s = 1/d and β = d. Real
> text is neither i.i.d. nor exactly Zipfian — fitted s ≈ 1 sits on the boundary where an infinite
> vocabulary cannot be normalized, and β = 1/s routinely fails on real corpora. Log–log least-squares
> slopes depend on the fitting window and are biased; the power-law MLE assumes a pure power tail above
> x_min, which a Pitman–Yor spectrum reaches only for large counts, and x_min is chosen by a heuristic.
> The Sibuya and Zipf limits of Pitman–Yor (Pitman 2006) are cited, not proved, and checked numerically.

## 6. Viz design intent — `VocabularyLawsLab` (one component, two linked panels)

All data from `viz_payload()`; mirror to the decimal with a comment citing `test_viz_payload` /
the relevant test. Marginalia tokens only.

**Panel A — "Fit the line."** Log–log scatter of the corpus rank–frequency curve (60 log-spaced points,
`payload.corpus`). The reader picks a fitting window from the six presets (`payload.windows`,
two-handle brush snapping to them); the OLS line redraws over the window and the readout shows its
slope. A fixed MLE line ($x_{\min} = 2$, $\hat s = 1.783$, through $(1548, 2)$) stays put.
*Lesson:* the OLS slope swings from −0.554 to −1.807 with the window; the MLE answers a different,
well-posed question. A shaded band marks the template head (ranks ≲ 100) vs. the open-class tail.

**Panel B — "Turn the discount."** Slider over $d \in \{0.3, \dots, 0.8\}$ (`payload.growth`, θ = 20,
n = 20,000). Main plot: $K_n$ path (points) and exact $\mathbb{E}K_n$ (curve) on log–log axes. Readouts:
naive Heaps slope vs. offset-corrected slope vs. $d$ (corrected within 0.03 of $d$ for every $d$; naive
always above). Inset: that restaurant's rank–frequency points with a reference slope $-1/d$.
*Lessons:* one knob moves both exponents ($\beta = d$, $s = 1/d$); the offset matters at finite $n$;
a single path's level is random (Thm 4(i)) while its slope is not.

**Link.** A shared exponent strip under both panels: "$\beta$ = … ⇔ $s = 1/\beta$ = …", fed by Panel B's
$d$, with Panel A's current OLS slope shown against it — the reader sees the corpus's $0.917$ vs. the
planted $1.667$.

## 7. Notebook cell map (`01_zipf_heaps_vocabulary_statistics.ipynb`, 27 cells)

| cells | content | asserts run |
|---|---|---|
| 0–1 | intro; imports, corpus stream summary | — |
| 2–3 | §1 log-spaced rank/count table | — |
| 4–5 | §2 Prop 1, spectrum, exponent map | `test_rank_is_count_tail`, `test_spectrum_exponent_maps_to_rank_exponent` |
| 6–7 | §3 ζ convergence, max-entropy | `test_zeta_*`, `test_zipf_maximizes_entropy_*` |
| 8–11 | §4 Karlin ratios table; saturation; hapax share | `test_karlin_*`, `test_finite_vocabulary_saturates` |
| 12–15 | §5 Theorem 2, MC, sampler cross-check; Sibuya shares | `test_py_*` |
| 16–19 | §6 OLS windows; MLE, scipy cross-check, OLS bias, KS $x_{\min}$ | `test_ols_*`, `test_mle_*`, `test_continuous_*` |
| 20–23 | §7 finance: real-text tension, saturation; generic miss; Theorem 2 recovery | `test_finance_*`, `test_full_text_*`, `test_templates_*`, `test_generic_*`, `test_theorem_two_*` |
| 24–26 | §8 viz payload printout; full harness | `_run_harness() == 0` |

Run:
```bash
uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/zipf-heaps-vocabulary-statistics/zipf_heaps_vocabulary_statistics.py
uv run --with numpy==2.5.4 --with scipy==1.18.1 --with jupyter --with ipykernel jupyter execute notebooks/zipf-heaps-vocabulary-statistics/01_zipf_heaps_vocabulary_statistics.ipynb
```

## 8. Cross-site links (all verified `status: "published"` on 2026-10-22)

| field | topic | relationship |
|---|---|---|
| `formalstatisticsPrereqs` | `discrete-distributions` | The Zipf and discrete power laws are discrete distributions on the positive integers; we use their pmfs, tails, and normalizers throughout. |
| `formalstatisticsPrereqs` | `maximum-likelihood` | The power-law exponent is estimated by maximum likelihood; consistency and the Fisher-information standard error come from the general MLE theory. |
| `formalcalculusPrereqs` | `series-convergence` | Whether a Zipf law can live on an infinite vocabulary is the convergence of the p-series ζ(s), settled by the integral test. |
| `formalcalculusPrereqs` | `improper-integrals` | Karlin's theorem reduces the expected vocabulary to an improper integral that evaluates to the Gamma function Γ(1 − 1/s). |
| `formalmlPrereqs` | `shannon-entropy` | Zipf's law is derived as the maximum-entropy distribution at fixed mean log-rank, with optimality from Gibbs' inequality (KL ≥ 0). |
| `formalmlPrereqs` | `bayesian-nonparametrics` | The Pitman–Yor restaurant generalizes the Chinese restaurant process of the Dirichlet process by a discount that turns logarithmic type growth into a power law. |
| `formalmlConnections` | `minimum-description-length` | Heavy-tailed vocabularies set the cost of coding rare words, the trade-off MDL-style subword vocabularies optimize downstream. |
| `formalstatisticsConnections` | `linear-regression` | The classical log–log least-squares fit of Zipf's exponent is a linear regression whose assumptions the rank plot violates; we show its bias. |
| `formalragConnections` | `vector-space-model-tfidf` | The Zipfian head is the stopword list and the long tail is where IDF weights are largest; TF-IDF inherits this topic's distribution. |

## 9. References (Chicago 17th, notes-and-bibliography; every entry has a URL)

1. Karlin, Samuel. "Central Limit Theorems for Certain Infinite Urn Schemes." *Journal of Mathematics
   and Mechanics* (now *Indiana University Mathematics Journal*) 17 (1967): 373–401.
   https://doi.org/10.1512/iumj.1968.17.17020. — paper; Theorem 1.
2. Gnedin, Alexander, Ben Hansen, and Jim Pitman. "Notes on the Occupancy Problem with Infinitely Many
   Boxes: General Asymptotics and Power Laws." *Probability Surveys* 4 (2007): 146–71.
   https://doi.org/10.1214/07-PS092. — paper; modern survey of Karlin's scheme.
3. Pitman, Jim, and Marc Yor. "The Two-Parameter Poisson–Dirichlet Distribution Derived from a Stable
   Subordinator." *Annals of Probability* 25, no. 2 (1997): 855–900. https://doi.org/10.1214/aop/1024404422. — paper.
4. Pitman, Jim. *Combinatorial Stochastic Processes*. Lecture Notes in Mathematics 1875. Berlin:
   Springer, 2006. https://doi.org/10.1007/b11601500. — book; Theorem 3.8, Corollary 3.9, Lemma 3.11.
5. Clauset, Aaron, Cosma Rohilla Shalizi, and M. E. J. Newman. "Power-Law Distributions in Empirical
   Data." *SIAM Review* 51, no. 4 (2009): 661–703. https://doi.org/10.1137/070710111. — paper; Prop 3, OLS critique.
6. Visser, Matt. "Zipf's Law, Power Laws and Maximum Entropy." *New Journal of Physics* 15, no. 4
   (2013): 043021. https://doi.org/10.1088/1367-2630/15/4/043021. — paper; Theorem 3.
7. Wendel, J. G. "Note on the Gamma Function." *American Mathematical Monthly* 55, no. 9 (1948): 563–64.
   https://doi.org/10.2307/2304460. — paper; Lemma B.
8. Zipf, George Kingsley. *Human Behavior and the Principle of Least Effort*. Cambridge, MA:
   Addison-Wesley, 1949. https://archive.org/details/in.ernet.dli.2015.90211. — book.
9. Heaps, H. S. *Information Retrieval: Computational and Theoretical Aspects*. New York: Academic
   Press, 1978. https://archive.org/details/informationretri0000heap. — book.
10. van Leijenhorst, D. C., and Th. P. van der Weide. "A Formal Derivation of Heaps' Law." *Information
    Sciences* 170 (2005): 263–72. https://doi.org/10.1016/j.ins.2004.03.006. — paper.
11. Ferrer i Cancho, Ramon, and Ricard V. Solé. "Two Regimes in the Frequency of Words and the Origins
    of Complex Lexicons: Zipf's Law Revisited." *Journal of Quantitative Linguistics* 8, no. 3 (2001):
    165–73. https://doi.org/10.1076/jqul.8.3.165.4101. — paper; §7 beat 1.
12. Piantadosi, Steven T. "Zipf's Word Frequency Law in Natural Language: A Critical Review and Future
    Directions." *Psychonomic Bulletin & Review* 21 (2014): 1112–30.
    https://doi.org/10.3758/s13423-014-0585-6. — paper; rigorFlag.
13. Baayen, R. Harald. *Word Frequency Distributions*. Dordrecht: Kluwer, 2001.
    https://doi.org/10.1007/978-94-010-0844-0. — book; LNRE background.
14. Goldwater, Sharon, Thomas L. Griffiths, and Mark Johnson. "Producing Power-Law Distributions and
    Damping Word Frequencies with Two-Stage Language Models." *Journal of Machine Learning Research* 12
    (2011): 2335–82. https://jmlr.org/papers/v12/goldwater11a.html. — paper; PY in language.
15. Teh, Yee Whye. "A Hierarchical Bayesian Language Model Based on Pitman–Yor Processes." In
    *Proceedings of COLING/ACL 2006*, 985–92. https://aclanthology.org/P06-1124/. — paper; §8 forward link.

## 10. Phase B notes

- MDX at `src/content/topics/zipf-heaps-vocabulary-statistics.mdx`; theorem numbering on the page may
  renumber (page order: Prop 1, Cor 1, Prop 2, Thm 3 → "Theorem 1" etc.) — keep the brief's proofs.
- `references[].type`: books 4, 8, 9, 13 → `book`; the rest `paper`.
- Graph: flip node `status` to `published`; remove the title from `planned` in `curriculum.ts`.
- `pnpm audit:cross-site` will report the nine outward links as missing reciprocals — expected until launch.

## 11. Phase B record (2026-10-22)

- Page numbering (page order): Thm 1 = max-entropy (brief Thm 3), Thm 2 = Karlin (brief Thm 1), Thm 3 =
  PY exact mean (brief Thm 2), Thm 4 = PY limits; Lemma 1 = unimodal sums, Lemma 2 = Wendel.
- The lab reads `src/components/viz/data/vocabulary-laws-lab.json`; the harness's
  `test_viz_mirror_matches_payload` requires it to equal `viz_payload()` exactly (harness now 27/27).
  Payload windows carry the OLS intercept too (placement only) and `mle.r_xmin = 1548`.
- Site fixes the first topic exposed: `mdx({ gfm: true })` (Astro 6.4 stopped inheriting GFM, so tables
  rendered as raw pipes) and `.prose` list/table styles in `global.css` (preflight stripped them).
