# Brief — `collocations-keyword-extraction`

Topic 26, the second `text-mining` topic. Status: **Phase A complete** (2026-10-26). Harness: 22/22 under `python -W error`,
3 s. Every number in §3 is the harness's, at 6 decimals.

Decisions (Jonathan, 2026-10-26): tests first, keywords second (about 60/40); the finance case study's gold is the
generator's own provenance (template text and names are fixed phrases, two adjacent restaurant draws are a null pair,
planted restaurant homes are sector keywords); Perron–Frobenius for positive stochastic matrices and the ℓ1
contraction are proved on the page. At the scaffold checkpoint: the selection effect is Prop 5 and the case study's
centerpiece; entity words are removed from keyness and TextRank candidates; Claude writes every body.

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Collocations and Keyword Extraction |
| `{{SUBTITLE}}` | A word pair is a 2×2 table, a point in a tetrahedron: PMI, X², and Dunning's G² measure its distance from independence in different ways, only an exact test survives rare pairs, and testing only the pairs you saw makes every rare sighting significant. Then keywords: keyness compares corpora, and TextRank, which compares nothing, ranks by frequency. |
| `{{SLUG}}` | `collocations-keyword-extraction` |
| `{{DOMAIN}}` | `text-mining` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[zipf-heaps-vocabulary-statistics, n-gram-models-smoothing]` (Topic 1 Defs 1–2, the hapax share; Topic 3 Def 1, Prop 2). In-site connection: `distributional-semantics-ppmi-svd` (Def 2 PMI, Prop 3 PPMI under independence — the page opens from it; that page's §… already forward-links here). |
| `{{FORMALML_PREREQS}}` | `random-walks` (Def 4 stationary distribution, Thm 1 π ∝ degree, Thm 3 convergence; its PageRank remark states the teleporting walk without proof), `shannon-entropy` (Def 5 mutual information, Prop 6 MI = 0 iff independent) |
| `{{FORMALRAG_PREREQS}}` | connections `vector-space-model-tfidf` (Def 4 TF-IDF, Thm 1 IDF as self-information: a within-collection weight, contrasted with keyness's two-corpus test, not re-derived), `set-metrics-precision-recall-map-mrr` (Def 1 precision at a cutoff) |
| `{{STATS_CALC_PREREQS}}` | formalStatistics `likelihood-ratio-tests-and-np` (Def 5 generalized likelihood ratio, Def 6 LRT rejection rule, Thm 4 scalar Wilks, Rem 13 vector Wilks stated without proof: Thm 2 proves the 2×2 case), `hypothesis-testing` (Def 12 Pearson goodness of fit, stated), `discrete-distributions` (Thms 15–17 hypergeometric; Ex 6 Fisher's exact test as a one-sided tail), `modes-of-convergence` (Thm 9 continuous mapping, Thm 10 Slutsky), `multiple-testing-and-false-discovery` (Thm 1 Bonferroni, Thm 5 BH); formalCalculus `metric-spaces` (Thm 4 Banach contraction, Ex 14 the Bellman operator: PageRank's discounted form) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/distributional-semantics-ppmi-svd/`, `notebooks/n-gram-models-smoothing/`, `notebooks/finance-corpus/` |

All sibling slugs above are `status: published` on GitHub `main`, and their block numbers match (checked 2026-10-26).
No sibling derives a collocation score, Pearson X²'s limit for a 2×2 table, G² ≈ X², Fisher's test as a conditional
test, keyness, PageRank's existence/uniqueness/rate on directed graphs, or TextRank (grep of every sibling's `main`,
2026-10-26).

**Scope against siblings.** formalStatistics proves scalar Wilks and states the vector case and Pearson's limit;
formalML proves stationarity and convergence for reversible walks and remarks on PageRank. This page proves the 2×2
limits directly (CLT + Slutsky + a Taylor bound), Lemma 1's conditional law, the selection result, and the Google
matrix's Perron–Frobenius conclusions by contraction.

## 1. Section outline (geometric first)

1. **Overview.** Topic 6's Prop 3 ended with shuffled text carrying more PPMI mass than real text, and pointed here
   for "a proper significance test." Association has to be tested, and the test has to be the right one, run on the
   right family.
2. **The 2×2 table as a point.** Def 1, Def 2, Def 3, Thm 1, Prop 1. Lab tab *Table geometry*.
3. **Rare pairs, in closed form.** Prop 2, Prop 3. Lab tab *Rare pairs*.
4. **Tests and their sizes.** Lemma 1, Def 4, Thm 2, Prop 4. Lab tab *Test size* (sizes).
5. **Testing only what you saw.** Prop 5, Rem 1 (many tests at once). Lab tab *Test size* (selection panel).
6. **Keyness.** Def 5, Prop 6, Rem 2 (TF-IDF is a different question), Rem 3 (Rayson and Garside's one-row LL).
7. **TextRank.** Def 6, Def 7, Thm 3, Thm 4, Prop 7. Lab tab *Keywords*.
8. **Finance case study** (`<FinanceCaseStudy>`): (a) the answer key; (b) ranking every bigram; (c) testing them: the
   selection in the corpus; (d) keyness by sector, and the synthetic corpus against EDGAR; (e) TextRank against
   frequency. Lab tab *Finance bigrams* and *Keywords*.
9. **Where this sits** / honest caveats / implementation.

## 2. Statements (numbered per type, in page order; same numbers in the `.py` and notebook)

Notation. A corpus has N bigram positions (adjacent tokens within a sentence, no padding). For an ordered pair
(w₁, w₂): a = c(w₁ w₂), r = c(w₁ ·), c = c(· w₂). E = rc/N. p̂ = table/N ∈ Δ³ with margins p̂_r = (r/N, 1 − r/N) and
p̂_c = (c/N, 1 − c/N). Natural logs throughout.

**Def 1 (The bigram's table).** [[a, b], [c′, d]] with b = r − a, c′ = c − a, d = N − r − c + a: rows "first token is
w₁ or not", columns "second token is w₂ or not".

**Def 2 (Independence surface).** S = {q ∈ Δ³ : q₁₁q₂₂ = q₁₂q₂₁}. *Claim (in the text, proved):* S is exactly the set of
product tables u vᵀ, u, v ∈ Δ¹ (det q = 0 ⟺ rank ≤ 1 ⟺ q = q_r q_cᵀ for a nonnegative table summing to 1). In Δ³
drawn as a regular tetrahedron, S is the doubly ruled saddle spanned by four of the six edges (the skew quadrilateral
p₁₁–p₁₂–p₂₂–p₂₁: u or v at a vertex), ruled by the segments with u fixed and with v fixed (the lab draws it as an 11 × 11
mesh).

**Def 3 (Association scores).** PMI = log(aN/(rc)) (Topic 6 Def 2 on one cell); t = (a − E)/√a (the "t-score": a
z-statistic whose variance is estimated by a: Manning and Schütze 1999, §5.3.1; Church and Hanks 1990, §4, suggest
t-scores); Pearson
X² = Σ (O − E)²/E; Dunning G² = 2 Σ O log(O/E), over the four cells with E_ij the product of the margins and 0 log 0 = 0.

**Thm 1 (The product of the margins is the KL projection; G² = 2N·Î).** For p ∈ Δ³ with margins p_r, p_c and any u, v ∈ Δ¹,
KL(p ‖ u vᵀ) = KL(p ‖ p_r p_cᵀ) + KL(p_r ‖ u) + KL(p_c ‖ v).
Hence p_r p_cᵀ is the unique closest point of S in KL, at distance I(p), the mutual information of the row and column
indicators; and G² = 2N·KL(p̂ ‖ p̂_r p̂_cᵀ) = 2N·Î. Since the multinomial log-likelihood of q is −N[H(p̂) + KL(p̂ ‖ q)], G²
is the generalized likelihood ratio statistic −2 log Λ (formalStatistics `likelihood-ratio-tests-and-np` Def 5) for
independence. G² = 2N × the mutual information is Moore (2004, §3, Fig. 2, Formula 5); the projection identity is proved
here. *Proof:* split log(p_ij/(u_i v_j)) into three logs and sum against p_ij; the last two sums are KL of the
margins (≥ 0, = 0 iff equal: formalML `shannon-entropy` Prop 4/6 via Gibbs). Harness: projection vs a BFGS solve
(10⁻⁶), the identity at 10⁻¹², G² = 2N·KL = 2N·Î at 10⁻⁹, G² and X² vs scipy `chi2_contingency`.

**Prop 1 (X² is Nφ̂²).** All four O − E equal ±(ad − bc′)/N, and Σ 1/E_ij = N³/(r(N−r)c(N−c)), so
X² = N·χ²(p̂ ‖ p̂_r p̂_cᵀ) = N(ad − bc′)²/(r(N−r)c(N−c)) = Nφ̂², with φ̂ the correlation of the two indicators. In particular
X² ≤ N. *Proof:* direct algebra (written out).

**Prop 2 (The hapax pair).** For the table [[1, 0], [0, N − 1]] (two words seen once each, and only together):
PMI = log N, the largest PMI of any table with a ≥ 1; X² = N, the largest X² of any table; t = 1 − 1/N;
G² = 2[log N − (N − 1) log(1 − 1/N)] = 2 log N + 2 − 1/N − O(N⁻²); and the exact one-sided p-value is 1/N.
*Proof:* substitution; aN/(rc) ≤ N/max(r, c) ≤ N since a ≤ min(r, c), with equality iff a = r = c = 1; |φ̂| ≤ 1
(Cauchy–Schwarz); P(A ≥ 1) = rc/N when r = c = 1 (Lemma 1). At the corpus's N = 57,052: PMI = 10.951718,
G² = 23.903419, t = 0.999982, p = 1.7528·10⁻⁵.

**Prop 3 (Perfect pairs: PMI falls as the evidence rises).** For [[k, 0], [0, N − k]], 1 ≤ k < N/3:
PMI = log(N/k) (decreasing), X² = N (constant), t = √k(1 − k/N) (increasing), G² = 2N·H(k/N) with H the binary
entropy in nats (increasing), and −log p = log C(N, k) (increasing; = PMI at k = 1). *Proof:* substitution; for
perfectly dependent indicators I = H(X); derivatives. Manning and Schütze (1999, §5.4) give the PMI half (for perfectly
dependent bigrams, "as they get rarer, their mutual information increases"); Dunning (1993, §6.1, Tables 2–3) shows the
X² ranking's overestimate for "items that occur only a few times." Prop 3 puts the four scores and the exact test in one
family.

**Lemma 1 (Conditioning on the margins).** If the N positions are i.i.d. with independent row and column indicators
(probabilities π, ρ), then given the margins (r, c), a is hypergeometric: P(a | r, c) = C(r, a)C(N − r, c − a)/C(N, c),
whatever π and ρ are. *Proof:* the multinomial probability factors as (N!/(a!b!c′!d!))·π^r(1−π)^{N−r}ρ^c(1−ρ)^{N−c}; the
two margins are independent binomials; divide, and N!/(a!b!c′!d!) = C(N, r)C(r, a)C(N − r, c′). Harness: exact
enumeration of every table at N = 7 against `hypergeom_pmf`, and scipy at random margins.

**Def 4 (Fisher's exact test).** One-sided p(a) = P(A ≥ a | r, c); two-sided: the total probability of tables no more
probable than the observed one. formalStatistics `discrete-distributions` Ex 6 computes the one-sided tail; Lemma 1
is why it is exact for every (π, ρ). It is valid conditionally (P(p ≤ α | r, c) ≤ α) and so unconditionally.

**Thm 2 (The χ²₁ limits, and G² ≈ X²).** Under independence with fixed 0 < π, ρ < 1:
(a) √N φ̂ ⇒ N(0, 1), so X² = Nφ̂² ⇒ χ²₁;
(b) G² − X² = O_p(N^{−1/2}), so G² ⇒ χ²₁.
This is the 2×2 case of the vector Wilks theorem that formalStatistics states without proof (Rem 13).
*Proof:* (a) p̂₁₁ − p̂₁·p̂·₁ = N⁻¹Σ(X_i − π)(Y_i − ρ) − (X̄ − π)(Ȳ − ρ); the first term is a mean of i.i.d. mean-zero
variables with variance π(1−π)ρ(1−ρ) (independence), so √N times it ⇒ N(0, π(1−π)ρ(1−ρ)) by the CLT; √N times the second
is O_p(N^{−1/2}); the denominator converges to √(π(1−π)ρ(1−ρ)) a.s.; Slutsky (fS `modes-of-convergence` Thm 10) and
continuous mapping (Thm 9). (b) With f(x) = x log(x/q) − (x − q), f(q) = f′(q) = 0 and f″ = 1/x, so
KL(p̂ ‖ q̂) = Σ (p̂ − q̂)²/(2ξ) with ξ between p̂ and q̂, and |2N·KL − N·χ²| ≤ N Σ|p̂ − q̂|³/(ξq̂); every |p̂_ij − q̂_ij| =
|φ̂|·√(…) = O_p(N^{−1/2}) by (a), and ξ, q̂ → q > 0. Harness: KS < 0.015 against N(0, 1) and χ²₁ at N = 10⁴ (20,000
draws), and median |G² − X²| falls by ≈ √10 per decade (0.001110 → 0.000362 → 0.000113 at N = 10³, 10⁴, 10⁵).

**Prop 4 (Exact sizes).** Given the margins, any test's size under independence is Σ_{a ∈ R} P(a | r, c) over its
rejection region (Lemma 1), so it can be computed exactly; Fisher's never exceeds α. *Proof:* Lemma 1;
super-uniformity of a discrete p-value. *Computed (N = 10⁴, r = c, α = 0.05):* X²'s size peaks at 0.162334 near
E = 0.176400, G²'s at 0.157057 near E = 1.904400, and Fisher's maximum is 0.049983. Below E ≈ 0.05, all three equal
P(A ≥ 1): Prop 5's regime. Dunning (1993, §4, Table 1) measures the normal approximation's error on binomial tails
(a factor of 40 at np = 0.1) and §5.3 notes that it reduces to Pearson's X²; Moore (2004, §6, Table 1) finds G²-based
p-values understate noise, i.e. overstate significance, for rare events, and recommends Fisher's test (§7). The exact test
is Fisher (1935, Example 1, pp. 48–50), which conditions on the margins as Lemma 1 does.

**Prop 5 (Testing only what you saw).** For any margins and the one-sided exact test at level α:
(a) P(reject) ≤ α; (b) if P(A ≥ 1) > α, P(reject | A ≥ 1) ≤ α / P(A ≥ 1); (c) if P(A ≥ 1) ≤ α, then every a ≥ 1 is
rejected: P(reject | A ≥ 1) = 1 while P(reject) = P(A ≥ 1) ≤ α. (d) Hence, over a family of null pairs, the number of
false rejections among the pairs that were *seen* is at least the number seen with P(seen) ≤ α, and a multiple-testing
correction computed over the observed pairs alone (m = #observed) controls nothing; computed over all m candidate pairs
(unseen pairs have p = 1) it controls what it promises. *Proof:* p(1) = P(A ≥ 1) and p is decreasing in a; Bayes.
Attribution: Moore (2004, §2.1, "When is Something Seen Only Once Significant?") makes the family-wide point for
singletons: a singleton–singleton co-occurrence has exact p = 2·10⁻⁶, but among 391,236,048 possible singleton pairs
782.5 are expected by chance; his §2.2 estimates noise by summing the expected count over all pairs with the same
marginals. Dunning (1993, §6.1) scores only the bigrams that appear and uses the statistic "as a measure." Prop 5 states
the conditional fact exactly and the finance stratum measures it. Harness: 200 random margin sets; the selection curve at the corpus stratum's N = 4,182 (P(reject | seen) = 1 exactly for
r ≤ 14, i.e. E ≤ 0.047; 0.042798 at r = 20).

**Rem 1 (Many tests at once).** A vocabulary of V types has V² candidate pairs. Bonferroni (fS Thm 1) holds under any
dependence; BH (fS Thm 5) assumes independence or positive dependence, which pairs sharing a word need not satisfy (its
arbitrary-dependence variant is fS Thm 6). Over all candidate pairs Bonferroni rejects nothing in the finance stratum
either, so the conclusion does not rest on BH's assumption. Frequency thresholds are the field's practical response:
Church and Hanks (1990, §4) set aside pairs with f(x, y) ≤ 5 ("the simple but arbitrary threshold"), and Evert (2004,
§4.3.3) excludes f < 3 and finds f ≥ 5 safe. A threshold fixed before looking shrinks the family to pairs whose P(seen)
is not negligible. Topic 28 takes up significance testing for NLP systems.

**Def 5 (Keyness).** For a target corpus T and a reference R (sizes N_T, N_R) and a word w with counts a in T and b in
R: the 2×2 table (w or not) × (T or R). Signed G² (positive when w is over-used in T) and the log₂ ratio
log₂[((a + ½)/N_T)/((b + ½)/N_R)] as an effect size.

**Prop 6 (Keyness is the same test).** Keyness G² = 2(N_T + N_R)·Î(word indicator; corpus indicator), the statistic of
Thm 1 with the corpus as the second variable, and the transposed table gives the same value. *Proof:* Thm 1; Î is
symmetric (fML `shannon-entropy` Prop 5).

**Rem 2 (TF-IDF answers another question).** TF-IDF (formalRAG `vector-space-model-tfidf` Def 4) weighs a term within
one collection by its document frequency there (Thm 1: IDF is the self-information of presence); keyness compares two
corpora and is a test. Not re-derived here.

**Rem 3 (Rayson and Garside's LL is G² over one row).** Rayson and Garside (2000, §2, Table 1) sum
2 Σ O log(O/E) over the word's row only. The omitted row (the other words) contributes a nonnegative term, at most
2(a − E₁)²[1/(N_T − E₁) + 1/(N_R − E₂)] by KL ≤ χ², so their LL ≤ G². They compare no χ² critical value: they sort by LL.
Harness: the bound in every sector, the largest relative gap below 0.07, and identical top-25 lists.

**Def 6 (Co-occurrence graph).** Remove a stop set; join two distinct words within W positions of each other in the
filtered sentence; weight = number of such co-occurrences (or 1 for the binary graph). Nodes are words with at least one
edge.

**Def 7 (PageRank).** P̃ = D⁻¹W with dangling rows replaced by the uniform row; the Google matrix
G = (1 − d)/n·𝟏𝟏ᵀ + d·P̃, 0 ≤ d < 1; PageRank is a row vector π ∈ Δ^{n−1} with πG = π. TextRank is PageRank on Def 6's
graph (Mihalcea and Tarau 2004, §2: they write S(V_i) = (1 − d) + d Σ S(V_j)/|Out(V_j)|, unnormalized, which is n·π
when no row dangles; d = 0.85; for keywords an undirected unweighted graph over nouns and adjectives, window 2 best
(§3.1–3.2, Table 1), convergence threshold 10⁻⁴).

**Thm 3 (Perron–Frobenius for positive stochastic matrices).** Let M be row-stochastic with every entry ≥ δ > 0
(the Google matrix has δ = (1 − d)/n, and then 1 − nδ = d). Then: (a) M has exactly one stationary distribution π;
(b) π_i ≥ δ for every i; (c) every eigenvalue λ ≠ 1 of M has |λ| ≤ 1 − nδ; (d) for the Google matrix,
π = ((1 − d)/n)·𝟏ᵀ(I − dP̃)⁻¹ = ((1 − d)/n)·𝟏ᵀ Σ_t dᵗP̃ᵗ.
*Proof:* write M = nδ·U + (1 − nδ)·Q with U the uniform matrix and Q stochastic. For a row vector y with y𝟏 = 0,
yM = (1 − nδ)·yQ and ‖yQ‖₁ ≤ ‖y‖₁ (Lemma: stochastic matrices do not expand ℓ1, for complex y too). So x ↦ xM is a
contraction of the simplex in ℓ1 with factor 1 − nδ; the simplex is complete; Banach (formalCalculus `metric-spaces`
Thm 4) gives (a). (b): π = πM ≥ δ·(π𝟏)·𝟏ᵀ. (c): if yM = λy with λ ≠ 1 then y𝟏 = 0 (since M𝟏 = 𝟏), so
|λ|‖y‖₁ = (1 − nδ)‖yQ‖₁ ≤ (1 − nδ)‖y‖₁. (d): ρ(dP̃) ≤ d < 1, so I − dP̃ is invertible and the Neumann series converges.
Haveliwala and Kamvar (2003) prove (c) for the Google matrix (with equality conditions). Harness: power iteration vs the
dense eigenvector (10⁻¹²) on random directed graphs with dangling rows, π_i ≥ (1 − d)/n, |λ₂| ≤ d.

**Thm 4 (Power iteration converges geometrically).** x_{t+1} = x_tG from any x₀ ∈ Δ satisfies
‖x_t − π‖₁ ≤ dᵗ‖x₀ − π‖₁ ≤ 2dᵗ and ‖x_{t+1} − x_t‖₁ ≤ dᵗ‖x₁ − x₀‖₁; stopping when a step falls below ε leaves an error at
most εd/(1 − d), after at most ⌈log(ε/‖x₁ − x₀‖₁)/log d⌉ + 1 steps. *Proof:* Thm 3's contraction; a geometric series.
This is the Bellman operator's argument (fC `metric-spaces` Ex 14) with d in place of the discount. Harness: both
inequalities at every step on random graphs; finance iteration counts within the bound.

**Prop 7 (Strength, and the d → 1 limit).** (a) On the weighted window-2 graph, a word's strength is
s(w) = 2c(w) − e(w) − 2σ(w), with c its count in the filtered stream, e its occurrences at a sentence's first or last
filtered position (a one-word sentence counts twice), and σ its adjacent self-pairs. (b) On an undirected graph without
isolated nodes and components C₁, …, C_m, every d gives each component exactly the mass |C_j|/n, and
lim_{d→1} π_d = Σ_j (|C_j|/n)·s|_{C_j}/vol(C_j). So as d → 1 TextRank ranks each component's words by strength, which
by (a) is twice the frequency up to boundary terms. (c) If the teleport distribution is s/vol instead of uniform
(a connected graph), π_d = s/vol for every d. *Proof of (c):* s/vol is stationary for P̃ and equals the teleport term,
so it is the fixed point. Grolmusz (2015) proves this personalization case and notes that with uniform teleportation
PageRank is not exactly proportional to degree; Boudin (2013, §5, Table 2: F@10 32.2 for both on Inspec) finds degree
centrality as good as TextRank, citing Grolmusz more broadly than his statement. *Proof of (a)–(b):* (a) count each occurrence's existing neighbors. (b) P̃
keeps each component closed, so π_d(C_j) = (1 − d)|C_j|/n + d·π_d(C_j); any limit point is stationary on each
component; an irreducible finite chain has one stationary distribution (its lazy version is ergodic: fML `random-walks`
Thm 3), which is s/vol (Thm 1). Harness: the identity holds exactly on the finance text (empty mismatch, with and without
the stop list); the limits at d = 1 − 10⁻⁷ to 10⁻⁶ on connected and two-component graphs; (c) at d ∈ {0.5, 0.85}
to 10⁻¹³, against a uniform-teleport gap above 10⁻³.

## 3. Findings (harness, 6 decimals)

**(a) The answer key.** `_TaggedGen` wraps every restaurant draw in sentinels and makes the same random calls, so the
stripped text equals `generate_corpus(300, 0)` byte for byte and the sentences equal `ng.corpus()["train"]`. N = 57,052
bigram positions, 13,724 types: 877 *fixed* (only ever template–template or name–name), 3,054 *null* (only ever two
adjacent draws, at least one from the shared vocabulary: independent given the sector), 1,000 *topical* (two adjacent
draws planted in one sector: associated through the sector), 8,793 *boundary*. `fc._tokens` drops commas, so list
neighbors ("freight, contracts") are adjacent draws too; both kinds are independent draws.

**(b) Ranking every bigram** (precision@k against *fixed*, ties broken at random in expectation):

| measure | top tie block | P@10 | P@50 | P@200 | P@500 | null + topical in top 50 |
|---|---|---|---|---|---|---|
| PMI | 97 | 0.051546 | 0.051546 | 0.057394 | 0.086865 | 0.948454 |
| t | 1 | 0.800000 | 0.900000 | 0.910000 | 0.822000 | 0 |
| X² | 196 | 0.530612 | 0.530612 | 0.540000 | 0.398000 | 0.469388 |
| G² | 1 | 0.800000 | 0.940000 | 0.930000 | 0.898000 | 0 |
| Fisher | 1 | 0.800000 | 0.940000 | 0.930000 | 0.896000 | 0 |
| frequency | 1 | 0.700000 | 0.860000 | 0.880000 | 0.780000 | 0 |

PMI's top 97 tie at log N (Prop 2's hapax pairs: 44 topical, 48 null, 5 fixed);
X²'s top 196 tie at N (every perfect pair, Prop 3). G², t, and Fisher put no null or topical pair in their top 500.
G²'s top two ("percent in", "for the") are *boundary*: template words next to an entity's closed-class word; they are
fixed text in fact, and the class is conservative. Fisher needs log-space tails: `sf` underflows to 0 for 46 pairs.

**(c) Testing them: the selection.** Inside the draw–draw stratum (N = 4,182 positions, 4,059 types, 2,128,632 candidate
pairs), 0.975117 of pairs are seen once. BH at q = 0.05 over the *observed* pairs rejects 2,218 of 3,055 null pairs
(0.726) and 863 of 1,000 topical; over *all* candidate pairs it rejects none of either (Bonferroni the same). At nominal
0.05 per pair, 0.762357 of observed null pairs reject. The pooled table repeats it: BH over the observed bigrams rejects
2,416 of 3,054 null pairs. The topical pairs are truly associated, but at one sighting each no valid procedure can find
them: the honest answer is "not enough data."

**(d) Keyness by sector** (entity words removed; P@25 against planted sector nouns):

| sector | docs | G² | log ratio | frequency |
|---|---|---|---|---|
| industrials | 142 | 1.00 | 1.00 | 0.08 |
| technology | 52 | 1.00 | 1.00 | 0.08 |
| banking | 40 | 0.96 | 1.00 | 0.08 |
| energy | 20 | 0.80 | 0.80 | 0.12 |
| retail | 27 | 1.00 | 1.00 | 0.08 |
| healthcare | 19 | 1.00 | 1.00 | 0.12 |

Energy's misses are its own template: "Crude production averaged … thousand barrels per day" occurs only in energy
filings, so "barrels", "averaged", "day", "crude" are key and correct, but not planted nouns. Synthetic corpus vs the EDGAR
excerpts: top words "percent, in, management, across, demand, was, million, halvorsen"; 0.76 of the top 50 are template-only
words. A reference corpus finds what makes the target unusual, and here that is its templates.

**(e) TextRank** (window 2, stop set = 100 most frequent words ∪ entity words, d = 0.85, P@25):

| sector | TextRank weighted | Spearman with frequency | binary | Spearman | frequency | keyness G² | iterations to 10⁻¹² |
|---|---|---|---|---|---|---|---|
| industrials | 0.40 | 0.849069 | 0.44 | 0.828389 | 0.24 | 1.00 | 142 |
| technology | 0.32 | 0.849610 | 0.36 | 0.803704 | 0.24 | 1.00 | 145 |
| banking | 0.40 | 0.836944 | 0.44 | 0.787838 | 0.36 | 0.96 | 147 |
| energy | 0.52 | 0.850071 | 0.52 | 0.742767 | 0.36 | 0.80 | 150 |
| retail | 0.32 | 0.832100 | 0.32 | 0.777393 | 0.16 | 1.00 | 148 |
| healthcare | 0.40 | 0.804751 | 0.40 | 0.734937 | 0.36 | 1.00 | 148 |

TextRank's rank correlation with frequency is 0.80–0.85 (weighted), and it beats raw frequency only modestly; keyness,
which has a reference corpus, beats both everywhere. Spearman with frequency rises with d (0.79 at 0.5 → 0.88 at 0.99 in
industrials), as Prop 7 predicts. Thm 4's bound ⌈log(10⁻¹²/2)/log 0.85⌉ + 1 = 176 steps; observed 142–150.

## 4. The lab: `CollocationLab` (5 tabs; every number from the payload, `src/components/viz/data/collocation-lab.json`)

1. **Table geometry.** A rotatable 2-D projection of the tetrahedron Δ³ with the independence saddle (11 × 11 mesh) and
   one margin preset's segment of tables (presets (N, r, c) = (100, 40, 50), (1000, 20, 30), (1000, 5, 5)); a slider
   on a moves the point; readouts PMI, t, X², G², exact p, with the crossing at a = E marked. *Learn:* the four scores
   are four distances from the same surface.
2. **Rare pairs.** k = 1…20 perfect pairs at N = 57,052: PMI falls, X² is flat at N, G² and −log p rise; the hapax
   readout. *Learn:* Dunning's inversion.
3. **Test size.** Exact size vs E (log axis, r = 1…450 at N = 10⁴) for X², G², Fisher, with α marked; second panel
   P(seen), P(reject), P(reject | seen) vs E at N = 4,182, and the corpus counts (observed vs all-pairs BH). *Learn:*
   each asymptotic test fails in its own regime; below E ≈ α everything "significant" was merely seen.
4. **Finance bigrams.** Measure pills; precision@k (k = 1…500) and null + topical share curves; the top 15 with class
   colors and counts. *Learn:* PMI's top is noise seen once; G² and Fisher rank fixed phrases.
5. **Keywords.** Sector pills; keyness top 10; TextRank precision and Spearman with frequency across d ∈ {0.5, 0.85, 0.95,
   0.99}, weighted vs binary; power-iteration step sizes (every 4th of the first 400) against dᵗ on industrials. *Learn:*
   TextRank without a reference corpus tracks frequency; keyness finds the sector.

Mobile: every readout row flex-wraps; the tetrahedron scales to the container width.

## 5. Checkpoint answers (Jonathan, 2026-10-26)

- Balance 60/40, tests first. Gold = the generator's provenance. Prove PF + contraction.
- Selection effect promoted to Prop 5 and the case study's centerpiece.
- Entity words excluded from keyness and TextRank candidates (names win keyness trivially).
- Claude writes every body.

## 5b. rigorFlag (draft)

The answer key is synthetic: templates make fixed phrases unusually easy to find (G² precision@500 0.898), and real
collocations are graded, not a yes/no class. The t-score is not a t test (its variance is estimated by the count). X²'s and
G²'s χ² p-values are asymptotic, and their exact sizes at small expected counts are computed here, not bounded in general.
Fisher's test is exact but conservative (discreteness). The "null" pairs are independent only up to the Pitman–Yor
restaurants' reinforcement and the generator's rule against repeating a noun within a list. Benjamini–Hochberg's guarantee
assumes independence or positive dependence among p-values, which pairs sharing a word need not have; Bonferroni gives the
same counts here. TextRank is run with a frequency stop list instead of Mihalcea and Tarau's part-of-speech filter (the
corpus has no tagger), on single words (they merge adjacent selected words into phrases), and keywords are scored against
planted sector nouns, not human-assigned keyphrases (their Inspec evaluation, Hulth 2003).

## 6. Notebook cell map (20 cells, `01_collocations_keyword_extraction.ipynb`)

1–2 setup and the class counts; 3–4 Thm 1 / Prop 1 numerically; 5–6 Prop 2–3; 7–8 Thm 2 and Prop 4; 9–12 Prop 5 and the
corpus selection; 13–14 rankings; 15–16 keyness and EDGAR; 17–18 Thm 3–4, Prop 7, TextRank; 19–20 the harness.

## 7. References (verified 2026-10-26 from the PDFs or official records; Chicago notes-bibliography)

- Benjamini, Yoav, and Yosef Hochberg. 1995. "Controlling the False Discovery Rate: A Practical and Powerful Approach to
  Multiple Testing." *Journal of the Royal Statistical Society: Series B (Methodological)* 57 (1): 289–300.
  https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Boudin, Florian. 2013. "A Comparison of Centrality Measures for Graph-Based Keyphrase Extraction." In *Proceedings of the
  Sixth International Joint Conference on Natural Language Processing*, 834–38. Nagoya. https://aclanthology.org/I13-1102/
  (not I13-1062, which is TopicRank.)
- Church, Kenneth Ward, and Patrick Hanks. 1990. "Word Association Norms, Mutual Information, and Lexicography."
  *Computational Linguistics* 16 (1): 22–29. https://aclanthology.org/J90-1003/ (§4: ordered pairs in a window of 5; the
  f ≤ 5 threshold; t-scores suggested.)
- Dunning, Ted. 1993. "Accurate Methods for the Statistics of Surprise and Coincidence." *Computational Linguistics* 19 (1):
  61–74. https://aclanthology.org/J93-1003/ (equations unnumbered: §5.2 λ and its χ² limit, without citing Wilks; §5.3 the
  binomial and multinomial forms; §6.1 the 31,777-word bigram study, Tables 2–4; §7 suggests "Fischer's exact method".)
- Evert, Stefan. 2004. "The Statistics of Word Cooccurrences: Word Pairs and Collocations." PhD diss., Universität
  Stuttgart (published 2005). https://elib.uni-stuttgart.de/handle/11682/2573 (URN urn:nbn:de:bsz:93-opus-23714; §2.2 the
  sampling models, §2.2.4 conditioning on the margins, §4.3.3 the frequency threshold. The elib page is behind bot
  protection; the handle comes from the author's page.)
- Fisher, R. A. 1922. "On the Interpretation of χ² from Contingency Tables, and the Calculation of P." *Journal of the Royal
  Statistical Society* 85 (1): 87–94. https://doi.org/10.2307/2340521 (degrees of freedom; not the exact test.)
- Fisher, R. A. 1935. "The Logic of Inductive Inference." *Journal of the Royal Statistical Society* 98 (1): 39–82.
  https://doi.org/10.2307/2342435 (Example 1, pp. 48–50: the exact test, conditioning on the margins.)
- Grolmusz, Vince. 2015. "A Note on the PageRank of Undirected Graphs." *Information Processing Letters* 115: 633–34.
  https://doi.org/10.1016/j.ipl.2015.02.015 (arXiv 1205.1960.)
- Haveliwala, Taher H., and Sepandar D. Kamvar. 2003. "The Second Eigenvalue of the Google Matrix." Technical report,
  Stanford InfoLab. http://ilpubs.stanford.edu:8090/582/ (Thm 1: |λ₂| ≤ c; Thm 2: = c with two irreducible closed subsets.
  The live server was down; read via the Wayback Machine.)
- Hulth, Anette. 2003. "Improved Automatic Keyword Extraction Given More Linguistic Knowledge." In *Proceedings of the 2003
  Conference on Empirical Methods in Natural Language Processing*, 216–23. https://aclanthology.org/W03-1028/
- Manning, Christopher D., and Hinrich Schütze. 1999. *Foundations of Statistical Natural Language Processing.* Cambridge,
  MA: MIT Press. Chap. 5, "Collocations," 141–77. https://nlp.stanford.edu/fsnlp/ (no DOI found; §5.3.1 t test, §5.3.3
  Pearson, §5.3.4 likelihood ratios, §5.4 PMI.)
- Mihalcea, Rada, and Paul Tarau. 2004. "TextRank: Bringing Order into Texts." In *Proceedings of the 2004 Conference on
  Empirical Methods in Natural Language Processing*, 404–11. Barcelona. https://aclanthology.org/W04-3252/ (the Anthology
  record says "Text"; the PDF says "Texts".)
- Moore, Robert C. 2004. "On Log-Likelihood-Ratios and the Significance of Rare Events." In *Proceedings of the 2004
  Conference on Empirical Methods in Natural Language Processing*, 333–40. Barcelona. https://aclanthology.org/W04-3243/
  (bilingual Hansard word association, not monolingual bigrams.)
- Page, Lawrence, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999. "The PageRank Citation Ranking: Bringing Order to
  the Web." Technical report, Stanford InfoLab. http://ilpubs.stanford.edu:8090/422/ (Def 1, Eq. 1; §2.5 random surfer;
  §6 ‖E‖₁ = 0.15. Read via the Wayback Machine.)
- Pearson, Karl. 1900. "On the Criterion That a Given System of Deviations from the Probable in the Case of a Correlated
  System of Variables Is Such That It Can Be Reasonably Supposed to Have Arisen from Random Sampling." *Philosophical
  Magazine*, 5th ser., 50 (302): 157–75. https://doi.org/10.1080/14786440009463897
- Rayson, Paul, and Roger Garside. 2000. "Comparing Corpora Using Frequency Profiling." In *The Workshop on Comparing
  Corpora*, 1–6. Hong Kong: ACL. https://doi.org/10.3115/1117729.1117730 (W00-0901.)
- Wilks, S. S. 1938. "The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses." *Annals of
  Mathematical Statistics* 9 (1): 60–62. https://doi.org/10.1214/aoms/1177732360

Not cited (unverified this session): Church et al. (1991) for the t-score, Brin and Page (1998) as the source of
d = 0.85, and Fisher's *Statistical Methods for Research Workers*, 5th ed., §21.02.
