# Brief — `knowledge-graph-embeddings`

**Knowledge-Graph Embeddings: TransE and RotatE.** This is roadmap #24 (domain `extraction-knowledge-graphs`). Both
prerequisites are published: `relation-extraction-distant-supervision` (#23), whose `extracted_kb()` this topic
completes, and `glove-embedding-geometry` (#8), whose finance case study found that "no offset is constant" in
co-occurrence vectors. It feeds `capstone-earnings-call-nlp` (#29).

**Status: Phase A done and approved by Jonathan (2026-10-26) with the recommended answers to §5: lead the case study
with finding 2; keep inheritance; trim runtime costs that move no number; proceed to Phase B.**
- **Phase B** shipped the MDX, `KnowledgeGraphEmbeddingLab` (5 tabs), and the graph and curriculum flip.
  - The runtime trim bought nothing: `evaluate` now batches its queries (payload unchanged, asserted by
    `test_viz_mirror_matches_payload`), but the time is ~30 s of training over 108 runs plus Topic 23's 25 s
    extraction, and neither shrinks without moving reported numbers. The harness runs 59.9–62.7 s, at or just over the 60 s rule.
  - Building the lab found that TransE's best plane run on S₃ collapses all six entities to one point (every distance
    is ‖w‖ = 2, ratio exactly 1). The page says so, and the harness asserts it.
  - Phase B corrections: Northgate Energy has one subsidiary (Halvorsen Industrial is the only parent with two, now the
    filtered-rank example); `tests/scaffold.test.ts` bans the word "architectures" (rephrased).
  - Gates: `pnpm test` 39/39, `astro sync`, `validate`, `audit:cross-site` (formalNLP 186 edges, 0 wrong direction,
    0 slug drift; all missing reciprocals deferred), `build`. Page: 0 `.katex-error`, 417 KaTeX spans, no `‘`. Every
    lab readout in 25 tab/toggle states equals the payload (checked by a script generated from the JSON), with both
    sliders driven by real key presses at both ends; at the mobile preset `innerWidth` and `scrollWidth` are 375 in all
    25 states. No console errors.
- `notebooks/knowledge-graph-embeddings/knowledge_graph_embeddings.py`: 16 tests, 59.9–62.7 s under `-W error` across runs (Topic 23's
  `extracted_kb()`, Topic 22's tagger and Topic 12's pretraining inside it, is 25 s of it). No waiver needed, barely (§5, question 5).
- `01_knowledge_graph_embeddings.ipynb` (22 cells) exits 0 under `jupyter nbconvert --execute`.
- Lab data `src/components/viz/data/knowledge-graph-embedding-lab.json` (46.4 kB) is written by `--write-payload` and
  pinned by `test_viz_mirror_matches_payload`.

**Phase A decisions** (plan mode, 2026-10-26, with Jonathan):
- **Scope:** the TransE/RotatE core plus a bilinear foil (DistMult, ComplEx) and the group-action view. Not chosen:
  evaluation-protocol theorems and a self-adversarial-sampling theorem (filtered rank and ties stay a definition).
- **Case study:** the gold KB's pattern tests, then Topic 23's extracted facts about the held-out companies, completed
  beside the training roster's facts.
- **`intermediate`, Claude writes all code.**
- **No sibling covers** TransE, RotatE, DistMult, ComplEx, link prediction, Hits@k, or filtered ranking (grepped all four
  siblings' `main`, 2026-10-26); formalRAG `set-metrics-precision-recall-map-mrr` Def 3 is MRR. No sibling states
  Euler's formula as a result.
- **Topic 8 is not imported** (its module pulls in Topic 6's pipeline); the page quotes its published finding in prose.

**What building the harness found** (each now an assert):
1. **The trained phases solve Thm 1's equations.** RotatE on the full KB puts COMPETES_WITH's 16 phases at {0, π}
   (median distance 0.000000 against π/4 for random phases), θ_PARENT + θ_SUB at 0 (median 0.000112), and
   θ_BASED − θ_HQ − θ_LOC at 0 (0.022656; random 1.570796). ACQUIRED ⇒ PARENT_OF shows as θ_ACQ ≈ θ_PARENT (0.056451).
2. **Expressivity cuts both ways under the closed-world assumption.** On split 0, each hidden fact is a *negative*
   during training. With phases at {0, π} a rotation scores (a, r, b) exactly as (b, r, a), so to push the hidden
   direction down RotatE moves its symmetric phases off the solutions (median 0.484901) and its inverse phases too
   (0.678214). It learns the false antisymmetry the incomplete KB teaches: held-out symmetry MRR 0.271977, the worst of
   the four. DistMult cannot represent antisymmetry (Prop 3), so it cannot learn it: 1.000000 on all three seeds.
3. **Each model wins the pattern its geometry favors, and only that one** (filtered MRR, split 0, 3 seeds):

   | | symmetry | inverse | composition | inheritance | direction |
   |---|---|---|---|---|---|
   | TransE | 0.459 | 0.694 | 0.352 | **0.815** | 0.997 |
   | RotatE | 0.272 | 0.676 | **0.912** | 0.286 | 0.992 |
   | DistMult | **1.000** | 0.574 | 0.807 | 0.022 | 0.500 |
   | ComplEx | 0.502 | **0.780** | 0.694 | 0.068 | 0.989 |

   TransE models composition exactly (Prop 1) yet learns it worst (0.352): exact representability is not what training
   finds. TransE's inheritance win is Thm 2(a)'s collapse at work: an N-to-1 relation (IN_SECTOR) pulls a sector's
   companies together, and a subsidiary lands near its parent. DistMult's direction accuracy is exactly 1/2 (Prop 3).
4. **Ties are real.** DistMult's direction accuracy first read 0.514: f(h, r, t) and f(t, r, h) are equal but computed in
   different orders, so they differ in the last bits. Def 2's tie tolerance (10⁻⁹ relative) makes it exactly 0.5.
5. **The competitor graph has odd cycles** (8 triangles; degrees 1 × 6, 2 × 10, 3 × 8 companies), so Thm 2(c) binds:
   an exact rotation would collapse every triangle. COMPETES_WITH is N-to-N (2.08 competitors per company), and "RotatE
   models symmetry" holds exactly only for matchings.
6. **No abelian model separates S₃'s corruptions** (Thm 3): the best trained ratios γ_min/ε are 1.000 (TransE, k = 1, by
   collapsing all six entities to one point),
   0.899 (k = 2), 0.907 (k = 8), 0.519, 0.995, 2.041 (RotatE), all below 5. The dihedral construction fits all 12 facts
   to 4.5 · 10⁻¹⁶ with entities 0.6 apart.
7. **Completion recovers the inverse, not the competitors.** At τ = 0.5 extraction gives 45 facts, all true, and misses
   28 WORLD facts about the held-out entities: 26 COMPETES_WITH with training-roster companies and 2 SUBSIDIARY_OF whose
   PARENT_OF was extracted. RotatE, DistMult, and ComplEx rank both SUBSIDIARY_OF facts first on every seed (TransE MRR
   0.556). Competitors are found only by sector proximity: Hits@10 0.833 (TransE), 0.808 (RotatE), 0.590 (DistMult),
   0.679 (ComplEx), MRR ≤ 0.332. At τ = 0.1 the KB gains 19 facts, 17 false (mostly reversed officer facts and swapped
   HQ cities and regions): TransE's Hits@10 falls most (0.833 → 0.641), RotatE's to 0.744, ComplEx's to 0.603, and
   DistMult's rises to 0.641 (52 queries, 3 seeds: small moves are noise).
8. **Validation is noisy at this scale.** Tuning on one seed picked RotatE γ = 3 at 500 steps (composition 0.731);
   averaging three seeds picks γ = 6 at 1,500 (0.912). Every pick is the seed-averaged argmax on split 1, asserted.
   Early stopping matters for the reason in finding 2: the longer the fit, the harder hidden facts are pushed down.
   Adding a 250-step checkpoint moved TransE (flat: 0.563 vs 0.557 at 500) and ComplEx (0.555 vs 0.500), and with them
   TransE's inheritance MRR (0.942 → 0.815) and ComplEx's symmetry (0.380 → 0.502); no qualitative claim changed.

---

## 0. Placeholder table

| Placeholder | Value |
|---|---|
| `{{TITLE}}` | Knowledge-Graph Embeddings: TransE and RotatE |
| `{{SUBTITLE}}` | Relations as translations or rotations of entity space: which patterns each geometry can satisfy exactly, what both abelian groups cannot, and what training on an incomplete KB actually learns. |
| `{{SLUG}}` | `knowledge-graph-embeddings` |
| `{{DOMAIN}}` | `extraction-knowledge-graphs` |
| `{{DIFFICULTY}}` | `intermediate` |
| `{{FINANCE_CASE_STUDY}}` | `true` |
| `{{PREREQUISITES}}` | `[relation-extraction-distant-supervision, glove-embedding-geometry]` |
| `{{FORMALML_PREREQS}}` | `tensor-decompositions` (Def 4, CP decomposition: DistMult is a CP model with shared entity factors) |
| `{{FORMALRAG_PREREQS}}` | `set-metrics-precision-recall-map-mrr` (Def 3, reciprocal rank and MRR); connections `negative-sampling-hard-negatives` (Prop 2: false negatives, the training-side twin of finding 2), `cross-encoders-reranking` (Prop 1, a bilinear scorer's rank bound), `embedding-dimension-lower-bounds` (Thm 1, sign-rank: ComplEx's expressiveness is a sign-rank statement), `graphrag-community-detection` (its entity graph is untyped; this one is typed and embedded) |
| `{{STATS_CALC_PREREQS}}` | formalCalculus `eigenvalues-eigenvectors` (Rem 2: a complex eigenvalue a + bi acts as rotation and scaling) and `linear-systems` (Ex 5 and Rem 3: rotation matrices through complex eigenvalues) |
| `{{RIGOR_FLAG}}` | §5b |
| `{{REFERENCE_NOTEBOOKS}}` | `notebooks/relation-extraction-distant-supervision/`, `notebooks/finance-corpus/` |

All eight sibling slugs are `status: published` on GitHub `main` and their block numbers match (checked 2026-10-26).
formalML `smooth-manifolds` has no numbered block on matrix Lie groups (a prose paragraph in "A Gallery of Smooth
Manifolds"), so it is not cited.

## 1. Section outline (geometric first)

1. **Overview.** A KB is a labeled directed graph; an embedding turns "is (h, r, t) true?" into geometry. The lab query:
   (Granite Basin Midstream, SUBSIDIARY_OF, Northgate Energy), a held-out fact extraction scored 0.401 (below 0.5),
   whose PARENT_OF inverse it did extract; RotatE ranks it first.
2. **Link prediction.** Def 1, Def 2.
3. **Translation: TransE.** From GloVe's parallelogram (Topic 8's Def 4 and its "no constant offset" finding) to offsets
   by design. Def 3, Prop 1, Prop 2. Lab tab *Translate or rotate*.
4. **Rotation: RotatE.** Lemma 1, Def 4, Thm 1. Lab tab *Relation phases*.
5. **Relations as group actions.** Def 5, Thm 2, Thm 3, Cor 1, then the pattern × model table. Lab tab
   *Non-commuting*.
6. **The bilinear foil.** Def 6, Props 3–5.
7. **Finance case study** (`<FinanceCaseStudy>`): (a) patterns held out (finding 2 leads); (b) completing what
   extraction found. Lab tabs *Link prediction* and *Completing extraction*.
8. **Where this sits** (hands the completed KB to the capstone) / honest caveats / implementation.

## 2. Statements (numbered per type, in page order; same numbers in the `.py` and notebook)

**Def 1 (knowledge graph, scoring, link prediction).** Entities 𝓔, relations 𝓡, a KG 𝒦 ⊆ 𝓔 × 𝓡 × 𝓔. A scoring function
f : 𝓔 × 𝓡 × 𝓔 → ℝ. A tail query (h, r, ?) ranks every t ∈ 𝓔 by f(h, r, t); a head query (?, r, t) ranks every h.

**Def 2 (filtered rank, MRR, Hits@k; Bordes et al. 2013; ties: Berrendorf et al. 2020).** For a test triple's tail
query with answer t, the filtered candidates are t and every t′ with (h, r, t′) ∉ 𝒦*, 𝒦* the known true triples. With
a = #{candidates scoring above t} and b = #{other candidates tied with t}, the realistic rank is 1 + a + b/2, the mean
of the optimistic rank 1 + a and the pessimistic 1 + a + b, equivalently the expected rank under uniform tie-breaking.
MRR is the mean of 1/rank over both queries of every test triple, Hits@k the share with rank ≤ k. MRR is formalRAG's
Def 3 with one relevant item per query.

**Def 3 (TransE; Bordes et al. 2013).** e_x ∈ ℝᵈ with ‖e_x‖ = 1, w_r ∈ ℝᵈ, f(h, r, t) = −‖e_h + w_r − e_t‖. Bordes et al.
train the margin loss Σ [γ + d(h, r, t) − d(h′, r, t′)]₊ over facts and their corruptions; the harness trains every
model under Sun et al.'s logistic loss (the page says so).

**Prop 1 (what translations represent exactly).** (a) If e_t = e_h + w₁ on all of r₁'s facts, then w₂ = −w₁ makes
every reversed pair a fact of r₂ exactly. (b) If r₁ then r₂ hold exactly along a path h → m → t, w₃ = w₁ + w₂ holds
exactly from h to t. (c) A symmetric pair (both (h, r, t) and (t, r, h)) holds exactly only if w_r = 0 and e_h = e_t.
*Proof:* (a), (b) substitute. (c) e_t = e_h + w and e_h = e_t + w give 2w = 0. ∎ (Wang et al. 2014 derive (c) for
what they call "reflexive" relations, meaning symmetric, and the N-to-1 collapse at ε = 0; cite them for both.) Random-instance asserts, including
max(‖e_h + w − e_t‖, ‖e_t + w − e_h‖) ≥ ‖w‖ for every h, t (the sum of the two vectors is 2w).

**Prop 2 (the norm constraint is load-bearing).** Scale every embedding by s > 0. Every distance scales by s, so each
margin term is [γ − sΔ]₊ with Δ = d(h′, r, t′) − d(h, r, t). If an embedding separates every fact from its corruptions
(Δ > 0), the margin loss is nonincreasing in s and 0 for s ≥ γ/min Δ, while every rank is unchanged. *Proof:* each term
is nonincreasing in s and vanishes once sΔ ≥ γ; ranks compare distances, which all scale by s. ∎ Bordes et al. (§2)
give the reason in one clause (without the constraint, training could "trivially minimize L by artificially increasing
entity embeddings norms"); Prop 2 is that clause made exact. They normalize entities every iteration, relations once. Asserted on a KG read
off a random TransE embedding (mutual nearest neighbors, 15 facts).

**Lemma 1 (multiplying by e^{iθ} rotates).** (a + ib)(cos θ + i sin θ) = (a cos θ − b sin θ) + i(a sin θ + b cos θ), the
rotation matrix applied to (a, b); |zw| = |z| when |w| = 1; e^{iα}e^{iβ} = e^{i(α+β)}. *Proof:* expand, using i² = −1;
the angle-addition formulas. ∎ (Cite formalCalculus `eigenvalues-eigenvectors` Rem 2 for the complex-eigenvalue view.)

**Def 4 (RotatE; Sun et al. 2019).** h, t ∈ ℂᵏ, r ∈ ℂᵏ with rᵢ = e^{iθᵢ}, f(h, r, t) = γ − ‖h ∘ r − t‖, ∘ coordinatewise.
(Sun et al. use the L1 norm over the coordinate moduli, their footnote 2; the harness uses the Euclidean norm, which
Prop 5 needs. The page says so.)

**Thm 1 (relation patterns as phase equations).** (Attribution: Sun et al.'s Lemmas 1–3 say only that RotatE "can
infer" each pattern, proved one direction in their Appendices B–D; the iff forms appear as claims in their §1. The page
says Thm 1 makes the lemmas precise and proves both directions.) (a) For an entity with every hᵢ ≠ 0,
both facts of a symmetric pair hold exactly iff r ∘ r = 1, i.e. θᵢ ∈ {0, π}; then t = h ∘ r, which differs from h on
every coordinate with θᵢ = π, so the pair need not collapse. (b) r₂ holds exactly on all of r₁'s reversed pairs iff
r₂ = r̄₁ (θ₂ = −θ₁). (c) r₃ holds exactly across every r₁-then-r₂ path iff r₃ = r₁ ∘ r₂ (θ₃ ≡ θ₁ + θ₂ mod 2π).
*Proof:* (a) t = h ∘ r and h = t ∘ r give h = h ∘ r ∘ r, so hᵢ(rᵢ² − 1) = 0 and rᵢ² = 1. Conversely r ∘ r = 1 makes
t ∘ r = h ∘ r ∘ r = h. (b) t = h ∘ r₁; then t ∘ r₂ = h iff h ∘ r₁ ∘ r₂ = h iff r₁ ∘ r₂ = 1 (hᵢ ≠ 0) iff r₂ = r̄₁
(|r₁ᵢ| = 1). (c) m = h ∘ r₁, t = m ∘ r₂ = h ∘ (r₁ ∘ r₂); t = h ∘ r₃ iff r₃ = r₁ ∘ r₂; phases add by Lemma 1. ∎ The
"iff" in (b), (c) needs a witness with hᵢ ≠ 0 on every coordinate (state it). Asserted on exact constructions; one
phase off by 0.3 fails by |hᵢ||e^{0.6i} − 1|.

**Def 5 (relation models as group actions).** A group G acts on entity space X by isometries; each relation is an
element g_r and f(h, r, t) = γ − d(g_r · e_h, e_t). TransE: G = (ℝᵈ, +), w · e = e + w. RotatE: G = U(1)ᵏ, r · h = h ∘ r.
Both groups are abelian. Head queries use g_r⁻¹: d(g e_h, e_t) = d(e_h, g⁻¹ e_t).

**Thm 2 (isometries collapse what they cannot separate).** Under Def 5, suppose every fact of r has
d(g_r e_h, e_t) ≤ ε. (a) Two heads with a common tail satisfy d(e_h, e_h′) ≤ 2ε; two tails of one head likewise. (b) A
symmetric pair satisfies d(g_r² e_h, e_h) ≤ 2ε; for TransE that is ‖w_r‖ ≤ ε and ‖e_h − e_t‖ ≤ 2ε. (c) On a cycle
x₀, …, x_{L−1} of odd length L whose edges all hold both ways, adjacent entities satisfy d(e_{x_i}, e_{x_{i+1}}) ≤ 2Lε.
At ε = 0 each says the entities coincide.
*Proof:* (a) d(e_h, e_h′) = d(g e_h, g e_h′) ≤ d(g e_h, e_t) + d(e_t, g e_h′). (b) d(g² e_h, e_h) ≤ d(g² e_h, g e_t) +
d(g e_t, e_h) = d(g e_h, e_t) + d(g e_t, e_h). For TransE g² e = e + 2w, and ‖e_h − e_t‖ ≤ ‖e_h + w − e_t‖ + ‖w‖.
(c) Write eᵢ = e_{xᵢ}, indices mod L. By induction d(gʲ e₀, eⱼ) ≤ jε (d(g^{j+1} e₀, e_{j+1}) ≤ d(g^{j+1} e₀, g eⱼ) +
d(g eⱼ, e_{j+1})), so d(g^L e₀, e₀) ≤ Lε. With L = 2m + 1, by (b) at each vertex, d(g^{2m} e₀, e₀) ≤ Σ_{j<m}
d(g^{2j+2} e₀, g^{2j} e₀) = m d(g² e₀, e₀) ≤ 2mε. So d(g e₀, e₀) ≤ d(g e₀, g^L e₀) + d(g^L e₀, e₀) =
d(e₀, g^{2m} e₀) + Lε ≤ (2L − 1)ε, and d(e₀, e₁) ≤ d(e₀, g e₀) + d(g e₀, e₁) ≤ 2Lε; relabel for any edge. ∎
Remark (Cai 2019, §4.3, frames relations as group elements and a symmetric relation as one of order 2): (b) at ε = 0
says g_r² fixes e_h. A free action of a torsion-free group (translations) then forces g_r = e,
which is Prop 1(c); U(1)ᵏ has 2ᵏ − 1 elements of order 2 (Thm 1(a)). Even cycles do not collapse (alternate signs with
θ = π); odd ones do for every isometric model. Asserted on the trained full-KB TransE and RotatE (IN_SECTOR, HQ_IN,
BASED_IN heads; COMPETES_WITH pairs; the 8 triangles) and on random odd cycles (L = 3, 5, 7).

**Thm 3 (abelian models cannot tell ab from ba).** Let S₃'s Cayley graph be a KG: entities the six permutations,
relation A: x ↦ xa (a a transposition) and B: x ↦ xb (b a 3-cycle), 12 facts. If G is abelian and every fact holds
within ε, then for every x the false triple (xa, B, xba) has d(g_B e_{xa}, e_{xba}) ≤ 5ε. So a model that keeps every
corruption at distance ≥ γ_min has γ_min ≤ 5ε.
*Proof:* the B fact at xa gives d(g_B e_{xa}, e_{xab}) ≤ ε; by isometry d(g_B e_{xa}, g_B g_A e_x) = d(e_{xa}, g_A e_x)
≤ ε; so d(e_{xab}, g_B g_A e_x) ≤ 2ε. Likewise (B at x, then A at xb) d(e_{xba}, g_A g_B e_x) ≤ 2ε. G abelian:
g_A g_B = g_B g_A, so d(e_{xab}, e_{xba}) ≤ 4ε, and d(g_B e_{xa}, e_{xba}) ≤ ε + 4ε. The triple is false because B is
a function and xab ≠ xba (ab ≠ ba). ∎ Asserted on 600 random abelian embeddings and the trained runs (finding 6).

**Cor 1 (a non-abelian action fits it exactly).** Let M : S₃ → O(2) send each permutation to the symmetry of an
equilateral triangle that permutes its vertices the same way (D₃ ≅ S₃). With a point p on no reflection axis, e_x =
M(x⁻¹)p and g_A = M(a)⁻¹, g_B = M(b)⁻¹ satisfy all 12 facts exactly with six distinct entities. *Proof:* e_{xg} =
M(g⁻¹x⁻¹)p = M(g)⁻¹M(x⁻¹)p = M(g)⁻¹ e_x; the six points are distinct because only the identity of D₃ fixes p. ∎
det g_A = −1 (a reflection), det g_B = 1 (a rotation by 2π/3), ‖g_A g_B − g_B g_A‖ = 2.449. Then cite QuatE, Rotate3D,
NagE, and Cai (2019) in prose for non-abelian KG embeddings.

**Pattern × model table (page, after Cor 1).** Exact representability: symmetry (TransE only by collapse; RotatE
θ ∈ {0, π}, matchings only, Thm 2(c); DistMult always; ComplEx Im r = 0), antisymmetry (TransE, RotatE, ComplEx; not
DistMult), inversion (TransE, RotatE, ComplEx; DistMult only as symmetric), composition (TransE, RotatE commutative only;
DistMult/ComplEx not in this sense), non-commuting composition (none of the four), N-to-1/1-to-N (distance models only
by collapse, Thm 2(a)).

**Def 6 (DistMult and ComplEx; Yang et al. 2015; Trouillon et al. 2016).** DistMult: f = ⟨e_h, w_r, e_t⟩ = Σ hᵢrᵢtᵢ, a
CP decomposition of the KG's 0/1 tensor with the head and tail factor matrices tied (formalML Def 4). ComplEx:
f = Re Σ hᵢ rᵢ t̄ᵢ over ℂᵏ.

**Prop 3 (DistMult is symmetric in every relation).** f(h, r, t) = f(t, r, h). So on every antisymmetric fact it ties
the reversal: direction accuracy is exactly 1/2. *Proof:* the product is commutative. ∎ (Yang et al. 2015 never mention
symmetry; Kazemi and Poole 2018, §2 and §5.1, assert it, and Sun et al.'s Table 2 lists it. Attribute it to them.)

**Prop 4 (ComplEx's antisymmetric part).** With z = h ∘ t̄, f(h, r, t) − f(t, r, h) = −2 Σ Im rᵢ Im zᵢ. ComplEx is
symmetric in r for all entities iff Im r = 0, and antisymmetric iff Re r = 0. *Proof:* f(t, r, h) = Re Σ rᵢ z̄ᵢ, and
Re(rz) − Re(r z̄) = Re(r(z − z̄)) = Re(2i r Im z) = −2 Im r Im z. ∎

**Prop 5 (RotatE is ComplEx on the unit torus).** If |rᵢ| = 1, ‖h ∘ r − t‖² = ‖h‖² + ‖t‖² − 2 Re⟨h, r, t̄⟩. *Proof:*
‖h ∘ r‖ = ‖h‖ (Lemma 1); expand |a − b|² = |a|² + |b|² − 2 Re(a b̄). ∎

## 3. Experiment grid (58.9 s)

- **Models:** k = 16 complex dimensions, so 32 real numbers per entity for every model. Full-batch Adam (lr 0.02),
  float64. Every corruption of every triple, with each query's known training answers excluded from its negatives.
  TransE projected onto the unit sphere after every step.
- **Tuning:** split 1 (another random hiding of the same patterns). Mean pattern MRR is averaged over seeds 0–2, on
  a grid of γ (TransE 0.5/1/1.5; RotatE 3/6/9) or λ (DistMult 0.0003/0.001/0.003; ComplEx 0.003/0.01/0.03) × steps
  250/500/1,000/1,500 (checkpoints of one run). Chosen: TransE γ 1 at 250, RotatE γ 6 at 1,500, DistMult λ 0.001 at
  1,500, ComplEx λ 0.01 at 250. Every γ and λ pick is interior; the step counts sit at the grid's ends (250 for
  TransE and ComplEx, 1,500 for RotatE and DistMult). 36 runs (checkpoints 250/500/1,000/1,500), 16 s.
- **Pattern tests:** split 0 (185 training facts; 25 + 13 + 18 + 13 hidden), seeds 0–2, 12 runs.
- **Completion:** 188 known + extracted (226 at τ = 0.5, 245 at τ = 0.1), seeds 0–2, 24 runs.
- **S₃:** TransE and RotatE at k ∈ {1, 2, 8}, γ ∈ {1, 2, 4}, 2 seeds, 2,000 steps; the best γ_min/ε.
- **Streams:** `default_rng([SEED, k])`, k = 3700 + seed (initializations), 3710 + split, 3720–3727 (random-instance
  checks), 3790 (Prop 2's planted embedding).

### 3a. Numbers the page quotes (harness, 6 decimals)

- Pattern MRR (sd) on split 0: TransE 0.458889 (0.003), 0.694123 (0.109), 0.352433 (0.027), 0.815034 (0.030);
  RotatE 0.271977 (0.010), 0.676496 (0.075), 0.912037 (0.015), 0.285815 (0.026); DistMult 1.000000 (0), 0.574089
  (0.036), 0.806574 (0.134), 0.022440 (0.011); ComplEx 0.502430 (0.210), 0.780448 (0.140), 0.694319 (0.014),
  0.068124 (0.060). Direction accuracy 0.996732, 0.991830, 0.500000, 0.988562. Train MRR 0.831601, 1, 1, 1.
- Phases (median residual), full KB: symmetry 0.000000, inverse 0.000112, composition 0.022656, implication
  0.056451; split 0: 0.484901, 0.678214, 0.202081, 0.215314; random: π/4 and π/2.
- S₃ (ε, γ_min, ratio): TransE k = 1 (2.000, 2.000, 1.000), k = 2 (4.146, 3.727, 0.899), k = 8 (4.125, 3.740, 0.907);
  RotatE k = 1 (4.385, 2.275, 0.519), k = 2 (0.911, 0.906, 0.995), k = 8 (2.474, 5.052, 2.042). D₃: error 4.5 · 10⁻¹⁶,
  separation 0.600, γ_min 0.600, commutator norm 2.449490.
- Completion τ = 0.5 (MRR, H@10): COMPETES_WITH TransE 0.214340/0.833333, RotatE 0.298544/0.807692, DistMult
  0.228832/0.589744, ComplEx 0.331923/0.679487; SUBSIDIARY_OF TransE 0.555556/1, others 1/1. τ = 0.1 COMPETES_WITH:
  0.174335/0.641026, 0.276031/0.743590, 0.279528/0.641026, 0.267646/0.602564.
- KB: 114 entities (48 PERSON, 37 ORG, 23 GPE, 6 SECTOR), 10 relations, 254 triples; competitor triangles 8.

## 4. The lab: `KnowledgeGraphEmbeddingLab` (5 tabs; every number from the payload)

1. **Translate or rotate.** The plane (k = 1). Toggle *pair* (a = 0.8 + 0.6i, b = −a) or *triangle* (a, ωa, ω²a), every
   pair stated both ways. Mode *rotate*: slider over θ (0–360° by 5°); mode *translate*: slider over s ∈ [−2, 2] along
   a → b. Draw the entities, the moved points, and each fact's residual; readouts are the payload's distances. Lesson:
   θ = 180° zeroes the pair; no translation brings the pair's larger residual below |b − a| = 2; no θ zeroes the
   triangle (Thm 2(c)).
2. **Relation phases.** A unit circle per relation with the 16 trained phases as points sized by coordinate weight.
   Toggle *full KB* / *split 0*. Panels: COMPETES_WITH (sym. distance to {0, π}), PARENT_OF vs SUBSIDIARY_OF (sum),
   BASED_IN vs HQ_IN + LOCATED_IN (residual), and the four medians against random phases.
3. **Non-commuting.** S₃'s Cayley graph: D₃'s exact layout (hexagon of 6 points, A edges and B edges) beside TransE's
   and RotatE's trained k = 1 layouts. A table of ε, γ_min, γ_min/ε at k = 1, 2, 8 against Thm 3's ceiling of 5.
4. **Link prediction.** A pattern picker, a query picker (69 tail queries of split 0), and a model selector. It shows
   the top 5 filtered candidates (the answer highlighted), the filtered rank, and the per-pattern MRR ± sd table plus
   direction accuracy.
5. **Completing extraction.** A τ toggle (0.5/0.1). The extracted facts (score; true or false), then the missing
   facts with each model's filtered rank (seed 0), and MRR/Hits@10 per relation (3 seeds). The overview's example
   is highlighted.

## 5. Checkpoint questions (for Jonathan)

1. **Lead the case study with finding 2** (the closed-world reversal: RotatE's symmetric phases break exactly where the
   held-out facts are negatives)? Recommended: yes, it is this page's own result and ties to #23's PU section.
2. **Keep "inheritance" as the fourth pattern?** It is not a KB rule, and TransE wins it through Thm 2(a)'s collapse.
   Recommended: keep, as the case where collapse is a useful bias.
3. **Prop 2 with the logistic loss.** We train under Sun et al.'s loss but keep Bordes's unit-sphere constraint for
   TransE, so Prop 2 explains the constraint. Recommended: keep Prop 2 as is and say which loss the harness trains.
4. **Five lab tabs**, as specified in §4? Recommended: yes.
5. **Runtime 59.9 s**, at the 60 s rule (25 s is Topic 23's `extracted_kb()`). Accept it, or buy headroom by trimming
   costs that move no reported number (the 600 random draws in Thm 3's check, S₃ at k = 2)? Recommended: trim those
   at Phase B start; shrinking a tuning grid would put a pick on an edge.

## 5b. rigorFlag (draft)

The pattern theorems (Prop 1, Thm 1, Cor 1) concern exact satisfaction. Trained models satisfy facts only to within a
margin, so "cannot model" means "cannot model at zero distance", and Thms 2–3 bound what a margin buys. The training
numbers come from a 254-triple KB, three seeds, one hidden split (tuned on another), and one loss; at this scale the
model ranking can move with the hyperparameters (finding 8). Training treats every unknown triple as false, so held-out
true facts are pushed down during training (finding 2). Filtered evaluation also treats the KB as complete, so a true
fact the KB lacks still counts against the model. Embeddings are transductive: an entity with no training fact has
only its random initialization (here every held-out entity has at least one extracted fact). RotatE is trained with
the Euclidean norm (Sun et al.: the sum of moduli).

## 6. Notebook cell map (22 cells)

1 title · 2 imports, KB counts · 3–4 Def 2 (realistic rank vs enumeration) · 5–6 TransE (Prop 1, Prop 2's margin
loss under scaling) · 7–8 RotatE (Lemma 1, Thm 1, full-KB phases) · 9–10 split 0's broken phases · 11–12 Thm 2
(competitor triangles) · 13–14 Thm 3, Cor 1 (S₃ ratios, D₃) · 15–16 Props 3–5, direction accuracy · 17–18 pattern
tests · 19–20 completion · 21–22 harness.

## 7. References (verified 2026-10-26: titles, venues, URLs, statement numbers)

- Bordes, Usunier, Garcia-Duran, Weston, and Yakhnenko (2013). "Translating Embeddings for Modeling Multi-relational
  Data." NIPS 26, 2787–95 (pages from citing bibliographies). No DOI:
  https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html. Eq. 1 margin loss
  (head *or* tail corrupted); §2 unit-norm entities and why; §4.2 "raw" vs "filtered"; §4.3 the 1.5 threshold for
  1-to-1/1-to-N/N-to-1/N-to-N; d is L1 or L2.
- Sun, Deng, Nie, and Tang (2019). "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." ICLR.
  https://arxiv.org/abs/1902.10197. Defs 1–3 (patterns), Lemmas 1–3 qualitative (App. B–D), Eq. 3 and footnote 2 (L1
  over moduli), Eqs. 4–6 (logistic loss with γ; self-adversarial weights), Table 2 (pattern support). Their footnote 6
  swaps the 1-to-N and N-to-1 labels relative to Bordes and Wang: do not copy it.
- Yang, Yih, He, Gao, and Deng (2015). "Embedding Entities and Relations for Learning and Inference in Knowledge Bases."
  ICLR. https://arxiv.org/abs/1412.6575. §3.2 diagonal bilinear ("Bilinear-diag (DISTMULT)").
- Trouillon, Welbl, Riedel, Gaussier, and Bouchard (2016). "Complex Embeddings for Simple Link Prediction." ICML, PMLR
  48: 2071–80. https://proceedings.mlr.press/v48/trouillon16.html. Eq. 9.
- Trouillon, Dance, Gaussier, Welbl, Riedel, and Bouchard (2017). "Knowledge Graph Completion via Complex Tensor
  Factorization." JMLR 18 (130): 1–38. https://jmlr.org/papers/v18/16-563.html. Thm 2 and Cor 1 (X = Re(EWE*)), Cor 3
  (sign-rank), Thm 4 (multi-relational).
- Wang, Zhang, Feng, and Chen (2014). "Knowledge Graph Embedding by Translating on Hyperplanes." AAAI 28 (1): 1112–19.
  https://doi.org/10.1609/aaai.v28i1.8870. TransE's flaws on "reflexive" (= symmetric), 1-to-N, N-to-1, N-to-N.
- Kazemi and Poole (2018). "SimplE Embedding for Link Prediction in Knowledge Graphs." NeurIPS 31.
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html. §5.1
  DistMult "forces relations to be symmetric" (asserted); Prop 2 and Cor 1 (translational models not fully expressive).
- Nickel, Tresp, and Kriegel (2011). "A Three-Way Model for Collective Learning on Multi-Relational Data." ICML, 809–16.
  https://icml.cc/2011/papers/438_icmlpaper.pdf.
- Cai (2019). "Group Representation Theory for Knowledge Graph Embedding." arXiv:1909.05100v1, withdrawn in v2 ("company
  policy"): cite v1 and say it was withdrawn. https://arxiv.org/abs/1909.05100v1.
- Yang, Sha, and Hong (2020). "NagE: Non-Abelian Group Embedding for Knowledge Graphs." CIKM, 1735–42.
  https://doi.org/10.1145/3340531.3411875. §3.2: non-commutative composition (mother's father vs father's mother).
- Zhang, Tay, Yao, and Liu (2019). "Quaternion Knowledge Graph Embeddings." NeurIPS 32.
  https://proceedings.neurips.cc/paper_files/paper/2019/hash/d961e9f236177d65d21100592edb0769-Abstract.html.
- Gao, Sun, Shan, Lin, and Wang (2020). "Rotate3D: Representing Relations as Rotations in Three-Dimensional Space for
  Knowledge Graph Embedding." CIKM, 385–94. https://doi.org/10.1145/3340531.3411889. ACM returns 403: cite from the
  abstract and say so.
- Berrendorf, Faerman, Vermue, and Tresp (2020). "Interpretable and Fair Comparison of Link Prediction or Entity
  Alignment Methods." WI-IAT, 371–74. https://doi.org/10.1109/WIIAT50758.2020.00053. The realistic rank is §2.2 Eq. 3
  of the longer arXiv version (2002.06914, retitled "On the Ambiguity of Rank-Based Evaluation…"); cite both and say
  which one we read.
- Sun, Vashishth, Sanyal, Talukdar, and Yang (2020). "A Re-evaluation of Knowledge Graph Completion Methods." ACL,
  5516–22. https://doi.org/10.18653/v1/2020.acl-main.489. §4: TOP/BOTTOM/RANDOM tie protocols; ConvKB ties 8,520 of
  14,541 negatives.
- Ruffinelli, Broscheit, and Gemulla (2020). "You CAN Teach an Old Dog New Tricks! On Training Knowledge Graph
  Embeddings." ICLR. https://openreview.net/forum?id=BkxSmlBFvr. Abstract: with appropriate training, architecture
  differences "often shrinks and sometimes even reverses" (supports finding 8).
