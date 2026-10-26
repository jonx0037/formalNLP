"""Named entity recognition — the reference implementation for formalNLP's twenty-second topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes from
``viz_payload``. The sections follow the page:

1. Encodings. BIO and BIOES are bijections between span sets and their valid tag sequences, and validity is a
   first-order constraint (Prop 1). Every BIO chain's span law is a BIOES chain's; with token-local features the
   inclusion is strict, and a one-word left window closes the gap (Prop 2).
2. Constrained decoding. Masking invalid starts, transitions, and ends conditions a chain on validity: Viterbi
   returns the best valid sequence and forward–backward the conditional marginals (Thm 1).
3. A CRF on a contextual encoder. The gradient of log p(z | x) with respect to the unary potentials is the gold
   indicators minus the node marginals, which the encoder's backward pass carries down (Thm 2); on frozen states the
   objective is Topic 21's, strictly concave under L2 (Cor 1).
4. Span evaluation. Micro F1 is a prediction-weighted mixture of the per-type F1s, and relaxing the match criterion
   can only raise precision, recall, and F1 (Prop 3).
5. F1-optimal span decoding. A span's marginal is one forward–backward product (Prop 4); the span set maximizing F1
   of expected counts is Dinkelbach's fixed point over a weighted-interval program, and with disjoint candidates it
   thresholds the marginals at F*/2 (Thm 3).
6. Finance case study: names held out by design. Train on documents about 18 companies, test on documents about
   them and about 6 others. A word-feature CRF memorizes names; frozen Topic 12 encoder states carry context; name
   dropout that substitutes <unk> teaches a shortcut, since a feature seen only on entity tokens argues against O
   (Prop 5), while dropout that deletes the name's features does not; plus a capitalization ablation, BIOES, and a
   softmax head.
7. Thm 3 on the finance taggers: threshold sweeps over span marginals, against each model's own F*.

Cross-checks: brute-force enumeration of every tag sequence and every span subset for Props 1–2 and 4 and Thms 1
and 3; complex-step and finite-difference derivatives for Thm 2; trust-region and L-BFGS optima for Cor 1 and
Prop 5.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/named-entity-recognition/named_entity_recognition.py
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "sequence-labeling-hmm-crf"))
sys.path.insert(0, str(_HERE.parent / "fine-tuning-low-rank-adaptation"))
import fine_tuning_low_rank_adaptation as ft  # noqa: E402 — module objects: their test_* stay out of this harness
import sequence_labeling_hmm_crf as sl  # noqa: E402

cm = ft.cm
ltc, fc = sl.ltc, sl.fc

# ONE stream per random-instance claim and per sampled quantity, each from default_rng([SEED, k]); k ≥ 3500 keeps
# this module's streams apart from Topics 1–21.
SEED = sl.SEED
_r = ltc._r


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k])


# --------------------------------------------------------------------------------------------
# 1. Encodings: span sets and tag sequences (Def 1, Def 2, Props 1–2).
# --------------------------------------------------------------------------------------------
#
# A span is (s, e, τ): tokens s, …, e − 1 form a mention of type τ. A span set is a set of pairwise disjoint spans
# (Def 1). BIO marks a span's first token B-τ and the rest I-τ; BIOES marks a one-token span S-τ and a longer span's
# tokens B-τ, I-τ, …, E-τ (Def 2).

TYPES = sl.TYPES
SCHEMES = ("bio", "bioes")


def tagset(scheme: str, types: tuple[str, ...] = TYPES) -> tuple[str, ...]:
    """O, then each type's tags in order: B, I (BIO) or B, I, E, S (BIOES). The BIO set equals Topic 21's TAGS."""
    return ("O", *(f"{p}-{t}" for t in types for p in ("BI" if scheme == "bio" else "BIES")))


def encode(S, T: int, scheme: str, types: tuple[str, ...] = TYPES) -> np.ndarray:
    """The tag sequence of a span set S on T tokens."""
    ix = {t: i for i, t in enumerate(tagset(scheme, types))}
    z = np.zeros(T, int)
    for s, e, ty in S:
        if scheme == "bio" or e - s > 1:
            z[s], z[s + 1:e] = ix["B-" + ty], ix["I-" + ty]
            if scheme == "bioes":
                z[e - 1] = ix["E-" + ty]
        else:
            z[s] = ix["S-" + ty]
    return z


def decode(z, scheme: str, types: tuple[str, ...] = TYPES) -> set[tuple[int, int, str]]:
    """The span set read off a tag sequence. On a valid sequence it inverts ``encode``; an invalid one is repaired
    conlleval-style: an I or E that does not continue an open span of its type opens one, and E or S closes it."""
    tags = tagset(scheme, types)
    out, cur = set(), None
    for t, k in enumerate(list(z) + [0]):
        tag = tags[k]
        if cur is not None and not (tag[0] in "IE" and tag[2:] == cur[1]):
            out.add((cur[0], t, cur[1]))
            cur = None
        if tag != "O" and cur is None:
            cur = (t, tag[2:])
        if cur is not None and tag[0] in "ES":
            out.add((cur[0], t + 1, cur[1]))
            cur = None
    return out


def masks(scheme: str, types: tuple[str, ...] = TYPES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prop 1's first-order constraint: which tags may start a sequence, follow each tag, and end one.
    BIO: I-τ only after B-τ or I-τ. BIOES: after B-τ or I-τ only I-τ or E-τ; elsewhere only O, B, or S; a sequence
    starts on O, B, or S and ends on O, E, or S."""
    tags = tagset(scheme, types)
    pre, ty = [t[0] for t in tags], [t[2:] for t in tags]
    K = len(tags)
    start = np.array([p in "OBS" for p in pre])
    if scheme == "bio":
        trans = np.array([[pre[k] != "I" or (pre[j] in "BI" and ty[j] == ty[k]) for k in range(K)] for j in range(K)])
        end = np.ones(K, bool)
    else:
        trans = np.array([[(pre[k] in "IE" and ty[j] == ty[k]) if pre[j] in "BI" else pre[k] in "OBS"
                           for k in range(K)] for j in range(K)])
        end = np.array([p in "OES" for p in pre])
    return start, trans, end


def is_valid(z, scheme: str, types: tuple[str, ...] = TYPES) -> bool:
    start, trans, end = masks(scheme, types)
    return bool(start[z[0]] and end[z[-1]] and all(trans[a, b] for a, b in zip(z[:-1], z[1:])))


def count_span_sets(T: int, E: int) -> int:
    """N(T) span sets on T tokens with E types: the last token is outside every span, or ends a span of length l
    and type τ, so N(T) = N(T − 1) + E Σ_{l=1}^{T} N(T − l), N(0) = 1."""
    N = [1]
    for t in range(1, T + 1):
        N.append(N[t - 1] + E * sum(N[t - l] for l in range(1, t + 1)))
    return N[T]


def span_sets(T: int, types: tuple[str, ...]) -> list[frozenset]:
    """Every span set on T tokens, by the same recursion (for the brute-force checks)."""
    out: list[list[frozenset]] = [[frozenset()]]
    for t in range(1, T + 1):
        cur = list(out[t - 1])
        for l in range(1, t + 1):
            for ty in types:
                cur += [S | {(t - l, t, ty)} for S in out[t - l]]
        out.append(cur)
    return out[T]


def collapse_for(types: tuple[str, ...] = TYPES) -> np.ndarray:
    """BIOES → BIO, tag by tag: S-τ ↦ B-τ, E-τ ↦ I-τ. On valid sequences it maps encode(S, ·, "bioes") to
    encode(S, ·, "bio"), a bijection between the two valid sets (Prop 2's relabeling)."""
    bio = {t: i for i, t in enumerate(tagset("bio", types))}
    return np.array([bio[{"S": "B", "E": "I"}.get(t[0], t[0]) + t[1:]] if t != "O" else 0
                     for t in tagset("bioes", types)])


# --------------------------------------------------------------------------------------------
# 2. Constrained decoding (Thm 1).
# --------------------------------------------------------------------------------------------


def constrain(U: np.ndarray, P: np.ndarray, scheme: str, types: tuple[str, ...] = TYPES
              ) -> tuple[np.ndarray, np.ndarray]:
    """Thm 1: the chain times 1[z valid]. Invalid starts and ends get U = −∞ at the first and last positions, invalid
    transitions P = −∞; the indicator factorizes over the chain's edges, so Topic 21's recursions apply unchanged."""
    start, trans, end = masks(scheme, types)
    U = U.copy()
    U[:, 0] = np.where(start, U[:, 0], -np.inf)
    U[:, -1] = np.where(end, U[:, -1], -np.inf)
    return U, np.where(trans, P, -np.inf)


# --------------------------------------------------------------------------------------------
# 3. Span marginals and F1-optimal decoding (Prop 4, Thm 3).
# --------------------------------------------------------------------------------------------


def span_marginals(U: np.ndarray, P: np.ndarray, scheme: str, types: tuple[str, ...] = TYPES) -> dict:
    """Prop 4 on a batch of constrained chains (U (n, T, K), shared P): q[(s, e, τ)] (n,) is the probability that the
    span set contains (s, e, τ). BIO: the paths with B-τ at s, I-τ through e − 1, and anything but I-τ at e (or the
    end); BIOES: S-τ at s, or B-τ, I-τ, …, E-τ. Each is a forward message, a product along the span, and a
    backward message: alpha_s(B) · Π (transition · unary) · beta_e / Z."""
    alpha, logZ = sl.chain_forward(U, P)
    beta = sl.chain_backward(U, P)
    n, T, K = U.shape
    ix = {t: i for i, t in enumerate(tagset(scheme, types))}
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for ty in types:
            b, i = ix["B-" + ty], ix["I-" + ty]
            if scheme == "bio":
                not_i = np.array([k != i for k in range(K)])
                for s in range(T):
                    acc, last = alpha[:, s, b], b
                    for e in range(s + 1, T + 1):
                        if e == T:
                            tail = beta[:, T - 1, last]
                        else:
                            tail = sl._lse((P[last] + U[:, e] + beta[:, e])[:, not_i], 1)
                        out[(s, e, ty)] = np.exp(acc + tail - logZ)
                        if e < T:
                            acc, last = acc + P[last, i] + U[:, e, i], i
            else:
                es, ss = ix["E-" + ty], ix["S-" + ty]
                for s in range(T):
                    out[(s, s + 1, ty)] = np.exp(alpha[:, s, ss] + beta[:, s, ss] - logZ)
                    acc, last = alpha[:, s, b], b
                    for e in range(s + 2, T + 1):
                        out[(s, e, ty)] = np.exp(acc + P[last, es] + U[:, e - 1, es] + beta[:, e - 1, es] - logZ)
                        acc, last = acc + P[last, i] + U[:, e - 1, i], i
    return out


def interval_dp(cands: list[tuple[int, int, str]], w: np.ndarray, T: int) -> list[int]:
    """The disjoint subset of candidate spans with the largest total weight (weighted interval scheduling): best[t] is
    the best total over spans inside [0, t). Returns the chosen candidates' indices; zero or negative weights are
    never chosen."""
    ending: dict[int, list[int]] = {}
    for j, c in enumerate(cands):
        ending.setdefault(c[1], []).append(j)
    best = np.zeros(T + 1)
    arg: list[int | None] = [None] * (T + 1)
    for t in range(1, T + 1):
        best[t], arg[t] = best[t - 1], None
        for j in ending.get(t, ()):
            v = best[cands[j][0]] + w[j]
            if w[j] > 0 and v > best[t]:
                best[t], arg[t] = v, j
    out, t = [], T
    while t > 0:
        if arg[t] is None:
            t -= 1
        else:
            out.append(arg[t])
            t = cands[arg[t]][0]
    return out[::-1]


def threshold_decode(sents: list[tuple[list, np.ndarray, int]], tau: float) -> list[list[int]]:
    """Per sentence (candidates, marginals q, length T): the disjoint spans maximizing Σ (q − τ)."""
    return [interval_dp(c, q - tau, T) for c, q, T in sents]


def f1_decode(sents: list[tuple[list, np.ndarray, int]], m: float) -> tuple[list[list[int]], float]:
    """Thm 3 by Dinkelbach's method: the disjoint span sets Ŝ (one per sentence) maximizing F1 of expected counts,
    2 Σ_{Ŝ} q / (|Ŝ| + m), where m = E|S| summed over the corpus. Each step solves max Σ (2q − F) by the interval
    program (= threshold_decode at τ = F/2) and sets F to the ratio of that solution; F rises strictly until no set
    beats it, so the loop ends after finitely many steps at F*, with the optimal set thresholded at F*/2."""
    F = 0.0
    while True:
        sel = threshold_decode(sents, F / 2)
        Q = sum(float(q[js].sum()) for (_, q, _), js in zip(sents, sel))
        k = sum(len(js) for js in sel)
        F_new = 2 * Q / (k + m)
        if F_new <= F + 1e-15:
            return sel, F
        F = F_new


# --------------------------------------------------------------------------------------------
# 4. Span evaluation (Def 4, Prop 3).
# --------------------------------------------------------------------------------------------

MATCHES = ("strict", "boundary", "type", "partial")


def _matches(g: tuple, p: tuple, how: str) -> bool:
    if how == "strict":
        return g == p
    if how == "boundary":
        return g[:2] == p[:2]
    overlap = g[0] < p[1] and p[0] < g[1]
    return overlap and (how == "partial" or g[2] == p[2])


def prf(golds: list[set], preds: list[set], how: str = "strict", ty: str | None = None) -> dict:
    """Def 4: a predicted span is correct if it matches some gold span of its sentence, a gold span is found if some
    predicted span matches it; precision and recall are those shares, F1 their harmonic mean. With ``ty``, only spans
    of that type count (strict matching). Under strict matching both shares count the same pairs, so
    F1 = 2 TP / (|G| + |P̂|)."""
    hp = hg = ng = npd = 0
    for G, Pd in zip(golds, preds):
        if ty is not None:
            G, Pd = {g for g in G if g[2] == ty}, {p for p in Pd if p[2] == ty}
        hp += sum(any(_matches(g, p, how) for g in G) for p in Pd)
        hg += sum(any(_matches(g, p, how) for p in Pd) for g in G)
        ng, npd = ng + len(G), npd + len(Pd)
    P_, R_ = hp / npd if npd else 0.0, hg / ng if ng else 0.0
    return {"p": P_, "r": R_, "f1": 2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0.0, "gold": ng, "pred": npd}


def micro_mixture(golds: list[set], preds: list[set], types: tuple[str, ...] = TYPES) -> tuple[float, dict]:
    """Prop 3(a): micro F1 = Σ_τ w_τ F1_τ with w_τ = (|G_τ| + |P̂_τ|) / Σ (|G| + |P̂|). Returns the mixture and w."""
    per = {t: prf(golds, preds, "strict", t) for t in types}
    tot = sum(r["gold"] + r["pred"] for r in per.values())
    w = {t: (r["gold"] + r["pred"]) / tot for t, r in per.items()}
    return sum(w[t] * per[t]["f1"] for t in types), w


# ============================================================================================
# Harness — exactness on small instances (Props 1–4, Thms 1–3, Cor 1).
# ============================================================================================


def _brute_valid(U: np.ndarray, P: np.ndarray, scheme: str, types: tuple[str, ...]):
    """Every valid tag sequence of one chain (U (T, K)), its score, and its probability under p(z | x, valid)."""
    paths, s = sl.brute_force(U, P)
    ok = np.array([is_valid(z, scheme, types) for z in paths])
    paths, s = paths[ok], s[ok]
    return paths, s, np.exp(s - (np.log(np.exp(s - s.max()).sum()) + s.max()))


def test_prop1_encodings_are_bijections():
    """Prop 1: for E = 1, 2 types and T ≤ 5 tokens, the valid sequences of each scheme are exactly the encodings of
    the N(T) span sets (decode ∘ encode = id, encode ∘ decode = id on valid sequences), N(T) also equals the
    transfer-matrix count start · M^{T−1} · end, and decode maps all K^T sequences onto the span sets (many-to-one)."""
    for E in (1, 2):
        types = TYPES[:E]
        for scheme in SCHEMES:
            K = len(tagset(scheme, types))
            start, trans, end = masks(scheme, types)
            for T in range(1, 6):
                sets = span_sets(T, types)
                assert len(sets) == len(set(sets)) == count_span_sets(T, E)
                assert all(decode(encode(S, T, scheme, types), scheme, types) == S for S in sets)
                seqs = list(itertools.product(range(K), repeat=T))
                valid = [z for z in seqs if is_valid(np.array(z), scheme, types)]
                assert len(valid) == len(sets)
                assert all(tuple(encode(decode(z, scheme, types), T, scheme, types)) == z for z in valid)
                M = trans.astype(int)
                assert start.astype(int) @ np.linalg.matrix_power(M, T - 1) @ end.astype(int) == len(sets)
                assert {frozenset(decode(z, scheme, types)) for z in seqs} == set(sets)
                assert len(seqs) > len(valid)
    rng = _rng(3520)  # decode reads BIO exactly as Topic 21's conlleval-style spans()
    for _ in range(300):
        z = rng.integers(0, len(sl.TAGS), int(rng.integers(1, 12)))
        assert decode(z, "bio") == sl.spans(z)


def test_prop2_bio_chains_are_bioes_chains():
    """Prop 2 (inclusion): relabeling a BIO chain through collapse_for() gives a BIOES chain whose constrained span law
    equals the BIO chain's constrained span law, for random potentials (E = 2, T ≤ 4)."""
    rng = _rng(3521)
    types = TYPES[:2]
    c = collapse_for(types)
    for _ in range(20):
        T = int(rng.integers(1, 5))
        Kb = len(tagset("bio", types))
        U, P = rng.normal(size=(T, Kb)) * 1.5, rng.normal(size=(Kb, Kb)) * 1.5
        laws = []
        for scheme, (Us, Ps) in (("bio", (U, P)), ("bioes", (U[:, c], P[np.ix_(c, c)]))):
            paths, _, p = _brute_valid(Us, Ps, scheme, types)
            law: dict = {}
            for z, pz in zip(paths, p):
                key = frozenset(decode(z, scheme, types))
                law[key] = law.get(key, 0.0) + pz
            laws.append(law)
        assert laws[0].keys() == laws[1].keys() and all(abs(laws[0][k] - laws[1][k]) < 1e-12 for k in laws[0])


PROP2_DATA = ((("a", "d"), {(0, 1, "ORG")}), (("c", "d"), {(0, 2, "ORG")}))  # one-token vs two-token span


def prop2_counterexample() -> dict:
    """Prop 2 (strict): the two-sentence data set PROP2_DATA. With token-local features (bias, the word), a BIO chain
    gives the two gold span sets total probability ≤ 1, since given z₁ = B the law of z₂ depends on x₂ = d alone;
    BIOES separates S (a) from B (c), and a BIO chain with the left word as a feature separates them too. Each is fit
    at λ = 10⁻³; returns the two gold probabilities per model (constrained marginal law of the span set)."""
    types = ("ORG",)
    out = {}
    for name, scheme, left in (("bio", "bio", 0), ("bioes", "bioes", 0), ("bio+left", "bio", 1)):
        feats = [sl.window_features(list(x), 0, left=left) for x, _ in PROP2_DATA]
        zs = [encode(S, len(x), scheme, types) for x, S in PROP2_DATA]
        vocab = sl.index_features(feats)
        K = len(tagset(scheme, types))
        data = sl.make_data(feats, zs, vocab, K)
        th = sl.crf_fit(data, 1e-3, gtol=1e-10)
        probs = []
        for f, (x, S) in zip(feats, PROP2_DATA):
            d1 = sl.make_data([f], [np.zeros(len(x), int)], vocab, K)
            U, P = sl.crf_potentials(th, d1.buckets[0].X, len(x), d1.F, K)
            paths, _, p = _brute_valid(U[0], P, scheme, types)
            probs.append(float(sum(pz for z, pz in zip(paths, p) if decode(z, scheme, types) == S)))
        out[name] = probs
    return out


def test_prop2_strict_for_token_local_features():
    """Prop 2 (strict): the token-local BIO chain's two gold probabilities sum to ≤ 1 for every θ (random draws) and
    at its fit; BIOES and BIO-with-left-word both exceed 0.95 on each sentence."""
    rng = _rng(3522)
    types = ("ORG",)
    feats = [sl.window_features(list(x), 0) for x, _ in PROP2_DATA]
    vocab = sl.index_features(feats)
    K = len(tagset("bio", types))
    for _ in range(200):
        th = rng.normal(size=sl.crf_size(len(vocab), K)) * 3
        tot = 0.0
        for f, (x, S) in zip(feats, PROP2_DATA):
            d1 = sl.make_data([f], [np.zeros(len(x), int)], vocab, K)
            U, P = sl.crf_potentials(th, d1.buckets[0].X, len(x), d1.F, K)
            paths, _, p = _brute_valid(U[0], P, "bio", types)
            tot += sum(pz for z, pz in zip(paths, p) if decode(z, "bio", types) == S)
        assert tot <= 1 + 1e-12
    ce = prop2_counterexample()
    assert sum(ce["bio"]) <= 1 + 1e-12
    assert min(ce["bioes"]) > 0.95 and min(ce["bio+left"]) > 0.95


def test_thm1_constrained_chain_is_the_conditional():
    """Thm 1: on random chains (both schemes, E = 2), the masked chain's Viterbi path is the highest-scoring valid
    path, its node marginals are the brute-force marginals given validity, and log Z_C − log Z = log p(valid)."""
    rng = _rng(3523)
    types = TYPES[:2]
    for trial in range(30):
        scheme = SCHEMES[trial % 2]
        K = len(tagset(scheme, types))
        T = int(rng.integers(1, 5))
        U, P = rng.normal(size=(T, K)) * 2, rng.normal(size=(K, K)) * 2
        Uc, Pc = constrain(U[None], P, scheme, types)
        paths, s, p = _brute_valid(U, P, scheme, types)
        path, best = sl.viterbi(Uc, Pc)
        assert abs(best[0] - s.max()) < 1e-12 and tuple(path[0]) == tuple(paths[np.argmax(s)])
        lzc, node, _ = sl.chain_marginals(Uc, Pc)
        for t in range(T):
            assert np.allclose(node[0, t], np.bincount(paths[:, t], p, K), atol=1e-12)
        all_paths, all_s = sl.brute_force(U, P)
        lz = np.log(np.exp(all_s - all_s.max()).sum()) + all_s.max()
        assert abs((lzc[0] - lz) - math.log(np.exp(s - lz).sum())) < 1e-12


def test_thm2_unary_gradient_is_gold_minus_marginals():
    """Thm 2: ∂ log p(z | x)/∂U_t(k) = 1[z_t = k] − γ_t(k) and ∂/∂P(j, k) = #{t: z_t = j, z_{t+1} = k} − Σ_t ξ_t(j, k),
    by complex step on random chains; with P = 0 the marginals are the per-token softmax (the softmax head)."""
    rng = _rng(3524)
    h = 1e-20
    for _ in range(10):
        T, K = int(rng.integers(2, 6)), int(rng.integers(2, 5))
        U, P = rng.normal(size=(1, T, K)), rng.normal(size=(K, K))
        z = rng.integers(0, K, (1, T))

        def logp(U_, P_):
            return (sl.path_score(U_, P_, z) - sl.chain_forward(U_, P_)[1])[0]

        _, node, pair = sl.chain_marginals(U, P)
        gU = np.array([[logp(U + 1j * h * np.eye(T * K)[t * K + k].reshape(1, T, K), P).imag / h for k in range(K)]
                       for t in range(T)])
        assert np.abs(gU - (np.eye(K)[z[0]] - node[0])).max() < 1e-12
        gP = np.array([[logp(U, P + 1j * h * np.eye(K * K)[j * K + k].reshape(K, K)).imag / h for k in range(K)]
                       for j in range(K)])
        obs = np.zeros((K, K))
        np.add.at(obs, (z[0, :-1], z[0, 1:]), 1)
        assert np.abs(gP - (obs - pair[0].sum(0))).max() < 1e-12
        node0 = sl.chain_marginals(U, np.zeros((K, K)))[1]
        soft = np.exp(U - U.max(2, keepdims=True))
        assert np.allclose(node0, soft / soft.sum(2, keepdims=True), atol=1e-14)


def test_thm2_chain_rule_into_the_encoder():
    """Thm 2 through the encoder: with U_t = W h_t + b on Topic 12's final states, ft.encoder_backward fed
    dL/dh_t = (γ_t − e_{z_t}) Wᵀ gives the gradient of L = −log p(z | x) in the encoder's weights, checked by central
    differences on token-embedding and attention coordinates (float64 copies; ReLU kinks are measure-zero)."""
    rng = _rng(3525)
    P_, blocks = encoder()
    K = len(sl.TAGS)
    W, b = rng.normal(size=(P_["E"].shape[1], K)) * 0.3, rng.normal(size=K) * 0.3
    Ptr = rng.normal(size=(K, K)) * 0.3
    tok = rng.integers(0, P_["E"].shape[0] - 1, (2, 7))
    z = rng.integers(0, K, (2, 7))

    def loss(Pp, bl):
        out = ft.final_states(Pp, bl, tok)[0]
        U = out @ W + b
        return float(-(sl.path_score(U, Ptr, z) - sl.chain_forward(U, Ptr)[1]).sum())

    out, r, caches = ft.final_states(P_, blocks, tok)
    U = out @ W + b
    node = sl.chain_marginals(U, Ptr)[1]
    G, Gb = ft.encoder_backward(P_, blocks, tok, out, r, caches, (node - np.eye(K)[z]) @ W.T)
    eps = 1e-6
    for leaf, key, g in ((P_, "E", G["E"]), (blocks[0]["attn"], "Wq", Gb[0]["attn"]["Wq"]),
                         (blocks[1]["attn"], "Wv", Gb[1]["attn"]["Wv"])):
        for _ in range(4):
            idx = tuple(int(rng.integers(0, s)) for s in leaf[key].shape)
            if key == "E":
                idx = (int(tok[0, int(rng.integers(0, 7))]), idx[1])
            old = leaf[key][idx]
            leaf[key][idx] = old + eps
            up = loss(P_, blocks)
            leaf[key][idx] = old - eps
            dn = loss(P_, blocks)
            leaf[key][idx] = old
            assert abs((up - dn) / (2 * eps) - g[idx]) < 1e-6 * max(1.0, abs(g[idx]))


def test_cor1_frozen_features_give_a_unique_optimum():
    """Cor 1: on dense real-valued token features (a stand-in for frozen encoder states), the penalized CRF objective
    is strictly concave, so trust-region fits from two starts and the L-BFGS fit reach one θ."""
    rng = _rng(3526)
    K, F = 3, 5
    zs = [rng.integers(0, K, int(rng.integers(1, 6))) for _ in range(25)]
    rows = [np.hstack([rng.normal(size=(len(z), F - 1)), np.ones((len(z), 1))]) for z in zs]
    data = build(rows, zs, K)
    a = sl.crf_fit(data, 0.05, gtol=1e-11)
    b = sl.crf_fit(data, 0.05, x0=rng.normal(size=sl.crf_size(F, K)) * 3, gtol=1e-11)
    c = sl.crf_fit_lbfgs(data, 0.05)
    assert np.abs(a - b).max() < 1e-8 and np.abs(a - c).max() < 1e-4


def test_prop4_span_marginals_match_brute_force():
    """Prop 4: span_marginals equals the brute-force probability that the span set contains each span, for both
    schemes (E = 2, T ≤ 4, constrained chains), and Σ_spans q = E|S| = the expected number of B and S tags."""
    rng = _rng(3527)
    types = TYPES[:2]
    for trial in range(16):
        scheme = SCHEMES[trial % 2]
        K = len(tagset(scheme, types))
        T = int(rng.integers(1, 5))
        U, P = rng.normal(size=(T, K)) * 2, rng.normal(size=(K, K)) * 2
        Uc, Pc = constrain(U[None], P, scheme, types)
        q = span_marginals(Uc, Pc, scheme, types)
        paths, _, p = _brute_valid(U, P, scheme, types)
        sets = [decode(z, scheme, types) for z in paths]
        for key, v in q.items():
            assert abs(v[0] - sum(pz for S, pz in zip(sets, p) if key in S)) < 1e-12
        node = sl.chain_marginals(Uc, Pc)[1][0]
        opens = [k for k, t in enumerate(tagset(scheme, types)) if t[0] in "BS"]
        assert abs(sum(v[0] for v in q.values()) - node[:, opens].sum()) < 1e-12


def _random_instance(rng: np.random.Generator, n_sent: int):
    """A few short constrained BIO chains (E = 2) as F1-decoding input: per sentence (candidates, q, T), and m."""
    types = TYPES[:2]
    K = len(tagset("bio", types))
    sents, m = [], 0.0
    for _ in range(n_sent):
        T = int(rng.integers(1, 4))
        U, P = constrain(rng.normal(size=(1, T, K)) * 2, rng.normal(size=(K, K)), "bio", types)
        q = span_marginals(U, P, "bio", types)
        cands = list(q)
        qs = np.array([q[c][0] for c in cands])
        sents.append((cands, qs, T))
        m += qs.sum()
    return sents, m


def _subsets(cands: list) -> list[list[int]]:
    out = []
    for r in range(len(cands) + 1):
        for js in itertools.combinations(range(len(cands)), r):
            spans_ = sorted((cands[j][0], cands[j][1]) for j in js)
            if all(a[1] <= b[0] for a, b in zip(spans_, spans_[1:])):
                out.append(list(js))
    return out


def test_thm3_dinkelbach_is_the_f1_optimum():
    """Thm 3: on random multi-sentence instances, f1_decode's F equals the maximum of 2 Σ_Ŝ q / (|Ŝ| + m) over every
    combination of disjoint span subsets (brute force), its chosen sets attain it, and they are exactly the
    interval-program solution at threshold F*/2."""
    rng = _rng(3528)
    for _ in range(12):
        sents, m = _random_instance(rng, int(rng.integers(1, 3)))
        best = 0.0
        for combo in itertools.product(*[_subsets(c) for c, _, _ in sents]):
            Q = sum(float(q[js].sum()) for (_, q, _), js in zip(sents, combo))
            best = max(best, 2 * Q / (sum(len(js) for js in combo) + m))
        sel, F = f1_decode(sents, m)
        Q = sum(float(q[js].sum()) for (_, q, _), js in zip(sents, sel))
        assert abs(F - best) < 1e-12 and abs(2 * Q / (sum(map(len, sel)) + m) - best) < 1e-12
        assert sel == threshold_decode(sents, F / 2)


def test_thm3_disjoint_candidates_threshold_at_half_f():
    """Thm 3 with disjoint candidates (Lipton, Elkan, and Naryanaswamy's setting): the optimal set is every item with
    q > F*/2, checked against brute force over all subsets of 10 items."""
    rng = _rng(3529)
    for _ in range(20):
        q = rng.beta(0.6, 0.6, 10)
        m = q.sum() + rng.uniform(0, 2)  # gold mass outside the candidates (spans no candidate covers)
        sents = [([(j, j + 1, "ORG") for j in range(10)], q, 10)]
        sel, F = f1_decode(sents, m)
        best = max(2 * q[list(js)].sum() / (len(js) + m) for r in range(11) for js in itertools.combinations(range(10), r))
        assert abs(F - best) < 1e-12 and sorted(sel[0]) == sorted(np.flatnonzero(q > F / 2).tolist())


def test_prop3_micro_mixture_and_match_ordering():
    """Prop 3 on random span sets: micro F1 equals the prediction-weighted mixture of per-type F1s, and precision,
    recall, and F1 never fall as the match relaxes (strict → boundary → partial, strict → type → partial)."""
    rng = _rng(3530)
    types = TYPES[:3]
    S = span_sets(4, types)
    for _ in range(50):
        golds, preds = [], []
        for _ in range(8):
            golds.append(set(S[rng.integers(len(S))]))
            preds.append(set(S[rng.integers(len(S))]))
        mix, w = micro_mixture(golds, preds, types)
        assert abs(prf(golds, preds)["f1"] - mix) < 1e-12 and abs(sum(w.values()) - 1) < 1e-12
        r = {h: prf(golds, preds, h) for h in MATCHES}
        for lo, hi in (("strict", "boundary"), ("boundary", "partial"), ("strict", "type"), ("type", "partial")):
            assert all(r[lo][k] <= r[hi][k] + 1e-12 for k in ("p", "r", "f1"))


# --------------------------------------------------------------------------------------------
# 5. Feature rows and data sets (dense, sparse, or both).
# --------------------------------------------------------------------------------------------


def build(rows: list, zs: list[np.ndarray], K: int) -> sl.Data:
    """Topic 21's Data from per-sentence feature rows (each (L, F), dense or sparse), bucketed by length — the
    dense-feature generalization of sl.make_data, which takes feature strings."""
    by_len: dict[int, list[int]] = {}
    for i, z in enumerate(zs):
        by_len.setdefault(len(z), []).append(i)
    F = rows[0].shape[1]
    obs_W = np.zeros((F, K))
    obs_trans, obs_start, obs_end = np.zeros((K, K)), np.zeros(K), np.zeros(K)
    buckets = []
    for L in sorted(by_len):
        X = sp.vstack([sp.csr_matrix(rows[i]) for i in by_len[L]], format="csr")
        z = np.array([zs[i] for i in by_len[L]], int).reshape(-1, L)
        buckets.append(sl.Bucket(X, z))
        obs_W += np.asarray(X.T @ np.eye(K)[z.ravel()])
        np.add.at(obs_trans, (z[:, :-1].ravel(), z[:, 1:].ravel()), 1)
        np.add.at(obs_start, z[:, 0], 1)
        np.add.at(obs_end, z[:, -1], 1)
    return sl.Data(tuple(buckets), F, K, len(zs), obs_W, obs_trans, obs_start, obs_end)


@lru_cache(maxsize=1)
def encoder() -> tuple[dict, list]:
    """Topic 12's filing encoder (pretrained on generate_corpus(300, 0), the full roster; ~21 s), as float64 copies."""
    m = ft.pretrained()
    P = {k: v.astype(np.float64) for k, v in m.P.items()}
    blocks = [{s: {k: v.astype(np.float64) for k, v in blk[s].items()} for s in blk} for blk in m.blocks]
    return P, blocks


def states(toks: tuple, unk: tuple | None) -> list[np.ndarray]:
    """LN(x_L) of Topic 12's encoder for each sentence, read alone with a trailing [SEP] (as sentences end in its
    pretraining stream): (L, d) per sentence. Words outside its vocabulary, and positions where ``unk`` is True,
    enter as [UNK]."""
    P, blocks = encoder()
    voc, V = cm.filing_vocab()
    UNK, SEP = V - 2, V - 1
    by_len: dict[int, list[int]] = {}
    for i, x in enumerate(toks):
        by_len.setdefault(len(x), []).append(i)
    out: list[np.ndarray | None] = [None] * len(toks)
    for L, idx in by_len.items():
        tok = np.array([[UNK if unk is not None and unk[i][t] else voc.get(w, UNK) for t, w in enumerate(toks[i])]
                        + [SEP] for i in idx])
        H = ft.final_states(P, blocks, tok)[0][:, :L]
        for j, i in enumerate(idx):
            out[i] = H[j]
    return out


# --------------------------------------------------------------------------------------------
# 6. Finance case study: names held out by design (Prop 5).
# --------------------------------------------------------------------------------------------
#
# fc.split_roster() holds out one company per sector (with its subsidiaries, officers, and flagship product). The
# taggers train on the first N_DOCS documents of a corpus about the other 18 and are tested on two corpora drawn
# alike: one about the same 18 ("seen" names) and one about the 6 held out ("unseen"), whose every ORG, PERSON, and
# PRODUCT mention contains a word the training text never shows. GPE, SECTOR, DATE, MONEY, and PERCENT are shared.

TRAIN_ROSTER, HELD_ROSTER = fc.split_roster()
TRAIN_RUN, TEST_RUN = (300, 0), (200, 1)  # generate_corpus(n_docs, seed) runs
N_DOCS = 30
UNK_WORD = "<unk>"
NAME_DROPOUT = 0.5  # fixed a priori (Lample et al. 2016 replace singletons by UNK at 0.5); never tuned
DROPOUT_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
TESTS = ("seen", "unseen")
_IDENTITY = ("w0=", "pre3=", "suf3=")


@dataclass(frozen=True)
class Sentences:
    toks: tuple[tuple[str, ...], ...]
    z: tuple[np.ndarray, ...]  # BIO tags (Topic 21's TAGS)
    caps: tuple[np.ndarray, ...]  # capitalized in the raw text and not sentence-initial
    spans: tuple[frozenset, ...]
    doc: tuple[int, ...]


def ner_sentences(docs) -> Sentences:
    """Topic 21's BIO sentences, plus each token's case read from the raw text (fc._tokens lowercases)."""
    xs, zs, doc = sl.bio_sentences(docs)
    caps = []
    for d in docs:
        for s, e in d.sentences:
            ms = re.findall(r"[A-Za-z]+", d.text[s:e])
            if ms:
                caps.append(np.array([t > 0 and w[0].isupper() for t, w in enumerate(ms)]))
    assert [len(c) for c in caps] == [len(x) for x in xs]
    return Sentences(tuple(map(tuple, xs)), tuple(zs), tuple(caps), tuple(frozenset(sl.spans(z)) for z in zs),
                     tuple(doc))


@lru_cache(maxsize=None)
def split(name: str) -> Sentences:
    """"train": the first N_DOCS documents of generate_corpus(300, 0, roster=TRAIN_ROSTER); "seen" and "unseen":
    generate_corpus(200, 1) on the training roster and on the held-out roster."""
    roster = HELD_ROSTER if name == "unseen" else TRAIN_ROSTER
    s = ner_sentences(fc.generate_corpus(*(TRAIN_RUN if name == "train" else TEST_RUN), roster=roster))
    if name != "train":
        return s
    k = int(np.searchsorted(s.doc, N_DOCS))
    return Sentences(s.toks[:k], s.z[:k], s.caps[:k], s.spans[:k], s.doc[:k])


@lru_cache(maxsize=1)
def train_words() -> frozenset[str]:
    return frozenset(w for x in split("train").toks for w in x)


@lru_cache(maxsize=1)
def _dropout_uniforms() -> tuple[np.ndarray, ...]:
    """One uniform per training token, drawn once (stream 3500): a token inside a gold span is dropped at rate p iff
    its uniform is below p, so the dropped sets are nested in p and shared by every model (common random numbers)."""
    rng = _rng(3500)
    return tuple(rng.random(len(x)) for x in split("train").toks)


@lru_cache(maxsize=None)
def dropped(p: float) -> tuple[np.ndarray, ...]:
    return tuple((u < p) & (z != 0) for u, z in zip(_dropout_uniforms(), split("train").z))


def word_features(toks, caps: np.ndarray | None, mask: np.ndarray | None, how: str) -> list[list[str]]:
    """Topic 21's finance features (the word, its 3-letter prefix and suffix, the words ±1 and ±2), plus "cap" when
    ``caps``. A token in ``mask`` is dropped: how = "substitute" rewrites it as <unk> (every feature then names
    <unk>); how = "delete" removes its own word, prefix, and suffix features and every neighbor's feature naming it,
    adding none."""
    v = [UNK_WORD if how == "substitute" and mask is not None and mask[t] else w for t, w in enumerate(toks)]
    fs = sl.fin_features(v, False)
    if how == "delete" and mask is not None:
        T = len(v)
        for t in range(T):
            keep = []
            for f in fs[t]:
                if f.startswith(_IDENTITY):
                    if mask[t]:
                        continue
                elif f[0] == "w" and f[1] in "+-":
                    u = t + int(f[1:f.index("=")])
                    if 0 <= u < T and mask[u]:
                        continue
                keep.append(f)
            fs[t] = keep
    if caps is not None:
        for t in np.flatnonzero(caps):
            fs[t].append("cap")
    return fs


def _sparse(feats: list[list[str]], vocab: dict[str, int]) -> sp.csr_matrix:
    idx, ptr = [], [0]
    for fs in feats:
        idx.extend(sorted({vocab[f] for f in fs if f in vocab}))
        ptr.append(len(idx))
    return sp.csr_matrix((np.ones(len(idx)), np.array(idx, int), np.array(ptr)), shape=(len(feats), len(vocab)))


@dataclass(frozen=True)
class Model:
    feats: str  # "words", "words+case", "encoder", or "both" (word features and encoder states side by side)
    dropout: str  # "none", "substitute", or "delete" (how a dropped token's word features change)
    p: float  # name-dropout rate; a dropped token also enters the encoder as [UNK]
    scheme: str  # "bio" or "bioes"
    head: str  # "crf" or "softmax" (an independent softmax per token)


@lru_cache(maxsize=None)
def _test_words(test: str, how: str) -> tuple:
    """A test split's word features: raw words, or (for substitution-trained taggers) every word unseen in training
    rewritten as <unk>, the test-time half of that recipe."""
    s, known = split(test), train_words()
    mask = [np.array([w not in known for w in x]) for x in s.toks] if how == "substitute" else None
    return tuple(word_features(x, None, None if mask is None else mask[i], how) for i, x in enumerate(s.toks))


@lru_cache(maxsize=None)
def _test_states(test: str, unk_names: bool) -> tuple:
    """A test split's encoder states; with ``unk_names`` every word unseen in training enters as [UNK] (the proxy
    for names never seen anywhere, even in pretraining)."""
    s, known = split(test), train_words()
    unk = tuple(np.array([w not in known for w in x]) for x in s.toks) if unk_names else None
    return tuple(states(s.toks, unk))


def _rows(model: Model, words: list | None, caps, H: list | None, vocab: dict) -> list:
    """Per-sentence feature rows: sparse word features, encoder states plus a bias, or both side by side."""
    rows = []
    for i in range(len(words if words is not None else H)):
        parts = []
        if model.feats != "encoder":
            fs = words[i] if caps is None else [f + ["cap"] * bool(c) for f, c in zip(words[i], caps[i])]
            parts.append(_sparse(fs, vocab))
        if model.feats in ("encoder", "both"):
            parts.append(sp.csr_matrix(np.hstack([H[i], np.ones((len(H[i]), 1))])))
        rows.append(parts[0] if len(parts) == 1 else sp.hstack(parts, format="csr"))
    return rows


@lru_cache(maxsize=None)
def fit(model: Model) -> dict:
    """A tagger trained on the training split. CRF: MAP under θ ~ N(0, I) (λ = 1/sentences on Topic 21's mean
    objective), by L-BFGS. Softmax head: Topic 17's objective on token rows (λ = 1/tokens, the same prior on the
    weights; intercepts unpenalized), by L-BFGS."""
    tr = split("train")
    mask = dropped(model.p) if model.dropout != "none" else None
    words = H = None
    vocab: dict = {}
    if model.feats != "encoder":
        words = [word_features(x, None, None if mask is None else mask[i], model.dropout)
                 for i, x in enumerate(tr.toks)]
        caps = tr.caps if model.feats == "words+case" else None
        vocab = sl.index_features([f if caps is None else [g + ["cap"] * bool(c) for g, c in zip(f, cc)]
                                   for f, cc in zip(words, caps or [None] * len(words))] if caps is not None else words)
    if model.feats in ("encoder", "both"):
        H = states(tr.toks, mask)
    rows = _rows(model, words, tr.caps if model.feats == "words+case" else None, H, vocab)
    K = len(tagset(model.scheme))
    zs = [encode(S, len(x), model.scheme) for x, S in zip(tr.toks, tr.spans)]
    if model.head == "crf":
        data = build(rows, zs, K)
        return {"theta": sl.crf_fit_lbfgs(data, 1.0 / len(zs)), "vocab": vocab, "F": data.F, "K": K}
    X = sp.vstack(rows, format="csr").toarray()
    y = np.concatenate(zs)
    r = sl.optimize.minimize(ltc.lr_objective, np.zeros(K * X.shape[1] + K), args=(X, y, K, 1.0 / len(y)), jac=True,
                             method="L-BFGS-B", options={"gtol": 1e-6, "ftol": 0.0, "maxiter": 5000, "maxcor": 30})
    return {"W": r.x[:K * X.shape[1]].reshape(K, X.shape[1]), "b": r.x[K * X.shape[1]:], "vocab": vocab,
            "F": X.shape[1], "K": K}


def _potentials(model: Model, f: dict, X, L: int) -> tuple[np.ndarray, np.ndarray]:
    if model.head == "crf":
        return sl.crf_potentials(f["theta"], X, L, f["F"], f["K"])
    Z = (np.asarray(X @ f["W"].T) + f["b"]).reshape(-1, L, f["K"])
    return Z - sl._lse(Z, 2)[..., None], np.zeros((f["K"], f["K"]))


@lru_cache(maxsize=None)
def chains(model: Model, test: str, unk_names: bool) -> tuple:
    """The test split's chains under a trained model, bucketed by length: (sentence indices, U, P) per bucket."""
    s, f = split(test), fit(model)
    words = _test_words(test, model.dropout) if model.feats != "encoder" else None
    H = _test_states(test, unk_names) if model.feats in ("encoder", "both") else None
    rows = _rows(model, words, s.caps if model.feats == "words+case" else None, H, f["vocab"])
    by_len: dict[int, list[int]] = {}
    for i, x in enumerate(s.toks):
        by_len.setdefault(len(x), []).append(i)
    out = []
    for L in sorted(by_len):
        X = sp.vstack([rows[i] for i in by_len[L]], format="csr")
        U, P = _potentials(model, f, X, L)
        out.append((tuple(by_len[L]), U, P))
    return tuple(out)


@lru_cache(maxsize=None)
def predictions(model: Model, test: str, unk_names: bool) -> dict:
    """Span sets decoded unconstrained (CRF: Viterbi; softmax: per-token argmax) and by constrained Viterbi (Thm 1),
    and the share of unconstrained sequences that are invalid."""
    n = len(split(test).toks)
    raw, con = [None] * n, [None] * n
    invalid = 0
    for idx, U, P in chains(model, test, unk_names):
        zr = sl.viterbi(U, P)[0] if model.head == "crf" else U.argmax(2)
        zc = sl.viterbi(*constrain(U, P, model.scheme))[0]
        for j, i in enumerate(idx):
            raw[i], con[i] = decode(zr[j], model.scheme), decode(zc[j], model.scheme)
            invalid += not is_valid(zr[j], model.scheme)
    return {"raw": raw, "constrained": con, "invalid": invalid / n}


@lru_cache(maxsize=None)
def f1_decoding(model: Model, test: str, unk_names: bool) -> dict:
    """Thm 3 on a test split: every sentence's span marginals from its constrained chain (Prop 4), the corpus's
    expected span count m, the F1-optimal span sets and their F*, and the candidates (pruned at q ≤ 10⁻⁶, which can
    never clear F*/2) for threshold sweeps."""
    n = len(split(test).toks)
    sents: list = [None] * n
    m = 0.0
    for idx, U, P in chains(model, test, unk_names):
        q = span_marginals(*constrain(U, P, model.scheme), model.scheme)
        keys = list(q)
        Q = np.stack([q[k] for k in keys], 1)
        m += float(Q.sum())
        for j, i in enumerate(idx):
            keep = np.flatnonzero(Q[j] > 1e-6)
            sents[i] = ([keys[k] for k in keep], Q[j, keep], U.shape[1])
    sel, F = f1_decode(sents, m)
    return {"sents": sents, "m": m, "F_star": F, "spans": [{sents[i][0][j] for j in js} for i, js in enumerate(sel)]}


def exposure_f1(test: str, preds: list[set]) -> dict:
    """Strict F1 on the mentions that contain a word the training split never shows ("unseen") and on the rest
    ("seen"), each side filtering gold and predicted spans by their own words. The seen-roster test has some unseen
    mentions too: names and cities the 30 training documents happen not to cover."""
    s, known = split(test), train_words()
    out = {}
    for side in ("seen", "unseen"):
        keep = (lambda x, sp_: any(w not in known for w in x[sp_[0]:sp_[1]]) == (side == "unseen"))
        golds = [{g for g in G if keep(x, g)} for x, G in zip(s.toks, s.spans)]
        pr = [{q for q in P_ if keep(x, q)} for x, P_ in zip(s.toks, preds)]
        out[side] = prf(golds, pr)["f1"]
    return out


def scores(test: str, preds: list[set]) -> dict:
    """Strict micro P/R/F1, per-type F1, F1 under every match scheme, Prop 3's mixture weights, and F1 on seen and
    unseen mentions."""
    golds = list(split(test).spans)
    out = prf(golds, preds)
    out["types"] = {t: prf(golds, preds, "strict", t)["f1"] for t in TYPES}
    out["schemes"] = {h: prf(golds, preds, h)["f1"] for h in MATCHES}
    out["weights"] = micro_mixture(golds, preds)[1]
    out["exposure"] = exposure_f1(test, preds)
    return out


GRID = {
    "words": Model("words", "none", 0.0, "bio", "crf"),
    "words, <unk> dropout": Model("words", "substitute", NAME_DROPOUT, "bio", "crf"),
    "words, deletion dropout": Model("words", "delete", NAME_DROPOUT, "bio", "crf"),
    "words, deletion dropout, BIOES": Model("words", "delete", NAME_DROPOUT, "bioes", "crf"),
    "words + case": Model("words+case", "none", 0.0, "bio", "crf"),
    "encoder": Model("encoder", "none", 0.0, "bio", "crf"),
    "both": Model("both", "none", 0.0, "bio", "crf"),
    "both, deletion dropout": Model("both", "delete", NAME_DROPOUT, "bio", "crf"),
    "softmax on encoder": Model("encoder", "none", 0.0, "bio", "softmax"),
}
HEADLINE = "both, deletion dropout"
SUBSTITUTE = "words, <unk> dropout"


def readings(model: Model) -> tuple[tuple[str, bool], ...]:
    """The (test, unk) pairs a model is scored on: word-only taggers read raw words; encoder taggers are read both with
    the encoder's own vocabulary ("ids": the held-out names were in its pretraining text) and with every word unseen in
    training entering as [UNK] ("unk": the test-time half of [UNK] dropout, and the input of names never seen anywhere)."""
    return tuple((t, u) for t in TESTS for u in ((False, True) if model.feats in ("encoder", "both") else (False,)))


def reading_key(test: str, unk: bool) -> str:
    return test + ("-unk" if unk else "")


@lru_cache(maxsize=None)
def evaluate(model: Model, test: str, unk_names: bool) -> dict:
    pr = predictions(model, test, unk_names)
    return {"raw": scores(test, pr["raw"]), "constrained": scores(test, pr["constrained"]), "invalid": pr["invalid"]}


def sweep_model(how: str, p: float) -> Model:
    return GRID["words"] if p == 0 else Model("words", how, p, "bio", "crf")


@lru_cache(maxsize=1)
def dropout_sweep() -> dict:
    """The word-feature CRF across name-dropout rates, both recipes: strict F1 and precision on each test (constrained
    Viterbi). The dropped sets are nested in p (one uniform per token)."""
    out = {}
    for how in ("substitute", "delete"):
        out[how] = {t: {"f1": [], "p": [], "r": []} for t in TESTS}
        for p in DROPOUT_SWEEP:
            for t in TESTS:
                r = evaluate(sweep_model(how, p), t, False)["constrained"]
                for k in ("f1", "p", "r"):
                    out[how][t][k].append(r[k])
    return out


# --------------------------------------------------------------------------------------------
# 7. Threshold sweeps (Thm 3 on the finance taggers) and the lab's example sentence.
# --------------------------------------------------------------------------------------------

THRESHOLDS = tuple(round(0.02 * k, 2) for k in range(1, 50))
F1_MODELS = ((HEADLINE, True), (SUBSTITUTE, False))  # (grid name, unk) pairs whose span marginals the lab sweeps


def _threshold_sets(sents: list, tau: float) -> list[list[int]]:
    """threshold_decode, faster: two overlapping spans are disjoint events (one span set cannot hold both), so their
    marginals sum to at most 1, and the candidates above τ ≥ 1/2 never overlap: the interval program keeps them all.
    Below 1/2 it runs only on sentences whose candidates above τ overlap."""
    out = []
    for c, q, T in sents:
        js = np.flatnonzero(q > tau)
        iv = sorted((c[j][0], c[j][1], j) for j in js)
        if all(x[1] <= y[0] for x, y in zip(iv, iv[1:])):
            out.append(sorted(js.tolist()))
        else:
            out.append(sorted(interval_dp(c, q - tau, T)))
    return out


@lru_cache(maxsize=None)
def threshold_curve(name: str, test: str, unk: bool) -> dict:
    """Decoding by threshold τ on the span marginals (the interval program with weights q − τ) for every τ in
    THRESHOLDS: the realized strict P/R/F1 against gold, and the model's own expected F1, 2 Σ_Ŝ q / (|Ŝ| + m).
    Thm 3: the expected-F1 curve peaks at τ = F*/2 (among all decoders, not only thresholds)."""
    d = f1_decoding(GRID[name], test, unk)
    golds = split(test).spans
    n_gold = sum(map(len, golds))
    hits = [np.array([c in G for c in cands], bool) for (cands, _, _), G in zip(d["sents"], golds)]
    out = {"tau": list(THRESHOLDS), "f1": [], "p": [], "r": [], "expected": []}
    for tau in THRESHOLDS:
        sel = _threshold_sets(d["sents"], tau)
        k = sum(map(len, sel))
        tp = sum(int(h[js].sum()) for h, js in zip(hits, sel))
        Q = sum(float(q[js].sum()) for (_, q, _), js in zip(d["sents"], sel))
        P_, R_ = (tp / k if k else 0.0), tp / n_gold
        out["p"].append(P_)
        out["r"].append(R_)
        out["f1"].append(2 * tp / (n_gold + k))
        out["expected"].append(2 * Q / (k + d["m"]))
    out.update(F_star=d["F_star"], m=d["m"], f1_decoded=prf(list(golds), d["spans"]),
               viterbi=evaluate(GRID[name], test, unk)["constrained"])
    return out


@lru_cache(maxsize=1)
def example_sentence() -> int:
    """The lab's sentence: the first unseen-test sentence with at least three gold spans, among them a multi-token ORG
    that the word CRF misses entirely (no predicted span overlaps it) and the headline tagger (unk reading) finds
    exactly."""
    s = split("unseen")
    words = predictions(GRID["words"], "unseen", False)["constrained"]
    head = predictions(GRID[HEADLINE], "unseen", True)["constrained"]
    for i, G in enumerate(s.spans):
        orgs = [g for g in G if g[2] == "ORG" and g[1] - g[0] > 1]
        if len(G) >= 3 and any(g in head[i] and not any(p[0] < g[1] and g[0] < p[1] for p in words[i]) for g in orgs):
            return i
    raise AssertionError("no example sentence")


@lru_cache(maxsize=1)
def pretraining_exposure() -> dict:
    """How many of the unseen test's name words (tokens of its ORG, PERSON, and PRODUCT spans that training never
    shows) are in Topic 12's encoder vocabulary, i.e. occurred at least FILING_MIN_COUNT times in its pretraining text."""
    s, known = split("unseen"), train_words()
    voc, _ = cm.filing_vocab()
    names = {w for x, G in zip(s.toks, s.spans) for (a, b, ty) in G if ty in ("ORG", "PERSON", "PRODUCT")
             for w in x[a:b] if w not in known}
    return {"name_words": len(names), "in_vocab": len(names & set(voc))}


# ============================================================================================
# Harness — the experiments' claims.
# ============================================================================================


def test_prop5_a_feature_seen_only_on_entities_argues_against_o():
    """Prop 5: at the penalized optimum, Σ_k W[f, k] = 0 for every feature, and a feature that fires only on tokens
    whose gold tag is in a set A has W[f, k] < 0 for every k ∉ A (so Σ_{k∈A} W[f, k] > 0) — on a toy data set fit to
    gradient 10⁻¹¹ by trust-ncg, and on the substitution-trained finance tagger, whose <unk> weight for O is negative."""
    rng = _rng(3531)
    K, A = 3, (1, 2)
    zs = [rng.integers(0, K, int(rng.integers(1, 6))) for _ in range(40)]
    feats = [[["bias", f"w0={int(rng.integers(0, 4))}"] + (["u"] if z_ in A and rng.random() < 0.5 else []) for z_ in z]
             for z in zs]
    vocab = sl.index_features(feats)
    data = sl.make_data(feats, zs, vocab, K)
    W = sl.crf_unpack(sl.crf_fit(data, 0.1, gtol=1e-11), data.F, K)[0]
    assert np.abs(W.sum(1)).max() < 1e-10
    u = vocab["u"]
    assert W[u, 0] < 0 and W[u, list(A)].sum() > 0
    f = fit(GRID[SUBSTITUTE])
    Wf = sl.crf_unpack(f["theta"], f["F"], f["K"])[0]
    j = f["vocab"]["w0=" + UNK_WORD]
    assert Wf[j, 0] < 0 and Wf[j, 0] == Wf[j].min() and abs(Wf[j].sum()) < 1e-3


def test_the_word_crf_memorizes_names():
    """With word features alone, the CRF tags seen ORG mentions at F1 > 0.9 and unseen ones at F1 0 (it predicts no
    ORG span on the held-out test): word identity explains every training label, so context earns little weight."""
    seen, unseen = (evaluate(GRID["words"], t, False)["constrained"] for t in TESTS)
    assert seen["types"]["ORG"] > 0.9 and unseen["types"]["ORG"] == 0.0
    assert not any(p[2] == "ORG" for P_ in predictions(GRID["words"], "unseen", False)["constrained"] for p in P_)


def test_dropout_recipes():
    """Name dropout at p = 0.5. Substituting <unk> teaches Prop 5's shortcut: precision falls below 0.55 on both tests
    (on the unseen test recall rises from below 0.5 to above 0.9). Deleting the dropped token's identity features
    raises strict F1 on both tests, the unseen one from 0.64 to above 0.88. Along the sweep, substitution's seen-test
    precision falls at every step; deletion's unseen-test F1 rises through p = 0.75 and falls at p = 1, where even the
    closed-class entity words (million, percent, quarter) never show their identity in training."""
    base = {t: evaluate(GRID["words"], t, False)["constrained"] for t in TESTS}
    sub = {t: evaluate(GRID[SUBSTITUTE], t, False)["constrained"] for t in TESTS}
    dele = {t: evaluate(GRID["words, deletion dropout"], t, False)["constrained"] for t in TESTS}
    for t in TESTS:
        assert sub[t]["p"] < 0.55 and dele[t]["f1"] > base[t]["f1"]
    assert base["unseen"]["r"] < 0.5 < 0.9 < sub["unseen"]["r"] and dele["unseen"]["f1"] > 0.88
    sw = dropout_sweep()
    assert all(a > b for a, b in zip(sw["substitute"]["seen"]["p"], sw["substitute"]["seen"]["p"][1:]))
    f = sw["delete"]["unseen"]["f1"]
    assert all(a < b for a, b in zip(f[:3], f[1:4])) and f[4] < f[3]


def test_case_is_most_of_the_signal():
    """The capitalization ablation: one "cap" feature (no dropout) lifts the word CRF's unseen-test F1 from 0.64 to
    above 0.9, past deletion dropout."""
    case = evaluate(GRID["words + case"], "unseen", False)["constrained"]["f1"]
    assert case > 0.9 and case > evaluate(GRID["words, deletion dropout"], "unseen", False)["constrained"]["f1"]


def test_encoder_context():
    """Frozen Topic 12 states carry context: the encoder CRF finds unseen ORGs at F1 > 0.5 where the word CRF finds
    none. Reading unseen words as [UNK] costs the undropped encoder taggers on the unseen test (they lose the
    pretraining exposure to the names), while it is what the [UNK]-dropout headline was trained for: there it scores
    above 0.97 on the unseen test, the best of the grid, and above 0.95 on the seen test."""
    assert evaluate(GRID["encoder"], "unseen", False)["constrained"]["types"]["ORG"] > 0.5
    for name in ("encoder", "both"):
        assert (evaluate(GRID[name], "unseen", True)["constrained"]["f1"]
                < evaluate(GRID[name], "unseen", False)["constrained"]["f1"])
    head = {t: evaluate(GRID[HEADLINE], t, True)["constrained"]["f1"] for t in TESTS}
    assert head["unseen"] > 0.97 and head["seen"] > 0.95
    others = [evaluate(m, "unseen", u)["constrained"]["f1"] for n, m in GRID.items() for t, u in readings(m)
              if t == "unseen" and not (n == HEADLINE and u)]
    assert head["unseen"] > max(others)


def test_crf_head_vs_softmax_head():
    """On frozen states, an independent softmax head decodes invalid BIO sequences (over 5% of sentences, on both
    tests); constraining it (Thm 1) removes them and raises its F1, but the CRF head on the same states still leads by
    more than 0.09 on both tests. The CRF heads decode invalid sequences in under 1% of sentences."""
    for t in TESTS:
        soft = evaluate(GRID["softmax on encoder"], t, False)
        crf = evaluate(GRID["encoder"], t, False)
        assert soft["invalid"] > 0.05 and soft["constrained"]["f1"] > soft["raw"]["f1"]
        assert crf["constrained"]["f1"] - soft["constrained"]["f1"] > 0.09
        for n, m in GRID.items():
            if m.head == "crf":
                assert all(evaluate(m, tt, u)["invalid"] < 0.011 for tt, u in readings(m))


def test_bioes_buys_nothing_here():
    """Prop 2's equality case: the word features see the neighbors, so BIOES is no more expressive than BIO, and with
    twice the tags it does not beat BIO on either test."""
    for t in TESTS:
        assert (evaluate(GRID["words, deletion dropout, BIOES"], t, False)["constrained"]["f1"]
                <= evaluate(GRID["words, deletion dropout"], t, False)["constrained"]["f1"])


def test_overlapping_spans_are_exclusive():
    """The fact behind _threshold_sets: on every finance chain the lab sweeps, two overlapping candidate spans have
    marginals summing to at most 1, and the fast threshold sets equal the interval program's at τ = 0.3 and 0.6."""
    for name, u in F1_MODELS:
        d = f1_decoding(GRID[name], "unseen", u)
        for c, q, T in d["sents"][:400]:
            for i, j in itertools.combinations(range(len(c)), 2):
                if c[i][0] < c[j][1] and c[j][0] < c[i][1]:
                    assert q[i] + q[j] <= 1 + 1e-9
        for tau in (0.3, 0.6):
            assert _threshold_sets(d["sents"], tau) == [sorted(x) for x in threshold_decode(d["sents"], tau)]


def c_argmax(c: dict) -> float:
    """The threshold at which a sweep's realized F1 peaks."""
    return c["tau"][int(np.argmax(c["f1"]))]


def test_thm3_on_the_finance_taggers():
    """F1-optimal decoding helps the miscalibrated substitution tagger by more than 0.09 on both tests and moves the
    headline by less than 0.01. The models' own F* misjudge their realized F1 (the headline's F* is below it), and the
    realized F1 peaks above τ = 1/2 for the substitution tagger and below F*/2 for the headline: Thm 3 is optimal for
    the model's beliefs, not the truth. Each sweep's expected-F1 curve peaks at the grid point nearest
    F*/2, and never exceeds F*."""
    for t in TESTS:
        sub = threshold_curve(SUBSTITUTE, t, False)
        assert sub["f1_decoded"]["f1"] - sub["viterbi"]["f1"] > 0.09
        head = threshold_curve(HEADLINE, t, True)
        assert abs(head["f1_decoded"]["f1"] - head["viterbi"]["f1"]) < 0.01 and head["F_star"] < head["f1_decoded"]["f1"]
        # Calibration shows in where the realized curve peaks: above 1/2 for the overconfident substitution tagger,
        # below F*/2 for the underconfident headline.
        assert c_argmax(sub) > 0.5 > sub["F_star"] / 2 + 0.15 and c_argmax(head) < head["F_star"] / 2
        for c in (sub, head):
            k = int(np.argmax(c["expected"]))
            assert abs(c["tau"][k] - c["F_star"] / 2) <= 0.01 + 1e-12 and max(c["expected"]) <= c["F_star"] + 1e-12


def test_prop3_on_the_finance_taggers():
    """Prop 3 on every grid tagger and reading: micro F1 equals the prediction-weighted mixture of per-type F1s, and
    F1 never falls as the match relaxes."""
    for m in GRID.values():
        for t, u in readings(m):
            golds, preds = list(split(t).spans), predictions(m, t, u)["constrained"]
            mix, _ = micro_mixture(golds, preds)
            r = evaluate(m, t, u)["constrained"]
            assert abs(r["f1"] - mix) < 1e-12
            sc = r["schemes"]
            assert sc["strict"] <= sc["boundary"] <= sc["partial"] and sc["strict"] <= sc["type"] <= sc["partial"]


def test_held_out_design():
    """The split: every ORG, PERSON, and PRODUCT mention of the held-out test contains a word training never shows;
    its other mentions do only for cities and regions the 30 training documents miss (GPE). Most of those name words
    were in the encoder's pretraining vocabulary. The example sentence follows its rule."""
    s, known = split("unseen"), train_words()
    for x, G in zip(s.toks, s.spans):
        for a, b, ty in G:
            if ty in ("ORG", "PERSON", "PRODUCT"):
                assert any(w not in known for w in x[a:b])
            elif ty != "GPE":
                assert all(w in known for w in x[a:b])
    e = pretraining_exposure()
    assert e["in_vocab"] > e["name_words"] / 2
    assert example_sentence() >= 0


def test_corpus_fingerprint():
    """The shared corpus generator is the one every topic pins (numpy 2.5.4)."""
    assert fc.CORPUS_FINGERPRINT == "b40dd72b3eeba520"


# ============================================================================================
# Summary and the lab's payload.
# ============================================================================================

VIZ_JSON = _HERE.parent.parent / "src" / "components" / "viz" / "data" / "named-entity-lab.json"


def summary() -> dict:
    out = {}
    for name, m in GRID.items():
        for t, u in readings(m):
            r = evaluate(m, t, u)
            c = r["constrained"]
            out[f"{name} [{reading_key(t, u)}]: F1 P R | ORG | unseen-mention F1"] = (
                f"{c['f1']:.6f} {c['p']:.6f} {c['r']:.6f} | {c['types']['ORG']:.6f} | {c['exposure']['unseen']:.6f}")
            if m.head == "softmax" or r["invalid"] > 0:
                out[f"{name} [{reading_key(t, u)}]: raw F1, invalid share"] = f"{r['raw']['f1']:.6f} {r['invalid']:.6f}"
    sw = dropout_sweep()
    for how in sw:
        for t in TESTS:
            out[f"sweep {how} [{t}] F1 by p"] = " ".join(f"{v:.6f}" for v in sw[how][t]["f1"])
            out[f"sweep {how} [{t}] P by p"] = " ".join(f"{v:.6f}" for v in sw[how][t]["p"])
    for name, u in F1_MODELS:
        for t in TESTS:
            c = threshold_curve(name, t, u)
            out[f"Thm 3 {name} [{reading_key(t, u)}]: F* Viterbi F1-decoded"] = (
                f"{c['F_star']:.6f} {c['viterbi']['f1']:.6f} {c['f1_decoded']['f1']:.6f}")
    ce = prop2_counterexample()
    out["Prop 2 gold probabilities bio | bioes | bio+left"] = " | ".join(
        " ".join(f"{v:.6f}" for v in ce[k]) for k in ("bio", "bioes", "bio+left"))
    e = pretraining_exposure()
    out["held-out name words, in encoder vocabulary"] = f"{e['name_words']} {e['in_vocab']}"
    for t in ("train", *TESTS):
        s = split(t)
        out[f"{t}: sentences, tokens, spans"] = f"{len(s.toks)} {sum(map(len, s.toks))} {sum(map(len, s.spans))}"
    return out


def _spans_json(S) -> list:
    return [[int(a), int(b), t] for a, b, t in sorted(S)]


def viz_payload() -> dict:
    """Every number the lab shows, rounded to the page's precision (3 decimals)."""
    s = split("unseen")
    i = example_sentence()
    x, G = s.toks[i], s.spans[i]
    one = ("ORG",)
    encodings = {
        "tokens": list(x), "spans": _spans_json(G),
        "tags": {sc: [tagset(sc)[k] for k in encode(G, len(x), sc)] for sc in SCHEMES},
        "masks": {sc: {"tags": list(tagset(sc, one)), **{k: v.astype(int).tolist() for k, v in
                                                          zip(("start", "trans", "end"), masks(sc, one))}}
                  for sc in SCHEMES},
        "counts": [{"T": T, "sets": count_span_sets(T, len(TYPES)), "bio": len(tagset("bio")) ** T,
                    "bioes": len(tagset("bioes")) ** T} for T in range(1, 7)],
        "prop2": {"data": [[list(w), _spans_json(S_)] for w, S_ in PROP2_DATA],
                  **{k: _r(v) for k, v in prop2_counterexample().items()}},
    }
    models = {}
    for name, m in GRID.items():
        rd = {}
        for t, u in readings(m):
            r = evaluate(m, t, u)
            c = r["constrained"]
            rd[reading_key(t, u)] = {
                "f1": _r(c["f1"]), "p": _r(c["p"]), "r": _r(c["r"]), "raw_f1": _r(r["raw"]["f1"]),
                "invalid": _r(r["invalid"]), "schemes": {h: _r(v) for h, v in c["schemes"].items()},
                "types": {k: _r(v) for k, v in c["types"].items()},
                "exposure": {k: _r(v) for k, v in c["exposure"].items()},
                "example": _spans_json(predictions(m, t, u)["constrained"][i]) if t == "unseen" else None}
        models[name] = {"feats": m.feats, "dropout": m.dropout, "p": m.p, "scheme": m.scheme, "head": m.head,
                        "readings": rd}
    sw = dropout_sweep()
    sweep = {"p": list(DROPOUT_SWEEP), **{how: {t: {k: _r(v) for k, v in sw[how][t].items()} for t in TESTS}
                                          for how in sw}}
    thresholds = {}
    for name, u in F1_MODELS:
        thresholds[name] = {"unk": u}
        for t in TESTS:
            c = threshold_curve(name, t, u)
            d = f1_decoding(GRID[name], t, u)
            ex = None
            if t == "unseen":
                cands, q, _ = d["sents"][i]
                ex = [[int(a), int(b), ty, _r(float(v))] for (a, b, ty), v in sorted(zip(cands, q)) if v >= 0.01]
            thresholds[name][t] = {"tau": c["tau"], "f1": _r(c["f1"]), "p": _r(c["p"]), "r": _r(c["r"]),
                                   "expected": _r(c["expected"]), "F_star": _r(c["F_star"]),
                                   "viterbi_f1": _r(c["viterbi"]["f1"]), "f1_decoded": _r(c["f1_decoded"]["f1"]),
                                   "example": ex}
    sizes = {t: {"sentences": len(split(t).toks), "spans": sum(map(len, split(t).spans))} for t in ("train", *TESTS)}
    return {"encodings": encodings, "models": models, "headline": HEADLINE, "sweep": sweep, "thresholds": thresholds,
            "exposure": pretraining_exposure(), "held_out": list(HELD_ROSTER), "n_docs": N_DOCS, "sizes": sizes}


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    assert json.loads(VIZ_JSON.read_text()) == json.loads(json.dumps(viz_payload())), VIZ_JSON


def _run_harness() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    print("\nSummary:")
    for k, v in summary().items():
        print(f"  {k:72} {v:.6f}" if isinstance(v, float) else f"  {k:72} {v}")
    return 1 if failed else 0


if __name__ == "__main__" and "--write-payload" in sys.argv:
    VIZ_JSON.write_text(json.dumps(viz_payload(), separators=(",", ":")))
    print("wrote", VIZ_JSON, VIZ_JSON.stat().st_size, "bytes")
elif __name__ == "__main__":
    sys.exit(_run_harness())
