"""Neural language models — the reference implementation for formalNLP's fifth topic.

Every claim the topic makes has an executable check here, and every number its D3 lab shows comes
from ``viz_payload``. The sections follow the page:

1. From counts to coordinates: Bengio's NPLM is softmax regression on learned context features. The
   softmax is positive everywhere, and every log-probability is Lipschitz in the context's embedding, so
   nearby contexts get nearby predictions: smoothing by geometry rather than by counts.
2. Training is maximum likelihood: the softmax is the categorical exponential family (its Jacobian is the
   covariance of the one-hot vector), the score is Σ a(p − y)ᵀ, and a stationary point matches every word's
   mean predicted probability to its training frequency — so a word never seen as a target has no finite
   MLE. The loss is convex in the output layer and not in the whole network.
3. Backpropagation: reverse accumulation on a DAG computes every adjoint in one sweep (checked against
   complex-step derivatives on random graphs and on both networks).
4. Recurrent networks: a bounded-state softmax RNN is a proper distribution on finite sentences; BPTT is
   reverse accumulation on the unrolled graph; ‖∂h_t/∂h_k‖ ≤ Π ‖diag(1 − h_j²) W_hᵀ‖ ≤ σ_max(W_h)^{t−k};
   an LSTM's cell path contributes Π diag(f_j), so the forget bias sets how far gradients reach.
5. The softmax bottleneck: a softmax over a d-dimensional state has a log-probability matrix of rank at
   most d + 2, on any set of contexts; a mixture of softmaxes escapes the bound.
6. The honest comparison: each model as an exact arithmetic code (Topic 4), in bits per character; dynamic
   evaluation is prequential coding, and the decoder replays the updates.
7. Finance case study: on EDGAR the neural models predict worse than uniform, dynamic evaluation recovers
   much of the loss, and the spelling of out-of-vocabulary words alone costs more than lzma's whole code.

Cross-checks: every hand-written gradient against complex-step derivatives (exact to rounding) and central
differences; softmax against ``scipy.special``; Topic 4's arithmetic decoder reproduces every coded text.

Run (executes the verification harness):
    uv run --with numpy==2.5.4 --with scipy==1.18.1 python notebooks/neural-language-models/neural_language_models.py
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, special

_HERE = Path(__file__).resolve().parent
for _dep in ("finance-corpus", "zipf-heaps-vocabulary-statistics", "subword-tokenization-bpe-unigram",
             "n-gram-models-smoothing", "perplexity-cross-entropy"):
    sys.path.insert(0, str(_HERE.parent / _dep))
import finance_corpus as fc  # noqa: E402
import n_gram_models_smoothing as ng  # noqa: E402
import perplexity_cross_entropy as pce  # noqa: E402

# ONE stream per random-instance claim and per trained model, each from default_rng([SEED, k]) (STARTER-PROMPT A1).
SEED = 20261023

BOS, EOS, UNK = ng.BOS, ng.EOS, ng.UNK
ORDER = ng.ORDER  # the NPLM sees the same two-word context as Topic 3's trigram models
LN2 = math.log(2)
TESTS = ("held", "edgar")
NGram = tuple[str, ...]
Params = dict[str, np.ndarray]

ALPHABET = pce.word_alphabet()  # Topic 3's V, sorted; the models predict these 3,427 symbols
INDEX = {w: i for i, w in enumerate(ALPHABET)}
V = len(ALPHABET)
EOS_ID, UNK_ID = INDEX[EOS], INDEX[UNK]
BOS_ID = V  # <s> is an input only: the embedding table has one extra row for it

DEV_DOCS, DEV_SEED = 40, 2  # a fresh generator run for model selection, like held's generate_corpus(100, 1)
DROP = 0.5  # each training token of a once-seen word becomes <unk> with this probability, redrawn every epoch

NPLM_CFG = {"d": 64, "h": 128, "epochs": 4, "batch": 256, "lr": 2e-3, "wd": 1e-4, "clip": 0.0}
RNN_CFG = {"d": 64, "h": 128, "epochs": 4, "batch": 32, "lr": 3e-3, "wd": 1e-4, "clip": 5.0}
DECAYED = frozenset({"E", "W1", "W2", "Wx", "Wh", "Wo"})  # L2 on the weight matrices, never on the biases
DYN_LRS = (0.001, 0.003, 0.01, 0.03)  # dynamic evaluation's SGD step sizes, chosen on dev
MIX_LAMBDAS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))  # weight on the neural model in model ⊕ KN, on dev


# --------------------------------------------------------------------------------------------
# Data: Topic 3's splits, a dev split, and the word ids the networks read.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def corpus() -> dict[str, tuple[NGram, ...]]:
    """Topic 3's train / held / edgar, plus dev: generate_corpus(DEV_DOCS, DEV_SEED), used only to choose the
    epoch, the mixture weight, and the dynamic-evaluation step. Held-out text never tunes anything."""
    dev = tuple(s for d in fc.generate_corpus(DEV_DOCS, DEV_SEED) for s in ng.split_sentences(d.text))
    return {**ng.corpus(), "dev": dev}


@lru_cache(maxsize=1)
def hapaxes() -> frozenset[str]:
    return frozenset(g[0] for g, n in ng.train_counts()[1].items() if n == 1 and g[0] != EOS)


def sentence_ids(s: NGram) -> list[int]:
    """Target ids of one sentence: each word (OOV → <unk>) and the final </s>."""
    return [INDEX.get(w, UNK_ID) for w in s] + [EOS_ID]


@lru_cache(maxsize=None)
def ids_of(test: str) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sentence_ids(s)) for s in corpus()[test])


@lru_cache(maxsize=1)
def _train_stream() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training text as one array [<s>, <s>, w₁ … wₙ, </s>] per sentence, the positions of its targets,
    and a mask of the tokens of once-seen words (the ones word dropout may hide)."""
    toks, pos, hap = [], [], []
    H = hapaxes()
    for s in corpus()["train"]:
        base = len(toks)
        toks += [BOS_ID, BOS_ID] + sentence_ids(s)
        pos += range(base + 2, len(toks))
        hap += [False, False] + [w in H for w in s] + [False]
    return np.array(toks), np.array(pos), np.array(hap)


def dropped_train_events(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One epoch's NPLM events: every token of a once-seen word becomes <unk> with probability DROP, in the
    context and as a target alike — Good–Turing's stand-in for the unseen (Topic 3), as word dropout."""
    toks, pos, hap = _train_stream()
    t = toks.copy()
    t[hap & (rng.random(len(t)) < DROP)] = UNK_ID
    return np.stack([t[pos - 2], t[pos - 1]], axis=1), t[pos]


def dropped_train_sentences(rng: np.random.Generator) -> list[list[int]]:
    """One epoch's RNN sentences under the same word dropout."""
    toks, pos, hap = _train_stream()
    t = toks.copy()
    t[hap & (rng.random(len(t)) < DROP)] = UNK_ID
    bounds = np.flatnonzero(t[pos] == EOS_ID) + 1
    return [list(x) for x in np.split(t[pos], bounds[:-1])]


@lru_cache(maxsize=None)
def events_of(test: str) -> tuple[np.ndarray, np.ndarray]:
    """(context ids, target ids) of a test text in ng.events order: two <s> pads, OOV → <unk>."""
    rows = [[INDEX.get(w, BOS_ID if w == BOS else UNK_ID) for w in g]
            for g in ng.events(ng.with_unk(corpus()[test]), ORDER)]
    a = np.array(rows)
    return a[:, :2], a[:, 2]


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax; complex inputs pass through unchanged in form (the complex step needs that)."""
    z = z - z.real.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


# --------------------------------------------------------------------------------------------
# 1. From counts to coordinates: the NPLM.
# --------------------------------------------------------------------------------------------


def init_nplm(rng: np.random.Generator, d: int, h: int, vocab: int = V, ctx: int = ORDER - 1) -> Params:
    """Def 1: E ∈ R^{(V+1)×d} (row V is <s>), a = tanh(x W₁ + b₁) on the concatenated context x ∈ R^{ctx·d},
    z = a W₂ + b₂, p = softmax(z). Bengio et al.'s direct connections are omitted, so Thm 6 applies with
    state dimension h."""
    return {"E": rng.normal(0, 0.1, (vocab + 1, d)), "W1": rng.normal(0, 1 / math.sqrt(ctx * d), (ctx * d, h)),
            "b1": np.zeros(h), "W2": rng.normal(0, 1 / math.sqrt(h), (h, vocab)), "b2": np.zeros(vocab)}


def _nplm_forward(P: Params, X: np.ndarray):
    x = P["E"][X].reshape(len(X), -1)
    a = np.tanh(x @ P["W1"] + P["b1"])
    return x, a, log_softmax(a @ P["W2"] + P["b2"])


def nplm_loss(P: Params, X: np.ndarray, Y: np.ndarray):
    """Mean cross-entropy (nats per event); complex parameters give a complex loss (for the complex step)."""
    _, _, lp = _nplm_forward(P, X)
    return -lp[np.arange(len(Y)), Y].mean()


def nplm_loss_grad(P: Params, X: np.ndarray, Y: np.ndarray) -> tuple[float, Params]:
    """Mean cross-entropy of the NPLM on events (X: n×2 context ids, Y: n target ids) and its gradient by
    hand-written reverse accumulation: ḡ_z = (p − y)/n (Thm 2) → W₂, b₂ → a → (1 − a²) → W₁, b₁ → the
    embedding rows of each event (np.add.at: only rows that occur get gradient)."""
    n = len(Y)
    x, a, lp = _nplm_forward(P, X)
    gz = np.exp(lp)
    gz[np.arange(n), Y] -= 1
    gz /= n
    ga = (gz @ P["W2"].T) * (1 - a * a)
    gE = np.zeros_like(P["E"])
    np.add.at(gE, X, (ga @ P["W1"].T).reshape(n, X.shape[1], -1))
    G = {"E": gE, "W1": x.T @ ga, "b1": ga.sum(0), "W2": a.T @ gz, "b2": gz.sum(0)}
    return float(-lp[np.arange(n), Y].mean()), G


def nplm_target_logprobs(P: Params, X: np.ndarray, Y: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = [_nplm_forward(P, X[i : i + chunk])[2][np.arange(len(Y[i : i + chunk])), Y[i : i + chunk]]
           for i in range(0, len(Y), chunk)]
    return np.concatenate(out)


def nplm_lipschitz(P: Params) -> float:
    """Thm 1's constant: √2 ‖W₂‖₂ ‖W₁‖₂ bounds |log p(w | x) − log p(w | x′)| / ‖x − x′‖₂ for every word w."""
    return math.sqrt(2) * np.linalg.norm(P["W2"], 2) * np.linalg.norm(P["W1"], 2)


def context_points(P: Params, X: np.ndarray) -> np.ndarray:
    """Each context as a point x(u) ∈ R^{2d}: its words' embeddings, concatenated."""
    return P["E"][X].reshape(len(X), -1)


def logprobs_at_points(P: Params, x: np.ndarray) -> np.ndarray:
    return log_softmax(np.tanh(x @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"])


# --------------------------------------------------------------------------------------------
# Training: Adam with dev-set model selection.
# --------------------------------------------------------------------------------------------

ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def adam_fit(P: Params, loss_grad, epoch_batches, cfg: dict, rng: np.random.Generator, monitors: dict):
    """Adam (Kingma and Ba 2015) with L2 weight decay on the weight matrices, optional global-norm clipping, and
    selection of the epoch with the lowest dev loss. epoch_batches(rng) yields one epoch's argument tuples for
    loss_grad; monitors maps a name to a function of the parameters, evaluated after every epoch."""
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    step, hist, best = 0, [], None
    for ep in range(cfg["epochs"]):
        for batch in epoch_batches(rng):
            _, G = loss_grad(P, *batch)
            if cfg["clip"]:
                norm = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
                if norm > cfg["clip"]:
                    G = {k: g * (cfg["clip"] / norm) for k, g in G.items()}
            step += 1
            c1, c2 = 1 - ADAM_B1 ** step, 1 - ADAM_B2 ** step
            for k in P:
                g = G[k] + cfg["wd"] * P[k] if k in DECAYED else G[k]
                m[k] *= ADAM_B1
                m[k] += (1 - ADAM_B1) * g
                s[k] *= ADAM_B2
                s[k] += (1 - ADAM_B2) * g * g
                P[k] -= cfg["lr"] * (m[k] / c1) / (np.sqrt(s[k] / c2) + ADAM_EPS)
        row = {name: float(fn(P)) for name, fn in monitors.items()}
        hist.append(row)
        if best is None or row["dev"] < best[1]["dev"]:
            best = (ep, row, {k: v.copy() for k, v in P.items()})
    return best[2], best[0], hist


@dataclass(frozen=True)
class Trained:
    params: Params
    epoch: int  # the dev-selected epoch (0-based)
    history: tuple[dict, ...]  # dev and EDGAR nats per token after each epoch


@lru_cache(maxsize=1)
def trained_nplm() -> Trained:
    cfg, rng = NPLM_CFG, np.random.default_rng([SEED, 100])
    P = init_nplm(rng, cfg["d"], cfg["h"])

    def batches(r):
        X, Y = dropped_train_events(r)
        perm = r.permutation(len(Y))
        for i in range(0, len(Y), cfg["batch"]):
            j = perm[i : i + cfg["batch"]]
            yield X[j], Y[j]

    mon = {t: (lambda P, t=t: -nplm_target_logprobs(P, *events_of(t)).mean()) for t in ("dev", "edgar")}
    best, ep, hist = adam_fit(P, nplm_loss_grad, batches, cfg, rng, mon)
    return Trained(best, ep, tuple(hist))


@lru_cache(maxsize=1)
def nplm_without_dropout() -> Trained:
    """Cor 1 on the corpus: the same NPLM trained ONE epoch on the raw training text, where <unk> never occurs
    as a target. (One epoch suffices to show the collapse, and keeps the harness under a minute.)"""
    cfg, rng = {**NPLM_CFG, "epochs": 1}, np.random.default_rng([SEED, 102])
    P = init_nplm(rng, cfg["d"], cfg["h"])
    toks, pos, _ = _train_stream()
    X, Y = np.stack([toks[pos - 2], toks[pos - 1]], axis=1), toks[pos]

    def batches(r):
        perm = r.permutation(len(Y))
        for i in range(0, len(Y), cfg["batch"]):
            yield X[perm[i : i + cfg["batch"]]], Y[perm[i : i + cfg["batch"]]]

    mon = {t: (lambda P, t=t: -nplm_target_logprobs(P, *events_of(t)).mean()) for t in ("dev", "edgar")}
    best, ep, hist = adam_fit(P, nplm_loss_grad, batches, cfg, rng, mon)
    return Trained(best, ep, tuple(hist))


@lru_cache(maxsize=None)
def unk_cost(name: str, test: str) -> dict[str, float]:
    """Mean predicted p(<unk>) over a text's contexts and the mean cost, in nats, of its <unk> events."""
    P = (nplm_without_dropout() if name == "nplm-nodrop" else trained_nplm()).params
    X, Y = events_of(test)
    lp = np.concatenate([_nplm_forward(P, X[i : i + 4096])[2][:, UNK_ID] for i in range(0, len(Y), 4096)])
    return {"mean_p_unk": float(np.exp(lp).mean()), "unk_nats": float(-lp[Y == UNK_ID].mean()),
            "nats": float(-nplm_target_logprobs(P, X, Y).mean())}


# --------------------------------------------------------------------------------------------
# 2. Training is maximum likelihood: the softmax, its score, and convexity.
# --------------------------------------------------------------------------------------------


def softmax_jacobian(z: np.ndarray) -> np.ndarray:
    """Prop 2: ∂p/∂z = diag(p) − ppᵀ, the covariance of the one-hot vector e_Y under Y ~ p (the log-partition's
    Hessian, formalStatistics exponential-families Thm 7.1)."""
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)


def output_layer_loss(theta: np.ndarray, A: np.ndarray, Y: np.ndarray, K: int):
    """Softmax regression on fixed features A (n × h): mean cross-entropy of θ = (W ∈ R^{h×K}, b ∈ R^K)."""
    h = A.shape[1]
    return -log_softmax(A @ theta[: h * K].reshape(h, K) + theta[h * K :])[np.arange(len(Y)), Y].mean()


def output_layer_loss_grad(theta: np.ndarray, A: np.ndarray, Y: np.ndarray, K: int):
    """The output-layer loss and its gradient (Aᵀ(P − Y), 𝟙ᵀ(P − Y))/n — Thm 2's score identity."""
    n, h = A.shape
    W, b = theta[: h * K].reshape(h, K), theta[h * K :]
    lp = log_softmax(A @ W + b)
    R = np.exp(lp)
    R[np.arange(n), Y] -= 1
    return float(-lp[np.arange(n), Y].mean()), np.concatenate([(A.T @ R).ravel(), R.sum(0)]) / n


def fit_output_layer(A: np.ndarray, Y: np.ndarray, K: int) -> np.ndarray:
    """A stationary point of the (convex) output-layer objective, by L-BFGS to a gradient of ~1e-11."""
    th0 = np.zeros(A.shape[1] * K + K)
    res = optimize.minimize(output_layer_loss_grad, th0, args=(A, Y, K), jac=True, method="L-BFGS-B",
                            options={"gtol": 1e-12, "ftol": 1e-16, "maxiter": 5000})
    return res.x


def permute_hidden(P: Params, perm: np.ndarray) -> Params:
    """The same function with its hidden units relabeled: Prop 3's symmetry."""
    return {**P, "W1": P["W1"][:, perm], "b1": P["b1"][perm], "W2": P["W2"][perm, :]}


@lru_cache(maxsize=1)
def nonconvexity() -> dict[str, float]:
    """Prop 3 on the trained NPLM: the loss on 4,096 dev events at θ, at θ with its hidden units in reverse
    order, and at their midpoint."""
    P = trained_nplm().params
    X, Y = events_of("dev")
    X, Y = X[:4096], Y[:4096]
    Q = permute_hidden(P, np.arange(NPLM_CFG["h"])[::-1])
    mid = {k: (P[k] + Q[k]) / 2 for k in P}
    return {"loss": float(nplm_loss(P, X, Y)), "loss_permuted": float(nplm_loss(Q, X, Y)),
            "loss_midpoint": float(nplm_loss(mid, X, Y))}


@lru_cache(maxsize=1)
def unk_moment() -> dict[str, float]:
    """Thm 2 on the corpus: the trained NPLM's mean predicted p(<unk>) over the training contexts (under one
    fresh draw of the word dropout) against the <unk> frequency that draw puts in the targets, and the
    expected rate DROP · n₁ / N."""
    P = trained_nplm().params
    X, Y = dropped_train_events(np.random.default_rng([SEED, 101]))
    p_unk = np.concatenate([np.exp(_nplm_forward(P, X[i : i + 4096])[2][:, UNK_ID]) for i in range(0, len(Y), 4096)])
    return {"mean_p_unk": float(p_unk.mean()), "target_rate": float((Y == UNK_ID).mean()),
            "expected_rate": DROP * len(hapaxes()) / len(Y)}


# --------------------------------------------------------------------------------------------
# 3. Backpropagation: reverse accumulation on a computational graph.
# --------------------------------------------------------------------------------------------


@dataclass
class Node:
    """One primitive of a computational graph (Def 2). args are the parents' values; const is the fixed matrix
    of a "matmul" node. vjp(ḡ) returns one cotangent per parent, J_{node, parent}ᵀ ḡ."""

    op: str
    parents: tuple[int, ...]
    value: np.ndarray
    args: tuple
    const: np.ndarray | None = None

    def vjp(self, g: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.op == "matmul":
            return (g @ self.const.T,)
        if self.op == "tanh":
            return (g * (1 - self.value**2),)
        if self.op == "mul":
            a, b = self.args
            return (g * b, g * a)
        if self.op == "add":
            return (g, g)
        if self.op == "lse":
            return (g * np.exp(self.args[0] - self.value),)
        return ()


@dataclass
class Graph:
    nodes: list[Node]


Program = list[tuple[str, tuple[int, ...], np.ndarray | None]]


def evaluate(program: Program, inputs: list[np.ndarray]) -> Graph:
    """Run a program — a list of (op, parent indices, constant) after the input nodes — on real or complex inputs."""
    nodes = [Node("input", (), x, ()) for x in inputs]
    for op, par, const in program:
        args = tuple(nodes[p].value for p in par)
        if op == "matmul":
            v = args[0] @ const
        elif op == "tanh":
            v = np.tanh(args[0])
        elif op == "mul":
            v = args[0] * args[1]
        elif op == "add":
            v = args[0] + args[1]
        else:  # "lse": log Σ exp, a scalar
            mx = args[0].real.max()
            v = mx + np.log(np.exp(args[0] - mx).sum())
        nodes.append(Node(op, par, v, args, const))
    return Graph(nodes)


def reverse_accumulate(graph: Graph, out: int) -> list[np.ndarray]:
    """Thm 3. Adjoints v̄ = ∂out/∂v for every node, in ONE sweep over the nodes in reverse topological order.
    graph.nodes[i].parents are indices < i; graph.nodes[i].vjp(ḡ) returns one cotangent per parent,
    J_{i,p}ᵀ ḡ. The scalar output's adjoint is seeded with 1; a node with fan-out sums its children's
    contributions."""
    adj = [np.zeros_like(n.value) for n in graph.nodes]
    adj[out] = np.ones_like(graph.nodes[out].value)
    for i in range(out, -1, -1):
        for p, g in zip(graph.nodes[i].parents, graph.nodes[i].vjp(adj[i])):
            adj[p] = adj[p] + g
    return adj


def random_program(rng: np.random.Generator, n_inputs: int, k: int, n_ops: int) -> Program:
    """A random DAG over k-vectors: matmul by a random matrix, tanh, elementwise product and sum, with parents
    drawn (biased toward recent nodes) so that fan-out is common; the output is log Σ exp of the last node."""
    prog: Program = []
    for i in range(n_inputs, n_inputs + n_ops):
        op = str(rng.choice(["matmul", "tanh", "mul", "add"]))
        pick = lambda: int(min(i - 1, np.floor(i * rng.beta(2.0, 1.0))))  # noqa: E731
        par = (pick(), pick()) if op in ("mul", "add") else (pick(),)
        prog.append((op, par, rng.normal(0, 1 / math.sqrt(k), (k, k)) if op == "matmul" else None))
    prog.append(("lse", (n_inputs + n_ops - 1,), None))
    return prog


def complex_step_grad(f, theta: np.ndarray, h: float = 1e-30) -> np.ndarray:
    """∂f/∂θ by the complex step (Squire and Trapp 1998): f(θ + ih eᵢ) = f(θ) + ih ∂ᵢf + O(h²), so Im f / h is
    the derivative with no subtractive cancellation — exact to rounding for real-analytic f."""
    g = np.zeros(theta.shape)
    for i in range(theta.size):
        t = theta.astype(complex)
        t.flat[i] += 1j * h
        g.flat[i] = np.imag(f(t)) / h
    return g


def central_diff_grad(f, theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros(theta.shape)
    for i in range(theta.size):
        tp, tm = theta.astype(float), theta.astype(float)
        tp.flat[i] += eps
        tm.flat[i] -= eps
        g.flat[i] = (float(np.real(f(tp))) - float(np.real(f(tm)))) / (2 * eps)
    return g


def param_grad(method, loss, P: Params) -> Params:
    """complex_step_grad or central_diff_grad applied to each parameter array of loss(P) in turn."""
    out = {}
    for k in P:
        out[k] = method(lambda t, k=k: loss({**P, k: t}), P[k])
    return out


# --------------------------------------------------------------------------------------------
# 4. Recurrent networks: the Elman RNN, BPTT, tightness, Jacobian products, the LSTM's cell path.
# --------------------------------------------------------------------------------------------


def init_rnn(rng: np.random.Generator, d: int, h: int, vocab: int = V) -> Params:
    """Def 3: h_t = tanh(x_t W_x + h_{t−1} W_h + b) from h₀ = 0, with x_t the embedding of input t (input 0 is
    <s>, row V of E), and p(· | prefix) = softmax(h_t W_o + b_o)."""
    return {"E": rng.normal(0, 0.1, (vocab + 1, d)), "Wx": rng.normal(0, 1 / math.sqrt(d), (d, h)),
            "Wh": rng.normal(0, 1 / math.sqrt(h), (h, h)), "b": np.zeros(h),
            "Wo": rng.normal(0, 1 / math.sqrt(h), (h, vocab)), "bo": np.zeros(vocab)}


def _pad(batch, bos: int = BOS_ID, eos: int = EOS_ID):
    """Targets Y (T × n, −1 past each sentence's end), inputs X (<s>, then the previous target), and the mask."""
    n, T = len(batch), max(map(len, batch))
    Y = np.full((T, n), -1)
    for j, s in enumerate(batch):
        Y[: len(s), j] = s
    X = np.vstack([np.full((1, n), bos), np.where(Y[:-1] < 0, eos, Y[:-1])])
    return X, Y, Y >= 0


def _rnn_forward(P: Params, batch):
    X, Y, M = _pad(batch, bos=len(P["E"]) - 1)
    xs = P["E"][X]
    h = np.zeros((X.shape[1], P["Wh"].shape[0]), dtype=xs.dtype)
    hs = []
    for t in range(len(X)):
        h = np.tanh(xs[t] @ P["Wx"] + h @ P["Wh"] + P["b"])
        hs.append(h)
    hs = np.stack(hs)
    return X, Y, M, xs, hs, log_softmax(hs @ P["Wo"] + P["bo"])


def _token_logprobs(Y, M, lp) -> np.ndarray:
    return np.take_along_axis(lp, np.where(M, Y, 0)[..., None], -1)[..., 0] * M


def rnn_loss(P: Params, batch):
    _, Y, M, _, _, lp = _rnn_forward(P, batch)
    return -_token_logprobs(Y, M, lp).sum() / M.sum()


def _rnn_backward(P: Params, X, Y, M, xs, hs, lp) -> Params:
    T, n = Y.shape
    ti, ji = np.nonzero(M)
    gz = np.exp(lp)
    gz[ti, ji, Y[ti, ji]] -= 1
    gz *= M[..., None] / M.sum()
    G = {"Wo": hs.reshape(-1, hs.shape[-1]).T @ gz.reshape(-1, gz.shape[-1]), "bo": gz.sum((0, 1))}
    dh_out = gz @ P["Wo"].T
    gWx, gWh, gb = np.zeros_like(P["Wx"]), np.zeros_like(P["Wh"]), np.zeros_like(P["b"])
    gxs, dh = np.empty_like(xs), np.zeros_like(hs[0])
    for t in range(T - 1, -1, -1):
        dz = (dh + dh_out[t]) * (1 - hs[t] ** 2)
        gxs[t] = dz @ P["Wx"].T
        gWx += xs[t].T @ dz
        if t:
            gWh += hs[t - 1].T @ dz
        gb += dz.sum(0)
        dh = dz @ P["Wh"].T
    gE = np.zeros_like(P["E"])
    np.add.at(gE, X, gxs)
    return {"E": gE, "Wx": gWx, "Wh": gWh, "b": gb, **G}


def rnn_loss_grad(P: Params, batch: list[list[int]]) -> tuple[float, Params]:
    """BPTT (Cor 3). Mean cross-entropy over a padded batch of sentences (ids ending in </s>; input 0 is <s>)
    under h_t = tanh(x_t W_x + h_{t−1} W_h + b), p_t = softmax(h_t W_o + b_o). Padded positions carry no loss.
    Tied weights: each gradient is the sum over time steps."""
    X, Y, M, xs, hs, lp = _rnn_forward(P, batch)
    return float(-_token_logprobs(Y, M, lp).sum() / M.sum()), _rnn_backward(P, X, Y, M, xs, hs, lp)


def rnn_sentence_logprobs(P: Params, sents, batch: int = 64) -> list[np.ndarray]:
    """log p of every target of every sentence, batched by length (sorting cuts the padding)."""
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    out: list[np.ndarray] = [None] * len(sents)  # type: ignore[list-item]
    for i in range(0, len(order), batch):
        idx = order[i : i + batch]
        _, Y, M, _, _, lp = _rnn_forward(P, [list(sents[j]) for j in idx])
        tok = _token_logprobs(Y, M, lp)
        for c, j in enumerate(idx):
            out[j] = tok[: len(sents[j]), c]
    return out


@lru_cache(maxsize=1)
def trained_rnn() -> Trained:
    cfg, rng = RNN_CFG, np.random.default_rng([SEED, 200])
    P = init_rnn(rng, cfg["d"], cfg["h"])

    def batches(r):
        S = dropped_train_sentences(r)
        perm = r.permutation(len(S))
        for i in range(0, len(S), cfg["batch"]):
            yield ([S[j] for j in perm[i : i + cfg["batch"]]],)

    mon = {t: (lambda P, t=t: -np.concatenate(rnn_sentence_logprobs(P, ids_of(t))).mean()) for t in ("dev", "edgar")}
    best, ep, hist = adam_fit(P, rnn_loss_grad, batches, cfg, rng, mon)
    return Trained(best, ep, tuple(hist))


def rnn_step(P: Params, h: np.ndarray, i: int) -> np.ndarray:
    return np.tanh(P["E"][i] @ P["Wx"] + h @ P["Wh"] + P["b"])


def rnn_states(P: Params, ids) -> np.ndarray:
    """The states h₁ … h_T a sentence drives (h_t predicts target t; input 0 is <s>)."""
    h, hs = np.zeros(P["Wh"].shape[0]), []
    for i in [len(P["E"]) - 1, *ids[:-1]]:
        h = rnn_step(P, h, i)
        hs.append(h)
    return np.array(hs)


def tightness_epsilon(Wo: np.ndarray, bo: np.ndarray, eos: int) -> float:
    """Thm 4's ε: a lower bound on p(</s> | h) over the whole cube ‖h‖∞ ≤ 1, from
    |z_j − z_eos| ≤ ‖W_o[:, j] − W_o[:, eos]‖₁ + (b_o,j − b_o,eos)."""
    gap = np.abs(Wo - Wo[:, [eos]]).sum(axis=0) + (bo - bo[eos])  # max over the cube of z_j − z_eos
    return float(np.exp(-special.logsumexp(gap)))  # p_eos = 1 / Σ_j e^{z_j − z_eos} ≥ 1 / Σ_j e^{gap_j}


def sample_sentences(P: Params, n: int, rng: np.random.Generator, max_len: int, eos: int = EOS_ID):
    """n sentences sampled from the RNN, each until </s> or max_len tokens: their lengths and, per sentence,
    the smallest p(</s> | h_t) along its path."""
    h = np.zeros((n, P["Wh"].shape[0]))
    x = np.full(n, len(P["E"]) - 1)
    alive, length, pmin = np.ones(n, bool), np.zeros(n, int), np.ones(n)
    for _ in range(max_len):
        h = np.tanh(P["E"][x] @ P["Wx"] + h @ P["Wh"] + P["b"])
        p = softmax(h @ P["Wo"] + P["bo"])
        pmin = np.where(alive, np.minimum(pmin, p[:, eos]), pmin)
        x = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        length += alive
        alive &= x != eos
        if not alive.any():
            break
    return length, pmin


def jacobian_product_norms(Wh: np.ndarray, hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thm 5 along one trajectory h₁ … h_T: for k = T−1, …, 0, the measured ‖∂h_T/∂h_k‖₂ and the trajectory
    bound Π_{j>k} ‖diag(1 − h_j²) W_hᵀ‖₂ (itself ≤ σ_max(W_h)^{T−k}). Entry m − 1 of each array is lag m."""
    M, bound = np.eye(len(Wh)), 1.0
    measured, bounds = [], []
    for j in range(len(hs) - 1, -1, -1):
        J = (1 - hs[j] ** 2)[:, None] * Wh.T  # ∂h_j/∂h_{j−1} in the column convention
        M = M @ J
        bound *= np.linalg.norm(J, 2)
        measured.append(np.linalg.norm(M, 2))
        bounds.append(bound)
    return np.array(measured), np.array(bounds)


def loss_sensitivity(P: Params, ids) -> tuple[list[list[float]], list[list[float]]]:
    """For each target t of a sentence, ‖∂ℓ_t/∂h_k‖₂ for k = 0, …, t (h_k the state that predicts target k) — how
    strongly the loss at t reaches back through the recurrence (the lab's Panel B) — and Thm 5's bound on it,
    ‖∂ℓ_t/∂h_t‖ Π_{j=k+1}^{t} ‖diag(1 − h_j²) W_hᵀ‖₂."""
    hs = rnn_states(P, ids)
    steps = [float(np.linalg.norm((1 - h**2)[:, None] * P["Wh"].T, 2)) for h in hs]
    out, bounds = [], []
    for t, y in enumerate(ids):
        g = softmax(hs[t] @ P["Wo"] + P["bo"])
        g[y] -= 1
        g = g @ P["Wo"].T  # ∂ℓ_t/∂h_t
        row, brow = [float(np.linalg.norm(g))], [float(np.linalg.norm(g))]
        for k in range(t, 0, -1):
            g = (g * (1 - hs[k] ** 2)) @ P["Wh"].T  # ∂ℓ_t/∂h_{k−1} = ∂ℓ_t/∂h_k · ∂h_k/∂h_{k−1}
            row.append(float(np.linalg.norm(g)))
            brow.append(brow[-1] * steps[k])
        out.append(row[::-1])
        bounds.append(brow[::-1])
    return out, bounds


def scaled_matrix(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """A Gaussian n × n matrix rescaled to spectral norm σ."""
    W = rng.normal(size=(n, n))
    return W * (sigma / np.linalg.norm(W, 2))


def tanh_trajectory(rng: np.random.Generator, Wh: np.ndarray, T: int, in_scale: float = 1.0) -> np.ndarray:
    h, hs = np.zeros(len(Wh)), []
    for _ in range(T):
        h = np.tanh(rng.normal(0, in_scale, len(Wh)) + h @ Wh)
        hs.append(h)
    return np.array(hs)


# The LSTM (Hochreiter and Schmidhuber 1997, with Gers et al.'s forget gate): gates in the order i, f, o, g.


def init_lstm(rng: np.random.Generator, k: int, h: int, forget_bias: float, out: int = 2) -> Params:
    P = {"Wx": rng.normal(0, 1 / math.sqrt(k), (k, 4 * h)), "Wh": rng.normal(0, 1 / math.sqrt(h), (h, 4 * h)),
         "b": np.zeros(4 * h), "Wo": rng.normal(0, 1 / math.sqrt(h), (h, out)), "bo": np.zeros(out)}
    P["b"][h : 2 * h] = forget_bias
    return P


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lstm_cell(P: Params, x: np.ndarray, h: np.ndarray, c: np.ndarray):
    """One step: c_t = f ⊙ c_{t−1} + i ⊙ g, h_t = o ⊙ tanh(c_t), every gate a function of (x_t, h_{t−1})."""
    H = h.shape[-1]
    z = x @ P["Wx"] + h @ P["Wh"] + P["b"]
    i, f, o, g = _sigmoid(z[..., :H]), _sigmoid(z[..., H : 2 * H]), _sigmoid(z[..., 2 * H : 3 * H]), np.tanh(z[..., 3 * H :])
    c = f * c + i * g
    return o * np.tanh(c), c, (i, f, o, g)


def lstm_cell_jacobian(P: Params, xs: np.ndarray, h0: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """∂c_T/∂c₀ (all paths), by the complex step: c₀ perturbed, h₀ = o₀ ⊙ tanh(c₀) recomputed with o₀ fixed."""
    o0 = h0 / np.tanh(c0)

    def run(c):
        h = o0 * np.tanh(c)
        for x in xs:
            h, c, _ = lstm_cell(P, x, h, c)
        return c

    H = len(c0)
    J = np.zeros((H, H))
    for i in range(H):
        c = c0.astype(complex)
        c[i] += 1e-30j
        J[:, i] = run(c).imag / 1e-30
    return J


def lstm_forget_path(P: Params, xs: np.ndarray, h0: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """Prop 4's cell-path term Π_t diag(f_t), as the vector of its diagonal."""
    h, c, prod = h0, c0, np.ones(len(c0))
    for x in xs:
        h, c, (_, f, _, _) = lstm_cell(P, x, h, c)
        prod = prod * f
    return prod


def zero_gate_recurrence(P: Params, scale: float = 0.0) -> Params:
    """Scale the recurrent weights of the i, f, g gates (not o): Prop 4's residual is linear in them."""
    H = P["Wh"].shape[0]
    Wh = P["Wh"].copy()
    for blk in (0, 1, 3):
        Wh[:, blk * H : (blk + 1) * H] *= scale
    return {**P, "Wh": Wh}


# The long-lag memory task: read a bit, then LAG noise symbols, then report the bit.

LAG, LAG_H, LAG_STEPS, LAG_BATCH, LAG_LR = 60, 16, 600, 64, 0.01
FORGET_BIASES = (1.0, 3.0)  # Gers et al.'s 1, and a bias whose cell path σ(3)^60 = 0.056 still reaches the bit


def lag_data(rng: np.random.Generator, n: int, lag: int = LAG) -> tuple[np.ndarray, np.ndarray]:
    """Inputs (lag + 1) × n one-hot ids: x₀ = the bit ∈ {0, 1}, then noise ∈ {2, 3}; the target is the bit."""
    bit = rng.integers(0, 2, n)
    return np.concatenate([bit[None], rng.integers(2, 4, (lag, n))]), bit


def lag_loss_grad(P: Params, X: np.ndarray, bit: np.ndarray, cell: str):
    """Cross-entropy of the final-step bit prediction, its gradient by BPTT, and ‖∂ℓ/∂s₀‖ — the gradient
    reaching the state s₀ that has just read the bit (for the LSTM, the total derivative in c₀)."""
    T, n = X.shape
    H = P["Wh"].shape[0]
    eye = np.eye(P["Wx"].shape[0])
    h, c, cache = np.zeros((n, H)), np.zeros((n, H)), []
    for t in range(T):
        hp, cp = h, c
        if cell == "lstm":
            h, c, gates = lstm_cell(P, eye[X[t]], h, c)
            cache.append((hp, cp, gates, np.tanh(c)))
        else:
            h = np.tanh(eye[X[t]] @ P["Wx"] + h @ P["Wh"] + P["b"])
            cache.append((hp, h))
    lp = log_softmax(h @ P["Wo"] + P["bo"])
    g = np.exp(lp)
    g[np.arange(n), bit] -= 1
    g /= n
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["Wo"], G["bo"] = h.T @ g, g.sum(0)
    dh, dc, reach = g @ P["Wo"].T, np.zeros((n, H)), 0.0
    for t in range(T - 1, -1, -1):
        if cell == "lstm":
            hp, cp, (i, f, o, gg), tc = cache[t]
            dct = dc + dh * o * (1 - tc * tc)  # the total ∂ℓ/∂c_t
            if t == 0:
                reach = float(np.linalg.norm(dct, axis=1).mean())
            dz = np.concatenate([dct * gg * i * (1 - i), dct * cp * f * (1 - f), dh * tc * o * (1 - o),
                                 dct * i * (1 - gg * gg)], 1)
            dc = dct * f
        else:
            hp, hh = cache[t]
            if t == 0:
                reach = float(np.linalg.norm(dh, axis=1).mean())
            dz = dh * (1 - hh * hh)
        G["Wx"] += eye[X[t]].T @ dz
        G["Wh"] += hp.T @ dz
        G["b"] += dz.sum(0)
        dh = dz @ P["Wh"].T
    return float(-lp[np.arange(n), bit].mean()), G, reach, float((lp.argmax(1) == bit).mean())


def lag_models() -> tuple[tuple[str, str, float], ...]:
    return (("elman", "elman", 0.0), *((f"lstm-{b:g}", "lstm", b) for b in FORGET_BIASES))


def init_lag_model(rng: np.random.Generator, cell: str, forget_bias: float) -> Params:
    if cell == "lstm":
        return init_lstm(rng, 4, LAG_H, forget_bias)
    return {"Wx": rng.normal(0, 1 / 2, (4, LAG_H)), "Wh": rng.normal(0, 1 / math.sqrt(LAG_H), (LAG_H, LAG_H)),
            "b": np.zeros(LAG_H), "Wo": rng.normal(0, 1 / math.sqrt(LAG_H), (LAG_H, 2)), "bo": np.zeros(2)}


@lru_cache(maxsize=1)
def long_lag_census(n_seeds: int = 12) -> dict[str, list[float]]:
    """‖∂ℓ/∂s₀‖ at initialization, for n_seeds independent initializations of each model."""
    rng = np.random.default_rng([SEED, 300])
    out: dict[str, list[float]] = {name: [] for name, _, _ in lag_models()}
    for _ in range(n_seeds):
        X, bit = lag_data(rng, 256)
        for name, cell, fb in lag_models():
            out[name].append(lag_loss_grad(init_lag_model(rng, cell, fb), X, bit, cell)[2])
    return out


@lru_cache(maxsize=1)
def long_lag_training() -> dict[str, dict[str, float]]:
    """Each model trained by Adam for LAG_STEPS minibatches, then scored on 2,000 fresh sequences."""
    out = {}
    for k, (name, cell, fb) in enumerate(lag_models()):
        rng = np.random.default_rng([SEED, 310 + k])
        P = init_lag_model(rng, cell, fb)
        m = {q: np.zeros_like(v) for q, v in P.items()}
        s = {q: np.zeros_like(v) for q, v in P.items()}
        X, bit = lag_data(rng, 256)
        reach0 = lag_loss_grad(P, X, bit, cell)[2]
        for step in range(1, LAG_STEPS + 1):
            X, bit = lag_data(rng, LAG_BATCH)
            _, G, _, _ = lag_loss_grad(P, X, bit, cell)
            for q in P:
                m[q] = ADAM_B1 * m[q] + (1 - ADAM_B1) * G[q]
                s[q] = ADAM_B2 * s[q] + (1 - ADAM_B2) * G[q] ** 2
                P[q] -= LAG_LR * (m[q] / (1 - ADAM_B1**step)) / (np.sqrt(s[q] / (1 - ADAM_B2**step)) + ADAM_EPS)
        X, bit = lag_data(rng, 2000)
        loss, _, _, acc = lag_loss_grad(P, X, bit, cell)
        out[name] = {"reach_init": reach0, "test_loss": loss, "accuracy": acc}
    return out


# --------------------------------------------------------------------------------------------
# 5. The softmax bottleneck.
# --------------------------------------------------------------------------------------------

BOTTLENECK_CONTEXTS = 1000  # the most frequent two-word training contexts


@lru_cache(maxsize=1)
def bottleneck_contexts() -> tuple[NGram, ...]:
    tot = ng.train_totals()[ORDER]
    return tuple(u for u, _ in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:BOTTLENECK_CONTEXTS])


def numerical_rank(L: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(L))


@lru_cache(maxsize=1)
def bottleneck_summary() -> dict[str, float]:
    """Thm 6 on the corpus: ranks of the log-probability matrices [log p(w | u)] over the 1,000 most frequent
    training contexts, for the NPLM (state dimension h), interpolated KN, and the RNN (over 1,000 held-out
    prefixes); and the share of the row-centered KN matrix's squared Frobenius norm beyond rank h + 1 (its
    best rank-(h + 1) approximation's residual, Eckart–Young)."""
    us = bottleneck_contexts()
    P = trained_nplm().params
    X = np.array([[INDEX.get(w, BOS_ID if w == BOS else UNK_ID) for w in u] for u in us])
    L_nplm = _nplm_forward(P, X)[2]
    kn = pce.word_vector("ikn")
    L_kn = np.log(np.array([kn(u) for u in us]))
    R = trained_rnn().params
    hs = np.concatenate([rnn_states(R, ids) for ids in ids_of("held")])[:BOTTLENECK_CONTEXTS]
    L_rnn = log_softmax(hs @ R["Wo"] + R["bo"])
    centered = L_kn - L_kn.mean(1, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    h = NPLM_CFG["h"]
    return {"contexts": len(us), "h": h, "rank_nplm": numerical_rank(L_nplm), "rank_rnn": numerical_rank(L_rnn),
            "rank_kn": numerical_rank(L_kn), "kn_energy_beyond": float((sv[h + 1 :] ** 2).sum() / (sv**2).sum())}


# --------------------------------------------------------------------------------------------
# 6. The honest comparison: per-token logprobs, mixtures, dynamic evaluation, exact codes.
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def kn_logprobs(test: str) -> np.ndarray:
    """Interpolated KN (Topic 3's fit) on every predicted token of a text, in ng.events order."""
    f = ng.prob_fn("ikn")
    return np.array([math.log(f(g[-1], g[:-1])) for g in ng.events(ng.with_unk(corpus()[test]), ORDER)])


def _split(flat: np.ndarray, test: str) -> list[np.ndarray]:
    return np.split(flat, np.cumsum([len(s) for s in ids_of(test)])[:-1])


@lru_cache(maxsize=None)
def static_logprobs(name: str, test: str) -> np.ndarray:
    if name == "kn":
        return kn_logprobs(test)
    if name == "nplm":
        return nplm_target_logprobs(trained_nplm().params, *events_of(test))
    return np.concatenate(rnn_sentence_logprobs(trained_rnn().params, ids_of(test)))


def dynamic_update(P: Params, ids, lr: float) -> None:
    """Dynamic evaluation's step (Mikolov et al. 2010; Krause et al. 2018): plain SGD on the summed loss of the
    sentence just coded. The decoder knows that sentence by then, so it can take the same step."""
    _, G = rnn_loss_grad(P, [list(ids)])
    for k in P:
        P[k] -= lr * len(ids) * G[k]


def dynamic_logprobs(P0: Params, sents, lr: float) -> list[np.ndarray]:
    """Score each sentence with the current parameters, then update on it: the prequential loss."""
    P = {k: v.copy() for k, v in P0.items()}
    out = []
    for ids in sents:
        X, Y, M, xs, hs, lp = _rnn_forward(P, [list(ids)])
        out.append(_token_logprobs(Y, M, lp)[:, 0])
        G = _rnn_backward(P, X, Y, M, xs, hs, lp)
        for k in P:
            P[k] -= lr * len(ids) * G[k]
    return out


DEV_DYN_SENTENCES = 400  # the dev prefix the step size is chosen on


@lru_cache(maxsize=1)
def dynamic_lr() -> float:
    P, dev = trained_rnn().params, ids_of("dev")[:DEV_DYN_SENTENCES]
    loss = {lr: -np.concatenate(dynamic_logprobs(P, dev, lr)).mean() for lr in DYN_LRS}
    return min(loss, key=loss.get)


@lru_cache(maxsize=None)
def dyn_logprobs(test: str) -> np.ndarray:
    return np.concatenate(dynamic_logprobs(trained_rnn().params, ids_of(test), dynamic_lr()))


@lru_cache(maxsize=None)
def mix_lambda(name: str) -> float:
    """The weight λ of the neural model in λ p_neural + (1 − λ) p_KN that minimizes dev log-loss."""
    a, b = np.exp(static_logprobs(name, "dev")), np.exp(kn_logprobs("dev"))
    return float(min(MIX_LAMBDAS, key=lambda lam: -np.log(lam * a + (1 - lam) * b).mean()))


COMPARED = ("kn", "nplm", "nplm+kn", "rnn", "rnn+kn", "rnn-dyn")


@lru_cache(maxsize=None)
def logprobs(name: str, test: str) -> np.ndarray:
    """log p of every predicted token (closed vocabulary: an OOV word counts as one <unk> event)."""
    if name == "rnn-dyn":
        return dyn_logprobs(test)
    if name.endswith("+kn"):
        base, lam = name[:-3], mix_lambda(name[:-3])
        return np.log(lam * np.exp(static_logprobs(base, test)) + (1 - lam) * np.exp(kn_logprobs(test)))
    return static_logprobs(name, test)


@lru_cache(maxsize=None)
def spelled_nats(test: str) -> np.ndarray:
    """Per sentence, −log S of its OOV words under Topic 4's speller — the same bits for every model here, since
    all of them share Topic 3's vocabulary."""
    a = pce.sentence_nll("ikn", test)
    return a[:, 3] - a[:, 0]


@lru_cache(maxsize=None)
def sentence_table(name: str, test: str) -> np.ndarray:
    """Per sentence: [closed NLL (nats), predicted tokens, characters, open NLL (closed + spelling)] — the
    layout of pce.sentence_nll, so Topic 4's ratio_estimate applies unchanged."""
    a = pce.sentence_nll("ikn", test)
    closed = np.array([-x.sum() for x in _split(logprobs(name, test), test)])
    return np.column_stack([closed, a[:, 1], a[:, 2], closed + spelled_nats(test)])


@lru_cache(maxsize=None)
def comparison_row(name: str, test: str) -> dict[str, float]:
    """Closed nats per token and open bits per character, each with its sentence-level standard error (Topic 4
    Thm 2), and the paired difference from interpolated KN in bpc."""
    a, k = sentence_table(name, test), sentence_table("kn", test)
    nats, nats_se = pce.ratio_estimate(a[:, 0], a[:, 1])
    bpc, bpc_se = pce.ratio_estimate(a[:, 3] / LN2, a[:, 2])
    d, d_se = pce.ratio_estimate((a[:, 3] - k[:, 3]) / LN2, a[:, 2])
    return {"nats": nats, "nats_se": nats_se, "pp": math.exp(nats), "bpc": bpc, "bpc_se": bpc_se,
            "bpc_minus_kn": d, "bpc_minus_kn_se": d_se}


@lru_cache(maxsize=None)
def event_classes(test: str) -> np.ndarray:
    """Per predicted token: 2 if its trigram occurs in training, 1 if only its bigram does, 0 if neither."""
    c3, c2 = ng.train_counts()[3], ng.train_counts()[2]
    return np.array([2 if g in c3 else 1 if g[1:] in c2 else 0 for g in ng.events(ng.with_unk(corpus()[test]), ORDER)])


@lru_cache(maxsize=1)
def class_breakdown() -> dict[str, dict[int, float]]:
    """Held-out nats per token by event class, for KN and the two networks."""
    cls = event_classes("held")
    return {name: {c: float(-logprobs(name, "held")[cls == c].mean()) for c in (0, 1, 2)}
            for name in ("kn", "nplm", "rnn")} | {"count": {c: int((cls == c).sum()) for c in (0, 1, 2)}}


def param_count(P: Params) -> int:
    return int(sum(v.size for v in P.values()))


def kn_table_size() -> int:
    """Stored n-gram counts of Topic 3's trigram KN (every order)."""
    return int(sum(len(c) for c in ng.train_counts().values()))


class RNNCoder:
    """The open-vocabulary RNN as a sequential model for pce.arithmetic_encode / arithmetic_decode (Topic 4's
    interface: start, probs, step). An in-vocabulary word is one symbol; an OOV word is <unk>, then its letters
    and an end mark from Topic 4's speller, exactly as pce.OpenVocabModel codes interpolated KN. With
    dyn_lr > 0 the parameters take dynamic_update's step on each sentence right after it is coded; the decoder
    replays the same steps, so the code stays lossless (prequential coding)."""

    def __init__(self, P0: Params, dyn_lr: float = 0.0):
        self.P0, self.dyn_lr, self.spell, self.vocab = P0, dyn_lr, pce.speller(1), ng.vocabulary()

    def start(self):
        self.P = {k: v.copy() for k, v in self.P0.items()} if self.dyn_lr else self.P0
        return self._sentence_start()

    def _sentence_start(self):
        return ("w", rnn_step(self.P, np.zeros(len(self.P["Wh"])), BOS_ID), ())

    def probs(self, state):
        if state[0] == "w":
            return ALPHABET, INDEX, softmax(state[1] @ self.P["Wo"] + self.P["bo"])
        return self.spell.alphabet, self.spell.index, self.spell(state[3])

    def step(self, state, sym: str):
        if state[0] == "c":
            _, h, ids, cctx = state
            return ("w", h, ids) if sym == EOS else ("c", h, ids, cctx[1:] + (sym,))
        _, h, ids = state
        ids = ids + (INDEX[sym],)
        if sym == EOS:
            if self.dyn_lr:
                dynamic_update(self.P, ids, self.dyn_lr)
            return self._sentence_start()
        h = rnn_step(self.P, h, INDEX[sym])
        return ("c", h, ids, (BOS,) * (pce.SPELL_ORDER - 1)) if sym == UNK else ("w", h, ids)

    def symbols(self, sents) -> list[str]:
        return pce.OpenVocabModel.symbols(self, sents)


def rnn_coder(name: str) -> RNNCoder:
    return RNNCoder(trained_rnn().params, dynamic_lr() if name == "rnn-dyn" else 0.0)


@lru_cache(maxsize=None)
def corpus_code(name: str, test: str, n_sents: int | None = None) -> pce.Code:
    """A whole test text (or its first n_sents sentences) as ONE arithmetic codeword under the open RNN."""
    m = rnn_coder(name)
    return pce.arithmetic_encode(m, m.symbols(corpus()[test][:n_sents]))


# --------------------------------------------------------------------------------------------
# 7. Finance case study: EDGAR.
# --------------------------------------------------------------------------------------------


def ppm_cache_spelled_nats(test: str) -> float:
    """OOV words coded with a PPM-C escape (Moffat 1990) in front of the speller: after n OOV tokens of u
    distinct types, a word already seen is sent as its cache entry with probability c(w)/(n + u), and a new
    word escapes with probability u/(n + u) — 1 for the first — and is then spelled. Adaptive, like lzma."""
    seen, total, nats = Counter(), 0, 0.0
    for s in corpus()[test]:
        for w in s:
            if w in ng.vocabulary():
                continue
            u = len(seen)
            if seen[w]:
                nats -= math.log(seen[w] / (total + u))
            else:
                nats += pce.spell_nll(w) - (math.log(u / (total + u)) if total else 0.0)
            seen[w] += 1
            total += 1
    return nats


@lru_cache(maxsize=None)
def lzma_given_bpc(test: str) -> float:
    """Topic 4's lzma figure: |C(train + test)| − |C(train)| bytes, in bits per character of the test text."""
    T, X = pce.text_of(corpus()["train"]).encode(), pce.text_of(corpus()[test]).encode()
    return (len(pce.compress("lzma", T + X)) - len(pce.compress("lzma", T))) * 8 / len(X)


@lru_cache(maxsize=1)
def finance_summary() -> dict[str, float]:
    out = {"uniform_nats": pce.uniform_log_loss(), "lzma_edgar_bpc": lzma_given_bpc("edgar")}
    for t in TESTS:
        nc = pce.n_chars(corpus()[t])
        out[f"spelled_bpc_{t}"] = float(spelled_nats(t).sum() / nc / LN2)
        out[f"cache_spelled_bpc_{t}"] = ppm_cache_spelled_nats(t) / nc / LN2
        closed = sentence_table("rnn-dyn", t)[:, 0].sum()
        out[f"rnn_dyn_cache_bpc_{t}"] = (closed + ppm_cache_spelled_nats(t)) / nc / LN2
        for name in COMPARED:
            r = comparison_row(name, t)
            out[f"{name}_{t}_nats"], out[f"{name}_{t}_bpc"] = r["nats"], r["bpc"]
    return out


# --------------------------------------------------------------------------------------------
# Viz payload — every number the lab shows (mirrored to the decimal in the TSX's JSON in Phase B).
# --------------------------------------------------------------------------------------------

LAB_MODELS = ("kn", "nplm", "rnn", "rnn-dyn")
LAB_SENTENCES = (
    ("held", ("cedar", "ridge", "bancorp", "owns", "brackenbury", "lending", "outright")),  # 3 unseen bigrams: sharing
    ("held", ("pellucid", "software", "based", "in", "dublin", "expanded", "its", "headcount", "clinicbase",
              "footprint")),  # "footprint" is predictable only from "expanded its …", beyond the trigram's reach
    ("edgar", ("book", "reserves", "are", "established", "for", "these", "plans")),  # Topic 4's lab sentence 3
)
THROUGH_SIGMAS = (0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
THROUGH_H, THROUGH_T, THROUGH_LAGS, THROUGH_DRAWS = 32, 60, 50, 16
_r = pce._r


def _sig(x: float, n: int = 4):
    """Round to n significant figures (for quantities spanning many decades, shown on log axes)."""
    return None if not math.isfinite(x) else float(f"{x:.{n}g}")


@lru_cache(maxsize=1)
def through_time() -> dict:
    """Panel B's synthetic curves: for W_h Gaussian with spectral norm σ, the median over draws of
    log₁₀ ‖∂h_T/∂h_{T−m}‖ for m = 1 … THROUGH_LAGS — tanh along a random trajectory, and linear ((W_hᵀ)^m) —
    with the median spectral radius ρ(W_h)."""
    rng = np.random.default_rng([SEED, 400])
    out = {"sigmas": list(THROUGH_SIGMAS), "tanh": [], "linear": [], "rho": []}
    for sigma in THROUGH_SIGMAS:
        tanh_rows, lin_rows, rhos = [], [], []
        for _ in range(THROUGH_DRAWS):
            Wh = scaled_matrix(rng, THROUGH_H, sigma)
            meas, _ = jacobian_product_norms(Wh, tanh_trajectory(rng, Wh, THROUGH_T))
            tanh_rows.append(np.log10(meas[:THROUGH_LAGS]))
            M, lin = np.eye(THROUGH_H), []
            for _ in range(THROUGH_LAGS):
                M = M @ Wh.T
                lin.append(math.log10(np.linalg.norm(M, 2)))
            lin_rows.append(lin)
            rhos.append(np.abs(np.linalg.eigvals(Wh)).max())
        out["tanh"].append([_r(x, 3) for x in np.median(tanh_rows, axis=0)])
        out["linear"].append([_r(x, 3) for x in np.median(lin_rows, axis=0)])
        out["rho"].append(_r(float(np.median(rhos)), 3))
    return out


DECAY_SENTENCES = 200


@lru_cache(maxsize=1)
def corpus_decay() -> dict[str, list]:
    """The trained RNN on the first DECAY_SENTENCES held-out sentences, from each sentence's last state h_T
    back: median ‖∂h_T/∂h_{T−m}‖ and median trajectory bound by lag m (lags with ≥ 30 sentences); the number of
    (sentence, lag) pairs where the measured norm exceeds the bound (Thm 5 says none); every per-step factor
    ‖diag(1 − h_j²) W_hᵀ‖₂ (its minimum and median); and the share of saturated units, |h| > 0.9."""
    P = trained_rnn().params
    by_lag_m: dict[int, list[float]] = {}
    by_lag_b: dict[int, list[float]] = {}
    violations, steps, sat = 0, [], []
    for ids in ids_of("held")[:DECAY_SENTENCES]:
        hs = rnn_states(P, ids)
        meas, bnd = jacobian_product_norms(P["Wh"], hs)
        violations += int(np.sum(meas > bnd * (1 + 1e-10)))
        steps.append(np.concatenate([bnd[:1], bnd[1:] / bnd[:-1]]))
        sat.append(np.abs(hs) > 0.9)
        for m in range(1, len(ids)):  # lag m reaches h_{T−m}, a state the sentence produced (lag T is h₀ = 0)
            by_lag_m.setdefault(m, []).append(meas[m - 1])
            by_lag_b.setdefault(m, []).append(bnd[m - 1])
    lags = [m for m in sorted(by_lag_m) if len(by_lag_m[m]) >= 30]
    return {"lags": lags, "measured": [float(np.median(by_lag_m[m])) for m in lags],
            "bound": [float(np.median(by_lag_b[m])) for m in lags], "n": [len(by_lag_m[m]) for m in lags],
            "violations": violations, "min_step_norm": float(np.concatenate(steps).min()),
            "median_step_norm": float(np.median(np.concatenate(steps))),
            "saturated_share": float(np.concatenate(sat).mean())}


def lab_sentence(test: str, words: NGram) -> dict:
    j = corpus()[test].index(words)
    ids = ids_of(test)[j]
    cls = _split(event_classes(test), test)[j]
    lps = {name: _split(logprobs(name, test), test)[j] for name in LAB_MODELS}
    tokens = []
    for i, (w, y) in enumerate(zip([*words, EOS], ids)):
        oov = w != EOS and w not in ng.vocabulary()
        tokens.append({"w": w, "oov": oov, "cls": int(cls[i]),
                       "spell_bits": _r(pce.spell_nll(w) / LN2, 3) if oov else None,
                       "bits": {name: _r(-lps[name][i] / LN2, 3) for name in LAB_MODELS}})
    P = trained_rnn().params
    hs = rnn_states(P, ids)
    sens, bound = loss_sensitivity(P, ids)
    return {"test": test, "index": j, "words": list(words), "chars": pce.n_chars([words]), "tokens": tokens,
            "sensitivity": [[_sig(x) for x in row] for row in sens],
            "sensitivity_bound": [[_sig(x) for x in row] for row in bound],
            "step_norms": [_r(float(np.linalg.norm((1 - h**2)[:, None] * P["Wh"].T, 2)), 4) for h in hs]}


@lru_cache(maxsize=1)
def viz_payload() -> dict:
    R = trained_rnn().params
    corpus_rows = {t: {name: {k: _r(v) for k, v in comparison_row(name, t).items()} for name in COMPARED} for t in TESTS}
    census, trained = long_lag_census(), long_lag_training()
    lag = {name: {"log10_reach": _r(math.log10(float(np.median(census[name]))), 3),
                  "accuracy": _r(trained[name]["accuracy"], 4), "test_loss": _sig(trained[name]["test_loss"]),
                  "log10_cell_path": None if cell != "lstm" else _r(LAG * math.log10(float(_sigmoid(fb))), 3)}
           for name, cell, fb in lag_models()}
    return {"models": list(LAB_MODELS), "sentences": [lab_sentence(t, w) for t, w in LAB_SENTENCES],
            "sigma_max": _r(float(np.linalg.norm(R["Wh"], 2)), 4),
            "rho": _r(float(np.abs(np.linalg.eigvals(R["Wh"])).max()), 4),
            "through_time": through_time(),
            "corpus_decay": {k: ([_sig(x) for x in v] if k in ("measured", "bound") else v)
                             for k, v in corpus_decay().items() if k in ("lags", "measured", "bound", "n")},
            "long_lag": {"lag": LAG, "models": lag}, "corpus": corpus_rows, "uniform_nats": _r(pce.uniform_log_loss()),
            "V": V, "lzma_edgar_bpc": _r(lzma_given_bpc("edgar"))}


# --------------------------------------------------------------------------------------------
# Summary — the numbers the page quotes.
# --------------------------------------------------------------------------------------------


def summary() -> dict[str, float]:
    out: dict[str, float] = {"dev_sentences": len(corpus()["dev"]), "dev_tokens": len(events_of("dev")[1])}
    n, r = trained_nplm(), trained_rnn()
    out["nplm_epoch"], out["rnn_epoch"] = n.epoch + 1, r.epoch + 1
    out["nplm_params"], out["rnn_params"], out["kn_ngrams"] = param_count(n.params), param_count(r.params), kn_table_size()
    out["nplm_lipschitz"] = nplm_lipschitz(n.params)
    out["lipschitz_ratio_median"], out["lipschitz_ratio_max"] = float(np.median(lipschitz_ratios())), float(lipschitz_ratios().max())
    for name in ("nplm", "nplm-nodrop"):
        for t in TESTS:
            out.update({f"{name}_{t}_{k}": v for k, v in unk_cost(name, t).items()})
    out["dyn_lr"] = dynamic_lr()
    for name in ("nplm", "rnn"):
        out[f"mix_lambda_{name}"] = mix_lambda(name)
    for t in TESTS:
        for name in COMPARED:
            for k, v in comparison_row(name, t).items():
                out[f"{t}_{name}_{k}"] = v
    for name, row in class_breakdown().items():
        for c, v in row.items():
            out[f"class{c}_{name}"] = v
    out.update({f"unk_{k}": v for k, v in unk_moment().items()})
    out.update({f"nonconvex_{k}": v for k, v in nonconvexity().items()})
    out["rnn_sigma_max"] = float(np.linalg.norm(r.params["Wh"], 2))
    out["rnn_rho"] = float(np.abs(np.linalg.eigvals(r.params["Wh"])).max())
    out["rnn_tightness_eps"] = tightness_epsilon(r.params["Wo"], r.params["bo"], EOS_ID)
    cd = corpus_decay()
    out.update({f"decay_{k}": cd[k] for k in ("min_step_norm", "median_step_norm", "saturated_share")})
    for m in (1, 4, 8, 12):
        if m in cd["lags"]:
            out[f"decay_lag{m}_measured"] = cd["measured"][cd["lags"].index(m)]
            out[f"decay_lag{m}_bound"] = cd["bound"][cd["lags"].index(m)]
    for name, row in long_lag_training().items():
        for k, v in row.items():
            out[f"lag_{name}_{k}"] = v
    for name, xs in long_lag_census().items():
        out[f"lag_census_{name}_median"] = float(np.median(xs))
    out.update({f"bottleneck_{k}": v for k, v in bottleneck_summary().items()})
    for t in TESTS:
        c = corpus_code("rnn-dyn", t)
        out[f"code_rnn_dyn_{t}_bits"], out[f"code_rnn_dyn_{t}_model_bits"] = c.length, c.model_bits
        out[f"code_rnn_dyn_{t}_bpc"] = c.length / pce.n_chars(corpus()[t])
    out.update({f"fin_{k}": v for k, v in finance_summary().items()})
    return out


# --------------------------------------------------------------------------------------------
# Verification harness — every claim the page makes.
# --------------------------------------------------------------------------------------------


def test_softmax_is_positive_and_proper():
    """Prop 1: every NPLM and RNN assigns every word positive probability in every context, and each next-word
    distribution sums to 1 — on random instances and on every held-out and EDGAR context of the trained models."""
    rng = np.random.default_rng([SEED, 1])
    for _ in range(50):
        P = init_nplm(rng, 3, 4, vocab=9)
        X = rng.integers(0, 10, (40, 2))
        p = np.exp(_nplm_forward(P, X)[2])
        assert np.all(p > 0) and np.allclose(p.sum(1), 1, atol=1e-12)
    for t in TESTS:
        for name in ("nplm", "rnn"):
            assert np.all(np.isfinite(static_logprobs(name, t)))
    P = trained_nplm().params
    p = np.exp(_nplm_forward(P, events_of("edgar")[0])[2])
    assert np.all(p > 0) and np.allclose(p.sum(1), 1, atol=1e-10)


def test_lipschitz_smoothing_random():
    """Thm 1 on random NPLMs: for every pair of context points and every word,
    |log p(w | x) − log p(w | x′)| ≤ √2 ‖W₂‖₂ ‖W₁‖₂ ‖x − x′‖₂."""
    rng = np.random.default_rng([SEED, 2])
    for _ in range(100):
        d, h, Vr = (int(v) for v in rng.integers(2, 7, 3))
        P = init_nplm(rng, d, h, vocab=Vr)
        P["W1"] *= rng.uniform(0.5, 4)
        P["W2"] *= rng.uniform(0.5, 4)
        x, x2 = rng.normal(0, 1, (200, 2 * d)), rng.normal(0, 1, (200, 2 * d))
        x2 = x + rng.uniform(0.001, 1, (200, 1)) * (x2 - x)
        lhs = np.abs(logprobs_at_points(P, x) - logprobs_at_points(P, x2)).max(1)
        assert np.all(lhs <= nplm_lipschitz(P) * np.linalg.norm(x - x2, axis=1) * (1 + 1e-12))


@lru_cache(maxsize=1)
def lipschitz_ratios() -> np.ndarray:
    """max_w |log p(w | x) − log p(w | x′)| / (L ‖x − x′‖) for 2,000 random pairs of distinct held-out contexts."""
    rng = np.random.default_rng([SEED, 3])
    P = trained_nplm().params
    X = events_of("held")[0]
    i, j = rng.integers(0, len(X), 2000), rng.integers(0, len(X), 2000)
    keep = np.any(X[i] != X[j], axis=1)
    xi, xj = context_points(P, X[i[keep]]), context_points(P, X[j[keep]])
    return np.abs(logprobs_at_points(P, xi) - logprobs_at_points(P, xj)).max(1) / (
        nplm_lipschitz(P) * np.linalg.norm(xi - xj, axis=1))


def test_lipschitz_smoothing_corpus():
    """Thm 1 on the trained NPLM: the bound holds for 2,000 random pairs of held-out contexts; the observed
    ratio max_w |Δ log p| / (L ‖Δx‖) is well below 1 (the bound is a worst case)."""
    ratio = lipschitz_ratios()
    assert ratio.max() <= 1 and np.median(ratio) < 0.1


def test_sharing_wins_where_counts_are_absent():
    """§1 on the corpus: on held-out tokens whose bigram never occurred in training, both networks beat KN;
    on tokens whose trigram did occur, KN beats the NPLM (counts are the better estimate where they exist)."""
    b = class_breakdown()
    assert b["nplm"][0] < b["kn"][0] and b["rnn"][0] < b["kn"][0]
    assert b["kn"][2] < b["nplm"][2]
    assert sum(b["count"].values()) == len(kn_logprobs("held"))


def test_softmax_exponential_family():
    """Prop 2: the softmax Jacobian is diag(p) − ppᵀ (complex-step check), which is Cov(e_Y) for Y ~ p, symmetric
    PSD, with 𝟙 in its null space and rank K − 1; ∇_z of −log p_y is p − e_y; softmax and logsumexp match scipy."""
    rng = np.random.default_rng([SEED, 4])
    for _ in range(100):
        K = int(rng.integers(2, 9))
        z = rng.normal(0, 2, K)
        J = softmax_jacobian(z)
        Jcs = np.column_stack([complex_step_grad(lambda t, i=i: softmax(t)[i], z) for i in range(K)]).T
        p = special.softmax(z)
        E1 = np.eye(K)
        cov = sum(p[k] * np.outer(E1[k], E1[k]) for k in range(K)) - np.outer(p, p)
        assert np.allclose(J, Jcs, atol=1e-13) and np.allclose(J, cov, atol=1e-15) and np.allclose(J, J.T)
        ev = np.linalg.eigvalsh(J)
        assert ev.min() > -1e-14 and np.allclose(J @ np.ones(K), 0, atol=1e-15)
        assert int(np.sum(ev > 1e-12 * ev.max())) == K - 1  # rank K − 1: only 𝟙 is annihilated
        y = int(rng.integers(K))
        g = complex_step_grad(lambda t: -log_softmax(t)[y], z)
        assert np.allclose(g, p - E1[y], atol=1e-13)
        assert abs(special.logsumexp(z) - (z - log_softmax(z))[0]) < 1e-12


def test_score_identity_and_moment_matching():
    """Thm 2 on random softmax regressions: the gradient is (Aᵀ(P − Y), 𝟙ᵀ(P − Y))/n (complex-step check); at a
    stationary point (L-BFGS) every class's mean predicted probability equals its frequency, and every feature's
    p-weighted mean equals its y-weighted mean; the objective is convex along random segments."""
    rng = np.random.default_rng([SEED, 5])
    for _ in range(30):
        n, h, K = int(rng.integers(60, 200)), int(rng.integers(1, 4)), int(rng.integers(2, 6))
        A = rng.normal(0, 1, (n, h))
        Y = rng.integers(0, K, n)
        Y[:K] = np.arange(K)  # every class occurs
        th = rng.normal(0, 1, h * K + K)
        _, g = output_layer_loss_grad(th, A, Y, K)
        assert np.allclose(g, complex_step_grad(lambda t: output_layer_loss(t, A, Y, K), th), atol=1e-12)
        th = fit_output_layer(A, Y, K)
        W, b = th[: h * K].reshape(h, K), th[h * K :]
        Pm = softmax(A @ W + b)
        Ym = np.eye(K)[Y]
        assert np.abs(Pm.mean(0) - Ym.mean(0)).max() < 1e-8
        assert np.abs(A.T @ (Pm - Ym)).max() / n < 1e-8
        t1, t2, lam = rng.normal(0, 2, th.size), rng.normal(0, 2, th.size), rng.uniform()
        L = lambda t: output_layer_loss(t, A, Y, K)  # noqa: E731
        assert L(lam * t1 + (1 - lam) * t2) <= lam * L(t1) + (1 - lam) * L(t2) + 1e-12


def test_zero_frequency_target_has_no_mle():
    """Cor 1: when a class never occurs as a target, its bias's partial derivative is its mean predicted
    probability, which is positive at every parameter value — so there is no stationary point, and gradient
    descent drives that bias down at every step and the class's probability toward 0. On the corpus: trained
    without word dropout, the NPLM's mean p(<unk>) falls below 1/|V| and EDGAR (57.6% <unk>) costs more than the
    uniform guess; with it, mean p(<unk>) tracks the dropout's target rate to 10% (Thm 2) and EDGAR beats uniform."""
    rng = np.random.default_rng([SEED, 6])
    for _ in range(20):
        n, h, K = 120, 2, 4
        A, Y = rng.normal(0, 1, (n, h)), rng.integers(0, K - 1, n)  # class K − 1 is absent
        th, prev = np.zeros(h * K + K), []
        for _ in range(300):
            _, g = output_layer_loss_grad(th, A, Y, K)
            assert g[-1] > 0  # ∂/∂b_{K−1} = mean p_{K−1} > 0
            th = th - 1.0 * g
            prev.append((th[-1], softmax(A @ th[: h * K].reshape(h, K) + th[h * K :])[:, -1].mean()))
        b, p = np.array(prev).T
        assert np.all(np.diff(b) < 0) and np.all(np.diff(p) < 0) and p[-1] < 0.05 * p[0]
    u = unk_moment()
    assert abs(u["mean_p_unk"] / u["target_rate"] - 1) < 0.1 and abs(u["target_rate"] / u["expected_rate"] - 1) < 0.05
    raw, drop = unk_cost("nplm-nodrop", "held"), unk_cost("nplm", "held")
    assert raw["mean_p_unk"] < 1 / V < drop["mean_p_unk"] and raw["unk_nats"] > drop["unk_nats"] + 3
    assert unk_cost("nplm-nodrop", "edgar")["nats"] > pce.uniform_log_loss() > unk_cost("nplm", "edgar")["nats"]


def test_network_is_not_convex():
    """Prop 3: relabeling hidden units leaves the loss unchanged, and the midpoint of the trained NPLM and its
    relabeling has a higher loss than both — impossible for a convex function, since convexity bounds the midpoint's
    loss by the endpoints' average. The symmetry itself is also checked on random small networks."""
    c = nonconvexity()
    assert abs(c["loss"] - c["loss_permuted"]) < 1e-10 and c["loss_midpoint"] > c["loss"] + 0.1
    rng = np.random.default_rng([SEED, 7])
    for _ in range(20):
        R = init_nplm(rng, 3, 5, vocab=6)
        X, Y = rng.integers(0, 7, (100, 2)), rng.integers(0, 6, 100)
        assert abs(nplm_loss(R, X, Y) - nplm_loss(permute_hidden(R, rng.permutation(5)), X, Y)) < 1e-12


def test_reverse_accumulation_random_dags():
    """Thm 3 on 200 random computational graphs with fan-out: one reverse sweep gives every input's adjoint,
    equal to its complex-step derivative (exact to rounding)."""
    rng = np.random.default_rng([SEED, 8])
    fanout = 0
    for _ in range(200):
        m, k, n_ops = int(rng.integers(1, 4)), int(rng.integers(2, 6)), int(rng.integers(3, 16))
        while True:  # chained products can overflow; redraw those graphs
            prog = random_program(rng, m, k, n_ops)
            xs = [rng.normal(0, 1, k) for _ in range(m)]
            g = evaluate(prog, xs)
            if max(float(np.abs(n.value).max()) for n in g.nodes) < 1e6:
                break
        adj = reverse_accumulate(g, len(g.nodes) - 1)
        children = Counter(p for _, par, _ in prog for p in par)
        fanout += any(c > 1 for c in children.values())
        for i in range(m):
            f = lambda t, i=i: evaluate(prog, [t if j == i else xs[j] for j in range(m)]).nodes[-1].value  # noqa: E731
            assert np.allclose(adj[i], complex_step_grad(f, xs[i]), rtol=1e-11, atol=1e-13)
    assert fanout > 150


def test_nplm_gradient():
    """Cor 2: the hand-written NPLM gradient equals complex-step derivatives (relative error < 1e-10) and central
    differences (< 1e-6) on random small instances; embedding rows of words absent from the batch get zero."""
    rng = np.random.default_rng([SEED, 9])
    for _ in range(20):
        P = init_nplm(rng, 3, 4, vocab=7)
        P["W2"] *= 2
        X, Y = rng.integers(0, 6, (30, 2)), rng.integers(0, 7, 30)
        _, G = nplm_loss_grad(P, X, Y)
        cs = param_grad(complex_step_grad, lambda Q: nplm_loss(Q, X, Y), P)
        cd = param_grad(central_diff_grad, lambda Q: nplm_loss(Q, X, Y), P)
        for k in P:
            scale = np.abs(cs[k]).max() + 1e-12
            assert np.abs(G[k] - cs[k]).max() / scale < 1e-10 and np.abs(G[k] - cd[k]).max() / scale < 1e-6
        unused = np.setdiff1d(np.arange(8), X)
        assert np.all(G["E"][unused] == 0)


def test_rnn_gradient():
    """Cor 3 (BPTT): the hand-written RNN gradient equals complex-step derivatives and central differences on
    random batches of padded sentences, and batching does not change it (padding carries no loss)."""
    rng = np.random.default_rng([SEED, 10])
    for _ in range(12):
        P = init_rnn(rng, 3, 4, vocab=6)
        P["Wh"] *= 2
        batch = [list(rng.integers(1, 6, int(rng.integers(1, 7)))) + [0] for _ in range(3)]
        _, G = rnn_loss_grad(P, batch)
        cs = param_grad(complex_step_grad, lambda Q: rnn_loss(Q, batch), P)
        cd = param_grad(central_diff_grad, lambda Q: rnn_loss(Q, batch), P)
        for k in P:
            scale = np.abs(cs[k]).max() + 1e-12
            assert np.abs(G[k] - cs[k]).max() / scale < 1e-10 and np.abs(G[k] - cd[k]).max() / scale < 1e-6
        n = [len(s) for s in batch]
        sep = [rnn_loss_grad(P, [s])[1] for s in batch]
        for k in P:
            assert np.allclose(G[k] * sum(n), sum(g[k] * m for g, m in zip(sep, n)), atol=1e-12)


def test_rnn_tight():
    """Thm 4: ε > 0 for random and trained RNNs; p(</s> | h) ≥ ε at every sampled and held-out state; so
    P(length > n) ≤ (1 − ε)ⁿ — checked on random RNNs with a non-negligible ε, where sampled tails sit under it."""
    rng = np.random.default_rng([SEED, 11])
    for _ in range(10):
        P = init_rnn(rng, 2, 3, vocab=4)
        P["Wo"] *= 0.5
        eps = tightness_epsilon(P["Wo"], P["bo"], 0)
        H = rng.uniform(-1, 1, (5000, 3))
        assert eps > 0 and softmax(H @ P["Wo"] + P["bo"])[:, 0].min() >= eps
        lengths, pmin = sample_sentences(P, 4000, rng, 400, eos=0)
        assert pmin.min() >= eps and lengths.max() < 400
        for n in (5, 10, 20):
            tail = (lengths > n).mean()
            assert tail <= (1 - eps) ** n + 3 * math.sqrt((1 - eps) ** n / 4000) + 1e-9
    R = trained_rnn().params
    eps = tightness_epsilon(R["Wo"], R["bo"], EOS_ID)
    hs = np.concatenate([rnn_states(R, ids) for ids in ids_of("held")[:500]])
    assert eps > 0 and softmax(hs @ R["Wo"] + R["bo"])[:, EOS_ID].min() >= eps
    lengths, pmin = sample_sentences(R, 200, rng, 120)
    assert pmin.min() >= eps and np.median(lengths) < 40


def test_jacobian_bound_random():
    """Thm 5 on random tanh RNNs: measured ‖∂h_T/∂h_k‖ ≤ trajectory bound ≤ σ_max^{T−k}, so σ_max < 1 forces
    geometric vanishing; for linear RNNs ‖W^m‖ ≥ ρ(W)^m (explosion once ρ > 1) and ‖W^m‖^{1/m} → ρ (Gelfand)."""
    rng = np.random.default_rng([SEED, 12])
    for sigma in (0.5, 0.9, 1.5, 3.0):
        for _ in range(10):
            Wh = scaled_matrix(rng, 8, sigma)
            meas, bnd = jacobian_product_norms(Wh, tanh_trajectory(rng, Wh, 40))
            lag = np.arange(1, 41)
            assert np.all(meas <= bnd * (1 + 1e-10)) and np.all(bnd <= sigma**lag * (1 + 1e-10))
    for _ in range(20):
        W = rng.normal(0, 1, (6, 6)) / math.sqrt(6) * rng.uniform(0.8, 2.0)
        rho = np.abs(np.linalg.eigvals(W)).max()
        Wm = np.linalg.matrix_power(W, 400)
        assert np.linalg.norm(Wm, 2) >= rho**400 * (1 - 1e-8)
        assert abs(np.linalg.norm(Wm, 2) ** (1 / 400) - rho) < 0.03 * rho


def test_jacobian_bound_corpus():
    """Thm 5 on the trained RNN: σ_max(W_h) > 1 and ρ(W_h) > 1, so neither the sufficient condition for
    vanishing nor the linear picture applies, and every per-step factor along 200 held-out sentences has norm
    above 1; yet the median measured norm at lag 12 is under a quarter of its lag-1 value while the trajectory
    bound has grown past 1,000 times it — the decay lives in the products (successive factors' expanding
    directions do not line up), where no per-step norm can see it — and no measured norm exceeds its bound."""
    R = trained_rnn().params
    assert np.linalg.norm(R["Wh"], 2) > 1 and np.abs(np.linalg.eigvals(R["Wh"])).max() > 1
    cd = corpus_decay()
    m, b = dict(zip(cd["lags"], cd["measured"])), dict(zip(cd["lags"], cd["bound"]))
    assert cd["violations"] == 0 and m[12] < 0.25 * m[1] and b[12] > 1000 * m[12] and cd["min_step_norm"] > 1


def test_lstm_cell_path():
    """Prop 4: ∂c_T/∂c₀ = Π diag(f_t) exactly when the i, f, g gates have no recurrent weights (o may keep its
    own), on random LSTMs; with those weights scaled by s the residual is O(s)."""
    rng = np.random.default_rng([SEED, 13])
    for _ in range(10):
        H = 4
        P = init_lstm(rng, 3, H, 1.0)
        xs = rng.normal(0, 1, (12, 3))
        c0 = rng.uniform(0.2, 1.0, H)
        h0 = 0.5 * np.tanh(c0)
        Z = zero_gate_recurrence(P, 0.0)
        assert np.allclose(lstm_cell_jacobian(Z, xs, h0, c0), np.diag(lstm_forget_path(Z, xs, h0, c0)), atol=1e-14)
        res = []
        for s in (1e-2, 1e-3):
            Q = zero_gate_recurrence(P, s)
            res.append(np.linalg.norm(lstm_cell_jacobian(Q, xs, h0, c0) - np.diag(lstm_forget_path(Q, xs, h0, c0))))
        assert 5 < res[0] / res[1] < 20


def test_long_lag_memory():
    """Prop 4 in practice, on a bit to be remembered across 60 noise symbols: at initialization the gradient
    reaching the bit is orders of magnitude larger for an LSTM with forget bias 3 than for one with bias 1 or for
    the Elman network (median over 12 initializations); after 600 Adam steps only the bias-3 LSTM has learned."""
    c = {k: np.median(v) for k, v in long_lag_census().items()}
    assert c["lstm-3"] > 30 * c["lstm-1"] and c["lstm-3"] > 30 * c["elman"]
    t = long_lag_training()
    assert t["lstm-3"]["accuracy"] > 0.95 and t["lstm-1"]["accuracy"] < 0.6 and t["elman"]["accuracy"] < 0.6


def test_softmax_bottleneck():
    """Thm 6: a softmax over a d-dimensional state has a log-probability matrix of rank ≤ d + 2 (and rank d + 2
    generically), on random instances and for the trained NPLM and RNN; a two-component mixture of softmaxes is
    not bounded that way; KN's matrix over the same contexts has far higher rank."""
    rng = np.random.default_rng([SEED, 14])
    for _ in range(20):
        d, N, K = int(rng.integers(1, 6)), 40, 30
        Hs, W, b = rng.normal(0, 1, (N, d)), rng.normal(0, 1, (d, K)), rng.normal(0, 1, K)
        assert numerical_rank(log_softmax(Hs @ W + b)) == d + 2
        H2, W2 = rng.normal(0, 1, (N, d)), rng.normal(0, 1, (d, K))
        mix = np.log(0.5 * softmax(Hs @ W + b) + 0.5 * softmax(H2 @ W2))
        assert numerical_rank(mix) > d + 2
    s = bottleneck_summary()
    assert s["rank_nplm"] <= s["h"] + 2 and s["rank_rnn"] <= RNN_CFG["h"] + 2 and s["rank_kn"] > 4 * (s["h"] + 2)


def test_neural_beats_kn_in_domain():
    """§6: in domain, in honest bits per character, RNN < NPLM < KN, each gap many paired standard errors; mixing
    with KN helps the NPLM; dynamic evaluation helps the RNN; the model ranking is the same per token."""
    r = {name: comparison_row(name, "held") for name in COMPARED}
    assert r["rnn"]["bpc"] < r["nplm"]["bpc"] < r["kn"]["bpc"]
    for name in ("nplm", "rnn", "rnn-dyn"):
        assert r[name]["bpc_minus_kn"] < -5 * r[name]["bpc_minus_kn_se"]
    assert r["nplm+kn"]["bpc"] < r["nplm"]["bpc"] and r["rnn-dyn"]["bpc"] < r["rnn"]["bpc"]
    order_bpc = sorted(COMPARED, key=lambda k: r[k]["bpc"])
    assert order_bpc == sorted(COMPARED, key=lambda k: r[k]["nats"])
    assert abs(r["kn"]["nats"] - 2.8551) < 1e-4 and abs(r["kn"]["bpc"] - 0.6371) < 1e-4  # Topic 4's numbers


def test_dynamic_evaluation_is_a_code():
    """§6: dynamic evaluation as an arithmetic code (Topic 4 Thm 4 with the model updated after every sentence):
    the decoder replays the updates and recovers EDGAR and the first 150 held-out sentences exactly; every code is
    within 2 bits of −log₂ Q̃ and equals the table's open NLL to 1e-6 relative; so does the static RNN's."""
    for name, test, n in (("rnn-dyn", "edgar", None), ("rnn", "edgar", None), ("rnn-dyn", "held", 150)):
        c, m = corpus_code(name, test, n), rnn_coder(name)
        assert pce.sents_from_symbols(pce.arithmetic_decode(m, c)) == list(corpus()[test][:n])
        assert pce.code_bounds_hold(c) and c.length - c.quantized_bits < 2
        if n is None:
            assert abs(c.model_bits * LN2 - sentence_table(name, test)[:, 3].sum()) < 1e-6 * c.model_bits
    c = corpus_code("rnn-dyn", "held")
    assert abs(c.model_bits * LN2 - sentence_table("rnn-dyn", "held")[:, 3].sum()) < 1e-6 * c.model_bits
    assert c.length - c.quantized_bits < 2


def test_finance_findings():
    """§7 on EDGAR: per token, both static networks beat KN and the uniform guess (they learned a <unk> rate; KN's
    is its uniform floor), and the NPLM beats the RNN — the in-domain order reversed; in-domain training hurts
    out of domain (at the dev-selected epoch EDGAR is worse than after the first, while dev is better); dynamic
    evaluation cuts the RNN's EDGAR loss by more than 3 nats
    per token; the spelling of OOV words alone costs more bits per character than lzma's whole code, so no model
    over this vocabulary and speller beats lzma; a PPM-style cache brings spelling below lzma, and the best model
    with it still loses."""
    f = finance_summary()
    for name in ("nplm", "rnn"):
        assert f[f"{name}_edgar_nats"] < f["uniform_nats"] < f["kn_edgar_nats"]
        h = (trained_nplm() if name == "nplm" else trained_rnn()).history
        assert h[-1]["edgar"] > h[0]["edgar"] and h[-1]["dev"] < h[0]["dev"]
    assert f["nplm_edgar_nats"] < f["rnn_edgar_nats"] and f["nplm_edgar_bpc"] < f["rnn_edgar_bpc"]
    assert f["rnn_held_bpc"] < f["nplm_held_bpc"]
    assert f["rnn-dyn_edgar_nats"] < f["rnn_edgar_nats"] - 3
    for name in COMPARED:
        assert f[f"{name}_edgar_bpc"] > f["spelled_bpc_edgar"] > f["lzma_edgar_bpc"]
    assert f["cache_spelled_bpc_edgar"] < f["lzma_edgar_bpc"] < f["rnn_dyn_cache_bpc_edgar"]
    assert abs(f["lzma_edgar_bpc"] - 2.908) < 1e-3  # Topic 4's number


def test_viz_payload():
    """The lab's data: every sentence's per-token bits are positive and sum to its model totals; sensitivities
    are finite; the through-time curves have one value per lag; σ_max and ρ match the trained RNN."""
    p = viz_payload()
    assert len(p["sentences"]) == len(LAB_SENTENCES)
    for s, (test, words) in zip(p["sentences"], LAB_SENTENCES):
        assert len(s["tokens"]) == len(s["words"]) + 1 == len(s["sensitivity"]) == len(s["sensitivity_bound"])
        assert all(len(row) == i + 1 for i, row in enumerate(s["sensitivity"]))
        for m, b in zip(s["sensitivity"], s["sensitivity_bound"]):
            assert all(0 < x <= y * (1 + 1e-3) for x, y in zip(m, b))  # Thm 5, through the 4-figure rounding
        for tok in s["tokens"]:
            assert all(v > 0 for v in tok["bits"].values())
        j = corpus()[test].index(words)
        for name in LAB_MODELS:  # the bars sum to the sentence's closed NLL in bits
            total = sum(tok["bits"][name] for tok in s["tokens"])
            assert abs(total - sentence_table(name, test)[j, 0] / LN2) < 0.01
    sens, bound = loss_sensitivity(trained_rnn().params, ids_of("held")[0])
    assert all(x <= y * (1 + 1e-12) for r, b in zip(sens, bound) for x, y in zip(r, b))
    tt = p["through_time"]
    assert all(len(row) == THROUGH_LAGS for row in tt["tanh"] + tt["linear"])
    assert p["sigma_max"] > 1 and all(x > 1 for s in p["sentences"] for x in s["step_norms"])
    L = p["long_lag"]["models"]
    assert L["lstm-3"]["log10_reach"] > L["lstm-1"]["log10_reach"] > L["elman"]["log10_reach"]


VIZ_JSON = Path(__file__).resolve().parents[2] / "src" / "components" / "viz" / "data" / "neural-lm-lab.json"


def test_viz_mirror_matches_payload():
    """The Viz ↔ Python invariant, enforced: the lab's data file equals viz_payload() exactly.
    Regenerate with json.dumps(viz_payload(), separators=(',', ':')) after changing any number here."""
    import json

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
    print("\nSummary (train generate_corpus(300, 0), dev generate_corpus(40, 2), held out generate_corpus(100, 1), EDGAR):")
    for k, v in summary().items():
        print(f"  {k:36} {v:.4g}" if isinstance(v, float) and 0 < abs(v) < 1e-3 else
              f"  {k:36} {v:.4f}" if isinstance(v, float) else f"  {k:36} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_harness())
