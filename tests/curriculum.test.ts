import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DOMAINS } from '../src/data/domains';
import { tracks } from '../src/data/curriculum';

interface Graph {
  nodes: { id: string; label: string; status: string; domain: string }[];
  edges: { source: string; target: string }[];
}
const graph: Graph = JSON.parse(readFileSync('src/data/curriculum-graph.json', 'utf8'));
const hasEdge = (source: string, target: string) =>
  graph.edges.some((e) => e.source === source && e.target === target);

test('29 unique nodes, planned or published, with valid domains, every domain populated', () => {
  assert.equal(graph.nodes.length, 29);
  assert.equal(new Set(graph.nodes.map((n) => n.id)).size, 29);
  assert.ok(graph.nodes.every((n) => n.status === 'planned' || n.status === 'published'));
  assert.ok(graph.nodes.every((n) => (DOMAINS as readonly string[]).includes(n.domain)));
  for (const d of DOMAINS) assert.ok(graph.nodes.some((n) => n.domain === d), d);
});

test('edges reference real nodes and the graph is acyclic', () => {
  const ids = new Set(graph.nodes.map((n) => n.id));
  for (const e of graph.edges) {
    assert.ok(ids.has(e.source), e.source);
    assert.ok(ids.has(e.target), e.target);
  }
  // Kahn's algorithm must consume every node.
  const indegree = new Map([...ids].map((id) => [id, 0]));
  for (const e of graph.edges) indegree.set(e.target, indegree.get(e.target)! + 1);
  const queue = [...ids].filter((id) => indegree.get(id) === 0);
  let consumed = 0;
  while (queue.length) {
    const id = queue.shift()!;
    consumed++;
    for (const e of graph.edges.filter((x) => x.source === id)) {
      indegree.set(e.target, indegree.get(e.target)! - 1);
      if (indegree.get(e.target) === 0) queue.push(e.target);
    }
  }
  assert.equal(consumed, 29);
});

test('the DAG fixes the build order the spec calls out', () => {
  assert.ok(hasEdge('transformer-block-residual-stream', 'contextual-embeddings-masked-lm'));
  assert.ok(hasEdge('knowledge-graph-embeddings', 'capstone-earnings-call-nlp'));
  assert.equal(graph.edges.length, 42);
});

test('tracks cover exactly DOMAINS, in layer order', () => {
  assert.deepEqual(tracks.map((t) => t.domain), [...DOMAINS]);
  assert.deepEqual([...new Set(tracks.map((t) => t.layer))],
    ['Foundations', 'Representations', 'Generation', 'Tasks', 'Evaluation']);
});

test('planned titles match the still-planned graph labels per domain', () => {
  for (const t of tracks) {
    assert.deepEqual([...t.planned],
      graph.nodes.filter((n) => n.domain === t.domain && n.status === 'planned').map((n) => n.label), t.domain);
  }
});

test('layer anchors never collide with domain anchors on /paths', async () => {
  const { layerAnchorId } = await import('../src/data/curriculum');
  const anchors = [...new Set(tracks.map((t) => t.layer))].map(layerAnchorId);
  assert.equal(layerAnchorId('Representations'), 'layer-representations');
  for (const a of anchors) assert.ok(!(DOMAINS as readonly string[]).includes(a), a);
});
