import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import matter from 'gray-matter';
import { publishInvariantErrors } from '../src/lib/publishInvariants';
import { tracks } from '../src/data/curriculum';

const graph = {
  nodes: [
    { id: 'a', label: 'Topic A', status: 'published' },
    { id: 'b', label: 'Topic B', status: 'planned' },
  ],
};

test('a consistent publish yields no errors', () => {
  assert.deepEqual(publishInvariantErrors([{ slug: 'a', status: 'published' }], graph, ['Topic B']), []);
});

test('an MDX slug with no graph node is an error', () => {
  assert.match(publishInvariantErrors([{ slug: 'zzz', status: 'draft' }], graph, ['Topic B']).join(), /zzz/);
});

test('published in MDX but planned in the graph is an error', () => {
  const errs = publishInvariantErrors(
    [{ slug: 'a', status: 'published' }, { slug: 'b', status: 'published' }], graph, ['Topic B']);
  assert.ok(errs.some((e) => e.includes('"b"') && e.includes('graph')), errs.join('\n'));
});

test('published in the graph without a published MDX file is an error', () => {
  assert.ok(publishInvariantErrors([], graph, ['Topic B']).some((e) => e.includes('"a"')));
});

test('a published topic still listed as planned is an error', () => {
  const errs = publishInvariantErrors([{ slug: 'a', status: 'published' }], graph, ['Topic A', 'Topic B']);
  assert.ok(errs.some((e) => e.includes('Topic A')), errs.join('\n'));
});

test('the repository satisfies the publish invariants', () => {
  const dir = 'src/content/topics';
  const topics = readdirSync(dir).filter((f) => f.endsWith('.mdx')).map((f) => ({
    slug: f.replace(/\.mdx$/, ''),
    status: String(matter(readFileSync(`${dir}/${f}`, 'utf8')).data.status),
  }));
  const realGraph = JSON.parse(readFileSync('src/data/curriculum-graph.json', 'utf8'));
  assert.deepEqual(publishInvariantErrors(topics, realGraph, tracks.flatMap((t) => [...t.planned])), []);
});
