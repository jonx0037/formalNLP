import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SITES, SELF, CROSS_SITE_FIELDS, parseField, reciprocalField } from '../scripts/lib/crossSite.mjs';
import { synthesizeCrossSiteNodes, humanReadableFromSlug } from '../src/lib/synthesizeCrossSiteNodes';

test('five sites, formalnlp is self', () => {
  assert.deepEqual(Object.keys(SITES).sort(),
    ['formalcalculus', 'formalml', 'formalnlp', 'formalrag', 'formalstatistics']);
  assert.equal(SELF, 'formalnlp');
  assert.equal(SITES.formalnlp.envVar, null);
  assert.equal(SITES.formalrag.url, 'https://formalrag.com');
  assert.equal(SITES.formalrag.envVar, 'FORMAL_RAG_PATH');
  assert.equal(SITES.formalml.defaultPath, '../formalML');
  assert.equal(CROSS_SITE_FIELDS.length, 10);
});

test('parses fields for every site, including the two newest', () => {
  assert.deepEqual(parseField('formalragPrereqs'), { targetSite: 'formalrag', direction: 'Prereqs' });
  assert.deepEqual(parseField('formalnlpConnections'), { targetSite: 'formalnlp', direction: 'Connections' });
  assert.deepEqual(parseField('formalmlPrereqs'), { targetSite: 'formalml', direction: 'Prereqs' });
  assert.equal(parseField('formalxyzPrereqs'), null);
  assert.equal(parseField('formalragprereqs'), null);
});

test('reciprocal flips direction and names the source site', () => {
  assert.equal(reciprocalField('formalnlp', 'Prereqs'), 'formalnlpConnections');
  assert.equal(reciprocalField('formalrag', 'Connections'), 'formalragPrereqs');
});

test('synthesizeCrossSiteNodes resolves formalrag prereqs to formalrag.com', () => {
  const ref = (topic: string, site: string) => ({ topic, site, relationship: 'x'.repeat(40) });
  const { nodes, edges } = synthesizeCrossSiteNodes([
    { id: 'word2vec-implicit-factorization',
      data: { formalragPrereqs: [ref('pmi-retrieval-value', 'formalrag')] as never,
              formalmlPrereqs: [ref('svd', 'formalml')] as never } },
  ]);
  const rag = nodes.find((n) => n.id === 'formalrag/pmi-retrieval-value');
  assert.ok(rag);
  assert.equal(rag.url, 'https://formalrag.com/topics/pmi-retrieval-value');
  assert.equal(rag.domain, 'formalrag');
  assert.equal(nodes.length, 2);
  assert.deepEqual(edges.find((e) => e.source === 'formalrag/pmi-retrieval-value'),
    { source: 'formalrag/pmi-retrieval-value', target: 'word2vec-implicit-factorization' });
});

test('NLP acronyms are upper-cased in synthesized labels', () => {
  assert.equal(humanReadableFromSlug('subword-tokenization-bpe-unigram'), 'Subword Tokenization BPE Unigram');
  assert.equal(humanReadableFromSlug('topic-models-lda'), 'Topic Models LDA');
});
