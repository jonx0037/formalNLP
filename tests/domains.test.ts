import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DOMAINS, DOMAIN_LABELS, SIBLING_SITES } from '../src/data/domains';

test('nine domains in spec order', () => {
  assert.deepEqual([...DOMAINS], [
    'text-as-data', 'language-modeling', 'representations', 'attention-transformers',
    'decoding-generation', 'classification-sentiment', 'extraction-knowledge-graphs',
    'text-mining', 'nlp-evaluation',
  ]);
});

test('every domain has its display label', () => {
  assert.deepEqual(DOMAIN_LABELS, {
    'text-as-data': 'Text as Data',
    'language-modeling': 'Language Modeling',
    'representations': 'Representations',
    'attention-transformers': 'Attention & Transformers',
    'decoding-generation': 'Decoding & Generation',
    'classification-sentiment': 'Classification & Sentiment',
    'extraction-knowledge-graphs': 'Information Extraction & Knowledge Graphs',
    'text-mining': 'Text Mining',
    'nlp-evaluation': 'Evaluation',
  });
});

test('four sibling sites, formalrag included', () => {
  assert.deepEqual([...SIBLING_SITES], ['formalml', 'formalcalculus', 'formalstatistics', 'formalrag']);
});
