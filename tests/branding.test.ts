import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execGrep } from './helpers';
import { domainColorScale } from '../src/components/viz/shared/colorScales';
import { DOMAINS, SIBLING_SITES } from '../src/data/domains';

// Identity strings, not sibling mentions: formalRAG may still be *named* as a
// sibling in comments and copy, but never as this site's title, owner, repo,
// tagline, or font.
const RAG_IDENTITY =
  /\| formalRAG|formalRAG\. All rights|jonx0037\/formalRAG|retrieval-augmented|Source Serif 4|(about|on|Browse all) formalRAG|retrieval and RAG/i;

test('no formalRAG identity remains in src/', () => {
  assert.deepEqual(execGrep(RAG_IDENTITY, 'src'), []);
});

test('pages are titled and owned as formalNLP', () => {
  const base = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
  assert.match(base, /\{title\} \| formalNLP/);
  assert.match(base, /The mathematics of language/);
  assert.match(readFileSync('src/components/ui/Footer.astro', 'utf8'), /github\.com\/jonx0037\/formalNLP/);
  for (const f of ['src/pages/index.astro', 'src/components/ui/Nav.astro']) {
    const src = readFileSync(f, 'utf8');
    assert.match(src, /formal<span[^>]*>NLP<\/span>/, f);
    assert.doesNotMatch(src, /formal<span[^>]*>RAG<\/span>/, f);
  }
});

test('domainColorScale covers every domain plus four siblings', () => {
  assert.deepEqual(domainColorScale.domain(), [...DOMAINS, ...SIBLING_SITES]);
  assert.deepEqual(domainColorScale.range(), [
    '#9F1D35', '#A16207', '#2F5D62', '#5B4B8A', '#C2410C', '#4D7C0F', '#1E4E79', '#8C5A3C', '#57534E',
    '#9ca3af', '#6b7280', '#4b5563', '#374151',
  ]);
});
