import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execGrep } from './helpers';
import { domainColorScale, domainColorScaleFor, DOMAIN_PALETTE } from '../src/components/viz/shared/colorScales';
import { contrastRatio } from '../src/lib/contrast';
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

test('domainColorScale covers every domain plus four siblings, in palette order', () => {
  assert.deepEqual(domainColorScale.domain(), [...DOMAINS, ...SIBLING_SITES]);
  assert.deepEqual(domainColorScale.range(), DOMAIN_PALETTE.light);
});

test('the dark scale serves the dark palette for the same domains', () => {
  const dark = domainColorScaleFor('dark');
  assert.deepEqual(dark.domain(), [...DOMAINS, ...SIBLING_SITES]);
  assert.equal(dark('text-as-data'), '#E0697C');
  assert.equal(domainColorScaleFor('light')('text-as-data'), '#9F1D35');
});

test('every graph color clears 3:1 against its theme background (WCAG non-text)', () => {
  assert.equal(DOMAIN_PALETTE.light.length, 13);
  assert.equal(DOMAIN_PALETTE.dark.length, 13);
  for (const c of DOMAIN_PALETTE.light) assert.ok(contrastRatio(c, '#FBF8F1') >= 3, `light ${c}`);
  for (const c of DOMAIN_PALETTE.dark) assert.ok(contrastRatio(c, '#1A1614') >= 3, `dark ${c}`);
});

test('both graphs follow the theme', () => {
  for (const f of ['src/components/viz/CurriculumGraph.tsx', 'src/components/viz/DAGGraph.tsx']) {
    const src = readFileSync(f, 'utf8');
    assert.match(src, /useTheme\(\)/, f);
    assert.match(src, /domainColorScaleFor\(theme\)/, f);
  }
});
