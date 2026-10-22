import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { execGrep } from './helpers';

test('the copied site shell is present', () => {
  for (const p of ['src/layouts/BaseLayout.astro', 'src/layouts/TopicLayout.astro',
                   'src/pages/index.astro', 'src/pages/paths.astro', 'src/content.config.ts',
                   'src/components/viz/CurriculumGraph.tsx', 'scripts/audit-cross-site-links.mjs']) {
    assert.ok(existsSync(p), p);
  }
});

test('no RAG-only artifacts were copied', () => {
  for (const p of ['src/components/ui/PipelineDiagram.astro', 'src/pages/architectures.astro',
                   'src/data/architectures.ts', 'src/data/pipeline.ts']) {
    assert.equal(existsSync(p), false, p);
  }
  const labs = readdirSync('src/components/viz').filter((f) => f.endsWith('Laboratory.tsx'));
  assert.deepEqual(labs, []);
});

test('no copied source references pipelineStage, modality, or the architectures page', () => {
  assert.deepEqual(execGrep(/pipelineStage|modality|PipelineDiagram|architectures/, 'src'), []);
});

test('astro site points at formalnlp.com', () => {
  assert.match(readFileSync('astro.config.mjs', 'utf8'), /site: 'https:\/\/formalnlp\.com'/);
});
