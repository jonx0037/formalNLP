import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { contrastRatio } from '../src/lib/contrast';

test('contrastRatio matches WCAG reference values', () => {
  assert.equal(contrastRatio('#000000', '#FFFFFF').toFixed(1), '21.0');
  assert.equal(contrastRatio('#FFFFFF', '#000000').toFixed(1), '21.0');
  assert.equal(contrastRatio('#FFFFFF', '#FFFFFF'), 1);
  assert.equal(contrastRatio('#777777', '#FFFFFF').toFixed(2), '4.48');
});

test('Marginalia text and accents meet WCAG AA (4.5:1) on both backgrounds', () => {
  const pairs = [
    ['#9F1D35', '#FBF8F1'], ['#A16207', '#FBF8F1'], ['#211A17', '#FBF8F1'],
    ['#E0697C', '#1A1614'], ['#D9A441', '#1A1614'], ['#EDE6DA', '#1A1614'],
  ];
  for (const [fg, bg] of pairs) {
    assert.ok(contrastRatio(fg, bg) >= 4.5, `${fg} on ${bg}: ${contrastRatio(fg, bg).toFixed(2)}`);
  }
});

test('global.css carries the Marginalia tokens for light and dark', () => {
  const css = readFileSync('src/styles/global.css', 'utf8');
  const [light, dark] = css.split('[data-theme="dark"]');
  const expectLight = {
    'bg': '#FBF8F1', 'text': '#211A17', 'accent': '#9F1D35', 'accent-secondary': '#A16207',
    'border': '#E6DCCB', 'definition-bg': '#F6E7E8', 'definition-border': '#9F1D35',
    'theorem-bg': '#F7EFDC', 'theorem-border': '#A16207',
  };
  const expectDark = {
    'bg': '#1A1614', 'surface': '#231E1B', 'text': '#EDE6DA', 'text-secondary': '#B3A99B',
    'accent': '#E0697C', 'accent-secondary': '#D9A441', 'border': '#3A322B',
    'definition-bg': '#3A1E24', 'theorem-bg': '#3A2F1A',
  };
  for (const [k, v] of Object.entries(expectLight)) assert.match(light, new RegExp(`--color-${k}: ${v};`), k);
  for (const [k, v] of Object.entries(expectDark)) assert.match(dark, new RegExp(`--color-${k}: ${v};`), k);
  assert.match(light, /--font-serif: 'Crimson Pro Variable'/);
  assert.match(light, /--font-sans: 'Inter Variable'/);
});
