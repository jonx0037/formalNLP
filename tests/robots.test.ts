import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { robotsTxt } from '../src/lib/robots';
import { PRELAUNCH, SITE_URL } from '../src/lib/siteConfig';

test('prelaunch disallows everything', () => {
  assert.equal(robotsTxt(true, SITE_URL), 'User-agent: *\nDisallow: /\n');
});

test('launch allows crawling', () => {
  assert.equal(
    robotsTxt(false, SITE_URL),
    'User-agent: *\nAllow: /\n\nSitemap: https://www.formalnlp.com/sitemap-index.xml\n',
  );
});

test('the shipped flag is launched and the site URL (www) matches astro.config', () => {
  assert.equal(PRELAUNCH, false);
  assert.equal(SITE_URL, 'https://www.formalnlp.com');
  assert.match(readFileSync('astro.config.mjs', 'utf8'), new RegExp(`site: '${SITE_URL}'`));
});

test('BaseLayout emits noindex, nofollow under PRELAUNCH', () => {
  const base = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
  assert.match(base, /PRELAUNCH && <meta name="robots" content="noindex, nofollow" \/>/);
});

test('every page carries the 1200x630 social card as an absolute og:image', () => {
  const base = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
  assert.match(base, /ogImage = '\/og-default\.png'/);
  assert.match(base, /<meta property="og:image" content=\{ogImageURL\} \/>/);
  const png = readFileSync('public/og-default.png');
  assert.equal(png.readUInt32BE(16), 1200);
  assert.equal(png.readUInt32BE(20), 630);
});
