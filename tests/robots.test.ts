import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { robotsTxt } from '../src/lib/robots';
import { PRELAUNCH, SITE_URL } from '../src/lib/siteConfig';

test('prelaunch disallows everything', () => {
  assert.equal(robotsTxt(true, SITE_URL), 'User-agent: *\nDisallow: /\n');
});

test('launch allows crawling', () => {
  assert.equal(robotsTxt(false, SITE_URL), 'User-agent: *\nAllow: /\n');
});

test('the shipped flag is prelaunch and the site URL matches astro.config', () => {
  assert.equal(PRELAUNCH, true);
  assert.equal(SITE_URL, 'https://formalnlp.com');
  assert.match(readFileSync('astro.config.mjs', 'utf8'), new RegExp(`site: '${SITE_URL}'`));
});

test('BaseLayout emits noindex, nofollow under PRELAUNCH', () => {
  const base = readFileSync('src/layouts/BaseLayout.astro', 'utf8');
  assert.match(base, /PRELAUNCH && <meta name="robots" content="noindex, nofollow" \/>/);
});
