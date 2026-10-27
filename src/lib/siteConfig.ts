// Site-wide constants. PRELAUNCH is the launch switch: while true, every page
// carries <meta name="robots" content="noindex, nofollow"> and robots.txt
// disallows all crawling. Flipped to false at launch (2026-10-27), once all 28
// topics + the capstone were published (docs/plans/formalnlp-site-design.md §10).
export const PRELAUNCH = false;

// The apex 308-redirects to www (as on every formal* site), so www is canonical.
export const SITE_URL = 'https://www.formalnlp.com';
