// Site-wide constants. PRELAUNCH is the launch switch: while true, every page
// carries <meta name="robots" content="noindex, nofollow"> and robots.txt
// disallows all crawling. Flip it to false only when all 28 topics + the
// capstone are published (see docs/plans/formalnlp-site-design.md §10).
export const PRELAUNCH = true;

export const SITE_URL = 'https://formalnlp.com';
