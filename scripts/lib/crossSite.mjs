/**
 * crossSite.mjs — the five formal* sites as data, plus cross-site field parsing.
 *
 * formalNLP is `SELF`; the four siblings are read from adjacent checkouts (or
 * the FORMAL_*_PATH env vars). The field regex matches all five sites — not
 * just the ones formalNLP links to — so the audit also sees `formalnlp*`
 * fields once the siblings add their reciprocals at launch.
 */

export const SELF = 'formalnlp';

export const SITES = {
  formalnlp: { label: 'formalNLP', url: 'https://www.formalnlp.com', envVar: null, defaultPath: '.' },
  formalml: { label: 'formalML', url: 'https://formalml.com', envVar: 'FORMAL_ML_PATH', defaultPath: '../formalML' },
  formalcalculus: { label: 'formalCalculus', url: 'https://formalcalculus.com', envVar: 'FORMAL_CALCULUS_PATH', defaultPath: '../formalCalculus' },
  formalstatistics: { label: 'formalStatistics', url: 'https://formalstatistics.com', envVar: 'FORMAL_STATISTICS_PATH', defaultPath: '../formalStatistics' },
  formalrag: { label: 'formalRAG', url: 'https://formalrag.com', envVar: 'FORMAL_RAG_PATH', defaultPath: '../formalRAG' },
};

export const CROSS_SITE_FIELDS = Object.keys(SITES).flatMap((site) => [
  `${site}Prereqs`,
  `${site}Connections`,
]);

const FIELD_RE = new RegExp(`^(${Object.keys(SITES).join('|')})(Prereqs|Connections)$`);

export function parseField(field) {
  const m = field.match(FIELD_RE);
  if (!m) return null;
  return { targetSite: m[1], direction: m[2] };
}

export function reciprocalField(sourceSite, direction) {
  const flipped = direction === 'Prereqs' ? 'Connections' : 'Prereqs';
  return `${sourceSite}${flipped}`;
}

// Frontmatter keys that look like cross-site arrays but aren't one of the ten
// valid fields (e.g. `formalRagPrereqs`, `formalmlPrereq`). Zod would silently
// strip them and the audit would silently skip them, so both call this.
export function unknownCrossSiteKeys(keys) {
  return keys.filter((k) => /^formal/i.test(k) && !CROSS_SITE_FIELDS.includes(k));
}
