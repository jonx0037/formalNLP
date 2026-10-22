import * as d3 from 'd3';
import { DOMAINS, SIBLING_SITES } from '../../../data/domains';

export type Theme = 'light' | 'dark';

// One color per key in [...DOMAINS, ...SIBLING_SITES] order. The nine NLP
// domains get an earthy palette tuned to Marginalia (oxblood and gold lead);
// the four sister sites get neutral slates so cross-site nodes read as
// "external". Each theme has its own palette, and every entry clears the 3:1
// WCAG non-text contrast bar against that theme's background
// (tests/branding.test.ts).
export const DOMAIN_PALETTE: Record<Theme, string[]> = {
  light: [
    '#9F1D35', '#A16207', '#2F5D62', '#5B4B8A', '#C2410C', '#4D7C0F', '#1E4E79', '#8C5A3C', '#57534E',
    '#78716C', '#6B7280', '#57606E', '#475569',
  ],
  dark: [
    '#E0697C', '#D9A441', '#5FB3B3', '#A99BE0', '#F08A5D', '#9CC45A', '#74A9DE', '#C9A27E', '#B8B2AB',
    '#D1D5DB', '#A8B0BD', '#8E99A8', '#7F8EA3',
  ],
};

const scales: Record<Theme, d3.ScaleOrdinal<string, string>> = {
  light: d3.scaleOrdinal<string, string>().domain([...DOMAINS, ...SIBLING_SITES]).range(DOMAIN_PALETTE.light),
  dark: d3.scaleOrdinal<string, string>().domain([...DOMAINS, ...SIBLING_SITES]).range(DOMAIN_PALETTE.dark),
};

export function domainColorScaleFor(theme: Theme): d3.ScaleOrdinal<string, string> {
  return scales[theme];
}

// Light-theme scale, for non-interactive or SSR contexts.
export const domainColorScale = scales.light;
