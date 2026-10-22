import * as d3 from 'd3';
import { DOMAINS, SIBLING_SITES } from '../../../data/domains';

// The nine NLP domains get an earthy palette tuned to Marginalia (oxblood and
// gold lead); the four sister sites get neutral grays so cross-site nodes read
// as "external" in the dependency graph. Explicit range so nothing cycles.
export const domainColorScale = d3
  .scaleOrdinal<string, string>()
  .domain([...DOMAINS, ...SIBLING_SITES])
  .range([
    '#9F1D35', // text-as-data
    '#A16207', // language-modeling
    '#2F5D62', // representations
    '#5B4B8A', // attention-transformers
    '#C2410C', // decoding-generation
    '#4D7C0F', // classification-sentiment
    '#1E4E79', // extraction-knowledge-graphs
    '#8C5A3C', // text-mining
    '#57534E', // nlp-evaluation
    '#9ca3af', // formalml
    '#6b7280', // formalcalculus
    '#4b5563', // formalstatistics
    '#374151', // formalrag
  ]);
