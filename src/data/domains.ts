// Single source of truth for formalNLP's nine curriculum domains. The content
// schema, curriculum tracks, graph colors, and tests all import from here, so
// a domain is added or renamed in exactly one place.
export const DOMAINS = [
  'text-as-data',
  'language-modeling',
  'representations',
  'attention-transformers',
  'decoding-generation',
  'classification-sentiment',
  'extraction-knowledge-graphs',
  'text-mining',
  'nlp-evaluation',
] as const;

export type Domain = (typeof DOMAINS)[number];

export const DOMAIN_LABELS: Record<Domain, string> = {
  'text-as-data': 'Text as Data',
  'language-modeling': 'Language Modeling',
  'representations': 'Representations',
  'attention-transformers': 'Attention & Transformers',
  'decoding-generation': 'Decoding & Generation',
  'classification-sentiment': 'Classification & Sentiment',
  'extraction-knowledge-graphs': 'Information Extraction & Knowledge Graphs',
  'text-mining': 'Text Mining',
  'nlp-evaluation': 'Evaluation',
};

// The four live formal* siblings formalNLP links to. formalNLP links UP into
// formalML and formalRAG and DOWN into the calculus/statistics foundations.
export const SIBLING_SITES = ['formalml', 'formalcalculus', 'formalstatistics', 'formalrag'] as const;

export type SiblingSite = (typeof SIBLING_SITES)[number];
