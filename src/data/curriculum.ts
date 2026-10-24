import { DOMAIN_LABELS } from './domains';

// Each track declares which curriculum *layer* it belongs to. The /paths page
// reads this to inject layer-section headers above the first track of each
// layer — so adding/removing/reordering tracks here is the single source of
// truth, and the presentation layer derives layer transitions from data.
//
// `planned` lists topic titles on the roadmap that are not yet authored as
// published MDX; it mirrors the node labels in curriculum-graph.json. As a
// topic is published, remove it from `planned` here.
// `dependsOnFormalML` surfaces a "need the ML foundations?" pointer on /paths.
export const tracks = [
  {
    domain: 'text-as-data',
    label: DOMAIN_LABELS['text-as-data'],
    layer: 'Foundations',
    description:
      'Text as a statistical object: the heavy-tailed laws of vocabulary and subword tokenization as a compression problem. The root of the dependency graph.',
    planned: [],
    dependsOnFormalML: false,
  },
  {
    domain: 'language-modeling',
    label: DOMAIN_LABELS['language-modeling'],
    layer: 'Foundations',
    description:
      'Assigning probabilities to sequences: n-gram estimation and smoothing, perplexity as cross-entropy, and the move to neural language models.',
    planned: [
    ],
    dependsOnFormalML: false,
  },
  {
    domain: 'representations',
    label: DOMAIN_LABELS['representations'],
    layer: 'Representations',
    description:
      'Meaning as geometry: co-occurrence statistics and PPMI, word2vec as implicit matrix factorization, GloVe, and contextual embeddings from masked language modeling.',
    planned: [
      'Contextual Embeddings and the Masked-LM Objective',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'attention-transformers',
    label: DOMAIN_LABELS['attention-transformers'],
    layer: 'Representations',
    description:
      'The transformer as mathematics: attention as kernel smoothing, positional encodings as rotations, the residual stream, and empirical scaling laws.',
    planned: [
      'Positional Encodings: Sinusoids, Rotations, and RoPE',
      'The Transformer Block and the Residual Stream',
      'Neural Scaling Laws and Compute-Optimal Training',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'decoding-generation',
    label: DOMAIN_LABELS['decoding-generation'],
    layer: 'Generation',
    description:
      'Turning a distribution into text: decoding as search, temperature and truncation sampling, and whether a language model knows what it does not know.',
    planned: [
      'Decoding as Search: Greedy and Beam Search',
      'Sampling: Temperature, Top-k, and Nucleus',
      'Calibration of Language Models',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'classification-sentiment',
    label: DOMAIN_LABELS['classification-sentiment'],
    layer: 'Tasks',
    description:
      'Labeling documents: generative versus discriminative linear classifiers, lexicon-based financial sentiment, aspect-level sentiment, and low-rank fine-tuning.',
    planned: [
      'Linear Text Classifiers: Generative vs. Discriminative',
      'Lexicon Sentiment and the Loughran–McDonald Dictionary',
      'Aspect-Based Sentiment Analysis',
      'Fine-Tuning and Low-Rank Adaptation',
    ],
    dependsOnFormalML: false,
  },
  {
    domain: 'extraction-knowledge-graphs',
    label: DOMAIN_LABELS['extraction-knowledge-graphs'],
    layer: 'Tasks',
    description:
      'From text to structure: sequence labeling with HMMs and CRFs, named entities, relation extraction under distant supervision, and knowledge-graph embeddings — closing with the earnings-call capstone.',
    planned: [
      'Sequence Labeling: From HMMs to Linear-Chain CRFs',
      'Named Entity Recognition',
      'Relation Extraction and Distant Supervision',
      'Knowledge-Graph Embeddings: TransE and RotatE',
      'Capstone: An Earnings-Call NLP Pipeline',
    ],
    dependsOnFormalML: false,
  },
  {
    domain: 'text-mining',
    label: DOMAIN_LABELS['text-mining'],
    layer: 'Tasks',
    description:
      'Unsupervised structure in corpora: latent Dirichlet allocation as a Bayesian model, and collocations and keywords by likelihood ratios and random walks.',
    planned: [
      'Topic Models: Latent Dirichlet Allocation',
      'Collocations and Keyword Extraction',
    ],
    dependsOnFormalML: true,
  },
  {
    domain: 'nlp-evaluation',
    label: DOMAIN_LABELS['nlp-evaluation'],
    layer: 'Evaluation',
    description:
      'Measuring language systems honestly: n-gram overlap metrics and their limits, and significance testing for paired NLP comparisons.',
    planned: [
      'Overlap Metrics: BLEU and ROUGE',
      'Significance Testing for NLP',
    ],
    dependsOnFormalML: false,
  },
] as const;

// Domain slug → display label, consumed by TopicLayout's domain badge.
export const domainLabelMap: Record<string, string> = DOMAIN_LABELS;

// /paths anchor for a layer header. Prefixed so a layer named like a domain
// ("Representations") never duplicates that domain's track anchor.
export function layerAnchorId(layer: string): string {
  return `layer-${layer.toLowerCase().replace(/\s+/g, '-')}`;
}
