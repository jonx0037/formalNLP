import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { DOMAINS, SIBLING_SITES } from './data/domains';
import { unknownCrossSiteKeys } from '../scripts/lib/crossSite.mjs';

// formalNLP links UP into formalML and formalRAG (representation learning,
// kernels, retrieval geometry) and DOWN into the calculus/statistics
// foundations — so all four live siblings appear in the site enum.
const crossSiteRef = z.object({
  topic: z.string(),
  site: z.enum(SIBLING_SITES),
  relationship: z.string().min(40),
});

const topics = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/topics' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    status: z.enum(['draft', 'review', 'published']),
    difficulty: z.enum(['foundational', 'intermediate', 'advanced']),
    prerequisites: z.array(z.string()).default([]),
    tags: z.array(z.string()),
    // Nine language-centric domains; see src/data/domains.ts.
    domain: z.enum(DOMAINS),
    videoId: z.string().nullable().default(null),
    notebookPath: z.string().nullable().default(null),
    githubUrl: z.string().url().nullable().default(null),
    datePublished: z.coerce.date().optional(),
    dateUpdated: z.coerce.date().optional(),
    estimatedReadTime: z.number().optional(),
    abstract: z.string(),
    // The recurring finance case-study thread (Applied NLP for Finance) + capstone.
    financeCaseStudy: z.boolean().default(false),
    // Honesty annotation: celebrated methods whose guarantees are heuristic
    // (Kneser–Ney's ad hoc discounts, empirical scaling-law fits, BLEU's
    // weak correlation with human judgment).
    rigorFlag: z.string().optional(),
    connections: z
      .array(
        z.object({
          topic: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    references: z
      .array(
        z.object({
          // 'documentation' kept: tokenizer/library docs can be load-bearing
          // references for the code pillar.
          type: z.enum(['paper', 'book', 'course', 'blog', 'video', 'documentation']),
          title: z.string(),
          authors: z.string().optional(),
          year: z.number().optional(),
          url: z.string().url().optional(),
          note: z.string().optional(),
        }),
      )
      .default([]),
    // Cross-site links: Prereqs point at a sibling topic this one builds on;
    // Connections are lateral. Reciprocals on the siblings are batched at launch.
    formalmlPrereqs: z.array(crossSiteRef).optional(),
    formalmlConnections: z.array(crossSiteRef).optional(),
    formalcalculusPrereqs: z.array(crossSiteRef).optional(),
    formalcalculusConnections: z.array(crossSiteRef).optional(),
    formalstatisticsPrereqs: z.array(crossSiteRef).optional(),
    formalstatisticsConnections: z.array(crossSiteRef).optional(),
    formalragPrereqs: z.array(crossSiteRef).optional(),
    formalragConnections: z.array(crossSiteRef).optional(),
  })
    // Keep unknown keys long enough to reject misspelled cross-site arrays
    // (e.g. `formalRagPrereqs`), which Zod would otherwise strip silently.
    .passthrough()
    .superRefine((data, ctx) => {
      for (const key of unknownCrossSiteKeys(Object.keys(data))) {
        ctx.addIssue({ code: 'custom', path: [key], message: `Unknown cross-site field "${key}"` });
      }
    }),
});

export const collections = { topics };
