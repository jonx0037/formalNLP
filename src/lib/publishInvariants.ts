// Publishing a topic touches three places that must agree: the MDX `status`,
// the curriculum-graph node `status`, and the `planned` titles in curriculum.ts.
// If they drift, /paths shows a published topic as an unclickable planned node,
// or the cross-site synthesizer emits an edge to a node that does not exist.
// Returns one human-readable message per violation (empty = consistent).

export interface TopicMeta {
  slug: string;
  status: string;
}

export interface GraphNodeMeta {
  id: string;
  label: string;
  status: string;
}

export function publishInvariantErrors(
  topics: TopicMeta[],
  graph: { nodes: GraphNodeMeta[] },
  plannedTitles: string[],
): string[] {
  const errors: string[] = [];
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const publishedSlugs = new Set(topics.filter((t) => t.status === 'published').map((t) => t.slug));
  const planned = new Set(plannedTitles);

  for (const t of topics) {
    const node = nodes.get(t.slug);
    if (!node) {
      errors.push(`MDX "${t.slug}" has no node in curriculum-graph.json (slug must equal node id)`);
    } else if (t.status === 'published' && node.status !== 'published') {
      errors.push(`MDX "${t.slug}" is published but its graph node is still "${node.status}"`);
    }
  }
  for (const n of graph.nodes) {
    if (n.status === 'published' && !publishedSlugs.has(n.id)) {
      errors.push(`graph node "${n.id}" is published but has no published MDX file`);
    }
    if (n.status === 'published' && planned.has(n.label)) {
      errors.push(`"${n.label}" is published but still listed in curriculum.ts planned`);
    }
  }
  return errors;
}
