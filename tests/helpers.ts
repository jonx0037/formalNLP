import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const TEXT_EXTENSIONS = /\.(astro|ts|tsx|mjs|js|css|json|md|mdx|svg)$/;

// Walk `dir` recursively and return `path:line` for every line matching `re`.
// Shared by the scaffold and branding tests to catch leftover formalRAG copy.
export function execGrep(re: RegExp, dir: string): string[] {
  const hits: string[] = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      hits.push(...execGrep(re, path));
    } else if (TEXT_EXTENSIONS.test(entry)) {
      readFileSync(path, 'utf8')
        .split('\n')
        .forEach((line, i) => {
          if (re.test(line)) hits.push(`${path}:${i + 1}: ${line.trim()}`);
        });
    }
  }
  return hits;
}
