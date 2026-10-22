// robots.txt body for the current launch state; served by src/pages/robots.txt.ts.
// `_siteUrl` is reserved for a Sitemap: line once a sitemap integration exists.
export function robotsTxt(prelaunch: boolean, _siteUrl: string): string {
  return prelaunch ? 'User-agent: *\nDisallow: /\n' : 'User-agent: *\nAllow: /\n';
}
