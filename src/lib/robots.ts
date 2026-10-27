// robots.txt body for the current launch state; served by src/pages/robots.txt.ts.
// At launch it points crawlers at @astrojs/sitemap's index.
export function robotsTxt(prelaunch: boolean, siteUrl: string): string {
  return prelaunch
    ? 'User-agent: *\nDisallow: /\n'
    : `User-agent: *\nAllow: /\n\nSitemap: ${siteUrl}/sitemap-index.xml\n`;
}
