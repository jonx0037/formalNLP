# Launch: search-engine checklist

These steps run on Jonathan's own accounts, after the launch PR (`PRELAUNCH = false`) is deployed. The
canonical host is `https://www.formalnlp.com`, and the apex 308-redirects to it.

## 1. Confirm the deploy is indexable

```bash
curl -s https://www.formalnlp.com/robots.txt
```

Expected: `Allow: /` and `Sitemap: https://www.formalnlp.com/sitemap-index.xml`.

```bash
curl -s https://www.formalnlp.com/ | grep -c 'noindex'
```

Expected: `0`. Then check that `https://www.formalnlp.com/sitemap-index.xml` returns 200 and that
`sitemap-0.xml` lists all 29 topics.

## 2. Google Search Console

1. At search.google.com/search-console, add a **Domain property** for `formalnlp.com`. It covers the apex
   and www.
2. Verify it with the DNS TXT record Google gives you, added at Namecheap (Advanced DNS → TXT, host `@`).
   Leave the A and CNAME records alone.
3. Go to Sitemaps and submit `https://www.formalnlp.com/sitemap-index.xml`.
4. Under URL Inspection, request indexing for `https://www.formalnlp.com/` and
   `https://www.formalnlp.com/paths/`.
5. After about a week, open Pages (indexing coverage). "Page with redirect" on apex URLs is expected.
   "Excluded by noindex" means a stale crawl from before launch, so request indexing again.

## 3. Bing Webmaster Tools

At bing.com/webmasters, choose **Import from Google Search Console**. That carries over the verification
and the sitemap. If you skip the import, verify by DNS and submit the same sitemap URL.

## 4. Link previews

Paste `https://www.formalnlp.com/` and one topic URL into a social-card validator, or a draft post. The
1200×630 card (`public/og-default.png`) should appear with the page's title and abstract.

## 5. After the sibling reciprocal PRs merge

Re-run `pnpm audit:cross-site` with the four siblings on `main`. In the formalNLP row of the summary,
Missing should be 0.
