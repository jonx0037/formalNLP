import type { APIRoute } from 'astro';
import { robotsTxt } from '../lib/robots';
import { PRELAUNCH, SITE_URL } from '../lib/siteConfig';

export const GET: APIRoute = () =>
  new Response(robotsTxt(PRELAUNCH, SITE_URL), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
