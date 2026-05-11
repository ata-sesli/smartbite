import type { RequestHandler } from './$types';
import { backendUrl } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch, url }) => {
  const search = new URLSearchParams();
  const runId = url.searchParams.get('run_id');
  const path = url.searchParams.get('path');
  if (runId) {
    search.set('run_id', runId);
  }
  if (path) {
    search.set('path', path);
  }

  const upstream = await fetch(backendUrl('/test64/manual-crop-recognition-review/assets', search));
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'content-type': upstream.headers.get('content-type') ?? 'application/octet-stream',
      'cache-control': 'public, max-age=60'
    }
  });
};
