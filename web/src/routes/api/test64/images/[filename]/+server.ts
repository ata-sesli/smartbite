import type { RequestHandler } from './$types';
import { backendUrl } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch, params }) => {
  const upstream = await fetch(backendUrl(`/test64/images/${encodeURIComponent(params.filename)}`));
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'content-type': upstream.headers.get('content-type') ?? 'application/octet-stream',
      'cache-control': 'public, max-age=60'
    }
  });
};
