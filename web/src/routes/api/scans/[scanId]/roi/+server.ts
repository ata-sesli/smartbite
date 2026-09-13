import type { RequestHandler } from './$types';
import { backendUrl } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch, params }) => {
  const upstream = await fetch(backendUrl(`/scans/${params.scanId}/roi`));
  const body = await upstream.arrayBuffer();
  return new Response(body, {
    status: upstream.status,
    headers: {
      'content-type': upstream.headers.get('content-type') || 'application/octet-stream'
    }
  });
};
