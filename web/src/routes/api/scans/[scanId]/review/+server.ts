import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch, params }) => {
  const upstream = await fetch(backendUrl(`/scans/${params.scanId}/review`));
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};

export const PUT: RequestHandler = async ({ fetch, params, request }) => {
  const body = await request.json();
  const upstream = await fetch(backendUrl(`/scans/${params.scanId}/review`), {
    method: 'PUT',
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
