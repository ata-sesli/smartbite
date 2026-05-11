import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch }) => {
  const upstream = await fetch(backendUrl('/test64/labels'));
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};

export const PUT: RequestHandler = async ({ fetch, request }) => {
  const body = await request.text();
  const upstream = await fetch(backendUrl('/test64/labels'), {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body
  });
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
