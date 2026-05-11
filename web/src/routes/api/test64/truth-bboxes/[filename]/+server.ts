import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const PUT: RequestHandler = async ({ fetch, params, request }) => {
  const body = await request.text();
  const upstream = await fetch(backendUrl(`/test64/truth-bboxes/${encodeURIComponent(params.filename)}`), {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body
  });
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
