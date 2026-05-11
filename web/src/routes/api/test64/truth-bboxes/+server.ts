import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch }) => {
  const upstream = await fetch(backendUrl('/test64/truth-bboxes'));
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
