import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const GET: RequestHandler = async ({ fetch, url }) => {
  const search = new URLSearchParams();
  const includeSuccess = url.searchParams.get('include_success');
  if (includeSuccess) {
    search.set('include_success', includeSuccess);
  }
  const upstream = await fetch(backendUrl('/test64/manual-crop-recognition-review', search));
  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
