import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const POST: RequestHandler = async ({ request, fetch }) => {
  const formData = await request.formData();
  const upstream = await fetch(backendUrl('/scans/oneshot'), {
    method: 'POST',
    body: formData
  });

  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
