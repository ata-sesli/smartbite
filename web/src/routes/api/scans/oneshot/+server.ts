import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { backendUrl, readResponse } from '$lib/server/backend';

export const POST: RequestHandler = async ({ request, fetch }) => {
  const contentType = request.headers.get('content-type') || '';
  const init = {
    method: 'POST',
    headers: contentType ? { 'content-type': contentType } : undefined,
    body: request.body,
    duplex: 'half'
  } as RequestInit & { duplex: 'half' };
  const upstream = await fetch(backendUrl('/scans/oneshot'), init);

  const payload = await readResponse(upstream);
  return json((payload ?? {}) as Record<string, unknown>, { status: upstream.status });
};
