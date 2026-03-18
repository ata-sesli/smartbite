# SmartBite Web (SvelteKit)

Minimal web console that is fully compatible with the SmartBite backend API.

## Backend compatibility
The app proxies these backend endpoints through SvelteKit server routes:
- `POST /scans`
- `GET /scans/{scan_id}`
- `PATCH /scans/{scan_id}`
- `GET /expiry`
- `POST /alerts/process`
- `GET /health`
- `GET /admin/metrics`

## Configuration
Set backend URL for proxying:

```bash
export SMARTBITE_BACKEND_URL=http://localhost:8000
```

## Run
```bash
cd web
pnpm install
pnpm dev
```

Then open: `http://localhost:5173`
