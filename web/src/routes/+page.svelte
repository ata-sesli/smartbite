<script lang="ts">
  type JsonRecord = Record<string, unknown>;

  let file: File | null = null;
  let qrCode = '';
  let userId = '';
  let metadata = '{"name":"","brand":"","category":""}';

  let scanIdInput = '';
  let correctionDate = '';
  let correctionReason = 'manual review resolved';

  let expiryFilterStatus = '';
  let expiryFilterUser = '';

  let loading = false;
  let health: JsonRecord | null = null;
  let metrics: JsonRecord | null = null;
  let scanCreateResult: JsonRecord | null = null;
  let scanResult: JsonRecord | null = null;
  let expiryList: JsonRecord | null = null;
  let alertsResult: JsonRecord | null = null;
  let error = '';

  function clearError() {
    error = '';
  }

  function handleFileChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    file = target?.files?.[0] ?? null;
  }

  function errorMessage(err: unknown): string {
    return err instanceof Error ? err.message : 'Unexpected error';
  }

  async function handleResponse(res: Response): Promise<JsonRecord> {
    const data = (await res.json().catch(() => ({}))) as JsonRecord;
    if (!res.ok) {
      const detail =
        typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`;
      throw new Error(detail);
    }
    return data;
  }

  async function checkHealth() {
    clearError();
    try {
      health = await handleResponse(await fetch('/api/health'));
    } catch (err) {
      error = errorMessage(err);
    }
  }

  async function loadMetrics() {
    clearError();
    try {
      metrics = await handleResponse(await fetch('/api/admin/metrics'));
    } catch (err) {
      error = errorMessage(err);
    }
  }

  async function createScan() {
    clearError();
    if (!file) {
      error = 'Please select an image file.';
      return;
    }

    loading = true;
    try {
      const formData = new FormData();
      formData.set('image', file);
      formData.set('qr_code', qrCode);
      formData.set('user_id', userId);
      if (metadata.trim()) {
        JSON.parse(metadata);
        formData.set('metadata', metadata);
      }

      scanCreateResult = await handleResponse(
        await fetch('/api/scans', {
          method: 'POST',
          body: formData
        })
      );
      const createdScanId = scanCreateResult?.scan_id;
      if (typeof createdScanId === 'string') {
        scanIdInput = createdScanId;
      }
    } catch (err) {
      error = errorMessage(err);
    } finally {
      loading = false;
    }
  }

  async function getScan() {
    clearError();
    if (!scanIdInput.trim()) {
      error = 'Scan ID is required.';
      return;
    }

    try {
      scanResult = await handleResponse(await fetch(`/api/scans/${scanIdInput.trim()}`));
    } catch (err) {
      error = errorMessage(err);
    }
  }

  async function patchScan() {
    clearError();
    if (!scanIdInput.trim()) {
      error = 'Scan ID is required for manual correction.';
      return;
    }
    if (!correctionReason.trim()) {
      error = 'Correction reason is required.';
      return;
    }

    try {
      const payload = {
        parsed_date: correctionDate || null,
        reason: correctionReason
      };
      const res = await fetch(`/api/scans/${scanIdInput.trim()}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const corrected = await handleResponse(res);
      scanResult = {
        ...(scanResult ?? {}),
        correction: corrected
      };
    } catch (err) {
      error = errorMessage(err);
    }
  }

  async function listExpiry() {
    clearError();
    try {
      const params = new URLSearchParams();
      if (expiryFilterStatus) params.set('status', expiryFilterStatus);
      if (expiryFilterUser) params.set('user_id', expiryFilterUser);
      const query = params.toString();
      const url = query ? `/api/expiry?${query}` : '/api/expiry';
      expiryList = await handleResponse(await fetch(url));
    } catch (err) {
      error = errorMessage(err);
    }
  }

  async function processAlerts() {
    clearError();
    try {
      alertsResult = await handleResponse(
        await fetch('/api/alerts/process', {
          method: 'POST'
        })
      );
    } catch (err) {
      error = errorMessage(err);
    }
  }
</script>

<main>
  <h1>SmartBite Minimal Web Console</h1>
  <p>Compatibility layer for the current SmartBite backend API.</p>

  {#if error}
    <section>
      <p><strong>Error:</strong> {error}</p>
    </section>
  {/if}

  <section>
    <h2>System</h2>
    <div class="grid two">
      <button on:click={checkHealth}>Check Health</button>
      <button class="secondary" on:click={loadMetrics}>Load Metrics</button>
    </div>
    {#if health}
      <pre>{JSON.stringify(health, null, 2)}</pre>
    {/if}
    {#if metrics}
      <pre>{JSON.stringify(metrics, null, 2)}</pre>
    {/if}
  </section>

  <section>
    <h2>Create Scan</h2>
    <div class="grid two">
      <label>
        QR Code
        <input bind:value={qrCode} placeholder="qr_123" />
      </label>
      <label>
        User ID
        <input bind:value={userId} placeholder="user_001" />
      </label>
    </div>
    <label>
      Image
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        on:change={handleFileChange}
      />
    </label>
    <label>
      Metadata JSON (optional)
      <textarea bind:value={metadata}></textarea>
    </label>
    <button on:click={createScan} disabled={loading}>{loading ? 'Uploading...' : 'Create Scan'}</button>
    {#if scanCreateResult}
      <pre>{JSON.stringify(scanCreateResult, null, 2)}</pre>
    {/if}
  </section>

  <section>
    <h2>Scan Details</h2>
    <div class="grid two">
      <label>
        Scan ID
        <input bind:value={scanIdInput} placeholder="uuid" />
      </label>
      <div class="small">Use scan ID from the create response.</div>
    </div>
    <div class="grid two">
      <button on:click={getScan}>Get Scan</button>
      <button class="secondary" on:click={listExpiry}>Refresh Expiry List</button>
    </div>
    {#if scanResult}
      <pre>{JSON.stringify(scanResult, null, 2)}</pre>
    {/if}
  </section>

  <section>
    <h2>Manual Correction</h2>
    <div class="grid two">
      <label>
        Corrected Parsed Date
        <input type="date" bind:value={correctionDate} />
      </label>
      <label>
        Reason
        <input bind:value={correctionReason} />
      </label>
    </div>
    <button class="danger" on:click={patchScan}>Apply Correction</button>
  </section>

  <section>
    <h2>Expiry List</h2>
    <div class="grid two">
      <label>
        Status Filter
        <select bind:value={expiryFilterStatus}>
          <option value="">(all)</option>
          <option value="safe">safe</option>
          <option value="expiring_soon">expiring_soon</option>
          <option value="expired">expired</option>
          <option value="manual_review_required">manual_review_required</option>
        </select>
      </label>
      <label>
        User Filter
        <input bind:value={expiryFilterUser} placeholder="user_001" />
      </label>
    </div>
    <button on:click={listExpiry}>Load Expiry</button>
    {#if expiryList}
      <pre>{JSON.stringify(expiryList, null, 2)}</pre>
    {/if}
  </section>

  <section>
    <h2>Alerts</h2>
    <button on:click={processAlerts}>Process Alerts</button>
    {#if alertsResult}
      <pre>{JSON.stringify(alertsResult, null, 2)}</pre>
    {/if}
  </section>
</main>
