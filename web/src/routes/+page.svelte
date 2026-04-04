<script lang="ts">
  import { browser } from '$app/environment';
  import { onDestroy, onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type AnalyzeMode = 'upload' | 'camera';
  type NoticeKind = 'error' | 'success' | 'info';
  type RequestState = 'idle' | 'submitting' | 'success' | 'error';

  type Notice = {
    kind: NoticeKind;
    text: string;
  };

  let analyzeMode: AnalyzeMode = 'upload';

  let uploadFile: File | null = null;
  let capturedFile: File | null = null;
  let capturedPreviewUrl = '';

  let metadata = '{"name":"","brand":"","category":""}';

  let scanIdInput = '';
  let correctionDate = '';
  let correctionReason = 'manual review resolved';

  let expiryFilterStatus = '';
  let expiryFilterUser = '';

  let notice: Notice | null = null;

  let analyzeState: RequestState = 'idle';
  let analyzeResult: JsonRecord | null = null;
  let analyzeError = '';
  let lastAnalyzeAt = '';

  let health: JsonRecord | null = null;
  let metrics: JsonRecord | null = null;
  let scanResult: JsonRecord | null = null;
  let expiryList: JsonRecord | null = null;
  let alertsResult: JsonRecord | null = null;

  let lastHealthCheckAt = '';

  let videoElement: HTMLVideoElement | null = null;
  let canvasElement: HTMLCanvasElement | null = null;
  let cameraOpen = false;
  let cameraError = '';
  let availableCameras: MediaDeviceInfo[] = [];
  let selectedCameraId = '';
  let mediaStream: MediaStream | null = null;

  const backendRouteLabel = '/api (SvelteKit proxy)';
  const metadataPlaceholder = '{"name":"","brand":"","category":""}';

  const healthStatus = (): 'unknown' | 'up' | 'down' => {
    if (!health) return 'unknown';
    const raw = health.status;
    if (typeof raw === 'string' && raw.toLowerCase() === 'ok') return 'up';
    return 'up';
  };

  const cameraSupported = (): boolean => {
    return browser && 'mediaDevices' in navigator && typeof navigator.mediaDevices.getUserMedia === 'function';
  };

  function setNotice(kind: NoticeKind, text: string): void {
    notice = { kind, text };
  }

  function clearNotice(): void {
    notice = null;
  }

  function errorMessage(err: unknown): string {
    return err instanceof Error ? err.message : 'Unexpected error';
  }

  function formatNow(): string {
    return new Date().toLocaleString();
  }

  function handleUploadFileChange(event: Event): void {
    const target = event.currentTarget as HTMLInputElement | null;
    uploadFile = target?.files?.[0] ?? null;
  }

  function revokeCapturedPreviewUrl(): void {
    if (capturedPreviewUrl) {
      URL.revokeObjectURL(capturedPreviewUrl);
      capturedPreviewUrl = '';
    }
  }

  function selectAnalyzeFile(): File | null {
    if (analyzeMode === 'upload') {
      return uploadFile;
    }
    return capturedFile;
  }

  function isJsonRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function readNestedString(root: JsonRecord, keyPath: string[]): string | null {
    let cursor: unknown = root;
    for (const key of keyPath) {
      if (!isJsonRecord(cursor) || !(key in cursor)) {
        return null;
      }
      cursor = cursor[key];
    }
    return typeof cursor === 'string' ? cursor : null;
  }

  function extractScanId(payload: JsonRecord | null): string {
    if (!payload) return '';
    const direct = payload.scan_id;
    if (typeof direct === 'string') return direct;
    const nested = readNestedString(payload, ['scan', 'scan_id']);
    return nested ?? '';
  }

  function extractParsedExpiry(payload: JsonRecord | null): string {
    if (!payload) return '-';
    const parsedCandidates: Array<string | null> = [
      typeof payload.parsed_date === 'string' ? payload.parsed_date : null,
      readNestedString(payload, ['result', 'parsed_date']),
      readNestedString(payload, ['parse_result', 'parsed_date']),
      readNestedString(payload, ['scan', 'parsed_date'])
    ];

    for (const value of parsedCandidates) {
      if (value) return value;
    }

    const classificationCandidates: Array<string | null> = [
      typeof payload.expiry_classification === 'string' ? payload.expiry_classification : null,
      readNestedString(payload, ['result', 'expiry_classification'])
    ];

    for (const value of classificationCandidates) {
      if (value) return `classification: ${value}`;
    }

    return '-';
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

  async function refreshCameraDevices(): Promise<void> {
    if (!cameraSupported()) return;

    const devices = await navigator.mediaDevices.enumerateDevices();
    availableCameras = devices.filter((item) => item.kind === 'videoinput');

    if (!availableCameras.length) {
      selectedCameraId = '';
      return;
    }

    if (!availableCameras.some((camera) => camera.deviceId === selectedCameraId)) {
      selectedCameraId = availableCameras[0].deviceId;
    }
  }

  async function startCamera(): Promise<void> {
    clearNotice();
    cameraError = '';

    if (!cameraSupported()) {
      cameraError = 'Camera is not supported in this browser.';
      return;
    }

    try {
      await stopCamera();

      const constraints: MediaStreamConstraints = {
        video: selectedCameraId
          ? { deviceId: { exact: selectedCameraId } }
          : { facingMode: { ideal: 'environment' } },
        audio: false
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoElement) {
        videoElement.srcObject = mediaStream;
        await videoElement.play();
      }

      cameraOpen = true;
      await refreshCameraDevices();
      setNotice('info', 'Camera preview is live. Capture a frame to analyze.');
    } catch (err) {
      const message = errorMessage(err);
      if (message.toLowerCase().includes('notallowed')) {
        cameraError = 'Camera permission denied. Allow camera access and try again.';
      } else {
        cameraError = message;
      }
      cameraOpen = false;
    }
  }

  async function stopCamera(): Promise<void> {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }

    if (videoElement) {
      videoElement.srcObject = null;
    }

    cameraOpen = false;
  }

  async function captureFrame(): Promise<void> {
    clearNotice();
    cameraError = '';

    if (!videoElement || !canvasElement) {
      cameraError = 'Camera preview is not ready yet.';
      return;
    }

    const width = videoElement.videoWidth;
    const height = videoElement.videoHeight;

    if (!width || !height) {
      cameraError = 'No camera frame available to capture.';
      return;
    }

    canvasElement.width = width;
    canvasElement.height = height;

    const context = canvasElement.getContext('2d');
    if (!context) {
      cameraError = 'Could not initialize canvas context.';
      return;
    }

    context.drawImage(videoElement, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvasElement?.toBlob((result) => resolve(result), 'image/jpeg', 0.95);
    });

    if (!blob) {
      cameraError = 'Could not capture a frame from camera preview.';
      return;
    }

    capturedFile = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
    revokeCapturedPreviewUrl();
    capturedPreviewUrl = URL.createObjectURL(blob);
    setNotice('success', 'Frame captured. You can now run one-shot analysis.');
  }

  function clearCapturedFrame(): void {
    capturedFile = null;
    revokeCapturedPreviewUrl();
  }

  async function submitAnalyze(): Promise<void> {
    clearNotice();
    analyzeError = '';

    const selectedFile = selectAnalyzeFile();
    if (!selectedFile) {
      const sourceLabel = analyzeMode === 'upload' ? 'upload' : 'camera capture';
      analyzeState = 'error';
      analyzeError = `Please select an image via ${sourceLabel} first.`;
      return;
    }

    analyzeState = 'submitting';

    try {
      const formData = new FormData();
      formData.set('image', selectedFile);

      if (metadata.trim()) {
        JSON.parse(metadata);
        formData.set('metadata', metadata);
      }

      analyzeResult = await handleResponse(
        await fetch('/api/scans/oneshot', {
          method: 'POST',
          body: formData
        })
      );

      analyzeState = 'success';
      lastAnalyzeAt = formatNow();

      const createdScanId = extractScanId(analyzeResult);
      if (createdScanId) {
        scanIdInput = createdScanId;
      }

      setNotice('success', 'One-shot analysis completed. Check the result panel below.');
    } catch (err) {
      analyzeState = 'error';
      analyzeError = errorMessage(err);
      setNotice('error', analyzeError);
    }
  }

  async function checkHealth(): Promise<void> {
    clearNotice();

    try {
      health = await handleResponse(await fetch('/api/health'));
      lastHealthCheckAt = formatNow();
      setNotice('success', 'Backend health check succeeded.');
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  async function loadMetrics(): Promise<void> {
    clearNotice();

    try {
      metrics = await handleResponse(await fetch('/api/admin/metrics'));
      setNotice('success', 'Metrics loaded.');
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  async function getScan(): Promise<void> {
    clearNotice();

    if (!scanIdInput.trim()) {
      setNotice('error', 'Scan ID is required.');
      return;
    }

    try {
      scanResult = await handleResponse(await fetch(`/api/scans/${scanIdInput.trim()}`));
      setNotice('success', `Scan ${scanIdInput.trim()} loaded.`);
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  async function patchScan(): Promise<void> {
    clearNotice();

    if (!scanIdInput.trim()) {
      setNotice('error', 'Scan ID is required for manual correction.');
      return;
    }

    if (!correctionReason.trim()) {
      setNotice('error', 'Correction reason is required.');
      return;
    }

    try {
      const payload = {
        parsed_date: correctionDate || null,
        reason: correctionReason
      };

      const corrected = await handleResponse(
        await fetch(`/api/scans/${scanIdInput.trim()}`, {
          method: 'PATCH',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload)
        })
      );

      scanResult = {
        ...(scanResult ?? {}),
        correction: corrected
      };

      setNotice('success', 'Manual correction submitted.');
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  async function listExpiry(): Promise<void> {
    clearNotice();

    try {
      const params = new URLSearchParams();
      if (expiryFilterStatus) params.set('status', expiryFilterStatus);
      if (expiryFilterUser) params.set('user_id', expiryFilterUser);

      const query = params.toString();
      const url = query ? `/api/expiry?${query}` : '/api/expiry';

      expiryList = await handleResponse(await fetch(url));
      setNotice('success', 'Expiry list loaded.');
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  async function processAlerts(): Promise<void> {
    clearNotice();

    try {
      alertsResult = await handleResponse(
        await fetch('/api/alerts/process', {
          method: 'POST'
        })
      );
      setNotice('success', 'Alert processing job completed.');
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  $: analyzeScanId = extractScanId(analyzeResult);
  $: parsedExpiryValue = extractParsedExpiry(analyzeResult);

  onMount(async () => {
    if (cameraSupported()) {
      await refreshCameraDevices();
    }

    await checkHealth();
  });

  onDestroy(() => {
    stopCamera();
    revokeCapturedPreviewUrl();
  });
</script>

<main class="console">
  <header class="console-header panel">
    <div>
      <p class="eyebrow">SmartBite</p>
      <h1>Minimal Testing Console</h1>
      <p class="subtitle">
        Internal workflow for backend validation and manual scan inspection.
      </p>
    </div>

    <div class="status-strip">
      <div class="status-item">
        <span>Backend</span>
        <strong class:up={healthStatus() === 'up'} class:unknown={healthStatus() !== 'up'}>
          {healthStatus() === 'up' ? 'Online' : 'Unknown'}
        </strong>
      </div>
      <div class="status-item">
        <span>Route</span>
        <strong>{backendRouteLabel}</strong>
      </div>
      <div class="status-item">
        <span>Last Health Check</span>
        <strong>{lastHealthCheckAt || '-'}</strong>
      </div>
    </div>
  </header>

  {#if notice}
    <section class={`notice ${notice.kind}`}>
      <p>{notice.text}</p>
    </section>
  {/if}

  <section class="panel analyze-panel">
    <div class="panel-heading">
      <h2>Analyze</h2>
      <p>Run immediate one-shot OCR analysis from image upload or camera capture.</p>
    </div>

    <div class="mode-switch" role="tablist" aria-label="Analyze mode selector">
      <button
        type="button"
        class:active={analyzeMode === 'upload'}
        on:click={() => {
          analyzeMode = 'upload';
          cameraError = '';
        }}>Upload Image</button
      >
      <button
        type="button"
        class:active={analyzeMode === 'camera'}
        on:click={async () => {
          analyzeMode = 'camera';
          await refreshCameraDevices();
        }}>Use Camera</button
      >
    </div>

    {#if analyzeMode === 'upload'}
      <div class="input-group">
        <label>
          Image file
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            on:change={handleUploadFileChange}
          />
        </label>
      </div>
    {:else}
      <div class="camera-area">
        {#if !cameraSupported()}
          <p class="muted">This browser does not support camera capture for this console.</p>
        {:else}
          <div class="camera-controls-row">
            <label>
              Camera
              <select bind:value={selectedCameraId} disabled={!availableCameras.length}>
                {#if !availableCameras.length}
                  <option value="">No camera detected</option>
                {/if}
                {#each availableCameras as camera}
                  <option value={camera.deviceId}>{camera.label || `Camera ${camera.deviceId.slice(0, 8)}`}</option>
                {/each}
              </select>
            </label>

            <div class="camera-buttons">
              <button type="button" class="secondary" on:click={startCamera}>Open Camera</button>
              <button type="button" class="ghost" on:click={stopCamera} disabled={!cameraOpen}>Stop</button>
              <button type="button" on:click={captureFrame} disabled={!cameraOpen}>Capture</button>
            </div>
          </div>

          {#if cameraError}
            <p class="error-inline">{cameraError}</p>
          {/if}

          <div class="camera-preview-grid">
            <div>
              <p class="small-title">Live Preview</p>
              <video bind:this={videoElement} autoplay playsinline muted></video>
            </div>
            <div>
              <p class="small-title">Captured Frame</p>
              {#if capturedPreviewUrl}
                <img src={capturedPreviewUrl} alt="Captured frame" class="captured-preview" />
                <button type="button" class="ghost" on:click={clearCapturedFrame}>Clear Capture</button>
              {:else}
                <div class="preview-empty">No captured frame yet.</div>
              {/if}
            </div>
          </div>

          <canvas bind:this={canvasElement} class="hidden-canvas"></canvas>
        {/if}
      </div>
    {/if}

    <label>
      Metadata JSON (optional)
      <textarea bind:value={metadata} placeholder={metadataPlaceholder}></textarea>
    </label>

    <div class="actions-row">
      <button type="button" on:click={submitAnalyze} disabled={analyzeState === 'submitting'}>
        {analyzeState === 'submitting' ? 'Submitting...' : 'Analyze Now'}
      </button>
      <button
        type="button"
        class="ghost"
        on:click={() => {
          clearNotice();
          analyzeError = '';
          analyzeResult = null;
          analyzeState = 'idle';
        }}>Clear Result</button
      >
    </div>
  </section>

  <section class="panel result-panel">
    <div class="panel-heading">
      <h2>Result</h2>
      <p>Inspect immediate API response after one-shot analysis.</p>
    </div>

    <div class="result-grid">
      <div class="result-item">
        <span>Status</span>
        <strong class={`pill ${analyzeState}`}>{analyzeState}</strong>
      </div>
      <div class="result-item">
        <span>Scan ID</span>
        <strong>{analyzeScanId || 'n/a (one-shot)'}</strong>
      </div>
      <div class="result-item">
        <span>Parsed Expiry</span>
        <strong>{parsedExpiryValue}</strong>
      </div>
      <div class="result-item">
        <span>Last Analyze At</span>
        <strong>{lastAnalyzeAt || '-'}</strong>
      </div>
    </div>

    {#if analyzeError}
      <p class="error-inline">{analyzeError}</p>
    {/if}

    {#if analyzeResult}
      <details>
        <summary>Raw JSON Response</summary>
        <pre>{JSON.stringify(analyzeResult, null, 2)}</pre>
      </details>
    {:else}
      <p class="muted">No analysis response yet. Submit an image from the Analyze section.</p>
    {/if}
  </section>

  <section class="panel secondary-tools">
    <div class="panel-heading">
      <h2>Secondary Tools</h2>
      <p>System checks, lookup, corrections, and expiry/alert operations.</p>
    </div>

    <div class="tools-grid">
      <details open>
        <summary>System</summary>
        <div class="tool-content">
          <div class="actions-row compact">
            <button type="button" class="secondary" on:click={checkHealth}>Check Health</button>
            <button type="button" class="ghost" on:click={loadMetrics}>Load Metrics</button>
          </div>

          {#if health}
            <details>
              <summary>Health Payload</summary>
              <pre>{JSON.stringify(health, null, 2)}</pre>
            </details>
          {/if}

          {#if metrics}
            <details>
              <summary>Metrics Payload</summary>
              <pre>{JSON.stringify(metrics, null, 2)}</pre>
            </details>
          {/if}
        </div>
      </details>

      <details>
        <summary>Scan Lookup</summary>
        <div class="tool-content">
          <label>
            Scan ID
            <input bind:value={scanIdInput} placeholder="uuid" />
          </label>
          <div class="actions-row compact">
            <button type="button" on:click={getScan}>Get Scan</button>
            <button type="button" class="ghost" on:click={listExpiry}>Refresh Expiry List</button>
          </div>

          {#if scanResult}
            <details>
              <summary>Scan Payload</summary>
              <pre>{JSON.stringify(scanResult, null, 2)}</pre>
            </details>
          {/if}
        </div>
      </details>

      <details>
        <summary>Manual Correction</summary>
        <div class="tool-content">
          <label>
            Corrected Parsed Date
            <input type="date" bind:value={correctionDate} />
          </label>
          <label>
            Correction Reason
            <input bind:value={correctionReason} />
          </label>

          <button type="button" class="danger" on:click={patchScan}>Apply Correction</button>
        </div>
      </details>

      <details>
        <summary>Expiry & Alerts</summary>
        <div class="tool-content">
          <div class="meta-grid">
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

          <div class="actions-row compact">
            <button type="button" class="secondary" on:click={listExpiry}>Load Expiry List</button>
            <button type="button" class="danger" on:click={processAlerts}>Process Alerts</button>
          </div>

          {#if expiryList}
            <details>
              <summary>Expiry Payload</summary>
              <pre>{JSON.stringify(expiryList, null, 2)}</pre>
            </details>
          {/if}

          {#if alertsResult}
            <details>
              <summary>Alerts Payload</summary>
              <pre>{JSON.stringify(alertsResult, null, 2)}</pre>
            </details>
          {/if}
        </div>
      </details>
    </div>
  </section>
</main>
