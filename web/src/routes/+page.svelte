<script lang="ts">
  import { browser } from '$app/environment';
  import { onDestroy, onMount } from 'svelte';
  import CropTruthPanel from '../components/CropTruthPanel.svelte';
  import DetectionReviewPanel from '../components/DetectionReviewPanel.svelte';
  import ExportPanel from '../components/ExportPanel.svelte';
  import FullPipelineReviewPanel from '../components/FullPipelineReviewPanel.svelte';
  import QueueList from '../components/QueueList.svelte';
  import RecognitionFailureReviewPanel from '../components/RecognitionFailureReviewPanel.svelte';
  import ReviewWorkspace from '../components/ReviewWorkspace.svelte';
  import Test64LabelPanel from '../components/Test64LabelPanel.svelte';

  type JsonRecord = Record<string, unknown>;
  type AnalyzeMode = 'upload' | 'camera';
  type ConsoleTab =
    | 'analyze'
    | 'queue'
    | 'review'
    | 'labeling'
    | 'cropTruth'
    | 'recognitionReview'
    | 'detectionReview'
    | 'productCropperReview'
    | 'fullPipelineReview'
    | 'dataset'
    | 'tools';
  type NoticeKind = 'error' | 'success' | 'info';
  type RequestState = 'idle' | 'submitting' | 'success' | 'error';
  type BatchState = 'pending' | 'queued' | 'processing' | 'done' | 'error' | 'timeout';
  type ReviewFilter = 'unreviewed' | 'reviewed' | 'all';
  type HistoryOrder = 'newest' | 'oldest';

  type Notice = {
    kind: NoticeKind;
    text: string;
  };

  type BatchItem = {
    id: string;
    file: File;
    name: string;
    status: BatchState;
    scanId: string;
    result: JsonRecord | null;
    error: string;
  };

  let activeTab: ConsoleTab = 'analyze';
  let analyzeMode: AnalyzeMode = 'upload';

  let uploadFile: File | null = null;
  let capturedFile: File | null = null;
  let capturedPreviewUrl = '';

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
  let batchItems: BatchItem[] = [];
  let batchRunning = false;
  let stopBatchRequested = false;
  let scanHistory: JsonRecord[] = [];
  let historyLoading = false;
  let reviewIndex = -1;
  let reviewItem: JsonRecord | null = null;
  let reviewFilter: ReviewFilter = 'unreviewed';
  let reviewCandidates: JsonRecord[] = [];
  let reviewUnreviewedCount = 0;
  let reviewReviewedCount = 0;
  let reviewDoneCount = 0;
  let reviewScanId = '';
  let reviewVerdict = '';
  let reviewAccepted = false;
  let reviewFinalText = '';
  let reviewFinalParsedDate = '';
  let reviewNotes = '';
  let reviewBbox: number[] | null = null;
  let exportOutputDir = '';
  let exportIncludeDetector = true;
  let exportIncludeRecognition = true;
  let exportResult: JsonRecord | null = null;
  let exportRunning = false;
  let reviewSaving = false;
  let queueingTestImages = false;
  let testImagesProgressActive = false;
  let testImagesCompleted = 0;
  let testImagesTotal = 0;
  let testImagesFailed = 0;
  let historyLimit = 50;
  let historyOrder: HistoryOrder = 'newest';

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
  const batchPollIntervalMs = 1500;
  const batchPollTimeoutMs = 180000;
  const testImageEstimatedMsPerScan = 15000;

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

  async function switchTab(tab: ConsoleTab): Promise<void> {
    activeTab = tab;
    if ((tab === 'queue' || tab === 'review') && !scanHistory.length) {
      await loadScanHistory(false);
    }
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
    const files = Array.from(target?.files ?? []);
    uploadFile = files[0] ?? null;
    batchItems = files.map((file) => ({
      id: randomId(),
      file,
      name: file.name,
      status: 'pending',
      scanId: '',
      result: null,
      error: ''
    }));
  }

  function randomId(): string {
    return browser && 'crypto' in window && 'randomUUID' in window.crypto
      ? window.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function normalizedScanStatus(value: unknown): string {
    return typeof value === 'string' ? value.trim().toLowerCase() : '';
  }

  function reviewState(item: JsonRecord): 'reviewed' | 'unreviewed' {
    const raw = statusText(item.review_verdict);
    return raw === '-' ? 'unreviewed' : 'reviewed';
  }

  function isScanDone(item: JsonRecord): boolean {
    return normalizedScanStatus(item.status) === 'done';
  }

  $: reviewCandidates = scanHistory.filter((item) => {
    if (!isScanDone(item)) return false;
    const state = reviewState(item);
    if (reviewFilter === 'all') return true;
    return reviewFilter === state;
  });

  $: {
    const done = scanHistory.filter((item) => isScanDone(item));
    reviewDoneCount = done.length;
    reviewReviewedCount = done.filter((item) => reviewState(item) === 'reviewed').length;
    reviewUnreviewedCount = done.filter((item) => reviewState(item) === 'unreviewed').length;
  }

  $: if (activeTab === 'review' && reviewCandidates.length > 0) {
    const selectedId = reviewItem ? statusText(reviewItem.scan_id) : '';
    const selectedStillVisible = reviewCandidates.some((item) => statusText(item.scan_id) === selectedId);
    if (!selectedId || !selectedStillVisible) {
      void openReview(reviewCandidates[0]);
    }
  }

  $: if (activeTab === 'review' && reviewCandidates.length === 0 && reviewItem) {
    clearReview();
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

  function updateBatchItem(id: string, patch: Partial<BatchItem>): void {
    batchItems = batchItems.map((item) => (item.id === id ? { ...item, ...patch } : item));
  }

  async function createQueuedScan(file: File): Promise<string> {
    const formData = new FormData();
    formData.set('image', file);
    formData.set('user_id', 'console_batch');
    formData.set('qr_code', `console-${randomId()}`);

    const created = await handleResponse(
      await fetch('/api/scans', {
        method: 'POST',
        body: formData
      })
    );
    const scanId = extractScanId(created);
    if (!scanId) {
      throw new Error('Backend did not return a scan ID.');
    }
    return scanId;
  }

  async function waitForScanResult(scanId: string, itemId: string): Promise<JsonRecord> {
    const deadline = Date.now() + batchPollTimeoutMs;

    while (Date.now() < deadline) {
      const payload = await handleResponse(await fetch(`/api/scans/${scanId}`));
      const status = typeof payload.status === 'string' ? payload.status : '';
      if (status === 'done') {
        return payload;
      }
      if (status === 'failed') {
        throw new Error('Scan failed during worker processing.');
      }
      if (status === 'queued' || status === 'processing') {
        updateBatchItem(itemId, { status: status as BatchState, result: payload });
      }
      await sleep(batchPollIntervalMs);
    }

    throw new Error('Batch item timed out while waiting for worker result.');
  }

  async function startBatch(): Promise<void> {
    clearNotice();
    if (!batchItems.length) {
      setNotice('error', 'Choose one or more image files before starting batch processing.');
      return;
    }
    if (batchRunning) return;

    batchRunning = true;
    stopBatchRequested = false;

    try {
      for (const item of batchItems) {
        if (stopBatchRequested) break;
        if (!['pending', 'error', 'timeout'].includes(item.status)) continue;

        try {
          updateBatchItem(item.id, { status: 'queued', error: '', result: null });
          const scanId = await createQueuedScan(item.file);
          updateBatchItem(item.id, { scanId, status: 'queued' });

          const result = await waitForScanResult(scanId, item.id);
          updateBatchItem(item.id, { status: 'done', result, error: '' });
          analyzeResult = result;
          analyzeState = 'success';
          scanIdInput = scanId;
          lastAnalyzeAt = formatNow();
        } catch (err) {
          const message = errorMessage(err);
          updateBatchItem(item.id, {
            status: message.toLowerCase().includes('timed out') ? 'timeout' : 'error',
            error: message
          });
        }
      }

      await loadScanHistory(false);
      setNotice('success', stopBatchRequested ? 'Batch stopped after current item.' : 'Batch processing completed.');
    } finally {
      batchRunning = false;
      stopBatchRequested = false;
    }
  }

  function clearBatch(): void {
    if (batchRunning) {
      setNotice('info', 'Batch is running. Use Stop After Current before clearing.');
      return;
    }
    batchItems = [];
  }

  function loadPayloadIntoResult(payload: JsonRecord | null): void {
    if (!payload) return;
    analyzeResult = payload;
    analyzeState = 'success';
    const scanId = extractScanId(payload);
    if (scanId) {
      scanIdInput = scanId;
    }
    lastAnalyzeAt = formatNow();
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
      loadPayloadIntoResult(scanResult);
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

  async function loadScanHistory(showNotice = true): Promise<void> {
    if (showNotice) clearNotice();
    historyLoading = true;

    try {
      const payload = await handleResponse(await fetch(`/api/scans?limit=${historyLimit}`));
      const items = payload.items;
      const filtered = Array.isArray(items) ? (items.filter(isJsonRecord) as JsonRecord[]) : [];
      scanHistory = sortScanHistory(filtered);
      if (showNotice) setNotice('success', 'Scan history loaded.');
    } catch (err) {
      if (showNotice) setNotice('error', errorMessage(err));
    } finally {
      historyLoading = false;
    }
  }

  async function queueAllTestImages(): Promise<void> {
    clearNotice();
    queueingTestImages = true;
    testImagesProgressActive = false;
    testImagesCompleted = 0;
    testImagesTotal = 0;
    testImagesFailed = 0;
    try {
      const payload = await handleResponse(
        await fetch('/api/test-images/queue', {
          method: 'POST'
        })
      );
      const queued = typeof payload.queued === 'number' ? payload.queued : 0;
      const skipped = typeof payload.skipped === 'number' ? payload.skipped : 0;
      const scanIds =
        Array.isArray(payload.scan_ids)
          ? payload.scan_ids.filter((value): value is string => typeof value === 'string' && value.length > 0)
          : [];

      testImagesTotal = queued;
      if (scanIds.length > 0) {
        await monitorTestImageQueue(scanIds);
      }
      await loadScanHistory(false);
      const unresolved = Math.max(0, testImagesTotal - testImagesCompleted);
      const failedSuffix = testImagesFailed ? `, failed ${testImagesFailed}` : '';
      const unresolvedSuffix = unresolved ? `, unresolved ${unresolved}` : '';
      setNotice(
        'success',
        `Queued ${queued} test images${skipped ? ` (${skipped} skipped)` : ''}. Completed ${testImagesCompleted}/${testImagesTotal}${failedSuffix}${unresolvedSuffix}.`
      );
    } catch (err) {
      setNotice('error', errorMessage(err));
    } finally {
      testImagesProgressActive = false;
      queueingTestImages = false;
    }
  }

  async function monitorTestImageQueue(scanIds: string[]): Promise<void> {
    testImagesProgressActive = true;
    const pending = new Set(scanIds);
    const terminal = new Set<string>();
    const estimatedDurationMs = scanIds.length * testImageEstimatedMsPerScan;
    const deadline = Date.now() + Math.max(600000, estimatedDurationMs, batchPollTimeoutMs * 2);

    while (pending.size > 0 && Date.now() < deadline) {
      const checks = Array.from(pending).map(async (scanId) => {
        try {
          const payload = await handleResponse(await fetch(`/api/scans/${scanId}`));
          const status = normalizedScanStatus(payload.status);
          if (status === 'done' || status === 'failed') {
            pending.delete(scanId);
            terminal.add(scanId);
            if (status === 'failed') testImagesFailed += 1;
          }
        } catch {
          // Keep waiting; transient network/proxy errors should not abort the whole monitor.
        }
      });

      await Promise.all(checks);
      testImagesCompleted = terminal.size;
      if (pending.size > 0) {
        await sleep(batchPollIntervalMs);
      }
    }
  }

  function statusText(value: unknown): string {
    return typeof value === 'string' && value ? value : '-';
  }

  function scanMomentMs(item: JsonRecord): number {
    const createdAt = typeof item.created_at === 'string' ? item.created_at : '';
    const parsed = Date.parse(createdAt);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function sortScanHistory(items: JsonRecord[]): JsonRecord[] {
    const direction = historyOrder === 'newest' ? -1 : 1;
    return [...items].sort((a, b) => {
      const delta = scanMomentMs(a) - scanMomentMs(b);
      if (delta !== 0) return direction * delta;
      return direction * statusText(a.scan_id).localeCompare(statusText(b.scan_id));
    });
  }

  function setHistoryLimit(value: number): void {
    const normalized = value === 100 || value === 200 ? value : 50;
    if (historyLimit === normalized) return;
    historyLimit = normalized;
    void loadScanHistory(false);
  }

  function setHistoryOrder(value: string): void {
    if (value !== 'newest' && value !== 'oldest') return;
    historyOrder = value;
    scanHistory = sortScanHistory(scanHistory);
  }

  function handleHistoryLimitChange(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    setHistoryLimit(Number(target?.value));
  }

  function handleHistoryOrderChange(event: Event): void {
    const target = event.currentTarget as HTMLSelectElement | null;
    setHistoryOrder(target?.value ?? '');
  }

  function historyPayload(item: JsonRecord): JsonRecord {
    const result = isJsonRecord(item.result) ? item.result : null;
    if (result) {
      return {
        scan_id: statusText(item.scan_id),
        status: statusText(item.status),
        result
      };
    }
    return item;
  }


  function numberValue(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  async function openReview(item: JsonRecord): Promise<void> {
    clearNotice();
    reviewItem = item;
    activeTab = 'review';
    reviewScanId = statusText(item.scan_id);
    reviewIndex = reviewCandidates.findIndex((scan) => statusText(scan.scan_id) === reviewScanId);
    const existing = isJsonRecord(item.review) ? item.review : null;
    reviewVerdict = statusText(existing?.verdict) === '-' ? '' : statusText(existing?.verdict);
    reviewAccepted = typeof existing?.accepted_for_training === 'boolean' ? existing.accepted_for_training : false;
    reviewFinalText = statusText(existing?.final_text) === '-' ? statusText(item.parsed_date) : statusText(existing?.final_text);
    reviewFinalParsedDate = statusText(existing?.final_parsed_date) === '-' ? statusText(item.parsed_date) : statusText(existing?.final_parsed_date);
    reviewFinalText = reviewFinalText === '-' ? '' : reviewFinalText;
    reviewFinalParsedDate = reviewFinalParsedDate === '-' ? '' : reviewFinalParsedDate;
    reviewNotes = statusText(existing?.notes) === '-' ? '' : statusText(existing?.notes);
    const bbox = Array.isArray(existing?.bbox_xyxy) ? existing.bbox_xyxy.map(numberValue) : [];
    reviewBbox = bbox.length === 4 && bbox.every((value) => value !== null) ? (bbox as number[]) : null;

    try {
      const payload = await handleResponse(await fetch(`/api/scans/${reviewScanId}/review`));
      const remote = isJsonRecord(payload.review) ? payload.review : null;
      if (remote) {
        reviewVerdict = statusText(remote.verdict);
        reviewAccepted = Boolean(remote.accepted_for_training);
        reviewFinalText = statusText(remote.final_text) === '-' ? reviewFinalText : statusText(remote.final_text);
        reviewFinalParsedDate = statusText(remote.final_parsed_date) === '-' ? reviewFinalParsedDate : statusText(remote.final_parsed_date);
        reviewNotes = statusText(remote.notes) === '-' ? '' : statusText(remote.notes);
        const remoteBbox = Array.isArray(remote.bbox_xyxy) ? remote.bbox_xyxy.map(numberValue) : [];
        reviewBbox = remoteBbox.length === 4 && remoteBbox.every((value) => value !== null) ? (remoteBbox as number[]) : reviewBbox;
      }
    } catch (err) {
      setNotice('error', errorMessage(err));
    }
  }

  function setReviewVerdict(value: string): void {
    reviewVerdict = value;
    if (value === 'correct') {
      if (reviewItem) {
        const parsed = statusText(reviewItem.parsed_date);
        const raw = statusText(reviewItem.raw_text);
        reviewFinalParsedDate = parsed === '-' ? '' : parsed;
        reviewFinalText = parsed !== '-' ? parsed : raw === '-' ? '' : raw;
      }
      reviewAccepted = Boolean(reviewFinalParsedDate || reviewFinalText);
      if (!reviewNotes.trim()) {
        reviewNotes = 'prediction verified';
      }
      return;
    }
    if (value === 'incorrect') {
      reviewAccepted = true;
    }
  }

  function clearReview(): void {
    reviewItem = null;
    reviewIndex = -1;
    reviewScanId = '';
    reviewVerdict = '';
    reviewAccepted = false;
    reviewFinalText = '';
    reviewFinalParsedDate = '';
    reviewNotes = '';
    reviewBbox = null;
  }

  async function saveReview(): Promise<void> {
    if (!reviewScanId) return;
    if (!reviewVerdict) {
      setNotice('error', 'Select Correct or False first.');
      return;
    }
    if (reviewVerdict === 'incorrect') {
      if (!reviewFinalParsedDate) {
        setNotice('error', 'Corrected date is required for False reviews.');
        return;
      }
      if (!reviewNotes.trim()) {
        setNotice('error', 'Feedback text is required for False reviews.');
        return;
      }
    }
    clearNotice();
    reviewSaving = true;
    try {
      const payload = {
        verdict: reviewVerdict,
        accepted_for_training: reviewAccepted,
        final_text: reviewFinalText || reviewFinalParsedDate || null,
        final_parsed_date: reviewFinalParsedDate || null,
        bbox_xyxy: reviewBbox,
        bbox_source: 'original_image',
        reviewer_id: 'console',
        notes: reviewNotes || null
      };
      await handleResponse(
        await fetch(`/api/scans/${reviewScanId}/review`, {
          method: 'PUT',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload)
        })
      );
      await loadScanHistory(false);
      const refreshedIndex = reviewCandidates.findIndex((scan) => statusText(scan.scan_id) === reviewScanId);
      if (refreshedIndex >= 0 && refreshedIndex < reviewCandidates.length) {
        reviewIndex = refreshedIndex;
        reviewItem = reviewCandidates[refreshedIndex] ?? reviewItem;
      }
      setNotice('success', 'Review saved');
    } catch (err) {
      setNotice('error', errorMessage(err));
    } finally {
      reviewSaving = false;
    }
  }

  function goToPreviousReview(): void {
    if (reviewIndex <= 0) return;
    const previous = reviewCandidates[reviewIndex - 1];
    if (previous) void openReview(previous);
  }

  function goToNextReview(): void {
    if (reviewIndex < 0 || reviewIndex >= reviewCandidates.length - 1) return;
    const next = reviewCandidates[reviewIndex + 1];
    if (next) void openReview(next);
  }

  function setReviewFilter(value: ReviewFilter): void {
    reviewFilter = value;
  }

  async function exportTrainingData(): Promise<void> {
    clearNotice();
    exportRunning = true;
    try {
      exportResult = await handleResponse(
        await fetch('/api/training-data/export', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            output_dir: exportOutputDir || null,
            include_detector: exportIncludeDetector,
            include_recognition: exportIncludeRecognition
          })
        })
      );
      setNotice('success', 'Training data export completed.');
    } catch (err) {
      setNotice('error', errorMessage(err));
    } finally {
      exportRunning = false;
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
    await loadScanHistory(false);
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

  <nav class="console-tabs" aria-label="Console sections">
    <button type="button" class:active={activeTab === 'analyze'} on:click={() => switchTab('analyze')}>Analyze</button>
    <button type="button" class:active={activeTab === 'queue'} on:click={() => switchTab('queue')}>Queue</button>
    <button type="button" class:active={activeTab === 'review'} on:click={() => switchTab('review')}>Review</button>
    <button type="button" class:active={activeTab === 'labeling'} on:click={() => switchTab('labeling')}>Labeling</button>
    <button type="button" class:active={activeTab === 'cropTruth'} on:click={() => switchTab('cropTruth')}>Crop Truth</button>
    <button type="button" class:active={activeTab === 'recognitionReview'} on:click={() => switchTab('recognitionReview')}>Recognition Review</button>
    <button type="button" class:active={activeTab === 'detectionReview'} on:click={() => switchTab('detectionReview')}>Detection Review</button>
    <button type="button" class:active={activeTab === 'productCropperReview'} on:click={() => switchTab('productCropperReview')}>Product Cropper</button>
    <button type="button" class:active={activeTab === 'fullPipelineReview'} on:click={() => switchTab('fullPipelineReview')}>Full Pipeline</button>
    <button type="button" class:active={activeTab === 'dataset'} on:click={() => switchTab('dataset')}>Dataset / Export</button>
    <button type="button" class:active={activeTab === 'tools'} on:click={() => switchTab('tools')}>Tools</button>
  </nav>

  {#if activeTab === 'analyze'}
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
            multiple
            accept="image/png,image/jpeg,image/webp"
            on:change={handleUploadFileChange}
          />
        </label>
        <p class="muted">Choose one file for one-shot analysis, or multiple files for sequential batch processing.</p>
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
      <p>Inspect the latest one-shot, batch, or history-loaded scan response.</p>
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


  <section class="panel batch-panel">
    <div class="panel-heading">
      <h2>Batch Queue</h2>
      <p>Process selected upload images sequentially through the async worker path.</p>
    </div>

    <div class="actions-row compact">
      <button type="button" on:click={startBatch} disabled={batchRunning || !batchItems.length}>
        {batchRunning ? 'Batch Running...' : 'Start Batch'}
      </button>
      <button
        type="button"
        class="secondary"
        on:click={() => {
          stopBatchRequested = true;
          setNotice('info', 'Batch will stop after the current item finishes.');
        }}
        disabled={!batchRunning}>Stop After Current</button
      >
      <button type="button" class="ghost" on:click={clearBatch} disabled={batchRunning}>Clear Batch</button>
    </div>

    {#if batchItems.length}
      <div class="batch-list">
        {#each batchItems as item}
          <article class="batch-row">
            <div>
              <strong>{item.name}</strong>
              <p class="muted">{item.scanId || 'No scan created yet.'}</p>
              {#if item.error}
                <p class="error-inline">{item.error}</p>
              {/if}
            </div>
            <span class={`pill ${item.status}`}>{item.status}</span>
            <div class="row-actions">
              <button
                type="button"
                class="ghost"
                on:click={() => loadPayloadIntoResult(item.result)}
                disabled={!item.result}>Load Result</button
              >
              {#if item.result}
                <details>
                  <summary>JSON</summary>
                  <pre>{JSON.stringify(item.result, null, 2)}</pre>
                </details>
              {/if}
            </div>
          </article>
        {/each}
      </div>
    {:else}
      <p class="muted">No batch files selected yet. Use the upload control above and select multiple images.</p>
    {/if}
  </section>


  {/if}

  {#if activeTab === 'queue'}
    <section class="panel">
      <div class="panel-heading">
        <h2>History View</h2>
        <p>Select how many items to load and order them by scan timestamp.</p>
      </div>
      <div class="meta-grid">
        <label>
          History Size
          <select value={String(historyLimit)} on:change={handleHistoryLimitChange}>
            <option value="50">50</option>
            <option value="100">100</option>
            <option value="200">200 (max)</option>
          </select>
        </label>

        <label>
          Order
          <select value={historyOrder} on:change={handleHistoryOrderChange}>
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </label>
      </div>
      <p class="muted">Showing {scanHistory.length} scans from history.</p>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <h2>Queue Helpers</h2>
        <p>Quick action for internal testing.</p>
      </div>
      <div class="actions-row compact">
        <button type="button" class="secondary" on:click={queueAllTestImages} disabled={queueingTestImages || historyLoading}>
          {queueingTestImages ? 'Queueing Test Images...' : 'Queue All test-images'}
        </button>
      </div>
      {#if testImagesTotal > 0}
        <div class="test-images-progress" aria-live="polite">
          {#if testImagesProgressActive}
            <span class="spinner" aria-hidden="true"></span>
          {/if}
          <strong>{testImagesCompleted}/{testImagesTotal}</strong>
        </div>
      {/if}
    </section>

    <QueueList
      items={scanHistory}
      loading={historyLoading}
      on:refresh={() => loadScanHistory()}
      on:review={(event) => openReview(event.detail.item)}
      on:loadResult={(event) => {
        loadPayloadIntoResult(historyPayload(event.detail.item));
        activeTab = 'analyze';
      }}
      on:useCorrection={(event) => {
        scanIdInput = statusText(event.detail.item.scan_id);
        scanResult = historyPayload(event.detail.item);
        loadPayloadIntoResult(historyPayload(event.detail.item));
        activeTab = 'tools';
      }}
    />
  {/if}

  {#if activeTab === 'labeling'}
    <Test64LabelPanel />
  {/if}

  {#if activeTab === 'cropTruth'}
    <CropTruthPanel />
  {/if}

  {#if activeTab === 'recognitionReview'}
    <RecognitionFailureReviewPanel />
  {/if}

  {#if activeTab === 'detectionReview'}
    <DetectionReviewPanel />
  {/if}

  {#if activeTab === 'productCropperReview'}
    <DetectionReviewPanel
      endpoint="/api/test64/product-cropper-review"
      title="Product Cropper Review"
      description="D2S product-cropper ROIs followed by expiry detection inside the cropped product regions."
      emptyText="No product-cropper audit items available."
      configSwitchDescription="Inspect product-first detector configs without mixing in recognition results."
    />
  {/if}

  {#if activeTab === 'fullPipelineReview'}
    <FullPipelineReviewPanel />
  {/if}

  {#if activeTab === 'review'}
    <section class="panel">
      <div class="panel-heading">
        <h2>Review Filters</h2>
        <p>Default is unreviewed scans.</p>
      </div>
      <div class="meta-grid">
        <label>
          History Size
          <select value={String(historyLimit)} on:change={handleHistoryLimitChange}>
            <option value="50">50</option>
            <option value="100">100</option>
            <option value="200">200 (max)</option>
          </select>
        </label>

        <label>
          Order
          <select value={historyOrder} on:change={handleHistoryOrderChange}>
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </label>
      </div>
      <div class="actions-row compact">
        <button type="button" class:secondary={reviewFilter !== 'unreviewed'} on:click={() => setReviewFilter('unreviewed')}>
          Unreviewed ({reviewUnreviewedCount})
        </button>
        <button type="button" class:secondary={reviewFilter !== 'reviewed'} on:click={() => setReviewFilter('reviewed')}>
          Reviewed ({reviewReviewedCount})
        </button>
        <button type="button" class:secondary={reviewFilter !== 'all'} on:click={() => setReviewFilter('all')}>
          All Done ({reviewDoneCount})
        </button>
      </div>
    </section>

    <QueueList
      items={reviewCandidates}
      loading={historyLoading}
      on:refresh={() => loadScanHistory()}
      on:review={(event) => openReview(event.detail.item)}
      on:loadResult={(event) => {
        loadPayloadIntoResult(historyPayload(event.detail.item));
        activeTab = 'analyze';
      }}
      on:useCorrection={(event) => {
        scanIdInput = statusText(event.detail.item.scan_id);
        scanResult = historyPayload(event.detail.item);
        loadPayloadIntoResult(historyPayload(event.detail.item));
        activeTab = 'tools';
      }}
    />

    <ReviewWorkspace
      item={reviewItem}
      index={reviewIndex}
      total={reviewCandidates.length}
      bind:verdict={reviewVerdict}
      bind:acceptedForTraining={reviewAccepted}
      bind:finalText={reviewFinalText}
      bind:finalParsedDate={reviewFinalParsedDate}
      bind:notes={reviewNotes}
      bind:bbox={reviewBbox}
      saving={reviewSaving}
      on:previous={goToPreviousReview}
      on:next={goToNextReview}
      on:close={clearReview}
      on:save={saveReview}
      on:verdictChange={(event) => setReviewVerdict(event.detail.verdict)}
    />
  {/if}

  {#if activeTab === 'dataset'}
    <ExportPanel
      bind:outputDir={exportOutputDir}
      bind:includeDetector={exportIncludeDetector}
      bind:includeRecognition={exportIncludeRecognition}
      result={exportResult}
      exporting={exportRunning}
      on:export={exportTrainingData}
    />
  {/if}

  {#if activeTab === 'tools'}
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
  {/if}
</main>
