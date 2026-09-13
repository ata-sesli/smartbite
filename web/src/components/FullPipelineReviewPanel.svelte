<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type NoticeKind = 'success' | 'error' | 'info';
  type ResultFilter = 'all' | 'remaining' | 'wrong' | 'manual' | 'correct';
  type Bbox = [number, number, number, number];
  type Polygon = Array<[number, number]>;

  type PipelineItem = {
    filename: string;
    imageUrl: string;
    verdict: string;
    expectedDate: string;
    expectedPrecision: string;
    predictedDate: string;
    exactMatch: boolean | null;
    httpStatus: number | null;
    finalStatus: string;
    rawText: string;
    normalizedText: string;
    recognitionConfidence: number | null;
    detectorConfidence: number | null;
    finalRecognitionBbox: Bbox | null;
    finalRecognitionPolygon: Polygon | null;
    finalCropPolicy: string;
    finalCropPaddingPx: number | null;
    reason: string;
    latencyMs: number | null;
    raw: JsonRecord;
  };

  type Notice = {
    kind: NoticeKind;
    text: string;
  } | null;

  export let title = 'Full Pipeline Results';
  export let description = 'Latest test64 direct benchmark run: detector, recognizer, parser, status, and latency per image.';
  export let initialFilter: ResultFilter = 'all';
  export let runIdOverride = '';

  let loading = false;
  let notice: Notice = null;
  let runId = '';
  let source = '';
  let endpoint = '';
  let createdAt = '';
  let summary: JsonRecord = {};
  let latency: JsonRecord = {};
  let items: PipelineItem[] = [];
  let selectedFilename = '';
  let filter: ResultFilter = initialFilter;
  let imageNaturalWidth = 0;
  let imageNaturalHeight = 0;
  let imageSizeFilename = '';

  $: allCount = items.length;
  $: wrongCount = items.filter((item) => item.verdict === 'wrong_date').length;
  $: manualCount = items.filter((item) => item.verdict === 'manual_review').length;
  $: remainingCount = items.filter((item) => item.exactMatch !== true).length;
  $: correctCount = items.filter((item) => item.verdict === 'correct_match').length;
  $: filteredItems = items.filter((item) => {
    if (filter === 'remaining') return item.exactMatch !== true;
    if (filter === 'wrong') return item.verdict === 'wrong_date';
    if (filter === 'manual') return item.verdict === 'manual_review';
    if (filter === 'correct') return item.verdict === 'correct_match';
    return true;
  });
  $: if (filteredItems.length > 0 && !filteredItems.some((item) => item.filename === selectedFilename)) {
    selectedFilename = filteredItems[0].filename;
  }
  $: active = filteredItems.find((item) => item.filename === selectedFilename) ?? filteredItems[0] ?? null;
  $: if (active?.filename !== imageSizeFilename) {
    imageSizeFilename = active?.filename ?? '';
    imageNaturalWidth = 0;
    imageNaturalHeight = 0;
  }

  function isRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function numberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function booleanOrNull(value: unknown): boolean | null {
    return typeof value === 'boolean' ? value : null;
  }

  function text(value: unknown): string {
    return typeof value === 'string' ? value : '';
  }

  function apiImageUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value.replace('/test64/images/', '/api/test64/images/');
  }

  function bboxOrNull(value: unknown): Bbox | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    const bbox = value.map((item) => (typeof item === 'number' && Number.isFinite(item) ? item : Number.NaN));
    return bbox.every(Number.isFinite) ? (bbox as Bbox) : null;
  }

  function polygonOrNull(value: unknown): Polygon | null {
    if (!Array.isArray(value) || value.length < 3) return null;
    const polygon: Polygon = [];
    for (const point of value) {
      if (!Array.isArray(point) || point.length < 2) return null;
      const x = point[0];
      const y = point[1];
      if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) return null;
      polygon.push([x, y]);
    }
    return polygon;
  }

  function mapItem(raw: unknown): PipelineItem | null {
    if (!isRecord(raw)) return null;
    return {
      filename: text(raw.filename),
      imageUrl: apiImageUrl(raw.image_url),
      verdict: text(raw.verdict),
      expectedDate: text(raw.expected_date),
      expectedPrecision: text(raw.expected_precision),
      predictedDate: text(raw.predicted_date ?? raw.detected_expiry_date),
      exactMatch: booleanOrNull(raw.exact_match),
      httpStatus: numberOrNull(raw.http_status),
      finalStatus: text(raw.final_status ?? raw.response_status),
      rawText: text(raw.raw_text),
      normalizedText: text(raw.normalized_text),
      recognitionConfidence: numberOrNull(raw.recognition_confidence),
      detectorConfidence: numberOrNull(raw.detector_confidence),
      finalRecognitionBbox: bboxOrNull(raw.final_recognition_bbox_xyxy),
      finalRecognitionPolygon: polygonOrNull(raw.final_recognition_polygon_json),
      finalCropPolicy: text(raw.final_crop_policy),
      finalCropPaddingPx: numberOrNull(raw.final_crop_padding_px),
      reason: text(raw.reason),
      latencyMs: numberOrNull(raw.latency_ms ?? raw.runtime_ms),
      raw: isRecord(raw.raw) ? raw.raw : raw
    };
  }

  async function handleResponse(res: Response): Promise<JsonRecord> {
    const data = (await res.json().catch(() => ({}))) as JsonRecord;
    if (!res.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`;
      throw new Error(detail);
    }
    return data;
  }

  async function loadResults(): Promise<void> {
    loading = true;
    notice = null;
    try {
      const search = new URLSearchParams();
      if (runIdOverride) search.set('run_id', runIdOverride);
      const payload = await handleResponse(
        await fetch(`/api/test64/full-pipeline-results${search.toString() ? `?${search}` : ''}`)
      );
      runId = text(payload.run_id);
      source = text(payload.source);
      endpoint = text(payload.endpoint);
      createdAt = text(payload.created_at);
      summary = isRecord(payload.summary) ? payload.summary : {};
      latency = isRecord(payload.latency) ? payload.latency : {};
      items = Array.isArray(payload.items)
        ? payload.items.map(mapItem).filter((item): item is PipelineItem => item !== null)
        : [];
      selectedFilename = items[0]?.filename ?? '';
      const loadedRemainingCount = items.filter((item) => item.exactMatch !== true).length;
      notice = {
        kind: 'success',
        text:
          filter === 'remaining'
            ? `Loaded ${loadedRemainingCount} remaining non-exact cases from ${items.length} full-pipeline results.`
            : `Loaded ${items.length} full-pipeline results.`
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      notice = { kind: 'error', text: message };
    } finally {
      loading = false;
    }
  }

  function selectItem(item: PipelineItem): void {
    selectedFilename = item.filename;
  }

  function setFilter(nextFilter: ResultFilter): void {
    filter = nextFilter;
  }

  function formatCount(value: unknown): string {
    return typeof value === 'number' ? String(value) : '-';
  }

  function formatRuntime(value: number | null | unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)} ms` : '-';
  }

  function formatPercent(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '-';
  }

  function formatConfidence(value: number | null): string {
    return value === null ? '-' : value.toFixed(3);
  }

  function formatBbox(value: Bbox | null): string {
    return value ? value.map((item) => Math.round(item)).join(', ') : '-';
  }

  function rectAttrs(bbox: Bbox): { x: number; y: number; width: number; height: number } {
    return {
      x: bbox[0],
      y: bbox[1],
      width: Math.max(0, bbox[2] - bbox[0]),
      height: Math.max(0, bbox[3] - bbox[1])
    };
  }

  function polygonPoints(polygon: Polygon): string {
    return polygon.map(([x, y]) => `${x},${y}`).join(' ');
  }

  function handleImageLoad(event: Event): void {
    const image = event.currentTarget as HTMLImageElement;
    imageNaturalWidth = image.naturalWidth;
    imageNaturalHeight = image.naturalHeight;
  }

  function verdictLabel(value: string): string {
    return value ? value.replaceAll('_', ' ') : 'unknown';
  }

  function resultLabel(item: PipelineItem): string {
    if (item.verdict === 'correct_match') return 'correct';
    if (item.verdict === 'wrong_date') return 'wrong';
    if (item.verdict === 'manual_review') return 'manual';
    return verdictLabel(item.verdict);
  }

  onMount(() => {
    void loadResults();
  });
</script>

<section class="panel recognition-review-panel full-pipeline-review-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
    <div class="recognition-review-actions">
      <button type="button" on:click={() => loadResults()} disabled={loading}>
        {loading ? 'Loading...' : 'Refresh'}
      </button>
    </div>
  </div>

  {#if notice}
    <div class={`notice ${notice.kind}`}>
      <p>{notice.text}</p>
    </div>
  {/if}

  <div class="recognition-review-stats">
    <div class="status-item">
      <span>Run</span>
      <strong>{runId || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Exact</span>
      <strong>{formatCount(summary.exact_matches)}/{formatCount(summary.total)}</strong>
    </div>
    <div class="status-item">
      <span>Accuracy</span>
      <strong>{formatPercent(summary.accuracy)}</strong>
    </div>
    <div class="status-item">
      <span>Parsed</span>
      <strong>{formatCount(summary.parsed_success)}</strong>
    </div>
    <div class="status-item">
      <span>Manual Review</span>
      <strong>{formatCount(summary.manual_review_required)}</strong>
    </div>
    <div class="status-item">
      <span>Wrong Dates</span>
      <strong>{formatCount(summary.wrong_parsed_dates)}</strong>
    </div>
    <div class="status-item">
      <span>Avg Latency</span>
      <strong>{formatRuntime(latency.avg_ms)}</strong>
    </div>
    <div class="status-item">
      <span>P90 / Max</span>
      <strong>{formatRuntime(latency.p90_ms)} / {formatRuntime(latency.max_ms)}</strong>
    </div>
  </div>

  <div class="full-pipeline-meta">
    <p class="muted">Source: {source || '-'} · Created: {createdAt || '-'} · Report: {runId || '-'}</p>
  </div>

  <div class="full-pipeline-filters" aria-label="Full pipeline result filters">
    <button type="button" class:active={filter === 'all'} on:click={() => setFilter('all')}>All {allCount}</button>
    <button type="button" class:active={filter === 'remaining'} on:click={() => setFilter('remaining')}>Remaining {remainingCount}</button>
    <button type="button" class:active={filter === 'wrong'} on:click={() => setFilter('wrong')}>Wrong {wrongCount}</button>
    <button type="button" class:active={filter === 'manual'} on:click={() => setFilter('manual')}>Manual {manualCount}</button>
    <button type="button" class:active={filter === 'correct'} on:click={() => setFilter('correct')}>Correct {correctCount}</button>
  </div>

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Full pipeline result images">
      {#if filteredItems.length === 0}
        <p class="muted">No full pipeline results match this filter.</p>
      {:else}
        {#each filteredItems as item, index}
          <button type="button" class:active={item.filename === active?.filename} on:click={() => selectItem(item)}>
            <strong>{index + 1}. {item.filename}</strong>
            <span>{resultLabel(item)} · {item.finalStatus || '-'}</span>
            <small>expected {item.expectedDate || '-'} · detected {item.predictedDate || '-'} · {formatRuntime(item.latencyMs)}</small>
          </button>
        {/each}
      {/if}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{active.filename}</h3>
            <p class="muted">Expected {active.expectedDate || '-'} ({active.expectedPrecision || 'unknown precision'}) · detected {active.predictedDate || '-'}</p>
          </div>
          <span class={`outcome-pill pipeline-outcome-${active.verdict}`}>{resultLabel(active)}</span>
        </div>

        <div class="detection-review-grid">
          <figure class="detection-review-image">
            {#if active.imageUrl}
              <div class="detection-review-image-frame">
                <img src={active.imageUrl} alt={`Full pipeline source ${active.filename}`} on:load={handleImageLoad} />
                {#if imageNaturalWidth > 0 && imageNaturalHeight > 0 && (active.finalRecognitionBbox || active.finalRecognitionPolygon)}
                  <svg
                    class="detection-review-overlay"
                    viewBox={`0 0 ${imageNaturalWidth} ${imageNaturalHeight}`}
                    preserveAspectRatio="xMidYMid meet"
                    aria-hidden="true"
                  >
                    {#if active.finalRecognitionPolygon}
                      <polygon class="final-recognition-box" points={polygonPoints(active.finalRecognitionPolygon)} />
                    {:else if active.finalRecognitionBbox}
                      {@const finalRect = rectAttrs(active.finalRecognitionBbox)}
                      <rect class="final-recognition-box" x={finalRect.x} y={finalRect.y} width={finalRect.width} height={finalRect.height} />
                    {/if}
                  </svg>
                {/if}
              </div>
            {:else}
              <div class="recognition-empty-crop">No image</div>
            {/if}
            <figcaption>
              Final recognizer bbox:
              {#if active.finalRecognitionBbox}
                [{formatBbox(active.finalRecognitionBbox)}]
              {:else}
                not available in this report
              {/if}
            </figcaption>
          </figure>

          <div class="recognition-winner-card">
            <h3>Endpoint Result</h3>
            <dl class="compact-dl">
              <dt>HTTP</dt>
              <dd>{formatCount(active.httpStatus)}</dd>
              <dt>Status</dt>
              <dd>{active.finalStatus || '-'}</dd>
              <dt>Exact Match</dt>
              <dd>{active.exactMatch === true ? 'yes' : active.exactMatch === false ? 'no' : '-'}</dd>
              <dt>Expected</dt>
              <dd>{active.expectedDate || '-'}</dd>
              <dt>Precision</dt>
              <dd>{active.expectedPrecision || '-'}</dd>
              <dt>Detected</dt>
              <dd>{active.predictedDate || '-'}</dd>
              <dt>Latency</dt>
              <dd>{formatRuntime(active.latencyMs)}</dd>
              <dt>Recognizer</dt>
              <dd>{formatConfidence(active.recognitionConfidence)}</dd>
              <dt>Detector</dt>
              <dd>{formatConfidence(active.detectorConfidence)}</dd>
              <dt>Final BBox</dt>
              <dd>{formatBbox(active.finalRecognitionBbox)}</dd>
              <dt>Crop Policy</dt>
              <dd>{active.finalCropPolicy || '-'}</dd>
              <dt>Padding</dt>
              <dd>{formatCount(active.finalCropPaddingPx)}</dd>
            </dl>
          </div>
        </div>

        <div class="detection-performance-grid">
          <article>
            <h3>OCR Evidence</h3>
            <dl class="compact-dl">
              <dt>Raw</dt>
              <dd>{active.rawText || '-'}</dd>
              <dt>Normalized</dt>
              <dd>{active.normalizedText || '-'}</dd>
            </dl>
          </article>
          <article>
            <h3>Reason</h3>
            <p>{active.reason || '-'}</p>
          </article>
        </div>

        <details class="full-pipeline-raw-row">
          <summary>Raw Result Row</summary>
          <pre>{JSON.stringify(active.raw, null, 2)}</pre>
        </details>
      {:else}
        <p class="muted">Select a full-pipeline result to inspect.</p>
      {/if}
    </section>
  </div>
</section>
