<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type Bbox = [number, number, number, number];
  type Polygon = Array<[number, number]>;
  type Filter = 'all' | 'unchecked' | 'true' | 'false' | 'exact' | 'wrong' | 'parsed' | 'manual';
  type Notice = { kind: 'success' | 'error' | 'info'; text: string } | null;

  type Review = {
    is_correct: boolean | null;
    corrected_date: string;
    updated_at: string;
  } | null;

  type Item = {
    filename: string;
    imageUrl: string;
    verdict: string;
    predictedDate: string;
    expectedDate: string;
    exactMatch: boolean | null;
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
    debugProfile: JsonRecord;
    review: Review;
    raw: JsonRecord;
  };

  let loading = false;
  let saving = false;
  let notice: Notice = null;
  let runId = '';
  let source = '';
  let createdAt = '';
  let summary: JsonRecord = {};
  let latency: JsonRecord = {};
  let items: Item[] = [];
  let selectedFilename = '';
  let filter: Filter = 'unchecked';
  let draftCorrect: boolean | null = null;
  let draftDate = '';
  let imageNaturalWidth = 0;
  let imageNaturalHeight = 0;
  let imageSizeFilename = '';

  $: allCount = items.length;
  $: checkedCount = items.filter((item) => item.review?.is_correct !== null && item.review?.is_correct !== undefined).length;
  $: trueCount = items.filter((item) => item.review?.is_correct === true).length;
  $: falseCount = items.filter((item) => item.review?.is_correct === false).length;
  $: labeledCount = items.filter((item) => item.expectedDate).length;
  $: exactCount = items.filter((item) => item.exactMatch === true).length;
  $: wrongCount = items.filter((item) => item.expectedDate && item.exactMatch === false).length;
  $: uncheckedCount = items.filter((item) => item.review?.is_correct === null || item.review === null).length;
  $: parsedCount = items.filter((item) => item.finalStatus === 'parsed_success').length;
  $: manualCount = items.filter((item) => item.finalStatus === 'manual_review_required').length;
  $: filteredItems = items.filter((item) => {
    if (filter === 'unchecked') return item.review?.is_correct === null || item.review === null;
    if (filter === 'true') return item.review?.is_correct === true;
    if (filter === 'false') return item.review?.is_correct === false;
    if (filter === 'exact') return item.exactMatch === true;
    if (filter === 'wrong') return item.expectedDate && item.exactMatch === false;
    if (filter === 'parsed') return item.finalStatus === 'parsed_success';
    if (filter === 'manual') return item.finalStatus === 'manual_review_required';
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

  function text(value: unknown): string {
    return typeof value === 'string' ? value : '';
  }

  function numberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
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

  function apiImageUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value.replace('/test-images/images/', '/api/test-images/images/');
  }

  function mapReview(value: unknown): Review {
    if (!isRecord(value)) return null;
    return {
      is_correct: typeof value.is_correct === 'boolean' ? value.is_correct : null,
      corrected_date: text(value.corrected_date),
      updated_at: text(value.updated_at)
    };
  }

  function mapItem(raw: unknown): Item | null {
    if (!isRecord(raw)) return null;
    return {
      filename: text(raw.filename),
      imageUrl: apiImageUrl(raw.image_url),
      verdict: text(raw.verdict),
      predictedDate: text(raw.predicted_date ?? raw.detected_expiry_date),
      expectedDate: text(raw.expected_date),
      exactMatch: typeof raw.exact_match === 'boolean' ? raw.exact_match : null,
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
      debugProfile: isRecord(raw.debug_profile) ? raw.debug_profile : {},
      review: mapReview(raw.review),
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
      const payload = await handleResponse(await fetch('/api/test-images/post-run-check'));
      runId = text(payload.run_id);
      source = text(payload.source);
      createdAt = text(payload.created_at);
      summary = isRecord(payload.summary) ? payload.summary : {};
      latency = isRecord(payload.latency) ? payload.latency : {};
      items = Array.isArray(payload.items)
        ? payload.items.map(mapItem).filter((item): item is Item => item !== null)
        : [];
      selectedFilename = filteredItems[0]?.filename ?? items[0]?.filename ?? '';
      syncDraft();
      notice = { kind: 'success', text: `Loaded ${items.length} blind test-image results.` };
    } catch (err) {
      notice = { kind: 'error', text: err instanceof Error ? err.message : 'Unexpected error' };
    } finally {
      loading = false;
    }
  }

  function syncDraft(): void {
    const selected = items.find((item) => item.filename === selectedFilename);
    draftCorrect = selected?.review?.is_correct ?? null;
    draftDate = selected?.review?.corrected_date ?? '';
  }

  function selectItem(item: Item): void {
    selectedFilename = item.filename;
    syncDraft();
  }

  function setFilter(nextFilter: Filter): void {
    filter = nextFilter;
  }

  async function saveReview(): Promise<void> {
    if (!active) return;
    saving = true;
    notice = null;
    try {
      const payload = await handleResponse(
        await fetch('/api/test-images/post-run-check', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            run_id: runId,
            filename: active.filename,
            is_correct: draftCorrect,
            corrected_date: draftDate
          })
        })
      );
      const review = mapReview(payload.review);
      items = items.map((item) => (item.filename === active.filename ? { ...item, review } : item));
      notice = { kind: 'success', text: `Saved review for ${active.filename}.` };
    } catch (err) {
      notice = { kind: 'error', text: err instanceof Error ? err.message : 'Unexpected error' };
    } finally {
      saving = false;
    }
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

  function reviewLabel(item: Item): string {
    if (item.exactMatch === true) return 'exact';
    if (item.expectedDate && item.exactMatch === false) return 'wrong';
    if (item.review?.is_correct === true) return 'true';
    if (item.review?.is_correct === false) return `false${item.review.corrected_date ? ` -> ${item.review.corrected_date}` : ''}`;
    return 'unchecked';
  }

  function outcomeClass(item: Item): string {
    if (item.exactMatch === true) return 'correct_match';
    if (item.expectedDate && item.exactMatch === false) return 'wrong_date';
    return item.finalStatus === 'parsed_success' ? 'correct_match' : 'manual_review';
  }

  onMount(() => {
    void loadResults();
  });
</script>

<section class="panel recognition-review-panel full-pipeline-review-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>Test Images Post-Run Check</h2>
      <p>Labeled full-pipeline results for test-images. Inspect exact/wrong cases, or continue manual true/false review if needed.</p>
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
      <span>Total</span>
      <strong>{formatCount(summary.total)}</strong>
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
      <span>Exact</span>
      <strong>{formatCount(summary.exact_matches)} / {labeledCount || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Accuracy</span>
      <strong>{formatPercent(summary.accuracy)}</strong>
    </div>
    <div class="status-item">
      <span>Wrong Dates</span>
      <strong>{formatCount(summary.wrong_parsed_dates)}</strong>
    </div>
    <div class="status-item">
      <span>Checked</span>
      <strong>{checkedCount}/{allCount}</strong>
    </div>
    <div class="status-item">
      <span>True / False</span>
      <strong>{trueCount} / {falseCount}</strong>
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

  <div class="full-pipeline-filters" aria-label="Blind result filters">
    <button type="button" class:active={filter === 'all'} on:click={() => setFilter('all')}>All {allCount}</button>
    <button type="button" class:active={filter === 'unchecked'} on:click={() => setFilter('unchecked')}>Unchecked {uncheckedCount}</button>
    <button type="button" class:active={filter === 'true'} on:click={() => setFilter('true')}>True {trueCount}</button>
    <button type="button" class:active={filter === 'false'} on:click={() => setFilter('false')}>False {falseCount}</button>
    <button type="button" class:active={filter === 'exact'} on:click={() => setFilter('exact')}>Exact {exactCount}</button>
    <button type="button" class:active={filter === 'wrong'} on:click={() => setFilter('wrong')}>Wrong {wrongCount}</button>
    <button type="button" class:active={filter === 'parsed'} on:click={() => setFilter('parsed')}>Parsed {parsedCount}</button>
    <button type="button" class:active={filter === 'manual'} on:click={() => setFilter('manual')}>Manual {manualCount}</button>
  </div>

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Blind test image results">
      {#if filteredItems.length === 0}
        <p class="muted">No test-image results match this filter.</p>
      {:else}
        {#each filteredItems as item, index}
          <button type="button" class:active={item.filename === active?.filename} on:click={() => selectItem(item)}>
            <strong>{index + 1}. {item.filename}</strong>
            <span>{item.finalStatus || '-'} · {reviewLabel(item)}</span>
            <small>detected {item.predictedDate || '-'} · expected {item.expectedDate || '-'} · {formatRuntime(item.latencyMs)}</small>
          </button>
        {/each}
      {/if}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{active.filename}</h3>
            <p class="muted">Detected {active.predictedDate || '-'} · Expected {active.expectedDate || '-'} · {active.finalStatus || '-'}</p>
          </div>
          <span class={`outcome-pill pipeline-outcome-${outcomeClass(active)}`}>{reviewLabel(active)}</span>
        </div>

        <div class="detection-review-grid">
          <figure class="detection-review-image">
            {#if active.imageUrl}
              <div class="detection-review-image-frame">
                <img src={active.imageUrl} alt={`Blind test image ${active.filename}`} on:load={handleImageLoad} />
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
            <figcaption>Final recognizer bbox: {active.finalRecognitionBbox ? `[${formatBbox(active.finalRecognitionBbox)}]` : 'not available'}</figcaption>
          </figure>

          <div class="recognition-winner-card">
            <h3>Blind Result</h3>
            <dl class="compact-dl">
              <dt>Status</dt>
              <dd>{active.finalStatus || '-'}</dd>
              <dt>Detected</dt>
              <dd>{active.predictedDate || '-'}</dd>
              <dt>Expected</dt>
              <dd>{active.expectedDate || '-'}</dd>
              <dt>Exact Match</dt>
              <dd>{active.exactMatch === null ? '-' : active.exactMatch ? 'yes' : 'no'}</dd>
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

            <div class="blind-review-form">
              <h3>Post-Run Check</h3>
              <div class="full-pipeline-filters compact" aria-label="Review decision">
                <button type="button" class:active={draftCorrect === true} on:click={() => (draftCorrect = true)}>True</button>
                <button type="button" class:active={draftCorrect === false} on:click={() => (draftCorrect = false)}>False</button>
              </div>
              {#if draftCorrect === false}
                <label>
                  Correct date
                  <input bind:value={draftDate} placeholder="YYYY-MM-DD or YYYY-MM" />
                </label>
              {/if}
              <button type="button" on:click={saveReview} disabled={saving}>
                {saving ? 'Saving...' : 'Save Check'}
              </button>
            </div>
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
            <h3>Timing Profile</h3>
            <pre>{JSON.stringify(active.debugProfile, null, 2)}</pre>
          </article>
        </div>
      {:else}
        <p class="muted">Select a blind test-image result to inspect.</p>
      {/if}
    </section>
  </div>
</section>
