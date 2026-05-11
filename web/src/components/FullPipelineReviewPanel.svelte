<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type NoticeKind = 'success' | 'error' | 'info';

  type PipelineItem = {
    filename: string;
    imageUrl: string;
    verdict: string;
    predictedDate: string;
    finalStatus: string;
    reason: string;
    parseableCandidates: number | null;
    selectedCandidates: number | null;
    runtimeMs: number | null;
    detectorCounts: JsonRecord;
    detectorPerformance: JsonRecord;
  };

  type Notice = {
    kind: NoticeKind;
    text: string;
  } | null;

  let loading = false;
  let notice: Notice = null;
  let runId = '';
  let source = '';
  let textDetectorMode = '';
  let detectorConfig: JsonRecord = {};
  let detectorTotals: JsonRecord = {};
  let summary: JsonRecord = {};
  let parseableCandidatesTotal: number | null = null;
  let averageRuntimeMs: number | null = null;
  let items: PipelineItem[] = [];
  let activeIndex = 0;

  $: active = items[activeIndex] ?? null;

  function isRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function numberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function text(value: unknown): string {
    return typeof value === 'string' ? value : '';
  }

  function apiImageUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value.replace('/test64/images/', '/api/test64/images/');
  }

  function mapItem(raw: unknown): PipelineItem | null {
    if (!isRecord(raw)) return null;
    return {
      filename: text(raw.filename),
      imageUrl: apiImageUrl(raw.image_url),
      verdict: text(raw.verdict),
      predictedDate: text(raw.predicted_date),
      finalStatus: text(raw.final_status),
      reason: text(raw.reason),
      parseableCandidates: numberOrNull(raw.parseable_candidates),
      selectedCandidates: numberOrNull(raw.selected_candidates),
      runtimeMs: numberOrNull(raw.runtime_ms),
      detectorCounts: isRecord(raw.detector_counts) ? raw.detector_counts : {},
      detectorPerformance: isRecord(raw.detector_performance) ? raw.detector_performance : {}
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

  async function loadReview(): Promise<void> {
    loading = true;
    notice = null;
    try {
      const payload = await handleResponse(await fetch('/api/test64/full-pipeline-review'));
      runId = text(payload.run_id);
      source = text(payload.source);
      textDetectorMode = text(payload.text_detector_mode);
      detectorConfig = isRecord(payload.detector_config) ? payload.detector_config : {};
      detectorTotals = isRecord(payload.detector_totals) ? payload.detector_totals : {};
      summary = isRecord(payload.summary) ? payload.summary : {};
      parseableCandidatesTotal = numberOrNull(payload.parseable_candidates_total);
      averageRuntimeMs = numberOrNull(payload.average_runtime_ms_per_scan);
      items = Array.isArray(payload.items)
        ? payload.items.map(mapItem).filter((item): item is PipelineItem => item !== null)
        : [];
      activeIndex = Math.min(activeIndex, Math.max(0, items.length - 1));
      notice = { kind: 'success', text: `Loaded ${items.length} full-pipeline items.` };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      notice = { kind: 'error', text: message };
    } finally {
      loading = false;
    }
  }

  function selectIndex(index: number): void {
    activeIndex = Math.max(0, Math.min(index, items.length - 1));
  }

  function formatCount(value: unknown): string {
    return typeof value === 'number' ? String(value) : '-';
  }

  function formatRuntime(value: number | null): string {
    return value === null ? '-' : `${Math.round(value)} ms`;
  }

  function formatDecimal(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '-';
  }

  function verdictLabel(value: string): string {
    return value ? value.replaceAll('_', ' ') : 'unknown';
  }

  function detectorCount(key: string): string {
    if (!active) return '-';
    return formatCount(active.detectorCounts[key]);
  }

  function performanceValue(key: string): string {
    if (!active) return '-';
    const value = active.detectorPerformance[key];
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
    if (typeof value === 'string') return value;
    return '-';
  }

  onMount(() => {
    void loadReview();
  });
</script>

<section class="panel recognition-review-panel full-pipeline-review-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>Full Pipeline Review</h2>
      <p>End-to-end test64 output: detection, ranking, recognition, parser, and final verdict.</p>
    </div>
    <div class="recognition-review-actions">
      <button type="button" on:click={() => loadReview()} disabled={loading}>
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
      <span>Detector</span>
      <strong>{textDetectorMode || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Correct</span>
      <strong>{formatCount(summary.correct_guess)}/{formatCount(summary.total)}</strong>
    </div>
    <div class="status-item">
      <span>Recognition Fail</span>
      <strong>{formatCount(summary.recognition_fail)}</strong>
    </div>
    <div class="status-item">
      <span>Wrong</span>
      <strong>{formatCount(summary.wrong_guess)}</strong>
    </div>
    <div class="status-item">
      <span>Parseable</span>
      <strong>{formatCount(parseableCandidatesTotal)}</strong>
    </div>
    <div class="status-item">
      <span>Avg Runtime</span>
      <strong>{formatDecimal(averageRuntimeMs)} ms</strong>
    </div>
    <div class="status-item">
      <span>CRAFT</span>
      <strong>{detectorConfig.craft_enabled === false ? 'off' : detectorConfig.craft_enabled === true ? 'on' : '-'}</strong>
    </div>
  </div>

  {#if source}
    <p class="muted">Source: {source}</p>
  {/if}

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Full pipeline review items">
      {#if items.length === 0}
        <p class="muted">No full pipeline items available.</p>
      {:else}
        {#each items as item, index}
          <button type="button" class:active={index === activeIndex} on:click={() => selectIndex(index)}>
            <strong>{index + 1}. {item.filename}</strong>
            <span>{verdictLabel(item.verdict)}</span>
            <small>pred {item.predictedDate || '-'} · {formatRuntime(item.runtimeMs)}</small>
          </button>
        {/each}
      {/if}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{active.filename}</h3>
            <p class="muted">Predicted {active.predictedDate || '-'} · {active.finalStatus || '-'}</p>
          </div>
          <span class="outcome-pill">{verdictLabel(active.verdict)}</span>
        </div>

        <div class="detection-review-grid">
          <figure class="detection-review-image">
            {#if active.imageUrl}
              <div class="detection-review-image-frame">
                <img src={active.imageUrl} alt={`Full pipeline source ${active.filename}`} />
              </div>
            {:else}
              <div class="recognition-empty-crop">No image</div>
            {/if}
          </figure>

          <div class="recognition-winner-card">
            <h3>Pipeline Flow</h3>
            <dl class="compact-dl">
              <dt>PP-OCRv5</dt>
              <dd>{detectorCount('ppocrv5_server_boxes')}</dd>
              <dt>CRAFT</dt>
              <dd>{detectorCount('craft_boxes')}</dd>
              <dt>Ensemble</dt>
              <dd>{detectorCount('ensemble_boxes')}</dd>
              <dt>Dedupe</dt>
              <dd>{performanceValue('detected_boxes_after_dedupe')}</dd>
              <dt>Probe</dt>
              <dd>{performanceValue('candidates_sent_to_mobile_probe')}</dd>
              <dt>Final Calls</dt>
              <dd>{performanceValue('final_recognition_call_count')}</dd>
              <dt>Selected</dt>
              <dd>{formatCount(active.selectedCandidates)}</dd>
              <dt>Parseable</dt>
              <dd>{formatCount(active.parseableCandidates)}</dd>
              <dt>Runtime</dt>
              <dd>{formatRuntime(active.runtimeMs)}</dd>
            </dl>
          </div>
        </div>

        <div class="detection-performance-grid">
          <article>
            <h3>Reason</h3>
            <p>{active.reason || '-'}</p>
          </article>
          <article>
            <h3>Detector Totals</h3>
            <pre>{JSON.stringify(detectorTotals, null, 2)}</pre>
          </article>
        </div>
      {:else}
        <p class="muted">Select a full-pipeline result to inspect.</p>
      {/if}
    </section>
  </div>
</section>
