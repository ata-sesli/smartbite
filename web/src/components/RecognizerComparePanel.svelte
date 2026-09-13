<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type Notice = {
    kind: 'success' | 'error' | 'info';
    text: string;
  } | null;

  type CompareResult = {
    label: string;
    runId: string;
    outcome: string;
    exactMatch: boolean;
    variantName: string;
    rawText: string;
    normalizedText: string;
    parsed: string;
    confidence: number | null;
    runtimeMs: number | null;
    parserReason: string;
    cropTransformUsed: string;
    selectedOrientation: string;
    winnerCropUrl: string;
  };

  type CompareRow = {
    filename: string;
    expected: string;
    notes: string;
    manualCropUrl: string;
    results: CompareResult[];
  };

  type ReportInfo = {
    label: string;
    reportPath: string;
    runId: string;
    generatedAt: string;
    modelDir: string;
    summary: JsonRecord;
  };

  let loading = false;
  let notice: Notice = null;
  let sourcePath = '';
  let updatedAt = '';
  let reports: ReportInfo[] = [];
  let rows: CompareRow[] = [];
  let activeIndex = 0;
  let active: CompareRow | null = null;
  let showOnlyChanged = false;

  $: changedRows = rows.filter(rowHasMiss);
  $: visibleRows = showOnlyChanged ? changedRows : rows;
  $: active = visibleRows[activeIndex] ?? visibleRows[0] ?? null;
  $: if (activeIndex >= visibleRows.length) activeIndex = Math.max(0, visibleRows.length - 1);

  function isRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function text(value: unknown): string {
    return typeof value === 'string' ? value : '';
  }

  function numberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function mapResult(value: unknown): CompareResult | null {
    if (!isRecord(value)) return null;
    return {
      label: text(value.label),
      runId: text(value.run_id),
      outcome: text(value.outcome),
      exactMatch: value.exact_match === true,
      variantName: text(value.variant_name),
      rawText: text(value.raw_text),
      normalizedText: text(value.normalized_text),
      parsed: text(value.parsed),
      confidence: numberOrNull(value.confidence),
      runtimeMs: numberOrNull(value.runtime_ms),
      parserReason: text(value.parser_reason),
      cropTransformUsed: text(value.crop_transform_used),
      selectedOrientation: text(value.selected_orientation),
      winnerCropUrl: text(value.winner_crop_url)
    };
  }

  function mapRow(value: unknown): CompareRow | null {
    if (!isRecord(value)) return null;
    return {
      filename: text(value.filename),
      expected: text(value.expected),
      notes: text(value.notes),
      manualCropUrl: text(value.manual_crop_url),
      results: Array.isArray(value.results)
        ? value.results.map(mapResult).filter((result): result is CompareResult => result !== null)
        : []
    };
  }

  function mapReport(value: unknown): ReportInfo | null {
    if (!isRecord(value)) return null;
    return {
      label: text(value.label),
      reportPath: text(value.report_path),
      runId: text(value.run_id),
      generatedAt: text(value.generated_at),
      modelDir: text(value.model_dir),
      summary: isRecord(value.summary) ? value.summary : {}
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

  async function loadCompare(): Promise<void> {
    loading = true;
    notice = null;
    try {
      const payload = await handleResponse(await fetch('/api/test64/recognizer-compare'));
      sourcePath = text(payload.source_path);
      updatedAt = text(payload.updated_at);
      reports = Array.isArray(payload.reports)
        ? payload.reports.map(mapReport).filter((report): report is ReportInfo => report !== null)
        : [];
      rows = Array.isArray(payload.rows)
        ? payload.rows.map(mapRow).filter((row): row is CompareRow => row !== null)
        : [];
      activeIndex = Math.min(activeIndex, Math.max(0, visibleRows.length - 1));
      notice = { kind: 'success', text: `Loaded ${rows.length} image-by-image recognizer comparisons.` };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      notice = { kind: 'error', text: message };
    } finally {
      loading = false;
    }
  }

  function selectIndex(index: number): void {
    activeIndex = Math.max(0, Math.min(index, visibleRows.length - 1));
  }

  function noteLines(value: string): string[] {
    return value
      .split('<br>')
      .map((line) => line.trim())
      .filter(Boolean);
  }

  function formatCount(value: unknown): string {
    return typeof value === 'number' ? String(value) : '-';
  }

  function formatConfidence(value: number | null): string {
    return value === null ? '-' : value.toFixed(3);
  }

  function formatRuntime(value: number | null): string {
    return value === null ? '-' : `${Math.round(value)} ms`;
  }

  function statusLabel(result: CompareResult): string {
    return result.exactMatch ? 'OK' : 'MISS';
  }

  function rowHasMiss(row: CompareRow): boolean {
    return row.results.some((result) => !result.exactMatch);
  }

  function resultSummary(row: CompareRow): string {
    return row.results.map((result) => `${result.label}:${statusLabel(result)}`).join(' ');
  }

  onMount(() => {
    void loadCompare();
  });
</script>

<section class="panel recognition-review-panel recognizer-compare-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>Recognizer Compare</h2>
      <p>Manual true crop, expected label, and recognizer winners side-by-side.</p>
    </div>
    <div class="recognition-review-actions">
      <label class="inline-toggle">
        <input type="checkbox" bind:checked={showOnlyChanged} on:change={() => selectIndex(0)} />
        Changed only
      </label>
      <button type="button" on:click={() => loadCompare()} disabled={loading}>
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
      <span>Images</span>
      <strong>{rows.length || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Changed</span>
      <strong>{changedRows.length || '-'}</strong>
    </div>
    {#each reports as report}
      <div class="status-item">
        <span>{report.label}</span>
        <strong>{formatCount(report.summary.exact_match_crops)}/{formatCount(report.summary.total_crops)}</strong>
      </div>
    {/each}
  </div>

  <p class="muted">
    Source: {sourcePath || '-'} · Updated: {updatedAt ? new Date(updatedAt).toLocaleString() : '-'}
  </p>

  <div class="recognition-review-layout recognizer-compare-layout">
    <aside class="recognition-review-list" aria-label="Recognizer comparison images">
      {#if visibleRows.length === 0}
        <p class="muted">No comparison rows available.</p>
      {:else}
        {#each visibleRows as row, index}
          <button
            type="button"
            class:active={row.filename === active?.filename}
            class:different={rowHasMiss(row)}
            on:click={() => selectIndex(index)}
          >
            <strong>{index + 1}. {row.filename}</strong>
            <span>{resultSummary(row)}</span>
            <small>Expected {row.expected || '-'}</small>
          </button>
        {/each}
      {/if}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{active.filename}</h3>
            <p class="muted">Expected {active.expected || '-'}</p>
          </div>
        </div>

        <div class="recognizer-compare-crop-row">
          <figure>
            {#if active.manualCropUrl}
              <img src={active.manualCropUrl} alt={`Manual true crop for ${active.filename}`} />
            {:else}
              <div class="recognition-empty-crop">No crop image</div>
            {/if}
            <figcaption>Manual true crop used for comparison</figcaption>
          </figure>

          {#if noteLines(active.notes).length}
            <div class="recognizer-compare-notes">
              <h3>Changed Case Notes</h3>
              {#each noteLines(active.notes) as line}
                <p>{line}</p>
              {/each}
            </div>
          {/if}
        </div>

        <div class="recognizer-result-grid">
          {#each active.results as result}
            <article class:exact={result.exactMatch}>
              <header>
                <div>
                  <h3>{result.label}</h3>
                  <p class="muted">{result.runId}</p>
                </div>
                <span class={`compare-pill ${result.exactMatch ? 'ok' : 'miss'}`}>{statusLabel(result)}</span>
              </header>

              {#if result.winnerCropUrl}
                <img src={result.winnerCropUrl} alt={`${active.filename} ${result.label} winner crop`} />
              {/if}

              <dl class="compact-dl">
                <dt>Raw</dt>
                <dd>{result.rawText || '-'}</dd>
                <dt>Normalized</dt>
                <dd>{result.normalizedText || '-'}</dd>
                <dt>Parsed</dt>
                <dd>{result.parsed || '-'}</dd>
                <dt>Outcome</dt>
                <dd>{result.outcome || '-'}</dd>
                <dt>Variant</dt>
                <dd>{result.variantName || '-'}</dd>
                <dt>Transform</dt>
                <dd>{result.cropTransformUsed || '-'}</dd>
                <dt>Orientation</dt>
                <dd>{result.selectedOrientation || '-'}</dd>
                <dt>Confidence</dt>
                <dd>{formatConfidence(result.confidence)}</dd>
                <dt>Runtime</dt>
                <dd>{formatRuntime(result.runtimeMs)}</dd>
                <dt>Parser</dt>
                <dd>{result.parserReason || '-'}</dd>
              </dl>
            </article>
          {/each}
        </div>
      {:else}
        <p class="muted">Select a crop to compare recognizer outputs.</p>
      {/if}
    </section>
  </div>
</section>

<style>
  .recognizer-compare-layout {
    grid-template-columns: minmax(290px, 0.34fr) minmax(0, 1fr);
  }

  .recognizer-compare-panel :global(.recognition-review-list button.different) {
    background: #fff7ed;
    border-color: #fdba74;
  }

  .recognizer-compare-panel :global(.recognition-review-list button.different span) {
    color: #9a3412;
  }

  .recognizer-compare-panel :global(.recognition-review-list button.different.active) {
    background: #ffedd5;
    border-color: #f97316;
  }

  .recognizer-compare-crop-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 0.45fr);
    gap: 0.8rem;
    align-items: start;
    margin-bottom: 0.9rem;
  }

  .recognizer-compare-crop-row figure,
  .recognizer-result-grid article,
  .recognizer-compare-notes {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #ffffff;
  }

  .recognizer-compare-crop-row figure {
    margin: 0;
    padding: 0.65rem;
  }

  .recognizer-compare-crop-row img {
    width: 100%;
    max-height: 220px;
    object-fit: contain;
    background: #0f1822;
    border-radius: 6px;
  }

  .recognizer-compare-crop-row figcaption {
    margin-top: 0.45rem;
    color: var(--secondary);
    font-size: 0.82rem;
  }

  .recognizer-compare-notes {
    padding: 0.7rem;
  }

  .recognizer-compare-notes h3 {
    margin: 0 0 0.45rem;
  }

  .recognizer-compare-notes p {
    margin: 0.2rem 0;
    color: var(--secondary);
  }

  .recognizer-result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.75rem;
  }

  .recognizer-result-grid article {
    padding: 0.7rem;
  }

  .recognizer-result-grid article.exact {
    border-color: rgba(22, 101, 52, 0.28);
  }

  .recognizer-result-grid header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.55rem;
  }

  .recognizer-result-grid h3 {
    margin: 0;
  }

  .recognizer-result-grid img {
    width: 100%;
    height: 110px;
    object-fit: contain;
    background: #0f1822;
    border-radius: 6px;
    margin-bottom: 0.6rem;
  }

  .compare-pill {
    display: inline-flex;
    min-width: 3.4rem;
    justify-content: center;
    border-radius: 999px;
    padding: 0.15rem 0.45rem;
    font-weight: 800;
    font-size: 0.72rem;
  }

  .compare-pill.ok {
    background: #dcfce7;
    color: #166534;
  }

  .compare-pill.miss {
    background: #fee2e2;
    color: #991b1b;
  }

  @media (max-width: 980px) {
    .recognizer-result-grid,
    .recognizer-compare-crop-row {
      grid-template-columns: 1fr;
    }
  }
</style>
