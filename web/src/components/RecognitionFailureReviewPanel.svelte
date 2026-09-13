<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type NoticeKind = 'success' | 'error' | 'info';

  type ReviewVariant = {
    variantName: string;
    cropUrl: string;
    rawText: string;
    normalizedText: string;
    confidence: number | null;
    runtimeMs: number | null;
    parserParsedDate: string;
    parserDatePrecision: string;
    parserParsedDay: number | null;
    parserParsedMonth: number | null;
    parserParsedYear: number | null;
    parserConfidence: number | null;
    parserReason: string;
    exactMatch: boolean;
    malformedRecoverable: boolean;
    genericText: boolean;
    cropTransformUsed: string;
    selectedOrientation: string;
    orientationCandidatesTried: string[];
    originalCropShape: number[] | null;
    normalizedCropShape: number[] | null;
    selectedTransformReason: string;
  };

  type ReviewItem = {
    filename: string;
    expectedDate: string;
    outcome: string;
    trueBbox: number[] | null;
    manualTrueCropUrl: string;
    manualTrueCropNormalizedUrl: string;
    winner: ReviewVariant | null;
    variants: ReviewVariant[];
  };

  type Notice = {
    kind: NoticeKind;
    text: string;
  } | null;

  let loading = false;
  let notice: Notice = null;
  let includeSuccess = false;
  let runId = '';
  let generatedAt = '';
  let recognizer = '';
  let totalItems = 0;
  let filteredCount = 0;
  let summary: JsonRecord = {};
  let items: ReviewItem[] = [];
  let activeIndex = 0;
  let active: ReviewItem | null = null;

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

  function bboxFrom(value: unknown): number[] | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    const parsed = value.map((item) => (typeof item === 'number' ? item : Number(item)));
    return parsed.every((item) => Number.isFinite(item)) ? parsed : null;
  }

  function stringList(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  }

  function numberList(value: unknown): number[] | null {
    if (!Array.isArray(value)) return null;
    const parsed = value.map((item) => (typeof item === 'number' ? item : Number(item)));
    return parsed.every((item) => Number.isFinite(item)) ? parsed : null;
  }

  function apiAssetUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value.replace(
      '/test64/manual-crop-recognition-review/assets',
      '/api/test64/manual-crop-recognition-review/assets'
    );
  }

  function mapVariant(raw: JsonRecord | null): ReviewVariant | null {
    if (!raw) return null;
    return {
      variantName: text(raw.variant_name),
      cropUrl: apiAssetUrl(raw.crop_url),
      rawText: text(raw.raw_text),
      normalizedText: text(raw.normalized_text),
      confidence: numberOrNull(raw.confidence),
      runtimeMs: numberOrNull(raw.runtime_ms),
      parserParsedDate: text(raw.parser_parsed_date),
      parserDatePrecision: text(raw.parser_date_precision),
      parserParsedDay: numberOrNull(raw.parser_parsed_day),
      parserParsedMonth: numberOrNull(raw.parser_parsed_month),
      parserParsedYear: numberOrNull(raw.parser_parsed_year),
      parserConfidence: numberOrNull(raw.parser_confidence),
      parserReason: text(raw.parser_reason),
      exactMatch: raw.exact_match === true,
      malformedRecoverable: raw.malformed_but_potentially_recoverable === true,
      genericText: raw.generic_text_output === true,
      cropTransformUsed: text(raw.crop_transform_used),
      selectedOrientation: text(raw.selected_orientation),
      orientationCandidatesTried: stringList(raw.orientation_candidates_tried),
      originalCropShape: numberList(raw.original_crop_shape),
      normalizedCropShape: numberList(raw.normalized_crop_shape),
      selectedTransformReason: text(raw.selected_transform_reason)
    };
  }

  function winnerFrom(raw: unknown): ReviewVariant | null {
    if (!isRecord(raw)) return null;
    return {
      variantName: text(raw.variant_name),
      cropUrl: apiAssetUrl(raw.crop_url),
      rawText: text(raw.parseq_raw_output ?? raw.raw_text),
      normalizedText: text(raw.parseq_normalized_output ?? raw.normalized_text),
      confidence: numberOrNull(raw.parseq_confidence ?? raw.confidence),
      runtimeMs: numberOrNull(raw.parseq_runtime_ms ?? raw.runtime_ms),
      parserParsedDate: text(raw.parser_parsed_date),
      parserDatePrecision: text(raw.parser_date_precision),
      parserParsedDay: numberOrNull(raw.parser_parsed_day),
      parserParsedMonth: numberOrNull(raw.parser_parsed_month),
      parserParsedYear: numberOrNull(raw.parser_parsed_year),
      parserConfidence: numberOrNull(raw.parser_confidence),
      parserReason: text(raw.parser_reason),
      exactMatch: raw.exact_match === true,
      malformedRecoverable: raw.malformed_but_potentially_recoverable === true,
      genericText: raw.generic_text_output === true,
      cropTransformUsed: text(raw.crop_transform_used),
      selectedOrientation: text(raw.selected_orientation),
      orientationCandidatesTried: stringList(raw.orientation_candidates_tried),
      originalCropShape: numberList(raw.original_crop_shape),
      normalizedCropShape: numberList(raw.normalized_crop_shape),
      selectedTransformReason: text(raw.selected_transform_reason)
    };
  }

  function mapItem(raw: JsonRecord): ReviewItem {
    const variants = Array.isArray(raw.variants)
      ? raw.variants.map((variant) => (isRecord(variant) ? mapVariant(variant) : null)).filter((variant): variant is ReviewVariant => variant !== null)
      : [];
    return {
      filename: text(raw.filename),
      expectedDate: text(raw.expected_date),
      outcome: text(raw.outcome),
      trueBbox: bboxFrom(raw.true_bbox_xyxy),
      manualTrueCropUrl: apiAssetUrl(raw.manual_true_crop_url),
      manualTrueCropNormalizedUrl: apiAssetUrl(raw.manual_true_crop_normalized_url),
      winner: winnerFrom(raw.winner),
      variants
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
    const search = new URLSearchParams();
    if (includeSuccess) search.set('include_success', 'true');
    try {
      const payload = await handleResponse(
        await fetch(`/api/test64/manual-crop-recognition-review${search.toString() ? `?${search}` : ''}`)
      );
      const rawItems = Array.isArray(payload.items) ? payload.items : [];
      items = rawItems.map((item) => (isRecord(item) ? mapItem(item) : null)).filter((item): item is ReviewItem => item !== null);
      runId = text(payload.run_id);
      generatedAt = text(payload.generated_at);
      summary = isRecord(payload.summary) ? payload.summary : {};
      recognizer = isRecord(payload.recognizer_config) ? text(payload.recognizer_config.recognizer) : '';
      totalItems = numberOrNull(payload.total_items) ?? items.length;
      filteredCount = numberOrNull(payload.filtered_count) ?? items.length;
      activeIndex = Math.min(activeIndex, Math.max(0, items.length - 1));
      notice = { kind: 'success', text: `Loaded ${items.length} recognition review items.` };
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

  function formatBbox(value: number[] | null): string {
    return value ? value.map((item) => Math.round(item)).join(', ') : 'not set';
  }

  function formatConfidence(value: number | null): string {
    return value === null ? '-' : value.toFixed(3);
  }

  function formatRuntime(value: number | null): string {
    return value === null ? '-' : `${Math.round(value)} ms`;
  }

  function formatShape(value: number[] | null): string {
    return value ? value.map((item) => Math.round(item)).join(' × ') : '-';
  }

  function formatParsed(variant: ReviewVariant): string {
    if (variant.parserParsedDate) return variant.parserParsedDate;
    if (variant.parserDatePrecision === 'month' && variant.parserParsedMonth && variant.parserParsedYear) {
      return `${String(variant.parserParsedMonth).padStart(2, '0')}/${variant.parserParsedYear}`;
    }
    return '-';
  }

  function transformLabel(variant: ReviewVariant): string {
    if (variant.cropTransformUsed) return variant.cropTransformUsed;
    if (variant.selectedOrientation) return variant.selectedOrientation;
    const prefix = variant.variantName.split('/')[0] ?? '';
    return prefix && prefix !== variant.variantName ? prefix : '-';
  }

  onMount(() => {
    void loadReview();
  });
</script>

<section class="panel recognition-review-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>Recognition Failure Review</h2>
      <p>Manual bbox crops only. No detection, ranking, or crop selection.</p>
    </div>
    <div class="recognition-review-actions">
      <label class="inline-toggle">
        <input type="checkbox" bind:checked={includeSuccess} on:change={() => loadReview()} />
        Include successes
      </label>
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
      <span>Recognizer</span>
      <strong>{recognizer || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Exact</span>
      <strong>{formatCount(summary.exact_match_crops)}/{formatCount(summary.total_crops)}</strong>
    </div>
    <div class="status-item">
      <span>Shown</span>
      <strong>{filteredCount}/{totalItems}</strong>
    </div>
  </div>

  {#if generatedAt}
    <p class="muted">Generated: {generatedAt}</p>
  {/if}

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Manual crop recognition review items">
      {#if items.length === 0}
        <p class="muted">No review items available.</p>
      {:else}
        {#each items as item, index}
          <button type="button" class:active={index === activeIndex} on:click={() => selectIndex(index)}>
            <strong>{index + 1}. {item.filename}</strong>
            <span>{item.outcome || 'unknown'}</span>
            <small>Expected {item.expectedDate || '-'}</small>
          </button>
        {/each}
      {/if}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{active.filename}</h3>
            <p class="muted">Expected {active.expectedDate || '-'} · bbox {formatBbox(active.trueBbox)}</p>
          </div>
          <span class="outcome-pill">{active.outcome || 'unknown'}</span>
        </div>

        <div class="recognition-crop-grid">
          <figure>
            {#if active.manualTrueCropNormalizedUrl || active.manualTrueCropUrl}
              <img src={active.manualTrueCropNormalizedUrl || active.manualTrueCropUrl} alt={`Manual true crop for ${active.filename}`} />
            {:else}
              <div class="recognition-empty-crop">No crop image</div>
            {/if}
            <figcaption>{active.manualTrueCropNormalizedUrl ? 'Manual normalized crop' : 'Manual true crop'}</figcaption>
          </figure>

          <div class="recognition-winner-card">
            <h3>Winner</h3>
            {#if active.winner}
              <dl class="compact-dl">
                <dt>Variant</dt>
                <dd>{active.winner.variantName || '-'}</dd>
                <dt>Transform</dt>
                <dd>{transformLabel(active.winner)}</dd>
                <dt>Raw</dt>
                <dd>{active.winner.rawText || '-'}</dd>
                <dt>Parsed</dt>
                <dd>{formatParsed(active.winner)}</dd>
                <dt>Confidence</dt>
                <dd>{formatConfidence(active.winner.confidence)}</dd>
                <dt>Exact</dt>
                <dd>{active.winner.exactMatch ? 'yes' : 'no'}</dd>
                <dt>Shape</dt>
                <dd>{formatShape(active.winner.originalCropShape)} → {formatShape(active.winner.normalizedCropShape)}</dd>
              </dl>
            {:else}
              <p class="muted">No winner recorded.</p>
            {/if}
          </div>
        </div>

        <div class="recognition-variant-grid">
          {#each active.variants as variant}
            <article class:exact={variant.exactMatch} class:recoverable={variant.malformedRecoverable} class:generic={variant.genericText}>
              <header>
                <h3>{variant.variantName || 'variant'}</h3>
                <span>{variant.exactMatch ? 'exact' : variant.malformedRecoverable ? 'recoverable' : variant.genericText ? 'generic' : 'miss'}</span>
              </header>
              {#if variant.cropUrl}
                <img src={variant.cropUrl} alt={`${active.filename} ${variant.variantName} crop`} />
              {/if}
              <dl class="compact-dl">
                <dt>Raw</dt>
                <dd>{variant.rawText || '-'}</dd>
                <dt>Normalized</dt>
                <dd>{variant.normalizedText || '-'}</dd>
                <dt>Parsed</dt>
                <dd>{formatParsed(variant)}</dd>
                <dt>Transform</dt>
                <dd>{transformLabel(variant)}</dd>
                <dt>Orientation</dt>
                <dd>{variant.selectedOrientation || '-'}</dd>
                <dt>Parser</dt>
                <dd>{variant.parserReason || '-'}</dd>
                <dt>Confidence</dt>
                <dd>{formatConfidence(variant.confidence)}</dd>
                <dt>Runtime</dt>
                <dd>{formatRuntime(variant.runtimeMs)}</dd>
                <dt>Shape</dt>
                <dd>{formatShape(variant.originalCropShape)} → {formatShape(variant.normalizedCropShape)}</dd>
              </dl>
            </article>
          {/each}
        </div>
      {:else}
        <p class="muted">Select a recognition failure to inspect.</p>
      {/if}
    </section>
  </div>
</section>
