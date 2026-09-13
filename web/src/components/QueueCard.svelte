<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  type JsonRecord = Record<string, unknown>;

  export let item: JsonRecord;

  const dispatch = createEventDispatcher<{
    review: { item: JsonRecord };
    loadResult: { item: JsonRecord };
    useCorrection: { item: JsonRecord };
  }>();

  const statusText = (value: unknown): string => (typeof value === 'string' && value ? value : '-');

  const proxyPreviewUrl = (value: unknown): string => {
    if (typeof value !== 'string' || !value) return '';
    return value.startsWith('/scans/') ? `/api${value}` : value;
  };

  const reviewStatus = (value: unknown): string => (statusText(value) === '-' ? 'unreviewed' : statusText(value));

  const formatScanTime = (value: unknown): string => {
    if (typeof value !== 'string' || !value) return '-';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return new Intl.DateTimeFormat(undefined, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }).format(parsed);
  };

  const displayTitle = (): string => {
    const parsed = statusText(item.parsed_date);
    if (parsed !== '-') return parsed;
    const finalStatus = statusText(item.final_status);
    if (finalStatus !== '-') return finalStatus;
    return statusText(item.status);
  };
</script>

<article class={`queue-card verdict-${reviewStatus(item.review_verdict)}`}>
  <div class="queue-main">
    <div class="queue-thumbnail-wrap">
      {#if item.image_url}
        <img class="queue-thumbnail" src={proxyPreviewUrl(item.image_url)} alt="Scan thumbnail" />
      {:else}
        <div class="queue-thumbnail placeholder">No image</div>
      {/if}
    </div>

    <div class="queue-summary">
      <strong class="queue-title">{displayTitle()}</strong>
      <div class="queue-badges">
        <span class={`pill ${statusText(item.status)}`}>{statusText(item.status)}</span>
        <span class={`pill review-${reviewStatus(item.review_verdict)}`}>{reviewStatus(item.review_verdict)}</span>
      </div>
      <p class="muted">Created: {formatScanTime(item.created_at)}</p>
      {#if item.original_filename}
        <p class="muted mono-line">{statusText(item.original_filename)}</p>
      {/if}
    </div>

    <div class="queue-actions">
      <button type="button" class="secondary" on:click={() => dispatch('review', { item })}>Review</button>
    </div>
  </div>

  <details class="queue-details">
    <summary>Details</summary>
    <div class="queue-details-content">
      <div class="queue-debug-grid">
        <span><strong>Parsed</strong>{statusText(item.parsed_date)}</span>
        <span><strong>Final</strong>{statusText(item.final_status)}</span>
        <span><strong>Class</strong>{statusText(item.expiry_classification)}</span>
        <span><strong>Detector</strong>{typeof item.detector_confidence === 'number' ? item.detector_confidence.toFixed(3) : '-'}</span>
        <span><strong>OCR Device</strong>{statusText(item.ocr_runtime_device)}</span>
        <span><strong>Scan ID</strong><span class="mono-line">{statusText(item.scan_id)}</span></span>
      </div>
      <div class="history-ocr compact">
        <span>Raw OCR</span>
        <p>{statusText(item.raw_text)}</p>
      </div>
      {#if item.reason}
        <p class="muted">Reason: {statusText(item.reason)}</p>
      {/if}

      <div class="queue-detail-actions">
        <button type="button" class="ghost" on:click={() => dispatch('loadResult', { item })}>Load Result</button>
        <button type="button" class="ghost" on:click={() => dispatch('useCorrection', { item })}>Use For Correction</button>
      </div>

      <details>
        <summary>JSON Payload</summary>
        <pre>{JSON.stringify(item, null, 2)}</pre>
      </details>
    </div>
  </details>
</article>
