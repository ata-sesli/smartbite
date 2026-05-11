<script lang="ts">
  type JsonRecord = Record<string, unknown>;

  export let item: JsonRecord | null = null;

  const statusText = (value: unknown): string => (typeof value === 'string' && value ? value : '-');
</script>

<section class="review-zone model-output-zone">
  <h3>Model Output</h3>

  {#if item}
    <div class="review-model-grid">
      <span><strong>Parsed Date</strong>{statusText(item.parsed_date)}</span>
      <span><strong>Final Status</strong>{statusText(item.final_status)}</span>
      <span><strong>Classification</strong>{statusText(item.expiry_classification)}</span>
      <span><strong>Detector Conf.</strong>{typeof item.detector_confidence === 'number' ? item.detector_confidence.toFixed(3) : '-'}</span>
      <span><strong>Reason</strong>{statusText(item.reason)}</span>
      <span><strong>OCR Device</strong>{statusText(item.ocr_runtime_device)}</span>
    </div>

    <details open>
      <summary>OCR Text</summary>
      <p class="review-ocr-text">{statusText(item.raw_text)}</p>
    </details>
  {:else}
    <p class="muted">Select an item from Queue to inspect model output.</p>
  {/if}
</section>
