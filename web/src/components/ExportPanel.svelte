<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  type JsonRecord = Record<string, unknown>;

  export let outputDir = '';
  export let includeDetector = true;
  export let includeRecognition = true;
  export let result: JsonRecord | null = null;
  export let exporting = false;

  const dispatch = createEventDispatcher<{ export: void }>();
</script>

<section class="panel dataset-panel">
  <div class="panel-heading">
    <h2>Dataset / Export</h2>
    <p>Run batch export for accepted reviews.</p>
  </div>

  <div class="dataset-grid">
    <label>
      Output directory (optional)
      <input bind:value={outputDir} placeholder="default: storage/training_exports/review_export" />
    </label>

    <div class="checkbox-row">
      <label><input type="checkbox" bind:checked={includeDetector} /> Detector labels</label>
      <label><input type="checkbox" bind:checked={includeRecognition} /> Recognition crops</label>
    </div>

    <button type="button" class="secondary" on:click={() => dispatch('export')} disabled={exporting}>
      {exporting ? 'Exporting...' : 'Export Accepted Reviews'}
    </button>
  </div>

  {#if result}
    <details open>
      <summary>Export Result</summary>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </details>
  {/if}
</section>
