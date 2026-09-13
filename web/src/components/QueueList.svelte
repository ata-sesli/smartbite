<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import QueueCard from './QueueCard.svelte';

  type JsonRecord = Record<string, unknown>;

  export let items: JsonRecord[] = [];
  export let loading = false;

  const dispatch = createEventDispatcher<{
    refresh: void;
    review: { item: JsonRecord };
    loadResult: { item: JsonRecord };
    useCorrection: { item: JsonRecord };
  }>();
</script>

<section class="panel queue-panel">
  <div class="history-heading-row">
    <div class="panel-heading">
      <h2>Review Queue</h2>
      <p>Pick a scan and move through review one-by-one.</p>
    </div>
    <button type="button" class="secondary history-refresh" on:click={() => dispatch('refresh')} disabled={loading}>
      {loading ? 'Loading...' : 'Refresh Queue'}
    </button>
  </div>

  {#if items.length}
    <div class="queue-list">
      {#each items as item}
        <QueueCard
          {item}
          on:review={(event) => dispatch('review', event.detail)}
          on:loadResult={(event) => dispatch('loadResult', event.detail)}
          on:useCorrection={(event) => dispatch('useCorrection', event.detail)}
        />
      {/each}
    </div>
  {:else}
    <p class="muted">No scans in queue yet.</p>
  {/if}
</section>
