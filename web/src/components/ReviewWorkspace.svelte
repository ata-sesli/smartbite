<script lang="ts">
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import ImageAnnotator from './ImageAnnotator.svelte';
  import ModelOutputPanel from './ModelOutputPanel.svelte';
  import ReviewDecisionForm from './ReviewDecisionForm.svelte';

  type JsonRecord = Record<string, unknown>;

  export let item: JsonRecord | null = null;
  export let index = -1;
  export let total = 0;
  export let verdict = '';
  export let acceptedForTraining = false;
  export let finalText = '';
  export let finalParsedDate = '';
  export let notes = '';
  export let bbox: number[] | null = null;
  export let saving = false;

  const dispatch = createEventDispatcher<{
    previous: void;
    next: void;
    save: void;
    close: void;
    verdictChange: { verdict: string };
    bboxChange: { bbox: number[] | null };
  }>();

  const statusText = (value: unknown): string => (typeof value === 'string' && value ? value : '-');

  function onKeydown(event: KeyboardEvent): void {
    if (!item) return;
    const tag = (event.target as HTMLElement | null)?.tagName?.toLowerCase() ?? '';
    const isInputLike = tag === 'input' || tag === 'textarea' || tag === 'select';
    if (isInputLike && event.key !== 'Enter') return;

    if (event.key.toLowerCase() === 'y') {
      dispatch('verdictChange', { verdict: 'correct' });
      event.preventDefault();
      return;
    }
    if (event.key.toLowerCase() === 'n') {
      dispatch('verdictChange', { verdict: 'incorrect' });
      event.preventDefault();
      return;
    }
    if (event.key === 'Enter' && !saving) {
      dispatch('save');
      event.preventDefault();
    }
  }

  onMount(() => {
    window.addEventListener('keydown', onKeydown);
  });

  onDestroy(() => {
    window.removeEventListener('keydown', onKeydown);
  });
</script>

<section class="panel review-workspace-panel">
  <div class="review-workspace-header">
    <div>
      <h2>Review Workspace</h2>
      {#if item}
        <p class="muted">Reviewing {index + 1} of {total} scans</p>
        <p class="muted"><strong>Status:</strong> {statusText(item.review_verdict) === '-' ? 'unreviewed' : statusText(item.review_verdict)}</p>
        <p class="muted"><strong>Extracted Date:</strong> {statusText(item.parsed_date)}</p>
      {:else}
        <p class="muted">Select a scan from Queue to start reviewing.</p>
      {/if}
    </div>

    <div class="review-nav-actions">
      <button type="button" class="ghost" on:click={() => dispatch('previous')} disabled={index <= 0}>← Previous</button>
      <button type="button" class="ghost" on:click={() => dispatch('next')} disabled={index < 0 || index >= total - 1}>Next →</button>
      <button type="button" class="ghost" on:click={() => dispatch('close')} disabled={!item}>Close</button>
    </div>
  </div>

  {#if item}
    <div class="review-workspace-grid">
      <ImageAnnotator
        imageUrl={statusText(item.image_url) === '-' ? '' : statusText(item.image_url)}
        roiUrl={statusText(item.roi_url) === '-' ? '' : statusText(item.roi_url)}
        bind:bbox
        on:bboxChange={(event) => dispatch('bboxChange', event.detail)}
      />

      <ModelOutputPanel {item} />

      <div class="review-zone right-zone">
        <ReviewDecisionForm
          bind:verdict
          bind:acceptedForTraining
          bind:finalText
          bind:finalParsedDate
          bind:notes
          on:verdictChange={(event) => dispatch('verdictChange', event.detail)}
        />

        <button type="button" class="save-review-button" on:click={() => dispatch('save')} disabled={saving || !verdict}>
          {saving ? 'Saving...' : 'Save Review'}
        </button>

        <p class="muted shortcut-hint">Shortcuts: Y = Correct, N = Incorrect, Enter = Save</p>
      </div>
    </div>
  {/if}
</section>
