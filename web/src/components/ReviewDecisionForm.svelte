<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';

  export let verdict = '';
  export let finalParsedDate = '';
  export let finalText = '';
  export let notes = '';
  export let acceptedForTraining = false;

  let correctedDateInput: HTMLInputElement | null = null;
  const dispatch = createEventDispatcher<{ verdictChange: { verdict: string } }>();

  async function setVerdict(value: 'correct' | 'incorrect'): Promise<void> {
    verdict = value;
    dispatch('verdictChange', { verdict: value });
    if (value === 'incorrect') {
      await tick();
      correctedDateInput?.focus();
    }
  }
</script>

<section class="review-zone decision-zone">
  <h3>Decision</h3>

  <p class="muted">Is the prediction correct?</p>
  <div class="review-flow-actions">
    <button
      type="button"
      class:active-decision={verdict === 'correct'}
      class="secondary"
      on:click={() => setVerdict('correct')}>✅ Correct</button
    >
    <button
      type="button"
      class:active-decision={verdict === 'incorrect'}
      class="danger"
      on:click={() => setVerdict('incorrect')}>❌ Incorrect</button
    >
  </div>

  {#if verdict === 'incorrect'}
    <div class="review-form">
      <label>
        Corrected date
        <input bind:this={correctedDateInput} type="date" bind:value={finalParsedDate} />
      </label>
      <label>
        Feedback notes
        <textarea bind:value={notes} placeholder="Explain what was wrong and what was corrected."></textarea>
      </label>
      <label>
        Corrected text (optional)
        <input bind:value={finalText} placeholder="23.07.2026" />
      </label>
      <label class="checkbox-inline">
        <input type="checkbox" bind:checked={acceptedForTraining} /> Accepted for training export
      </label>
    </div>
  {:else if verdict === 'correct'}
    <label class="checkbox-inline">
      <input type="checkbox" bind:checked={acceptedForTraining} /> Accepted for training export
    </label>
  {/if}
</section>
