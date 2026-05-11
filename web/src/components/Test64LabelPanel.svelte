<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type NoticeKind = 'success' | 'error' | 'info';

  type Notice = {
    kind: NoticeKind;
    text: string;
  } | null;

  type LabelRow = {
    filename: string;
    imageUrl: string;
    expectedDay: number | null;
    expectedMonth: number | null;
    expectedYear: number | null;
    updatedAt: string | null;
    dayInput: string;
    monthInput: string;
    yearInput: string;
    saving: boolean;
    error: string;
    saveState: 'idle' | 'saved' | 'error';
    saveMessage: string;
    imageExpanded: boolean;
  };

  let rows: LabelRow[] = [];
  let loading = false;
  let notice: Notice = null;
  let totalCandidates = 0;
  let labeledCount = 0;

  function isJsonRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function toIntOrNull(value: unknown): number | null {
    if (value === null || value === undefined) return null;
    if (typeof value === 'number') {
      return Number.isFinite(value) ? Math.trunc(value) : null;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (!trimmed) return null;
      const parsed = Number.parseInt(trimmed, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
  }

  function toInput(value: number | null): string {
    return value === null ? '' : String(value);
  }

  function mapRow(raw: JsonRecord): LabelRow {
    const filename = typeof raw.filename === 'string' ? raw.filename : '';
    const expectedDay = typeof raw.expected_day === 'number' ? raw.expected_day : null;
    const expectedMonth = typeof raw.expected_month === 'number' ? raw.expected_month : null;
    const expectedYear = typeof raw.expected_year === 'number' ? raw.expected_year : null;
    const updatedAt = typeof raw.updated_at === 'string' ? raw.updated_at : null;
    const isLabeled = expectedMonth !== null && expectedYear !== null;
    return {
      filename,
      imageUrl: `/api/test64/images/${encodeURIComponent(filename)}`,
      expectedDay,
      expectedMonth,
      expectedYear,
      updatedAt,
      dayInput: toInput(expectedDay),
      monthInput: toInput(expectedMonth),
      yearInput: toInput(expectedYear),
      saving: false,
      error: '',
      saveState: 'idle',
      saveMessage: '',
      imageExpanded: !isLabeled
    };
  }

  function isPersistedLabeled(row: LabelRow): boolean {
    return row.expectedMonth !== null && row.expectedYear !== null;
  }

  function toggleImagePreview(index: number): void {
    const row = rows[index];
    if (!row) return;
    if (!isPersistedLabeled(row)) return;
    row.imageExpanded = !row.imageExpanded;
    rows = rows.slice();
  }

  function formatUpdatedAt(value: string | null): string {
    if (!value) return '-';
    const parsed = Date.parse(value);
    if (!Number.isFinite(parsed)) return value;
    return new Date(parsed).toLocaleString(undefined, {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  function setNotice(kind: NoticeKind, text: string): void {
    notice = { kind, text };
  }

  async function handleResponse(res: Response): Promise<JsonRecord> {
    const data = (await res.json().catch(() => ({}))) as JsonRecord;
    if (!res.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`;
      throw new Error(detail);
    }
    return data;
  }

  async function loadLabels(showNotice = false): Promise<void> {
    loading = true;
    if (showNotice) notice = null;
    try {
      const payload = await handleResponse(await fetch('/api/test64/labels'));
      const items = Array.isArray(payload.items) ? payload.items.filter(isJsonRecord) : [];
      rows = items.map(mapRow).filter((item) => item.filename.length > 0);
      totalCandidates = typeof payload.total_candidates === 'number' ? payload.total_candidates : rows.length;
      labeledCount = typeof payload.labeled_count === 'number' ? payload.labeled_count : rows.filter((item) => item.expectedMonth !== null && item.expectedYear !== null).length;
      if (showNotice) setNotice('success', 'test64 labels refreshed.');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      setNotice('error', message);
    } finally {
      loading = false;
    }
  }

  async function saveRow(index: number): Promise<void> {
    const row = rows[index];
    if (!row) return;
    row.error = '';
    notice = null;

    const day = toIntOrNull(row.dayInput);
    const month = toIntOrNull(row.monthInput);
    const year = toIntOrNull(row.yearInput);

    if (month === null || month < 1 || month > 12) {
      row.error = 'Month is required (1-12).';
      row.saveState = 'error';
      row.saveMessage = 'Validation failed.';
      rows = rows.slice();
      return;
    }
    if (year === null || year < 2000 || year > 2200) {
      row.error = 'Year is required (2000-2200).';
      row.saveState = 'error';
      row.saveMessage = 'Validation failed.';
      rows = rows.slice();
      return;
    }
    if (day !== null && (day < 1 || day > 31)) {
      row.error = 'Day must be 1-31 when provided.';
      row.saveState = 'error';
      row.saveMessage = 'Validation failed.';
      rows = rows.slice();
      return;
    }

    row.saving = true;
    row.saveState = 'idle';
    row.saveMessage = '';
    rows = rows.slice();
    try {
      const payload = await handleResponse(
        await fetch('/api/test64/labels', {
          method: 'PUT',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            filename: row.filename,
            day,
            month,
            year
          })
        })
      );
      const item = isJsonRecord(payload.item) ? mapRow(payload.item) : null;
      if (item) {
        rows[index] = {
          ...item,
          updatedAt: item.updatedAt ?? new Date().toISOString(),
          imageExpanded: false,
          saving: false,
          error: '',
          saveState: 'saved',
          saveMessage: 'Saved ✅'
        };
      } else {
        rows[index].expectedDay = day;
        rows[index].expectedMonth = month;
        rows[index].expectedYear = year;
        rows[index].saving = false;
        rows[index].updatedAt = new Date().toISOString();
        rows[index].imageExpanded = false;
        rows[index].saveState = 'saved';
        rows[index].saveMessage = 'Saved ✅';
      }
      labeledCount = rows.filter((entry) => toIntOrNull(entry.monthInput) !== null && toIntOrNull(entry.yearInput) !== null).length;
      rows = rows.slice();
      setNotice('success', `Saved label for ${row.filename}.`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      rows[index].saving = false;
      rows[index].error = message;
      rows[index].saveState = 'error';
      rows[index].saveMessage = 'Save failed';
      rows = rows.slice();
    }
  }

  onMount(() => {
    void loadLabels();
  });
</script>

<section class="panel">
  <div class="panel-heading">
    <h2>test64 Benchmark Labels</h2>
    <p>Set expected expiry dates before rescanning. Day can be empty if only month/year exists.</p>
  </div>

  <div class="actions-row compact">
    <button type="button" class="secondary small" on:click={() => loadLabels(true)} disabled={loading}>
      {loading ? 'Loading...' : 'Refresh Labels'}
    </button>
    <p class="muted">Labeled {labeledCount}/{totalCandidates}</p>
  </div>

  {#if notice}
    <div class={`notice ${notice.kind}`}>
      <p>{notice.text}</p>
    </div>
  {/if}

  {#if loading && rows.length === 0}
    <p class="muted">Loading test64 images...</p>
  {:else if rows.length === 0}
    <p class="muted">No files found in test64.</p>
  {:else}
    <div class="test64-label-list">
      {#each rows as row, index (row.filename)}
        <article class="test64-label-card">
          {#if !isPersistedLabeled(row)}
            <div class="test64-image-wrap">
              <img src={row.imageUrl} alt={`test64 ${row.filename}`} loading="lazy" />
            </div>
          {:else if row.imageExpanded}
            <button
              type="button"
              class="test64-image-wrap test64-image-toggle"
              on:click={() => toggleImagePreview(index)}
              aria-label={`Hide preview for ${row.filename}`}
            >
              <img src={row.imageUrl} alt={`test64 ${row.filename}`} loading="lazy" />
            </button>
          {:else}
            <button
              type="button"
              class="test64-image-placeholder"
              on:click={() => toggleImagePreview(index)}
              aria-label={`Show preview for ${row.filename}`}
            >
              <span>Labeled item</span>
              <strong>Click to show image</strong>
            </button>
          {/if}
          <div class="test64-label-main">
            <strong class="test64-filename">{row.filename}</strong>
            <p class="muted">Last updated: {formatUpdatedAt(row.updatedAt)}</p>
            <div class="test64-label-grid">
              <label>
                Day (optional)
                <input type="number" min="1" max="31" bind:value={row.dayInput} placeholder="-" />
              </label>
              <label>
                Month
                <input type="number" min="1" max="12" bind:value={row.monthInput} placeholder="MM" />
              </label>
              <label>
                Year
                <input type="number" min="2000" max="2200" bind:value={row.yearInput} placeholder="YYYY" />
              </label>
            </div>
            {#if row.error}
              <p class="error-inline">{row.error}</p>
            {/if}
            <div class="test64-label-footer">
              <div class="test64-save-status" aria-live="polite">
                {#if row.saving}
                  <span class="muted">Saving...</span>
                {:else if row.saveState === 'saved'}
                  <span class="saved-indicator">{row.saveMessage}</span>
                {:else if row.saveState === 'error'}
                  <span class="error-indicator">{row.saveMessage}</span>
                {/if}
              </div>
              <div class="test64-label-actions">
                <button type="button" class="small" on:click={() => saveRow(index)} disabled={row.saving}>
                  {row.saving ? 'Saving...' : 'Save Label'}
                </button>
              </div>
            </div>
          </div>
        </article>
      {/each}
    </div>
  {/if}
</section>
