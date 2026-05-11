<script lang="ts">
  import { onMount } from 'svelte';
  import ImageAnnotator from './ImageAnnotator.svelte';

  type JsonRecord = Record<string, unknown>;
  type NoticeKind = 'success' | 'error' | 'info';

  type TruthItem = {
    filename: string;
    imageUrl: string;
    expectedDay: number | null;
    expectedMonth: number | null;
    expectedYear: number | null;
    imageWidth: number;
    imageHeight: number;
    bbox: number[] | null;
    draftBbox: number[] | null;
    polygon: number[][] | null;
    draftPolygon: number[][] | null;
    rotationDegrees: number;
    draftRotationDegrees: number;
    updatedAt: string | null;
    saving: boolean;
    saveState: 'idle' | 'saved' | 'error';
    saveMessage: string;
    error: string;
  };

  type Notice = {
    kind: NoticeKind;
    text: string;
  } | null;

  let items: TruthItem[] = [];
  let activeIndex = 0;
  let loading = false;
  let notice: Notice = null;
  let annotatedCount = 0;
  let totalCount = 0;
  let manifestPath = '';
  let active: TruthItem | null = null;

  $: active = items[activeIndex] ?? null;

  function isJsonRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function toNumberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function bboxFrom(value: unknown): number[] | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    const parsed = value.map((item) => (typeof item === 'number' ? item : Number(item)));
    return parsed.every((item) => Number.isFinite(item)) ? parsed : null;
  }

  function polygonFrom(value: unknown): number[][] | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    const parsed = value.map((point) => {
      if (!Array.isArray(point) || point.length !== 2) return null;
      const x = typeof point[0] === 'number' ? point[0] : Number(point[0]);
      const y = typeof point[1] === 'number' ? point[1] : Number(point[1]);
      return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
    });
    return parsed.every((point): point is number[] => point !== null) ? parsed : null;
  }

  function mapItem(raw: JsonRecord): TruthItem {
    const bbox = bboxFrom(raw.true_bbox_xyxy);
    const polygon = polygonFrom(raw.true_polygon_xy);
    const rotationDegrees = toNumberOrNull(raw.annotation_rotation_degrees) ?? 0;
    return {
      filename: typeof raw.filename === 'string' ? raw.filename : '',
      imageUrl: typeof raw.image_url === 'string' ? raw.image_url : '',
      expectedDay: toNumberOrNull(raw.expected_day),
      expectedMonth: toNumberOrNull(raw.expected_month),
      expectedYear: toNumberOrNull(raw.expected_year),
      imageWidth: toNumberOrNull(raw.image_width) ?? 0,
      imageHeight: toNumberOrNull(raw.image_height) ?? 0,
      bbox,
      draftBbox: bbox ? bbox.slice() : null,
      polygon,
      draftPolygon: polygon ? polygon.map((point) => point.slice()) : null,
      rotationDegrees,
      draftRotationDegrees: rotationDegrees,
      updatedAt: typeof raw.updated_at === 'string' ? raw.updated_at : null,
      saving: false,
      saveState: 'idle',
      saveMessage: '',
      error: ''
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

  function setNotice(kind: NoticeKind, text: string): void {
    notice = { kind, text };
  }

  async function loadTruthBboxes(showNotice = false): Promise<void> {
    loading = true;
    if (showNotice) notice = null;
    try {
      const payload = await handleResponse(await fetch('/api/test64/truth-bboxes'));
      const auditSet = Array.isArray(payload.audit_set) ? payload.audit_set.filter((item): item is string => typeof item === 'string') : [];
      const rawItems = isJsonRecord(payload.items) ? payload.items : {};
      items = auditSet
        .map((filename) => (isJsonRecord(rawItems[filename]) ? mapItem(rawItems[filename] as JsonRecord) : null))
        .filter((item): item is TruthItem => item !== null && item.filename.length > 0);
      annotatedCount = typeof payload.annotated_count === 'number' ? payload.annotated_count : items.filter((item) => item.bbox !== null).length;
      totalCount = typeof payload.total_count === 'number' ? payload.total_count : items.length;
      manifestPath = typeof payload.manifest_path === 'string' ? payload.manifest_path : '';
      activeIndex = Math.min(activeIndex, Math.max(0, items.length - 1));
      if (showNotice) setNotice('success', 'Crop truth annotations refreshed.');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      setNotice('error', message);
    } finally {
      loading = false;
    }
  }

  function activeItem(): TruthItem | null {
    return active;
  }

  function expectedDate(item: TruthItem): string {
    const month = item.expectedMonth === null ? 'MM' : String(item.expectedMonth).padStart(2, '0');
    const year = item.expectedYear === null ? 'YYYY' : String(item.expectedYear);
    if (item.expectedDay === null) return `${month}/${year}`;
    return `${String(item.expectedDay).padStart(2, '0')}/${month}/${year}`;
  }

  function formatUpdatedAt(value: string | null): string {
    if (!value) return '-';
    const parsed = Date.parse(value);
    if (!Number.isFinite(parsed)) return value;
    return new Date(parsed).toLocaleString();
  }

  function formatBbox(value: number[] | null): string {
    return value ? value.map((item) => Math.round(item)).join(', ') : 'not set';
  }

  function selectIndex(index: number): void {
    activeIndex = Math.max(0, Math.min(index, items.length - 1));
  }

  function nextUnlabeled(): void {
    if (items.length === 0) return;
    for (let offset = 1; offset <= items.length; offset += 1) {
      const index = (activeIndex + offset) % items.length;
      if (!items[index].bbox) {
        activeIndex = index;
        return;
      }
    }
    setNotice('info', 'All test64 images have saved crop truth boxes.');
  }

  async function saveActive(): Promise<void> {
    const item = activeItem();
    if (!item) return;
    item.saving = true;
    item.error = '';
    item.saveState = 'idle';
    item.saveMessage = '';
    items = items.slice();
    try {
      const payload = await handleResponse(
        await fetch(`/api/test64/truth-bboxes/${encodeURIComponent(item.filename)}`, {
          method: 'PUT',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            true_bbox_xyxy: item.draftBbox,
            true_polygon_xy: item.draftBbox ? item.draftPolygon : null,
            annotation_rotation_degrees: item.draftBbox ? item.draftRotationDegrees : null
          })
        })
      );
      const saved = isJsonRecord(payload.item) ? mapItem(payload.item) : null;
      if (saved) {
        saved.saveState = 'saved';
        saved.saveMessage = 'Saved';
        items[activeIndex] = saved;
      } else {
        item.bbox = item.draftBbox ? item.draftBbox.slice() : null;
        item.polygon = item.draftPolygon ? item.draftPolygon.map((point) => point.slice()) : null;
        item.rotationDegrees = item.draftRotationDegrees;
        item.saving = false;
        item.saveState = 'saved';
        item.saveMessage = 'Saved';
      }
      annotatedCount = typeof payload.annotated_count === 'number' ? payload.annotated_count : items.filter((entry) => entry.bbox !== null).length;
      totalCount = typeof payload.total_count === 'number' ? payload.total_count : items.length;
      manifestPath = typeof payload.manifest_path === 'string' ? payload.manifest_path : manifestPath;
      items = items.slice();
      setNotice('success', `Saved crop truth bbox for ${item.filename}.`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      item.saving = false;
      item.error = message;
      item.saveState = 'error';
      item.saveMessage = 'Save failed';
      items = items.slice();
    }
  }

  function clearActive(): void {
    const item = activeItem();
    if (!item) return;
    item.draftBbox = null;
    item.draftPolygon = null;
    item.error = '';
    item.saveState = 'idle';
    item.saveMessage = '';
    items = items.slice();
  }

  function updateDraftBbox(event: CustomEvent<{ bbox: number[] | null; polygon?: number[][] | null; rotationDegrees?: number }>): void {
    const item = activeItem();
    if (!item) return;
    const bbox = event.detail.bbox;
    const polygon = event.detail.polygon ?? null;
    item.draftBbox = bbox ? bbox.slice() : null;
    item.draftPolygon = polygon ? polygon.map((point) => point.slice()) : null;
    item.draftRotationDegrees =
      typeof event.detail.rotationDegrees === 'number' && Number.isFinite(event.detail.rotationDegrees)
        ? event.detail.rotationDegrees
        : item.draftRotationDegrees;
    item.error = '';
    item.saveState = 'idle';
    item.saveMessage = '';
    items = items.slice();
  }

  onMount(() => {
    void loadTruthBboxes();
  });
</script>

<section class="panel crop-truth-panel">
  <div class="panel-heading">
    <h2>Crop Truth Annotation</h2>
    <p>Draw the tight expiry-date box for every test64 image. Existing saved boxes are preserved.</p>
  </div>

  <div class="actions-row compact">
    <button type="button" class="secondary small" on:click={() => loadTruthBboxes(true)} disabled={loading}>
      {loading ? 'Loading...' : 'Refresh'}
    </button>
    <button type="button" class="secondary small" on:click={nextUnlabeled} disabled={items.length === 0 || annotatedCount >= totalCount}>
      Next Unlabeled
    </button>
    <p class="muted">Annotated {annotatedCount}/{totalCount}</p>
  </div>

  {#if notice}
    <div class={`notice ${notice.kind}`}>
      <p>{notice.text}</p>
    </div>
  {/if}

  {#if loading && items.length === 0}
    <p class="muted">Loading crop truth set...</p>
  {:else if items.length === 0}
    <p class="muted">No crop truth images available.</p>
  {:else if active}
    <div class="crop-truth-layout">
      <aside class="crop-truth-list" aria-label="Crop truth test64 images">
        {#each items as row, index (row.filename)}
          <button type="button" class:active={index === activeIndex} on:click={() => selectIndex(index)}>
            <span>{index + 1}. {row.filename}</span>
            <strong class:boxed={row.bbox}>{row.bbox ? 'boxed' : 'open'}</strong>
          </button>
        {/each}
      </aside>

      <div class="crop-truth-workspace">
        <div class="crop-truth-meta">
          <div class="crop-truth-meta-summary">
            <strong>{active.filename}</strong>
            <p class="muted">Expected: {expectedDate(active)} · Image: {active.imageWidth} × {active.imageHeight}</p>
            <p class="muted crop-truth-live-line">Saved bbox: {formatBbox(active.bbox)} · Draft bbox: {formatBbox(active.draftBbox)}</p>
            <p class="muted crop-truth-live-line">Draft rotation: {active.draftRotationDegrees.toFixed(1)}° · Exact polygon: {active.draftPolygon ? 'set' : 'not set'}</p>
            <p class="muted">Last updated: {formatUpdatedAt(active.updatedAt)}</p>
          </div>
          <div class="crop-truth-actions">
            <button type="button" class="secondary small" on:click={() => selectIndex(activeIndex - 1)} disabled={activeIndex === 0}>
              Previous
            </button>
            <button type="button" class="secondary small" on:click={() => selectIndex(activeIndex + 1)} disabled={activeIndex >= items.length - 1}>
              Next
            </button>
            <button type="button" class="secondary small" on:click={nextUnlabeled} disabled={annotatedCount >= totalCount}>
              Next Unlabeled
            </button>
            <button type="button" class="ghost small" on:click={clearActive} disabled={!active.draftBbox || active.saving}>
              Clear
            </button>
            <button type="button" class="small" on:click={saveActive} disabled={active.saving}>
              {active.saving ? 'Saving...' : 'Save BBox'}
            </button>
          </div>
        </div>

        <ImageAnnotator
          imageUrl={active.imageUrl}
          bbox={active.draftBbox}
          polygon={active.draftPolygon}
          rotationDegrees={active.draftRotationDegrees}
          on:bboxChange={updateDraftBbox}
        />

        {#if active.error}
          <p class="error-inline">{active.error}</p>
        {:else if active.saveState === 'saved'}
          <p class="saved-indicator">{active.saveMessage}</p>
        {:else if active.saveState === 'error'}
          <p class="error-indicator">{active.saveMessage}</p>
        {/if}

        {#if manifestPath}
          <p class="muted">Manifest: {manifestPath}</p>
        {/if}
      </div>
    </div>
  {/if}
</section>
