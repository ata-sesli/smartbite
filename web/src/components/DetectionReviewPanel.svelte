<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type NoticeKind = 'success' | 'error' | 'info';

  type DetectorBox = {
    bboxXyxy: [number, number, number, number];
    polygonXy: [number, number][] | null;
    confidence: number | null;
    source: string;
    sources: string[];
    variantName: string;
  };

  type ProductRoi = {
    bboxXyxy: [number, number, number, number];
    confidence: number | null;
    source: string;
  };

  type DetectorAuditConfigResult = {
    configName: string;
    verdict: string;
    boxCount: number | null;
    productRoiCount: number | null;
    bestIou: number | null;
    truthCoverageRatio: number | null;
    runtimeMs: number | null;
    bestBox: DetectorBox | null;
    detectorBoxes: DetectorBox[];
    productRois: ProductRoi[];
  };

  type DetectorAuditItem = {
    filename: string;
    imageUrl: string;
    truthBboxXyxy: [number, number, number, number] | null;
    truthPolygonXy: [number, number][] | null;
    configs: DetectorAuditConfigResult[];
  };

  type Notice = {
    kind: NoticeKind;
    text: string;
  } | null;

  export let endpoint = '/api/test64/detection-review';
  export let title = 'Detection Review';
  export let description = 'Detector-only truth overlap. No recognition, parsing, or full-pipeline verdicts.';
  export let emptyText = 'No detector audit items available.';
  export let configSwitchDescription = 'Switch configs without mixing in recognition results.';

  let loading = false;
  let notice: Notice = null;
  let runId = '';
  let source = '';
  let summary: JsonRecord = {};
  let configs: JsonRecord[] = [];
  let items: DetectorAuditItem[] = [];
  let activeIndex = 0;
  let activeConfigIndex = 0;
  let imageNaturalWidth = 0;
  let imageNaturalHeight = 0;

  $: active = items[activeIndex] ?? null;
  $: activeConfigs = active?.configs ?? [];
  $: if (activeConfigIndex >= activeConfigs.length) activeConfigIndex = 0;
  $: selectedConfig = activeConfigs[activeConfigIndex] ?? null;
  $: selectedSummary = selectedConfig ? configSummary(selectedConfig.configName) : null;

  function isRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function numberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function text(value: unknown): string {
    return typeof value === 'string' ? value : '';
  }

  function stringList(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
  }

  function apiImageUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value.replace('/test64/images/', '/api/test64/images/');
  }

  function numberPair(value: unknown): [number, number] | null {
    if (!Array.isArray(value) || value.length !== 2) return null;
    const [x, y] = value;
    if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    return [x, y];
  }

  function xyxy(value: unknown): [number, number, number, number] | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    if (!value.every((item) => typeof item === 'number' && Number.isFinite(item))) return null;
    return [value[0], value[1], value[2], value[3]];
  }

  function mapDetectorBox(raw: unknown): DetectorBox | null {
    if (!isRecord(raw) || !Array.isArray(raw.bbox_xyxy) || raw.bbox_xyxy.length !== 4) return null;
    const bbox = raw.bbox_xyxy;
    if (!bbox.every((value) => typeof value === 'number' && Number.isFinite(value))) return null;
    const polygon = Array.isArray(raw.polygon_xy)
      ? raw.polygon_xy.map(numberPair).filter((point): point is [number, number] => point !== null)
      : [];
    return {
      bboxXyxy: [bbox[0], bbox[1], bbox[2], bbox[3]],
      polygonXy: polygon.length >= 3 ? polygon : null,
      confidence: numberOrNull(raw.confidence),
      source: text(raw.source),
      sources: stringList(raw.sources),
      variantName: text(raw.variant_name)
    };
  }

  function mapProductRoi(raw: unknown): ProductRoi | null {
    if (!isRecord(raw) || !Array.isArray(raw.bbox_xyxy) || raw.bbox_xyxy.length !== 4) return null;
    const bbox = raw.bbox_xyxy;
    if (!bbox.every((value) => typeof value === 'number' && Number.isFinite(value))) return null;
    return {
      bboxXyxy: [bbox[0], bbox[1], bbox[2], bbox[3]],
      confidence: numberOrNull(raw.confidence),
      source: text(raw.source)
    };
  }

  function mapAuditConfig(raw: unknown): DetectorAuditConfigResult | null {
    if (!isRecord(raw)) return null;
    return {
      configName: text(raw.config_name),
      verdict: text(raw.verdict),
      boxCount: numberOrNull(raw.box_count),
      productRoiCount: numberOrNull(raw.product_roi_count),
      bestIou: numberOrNull(raw.best_iou),
      truthCoverageRatio: numberOrNull(raw.truth_coverage_ratio),
      runtimeMs: numberOrNull(raw.runtime_ms),
      bestBox: mapDetectorBox(raw.best_box),
      detectorBoxes: Array.isArray(raw.detector_boxes)
        ? raw.detector_boxes.map(mapDetectorBox).filter((box): box is DetectorBox => box !== null)
        : [],
      productRois: Array.isArray(raw.product_rois)
        ? raw.product_rois.map(mapProductRoi).filter((roi): roi is ProductRoi => roi !== null)
        : []
    };
  }

  function mapAuditItem(raw: unknown): DetectorAuditItem | null {
    if (!isRecord(raw)) return null;
    const truthPolygon = Array.isArray(raw.truth_polygon_xy)
      ? raw.truth_polygon_xy.map(numberPair).filter((point): point is [number, number] => point !== null)
      : [];
    return {
      filename: text(raw.filename),
      imageUrl: apiImageUrl(raw.image_url),
      truthBboxXyxy: xyxy(raw.truth_bbox_xyxy),
      truthPolygonXy: truthPolygon.length >= 3 ? truthPolygon : null,
      configs: Array.isArray(raw.configs)
        ? raw.configs.map(mapAuditConfig).filter((item): item is DetectorAuditConfigResult => item !== null)
        : []
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
    try {
      const payload = await handleResponse(await fetch(endpoint));
      runId = text(payload.run_id);
      source = text(payload.source);
      summary = isRecord(payload.summary) ? payload.summary : {};
      configs = Array.isArray(payload.configs) ? payload.configs.filter(isRecord) : [];
      items = Array.isArray(payload.items)
        ? payload.items.map(mapAuditItem).filter((item): item is DetectorAuditItem => item !== null)
        : [];
      activeIndex = Math.min(activeIndex, Math.max(0, items.length - 1));
      notice = { kind: 'success', text: `Loaded ${items.length} ${title.toLowerCase()} items.` };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      notice = { kind: 'error', text: message };
    } finally {
      loading = false;
    }
  }

  function selectIndex(index: number): void {
    activeIndex = Math.max(0, Math.min(index, items.length - 1));
    activeConfigIndex = 0;
    imageNaturalWidth = 0;
    imageNaturalHeight = 0;
  }

  function configSummary(configName: string): JsonRecord | null {
    const root = isRecord(summary.configs) ? summary.configs : {};
    const value = root[configName];
    return isRecord(value) ? value : null;
  }

  function formatCount(value: unknown): string {
    return typeof value === 'number' ? String(value) : '-';
  }

  function formatRuntime(value: number | null): string {
    return value === null ? '-' : `${Math.round(value)} ms`;
  }

  function formatDecimal(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '-';
  }

  function percent(value: number | null): string {
    return value === null ? '-' : `${(value * 100).toFixed(1)}%`;
  }

  function verdictLabel(value: string): string {
    return value ? value.replaceAll('_', ' ') : 'unknown';
  }

  function handleImageLoad(event: Event): void {
    const image = event.currentTarget as HTMLImageElement;
    imageNaturalWidth = image.naturalWidth;
    imageNaturalHeight = image.naturalHeight;
  }

  function rectFromXyxy([x1, y1, x2, y2]: [number, number, number, number]): {
    x: number;
    y: number;
    width: number;
    height: number;
  } {
    return {
      x: Math.min(x1, x2),
      y: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1)
    };
  }

  function rectAttrs(box: DetectorBox): { x: number; y: number; width: number; height: number } {
    return rectFromXyxy(box.bboxXyxy);
  }

  function roiRectAttrs(roi: ProductRoi): { x: number; y: number; width: number; height: number } {
    return rectFromXyxy(roi.bboxXyxy);
  }

  function polygonPoints(box: DetectorBox): string {
    return (box.polygonXy ?? []).map(([x, y]) => `${x},${y}`).join(' ');
  }

  function pointsString(points: [number, number][]): string {
    return points.map(([x, y]) => `${x},${y}`).join(' ');
  }

  onMount(() => {
    void loadReview();
  });
</script>

<section class="panel recognition-review-panel detection-review-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
    <div class="recognition-review-actions">
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
      <span>Audit Run</span>
      <strong>{runId || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Configs</span>
      <strong>{configs.length}</strong>
    </div>
    <div class="status-item">
      <span>Images</span>
      <strong>{items.length}</strong>
    </div>
    {#if selectedSummary}
      <div class="status-item">
        <span>Covered</span>
        <strong>{formatCount(selectedSummary.covered)}/{formatCount(selectedSummary.images)}</strong>
      </div>
      <div class="status-item">
        <span>No Box</span>
        <strong>{formatCount(selectedSummary.no_box)}</strong>
      </div>
      <div class="status-item">
        <span>Mean Truth</span>
        <strong>{formatDecimal(Number(selectedSummary.mean_truth_coverage_ratio ?? 0) * 100)}%</strong>
      </div>
      <div class="status-item">
        <span>Avg Boxes</span>
        <strong>{formatDecimal(selectedSummary.average_box_count)}</strong>
      </div>
      {#if selectedConfig?.productRoiCount !== null && selectedConfig?.productRoiCount !== undefined}
        <div class="status-item">
          <span>Product ROIs</span>
          <strong>{formatCount(selectedConfig.productRoiCount)}</strong>
        </div>
      {/if}
      <div class="status-item">
        <span>Avg Runtime</span>
        <strong>{formatDecimal(selectedSummary.average_runtime_ms)} ms</strong>
      </div>
    {/if}
  </div>

  {#if source}
    <p class="muted">Source: {source}</p>
  {/if}

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Detector audit items">
      {#if items.length === 0}
        <p class="muted">{emptyText}</p>
      {:else}
        {#each items as item, index}
          {@const config = item.configs[activeConfigIndex] ?? item.configs[0]}
          <button type="button" class:active={index === activeIndex} on:click={() => selectIndex(index)}>
            <strong>{index + 1}. {item.filename}</strong>
            <span>{verdictLabel(config?.verdict ?? '')}</span>
            <small>
              boxes {formatCount(config?.boxCount)} · truth {percent(config?.truthCoverageRatio ?? null)}
            </small>
          </button>
        {/each}
      {/if}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{active.filename}</h3>
            <p class="muted">Manual truth bbox vs selected detector config.</p>
          </div>
          <span class="outcome-pill">{selectedConfig?.verdict ? verdictLabel(selectedConfig.verdict) : '-'}</span>
        </div>

        <div class="detection-review-grid">
          <figure class="detection-review-image">
            {#if active.imageUrl}
              <div class="detection-review-image-frame">
                <img src={active.imageUrl} alt={`Detector audit source ${active.filename}`} on:load={handleImageLoad} />
                {#if imageNaturalWidth > 0 && imageNaturalHeight > 0 && (selectedConfig?.detectorBoxes.length || selectedConfig?.productRois.length || active.truthPolygonXy || active.truthBboxXyxy || selectedConfig?.bestBox)}
                  <svg
                    class="detection-review-overlay"
                    viewBox={`0 0 ${imageNaturalWidth} ${imageNaturalHeight}`}
                    preserveAspectRatio="xMidYMid meet"
                    aria-hidden="true"
                  >
                    {#each selectedConfig?.productRois ?? [] as roi}
                      {@const roiRect = roiRectAttrs(roi)}
                      <rect class="product-roi-box" x={roiRect.x} y={roiRect.y} width={roiRect.width} height={roiRect.height} />
                    {/each}
                    {#each selectedConfig?.detectorBoxes ?? [] as box}
                      {#if box.polygonXy}
                        <polygon points={polygonPoints(box)} />
                      {:else}
                        {@const rect = rectAttrs(box)}
                        <rect x={rect.x} y={rect.y} width={rect.width} height={rect.height} />
                      {/if}
                    {/each}
                    {#if active.truthPolygonXy}
                      <polygon class="truth-box" points={pointsString(active.truthPolygonXy)} />
                    {:else if active.truthBboxXyxy}
                      {@const truthRect = rectFromXyxy(active.truthBboxXyxy)}
                      <rect class="truth-box" x={truthRect.x} y={truthRect.y} width={truthRect.width} height={truthRect.height} />
                    {/if}
                    {#if selectedConfig?.bestBox}
                      {#if selectedConfig.bestBox.polygonXy}
                        <polygon class="best-audit-box" points={polygonPoints(selectedConfig.bestBox)} />
                      {:else}
                        {@const bestRect = rectAttrs(selectedConfig.bestBox)}
                        <rect class="best-audit-box" x={bestRect.x} y={bestRect.y} width={bestRect.width} height={bestRect.height} />
                      {/if}
                    {/if}
                  </svg>
                {/if}
              </div>
            {:else}
              <div class="recognition-empty-crop">No image</div>
            {/if}
            <figcaption>
              {formatCount(selectedConfig?.boxCount)} detector boxes drawn
              {#if selectedConfig?.productRoiCount !== null && selectedConfig?.productRoiCount !== undefined}
                · {formatCount(selectedConfig.productRoiCount)} product ROIs
              {/if}
            </figcaption>
          </figure>

          <div class="recognition-winner-card">
            <h3>Detector Audit</h3>
            <dl class="compact-dl">
              <dt>Config</dt>
              <dd>{selectedConfig?.configName || '-'}</dd>
              <dt>Verdict</dt>
              <dd>{selectedConfig?.verdict ? verdictLabel(selectedConfig.verdict) : '-'}</dd>
              <dt>Boxes</dt>
              <dd>{formatCount(selectedConfig?.boxCount)}</dd>
              {#if selectedConfig?.productRoiCount !== null && selectedConfig?.productRoiCount !== undefined}
                <dt>Product ROIs</dt>
                <dd>{formatCount(selectedConfig.productRoiCount)}</dd>
              {/if}
              <dt>Best IoU</dt>
              <dd>{formatDecimal(selectedConfig?.bestIou)}</dd>
              <dt>Truth Coverage</dt>
              <dd>{percent(selectedConfig?.truthCoverageRatio ?? null)}</dd>
              <dt>Runtime</dt>
              <dd>{formatRuntime(selectedConfig?.runtimeMs ?? null)}</dd>
            </dl>
          </div>
        </div>

        {#if activeConfigs.length > 0}
          <div class="detector-audit-panel">
            <div class="recognition-review-title-row">
              <div>
                <h3>Compare Detector Configs</h3>
                <p class="muted">{configSwitchDescription}</p>
              </div>
            </div>
            <div class="detector-audit-configs">
              {#each activeConfigs as config, index}
                <button type="button" class:active={index === activeConfigIndex} on:click={() => (activeConfigIndex = index)}>
                  <strong>{config.configName}</strong>
                  <span>{verdictLabel(config.verdict)}</span>
                  <small>
                    boxes {formatCount(config.boxCount)} · truth {percent(config.truthCoverageRatio)}
                    · {formatRuntime(config.runtimeMs)}
                  </small>
                </button>
              {/each}
            </div>
          </div>
        {/if}
      {:else}
        <p class="muted">Select a detector audit item to inspect.</p>
      {/if}
    </section>
  </div>
</section>
