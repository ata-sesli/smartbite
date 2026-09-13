<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type Bbox = [number, number, number, number];
  type Polygon = Array<[number, number]>;

  type DetectorBox = {
    bboxXyxy: Bbox;
    polygonXy: Polygon | null;
    confidence: number | null;
    detectorName: string;
  };

  type DetectorResult = {
    configName: string;
    verdict: string;
    boxCount: number | null;
    truthCoverageRatio: number | null;
    bestIou: number | null;
    runtimeMs: number | null;
    detectorBoxes: DetectorBox[];
  };

  type ComparisonItem = {
    filename: string;
    imageUrl: string;
    configs: DetectorResult[];
  };

  type Notice = { kind: 'success' | 'error' | 'info'; text: string } | null;

  let loading = false;
  let notice: Notice = null;
  let runId = '';
  let reportPath = '';
  let candidateCap = 0;
  let configs: JsonRecord[] = [];
  let summary: JsonRecord = {};
  let items: ComparisonItem[] = [];
  let activeIndex = 0;
  let imageNaturalWidth = 0;
  let imageNaturalHeight = 0;

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

  function apiImageUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value.replace('/test64/images/', '/api/test64/images/');
  }

  function numberPair(value: unknown): [number, number] | null {
    if (!Array.isArray(value) || value.length < 2) return null;
    const [x, y] = value;
    if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    return [x, y];
  }

  function xyxy(value: unknown): Bbox | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    if (!value.every((item) => typeof item === 'number' && Number.isFinite(item))) return null;
    return [value[0], value[1], value[2], value[3]];
  }

  function mapDetectorBox(raw: unknown, fallbackDetectorName: string): DetectorBox | null {
    if (!isRecord(raw)) return null;
    const bbox = xyxy(raw.bbox_xyxy);
    if (!bbox) return null;
    const polygon = Array.isArray(raw.polygon_xy)
      ? raw.polygon_xy.map(numberPair).filter((point): point is [number, number] => point !== null)
      : [];
    return {
      bboxXyxy: bbox,
      polygonXy: polygon.length >= 3 ? polygon : null,
      confidence: numberOrNull(raw.confidence),
      detectorName: text(raw.detector_name) || fallbackDetectorName
    };
  }

  function mapResult(raw: unknown): DetectorResult | null {
    if (!isRecord(raw)) return null;
    const configName = text(raw.config_name);
    return {
      configName,
      verdict: text(raw.verdict),
      boxCount: numberOrNull(raw.box_count),
      truthCoverageRatio: numberOrNull(raw.truth_coverage_ratio),
      bestIou: numberOrNull(raw.best_iou),
      runtimeMs: numberOrNull(raw.runtime_ms),
      detectorBoxes: Array.isArray(raw.detector_boxes)
        ? raw.detector_boxes.map((box) => mapDetectorBox(box, configName)).filter((box): box is DetectorBox => box !== null)
        : []
    };
  }

  function mapItem(raw: unknown): ComparisonItem | null {
    if (!isRecord(raw)) return null;
    return {
      filename: text(raw.filename),
      imageUrl: apiImageUrl(raw.image_url),
      configs: Array.isArray(raw.configs)
        ? raw.configs.map(mapResult).filter((item): item is DetectorResult => item !== null)
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

  async function loadComparison(): Promise<void> {
    loading = true;
    notice = null;
    try {
      const payload = await handleResponse(await fetch('/api/test64/detector-comparison'));
      runId = text(payload.run_id);
      reportPath = text(payload.report_path);
      candidateCap = typeof payload.candidate_cap === 'number' ? payload.candidate_cap : 0;
      configs = Array.isArray(payload.configs) ? payload.configs.filter(isRecord) : [];
      summary = isRecord(payload.summary) ? payload.summary : {};
      items = Array.isArray(payload.items)
        ? payload.items.map(mapItem).filter((item): item is ComparisonItem => item !== null)
        : [];
      activeIndex = Math.min(activeIndex, Math.max(0, items.length - 1));
      notice = { kind: 'success', text: `Loaded ${items.length} detector comparison images.` };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      notice = { kind: 'error', text: message };
    } finally {
      loading = false;
    }
  }

  function selectIndex(index: number): void {
    activeIndex = Math.max(0, Math.min(index, items.length - 1));
    imageNaturalWidth = 0;
    imageNaturalHeight = 0;
  }

  function summaryFor(configName: string): JsonRecord | null {
    const root = isRecord(summary.configs) ? summary.configs : {};
    const value = root[configName];
    return isRecord(value) ? value : null;
  }

  function configColor(configName: string): string {
    const config = configs.find((item) => text(item.name) === configName);
    return text(config?.color) || '#f59e0b';
  }

  function detectorLabel(configName: string): string {
    if (configName === 'yolo26s_obb_best') return 'YOLO';
    if (configName === 'ppocrv5_mobile_pretrained') return 'PP mobile';
    return configName;
  }

  function formatCount(value: unknown): string {
    return typeof value === 'number' ? String(value) : '-';
  }

  function formatRuntime(value: number | null | unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)} ms` : '-';
  }

  function percent(value: number | null | unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '-';
  }

  function decimal(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '-';
  }

  function verdictLabel(value: string): string {
    return value ? value.replaceAll('_', ' ') : 'unknown';
  }

  function rectFromXyxy([x1, y1, x2, y2]: Bbox): { x: number; y: number; width: number; height: number } {
    return {
      x: Math.min(x1, x2),
      y: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1)
    };
  }

  function polygonPoints(polygon: Polygon): string {
    return polygon.map(([x, y]) => `${x},${y}`).join(' ');
  }

  function handleImageLoad(event: Event): void {
    const image = event.currentTarget as HTMLImageElement;
    imageNaturalWidth = image.naturalWidth;
    imageNaturalHeight = image.naturalHeight;
  }

  onMount(() => {
    void loadComparison();
  });
</script>

<section class="panel recognition-review-panel detector-comparison-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>Detector Comparison</h2>
      <p>YOLO OBB and pretrained PP-OCRv5 mobile detection on test64.</p>
    </div>
    <div class="recognition-review-actions">
      <button type="button" on:click={() => loadComparison()} disabled={loading}>
        {loading ? 'Loading...' : 'Refresh'}
      </button>
    </div>
  </div>

  {#if notice}
    <div class={`notice ${notice.kind}`}>
      <p>{notice.text}</p>
    </div>
  {/if}

  <div class="recognition-review-stats detector-comparison-summary">
    {#each configs as config}
      {@const configName = text(config.name)}
      {@const configSummary = summaryFor(configName)}
      <div class="status-item detector-summary-card" style={`border-color: ${configColor(configName)}`}>
        <span>{detectorLabel(configName)}</span>
        <strong>{formatCount(configSummary?.covered)}/{formatCount(configSummary?.images)} covered</strong>
        <small>
          avg {formatRuntime(configSummary?.average_runtime_ms)}
          · boxes {decimal(configSummary?.average_box_count)}
          · no box {formatCount(configSummary?.no_box)}
        </small>
      </div>
    {/each}
  </div>

  <p class="muted">Candidate cap: {candidateCap || '-'} · Report: {reportPath || runId || '-'}</p>

  <div class="detector-legend" aria-label="Detector colors">
    {#each configs as config}
      {@const configName = text(config.name)}
      <span><i style={`background: ${configColor(configName)}`}></i>{detectorLabel(configName)}</span>
    {/each}
  </div>

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Detector comparison images">
      {#if items.length === 0}
        <p class="muted">No detector comparison report is available.</p>
      {:else}
        {#each items as item, index}
          <button type="button" class:active={index === activeIndex} on:click={() => selectIndex(index)}>
            <strong>{index + 1}. {item.filename}</strong>
            <span>
              {#each item.configs as result, resultIndex}
                {resultIndex > 0 ? ' · ' : ''}{detectorLabel(result.configName)}:{verdictLabel(result.verdict)}
              {/each}
            </span>
            <small>
              {#each item.configs as result, resultIndex}
                {resultIndex > 0 ? ' · ' : ''}{detectorLabel(result.configName)} {formatCount(result.boxCount)}
              {/each}
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
            <p class="muted">All detector boxes are overlaid together. Manual truth boxes are intentionally hidden.</p>
          </div>
        </div>

        <div class="detection-review-grid">
          <figure class="detection-review-image detector-comparison-image">
            {#if active.imageUrl}
              <div class="detection-review-image-frame">
                <img src={active.imageUrl} alt={`Detector comparison source ${active.filename}`} on:load={handleImageLoad} />
                {#if imageNaturalWidth > 0 && imageNaturalHeight > 0}
                  <svg
                    class="detection-review-overlay detector-comparison-overlay"
                    viewBox={`0 0 ${imageNaturalWidth} ${imageNaturalHeight}`}
                    preserveAspectRatio="xMidYMid meet"
                    aria-hidden="true"
                  >
                    {#each active.configs as result}
                      {@const color = configColor(result.configName)}
                      {#each result.detectorBoxes as box}
                        {#if box.polygonXy}
                          <polygon
                            points={polygonPoints(box.polygonXy)}
                            style={`stroke: ${color}; fill: transparent;`}
                          />
                        {:else}
                          {@const rect = rectFromXyxy(box.bboxXyxy)}
                          <rect
                            x={rect.x}
                            y={rect.y}
                            width={rect.width}
                            height={rect.height}
                            style={`stroke: ${color}; fill: transparent;`}
                          />
                        {/if}
                      {/each}
                    {/each}
                  </svg>
                {/if}
              </div>
            {:else}
              <div class="recognition-empty-crop">No image</div>
            {/if}
            <figcaption>
              {#each active.configs as result, index}
                {index > 0 ? ' · ' : ''}{detectorLabel(result.configName)} {formatCount(result.boxCount)}
              {/each}
            </figcaption>
          </figure>

          <div class="recognition-winner-card">
            <h3>Per-Detector Result</h3>
            <div class="detector-result-stack">
              {#each active.configs as result}
                <article style={`border-color: ${configColor(result.configName)}`}>
                  <strong>{detectorLabel(result.configName)}</strong>
                  <dl class="compact-dl">
                    <dt>Verdict</dt>
                    <dd>{verdictLabel(result.verdict)}</dd>
                    <dt>Boxes</dt>
                    <dd>{formatCount(result.boxCount)}</dd>
                    <dt>Truth Coverage</dt>
                    <dd>{percent(result.truthCoverageRatio)}</dd>
                    <dt>Best IoU</dt>
                    <dd>{percent(result.bestIou)}</dd>
                    <dt>Runtime</dt>
                    <dd>{formatRuntime(result.runtimeMs)}</dd>
                  </dl>
                </article>
              {/each}
            </div>
          </div>
        </div>
      {:else}
        <p class="muted">Select an image to inspect detector overlays.</p>
      {/if}
    </section>
  </div>
</section>
