<script lang="ts">
  import { onMount } from 'svelte';

  type JsonRecord = Record<string, unknown>;
  type Bbox = [number, number, number, number];
  type Notice = { kind: 'success' | 'error' | 'info'; text: string } | null;

  let loading = false;
  let notice: Notice = null;
  let runId = '';
  let createdAt = '';
  let stableReport = '';
  let summary: JsonRecord = {};
  let items: JsonRecord[] = [];
  let selectedFilename = '';
  let filter = 'all';
  let imageNaturalWidth = 0;
  let imageNaturalHeight = 0;
  let imageSizeFilename = '';

  $: classCounts = isRecord(summary.class_counts) ? summary.class_counts : {};
  $: filters = ['all', ...Object.keys(classCounts).sort()];
  $: filteredItems = items.filter((item) => {
    if (filter === 'all') return true;
    return text(item.class) === filter;
  });
  $: if (filteredItems.length > 0 && !filteredItems.some((item) => text(item.filename) === selectedFilename)) {
    selectedFilename = text(filteredItems[0].filename);
  }
  $: active = filteredItems.find((item) => text(item.filename) === selectedFilename) ?? filteredItems[0] ?? null;
  $: if (text(active?.filename) !== imageSizeFilename) {
    imageSizeFilename = text(active?.filename);
    imageNaturalWidth = 0;
    imageNaturalHeight = 0;
  }

  function isRecord(value: unknown): value is JsonRecord {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  function text(value: unknown): string {
    return typeof value === 'string' ? value : '';
  }

  function numberOrNull(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  function list(value: unknown): JsonRecord[] {
    return Array.isArray(value) ? value.filter(isRecord) : [];
  }

  function arrayLength(value: unknown): number {
    return Array.isArray(value) ? value.length : 0;
  }

  function apiUrl(value: unknown): string {
    if (typeof value !== 'string' || !value) return '';
    return value
      .replace('/test64/images/', '/api/test64/images/')
      .replace('/test64/mobile-oracle-forensic-audit/assets', '/api/test64/mobile-oracle-forensic-audit/assets');
  }

  function bboxOrNull(value: unknown): Bbox | null {
    if (!Array.isArray(value) || value.length !== 4) return null;
    const bbox = value.map((item) => (typeof item === 'number' && Number.isFinite(item) ? item : Number.NaN));
    return bbox.every(Number.isFinite) ? (bbox as Bbox) : null;
  }

  function recordValue(value: unknown): JsonRecord {
    return isRecord(value) ? value : {};
  }

  function truthBbox(item: JsonRecord | null): Bbox | null {
    return bboxOrNull(recordValue(item?.truth).bbox_xyxy);
  }

  function selectedBbox(item: JsonRecord | null): Bbox | null {
    return bboxOrNull(recordValue(item?.selected).final_recognition_bbox_xyxy);
  }

  function overlapBbox(item: JsonRecord | null): Bbox | null {
    return bboxOrNull(recordValue(item?.truth_overlap).bbox_xyxy);
  }

  function rectAttrs(bbox: Bbox): { x: number; y: number; width: number; height: number } {
    return {
      x: bbox[0],
      y: bbox[1],
      width: Math.max(0, bbox[2] - bbox[0]),
      height: Math.max(0, bbox[3] - bbox[1])
    };
  }

  function rectAttr(bbox: Bbox | null, attr: 'x' | 'y' | 'width' | 'height'): number {
    return bbox ? rectAttrs(bbox)[attr] : 0;
  }

  function formatBbox(value: unknown): string {
    const bbox = bboxOrNull(value);
    return bbox ? bbox.map((item) => Math.round(item)).join(', ') : '-';
  }

  function formatCount(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? String(value) : '-';
  }

  function formatRuntime(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)} ms` : '-';
  }

  function formatConfidence(value: unknown): string {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '-';
  }

  function classLabel(value: unknown): string {
    return text(value).replaceAll('_', ' ') || 'unclassified';
  }

  function expectedLabel(item: JsonRecord): string {
    const expected = recordValue(item.expected);
    return text(expected.label) || text(item.expected) || '-';
  }

  function detectedLabel(item: JsonRecord): string {
    return text(item.detected) || '-';
  }

  function selectedParsedDate(item: JsonRecord): string {
    const parsed = recordValue(recordValue(item.selected).parsed);
    return text(parsed.parsed_date) || '-';
  }

  function selectedRecognition(item: JsonRecord): JsonRecord {
    return recordValue(recordValue(item.selected).recognition);
  }

  function truthCrop(item: JsonRecord): JsonRecord {
    return recordValue(item.truth_crop_oracle);
  }

  function topFinalCrops(item: JsonRecord): JsonRecord[] {
    const candidates = list(item.final_candidates);
    const selectedFirst = [
      ...candidates.filter((candidate) => candidate.selected_final === true),
      ...candidates.filter((candidate) => candidate.selected_final !== true)
    ];
    const crops: JsonRecord[] = [];
    for (const candidate of selectedFirst) {
      for (const crop of list(candidate.final_crops)) {
        crops.push({ ...crop, candidate_id: candidate.candidate_id, candidate_type: candidate.candidate_type });
      }
      if (crops.length >= 6) break;
    }
    return crops.slice(0, 6);
  }

  function parsedVariants(crop: JsonRecord): JsonRecord[] {
    return list(crop.svtr_variants)
      .filter((variant) => text(recordValue(variant.best_parse).parsed_date))
      .slice(0, 8);
  }

  function rawVariants(crop: JsonRecord): JsonRecord[] {
    return list(crop.svtr_variants).slice(0, 8);
  }

  function variantRawText(variant: JsonRecord): string {
    return text(recordValue(variant.recognition).raw_text) || '-';
  }

  function variantParsedDate(variant: JsonRecord): string {
    return text(recordValue(variant.best_parse).parsed_date) || '-';
  }

  function variantConfidence(variant: JsonRecord): string {
    return formatConfidence(recordValue(variant.recognition).confidence);
  }

  function blockerText(item: JsonRecord): string {
    return text(item.blocker) || text(item.stable_reason) || '-';
  }

  async function handleResponse(res: Response): Promise<JsonRecord> {
    const data = (await res.json().catch(() => ({}))) as JsonRecord;
    if (!res.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`;
      throw new Error(detail);
    }
    return data;
  }

  async function loadReport(): Promise<void> {
    loading = true;
    notice = null;
    try {
      const payload = await handleResponse(await fetch('/api/test64/mobile-oracle-forensic-audit'));
      runId = text(payload.run_id);
      createdAt = text(payload.created_at);
      stableReport = text(payload.stable_report);
      summary = isRecord(payload.summary) ? payload.summary : {};
      items = Array.isArray(payload.items) ? payload.items.filter(isRecord) : [];
      selectedFilename = text(items[0]?.filename);
      notice = { kind: 'success', text: `Loaded ${items.length} forensic rows.` };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error';
      notice = { kind: 'error', text: message };
    } finally {
      loading = false;
    }
  }

  function handleImageLoad(event: Event): void {
    const image = event.currentTarget as HTMLImageElement;
    imageNaturalWidth = image.naturalWidth;
    imageNaturalHeight = image.naturalHeight;
  }

  onMount(() => {
    void loadReport();
  });
</script>

<section class="panel recognition-review-panel oracle-forensic-panel">
  <div class="panel-heading recognition-review-heading">
    <div>
      <h2>Oracle Forensics</h2>
      <p>Latest stable 39/64 failure audit with source images, oracle crops, candidate crops, SVTR variants, parser output, and final evidence.</p>
    </div>
    <div class="recognition-review-actions">
      <button type="button" on:click={() => loadReport()} disabled={loading}>
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
      <span>Run</span>
      <strong>{runId || '-'}</strong>
    </div>
    <div class="status-item">
      <span>Total</span>
      <strong>{formatCount(summary.total)}</strong>
    </div>
    <div class="status-item">
      <span>Selector Fixes</span>
      <strong>{formatCount(arrayLength(summary.selector_can_be_fixed_safely))}</strong>
    </div>
    <div class="status-item">
      <span>Recognizer/Data</span>
      <strong>{formatCount(arrayLength(summary.recognizer_or_data_blocker))}</strong>
    </div>
  </div>

  <div class="forensic-filter-row">
    {#each filters as option}
      <button type="button" class:active={filter === option} on:click={() => (filter = option)}>
        {option === 'all' ? 'All' : option.split('_')[0]}
        <small>{option === 'all' ? items.length : formatCount(classCounts[option])}</small>
      </button>
    {/each}
  </div>

  <div class="recognition-review-layout">
    <aside class="recognition-review-list" aria-label="Oracle forensic rows">
      {#each filteredItems as item}
        <button
          type="button"
          class:active={selectedFilename === text(item.filename)}
          on:click={() => (selectedFilename = text(item.filename))}
        >
          <span>{classLabel(item.class)}</span>
          <strong>{text(item.filename)}</strong>
          <small>expected {expectedLabel(item)} · detected {detectedLabel(item)}</small>
        </button>
      {/each}
    </aside>

    <section class="recognition-review-detail">
      {#if active}
        <div class="recognition-review-title-row">
          <div>
            <h3>{text(active.filename)}</h3>
            <p class="muted">Expected {expectedLabel(active)} · detected {detectedLabel(active)} · selected {selectedParsedDate(active)}</p>
          </div>
          <span class={`outcome-pill oracle-class-${text(active.class).split('_')[0]}`}>{classLabel(active.class)}</span>
        </div>

        <div class="detection-review-grid">
          <figure class="detection-review-image">
            {#if apiUrl(active.image_url)}
              <div class="detection-review-image-frame">
                <img src={apiUrl(active.image_url)} alt={`Forensic source ${text(active.filename)}`} on:load={handleImageLoad} />
                {#if imageNaturalWidth > 0 && imageNaturalHeight > 0}
                  <svg
                    class="detection-review-overlay"
                    viewBox={`0 0 ${imageNaturalWidth} ${imageNaturalHeight}`}
                    preserveAspectRatio="xMidYMid meet"
                    aria-hidden="true"
                  >
                    {#if truthBbox(active)}
                      <rect
                        class="truth-box"
                        x={rectAttr(truthBbox(active), 'x')}
                        y={rectAttr(truthBbox(active), 'y')}
                        width={rectAttr(truthBbox(active), 'width')}
                        height={rectAttr(truthBbox(active), 'height')}
                      />
                    {/if}
                    {#if overlapBbox(active)}
                      <rect
                        class="best-audit-box"
                        x={rectAttr(overlapBbox(active), 'x')}
                        y={rectAttr(overlapBbox(active), 'y')}
                        width={rectAttr(overlapBbox(active), 'width')}
                        height={rectAttr(overlapBbox(active), 'height')}
                      />
                    {/if}
                    {#if selectedBbox(active)}
                      <rect
                        class="final-recognition-box"
                        x={rectAttr(selectedBbox(active), 'x')}
                        y={rectAttr(selectedBbox(active), 'y')}
                        width={rectAttr(selectedBbox(active), 'width')}
                        height={rectAttr(selectedBbox(active), 'height')}
                      />
                    {/if}
                  </svg>
                {/if}
              </div>
            {:else}
              <div class="recognition-empty-crop">No image</div>
            {/if}
            <figcaption>
              Truth [{formatBbox(recordValue(active.truth).bbox_xyxy)}] · selected [{formatBbox(recordValue(active.selected).final_recognition_bbox_xyxy)}]
            </figcaption>
          </figure>

          <div class="recognition-winner-card">
            <h3>Selected Result</h3>
            <dl class="compact-dl">
              <dt>Status</dt>
              <dd>{text(recordValue(active.selected).status) || text(active.stable_status) || '-'}</dd>
              <dt>Class</dt>
              <dd>{classLabel(active.class)}</dd>
              <dt>Expected</dt>
              <dd>{expectedLabel(active)}</dd>
              <dt>Detected</dt>
              <dd>{detectedLabel(active)}</dd>
              <dt>Parsed</dt>
              <dd>{selectedParsedDate(active)}</dd>
              <dt>Variant</dt>
              <dd>{text(recordValue(active.selected).recognition_variant) || '-'}</dd>
              <dt>Raw Text</dt>
              <dd>{text(selectedRecognition(active).raw_text) || '-'}</dd>
              <dt>SVTR</dt>
              <dd>{formatConfidence(selectedRecognition(active).confidence)}</dd>
              <dt>Runtime</dt>
              <dd>{formatRuntime(active.runtime_ms)}</dd>
              <dt>Truth Overlap</dt>
              <dd>{text(recordValue(active.truth_overlap).verdict) || '-'}</dd>
            </dl>
          </div>
        </div>

        <div class="detection-performance-grid">
          <article>
            <h3>Blocker</h3>
            <p>{blockerText(active)}</p>
          </article>
          <article>
            <h3>Stage Counts</h3>
            <dl class="compact-dl">
              <dt>YOLO OBB</dt>
              <dd>{formatCount(list(active.yolo_obb_proposals).length)}</dd>
              <dt>Probe</dt>
              <dd>{formatCount(list(active.probe_candidates).length)}</dd>
              <dt>Final</dt>
              <dd>{formatCount(list(active.final_candidates).length)}</dd>
              <dt>Date Evidence</dt>
              <dd>{formatCount(list(active.date_evidence).length)}</dd>
            </dl>
          </article>
        </div>

        <div class="forensic-crop-grid">
          <article>
            <h3>Oracle Truth Crop</h3>
            {#if apiUrl(truthCrop(active).crop_url)}
              <img class="forensic-crop-image" src={apiUrl(truthCrop(active).crop_url)} alt="Oracle truth crop" />
            {:else}
              <div class="recognition-empty-crop">No oracle crop</div>
            {/if}
            <p class="muted">{text(truthCrop(active).reason) || 'oracle crop parsed normally'} · reliable {truthCrop(active).reliable === true ? 'yes' : 'no'}</p>
            <div class="forensic-variant-list">
              {#each list(truthCrop(active).variants).slice(0, 8) as variant}
                <span>{text(variant.variant)}: {variantRawText(variant)} ({variantParsedDate(variant)})</span>
              {/each}
            </div>
          </article>

          <article>
            <h3>Date Evidence Clusters</h3>
            {#if list(active.date_evidence_clusters).length}
              <div class="forensic-table-wrap">
                <table class="forensic-table">
                  <thead>
                    <tr><th>Date</th><th>Precision</th><th>Support</th><th>Expected</th></tr>
                  </thead>
                  <tbody>
                    {#each list(active.date_evidence_clusters) as cluster}
                      <tr>
                        <td>{text(cluster.parsed_date) || '-'}</td>
                        <td>{text(cluster.date_precision) || '-'}</td>
                        <td>{formatCount(cluster.support)}</td>
                        <td>{cluster.matches_expected === true ? 'yes' : 'no'}</td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {:else}
              <p class="muted">No parseable DateEvidence cluster.</p>
            {/if}
          </article>
        </div>

        <section class="forensic-candidates">
          <div class="recognition-review-title-row">
            <h3>Final Candidate Crops</h3>
            <p class="muted">Top selected/shortlisted crops with parseable SVTR rows first.</p>
          </div>
          {#if topFinalCrops(active).length}
            <div class="forensic-candidate-grid">
              {#each topFinalCrops(active) as crop}
                <article class="forensic-candidate-card">
                  {#if apiUrl(crop.crop_url)}
                    <img class="forensic-crop-image" src={apiUrl(crop.crop_url)} alt={`Candidate crop ${formatCount(crop.crop_index)}`} />
                  {:else}
                    <div class="recognition-empty-crop">No crop image</div>
                  {/if}
                  <dl class="compact-dl">
                    <dt>Candidate</dt>
                    <dd>{text(crop.candidate_id)} · {text(crop.candidate_type)}</dd>
                    <dt>BBox</dt>
                    <dd>{formatBbox(crop.bbox_xyxy)}</dd>
                    <dt>Overlap</dt>
                    <dd>{text(recordValue(crop.truth_overlap).verdict) || '-'}</dd>
                  </dl>
                  <div class="forensic-table-wrap">
                    <table class="forensic-table">
                      <thead>
                        <tr><th>Variant</th><th>Raw</th><th>Parsed</th><th>Conf</th></tr>
                      </thead>
                      <tbody>
                        {#each (parsedVariants(crop).length ? parsedVariants(crop) : rawVariants(crop)) as variant}
                          <tr>
                            <td>{text(variant.variant)} {text(variant.orientation)}</td>
                            <td>{variantRawText(variant)}</td>
                            <td>{variantParsedDate(variant)}</td>
                            <td>{variantConfidence(variant)}</td>
                          </tr>
                        {/each}
                      </tbody>
                    </table>
                  </div>
                </article>
              {/each}
            </div>
          {:else}
            <p class="muted">No final candidate crops recorded.</p>
          {/if}
        </section>

        <details class="full-pipeline-raw-row">
          <summary>Raw Forensic Row</summary>
          <pre>{JSON.stringify(active, null, 2)}</pre>
        </details>

        <p class="muted">Stable report: {stableReport || '-'} · created {createdAt || '-'}</p>
      {:else}
        <p class="muted">No forensic rows available.</p>
      {/if}
    </section>
  </div>
</section>
