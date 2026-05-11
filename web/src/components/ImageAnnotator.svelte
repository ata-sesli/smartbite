<script lang="ts">
  import { createEventDispatcher, onDestroy } from 'svelte';

  export let imageUrl = '';
  export let roiUrl = '';
  export let bbox: number[] | null = null;
  export let polygon: number[][] | null = null;
  export let rotationDegrees = 0;

  let wrapElement: HTMLButtonElement | null = null;
  let imageElement: HTMLImageElement | null = null;
  let naturalWidth = 0;
  let naturalHeight = 0;
  let drawing = false;
  let rotating = false;
  let drawStartClient: { x: number; y: number } | null = null;
  let rotateStart: { x: number; angle: number } | null = null;
  let previewBbox: number[] | null = null;
  let previewPolygon: number[][] | null = null;
  let renderedBbox: number[] | null = null;
  let renderedPolygon: number[][] | null = null;
  let showRoiOverlay = false;

  type BboxChangeDetail = {
    bbox: number[] | null;
    polygon: number[][] | null;
    rotationDegrees: number;
  };

  const dispatch = createEventDispatcher<{ bboxChange: BboxChangeDetail }>();

  const proxyUrl = (value: string): string =>
    value.startsWith('/scans/') || value.startsWith('/test64/') ? `/api${value}` : value;

  $: renderedBbox = previewBbox ?? bbox;
  $: renderedPolygon = previewPolygon ?? polygon ?? polygonFromBbox(renderedBbox);

  const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));

  function polygonFromBbox(value: number[] | null): number[][] | null {
    if (!value) return null;
    return [
      [value[0], value[1]],
      [value[2], value[1]],
      [value[2], value[3]],
      [value[0], value[3]]
    ];
  }

  function bboxFromPolygon(points: number[][]): number[] {
    const xs = points.map((point) => clamp(point[0], 0, naturalWidth));
    const ys = points.map((point) => clamp(point[1], 0, naturalHeight));
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }

  function tinyBboxFromPoint(point: { x: number; y: number }): number[] {
    const x1 = clamp(point.x, 0, Math.max(0, naturalWidth - 1));
    const y1 = clamp(point.y, 0, Math.max(0, naturalHeight - 1));
    return [x1, y1, clamp(x1 + 1, 0, naturalWidth), clamp(y1 + 1, 0, naturalHeight)];
  }

  function dispatchAnnotation(nextBbox: number[] | null, nextPolygon: number[][] | null): void {
    dispatch('bboxChange', { bbox: nextBbox, polygon: nextPolygon, rotationDegrees });
  }

  const displayedImageRect = (): DOMRect | null => {
    if (!wrapElement || !naturalWidth || !naturalHeight) return null;
    const elementRect = wrapElement.getBoundingClientRect();
    if (!elementRect.width || !elementRect.height) return null;

    const naturalRatio = naturalWidth / naturalHeight;
    const elementRatio = elementRect.width / elementRect.height;
    if (elementRatio > naturalRatio) {
      const width = elementRect.height * naturalRatio;
      const left = elementRect.left + (elementRect.width - width) / 2;
      return new DOMRect(left, elementRect.top, width, elementRect.height);
    }

    const height = elementRect.width / naturalRatio;
    const top = elementRect.top + (elementRect.height - height) / 2;
    return new DOMRect(elementRect.left, top, elementRect.width, height);
  };

  const pointFromClient = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const rect = displayedImageRect();
    if (!rect) return null;
    if (!rect.width || !rect.height) return null;

    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const radians = (-rotationDegrees * Math.PI) / 180;
    const dx = clientX - centerX;
    const dy = clientY - centerY;
    const unrotatedX = centerX + dx * Math.cos(radians) - dy * Math.sin(radians);
    const unrotatedY = centerY + dx * Math.sin(radians) + dy * Math.cos(radians);

    const x = clamp(((unrotatedX - rect.left) / rect.width) * naturalWidth, 0, naturalWidth);
    const y = clamp(((unrotatedY - rect.top) / rect.height) * naturalHeight, 0, naturalHeight);
    return { x, y };
  };

  function updateSelectionFromClientRect(start: { x: number; y: number }, end: { x: number; y: number }): void {
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    const right = Math.max(start.x, end.x);
    const bottom = Math.max(start.y, end.y);
    const points = [
      pointFromClient(left, top),
      pointFromClient(right, top),
      pointFromClient(right, bottom),
      pointFromClient(left, bottom)
    ];
    if (points.some((point) => point === null)) return;
    const nextPolygon = points.map((point) => [point?.x ?? 0, point?.y ?? 0]);
    const nextBbox = bboxFromPolygon(nextPolygon);
    previewBbox = nextBbox;
    previewPolygon = nextPolygon;
    bbox = nextBbox;
    polygon = nextPolygon;
    dispatchAnnotation(bbox, polygon);
  }

  function stopWindowTracking(): void {
    window.removeEventListener('mousemove', moveDraw);
    window.removeEventListener('mouseup', finishDraw);
    window.removeEventListener('mousemove', moveRotate);
    window.removeEventListener('mouseup', finishRotate);
  }

  function startDraw(event: MouseEvent): void {
    if (event.button !== 0) return;
    const point = pointFromClient(event.clientX, event.clientY);
    if (!point) return;
    event.preventDefault();
    drawing = true;
    drawStartClient = { x: event.clientX, y: event.clientY };
    previewBbox = tinyBboxFromPoint(point);
    previewPolygon = polygonFromBbox(previewBbox);
    bbox = previewBbox;
    polygon = previewPolygon;
    dispatchAnnotation(bbox, polygon);
    stopWindowTracking();
    window.addEventListener('mousemove', moveDraw);
    window.addEventListener('mouseup', finishDraw);
  }

  function moveDraw(event: MouseEvent): void {
    if (!drawing || !drawStartClient) return;
    event.preventDefault();
    updateSelectionFromClientRect(drawStartClient, { x: event.clientX, y: event.clientY });
  }

  function finishDraw(event: MouseEvent): void {
    if (!drawing) return;
    if (drawStartClient) {
      updateSelectionFromClientRect(drawStartClient, { x: event.clientX, y: event.clientY });
    }
    drawing = false;
    drawStartClient = null;
    stopWindowTracking();
    if (bbox && (bbox[2] - bbox[0] < 2 || bbox[3] - bbox[1] < 2)) {
      bbox = null;
      previewBbox = null;
      polygon = null;
      previewPolygon = null;
      dispatchAnnotation(null, null);
      return;
    }
    previewBbox = null;
    previewPolygon = null;
    dispatchAnnotation(bbox, polygon);
  }

  function startRotate(event: MouseEvent): void {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    rotating = true;
    rotateStart = { x: event.clientX, angle: rotationDegrees };
    stopWindowTracking();
    window.addEventListener('mousemove', moveRotate);
    window.addEventListener('mouseup', finishRotate);
  }

  function moveRotate(event: MouseEvent): void {
    if (!rotating || !rotateStart) return;
    event.preventDefault();
    const sensitivity = event.shiftKey ? 0.05 : 0.2;
    rotationDegrees = clamp(rotateStart.angle + (event.clientX - rotateStart.x) * sensitivity, -180, 180);
    dispatchAnnotation(bbox, polygon);
  }

  function finishRotate(): void {
    rotating = false;
    rotateStart = null;
    stopWindowTracking();
    dispatchAnnotation(bbox, polygon);
  }

  function resetRotation(): void {
    rotationDegrees = 0;
    dispatchAnnotation(bbox, polygon);
  }

  function clearBbox(): void {
    drawing = false;
    drawStartClient = null;
    stopWindowTracking();
    bbox = null;
    previewBbox = null;
    polygon = null;
    previewPolygon = null;
    dispatchAnnotation(null, null);
  }

  onDestroy(() => {
    stopWindowTracking();
  });
</script>

<section class="review-zone image-zone">
  <div class="image-zone-header">
    <h3>Image</h3>
    <div class="checkbox-row">
      <label class="checkbox-inline">
        <input type="checkbox" bind:checked={showRoiOverlay} disabled={!roiUrl} /> ROI overlay
      </label>
      <button
        type="button"
        class="ghost small"
        aria-label="Hold and drag left or right to rotate the annotation image"
        on:mousedown={startRotate}
      >
        {rotating ? 'Rotating' : 'Rotate'} {rotationDegrees.toFixed(1)}°
      </button>
      <button type="button" class="ghost small" on:click={resetRotation} disabled={Math.abs(rotationDegrees) < 0.05}>
        Reset rotation
      </button>
      <button type="button" class="ghost small" on:click={clearBbox} disabled={!bbox}>Clear bbox</button>
    </div>
  </div>

  {#if imageUrl}
    <button
      type="button"
      class="review-image-wrap"
      aria-label="Drag on the image to draw an expiry date bounding box"
      bind:this={wrapElement}
      on:mousedown={startDraw}
    >
      <div class="review-image-stage" style={`transform: rotate(${rotationDegrees}deg);`}>
        <img
          bind:this={imageElement}
          src={proxyUrl(imageUrl)}
          alt="Review source"
          draggable="false"
          on:load={() => {
            naturalWidth = imageElement?.naturalWidth ?? 0;
            naturalHeight = imageElement?.naturalHeight ?? 0;
          }}
        />
        {#if renderedPolygon && naturalWidth && naturalHeight}
          <svg
            class="review-bbox-overlay"
            viewBox={`0 0 ${naturalWidth} ${naturalHeight}`}
            preserveAspectRatio="xMidYMid meet"
            aria-hidden="true"
          >
            <polygon points={renderedPolygon.map((point) => `${point[0]},${point[1]}`).join(' ')}></polygon>
          </svg>
        {/if}
      </div>
      {#if showRoiOverlay && roiUrl}
        <div class="roi-overlay">
          <img src={proxyUrl(roiUrl)} alt="ROI reference" />
        </div>
      {/if}
    </button>
    <p class="muted review-bbox-status">
      Current bbox: {renderedBbox ? renderedBbox.map((value) => Math.round(value)).join(', ') : 'not set'}
    </p>
  {:else}
    <p class="muted">No image available for this scan.</p>
  {/if}
</section>
