from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.ai.onnx_inference import SVTROnnxTextRecognizer
from app.ai.parser import ExpiryDateParser
from app.ai.preprocess import ImageVariant, ROIImagePreprocessor
from app.ai.rapidocr_text_detector import RapidOCRTextProposalDetector
from app.infra.settings import PROJECT_ROOT, get_settings


DEFAULT_VARIANTS = (
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "unsharp_gray",
    "stamp_blackhat",
)


@dataclass(slots=True)
class ProposalInfo:
    index: int
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: np.ndarray
    confidence: float
    center: np.ndarray
    angle_deg: float
    long_axis: np.ndarray
    short_axis: np.ndarray
    long_len: float
    short_len: float
    area: float


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _normalize_angle(angle: float) -> float:
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _angle_diff(a: float, b: float) -> float:
    diff = abs(_normalize_angle(a - b))
    return min(diff, 180.0 - diff)


def _proposal_geometry(index: int, proposal: Any) -> ProposalInfo | None:
    polygon_raw = getattr(proposal, "polygon_xy", None)
    if polygon_raw:
        polygon = np.asarray(polygon_raw, dtype=np.float32)
    else:
        x1, y1, x2, y2 = proposal.bbox_xyxy
        polygon = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if polygon.shape != (4, 2):
        return None

    ordered = _order_points(polygon)
    width_top = float(np.linalg.norm(ordered[1] - ordered[0]))
    width_bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    height_left = float(np.linalg.norm(ordered[3] - ordered[0]))
    height_right = float(np.linalg.norm(ordered[2] - ordered[1]))
    width = max(width_top, width_bottom)
    height = max(height_left, height_right)
    if width <= 1.0 or height <= 1.0:
        return None

    if width >= height:
        long_axis = _unit(ordered[1] - ordered[0])
        short_axis = _unit(ordered[3] - ordered[0])
        long_len = width
        short_len = height
    else:
        long_axis = _unit(ordered[3] - ordered[0])
        short_axis = _unit(ordered[1] - ordered[0])
        long_len = height
        short_len = width

    angle = _normalize_angle(float(np.degrees(np.arctan2(long_axis[1], long_axis[0]))))
    center = ordered.mean(axis=0)
    x1, y1, x2, y2 = proposal.bbox_xyxy
    return ProposalInfo(
        index=index,
        bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
        polygon_xy=ordered,
        confidence=float(getattr(proposal, "confidence", 0.0) or 0.0),
        center=center,
        angle_deg=angle,
        long_axis=long_axis,
        short_axis=short_axis,
        long_len=long_len,
        short_len=short_len,
        area=float(cv2.contourArea(ordered)),
    )


def _neighbor_candidates(anchor: ProposalInfo, proposals: list[ProposalInfo]) -> list[tuple[float, ProposalInfo]]:
    neighbors: list[tuple[float, ProposalInfo]] = []
    for proposal in proposals:
        if proposal.index == anchor.index:
            continue
        if _angle_diff(anchor.angle_deg, proposal.angle_deg) > 24.0:
            continue
        delta = proposal.center - anchor.center
        along = abs(float(np.dot(delta, anchor.long_axis)))
        across = abs(float(np.dot(delta, anchor.short_axis)))
        max_along = max(260.0, 4.2 * max(anchor.long_len, proposal.long_len))
        max_across = max(90.0, 3.2 * max(anchor.short_len, proposal.short_len))
        if along > max_along or across > max_across:
            continue
        distance = float(np.hypot(along, across))
        score = distance - (proposal.confidence * 65.0)
        neighbors.append((score, proposal))
    neighbors.sort(key=lambda item: item[0])
    return neighbors


def _build_groups(proposals: list[ProposalInfo], *, top_k: int, max_groups: int) -> list[tuple[int, ...]]:
    usable = proposals[:top_k]
    groups: set[tuple[int, ...]] = set()
    by_index = {proposal.index: proposal for proposal in usable}
    for anchor in usable:
        neighbors = [item[1] for item in _neighbor_candidates(anchor, usable)[:7]]
        groups.add((anchor.index,))
        for size in (1, 2, 3):
            for combo in itertools.combinations(neighbors, size):
                groups.add(tuple(sorted([anchor.index, *(item.index for item in combo)])))
        if len(neighbors) >= 4:
            groups.add(tuple(sorted([anchor.index, *(item.index for item in neighbors[:4])])))

    def group_key(group: tuple[int, ...]) -> tuple[float, int, float]:
        items = [by_index[idx] for idx in group]
        confidence = sum(item.confidence for item in items) / max(1, len(items))
        area = sum(item.area for item in items)
        return (confidence + 0.045 * len(items), len(items), area)

    return sorted(groups, key=group_key, reverse=True)[:max_groups]


def _expanded_oriented_crop(
    image: np.ndarray,
    points: np.ndarray,
    *,
    pad_x: int,
    pad_y: int,
) -> tuple[np.ndarray | None, list[list[float]] | None]:
    if points.shape[0] < 4:
        return None, None
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = _order_points(cv2.boxPoints(rect))
    width = max(float(np.linalg.norm(box[1] - box[0])), float(np.linalg.norm(box[2] - box[3])))
    height = max(float(np.linalg.norm(box[3] - box[0])), float(np.linalg.norm(box[2] - box[1])))
    if width <= 2.0 or height <= 2.0:
        return None, None

    center = box.mean(axis=0)
    axis_x = _unit(box[1] - box[0])
    axis_y = _unit(box[3] - box[0])
    width += 2.0 * float(pad_x)
    height += 2.0 * float(pad_y)
    expanded = np.asarray(
        [
            center - axis_x * (width / 2.0) - axis_y * (height / 2.0),
            center + axis_x * (width / 2.0) - axis_y * (height / 2.0),
            center + axis_x * (width / 2.0) + axis_y * (height / 2.0),
            center - axis_x * (width / 2.0) + axis_y * (height / 2.0),
        ],
        dtype=np.float32,
    )
    out_w = max(3, int(round(width)))
    out_h = max(3, int(round(height)))
    dst = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(expanded, dst)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] > crop.shape[1] * 1.25:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop, expanded.tolist()


def _recognition_variants(crop: np.ndarray, allowed: tuple[str, ...]) -> list[ImageVariant]:
    variants = [ImageVariant("original_color_tight", crop.copy(), purpose="recognition")]
    preprocessor = ROIImagePreprocessor()
    seen = {"original_color_tight"}
    for variant in preprocessor.recognition_variants(crop, allowed_names=tuple(name for name in allowed if name != "original_color_tight")):
        if variant.name in allowed and variant.name not in seen:
            variants.append(variant)
            seen.add(variant.name)
    return variants


def _rotation_variants(image: np.ndarray, *, try_all: bool) -> list[tuple[str, np.ndarray]]:
    variants = [("original", image)]
    if try_all:
        variants.extend(
            [
                ("rot180", cv2.rotate(image, cv2.ROTATE_180)),
                ("rot90_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                ("rot90_ccw", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]
        )
    return variants


def _parsed_label(parsed: Any) -> str | None:
    if parsed.parsed_month is None or parsed.parsed_year is None:
        return None
    if parsed.date_precision == "month" or parsed.parsed_day is None:
        return f"{int(parsed.parsed_year):04d}-{int(parsed.parsed_month):02d}"
    return date(int(parsed.parsed_year), int(parsed.parsed_month), int(parsed.parsed_day)).isoformat()


def _result_rank(result: dict[str, Any]) -> tuple[float, float, float, int]:
    parsed = result.get("parsed_label")
    precision_bonus = 2.0 if parsed and result.get("date_precision") == "day" else 1.0 if parsed else 0.0
    parser_conf = float(result.get("parser_confidence") or 0.0)
    ocr_conf = float(result.get("ocr_confidence") or 0.0)
    raw = str(result.get("raw_text") or "")
    useful_chars = sum(1 for char in raw if char.isdigit() or char in "/.-")
    return (precision_bonus, parser_conf, ocr_conf, useful_chars)


def _write_contact_sheet(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    thumb_w = 360
    label_h = 82
    rows = min(16, len(results))
    cols = 2
    cell_w = thumb_w
    cell_h = 210 + label_h
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for idx, result in enumerate(results[: rows * cols]):
        crop_path = result.get("crop_path")
        if not crop_path:
            continue
        crop = cv2.imread(str(crop_path))
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        scale = min((thumb_w - 12) / max(1, w), 200 / max(1, h))
        resized = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * cell_h
        sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        text_y = y0 + 214
        lines = [
            f"{idx + 1}. group={result.get('group_indices')} pad={result.get('pad')}",
            f"{result.get('raw_text')} -> {result.get('parsed_label')}",
            f"ocr={result.get('ocr_confidence')} parser={result.get('parser_confidence')}",
        ]
        for line in lines:
            cv2.putText(sheet, line[:52], (x0 + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 30, 40), 1, cv2.LINE_AA)
            text_y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def run_probe(args: argparse.Namespace) -> int:
    image_path = _resolve_path(Path(args.image))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SystemExit(f"image not found or unreadable: {image_path}")

    settings = get_settings()
    output_root = _resolve_path(Path(args.output_dir)) / f"{image_path.stem}_{_stamp()}"
    crops_dir = output_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector = RapidOCRTextProposalDetector(
        ocr_version="PP-OCRv5",
        model_type="mobile",
        lang_type="ch",
        limit_side_len=int(args.limit_side_len),
        limit_type="max",
        max_candidates=int(args.max_candidates),
        max_boxes=int(args.max_boxes),
        min_confidence=float(args.min_confidence),
        min_box_area=8,
    )
    detect_start = perf_counter()
    proposals_raw, detect_error = detector.detect(image)
    detect_ms = (perf_counter() - detect_start) * 1000.0
    proposals = [item for item in (_proposal_geometry(idx, proposal) for idx, proposal in enumerate(proposals_raw)) if item is not None]
    print(f"RapidOCR proposals: {len(proposals)} accepted in {detect_ms:.1f} ms")
    if detect_error:
        print(f"RapidOCR warning: {detect_error}")

    groups = _build_groups(proposals, top_k=int(args.top_k), max_groups=int(args.max_groups))
    print(f"Oriented groups to try: {len(groups)}")

    recognizer = SVTROnnxTextRecognizer(
        model_path=Path(settings.svtrv2_rec_onnx_path),
        paddle_model_dir=Path(settings.svtrv2_rec_model_dir),
        runtime_device=str(settings.svtrv2_device_mode),
    )
    parser = ExpiryDateParser()
    reference_date = date.fromisoformat(args.reference_date) if args.reference_date else date.today()
    allowed_variants = tuple(args.variants.split(",")) if args.variants else DEFAULT_VARIANTS
    by_index = {proposal.index: proposal for proposal in proposals}
    results: list[dict[str, Any]] = []
    expected = args.expected.strip() if args.expected else ""

    probe_start = perf_counter()
    for group_number, group in enumerate(groups, start=1):
        points = np.concatenate([by_index[idx].polygon_xy for idx in group], axis=0)
        for pad_x, pad_y in ((0, 0), (6, 3), (14, 5)):
            base_crop, expanded_polygon = _expanded_oriented_crop(image, points, pad_x=pad_x, pad_y=pad_y)
            if base_crop is None or base_crop.size == 0:
                continue
            crop_base = crops_dir / f"group_{group_number:03d}_{'-'.join(map(str, group))}_pad{pad_x}x{pad_y}.png"
            cv2.imwrite(str(crop_base), base_crop)
            for rotation_name, rotated in _rotation_variants(base_crop, try_all=bool(args.try_all_rotations)):
                for variant in _recognition_variants(rotated, allowed_variants):
                    rec = recognizer.recognize(variant.image)
                    text = rec.normalized_text or rec.raw_text
                    parsed = parser.parse(text, reference_date=reference_date)
                    parsed_label = _parsed_label(parsed)
                    result = {
                        "group_indices": list(group),
                        "group_number": group_number,
                        "pad": [pad_x, pad_y],
                        "expanded_polygon_xy": expanded_polygon,
                        "rotation": rotation_name,
                        "variant": variant.name,
                        "raw_text": rec.raw_text,
                        "normalized_text": rec.normalized_text,
                        "ocr_confidence": None if rec.confidence is None else round(float(rec.confidence), 4),
                        "ocr_reason": rec.reason,
                        "parsed_label": parsed_label,
                        "date_precision": parsed.date_precision,
                        "parser_confidence": round(float(parsed.confidence), 4),
                        "parser_reason": parsed.reason,
                        "parser_candidates": parsed.candidates,
                        "crop_path": str(crop_base),
                    }
                    results.append(result)
                    if expected and parsed_label == expected:
                        elapsed = (perf_counter() - probe_start) * 1000.0
                        print(
                            "FOUND expected "
                            f"{expected} raw={rec.raw_text!r} conf={result['ocr_confidence']} "
                            f"group={list(group)} pad={[pad_x, pad_y]} variant={variant.name} "
                            f"elapsed_ms={elapsed:.1f}"
                        )
                        if args.stop_on_expected:
                            break
                if args.stop_on_expected and expected and results and results[-1].get("parsed_label") == expected:
                    break
            if args.stop_on_expected and expected and results and results[-1].get("parsed_label") == expected:
                break
        if args.stop_on_expected and expected and results and results[-1].get("parsed_label") == expected:
            break

    results.sort(key=_result_rank, reverse=True)
    top_results = results[: int(args.report_top)]
    payload = {
        "image": str(image_path),
        "output_root": str(output_root),
        "rapidocr_detect_ms": round(detect_ms, 2),
        "proposal_count": len(proposals),
        "group_count": len(groups),
        "expected": expected or None,
        "top_results": top_results,
        "proposals": [
            {
                "index": proposal.index,
                "bbox_xyxy": list(proposal.bbox_xyxy),
                "polygon_xy": proposal.polygon_xy.tolist(),
                "confidence": round(proposal.confidence, 4),
                "angle_deg": round(proposal.angle_deg, 2),
                "long_len": round(proposal.long_len, 2),
                "short_len": round(proposal.short_len, 2),
            }
            for proposal in proposals
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "probe_report.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_contact_sheet(output_root / "top_candidates_contact_sheet.jpg", top_results)

    print("Top parsed/recognized candidates:")
    for idx, result in enumerate(top_results[:10], start=1):
        print(
            f"{idx:02d}. parsed={result.get('parsed_label')} raw={result.get('raw_text')!r} "
            f"ocr={result.get('ocr_confidence')} parser={result.get('parser_confidence')} "
            f"group={result.get('group_indices')} pad={result.get('pad')} variant={result.get('variant')}"
        )
    print(f"Report: {output_root / 'probe_report.json'}")
    print(f"Contact sheet: {output_root / 'top_candidates_contact_sheet.jpg'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe oriented RapidOCR proposal grouping before production pipeline changes.")
    parser.add_argument("--image", default="test64/IMG_0885.JPG")
    parser.add_argument("--output-dir", default="artifacts/forensics/oriented_rapidocr_group_probe")
    parser.add_argument("--expected", default="2026-05-03")
    parser.add_argument("--reference-date", default=str(date.today()))
    parser.add_argument("--limit-side-len", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--min-confidence", type=float, default=0.08)
    parser.add_argument("--top-k", type=int, default=28)
    parser.add_argument("--max-groups", type=int, default=220)
    parser.add_argument("--report-top", type=int, default=32)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--try-all-rotations", action="store_true")
    parser.add_argument("--stop-on-expected", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_probe(parse_args()))
