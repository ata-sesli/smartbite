from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import re
from typing import Any

import numpy as np

from app.ai.date_role import role_selection_key, score_date_role


DATE_PATTERN = re.compile(r"\b\d{1,4}[./:-]\d{1,2}(?:[./:-]\d{1,4})?\b")


@dataclass(slots=True)
class ProposalBox:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float | None
    source: str
    sources: tuple[str, ...]
    variant_name: str | None
    polygon_xy: tuple[tuple[float, float], ...] | None = None


@dataclass(slots=True)
class RankedExpiryCandidate:
    candidate_id: str
    candidate_type: str
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: tuple[tuple[float, float], ...] | None
    detector_confidence: float | None
    detector_sources: tuple[str, ...]
    detector_variant: str | None
    member_indices: list[int]
    member_bboxes: list[tuple[int, int, int, int]]
    geometry_features: dict[str, float]
    geometry_score: float
    score_breakdown: dict[str, float]
    total_score: float
    recognition_bbox: tuple[int, int, int, int] | None = None
    evidence_bbox: tuple[int, int, int, int] | None = None
    selected_geometry: bool = False
    selected_final: bool = False
    final_rank_before_force_include: int | None = None
    final_rank_after_force_include: int | None = None
    force_included_reason: str | None = None
    final_crop_bbox: tuple[int, int, int, int] | None = None
    final_crop_padding_px: int | None = None
    final_crop_policy: str | None = None
    probe_text: str = ""
    probe_normalized_text: str = ""
    probe_confidence: float | None = None
    probe_reason: str | None = None
    recognition_variant_image: np.ndarray | None = None
    selected_recognition_output: Any | None = None
    selected_recognition_variant: str = ""
    context_probe_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    context_probe_indices: list[int] = field(default_factory=list)
    context_probe_texts: list[str] = field(default_factory=list)
    context_probe_confidences: list[float | None] = field(default_factory=list)
    rapidocr_probe_text: str | None = None
    rapidocr_probe_normalized_text: str | None = None
    rapidocr_probe_confidence: float | None = None
    rapidocr_probe_score: float | None = None
    rapidocr_probe_orientation: str | None = None
    rapidocr_probe_polygon_used: bool = False
    adjacent_expiry_keyword: bool = False
    adjacent_production_keyword: bool = False
    keyword_relation: str | None = None
    multiline_split_source_id: str | None = None
    line_index_in_group: int | None = None
    scan_offset_x: int = 0
    scan_offset_y: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.bbox_xyxy

    def crop(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return image[y1:y2, x1:x2]


class ExpiryCandidateEngine:
    def __init__(
        self,
        *,
        max_group_candidates_per_roi: int = 100,
        geometry_top_n: int = 12,
        context_probe_enabled: bool = True,
        context_max_boxes_per_candidate: int = 2,
        require_multiline_for_line_candidates: bool = False,
    ) -> None:
        self.max_group_candidates_per_roi = max(0, int(max_group_candidates_per_roi))
        self.geometry_top_n = max(1, int(geometry_top_n))
        self.context_probe_enabled = bool(context_probe_enabled)
        self.context_max_boxes_per_candidate = max(0, int(context_max_boxes_per_candidate))
        self.require_multiline_for_line_candidates = bool(require_multiline_for_line_candidates)

    def build_ranked_candidates(
        self,
        image: np.ndarray,
        proposals: list[ProposalBox],
    ) -> list[RankedExpiryCandidate]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []
        if not proposals:
            return [
                self._candidate_from_bbox(
                    candidate_id="fallback_full",
                    candidate_type="fallback",
                    bbox=(0, 0, w, h),
                    image_shape=(h, w),
                    proposals=[],
                    member_indices=[],
                    polygon_xy=None,
                    neighbor_counts={},
                )
            ]

        candidates: list[RankedExpiryCandidate] = []
        seen: set[tuple[int, int, int, int, tuple[int, ...], str]] = set()
        neighbor_counts = self._neighbor_counts(proposals)

        def add(
            candidate_type: str,
            bbox: tuple[int, int, int, int],
            member_indices: list[int],
            polygon: tuple[tuple[float, float], ...] | None,
        ) -> None:
            key = (*bbox, tuple(member_indices), candidate_type)
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                self._candidate_from_bbox(
                    candidate_id=f"cand_{len(candidates) + 1}",
                    candidate_type=candidate_type,
                    bbox=bbox,
                    image_shape=(h, w),
                    proposals=proposals,
                    member_indices=member_indices,
                    polygon_xy=polygon,
                    neighbor_counts=neighbor_counts,
                )
            )

        for idx, proposal in enumerate(proposals):
            add("single", proposal.bbox_xyxy, [idx], proposal.polygon_xy)

        line_clusters = self._cluster_boxes_into_lines(proposals)
        multiline_source_id = "multiline_block_1" if len(line_clusters) > 1 else None
        if (not self.require_multiline_for_line_candidates) or len(line_clusters) > 1:
            for line_index, line in enumerate(line_clusters):
                if len(line) < 2:
                    continue
                add("line", self._union_many_bboxes([proposals[idx].bbox_xyxy for idx in line]), line, None)
                if multiline_source_id and candidates:
                    candidates[-1].multiline_split_source_id = multiline_source_id
                    candidates[-1].line_index_in_group = line_index

        group_count = 0
        for i in range(len(proposals)):
            if self.max_group_candidates_per_roi <= 0 or group_count >= self.max_group_candidates_per_roi:
                break
            if not self._box_has_grouping_geometry(proposals[i].bbox_xyxy, image_shape=(h, w)):
                continue
            neighbors = [
                (self._box_center_distance(proposals[i].bbox_xyxy, proposals[j].bbox_xyxy), j)
                for j in range(len(proposals))
                if j != i
                and self._box_has_grouping_geometry(proposals[j].bbox_xyxy, image_shape=(h, w))
                and self._boxes_are_groupable(proposals[i].bbox_xyxy, proposals[j].bbox_xyxy)
            ]
            neighbors.sort(key=lambda item: item[0])
            for _distance, j in neighbors[:2]:
                if i >= j:
                    continue
                add("group", self._union_bbox(proposals[i].bbox_xyxy, proposals[j].bbox_xyxy), [i, j], None)
                group_count += 1
                if group_count >= self.max_group_candidates_per_roi:
                    break

        candidates.sort(key=lambda item: item.geometry_score, reverse=True)
        for idx, candidate in enumerate(candidates, start=1):
            candidate.candidate_id = f"cand_{idx}"
        return candidates

    @staticmethod
    def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    @classmethod
    def _union_many_bboxes(cls, bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        out = bboxes[0]
        for bbox in bboxes[1:]:
            out = cls._union_bbox(out, bbox)
        return out

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _boxes_are_groupable(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        aw, ah = max(1, a[2] - a[0]), max(1, a[3] - a[1])
        bw, bh = max(1, b[2] - b[0]), max(1, b[3] - b[1])
        a_cx, a_cy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        b_cx, b_cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        center_distance = float(np.hypot(a_cx - b_cx, a_cy - b_cy))
        diag_ref = max(np.hypot(aw, ah), np.hypot(bw, bh))
        x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        x_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        y_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return bool(
            center_distance <= (diag_ref * 2.4)
            and (
                y_overlap >= min(ah, bh) * 0.25
                or x_overlap >= min(aw, bw) * 0.25
                or (x_gap <= max(aw, bw) * 1.5 and y_gap <= max(ah, bh) * 1.5)
            )
        )

    @staticmethod
    def _geometry_size_score(area_ratio: float) -> float:
        if area_ratio < 0.0006:
            return -0.6
        if area_ratio < 0.002:
            return -0.2
        if area_ratio <= 0.12:
            return 0.55
        if area_ratio <= 0.32:
            return 0.2
        return -0.35

    @staticmethod
    def _geometry_aspect_score(aspect: float) -> float:
        if aspect < 0.5:
            return -0.2
        if aspect < 1.0:
            return 0.05
        if aspect <= 8.0:
            return 0.35
        if aspect <= 14.0:
            return 0.2
        return -0.1

    @staticmethod
    def _geometry_edge_score(edge_proximity: float) -> float:
        if edge_proximity < 0.01:
            return -0.18
        if edge_proximity < 0.03:
            return -0.07
        if edge_proximity < 0.12:
            return 0.08
        return 0.14

    def _candidate_from_bbox(
        self,
        *,
        candidate_id: str,
        candidate_type: str,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
        proposals: list[ProposalBox],
        member_indices: list[int],
        polygon_xy: tuple[tuple[float, float], ...] | None,
        neighbor_counts: dict[int, int] | None = None,
    ) -> RankedExpiryCandidate:
        h, w = image_shape
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area_ratio = float(bw * bh) / float(max(1, h * w))
        aspect = float(bw) / float(bh)
        line_likeness = min(1.0, aspect / 6.0) if aspect >= 1.0 else max(0.0, aspect * 0.4)
        edge_dist = min(bbox[0], bbox[1], max(0, w - bbox[2]), max(0, h - bbox[3]))
        edge_proximity = float(edge_dist) / float(max(1, min(h, w)))
        confidences = [
            float(proposals[idx].confidence)
            for idx in member_indices
            if idx < len(proposals) and proposals[idx].confidence is not None
        ]
        det_conf_avg = sum(confidences) / len(confidences) if confidences else 0.0
        member_neighbor_counts = [neighbor_counts.get(idx, 0) for idx in member_indices] if neighbor_counts else []
        neighbor_avg = float(sum(member_neighbor_counts) / len(member_neighbor_counts)) if member_neighbor_counts else 0.0
        detector_sources = tuple(
            dict.fromkeys(
                source
                for idx in member_indices
                if idx < len(proposals)
                for source in proposals[idx].sources
            )
        ) or ("fallback",)
        detector_variants = tuple(
            dict.fromkeys(
                proposals[idx].variant_name
                for idx in member_indices
                if idx < len(proposals) and proposals[idx].variant_name
            )
        )

        size_score = self._geometry_size_score(area_ratio)
        aspect_score = self._geometry_aspect_score(aspect)
        edge_score = self._geometry_edge_score(edge_proximity)
        neighbor_score = min(0.2, neighbor_avg * 0.05)
        det_score = (det_conf_avg - 0.5) * 0.25 if confidences else 0.0
        group_bonus = 0.1 if candidate_type == "group" else 0.0
        fallback_bonus = 0.05 if candidate_type == "fallback" else 0.0
        multi_source_bonus = 0.25 if len(detector_sources) > 1 else 0.0
        geometry_score = (
            size_score
            + aspect_score
            + edge_score
            + neighbor_score
            + det_score
            + group_bonus
            + fallback_bonus
            + multi_source_bonus
        )
        context_matches = self._context_probe_matches(
            candidate_bbox=bbox,
            member_indices=member_indices,
            proposals=proposals,
        )
        context_bboxes = [context_bbox for _idx, context_bbox, _relation in context_matches]
        context_indices = [idx for idx, _context_bbox, _relation in context_matches]
        keyword_relation = context_matches[0][2] if context_matches else None

        return RankedExpiryCandidate(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            bbox_xyxy=bbox,
            polygon_xy=polygon_xy,
            detector_confidence=det_conf_avg if confidences else None,
            detector_sources=detector_sources,
            detector_variant=detector_variants[0] if detector_variants else None,
            member_indices=member_indices,
            member_bboxes=[proposals[idx].bbox_xyxy for idx in member_indices if idx < len(proposals)],
            geometry_features={
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect, 4),
                "line_likeness": round(line_likeness, 4),
                "edge_proximity": round(edge_proximity, 4),
                "neighbor_count": round(neighbor_avg, 2),
                "det_conf_avg": round(det_conf_avg, 4),
            },
            geometry_score=geometry_score,
            score_breakdown={
                "size_score": size_score,
                "aspect_score": aspect_score,
                "edge_score": edge_score,
                "neighbor_score": neighbor_score,
                "det_score": det_score,
                "group_bonus": group_bonus,
                "fallback_bonus": fallback_bonus,
                "multi_source_bonus": multi_source_bonus,
            },
            recognition_bbox=bbox,
            evidence_bbox=self._union_many_bboxes([bbox, *context_bboxes]) if context_bboxes else bbox,
            context_probe_bboxes=context_bboxes,
            context_probe_indices=context_indices,
            keyword_relation=keyword_relation,
            total_score=geometry_score,
        )

    def _cluster_boxes_into_lines(self, proposals: list[ProposalBox]) -> list[list[int]]:
        ordered = sorted(
            range(len(proposals)),
            key=lambda idx: (
                (proposals[idx].bbox_xyxy[1] + proposals[idx].bbox_xyxy[3]) / 2.0,
                proposals[idx].bbox_xyxy[0],
            ),
        )
        lines: list[list[int]] = []
        for idx in ordered:
            box = proposals[idx].bbox_xyxy
            cy = (box[1] + box[3]) / 2.0
            height = max(1, box[3] - box[1])
            placed = False
            for line in lines:
                line_boxes = [proposals[item].bbox_xyxy for item in line]
                line_cy = sum((item[1] + item[3]) / 2.0 for item in line_boxes) / len(line_boxes)
                line_height = sum(max(1, item[3] - item[1]) for item in line_boxes) / len(line_boxes)
                y_overlap = max(
                    0,
                    min(max(item[3] for item in line_boxes), box[3])
                    - max(min(item[1] for item in line_boxes), box[1]),
                )
                if abs(cy - line_cy) <= max(height, line_height) * 0.65 or y_overlap >= min(height, line_height) * 0.35:
                    line.append(idx)
                    placed = True
                    break
            if not placed:
                lines.append([idx])
        return [sorted(line, key=lambda idx: proposals[idx].bbox_xyxy[0]) for line in lines]

    @staticmethod
    def _neighbor_counts(proposals: list[ProposalBox]) -> dict[int, int]:
        counts = {idx: 0 for idx in range(len(proposals))}
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                if ExpiryCandidateEngine._boxes_are_groupable(proposals[i].bbox_xyxy, proposals[j].bbox_xyxy):
                    counts[i] += 1
                    counts[j] += 1
        return counts

    @staticmethod
    def _box_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        return float(
            np.hypot(
                ((a[0] + a[2]) / 2.0) - ((b[0] + b[2]) / 2.0),
                ((a[1] + a[3]) / 2.0) - ((b[1] + b[3]) / 2.0),
            )
        )

    @staticmethod
    def _box_has_grouping_geometry(bbox: tuple[int, int, int, int], *, image_shape: tuple[int, int]) -> bool:
        h, w = image_shape
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area_ratio = float(bw * bh) / float(max(1, h * w))
        aspect = float(bw) / float(bh)
        return bool(0.0003 <= area_ratio <= 0.18 and 0.35 <= aspect <= 14.0)

    @staticmethod
    def _bbox_vertical_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return float(overlap) / float(max(1, min(a[3] - a[1], b[3] - b[1])))

    @staticmethod
    def _bbox_horizontal_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        return float(overlap) / float(max(1, min(a[2] - a[0], b[2] - b[0])))

    @classmethod
    def _context_relation(
        cls,
        candidate_bbox: tuple[int, int, int, int],
        context_bbox: tuple[int, int, int, int],
    ) -> str | None:
        cx1, cy1, cx2, cy2 = candidate_bbox
        bx1, by1, bx2, by2 = context_bbox
        candidate_h = max(1, cy2 - cy1)
        context_h = max(1, by2 - by1)
        ref_h = max(candidate_h, context_h)
        vertical_overlap = cls._bbox_vertical_overlap_ratio(candidate_bbox, context_bbox)
        if bx2 <= cx1 and vertical_overlap >= 0.45 and cx1 - bx2 <= 4 * ref_h:
            return "left_same_line"
        if by2 <= cy1:
            gap = cy1 - by2
            horizontal_overlap = cls._bbox_horizontal_overlap_ratio(candidate_bbox, context_bbox)
            candidate_cx = (cx1 + cx2) / 2.0
            context_cx = (bx1 + bx2) / 2.0
            center_aligned = abs(candidate_cx - context_cx) <= max(cx2 - cx1, bx2 - bx1)
            if gap <= 2 * ref_h and (horizontal_overlap >= 0.25 or center_aligned):
                return "above_aligned"
        return None

    def _context_probe_matches(
        self,
        *,
        candidate_bbox: tuple[int, int, int, int],
        member_indices: list[int],
        proposals: list[ProposalBox],
    ) -> list[tuple[int, tuple[int, int, int, int], str]]:
        if not self.context_probe_enabled or self.context_max_boxes_per_candidate <= 0:
            return []
        excluded = set(member_indices)
        matches: list[tuple[tuple[int, int, float], int, tuple[int, int, int, int], str]] = []
        cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
        cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
        for idx, proposal in enumerate(proposals):
            if idx in excluded:
                continue
            bbox = proposal.bbox_xyxy
            relation = self._context_relation(candidate_bbox, bbox)
            if relation is None:
                continue
            bx = (bbox[0] + bbox[2]) / 2.0
            by = (bbox[1] + bbox[3]) / 2.0
            distance = float(np.hypot(cx - bx, cy - by))
            priority = 0 if relation == "left_same_line" else 1
            matches.append(((priority, int(distance), -float(proposal.confidence or 0.0)), idx, bbox, relation))
        matches.sort(key=lambda item: item[0])
        return [(idx, bbox, relation) for _key, idx, bbox, relation in matches[: self.context_max_boxes_per_candidate]]

    @staticmethod
    def normalize_ocr_confusions(text: str) -> str:
        mapping = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})

        def replace_token(match: "re.Match[str]") -> str:
            token = match.group(0)
            has_digit_or_separator = any(ch.isdigit() or ch in "/.-:" for ch in token)
            has_confusable_chars = any(ch in "OILSB" for ch in token)
            return token.translate(mapping) if has_digit_or_separator and has_confusable_chars else token

        return re.sub(r"[A-Z0-9/.\-:]+", replace_token, text.upper())

    @classmethod
    def build_parse_inputs_from_text(cls, raw_text: str, normalized_text: str | None = None) -> list[str]:
        base = cls.normalize_ocr_confusions(normalized_text or "")
        raw = cls.normalize_ocr_confusions(raw_text or "")
        inputs: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = " ".join(value.strip().upper().split())
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            inputs.append(normalized)

        add(base)
        for line in raw.splitlines():
            if any(ch.isdigit() for ch in line):
                add(line)
        tokens = [token for token in raw.split() if any(ch.isdigit() for ch in token)]
        for token in tokens:
            if any(sep in token for sep in ("/", "-", ".")):
                add(token)
        for index in range(len(tokens) - 1):
            add(f"{tokens[index]} {tokens[index + 1]}")
        for index in range(len(tokens) - 2):
            add(f"{tokens[index]} {tokens[index + 1]} {tokens[index + 2]}")
        return inputs or ([base] if base else [raw])

    @staticmethod
    def parse_input_selection_key(text: str, parsed: Any) -> tuple[int, int, int, int, float]:
        parsed_date = getattr(parsed, "parsed_date", None)
        confidence = float(getattr(parsed, "confidence", 0.0) or 0.0)
        return role_selection_key(evidence=score_date_role(text), parsed_date=parsed_date, specificity=1, confidence=confidence)

    @staticmethod
    def select_final_date_for_test(items: list[dict[str, Any]]) -> dict[str, Any]:
        def key(item: dict[str, Any]) -> tuple[int, int, int, float, float]:
            role = score_date_role(str(item.get("text") or ""))
            return (
                role.priority,
                role.non_production,
                int(bool(item.get("parsed_date"))),
                float(item.get("candidate_score") or 0.0),
                float(item.get("ocr_score") or 0.0),
            )

        return max(items, key=key)
