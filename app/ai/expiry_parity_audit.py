from __future__ import annotations


BBox = tuple[int, int, int, int]


def bbox_overlap(a: BBox, b: BBox) -> tuple[float, float]:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = max(1, area_a + area_b - inter)
    containment = inter / float(max(1, min(area_a, area_b)))
    return inter / float(union), containment


def bbox_coverage(target: BBox, candidate: BBox) -> float:
    ix1 = max(target[0], candidate[0])
    iy1 = max(target[1], candidate[1])
    ix2 = min(target[2], candidate[2])
    iy2 = min(target[3], candidate[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (target[2] - target[0]) * (target[3] - target[1]))
    return inter / float(area)


def classify_failure_bucket(
    *,
    truth_bbox: BBox | None,
    proposal_bboxes: list[BBox],
    candidate_bboxes: list[BBox],
    correct_candidate_exists: bool,
    parser_recovered_date: bool,
    selected_wrong_date: bool,
    best_truth_crop_text: str | None,
) -> str:
    if truth_bbox is None:
        return "label_or_benchmark_issue"
    proposal_overlaps = [bbox_overlap(truth_bbox, box) for box in proposal_bboxes]
    if not proposal_overlaps or max(score[1] for score in proposal_overlaps) < 0.25:
        return "proposal_missing_truth"
    best_truth_coverage = max(bbox_coverage(truth_bbox, box) for box in proposal_bboxes)
    if best_truth_coverage < 0.70:
        return "proposal_truncates_truth"
    if selected_wrong_date and correct_candidate_exists:
        return "candidate_selection_wrong"
    if best_truth_crop_text and not parser_recovered_date:
        return "parser_gap"
    if not correct_candidate_exists:
        return "recognizer_misreads_truth_crop"
    return "label_or_benchmark_issue"
