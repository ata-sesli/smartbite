import numpy as np

from app.ai.expiry_candidate_engine import ExpiryCandidateEngine, ProposalBox


def test_engine_builds_single_line_group_and_fallback_candidates() -> None:
    engine = ExpiryCandidateEngine(max_group_candidates_per_roi=20)
    image = np.full((160, 320, 3), 255, dtype=np.uint8)
    proposals = [
        ProposalBox((20, 30, 92, 50), 0.91, "yolo26s_obb", ("yolo26s_obb",), "raw", None),
        ProposalBox((100, 30, 176, 50), 0.88, "yolo26s_obb", ("yolo26s_obb",), "raw", None),
        ProposalBox((22, 64, 160, 86), 0.77, "yolo26s_obb", ("yolo26s_obb",), "raw", None),
    ]

    candidates = engine.build_ranked_candidates(image, proposals)

    assert {"single", "line", "group"} <= {candidate.candidate_type for candidate in candidates}
    assert all(candidate.detector_sources == ("yolo26s_obb",) for candidate in candidates)


def test_engine_builds_full_fallback_when_no_proposals() -> None:
    engine = ExpiryCandidateEngine()
    image = np.full((160, 320, 3), 255, dtype=np.uint8)

    candidates = engine.build_ranked_candidates(image, [])

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "fallback_full"
    assert candidates[0].bbox_xyxy == (0, 0, 320, 160)


def test_engine_prefers_expiry_role_over_higher_ocr_confidence_mfg() -> None:
    engine = ExpiryCandidateEngine()
    selected = engine.select_final_date_for_test(
        [
            {"text": "MFG 01/01/2026", "parsed_date": "2026-01-01", "ocr_score": 0.99, "candidate_score": 2.0},
            {"text": "EXP 01/01/2027", "parsed_date": "2027-01-01", "ocr_score": 0.72, "candidate_score": 1.5},
        ]
    )

    assert selected["parsed_date"] == "2027-01-01"
