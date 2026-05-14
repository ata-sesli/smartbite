from app.ai.expiry_parity_audit import classify_failure_bucket


def test_classifies_missing_proposal_when_no_overlap() -> None:
    bucket = classify_failure_bucket(
        truth_bbox=(100, 100, 180, 130),
        proposal_bboxes=[(10, 10, 60, 30)],
        candidate_bboxes=[],
        correct_candidate_exists=False,
        parser_recovered_date=False,
        selected_wrong_date=False,
        best_truth_crop_text=None,
    )

    assert bucket == "proposal_missing_truth"


def test_classifies_truncated_proposal_when_overlap_is_small() -> None:
    bucket = classify_failure_bucket(
        truth_bbox=(100, 100, 180, 130),
        proposal_bboxes=[(100, 100, 125, 130)],
        candidate_bboxes=[],
        correct_candidate_exists=False,
        parser_recovered_date=False,
        selected_wrong_date=False,
        best_truth_crop_text=None,
    )

    assert bucket == "proposal_truncates_truth"


def test_classifies_selection_when_correct_candidate_exists_but_wrong_date_selected() -> None:
    bucket = classify_failure_bucket(
        truth_bbox=(100, 100, 180, 130),
        proposal_bboxes=[(95, 95, 185, 135)],
        candidate_bboxes=[(95, 95, 185, 135)],
        correct_candidate_exists=True,
        parser_recovered_date=True,
        selected_wrong_date=True,
        best_truth_crop_text="EXP 12/05/2027",
    )

    assert bucket == "candidate_selection_wrong"
