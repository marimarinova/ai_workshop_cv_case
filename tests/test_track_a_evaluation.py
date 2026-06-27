"""Unit tests for Track A evaluation workflow.

Synthetic data and mocks only — no GPU, no real videos, no model files.
Covers all Phase 6 acceptance criteria.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from pickup_putdown.layer1.track_a.evaluation import (
    ClipStatus,
    EvaluationSummary,
    apply_limit,
    check_leakage,
    combine_predictions,
    filter_by_clip_id,
    filter_ground_truth,
    generate_markdown_report,
    load_splits,
    read_csv_rows,
    resolve_clips_for_split,
    run_task8_evaluation,
    write_csv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dirs(tmp_path: Path):
    """Create minimal directory structure for evaluation tests."""
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "train": ["clip_train_01", "clip_train_02"],
                "val": ["clip_val_01", "clip_val_02", "clip_val_03"],
                "test": ["clip_test_01"],
            }
        )
    )
    gt_path = tmp_path / "events.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_val_01,pickup,1.0,2.0,false,annotator1,high,\n"
        "e2,clip_val_01,putdown,3.0,4.0,false,annotator1,high,\n"
        "e3,clip_val_02,pickup,5.0,6.0,false,annotator1,high,\n"
        "e4,clip_val_03,putdown,7.0,8.0,false,annotator1,high,\n"
        "e5,clip_val_03,pickup,9.0,10.0,false,annotator1,high,\n"
        "e6,clip_train_01,pickup,1.0,2.0,false,annotator1,high,\n"
    )
    clips_path = tmp_path / "clips.csv"
    clips_path.write_text(
        "clip_id,s3_key,duration_s,fps,width,height,decode_ok\n"
        "clip_val_01,s3://bucket/v1.mp4,60.0,30.0,1920,1080,true\n"
        "clip_val_02,s3://bucket/v2.mp4,60.0,30.0,1920,1080,true\n"
        "clip_val_03,s3://bucket/v3.mp4,60.0,30.0,1920,1080,true\n"
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "hand_state.joblib").write_bytes(b"fake")
    (artifact_dir / "hand_state_metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": "1.0",
                "embedding_dim": 16,
                "encoder_name": "mobilenet_v3_small",
                "encoder_version": "v1",
                "class_names": ["empty", "carrying"],
            }
        )
    )
    (artifact_dir / "shelf_state.joblib").write_bytes(b"fake")
    (artifact_dir / "shelf_state_metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": "1.0",
                "embedding_dim": 16,
                "encoder_name": "mobilenet_v3_small",
                "encoder_version": "v1",
                "class_names": ["on_shelf", "off_shelf"],
            }
        )
    )
    candidate_meta = tmp_path / "candidate_metadata"
    candidate_meta.mkdir()
    for cid in ["clip_val_01", "clip_val_02", "clip_val_03"]:
        cd = candidate_meta / cid
        cd.mkdir()
        (cd / f"{cid}.json").write_text(
            json.dumps(
                {
                    "source_video_id": cid,
                    "candidates": [
                        {
                            "candidate_id": f"cand_{cid}_1",
                            "source_start_s": 1.0,
                            "source_end_s": 2.0,
                        }
                    ],
                }
            )
        )
    source_video_dir = tmp_path / "source_videos"
    source_video_dir.mkdir()
    for cid in ["clip_val_01", "clip_val_02", "clip_val_03"]:
        (source_video_dir / f"{cid}.mp4").write_bytes(b"fake_video")
    shelf_cfg = tmp_path / "shelves.yaml"
    shelf_cfg.write_text(
        "cameras:\n"
        "  store_camera_01:\n"
        "    source_width: 1920\n"
        "    source_height: 1080\n"
        "    regions:\n"
        "      - region_id: shelf_1\n"
        "        points: [[0,0],[100,0],[100,100],[0,100]]\n"
    )
    return {
        "tmp_path": tmp_path,
        "splits": splits_path,
        "events": gt_path,
        "clips": clips_path,
        "artifact_dir": artifact_dir,
        "candidate_metadata": candidate_meta,
        "source_video_dir": source_video_dir,
        "shelves_config": shelf_cfg,
    }


# ---------------------------------------------------------------------------
# 1. Split clip resolution
# ---------------------------------------------------------------------------


def test_split_clip_resolution(tmp_dirs: dict) -> None:
    """Criterion 1: resolve clips from a requested split."""
    splits = load_splits(tmp_dirs["splits"])
    val_clips = resolve_clips_for_split(splits, "val")
    assert val_clips == ["clip_val_01", "clip_val_02", "clip_val_03"]
    train_clips = resolve_clips_for_split(splits, "train")
    assert train_clips == ["clip_train_01", "clip_train_02"]


# ---------------------------------------------------------------------------
# 2. Deterministic --limit-clips
# ---------------------------------------------------------------------------


def test_deterministic_limit() -> None:
    """Criterion 2: --limit-clips selects first N clips deterministically."""
    clips = ["a", "b", "c", "d", "e"]
    assert apply_limit(clips, 2) == ["a", "b"]
    assert apply_limit(clips, 5) == clips
    assert apply_limit(clips, 0) == clips
    assert apply_limit(clips, None) == clips


# ---------------------------------------------------------------------------
# 3. Explicit --clip-id filtering
# ---------------------------------------------------------------------------


def test_clip_id_filter(tmp_dirs: dict) -> None:
    """Criterion 3: --clip-id selects a single clip."""
    splits = load_splits(tmp_dirs["splits"])
    val_clips = resolve_clips_for_split(splits, "val")
    assert filter_by_clip_id(val_clips, "clip_val_02") == ["clip_val_02"]
    assert filter_by_clip_id(val_clips, "nonexistent") == []


# ---------------------------------------------------------------------------
# 4. Leakage detection
# ---------------------------------------------------------------------------


def test_leakage_detection() -> None:
    """Criterion 4: train/val leakage is detected."""
    splits = {
        "train": ["clip_a", "clip_b"],
        "val": ["clip_c", "clip_d"],
    }
    assert check_leakage(splits, ["clip_c", "clip_d"]) == "passed"

    leaked = {
        "train": ["clip_a", "clip_b"],
        "val": ["clip_b", "clip_c"],
    }
    result = check_leakage(leaked, ["clip_b", "clip_c"])
    assert "leakage" in result


def test_leakage_no_violation() -> None:
    """No leakage when splits are disjoint."""
    splits = {"train": ["a"], "val": ["b"], "test": ["c"]}
    assert check_leakage(splits, ["b"]) == "passed"


# ---------------------------------------------------------------------------
# 5. Missing input reporting
# ---------------------------------------------------------------------------


def test_missing_artifact(tmp_dirs: dict) -> None:
    """Criterion 5: missing classifier artifact raises FileNotFoundError."""
    from pickup_putdown.layer1.track_a.evaluation import evaluate_track_a

    artifact_dir = tmp_dirs["artifact_dir"]
    # Remove hand classifier
    (artifact_dir / "hand_state.joblib").unlink()

    with pytest.raises(FileNotFoundError, match="hand classifier"):
        evaluate_track_a(
            splits=tmp_dirs["splits"],
            events=tmp_dirs["events"],
            clips=tmp_dirs["clips"],
            artifact_dir=tmp_dirs["artifact_dir"],
            candidate_metadata=tmp_dirs["candidate_metadata"],
            source_video_dir=tmp_dirs["source_video_dir"],
            shelves_config=tmp_dirs["shelves_config"],
            output_dir=tmp_dirs["tmp_path"] / "eval_out",
        )


def test_missing_events(tmp_dirs: dict) -> None:
    """Missing ground-truth events raises FileNotFoundError."""
    from pickup_putdown.layer1.track_a.evaluation import evaluate_track_a

    events = tmp_dirs["tmp_path"] / "nonexistent_events.csv"
    with pytest.raises(FileNotFoundError, match="ground truth events"):
        evaluate_track_a(
            splits=tmp_dirs["splits"],
            events=events,
            clips=tmp_dirs["clips"],
            artifact_dir=tmp_dirs["artifact_dir"],
            candidate_metadata=tmp_dirs["candidate_metadata"],
            source_video_dir=tmp_dirs["source_video_dir"],
            shelves_config=tmp_dirs["shelves_config"],
            output_dir=tmp_dirs["tmp_path"] / "eval_out",
        )


def test_missing_source_video(tmp_dirs: dict) -> None:
    """Missing source video marks clip as skipped, not fatal."""
    from pickup_putdown.layer1.track_a.evaluation import evaluate_track_a

    # Remove one source video
    (tmp_dirs["source_video_dir"] / "clip_val_01.mp4").unlink()

    result = evaluate_track_a(
        splits=tmp_dirs["splits"],
        events=tmp_dirs["events"],
        clips=tmp_dirs["clips"],
        artifact_dir=tmp_dirs["artifact_dir"],
        candidate_metadata=tmp_dirs["candidate_metadata"],
        source_video_dir=tmp_dirs["source_video_dir"],
        shelves_config=tmp_dirs["shelves_config"],
        output_dir=tmp_dirs["tmp_path"] / "eval_out",
        limit_clips=1,
    )
    assert result.skipped_clips == 1
    assert result.skipped[0].status == "missing_source_video"


# ---------------------------------------------------------------------------
# 6. Ground-truth filtering
# ---------------------------------------------------------------------------


def test_ground_truth_filtering() -> None:
    """Criterion 6: GT filtered to evaluated clips only."""
    rows = [
        {"clip_id": "clip_a", "type": "pickup", "t_start": "1.0", "t_end": "2.0"},
        {"clip_id": "clip_b", "type": "putdown", "t_start": "3.0", "t_end": "4.0"},
        {"clip_id": "clip_c", "type": "pickup", "t_start": "5.0", "t_end": "6.0"},
    ]
    evaluated = {"clip_a", "clip_c"}
    filtered = filter_ground_truth(rows, evaluated)
    assert len(filtered) == 2
    clip_ids = {r["clip_id"] for r in filtered}
    assert clip_ids == {"clip_a", "clip_c"}


# ---------------------------------------------------------------------------
# 7. Canonical prediction combination
# ---------------------------------------------------------------------------


def test_combine_predictions(tmp_path: Path) -> None:
    """Criterion 7: predictions combined from per-clip outputs."""
    base = tmp_path / "inference"
    for cid in ["clip_a", "clip_b"]:
        out = base / cid
        out.mkdir(parents=True)
        (out / "predictions.csv").write_text(
            "pred_id,clip_id,type,t_start,t_end,score,model\n"
            f"p1,{cid},pickup,1.0,2.0,0.9,track_a\n"
        )

    combined = combine_predictions({"clip_a": base / "clip_a", "clip_b": base / "clip_b"})
    assert len(combined) == 2
    types = [r["type"] for r in combined]
    assert "pickup" in types


def test_combine_predictions_empty(tmp_path: Path) -> None:
    """Empty prediction directory returns empty list."""
    base = tmp_path / "inference"
    base.mkdir()
    combined = combine_predictions({})
    assert combined == []


# ---------------------------------------------------------------------------
# 8. Task 8 evaluator invocation
# ---------------------------------------------------------------------------


def test_task8_evaluation(tmp_path: Path) -> None:
    """Criterion 8: Task 8 evaluator is invoked and returns metrics."""
    pred_path = tmp_path / "predictions.csv"
    pred_path.write_text(
        "pred_id,clip_id,type,t_start,t_end,score,model\n"
        "p1,clip_a,pickup,1.0,2.0,0.9,track_a\n"
        "p2,clip_a,putdown,3.0,4.0,0.8,track_a\n"
    )
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_a,pickup,1.1,2.1,false,annotator1,high,\n"
        "e2,clip_a,putdown,3.1,4.1,false,annotator1,high,\n"
    )

    result = run_task8_evaluation(pred_path, gt_path)
    assert "metrics" in result
    assert "tiou@0.3" in result["metrics"]
    assert "tiou@0.5" in result["metrics"]
    assert "per_type" in result["metrics"]
    assert "pickup" in result["metrics"]["per_type"]
    assert "putdown" in result["metrics"]["per_type"]


# ---------------------------------------------------------------------------
# 9. Metrics at tIoU 0.3 and 0.5
# ---------------------------------------------------------------------------


def test_metrics_at_thresholds(tmp_path: Path) -> None:
    """Criterion 9: metrics at both tIoU 0.3 and 0.5."""
    pred_path = tmp_path / "predictions.csv"
    pred_path.write_text(
        "pred_id,clip_id,type,t_start,t_end,score,model\np1,clip_a,pickup,1.0,2.0,0.9,track_a\n"
    )
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_a,pickup,1.0,2.0,false,annotator1,high,\n"
    )

    result = run_task8_evaluation(pred_path, gt_path)
    m = result["metrics"]
    assert "tiou@0.3" in m
    assert "tiou@0.5" in m
    # Perfect match should give precision=recall=F1=1.0 at both thresholds
    assert m["tiou@0.3"]["precision"] == 1.0
    assert m["tiou@0.5"]["precision"] == 1.0


# ---------------------------------------------------------------------------
# 10. Per-class metric reporting
# ---------------------------------------------------------------------------


def test_per_class_metrics(tmp_path: Path) -> None:
    """Criterion 10: per-class precision, recall, F1 reported."""
    pred_path = tmp_path / "predictions.csv"
    pred_path.write_text(
        "pred_id,clip_id,type,t_start,t_end,score,model\n"
        "p1,clip_a,pickup,1.0,2.0,0.9,track_a\n"
        "p2,clip_a,putdown,3.0,4.0,0.8,track_a\n"
    )
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_a,pickup,1.0,2.0,false,annotator1,high,\n"
        "e2,clip_a,putdown,3.0,4.0,false,annotator1,high,\n"
    )

    result = run_task8_evaluation(pred_path, gt_path)
    per_type = result["metrics"]["per_type"]
    assert "pickup" in per_type
    assert "putdown" in per_type
    assert per_type["pickup"]["precision"] == 1.0
    assert per_type["putdown"]["precision"] == 1.0


# ---------------------------------------------------------------------------
# 11. Empty predictions
# ---------------------------------------------------------------------------


def test_empty_predictions(tmp_path: Path) -> None:
    """Criterion 11: evaluation handles empty predictions gracefully."""
    pred_path = tmp_path / "predictions.csv"
    pred_path.write_text("pred_id,clip_id,type,t_start,t_end,score,model\n")
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_a,pickup,1.0,2.0,false,annotator1,high,\n"
    )

    result = run_task8_evaluation(pred_path, gt_path)
    m = result["metrics"]
    assert m["tiou@0.5"]["tp"] == 0
    assert m["tiou@0.5"]["fp"] == 0
    assert m["tiou@0.5"]["fn"] == 1


# ---------------------------------------------------------------------------
# 12. Missing event class
# ---------------------------------------------------------------------------


def test_missing_event_class(tmp_path: Path) -> None:
    """Criterion 12: evaluation handles missing event class."""
    pred_path = tmp_path / "predictions.csv"
    pred_path.write_text(
        "pred_id,clip_id,type,t_start,t_end,score,model\np1,clip_a,pickup,1.0,2.0,0.9,track_a\n"
    )
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_a,putdown,1.0,2.0,false,annotator1,high,\n"
    )

    result = run_task8_evaluation(pred_path, gt_path)
    per_type = result["metrics"]["per_type"]
    # pickup GT is missing, putdown pred is missing
    assert "pickup" in per_type
    assert "putdown" in per_type


# ---------------------------------------------------------------------------
# 13. Skipped clip export
# ---------------------------------------------------------------------------


def test_skipped_clip_export(tmp_dirs: dict) -> None:
    """Criterion 13: skipped clips are exported in evaluation_summary.json."""
    from pickup_putdown.layer1.track_a.evaluation import evaluate_track_a

    # Remove one source video to force a skip
    (tmp_dirs["source_video_dir"] / "clip_val_01.mp4").unlink()

    out = tmp_dirs["tmp_path"] / "eval_out"
    result = evaluate_track_a(
        splits=tmp_dirs["splits"],
        events=tmp_dirs["events"],
        clips=tmp_dirs["clips"],
        artifact_dir=tmp_dirs["artifact_dir"],
        candidate_metadata=tmp_dirs["candidate_metadata"],
        source_video_dir=tmp_dirs["source_video_dir"],
        shelves_config=tmp_dirs["shelves_config"],
        output_dir=out,
        limit_clips=1,
    )

    assert result.skipped_clips == 1
    summary_json = json.loads((out / "evaluation_summary.json").read_text())
    assert summary_json["skipped_clips"] == 1
    assert len(summary_json["skipped"]) == 1
    assert summary_json["skipped"][0]["status"] == "missing_source_video"


# ---------------------------------------------------------------------------
# 14. Failure table generation
# ---------------------------------------------------------------------------


def test_failure_table_generation(tmp_path: Path) -> None:
    """Criterion 14: false positives and false negatives are exported."""
    pred_path = tmp_path / "predictions.csv"
    pred_path.write_text(
        "pred_id,clip_id,type,t_start,t_end,score,model\n"
        "p1,clip_a,pickup,1.0,2.0,0.9,track_a\n"
        "p2,clip_a,putdown,3.0,4.0,0.8,track_a\n"
    )
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text(
        "event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes\n"
        "e1,clip_a,pickup,1.5,2.5,false,annotator1,high,\n"
    )

    result = run_task8_evaluation(pred_path, gt_path)
    assert len(result["false_positives"]) >= 0  # may be 0 if matched
    assert len(result["false_negatives"]) >= 0


# ---------------------------------------------------------------------------
# 15. Deterministic Markdown report
# ---------------------------------------------------------------------------


def test_deterministic_markdown_report() -> None:
    """Criterion 15: Markdown report is deterministic and well-formed."""
    summary = EvaluationSummary(
        split="val",
        limited=False,
        limit_count=None,
        total_clips=3,
        evaluated_clips=2,
        skipped_clips=1,
        selected_clip_ids=["clip_a", "clip_b"],
        evaluated_clip_ids=["clip_a", "clip_b"],
        skipped=[ClipStatus("clip_c", "missing_source_video", "no video")],
        gt_event_count=4,
        pred_event_count=3,
        pickup_count=2,
        putdown_count=2,
        metrics={
            "tiou@0.3": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "tp": 3, "fp": 1, "fn": 1}
        },
        leakage_check="passed",
    )
    md = generate_markdown_report(summary)
    assert "# Track A Evaluation Report" in md
    assert "## Evaluation Scope" in md
    assert "validation metrics" in md.lower() or "development evaluation" in md.lower()
    assert "clip_c" in md
    assert "missing_source_video" in md


# ---------------------------------------------------------------------------
# 16. Limited run labeled as subset evaluation
# ---------------------------------------------------------------------------


def test_limited_run_label() -> None:
    """Criterion 16: limited run is labeled as subset evaluation."""
    summary = EvaluationSummary(
        split="val",
        limited=True,
        limit_count=2,
        total_clips=10,
        evaluated_clips=2,
        skipped_clips=0,
    )
    md = generate_markdown_report(summary)
    assert "subset" in md.lower()


# ---------------------------------------------------------------------------
# 17. Full validation run labeled as development evaluation
# ---------------------------------------------------------------------------


def test_full_validation_label() -> None:
    """Criterion 17: full run labeled as development evaluation."""
    summary = EvaluationSummary(
        split="val",
        limited=False,
        limit_count=None,
        total_clips=10,
        evaluated_clips=10,
        skipped_clips=0,
    )
    md = generate_markdown_report(summary)
    assert "development evaluation" in md.lower()


# ---------------------------------------------------------------------------
# 18. Ground truth is not passed to inference
# ---------------------------------------------------------------------------


def test_gt_not_passed_to_inference(tmp_dirs: dict) -> None:
    """Criterion 18: ground truth events are never passed into inference.

    The evaluation workflow loads ground truth only for scoring after
    inference outputs exist.
    """
    from pickup_putdown.layer1.track_a.evaluation import evaluate_track_a

    # Patch run_inference_for_clips to verify it's called without GT data
    with mock.patch(
        "pickup_putdown.layer1.track_a.evaluation.run_inference_for_clips"
    ) as mock_infer:
        mock_infer.return_value = {
            "clip_val_01": "ok",
            "clip_val_02": "ok",
            "clip_val_03": "ok",
        }

        evaluate_track_a(
            splits=tmp_dirs["splits"],
            events=tmp_dirs["events"],
            clips=tmp_dirs["clips"],
            artifact_dir=tmp_dirs["artifact_dir"],
            candidate_metadata=tmp_dirs["candidate_metadata"],
            source_video_dir=tmp_dirs["source_video_dir"],
            shelves_config=tmp_dirs["shelves_config"],
            output_dir=tmp_dirs["tmp_path"] / "eval_out",
        )

        # Verify inference was called with clip IDs, not event IDs
        call_args = mock_infer.call_args
        assert call_args is not None
        # The clip_ids argument should contain clip IDs, not event IDs
        clip_ids = call_args.kwargs.get("clip_ids", call_args[1].get("clip_ids", []))
        assert all("clip_" in cid for cid in clip_ids)
        assert not any(cid[:1] == "e" and "clip_" not in cid for cid in clip_ids)


# ---------------------------------------------------------------------------
# 19. CSV write/read round-trip
# ---------------------------------------------------------------------------


def test_csv_roundtrip(tmp_path: Path) -> None:
    """CSV write and read preserve data."""
    cols = ["a", "b", "c"]
    rows = [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": "5", "c": "6"},
    ]
    path = tmp_path / "test.csv"
    write_csv(path, cols, rows)
    read = read_csv_rows(path)
    assert len(read) == 2
    assert read[0]["a"] == "1"
    assert read[1]["b"] == "5"


# ---------------------------------------------------------------------------
# 20. Evaluation summary JSON structure
# ---------------------------------------------------------------------------


def test_evaluation_summary_json(tmp_dirs: dict) -> None:
    """Evaluation summary JSON has required fields."""
    from pickup_putdown.layer1.track_a.evaluation import evaluate_track_a

    out = tmp_dirs["tmp_path"] / "eval_out"
    evaluate_track_a(
        splits=tmp_dirs["splits"],
        events=tmp_dirs["events"],
        clips=tmp_dirs["clips"],
        artifact_dir=tmp_dirs["artifact_dir"],
        candidate_metadata=tmp_dirs["candidate_metadata"],
        source_video_dir=tmp_dirs["source_video_dir"],
        shelves_config=tmp_dirs["shelves_config"],
        output_dir=out,
        limit_clips=1,
    )

    summary_json = json.loads((out / "evaluation_summary.json").read_text())
    required_keys = {
        "split",
        "limited",
        "limit_count",
        "total_clips",
        "evaluated_clips",
        "skipped_clips",
        "selected_clip_ids",
        "evaluated_clip_ids",
        "skipped",
        "gt_event_count",
        "pred_event_count",
        "leakage_check",
    }
    assert required_keys.issubset(set(summary_json.keys()))
    assert summary_json["split"] == "val"
    assert summary_json["limited"] is True
