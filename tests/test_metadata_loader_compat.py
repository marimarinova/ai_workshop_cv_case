"""Tests for Task 6.2 metadata loader compatibility fix.

Validates that the candidate metadata loader correctly handles:
- Source-level Task 6.1 metadata with nested .candidates[]
- Recursive directory discovery
- Source metadata inheritance (source_video_id -> clip_id, source_bucket, source_key)
- Zero-candidate source file skipping
- Flat candidate format backward compatibility
- Validation and error handling for malformed metadata
- Duplicate candidate ID detection
- Deterministic ordering
- Pilot selection from flattened collection
- End-to-end task generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pickup_putdown.annotation.import_export import (
    _load_candidate_metadata_from_dir,
    build_candidate_tasks,
    select_candidate_pilot,
)
from pickup_putdown.annotation.schemas import VideoUrlMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source_metadata(
    source_video_id: str,
    candidates: list[dict[str, Any]],
    source_bucket: str | None = None,
    source_key: str | None = None,
) -> dict[str, Any]:
    """Build a source-level metadata dict matching Task 6.1 output format."""
    meta: dict[str, Any] = {
        "source_video_id": source_video_id,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if source_bucket:
        meta["source_bucket"] = source_bucket
    if source_key:
        meta["source_key"] = source_key
    return meta


def _make_candidate(
    candidate_id: str,
    candidate_key: str,
    source_start_s: float,
    source_end_s: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build a candidate record."""
    return {
        "candidate_id": candidate_id,
        "candidate_key": candidate_key,
        "source_start_s": source_start_s,
        "source_end_s": source_end_s,
        **extra,
    }


# ---------------------------------------------------------------------------
# Test 1: Recursive discovery of nested metadata files
# ---------------------------------------------------------------------------


class TestRecursiveDiscovery:
    def test_rglob_finds_nested_json(self, tmp_path: Path) -> None:
        subdir = tmp_path / "source_a"
        subdir.mkdir()
        meta = _make_source_metadata(
            "source_a",
            [_make_candidate("c1", "s3://b/k/c1.mp4", 0.0, 5.0)],
        )
        (subdir / "source_a.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.source_files_scanned == 1
        assert stats.candidates_loaded == 1
        assert candidates[0]["candidate_id"] == "c1"

    def test_deeply_nested_discovery(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        meta = _make_source_metadata(
            "deep_source",
            [_make_candidate("dc1", "s3://b/k/dc1.mp4", 0.0, 3.0)],
        )
        (deep / "deep.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.source_files_scanned == 1
        assert stats.candidates_loaded == 1


# ---------------------------------------------------------------------------
# Test 2: Loading multiple candidates from one source-level JSON
# ---------------------------------------------------------------------------


class TestMultipleCandidatesFromOneSource:
    def test_two_candidates_loaded(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "src_01",
            [
                _make_candidate("c1", "s3://b/c1.mp4", 0.0, 5.0),
                _make_candidate("c2", "s3://b/c2.mp4", 10.0, 15.0),
            ],
        )
        (tmp_path / "src_01.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 2
        ids = [c["candidate_id"] for c in candidates]
        assert ids == ["c1", "c2"]

    def test_many_candidates_from_one_source(self, tmp_path: Path) -> None:
        cands = [
            _make_candidate(f"c{i:03d}", f"s3://b/c{i:03d}.mp4", float(i), float(i + 5))
            for i in range(50)
        ]
        meta = _make_source_metadata("bulk_src", cands)
        (tmp_path / "bulk.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 50


# ---------------------------------------------------------------------------
# Test 3: Mapping source_video_id to clip_id
# ---------------------------------------------------------------------------


class TestSourceVideoIdToClipId:
    def test_clip_id_inherited(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "my_source_video",
            [_make_candidate("c1", "s3://b/c1.mp4", 0.0, 5.0)],
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        assert candidates[0]["clip_id"] == "my_source_video"

    def test_explicit_clip_id_not_overridden(self, tmp_path: Path) -> None:
        """Candidate-level clip_id takes precedence when explicitly present."""
        cand = _make_candidate("c1", "s3://b/c1.mp4", 0.0, 5.0)
        cand["clip_id"] = "explicit_clip"
        meta = _make_source_metadata("my_source", [cand])
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        assert candidates[0]["clip_id"] == "explicit_clip"


# ---------------------------------------------------------------------------
# Test 4: Inheriting source_bucket and source_key
# ---------------------------------------------------------------------------


class TestSourceMetadataInheritance:
    def test_source_bucket_inherited(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "src",
            [_make_candidate("c1", "s3://b/c1.mp4", 0.0, 5.0)],
            source_bucket="chillnbite-cameras",
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        assert candidates[0].get("source_bucket") == "chillnbite-cameras"

    def test_source_key_inherited(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "src",
            [_make_candidate("c1", "s3://b/c1.mp4", 0.0, 5.0)],
            source_key="anon/source/example.mp4",
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        assert candidates[0].get("source_key") == "anon/source/example.mp4"


# ---------------------------------------------------------------------------
# Test 5: Preserving candidate_key
# ---------------------------------------------------------------------------


class TestCandidateKeyPreserved:
    def test_candidate_key_preserved(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "src",
            [_make_candidate("c1", "s3://bucket/candidates/c1.mp4", 0.0, 5.0)],
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        assert candidates[0]["candidate_key"] == "s3://bucket/candidates/c1.mp4"


# ---------------------------------------------------------------------------
# Test 6: Ignoring candidate_count: 0
# ---------------------------------------------------------------------------


class TestZeroCandidateCount:
    def test_zero_candidate_count_skipped(self, tmp_path: Path) -> None:
        meta = _make_source_metadata("empty_src", [])
        (tmp_path / "empty.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.source_files_scanned == 1
        assert stats.zero_candidate_sources_skipped == 1
        assert stats.candidates_loaded == 0
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# Test 7: Ignoring empty candidates array
# ---------------------------------------------------------------------------


class TestEmptyCandidatesArray:
    def test_empty_candidates_array_skipped(self, tmp_path: Path) -> None:
        raw = {
            "source_video_id": "src_empty",
            "candidate_count": 0,
            "candidates": [],
        }
        (tmp_path / "e.json").write_text(json.dumps(raw))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.zero_candidate_sources_skipped == 1
        assert stats.candidates_loaded == 0

    def test_absent_candidates_key_flat_fallback(self, tmp_path: Path) -> None:
        """A dict without 'candidates' key is treated as flat candidate."""
        raw = {
            "candidate_id": "flat_c",
            "clip_id": "clip_x",
            "source_start_s": 0.0,
            "source_end_s": 5.0,
            "candidate_key": "s3://b/flat.mp4",
        }
        (tmp_path / "flat.json").write_text(json.dumps(raw))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 1
        assert candidates[0]["candidate_id"] == "flat_c"


# ---------------------------------------------------------------------------
# Test 8: Deterministic ordering across directories and files
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    def test_sorted_by_path_then_index(self, tmp_path: Path) -> None:
        # Create files in non-alphabetical order
        for name, cid in [("b_src", "b1"), ("a_src", "a1"), ("c_src", "c1")]:
            subdir = tmp_path / name
            subdir.mkdir()
            meta = _make_source_metadata(
                name,
                [_make_candidate(cid, f"s3://b/{cid}.mp4", 0.0, 5.0)],
            )
            (subdir / f"{name}.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        ids = [c["candidate_id"] for c in candidates]
        assert ids == ["a1", "b1", "c1"]

    def test_repeated_calls_same_order(self, tmp_path: Path) -> None:
        for i in range(3):
            subdir = tmp_path / f"src_{i}"
            subdir.mkdir()
            meta = _make_source_metadata(
                f"src_{i}",
                [
                    _make_candidate(f"c{i}_0", f"s3://b/{i}_0.mp4", 0.0, 2.0),
                    _make_candidate(f"c{i}_1", f"s3://b/{i}_1.mp4", 3.0, 5.0),
                ],
            )
            (subdir / f"{i}.json").write_text(json.dumps(meta))

        r1, _ = _load_candidate_metadata_from_dir(tmp_path)
        r2, _ = _load_candidate_metadata_from_dir(tmp_path)
        assert [c["candidate_id"] for c in r1] == [c["candidate_id"] for c in r2]


# ---------------------------------------------------------------------------
# Test 9: Supporting existing flat candidate format
# ---------------------------------------------------------------------------


class TestFlatCandidateFormat:
    def test_flat_dict_single(self, tmp_path: Path) -> None:
        raw = {
            "candidate_id": "flat_1",
            "clip_id": "clip_flat",
            "source_start_s": 0.0,
            "source_end_s": 10.0,
            "candidate_key": "s3://b/flat.mp4",
        }
        (tmp_path / "flat.json").write_text(json.dumps(raw))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 1
        assert candidates[0]["candidate_id"] == "flat_1"
        assert candidates[0]["clip_id"] == "clip_flat"

    def test_flat_array_in_file(self, tmp_path: Path) -> None:
        raw = [
            {
                "candidate_id": "a1",
                "clip_id": "clip_a",
                "source_start_s": 0.0,
                "source_end_s": 5.0,
                "candidate_key": "s3://b/a.mp4",
            },
            {
                "candidate_id": "a2",
                "clip_id": "clip_a",
                "source_start_s": 10.0,
                "source_end_s": 15.0,
                "candidate_key": "s3://b/a2.mp4",
            },
        ]
        (tmp_path / "array.json").write_text(json.dumps(raw))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 2

    def test_mixed_source_and_flat(self, tmp_path: Path) -> None:
        """Source-level and flat files coexist in same directory."""
        # Source-level
        meta = _make_source_metadata(
            "src_nested",
            [_make_candidate("n1", "s3://b/n1.mp4", 0.0, 5.0)],
        )
        (tmp_path / "01_nested.json").write_text(json.dumps(meta))

        # Flat
        flat = {
            "candidate_id": "f1",
            "clip_id": "clip_f",
            "source_start_s": 0.0,
            "source_end_s": 5.0,
            "candidate_key": "s3://b/f1.mp4",
        }
        (tmp_path / "02_flat.json").write_text(json.dumps(flat))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 2
        ids = [c["candidate_id"] for c in candidates]
        assert ids == ["n1", "f1"]


# ---------------------------------------------------------------------------
# Test 10: Missing source_video_id with non-empty candidates
# ---------------------------------------------------------------------------


class TestMissingSourceVideoId:
    def test_missing_source_video_id_error(self, tmp_path: Path) -> None:
        raw = {
            "candidate_count": 1,
            "candidates": [
                _make_candidate("c1", "s3://b/c1.mp4", 0.0, 5.0),
            ],
        }
        (tmp_path / "bad.json").write_text(json.dumps(raw))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 0
        assert len(stats.errors) == 1
        assert "source_video_id" in stats.errors[0]


# ---------------------------------------------------------------------------
# Test 11: Missing nested candidate_id
# ---------------------------------------------------------------------------


class TestMissingNestedCandidateId:
    def test_missing_candidate_id_error(self, tmp_path: Path) -> None:
        cand = {
            "candidate_key": "s3://b/c1.mp4",
            "source_start_s": 0.0,
            "source_end_s": 5.0,
        }
        meta = _make_source_metadata("src", [cand])
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 0
        assert len(stats.errors) == 1
        assert "candidate_id" in stats.errors[0]


# ---------------------------------------------------------------------------
# Test 12: Missing nested candidate_key
# ---------------------------------------------------------------------------


class TestMissingNestedCandidateKey:
    def test_missing_candidate_key_error(self, tmp_path: Path) -> None:
        cand = {
            "candidate_id": "c1",
            "source_start_s": 0.0,
            "source_end_s": 5.0,
        }
        meta = _make_source_metadata("src", [cand])
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 0
        assert len(stats.errors) == 1
        assert "candidate_key" in stats.errors[0]


# ---------------------------------------------------------------------------
# Test 13: Invalid .candidates type
# ---------------------------------------------------------------------------


class TestInvalidCandidatesType:
    def test_candidates_not_list_error(self, tmp_path: Path) -> None:
        raw = {
            "source_video_id": "src",
            "candidate_count": 1,
            "candidates": {"c1": "bad"},
        }
        (tmp_path / "s.json").write_text(json.dumps(raw))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 0
        assert len(stats.errors) == 1
        assert "must be a list" in stats.errors[0]


# ---------------------------------------------------------------------------
# Test 14: Duplicate candidate IDs
# ---------------------------------------------------------------------------


class TestDuplicateCandidateIds:
    def test_duplicate_in_same_source(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "src",
            [
                _make_candidate("dup", "s3://b/dup1.mp4", 0.0, 5.0),
                _make_candidate("dup", "s3://b/dup2.mp4", 10.0, 15.0),
            ],
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 1
        assert len(stats.errors) == 1
        assert "Duplicate" in stats.errors[0]

    def test_duplicate_across_sources(self, tmp_path: Path) -> None:
        for i in range(2):
            subdir = tmp_path / f"src_{i}"
            subdir.mkdir()
            meta = _make_source_metadata(
                f"src_{i}",
                [_make_candidate("same_id", f"s3://b/{i}.mp4", 0.0, 5.0)],
            )
            (subdir / f"{i}.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 1
        assert len(stats.errors) == 1
        assert "Duplicate" in stats.errors[0]


# ---------------------------------------------------------------------------
# Test 15: Pilot selection from flattened internal candidate collection
# ---------------------------------------------------------------------------


class TestPilotSelectionFromFlattened:
    def test_pilot_from_source_level_metadata(self, tmp_path: Path) -> None:
        for i in range(10):
            subdir = tmp_path / f"src_{i:02d}"
            subdir.mkdir()
            meta = _make_source_metadata(
                f"src_{i:02d}",
                [
                    _make_candidate(
                        f"cand_{i:02d}_0",
                        f"s3://b/{i:02d}_0.mp4",
                        0.0,
                        5.0,
                    ),
                    _make_candidate(
                        f"cand_{i:02d}_1",
                        f"s3://b/{i:02d}_1.mp4",
                        10.0,
                        15.0,
                    ),
                ],
            )
            (subdir / f"{i:02d}.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 20

        selected = select_candidate_pilot(candidates, limit=5, seed=42)
        assert len(selected) == 5
        # Deterministic
        selected2 = select_candidate_pilot(candidates, limit=5, seed=42)
        assert [c["candidate_id"] for c in selected] == [c["candidate_id"] for c in selected2]

    def test_pilot_respects_clip_id_inheritance(self, tmp_path: Path) -> None:
        meta = _make_source_metadata(
            "pilot_src",
            [
                _make_candidate(f"pc{i}", f"s3://b/pc{i}.mp4", float(i * 5), float(i * 5 + 3))
                for i in range(10)
            ],
        )
        (tmp_path / "p.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        selected = select_candidate_pilot(candidates, limit=3, seed=42)
        for c in selected:
            assert c["clip_id"] == "pilot_src"


# ---------------------------------------------------------------------------
# Test 16: End-to-end task generation from realistic Task 6.1 fixture
# ---------------------------------------------------------------------------


class TestEndToEndTaskGeneration:
    def test_full_pipeline_from_source_metadata(self, tmp_path: Path) -> None:
        """Realistic fixture: two source files, one with candidates, one empty."""
        # Source with candidates
        meta_with = _make_source_metadata(
            "D2_S20260526201736_E20260526201856_anon",
            [
                _make_candidate(
                    "cand_3f54241aeb53",
                    ".local/candidate_staging/candidates/D2_S20260526201736_E20260526201856_anon/cand_3f54241aeb53.mp4",
                    0.0,
                    3.75,
                    duration_s=3.75,
                    codec="h264",
                    pixel_format="yuv420p",
                ),
                _make_candidate(
                    "cand_4be6d4e2cd70",
                    ".local/candidate_staging/candidates/D2_S20260526201736_E20260526201856_anon/cand_4be6d4e2cd70.mp4",
                    0.0,
                    3.9,
                    duration_s=3.9,
                    codec="h264",
                    pixel_format="yuv420p",
                ),
            ],
            source_bucket="chillnbite-cameras",
            source_key="anon/source/example.mp4",
        )
        src_dir = tmp_path / "D2_S20260526201736_E20260526201856_anon"
        src_dir.mkdir()
        (src_dir / "D2_S20260526201736_E20260526201856_anon.json").write_text(
            json.dumps(meta_with)
        )

        # Source with zero candidates
        meta_empty = _make_source_metadata(
            "D2_S20260526122832_E20260526123358_anon",
            [],
        )
        empty_dir = tmp_path / "D2_S20260526122832_E20260526123358_anon"
        empty_dir.mkdir()
        (empty_dir / "D2_S20260526122832_E20260526123358_anon.json").write_text(
            json.dumps(meta_empty)
        )

        # Load
        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.source_files_scanned == 2
        assert stats.zero_candidate_sources_skipped == 1
        assert stats.candidates_loaded == 2
        assert len(stats.errors) == 0

        # clip_id inheritance
        for c in candidates:
            assert c["clip_id"] == "D2_S20260526201736_E20260526201856_anon"
            assert c.get("source_bucket") == "chillnbite-cameras"
            assert c.get("source_key") == "anon/source/example.mp4"

        # Build tasks with s3_key mode (no local files needed)
        tasks, errors = build_candidate_tasks(
            candidates,
            video_url_mode=VideoUrlMode.S3_KEY,
        )
        assert not errors
        assert len(tasks) == 2

        # Verify task data
        task_ids = sorted(t.data["candidate_id"] for t in tasks)
        assert task_ids == ["cand_3f54241aeb53", "cand_4be6d4e2cd70"]

        # clip_id in task data is source_video_id, not candidate_id
        for task in tasks:
            assert task.data["clip_id"] == "D2_S20260526201736_E20260526201856_anon"

        # Build tasks with s3_storage mode
        tasks_s3, errors_s3 = build_candidate_tasks(
            candidates,
            video_url_mode=VideoUrlMode.S3_STORAGE,
            s3_bucket="chillnbite-cameras",
        )
        assert not errors_s3
        assert len(tasks_s3) == 2
        # Video URL should be s3:// format
        for task in tasks_s3:
            assert task.data["video"].startswith("s3://chillnbite-cameras/")

    def test_s3_storage_no_local_files_needed(self, tmp_path: Path) -> None:
        """s3_storage mode should work without local video files."""
        meta = _make_source_metadata(
            "src_s3",
            [
                _make_candidate(
                    "s3c1",
                    "anon/candidates/videos/src_s3/s3c1.mp4",
                    0.0,
                    5.0,
                ),
            ],
            source_bucket="chillnbite-cameras",
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, _ = _load_candidate_metadata_from_dir(tmp_path)
        tasks, errors = build_candidate_tasks(
            candidates,
            video_url_mode=VideoUrlMode.S3_STORAGE,
            s3_bucket="chillnbite-cameras",
        )
        assert not errors
        assert len(tasks) == 1
        assert (
            tasks[0].data["video"]
            == "s3://chillnbite-cameras/anon/candidates/videos/src_s3/s3c1.mp4"
        )

    def test_malformed_candidate_fails_not_skipped(self, tmp_path: Path) -> None:
        """A malformed candidate produces an error, not silent skip."""
        meta = _make_source_metadata(
            "src_bad",
            [
                _make_candidate("good", "s3://b/good.mp4", 0.0, 5.0),
                {
                    "candidate_id": "bad_timing",
                    "candidate_key": "s3://b/bad.mp4",
                    "source_start_s": 10.0,
                    "source_end_s": 5.0,
                },
            ],
        )
        (tmp_path / "s.json").write_text(json.dumps(meta))

        candidates, stats = _load_candidate_metadata_from_dir(tmp_path)
        assert stats.candidates_loaded == 1
        assert len(stats.errors) == 1
        assert "bad_timing" in stats.errors[0]
        assert "invalid interval" in stats.errors[0].lower()
