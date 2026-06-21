Overall Assessment: 6.5/10
Solid architecture with good separation of concerns, but several quality issues need attention.
Strengths
- Clean modular structure — common/, ingestion/, perception/ are well-separated concerns
- Pydantic schemas — canonical data models with validators provide good type safety
- Good test coverage — 11 test files covering schemas, cache, registry, config, shelf regions, person tracker, previews, active spans, video probe
- Configuration system — YAML + env-var overrides with deep merge is well-designed
- Deterministic IDs — SHA-256 based clip IDs ensure reproducibility
- Logging — consistent structured logging throughout
- Cache with eviction — bounded download cache with file locking is well-implemented
Critical Issues
1. cli.py — 713 lines, massive function (lines 322-651)
The triage command is a ~330-line monolithic function that does too much:
- Video resolution
- Probe metadata
- Run person tracker
- Derive active spans
- Render previews
- Write parquet files
- Generate sampling reports
- Write run metadata
Fix: Extract into a TriagePipeline class or separate functions (run_triage_for_video, write_triage_outputs, etc.)
2. cli.py — triage_comparison uses placeholder values (lines 69-70, 86)
clip_duration_s=cfg_1.target_fps * 100,  # placeholder, corrected below
This is a bug — target_fps * 100 is not the actual clip duration. The comparison results are unreliable.
3. cli.py — main() is a bare wrapper (lines 280-282)
def main() -> None:
    app()
Redundant — app() is already the entry point. The pyproject.toml script points to main() but it adds nothing.
4. cli.py — @app.command() decorators on non-function objects (lines 285-713)
The triage and triage_comparison functions are defined after the if __name__ == "__main__" block and after main(), which is structurally confusing. The TriageCommand class and _resolve_video_paths function are interleaved.
Medium Issues
5. schemas.py — Repeated validator pattern (lines 70-84, 103-117, 141-148, 165-179)
The t_end_after_start validator uses hasattr(info, "data") and info.data.get("t_start") — this is a fragile Pydantic v2 pattern that may not work reliably across versions. A custom root validator or model_validator would be more robust.
6. schemas.py — PersonObservation and TrackSummary are "internal" but exported (lines 214-257)
These live in the public schema module but are only used internally by perception/. They should either be in a separate internal module or clearly documented as part of the public API.
7. cache.py — _download_fn type is Any (line 197)
_download_fn: Any = None
Should use Callable[[str, Path], None] | None for type safety.
8. cache.py — Lock file not cleaned up on exception (lines 258-268)
If _download raises between _acquire_lock and _release_lock, the finally block releases it. But if _release_lock itself fails, the lock file handle may leak. The _lock_file attribute is set inside _acquire_lock but checked in _release_lock — if _acquire_lock fails partway, _release_lock may crash on hasattr check.
9. active_spans.py — _Interval is mutable dataclass (lines 13-25)
The merge logic mutates last.t_start and last.t_end in-place. This is subtle and could be confusing. Consider returning new objects instead.
10. comparison.py — _clone_cfg is verbose (lines 124-143)
Manually copying all 15 fields. Pydantic's .model_copy() would be cleaner:
return cfg.model_copy(update={"target_fps": target_fps})
11. perception/__init__.py is empty
No __all__ exports defined. Callers import from deep paths (from pickup_putdown.perception.active_spans import ...) which is fine but the module doesn't signal its public API.
12. video_probe.py — _validate_decode always runs even when ffprobe fails (line 269)
Actually, it's only called after ffprobe succeeds, so this is correct. But the function signature takes ffprobe: str when it could just look it up again — minor inconsistency.
Minor Issues
13. cli.py — import re inside function (line 115)
Should be at module level.
14. cli.py — import hashlib, json, subprocess inside triage (lines 357-359)
Should be at module level.
15. cli.py — import pyarrow inside triage (line 362)
Should be at module level.
16. config.py — _build_env_overrides section whitelist is hardcoded (lines 98-107)
The list of valid sections (storage, triage, tracker, proposals, data, output, cache, results) doesn't match AppConfig fields exactly (missing results). If a new section is added to AppConfig, it must be added here too.
17. config.py — AppConfig has string fields for paths (data_dir, output_dir, etc.) (lines 65-68)
These should be Path types or at least validated.
18. shelf_regions.py — _all_collinear uses magic epsilon 1e-6 (line 104)
Should be a named constant.
19. previews.py — _open_video_writer doesn't close writer on failure (line 583)
If cv2.VideoWriter opens but subsequent write fails, the writer is released in the finally block of render_triage_preview, so this is actually fine.
20. sampling_report.py — observations_to_report_rows is unused (lines 160-183)
This function exists but is never called from anywhere in the codebase. Dead code.
21. common/run_metadata.py — RunMetadata model exists but is barely used
Only referenced in triage command's metadata JSON (lines 626-639), but the JSON is manually constructed rather than using the model.
22. exceptions.py — get_exit_code and exit_with_error are defined but never used
No code in the repo calls these. They're dead code.
23. cli.py — _s3_download_fn reloads config every call (line 257)
cfg = load_config()
This is called once per download in the ingest loop, so it's not a performance issue, but it's wasteful. Config should be loaded once and passed in.
24. perception/person_tracker.py — _model stored as object (line 37)
Should use YOLO | None for better type hints. The lazy import of ultralytics.YOLO makes this harder but TYPE_CHECKING block would help.
25. perception/person_tracker.py — run() creates clip_id from filename (line 273)
f"clip_{stem}" — this is fragile. If the same video file appears in different directories, clip IDs collide.
26. tests/ — Missing tests for cli.py, config.py (partial), video_probe.py, active_spans.py, comparison.py, sampling_report.py, shelf_regions.py
test_config.py exists but test_video_probe.py likely tests only happy paths. No tests for the CLI commands or the comparison module.
27. pyproject.toml — mypy strict mode but many Any usages
The strict = true setting conflicts with widespread Any types in cache.py, clip_registry.py, and elsewhere. This will cause mypy failures.
28. AGENTS.md — References CCE tools that may not be available
The agent instructions reference context_search, session_recall, etc. which are tool-specific. This is fine for this environment but worth noting.
Refactoring Priorities
Priority	Issue
P0	comparison.py placeholder clip_duration_s
P0	cli.py triage — extract pipeline logic
P1	Dead code: observations_to_report_rows, get_exit_code, exit_with_error
P1	schemas.py validator fragility
P1	comparison.py — use model_copy()
P2	Module-level imports in cli.py
P2	cache.py — type Any for _download_fn
P2	Missing tests for cli.py, comparison.py, active_spans.py
P3	config.py — section whitelist sync
P3	perception/__init__.py — add __all__
P3	run_metadata.py — use model instead of manual dict
Summary
The codebase is well-structured and mostly correct with good separation of concerns and solid testing for core data models. The biggest issues are:
1. A real bug in comparison.py using placeholder values for clip duration
2. One massive CLI function (~330 lines) that needs extraction
3. Several pieces of dead code that should be removed
4. Type safety gaps that conflict with the strict mypy config
The architecture is sound — the main work is refactoring the CLI, fixing the comparison bug, and cleaning up dead code.