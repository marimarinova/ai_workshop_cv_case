# Candidate Generation — Process All Report

**Generated:** 2026-06-23T16:30Z

**Status: NOT RUNNING** — No `pickup-putdown` or `process_all` processes active. The script started **Run 3/10** at `16:14:14Z` but the process is no longer running (likely killed or crashed mid-run).

## Ledger Summary

| State | Count |
|---|---|
| Total entries | 88 |
| Completed (generated=true) | 75 |
| Ready for processing | 13 |
| Skipped (permanent fail) | 21 |

## Current Session

- **Run ID:** `local_1c19d3c8d68b`
- **Started:** 14:47:40Z
- **Current run:** 3 of 10
- **Videos selected this run:** 13

| Metric | Value |
|---|---|
| Cumulative completed | 28 (15 + 13) |
| Cumulative failed | 58 (42 + 16) |
| Cumulative candidates | 628 (453 + 175) |

## 13 Remaining Videos

All are longer clips from May 20–26. Two are in the failed-JSON retry tracker:

| Video | Status |
|---|---|
| `D2_S20260520165727_E20260520171509_anon` | Skipped permanently (3+ retries) |
| `D2_S20260522010750_E20260522011738_anon` | 1 retry in failed JSON |
| `D2_S20260522011748_E20260522013928_anon` | Timed out (3600s triage) |
| `D2_S20260522103722_E20260522103914_anon` | Worker timeout |
| `D2_S20260522152016_E20260522152704_anon` | Worker timeout |
| `D2_S20260522160804_E20260522161302_anon` | Worker timeout |
| `D2_S20260522161302_E20260522161916_anon` | Worker timeout |
| `D2_S20260522163920_E20260522164344_anon` | Worker timeout |
| `D2_S20260522180722_E20260522181742_anon` | Skipped permanently |
| `D2_S20260522182654_E20260522184754_anon` | Skipped permanently |
| `D2_S20260526120450_E20260526121058_anon` | Worker timeout |
| `D2_S20260526121058_E20260526122012_anon` | Skipped permanently |
| `D2_S20260526122012_E20260526122832_anon` | Skipped permanently |

## Failure Pattern

The 13 remaining videos share a consistent failure mode: **Task 3 worker timeout** (`RuntimeError: Worker 0/1 error: Worker 0/1: timeout`). These are all longer clips from May 22–26. The frame queue worker times out waiting for frames during the triage pipeline.

## Session History (Jun 23)

| Session | Start | Videos | Completed | Failed | Candidates | End reason |
|---|---|---|---|---|---|---|
| 1 | 05:13 | 11 | 0 | 0 | 0 | All 10 runs returned 0 (command-level issue) |
| 2 | 05:26 | 8→10 | 0 | 0 | 0 | Same — 0 across all runs |
| 3 | 05:28 | 12 | 12 | 10 | 323 | STUCK after run 3 |
| 4 | 06:41 | 21 | 11 | 48 | 137 | STUCK after run 3 |
| 5 | 07:57 | 9 | 0 | 18 | 0 | STUCK after run 1 |
| 6 | 08:37 | 34 | 8 | 52 | 85 | Stopped after run 1 |
| 7 | 11:39 | 20 | — | — | — | Stopped after run 1 start |
| 8 (current) | 12:33 | 41 | 28 | 58 | 628 | Process stopped during run 3 |

## Summary

Processing produced **75 completed videos** and **628 candidates** across the session. The 13 remaining videos are all hitting worker timeouts during triage. The process is **not currently running** — it stopped partway through run 3. The skip file has grown to 21 permanently-failed videos across all sessions. The remaining 13 need either the worker timeout issue fixed or manual removal from the skip list to retry.
