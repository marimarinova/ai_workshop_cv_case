# Post-VLM Annotation Steps

Run these after `pickup-putdown annotate-vlm` completes.

## 1. Finalize Task 7

```bash
pickup-putdown finalize-task-7   --vlm-output-dir .local/vlm_annotations   --candidate-metadata-dir .local/candidate_staging/candidates   --source-videos-dir .local/source_videos   --output-dir .local/task_7_vlm
```

Check the outputs:

```bash
jq . .local/task_7_vlm/summary.json
wc -l .local/task_7_vlm/{clips.csv,events.csv,processing.csv}
```

## 2. Create the manual-review list

This selects all VLM-positive candidates, all deduplication candidates, and 10 deterministic negative samples.

```bash
mkdir -p .local/task_7_review

python - <<'PY'
import csv
import json
import random
from pathlib import Path

normalized_dir = Path(".local/vlm_annotations/normalized")
dedup_path = Path(".local/task_7_vlm/dedup_audit.json")
output_path = Path(".local/task_7_review/review_manifest.csv")

records = {}

for path in sorted(normalized_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    candidate_id = data["candidate_id"]

    records[candidate_id] = {
        "candidate_id": candidate_id,
        "clip_id": data.get("clip_id", ""),
        "review_groups": set(),
        "video_path": data.get("video_path", ""),
        "json_path": str(path),
        "event_count": len(data.get("events", [])),
        "reviewed": "false",
        "review_notes": "",
    }

for record in records.values():
    if record["event_count"] > 0:
        record["review_groups"].add("vlm_positive")


def candidate_ids(value):
    if isinstance(value, dict):
        if value.get("candidate_id"):
            yield value["candidate_id"]
        for child in value.values():
            yield from candidate_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from candidate_ids(child)


if dedup_path.exists():
    audit = json.loads(dedup_path.read_text(encoding="utf-8"))
    for candidate_id in set(candidate_ids(audit)):
        if candidate_id in records:
            records[candidate_id]["review_groups"].add("dedup_overlap")

negative_ids = sorted(
    candidate_id
    for candidate_id, record in records.items()
    if record["event_count"] == 0
)
random.Random(42).shuffle(negative_ids)

for candidate_id in negative_ids[:10]:
    records[candidate_id]["review_groups"].add("negative_sample")

selected = [
    record for record in records.values() if record["review_groups"]
]
selected.sort(
    key=lambda record: (
        record["event_count"] == 0,
        record["clip_id"],
        record["candidate_id"],
    )
)

columns = [
    "candidate_id",
    "clip_id",
    "review_groups",
    "video_path",
    "json_path",
    "event_count",
    "reviewed",
    "review_notes",
]

with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()

    for record in selected:
        row = dict(record)
        row["review_groups"] = ";".join(sorted(record["review_groups"]))
        writer.writerow(row)

print(f"Wrote {len(selected)} candidates to {output_path}")
PY
```

Open each video, edit its normalized JSON, and mark the row as reviewed:

```bash
mpv --osd-level=3 --osd-fractions <video_path>
```

After review, rerun `finalize-task-7` to regenerate the canonical Task 7 outputs.

