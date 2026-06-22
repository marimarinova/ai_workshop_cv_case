"""Markdown + HTML reports, JSON serializer, and a failure gallery."""

from __future__ import annotations

import html
import json
from typing import Any

from .class_aware_matching import by_clip, drop_ignored, evaluate_class_aware, match_one_to_one
from .contracts import VALID_TYPES, type_name
from .intervals import Criterion


def _fmt(v: object) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def render_markdown(metrics: dict[str, Any], model_name: str = "model") -> str:
    """Render an aggregate_metrics dict as markdown."""
    safe_name = html.escape(str(model_name))
    lines = [f"# Evaluation report - {safe_name}", ""]
    for k, v in metrics.items():
        if k == "confusion":
            lines += [
                "## Type confusion",
                "| gt \\ pred | " + " | ".join(VALID_TYPES) + " |",
                "|" + "---|" * (len(VALID_TYPES) + 1),
            ]
            for a in VALID_TYPES:
                lines.append(f"| {a} | " + " | ".join(str(v[a][b]) for b in VALID_TYPES) + " |")
        elif isinstance(v, dict):
            lines.append(f"## {k}")
            lines.append(
                ", ".join(f"{kk}={_fmt(vv)}" for kk, vv in v.items() if not isinstance(vv, dict))
            )
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    lines.append(
                        f"- {kk}: " + ", ".join(f"{k2}={_fmt(v2)}" for k2, v2 in vv.items())
                    )
        else:
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    return "\n".join(lines)


def render_html(metrics: dict[str, Any], model_name: str = "model") -> str:
    """Render an aggregate_metrics dict as standalone HTML (values escaped)."""
    esc = html.escape
    parts = [f"<h1>Evaluation report &mdash; {esc(str(model_name))}</h1>"]
    for k, v in metrics.items():
        if k == "confusion":
            head = "".join(f"<th>{esc(b)}</th>" for b in VALID_TYPES)
            parts.append(
                f"<h2>Type confusion</h2><table border='1' cellpadding='4'><tr><th>gt \\ pred</th>{head}</tr>"
            )
            for a in VALID_TYPES:
                cells = "".join(f"<td>{v[a][b]}</td>" for b in VALID_TYPES)
                parts.append(f"<tr><th>{esc(a)}</th>{cells}</tr>")
            parts.append("</table>")
        elif isinstance(v, dict):
            parts.append(f"<h2>{esc(k)}</h2><ul>")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    inner = ", ".join(f"{esc(str(k2))}={esc(_fmt(v2))}" for k2, v2 in vv.items())
                    parts.append(f"<li>{esc(str(kk))}: {inner}</li>")
                else:
                    parts.append(f"<li>{esc(str(kk))}: {esc(_fmt(vv))}</li>")
            parts.append("</ul>")
        else:
            parts.append(f"<p><b>{esc(str(k))}</b>: {esc(str(v))}</p>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Evaluation report - {esc(str(model_name))}</title></head><body>"
        + "".join(parts)
        + "</body></html>"
    )


def metrics_to_json(metrics: dict[str, Any], **kw: Any) -> str:
    """Serialize a metrics dict to a JSON string (all keys are strings)."""
    return json.dumps(metrics, **kw)


def failure_gallery(
    events: Any, preds: Any, criterion: Criterion | None = None, ignores: Any = ()
) -> dict[str, Any]:
    """False positives, false negatives, and type-confusion pairs for review."""
    criterion = criterion or Criterion("tiou", 0.5)
    r = evaluate_class_aware(events, preds, criterion, ignores)
    # type confusions: temporal (type-agnostic) matches whose types differ
    ev = drop_ignored(events, ignores)
    pr = drop_ignored(preds, ignores)
    ge, gp = by_clip(ev), by_clip(pr)
    type_confusions: list[dict[str, Any]] = []
    for clip in set(ge) | set(gp):
        m = match_one_to_one(ge.get(clip, []), gp.get(clip, []), criterion)
        for g, p in m.matched:
            if type_name(g.type) != type_name(p.type):
                type_confusions.append(
                    {
                        "clip_id": g.clip_id,
                        "gt_type": type_name(g.type),
                        "pred_type": type_name(p.type),
                        "t_start": g.t_start,
                        "t_end": g.t_end,
                    }
                )
    return {
        "false_positives": [
            {
                "clip_id": p.clip_id,
                "type": type_name(p.type),
                "t_start": p.t_start,
                "t_end": p.t_end,
                "score": getattr(p, "score", None),
            }
            for p in r.unmatched_pred
        ],
        "false_negatives": [
            {
                "clip_id": g.clip_id,
                "type": type_name(g.type),
                "t_start": g.t_start,
                "t_end": g.t_end,
                "hard_case": getattr(g, "hard_case", None),
            }
            for g in r.unmatched_gt
        ],
        "type_confusions": type_confusions,
    }
