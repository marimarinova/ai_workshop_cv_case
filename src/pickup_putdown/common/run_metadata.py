"""Run metadata for reproducibility."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    git_commit: str = ""
    dataset_version: str = ""
    split_version: str = ""
    config: str = ""
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    model_identifier: str = ""
    checkpoint_hash: str = ""
    timestamp: str = Field(default_factory=lambda: _now_iso())

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()
