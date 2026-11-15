"""Per-stage JSONL checkpoints and the merged ``all_metadata.json.gz``."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Latest entry per image name."""
    entries: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                entries[entry["name"]] = entry
    return entries


def load_generated(run_path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(run_path / "generation_metadata.jsonl")
    if not records:
        raise SystemExit(f"No generation_metadata.jsonl in {run_path}. Run the generation step first.")
    return sorted(records.values(), key=lambda r: (r.get("mode", 0), r.get("index", 0)))


def merge(records: list[dict[str, Any]], run_path: Path, stage_names: list[str]) -> list[dict[str, Any]]:
    """Combine generation records and every stage checkpoint into final records."""
    checkpoints = {name: read_jsonl(run_path / "extraction" / f"{name}.jsonl") for name in stage_names}
    merged = []
    for record in records:
        final = dict(record)
        for entries in checkpoints.values():
            entry = entries.get(record["name"])
            if entry and "fields" in entry:
                final.update(entry["fields"])
        merged.append(final)
    return merged


def write_gzip_json(data: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as file:
        json.dump(data, file, separators=(",", ":"))
    os.replace(temporary, path)
