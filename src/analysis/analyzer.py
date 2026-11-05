"""Compute the report data and persist it as a versioned JSON artifact.

The artifact holds one ``report_data["chapter_N"]`` payload per report
chapter, plus per-chapter build metadata. The first run (or any run without a
chapter selection) builds every chapter. Later runs can rebuild only some
chapters and keep the rest from the existing artifact:

    python -m src.analysis.analyzer                  # build everything
    python -m src.analysis.analyzer --chapters 3 7   # rebuild chapters 3 and 7
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .chapters import CHAPTER_BUILDERS, CHAPTER_TITLES, normalize_chapter
from .preprocessing import preprocess

SCHEMA_VERSION = 2
# Config keys that only control how the analysis runs, not what it computes.
RUNTIME_CONFIG_KEYS = frozenset({"verbose"})


def _stored_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in RUNTIME_CONFIG_KEYS}


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()[:16]


def _data_fingerprint(data: list[dict[str, Any]]) -> str:
    """Cheap identity of the dataset: which images, prompts and modes it holds."""
    return _fingerprint([
        (record.get("name"), record.get("prompt"), record.get("mode"))
        for record in data if isinstance(record, dict)
    ])


def _select_chapters(chapters: Iterable[str | int] | None) -> list[str]:
    if chapters is None:
        return list(CHAPTER_BUILDERS)
    selected = {normalize_chapter(chapter) for chapter in chapters}
    return [chapter for chapter in CHAPTER_BUILDERS if chapter in selected]


def _usable_previous(previous: dict[str, Any] | None) -> dict[str, Any] | None:
    if previous is None:
        return None
    if previous.get("schema_version") != SCHEMA_VERSION:
        warnings.warn(
            f"Existing artifact has schema version {previous.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}; rebuilding every chapter.",
            stacklevel=3,
        )
        return None
    if not isinstance(previous.get("report_data"), dict):
        return None
    return previous


def analyze(
    data: list[dict[str, Any]],
    config: dict[str, Any],
    chapters: Iterable[str | int] | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute report data and return a JSON-serializable artifact.

    Args:
        data: Raw image metadata records.
        config: Analysis configuration. ``verbose`` controls progress output
            and is not stored in the artifact.
        chapters: Chapters to (re)build, e.g. ``[3, "chapter_5"]``. ``None``
            builds every chapter.
        previous: An existing artifact to update. Chapters that are not
            selected are copied from it; chapters it lacks are always built.
            Without a usable previous artifact every chapter is built.
    """
    verbose = int(config.get("verbose", 0) or 0)
    stored_config = _stored_config(config)
    previous = _usable_previous(previous)
    previous_data = previous["report_data"] if previous else {}
    previous_meta = previous.get("chapters", {}) if previous else {}

    selected = _select_chapters(chapters)
    to_build = [
        chapter for chapter in CHAPTER_BUILDERS
        if chapter in selected or chapter not in previous_data
    ]
    if verbose > 0:
        print(f"Building {len(to_build)}/{len(CHAPTER_BUILDERS)} chapters: "
              + ", ".join(to_build))

    start = time.perf_counter()
    preprocessed_data = preprocess(data)
    if verbose > 1:
        print(f"  preprocessing: {time.perf_counter() - start:.1f}s")

    config_fingerprint = _fingerprint(stored_config)
    data_fingerprint = _data_fingerprint(data)
    report_data: dict[str, Any] = {}
    chapter_meta: dict[str, Any] = {}
    for chapter in CHAPTER_BUILDERS:
        if chapter not in to_build:
            report_data[chapter] = previous_data[chapter]
            chapter_meta[chapter] = previous_meta.get(chapter, {})
            continue
        start = time.perf_counter()
        report_data[chapter] = CHAPTER_BUILDERS[chapter](preprocessed_data, config)
        duration = time.perf_counter() - start
        chapter_meta[chapter] = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_seconds": round(duration, 2),
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
        }
        if verbose > 0:
            print(f"  {chapter} ({CHAPTER_TITLES[chapter]}): {duration:.1f}s")

    stale = [
        chapter for chapter, meta in chapter_meta.items()
        if chapter not in to_build and (
            meta.get("config_fingerprint") != config_fingerprint
            or meta.get("data_fingerprint") != data_fingerprint
        )
    ]
    if stale:
        warnings.warn(
            "These chapters were kept from the previous artifact but were built "
            "from a different config or dataset: " + ", ".join(stale)
            + ". Rebuild them, or run without --chapters to rebuild everything.",
            stacklevel=2,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "config": stored_config,
        "chapters": chapter_meta,
        "report_data": report_data,
    }


def load_analysis(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_analysis(
    data: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: str | Path,
    chapters: Iterable[str | int] | None = None,
) -> Path:
    """Compute the analysis and persist it for a later report build.

    With ``chapters=None`` every chapter is built. With a chapter selection the
    existing artifact at ``output_path`` is updated in place: only the selected
    chapters (and any it is missing) are rebuilt. If no compatible artifact
    exists yet, every chapter is built.
    """
    path = Path(output_path)
    previous = None
    if chapters is not None:
        if path.exists():
            previous = load_analysis(path)
        elif int(config.get("verbose", 0) or 0) > 0:
            print(f"No artifact at {path}; building every chapter.")
    artifact = analyze(data, config, chapters=chapters, previous=previous)

    # Serialize fully before touching the file, and replace it atomically, so a
    # failure never leaves a truncated or half-updated artifact behind.
    text = json.dumps(artifact, ensure_ascii=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute the fairness-report data artifact.",
    )
    parser.add_argument(
        "--chapters", "-c", nargs="+", metavar="N",
        help="Rebuild only these chapters (e.g. 3 5 7) and keep the others "
             "from the existing artifact. Default: rebuild every chapter.",
    )
    parser.add_argument("--config", default="assets/config.json")
    parser.add_argument("--data", default="assets/all_metadata.json.gz")
    parser.add_argument("--output", default="assets/analysis.json")
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--list", action="store_true",
                        help="List chapters and when they were last built, then exit.")
    args = parser.parse_args(argv)

    if args.list:
        meta = load_analysis(args.output).get("chapters", {}) if Path(args.output).exists() else {}
        for chapter, title in CHAPTER_TITLES.items():
            info = meta.get(chapter, {})
            built = info.get("generated_at", "never")
            took = f" ({info['duration_seconds']}s)" if "duration_seconds" in info else ""
            print(f"{chapter.split('_')[1]:>2}  {title:<45} {built}{took}")
        return

    chapters = args.chapters
    if chapters is not None:
        # Validate before loading the (large) dataset.
        try:
            chapters = _select_chapters(chapters)
        except ValueError as error:
            parser.error(str(error))

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["verbose"] = args.verbose
    data = load_metadata(args.data)
    start = time.perf_counter()
    path = save_analysis(data, config, args.output, chapters=chapters)
    print(f"Wrote {path} in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
