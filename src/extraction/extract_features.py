"""Step 3 - extract the features needed by the analysis and write ONE file.

Heavy step: meant to run on a GPU, e.g. in the same Colab runtime that
generated the images::

    !pip install -r requirements-colab.txt
    !python -m src.extraction.extract_features --run /content/drive/MyDrive/ethics/PixArtSigma

Input  (``--run``, produced by ``src.generation.generate_images``)::

    generation_metadata.jsonl, images/*.png, config.json

Output (same folder)::

    all_metadata.json.gz         one record per image with every extracted feature
    config.json                  updated while the run progresses (stages, models, features)
    extraction/<stage>.jsonl     per-stage checkpoints (what makes the run resumable)

The analysis and report steps need nothing else, so they can run locally::

    python -m src.analysis.analyzer --config <run>/config.json --data <run>/all_metadata.json.gz

Features are extracted by independent *stages*, each loading one model at a
time so a Colab GPU is enough. Every stage checkpoints after each image: after a
disconnection, run the same command again and it continues where it stopped.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

from ..run_config import RunConfig, now_iso, setup_logging
from .checkpoints import load_generated, merge, read_jsonl, write_gzip_json
from .stages import STAGES, Stage


log = logging.getLogger("extraction")


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_stage(stage: Stage, records: list[dict[str, Any]], run_path: Path,
              config: RunConfig) -> None:
    from PIL import Image

    checkpoint = run_path / "extraction" / f"{stage.name}.jsonl"
    checkpoint.parent.mkdir(exist_ok=True)
    finished = {name for name, entry in read_jsonl(checkpoint).items() if not entry.get("error")}
    usable = [r for r in records if not r.get("generation_failed")]
    todo = [r for r in usable if r["name"] not in finished]

    state = {"model": stage.model, "features": stage.features, "settings": stage.settings,
             "total": len(usable), "done": len(usable) - len(todo), "failed": 0,
             "status": "running", "started_at": now_iso()}
    config.section("extraction").setdefault("stages", {})[stage.name] = state
    config.save()
    log.info("[%s] %d to process (%d already done)", stage.name, len(todo), len(finished))

    if todo:
        stage.load()
    try:
        with checkpoint.open("a", encoding="utf-8") as out:
            for position, record in enumerate(todo, 1):
                entry: dict[str, Any] = {"name": record["name"]}
                try:
                    with Image.open(run_path / record["image"]) as image:
                        image.load()
                        entry["fields"] = stage.process(image, record)
                    state["done"] += 1
                except Exception as error:  # keep going; failed images are retried on resume
                    log.warning("[%s] %s failed: %s", stage.name, record["name"], error)
                    entry["error"] = f"{type(error).__name__}: {error}"
                    state["failed"] += 1
                out.write(json.dumps(entry) + "\n")
                out.flush()
                if position % 25 == 0:
                    log.info("[%s] %d/%d", stage.name, position, len(todo))
                    config.save(force=False)
    finally:
        stage.unload()
        free_memory()

    state["status"] = "completed" if not state["failed"] else "completed_with_errors"
    state["finished_at"] = now_iso()
    config.save()


def refresh_config(config: RunConfig) -> None:
    """Derive the declared features/models from the stages that have completed."""
    stages = config.section("extraction").get("stages", {})
    done = [state for state in stages.values() if state["status"].startswith("completed")]
    config.update(
        extracted_feats=[feat for state in done for feat in state["features"]],
        extraction_models=[state["model"] for state in done if state["model"]],
    )


def run(run_dir: str | Path, stage_names: list[str] | None = None, merge_only: bool = False) -> Path:
    run_path = Path(run_dir)
    config = RunConfig(run_path / "config.json")
    records = load_generated(run_path)
    stage_names = stage_names or list(STAGES)

    extraction = config.section("extraction")
    extraction.setdefault("started_at", now_iso())
    extraction.update(status="running", resumed_at=now_iso())
    config.save()

    if not merge_only:
        for name in stage_names:
            run_stage(STAGES[name](), records, run_path, config)
            refresh_config(config)

    output = run_path / "all_metadata.json.gz"
    # Merge every stage that has a checkpoint, not just those run now.
    available = [name for name in STAGES if (run_path / "extraction" / f"{name}.jsonl").exists()]
    write_gzip_json(merge(records, run_path, available), output)
    config.update_section(
        "extraction", status="completed", finished_at=now_iso(),
        output=output.name, num_records=len(records), merged_stages=available,
    )
    log.info("Wrote %s (%d records)", output, len(records))
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract image features into a single all_metadata.json.gz.")
    parser.add_argument("--run", "-r", required=True,
                        help="Folder produced by the generation step (images, metadata, config.json).")
    parser.add_argument("--stages", "-s", nargs="+", choices=list(STAGES), default=None,
                        help=f"Stages to run (default: all). Available: {', '.join(STAGES)}.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Do not run any model, only rebuild all_metadata.json.gz from the checkpoints.")
    args = parser.parse_args(argv)
    setup_logging()
    run(args.run, args.stages, args.merge_only)


if __name__ == "__main__":
    main()
