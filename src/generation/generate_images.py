"""Step 2 - generate the images (and their metadata) from a list of prompts.

Heavy step: meant to run on a GPU, e.g. in a Colab runtime::

    !git clone <this repo> && cd progEthics
    !pip install -U -r requirements-colab-generation.txt
    !python -m src.generation.generate_images --model pixart-sigma --out /content/drive/MyDrive/ethics/PixArtSigma

Output, inside ``--out``::

    images/<name>.png            one image per prompt
    generation_metadata.jsonl    one record per image: name, prompt, mode, seed, settings...
    config.json                  run configuration, updated while the run progresses

The run is resumable: re-running the same command skips the images that already
exist, so a disconnected Colab session loses nothing.

Auditing another model
----------------------
Only the *model adapter* changes between audits. Each model has its own file in
``src/generation/models/`` (sd15, pixart_sigma, flux_schnell, flux2_klein), and
``models/__init__.py`` registers them in ``MODELS``. To audit a new model either
add a file there defining a ``DiffusersGenerator`` (any diffusers text-to-image
pipeline) and register it, or subclass ``ImageGenerator`` for a non-diffusers
model and pass it with ``--model my_module:MyGenerator``. The rest of the pipeline (prompts,
metadata, extraction, analysis, report) is model-agnostic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from ..run_config import PROMPT_MODES, RunConfig, environment_info, now_iso, setup_logging
from .models import MODELS, DiffusersGenerator, ImageGenerator  # noqa: F401 (re-exported)

log = logging.getLogger("generation")

DEFAULT_PROMPT_FILES = {
    0: "assets/control_prompts.txt",
    1: "assets/single_var_prompts.txt",
    2: "assets/double_var_prompts.txt",
}


# --------------------------------------------------------------------------
# Model selection - the adapters live in ./models, one file per model
# --------------------------------------------------------------------------
def resolve_model(spec: str) -> ImageGenerator:
    """A registry key, or ``package.module:Name`` (an ImageGenerator class or instance)."""
    if spec in MODELS:
        return MODELS[spec]
    if ":" in spec:
        module_name, attribute = spec.split(":", 1)
        target = getattr(importlib.import_module(module_name), attribute)
        return target() if isinstance(target, type) else target
    raise SystemExit(f"Unknown model {spec!r}. Known: {', '.join(MODELS)} or 'module:Class'.")


# --------------------------------------------------------------------------
# Prompts and resumable metadata
# --------------------------------------------------------------------------
def read_prompts(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_jobs(prompt_files: dict[int, str]) -> list[dict[str, Any]]:
    """One job per image to generate: ``{name, prompt, mode, index, global_index}``."""
    jobs = []
    for mode, path in sorted(prompt_files.items()):
        for index, prompt in enumerate(read_prompts(path)):
            jobs.append({"name": f"m{mode}_{index}", "prompt": prompt, "mode": mode,
                         "index": index, "global_index": len(jobs)})
    return jobs


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per image name from the append-only metadata file."""
    records: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["name"]] = record
    return records


def parse_prompt_files(entries: list[str] | None) -> dict[int, str]:
    if not entries:
        return {mode: path for mode, path in DEFAULT_PROMPT_FILES.items() if Path(path).exists()}
    files = {}
    for entry in entries:
        mode, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--prompts expects MODE=PATH, got {entry!r}")
        files[int(mode)] = path
    return files


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run(
    generator: ImageGenerator,
    prompt_files: dict[int, str],
    out_dir: str | Path,
    seed: int = 0,
    batch_size: int = 1,
    limit: int | None = None,
) -> Path:
    out = Path(out_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "generation_metadata.jsonl"

    config = RunConfig(out / "config.json")
    jobs = build_jobs(prompt_files)
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        raise SystemExit("No prompts found. Check --prompts / the assets/*_prompts.txt files.")

    existing = read_records(metadata_path)
    todo = [
        job for job in jobs
        if not (job["name"] in existing and not existing[job["name"]].get("generation_failed")
                and (images_dir / f"{job['name']}.png").exists())
    ]
    settings = generator.settings
    config.update(
        T2I_model=generator.name,
        model_version=generator.version,
        generation_settings={**settings, "seed_base": seed, "batch_size": batch_size},
        audit_date=config.get("audit_date") or now_iso()[:10],
    )
    config.update_section(
        "generation",
        status="running",
        started_at=config.section("generation").get("started_at") or now_iso(),
        resumed_at=now_iso(),
        prompt_files={str(mode): str(path) for mode, path in prompt_files.items()},
        prompts_per_mode={
            str(mode): sum(job["mode"] == mode for job in jobs) for mode in prompt_files
        },
        total=len(jobs),
        done=len(jobs) - len(todo),
        failed=0,
        environment=environment_info(),
    )
    log.info("%d images to generate (%d already done)", len(todo), len(jobs) - len(todo))

    if todo:
        log.info("Loading %s", generator.name)
        generator.load()

    done, failed = len(jobs) - len(todo), 0
    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            seeds = [seed + job["global_index"] for job in batch]
            try:
                images = generator.generate([job["prompt"] for job in batch], seeds)
                errors: list[str | None] = [None] * len(batch)
            except Exception as error:  # one bad batch must not lose the whole run
                log.exception("Generation failed for %s", [job["name"] for job in batch])
                images, errors = [None] * len(batch), [f"{type(error).__name__}: {error}"] * len(batch)

            for job, image, job_seed, error in zip(batch, images, seeds, errors):
                record = {
                    "name": job["name"], "prompt": job["prompt"], "mode": job["mode"],
                    "index": job["index"], "image": f"images/{job['name']}.png", "seed": job_seed,
                    "generation_settings": {**settings, "batch_size": batch_size},
                }
                if image is None:
                    record.update(generation_failed=True, error=error)
                    failed += 1
                else:
                    image.save(images_dir / f"{job['name']}.png")
                    done += 1
                metadata_file.write(json.dumps(record) + "\n")
                metadata_file.flush()
            log.info("%d/%d images (%d failed)", done, len(jobs), failed)
            config.update_section("generation", force=False, done=done, failed=failed)

    config.update_section(
        "generation", status="completed" if not failed else "completed_with_errors",
        finished_at=now_iso(), done=done, failed=failed,
    )
    log.info("Generation finished: %d images, %d failed -> %s", done, failed, out)
    return metadata_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate images from prompts with a text-to-image model.")
    parser.add_argument("--model", "-m", default="pixart-sigma",
                        help=f"Registry key ({', '.join(MODELS)}) or 'package.module:Class'.")
    parser.add_argument("--out", "-o", help="Output folder (images, metadata, config.json).")
    parser.add_argument("--prompts", "-p", action="append", metavar="MODE=PATH",
                        help="Prompt file for a prompt mode (0=control, 1=single, 2=double). "
                             "Repeatable. Default: assets/{control,single_var,double_var}_prompts.txt.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; image i uses seed + i.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Only generate the first N prompts (smoke test).")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    if args.list_models:
        for key, model in MODELS.items():
            print(f"{key:14} {model.name}  {model.settings}")
        return
    if not args.out:
        parser.error("--out is required")
    setup_logging()
    run(resolve_model(args.model), parse_prompt_files(args.prompts), args.out,
        seed=args.seed, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
