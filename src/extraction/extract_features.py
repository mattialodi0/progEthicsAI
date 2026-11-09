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
import gzip
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..run_config import RunConfig, now_iso, setup_logging

log = logging.getLogger("extraction")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch / tuple values to plain JSON types."""
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    if hasattr(obj, "tolist"):  # numpy arrays / scalars, torch tensors
        return obj.tolist()
    return obj


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def best_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Stages. Each one produces a few fields of the final record.
# --------------------------------------------------------------------------
class Stage(ABC):
    name: str                 # key used on the command line and in the config
    features: list[str]       # analysis-level features it provides
    model: str                # measurement model, reported in the config (empty = none)
    settings: dict[str, Any] = {}

    def load(self) -> None:
        """Load the model (called only if there is something to process)."""

    def unload(self) -> None:
        """Release the model."""

    @abstractmethod
    def process(self, image: Any, record: dict[str, Any]) -> dict[str, Any]:
        """Return the new fields for one record."""


class ImageMetricsStage(Stage):
    """Brightness, saturation and contrast of the image (no model)."""

    name, features, model = "image_metrics", ["image_metrics"], ""

    def process(self, image, record):
        import numpy as np

        gray = np.asarray(image.convert("L"), dtype=np.float32)
        saturation = np.asarray(image.convert("HSV"), dtype=np.float32)[..., 1]
        return {"image_metrics": {
            "brightness": round(float(gray.mean()), 3),
            "saturation": round(float(saturation.mean()), 3),
            "contrast": round(float(gray.std()), 3),
        }}


class FacesStage(Stage):
    """Perceived age, gender, race and emotion of every detected face (DeepFace)."""

    name, features, model = "faces", ["age", "gender", "race", "emotion"], "DeepFace"
    settings = {"actions": ["age", "gender", "race", "emotion"], "detector_backend": "mtcnn",
                "align": True, "enforce_detection": True}

    def load(self) -> None:
        from deepface import DeepFace

        self._deepface = DeepFace

    def process(self, image, record):
        import numpy as np

        try:
            result = self._deepface.analyze(
                img_path=np.asarray(image.convert("RGB"))[..., ::-1],  # DeepFace expects BGR
                actions=self.settings["actions"],
                detector_backend=self.settings["detector_backend"],
                align=self.settings["align"],
                enforce_detection=self.settings["enforce_detection"],
            )
        except ValueError:  # DeepFace raises ValueError when no face is found
            result = []
        return {"facial_feats": to_jsonable(result)}


class FlorenceStage(Stage):
    """Detailed caption and open-vocabulary object labels (Florence-2)."""

    name, features, model = "florence", ["caption", "items_florence"], "Florence-2-base"
    settings = {"model_id": "microsoft/Florence-2-base", "caption_task": "<MORE_DETAILED_CAPTION>",
                "object_task": "<OD>", "num_beams": 3, "max_new_tokens": 512}

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = best_device()
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        model_id = self.settings["model_id"]
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self._dtype, trust_remote_code=True
        ).to(self._device).eval()

    def unload(self) -> None:
        self._model = self._processor = None

    def _run(self, image, task: str):
        import torch

        inputs = self._processor(text=task, images=image, return_tensors="pt").to(
            self._device, self._dtype
        )
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self.settings["max_new_tokens"],
                num_beams=self.settings["num_beams"],
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(text, task=task, image_size=image.size)[task]

    def process(self, image, record):
        image = image.convert("RGB")
        return {
            "caption": self._run(image, self.settings["caption_task"]),
            "items_florence": to_jsonable(self._run(image, self.settings["object_task"])),
        }


class ClipStage(Stage):
    """Normalised CLIP embeddings of the image and of its prompt."""

    name, features, model = "clip", ["embedding"], "CLIP ViT-B-32 (OpenAI)"
    settings = {"architecture": "ViT-B-32", "pretrained": "openai"}

    def load(self) -> None:
        import open_clip

        self._device = best_device()
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.settings["architecture"], pretrained=self.settings["pretrained"]
        )
        self._tokenizer = open_clip.get_tokenizer(self.settings["architecture"])
        self._model = self._model.to(self._device).eval()
        self._prompt_cache: dict[str, Any] = {}

    def unload(self) -> None:
        self._model = None

    def process(self, image, record):
        import torch

        prompt = record["prompt"]
        with torch.no_grad():
            if prompt not in self._prompt_cache:  # prompts repeat many times
                text = self._model.encode_text(self._tokenizer(prompt).to(self._device))
                self._prompt_cache[prompt] = (text / text.norm(dim=-1, keepdim=True)).cpu().tolist()
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
            embedding = self._model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return {"prompt_embedding": self._prompt_cache[prompt],
                "image_embedding": embedding.cpu().tolist()}


class YoloStage(Stage):
    """COCO object boxes (YOLO11-nano); used for the number of people per image."""

    name, features, model = "yolo", ["items"], "yolo11n"
    settings = {"weights": "yolo11n.pt"}

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.settings["weights"])

    def unload(self) -> None:
        self._model = None

    def process(self, image, record):
        boxes = self._model(image.convert("RGB"), verbose=False)[0].boxes
        return {"items": {
            "xyxy": boxes.xyxy.cpu().tolist(),
            "conf": boxes.conf.cpu().tolist(),
            "cls": boxes.cls.cpu().tolist(),
            "orig_shape": list(boxes.orig_shape),
        }}


STAGES: dict[str, type[Stage]] = {
    stage.name: stage for stage in (ImageMetricsStage, FacesStage, FlorenceStage, ClipStage, YoloStage)
}


# --------------------------------------------------------------------------
# Checkpointing and merging
# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Latest entry per image name."""
    entries: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                entries[entry["name"]] = entry
    return entries


def load_generated(run: Path) -> list[dict[str, Any]]:
    records = read_jsonl(run / "generation_metadata.jsonl")
    if not records:
        raise SystemExit(f"No generation_metadata.jsonl in {run}. Run the generation step first.")
    return sorted(records.values(), key=lambda r: (r.get("mode", 0), r.get("index", 0)))


def run_stage(stage: Stage, records: list[dict[str, Any]], run: Path, config: RunConfig) -> None:
    from PIL import Image

    checkpoint = run / "extraction" / f"{stage.name}.jsonl"
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
            for number, record in enumerate(todo, 1):
                entry: dict[str, Any] = {"name": record["name"]}
                try:
                    with Image.open(run / record["image"]) as image:
                        image.load()
                        entry["fields"] = stage.process(image, record)
                    state["done"] += 1
                except Exception as error:  # keep going; failed images are retried on resume
                    log.warning("[%s] %s failed: %s", stage.name, record["name"], error)
                    entry["error"] = f"{type(error).__name__}: {error}"
                    state["failed"] += 1
                out.write(json.dumps(entry) + "\n")
                out.flush()
                if number % 25 == 0:
                    log.info("[%s] %d/%d", stage.name, number, len(todo))
                    config.save(force=False)
    finally:
        stage.unload()
        free_memory()

    state["status"] = "completed" if not state["failed"] else "completed_with_errors"
    state["finished_at"] = now_iso()
    config.save()


def merge(records: list[dict[str, Any]], run: Path, stage_names: list[str]) -> list[dict[str, Any]]:
    """Combine generation records and every stage checkpoint into final records."""
    checkpoints = {name: read_jsonl(run / "extraction" / f"{name}.jsonl") for name in stage_names}
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
