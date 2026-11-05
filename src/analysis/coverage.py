"""Chapter 2 data: dataset and generation coverage."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any


def _matches(prompt: str, value: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


def _present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, tuple, dict)) and not value:
        return False
    return True


def _feature_available(record: dict[str, Any], feature: str) -> bool:
    if feature in {"age", "gender", "race", "emotion"}:
        faces = record.get("facial_feats")
        return isinstance(faces, (list, tuple)) and any(
            isinstance(face, dict)
            and face.get("face_confidence", -1) != -1
            and face.get("age" if feature == "age" else f"dominant_{feature}") is not None
            for face in faces
        )
    if feature == "caption":
        value = record.get("caption")
        return isinstance(value, str) and bool(value.strip())
    if feature == "embedding":
        return _present(record.get("image_embedding"))
    if feature == "items":
        return _present(record.get("items"))
    return _present(record.get(feature))


def _failed_generation(record: dict[str, Any]) -> bool:
    for key in ("generation_failed", "failed", "error", "generation_error"):
        value = record.get(key)
        if value is True or (isinstance(value, str) and value.strip()):
            return True
    return False


def _conditioning_label(attribute: str) -> str:
    return {
        "ages": "age",
        "continents": "continent",
        "emotions": "emotion",
        "genders": "gender",
        "genders_noun": "gender",
        "jobs": "job",
        "positive_traits": "positive trait",
        "races": "race",
        "statuses": "status",
        "toxic_traits": "toxic trait",
        "workplaces": "workplace",
    }.get(attribute, attribute)


def build_coverage(
    data: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Collect coverage metrics directly from image metadata and configuration."""
    prompt_modes = config.get("prompt_modes", {})
    mode_labels = (
        {str(key): str(value) for key, value in prompt_modes.items()}
        if isinstance(prompt_modes, dict)
        else {}
    )
    mode_counts: Counter[str] = Counter()
    prompt_images: Counter[str] = Counter()
    conditioned_images: Counter[str] = Counter()
    prompt_variables = config.get("prompt_variables", {})
    prompt_attributes = (
        {str(key): Counter() for key in prompt_variables}
        if isinstance(prompt_variables, dict)
        else {}
    )

    for record in data:
        if not isinstance(record, dict):
            continue
        mode = str(record.get("mode", "unknown"))
        mode_counts[mode_labels.get(mode, mode)] += 1
        prompt = record.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompt_images[prompt.strip()] += 1
            if isinstance(prompt_variables, dict):
                # A prompt counts once per conditioning label, even when two
                # configured variables (e.g. genders and genders_noun) share it.
                matched_labels: set[str] = set()
                for attribute, values in prompt_variables.items():
                    if not isinstance(values, list):
                        continue
                    for value in values:
                        if isinstance(value, str) and _matches(prompt, value):
                            prompt_attributes[str(attribute)][value] += 1
                            matched_labels.add(_conditioning_label(str(attribute)))
                for label in matched_labels:
                    conditioned_images[label] += 1

    total = len(data)
    feature_names = ["caption", "embedding", "items", "age", "gender", "race", "emotion"]
    valid_detection = {
        feature: {
            "valid": sum(_feature_available(record, feature) for record in data),
            "total": total,
        }
        for feature in feature_names
    }
    settings = config.get("generation_settings", config.get("generation", {}))
    record_settings: Counter[str] = Counter()
    for record in data:
        if not isinstance(record, dict):
            continue
        value = record.get("generation_settings", record.get("settings"))
        if isinstance(value, dict):
            record_settings[json.dumps(value, sort_keys=True)] += 1
    settings_distribution: Any
    if record_settings:
        settings_distribution = [
            {"settings": json.loads(value), "images": count}
            for value, count in record_settings.items()
        ]
    else:
        settings_distribution = settings if isinstance(settings, dict) else {}

    return {
        "mode_counts": dict(sorted(mode_counts.items())),
        "attribute_counts": {
            attribute: dict(sorted(counts.items()))
            for attribute, counts in sorted(prompt_attributes.items())
        },
        "conditioned_images": dict(sorted(conditioned_images.items())),
        "prompt_images": dict(sorted(prompt_images.items())),
        "valid_detection": valid_detection,
        "missing_outputs": {
            "captions": total - valid_detection["caption"]["valid"],
            "embeddings": total - valid_detection["embedding"]["valid"],
            "demographic_detections": sum(
                total - valid_detection[feature]["valid"]
                for feature in ("age", "gender", "race", "emotion")
            ),
            "failed_generations": sum(_failed_generation(record) for record in data),
        },
        "generation_settings_distribution": settings_distribution,
    }
