"""Chapter 2 data: dataset and generation coverage."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from .general_statistics import CONFIDENT_PROBABILITY, _top_probability
from .preprocessing import PROBABILITY_ATTRIBUTES


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


def _detection_by_group(
    data: list[dict[str, Any]], mode_labels: dict[str, str]
) -> dict[str, dict[str, dict[str, int]]]:
    """Images and images with a detected face, per prompt mode and prompt value.

    Face-based features exist only where a face was detected, so a subgroup
    with a lower detection rate is under-represented in every face-based
    result. Uses the ``valid`` flag set by preprocessing.
    """
    groups: dict[str, dict[str, Counter[str]]] = {}

    def add(attribute: str, value: str, valid: bool) -> None:
        bucket = groups.setdefault(attribute, {}).setdefault(value, Counter())
        bucket["total"] += 1
        bucket["face_detected"] += int(valid)

    for record in data:
        if not isinstance(record, dict):
            continue
        valid = record.get("valid") is True
        mode = str(record.get("mode", "unknown"))
        add("prompt_mode", mode_labels.get(mode, mode), valid)
        features = record.get("prompt_feats")
        if isinstance(features, dict):
            for attribute, value in features.items():
                if value is not None and str(value).strip():
                    add(attribute, str(value).strip().lower(), valid)
    return {
        attribute: {value: dict(counts) for value, counts in sorted(values.items())}
        for attribute, values in sorted(groups.items())
    }


def _classifier_confidence(data: list[dict[str, Any]]) -> dict[str, Any]:
    """How sure the face model is of the class it reports, per attribute.

    A label whose top class has a low probability is close to a guess between
    two classes. The share of such labels is reported per attribute and per
    requested value of the same attribute, because uncertainty that
    concentrates on some groups weakens every result about them.
    """
    result: dict[str, Any] = {}
    for attribute in PROBABILITY_ATTRIBUTES:
        overall: list[float] = []
        by_requested: dict[str, list[float]] = {}
        for record in data:
            if not isinstance(record, dict):
                continue
            top = _top_probability(record, attribute)
            if top is None:
                continue
            overall.append(top)
            features = record.get("prompt_feats")
            requested = features.get(attribute) if isinstance(features, dict) else None
            group = str(requested).strip().lower() if requested is not None else "not requested"
            by_requested.setdefault(group, []).append(top)

        def summary(values: list[float]) -> dict[str, Any]:
            ordered = sorted(values)
            return {
                "n": len(ordered),
                "confident": sum(value >= CONFIDENT_PROBABILITY for value in ordered),
                "median": round(ordered[len(ordered) // 2], 4) if ordered else None,
            }

        result[attribute] = {
            **summary(overall),
            "by_requested": {group: summary(values)
                             for group, values in sorted(by_requested.items())},
        }
    return {"threshold": CONFIDENT_PROBABILITY, "attributes": result}


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
        "detection_by_group": _detection_by_group(data, mode_labels),
        "classifier_confidence": _classifier_confidence(data),
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
