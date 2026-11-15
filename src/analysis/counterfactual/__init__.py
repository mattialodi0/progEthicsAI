"""Chapter 6 data: matched counterfactual pairs."""

from __future__ import annotations

from typing import Any

from .attribute import attribute_analysis
from .common import CONTROL_ATTRIBUTES, DETECTED_ATTRIBUTES, NUMERICAL_FEATURES
from .transitions import control_pairs, double_pairs


COUNTERFACTUAL_ATTRIBUTES = ("gender", "race", "age", "emotion", "job", "workplace")


def _mode_key(config: dict[str, Any] | None, label: str, default: str) -> str:
    """Return the raw ``mode`` value whose configured label is ``label``."""
    modes = (config or {}).get("prompt_modes")
    if isinstance(modes, dict):
        for key, value in modes.items():
            if str(value).strip().lower() == label:
                return str(key)
    return default


def build_counterfactual_analysis(
    data: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build all chapter-6 data without retaining raw records or embeddings."""
    control_mode = _mode_key(config, "control", "0")
    double_mode = _mode_key(config, "double variable", "2")
    results = {
        attribute: attribute_analysis(data, attribute)
        for attribute in COUNTERFACTUAL_ATTRIBUTES
    }
    return {
        "attributes": list(COUNTERFACTUAL_ATTRIBUTES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "by_attribute": results,
        "control_analysis": {
            "control_attributes": list(CONTROL_ATTRIBUTES),
            "detected_attributes": list(DETECTED_ATTRIBUTES),
            "pairs": control_pairs(data, control_mode),
        },
        "double_analysis": {
            "control_attributes": list(CONTROL_ATTRIBUTES),
            "detected_attributes": list(DETECTED_ATTRIBUTES),
            "pairs": double_pairs(data, double_mode),
        },
        "caveat": (
            "Embedding distance is not itself a bias score: the requested "
            "attribute should change some image content. Interpret distances "
            "alongside matched adherence and unrelated-feature differences."
        ),
    }
