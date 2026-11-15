"""Chapter 3: compact payloads for the composition and diversity charts.

The charts are drawn in the browser (``static/chapter_3.js``) for the prompt
modes the reader selects, so the payload holds one entry per prompt mode.
"""

from __future__ import annotations

from typing import Any


def _metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """Keep only the numbers the charts use (the artifact also holds every
    pairwise similarity, which would bloat the page)."""
    gender_entropy = (stats.get("demographic_entropy") or {}).get("gender") or {}
    return {
        "n_images": stats.get("n_images"),
        "embedding_dispersion": stats.get("embedding_dispersion"),
        "pairwise_similarity": (stats.get("pairwise_similarity") or {}).get("mean"),
        "demographic_entropy": gender_entropy.get("normalized_entropy"),
        "object_unique": stats.get("object_unique"),
    }


def _mode_stats(stats: dict[str, Any], mode: str) -> dict[str, Any] | None:
    by_mode = stats.get("by_prompt_mode")
    if not by_mode:
        return stats
    return next(
        (value for key, value in by_mode.items()
         if str(key).lower() == mode.lower() and isinstance(value, dict)),
        None,
    )


def _detected_counts(slices: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(name): (value or {}).get("detected", {}).get("counts", {})
        for name, value in slices.items() if isinstance(value, dict)
    }


def composition_payload(
    chapter_data: dict[str, Any], modes: list[str]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """Detected counts per dimension and prompt mode, and per job / workplace."""
    composition = chapter_data.get("composition", {})
    composition_data: dict[str, dict[str, dict[str, Any]]] = {}
    context_data: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in chapter_data.get("dimensions", []):
        by_slice = composition.get(dimension, {}).get("by_slice", {})
        slices = by_slice.get("prompt_mode", {})
        composition_data[dimension] = {
            mode: next(
                (value.get("detected", {}).get("counts", {})
                 for key, value in slices.items()
                 if str(key).lower() == mode.lower() and isinstance(value, dict)),
                {},
            )
            for mode in modes
        }
        context_data[dimension] = {
            context: _detected_counts(by_slice.get(context, {}))
            for context in ("job", "workplace") if by_slice.get(context)
        }
    return composition_data, context_data


def diversity_payload(diversity: dict[str, Any], modes: list[str]) -> dict[str, dict[str, Any]]:
    """Diversity metrics per grouping, prompt mode and group."""
    diversity_data: dict[str, dict[str, Any]] = {}
    for group_name, values in (diversity.get("groups") or {}).items():
        if not isinstance(values, dict):
            continue
        diversity_data[group_name] = {}
        for mode in modes:
            mode_values: dict[str, Any] = {}
            for name, stats in values.items():
                if not isinstance(stats, dict):
                    continue
                if group_name == "prompt_mode":
                    if str(name).lower() == mode.lower():
                        mode_values[str(name)] = _metrics(stats)
                else:
                    mode_stats = _mode_stats(stats, mode)
                    if mode_stats is not None:
                        mode_values[str(name)] = _metrics(mode_stats)
            diversity_data[group_name][mode] = mode_values
    return diversity_data
