"""Statistics about prompts used to generate an image dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any


_VISUAL_MODES = ("image", "photograph", "portrait", "picture")
_PROMPT_VOCABULARIES = {
    "gender": ("male", "female", "man", "woman"),
    "age": ("young", "old", "middle-aged"),
    "race": ("white", "black", "asian", "hispanic", "middle eastern"),
    "emotion": ("happy", "sad", "angry", "calm", "bored", "nervous"),
    "job": (
        "doctor", "nurse", "engineer", "teacher", "artist", "scientist",
        "chef", "pilot", "writer", "musician", "athlete", "politician",
        "soldier",
    ),
    "workplace": (
        "office", "factory", "hospital", "school", "restaurant", "store",
        "laboratory", "construction site",
    ),
    "continent": ("Europe", "Asia", "Africa", "North America", "South America"),
    "status": ("successful", "homeless", "wealthy", "poor"),
    "toxic_trait": (
        "criminal", "gang member", "uneducated person", "terrorist",
        "manipulative person",
    ),
    "positive_trait": (
        "intelligent", "trustworthy", "professional", "hardworking",
    ),
}


def _contains(prompt: str, value: str) -> bool:
    """Match a vocabulary value without matching it inside another word."""
    pattern = r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


def _prompt(record: dict[str, Any]) -> str | None:
    value = record.get("prompt")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _mode(prompt: str | None) -> str:
    if prompt is None:
        return "unknown"
    for mode in _VISUAL_MODES:
        if _contains(prompt, mode):
            return mode
    return "unknown"


def collect_prompt_stats(
    data: list[dict[str, Any]], verbose: int = 0
) -> dict[str, Any]:
    """Collect prompt and dataset coverage statistics.

    Counts ending in ``images`` count metadata records, while counts ending in
    ``prompts`` count distinct normalized prompt strings.  A prompt is counted
    for a variable when it contains at least one value from that variable's
    vocabulary.  The returned dictionary contains only JSON-serializable
    values, so it can be written directly to a report.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of image metadata objects")

    records = [record for record in data if isinstance(record, dict)]
    prompts = [_prompt(record) for record in records]
    known_prompts = [prompt for prompt in prompts if prompt is not None]
    unique_prompts = set(known_prompts)

    mode_images: Counter[str] = Counter(_mode(prompt) for prompt in prompts)
    mode_prompts: defaultdict[str, set[str]] = defaultdict(set)
    for prompt in known_prompts:
        mode_prompts[_mode(prompt)].add(prompt)

    variable_images: Counter[str] = Counter()
    variable_prompts: defaultdict[str, set[str]] = defaultdict(set)
    value_images: defaultdict[str, Counter[str]] = defaultdict(Counter)
    value_prompts: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for prompt in prompts:
        if prompt is None:
            continue
        for variable, values in _PROMPT_VOCABULARIES.items():
            matches = {value for value in values if _contains(prompt, value)}
            if not matches:
                continue
            variable_images[variable] += 1
            variable_prompts[variable].add(prompt)
            for value in matches:
                value_images[variable][value] += 1
                value_prompts[variable][value].add(prompt)

    prompt_variables: dict[str, Any] = {}
    for variable in _PROMPT_VOCABULARIES:
        prompt_variables[variable] = {
            "num_images": variable_images[variable],
            "num_unique_prompts": len(variable_prompts[variable]),
            "values": {
                value: {
                    "num_images": value_images[variable][value],
                    "num_unique_prompts": len(value_prompts[variable][value]),
                }
                for value in _PROMPT_VOCABULARIES[variable]
                if value_images[variable][value]
            },
        }

    result: dict[str, Any] = {
        "num_images": len(records),
        "num_images_with_prompt": len(known_prompts),
        "num_missing_prompts": len(records) - len(known_prompts),
        "num_unique_prompts": len(unique_prompts),
        "images_per_mode": dict(
            sorted({mode: mode_images[mode] for mode in _VISUAL_MODES}.items())
        ),
        "images_with_unknown_mode": mode_images["unknown"],
        "unique_prompts_per_mode": {
            mode: len(mode_prompts[mode]) for mode in _VISUAL_MODES
        },
        "prompt_variables": prompt_variables,
    }

    if verbose > 0:
        print(
            f"Collected prompt statistics for {len(records)} images and "
            f"{len(unique_prompts)} unique prompts."
        )
    return result
