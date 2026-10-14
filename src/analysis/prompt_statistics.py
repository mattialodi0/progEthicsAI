from __future__ import annotations
from collections import Counter, defaultdict
import re
from typing import Any


def _contains(prompt: str, value: str) -> bool:
    """Match a vocabulary value without matching it inside another word."""
    pattern = r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None

def _prompt(record: dict[str, Any]) -> str | None:
    value = record.get("prompt")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None

def _mode(prompt: str | None, mode: str) -> str:
    if prompt is None:
        return "unknown"

def collect_prompt_stats(
    data: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Collect prompt and dataset coverage statistics.

    Counts ending in `images` count metadata records, while counts ending in
    `prompts` count distinct normalized prompt strings.  A prompt is counted
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
    
    vocabularies = config.get('prompt_variables', {})
    variable_images: Counter[str] = Counter()
    variable_prompts: defaultdict[str, set[str]] = defaultdict(set)
    value_images: defaultdict[str, Counter[str]] = defaultdict(Counter)
    value_prompts: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for prompt in prompts:
        if prompt is None:
            continue
        for variable, values in vocabularies.items():
            matches = {value for value in values if _contains(prompt, value)}
            if not matches:
                continue
            variable_images[variable] += 1
            variable_prompts[variable].add(prompt)
            for value in matches:
                value_images[variable][value] += 1
                value_prompts[variable][value].add(prompt)

    prompt_variables: dict[str, Any] = {}
    for variable in vocabularies:
        prompt_variables[variable] = {
            "num_images": variable_images[variable],
            "num_unique_prompts": len(variable_prompts[variable]),
            "values": {
                value: {
                    "num_images": value_images[variable][value],
                    "num_unique_prompts": len(value_prompts[variable][value]),
                }
                for value in vocabularies[variable]
                if value_images[variable][value]
            },
        }

    result: dict[str, Any] = {
        "num_images": len(records),
        "num_images_with_prompt": len(known_prompts),
        "num_unique_prompts": len(unique_prompts),
        "prompt_variables": prompt_variables,
    }

    if config.get('verbose', 0) > 0:
        print(
            f"Collected prompt statistics for {len(records)} images and "
            f"{len(unique_prompts)} unique prompts."
        )
    return result
