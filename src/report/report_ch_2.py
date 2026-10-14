"""Data preparation and renderer for chapter 2: Dataset and generation coverage."""

from __future__ import annotations

from collections import Counter
import html
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
                for attribute, values in prompt_variables.items():
                    if not isinstance(values, list):
                        continue
                    matched = False
                    for value in values:
                        if isinstance(value, str) and _matches(prompt, value):
                            prompt_attributes[str(attribute)][value] += 1
                            matched = True
                    if matched:
                        conditioned_images[_conditioning_label(str(attribute))] += 1

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
    record_settings = Counter()
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


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _bar_table(values: dict[str, int]) -> str:
    maximum = max(values.values(), default=0)
    body = ""
    for label, count in values.items():
        width = 100 * count / maximum if maximum else 0
        body += (
            f'<tr><td>{html.escape(str(label))}</td><td>{count}</td><td>'
            f'<div class="bar"><span style="width:{width:.2f}%"></span></div>'
            "</td></tr>"
        )
    return (
        "<table><thead><tr><th>Value</th><th>Images</th><th>Distribution</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _conditioning_plot(conditioned_images: dict[str, int], total: int) -> str:
    """Render the share of generated images conditioned on each variable."""
    rows = []
    for attribute, count in conditioned_images.items():
        percentage = 100 * count / total if total else 0
        rows.append(
            f'<div class="plot-row"><span class="plot-label">{html.escape(attribute)}</span>'
            f'<div class="plot-track"><span style="width:{percentage:.2f}%"></span></div>'
            f'<span class="plot-value">{percentage:.1f}% ({count})</span></div>'
        )
    return "".join(rows) or '<p class="note">No configured prompt variables found.</p>'


def render_chapter_two(coverage: dict[str, Any]) -> str:
    """Render chapter 2 as expandable, readable HTML tables."""
    mode_rows = [[mode, count] for mode, count in coverage["mode_counts"].items()]
    feature_rows = [
        [
            feature,
            values["valid"],
            values["total"],
            f"{100 * values['valid'] / values['total']:.1f}%" if values["total"] else "0.0%",
        ]
        for feature, values in coverage["valid_detection"].items()
    ]
    missing = coverage["missing_outputs"]
    prompt_rows = [[prompt, count] for prompt, count in coverage["prompt_images"].items()]
    attribute_sections = "".join(
        f"<details><summary>{html.escape(attribute)}</summary>{_bar_table(values)}</details>"
        for attribute, values in coverage["attribute_counts"].items()
    )
    prompt_html = _table(["Prompt", "Images"], prompt_rows) if prompt_rows else "<p>No prompts found.</p>"
    return f"""
  <div class="eyebrow">Chapter 2</div>
  <h2>Dataset and generation coverage</h2>
  <p class="note">This chapter describes what was actually tested. Counts are
  based on metadata records and prompt text; missing detections are not treated
  as evidence that an attribute is absent from an image.</p>
  <section><h3>Prompts by mode</h3>{_table(["Mode", "Images"], mode_rows)}</section>
  <section><h3>Prompt attributes</h3>
    <p class="note">Bars show the relative number of generated images whose
    prompt contains each configured value.</p>
    <h4>Images conditioned by prompt variable</h4>
    <div class="conditioning-plot" role="img"
      aria-label="Percentage of generated images conditioned by each prompt variable">
      {_conditioning_plot(coverage["conditioned_images"], sum(coverage["mode_counts"].values()))}
    </div>
    {attribute_sections}
  </section>
  <section><h3>Images per prompt</h3>
    <p>{len(coverage["prompt_images"])} unique prompts produced
    {sum(coverage["prompt_images"].values())} images. Expand the list only when
    prompt-level counts are needed.</p>
    <details><summary>Show prompt image counts</summary>{prompt_html}</details>
  </section>
  <section><h3>Valid detection rate by extracted feature</h3>
    {_table(["Feature", "Valid", "Total", "Rate"], feature_rows)}
    <h3>Missing outputs and failed generations</h3>
    {_table(["Output", "Missing or failed"], [
        [key.replace("_", " ").title(), value] for key, value in missing.items()
    ])}
  </section>
  <section><h3>Generation settings distribution</h3>
    <pre>{html.escape(str(coverage["generation_settings_distribution"]))}</pre>
    <p class="note">Per-image distributions require settings to be stored in
    each metadata record.</p>
  </section>
"""
