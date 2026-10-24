"""Data preparation and renderer for chapter 2: Dataset and generation coverage."""

from __future__ import annotations

from collections import Counter
import html
import json
import re
from typing import Any

import plotly.graph_objects as go

from .plotly_helpers import chart, numeric

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


def render_chapter_two(coverage: dict[str, Any]) -> str:
    """Render chapter 2 with compact charts and inline summaries."""
    mode_counts = coverage["mode_counts"]
    mode_chart = chart(
        go.Figure(go.Bar(x=list(mode_counts), y=list(mode_counts.values()))),
        title="",
    )
    detection = coverage["valid_detection"]
    features = list(detection)
    total = max((detection[name]["total"] for name in features), default=0)
    def detection_row(name: str) -> str:
        item = detection[name]
        percent = 100 * numeric(item["valid"]) / item["total"] if item["total"] else 0
        return (
            f"<tr><th scope='row'>{html.escape(name)}</th>"
            f"<td>{item['valid']}/{item['total']}</td>"
            f"<td><div class='bar' role='img' aria-label='{item['valid']} valid "
            f"out of {item['total']}'><span style='width:{percent:.1f}%'></span>"
            f"</div></td><td>{percent:.1f}%</td></tr>"
        )
    detection_rows = "".join(
        detection_row(name)
        for name in features
    )
    detection_table = (
        "<table><thead><tr><th>Feature</th><th>Valid / total</th>"
        "<th>Rate</th><th>Percent</th></tr></thead><tbody>"
        f"{detection_rows}</tbody></table>"
    )
    prompt_counts = coverage["prompt_images"]
    prompt_chart = (
        "<details class='prompt-list'>"
        "<summary>Show all prompts</summary>"
        "<ul>"
        + "".join(
            f"<li>{html.escape(prompt)} "
            f"<span class='note'>({count} images)</span></li>"
            for prompt, count in prompt_counts.items()
        )
        + "</ul></details>"
        if prompt_counts
        else '<p class="note">No prompts found.</p>'
    )
    def attribute_section(attribute: str, values: dict[str, int]) -> str:
        rows = ""
        for value, count in values.items():
            percent = 100 * numeric(count) / total if total else 0
            rows += (
                f"<tr><th scope='row'>{html.escape(str(value))}</th><td>{count}</td>"
                f"<td><div class='bar'><span style='width:{percent:.1f}%'></span>"
                "</div></td></tr>"
            )
        return (
            "<article class='plot-card'><h4>"
            f"{html.escape(attribute)}</h4><table><thead><tr><th>Value</th>"
            "<th>Images</th><th>Share</th></tr></thead><tbody>"
            f"{rows}</tbody></table></article>"
        )

    attribute_sections = "".join(
        attribute_section(attribute.capitalize().replace("_", " "), values)
        for attribute, values in coverage["attribute_counts"].items()
    )
    conditioned = coverage["conditioned_images"]
    conditioning_chart = chart(
        go.Figure(go.Bar(
            x=list(conditioned.values()),
            y=list(conditioned),
            orientation="h",
            customdata=[
                100 * numeric(value) / sum(mode_counts.values())
                if sum(mode_counts.values()) else 0
                for value in conditioned.values()
            ],
            hovertemplate="%{y}: %{x} images (%{customdata:.1f}%)<extra></extra>",
        )),
        title="Images conditioned by prompt variable",
    ) if conditioned else '<p class="note">No configured prompt variables found.</p>'
    return f"""
    <div id="chapter-container">
      <div class="eyebrow">Chapter 2</div>
      <h2 id="chapter-2">Dataset and generation coverage</h2>
      <p class="note">This chapter describes what was actually tested. 
      We reported the prompts used to generate the images, with their types, attributes and counts.</p>
      <section><h3>Prompt modes</h3>
       <p class="note">Distribution of prompts by their mode.</p>{mode_chart}</section>
      <section id="chapter-2-attributes"><h3>Prompt attributes</h3>
        <h4>Prompts per attribute</h4>
        {conditioning_chart}
        <h4>Prompt variables</h4>
        {attribute_sections}
      </section>
      <section id="chapter-2-prompts"><h3>Images per prompt</h3>
        <p>{len(coverage["prompt_images"])} unique prompts produced
        {sum(coverage["prompt_images"].values())} images.</p>
        {prompt_chart}
      </section>
      <section id="chapter-2-detections"><h3>Valid detection rate by extracted feature</h3>
        {detection_table}
      </section>
    </div>
"""
