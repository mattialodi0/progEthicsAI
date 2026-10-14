"""Renderer for chapter 1: Executive summary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
from typing import Any


@dataclass(frozen=True)
class RenderedChapter:
    html: str
    data: dict[str, Any]


def _first_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return "Not specified"


def _generation_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("generation_settings", config.get("generation"))
    return settings if isinstance(settings, dict) else {}


def _metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<strong class="metric-value">{html.escape(str(value))}</strong>'
        "</div>"
    )


def _summary(
    data: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt_stats = analysis_results.get("prompt_stats", {})
    general_stats = analysis_results.get("general_stats", {})
    invalid_images = general_stats.get("invalid_images", {})
    generated_images = prompt_stats.get("num_images", len(data))
    failed_features = invalid_images.get("count", 0)
    if not isinstance(generated_images, int) or generated_images < 0:
        raise ValueError("prompt_stats.num_images must be a non-negative integer")
    if not isinstance(failed_features, int) or failed_features < 0:
        raise ValueError("general_stats.invalid_images.count must be a non-negative integer")
    return {
        "model_name": _first_config_value(config, "model_name", "model", "T2I_model"),
        "model_version": _first_config_value(config, "model_version", "version"),
        "generation_settings": _generation_settings(config),
        "audit_date": config.get(
            "audit_date", datetime.now().astimezone().date().isoformat()
        ),
        "num_prompts": prompt_stats.get("num_unique_prompts", 0),
        "generated_images": generated_images,
        "valid_images": max(0, generated_images - failed_features),
        "failed_feature_detections": failed_features,
    }


def render_chapter_one(
    data: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    config: dict[str, Any],
) -> RenderedChapter:
    """Render the executive summary chapter."""
    summary = _summary(data, analysis_results, config)
    settings_json = json.dumps(summary["generation_settings"], indent=2, ensure_ascii=True)
    content = f"""
  <div class="eyebrow">Chapter 1</div>
  <h2>Executive summary</h2>
  <p class="subtitle">Audit date: {html.escape(str(summary["audit_date"]))}</p>
  <section aria-labelledby="model-heading">
    <h3 id="model-heading">Model and audit configuration</h3>
    <p><strong>Model:</strong> {html.escape(str(summary["model_name"]))}</p>
    <p><strong>Version:</strong> {html.escape(str(summary["model_version"]))}</p>
    <details><summary>Show generation settings</summary>
      <pre>{html.escape(settings_json)}</pre>
    </details>
  </section>
  <h3>Audit coverage</h3>
  <div class="metrics">
    {_metric_card("Prompts", summary["num_prompts"])}
    {_metric_card("Generated images", summary["generated_images"])}
    {_metric_card("Valid images", summary["valid_images"])}
    {_metric_card("Failed feature detections", summary["failed_feature_detections"])}
  </div>
  <section aria-labelledby="fairness-heading">
    <h3 id="fairness-heading">Fairness overview</h3>
    <p class="placeholder">Fairness findings will be added in a later chapter.</p>
  </section>
  <section aria-labelledby="scope-heading">
    <h3 id="scope-heading">What this audit establishes</h3>
    <p>This audit summarizes the generated sample and feature detections
    available from the configured analysis pipeline. It can identify patterns
    and disparities in this dataset under these prompts and settings.</p>
    <p class="note">It cannot establish that the model is fair or unfair in
    general, prove causal bias, or predict results for untested prompts,
    populations, models, or settings.</p>
  </section>
"""
    return RenderedChapter(content, summary)