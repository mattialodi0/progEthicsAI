"""Renderer for chapter 1: Executive summary."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from typing import Any


@dataclass(frozen=True)
class RenderedChapter:
    html: str
    data: dict[str, Any]


def _metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<strong class="metric-value">{html.escape(str(value))}</strong>'
        "</div>"
    )


def render_chapter_one(
    summary: dict[str, Any],
) -> RenderedChapter:
    """Render the executive summary chapter."""
    settings_json = json.dumps(summary["generation_settings"], indent=2, ensure_ascii=True)
    content = f"""
  <div class="eyebrow">Chapter 1</div>
  <h2 id="chapter-1">Executive summary</h2>
  <p class="subtitle">Audit date: {html.escape(str(summary["audit_date"]))}</p>
  <section id="chapter-1-model" aria-labelledby="model-heading">
    <h3 id="model-heading">Model and audit configuration</h3>
    <p><strong>Model:</strong> {html.escape(str(summary["model_name"]))}</p>
    <p><strong>Version:</strong> {html.escape(str(summary["model_version"]))}</p>
    <details><summary>Show generation settings</summary>
      <pre>{html.escape(settings_json)}</pre>
    </details>
  </section>
  <h3 id="chapter-1-coverage">Audit coverage</h3>
  <div class="metrics">
    {_metric_card("Prompts", summary["num_prompts"])}
    {_metric_card("Generated images", summary["generated_images"])}
    {_metric_card("Valid images", summary["valid_images"])}
    {_metric_card("Failed feature detections", summary["failed_feature_detections"])}
  </div>
  <section id="chapter-1-fairness" aria-labelledby="fairness-heading">
    <h3 id="fairness-heading">Fairness overview</h3>
    <p class="placeholder">Fairness findings will be added in a later chapter.</p>
  </section>
  <section id="chapter-1-scope" aria-labelledby="scope-heading">
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