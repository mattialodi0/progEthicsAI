"""Renderer for chapter 1: Executive summary."""

from __future__ import annotations

import html
import json
from typing import Any

from .explain import chapter_header, esc, guide


def _metric_card(label: str, value: Any, help_text: str = "", href: str = "") -> str:
    tag = "a" if href else "div"
    link = f' href="{esc(href)}"' if href else ""
    help_html = f'<span class="metric-help">{help_text}</span>' if help_text else ""
    return (
        f'<{tag} class="metric"{link}>'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<strong class="metric-value">{value}</strong>{help_html}'
        f"</{tag}>"
    )


def render_chapter_one(
    summary: dict[str, Any],
    indicators: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Render the executive summary chapter.

    ``indicators`` are (label, value, explanation, link) tiles computed from
    the other chapters by ``insights.headline_indicators``.
    """
    settings = summary.get("generation_settings") or {}
    settings_html = (
        f"<details><summary>Show generation settings</summary>"
        f"<pre>{html.escape(json.dumps(settings, indent=2, ensure_ascii=True))}</pre></details>"
        if settings else
        '<p class="note">No generation settings (seed, steps, guidance scale, resolution) '
        "were recorded. Record them to make the audit reproducible.</p>"
    )
    generated = summary.get("generated_images") or 0
    valid = summary.get("valid_images") or 0
    failed = summary.get("failed_feature_detections") or 0
    valid_share = f"{100 * valid / generated:.1f}% of images" if generated else ""
    failed_share = f"{100 * failed / generated:.1f}% of images" if generated else ""
    per_prompt = (f"≈ {generated / summary['num_prompts']:.0f} images per prompt"
                  if summary.get("num_prompts") else "")
    tiles = "".join(
        _metric_card(label, value, help_text, href)
        for label, value, help_text, href in (indicators or [])
    )
    fairness_html = (
        f'<div class="metrics">{tiles}</div>'
        + guide(
            look="These are separate indicators, not one fairness score. A default gender far "
                 "from 50/50, a wide adherence spread or a Cramér's V above 0.3 are the "
                 "things worth following up in the linked chapters.",
        )
        + '<p class="note">A full cross-chapter synthesis of findings is in chapter 9.</p>'
        if tiles else
        '<p class="placeholder">Headline indicators are not available for this artifact.</p>'
    )
    return f"""
  {chapter_header(1, "Executive summary",
                  "What was audited, how much evidence is there, and what stands out?",
                  "<p>This chapter is a one-page overview: the model and configuration, the "
                  "size of the generated sample, a handful of headline indicators drawn from "
                  "the later chapters, and what this kind of audit can and cannot "
                  "establish.</p>")}
  <section id="chapter-1-model" aria-labelledby="model-heading">
    <h3 id="model-heading">Model and audit configuration</h3>
    <table><tbody>
      <tr><th scope="row">Model</th><td>{esc(summary.get("model_name", "?"))}</td></tr>
      <tr><th scope="row">Version</th><td>{esc(summary.get("model_version", "?"))}</td></tr>
      <tr><th scope="row">Audit date</th><td>{esc(summary.get("audit_date", "?"))}</td></tr>
    </tbody></table>
    {settings_html}
  </section>
  <section id="chapter-1-coverage" aria-labelledby="chapter-1-coverage-heading">
    <h3 id="chapter-1-coverage-heading">Audit coverage</h3>
    <div class="metrics">
      {_metric_card("Prompts", esc(summary.get("num_prompts", "?")), "Distinct prompt texts.")}
      {_metric_card("Generated images", esc(generated), esc(per_prompt))}
      {_metric_card("Images with a detected face", esc(valid), esc(valid_share),
                    "#chapter-2-detections")}
      {_metric_card("Images without a detected face", esc(failed),
                    esc(failed_share) + " · excluded from face-based results",
                    "#chapter-2-detections")}
    </div>
    <p class="note">Face-based features (gender, race, age, emotion) exist only for images
    with a detected face. Objects, captions and embeddings are available for every image.
    Per-feature detection rates are in chapter 2.</p>
  </section>
  <section id="chapter-1-fairness" aria-labelledby="fairness-heading">
    <h3 id="fairness-heading">Headline indicators</h3>
    {fairness_html}
  </section>
  <section id="chapter-1-scope" aria-labelledby="scope-heading">
    <h3 id="scope-heading">What this audit can and cannot establish</h3>
    <p><strong>What it can show.</strong></p>
    <ul>
      <li>Which demographic groups the model produces by default, and how varied they are.</li>
      <li>Whether requested attributes are followed equally well for different groups.</li>
      <li>Which prompt attributes co-occur with which detected demographics, objects,
      emotions and caption words in this sample.</li>
      <li>Whether changing a single prompt word changes unrelated image properties.</li>
    </ul>
    <p><strong>What it cannot show.</strong></p>
    <ul>
      <li>That the model is fair or unfair in general: the results hold for these prompts,
      this model version and these generation settings.</li>
      <li>Why a pattern exists (training data, architecture). An output audit can't tell.</li>
      <li>Ground truth about demographics: every label is a detector prediction with its
      own error rates.</li>
      <li>Whether a difference is harmful. That is a judgement made with the evidence,
      not by it.</li>
    </ul>
  </section>
"""
