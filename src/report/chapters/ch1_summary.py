"""Renderer for chapter 1: Executive summary."""

from __future__ import annotations

import html
import json
from typing import Any

from ..explain import badge, callout, chapter_header, esc, guide
from .ch9_format import BAND_TONE, FLAG_TEXT

_VERDICT = {"large": "Large", "medium": "Moderate", "small": "Small",
            "negligible": "Negligible", None: "None supported"}


def _scorecard(rows: list[dict[str, Any]]) -> str:
    """Compact table: one row per fairness question, from the chapter-9 synthesis."""
    body = []
    for row in rows:
        tests = row.get("tests") or 0
        supported = row.get("supported") or 0
        share = 100 * supported / tests if tests else 0
        reliable = row.get("reliable") or {}
        reliable_text = ", ".join(
            f"{reliable[key]} reliably {word}"
            for key, word in (("large", "large"), ("medium", "moderate")) if reliable.get(key)
        )
        flags = "".join(
            " " + badge("⚠ " + FLAG_TEXT.get(code, code), "mid")
            for code in row.get("example_flags") or [] if code
        )
        example = (esc(row["example"]) + flags) if row.get("example") else (
            '<span class="note">No result reaches the support threshold.</span>')
        body.append(
            "<tr>"
            f'<th scope="row"><a href="{esc(row.get("link", "#chapter-9"))}">'
            f'{esc(row.get("title", ""))}</a>'
            f'<span class="scorecard-question">{esc(row.get("question", ""))}</span></th>'
            f'<td>{badge(_VERDICT.get(row.get("band"), "n/a"), BAND_TONE.get(row.get("band"), "neutral"))}</td>'
            f'<td><div class="ratio"><div class="bar" role="img" aria-label="{supported} of '
            f'{tests} tests supported"><span style="width:{share:.0f}%"></span></div>'
            f"<span>{supported} / {tests}</span></div>"
            + (f'<span class="scorecard-question">{esc(reliable_text)}</span>' if reliable_text else "")
            + f"</td><td>{example}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table class="scorecard"><thead><tr>'
        "<th>Fairness question</th><th>Largest reliable disparity</th>"
        "<th>Supported / tested</th><th>Strongest example</th></tr></thead>"
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


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
    conclusions: list[str] | None = None,
    scorecard: list[dict[str, Any]] | None = None,
) -> str:
    """Render the executive summary chapter.

    ``indicators`` are (label, value, explanation, link) tiles computed from
    the other chapters by ``insights.headline_indicators``; ``conclusions`` are
    the data-driven points of ``insights.conclusive_insights`` and
    ``scorecard`` the per-question rows of ``insights.question_scorecard``.
    """
    settings = summary.get("generation_settings") or {}
    settings_html = (
        f"<details><summary>Show all generation settings</summary>"
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
    scorecard_html = _scorecard(scorecard) if scorecard else ""
    conclusions_html = (
        '<section id="chapter-1-conclusions" aria-labelledby="conclusions-heading">'
        '<h3 id="conclusions-heading">Main conclusions</h3>'
        + scorecard_html
        + (callout("finding", conclusions, "What the audit found") if conclusions else "")
        + '<p class="note">Verdict = effect band of the strongest supported result at the '
          'lower end of its 95% confidence interval. The full answers are in '
          '<a href="#chapter-9-answers">chapter 9.1</a>. They describe measured differences '
          "in this sample, not whether the model is fair overall; how far the numbers can be "
          'trusted is explained in <a href="#chapter-10">chapter 10</a>.</p></section>'
        if scorecard or conclusions else ""
    )
    return f"""
  {chapter_header(1, "Executive summary",
                  "Does the model treat different social groups fairly, and how strong is the "
                  "evidence?",
                  "<p>This audit probes a text-to-image model for <strong>bias and "
                  "fairness</strong>: who it draws when the prompt does not say, whether it "
                  "follows requests equally well for every group, which demographics it ties "
                  "to jobs, places and traits, and whether changing a single word changes "
                  "unrelated parts of the image. This chapter is a one-page overview: the "
                  "main conclusions, the model and sample, headline indicators from the later "
                  "chapters, and what this kind of audit can and cannot establish.</p>")}
  {conclusions_html}
  <section id="chapter-1-model" aria-labelledby="model-heading">
    <h3 id="model-heading">Generation settings</h3>
    <p class="note">Model <strong>{esc(summary.get("model_name", "?"))}</strong>
    (version {esc(summary.get("model_version", "?"))}), audited on
    {esc(summary.get("audit_date", "?"))}.</p>
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
