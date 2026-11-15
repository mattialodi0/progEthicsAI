"""Renderer for chapter 2: Dataset and generation coverage.

The data is built by ``src.analysis.coverage.build_coverage``.
"""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from ..explain import chapter_header, esc, findings, guide
from ..insights import chapter_two
from ..plotly_helpers import chart, numeric
from .ch2_tables import (
    attribute_sections,
    confidence_section,
    detection_table,
    missing_outputs_table,
)

_MODE_HELP = {
    "control": "gender, race, age and job all fixed",
    "single variable": "one attribute only, so the model fills in the rest",
    "double variable": "two attributes combined",
}


def _mode_chart(mode_counts: dict[str, Any], total_images: float) -> str:
    figure = go.Figure(go.Bar(
        x=list(mode_counts), y=list(mode_counts.values()),
        text=[f"{value} ({100 * numeric(value) / total_images:.0f}%)" if total_images else value
              for value in mode_counts.values()],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} images<extra></extra>",
    ))
    figure.update_layout(yaxis={"title": "Images"}, xaxis={"title": "Prompt mode"})
    return chart(figure, title="", height=320)


def _conditioning_chart(conditioned: dict[str, Any], total_images: float) -> str:
    """Horizontal bars: images whose prompt specifies each variable."""
    if not conditioned:
        return '<p class="note">No configured prompt variables found.</p>'
    ordered = sorted(conditioned.items(), key=lambda item: numeric(item[1]))
    return chart(
        go.Figure(go.Bar(
            x=[value for _, value in ordered],
            y=[name for name, _ in ordered],
            orientation="h",
            text=[value for _, value in ordered], textposition="outside", cliponaxis=False,
            customdata=[100 * numeric(value) / total_images if total_images else 0
                        for _, value in ordered],
            hovertemplate="%{y}: %{x} images (%{customdata:.1f}% of all images)<extra></extra>",
        )).update_layout(xaxis={"title": "Images whose prompt specifies the variable"}),
        title="", height=max(300, 40 + 32 * len(ordered)),
    )


def _per_prompt_text(prompt_counts: dict[str, Any]) -> str:
    per_prompt = [numeric(value) for value in prompt_counts.values()]
    if per_prompt and min(per_prompt) == max(per_prompt):
        return (f"Every prompt was generated <strong>{int(per_prompt[0])}</strong> "
                "times, so prompts carry equal weight.")
    if per_prompt:
        return (f"Images per prompt range from {int(min(per_prompt))} to "
                f"{int(max(per_prompt))}; prompts with more images weigh more "
                "in pooled statistics.")
    return ""


def _prompt_list(prompt_counts: dict[str, Any]) -> str:
    if not prompt_counts:
        return '<p class="note">No prompts found.</p>'
    return (
        "<details class='prompt-list'><summary>Show all prompts</summary><ul>"
        + "".join(
            f"<li>{html.escape(prompt)} <span class='note'>({count} images)</span></li>"
            for prompt, count in prompt_counts.items()
        )
        + "</ul></details>"
    )


def render_chapter_two(coverage: dict[str, Any]) -> str:
    """Render chapter 2 with compact charts and inline summaries."""
    mode_counts = coverage.get("mode_counts", {})
    total_images = sum(numeric(value) for value in mode_counts.values())
    mode_list = "".join(
        f"<li><strong>{esc(mode)}</strong> — {_MODE_HELP.get(str(mode).lower(), '')}</li>"
        for mode in mode_counts
    )
    detection = coverage.get("valid_detection", {})
    total = max((detection[name]["total"] for name in detection), default=0)
    prompt_counts = coverage.get("prompt_images", {})

    return f"""
  {chapter_header(2, "Dataset and generation coverage",
      "What exactly was tested, and is the evidence evenly spread?",
      "<p>Before reading any disparity, check what was generated and measured. An apparent "
      "bias can be an artifact of an unbalanced prompt set, of unequal sample sizes, or of "
      "a detector that fails more often for some images. This chapter documents the prompt "
      "design, how many images back each prompt attribute, and how often each feature "
      "could be extracted.</p>")}
  {findings(chapter_two(coverage))}
  <section id="chapter-2-modes"><h3>2.1 Prompt modes</h3>
    <p>Single-variable prompts show what the model assumes by default; control prompts are what chapter 6 needs.</p>
    <ul>{mode_list}</ul>
    <article class="plot-card">{_mode_chart(mode_counts, total_images)}</article>
  </section>
  <section id="chapter-2-attributes"><h3>2.2 Prompt attributes</h3>
    <p>How many images mention each variable, and how its values are spread. Variables
    backed by few images give imprecise results later on.</p>
    <article class="plot-card"><h4>Images conditioned by each prompt variable</h4>
      {_conditioning_chart(coverage.get("conditioned_images", {}), total_images)}</article>
    <div class="plot-grid" style="align-items:start">{attribute_sections(coverage.get("attribute_counts", {}), total)}</div>
  </section>
  <section id="chapter-2-prompts"><h3>2.3 Images per prompt</h3>
    <p>{len(prompt_counts)} unique prompts produced {sum(prompt_counts.values())} images.
    {_per_prompt_text(prompt_counts)} Repeated generations of the same prompt capture the model's
    randomness: a pattern that appears in only one of ten generations is weak evidence.</p>
    {_prompt_list(prompt_counts)}
  </section>
  <section id="chapter-2-detections"><h3>2.4 Valid detection rate by extracted feature</h3>
    <p>Share of images where each extractor returned a usable value.</p>
    {guide(
        caution="If detection fails more often for some groups (older people, darker skin, "
                "faces far from the camera), a disparity later on may come from the "
                "pipeline rather than the model.",
    )}
    {detection_table(detection)}
    {missing_outputs_table(coverage.get("missing_outputs") or {})}
    {confidence_section(coverage.get("classifier_confidence") or {})}
  </section>
"""
