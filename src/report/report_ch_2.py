"""Renderer for chapter 2: Dataset and generation coverage.

The data is built by ``src.analysis.coverage.build_coverage``.
"""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from .explain import chapter_header, esc, findings, guide, label
from .insights import chapter_two
from .plotly_helpers import chart, numeric

_MODE_HELP = {
    "control": "gender, race, age and job all fixed",
    "single variable": "one attribute only, so the model fills in the rest",
    "double variable": "two attributes combined",
}


def _bar_class(rate: float) -> str:
    return "bar bad" if rate < 70 else "bar warn" if rate < 90 else "bar"


def render_chapter_two(coverage: dict[str, Any]) -> str:
    """Render chapter 2 with compact charts and inline summaries."""
    mode_counts = coverage.get("mode_counts", {})
    total_images = sum(numeric(value) for value in mode_counts.values())
    mode_figure = go.Figure(go.Bar(
        x=list(mode_counts), y=list(mode_counts.values()),
        text=[f"{value} ({100 * numeric(value) / total_images:.0f}%)" if total_images else value
              for value in mode_counts.values()],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y} images<extra></extra>",
    ))
    mode_figure.update_layout(yaxis={"title": "Images"}, xaxis={"title": "Prompt mode"})
    mode_chart = chart(mode_figure, title="", height=320)
    mode_list = "".join(
        f"<li><strong>{esc(mode)}</strong> — {_MODE_HELP.get(str(mode).lower(), '')}</li>"
        for mode in mode_counts
    )

    detection = coverage.get("valid_detection", {})
    total = max((detection[name]["total"] for name in detection), default=0)

    def detection_row(name: str) -> str:
        item = detection[name]
        percent = 100 * numeric(item["valid"]) / item["total"] if item["total"] else 0
        return (
            f"<tr><th scope='row'>{html.escape(label(name))}</th>"
            f"<td>{item['valid']} / {item['total']}</td>"
            f"<td><div class='{_bar_class(percent)}' role='img' aria-label='{percent:.1f}% "
            f"valid'><span style='width:{percent:.1f}%'></span></div></td>"
            f"<td>{percent:.1f}%</td><td>{item['total'] - item['valid']}</td></tr>"
        )

    detection_table = (
        "<table><thead><tr><th>Feature</th><th>Valid / total</th>"
        "<th>Rate</th><th>Percent</th><th>Missing</th></tr></thead><tbody>"
        + "".join(detection_row(name) for name in detection)
        + "</tbody></table>"
    )
    missing = coverage.get("missing_outputs") or {}
    missing_html = (
        "<h4>Missing outputs</h4><table><thead><tr><th>Output</th><th>Missing values</th>"
        "</tr></thead><tbody>" + "".join(
            f"<tr><th scope='row'>{esc(label(name))}</th><td>{esc(value)}</td></tr>"
            for name, value in missing.items())
        + "</tbody></table><p class='note'>Missing demographic detections are counted per "
        "feature (gender, race, age and emotion each), so one image without a face adds 4.</p>"
        if missing else ""
    )

    prompt_counts = coverage.get("prompt_images", {})
    per_prompt = [numeric(value) for value in prompt_counts.values()]
    if per_prompt and min(per_prompt) == max(per_prompt):
        per_prompt_text = (f"Every prompt was generated <strong>{int(per_prompt[0])}</strong> "
                           "times, so prompts carry equal weight.")
    elif per_prompt:
        per_prompt_text = (f"Images per prompt range from {int(min(per_prompt))} to "
                           f"{int(max(per_prompt))}; prompts with more images weigh more "
                           "in pooled statistics.")
    else:
        per_prompt_text = ""
    prompt_list = (
        "<details class='prompt-list'><summary>Show all prompts</summary><ul>"
        + "".join(
            f"<li>{html.escape(prompt)} <span class='note'>({count} images)</span></li>"
            for prompt, count in prompt_counts.items()
        )
        + "</ul></details>"
        if prompt_counts else '<p class="note">No prompts found.</p>'
    )

    def attribute_section(attribute: str, values: dict[str, int]) -> str:
        top = max((numeric(count) for count in values.values()), default=0)
        rows = ""
        for value, count in sorted(values.items(), key=lambda item: -numeric(item[1])):
            width = 100 * numeric(count) / top if top else 0
            share = 100 * numeric(count) / total if total else 0
            rows += (
                f"<tr><th scope='row'>{html.escape(str(value))}</th><td>{count}</td>"
                f"<td><div class='bar' title='{share:.1f}% of all images'>"
                f"<span style='width:{width:.1f}%'></span></div></td></tr>"
            )
        return (
            f"<article class='plot-card'><h4>{html.escape(attribute)}</h4>"
            "<table><thead><tr><th>Value</th><th>Images</th><th>Relative count</th></tr>"
            f"</thead><tbody>{rows}</tbody></table></article>"
        )

    attribute_sections = "".join(
        attribute_section(label(attribute), values)
        for attribute, values in coverage.get("attribute_counts", {}).items()
    )
    conditioned = coverage.get("conditioned_images", {})
    ordered = sorted(conditioned.items(), key=lambda item: numeric(item[1]))
    conditioning_chart = chart(
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
    ) if conditioned else '<p class="note">No configured prompt variables found.</p>'

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
    <article class="plot-card">{mode_chart}</article>
  </section>
  <section id="chapter-2-attributes"><h3>2.2 Prompt attributes</h3>
    <p>How many images mention each variable, and how its values are spread. Variables
    backed by few images give imprecise results later on.</p>
    <article class="plot-card"><h4>Images conditioned by each prompt variable</h4>
      {conditioning_chart}</article>
    <div class="plot-grid" style="align-items:start">{attribute_sections}</div>
  </section>
  <section id="chapter-2-prompts"><h3>2.3 Images per prompt</h3>
    <p>{len(prompt_counts)} unique prompts produced {sum(prompt_counts.values())} images.
    {per_prompt_text} Repeated generations of the same prompt capture the model's
    randomness: a pattern that appears in only one of ten generations is weak evidence.</p>
    {prompt_list}
  </section>
  <section id="chapter-2-detections"><h3>2.4 Valid detection rate by extracted feature</h3>
    <p>Share of images where each extractor returned a usable value.</p>
    {guide(
        caution="If detection fails more often for some groups (older people, darker skin, "
                "faces far from the camera), a disparity later on may come from the "
                "pipeline rather than the model.",
    )}
    {detection_table}
    {missing_html}
  </section>
"""
