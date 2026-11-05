"""Renderer for chapter 2: Dataset and generation coverage.

The data is built by ``src.analysis.coverage.build_coverage``.
"""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from .plotly_helpers import chart, numeric


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
      <div class="eyebrow">Chapter 2</div>
      <h2 id="chapter-2">Dataset and generation coverage</h2>
      <p class="note">This chapter describes what was actually tested. 
      We reported the prompts used to generate the images, with their types, attributes and counts.</p>
      <section id="chapter-2-modes"><h3>Prompt modes</h3>
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
"""
