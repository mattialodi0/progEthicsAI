"""Renderer for chapter 5: Conditional distributions."""

from __future__ import annotations

import json
from typing import Any


def render_chapter_five(bias_stats: dict[str, Any]) -> str:
    """Render selectable prompt/output association heatmaps."""
    serialized = json.dumps(bias_stats, ensure_ascii=True).replace("</", "<\\/")
    return f"""
  <div class="eyebrow">Chapter 5</div>
  <h2 id="chapter-5">Conditional distributions</h2>
  <p class="subtitle">Whether prompt attributes systematically co-occur with
  detected properties of generated images.</p>
  <section id="chapter-5-single-variable" aria-labelledby="chapter-5-single-heading">
    <h3 id="chapter-5-single-heading">5.1 Single-variable–attribute association matrices</h3>
    <p>Each heatmap shows the detected output distribution conditional on one
    prompt category. Percentages are normalized within each row; counts are
    included in the hover text. Images missing either the prompt category or
    detected feature are excluded from that matrix.</p>
    <div id="chapter-5-controls" class="checkbox-grid" aria-label="Association plot selectors"></div>
    <div id="chapter-5-associations" class="plot-grid"></div>
  </section>
  <script>
    const chapterFiveData = {serialized};
    const chapterFiveLabels = {{
      gender: "Gender", race: "Race", age: "Age", emotion: "Emotion",
      job: "Job", workplace: "Workplace", continent: "Continent",
      status: "Status", toxic_trait: "Toxic trait",
      positive_trait: "Positive trait"
    }};
    const chapterFiveTitle = (value) =>
      chapterFiveLabels[value] || value.replaceAll("_", " ");
    const chapterFiveSafe = (value) =>
      String(value).replace(/[&<>"']/g, (character) =>
        ({{"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}})[character]);
    const chapterFiveSelected = (name) => new Set(Array.from(
      document.querySelectorAll(`#chapter-5-controls input[name="${{name}}"]:checked`)
    ).map((input) => input.value));
    function chapterFiveRender() {{
      const prompts = chapterFiveSelected("prompt-attribute");
      const outputs = chapterFiveSelected("output-feature");
      const cards = [];
      const plots = [];
      prompts.forEach((promptAttribute) => outputs.forEach((outputFeature) => {{
        const matrix = chapterFiveData.associations?.[promptAttribute]?.[outputFeature];
        if (!matrix || !matrix.total) return;
        const z = matrix.rows.map((row) => matrix.columns.map(
          (column) => matrix.percentages[row][column]));
        const text = matrix.rows.map((row) => matrix.columns.map(
          (column) => `${{matrix.counts[row][column]}} observations`));
        const plotId = `chapter-five-${{promptAttribute}}-${{outputFeature}}`;
        cards.push(`<article class="plot-card"><h4>Prompt ${{chapterFiveSafe(
          chapterFiveTitle(promptAttribute))}} vs detected ${{chapterFiveSafe(
          chapterFiveTitle(outputFeature))}}</h4><p class="note">${{matrix.total}}
          paired observations</p><div id="${{plotId}}" style="height:360px"></div></article>`);
        plots.push([plotId, matrix, z, text, promptAttribute, outputFeature]);
      }}));
      document.getElementById("chapter-5-associations").innerHTML = cards.join("") ||
        '<p class="note">No paired observations are available for the selected plots.</p>';
      plots.forEach(([plotId, matrix, z, text, promptAttribute, outputFeature]) =>
        Plotly.newPlot(plotId, [{{
          type: "heatmap", x: matrix.columns, y: matrix.rows, z,
          text, hovertemplate: "Prompt: %{{y}}<br>Detected: %{{x}}<br>%{{z:.1f}}%<br>%{{text}}<extra></extra>",
          texttemplate: "%{{z:.1f}}%", colorscale: "Blues", zmin: 0, zmax: 100,
          colorbar: {{title: "%"}}
        }}], {{margin: {{l: 90, r: 25, t: 20, b: 70}}, height: 340,
          xaxis: {{title: "Detected ${{chapterFiveTitle(outputFeature)}}"}},
          yaxis: {{title: "Prompt ${{chapterFiveTitle(promptAttribute)}}"}}
        }}, {{responsive: true, displaylogo: false}}));
    }}
    const chapterFiveControl = (group, value, checked) =>
      `<label><input type="checkbox" name="${{group}}" value="${{value}}" ${{checked ? "checked" : ""}}>
      ${{chapterFiveSafe(chapterFiveTitle(value))}}</label>`;
    const chapterFivePromptAttributes = chapterFiveData.prompt_attributes || [];
    const chapterFiveOutputFeatures = chapterFiveData.output_features || [];
    document.getElementById("chapter-5-controls").innerHTML =
      '<fieldset><legend>Prompt attributes</legend>' +
      chapterFivePromptAttributes.map((value, index) =>
        chapterFiveControl("prompt-attribute", value, index < 3)).join("") +
      '</fieldset><fieldset><legend>Detected features</legend>' +
      chapterFiveOutputFeatures.map((value, index) =>
        chapterFiveControl("output-feature", value, index < 3)).join("") + '</fieldset>';
    document.querySelectorAll("#chapter-5-controls input").forEach((input) =>
      input.addEventListener("change", chapterFiveRender));
    chapterFiveRender();
  </script>
"""