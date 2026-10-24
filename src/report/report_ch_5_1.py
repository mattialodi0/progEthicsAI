"""Renderer for chapter 5: Conditional distributions."""

from __future__ import annotations

import json
from typing import Any


def render_chapter_five(bias_stats: dict[str, Any]) -> str:
    """Render selectable conditional bar or pie charts."""
    serialized = json.dumps(bias_stats, ensure_ascii=True).replace("</", "<\\/")
    return f"""
    <div id="chapter-container">
  <div class="eyebrow">Chapter 5</div>
  <h2 id="chapter-5">Conditional distributions and Association analysis</h2>
  <p class="subtitle"> This chapter is the core of the analysis and examines whether prompt attributes 
  systematically co-occur with detected properties of generated images. 
  Such information is crucial for understanding the model's behavior and identifying potential biases.</p>
  <section id="chapter-5-single-variable" aria-labelledby="chapter-5-single-heading">
    <h3 id="chapter-5-single-heading">5.1 Single-variable–attribute association matrices</h3>
    <p>Each plot shows the detected output distribution conditional on one
    prompt category. Percentages are normalized within each prompt category.
    Images missing either the prompt category or detected feature are excluded
    from that plot. Cramér's V summarizes association strength from the
    contingency table, while the Pearson chi-square p-value indicates the
    uncertainty of the association test.</p>
    <div id="chapter-5-controls" class="checkbox-grid" aria-label="Association plot selectors"></div>
    <div id="chapter-5-associations" class=""></div>
  </section>
  <section id="chapter-5-double-variable" aria-labelledby="chapter-5-double-heading">
    <h3 id="chapter-5-double-heading">5.2 Double-variable–attribute association matrices</h3>
    <p>Each Seaborn heatmap shows the two prompt attributes together on one
    combined axis, with categories of the selected detected feature on the
    other axis. Cell values are percentages conditional on each prompt pair;
    annotations include the paired observation counts. Select a prompt pair
    and detected feature below.</p>
    <div id="chapter-5-double-controls" class="checkbox-grid"
      aria-label="Double-variable association selectors"></div>
    <div id="chapter-5-double-association" class="plot-card"></div>
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
    const chapterFiveChartType = () =>
      document.querySelector("#chapter-5-controls input[name='chart-type']:checked").value;
    function chapterFiveRender() {{
      const prompts = chapterFiveSelected("prompt-attribute");
      const outputs = chapterFiveSelected("output-feature");
      const chartType = chapterFiveChartType();
      const cards = [];
      const plots = [];
      prompts.forEach((promptAttribute) => outputs.forEach((outputFeature) => {{
        const matrix = chapterFiveData.associations?.[promptAttribute]?.[outputFeature];
        if (!matrix || !matrix.total) return;
        const plotId = `chapter-five-${{promptAttribute}}-${{outputFeature}}`;
        const statistics = matrix.statistics || {{}};
        cards.push(`<article class="plot-card"><h4>Prompt ${{chapterFiveSafe(chapterFiveTitle(promptAttribute))}} vs detected ${{chapterFiveSafe(chapterFiveTitle(outputFeature))}}</h4><p class="note">${{matrix.total}}
          paired observations</p><p class="note">Cramér's V:
          ${{Number(statistics.cramers_v || 0).toFixed(3)}} (association strength)
          · Chi-square:
          ${{Number(statistics.chi_square || 0).toFixed(3)}} (df=${{statistics.degrees_of_freedom || 0}})</p>
          <div id="${{plotId}}" style="height:360px"></div></article>`);
        plots.push([plotId, matrix, promptAttribute, outputFeature, chartType]);
      }}));
      document.getElementById("chapter-5-associations").innerHTML = cards.join("") ||
        '<p class="note">No paired observations are available for the selected plots.</p>';
      plots.forEach(([plotId, matrix, promptAttribute, outputFeature, chartType]) => {{
        if (chartType === "pie") {{
          const columnCount = Math.min(2, Math.max(matrix.rows.length, 1));
          const rowCount = Math.ceil(matrix.rows.length / columnCount);
          const chartHeight = Math.max(340, 230 * rowCount);
          document.getElementById(plotId).style.height = `${{chartHeight}}px`;
          const traces = matrix.rows.map((row, index) => ({{
            type: "pie",
            labels: matrix.columns,
            values: matrix.columns.map((column) => matrix.percentages[row][column]),
            text: matrix.columns.map((column) =>
              `${{matrix.counts[row][column]}} observations`),
            hovertemplate: "Detected: %{{label}}<br>%{{value:.1f}}%<br>%{{text}}<extra></extra>",
            name: row,
            domain: {{row: Math.floor(index / columnCount), column: index % columnCount}},
            title: {{text: row}}
          }}));
          Plotly.newPlot(plotId, traces, {{
            grid: {{rows: rowCount, columns: columnCount, pattern: "independent"}},
            margin: {{l: 30, r: 30, t: 20, b: 30}}, height: chartHeight,
            showlegend: true
          }}, {{responsive: true, displaylogo: false}});
        }} else {{
          const traces = matrix.columns.map((column) => ({{
            type: "bar",
            name: column,
            x: matrix.rows,
            y: matrix.rows.map((row) => matrix.percentages[row][column]),
            customdata: matrix.rows.map((row) => matrix.counts[row][column]),
            hovertemplate: "Prompt: %{{x}}<br>Detected: ${{column}}<br>%{{y:.1f}}%<br>%{{customdata}} observations<extra></extra>"
          }}));
          Plotly.newPlot(plotId, traces, {{
            barmode: "group", yaxis: {{title: "Detected percentage (%)", range: [0, 100]}},
            xaxis: {{title: `Prompt ${{chapterFiveTitle(promptAttribute)}}`}},
            margin: {{l: 60, r: 25, t: 20, b: 75}}, height: 340
          }}, {{responsive: true, displaylogo: false}});
        }}
      }});
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
        chapterFiveControl("output-feature", value, index == 3)).join("") +
      '</fieldset><fieldset><legend>Chart type</legend>' +
      '<label><input type="radio" name="chart-type" value="bar" checked> Bar</label>' +
      '<label><input type="radio" name="chart-type" value="pie"> Pie</label></fieldset>';
    document.querySelectorAll("#chapter-5-controls input").forEach((input) =>
      input.addEventListener("change", chapterFiveRender));
    chapterFiveRender();
    const chapterFiveDouble = chapterFiveData.double_associations || {{}};
    const chapterFiveDoubleEntries = Object.entries(chapterFiveDouble);
    const chapterFiveDoubleLabels = (key) => {{
      const item = chapterFiveDouble[key];
      return `${{chapterFiveTitle(item.first_attribute)}} ×
        ${{chapterFiveTitle(item.second_attribute)}}`;
    }};
    const chapterFiveRenderDouble = () => {{
      const pair = document.getElementById("chapter-5-double-pair").value;
      const output = document.getElementById("chapter-5-double-output").value;
      const item = chapterFiveDouble[pair];
      const matrix = item?.outputs?.[output];
      const target = document.getElementById("chapter-5-double-association");
      if (!matrix) {{
        target.innerHTML = '<p class="note">No paired observations are available.</p>';
        return;
      }}
      target.innerHTML = `<h4>Prompt ${{chapterFiveSafe(chapterFiveDoubleLabels(pair))}}
        vs detected ${{chapterFiveSafe(chapterFiveTitle(output))}}</h4>
        <p class="note">${{matrix.total}} paired observations. The heatmap was
        generated with Seaborn during analysis.</p>
        <img src="${{matrix.image}}" alt="Seaborn heatmap of
        ${{chapterFiveSafe(chapterFiveDoubleLabels(pair))}} and
        ${{chapterFiveSafe(chapterFiveTitle(output))}}" style="max-width:100%">`;
    }};
    document.getElementById("chapter-5-double-controls").innerHTML =
      '<label>Prompt pair <select id="chapter-5-double-pair">' +
      chapterFiveDoubleEntries.map(([key]) =>
        `<option value="${{key}}">${{chapterFiveSafe(chapterFiveDoubleLabels(key))}}</option>`
      ).join("") + '</select></label>' +
      '<label>Detected feature <select id="chapter-5-double-output">' +
      (chapterFiveDoubleEntries.length ? Object.keys(
        chapterFiveDouble[chapterFiveDoubleEntries[0][0]].outputs).map((value) =>
        `<option value="${{value}}">${{chapterFiveSafe(chapterFiveTitle(value))}}</option>`
      ).join("") : "") + '</select></label>';
    const doublePairSelect = document.getElementById("chapter-5-double-pair");
    const doubleOutputSelect = document.getElementById("chapter-5-double-output");
    if (doublePairSelect && doubleOutputSelect) {{
      doublePairSelect.addEventListener("change", () => {{
        const outputs = Object.keys(chapterFiveDouble[doublePairSelect.value].outputs);
        doubleOutputSelect.innerHTML = outputs.map((value) =>
          `<option value="${{value}}">${{chapterFiveSafe(chapterFiveTitle(value))}}</option>`
        ).join("");
        chapterFiveRenderDouble();
      }});
      doubleOutputSelect.addEventListener("change", chapterFiveRenderDouble);
      chapterFiveRenderDouble();
    }} else {{
      document.getElementById("chapter-5-double-association").innerHTML =
        '<p class="note">No double-variable observations are available.</p>';
    }}
  </script>
  </div>
"""