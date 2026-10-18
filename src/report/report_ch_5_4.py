"""Renderer for chapter 5.4: intersectional analysis."""

from __future__ import annotations

import json
from typing import Any


def render_intersectional_analysis(bias_stats: dict[str, Any]) -> str:
    """Render joint-group heatmaps, comparisons, and interaction tests."""
    serialized = json.dumps(
        bias_stats.get("intersectional", {}), ensure_ascii=True
    ).replace("</", "<\\/")
    return f"""
  <section id="chapter-5-intersectional" aria-labelledby="chapter-5-intersectional-heading">
    <h3 id="chapter-5-intersectional-heading">5.4 Intersectional analysis</h3>
    <p>This analysis tests whether a joint prompt group differs from what would
    be expected from its constituent groups. Heatmap annotations show sample
    sizes, small multiples compare one detected category across intersections,
    and the forest plot shows the joint-group difference from the average of
    the two constituent-group probabilities with 95% confidence intervals.
    Interaction models include both main effects and their interaction when at
    least 20 observations and an estimable design are available.</p>
    <div id="chapter-5-four-controls" class="checkbox-grid"
      aria-label="Intersectional analysis selectors"></div>
    <div id="chapter-5-four-analysis"></div>
  </section>
  <script>
    const chapterFiveFourData = {serialized};
    const chapterFiveFourLabels = {{
      gender: "Gender", race: "Race", age: "Age", emotion: "Emotion",
      job: "Job", workplace: "Workplace", continent: "Continent",
      status: "Status", toxic_trait: "Toxic trait", positive_trait: "Positive trait"
    }};
    const chapterFiveFourTitle = (value) =>
      chapterFiveFourLabels[value] || String(value).replaceAll("_", " ");
    const chapterFiveFourSafe = (value) =>
      String(value).replace(/[&<>"']/g, (character) =>
        ({{"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}})[character]);
    const chapterFiveFourRender = () => {{
      const pair = document.getElementById("chapter-five-four-pair").value;
      const output = document.getElementById("chapter-five-four-output").value;
      const category = document.getElementById("chapter-five-four-category").value;
      const item = chapterFiveFourData[pair];
      const matrix = item?.outputs?.[output];
      const target = document.getElementById("chapter-5-four-analysis");
      if (!matrix) {{
        target.innerHTML = '<p class="note">No intersectional observations are available.</p>';
        return;
      }}
      const labels = matrix.rows.map((row) => row.group);
      const values = matrix.rows.map((row) =>
        Number(row.values[category]?.percentage || 0));
      const sampleSizes = matrix.rows.map((row) => row.n);
      const heatmap = matrix.categories.map((value) => ({{
        z: [matrix.rows.map((row) => Number(row.values[value]?.percentage || 0))],
        x: labels, y: [value], type: "heatmap", colorscale: "Blues",
        text: [matrix.rows.map((row) => `n=${{row.n}}`)], texttemplate: "%{{z:.1f}}%<br>%{{text}}",
        hovertemplate: "%{{x}}<br>%{{y}}: %{{z:.1f}}%<br>%{{text}}<extra></extra>",
        showscale: false
      }}));
      const selectedForest = matrix.forest.filter((row) => row.category === category);
      const modelText = (matrix.models || []).map((model) =>
        `<li>${{chapterFiveFourSafe(model.output_category)}}:
        χ²-like interaction statistic ${{model.interaction_statistic.toFixed(2)}},
        df=${{model.interaction_df}}, p=${{model.interaction_p_value.toFixed(3)}}
        (n=${{model.n}})</li>`).join("");
      target.innerHTML = `<p class="note">${{matrix.total}} paired observations.
        Selected category: <strong>${{chapterFiveFourSafe(category)}}</strong>.</p>
        <div>
          <article class="plot-card"><h4>Intersectional heatmap</h4>
            <div id="chapter-five-four-heatmap" style="height:360px"></div></article>
          <article class="plot-card"><h4>Small multiple: same feature across joint groups</h4>
            <div id="chapter-five-four-small-multiple" style="height:360px"></div></article>
          <article class="plot-card"><h4>Forest plot: joint minus constituent average</h4>
            <div id="chapter-five-four-forest" style="height:420px"></div></article>
        </div>
        <h4>Interaction model results</h4>
        ${{modelText ? `<ul>${{modelText}}</ul>` :
          '<p class="note">Models were not fit: sample size or design was insufficient.</p>'}}`;
      Plotly.newPlot("chapter-five-four-heatmap", heatmap, {{
        title: `${{chapterFiveFourTitle(item.first_attribute)}} ×
          ${{chapterFiveFourTitle(item.second_attribute)}} → ${{chapterFiveFourTitle(output)}}`,
        xaxis: {{tickangle: -45}}, margin: {{l: 90, r: 20, t: 55, b: 120}}
      }}, {{responsive: true, displaylogo: false}});
      Plotly.newPlot("chapter-five-four-small-multiple", [{{
        type: "bar", x: labels, y: values, text: sampleSizes.map((n) => `n=${{n}}`),
        textposition: "auto", hovertemplate: "%{{x}}<br>%{{y:.1f}}%<br>%{{text}}<extra></extra>"
      }}], {{yaxis: {{title: "Detected percentage (%)", range: [0, 100]}},
        xaxis: {{tickangle: -45}}, margin: {{l: 55, r: 20, t: 20, b: 120}}
      }}, {{responsive: true, displaylogo: false}});
      Plotly.newPlot("chapter-five-four-forest", [{{
        type: "scatter", mode: "markers",
        x: selectedForest.map((row) => row.difference),
        y: selectedForest.map((row) => row.group),
        error_x: {{type: "data", symmetric: false,
          array: selectedForest.map((row) => row.upper - row.difference),
          arrayminus: selectedForest.map((row) => row.difference - row.lower)}},
        customdata: selectedForest.map((row) => [`n=${{row.n}}`, row.category]),
        hovertemplate: "%{{y}}<br>Difference: %{{x:.1f}} percentage points<br>" +
          "%{{customdata[0]}}<extra></extra>"
      }}], {{xaxis: {{title: "Joint minus constituent average (percentage points)"}},
        shapes: [{{type: "line", x0: 0, x1: 0, y0: 0, y1: 1,
          yref: "paper", line: {{dash: "dash", color: "#829ab1"}}}}],
        margin: {{l: 140, r: 25, t: 20, b: 70}}
      }}, {{responsive: true, displaylogo: false}});
    }};
    const chapterFiveFourEntries = Object.entries(chapterFiveFourData);
    document.getElementById("chapter-5-four-controls").innerHTML =
      '<label>Prompt pair <select id="chapter-five-four-pair">' +
      chapterFiveFourEntries.map(([key, item]) =>
        `<option value="${{key}}">${{chapterFiveFourTitle(item.first_attribute)}} ×
        ${{chapterFiveFourTitle(item.second_attribute)}}</option>`).join("") +
      '</select></label><label>Detected feature <select id="chapter-five-four-output"></select></label>' +
      '<label>Category <select id="chapter-five-four-category"></select></label>';
    const pairSelect = document.getElementById("chapter-five-four-pair");
    const outputSelect = document.getElementById("chapter-five-four-output");
    const categorySelect = document.getElementById("chapter-five-four-category");
    const chapterFiveFourRefreshCategories = () => {{
      const item = chapterFiveFourData[pairSelect.value];
      categorySelect.innerHTML = item.outputs[outputSelect.value].categories.map((value) =>
        `<option value="${{value}}">${{chapterFiveFourSafe(value)}}</option>`).join("");
      chapterFiveFourRender();
    }};
    const chapterFiveFourRefreshOptions = () => {{
      const item = chapterFiveFourData[pairSelect.value];
      outputSelect.innerHTML = Object.keys(item.outputs).map((value) =>
        `<option value="${{value}}">${{chapterFiveFourTitle(value)}}</option>`).join("");
      chapterFiveFourRefreshCategories();
    }};
    pairSelect.addEventListener("change", chapterFiveFourRefreshOptions);
    outputSelect.addEventListener("change", chapterFiveFourRefreshCategories);
    categorySelect.addEventListener("change", chapterFiveFourRender);
    if (chapterFiveFourEntries.length) chapterFiveFourRefreshOptions();
    else document.getElementById("chapter-5-four-analysis").innerHTML =
      '<p class="note">No intersectional observations are available.</p>';
  </script>
"""
