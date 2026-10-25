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
    be expected from its constituent groups.</p>
    <p>Heatmap annotations show sample sizes, small multiples compare one detected category across intersections,
    and the forest plot shows the joint-group difference from the average of
    the two constituent-group probabilities with 95% confidence intervals.
    The interaction-effect plot shows the additional percentage-point effect
    of each attribute combination after accounting for both main effects.
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
      const heatmap = [{{
        z: matrix.rows.map((row) =>
          matrix.categories.map((value) =>
            Number(row.values[value]?.percentage || 0))),
        x: matrix.categories, y: labels, type: "heatmap", colorscale: "Blues",
        text: matrix.rows.map((row) =>
          matrix.categories.map((value) =>
            `n=${{row.values[value]?.count || 0}}`)),
        texttemplate: "%{{z:.1f}}%<br>%{{text}}",
        hovertemplate: "%{{y}}<br>%{{x}}: %{{z:.1f}}%<br>%{{text}}<extra></extra>",
        showscale: false
      }}];
      const selectedForest = matrix.forest.filter((row) => row.category === category);
      const selectedModel = (matrix.models || []).find((model) =>
        model.output_category === category);
      const selectedDifferenceInDifferences =
        matrix.difference_in_differences?.[category];
      const interactionTerms = selectedModel?.interaction_terms || [];
      const interactionEffect = (term) =>
        Number(term.effect_percentage_points ?? (Number(term.estimate) * 100));
      const modelText = (matrix.models || []).map((model) =>
        `<li>${{chapterFiveFourSafe(model.output_category)}}:
        χ²-like interaction statistic ${{model.interaction_statistic.toFixed(2)}},
        df=${{model.interaction_df}}, p=${{model.interaction_p_value.toFixed(3)}}
        (n=${{model.n}})</li>`).join("");
      target.innerHTML = `<p class="note">${{matrix.total}} paired observations.
        Selected category: <strong>${{chapterFiveFourSafe(category)}}</strong>.</p>
        <div>
          <article class="plot-card"><h4>5.4.1 Intersectional heatmap</h4>
            <div id="chapter-five-four-heatmap" style="height:360px"></div></article>
          <article class="plot-card"><h4>5.4.2 Same feature across joint groups</h4>
            <div id="chapter-five-four-small-multiple" style="height:360px"></div></article>
          <article class="plot-card"><h4>5.4.3 Forest plot: joint minus constituent average</h4>
            <p>The forest plot shows the difference between the joint-group probability and the average of
            the two constituent-group probabilities for the selected category. </p>
            <p>Difference > 0: the outcome is more common in the joint group than the constituent average would suggest.</p>
            <p>Difference < 0: the outcome is less common.</p>
            <p>Difference near 0: the joint group is close to the constituent average.</p>
            <p>Error bars show 95% confidence intervals.</p>
            <div id="chapter-five-four-forest" style="height:420px"></div></article>
          <article class="plot-card"><h4>5.4.4 Interaction effects</h4>
            <p>The interaction effect is β3 in
            <em>Y = β0 + β1G + β2R + β3(G × R)</em>. It is the additional
            change in the selected output probability, in percentage points,
            associated with the combination of the displayed non-reference
            levels after accounting for their individual effects. The
            interaction p-value above tests all interaction terms jointly.</p>
            <div id="chapter-five-four-interaction" style="height:420px"></div></article>
          <article class="plot-card"><h4>5.4.5 Difference-in-differences plot</h4>
            <p>This plot shows the selected category across the levels of
            ${{chapterFiveFourTitle(item.second_attribute)}} for the two
            ${{chapterFiveFourTitle(item.first_attribute)}} levels used in the
            comparison. The difference-in-differences compares how the gap
            between those lines changes from the baseline to the comparison horizontal-axis level.</p>
            <p class="note">${{selectedDifferenceInDifferences ?
              `Difference-in-differences: <strong>${{
                selectedDifferenceInDifferences.difference_in_differences_percentage_points.toFixed(2)
              }} percentage points</strong>` :
              "Difference-in-differences is unavailable because the selected pair does not have two complete levels for the selected category."}}</p>
            <div id="chapter-five-four-difference-in-differences" style="height:420px"></div></article>
        </div>
        <h4>Interaction model results</h4>
        ${{modelText ? `<ul>${{modelText}}</ul>` :
          '<p class="note">Models were not fit: sample size or design was insufficient.</p>'}}`;
      Plotly.newPlot("chapter-five-four-heatmap", heatmap, {{
        title: `${{chapterFiveFourTitle(item.first_attribute)}} × ${{chapterFiveFourTitle(item.second_attribute)}} → ${{chapterFiveFourTitle(output)}}`,
        xaxis: {{title: "Detected category", tickangle: -45}},
        yaxis: {{title: "Prompt attribute pair"}},
        margin: {{l: 160, r: 20, t: 55, b: 120}}
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
      Plotly.newPlot("chapter-five-four-interaction", [{{
        type: "bar", orientation: "h",
        x: interactionTerms.map(interactionEffect),
        y: interactionTerms.map((term) => `${{term.first}} × ${{term.second}}`),
        customdata: interactionTerms.map((term) => [term.estimate, category]),
        marker: {{color: interactionTerms.map((term) =>
          interactionEffect(term) >= 0 ? "#486581" : "#d64545")}},
        text: interactionTerms.map((term) =>
          `${{interactionEffect(term).toFixed(2)}} pp`),
        textposition: "outside",
        cliponaxis: false,
        hovertemplate: "%{{y}}<br>Interaction effect: %{{x:.2f}} percentage points<br>" +
          "β3: %{{customdata[0]:.4f}}<extra></extra>"
      }}], {{
        xaxis: {{title: "Additional interaction effect (percentage points)"}},
        yaxis: {{title: "Attribute combination", automargin: true}},
        shapes: [{{type: "line", x0: 0, x1: 0, y0: 0, y1: 1,
          yref: "paper", line: {{dash: "dash", color: "#829ab1"}}}}],
        margin: {{l: 170, r: 80, t: 20, b: 70}}
      }}, {{responsive: true, displaylogo: false}});
      const differencePlot = document.getElementById(
        "chapter-five-four-difference-in-differences");
      if (!selectedDifferenceInDifferences) {{
        differencePlot.innerHTML =
          '<p class="note">No complete difference-in-differences comparison is available.</p>';
      }} else {{
        const firstLevels = selectedDifferenceInDifferences.first_levels;
        const points = selectedDifferenceInDifferences.points;
        const traces = firstLevels.map((first) => {{
          const linePoints = points
            .filter((point) => point.first === first)
            .sort((left, right) =>
              selectedDifferenceInDifferences.second_levels.indexOf(left.second) -
              selectedDifferenceInDifferences.second_levels.indexOf(right.second));
          return {{
            type: "scatter", mode: "lines+markers", name: first,
            x: linePoints.map((point) => point.second),
            y: linePoints.map((point) => point.percentage),
            customdata: linePoints.map((point) => point.n),
            hovertemplate: `${{first}}<br>%{{x}}: %{{y:.1f}}%<br>` +
              "n=%{{customdata}}<extra></extra>"
          }};
        }});
        Plotly.newPlot(differencePlot, traces, {{
          xaxis: {{title: chapterFiveFourTitle(item.second_attribute)}},
          yaxis: {{title: "Selected category proportion (%)", range: [0, 100]}},
          legend: {{title: {{text: chapterFiveFourTitle(item.first_attribute)}}}},
          margin: {{l: 65, r: 25, t: 20, b: 70}}
        }}, {{responsive: true, displaylogo: false}});
      }}
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
