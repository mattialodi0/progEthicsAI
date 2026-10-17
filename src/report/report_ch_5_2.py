"""Renderer for chapter 5.3: stereotypical item associations."""

from __future__ import annotations

import json
from typing import Any


def render_stereotypical_associations(bias_stats: dict[str, Any]) -> str:
    """Render the most frequent object associations for jobs and workplaces."""
    associations = bias_stats.get("stereotypical_associations", {})
    serialized = json.dumps(associations, ensure_ascii=True).replace("</", "<\\/")
    return f"""
  <section id="chapter-5-stereotypical" aria-labelledby="chapter-5-stereotypical-heading">
    <h3 id="chapter-5-stereotypical-heading">5.3 Stereotypical item associations</h3>
    <p>These tables show which detected objects most often co-occur with each
    prompted job or workplace. Each object is counted once per image, even if
    detected multiple times. Lift compares the conditional prevalence for a
    group with the object's overall prevalence:
    <strong>Lift(A, B) = P(B | A) / P(B)</strong>. Values above 1 indicate
    over-representation, while values below 1 indicate under-representation.
    Generic person and face labels are excluded.</p>
    <div class="checkbox-grid" aria-label="Stereotypical association selector">
      <label>Prompt context
        <select id="chapter-five-three-attribute"></select>
      </label>
    </div>
    <div id="chapter-five-three-table"></div>
    <div id="chapter-five-three-plot" style="height:420px"></div>
  </section>
  <script>
    const chapterFiveThreeData = {serialized};
    const chapterFiveThreeSafe = (value) =>
      String(value).replace(/[&<>"']/g, (character) =>
        ({{"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}})[character]);
    const chapterFiveThreeTitle = (value) =>
      String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
    const chapterFiveThreeFormat = (value) =>
      value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
    const chapterFiveThreeRender = () => {{
      const attribute = document.getElementById("chapter-five-three-attribute").value;
      const data = chapterFiveThreeData[attribute];
      const target = document.getElementById("chapter-five-three-table");
      if (!data || !data.rows.length) {{
        target.innerHTML = '<p class="note">No object associations are available.</p>';
        Plotly.purge("chapter-five-three-plot");
        return;
      }}
      target.innerHTML = `<p class="note">${{data.baseline_observations}}
        images with a detected object and prompted ${{chapterFiveThreeTitle(attribute)}}.</p>
        <table><thead><tr><th>${{chapterFiveThreeTitle(attribute)}}</th>
        <th>Observations</th><th>Most common objects</th></tr></thead><tbody>` +
        data.rows.map((row) => `<tr><td>${{chapterFiveThreeSafe(row.group)}}</td>
          <td>${{row.observations}}</td><td>` +
          row.top_objects.map((item) => `<div><strong>${{chapterFiveThreeSafe(item.object)}}</strong>:
            ${{(100 * item.probability).toFixed(1)}}% (n=${{item.count}}, lift=${{
              chapterFiveThreeFormat(item.lift)}})</div>`).join("") +
          `</td></tr>`).join("") + "</tbody></table>";

      const plotRows = data.rows.flatMap((row) =>
        row.top_objects.slice(0, 3).map((item) => ({{
          group: row.group, object: item.object, lift: item.lift || 0
        }})));
      Plotly.newPlot("chapter-five-three-plot", [{{
        type: "bar",
        x: plotRows.map((item) => `${{item.group}}: ${{item.object}}`),
        y: plotRows.map((item) => item.lift),
        customdata: plotRows.map((item) => [item.group, item.object]),
        hovertemplate: "Group: %{{customdata[0]}}<br>Object: %{{customdata[1]}}<br>" +
          "Lift: %{{y:.2f}}<extra></extra>"
      }}], {{
        title: "Top object lifts by prompted " + chapterFiveThreeTitle(attribute),
        xaxis: {{title: "Prompt group and object", tickangle: -45}},
        yaxis: {{title: "Lift", rangemode: "tozero"}},
        margin: {{l: 60, r: 25, t: 55, b: 130}}
      }}, {{responsive: true, displaylogo: false}});
    }};
    const chapterFiveThreeAttributes = Object.keys(chapterFiveThreeData);
    document.getElementById("chapter-five-three-attribute").innerHTML =
      chapterFiveThreeAttributes.map((value) =>
        `<option value="${{value}}">${{chapterFiveThreeTitle(value)}}</option>`).join("");
    document.getElementById("chapter-five-three-attribute").addEventListener(
      "change", chapterFiveThreeRender);
    if (chapterFiveThreeAttributes.length) {{
      chapterFiveThreeRender();
    }} else {{
      document.getElementById("chapter-five-three-table").innerHTML =
        '<p class="note">No object associations are available.</p>';
    }}
  </script>
"""