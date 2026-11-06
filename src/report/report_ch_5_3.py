"""Renderer for chapter 5.3: stereotypical item associations."""

from __future__ import annotations

import json
from typing import Any

from .explain import findings, guide
from .insights import stereotypes


def render_stereotypical_associations(bias_stats: dict[str, Any]) -> str:
    """Render the most frequent object associations for prompt contexts."""
    associations = bias_stats.get("stereotypical_associations", {})
    serialized = json.dumps(associations, ensure_ascii=True).replace("</", "<\\/")
    return f"""
  <section id="chapter-5-stereotypical" aria-labelledby="chapter-5-stereotypical-heading">
    <h3 id="chapter-5-stereotypical-heading">5.3 Stereotypical item associations</h3>
    <p>Which detected objects co-occur unusually often with each prompted job, workplace,
    continent, toxic trait or positive trait? Each object is counted once per image, and
    generic person and face labels are excluded. Over-representation is measured with
    <strong>lift</strong>:</p>
    <p style="text-align:center"><strong>Lift(A, B) = P(B | A) / P(B)</strong></p>
    <p>where A is the prompt group and B the object. A lift of 3 means the object appears
    three times more often in that group than across all prompts of this context.</p>
    {findings(stereotypes(associations))}
    {guide(
        read=["Each bar is one group → object pair, sorted by lift, on a logarithmic axis; "
              "the dashed line at 1 marks “as common as overall”.",
              "Hover for P(B | A) — the share of the group's images containing the object — "
              "and the number of images. The minimum-count filter hides objects seen in "
              "fewer images, whose lift is unstable."],
        look=["Objects the prompt does <em>not</em> require. A guitar for “musician” or a "
              "rifle for “soldier” is expected; weapons for a continent, handcuffs for an "
              "ethnicity or luxury goods only for some groups would be stereotyped "
              "depictions.",
              "Toxic and positive traits: which props the model uses to signal “criminal”, "
              "“terrorist” or “trustworthy”, and whether these props also appear for "
              "certain demographics in chapter 8's captions.",
              "High lift <em>and</em> high P(B | A): a prop that defines most of the group's "
              "images is a strong visual template."],
        caution=["Lift is a ratio: an object seen in 1 image of a 10-image group and nowhere "
                 "else gets a huge lift. Keep the minimum count at 3 or more for conclusions.",
                 "A high lift alone does not make an association harmful; that judgement "
                 "needs the context of what the object signals.",
                 "Objects come from an automatic detector with a fixed vocabulary; missing "
                 "or wrong detections are possible."],
    )}
    <div class="controls" aria-label="Stereotypical association selector">
      <label>Prompt context <select id="chapter-five-three-attribute"></select></label>
      <label>Minimum images with the object <select id="chapter-five-three-min">
        <option value="1">1</option><option value="2">2</option>
        <option value="3" selected>3</option><option value="5">5</option></select></label>
    </div>
    <article class="plot-card"><div id="chapter-five-three-plot"></div></article>
    <details><summary>Full table: most common objects per group</summary>
      <div id="chapter-five-three-table" class="table-wrap"></div>
    </details>
  </section>
  <script>
    const chapterFiveThreeData = {serialized};
    const chapterFiveThreeLabels = {{
      job: "Job", workplace: "Workplace", continent: "Continent",
      toxic_trait: "Toxic trait", positive_trait: "Positive trait"
    }};
    const chapterFiveThreeSafe = (value) =>
      String(value).replace(/[&<>"']/g, (character) =>
        ({{"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}})[character]);
    const chapterFiveThreeTitle = (value) =>
      chapterFiveThreeLabels[value] ||
      String(value).replaceAll("_", " ").replace(/\\b\\w/g, (letter) => letter.toUpperCase());
    const chapterFiveThreeFormat = (value) =>
      value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
    const chapterFiveThreeRender = () => {{
      const attribute = document.getElementById("chapter-five-three-attribute").value;
      const minimum = Number(document.getElementById("chapter-five-three-min").value);
      const data = chapterFiveThreeData[attribute];
      const target = document.getElementById("chapter-five-three-table");
      const plot = document.getElementById("chapter-five-three-plot");
      if (!data || !data.rows.length) {{
        target.innerHTML = '<p class="note">No object associations are available.</p>';
        Plotly.purge(plot);
        return;
      }}
      target.innerHTML = `<p class="note">${{data.baseline_observations}}
        observations with a prompted ${{chapterFiveThreeTitle(attribute)}}. Each entry:
        share of the group's images containing the object (n = images, lift).</p>
        <table><thead><tr><th>${{chapterFiveThreeTitle(attribute)}}</th>
        <th>Images</th><th>Most common objects</th></tr></thead><tbody>` +
        data.rows.map((row) => `<tr><td>${{chapterFiveThreeSafe(row.group)}}</td>
          <td>${{row.observations}}</td><td>` +
          row.top_objects.map((item) => `<div><strong>${{chapterFiveThreeSafe(item.object)}}</strong>:
            ${{(100 * item.probability).toFixed(0)}}% (n=${{item.count}}, lift ${{
              chapterFiveThreeFormat(item.lift)}})</div>`).join("") +
          `</td></tr>`).join("") + "</tbody></table>";

      const plotRows = data.rows.flatMap((row) =>
        row.top_objects.filter((item) => item.count >= minimum && item.lift > 0)
          .map((item) => ({{group: row.group, item}})))
        .sort((left, right) => right.item.lift - left.item.lift)
        .slice(0, 25).reverse();
      if (!plotRows.length) {{
        Plotly.purge(plot);
        plot.innerHTML = '<p class="note">No object reaches the minimum count.</p>';
        return;
      }}
      plot.innerHTML = "";
      const groups = Array.from(new Set(plotRows.map((entry) => entry.group))).sort();
      const traces = groups.map((group) => {{
        const rows = plotRows.filter((entry) => entry.group === group);
        return {{
          type: "bar", orientation: "h", name: group,
          y: rows.map((entry) => `${{entry.group}} → ${{entry.item.object}}`),
          x: rows.map((entry) => entry.item.lift),
          customdata: rows.map((entry) => [100 * entry.item.probability, entry.item.count]),
          hovertemplate: "%{{y}}<br>Lift: %{{x:.2f}}<br>In %{{customdata[0]:.0f}}% of the group's images" +
            " (%{{customdata[1]}} images)<extra></extra>"
        }};
      }});
      Plotly.newPlot(plot, traces, {{
        title: `Most over-represented objects by prompted ${{chapterFiveThreeTitle(attribute).toLowerCase()}}`,
        barmode: "stack", height: Math.max(320, 90 + 24 * plotRows.length),
        xaxis: {{title: "Lift (log scale)", type: "log", tickvals: [0.5, 1, 2, 5, 10, 20, 50, 100]}},
        yaxis: {{categoryorder: "array",
          categoryarray: plotRows.map((entry) => `${{entry.group}} → ${{entry.item.object}}`)}},
        legend: {{title: {{text: chapterFiveThreeTitle(attribute)}}}},
        margin: {{l: 10, r: 20, t: 45, b: 50}},
        shapes: [{{type: "line", x0: 1, x1: 1, y0: 0, y1: 1, yref: "paper",
          line: {{dash: "dash", color: "#52606d"}}}}]
      }});
    }};
    const chapterFiveThreeAttributes = Object.keys(chapterFiveThreeData);
    document.getElementById("chapter-five-three-attribute").innerHTML =
      chapterFiveThreeAttributes.map((value) =>
        `<option value="${{value}}">${{chapterFiveThreeTitle(value)}}</option>`).join("");
    ["chapter-five-three-attribute", "chapter-five-three-min"].forEach((id) =>
      document.getElementById(id).addEventListener("change", chapterFiveThreeRender));
    if (chapterFiveThreeAttributes.length) {{
      chapterFiveThreeRender();
    }} else {{
      document.getElementById("chapter-five-three-table").innerHTML =
        '<p class="note">No object associations are available.</p>';
    }}
  </script>
"""
