"""Renderer for chapter 7: image and embedding analysis.

The data is built by ``src.analysis.visual_features.build_visual_feature_analysis``.
"""

from __future__ import annotations

import json
from typing import Any


def render_chapter_seven(chapter_data: dict[str, Any]) -> str:
    """Render interactive feature and embedding plots."""
    serialized = json.dumps(chapter_data, ensure_ascii=True).replace("</", "<\\/")
    return f"""
  <div class="eyebrow">Chapter 7</div>
  <h2 id="chapter-7">Visual feature and embedding analysis</h2>
  <p class="subtitle">Measurable image properties and semantic representation are
  compared using the same observations and, for embeddings, the same projection.</p>
  <section id="chapter-7-features"><h3>7.1 Image-feature distributions</h3>
    <p>Histograms show overall distributions and box plots compare subgroups.</p>
    <div id="chapter-7-feature-controls" class="checkbox-grid"></div>
    <div class=""><article class="plot-card"><div id="chapter-7-feature-plot"></div></article>
    <article class="plot-card"><div id="chapter-7-feature-groups"></div></article></div>
  </section>
  <section id="chapter-7-embeddings"><h3>7.2 Embedding distributions and cluster separability</h3>
    <p>PCA is fit once to all embeddings, so comparable groups share coordinates.
    UMAP is shown when available. Silhouette scores and within/between Euclidean
    distances are descriptive and depend on the selected grouping.</p>
    <div id="chapter-7-embedding-controls" class="checkbox-grid"></div>
    <div id="chapter-7-separability"></div>
    <article class="plot-card"><div id="chapter-7-embedding-plot"></div></article>
    <div class="plot-grid">
        <article class="plot-card"><div id="chapter-7-distance-plot"></div></article>
        <article class="plot-card"><div id="chapter-7-cluster-separability"></div></article>
    </div>
  </section>
  <script>
    const chapterSevenData = {serialized};
    const chapterSevenLabels = {{
      brightness: "Brightness", saturation: "Saturation", contrast: "Contrast",
      n_people: "Number of people", person_area_ratio: "Person area ratio",
      face_area_ratio: "Face area ratio", prompt_mode: "Prompt mode"
    }};
    const chapterSevenTitle = (value) => chapterSevenLabels[value] ||
      String(value).replaceAll("_", " ");
    const chapterSevenState = {{feature: "brightness", grouping: "prompt_mode", projection: "pca"}};
    function renderChapterSeven() {{
      // Both subgroup selectors drive the same grouping; keep them in sync.
      ["chapter-7-group", "chapter-7-embedding-group"].forEach((id) => {{
        const select = document.getElementById(id);
        if (select) select.value = chapterSevenState.grouping;
      }});
      const feature = chapterSevenData.image_features[chapterSevenState.feature];
      const groups = feature.by_group[chapterSevenState.grouping] || {{}};
      Plotly.newPlot("chapter-7-feature-plot", [{{type: "histogram", x: feature.values,
        name: "All images"}}], {{title: chapterSevenTitle(chapterSevenState.feature),
        xaxis: {{title: chapterSevenTitle(chapterSevenState.feature)}}, yaxis: {{title: "Images"}}}},
        {{responsive: true, displaylogo: false}});
      Plotly.newPlot("chapter-7-feature-groups", Object.entries(groups).map(([name, values]) =>
        ({{type: "box", y: values, name, boxpoints: "outliers"}})),
        {{title: `Subgroups by ${{chapterSevenTitle(chapterSevenState.grouping)}}`}},
        {{responsive: true, displaylogo: false}});
      const points = chapterSevenData.embedding.points.filter((point) => point[chapterSevenState.projection]);
      const pointsByLabel = new Map();
      points.forEach((point) => {{
        const label = point.groups[chapterSevenState.grouping] || "unlabelled";
        if (!pointsByLabel.has(label)) pointsByLabel.set(label, []);
        pointsByLabel.get(label).push(point);
      }});
      const embeddingTraces = [...pointsByLabel.entries()].map(([label, groupPoints]) => ({{
        x: groupPoints.map(p => p[chapterSevenState.projection][0]),
        y: groupPoints.map(p => p[chapterSevenState.projection][1]),
        mode: "markers", type: "scatter", name: label,
        hovertemplate: `${{label}}<br>x=%{{x:.3f}}, y=%{{y:.3f}}<extra></extra>`
      }}));
      Plotly.newPlot("chapter-7-embedding-plot", embeddingTraces,
        {{title: `${{chapterSevenState.projection.toUpperCase()}} projection`,
          legend: {{title: {{text: chapterSevenTitle(chapterSevenState.grouping)}}}}}},
        {{responsive: true, displaylogo: false}});
      const distances = chapterSevenData.embedding.groups[chapterSevenState.grouping].distances;
      Plotly.newPlot("chapter-7-distance-plot", [
        {{type: "box", y: distances.within, name: "Within group"}},
        {{type: "box", y: distances.between, name: "Between groups"}}
      ], {{title: "Embedding distance distributions", yaxis: {{title: "Euclidean distance"}}}},
      {{responsive: true, displaylogo: false}});
      const separabilityGroups = chapterSevenData.embedding.groups || {{}};
      const separabilityEntries = Object.entries(separabilityGroups);
      const separabilityElement = document.getElementById("chapter-7-cluster-separability");
      if (!separabilityEntries.length) {{
        separabilityElement.innerHTML =
          '<p class="note">No cluster separability measurements are available.</p>';
      }} else {{
        const separabilityTrace = {{
          type: "bar",
          x: separabilityEntries.map(([grouping]) => chapterSevenTitle(grouping)),
          y: separabilityEntries.map(([, item]) => item.separability.score),
          customdata: separabilityEntries.map(([, item]) => [
            item.separability.n_images, item.separability.n_clusters
          ]),
          hovertemplate: "%{{x}}<br>Silhouette score: %{{y:.3f}}" +
            "<br>Embeddings: %{{customdata[0]}}<br>Groups: %{{customdata[1]}}<extra></extra>"
        }};
        Plotly.newPlot("chapter-7-cluster-separability", [separabilityTrace], {{
          title: "Cluster separability by attribute",
          xaxis: {{title: "Grouping attribute"}},
          yaxis: {{title: "Silhouette score", range: [-1, 1]}},
          margin: {{b: 100}}
        }}, {{responsive: true, displaylogo: false}});
      }}
      const separation = chapterSevenData.embedding.groups[chapterSevenState.grouping].separability;
      document.getElementById("chapter-7-separability").innerHTML =
        `<p class="note">Silhouette score: <strong>${{separation.score === null ? "Not available" :
        Number(separation.score).toFixed(3)}}</strong> · ${{separation.n_images}} embeddings ·
        ${{separation.n_clusters}} groups.</p>`;
    }}
    const featureOptions = Object.keys(chapterSevenData.image_features);
    const groupOptions = chapterSevenData.embedding.group_by;
    document.getElementById("chapter-7-feature-controls").innerHTML =
      `<label>Feature <select id="chapter-7-feature">${{featureOptions.map(f =>
      `<option value="${{f}}">${{chapterSevenTitle(f)}}</option>`).join("")}}</select></label>
      <label>Subgroup <select id="chapter-7-group">${{groupOptions.map(g =>
      `<option value="${{g}}">${{chapterSevenTitle(g)}}</option>`).join("")}}</select></label>`;
    document.getElementById("chapter-7-embedding-controls").innerHTML =
      `<label>Group labels <select id="chapter-7-embedding-group">${{groupOptions.map(g =>
      `<option value="${{g}}">${{chapterSevenTitle(g)}}</option>`).join("")}}</select></label>
      <label>Projection <select id="chapter-7-projection"><option value="pca">PCA</option>
      <option value="umap" ${{chapterSevenData.embedding.projections.umap_available ? "" : "disabled"}}>UMAP</option>
      </select></label>`;
    document.getElementById("chapter-7-feature").onchange = (event) => {{
      chapterSevenState.feature = event.target.value; renderChapterSeven();
    }};
    document.getElementById("chapter-7-group").onchange = (event) => {{
      chapterSevenState.grouping = event.target.value; renderChapterSeven();
    }};
    document.getElementById("chapter-7-embedding-group").onchange = (event) => {{
      chapterSevenState.grouping = event.target.value; renderChapterSeven();
    }};
    document.getElementById("chapter-7-projection").onchange = (event) => {{
      chapterSevenState.projection = event.target.value; renderChapterSeven();
    }};
    renderChapterSeven();
  </script>
"""
