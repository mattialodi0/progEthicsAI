"""Renderer for chapter 7: image and embedding analysis.

The data is built by ``src.analysis.visual_features.build_visual_feature_analysis``.
"""

from __future__ import annotations

import json
from typing import Any

from .explain import chapter_header, findings, guide, metric_defs
from .insights import chapter_seven


def _quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _box_stats(values: list[Any]) -> dict[str, float] | None:
    """Summarise a distribution for a precomputed Plotly box.

    The artifact stores every pairwise distance (tens of thousands per
    grouping); shipping only the box statistics keeps the report small.
    """
    ordered = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not ordered:
        return None
    q1, median, q3 = (_quantile(ordered, q) for q in (0.25, 0.5, 0.75))
    spread = 1.5 * (q3 - q1)
    return {
        "q1": q1, "median": median, "q3": q3,
        "lowerfence": next(value for value in ordered if value >= q1 - spread),
        "upperfence": next(value for value in reversed(ordered) if value <= q3 + spread),
        "mean": sum(ordered) / len(ordered), "n": len(ordered),
    }


def _compact(chapter_data: dict[str, Any]) -> dict[str, Any]:
    embedding = dict(chapter_data.get("embedding") or {})
    groups = {}
    for name, item in (embedding.get("groups") or {}).items():
        distances = item.get("distances") or {}
        groups[name] = {
            "separability": item.get("separability") or {},
            "distances": {key: _box_stats(values or []) for key, values in distances.items()},
        }
    embedding["groups"] = groups
    return {**chapter_data, "embedding": embedding}


def render_chapter_seven(chapter_data: dict[str, Any]) -> str:
    """Render interactive feature and embedding plots."""
    serialized = json.dumps(_compact(chapter_data), ensure_ascii=True).replace("</", "<\\/")
    return f"""
  {chapter_header(7, "Visual feature and embedding analysis",
      "Are some groups depicted differently — lighter or darker, closer or farther, more "
      "or less distinct — independently of what the prompt asked for?",
      "<p>Chapters 3–6 work with categorical labels. This chapter looks at continuous image "
      "properties and at the CLIP embedding space, where every image is a point and "
      "similar images lie close together. Differences in framing, colour or composition "
      "between demographic groups are a subtle form of representational bias: e.g. one "
      "group consistently shown in darker, lower-contrast images, or as small figures "
      "rather than portraits.</p>")}
  {findings(chapter_seven(chapter_data))}
  {metric_defs([
      ("Brightness", "Mean pixel luminance of the image (0 = black).",
       "Systematic differences between groups can reflect lighting or skin-tone rendering "
       "choices."),
      ("Saturation / contrast", "Colour intensity and the spread of luminance values.",
       "Lower values give a duller, flatter, often more sombre look."),
      ("Number of people", "People detected in the image.", "More than 1 for a single-person "
       "prompt means the model added people (crowds, colleagues)."),
      ("Person / face area ratio", "Share of the image covered by person and by face "
       "bounding boxes.", "High face ratio = close-up portrait; low = distant figure. "
       "Portraits individualise, distant figures depersonalise."),
      ("Silhouette score", "Mean of (b − a) / max(a, b) over images, where a is the mean "
       "distance to the own group and b to the nearest other group.", "−1 to 1; near 0 = "
       "groups overlap, &gt; 0.25 = noticeable structure, &gt; 0.5 = clearly separate."),
  ])}
  <section id="chapter-7-features"><h3>7.1 Image-feature distributions</h3>
    {guide(
        read=["The histogram shows the feature over all images. The box plots split it by "
              "the selected prompt attribute: the line is the median, the box the middle "
              "50%, the dashed line the mean; dots are outliers.",
              "The dashed vertical/horizontal reference is the overall median."],
        look=["Boxes of demographic groups (gender, race, age) that sit clearly above or "
              "below the others for brightness, contrast or saturation: a group rendered "
              "with a different visual style.",
              "Face or person area ratios that differ by group: one group shown as close "
              "portraits, another as distant or partial figures.",
              "Number of people &gt; 1 concentrated in some groups (e.g. crowds for some "
              "continents or statuses)."],
        caution="Brightness and colour also depend on skin tone and on settings that the "
                "prompt implies (e.g. a construction site vs an office). Compare groups "
                "within the same prompt mode where possible.",
    )}
    <div id="chapter-7-feature-controls" class="checkbox-grid"></div>
    <div class="plot-grid"><article class="plot-card"><div id="chapter-7-feature-plot"></div></article>
    <article class="plot-card"><div id="chapter-7-feature-groups"></div></article></div>
  </section>
  <section id="chapter-7-embeddings"><h3>7.2 Embedding distributions and cluster separability</h3>
    <p>Each image's CLIP embedding is projected to two dimensions. PCA is fit once to all
    embeddings, so every grouping is shown in the same coordinates and colourings are
    directly comparable. UMAP is shown when available.</p>
    {guide(
        read=["Scatter: one point per image, coloured by the selected prompt attribute.",
              "Distance plot: distances between images of the same group vs of different "
              "groups.",
              "Separability bars: silhouette score for every grouping; the selected one is "
              "highlighted."],
        look=["Groups that form their own cluster although the prompt only changed a "
              "demographic word: the model renders that group in a distinct visual "
              "“register” (setting, style, clothing), beyond the person.",
              "Within-group distances clearly smaller than between-group distances.",
              "Compare demographic groupings with job / workplace: a scene-defining attribute "
              "is expected to separate more."],
        caution=["High separability means the information is encoded in the embeddings, not "
                 "that the encoding is unfair: a requested attribute must leave a trace.",
                 "2-D projections distort distances; rely on the silhouette scores and "
                 "distance distributions rather than on the visual impression."],
    )}
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
      face_area_ratio: "Face area ratio", prompt_mode: "Prompt mode",
      toxic_trait: "Toxic trait", positive_trait: "Positive trait"
    }};
    const chapterSevenTitle = (value) => chapterSevenLabels[value] ||
      String(value).replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
    const chapterSevenState = {{feature: "brightness", grouping: "prompt_mode", projection: "pca"}};
    const chapterSevenMedian = (values) => {{
      const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
      if (!sorted.length) return null;
      const middle = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    }};
    function renderChapterSeven() {{
      // Both subgroup selectors drive the same grouping; keep them in sync.
      ["chapter-7-group", "chapter-7-embedding-group"].forEach((id) => {{
        const select = document.getElementById(id);
        if (select) select.value = chapterSevenState.grouping;
      }});
      const featureName = chapterSevenTitle(chapterSevenState.feature);
      const groupingName = chapterSevenTitle(chapterSevenState.grouping);
      const feature = chapterSevenData.image_features[chapterSevenState.feature];
      const groups = feature.by_group[chapterSevenState.grouping] || {{}};
      const median = chapterSevenMedian(feature.values);
      const reference = (axis) => median === null ? [] : [{{type: "line",
        [axis + "0"]: median, [axis + "1"]: median,
        [(axis === "x" ? "y" : "x") + "0"]: 0, [(axis === "x" ? "y" : "x") + "1"]: 1,
        [(axis === "x" ? "y" : "x") + "ref"]: "paper",
        layer: "above", line: {{dash: "dash", color: "#243b53", width: 2}}}}];
      Plotly.newPlot("chapter-7-feature-plot", [{{type: "histogram", x: feature.values,
        name: "All images", marker: {{color: "#0072B2"}}}}], {{
        title: `${{featureName}}: all images`,
        xaxis: {{title: featureName}}, yaxis: {{title: "Images"}}, bargap: 0.05,
        shapes: reference("x"), height: 380}});
      const boxes = Object.entries(groups)
        .map(([name, values]) => [name, values, chapterSevenMedian(values)])
        .sort((left, right) => (left[2] ?? 0) - (right[2] ?? 0));
      Plotly.newPlot("chapter-7-feature-groups", boxes.map(([name, values]) =>
        ({{type: "box", y: values, name: `${{name}} (n=${{values.length}})`,
          boxpoints: "outliers", boxmean: true, showlegend: false}})),
        {{title: `${{featureName}} by ${{groupingName.toLowerCase()}}`,
          yaxis: {{title: featureName}}, xaxis: {{tickangle: -30}},
          shapes: reference("y"), height: 380}});
      const points = chapterSevenData.embedding.points.filter((point) => point[chapterSevenState.projection]);
      const pointsByLabel = new Map();
      points.forEach((point) => {{
        const label = point.groups[chapterSevenState.grouping] || "not specified";
        if (!pointsByLabel.has(label)) pointsByLabel.set(label, []);
        pointsByLabel.get(label).push(point);
      }});
      const embeddingTraces = [...pointsByLabel.entries()]
        .sort((left, right) => left[0] === "not specified" ? 1 : right[0] === "not specified" ? -1 :
          left[0].localeCompare(right[0]))
        .map(([label, groupPoints]) => ({{
          x: groupPoints.map(p => p[chapterSevenState.projection][0]),
          y: groupPoints.map(p => p[chapterSevenState.projection][1]),
          mode: "markers", type: "scattergl", name: `${{label}} (${{groupPoints.length}})`,
          marker: label === "not specified" ? {{size: 4, color: "#cbd2d9", opacity: 0.5}} :
            {{size: 6, opacity: 0.75}},
          hovertemplate: `${{label}}<br>x=%{{x:.3f}}, y=%{{y:.3f}}<extra></extra>`
        }}));
      Plotly.newPlot("chapter-7-embedding-plot", embeddingTraces, {{
        title: `${{chapterSevenState.projection.toUpperCase()}} projection of CLIP embeddings, coloured by ${{groupingName.toLowerCase()}}`,
        xaxis: {{title: "Component 1", zeroline: false}}, yaxis: {{title: "Component 2", zeroline: false}},
        legend: {{title: {{text: groupingName}}, orientation: "v", x: 1.02, y: 1}},
        height: 520, margin: {{r: 10}}}});
      const distances = chapterSevenData.embedding.groups[chapterSevenState.grouping].distances;
      const box = (name, stats) => stats ? {{type: "box", name, q1: [stats.q1],
        median: [stats.median], q3: [stats.q3], lowerfence: [stats.lowerfence],
        upperfence: [stats.upperfence], mean: [stats.mean], x: [name],
        hovertemplate: `${{name}}<br>median %{{median:.3f}}<br>${{stats.n}} image pairs<extra></extra>`}} : null;
      Plotly.newPlot("chapter-7-distance-plot",
        [box("Within group", distances.within), box("Between groups", distances.between)]
          .filter(Boolean),
        {{title: `Embedding distances by ${{groupingName.toLowerCase()}}`,
          yaxis: {{title: "Euclidean distance"}}, showlegend: false, height: 380}});
      const separabilityEntries = Object.entries(chapterSevenData.embedding.groups || {{}})
        .filter(([, item]) => item.separability && item.separability.score !== null)
        .sort((left, right) => right[1].separability.score - left[1].separability.score);
      if (!separabilityEntries.length) {{
        document.getElementById("chapter-7-cluster-separability").innerHTML =
          '<p class="note">No cluster separability measurements are available.</p>';
      }} else {{
        Plotly.newPlot("chapter-7-cluster-separability", [{{
          type: "bar",
          x: separabilityEntries.map(([grouping]) => chapterSevenTitle(grouping)),
          y: separabilityEntries.map(([, item]) => item.separability.score),
          marker: {{color: separabilityEntries.map(([grouping]) =>
            grouping === chapterSevenState.grouping ? "#D55E00" : "#0072B2")}},
          customdata: separabilityEntries.map(([, item]) => [
            item.separability.n_images, item.separability.n_clusters]),
          hovertemplate: "%{{x}}<br>Silhouette score: %{{y:.3f}}" +
            "<br>Embeddings: %{{customdata[0]}}<br>Groups: %{{customdata[1]}}<extra></extra>"
        }}], {{
          title: "Cluster separability by grouping (selected in orange)",
          xaxis: {{tickangle: -35}},
          yaxis: {{title: "Silhouette score", range: [-0.1, 0.6]}},
          shapes: [{{type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.25, y1: 0.25,
            line: {{dash: "dot", color: "#a35f00"}}}}],
          annotations: [{{xref: "paper", x: 1, y: 0.25, yanchor: "bottom", xanchor: "right",
            text: "noticeable structure", showarrow: false, font: {{size: 11, color: "#a35f00"}}}}],
          height: 380, margin: {{b: 100}}
        }});
      }}
      const separation = chapterSevenData.embedding.groups[chapterSevenState.grouping].separability;
      const score = separation.score === null ? null : Number(separation.score);
      const reading = score === null ? "" : score < 0.1 ? "groups overlap almost completely" :
        score < 0.25 ? "weak structure" : score < 0.5 ? "noticeable structure" : "clearly separated groups";
      document.getElementById("chapter-7-separability").innerHTML =
        `<p class="note">Silhouette score for <strong>${{groupingName}}</strong>:
        <strong>${{score === null ? "Not available" : score.toFixed(3)}}</strong>
        ${{reading ? `(${{reading}})` : ""}} · ${{separation.n_images}} embeddings ·
        ${{separation.n_clusters}} groups.</p>`;
    }}
    const featureOptions = Object.keys(chapterSevenData.image_features);
    const groupOptions = chapterSevenData.embedding.group_by;
    document.getElementById("chapter-7-feature-controls").innerHTML =
      `<label>Feature <select id="chapter-7-feature">${{featureOptions.map(f =>
      `<option value="${{f}}">${{chapterSevenTitle(f)}}</option>`).join("")}}</select></label>
      <label>Split by prompt attribute <select id="chapter-7-group">${{groupOptions.map(g =>
      `<option value="${{g}}">${{chapterSevenTitle(g)}}</option>`).join("")}}</select></label>`;
    document.getElementById("chapter-7-embedding-controls").innerHTML =
      `<label>Colour by prompt attribute <select id="chapter-7-embedding-group">${{groupOptions.map(g =>
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
