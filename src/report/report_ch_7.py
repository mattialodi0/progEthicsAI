"""Data preparation and renderer for chapter 7: image and embedding analysis."""

from __future__ import annotations

import json
from itertools import combinations
from math import sqrt
from typing import Any

from ..analysis.general_statistics import _distance, _embedding, _record_mode

IMAGE_FEATURES = (
    "brightness", "saturation", "contrast", "n_people",
    "person_area_ratio", "face_area_ratio",
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _feature_value(record: dict[str, Any], feature: str) -> float | None:
    if feature in {"brightness", "saturation", "contrast"}:
        metrics = record.get("image_metrics")
        return _number(metrics.get(feature) if isinstance(metrics, dict) else None)
    return _number(record.get(feature))


def _group_value(record: dict[str, Any], group_by: str, config: dict[str, Any]) -> str | None:
    if group_by == "prompt_mode":
        return _record_mode(record, config)
    features = record.get("prompt_feats")
    value = features.get(group_by) if isinstance(features, dict) else None
    return str(value).strip().lower() if value is not None and str(value).strip() else None


def _pca(vectors: list[list[float]]) -> list[list[float]] | None:
    if len(vectors) < 2:
        return None
    dimensions = min(len(vector) for vector in vectors)
    if not dimensions:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        matrix = np.asarray([vector[:dimensions] for vector in vectors], dtype=float)
        matrix -= matrix.mean(axis=0)
        _, _, components = np.linalg.svd(matrix, full_matrices=False)
        projection = matrix @ components[:2].T
    except (np.linalg.LinAlgError, ValueError):
        return None
    if projection.shape[1] == 1:
        projection = np.column_stack((projection[:, 0], np.zeros(len(vectors))))
    return [[round(float(x), 6), round(float(y), 6)] for x, y in projection]


def _umap(vectors: list[list[float]]) -> list[list[float]] | None:
    if len(vectors) < 3:
        return None
    try:
        import umap
        dimensions = min(len(vector) for vector in vectors)
        projection = umap.UMAP(
            n_components=2, n_neighbors=max(2, min(15, len(vectors) - 1)),
            random_state=42,
        ).fit_transform([vector[:dimensions] for vector in vectors])
    except (ImportError, ValueError):
        return None
    return [[round(float(x), 6), round(float(y), 6)] for x, y in projection]


def _distance_distributions(
    vectors: list[list[float]], labels: list[str | None], max_points: int = 600
) -> dict[str, Any]:
    if len(vectors) > max_points:
        indices = [int(index * len(vectors) / max_points) for index in range(max_points)]
        vectors = [vectors[index] for index in indices]
        labels = [labels[index] for index in indices]
    within: list[float] = []
    between: list[float] = []
    for left, right in combinations(range(len(vectors)), 2):
        value = round(_distance(vectors[left], vectors[right]), 6)
        (within if labels[left] == labels[right] else between).append(value)
    return {"within": within, "between": between}


def _silhouette(vectors: list[list[float]], labels: list[str | None]) -> float | None:
    groups = sorted({label for label in labels if label is not None})
    if len(groups) < 2:
        return None
    scores: list[float] = []
    for index, label in enumerate(labels):
        if label is None:
            continue
        own = [
            _distance(vectors[index], vectors[j])
            for j, other in enumerate(labels) if other == label and j != index
        ]
        means = [
            sum(_distance(vectors[index], vectors[j]) for j, other in enumerate(labels) if other == group)
            / labels.count(group)
            for group in groups if group != label
        ]
        if means:
            a = sum(own) / len(own) if own else 0.0
            b = min(means)
            scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return round(sum(scores) / len(scores), 6) if scores else None


def build_visual_feature_analysis(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Build JSON-safe data for all chapter-7 plots."""
    attributes = sorted({
        key for record in data
        for key in (record.get("prompt_feats", {}) if isinstance(record.get("prompt_feats"), dict) else {})
        if any(
            isinstance(item.get("prompt_feats"), dict)
            and item["prompt_feats"].get(key) is not None for item in data
        )
    })
    groupings = ["prompt_mode", *attributes]
    features: dict[str, Any] = {}
    for feature in IMAGE_FEATURES:
        features[feature] = {
            "values": [
                value for record in data
                if (value := _feature_value(record, feature)) is not None
            ],
            "by_group": {
                grouping: {
                    group: [
                        value for record in data
                        if _group_value(record, grouping, config) == group
                        and (value := _feature_value(record, feature)) is not None
                    ]
                    for group in sorted(
                        set(
                            _group_value(record, grouping, config)
                            for record in data
                            if _group_value(record, grouping, config) is not None
                        )
                    )
                }
                for grouping in groupings
            },
        }

    correlation_values = {
        feature: [_feature_value(record, feature) for record in data]
        for feature in IMAGE_FEATURES
    }
    correlation_matrix = []
    for first in IMAGE_FEATURES:
        row = []
        for second in IMAGE_FEATURES:
            pairs = [
                (left, right) for left, right in zip(
                    correlation_values[first], correlation_values[second]
                ) if left is not None and right is not None
            ]
            if len(pairs) < 2:
                row.append(None)
                continue
            left_mean = sum(left for left, _ in pairs) / len(pairs)
            right_mean = sum(right for _, right in pairs) / len(pairs)
            numerator = sum((left - left_mean) * (right - right_mean) for left, right in pairs)
            denominator = sqrt(
                sum((left - left_mean) ** 2 for left, _ in pairs)
                * sum((right - right_mean) ** 2 for _, right in pairs)
            )
            row.append(round(numerator / denominator, 6) if denominator else 0.0)

    records = [
        (record, vector) for record in data
        if (vector := _embedding(record)) is not None
    ]
    vectors = [vector for _, vector in records]
    pca = _pca(vectors)
    umap = _umap(vectors)
    points = [{
        "pca": pca[index] if pca else None,
        "umap": umap[index] if umap else None,
        "groups": {
            grouping: _group_value(record, grouping, config)
            for grouping in groupings
        },
    } for index, (record, _) in enumerate(records)]
    embedding_groups = {}
    for grouping in groupings:
        labels = [_group_value(record, grouping, config) for record, _ in records]
        distances = _distance_distributions(vectors, labels)
        embedding_groups[grouping] = {
            "separability": {
                "score": _silhouette(vectors, labels),
                "n_images": len(vectors),
                "n_clusters": len({label for label in labels if label is not None}),
            },
            "distances": distances,
        }
    return {
        "image_features": features,
        "correlation": {"features": list(IMAGE_FEATURES), "matrix": correlation_matrix},
        "embedding": {
            "n_embeddings": len(vectors),
            "group_by": groupings,
            "points": points,
            "projections": {"pca_available": pca is not None, "umap_available": umap is not None},
            "groups": embedding_groups,
        },
    }


def render_chapter_seven(chapter_data: dict[str, Any]) -> str:
    """Render interactive feature and embedding plots."""
    serialized = json.dumps(chapter_data, ensure_ascii=True).replace("</", "<\\/")
    return f"""
  <div class="eyebrow">Chapter 7</div>
  <h2 id="chapter-7">Visual feature and embedding analysis</h2>
  <p class="subtitle">Measurable image properties and semantic representation are
  compared using the same observations and, for embeddings, the same projection.</p>
  <section id="chapter-7-features"><h3>7.1 Image-feature distributions</h3>
    <p>Histograms show overall distributions; box plots compare subgroups, and the
    heatmap shows Pearson correlations for the six available numerical features.</p>
    <div id="chapter-7-feature-controls" class="checkbox-grid"></div>
    <div class="plot-grid"><article class="plot-card"><div id="chapter-7-feature-plot"></div></article>
      <article class="plot-card"><div id="chapter-7-feature-groups"></div></article></div>
    <article class="plot-card"><div id="chapter-7-correlation"></div></article>
  </section>
  <section id="chapter-7-embeddings"><h3>7.2 Embedding distributions and cluster separability</h3>
    <p>PCA is fit once to all embeddings, so comparable groups share coordinates.
    UMAP is shown when available. Silhouette scores and within/between Euclidean
    distances are descriptive and depend on the selected grouping.</p>
    <div id="chapter-7-embedding-controls" class="checkbox-grid"></div>
    <div class="plot-grid"><article class="plot-card"><div id="chapter-7-embedding-plot"></div></article>
      <article class="plot-card"><div id="chapter-7-distance-plot"></div></article></div>
    <div id="chapter-7-separability"></div>
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
      const correlation = chapterSevenData.correlation;
      Plotly.newPlot("chapter-7-correlation", [{{z: correlation.matrix,
        x: correlation.features.map(chapterSevenTitle), y: correlation.features.map(chapterSevenTitle),
        type: "heatmap", zmin: -1, zmax: 1, colorscale: "RdBu", reversescale: true}}],
        {{title: "Feature correlation matrix", margin: {{l: 130, b: 130}}}},
        {{responsive: true, displaylogo: false}});
      const points = chapterSevenData.embedding.points.filter((point) => point[chapterSevenState.projection]);
      const pointLabels = points.map(p => p.groups[chapterSevenState.grouping] || "unlabelled");
      const labelCodes = Object.fromEntries([...new Set(pointLabels)].map((label, index) => [label, index]));
      Plotly.newPlot("chapter-7-embedding-plot", [{{x: points.map(p => p[chapterSevenState.projection][0]),
        y: points.map(p => p[chapterSevenState.projection][1]), mode: "markers", type: "scatter",
        text: pointLabels,
        marker: {{color: pointLabels.map(label => labelCodes[label]), colorscale: "Turbo",
          showscale: false}},
        hovertemplate: "%{{text}}<br>x=%{{x:.3f}}, y=%{{y:.3f}}<extra></extra>"}}],
        {{title: `${{chapterSevenState.projection.toUpperCase()}} projection`}},
        {{responsive: true, displaylogo: false}});
      const distances = chapterSevenData.embedding.groups[chapterSevenState.grouping].distances;
      Plotly.newPlot("chapter-7-distance-plot", [
        {{type: "box", y: distances.within, name: "Within group"}},
        {{type: "box", y: distances.between, name: "Between groups"}}
      ], {{title: "Embedding distance distributions", yaxis: {{title: "Euclidean distance"}}}},
      {{responsive: true, displaylogo: false}});
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
