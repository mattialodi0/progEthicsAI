"""Renderer for chapter 3: Demographic representation and diversity."""

from __future__ import annotations

import html
import json
from typing import Any

import plotly.graph_objects as go

from .plotly_helpers import chart, numeric


def _metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """Keep only the numbers the charts use (the artifact also holds every
    pairwise similarity, which would bloat the page)."""
    gender_entropy = (stats.get("demographic_entropy") or {}).get("gender") or {}
    return {
        "n_images": stats.get("n_images"),
        "embedding_dispersion": stats.get("embedding_dispersion"),
        "pairwise_similarity": (stats.get("pairwise_similarity") or {}).get("mean"),
        "demographic_entropy": gender_entropy.get("normalized_entropy"),
        "object_unique": stats.get("object_unique"),
    }


def _mode_stats(stats: dict[str, Any], mode: str) -> dict[str, Any] | None:
    by_mode = stats.get("by_prompt_mode")
    if not by_mode:
        return stats
    return next(
        (value for key, value in by_mode.items()
         if str(key).lower() == mode.lower() and isinstance(value, dict)),
        None,
    )


def _rare_chart(rare: dict[str, Any]) -> str:
    entries = [(str(name), len(values)) for name, values in rare.items()
               if isinstance(values, list) and values]
    if not entries:
        return '<p class="note">No prompt category occurs exactly once.</p>'
    figure = go.Figure(go.Bar(
        x=[count for _, count in entries], y=[name for name, _ in entries], orientation="h",
        customdata=[", ".join(str(item.get("value")) for item in rare[name]) for name, _ in entries],
        hovertemplate="%{y}: %{x}<br>%{customdata}<extra></extra>",
    ))
    figure.update_xaxes(title="Categories occurring exactly once")
    return chart(figure, title="", height=max(240, 60 + 40 * len(entries)))


def _sample_count_chart(rows: list[dict[str, Any]]) -> str:
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return '<p class="note">Not enough prompt-mode groups for this comparison.</p>'
    figure = go.Figure(go.Scatter(
        x=[row.get("sample_count") for row in rows],
        y=[numeric(row.get("mean_embedding_dispersion")) for row in rows],
        mode="lines+markers",
        hovertemplate="Groups with at least %{x} images<br>Mean dispersion: %{y:.3f}<extra></extra>",
    ))
    figure.update_xaxes(title="Minimum images per prompt-mode group", type="category")
    figure.update_yaxes(title="Mean embedding dispersion")
    return chart(figure, title="", height=320)


def render_chapter_three(chapter_data: dict[str, Any]) -> str:
    """Render dynamically aggregated composition and diversity charts."""
    modes = [str(mode) for mode in chapter_data.get("prompt_modes", [])]
    composition = chapter_data.get("composition", {})
    composition_data: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in chapter_data.get("dimensions", []):
        slices = composition.get(dimension, {}).get("by_slice", {}).get("prompt_mode", {})
        composition_data[dimension] = {
            mode: next(
                (value.get("detected", {}).get("counts", {})
                 for key, value in slices.items()
                 if str(key).lower() == mode.lower() and isinstance(value, dict)),
                {},
            )
            for mode in modes
        }

    diversity = chapter_data.get("diversity", {})
    if not isinstance(diversity, dict):
        diversity = {}
    diversity_data: dict[str, dict[str, Any]] = {}
    for group_name, values in (diversity.get("groups") or {}).items():
        if not isinstance(values, dict):
            continue
        diversity_data[group_name] = {}
        for mode in modes:
            mode_values: dict[str, Any] = {}
            for name, stats in values.items():
                if not isinstance(stats, dict):
                    continue
                if group_name == "prompt_mode":
                    if str(name).lower() == mode.lower():
                        mode_values[str(name)] = _metrics(stats)
                else:
                    mode_stats = _mode_stats(stats, mode)
                    if mode_stats is not None:
                        mode_values[str(name)] = _metrics(mode_stats)
            diversity_data[group_name][mode] = mode_values

    composition_cards = "".join(
        f'<article class="plot-card"><h4>{html.escape(str(dimension).title())}</h4>'
        f'<div id="chapter-3-composition-{index}" class="chapter-3-chart"></div></article>'
        for index, dimension in enumerate(composition_data)
    )
    diversity_cards = "".join(
        f'<article class="plot-card"><h4>{html.escape(str(group_name).replace("_", " ").title())}</h4>'
        + '<div class="plot-grid">'
        + "".join(
            f'<div id="chapter-3-diversity-{index}-{metric}" class="chapter-3-chart"></div>'
            for metric in ("embedding_dispersion", "pairwise_similarity",
                           "demographic_entropy", "object_unique")
        )
        + "</div></article>"
        for index, group_name in enumerate(diversity_data)
    )
    selector_options = "".join(
        f'<label><input type="checkbox" value="{html.escape(mode)}" checked> '
        f'{html.escape(mode)}</label>'
        for mode in modes
    )
    composition_json = json.dumps(composition_data, ensure_ascii=True).replace("</", "<\\/")
    diversity_json = json.dumps(diversity_data, ensure_ascii=True).replace("</", "<\\/")
    dimensions_json = json.dumps(list(composition_data), ensure_ascii=True)
    groups_json = json.dumps(list(diversity_data), ensure_ascii=True)
    return f"""
  <div class="eyebrow">Chapter 3</div>
  <h2 id="chapter-3">Demographic representation and diversity</h2>
  <p class="subtitle">This chapter shows the detected demographic composition of generated outputs and
  the diversity of those outputs, for each attribute we can detect automatically.</p>
  <section id="chapter-3-controls" aria-labelledby="chapter-3-controls-heading">
    <h3 id="chapter-3-controls-heading">Prompt modes included</h3>
    <p class="note">Select the prompt modes used to aggregate the charts below.</p>
    <div id="chapter-3-mode-selector" class="checkbox-grid">{selector_options}</div>
  </section>
  <section id="chapter-3-distributions" aria-labelledby="chapter-3-distributions-heading">
    <h3 id="chapter-3-distributions-heading">3.1 Demographic distributions</h3>
    <p>These pie charts show the detected demographic composition of the generated
    images in the selected prompt modes.</p>
    <div id="chapter-3-composition"><div class="plot-grid">{composition_cards}</div></div>
  </section>
  <section id="chapter-3-diversity" aria-labelledby="chapter-3-diversity-heading">
    <h3 id="chapter-3-diversity-heading">3.2 Diversity</h3>
    <p>Diversity is measured with four metrics. When several prompt modes are
    selected, each group's value is the image-weighted mean of its per-mode values.</p>
    <ul>
      <li><b>Embedding dispersion</b>: the average Euclidean distance between every pair of
        CLIP image embeddings in a group. Higher values mean the images are more different in
        feature space, suggesting greater visual or semantic variety.</li>
      <li><b>Mean pairwise similarity</b>: the average cosine similarity between every pair of
        image embeddings. Higher values mean the images are more alike, so it reads in the
        opposite direction from dispersion.</li>
      <li><b>Normalized gender entropy</b>: how evenly the detected gender categories are
        represented in a group. Near 1 means evenly represented; near 0 means one category
        dominates or only one is present.</li>
      <li><b>Unique objects</b>: the number of distinct object labels detected by Florence-2 across
        the images in the group (lowercased, trimmed and deduplicated).</li>
    </ul>
    <div id="chapter-3-diversity-scores">{diversity_cards}</div>
    <div class="plot-grid">
      <article class="plot-card"><h4>Rare categories by attribute</h4>
        <p class="note">Number of prompt categories occurring in exactly one image
        (all prompt modes). Hover to see them.</p>
        {_rare_chart(diversity.get("rare_categories") or {})}</article>
      <article class="plot-card"><h4>Diversity versus sample count</h4>
        <p class="note">Mean embedding dispersion of the prompt-mode groups with at least the
        given number of images (all prompt modes), to check that dispersion is not driven by
        group size.</p>
        {_sample_count_chart(diversity.get("by_sample_count") or [])}</article>
    </div>
  </section>
  <script>
    const chapterThreeComposition = {composition_json};
    const chapterThreeDiversity = {diversity_json};
    const chapterThreeDimensions = {dimensions_json};
    const chapterThreeGroups = {groups_json};
    const chapterThreeMetricLabels = {{
      embedding_dispersion: "Embedding dispersion",
      pairwise_similarity: "Mean pairwise similarity",
      demographic_entropy: "Normalized gender entropy",
      object_unique: "Unique objects"
    }};
    const chapterThreeMetricColors = {{
      embedding_dispersion: "#0072B2",
      pairwise_similarity: "#D55E00",
      demographic_entropy: "#009E73",
      object_unique: "#E69F00"
    }};
    const chapterThreeRender = () => {{
      const selected = new Set(Array.from(document.querySelectorAll(
        "#chapter-3-mode-selector input:checked")).map((input) => input.value));
      chapterThreeDimensions.forEach((dimension, index) => {{
        const counts = {{}};
        Object.entries(chapterThreeComposition[dimension]).forEach(([mode, values]) => {{
          if (selected.has(mode)) Object.entries(values).forEach(([label, value]) =>
            counts[label] = (counts[label] || 0) + Number(value));
        }});
        Plotly.react("chapter-3-composition-" + index, [{{
          labels: Object.keys(counts), values: Object.values(counts), type: "pie"
        }}], {{height: 340, margin: {{t: 20, b: 20, l: 20, r: 20}}, showlegend: true}},
        {{responsive: true, displaylogo: false}});
      }});
      chapterThreeGroups.forEach((group, groupIndex) => {{
        Object.keys(chapterThreeMetricLabels).forEach((metric) => {{
          const aggregate = {{}};
          Object.entries(chapterThreeDiversity[group]).forEach(([mode, values]) => {{
            if (!selected.has(mode)) return;
            Object.entries(values).forEach(([name, stats]) => {{
              const raw = stats[metric];
              if (raw === null || raw === undefined) return;
              const value = Number(raw);
              if (!Number.isFinite(value)) return;
              const weight = Number(stats.n_images) || 1;
              aggregate[name] = aggregate[name] || {{sum: 0, weight: 0}};
              aggregate[name].sum += value * weight;
              aggregate[name].weight += weight;
            }});
          }});
          const names = Object.keys(aggregate);
          const values = names.map((name) => aggregate[name].sum / aggregate[name].weight);
          Plotly.react("chapter-3-diversity-" + groupIndex + "-" + metric, [{{
            x: values, y: names, type: "bar", orientation: "h",
            marker: {{color: chapterThreeMetricColors[metric]}},
            hovertemplate: "%{{y}}: %{{x:.3f}}<extra></extra>"
          }}], {{title: chapterThreeMetricLabels[metric], height: Math.max(220, names.length * 28),
            margin: {{t: 45, b: 45, l: 130, r: 20}}}}, {{responsive: true, displaylogo: false}});
        }});
      }});
    }};
    document.querySelectorAll("#chapter-3-mode-selector input").forEach((input) =>
      input.addEventListener("change", chapterThreeRender));
    chapterThreeRender();
  </script>
"""
