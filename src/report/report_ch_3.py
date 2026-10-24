"""Renderer for chapter 3: Demographic representation and diversity."""

from __future__ import annotations

import base64
from io import BytesIO
import html
import json
from typing import Any


_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#332288",
    "#117733",
    "#AA4499",
]


def _pyplot() -> Any:
    """Load a non-interactive pyplot backend only when report rendering needs it."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _figure_data_uri(figure: Any, plt: Any) -> str:
    output = BytesIO()
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _distribution_plot(
    dimension: str,
    distribution: dict[str, Any],
    plt: Any,
) -> str | None:
    counts = [
        (str(label), float(count))
        for label, count in (distribution.get("counts") or {}).items()
        if _finite_number(count) and float(count) > 0
    ]
    if not counts:
        return None

    labels, values = zip(*counts)
    figure, axis = plt.subplots(figsize=(5.8, 4.1), dpi=120)
    figure.patch.set_facecolor("#ffffff")
    axis.pie(
        values,
        labels=labels,
        colors=[_PALETTE[index % len(_PALETTE)] for index in range(len(values))],
        autopct=lambda value: f"{value:.1f}%" if value >= 3 else "",
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        textprops={"color": "#17202a", "fontsize": 9},
    )
    axis.set_title("")
    return _figure_data_uri(figure, plt)


def _finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _metric_plot(
    entries: list[tuple[str, Any]],
    metric: str,
    label: str,
    color: str,
    plt: Any,
) -> str | None:
    values = [
        (str(name), float(stats[metric]))
        for name, stats in entries
        if isinstance(stats, dict) and _finite_number(stats.get(metric))
    ]
    if not values:
        return None

    values.sort(key=lambda item: item[1])
    names, numbers = zip(*values)
    height = max(2.2, 0.42 * len(values) + 0.75)
    figure, axis = plt.subplots(figsize=(7.4, height), dpi=120)
    figure.patch.set_facecolor("#ffffff")
    bars = axis.barh(names, numbers, color=color, alpha=0.9)
    axis.set_title(label, loc="left", color="#102a43", pad=10, fontweight="bold")
    axis.grid(axis="x", color="#d9e2ec", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, colors="#334e68")
    axis.tick_params(axis="x", colors="#52606d")
    for bar, number in zip(bars, numbers):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {number:.3f}",
            va="center",
            color="#334e68",
            fontsize=8,
        )
    figure.tight_layout()
    return _figure_data_uri(figure, plt)


def build_demographic_diversity(
    analysis_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Expose precomputed chapter-3 metrics without recomputing them."""
    general = analysis_results.get("general_stats", {})
    modes = config.get("prompt_modes", {})
    mode_labels = (
        {str(key): str(value) for key, value in modes.items()}
        if isinstance(modes, dict)
        else {}
    )
    composition = general.get("demographic_composition", {})
    diversity = general.get("diversity_metrics", {})
    return {
        "dimensions": ["gender", "race", "age", "emotion"],
        "composition": composition if isinstance(composition, dict) else {},
        "diversity": diversity if isinstance(diversity, dict) else {},
        "prompt_modes": sorted(
            diversity.get("available_prompt_modes", mode_labels.values())
            if isinstance(diversity, dict)
            else mode_labels.values()
        ),
    }


def render_chapter_three(chapter_data: dict[str, Any]) -> str:
    """Render chapter 3 charts with pyplot and keep only filtering in the HTML."""
    plt = _pyplot()
    modes = [str(mode) for mode in chapter_data.get("prompt_modes", [])]

    composition_cards: list[str] = []
    composition = chapter_data.get("composition", {})
    for dimension in chapter_data.get("dimensions", []):
        item = composition.get(dimension, {})
        for mode in modes:
            prompt_mode_slices = item.get("by_slice", {}).get("prompt_mode", {})
            slice_data = next(
                (
                    value for key, value in prompt_mode_slices.items()
                    if str(key).lower() == mode.lower()
                ),
                {},
            )
            counts: dict[str, float] = {}
            for label, count in (slice_data.get("detected", {}).get("counts") or {}).items():
                if _finite_number(count):
                    counts[str(label)] = counts.get(str(label), 0) + float(count)
            image = _distribution_plot(dimension, {"counts": counts}, plt)
            if image:
                composition_cards.append(
                    f'<article class="plot-card chapter-3-mode-group" '
                    f'data-modes="{html.escape(mode)}"><h4>{html.escape(dimension.title())}'
                    f' - {html.escape(mode)}</h4><img class="pyplot-figure" src="{image}" alt="'
                    f'{html.escape(dimension)} distribution for {html.escape(mode)}"></article>'
                )
    composition_sections = [
        '<div class="plot-grid">' + "".join(composition_cards) + "</div>"
    ] if composition_cards else []

    diversity = chapter_data.get("diversity", {})
    diversity_sections: list[str] = []
    groups = diversity.get("groups", {}) if isinstance(diversity, dict) else {}
    for group_name, values in groups.items():
        if not isinstance(values, dict):
            continue
        has_mode_breakdown = any(
            isinstance(stats, dict) and isinstance(stats.get("by_prompt_mode"), dict)
            for stats in values.values()
        )
        for mode in modes:
            if group_name == "prompt_mode":
                entries = [
                    (name, stats)
                    for name, stats in values.items()
                    if str(name).lower() == mode.lower()
                ]
            elif not has_mode_breakdown:
                if mode != modes[0]:
                    continue
                entries = [
                    (name, stats)
                    for name, stats in values.items()
                    if isinstance(stats, dict)
                ]
            else:
                entries = [
                    (
                        name,
                        next(
                            (
                                value
                                for key, value in stats.get("by_prompt_mode", {}).items()
                                if str(key).lower() == mode.lower()
                            ),
                            {},
                        ),
                    )
                    for name, stats in values.items()
                    if isinstance(stats, dict)
                ]
            entries = [(name, stats) for name, stats in entries if isinstance(stats, dict)]
            if not entries:
                continue
            # Nested metrics are normalized here so plotting stays in Python.
            normalized_entries = [
                (
                    name,
                    {
                        "embedding_dispersion": stats.get("embedding_dispersion"),
                        "pairwise_similarity": (
                            stats.get("pairwise_similarity") or {}
                        ).get("mean"),
                        "demographic_entropy": (
                            (stats.get("demographic_entropy") or {}).get("gender") or {}
                        ).get("normalized_entropy"),
                        "object_unique": stats.get("object_unique"),
                    },
                )
                for name, stats in entries
            ]
            plots = [
                _metric_plot(normalized_entries, metric, label, color, plt)
                for metric, label, color in (
                    ("embedding_dispersion", "Embedding dispersion", "#0072B2"),
                    ("pairwise_similarity", "Mean pairwise similarity", "#D55E00"),
                    ("demographic_entropy", "Normalized gender entropy", "#009E73"),
                    ("object_unique", "Unique objects", "#E69F00"),
                )
            ]
            plot_html = "".join(
                f'<img class="pyplot-figure" src="{plot}" alt="{html.escape(label)}">'
                for plot, label in zip(
                    plots,
                    (
                        "Embedding dispersion",
                        "Mean pairwise similarity",
                        "Normalized gender entropy",
                        "Unique objects",
                    ),
                )
                if plot
            )
            if plot_html:
                mode_attribute = (
                    f' class="plot-card chapter-3-mode-group" '
                    f'data-modes="{html.escape(mode)}"'
                    if has_mode_breakdown or group_name == "prompt_mode"
                    else ' class="plot-card"'
                )
                diversity_sections.append(
                    f"<article{mode_attribute}><h4>{html.escape(str(group_name).replace('_', ' '))}"
                    f'{" - " + html.escape(mode) if has_mode_breakdown or group_name == "prompt_mode" else ""}'
                    f'</h4><div class="plot-grid">{plot_html}</div></article>'
                )

    rare_entries = [
        (str(name), {"frequency": len(values) if isinstance(values, list) else 0})
        for name, values in (diversity.get("rare_categories", {}) or {}).items()
        if isinstance(values, list) and values
    ]
    rare_plot = _metric_plot(
        rare_entries,
        "frequency",
        "Rare categories by attribute",
        "#332288",
        plt,
    )
    if rare_plot:
        diversity_sections.append(
            '<article class="plot-card"><h4>Rare categories by attribute</h4>'
            '<p class="note">Number of prompt categories occurring exactly once, grouped by attribute.</p>'
            f'<img class="pyplot-figure" src="{rare_plot}" alt="Rare-category frequency"></article>'
        )

    sample_entries = [
        (
            str(entry.get("sample_count")),
            {"dispersion": entry.get("mean_embedding_dispersion")},
        )
        for entry in (diversity.get("by_sample_count", []) or [])
        if isinstance(entry, dict)
    ]
    sample_plot = _metric_plot(
        sample_entries,
        "dispersion",
        "Mean embedding dispersion",
        "#117733",
        plt,
    )
    if sample_plot:
        diversity_sections.append(
            '<article class="plot-card"><h4>Diversity versus sample count</h4>'
            f'<img class="pyplot-figure" src="{sample_plot}" alt="Diversity versus sample count"></article>'
        )

    selector_options = "".join(
        f'<label><input type="checkbox" value="{html.escape(mode)}" '
        f'checked> {html.escape(mode)}</label>'
        for mode in modes
    )
    return f"""
<div id="chapter-container">
  <div class="eyebrow">Chapter 3</div>
  <h2 id="chapter-3">Demographic representation and diversity</h2>
  <p class="subtitle">This chapter shows the detected demographic composition of generated outputs and
  the diversity of those outputs, for each attribute we can detect automatically.</p>
  <section id="chapter-3-controls" aria-labelledby="chapter-3-controls-heading">
    <h3 id="chapter-3-controls-heading">Prompt modes included</h3>
    <p class="note">Select the prompt modes used in the plots below.</p>
    <div id="chapter-3-mode-selector" class="checkbox-grid ">{selector_options}</div>
  </section>
  <section id="chapter-3-distributions" aria-labelledby="chapter-3-distributions-heading">
    <h3 id="chapter-3-distributions-heading">3.1 Demographic distributions</h3>
    <p>These pie charts show the overall detected demographic composition of the generated images.</p>
    <div id="chapter-3-composition">{"".join(composition_sections) or '<p class="note">No detected observations.</p>'}</div>
  </section>
  <section id="chapter-3-diversity" aria-labelledby="chapter-3-diversity-heading">
    <h3 id="chapter-3-diversity-heading">3.2 Diversity</h3>
    <p>Diversity is here measured with 4 different metrics:</p>
    <p>
      <ul>
        <li><b>Embedding dispersion</b>: the average Euclidean distance between every pair of image embeddings in a group.
          The embeddings are feature vectors, derived from CLIP-based image features.
          Higher values mean the images are more different in feature space, suggesting greater visual or semantic variety.</li>
        <li><b>Mean pairwise similarity</b>: the average cosine similarity between every pair of image embeddings.
          Higher values mean the generated images are more alike. Lower values indicate greater variation. This metric therefore has the opposite interpretation from dispersion. </li>
        <li><b>Normalized gender entropy</b>: the diversity of detected gender categories within a group, based on the entropy of the detected gender labels.
          A value near 1 means the observed gender categories are relatively evenly represented. A value near 0 means one category dominates, or only one category is present.</li>
        <li><b>Unique objects</b>: the number of distinct object labels detected by Florence-2 across the images in the group. Labels are normalized to lowercase, trimmed, and deduplicated.
          Higher values mean a broader range of detected object types occurs in the images.</li>
      </ul>
    </p>

    <div id="chapter-3-diversity-scores">{"".join(diversity_sections) or '<p class="note">No diversity observations.</p>'}</div>
  </section>
  <script>
    const chapterThreeGroups = Array.from(document.querySelectorAll(".chapter-3-mode-group"));
    const chapterThreeRender = () => {{
      const selected = new Set(Array.from(document.querySelectorAll(
        "#chapter-3-mode-selector input:checked")).map((input) => input.value));
      chapterThreeGroups.forEach((group) => {{
        group.hidden = !selected.has(group.dataset.modes);
      }});
    }};
    document.querySelectorAll("#chapter-3-mode-selector input").forEach((input) =>
      input.addEventListener("change", chapterThreeRender));
    chapterThreeRender();
  </script>
</div>
"""


def render_chapter_three(chapter_data: dict[str, Any]) -> str:
    """Render one dynamically aggregated chart per chapter-3 view."""
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
    groups = diversity.get("groups", {}) if isinstance(diversity, dict) else {}
    diversity_data: dict[str, dict[str, Any]] = {}
    for group_name, values in groups.items():
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
                        mode_values[str(name)] = stats
                else:
                    mode_stats = next(
                        (value for key, value in stats.get("by_prompt_mode", {}).items()
                         if str(key).lower() == mode.lower()),
                        stats if not stats.get("by_prompt_mode") else {},
                    )
                    if isinstance(mode_stats, dict):
                        mode_values[str(name)] = mode_stats
            diversity_data[group_name][mode] = mode_values

    composition_cards = "".join(
        f'<article class="plot-card"><h4>{html.escape(str(dimension).title())}</h4>'
        f'<div id="chapter-3-composition-{index}" class="chapter-3-chart"></div></article>'
        for index, dimension in enumerate(composition_data)
    )
    diversity_cards = "".join(
        f'<article class="plot-card"><h4>{html.escape(str(group_name).replace("_", " "))}</h4>'
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
<div id="chapter-container">
  <div class="eyebrow">Chapter 3</div>
  <h2 id="chapter-3">Demographic representation and diversity</h2>
  <p class="subtitle">This chapter shows the detected demographic composition of generated outputs and
  the diversity of those outputs, for each attribute we can detect automatically.</p>
  <section id="chapter-3-controls" aria-labelledby="chapter-3-controls-heading">
    <h3 id="chapter-3-controls-heading">Prompt modes included</h3>
    <p class="note">Select the prompt modes used to aggregate each chart below.</p>
    <div id="chapter-3-mode-selector" class="checkbox-grid">{selector_options}</div>
  </section>
  <section id="chapter-3-distributions" aria-labelledby="chapter-3-distributions-heading">
    <h3 id="chapter-3-distributions-heading">3.1 Demographic distributions</h3>
    <div id="chapter-3-composition"><div class="plot-grid">{composition_cards}</div></div>
  </section>
  <section id="chapter-3-diversity" aria-labelledby="chapter-3-diversity-heading">
    <h3 id="chapter-3-diversity-heading">3.2 Diversity</h3>
    <div id="chapter-3-diversity-scores">{diversity_cards}</div>
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
    const chapterThreeMetricValue = (stats, metric) => {{
      if (metric === "pairwise_similarity") {{
        return stats.pairwise_similarity?.mean ?? stats.pairwise_similarity;
      }}
      if (metric === "demographic_entropy") {{
        return stats.demographic_entropy?.gender?.normalized_entropy
          ?? stats.demographic_entropy?.gender?.entropy
          ?? stats.normalized_entropy;
      }}
      return stats[metric];
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
        }}], {{height: 340, margin: {{t: 20, b: 20, l: 20, r: 20}}, showlegend: true}});
      }});
      chapterThreeGroups.forEach((group, groupIndex) => {{
        Object.keys(chapterThreeMetricLabels).forEach((metric) => {{
          const aggregate = {{}};
          Object.entries(chapterThreeDiversity[group]).forEach(([mode, values]) => {{
            if (!selected.has(mode)) return;
            Object.entries(values).forEach(([name, stats]) => {{
              const value = Number(chapterThreeMetricValue(stats, metric));
              if (!Number.isFinite(value)) return;
              const weight = Number(stats.n_images) || 1;
              aggregate[name] = aggregate[name] || {{sum: 0, weight: 0}};
              aggregate[name].sum += value * weight;
              aggregate[name].weight += weight;
            }});
          }});
          const names = Object.keys(aggregate);
          const values = names.map((name) => aggregate[name].sum / aggregate[name].weight);
          const color = chapterThreeMetricColors[metric];
          Plotly.react("chapter-3-diversity-" + groupIndex + "-" + metric, [{{
            x: values, y: names, type: "bar", orientation: "h",
            marker: {{
              color: names.map((name, index) => {{
                const shade = Math.max(0.55, 1 - (index % 5) * 0.1);
                return color + Math.round(shade * 255).toString(16).padStart(2, "0");
              }})
            }}
          }}], {{title: chapterThreeMetricLabels[metric], height: Math.max(220, names.length * 28),
            margin: {{t: 45, b: 45, l: 130, r: 20}}}});
        }});
      }});
    }};
    document.querySelectorAll("#chapter-3-mode-selector input").forEach((input) =>
      input.addEventListener("change", chapterThreeRender));
    chapterThreeRender();
  </script>
</div>
"""
