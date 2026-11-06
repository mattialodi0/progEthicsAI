"""Renderer for chapter 3: Demographic representation and diversity."""

from __future__ import annotations

import html
import json
from typing import Any

import plotly.graph_objects as go

from .explain import chapter_header, findings, guide, metric_defs
from .insights import chapter_three
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
        return '<p class="note">No prompt category occurs in exactly one image, so no group ' \
               'estimate rests on a single generation.</p>'
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


def _detected_counts(slices: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(name): (value or {}).get("detected", {}).get("counts", {})
        for name, value in slices.items() if isinstance(value, dict)
    }


def render_chapter_three(chapter_data: dict[str, Any]) -> str:
    """Render dynamically aggregated composition and diversity charts."""
    modes = [str(mode) for mode in chapter_data.get("prompt_modes", [])]
    composition = chapter_data.get("composition", {})
    composition_data: dict[str, dict[str, dict[str, Any]]] = {}
    context_data: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in chapter_data.get("dimensions", []):
        by_slice = composition.get(dimension, {}).get("by_slice", {})
        slices = by_slice.get("prompt_mode", {})
        composition_data[dimension] = {
            mode: next(
                (value.get("detected", {}).get("counts", {})
                 for key, value in slices.items()
                 if str(key).lower() == mode.lower() and isinstance(value, dict)),
                {},
            )
            for mode in modes
        }
        context_data[dimension] = {
            context: _detected_counts(by_slice.get(context, {}))
            for context in ("job", "workplace") if by_slice.get(context)
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
        f'<article class="plot-card"><h4>Detected {html.escape(str(dimension))}</h4>'
        f'<div id="chapter-3-composition-{index}"></div></article>'
        for index, dimension in enumerate(composition_data)
    )
    selector_options = "".join(
        f'<label><input type="checkbox" value="{html.escape(mode)}" checked> '
        f'{html.escape(mode)}</label>'
        for mode in modes
    )
    group_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.replace("_", " "))}</option>'
        for name in diversity_data
    )
    dimension_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in context_data
    )
    dump = lambda value: json.dumps(value, ensure_ascii=True).replace("</", "<\\/")  # noqa: E731
    metric_table = metric_defs([
        ("Embedding dispersion", "Mean Euclidean distance between every pair of CLIP image "
         "embeddings in the group.", "Higher = more visual / semantic variety. Compare groups "
         "with each other; the absolute value has no natural threshold."),
        ("Mean pairwise similarity", "Mean cosine similarity between every pair of image "
         "embeddings in the group.", "Higher = images look more alike (more repetitive). "
         "Reads in the opposite direction from dispersion."),
        ("Normalized gender entropy", "Shannon entropy of the detected gender distribution "
         "divided by its maximum.", "1 = genders evenly represented; 0 = only one gender "
         "detected. Expected to be ~0 when the prompt requests a gender."),
        ("Unique objects", "Number of distinct object labels detected across the group's "
         "images.", "Higher = richer scenes. Grows with group size, so compare groups of "
         "similar size."),
    ])
    return f"""
  {chapter_header(3, "Demographic representation and diversity",
      "Who does the model generate, and how varied are the images it produces?",
      "<p>Two different questions are kept apart here. <strong>Representation</strong> "
      "(3.1) asks which demographic groups appear in the generated images. "
      "<strong>Diversity</strong> (3.2) asks how varied the images are within each group. "
      "A model can produce a balanced number of men and women while drawing every woman "
      "in the same way — or produce varied images of only one group.</p>")}
  {findings(chapter_three(chapter_data))}
  <div class="controls" id="chapter-3-mode-selector" aria-label="Prompt modes included">
    <strong>Prompt modes included in 3.1 and 3.2:</strong>{selector_options}
  </div>
  <section id="chapter-3-distributions" aria-labelledby="chapter-3-distributions-heading">
    <h3 id="chapter-3-distributions-heading">3.1 Demographic distributions</h3>
    <p>Detected demographic composition of the generated images in each selected prompt
    mode, plus the pooled total of the selection.</p>
    {guide(
        read="Each horizontal bar sums to 100% and is split by the detected category; "
             "hover for counts. Compare the rows: the single-variable row shows the model's "
             "default when the attribute is (mostly) not requested.",
        look=["In single-variable prompts, a demographic far from the population you would "
              "expect (e.g. 90% male for gender-neutral prompts) indicates the model fills "
              "unspecified attributes with a default group.",
              "Categories that never or almost never appear indicate erasure.",
              "In control and double-variable prompts the composition is largely determined "
              "by what was requested; differences from the requested mix are measured in "
              "chapter 4."],
        caution=["Composition must be read against the prompt design: if 70% of prompts "
                 "request one category, a 70% output share is not evidence of bias.",
                 "Labels are DeepFace predictions over a small fixed set of categories "
                 "(e.g. 7 race classes), so they cannot capture the full diversity of "
                 "people and may be systematically wrong for some groups."],
    )}
    <div class="plot-grid">{composition_cards}</div>
  </section>
  <section id="chapter-3-context" aria-labelledby="chapter-3-context-heading">
    <h3 id="chapter-3-context-heading">3.1.1 Demographic composition by job and workplace</h3>
    <p>The detected composition for each job or workplace, with all prompt modes pooled
    (explicit gender, race or age requests pull the bars toward balance, so real skews
    are probably a bit stronger). Jobs dominated by one group, or opposite skews for
    related jobs, are what to look for.</p>
    <div class="controls">
      <label>Detected attribute <select id="chapter-3-context-dimension">{dimension_options}</select></label>
      <label>Split by <select id="chapter-3-context-split">
        <option value="job">Job</option><option value="workplace">Workplace</option></select></label>
    </div>
    <article class="plot-card"><div id="chapter-3-context-plot"></div></article>
  </section>
  <section id="chapter-3-diversity" aria-labelledby="chapter-3-diversity-heading">
    <h3 id="chapter-3-diversity-heading">3.2 Diversity</h3>
    <p>Diversity of the generated images within each group, measured four ways. When several
    prompt modes are selected, each value is the image-weighted mean of its per-mode values.</p>
    {metric_table}
    {guide(
        read="Each panel ranks the groups of the selected grouping on one metric; the dashed "
             "line is the image-weighted mean of the groups shown. Hover for the number of "
             "images.",
        look=["A group with clearly lower dispersion (or higher similarity) than comparable "
              "groups is depicted more uniformly — a sign of a narrow, possibly stereotyped "
              "visual template.",
              "Gender entropy near 0 for a job whose prompt did not specify gender means one "
              "gender is drawn almost always."],
        caution="Higher diversity is not automatically better: a prompt that specifies more "
                "attributes (control mode) should produce less variety than an open prompt, "
                "and some subjects (e.g. a soldier's uniform) are visually constrained.",
    )}
    <div class="controls">
      <label>Group images by <select id="chapter-3-diversity-group">{group_options}</select></label>
    </div>
    <div class="plot-grid">
      <article class="plot-card"><div id="chapter-3-diversity-embedding_dispersion"></div></article>
      <article class="plot-card"><div id="chapter-3-diversity-pairwise_similarity"></div></article>
      <article class="plot-card"><div id="chapter-3-diversity-demographic_entropy"></div></article>
      <article class="plot-card"><div id="chapter-3-diversity-object_unique"></div></article>
    </div>
    <details><summary>Robustness checks: rare categories and sample size</summary>
      <div class="plot-grid">
        <article class="plot-card"><h4>Rare categories by attribute</h4>
          <p class="note">Prompt categories occurring in exactly one image (all prompt
          modes). Estimates for these groups rest on a single generation.</p>
          {_rare_chart(diversity.get("rare_categories") or {})}</article>
        <article class="plot-card"><h4>Diversity versus sample count</h4>
          <p class="note">Mean embedding dispersion of the prompt-mode groups with at least
          the given number of images. A flat line means dispersion is not driven by group
          size.</p>
          {_sample_count_chart(diversity.get("by_sample_count") or [])}</article>
      </div>
    </details>
  </section>
  <script>
    const chapterThreeComposition = {dump(composition_data)};
    const chapterThreeContext = {dump(context_data)};
    const chapterThreeDiversity = {dump(diversity_data)};
    const chapterThreeDimensions = {dump(list(composition_data))};
    const chapterThreeMetricLabels = {{
      embedding_dispersion: "Embedding dispersion (higher = more varied)",
      pairwise_similarity: "Mean pairwise similarity (higher = more alike)",
      demographic_entropy: "Normalized gender entropy (1 = balanced)",
      object_unique: "Unique objects"
    }};
    const chapterThreeMetricColors = {{
      embedding_dispersion: "#0072B2", pairwise_similarity: "#D55E00",
      demographic_entropy: "#009E73", object_unique: "#E69F00"
    }};
    const chapterThreePalette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#8C6D31", "#7F7F7F", "#332288", "#B8336A"];
    // One stable colour per category, shared by every chart of the chapter.
    const chapterThreeColors = {{}};
    Object.entries(chapterThreeComposition).forEach(([dimension, byMode]) => {{
      const categories = new Set();
      Object.values(byMode).forEach((counts) => Object.keys(counts).forEach((c) => categories.add(c)));
      Object.values(chapterThreeContext[dimension] || {{}}).forEach((split) =>
        Object.values(split).forEach((counts) => Object.keys(counts).forEach((c) => categories.add(c))));
      chapterThreeColors[dimension] = {{}};
      Array.from(categories).sort().forEach((category, index) =>
        chapterThreeColors[dimension][category] = chapterThreePalette[index % chapterThreePalette.length]);
    }});
    // rows: [[label, counts]] -> 100% stacked horizontal bars.
    const chapterThreeStacked = (element, dimension, rows, height) => {{
      const categories = Object.keys(chapterThreeColors[dimension] || {{}});
      const totals = rows.map(([, counts]) =>
        Object.values(counts).reduce((sum, value) => sum + Number(value), 0));
      const labels = rows.map(([name], index) => `${{name}} (n=${{totals[index]}})`);
      const traces = categories.map((category) => ({{
        type: "bar", orientation: "h", name: category,
        y: labels,
        x: rows.map(([, counts], index) =>
          totals[index] ? 100 * Number(counts[category] || 0) / totals[index] : 0),
        customdata: rows.map(([, counts]) => Number(counts[category] || 0)),
        marker: {{color: chapterThreeColors[dimension][category]}},
        hovertemplate: `%{{y}}<br>${{category}}: %{{x:.1f}}% (%{{customdata}} faces)<extra></extra>`
      }})).filter((trace) => trace.customdata.some((value) => value > 0));
      Plotly.react(element, traces, {{
        barmode: "stack", height: height || Math.max(240, 120 + 34 * rows.length),
        margin: {{t: 40, b: 45, l: 10, r: 10}},
        xaxis: {{title: "Share of detected faces (%)", range: [0, 100]}},
        yaxis: {{autorange: "reversed"}},
        legend: {{orientation: "h", x: 0, y: 1.02, yanchor: "bottom", traceorder: "normal"}}
      }});
    }};
    const chapterThreeSelected = () => new Set(Array.from(document.querySelectorAll(
      "#chapter-3-mode-selector input:checked")).map((input) => input.value));
    const chapterThreeRenderComposition = () => {{
      const selected = chapterThreeSelected();
      chapterThreeDimensions.forEach((dimension, index) => {{
        const rows = [];
        const pooled = {{}};
        Object.entries(chapterThreeComposition[dimension]).forEach(([mode, counts]) => {{
          if (!selected.has(mode)) return;
          rows.push([mode, counts]);
          Object.entries(counts).forEach(([label, value]) =>
            pooled[label] = (pooled[label] || 0) + Number(value));
        }});
        if (rows.length > 1) rows.push(["All selected", pooled]);
        chapterThreeStacked("chapter-3-composition-" + index, dimension, rows);
      }});
    }};
    const chapterThreeRenderContext = () => {{
      const dimension = document.getElementById("chapter-3-context-dimension").value;
      const split = document.getElementById("chapter-3-context-split").value;
      const data = (chapterThreeContext[dimension] || {{}})[split] || {{}};
      const first = Object.keys(chapterThreeColors[dimension] || {{}})[0];
      const share = (counts) => {{
        const total = Object.values(counts).reduce((sum, value) => sum + Number(value), 0);
        return total ? Number(counts[first] || 0) / total : 0;
      }};
      const rows = Object.entries(data).sort((left, right) => share(right[1]) - share(left[1]));
      const element = document.getElementById("chapter-3-context-plot");
      if (!rows.length) {{
        element.innerHTML = '<p class="note">No composition is available for this split.</p>';
        return;
      }}
      chapterThreeStacked(element, dimension, rows);
    }};
    const chapterThreeRenderDiversity = () => {{
      const selected = chapterThreeSelected();
      const group = document.getElementById("chapter-3-diversity-group").value;
      Object.keys(chapterThreeMetricLabels).forEach((metric) => {{
        const aggregate = {{}};
        Object.entries(chapterThreeDiversity[group] || {{}}).forEach(([mode, values]) => {{
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
        const entries = Object.entries(aggregate)
          .map(([name, item]) => [name, item.sum / item.weight, item.weight])
          .sort((left, right) => left[1] - right[1]);
        const element = "chapter-3-diversity-" + metric;
        const totalWeight = entries.reduce((sum, entry) => sum + entry[2], 0);
        const mean = totalWeight ?
          entries.reduce((sum, entry) => sum + entry[1] * entry[2], 0) / totalWeight : null;
        Plotly.react(element, [{{
          x: entries.map((entry) => entry[1]), y: entries.map((entry) => entry[0]),
          customdata: entries.map((entry) => entry[2]),
          type: "bar", orientation: "h",
          marker: {{color: chapterThreeMetricColors[metric]}},
          hovertemplate: "%{{y}}: %{{x:.3f}}<br>%{{customdata}} images<extra></extra>"
        }}], {{
          title: chapterThreeMetricLabels[metric],
          height: Math.max(240, 80 + entries.length * 24),
          margin: {{t: 40, b: 40, l: 10, r: 20}},
          shapes: mean === null ? [] : [{{type: "line", x0: mean, x1: mean, y0: 0, y1: 1,
            yref: "paper", line: {{dash: "dash", color: "#52606d", width: 1.5}}}}]
        }});
      }});
    }};
    const chapterThreeRender = () => {{
      chapterThreeRenderComposition();
      chapterThreeRenderDiversity();
    }};
    document.querySelectorAll("#chapter-3-mode-selector input").forEach((input) =>
      input.addEventListener("change", chapterThreeRender));
    document.getElementById("chapter-3-diversity-group")
      .addEventListener("change", chapterThreeRenderDiversity);
    ["chapter-3-context-dimension", "chapter-3-context-split"].forEach((id) =>
      document.getElementById(id).addEventListener("change", chapterThreeRenderContext));
    chapterThreeRender();
    chapterThreeRenderContext();
  </script>
"""
