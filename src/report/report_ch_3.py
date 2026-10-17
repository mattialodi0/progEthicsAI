"""Renderer for chapter 3: Demographic representation and diversity."""

from __future__ import annotations

from typing import Any


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
    """Render detected distributions and SVG diversity plots."""
    import json

    serialized = json.dumps(chapter_data, ensure_ascii=True).replace("</", "<\\/")
    return f"""
  <div class="eyebrow">Chapter 3</div>
  <h2 id="chapter-3">Demographic representation and diversity</h2>
  <p class="subtitle">Who the model generates and how varied those outputs are are
  reported separately.</p>
  <section id="chapter-3-controls" aria-labelledby="chapter-3-controls-heading">
    <h3 id="chapter-3-controls-heading">Prompt modes included</h3>
    <p class="note">Select the prompt modes used in the detected distributions and
    diversity plots. Metrics are precomputed by the analysis pipeline.</p>
    <div id="chapter-3-mode-selector" class="checkbox-grid"></div>
  </section>
  <section id="chapter-3-distributions" aria-labelledby="chapter-3-distributions-heading">
    <h3 id="chapter-3-distributions-heading">3.1 Demographic distributions</h3>
    <p>These plots show detected output composition only. Prompt specifications are
    excluded because requested and detected observations do not share the same base.</p>
    <div id="chapter-3-composition"></div>
  </section>
  <section id="chapter-3-diversity" aria-labelledby="chapter-3-diversity-heading">
    <h3 id="chapter-3-diversity-heading">3.2 Diversity</h3>
    <p>Each metric is shown as a compact, consistently scaled bar chart. Empty
    metrics and groups are omitted rather than rendered as misleading zeroes.</p>
    <div id="chapter-3-diversity-scores"></div>
    <div id="chapter-3-rare-categories"></div>
    <div id="chapter-3-sample-count"></div>
  </section>
  <script>
    const chapterThreeData = {serialized};
    const chapterThreePalette = [
      "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#F0E442", "#332288", "#117733", "#AA4499"
    ];
    const chapterThreeNumber = (value) =>
      value === null || value === undefined ? "—" : Number(value).toFixed(3);
    const chapterThreeSelectedModes = () => new Set(Array.from(document.querySelectorAll(
      "#chapter-3-mode-selector input:checked")).map((input) => input.value));
    const chapterThreeCombineDetected = (slices) => {{
      const counts = {{}};
      let total = 0;
      slices.forEach((slice) => Object.entries(
        (slice.detected || {{}}).counts || {{}}).forEach(([label, count]) => {{
          counts[label] = (counts[label] || 0) + Number(count);
          total += Number(count);
        }}));
      return {{counts, percentages: Object.fromEntries(Object.entries(counts).map(
        ([label, count]) => [label, total ? 100 * count / total : 0]))}};
    }};
    const chapterThreeComposition = (dimension, distribution) => {{
      const counts = Object.entries(distribution.counts || {{}})
        .filter(([, count]) => Number(count) > 0);
      const total = counts.reduce((sum, [, count]) => sum + Number(count), 0);
      if (!total) return '<p class="note">No detected observations.</p>';
      const values = counts.map(([label, count]) => [label, 100 * Number(count) / total]);
      const gradient = values.map(([, value], index) => {{
        const start = values.slice(0, index).reduce((sum, [, item]) => sum + item, 0);
        return `${{chapterThreePalette[index % chapterThreePalette.length]}} ${{start}}% ${{start + value}}%`;
      }}).join(", ");
      const legend = values.map(([label, value], index) =>
        `<span class="plot-legend"><i style="background:${{
          chapterThreePalette[index % chapterThreePalette.length]}}"></i>${{label}}
        (${{value.toFixed(1)}}%)</span>`).join("");
      return `<div class="pie-layout"><div class="pie-chart" style="background:conic-gradient(${{
        gradient}})" role="img" aria-label="${{dimension}} distribution"></div>
        <div>${{legend}}</div></div>`;
    }};
    const chapterThreeMetricSvg = (entries, metric, label, color) => {{
      const values = entries.map(([name, stats]) => [
        name, Number(stats[metric])]).filter(([, value]) => Number.isFinite(value));
      if (!values.length) return "";
      const max = Math.max(...values.map(([, value]) => value), 1);
      const rowHeight = 38;
      const chartWidth = 720;
      const chartHeight = Math.max(70, values.length * rowHeight + 20);
      const bars = values.map(([name, value], index) => {{
        const y = index * rowHeight + 8;
        const barWidth = chartWidth * value / max;
        return `<text x="0" y="${{y + 17}}" class="svg-label">${{name}}</text>
          <rect x="155" y="${{y}}" width="${{barWidth}}" height="22"
            fill="${{color}}"><title>${{name}}: ${{chapterThreeNumber(value)}}</title></rect>
          <text x="${{Math.min(160 + barWidth, 690)}}" y="${{y + 17}}"
            class="svg-value">${{chapterThreeNumber(value)}}</text>`;
      }}).join("");
      return `<figure class="svg-figure"><figcaption><strong>${{label}}</strong></figcaption>
        <svg viewBox="0 0 760 ${{chartHeight}}" role="img" aria-label="${{label}}">
          ${{bars || '<text x="0" y="20">No observations</text>'}}</svg></figure>`;
    }};
    const chapterThreeMetricPlot = (entries, metric, label, color) =>
      chapterThreeMetricSvg(entries, metric, label, color);
    function renderChapterThree() {{
      const selected = chapterThreeSelectedModes();
      document.getElementById("chapter-3-composition").innerHTML =
        chapterThreeData.dimensions.map((dimension) => {{
          const item = chapterThreeData.composition[dimension] || {{}};
          const slices = Object.entries(item.by_slice?.prompt_mode || {{}})
            .filter(([mode]) => selected.has(mode)).map(([, value]) => value);
          const distribution = chapterThreeCombineDetected(slices);
          return `<article class="plot-card"><h4>${{dimension}}</h4>
            ${{chapterThreeComposition(dimension, distribution)}}
            </article>`;
        }}).join("");

      const groups = chapterThreeData.diversity.groups || {{}};
      document.getElementById("chapter-3-diversity-scores").innerHTML =
        Object.entries(groups).map(([type, values]) => {{
          const entries = Object.entries(values).filter(([name]) =>
            type !== "prompt_mode" || selected.has(name));
          const diversityEntries = entries.map(([name, stats]) => [name, {{
            dispersion: stats.embedding_dispersion,
            similarity: stats.pairwise_similarity?.mean,
            entropy: stats.demographic_entropy?.gender?.normalized_entropy,
            captions: stats.caption_unique,
            objects: stats.object_unique
          }}]);
          const plots = [
            chapterThreeMetricPlot(diversityEntries, "dispersion",
              "Embedding dispersion", "#0072B2"),
            chapterThreeMetricPlot(diversityEntries, "similarity",
              "Mean pairwise similarity", "#D55E00"),
            chapterThreeMetricPlot(diversityEntries, "entropy",
              "Normalized gender entropy", "#009E73"),
            chapterThreeMetricPlot(diversityEntries, "captions",
              "Unique captions", "#CC79A7"),
            chapterThreeMetricPlot(diversityEntries, "objects",
              "Unique objects", "#E69F00")
          ].filter(Boolean).join("");
          return plots ? `<article class="plot-card"><h4>${{type.replaceAll("_", " ")}}</h4>
            <div class="plot-grid">
              ${{plots}}
            </div></article>` : "";
        }}).join("") || '<p class="note">No diversity observations.</p>';

      const rare = chapterThreeData.diversity.rare_categories || {{}};
      const rareEntries = Object.entries(rare).map(([name, values]) =>
        [name, {{frequency: Array.isArray(values) ? values.length : 0}}])
        .filter(([, stats]) => stats.frequency > 0);
      document.getElementById("chapter-3-rare-categories").innerHTML =
        rareEntries.length ? `<article class="plot-card"><h4>Rare-category frequency</h4>
        ${{chapterThreeMetricPlot(rareEntries, "frequency",
          "Categories occurring once", "#332288")}}</article>` : "";
      const sampleCounts = chapterThreeData.diversity.by_sample_count || [];
      document.getElementById("chapter-3-sample-count").innerHTML = sampleCounts.length ?
        `<article class="plot-card"><h4>Diversity versus sample count</h4>
        ${{chapterThreeMetricPlot(sampleCounts.map((entry) => [
          String(entry.sample_count), {{dispersion: entry.mean_embedding_dispersion}}]),
          "dispersion", "Mean embedding dispersion", "#117733")}}</article>` : "";
    }}
    chapterThreeData.prompt_modes.forEach((mode, index) => {{
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${{mode}}" ${{index <= 3 ? "checked" : ""}}> ${{mode}}`;
      label.querySelector("input").addEventListener("change", renderChapterThree);
      document.getElementById("chapter-3-mode-selector").appendChild(label);
    }});
  </script>
"""
