    const chapterThreeMetricLabels = {
      embedding_dispersion: "Embedding dispersion (higher = more varied)",
      pairwise_similarity: "Mean pairwise similarity (higher = more alike)",
      demographic_entropy: "Normalized gender entropy (1 = balanced)",
      object_unique: "Unique objects"
    };
    const chapterThreeMetricColors = {
      embedding_dispersion: "#0072B2", pairwise_similarity: "#D55E00",
      demographic_entropy: "#009E73", object_unique: "#E69F00"
    };
    const chapterThreePalette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#8C6D31", "#7F7F7F", "#332288", "#B8336A"];
    // One stable colour per category, shared by every chart of the chapter.
    const chapterThreeColors = {};
    Object.entries(chapterThreeComposition).forEach(([dimension, byMode]) => {
      const categories = new Set();
      Object.values(byMode).forEach((counts) => Object.keys(counts).forEach((c) => categories.add(c)));
      Object.values(chapterThreeContext[dimension] || {}).forEach((split) =>
        Object.values(split).forEach((counts) => Object.keys(counts).forEach((c) => categories.add(c))));
      chapterThreeColors[dimension] = {};
      Array.from(categories).sort().forEach((category, index) =>
        chapterThreeColors[dimension][category] = chapterThreePalette[index % chapterThreePalette.length]);
    });
    // rows: [[label, counts]] -> 100% stacked horizontal bars.
    const chapterThreeStacked = (element, dimension, rows, height) => {
      const categories = Object.keys(chapterThreeColors[dimension] || {});
      const totals = rows.map(([, counts]) =>
        Object.values(counts).reduce((sum, value) => sum + Number(value), 0));
      const labels = rows.map(([name], index) => `${name} (n=${totals[index]})`);
      const traces = categories.map((category) => {
        const shares = rows.map(([, counts], index) =>
          totals[index] ? 100 * Number(counts[category] || 0) / totals[index] : 0);
        return {
          type: "bar", orientation: "h", name: category,
          y: labels,
          x: shares,
          // Label segments wide enough to hold the text; hover gives the rest.
          text: shares.map((share) => share >= 8 ? share.toFixed(0) + "%" : ""),
          textposition: "inside", insidetextanchor: "middle", textangle: 0,
          textfont: {size: 11},
          customdata: rows.map(([, counts]) => Number(counts[category] || 0)),
          marker: {color: chapterThreeColors[dimension][category]},
          hovertemplate: `%{y}<br>${category}: %{x:.1f}% (%{customdata} faces)<extra></extra>`
        };
      }).filter((trace) => trace.customdata.some((value) => value > 0));
      Plotly.react(element, traces, {
        barmode: "stack", height: height || Math.max(240, 120 + 34 * rows.length),
        uniformtext: {mode: "hide", minsize: 9},
        margin: {t: 40, b: 45, l: 10, r: 10},
        xaxis: {title: "Share of detected faces (%)", range: [0, 100]},
        yaxis: {autorange: "reversed"},
        legend: {orientation: "h", x: 0, y: 1.02, yanchor: "bottom", traceorder: "normal"}
      });
    };
    const chapterThreeSelected = () => new Set(Array.from(document.querySelectorAll(
      "#chapter-3-mode-selector input:checked")).map((input) => input.value));
    const chapterThreeRenderComposition = () => {
      const selected = chapterThreeSelected();
      chapterThreeDimensions.forEach((dimension, index) => {
        const rows = [];
        const pooled = {};
        Object.entries(chapterThreeComposition[dimension]).forEach(([mode, counts]) => {
          if (!selected.has(mode)) return;
          rows.push([mode, counts]);
          Object.entries(counts).forEach(([label, value]) =>
            pooled[label] = (pooled[label] || 0) + Number(value));
        });
        if (rows.length > 1) rows.push(["All selected", pooled]);
        chapterThreeStacked("chapter-3-composition-" + index, dimension, rows);
      });
    };
    const chapterThreeRenderContext = () => {
      const dimension = document.getElementById("chapter-3-context-dimension").value;
      const split = document.getElementById("chapter-3-context-split").value;
      const data = (chapterThreeContext[dimension] || {})[split] || {};
      const first = Object.keys(chapterThreeColors[dimension] || {})[0];
      const share = (counts) => {
        const total = Object.values(counts).reduce((sum, value) => sum + Number(value), 0);
        return total ? Number(counts[first] || 0) / total : 0;
      };
      const rows = Object.entries(data).sort((left, right) => share(right[1]) - share(left[1]));
      const element = document.getElementById("chapter-3-context-plot");
      if (!rows.length) {
        element.innerHTML = '<p class="note">No composition is available for this split.</p>';
        return;
      }
      chapterThreeStacked(element, dimension, rows);
    };
    const chapterThreeRenderDiversity = () => {
      const selected = chapterThreeSelected();
      const group = document.getElementById("chapter-3-diversity-group").value;
      Object.keys(chapterThreeMetricLabels).forEach((metric) => {
        const aggregate = {};
        Object.entries(chapterThreeDiversity[group] || {}).forEach(([mode, values]) => {
          if (!selected.has(mode)) return;
          Object.entries(values).forEach(([name, stats]) => {
            const raw = stats[metric];
            if (raw === null || raw === undefined) return;
            const value = Number(raw);
            if (!Number.isFinite(value)) return;
            const weight = Number(stats.n_images) || 1;
            aggregate[name] = aggregate[name] || {sum: 0, weight: 0};
            aggregate[name].sum += value * weight;
            aggregate[name].weight += weight;
          });
        });
        const entries = Object.entries(aggregate)
          .map(([name, item]) => [name, item.sum / item.weight, item.weight])
          .sort((left, right) => left[1] - right[1]);
        const element = "chapter-3-diversity-" + metric;
        const totalWeight = entries.reduce((sum, entry) => sum + entry[2], 0);
        const mean = totalWeight ?
          entries.reduce((sum, entry) => sum + entry[1] * entry[2], 0) / totalWeight : null;
        Plotly.react(element, [{
          x: entries.map((entry) => entry[1]), y: entries.map((entry) => entry[0]),
          customdata: entries.map((entry) => entry[2]),
          type: "bar", orientation: "h",
          marker: {color: chapterThreeMetricColors[metric]},
          hovertemplate: "%{y}: %{x:.3f}<br>%{customdata} images<extra></extra>"
        }], {
          title: chapterThreeMetricLabels[metric],
          height: Math.max(240, 80 + entries.length * 24),
          margin: {t: 40, b: 40, l: 10, r: 20},
          shapes: mean === null ? [] : [{type: "line", x0: mean, x1: mean, y0: 0, y1: 1,
            yref: "paper", line: {dash: "dash", color: "#52606d", width: 1.5}}]
        });
      });
    };
    const chapterThreeRender = () => {
      chapterThreeRenderComposition();
      chapterThreeRenderDiversity();
    };
    document.querySelectorAll("#chapter-3-mode-selector input").forEach((input) =>
      input.addEventListener("change", chapterThreeRender));
    document.getElementById("chapter-3-diversity-group")
      .addEventListener("change", chapterThreeRenderDiversity);
    ["chapter-3-context-dimension", "chapter-3-context-split"].forEach((id) =>
      document.getElementById(id).addEventListener("change", chapterThreeRenderContext));
    chapterThreeRender();
    chapterThreeRenderContext();
