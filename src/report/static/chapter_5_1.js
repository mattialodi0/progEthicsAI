    const chapterFiveLabels = {
      gender: "Gender", race: "Race", age: "Age", emotion: "Emotion",
      job: "Job", workplace: "Workplace", continent: "Continent",
      status: "Status", toxic_trait: "Toxic trait",
      positive_trait: "Positive trait"
    };
    const chapterFiveTitle = (value) =>
      chapterFiveLabels[value] || value.replaceAll("_", " ");
    const chapterFiveBand = (value) => value < 0.1 ? ["negligible", "neutral"] :
      value < 0.3 ? ["weak", "low"] : value < 0.5 ? ["moderate", "mid"] : ["strong", "high"];
    const chapterFiveSafe = (value) =>
      String(value).replace(/[&<>"']/g, (character) =>
        ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[character]);
    const chapterFiveSelected = (name) => new Set(Array.from(
      document.querySelectorAll(`#chapter-5-controls input[name="${name}"]:checked`)
    ).map((input) => input.value));
    const chapterFiveChartType = () =>
      document.querySelector("#chapter-5-controls input[name='chart-type']:checked").value;
    // Adds the pooled "All values" row used as the comparison baseline.
    const chapterFiveWithOverall = (matrix) => {
      const overall = {};
      let total = 0;
      matrix.rows.forEach((row) => matrix.columns.forEach((column) => {
        const count = Number(matrix.counts[row][column] || 0);
        overall[column] = (overall[column] || 0) + count;
        total += count;
      }));
      const rows = [...matrix.rows, "All values"];
      const counts = {...matrix.counts, "All values": overall};
      const percentages = {...matrix.percentages, "All values": Object.fromEntries(
        matrix.columns.map((column) => [column, total ? 100 * overall[column] / total : 0]))};
      const sizes = Object.fromEntries(rows.map((row) => [row,
        matrix.columns.reduce((sum, column) => sum + Number(counts[row][column] || 0), 0)]));
      return {rows, counts, percentages, sizes, columns: matrix.columns};
    };
    function chapterFiveRender() {
      const prompts = chapterFiveSelected("prompt-attribute");
      const outputs = chapterFiveSelected("output-feature");
      const chartType = chapterFiveChartType();
      const cards = [];
      const plots = [];
      prompts.forEach((promptAttribute) => outputs.forEach((outputFeature) => {
        const matrix = chapterFiveData.associations?.[promptAttribute]?.[outputFeature];
        if (!matrix || !matrix.total) return;
        const plotId = `chapter-five-${promptAttribute}-${outputFeature}`;
        const statistics = matrix.statistics || {};
        const value = Number(statistics.cramers_v || 0);
        const [band, tone] = chapterFiveBand(value);
        const pValue = Number(statistics.p_value ?? 1);
        const self = promptAttribute === outputFeature ?
          ' <span class="badge badge-neutral">adherence, see ch. 4</span>' : "";
        cards.push(`<article class="plot-card"><h4>Prompt ${chapterFiveSafe(chapterFiveTitle(promptAttribute))}
          → detected ${chapterFiveSafe(chapterFiveTitle(outputFeature))}${self}</h4>
          <div class="stat-line"><span><b>n</b> = ${matrix.total}</span>
          <span><b>Cramér's V</b> = ${value.toFixed(3)} <span class="badge badge-${tone}">${band}</span></span>
          <span><b>χ²</b> = ${Number(statistics.chi_square || 0).toFixed(1)}
          (df=${statistics.degrees_of_freedom || 0}, p ${pValue < 0.001 ? "&lt; 0.001" : "= " + pValue.toFixed(3)})</span></div>
          <div id="${plotId}"></div></article>`);
        plots.push([plotId, matrix, promptAttribute, outputFeature, chartType]);
      }));
      document.getElementById("chapter-5-associations").innerHTML = cards.join("") ||
        '<p class="note">No paired observations are available for the selected plots.</p>';
      plots.forEach(([plotId, rawMatrix, promptAttribute, outputFeature, chartType]) => {
        const matrix = chapterFiveWithOverall(rawMatrix);
        const rowLabels = matrix.rows.map((row) => `${row} (n=${matrix.sizes[row]})`);
        if (chartType === "pie") {
          const columnCount = Math.min(3, Math.max(matrix.rows.length, 1));
          const rowCount = Math.ceil(matrix.rows.length / columnCount);
          const chartHeight = Math.max(320, 220 * rowCount);
          const traces = matrix.rows.map((row, index) => ({
            type: "pie", sort: false,
            labels: matrix.columns,
            values: matrix.columns.map((column) => matrix.percentages[row][column]),
            text: matrix.columns.map((column) => `${matrix.counts[row][column]} observations`),
            hovertemplate: "Detected: %{label}<br>%{value:.1f}%<br>%{text}<extra></extra>",
            textinfo: "percent", name: row,
            domain: {row: Math.floor(index / columnCount), column: index % columnCount},
            title: {text: rowLabels[index]}
          }));
          Plotly.newPlot(plotId, traces, {
            grid: {rows: rowCount, columns: columnCount, pattern: "independent"},
            margin: {l: 20, r: 20, t: 30, b: 20}, height: chartHeight, showlegend: true
          });
        } else if (chartType === "heatmap") {
          Plotly.newPlot(plotId, [{
            type: "heatmap",
            x: matrix.columns, y: rowLabels,
            z: matrix.rows.map((row) => matrix.columns.map((column) => matrix.percentages[row][column])),
            customdata: matrix.rows.map((row) => matrix.columns.map((column) => matrix.counts[row][column])),
            texttemplate: "%{z:.0f}%",
            colorscale: [[0, "#f7fbff"], [0.25, "#c6dbef"], [0.5, "#6baed6"], [0.75, "#2171b5"], [1, "#08306b"]], zmin: 0, zmax: 100, xgap: 1, ygap: 1,
            hovertemplate: "Prompt: %{y}<br>Detected: %{x}<br>" +
              "%{z:.1f}%<br>%{customdata} observations<extra></extra>",
            colorbar: {title: "% of row"}
          }], {
            xaxis: {title: {text: `Detected ${chapterFiveTitle(outputFeature)}`}, tickangle: -30},
            yaxis: {title: {text: `Prompt ${chapterFiveTitle(promptAttribute)}`}, autorange: "reversed"},
            margin: {l: 20, r: 25, t: 20, b: 90}, height: Math.max(300, 110 + 34 * matrix.rows.length)
          });
        } else if (chartType === "grouped") {
          const traces = matrix.columns.map((column) => ({
            type: "bar", name: column, x: rowLabels,
            y: matrix.rows.map((row) => matrix.percentages[row][column]),
            customdata: matrix.rows.map((row) => matrix.counts[row][column]),
            hovertemplate: `%{x}<br>Detected ${chapterFiveSafe(column)}: %{y:.1f}%<br>%{customdata} observations<extra></extra>`
          }));
          Plotly.newPlot(plotId, traces, {
            barmode: "group", yaxis: {title: "Share of detected faces (%)", range: [0, 100]},
            xaxis: {title: `Prompt ${chapterFiveTitle(promptAttribute)}`},
            legend: {title: {text: `Detected ${chapterFiveTitle(outputFeature)}`}},
            margin: {l: 60, r: 25, t: 20, b: 75}, height: 360
          });
        } else {
          const traces = matrix.columns.map((column) => ({
            type: "bar", orientation: "h", name: column, y: rowLabels,
            x: matrix.rows.map((row) => matrix.percentages[row][column]),
            customdata: matrix.rows.map((row) => matrix.counts[row][column]),
            hovertemplate: `%{y}<br>Detected ${chapterFiveSafe(column)}: %{x:.1f}%<br>%{customdata} observations<extra></extra>`
          }));
          Plotly.newPlot(plotId, traces, {
            barmode: "stack",
            xaxis: {title: "Share of detected faces (%)", range: [0, 100]},
            yaxis: {autorange: "reversed"},
            legend: {title: {text: `Detected ${chapterFiveTitle(outputFeature)}:`}, orientation: "h",
              x: 0, y: 1.02, yanchor: "bottom", traceorder: "normal"},
            margin: {l: 10, r: 20, t: 40, b: 50}, height: Math.max(280, 140 + 30 * matrix.rows.length),
            shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y",
              y0: matrix.rows.length - 1.5, y1: matrix.rows.length - 0.5,
              line: {color: "#243b53", width: 1.5}, fillcolor: "rgba(0,0,0,0)"}]
          });
        }
      });
    }
    const chapterFiveControl = (group, value, checked) =>
      `<label><input type="checkbox" name="${group}" value="${value}" ${checked ? "checked" : ""}>
      ${chapterFiveSafe(chapterFiveTitle(value))}</label>`;
    const chapterFivePromptAttributes = chapterFiveData.prompt_attributes || [];
    const chapterFiveOutputFeatures = chapterFiveData.output_features || [];
    const chapterFiveDefaultPrompts = new Set(["job", "race"]);
    const chapterFiveDefaultOutputs = new Set(["gender", "emotion"]);
    document.getElementById("chapter-5-controls").innerHTML =
      '<fieldset><legend>Prompt attributes</legend>' +
      chapterFivePromptAttributes.map((value) =>
        chapterFiveControl("prompt-attribute", value, chapterFiveDefaultPrompts.has(value))).join("") +
      '</fieldset><fieldset><legend>Detected features</legend>' +
      chapterFiveOutputFeatures.map((value) =>
        chapterFiveControl("output-feature", value, chapterFiveDefaultOutputs.has(value))).join("") +
      '</fieldset><fieldset><legend>Chart type</legend>' +
      '<label><input type="radio" name="chart-type" value="stacked" checked> Stacked</label>' +
      '<label><input type="radio" name="chart-type" value="heatmap"> Heatmap</label>' +
      '<label><input type="radio" name="chart-type" value="grouped"> Grouped</label>' +
      '<label><input type="radio" name="chart-type" value="pie"> Pie</label></fieldset>';
    document.querySelectorAll("#chapter-5-controls input").forEach((input) =>
      input.addEventListener("change", chapterFiveRender));
    chapterFiveRender();
    const chapterFiveDouble = chapterFiveData.double_associations || {};
    const chapterFiveDoubleEntries = Object.entries(chapterFiveDouble);
    const chapterFiveDoubleLabels = (key) => {
      const item = chapterFiveDouble[key];
      return `${chapterFiveTitle(item.first_attribute)} ×
        ${chapterFiveTitle(item.second_attribute)}`;
    };
    const chapterFiveRenderDouble = () => {
      const pair = document.getElementById("chapter-5-double-pair").value;
      const output = document.getElementById("chapter-5-double-output").value;
      const item = chapterFiveDouble[pair];
      const matrix = item?.outputs?.[output];
      const target = document.getElementById("chapter-5-double-association");
      if (!matrix) {
        target.innerHTML = '<p class="note">No paired observations are available.</p>';
        return;
      }
      target.innerHTML = `<h4>Prompt ${chapterFiveSafe(chapterFiveDoubleLabels(pair))}
        → detected ${chapterFiveSafe(chapterFiveTitle(output))}</h4>
        <p class="note">${matrix.total} paired observations. Rows are prompt pairs
        (n in brackets), columns the detected categories.</p>
        <div id="chapter-5-double-plot"></div>`;
      const pairs = matrix.columns;
      const sizes = pairs.map((pairLabel) => matrix.rows.reduce(
        (sum, row) => sum + Number(matrix.counts[row][pairLabel] || 0), 0));
      const rowHeight = pairs.length > 12 ? 26 : 38;
      Plotly.newPlot("chapter-5-double-plot", [{
        type: "heatmap", x: matrix.rows,
        y: pairs.map((pairLabel, index) => `${pairLabel} (n=${sizes[index]})`),
        z: pairs.map((pairLabel) => matrix.rows.map((row) => matrix.percentages[row][pairLabel])),
        customdata: pairs.map((pairLabel) => matrix.rows.map((row) => matrix.counts[row][pairLabel])),
        texttemplate: "%{z:.0f}%", textfont: {size: pairs.length > 12 ? 11 : 13},
        colorscale: [[0, "#f7fbff"], [0.25, "#c6dbef"], [0.5, "#6baed6"], [0.75, "#2171b5"], [1, "#08306b"]],
        zmin: 0, zmax: 100, xgap: 1, ygap: 1,
        hovertemplate: "Prompt: %{y}<br>Detected: %{x}<br>%{z:.1f}%<br>%{customdata} images<extra></extra>",
        colorbar: {title: "% of row"}
      }], {
        xaxis: {title: {text: `Detected ${chapterFiveTitle(output)}`}, side: "top"},
        yaxis: {autorange: "reversed", automargin: true, tickfont: {size: 12}},
        margin: {l: 10, r: 20, t: 70, b: 20}, height: Math.max(300, 130 + rowHeight * pairs.length)
      });
    };
    document.getElementById("chapter-5-double-controls").innerHTML =
      '<label>Prompt pair <select id="chapter-5-double-pair">' +
      chapterFiveDoubleEntries.map(([key]) =>
        `<option value="${key}">${chapterFiveSafe(chapterFiveDoubleLabels(key))}</option>`
      ).join("") + '</select></label>' +
      '<label>Detected feature <select id="chapter-5-double-output">' +
      (chapterFiveDoubleEntries.length ? Object.keys(
        chapterFiveDouble[chapterFiveDoubleEntries[0][0]].outputs).map((value) =>
        `<option value="${value}">${chapterFiveSafe(chapterFiveTitle(value))}</option>`
      ).join("") : "") + '</select></label>';
    const doublePairSelect = document.getElementById("chapter-5-double-pair");
    const doubleOutputSelect = document.getElementById("chapter-5-double-output");
    if (doublePairSelect && doubleOutputSelect && chapterFiveDoubleEntries.length) {
      doublePairSelect.addEventListener("change", () => {
        const outputs = Object.keys(chapterFiveDouble[doublePairSelect.value].outputs);
        doubleOutputSelect.innerHTML = outputs.map((value) =>
          `<option value="${value}">${chapterFiveSafe(chapterFiveTitle(value))}</option>`
        ).join("");
        chapterFiveRenderDouble();
      });
      doubleOutputSelect.addEventListener("change", chapterFiveRenderDouble);
      chapterFiveRenderDouble();
    } else {
      document.getElementById("chapter-5-double-association").innerHTML =
        '<p class="note">No double-variable observations are available.</p>';
    }
