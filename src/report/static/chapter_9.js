(() => {
  const payload = JSON.parse(document.getElementById("chapter-9-data").textContent);
  const rows = payload.rows;
  const lenses = payload.lenses;
  const smallN = payload.small_n;
  const alpha = payload.alpha;
  const lensColor = {};
  const palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
                   "#8C6D31", "#7F7F7F", "#332288", "#B8336A", "#44AA99"];
  Object.keys(lenses).forEach((lens, index) => { lensColor[lens] = palette[index % palette.length]; });
  const tierColor = {supported: "#c0392b", descriptive: "#c97a00", negligible: "#9aa5b1",
                     context: "#829ab1", measurement: "#2f6fb0"};
  const tierLabel = payload.tiers;
  const el = (id) => document.getElementById(id);
  const controls = {
    lens: el("ch9-lens"), attribute: el("ch9-attribute"), tier: el("ch9-tier"),
    minN: el("ch9-min-n"), question: el("ch9-question"), search: el("ch9-search"),
    specific: el("ch9-specific")
  };
  const options = (select, values, labels) => {
    select.innerHTML += values.map((value) =>
      '<option value="' + value + '">' + (labels ? labels[value] || value : value) + "</option>").join("");
  };
  options(controls.lens, Object.keys(lenses), Object.fromEntries(
    Object.entries(lenses).map(([key, value]) => [key, value.label])));
  options(controls.attribute, Array.from(new Set(rows.map((row) => row.attribute)
    .filter(Boolean))).sort());
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
  const fmtEffect = (row) => {
    const v = row.effect;
    if (v === null || v === undefined) return "–";
    const [lo, hi] = row.ci || [null, null];
    const both = lo !== null && hi !== null && lo !== undefined && hi !== undefined;
    if (row.effect_unit === "pp") return (100 * v >= 0 ? "+" : "") + (100 * v).toFixed(0) + " pp" +
      (both ? " [" + (100 * lo).toFixed(0) + ", " + (100 * hi).toFixed(0) + "]" : "");
    if (row.effect_unit === "share") return (100 * v).toFixed(0) + "%" +
      (both ? " [" + (100 * lo).toFixed(0) + "%, " + (100 * hi).toFixed(0) + "%]" : "");
    if (row.effect_unit === "ratio") return v.toFixed(2) + "×";
    if (row.effect_unit === "value") return (v >= 0 ? "+" : "") + v.toPrecision(3) +
      (both ? " [" + lo.toPrecision(3) + ", " + hi.toPrecision(3) + "]" : "");
    return v.toFixed(2) + (both ? " [" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]" : "");
  };
  const fmtQ = (q) => q === null || q === undefined ? "–" : (q < 0.001 ? "<0.001" : q.toFixed(3));
  const rankKey = (row) => [-Math.min(row.strength_low ?? 0, 3), -(row.n ?? 0), -(row.strength ?? 0)];
  const compare = (a, b) => {
    const ka = rankKey(a), kb = rankKey(b);
    for (let i = 0; i < ka.length; i += 1) if (ka[i] !== kb[i]) return ka[i] - kb[i];
    return 0;
  };
  let sortField = null, sortAsc = false;
  // ``anyTier``: the reliability plot compares tiers, so it ignores the tier
  // filter and shows every substantive result.
  const filtered = (anyTier) => {
    const lens = controls.lens.value, attribute = controls.attribute.value;
    const tier = anyTier ? "substantive" : controls.tier.value;
    const question = controls.question.value;
    const minN = Number(controls.minN.value) || 0;
    const search = controls.search.value.trim().toLowerCase();
    const specific = controls.specific.checked;
    return rows.filter((row) =>
      (!lens || row.lens === lens) &&
      (!attribute || row.attribute === attribute) &&
      (!question || row.question === question) &&
      (tier === "all" || (tier === "supported" ? row.tier === "supported"
        : tier === "substantive" ? !["context", "measurement"].includes(row.tier)
        : row.tier === tier)) &&
      (row.n === null || row.n === undefined || row.n >= minN) &&
      (!specific || row.level === "specific") &&
      (!search || (/^f\d+$/.test(search) ? row.cluster === search
        : (row.statement + " " + row.id).toLowerCase().includes(search))));
  };
  const short = (text, size) => text.length > size ? text.slice(0, size - 1) + "…" : text;
  // Plotly keeps its state on the div: wiping innerHTML of a plotted div breaks
  // later Plotly.react calls, so purge before showing a note, and remove only the
  // note before plotting again.
  const showEmpty = (target) => {
    Plotly.purge(target);
    target.innerHTML = '<p class="note">No result matches the filters.</p>';
  };
  const clearNote = (target) => target.querySelectorAll(":scope > p.note").forEach((node) => node.remove());
  const drawForest = (data) => {
    const ranked = data.filter((row) => row.strength !== null && row.strength !== undefined)
      .sort(compare).slice(0, 30).reverse();
    const target = el("ch9-forest");
    if (!ranked.length) { showEmpty(target); return; }
    clearNote(target);
    const traces = Object.keys(lenses).map((lens) => {
      const items = ranked.filter((row) => row.lens === lens);
      return {
        type: "scatter", mode: "markers", name: lenses[lens].label,
        x: items.map((row) => row.strength),
        y: items.map((row) => row.id),
        marker: {color: lensColor[lens], size: 10,
                 symbol: items.map((row) => row.tier === "supported" ? "circle" : "circle-open")},
        error_x: {type: "data", symmetric: false, color: lensColor[lens], thickness: 1.5,
                  array: items.map((row) => Math.max(0, (row.strength_ci?.[1] ?? row.strength) - row.strength)),
                  arrayminus: items.map((row) => Math.max(0, row.strength - (row.strength_ci?.[0] ?? row.strength)))},
        customdata: items.map((row) => [escapeHtml(row.statement), fmtEffect(row), row.n, fmtQ(row.q_value),
                                        tierLabel[row.tier] || row.tier]),
        hovertemplate: "%{customdata[0]}<br>Effect %{customdata[1]} · n = %{customdata[2]} · q %{customdata[3]}" +
                       "<br>Strength %{x:.2f} · %{customdata[4]}<extra></extra>"
      };
    }).filter((trace) => trace.x.length);
    const labels = Object.fromEntries(ranked.map((row) => [row.id, short(row.statement, 70)]));
    Plotly.react(target, traces, {
      height: Math.max(420, 26 * ranked.length + 120),
      margin: {l: 430, r: 20, t: 20, b: 50},
      xaxis: {title: "Strength (1 = small, 2 = medium, 3 = large); bars = 95% CI", range: [0, 6.3], zeroline: false},
      yaxis: {type: "category", categoryorder: "array", categoryarray: ranked.map((row) => row.id),
              tickmode: "array", tickvals: ranked.map((row) => row.id),
              ticktext: ranked.map((row) => labels[row.id]), automargin: false, tickfont: {size: 11}},
      shapes: [1, 2, 3].map((x) => ({type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1,
                                     line: {dash: "dot", color: "#9aa5b1", width: 1}})),
      legend: {orientation: "h", y: -0.12}
    });
  };
  const drawScatter = (data) => {
    const target = el("ch9-reliability");
    const usable = data.filter((row) => row.strength !== null && row.strength !== undefined && row.n);
    if (!usable.length) { showEmpty(target); return; }
    clearNote(target);
    const tiers = Array.from(new Set(usable.map((row) => row.tier)));
    const traces = tiers.map((tier) => {
      const items = usable.filter((row) => row.tier === tier);
      return {
        type: "scattergl", mode: "markers", name: tierLabel[tier] || tier,
        x: items.map((row) => row.n), y: items.map((row) => row.strength),
        marker: {color: tierColor[tier] || "#7f7f7f", size: 7, opacity: 0.65},
        customdata: items.map((row) => [escapeHtml(short(row.statement, 140)), fmtEffect(row),
                                        fmtQ(row.q_value), lenses[row.lens].label]),
        hovertemplate: "%{customdata[0]}<br>%{customdata[3]} · effect %{customdata[1]} · q %{customdata[2]}" +
                       "<br>n = %{x} · strength %{y:.2f}<extra></extra>"
      };
    });
    Plotly.react(target, traces, {
      height: 460, margin: {l: 60, r: 20, t: 30, b: 60},
      xaxis: {title: "n of the smallest compared group (log scale)", type: "log"},
      yaxis: {title: "Strength", range: [0, 6.3]},
      shapes: [
        {type: "rect", xref: "x", yref: "y", x0: 1, x1: smallN, y0: 3, y1: 6.3,
         fillcolor: "#c0392b", opacity: 0.07, line: {width: 0}},
        ...[1, 2, 3].map((y) => ({type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
                                  line: {dash: "dot", color: "#9aa5b1", width: 1}}))
      ],
      annotations: [{x: 0, y: 1.0, xref: "paper", yref: "paper", xanchor: "left", yanchor: "bottom",
                     text: "shaded: large effect, n < " + smallN, showarrow: false,
                     font: {size: 11, color: "#9b2c1f"}}],
      legend: {orientation: "h", y: -0.18}
    });
  };
  const columns = [
    ["statement", "Result"], ["lens", "Lens"], ["effect", "Effect [95% CI]"],
    ["strength", "Strength"], ["n", "n"], ["q_value", "q"], ["tier", "Tier"]
  ];
  const drawTable = (data) => {
    let items = data.slice();
    if (sortField) {
      items.sort((a, b) => {
        const x = a[sortField], y = b[sortField];
        const nx = x === null || x === undefined, ny = y === null || y === undefined;
        if (nx || ny) return nx === ny ? 0 : (nx ? 1 : -1);
        return (x > y ? 1 : x < y ? -1 : 0) * (sortAsc ? 1 : -1);
      });
    } else {
      items.sort(compare);
    }
    const shown = items.slice(0, 250);
    el("ch9-count").textContent = data.length + " result(s) match" +
      (data.length > shown.length ? "; showing the first " + shown.length + "." : ".");
    const head = columns.map(([key, name]) =>
      '<th scope="col"><button type="button" class="sort" data-field="' + key + '">' + name +
      (sortField === key ? (sortAsc ? " ▲" : " ▼") : "") + "</button></th>").join("");
    const body = shown.map((row) => "<tr><td>" + escapeHtml(row.statement) +
      (row.flags_text ? '<br><span class="note">⚠ ' + escapeHtml(row.flags_text) + "</span>" : "") +
      "</td><td>" + '<a href="' + row.link + '">' + escapeHtml(lenses[row.lens].label) + "</a></td><td>" +
      fmtEffect(row) + "</td><td>" + (row.strength === null || row.strength === undefined ? "–" : row.strength.toFixed(2)) +
      "</td><td>" + (row.n ?? "–") + "</td><td>" + fmtQ(row.q_value) + "</td><td>" +
      escapeHtml(tierLabel[row.tier] || row.tier) + (row.cluster ? " · " + row.cluster : "") + "</td></tr>").join("");
    el("ch9-table").innerHTML = "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>";
    el("ch9-table").querySelectorAll("button.sort").forEach((button) => {
      button.onclick = () => {
        const field = button.dataset.field;
        if (sortField === field) sortAsc = !sortAsc; else { sortField = field; sortAsc = field === "statement"; }
        drawTable(filtered(false));
      };
    });
  };
  const csvField = (value) => {
    if (value === null || value === undefined) return "";
    const text = Array.isArray(value) ? value.join(";") : String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  };
  el("ch9-download").onclick = () => {
    const data = filtered(false);
    const fields = payload.fields.concat(["flags_text"]);
    const csv = [fields.join(",")].concat(data.map((row) => fields.map((field) => csvField(row[field])).join(","))).join("\n");
    const blob = new Blob([csv], {type: "text/csv"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "fairness_results.csv";
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  const draw = () => {
    const data = filtered(false);
    drawForest(data); drawScatter(filtered(true)); drawTable(data);
  };
  Object.values(controls).forEach((control) => control.addEventListener(
    control.tagName === "INPUT" && control.type !== "checkbox" ? "input" : "change", draw));
  document.querySelectorAll("[data-ch9-filter]").forEach((link) => link.addEventListener("click", () => {
    const [field, value] = link.dataset.ch9Filter.split(":");
    if (field === "search") controls.search.value = value;
    if (field === "lens") controls.lens.value = value;
    controls.tier.value = "substantive";
    el("ch9-explorer-details").open = true;
    draw();
  }));
  draw();
})();
