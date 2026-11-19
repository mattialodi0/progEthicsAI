  (() => {
    const colorway = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#8C6D31", "#7F7F7F", "#332288", "#B8336A", "#44AA99", "#999933"];
    const font = {family: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
      size: 12.5, color: "#243b53"};
    const withDefaults = (layout) => {
      const result = Object.assign({}, layout || {});
      result.font = Object.assign({}, font, result.font || {});
      result.colorway = result.colorway || colorway;
      result.hoverlabel = Object.assign({font: {family: font.family}}, result.hoverlabel || {});
      result.paper_bgcolor = result.paper_bgcolor || "#ffffff";
      result.plot_bgcolor = result.plot_bgcolor || "#ffffff";
      ["xaxis", "yaxis"].forEach((axis) => {
        result[axis] = Object.assign({gridcolor: "#e6ebf0", zerolinecolor: "#bcccdc",
          automargin: true}, result[axis] || {});
      });
      if (result.title && typeof result.title === "string") {
        result.title = {text: result.title, x: 0, xanchor: "left", font: {size: 14}};
      }
      return result;
    };
    const config = (value) => Object.assign({responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]}, value || {});
    // Lazy rendering: a chart is drawn only when its container comes near the
    // viewport (or a closed <details> holding it is opened). The report has
    // dozens of charts; drawing them all on load makes the page slow to open.
    // A later call for a chart that is still waiting replaces its arguments.
    const pending = new Map();
    const draw = (element) => {
      const job = pending.get(element);
      if (!job) return;
      pending.delete(element);
      lazy?.unobserve(element);
      job.original.apply(Plotly, [element].concat(job.args));
    };
    const lazy = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
      entries.forEach((entry) => { if (entry.isIntersecting) draw(entry.target); });
    }, {rootMargin: "800px 0px"}) : null;
    window.reportFlushCharts = () => Array.from(pending.keys()).forEach(draw);
    ["newPlot", "react"].forEach((name) => {
      const original = Plotly[name];
      Plotly[name] = (element, data, layout, options) => {
        if (!Array.isArray(data)) return original.call(Plotly, element, data, layout, options);
        const args = [data, withDefaults(layout), config(options)];
        const target = typeof element === "string" ? document.getElementById(element) : element;
        // Already drawn (filter changes) or no observer: draw immediately.
        if (!lazy || !target || target._fullLayout) {
          return original.apply(Plotly, [element].concat(args));
        }
        if (!pending.has(target)) lazy.observe(target);
        pending.set(target, {original, args});
        return Promise.resolve(target);
      };
    });
    const purge = Plotly.purge;
    Plotly.purge = (element) => {
      const target = typeof element === "string" ? document.getElementById(element) : element;
      if (target && pending.has(target)) { pending.delete(target); lazy?.unobserve(target); }
      return purge.call(Plotly, element);
    };
  })();
