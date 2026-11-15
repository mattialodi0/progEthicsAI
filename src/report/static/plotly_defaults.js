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
    ["newPlot", "react"].forEach((name) => {
      const original = Plotly[name];
      Plotly[name] = (element, data, layout, options) =>
        Array.isArray(data)
          ? original.call(Plotly, element, data, withDefaults(layout), config(options))
          : original.call(Plotly, element, data, layout, options);
    });
  })();
