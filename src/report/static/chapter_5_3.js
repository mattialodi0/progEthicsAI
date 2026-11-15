    const chapterFiveThreeLabels = {
      job: "Job", workplace: "Workplace", continent: "Continent",
      toxic_trait: "Toxic trait", positive_trait: "Positive trait"
    };
    const chapterFiveThreeSafe = (value) =>
      String(value).replace(/[&<>"']/g, (character) =>
        ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[character]);
    const chapterFiveThreeTitle = (value) =>
      chapterFiveThreeLabels[value] ||
      String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
    const chapterFiveThreeFormat = (value) =>
      value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
    const chapterFiveThreeRender = () => {
      const attribute = document.getElementById("chapter-five-three-attribute").value;
      const minimum = Number(document.getElementById("chapter-five-three-min").value);
      const data = chapterFiveThreeData[attribute];
      const target = document.getElementById("chapter-five-three-table");
      const plot = document.getElementById("chapter-five-three-plot");
      if (!data || !data.rows.length) {
        target.innerHTML = '<p class="note">No object associations are available.</p>';
        Plotly.purge(plot);
        return;
      }
      target.innerHTML = `<p class="note">${data.baseline_observations}
        observations with a prompted ${chapterFiveThreeTitle(attribute)}. Each entry:
        share of the group's images containing the object (n = images, lift).</p>
        <table><thead><tr><th>${chapterFiveThreeTitle(attribute)}</th>
        <th>Images</th><th>Most common objects</th></tr></thead><tbody>` +
        data.rows.map((row) => `<tr><td>${chapterFiveThreeSafe(row.group)}</td>
          <td>${row.observations}</td><td>` +
          row.top_objects.map((item) => `<div><strong>${chapterFiveThreeSafe(item.object)}</strong>:
            ${(100 * item.probability).toFixed(0)}% (n=${item.count}, lift ${
              chapterFiveThreeFormat(item.lift)})</div>`).join("") +
          `</td></tr>`).join("") + "</tbody></table>";

      const plotRows = data.rows.flatMap((row) =>
        row.top_objects.filter((item) => item.count >= minimum && item.lift > 0)
          .map((item) => ({group: row.group, item})))
        .sort((left, right) => right.item.lift - left.item.lift)
        .slice(0, 25).reverse();
      if (!plotRows.length) {
        Plotly.purge(plot);
        plot.innerHTML = '<p class="note">No object reaches the minimum count.</p>';
        return;
      }
      plot.innerHTML = "";
      const groups = Array.from(new Set(plotRows.map((entry) => entry.group))).sort();
      const traces = groups.map((group) => {
        const rows = plotRows.filter((entry) => entry.group === group);
        return {
          type: "bar", orientation: "h", name: group,
          y: rows.map((entry) => `${entry.group} → ${entry.item.object}`),
          x: rows.map((entry) => entry.item.lift),
          customdata: rows.map((entry) => [100 * entry.item.probability, entry.item.count]),
          hovertemplate: "%{y}<br>Lift: %{x:.2f}<br>In %{customdata[0]:.0f}% of the group's images" +
            " (%{customdata[1]} images)<extra></extra>"
        };
      });
      Plotly.newPlot(plot, traces, {
        title: `Most over-represented objects by prompted ${chapterFiveThreeTitle(attribute).toLowerCase()}`,
        barmode: "stack", height: Math.max(320, 90 + 24 * plotRows.length),
        xaxis: {title: "Lift (log scale)", type: "log", tickvals: [0.5, 1, 2, 5, 10, 20, 50, 100]},
        yaxis: {categoryorder: "array",
          categoryarray: plotRows.map((entry) => `${entry.group} → ${entry.item.object}`)},
        legend: {title: {text: chapterFiveThreeTitle(attribute)}},
        margin: {l: 10, r: 20, t: 45, b: 50},
        shapes: [{type: "line", x0: 1, x1: 1, y0: 0, y1: 1, yref: "paper",
          line: {dash: "dash", color: "#52606d"}}]
      });
    };
    const chapterFiveThreeAttributes = Object.keys(chapterFiveThreeData);
    document.getElementById("chapter-five-three-attribute").innerHTML =
      chapterFiveThreeAttributes.map((value) =>
        `<option value="${value}">${chapterFiveThreeTitle(value)}</option>`).join("");
    ["chapter-five-three-attribute", "chapter-five-three-min"].forEach((id) =>
      document.getElementById(id).addEventListener("change", chapterFiveThreeRender));
    if (chapterFiveThreeAttributes.length) {
      chapterFiveThreeRender();
    } else {
      document.getElementById("chapter-five-three-table").innerHTML =
        '<p class="note">No object associations are available.</p>';
    }
