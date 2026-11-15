"""Renderer for chapter 5: Conditional distributions."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from ..explain import chapter_header, findings, guide, label, metric_defs
from ..insights import chapter_five
from ..plotly_helpers import chart
from ..static_files import script_json, static_text

_SCRIPT = static_text("chapter_5_1.js")


def _overview(bias_stats: dict[str, Any]) -> str:
    """Heatmap of Cramér's V for every prompt attribute × detected feature."""
    prompts = bias_stats.get("prompt_attributes") or []
    outputs = bias_stats.get("output_features") or []
    associations = bias_stats.get("associations") or {}
    z, text, hover = [], [], []
    for prompt in prompts:
        z_row, text_row, hover_row = [], [], []
        for output in outputs:
            matrix = (associations.get(prompt) or {}).get(output) or {}
            stats = matrix.get("statistics") or {}
            value = stats.get("cramers_v")
            p_value = stats.get("p_value")
            n = matrix.get("total", 0)
            if value is None or not n:
                z_row.append(None)
                text_row.append("")
                hover_row.append("no data")
                continue
            star = "*" if p_value is not None and p_value < 0.05 else ""
            same = " (adherence)" if prompt == output else ""
            z_row.append(value)
            text_row.append(f"{value:.2f}{star}{'†' if same else ''}")
            hover_row.append(
                f"V={value:.3f}{same}<br>p={p_value if p_value is None else f'{p_value:.3g}'}"
                f"<br>n={n}"
            )
        z.append(z_row)
        text.append(text_row)
        hover.append(hover_row)
    if not z:
        return '<p class="note">No association statistics are available.</p>'
    figure = go.Figure(go.Heatmap(
        x=[f"Detected {label(output).lower()}" for output in outputs],
        y=[f"Prompt {label(prompt).lower()}" for prompt in prompts],
        z=z, text=text, customdata=hover,
        texttemplate="%{text}", textfont={"size": 13},
        colorscale=[[0, "#f7fbff"], [0.2, "#c6dbef"], [0.4, "#6baed6"],
                    [0.6, "#2171b5"], [1, "#08306b"]],
        zmin=0, zmax=1, xgap=2, ygap=2,
        colorbar={"title": "Cramér's V", "tickvals": [0, 0.1, 0.3, 0.5, 1],
                  "ticktext": ["0", "0.1 weak", "0.3 moderate", "0.5 strong", "1"]},
        hovertemplate="%{y} → %{x}<br>%{customdata}<extra></extra>",
    ))
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(side="top")
    return chart(figure, title="", height=120 + 42 * len(prompts))


def render_chapter_five(bias_stats: dict[str, Any]) -> str:
    """Render selectable conditional bar, pie, or heatmap charts."""
    # Only the keys used here: 5.3 and 5.4 embed their own sections, and the
    # intersectional data is large.
    used = {
        key: bias_stats.get(key)
        for key in ("prompt_attributes", "output_features", "associations",
                    "double_associations")
    }
    # 5.2 is drawn client-side; drop the pre-rendered PNGs (large and too small to read).
    used["double_associations"] = {
        pair: {**item, "outputs": {
            name: {key: value for key, value in matrix.items() if key != "image"}
            for name, matrix in (item.get("outputs") or {}).items()}}
        for pair, item in (used["double_associations"] or {}).items()
    }
    serialized = script_json(used)
    return f"""
  {chapter_header(5, "Conditional distributions and association analysis",
      "Do prompt attributes systematically co-occur with particular detected "
      "demographics, emotions or objects?",
      "<p>This is the core of the audit. For every prompt attribute (job, workplace, "
      "continent, status, trait, and the demographic attributes themselves) it compares "
      "the distribution of each detected feature across the attribute's values. If prompts "
      "for “nurse” produce mostly women while prompts for “engineer” produce mostly men, "
      "the model associates the occupation with a gender. The chapter moves from single "
      "attributes (5.1) to pairs (5.2), to objects (5.3), and to genuinely intersectional "
      "effects (5.4).</p>")}
  {findings(chapter_five(bias_stats))}
  {metric_defs([
      ("Conditional distribution", "P(detected category | prompt value): the share of "
       "faces with each detected category among images with one prompt value.",
       "Compare each row with the “All values” row: large deviations are the association."),
      ("Cramér's V", "Effect size from the chi-square test of the contingency table, "
       "scaled to 0–1.", "&lt; 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, &gt; 0.5 "
       "strong. Compare strengths across pairs with this, not with p-values."),
      ("Chi-square p-value", "Probability of an association at least this strong if prompt "
       "value and detected feature were independent.", "&lt; 0.05: probably not noise. With "
       "small n or many sparse cells the test is approximate; with large n tiny effects "
       "become “significant”."),
  ], open_=False)}
  <section id="chapter-5-overview" aria-labelledby="chapter-5-overview-heading">
    <h3 id="chapter-5-overview-heading">5.0 Association overview</h3>
    <p>Association strength between every prompt attribute and every detected feature,
    in one view. Use it to decide which pairs to inspect in 5.1.</p>
    {guide(
        read=["Each cell is Cramér's V for one prompt attribute (row) and one detected "
              "feature (column); darker means stronger. <strong>*</strong> marks p &lt; 0.05.",
              "Cells marked <strong>†</strong> compare an attribute with itself (e.g. prompt "
              "gender → detected gender): they measure adherence (chapter 4), not bias, and "
              "should be strong."],
        look=["Moderate or strong values (V &gt; 0.3) between a <em>non-demographic</em> "
              "prompt attribute (job, workplace, status, trait, continent) and a detected "
              "demographic are candidate stereotypes.",
              "Associations between one demographic prompt and a <em>different</em> detected "
              "demographic or emotion (e.g. prompt race → detected emotion) indicate that "
              "the model links groups to characteristics nobody asked for."],
        caution=["Some associations are expected and benign (e.g. continent → detected race "
                 "partly reflects real demographics); judge each in context.",
                 "Rows built from few images (status, traits, continents: ~40–50 images) can "
                 "show large V by chance; check n on hover."],
    )}
    <article class="plot-card">{_overview(bias_stats)}</article>
  </section>
  <section id="chapter-5-single-variable" aria-labelledby="chapter-5-single-heading">
    <h3 id="chapter-5-single-heading">5.1 Single-variable–attribute association matrices</h3>
    <p>Detected distribution conditional on one prompt attribute. Select prompt attributes,
    detected features and the chart type; one chart is drawn per combination.</p>
    {guide(
        read=["<strong>Stacked</strong> (default): one 100% bar per prompt value plus an "
              "“All values” reference bar. <strong>Heatmap</strong>: the same percentages as "
              "a grid. <strong>Grouped</strong> and <strong>pie</strong> show the same data "
              "differently.",
              "Above each chart: number of paired observations, Cramér's V with its strength "
              "band and the chi-square p-value."],
        look=["Bars that look very different from the “All values” bar: that prompt value "
              "pulls the detected feature in a particular direction.",
              "Consistent direction across related values (e.g. all care jobs skew female, "
              "all technical jobs skew male).",
              "Emotion differences between demographic groups (e.g. one race detected as "
              "angry or sad more often)."],
        caution="When the prompt attribute is a demographic, its rows include prompts that "
                "also request other demographics (double-variable and control modes). "
                "Differences can partly reflect the prompt mix; chapter 6 compares matched "
                "prompts that differ in only one word.",
    )}
    <div id="chapter-5-controls" class="checkbox-grid" aria-label="Association plot selectors"></div>
    <div id="chapter-5-associations" class=""></div>
  </section>
  <section id="chapter-5-double-variable" aria-labelledby="chapter-5-double-heading">
    <h3 id="chapter-5-double-heading">5.2 Double-variable–attribute association matrices</h3>
    <p>Detected distribution for prompts that specify two attributes together (e.g. “a black
    nurse”). The heatmap shows both prompt attributes on one combined axis.</p>
    {guide(
        read="Rows are prompt pairs, columns are detected categories; cell values are "
             "percentages within each prompt pair, annotations include the counts.",
        look=["Rows whose pattern differs from the other rows sharing one of their "
              "attributes, e.g. “female engineer” vs “female nurse” vs “male engineer”.",
              "Whether the stereotype seen in 5.1 survives when the demographic is "
              "explicitly requested (e.g. does “female doctor” still produce men?)."],
        caution="This view shows joint distributions; whether a combination is more than the "
                "sum of its parts is tested formally in 5.4.",
    )}
    <div id="chapter-5-double-controls" class="checkbox-grid"
      aria-label="Double-variable association selectors"></div>
    <div id="chapter-5-double-association" class="plot-card"></div>
  </section>
  <script>
    const chapterFiveData = {serialized};
{_SCRIPT}  </script>
"""
