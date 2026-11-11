"""Build the HTML fairness-audit report from chapter renderers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
import webbrowser

from .insights import conclusive_insights, headline_indicators
from .report_intro import render_reading_guide
from .report_ch_1 import render_chapter_one
from .report_ch_2 import render_chapter_two
from .report_ch_3 import render_chapter_three
from .report_ch_4 import render_chapter_four
from .report_ch_5_1 import render_chapter_five
from .report_ch_5_3 import render_stereotypical_associations
from .report_ch_5_4 import render_intersectional_analysis
from .report_ch_6 import render_chapter_six
from .report_ch_7 import render_chapter_seven
from .report_ch_8 import render_chapter_eight
from .report_ch_9 import render_chapter_nine
from .report_ch_10 import render_chapter_ten
from ..analysis.synthesis import build_synthesis


# (anchor id, label, level) — level 0 is a chapter, 1 a section.
_INDEX = (
    ("report-title", "Fairness report", -1),
    ("reading-guide", "How to read this report", -1),
    ("chapter-1", "1. Executive summary", 0),
    ("chapter-1-conclusions", "Main conclusions", 1),
    ("chapter-1-model", "Model and configuration", 1),
    ("chapter-1-coverage", "Audit coverage", 1),
    ("chapter-1-fairness", "Headline indicators", 1),
    ("chapter-1-scope", "Scope and limitations", 1),
    ("chapter-2", "2. Dataset and generation coverage", 0),
    ("chapter-2-modes", "Prompt modes", 1),
    ("chapter-2-attributes", "Prompt attributes", 1),
    ("chapter-2-prompts", "Images per prompt", 1),
    ("chapter-2-detections", "Detection rates", 1),
    ("chapter-3", "3. Representation and diversity", 0),
    ("chapter-3-distributions", "Demographic distributions", 1),
    ("chapter-3-context", "Composition by job and workplace", 1),
    ("chapter-3-diversity", "Diversity", 1),
    ("chapter-4", "4. Prompt adherence", 0),
    ("chapter-4-attributes", "Overall adherence", 1),
    ("chapter-4-modes", "Prompt modes", 1),
    ("chapter-4-demographics", "Demographic categories", 1),
    ("chapter-4-combinations", "Attribute combinations", 1),
    ("chapter-5", "5. Associations", 0),
    ("chapter-5-overview", "Association overview", 1),
    ("chapter-5-single-variable", "Single-variable associations", 1),
    ("chapter-5-double-variable", "Double-variable associations", 1),
    ("chapter-5-stereotypical", "Stereotypical item associations", 1),
    ("chapter-5-intersectional", "Intersectional analysis", 1),
    ("chapter-6", "6. Counterfactual analysis", 0),
    ("chapter-6-overview", "Overview", 1),
    ("chapter-6-features", "Feature counterfactuals", 1),
    ("chapter-6-embeddings", "Embedding counterfactuals", 1),
    ("chapter-7", "7. Visual features and embeddings", 0),
    ("chapter-7-features", "Image-feature distributions", 1),
    ("chapter-7-embeddings", "Embedding separability", 1),
    ("chapter-8", "8. Caption and semantic associations", 0),
    ("chapter-8-prompt", "Prompt-word associations", 1),
    ("chapter-8-attributes", "Word-attribute associations", 1),
    ("chapter-8-sentiment", "Description sentiment", 1),
    ("chapter-8-diversity", "Vocabulary diversity", 1),
    ("chapter-8-vocabulary", "Stopwords and vocabulary", 1),
    ("chapter-9", "9. Cross-analysis and key findings", 0),
    ("chapter-9-answers", "Fairness questions", 1),
    ("chapter-9-findings", "Key findings", 1),
    ("chapter-9-matrix", "Evidence matrix", 1),
    ("chapter-9-ranked", "Ranked effect sizes", 1),
    ("chapter-9-reliability", "Effect vs reliability", 1),
    ("chapter-9-measurement", "Measurement caveats", 1),
    ("chapter-9-explorer", "Results explorer", 1),
    ("chapter-10", "10. Methodology and limitations", 0),
    ("chapter-10-pipeline", "Audit pipeline", 1),
    ("chapter-10-models", "Measurement models", 1),
    ("chapter-10-statistics", "How the analysis works", 1),
    ("chapter-10-limitations", "Limitations", 1),
    ("chapter-10-reproducibility", "Reproducibility", 1),
)

_STYLE = """
    :root { color-scheme: light;
      --bg: #f4f6f8; --surface: #ffffff; --border: #d9e2ec; --text: #17202a;
      --muted: #52606d; --accent: #2f5d8a; --accent-soft: #eaf1f8;
      --read: #2f6fb0; --read-bg: #eef5fc; --look: #1d7a55; --look-bg: #ecf7f1;
      --caution: #a35f00; --caution-bg: #fdf5e8; --finding: #6a4bb5; --finding-bg: #f3effb;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); margin: 0; line-height: 1.55; }
    .report-layout { display: grid; gap: 2rem; grid-template-columns: 240px minmax(0, 1fr);
      margin: 0 auto; max-width: 1360px; padding: 2rem 1rem 4rem; }
    main { min-width: 0; }
    .sidebar { align-self: start; position: sticky; top: 1rem; }
    .sidebar nav { background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; box-shadow: 0 2px 5px #102a430d; padding: 1rem;
      overflow-y: auto; max-height: 92vh; }
    .sidebar-title { color: var(--muted); font-size: .75rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase; margin-bottom: .4rem; }
    .sidebar a { border-left: 3px solid transparent; color: #334e68; display: block;
      font-size: .92rem; padding: .35rem .6rem; text-decoration: none; }
    .sidebar a:hover, .sidebar a.active { background: #f0f4f8;
      border-left-color: var(--accent); color: #102a43; }
    .sidebar .supitem { font-weight: 700; margin-top: .35rem; }
    .sidebar .subitem { font-size: .85rem; padding-left: 1.25rem; }
    @media (max-width: 800px) { .report-layout { display: block; }
      .sidebar { margin-bottom: 1rem; position: static; } .sidebar nav {
        display: flex; gap: .25rem; overflow-x: auto; white-space: nowrap; }
      .sidebar-title { display: none; } .sidebar .subitem { display: none; }
      .suggestion-toggle { flex: none; margin: 0 .5rem 0 0; width: auto; } }
    .report-header { border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;
      padding-bottom: 1rem; }
    .report-header h1 { font-size: 2.1rem; margin: 0 0 .25rem; }
    .eyebrow { color: var(--muted); font-size: .8rem; font-weight: 700;
      letter-spacing: .08em; margin-top: 3rem; text-transform: uppercase; }
    h2 { font-size: 1.7rem; margin: .2rem 0 .6rem; }
    h3 { font-size: 1.25rem; margin-top: 0; }
    h4 { margin: .2rem 0 .5rem; }
    p, li { max-width: 88ch; }
    .subtitle { color: var(--muted); margin-top: 0; }
    .question { background: var(--accent-soft); border-left: 4px solid var(--accent);
      border-radius: 0 8px 8px 0; font-size: 1.05rem; font-weight: 600; margin: .5rem 0 1rem;
      max-width: none; padding: .7rem 1rem; }
    .question span { color: var(--accent); display: block; font-size: .72rem;
      letter-spacing: .08em; text-transform: uppercase; }
    .intro { margin-bottom: 1rem; }
    .metrics { display: grid; gap: 1rem; grid-template-columns:
      repeat(auto-fit, minmax(min(190px, 100%), 1fr)); margin: 1rem 0; }
    .metric, section, details { background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; box-shadow: 0 2px 5px #102a430d; }
    section { margin-top: 1.25rem; }
    .metric { color: inherit; padding: 1rem; text-decoration: none; }
    a.metric:hover { border-color: var(--accent); }
    .metric-label { color: var(--muted); display: block; font-size: .85rem; }
    .metric-value { display: block; font-size: 1.6rem; font-weight: 700; margin-top: .25rem; }
    .metric-help { color: var(--muted); display: block; font-size: .8rem; margin-top: .35rem; }
    section, details { padding: 1rem 1.25rem; }
    details { margin-top: 1rem; } summary { cursor: pointer; font-weight: 700; }
    pre { background: var(--bg); border-radius: 6px; overflow: auto; padding: 1rem; }
    table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
    th, td { border-bottom: 1px solid var(--border); padding: .5rem; text-align: left;
      vertical-align: top; }
    th { color: var(--muted); font-size: .85rem; }
    .table-wrap { overflow-x: auto; }
    .bar { background: var(--border); border-radius: 99px; height: .55rem; min-width: 3rem;
      overflow: hidden; }
    .bar span { background: var(--accent); display: block; height: 100%; }
    .bar.warn span { background: #c97a00; } .bar.bad span { background: #c0392b; }
    .placeholder { border-left: 4px solid #829ab1; color: var(--muted); padding-left: 1rem; }
    .note { color: var(--muted); font-size: .92rem; }
    .checkbox-grid, .controls, .ch8-controls { align-items: flex-end; background: #f8fafc;
      border: 1px solid var(--border); border-radius: 8px; display: flex; flex-wrap: wrap;
      gap: .6rem 1.25rem; margin: .75rem 0; padding: .65rem .9rem; }
    .checkbox-grid:empty { display: none; }
    .checkbox-grid label, .controls label, .ch8-controls label { cursor: pointer; font-size: .92rem; }
    .checkbox-grid select, .controls select, .ch8-controls select { display: block; font: inherit; margin-top: .2rem;
      padding: .25rem .4rem; }
    fieldset { border: 1px solid var(--border); border-radius: 6px; padding: .4rem .7rem; }
    fieldset label { margin-right: .7rem; white-space: nowrap; }
    legend { color: var(--muted); font-size: .8rem; font-weight: 700; }
    .plot-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      margin-top: 1rem; min-width: 0; overflow: visible; padding: 1rem; }
    .plot-card > div { min-width: 0; max-width: 100%; }
    .plot-grid { display: grid; gap: 1rem; grid-template-columns:
      repeat(auto-fit, minmax(min(340px, 100%), 1fr)); }
    .plot-grid > .plot-card { margin-top: 0; }
    @media (max-width: 700px) { .plot-grid { grid-template-columns: 1fr; } }
    .guide { display: grid; gap: .8rem; grid-template-columns:
      repeat(auto-fit, minmax(min(300px, 100%), 1fr)); margin: 1rem 0; }
    .callout { border: 1px solid; border-left-width: 4px; border-radius: 8px;
      font-size: .93rem; padding: .7rem 1rem; }
    .callout p, .callout li { max-width: none; }
    .callout ul { margin: .3rem 0 0; padding-left: 1.2rem; }
    .callout li { margin-bottom: .3rem; }
    .callout p { margin: .3rem 0 0; }
    .callout-title { font-size: .75rem; font-weight: 800; letter-spacing: .07em;
      text-transform: uppercase; }
    .callout-read { background: var(--read-bg); border-color: var(--read); }
    .callout-read .callout-title { color: var(--read); }
    .callout-look { background: var(--look-bg); border-color: var(--look); }
    .callout-look .callout-title { color: var(--look); }
    .callout-caution { background: var(--caution-bg); border-color: var(--caution); }
    .callout-caution .callout-title { color: var(--caution); }
    /* Findings are data, not advice: a full-border card with a solid title
       band, so they differ from the suggestion call-outs in shape, not only colour. */
    .callout-finding { background: var(--surface); border: 2px solid var(--finding);
      border-radius: 6px; box-shadow: 0 2px 6px #6a4bb51f; margin: 1rem 0; padding: 0; }
    .callout-finding .callout-title { background: var(--finding); color: #fff;
      padding: .45rem 1rem; }
    .callout-finding .callout-title::before { content: ""; margin-right: .5rem; }
    .callout-finding > ul, .callout-finding > p { margin: 0; padding: .6rem 1rem .2rem 2.2rem; }
    .callout-finding > p { padding-left: 1rem; }
    .callout-finding > .callout-foot { border-top: 1px dashed var(--border);
      margin-top: .4rem; padding: .45rem 1rem .6rem; }
    .suggestion-toggle { align-items: center; background: var(--accent-soft);
      border: 1px solid var(--accent); border-radius: 6px; color: var(--accent);
      cursor: pointer; display: flex; font: inherit; font-size: .85rem; font-weight: 600;
      gap: .5rem; margin-bottom: .8rem; padding: .4rem .6rem; text-align: left; width: 100%; }
    .suggestion-toggle:hover { background: #dce8f4; }
    .suggestion-toggle[aria-pressed="false"] { background: var(--surface); color: var(--muted);
      border-color: var(--border); }
    .suggestion-toggle .toggle-dot { border: 2px solid currentColor; border-radius: 3px;
      flex: none; height: .8rem; width: .8rem; }
    .suggestion-toggle[aria-pressed="true"] .toggle-dot { background: currentColor; }
    body.hide-suggestions .suggestion { display: none; }
    .callout-foot { color: var(--muted); font-size: .8rem; font-style: italic; }
    details.defs { box-shadow: none; }
    details.defs table th[scope=row] { color: var(--text); white-space: nowrap; }
    details.defs td { font-size: .9rem; }
    .glossary { display: grid; gap: .3rem 1rem; grid-template-columns: max-content 1fr; }
    .glossary dt { font-weight: 700; } .glossary dd { margin: 0; font-size: .92rem; }
    @media (max-width: 700px) { .glossary { grid-template-columns: 1fr; } }
    .lens-grid { display: grid; gap: .8rem; grid-template-columns:
      repeat(auto-fit, minmax(min(200px, 100%), 1fr)); margin: 1rem 0; }
    .lens { background: var(--accent-soft); border-radius: 8px; color: inherit;
      display: flex; flex-direction: column; gap: .3rem; padding: .8rem 1rem;
      text-decoration: none; }
    .lens:hover { outline: 2px solid var(--accent); }
    .lens span { font-size: .92rem; } .lens em { color: var(--muted); font-size: .8rem; }
    .questions li { margin-bottom: .35rem; }
    .reading-guide { border-top: 4px solid var(--accent); }
    .badge { border-radius: 99px; display: inline-block; font-size: .75rem; font-weight: 700;
      padding: .05rem .5rem; }
    .badge-neutral { background: #e4e7eb; color: #3e4c59; }
    .badge-low { background: #e0ecf9; color: #1f4f82; }
    .badge-mid { background: #fdebd0; color: #8a4b00; }
    .badge-high { background: #f9d7d3; color: #9b2c1f; }
    .stat-line { display: flex; flex-wrap: wrap; gap: .4rem 1rem; font-size: .9rem; }
    .stat-line b { font-weight: 700; }
    .pyplot-figure { background: #fff; display: block; height: auto; max-width: 100%; width: 100%; }
    .chapter { margin: 20px 0; }
"""

# Applies one look to every chart: the Python-rendered figures and the ones
# drawn by the chapter scripts. Layout values given by a chart win.
_PLOTLY_DEFAULTS = """
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
"""

_SCRIPT = """
  // Show / hide the reading-suggestion call-outs (How to read, What to look
  // for, Caveats). Visible by default; the choice is remembered per browser.
  (() => {
    const button = document.getElementById("suggestion-toggle");
    if (!button) return;
    const apply = (visible) => {
      document.body.classList.toggle("hide-suggestions", !visible);
      button.setAttribute("aria-pressed", String(visible));
      button.querySelector(".toggle-state").textContent = visible ? "shown" : "hidden";
    };
    let visible = true;
    try { visible = localStorage.getItem("report-suggestions") !== "hidden"; } catch (e) {}
    apply(visible);
    button.addEventListener("click", () => {
      visible = !visible;
      apply(visible);
      try { localStorage.setItem("report-suggestions", visible ? "shown" : "hidden"); } catch (e) {}
    });
  })();
  // Highlight the sidebar entry of the section currently in view. Only
  // elements that have a sidebar link are observed, so plot containers whose
  // ids also start with "chapter-" cannot clear the highlight.
  const sectionLinks = new Map(Array.from(
    document.querySelectorAll("[data-section-link]")
  ).map((link) => [link.dataset.sectionLink, link]));
  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        sectionLinks.forEach((link) => link.classList.remove("active"));
        sectionLinks.get(entry.target.id)?.classList.add("active");
      }
    });
  }, { rootMargin: "-15% 0px -70% 0px" });
  sectionLinks.forEach((link, id) => {
    const section = document.getElementById(id);
    if (section) sectionObserver.observe(section);
  });
  // Charts drawn inside a closed <details> get a default width; resize them
  // to their container when the block is opened.
  document.addEventListener("toggle", (event) => {
    if (event.target.open) {
      event.target.querySelectorAll(".js-plotly-plot").forEach((plot) => Plotly.Plots.resize(plot));
    }
  }, true);
"""


def _sidebar() -> str:
    classes = {-1: "", 0: "supitem", 1: "subitem"}
    return "\n".join(
        f'      <a class="{classes[level]}" href="#{anchor}" '
        f'data-section-link="{anchor}">{html.escape(text)}</a>'
        for anchor, text, level in _INDEX
    )


def build_report(
    analysis_artifact: dict[str, Any] | str | Path,
    *,
    open_report: bool = True,
    wait_for_close: bool = True,
    output_dir: str | Path | None = None,
) -> Path:
    """Render a previously saved analysis artifact.

    Analysis must be completed by ``analysis.analyzer`` first. The builder
    intentionally does not accept raw metadata, so rendering is deterministic
    and cannot accidentally repeat expensive feature analysis.

    ``open_report`` and ``wait_for_close`` are explicit so notebooks and tests
    can render without blocking, while normal analysis runs remain interactive.
    """
    if isinstance(analysis_artifact, (str, Path)):
        artifact_path = Path(analysis_artifact)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    elif isinstance(analysis_artifact, dict):
        artifact = analysis_artifact
    else:
        raise TypeError("analysis_artifact must be a path or dictionary")
    if artifact.get("schema_version") != 2:
        raise ValueError(
            "Unsupported analysis artifact schema version "
            f"{artifact.get('schema_version')!r}; rerun `python -m src.analysis.analyzer`"
        )
    config = artifact.get("config")
    report_data = artifact.get("report_data")
    if not isinstance(config, dict):
        raise ValueError("Analysis artifact config must be a dictionary")
    if not isinstance(report_data, dict):
        raise ValueError("Analysis artifact report_data must be a dictionary")

    required_chapters = ("chapter_1", "chapter_2", "chapter_3", "chapter_4", "chapter_5", "chapter_7")
    missing_chapters = [chapter for chapter in required_chapters if chapter not in report_data]
    if missing_chapters:
        raise ValueError(
            "Analysis artifact is missing required report data: "
            + ", ".join(missing_chapters)
        )
    if not all(isinstance(report_data[chapter], dict) for chapter in required_chapters):
        raise ValueError("Each required report-data chapter must be a dictionary")

    # Chapter 9 is derived from the other chapters' payloads. It is cheap and
    # deterministic, so it is always recomputed here: the copy stored in the
    # artifact (for export) can then never be stale, and artifacts written
    # before chapter 9 existed still render.
    synthesis = build_synthesis(report_data, config)
    report_data = {**report_data, "chapter_9": synthesis}

    bias_stats = report_data["chapter_5"]
    summary = report_data["chapter_1"]
    chapters = [
        render_chapter_one(summary, headline_indicators(report_data),
                           conclusive_insights(report_data)),
        render_chapter_two(report_data["chapter_2"]),
        render_chapter_three(report_data["chapter_3"]),
        render_chapter_four(report_data["chapter_4"]),
        render_chapter_five(bias_stats)
        + render_stereotypical_associations(bias_stats)
        + render_intersectional_analysis(bias_stats),
        render_chapter_six(report_data.get("chapter_6", {})),
        render_chapter_seven(report_data["chapter_7"]),
        render_chapter_eight(report_data.get("chapter_8", {})),
        render_chapter_nine(synthesis),
        render_chapter_ten(report_data, config),
    ]
    chapters_html = "\n".join(
        f'<div class="chapter">{chapter}</div>' for chapter in chapters
    )

    if output_dir is None:
        output_dir = config.get("report_dir", "assets/report")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = config.get("report_filename", "fairness_report.html")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("report_filename must be a filename, not a path")
    report_path = output_dir / filename
    if report_path.suffix.lower() != ".html":
        raise ValueError("report_filename must use the .html extension")

    model = html.escape(str(summary.get("model_name", "Unknown model")))
    subtitle = (
        f"Text-to-image fairness audit of <strong>{model}</strong> · "
        f"{html.escape(str(summary.get('generated_images', '?')))} images from "
        f"{html.escape(str(summary.get('num_prompts', '?')))} prompts · "
        f"audit date {html.escape(str(summary.get('audit_date', '?')))}"
    )
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fairness report – {model}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script>{_PLOTLY_DEFAULTS}</script>
  <style>{_STYLE}</style>
</head>
<body>
<div class="report-layout">
  <aside class="sidebar" aria-label="Report index">
    <nav>
      <button type="button" class="suggestion-toggle" id="suggestion-toggle"
        aria-pressed="true" title="Show or hide the How to read / What to look for / Caveats boxes">
        <span class="toggle-dot" aria-hidden="true"></span>
        <span>Reading suggestions: <span class="toggle-state">shown</span></span>
      </button>
      <div class="sidebar-title">Report index</div>
{_sidebar()}
    </nav>
  </aside>
  <main>
    <header class="report-header">
      <h1 id="report-title">Fairness report</h1>
      <p class="subtitle">{subtitle}</p>
    </header>
    {render_reading_guide()}
    {chapters_html}
  </main>
</div>
<script>{_SCRIPT}</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    if open_report:
        webbrowser.open(report_path.resolve().as_uri())
        if wait_for_close:
            try:
                input("Fairness report is open. Press Enter when you are finished viewing it. ")
            except EOFError:
                pass
    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a fairness report from a saved analysis artifact."
    )
    parser.add_argument(
        "--input", "-i", default="assets/analysis.json",
        help="Path to the analysis JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Folder the report is written to (default: report_dir from the "
             "analysis config, else assets/report)",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open the report")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for input after opening the report",
    )
    args = parser.parse_args()
    build_report(
        args.input,
        output_dir=args.output_dir,
        open_report=not args.no_open,
        wait_for_close=not args.no_wait,
    )
