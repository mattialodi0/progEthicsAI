"""Build the HTML fairness-audit report from chapter renderers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import webbrowser

from .report_ch_1 import render_chapter_one
from .report_ch_2 import render_chapter_two
from .report_ch_3 import render_chapter_three
from .report_ch_4 import render_chapter_four
from .report_ch_5_1 import render_chapter_five
from .report_ch_5_3 import render_stereotypical_associations
from .report_ch_5_4 import render_intersectional_analysis
from .report_ch_7 import render_chapter_seven


def build_report(
    analysis_artifact: dict[str, Any] | str | Path,
    *,
    open_report: bool = True,
    wait_for_close: bool = True,
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
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported analysis artifact schema version")
    config = artifact.get("config")
    analysis_results = artifact.get("analysis_results")
    report_data = artifact.get("report_data")
    if not isinstance(config, dict):
        raise ValueError("Analysis artifact config must be a dictionary")
    if not isinstance(analysis_results, dict):
        raise ValueError("Analysis artifact analysis_results must be a dictionary")
    if not isinstance(report_data, dict):
        raise ValueError("Analysis artifact report_data must be a dictionary")

    chapter_one = render_chapter_one(report_data["chapter_1"])
    coverage = report_data["chapter_2"]
    chapter_two = render_chapter_two(coverage)
    chapter_three_data = report_data["chapter_3"]
    chapter_three = render_chapter_three(chapter_three_data)
    chapter_four = render_chapter_four(report_data["chapter_4"])
    bias_stats = report_data["chapter_5"]
    chapter_five = (
        render_chapter_five(bias_stats)
        + render_stereotypical_associations(bias_stats)
        + render_intersectional_analysis(bias_stats)
    )
    chapter_seven = render_chapter_seven(report_data["chapter_7"])
    report_data = {
        "chapter_1": chapter_one.data,
        "chapter_2": coverage,
        "chapter_3": chapter_three_data,
        "chapter_4": report_data["chapter_4"],
        "chapter_5": bias_stats,
        "chapter_7": report_data["chapter_7"],
    }
    serialized_data = json.dumps(report_data, ensure_ascii=True).replace("</", "<\\/")

    output_dir = Path(config.get("report_dir", "assets/report"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / config.get("report_filename", "fairness_report.html")
    if report_path.suffix.lower() != ".html":
        raise ValueError("report_filename must use the .html extension")

    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fairness report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ background: #f4f6f8; color: #17202a; margin: 0; }}
    .report-layout {{ display: grid; gap: 2rem; grid-template-columns: 220px minmax(0, 1fr);
      margin: 0 auto; max-width: 1320px; padding: 2rem 1rem 4rem; }}
    main {{ min-width: 0; }}
    .sidebar {{ align-self: start; position: sticky; top: 1rem; }}
    .sidebar nav {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px;
      box-shadow: 0 2px 5px #102a430d; padding: 1rem; overflow-y: scroll; max-height: 90vh; }}
    .sidebar-title {{ color: #52606d; font-size: .8rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase; }}
    .sidebar a {{ border-left: 3px solid transparent; color: #334e68; display: block;
      padding: .45rem .6rem; text-decoration: none; }}
    .sidebar a:hover, .sidebar a.active {{ background: #f0f4f8;
      border-left-color: #486581; color: #102a43; }}
    .sidebar .supitem {{ font-weight: bold; }}
    .sidebar .subitem {{ font-size: .9rem; padding-left: 1.25rem; }}
    @media (max-width: 800px) {{ .report-layout {{ display: block; }}
      .sidebar {{ margin-bottom: 1rem; position: static; }} .sidebar nav {{
        display: flex; gap: .25rem; overflow-x: auto; white-space: nowrap; }}
      .sidebar-title {{ display: none; }} .sidebar .subitem {{ display: none; }} }}
    .eyebrow {{ color: #52606d; font-size: .8rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase; }}
    h1 {{ margin-bottom: .35rem; }} h2 {{ margin-top: 2rem; }}
    .subtitle {{ color: #52606d; margin-top: 0; }}
    .metrics {{ display: grid; gap: 1rem; grid-template-columns:
      repeat(auto-fit, minmax(170px, 1fr)); }}
    .metric, section, details {{ background: white; border: 1px solid #d9e2ec;
      border-radius: 8px; box-shadow: 0 2px 5px #102a430d; }}
    .metric {{ padding: 1rem; }} .metric-label {{ color: #52606d; display: block;
      font-size: .9rem; }} .metric-value {{ display: block; font-size: 1.6rem;
      margin-top: .35rem; }} section, details {{ padding: 1rem 1.25rem; }}
    details {{ margin-top: 1rem; }} summary {{ cursor: pointer; font-weight: 700; }}
    pre {{ background: #f4f6f8; border-radius: 6px; overflow: auto; padding: 1rem; }}
    table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: .55rem; text-align: left; }}
    th {{ color: #52606d; font-size: .85rem; }} .bar {{ background: #d9e2ec;
      border-radius: 99px; height: .55rem; min-width: 3rem; overflow: hidden; }}
    .bar span {{ background: #486581; display: block; height: 100%; }}
    .conditioning-plot {{ display: grid; gap: .65rem; margin: 1rem 0; }}
    .plot-row {{ align-items: center; display: grid; gap: .75rem;
      grid-template-columns: minmax(110px, 1fr) 3fr auto; }}
    .plot-label, .plot-value {{ font-size: .9rem; }} .plot-value {{ color: #52606d;
      min-width: 90px; text-align: right; }} .plot-track {{ background: #d9e2ec;
      border-radius: 99px; height: .7rem; overflow: hidden; }}
    .plot-track span {{ background: #486581; display: block; height: 100%; }}
    .placeholder {{ border-left: 4px solid #829ab1; color: #52606d; padding-left: 1rem; }}
    .note {{ color: #52606d; font-size: .95rem; }}
    .checkbox-grid {{ display: flex; flex-wrap: wrap; gap: .75rem 1.25rem; }}
    .checkbox-grid label {{ cursor: pointer; }}
    .plot-card {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px;
      margin-top: 1rem; padding: 1rem; }}
    .plot-grid {{ display: grid; gap: 1rem; grid-template-columns:
      repeat(auto-fit, minmax(320px, 1fr)); }}
    .svg-figure {{ margin: .75rem 0; min-width: 0; }}
    .svg-figure svg {{ background: #f8fafc; border: 1px solid #d9e2ec;
      border-radius: 6px; display: block; max-width: 100%; min-height: 80px;
      width: 100%; }}
    .svg-label {{ fill: #243b53; font-size: 13px; }}
    .svg-value {{ fill: #243b53; font-size: 12px; }}
    .plot-legend {{ display: inline-block; font-size: .82rem; margin: .35rem .75rem 0 0; }}
    .plot-legend i {{ border-radius: 2px; display: inline-block; height: .75rem;
      margin-right: .3rem; vertical-align: -.1rem; width: .75rem; }}
    .pie-layout {{ align-items: center; display: flex; flex-wrap: wrap; gap: 1.25rem; }}
    .pie-chart {{ border: 1px solid #d9e2ec; border-radius: 50%; flex: 0 0 190px;
      height: 190px; width: 190px; }}
  </style>
</head>
<body>
<div class="report-layout">
  <aside class="sidebar" aria-label="Report index">
    <nav>
      <div class="sidebar-title">Report index</div>
      <a href="#report-title" data-section-link="report-title">Fairness report</a>
      <a class="supitem" href="#chapter-1" data-section-link="chapter-1">1. Executive summary</a>
      <a class="subitem" href="#chapter-1-model" data-section-link="chapter-1-model">Model and configuration</a>
      <a class="subitem" href="#chapter-1-coverage" data-section-link="chapter-1-coverage">Audit coverage</a>
      <a class="subitem" href="#chapter-1-fairness" data-section-link="chapter-1-fairness">Fairness overview</a>
      <a class="subitem" href="#chapter-1-scope" data-section-link="chapter-1-scope">Scope and limitations</a>
      <a class="supitem" href="#chapter-2" data-section-link="chapter-2">2. Dataset and generation coverage</a>
      <a class="subitem" href="#chapter-2-attributes" data-section-link="chapter-2-attributes">Prompt attributes</a>
      <a class="subitem" href="#chapter-2-prompts" data-section-link="chapter-2-prompts">Images per prompt</a>
      <a class="subitem" href="#chapter-2-detections" data-section-link="chapter-2-detections">Detection rates</a>
      <a class="subitem" href="#chapter-2-settings" data-section-link="chapter-2-settings">Generation settings</a>
      <a class="supitem" href="#chapter-3" data-section-link="chapter-3">3. Demographic representation and diversity</a>
      <a class="subitem" href="#chapter-3-controls" data-section-link="chapter-3-controls">Prompt modes</a>
      <a class="subitem" href="#chapter-3-distributions" data-section-link="chapter-3-distributions">Demographic distributions</a>
      <a class="subitem" href="#chapter-3-diversity" data-section-link="chapter-3-diversity">Diversity</a>
      <a class="supitem" href="#chapter-4" data-section-link="chapter-4">4. Prompt adherence</a>
      <a class="subitem" href="#chapter-4-attributes" data-section-link="chapter-4-attributes">Overall adherence</a>
      <a class="subitem" href="#chapter-4-modes" data-section-link="chapter-4-modes">Prompt modes</a>
      <a class="subitem" href="#chapter-4-demographics" data-section-link="chapter-4-demographics">Demographic categories</a>
      <a class="subitem" href="#chapter-4-combinations" data-section-link="chapter-4-combinations">Attribute combinations</a>
      <a class="supitem" href="#chapter-5" data-section-link="chapter-5">5. Conditional distributions and Association analysis</a>
      <a class="subitem" href="#chapter-5-single-variable" data-section-link="chapter-5-single-variable">Single-variable associations</a>
      <a class="subitem" href="#chapter-5-double-variable" data-section-link="chapter-5-double-variable">Double-variable associations</a>
      <a class="subitem" href="#chapter-5-stereotypical" data-section-link="chapter-5-stereotypical">Stereotypical item associations</a>
      <a class="subitem" href="#chapter-5-intersectional" data-section-link="chapter-5-intersectional">Intersectional analysis</a>
      <a class="supitem" href="#chapter-7" data-section-link="chapter-7">7. Visual feature and embedding analysis</a>
      <a class="subitem" href="#chapter-7-features" data-section-link="chapter-7-features">Image-feature distributions</a>
      <a class="subitem" href="#chapter-7-embeddings" data-section-link="chapter-7-embeddings">Embedding separability</a>
    </nav>
  </aside>
  <main>
    <h1 id="report-title">Fairness report</h1>
    <p class="subtitle">Interactive audit report</p>
    {chapter_one.html}
    {chapter_two}
    {chapter_three}
    {chapter_four}
    {chapter_five}
    {chapter_seven}
  </main>
</div>
<script>
  const reportData = {serialized_data};
  renderChapterThree();
  const reportSections = document.querySelectorAll("[data-section-link]");
  const sectionLinks = new Map(Array.from(reportSections).map((link) => [
    link.dataset.sectionLink, link
  ]));
  const sectionObserver = new IntersectionObserver((entries) => {{
    entries.forEach((entry) => {{
      if (entry.isIntersecting) {{
        sectionLinks.forEach((link) => link.classList.remove("active"));
        sectionLinks.get(entry.target.id)?.classList.add("active");
      }}
    }});
  }}, {{ rootMargin: "-15% 0px -70% 0px" }});
  document.querySelectorAll("[id^='chapter-'], #report-title").forEach((section) => {{
    sectionObserver.observe(section);
  }});
</script>
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
    parser.add_argument("analysis_artifact", help="Path to the analysis JSON file")
    parser.add_argument("--no-open", action="store_true", help="Do not open the report")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for input after opening the report",
    )
    args = parser.parse_args()
    build_report(
        args.analysis_artifact,
        open_report=not args.no_open,
        wait_for_close=not args.no_wait,
    )
