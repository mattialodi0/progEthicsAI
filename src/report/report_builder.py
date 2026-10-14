"""Build the HTML fairness-audit report from chapter renderers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .report_ch_1 import render_chapter_one
from .report_ch_2 import build_coverage, render_chapter_two


def build_report(
    data: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Write the interactive fairness report and return its path.

    Each chapter owns its data preparation and HTML rendering in a separate
    module.  The builder only validates inputs, assembles the document, and
    exposes structured data to future interactive chapters.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if not isinstance(analysis_results, dict):
        raise TypeError("analysis_results must be a dictionary")
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    chapter_one = render_chapter_one(data, analysis_results, config)
    coverage = build_coverage(data, config)
    chapter_two = render_chapter_two(coverage)
    report_data = {
        "chapter_1": chapter_one.data,
        "chapter_2": coverage,
    }
    serialized_data = json.dumps(report_data, ensure_ascii=True).replace("</", "<\\/")

    output_dir = Path(config.get("report_dir", "report"))
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
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ background: #f4f6f8; color: #17202a; margin: 0; }}
    main {{ margin: 0 auto; max-width: 1100px; padding: 2rem 1rem 4rem; }}
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
  </style>
</head>
<body>
<main>
  <h1>Fairness report</h1>
  <p class="subtitle">Interactive audit report</p>
  {chapter_one.html}
  {chapter_two}
</main>
<script>
  const reportData = {serialized_data};
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path
