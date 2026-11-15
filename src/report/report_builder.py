"""Build the HTML fairness-audit report from a saved analysis artifact.

    python -m src.report.report_builder -i assets/analysis.json --no-wait
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any
import webbrowser

from ..analysis.synthesis import build_synthesis
from .chapters import (
    render_chapter_eight,
    render_chapter_five,
    render_chapter_four,
    render_chapter_nine,
    render_chapter_one,
    render_chapter_seven,
    render_chapter_six,
    render_chapter_ten,
    render_chapter_three,
    render_chapter_two,
    render_intersectional_analysis,
    render_reading_guide,
    render_stereotypical_associations,
)
from .insights import conclusive_insights, headline_indicators
from .page import render_page

SCHEMA_VERSION = 2
REQUIRED_CHAPTERS = ("chapter_1", "chapter_2", "chapter_3", "chapter_4", "chapter_5", "chapter_7")


def _load_artifact(analysis_artifact: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(analysis_artifact, (str, Path)):
        return json.loads(Path(analysis_artifact).read_text(encoding="utf-8"))
    if isinstance(analysis_artifact, dict):
        return analysis_artifact
    raise TypeError("analysis_artifact must be a path or dictionary")


def _validate(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the artifact's (config, report_data) after checking their shape."""
    if artifact.get("schema_version") != SCHEMA_VERSION:
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

    missing_chapters = [chapter for chapter in REQUIRED_CHAPTERS if chapter not in report_data]
    if missing_chapters:
        raise ValueError(
            "Analysis artifact is missing required report data: "
            + ", ".join(missing_chapters)
        )
    if not all(isinstance(report_data[chapter], dict) for chapter in REQUIRED_CHAPTERS):
        raise ValueError("Each required report-data chapter must be a dictionary")
    return config, report_data


def _render_chapters(report_data: dict[str, Any], config: dict[str, Any]) -> str:
    bias_stats = report_data["chapter_5"]
    chapters = [
        render_chapter_one(report_data["chapter_1"], headline_indicators(report_data),
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
        render_chapter_nine(report_data["chapter_9"]),
        render_chapter_ten(report_data, config),
    ]
    return "\n".join(f'<div class="chapter">{chapter}</div>' for chapter in chapters)


def _report_path(config: dict[str, Any], output_dir: str | Path | None) -> Path:
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
    return report_path


def _subtitle(summary: dict[str, Any], model: str) -> str:
    return (
        f"Text-to-image fairness audit of <strong>{model}</strong> · "
        f"{html.escape(str(summary.get('generated_images', '?')))} images from "
        f"{html.escape(str(summary.get('num_prompts', '?')))} prompts · "
        f"audit date {html.escape(str(summary.get('audit_date', '?')))}"
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
    config, report_data = _validate(_load_artifact(analysis_artifact))

    # Chapter 9 is derived from the other chapters' payloads. It is cheap and
    # deterministic, so it is always recomputed here: the copy stored in the
    # artifact (for export) can then never be stale, and artifacts written
    # before chapter 9 existed still render.
    report_data = {**report_data, "chapter_9": build_synthesis(report_data, config)}

    body = render_reading_guide() + "\n    " + _render_chapters(report_data, config)
    report_path = _report_path(config, output_dir)
    summary = report_data["chapter_1"]
    model = html.escape(str(summary.get("model_name", "Unknown model")))
    report_path.write_text(render_page(model, _subtitle(summary, model), body),
                           encoding="utf-8")

    if open_report:
        webbrowser.open(report_path.resolve().as_uri())
        if wait_for_close:
            try:
                input("Fairness report is open. Press Enter when you are finished viewing it. ")
            except EOFError:
                pass
    return report_path


def main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args(argv)
    build_report(
        args.input,
        output_dir=args.output_dir,
        open_report=not args.no_open,
        wait_for_close=not args.no_wait,
    )


if __name__ == "__main__":
    main()
