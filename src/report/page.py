"""The HTML page around the chapters: head, sidebar index, header and scripts."""

from __future__ import annotations

import html

from .static_files import static_text


# (anchor id, label, level) — level 0 is a chapter, 1 a section.
INDEX = (
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


def sidebar() -> str:
    classes = {-1: "", 0: "supitem", 1: "subitem"}
    return "\n".join(
        f'      <a class="{classes[level]}" href="#{anchor}" '
        f'data-section-link="{anchor}">{html.escape(text)}</a>'
        for anchor, text, level in INDEX
    )


def render_page(model: str, subtitle: str, body: str) -> str:
    """Return the complete report document.

    ``model`` and ``subtitle`` must already be escaped; ``body`` holds the
    reading guide and the rendered chapters.
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fairness report – {model}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script>
{static_text("plotly_defaults.js")}</script>
  <style>
{static_text("report.css")}</style>
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
{sidebar()}
    </nav>
  </aside>
  <main>
    <header class="report-header">
      <h1 id="report-title">Fairness report</h1>
      <p class="subtitle">{subtitle}</p>
    </header>
    {body}
  </main>
</div>
<script>
{static_text("report.js")}</script>
</body>
</html>
"""
