"""Renderer for chapter 3: Demographic representation and diversity."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from ..explain import chapter_header, findings, guide, metric_defs
from ..insights import chapter_three
from ..plotly_helpers import chart, numeric
from ..static_files import script_json, static_text
from .ch3_data import composition_payload, diversity_payload

_SCRIPT = static_text("chapter_3.js")


def _rare_chart(rare: dict[str, Any]) -> str:
    entries = [(str(name), len(values)) for name, values in rare.items()
               if isinstance(values, list) and values]
    if not entries:
        return '<p class="note">No prompt category occurs in exactly one image, so no group ' \
               'estimate rests on a single generation.</p>'
    figure = go.Figure(go.Bar(
        x=[count for _, count in entries], y=[name for name, _ in entries], orientation="h",
        customdata=[", ".join(str(item.get("value")) for item in rare[name]) for name, _ in entries],
        hovertemplate="%{y}: %{x}<br>%{customdata}<extra></extra>",
    ))
    figure.update_xaxes(title="Categories occurring exactly once")
    return chart(figure, title="", height=max(240, 60 + 40 * len(entries)))


def _sample_count_chart(rows: list[dict[str, Any]]) -> str:
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return '<p class="note">Not enough prompt-mode groups for this comparison.</p>'
    figure = go.Figure(go.Scatter(
        x=[row.get("sample_count") for row in rows],
        y=[numeric(row.get("mean_embedding_dispersion")) for row in rows],
        mode="lines+markers",
        hovertemplate="Groups with at least %{x} images<br>Mean dispersion: %{y:.3f}<extra></extra>",
    ))
    figure.update_xaxes(title="Minimum images per prompt-mode group", type="category")
    figure.update_yaxes(title="Mean embedding dispersion")
    return chart(figure, title="", height=320)


def render_chapter_three(chapter_data: dict[str, Any]) -> str:
    """Render dynamically aggregated composition and diversity charts."""
    modes = [str(mode) for mode in chapter_data.get("prompt_modes", [])]
    composition_data, context_data = composition_payload(chapter_data, modes)
    diversity = chapter_data.get("diversity", {})
    if not isinstance(diversity, dict):
        diversity = {}
    diversity_data = diversity_payload(diversity, modes)

    composition_cards = "".join(
        f'<article class="plot-card"><h4>Detected {html.escape(str(dimension))}</h4>'
        f'<div id="chapter-3-composition-{index}"></div></article>'
        for index, dimension in enumerate(composition_data)
    )
    selector_options = "".join(
        f'<label><input type="checkbox" value="{html.escape(mode)}" checked> '
        f'{html.escape(mode)}</label>'
        for mode in modes
    )
    group_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.replace("_", " "))}</option>'
        for name in diversity_data
    )
    dimension_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in context_data
    )
    metric_table = metric_defs([
        ("Embedding dispersion", "Mean Euclidean distance between every pair of CLIP image "
         "embeddings in the group.", "Higher = more visual / semantic variety. Compare groups "
         "with each other; the absolute value has no natural threshold."),
        ("Mean pairwise similarity", "Mean cosine similarity between every pair of image "
         "embeddings in the group.", "Higher = images look more alike (more repetitive). "
         "Reads in the opposite direction from dispersion."),
        ("Normalized gender entropy", "Shannon entropy of the detected gender distribution "
         "divided by its maximum.", "1 = genders evenly represented; 0 = only one gender "
         "detected. Expected to be ~0 when the prompt requests a gender."),
        ("Unique objects", "Number of distinct object labels detected across the group's "
         "images.", "Higher = richer scenes. Grows with group size, so compare groups of "
         "similar size."),
    ])
    return f"""
  {chapter_header(3, "Demographic representation and diversity",
      "Who does the model generate, and how varied are the images it produces?",
      "<p>Two different questions are kept apart here. <strong>Representation</strong> "
      "(3.1) asks which demographic groups appear in the generated images. "
      "<strong>Diversity</strong> (3.2) asks how varied the images are within each group. "
      "A model can produce a balanced number of men and women while drawing every woman "
      "in the same way — or produce varied images of only one group.</p>")}
  {findings(chapter_three(chapter_data))}
  <div class="controls" id="chapter-3-mode-selector" aria-label="Prompt modes included">
    <strong>Prompt modes included in 3.1 and 3.2:</strong>{selector_options}
  </div>
  <section id="chapter-3-distributions" aria-labelledby="chapter-3-distributions-heading">
    <h3 id="chapter-3-distributions-heading">3.1 Demographic distributions</h3>
    <p>Detected demographic composition of the generated images in each selected prompt
    mode, plus the pooled total of the selection.</p>
    {guide(
        read="Each horizontal bar sums to 100% and is split by the detected category; "
             "hover for counts. Compare the rows: the single-variable row shows the model's "
             "default when the attribute is (mostly) not requested.",
        look=["In single-variable prompts, a demographic far from the population you would "
              "expect (e.g. 90% male for gender-neutral prompts) indicates the model fills "
              "unspecified attributes with a default group.",
              "Categories that never or almost never appear indicate erasure.",
              "In control and double-variable prompts the composition is largely determined "
              "by what was requested; differences from the requested mix are measured in "
              "chapter 4."],
        caution=["Composition must be read against the prompt design: if 70% of prompts "
                 "request one category, a 70% output share is not evidence of bias.",
                 "Labels are DeepFace predictions over a small fixed set of categories "
                 "(e.g. 7 race classes), so they cannot capture the full diversity of "
                 "people and may be systematically wrong for some groups."],
    )}
    <div class="plot-grid">{composition_cards}</div>
  </section>
  <section id="chapter-3-context" aria-labelledby="chapter-3-context-heading">
    <h3 id="chapter-3-context-heading">3.1.1 Demographic composition by job and workplace</h3>
    <p>The detected composition for each job or workplace, with all prompt modes pooled
    (explicit gender, race or age requests pull the bars toward balance, so real skews
    are probably a bit stronger). Jobs dominated by one group, or opposite skews for
    related jobs, are what to look for.</p>
    <div class="controls">
      <label>Detected attribute <select id="chapter-3-context-dimension">{dimension_options}</select></label>
      <label>Split by <select id="chapter-3-context-split">
        <option value="job">Job</option><option value="workplace">Workplace</option></select></label>
    </div>
    <article class="plot-card"><div id="chapter-3-context-plot"></div></article>
  </section>
  <section id="chapter-3-diversity" aria-labelledby="chapter-3-diversity-heading">
    <h3 id="chapter-3-diversity-heading">3.2 Diversity</h3>
    <p>Diversity of the generated images within each group, measured four ways. When several
    prompt modes are selected, each value is the image-weighted mean of its per-mode values.</p>
    {metric_table}
    {guide(
        read="Each panel ranks the groups of the selected grouping on one metric; the dashed "
             "line is the image-weighted mean of the groups shown. Hover for the number of "
             "images.",
        look=["A group with clearly lower dispersion (or higher similarity) than comparable "
              "groups is depicted more uniformly — a sign of a narrow, possibly stereotyped "
              "visual template.",
              "Gender entropy near 0 for a job whose prompt did not specify gender means one "
              "gender is drawn almost always."],
        caution="Higher diversity is not automatically better: a prompt that specifies more "
                "attributes (control mode) should produce less variety than an open prompt, "
                "and some subjects (e.g. a soldier's uniform) are visually constrained.",
    )}
    <div class="controls">
      <label>Group images by <select id="chapter-3-diversity-group">{group_options}</select></label>
    </div>
    <div class="plot-grid">
      <article class="plot-card"><div id="chapter-3-diversity-embedding_dispersion"></div></article>
      <article class="plot-card"><div id="chapter-3-diversity-pairwise_similarity"></div></article>
      <article class="plot-card"><div id="chapter-3-diversity-demographic_entropy"></div></article>
      <article class="plot-card"><div id="chapter-3-diversity-object_unique"></div></article>
    </div>
    <details><summary>Robustness checks: rare categories and sample size</summary>
      <div class="plot-grid">
        <article class="plot-card"><h4>Rare categories by attribute</h4>
          <p class="note">Prompt categories occurring in exactly one image (all prompt
          modes). Estimates for these groups rest on a single generation.</p>
          {_rare_chart(diversity.get("rare_categories") or {})}</article>
        <article class="plot-card"><h4>Diversity versus sample count</h4>
          <p class="note">Mean embedding dispersion of the prompt-mode groups with at least
          the given number of images. A flat line means dispersion is not driven by group
          size.</p>
          {_sample_count_chart(diversity.get("by_sample_count") or [])}</article>
      </div>
    </details>
  </section>
  <script>
    const chapterThreeComposition = {script_json(composition_data)};
    const chapterThreeContext = {script_json(context_data)};
    const chapterThreeDiversity = {script_json(diversity_data)};
    const chapterThreeDimensions = {script_json(list(composition_data))};
{_SCRIPT}  </script>
"""
