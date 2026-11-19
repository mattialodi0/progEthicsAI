"""Renderer for chapter 7: image and embedding analysis.

The data is built by ``src.analysis.visual_features.build_visual_feature_analysis``.
"""

from __future__ import annotations

from typing import Any

from ..explain import chapter_header, findings, guide, metric_defs
from ..insights import chapter_seven
from ..static_files import script_json, static_text

_SCRIPT = static_text("chapter_7.js")


def _quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _box_stats(values: list[Any]) -> dict[str, float] | None:
    """Summarise a distribution for a precomputed Plotly box.

    The artifact stores every pairwise distance (tens of thousands per
    grouping); shipping only the box statistics keeps the report small.
    """
    ordered = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not ordered:
        return None
    q1, median, q3 = (_quantile(ordered, q) for q in (0.25, 0.5, 0.75))
    spread = 1.5 * (q3 - q1)
    return {
        "q1": q1, "median": median, "q3": q3,
        "lowerfence": next(value for value in ordered if value >= q1 - spread),
        "upperfence": next(value for value in reversed(ordered) if value <= q3 + spread),
        "mean": sum(ordered) / len(ordered), "n": len(ordered),
    }


def _compact(chapter_data: dict[str, Any]) -> dict[str, Any]:
    embedding = dict(chapter_data.get("embedding") or {})
    groups = {}
    for name, item in (embedding.get("groups") or {}).items():
        distances = item.get("distances") or {}
        groups[name] = {
            "separability": item.get("separability") or {},
            "distances": {key: _box_stats(values or []) for key, values in distances.items()},
        }
    embedding["groups"] = groups
    return {**chapter_data, "embedding": embedding}


def render_chapter_seven(chapter_data: dict[str, Any]) -> str:
    """Render interactive feature and embedding plots."""
    serialized = script_json(_compact(chapter_data))
    return f"""
  {chapter_header(7, "Visual features and embeddings",
      "Are some groups depicted differently — lighter or darker, closer or farther, more "
      "or less distinct — independently of what the prompt asked for?",
      "<p>Chapters 3–6 work with categorical labels. This chapter looks at continuous image "
      "properties and at the CLIP embedding space, where every image is a point and "
      "similar images lie close together. Differences in framing, colour or composition "
      "between demographic groups are a subtle form of representational bias: e.g. one "
      "group consistently shown in darker, lower-contrast images, or as small figures "
      "rather than portraits.</p>")}
  {findings(chapter_seven(chapter_data))}
  {metric_defs([
      ("Brightness", "Mean pixel luminance of the image (0 = black).",
       "Systematic differences between groups can reflect lighting or skin-tone rendering "
       "choices."),
      ("Saturation / contrast", "Colour intensity and the spread of luminance values.",
       "Lower values give a duller, flatter, often more sombre look."),
      ("Number of people", "People detected in the image.", "More than 1 for a single-person "
       "prompt means the model added people (crowds, colleagues)."),
      ("Person / face area ratio", "Share of the image covered by person and by face "
       "bounding boxes.", "High face ratio = close-up portrait; low = distant figure. "
       "Portraits individualise, distant figures depersonalise."),
      ("Detected age", "Age in years estimated by the face model (mean over the faces of "
       "the image), only for prompts that do not request an age.", "Shows how old the model "
       "draws a job, gender, race or status by default, without the young / middle-aged / "
       "old cut-offs. The estimates are compressed towards 30–40: compare groups, do not "
       "read the years literally."),
      ("Silhouette score", "Mean of (b − a) / max(a, b) over images, where a is the mean "
       "distance to the own group and b to the nearest other group.", "−1 to 1; near 0 = "
       "groups overlap, &gt; 0.25 = noticeable structure, &gt; 0.5 = clearly separate."),
  ])}
  <section id="chapter-7-features"><h3>7.1 Image-feature distributions</h3>
    {guide(
        read=["The histogram shows the feature over all images. The box plots split it by "
              "the selected prompt attribute: the line is the median, the box the middle "
              "50%, the dashed line the mean; dots are outliers.",
              "The dashed vertical/horizontal reference is the overall median."],
        look=["Boxes of demographic groups (gender, race, age) that sit clearly above or "
              "below the others for brightness, contrast or saturation: a group rendered "
              "with a different visual style.",
              "Face or person area ratios that differ by group: one group shown as close "
              "portraits, another as distant or partial figures.",
              "Number of people &gt; 1 concentrated in some groups (e.g. crowds for some "
              "continents or statuses).",
              "Detected age that differs by job, gender or status (e.g. women drawn younger "
              "than men, some jobs drawn older): an age stereotype that the three age groups "
              "of chapter 5 are too coarse to show."],
        caution="Brightness and colour also depend on skin tone and on settings that the "
                "prompt implies (e.g. a construction site vs an office). Compare groups "
                "within the same prompt mode where possible.",
    )}
    <div id="chapter-7-feature-controls" class="checkbox-grid"></div>
    <div class="plot-grid"><article class="plot-card"><div id="chapter-7-feature-plot"></div></article>
    <article class="plot-card"><div id="chapter-7-feature-groups"></div></article></div>
  </section>
  <section id="chapter-7-embeddings"><h3>7.2 Embedding distributions and cluster separability</h3>
    <p>Each image's CLIP embedding is projected to two dimensions. PCA is fit once to all
    embeddings, so every grouping is shown in the same coordinates and colourings are
    directly comparable. UMAP is shown when available.</p>
    {guide(
        read=["Scatter: one point per image, coloured by the selected prompt attribute.",
              "Distance plot: distances between images of the same group vs of different "
              "groups.",
              "Separability bars: silhouette score for every grouping; the selected one is "
              "highlighted."],
        look=["Groups that form their own cluster although the prompt only changed a "
              "demographic word: the model renders that group in a distinct visual "
              "“register” (setting, style, clothing), beyond the person.",
              "Within-group distances clearly smaller than between-group distances.",
              "Compare demographic groupings with job / workplace: a scene-defining attribute "
              "is expected to separate more."],
        caution=["High separability means the information is encoded in the embeddings, not "
                 "that the encoding is unfair: a requested attribute must leave a trace.",
                 "2-D projections distort distances; rely on the silhouette scores and "
                 "distance distributions rather than on the visual impression."],
    )}
    <div id="chapter-7-embedding-controls" class="checkbox-grid"></div>
    <div id="chapter-7-separability"></div>
    <article class="plot-card"><div id="chapter-7-embedding-plot"></div></article>
    <div class="plot-grid">
        <article class="plot-card"><div id="chapter-7-distance-plot"></div></article>
        <article class="plot-card"><div id="chapter-7-cluster-separability"></div></article>
    </div>
  </section>
  <script>
    const chapterSevenData = {serialized};
{_SCRIPT}  </script>
"""
