"""Renderer for chapter 5.4: intersectional analysis."""

from __future__ import annotations

from typing import Any

from ..explain import findings, guide, metric_defs
from ..insights import intersectional
from ..static_files import script_json, static_text

_SCRIPT = static_text("chapter_5_4.js")


def render_intersectional_analysis(bias_stats: dict[str, Any]) -> str:
    """Render joint-group heatmaps, comparisons, and interaction tests."""
    data = bias_stats.get("intersectional", {})
    serialized = script_json(data)
    return f"""
  <section id="chapter-5-intersectional" aria-labelledby="chapter-5-intersectional-heading">
    <h3 id="chapter-5-intersectional-heading">5.4 Intersectional analysis</h3>
    <p>Does a joint group (e.g. “black female”, “asian nurse”) behave differently from what
    its two constituent groups would predict? An intersectional effect exists when the
    combination is more — or less — than the sum of its parts: for instance, if the model
    draws women and Black people with typical rates of a feature, but Black women with a
    very different rate. Ordinary demographic differences (chapter 5.1) cannot reveal
    this.</p>
    <p class="note">Images whose prompt itself requests the detected feature are left out
    (e.g. “young white female nurse” is not used for race × job → detected gender), so every
    share describes the model's own choice rather than adherence to a request.</p>
    {findings(intersectional(data))}
    {metric_defs([
        ("Joint − constituent average (forest plot)", "P(category | joint group) minus the "
         "mean of P(category | first group) and P(category | second group), with a 95% CI.",
         "0 = the joint group is what its parts predict. A CI that excludes 0 is a reliable "
         "deviation; its sign gives the direction."),
        ("Interaction effect β3", "Coefficient of the G × R term in "
         "<em>Y = β0 + β1G + β2R + β3(G × R)</em>, for a binary outcome (has the selected "
         "category), in percentage points.", "The extra change for that combination after "
         "both main effects. Large |β3| = strong intersectional effect."),
        ("Interaction p-value", "Likelihood-ratio-style test comparing the model with "
         "and without interaction terms (all interaction terms = 0).",
         "&lt; 0.05 suggests a real intersectional effect. Many models are tested, so a few "
         "small p-values are expected by chance."),
        ("Difference-in-differences", "How the gap between two levels of the first "
         "attribute changes between two levels of the second attribute.",
         "Non-parallel lines = the effect of one attribute depends on the other."),
    ])}
    {guide(
        read="Select a prompt pair, a detected feature and one detected category. All five "
             "plots then describe the share of faces detected in that category. Every group "
             "shows its n; with only 10 images per joint group, ±10 pp is a single image.",
        look=["Forest-plot points whose CI does not cross the dashed zero line.",
              "Interaction bars that are large and consistent in direction for one group "
              "(e.g. every combination with “female” moves the same way).",
              "Crossing or diverging lines in the difference-in-differences plot.",
              "The same joint group standing out for several detected features."],
        caution="Intersectional cells are the smallest samples in the report. Treat single "
                "deviations as hypotheses to re-test with more generations, not as findings.",
    )}
    <div id="chapter-5-four-controls" class="checkbox-grid"
      aria-label="Intersectional analysis selectors"></div>
    <div id="chapter-5-four-analysis"></div>
  </section>
  <script>
    const chapterFiveFourData = {serialized};
{_SCRIPT}  </script>
"""
