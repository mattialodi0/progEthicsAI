"""Renderer for chapter 4: Prompt adherence."""

from __future__ import annotations

from typing import Any

from ..explain import chapter_header, findings, guide, metric_defs
from ..insights import chapter_four
from ..measurable import unmeasurable_by_attribute
from .ch4_charts import accuracy_chart, adherence_matrix, demographic_chart, mode_chart
from .ch4_uncertainty import age_years_card, probability_card


def render_chapter_four(adherence: dict[str, Any]) -> str:
    """Render measurements computed by the analysis pipeline."""
    attributes = adherence.get("by_attribute", {})
    modes = adherence.get("by_prompt_mode", {})
    combinations = adherence.get("combinations", {})
    attribute_names = adherence.get("attributes", [])
    unmeasurable = unmeasurable_by_attribute(adherence)
    attribute_html = "".join(
        adherence_matrix(attribute, measurement)
        for attribute, measurement in attributes.items()
    ) or '<p class="note">No adherence observations are available.</p>'
    demographic_html = "".join(
        demographic_chart(category, groups, attribute_names, unmeasurable.get(category))
        for category, groups in adherence.get("by_demographic_category", {}).items()
        if groups
    ) or '<p class="note">No demographic adherence breakdown is available.</p>'
    combination_html = "".join(
        adherence_matrix(name, measurement, collapsible=True)
        for name, measurement in combinations.items()
        if measurement.get("total", 0) > 0
    ) or '<p class="note">No two-attribute prompt combinations are available.</p>'

    return f"""
  {chapter_header(4, "Prompt adherence",
      "When a prompt asks for a specific gender, race, age or emotion, does the image "
      "show it — and equally well for every group?",
      "<p>Adherence connects the input prompt to the measured image. It is a "
      "<strong>quality-of-service</strong> question: a model that reliably draws "
      "“a male doctor” but often turns “a female doctor” into a man serves the two groups "
      "unequally, even if it never produces offensive content. Low adherence for one group "
      "also tells you that the model's default (chapter 3) overrides explicit "
      "instructions.</p>")}
  {findings(chapter_four(adherence))}
  {metric_defs([
      ("Adherence (accuracy)", "Images whose detected value equals the requested value ÷ "
       "images that request the value and have a detection.", "100% = always followed. "
       "Compare across groups of the same attribute: the gap is the disparity."),
      ("Unmatched / without detection", "Images that request the attribute but where the "
       "detector returned nothing (usually no face found).", "Reported separately so that "
       "excluding failed detections cannot hide a group-specific failure."),
      ("Confusion matrix cell", "Share of images requesting the row value that were "
       "detected as the column value.", "The diagonal is adherence; off-diagonal cells show "
       "<em>what the model drew instead</em>."),
      ("Detected age (years)", "Age estimated by the face model, before it is cut into "
       "young (&lt; 30), middle-aged (30–49) and old (50+).", "Shows where each requested "
       "age term lands and whether two terms are told apart, without depending on the "
       "cut-offs. A relative scale: compare terms, do not read the years literally."),
      ("Probability-weighted adherence", "Mean probability that the face model gives to the "
       "requested class.", "Close to the adherence = labels are clear-cut. Much lower = "
       "many hits are narrow wins between two classes."),
      ("Confident-label adherence", "Adherence on the images whose most likely class has a "
       "probability of at least 60%.", "Check how many images remain: a group with few "
       "confident labels is one the face model cannot measure well."),
  ], open_=False)}
  <section id="chapter-4-attributes">
    <h3>4.1 Overall adherence by attribute</h3>
    {guide(
        read=["The bar chart summarises adherence per attribute (blue ≥ 80%, amber 60–80%, "
              "red &lt; 60%).",
              "In each confusion matrix, rows are the requested value and columns the "
              "detected value; each row sums to 100%. Cell text: row percentage and (count). "
              "Outlined cells are the diagonal, i.e. the adherence.",
              "Rows marked (n/a) request a value the detector cannot output: they can "
              "never match, so the adherence is also given without them."],
        look=["Rows with a weak diagonal: the requested value is often not drawn.",
              "Asymmetric errors: if “female” is often detected as “male” but never the "
              "reverse, the model drifts towards a default group.",
              "Systematic substitutions (e.g. “old” → “middle-aged”), which show what the "
              "model considers the typical version of a group."],
        caution=["Errors can come from the detector rather than the model, especially for "
                 "age and race. Detector error rates can themselves differ across groups.",
                 "Emotion adherence is limited by the label sets: DeepFace detects angry, "
                 "disgust, fear, happy, sad, surprise and neutral, so requested emotions such "
                 "as calm, bored or nervous can never match exactly.",
                 "Requested and detected label names must match for a hit; a requested value "
                 "with 0% adherence may simply use a different label than the detector "
                 "(e.g. “hispanic” vs “latino hispanic”)."],
    )}
    <article class="plot-card"><h4>Adherence per attribute</h4>{accuracy_chart(attributes)}</article>
    <div class="plot-grid matrix-grid">{attribute_html}</div>
    {age_years_card(adherence.get("age_numeric") or {})}
    {probability_card(adherence.get("probabilities") or {})}
  </section>
  <section id="chapter-4-modes">
    <h3>4.2 Adherence by prompt mode</h3>
    <p>Does adding more attributes to a prompt make the model follow each of them less?</p>
    {guide(
        read="Grouped bars: one colour per attribute, one group per prompt mode. Missing bars "
             "mean the attribute is never requested in that mode.",
        look="Adherence that falls from single-variable to double-variable to control prompts "
             "means attributes compete: the model drops some requested attributes when "
             "several are combined. Compare which attribute is sacrificed.",
    )}
    <article class="plot-card">{mode_chart(modes, attribute_names)}</article>
  </section>
  <section id="chapter-4-demographics">
    <h3>4.3 Adherence by demographic category</h3>
    <p>For each requested value of a demographic attribute, the bars show how often each
    requested attribute was detected as requested.</p>
    {guide(
        read="In “By requested gender”, the <em>gender</em> bars give the per-group adherence "
             "to gender itself; the other bars show whether, e.g., race or age is followed "
             "less reliably when the prompt also asks for a woman.",
        look=["A lower bar for one group means prompts mentioning that group are followed "
              "less reliably — an unequal quality of service.",
              "Gaps of more than ~10 percentage points with n ≥ 30 per group deserve "
              "attention; smaller groups need larger gaps to be convincing."],
        caution="A lower bar can also mean the detector is less accurate for that group.",
    )}
    <div class="plot-grid">{demographic_html}</div>
  </section>
  <section id="chapter-4-combinations">
    <h3>4.4 Adherence to combinations of two attributes</h3>
    <p>Joint adherence: both requested attributes must be detected correctly. Expand a
    combination to see its confusion matrix.</p>
    {guide(
        read="Rows are requested pairs (e.g. “female + black”), columns are detected pairs.",
        look="Pairs whose joint adherence is much lower than the adherence of their two "
             "attributes separately: the combination itself is hard for the model (e.g. it "
             "can draw women and can draw Black people, but often fails at Black women). "
             "This is an intersectional quality-of-service gap.",
    )}
    {combination_html}
  </section>
"""
