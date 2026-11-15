"""Renderer for chapter 5.3: stereotypical item associations."""

from __future__ import annotations

from typing import Any

from ..explain import findings, guide
from ..insights import stereotypes
from ..static_files import script_json, static_text

_SCRIPT = static_text("chapter_5_3.js")


def render_stereotypical_associations(bias_stats: dict[str, Any]) -> str:
    """Render the most frequent object associations for prompt contexts."""
    associations = bias_stats.get("stereotypical_associations", {})
    serialized = script_json(associations)
    return f"""
  <section id="chapter-5-stereotypical" aria-labelledby="chapter-5-stereotypical-heading">
    <h3 id="chapter-5-stereotypical-heading">5.3 Stereotypical item associations</h3>
    <p>Which detected objects co-occur unusually often with each prompted job, workplace,
    continent, toxic trait or positive trait? Each object is counted once per image, and
    generic person and face labels are excluded. Over-representation is measured with
    <strong>lift</strong>:</p>
    <p style="text-align:center"><strong>Lift(A, B) = P(B | A) / P(B)</strong></p>
    <p>where A is the prompt group and B the object. A lift of 3 means the object appears
    three times more often in that group than across all prompts of this context.</p>
    {findings(stereotypes(associations))}
    {guide(
        read=["Each bar is one group → object pair, sorted by lift, on a logarithmic axis; "
              "the dashed line at 1 marks “as common as overall”.",
              "Hover for P(B | A) — the share of the group's images containing the object — "
              "and the number of images. The minimum-count filter hides objects seen in "
              "fewer images, whose lift is unstable."],
        look=["Objects the prompt does <em>not</em> require. A guitar for “musician” or a "
              "rifle for “soldier” is expected; weapons for a continent, handcuffs for an "
              "ethnicity or luxury goods only for some groups would be stereotyped "
              "depictions.",
              "Toxic and positive traits: which props the model uses to signal “criminal”, "
              "“terrorist” or “trustworthy”, and whether these props also appear for "
              "certain demographics in chapter 8's captions.",
              "High lift <em>and</em> high P(B | A): a prop that defines most of the group's "
              "images is a strong visual template."],
        caution=["Lift is a ratio: an object seen in 1 image of a 10-image group and nowhere "
                 "else gets a huge lift. Keep the minimum count at 3 or more for conclusions.",
                 "A high lift alone does not make an association harmful; that judgement "
                 "needs the context of what the object signals.",
                 "Objects come from an automatic detector with a fixed vocabulary; missing "
                 "or wrong detections are possible."],
    )}
    <div class="controls" aria-label="Stereotypical association selector">
      <label>Prompt context <select id="chapter-five-three-attribute"></select></label>
      <label>Minimum images with the object <select id="chapter-five-three-min">
        <option value="1">1</option><option value="2">2</option>
        <option value="3" selected>3</option><option value="5">5</option></select></label>
    </div>
    <article class="plot-card"><div id="chapter-five-three-plot"></div></article>
    <details><summary>Full table: most common objects per group</summary>
      <div id="chapter-five-three-table" class="table-wrap"></div>
    </details>
  </section>
  <script>
    const chapterFiveThreeData = {serialized};
{_SCRIPT}  </script>
"""
