"""Renderer for chapter 9: cross-analysis synthesis and key findings.

All content comes from ``report_data["chapter_9"]`` (built by
``analysis.synthesis``): nothing here is specific to one model or dataset.
Static parts (answers, finding cards, evidence matrix, measurement caveats)
are rendered in Python (``ch9_sections``); the ranked-effects plot, the
effect-versus-reliability plot and the results explorer are drawn in the
browser (``ch9_explorer`` and ``static/chapter_9.js``).
"""

from __future__ import annotations

from typing import Any

from ..explain import chapter_header, esc, guide, metric_defs, pct
from ..static_files import static_text
from .ch9_explorer import explorer_controls, results_payload
from .ch9_sections import (
    render_answers,
    render_finding_cards,
    render_matrix,
    render_measurement,
    render_overview,
)

_STYLE = "\n  <style>\n" + static_text("chapter_9.css") + "  </style>"
_SCRIPT = "\n" + static_text("chapter_9.js")


_TIER_HELP = {
    "supported": "q below α after the audit-wide correction and at least a small effect.",
    "descriptive": "At least a small effect, but not statistically supported after correction "
                   "(often a small group).",
    "negligible": "Effect below the small threshold, whatever its p-value.",
    "context": "Expected or descriptive information (e.g. a caption naming the requested job, "
               "objects of the requested workplace, embedding separability of a requested "
               "attribute). Not counted as a disparity.",
    "measurement": "A property of the measurement pipeline (face-detection gaps, label "
                   "vocabulary mismatches) rather than of the model.",
}


def _definitions(data: dict[str, Any]) -> str:
    """Metric definitions and the effect-size thresholds of every family."""
    settings = data.get("settings") or {}
    alpha = settings.get("alpha", 0.05)
    evidence_levels = settings.get("evidence_levels") or {}
    correction = data.get("correction") or {}
    family_rows = "".join(
        f"<tr><th scope='row'>{esc(value['label'])}</th><td>"
        + " / ".join(f"{threshold:g}" for threshold in value["thresholds"])
        + "</td></tr>"
        for value in (settings.get("families") or {}).values()
    )
    return f"""{metric_defs([
      ("Strength", "The effect size mapped onto a common scale using the conventional "
       "small / medium / large thresholds of its own family (table below): 1 = small, "
       "2 = medium, 3 = large; above 3 it keeps growing linearly (capped at 6).",
       "Makes a gap in percentage points, a mean difference and a Cramér's V comparable. "
       "Bands: &lt; 1 negligible, 1–2 small, 2–3 medium, ≥ 3 large."),
      ("Conservative strength", "The strength at the lower end of the effect's 95% "
       "confidence interval.", "Used for ranking and for the verdicts in 9.1: the smallest "
       "effect still compatible with the data, so a huge gap in 8 images does not outrank a "
       "solid one in 200. A result is <em>reliably</em> large (or moderate) when it stays in "
       "that band at this lower end."),
      ("q-value", f"Benjamini–Hochberg adjusted p-value over all {correction.get('tests', 0):,} "
       "tests of the audit, including the candidate objects and caption words screened "
       "before the top ones were stored.", f"A result is <em>supported</em> when q &lt; {alpha} "
       "and its strength is at least small. The expected share of false discoveries among "
       f"supported results is at most {pct(alpha, 0)}."),
      ("Comparison standard", "Every specific result compares a subgroup with the "
       "<em>other values of the same prompt attribute</em>, using only images whose prompt "
       "does not request the measured feature (e.g. job × detected gender without “female "
       "doctor” prompts).", "Removes the most common confound: a difference that is just the "
       "prompt asking for it."),
      ("Evidence level", "<em>" + esc(evidence_levels.get("confirmed", "")) + "</em>: supported "
       "in a pooled comparison and in matched prompt pairs that change only this attribute. "
       "<em>" + esc(evidence_levels.get("convergent", "")) + "</em>: supported by two or more "
       "measurement instruments. <em>" + esc(evidence_levels.get("supported", "")) + "</em>: "
       "one instrument.", "Matched pairs hold the rest of the prompt fixed, so they rule out "
       "template differences; several instruments rule out a single detector's error."),
      ("Stereotype candidate", "A supported link between a social-group attribute "
       "(gender, race, age, continent, status, traits) and non-requested content, or between "
       "any prompt attribute and a non-requested demographic.", "A pattern that matches the "
       "shape of a stereotype and deserves human review. It is not a verdict."),
  ], open_=False)}
  <details class="defs"><summary>Effect-size families and thresholds (small / medium / large)</summary>
    <table><thead><tr><th>Family</th><th>Thresholds</th></tr></thead><tbody>{family_rows}</tbody></table>
    <p class="note">Proportion gaps use Cohen's h, mean differences Hedges' g (or d_z for paired
    differences), contingency tables Cohen's w = V·√(min(r, c) − 1), group differences on
    continuous features ε², embedding structure the silhouette above its permutation null.</p>
  </details>"""


def _question_sections(data: dict[str, Any], rows: dict[str, dict[str, Any]],
                       clusters: dict[str, dict[str, Any]]) -> str:
    """9.1-9.3: answers to the fairness questions, key findings, evidence matrix."""
    return f"""<section id="chapter-9-answers">
    <h3>9.1 Answers to the fairness questions</h3>
    <p>The five questions of the reading guide, answered from the results table. The verdict
    is the <strong>largest reliable disparity found</strong>: the effect band of the
    top-ranked supported result at the <em>lower</em> end of its 95% confidence interval (a
    supported result counts as at least small). It is not a grade for the whole question, so
    each card also says how many supported results stay large or moderate at that lower end.
    The three listed results are the strongest by conservative strength (ties broken by
    evidence level and sample size).</p>
    {guide(
        read="Each card counts the specific tests that address the question and how many are "
             "supported. Below each result: its effect with 95% CI, the strength (point "
             "estimate, then the lower end of its interval), the smallest group size, the "
             "q-value and a link to the chapter that measured it.",
        look=["A large verdict backed by results with large n and no ⚠ caveats.",
              "The share of tests that are reliably large: one large result among hundreds of "
              "tests is a different situation from dozens. This is the number to use when "
              "comparing models.",
              "Questions where nothing is supported despite many tests: an informative null, "
              "if the samples are large enough to detect a small effect."],
        caution=["The verdict is a maximum: the more tests a question has, the more likely one "
                 "of them is large. Read the counts, the listed results and the evidence matrix "
                 "before generalising to the whole question.",
                 "Representational quality only counts differences between social groups "
                 "(gender, race, age, continent, status, traits). Brightness, framing or caption "
                 "tone that differ between jobs or workplaces describe the requested scene; they "
                 "stay in the results table under the question “Context”."],
    )}
    {render_answers(data, rows)}
  </section>

  <section id="chapter-9-findings">
    <h3>9.2 Key findings</h3>
    <p>The strongest supported findings, grouped by the claim they make and ranked by
    conservative strength, then evidence level and sample size. At most two findings per
    attribute × feature theme and three per lens are listed, so one dominant pattern does not
    hide the others. Expand a finding to see every result behind it.</p>
    {guide(
        read="Badges give the evidence level, the effect band and the number of independent "
             "sources; ⚠ badges are caveats from 9.6 (hover for details).",
        look=["Counterfactual-confirmed and convergent findings first: they survive a change "
              "of instrument or of prompt template.",
              "Stereotype candidates linking a social group to roles, emotions, objects or "
              "words."],
        caution=["Every demographic label is a detector prediction. A finding about "
                 "“detected female” is a finding about what the face model reports.",
                 "Images in matched pairs are partly the same images as in pooled comparisons; "
                 "confirmation shows the pattern is not a template artefact, it is not an "
                 "independent replication."],
    )}
    {render_finding_cards(data, rows, clusters)}
  </section>

  <section id="chapter-9-matrix">
    <h3>9.3 Evidence matrix</h3>
    <p>One row per prompt attribute, one column per lens. A disparity that appears in several
    columns of the same row is a robust property of how the model treats that attribute.</p>
    {render_matrix(data, rows)}
  </section>"""


def _interactive_sections(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    """9.4-9.7: ranked effects, reliability, measurement caveats, explorer."""
    settings = data.get("settings") or {}
    return f"""<section id="chapter-9-ranked">
    <h3>9.4 Ranked effect sizes</h3>
    <p>The filters below drive 9.4, 9.5 and the explorer in 9.7. The plot shows the 30
    strongest results that match, with the 95% CI of their strength.</p>
    {explorer_controls(data)}
    {guide(
        read="Each row is one result; colour = lens; filled marker = supported, open = not "
             "supported; a triangle ▶ means the strength reached the scale's cap, so the "
             "true value may be higher. Dotted lines mark the small, medium and large "
             "thresholds. Hover for the full statement, effect, n and q.",
        look="Long bars reaching down towards 0: a large point estimate with little "
             "certainty. Narrow bars far to the right are the most reliable large effects.",
        caution="Strength puts different metrics on one scale using conventions, not a "
                "physical unit; compare bands, not decimals.",
    )}
    <article class="plot-card"><div id="ch9-forest"></div></article>
  </section>

  <section id="chapter-9-reliability">
    <h3>9.5 Effect versus reliability</h3>
    <p>Strength against the size of the smallest compared group. Points in the shaded corner
    are large effects measured on fewer than {esc(settings.get("small_n", 30))} images:
    candidates for re-testing with more generations rather than conclusions.</p>
    <article class="plot-card"><div id="ch9-reliability"></div></article>
  </section>

  <section id="chapter-9-measurement">
    <h3>9.6 Measurement reliability and caveats</h3>
    <p>Problems in the measurement pipeline can create or hide disparities. These checks are
    applied automatically and attached as ⚠ flags to the affected results.</p>
    {render_measurement(data, rows)}
  </section>

  <section id="chapter-9-explorer">
    <h3>9.7 Results explorer</h3>
    <p>The full results table (filtered as above). Click a column to sort; the default order is
    the ranking used in 9.2.</p>
    <p><button type="button" class="button" id="ch9-download">Download filtered results
    (CSV)</button></p>
    <details class="defs" id="ch9-explorer-details">
      <summary>Show the results table (<span id="ch9-count" aria-live="polite"></span>)</summary>
      <div id="ch9-table" class="table-wrap"></div>
    </details>
  </section>"""


def _method(data: dict[str, Any]) -> str:
    """Collapsible method and limitations of the synthesis."""
    tiers = (data.get("settings") or {}).get("tiers") or {}
    correction = data.get("correction") or {}
    return f"""<details class="defs" id="chapter-9-method"><summary>Method and limitations of the synthesis</summary>
    <dl class="tier-list">
      {''.join(f"<dt>{esc(name)}</dt><dd>{esc(_TIER_HELP.get(key, ''))}</dd>" for key, name in tiers.items())}
    </dl>
    <ul>
      <li>Intervals: Wilson score intervals for proportions and Newcombe's method for their
      differences; t intervals for means; approximate intervals for Hedges' g; a multinomial
      bootstrap for Cramér's V; Wald intervals for log odds ratios of caption words.</li>
      <li>Tests: two-proportion z-tests, Fisher's exact test (objects), chi-square tests
      (contingency tables, adherence by requested value, parity), Mann–Whitney and
      Kruskal–Wallis tests (image features), one-sample t-tests (paired differences), a
      permutation null (silhouette).</li>
      <li>Multiple testing: one Benjamini–Hochberg family over the whole audit
      ({correction.get('tests', 0):,} tests). Screened candidates that were not stored are
      counted with p = 1, which is conservative. Tests are not independent (the same images
      appear in several comparisons); BH remains valid under the positive dependence typical
      here, but treat q-values near α as borderline.</li>
      <li>Default representation is compared with parity over the detector's categories.
      Parity is a reference point, not a claim about the correct distribution.</li>
      <li>Linking results into findings uses exact matches of (attribute, subgroup, feature,
      category, direction); caption words are linked to demographic categories through a small
      lexicon (configurable with <code>synthesis_concept_lexicon</code>). Links can be missed
      when two instruments use different words for the same thing.</li>
      <li>Pairs in chapter 6 are matched by prompt template and generation order, not by
      seed. Each side is therefore treated as an independent sample.</li>
      <li>The synthesis cannot establish causes (training data, architecture) and inherits
      every limitation of the detectors, the captioner and CLIP.</li>
    </ul>
  </details>"""


def render_chapter_nine(data: dict[str, Any]) -> str:
    """Render the cross-analysis chapter from the synthesis payload."""
    if not data or not data.get("results"):
        return (chapter_header(9, "Cross-analysis and key findings",
                               "Which disparities are robust across the audit?",
                               "<p>The synthesis data is not available.</p>")
                + '<p class="placeholder">Rerun <code>python -m src.analysis.analyzer</code> '
                  "to build chapter 9.</p>")
    rows = {row["id"]: row for row in data.get("results") or []}
    clusters = {cluster["id"]: cluster for cluster in data.get("clusters") or []}
    return f"""
  {_STYLE}
  {chapter_header(9, "Cross-analysis and key findings",
      "Which disparities are large, statistically reliable and confirmed by more than one "
      "kind of evidence — and which are not?",
      "<p>Chapters 3–8 each look at the model through one lens. This chapter puts every "
      "effect they measured into <strong>one results table</strong> with a common format: "
      "the subgroup, what it is compared with, the effect in its own unit with a 95% "
      "confidence interval, a standardized <em>strength</em>, the sample size and a p-value. "
      "All p-values of the audit are then corrected <em>together</em> for multiple testing, "
      "and results that make the same claim are linked across independent evidence "
      "sources — the face detector, the object detector, the captions, pixel statistics and "
      "matched counterfactual prompts. A pattern seen by several instruments, or one that "
      "persists when only a single prompt word is changed, is much stronger evidence than a "
      "single chart.</p>"
      "<p>Everything below is computed from the analysis artifact; a different model or "
      "dataset produces different findings. Findings are classified by the evidence behind "
      "them, not labelled “fair” or “unfair”: whether a disparity is harmful is a judgement "
      "the evidence informs.</p>")}
  {render_overview(data, rows, clusters)}
  {_definitions(data)}

  {_question_sections(data, rows, clusters)}

  {_interactive_sections(data, rows)}

  {_method(data)}
  <script type="application/json" id="chapter-9-data">{results_payload(data)}</script>
  <script>{_SCRIPT}</script>
"""
