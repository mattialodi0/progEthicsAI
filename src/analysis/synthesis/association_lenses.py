"""Chapter 9 lenses read from chapter 5: demographic, intersectional and object
associations."""

from __future__ import annotations

import math
from typing import Any

from ..stats_utils import cohen_h, finite, two_proportion_p
from .constants import DEMOGRAPHIC, SOCIAL_ATTRIBUTES
from .table import ResultTable, h_interval, human, p_from_ci, pct, proportion_gap


def association_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    chapter = report_data.get("chapter_5") or {}
    source = chapter.get("associations_unrequested")
    fallback = not isinstance(source, dict)
    if fallback:
        table.warnings.append("Chapter 5 has no unrequested-only associations; using all "
                              "images (rebuild chapter 5).")
        source = chapter.get("associations") or {}
    min_cell = table.settings["min_cell"]
    for attribute, outputs in source.items():
        for feature, matrix in (outputs or {}).items():
            if attribute == feature or not isinstance(matrix, dict):
                continue
            counts = matrix.get("counts") or {}
            columns = matrix.get("columns") or []
            total = matrix.get("total") or 0
            stats = matrix.get("statistics") or {}
            if total < 2 or len(counts) < 2 or len(columns) < 2:
                continue
            scope = (f"images whose prompt does not request {feature}" if not fallback
                     else "all images")
            v = finite(stats.get("cramers_v")) or 0.0
            factor = math.sqrt(max(1, min(len(counts), len(columns)) - 1))
            interval = stats.get("cramers_v_ci") or [None, None]
            p_value = finite(stats.get("p_value_exact"))
            if p_value is None:
                p_value = finite(stats.get("p_value"))
            table.add(
                lens="association", level="omnibus", attribute=attribute, feature=feature,
                scope=scope, metric="Cramér's V", effect=v,
                ci=[finite(interval[0]), finite(interval[1])], family="w",
                magnitude=v * factor,
                magnitude_ci=[(finite(interval[0]) or 0) * factor if interval[0] is not None
                              else None,
                              (finite(interval[1]) or 0) * factor if interval[1] is not None
                              else None],
                n=min((sum(row.values()) for row in counts.values()), default=0),
                n_total=total, p_value=p_value, question="stereotyping",
                stereotype_candidate=attribute in SOCIAL_ATTRIBUTES or feature in DEMOGRAPHIC,
                details={"cramers_v_corrected": stats.get("cramers_v_corrected")},
                statement=(f"Prompt {human(attribute)} is associated with detected {feature} "
                           f"(Cramér's V = {v:.2f}; {scope})."),
            )
            column_totals = {column: sum(row.get(column, 0) for row in counts.values())
                             for column in columns}
            for group, row_counts in counts.items():
                n = sum(row_counts.values())
                rest_n = total - n
                if n < min_cell or rest_n < min_cell:
                    continue
                for category in columns:
                    k = row_counts.get(category, 0)
                    rest_k = column_totals[category] - k
                    share, rest_share = k / n, rest_k / rest_n
                    if share <= rest_share:
                        continue  # under-representation is the mirror of another cell
                    gap = proportion_gap(k, n, rest_k, rest_n)
                    table.add(
                        lens="association", attribute=attribute, subgroup=group,
                        reference=f"other {human(attribute)} values", feature=feature,
                        category=category, scope=scope, metric=f"share detected as {category}",
                        **gap,
                        n=min(n, rest_n), n_total=total,
                        p_value=two_proportion_p(k, n, rest_k, rest_n),
                        question="stereotyping",
                        stereotype_candidate=(attribute in SOCIAL_ATTRIBUTES
                                              or feature in DEMOGRAPHIC),
                        keys=[(attribute, group, feature, category, "+")],
                        details={"share": share, "rest_share": rest_share, "count": k},
                        statement=(f"Prompt {human(attribute)} = {group}: detected {feature} "
                                   f"is {category} in {pct(share)} of {n} faces vs "
                                   f"{pct(rest_share)} for other {human(attribute)} values."),
                    )


def intersectional_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    data = (report_data.get("chapter_5") or {}).get("intersectional") or {}
    min_cell = table.settings["min_cell"]
    for pair, item in data.items():
        first = item.get("first_attribute")
        second = item.get("second_attribute")
        for feature, output in (item.get("outputs") or {}).items():
            if feature in (first, second):
                # The chapter's expectation averages the two single-attribute
                # groups; when one of them *requests* the detected feature that
                # average is not a meaningful baseline (female × asian → race
                # is ~100% asian by request). Joint-group adherence is covered
                # by the adherence-by-category results instead.
                continue
            question = "stereotyping"
            joint = {row.get("group"): row for row in output.get("rows", [])}
            for forest in output.get("forest", []):
                difference = finite(forest.get("difference"))
                low, high = finite(forest.get("lower")), finite(forest.get("upper"))
                n = forest.get("n") or 0
                if difference is None or low is None or high is None or n < min_cell:
                    continue
                if difference <= 0:
                    continue  # mirrored by the over-represented category
                group = forest.get("group")
                category = forest.get("category")
                observed = ((joint.get(group) or {}).get("values", {}).get(category, {})
                            .get("percentage"))
                observed = (finite(observed) or 0) / 100
                expected = observed - difference / 100
                table.add(
                    lens="intersectional", attribute=pair, subgroup=group,
                    reference=f"average of the {first} and {second} groups alone",
                    feature=feature, category=category,
                    metric="joint − expected share", effect=difference / 100,
                    effect_unit="pp", ci=[low / 100, high / 100], family="h",
                    magnitude=cohen_h(observed, expected),
                    magnitude_ci=h_interval(expected, low / 100, high / 100),
                    n=n, n_total=output.get("total"),
                    p_value=p_from_ci(difference, low, high), question=question,
                    stereotype_candidate=question == "stereotyping",
                    keys=[(pair, group, feature, category, "+")],
                    details={"observed": observed, "expected": expected,
                             "attributes": [first, second]},
                    statement=(f"{group}: detected {feature} = {category} in "
                               f"{pct(observed)} of {n} faces, {abs(difference):.0f} pp more "
                               f"than expected from {first} and {second} separately."),
                )
            for model in output.get("models", []) or []:
                p_value = finite(model.get("interaction_p_value"))
                terms = model.get("interaction_terms") or []
                largest = max((abs(finite(term.get("effect_percentage_points")) or 0)
                               for term in terms), default=None)
                if p_value is None:
                    continue
                table.add(
                    lens="intersectional", level="omnibus", attribute=pair, feature=feature,
                    category=model.get("output_category"), metric="interaction test",
                    effect=(largest / 100) if largest is not None else None,
                    effect_unit="pp", n=model.get("n"), n_total=model.get("n"),
                    p_value=p_value, question=question,
                    details={"attributes": [first, second], "df": model.get("interaction_df")},
                    statement=(f"The {first} × {second} interaction changes the share of "
                               f"detected {feature} = {model.get('output_category')} "
                               f"(largest interaction term {largest or 0:.0f} pp)."),
                )


def object_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    data = (report_data.get("chapter_5") or {}).get("stereotypical_associations") or {}
    for attribute, item in data.items():
        for row in item.get("rows", []):
            group = row.get("group")
            n = row.get("observations") or 0
            listed = row.get("associated_objects")
            if listed is None:
                table.warnings.append("Chapter 5.3 has no tested object associations; "
                                      "rebuild chapter 5.")
                return
            tested = row.get("objects_tested", len(listed))
            table.hidden_tests += max(0, tested - len(listed))
            for entry in listed:
                k = entry.get("count", 0)
                rest_k = entry.get("rest_count")
                rest_n = entry.get("rest_observations")
                if rest_k is None or not rest_n or not n:
                    continue
                share, rest_share = k / n, rest_k / rest_n
                gap = proportion_gap(k, n, rest_k, rest_n)
                social = attribute in SOCIAL_ATTRIBUTES
                lift = finite(entry.get("lift"))
                table.add(
                    lens="objects", attribute=attribute, subgroup=group,
                    reference=f"other {human(attribute)} values", feature="object",
                    category=entry.get("object"), metric="share of images with object",
                    **gap,
                    n=min(n, rest_n), n_total=n + rest_n,
                    p_value=finite(entry.get("p_value")),
                    question="stereotyping" if social else "context",
                    stereotype_candidate=social,
                    # Job and workplace objects are mostly what the prompt asks
                    # for (a chef with a cutting board): context, not bias.
                    context=not social,
                    keys=[(attribute, group, "object", entry.get("object"), "+")],
                    details={"share": share, "rest_share": rest_share, "lift": lift,
                             "count": k},
                    statement=(f"Prompt {human(attribute)} = {group}: “{entry.get('object')}” "
                               f"appears in {pct(share)} of {n} images vs "
                               f"{pct(rest_share, 1)} for other {human(attribute)} values"
                               + (f" (lift {lift:.1f})." if lift else ".")),
                )
