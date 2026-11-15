"""Chapter 9 lenses read from chapters 3, 6 and 7: matched counterfactuals,
image features, embedding separability and output diversity."""

from __future__ import annotations

from typing import Any

from ..stats_utils import cohen_h, finite, normal_p
from .constants import DEMOGRAPHIC, DEMOGRAPHIC_FEATURES, FACE_FEATURES, SOCIAL_ATTRIBUTES
from .table import ResultTable, h_interval, human, pct, scene_question


def counterfactual_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    chapter = report_data.get("chapter_6") or {}
    min_cell = table.settings["min_cell"]
    for attribute, item in (chapter.get("by_attribute") or {}).items():
        # Changing a social-group word is the counterfactual-consistency
        # question; changing the job or workplace and seeing the detected
        # demographics move is occupational / contextual stereotyping.
        social_change = attribute in SOCIAL_ATTRIBUTES
        effects = item.get("side_effects")
        if effects is None:
            table.warnings.append("Chapter 6 has no side-effect tests; rebuild chapter 6.")
            effects = []
        for effect in effects:
            difference = finite(effect.get("difference"))
            n_first, n_second = effect.get("n_first", 0), effect.get("n_second", 0)
            if difference is None or difference <= 0 or min(n_first, n_second) < min_cell:
                continue
            first, second = effect.get("first"), effect.get("second")
            feature, category = effect.get("feature"), effect.get("category")
            first_share = finite(effect.get("first_share")) or 0.0
            second_share = finite(effect.get("second_share")) or 0.0
            low, high = finite(effect.get("ci_low")), finite(effect.get("ci_high"))
            social = attribute in SOCIAL_ATTRIBUTES or feature in DEMOGRAPHIC
            table.add(
                lens="counterfactual", attribute=attribute, subgroup=second,
                reference=f"matched {first} prompts", feature=feature, category=category,
                scope="prompt pairs identical except for this attribute",
                metric=f"share detected as {category}", effect=difference, effect_unit="pp",
                ci=[low, high], family="h", magnitude=cohen_h(second_share, first_share),
                magnitude_ci=h_interval(first_share, low, high),
                n=min(n_first, n_second), n_total=n_first + n_second,
                p_value=finite(effect.get("p_value")),
                question=("counterfactual" if social_change else
                          "stereotyping" if feature in FACE_FEATURES else "context"),
                stereotype_candidate=social,
                keys=[(attribute, second, feature, category, "+")],
                details={"first": first, "second": second, "first_share": first_share,
                         "second_share": second_share},
                statement=(f"Same prompt, only {human(attribute)} changed {first} → {second}: "
                           f"detected {feature} = {category} goes from {pct(first_share)} to "
                           f"{pct(second_share)} ({n_first} vs {n_second} images)."),
            )
        conditions = sorted((item.get("adherence") or {}).keys())
        if len(conditions) == 2:
            # Paired numerical differences are (second − first) of the
            # alphabetically ordered values; only meaningful for 2 values.
            first, second = conditions
            for feature, stats in (item.get("numerical_differences") or {}).items():
                mean = finite(stats.get("mean"))
                dz = finite(stats.get("dz"))
                if mean is None or dz is None or (stats.get("n") or 0) < min_cell:
                    continue
                higher = second if mean > 0 else first
                low, high = finite(stats.get("ci_low")), finite(stats.get("ci_high"))
                n = stats.get("n")
                sd = finite(stats.get("sd")) or 0.0
                demographic = feature in DEMOGRAPHIC_FEATURES
                table.add(
                    lens="counterfactual", attribute=attribute, subgroup=second,
                    reference=f"matched {first} prompts", feature=f"visual:{feature}",
                    scope="prompt pairs identical except for this attribute",
                    metric=f"mean paired change in {human(feature)}", effect=mean,
                    effect_unit="value", ci=[low, high], family="dz", magnitude=dz,
                    magnitude_ci=[low / sd if sd and low is not None else None,
                                  high / sd if sd and high is not None else None],
                    n=n, n_total=n, p_value=finite(stats.get("p_value")),
                    question=("counterfactual" if social_change else
                              "stereotyping" if demographic else "context"),
                    stereotype_candidate=demographic,
                    keys=[(attribute, higher, f"visual:{feature}", "", "+")],
                    statement=(f"Same prompt, only {human(attribute)} changed {first} → "
                               f"{second}: {human(feature)} changes by {mean:+.3f} on average "
                               f"(d_z = {dz:+.2f}, {n} pairs)."),
                )
        embeddings = item.get("embeddings") or {}
        between = finite((embeddings.get("between_pair_euclidean_distance") or {}).get("mean"))
        within = finite((embeddings.get("within_condition_euclidean_distance") or {}).get("mean"))
        if between is not None and within:
            table.add(
                lens="counterfactual", level="omnibus", attribute=attribute,
                feature="embedding", metric="between ÷ within distance",
                effect=between / within, effect_unit="ratio", n=item.get("pairs"),
                n_total=item.get("pairs"), context=True, question="counterfactual",
                statement=(f"Changing only {human(attribute)} moves the CLIP embedding "
                           f"{between / within:.2f}× as far as two generations with the same "
                           f"{human(attribute)} value differ."),
            )


def visual_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    features = (report_data.get("chapter_7") or {}).get("image_features") or {}
    min_cell = table.settings["min_cell"]
    for feature, item in features.items():
        tests = item.get("tests")
        if tests is None:
            table.warnings.append("Chapter 7 has no image-feature tests; rebuild chapter 7.")
            return
        # Detected age is a face-model estimate of a demographic, not a pixel
        # statistic: a group drawn older or younger is a stereotyping result.
        demographic = feature in DEMOGRAPHIC_FEATURES
        for attribute, result in tests.items():
            if attribute == "prompt_mode":
                continue
            extra = ({"source": "Face detector", "question": "stereotyping",
                      "stereotype_candidate": True} if demographic
                     else {"question": scene_question(attribute)})
            epsilon = finite(result.get("epsilon_squared"))
            groups = result.get("groups") or {}
            if epsilon is not None:
                table.add(
                    lens="visual", level="omnibus", attribute=attribute,
                    feature=f"visual:{feature}", metric="ε² (Kruskal–Wallis)", effect=epsilon,
                    family="epsilon2", magnitude=epsilon,
                    n=min((group.get("n") or 0 for group in groups.values()), default=0),
                    n_total=result.get("n"), p_value=finite(result.get("p_value")),
                    **extra,
                    statement=(f"{human(feature).capitalize()} differs between prompt "
                               f"{human(attribute)} values (ε² = {epsilon:.3f})."),
                )
            for group, stats in groups.items():
                g = finite(stats.get("hedges_g"))
                n = stats.get("n") or 0
                rest_n = stats.get("rest_n") or 0
                if g is None or g <= 0 or min(n, rest_n) < min_cell:
                    continue  # lower values are the mirror of a higher group
                mean, rest = finite(stats.get("mean")), finite(stats.get("rest_mean"))
                table.add(
                    lens="visual", attribute=attribute, subgroup=group,
                    reference=f"other {human(attribute)} values", feature=f"visual:{feature}",
                    metric=f"difference in {human(feature)}",
                    effect=(mean - rest) if mean is not None and rest is not None else None,
                    effect_unit="value", family="g", magnitude=g,
                    magnitude_ci=[finite(stats.get("g_ci_low")), finite(stats.get("g_ci_high"))],
                    ci=[None, None], n=min(n, rest_n), n_total=n + rest_n,
                    p_value=finite(stats.get("p_value")), **extra,
                    keys=[(attribute, group, f"visual:{feature}", "", "+")],
                    details={"mean": mean, "rest_mean": rest},
                    statement=(f"Images for prompt {human(attribute)} = {group} have higher "
                               f"{human(feature)} than other {human(attribute)} values "
                               f"(mean {mean:.3g} vs {rest:.3g}, g = {g:+.2f})."),
                )


def embedding_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    groups = ((report_data.get("chapter_7") or {}).get("embedding") or {}).get("groups") or {}
    for attribute, item in groups.items():
        if attribute == "prompt_mode":
            continue
        separability = item.get("separability") or {}
        score = finite(separability.get("score"))
        if score is None:
            continue
        null = separability.get("null") or {}
        probe = separability.get("probe") or {}
        null_mean, null_sd = finite(null.get("null_mean")), finite(null.get("null_sd"))
        p_value = (normal_p((score - null_mean) / null_sd)
                   if null_mean is not None and null_sd else finite(null.get("p_value")))
        accuracy = finite(probe.get("balanced_accuracy"))
        probe_text = (f"; a linear probe recovers it with {pct(accuracy)} balanced accuracy "
                      f"(chance {pct(finite(probe.get('chance')))})" if accuracy is not None
                      else "")
        table.add(
            lens="embedding", level="omnibus", attribute=attribute, feature="embedding",
            metric="silhouette vs permutation null", effect=score, family="silhouette",
            magnitude=score - (null_mean or 0.0), n=probe.get("n_images"),
            n_total=separability.get("n_images"), p_value=p_value, context=True,
            question="quality",
            details={"null_mean": null_mean, "null_sd": null_sd, "probe": probe},
            statement=(f"CLIP embeddings group images by prompt {human(attribute)}: "
                       f"silhouette {score:.2f} vs {null_mean if null_mean is not None else 0:.2f} "
                       f"under shuffled labels{probe_text}."),
        )


def diversity_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    groups = (((report_data.get("chapter_3") or {}).get("diversity") or {})
              .get("groups") or {}).get("demographic_group") or {}
    values = {group: (finite(stats.get("embedding_dispersion")), stats.get("n_images") or 0)
              for group, stats in groups.items()}
    values = {group: value for group, value in values.items() if value[0] is not None}
    if len(values) < 2:
        return
    low = min(values, key=lambda group: values[group][0])
    high = max(values, key=lambda group: values[group][0])
    table.add(
        lens="diversity", level="omnibus", attribute="detected gender", subgroup=low,
        reference=high, feature="embedding dispersion",
        metric="ratio of embedding dispersion", effect=values[low][0] / values[high][0],
        effect_unit="ratio", n=min(values[low][1], values[high][1]),
        n_total=sum(value[1] for value in values.values()), context=True, question="quality",
        details={group: {"dispersion": value[0], "n": value[1]}
                 for group, value in values.items()},
        statement=(f"Images detected as {low} are less varied than images detected as {high} "
                   f"(mean pairwise embedding distance {values[low][0]:.3f} vs "
                   f"{values[high][0]:.3f}). Group sizes differ and the groups come from "
                   "different prompts, so this is descriptive only."),
    )
