"""Chapter 5 data: conditional, intersectional and object associations."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .associations import association_matrix
from .common import OUTPUT_FEATURES, PROMPT_ATTRIBUTES
from .double import double_association
from .intersectional import intersectional_analysis
from .objects import stereotypical_associations


def collect_bias_stats(
    data: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Precompute prompt-attribute/output-feature association matrices.

    Each matrix is conditioned on a prompt category: a row sums to 100% over
    the detected output categories when at least one paired observation exists.
    Images missing either side are excluded from that matrix and are represented
    by its sample counts and total.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of image metadata objects")
    records = [record for record in data if isinstance(record, dict)]
    associations = {
        prompt_attribute: {
            output_feature: association_matrix(
                records, prompt_attribute, output_feature
            )
            for output_feature in OUTPUT_FEATURES
        }
        for prompt_attribute in PROMPT_ATTRIBUTES
    }
    available = [
        prompt_attribute
        for prompt_attribute, features in associations.items()
        if any(matrix["total"] for matrix in features.values())
    ]
    available_features = [
        output_feature
        for output_feature in OUTPUT_FEATURES
        if any(
            associations[prompt_attribute][output_feature]["total"]
            for prompt_attribute in PROMPT_ATTRIBUTES
        )
    ]
    double_associations = {}
    for first_attribute, second_attribute in combinations(PROMPT_ATTRIBUTES, 2):
        association = double_association(
            records, first_attribute, second_attribute
        )
        if association["outputs"]:
            key = f"{first_attribute}__{second_attribute}"
            double_associations[key] = association
    # Same matrices restricted to images that do not request the detected
    # feature: the common comparison standard used by the synthesis chapter.
    associations_unrequested = {
        prompt_attribute: {
            output_feature: association_matrix(
                records, prompt_attribute, output_feature, unrequested_only=True
            )
            for output_feature in OUTPUT_FEATURES
            if output_feature != prompt_attribute
        }
        for prompt_attribute in PROMPT_ATTRIBUTES
    }
    return {
        "prompt_attributes": available,
        "output_features": available_features,
        "associations": associations,
        "associations_unrequested": associations_unrequested,
        "double_associations": double_associations,
        "intersectional": intersectional_analysis(records),
        "stereotypical_associations": stereotypical_associations(records),
    }
