"""Chapter 5.2: detected distributions for prompts that set two attributes."""

from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO
from typing import Any

from ..records import detected_value
from .common import OUTPUT_FEATURES


def _heatmap_data_uri(
    values: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]],
    rows: list[str],
    columns: list[str],
    title: str,
) -> str:
    """Render one double-condition matrix with Seaborn outside the report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "Chapter 5.2 requires matplotlib, pandas, and seaborn"
        ) from error

    frame = pd.DataFrame(
        [[values[row][column] for column in columns] for row in rows],
        index=rows,
        columns=columns,
    )
    annotations = pd.DataFrame(
        [
            [
                f"{values[row][column]:.1f}%\n(n={counts[row][column]})"
                for column in columns
            ]
            for row in rows
        ],
        index=rows,
        columns=columns,
    )
    width = max(6.0, 1.15 * len(columns) + 2.5)
    height = max(4.0, 0.65 * len(rows) + 2.0)
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        frame.T,
        annot=annotations.T,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Detected percentage (%)"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_ylabel("Prompt attribute pair")
    axis.set_xlabel("Detected feature category")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=130)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def double_association(
    data: list[dict[str, Any]],
    first_attribute: str,
    second_attribute: str,
) -> dict[str, Any]:
    counts_by_output: dict[str, Counter[str]] = {
        output: Counter() for output in OUTPUT_FEATURES
    }

    for record in data:
        prompt_features = record.get("prompt_feats", {})
        if not isinstance(prompt_features, dict):
            continue
        first_value = prompt_features.get(first_attribute)
        second_value = prompt_features.get(second_attribute)
        if first_value is None or second_value is None:
            continue
        first_label = str(first_value).strip().lower()
        second_label = str(second_value).strip().lower()
        if not first_label or not second_label:
            continue
        detected = {
            output: detected_value(record, output)
            for output in OUTPUT_FEATURES
        }
        for output, value in detected.items():
            if value is None:
                continue
            pair_label = f"{first_label} × {second_label}"
            counts_by_output[output][pair_label, str(value).strip().lower()] += 1

    outputs: dict[str, Any] = {}
    for output in OUTPUT_FEATURES:
        pair_counts = counts_by_output[output]
        columns = sorted({pair for pair, _ in pair_counts})
        rows = sorted({detected for _, detected in pair_counts})
        counts = {
            row: {
                column: pair_counts.get((column, row), 0)
                for column in columns
            }
            for row in rows
        }
        total = sum(sum(row.values()) for row in counts.values())
        if not total:
            continue
        pair_percentages = {
            row: {
                column: round(
                    100.0 * counts[row][column]
                    / max(1, sum(counts[detected][column] for detected in rows)),
                    4,
                )
                for column in columns
            }
            for row in rows
        }
        title = (
            f"Prompt {first_attribute} x {second_attribute} | "
            f"detected {output}"
        )
        outputs[output] = {
            "output_feature": output,
            "rows": rows,
            "columns": columns,
            "counts": counts,
            "percentages": pair_percentages,
            "image": _heatmap_data_uri(
                pair_percentages, counts, rows, columns, title
            ),
            "total": total,
        }

    return {
        "first_attribute": first_attribute,
        "second_attribute": second_attribute,
        "outputs": outputs,
    }
