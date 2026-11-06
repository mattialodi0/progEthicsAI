"""Small, dependency-light statistics helpers shared by the chapter builders.

Everything here works on plain Python numbers. ``numpy`` and ``scipy`` are
imported lazily (both are installed with scikit-learn) and only where an exact
distribution is needed.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

Z95 = 1.959963984540054


def finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def regularized_gamma_q(shape: float, value: float) -> float:
    """Return the regularized upper incomplete gamma function Q(a, x)."""
    if value <= 0:
        return 1.0
    if value < shape + 1.0:
        term = 1.0 / shape
        series = term
        for index in range(1, 1000):
            term *= value / (shape + index)
            series += term
            if abs(term) <= abs(series) * 3e-14:
                break
        return max(
            0.0,
            min(1.0, 1.0 - series * math.exp(-value + shape * math.log(value)
                                                - math.lgamma(shape))),
        )

    tiny = 1e-300
    numerator = 1.0
    denominator = value + 1.0 - shape
    if abs(denominator) < tiny:
        denominator = tiny
    fraction = numerator / denominator
    for index in range(1, 1000):
        an = -index * (index - shape)
        if abs(denominator) < tiny:
            denominator = tiny
        if abs(numerator) < tiny:
            numerator = tiny
        denominator = an / denominator + value + 1.0 - shape
        if abs(denominator) < tiny:
            denominator = tiny
        numerator = (value + index - shape) + an / numerator
        if abs(numerator) < tiny:
            numerator = tiny
        denominator = 1.0 / denominator
        delta = numerator * denominator
        fraction *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return max(
        0.0,
        min(1.0, math.exp(-value + shape * math.log(value)
                          - math.lgamma(shape)) * fraction),
    )


def chi2_sf(statistic: float, degrees_of_freedom: int) -> float:
    """Upper-tail probability of the chi-square distribution."""
    if degrees_of_freedom <= 0:
        return 1.0
    return regularized_gamma_q(degrees_of_freedom / 2, max(0.0, statistic) / 2)


def normal_p(z: float | None) -> float | None:
    """Two-sided p-value of a standard-normal statistic."""
    if z is None or not math.isfinite(z):
        return None
    return math.erfc(abs(z) / math.sqrt(2))


def wilson(count: float, total: float, z: float = Z95) -> tuple[float | None, float | None]:
    """Wilson score interval of a binomial proportion."""
    if not total:
        return None, None
    share = count / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_difference(
    count_a: float, total_a: float, count_b: float, total_b: float
) -> tuple[float | None, float | None, float | None]:
    """Difference p_a − p_b with Newcombe's hybrid-score 95% interval."""
    if not total_a or not total_b:
        return None, None, None
    p_a, p_b = count_a / total_a, count_b / total_b
    low_a, high_a = wilson(count_a, total_a)
    low_b, high_b = wilson(count_b, total_b)
    difference = p_a - p_b
    low = difference - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    high = difference + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return difference, max(-1.0, low), min(1.0, high)


def two_proportion_p(count_a: float, total_a: float, count_b: float, total_b: float) -> float | None:
    """Two-sided pooled z-test for p_a = p_b."""
    if not total_a or not total_b:
        return None
    pooled = (count_a + count_b) / (total_a + total_b)
    variance = pooled * (1 - pooled) * (1 / total_a + 1 / total_b)
    if variance <= 0:
        return 1.0
    return normal_p((count_a / total_a - count_b / total_b) / math.sqrt(variance))


def one_proportion_p(count: float, total: float, reference: float) -> float | None:
    """Two-sided score test of a proportion against a fixed reference value."""
    if not total or not 0 < reference < 1:
        return None
    return normal_p((count / total - reference)
                    / math.sqrt(reference * (1 - reference) / total))


def cohen_h(p_a: float | None, p_b: float | None) -> float | None:
    """Cohen's h effect size between two proportions (signed, a − b)."""
    if p_a is None or p_b is None:
        return None
    p_a = min(1.0, max(0.0, p_a))
    p_b = min(1.0, max(0.0, p_b))
    return 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b))


def t_critical(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return Z95
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, degrees_of_freedom))
    except ImportError:  # pragma: no cover - scipy ships with scikit-learn
        return Z95


def mean_summary(values: Iterable[float]) -> dict[str, Any]:
    """Mean, standard deviation and t-based 95% interval of the mean."""
    clean = [float(value) for value in values if finite(value) is not None]
    n = len(clean)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci_low": None, "ci_high": None}
    mean = sum(clean) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in clean) / (n - 1)) if n > 1 else 0.0
    half = t_critical(n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def hedges_g(first: Sequence[float], second: Sequence[float]) -> dict[str, Any]:
    """Hedges' g (first − second) with an approximate 95% interval."""
    a = mean_summary(first)
    b = mean_summary(second)
    n_a, n_b = a["n"], b["n"]
    if n_a < 2 or n_b < 2:
        return {"g": None, "ci_low": None, "ci_high": None}
    pooled = math.sqrt(((n_a - 1) * a["sd"] ** 2 + (n_b - 1) * b["sd"] ** 2) / (n_a + n_b - 2))
    if not pooled:
        return {"g": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    g = correction * (a["mean"] - b["mean"]) / pooled
    se = math.sqrt((n_a + n_b) / (n_a * n_b) + g * g / (2 * (n_a + n_b)))
    return {"g": g, "ci_low": g - Z95 * se, "ci_high": g + Z95 * se}


def mann_whitney_p(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 2 or len(second) < 2:
        return None
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(first, second, alternative="two-sided").pvalue)
    except (ImportError, ValueError):
        return None


def kruskal(groups: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Kruskal–Wallis H test with the epsilon-squared effect size."""
    usable = [list(group) for group in groups if len(group) >= 2]
    n = sum(len(group) for group in usable)
    if len(usable) < 2 or n < 3:
        return {"h": None, "p_value": None, "epsilon_squared": None, "n": n}
    try:
        from scipy.stats import kruskal as scipy_kruskal
        statistic, p_value = scipy_kruskal(*usable)
    except (ImportError, ValueError):
        return {"h": None, "p_value": None, "epsilon_squared": None, "n": n}
    statistic = float(statistic) if math.isfinite(statistic) else 0.0
    p_value = float(p_value) if math.isfinite(p_value) else 1.0
    return {"h": statistic, "p_value": p_value,
            "epsilon_squared": max(0.0, statistic / (n - 1)), "n": n}


def one_sample_t_p(values: Sequence[float]) -> float | None:
    """Two-sided one-sample t-test p-value against zero."""
    summary = mean_summary(values)
    if summary["n"] < 2:
        return None
    if not summary["sd"]:
        return 1.0 if summary["mean"] == 0 else 0.0
    statistic = summary["mean"] / (summary["sd"] / math.sqrt(summary["n"]))
    try:
        from scipy.stats import t
        return float(2 * t.sf(abs(statistic), summary["n"] - 1))
    except ImportError:  # pragma: no cover
        return normal_p(statistic)


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Benjamini–Hochberg adjusted p-values (q-values); ``None`` passes through."""
    indexed = [(index, value) for index, value in enumerate(p_values)
               if value is not None and math.isfinite(value)]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    running = 1.0
    for rank in range(total, 0, -1):
        index, value = indexed[rank - 1]
        running = min(running, value * total / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def contingency_statistics(
    table: Sequence[Sequence[float]],
    *,
    bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Chi-square test, Cramér's V (raw and bias-corrected) and residuals.

    ``table`` is rows × columns of counts. Adjusted (standardized Pearson)
    residuals are approximately standard normal under independence, so
    |r| > 1.96 marks a cell that deviates from independence at the 5% level.
    With ``bootstrap`` > 0, a percentile interval for V is estimated by
    resampling the table from its own multinomial distribution.
    """
    rows = [list(map(float, row)) for row in table]
    n_rows = len(rows)
    n_columns = len(rows[0]) if rows else 0
    total = sum(sum(row) for row in rows)
    row_totals = [sum(row) for row in rows]
    column_totals = [sum(row[column] for row in rows) for column in range(n_columns)]
    usable_rows = sum(1 for value in row_totals if value)
    usable_columns = sum(1 for value in column_totals if value)
    degrees_of_freedom = max(0, (usable_rows - 1) * (usable_columns - 1))
    empty = {
        "chi_square": 0.0, "degrees_of_freedom": degrees_of_freedom, "p_value": 1.0,
        "cramers_v": 0.0, "cramers_v_corrected": 0.0, "cramers_v_ci": [None, None],
        "adjusted_residuals": [[0.0] * n_columns for _ in range(n_rows)],
    }
    if not total or degrees_of_freedom == 0:
        return empty

    def _v(values: list[list[float]]) -> tuple[float, float, float]:
        local_total = sum(sum(row) for row in values)
        r_totals = [sum(row) for row in values]
        c_totals = [sum(row[column] for row in values) for column in range(n_columns)]
        statistic = 0.0
        for i, row in enumerate(values):
            for j, observed in enumerate(row):
                expected = r_totals[i] * c_totals[j] / local_total if local_total else 0
                if expected:
                    statistic += (observed - expected) ** 2 / expected
        k = sum(1 for value in r_totals if value)
        r = sum(1 for value in c_totals if value)
        smaller = min(k - 1, r - 1)
        v = math.sqrt(statistic / (local_total * smaller)) if smaller > 0 and local_total else 0.0
        # Bergsma (2013) bias correction.
        if local_total > 1 and smaller > 0:
            phi2 = statistic / local_total
            phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (local_total - 1))
            k_corrected = k - (k - 1) ** 2 / (local_total - 1)
            r_corrected = r - (r - 1) ** 2 / (local_total - 1)
            denominator = min(k_corrected - 1, r_corrected - 1)
            corrected = math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0
        else:
            corrected = 0.0
        return statistic, v, corrected

    statistic, v, corrected = _v(rows)
    residuals = []
    for i, row in enumerate(rows):
        values = []
        for j, observed in enumerate(row):
            expected = row_totals[i] * column_totals[j] / total
            scale = expected * (1 - row_totals[i] / total) * (1 - column_totals[j] / total)
            values.append((observed - expected) / math.sqrt(scale) if scale > 0 else 0.0)
        residuals.append(values)
    interval: list[float | None] = [None, None]
    if bootstrap > 0:
        import numpy as np

        generator = np.random.default_rng(seed)
        flat = np.asarray(rows, dtype=float).ravel()
        probabilities = flat / flat.sum()
        samples = generator.multinomial(int(total), probabilities, size=bootstrap)
        estimates = sorted(
            _v(sample.reshape(n_rows, n_columns).tolist())[1] for sample in samples
        )
        interval = [estimates[int(0.025 * (bootstrap - 1))],
                    estimates[int(round(0.975 * (bootstrap - 1)))]]
    return {
        "chi_square": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": chi2_sf(statistic, degrees_of_freedom),
        "cramers_v": v,
        "cramers_v_corrected": corrected,
        "cramers_v_ci": interval,
        "adjusted_residuals": residuals,
    }


def rounded(value: Any, digits: int = 6) -> Any:
    """Round floats (recursively in lists/dicts) for compact JSON."""
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    return value
