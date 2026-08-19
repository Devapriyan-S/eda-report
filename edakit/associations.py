"""Association strength between columns of any type.

``df.corr()`` only speaks Pearson, and Pearson only speaks numeric-to-numeric
and only sees straight lines. A dataset where `city` almost perfectly
determines `delivery_time`, or where `age` and `income` are related through a
curve, shows up as nothing at all.

This module measures every pair on a common 0–1 scale:

* numeric ↔ numeric      — Spearman (rank), which catches any monotone relation
* categorical ↔ categorical — Cramér's V, bias-corrected
* numeric ↔ categorical  — the correlation ratio η, i.e. how much of the
  numeric column's variance the grouping explains

All three land in [0, 1] where 0 is independence and 1 is complete
determination, so one matrix can hold the whole dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V with Bergsma's bias correction.

    The uncorrected statistic is biased upward when either variable has many
    levels relative to the sample — with 20 categories and 200 rows it reports
    a strong association between two independent columns.
    """
    table = pd.crosstab(a, b)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corrected - 1, r_corrected - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio η: the share of a numeric column's variance explained
    by a categorical grouping. This is what a Pearson correlation cannot see."""
    df = pd.DataFrame({"c": categories, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if len(df) < 3 or df["c"].nunique() < 2:
        return 0.0
    grand_mean = df["v"].mean()
    total_ss = ((df["v"] - grand_mean) ** 2).sum()
    if total_ss <= 0:
        return 0.0
    between_ss = sum(
        len(g) * (g["v"].mean() - grand_mean) ** 2
        for _, g in df.groupby("c", observed=True)
    )
    return float(np.sqrt(max(0.0, between_ss / total_ss)))


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Absolute Spearman correlation — monotone, not merely linear."""
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                       "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(df) < 3 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return 0.0
    rho = stats.spearmanr(df["a"], df["b"]).statistic
    return 0.0 if np.isnan(rho) else float(abs(rho))


def pearson(a: pd.Series, b: pd.Series) -> float:
    """Signed Pearson correlation, kept alongside Spearman.

    The gap between the two is informative: a strong Spearman with a weak
    Pearson means the relationship is real but curved.
    """
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                       "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(df) < 3 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return 0.0
    r = stats.pearsonr(df["a"], df["b"]).statistic
    return 0.0 if np.isnan(r) else float(r)


def association_matrix(
    df: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> tuple[list[str], np.ndarray, list[dict]]:
    """Pairwise association for every usable column pair.

    Returns the column order, a symmetric matrix in [0, 1], and a list of the
    strongest pairs with the method used and a plain-language reading.
    """
    columns = [c for c in df.columns if c in numeric or c in categorical]
    n = len(columns)
    matrix = np.eye(n)
    pairs: list[dict] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = columns[i], columns[j]
            a_num, b_num = a in numeric, b in numeric

            if a_num and b_num:
                strength = spearman(df[a], df[b])
                lin = pearson(df[a], df[b])
                method, detail = "spearman", {"pearson": round(lin, 4)}
            elif not a_num and not b_num:
                strength = cramers_v(df[a].astype(str), df[b].astype(str))
                method, detail = "cramers_v", {}
            else:
                cat, num = (b, a) if a_num else (a, b)
                strength = correlation_ratio(df[cat].astype(str), df[num])
                method, detail = "correlation_ratio", {"grouping": cat}

            matrix[i, j] = matrix[j, i] = strength
            pairs.append({"a": a, "b": b, "strength": round(float(strength), 4),
                          "method": method, **detail})

    pairs.sort(key=lambda p: -p["strength"])
    return columns, matrix, pairs


def missingness_correlation(df: pd.DataFrame, min_missing: int = 1) -> list[dict]:
    """Do columns go missing *together*?

    Two columns whose blanks coincide almost always means one upstream event
    dropped both — a failed join, a form section nobody filled in, a sensor
    outage. That is a different problem from two columns that are each
    independently patchy, and dropna() will treat them very differently.
    """
    flags = {c: df[c].isna() for c in df.columns if df[c].isna().sum() >= min_missing}
    names = list(flags)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            fa, fb = flags[a], flags[b]
            # Jaccard: of the rows missing either, how many are missing both.
            union = (fa | fb).sum()
            if union == 0:
                continue
            overlap = float((fa & fb).sum() / union)
            if overlap > 0.5:
                out.append({"a": a, "b": b, "overlap": round(overlap, 4),
                            "both": int((fa & fb).sum())})
    out.sort(key=lambda x: -x["overlap"])
    return out
