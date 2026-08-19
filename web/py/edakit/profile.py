"""Per-column profiling: role, distribution, and everything the report needs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_DISCRETE_NUMERIC = 12
IDENTIFIER_UNIQUE_RATIO = 0.9
TEXT_TOKEN_THRESHOLD = 3.0


def infer_role(s: pd.Series) -> str:
    non_null = s.dropna()
    if non_null.empty:
        return "empty"
    if non_null.nunique() <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(non_null):
        return "categorical"

    if pd.api.types.is_numeric_dtype(non_null):
        integral = pd.api.types.is_integer_dtype(non_null) or non_null.mod(1).eq(0).all()
        n_unique = non_null.nunique()
        if integral and n_unique <= MAX_DISCRETE_NUMERIC:
            return "categorical"
        if integral and n_unique == len(non_null) and n_unique > 20:
            # Dense contiguous range means a row index; a rounded price is not.
            span = float(non_null.max() - non_null.min()) + 1
            if span <= n_unique * 1.5:
                return "identifier"
        return "numeric"

    as_str = non_null.astype(str)
    sample = as_str.head(200)
    if sample.str.contains(r"[-/:]").any():
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            return "datetime"

    n_unique = non_null.nunique()
    if as_str.head(500).str.split().str.len().mean() >= TEXT_TOKEN_THRESHOLD:
        return "text"
    if n_unique / len(non_null) > IDENTIFIER_UNIQUE_RATIO and n_unique > 50:
        return "identifier"
    return "categorical"


def _histogram(values: np.ndarray, bins: int = 24) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {}
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return {"edges": [lo, lo + 1], "counts": [int(values.size)]}
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.histogram(values, bins=edges)[0]
    return {"edges": [round(float(e), 6) for e in edges],
            "counts": [int(c) for c in counts]}


def profile_column(s: pd.Series, n_rows: int) -> dict[str, Any]:
    role = infer_role(s)
    non_null = s.dropna()
    n_missing = int(s.isna().sum())

    out: dict[str, Any] = {
        "name": str(s.name),
        "role": role,
        "dtype": str(s.dtype),
        "n_valid": int(len(non_null)),
        "n_missing": n_missing,
        "missing_pct": round(100.0 * n_missing / n_rows, 2) if n_rows else 0.0,
        "n_unique": int(non_null.nunique()) if len(non_null) else 0,
        "stats": {},
    }

    if role == "numeric":
        v = pd.to_numeric(non_null, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            out["stats"] = {
                "mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
                "min": round(float(v.min()), 4),
                "q1": round(float(q1), 4),
                "median": round(float(med), 4),
                "q3": round(float(q3), 4),
                "max": round(float(v.max()), 4),
                "skew": round(float(pd.Series(v).skew()), 4) if v.size > 2 else 0.0,
                "n_outliers": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()),
                "n_zero": int((v == 0).sum()),
                "n_negative": int((v < 0).sum()),
            }
            out["histogram"] = _histogram(v)

    elif role in {"categorical", "text", "identifier"}:
        counts = non_null.astype(str).value_counts()
        if len(counts):
            out["stats"] = {
                "top_value": str(counts.index[0]),
                "top_count": int(counts.iloc[0]),
                "top_share": round(float(counts.iloc[0] / len(non_null)), 4),
                "levels": [{"value": str(k), "count": int(v)} for k, v in counts.head(12).items()],
            }

    elif role == "datetime":
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed").dropna()
        if len(parsed):
            out["stats"] = {
                "min": str(parsed.min().date()),
                "max": str(parsed.max().date()),
                "span_days": int((parsed.max() - parsed.min()).days),
            }
    return out


def profile_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [profile_column(df[c], len(df)) for c in df.columns]
