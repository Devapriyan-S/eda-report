"""JSON bridge between the browser UI and edakit."""

import io
import json
import traceback
import warnings

import numpy as np
import pandas as pd

from edakit import (association_matrix, build_findings, missingness_correlation,
                    profile_dataframe)

warnings.filterwarnings("ignore")

# Association is O(columns^2) with a chi-square or ANOVA per pair, so a
# 200-column file would take minutes. Beyond this only the columns most likely
# to matter are paired, and the UI says so.
MAX_ASSOCIATION_COLUMNS = 30


def _json(payload) -> str:
    return json.dumps(payload, default=str)


def analyse(text: str, name: str) -> str:
    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty:
            raise ValueError("The file parsed to zero rows.")

        profiles = profile_dataframe(df)
        by_role = {}
        for p in profiles:
            by_role.setdefault(p["role"], []).append(p["name"])

        numeric = by_role.get("numeric", [])
        categorical = by_role.get("categorical", [])
        candidates = numeric + categorical
        truncated = 0
        if len(candidates) > MAX_ASSOCIATION_COLUMNS:
            # Keep the columns with the most variation — a column that is 99%
            # one value cannot be strongly associated with anything anyway.
            ranked = sorted(
                candidates,
                key=lambda c: -min(df[c].nunique(), len(df) - df[c].isna().sum()),
            )
            truncated = len(candidates) - MAX_ASSOCIATION_COLUMNS
            keep = set(ranked[:MAX_ASSOCIATION_COLUMNS])
            numeric = [c for c in numeric if c in keep]
            categorical = [c for c in categorical if c in keep]

        cols, matrix, pairs = association_matrix(df, numeric, categorical)
        missing_pairs = missingness_correlation(df)
        findings = build_findings(df, profiles, pairs, missing_pairs)

        return _json({
            "ok": True,
            "name": name,
            "shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024 ** 2, 3),
            "duplicates": int(df.duplicated().sum()),
            "total_missing": int(df.isna().sum().sum()),
            "roles": {k: len(v) for k, v in by_role.items()},
            "profiles": profiles,
            "association": {
                "columns": cols,
                "matrix": [[round(float(v), 4) for v in row] for row in matrix],
                "pairs": pairs[:25],
                "truncated": truncated,
            },
            "missing_pairs": missing_pairs[:8],
            "findings": [f.to_dict() for f in findings],
            "preview": {
                "columns": list(df.columns)[:14],
                "rows": df.head(8).iloc[:, :14].astype(object)
                          .where(df.head(8).iloc[:, :14].notna(), None).values.tolist(),
            },
        })
    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()})
