"""Turn a profiled dataset into ranked, plainly-worded findings.

The output of most auto-EDA tools is a wall of charts and a correlation
heatmap, leaving the reader to spot what matters. This module does the spotting:
each finding carries a severity, a one-line statement of the fact, and — where
it changes what you should do — a sentence on the consequence.

Findings are ranked by how much they would change an analysis, not by how
interesting they look.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"critical": 0, "warning": 1, "note": 2, "info": 3}


@dataclass
class Finding:
    severity: str        # critical | warning | note | info
    category: str
    title: str
    detail: str
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_findings(
    df: pd.DataFrame,
    profiles: list[dict],
    pairs: list[dict],
    missing_pairs: list[dict],
    *,
    strong_association: float = 0.6,
) -> list[Finding]:
    out: list[Finding] = []
    n = len(df)
    by_name = {p["name"]: p for p in profiles}

    # ── Structural problems ──────────────────────────────────
    dupes = int(df.duplicated().sum())
    if dupes:
        out.append(Finding(
            "critical" if dupes > n * 0.05 else "warning", "duplicates",
            f"{dupes:,} duplicate rows ({_pct(dupes / n)} of the data)",
            "Identical rows inflate every count and can leak across a train/test "
            "split, making a model look better than it is. Check whether they are "
            "genuine repeat events or an artefact of a join.",
            [],
        ))

    empty = [p["name"] for p in profiles if p["missing_pct"] >= 100]
    if empty:
        out.append(Finding(
            "warning", "empty",
            f"{len(empty)} column(s) are entirely empty",
            f"{', '.join(empty[:6])} contain no values at all and can be dropped.",
            empty,
        ))

    constant = [p["name"] for p in profiles if p["role"] == "constant"]
    if constant:
        out.append(Finding(
            "note", "constant",
            f"{len(constant)} column(s) hold a single value",
            f"{', '.join(constant[:6])} carry no information and cannot help any model.",
            constant,
        ))

    # ── Missingness ──────────────────────────────────────────
    heavy = [p for p in profiles if 0 < p["missing_pct"] < 100]
    severe = [p for p in heavy if p["missing_pct"] > 40]
    if severe:
        worst = max(severe, key=lambda p: p["missing_pct"])
        out.append(Finding(
            "critical", "missing",
            f"{len(severe)} column(s) are more than 40% missing",
            f"{worst['name']} is {worst['missing_pct']:.0f}% empty. Imputing a column "
            f"this sparse mostly invents data; dropping it is usually more honest.",
            [p["name"] for p in severe],
        ))
    moderate = [p for p in heavy if 5 < p["missing_pct"] <= 40]
    if moderate:
        out.append(Finding(
            "warning", "missing",
            f"{len(moderate)} column(s) have 5–40% missing values",
            f"{', '.join(p['name'] for p in moderate[:5])}. A row-wise dropna() would "
            f"discard {_pct(float(df[[p['name'] for p in moderate]].isna().any(axis=1).mean()))} "
            f"of the dataset — impute per column instead.",
            [p["name"] for p in moderate],
        ))
    for mp in missing_pairs[:3]:
        out.append(Finding(
            "warning", "missing-pattern",
            f"{mp['a']} and {mp['b']} go missing together",
            f"{_pct(mp['overlap'])} of the rows missing either are missing both "
            f"({mp['both']:,} rows). Values that disappear in pairs usually point to "
            f"one upstream cause — a failed join or a skipped form section — not to "
            f"two independent gaps.",
            [mp["a"], mp["b"]],
        ))

    # ── Distributions ────────────────────────────────────────
    for p in profiles:
        if p["role"] != "numeric":
            continue
        s = p.get("stats", {})
        skew = s.get("skew")
        if skew is not None and abs(skew) > 2:
            out.append(Finding(
                "note", "skew",
                f"{p['name']} is strongly skewed (skew {skew:.1f})",
                "A linear model will be dominated by the tail. A log or Box-Cox "
                "transform usually helps; tree models are unaffected.",
                [p["name"]],
            ))
        outliers = s.get("n_outliers", 0)
        if outliers and outliers / max(p["n_valid"], 1) > 0.05:
            out.append(Finding(
                "note", "outliers",
                f"{p['name']} has {outliers:,} outliers "
                f"({_pct(outliers / p['n_valid'])} beyond 1.5×IQR)",
                f"Range is {s.get('min')} to {s.get('max')} against a median of "
                f"{s.get('median')}. Check whether these are errors or a genuine "
                f"heavy tail before removing anything.",
                [p["name"]],
            ))
        if s.get("n_zero", 0) / max(p["n_valid"], 1) > 0.5:
            out.append(Finding(
                "note", "zeros",
                f"{p['name']} is more than half zeros",
                "A zero-inflated column often means two populations — those with the "
                "behaviour and those without. Modelling them separately usually beats "
                "one model over both.",
                [p["name"]],
            ))

    for p in profiles:
        if p["role"] != "categorical":
            continue
        s = p.get("stats", {})
        top_share = s.get("top_share", 0)
        if top_share > 0.95:
            out.append(Finding(
                "note", "imbalance",
                f"{p['name']} is {_pct(top_share)} one value",
                f"'{s.get('top_value')}' dominates. As a feature this is nearly "
                f"constant; as a target it makes accuracy meaningless.",
                [p["name"]],
            ))

    # ── Relationships ────────────────────────────────────────
    strong = [p for p in pairs if p["strength"] >= strong_association]
    for pair in strong[:6]:
        method = {
            "spearman": "rank correlation",
            "cramers_v": "Cramér's V",
            "correlation_ratio": "correlation ratio",
        }[pair["method"]]

        detail = (f"{method} {pair['strength']:.2f}. ")
        if pair["method"] == "spearman":
            lin = abs(pair.get("pearson", 0))
            if pair["strength"] - lin > 0.2:
                detail += (f"Pearson is only {lin:.2f}, so the relationship is real "
                           f"but curved — a linear model will underfit it. ")
            else:
                detail += "Near-linear. "
            detail += ("Two features this correlated make regression coefficients "
                       "unstable; consider keeping one.")
        elif pair["method"] == "correlation_ratio":
            detail += (f"Knowing {pair['grouping']} explains most of the variance in "
                       f"the other column — a Pearson correlation would have shown "
                       f"nothing here.")
        else:
            detail += ("These two categoricals carry largely the same information.")

        out.append(Finding(
            "warning" if pair["strength"] > 0.9 else "note",
            "association",
            f"{pair['a']} and {pair['b']} are strongly related",
            detail, [pair["a"], pair["b"]],
        ))

    # ── Identifiers ──────────────────────────────────────────
    ids = [p["name"] for p in profiles if p["role"] == "identifier"]
    if ids:
        out.append(Finding(
            "info", "identifier",
            f"{len(ids)} column(s) look like identifiers",
            f"{', '.join(ids[:5])} are near-unique per row. Useful for joining, "
            f"harmful as model features — they let a model memorise rows.",
            ids,
        ))

    if not out:
        out.append(Finding(
            "info", "clean", "Nothing notable found",
            "No duplicates, no heavy missingness, no extreme skew, and no pair of "
            "columns strongly enough related to cause trouble.", [],
        ))

    out.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    return out
