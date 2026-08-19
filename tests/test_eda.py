"""The associations must find relationships that df.corr() cannot see."""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from edakit import (profile_dataframe, infer_role, association_matrix, cramers_v,
                    correlation_ratio, spearman, missingness_correlation, build_findings)

failures = []
def check(label, cond, detail=""):
    if not cond: failures.append(label)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {label}{('  — ' + detail) if detail else ''}")

rng = np.random.default_rng(13)
N = 1200

print("\n1. Column roles\n")
df = pd.DataFrame({
    "order_id":  [f"ORD{i:06d}" for i in range(N)],
    "row_index": np.arange(N),
    "price":     rng.gamma(2, 90, N).round(2),
    "rating":    rng.integers(1, 6, N),
    "city":      rng.choice(["Chennai","Mumbai","Delhi"], N),
    "ordered_at": pd.date_range("2023-01-01", periods=N, freq="h").astype(str),
    "review":    rng.choice(["great product highly recommended","poor quality do not buy"], N),
    "country":   ["IN"] * N,
})
roles = {p["name"]: p["role"] for p in profile_dataframe(df)}
for name, want in [("order_id","identifier"), ("row_index","identifier"), ("price","numeric"),
                   ("rating","categorical"), ("city","categorical"),
                   ("ordered_at","datetime"), ("review","text"), ("country","constant")]:
    check(f"{name} -> {want}", roles[name] == want, f"got {roles[name]}")

print("\n2. Cramér's V\n")
a = pd.Series(rng.choice(list("ABC"), N))
check("independent categoricals ~ 0", cramers_v(a, pd.Series(rng.choice(list("XYZ"), N))) < 0.1,
      f"{cramers_v(a, pd.Series(rng.choice(list('XYZ'), N))):.4f}")
check("identical categoricals ~ 1", cramers_v(a, a.copy()) > 0.95, f"{cramers_v(a, a.copy()):.4f}")
# The bias correction is the point: 30 levels on 200 rows fools the raw statistic.
small_a = pd.Series(rng.choice([f"L{i}" for i in range(30)], 200))
small_b = pd.Series(rng.choice([f"M{i}" for i in range(30)], 200))
v = cramers_v(small_a, small_b)
check("bias-corrected on many levels / few rows", v < 0.15, f"{v:.4f} (uncorrected would be ~0.7)")

print("\n3. Correlation ratio finds what Pearson cannot\n")
groups = rng.choice(["low","mid","high"], N)
value = pd.Series(np.where(groups=="low", 10, np.where(groups=="mid", 50, 200))
                  + rng.normal(0, 6, N))
eta = correlation_ratio(pd.Series(groups), value)
check("category strongly determines the numeric column", eta > 0.9, f"eta {eta:.3f}")
check("unrelated category scores low",
      correlation_ratio(pd.Series(rng.choice(list("PQR"), N)), value) < 0.15,
      f"{correlation_ratio(pd.Series(rng.choice(list('PQR'), N)), value):.3f}")

print("\n4. Spearman sees curves that Pearson misses\n")
x = pd.Series(np.linspace(-3, 3, N))
# Multiplicative noise, which is what an exponential relationship actually has.
# Additive noise of a fixed size swamps exp(-3) = 0.05 entirely and breaks
# monotonicity at the low end, so Spearman would be penalised for the test's
# construction rather than for the measure.
y = pd.Series(np.exp(x) * np.exp(rng.normal(0, 0.12, N)))
from edakit import pearson
sp, pe = spearman(x, y), abs(pearson(x, y))
print(f"     exponential relation: Spearman {sp:.3f}  Pearson {pe:.3f}")
check("Spearman near 1 on a monotone curve", sp > 0.97, f"{sp:.3f}")
check("Spearman clearly exceeds Pearson", sp - pe > 0.15, f"gap {sp-pe:.3f}")

print("\n5. Association matrix across mixed types\n")
mixed = pd.DataFrame({"tier": groups, "spend": value,
                      "noise": rng.normal(0, 1, N),
                      "letter": rng.choice(list("AB"), N)})
cols, matrix, pairs = association_matrix(mixed, ["spend","noise"], ["tier","letter"])
check("matrix is square and symmetric",
      matrix.shape == (4,4) and bool(np.allclose(matrix, matrix.T)))
check("diagonal is 1", bool(np.allclose(np.diag(matrix), 1.0)))
check("all values in [0,1]", bool(matrix.min() >= 0 and matrix.max() <= 1.0001))
top = pairs[0]
check("strongest pair is tier<->spend", {top["a"], top["b"]} == {"tier","spend"},
      f"{top['a']}<->{top['b']} {top['strength']:.3f}")
check("that pair used the correlation ratio", top["method"] == "correlation_ratio")
print(f"     top 3: " + "  ".join(f"{p['a']}~{p['b']}:{p['strength']:.2f}" for p in pairs[:3]))

print("\n6. Missingness that travels together\n")
mdf = pd.DataFrame({"a": rng.normal(0,1,500), "b": rng.normal(0,1,500), "c": rng.normal(0,1,500)})
joint = rng.choice(500, 120, replace=False)
mdf.loc[joint, "a"] = np.nan
mdf.loc[joint, "b"] = np.nan                      # a and b fail together
mdf.loc[rng.choice(500, 60, replace=False), "c"] = np.nan   # c fails alone
mp = missingness_correlation(mdf)
check("detects the paired failure", any({m["a"],m["b"]} == {"a","b"} for m in mp), str(mp))
check("does not flag the independent one", not any("c" in (m["a"],m["b"]) for m in mp))

print("\n7. Findings are ranked and specific\n")
dirty = pd.DataFrame({
    "id":      [f"X{i}" for i in range(400)],
    "mostly_missing": [1.0 if i < 40 else np.nan for i in range(400)],
    "skewed":  np.concatenate([rng.gamma(1, 2, 390), [900, 950, 1000, 1100, 1200,
                                                       1300, 1400, 1500, 1600, 1700]]),
    "flat":    ["same"] * 400,
    "grp":     rng.choice(["a","b"], 400),
})
dirty = pd.concat([dirty, dirty.head(30)])        # duplicates
prof = profile_dataframe(dirty)
_, _, prs = association_matrix(dirty, ["mostly_missing","skewed"], ["grp"])
found = build_findings(dirty, prof, prs, missingness_correlation(dirty))
cats = [f.category for f in found]
for f in found[:6]:
    print(f"     [{f.severity:8}] {f.title}")
check("duplicates flagged", "duplicates" in cats)
check("heavy missingness flagged as critical",
      any(f.category == "missing" and f.severity == "critical" for f in found))
check("constant column flagged", "constant" in cats)
check("skew flagged", "skew" in cats)
check("identifier flagged", "identifier" in cats)
check("critical findings come first", found[0].severity == "critical", found[0].severity)

print("\n8. A clean dataset produces no alarm\n")
clean = pd.DataFrame({"a": rng.normal(0,1,600), "b": rng.normal(5,2,600),
                      "g": rng.choice(["x","y","z"], 600)})
cf = build_findings(clean, profile_dataframe(clean),
                    association_matrix(clean, ["a","b"], ["g"])[2],
                    missingness_correlation(clean))
check("no critical or warning findings on clean data",
      all(f.severity in {"note","info"} for f in cf),
      str([(f.severity, f.title) for f in cf]))

print("\n" + "=" * 68)
print("FAILURES:" if failures else "All EDA checks passed.")
for f in failures: print("  -", f)
sys.exit(1 if failures else 0)
