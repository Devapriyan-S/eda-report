"""edakit — profile a dataset and report what actually matters about it.

    from edakit import profile_dataframe, association_matrix, build_findings

Associations are measured on a common 0-1 scale across every column type, so a
categorical column that determines a numeric one is visible — something a plain
`df.corr()` cannot show.
"""

from .associations import (association_matrix, correlation_ratio, cramers_v,
                           missingness_correlation, pearson, spearman)
from .findings import Finding, build_findings
from .profile import infer_role, profile_column, profile_dataframe

__version__ = "1.0.0"
__all__ = [
    "profile_dataframe", "profile_column", "infer_role",
    "association_matrix", "cramers_v", "correlation_ratio", "spearman", "pearson",
    "missingness_correlation", "build_findings", "Finding",
]
