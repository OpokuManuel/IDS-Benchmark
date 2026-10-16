"""Feature handling shared by the models and the leakage diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from .data import CAT_COLS, TARGET_COLS, ID_COL

# Raw TCP sequence numbers: effectively random per connection, but they can act as
# connection fingerprints. Kept as model features by default (ablated in the report),
# but excluded from the *distance* used for leakage diagnostics.
SEQ_COLS = ["stcpb", "dtcpb"]
NON_FEATURES = TARGET_COLS + [ID_COL, "row_id", "vec", "src"]


def model_columns(df: pd.DataFrame, drop: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in NON_FEATURES and c not in drop]
    cat = [c for c in cols if c in CAT_COLS]
    num = [c for c in cols if c not in CAT_COLS]
    return num, cat


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def make_preprocessor(df: pd.DataFrame, scale: bool, drop: tuple[str, ...] = ()) -> ColumnTransformer:
    num, cat = model_columns(df, drop)
    num_steps = [("log", FunctionTransformer(_log1p, feature_names_out="one-to-one"))]
    if scale:
        num_steps.append(("std", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(max_categories=12, handle_unknown="infrequent_if_exist",
                              sparse_output=False), cat),
    ])


def distance_space(df: pd.DataFrame) -> np.ndarray:
    """Standardised log-feature space used ONLY to measure near-duplicate leakage."""
    num, _ = model_columns(df, drop=tuple(SEQ_COLS))
    num = [c for c in num if c not in CAT_COLS]
    X = _log1p(df[num].to_numpy(dtype=float))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cats = pd.get_dummies(df[CAT_COLS].astype(str), dtype=float).to_numpy()
    # categorical match counts as a full standardised unit of difference
    return np.hstack([X, cats]).astype(np.float32)
