"""The three baselines. Hyper-parameters were chosen with scripts/00_tune_on_validation.py, which uses
only an inner validation split of one TRAIN fold (never test data), then frozen here."""
from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .features import make_preprocessor

MODEL_NAMES = ["logreg", "random_forest", "grad_boosting"]


def build_model(name: str, df_example, seed: int, drop: tuple[str, ...] = ()):
    if name == "logreg":
        pre = make_preprocessor(df_example, scale=True, drop=drop)
        clf = LogisticRegression(C=10.0, max_iter=500, random_state=seed)
    elif name == "random_forest":
        pre = make_preprocessor(df_example, scale=False, drop=drop)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
    elif name == "grad_boosting":
        pre = make_preprocessor(df_example, scale=False, drop=drop)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])
