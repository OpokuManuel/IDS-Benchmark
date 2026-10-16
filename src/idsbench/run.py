"""Run one (protocol, model, fold) experiment and persist it (resumable)."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from . import SEED
from .data import ROOT, load_raw, clean_pooled, feature_cols
from .metrics import evaluate
from .models import build_model
from .splits import make_groups, p1_folds, p2_folds, val_split

OUT = ROOT / "results" / "raw"


def _score(model, X):
    proba = model.predict_proba(X)
    classes = list(model.named_steps["clf"].classes_)
    return 1.0 - proba[:, classes.index("Normal")], model.named_steps["clf"].predict(model.named_steps["pre"].transform(X)) if False else np.array(classes)[proba.argmax(1)]


def get_data():
    tr, te = load_raw()
    clean, rep = clean_pooled(tr, te)
    return tr, te, clean, rep


def iter_folds(protocol, tr, te, clean, seed, n_splits=5, bins=2):
    """Yield (fold, train_df, test_df, val_idx_fit, val_idx_val, grouped) with indices into the df used."""
    if protocol == "P0_official":
        y = tr["attack_cat"].to_numpy()
        a, b = val_split(np.arange(len(tr)), y, None, False, seed)
        yield 0, tr, te, a, b
        return
    y = clean["attack_cat"].to_numpy()
    if protocol == "P1_random_clean":
        for k, (tri, tei) in enumerate(p1_folds(y, n_splits, seed)):
            a, b = val_split(tri, y, None, False, seed + k)
            yield k, clean.iloc[tri], clean.iloc[tei], a, b  # a,b index into clean
    elif protocol.startswith("P2_grouped"):
        g = make_groups(clean, bins)
        for k, (tri, tei) in enumerate(p2_folds(y, g, n_splits, seed)):
            a, b = val_split(tri, y, g, True, seed + k)
            yield k, clean.iloc[tri], clean.iloc[tei], a, b
    else:
        raise ValueError(protocol)


def run_protocol(protocol, model_name, seed=SEED, n_splits=5, bins=2, drop=(), tag="", only_folds=None):
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te, clean, _ = get_data()
    feats = feature_cols(tr)
    for k, trn, tst, ia, iv in iter_folds(protocol, tr, te, clean, seed, n_splits, bins):
        if only_folds is not None and k not in only_folds:
            continue
        name = f"{protocol}{tag}__{model_name}__seed{seed}__fold{k}"
        path = OUT / f"{name}.json"
        if path.exists():
            continue
        base = tr if protocol == "P0_official" else clean
        fit, val = base.iloc[ia], base.iloc[iv]
        m = build_model(model_name, fit[feats], seed, drop)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(fit[feats], fit["attack_cat"])
        fit_s = time.time() - t0
        s_val, _ = _score(m, val[feats]); s_te, pred = _score(m, tst[feats])
        res = evaluate(tst["attack_cat"].to_numpy(), pred, s_te, val["attack_cat"].to_numpy(), s_val)
        res.update(protocol=protocol + tag, model=model_name, seed=seed, fold=k, n_fit=len(fit), n_val=len(val),
                   n_test=len(tst), fit_seconds=round(fit_s, 1), dropped_features=list(drop), bins=bins)
        path.write_text(json.dumps(res))
        # out-of-fold predictions (row ids of the cleaned frame, or official ids for P0)
        rid = tst["row_id"].to_numpy() if "row_id" in tst else tst["id"].to_numpy()
        pd.DataFrame({"row_id": rid, "fold": k, "y_true": tst["attack_cat"].to_numpy(), "y_pred": pred,
                      "score": np.round(s_te, 5)}).to_csv(OUT / f"{name}.pred.csv.gz", index=False)
        print(f"[done] {name} acc={res['accuracy']:.4f} mF1={res['macro_f1']:.4f} auc={res['roc_auc']:.4f} fit={fit_s:.0f}s", flush=True)
