"""Split protocols. P0 official, P1 random-on-clean, P2 leakage-safe grouped."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split


def make_groups(df: pd.DataFrame, bins_per_octave: int = 1, use_dur: bool = False) -> np.ndarray:
    """Near-duplicate / 'session proxy' groups.

    Flows that share protocol, service, state, both TTLs and the same coarse
    log-scale size profile (packets and bytes, each direction) get one group id.
    These are the flows an attack tool or traffic generator emits repeatedly, i.e.
    exactly what a random split would scatter across train and test.
    No label is used. Finer `bins_per_octave` -> more, smaller groups.
    """
    parts = [df["proto"], df["service"], df["state"], df["sttl"], df["dttl"]]
    for c in ["spkts", "dpkts", "sbytes", "dbytes"] + (["dur"] if use_dur else []):
        v = np.log2(df[c].to_numpy(float) + 1.0) * bins_per_octave
        parts.append(pd.Series(np.floor(v).astype(int), index=df.index))
    sig = pd.concat(parts, axis=1).astype(str).agg("|".join, axis=1)
    return pd.factorize(sig)[0]


def p1_folds(y, n_splits, seed):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y)), y)


def p2_folds(y, groups, n_splits, seed):
    sgk = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    yield from sgk.split(np.zeros(len(y)), y, groups)


def val_split(train_idx, y, groups, grouped: bool, seed: int, frac: float = 0.15):
    """Carve a validation set (for threshold choice) out of the TRAIN fold only,
    using the same protocol as the outer split."""
    ti = np.asarray(train_idx)
    if grouped:
        k = int(round(1 / frac))
        sgk = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
        a, b = next(sgk.split(np.zeros(len(ti)), y[ti], groups[ti]))
    else:
        a, b = train_test_split(np.arange(len(ti)), test_size=frac, stratify=y[ti], random_state=seed)
    return ti[a], ti[b]
