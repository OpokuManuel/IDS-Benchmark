"""Loading, auditing and cleaning the official UNSW-NB15 train/test CSVs.

The official CSVs carry NO timestamps and NO IP addresses, so a true
time- or session-based split is impossible with them. See README, "Limitations".
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TRAIN_CSV = RAW / "UNSW_NB15_training-set.csv"
TEST_CSV = RAW / "UNSW_NB15_testing-set.csv"

CAT_COLS = ["proto", "service", "state"]
TARGET_COLS = ["attack_cat", "label"]
ID_COL = "id"
CLASSES = ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
           "Generic", "Reconnaissance", "Shellcode", "Worms"]


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official partitions exactly as published (BOM stripped, labels trimmed)."""
    tr = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    te = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    for d in (tr, te):
        d["attack_cat"] = d["attack_cat"].str.strip()
        assert set(d["attack_cat"]) <= set(CLASSES), set(d["attack_cat"]) - set(CLASSES)
    return tr, te


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGET_COLS + [ID_COL]]


def clean_pooled(tr: pd.DataFrame, te: pd.DataFrame):
    """Pool both official files, then:

    1. drop vectors whose identical features carry BOTH benign and malicious
       labels (binary ground truth is contradictory),
    2. collapse exact duplicate feature vectors to one row,
    3. for vectors with several attack categories (same binary label), keep the
       most frequent category (ties -> alphabetical).

    Returns (clean_df, report_dict). `clean_df` has a stable `row_id`.
    """
    feats = feature_cols(tr)
    al = pd.concat([tr.assign(src="train"), te.assign(src="test")], ignore_index=True)
    n_raw = len(al)
    # integer code per unique feature vector
    al["vec"] = al.groupby(feats, sort=False).ngroup()
    g = al.groupby("vec")
    n_bin = g["label"].transform("nunique")
    n_cat = g["attack_cat"].transform("nunique")
    rep = {
        "rows_raw": n_raw,
        "rows_official_train": len(tr), "rows_official_test": len(te),
        "unique_vectors": int(al["vec"].nunique()),
        "exact_duplicate_rows_removed": int(n_raw - al["vec"].nunique()),
        "vectors_binary_conflict": int(al.loc[n_bin > 1, "vec"].nunique()),
        "rows_in_binary_conflict": int((n_bin > 1).sum()),
        "vectors_category_conflict_only": int(al.loc[(n_cat > 1) & (n_bin == 1), "vec"].nunique()),
        "rows_in_category_conflict": int((n_cat > 1).sum()),
    }
    al = al[n_bin == 1]
    # majority category per vector (ties alphabetical)
    cnt = al.groupby(["vec", "attack_cat"]).size().rename("n").reset_index()
    cnt = cnt.sort_values(["vec", "n", "attack_cat"], ascending=[True, False, True])
    maj = cnt.drop_duplicates("vec").set_index("vec")["attack_cat"]
    clean = al.drop_duplicates("vec").copy()
    clean["attack_cat"] = clean["vec"].map(maj)
    clean["label"] = (clean["attack_cat"] != "Normal").astype(int)
    clean = clean.sort_values("vec").reset_index(drop=True)
    clean["row_id"] = np.arange(len(clean))
    rep["rows_clean"] = len(clean)
    rep["clean_class_counts"] = clean["attack_cat"].value_counts().to_dict()
    return clean, rep
