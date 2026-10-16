"""Data-quality + leakage audit. Writes results/audit.json, data_quality.csv, leakage_audit.csv."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from idsbench import SEED
from idsbench.data import ROOT, CLASSES, load_raw, clean_pooled, feature_cols
from idsbench.features import distance_space
from idsbench.splits import make_groups, p1_folds, p2_folds

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
NEAR = 0.1          # "near-duplicate": 1-NN distance below 0.1 in the standardised log-feature space
N_Q = 2000          # test rows sampled per fold for the nearest-neighbour search


def nn_dist(Xtr, Xq):
    a2 = (Xtr ** 2).sum(1); out = []
    for i in range(0, len(Xq), 400):
        b = Xq[i:i + 400]
        d = (b ** 2).sum(1)[:, None] + a2[None, :] - 2 * b @ Xtr.T
        out.append(np.sqrt(np.clip(d.min(1), 0, None)))
    return np.concatenate(out)


def summarise(d):
    return dict(median_nn_dist=float(np.median(d)), frac_near_twin=float((d < NEAR).mean()),
                frac_within_0p5=float((d < 0.5).mean()))


tr, te = load_raw(); feats = feature_cols(tr)
clean, rep = clean_pooled(tr, te)
audit = {"cleaning": rep}

# ---- A. data quality, per class ------------------------------------------------------
raw = pd.concat([tr, te], ignore_index=True)
raw["vec"] = raw.groupby(feats, sort=False).ngroup()
ncat = raw.groupby("vec")["attack_cat"].transform("nunique")
rows = []
for c in CLASSES:
    r = raw[raw.attack_cat == c]
    rows.append(dict(attack_cat=c, raw_rows=len(r), unique_vectors=int(r.vec.nunique()),
                     duplicate_share=1 - r.vec.nunique() / len(r),
                     share_rows_in_category_conflict=float((ncat[r.index] > 1).mean()),
                     clean_rows=int((clean.attack_cat == c).sum())))
dq = pd.DataFrame(rows); dq.to_csv(RES / "data_quality.csv", index=False)
print(dq.round(3).to_string())

# shortcut evidence: what is "service == dns" in the official TRAINING file vs the cleaned pool
audit["dns_share_train_by_class"] = (tr[tr.service == "dns"].attack_cat.value_counts(normalize=True).round(4).to_dict())
audit["udp_dns_rows_train"] = int(((tr.proto == "udp") & (tr.service == "dns")).sum())
audit["classes_official_train"] = tr.attack_cat.value_counts().to_dict()
audit["classes_official_test"] = te.attack_cat.value_counts().to_dict()

# ---- B. leakage in the OFFICIAL split (raw rows) ---------------------------------------
tr_keys = set(map(tuple, tr[feats].to_numpy()))
audit["official_test_rows_with_exact_twin_in_train"] = int(sum(tuple(r) in tr_keys for r in te[feats].to_numpy()))
audit["official_test_rows"] = len(te)
audit["official_train_duplicate_rows"] = int(tr.duplicated(feats).sum())
audit["official_test_duplicate_rows"] = int(te.duplicated(feats).sum())
Xraw = distance_space(pd.concat([tr, te], ignore_index=True))
Xtr_raw, Xte_raw = Xraw[:len(tr)], Xraw[len(tr):]
q = rng.choice(len(te), N_Q, replace=False)
leak = [dict(protocol="P0_official (raw)", fold=0, **summarise(nn_dist(Xtr_raw, Xte_raw[q])))]

# ---- C. leakage in the cleaned-data protocols --------------------------------------------
X = distance_space(clean); y = clean.attack_cat.to_numpy()
def add(name, folds, groups=None):
    for k, (tri, tei) in enumerate(folds):
        s = rng.choice(tei, min(N_Q, len(tei)), replace=False)
        rec = dict(protocol=name, fold=k, **summarise(nn_dist(X[tri], X[s])))
        if groups is not None:
            rec["groups_shared_train_test"] = int(len(set(groups[tri]) & set(groups[tei])))
        leak.append(rec)
add("P1_random_clean", p1_folds(y, 5, SEED))
for bins in (1, 2, 4):
    g = make_groups(clean, bins)
    audit[f"groups_bins{bins}"] = dict(n_groups=int(g.max() + 1), largest=int(np.bincount(g).max()),
                                      singletons=int((np.bincount(g) == 1).sum()))
    add("P2_grouped" + ("" if bins == 2 else f"_bins{bins}"), p2_folds(y, g, 5, SEED), g)
L = pd.DataFrame(leak); L.to_csv(RES / "leakage_audit.csv", index=False)
print(L.groupby("protocol")[["median_nn_dist", "frac_near_twin", "frac_within_0p5"]].mean().round(3))
(RES / "audit.json").write_text(json.dumps(audit, indent=1, default=int))
