"""Aggregate raw runs into the tables and figures used by the README and the PDF write-up."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from idsbench.data import ROOT, CLASSES, load_raw, clean_pooled
from idsbench.metrics import evaluate

RES = ROOT / "results"; FIG = RES / "figures"; FIG.mkdir(exist_ok=True, parents=True)
MODELS = ["logreg", "random_forest", "grad_boosting"]
LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest", "grad_boosting": "Gradient boosting"}
COL = {"logreg": "#8c8c8c", "random_forest": "#1f6f8b", "grad_boosting": "#d1495b"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

# ---------- load every run -------------------------------------------------------------
runs = []
for p in sorted(glob.glob(str(RES / "raw" / "*.json"))):
    r = json.load(open(p)); r["pred"] = pd.read_csv(p.replace(".json", ".pred.csv.gz")); runs.append(r)
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred", "per_class", "confusion")} for r in runs])
print(df.groupby(["protocol", "model", "seed"]).size().unstack("seed").to_string())

def pooled(protocol, model, seed):
    rs = [r for r in runs if r["protocol"] == protocol and r["model"] == model and r["seed"] == seed]
    need = 1 if protocol == "P0_official" else 5
    if len(rs) < need: return None          # never aggregate a partially finished seed
    P = pd.concat([r["pred"] for r in rs], ignore_index=True)
    m = evaluate(P.y_true.to_numpy(), P.y_pred.to_numpy(), P.score.to_numpy())
    m["val_thr_tpr"] = np.mean([r["val_thr_tpr"] for r in rs]); m["val_thr_fpr"] = np.mean([r["val_thr_fpr"] for r in rs])
    m["val_thr_fpr_sd"] = float(np.std([r["val_thr_fpr"] for r in rs])) if len(rs) > 1 else np.nan
    m["val_thr_tpr_sd"] = float(np.std([r["val_thr_tpr"] for r in rs])) if len(rs) > 1 else np.nan
    m["n_folds"] = len(rs); m["P"] = P
    return m

KEYS = ["accuracy", "macro_f1", "roc_auc", "pr_auc", "fpr_at_tpr95", "fpr_at_tpr99", "val_thr_tpr", "val_thr_fpr"]
rows, POOL = [], {}
for proto in sorted(df.protocol.unique()):
    for model in MODELS + []:
        seeds = sorted(df[(df.protocol == proto) & (df.model == model)].seed.unique())
        ms = []
        for s in seeds:
            m = pooled(proto, model, s)
            if m: ms.append(m); POOL[(proto, model, s)] = m
        if not ms: continue
        row = dict(protocol=proto, model=model, n_seeds=len(ms), n_folds_per_seed=ms[0]["n_folds"])
        for k in KEYS:
            v = np.array([m[k] for m in ms]); row[k] = v.mean(); row[k + "_sd"] = v.std() if len(v) > 1 else np.nan
        rows.append(row)
S = pd.DataFrame(rows); S.to_csv(RES / "summary_by_protocol.csv", index=False)
print(S[["protocol", "model", "n_seeds", "accuracy", "macro_f1", "roc_auc", "fpr_at_tpr95", "val_thr_tpr", "val_thr_fpr"]].round(3).to_string())

# ---------- per-class table (P2 primary, seed 42) ------------------------------------
pc = []
for proto in ["P0_official", "P1_random_clean", "P2_grouped"]:
    for model in MODELS:
        m = POOL.get((proto, model, 42))
        if not m: continue
        for c in CLASSES:
            v = m["per_class"][c]; pc.append(dict(protocol=proto, model=model, attack_cat=c, **v))
PC = pd.DataFrame(pc); PC.to_csv(RES / "per_class_metrics.csv", index=False)

# ---------- figure 1: protocol inflation ---------------------------------------------------
protos = [("P0_official", "P0 official\n(raw split)"), ("P1_random_clean", "P1 random\n(deduplicated)"), ("P2_grouped", "P2 grouped\n(leakage-safe)")]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
for a, (key, title) in zip(ax, [("macro_f1", "Macro-F1 (10 classes)"), ("roc_auc", "Binary ROC-AUC"), ("fpr_at_tpr95", "False-positive rate at 95% detection")]):
    w = 0.26
    for i, m in enumerate(MODELS):
        vals, errs = [], []
        for p, _ in protos:
            r = S[(S.protocol == p) & (S.model == m)]
            vals.append(float(r[key].iloc[0]) if len(r) else np.nan); errs.append(float(r[key + "_sd"].iloc[0]) if len(r) else 0)
        a.bar(np.arange(3) + (i - 1) * w, vals, w, yerr=np.nan_to_num(errs), color=COL[m], label=LABEL[m], capsize=2, error_kw=dict(lw=0.8))
    a.set_xticks(range(3)); a.set_xticklabels([t for _, t in protos], fontsize=7.5); a.set_title(title, fontsize=9)
    if key == "roc_auc": a.set_ylim(0.5, 1.0)
ax[0].legend(fontsize=7, frameon=False, loc="upper right"); plt.tight_layout(); plt.savefig(FIG / "fig1_protocol_comparison.png"); plt.close()

# ---------- figure 2: leakage audit ------------------------------------------------------
L = pd.read_csv(RES / "leakage_audit.csv")
order = ["P0_official (raw)", "P1_random_clean", "P2_grouped_bins4", "P2_grouped", "P2_grouped_bins1"]
nice = {"P0_official (raw)": "P0 official", "P1_random_clean": "P1 random", "P2_grouped_bins4": "P2 fine\n(4 bins/oct)", "P2_grouped": "P2 primary\n(2 bins/oct)", "P2_grouped_bins1": "P2 coarse\n(1 bin/oct)"}
g = L.groupby("protocol").mean(numeric_only=True).reindex(order)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax[0].bar(range(len(g)), g.frac_near_twin * 100, color=["#8c8c8c", "#e0a458", "#7fb7be", "#1f6f8b", "#16425b"])
ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels([nice[o] for o in g.index], fontsize=6.5); ax[0].set_ylabel("% of test flows with a\nnear-twin in training"); ax[0].set_title("Near-duplicate leakage", fontsize=9)
ax[1].bar(range(len(g)), g.median_nn_dist, color=["#8c8c8c", "#e0a458", "#7fb7be", "#1f6f8b", "#16425b"])
ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([nice[o] for o in g.index], fontsize=6.5); ax[1].set_ylabel("median nearest-neighbour distance"); ax[1].set_title("Distance test flow -> training set", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "fig2_leakage_audit.png"); plt.close()

# ---------- figure 3: per-class recall P0 vs P1 vs P2 (random forest) ----------------------
fig, ax = plt.subplots(figsize=(7.4, 2.9)); w = 0.27
for i, (p, t) in enumerate(protos):
    sub = PC[(PC.protocol == p) & (PC.model == "random_forest")].set_index("attack_cat").reindex(CLASSES)
    ax.bar(np.arange(len(CLASSES)) + (i - 1) * w, sub.recall, w, label=t.replace("\n", " "), color=["#8c8c8c", "#e0a458", "#1f6f8b"][i])
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30, ha="right"); ax.set_ylabel("recall"); ax.set_title("Per-class recall, random forest", fontsize=9); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
plt.tight_layout(); plt.savefig(FIG / "fig3_per_class_recall_rf.png"); plt.close()

# ---------- confusion matrix (best model under P2 by macro-F1) ----------------------------
p2 = S[S.protocol == "P2_grouped"].sort_values("macro_f1", ascending=False)
best = p2.model.iloc[0]
cm = np.array(POOL[("P2_grouped", best, 42)]["confusion"], float); cmn = cm / cm.sum(1, keepdims=True).clip(1)
pd.DataFrame(cm.astype(int), index=CLASSES, columns=CLASSES).to_csv(RES / f"confusion_P2_{best}.csv")
fig, ax = plt.subplots(figsize=(5.2, 4.4)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7); ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=7)
for i in range(10):
    for j in range(10):
        if cmn[i, j] >= 0.02: ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > .5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{LABEL[best]}, P2 grouped: row-normalised (%)", fontsize=9); plt.tight_layout(); plt.savefig(FIG / "fig4_confusion_P2_best.png"); plt.close()

# ---------- ROC curves (pooled OOF, seed 42) -------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
for a, proto, t in zip(ax, ["P0_official", "P2_grouped"], ["P0 official split", "P2 grouped (leakage-safe)"]):
    for m in MODELS:
        P = POOL[(proto, m, 42)]["P"]; yb = (P.y_true != "Normal").astype(int)
        f, tp, _ = roc_curve(yb, P.score); a.plot(np.clip(f, 1e-4, 1), tp, color=COL[m], label=LABEL[m], lw=1.3)
    a.axhline(0.95, color="k", lw=0.5, ls=":"); a.set_xscale("log"); a.set_xlim(1e-3, 1); a.set_ylim(0.3, 1.0)
    a.set_xlabel("false-positive rate"); a.set_ylabel("detection rate (TPR)"); a.set_title(t, fontsize=9)
ax[0].legend(fontsize=7, frameon=False, loc="lower right"); plt.tight_layout(); plt.savefig(FIG / "fig5_roc.png"); plt.close()

# ---------- sensitivity to group granularity ----------------------------------------------
sens = [("P1_random_clean", "random"), ("P2_grouped_bins4", "grouped\n4 bins/oct"), ("P2_grouped", "grouped\n2 bins/oct"), ("P2_grouped_bins1", "grouped\n1 bin/oct")]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
for a, key, t in zip(ax, ["macro_f1", "fpr_at_tpr95"], ["Macro-F1", "FPR at 95% detection"]):
    for m in MODELS:
        v = [float(S[(S.protocol == p) & (S.model == m)][key].iloc[0]) if len(S[(S.protocol == p) & (S.model == m)]) else np.nan for p, _ in sens]
        a.plot(range(4), v, "-o", color=COL[m], label=LABEL[m], ms=3)
    a.set_xticks(range(4)); a.set_xticklabels([n for _, n in sens], fontsize=7); a.set_title(t, fontsize=9); a.set_xlabel("stricter near-duplicate grouping  ->")
ax[0].legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig(FIG / "fig6_sensitivity.png"); plt.close()

# ---------- ablation (random forest, P2) ------------------------------------------------
abl = [("P2_grouped", "all features"), ("P2_grouped_noTTL", "no TTL /\nct_state_ttl"), ("P2_grouped_noSEQ", "no TCP seq.\nnumbers"), ("P2_grouped_noCAT", "no proto /\nservice / state")]
rows = []
for p, n in abl:
    r = S[(S.protocol == p) & (S.model == "random_forest")]
    if len(r): rows.append(dict(variant=n.replace("\n", " "), protocol=p, **{k: float(r[k].iloc[0]) for k in ["accuracy", "macro_f1", "roc_auc", "fpr_at_tpr95", "val_thr_tpr", "val_thr_fpr"]}))
A = pd.DataFrame(rows); A.to_csv(RES / "ablation_rf_P2.csv", index=False); print(A.round(3).to_string())
if len(A) > 1:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for a, key, t in zip(ax, ["macro_f1", "fpr_at_tpr95"], ["Macro-F1", "FPR at 95% detection"]):
        a.bar(range(len(A)), A[key], color=["#1f6f8b"] + ["#9ec5d1"] * (len(A) - 1)); a.set_xticks(range(len(A))); a.set_xticklabels([n for _, n in abl][:len(A)], fontsize=6.5); a.set_title("RF ablation: " + t, fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig7_ablation.png"); plt.close()

# ---------- error analysis for the best model under P2 (seed 42) -----------------------
tr, te = load_raw(); clean, rep = clean_pooled(tr, te); c = clean.set_index("row_id")
P = POOL[("P2_grouped", best, 42)]["P"].copy(); P = P.join(c[["proto", "service", "state"]], on="row_id")
ea = {"best_model": best}
cmdf = pd.DataFrame(cm.astype(int), index=CLASSES, columns=CLASSES)
off = [(a, b, int(cmdf.loc[a, b]), float(cmdf.loc[a, b] / cmdf.loc[a].sum())) for a in CLASSES for b in CLASSES if a != b]
ea["top_confusions"] = sorted(off, key=lambda x: -x[2])[:8]
yb = (P.y_true != "Normal")
# error mass by true class
err = P[P.y_true != P.y_pred].y_true.value_counts()
ea["errors_by_true_class"] = err.to_dict(); ea["n_errors"] = int(err.sum()); ea["n_test"] = len(P)
# binary view at the 95%-detection oracle threshold
from idsbench.metrics import threshold_for_tpr
t = threshold_for_tpr(yb.astype(int).to_numpy(), P.score.to_numpy(), 0.95); P["alarm"] = P.score >= t
ea["missed_attacks_by_class_at_tpr95"] = (P[yb & ~P.alarm].y_true.value_counts() / P[yb].y_true.value_counts()).dropna().round(3).to_dict()
fp = P[~yb & P.alarm]; nrm = P[~yb]
ea["false_alarm_rate_by_service"] = {s: round(float(g.alarm.mean()), 3) for s, g in nrm.groupby("service") if len(g) >= 300}
ea["false_alarm_share_top_service"] = fp.service.value_counts(normalize=True).head(3).round(3).to_dict()
(RES / "error_analysis.json").write_text(json.dumps(ea, indent=1, default=str)); print(json.dumps(ea, indent=1, default=str))
