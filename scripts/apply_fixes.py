"""One-off corrections after the full re-run. Safe to run twice (already-applied edits are skipped)."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = "Emmanuel Atta Opoku"

def read(p): return (ROOT / p).read_text(encoding="utf-8")
def write(p, s): (ROOT / p).write_text(s, encoding="utf-8", newline="\n")

def sub(path, old, new):
    s = read(path)
    if new in s:
        print(f"  skipped (already applied): {path}: {old[:50]!r}"); return
    assert s.count(old) == 1, f"{path}: expected exactly 1 match, found {s.count(old)} for: {old[:70]!r}"
    write(path, s.replace(old, new)); print(f"  fixed: {path}: {old[:50]!r}")

S = pd.read_csv(ROOT / "results" / "summary_by_protocol.csv").set_index(["protocol", "model"])
dq = pd.read_csv(ROOT / "results" / "data_quality.csv").set_index("attack_cat")
tr = pd.read_csv(ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv", encoding="utf-8-sig")
u = tr[(tr.proto == "udp") & (tr.service == "dns")]
generic, attack = (u.attack_cat.str.strip() == "Generic").mean() * 100, (u.label == 1).mean() * 100
p1, p2 = S.loc[("P1_random_clean", "grad_boosting")], S.loc[("P2_grouped", "grad_boosting")]
dup_generic = dq.loc["Generic", "duplicate_share"] * 100

print("README.md")
R = "README.md"
sub(R, "**Author:** Opoku Emmanuel Atta", f"**Author:** {NAME}")
sub(R, "All seeds fixed (42).", "Seeds are fixed (42, plus 43 and 44 for the repeats of the grouped protocol).")
sub(R, "## 3. Results (pooled out-of-fold predictions; every flow is tested once)",
    "## 3. Results (pooled out-of-fold predictions; every flow is tested once; P2 rows are means over three grouping seeds)")
sub(R, "costs gradient boosting 0.27 macro-F1 and 0.06 FPR at 95% detection",
    f"costs gradient boosting {p1.macro_f1 - p2.macro_f1:.2f} macro-F1 and {p2.fpr_at_tpr95 - p1.fpr_at_tpr95:.2f} FPR at 95% detection")
sub(R, "gave 95.3% detection at **19.4%** FPR for gradient boosting under P2",
    f"gave {p2.val_thr_tpr * 100:.1f}% detection at **{p2.val_thr_fpr * 100:.1f}%** FPR for gradient boosting under P2 (mean over three seeds)")
sub(R, "Generic being 88% duplicates", f"Generic being {dup_generic:.0f}% duplicates")
sub(R, "The random forest ablation shows",
    "The results are reproducible with `scripts/04_dns_shortcut_check.py`. The random forest ablation shows")
sub(R, "2. **One grouping seed.** All P2 numbers use fold assignment seed 42; I did not complete repeats with other seeds (`scripts/02_run_experiments.py --stage seeds` is provided but its results are not in this report). Fold-level spread is reported instead.",
    "2. **Three grouping seeds.** The P2 results in the main table are means over fold-assignment seeds 42, 43 and 44 (each pooled over five folds); the spread across seeds is in the `*_sd` columns of `results/summary_by_protocol.csv`. P0, P1, the ablation and the sensitivity analysis, and the per-class tables, confusion matrix and ROC curves, use seed 42 only.")

s = read(R).split("\n")
head = next(i for i, l in enumerate(s) if l.startswith("| Protocol | Model | Accuracy"))
end = head + 2
while end < len(s) and s[end].startswith("|"): end += 1
labels = {"P0_official": "P0 official", "P1_random_clean": "P1 random", "P2_grouped": "**P2 grouped**"}
models = {"logreg": "Logistic reg.", "random_forest": "Random forest", "grad_boosting": "Grad. boosting"}
rows = [f"| {labels[p]} | {models[m]} | {S.loc[(p, m), 'accuracy']:.3f} | {S.loc[(p, m), 'macro_f1']:.3f} | {S.loc[(p, m), 'roc_auc']:.3f} | "
        f"{S.loc[(p, m), 'fpr_at_tpr95']:.3f} | {S.loc[(p, m), 'fpr_at_tpr99']:.3f} |" for p in labels for m in models]
s[head + 2:end] = rows; write(R, "\n".join(s)); print("  rebuilt: results table in README.md")

print("report/build_report.py")
B = "report/build_report.py"
sub(B, '--author "Your Name" """', f'--author "{NAME}" """')
sub(B, 'default="Opoku Emmanuel Atta"', f'default="{NAME}"')
sub(B, "(pooled out-of-fold predictions, seed 42)", "(pooled out-of-fold predictions; P2 rows are means over three seeds)")
sub(B, "while Generic (0.07), Reconnaissance (0.15) and DoS (0.25) collapse, although they score 0.97, 0.81 and 0.08 under P0. ",
    "while Generic (0.07) and Reconnaissance (0.15) collapse, although they score 0.97 and 0.81 under the official split (which uses different cleaning); DoS is weak in both (0.08 official, 0.25 grouped). ")
sub(B, "All P2 numbers use one fold-assignment seed (fold-level spread is reported instead; AUC ranges 0.59-0.96 for logistic regression). ",
    "The P2 results are means over three fold-assignment seeds; the AUC of logistic regression varies widely between folds (0.59 to 0.96). ")
t = read(B)
pat = re.compile(r"A spurious correlation is visible directly: 93%.*?Exploits\.", re.S)
if pat.search(t):
    t = pat.sub(f"A possible shortcut is visible in the data: {attack:.0f}% of UDP/DNS flows in the official training file are attacks ({generic:.0f}% Generic), so a model can learn that DNS means attack.", t, count=1)
    write(B, t); print("  fixed: DNS shortcut sentence")
else:
    print("  skipped (already applied): DNS shortcut sentence")

print("requirements.txt")
req = read("requirements.txt")
if "pypdf" not in req:
    write("requirements.txt", req.rstrip("\n") + "\npypdf==5.9.0\n"); print("  added: pypdf==5.9.0")
else:
    print("  skipped (already present): pypdf")
print("\nDone. Next: pip install pypdf==5.9.0, python report/build_report.py, pytest -q, then git diff")