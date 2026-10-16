"""Small, fixed hyper-parameter sweep using ONLY an inner validation split of one TRAIN fold.
No test data is touched. The chosen configs are then frozen in src/idsbench/models.py."""
import sys, time, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from idsbench import SEED
from idsbench.data import ROOT, load_raw, clean_pooled, feature_cols, CLASSES
from idsbench.features import make_preprocessor
from idsbench.splits import make_groups, p2_folds, val_split

tr, te = load_raw(); clean, _ = clean_pooled(tr, te); f = feature_cols(tr)
g, y = make_groups(clean, 2), clean.attack_cat.to_numpy()
tri, _ = next(p2_folds(y, g, 5, SEED))                       # outer fold 0: only its TRAIN part is used
a, b = val_split(tri, y, g, True, SEED, frac=0.2)            # grouped inner validation
fit, val = clean.iloc[a], clean.iloc[b]
print(len(fit), len(val), val.attack_cat.value_counts().to_dict(), flush=True)
grid = {
 "logreg": [("C=0.1", lambda: LogisticRegression(C=0.1, max_iter=500)), ("C=1", lambda: LogisticRegression(C=1.0, max_iter=500)), ("C=10", lambda: LogisticRegression(C=10.0, max_iter=500))],
 "random_forest": [("n=100,leaf=1", lambda: RandomForestClassifier(100, random_state=SEED)), ("n=100,leaf=3", lambda: RandomForestClassifier(100, min_samples_leaf=3, random_state=SEED)),
                   ("n=100,leaf=1,bal", lambda: RandomForestClassifier(100, class_weight="balanced_subsample", random_state=SEED))],
 "grad_boosting": [("it=100,lr=.1", lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=SEED)),
                   ("it=300,lr=.1", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=SEED)),
                   ("it=300,lr=.05,leaf15,l2=1", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=SEED))],
}
out = []
for name, cands in grid.items():
    for label, mk in cands:
        pre = make_preprocessor(fit[f], scale=(name == "logreg"))
        m = Pipeline([("pre", pre), ("clf", mk())]); t = time.time(); m.fit(fit[f], fit.attack_cat)
        P = m.predict_proba(val[f]); cl = list(m.named_steps["clf"].classes_); pred = np.array(cl)[P.argmax(1)]
        yb = (val.attack_cat != "Normal").astype(int); s = 1 - P[:, cl.index("Normal")]
        r = dict(model=name, config=label, val_macro_f1=round(f1_score(val.attack_cat, pred, labels=CLASSES, average="macro", zero_division=0), 4),
                 val_acc=round(float((pred == val.attack_cat).mean()), 4), val_auc=round(roc_auc_score(yb, s), 4), secs=round(time.time() - t))
        print(r, flush=True); out.append(r)
(ROOT / "results" / "tuning_inner_validation.json").write_text(json.dumps(out, indent=1))
