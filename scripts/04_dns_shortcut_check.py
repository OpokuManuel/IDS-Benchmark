"""Reproducible check of the 'DNS means attack' shortcut (prints numbers; writes nothing)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import pandas as pd
from idsbench import SEED
from idsbench.data import load_raw, clean_pooled, feature_cols
from idsbench.models import build_model
from idsbench.splits import make_groups, p2_folds

tr, te = load_raw()
for name, d in (("official train", tr), ("official test", te)):
    u = d[(d.proto == "udp") & (d.service == "dns")]
    print(f"{name}: {len(u):,} UDP/DNS flows | Generic {(u.attack_cat == 'Generic').mean():.1%} | "
          f"any attack {(u.label == 1).mean():.1%} | Normal {(u.label == 0).mean():.1%}")

clean, _ = clean_pooled(tr, te)
feats = feature_cols(tr)
y = clean.attack_cat.to_numpy()
groups = make_groups(clean, 2)
tri, tei = next(p2_folds(y, groups, 5, SEED))
big = pd.Series(groups[tei]).value_counts().index[0]
sub = clean.iloc[tei[groups[tei] == big]]
print(f"\nFold 0, largest test group: {len(sub):,} flows; true classes: {sub.attack_cat.value_counts().to_dict()}")
print("most common proto / service / state / sttl:", sub.proto.mode()[0], sub.service.mode()[0], sub.state.mode()[0], sub.sttl.mode()[0])
train = clean.iloc[tri]
tu = train[(train.proto == "udp") & (train.service == "dns")]
print(f"UDP/DNS flows in this fold's training data: {len(tu):,} ({tu.attack_cat.value_counts().to_dict()})")
for m in ("logreg", "random_forest"):
    model = build_model(m, train[feats], SEED)
    model.fit(train[feats], train["attack_cat"])
    print(f"{m}: predictions on that group -> {pd.Series(model.predict(sub[feats])).value_counts().to_dict()}")