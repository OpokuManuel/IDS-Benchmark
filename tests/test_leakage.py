"""Guards for the claims the benchmark makes. Run: pytest -q"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from idsbench.data import load_raw, clean_pooled, feature_cols
from idsbench.splits import make_groups, p1_folds, p2_folds, val_split


@pytest.fixture(scope="module")
def data():
    tr, te = load_raw()
    clean, rep = clean_pooled(tr, te)
    return tr, te, clean, rep


def test_official_row_counts(data):
    tr, te, _, _ = data
    assert (len(tr), len(te)) == (175_341, 82_332)       # documented by UNSW


def test_clean_has_no_duplicate_vectors_or_binary_conflicts(data):
    tr, _, clean, _ = data
    assert not clean.duplicated(feature_cols(tr)).any()
    assert set(clean.label) == {0, 1}
    assert ((clean.attack_cat != "Normal").astype(int) == clean.label).all()


@pytest.mark.parametrize("bins", [1, 2, 4])
@pytest.mark.parametrize("seed", [42, 43])
def test_grouped_folds_share_no_group_and_cover_everything(data, bins, seed):
    _, _, clean, _ = data
    g, y = make_groups(clean, bins), clean.attack_cat.to_numpy()
    seen = []
    for tri, tei in p2_folds(y, g, 5, seed):
        assert not (set(g[tri]) & set(g[tei]))
        assert not (set(tri) & set(tei))
        seen.append(tei)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(len(clean)))   # each row tested exactly once


def test_validation_split_stays_inside_train_and_is_group_disjoint(data):
    _, _, clean, _ = data
    g, y = make_groups(clean, 2), clean.attack_cat.to_numpy()
    tri, tei = next(p2_folds(y, g, 5, 42))
    a, b = val_split(tri, y, g, True, 42)
    assert set(a) | set(b) == set(tri) and not (set(a) & set(b))
    assert not (set(g[a]) & set(g[b])) and not (set(b) & set(tei))


def test_groups_never_use_labels_and_are_deterministic(data):
    _, _, clean, _ = data
    g1, g2 = make_groups(clean, 2), make_groups(clean.drop(columns=["attack_cat", "label"]), 2)
    assert np.array_equal(g1, g2)


def test_random_protocol_is_reproducible(data):
    _, _, clean, _ = data
    y = clean.attack_cat.to_numpy()
    f1 = [t.tolist() for _, t in p1_folds(y, 5, 42)]
    f2 = [t.tolist() for _, t in p1_folds(y, 5, 42)]
    assert f1 == f2
