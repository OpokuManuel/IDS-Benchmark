"""Evaluation beyond accuracy: per-class P/R, FPR at a fixed detection rate, etc."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve)
from .data import CLASSES


def fpr_at_tpr(y_bin, score, target_tpr):
    """Smallest FPR achievable at detection rate >= target (threshold picked on THESE labels)."""
    fpr, tpr, thr = roc_curve(y_bin, score)
    i = np.searchsorted(tpr, target_tpr, side="left")
    return float(fpr[min(i, len(fpr) - 1)])


def threshold_for_tpr(y_bin, score, target_tpr):
    fpr, tpr, thr = roc_curve(y_bin, score)
    i = np.searchsorted(tpr, target_tpr, side="left")
    return float(thr[min(i, len(thr) - 1)])


def transferred_operating_point(y_val, s_val, y_test, s_test, target_tpr=0.95):
    """Pick the threshold on validation data only, then measure it on test data.
    This is the honest operating point: no test labels were used to set it."""
    t = threshold_for_tpr(y_val, s_val, target_tpr)
    pred = s_test >= t
    pos, neg = y_test == 1, y_test == 0
    return float(pred[pos].mean()), float(pred[neg].mean())


def evaluate(y_true_cat, y_pred_cat, score, y_val=None, s_val=None):
    y_bin = (y_true_cat != "Normal").astype(int)
    p, r, f, s = precision_recall_fscore_support(y_true_cat, y_pred_cat, labels=CLASSES, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true_cat, y_pred_cat),
        "macro_f1": f1_score(y_true_cat, y_pred_cat, labels=CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_cat, y_pred_cat, labels=CLASSES, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_bin, score),
        "pr_auc": average_precision_score(y_bin, score),
        "fpr_at_tpr95": fpr_at_tpr(y_bin, score, 0.95),
        "fpr_at_tpr99": fpr_at_tpr(y_bin, score, 0.99),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion": confusion_matrix(y_true_cat, y_pred_cat, labels=CLASSES).tolist(),
    }
    if y_val is not None:
        tpr, fpr = transferred_operating_point((y_val != "Normal").astype(int), s_val, y_bin, score, 0.95)
        out.update(val_thr_tpr=tpr, val_thr_fpr=fpr)
    return out
