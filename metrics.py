"""Step 5: evaluation metrics written from scratch. Positive class = churn = 1."""
import numpy as np


def _div(a, b):
    return float(a / b) if b else 0.0


def confusion_matrix(y, pred):
    """[[TN, FP], [FN, TP]] - rows = actual, columns = predicted."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return np.array([[np.sum((y == 0) & (pred == 0)), np.sum((y == 0) & (pred == 1))],
                     [np.sum((y == 1) & (pred == 0)), np.sum((y == 1) & (pred == 1))]])


def report(y, pred):
    (tn, fp), (fn, tp) = confusion_matrix(y, pred)
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": _div(tp + tn, tp + tn + fp + fn), "precision": precision, "recall": recall,
            "specificity": _div(tn, tn + fp), "f1": _div(2 * precision * recall, precision + recall)}


def roc_curve(y, scores):
    """Sweep the threshold from high to low; each distinct score gives one (FPR, TPR) point."""
    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]       # last row of each group of tied scores
    return np.r_[0.0, fps[last] / (1 - y).sum()], np.r_[0.0, tps[last] / y.sum()]


def auc(x, y):
    """Trapezoid rule: sum of (x[i+1]-x[i]) * (y[i]+y[i+1]) / 2."""
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def roc_auc_score(y, scores):
    return auc(*roc_curve(y, scores))


def threshold_sweep(y, proba):
    return [{**report(y, proba >= t), "threshold": float(t)} for t in np.round(np.arange(0.02, 0.99, 0.01), 2)]


def best_f1_threshold(y, proba):
    """Threshold with the highest F1 (call this on VALIDATION data only)."""
    return max(threshold_sweep(y, proba), key=lambda r: (r["f1"], -r["threshold"]))["threshold"]


def threshold_for_min_recall(y, proba, min_recall):
    """Constraint 'catch at least X% of churners': the HIGHEST threshold (fewest false alarms) meeting it."""
    ok = [r for r in threshold_sweep(y, proba) if r["recall"] >= min_recall]
    return max(r["threshold"] for r in ok) if ok else None
