"""Stream evaluation: known-class balanced accuracy, known-vs-novel balanced accuracy, novel ARI."""

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)

METRIC_KEYS = ("known_balanced_accuracy", "known_vs_novel_balanced_accuracy", "novel_ari")


def _matched_balanced_accuracy(y_true, y_pred):
    """Balanced accuracy after one-to-one Hungarian matching of predicted to true labels."""
    classes, labels = np.unique(y_true), np.unique(y_pred)
    R = np.array([[np.mean(y_pred[y_true == c] == e) for e in labels] for c in classes])
    rows, cols = linear_sum_assignment(R, maximize=True)
    return float(R[rows, cols].sum() / len(classes))


def known_class_balanced_accuracy(y_true, y_pred, n_known, match_labels=False):
    """`match_labels` only for unsupervised methods (arbitrary cluster ids)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true < n_known
    if len(np.unique(y_true[mask])) < 2:
        return float("nan")
    if match_labels:
        return _matched_balanced_accuracy(y_true[mask], y_pred[mask])
    return float(balanced_accuracy_score(y_true[mask], y_pred[mask]))


def known_vs_novel_balanced_accuracy(y_true, y_pred, n_known, known_ids):
    true_known = np.asarray(y_true) < n_known
    if len(np.unique(true_known)) < 2:
        return float("nan")
    pred_known = np.isin(np.asarray(y_pred), list(known_ids))
    return float(balanced_accuracy_score(true_known.astype(int), pred_known.astype(int)))


def novel_ari(y_true, y_pred, n_known):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true >= n_known
    if len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(adjusted_rand_score(y_true[mask], y_pred[mask]))


def evaluate_stream(y_true, y_pred, n_known, known_ids, match_labels=False):
    """All three criteria on the online phase only; `known_ids` from the full run."""
    return dict(zip(METRIC_KEYS, (
        known_class_balanced_accuracy(y_true, y_pred, n_known, match_labels),
        known_vs_novel_balanced_accuracy(y_true, y_pred, n_known, known_ids),
        novel_ari(y_true, y_pred, n_known))))


def mask_undefined(res, undefined):
    """Sets NaN (in place) for criteria the method does not support ("crit1".."crit3")."""
    for crit in undefined:
        res[METRIC_KEYS[int(crit[-1]) - 1]] = float("nan")
    return res
