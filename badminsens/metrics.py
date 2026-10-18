"""Classification metrics for stroke models."""
import numpy as np
from sklearn import metrics as skm


def evaluate(y, proba, classes):
    """y: true labels, proba: (N, K) class probabilities in `classes` order."""
    classes = list(classes)
    proba = np.clip(np.asarray(proba, float), 1e-7, 1)
    proba = proba / proba.sum(1, keepdims=True)
    pred = np.asarray(classes)[proba.argmax(1)]
    yi = np.searchsorted(classes, y)
    p, r, f, s = skm.precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(skm.accuracy_score(y, pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y, pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_f1": float(skm.f1_score(y, pred, average="weighted")),
        "cohen_kappa": float(skm.cohen_kappa_score(y, pred)),
        "mcc": float(skm.matthews_corrcoef(y, pred)),
        "log_loss": float(skm.log_loss(yi, proba, labels=range(len(classes)))),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion": skm.confusion_matrix(y, pred, labels=classes).tolist(),
    }
    try:
        out["roc_auc_ovr"] = float(skm.roc_auc_score(yi, proba, multi_class="ovr", labels=range(len(classes))))
    except ValueError:
        out["roc_auc_ovr"] = None
    return out


SUMMARY_KEYS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                "cohen_kappa", "mcc", "log_loss", "roc_auc_ovr"]
