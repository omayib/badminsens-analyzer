"""Configuration search loop for the stroke classifiers.

    python -m badminsens.tune                                  # all families, target 90 %
    python -m badminsens.tune --families svm cnn --trials 20 --target 0.9

Every configuration is scored by leave-one-player-out cross-validation on the 9
development players (train + val split). The 3 test players are never used here:
they are reserved for the final benchmark (badminsens.benchmark), so the reported
test accuracy is not tuned on.

Per family: trial 0 = default configuration, then random configurations, then
mutations of the best one so far. A family stops when its CV accuracy reaches
--target or its budget is used. If no family reached the target, a refinement
round keeps mutating the best families (--refine trials each).

Every trial (config + metrics) is appended to experiments/tuning_log.jsonl and the
best configuration per family is written to experiments/best_configs.json.
"""
import argparse
import datetime
import json
import os
import random
import time

import numpy as np

from . import dataset, metrics
from .models import ALL_MODELS, FEATURE_MODELS, full_config

EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")

NN_COMMON = {
    "epochs": [30, 60, 100], "batch": [16, 32, 64], "optimizer": ["adamw", "adam", "sgd"],
    "lr": [5e-4, 1e-3, 2e-3, 5e-3], "weight_decay": [1e-4, 1e-3, 1e-2, 5e-2],
    "scheduler": ["cosine", "onecycle", "none"], "label_smoothing": [0.0, 0.1, 0.2],
    "dropout": [0.1, 0.3, 0.5], "dense": [[], [64], [128], [128, 64]],
    "norm": ["session_z", "session_center", "none"],
    "aug_session_rot": [0.0, 15.0, 30.0], "aug_stroke_rot": [0.0, 8.0, 15.0], "aug_noise": [0.0, 0.02],
}
SGD_LR = [1e-2, 3e-2, 5e-2]
# sequence nets: 25 Hz input and <= 60 epochs keep a CV trial affordable on CPU
SEQ = {"seq_lowpass": [12.0, 20.0], "seq_decim": [4], "epochs": [30, 60]}
SEQ_START = {"seq_decim": 4, "epochs": 40}

SPACES = {
    "svm": {"features": ["paper", "ours", "both"], "norm": ["session_z", "session_center", "none"],
            "C": [0.3, 1.0, 3.0, 10.0], "gamma": ["scale", 3e-4, 1e-3, 3e-3], "aug_copies": [0, 2, 4],
            "aug_session_rot": [0.0, 15.0, 30.0]},
    "logreg": {"features": ["paper", "ours", "both"], "norm": ["session_z", "session_center", "none"],
               "C": [0.003, 0.01, 0.03, 0.1, 0.3], "aug_copies": [0, 2, 4]},
    "rf": {"features": ["paper", "ours", "both"], "norm": ["session_z", "session_center", "none"],
           "max_features": ["sqrt", 0.1, 0.3], "min_samples_leaf": [1, 2, 4], "aug_copies": [0, 2, 4]},
    "et": {"features": ["paper", "ours", "both"], "norm": ["session_z", "session_center", "none"],
           "max_features": ["sqrt", 0.1, 0.3], "min_samples_leaf": [1, 2, 4], "aug_copies": [0, 2, 4]},
    "hgb": {"features": ["paper", "ours", "both"], "norm": ["session_z", "session_center", "none"],
            "learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [7, 15, 31], "l2_regularization": [0.0, 1.0],
            "aug_copies": [0, 2, 4]},
    "mlp": {**NN_COMMON, "features": ["paper", "ours", "both"],
            "dense": [[128], [256, 128], [512, 128], [256, 128, 64]]},
    "cnn": {**NN_COMMON, **SEQ, "channels": [[32, 64], [32, 64, 128], [64, 128, 128], [32, 64, 128, 128]],
            "kernel": [5, 7, 9]},
    "tcn": {**NN_COMMON, **SEQ, "channels": [[32, 32, 64, 64], [64, 64, 64, 64], [32, 64, 64]], "kernel": [3, 5, 7]},
    "lstm": {**NN_COMMON, **SEQ, "hidden": [32, 64, 128], "layers": [1, 2], "bidirectional": [True, False]},
    "gru": {**NN_COMMON, **SEQ, "hidden": [32, 64, 128], "layers": [1, 2], "bidirectional": [True, False]},
    "cnn_lstm": {**NN_COMMON, **SEQ, "channels": [[32, 64], [64, 128]], "hidden": [32, 64, 128], "layers": [1, 2]},
    "transformer": {**NN_COMMON, **SEQ, "d_model": [32, 64, 96], "heads": [2, 4], "layers": [1, 2, 3]},
}
# trial budget per family (slow recurrent nets get fewer)
BUDGET = {"svm": 16, "logreg": 6, "rf": 8, "et": 8, "hgb": 8, "mlp": 12, "cnn": 14, "tcn": 14,
          "lstm": 8, "gru": 8, "cnn_lstm": 10, "transformer": 8}


def _fix(cfg):
    if cfg.get("optimizer") == "sgd" and cfg.get("lr", 1) < 1e-2:
        cfg["lr"] = 3e-2
    if cfg.get("optimizer") != "sgd" and cfg.get("lr", 0) >= 1e-2:
        cfg["lr"] = 2e-3
    return cfg


def sample(family, rng):
    cfg = {"model": family}
    for k, v in SPACES[family].items():
        cfg[k] = rng.choice(v)
    if cfg.get("optimizer") == "sgd":
        cfg["lr"] = rng.choice(SGD_LR)
    return _fix(cfg)


def mutate(cfg, family, rng, n=2):
    new = dict(cfg)
    for k in rng.sample(sorted(SPACES[family]), n):
        choices = [v for v in SPACES[family][k] if v != cfg.get(k)] or SPACES[family][k]
        new[k] = rng.choice(choices)
    if new.get("optimizer") == "sgd" and new.get("lr", 1) < 1e-2:
        new["lr"] = rng.choice(SGD_LR)
    return _fix(new)


# --------------------------------------------------------------------------- cross-validation

_DATA = None


def skewed_refs(y, rng, n_refs=3, size=40):
    """Index sets simulating sessions with an unbalanced stroke mix (Dirichlet(1) proportions)."""
    classes = sorted(set(y))
    refs = []
    for _ in range(n_refs):
        p = rng.dirichlet(np.ones(len(classes)))
        cls = rng.choice(len(classes), size=size, p=p)
        refs.append(np.array([rng.choice(np.flatnonzero(y == classes[c])) for c in cls]))
    return refs


def _fold(cfg, test_player, threads=1):
    global _DATA
    import torch

    torch.set_num_threads(threads)
    if _DATA is None:
        _DATA = dataset.load()
    from .models import Model

    d = _DATA
    dev = np.isin(d["player"], dataset.SPLIT["train"] + dataset.SPLIT["val"])
    tr, te = dev & (d["player"] != test_player), d["player"] == test_player
    m = Model({**cfg, "n_jobs": threads}).fit(d["acc"][tr], d["gyr"][tr], d["y"][tr], d["player"][tr])
    A, G, Y, P = d["acc"][te], d["gyr"][te], d["y"][te], d["player"][te]
    cl = np.asarray(m.classes)
    m.cfg["balance_iters"] = 0
    proba = m.predict_proba(A, G, P)
    # same player, session statistics from unbalanced subsets of their strokes,
    # without (skew) and with (skew_bal) class-balanced re-estimation at inference
    skew, skew_bal = [], []
    for r in skewed_refs(Y, np.random.default_rng(int(test_player[4:]))):
        ref = (A[r], G[r], P[r])
        for iters, acc_list in ((0, skew), (3, skew_bal)):
            m.cfg["balance_iters"] = iters
            acc_list.append(float((cl[m.predict_proba(A, G, P, ref=ref).argmax(1)] == Y).mean()))
        if m.cfg["norm"] == "none":
            break
    return np.flatnonzero(te), proba, m.classes, float(np.mean(skew)), float(np.mean(skew_bal))


def cross_validate(cfg, d, jobs):
    from joblib import Parallel, delayed

    players = dataset.SPLIT["train"] + dataset.SPLIT["val"]
    if jobs == 1:  # in-process: the networks use the GPU, no extra processes / memory
        out = [_fold(cfg, p, threads=6) for p in players]
    else:
        out = Parallel(n_jobs=jobs, backend="loky")(delayed(_fold)(cfg, p, max(1, 6 // jobs)) for p in players)
    classes = out[0][2]
    idx = np.concatenate([o[0] for o in out])
    proba = np.concatenate([o[1] for o in out])
    y = d["y"][idx]
    r = metrics.evaluate(y, proba, classes)
    per = {p: float((np.asarray(classes)[o[1].argmax(1)] == d["y"][o[0]]).mean()) for p, o in zip(players, out)}
    r["per_player"] = per
    r["player_acc_std"] = float(np.std(list(per.values())))
    r["skewed_session_accuracy"] = float(np.mean([o[3] for o in out]))
    r["skewed_session_accuracy_balanced"] = float(np.mean([o[4] for o in out]))
    return r


# --------------------------------------------------------------------------- loop

def run_family(family, d, args, log, rng, best=None, budget=None, phase="search", warm=None):
    budget = budget or BUDGET[family]
    tried = set()
    for t in range(budget):
        if best is None:
            cfg = {"model": family, **(SEQ_START if "seq_decim" in SPACES[family] else {})}
        elif t == 1 and warm is not None and phase == "search":
            # warm start: the best configuration of an earlier tuning run
            cfg = {k: v for k, v in warm["cfg"].items() if k not in ("align", "n_jobs")}
            if cfg.get("norm") not in SPACES[family]["norm"]:
                cfg["norm"] = SPACES[family]["norm"][0]
        elif t < budget // 2 and phase == "search":
            cfg = sample(family, rng)
        else:
            cfg = mutate(best["cfg"], family, rng)
        key = json.dumps(full_config(cfg), sort_keys=True, default=str)
        if key in tried:
            cfg = mutate(cfg, family, rng, 1)
            key = json.dumps(full_config(cfg), sort_keys=True, default=str)
        tried.add(key)
        t0 = time.time()
        try:
            r = cross_validate(cfg, d, args.jobs)
        except Exception as e:  # a bad config must not kill the loop
            print(f"  [{family} {phase} {t}] failed: {e!r}", flush=True)
            continue
        rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "family": family, "phase": phase,
               "trial": t, "cfg": full_config(cfg), "seconds": round(time.time() - t0, 1),
               **{k: r[k] for k in metrics.SUMMARY_KEYS}, "per_player": r["per_player"],
               "player_acc_std": r["player_acc_std"], "skewed_session_accuracy": r["skewed_session_accuracy"],
               "skewed_session_accuracy_balanced": r["skewed_session_accuracy_balanced"],
               "per_class": r["per_class"], "confusion": r["confusion"]}
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()
        is_best = best is None or r["accuracy"] > best["accuracy"]
        if is_best:
            best = rec
        print(f"  [{family:11s} {phase} {t:2d}] cv acc {r['accuracy']:.3f}  skew {r['skewed_session_accuracy']:.3f}/{r['skewed_session_accuracy_balanced']:.3f}  "
              f"f1 {r['macro_f1']:.3f}  "
              f"kappa {r['cohen_kappa']:.3f}  {rec['seconds']:6.1f}s {'*' if is_best else ' '} "
              f"{_short(cfg)}", flush=True)
        if best["accuracy"] >= args.target:
            print(f"  -> {family} reached target {args.target:.2f} (cv acc {best['accuracy']:.3f})", flush=True)
            break
    return best


def _short(cfg):
    return {k: v for k, v in cfg.items() if k != "model"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", choices=ALL_MODELS,
                    default=["svm", "logreg", "rf", "et", "hgb", "mlp", "cnn", "tcn", "cnn_lstm", "transformer",
                             "gru", "lstm"])
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--trials", type=int, help="override the per-family budget")
    ap.add_argument("--refine", type=int, default=10, help="extra trials per top family if target not reached")
    ap.add_argument("--norms", nargs="+", default=["session_center", "none"],
                    choices=["session_center", "none", "session_z"],
                    help="session normalizations allowed in the search (session_z is fragile across devices)")
    ap.add_argument("--warm-start", help="tuning log of an earlier run; its best config per family is trial 1")
    ap.add_argument("--resume", action="store_true",
                    help="continue from each family's best logged config (mutations only)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel CV worker processes (1 = in-process, uses the GPU; >1 needs ~0.5 GB RAM each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=EXP_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for space in SPACES.values():
        space["norm"] = list(args.norms)
    d = dataset.load()
    rng = random.Random(args.seed)
    best_path = os.path.join(args.out, "best_configs.json")
    bests = _bests_from_log(os.path.join(args.out, "tuning_log.jsonl"), args.norms)
    warm = _bests_from_log(args.warm_start, args.norms) if args.warm_start else {}
    print(f"CV: leave-one-player-out on {len(dataset.SPLIT['train'] + dataset.SPLIT['val'])} dev players, "
          f"target accuracy {args.target:.2f}", flush=True)
    with open(os.path.join(args.out, "tuning_log.jsonl"), "a") as log:
        for fam in args.families:
            if args.resume and fam in bests:
                b = run_family(fam, d, args, log, rng, best=bests[fam], budget=args.trials, phase="refine")
            else:
                b = run_family(fam, d, args, log, rng, budget=args.trials, warm=warm.get(fam))
            if b and (fam not in bests or b["accuracy"] > bests[fam]["accuracy"]):
                bests[fam] = b
            _save(bests, best_path)
        reached = [f for f in args.families if bests.get(f, {}).get("accuracy", 0) >= args.target]
        if not reached and args.refine:
            top = sorted(args.families, key=lambda f: -bests.get(f, {}).get("accuracy", 0))[:3]
            print(f"\nTarget not reached; refining {top}", flush=True)
            for fam in top:
                b = run_family(fam, d, args, log, rng, best=bests[fam], budget=args.refine, phase="refine")
                if b["accuracy"] > bests[fam]["accuracy"]:
                    bests[fam] = b
                _save(bests, best_path)

    print("\nBest per family (leave-one-player-out CV on dev players):")
    for fam, b in sorted(bests.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {fam:11s} acc {b['accuracy']:.3f}  skew {b['skewed_session_accuracy']:.3f}  bal-acc {b['balanced_accuracy']:.3f}  f1 {b['macro_f1']:.3f}  "
              f"kappa {b['cohen_kappa']:.3f}  mcc {b['mcc']:.3f}  auc {b['roc_auc_ovr']:.3f}")


def _bests_from_log(path, norms=None):
    bests = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if norms and r["cfg"].get("norm") not in norms:
                continue
            if r["family"] not in bests or r["accuracy"] > bests[r["family"]]["accuracy"]:
                bests[r["family"]] = r
    return bests


def _save(bests, path):
    with open(path, "w") as f:
        json.dump(bests, f, indent=1, default=str)


if __name__ == "__main__":
    main()
