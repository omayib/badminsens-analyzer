"""Choose the shipped default model: leave-one-player-out over all 12 players.

    python -m badminsens.select                     # after tune + benchmark
    python -m badminsens.select --models svm gru cnn

The held-out test set has only 3 players (217 strokes) and the model ranking on it
differs from the CV ranking on the 9 dev players: player-to-player variation is
larger than the differences between models. After the test results are recorded,
this step evaluates the finalist configurations and their ensembles with
leave-one-player-out over all 12 players (4x more unseen players than the test set)
and makes the best one the default in models/registry.json.

Scores: accuracy on each unseen player with balanced session statistics and with
statistics from unbalanced subsets (skewed sessions), both averaged over players.
Out-of-fold probabilities are cached in experiments/select_oof.npz, so ensembles are
scored without retraining.
"""
import argparse
import itertools
import json
import os

import numpy as np

from . import dataset, metrics
from .models import Model
from .tune import EXP_DIR, _bests_from_log, skewed_refs

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def oof(cfg, d):
    """Out-of-fold probabilities for all 12 players: balanced + 3 skewed refs (balance 0 and 3)."""
    P = d["player"]
    players = sorted(set(P), key=lambda p: int(p[4:]))
    bal = np.zeros((len(P), 4))
    skew = {0: np.zeros((3, len(P), 4)), 3: np.zeros((3, len(P), 4))}
    for p in players:
        te = P == p
        m = Model(cfg).fit(d["acc"][~te], d["gyr"][~te], d["y"][~te], P[~te])
        A, G, Y, PP = d["acc"][te], d["gyr"][te], d["y"][te], P[te]
        m.cfg["balance_iters"] = 0
        bal[te] = m.predict_proba(A, G, PP)
        for k, r in enumerate(skewed_refs(Y, np.random.default_rng(int(p[4:])))):
            for it in skew:
                m.cfg["balance_iters"] = it
                skew[it][k, te] = m.predict_proba(A, G, PP, ref=(A[r], G[r], PP[r]))
        print(f"    {cfg['model']} fold {p} acc {(np.asarray(m.classes)[bal[te].argmax(1)] == Y).mean():.3f}", flush=True)
    return bal, skew[0], skew[3], m.classes


def score(bal, sk0, sk3, y, P, classes):
    cl = np.asarray(classes)
    per = {p: float((cl[bal[P == p].argmax(1)] == y[P == p]).mean()) for p in np.unique(P)}

    def skew_acc(sk):
        return float(np.mean([(cl[s[P == p].argmax(1)] == y[P == p]).mean() for s in sk for p in np.unique(P)]))

    r = metrics.evaluate(y, bal, classes)
    return {"accuracy": r["accuracy"], "player_mean": float(np.mean(list(per.values()))),
            "player_min": float(min(per.values())), "macro_f1": r["macro_f1"], "cohen_kappa": r["cohen_kappa"],
            "skew": skew_acc(sk0), "skew_bal": skew_acc(sk3), "per_player": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["svm", "rf", "et", "hgb", "mlp", "cnn", "gru", "transformer"])
    ap.add_argument("--norms", nargs="+", default=["session_center", "none"])
    ap.add_argument("--max-ensemble", type=int, default=3)
    ap.add_argument("--models-dir", default=MODELS_DIR)
    args = ap.parse_args()

    d = dataset.load()
    y, P = d["y"], d["player"]
    best = _bests_from_log(os.path.join(EXP_DIR, "tuning_log.jsonl"), args.norms)
    cache = os.path.join(EXP_DIR, "select_oof.npz")
    store = dict(np.load(cache, allow_pickle=True)) if os.path.exists(cache) else {}
    for name in args.models:
        if f"{name}_bal" in store:
            continue
        print(f"  {name}: 12-fold leave-one-player-out", flush=True)
        cfg = {k: v for k, v in best[name]["cfg"].items() if k not in ("seed", "n_jobs")}
        bal, s0, s3, classes = oof(cfg, d)
        store.update({f"{name}_bal": bal, f"{name}_s0": s0, f"{name}_s3": s3})
        np.savez(cache, **store)
    classes = sorted(set(y))

    results = {}
    for k in range(1, args.max_ensemble + 1):
        for combo in itertools.combinations(args.models, k):
            bal = np.mean([store[f"{c}_bal"] for c in combo], 0)
            s0 = np.mean([store[f"{c}_s0"] for c in combo], 0)
            s3 = np.mean([store[f"{c}_s3"] for c in combo], 0)
            results["+".join(combo)] = score(bal, s0, s3, y, P, classes)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["player_mean"])
    print(f"\n{'model / ensemble':28s} {'acc':>6s} {'player mean':>11s} {'worst player':>12s} {'F1':>6s} "
          f"{'skew':>6s} {'skew+bal':>8s}")
    for name, r in ranked[:15] + [kv for kv in ranked if "+" not in kv[0] and kv not in ranked[:15]]:
        print(f"{name:28s} {r['accuracy']:6.3f} {r['player_mean']:11.3f} {r['player_min']:12.3f} "
              f"{r['macro_f1']:6.3f} {r['skew']:6.3f} {r['skew_bal']:8.3f}")
    with open(os.path.join(EXP_DIR, "selection.json"), "w") as f:
        json.dump({"ranked": ranked}, f, indent=1)

    # default: best mean per-player accuracy; balance setting from the skewed-session scores
    win, r = ranked[0]
    reg_path = os.path.join(args.models_dir, "registry.json")
    reg = json.load(open(reg_path))
    members = win.split("+")
    missing = [m for m in members if m not in reg["models"]]
    if missing:
        raise SystemExit(f"shipped models missing for {missing}; run badminsens.benchmark first")
    if len(members) > 1:
        reg["models"]["selected_ensemble"] = {"family": "ensemble", "members": members,
                                              "loso12_accuracy": r["accuracy"]}
        reg["default"] = "selected_ensemble"
    else:
        reg["default"] = win
    reg["default_rule"] = ("best mean per-player accuracy, leave-one-player-out over all 12 players "
                           "(badminsens.select), after the held-out test evaluation")
    reg["balance_iters"] = 3 if r["skew_bal"] > r["skew"] else 0
    for name, rr in results.items():
        if name in reg["models"]:
            reg["models"][name]["loso12"] = {k: rr[k] for k in ("accuracy", "player_mean", "player_min", "skew", "skew_bal")}
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=1, default=str)
    print(f"\ndefault -> {reg['default']} ({win}), balance_iters={reg['balance_iters']}")


if __name__ == "__main__":
    main()
