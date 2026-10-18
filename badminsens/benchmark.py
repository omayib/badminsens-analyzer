"""Final benchmark of the tuned configurations, then ship them.

    python -m badminsens.tune         # 1. search configurations (CV on the 9 dev players)
    python -m badminsens.benchmark    # 2. evaluate once on the 3 test players, ship

For every family, the configuration with the best CV accuracy in experiments/tuning_log.jsonl
(restricted to the allowed session normalizations):
  * train on the dev players (train + val split), evaluate on the held-out test players
    (never used during tuning), mean +/- std over --seeds seeds, full metric set;
  * robustness: each test player also scored with session statistics from unbalanced
    subsets of their strokes, without and with class-balanced re-estimation;
  * ensemble = average of the 3 families with the best CV accuracy.
Ship: retrain every configuration on all 12 players + the overhead gate, write
models/registry.json (default = best CV accuracy, i.e. chosen without the test set)
and the report models/BENCHMARK.md.
"""
import argparse
import datetime
import json
import os
import time

import numpy as np

from . import dataset, metrics
from .gate import OverheadGate
from .models import Model, full_config
from .tune import EXP_DIR, _bests_from_log, skewed_refs

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
GATE_QUANTILE = 0.97
NN_TRAIN = ["dense", "dropout", "epochs", "batch", "optimizer", "lr", "weight_decay", "scheduler",
            "label_smoothing", "aug_session_rot", "aug_stroke_rot", "aug_noise"]
SEQ_IN = ["align", "norm", "seq_lowpass", "seq_decim"]
CFG_KEYS = {
    "svm": ["features", "align", "norm", "C", "gamma", "aug_copies", "aug_session_rot", "aug_stroke_rot"],
    "logreg": ["features", "align", "norm", "C", "aug_copies"],
    "rf": ["features", "align", "norm", "n_estimators", "max_features", "min_samples_leaf", "aug_copies"],
    "et": ["features", "align", "norm", "n_estimators", "max_features", "min_samples_leaf", "aug_copies"],
    "hgb": ["features", "align", "norm", "max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization",
            "aug_copies"],
    "mlp": ["features", "align", "norm"] + NN_TRAIN,
    "cnn": SEQ_IN + ["channels", "kernel"] + NN_TRAIN,
    "tcn": SEQ_IN + ["channels", "kernel"] + NN_TRAIN,
    "lstm": SEQ_IN + ["hidden", "layers", "bidirectional"] + NN_TRAIN,
    "gru": SEQ_IN + ["hidden", "layers", "bidirectional"] + NN_TRAIN,
    "cnn_lstm": SEQ_IN + ["channels", "kernel", "hidden", "layers", "bidirectional"] + NN_TRAIN,
    "transformer": SEQ_IN + ["d_model", "heads", "layers"] + NN_TRAIN,
}
# scikit-learn settings that fall back to these values when not in the config (models._sk_estimator)
SK_DEFAULTS = {"svm": {"C": 1.0, "gamma": "scale"}, "logreg": {"C": 0.1},
               "rf": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1},
               "et": {"n_estimators": 600, "max_features": "sqrt", "min_samples_leaf": 2},
               "hgb": {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 1.0}}


def evaluate_on_test(cfg, d, idx, seed):
    dev, te = idx["dev"], idx["test"]
    t = time.time()
    m = Model({**cfg, "seed": seed}).fit(d["acc"][dev], d["gyr"][dev], d["y"][dev], d["player"][dev])
    train_s = time.time() - t
    A, G, Y, P = d["acc"][te], d["gyr"][te], d["y"][te], d["player"][te]
    m.cfg["balance_iters"] = 0
    t = time.time()
    proba = m.predict_proba(A, G, P)
    infer_ms = (time.time() - t) / len(te) * 1000
    cl = np.asarray(m.classes)
    skew = {0: [], 3: []}
    for p in np.unique(P):
        k = P == p
        for r in skewed_refs(Y[k], np.random.default_rng(int(p[4:]))):
            ref = (A[k][r], G[k][r], P[k][r])
            for it in skew:
                m.cfg["balance_iters"] = it
                skew[it].append(float((cl[m.predict_proba(A[k], G[k], P[k], ref=ref).argmax(1)] == Y[k]).mean()))
    r = metrics.evaluate(Y, proba, m.classes)
    r.update(train_s=train_s, infer_ms_per_stroke=infer_ms, skewed_session_accuracy=float(np.mean(skew[0])),
             skewed_session_accuracy_balanced=float(np.mean(skew[3])),
             per_player={p: float((cl[proba[P == p].argmax(1)] == Y[P == p]).mean()) for p in np.unique(P)})
    return r, proba, m.classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="BADS_CLL_OPENACCESS_V1")
    ap.add_argument("--log", default=os.path.join(EXP_DIR, "tuning_log.jsonl"),
                    help="tuning log; the best allowed configuration per family is benchmarked")
    ap.add_argument("--norms", nargs="+", default=["session_center", "none"],
                    choices=["session_center", "none", "session_z"])
    ap.add_argument("--families", nargs="+", help="subset of families (default: all tuned)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=MODELS_DIR)
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-render BENCHMARK.md from benchmark.json")
    ap.add_argument("--update", action="store_true",
                    help="merge the given --families into the existing benchmark.json / registry.json")
    args = ap.parse_args()

    if args.report_only:
        _markdown(json.load(open(os.path.join(args.out, "benchmark.json"))), os.path.join(args.out, "BENCHMARK.md"))
        return
    best = _bests_from_log(args.log, args.norms)
    fams = [f for f in (args.families or best) if f in best]
    d = dataset.load(args.data)
    s = dataset.split_indices(d)
    idx = {"dev": np.r_[s["train"], s["val"]], "test": s["test"]}
    Y = d["y"][idx["test"]]
    print(f"dev {len(idx['dev'])} strokes (9 players) / test {len(idx['test'])} strokes "
          f"({', '.join(dataset.SPLIT['test'])})", flush=True)

    rows, test_proba = [], {}
    for fam in fams:
        cfg = {k: v for k, v in best[fam]["cfg"].items() if k not in ("seed", "n_jobs")}
        runs = []
        for seed in range(args.seeds):
            r, proba, classes = evaluate_on_test(cfg, d, idx, seed)
            runs.append(r)
            if seed == 0:
                test_proba[fam] = proba
            print(f"  {fam:11s} seed {seed}: test acc {r['accuracy']:.3f}  f1 {r['macro_f1']:.3f}  "
                  f"skew {r['skewed_session_accuracy']:.3f}/{r['skewed_session_accuracy_balanced']:.3f}", flush=True)
        rows.append(_row(fam, cfg, best[fam], runs))

    old_bench = None
    if args.update and os.path.exists(os.path.join(args.out, "benchmark.json")):
        old_bench = json.load(open(os.path.join(args.out, "benchmark.json")))
        rows += [r for r in old_bench["rows"] if r["model"] not in fams and r["family"] != "ensemble"]
    all_fams = sorted({r["model"] for r in rows if r["family"] != "ensemble"} | set(fams))
    top = sorted(all_fams, key=lambda f: -best[f]["accuracy"])[:3]
    fresh_ens = len(top) >= 2 and all(f in test_proba for f in top)
    if old_bench and not fresh_ens:  # keep the recorded ensemble row
        rows += [r for r in old_bench["rows"] if r["family"] == "ensemble"]
    if fresh_ens:
        pr = np.mean([test_proba[f] for f in top], 0)
        r = metrics.evaluate(Y, pr, classes)
        rows.append({"model": "ensemble", "family": "ensemble", "cfg": {"members": top},
                     "cv_accuracy": float(np.mean([best[f]["accuracy"] for f in top])),
                     "test": {k: [r[k], 0.0] for k in metrics.SUMMARY_KEYS},
                     "test_seed0": r, "skew": None, "skew_bal": None, "train_s": None, "infer_ms": None})
        print(f"  ensemble {top}: test acc {r['accuracy']:.3f}  f1 {r['macro_f1']:.3f}")
    rows.sort(key=lambda r: -r["test"]["accuracy"][0])

    os.makedirs(args.out, exist_ok=True)
    tuning = _tuning_summary()
    bench = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "split": dataset.SPLIT,
             "seeds": args.seeds, "classes": classes, "rows": rows, "tuning": tuning}
    with open(os.path.join(args.out, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=1, default=str)
    _markdown(bench, os.path.join(args.out, "BENCHMARK.md"))
    print(f"\n{'model':12s} {'CV acc':>7s} {'test acc':>15s} {'macro F1':>9s} {'kappa':>6s} {'skew':>6s} {'skew+bal':>8s}")
    for r in rows:
        t = r["test"]
        print(f"{r['model']:12s} {r['cv_accuracy']:7.3f} {t['accuracy'][0]:.3f} ± {t['accuracy'][1]:.3f}   "
              f"{t['macro_f1'][0]:8.3f} {t['cohen_kappa'][0]:6.3f} "
              f"{r['skew'] if r['skew'] is None else round(r['skew'], 3)!s:>6s} "
              f"{r['skew_bal'] if r['skew_bal'] is None else round(r['skew_bal'], 3)!s:>8s}")

    if args.no_ship:
        return
    print("\nShipping: retraining on all 12 players ...", flush=True)
    if not args.update:
        gate = OverheadGate(GATE_QUANTILE).fit(d["acc"], d["gyr"], d["y"])
        gate.save(os.path.join(args.out, "gate.joblib"))
    # real sessions have an unbalanced stroke mix: ship class-balanced session statistics
    # when they help on the skewed CV sessions
    bal = [r for r in rows if r["family"] != "ensemble"]
    use_bal = np.mean([best[r["model"]]["skewed_session_accuracy_balanced"] for r in bal]) > \
        np.mean([best[r["model"]]["skewed_session_accuracy"] for r in bal])
    default = max(fams, key=lambda f: best[f]["accuracy"])
    reg_path = os.path.join(args.out, "registry.json")
    if args.update and os.path.exists(reg_path):
        reg = json.load(open(reg_path))
        for r in rows:
            if r["model"] in fams:
                m = Model(r["cfg"]).fit(d["acc"], d["gyr"], d["y"], d["player"])
                m.save(os.path.join(args.out, f"{r['model']}.joblib"))
                reg["models"][r["model"]].update(config=m.cfg, test_accuracy=r["test"]["accuracy"][0],
                                                  test_macro_f1=r["test"]["macro_f1"][0])
                print(f"  saved {r['model']}.joblib", flush=True)
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=1, default=str)
        return
    reg = {"created": bench["created"], "trained_on": "all 12 BADS players", "classes": classes,
           "default": default, "default_rule": "best leave-one-player-out CV accuracy (test set not used)",
           "balance_iters": 3 if use_bal else 0,
           "gate": {"file": "gate.joblib", "quantile": GATE_QUANTILE}, "models": {}}
    for r in rows:
        if r["family"] == "ensemble":
            reg["models"]["ensemble"] = {"family": "ensemble", "members": r["cfg"]["members"],
                                         "cv_accuracy": r["cv_accuracy"], "test_accuracy": r["test"]["accuracy"][0]}
            continue
        m = Model(r["cfg"]).fit(d["acc"], d["gyr"], d["y"], d["player"])
        fn = f"{r['model']}.joblib"
        m.save(os.path.join(args.out, fn))
        reg["models"][r["model"]] = {"file": fn, "family": m.family, "config": m.cfg,
                                     "cv_accuracy": r["cv_accuracy"], "test_accuracy": r["test"]["accuracy"][0],
                                     "test_macro_f1": r["test"]["macro_f1"][0]}
        print(f"  saved {fn}", flush=True)
    with open(os.path.join(args.out, "registry.json"), "w") as f:
        json.dump(reg, f, indent=1, default=str)
    print(f"default model: {default} (balance_iters={reg['balance_iters']}) -> {args.out}/registry.json")


def _row(fam, cfg, cv, runs):
    agg = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in metrics.SUMMARY_KEYS}
    return {"model": fam, "family": Model(cfg).family, "cfg": cfg, "cv_accuracy": cv["accuracy"],
            "cv": {k: cv[k] for k in metrics.SUMMARY_KEYS + ["skewed_session_accuracy",
                                                              "skewed_session_accuracy_balanced"]},
            "test": agg, "test_seed0": runs[0],
            "skew": float(np.mean([r["skewed_session_accuracy"] for r in runs])),
            "skew_bal": float(np.mean([r["skewed_session_accuracy_balanced"] for r in runs])),
            "train_s": float(np.mean([r["train_s"] for r in runs])),
            "infer_ms": float(np.mean([r["infer_ms_per_stroke"] for r in runs]))}


def _tuning_summary():
    path = os.path.join(EXP_DIR, "tuning_log.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        r = json.loads(line)
        o = out.setdefault(r["family"], {"trials": 0, "best": 0.0, "worst": 1.0, "seconds": 0.0})
        o["trials"] += 1
        o["best"], o["worst"] = max(o["best"], r["accuracy"]), min(o["worst"], r["accuracy"])
        o["seconds"] += r["seconds"]
    return out


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def _markdown(b, path):
    cl = b["classes"]
    names = {"ForehandKill": "smash", "ForehandHigh": "clear", "ForehandLob": "drop", "BackhandTransition": "backhand clear"}
    L = [f"# Stroke classifier benchmark ({b['created'][:10]})", "",
         f"- Tuning: leave-one-player-out CV on 9 dev players ({', '.join(b['split']['train'] + b['split']['val'])}).",
         f"- Test: {', '.join(b['split']['test'])}, never used during tuning; mean ± std over {b['seeds']} seeds.",
         "- *skew*: test accuracy when session statistics come from an unbalanced subset of the player's strokes; "
         "*skew+bal*: same with class-balanced re-estimation.", "",
         "## Results (test players)", "",
         "| model | CV acc | test acc | bal. acc | macro P | macro R | macro F1 | Cohen κ | MCC | log-loss | ROC-AUC | skew | skew+bal | train s | ms/stroke |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        t = r["test"]
        f = lambda k: f"{t[k][0]:.3f}" if t[k][0] is not None else ""  # noqa: E731
        L.append(f"| {r['model']} | {r['cv_accuracy']:.3f} | {t['accuracy'][0]:.3f} ± {t['accuracy'][1]:.3f} | "
                 f"{f('balanced_accuracy')} | {f('macro_precision')} | {f('macro_recall')} | {f('macro_f1')} | "
                 f"{f('cohen_kappa')} | {f('mcc')} | {f('log_loss')} | {f('roc_auc_ovr')} | "
                 f"{'' if r['skew'] is None else format(r['skew'], '.3f')} | "
                 f"{'' if r['skew_bal'] is None else format(r['skew_bal'], '.3f')} | "
                 f"{'' if r['train_s'] is None else format(r['train_s'], '.1f')} | "
                 f"{'' if r['infer_ms'] is None else format(r['infer_ms'], '.2f')} |")
    L += ["", "## Per-class precision / recall / F1 (test, seed 0)", "",
          "| model | " + " | ".join(names[c] for c in cl) + " |", "|---|" + "---|" * len(cl)]
    for r in b["rows"]:
        pc = r["test_seed0"]["per_class"]
        L.append(f"| {r['model']} | " + " | ".join(
            f"{pc[c]['precision']:.2f} / {pc[c]['recall']:.2f} / {pc[c]['f1']:.2f}" for c in cl) + " |")
    L += ["", "## Configurations", ""]
    for r in b["rows"]:
        if r["family"] == "ensemble":
            L.append(f"**ensemble**: average of {', '.join(r['cfg']['members'])}")
            L.append("")
            continue
        cfg = {**full_config({}), **SK_DEFAULTS.get(r["model"], {}), **r["cfg"]}
        if r["model"] == "cnn_lstm":
            cfg["channels"] = cfg["channels"][:2]
        keys = CFG_KEYS[r["model"]]
        L.append(f"**{r['model']}** ({r['family']})")
        L.append("")
        L.append("| " + " | ".join(keys) + " |")
        L.append("|" + "---|" * len(keys))
        L.append("| " + " | ".join(_fmt(cfg[k]) for k in keys) + " |")
        L.append("")
    if b["tuning"]:
        L += ["## Tuning loop", "", "| family | trials | worst CV acc | best CV acc | CPU time |", "|---|---|---|---|---|"]
        for fam, t in sorted(b["tuning"].items(), key=lambda kv: -kv[1]["best"]):
            L.append(f"| {fam} | {t['trials']} | {t['worst']:.3f} | {t['best']:.3f} | {t['seconds'] / 60:.0f} min |")
        L.append("")
    sel = os.path.join(EXP_DIR, "selection.json")
    if os.path.exists(sel):
        ranked = json.load(open(sel))["ranked"]
        singles = [kv for kv in ranked if "+" not in kv[0]]
        L += ["## Default-model selection: leave-one-player-out over all 12 players", "",
              "Run after the test evaluation above (`python -m badminsens.select`). Per-player accuracy "
              "averaged over the 12 players; ensembles average the members' probabilities.", "",
              "| model / ensemble | accuracy | player mean | worst player | macro F1 | skew | skew+bal |",
              "|---|---|---|---|---|---|---|"]
        for name, r in ranked[:10] + [kv for kv in singles if kv not in ranked[:10]]:
            L.append(f"| {name} | {r['accuracy']:.3f} | {r['player_mean']:.3f} | {r['player_min']:.3f} | "
                     f"{r['macro_f1']:.3f} | {r['skew']:.3f} | {r['skew_bal']:.3f} |")
        L.append("")
    L += ["## Confusion matrices (test, seed 0; rows = true, columns = predicted)", "",
          "Order: " + ", ".join(names[c] for c in cl), ""]
    for r in b["rows"]:
        L += [f"**{r['model']}**", "```"] + [" ".join(f"{v:4d}" for v in row) for row in r["test_seed0"]["confusion"]] + ["```", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
