"""Count and classify badminton strokes in a Garmin .fit file with a shipped model.

    python -m badminsens.predict ACTIVITY.fit                 # default model (best CV accuracy)
    python -m badminsens.predict ACTIVITY.fit --model svm     # any shipped model / ensemble
    python -m badminsens.predict ACTIVITY.fit --model all     # compare all shipped models
    python -m badminsens.predict ACTIVITY.fit --no-gate       # force every swing into a class

Pipeline: swing detection -> overhead gate (is this one of the dataset's 4 overhead
strokes at all? otherwise "Other") -> session normalization over the accepted swings ->
classifier. Writes <fit>_strokes.csv (one row per swing) and <fit>_summary.json.
"""
import argparse
import csv
import datetime
import json
import os
from collections import Counter

import numpy as np

from . import models
from .detect import cut_windows, detect_strokes
from .gate import OverheadGate
from .io import load_fit

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OTHER = "Other (not a dataset stroke)"

# BADS label -> stroke name (BadminSense paper) and side. All 4 are overhead strokes.
STROKES = {
    "ForehandKill": ("Smash", "forehand"),
    "ForehandHigh": ("Clear", "forehand"),
    "ForehandLob": ("Drop", "forehand"),
    "BackhandTransition": ("Backhand clear", "backhand"),
}


def registry(models_dir=MODELS_DIR):
    with open(os.path.join(models_dir, "registry.json")) as f:
        return json.load(f)


def load_model(name_or_path="default", models_dir=MODELS_DIR):
    if os.path.isfile(name_or_path):
        return models.load(name_or_path)
    reg = registry(models_dir)
    name = reg["default"] if name_or_path == "default" else name_or_path
    if name not in reg["models"]:
        raise SystemExit(f"unknown model '{name}'; shipped: {', '.join(reg['models'])}")
    entry = reg["models"][name]
    if entry["family"] == "ensemble":
        return models.EnsembleModel([load_model(m, models_dir) for m in entry["members"]])
    return models.load(os.path.join(models_dir, entry["file"]))


def detect(fit_path, min_peak_dps=800.0, segments=None):
    """All swing windows of a .fit file: (acc, gyr, info rows).

    segments: the file's load_fit() output, if already loaded (avoids parsing it twice)."""
    accs, gyrs, info = [], [], []
    for seg in load_fit(fit_path) if segments is None else segments:
        peaks = detect_strokes(seg, min_peak_dps=min_peak_dps)
        if len(peaks) == 0:
            continue
        a, g = cut_windows(seg, peaks)
        accs.append(a)
        gyrs.append(g)
        for p in peaks:
            t = float(seg["t"][p])
            info.append({
                "time_s": round(t, 2),
                "clock": (seg["t0"] + datetime.timedelta(seconds=t)).strftime("%H:%M:%S.%f")[:-4]
                if seg["t0"] else "",
                "peak_gyro_dps": round(float(np.degrees(np.linalg.norm(seg["gyr"][p]))), 0),
                "peak_acc_g": round(float(np.linalg.norm(seg["acc"][p - 10:p + 10], axis=1).max() / 9.80665), 2),
            })
    if not info:
        return np.zeros((0, 200, 3)), np.zeros((0, 200, 3)), []
    return np.concatenate(accs), np.concatenate(gyrs), info


def set_balance(model, iters):
    for m in getattr(model, "members", [model]):
        m.cfg["balance_iters"] = iters


def classify(model, acc, gyr, info, accepted, min_conf=0.5):
    """Classify the accepted swings (one session); the rest are 'Other'."""
    rows = [{**base, "stroke": OTHER, "best_guess": "", "label": "", "confidence": "",
             "gate": "rejected", **{f"p_{c}": "" for c in model.classes}} for base in info]
    idx = np.flatnonzero(accepted)
    if len(idx):
        proba = model.predict_proba(acc[idx], gyr[idx])
        for i, pr in zip(idx, proba):
            k = int(pr.argmax())
            c = model.classes[k]
            rows[i].update(stroke=STROKES[c][0] if pr[k] >= min_conf else "Uncertain",
                           best_guess=STROKES[c][0], label=c, confidence=round(float(pr[k]), 3), gate="accepted",
                           **{f"p_{cl}": round(float(v), 3) for cl, v in zip(model.classes, pr)})
    return rows


def summarize(rows):
    n = len(rows)
    acc = [r for r in rows if r["gate"] == "accepted"]
    k = len(acc)
    pct = lambda x, tot: round(100 * x / tot, 1) if tot else 0.0  # noqa: E731
    best = Counter(r["label"] for r in acc)
    conf = Counter(r["stroke"] for r in acc)
    side = Counter(STROKES[c][1] for c in best.elements())
    return {
        "total_swings": n,
        "dataset_strokes": k,
        "other_swings": n - k,
        "strokes": {STROKES[c][0]: {"count": best[c], "percent_of_dataset_strokes": pct(best[c], k),
                                    "percent_of_all_swings": pct(best[c], n), "confident": conf[STROKES[c][0]]}
                    for c in STROKES},
        "other": {"count": n - k, "percent_of_all_swings": pct(n - k, n)},
        "sides": {s: {"count": side[s], "percent_of_dataset_strokes": pct(side[s], k)} for s in ("forehand", "backhand")},
        "uncertain": conf["Uncertain"],
        "mean_confidence": round(float(np.mean([r["confidence"] for r in acc])), 3) if k else None,
    }


def print_summary(s, title):
    n, k = s["total_swings"], s["dataset_strokes"]
    print(f"\n{title}: {n} swings, {k} classified as dataset strokes, {s['other_swings']} other")
    print(f"  {'stroke':28s}{'count':>6s}  {'% of strokes':>12s}  {'% of swings':>11s}")
    for name, v in s["strokes"].items():
        print(f"  {name:28s}{v['count']:6d}  {v['percent_of_dataset_strokes']:11.1f}%  {v['percent_of_all_swings']:10.1f}%")
    print(f"  {OTHER:28s}{s['other']['count']:6d}  {'':12s}  {s['other']['percent_of_all_swings']:10.1f}%")
    f, b = s["sides"]["forehand"], s["sides"]["backhand"]
    print(f"  {'-' * 62}")
    print(f"  forehand {f['count']} ({f['percent_of_dataset_strokes']}%) / backhand {b['count']} "
          f"({b['percent_of_dataset_strokes']}%) of the dataset strokes")
    print(f"  uncertain (confidence < threshold): {s['uncertain']}   mean confidence {s['mean_confidence']}")


def analyze(fit_path, model="default", models_dir=MODELS_DIR, min_peak_dps=800.0, min_conf=0.5,
            gate_quantile=None, balance=None):
    """One call for notebooks: detect -> gate -> classify. Returns (rows, summary)."""
    reg = registry(models_dir)
    acc, gyr, info = detect(fit_path, min_peak_dps)
    if not info:
        return [], summarize([])
    gate = OverheadGate.load(os.path.join(models_dir, reg["gate"]["file"]))
    if gate_quantile:
        gate.set_quantile(gate_quantile)
    m = load_model(model, models_dir)
    set_balance(m, reg.get("balance_iters", 0) if balance is None else balance)
    rows = classify(m, acc, gyr, info, gate.accept(acc, gyr), min_conf)
    return rows, summarize(rows)


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit", help="Garmin .fit activity file with accelerometer/gyroscope data")
    ap.add_argument("--model", default="default",
                    help="shipped model name (see models/registry.json), 'ensemble', 'all', or a model file")
    ap.add_argument("--models-dir", default=MODELS_DIR)
    ap.add_argument("--out", help="CSV output path (default: <fit name>_strokes.csv next to the .fit)")
    ap.add_argument("--min-peak-dps", type=float, default=800.0,
                    help="gyro peak needed to count a swing (lower = more, weaker swings)")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="below this probability a stroke is counted as 'Uncertain'")
    ap.add_argument("--gate-quantile", type=float,
                    help="overhead gate strictness: share of dataset strokes it accepts (default from registry)")
    ap.add_argument("--no-gate", action="store_true", help="classify every swing (forces non-dataset strokes)")
    ap.add_argument("--balance", type=int,
                    help="class-balanced session statistics iterations (default from registry)")
    args = ap.parse_args()

    reg = registry(args.models_dir)
    out = args.out or os.path.splitext(args.fit)[0] + "_strokes.csv"
    acc, gyr, info = detect(args.fit, args.min_peak_dps)
    print(f"{len(info)} swings detected in {os.path.basename(args.fit)}")
    if not info:
        return
    gate = OverheadGate.load(os.path.join(args.models_dir, reg["gate"]["file"]))
    if args.gate_quantile:
        gate.set_quantile(args.gate_quantile)
    accepted = np.ones(len(info), bool) if args.no_gate else gate.accept(acc, gyr)
    balance = reg.get("balance_iters", 0) if args.balance is None else args.balance

    names = ([reg["default"]] + [n for n in reg["models"] if n != reg["default"]]) if args.model == "all" \
        else [reg["default"] if args.model == "default" else args.model]
    results = {}
    for n in names:
        m = load_model(n, args.models_dir)
        set_balance(m, balance)
        results[n] = classify(m, acc, gyr, info, accepted, args.min_conf)
    summaries = {n: summarize(r) for n, r in results.items()}
    main_name = names[0]
    rows = results[main_name]
    if len(names) > 1:
        ref = [r["label"] for r in rows]
        print(f"\n{'model':18s} {'CV acc':>7s} {'test acc':>8s} {'12-player':>9s} "
              + " ".join(f"{v[0][:14]:>14s}" for v in STROKES.values()) + f"  agree w/ {main_name}")
        for n, r in results.items():
            s = summaries[n]
            agree = np.mean([a["label"] == b for a, b in zip(r, ref) if a["gate"] == "accepted"]) if any(accepted) else 0
            s["agreement_with_default"] = round(float(agree), 3)
            e = reg["models"][n]
            l12 = e.get("loso12", {}).get("accuracy", e.get("loso12_accuracy", float("nan")))
            fmt = lambda v, w: f"{v:{w}.3f}" if v == v else f"{'-':>{w}s}"  # noqa: E731
            print(f"{n:18s} {fmt(e.get('cv_accuracy', float('nan')), 7)} {fmt(e.get('test_accuracy', float('nan')), 8)} "
                  f"{fmt(l12, 9)} "
                  + " ".join(f"{s['strokes'][v[0]]['percent_of_dataset_strokes']:13.1f}%" for v in STROKES.values())
                  + f"  {agree:.1%}")
        rows = [{**row, **{f"pred_{n}": results[n][i]["label"] for n in names}} for i, row in enumerate(rows)]
    print_summary(summaries[main_name], f"model {main_name}")
    _write_csv(rows, out)
    summary = {"file": os.path.basename(args.fit), "gate_quantile": None if args.no_gate else gate.quantile,
               "balance_iters": balance}
    summary.update({"model": main_name, **summaries[main_name]} if len(names) == 1 else {"models": summaries})
    with open(os.path.splitext(out)[0].removesuffix("_strokes") + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nper-swing details: {out}")


if __name__ == "__main__":
    main()
