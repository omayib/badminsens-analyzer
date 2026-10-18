"""FIT parsing and feature engineering for playstyle analysis.

    python -m src.parser raw/*.fit            # print the feature vector of each session

One pass over a Garmin .fit file gives:
  * swings      one row per detected swing: time, game, stroke (badminsens overhead gate +
                classifier), peak wrist rotation / acceleration, estimated racket-head speed
  * meta        session totals the watch recorded (heart rate, training effect, calories)
                and the user's heart-rate profile (max / resting HR)
  * hr          heart-rate samples (minutes since start, bpm)
  * features    the session feature vector used by src.model (see FEATURES)

Swings are detected and classified with the shipped badminsens models (BADS overhead
strokes: smash, clear, drop, backhand clear; everything else is "Other").
"""
import datetime as dt
import hashlib
import math
import os
import pickle
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from badminsens.gate import OverheadGate  # noqa: E402
from badminsens.io import load_fit  # noqa: E402
from badminsens.predict import MODELS_DIR, STROKES, classify, detect, load_model, registry, set_balance  # noqa: E402

RACKET_M = 0.7          # wrist -> racket head distance for the speed estimate (same as badminsens.plot)
GYRO_SAT_DPS = 1990.0   # Forerunner 255 gyro clips at +/-2000 deg/s per axis
RALLY_GAP_S = 4.0       # consecutive swings closer than this belong to one rally
MIN_REST_MIN = 2.0      # an IMU gap this long is a rest between games (watch app paused)
CACHE_VERSION = 3

# stroke codes used in features and sequences
CODE = {"ForehandKill": "smash", "ForehandHigh": "clear", "ForehandLob": "drop",
        "BackhandTransition": "backhand", "": "other"}

# session feature vector, in the order src.model expects (name: description)
FEATURES = {
    "active_min": "minutes of play (IMU recording, rests excluded)",
    "swings_per_min": "swings per active minute",
    "overhead_share": "share of swings that are one of the 4 overhead strokes",
    "smash_share": "smashes / all swings",
    "clear_share": "clears / all swings",
    "drop_share": "drops / all swings",
    "backhand_share": "backhand clears / all swings",
    "attack_ratio": "smashes / overhead forehand strokes (smash + clear + drop)",
    "smash_p75_dps": "75th percentile peak wrist rotation of smashes (deg/s)",
    "backhand_med_dps": "median peak wrist rotation of backhand clears (deg/s)",
    "smash_drop_gap_dps": "median smash minus median drop peak wrist rotation (deg/s)",
    "swing_p90_kmh": "90th percentile estimated racket-head speed, all swings (km/h)",
    "swing_med_kmh": "median estimated racket-head speed, all swings (km/h)",
    "acc_p90_g": "90th percentile peak wrist acceleration (g)",
    "saturated_share": "share of swings that hit the gyro range limit (true speed higher)",
    "speed_cv": "median coefficient of variation of swing speed within each stroke type",
    "mean_confidence": "mean classifier confidence on overhead strokes",
    "rally_len": "mean swings per rally (swings < 4 s apart)",
    "shot_interval_s": "median time between own shots inside a rally (s)",
    "transition_entropy": "Markov transition entropy between consecutive stroke types (bits, 0..log2 5)",
    "speed_retention": "median swing speed in the last third of play / first third",
    "hr_avg_pct_hrr": "average heart rate as % of heart-rate reserve",
    "hr_high_share": "share of time above 80 % of max heart rate",
    "aerobic_te": "Garmin aerobic training effect (0-5)",
    "anaerobic_te": "Garmin anaerobic training effect (0-5)",
    "move_g": "RMS dynamic wrist acceleration between swings (g): footwork / movement",
}


def file_id(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_meta(path):
    """Session totals, user HR profile, HR samples and the UTC offset in one fitparse pass."""
    import fitparse

    meta, hr, offset = {}, [], None
    for m in fitparse.FitFile(path).get_messages(
            ["session", "user_profile", "zones_target", "record", "timestamp_correlation", "activity"]):
        v = m.get_values()
        if m.name == "record":
            if v.get("timestamp") is not None and v.get("heart_rate") is not None:
                hr.append((v["timestamp"], v["heart_rate"]))
        elif m.name == "session":
            for k in ("start_time", "total_timer_time", "total_elapsed_time", "total_calories", "avg_heart_rate",
                      "max_heart_rate", "total_training_effect", "total_anaerobic_training_effect",
                      "sport", "sub_sport"):
                if v.get(k) is not None:
                    meta[k] = v[k]
        elif m.name == "user_profile":
            for k in ("resting_heart_rate",):  # only the HR profile; no personal details are kept
                if v.get(k) is not None:
                    meta["user_" + k] = v[k]
        elif m.name == "zones_target":
            if v.get("max_heart_rate"):
                meta["user_max_heart_rate"] = v["max_heart_rate"]
        elif offset is None and v.get("local_timestamp") and v.get("timestamp"):
            if isinstance(v["local_timestamp"], dt.datetime):
                offset = round((v["local_timestamp"] - v["timestamp"]).total_seconds() / 3600, 2)
    meta["utc_offset_h"] = 7.0 if offset is None else offset
    return meta, hr


def game_spans(segments, min_rest_min=MIN_REST_MIN):
    spans = []
    for s in segments:
        a, b = s["t"][0] / 60, s["t"][-1] / 60
        if spans and a - spans[-1][1] < min_rest_min:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def _movement_g(segments, swing_times_s, guard_s=1.0):
    """RMS of |acc| - 1 g (low-passed) outside swing windows: footwork / body movement."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, 5.0, fs=100.0, output="sos")
    num = den = 0.0
    sw = np.sort(np.asarray(swing_times_s, float))
    for s in segments:
        if len(s["t"]) < 50:
            continue
        mag = np.linalg.norm(s["acc"], axis=1) / 9.80665
        dyn = sosfiltfilt(sos, mag - 1.0)
        t = s["t"]
        keep = np.ones(len(t), bool)
        if len(sw):
            j = np.searchsorted(sw, t)
            near = np.minimum(np.abs(t - sw[np.clip(j, 0, len(sw) - 1)]), np.abs(t - sw[np.clip(j - 1, 0, len(sw) - 1)]))
            keep = near > guard_s
        num += float((dyn[keep] ** 2).sum())
        den += int(keep.sum())
    return math.sqrt(num / den) if den else float("nan")


def _entropy_rate(seqs, k=5):
    """Markov-chain entropy rate (bits) of stroke-type sequences, with add-0.5 smoothing."""
    cats = list(dict.fromkeys(CODE.values()))
    T = np.full((k, k), 0.5)
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            T[cats.index(a), cats.index(b)] += 1
    rows = T.sum(1)
    P = T / rows[:, None]
    pi = rows / rows.sum()
    return float(-(pi[:, None] * P * np.log2(P)).sum())


def session_features(swings, meta, hr, games):
    """Feature vector (dict, keys = FEATURES) of one session or a slice of it."""
    n = len(swings)
    active = sum(b - a for a, b in games) or 1e-9
    codes = [s["code"] for s in swings]
    c = Counter(codes)
    kmh = np.array([s["racket_kmh"] for s in swings], float)
    dps = np.array([s["peak_gyro_dps"] for s in swings], float)
    acc = np.array([s["peak_acc_g"] for s in swings], float)
    t = np.array([s["time_s"] for s in swings], float)
    fwd = c["smash"] + c["clear"] + c["drop"]
    over = fwd + c["backhand"]
    of = lambda code: dps[[k == code for k in codes]] if n else np.array([])  # noqa: E731
    smash, drop, back = of("smash"), of("drop"), of("backhand")

    cvs = []
    for code in ("smash", "clear", "drop", "backhand", "other"):
        v = kmh[[k == code for k in codes]]
        if len(v) >= 5:
            cvs.append(v.std() / v.mean())

    # rallies: runs of swings < RALLY_GAP_S apart
    rallies, cur = [], []
    for i in range(n):
        if cur and t[i] - t[i - 1] > RALLY_GAP_S:
            rallies.append(cur)
            cur = []
        cur.append(i)
    if cur:
        rallies.append(cur)
    intervals = [t[r[j + 1]] - t[r[j]] for r in rallies for j in range(len(r) - 1)]

    # speed retention: last third of play vs first third (by swing order)
    third = n // 3
    retention = float(np.median(kmh[-third:]) / np.median(kmh[:third])) if third >= 10 else float("nan")

    hmax = meta.get("user_max_heart_rate") or 220 - 35
    hrest = meta.get("user_resting_heart_rate") or 60
    hv = np.array([h for _, h in hr], float)
    pct_hrr = float(np.mean((hv - hrest) / (hmax - hrest)) * 100) if len(hv) else float("nan")
    high = float(np.mean(hv >= 0.8 * hmax)) if len(hv) else float("nan")

    shares = {k: c[k] / n if n else 0.0 for k in ("smash", "clear", "drop", "backhand")}
    return {
        "active_min": active,
        "swings_per_min": n / active,
        "overhead_share": over / n if n else 0.0,
        "smash_share": shares["smash"],
        "clear_share": shares["clear"],
        "drop_share": shares["drop"],
        "backhand_share": shares["backhand"],
        "attack_ratio": c["smash"] / fwd if fwd else float("nan"),
        "smash_p75_dps": float(np.percentile(smash, 75)) if len(smash) >= 3 else float("nan"),
        "backhand_med_dps": float(np.median(back)) if len(back) >= 3 else float("nan"),
        "smash_drop_gap_dps": float(np.median(smash) - np.median(drop)) if len(smash) >= 3 and len(drop) >= 3
        else float("nan"),
        "swing_p90_kmh": float(np.percentile(kmh, 90)) if n else float("nan"),
        "swing_med_kmh": float(np.median(kmh)) if n else float("nan"),
        "acc_p90_g": float(np.percentile(acc, 90)) if n else float("nan"),
        "saturated_share": float(np.mean([s["saturated"] for s in swings])) if n else 0.0,
        "speed_cv": float(np.median(cvs)) if cvs else float("nan"),
        "mean_confidence": float(np.mean([s["confidence"] for s in swings if s["code"] != "other"] or [np.nan])),
        "rally_len": float(np.mean([len(r) for r in rallies])) if rallies else float("nan"),
        "shot_interval_s": float(np.median(intervals)) if intervals else float("nan"),
        "transition_entropy": _entropy_rate([[codes[i] for i in r] for r in rallies]),
        "speed_retention": retention,
        "hr_avg_pct_hrr": pct_hrr,
        "hr_high_share": high,
        "aerobic_te": float(meta.get("total_training_effect", float("nan"))),
        "anaerobic_te": float(meta.get("total_anaerobic_training_effect", float("nan"))),
        "move_g": meta.get("_move_g", float("nan")),
    }


def parse_fit(path, model="default", min_peak_dps=800.0, min_rest_min=MIN_REST_MIN, cache_dir=None,
              models_dir=MODELS_DIR):
    """Full analysis of one .fit file -> dict(meta, swings, hr, games, features, per_game)."""
    fid = file_id(path)
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"{fid}_{model}_{int(min_peak_dps)}_v{CACHE_VERSION}.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                return pickle.load(f)

    meta, hr_raw = _read_meta(path)
    segments = load_fit(path)
    games = game_spans(segments, min_rest_min)
    acc, gyr, info = detect(path, min_peak_dps, segments=segments)

    reg = registry(models_dir)
    if info:
        gate = OverheadGate.load(os.path.join(models_dir, reg["gate"]["file"]))
        m = load_model(model, models_dir)
        set_balance(m, reg.get("balance_iters", 0))
        rows = classify(m, acc, gyr, info, gate.accept(acc, gyr))
        sat = (np.abs(np.degrees(gyr[:, 80:110])) >= GYRO_SAT_DPS).any(axis=(1, 2))
    else:
        rows, sat = [], np.zeros(0, bool)

    mid = np.array([(a + b) / 2 for a, b in games]) if games else np.zeros(1)
    swings = []
    for r, s in zip(rows, sat):
        tm = r["time_s"] / 60
        g = next((i for i, (a, b) in enumerate(games) if a - 0.05 <= tm <= b + 0.05), int(np.abs(mid - tm).argmin()))
        code = CODE[r["label"] if r["gate"] == "accepted" else ""]
        swings.append({
            "time_s": r["time_s"], "clock": r["clock"], "game": g + 1, "code": code,
            "stroke": STROKES[r["label"]][0] if code != "other" else "Other",
            "confidence": r["confidence"] if r["confidence"] != "" else float("nan"),
            "peak_gyro_dps": r["peak_gyro_dps"], "peak_acc_g": r["peak_acc_g"],
            "racket_kmh": round(math.radians(r["peak_gyro_dps"]) * RACKET_M * 3.6, 1), "saturated": bool(s),
        })
    meta["_move_g"] = _movement_g(segments, [s["time_s"] for s in swings])

    t0 = hr_raw[0][0] if hr_raw else meta.get("start_time")
    start = meta.get("start_time") or t0
    hr = [((ts - start).total_seconds() / 60, h) for ts, h in hr_raw] if start else []
    local = start + dt.timedelta(hours=meta["utc_offset_h"]) if start else None

    feats = session_features(swings, meta, hr, games)
    per_game = []
    for i, (a, b) in enumerate(games):
        sw = [s for s in swings if s["game"] == i + 1]
        h = [(tm, v) for tm, v in hr if a <= tm <= b]
        if len(sw) >= 30:
            gm = {**meta, "total_training_effect": float("nan"), "total_anaerobic_training_effect": float("nan")}
            per_game.append({"game": i + 1, "start_min": a, "end_min": b,
                             "features": session_features(sw, gm, h, [(a, b)])})

    out = {
        "file_id": fid,
        "file_name": os.path.basename(path),
        "activity_id": os.path.basename(path).split("_")[0],
        "start_local": local.isoformat() if local else None,
        "model": model,
        "meta": {k: (v.isoformat() if isinstance(v, dt.datetime) else v) for k, v in meta.items()},
        "games": games,
        "swings": swings,
        "hr": hr,
        "features": feats,
        "per_game": per_game,
    }
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit", nargs="+")
    ap.add_argument("--model", default="default")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache", "sessions"))
    args = ap.parse_args()
    for p in args.fit:
        s = parse_fit(p, args.model, cache_dir=args.cache)
        print(f"\n{s['file_name']}  {s['start_local']}  {len(s['swings'])} swings, {len(s['games'])} games")
        for k, v in s["features"].items():
            print(f"  {k:20s} {v:9.3f}")


if __name__ == "__main__":
    main()
