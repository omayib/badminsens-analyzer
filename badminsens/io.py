"""Loaders for Garmin .fit IMU data and the BADS dataset.

Both loaders return data on a uniform 100 Hz grid in SI units:
accelerometer in m/s^2 (gravity included), gyroscope in rad/s.
"""
import glob
import json
import os

import numpy as np

FS = 100.0  # common sample rate (Hz)
G = 9.80665

# Garmin Forerunner 255 raw IMU scaling (determined empirically, see README):
#   accel: 1 mg per LSB, +/-8 g range  (saturates at +/-8192). Garmin reports
#          the opposite sign to Android (reads -1 g on z when lying face up),
#          so the sign is flipped to match the BADS/Android convention.
#   gyro : 16.384 LSB per deg/s, +/-2000 deg/s range (saturates at +/-32767)
GARMIN_ACC_SCALE = -G / 1000.0
GARMIN_GYRO_SCALE = np.radians(1.0 / 16.384)

# Garmin sensor frame -> BADS (Android smartwatch) sensor frame, applied as
# v_bads = GARMIN_TO_BADS @ v_garmin (gyro also multiplied by det for mirrors).
# A search over all 48 signed axis permutations, matching swing statistics
# against BADS, showed the axes coincide once the accel sign is flipped.
GARMIN_TO_BADS = np.eye(3)

# The IMU messages' timestamp + sample_time_offset run 1024/1000 too fast: their "ms" are
# really 1/1024 s ticks (the 8-tick spacing is 128 Hz, not 125 Hz), and the timestamp base
# is minutes off the activity clock. Fitted on the watch's timer start/stop events (one per
# app resume/pause), the rate 1000/1024 matches every block boundary to within 1 s in both
# recorded sessions; the offset is then taken from the timer events.
GARMIN_IMU_CLOCK = 1000.0 / 1024.0


def _resample(t_ms, X, t_new_ms):
    return np.stack([np.interp(t_new_ms, t_ms, X[:, i]) for i in range(X.shape[1])], 1)


def load_fit(path, max_gap_ms=100, transform=None):
    """Read accelerometer_data / gyroscope_data messages from a Garmin .fit file.

    Returns a list of continuous segments (gaps > max_gap_ms split segments).
    Each segment is a dict with t (seconds since activity start, absolute
    datetime of start in 't0'), acc (N,3) m/s^2 and gyr (N,3) rad/s in the
    BADS frame. IMU times are corrected onto the activity clock (see
    GARMIN_IMU_CLOCK), so they line up with heart rate and the timer events.
    """
    import datetime as dt

    import fitparse

    ff = fitparse.FitFile(path)
    rows = {"accelerometer_data": [], "gyroscope_data": []}
    timer, first_record = [], None
    for m in ff.get_messages([*rows, "event", "record"]):
        v = m.get_values()
        if m.name == "event":
            if v.get("event") == "timer" and v.get("timestamp") is not None:
                timer.append((v["timestamp"], v.get("event_type")))
            continue
        if m.name == "record":
            if first_record is None and v.get("timestamp") is not None:
                first_record = v["timestamp"]
            continue
        if v.get("timestamp") is None or v.get("sample_time_offset") is None:
            continue
        base = v["timestamp"].timestamp() * 1000.0 + (v.get("timestamp_ms") or 0)
        k = "accel" if m.name == "accelerometer_data" else "gyro"
        xs, ys, zs = (v.get(f"calibrated_{k}_{a}") for a in "xyz")
        if xs is None:
            continue
        for i, off in enumerate(v["sample_time_offset"]):
            if off is None or xs[i] is None:
                continue
            rows[m.name].append((base + off, xs[i], ys[i], zs[i]))

    A = np.array(rows["accelerometer_data"], float)
    Gy = np.array(rows["gyroscope_data"], float)
    if len(A) == 0 or len(Gy) == 0:
        raise ValueError(
            f"{path}: no accelerometer/gyroscope data. Enable raw sensor logging "
            "on the watch (the file must contain accelerometer_data and gyroscope_data)."
        )
    A = A[np.argsort(A[:, 0], kind="stable")]
    Gy = Gy[np.argsort(Gy[:, 0], kind="stable")]
    # drop the zero-padding samples Garmin sometimes emits
    A = A[~(A[:, 1:] == 0).all(1)]
    Gy = Gy[~(Gy[:, 1:] == 0).all(1)]
    # remove duplicate timestamps
    A = A[np.r_[True, np.diff(A[:, 0]) > 0]]
    Gy = Gy[np.r_[True, np.diff(Gy[:, 0]) > 0]]

    # IMU clock -> activity clock: true rate first, then ms since the activity start
    t_imu0 = min(A[0, 0], Gy[0, 0])
    A[:, 0] = (A[:, 0] - t_imu0) * GARMIN_IMU_CLOCK
    Gy[:, 0] = (Gy[:, 0] - t_imu0) * GARMIN_IMU_CLOCK
    t0, offset_ms = _activity_start(A[:, 0], timer, first_record)
    if t0 is None:  # no timer events or records: keep the IMU's own (approximate) start time
        t0 = dt.datetime.fromtimestamp(t_imu0 / 1000.0)
    A[:, 0] += offset_ms
    Gy[:, 0] += offset_ms

    R = GARMIN_TO_BADS if transform is None else transform
    det = np.sign(np.linalg.det(R))
    t_start = 0.0

    segments = []
    breaks = np.where(np.diff(A[:, 0]) > max_gap_ms)[0]
    for s, e in zip(np.r_[0, breaks + 1], np.r_[breaks + 1, len(A)]):
        ta = A[s:e, 0]
        if ta[-1] - ta[0] < 2000:
            continue
        g_mask = (Gy[:, 0] >= ta[0]) & (Gy[:, 0] <= ta[-1])
        if g_mask.sum() < 10:
            continue
        t_new = np.arange(ta[0], ta[-1], 1000.0 / FS)
        acc = _resample(ta, A[s:e, 1:], t_new) * GARMIN_ACC_SCALE
        gyr = _resample(Gy[g_mask, 0], Gy[g_mask, 1:], t_new) * GARMIN_GYRO_SCALE
        segments.append(
            {
                "t": (t_new - t_start) / 1000.0,
                "t0": t0,
                "acc": acc @ R.T,
                "gyr": det * (gyr @ R.T),
            }
        )
    return segments


def _activity_start(t_ms, timer, first_record, block_gap_ms=1000.0):
    """Activity start (datetime) and where the first IMU sample falls after it (ms).

    t_ms: IMU sample times in ms since the first sample, already at the true rate.
    With one timer interval (start -> stop) per IMU block, the offset is the mean over all
    block boundaries; otherwise the first IMU sample is taken to be the first timer start.
    """
    starts = [ts for ts, kind in timer if kind == "start"]
    if not starts:
        return first_record, 0.0
    t0 = starts[0]
    intervals, cur = [], None
    for ts, kind in timer:
        if kind == "start":
            cur = cur or ts
        elif cur is not None:  # stop / stop_all
            intervals.append(((cur - t0).total_seconds() * 1000.0, (ts - t0).total_seconds() * 1000.0))
            cur = None
    br = np.flatnonzero(np.diff(t_ms) > block_gap_ms)
    blocks = list(zip(t_ms[np.r_[0, br + 1]], t_ms[np.r_[br, len(t_ms) - 1]]))
    if len(blocks) == len(intervals):
        return t0, float(np.mean(np.array(intervals) - np.array(blocks)))
    return t0, 0.0


def _bads_window(data, n=200):
    a, g = data["ACCELEROMETER"], data["GYROSCOPE"]
    ta, tg = np.asarray(a["timestamp"], float), np.asarray(g["timestamp"], float)
    t0 = max(ta[0], tg[0])
    t_new = t0 + np.arange(n) * 1000.0 / FS
    acc = _resample(ta, np.c_[a["Gx"], a["Gy"], a["Gz"]], t_new)
    gyr = _resample(tg, np.c_[g["angularSpeedX"], g["angularSpeedY"], g["angularSpeedZ"]], t_new)
    return acc, gyr


def load_bads(root):
    """Load all labelled strokes of the BADS dataset.

    Returns a dict: acc (N,200,3), gyr (N,200,3), y (N,) stroke label, and
    per-stroke metadata player, gender, level.
    """
    accs, gyrs, labels, meta = [], [], [], []
    for fn in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        gender, level, _, player = os.path.basename(os.path.dirname(fn)).split("_")[:4]
        with open(fn) as f:
            for s in json.load(f):
                if not s or not s.get("label") or not s.get("data"):
                    continue
                d = s["data"]
                if not d.get("ACCELEROMETER") or not d.get("GYROSCOPE"):
                    continue
                acc, gyr = _bads_window(d)
                accs.append(acc)
                gyrs.append(gyr)
                labels.append(s["label"]["actionType"])
                meta.append((player, gender, level))
    player, gender, level = np.array(meta).T
    return {"acc": np.array(accs), "gyr": np.array(gyrs), "y": np.array(labels),
            "player": player, "gender": gender, "level": level}
