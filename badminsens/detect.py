"""Stroke (swing) detection in a continuous IMU stream."""
import numpy as np
from scipy.signal import find_peaks

from .io import FS

WIN = 200  # 2 s windows, as in BADS
PEAK_POS = 94  # BADS swings peak at ~47% of the window


def detect_strokes(seg, min_peak_dps=800.0, min_gap_s=0.8):
    """Return sample indices of swing peaks (gyro magnitude maxima)."""
    gm = np.degrees(np.linalg.norm(seg["gyr"], axis=1))
    peaks, _ = find_peaks(
        gm, height=min_peak_dps, distance=int(min_gap_s * FS), prominence=min_peak_dps * 0.5
    )
    return peaks[(peaks >= PEAK_POS) & (peaks + WIN - PEAK_POS <= len(gm))]


def cut_windows(seg, peaks):
    idx = peaks[:, None] - PEAK_POS + np.arange(WIN)[None, :]
    return seg["acc"][idx], seg["gyr"][idx]
