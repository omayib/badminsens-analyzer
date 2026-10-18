"""Feature extraction and augmentation for 2 s stroke windows (200 x 3 acc, 200 x 3 gyr)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from .io import FS

# Garmin accel carries less high-frequency energy than the BADS watch, so both
# domains are low-passed to a common bandwidth before feature extraction.
_SOS = butter(4, 12.0 / (FS / 2), output="sos")

PHASES = [(0, 60), (60, 85), (85, 105), (105, 130), (130, 200)]  # prep / backswing / impact / follow / recovery
N_RESAMPLE = 25


def align_to_peak(acc, gyr, pos=94):
    """Shift each window so the (12 Hz low-passed) gyro magnitude peak sits at `pos`.

    BADS windows were cut by hand and their peak jitters by ~+/-6 samples; windows cut by
    detect.py are already peak-aligned, so this makes training and inference consistent.
    Edge samples are repeated where the shift runs past the window.
    """
    pk = np.linalg.norm(sosfiltfilt(_SOS, gyr, axis=1), axis=2).argmax(1)
    idx = np.clip(np.arange(acc.shape[1])[None, :] + (pk - pos)[:, None], 0, acc.shape[1] - 1)
    rows = np.arange(len(acc))[:, None]
    return acc[rows, idx], gyr[rows, idx]


def augment_once(acc, gyr, rng, max_rot_deg=20.0):
    """One random copy: wrist-orientation rotation, time shift and amplitude scaling."""
    R = Rotation.from_rotvec(
        rng.normal(size=(len(acc), 3)) * np.radians(max_rot_deg) / np.sqrt(3)
    ).as_matrix()
    a = np.einsum("nij,ntj->nti", R, acc)
    g = np.einsum("nij,ntj->nti", R, gyr)
    shift = rng.integers(-6, 7, size=len(acc))
    a = np.stack([np.roll(x, s, 0) for x, s in zip(a, shift)])
    g = np.stack([np.roll(x, s, 0) for x, s in zip(g, shift)])
    # accel amplitude varies a lot between devices/players; gyro less so
    a = a * rng.uniform(0.5, 1.1, size=(len(a), 1, 1))
    g = g * rng.uniform(0.8, 1.15, size=(len(g), 1, 1))
    return a, g


def augment(acc, gyr, rng, n_copies=4, max_rot_deg=20.0):
    """The originals followed by n_copies augmented copies."""
    A, Gs = [acc], [gyr]
    for _ in range(n_copies):
        a, g = augment_once(acc, gyr, rng, max_rot_deg)
        A.append(a)
        Gs.append(g)
    return np.concatenate(A), np.concatenate(Gs)


def to_sequence(acc, gyr, lowpass=12.0, decim=2):
    """acc, gyr: (N, 200, 3) -> (N, 200 // decim, 8) input for the sequence networks.

    Low-pass (default 12 Hz, as for the features) then decimation (default 50 Hz, no aliasing).
    Channels: acc xyz + |acc| in g, gyro xyz + |gyro| in 1000 deg/s.
    """
    sos = _SOS if lowpass == 12.0 else butter(4, lowpass / (FS / 2), output="sos")
    a = sosfiltfilt(sos, acc, axis=1)[:, ::decim] / 9.80665
    w = np.degrees(sosfiltfilt(sos, gyr, axis=1)[:, ::decim]) / 1000.0
    return np.concatenate(
        [a, np.linalg.norm(a, axis=2, keepdims=True), w, np.linalg.norm(w, axis=2, keepdims=True)], 2
    ).astype(np.float32)


def _stats(x):
    return np.stack(
        [x.mean(1), x.std(1), x.min(1), x.max(1),
         np.percentile(x, 10, axis=1), np.percentile(x, 90, axis=1),
         x.argmax(1) / x.shape[1], x.argmin(1) / x.shape[1]],
        -1,
    ).reshape(len(x), -1)


def extract(acc, gyr):
    """acc, gyr: (N, 200, 3) in m/s^2 and rad/s -> feature matrix (N, F)."""
    acc = sosfiltfilt(_SOS, acc, axis=1)
    gyr = sosfiltfilt(_SOS, gyr, axis=1)
    g = 9.80665
    a = acc / g  # in g
    w = np.degrees(gyr) / 1000.0  # in 1000 deg/s
    am = np.linalg.norm(a, axis=2, keepdims=True)
    wm = np.linalg.norm(w, axis=2, keepdims=True)
    # gravity direction (orientation) before and after the stroke
    pre = acc[:, :50].mean(1)
    post = acc[:, 150:].mean(1)
    pre /= np.linalg.norm(pre, axis=1, keepdims=True) + 1e-9
    post /= np.linalg.norm(post, axis=1, keepdims=True) + 1e-9
    # rotation axis at peak angular speed
    pk = wm[..., 0].argmax(1)
    wpk = w[np.arange(len(w)), pk]
    wdir = wpk / (np.linalg.norm(wpk, axis=1, keepdims=True) + 1e-9)
    apk = a[np.arange(len(a)), pk]

    X = np.concatenate([a, am, w, wm], axis=2)  # (N, 200, 8)
    feats = [_stats(X)]
    for s, e in PHASES:
        feats.append(X[:, s:e].mean(1))
        feats.append(X[:, s:e].std(1))
    # coarse shape of each channel (25 samples over the window)
    idx = np.linspace(0, X.shape[1] - 1, N_RESAMPLE).astype(int)
    feats.append(X[:, idx].reshape(len(X), -1))
    # integrated rotation per axis (net wrist rotation)
    feats.append(w.sum(1) / FS)
    feats.append(np.abs(w).sum(1) / FS)
    # axis cross-correlations
    feats.append(np.stack([np.mean(w[..., i] * w[..., j], 1) for i, j in [(0, 1), (0, 2), (1, 2)]], 1))
    feats.append(np.stack([np.mean(a[..., i] * w[..., j], 1) for i in range(3) for j in range(3)], 1))
    feats += [pre, post, wdir, wpk, apk, (pre * post).sum(1, keepdims=True)]
    return np.concatenate(feats, 1).astype(np.float32)


# --------------------------------------------------------------------------- paper features
# BadminSense (Chen et al., CHI '26) section 5.3.1: drop 20 frames at both ends, 2nd-order
# 20 Hz low-pass, then 23 time/frequency descriptors per IMU axis (6 x 23 per stroke).
_SOS20 = butter(2, 20.0 / (FS / 2), output="sos")
TRIM = 20


def _axis_features(x):
    """x: (N, T, C) -> (N, C, 23)."""
    from scipy.signal import find_peaks, welch
    from scipy.stats import kurtosis, skew

    n, t, c = x.shape
    dx = np.diff(x, axis=1) * FS
    pct = np.percentile(x, [10, 25, 50, 75, 90], axis=1)  # (5, N, C)
    peaks = np.array([[len(find_peaks(x[i, :, j], prominence=0.25 * (np.ptp(x[i, :, j]) + 1e-9))[0])
                       for j in range(c)] for i in range(n)], float)
    spec = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2  # (N, F, C)
    freqs = np.fft.rfftfreq(t, 1 / FS)
    p = spec / (spec.sum(1, keepdims=True) + 1e-12)
    _, psd = welch(x, fs=FS, nperseg=64, axis=1)
    f = [
        x.sum(1), x.mean(1), x.var(1), x.std(1), skew(x, axis=1), kurtosis(x, axis=1), *pct, peaks,
        dx.mean(1), dx.std(1), skew(dx, axis=1), kurtosis(dx, axis=1),
        np.percentile(dx, 10, axis=1), np.percentile(dx, 90, axis=1),
        np.log1p(spec.sum(1)), (p * freqs[None, :, None]).sum(1), -(p * np.log(p + 1e-12)).sum(1),
        np.log1p(psd.max(1)), np.log1p(psd.mean(1)),
    ]
    return np.nan_to_num(np.stack(f, -1))


def extract_paper(acc, gyr, magnitude=False):
    """acc (m/s^2), gyr (rad/s): (N, 200, 3) -> (N, 6*23) [or 8*23 with magnitude channels]."""
    x = np.concatenate([acc, gyr], 2)[:, TRIM:-TRIM]
    x = sosfiltfilt(_SOS20, x, axis=1)
    if magnitude:
        x = np.concatenate([x, np.linalg.norm(x[..., :3], axis=2, keepdims=True),
                            np.linalg.norm(x[..., 3:], axis=2, keepdims=True)], 2)
    return _axis_features(x).reshape(len(x), -1).astype(np.float32)
