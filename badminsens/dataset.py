"""Pre-process the BADS dataset into a cached array file and split it by player.

    python -m badminsens.dataset --data BADS_CLL_OPENACCESS_V1

Each labelled stroke becomes a 2 s window (200 samples at 100 Hz) of
accelerometer (m/s^2) and gyroscope (rad/s) data. The split is by player, so
validation/test accuracy measures performance on people the model never saw -
the situation when analysing a new user's Garmin recording.
"""
import argparse
import os
from collections import Counter

import numpy as np

from .io import load_bads

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bads_windows.npz")

# 12 players: Play1..10 men (High: 4, 6 / Medium: 2, 5, 7, 8, 10 / Low: 1, 3, 9),
# Play11..12 women (Low). Every split mixes skill levels, train and test both
# contain a woman and a high-level player.
SPLIT = {
    "train": ["Play1", "Play2", "Play4", "Play7", "Play8", "Play10", "Play11"],
    "val": ["Play3", "Play5"],
    "test": ["Play6", "Play9", "Play12"],
}


def load(root="BADS_CLL_OPENACCESS_V1", cache=DEFAULT_CACHE, rebuild=False):
    """Return dict with acc (N,200,3), gyr (N,200,3), y, player, gender, level."""
    if cache and os.path.exists(cache) and not rebuild:
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    d = load_bads(root)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **d)
    return d


def split_indices(d, split=SPLIT):
    return {name: np.flatnonzero(np.isin(d["player"], players)) for name, players in split.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="BADS_CLL_OPENACCESS_V1")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    args = ap.parse_args()
    d = load(args.data, args.cache, rebuild=True)
    print(f"{len(d['y'])} strokes -> {args.cache}")
    for name, idx in split_indices(d).items():
        c = Counter(d["y"][idx])
        print(f"{name:5s} {len(idx):4d} strokes, players {sorted(set(d['player'][idx].tolist()), key=lambda p: int(p[4:]))}: "
              + ", ".join(f"{k}={c[k]}" for k in sorted(c)))


if __name__ == "__main__":
    main()
