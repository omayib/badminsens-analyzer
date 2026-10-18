"""Overhead-stroke gate: is a swing one of the dataset's stroke types at all?

BADS only contains overhead strokes (clear, smash, drop, backhand clear). In a real
session most swings are something else (lift, net shot, drive, serve, defence) and a
classifier would force them into the nearest overhead class. The gate compares each
swing's orientation signature with the dataset's strokes and marks the rest as
"Other":

  * gravity direction before the swing (how the forearm is held: raised for overhead)
  * gravity direction after the swing (follow-through)
  * rotation axis at peak angular speed (forehand vs backhand, overhead vs side-arm)

A swing is accepted if its Mahalanobis distance to the nearest stroke class (shared
covariance) is below the `quantile` of the training strokes' distances. Unit
vectors only, so it is insensitive to the amplitude differences between watches.
"""
import joblib
import numpy as np
from scipy.signal import sosfiltfilt

from .features import _SOS, align_to_peak


def signature(acc, gyr):
    acc, gyr = align_to_peak(acc, gyr)
    a = sosfiltfilt(_SOS, acc, axis=1)
    w = sosfiltfilt(_SOS, gyr, axis=1)

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

    pre = unit(a[:, 20:60].mean(1))
    post = unit(a[:, 150:190].mean(1))
    pk = np.linalg.norm(w, axis=2).argmax(1)
    wpk = unit(w[np.arange(len(w)), pk])
    return np.hstack([pre, post, wpk])


class OverheadGate:
    def __init__(self, quantile=0.95):
        self.quantile = quantile

    def fit(self, acc, gyr, y):
        F = signature(acc, gyr)
        self.classes = sorted(set(y))
        self.means = np.array([F[y == c].mean(0) for c in self.classes])
        cov = sum(np.cov(F[y == c].T) * (y == c).sum() for c in self.classes) / len(y)
        self.icov = np.linalg.inv(cov + 1e-3 * np.eye(F.shape[1]))
        self.train_distances = np.sort(self.distance(acc, gyr))
        self.threshold = float(np.quantile(self.train_distances, self.quantile))
        return self

    def set_quantile(self, q):
        self.quantile, self.threshold = q, float(np.quantile(self.train_distances, q))
        return self

    def distance(self, acc, gyr):
        F = signature(acc, gyr)
        D = F[:, None, :] - self.means[None]
        return np.sqrt(np.einsum("nki,ij,nkj->nk", D, self.icov, D).min(1))

    def accept(self, acc, gyr):
        return self.distance(acc, gyr) <= self.threshold

    def save(self, path):
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path):
        g = cls()
        g.__dict__.update(joblib.load(path))
        return g
