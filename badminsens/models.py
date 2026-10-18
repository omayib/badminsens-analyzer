"""Config-driven stroke classifiers: every setting of a model is one dict.

    cfg = {"model": "cnn", "norm": "session_z", "epochs": 60, "optimizer": "adamw", "lr": 2e-3, ...}
    m = Model(cfg).fit(acc, gyr, y, groups)        # groups = player / session id per stroke
    proba = m.predict_proba(acc, gyr, groups)

Families
  feature models (scikit-learn): svm, logreg, rf, et, hgb   - on hand-crafted features
  neural nets (PyTorch):         mlp                         - on hand-crafted features
                                 cnn, lstm, gru, cnn_lstm, tcn, transformer - on the IMU sequence

Session normalization ("norm")
  Every player/session is normalized against its own strokes before the classifier sees
  them ("session_center": subtract the session mean, "session_z": also divide by the
  session std, "none"). This removes individual style and device offsets (the main
  cross-user error). At inference a .fit recording is one session. session_z proved
  fragile on Garmin data (per-feature scaling depends on device and stroke mix), so the
  shipped models use session_center or none.
"""
import copy

import joblib
import numpy as np

from . import features

FEATURE_MODELS = ["svm", "logreg", "rf", "et", "hgb"]
SEQUENCE_NETS = ["cnn", "lstm", "gru", "cnn_lstm", "tcn", "transformer"]
NEURAL_MODELS = ["mlp"] + SEQUENCE_NETS
ALL_MODELS = FEATURE_MODELS + NEURAL_MODELS

DEFAULTS = {
    "features": "both",       # paper | ours | both            (feature models, mlp)
    "align": True,            # shift windows so the swing peak is at the same sample
    "norm": "session_center", # session_center | none | session_z (z-scaling is not robust across devices)
    "seed": 0,
    # augmentation: whole-session rotation (watch worn differently) + per-stroke jitter
    "aug_copies": 4, "aug_session_rot": 15.0, "aug_stroke_rot": 8.0, "aug_shift": 6,
    "aug_acc_scale": [0.7, 1.1], "aug_gyr_scale": [0.85, 1.15], "aug_noise": 0.0,
    # sequence input
    "seq_lowpass": 12.0, "seq_decim": 2,
    # neural training
    "epochs": 60, "batch": 32, "optimizer": "adamw", "lr": 2e-3, "weight_decay": 1e-2,
    "scheduler": "cosine", "label_smoothing": 0.1, "dropout": 0.3, "dense": [64],
    # architecture
    "channels": [32, 64, 128], "kernel": 7, "hidden": 64, "layers": 2, "bidirectional": True,
    "d_model": 64, "heads": 4,
    "device": "auto",         # auto (cuda if available) | cpu | cuda
    # inference: re-estimate session statistics with class-balanced weights from the
    # model's own predictions (robust to sessions with an unbalanced stroke mix)
    "balance_iters": 0,
}


_CUDA_OK = None


def _device(cfg):
    """cuda if requested/available and actually usable (a stuck driver falls back to cpu)."""
    global _CUDA_OK
    import torch

    want = cfg.get("device", "auto")
    if want == "cpu":
        return "cpu"
    if _CUDA_OK is None:
        try:
            _CUDA_OK = bool(torch.cuda.is_available() and torch.ones(1, device="cuda").sum().item() == 1)
        except Exception:
            _CUDA_OK = False
    return "cuda" if _CUDA_OK else "cpu"


def full_config(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    return out


# --------------------------------------------------------------------------- inputs

def _raw_features(acc, gyr, kind):
    if kind == "paper":
        return features.extract_paper(acc, gyr)
    if kind == "ours":
        return features.extract(acc, gyr)
    return np.hstack([features.extract_paper(acc, gyr), features.extract(acc, gyr)])


def _sequence(acc, gyr, cfg):
    return features.to_sequence(acc, gyr, lowpass=cfg["seq_lowpass"], decim=cfg["seq_decim"])


def session_normalize(X, groups, mode, ref=None, weights=None):
    """Standardize rows of X per group. X: (N, F) or (N, T, C) (stats over strokes and time).

    ref: optional (X_ref, groups_ref) providing the statistics instead of X itself.
    weights: optional per-row weights of the statistics' source rows.
    """
    if mode == "none":
        return X
    Xr, gr = (X, groups) if ref is None else ref
    w_all = np.ones(len(Xr)) if weights is None else np.asarray(weights, float)
    out = np.empty_like(X)
    for g in np.unique(groups):
        src, w = Xr[gr == g], w_all[gr == g]
        w = (w / w.sum()).reshape((-1,) + (1,) * (src.ndim - 1))
        if src.ndim == 3:
            w = w / src.shape[1]
            mu = (w * src).sum((0, 1))
            sd = np.sqrt((w * (src - mu) ** 2).sum((0, 1)))
        else:
            mu = (w * src).sum(0)
            sd = np.sqrt((w * (src - mu) ** 2).sum(0))
        out[groups == g] = X[groups == g] - mu
        if mode == "session_z":
            out[groups == g] /= sd + 1e-6
    return out


def augment_sessions(acc, gyr, groups, rng, cfg):
    """One augmented copy of every session: session-level rotation + per-stroke jitter."""
    from scipy.spatial.transform import Rotation

    n = len(acc)
    ug, inv = np.unique(groups, return_inverse=True)
    rs = Rotation.from_rotvec(rng.normal(size=(len(ug), 3)) * np.radians(cfg["aug_session_rot"]) / np.sqrt(3))
    rp = Rotation.from_rotvec(rng.normal(size=(n, 3)) * np.radians(cfg["aug_stroke_rot"]) / np.sqrt(3))
    R = (rs[inv] * rp).as_matrix()
    a = np.einsum("nij,ntj->nti", R, acc)
    g = np.einsum("nij,ntj->nti", R, gyr)
    if cfg["aug_shift"]:
        s = rng.integers(-cfg["aug_shift"], cfg["aug_shift"] + 1, size=n)
        a = np.stack([np.roll(x, k, 0) for x, k in zip(a, s)])
        g = np.stack([np.roll(x, k, 0) for x, k in zip(g, s)])
    # amplitude: mostly per session (device / player strength), a little per stroke
    sa = rng.uniform(*cfg["aug_acc_scale"], size=len(ug))[inv] * rng.uniform(0.95, 1.05, n)
    sg = rng.uniform(*cfg["aug_gyr_scale"], size=len(ug))[inv] * rng.uniform(0.95, 1.05, n)
    a, g = a * sa[:, None, None], g * sg[:, None, None]
    if cfg["aug_noise"]:
        a = a + rng.normal(scale=cfg["aug_noise"] * 9.81, size=a.shape)
        g = g + rng.normal(scale=cfg["aug_noise"] * 5.0, size=g.shape)
    return a, g


# --------------------------------------------------------------------------- networks

def build_net(cfg, n_in, n_classes):
    import torch
    from torch import nn

    k, dr = cfg["kernel"], cfg["dropout"]

    def conv(i, o, kk=k, dil=1):
        return nn.Sequential(nn.Conv1d(i, o, kk, padding=dil * (kk // 2), dilation=dil), nn.BatchNorm1d(o), nn.ReLU())

    def head(d):
        layers, prev = [], d
        for u in cfg["dense"]:
            layers += [nn.Dropout(dr), nn.Linear(prev, u), nn.ReLU()]
            prev = u
        return nn.Sequential(*layers, nn.Dropout(dr), nn.Linear(prev, n_classes))

    class TCNBlock(nn.Module):
        def __init__(self, c_in, c, dil):
            super().__init__()
            self.f = nn.Sequential(conv(c_in, c, dil=dil), nn.Dropout(dr * 0.5), conv(c, c, dil=dil))
            self.skip = nn.Conv1d(c_in, c, 1) if c_in != c else nn.Identity()

        def forward(self, x):
            return self.f(x) + self.skip(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            arch = self.arch = cfg["model"]
            ch = cfg["channels"]
            if arch == "mlp":
                d = n_in
            elif arch == "cnn":
                blocks, prev = [], n_in
                for i, c in enumerate(ch):
                    blocks.append(conv(prev, c))
                    if i < len(ch) - 1:
                        blocks.append(nn.MaxPool1d(2))
                    prev = c
                self.body, d = nn.Sequential(*blocks), 2 * ch[-1]
            elif arch == "tcn":
                blocks, prev = [], n_in
                for i, c in enumerate(ch):
                    blocks.append(TCNBlock(prev, c, 2 ** i))
                    prev = c
                self.body, d = nn.Sequential(*blocks), 2 * ch[-1]
            elif arch in ("lstm", "gru", "cnn_lstm"):
                rnn = nn.GRU if arch == "gru" else nn.LSTM
                prev = n_in
                if arch == "cnn_lstm":
                    self.conv = nn.Sequential(conv(n_in, ch[0]), nn.MaxPool1d(2), conv(ch[0], ch[1]), nn.MaxPool1d(2))
                    prev = ch[1]
                self.rnn = rnn(prev, cfg["hidden"], num_layers=cfg["layers"], batch_first=True,
                               bidirectional=cfg["bidirectional"], dropout=dr if cfg["layers"] > 1 else 0.0)
                d = 2 * cfg["hidden"] * (2 if cfg["bidirectional"] else 1)
            elif arch == "transformer":
                dm = cfg["d_model"]
                self.proj = nn.Linear(n_in, dm)
                self.pos = nn.Parameter(torch.zeros(1, 256, dm))
                layer = nn.TransformerEncoderLayer(dm, cfg["heads"], 2 * dm, dr, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, cfg["layers"])
                d = 2 * dm
            else:
                raise ValueError(arch)
            self.head = head(d)

        def forward(self, x):  # x: (B, F) for mlp, (B, T, C) otherwise
            a = self.arch
            if a == "mlp":
                return self.head(x)
            if a in ("cnn", "tcn"):
                h = self.body(x.transpose(1, 2)).transpose(1, 2)
            elif a == "transformer":
                h = self.enc(self.proj(x) + self.pos[:, : x.shape[1]])
            else:
                if a == "cnn_lstm":
                    x = self.conv(x.transpose(1, 2)).transpose(1, 2)
                h, _ = self.rnn(x)
            return self.head(torch.cat([h.mean(1), h.amax(1)], 1))

    return Net()


def _sk_estimator(cfg):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    m, s = cfg["model"], cfg["seed"]
    if m == "svm":
        # probabilities: temperature-scaled softmax over the one-vs-rest decision function
        # (SVC(probability=True) is deprecated in scikit-learn 1.9)
        return make_pipeline(StandardScaler(), SVC(C=cfg.get("C", 1.0), gamma=cfg.get("gamma", "scale"),
                                                   kernel=cfg.get("svm_kernel", "rbf"),
                                                   decision_function_shape="ovr", random_state=s))
    if m == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=cfg.get("C", 0.1), max_iter=5000))
    if m == "rf":
        return RandomForestClassifier(n_estimators=cfg.get("n_estimators", 500), max_features=cfg.get("max_features", "sqrt"),
                                      min_samples_leaf=cfg.get("min_samples_leaf", 1), n_jobs=cfg.get("n_jobs", -1),
                                      random_state=s)
    if m == "et":
        return ExtraTreesClassifier(n_estimators=cfg.get("n_estimators", 600), max_features=cfg.get("max_features", "sqrt"),
                                    min_samples_leaf=cfg.get("min_samples_leaf", 2), n_jobs=cfg.get("n_jobs", -1),
                                    random_state=s)
    if m == "hgb":
        return HistGradientBoostingClassifier(max_iter=cfg.get("max_iter", 300), learning_rate=cfg.get("learning_rate", 0.06),
                                              max_leaf_nodes=cfg.get("max_leaf_nodes", 15),
                                              l2_regularization=cfg.get("l2_regularization", 1.0), random_state=s)
    raise ValueError(m)


# --------------------------------------------------------------------------- model

class Model:
    def __init__(self, cfg):
        self.cfg = full_config(cfg)
        self.name = self.cfg["model"]
        self.family = ("features" if self.name in FEATURE_MODELS
                       else "mlp" if self.name == "mlp" else "sequence")
        self.classes, self.est, self.net, self.mu, self.sd = None, None, None, None, None
        self.svm_temp = 1.0

    # inputs --------------------------------------------------------------
    def _raw(self, acc, gyr):
        if self.cfg.get("align", False):
            acc, gyr = features.align_to_peak(acc, gyr)
        if self.family == "sequence":
            return _sequence(acc, gyr, self.cfg)
        return _raw_features(acc, gyr, self.cfg["features"])

    def _inputs(self, acc, gyr, groups, ref=None):
        X = self._raw(acc, gyr)
        if ref is not None:
            ref = (self._raw(*ref[:2]), ref[2])
        X = session_normalize(X, groups, self.cfg["norm"], ref)
        if self.mu is not None:
            X = (X - self.mu) / self.sd
        return X.astype(np.float32)

    def _training_set(self, acc, gyr, groups, rng, copies):
        """Originals + augmented copies; each copy of a session is normalized as its own session."""
        Xs = [self._inputs(acc, gyr, groups)]
        for _ in range(copies):
            a, g = augment_sessions(acc, gyr, groups, rng, self.cfg)
            Xs.append(self._inputs(a, g, groups))
        return np.concatenate(Xs)

    # fit -----------------------------------------------------------------
    def fit(self, acc, gyr, y, groups):
        self.classes = sorted(np.unique(y).tolist())
        rng = np.random.default_rng(self.cfg["seed"])
        if self.family == "features":
            X = self._training_set(acc, gyr, groups, rng, self.cfg["aug_copies"])
            yy = np.tile(y, self.cfg["aug_copies"] + 1)
            self.est = _sk_estimator(self.cfg).fit(X, yy)
            if self.name == "svm":
                self._calibrate_svm(X, yy, np.tile(groups, self.cfg["aug_copies"] + 1))
        else:
            self._fit_net(acc, gyr, y, groups, rng)
        return self

    def _calibrate_svm(self, X, y, groups):
        """Softmax temperature for the decision function, fitted on out-of-fold decision values
        (grouped by player/session, so augmented copies of a session stay in one fold).
        Calibrating on the training fit itself would give near 0/1 probabilities."""
        from sklearn.base import clone
        from sklearn.model_selection import GroupKFold

        dfun = np.zeros((len(y), len(self.classes)))
        n_splits = min(3, len(np.unique(groups)))
        if n_splits < 2:
            dfun = self.est.decision_function(X)
        else:
            for tr, va in GroupKFold(n_splits).split(X, y, groups):
                dfun[va] = clone(self.est).fit(X[tr], y[tr]).decision_function(X[va])
        yi = np.searchsorted(self.classes, y)
        best = (np.inf, 1.0)
        for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
            p = _softmax(dfun / t)
            best = min(best, (-np.log(p[np.arange(len(y)), yi] + 1e-9).mean(), t))
        self.svm_temp = best[1]

    def _fit_net(self, acc, gyr, y, groups, rng):
        import torch
        from torch import nn

        cfg = self.cfg
        torch.manual_seed(cfg["seed"])
        yi = torch.tensor(np.searchsorted(self.classes, y))
        base = self._inputs(acc, gyr, groups)
        axes = (0,) if base.ndim == 2 else (0, 1)
        self.mu, self.sd = base.mean(axes), base.std(axes) + 1e-6
        base = (base - self.mu) / self.sd
        dev = _device(cfg)
        net = build_net(cfg, base.shape[-1], len(self.classes)).to(dev)
        yi = yi.to(dev)
        base_t = torch.tensor(base, device=dev)
        params = net.parameters()
        if cfg["optimizer"] == "sgd":
            opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
        elif cfg["optimizer"] == "adam":
            opt = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:
            opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        E = cfg["epochs"]
        steps = int(np.ceil(2 * len(y) / cfg["batch"]))
        if cfg["scheduler"] == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=E * steps + 10)
        elif cfg["scheduler"] == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E * steps)
        else:
            sched = None
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
        net.train()
        for _ in range(E):
            # originals + one fresh augmented copy of every session per epoch
            a, g = augment_sessions(acc, gyr, groups, rng, cfg)
            xt = torch.cat([base_t, torch.tensor(self._inputs(a, g, groups), device=dev)])
            yt = torch.cat([yi, yi])
            for b in torch.randperm(len(xt), device=dev).split(cfg["batch"]):
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss_fn(net(xt[b]), yt[b]).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if sched is not None:
                    sched.step()
        self.net = net.eval()

    # predict -------------------------------------------------------------
    def predict_proba(self, acc, gyr, groups=None, ref=None):
        """groups: session id per stroke (default: all one session).
        ref: optional (acc, gyr, groups) whose strokes supply the session statistics."""
        if groups is None:
            groups = np.zeros(len(acc), int)
        raw = self._raw(acc, gyr)
        if self.cfg["norm"] == "none":
            return self._classify(self._scale(raw))
        rraw, rg = (raw, groups) if ref is None else (self._raw(*ref[:2]), ref[2])
        w = np.ones(len(rraw))
        for _ in range(self.cfg.get("balance_iters", 0)):
            # weights that give every (soft-)predicted class the same total mass per session
            pr = self._classify(self._scale(session_normalize(rraw, rg, self.cfg["norm"], (rraw, rg), w)))
            w = np.zeros(len(rraw))
            for g in np.unique(rg):
                m = rg == g
                mass = pr[m].sum(0) + 1e-3
                w[m] = (pr[m] / mass).sum(1)
        return self._classify(self._scale(session_normalize(raw, groups, self.cfg["norm"], (rraw, rg), w)))

    def _scale(self, X):
        if self.mu is not None:
            X = (X - self.mu) / self.sd
        return X.astype(np.float32)

    def _classify(self, X):
        if self.family == "features":
            if self.name == "svm":
                return _softmax(self.est.decision_function(X) / self.svm_temp)
            return self.est.predict_proba(X)
        import torch

        dev = next(self.net.parameters()).device
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X, device=dev)), 1).cpu().numpy()

    # io ------------------------------------------------------------------
    def save(self, path):
        b = {"cfg": self.cfg, "classes": self.classes, "mu": self.mu, "sd": self.sd, "svm_temp": self.svm_temp}
        if self.family == "features":
            b["est"] = self.est
        else:
            b["state_dict"] = {k: v.cpu() for k, v in self.net.state_dict().items()}
        joblib.dump(b, path, compress=3)

    @classmethod
    def load(cls, path):
        b = joblib.load(path)
        m = cls(b["cfg"])
        m.classes, m.mu, m.sd, m.svm_temp = b["classes"], b["mu"], b["sd"], b.get("svm_temp", 1.0)
        # A feature that was (numerically) constant in training, e.g. the gyro peak position
        # of peak-aligned windows, has a ~0 spread; one off-pattern stroke would then explode
        # after scaling and, via session centering, shift the whole session. Leave it unscaled.
        if m.sd is not None:
            m.sd = np.where(np.asarray(m.sd) < 1e-3, 1.0, m.sd).astype(np.asarray(m.sd).dtype)
        if m.family == "features":
            m.est = b["est"]
            for step in getattr(m.est, "named_steps", {}).values():
                if hasattr(step, "scale_"):
                    step.scale_ = np.where(step.scale_ < 1e-3, 1.0, step.scale_)
        else:
            m.net = build_net(m.cfg, np.shape(m.mu)[-1], len(m.classes))
            m.net.load_state_dict(b["state_dict"])
            m.net.to(_device(m.cfg)).eval()
        return m


class EnsembleModel:
    """Average of the probabilities of several models."""

    def __init__(self, members, name="ensemble"):
        self.name, self.members, self.family = name, members, "ensemble"
        self.classes = members[0].classes
        assert all(m.classes == self.classes for m in members)
        self.cfg = {"model": "ensemble", "members": [m.cfg for m in members]}

    def predict_proba(self, acc, gyr, groups=None, ref=None):
        return np.mean([m.predict_proba(acc, gyr, groups, ref) for m in self.members], 0)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(path):
    return Model.load(path)


def clone_cfg(cfg):
    return copy.deepcopy(cfg)
