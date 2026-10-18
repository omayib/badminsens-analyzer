"""Playstyle model: session features -> attribute scores, playstyle archetype and skill tier.

    python -m src.model fit                  # fit the skill index on BADS, write models/playstyle.json
    python -m src.model validate raw/*.fit   # tune the archetype temperature, stability report

Three layers, following references/badminton-play-mode-equipments.md:

1. Attributes (0-100): Power, Speed, Control, Endurance, Agility. Each is a weighted mean of
   anchored feature scores; an anchor (lo, hi) maps lo -> 10 and hi -> 90 on a logistic curve.
   Anchors for wrist speed come from the BADS players (12 players, Low/Medium/High level);
   the others are physiological / coaching ranges.
2. Archetype: Attacking, Defensive, Control or All-round. The attribute profile is centred on
   its own mean (relative strengths) and compared with prototype shapes; a flat profile is
   All-round. Probabilities are a softmax over squared distances with temperature T, tuned
   so the top probability matches how often the archetype survives a bootstrap of the swings.
3. Skill tier: Low / Medium / High from a skill index fitted on BADS levels (smash speed,
   backhand speed, smash-drop separation; gyro clipped to the Garmin's +/-2000 deg/s range)
   and validated leave-one-player-out.

Weights live in models/playstyle.json; load() falls back to the Hugging Face repo.
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "models", "playstyle.json")
HF_REPO = "omayib/badak"
ATTRS = ["Power", "Speed", "Control", "Endurance", "Agility"]
LEVELS = ["Low", "Medium", "High"]

# attribute -> [(feature, lo, hi, weight)]; lo > hi means "lower is better"
DEFAULT_ATTRIBUTES = {
    "Power": [("smash_p75_dps", 2500, 3000, 2.0), ("swing_p90_kmh", 90, 130, 1.0),
              ("acc_p90_g", 7.0, 11.0, 1.0), ("attack_ratio", 0.25, 0.60, 1.0)],
    "Speed": [("swings_per_min", 6.0, 12.0, 1.5), ("shot_interval_s", 2.6, 1.6, 1.0),
              ("swing_med_kmh", 45, 75, 1.0)],
    "Control": [("attack_ratio", 0.60, 0.25, 1.0), ("speed_cv", 0.35, 0.18, 1.0),
                ("mean_confidence", 0.60, 0.85, 1.0), ("transition_entropy", 1.3, 2.1, 1.0)],
    "Endurance": [("active_min", 20, 90, 1.0), ("hr_avg_pct_hrr", 35, 70, 1.0),
                  ("speed_retention", 0.60, 1.00, 1.5), ("aerobic_te", 1.5, 4.0, 1.0)],
    "Agility": [("move_g", 0.15, 0.30, 1.5), ("backhand_share", 0.08, 0.20, 1.0),
                ("rally_len", 1.8, 3.2, 1.0), ("anaerobic_te", 0.5, 3.0, 1.0)],
}

# prototype shapes over ATTRS, relative to the player's own mean (All-round = flat)
DEFAULT_ARCHETYPES = {
    "Attacking": [22, -2, -12, -2, -6],
    "Defensive": [-12, 10, -2, 2, 12],
    "Control": [-10, -6, 22, 2, -8],
    "All-round": [0, 0, 0, 0, 0],
}

ARCHETYPE_INFO = {
    "Attacking": "Rear-court power: a high share of smashes and fast overhead swings.",
    "Defensive": "Fast, reactive play: quick shot tempo, lots of movement and backhand work.",
    "Control": "Rally construction: drops and clears over smashes, consistent swing speed, varied patterns.",
    "All-round": "No single dominant trait: power, tempo and control in balance.",
}


def anchored(x, lo, hi):
    """Logistic score 0..1 with lo -> 0.1 and hi -> 0.9 (works for lo > hi too)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    z = (x - (lo + hi) / 2) / ((hi - lo) / 2)  # lo -> -1, hi -> +1
    return 1 / (1 + math.exp(-math.log(9) * z))


class PlaystyleModel:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.attributes = {a: [tuple(c) for c in comps] for a, comps in cfg.get("attributes", DEFAULT_ATTRIBUTES).items()}
        self.archetypes = cfg.get("archetypes", DEFAULT_ARCHETYPES)
        self.temperature = cfg.get("temperature", 60.0)
        self.skill = cfg.get("skill")  # fitted by fit_skill()
        self.validation = cfg.get("validation", {})

    # ---------- layer 1: attributes ----------
    def attribute_scores(self, f):
        out = {}
        for a, comps in self.attributes.items():
            num = den = 0.0
            for feat, lo, hi, w in comps:
                s = anchored(f.get(feat), lo, hi)
                if s is not None:
                    num, den = num + w * s, den + w
            out[a] = round(100 * num / den, 1) if den else 50.0
        return out

    def components(self, f):
        """Per-feature contribution table (for explanations in the app)."""
        rows = []
        for a, comps in self.attributes.items():
            for feat, lo, hi, w in comps:
                s = anchored(f.get(feat), lo, hi)
                rows.append({"attribute": a, "feature": feat, "value": f.get(feat), "lo": lo, "hi": hi, "weight": w,
                             "score": None if s is None else round(100 * s, 1)})
        return rows

    # ---------- layer 2: archetype ----------
    def archetype_probs(self, scores, temperature=None):
        T = temperature or self.temperature
        x = np.array([scores[a] for a in ATTRS], float)
        x = x - x.mean()
        names = list(self.archetypes)
        d2 = np.array([((x - np.array(self.archetypes[k], float)) ** 2).sum() for k in names])
        logit = -d2 / (2 * T ** 2) * 25  # T in score points; 25 keeps T ~ tens
        p = np.exp(logit - logit.max())
        p /= p.sum()
        return dict(zip(names, (round(float(v), 3) for v in p)))

    # ---------- layer 3: skill tier ----------
    def skill_index(self, f):
        if not self.skill:
            return None
        zs = [(f[k] - m) / s for k, m, s in zip(self.skill["features"], self.skill["mean"], self.skill["std"])
              if f.get(k) is not None and not math.isnan(f[k])]
        return float(np.mean(zs)) if zs else None

    def skill_tier(self, f):
        idx = self.skill_index(f)
        if idx is None:
            return None, None
        lo, hi = self.skill["cutpoints"]
        return (LEVELS[0] if idx < lo else LEVELS[1] if idx < hi else LEVELS[2]), round(idx, 2)

    # ---------- all together ----------
    def predict(self, features):
        scores = self.attribute_scores(features)
        probs = self.archetype_probs(scores)
        tier, idx = self.skill_tier(features)
        top = max(probs, key=probs.get)
        return {"attributes": scores, "archetype": top, "archetype_probs": probs, "skill_tier": tier,
                "skill_index": idx, "description": ARCHETYPE_INFO[top]}

    def to_dict(self):
        return {"attributes": {a: [list(c) for c in comps] for a, comps in self.attributes.items()},
                "archetypes": self.archetypes, "temperature": self.temperature, "skill": self.skill,
                "validation": self.validation}

    def save(self, path=CONFIG_PATH):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


def load(path=CONFIG_PATH, repo_id=HF_REPO):
    """Local models/playstyle.json, else download it from the Hugging Face repo."""
    if not os.path.exists(path):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id, "models/playstyle.json")
        except Exception:  # offline and no local file: untrained defaults
            return PlaystyleModel()
    with open(path) as f:
        return PlaystyleModel(json.load(f))


# ---------------------------------------------------------------- fitting / validation
SKILL_FEATURES = ["smash_p75_dps", "backhand_med_dps", "smash_drop_gap_dps"]


def bads_player_features(npz=os.path.join(ROOT, "data", "bads_windows.npz")):
    """Per-player skill features on BADS, gyro/acc clipped to the Garmin sensor range."""
    d = np.load(npz)
    g = np.linalg.norm(np.clip(np.degrees(d["gyr"]), -2000, 2000), axis=2).max(1)
    players = sorted(set(d["player"].tolist()), key=lambda s: int(s[4:]))
    X, y = [], []
    for p in players:
        m = d["player"] == p
        sm, dr, bh = (g[m & (d["y"] == c)] for c in ("ForehandKill", "ForehandLob", "BackhandTransition"))
        X.append([np.percentile(sm, 75), np.median(bh), np.median(sm) - np.median(dr)])
        y.append(LEVELS.index(str(d["level"][m][0])))
    return players, np.array(X), np.array(y)


def _fit_index(X, y):
    mean, std = X.mean(0), X.std(0)
    idx = ((X - mean) / std).mean(1)
    lv = [idx[y == k].mean() for k in range(3)]
    return mean, std, [(lv[0] + lv[1]) / 2, (lv[1] + lv[2]) / 2]


def fit_skill(model):
    from scipy.stats import spearmanr

    players, X, y = bads_player_features()
    # leave-one-player-out: fit on 11, place the 12th
    pred, held = [], []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mean, std, cut = _fit_index(X[tr], y[tr])
        v = float(((X[i] - mean) / std).mean())
        held.append(v)
        pred.append(0 if v < cut[0] else 1 if v < cut[1] else 2)
    pred = np.array(pred)
    mean, std, cut = _fit_index(X, y)
    rho = spearmanr(y, held)
    per_feat = {k: round(float(spearmanr(y, X[:, j]).statistic), 2) for j, k in enumerate(SKILL_FEATURES)}
    model.skill = {"features": SKILL_FEATURES, "mean": mean.round(1).tolist(), "std": std.round(1).tolist(),
                   "cutpoints": [round(c, 3) for c in cut], "levels": LEVELS}
    model.validation["skill_loso"] = {
        "players": len(y), "exact_accuracy": round(float((pred == y).mean()), 3),
        "within_one_level": round(float((np.abs(pred - y) <= 1).mean()), 3),
        "majority_baseline": round(float(np.bincount(y).max() / len(y)), 3),
        "spearman_rho_heldout_index_vs_level": round(float(rho.statistic), 3),
        "spearman_p": round(float(rho.pvalue), 4), "per_feature_rho": per_feat,
        "per_player": [{"player": p, "level": LEVELS[t], "predicted": LEVELS[q], "index": round(v, 2)}
                       for p, t, q, v in zip(players, y, pred, held)],
    }
    return model


def _bootstrap_swings(session, rng):
    from .parser import session_features

    sw = session["swings"]
    # resample whole rallies-ish blocks: 20-swing blocks keep the temporal structure
    blocks = [sw[i:i + 20] for i in range(0, len(sw), 20)]
    pick = rng.integers(0, len(blocks), len(blocks))
    order = sorted(pick)
    res = []
    for j, b in enumerate(order):
        res.extend(blocks[b])
    meta = {**session["meta"], "_move_g": session["features"]["move_g"]}
    return session_features(res, meta, session["hr"], session["games"])


def validate(model, sessions, n_boot=200, seed=0, temps=(10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80)):
    """Choose the temperature whose top probability best matches bootstrap agreement.

    For each session the archetype is recomputed on n_boot block-bootstraps of its swings;
    the share of resamples that keep the full-session archetype is the empirical
    reliability. The chosen T minimises the gap |mean top prob - mean agreement|
    (a calibration criterion, no labels needed). Per-game agreement is reported too."""
    rng = np.random.default_rng(seed)
    boot = {s["activity_id"]: [model.attribute_scores(_bootstrap_swings(s, rng)) for _ in range(n_boot)]
            for s in sessions}
    full = {s["activity_id"]: model.attribute_scores(s["features"]) for s in sessions}
    table = []
    for T in temps:
        tops, agree = [], []
        for sid, sc in full.items():
            p = model.archetype_probs(sc, T)
            top = max(p, key=p.get)
            tops.append(p[top])
            bt = [max(q := model.archetype_probs(b, T), key=q.get) for b in boot[sid]]
            agree.append(np.mean([b == top for b in bt]))
        table.append({"T": T, "mean_top_prob": round(float(np.mean(tops)), 3),
                      "mean_bootstrap_agreement": round(float(np.mean(agree)), 3),
                      "gap": round(abs(float(np.mean(tops)) - float(np.mean(agree))), 3)})
    best = min(table, key=lambda r: r["gap"])
    model.temperature = float(best["T"])

    attr_sd = {a: round(float(np.mean([np.std([b[a] for b in boot[sid]]) for sid in boot])), 2) for a in ATTRS}
    games, same = [], 0
    for s in sessions:
        top = model.predict(s["features"])["archetype"]
        for g in s["per_game"]:
            gp = model.predict(g["features"])
            games.append({"session": s["activity_id"], "game": g["game"], "archetype": gp["archetype"],
                          "session_archetype": top})
            same += gp["archetype"] == top
    model.validation["archetype"] = {
        "criterion": "min |mean top-archetype probability - mean bootstrap agreement| over sessions",
        "n_sessions": len(sessions), "n_bootstrap": n_boot, "temperature_search": table, "chosen_T": model.temperature,
        "bootstrap_attribute_sd": attr_sd,
        "per_game_agreement_with_session": round(same / len(games), 3) if games else None, "per_game": games,
        "sessions": [{"session": s["activity_id"], **model.predict(s["features"])} for s in sessions],
    }
    return model


def main():
    import argparse

    from .parser import parse_fit

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "validate", "predict"])
    ap.add_argument("fit_files", nargs="*")
    ap.add_argument("--cache", default=os.path.join(ROOT, ".cache", "sessions"))
    args = ap.parse_args()
    m = load() if os.path.exists(CONFIG_PATH) else PlaystyleModel()
    if args.cmd == "fit":
        fit_skill(m)
        m.save()
        print(json.dumps(m.validation["skill_loso"], indent=1))
        print(f"-> {CONFIG_PATH}")
        return
    sessions = sorted((parse_fit(p, cache_dir=args.cache) for p in args.fit_files), key=lambda s: s["start_local"])
    if args.cmd == "validate":
        validate(m, sessions)
        m.save()
        v = m.validation["archetype"]
        for r in v["temperature_search"]:
            print(r)
        print("chosen T", v["chosen_T"], "| bootstrap attribute SD", v["bootstrap_attribute_sd"],
              "| per-game agreement", v["per_game_agreement_with_session"])
    for s in sessions:
        p = m.predict(s["features"])
        print(f"{s['activity_id']} {s['start_local'][:16]} {p['archetype']:10s} "
              f"{' '.join(f'{a[:3]} {v:4.0f}' for a, v in p['attributes'].items())}  skill {p['skill_tier']} "
              f"({p['skill_index']})  {p['archetype_probs']}")


if __name__ == "__main__":
    main()
