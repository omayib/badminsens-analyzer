"""Session charts: heart rate + strokes, swing speed, and when each stroke was played.

    python -m badminsens.plot ACTIVITY.fit                          # all charts + both report pages
    python -m badminsens.plot ACTIVITY.fit --charts hr speed        # a selection
    python -m badminsens.plot ACTIVITY.fit [--model svm] [--min-rest 2]

Charts (PNG next to the .fit):
  hr             <fit>_hr_strokes.png          heart rate on top, one row of ticks per stroke type
  speed          <fit>_swing_speed.png         average racket speed (km/j) per 5 minutes, per stroke
  speed-simple   <fit>_swing_speed_simple.png  average (estimated) racket speed per stroke and game
  timeline       <fit>_stroke_timeline.png     number of each stroke per 2 minutes
  report-desktop <fit>_report_desktop.png      all four on one landscape page (2 x 2 grid)
  report-mobile  <fit>_report_mobile.png       all four in one narrow column for phones

Games: pausing the watch app between games leaves gaps in the IMU data; any gap of at
least --min-rest minutes (default 2) is shown as rest and starts a new game.
"""
import argparse
import datetime as dt
import functools
import os

import numpy as np

from .predict import OTHER, STROKES, classify, detect, load_model, registry, set_balance
from .gate import OverheadGate
from .io import load_fit

# categorical slots 1-4 (validated for colour-vision deficiency); 'Other' is muted gray
COLORS = {"ForehandKill": "#2a78d6", "ForehandHigh": "#eb6834", "ForehandLob": "#1baf7a",
          "BackhandTransition": "#eda100"}
SURF, INK, INK2, MUTED, GRID, HR_INK = "#fcfcfb", "#0b0b0b", "#52514e", "#b5b4ae", "#e4e3df", "#3a3a38"
# blue ramp steps 250..600: games get evenly spaced steps, light (game 1) -> dark (last);
# the light end is the lightest step that still clears 2:1 on the surface
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
RACKET_M = 0.7  # assumed wrist -> racket head distance for the km/j (km/h) estimate
SPEED_UNIT = "km/j"  # km per jam (hour)
TIMELINE_BIN_MIN = 2.0
SPEED_BIN_MIN = 5.0  # average swing speed per 5 minutes: enough shots per bar to be a stable average
DEFAULT_UTC_OFFSET_H = 7  # WIB, used when the file does not record the watch's local time
TZ_NAMES = {7: "WIB", 8: "WITA", 9: "WIT"}


@functools.lru_cache(maxsize=8)
def _fit_meta(fit_path):
    """Heart-rate records and the watch's UTC offset (hours), read in one pass and cached."""
    import fitparse

    hr, offset = [], None
    for m in fitparse.FitFile(fit_path).get_messages(["record", "timestamp_correlation"]):
        if m.name == "record":
            t, h = m.get_value("timestamp"), m.get_value("heart_rate")
            if t is not None and h is not None:
                hr.append((t, h))
        elif offset is None and m.get_value("local_timestamp") and m.get_value("timestamp"):
            offset = round((m.get_value("local_timestamp") - m.get_value("timestamp")).total_seconds() / 3600, 2)
    return tuple(hr), DEFAULT_UTC_OFFSET_H if offset is None else offset


def heart_rate(fit_path):
    return list(_fit_meta(fit_path)[0])


def local_time(fit_path, t_utc):
    """(local datetime, zone name) for a UTC datetime from the file, e.g. WIB for UTC+7."""
    off = _fit_meta(fit_path)[1]
    name = TZ_NAMES.get(off, f"UTC{off:+g}")
    return t_utc + dt.timedelta(hours=off), name


def game_spans(segments, min_rest_min=2.0):
    """(start, end) in minutes of each game: load_fit segments joined across gaps < min_rest_min."""
    spans = []
    for s in segments:
        a, b = s["t"][0] / 60, s["t"][-1] / 60
        if spans and a - spans[-1][1] < min_rest_min:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def _game_of(t_min, games):
    """Index of the game containing each time (nearest game for swings on a boundary)."""
    mid = np.array([(a + b) / 2 for a, b in games])
    return np.array([next((i for i, (a, b) in enumerate(games) if a - 0.05 <= t <= b + 0.05),
                          int(np.abs(mid - t).argmin())) for t in t_min], int)


def _game_colors(n):
    return [BLUE_RAMP[i] for i in np.linspace(0, len(BLUE_RAMP) - 1, n).astype(int)] if n > 1 else [BLUE_RAMP[4]]


def _swings(rows):
    """time (min), peak wrist speed (deg/s) and BADS label ('' = not a dataset stroke) per swing."""
    t = np.array([r["time_s"] for r in rows], float) / 60
    v = np.array([r["peak_gyro_dps"] for r in rows], float)
    lab = np.array([r["label"] if r["gate"] == "accepted" else "" for r in rows])
    return t, v, lab


def _kmh(dps):
    return np.radians(dps) * RACKET_M * 3.6


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def _shade_rests(ax, games, label=False):
    for (_, b), (a, _) in zip(games[:-1], games[1:]):
        ax.axvspan(b, a, color=GRID, alpha=0.45, lw=0, zorder=0)
        if label:
            ax.text((a + b) / 2, 0.08, f"rest (paused)\n{a - b:.0f} min", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=INK2)


def _canvas(fig, out, figsize, **kw):
    """Axes on `fig` (a Figure or a SubFigure of a report page), or on a new figure."""
    if fig is None:
        import matplotlib

        if out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, dpi=150, facecolor=SURF)
    return fig, fig.subplots(**kw)


def _size_in(fig):
    """(width, height) of a Figure or SubFigure in inches."""
    return fig.bbox.width / fig.figure.dpi, fig.bbox.height / fig.figure.dpi


def _header(fig, title, subtitle="", left_in=0.6, compact=False, title_size=14):
    """Title and a wrapped subtitle at a fixed distance from the top, whatever the panel size.
    Returns the figure fraction just below the header (for subplots_adjust(top=...))."""
    import textwrap

    w, h = _size_in(fig)
    x, y = left_in / w, 1 - 0.2 / h
    fig.text(x, y, title, fontsize=title_size, weight="bold", color=INK, va="top")
    used = 0.2 + title_size / 72 * 1.3
    if subtitle:
        chars = int((w - left_in - 0.2) * 72 / (10 * 0.55))  # ~ characters per line at 10 pt
        text = "\n".join(textwrap.fill(p, chars) for p in subtitle.split("\n"))
        fig.text(x, 1 - (used + 0.06) / h, text, fontsize=10, color=INK2, va="top", linespacing=1.35)
        used += 0.06 + (text.count("\n") + 1) * 10 / 72 * 1.35
    return 1 - (used + 0.15) / h


def _footnote(fig, text, left_in=0.6):
    """Small wrapped note at the bottom. Returns the figure fraction just above it."""
    import textwrap

    w, h = _size_in(fig)
    chars = int((w - left_in - 0.2) * 72 / (7.5 * 0.55))
    text = textwrap.fill(text, chars)
    fig.text(left_in / w, 0.12 / h, text, fontsize=7.5, color=INK2, va="bottom", linespacing=1.3)
    return (0.12 + (text.count("\n") + 1) * 7.5 / 72 * 1.3) / h


def plot_session(fit_path, rows, model_name="", out=None, fig=None, compact=False):
    """Heart rate with one row of stroke ticks per type underneath.
    rows: output of predict.classify (needs 'clock', 'label', 'gate'). Returns the figure."""
    hr = heart_rate(fit_path)
    t0 = hr[0][0] if hr else None

    def minutes(clock):
        c = dt.datetime.strptime(clock, "%H:%M:%S.%f").time()
        base = t0 or dt.datetime.strptime(rows[0]["clock"], "%H:%M:%S.%f")
        return (dt.datetime.combine(base.date(), c) - base.replace(tzinfo=None)).total_seconds() / 60

    fig, (ax, axs) = _canvas(fig, out, (12, 6.4), nrows=2, ncols=1, sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.7], "hspace": 0.08})
    for a in (ax, axs):
        a.set_facecolor(SURF)
    smin = np.array([minutes(r["clock"]) for r in rows])
    xmax = smin.max() + 0.5 if len(smin) else 1
    if hr:
        t = np.array([(ts - t0).total_seconds() / 60 for ts, _ in hr])
        h = np.array([v for _, v in hr], float)
        cut = np.flatnonzero(np.diff(t) > 1.0) + 1  # break the line across recording gaps
        ax.plot(np.insert(t, cut, np.nan), np.insert(h, cut, np.nan), color=HR_INK, lw=2,
                solid_capstyle="round", solid_joinstyle="round")
        ax.axhline(h.mean(), color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        if len(t) > 1:
            k = int(np.argmax(np.diff(t)))
            if t[k + 1] - t[k] > 3:
                mid = (t[k] + t[k + 1]) / 2
                ax.text(mid, h.min() - 4, f"rest ({t[k + 1] - t[k]:.0f} min)", color=INK2, fontsize=8.5,
                        ha="center", va="top", style="italic")
                ax.text(mid, h.mean() + 1.5, f"avg {h.mean():.0f} bpm", color=INK2, fontsize=9, ha="center",
                        va="bottom")
            else:
                ax.text(0.5, h.mean() + 1.5, f"avg {h.mean():.0f} bpm", color=INK2, fontsize=9, va="bottom")
        i = int(h.argmax())
        ax.scatter([t[i]], [h[i]], s=40, color=HR_INK, edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(f"max {h[i]:.0f} bpm", (t[i], h[i]), xytext=(8, 4), textcoords="offset points", color=INK,
                    fontsize=9, ha="right" if t[i] > 0.8 * t[-1] else "left")
        ax.set_ylim(max(0, h.min() - 14), h.max() + 10)
        xmax = max(xmax, t[-1] + 0.5)
        lt, zone = local_time(fit_path, t0)
        start = f"started {lt:%H:%M} {zone}"
    else:
        ax.text(0.5, 0.5, "no heart-rate data in this file", transform=ax.transAxes, ha="center", color=INK2)
        start = ""
    ax.set_ylabel("heart rate (bpm)", color=INK2, fontsize=9)

    other = "Other" if compact else OTHER
    rows_def = [(lab, STROKES[lab][0], COLORS[lab]) for lab in COLORS] + [("", other, MUTED)]
    ticks = []
    for i, (lab, name, col) in enumerate(rows_def):
        y = len(rows_def) - 1 - i
        xs = [m for m, r in zip(smin, rows) if (r["label"] == lab if lab else r["gate"] != "accepted")]
        axs.vlines(xs, y - 0.34, y + 0.34, color=col, lw=1.3)
        axs.axhline(y, color=GRID, lw=0.6, zorder=0)
        ticks.append((y, f"{name} ({len(xs)})"))
    axs.set_yticks([y for y, _ in ticks])
    axs.set_yticklabels([s for _, s in ticks], fontsize=9, color=INK)
    axs.set_ylim(-0.6, len(rows_def) - 0.4)
    axs.set_xlabel("minutes since start", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for a in (ax, axs):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9, length=0)
    ax.spines["bottom"].set_visible(False)
    axs.set_xlim(0, xmax)
    w, h_in = _size_in(fig)
    left_in = 1.5 if compact else 2.4
    sub = " · ".join(p for p in [start, model_name and f"model {model_name}",
                                 None if compact else os.path.basename(fit_path)] if p)
    top = _header(fig, "Heart rate and strokes", sub, left_in=0.25, compact=compact)
    fig.subplots_adjust(left=left_in / w, right=1 - 0.2 / w, top=top, bottom=0.55 / h_in)
    if out:
        fig.savefig(out, facecolor=SURF)
    return fig


def plot_swing_speed(rows, games, title="", out=None, fig=None, compact=False):
    """Average estimated racket speed (km/j) per 5 minutes: one bar panel per stroke type plus
    'Other'. Bars of fewer than 3 shots are drawn light. Returns the figure."""
    from matplotlib.ticker import MultipleLocator

    t, v, lab = _swings(rows)
    kmh = _kmh(v)
    panels = [(k, STROKES[k][0], COLORS[k]) for k in COLORS] + [("", "Other swings", MUTED)]
    fig, axes = _canvas(fig, out, (12, 9.8), nrows=len(panels), ncols=1, sharex=True, sharey=True,
                        gridspec_kw={"hspace": 0.55 if compact else 0.45})
    bars, best = [], None  # best: fastest well-sampled bar of an overhead stroke
    for ax, (k, name, col) in zip(axes, panels):
        _style(ax)
        m = lab == k
        for gi, (a, b) in enumerate(games):  # bins restart at each game so no bar straddles a rest
            edges = np.arange(a, b + 0.01, SPEED_BIN_MIN)
            edges = np.r_[edges, b] if b - edges[-1] > SPEED_BIN_MIN / 2 else np.r_[edges[:-1], b]  # short tail joins the last bin
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = m & (t >= lo) & (t < hi + (1e-6 if hi == b else 0))
                if not sel.any():
                    continue
                val, cnt = float(kmh[sel].mean()), int(sel.sum())
                few = cnt < 3
                ax.bar((lo + hi) / 2, val, (hi - lo) * 0.84, color=col, alpha=0.35 if few else 1, zorder=3)
                bars.append(val)
                if not few:
                    ax.text((lo + hi) / 2, val + 3, f"{val:.0f}", ha="center", va="bottom", color=INK,
                            fontsize=7 if compact else 8.5)
                    if k and (best is None or val > best[0]):
                        best = (val, name, gi, lo, hi)
            if ax is axes[0] and len(games) > 1:
                ax.text((a + b) / 2, 1.55 if compact else 1.4, f"Game {gi + 1}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=9.5 if compact else 10.5, color=INK, weight="bold")
        _shade_rests(ax, games)
        if ax is axes[-1] and not compact:
            for (_, b), (a, _) in zip(games[:-1], games[1:]):
                ax.text((a + b) / 2, 0.5, "rest", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=9, color=INK2)
        avg = f"average {kmh[m].mean():.0f} {SPEED_UNIT}" if m.any() else "none"
        ax.set_title(f"{name} · {m.sum()} shots · {avg}", loc="left", fontsize=9 if compact else 10, color=INK,
                     weight="bold" if k else "normal", pad=3)
    top_val = max(bars + [100])
    axes[0].set_ylim(0, np.ceil(top_val * 1.15 / 25) * 25)
    axes[0].yaxis.set_major_locator(MultipleLocator(50))
    axes[-1].set_xlabel(f"minutes since start  ·  each bar = average of {SPEED_BIN_MIN:.0f} minutes"
                        + ("  ·  shaded = rest" if compact and len(games) > 1 else ""), color=INK2,
                        fontsize=8.5 if compact else 9)
    axes[-1].set_xlim(0, (games[-1][1] if games else t.max()) + 0.5)
    w, h = _size_in(fig)
    left_in = 0.6 if compact else 0.8
    fig.text(0.12 / w, 0.5, f"racket speed ({SPEED_UNIT}, estimated)", rotation=90, va="center", color=INK2,
             fontsize=9)
    sub = (f"Fastest 5 minutes: {best[1].lower()}, {best[0]:.0f} {SPEED_UNIT} (game {best[2] + 1}, "
           f"{best[3]:.0f}–{best[4]:.0f} min after the start). " if best else "")
    sub += (f"Each bar is the average speed of that shot over {SPEED_BIN_MIN:.0f} minutes; "
            "light bars have fewer than 3 shots, so read them with care.")
    if title and not compact:
        sub += f"\n{title}"
    top = _header(fig, "Swing speed over time", sub, left_in=0.25, compact=compact)
    if len(games) > 1:
        top -= (0.45 if compact else 0.4) / h
    fig.subplots_adjust(left=left_in / w, right=1 - 0.15 / w, top=top - 0.2 / h, bottom=0.55 / h)
    if out:
        fig.savefig(out, facecolor=SURF)
    return fig


def plot_speed_by_game(rows, games, title="", out=None, fig=None, compact=False):
    """Simple grouped bars: average estimated racket-head speed (km/j) per stroke type and game."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    t, v, lab = _swings(rows)
    g = _game_of(t, games)
    n = len(games)
    colors = _game_colors(n)
    fig, ax = _canvas(fig, out, (10, 5.8))
    _style(ax)
    ax.tick_params(labelsize=9.5 if compact else 10)
    bw = min(0.24, 0.8 / n)
    avg = {}
    for si, k in enumerate(COLORS):
        for gi in range(n):
            m = (lab == k) & (g == gi)
            if not m.any():
                continue
            val = float(_kmh(v[m]).mean())
            avg[k, gi] = (val, int(m.sum()))
            x = si + (gi - (n - 1) / 2) * (bw + 0.02)
            ax.bar(x, val, bw, color=colors[gi], zorder=3)
            ax.text(x, val + 2, f"{val:.0f}", ha="center", va="bottom", color=INK, weight="bold",
                    fontsize=(8.5 if compact else 10) if n <= 4 else 7.5)
            if m.sum() < 5:  # flag averages of very few shots
                ax.text(x, val / 2, f"only\n{m.sum()}" if compact else f"only\n{m.sum()}\nshot{'s' if m.sum() > 1 else ''}",
                        ha="center", va="center", fontsize=7 if compact or n > 4 else 8.5, weight="bold",
                        color=INK if colors[gi] == BLUE_RAMP[0] else "white")
    ax.set_xticks(range(len(COLORS)))
    names = [STROKES[k][0].replace(" ", "\n") if compact else STROKES[k][0] for k in COLORS]
    ax.set_xticklabels([f"{s}\n({(lab == k).sum()}{'' if compact else ' shots'})" for s, k in zip(names, COLORS)],
                       color=INK)
    ax.set_ylabel(f"avg racket speed ({SPEED_UNIT}, est.)" if compact else f"average racket speed ({SPEED_UNIT}, estimated)",
                  color=INK2, fontsize=9 if compact else 9.5)
    top_val = max([a for a, _ in avg.values()] + [100])
    ax.set_ylim(0, np.ceil(top_val * 1.15 / 50) * 50)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    w, h = _size_in(fig)
    left_in = 0.7 if compact else 0.8
    top = _header(fig, "How fast you swung" + (", game by game" if n > 1 else ""), _speed_takeaway(avg, n),
                  left_in=0.25, compact=compact)
    if n > 1:
        ax.legend(handles=[Patch(color=colors[gi], label=f"Game {gi + 1}") for gi in range(n)], loc="lower left",
                  bbox_to_anchor=(0, 1.0), frameon=False, ncol=min(n, 6), fontsize=9 if compact else 10,
                  labelcolor=INK, handlelength=1.2, columnspacing=1.2)
        top -= 0.35 / h
    bottom = _footnote(fig, f"Speed estimated from wrist rotation (watch gyro × {RACKET_M} m to the racket head); "
                       "compare strokes and games, not exact values. Smashes read a little low (watch sensor limit)."
                       + (f"  {title}" if title and not compact else ""), left_in=0.25)
    fig.subplots_adjust(left=left_in / w, right=1 - 0.2 / w, top=top, bottom=bottom + (0.95 if compact else 0.6) / h)
    if out:
        fig.savefig(out, facecolor=SURF)
    return fig


def _speed_takeaway(avg, n):
    """One plain sentence: the fastest stroke and the biggest change from the first to the last game."""
    total = {}
    for (k, _), (val, c) in avg.items():
        s, cnt = total.get(k, (0.0, 0))
        total[k] = (s + val * c, cnt + c)
    if not total:
        return "No overhead strokes found."
    fast = max(total, key=lambda k: total[k][0] / total[k][1])
    msg = f"Fastest stroke: {STROKES[fast][0].lower()} (about {total[fast][0] / total[fast][1]:.0f} {SPEED_UNIT} on average)."
    # only strokes with enough shots in both games to compare
    ch = {k: avg[k, n - 1][0] - avg[k, 0][0] for k in COLORS
          if n > 1 and (k, 0) in avg and (k, n - 1) in avg and min(avg[k, 0][1], avg[k, n - 1][1]) >= 5}
    if ch:
        k = max(ch, key=lambda c: abs(ch[c]))
        if abs(ch[k]) >= 3:
            msg += (f" Biggest change from game 1 to game {n}: {STROKES[k][0].lower()} "
                    f"{'up' if ch[k] > 0 else 'down'} {abs(ch[k]):.0f} {SPEED_UNIT}.")
    return msg


def plot_stroke_timeline(rows, games, title="", out=None, fig=None, compact=False):
    """Simple bars: how many of each stroke per 2 minutes, one row per stroke plus all swings."""
    from matplotlib.ticker import MaxNLocator

    t, _, lab = _swings(rows)
    panels = [(None, "All swings", MUTED)] + [(k, STROKES[k][0], COLORS[k]) for k in COLORS]
    fig, axes = _canvas(fig, out, (12, 8.6), nrows=len(panels), ncols=1, sharex=True,
                        gridspec_kw={"hspace": 0.55 if compact else 0.35, "height_ratios": [1.2, 1, 1, 1, 1]})
    for ax, (k, name, col) in zip(axes, panels):
        _style(ax)
        ax.tick_params(labelsize=8.5)
        m = np.ones(len(t), bool) if k is None else lab == k
        for gi, (a, b) in enumerate(games):  # bins restart at each game so no bar straddles a rest
            edges = np.arange(a, b + TIMELINE_BIN_MIN, TIMELINE_BIN_MIN)
            cnt, _ = np.histogram(t[m], edges)
            ax.bar(edges[:-1] + TIMELINE_BIN_MIN / 2, cnt, TIMELINE_BIN_MIN * 0.82, color=col, zorder=3)
            if k is None and len(games) > 1:
                ax.text((a + b) / 2, 1.5 if compact else 1.18, f"Game {gi + 1}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=9.5 if compact else 10.5, color=INK, weight="bold")
        _shade_rests(ax, games)
        if ax is axes[-1] and not compact:
            for (_, b), (a, _) in zip(games[:-1], games[1:]):
                ax.text((a + b) / 2, 0.5, "rest", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=9, color=INK2)
        ax.set_ylim(0, max(1, ax.get_ylim()[1]) * 1.05)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        if compact:  # names above the panels: a label column would take a third of a phone screen
            ax.set_title(f"{name} · {m.sum()}", loc="left", fontsize=9, color=INK, weight="bold" if k else "normal",
                         pad=3)
        else:
            ax.text(-0.045, 0.5, f"{name}\n{m.sum()} total", transform=ax.transAxes, ha="right", va="center",
                    fontsize=10, color=INK, weight="bold" if k else "normal")
    axes[-1].set_xlim(0, (games[-1][1] if games else t.max()) + 0.5)
    note = f"each bar = shots in {TIMELINE_BIN_MIN:.0f} min"
    axes[-1].set_xlabel(f"minutes since start  ·  {note}" + ("  ·  shaded = rest" if compact and len(games) > 1
                                                             else f"  ·  {title}" if title else ""),
                        color=INK2, fontsize=8.5 if compact else 9)
    w, h = _size_in(fig)
    left_in = 0.45 if compact else 2.0
    top = _header(fig, "When you played each shot", _timeline_takeaway(t, lab, games), left_in=0.25,
                  compact=compact)
    top -= (0.55 if compact else 0.35) / h if len(games) > 1 else (0.2 / h if compact else 0)
    fig.subplots_adjust(left=left_in / w, right=1 - 0.15 / w, top=top, bottom=0.55 / h)
    if out:
        fig.savefig(out, facecolor=SURF)
    return fig


def _timeline_takeaway(t, lab, games):
    """Plain sentences: the most played stroke overall and per game, and the busiest game."""
    acc = lab != ""
    if not acc.any():
        return "No overhead strokes found."
    name = lambda k: STROKES[k][0].lower()  # noqa: E731
    counts = {k: int((lab == k).sum()) for k in COLORS}
    top = max(counts, key=counts.get)
    msg = f"Most played: {name(top)} ({100 * counts[top] / acc.sum():.0f}% of your overhead strokes)."
    if len(games) > 1:
        g = _game_of(t, games)
        per = []
        for gi in range(len(games)):
            c = {k: int(((lab == k) & (g == gi)).sum()) for k in COLORS}
            per.append(max(c, key=c.get) if sum(c.values()) else None)
        if len(set(per)) > 1:
            msg += " Top stroke per game: " + ", ".join(f"game {i + 1} {name(k)}" for i, k in enumerate(per) if k) + "."
        rate = [((g == gi).sum()) / max(b - a, 1e-9) for gi, (a, b) in enumerate(games)]
        busy = int(np.argmax(rate))
        msg += f"\nBusiest game: game {busy + 1} ({rate[busy]:.1f} swings per minute)."
    return msg


# One page with all four charts. Heights in inches per panel; desktop is a 2 x 2 grid,
# mobile a single narrow column (fonts stay in points, so they read about twice as large
# relative to the page on a phone).
REPORT = {
    "desktop": {"width": 24.0, "dpi": 100, "header": 1.25, "rows": [[("speed-simple", 1), ("timeline", 1)],
                                                                      [("hr", 1), ("speed", 1)]],
                "row_heights": [8.6, 9.8]},
    "mobile": {"width": 6.5, "dpi": 200, "header": 1.3, "rows": [[("speed-simple", 1)], [("timeline", 1)],
                                                                  [("hr", 1)], [("speed", 1)]],
               "row_heights": [6.4, 8.8, 6.0, 11.0]},
}


def plot_report(fit_path, rows, games, title="", model_name="", layout="desktop", out=None):
    """All four charts on one page: layout 'desktop' (2 x 2, landscape) or 'mobile' (one column)."""
    import matplotlib

    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = REPORT[layout]
    compact = layout == "mobile"
    heights = [spec["header"], *spec["row_heights"]]
    fig = plt.figure(figsize=(spec["width"], sum(heights)), dpi=spec["dpi"], facecolor=SURF)
    subs = fig.subfigures(len(heights), 1, height_ratios=heights, facecolor=SURF, hspace=0)
    _report_header(subs[0], fit_path, rows, games, model_name, compact)
    for sub, row in zip(subs[1:], spec["rows"]):
        cells = [sub] if len(row) == 1 else sub.subfigures(1, len(row), facecolor=SURF, wspace=0)
        for cell, (chart, _) in zip(cells, row):
            if chart == "hr":
                plot_session(fit_path, rows, model_name, fig=cell, compact=compact)
            elif chart == "speed":
                plot_swing_speed(rows, games, title, fig=cell, compact=compact)
            elif chart == "speed-simple":
                plot_speed_by_game(rows, games, title, fig=cell, compact=compact)
            else:
                plot_stroke_timeline(rows, games, title, fig=cell, compact=compact)
        # hairline between chart rows
        sub.add_artist(matplotlib.lines.Line2D([0.02, 0.98], [1, 1], transform=sub.transSubfigure, color=GRID, lw=1))
    if out:
        fig.savefig(out, facecolor=SURF)
    return fig


def _report_header(fig, fit_path, rows, games, model_name, compact):
    t, _, lab = _swings(rows)
    hr = [v for _, v in heart_rate(fit_path)]
    n = len(games)
    when = ""
    if rows and rows[0].get("clock"):
        start = next((ts for ts, _ in heart_rate(fit_path)), None)
        if start:
            lt, zone = local_time(fit_path, start)
            when = f"{lt:%d %b %Y, %H:%M} {zone}"
    stats = [when, f"{games[-1][1]:.0f} min" if games else "", f"{n} game{'s' if n != 1 else ''}",
             f"{len(rows)} swings", f"{int((lab != '').sum())} overhead strokes",
             f"avg heart rate {np.mean(hr):.0f} bpm" if hr else ""]
    name = os.path.basename(fit_path).removesuffix(".fit").removesuffix("_ACTIVITY")
    _header(fig, f"Badminton session {name}", "  ·  ".join(s for s in stats if s)
            + (f"\nmodel {model_name}" if model_name else ""), left_in=0.25, compact=compact,
            title_size=16 if compact else 20)


CHARTS = {"hr": "_hr_strokes.png", "speed": "_swing_speed.png", "speed-simple": "_swing_speed_simple.png",
          "timeline": "_stroke_timeline.png", "report-desktop": "_report_desktop.png",
          "report-mobile": "_report_mobile.png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit")
    ap.add_argument("--model", default="default")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--charts", nargs="+", choices=["all", *CHARTS], default=["all"],
                    help="which charts to draw (default: all)")
    ap.add_argument("--min-rest", type=float, default=2.0,
                    help="a gap in the IMU data of at least this many minutes is a rest between games")
    ap.add_argument("--out", help="PNG path of the heart-rate chart (default: <fit>_hr_strokes.png)")
    args = ap.parse_args()
    from .predict import MODELS_DIR

    mdir = args.models_dir or MODELS_DIR
    reg = registry(mdir)
    name = reg["default"] if args.model == "default" else args.model
    segments = load_fit(args.fit)
    acc, gyr, info = detect(args.fit, segments=segments)
    if not info:
        raise SystemExit("no swings detected; nothing to plot")
    gate = OverheadGate.load(os.path.join(mdir, reg["gate"]["file"]))
    model = load_model(name, mdir)
    set_balance(model, reg.get("balance_iters", 0))
    rows = classify(model, acc, gyr, info, gate.accept(acc, gyr))
    games = game_spans(segments, args.min_rest)
    base = os.path.splitext(args.fit)[0]
    label = f"{os.path.basename(base).removesuffix('_ACTIVITY')}, model {name}"
    charts = list(CHARTS) if "all" in args.charts else args.charts
    for c in charts:
        out = (args.out if c == "hr" and args.out else base + CHARTS[c])
        if c == "hr":
            plot_session(args.fit, rows, name, out)
        elif c == "speed":
            plot_swing_speed(rows, games, label, out)
        elif c == "speed-simple":
            plot_speed_by_game(rows, games, label, out)
        elif c == "timeline":
            plot_stroke_timeline(rows, games, label, out)
        else:
            plot_report(args.fit, rows, games, label, name, c.removeprefix("report-"), out)
        print(f"chart written to {out}")
        import matplotlib.pyplot as plt

        plt.close("all")


if __name__ == "__main__":
    main()
