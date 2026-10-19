"""Badak playstyle analyzer: Streamlit app.

    streamlit run app.py

Upload Garmin .fit files recorded with the badak watch app -> stroke detection and
classification (badminsens) -> playstyle attributes, archetype and skill tier (src.model)
-> racket and string-tension recommendations (src.recommendation). Past analyses are kept
per browser in SQLite (server) and mirrored into the browser's IndexedDB.
"""
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import model as pmodel
from src.history import History, compact
from src.parser import FEATURES, parse_fit, resolve_model
from src.recommendation import racket_svg, recommend

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache", "sessions")
ATTRS = pmodel.ATTRS
STROKE_ORDER = [("smash", "Smash"), ("clear", "Clear"), ("drop", "Drop"), ("backhand", "Backhand clear"),
                ("other", "Other swings")]

st.set_page_config(page_title="Badak Playstyle", page_icon="🏸", layout="wide")

# ------------------------------------------------------------------ theme tokens
DARK = (getattr(st.context, "theme", None) is not None and st.context.theme.type == "dark")
C = {
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"] if DARK
    else ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "muted": "#75746f" if DARK else "#b5b4ae",
    "ink": "#f0efec" if DARK else "#0b0b0b",
    "ink2": "#c3c2b7" if DARK else "#52514e",
    "grid": "#333331" if DARK else "#e4e3df",
}
STROKE_COLOR = dict(zip([k for k, _ in STROKE_ORDER], C["series"][:4] + [C["muted"]]))

st.markdown("""<style>
.badak-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 16px; height:100%;}
.badak-card h4 {margin:0 0 4px 0; font-size:1.02rem;}
.badak-card .sub {opacity:.72; font-size:.85rem; margin-bottom:8px;}
.badak-card table {width:100%; font-size:.86rem; border-collapse:collapse;}
.badak-card td {padding:2px 0; border:none;}
.badak-card td:last-child {text-align:right; font-variant-numeric:tabular-nums;}
.badak-row {display:flex; gap:14px; align-items:flex-start;}
.badak-pill {display:inline-block; padding:1px 9px; border-radius:99px; font-size:.78rem;
             border:1px solid rgba(128,128,128,.35); margin-right:4px;}
.badak-hero {font-size:2.1rem; font-weight:650; line-height:1.1; margin:2px 0 6px 0;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ browser bridge (id + IndexedDB mirror)
BRIDGE_JS = """
export default function(component) {
  const { data, setStateValue } = component;
  const last = (window.__badak = window.__badak || {});
  const send = (k, v) => { const s = JSON.stringify(v); if (last[k] !== s) { last[k] = s; setStateValue(k, v); } };
  let uid = null;
  try { uid = localStorage.getItem('badak_uid');
        if (!uid) { uid = crypto.randomUUID(); localStorage.setItem('badak_uid', uid); } } catch (e) {}
  if (!uid) { uid = last.uid || (last.uid = crypto.randomUUID()); }
  send('uid', uid);
  if (!window.indexedDB) { send('ids', []); return; }
  const req = indexedDB.open('badak', 1);
  req.onupgradeneeded = () => req.result.createObjectStore('sessions', { keyPath: 'id' });
  req.onerror = () => send('ids', []);
  req.onsuccess = () => {
    const db = req.result;
    const tx = db.transaction('sessions', 'readwrite');
    const store = tx.objectStore('sessions');
    const d = data || {};
    if (d.clear) store.clear();
    (d.del || []).forEach((id) => store.delete(id));
    (d.put || []).forEach((rec) => store.put(rec));
    tx.oncomplete = () => {
      const r = db.transaction('sessions').objectStore('sessions').getAll();
      r.onsuccess = () => {
        send('ids', r.result.map((x) => x.id).sort());
        if (d.want) send('records', r.result);
      };
    };
  };
}
"""
_bridge = st.components.v2.component("badak_bridge", js=BRIDGE_JS)


@st.cache_resource
def history():
    return History()


@st.cache_resource
def playstyle():
    return pmodel.load()


ss = st.session_state
ss.setdefault("pending_put", {})
ss.setdefault("pending_del", set())
ss.setdefault("clear_idb", False)
ss.setdefault("restored", False)

with st.sidebar:
    bridge = _bridge(key="bridge", data={"put": list(ss.pending_put.values()), "del": sorted(ss.pending_del),
                                         "clear": ss.clear_idb, "want": not ss.restored},
                     height=0, on_uid_change=lambda: None, on_ids_change=lambda: None,
                     on_records_change=lambda: None)
uid = bridge.get("uid") if bridge else None
idb_ids = set(bridge.get("ids") or []) if bridge else set()
# what IndexedDB now holds is no longer pending
ss.pending_put = {k: v for k, v in ss.pending_put.items() if k not in idb_ids}
ss.pending_del = {k for k in ss.pending_del if k in idb_ids}
if ss.clear_idb and not idb_ids:
    ss.clear_idb = False

H = history()
M = playstyle()
if uid is None:
    st.info("Connecting to your browser storage...")
    st.stop()

# restore from IndexedDB when the server has nothing for this browser (e.g. after a redeploy)
if not ss.restored and bridge.get("records") is not None:
    ss.restored = True
    have = {r["id"] for r in H.all(uid)}
    for rec in bridge.get("records") or []:
        if rec.get("schema") == 1 and rec["id"] not in have:
            H.save(uid, rec)
records = H.all(uid)
# mirror anything IndexedDB is missing
for r in records:
    if r["id"] not in idb_ids and r["id"] not in ss.pending_put and r["id"] not in ss.pending_del:
        ss.pending_put[r["id"]] = r


# ------------------------------------------------------------------ helpers
def label(rec):
    t = rec.get("start_local")
    when = datetime.fromisoformat(t).strftime("%a %d %b %Y, %H:%M") if t else rec["file_name"]
    return f"{when} · {rec['activity_id']}"


def predict(features):
    return M.predict(features)


def mean_features(recs):
    keys = recs[0]["features"].keys()
    out = {}
    for k in keys:
        v = [r["features"][k] for r in recs if r["features"].get(k) is not None and not _nan(r["features"][k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def _nan(x):
    return isinstance(x, float) and math.isnan(x)


def fmt(x, nd=1):
    return "–" if x is None or _nan(x) else f"{x:,.{nd}f}"


def base_layout(fig, height=380, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["ink2"], size=13),
                      legend=dict(orientation="h", y=-0.12, x=0), hoverlabel=dict(font_size=13), **kw)
    return fig


def analyze_uploads(files):
    os.makedirs(os.path.join(ROOT, ".cache", "uploads"), exist_ok=True)
    added, errors = [], []
    prog = st.progress(0.0, text="Starting…")
    for i, f in enumerate(files):
        prog.progress(i / len(files), text=f"Analyzing {f.name} ({i + 1}/{len(files)}) — up to a minute per hour of play")
        with tempfile.NamedTemporaryFile(suffix=".fit", dir=os.path.join(ROOT, ".cache", "uploads"),
                                         prefix=os.path.splitext(f.name)[0] + "_", delete=False) as tmp:
            tmp.write(f.getbuffer())
            path = tmp.name
        try:
            s = parse_fit(path, cache_dir=CACHE)
            s["file_name"] = f.name
            s["activity_id"] = f.name.split("_")[0].split(".")[0]
            if len(s["swings"]) < 20:
                errors.append(f"{f.name}: only {len(s['swings'])} swings found — is this a badminton recording "
                              "with raw sensor logging on?")
                continue
            rec = compact(s, predict(s["features"]))
            H.save(uid, rec)
            ss.pending_put[rec["id"]] = rec
            ss.pending_del.discard(rec["id"])
            added.append(rec["id"])
        except Exception as e:  # noqa: BLE001 - shown to the user
            errors.append(f"{f.name}: {e}")
        finally:
            os.remove(path)
    prog.empty()
    return added, errors


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🏸 Badak Playstyle")
    st.caption("Garmin .fit recordings from the [badak](https://github.com/omayib/badak) watch app "
               "(raw accelerometer + gyroscope, racket wrist).")
    up = st.file_uploader("Upload .fit files", type=["fit"], accept_multiple_files=True, key=f"up{ss.get('upn', 0)}")
    if up and st.button(f"Analyze {len(up)} file{'s' * (len(up) > 1)}", type="primary", width="stretch"):
        added, errors = analyze_uploads(up)
        for e in errors:
            st.error(e)
        if added:
            ss.focus = added[-1]
            ss.upn = ss.get("upn", 0) + 1
            st.rerun()

    if records:
        st.divider()
        st.subheader(f"History ({len(records)})")
        by_id = {r["id"]: r for r in records}
        ids = [r["id"] for r in records]
        if ss.get("focus") not in by_id:
            ss.focus = ids[-1]
        focus_id = st.selectbox("Focus session", ids[::-1], format_func=lambda i: label(by_id[i]), key="focus")
        with st.expander(f"Sessions in trends and averages ({len(ss.get('compare', ids))} of {len(ids)})"):
            compare = st.multiselect("Sessions", ids, default=ids, format_func=lambda i: label(by_id[i]),
                                     key="compare", label_visibility="collapsed")
        with st.expander("Manage history"):
            st.download_button("Export history (.json)", json.dumps(records), "badak_history.json",
                               "application/json", width="stretch")
            imp = st.file_uploader("Import history (.json)", type=["json"], key="imp")
            if imp is not None and st.button("Import", width="stretch"):
                n = 0
                for rec in json.load(imp):
                    if isinstance(rec, dict) and rec.get("schema") == 1:
                        H.save(uid, rec)
                        n += 1
                st.success(f"Imported {n} sessions")
                st.rerun()
            if st.button("Delete focus session", width="stretch"):
                H.delete(uid, focus_id)
                ss.pending_put.pop(focus_id, None)
                ss.pending_del.add(focus_id)
                del ss["focus"]
                st.rerun()
            if st.button("Delete all history", width="stretch"):
                H.clear(uid)
                ss.pending_put, ss.pending_del, ss.clear_idb = {}, set(), True
                st.rerun()
        st.caption("History is kept on this server for this browser and mirrored in your browser's IndexedDB.")
    st.caption("Uploaded .fit files are deleted right after analysis; only the derived summary is kept.")

if not records:
    st.title("Your badminton playstyle, from your watch")
    st.markdown(
        "Upload one or more Garmin `.fit` files in the sidebar. For each session the app\n\n"
        "1. finds every swing and classifies the overhead strokes (smash, clear, drop, backhand clear),\n"
        "2. scores **Power, Speed, Control, Endurance and Agility** and places you on the "
        "Attacking / Defensive / Control / All-round map,\n"
        "3. estimates a skill tier and suggests a **racket profile and string tension**.\n\n"
        "Analyses are saved, so next time you can browse them without uploading again.")
    st.stop()

focus = by_id[focus_id]
pred = predict(focus["features"])
sel = [by_id[i] for i in compare] or [focus]

# ------------------------------------------------------------------ header
st.title("Playstyle analysis")
f = focus["features"]
m = focus["meta"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Session", datetime.fromisoformat(focus["start_local"]).strftime("%d %b") if focus["start_local"] else "–",
          help=f"{label(focus)} · {focus['file_name']}")
c2.metric("Active play", f"{f['active_min']:.0f} min", help=f"{len(focus['games'])} game block(s); rests excluded")
c3.metric("Swings", f"{len(focus['swings'])}", help=f"{f['swings_per_min']:.1f} per minute of play")
c4.metric("Avg heart rate", f"{m.get('avg_heart_rate') or '–'} bpm", help=f"max {m.get('max_heart_rate') or '–'} bpm")
c5.metric("Training effect", f"{fmt(m.get('total_training_effect'))} / {fmt(m.get('total_anaerobic_training_effect'))}",
          help="Garmin aerobic / anaerobic training effect (0–5)")

tab_style, tab_trend, tab_gear, tab_sess, tab_about = st.tabs(
    ["Playstyle", "Trends", "Equipment", "Session details", "Method"])

# ------------------------------------------------------------------ playstyle
with tab_style:
    left, right = st.columns([1.35, 1])
    with left:
        fig = go.Figure()
        theta = ATTRS + [ATTRS[0]]
        if len(sel) > 1:
            avg = predict(mean_features(sel))["attributes"]
            fig.add_trace(go.Scatterpolar(r=[avg[a] for a in theta], theta=theta, name=f"Average of {len(sel)} sessions",
                                          line=dict(color=C["muted"], width=2, dash="dot"), fill="none",
                                          hovertemplate="%{theta}: %{r:.0f}<extra>average</extra>"))
        fig.add_trace(go.Scatterpolar(r=[pred["attributes"][a] for a in theta], theta=theta, name="Focus session",
                                      line=dict(color=C["series"][0], width=2), fill="toself",
                                      fillcolor=C["series"][0] + "33", marker=dict(size=8),
                                      hovertemplate="%{theta}: %{r:.0f}<extra>focus session</extra>"))
        fig.update_polars(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], gridcolor=C["grid"],
                                          linecolor=C["grid"], tickfont=dict(size=10, color=C["ink2"])),
                          angularaxis=dict(rotation=90, direction="clockwise", gridcolor=C["grid"], linecolor=C["grid"], tickfont=dict(size=14, color=C["ink"])),
                          bgcolor="rgba(0,0,0,0)")
        base_layout(fig, 430, showlegend=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.caption("Archetype")
        st.markdown(f'<div class="badak-hero">{pred["archetype"]}</div>', unsafe_allow_html=True)
        st.write(pred["description"])
        if pred["skill_tier"]:
            st.markdown(f'<span class="badak-pill">Skill tier: <b>{pred["skill_tier"]}</b></span>'
                        f'<span class="badak-pill">index {pred["skill_index"]:+.2f}</span>', unsafe_allow_html=True)
        probs = pred["archetype_probs"]
        names = list(probs)[::-1]
        bar = go.Figure(go.Bar(x=[probs[n] * 100 for n in names], y=names, orientation="h",
                               marker=dict(color=[C["series"][0] if n == pred["archetype"] else C["muted"] for n in names],
                                           cornerradius=4),
                               text=[f"{probs[n]:.0%}" for n in names], textposition="outside",
                               hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        base_layout(bar, 220, xaxis=dict(range=[0, 115], visible=False), yaxis=dict(tickfont=dict(color=C["ink"])),
                    showlegend=False, bargap=0.35)
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
        st.caption("Probabilities are calibrated against bootstrap resampling of your swings (see Method).")

    cols = st.columns(5)
    for col, a in zip(cols, ATTRS):
        delta = None
        if len(sel) > 1:
            delta = f"{pred['attributes'][a] - avg[a]:+.0f} vs avg"
        col.metric(a, f"{pred['attributes'][a]:.0f}", delta, delta_color="off" if delta is None else "normal")

    with st.expander("What drives each score"):
        comp = pd.DataFrame(M.components(f))
        comp["description"] = comp["feature"].map(FEATURES)
        comp["value"] = comp["value"].map(lambda v: fmt(v, 2))
        comp["range (10 → 90)"] = comp.apply(lambda r: f"{r['lo']} → {r['hi']}", axis=1)
        st.dataframe(comp[["attribute", "description", "value", "range (10 → 90)", "weight", "score"]],
                     hide_index=True, width="stretch")
        if f.get("saturated_share", 0) > 0.1:
            st.caption(f"{f['saturated_share']:.0%} of swings hit the watch gyroscope's 2000 °/s limit, so the "
                       "hardest swings read lower than they were: Power is a lower bound.")

# ------------------------------------------------------------------ trends
with tab_trend:
    if len(sel) < 2:
        st.info("Select at least two sessions in the sidebar to see trends.")
    else:
        rows = []
        for r in sel:
            p = predict(r["features"])
            rows.append({"date": datetime.fromisoformat(r["start_local"]), "session": r["activity_id"],
                         "archetype": p["archetype"], "skill": p["skill_tier"], **p["attributes"],
                         "smash speed (°/s)": r["features"]["smash_p75_dps"],
                         "swings / min": r["features"]["swings_per_min"],
                         "% HR reserve": r["features"]["hr_avg_pct_hrr"],
                         "active min": r["features"]["active_min"]})
        df = pd.DataFrame(rows).sort_values("date")
        attr_pick = st.segmented_control("Attributes", ATTRS, default=ATTRS, selection_mode="multi")
        fig = go.Figure()
        shown = [a for a in ATTRS if a in (attr_pick or [])]
        # direct end labels, nudged apart so they never overlap (>= 5 points in score units)
        ends = sorted(shown, key=lambda a: df[a].iloc[-1])
        ypos, prev = {}, -1e9
        for a in ends:
            ypos[a] = prev = max(df[a].iloc[-1], prev + 5)
        for i, a in enumerate(ATTRS):  # colour follows the attribute, never its rank
            if a not in shown:
                continue
            fig.add_trace(go.Scatter(x=df["date"], y=df[a], name=a, mode="lines+markers",
                                     line=dict(color=C["series"][i], width=2), marker=dict(size=8, line=dict(width=2, color="white" if not DARK else "#1a1a19")),
                                     hovertemplate="%{x|%d %b}: %{y:.0f}<extra>" + a + "</extra>"))
            fig.add_annotation(x=df["date"].iloc[-1], y=ypos[a], text=a, showarrow=False, xanchor="left",
                               xshift=10, font=dict(color=C["ink2"], size=12))
        base_layout(fig, 420, hovermode="x unified", yaxis=dict(range=[0, 100], gridcolor=C["grid"], title="score"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        fig.update_layout(margin=dict(r=90))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        arche = Counter(df["archetype"])
        st.markdown("**Archetype per session:** " + " · ".join(f"{k} ×{v}" for k, v in arche.most_common()))
        show = df.copy()
        show["date"] = show["date"].dt.strftime("%d %b %H:%M")
        st.dataframe(show.round(1), hide_index=True, width="stretch")

        st.subheader("Game by game (focus session)")
        if focus["per_game"]:
            g = pd.DataFrame([{"game": x["game"], "minutes": round(x["end_min"] - x["start_min"], 1),
                               **predict(x["features"])["attributes"],
                               "archetype": predict(x["features"])["archetype"]} for x in focus["per_game"]])
            st.dataframe(g, hide_index=True, width="stretch")
            st.caption("Single games are short, so their scores move more than session scores "
                       "(see the per-game agreement figure in Method).")

# ------------------------------------------------------------------ equipment
with tab_gear:
    opts = ["Focus session"] + ([f"Average of {len(sel)} selected sessions"] if len(sel) > 1 else [])
    basis = st.radio("Base the recommendation on", opts, horizontal=True, index=len(opts) - 1)
    if basis.startswith("Average"):
        feats = mean_features(sel)
    else:
        feats = f
    p = predict(feats)
    rec = recommend(p, feats)
    st.caption(f"Profile used: **{p['archetype']}**, skill tier **{p['skill_tier']}** · "
               + ", ".join(f"{a} {p['attributes'][a]:.0f}" for a in ATTRS))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Balance", rec["balance"])
    k1.caption(f"balance point {rec['balance_mm'][0]}–{rec['balance_mm'][1]} mm")
    k2.metric("Weight class", rec["weight"])
    k2.caption(f"{rec['weight_g'][0]:.0f}–{rec['weight_g'][1]:.0f} g unstrung")
    k3.metric("Shaft", rec["shaft"])
    k3.caption("flex")
    k4.metric("String tension", f"{rec['tension_lbs'][0]}–{rec['tension_lbs'][1]} lbs")
    k4.caption(f"≈ {rec['tension_lbs'][0] * 0.4536:.1f}–{rec['tension_lbs'][1] * 0.4536:.1f} kg")
    with st.expander("Why these settings", expanded=True):
        for key, title in [("balance", "Balance"), ("weight", "Weight"), ("shaft", "Shaft"), ("tension", "Tension")]:
            st.markdown(f"**{title}.** {rec['reasons'][key]}  \n<small>Evidence: {rec['evidence'][key]}</small>",
                        unsafe_allow_html=True)

    st.subheader("Racket profiles that fit")
    cards = st.columns(3)
    for col, r in zip(cards, rec["rackets"]):
        svg = racket_svg(r, accent=C["series"][0], ink=C["ink2"])
        col.markdown(f"""<div class="badak-card"><div class="badak-row"><div>{svg}</div><div style="flex:1">
<h4>{r['name']}</h4><div class="sub">{r['suits']}</div>
<span class="badak-pill">{r['match']:.0%} match</span>
<table>
<tr><td>Balance</td><td>{r['balance']}</td></tr>
<tr><td>Balance point</td><td>{r['balance_mm'][0]}–{r['balance_mm'][1]} mm</td></tr>
<tr><td>Weight</td><td>{r['weight']} · {r['weight_g'][0]:.0f}–{r['weight_g'][1]:.0f} g</td></tr>
<tr><td>Shaft</td><td>{r['shaft']}</td></tr>
<tr><td>Head</td><td>{r['head']}</td></tr>
<tr><td>String at</td><td>{rec['tension_lbs'][0]}–{rec['tension_lbs'][1]} lbs</td></tr>
</table></div></div></div>""", unsafe_allow_html=True)
    st.caption("Diagram: shading of the head grows with head weight, the orange dot marks the balance point "
               "(mm from the butt of a 675 mm frame), the shaft line is thicker when stiffer. "
               + rec["caveat"])

# ------------------------------------------------------------------ session details
with tab_sess:
    sw = pd.DataFrame(focus["swings"], columns=["time_s", "game", "code", "kmh", "saturated"])
    a, b = st.columns([1, 1.4])
    with a:
        st.subheader("Stroke mix")
        cnt = sw["code"].value_counts()
        names = [n for _, n in STROKE_ORDER][::-1]
        codes = [c for c, _ in STROKE_ORDER][::-1]
        vals = [int(cnt.get(c, 0)) for c in codes]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker=dict(color=[STROKE_COLOR[c] for c in codes],
                                                                              cornerradius=4),
                               text=[f"{v} ({v / len(sw):.0%})" for v in vals], textposition="outside",
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        base_layout(fig, 300, xaxis=dict(visible=False, range=[0, max(vals) * 1.3]), showlegend=False, bargap=0.3,
                    yaxis=dict(tickfont=dict(color=C["ink"])))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("“Other swings” are lifts, net shots, drives, serves and defence: the stroke model only knows "
                   "the four overhead strokes of the BADS dataset.")
    with b:
        st.subheader("Heart rate")
        hr = pd.DataFrame(focus["hr"], columns=["min", "bpm"])
        fig = go.Figure()
        for i, (g0, g1) in enumerate(focus["games"]):
            fig.add_vrect(x0=g0, x1=g1, fillcolor=C["series"][0], opacity=0.07, line_width=0,
                          annotation_text=f"game {i + 1}" if len(focus["games"]) > 1 else "", annotation_position="top left",
                          annotation_font=dict(size=11, color=C["ink2"]))
        brk = hr["min"].diff() > 1
        hr.loc[brk, "bpm"] = None
        fig.add_trace(go.Scatter(x=hr["min"], y=hr["bpm"], mode="lines", line=dict(color=C["ink2"], width=2),
                                 connectgaps=False, hovertemplate="%{x:.0f} min: %{y} bpm<extra></extra>"))
        base_layout(fig, 300, showlegend=False, xaxis=dict(title="minutes since start", gridcolor=C["grid"]),
                    yaxis=dict(title="bpm", gridcolor=C["grid"]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Racket speed by stroke and game")
    agg = sw.groupby(["code", "game"])["kmh"].mean().reset_index()
    fig = go.Figure()
    for c, n in STROKE_ORDER:
        d = agg[agg["code"] == c]
        fig.add_trace(go.Bar(x=[f"game {g}" for g in d["game"]], y=d["kmh"], name=n,
                             marker=dict(color=STROKE_COLOR[c], cornerradius=4),
                             hovertemplate=n + " · %{x}: %{y:.0f} km/h<extra></extra>"))
    base_layout(fig, 340, barmode="group", bargap=0.25, bargroupgap=0.08,
                yaxis=dict(title="avg racket-head speed (km/h, est.)", gridcolor=C["grid"]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Speed is estimated from peak wrist rotation × 0.7 m (wrist to racket head).")

# ------------------------------------------------------------------ method
with tab_about:
    v = M.validation
    st.markdown("""
**Pipeline.** Swings are found as gyroscope peaks (≥ 800 °/s). An overhead gate keeps the swings that look like
one of the four BADS strokes, and a classifier trained on BADS (12 players) labels them. Everything else is
“Other”. Session features (stroke shares, swing speed, tempo, rally length, Markov transition entropy between
stroke types, heart-rate load, speed retention, movement between swings) feed five attribute scores.
Each feature has an anchor range that maps to 10–90 points, and an attribute is a weighted mean of its features.

**Archetype.** The attribute profile is centred on its own mean (your relative strengths) and compared with
prototype shapes. Attacking leans on Power, Defensive on Speed and Agility, Control on Control. A flat
profile is All-round. The softmax temperature was tuned so the top probability matches how often the archetype
survives block-bootstrap resampling of the swings.

**Skill tier.** A skill index (smash speed, backhand speed, smash–drop speed gap, gyro clipped to the Garmin's
±2000 °/s range) was fitted on the BADS players' Low / Medium / High levels and validated leave-one-player-out.
""")
    stroke_model = resolve_model()
    st.markdown(f"- Stroke model on this server: **{'SVM + CNN + Transformer ensemble' if stroke_model == 'default' else stroke_model.upper()}**"
                + ("" if stroke_model == "default" else " (no PyTorch here; 87.1% vs the ensemble's 87.5%, leave-one-player-out)."))
    if v.get("skill_loso"):
        s = v["skill_loso"]
        st.markdown(f"- Skill tier, leave-one-player-out on {s['players']} BADS players: **{s['exact_accuracy']:.0%}** exact "
                    f"(majority baseline {s['majority_baseline']:.0%}), {s['within_one_level']:.0%} within one level, "
                    f"Spearman ρ = {s['spearman_rho_heldout_index_vs_level']:.2f} (p = {s['spearman_p']}).")
    if v.get("archetype"):
        a = v["archetype"]
        st.markdown(f"- Archetype temperature T = {a['chosen_T']:g}, tuned on {a['n_sessions']} sessions × "
                    f"{a['n_bootstrap']} bootstraps; a single game agrees with its whole session "
                    f"{a['per_game_agreement_with_session']:.0%} of the time.")
        st.markdown("- Typical bootstrap spread of a score (SD, points): "
                    + ", ".join(f"{k} {x:.1f}" for k, x in a["bootstrap_attribute_sd"].items()))
    st.markdown("""
**Limits.** There are no labelled playstyles for these recordings, so the archetype is a transparent,
knowledge-based mapping, not a trained classifier. The skill index comes from 12 players on a different watch.
Equipment links to play mode are mostly coaching consensus. The best-supported piece is that string tension
affects shuttle speed and that skill moderates it (Vanasant 2013; Zhu 2013, 2014).
""")
