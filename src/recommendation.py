"""Rule-based equipment mapping: playstyle + skill + intensity -> racket and string tension.

The rules follow references/badminton-play-mode-equipments.md. Equipment *tendencies* per
mode are coaching consensus (balance, weight) or partial mechanical evidence (shaft); the
string-tension rule is the best supported part:
  * lower tension -> higher shuttle velocity, 22 > 24 > 26 > 28 > 30 lb (Vanasant et al. 2013)
  * experts adapt to any tension, novices are constrained by it (Zhu 2013, 2014)
so tension is set primarily by the skill tier, then nudged by playstyle and fatigue.
Every recommendation carries its rationale and evidence strength.
"""
from dataclasses import asdict, dataclass, field

BALANCE_MM = {"Head-heavy": (295, 310), "Even balance": (285, 295), "Head-light": (270, 285)}
WEIGHT_G = {"3U": (85, 89.9), "4U": (80, 84.9), "5U": (75, 79.9)}
BASE_TENSION = {"Low": (21, 24), "Medium": (24, 27), "High": (27, 30)}
EVIDENCE = {
    "balance": "Coaching consensus (ActiveSG, BadmintonHQ); not a formal tactical theory",
    "weight": "Practitioner knowledge; little direct research",
    "shaft": "Partial mechanical evidence (Chalmers simulation report)",
    "tension": "Moderate-to-strong for shuttle velocity (Vanasant 2013) and expertise interaction (Zhu 2013)",
}


@dataclass
class RacketProfile:
    key: str
    name: str
    balance: str
    weight: str
    shaft: str
    head: str
    suits: str
    balance_mm: tuple = field(default=(0, 0))
    weight_g: tuple = field(default=(0, 0))


# generic racket profiles (no brands): one per balance x shaft combination that is common
CATALOG = [
    RacketProfile("hh-stiff", "Rear-court attack frame", "Head-heavy", "3U", "Stiff", "Isometric",
                  "Fast, well-timed full smashes; advanced attackers"),
    RacketProfile("hh-medium", "Power all-rounder", "Head-heavy", "4U", "Medium", "Isometric",
                  "Attack-minded players still building swing speed"),
    RacketProfile("even-medium", "Balanced all-court frame", "Even balance", "4U", "Medium", "Isometric",
                  "All-round construction: mix of attack, defence and net play"),
    RacketProfile("even-stiff", "Control precision frame", "Even balance", "4U", "Stiff", "Isometric",
                  "Placement and deception for players with a fast, consistent swing"),
    RacketProfile("hl-flex", "Speed defence frame", "Head-light", "5U", "Flexible", "Isometric",
                  "Quick drives, blocks and net exchanges; easy repulsion"),
    RacketProfile("hl-medium", "Fast-handling doubles frame", "Head-light", "4U", "Medium", "Isometric",
                  "Front-court interception and flat exchanges"),
    RacketProfile("even-flex", "Forgiving starter frame", "Even balance", "5U", "Flexible", "Isometric",
                  "Players developing technique; shaft helps shuttle depth"),
]
for _r in CATALOG:
    _r.balance_mm, _r.weight_g = BALANCE_MM[_r.balance], WEIGHT_G[_r.weight]


def recommend(prediction, features):
    """prediction: src.model.PlaystyleModel.predict output; features: session feature dict."""
    a = prediction["attributes"]
    probs = prediction["archetype_probs"]
    style = prediction["archetype"]
    tier = prediction.get("skill_tier") or "Medium"
    reasons = {}

    # --- balance: attack affinity vs speed/defence affinity
    attack = probs.get("Attacking", 0) + 0.5 * max(0.0, (a["Power"] - (a["Speed"] + a["Agility"]) / 2) / 40)
    defend = probs.get("Defensive", 0) + 0.5 * max(0.0, ((a["Speed"] + a["Agility"]) / 2 - a["Power"]) / 40)
    if attack >= 0.5 and attack > defend:
        balance = "Head-heavy"
        reasons["balance"] = (f"Attacking affinity {attack:.0%}: more head mass carries momentum into smashes "
                              f"and clears (Power {a['Power']:.0f}).")
    elif defend >= 0.5:
        balance = "Head-light"
        reasons["balance"] = (f"Defensive affinity {defend:.0%}: a light head swings faster for blocks, drives "
                              f"and net play (Speed {a['Speed']:.0f}, Agility {a['Agility']:.0f}).")
    else:
        balance = "Even balance"
        reasons["balance"] = (f"No dominant attack ({attack:.0%}) or defence ({defend:.0%}) affinity: even "
                              f"balance trades power against handling.")

    # --- weight class
    if balance == "Head-heavy" and tier == "High" and a["Endurance"] >= 50:
        weight = "3U"
        reasons["weight"] = "Fast swing and enough stamina to carry extra mass through long games."
    elif balance == "Head-light" or tier == "Low" or a["Endurance"] < 35:
        weight = "5U"
        reasons["weight"] = ("Lighter frame keeps swing speed up late in the session"
                             + (f" (Endurance {a['Endurance']:.0f})." if a["Endurance"] < 35 else "."))
    else:
        weight = "4U"
        reasons["weight"] = "4U is the common middle ground for club players."

    # --- shaft stiffness: stiff rewards fast, well-timed swings; flex helps slower swings
    smash = features.get("smash_p75_dps") or 0
    if tier == "High" and smash >= 2800:
        shaft = "Stiff"
        reasons["shaft"] = f"Smash wrist speed {smash:.0f} deg/s (BADS High-level range): a stiff shaft returns it with precise timing."
    elif tier == "Low" or smash < 2550:
        shaft = "Flexible"
        reasons["shaft"] = "A flexible shaft adds kick for slower or still-developing swings."
    else:
        shaft = "Medium"
        reasons["shaft"] = "Medium flex: some kick without losing timing."

    # --- string tension: skill tier sets the range, style/fatigue nudge it
    lo, hi = BASE_TENSION[tier]
    notes = [f"{tier} skill tier -> base {lo}-{hi} lbs (experts handle higher tension; Zhu 2013)."]
    if style == "Control" or a["Control"] >= 65:
        lo, hi = lo + 1, hi + 1
        notes.append("+1 lb for placement and a crisper feel (Control-leaning).")
    if style == "Attacking" and tier != "High":
        lo, hi = lo - 1, hi - 1
        notes.append("-1 lb: lower tension gives more shuttle speed for the same swing (Vanasant 2013).")
    if a["Endurance"] < 35:
        lo, hi = lo - 1, hi - 1
        notes.append("-1 lb: a larger sweet spot forgives the off-centre hits of a tired arm.")
    lo, hi = max(18, lo), min(32, hi)
    reasons["tension"] = " ".join(notes)

    target = {"balance": balance, "weight": weight, "shaft": shaft}
    ranked = sorted(CATALOG, key=lambda r: -_match(r, target))
    rackets = [{**asdict(r), "match": round(_match(r, target) / 3, 2)} for r in ranked[:3]]
    return {
        "balance": balance, "balance_mm": BALANCE_MM[balance], "weight": weight, "weight_g": WEIGHT_G[weight],
        "shaft": shaft, "tension_lbs": (lo, hi), "reasons": reasons, "evidence": EVIDENCE, "rackets": rackets,
        "caveat": ("Equipment does not determine play mode; it changes the cost and timing of stroke patterns. "
                   "Treat these as a starting point to test, not a prescription."),
    }


def _match(r, t):
    order = {"balance": ["Head-light", "Even balance", "Head-heavy"], "weight": ["5U", "4U", "3U"],
             "shaft": ["Flexible", "Medium", "Stiff"]}
    return sum(1 - abs(order[k].index(getattr(r, k)) - order[k].index(t[k])) / 2 for k in order)


def racket_svg(r, accent="#2a78d6", ink="#52514e", width=96):
    """Racket silhouette: head shading grows with head weight, a dot marks the balance point."""
    lo, hi = r["balance_mm"]
    bp = (lo + hi) / 2  # mm from the butt; the frame is 675 mm long
    h = 300
    y = lambda mm: h - 8 - (mm / 675) * (h - 16)  # noqa: E731  butt at the bottom
    heavy = {"Head-heavy": 0.55, "Even balance": 0.32, "Head-light": 0.14}[r["balance"]]
    shaft_w = {"Stiff": 3.2, "Medium": 2.6, "Flexible": 2.0}[r["shaft"]]
    cx = 60
    strings = "".join(f'<line x1="{cx - 30}" y1="{y(675 - 12) + 8 + i * 12}" x2="{cx + 30}" y2="{y(675 - 12) + 8 + i * 12}"/>'
                      for i in range(8))
    strings += "".join(f'<line x1="{cx - 24 + i * 12}" y1="{y(675) + 6}" x2="{cx - 24 + i * 12}" y2="{y(430) - 6}"/>'
                       for i in range(5))
    return f"""<svg viewBox="0 0 120 {h}" width="{width}" role="img" aria-label="{r['name']} racket diagram">
  <g fill="none" stroke="{ink}" stroke-linecap="round">
    <ellipse cx="{cx}" cy="{(y(675) + y(430)) / 2}" rx="36" ry="{(y(430) - y(675)) / 2}" stroke-width="4"
             stroke="{accent}" fill="{accent}" fill-opacity="{heavy}"/>
    <g stroke-width="0.8" stroke-opacity="0.55">{strings}</g>
    <line x1="{cx}" y1="{y(430)}" x2="{cx}" y2="{y(200)}" stroke-width="{shaft_w}"/>
    <rect x="{cx - 7}" y="{y(200)}" width="14" height="{y(0) - y(200)}" rx="5" fill="{ink}" fill-opacity="0.18" stroke-width="1.5"/>
  </g>
  <circle cx="{cx}" cy="{y(bp)}" r="6" fill="#eb6834" stroke="#fff" stroke-width="2"/>
  <text x="{cx + 12}" y="{y(bp) + 4}" font-size="11" fill="{ink}" font-family="sans-serif">{bp:.0f} mm</text>
</svg>"""
