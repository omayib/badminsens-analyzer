---
title: Badak Playstyle
emoji: 🏸
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-nc-nd-4.0
models:
- omayib/badak
short_description: Badminton playstyle and racket advice from Garmin .fit files
---

# Badak Playstyle

Upload Garmin `.fit` recordings made with the [badak](https://github.com/omayib/badak) watch app
(raw accelerometer + gyroscope, watch on the racket wrist, right-handed). For each session the app:

1. finds every swing and classifies the overhead strokes (smash, clear, drop, backhand clear),
2. scores **Power, Speed, Control, Endurance and Agility** and places you on the
   Attacking / Defensive / Control / All-round map, with a skill tier,
3. suggests a racket profile (balance, weight, shaft) and string tension, with the reasoning and
   evidence strength behind each.

Past analyses stay available for your browser: they are kept on the Space and mirrored in your
browser's IndexedDB, so they come back even after the Space restarts. Uploaded files are deleted
right after analysis.

Models, method and limits: [omayib/badak](https://huggingface.co/omayib/badak). Trained on the
BADS dataset from BadminSense (Chen et al., CHI 2026), CC BY-NC-ND 4.0: non-commercial use only.
