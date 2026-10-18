---
license: cc-by-nc-nd-4.0
language:
- en
library_name: scikit-learn
pipeline_tag: tabular-classification
tags:
- badminton
- imu
- smartwatch
- garmin
- sports
- human-activity-recognition
- time-series-classification
- playstyle
- streamlit
- pytorch
metrics:
- accuracy
- f1
---

# 🏸 badak: how many smashes did I actually hit?

![Heart rate and strokes from one of my badminton matches](assets/hr_strokes.png)

*One of my matches: heart rate on top, and below it every stroke the model found, sorted by type.
That picture is what this whole project was about.*

## Why I built this

I play badminton, and after every match I wonder how I actually played. How many smashes did I hit?
Do I play more backhands than I think? Am I clearing enough, or dropping too much?

My watch is a Garmin Forerunner 255. It tracks runs and rides very well, but for badminton it gives
me a heart rate curve, calories and a time, and nothing about the strokes themselves. There is no
badminton mode that tells you what you hit.

Things got interesting when I found the paper
**[BadminSense: Enabling Fine-Grained Badminton Stroke Evaluation on a Single Smartwatch](https://doi.org/10.1145/3772318.3790998)**
(Chen, Chen, Liu, Ke and Sun, CHI 2026). The authors recorded 848 badminton strokes from 12 players
with a smartwatch, labelled every one by hand, and **released the dataset publicly**. A huge thank
you to them: without that dataset this project would not exist. 🙏

So the plan became:

1. **Record the raw motion myself.** I wrote a small Garmin watch app that logs the watch's
   accelerometer and gyroscope (128 samples per second) during a session, with the watch on my
   racket wrist. The app is open source: **[github.com/omayib/badak](https://github.com/omayib/badak)**.
2. **Let Garmin do the syncing.** The session syncs to Garmin Connect like any other activity, and I
   download the `.fit` file from there.
3. **Teach a model once, reuse it forever.** Train classifiers on the BadminSense data, make them
   understand *my* Garmin recordings, and analyse any new `.fit` file in a few seconds.

That's what this repository is: the trained models, the code, and a notebook you can run in your browser.

## What it tells you

For every swing in your recording you get one of these:

| stroke | what it is |
|---|---|
| **Smash** | forehand overhead smash |
| **Clear** | forehand overhead clear (high and deep) |
| **Drop** | forehand overhead drop shot |
| **Backhand clear** | backhand overhead clear |
| **Other** | anything else: lifts, net shots, drives, serves, defence, practice swings |

The dataset only covers these **four overhead strokes**, so an honest model has to say "that's
something else" instead of forcing every swing into one of them. My first version didn't do that
and told me 308 of my 484 swings were backhands. Now, if a swing doesn't look like any of the four,
it's counted as *Other*.

For the match in the picture above (with the SVM model): **65 smashes, 13 clears, 52 drops,
61 backhand clears**, and 293 other swings.

## Try it on your own recording (Google Colab, nothing to install locally)

1. Download [`badak_colab.ipynb`](badak_colab.ipynb) from this repository.
2. Open [Google Colab](https://colab.research.google.com), then **File → Upload notebook** and pick it.
3. Run the cells from top to bottom. When asked, upload your `.fit` file.

You get the stroke counts, the chart above for your own session, and a comparison across models.
The only install is one line inside Colab (`fitparse` to read Garmin files, plus the scikit-learn
version the models were saved with). Nothing touches your own computer.

If you'd rather paste code than download a notebook, these cells do the same:

```python
!pip install -q fitparse scikit-learn==1.9.1
```

```python
from huggingface_hub import snapshot_download
import sys
repo = snapshot_download("omayib/badak")      # models + code from this repo
sys.path.insert(0, repo)

from badminsens.predict import analyze, print_summary
from badminsens.plot import plot_session, plot_report, game_spans
from badminsens.io import load_fit
from google.colab import files

fit = next(iter(files.upload()))              # pick your .fit file
rows, summary = analyze(fit)                  # default model; try model="svm", "gru", ...
print_summary(summary, "my session")
plot_session(fit, rows, "default")            # heart rate + strokes chart
games = game_spans(load_fit(fit))             # pausing the watch app between games = rest
plot_report(fit, rows, games, layout="mobile")   # all charts on one page ("desktop" for 2 x 2)
```

### Choosing a model

`snapshot_download` fetches the code and **all** the models (about 20 MB). You pick one when you
analyse, with `model=`:

```python
rows, summary = analyze(fit)                # default: SVM + CNN + Transformer ensemble
rows, summary = analyze(fit, model="svm")   # or "gru", "rf", "et", "hgb", "mlp", "cnn", "transformer", ...
```

| model | when to use it |
|---|---|
| `"default"` | Not sure? Use this. Best overall, and it holds up when a match has an uneven mix of strokes. |
| `"svm"` | One fast, simple model, nearly as good as the default and very decisive. |
| `"gru"` | A second opinion. Best on the held-out test players and steady with a lopsided stroke mix. |
| `"rf"` / `"et"` | Tree models only, no neural networks. |

If two models disagree on your file, compare several (notebook step 6). The spread shows how
certain each count is; clear is usually the least certain. To download just one model:

```python
repo = snapshot_download("omayib/badak", allow_patterns=[
    "badminsens/*", "models/registry.json", "models/gate.joblib", "models/svm.joblib"])
```

(The default needs `svm.joblib`, `cnn.joblib` and `transformer.joblib`.)

**What your file needs:** raw accelerometer and gyroscope data (the `accelerometer_data` and
`gyroscope_data` messages in the `.fit`), recorded on the **racket wrist**, right-handed. A normal
Garmin activity file only stores 1-second summaries, and there's nothing in it to classify. To
record the right kind of file, use my Garmin watch app:
**[github.com/omayib/badak](https://github.com/omayib/badak)**.

### On your own computer

```bash
git clone https://huggingface.co/omayib/badak && cd badak
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m badminsens.predict my_session.fit             # counts + per-swing CSV
.venv/bin/python -m badminsens.predict my_session.fit --model all # compare every model
.venv/bin/python -m badminsens.plot my_session.fit                # all charts + desktop/mobile pages
```

## Playstyle and racket recommendations

Counting strokes was step one. The next question was *what kind of player am I*, and what racket
and string tension suit that. `src/` and the Streamlit app `app.py` answer it in three layers.
All the weights are in one readable JSON file, [`models/playstyle.json`](models/playstyle.json).

**1. Five attribute scores (0–100).** Each session becomes a feature vector of 25 numbers
(`src/parser.py`, `FEATURES`): stroke shares, smash wrist speed, racket-head speed, tempo, rally length,
the Markov transition entropy between stroke types (after the Markov-chain hitting-sequence paper), heart-rate load,
how well swing speed holds up late in the session, and movement between swings. Each feature has an
anchor range that maps to 10–90 points, and an attribute is a weighted mean of its features:

| attribute | built from |
|---|---|
| **Power** | smash wrist speed (anchors from the BADS players), racket-head speed, peak acceleration, smash share |
| **Speed** | swings per minute, time between own shots, median swing speed |
| **Control** | drops + clears over smashes, consistency of swing speed, classifier confidence, stroke variety |
| **Endurance** | minutes played, heart-rate reserve, speed retention, Garmin aerobic training effect |
| **Agility** | movement between swings, backhand share, rally length, anaerobic training effect |

**2. Archetype: Attacking, Defensive, Control or All-round.** The five scores are centred on their
own mean (relative strengths) and compared with prototype shapes. A flat profile is All-round. The
softmax temperature is the one hyper-parameter, and with no playstyle labels it was tuned by
calibration: on my 8 sessions × 200 block-bootstraps of the swings, T = 45 makes the top probability
(79%) match how often the archetype survives resampling (77%).

**3. Skill tier: Low / Medium / High.** BADS records each player's level. Smash speed, backhand
speed and the smash-drop speed gap (gyro clipped to the Garmin's ±2000 °/s range) each correlate
with it. Averaged into one index and checked leave-one-player-out on the 12 players:

| | |
|---|---|
| exact level | **75%** (majority baseline 42%) |
| within one level | 100% |
| Spearman ρ, held-out index vs level | 0.85 (p = 0.0005) |

**Equipment** (`src/recommendation.py`) is rule-based and follows the evidence table in the
references: balance from attack vs defence affinity, weight class from skill and endurance, shaft
stiffness from smash speed and skill, and string tension set mainly by skill tier (Low 21–24,
Medium 24–27, High 27–30 lbs), then nudged by style and fatigue. Lower tension gives more shuttle
speed (Vanasant 2013), and experts cope with any tension (Zhu 2013, 2014). Every suggestion comes
with its reason and how strong the evidence behind it is.

### Input and output

```python
from huggingface_hub import snapshot_download
import sys
repo = snapshot_download("omayib/badak"); sys.path.insert(0, repo)

from src.parser import parse_fit           # .fit -> swings, games, heart rate, 25 features
from src.model import load                 # models/playstyle.json (downloads from the Hub if missing)
from src.recommendation import recommend

session = parse_fit("my_session.fit")
pred = load().predict(session["features"])
# {'attributes': {'Power': 75.5, 'Speed': 58.0, 'Control': 40.3, 'Endurance': 42.0, 'Agility': 31.9},
#  'archetype': 'Attacking', 'archetype_probs': {'Attacking': 0.996, ...},
#  'skill_tier': 'Medium', 'skill_index': 0.15, 'description': '...'}
rec = recommend(pred, session["features"])
# {'balance': 'Head-heavy', 'weight': '4U', 'shaft': 'Medium', 'tension_lbs': [23, 26], 'reasons': {...}, ...}
```

Run the app locally with `streamlit run app.py`. It takes several `.fit` files at once and keeps
your history (SQLite on the server, mirrored into your browser's IndexedDB), so you can browse past
sessions without uploading again.

**Honest limits of this part.** Nobody has labelled playstyles for these recordings, so the archetype
is a transparent, knowledge-based mapping, not a trained classifier. A single game agrees with its whole
session's archetype only 59% of the time, so trust session and multi-session averages over single
games. The skill index comes from 12 players on a Samsung watch. About 20% of my swings clip the Garmin
gyroscope's range, which makes Power a lower bound. Equipment links to play mode are mostly coaching
consensus, so treat the suggestions as a starting point to test.

## Making a smartwatch dataset understand a Garmin recording

The BadminSense strokes were recorded on a Samsung Galaxy Watch. My recordings come from a Garmin.
Same idea, different device, so the dataset and the Garmin files had to be brought onto common
ground first. Here is what happens, in order:

1. **Same units.** Garmin stores raw numbers. The accelerometer is in milli-g, the gyroscope in
   steps of 1/16.384 °/s. Both are converted to physical units (m/s² and rad/s), the same as the
   dataset.
2. **Same direction.** Garmin's accelerometer reports gravity with the opposite sign to Android
   watches, so it's flipped. I didn't just assume the axes line up: I checked it two ways. First, that
   the Garmin gyroscope and accelerometer agree with each other physically. Second, that out of all 24
   possible ways of rotating the watch's axes, the one used makes my swings look most like the
   dataset's strokes (32% of swings fit, against 10% or less for any other rotation).
3. **Same timing.** Garmin samples at 128 Hz, the dataset at 100 Hz. Everything is resampled to 100 Hz. (The .fit IMU timestamps count 1/1024 s ticks as ms, so they read 125 Hz and run 2.4% fast; `io.load_fit` corrects them onto the activity clock using the timer events.)
4. **Same stroke window.** In the dataset every stroke is a 2-second clip. In a Garmin session I find
   swings automatically: every burst of wrist rotation faster than 800 °/s is a candidate. I cut the
   same 2-second clip around each one, with the moment of peak rotation at the same position. I also
   re-aligned the dataset's clips on their peak, which made the models noticeably more accurate.
5. **Same bandwidth.** Both sources are low-pass filtered, so a difference in sensor sharpness
   between the two watches doesn't look like a difference in stroke.
6. **Is it one of the four strokes at all?** A small "overhead gate" checks how the forearm is held
   before and after the swing and around which axis the wrist turns. If that doesn't look like any of
   the four dataset strokes, the swing is counted as *Other*. It still lets through about 91% of real
   dataset strokes from players it has never seen.
7. **You are not the dataset's players.** Everyone swings a bit differently, and every watch sits a
   bit differently on the wrist. Each session is therefore compared against its own average, which
   takes out personal and device offsets. Because a real match doesn't have a nicely balanced mix of
   strokes, that average is re-weighted so each stroke type counts equally.
8. **Robust training.** During training, the strokes were randomly rotated (the watch worn a little
   differently), shifted in time and scaled in strength, so the models don't memorise one exact
   recording setup.

## How good is it?

I tried 12 kinds of models: SVM, logistic regression, random forest, extra trees, gradient boosting
and an MLP on hand-crafted features, and a CNN, TCN, LSTM, GRU, CNN-LSTM and Transformer on the raw
motion signal. A tuning loop tried 174 configurations (epochs, layer sizes, optimizers, learning
rates, dropout, …), always scoring on players the model had never seen.

| model | accuracy on 3 unseen players | 12 players, each held out once |
|---|---|---|
| **SVM + CNN + Transformer** (default) | | **87.5%** |
| GRU | **87.3%** | 82.1% |
| random forest | 84.2% | 86.1% |
| SVM | 81.4% | 87.1% |

What I learned:

- **Around 85–88% on a new player** is where it lands. The paper reports 91%, but on a single random
  split. Repeating their method over many splits averages closer to 79%, so "which players you test
  on" matters a lot. Accuracy on one player ranges from about 73% to 99%.
- **Backhand clears and drops are recognised very reliably. Clear vs smash is the hard part.** Both
  start with the same swing, so clears tend to be undercounted.
- **I can't measure accuracy on my own Garmin recordings**, because nobody labelled my strokes.
  Comparing several models on the same file (`--model all`) gives a sense of how sure the counts are.

Full numbers are in [`models/BENCHMARK.md`](models/BENCHMARK.md): precision, recall and F1 per
stroke, Cohen's κ, MCC, ROC-AUC, confusion matrices and every model's exact configuration.

## Limitations

- Only the four **overhead** strokes are recognised. Everything else is *Other*.
- Right-handed players, watch on the **racket wrist**.
- Trained on 12 players (10 men, 2 women), one watch model, one racket. Your mileage may vary.
- The model files are Python pickles (`.joblib`). Loading a pickle can run code, so only load them from
  a source you trust.

## Repository contents

| path | what's inside |
|---|---|
| `badak_colab.ipynb` | the Colab notebook |
| `badminsens/` | the code: reading `.fit`, swing detection, gate, models, training, plotting |
| `src/` | playstyle layer: `parser.py` (features), `model.py` (scores, archetype, skill), `recommendation.py`, `history.py` |
| `app.py` | the Streamlit app |
| `models/` | 12 trained models + the gate, `registry.json` (configs), `BENCHMARK.md` (results), `playstyle.json` (playstyle weights + validation) |
| `experiments/` | every tuning trial and the model-selection results |
| `DOCS.md` | technical documentation, including how to retrain everything |

## License and credits

The models are trained on the **BADS dataset** from BadminSense, released under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
([dataset repository](https://github.com/taizhouchen/BadminSense_Dataset)). This repository follows
the same terms: **non-commercial use only**, with credit to the original authors. If you use it,
please cite their paper:

```bibtex
@inproceedings{chen2026badminsense,
  title     = {BadminSense: Enabling Fine-Grained Badminton Stroke Evaluation on a Single Smartwatch},
  author    = {Chen, Taizhou and Chen, Kai and Liu, Xingyu and Ke, Pingchuan and Sun, Zhida},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  year      = {2026},
  doi       = {10.1145/3772318.3790998}
}
```

Thanks again to Taizhou Chen, Kai Chen, Xingyu Liu, Pingchuan Ke and Zhida Sun for making their data available. 🏸
