# badminsens-analyzer

Counts and classifies badminton strokes in Garmin (Forerunner 255) `.fit` files with models
trained on the BADS smartwatch dataset (`BADS_CLL_OPENACCESS_V1`, BadminSense paper, `paper.pdf`).
The recordings come from a custom Garmin watch app that logs raw IMU data:
[github.com/omayib/badak](https://github.com/omayib/badak). Trained models and a Colab notebook are
on Hugging Face: [omayib/badak](https://huggingface.co/omayib/badak).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m badminsens.predict path/to/ACTIVITY.fit                # default model
.venv/bin/python -m badminsens.predict path/to/ACTIVITY.fit --model gru    # any shipped model
.venv/bin/python -m badminsens.predict path/to/ACTIVITY.fit --model all    # compare all models
.venv/bin/python -m badminsens.plot path/to/ACTIVITY.fit                   # 4 charts + desktop/mobile report pages
.venv/bin/python -m badminsens.plot path/to/ACTIVITY.fit --charts speed-simple  # just one
```

In Python or a notebook: `rows, summary = badminsens.predict.analyze("ACTIVITY.fit", model="svm")`,
then `badminsens.plot.plot_session("ACTIVITY.fit", rows)`. The charts: `hr` heart rate + strokes, `speed` average racket
speed (km/j) per 5 minutes, `speed-simple` average racket speed per stroke and game (km/j = km per hour,
estimated), times in the watch's local zone (WIB), `timeline` strokes per
2 minutes; `report-desktop` / `report-mobile` put all four on one page (a 2 x 2 grid, or one
narrow column for phones). A gap in the IMU data of at least `--min-rest` minutes (default 2, i.e. the watch app paused)
is drawn as rest and starts a new game. `badak_colab.ipynb` runs the same in
Google Colab, straight from the Hugging Face repo `omayib/badak`.

Output: counts and percentages per stroke in the terminal, `<ACTIVITY>_summary.json`, and
`<ACTIVITY>_strokes.csv` (one row per swing: time, stroke, confidence, gate decision, class
probabilities; with `--model all` every model's label).

Options: `--min-peak-dps` (swing detection, default 800 °/s), `--min-conf` (below this a stroke is
also flagged `Uncertain`, default 0.5), `--gate-quantile` (overhead-gate strictness, default 0.97),
`--no-gate` (force every swing into a class), `--balance` (class-balanced session statistics
iterations; default from the registry).

## What it can and cannot count

BADS contains only **4 overhead strokes**, all right-handed with the watch on the racket wrist:

| BADS label | stroke | side |
|---|---|---|
| `ForehandKill` | Smash | forehand |
| `ForehandHigh` | Clear | forehand |
| `ForehandLob` | Drop (吊球, "lob" is a mistranslation; it is **not** an underhand lift) | forehand |
| `BackhandTransition` | Backhand clear | backhand |

A match also contains lifts, net shots, drives, serves, defence and non-hitting swings. Earlier
versions forced all of them into one of the 4 classes, which inflated "backhand" (308 of 484
swings). Now an **overhead gate** first checks whether a swing looks like a dataset stroke at all
(forearm orientation before and after the swing, rotation axis at peak speed). The gate is
Mahalanobis distance to the stroke classes, accepting 97% of training strokes and 91% of strokes
of unseen players. Rejected swings are reported as `Other (not a dataset stroke)`.

Result for `24469883647_ACTIVITY.fit` (match, right wrist, right-handed), default model:

| | count | % of dataset strokes | % of all swings |
|---|---|---|---|
| Smash | 70 | 36.6 | 14.5 |
| Clear | 7 | 3.7 | 1.4 |
| Drop | 48 | 25.1 | 9.9 |
| Backhand clear | 66 | 34.6 | 13.6 |
| Other (not a dataset stroke) | 293 | | 60.5 |

Forehand 65% / backhand 35% of the dataset strokes. The other models give 28–41% smash, 4–18%
clear, 12–27% drop and 28–42% backhand clear (`--model all`). Treat that spread as the
uncertainty: there are no labelled Garmin strokes, so accuracy on your watch cannot be measured.
Clear is probably undercounted, because clear→smash is the main error on the dataset too (clear
recall 0.3–0.66).

## Playstyle app

```bash
.venv/bin/streamlit run app.py                     # upload .fit files in the browser
.venv/bin/python -m src.parser raw/*.fit           # print each session's 25 playstyle features
.venv/bin/python -m src.model fit                  # fit the skill index on BADS (leave-one-player-out report)
.venv/bin/python -m src.model validate raw/*.fit   # tune the archetype temperature by bootstrap calibration
```

`src/parser.py` turns a `.fit` file into swings (classified with the models above), games, heart rate
and a feature vector. `src/model.py` scores Power / Speed / Control / Endurance / Agility, the
archetype (Attacking / Defensive / Control / All-round) and a skill tier. `src/recommendation.py` maps
these to racket balance, weight, shaft and string tension, and `src/history.py` keeps past analyses in
SQLite (`.appdata/`, mirrored into the browser's IndexedDB). Weights and validation results are in
`models/playstyle.json`; method and limits in `MODEL_CARD.md`.

Deploy on Streamlit Community Cloud with main file `streamlit_cloud/streamlit_app.py` and Python 3.12
or 3.13 (numpy 2.5 needs >= 3.12). That folder's `requirements.txt` leaves out torch, and without torch
the app uses the SVM stroke model instead of the ensemble (87.1% vs 87.5% leave-one-player-out). `deploy/build_space.sh` assembles a Docker
Hugging Face Space instead (needs a PRO account).

## Pipeline

```bash
.venv/bin/python -m badminsens.dataset     # 1. BADS JSON -> data/bads_windows.npz, player split
.venv/bin/python -m badminsens.tune        # 2. configuration search loop (CV on 9 dev players)
.venv/bin/python -m badminsens.benchmark   # 3. evaluate once on 3 test players, ship to models/
.venv/bin/python -m badminsens.select      # 4. choose the default (leave-one-player-out, 12 players)
```

1. **Pre-processing** (`dataset.py`, `io.py`, `features.py`): 848 strokes as 2 s windows at 100 Hz
   (acc m/s², gyro rad/s), aligned so the gyro peak is at sample 94 (like windows cut from a .fit).
2. **Split by player**: train 7 players (490 strokes) + val 2 (141) = 9 *dev* players used for
   tuning; **test 3 players (217 strokes: Play6, Play9, Play12)** used once, after tuning.
3. **Features**: the paper's 23 time/frequency descriptors per axis (trim 200 ms, 20 Hz low-pass,
   stats of signal and derivative, FFT energy/centroid/entropy, Welch PSD) + our 378 shape and
   orientation features; sequence nets get the 12/20 Hz low-passed signal at 25 Hz (8 channels).
4. **Session normalization**: each player/session is centred on its own mean (removes individual
   style and device offsets). At inference the Garmin session mean is computed over the gate-accepted
   swings, re-weighted so each predicted class counts equally (`balance_iters`), because a real
   session does not have the dataset's balanced stroke mix. (Z-scaling per session scored slightly
   higher in CV but broke on Garmin data, so it is excluded.)
5. **Augmentation**: session-level wrist rotation (watch worn differently), per-stroke jitter,
   time shift, amplitude scaling, noise.

## The tuning loop

`badminsens.tune` searched 12 model families: 174 configurations in the final run, logged in
`experiments/tuning_log.jsonl` (plus 80 in an earlier run before peak alignment,
`tuning_log_v1_unaligned.jsonl`) (config + accuracy, balanced accuracy, macro P/R/F1, Cohen κ, MCC,
log-loss, ROC-AUC, per-player accuracy, confusion matrix, skewed-session accuracy). Per family it
tries the default, then random configurations, then mutations of the best one, and stops when
leave-one-player-out CV accuracy reaches the 0.90 target or the budget runs out. Searched settings:

- **SVM / LR / RF / ET / HGB**: feature set, normalization, C, γ, trees, max_features, leaf size,
  learning rate, leaf nodes, L2, number of augmented copies.
- **Neural nets**: epochs (30/60/100), batch (16/32/64), optimizer (AdamW/Adam/SGD+Nesterov),
  learning rate, weight decay, scheduler (cosine/one-cycle/none), label smoothing, dropout, dense
  head (none, 64, 128, 128-64, 256-128, …), conv channels and kernel, RNN hidden size, layers and
  direction, transformer width, heads and layers, input low-pass and rate, augmentation strength.

**No configuration reached 0.90 on unseen players.** The best was SVM at 0.892 CV. The biggest
gains came from the data rather than from hyperparameters: session normalization (+7 points) and
peak alignment (+1.5).

## Results

Full tables (all metrics, per-class P/R/F1, configurations, confusion matrices, tuning summary):
[`models/BENCHMARK.md`](models/BENCHMARK.md).

| model | CV acc (9 dev) | **test acc (3 unseen)** | macro F1 | κ | skewed session | 12-player LOPO |
|---|---|---|---|---|---|---|
| **svm + cnn + transformer** (default) | | | | | 0.845 / 0.856* | **0.875** |
| gru | 0.794 | **0.873 ± 0.021** | 0.870 | 0.829 | 0.861 | 0.821 |
| rf | 0.870 | 0.842 | 0.825 | 0.788 | 0.722 | 0.861 |
| transformer | 0.783 | 0.839 | 0.830 | 0.784 | 0.827 | 0.829 |
| et | 0.872 | 0.836 | 0.819 | 0.780 | 0.721 | 0.853 |
| svm | **0.892** | 0.814 | 0.783 | 0.751 | 0.716 | 0.871 |
| hgb | 0.878 | 0.810 | 0.788 | 0.745 | 0.728 | 0.861 |
| cnn | 0.792 | 0.810 | 0.811 | 0.745 | 0.805 | 0.807 |
| mlp | 0.881 | 0.794 | 0.755 | 0.724 | 0.743 | 0.856 |
| logreg | 0.853 | 0.779 | 0.738 | 0.703 | 0.684 | |
| lstm | 0.778 | 0.768 | 0.769 | 0.689 | 0.767 | |
| cnn_lstm | 0.759 | 0.754 | 0.743 | 0.672 | 0.757 | |
| tcn | 0.799 | 0.736 | 0.741 | 0.647 | 0.735 | |

*skewed session: test accuracy when the session statistics come from an unbalanced mix of the
player's strokes (without / with class balancing); for the default it is the 12-player value.*

How to read it:

- The **CV and test rankings disagree**: SVM is best on the 9 dev players and GRU on the 3 test
  players. Accuracy varies more between players than between models. Every model scores about 0.73
  on Play12 and 0.85–0.99 on Play9. With 3 test players, one standard error is about ±2.5 points.
- The **default was therefore chosen after the test evaluation**, by leave-one-player-out over all
  12 players (`badminsens.select`, 4× more unseen players). That makes the ensemble's number more
  reliable, but it is not an untouched test score.
- The paper reports 91.43% (SVM, one random leave-3-users-out split). Reproducing its pipeline
  gives 78.9% ± 5.4 over 20 random 3-player splits (best split 88.1%). Split luck explains most of
  the gap.
- Backhand clear is almost perfect, drop is good (F1 ≈ 0.9), and **clear vs smash** is the
  remaining error, as the paper also notes.

Best configurations (all settings in `models/registry.json`):

| model | key settings |
|---|---|
| svm | both feature sets, session-centred, RBF C=0.3, γ=3e-4, 4 augmented copies |
| cnn | conv 64-128-128, kernel 9, dense 128, dropout 0.3, 30 epochs, batch 32, AdamW lr 5e-4, wd 1e-4, cosine, label smoothing 0.2, 20 Hz → 25 Hz input, session-centred |
| transformer | d_model 32, 4 heads, 2 layers, no dense head, dropout 0.3, 30 epochs, batch 32, Adam lr 5e-4, wd 1e-3, cosine, session-centred |
| gru | 2-layer unidirectional GRU, 64 hidden, dense 128, dropout 0.5, 60 epochs, batch 32, SGD+Nesterov lr 0.01, cosine, no session normalization |

## Requirements for your recordings

- Raw accelerometer + gyroscope logging must be enabled (like `24469883647_ACTIVITY.fit`). The Garmin
  watch app at [github.com/omayib/badak](https://github.com/omayib/badak) records this kind of file.
- Watch on the **racket-hand wrist**, right-handed (as in BADS). The Garmin→BADS axis mapping was
  verified from the data: the gyro/accel physics check passed, and the identity rotation fits the
  dataset's stroke orientations far better than the other 23 rotations (32% vs ≤10% of swings in
  distribution).

## Garmin ↔ BADS sensor mapping

| | Garmin raw | Conversion |
|---|---|---|
| accel | 1 mg/LSB, ±8 g, **sign opposite to Android** | × −9.80665/1000 → m/s² |
| gyro | 16.384 LSB/(°/s), ±2000 °/s | ÷ 16.384 → °/s → rad/s |

## Notes

- Training uses the GPU when CUDA works and falls back to CPU otherwise. On this machine the
  driver was stuck (`CUDA unknown error`; fix with `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`),
  so everything was trained on CPU.
- `experiments/`: tuning logs (v1 = before peak alignment, kept for reference), selection results
  and run logs.
