# Stroke classifier benchmark (2026-09-25)

- Tuning: leave-one-player-out CV on 9 dev players (Play1, Play2, Play4, Play7, Play8, Play10, Play11, Play3, Play5).
- Test: Play6, Play9, Play12, never used during tuning; mean ± std over 3 seeds.
- *skew*: test accuracy when session statistics come from an unbalanced subset of the player's strokes; *skew+bal*: same with class-balanced re-estimation.

## Results (test players)

| model | CV acc | test acc | bal. acc | macro P | macro R | macro F1 | Cohen κ | MCC | log-loss | ROC-AUC | skew | skew+bal | train s | ms/stroke |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gru | 0.794 | 0.873 ± 0.021 | 0.868 | 0.881 | 0.868 | 0.870 | 0.829 | 0.833 | 0.402 | 0.961 | 0.861 | 0.861 | 65.3 | 0.11 |
| rf | 0.870 | 0.842 ± 0.006 | 0.839 | 0.866 | 0.839 | 0.825 | 0.788 | 0.803 | 0.411 | 0.974 | 0.722 | 0.756 | 3.6 | 0.77 |
| transformer | 0.783 | 0.839 ± 0.000 | 0.834 | 0.867 | 0.834 | 0.830 | 0.784 | 0.798 | 0.484 | 0.936 | 0.827 | 0.822 | 26.3 | 0.09 |
| et | 0.872 | 0.836 ± 0.002 | 0.833 | 0.858 | 0.833 | 0.819 | 0.780 | 0.795 | 0.398 | 0.970 | 0.721 | 0.762 | 2.4 | 0.82 |
| svm | 0.892 | 0.814 ± 0.002 | 0.810 | 0.846 | 0.810 | 0.783 | 0.751 | 0.774 | 0.527 | 0.940 | 0.716 | 0.779 | 4.1 | 0.91 |
| hgb | 0.878 | 0.810 ± 0.004 | 0.807 | 0.819 | 0.807 | 0.788 | 0.745 | 0.760 | 1.032 | 0.946 | 0.728 | 0.783 | 11.9 | 0.52 |
| cnn | 0.792 | 0.810 ± 0.023 | 0.810 | 0.816 | 0.810 | 0.811 | 0.745 | 0.747 | 0.562 | 0.949 | 0.805 | 0.804 | 13.1 | 0.11 |
| mlp | 0.881 | 0.794 ± 0.006 | 0.790 | 0.833 | 0.790 | 0.755 | 0.724 | 0.751 | 0.554 | 0.938 | 0.743 | 0.754 | 24.7 | 0.47 |
| ensemble | 0.884 | 0.793 ± 0.000 | 0.788 | 0.824 | 0.788 | 0.748 | 0.722 | 0.750 | 0.510 | 0.945 |  |  |  |  |
| logreg | 0.853 | 0.779 ± 0.000 | 0.774 | 0.786 | 0.774 | 0.738 | 0.703 | 0.728 | 0.817 | 0.900 | 0.684 | 0.705 | 0.6 | 0.57 |
| lstm | 0.778 | 0.768 ± 0.035 | 0.760 | 0.797 | 0.760 | 0.769 | 0.689 | 0.693 | 0.822 | 0.888 | 0.767 | 0.767 | 27.4 | 0.17 |
| cnn_lstm | 0.759 | 0.754 ± 0.022 | 0.756 | 0.788 | 0.756 | 0.743 | 0.672 | 0.692 | 1.609 | 0.935 | 0.757 | 0.753 | 23.0 | 0.11 |
| tcn | 0.799 | 0.736 ± 0.009 | 0.734 | 0.758 | 0.734 | 0.741 | 0.647 | 0.650 | 0.766 | 0.893 | 0.735 | 0.733 | 38.8 | 0.15 |

## Per-class precision / recall / F1 (test, seed 0)

| model | backhand clear | clear | smash | drop |
|---|---|---|---|---|
| gru | 1.00 / 1.00 / 1.00 | 0.83 / 0.66 / 0.74 | 0.77 / 0.95 / 0.85 | 0.96 / 0.90 / 0.92 |
| rf | 1.00 / 1.00 / 1.00 | 0.92 / 0.43 / 0.59 | 0.70 / 0.97 / 0.81 | 0.87 / 0.98 / 0.92 |
| transformer | 1.00 / 0.98 / 0.99 | 0.86 / 0.45 / 0.59 | 0.66 / 0.97 / 0.79 | 0.96 / 0.94 / 0.95 |
| et | 1.00 / 1.00 / 1.00 | 0.88 / 0.42 / 0.56 | 0.68 / 0.95 / 0.79 | 0.89 / 0.98 / 0.93 |
| svm | 1.00 / 1.00 / 1.00 | 0.89 / 0.30 / 0.45 | 0.66 / 0.97 / 0.79 | 0.84 / 0.98 / 0.90 |
| hgb | 1.00 / 1.00 / 1.00 | 0.79 / 0.36 / 0.49 | 0.67 / 0.92 / 0.77 | 0.85 / 0.98 / 0.91 |
| cnn | 1.00 / 1.00 / 1.00 | 0.61 / 0.58 / 0.60 | 0.69 / 0.73 / 0.71 | 0.94 / 0.92 / 0.93 |
| mlp | 1.00 / 1.00 / 1.00 | 0.93 / 0.25 / 0.39 | 0.64 / 0.97 / 0.77 | 0.82 / 0.98 / 0.90 |
| ensemble | 1.00 / 1.00 / 1.00 | 0.85 / 0.21 / 0.33 | 0.64 / 0.97 / 0.77 | 0.81 / 0.98 / 0.89 |
| logreg | 1.00 / 1.00 / 1.00 | 0.69 / 0.21 / 0.32 | 0.62 / 0.95 / 0.75 | 0.83 / 0.94 / 0.88 |
| lstm | 1.00 / 1.00 / 1.00 | 0.57 / 0.64 / 0.60 | 0.74 / 0.83 / 0.78 | 0.94 / 0.67 / 0.78 |
| cnn_lstm | 1.00 / 1.00 / 1.00 | 0.53 / 0.57 / 0.55 | 0.64 / 0.69 / 0.67 | 1.00 / 0.81 / 0.90 |
| tcn | 1.00 / 1.00 / 1.00 | 0.46 / 0.62 / 0.53 | 0.65 / 0.59 / 0.62 | 0.94 / 0.67 / 0.78 |

## Configurations

**gru** (sequence)

| align | norm | seq_lowpass | seq_decim | hidden | layers | bidirectional | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| True | none | 20 | 4 | 64 | 2 | False | [128] | 0.5 | 60 | 32 | sgd | 0.01 | 0.0001 | cosine | 0.1 | 30 | 0 | 0.02 |

**rf** (features)

| features | align | norm | n_estimators | max_features | min_samples_leaf | aug_copies |
|---|---|---|---|---|---|---|
| both | True | session_center | 500 | sqrt | 1 | 4 |

**transformer** (sequence)

| align | norm | seq_lowpass | seq_decim | d_model | heads | layers | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| True | session_center | 12 | 4 | 32 | 4 | 2 | [] | 0.3 | 30 | 32 | adam | 0.0005 | 0.001 | cosine | 0 | 15 | 15 | 0.02 |

**et** (features)

| features | align | norm | n_estimators | max_features | min_samples_leaf | aug_copies |
|---|---|---|---|---|---|---|
| both | True | session_center | 600 | 0.3 | 2 | 2 |

**svm** (features)

| features | align | norm | C | gamma | aug_copies | aug_session_rot | aug_stroke_rot |
|---|---|---|---|---|---|---|---|
| both | True | session_center | 0.3 | 0.0003 | 4 | 15 | 8 |

**hgb** (features)

| features | align | norm | max_iter | learning_rate | max_leaf_nodes | l2_regularization | aug_copies |
|---|---|---|---|---|---|---|---|
| both | True | session_center | 300 | 0.03 | 31 | 0 | 4 |

**cnn** (sequence)

| align | norm | seq_lowpass | seq_decim | channels | kernel | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| True | session_center | 20 | 4 | [64, 128, 128] | 9 | [128] | 0.3 | 30 | 32 | adamw | 0.0005 | 0.0001 | cosine | 0.2 | 30 | 0 | 0.02 |

**mlp** (mlp)

| features | align | norm | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both | True | session_center | [128] | 0.5 | 60 | 64 | adamw | 0.005 | 0.05 | cosine | 0.1 | 15 | 8 | 0.02 |

**ensemble**: average of svm, mlp, hgb

**logreg** (features)

| features | align | norm | C | aug_copies |
|---|---|---|---|---|
| both | True | session_center | 0.1 | 0 |

**lstm** (sequence)

| align | norm | seq_lowpass | seq_decim | hidden | layers | bidirectional | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| True | none | 12 | 4 | 64 | 2 | True | [64] | 0.3 | 40 | 32 | adamw | 0.002 | 0.0001 | cosine | 0.1 | 15 | 8 | 0 |

**cnn_lstm** (sequence)

| align | norm | seq_lowpass | seq_decim | channels | kernel | hidden | layers | bidirectional | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| True | session_center | 20 | 4 | [64, 128] | 7 | 64 | 2 | True | [64] | 0.5 | 60 | 64 | sgd | 0.03 | 0.0001 | none | 0 | 30 | 15 | 0.02 |

**tcn** (sequence)

| align | norm | seq_lowpass | seq_decim | channels | kernel | dense | dropout | epochs | batch | optimizer | lr | weight_decay | scheduler | label_smoothing | aug_session_rot | aug_stroke_rot | aug_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| True | session_center | 12 | 4 | [64, 64, 64, 64] | 7 | [128, 64] | 0.3 | 40 | 32 | adam | 0.002 | 0.01 | cosine | 0.1 | 15 | 8 | 0 |

## Tuning loop

| family | trials | worst CV acc | best CV acc | CPU time |
|---|---|---|---|---|
| svm | 30 | 0.780 | 0.892 | 3 min |
| mlp | 20 | 0.807 | 0.881 | 28 min |
| hgb | 16 | 0.796 | 0.878 | 12 min |
| et | 16 | 0.780 | 0.876 | 3 min |
| rf | 16 | 0.751 | 0.870 | 5 min |
| logreg | 14 | 0.780 | 0.865 | 1 min |
| gru | 8 | 0.769 | 0.819 | 22 min |
| tcn | 14 | 0.758 | 0.816 | 40 min |
| cnn_lstm | 10 | 0.731 | 0.805 | 13 min |
| lstm | 8 | 0.273 | 0.803 | 15 min |
| transformer | 8 | 0.739 | 0.800 | 19 min |
| cnn | 14 | 0.674 | 0.799 | 10 min |

## Default-model selection: leave-one-player-out over all 12 players

Run after the test evaluation above (`python -m badminsens.select`). Per-player accuracy averaged over the 12 players; ensembles average the members' probabilities.

| model / ensemble | accuracy | player mean | worst player | macro F1 | skew | skew+bal |
|---|---|---|---|---|---|---|
| svm+cnn+transformer | 0.875 | 0.873 | 0.750 | 0.879 | 0.845 | 0.856 |
| svm+cnn | 0.873 | 0.873 | 0.750 | 0.876 | 0.804 | 0.839 |
| et+mlp | 0.873 | 0.872 | 0.750 | 0.876 | 0.792 | 0.809 |
| svm+mlp+transformer | 0.873 | 0.872 | 0.750 | 0.876 | 0.821 | 0.846 |
| svm | 0.871 | 0.872 | 0.750 | 0.874 | 0.794 | 0.838 |
| svm+rf | 0.871 | 0.872 | 0.750 | 0.874 | 0.794 | 0.839 |
| svm+et | 0.871 | 0.872 | 0.750 | 0.874 | 0.794 | 0.839 |
| svm+mlp | 0.871 | 0.872 | 0.750 | 0.874 | 0.794 | 0.836 |
| svm+rf+mlp | 0.871 | 0.872 | 0.750 | 0.874 | 0.793 | 0.835 |
| svm+et+mlp | 0.871 | 0.872 | 0.750 | 0.874 | 0.793 | 0.833 |
| hgb | 0.861 | 0.861 | 0.733 | 0.863 | 0.777 | 0.818 |
| rf | 0.861 | 0.859 | 0.750 | 0.865 | 0.760 | 0.784 |
| mlp | 0.856 | 0.856 | 0.750 | 0.860 | 0.798 | 0.801 |
| et | 0.853 | 0.851 | 0.750 | 0.856 | 0.757 | 0.781 |
| transformer | 0.829 | 0.827 | 0.733 | 0.835 | 0.817 | 0.815 |
| gru | 0.821 | 0.819 | 0.717 | 0.829 | 0.819 | 0.819 |
| cnn | 0.807 | 0.807 | 0.569 | 0.817 | 0.808 | 0.806 |

## Confusion matrices (test, seed 0; rows = true, columns = predicted)

Order: backhand clear, clear, smash, drop

**gru**
```
  57    0    0    0
   0   35   17    1
   0    2   56    1
   0    5    0   43
```

**rf**
```
  57    0    0    0
   0   23   24    6
   0    1   57    1
   0    1    0   47
```

**transformer**
```
  56    0    1    0
   0   24   28    1
   0    1   57    1
   0    3    0   45
```

**et**
```
  57    0    0    0
   0   22   26    5
   0    2   56    1
   0    1    0   47
```

**svm**
```
  57    0    0    0
   0   16   29    8
   0    1   57    1
   0    1    0   47
```

**hgb**
```
  57    0    0    0
   0   19   27    7
   0    4   54    1
   0    1    0   47
```

**cnn**
```
  57    0    0    0
   0   31   19    3
   0   16   43    0
   0    4    0   44
```

**mlp**
```
  57    0    0    0
   0   13   31    9
   0    1   57    1
   0    0    1   47
```

**ensemble**
```
  57    0    0    0
   0   11   32   10
   0    1   57    1
   0    1    0   47
```

**logreg**
```
  57    0    0    0
   0   11   34    8
   0    2   56    1
   0    3    0   45
```

**lstm**
```
  57    0    0    0
   0   34   17    2
   0   10   49    0
   0   16    0   32
```

**cnn_lstm**
```
  57    0    0    0
   0   30   23    0
   0   18   41    0
   0    9    0   39
```

**tcn**
```
  57    0    0    0
   0   33   19    1
   0   23   35    1
   0   16    0   32
```
