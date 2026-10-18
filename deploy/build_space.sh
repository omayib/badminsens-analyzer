#!/usr/bin/env bash
# Assemble the Hugging Face Space folder: deploy/build_space.sh OUT_DIR, then
#   hf upload omayib/badak-playstyle OUT_DIR . --repo-type space
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: deploy/build_space.sh OUT_DIR}
rm -rf "$OUT" && mkdir -p "$OUT/models" "$OUT/src" "$OUT/badminsens"
cp -r deploy/space/. "$OUT/"
cp app.py "$OUT/"
cp src/__init__.py src/parser.py src/model.py src/recommendation.py src/history.py "$OUT/src/"
cp badminsens/*.py "$OUT/badminsens/"
# the default model is the svm + cnn + transformer ensemble, plus the overhead gate
cp models/registry.json models/gate.joblib models/svm.joblib models/cnn.joblib models/transformer.joblib \
   models/playstyle.json "$OUT/models/"
