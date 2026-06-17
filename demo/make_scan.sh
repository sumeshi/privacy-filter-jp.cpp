#!/usr/bin/env bash
# Render the single-document PII-scan TUI (pii_scan.py) for the "scan" scene,
# record it with recorder-for-agents, trim the lead-in, append a branding outro.
# The scene + real spans + real tok/s come from gen_scan.py (pf-cli output).
#
#   ./make_scan.sh                 # real time
#   DILATE=2 ./make_scan.sh        # slowed 2x
#
# env: RECORDER, DILATE, WIDTH/HEIGHT/FPS/FONTSIZE, HOLD, CARD, LINK
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RECORDER=${RECORDER:-/home/rich/python/recorder-for-agents}
SCENE=scan
OUT=${1:-pii_scan.mp4}
DILATE=${DILATE:-1}
W=${WIDTH:-1280}; H=${HEIGHT:-720}; FS=${FONTSIZE:-16}; FPS=${FPS:-30}
HOLD=${HOLD:-1.4}; CARD=${CARD:-3.5}
LINK=${LINK:-github.com/richiejp/privacy-filter.cpp}
SDIR="$HERE/traces/$SCENE"

[ -d "$SDIR" ] || { echo "no scene at $SDIR (run gen_scan.py)"; exit 1; }
[ -x "$RECORDER/record.sh" ] || { echo "recorder not found at $RECORDER"; exit 1; }

# capture length: start delay + scan (scaled) + hold + settle + card + buffer
DUR=$(python3 - "$SDIR" "$DILATE" "$HOLD" "$CARD" <<'PY'
import json, sys, math
from pathlib import Path
d = Path(sys.argv[1]); dil, hold, card = map(float, sys.argv[2:5])
c = json.load(open(d / "content.json")); e = json.load(open(d / "engines.json"))[0]
proc = c["n_tokens"] / e["tps"]
print(int(math.ceil(1.0 + proc * dil + hold + 0.7 + card + 1.0)))
PY
)
echo "[make-scan] ${W}x${H}@${FPS} fs=${FS} dilate=${DILATE} duration=${DUR}s -> out/$OUT"

WORK="$HERE" BG="#0d1117" FG="#d7dde5" FONTSIZE="$FS" DURATION="$DUR" \
  WIDTH="$W" HEIGHT="$H" FPS="$FPS" START_DELAY=1.0 END_HOLD=0.2 \
  "$RECORDER/record.sh" \
  "python3 pii_scan.py --scene traces/$SCENE --dilate $DILATE --hold $HOLD --card $CARD --link '$LINK'" \
  "$OUT"

RAW="$HERE/out/$OUT"; NOEXT="${OUT%.mp4}"
if [ -f "$RECORDER/examples/duel/trim_lead.sh" ]; then
  bash "$RECORDER/examples/duel/trim_lead.sh" "$RAW" "$HERE/out/.trim_$SCENE.mp4" \
    && mv "$HERE/out/.trim_$SCENE.mp4" "$RAW"
fi
if [ -f "$RECORDER/examples/duel/outro.sh" ]; then
  OW="$W" OH="$H" TITLE="privacy-filter.cpp" \
  LINK1="github.com/richiejp/privacy-filter.cpp" \
  LINK2="on-device NER · Raspberry Pi 5 · real PII spans" \
    bash "$RECORDER/examples/duel/outro.sh" "$RAW" "$HERE/out/${NOEXT}_final.mp4"
  echo "-> $HERE/out/${NOEXT}_final.mp4"
else
  echo "-> $RAW"
fi
