#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/project-03
source envs/openvino/bin/activate

SEED="${RUN_SEED:-101}"
PHASE_N="${PHASE_N:-50}"
FRAMES_DIR="${FRAMES_DIR:-data/frames}"
SOURCE="${SOURCE:-src/pipeline_real_tiles_fixed.py}"

ROOT="${RUN_ROOT:-runs/$(date +%Y%m%d-%H%M%S)-phase1-seed-$SEED}"

mkdir -p "$ROOT"
echo "$ROOT" > runs/phase1_latest.txt

# Preserve experiment metadata.
{
    date -Is
    hostname
    git rev-parse HEAD
    git status --short
    python --version
    python -c "import openvino as ov; print('OpenVINO', ov.__version__)"
    echo "Seed: $SEED"
    echo "Requests per run: $((PHASE_N * 3))"
    echo "Frame directory: $FRAMES_DIR"
    echo "Source: $SOURCE"
    echo "Image count:"
    find "$FRAMES_DIR" -type f -iname '*.jpg' | wc -l
    df -h .
} > "$ROOT/metadata.txt" 2>&1

# Produce a fixed, randomized execution order.
python - <<'PY' > "$ROOT/manifest.tsv"
import itertools
import random

cases = list(itertools.product(
    ("1.2", "2.2"),
    (1, 3, 5),
    ("GPU", "NPU")
))

random.Random(1703).shuffle(cases)

for rate, total_tiles, device in cases:
    # 1 tile: no thumbnail.
    # 3 tiles: 2 spatial + thumbnail.
    # 5 tiles: 4 spatial + thumbnail.
    print(rate, total_tiles, device, sep="\t")
PY


printf 'rate\ttiles\tdevice\tseed\texit_code\twall_seconds\n' \
    > "$ROOT/status.tsv"

while IFS=$'\t' read -r RATE TOTAL_TILES DEVICE; do
    DIR="$ROOT/rate-$RATE/tiles-$TOTAL_TILES/$DEVICE"
    mkdir -p "$DIR"

    if [[ "$TOTAL_TILES" == "1" ]]; then
        SPATIAL=1
        EXTRA=()
    elif [[ "$TOTAL_TILES" == "3" ]]; then
        SPATIAL=2
        EXTRA=(--thumbnail)
    elif [[ "$TOTAL_TILES" == "5" ]]; then
        SPATIAL=4
        EXTRA=(--thumbnail)
    else
        echo "Unsupported tile configuration" >&2
        exit 1
    fi

    echo "================================================"
    echo "Rate=$RATE, tiles=$TOTAL_TILES, device=$DEVICE"
    echo "Seed=$SEED, requests=$((PHASE_N * 3))"

    START=$SECONDS

    if /usr/bin/time -v -o "$DIR/resource.txt" \
       timeout 600s python -u "$SOURCE" \
        --tiles "$SPATIAL" \
        "${EXTRA[@]}" \
        --worker-mode serial \
        --vision-device "$DEVICE" \
        --phase-rates "$RATE,$RATE,$RATE" \
        --phase-requests "$PHASE_N" \
        --seed "$SEED" \
        --warmups 2 \
        --max-tokens 32 \
        --frames-dir "$FRAMES_DIR" \
        --out "$DIR/results.csv" \
        > "$DIR/output.log" 2>&1; then
        STATUS=0
    else
        STATUS=$?
    fi

    WALL=$((SECONDS - START))

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RATE" "$TOTAL_TILES" "$DEVICE" \
        "$SEED" "$STATUS" "$WALL" \
        >> "$ROOT/status.tsv"

    if [[ "$STATUS" -eq 0 ]]; then
        grep -E \
          '^(Completed:|e2e_ms |vision_wait_ms |vision_ms |decoder_wait_ms |ttft_ms |decode_ms )' \
          "$DIR/output.log" || true
    else
        echo "FAILED with exit code $STATUS"
        tail -25 "$DIR/output.log"
    fi

    echo "Elapsed: ${WALL}s"
done < "$ROOT/manifest.tsv"

echo
echo "ALL CONFIGURATIONS ATTEMPTED"
echo "Results: $ROOT"
