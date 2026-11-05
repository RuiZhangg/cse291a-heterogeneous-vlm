#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/project-03
source envs/openvino/bin/activate
ulimit -c 0

SEED="${RUN_SEED:-101}"
N="${PHASE_N:-50}"
SOURCE="src/pipeline_real_tiles_fixed.py"

ROOT="runs/$(date +%Y%m%d-%H%M%S)-phase1-rate-sweep-seed-$SEED"
mkdir -p "$ROOT"
echo "$ROOT" > runs/phase1_rate_sweep_latest.txt

cp "$SOURCE" "$ROOT/source_snapshot.py"

{
    date -Is
    hostname
    git rev-parse HEAD
    sha256sum "$SOURCE"
    python --version
    python -c "import openvino; print(openvino.__version__)"
    echo "Seed=$SEED"
    echo "Requests per configuration=$((3*N))"
    echo "Source=$SOURCE"
} > "$ROOT/metadata.txt" 2>&1

# Randomize the order of workload pairs.
# Keep GPU and NPU executions adjacent within each pair.
python - <<'PY' > "$ROOT/manifest.tsv"
import random

cases = (
    [(rate, 3) for rate in ("1.5", "1.8", "2.0")]
    + [(rate, 5) for rate in (
        "0.8", "1.0", "1.4", "1.6", "1.8"
    )]
)

rng = random.Random(713)
rng.shuffle(cases)

for rate, tiles in cases:
    devices = ["GPU", "NPU"]
    rng.shuffle(devices)
    for device in devices:
        print(rate, tiles, device, sep="\t")
PY

printf 'rate\ttiles\tdevice\tseed\texit_code\twall_seconds\n' \
    > "$ROOT/status.tsv"

while IFS=$'\t' read -r RATE TILES DEVICE; do

    DIR="$ROOT/rate-$RATE/tiles-$TILES/$DEVICE"
    mkdir -p "$DIR"

    if [[ "$TILES" == "3" ]]; then
        SPATIAL=2
    elif [[ "$TILES" == "5" ]]; then
        SPATIAL=4
    else
        echo "Invalid tile count: $TILES" >&2
        exit 1
    fi

    echo "============================================"
    echo "rate=$RATE tiles=$TILES device=$DEVICE"

    START=$SECONDS
    STATUS=0

    if timeout 650s /usr/bin/time -v \
        -o "$DIR/resource.txt" \
        python -u "$SOURCE" \
        --tiles "$SPATIAL" \
        --thumbnail \
        --worker-mode serial \
        --vision-device "$DEVICE" \
        --phase-rates "$RATE,$RATE,$RATE" \
        --phase-requests "$N" \
        --seed "$SEED" \
        --warmups 2 \
        --max-tokens 32 \
        --frames-dir data/frames \
        --out "$DIR/results.csv" \
        > "$DIR/output.log" 2>&1; then
        :
    else
        STATUS=$?
    fi

    WALL=$((SECONDS-START))

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RATE" "$TILES" "$DEVICE" \
        "$SEED" "$STATUS" "$WALL" \
        >> "$ROOT/status.tsv"

    if [[ "$STATUS" -eq 0 ]]; then
        grep -E \
          '^(Completed:|e2e_ms |vision_wait_ms |decoder_wait_ms |vision_ms |decode_ms )' \
          "$DIR/output.log" || true
    else
        echo "FAILED: $STATUS"
        tail -20 "$DIR/output.log"
    fi

    echo "Wall time: ${WALL}s"

done < "$ROOT/manifest.tsv"

echo "Finished. Results: $ROOT"
