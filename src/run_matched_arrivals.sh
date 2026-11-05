#!/usr/bin/env bash
set -Eeuo pipefail

cd /data/project-03
source envs/openvino/bin/activate
ulimit -c 0

SEED="${RUN_SEED:-101}"

ROOT="runs/$(date +%Y%m%d-%H%M%S)-matched-arrivals-seed-$SEED"
mkdir -p "$ROOT"

echo "$ROOT" > runs/matched_arrivals_latest.txt

cp src/pipeline_arrival_matched.py \
   "$ROOT/source_snapshot.py"

{
    date -Is
    hostname
    git rev-parse HEAD
    sha256sum src/pipeline_arrival_matched.py
    echo "Seed=$SEED"
} > "$ROOT/metadata.txt"

printf 'rate\ttiles\tmode\tdevice\texit_code\n' \
    > "$ROOT/status.tsv"

for SPEC in "1.5 2" "1.0 4"; do
    read -r RATE SPATIAL <<< "$SPEC"

    TILES=$((SPATIAL + 1))

    for MODE in periodic poisson-matched; do
        for DEVICE in GPU NPU; do

            DIR="$ROOT/rate-$RATE/tiles-$TILES/$MODE/$DEVICE"
            mkdir -p "$DIR"

            echo "======================================"
            echo "$TILES tiles | $RATE req/s | $MODE | $DEVICE"

            STATUS=0

            if timeout 500s /usr/bin/time -v \
                -o "$DIR/resource.txt" \
                python -u src/pipeline_arrival_matched.py \
                    --tiles "$SPATIAL" \
                    --thumbnail \
                    --worker-mode serial \
                    --vision-device "$DEVICE" \
                    --arrival-mode "$MODE" \
                    --phase-rates "$RATE,$RATE,$RATE" \
                    --phase-requests 50 \
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

            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$RATE" "$TILES" "$MODE" \
                "$DEVICE" "$STATUS" \
                >> "$ROOT/status.tsv"

            if [[ "$STATUS" -eq 0 ]]; then
                grep -E \
                    'Arrival mode:|Completed:|e2e_ms |vision_wait_ms |decoder_wait_ms ' \
                    "$DIR/output.log" || true
            else
                echo "FAILED: $STATUS"
                tail -15 "$DIR/output.log"
            fi
        done
    done
done

echo "Results: $ROOT"
