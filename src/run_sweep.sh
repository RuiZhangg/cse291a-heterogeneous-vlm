#!/usr/bin/env bash
set -u

cd /data/project-03 || exit 1

SWEEP_DIR="runs/$(date +%Y%m%d-%H%M%S)-arrival-sweep"
mkdir -p "$SWEEP_DIR"

for RATE in 2 4 5 0; do
    for REP in 1 2 3; do

        if (( REP % 2 == 1 )); then
            DEVICES=(GPU NPU)
        else
            DEVICES=(NPU GPU)
        fi

        for DEVICE in "${DEVICES[@]}"; do
            RUN_DIR="$SWEEP_DIR/rate-${RATE}-rep-${REP}-${DEVICE}"
            mkdir -p "$RUN_DIR"

            echo "===== Rate=$RATE Rep=$REP Device=$DEVICE ====="

            (
                ulimit -c 0
                timeout 120s python -u src/pipeline_bench.py \
                    --vision-device "$DEVICE" \
                    --rate "$RATE" \
                    --requests 24 \
                    --warmups 2 \
                    --out "$RUN_DIR/results.csv"
            ) > "$RUN_DIR/output.log" 2>&1

            RESULT=$?
            echo "$RESULT" > "$RUN_DIR/exit_code.txt"

            if [ "$RESULT" -eq 0 ]; then
                grep -E '^(Completed:|e2e_ms |vision_wait_ms |vision_ms |decoder_wait_ms |decode_ms )' \
                    "$RUN_DIR/output.log"
            else
                echo "FAILED: exit code $RESULT"
                tail -15 "$RUN_DIR/output.log"
            fi
        done
    done
done

echo "All experiment outputs: $SWEEP_DIR"
