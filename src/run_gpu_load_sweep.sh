#!/usr/bin/env bash
set -euo pipefail

cd /data/project-03

ROOT="runs/$(date +%Y%m%d-%H%M%S)-gpu-load-sweep"
mkdir -p "$ROOT"

for RATE in 0 2 5 8; do
    DIR="$ROOT/gpu-load-$RATE"
    mkdir -p "$DIR"

    echo "===== GPU background load: $RATE req/s ====="

    BG_PID=""

    if [ "$RATE" != "0" ]; then
        python -u src/vision_only_load.py \
            --device GPU \
            --rate "$RATE" \
            --duration 35 \
            --ready "$DIR/bg.ready" \
            --go "$DIR/bg.go" \
            > "$DIR/background.log" 2>&1 &

        BG_PID=$!

        for i in $(seq 1 150); do
            [ -f "$DIR/bg.ready" ] && break
            if ! kill -0 "$BG_PID" 2>/dev/null; then
                echo "Background failed"
                cat "$DIR/background.log"
                exit 1
            fi
            sleep 0.2
        done

        if [ ! -f "$DIR/bg.ready" ]; then
            echo "Background initialization timeout"
            kill "$BG_PID" 2>/dev/null || true
            exit 1
        fi

        touch "$DIR/bg.go"
        sleep 1
    fi

    STATUS=0
    timeout 80s python -u src/pipeline_video.py \
        --vision-device NPU \
        --rate 2 \
        --requests 30 \
        --warmups 2 \
        --max-tokens 64 \
        --frames-dir data/frames \
        --out "$DIR/results.csv" \
        > "$DIR/output.log" 2>&1 || STATUS=$?

    echo "$STATUS" > "$DIR/exit_code.txt"

    if [ -n "$BG_PID" ]; then
        kill "$BG_PID" 2>/dev/null || true
        wait "$BG_PID" 2>/dev/null || true
    fi

    if [ "$STATUS" -ne 0 ]; then
        echo "Foreground failed: $STATUS"
        tail -20 "$DIR/output.log"
    fi
done

echo "Results: $ROOT"
