#!/usr/bin/env bash
# Optional sensitivity study: Does shortening serialized language work remove the reversal?
# This does NOT test a concurrent decoder: language worker remains serialized.
set -uo pipefail
cd /data/project-03
source envs/openvino/bin/activate
ROOT="runs/$(date +%Y%m%d-%H%M%S)-short-output-ablation"
mkdir -p "$ROOT"
echo "$ROOT" > runs/short_output_latest.txt
printf 'tiles\trate\tarrival\tdevice\texit\n' > "$ROOT/status.tsv"
for CONFIG in '3 1.5 2' '5 1.0 4'; do
  read -r TILES RATE SPATIAL <<< "$CONFIG"
  for MODE in periodic poisson-matched; do
    for DEVICE in GPU NPU; do
      DIR="$ROOT/tiles-$TILES/rate-$RATE/$MODE/$DEVICE"
      mkdir -p "$DIR"
      echo "==== tiles=$TILES rate=$RATE mode=$MODE device=$DEVICE ===="
      timeout 260s python -u src/pipeline_arrival_matched.py \
        --tiles "$SPATIAL" --thumbnail \
        --worker-mode serial --vision-device "$DEVICE" \
        --arrival-mode "$MODE" --phase-rates "$RATE,$RATE,$RATE" \
        --phase-requests 50 --seed 202 --warmups 2 --max-tokens 2 \
        --frames-dir data/frames --out "$DIR/results.csv" > "$DIR/run.log" 2>&1
      STATUS=$?
      printf '%s\t%s\t%s\t%s\t%s\n' "$TILES" "$RATE" "$MODE" "$DEVICE" "$STATUS" >> "$ROOT/status.tsv"
      if [ "$STATUS" -eq 0 ]; then grep -E '^(Completed:|e2e_ms |vision_wait_ms |decoder_wait_ms )' "$DIR/run.log"; else tail -15 "$DIR/run.log"; fi
    done
  done
done
echo "Results: $ROOT"
