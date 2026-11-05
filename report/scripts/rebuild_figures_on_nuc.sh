#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
mkdir -p report/figures
PROFILES=results/phase1/final/all_profiles.csv
[[ -f "$PROFILES" ]] || { echo "Missing $PROFILES" >&2; exit 1; }
# Main two figures: reproducible from the committed, audited aggregate CSV.
python report/scripts/make_final_figures.py \
  --profiles "$PROFILES" --out report/figures
# Supplemental short-output figure: reproducible only when the raw traces are available.
FULL=runs/20261104-125358-matched-arrivals-seed-202
SHORT=runs/20261104-202850-short-output-ablation
if [[ -d "$FULL" && -d "$SHORT" ]]; then
  python report/plot_short_output_from_raw.py \
    --full "$FULL" --short "$SHORT" --out report/figures
else
  echo "Skipping optional short-output figure: raw traces are unavailable." >&2
fi
printf 'Figure files written to %s/report/figures\n' "$ROOT"
