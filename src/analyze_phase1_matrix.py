import argparse
import csv
from pathlib import Path
from collections import Counter

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
a = ap.parse_args()

root = Path(a.root)
summaries = []
by_case = {}

def numbers(rows, field):
    return np.asarray(
        [float(r[field]) for r in rows],
        dtype=np.float64
    )

def percentile(x, p):
    return float(np.percentile(x, p))

for path in sorted(root.glob(
    "rate-*/tiles-*/*/results.csv"
)):
    device = path.parent.name
    tiles = int(path.parent.parent.name.split("-")[1])
    rate = float(path.parent.parent.parent.name.split("-")[1])

    with path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("EMPTY:", path)
        continue

    devices = Counter(r["device"] for r in rows)

    if devices != {device: len(rows)}:
        print("WARNING: Device mismatch:", path, devices)

    e2e = numbers(rows, "e2e_ms")
    arrivals = numbers(rows, "arrival")
    completions = numbers(rows, "complete")

    offered_span = arrivals.max() - arrivals.min()
    completion_span = completions.max() - arrivals.min()

    # These differ when requests queue substantially.
    observed_arrival_rate = (
        (len(rows) - 1) / offered_span
        if offered_span > 0 else 0
    )
    completion_throughput = (
        len(rows) / completion_span
        if completion_span > 0 else 0
    )

    def mean(field):
        return float(numbers(rows, field).mean())

    # Approximation using component timings.
    # Exact E2E first-token timestamps should be added later.
    e2e_ttft = (
        numbers(rows, "vision_wait_ms")
        + numbers(rows, "vision_ms")
        + numbers(rows, "decoder_wait_ms")
        + numbers(rows, "prep_ms")
        + numbers(rows, "ttft_ms")
    )

    record = {
        "nominal_rate": rate,
        "tiles": tiles,
        "device": device,
        "requests": len(rows),
        "observed_arrival_rate": observed_arrival_rate,
        "completion_throughput": completion_throughput,
        "e2e_mean_ms": float(e2e.mean()),
        "e2e_p50_ms": percentile(e2e, 50),
        "e2e_p95_ms": percentile(e2e, 95),
        "vision_wait_mean_ms": mean("vision_wait_ms"),
        "vision_mean_ms": mean("vision_ms"),
        "decoder_wait_mean_ms": mean("decoder_wait_ms"),
        "decode_mean_ms": mean("decode_ms"),
        "internal_ttft_mean_ms": mean("ttft_ms"),
        "approx_e2e_ttft_mean_ms": float(e2e_ttft.mean()),
        "mean_output_tokens": mean("tokens"),
    }

    summaries.append(record)
    by_case[(rate, tiles, device)] = rows

if not summaries:
    raise SystemExit("No results found")

out = root / "summary.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=summaries[0].keys()
    )
    writer.writeheader()
    writer.writerows(summaries)

print("\n========== PLACEMENT COMPARISON ==========")

for rate in sorted(set(r["nominal_rate"] for r in summaries)):
    for tiles in (1, 3, 5):
        g = next((
            r for r in summaries
            if r["nominal_rate"] == rate
            and r["tiles"] == tiles
            and r["device"] == "GPU"
        ), None)

        n = next((
            r for r in summaries
            if r["nominal_rate"] == rate
            and r["tiles"] == tiles
            and r["device"] == "NPU"
        ), None)

        if not g or not n:
            print(rate, tiles, "Incomplete pair")
            continue

        improvement = (
            100 * (g["e2e_p50_ms"] - n["e2e_p50_ms"])
            / g["e2e_p50_ms"]
        )

        gpu_rows = {
            int(r["id"]): r
            for r in by_case[(rate, tiles, "GPU")]
        }
        npu_rows = {
            int(r["id"]): r
            for r in by_case[(rate, tiles, "NPU")]
        }

        ids = set(gpu_rows) & set(npu_rows)

        matched_tokens = sum(
            gpu_rows[i]["tokens"] == npu_rows[i]["tokens"]
            for i in ids
        )

        matched_captions = sum(
            gpu_rows[i]["caption"] == npu_rows[i]["caption"]
            for i in ids
        )

        print(f"\nRate={rate}, tiles={tiles}")
        print(
            f"  Observed arrival rates: "
            f"GPU={g['observed_arrival_rate']:.2f}/s, "
            f"NPU={n['observed_arrival_rate']:.2f}/s"
        )
        print(
            f"  GPU p50/p95: "
            f"{g['e2e_p50_ms']:.1f}/"
            f"{g['e2e_p95_ms']:.1f} ms"
        )
        print(
            f"  NPU p50/p95: "
            f"{n['e2e_p50_ms']:.1f}/"
            f"{n['e2e_p95_ms']:.1f} ms"
        )
        print(
            f"  Median latency improvement using NPU: "
            f"{improvement:+.1f}%"
        )
        print(
            "  Decoder queue mean (GPU/NPU):",
            f"{g['decoder_wait_mean_ms']:.1f}/"
            f"{n['decoder_wait_mean_ms']:.1f} ms"
        )
        print(
            f"  Matched outputs: "
            f"tokens={matched_tokens}/{len(ids)}, "
            f"captions={matched_captions}/{len(ids)}"
        )

print("\nSaved:", out)
