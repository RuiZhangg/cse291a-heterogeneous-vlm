import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])

stages = [
    "vision_wait_ms",
    "vision_ms",
    "decoder_wait_ms",
    "prep_ms",
    "decode_ms",
]

def describe(x):
    x = np.asarray(x, dtype=float)
    return (
        f"mean={x.mean():7.1f}  "
        f"p50={np.median(x):7.1f}  "
        f"p95={np.percentile(x,95):7.1f}"
    )

for rate in (2, 4.5, 5):
    for device in ("GPU", "NPU"):
        path = (
            root / f"rate-{rate}" / device / "results.csv"
        )

        if not path.exists():
            print("MISSING:", path)
            continue

        with path.open() as f:
            rows = list(csv.DictReader(f))

        def values(column):
            return np.array([
                float(r[column]) for r in rows
            ])

        e2e = values("e2e_ms")
        tokens = values("tokens")

        print(f"\n===== {rate} req/s | {device} =====")
        print("Requests:", len(rows))
        print("E2E latency:   ", describe(e2e))

        for stage in stages:
            print(f"{stage:17s}", describe(values(stage)))

        print("TTFT:          ", describe(values("ttft_ms")))
        print("Tokens:        ", describe(tokens))

        # Verify timestamp coverage and decomposition.
        component_sum = sum(values(s) for s in stages)
        residual = e2e - component_sum

        print(
            "Timing residual:",
            describe(residual)
        )

        # Approximate inter-token wall time.
        mask = tokens > 1

        if mask.any():
            per_token = (
                values("decode_ms")[mask]
                - values("ttft_ms")[mask]
            ) / (tokens[mask] - 1)

            print(
                "Inter-token ms:",
                describe(per_token)
            )

        # A fixed 350 ms exploratory deadline.
        print(
            "Deadline miss:",
            round(100 * np.mean(e2e > 350), 1), "%"
        )

        n = min(20, len(rows) // 2)

        if n:
            print(
                "First/last 20 median:",
                round(float(np.median(e2e[:n])),1),
                round(float(np.median(e2e[-n:])),1)
            )
