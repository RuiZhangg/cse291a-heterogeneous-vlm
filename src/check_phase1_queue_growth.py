import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])

for rate in ("1.2", "2.2"):
    for tiles in (1, 3, 5):
        for device in ("GPU", "NPU"):
            path = (
                root / f"rate-{rate}" /
                f"tiles-{tiles}" / device / "results.csv"
            )

            if not path.exists():
                continue

            with path.open() as f:
                rows = sorted(
                    csv.DictReader(f),
                    key=lambda r: int(r["id"])
                )

            def arr(field, subset=rows):
                return np.array([
                    float(r[field]) for r in subset
                ])

            print(
                f"\nRate={rate} tiles={tiles} device={device}"
            )

            arrivals = arr("arrival")
            completes = arr("complete")

            observed_rate = (
                (len(rows)-1) /
                (arrivals[-1]-arrivals[0])
            )

            drain = (
                completes.max() - arrivals.max()
            )

            print(
                f"Observed arrival rate: "
                f"{observed_rate:.3f} req/s"
            )

            print(
                f"Time from last arrival to last completion: "
                f"{drain:.2f}s"
            )

            print(
                "Mean generated tokens:",
                round(arr("tokens").mean(), 2)
            )

            for start in range(0, len(rows), 30):
                sub = rows[start:start+30]

                print(
                    f"  Requests {start:03d}-{start+len(sub)-1:03d}:",
                    f"E2E p50={np.median(arr('e2e_ms',sub)):.0f}ms",
                    f"vision_wait_mean={arr('vision_wait_ms',sub).mean():.0f}ms",
                    f"decoder_wait_mean={arr('decoder_wait_ms',sub).mean():.0f}ms"
                )
