import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])

for rate, tiles in (("1.5", 3), ("1.0", 5)):
    print(f"\n===== {tiles} tiles, {rate} req/s =====")

    for mode in ("periodic", "poisson-matched"):
        print(f"\nArrival: {mode}")

        for device in ("GPU", "NPU"):
            path = (
                root / f"rate-{rate}" /
                f"tiles-{tiles}" /
                mode / device / "results.csv"
            )

            if not path.exists():
                print(device, "MISSING")
                continue

            with path.open() as f:
                rows = sorted(
                    csv.DictReader(f),
                    key=lambda r: int(r["id"])
                )

            def v(key, subset=rows):
                return np.array([
                    float(r[key]) for r in subset
                ])

            arrivals = v("arrival")
            gaps = np.diff(arrivals)
            e2e = v("e2e_ms")

            early = rows[:30]
            late = rows[-30:]

            rate_actual = (
                (len(rows) - 1) /
                (arrivals[-1] - arrivals[0])
            )

            gap_cv = gaps.std() / gaps.mean()

            print(
                f"{device}: "
                f"arrival={rate_actual:.3f}/s "
                f"gap_CV={gap_cv:.2f} "
                f"p50={np.median(e2e):.1f}ms "
                f"p95={np.percentile(e2e,95):.1f}ms"
            )

            print(
                f"    vision_wait_mean="
                f"{v('vision_wait_ms').mean():.1f}ms "
                f"decoder_wait_mean="
                f"{v('decoder_wait_ms').mean():.1f}ms"
            )

            print(
                "    early/late p50:",
                f"{np.median(v('e2e_ms',early)):.1f}/"
                f"{np.median(v('e2e_ms',late)):.1f}ms"
            )
