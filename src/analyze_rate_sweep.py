import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
cases = {}

for path in sorted(root.glob(
    "rate-*/tiles-*/*/results.csv"
)):
    rate = float(path.parts[-4].replace("rate-", ""))
    tiles = int(path.parts[-3].replace("tiles-", ""))
    device = path.parts[-2]

    with path.open() as f:
        rows = sorted(
            csv.DictReader(f),
            key=lambda r: int(r["id"])
        )

    def v(key, subset=rows):
        return np.asarray([
            float(r[key]) for r in subset
        ])

    arrivals = v("arrival")
    completes = v("complete")
    e2e = v("e2e_ms")
    vw = v("vision_wait_ms")
    dw = v("decoder_wait_ms")

    first = rows[:30]
    last = rows[-30:]

    result = {
        "arrival_rate": (
            (len(rows)-1) /
            (arrivals[-1]-arrivals[0])
        ),
        "throughput": (
            len(rows) /
            (completes.max()-arrivals.min())
        ),
        "p50": np.median(e2e),
        "p95": np.percentile(e2e,95),
        "vision_wait_mean": vw.mean(),
        "decoder_wait_mean": dw.mean(),
        "early_p50": np.median(v("e2e_ms", first)),
        "late_p50": np.median(v("e2e_ms", last)),
        "early_wait": (
            v("vision_wait_ms", first).mean()
            + v("decoder_wait_ms", first).mean()
        ),
        "late_wait": (
            v("vision_wait_ms", last).mean()
            + v("decoder_wait_ms", last).mean()
        ),
        "drain_s": completes.max()-arrivals.max(),
        "tokens_mean": v("tokens").mean(),
        "n":len(rows),
    }
    cases[(rate,tiles,device)] = result

for rate, tiles in sorted({
    (r,t) for (r,t,d) in cases
}):
    print(f"\n===== {tiles} tiles, rate={rate}/s =====")

    for device in ("GPU","NPU"):
        r = cases.get((rate,tiles,device))
        if r is None:
            print(device, "MISSING")
            continue

        print(
            f"{device}: "
            f"observed_arrival={r['arrival_rate']:.2f}/s "
            f"throughput={r['throughput']:.2f}/s "
            f"p50={r['p50']:.1f}ms "
            f"p95={r['p95']:.1f}ms "
            f"vision_wait={r['vision_wait_mean']:.1f}ms "
            f"decoder_wait={r['decoder_wait_mean']:.1f}ms"
        )

        print(
            f"    first30/last30 p50: "
            f"{r['early_p50']:.0f}/{r['late_p50']:.0f}ms; "
            f"combined wait: "
            f"{r['early_wait']:.0f}/{r['late_wait']:.0f}ms; "
            f"drain={r['drain_s']:.1f}s"
        )

    g = cases.get((rate,tiles,"GPU"))
    n = cases.get((rate,tiles,"NPU"))

    if g and n:
        improvement = (
            100*(g["p50"]-n["p50"])/g["p50"]
        )
        print(
            f"NPU median improvement: "
            f"{improvement:+.1f}%"
        )
