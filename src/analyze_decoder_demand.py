import csv
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])

def load(tokens, mode):
    path = root / f"tokens-{tokens}" / mode / "results.csv"
    with path.open() as f:
        return list(csv.DictReader(f))

def values(rows, key):
    return np.array([float(r[key]) for r in rows])

def overlap(intervals_a, intervals_b):
    total = 0.0
    for a0, a1 in intervals_a:
        for b0, b1 in intervals_b:
            total += max(0.0, min(a1,b1) - max(a0,b0))
    return total

for tokens in (2,6,64):
    print(f"\n========== MAX TOKENS {tokens} ==========")

    outputs = {}

    for mode in ("serial", "parallel"):
        rows = load(tokens,mode)
        outputs[mode] = {int(r["id"]):r for r in rows}

        e2e = values(rows,"e2e_ms")
        vwait = values(rows,"vision_wait_ms")
        dwait = values(rows,"decoder_wait_ms")
        vtime = values(rows,"vision_ms")
        dtime = values(rows,"decode_ms")

        span = (
            values(rows,"complete").max()
            - values(rows,"arrival").min()
        )

        gpu_vision = [
            (float(r["vision_start"]),
             float(r["vision_end"]))
            for r in rows if r["device"]=="GPU"
        ]

        npu_vision = [
            (float(r["vision_start"]),
             float(r["vision_end"]))
            for r in rows if r["device"]=="NPU"
        ]

        decoding = [
            (float(r["decode_start"]),
             float(r["complete"]))
            for r in rows
        ]

        decoder_overlap = overlap(gpu_vision,decoding)
        decoder_total = sum(b-a for a,b in decoding)

        print(f"\n{mode.upper()}")
        print("Completion throughput:",
              round(len(rows)/span,2), "req/s")
        print("Actual mean tokens:",
              round(values(rows,"tokens").mean(),2))
        print("E2E mean / p50 / p95:",
              *(round(x,1) for x in (
                  e2e.mean(),np.median(e2e),
                  np.percentile(e2e,95)
              )))
        print("Mean vision wait:",round(vwait.mean(),1))
        print("Mean vision execution:",round(vtime.mean(),1))
        print("Mean decoder wait:",round(dwait.mean(),1))
        print("Mean decoder execution:",round(dtime.mean(),1))
        print("GPU/NPU vision overlap:",
              round(overlap(gpu_vision,npu_vision),2),"s")
        print("Decoder time overlapping GPU vision:",
              round(100*decoder_overlap/decoder_total,1),"%")

    s,p = outputs["serial"],outputs["parallel"]
    ids = sorted(set(s)&set(p))

    print("\nPAIRED-CONTROL VALIDATION")
    print("Same device:",
          sum(s[i]["device"]==p[i]["device"] for i in ids),
          "/",len(ids))
    print("Same token count:",
          sum(s[i]["tokens"]==p[i]["tokens"] for i in ids),
          "/",len(ids))
    print("Same caption:",
          sum(s[i]["caption"]==p[i]["caption"] for i in ids),
          "/",len(ids))
