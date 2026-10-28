# CSE 291A Option 2: Heterogeneous VLM Exploration
Date: 2026-10-28
Machine: Intel ASUS NUC 16 Pro
Team: 03
Workspace: /data/project-03

## System

InternVL2-1B INT4 OpenVINO IR.

CPU: image preprocessing, application queues.
Vision: OpenVINO on GPU or NPU, 448x448 static input.
Text embeddings: GPU.
Language decoder: GPU, stateful autoregressive inference.

Vision features: 256 x 896 per single image tile.

Both vision models can remain resident and are selectable
per request, without recompilation.

## Compatibility

- Full VLM on GPU: works.
- Full VLM on CPU: works, but much slower.
- Full VLM on NPU: compilation failure for downloaded model.
- Vision encoder on NPU with dynamic shapes: fails.
- Vision encoder on NPU with static 448x448: works.
- GPU/NPU vision feature cosine similarity: approximately 0.999385.
- Mixed NPU-vision/GPU-language caption generation: works.

## Main empirical findings

1. Under low load, all-GPU has lower latency than NPU+GPU.

2. Under high load, GPU vision execution competes with
   GPU language decoding. Moving vision to NPU can improve
   pipeline throughput and queueing latency.

3. On a fixed 14-token caption workload:
   - 24-request burst GPU: approximately 4.45 req/s.
   - 24-request burst NPU+GPU: approximately 5.37 req/s.

4. On equal-token 32/64-token workloads:
   - 32 tokens: GPU 2.98, NPU+GPU 3.35 req/s.
   - 64 tokens: GPU 1.85, NPU+GPU 1.97 req/s.
   - Advantage decreases as language generation dominates.

5. CPU vision is too slow for tested high-rate workloads:
   approximately 860 ms per image and 1.15 req/s
   under saturation.

6. Real video-frame tests:
   - At 4.0 and 4.5 req/s, all-GPU has lower latency.
   - At 5.0 req/s, NPU+GPU has lower latency.
   - Captions are plausible, but require visual quality review.
   - GPU/NPU exact caption match: 19/32 images.

7. Phase-aware 4->5->4 periodic routing:
   - Static GPU overall p95: 369.8 ms.
   - Static NPU overall p95: 353.5 ms.
   - Phase-aware overall p95: 325.3 ms.

8. Under Poisson arrivals, phase-aware routing may
   underperform static NPU because rate averages
   do not capture short bursts and backlog.

9. Simple online queue-aware routing:
   - Reduces mean latency versus static NPU across
     three tested Poisson traces (4->5->4).
   - Reduces p95 in two of three traces.
   - Selects NPU for approximately 73-76% of requests.
   - Requires comparison against matched random routing.

## Important limitations

- Only one machine and one VLM.
- Single 448x448 image tile; no multi-tile preprocessing.
- Video frames are predecoded and preprocessed.
- Single GPU decoder worker; no concurrent decoder batching.
- Only frame-level captions; no temporal understanding.
- Captions have differing output-token lengths between devices.
- Limited quality evaluation and no independent annotations.
- Results on a shared machine may be affected by external load.
- These are exploratory measurements, not final statistical claims.

## Potential Phase 1 question

When does offloading visual encoding to NPU benefit
streaming vision-language inference compared with all-GPU,
and what workload characteristics determine the crossover?

## Potential Phase 2 question

Can an online scheduler use queue state and measured
service times to outperform static GPU/NPU placement
under irregular and changing video-frame workloads?

## Next research tasks

- Replicate with more seeds and representative videos.
- Introduce variable output lengths and multi-tile image inputs.
- Evaluate caption quality and quality/performance tradeoffs.
- Evaluate smarter admission, routing, and queueing policies.
- Measure full video preprocessing and streaming latency.

## Final exploration: Online scheduling controls

### Random-routing control: Poisson 4 -> 5 -> 4 req/s

Both policies use approximately 25% GPU vision and 75% NPU vision.
Random routing chooses devices independently of queue state.

| Seed | Random mean (ms) | Online mean (ms) | Online improvement |
|---|---:|---:|---:|
| 101 | 508.1 | 411.9 | 18.9% |
| 202 | 651.3 | 499.6 | 23.3% |
| 303 | 623.7 | 493.4 | 20.9% |

Online routing beat random routing in mean, median, and p95
latency for all three paired traces.

This suggests queue-state information adds value beyond using
NPU vision for most requests.

### Quiet -> busy -> quiet: Poisson 2 -> 5 -> 2 req/s

| Seed | GPU mean | NPU mean | Online mean |
|---|---:|---:|---:|
| 202 | 492.5 | 423.1 | 351.9 |
| 303 | 563.5 | 452.3 | 361.3 |

All values are milliseconds.

For seed 101, online median was 285.75 ms, compared with
310.18 ms on GPU and 346.24 ms on NPU. Its p95 was slightly
worse than static NPU.

Online routing achieved the best mean latency in seeds 202
and 303 and the best median latency in all three tested traces.

### Revised conclusions

1. Mixed NPU-vision/GPU-language inference works.
2. All-GPU has better isolated and lightly loaded latency.
3. NPU offloading can improve throughput and tail latency
   under GPU contention.
4. CPU offloading is substantially slower on this hardware.
5. The preferred placement depends on arrival rate and
   generation workload.
6. Phase-average-rate routing is insufficient for bursty traffic.
7. Online queue-aware routing is technically feasible.
8. Queue-aware placement outperformed matched random routing
   in all three exploratory trials.
9. Online routing does not consistently outperform static NPU
   on every tail-latency metric.

### Research direction

Phase 1:
Characterize the performance crossover and quantify contention
in heterogeneous streaming vision-language inference.

Phase 2:
Develop and evaluate online placement policies using device
availability, decoder backlog, predicted service time, and
workload intensity.

### Remaining limitations

- Single hardware platform and VLM.
- Single 448x448 image tile per frame.
- Predecoded and preprocessed frames.
- Single stateful GPU decoder worker.
- Limited caption-quality evaluation.
- Only three stochastic seeds for key comparisons.
- No independent temporal video reasoning.
- No rigorous analysis of external shared-machine load.
- Current online policy is a baseline heuristic, not an
  established optimal scheduler.
