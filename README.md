# CSE 291A — Team 03 | GPU/NPU-Aware Vision-Language Inference

**Project option:** 2 — CPU/GPU/NPU-Aware Agent Pipelines  
**Team member:** Rui Zhang (`ruz093@ucsd.edu`)

This project investigates when the visual encoder of a vision-language model should run on an integrated GPU or an NPU. On an Intel NUC 16 Pro, we run the visual encoder of InternVL2-1B INT4 on either device and run text embeddings and language generation on the GPU. Phase 1 profiles request and stage latency as image tile count, text prompt length, request rate, and arrival pattern change. Phase 2 will investigate online, workload-aware placement.

## Environment

- **Machine:** Intel NUC 16 Pro (`wuklab-NUC16`), Ubuntu 24.04, integrated Intel GPU and NPU.
- **Python:** 3.12.3, using the project environment `envs/openvino/`.
- **Inference framework:** OpenVINO 2026.1.0 (`2026.1.0-21367-63e31528c62-releases/2026/1`).
- **Model:** [`OpenVINO/InternVL2-1B-int4-ov`](https://huggingface.co/OpenVINO/InternVL2-1B-int4-ov), pinned to revision `3d04b8079cffb4aa6ebd27b80bcd0f54d8dffadb`.
- **Model directory:** `/data/project-03/models/InternVL2-1B-int4-ov/`. The current pipeline scripts refer to this absolute path.

Model weights, virtual environments, and OpenVINO caches are excluded from Git. To restore the exact model revision on the lab machine:

```bash
cd /data/project-03
source envs/openvino/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenVINO/InternVL2-1B-int4-ov",
    revision="3d04b8079cffb4aa6ebd27b80bcd0f54d8dffadb",
    local_dir="models/InternVL2-1B-int4-ov",
)
PY
```

The pinned revision is also recorded in `results/phase1/metadata/model_revision.txt`. SHA-256 manifests document the 27 local model files and the 40 benchmark images. Verify the local files from the repository root with:

```bash
sha256sum -c results/phase1/metadata/model_files.sha256
sha256sum -c results/phase1/metadata/frames.sha256
```

## Input data

The benchmark uses **40 JPEG frames**, with 20 frames from each of two Intel sample videos:

- **People detection:** [`people-detection.mp4`](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4), saved as `data/frames/people/frame_001.jpg` through `frame_020.jpg`.
- **Person, bicycle, and car detection:** [`person-bicycle-car-detection.mp4`](https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4), saved as `data/frames/traffic/frame_001.jpg` through `frame_020.jpg`.

The extracted frames are included in the repository, so rerunning the benchmarks does not require extracting the videos again. `results/phase1/metadata/frames.sha256` identifies the exact inputs. The frame sampling procedure is not specified here; use the committed files for exact reproduction. Check upstream usage and redistribution terms before publishing the image files in a public repository.

An additional [PyTorch Hub dog image](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg) was used for early development and is **not** part of the 40-frame benchmark.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/` | VLM pipeline, profiling variants, benchmark runners, and analysis scripts |
| `data/frames/` | The 40 committed benchmark frames |
| `models/` | Downloaded model files, excluded from Git |
| `runs/` | Timestamped raw run directories, excluded from Git |
| `results/phase1/final/` | Curated per-configuration summaries and audit tables |
| `results/phase1/logs/` | Selected console and profiling logs |
| `results/phase1/metadata/` | Environment, revision, device information, and checksums |
| `results/phase1/raw_runs/` | Selected request-level CSVs, if packaged for submission |
| `report/` | LaTeX report, plotting scripts, and generated figures |

## Quick smoke test

Run on the lab NUC from the repository root:

```bash
cd /data/project-03
source envs/openvino/bin/activate
mkdir -p runs/smoke
python src/pipeline_real_tiles_fixed.py \
    --tiles 1 --vision-device GPU --worker-mode serial \
    --phase-rates 1,1,1 --phase-requests 2 --seed 101 \
    --warmups 1 --max-tokens 2 --frames-dir data/frames \
    --out runs/smoke/results.csv
```

A successful run reports `Completed: 6` and writes six request-level records to `runs/smoke/results.csv`. Vision can also run on the NPU by changing `--vision-device` to `NPU`.

## Reproduce the main Phase 1 experiment

The main experiment compares periodic arrivals with rate-matched irregular arrivals at **1.5 requests/s with 3 image tiles** and **1.0 requests/s with 5 image tiles**. Each configuration processes 150 requests. The GPU and NPU runs use matched arrival schedules, and the comparison is repeated for three seeds. Language generation uses one serialized GPU worker.

```bash
cd /data/project-03
source envs/openvino/bin/activate
for SEED in 101 202 303; do
    RUN_SEED="$SEED" bash src/run_matched_arrivals.sh
done
```

Runs are saved under timestamped directories in `runs/`. To recalculate summary tables from available raw runs and regenerate the figures:

```bash
python src/phase1_audit_plots.py --base runs --out results/phase1/final
bash report/scripts/rebuild_figures_on_nuc.sh
```

The two main report figures can also be recreated using `results/phase1/final/all_profiles.csv` and `report/scripts/make_final_figures.py`. The optional short-output figure requires the matching raw request CSVs. All report figures are generated in Python.

## Results and scope

Phase 1 finds that the faster isolated GPU vision encoder is generally preferable for regularly spaced requests, while offloading vision to NPU can improve request latency under irregular arrivals. The placement reversal appears in all three tested seeds for the two selected workloads. Shorter output generations reduce the median NPU advantage, indicating that downstream language work matters to placement decisions.

Measurements include per-stage and end-to-end latency, queue waiting, compilation, initialization, host-process peak RSS, and selected host-side feature-copy and preparation operations. Physical host-to-accelerator transfer and accelerator-specific memory are not isolated. Results characterize the tested workload and serialized language worker rather than all VLM serving architectures.

See `report/phase1_report.tex` and the curated results under `results/phase1/` for details. Phase 2 will evaluate a queue-, device-state-, and request-complexity-aware placement policy against fixed placement baselines on held-out workloads.
