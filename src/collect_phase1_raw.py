#!/usr/bin/env python3
"""Copy small Phase 1 raw results, keeping paths and hashes for reproducibility.

Run on wuklab NUC from repository root:
  python src/collect_phase1_raw.py

The script skips large media and model weights. It doesn't change raw runs.
"""
from pathlib import Path
from hashlib import sha256
import shutil

root = Path.cwd()
base = root / 'runs'
dest = root / 'results' / 'phase1' / 'raw_runs'
patterns = (
    '20261104-*-phase1-seed-*',
    '20261104-*-phase1-rate-sweep-seed-*',
    '20261104-*-matched-arrivals-seed-*',
    '20261104-*-prompt-length',
    '20261104-*-short-output-ablation',
)
allowed = {'results.csv', 'output.log', 'status.tsv', 'manifest.tsv',
           'metadata.txt', 'resource.txt', 'exit_code.txt'}
folders = sorted({p for pat in patterns for p in base.glob(pat) if p.is_dir()})
if not folders:
    raise SystemExit('No expected Phase 1 run folders found in runs/')
count = 0
for folder in folders:
    for f in folder.rglob('*'):
        if f.is_file() and f.name in allowed:
            target = dest / f.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            count += 1

manifest = dest / 'SHA256SUMS.txt'
with manifest.open('w') as out:
    for f in sorted(dest.rglob('*')):
        if f.is_file() and f != manifest:
            out.write(f'{sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest)}\n')
print(f'Copied {count} small files from {len(folders)} run directories to {dest}')
print('Manifest:', manifest)
print('Inspect size before staging: du -sh results/phase1/raw_runs')
