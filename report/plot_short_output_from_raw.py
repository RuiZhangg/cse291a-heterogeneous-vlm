#!/usr/bin/env python3
"""Reproduce the short-output sensitivity figure from NUC request-level CSVs.

Example (on /data/project-03):
 python report/plot_short_output_from_raw.py \
   --full runs/20261104-125358-matched-arrivals-seed-202 \
   --short runs/20261104-202850-short-output-ablation \
   --out report

This script does NOT infer anything from console summaries. Both inputs must
have real per-request results.csv files from the NUC benchmark scripts.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--full',required=True,type=Path)
p.add_argument('--short',required=True,type=Path)
p.add_argument('--out',required=True,type=Path)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
rows=[]
for tiles,rate in ((3,1.5),(5,1.0)):
 for mode in ('periodic','poisson-matched'):
  for dev in ('GPU','NPU'):
   for max_tok, root in ((32,a.full),(2,a.short)):
    if max_tok==32:
     path=root/f'rate-{rate}/tiles-{tiles}/{mode}/{dev}/results.csv'
    else:
     path=root/f'tiles-{tiles}/rate-{rate}/{mode}/{dev}/results.csv'
    if not path.is_file():
     raise SystemExit(f'Missing raw file: {path}')
    with path.open() as f:
     records=list(csv.DictReader(f))
    if len(records)!=150:
     raise SystemExit(f'Expected 150 requests at {path}, got {len(records)}')
    devices={r['device'] for r in records}
    if devices!={dev}:
     raise SystemExit(f'Device mismatch: {path}: {devices}')
    latency=np.array([float(r['e2e_ms']) for r in records])
    tokens=np.array([int(r['tokens']) for r in records])
    rows.append(dict(tiles=tiles,rate=rate,mode=mode,device=dev,max_tokens=max_tok,
                     n=len(records),p50_ms=np.median(latency),
                     p95_ms=np.percentile(latency,95),tokens_mean=np.mean(tokens)))
summary=a.out/'short_output_ablation_summary.csv'
with summary.open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print('Saved',summary)
fig,axs=plt.subplots(1,2,figsize=(9.4,3.55),constrained_layout=True)
colors={'GPU':'C0','NPU':'C1'}
for i,(tiles,rate) in enumerate(((3,1.5),(5,1.0))):
 ax=axs[i]
 x=np.arange(2)
 for cap,hatched,offset in ((32,'',-.18),(2,'//',.18)):
  for dev in ('GPU','NPU'):
   vals=[]
   for mode in ('periodic','poisson-matched'):
    data=[r for r in rows if r['tiles']==tiles and r['mode']==mode and r['device']==dev and r['max_tokens']==cap]
    assert len(data)==1
    vals.append(data[0]['p50_ms'])
   shift=offset+(-.085 if dev=='GPU' else .085)
   ax.bar(x+shift,vals,width=.16,color=colors[dev],hatch=hatched,
          label=f'{dev} vision, max {cap} tokens',edgecolor='white',linewidth=.4)
 ax.set_xticks(x,['Periodic','Irregular'])
 ax.set_ylabel('End-to-end median latency (ms)')
 ax.set_title(f'{tiles} visual tiles · {rate:g} req/s · seed 202')
 ax.grid(axis='y',alpha=.2)
 ax.set_axisbelow(True)
axs[0].legend(loc='upper left',fontsize=7,frameon=False,ncol=1)
fig.suptitle('Generation-length sensitivity: fixed serial GPU language worker',fontsize=11)
for ext in ('png','pdf'):
 fig.savefig(a.out/f'fig3_short_output_sensitivity.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print('Saved',a.out/'fig3_short_output_sensitivity.png')
