#!/usr/bin/env python3
"""Audit and plot CSE 291A Option 2 Phase 1 inference measurements.

Run from /data/project-03:
 python src/phase1_audit_plots.py --base runs --out results/phase1/final
No web access, no pandas. Requires numpy + matplotlib.
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = argparse.ArgumentParser()
P.add_argument('--base', default='runs', help='directory holding dated run folders')
P.add_argument('--out', default='results/phase1/final')
A = P.parse_args()
base, out = Path(A.base), Path(A.out)
out.mkdir(parents=True, exist_ok=True)

# Matches the exact experiment families printed in their run directories.
families = {
 'matched': sorted(base.glob('*-matched-arrivals-seed-*')),
 'matrix': sorted(base.glob('*-phase1-seed-*')),
 'rates': sorted(base.glob('*-phase1-rate-sweep-seed-*')),
 'prompt': sorted(base.glob('*-prompt-length*')),
}
for name, roots in families.items():
 print(f'{name}: {len(roots)} directories')
 for root in roots: print('  ', root)

fieldnames = ['family','run','seed','rate_nominal','tiles','arrival_mode','device','n',
 'arrival_realized','throughput','p50_ms','p95_ms','mean_ms','vision_wait_mean_ms',
 'decoder_wait_mean_ms','vision_mean_ms','decode_mean_ms','first_token_internal_mean_ms',
 'output_tokens_mean','first30_p50_ms','last30_p50_ms','drain_s','max_host_rss_mib']
records=[]
raw_by_key={}
errors=[]


def fnums(rows, col):
 return np.array([float(r[col]) for r in rows],dtype=float)

def compute(family, root, path, mode, tiles, device, rate, seed):
 try:
  with path.open(newline='') as f: rows=list(csv.DictReader(f))
  rows.sort(key=lambda r:int(r['id']))
  if not rows: raise ValueError('empty CSV')
  n=len(rows)
  arrivals=fnums(rows,'arrival'); completes=fnums(rows,'complete')
  e2e=fnums(rows,'e2e_ms')
  realrate=(n-1)/(arrivals[-1]-arrivals[0]) if n>1 and arrivals[-1]>arrivals[0] else float('nan')
  throughput=n/(completes.max()-arrivals.min()) if completes.max()>arrivals.min() else float('nan')
  maximum_rss=float('nan')
  resource=path.parent/'resource.txt'
  if resource.exists():
   m=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resource.read_text(errors='replace'))
   if m: maximum_rss=int(m.group(1))/1024.0
  expected={device:n}
  dev_counts=dict(Counter(r['device'] for r in rows))
  if dev_counts!=expected: errors.append(f'DEVICE MISMATCH {path}: {dev_counts} expected {expected}')
  values=dict(family=family,run=root.name,seed=seed,rate_nominal=rate,tiles=tiles,
    arrival_mode=mode,device=device,n=n,arrival_realized=realrate,throughput=throughput,
    p50_ms=float(np.median(e2e)),p95_ms=float(np.percentile(e2e,95)),mean_ms=float(e2e.mean()),
    vision_wait_mean_ms=float(fnums(rows,'vision_wait_ms').mean()),
    decoder_wait_mean_ms=float(fnums(rows,'decoder_wait_ms').mean()),
    vision_mean_ms=float(fnums(rows,'vision_ms').mean()),
    decode_mean_ms=float(fnums(rows,'decode_ms').mean()),
    first_token_internal_mean_ms=float(fnums(rows,'ttft_ms').mean()),
    output_tokens_mean=float(fnums(rows,'tokens').mean()),
    first30_p50_ms=float(np.median(e2e[:30])),last30_p50_ms=float(np.median(e2e[-30:])),
    drain_s=float(completes.max()-arrivals.max()),max_host_rss_mib=maximum_rss)
  records.append(values)
  raw_by_key[(family,root.name,rate,tiles,mode,device)]={int(r['id']):r for r in rows}
 except Exception as exc: errors.append(f'FAILED READ {path}: {exc}')

for root in families['matched']:
 seed_match=re.search(r'seed-(\d+)$',root.name)
 seed=int(seed_match.group(1)) if seed_match else None
 for path in root.glob('rate-*/tiles-*/*/*/results.csv'):
  parts=path.relative_to(root).parts
  rate=float(parts[0][5:]); tiles=int(parts[1][6:]); mode=parts[2]; device=parts[3]
  compute('matched',root,path,mode,tiles,device,rate,seed)
for family in ('matrix','rates'):
 for root in families[family]:
  seed_match=re.search(r'seed-(\d+)$',root.name)
  seed=int(seed_match.group(1)) if seed_match else None
  for path in root.glob('rate-*/tiles-*/*/results.csv'):
   parts=path.relative_to(root).parts
   rate=float(parts[0][5:]); tiles=int(parts[1][6:]); device=parts[2]
   compute(family,root,path,'poisson',tiles,device,rate,seed)

# Prompt experiment path includes source's timing of first language token.
prompt=[]
for root in families['prompt']:
 for path in root.glob('tiles-*/repeat-*/*/results.csv'):
  parts=path.relative_to(root).parts
  tiles=int(parts[0][6:]); repeat=int(parts[1][7:]); device=parts[2]
  try:
   with path.open(newline='') as f: rows=list(csv.DictReader(f))
   log=path.parent/'output.log'
   match=re.search(r'Text question tokens:\s*(\d+)',log.read_text(errors='replace')) if log.exists() else None
   text_tokens=int(match.group(1)) if match else {0:11,32:107,128:395}.get(repeat)
   prompt.append(dict(root=root.name,tiles=tiles,repeat=repeat,device=device,
     prompt_tokens=text_tokens,n=len(rows),first_token_internal_ms=float(fnums(rows,'ttft_ms').mean()),
     vision_mean_ms=float(fnums(rows,'vision_ms').mean()),
     p50_ms=float(np.median(fnums(rows,'e2e_ms')))))
  except Exception as exc: errors.append(f'PROMPT READ {path}: {exc}')

def save_csv(filename, entries, cols):
 with (out/filename).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(entries)

save_csv('all_profiles.csv',records,fieldnames)
if prompt:save_csv('prompt_profiles.csv',prompt,list(prompt[0]))

# Repeated seeds: use mean and standard deviation of per-run medians, never pseudo-replicate requests.
summary=[]
groups=defaultdict(list)
for r in records:
 if r['family']=='matched':groups[(r['tiles'],r['rate_nominal'],r['arrival_mode'],r['device'])].append(r)
for (tiles,rate,mode,device),items in sorted(groups.items()):
 vals=[r['p50_ms'] for r in items]
 p95=[r['p95_ms'] for r in items]
 summary.append(dict(tiles=tiles,rate=rate,arrival_mode=mode,device=device,
  seeds=','.join(map(str,sorted(r['seed'] for r in items))),n_runs=len(items),
  per_run_medians_ms=';'.join(f'{v:.2f}' for v in vals),
  mean_run_median_ms=statistics.mean(vals),
  sd_run_median_ms=statistics.stdev(vals) if len(vals)>1 else 0,
  mean_run_p95_ms=statistics.mean(p95),
  mean_vision_wait_ms=statistics.mean(r['vision_wait_mean_ms'] for r in items),
  mean_decoder_wait_ms=statistics.mean(r['decoder_wait_mean_ms'] for r in items),
  mean_host_rss_mib=statistics.mean(r['max_host_rss_mib'] for r in items if np.isfinite(r['max_host_rss_mib'])) if any(np.isfinite(r['max_host_rss_mib']) for r in items) else float('nan')))
if summary:save_csv('three_seed_summary.csv',summary,list(summary[0]))

# Check matched output length and caption equality for each GPU / NPU pair.
matched=[]
for root in families['matched']:
 for rate,tiles in [(1.5,3),(1.0,5)]:
  for mode in ('periodic','poisson-matched'):
   gr=raw_by_key.get(('matched',root.name,rate,tiles,mode,'GPU'))
   nr=raw_by_key.get(('matched',root.name,rate,tiles,mode,'NPU'))
   if gr is None or nr is None: continue
   keys=sorted(set(gr)&set(nr))
   matched.append(dict(root=root.name,rate=rate,tiles=tiles,mode=mode,n=len(keys),
      same_token_count=sum(gr[i]['tokens']==nr[i]['tokens'] for i in keys),
      same_caption=sum(gr[i]['caption']==nr[i]['caption'] for i in keys)))
if matched: save_csv('output_agreement.csv',matched,list(matched[0]))

# Aggregate compile log lines from all outputs; compilation measurements from warmed code cache
# are not interchangeable with cold first-ever NPU compilation.
compile_pat=re.compile(r'^(vision_embeddings_model|text_embeddings_model|language_model)\s+(GPU|NPU|CPU)\s+compile_s\s+([0-9.]+)\s+execution_devices\s+(.+)$',re.M)
compile_rows=[]
for family in ('matched','matrix','rates','prompt'):
 for root in families[family]:
  for log in root.rglob('output.log'):
   try:
    for name,device,secs,execdev in compile_pat.findall(log.read_text(errors='replace')):
     compile_rows.append(dict(run=root.name,filename=str(log.relative_to(root)),model=name,device=device,compile_s=float(secs),execution_device=execdev.strip()))
   except Exception as exc:errors.append(f'COMPILATION READ {log}: {exc}')
if compile_rows:save_csv('compile_observations.csv',compile_rows,list(compile_rows[0]))

# Tables for the report.
if summary:
 fig,axes=plt.subplots(1,2,figsize=(10.8,3.5),sharey=False)
 for ax,tiles,rate in zip(axes,[3,5],[1.5,1.0]):
  x=np.arange(2);width=.32
  for k,device in enumerate(['GPU','NPU']):
   ys=[]; es=[]
   for mode in ('periodic','poisson-matched'):
    hit=[r for r in summary if r['tiles']==tiles and r['rate']==rate and r['arrival_mode']==mode and r['device']==device]
    ys.append(hit[0]['mean_run_median_ms'] if hit else np.nan)
    es.append(hit[0]['sd_run_median_ms'] if hit else np.nan)
   ax.bar(x+(k-.5)*width,ys,width,yerr=es,capsize=3,label=f'{device} vision')
  ax.set_xticks(x,['Periodic','Irregular'])
  ax.set_title(f'{tiles} visual tiles · {rate:g} req/s')
  ax.set_ylabel('Mean of run medians (ms)')
  ax.legend(fontsize=8,frameon=False)
  ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('Placement preference reverses with arrival pattern (three seeds)',fontsize=12)
 fig.tight_layout();fig.savefig(out/'fig1_placement_reversal.png',dpi=210,bbox_inches='tight');plt.close(fig)

 fig,axes=plt.subplots(1,2,figsize=(10.8,3.2),sharey=False)
 for ax,tiles,rate in zip(axes,[3,5],[1.5,1.0]):
  keys=[('GPU','periodic'),('NPU','periodic'),('GPU','poisson-matched'),('NPU','poisson-matched')]
  vis=[];dec=[]
  for dev,mode in keys:
   hit=[r for r in summary if r['tiles']==tiles and r['rate']==rate and r['arrival_mode']==mode and r['device']==dev]
   vis.append(hit[0]['mean_vision_wait_ms'] if hit else np.nan)
   dec.append(hit[0]['mean_decoder_wait_ms'] if hit else np.nan)
  x=np.arange(4)
  ax.bar(x,vis,label='Vision wait')
  ax.bar(x,dec,bottom=np.asarray(vis),label='Decoder wait')
  ax.set_xticks(x,['GPU\nperiodic','NPU\nperiodic','GPU\nirregular','NPU\nirregular'],fontsize=8)
  ax.set_title(f'{tiles} tiles · {rate:g}/s')
  ax.set_ylabel('Mean queue wait per request (ms)')
  ax.legend(fontsize=8,frameon=False)
  ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.suptitle('Queueing shifts from language to vision when offloading',fontsize=12)
 fig.tight_layout();fig.savefig(out/'fig2_queue_decomposition.png',dpi=210,bbox_inches='tight');plt.close(fig)

if any(r['family'] in ('matrix','rates') for r in records):
 fig,axes=plt.subplots(1,2,figsize=(10.8,3.35))
 for ax,tiles in zip(axes,(3,5)):
  for dev in ('GPU','NPU'):
   rr=sorted((r for r in records if r['family'] in ('matrix','rates') and r['seed']==101 and r['tiles']==tiles and r['device']==dev),key=lambda r:r['arrival_realized'])
   ax.plot([r['arrival_realized'] for r in rr],[r['p50_ms'] for r in rr],marker='o',ms=4,label=dev)
  ax.set_title(f'{tiles} visual tiles')
  ax.set_xlabel('Observed arrival rate (requests/s)')
  ax.set_ylabel('Median end-to-end latency (ms; log scale)')
  ax.set_yscale('log');ax.legend(frameon=False)
  ax.grid(alpha=.2)
 fig.suptitle('Load and complexity-dependent crossover / overload (development trace)',fontsize=12)
 fig.tight_layout();fig.savefig(out/'fig3_arrival_rate_sweep.png',dpi=210,bbox_inches='tight');plt.close(fig)

if prompt:
 fig,ax=plt.subplots(figsize=(6.9,3.4))
 for tiles in (1,3):
  for dev in ('GPU','NPU'):
   rr=sorted((r for r in prompt if r['tiles']==tiles and r['device']==dev),key=lambda r:r['prompt_tokens'])
   if rr:ax.plot([r['prompt_tokens'] for r in rr],[r['first_token_internal_ms'] for r in rr],marker='o',label=f'{tiles} tile(s), {dev} vision')
 ax.set_xlabel('Text question tokens (excludes image tokens)')
 ax.set_ylabel('Internal first-token computation (ms)')
 ax.set_title('Prompt length increases language prefill cost')
 ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'fig4_prompt_length.png',dpi=210,bbox_inches='tight');plt.close(fig)

# Executive audit of data available for final report.
with (out/'audit.txt').open('w') as f:
 f.write('PHASE 1 RAW-DATA AUDIT\n=====================\n')
 for name,roots in families.items():f.write(f'{name}: {len(roots)} run directories\n')
 f.write(f'Parsed profile configs: {len(records)}\n')
 f.write(f'Parsed matched-arrival seed configurations: {sum(r["family"]=="matched" for r in records)}\n')
 f.write(f'Parsed prompt configs: {len(prompt)}\n')
 f.write(f'Compiler observations: {len(compile_rows)}\n')
 f.write(f'Matched output comparisons: {len(matched)}\n')
 f.write(f'Max RSS observations: {sum(np.isfinite(r["max_host_rss_mib"]) for r in records)}\n')
 f.write('\nNOT YET VERIFIED AUTOMATICALLY: startup/init time distinct from compilation; actual host->device transfer; GPU/NPU device memory; source image provenance/checkpoint revision; full source snapshots.\n')
 f.write('\nWARNINGS:\n')
 for e in errors: f.write(e+'\n')

print('\nSaved outputs to',out)
for item in sorted(out.glob('fig*.png')):print('Figure:',item)
print('Profiles:',out/'all_profiles.csv')
print('Audit:',out/'audit.txt')
if errors:
 print('WARNING:',len(errors),'parse/device issues; read audit.txt')
