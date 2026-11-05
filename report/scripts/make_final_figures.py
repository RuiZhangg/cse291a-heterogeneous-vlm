#!/usr/bin/env python3
"""Phase 1 final plots: run-level points and log-scale visible queues."""
import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ap=argparse.ArgumentParser()
ap.add_argument('--profiles',default='all_profiles.csv')
ap.add_argument('--out',default='.')
a=ap.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(a.profiles)
df=df[df.family=='matched'].copy()
df['arrival_mode']=df.arrival_mode.map({'periodic':'Periodic','poisson-matched':'Irregular'})
order=['Periodic','Irregular']
styles={'GPU':('o','C0'),'NPU':('s','C1')}
# Main figure: four panels with p50 and p95; each marker is one independent seed,
# bars/connecting line are the mean across independent seeds (n=3).
fig,axs=plt.subplots(2,2,figsize=(10.7,5.9),sharex='col',constrained_layout=True)
for c,(tiles,rate) in enumerate([(3,1.5),(5,1.0)]):
    dd=df[(df.tiles==tiles)&(np.isclose(df.rate_nominal,rate))]
    for r,(metric,label) in enumerate([('p50_ms','Median (p50)'),('p95_ms','Tail (p95)')]):
        ax=axs[r,c]
        for dev,(marker,col) in styles.items():
            means=[]
            for k,mode in enumerate(['Periodic','Irregular']):
                v=dd[(dd.device==dev)&(dd.arrival_mode==mode)].sort_values('seed')[metric].to_numpy()
                assert len(v)==3,(dev,mode,tiles,metric,v)
                means.append(np.mean(v))
                offsets=np.array([-0.058,0.,0.058]) + (-0.045 if dev=='GPU' else 0.045)
                ax.scatter(k+offsets,v,marker=marker,color=col,s=20,alpha=.55,zorder=3,linewidths=0)
            ax.plot([0,1],means,marker=marker,color=col,label=dev+' vision',linewidth=2,markersize=6,zorder=4)
        ax.set_xticks([0,1],order)
        ax.set_xlim(-.20,1.20)
        ax.set_ylabel(label+' latency (ms)')
        ax.set_title(f'{tiles} visual tiles · {rate:g} req/s' if r==0 else '')
        ax.grid(axis='y',alpha=.22)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p: f'{x:,.0f}'))
        if r==0: ax.legend(loc='upper left',fontsize=9,frameon=False)
fig.suptitle('Request latency across arrival patterns and three seeds',fontsize=14,fontweight='semibold')
fig.savefig(out/'fig1_placement_reversal_p50_p95.png',dpi=210,bbox_inches='tight')
fig.savefig(out/'fig1_placement_reversal_p50_p95.pdf',bbox_inches='tight')
plt.close(fig)

# queue graph using LOG SCALE so periodic ~0.1ms is visible.
# 3seed average wait per request for each stage. Zero not present; measured >=0.09ms.
fig,axs=plt.subplots(1,2,figsize=(10.7,3.25),sharey=True,constrained_layout=True)
for c,(tiles,rate) in enumerate([(3,1.5),(5,1.0)]):
    ax=axs[c]
    dd=df[(df.tiles==tiles)&(np.isclose(df.rate_nominal,rate))]
    labels=['GPU\nPeriodic','NPU\nPeriodic','GPU\nIrregular','NPU\nIrregular']
    xx=np.arange(4)
    vw=[]; dw=[]
    for device,mode in [('GPU','Periodic'),('NPU','Periodic'),('GPU','Irregular'),('NPU','Irregular')]:
        sub=dd[(dd.device==device)&(dd.arrival_mode==mode)]
        assert len(sub)==3
        vw.append(sub.vision_wait_mean_ms.mean())
        dw.append(sub.decoder_wait_mean_ms.mean())
    # Side-by-side bars instead of stacked on log axis (which is misleading).
    ax.bar(xx-.18,vw,width=.33,label='Vision queue',zorder=3)
    ax.bar(xx+.18,dw,width=.33,label='Language queue',zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(0.055,1600)
    ax.set_yticks([0.1,1,10,100,1000])
    ax.set_yticklabels(['0.1','1','10','100','1,000'])
    ax.set_xticks(xx,labels,fontsize=9)
    ax.set_title(f'{tiles} tiles · {rate:g} req/s')
    ax.grid(axis='y',which='major',alpha=.25)
    ax.set_axisbelow(True)
    for x,v,d in zip(xx,vw,dw):
        for xpos,val in [(x-.18,v),(x+.18,d)]:
            ax.annotate(f'{val:.1f}' if val<1 else f'{val:.0f}',(xpos,val),
                xytext=(0,3),textcoords='offset points',ha='center',fontsize=7.5,rotation=0)
axs[0].set_ylabel('Mean queue wait per request (ms, log scale)')
axs[1].legend(loc='upper left',fontsize=9,frameon=False)
fig.suptitle('Where requests wait: vision vs language stage',fontsize=14,fontweight='semibold')
fig.savefig(out/'fig2_queue_wait_log.png',dpi=210,bbox_inches='tight')
fig.savefig(out/'fig2_queue_wait_log.pdf',bbox_inches='tight')
plt.close(fig)
print('Generated new figures in',out)
