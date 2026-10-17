"""Matplotlib Agg renderers; all figures have auditable numeric sidecars."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from common import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
def footer(fig,text,mock):
 full=('MOCK - ENGINEERING FIXTURE ONLY | ' if mock else '')+text
 wrapped='\n'.join(textwrap.fill(line,width=int(fig.get_figwidth()*15)) for line in full.split('\n'))
 fig.text(.02,.015,wrapped,fontsize=8,color='#a13333' if mock else '#444444')
def curve(item,path,mock=False):
 need(item['status']=='ready','Incomplete real curve cannot be rendered');fig,ax=plt.subplots(figsize=(8,4.8));x=[p['x'] for p in item['points']];y=[p['mean'] for p in item['points']];std=[p['std_ddof1'] for p in item['points']]
 ax.errorbar(x,y,yerr=std,marker='o',capsize=4,lw=1.8,color='#176c9c');ax.set_xscale('log',base=2);ax.set_xticks(x,[str(v) for v in x]);ax.set_xlabel('BN statistics window B' if item['kind']=='Bscan' else 'Observed images N');ax.set_ylabel(item['metric']+' ['+item['unit']+']');ax.grid(alpha=.2);ax.set_title(item['config']+' | '+item['line']+' | '+item['domain']+' | '+item['head'])
 footer(fig,'Source SHA '+str(item['source_sha256'])[:16]+' | '+str(item.get('fixed_protocol',{}))+' | mean +/- sample SD, 5 ODR observation streams; one source training\nCurrent H/mixed scores use H-accepted subset; paper Table4 onward states all samples: denominators differ.',mock);fig.tight_layout(rect=(0,.14,1,1));fig.savefig(path);plt.close(fig)
def distribution(item,path,mock=False):
 fig,axes=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.35,1]});ax=axes[0]
 if item['n_common']:
  edges=np.asarray(item['bin_edges']);ax.bar(edges[:-1],item['bin_counts'],width=np.diff(edges),align='edge',color='#176c9c');ax.axvline(0,color='#a94335',lw=1);ax.set_xlabel('ODR - source final score [dimensionless]');ax.set_ylabel('Common-valid sample count')
 else:ax.text(.5,.5,'No common-valid pairs\nDistribution unavailable',ha='center',va='center',transform=ax.transAxes)
 names=['both_accepted','accepted_to_rejected','rejected_to_accepted','both_rejected'];axes[1].bar(range(4),[item['states'][n] for n in names],color=['#176c9c','#b7523e','#5a9370','#888888']);axes[1].set_xticks(range(4),['both valid','valid to reject','reject to valid','both rejected'],rotation=25,ha='right');axes[1].set_ylabel('All stable sample IDs');axes[1].set_title('Failure transitions: n='+str(item['n_total']));ax.set_title('Same-ID common-valid n='+str(item['n_common']));fig.suptitle(item['config']+' | '+item['line']+' | '+item['domain']+' | fixed seed 2021')
 footer(fig,'Source SHA '+item['source_sha256'][:16]+' | N='+str(item['N'])+' B='+str(item['B'])+' | excluded IDs never assigned zero/Inf\nCurrent histogram: same-ID common-valid intersection. Paper Table4 onward all-sample metric is a different denominator.',mock);fig.tight_layout(rect=(0,.14,1,.94));fig.savefig(path);plt.close(fig)
def parameters(item,path,mock=False):
 rows=item['rows'];labels=[r['config'].replace('table4_','').replace('_ehs','') for r in rows];fig,axes=plt.subplots(1,2,figsize=(11,4.8));x=np.arange(len(rows));bottom=np.zeros(len(rows));parts=[('efficientnet_encoder','EfficientNet'),('bifpn','BiFPN'),('heatmap','H head'),('efficientpose','E head'),('segmentation','S head')]
 for key,label in parts:
  y=np.array([int(r[key])/1e6 for r in rows]);axes[0].bar(x,y,bottom=bottom,label=label);bottom+=y
 axes[0].set_ylabel('Whole SPNv2 parameters [million]');axes[0].legend(fontsize=8);axes[0].set_title('Actual CPU model construction')
 y=[100*int(r['odr_updated_backbone_norm_affine'])/int(r['efficientdet_backbone']) for r in rows];axes[1].bar(x,y,color='#176c9c');axes[1].set_ylabel('ODR tuned / EfficientDet backbone [%]');axes[1].set_title('Backbone BN/GN affine only; head updates = 0')
 for ax in axes:ax.set_xticks(x,labels,rotation=22,ha='right');ax.grid(axis='y',alpha=.18)
 footer(fig,'Table6 PARTIAL: static parameter counts only. GPU latency, throughput and peak memory remain pending.',mock);fig.tight_layout(rect=(0,.075,1,1));fig.savefig(path);plt.close(fig)


def benchmark(item,path,mock=False):
 need(item['status']=='accepted' or mock,'Unaccepted GPU benchmark cannot be plotted');data=item['measurements'];labels=[r['name'].replace('table4_','').replace('_ehs','') for r in data];x=np.arange(len(data));fig,axes=plt.subplots(1,3,figsize=(13,4.8))
 axes[0].bar(x,[r['statistics']['mean_ms'] for r in data],yerr=[r['statistics']['population_std_ms'] for r in data],capsize=4,color='#176c9c');axes[0].set_ylabel('GPU forward latency [ms]');axes[0].set_title('30 samples: mean +/- population SD')
 axes[1].bar(x,[r['statistics']['images_per_second'] for r in data],color='#5a9370');axes[1].set_ylabel('Images / s (1000 / mean_ms)');axes[1].set_title('Physical batch = 1')
 axes[2].bar(x-.15,[r['memory_bytes']['peak_allocated']/2**20 for r in data],width=.3,label='peak allocated');axes[2].bar(x+.15,[r['memory_bytes']['peak_reserved']/2**20 for r in data],width=.3,label='peak reserved');axes[2].set_ylabel('GPU memory [MiB]');axes[2].legend(fontsize=8);axes[2].set_title('Model/input resident + forward peaks')
 for ax in axes:ax.set_xticks(x,labels,rotation=25,ha='right');ax.grid(axis='y',alpha=.15)
 fig.suptitle(str(item.get('device',{}).get('name','MOCK DEVICE'))+' | float32 | input 1x3x512x768 | all EHS heads')
 footer(fig,'Pure forward CUDA-event timing. Excludes loading, preprocessing, H2D and PnP; not end-to-end speed or ODR adaptation time.',mock);fig.tight_layout(rect=(0,.075,1,.94));fig.savefig(path);plt.close(fig)
