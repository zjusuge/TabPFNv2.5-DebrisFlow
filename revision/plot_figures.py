"""Render saved results through the original figure code with portable paths."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RES=ROOT/'archived'
EXP=ROOT/'results'
SRC=ROOT/'plotting'
OUT=ROOT/'rerun_figures'
CURRENT=''
MANIFEST=[]
from pathlib import Path

import argparse,ast,re,json,gc

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.text import Text

from matplotlib.patches import Patch,Rectangle,FancyBboxPatch,Polygon

from matplotlib.lines import Line2D

from scipy import stats

from scipy.stats import gaussian_kde

from PIL import Image

MC={'TabPFN':'#D55E00','RF':'#0072B2','XGBoost':'#332288','GBR':'#009E73','AdaBoost':'#CC79A7','KNN':'#E69F00','SVR':'#56B4E9','Ridge':'#999999','Lasso':'#882255','ElasticNet':'#44AA99','MLP':'#BBBBBB'}

SC={'***':'#D55E00','**':'#E69F00','*':'#DDCC77','n.s.':'#CCCCCC'}

BASELINES=['Ridge','Lasso','ElasticNet','SVR','KNN','RF','GBR','XGBoost','AdaBoost','MLP']

def style(size=12):
    plt.style.use('default');plt.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial','DejaVu Sans'],'font.size':size,'axes.labelsize':size+1,'axes.titlesize':size+1,'xtick.labelsize':size,'ytick.labelsize':size,'legend.fontsize':size-1,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,'mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def PL(ax,label):ax.text(-.10,1.04,label,transform=ax.transAxes,fontsize=15,fontweight='bold',ha='left',va='bottom')

def ensure_arial(axes):
    for ax in axes:
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontfamily('Arial')

def save(fig,name=None,legacy=False):
    name=name or CURRENT
    if name in ['Fig05','Fig07']:
        ax=fig.axes[3]
        for text in list(ax.texts):
            if 'Higher TabPFN' in text.get_text():
                ax.set_title(text.get_text(),fontsize=12,pad=12);text.remove()
        ax.legend(loc='lower right',fontsize=11,framealpha=.95)
    if name=='Fig08':
        fig.axes[0].legend([Line2D([0],[0],marker='o',color='#777',ls='',ms=7),Line2D([0],[0],marker='s',color='#777',ls='',ms=7)],['Set A: with deposits','Set B: without deposits'],loc='lower center',bbox_to_anchor=(.5,-.38),ncol=2,fontsize=10.5)
        for text in list(fig.axes[2].texts):
            if text.get_text()=='Set A > Set B':text.remove()
    if name=='Fig06':
        for ax in fig.axes[:2]:ax.legend(loc='upper left',fontsize=11,framealpha=.95)
        for ax in fig.axes[2:]:
            legend=ax.get_legend()
            label=legend.get_texts()[0].get_text()
            legend.remove();ax.set_title(label,fontsize=11,pad=10)
    for text in fig.findobj(Text):text.set_fontfamily(['Arial','DejaVu Sans'])
    if legacy:
        factor=7.2/fig.get_figwidth()
        for text in fig.findobj(Text):
            minimum=(8.8 if '$' in text.get_text() else 6.4)/factor
            text.set_fontsize(max(minimum,text.get_fontsize())*factor)
        fig.set_size_inches(7.2,fig.get_figheight()*factor)
        fig.tight_layout(pad=1.5,h_pad=2.1,w_pad=2.0)
    fig.savefig(OUT/f'{name}.jpg',dpi=900,format='jpg',bbox_inches='tight',pad_inches=.08,pil_kwargs={'quality':97,'subsampling':0})
    # Internal vector copy enables actual font-size QA; the delivery folder is JPG only.
    with Image.open(OUT/f'{name}.jpg') as im:
        assert im.info.get('dpi')==(900,900)
        MANIFEST.append({'figure':name,'pixels':list(im.size),'dpi':list(im.info['dpi'])})
    plt.close(fig);gc.collect();print(name,'900 dpi JPG saved',flush=True)

def SAVE_FIG(fig,unused):save(fig,legacy=True)

def learning_final():
    style(8);models=['TabPFN','RF','XGBoost','SVR','Ridge','GBR','MLP'];markers=['o','s','^','D','v','P','X'];fig=plt.figure(figsize=(7.2,7.3));gs=fig.add_gridspec(3,2,height_ratios=[2.7,1.2,2.8],hspace=.70,wspace=.34)
    for j,frame in enumerate([lc_sum_A,lc_sum_B]):
        for row,selection in [(0,models[:-1]),(1,['MLP'])]:
            ax=fig.add_subplot(gs[row,j])
            for model in selection:
                s=frame[frame.Model==model].sort_values('n_train');ax.plot(s.n_train,s.R2_mean,color=MC[model],marker=markers[models.index(model)],ms=3,lw=1.4 if model=='TabPFN' else 1,label=model);ax.fill_between(s.n_train,s.R2_mean-s.R2_std,s.R2_mean+s.R2_std,color=MC[model],alpha=.09)
            ax.set_ylabel('R², mean ± SD');ax.set_xlabel('Training catchments');ax.axhline(0,c='#555',lw=.6)
            if row==0:ax.set_title(f'({"a" if j==0 else "b"}) Set {"A" if j==0 else "B"}: six models',loc='left',fontweight='bold',fontsize=9)
            else:ax.set_title('MLP: separate full-range axis',loc='left',fontsize=8)
            if row==0 and j==1:ax.legend(fontsize=6.3,ncol=2,loc='lower right')
    ax=fig.add_subplot(gs[2,0]);x=np.arange(7)
    for off,n,alpha in [(-.18,10,.4),(.18,48,.95)]:
        vals=lc_sum_A[lc_sum_A.n_train==n].set_index('Model').loc[models].R2_mean;ax.bar(x+off,vals,.33,color=[MC[m] for m in models],alpha=alpha,label=f'Training n = {n}',edgecolor='white')
    ax.set_xticks(x,models,rotation=30);ax.set_ylabel('Mean test-fold R²');ax.axhline(0,c='#555',lw=.7);ax.set_title('(c) Set A: 10 and 48 training catchments',loc='left',fontsize=8.5,fontweight='bold');ax.legend(fontsize=6.5,loc='lower left');ax.set_ylim(min(vals.min(),lc_sum_A.R2_mean.min())-.6,1.2)
    ax=fig.add_subplot(gs[2,1])
    for label,frame,mk,color in [('A',lc_sum_A,'o',MC['TabPFN']),('B',lc_sum_B,'s',MC['RF'])]:
        sizes=sorted(frame.n_train.unique());delta=[]
        for n in sizes:
            s=frame[frame.n_train==n].set_index('Model').R2_mean;delta.append(s['TabPFN']-s.drop('TabPFN').max())
        ax.plot(sizes,delta,c=color,marker=mk,ms=4,lw=1.5,label=f'Set {label}')
    ax.axhline(0,c='#555',lw=.8);ax.set(xlabel='Training catchments',ylabel='ΔR², TabPFN − highest baseline mean');ax.set_title('(d) Margin over the best baseline mean',loc='left',fontsize=8.5,fontweight='bold');ax.legend(fontsize=7,loc='lower right');save(fig,'Fig09')

def supplements():
    style(8);summ=json.loads((EXP/'results_summary.json').read_text());t=pd.DataFrame(summ['transform']);f=pd.DataFrame(summ['foundation_folds']);fig,axs=plt.subplots(1,2,figsize=(7.2,3.7),sharey=True)
    for ax,tag in zip(axs,['A','B']):
        order=['Ridge','Lasso','ElasticNet','MLP'];p=t[t.dataset==tag]
        for off,trans,alpha,hatch in [(-.18,'raw',.4,'//'),(.18,'log',.95,'')]:
            vals=p[p['transform']==trans].set_index('model').loc[order].R2;ax.bar(np.arange(4)+off,vals,.33,color=[MC[m] for m in order],alpha=alpha,hatch=hatch,label=trans.title(),edgecolor='white')
        ref=f[(f.dataset==tag)&(f.kind=='new_random_cv')].R2.mean();ax.axhline(ref,c=MC['TabPFN'],ls='--',lw=1.3,label='TabPFN, 8 members');ax.set_xticks(range(4),order,rotation=25);ax.set_title(f'Set {tag}',loc='left',fontweight='bold');ax.set_ylabel('Mean outer-fold R²');ax.set_ylim(0,1.02)
    axs[1].legend(fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3);fig.tight_layout();save(fig,'FigS1')
    g=pd.DataFrame(summ['group_pooled']);fig,axs=plt.subplots(1,2,figsize=(7.2,3.6))
    for ax,metric in zip(axs,['R2','RMSE']):
        for off,m in [(-.18,'TabPFN'),(.18,'RF')]:
            v=g[g.model==m].set_index('dataset').loc[['A','B','Korea'],metric];bars=ax.bar(np.arange(3)+off,v,.33,color=MC[m],alpha=.85,label=m)
            for b,value in zip(bars,v):ax.text(b.get_x()+b.get_width()/2,value+.01,f'{value:.3f}',ha='center',fontsize=7)
        ax.set_xticks(range(3),['China A','China B','Korea']);ax.set_ylabel('Pooled R²' if metric=='R2' else 'Pooled RMSE in log₁₀ units');ax.margins(y=.18)
    axs[0].legend(loc='lower left',fontsize=7);fig.tight_layout();save(fig,'FigS2')
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.4))
    for ax,tag in zip(axs,['A','Korea']):
        d=pd.concat([pd.read_csv(p) for p in EXP.glob(f'TabPFN_{tag}_*_split_conformal.csv')]).sort_values('y_true');x=np.arange(len(d));inside=(d.y_true>=d.lo)&(d.y_true<=d.hi)
        ax.vlines(x,d.lo,d.hi,color=MC['RF'],alpha=.45,lw=1.1);ax.scatter(x,d.y_true,c='#333',s=13,label='Observed',zorder=3);ax.scatter(x,d.y_pred,c=MC['TabPFN'],s=12,label='Predicted',zorder=3);ax.scatter(x[~inside],d.y_true[~inside],s=27,facecolors='none',edgecolors='#A23B3B',lw=.9,label='Outside interval',zorder=4);ax.set_title(f'{"Longmen Shan A" if tag=="A" else "Korea"}: {inside.sum()}/{len(d)} covered at 90% nominal',loc='left',fontsize=9,fontweight='bold');ax.set(xlabel='Test observations ordered by observed volume',ylabel='Log₁₀ volume')
    axs[0].legend(fontsize=7,ncol=3,loc='upper left');fig.tight_layout(pad=1.4,h_pad=2.2);save(fig,'FigS3')
    r=pd.read_csv(EXP/'SHAP_region_mean_absolute.csv',index_col=0);data=pd.read_csv(EXP/'longmenshan_with_source_region.csv');fig,ax=plt.subplots(figsize=(7.2,3.8));im=ax.imshow(r.to_numpy(),cmap='Reds',aspect='auto',vmin=0);ax.set_xticks(range(6),['A','H','L','D','J','Vlandslide']);ax.set_yticks(range(6),[f'{name}, n = {(data.Region==name).sum()}' for name in r.index]);ax.grid(False)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{r.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if r.iloc[i,j]>.6*r.to_numpy().max() else '#222')
    fig.colorbar(im,ax=ax,label='Mean absolute SHAP in log₁₀ units');fig.tight_layout();save(fig,'FigS4')
def cell(name):
    exec(compile((SRC/name).read_text(encoding='utf-8'),str(SRC/name),'exec'),globals())

def original_figures():
    global CURRENT,comp_AB,lc_sum_A,lc_sum_B,MC
    style()
    cell('original_cell3_adapted.py')
    audit=json.loads((EXP/'paired_statistics_audit.json').read_text())
    for tag,name in [('FSA','Fig05'),('FSB','Fig07')]:
        records=[]
        for item in audit:
            if item['dataset']==tag and item['mode']=='bo':
                p=item['p_holm']
                sig='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
                records.append(dict(Model=item['model'],mean_dR2=np.mean(item['repeat_differences']),effect_r=item['rank_biserial'],sig=sig))
        CURRENT=name
        plot_model_comparison(pd.read_csv(RES/(tag+'_per_fold_R2.csv')),pd.DataFrame(records),'Set A' if tag=='FSA' else 'Set B',name)
    CURRENT='Fig06';style();cell('original_cell6_adapted.py')
    comp_AB=pd.read_csv(RES/'comparison_A_vs_B.csv')
    lc_sum_A=pd.read_csv(RES/'FSA_learning_curve_summary.csv')
    lc_sum_B=pd.read_csv(RES/'FSB_learning_curve_summary.csv')
    CURRENT='Fig08';style();cell('original_cell8_adapted.py')
    CURRENT='Fig09';learning_final()
    cell('original_UQ_adapted.py')
    primary_palette=MC.copy()
    cell('original_Korea_adapted.py')
    MC=primary_palette

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part',choices=['main','supplements','all'],default='all')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'rerun_figures')
    args=ap.parse_args();OUT=args.output_dir.expanduser().resolve()
    if OUT==ROOT/'figures' or OUT.is_relative_to(ROOT/'figures'):
        ap.error('Choose an output directory outside the delivered figures directory.')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.part in ['main','all']:original_figures()
    if args.part in ['supplements','all']:supplements()
    (OUT/'export_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
