"""Adapt the supplied plotting cells; render revised figures as 900 dpi JPEG.

Figures 5–9 retain the supplied code's multi-panel composition and palette.
Figures 3–4 inherit the original schematic structure. Figures 10–11 inherit
the visual style while using corrected uncertainty and benchmark definitions.
All numerical panels read saved results; no display-value overrides are used.
"""
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

HERE=Path(__file__).resolve().parent;WORK=HERE.parent;ROOT=WORK
BASE=WORK/'rerun_current_inputs';RES=WORK/'archived';EXP=WORK/'results'
OUT=WORK/'rerun_current_figures';QA=OUT/'qa';SRC=OUT/'adapted_sources'
for d in [OUT,QA,SRC]:d.mkdir(exist_ok=True,parents=True)
MC={'TabPFN':'#D55E00','RF':'#0072B2','XGBoost':'#332288','GBR':'#009E73','AdaBoost':'#CC79A7','KNN':'#E69F00','SVR':'#56B4E9','Ridge':'#999999','Lasso':'#882255','ElasticNet':'#44AA99','MLP':'#BBBBBB'}
SC={'***':'#D55E00','**':'#E69F00','*':'#DDCC77','n.s.':'#CCCCCC'}
BASELINES=['Ridge','Lasso','ElasticNet','SVR','KNN','RF','GBR','XGBoost','AdaBoost','MLP']
CURRENT='';MANIFEST=[]
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
    fig.savefig(QA/f'{name}_preview.png',dpi=300,bbox_inches='tight',pad_inches=.08)
    # Internal vector copy enables actual font-size QA; the delivery folder is JPG only.
    fig.savefig(QA/f'{name}_font_audit.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(QA/f'{name}_font_audit.svg',bbox_inches='tight',pad_inches=.08)
    with Image.open(OUT/f'{name}.jpg') as im:
        assert im.info.get('dpi')==(900,900)
        MANIFEST.append({'figure':name,'pixels':list(im.size),'dpi':list(im.info['dpi'])})
    plt.close(fig);gc.collect();print(name,'900 dpi JPG saved',flush=True)
def SAVE_FIG(fig,unused):save(fig,legacy=True)

def adapted(cell,transform=None):
    src=(HERE/f'绘图可视化_cell{cell}.py').read_text(encoding='utf-8')
    if transform:src=transform(src)
    src=src.replace('plt.show()','plt.close("all")')
    (SRC/f'original_cell{cell}_adapted.py').write_text(src,encoding='utf-8')
    exec(compile(src,f'adapted_cell{cell}','exec'),globals())

def comparison_figures():
    style();r2_A=pd.read_csv(RES/'FSA_per_fold_R2.csv');r2_B=pd.read_csv(RES/'FSB_per_fold_R2.csv')
    audit=json.loads((EXP/'paired_statistics_audit.json').read_text())
    def adapt(s):
        start=s.index('    ax.text(panel_a_info_pos[0]');end=s.index('    ax.text(0.02, 0.05',start)
        s=s[:start]+'''    ax.set_title(f"{set_label}: TabPFN mean R² = {tabpfn_mean:.4f}", fontsize=13, pad=12)\n\n'''+s[end:]
        start=s.index('    leg_handles =');end=s.index('    PL(ax, "(c)")',start)
        s=s[:start]+'''    ax.set_title("Ten paired repeat means; Holm correction", fontsize=12, pad=12)\n    ax.set_xlim(0, 1.13)\n'''+s[end:]
        s=s.replace('x_txt = min(row["effect_r"] + 0.03, 0.92)','x_txt = row["effect_r"] + 0.025')
        s=s.replace('    PL(ax, "(b)")','    ax.set_xlim(min(0, wdf["mean_dR2"].min()) - 0.01, wdf["mean_dR2"].max() * 1.27)\n    PL(ax, "(b)")')
        s=s.replace('TabPFN superior in','Higher TabPFN R² in')
        s=s.replace('fontsize=11.5, loc=panel_d_legend_loc','fontsize=11.5, loc="upper left"')
        return s
    adapted(3,adapt)
    for tag,frame,name in [('FSA',r2_A,'Fig05'),('FSB',r2_B,'Fig07')]:
        records=[]
        for item in audit:
            if item['dataset']==tag and item['mode']=='bo':
                p=item['p_holm'];sig='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
                records.append({'Model':item['model'],'mean_dR2':np.mean(item['repeat_differences']),'effect_r':item['rank_biserial'],'sig':sig})
        global CURRENT;CURRENT=name
        plot_model_comparison(frame,pd.DataFrame(records),'Set A' if tag=='FSA' else 'Set B',name)
    CURRENT='Fig06';style();adapted(6,lambda s:s.replace('log$_{10}$($V_0$)','log₁₀ volume').replace('Std =','SD ='))

def ablation_learning():
    global CURRENT,comp_AB,lc_sum_A,lc_sum_B
    comp_AB=pd.read_csv(RES/'comparison_A_vs_B.csv');lc_sum_A=pd.read_csv(RES/'FSA_learning_curve_summary.csv');lc_sum_B=pd.read_csv(RES/'FSB_learning_curve_summary.csv')
    CURRENT='Fig08';style()
    def ab(s):
        s=s.replace('Mean R² (BO-optimised)','Mean test-fold R²').replace('R² — Set','R², set')
        s=s.replace('PL(ax, "(a)")','ax.set_xlim(comp["R2_B"].min()-.06,1.03)\nPL(ax, "(a)")')
        s=s.replace('PL(ax, "(b)")','ax.set_xlim(0,comp_b["Delta_AB"].max()*1.24)\nPL(ax, "(b)")')
        s=s.replace('PL(ax, "(d)")','ax.set_xlim(0,comp_d["R2_rel_drop"].max()*1.23)\nPL(ax, "(d)")')
        s=s.replace('r"Relative R² decrease after removing $V_{land}$ (%)"','"Relative R² decrease without deposits (%)"')
        s=s.replace('xytext=(ox, oy), textcoords="offset points",', '''xytext={"TabPFN":(.87,.91),"RF":(.95,.75),"GBR":(.60,.88),"XGBoost":(.84,.45),"AdaBoost":(.42,.78),"KNN":(.37,.65),"SVR":(.72,.35),"Ridge":(.23,.34),"Lasso":(.24,.20),"ElasticNet":(.59,.23),"MLP":(.60,.09)}[m], textcoords="data",''')
        s=s.replace('ax.set_xlim(r2_lo, r2_hi)','ax.set_xlim(r2_lo, 1.10)')
        return s
    adapted(8,ab)
    CURRENT='Fig09';style()
    def learning(s):
        # Preserve four panels while retaining every true negative mean and SD band.
        s=s.replace('mu_c = np.clip(mu, CLIP, None)','mu_c = mu').replace('lo = np.clip(mu - sd, CLIP, None)','lo = mu - sd')
        s=s.replace('ax.set_ylim(-0.55, 1.05)','ax.set_ylim(min((lc_sum["R2_mean"]-lc_sum["R2_std"]).min()-.10,-.1), max(1.05,(lc_sum["R2_mean"]+lc_sum["R2_std"]).max()+.08))')
        s=s.replace('"R² < −0.5 clipped for readability"','"All means and SD bands retained"')
        s=s.replace('r"Training sample size ($n_{train}$)"','"Training catchments"')
        s=s.replace('fr"$n_{{train}}$ = {n_small}"','f"Training n = {n_small}"').replace('fr"$n_{{train}}$ = {n_large}"','f"Training n = {n_large}"')
        s=s.replace('loc="lower left")','loc="upper left")')
        # Endpoint labels and a full-range axis retain the original design.
        s=s.replace('ax.set_ylim(y_lo, y_hi)','ax.set_ylim(y_lo-.2, y_hi+.65)')
        s=s.replace('PL(ax, "(d)")','ax.set_title("Descriptive margin over the highest baseline mean", fontsize=11, pad=14)\nPL(ax, "(d)")')
        return s
    # Preserve the original four evidence roles and separate the extreme MLP range.
    (SRC/'original_cell9_adapted.py').write_text(learning((HERE/'绘图可视化_cell9.py').read_text(encoding='utf-8')),encoding='utf-8')
    learning_final()

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

def box(ax,x,y,w,h,text='',fc='#E7F0E0',ec='#4E704E',fs=8,bold=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.012',facecolor=fc,edgecolor=ec,lw=.8);ax.add_patch(p)
    if text:ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal')
def arrow(ax,a,b,color='#42656D',lw=1.2):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':color,'lw':lw,'mutation_scale':12})
def header(ax,y,label,color):box(ax,.015,y,.97,.043,label,fc=color,ec=color,fs=10,bold=True)
def grid(ax,x,y,w,h,rows=4,cols=5,query=False):
    colors=['#AEC9A2','#DABAA9','#B0CED1','#D8CBAD','#9DBD9D']
    for r in range(rows):
        for c in range(cols):
            col='#F6DDB0' if query else colors[c%5]
            ax.add_patch(Rectangle((x+c*w/cols,y+r*h/rows),w/cols,h/rows,fc=col,ec='white',lw=1))
    ax.add_patch(Rectangle((x,y),w,h,fc='none',ec='#58716A',lw=.75))
def schematics():
    style(8);fig,ax=plt.subplots(figsize=(7.2,8.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    header(ax,.944,'1  Data and paired predictor design','#BFD6E5')
    box(ax,.03,.823,.29,.100,'Longmen Shan\n60 catchments\nMaximum recorded event volume',fc='#EFF5F8',ec='#6C9EAF',fs=8)
    box(ax,.355,.823,.29,.100,'Log₁₀ response\nMatched observations and splits\nPositive predictors for log sensitivity',fc='#EFF5F8',ec='#6C9EAF',fs=8)
    box(ax,.68,.823,.29,.100,'Set A: A, H, L, D, J, Vlandslide\nSet B: A, H, L, D, J\nPaired source-information test',fc='#EFF5F8',ec='#6C9EAF',fs=8)
    arrow(ax,(.32,.873),(.35,.873));arrow(ax,(.645,.873),(.674,.873));arrow(ax,(.5,.815),(.5,.790))
    header(ax,.741,'2  Complementary modelling approaches','#CADBB4')
    box(ax,.03,.624,.40,.100,'TabPFN-2.5\nPretrained weights + labelled context\nIn-context regression',fc='#ECF2E3',fs=9)
    box(ax,.47,.624,.50,.100,'Ridge · Lasso · Elastic Net · SVR · KNN\nRF · GBR · XGBoost · AdaBoost · MLP\nTraining-only preprocessing and tuning',fc='#ECF2E3',fs=8.5)
    arrow(ax,(.5,.616),(.5,.590))
    header(ax,.542,'3  Matched primary evaluation','#B5DEDD')
    box(ax,.03,.420,.44,.103,fc='#EFF8F7',ec='#508C8B')
    ax.text(.25,.491,'Outer: 10 repetitions × 5 folds\nInner: 3 folds; 30 tuning trials',ha='center',va='center',fontsize=8.5)
    for i,label in enumerate(['Train','Train','Train','Train','Test']):box(ax,.065+i*.074,.433,.067,.027,label,fc='#CEE1BE' if i<4 else '#F3D5A9',ec='#7E9B83',fs=7)
    box(ax,.53,.420,.44,.103,'R² · RMSE · MAE\nTen paired repeat means\nWilcoxon comparisons + Holm correction',fc='#EFF8F7',ec='#508C8B',fs=8.7)
    arrow(ax,(.5,.412),(.5,.389))
    header(ax,.340,'4  Source value and robustness','#C6D7E8')
    for x,txt in zip([.03,.355,.68],['Matched A/B ablation\nLearning curves','Raw/log predictors\n30/100 trials; R²/RMSE','Whole-region holdouts\nChina and Korea']):box(ax,x,.249,.29,.072,txt,fc='#F0F4F9',ec='#6C86AB',fs=8.7)
    arrow(ax,(.5,.240),(.5,.215))
    header(ax,.167,'5  Uncertainty and environmental interpretation','#DDBED8')
    for x,txt in zip([.03,.355,.68],['Residual intervals\nSeparate split conformal','Global and local SHAP\nRegion-group attribution','Independent Korean benchmark\n63 events; three districts']):box(ax,x,.062,.29,.086,txt,fc='#F8F1F7',ec='#A07A99',fs=8.2)
    ax.text(.5,.020,'Primary benchmark and additional sensitivity protocols are reported separately.',ha='center',fontsize=7.5,color='#52626D')
    save(fig,'Fig03')

    fig,ax=plt.subplots(figsize=(7.2,7.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    header(ax,.945,'a  Pretraining across synthetic supervised tasks','#B9D7C3')
    ax.text(.5,.916,'Synthetic task prior',ha='center',fontsize=9,color='#315E45')
    for i,x in enumerate([.06,.395,.73]):
        grid(ax,x,.794,.21,.089);ax.text(x+.105,.778,f'Task {i+1}' if i<2 else 'Task m',ha='center',fontsize=8)
        arrow(ax,(.5,.903),(x+.105,.889));arrow(ax,(x+.105,.756),(.5,.721))
    box(ax,.18,.663,.64,.059,'Expected predictive log loss across tasks\nPretrained network weights',fc='#EDF5EC',ec='#527F61',fs=9)
    arrow(ax,(.5,.655),(.5,.627))
    header(ax,.576,'b  Inference on catchment tables','#C1D5E9')
    ax.text(.07,.548,'Labelled training context',fontsize=8.7,fontweight='bold');ax.text(.585,.548,'Query predictors',fontsize=8.7,fontweight='bold')
    grid(ax,.07,.456,.34,.074,rows=3,cols=5);grid(ax,.59,.473,.30,.027,rows=1,cols=5,query=True)
    ax.text(.448,.493,'+',fontsize=19,ha='center');ax.text(.07,.438,'Catchment descriptors and observed log volume',fontsize=7)
    ax.text(.59,.455,'Target withheld',fontsize=7,color='#9C601F')
    arrow(ax,(.25,.426),(.5,.395));arrow(ax,(.75,.426),(.5,.395))
    box(ax,.30,.350,.40,.045,'Feature and target embeddings',fc='#EDF1F4',ec='#8294A0',fs=9)
    arrow(ax,(.5,.344),(.5,.316))
    box(ax,.04,.126,.92,.190,fc='#F1F6F9',ec='#628090')
    ax.text(.5,.294,'Alternating attention',ha='center',fontsize=10,fontweight='bold',color='#344C62')
    box(ax,.07,.152,.40,.112,fc='#DDEEEE',ec='#628F90')
    ax.text(.27,.244,'Feature attention',ha='center',fontsize=9,fontweight='bold')
    for x,t in zip([.11,.22,.33],['Area','Relief','Source']):box(ax,x,.185,.084,.031,t,fc='#B7D5D1',ec='#719B95',fs=7)
    for a,b in [((.193,.201),(.217,.201)),((.303,.201),(.327,.201))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'<->','color':'#42656D','lw':1.1})
    ax.text(.27,.164,'Dependencies within a catchment',ha='center',fontsize=7.5)
    box(ax,.53,.152,.40,.112,fc='#E3EAF5',ec='#7083A1')
    ax.text(.73,.244,'Sample attention',ha='center',fontsize=9,fontweight='bold')
    ax.text(.61,.200,'Training\ncatchments',ha='center',va='center',fontsize=8);ax.text(.85,.200,'Query\ncatchment',ha='center',va='center',fontsize=8)
    arrow(ax,(.68,.201),(.783,.201));ax.text(.73,.164,'Relationships across catchments',ha='center',fontsize=7.5)
    arrow(ax,(.5,.12),(.5,.089))
    box(ax,.17,.030,.66,.059,'Predictive decoder\nPoint prediction and conditional predictive distribution',fc='#FAE8CF',ec='#C89C65',fs=8.8)
    save(fig,'Fig04')

def uq_korea():
    style(8)
    d=pd.read_csv(BASE/'UQ_conformal_results/UQ_XConf_samples_FSA.csv');q=d.groupby('alpha').agg(cov=('xc_cov','mean'),width=('xc_wid','mean')).reset_index().sort_values('alpha',ascending=False)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2));orange=MC['TabPFN'];blue=MC['RF']
    ax=axs[0,0];ax.plot(1-q.alpha,q['cov'],'o-',color=blue,lw=1.5,ms=4);ax.plot([.45,1],[.45,1],'--',color='#777',lw=.9);ax.set(xlabel='Nominal coverage',ylabel='Empirical coverage',xlim=(.45,1.01),ylim=(.45,1.01));ax.set_title('553/600 covered at 90% nominal',fontsize=8.5)
    ax=axs[0,1];ax.plot(1-q.alpha,q.width,'o-',color=orange,lw=1.5,ms=4);ax.set(xlabel='Nominal coverage',ylabel='Mean interval width in log₁₀ units')
    # Fixed first repetition illustrates issued intervals, without median aggregation.
    ninety=d[np.isclose(d.alpha,.10)];rep=ninety[ninety.fold<5].sort_values('y_obs');assert len(rep)==60
    ax=axs[1,0];x=np.arange(60);ax.vlines(x,rep.xc_lo,rep.xc_hi,color=blue,alpha=.5,lw=.8);ax.scatter(x,rep.y_obs,c='#272727',s=8,label='Observed',zorder=3);ax.scatter(x,rep.y_pred,c=orange,s=7,label='Predicted',zorder=3);ax.set(xlabel='Catchments ordered by observed volume',ylabel='Log₁₀ volume');ax.legend(fontsize=6.8,loc='upper left');ax.set_title('Issued 90% intervals: first repetition',fontsize=8.5)
    ax=axs[1,1];ax.scatter(ninety.xc_wid,abs(ninety.y_pred-ninety.y_obs),s=8,alpha=.3,color=blue,edgecolors='none');xx=np.linspace(ninety.xc_wid.min(),ninety.xc_wid.max(),100);ax.plot(xx,xx/2,'--',c=orange,lw=1.1,label='Half-width threshold');ax.set(xlabel='90% interval width in log₁₀ units',ylabel='Absolute test prediction error');ax.legend(fontsize=7,loc='upper left');ax.set_title('600 repeated test predictions',fontsize=8.5)
    for a,l in zip(axs.flat,'abcd'):a.text(-.15,1.06,f'({l})',transform=a.transAxes,fontweight='bold',fontsize=10)
    fig.tight_layout(pad=1.8,h_pad=2.5,w_pad=2.2);save(fig,'Fig10')
    k=pd.read_csv(BASE/'TabPFN_Korea_results/all_folds.csv');agg=k.groupby('model')[['R2','RMSE','MAE']].agg(['mean','std']);order=agg.sort_values(('R2','mean'),ascending=False).index.tolist();fig=plt.figure(figsize=(7.2,6.2));gs=fig.add_gridspec(2,2,wspace=.43,hspace=.47)
    sub=gs[0,0].subgridspec(2,1,height_ratios=[5,1.2],hspace=.55);a=fig.add_subplot(sub[0]);b=fig.add_subplot(sub[1]);main=[m for m in order if m!='MLP']
    for ax,models in [(a,main),(b,['MLP'])]:
        s=agg.loc[models];ys=np.arange(len(models));ax.barh(ys,s.R2['mean'],color=[MC[m] for m in models],alpha=.85,height=.7);ax.errorbar(s.R2['mean'],ys,xerr=s.R2['std'],fmt='none',ecolor='#444',capsize=2,lw=.65);ax.set_yticks(ys,models,fontsize=6.5);ax.invert_yaxis()
    a.set_title('(a) R² across 50 test folds',loc='left',fontsize=9,fontweight='bold');b.set_xlabel('Mean ± SD; MLP uses a wider scale',fontsize=7);b.tick_params(labelsize=6)
    sub2=gs[0,1].subgridspec(2,1,height_ratios=[5,1.2],hspace=.55);vals={m:k.loc[k.model==m,'R2'].to_numpy().reshape(10,5).mean(axis=1) for m in order}
    for row,models in [(0,[m for m in order if m not in ['TabPFN','MLP']]),(1,['MLP'])]:
        ax=fig.add_subplot(sub2[row]);delta=np.array([vals['TabPFN']-vals[m] for m in models]);ys=np.arange(len(models));ax.barh(ys,delta.mean(axis=1),color=[MC[m] for m in models],alpha=.8);ax.errorbar(delta.mean(axis=1),ys,xerr=delta.std(axis=1,ddof=1),fmt='none',ecolor='#444',lw=.7,capsize=2);ax.set_yticks(ys,models,fontsize=6.5);ax.invert_yaxis();ax.axvline(0,c='#777',lw=.7);ax.tick_params(labelsize=6)
        if row==0:ax.set_title('(b) TabPFN minus each baseline',loc='left',fontsize=9,fontweight='bold')
        else:ax.set_xlabel('Paired ΔR², mean ± SD; MLP wider scale',fontsize=6.6)
    ax=fig.add_subplot(gs[1,0]);metric=agg.loc[order,[(s,'mean') for s in ['R2','RMSE','MAE']]].to_numpy();norm=metric.copy()
    for j in range(3):
        lo,hi=metric[:,j].min(),metric[:,j].max();norm[:,j]=(metric[:,j]-lo)/(hi-lo)
        if j>0:norm[:,j]=1-norm[:,j]
    ax.imshow(norm,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(3),['R²','RMSE','MAE']);ax.set_yticks(range(11),order,fontsize=6.7);ax.grid(False)
    for i in range(11):
        for j in range(3):ax.text(j,i,f'{metric[i,j]:.4f}',ha='center',va='center',fontsize=6.4,color='white' if norm[i,j]>.6 else '#222')
    ax.set_title('(c) Mean metrics; darker is better',loc='left',fontsize=9,fontweight='bold')
    ax=fig.add_subplot(gs[1,1]);tab=k[k.model=='TabPFN'].R2.to_numpy();rf=k[k.model=='RF'].R2.to_numpy();win=tab>rf
    for mask,color,label in [(win,orange,'Higher TabPFN R²'),(~win,blue,'Higher or equal RF R²')]:ax.scatter(rf[mask],tab[mask],s=17,color=color,alpha=.75,label=label,edgecolor='white',lw=.3)
    lo=min(tab.min(),rf.min())-.03;hi=max(tab.max(),rf.max())+.03;ax.plot([lo,hi],[lo,hi],'--',c='#666',lw=.8);ax.set(xlabel='RF test-fold R²',ylabel='TabPFN test-fold R²',xlim=(lo,hi),ylim=(lo,hi));ax.legend(fontsize=6.2,loc='upper left');ax.set_title('(d) Matched folds: TabPFN and RF',loc='left',fontsize=9,fontweight='bold')
    save(fig,'Fig11')

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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['schematics','original','uq_korea','supplements','all'],default='all');args=parser.parse_args()
    if args.part in ['schematics','all']:schematics()
    if args.part in ['original','all']:comparison_figures();ablation_learning()
    if args.part in ['uq_korea','all']:uq_korea()
    if args.part in ['supplements','all']:supplements()
    (QA/f'export_manifest_{args.part}.json').write_text(json.dumps(MANIFEST,indent=2),encoding='utf-8')
