from pathlib import Path
import sys,re,json,gc,hashlib
R=Path(__file__).resolve().parent;HERE=R/'plotting_current';sys.path.insert(0,str(HERE))
import restore_original_style as b
import refine_original_figures as f
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import pandas as pd,numpy as np
from PIL import Image
OUT=R/'rerun_current_figures';QA=OUT/'qa';SRC=OUT/'adapted_sources'
for d in [OUT,QA,SRC]:d.mkdir(parents=True,exist_ok=True)
import shutil
inputs=OUT/'inputs'
for sub in ['TabPFN_BO_experiment_results','TabPFN_Korea_results','UQ_conformal_results','SHAP_Analysis_Figures_Optimized_900dpi_JPG/_cache']:(inputs/sub).mkdir(parents=True,exist_ok=True)
for fp in (R/'archived').glob('*.csv'):shutil.copy2(fp,inputs/'TabPFN_BO_experiment_results'/fp.name)
for a,z in [('Korea_all_folds.csv','TabPFN_Korea_results/all_folds.csv'),('UQ_XConf_samples_FSA.csv','UQ_conformal_results/UQ_XConf_samples_FSA.csv'),('UQ_XConf_summary_FSA.csv','UQ_conformal_results/UQ_XConf_summary_FSA.csv'),('all_results_tabpfn_exact_shap.npz','SHAP_Analysis_Figures_Optimized_900dpi_JPG/_cache/all_results_tabpfn_exact_shap.npz')]:shutil.copy2(R/'archived'/a,inputs/z)
shutil.copy2(R/'data/debris_flow_longmenshan.xlsx',inputs/'debris_flow_longmenshan.xlsx')
b.BASE=inputs;b.RES=R/'archived';b.EXP=R/'results';b.SRC=SRC;b.QA=QA;b.OUT=OUT;f.SRC=SRC;f.QA=QA
manifest=[]
def legend(ax,loc='best',ncol=1,labels=None):
 old=ax.get_legend()
 if old is None:return
 handles=old.legend_handles;txt=[x.get_text() for x in old.get_texts()]
 title=old.get_title().get_text()
 old.remove()
 ax.legend(handles,labels or txt,loc=loc,ncol=ncol,fontsize=8,frameon=False,handlelength=1.35,handletextpad=.45,borderpad=.25,labelspacing=.25,columnspacing=.7,title=title or None,title_fontsize=8)
def inside_legend(ax,loc='upper left',anchor=None,ncol=1):
 old=ax.get_legend()
 if old is None:return
 handles=old.legend_handles;labels=[t.get_text() for t in old.get_texts()];title=old.get_title().get_text()
 old.remove()
 kw={} if anchor is None else {'bbox_to_anchor':anchor}
 ax.legend(handles,labels,loc=loc,ncol=ncol,fontsize=7.7,title=title or None,title_fontsize=7.7,frameon=True,framealpha=.95,edgecolor='lightgray',handlelength=1.5,handletextpad=.45,labelspacing=.25,borderpad=.3,columnspacing=.7,**kw)

def note(ax,xy,txt,ha='left',va='bottom',color='gray',face='white',size=8):
 return ax.text(*xy,txt,transform=ax.transAxes,ha=ha,va=va,fontsize=size,bbox=dict(boxstyle='round,pad=.28',fc=face,ec=color,lw=.55,alpha=.95))

def save(fig,name=None,legacy=False):
 from matplotlib.collections import PathCollection
 from matplotlib.patches import Patch
 name=name or b.CURRENT
 name={'Fig02':'Fig2','Fig05':'Fig5','Fig06':'Fig6','Fig07':'Fig7','Fig08':'Fig8','Fig09':'Fig9'}.get(name,name)
 fac=min(1,7.2/fig.get_figwidth())
 if fac<1:
  fig.set_size_inches(7.2,fig.get_figheight()*fac)
  for ax in fig.axes:
   for art in ax.collections:
    if isinstance(art,PathCollection):art.set_sizes(art.get_sizes()*fac**2)
   for line in ax.lines:
    line.set_markersize(max(2,line.get_markersize()*fac));line.set_linewidth(max(.6,line.get_linewidth()*fac))
 for t in fig.findobj(Text):
  t.set_fontfamily(['Arial','DejaVu Sans']);t.set_fontsize(max(7.7,t.get_fontsize()*fac))
  if re.fullmatch(r'\([a-h]\)',t.get_text().strip()):t.set_bbox(None);t.set_fontsize(10);t.set_fontweight('bold')
 for ax in fig.axes:
  ax.tick_params(labelsize=7.7);ax.xaxis.label.set_fontsize(8.6);ax.yaxis.label.set_fontsize(8.6)
  # Names live in captions. Existing informational annotations remain inside.
  for loc in ['left','center','right']:ax.set_title('',loc=loc)
  old=ax.get_legend()
  if old:
   for t in old.get_texts():t.set_fontsize(7.7)
   old.get_title().set_fontsize(7.7)
 if name in ['Fig5','Fig7']:
  a,bb,c,d=fig.axes[:4]
  # All paired mean differences are positive, so the zero line is the spine.
  for ln in list(bb.lines):
   if len(ln.get_xdata())==2 and np.allclose(ln.get_xdata(),0):ln.remove()
  bb.set_xlim(0,bb.get_xlim()[1])
  tag='A' if name=='Fig5' else 'B';n=6 if tag=='A' else 5
  mean=pd.read_csv(b.RES/f'FS{tag}_per_fold_R2.csv').TabPFN.mean()
  note(a,(.025,.24) if tag=='A' else (.97,.055),f'Feature Set {tag} ({n} features)\nTabPFN mean R² = {mean:.4f}',ha='left' if tag=='A' else 'right',color='#D55E00',face='#FFF0E8')
  # Keep the original colour-coded significance key, in added internal space.
  records={x['model']:x for x in json.loads((b.EXP/'paired_statistics_audit.json').read_text()) if x['dataset']=='FS'+tag and x['mode']=='bo'}
  order=['MLP','KNN','Ridge','Lasso','ElasticNet','SVR','GBR','XGBoost','AdaBoost','RF'] if tag=='A' else ['Ridge','Lasso','ElasticNet','MLP','SVR','KNN','XGBoost','AdaBoost','GBR','RF']
  c.clear();ys=np.arange(10)[::-1]
  for y,m in zip(ys,order):
   val=records[m]['rank_biserial'];p=records[m]['p_holm'];sig='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
   c.barh(y,val,color=b.SC[sig],edgecolor='white',height=.65,alpha=.85)
   c.text(val+.025,y,sig,va='center',fontsize=8,fontweight='bold')
  c.set_yticks(ys,order,fontsize=7.7);c.set_xlabel('Effect size (rank-biserial r)',fontsize=8.6);c.tick_params(labelsize=7.7)
  c.set_ylim(c.get_ylim()[0]-2.5,c.get_ylim()[1]);c.set_xlim(0,1.13)
  c.legend([Patch(fc=b.SC[s]) for s in ['***','**','*','n.s.']],[r'$p_{\mathrm{Holm}}$ < 0.001',r'$p_{\mathrm{Holm}}$ < 0.01',r'$p_{\mathrm{Holm}}$ < 0.05',r'$p_{\mathrm{Holm}}$ ≥ 0.05'],loc='lower left',ncol=2,fontsize=7.7,framealpha=.95,edgecolor='lightgray',columnspacing=.8,handlelength=1.2,borderpad=.25)
  inside_legend(d,'upper left')
  for t in d.texts:
   if 'Higher TabPFN' in t.get_text():
    t.set_position((.98,.03));t.set_fontsize(7.5);t.set_bbox(dict(boxstyle='round,pad=.25',fc='white',ec='gray',lw=.5));t.set_ha('right')
 if name=='Fig6':
  for a in fig.axes[:2]:inside_legend(a,'upper left',(0,.72))
  for a in fig.axes[2:]:inside_legend(a,'upper right')
  for a in fig.axes:
   a.set_xlabel(r'Observed log$_{10}(V_0)$')
   if a in fig.axes[:2]:a.set_ylabel(r'Predicted log$_{10}(V_0)$')
 if name=='Fig2':
  axes=[a for a in fig.axes if a.get_label()!='<colorbar>']
  inside_legend(axes[1],'upper right');inside_legend(axes[2],'upper left',(.11,.86));inside_legend(axes[3],'upper left',(0,1.0))
  axes[1].set_ylim(axes[1].get_ylim()[0],5.7)
 if name=='Fig8':
  a=fig.axes[0];a.set_ylim(a.get_ylim()[0]+1.1,a.get_ylim()[1])
  a.legend([Line2D([],[],marker='o',color='#777',ls=''),Line2D([],[],marker='s',color='#777',ls='')],['Set A','Set B'],loc='lower left',fontsize=7.7,ncol=2,framealpha=.95)
 if name=='Fig9':
  # Retain separate MLP ranges: restoring the original clipping would hide data.
  a,mlpa,bb,mlpb,c,d=fig.axes
  a.legend(*bb.get_legend_handles_labels(),loc='lower right',ncol=2,fontsize=7.7)
  for ax in [a,bb]:inside_legend(ax,'lower right',ncol=2)
  for ax in [mlpa,mlpb]:ax.text(.02,.83,'MLP',transform=ax.transAxes,fontsize=7.7)
  inside_legend(c,'lower left');inside_legend(d,'upper left')
  for bar in c.patches:
   val=bar.get_height();c.text(bar.get_x()+bar.get_width()/2,val/2 if val < -1 else val+(.035 if val>=0 else -.035),f'{val:.4f}',rotation=90,ha='center',va='center' if val < -1 else 'bottom' if val>=0 else 'top',fontsize=7,color='#555555',fontweight='bold')
  c.set_ylim(c.get_ylim()[0]-.25,1.95)
  d.set_ylim(d.get_ylim()[0],.10);d.set_ylabel('ΔR² (TabPFN − best baseline)',fontsize=8.6)
  for ax in [a,bb,mlpa,mlpb,d]:ax.set_xlabel(r'Training sample size ($n_{train}$)',fontsize=8.6)
  d.lines[1].set_linestyle('--')
  a.text(.03,.92,'Set A',transform=a.transAxes,fontsize=8);bb.text(.03,.92,'Set B',transform=bb.transAxes,fontsize=8)
  # Match the original curve markers and line patterns.
  patterns=['-','--','--',':','--','-.',':']
  for ax in [a,bb,mlpa,mlpb]:
   for ln in ax.lines:
    if ln.get_label() in ['TabPFN','RF','XGBoost','SVR','Ridge','GBR','MLP']:ln.set_linestyle(patterns[['TabPFN','RF','XGBoost','SVR','Ridge','GBR','MLP'].index(ln.get_label())])
 if name=='Fig10':
  a,bb,c,d=fig.axes[:4]
  inside_legend(a,'upper left',(0,.82));inside_legend(bb,'lower right')
  note(a,(.98,.025),'First repetition: 56/60 covered\nMedian width: 0.713 log₁₀ units',ha='right',size=7.5)
  c.annotate('90%: median = 0.708',xy=(5,.708),xytext=(1.4,1.32),fontsize=7.5,bbox=dict(boxstyle='round,pad=.2',fc='white',ec='gray',lw=.5),arrowprops=dict(arrowstyle='-',lw=.6,color='gray'))
  d.set_ylim(-1.6,3.8);inside_legend(d,'lower right',ncol=3)
 if name=='Fig11':
  for ax in fig.axes:
   for im in ax.images:
    im.set_cmap('RdYlBu_r')
    for t in ax.texts:
     x,y=t.get_position()
     if int(x)==x and int(y)==y and 0<=y<im.get_array().shape[0] and 0<=x<im.get_array().shape[1]:
      rgba=im.cmap(im.norm(im.get_array()[int(y),int(x)]));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2];t.set_color('white' if lum<.55 else '#222222')
  for ax in fig.axes:
   if ax.get_legend():inside_legend(ax,'lower right')
  # Separate MLP axes keep their identity without subplot titles.
  for ax in fig.axes:
   if len(ax.get_yticklabels())==1 and ax.get_yticklabels()[0].get_text()=='MLP':pass
 if name=='Fig12':
  a,bb,c,d=fig._source_axes
  fig.set_size_inches(7.2,6.6)
  a.set_xlabel(r'Mean |SHAP value| (log$_{10}$ scale)');bb.set_xlabel(r'SHAP value (log$_{10}$ scale)')
  c.set_xlabel(r'$V_{\mathrm{landslide}}$ ($10^4$ m$^3$)');c.set_ylabel(r'SHAP value (log$_{10}$ scale)')
  for p,col in zip(a.patches[-6:],['#B71C1C','#C62828','#D32F2F','#E53935','#EF9A9A','#FFCDD2']):p.set_facecolor(col)
  for t in a.texts:
   if 'cumulative contribution' in t.get_text():t.set_text('Top-2 cumul.: 58%\nTop-4 cumul.: 92%');t.set_fontsize(7.7)
  for t in list(d.texts):
   if 'Upper' in t.get_text() or 'Lower' in t.get_text():t.remove()
   elif t.get_bbox_patch() is not None:t.remove()
  # Triangle definitions are in the caption; no overprinted matrix values.
  cb=fig._source_cbs[1];cb.set_label(r'$A$ (km$^2$)',fontsize=7.7);cb.ax.tick_params(labelsize=7)
  fig._source_cbs[0].set_label('Feature value\n(normalised)',fontsize=7.7)
  fig._source_cbs[2].set_label(r'Pearson $r$',fontsize=7.7)
 if name=='Fig13':
  a,bb,c,d=fig._source_axes;fig.set_size_inches(7.2,6.8)
  c.set_xlabel(r'Mean |SHAP value| (log$_{10}$ scale)')
  d.set_xlabel('Sample index (sorted by predicted volume)')
  d.set_xticklabels(['1','12','24','36','48','60'])
  fig._top_axis.set_xlabel(r'Predicted $V_0$ ($10^4$ m$^3$)',fontsize=7.7,labelpad=5)
  fig._source_cbs[0].set_label(r'SHAP value (log$_{10}$ scale)',fontsize=7.7)
  leg=c.get_legend()
  if leg:
   leg.set_title('Observed-volume group')
   for t in leg.get_texts():t.set_text(t.get_text().split(' ')[0]+' (n = 20)')
   inside_legend(c,'lower right')
 if name=='FigS1':
  inside_legend(fig.axes[1],'upper right');fig.axes[0].text(.03,.93,'Set A',transform=fig.axes[0].transAxes);fig.axes[1].text(.03,.93,'Set B',transform=fig.axes[1].transAxes)
 if name=='FigS2':fig.axes[0].set_ylim(0,1.14);inside_legend(fig.axes[0],'upper right',ncol=2)
 if name=='FigS3':inside_legend(fig.axes[0],'upper left',ncol=3)
 if name not in ['Fig12','Fig13','Fig9','Fig2']:
  fig.tight_layout(pad=1.2,h_pad=1.2,w_pad=1.2)
 elif name=='Fig9':fig.subplots_adjust(left=.10,right=.98,bottom=.08,top=.95,wspace=.34,hspace=.60)
 elif name=='Fig2':
  fig.set_size_inches(7.2,6.8)
  boxes=[[.10,.564,.35,.366],[.61,.564,.35,.366],[.10,.10,.35,.366],[.61,.10,.35,.366]]
  for ax,pos in zip(axes,boxes):ax.set_position(pos)
  axes[0].set_aspect('auto')
  cb=[a for a in fig.axes if a.get_label()=='<colorbar>'][0];cb.set_position([.462,.60,.012,.30])
 else:
  boxes=[[.12,.574,.32,.356],[.60,.574,.31,.356],[.12,.12,.32,.346],[.60,.12,.31,.346]]
  for ax,pos in zip(fig._source_axes,boxes):ax.set_position(pos)
  if name=='Fig12':
   fig._source_axes[3].set_aspect('auto')
   fig._source_cbs[0].ax.set_position([.924,.62,.012,.27]);fig._source_cbs[2].ax.set_position([.924,.14,.012,.30])
   # Lower-right inset colour key: high-volume observations occupy the upper half.
   pos=fig._source_axes[2].get_position();fig._source_cbs[1].ax.set_position([pos.x0+.74*pos.width,pos.y0+.06*pos.height,.010,.17])
  else:
   fig._top_axis.set_position(boxes[3]);fig._source_cbs[0].ax.set_position([.924,.14,.012,.29])
 # Fixed figure coordinates guarantee column and row alignment of panel letters.
 if name in ['Fig12','Fig13']:main_axes=fig._source_axes
 elif name=='Fig9':main_axes=[fig.axes[i] for i in [0,2,4,5]]
 else:main_axes=[a for a in fig.axes if a.get_label()!='<colorbar>']
 for ax in fig.axes:
  for t in list(ax.texts):
   if re.fullmatch(r'\([a-h]\)',t.get_text().strip()):t.remove()
 if len(main_axes)>=4:
  # Korea contains extra MLP axes; identify the four full panels by their size.
  if name=='Fig11':
   candidates=[a for a in main_axes if a.get_position().height>.13]
   main_axes=sorted(candidates,key=lambda a:a.get_position().x0)
   left=sorted(main_axes[:2],key=lambda a:-a.get_position().y1);right=sorted(main_axes[2:],key=lambda a:-a.get_position().y1)
   main_axes=[left[0],right[0],left[1],right[1]]
  for i,ax in enumerate(main_axes[:4]):
   pos=ax.get_position();fig.text(pos.x0-.028,pos.y1+.012 if name!='Fig13' or i<2 else pos.y1+.070,f'({chr(97+i)})',fontsize=10,fontweight='bold',ha='left',va='bottom')
 for t in fig.findobj(Text):t.set_fontfamily(['Arial','DejaVu Sans'])
 fig.canvas.draw()
 assert all(not ax.get_title(loc=loc) for ax in fig.axes for loc in ['left','center','right'])
 fig.savefig(QA/f'{name}.jpg',dpi=200,bbox_inches='tight',pad_inches=.06,pil_kwargs={'quality':95})
 fig.savefig(QA/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
 if '--preview' not in sys.argv:fig.savefig(OUT/f'{name}.jpg',dpi=900,format='jpg',bbox_inches='tight',pad_inches=.06,pil_kwargs={'quality':98,'subsampling':0})
 info={'figure':name,'subplot_titles':0,'panel_letter_boxes':0,'axes_positions':[list(a.get_position().bounds) for a in main_axes]}
 (QA/f'{name}_audit.json').write_text(json.dumps(info,indent=2))
 plt.close(fig);gc.collect();print(name,'ready',flush=True)

b.save=save;b.SAVE_FIG=lambda fig,unused:save(fig,legacy=True)
def feature():
 b.style();b.CURRENT='Fig02';b.df_raw=pd.read_excel(b.BASE/'debris_flow_longmenshan.xlsx');b.df_raw['log_V0']=np.log10(b.df_raw['V0_1e4m3'])
 b.adapted(2)
def shap():
 s=(HERE/'shap_source.py').read_text(encoding='utf-8')
 start=s.index('DATA_DIR =');end=s.index('DPI =',start)
 s=s[:start]+f'DATA_DIR = Path({str(b.BASE)!r})\nFIG_DIR = Path({str(OUT)!r})\nCACHE_FILE = DATA_DIR / "SHAP_Analysis_Figures_Optimized_900dpi_JPG/_cache/all_results_tabpfn_exact_shap.npz"\n\n'+s[end:]
 start=s.index('def _save(');end=s.index('def _restore_all_spines',start)
 s=s[:start]+'''def _save(fig, stem):
    if 'Global' in stem:
        fig._source_axes=[ax_a,ax_b,ax_c,ax_d]
        fig._source_cbs=[cb_b,cbar_c,cb_d]
    else:
        fig._source_axes=[ax2a,ax2b,ax2c,ax2d]
        fig._source_cbs=[cb_hm];fig._top_axis=ax2d_top
    save(fig, 'Fig12' if 'Global' in stem else 'Fig13', legacy=True)

'''+s[end:]
 s=s.replace('bbox=box,','bbox=None,')
 s=s.replace("SORTED_NAMES = [DISPLAY_NAME_MAP[FEAT_LABELS[i]] for i in rank_idx]","SORTED_NAMES = [SYMUNIT[FEAT_LABELS[i]][0] for i in rank_idx]")
 s=s.replace("r'$D_f$'","r'$D$'").replace("r'$V_L$'","r'$V_{\\mathrm{landslide}}$'")
 s=s.replace('small_rep_idx = representative_index(group_chunks[0], y)','small_rep_idx = 54  # original published-figure example, verified against cached predictors')
 s=s.replace('large_rep_idx = representative_index(group_chunks[2], y)','large_rep_idx = 12  # original published-figure example, verified against cached predictors')
 start=s.index('def _true_waterfall(');end=s.index('# -----------------------------------------------------------------------------\n# 6.',start)
 s=s[:start]+(HERE/'shap_local_function.py').read_text(encoding='utf-8')+'\n'+s[end:]
 (SRC/'original_SHAP_adapted.py').write_text(s,encoding='utf-8');exec(compile(s,'SHAP_adapted','exec'),{'save':save})
part=sys.argv[1] if len(sys.argv)>1 else 'all'
if part in ['all','feature']:feature()
if part in ['all','compare']:b.comparison_figures()
if part in ['all','ablation']:b.ablation_learning()
if part in ['all','uq']:f.uq()
if part in ['all','korea']:f.korea()
if part in ['all','shap']:shap()
if part in ['all','supplements']:b.supplements()
