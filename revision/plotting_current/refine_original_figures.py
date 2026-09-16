"""Targeted edits to the author's original plotting cells and editable schematics.
Quantitative grid contract: preserve the original panel roles and palette; correct
statistics and use every saved observation. JPEG 900 dpi is the delivery format.
Internal vector exports are used only for font and layout audits.
"""
from pathlib import Path
import zipfile,re,json,difflib
import numpy as np
import pandas as pd
import restore_original_style as base

HERE=Path(__file__).resolve().parent;WORK=HERE.parent;ROOT=WORK
SRC=WORK/'rerun_current_figures/adapted_sources';QA=WORK/'rerun_current_figures/qa'

def keep_diff(name,old,new):
    (SRC/name).write_text(new,encoding='utf-8')
    (SRC/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original',tofile='revised')),encoding='utf-8')

def edit_schematics():
    from lxml import etree
    src=ROOT/'A投稿期刊/论文的图/PPT绘图/演示文稿1.pptx'
    out=SRC/'Figures_3_4_original_design_revised.pptx'
    changes={'Zero-shot inference':'In-context inference',
        'Cross-validated conformal calibration':'Residual-based prediction intervals',
        'Transformer Encoder Stack (':'Transformer Encoder Stack: ',
        'L/2 alternating pairs), L = 18~24 layers':'18 layers, each with feature and sample attention',
        'Per-cell embedding, feature group size = 3':'Feature-group embedding, group size = 3'}
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    count={k:0 for k in changes}
    with zipfile.ZipFile(src) as zin,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            raw=zin.read(item.filename)
            if re.fullmatch(r'ppt/slides/slide[1-4]\.xml',item.filename):
                tree=etree.fromstring(raw)
                for p in tree.findall('.//a:p',ns):
                    texts=p.findall('.//a:t',ns)
                    full=''.join(t.text or '' for t in texts)
                    if 'Transformer Encoder Stack' in full:
                        texts[0].text='Transformer Encoder Stack: 18 layers'
                        for t in texts[1:]:t.text=''
                        count['Transformer Encoder Stack (']+=1
                        continue
                    for t in texts:
                        if t.text in changes:
                            old=t.text;t.text=changes[old];count[old]+=1
                raw=etree.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
            zout.writestr(item,raw)
    (QA/'schematic_edits.json').write_text(json.dumps(count,indent=2),encoding='utf-8')
    print('Original schematic layouts preserved; labels updated',flush=True)

def uq():
    old=(HERE/'不确定性计算及可视化_cell1.py').read_text(encoding='utf-8')
    s=old
    start=s.index('DATA_DIR =');end=s.index('FILE_SAMPLES =',start)
    s=s[:start]+f'DATA_DIR = Path({str(base.BASE)!r})\nUQ_DIR = DATA_DIR / "UQ_conformal_results"\n'+s[end:]
    start=s.index('df_agg =');end=s.index('n_unique =',start)
    s=s[:start]+'''# Fixed first repetition for the sample and fan-chart displays.
df_agg = df_s[df_s[COL_FOLD] < 5].copy()
df_agg['covered'] = ((df_agg[COL_YOBS] >= df_agg[COL_LO]) & (df_agg[COL_YOBS] <= df_agg[COL_HI])).astype(int)
df_agg['n_folds'] = 1
assert df_agg.groupby(COL_ALPHA).size().eq(60).all()
'''+s[end:]
    # Remove invalid independent-binomial intervals for repeated catchments.
    start=s.index('ax.axhline(res_ceiling');end=s.index('ax.set_xlabel("Nominal coverage"',start)
    s=s[:start]+'''ax.plot(coverages_b, empirical_b, 'o-', color=C_BLUE, ms=7, lw=2,
        markeredgecolor='white', label='Residual intervals', zorder=3)
ax.set_title('All 600 outer-test predictions per level', fontsize=14)
'''+s[end:]
    s=s.replace('alphas_c  = sorted(df_agg[COL_ALPHA].unique(), reverse=True)','alphas_c  = sorted(df_s[COL_ALPHA].unique(), reverse=True)')
    s=s.replace('w = df_agg.loc[np.abs(df_agg[COL_ALPHA] - a) < 1e-6, "xc_wid"].values','w = df_s.loc[np.abs(df_s[COL_ALPHA] - a) < 1e-6, "xc_wid"].values')
    start=s.index('    std_v =');end=s.index('    vp =',start)
    s=s[:start]+'    w_plot = w_data\n'+s[end:]
    s=s.replace('showfliers=False','showfliers=True')
    s=s.replace('label="Median prediction"','label="Prediction"')
    s=s.replace('f"Empirical coverage: {emp_cov_90:.1f}%\\n"','f"First repetition: {emp_cov_90:.1f}% covered\\n"')
    s=s.replace('f"#{sid}  "','f"No. {sid+1}  "')
    s=s.replace('ax.legend(handles=legend_handles, loc="lower right",','ax.legend(handles=legend_handles, loc="upper center",')
    s=s.replace('fontsize=12, ncol=2','fontsize=12, ncol=4')
    s=s.replace('bbox_to_anchor=(0.98, 0.02)','bbox_to_anchor=(0.50, -0.22)')
    s=s.replace('ax_i.text(0.02, 0.98, tag,','ax_i.text(0.00, 1.12, tag,')
    s=s.replace('ax.set_ylabel(YLABEL_VOLUME, fontweight="normal")', 'ax.set_ylabel("Debris-flow volume\\n" + r"$\\log_{10}(V_0 / 10^4\\ \\mathrm{m}^3)$", fontweight="normal")')
    s=s.replace('Prediction interval width (log$_{10}$ units)', 'Interval width (log$_{10}$ units)')
    s=s.replace('ax.set_ylabel("Representative catchment", fontweight="normal")','ax.set_ylabel("Selected catchment", fontweight="normal")')
    start=s.index('y_labs = []');end=s.index('NEST_COLORS',start)
    s=s[:start]+"y_labs = [f'No. {sid+1}' for sid in PICK_IDS]\n\n"+s[end:]
    s=s.replace('bbox_to_anchor=(0.00, 0.82)','bbox_to_anchor=(0.00, 1.00)')
    s=s.replace('ax.legend(loc="lower right", frameon=True, framealpha=0.95,','ax.legend(loc="upper left", frameon=True, framealpha=0.95,')
    start=s.index('ax.text(0.97, 0.05, info_a,');end=s.index('ax.set_xlabel("Sample index',start)
    s=s[:start]+"ax.set_title(f'First repetition; median width = {med_wid_90:.3f}', fontsize=14)\n\n"+s[end:]
    s=s.replace('bbox_to_anchor=(0.00, 1.00)','bbox_to_anchor=(.5, -.20), ncol=3')
    s=s.replace('ax.legend(loc="upper left", frameon=True, framealpha=0.95,','ax.legend(loc="upper center", frameon=True, framealpha=0.95,')
    # Panel b legend has no inherited anchor; place it below its axis.
    s=s.replace('fontsize=12, edgecolor="0.80")             # ★ 10 → 12','fontsize=12, edgecolor="0.80", bbox_to_anchor=(.5,-.20), ncol=2)')
    start=s.index('try:\n    idx90');end=s.index('# (d)',start)
    s=s[:start]+"ax.set_title(f'90% median width = {np.median(width_dists[labels_c.index(\"90%\")]):.3f}', fontsize=14)\n\n"+s[end:]
    s=s[:s.index('save_path =')]+"\nbase.save(fig,'Fig10',legacy=True)\n"
    s=s.replace('"mathtext.fontset":    "custom"','"mathtext.fontset":    "dejavusans"')
    s=s.replace('figsize=(15, 10.5)','figsize=(15, 12)')
    s=s.replace('label=r"$\\pm$5 pp tolerance band"','label="Reference band: ±5 pp"')
    keep_diff('original_UQ_adapted.py',old,s)
    exec(compile(s,'original_UQ_adapted.py','exec'),{'base':base})

def korea():
    old=(HERE/'数据试验_cell2.py').read_text(encoding='utf-8')
    s=old
    start=s.index('models_all =');end=s.index('#  COLOUR PALETTE',start)
    # Keep downstream variable names so original plotting panels remain reusable.
    data='''import pandas as pd
k=pd.read_csv(BASE/'TabPFN_Korea_results/all_folds.csv')
agg=k.groupby('model')[['R2','RMSE','MAE']].agg(['mean','std'])
models_all=agg.sort_values(('R2','mean'),ascending=False).index.tolist()
r2_mean=agg.loc[models_all,('R2','mean')].to_numpy()
r2_std=agg.loc[models_all,('R2','std')].to_numpy()
rmse_mean=agg.loc[models_all,('RMSE','mean')].to_numpy()
mae_mean=agg.loc[models_all,('MAE','mean')].to_numpy()
models_10=models_all
metric_data=np.column_stack([r2_mean,rmse_mean,mae_mean])
metric_labels=['R² ↑','RMSE ↓','MAE ↓']
metric_higher=[True,False,False]
repeat={m:k[k.model==m].R2.to_numpy().reshape(10,5).mean(axis=1) for m in models_all}
'''
    s=s[:start]+data+'\n# '+s[end:]
    start=s.index('colors_a =');end=s.index('#  (c)  MULTI-METRIC',start)
    panels='''# Retain vertical ranking bars; show MLP variability on its own full scale.
sub=ax_a.get_subplotspec().subgridspec(2,1,height_ratios=[3,1],hspace=.65)
ax_a.remove();ax_a=fig.add_subplot(sub[0]);ax_m=fig.add_subplot(sub[1])
main=[m for m in models_all if m!='MLP']
for ax,mods in [(ax_a,main),(ax_m,['MLP'])]:
    x=np.arange(len(mods));v=agg.loc[mods,('R2','mean')].to_numpy();sd=agg.loc[mods,('R2','std')].to_numpy()
    ax.bar(x,v,yerr=sd,capsize=3.5,color=[MC[m] for m in mods],edgecolor='white',width=.72,error_kw=dict(lw=1,ecolor='#616161'))
    ax.set_xticks(x,mods,rotation=38 if len(mods)>1 else 0,ha='right' if len(mods)>1 else 'center')
    ax.axhline(0,c='grey',lw=.6);ax.set_ylabel('R², mean ± SD')
    if len(mods)>1:ax.set_ylim(min(0,(v-sd).min()-.05),max(1.05,(v+sd).max()+.08))
    else:ax.set_ylim((v-sd).min()-.25,(v+sd).max()+.25);ax.set_title('MLP: separate full-range axis',fontsize=12)
ax_a.set_title('Korean benchmark: 50 test folds',fontsize=14)
# Matched-fold scatter replaces cross-dataset retention ratios.
tab=k[k.model=='TabPFN'].R2.to_numpy();rf=k[k.model=='RF'].R2.to_numpy();win=tab>rf
for mask,col,label in [(win,C['TabPFN'],'Higher TabPFN R²'),(~win,C['tree'],'Higher or equal RF R²')]:
    ax_b.scatter(rf[mask],tab[mask],s=38,c=col,alpha=.8,label=label,edgecolor='white',lw=.5)
lo=min(tab.min(),rf.min())-.03;hi=max(tab.max(),rf.max())+.03
ax_b.plot([lo,hi],[lo,hi],'--',c='#777',lw=1)
ax_b.set(xlabel='RF test-fold R²',ylabel='TabPFN test-fold R²',xlim=(lo,hi),ylim=(lo,hi))
ax_b.legend(fontsize=11,loc='lower right');ax_b.set_title('Matched Korean test folds',fontsize=14)
# '''
    s=s[:start]+panels+s[end:]
    s=s.replace('cmap="RdYlBu_r"','cmap="Blues"').replace('tc = "white" if bg > 0.72 or bg < 0.28 else "#222"','tc = "white" if bg > 0.60 else "#222"')
    s=s.replace('cbar.set_label("Normalized score", fontsize=13)','cbar.set_label("Within-metric normalized score", fontsize=13)')
    start=s.index('idx9 =');end=s.index('#  SUBPLOT LABELS',start)
    paneld='''sub=ax_d.get_subplotspec().subgridspec(2,1,height_ratios=[5,1],hspace=.65)
ax_d.remove();ax_d=fig.add_subplot(sub[0]);ax_dm=fig.add_subplot(sub[1])
for ax,mods in [(ax_d,[m for m in models_all if m not in ['TabPFN','MLP']]),(ax_dm,['MLP'])]:
    delta=np.array([repeat['TabPFN']-repeat[m] for m in mods]);ys=np.arange(len(mods))
    ax.barh(ys,delta.mean(axis=1),color=[MC[m] for m in mods],edgecolor='white',height=.62)
    ax.errorbar(delta.mean(axis=1),ys,xerr=delta.std(axis=1,ddof=1),fmt='none',ecolor='#555',capsize=3,lw=1)
    ax.set_yticks(ys,mods);ax.invert_yaxis();ax.axvline(0,c='#777',lw=.7)
    ax.set_xlabel('Paired ΔR², mean ± SD')
ax_d.set_title('TabPFN minus baseline: ten repeat means',fontsize=13)
ax_dm.set_title('MLP: separate full-range axis',fontsize=12)
# '''
    s=s[:start]+paneld+s[end:]
    start=s.index('LABEL_FS =')
    s=s[:start]+'''for ax,label in [(ax_a,'a'),(ax_b,'b'),(ax_c,'c'),(ax_d,'d')]:
    ax.text(-.13,1.12,f'({label})',transform=ax.transAxes,fontweight='bold',fontsize=17)
fig.subplots_adjust(hspace=.5,wspace=.4)
base.save(fig,'Fig11',legacy=True)
'''
    s=s.replace('"mathtext.fontset":   "custom"','"mathtext.fontset":   "dejavusans"')
    s=s.replace('figsize=(15, 12)','figsize=(15, 14)').replace('hspace=.65','hspace=1.15')
    keep_diff('original_Korea_adapted.py',old,s)
    exec(compile(s,'original_Korea_adapted.py','exec'),{'BASE':base.BASE,'base':base})

if __name__=='__main__':
    import sys
    part=sys.argv[1] if len(sys.argv)>1 else 'all'
    if part in ['all','schematics']:edit_schematics()
    if part in ['all','uq']:uq()
    if part in ['all','korea']:korea()
