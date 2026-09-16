"""
Figure 12 visualization — v2 with consistent font sizes
Fig. 12 — Cross-regional validation on the South Korean debris-flow dataset
Publication-quality 2×2 figure  (CNS-level standard)
──────────────────────────────────────────
v2 CHANGES vs v1:
  ★ ALL font sizes unified with Fig 10/11 style
    - axes.labelsize   13 → 15
    - axes.titlesize   13 → 15
    - xtick/ytick      11 → 13
    - legend.fontsize   9 → 12
    - font.size         12 → 14
    - subplot labels    15 → 17
    - all inline text scaled proportionally

v2.1 CHANGES:
  ★ Subfigure (c) heatmap cell values are now uniformly shown with 4 decimals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, Rectangle
import os

# ================================================================
#  CNS-level publication style — ★ FONT: unified with Fig 10/11
# ================================================================
plt.style.use("default")
plt.rcParams.update({
    "font.family":        "Arial",
    "font.sans-serif":    ["Arial"],
    "mathtext.fontset":   "dejavusans",
    "mathtext.rm":        "Arial",
    "mathtext.it":        "Arial:italic",
    "mathtext.bf":        "Arial:bold",
    "font.size":          14,
    "axes.labelsize":     15,
    "axes.titlesize":     15,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    12,
    "figure.dpi":         300,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.8,
    "savefig.dpi":        900,
    "savefig.format":     "jpg",
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.0,
    "axes.unicode_minus": False,
})

# ================================================================
#  REAL EXPERIMENTAL DATA  (from 50-fold nested CV)
# ================================================================
import pandas as pd
k=pd.read_csv(RES/'Korea_all_folds.csv')
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

# #  COLOUR PALETTE
# ================================================================
C = dict(
    TabPFN="#2C73D2",
    tree="#8B8B8B",
    linear="#5FAD56",
    KNN="#9B59B6",
    SVR="#E67E22",
    MLP="#C0392B",
    RI="#5B9BD5",
    RII="#ED7D31",
    sig="#C0392B",
    ns="#AAAAAA"
)

MC = dict(
    TabPFN=C["TabPFN"],
    RF=C["tree"],
    XGBoost=C["tree"],
    AdaBoost=C["tree"],
    GBR=C["tree"],
    Lasso=C["linear"],
    ElasticNet=C["linear"],
    Ridge=C["linear"],
    KNN=C["KNN"],
    SVR=C["SVR"],
    MLP=C["MLP"]
)

# ================================================================
#  FIGURE  (2 x 2)
# ================================================================
fig, axes = plt.subplots(
    2, 2, figsize=(15, 14),
    gridspec_kw={
        "width_ratios": [1.15, 1.0],
        "height_ratios": [1.0, 1.15]
    }
)
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# ─────────────────────────────────────────────────────
#  (a)  R² RANKING — Korean dataset, all 11 models
# ─────────────────────────────────────────────────────
# Retain vertical ranking bars; show MLP variability on its own full scale.
sub=ax_a.get_subplotspec().subgridspec(2,1,height_ratios=[3,1],hspace=1.15)
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
# #  (c)  MULTI-METRIC HEATMAP — 10 models
#       ★ all displayed values use 4 decimals
# ─────────────────────────────────────────────────────
nmod, nmet = metric_data.shape

norm = np.zeros_like(metric_data)
for j in range(nmet):
    col = metric_data[:, j]
    lo, hi = col.min(), col.max()
    span = max(hi - lo, 1e-12)
    norm[:, j] = (col - lo) / span if metric_higher[j] else (hi - col) / span

for sp in ax_c.spines.values():
    sp.set_visible(False)

im = ax_c.imshow(
    norm,
    aspect="auto",
    cmap="Blues",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

for i in range(nmod):
    for j in range(nmet):
        v = metric_data[i, j]
        col = metric_data[:, j]
        best = (v == col.max()) if metric_higher[j] else (v == col.min())
        bg = norm[i, j]
        tc = "white" if bg > 0.60 else "#222"
        txt = f"{v:.4f}"
        ax_c.text(
            j, i, txt,
            ha="center", va="center",
            fontsize=11 if not best else 11.5,
            fontweight="bold" if best else "normal",
            color=tc
        )

ax_c.add_patch(Rectangle(
    (-0.5, -0.5), nmet, 1,
    fill=False, ec=C["TabPFN"], lw=2.5, zorder=10
))

ax_c.set_xticks(np.arange(nmet))
ax_c.set_xticklabels(metric_labels, fontsize=13, fontweight="bold")
ax_c.xaxis.tick_top()
ax_c.xaxis.set_label_position("top")
ax_c.set_yticks(np.arange(nmod))
ax_c.set_yticklabels(models_10, fontsize=13)
ax_c.tick_params(axis="y", length=0, pad=8)
ax_c.tick_params(axis="x", length=0, pad=6)

ax_c.grid(False)
ax_c.set_xticks(np.arange(nmet + 1) - 0.5, minor=True)
ax_c.set_yticks(np.arange(nmod + 1) - 0.5, minor=True)
ax_c.grid(which="minor", color="white", linewidth=2)
ax_c.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(im, ax=ax_c, fraction=0.032, pad=0.025, shrink=0.82)
cbar.set_label("Within-metric normalized score", fontsize=13)
cbar.ax.tick_params(labelsize=12)

ax_c.text(
    0.50, -0.06,
    r"Bold = column-best;  $\uparrow$ higher is better;  $\downarrow$ lower is better",
    transform=ax_c.transAxes,
    fontsize=10,
    ha="center",
    va="top",
    color="#555",
    fontstyle="italic"
)

# ─────────────────────────────────────────────────────
#  (d)  Delta-R² WITH WILCOXON SIGNIFICANCE
# ─────────────────────────────────────────────────────
sub=ax_d.get_subplotspec().subgridspec(2,1,height_ratios=[5,1],hspace=1.15)
ax_d.remove();ax_d=fig.add_subplot(sub[0]);ax_dm=fig.add_subplot(sub[1])
for ax,mods in [(ax_d,[m for m in models_all if m not in ['TabPFN','MLP']]),(ax_dm,['MLP'])]:
    delta=np.array([repeat['TabPFN']-repeat[m] for m in mods]);ys=np.arange(len(mods))
    ax.barh(ys,delta.mean(axis=1),color=[MC[m] for m in mods],edgecolor='white',height=.62)
    ax.errorbar(delta.mean(axis=1),ys,xerr=delta.std(axis=1,ddof=1),fmt='none',ecolor='#555',capsize=3,lw=1)
    ax.set_yticks(ys,mods);ax.invert_yaxis();ax.axvline(0,c='#777',lw=.7)
    ax.set_xlabel('Paired ΔR², mean ± SD')
ax_d.set_title('TabPFN minus baseline: ten repeat means',fontsize=13)
ax_dm.set_title('MLP: separate full-range axis',fontsize=12)
# #  SUBPLOT LABELS
# ================================================================
for ax,label in [(ax_a,'a'),(ax_b,'b'),(ax_c,'c'),(ax_d,'d')]:
    ax.text(-.13,1.12,f'({label})',transform=ax.transAxes,fontweight='bold',fontsize=17)
fig.subplots_adjust(hspace=.5,wspace=.4)
save(fig,'Fig11',legacy=True)
