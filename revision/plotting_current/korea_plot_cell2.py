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
    "mathtext.fontset":   "custom",
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
models_all = [
    "TabPFN", "RF", "XGBoost", "AdaBoost", "GBR",
    "Lasso", "ElasticNet", "Ridge", "KNN", "SVR", "MLP"
]

r2_mean = np.array([
    0.7948, 0.7726, 0.7644, 0.7622, 0.7454,
    0.7205, 0.7195, 0.7182, 0.6952, 0.6743, -0.0988
])
r2_std = np.array([
    0.1015, 0.1058, 0.1151, 0.1043, 0.1257,
    0.1108, 0.1131, 0.1148, 0.1435, 0.1296, 2.7480
])

rmse_mean = np.array([
    0.1859, 0.1959, 0.1987, 0.2015, 0.2061,
    0.2188, 0.2190, 0.2194, 0.2267, 0.2362, 0.3330
])

mae_mean = np.array([
    0.1556, 0.1609, 0.1598, 0.1646, 0.1664,
    0.1796, 0.1793, 0.1795, 0.1800, 0.1948, 0.2667
])

# Region I (Table 2, Feature Set A)
r2_R1 = np.array([
    0.920, 0.890, 0.880, 0.870, 0.880,
    0.490, 0.500, 0.490, 0.750, 0.650
])
std_R1 = np.array([
    0.040, 0.050, 0.060, 0.070, 0.060,
    0.190, 0.180, 0.200, 0.110, 0.150
])

# Wilcoxon (9 baselines, MLP excluded)
stat_names = [
    "RF", "GBR", "XGBoost", "AdaBoost",
    "Ridge", "Lasso", "ElasticNet", "SVR", "KNN"
]
delta_r2_stat = np.array([
    0.0222, 0.0495, 0.0305, 0.0326,
    0.0766, 0.0744, 0.0753, 0.1206, 0.0997
])
sig_label = [
    "n.s.", "n.s.", "n.s.", "n.s.",
    "***", "***", "***", "***", "***"
]

# ================================================================
#  DERIVED HEATMAP METRICS  (10 models, MLP excluded)
# ================================================================
models_10 = models_all[:-1]
r2_10     = r2_mean[:-1]
r2s_10    = r2_std[:-1]
rmse_10   = rmse_mean[:-1]
mae_10    = mae_mean[:-1]

one_cv = 1.0 - r2s_10 / np.abs(r2_10)
r2_hm  = 2.0 * r2_R1 * r2_10 / (r2_R1 + r2_10)

metric_data = np.column_stack([r2_10, rmse_10, mae_10, one_cv, r2_hm])
metric_labels = [
    r"R$^{2}$ $\uparrow$",
    r"RMSE $\downarrow$",
    r"MAE $\downarrow$",
    r"1-CV $\uparrow$",
    r"R$^{2}$-HM $\uparrow$"
]
metric_higher = [True, False, False, True, True]

# ================================================================
#  DATA VERIFICATION
# ================================================================
ret_pct = r2_10[0] / r2_R1[0] * 100
print("=" * 60)
print("  DATA VERIFICATION")
print("=" * 60)
print(f"  Region II  TabPFN R2 = {r2_mean[0]:.4f} +/- {r2_std[0]:.4f}")
print(f"  Region I   TabPFN R2 = {r2_R1[0]:.3f}  +/- {std_R1[0]:.3f}")
print(f"  R2 retention         = {ret_pct:.1f}%")
print(f"  TabPFN RMSE          = {rmse_mean[0]:.4f}")
print(f"  TabPFN MAE           = {mae_mean[0]:.4f}")
print(f"  TabPFN 1-CV          = {one_cv[0]:.4f}")
print(f"  TabPFN R2-HM         = {r2_hm[0]:.4f}")
n_sig = sum(1 for s in sig_label if s == "***")
n_ns  = sum(1 for s in sig_label if s == "n.s.")
print(f"  Wilcoxon: {n_sig} significant,  {n_ns} not significant")
print("=" * 60)

# ================================================================
#  COLOUR PALETTE
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
    2, 2, figsize=(15, 12),
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
colors_a = [MC[m] for m in models_all]
std_cap = r2_std.copy()
std_cap[-1] = min(std_cap[-1], 0.35)

ax_a.bar(
    np.arange(len(models_all)),
    r2_mean,
    yerr=std_cap,
    capsize=3.5,
    color=colors_a,
    edgecolor="white",
    linewidth=0.6,
    width=0.72,
    zorder=3,
    error_kw=dict(lw=1.0, capthick=0.8, ecolor="#616161")
)

# Rank labels inside bars
for i, (m, v) in enumerate(zip(models_all, r2_mean)):
    if v > 0.15:
        tc = "white" if v > 0.50 else "#333"
        fx = [] if v > 0.50 else [pe.withStroke(linewidth=2.5, foreground="white")]
        ax_a.text(
            i, v * 0.50, f"#{i+1}",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=tc,
            zorder=5, path_effects=fx
        )

# TabPFN value annotation
ax_a.annotate(
    f"{r2_mean[0]:.4f}",
    xy=(0, r2_mean[0]),
    xytext=(0, 0.95),
    fontsize=13,
    fontweight="bold",
    color=C["TabPFN"],
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="-", color=C["TabPFN"], lw=1)
)

# #11 + Generalization failure
ax_a.annotate(
    "#11\nGeneralization failure",
    xy=(10, r2_mean[-1] - 0.02),
    xytext=(7.2, -0.22),
    fontsize=11,
    fontweight="bold",
    color=C["MLP"],
    ha="center",
    arrowprops=dict(
        arrowstyle="->",
        color=C["MLP"],
        lw=1.2,
        connectionstyle="arc3,rad=0.20"
    ),
    bbox=dict(boxstyle="round,pad=0.25", fc="#FFF0F0", ec=C["MLP"], alpha=0.90)
)

ax_a.set_xticks(np.arange(len(models_all)))
ax_a.set_xticklabels(models_all, rotation=38, ha="right", fontsize=13)
ax_a.set_ylabel(r"Coefficient of determination (R$^{2}$)")
ax_a.set_xlabel("Machine learning model")
ax_a.set_ylim(-0.30, 1.10)
ax_a.axhline(0, color="grey", linewidth=0.6)
ax_a.set_axisbelow(True)

# ─────────────────────────────────────────────────────
#  (b)  CROSS-REGIONAL R² COMPARISON — 10 models
# ─────────────────────────────────────────────────────
xb = np.arange(len(models_10))
w = 0.36

ax_b.bar(
    xb - w / 2, r2_R1, w,
    yerr=std_R1, capsize=2.5,
    error_kw=dict(lw=0.9, color="#555"),
    color=C["RI"], edgecolor="white", linewidth=0.5,
    label="Region I", zorder=3
)
ax_b.bar(
    xb + w / 2, r2_10, w,
    yerr=r2s_10, capsize=2.5,
    error_kw=dict(lw=0.9, color="#555"),
    color=C["RII"], edgecolor="white", linewidth=0.5,
    label="Region II", zorder=3
)

ax_b.annotate(
    f"R$^{{2}}$ retention: {ret_pct:.1f}%",
    xy=(w / 2, r2_10[0] + r2s_10[0] + 0.02),
    xytext=(-0.2, 1.08),
    fontsize=12.5,
    fontweight="bold",
    color="#333",
    bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFCC", ec="#999", alpha=0.95),
    arrowprops=dict(arrowstyle="->", color="#666", lw=1.2)
)

ax_b.text(
    0.98, 0.02,
    r"MLP excluded" "\n" r"(Region II: R$^{2}$ < 0)",
    transform=ax_b.transAxes,
    fontsize=10.5,
    color=C["MLP"],
    ha="right",
    va="bottom",
    fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec=C["MLP"], alpha=0.80)
)

ax_b.set_xticks(xb)
ax_b.set_xticklabels(models_10, rotation=38, ha="right", fontsize=13)
ax_b.set_ylabel(r"Coefficient of determination (R$^{2}$)")
ax_b.set_xlabel("Machine learning model")
ax_b.set_ylim(0, 1.25)
ax_b.set_axisbelow(True)
ax_b.legend(fontsize=12, loc="upper right", framealpha=0.95, edgecolor="lightgray")

# ─────────────────────────────────────────────────────
#  (c)  MULTI-METRIC HEATMAP — 10 models
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
    cmap="RdYlBu_r",
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
        tc = "white" if bg > 0.72 or bg < 0.28 else "#222"
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
cbar.set_label("Normalized score", fontsize=13)
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
idx9 = np.argsort(delta_r2_stat)
nm_s = [stat_names[i] for i in idx9]
dr_s = delta_r2_stat[idx9]
sg_s = [sig_label[i] for i in idx9]

yd = np.arange(len(nm_s))
cols_d = [C["sig"] if s == "***" else C["ns"] for s in sg_s]

ax_d.barh(
    yd, dr_s,
    color=cols_d,
    edgecolor="white",
    linewidth=0.5,
    height=0.62,
    zorder=3
)

for i, (v, s) in enumerate(zip(dr_s, sg_s)):
    sc = C["sig"] if s == "***" else "#455A64"
    ax_d.text(
        v + 0.002, i, s,
        va="center", ha="left",
        fontsize=13, fontweight="bold", color=sc
    )

ax_d.set_yticks(yd)
ax_d.set_yticklabels(nm_s, fontsize=13)
ax_d.set_xlabel(r"$\Delta$R$^{2}$ (TabPFN $-$ Baseline)")
ax_d.set_xlim(0, max(dr_s) * 1.30)
ax_d.set_axisbelow(True)

leg_d = []
if any(s == "***" for s in sg_s):
    leg_d.append(Patch(
        fc=C["sig"],
        label=r"Significant (Holm-Bonferroni, $\alpha$ = 0.05)"
    ))
if any(s == "n.s." for s in sg_s):
    leg_d.append(Patch(fc=C["ns"], label="Not significant"))

ax_d.legend(
    handles=leg_d,
    fontsize=11,
    loc="lower right",
    framealpha=0.95,
    edgecolor="lightgray"
)

ax_d.text(
    0.98, 0.03,
    r"MLP excluded ($\Delta$R$^{2}$ = 0.89)",
    transform=ax_d.transAxes,
    fontsize=9.5,
    ha="right",
    va="bottom",
    style="italic",
    color="#757575"
)

# ================================================================
#  SUBPLOT LABELS
# ================================================================
LABEL_FS = 17

label_kw = dict(
    fontsize=LABEL_FS,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.98, edgecolor="lightgray"),
    verticalalignment="top",
    horizontalalignment="left",
    zorder=10
)

# (a) placed at lower-left
ax_a.text(
    0.02, 0.08, "(a)",
    transform=ax_a.transAxes,
    fontsize=LABEL_FS,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.98, edgecolor="lightgray"),
    va="bottom",
    ha="left",
    zorder=10
)

# (b) and (d) at upper-left
for ax, lbl in zip([ax_b, ax_d], ["(b)", "(d)"]):
    ax.text(0.02, 0.98, lbl, transform=ax.transAxes, **label_kw)

# (c) offset above heatmap
ax_c.text(-0.08, 1.10, "(c)", transform=ax_c.transAxes, **label_kw)

# ================================================================
#  LAYOUT  &  POST-PROCESSING
# ================================================================
plt.tight_layout(pad=2.0)

fig.canvas.draw()
for tl, mn in zip(ax_c.get_yticklabels(), models_10):
    tl.set_color(MC[mn])
    if mn == "TabPFN":
        tl.set_fontweight("bold")

for ax in axes.flat:
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Arial")
        tick.set_fontsize(13)

# ================================================================
#  SAVE
# ================================================================
save_dir = os.path.join(os.getcwd(), "manuscript_figures")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "fig12_cross_regional_validation.jpg")

fig.savefig(
    save_path,
    dpi=900,
    format="jpg",
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor="white",
    edgecolor="none"
)

print(f"\n{'='*60}")
print(f"  Fig. 12 saved  ->  {save_path}")
print(f"{'='*60}")
print(f"  FONT SIZE CHANGES (v1 → v2):")
print(f"    font.size:        12 → 14")
print(f"    axes.labelsize:   13 → 15")
print(f"    axes.titlesize:   13 → 15")
print(f"    xtick/ytick:      11 → 13")
print(f"    legend.fontsize:   9 → 12")
print(f"    subplot labels:   15 → 17")
print(f"    bar rank text:     9 → 11")
print(f"    TabPFN annotate:  11 → 13")
print(f"    heatmap cells:  all → 4 decimals")
print(f"    metric labels:  10.5 → 13")
print(f"    colorbar label:   10 → 13")
print(f"    Wilcoxon sig:     11 → 13")
print(f"    all inline text scaled +2pt")
print(f"  DATA CHECK:")
print(f"    [OK] TabPFN R2 = {r2_mean[0]:.4f}")
print(f"    [OK] R2 retention = {ret_pct:.1f}%")
print(f"    [OK] heatmap values shown with 4 decimals")
print(f"    [OK] 900 dpi JPG")
print(f"{'='*60}")

plt.show()