#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════
part2 不确定性可视化
Fig. 9 — Cross-Conformal Prediction Uncertainty Quantification
TabPFN · Debris-Flow Volume · Longmenshan Fault Zone
═══════════════════════════════════════════════════════════════════════════
v3.3 字号放大版:
  全局 rcParams 及逐面板字号统一提升 +2pt
═══════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ═══════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 1.  PATHS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).resolve().parent.parent / 'rerun_current_figures' / 'inputs'
UQ_DIR   = DATA_DIR / "UQ_conformal_results"

SAVE_DIR = DATA_DIR / "Fig9_UQ_CrossConformal_publication"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FILE_SAMPLES = UQ_DIR / "UQ_XConf_samples_FSA.csv"
FILE_SUMMARY = UQ_DIR / "UQ_XConf_summary_FSA.csv"

COL_FOLD  = "fold"
COL_SID   = "idx"
COL_YOBS  = "y_obs"
COL_YPRED = "y_pred"
COL_ALPHA = "alpha"
COL_LO    = "xc_lo"
COL_HI    = "xc_hi"
COL_WID   = "xc_wid"
COL_COV   = "xc_cov"

NEST_ALPHAS = [0.50, 0.30, 0.10, 0.05, 0.01]
NEST_LABELS = ["50%", "70%", "90%", "95%", "99%"]

# ═══════════════════════════════════════════════════════════════
# 2.  CNS-LEVEL FIGURE STYLE  ★ 全局字号 +2pt
# ═══════════════════════════════════════════════════════════════
plt.style.use('default')
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":    "custom",
    "mathtext.rm":         "Arial",
    "mathtext.it":         "Arial:italic",
    "mathtext.bf":         "Arial:bold",
    "font.size":           14,          # ★ 12 → 14
    "axes.labelsize":      15,          # ★ 13 → 15
    "axes.titlesize":      15,          # ★ 13 → 15
    "xtick.labelsize":     13,          # ★ 11 → 13
    "ytick.labelsize":     13,          # ★ 11 → 13
    "legend.fontsize":     12,          # ★ 10 → 12
    "figure.dpi":          300,
    "axes.grid":           True,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.8,
    "savefig.dpi":         900,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.1,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      1.0,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# ═══════════════════════════════════════════════════════════════
# 3.  LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  Fig. 9 — Cross-Conformal UQ · Publication Figure (v3.3 font-up)")
print("  TabPFN · Debris-Flow Volume · Longmenshan Fault Zone")
print("=" * 70)

for fp, tag in [(FILE_SAMPLES, "samples"), (FILE_SUMMARY, "summary")]:
    if not fp.exists():
        raise FileNotFoundError(
            f"\n❌  {tag} CSV not found:\n"
            f"    {fp}\n"
            f"    → Run Part 1 (computation script v4.2b) first.\n"
            f"    → Expected output directory: {UQ_DIR}")

df_s = pd.read_csv(FILE_SAMPLES)
df_m = pd.read_csv(FILE_SUMMARY)

print(f"  ✔  samples : {len(df_s):,} rows   columns = {list(df_s.columns)}")
print(f"  ✔  summary : {len(df_m)} rows")

required = [COL_FOLD, COL_SID, COL_YOBS, COL_YPRED,
            COL_ALPHA, COL_LO, COL_HI, COL_WID, COL_COV]
missing  = [c for c in required if c not in df_s.columns]
if missing:
    raise KeyError(
        f"Missing columns in CSV: {missing}\n"
        f"Available: {list(df_s.columns)}")

# ═══════════════════════════════════════════════════════════════
# 4.  AGGREGATE ACROSS CV FOLDS
# ═══════════════════════════════════════════════════════════════
df_agg = (df_s
          .groupby([COL_SID, COL_ALPHA])
          .agg(
              y_obs   = (COL_YOBS,  "first"),
              y_pred  = (COL_YPRED, "median"),
              xc_lo   = (COL_LO,    "median"),
              xc_hi   = (COL_HI,    "median"),
              xc_wid  = (COL_WID,   "median"),
              xc_cov  = (COL_COV,   "mean"),
              n_folds = (COL_FOLD,  "count"),
          )
          .reset_index())

df_agg["covered"] = (df_agg["xc_cov"] >= 0.5).astype(int)

n_unique = df_agg[COL_SID].nunique()
print(f"  ✔  Aggregated: {n_unique} unique samples, "
      f"~{df_agg['n_folds'].median():.0f} folds per sample")

# ═══════════════════════════════════════════════════════════════
# 5.  PREPARE DATA FOR EACH PANEL
# ═══════════════════════════════════════════════════════════════

# ────── Panel (a): 90 % intervals ──────
ALPHA_90 = 0.10
df_90 = (df_agg[np.abs(df_agg[COL_ALPHA] - ALPHA_90) < 1e-6]
         .copy()
         .sort_values("y_obs")
         .reset_index(drop=True))

n_samples  = len(df_90)
n_covered  = int(df_90["covered"].sum())
n_missed   = n_samples - n_covered
emp_cov_90 = n_covered / n_samples * 100
med_wid_90 = df_90["xc_wid"].median()

print(f"\n  Panel (a)  90% PI: n = {n_samples}, "
      f"coverage = {emp_cov_90:.1f}%, "
      f"median width = {med_wid_90:.3f}")

# ────── Panel (b): multi-level calibration ──────
all_alphas = sorted(df_s[COL_ALPHA].unique())

coverages_b, empirical_b = [], []
ci_lo_b, ci_hi_b = [], []

for a in all_alphas:
    sub = df_s[np.abs(df_s[COL_ALPHA] - a) < 1e-6]
    nom = 1.0 - a
    emp = sub[COL_COV].mean()
    n   = len(sub)
    z   = 1.96
    den = 1.0 + z**2 / n
    ctr = (emp + z**2 / (2.0 * n)) / den
    mar = z * np.sqrt((emp * (1 - emp) + z**2 / (4 * n)) / n) / den
    coverages_b.append(nom)
    empirical_b.append(emp)
    ci_lo_b.append(max(ctr - mar, 0.0))
    ci_hi_b.append(min(ctr + mar, 1.0))

coverages_b = np.asarray(coverages_b)
empirical_b = np.asarray(empirical_b)
ci_lo_b     = np.asarray(ci_lo_b)
ci_hi_b     = np.asarray(ci_hi_b)
res_ceiling = (n_samples - 1) / n_samples

# ────── Panel (c): width distributions ──────
alphas_c  = sorted(df_agg[COL_ALPHA].unique(), reverse=True)
labels_c  = [f"{(1-a)*100:.0f}%" for a in alphas_c]

width_dists = []
for a in alphas_c:
    w = df_agg.loc[np.abs(df_agg[COL_ALPHA] - a) < 1e-6, "xc_wid"].values
    width_dists.append(w)

print(f"  Panel (c)  levels: {labels_c[0]} → {labels_c[-1]}  "
      f"({len(width_dists)} bins)")

# ────── Panel (d): representative catchments ──────
df_sorted = df_90.sort_values("y_obs").reset_index(drop=True)
n_s = len(df_sorted)

pctl_pos = [0.10, 0.35, 0.65, 0.90]
pick_rows = sorted(set(
    min(int(round(p * (n_s - 1))), n_s - 1) for p in pctl_pos))

if len(pick_rows) < 4:
    extras = [i for i in range(n_s) if i not in pick_rows]
    pick_rows = sorted((pick_rows + extras[:4 - len(pick_rows)])[:4])

PICK_IDS = [int(df_sorted.iloc[i][COL_SID]) for i in pick_rows]

V0_DISPLAY = {}
for sid in PICK_IDS:
    obs_log = df_sorted.loc[
        df_sorted[COL_SID] == sid, "y_obs"].iloc[0]
    V0_DISPLAY[sid] = round(10**obs_log, 1)

print(f"  Panel (d)  catchments: {PICK_IDS}")
print(f"             V₀ (×10⁴ m³): {V0_DISPLAY}")

nest_data = {}
for sid in PICK_IDS:
    sub = df_agg[df_agg[COL_SID] == sid]
    if sub.empty:
        continue
    nest_data[sid] = {
        "obs":  sub["y_obs"].iloc[0],
        "pred": sub["y_pred"].iloc[0],
    }
    for a in NEST_ALPHAS:
        row = sub[np.abs(sub[COL_ALPHA] - a) < 1e-6]
        if len(row):
            nest_data[sid][a] = (
                row["xc_lo"].iloc[0],
                row["xc_hi"].iloc[0],
                row["y_pred"].iloc[0])


# ═══════════════════════════════════════════════════════════════
# 6.  CREATE FOUR-PANEL FIGURE  (v3.3 font-up)
# ═══════════════════════════════════════════════════════════════
print("\n  🎨  Rendering publication figure (v3.3 font-up) …")

C_TEAL      = "#2E8B7E"
C_CORAL     = "#D2691E"
C_BLUE      = "#1F77B4"
C_LIGHTBLUE = "#4DBECC"
C_BAR_COV   = "#7EC8BD"
C_BAR_MISS  = "#E8A87C"
C_VIO_FILL  = "#7CB9E8"
C_VIO_EDGE  = "#3A7CA5"
C_PRED_MK   = "#2C3E50"

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

YLABEL_VOLUME = (r"Debris-flow volume, "
                 r"log$_{10}$($V_{\mathrm{0}}$ / 10$^{4}$ m$^{3}$)")


# ──────────────────────────────────────────────────────────────
# (a)  90 % Cross-Conformal Prediction Intervals
# ──────────────────────────────────────────────────────────────
ax = axes[0, 0]

for i in range(n_samples):
    row = df_90.iloc[i]
    c = C_BAR_COV if row["covered"] else C_BAR_MISS
    ax.plot([i, i], [row["xc_lo"], row["xc_hi"]],
            color=c, lw=1.5, alpha=0.55,
            solid_capstyle="round", zorder=1)

ax.plot(range(n_samples), df_90["y_pred"].values,
        color=C_LIGHTBLUE, lw=1.3, alpha=0.85,
        label="Median prediction", zorder=2)

mask_cov = df_90["covered"].astype(bool)

ax.scatter(df_90.index[mask_cov],
           df_90.loc[mask_cov, "y_obs"],
           c=C_TEAL, s=32, zorder=3,
           edgecolors="white", linewidths=0.4,
           label=f"Covered ({n_covered}/{n_samples})")

ax.scatter(df_90.index[~mask_cov],
           df_90.loc[~mask_cov, "y_obs"],
           c=C_CORAL, s=58, marker="D", zorder=4,
           edgecolors="white", linewidths=0.5,
           label=f"Missed ({n_missed}/{n_samples})")

info_a = (f"Empirical coverage: {emp_cov_90:.1f}%\n"
          f"Median width: {med_wid_90:.3f} log$_{{10}}$ units")
ax.text(0.97, 0.05, info_a,
        transform=ax.transAxes, fontsize=13,        # ★ 11 → 13
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4",
                  fc="white", ec="0.70", alpha=0.95))

ax.set_xlabel("Sample index (sorted by observed value)",
              fontweight="normal")
ax.set_ylabel(YLABEL_VOLUME, fontweight="normal")

ax.legend(loc="upper left", frameon=True, framealpha=0.95,
          fontsize=12, edgecolor="0.80",             # ★ 10 → 12
          bbox_to_anchor=(0.00, 0.82))


# ──────────────────────────────────────────────────────────────
# (b)  Multi-Level Coverage Calibration Curve
# ──────────────────────────────────────────────────────────────
ax = axes[0, 1]

x_diag = np.linspace(0.44, 1.04, 300)
ax.fill_between(x_diag, x_diag - 0.05, x_diag + 0.05,
                color="0.88", alpha=0.50,
                label=r"$\pm$5 pp tolerance band", zorder=0)
ax.plot(x_diag, x_diag, "--", color="0.55", lw=1.2,
        label="Ideal calibration", zorder=1)

ax.axhline(res_ceiling, color=C_CORAL, ls=":", lw=1.2,
           label=f"Resolution ceiling ({res_ceiling*100:.1f}%)",
           zorder=1)

ax.errorbar(coverages_b, empirical_b,
            yerr=[empirical_b - ci_lo_b,
                  ci_hi_b - empirical_b],
            fmt="o-", color=C_BLUE, ms=7,
            capsize=4, lw=2.0, markeredgecolor="white",
            markeredgewidth=0.5,
            label="Cross-conformal (TabPFN)", zorder=3)

ax.set_xlabel("Nominal coverage", fontweight="normal")
ax.set_ylabel("Empirical coverage", fontweight="normal")
ax.set_xlim(0.46, 1.03)
ax.set_ylim(0.46, 1.06)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(loc="lower right", frameon=True, framealpha=0.95,
          fontsize=12, edgecolor="0.80")             # ★ 10 → 12


# ──────────────────────────────────────────────────────────────
# (c)  Prediction Interval Width — Violin + Box Plots
# ──────────────────────────────────────────────────────────────
ax = axes[1, 0]

pos = list(range(len(width_dists)))

for idx_v, (w_data, p) in enumerate(zip(width_dists, pos)):
    if len(w_data) < 2:
        continue
    std_v = np.std(w_data)
    if std_v < 0.02:
        w_plot = w_data + np.random.normal(0, 0.008, len(w_data))
    else:
        w_plot = w_data

    vp = ax.violinplot([w_plot], positions=[p],
                       showmedians=False, showextrema=False,
                       widths=0.75)
    for pc in vp["bodies"]:
        pc.set_facecolor(C_VIO_FILL)
        pc.set_edgecolor(C_VIO_EDGE)
        pc.set_alpha(0.55)
        pc.set_linewidth(0.8)

bp = ax.boxplot(width_dists, positions=pos,
                widths=0.13, patch_artist=True,
                showfliers=False, zorder=3)
for patch in bp["boxes"]:
    patch.set_facecolor(C_VIO_EDGE)
    patch.set_alpha(0.80)
for el in ("whiskers", "caps"):
    for ln in bp[el]:
        ln.set_color(C_VIO_EDGE)
        ln.set_linewidth(1.0)
for ml in bp["medians"]:
    ml.set_color("white")
    ml.set_linewidth(1.6)

ax.set_xticks(pos)
ax.set_xticklabels(labels_c)
ax.set_xlabel("Nominal coverage level", fontweight="normal")
ax.set_ylabel(r"Prediction interval width (log$_{10}$ units)",
              fontweight="normal")

try:
    idx90 = labels_c.index("90%")
    med90 = np.median(width_dists[idx90])
    all_widths_flat = np.concatenate(width_dists)
    y_span = np.max(all_widths_flat) - np.min(all_widths_flat)
    ann_x = max(idx90 - 2.0, 0.5)
    ann_y = med90 + y_span * 0.10
    ax.annotate(
        f"90%: median = {med90:.2f}",
        xy=(idx90, med90),
        xytext=(ann_x, ann_y),
        fontsize=13, color="0.25",                   # ★ 11 → 13
        arrowprops=dict(arrowstyle="-", color="0.50", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.3",
                  fc="white", ec="0.70", alpha=0.95))
except ValueError:
    pass


# ──────────────────────────────────────────────────────────────
# (d)  Nested Fan-Chart for Representative Catchments
# ──────────────────────────────────────────────────────────────
ax = axes[1, 1]

y_pos  = list(range(len(PICK_IDS)))

y_labs = []
for sid in PICK_IDS:
    v0 = V0_DISPLAY.get(sid, "?")
    y_labs.append(
        f"#{sid}  "
        r"($V_{\mathrm{0}}$" + f"={v0}" + r"$\times\!10^{4}$m$^{3}$)")

NEST_COLORS  = ["#08306B", "#2171B5", "#4292C6", "#9ECAE1", "#DEEBF7"]
BAR_HEIGHTS  = [0.20,       0.34,      0.50,      0.62,      0.74]

for k in range(len(NEST_ALPHAS) - 1, -1, -1):
    a     = NEST_ALPHAS[k]
    color = NEST_COLORS[k]
    h     = BAR_HEIGHTS[k]
    z_ord = 2 + (len(NEST_ALPHAS) - k)
    for j, sid in enumerate(PICK_IDS):
        if sid in nest_data and a in nest_data[sid]:
            lo, hi, _ = nest_data[sid][a]
            ax.barh(y_pos[j], hi - lo, left=lo,
                    height=h, color=color, alpha=0.82,
                    edgecolor="none", zorder=z_ord)

for j, sid in enumerate(PICK_IDS):
    if sid not in nest_data:
        continue
    ax.plot(nest_data[sid]["obs"], y_pos[j], "D",
            color=C_CORAL, ms=9.5, zorder=10,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.plot(nest_data[sid]["pred"], y_pos[j], "o",
            color=C_PRED_MK, ms=7.5, zorder=10,
            markeredgecolor="white", markeredgewidth=1.2)

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labs, fontsize=12)              # ★ 10 → 12
ax.set_ylabel("Representative catchment", fontweight="normal")
ax.set_xlabel(YLABEL_VOLUME, fontweight="normal")

legend_handles = [
    Patch(fc=NEST_COLORS[0], alpha=0.82, label="50%"),
    Patch(fc=NEST_COLORS[1], alpha=0.82, label="70%"),
    Patch(fc=NEST_COLORS[2], alpha=0.82, label="90%"),
    Patch(fc=NEST_COLORS[3], alpha=0.82, label="95%"),
    Patch(fc=NEST_COLORS[4], alpha=0.82, label="99%"),
    Line2D([], [], marker="D", color="w", markerfacecolor=C_CORAL,
           ms=8, markeredgecolor="white", label="Observed"),
    Line2D([], [], marker="o", color="w", markerfacecolor=C_PRED_MK,
           markeredgecolor="white", ms=7, label="Predicted"),
]
ax.legend(handles=legend_handles, loc="lower right",
          frameon=True, framealpha=0.95,
          fontsize=12, ncol=2, edgecolor="0.80",     # ★ 10 → 12
          handletextpad=0.5, columnspacing=1.0,
          bbox_to_anchor=(0.98, 0.02))


# ═══════════════════════════════════════════════════════════════
# 7.  SUBPLOT LABELS & GLOBAL FORMATTING
# ═══════════════════════════════════════════════════════════════
panel_tags = ["(a)", "(b)", "(c)", "(d)"]

for ax_i, tag in zip(axes.flat, panel_tags):
    ax_i.text(0.02, 0.98, tag,
              transform=ax_i.transAxes,
              fontsize=17, fontweight="bold",         # ★ 15 → 17
              verticalalignment="top",
              horizontalalignment="left",
              bbox=dict(boxstyle="round,pad=0.25",
                        facecolor="white", alpha=0.98,
                        edgecolor="lightgray"),
              zorder=10)

for ax_i in axes.flat:
    ax_i.grid(True, alpha=0.25, linewidth=0.8)
    for tick in ax_i.get_xticklabels() + ax_i.get_yticklabels():
        tick.set_fontfamily("Arial")
        tick.set_fontsize(13)                        # ★ 11 → 13

plt.tight_layout(pad=2.2, h_pad=2.8, w_pad=2.5)


# ═══════════════════════════════════════════════════════════════
# 8.  SAVE
# ═══════════════════════════════════════════════════════════════
save_path = SAVE_DIR / "Fig9_UQ_CrossConformal_4panel_v3final.jpg"

fig.savefig(str(save_path),
            dpi=900, format="jpg",
            bbox_inches="tight", pad_inches=0.08,
            facecolor="white", edgecolor="none",
            pil_kwargs={"quality": 95})

plt.show()

# ═══════════════════════════════════════════════════════════════
# 9.  SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print(f"  🏆  Figure saved! (v3.3 font-up)")
print(f"{'═' * 70}")
print(f"  📁  Folder : {SAVE_DIR}")
print(f"  📊  File   : {save_path.name}")
print(f"  📐  DPI    : 900  |  Size : 15 × 10.5 in")
print(f"{'─' * 70}")
print(f"  ✅  (a) 90% PI: {n_samples} samples, "
      f"cov = {emp_cov_90:.1f}%, width = {med_wid_90:.3f}")
print(f"  ✅  (b) Calibration: {len(all_alphas)} levels")
print(f"  ✅  (c) Violin: {labels_c[0]} → {labels_c[-1]}")
print(f"  ✅  (d) Fan-chart: {len(PICK_IDS)} catchments")
print(f"{'─' * 70}")
print(f"  🔧  v3.2→v3.3 改动 (全局 +2pt):")
print(f"      rcParams font.size      12→14")
print(f"      rcParams axes.labelsize  13→15")
print(f"      rcParams axes.titlesize  13→15")
print(f"      rcParams xtick/ytick     11→13")
print(f"      rcParams legend.fontsize 10→12")
print(f"      (a) info box 11→13, legend 10→12")
print(f"      (b) legend 10→12")
print(f"      (c) annotation 11→13")
print(f"      (d) y-tick 10→12, legend 10→12")
print(f"      §7 panel tags 15→17, tick覆写 11→13")
print(f"{'═' * 70}\n  ✅  Done.\n")