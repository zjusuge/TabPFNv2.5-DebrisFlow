"""
Cell 3 ── Fig 2: Feature Analysis (Spearman heatmap, violin, distribution, scatter)
"""

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ═══════ (a) Spearman correlation heatmap ═══════
ax = axes[0, 0]

corr_cols = ["A_km2", "H_m", "L_km", "D_km", "J_permille",
             "V_landslide_1e4m3", "log_V0"]
corr_labels = [r"$A$", r"$H$", r"$L$", r"$D_f$",
               r"$J$", r"$V_{land}$", r"log$V_0$"]
corr_mat = df_raw[corr_cols].corr(method="spearman")
n_var = len(corr_labels)

ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)

for i in range(n_var + 1):
    ax.axhline(i - 0.5, color="white", linewidth=1.5)
    ax.axvline(i - 0.5, color="white", linewidth=1.5)

ax.set_xticks(range(n_var))
ax.set_yticks(range(n_var))
ax.set_xticklabels(corr_labels, fontsize=11)
ax.set_yticklabels(corr_labels, fontsize=11)
ax.tick_params(length=0)

for i in range(n_var):
    for j in range(n_var):
        val = corr_mat.values[i, j]
        txt_color = "white" if abs(val) > 0.55 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10.5, color=txt_color,
                fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label("Spearman ρ", fontsize=12)
cbar.ax.tick_params(labelsize=10)
PL(ax, "(a)")

# ═══════ (b) Violin plots and boxplots (z-score) ═══════
ax = axes[0, 1]

feat_violin = ["V_landslide_1e4m3", "A_km2", "H_m",
               "J_permille", "L_km", "D_km"]
vlabels = [r"$V_{land}$", r"$A$", r"$H$", r"$J$", r"$L$", r"$D_f$"]
is_land = [True, False, False, False, False, False]
CLR_LAND = "#D55E00"
CLR_TOPO = "#0072B2"

z_data = []
for col in feat_violin:
    v = df_raw[col].values.astype(float)
    z_data.append((v - v.mean()) / v.std())

parts = ax.violinplot(z_data, positions=range(len(feat_violin)),
                      showmedians=False, showextrema=False)
for idx, body in enumerate(parts["bodies"]):
    c = CLR_LAND if is_land[idx] else CLR_TOPO
    body.set_facecolor(c)
    body.set_alpha(0.25)
    body.set_edgecolor(c)
    body.set_linewidth(0.8)

bp = ax.boxplot(z_data, positions=range(len(feat_violin)), widths=0.18,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.5,
                                markerfacecolor="#555555",
                                markeredgecolor="#555555"),
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8))
for patch in bp["boxes"]:
    patch.set_facecolor("white")
    patch.set_alpha(0.9)
    patch.set_linewidth(0.8)

ax.set_xticks(range(len(vlabels)))
ax.set_xticklabels(vlabels, fontsize=11)
ax.set_ylabel("Standardised value (z-score)")
ax.legend(
    [Patch(fc=CLR_LAND, alpha=0.3, ec=CLR_LAND, lw=0.8),
     Patch(fc=CLR_TOPO, alpha=0.3, ec=CLR_TOPO, lw=0.8)],
    ["Landslide-derived", "Topographic / Tectonic"],
    fontsize=10, loc="upper right", framealpha=0.95,
    edgecolor="lightgray")
PL(ax, "(b)")

# ═══════ (c) log₁₀(V₀) probability distribution; prevent overlap between the legend and panel (c) label ═══════
ax = axes[1, 0]

lv = df_raw["log_V0"].values
ax.hist(lv, bins=12, density=True, alpha=0.4,
        color="#BBBBBB", edgecolor="#888888", linewidth=0.8)
xk = np.linspace(lv.min() - 0.5, lv.max() + 0.5, 300)
kde = gaussian_kde(lv)
ax.plot(xk, kde(xk), color="#D55E00", lw=2.5, label="KDE", zorder=3)
ax.axvline(lv.mean(), color="#D55E00", ls="--", lw=1.5,
           label=f"Mean = {lv.mean():.2f}")
ax.axvline(np.median(lv), color="#0072B2", ls="-.", lw=1.5,
           label=f"Median = {np.median(lv):.2f}")

ax.set_xlabel(r"log$_{10}$($V_0$)  [$V_0$ in ×10$^4$ m³]")
ax.set_ylabel("Probability density")

# ★★ Use bbox_to_anchor to place the legend below the panel (c) label without overlap
ax.legend(fontsize=10, loc="upper left", framealpha=0.95,
          edgecolor="lightgray",
          bbox_to_anchor=(0.12, 0.88))

stxt = (f"n = {len(lv)}\n"
        f"Range: [{lv.min():.2f}, {lv.max():.2f}]\n"
        f"σ = {lv.std():.3f}\n"
        f"Skewness = {stats.skew(lv):.3f}\n"
        f"Kurtosis = {stats.kurtosis(lv):.3f}")
ax.text(0.97, 0.97, stxt, transform=ax.transAxes, fontsize=10,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  alpha=0.95, edgecolor="gray", linewidth=0.5))
PL(ax, "(c)")

# ═══════ (d) V_land vs log₁₀(V₀) scatterplot ═══════
ax = axes[1, 1]

vl = df_raw["V_landslide_1e4m3"].values
J = df_raw["J_permille"].values
lv = df_raw["log_V0"].values

J_thr = np.percentile(J, [33.3, 66.7])
m_hi = J > J_thr[1]
m_mi = (J >= J_thr[0]) & (J <= J_thr[1])
m_lo = J < J_thr[0]

ax.scatter(vl[m_hi], lv[m_hi], c="#D55E00", s=48, alpha=0.75,
           edgecolors="white", lw=0.5, zorder=3,
           label=f"J > {J_thr[1]:.0f} ‰  (steep)")
ax.scatter(vl[m_mi], lv[m_mi], c="#E69F00", s=48, alpha=0.75,
           edgecolors="white", lw=0.5, zorder=3,
           label=f"J = {J_thr[0]:.0f}–{J_thr[1]:.0f} ‰  (moderate)")
ax.scatter(vl[m_lo], lv[m_lo], c="#0072B2", s=48, alpha=0.75,
           edgecolors="white", lw=0.5, zorder=3,
           label=f"J < {J_thr[0]:.0f} ‰  (gentle)")

z = np.polyfit(vl, lv, 1)
xf = np.linspace(vl.min(), vl.max(), 200)
ax.plot(xf, np.polyval(z, xf), "--", color="#666666", lw=1.2, alpha=0.7)

rho, pval = stats.spearmanr(vl, lv)
ax.set_xlabel(r"Landslide volume $V_{land}$ (×10$^4$ m³)")
ax.set_ylabel(r"log$_{10}$($V_0$)")
ax.legend(fontsize=9.5, title="Channel gradient (J)",
          title_fontsize=10.5, loc="upper left",
          framealpha=0.95,
          bbox_to_anchor=(0.0, 0.88), edgecolor="lightgray")

pval_str = "p < 0.001" if pval < 0.001 else f"p = {pval:.4f}"
ax.text(0.97, 0.05, f"Spearman ρ = {rho:.3f}\n{pval_str}",
        transform=ax.transAxes, fontsize=10.5, ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  alpha=0.95, edgecolor="gray", linewidth=0.5))
PL(ax, "(d)")

ensure_arial(axes.flat)
plt.tight_layout(pad=1.8)
SAVE_FIG(fig, "fig2_feature_analysis.jpg")
plt.show()