"""
Cell 10 ── Fig 8: Learning Curves
           ★ Panel (c) label at the lower right
"""

LC_MODELS = ["TabPFN", "RF", "XGBoost", "SVR", "Ridge", "GBR", "MLP"]
LC_STYLES = {
    "TabPFN": dict(c="#D55E00", ls="-", marker="o", lw=2.5),
    "RF": dict(c="#0072B2", ls="--", marker="s", lw=1.8),
    "XGBoost": dict(c="#332288", ls="--", marker="^", lw=1.8),
    "SVR": dict(c="#56B4E9", ls=":", marker="D", lw=1.8),
    "Ridge": dict(c="#999999", ls="--", marker="v", lw=1.5),
    "GBR": dict(c="#009E73", ls="-.", marker="P", lw=1.8),
    "MLP": dict(c="#BBBBBB", ls=":", marker="X", lw=1.5),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ═══════════════════════════════════════════════════════
# (a) & (b) Learning curves: Set A / Set B
# ═══════════════════════════════════════════════════════
CLIP = -0.5

for ci, (lc_sum, label) in enumerate([
    (lc_sum_A, "Feature Set A (6 features)"),
    (lc_sum_B, "Feature Set B (5 features)")]):
    ax = axes[0, ci]
    for m in LC_MODELS:
        sub = lc_sum[lc_sum["Model"] == m].sort_values("n_train")
        if len(sub) == 0:
            continue
        ns = sub["n_train"].values
        mu = sub["R2_mean"].values
        sd = sub["R2_std"].values
        st = LC_STYLES[m]

        mu_c = np.clip(mu, CLIP, None)
        lo = np.clip(mu - sd, CLIP, None)
        hi = mu + sd

        ax.plot(ns, mu_c, color=st["c"], ls=st["ls"],
                marker=st["marker"], ms=5, lw=st["lw"],
                label=m, zorder=3)
        ax.fill_between(ns, lo, hi, color=st["c"], alpha=0.08)

    ax.set_xlabel(r"Training sample size ($n_{train}$)")
    ax.set_ylabel("R² (mean ± s.d.)")
    ax.set_title(label, fontsize=14, fontweight="bold", pad=8)
    ax.set_ylim(-0.55, 1.05)
    ax.legend(fontsize=11, loc="lower right", ncol=2,
              framealpha=0.95, edgecolor="lightgray")
    ax.text(0.03, 0.03, "R² < −0.5 clipped for readability",
            transform=ax.transAxes, fontsize=10.5,
            color="gray", style="italic")
    PL(ax, f"({'a' if ci == 0 else 'b'})")

# ═══════════════════════════════════════════════════════
# (c) Small-sample versus large-sample R² comparison (Set A); panel (c) label at the lower right
# ═══════════════════════════════════════════════════════
ax = axes[1, 0]
n_small = int(lc_sum_A["n_train"].min())
n_large = int(lc_sum_A["n_train"].max())

x = np.arange(len(LC_MODELS))
w = 0.35
vals_s, vals_l = [], []
for m in LC_MODELS:
    sub = lc_sum_A[lc_sum_A["Model"] == m]
    rs = sub.loc[sub["n_train"] == n_small, "R2_mean"]
    rl = sub.loc[sub["n_train"] == n_large, "R2_mean"]
    vals_s.append(float(rs.iloc[0]) if len(rs) > 0 else np.nan)
    vals_l.append(float(rl.iloc[0]) if len(rl) > 0 else np.nan)

bars1 = ax.bar(x - w / 2, vals_s, w, alpha=0.45,
               color=[MC[m] for m in LC_MODELS],
               edgecolor="white",
               label=fr"$n_{{train}}$ = {n_small}")

bars2 = ax.bar(x + w / 2, vals_l, w,
               color=[MC[m] for m in LC_MODELS],
               edgecolor="white",
               label=fr"$n_{{train}}$ = {n_large}")

for bar, val in zip(bars1, vals_s):
    if not np.isnan(val):
        if val >= 0:
            yp = val + 0.03
            va_pos = "bottom"
        else:
            yp = val - 0.03
            va_pos = "top"
        ax.text(bar.get_x() + bar.get_width() / 2, yp,
                f"{val:.4f}", ha="center", va=va_pos,
                fontsize=10, fontweight="bold",
                color="#777777", rotation=90)

for bar, val in zip(bars2, vals_l):
    if not np.isnan(val):
        if val >= 0:
            yp = val + 0.03
            va_pos = "bottom"
        else:
            yp = val - 0.03
            va_pos = "top"
        ax.text(bar.get_x() + bar.get_width() / 2, yp,
                f"{val:.4f}", ha="center", va=va_pos,
                fontsize=10, fontweight="bold",
                color="#333333", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(LC_MODELS, fontsize=12, rotation=30,
                   ha="right")
ax.set_ylabel("R²")
ax.axhline(0, color="black", lw=0.5)
ax.legend(fontsize=12, framealpha=0.95, edgecolor="lightgray",
          loc="lower left")

all_vals = [v for v in vals_s + vals_l if not np.isnan(v)]
y_lo = min(min(all_vals) - 0.25, -0.15)
y_hi = max(all_vals) + 0.18
ax.set_ylim(y_lo, y_hi)

# ★ Move the panel (c) label from the upper left to the lower right
ax.text(0.98, 0.05, "(c)", transform=ax.transAxes,
        fontsize=17, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  alpha=0.98, edgecolor="lightgray"),
        verticalalignment="bottom", horizontalalignment="right", zorder=10)

# ═══════════════════════════════════════════════════════
# (d) TabPFN performance margin over the best baseline
# ═══════════════════════════════════════════════════════
ax = axes[1, 1]

all_deltas = []
for si, (lc_sum, ls_, clr, mk, lab) in enumerate([
    (lc_sum_A, "-", "#D55E00", "o", "Set A"),
    (lc_sum_B, "--", "#0072B2", "s", "Set B")]):

    sizes = sorted(lc_sum["n_train"].unique())
    deltas = []
    for n in sizes:
        sub = lc_sum[lc_sum["n_train"] == n]
        tab = sub[sub["Model"] == "TabPFN"]
        base = sub[sub["Model"] != "TabPFN"]
        if len(tab) > 0 and len(base) > 0:
            d = tab["R2_mean"].iloc[0] - base["R2_mean"].max()
            deltas.append(d)
        else:
            deltas.append(np.nan)
    all_deltas.extend([d for d in deltas if not np.isnan(d)])
    ax.plot(sizes, deltas, color=clr, ls=ls_, marker=mk,
            ms=7, lw=2, label=lab, zorder=3)

ax.axhline(0, color="black", lw=0.8)

y_lower = min(min(all_deltas) - 0.02, -0.12)
y_upper = max(all_deltas) + 0.02
ax.set_ylim(y_lower, y_upper)

if y_lower < 0:
    ax.axhspan(y_lower, 0, alpha=0.04, color="#D55E00", zorder=0)

ax.set_xlabel(r"Training sample size ($n_{train}$)")
ax.set_ylabel("ΔR² (TabPFN − best baseline)")
ax.legend(fontsize=12, framealpha=0.95, edgecolor="lightgray",
          loc="upper left", bbox_to_anchor=(0.0, 0.95))
ax.text(0.97, 0.03, "Below 0 → baseline outperforms TabPFN",
        transform=ax.transAxes, fontsize=11, ha="right",
        color="gray", style="italic")
PL(ax, "(d)")

# ═══════════════════════════════════════════════════════
# Finalize the figure
# ═══════════════════════════════════════════════════════
ensure_arial(axes.flat)
plt.tight_layout(pad=1.8)
SAVE_FIG(fig, "fig8_learning_curves.jpg")
plt.show()