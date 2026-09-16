"""
Cell 9 ── Fig 7: Feature Ablation
         ★ Panel (b) Mean Δ label at the lower right
"""

comp = comp_AB.copy()
comp = comp.sort_values("R2_A", ascending=False).reset_index(drop=True)
comp["R2_rel_drop"] = np.where(
    comp["R2_A"] > 0,
    (comp["Delta_AB"] / comp["R2_A"]) * 100,
    np.nan)
mean_delta = comp["Delta_AB"].mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ═══════ (a) Dumbbell plot ═══════
ax = axes[0, 0]
n_m = len(comp)
y_pos = np.arange(n_m)
for i, row in comp.iterrows():
    ax.plot([row["R2_B"], row["R2_A"]], [i, i],
            "-", color="#CCCCCC", lw=2.5, zorder=1)
    ax.scatter(row["R2_A"], i, c=MC.get(row["Model"], "#888"),
               marker="o", s=95, zorder=3, edgecolors="white", lw=0.8)
    ax.scatter(row["R2_B"], i, c=MC.get(row["Model"], "#888"),
               marker="s", s=95, zorder=3, edgecolors="white", lw=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(comp["Model"], fontsize=12)
ax.set_xlabel("Mean test-fold R²")
ax.legend(
    [Line2D([0], [0], marker="o", color="gray", ls="", ms=9),
     Line2D([0], [0], marker="s", color="gray", ls="", ms=9)],
    ["Set A (6 features)", "Set B (5 features)"],
    fontsize=12, loc="lower right", framealpha=0.95,
    edgecolor="lightgray")
ax.invert_yaxis()
ax.set_xlim(comp["R2_B"].min()-.06,1.03)
PL(ax, "(a)")

# ═══════ (b) ΔR² bar chart; Mean Δ at the lower right ═══════
ax = axes[0, 1]
comp_b = comp.sort_values("Delta_AB",
                          ascending=True).reset_index(drop=True)
y_pos = np.arange(len(comp_b))
ax.barh(y_pos, comp_b["Delta_AB"],
        color=[MC.get(m, "#AAA") for m in comp_b["Model"]],
        edgecolor="white", height=0.65, alpha=0.85)
ax.axvline(mean_delta, ls=":", color="#0072B2", lw=1.5, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(comp_b["Model"], fontsize=12)
ax.set_xlabel("ΔR² (Set A − Set B)")
for i, row in comp_b.iterrows():
    sign = "+" if row["Delta_AB"] >= 0 else ""
    ax.text(row["Delta_AB"] + 0.003, i,
            f"{sign}{row['Delta_AB']:.4f}", va="center",
            fontsize=11.5, fontweight="bold")
# ★ Move Mean Δ from the upper right to the lower right
ax.text(0.97, 0.05,  # ★ 0.97 → 0.05
        f"Mean Δ = {mean_delta:+.4f}",
        transform=ax.transAxes, color="#0072B2", fontsize=12.5,
        ha="right", va="bottom",  # ★ top → bottom
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#0072B2", lw=0.5))
ax.set_xlim(0,comp_b["Delta_AB"].max()*1.24)
PL(ax, "(b)")

# ═══════ (c) R²(A) versus R²(B) scatter ═══════
ax = axes[1, 0]
all_r2 = pd.concat([comp["R2_A"], comp["R2_B"]])
pad = 0.08
r2_lo = max(all_r2.min() - pad, -0.15)
r2_hi = min(all_r2.max() + pad, 1.05)

label_offsets = {
    "TabPFN": (12, 14),
    "RF": (12, -16),
    "GBR": (-75, 10),
    "XGBoost": (12, -28),
    "AdaBoost": (-80, -10),
    "KNN": (-55, 12),
    "SVR": (12, -8),
    "Ridge": (12, 8),
    "ElasticNet": (12, -12),
    "Lasso": (12, 10),
    "MLP": (12, -14),
}

for _, row in comp.iterrows():
    m = row["Model"]
    ax.scatter(row["R2_A"], row["R2_B"],
               c=MC.get(m, "#888"), s=110,
               edgecolors="white", lw=0.8, zorder=3)
    ox, oy = label_offsets.get(m, (10, 6))
    ax.annotate(m,
                xy=(row["R2_A"], row["R2_B"]),
                xytext={"TabPFN":(.87,.91),"RF":(.95,.75),"GBR":(.60,.88),"XGBoost":(.84,.45),"AdaBoost":(.42,.78),"KNN":(.37,.65),"SVR":(.72,.35),"Ridge":(.23,.34),"Lasso":(.24,.20),"ElasticNet":(.59,.23),"MLP":(.60,.09)}[m], textcoords="data",
                fontsize=11, fontweight="bold",
                color=MC.get(m, "#888"),
                arrowprops=dict(arrowstyle="-",
                                color="#CCCCCC", lw=0.5,
                                shrinkA=0, shrinkB=3))

ax.plot([r2_lo, r2_hi], [r2_lo, r2_hi], "--", color="#999999",
        lw=1.2, alpha=0.6)
ax.set_xlim(r2_lo, 1.10)
ax.set_ylim(r2_lo, r2_hi)
ax.set_xlabel("R², set A (6 features)")
ax.set_ylabel("R², set B (5 features)")
ax.text(0.65, 0.15, "Set A > Set B", transform=ax.transAxes,
        fontsize=13, color="#0072B2", alpha=0.35,
        style="italic", fontweight="bold")
PL(ax, "(c)")

# ═══════ (d) Relative R² decrease in percent ═══════
ax = axes[1, 1]
comp_d = (comp.dropna(subset=["R2_rel_drop"])
          .sort_values("R2_rel_drop", ascending=True)
          .reset_index(drop=True))
y_pos = np.arange(len(comp_d))
ax.barh(y_pos, comp_d["R2_rel_drop"],
        color=[MC.get(m, "#AAA") for m in comp_d["Model"]],
        edgecolor="white", height=0.65, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(comp_d["Model"], fontsize=12)
ax.set_xlabel("Relative R² decrease without deposits (%)")
for i, row in comp_d.iterrows():
    ax.text(row["R2_rel_drop"] + 0.8, i,
            f"{row['R2_rel_drop']:.1f}%", va="center",
            fontsize=11.5, fontweight="bold")
ax.set_xlim(0,comp_d["R2_rel_drop"].max()*1.23)
PL(ax, "(d)")

ensure_arial(axes.flat)
plt.tight_layout(pad=1.8)
SAVE_FIG(fig, "fig7_feature_ablation.jpg")
plt.close("all")
