"""
Cell 4 ── 定义可复用的模型对比函数（供 Fig 4 和 Fig 6 调用）
         ★ 新增 panel_a_info_ha 参数，控制信息框水平对齐
"""


def plot_model_comparison(r2_df, wilcoxon_df, set_label, fig_name,
                          panel_a_info_pos=(0.98, 0.97),
                          panel_a_info_va="top",
                          panel_a_info_ha="right",  # ★ 新增参数
                          panel_c_legend_loc="lower right",
                          panel_d_legend_loc="upper left"):
    """
    4-panel model comparison figure (CNS-level).
    """
    # ── 准备每折数据 ──
    model_r2 = {}
    model_r2["TabPFN"] = r2_df["TabPFN"].dropna().values
    for m in BASELINES:
        col = f"{m}_bo"
        if col in r2_df.columns:
            model_r2[m] = r2_df[col].dropna().values

    order = sorted(model_r2, key=lambda x: np.median(model_r2[x]),
                   reverse=True)
    tabpfn_mean = np.mean(model_r2["TabPFN"])

    baselines_only = {k: v for k, v in model_r2.items() if k != "TabPFN"}
    best_bl = max(baselines_only, key=lambda x: np.median(baselines_only[x]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ═══════════════════════════════════════════════════
    # (a) Boxplot + jittered strip
    # ═══════════════════════════════════════════════════
    ax = axes[0, 0]
    box_data = [model_r2[m] for m in order]
    pos = np.arange(len(order))

    bp = ax.boxplot(box_data, positions=pos, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=0.8, color="#555555"),
                    capprops=dict(lw=0.8, color="#555555"))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(MC.get(m, "#AAA"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for i, m in enumerate(order):
        jitter = rng.uniform(-0.15, 0.15, len(model_r2[m]))
        ax.scatter(pos[i] + jitter, model_r2[m],
                   c=MC.get(m, "#AAA"), s=10, alpha=0.3,
                   edgecolors="none", zorder=2)

    ax.axhline(tabpfn_mean, ls="--", color="#D55E00", lw=1.2, alpha=0.7)
    ax.set_xticks(pos)
    ax.set_xticklabels(order, fontsize=12, rotation=35,
                       ha="right")
    ax.set_ylabel("R² (repeated nested CV)")

    # ★ 使用参数化位置 + ha
    ax.set_title(f"{set_label}: TabPFN mean R² = {tabpfn_mean:.4f}", fontsize=13, pad=12)

    ax.text(0.02, 0.05, "(a)", transform=ax.transAxes,
            fontsize=17, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.98, edgecolor="lightgray"),
            verticalalignment="bottom", horizontalalignment="left", zorder=10)

    # ═══════════════════════════════════════════════════
    # (b) ΔR² 水平条形图
    # ═══════════════════════════════════════════════════
    ax = axes[0, 1]
    wdf = wilcoxon_df.sort_values("mean_dR2",
                                  ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(wdf))
    ax.barh(y_pos, wdf["mean_dR2"],
            color=[MC.get(m, "#AAA") for m in wdf["Model"]],
            edgecolor="white", height=0.65, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wdf["Model"], fontsize=12)
    ax.set_xlabel("ΔR² (TabPFN − Baseline)")
    ax.axvline(0, color="black", lw=0.5)
    for i, row in wdf.iterrows():
        sign = "+" if row["mean_dR2"] >= 0 else ""
        offset = 0.003 if row["mean_dR2"] >= 0 else -0.003
        ha = "left" if row["mean_dR2"] >= 0 else "right"
        ax.text(row["mean_dR2"] + offset, i,
                f"{sign}{row['mean_dR2']:.4f}",
                va="center", ha=ha, fontsize=11.5,
                fontweight="bold")

    ax.set_xlim(min(0, wdf["mean_dR2"].min()) - 0.01, wdf["mean_dR2"].max() * 1.27)
    PL(ax, "(b)")

    # ═══════════════════════════════════════════════════
    # (c) 效应量 + 显著性
    # ═══════════════════════════════════════════════════
    ax = axes[1, 0]
    wdf_c = wilcoxon_df.sort_values("effect_r",
                                    ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(wdf_c))
    bar_colors = [SC.get(str(row["sig"]).strip(), "#CCCCCC")
                  for _, row in wdf_c.iterrows()]
    ax.barh(y_pos, wdf_c["effect_r"], color=bar_colors,
            edgecolor="white", height=0.65, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wdf_c["Model"], fontsize=12)
    ax.set_xlabel("Effect size (rank-biserial r)")
    for i, row in wdf_c.iterrows():
        x_txt = row["effect_r"] + 0.025
        ax.text(x_txt, i, str(row["sig"]).strip(),
                va="center", fontsize=12.5,
                fontweight="bold", color="#333333")

    ax.set_title("Ten paired repeat means; Holm correction", fontsize=12, pad=12)
    ax.set_xlim(0, 1.13)
    PL(ax, "(c)")

    # ═══════════════════════════════════════════════════
    # (d) 每折散点：TabPFN vs 最优 baseline
    # ═══════════════════════════════════════════════════
    ax = axes[1, 1]
    ref = model_r2["TabPFN"]
    other = model_r2[best_bl]
    n_min = min(len(ref), len(other))
    ref_c, other_c = ref[:n_min], other[:n_min]

    wins_t = int(np.sum(ref_c > other_c))
    wins_b = int(np.sum(other_c > ref_c))
    mask_win = ref_c > other_c

    ax.scatter(other_c[mask_win], ref_c[mask_win],
               c="#D55E00", s=38, alpha=0.65,
               edgecolors="white", lw=0.4, zorder=3,
               label=f"TabPFN wins (n = {wins_t})")
    ax.scatter(other_c[~mask_win], ref_c[~mask_win],
               c=MC[best_bl], s=38, alpha=0.65,
               edgecolors="white", lw=0.4, zorder=3,
               label=f"{best_bl} wins (n = {wins_b})")

    lo = min(ref_c.min(), other_c.min()) - 0.05
    hi = max(ref_c.max(), other_c.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color="#666666", lw=1.2,
            alpha=0.6, label="Identity line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{best_bl} R² (per fold)")
    ax.set_ylabel("TabPFN R² (per fold)")

    ax.legend(fontsize=11.5, loc="upper left",
              framealpha=0.95, edgecolor="lightgray")

    pct = wins_t / n_min * 100
    ax.text(0.97, 0.03,
            f"Higher TabPFN R² in {wins_t}/{n_min} folds ({pct:.0f}%)",
            transform=ax.transAxes, fontsize=12, ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", lw=0.5))

    ax.text(0.02, 0.05, "(d)", transform=ax.transAxes,
            fontsize=17, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.98, edgecolor="lightgray"),
            verticalalignment="bottom", horizontalalignment="left", zorder=10)

    ensure_arial(axes.flat)
    plt.tight_layout(pad=1.8)
    SAVE_FIG(fig, fig_name)
    plt.close("all")


print("✅ plot_model_comparison() 已定义（新增 panel_a_info_ha）。")
