"""
Cell 7 ── Fig 5: Predicted vs Observed + Residual Diagnostics
         ★ Increase global font sizes by 2 pt to match Fig9 v3.3
"""
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pred_df = pd.read_csv(RES / "predictions_setA_cv5fold.csv")
obs = pred_df["observed"].values
p_tab = pred_df["pred_tabpfn"].values
p_rf = pred_df["pred_rf"].values


def _metrics(yt, yp):
    return (r2_score(yt, yp),
            np.sqrt(mean_squared_error(yt, yp)),
            mean_absolute_error(yt, yp))


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

models_info = [
    (p_tab, "TabPFN", "#D55E00", "#FFF0E8", "#D55E00"),
    (p_rf, "Random Forest", "#0072B2", "#E8F4FF", "#0072B2"),
]

for ci, (pred, name, clr, bg, ec) in enumerate(models_info):
    r2, rmse, mae = _metrics(obs, pred)
    resid = pred - obs

    # ── Top row: predictions versus observations ──
    ax = axes[0, ci]
    ax.scatter(obs, pred, c=clr, s=48, alpha=0.7,
               edgecolors="white", lw=0.5, zorder=3)

    lo_ = min(obs.min(), pred.min()) - 0.2
    hi_ = max(obs.max(), pred.max()) + 0.2
    ax.plot([lo_, hi_], [lo_, hi_], "k-", lw=1.2, label="1:1 line")
    ax.fill_between([lo_, hi_],
                    [lo_ - 0.3, hi_ - 0.3],
                    [lo_ + 0.3, hi_ + 0.3],
                    alpha=0.06, color="#888888", label="±0.3 log unit")
    ax.fill_between([lo_, hi_],
                    [lo_ - 0.15, hi_ - 0.15],
                    [lo_ + 0.15, hi_ + 0.15],
                    alpha=0.10, color="#888888", label="±0.15 log unit")
    ax.set_xlim(lo_, hi_)
    ax.set_ylim(lo_, hi_)
    ax.set_xlabel(r"Observed log₁₀ volume")
    ax.set_ylabel(r"Predicted log₁₀ volume")

    info = (f"{name}\nn = {len(obs)}\n"
            f"R² = {r2:.4f}\n"
            f"RMSE = {rmse:.4f}\n"
            f"MAE = {mae:.4f}")
    ax.text(0.97, 0.03, info, transform=ax.transAxes, fontsize=12,  # ★ 10 → 12
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc=bg, ec=ec, lw=0.8,
                      alpha=0.95))

    ax.legend(fontsize=11, loc="upper left", framealpha=0.95,  # ★ 9 → 11
              edgecolor="lightgray", bbox_to_anchor=(0.0, 0.72))
    PL(ax, f"({'a' if ci == 0 else 'b'})")

    # ── Bottom row: residual diagnostics ──
    ax = axes[1, ci]
    ax.scatter(obs, resid, c=clr, s=48, alpha=0.7,
               edgecolors="white", lw=0.5, zorder=3)
    ax.axhline(0, color="black", lw=1.2)
    ax.axhline(mae, color=clr, ls=":", lw=1.2, alpha=0.6,
               label=f"±MAE = ±{mae:.3f}")
    ax.axhline(-mae, color=clr, ls=":", lw=1.2, alpha=0.6)

    ax.set_xlabel(r"Observed log₁₀ volume")
    ax.set_ylabel("Residual (Predicted − Observed)")
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95,  # ★ 10 → 12
              edgecolor="lightgray")

    res_txt = (f"Mean residual = {resid.mean():.4f}\n"
               f"SD = {resid.std():.4f}")
    ax.text(0.04, 0.05, res_txt, transform=ax.transAxes, fontsize=12,  # ★ 10 → 12
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", lw=0.5, alpha=0.95))
    PL(ax, f"({'c' if ci == 0 else 'd'})")

ensure_arial(axes.flat)
plt.tight_layout(pad=1.8)
SAVE_FIG(fig, "fig5_predicted_vs_observed.jpg")
plt.close("all")
