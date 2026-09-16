# =============================================================================
# Part 2 — Optimized SHAP visualisation (900-dpi JPG only)
# =============================================================================

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess

    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / 'rerun_current_figures' / 'inputs'

FIG_DIR = DATA_DIR / 'SHAP_Analysis_Figures_Optimized_900dpi_JPG'
CACHE_FILE = FIG_DIR / '_cache' / 'all_results_tabpfn_exact_shap.npz'

DPI = 900
FMT = 'jpg'
SEED = 42

assert CACHE_FILE.exists(), f'Cache file not found:\n{CACHE_FILE}\nPlease run Part 1 first.'

# Remove old figures with the same names to avoid confusion
for old_name in [
    'Fig10_Global_SHAP_Attribution.jpg',
    'Fig11_Local_and_Scale_Dependent_SHAP.jpg'
]:
    old_file = FIG_DIR / old_name
    if old_file.exists():
        old_file.unlink()

# -----------------------------------------------------------------------------
# 1. Matplotlib settings — robust font fallback, no missing glyph issues
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'axes.unicode_minus': False,

    'font.size': 13,
    'axes.labelsize': 13.5,
    'axes.titlesize': 14.5,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5,
    'legend.fontsize': 10.5,

    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.5,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,

    'savefig.dpi': DPI,
    'savefig.format': FMT,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.18,
})

# -----------------------------------------------------------------------------
# 2. Load cached real-data SHAP results
# -----------------------------------------------------------------------------
print('=' * 78)
print('Part 2 — Optimized SHAP visualisation (900-dpi JPG only)')
print('=' * 78)
print(f'Cache file   : {CACHE_FILE}')
print(f'Output folder: {FIG_DIR}')

cache = np.load(CACHE_FILE, allow_pickle=False)

shap_values = cache['shap_values']
base_values = cache['base_values']
base_value_mean = float(cache['base_value_mean'])
X = cache['X']
y = cache['y']  # original V0, ×10^4 m^3
y_log = cache['y_log']
y_pred = cache['y_pred']  # full-data model prediction in log10(V0)
sample_id = cache['sample_id']
FEATURE_COLS = cache['feature_cols'].tolist()
FEAT_LABELS = cache['feature_labels'].tolist()

cache.close()

n_samples, n_features = X.shape

print(f'Samples  : {n_samples}')
print(f'Features : {n_features}')
print(f'Mean baseline E[f(x)] : {base_value_mean:.4f} [log10(V0)]')
print(f'Base value SD         : {np.std(base_values):.3e}')

# -----------------------------------------------------------------------------
# 3. Constants, labels, and colours
# -----------------------------------------------------------------------------
SYMUNIT = {
    'Catchment area': (r'$A$', r'km$^2$'),
    'Relative relief': (r'$H$', r'm'),
    'Channel length': (r'$L$', r'km'),
    'Fault distance': (r'$D_f$', r'km'),
    'Channel gradient': (r'$J$', '‰'),
    'Landslide volume': (r'$V_L$', r'$\times 10^4$ m$^3$'),
}
WFALL_SYM = [r'$A$', r'$H$', r'$L$', r'$D_f$', r'$J$', r'$V_L$']

BAR_COLORS = ['#8E0000', '#B71C1C', '#C62828', '#D84315', '#EF6C00', '#FFB74D']
POS_COL = '#C62828'
NEG_COL = '#1565C0'
MAG_CLR = ['#2E7D32', '#F9A825', '#B71C1C']
MAG_NAMES = ['Small', 'Medium', 'Large']

DISPLAY_NAME_MAP = {
    'Catchment area': 'Catchment area',
    'Relative relief': 'Relative relief',
    'Channel length': 'Channel length',
    'Fault distance': 'Fault distance',
    'Channel gradient': 'Channel gradient',
    'Landslide volume': 'Landslide volume',
}

# -----------------------------------------------------------------------------
# 4. Core summaries
# -----------------------------------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
rank_idx = np.argsort(mean_abs_shap)[::-1]
SORTED_NAMES = [DISPLAY_NAME_MAP[FEAT_LABELS[i]] for i in rank_idx]
fname2idx = {name: i for i, name in enumerate(FEAT_LABELS)}

# Global bootstrap CI for mean |SHAP|
rng_global = np.random.default_rng(SEED)
B_BOOT_GLOBAL = 2000
boots_global = np.empty((B_BOOT_GLOBAL, n_features), dtype=np.float64)

for b in range(B_BOOT_GLOBAL):
    idx_b = rng_global.choice(n_samples, n_samples, replace=True)
    boots_global[b] = np.abs(shap_values[idx_b]).mean(axis=0)

ci_lo_global = np.percentile(boots_global, 2.5, axis=0)
ci_hi_global = np.percentile(boots_global, 97.5, axis=0)

# Equal-size tertiles by ordered observed V0 (strictly 20/20/20 for n=60)
order_obs = np.argsort(y)
group_chunks = np.array_split(order_obs, 3)
mag_group = np.empty(n_samples, dtype=int)
for gi, chunk in enumerate(group_chunks):
    mag_group[chunk] = gi

# Group-level bootstrap CI for mean |SHAP|
rng_group = np.random.default_rng(SEED + 1)
B_BOOT_GROUP = 1500
group_stats = []

for gi, chunk in enumerate(group_chunks):
    group_mean = np.mean(np.abs(shap_values[chunk]), axis=0)
    boots_group = np.empty((B_BOOT_GROUP, n_features), dtype=np.float64)

    for b in range(B_BOOT_GROUP):
        idx_b = rng_group.choice(chunk, size=len(chunk), replace=True)
        boots_group[b] = np.abs(shap_values[idx_b]).mean(axis=0)

    group_stats.append({
        'indices': chunk,
        'n': len(chunk),
        'mean_abs': group_mean,
        'ci_lo': np.percentile(boots_group, 2.5, axis=0),
        'ci_hi': np.percentile(boots_group, 97.5, axis=0),
        'median_v0': float(np.median(y[chunk])),
        'min_v0': float(np.min(y[chunk])),
        'max_v0': float(np.max(y[chunk])),
    })


# Representative small/large catchments: closest to group median observed V0
def representative_index(chunk, obs_y):
    chunk = np.asarray(chunk, dtype=int)
    med = np.median(obs_y[chunk])
    return int(chunk[np.argmin(np.abs(obs_y[chunk] - med))])


small_rep_idx = representative_index(group_chunks[0], y)
large_rep_idx = representative_index(group_chunks[2], y)

print('\nGroup summary (equal-size tertiles by observed V0):')
for gi, gs in enumerate(group_stats):
    print(
        f"  {MAG_NAMES[gi]:<6s} | n = {gs['n']:>2d} | "
        f"median V0 = {gs['median_v0']:.2f} | "
        f"range = [{gs['min_v0']:.2f}, {gs['max_v0']:.2f}]"
    )

print('\nRepresentative samples for true waterfall plots:')
print(f'  Small representative sample ID : {sample_id[small_rep_idx]} | observed V0 = {y[small_rep_idx]:.2f}')
print(f'  Large representative sample ID : {sample_id[large_rep_idx]} | observed V0 = {y[large_rep_idx]:.2f}')

vi_VL = fname2idx['Landslide volume']
vi_A = fname2idx['Catchment area']


# -----------------------------------------------------------------------------
# 5. Helper functions
# -----------------------------------------------------------------------------
def _fmt_num(v):
    v = float(v)
    if abs(v) >= 1000:
        return f'{v:.0f}'
    if abs(v) >= 100:
        return f'{v:.1f}'
    if abs(v) >= 10:
        return f'{v:.1f}'
    if abs(v) >= 1:
        return f'{v:.2f}'
    return f'{v:.3f}'


def _panel_label(ax, tag, loc='upper-left', fs=15.5, pad=0.22):
    box = dict(
        boxstyle=f'round,pad={pad}',
        facecolor='white',
        edgecolor='#BDBDBD',
        alpha=0.98,
        linewidth=0.8
    )
    presets = {
        'upper-left': (0.02, 0.98, 'left', 'top'),
        'upper-right': (0.98, 0.98, 'right', 'top'),
        'lower-left': (0.02, 0.03, 'left', 'bottom'),
        'lower-right': (0.98, 0.03, 'right', 'bottom'),
    }
    x, y_, ha, va = presets[loc]
    ax.text(
        x, y_, tag,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=fs,
        fontweight='bold',
        bbox=box,
        zorder=200
    )


def _save(fig, stem):
    out_path = FIG_DIR / f'{stem}.{FMT}'
    try:
        fig.savefig(
            out_path,
            dpi=DPI,
            format=FMT,
            bbox_inches='tight',
            pad_inches=0.18,
            facecolor='white',
            edgecolor='none',
            pil_kwargs={'quality': 95, 'subsampling': 0, 'optimize': True}
        )
    except TypeError:
        fig.savefig(
            out_path,
            dpi=DPI,
            format=FMT,
            bbox_inches='tight',
            pad_inches=0.18,
            facecolor='white',
            edgecolor='none',
            pil_kwargs={'quality': 95}
        )
    plt.close(fig)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'  Saved: {out_path.name}  ({size_mb:.2f} MB)')


def _restore_all_spines(ax):
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(True)


def _beeswarm_offsets(vals, nbins=30, width=0.42):
    vals = np.asarray(vals, dtype=np.float64)
    offsets = np.zeros(len(vals), dtype=np.float64)
    spread = np.ptp(vals)

    if spread < 0.08:
        width = max(width, 0.54)
        nbins = max(nbins, 40)
    elif spread < 0.20:
        width = max(width, 0.48)

    edges = np.linspace(vals.min() - 1e-12, vals.max() + 1e-12, nbins + 1)
    bins = np.clip(np.digitize(vals, edges) - 1, 0, nbins - 1)

    for b in range(nbins):
        mask = bins == b
        count = np.sum(mask)
        if count > 1:
            offsets[mask] = np.linspace(-width / 2, width / 2, count)

    return offsets


def _smooth_xy(x, y_arr, frac=0.35):
    mask = np.isfinite(x) & np.isfinite(y_arr)
    x0 = np.asarray(x[mask], dtype=np.float64)
    y0 = np.asarray(y_arr[mask], dtype=np.float64)

    order = np.argsort(x0)
    x0 = x0[order]
    y0 = y0[order]

    if len(x0) < 5:
        return x0, y0

    if HAS_LOWESS:
        res = _sm_lowess(y0, x0, frac=frac, return_sorted=True)
        return res[:, 0], res[:, 1]

    win = max(5, len(x0) // 6)
    kernel = np.ones(win) / win
    y_sm = np.convolve(y0, kernel, mode='same')
    return x0, y_sm


def _smooth_ci_band(ax, x, y_arr, frac=0.35, n_boot=250, alpha=0.12, color='#424242', seed=42):
    rng = np.random.default_rng(seed)
    x_ref, _ = _smooth_xy(x, y_arr, frac=frac)
    if len(x_ref) < 5:
        return

    curves = []
    n = len(x)

    for _ in range(n_boot):
        idx_b = rng.choice(n, size=n, replace=True)
        xb = np.asarray(x[idx_b], dtype=np.float64)
        yb = np.asarray(y_arr[idx_b], dtype=np.float64)

        try:
            xs, ys = _smooth_xy(xb, yb, frac=frac)
            if len(xs) >= 5:
                curves.append(np.interp(x_ref, xs, ys))
        except Exception:
            continue

    if len(curves) < 30:
        return

    curves = np.asarray(curves, dtype=np.float64)
    lo = np.percentile(curves, 2.5, axis=0)
    hi = np.percentile(curves, 97.5, axis=0)

    ax.fill_between(x_ref, lo, hi, color=color, alpha=alpha, zorder=2)


def _true_waterfall(ax, idx, tag, group_label):
    """
    True cumulative SHAP waterfall:
    starts from sample-specific baseline and accumulates feature contributions.
    """
    sv = np.asarray(shap_values[idx], dtype=np.float64)
    xv = np.asarray(X[idx], dtype=np.float64)
    sid = int(sample_id[idx])

    base_i = float(base_values[idx])
    pred_log = float(y_pred[idx])
    pred_raw = float(10 ** pred_log)
    obs_raw = float(y[idx])

    order = np.argsort(np.abs(sv))[::-1]
    ordered_sv = sv[order]

    y_pos = np.arange(len(order), 0, -1)
    starts = np.zeros(len(order), dtype=np.float64)
    ends = np.zeros(len(order), dtype=np.float64)

    running = base_i
    for k, j in enumerate(order):
        starts[k] = running
        running = running + sv[j]
        ends[k] = running

    # The prediction should equal the final cumulative value; plot that cumulative value for consistent numerical display
    final_cum = float(ends[-1])

    all_x = np.concatenate([[base_i], starts, ends, [pred_log]])
    x_span = max(all_x.max() - all_x.min(), 0.25)
    x_pad = 0.22 * x_span
    x_min = all_x.min() - x_pad
    x_max = all_x.max() + x_pad
    ax.set_xlim(x_min, x_max)

    # Background bands
    for i, yy in enumerate(y_pos):
        if i % 2 == 0:
            ax.axhspan(yy - 0.42, yy + 0.42, color='#F7F7F7', zorder=0)

    # Baseline and prediction lines
    ax.axvline(base_i, color='#616161', lw=0.9, ls='--', zorder=1)
    ax.axvline(final_cum, color='#424242', lw=0.9, ls=':', zorder=1)

    label_threshold = 0.13 * (x_max - x_min)

    for k, j in enumerate(order):
        s = starts[k]
        e = ends[k]
        delta = e - s
        left = min(s, e)
        width = abs(delta)
        color = POS_COL if delta >= 0 else NEG_COL

        ax.barh(
            y_pos[k], width, left=left, height=0.56,
            color=color, edgecolor='white', lw=0.7, alpha=0.92, zorder=3
        )

        # Connector lines
        if k < len(order) - 1:
            ax.plot([e, e], [y_pos[k] - 0.28, y_pos[k + 1] + 0.28],
                    color='#9E9E9E', lw=0.85, zorder=2)

        # Increment annotations
        delta_txt = f'{delta:+.3f}'
        if width >= label_threshold:
            ax.text(
                (s + e) / 2, y_pos[k], delta_txt,
                ha='center', va='center',
                fontsize=10.6, fontweight='bold', color='white',
                zorder=5,
                path_effects=[pe.withStroke(linewidth=1.0, foreground='none')]
            )
        else:
            pad = 0.02 * (x_max - x_min)
            if delta >= 0:
                ax.text(
                    e + pad, y_pos[k], delta_txt,
                    ha='left', va='center',
                    fontsize=10.4, fontweight='bold', color='#444444', zorder=5
                )
            else:
                ax.text(
                    e - pad, y_pos[k], delta_txt,
                    ha='right', va='center',
                    fontsize=10.4, fontweight='bold', color='#444444', zorder=5
                )

    # Y-axis labels: variable symbols and observed values
    y_labels = []
    for j in order:
        sym, unit = SYMUNIT[FEAT_LABELS[j]]
        val_txt = _fmt_num(xv[j])
        y_labels.append(f'{sym} = {val_txt}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11.5)
    ax.set_ylim(0.35, len(order) + 0.80)

    ax.set_xlabel(r'Model output, log$_{10}(V_0)$', fontsize=13)
    ax.set_axisbelow(True)

    # Baseline information
    ax.text(
        0.02, 0.03,
        f'Baseline $E[f(x)]$ = {base_i:.2f}',
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=10.8, color='#424242',
        bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#BDBDBD', lw=0.7, alpha=0.96),
        zorder=50
    )

    # Sample information
    info_text = (
        f'{group_label} representative catchment\n'
        f'Sample ID = {sid}\n'
        f'Observed $V_0$ = {_fmt_num(obs_raw)} ×10$^4$ m$^3$\n'
        f'Predicted $V_0$ = {_fmt_num(pred_raw)} ×10$^4$ m$^3$'
    )
    info_fc = '#E3F2FD' if pred_log < base_i else '#FFF3E0'
    info_ec = NEG_COL if pred_log < base_i else '#E65100'
    info_tc = NEG_COL if pred_log < base_i else POS_COL

    ax.text(
        0.98, 0.97, info_text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=11.2, fontweight='bold', color=info_tc,
        bbox=dict(boxstyle='round,pad=0.36', fc=info_fc, ec=info_ec, lw=1.1, alpha=0.97),
        zorder=60
    )

    _panel_label(ax, tag)


# -----------------------------------------------------------------------------
# 6. Figure 10 — Global SHAP attribution
# -----------------------------------------------------------------------------
print('\nGenerating Figure 10 ...')

fig1 = plt.figure(figsize=(15.8, 13.0), facecolor='white')
gs1 = GridSpec(
    2, 2, figure=fig1,
    left=0.14, right=0.93, bottom=0.06, top=0.97,
    wspace=0.52, hspace=0.18
)

# (a) Mean absolute SHAP values with bootstrap CI
ax_a = fig1.add_subplot(gs1[0, 0])

vals_sorted = mean_abs_shap[rank_idx]
ci_lo_sorted = ci_lo_global[rank_idx]
ci_hi_sorted = ci_hi_global[rank_idx]
indiv_pct = vals_sorted / vals_sorted.sum() * 100.0
cum_pct = np.cumsum(indiv_pct)

ypos = np.arange(n_features)[::-1]

for i in range(n_features):
    if i % 2 == 0:
        ax_a.axhspan(ypos[i] - 0.38, ypos[i] + 0.38, color='#F5F5F5', zorder=0)

for i in range(n_features):
    err_lo = max(vals_sorted[i] - ci_lo_sorted[i], 0.0)
    err_hi = max(ci_hi_sorted[i] - vals_sorted[i], 0.0)

    ax_a.barh(
        ypos[i], vals_sorted[i], height=0.54,
        color=BAR_COLORS[i], edgecolor='none', zorder=3,
        xerr=np.array([[err_lo], [err_hi]]),
        error_kw=dict(ecolor='#424242', lw=1.0, capsize=4, capthick=1.0)
    )

label_x_max = max(ci_hi_sorted.max(), vals_sorted.max()) + 0.72 * vals_sorted.max()
for i in range(n_features):
    tx = ci_hi_sorted[i] + 0.05 * vals_sorted.max()
    ax_a.text(
        tx, ypos[i],
        f'{vals_sorted[i]:.3f} ({indiv_pct[i]:.1f}%)',
        va='center', ha='left',
        fontsize=10.9, fontweight='bold', color='#333333', zorder=5
    )

ax_a.text(
    0.97, 0.06,
    f'Top-2 cumulative contribution = {cum_pct[1]:.0f}%\n'
    f'Top-4 cumulative contribution = {cum_pct[3]:.0f}%',
    transform=ax_a.transAxes,
    fontsize=10.8, ha='right', va='bottom',
    color='#1565C0', fontstyle='italic', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.34', fc='#E3F2FD', ec='#90CAF9', lw=0.8, alpha=0.96),
    zorder=20
)

ax_a.set_yticks(ypos)
ax_a.set_yticklabels(SORTED_NAMES, fontsize=12.2)
ax_a.set_xlim(0, label_x_max)
ax_a.set_xlabel(r'Mean absolute SHAP value (contribution on log$_{10}(V_0)$ scale)', fontsize=13)
ax_a.grid(axis='x')
ax_a.grid(axis='y', visible=False)
_panel_label(ax_a, '(a)')

# (b) Beeswarm plot
ax_b = fig1.add_subplot(gs1[0, 1])

X_norm = np.zeros_like(X, dtype=np.float64)
for j in range(n_features):
    col_min, col_max = X[:, j].min(), X[:, j].max()
    X_norm[:, j] = (X[:, j] - col_min) / (col_max - col_min + 1e-12)

for rk in range(n_features):
    y_center = n_features - 1 - rk
    if rk % 2 == 0:
        ax_b.axhspan(y_center - 0.45, y_center + 0.45, color='#F5F5F5', zorder=0)

for rk, feat_idx in enumerate(rank_idx):
    y_center = n_features - 1 - rk
    sv = shap_values[:, feat_idx]
    offsets = _beeswarm_offsets(sv, nbins=30, width=0.42)
    order = np.argsort(X_norm[:, feat_idx])

    sc = ax_b.scatter(
        sv[order], y_center + offsets[order],
        c=X_norm[order, feat_idx], cmap='RdBu_r',
        s=34, alpha=0.84, edgecolors='#888888', linewidths=0.15,
        vmin=0, vmax=1, zorder=3
    )
    ax_b.plot(
        np.median(sv), y_center, '|', color='black', ms=14, mew=1.8, zorder=6,
        path_effects=[pe.withStroke(linewidth=3.0, foreground='white')]
    )

ax_b.axvline(0, color='grey', lw=0.6, ls='--', zorder=1)
ax_b.set_yticks(range(n_features))
ax_b.set_yticklabels(SORTED_NAMES[::-1], fontsize=12.2)
ax_b.set_xlabel(r'SHAP value (contribution on log$_{10}(V_0)$ scale)', fontsize=13)
ax_b.grid(axis='x')
ax_b.grid(axis='y', visible=False)

sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=Normalize(0, 1))
sm.set_array([])
cb_b = fig1.colorbar(sm, ax=ax_b, fraction=0.035, pad=0.04, aspect=25)
cb_b.set_ticks([0, 0.5, 1.0])
cb_b.set_ticklabels(['Low', 'Mid', 'High'])
cb_b.set_label('Normalised feature value', fontsize=12, rotation=90, labelpad=14)

_panel_label(ax_b, '(b)', loc='lower-left')

# (c) Dependence plot for landslide volume with catchment area as interaction variable
ax_c = fig1.add_subplot(gs1[1, 0])

x_vl = X[:, vi_VL]
sv_vl = shap_values[:, vi_VL]
c_a = X[:, vi_A]

sc_c = ax_c.scatter(
    x_vl, sv_vl, c=c_a, cmap='RdYlBu_r',
    s=58, alpha=0.84, edgecolors='grey', lw=0.3, zorder=3
)

x_sm, y_sm = _smooth_xy(x_vl, sv_vl, frac=0.35)
ax_c.plot(x_sm, y_sm, color='#424242', lw=2.2, alpha=0.88, zorder=4)
_smooth_ci_band(ax_c, x_vl, sv_vl, frac=0.35, n_boot=250, seed=SEED)

ax_c.axhline(0, color='grey', lw=0.6, ls='--', zorder=1)
ax_c.set_xlabel(r'Landslide volume, $V_L$ ($\times 10^4$ m$^3$)', fontsize=13)
ax_c.set_ylabel(r'SHAP value (contribution on log$_{10}(V_0)$ scale)', fontsize=13)

cbar_c = fig1.colorbar(sc_c, ax=ax_c, fraction=0.04, pad=0.03, aspect=22)
cbar_c.set_label(r'Catchment area, $A$ (km$^2$)', fontsize=12, rotation=90, labelpad=14)

r_val, p_val = pearsonr(x_vl, sv_vl)
ax_c.text(
    0.03, 0.97,
    f'Pearson $r$ = {r_val:.3f}\n$p$ = {p_val:.1e}',
    transform=ax_c.transAxes,
    fontsize=10.8, ha='left', va='top',
    bbox=dict(boxstyle='round,pad=0.30', fc='#F5F5F5', ec='#BDBDBD', lw=0.6, alpha=0.95),
    zorder=20
)
_panel_label(ax_c, '(c)', loc='lower-right')

# (d) Dual correlation matrix: upper triangle = attribution correlation; lower triangle = feature correlation
ax_d = fig1.add_subplot(gs1[1, 1])
_restore_all_spines(ax_d)
ax_d.grid(False)

X_ranked = X[:, rank_idx]
S_ranked = shap_values[:, rank_idx]

feat_corr = np.nan_to_num(np.corrcoef(X_ranked.T), nan=0.0)
shap_corr = np.nan_to_num(np.corrcoef(S_ranked.T), nan=0.0)

display_mat = np.full((n_features, n_features), np.nan, dtype=np.float64)
for i in range(n_features):
    for j in range(n_features):
        if i < j:
            display_mat[i, j] = shap_corr[i, j]
        elif i > j:
            display_mat[i, j] = feat_corr[i, j]

masked_display = np.ma.masked_invalid(display_mat)
cmap_corr = plt.cm.RdBu_r.copy()
cmap_corr.set_bad('#F0F0F0')

im_d = ax_d.imshow(masked_display, cmap=cmap_corr, vmin=-1, vmax=1, aspect='equal', interpolation='nearest')

for i in range(n_features):
    for j in range(n_features):
        if i == j:
            sym, _ = SYMUNIT[FEAT_LABELS[rank_idx[i]]]
            ax_d.text(
                j, i, sym,
                ha='center', va='center',
                fontsize=12.0, fontweight='bold', color='#333333',
                bbox=dict(boxstyle='round,pad=0.14', fc='#EEEEEE', ec='#BDBDBD', lw=0.5)
            )
        else:
            v = display_mat[i, j]
            tc = 'white' if abs(v) > 0.55 else '#333333'
            ax_d.text(
                j, i, f'{v:.2f}',
                ha='center', va='center',
                fontsize=10.8, fontweight='bold', color=tc
            )

ax_d.plot([-0.5, n_features - 0.5], [-0.5, n_features - 0.5],
          color='#757575', lw=1.4, alpha=0.65, zorder=10)

ax_d.set_xticks(range(n_features))
ax_d.set_xticklabels(SORTED_NAMES, rotation=45, ha='right', fontsize=10.8)
ax_d.set_yticks(range(n_features))
ax_d.set_yticklabels(SORTED_NAMES, fontsize=11.2)

cb_d = fig1.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04, aspect=22)
cb_d.set_label('Pearson correlation coefficient', fontsize=12, rotation=90, labelpad=14)
cb_d.outline.set_linewidth(0.5)

# Place explanatory text outside the matrix to keep the values visible
ax_d.text(
    0.99, 1.05, 'Upper triangle: attribution correlation',
    transform=ax_d.transAxes, ha='right', va='bottom',
    fontsize=10.5, color='#B71C1C', fontstyle='italic', fontweight='bold'
)
ax_d.text(
    0.01, -0.14, 'Lower triangle: feature correlation',
    transform=ax_d.transAxes, ha='left', va='top',
    fontsize=10.5, color='#1565C0', fontstyle='italic', fontweight='bold'
)

_panel_label(ax_d, '(d)')

_save(fig1, 'Fig10_Global_SHAP_Attribution')

# -----------------------------------------------------------------------------
# 7. Figure 11 — Local and scale-dependent SHAP attribution
# -----------------------------------------------------------------------------
print('Generating Figure 11 ...')

fig2 = plt.figure(figsize=(16.2, 13.4), facecolor='white')
gs2 = GridSpec(
    2, 2, figure=fig2,
    left=0.13, right=0.94, bottom=0.06, top=0.97,
    wspace=0.52, hspace=0.18
)

# (a) Representative small-volume catchment — true waterfall
ax2a = fig2.add_subplot(gs2[0, 0])
_true_waterfall(ax2a, small_rep_idx, '(a)', 'Small-volume')

# (b) Representative large-volume catchment — true waterfall
ax2b = fig2.add_subplot(gs2[0, 1])
_true_waterfall(ax2b, large_rep_idx, '(b)', 'Large-volume')

# (c) Grouped mean |SHAP| by debris-flow magnitude class
ax2c = fig2.add_subplot(gs2[1, 0])

bar_h = 0.22
y_ctr = np.arange(n_features)[::-1]
offsets = np.array([-bar_h - 0.03, 0.0, bar_h + 0.03])

for i in range(n_features):
    if i % 2 == 0:
        ax2c.axhspan(y_ctr[i] - 0.42, y_ctr[i] + 0.42, color='#F8F8F8', zorder=0)

for gi, gs in enumerate(group_stats):
    mean_abs_g = gs['mean_abs']
    ci_lo_g = gs['ci_lo']
    ci_hi_g = gs['ci_hi']
    n_g = gs['n']
    med_v0 = gs['median_v0']

    for rk, feat_idx in enumerate(rank_idx):
        yy = y_ctr[rk] + offsets[gi]
        val = mean_abs_g[feat_idx]
        err_lo = max(val - ci_lo_g[feat_idx], 0.0)
        err_hi = max(ci_hi_g[feat_idx] - val, 0.0)

        ax2c.barh(
            yy, val, height=bar_h,
            color=MAG_CLR[gi], edgecolor='white', lw=0.6, alpha=0.87, zorder=3,
            xerr=np.array([[err_lo], [err_hi]]),
            error_kw=dict(ecolor='#424242', lw=0.8, capsize=2, capthick=0.6),
            label=(
                f"{MAG_NAMES[gi]} (n={n_g}, median $V_0$={_fmt_num(med_v0)})"
                if rk == 0 else ''
            )
        )

        if val >= 0.008:
            ax2c.text(
                ci_hi_g[feat_idx] + 0.0055, yy, f'{val:.3f}',
                ha='left', va='center',
                fontsize=9.5, fontweight='bold', color='#333333', zorder=5
            )

ax2c.axvline(0, color='#757575', lw=0.6, zorder=1)
ax2c.set_yticks(y_ctr)
ax2c.set_yticklabels(SORTED_NAMES, fontsize=12.0)
ax2c.set_xlabel(r'Mean absolute SHAP value (contribution on log$_{10}(V_0)$ scale)', fontsize=13)
ax2c.grid(axis='x')
ax2c.grid(axis='y', visible=False)

handles, labels = ax2c.get_legend_handles_labels()
unique_handles, unique_labels = [], []
seen = set()
for h, lab in zip(handles, labels):
    if lab not in seen and lab != '':
        unique_handles.append(h)
        unique_labels.append(lab)
        seen.add(lab)

ax2c.legend(
    unique_handles, unique_labels,
    loc='lower right',
    fontsize=9.8,
    framealpha=0.97,
    facecolor='white',
    edgecolor='#BDBDBD',
    title='Observed debris-flow magnitude class',
    title_fontsize=10.2,
    handlelength=1.5,
    handletextpad=0.5,
    labelspacing=0.35
)

_panel_label(ax2c, '(c)')

# (d) Sample-level SHAP heatmap sorted by predicted V0
ax2d = fig2.add_subplot(gs2[1, 1])
_restore_all_spines(ax2d)
ax2d.grid(False)

sort_order = np.argsort(y_pred)
shap_sorted = shap_values[sort_order][:, rank_idx].T
sample_id_sorted = sample_id[sort_order]
pred_sorted_log = y_pred[sort_order]

abs_lim = np.max(np.abs(shap_sorted))
norm_hm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0, vmax=abs_lim)

im_hm = ax2d.imshow(
    shap_sorted, aspect='auto', cmap='RdBu_r',
    norm=norm_hm, interpolation='nearest', origin='upper'
)

for yi in range(n_features - 1):
    ax2d.axhline(yi + 0.5, color='white', lw=1.15, zorder=5)

ax2d.set_yticks(range(n_features))
ax2d.set_yticklabels(SORTED_NAMES, fontsize=11.2)

n_xticks = 6
xtick_pos = np.linspace(0, n_samples - 1, n_xticks).astype(int)
ax2d.set_xticks(xtick_pos)
ax2d.set_xticklabels([str(sample_id_sorted[i]) for i in xtick_pos], fontsize=10.8)
ax2d.set_xlabel(r'Sample ID (ordered by predicted debris-flow volume)', fontsize=13)

# Top axis: predicted V0 on original scale
ax2d_top = ax2d.twiny()
ax2d_top.set_xlim(ax2d.get_xlim())
ax2d_top.set_xticks(xtick_pos)
ax2d_top.set_xticklabels([_fmt_num(10 ** pred_sorted_log[i]) for i in xtick_pos], fontsize=11.0, color='#757575')
ax2d_top.set_xlabel(r'Predicted debris-flow volume, $V_0$ ($\times 10^4$ m$^3$)', fontsize=13, color='#757575',
                    labelpad=8)
ax2d_top.tick_params(colors='#757575', labelsize=11.0)
ax2d_top.spines['top'].set_visible(True)
ax2d_top.spines['top'].set_color('#757575')
ax2d_top.spines['top'].set_linewidth(0.6)

cb_hm = fig2.colorbar(im_hm, ax=ax2d, fraction=0.040, pad=0.04, aspect=22)
cb_hm.set_label(r'SHAP value (contribution on log$_{10}(V_0)$ scale)', fontsize=12, rotation=90, labelpad=14)
cb_hm.outline.set_linewidth(0.5)

_panel_label(ax2d, '(d)')

_save(fig2, 'Fig11_Local_and_Scale_Dependent_SHAP')

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
print('\n' + '=' * 78)
print('All optimized figures have been exported successfully.')
print('=' * 78)

for f in sorted(FIG_DIR.glob('*.jpg')):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f'  {f.name}  ({size_mb:.2f} MB)')

print(f'\nAll figures are 900-dpi JPG files and have been saved to:\n{FIG_DIR}')