"""
The paper's two figures (Figure 1 in Sec. 4.3, Figure 2 in App. I), from saved JSON only.

  fig2_block_vs_full    <- results/camera_ready/matched_summary.json (across-seed geometric means
                           and pointwise 95% t-intervals of E, X and S, validation split) and
                           results/camera_ready/section4_numbers.json (raw E/X per checkpoint)
  fig4_conductance      <- results/camera_ready/conductance_stats_matched.json (cumulative
                           distribution of g from the log10 g histograms)
                           and results/pretrained/scan_probe_large_mamba-370m.json (per-layer
                           fraction of g below 0.01 and least-squares scan-probe slope, tau = 10-100)

Colors: the Okabe-Ito condition colors (slow orange, default blue, fast
green) identify initializations; curve types use colors outside that set (full model black, block
formula reddish purple, scan probe gray) with distinct line styles. Sized for the 5.5-inch NeurIPS
text width at 8 pt. Writes PDF + PNG to figures/ and, if paper/figures/ exists, copies the PDF there.

Usage:
    python3 scripts/make_camera_ready_figures.py
"""

import json
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COND = [('slow', 'slow', '#D55E00'), ('default', 'default', '#0072B2'), ('fast', 'fast', '#009E73')]
CURVES = [('E', 'Full model', '#000000', '-'),
          ('X', 'Block formula (frozen selectivity)', '#CC79A7', '--'),
          ('S', 'Scan probe (scan-level formula)', '#7F7F7F', ':')]
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'lines.linewidth': 1.3, 'axes.linewidth': 0.6,
                     'font.family': 'serif', 'mathtext.fontset': 'stix'})


def save(fig, name):
    os.makedirs('figures', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/{name}.{ext}', bbox_inches='tight', dpi=300)
    copied = os.path.isdir('paper/figures')   # optional copy into a LaTeX tree
    if copied:
        shutil.copy(f'figures/{name}.pdf', f'paper/figures/{name}.pdf')
    plt.close(fig)
    print(f'wrote figures/{name}.pdf/.png' + (f' and paper/figures/{name}.pdf' if copied else ''))


def grid(ax):
    ax.grid(True, which='major', color='#d9d9d9', linewidth=0.4)
    ax.set_axisbelow(True)


def fig2():
    summ = json.load(open('results/camera_ready/matched_summary.json'))['across_seeds']['validation']
    sec4 = json.load(open('results/camera_ready/section4_numbers.json'))
    lags = np.arange(1, 1024)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.2))
    for ax, (cond, label, _) in zip(axes.flat[:3], COND):
        for key, name, color, ls in CURVES:
            d = summ[cond][key]
            ax.fill_between(lags, d['pointwise_t95_lo'], d['pointwise_t95_hi'], color=color,
                            alpha=0.18, linewidth=0)
            ax.plot(lags, d['geometric_mean_curve_lag1_to_1023'], color=color, ls=ls, label=name)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 1023)
        ax.set_ylim(1e-14, 2)
        ax.set_title(f'({"abc"[[c[0] for c in COND].index(cond)]}) {label} $A$-initialization (3 seeds)')
        ax.set_xlabel(r'lag $\tau$')
        ax.set_ylabel(r'ERF, normalized at $\tau = 1$')
        grid(ax)
    ax = axes.flat[3]
    curves = sec4['raw_E_over_X_curves_validation_lag1_to_1023']
    for key, ratio in curves.items():
        if key != 'default_seed3':
            ax.plot(lags, ratio, color='#9e9e9e', linewidth=0.7)
    ax.plot(lags, curves['default_seed3'], color='#000000', linewidth=1.1)
    ax.annotate('default, seed 3', xy=(420, curves['default_seed3'][419]), xytext=(40, 3.2),
                fontsize=7, arrowprops=dict(arrowstyle='-', color='#000000', lw=0.5))
    ax.axhline(1.0, color='#4d4d4d', linewidth=0.6)
    ax.plot([1, 500], [2.3, 2.3], color='#CC79A7', ls='--', linewidth=0.9)
    ax.text(1.3, 2.5, r'$2.3\times$ ($\tau \leq 500$)', fontsize=7, color='#000000')
    ax.axvline(500, color='#4d4d4d', ls=':', linewidth=0.7)
    ax.set_xscale('log')
    ax.set_xlim(1, 1023)
    ax.set_ylim(0.9, 5.4)
    ax.set_title('(d) full model / block formula, raw (9 models)')
    ax.set_xlabel(r'lag $\tau$')
    ax.set_ylabel('ratio')
    grid(ax)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, 'fig2_block_vs_full')


def fig4():
    cs = json.load(open('results/camera_ready/conductance_stats_matched.json'))['models']
    p370 = json.load(open('results/pretrained/scan_probe_large_mamba-370m.json'))['layers']
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.95))
    edges = np.linspace(-6.0, 0.0, 61)
    for cond, label, color in COND:
        frac = np.mean([cs[f'{cond}_seed{s}']['log10_g_histogram']['fraction'] for s in (1, 2, 3)], axis=0)
        a.plot(10 ** edges[1:], np.cumsum(frac), color=color, label=label, linewidth=1.1,
               ls={'slow': '-', 'default': '--', 'fast': ':'}[cond])  # styles keep coinciding curves visible
    a.axvline(0.01, color='#4d4d4d', ls=':', linewidth=0.7)
    a.text(0.012, 0.62, 'coast:\n$g < 0.01$', fontsize=7, color='#000000')
    a.set_xscale('log')
    a.set_xlim(1e-6, 1)
    a.set_ylim(0, 1)
    a.set_xlabel(r'conductance $g$')
    a.set_ylabel('cumulative fraction')
    a.set_title('(a) single-layer models, 3 seeds each')
    a.legend(frameon=False, loc='upper left', title=r'$A$-init', title_fontsize=7, handlelength=2.6)
    grid(a)
    frac = np.array([v['g_stats']['frac_below_0.01'] for v in p370.values()])
    slope = np.array([v['slope_ls']['10-100']['value'] for v in p370.values()])
    lo = np.array([v['slope_ls']['10-100']['ci95'][0] for v in p370.values()])
    hi = np.array([v['slope_ls']['10-100']['ci95'][1] for v in p370.values()])
    b.errorbar(100 * frac, slope, yerr=[slope - lo, hi - slope], fmt='o', ms=3, color='#000000',
               ecolor='#9e9e9e', elinewidth=0.6, capsize=0)
    for name, off in (('L34', (-18, 3)), ('L47', (4, 3))):
        i = list(p370).index(name)
        b.annotate(name, xy=(100 * frac[i], slope[i]), xytext=off, textcoords='offset points', fontsize=7)
    b.set_xscale('log')
    b.set_xlabel(r'conductances below 0.01 (%)')
    b.set_ylabel(r'scan-probe slope ($\tau = 10$–$100$)')
    b.set_title('(b) Mamba-370M, 48 layers')
    b.set_ylim(top=0.9)                      # headroom so the L34 label sits inside the axes
    grid(b)
    fig.tight_layout()
    save(fig, 'fig4_conductance')


if __name__ == '__main__':
    fig2()
    fig4()
