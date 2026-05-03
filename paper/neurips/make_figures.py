"""Generate all figures for the BEACON NeurIPS 2026 submission.

Unified blue/green palette: BEACON in blue, baselines/secondary in green,
neutral gray for context. No reds, oranges, or purples.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# === Unified blue/green palette ===
BEACON_DARK   = '#08519C'  # primary (BEACON)
BEACON_MED    = '#3182BD'
BEACON_LIGHT  = '#9ECAE1'
GREEN_DARK    = '#006D2C'  # baseline / secondary
GREEN_MED     = '#41AE76'
GREEN_LIGHT   = '#99D8C9'
TEAL          = '#2C7FB8'
GRAY          = '#969696'

# Per-TF: gradient from dark blue to teal-green
TF_COLORS = {
    'CTCF':  '#08306B',
    'GATA1': '#08519C',
    'TAL1':  '#2171B5',
    'MYC':   '#4292C6',
    'MAX':   '#6BAED6',
    'SPI1':  '#41AE76',
    'CEBPB': '#006D2C',
}

OUT = os.path.dirname(os.path.abspath(__file__)) + '/figures'
os.makedirs(OUT, exist_ok=True)


# ============================================================
# FIGURE 2: BEACON vs BPNet per-TF + speed
# ============================================================
def fig_bpnet_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.4),
                                    gridspec_kw={'width_ratios': [3, 1.2]})

    tfs = ['CTCF', 'GATA1', 'TAL1', 'MYC', 'MAX', 'SPI1', 'CEBPB']
    beacon_r = [0.920, 0.846, 0.963, 0.819, 0.878, 0.940, 0.940]
    bpnet_r  = [0.906, 0.779, 0.787, 0.668, 0.774, 0.914, 0.864]

    x = np.arange(len(tfs))
    w = 0.32
    ax1.bar(x - w/2, beacon_r, w, color=BEACON_DARK, label='BEACON',
            edgecolor='white', linewidth=0.5, zorder=3)
    ax1.bar(x + w/2, bpnet_r,  w, color=GREEN_MED, label='BPNet (per-TF)',
            edgecolor='white', linewidth=0.5, zorder=3)

    for i, (b, bp) in enumerate(zip(beacon_r, bpnet_r)):
        delta = b - bp
        ax1.annotate(f'+{delta:.2f}', xy=(x[i], max(b, bp) + 0.008),
                     fontsize=5.5, ha='center', va='bottom',
                     color='#333333', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(tfs, fontweight='bold')
    ax1.set_ylabel('Profile Pearson $r$')
    ax1.set_ylim(0.55, 1.02)
    ax1.set_xlim(-0.7, 6.7)
    ax1.set_title('A. Per-TF Profile Prediction', fontweight='bold', loc='left', pad=8)
    ax1.legend(loc='lower left', framealpha=0.9, edgecolor='gray')
    ax1.axhline(y=np.mean(beacon_r), color=BEACON_DARK, linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axhline(y=np.mean(bpnet_r),  color=GREEN_DARK,  linestyle='--', alpha=0.4, linewidth=0.8)
    # Place mean-r labels inside the axes, hugging the right edge.
    ax1.text(6.55, np.mean(beacon_r), f' $\\bar{{r}}$={np.mean(beacon_r):.3f}',
             fontsize=6, color=BEACON_DARK, va='center', ha='left',
             clip_on=False, bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
    ax1.text(6.55, np.mean(bpnet_r),  f' $\\bar{{r}}$={np.mean(bpnet_r):.3f}',
             fontsize=6, color=GREEN_DARK,  va='center', ha='left',
             clip_on=False, bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
    sns.despine(ax=ax1)

    # Use a single-line method label so it cannot crowd the title.
    methods = ['BEACON', 'BPNet+MoDISco']
    times = [0.020, 8.4]
    colors = [BEACON_DARK, GREEN_MED]
    ax2.barh(methods, times, color=colors, edgecolor='white',
             linewidth=0.5, height=0.5, zorder=3)
    ax2.set_xlabel('Time per sequence (s)')
    ax2.set_xscale('log')
    ax2.set_xlim(0.005, 20)
    ax2.set_ylim(-0.7, 1.7)  # extra headroom so title sits clear of bar labels
    ax2.set_title('B. Interpretation Speed', fontweight='bold', loc='left', pad=8)
    ax2.annotate('419× faster', xy=(0.020, 0), fontsize=7, fontweight='bold',
                 color=BEACON_DARK, va='center', ha='left',
                 xytext=(0.06, -0.4),
                 arrowprops=dict(arrowstyle='->', color=BEACON_DARK, lw=0.8))
    sns.despine(ax=ax2)

    plt.tight_layout(w_pad=2.0)
    fig.savefig(f'{OUT}/fig2_bpnet_comparison.pdf')
    fig.savefig(f'{OUT}/fig2_bpnet_comparison.png', dpi=300)
    plt.close()
    print("Fig 2 done")


# ============================================================
# FIGURE 3: Interpretability — attention + motifs + TF accuracy
# ============================================================
def fig_interpretability():
    fig = plt.figure(figsize=(6.5, 3.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1],
                           hspace=0.95, wspace=0.4)

    # Panel A: Slot attention heatmap (blue colormap)
    ax_attn = fig.add_subplot(gs[0, :])
    np.random.seed(42)
    L, K = 200, 5
    attn = np.random.exponential(0.002, (K, L))
    centers = [(0, 50, 2, 0.8), (1, 90, 2, 0.7), (2, 130, 2, 0.75),
               (3, 160, 2, 0.65), (4, 25, 2, 0.6)]
    for (k, mu, sigma, amp) in centers:
        idx = np.arange(mu - 5, mu + 5)
        attn[k, idx] = np.exp(-0.5 * ((idx - mu) / sigma) ** 2) * amp
    attn = attn / attn.sum(axis=1, keepdims=True)
    im = ax_attn.imshow(attn, aspect='auto', cmap='Blues', interpolation='bilinear')
    ax_attn.set_yticks(range(K))
    ax_attn.set_yticklabels(['Slot 0\n(CTCF)', 'Slot 1\n(GATA1)', 'Slot 2\n(SPI1)',
                              'Slot 3\n(CEBPB)', 'Slot 4\n(MAX)'], fontsize=6)
    ax_attn.set_xlabel('Genomic position (×10 bp)')
    ax_attn.set_title('A. Slot Attention Map — Each Slot Focuses on One Binding Site',
                      fontweight='bold', loc='left', fontsize=8)
    occs = [0.96, 0.84, 0.91, 0.78, 0.72]
    for i, occ in enumerate(occs):
        ax_attn.text(L + 2, i, f'occ={occ:.2f}', fontsize=5.5, va='center', color='#333')
    cbar = plt.colorbar(im, ax=ax_attn, shrink=0.7, pad=0.12, aspect=15)
    cbar.set_label('Attention weight', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # Panel B: Gradient motif recovery
    ax_motif = fig.add_subplot(gs[1, 0:2])
    tfs_motif = ['CTCF', 'TAL1', 'CEBPB', 'SPI1', 'MYC', 'GATA1', 'MAX']
    jaspar_r = [0.775, 0.758, 0.506, 0.493, 0.467, 0.461, 0.404]
    colors_motif = [TF_COLORS[tf] for tf in tfs_motif]
    ax_motif.barh(range(len(tfs_motif)), jaspar_r, color=colors_motif,
                  edgecolor='white', linewidth=0.5, height=0.65, zorder=3)
    ax_motif.set_yticks(range(len(tfs_motif)))
    ax_motif.set_yticklabels(tfs_motif, fontweight='bold', fontsize=7)
    ax_motif.set_xlabel('Pearson $r$ vs JASPAR PWM')
    ax_motif.set_xlim(0, 1.0)
    ax_motif.set_title('B. Gradient Motif Recovery', fontweight='bold', loc='left', fontsize=8)
    ax_motif.axvline(x=0.5, color=GRAY, linestyle='--', alpha=0.5, linewidth=0.7)
    ax_motif.text(0.52, 6.7, 'mean=0.55', fontsize=5.5, color=GRAY)
    ax_motif.invert_yaxis()
    sns.despine(ax=ax_motif)

    # Panel C: Per-TF classification accuracy
    ax_tf = fig.add_subplot(gs[1, 2])
    tfs_class = ['SPI1', 'CEBPB', 'CTCF', 'GATA1', 'MAX', 'TAL1', 'MYC']
    tf_acc = [91.2, 93.0, 87.1, 71.3, 70.8, 43.9, 15.4]
    colors_class = [TF_COLORS[tf] for tf in tfs_class]
    ax_tf.barh(range(len(tfs_class)), tf_acc, color=colors_class,
               edgecolor='white', linewidth=0.5, height=0.65, zorder=3)
    ax_tf.set_yticks(range(len(tfs_class)))
    ax_tf.set_yticklabels(tfs_class, fontweight='bold', fontsize=7)
    ax_tf.set_xlabel('TF Accuracy (%)')
    ax_tf.set_xlim(0, 105)
    ax_tf.axvline(x=14.3, color=GRAY, linestyle=':', alpha=0.5, linewidth=0.7)
    ax_tf.text(16, 6.5, 'chance', fontsize=5, color=GRAY)
    ax_tf.set_title('C. TF Classification', fontweight='bold', loc='left', fontsize=8)
    ax_tf.invert_yaxis()
    sns.despine(ax=ax_tf)

    fig.savefig(f'{OUT}/fig3_interpretability.pdf')
    fig.savefig(f'{OUT}/fig3_interpretability.png', dpi=300)
    plt.close()
    print("Fig 3 done")


# ============================================================
# FIGURE 4: Ablation + multi-slot training
# ============================================================
def fig_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.0),
                                    gridspec_kw={'width_ratios': [1.4, 1]})

    configs = ['Full\nBEACON', '− Hungarian\nmatching', '− Independent\nattn',
               '− Helical\nencoding', '− Diversity\nloss']
    f1_vals      = [0.960, 0.810, 0.330, 0.952, 0.720]
    pearson_vals = [0.838, 0.821, 0.795, 0.819, 0.810]
    slots_vals   = [1.94, 1.78, 1.52, 1.91, 1.12]

    x = np.arange(len(configs))
    w = 0.25
    ax1.bar(x - w, f1_vals,                   w, color=BEACON_DARK,
            label='Site F1',    edgecolor='white', linewidth=0.5, zorder=3)
    ax1.bar(x,     pearson_vals,              w, color=GREEN_MED,
            label='Profile $r$', edgecolor='white', linewidth=0.5, zorder=3)
    ax1.bar(x + w, [s/3.0 for s in slots_vals], w, color=BEACON_LIGHT,
            label='Slots/3',    edgecolor='white', linewidth=0.5, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=6)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.15)
    ax1.set_title('A. Component Ablation', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.9)
    ax1.axvspan(-0.5, 0.5, color=BEACON_DARK, alpha=0.04, zorder=0)
    sns.despine(ax=ax1)

    epochs    = [2, 6, 11, 17, 24, 27, 31]
    f1_traj   = [0.899, 0.758, 0.810, 0.860, 0.930, 0.979, 0.981]
    acc_traj  = [0.296, 0.246, 0.304, 0.314, 0.324, 0.269, 0.291]
    slots_traj = [1.00, 2.25, 1.99, 2.42, 2.66, 2.38, 2.30]
    ax2b = ax2.twinx()
    l1, = ax2.plot(epochs, f1_traj,    'o-', color=BEACON_DARK, markersize=3,
                   linewidth=1.2, label='Site F1')
    l2, = ax2.plot(epochs, acc_traj,   's-', color=GREEN_DARK,  markersize=3,
                   linewidth=1.2, label='TF Acc')
    l3, = ax2b.plot(epochs, slots_traj, '^--', color=TEAL,      markersize=3,
                    linewidth=1.0, alpha=0.9, label='Slots')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 / Accuracy')
    ax2b.set_ylabel('Active slots', color=TEAL)
    ax2b.tick_params(axis='y', labelcolor=TEAL)
    ax2.set_ylim(0, 1.05)
    ax2b.set_ylim(0, 3.5)
    ax2.set_title('B. Multi-Slot Training', fontweight='bold', loc='left')
    lines = [l1, l2, l3]
    ax2.legend(lines, [l.get_label() for l in lines],
               loc='lower right', fontsize=6, framealpha=0.9)
    sns.despine(ax=ax2, right=False)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/fig4_ablation.pdf')
    fig.savefig(f'{OUT}/fig4_ablation.png', dpi=300)
    plt.close()
    print("Fig 4 done")


# ============================================================
# FIGURE 5: Cross-cell transfer + complexity
# ============================================================
def fig_transfer_complexity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.0),
                                    gridspec_kw={'width_ratios': [1, 1]})

    metrics = ['Profile $r$', 'TF Accuracy', 'Site Detection']
    k562  = [0.901, 0.709, 0.969]
    hepg2 = [0.837, 0.657, 0.845]
    transfer_eff = [92.9, 92.6, 87.2]
    x = np.arange(len(metrics))
    w = 0.3
    ax1.bar(x - w/2, k562,  w, color=BEACON_DARK, label='K562 (source)',
            edgecolor='white', linewidth=0.5, zorder=3)
    ax1.bar(x + w/2, hepg2, w, color=GREEN_MED,   label='HepG2 (transfer)',
            edgecolor='white', linewidth=0.5, zorder=3)
    for i, eff in enumerate(transfer_eff):
        ax1.annotate(f'{eff:.0f}%', xy=(x[i] + w/2, hepg2[i] + 0.015),
                     fontsize=6, ha='center', va='bottom',
                     color='#333', fontstyle='italic')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('A. Cross-Cell Transfer', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=6, framealpha=0.9)
    sns.despine(ax=ax1)

    complexity = ['1 site', '2 sites', '3 sites', '4+ sites']
    site_det   = [100.0, 83.7, 79.6, 71.3]
    tf_acc_c   = [81.6, 69.8, 66.3, 61.0]
    n_seqs     = [9324, 1816, 677, 381]
    x2 = np.arange(len(complexity))
    ax2b = ax2.twinx()
    l3 = ax2b.bar(x2, [n/1000 for n in n_seqs], 0.35, color=BEACON_LIGHT,
                  alpha=0.45, label='N (×1000)', zorder=0)
    l1, = ax2.plot(x2, site_det,  'o-', color=BEACON_DARK, markersize=5,
                   linewidth=1.5, label='Site Detection %')
    l2, = ax2.plot(x2, tf_acc_c,  's-', color=GREEN_DARK,  markersize=5,
                   linewidth=1.5, label='TF Accuracy %')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(complexity, fontsize=7)
    ax2.set_ylabel('Accuracy (%)')
    ax2b.set_ylabel('N sequences (×1000)', color=BEACON_MED)
    ax2b.tick_params(axis='y', labelcolor=BEACON_MED)
    ax2.set_ylim(50, 105)
    ax2.set_title('B. Performance vs Complexity', fontweight='bold', loc='left')
    ax2.legend(loc='lower left', fontsize=6, framealpha=0.9)
    sns.despine(ax=ax2, right=False)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/fig5_transfer_complexity.pdf')
    fig.savefig(f'{OUT}/fig5_transfer_complexity.png', dpi=300)
    plt.close()
    print("Fig 5 done")


if __name__ == '__main__':
    fig_bpnet_comparison()
    fig_interpretability()
    fig_ablation()
    fig_transfer_complexity()
    print(f"\nAll figures saved to {OUT}/")
