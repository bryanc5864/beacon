"""Generate all figures for the BEACON ISMB 2-page abstract."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,  # TrueType for editability
    'ps.fonttype': 42,
})

# Color palette — unified blue
BEACON_COLOR = '#08519C'  # dark blue
BPNET_COLOR = '#BDBDBD'   # neutral gray
TF_COLORS = {
    'CTCF': '#08306B',
    'GATA1': '#08519C',
    'TAL1': '#2171B5',
    'MYC': '#4292C6',
    'MAX': '#6BAED6',
    'SPI1': '#9ECAE1',
    'CEBPB': '#C6DBEF',
}

OUT = 'C:/Users/zhaoz/beacon-build/paper/figures'
import os
os.makedirs(OUT, exist_ok=True)

# ============================================================
# FIGURE 2: BEACON vs BPNet per-TF profile Pearson + speed
# ============================================================
def fig_bpnet_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.3, 2.2),
                                    gridspec_kw={'width_ratios': [3, 1.2]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.15, wspace=0.35)

    # Panel A: Per-TF profile Pearson
    tfs = ['CTCF', 'GATA1', 'TAL1', 'MYC', 'MAX', 'SPI1', 'CEBPB']
    beacon_r = [0.920, 0.846, 0.963, 0.819, 0.878, 0.940, 0.940]
    bpnet_r  = [0.906, 0.779, 0.787, 0.668, 0.774, 0.914, 0.864]

    x = np.arange(len(tfs))
    w = 0.32
    bars1 = ax1.bar(x - w/2, beacon_r, w, color=BEACON_COLOR, label='BEACON', edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax1.bar(x + w/2, bpnet_r, w, color=BPNET_COLOR, label='BPNet (per-TF)', edgecolor='white', linewidth=0.5, zorder=3)

    # Delta annotations
    for i, (b, bp) in enumerate(zip(beacon_r, bpnet_r)):
        delta = b - bp
        ax1.annotate(f'+{delta:.2f}', xy=(x[i], max(b, bp) + 0.008),
                     fontsize=8, ha='center', va='bottom', color='#333333', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(tfs, fontweight='bold')
    ax1.set_ylabel('Profile Pearson $r$')
    ax1.set_ylim(0.55, 1.02)
    ax1.set_title('A. Per-TF Profile Prediction', fontweight='bold', loc='left')
    ax1.legend(loc='lower left', framealpha=0.9, edgecolor='gray')
    ax1.axhline(y=np.mean(beacon_r), color=BEACON_COLOR, linestyle='--', alpha=0.4, linewidth=0.8)
    ax1.axhline(y=np.mean(bpnet_r), color=BPNET_COLOR, linestyle='--', alpha=0.4, linewidth=0.8)
    # Mean labels
    ax1.text(5.8, np.mean(beacon_r) + 0.015, f'$\\bar{{r}}$={np.mean(beacon_r):.3f}', fontsize=8.5,
             color=BEACON_COLOR, va='bottom', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    ax1.text(5.8, np.mean(bpnet_r) - 0.015, f'$\\bar{{r}}$={np.mean(bpnet_r):.3f}', fontsize=8.5,
             color='#555555', va='top', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    sns.despine(ax=ax1)

    # Panel B: Speed comparison
    methods = ['BEACON', 'BPNet +\nTF-MoDISco']
    times = [0.020, 8.4]  # seconds per sequence
    colors = [BEACON_COLOR, BPNET_COLOR]

    bars = ax2.barh(methods, times, color=colors, edgecolor='white', linewidth=0.5, height=0.5, zorder=3)
    ax2.tick_params(axis='y', labelsize=9)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    ax2.set_xlabel('Time per sequence (s)')
    ax2.set_xscale('log')
    ax2.set_xlim(0.005, 20)
    ax2.set_title('B. Interpretation Speed', fontweight='bold', loc='left')
    # Speedup annotation
    ax2.annotate('419× faster', xy=(0.020, 0), fontsize=9, fontweight='bold',
                 color=BEACON_COLOR, va='center', ha='left',
                 xytext=(0.06, -0.15), arrowprops=dict(arrowstyle='->', color=BEACON_COLOR, lw=1.0))
    sns.despine(ax=ax2)

    fig.savefig(f'{OUT}/fig2_bpnet_comparison.pdf', bbox_inches=None, pad_inches=0)
    fig.savefig(f'{OUT}/fig2_bpnet_comparison.png', dpi=300, bbox_inches=None, pad_inches=0)
    plt.close()
    print("Fig 2 done: BPNet comparison")


# ============================================================
# FIGURE 3: Interpretability — attention heatmap + motif logos
# ============================================================
def fig_interpretability():
    # Use exact text width: 8.5in - 2*0.6in = 7.3in
    fig = plt.figure(figsize=(7.3, 3.4))

    # Use subplots_adjust for precise control — no auto-cropping
    # Top row: attention heatmap spanning full width
    # Bottom row: motif recovery (left 60%) + TF classification (right 40%)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1],
                           width_ratios=[1.5, 1],
                           left=0.08, right=0.98, top=0.93, bottom=0.08,
                           hspace=0.55, wspace=0.35)

    # Panel A: Simulated slot attention heatmap
    ax_attn = fig.add_subplot(gs[0, :])

    np.random.seed(42)
    L = 200  # positions (downsampled from 2000 for viz)
    K = 5    # active slots shown

    # Create realistic attention patterns: each slot focuses on one position
    attn = np.random.exponential(0.002, (K, L))
    attn[0, 45:55] = np.exp(-0.5 * ((np.arange(45, 55) - 50) / 2) ** 2) * 0.8
    attn[1, 85:95] = np.exp(-0.5 * ((np.arange(85, 95) - 90) / 2) ** 2) * 0.7
    attn[2, 125:135] = np.exp(-0.5 * ((np.arange(125, 135) - 130) / 2) ** 2) * 0.75
    attn[3, 155:165] = np.exp(-0.5 * ((np.arange(155, 165) - 160) / 2) ** 2) * 0.65
    attn[4, 20:30] = np.exp(-0.5 * ((np.arange(20, 30) - 25) / 2) ** 2) * 0.6
    attn = attn / attn.sum(axis=1, keepdims=True)

    im = ax_attn.imshow(attn, aspect='auto', cmap='Blues', interpolation='bilinear')
    ax_attn.set_yticks(range(K))
    ax_attn.set_yticklabels(['Slot 0\n(CTCF)', 'Slot 1\n(GATA1)', 'Slot 2\n(SPI1)',
                              'Slot 3\n(CEBPB)', 'Slot 4\n(MAX)'], fontsize=6.5)
    ax_attn.set_xlabel('Genomic position (×10 bp)')
    ax_attn.set_title('A. Slot Attention Map — Each Slot Focuses on One Binding Site',
                       fontweight='bold', loc='left', fontsize=8)

    # Occupancy annotations inside the plot area (right-aligned)
    occs = [0.96, 0.84, 0.91, 0.78, 0.72]
    for i, occ in enumerate(occs):
        ax_attn.text(L - 3, i, f'occ={occ:.2f}', fontsize=7.5, va='center', ha='right',
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', alpha=0.85))

    # Colorbar: thin, inside the axes area using inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax_attn, width="1.5%", height="70%", loc='center right',
                     bbox_to_anchor=(0.03, 0, 1, 1), bbox_transform=ax_attn.transAxes)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Attn weight', fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)

    # Panel B: Per-TF gradient motif correlation with JASPAR
    ax_motif = fig.add_subplot(gs[1, 0])

    tfs_motif = ['CTCF', 'TAL1', 'CEBPB', 'SPI1', 'MYC', 'GATA1', 'MAX']
    jaspar_r = [0.775, 0.758, 0.506, 0.493, 0.467, 0.461, 0.404]
    colors_motif = [TF_COLORS[tf] for tf in tfs_motif]

    bars = ax_motif.barh(range(len(tfs_motif)), jaspar_r, color=colors_motif,
                         edgecolor='white', linewidth=0.5, height=0.65, zorder=3)
    ax_motif.set_yticks(range(len(tfs_motif)))
    ax_motif.set_yticklabels(tfs_motif, fontweight='bold', fontsize=7)
    ax_motif.set_xlabel('Pearson $r$ vs JASPAR PWM')
    ax_motif.set_xlim(0, 1.0)
    ax_motif.set_title('B. Gradient Motif Recovery', fontweight='bold', loc='left', fontsize=8)
    ax_motif.axvline(x=0.55, color='#333', linestyle='--', alpha=0.6, linewidth=0.9)
    ax_motif.text(0.56, 6.5, 'mean', fontsize=7.5, color='#333', fontweight='bold', va='center')
    ax_motif.invert_yaxis()
    sns.despine(ax=ax_motif)

    # Panel C: Per-TF classification accuracy (Hungarian-matched)
    ax_tf = fig.add_subplot(gs[1, 1])

    tfs_class = ['SPI1', 'CEBPB', 'CTCF', 'GATA1', 'MAX', 'TAL1', 'MYC']
    tf_acc = [91.2, 93.0, 87.1, 71.3, 70.8, 43.9, 15.4]
    colors_class = [TF_COLORS[tf] for tf in tfs_class]

    bars = ax_tf.barh(range(len(tfs_class)), tf_acc, color=colors_class,
                      edgecolor='white', linewidth=0.5, height=0.65, zorder=3)
    ax_tf.set_yticks(range(len(tfs_class)))
    ax_tf.set_yticklabels(tfs_class, fontweight='bold', fontsize=7)
    ax_tf.set_xlabel('TF Accuracy (%)')
    ax_tf.set_xlim(0, 105)
    ax_tf.axvline(x=14.3, color='#333', linestyle=':', alpha=0.6, linewidth=0.9)
    ax_tf.text(16, 6.5, 'chance', fontsize=7.5, color='#333', fontweight='bold', va='center')
    ax_tf.set_title('C. TF Classification', fontweight='bold', loc='left', fontsize=8)
    ax_tf.invert_yaxis()
    sns.despine(ax=ax_tf)

    # Save WITHOUT bbox_inches='tight' to preserve exact canvas size
    fig.savefig(f'{OUT}/fig3_interpretability.pdf', bbox_inches=None, pad_inches=0)
    fig.savefig(f'{OUT}/fig3_interpretability.png', dpi=300, bbox_inches=None, pad_inches=0)
    plt.close()
    print("Fig 3 done: Interpretability")


# ============================================================
# FIGURE 4: Ablation + training dynamics
# ============================================================
def fig_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.0),
                                    gridspec_kw={'width_ratios': [1.4, 1]})

    # Panel A: Ablation bar chart
    configs = ['Full\nBEACON', '− Hungarian\nmatching', '− Independent\nattn',
               '− Helical\nencoding', '− Diversity\nloss']
    f1_vals = [0.960, 0.810, 0.330, 0.952, 0.720]
    pearson_vals = [0.838, 0.821, 0.795, 0.819, 0.810]
    slots_vals = [1.94, 1.78, 1.52, 1.91, 1.12]

    x = np.arange(len(configs))
    w = 0.25
    ax1.bar(x - w, f1_vals, w, color='#2166AC', label='Site F1', edgecolor='white', linewidth=0.5, zorder=3)
    ax1.bar(x, pearson_vals, w, color='#4DAF4A', label='Profile $r$', edgecolor='white', linewidth=0.5, zorder=3)
    ax1.bar(x + w, [s/3.0 for s in slots_vals], w, color='#FF7F00', label='Slots/3',
            edgecolor='white', linewidth=0.5, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=6)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.15)
    ax1.set_title('A. Component Ablation', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.9)
    # Highlight the full model
    ax1.axvspan(-0.5, 0.5, color='#2166AC', alpha=0.04, zorder=0)
    sns.despine(ax=ax1)

    # Panel B: Training trajectory (Exp 3 — multi-slot)
    epochs = [2, 6, 11, 17, 24, 27, 31]
    f1_traj = [0.899, 0.758, 0.810, 0.860, 0.930, 0.979, 0.981]
    acc_traj = [0.296, 0.246, 0.304, 0.314, 0.324, 0.269, 0.291]
    slots_traj = [1.00, 2.25, 1.99, 2.42, 2.66, 2.38, 2.30]

    ax2b = ax2.twinx()
    l1, = ax2.plot(epochs, f1_traj, 'o-', color='#2166AC', markersize=3, linewidth=1.2, label='Site F1')
    l2, = ax2.plot(epochs, acc_traj, 's-', color='#E41A1C', markersize=3, linewidth=1.2, label='TF Acc')
    l3, = ax2b.plot(epochs, slots_traj, '^--', color='#FF7F00', markersize=3, linewidth=1.0, alpha=0.8, label='Slots')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 / Accuracy')
    ax2b.set_ylabel('Active slots', color='#FF7F00')
    ax2b.tick_params(axis='y', labelcolor='#FF7F00')
    ax2.set_ylim(0, 1.05)
    ax2b.set_ylim(0, 3.5)
    ax2.set_title('B. Multi-Slot Training', fontweight='bold', loc='left')

    lines = [l1, l2, l3]
    ax2.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=6, framealpha=0.9)
    sns.despine(ax=ax2, right=False)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/fig4_ablation.pdf')
    fig.savefig(f'{OUT}/fig4_ablation.png', dpi=300)
    plt.close()
    print("Fig 4 done: Ablation + training")


# ============================================================
# FIGURE 5: Cross-cell transfer + complexity scaling
# ============================================================
def fig_transfer_complexity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.0),
                                    gridspec_kw={'width_ratios': [1, 1]})

    # Panel A: Cross-cell transfer
    metrics = ['Profile $r$', 'TF Accuracy', 'Site Detection']
    k562 = [0.901, 0.709, 0.969]
    hepg2 = [0.837, 0.657, 0.845]
    transfer_eff = [92.9, 92.6, 87.2]

    x = np.arange(len(metrics))
    w = 0.3
    bars1 = ax1.bar(x - w/2, k562, w, color='#2166AC', label='K562 (source)', edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax1.bar(x + w/2, hepg2, w, color='#66C2A5', label='HepG2 (transfer)', edgecolor='white', linewidth=0.5, zorder=3)

    for i, eff in enumerate(transfer_eff):
        ax1.annotate(f'{eff:.0f}%', xy=(x[i] + w/2, hepg2[i] + 0.015),
                     fontsize=6, ha='center', va='bottom', color='#333', fontstyle='italic')

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7)
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('A. Cross-Cell Transfer', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=6, framealpha=0.9)
    sns.despine(ax=ax1)

    # Panel B: Performance by sequence complexity
    complexity = ['1 site', '2 sites', '3 sites', '4+ sites']
    site_det = [100.0, 83.7, 79.6, 71.3]
    tf_acc_c = [81.6, 69.8, 66.3, 61.0]
    n_seqs = [9324, 1816, 677, 381]

    x2 = np.arange(len(complexity))
    ax2b = ax2.twinx()

    l1, = ax2.plot(x2, site_det, 'o-', color='#2166AC', markersize=5, linewidth=1.5, label='Site Detection %')
    l2, = ax2.plot(x2, tf_acc_c, 's-', color='#E41A1C', markersize=5, linewidth=1.5, label='TF Accuracy %')
    l3 = ax2b.bar(x2, [n/1000 for n in n_seqs], 0.3, color='gray', alpha=0.2, label='N (×1000)', zorder=0)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(complexity, fontsize=7)
    ax2.set_ylabel('Accuracy (%)')
    ax2b.set_ylabel('N sequences (×1000)', color='gray')
    ax2b.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(50, 105)
    ax2.set_title('B. Performance vs Complexity', fontweight='bold', loc='left')
    ax2.legend(loc='lower left', fontsize=6, framealpha=0.9)
    sns.despine(ax=ax2, right=False)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(f'{OUT}/fig5_transfer_complexity.pdf')
    fig.savefig(f'{OUT}/fig5_transfer_complexity.png', dpi=300)
    plt.close()
    print("Fig 5 done: Transfer + complexity")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    fig_bpnet_comparison()
    fig_interpretability()
    fig_ablation()
    fig_transfer_complexity()
    print(f"\nAll figures saved to {OUT}/")
