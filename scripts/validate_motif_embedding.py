#!/usr/bin/env python3
"""
A2 Validation: Open-Vocabulary Motif Embedding Quality.

Validates the MotifEmbeddingHead's ability to organize TF motifs in embedding
space.  Runs five analyses on the motif embedding outputs:

  1. Embedding clustering quality (silhouette score)
  2. Anchor proximity per known TF
  3. Prototype utilization
  4. TF classification accuracy (standard vs motif head)
  5. Novel motif detection fraction

Produces a formatted results table, a JSON results file, and a 2x2
diagnostic figure.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from beacon.models import BEACON
from beacon.data.dataset import BEACONDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional sklearn imports -------------------------------------------------
try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

TF_NAMES = ["CTCF", "GATA1", "TAL1", "MYC", "MAX", "SPI1", "CEBPB"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A2 Validation: Open-Vocabulary Motif Embedding Quality"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/phase10_slot_ablation/slots_16/"
            "beacon_20260203_231900/best_model.pt"
        ),
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/multi_tf_k562/test.h5"),
        help="Path to test HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/a2_motif_embedding_validation"),
        help="Directory for output files",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum samples per TF",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Resolve relative paths
    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path

    test_data_path = args.test_data
    if not test_data_path.is_absolute():
        test_data_path = base_dir / test_data_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("A2 Validation: Open-Vocabulary Motif Embedding Quality")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Test data  : {test_data_path}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {device}")
    print(f"Max samples: {args.max_samples}")

    print("\n--- Loading Model ---")
    model = BEACON(
        seq_len=2000,
        input_channels=4,
        backbone_type="dilated",
        backbone_dim=128,
        backbone_layers=4,
        n_slots=16,
        slot_dim=128,
        n_iterations=3,
        n_tfs=7,
        position_mode="gaussian",
        dropout=0.0,
        attention_mode="independent",
        use_motif_embedding_head=True,
        motif_dim=64,
        n_prototypes=64,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    # strict=False allows loading checkpoints that lack motif head weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} parameters on {device}")

    if model.motif_embedding_head is None:
        print("ERROR: MotifEmbeddingHead was not created. Aborting.")
        return 1

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    print("\n--- Loading Test Data ---")
    dataset = BEACONDataset(
        str(test_data_path),
        seq_length=2000,
        augment=False,
        n_slots=16,
        extract_peaks=True,
    )
    print(f"  {len(dataset)} test samples")

    # Organize samples by TF
    tf_samples = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        tf_idx = sample["tf_index"].item() if "tf_index" in sample else -1
        if 0 <= tf_idx < len(TF_NAMES) and len(tf_samples[tf_idx]) < args.max_samples:
            tf_samples[tf_idx].append(sample)

    for tf_idx in sorted(tf_samples.keys()):
        print(f"  {TF_NAMES[tf_idx]}: {len(tf_samples[tf_idx])} samples")

    # ------------------------------------------------------------------
    # 3. Collect forward-pass outputs
    # ------------------------------------------------------------------
    print("\n--- Collecting Forward-Pass Outputs ---")

    # Accumulators across all samples
    all_motif_embeddings = []   # list of [K, motif_dim]
    all_proto_assignments = []  # list of [K, n_prototypes]
    all_occupancy = []          # list of [K]
    all_tf_logits = []          # list of [K, n_tfs]
    all_motif_tf_logits = []    # list of [K, n_tfs]
    all_gt_tf_idx = []          # list of int (one per sample)

    for tf_idx in sorted(tf_samples.keys()):
        tf_name = TF_NAMES[tf_idx]
        samples = tf_samples[tf_idx]
        n = len(samples)
        print(f"\n  Processing {tf_name} ({n} samples)")

        for si, sample in enumerate(samples):
            if (si + 1) % 50 == 0 or si == 0:
                print(f"    sample {si + 1}/{n}")

            seq = sample["sequence"]  # [L, 4]

            with torch.no_grad():
                outputs = model(seq.unsqueeze(0).to(device))

            motif_emb = outputs["motif_embeddings"][0].cpu().numpy()        # [K, motif_dim]
            proto_ass = outputs["prototype_assignments"][0].cpu().numpy()    # [K, n_prototypes]
            occ = outputs["occupancy"][0, :, 0].cpu().numpy()               # [K]
            tf_log = outputs["tf_logits"][0].cpu().numpy()                   # [K, n_tfs]
            motif_tf_log = outputs["motif_tf_logits"][0].cpu().numpy()       # [K, n_tfs]

            all_motif_embeddings.append(motif_emb)
            all_proto_assignments.append(proto_ass)
            all_occupancy.append(occ)
            all_tf_logits.append(tf_log)
            all_motif_tf_logits.append(motif_tf_log)
            all_gt_tf_idx.append(tf_idx)

    n_samples = len(all_gt_tf_idx)
    n_slots = all_motif_embeddings[0].shape[0]
    motif_dim = all_motif_embeddings[0].shape[1]
    n_prototypes = all_proto_assignments[0].shape[1]
    print(f"\n  Collected {n_samples} samples, {n_slots} slots each")

    # ------------------------------------------------------------------
    # 4. Build per-slot arrays with labels
    # ------------------------------------------------------------------
    # For analyses we label each slot by the ground-truth TF of its sample.
    occ_threshold = 0.3

    slot_embeddings_all = np.concatenate(all_motif_embeddings, axis=0)      # [N*K, motif_dim]
    slot_occupancy_all = np.concatenate(all_occupancy, axis=0)              # [N*K]
    slot_proto_all = np.concatenate(all_proto_assignments, axis=0)          # [N*K, n_prototypes]
    slot_tf_logits_all = np.concatenate(all_tf_logits, axis=0)              # [N*K, n_tfs]
    slot_motif_tf_logits_all = np.concatenate(all_motif_tf_logits, axis=0)  # [N*K, n_tfs]

    # Ground-truth label for each slot = sample's TF
    slot_gt_labels = np.repeat(np.array(all_gt_tf_idx), n_slots)            # [N*K]

    # Active slot mask
    active_mask = slot_occupancy_all > occ_threshold
    n_active = int(active_mask.sum())
    print(f"  Active slots (occ > {occ_threshold}): {n_active} / {len(active_mask)}")

    # ------------------------------------------------------------------
    # 5a. Embedding clustering quality (silhouette score)
    # ------------------------------------------------------------------
    print("\n--- Analysis (a): Embedding Clustering Quality ---")
    silhouette = float("nan")
    if not HAS_SKLEARN:
        print("  sklearn not installed -- skipping silhouette score")
    elif n_active < 2:
        print("  Not enough active slots for silhouette score")
    else:
        active_embs = slot_embeddings_all[active_mask]
        active_labels = slot_gt_labels[active_mask]
        n_unique = len(np.unique(active_labels))
        if n_unique < 2:
            print("  Only one TF label among active slots -- cannot compute silhouette")
        else:
            silhouette = float(silhouette_score(active_embs, active_labels))
            print(f"  Silhouette score: {silhouette:.4f}")

    # ------------------------------------------------------------------
    # 5b. Anchor proximity per TF
    # ------------------------------------------------------------------
    print("\n--- Analysis (b): Anchor Proximity ---")
    anchors_np = model.motif_embedding_head.anchors.detach().cpu().numpy()  # [n_tfs, motif_dim]
    # Normalize anchors (same as forward pass)
    anchors_norm = anchors_np / (np.linalg.norm(anchors_np, axis=-1, keepdims=True) + 1e-8)

    anchor_distances_per_tf = {}
    for tf_idx in range(len(TF_NAMES)):
        tf_mask = active_mask & (slot_gt_labels == tf_idx)
        if tf_mask.sum() == 0:
            anchor_distances_per_tf[TF_NAMES[tf_idx]] = float("nan")
            continue
        embs = slot_embeddings_all[tf_mask]
        embs_norm = embs / (np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-8)
        dists = np.linalg.norm(embs_norm - anchors_norm[tf_idx][None, :], axis=-1)
        anchor_distances_per_tf[TF_NAMES[tf_idx]] = float(np.mean(dists))
        print(f"  {TF_NAMES[tf_idx]}: mean L2 to anchor = {anchor_distances_per_tf[TF_NAMES[tf_idx]]:.4f}")

    mean_anchor_dist = float(np.nanmean(list(anchor_distances_per_tf.values())))
    print(f"  Mean across TFs: {mean_anchor_dist:.4f}")

    # ------------------------------------------------------------------
    # 5c. Prototype utilization
    # ------------------------------------------------------------------
    print("\n--- Analysis (c): Prototype Utilization ---")
    if n_active > 0:
        active_proto = slot_proto_all[active_mask]  # [n_active, n_prototypes]
        total_mass_per_proto = active_proto.sum(axis=0)  # [n_prototypes]
        total_mass = total_mass_per_proto.sum()
        if total_mass > 0:
            frac_per_proto = total_mass_per_proto / total_mass
        else:
            frac_per_proto = np.zeros(n_prototypes)
        utilized = int((frac_per_proto > 0.01).sum())
    else:
        frac_per_proto = np.zeros(n_prototypes)
        utilized = 0

    utilization_frac = utilized / n_prototypes if n_prototypes > 0 else 0.0
    print(f"  Prototypes with >1% mass: {utilized} / {n_prototypes} ({utilization_frac:.1%})")

    # ------------------------------------------------------------------
    # 5d. TF classification accuracy (standard vs motif head)
    # ------------------------------------------------------------------
    print("\n--- Analysis (d): TF Classification Accuracy ---")
    if n_active > 0:
        active_tf_logits = slot_tf_logits_all[active_mask]
        active_motif_tf_logits = slot_motif_tf_logits_all[active_mask]
        active_gt = slot_gt_labels[active_mask]

        standard_preds = active_tf_logits.argmax(axis=1)
        motif_preds = active_motif_tf_logits.argmax(axis=1)

        standard_acc = float((standard_preds == active_gt).mean())
        motif_acc = float((motif_preds == active_gt).mean())
    else:
        standard_acc = 0.0
        motif_acc = 0.0

    print(f"  Standard head accuracy: {standard_acc:.4f}")
    print(f"  Motif head accuracy:    {motif_acc:.4f}")

    # ------------------------------------------------------------------
    # 5e. Novel motif detection
    # ------------------------------------------------------------------
    print("\n--- Analysis (e): Novel Motif Detection ---")
    if n_active > 0:
        active_embs_t = torch.from_numpy(slot_embeddings_all[active_mask]).float()
        with torch.no_grad():
            novel_mask = model.motif_embedding_head.get_novel_motifs(
                active_embs_t.unsqueeze(0)
            )  # [1, n_active]
        novel_mask_np = novel_mask[0].cpu().numpy()
        n_novel = int(novel_mask_np.sum())
        novel_frac = n_novel / n_active
    else:
        n_novel = 0
        novel_frac = 0.0

    print(f"  Novel slots: {n_novel} / {n_active} ({novel_frac:.1%})")

    # ------------------------------------------------------------------
    # 6. Print formatted results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: A2 Open-Vocabulary Motif Embedding Quality")
    print("=" * 70)

    print(f"\n{'Metric':<45} {'Value':>15}")
    print("-" * 62)

    rows = [
        ("Embedding silhouette score", f"{silhouette:.4f}"),
        ("Mean anchor L2 distance", f"{mean_anchor_dist:.4f}"),
        (f"Prototype utilization (>1% mass)", f"{utilized}/{n_prototypes} ({utilization_frac:.1%})"),
        ("Standard head TF accuracy", f"{standard_acc:.4f}"),
        ("Motif head TF accuracy", f"{motif_acc:.4f}"),
        ("Novel motif fraction", f"{novel_frac:.4f} ({n_novel}/{n_active})"),
        ("Active slots (occ > 0.3)", f"{n_active}"),
        ("Total slots evaluated", f"{len(active_mask)}"),
    ]
    for label, val in rows:
        print(f"  {label:<43} {val:>15}")

    print(f"\n{'Per-TF Anchor Distance':=<62}")
    print(f"  {'TF':<10} {'Mean L2':>10}")
    print(f"  {'-'*22}")
    for tf_name in TF_NAMES:
        d = anchor_distances_per_tf.get(tf_name, float("nan"))
        print(f"  {tf_name:<10} {d:>10.4f}")

    # ------------------------------------------------------------------
    # 7. Save results JSON
    # ------------------------------------------------------------------
    save_data = {
        "validation": "A2_motif_embedding_quality",
        "checkpoint": str(checkpoint_path),
        "test_data": str(test_data_path),
        "max_samples_per_tf": args.max_samples,
        "n_samples": n_samples,
        "n_slots": n_slots,
        "motif_dim": motif_dim,
        "n_prototypes": n_prototypes,
        "occ_threshold": occ_threshold,
        "n_active_slots": n_active,
        "silhouette_score": silhouette if not np.isnan(silhouette) else None,
        "mean_anchor_distance": mean_anchor_dist,
        "anchor_distance_per_tf": {
            k: (v if not np.isnan(v) else None)
            for k, v in anchor_distances_per_tf.items()
        },
        "prototype_utilization": {
            "utilized": utilized,
            "total": n_prototypes,
            "fraction": utilization_frac,
        },
        "tf_accuracy": {
            "standard_head": standard_acc,
            "motif_head": motif_acc,
        },
        "novel_motif_detection": {
            "n_novel": n_novel,
            "n_active": n_active,
            "fraction": novel_frac,
        },
    }

    results_path = output_dir / "a2_motif_embedding_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 8. Create matplotlib visualization (2x2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors_per_tf = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b",
                     "#76b7b2", "#edc948", "#b07aa1"]

    # --- Panel A: 2D t-SNE of motif embeddings colored by TF ---
    ax = axes[0, 0]
    if HAS_SKLEARN and n_active >= 2:
        active_embs = slot_embeddings_all[active_mask]
        active_labels = slot_gt_labels[active_mask]
        perplexity = min(30, max(5, n_active // 4))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed)
        coords = tsne.fit_transform(active_embs)
        for tf_idx in range(len(TF_NAMES)):
            mask_tf = active_labels == tf_idx
            if mask_tf.sum() > 0:
                ax.scatter(
                    coords[mask_tf, 0], coords[mask_tf, 1],
                    c=colors_per_tf[tf_idx], label=TF_NAMES[tf_idx],
                    s=12, alpha=0.6, edgecolors="none",
                )
        ax.legend(fontsize=7, markerscale=2, loc="best")
        ax.set_title("t-SNE of Motif Embeddings (active slots)")
    else:
        ax.text(
            0.5, 0.5,
            "sklearn not installed\nor insufficient data",
            transform=ax.transAxes, ha="center", va="center", fontsize=12,
        )
        ax.set_title("t-SNE (unavailable)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # --- Panel B: Anchor distance per TF (bar chart) ---
    ax = axes[0, 1]
    tf_names_plot = []
    tf_dists_plot = []
    tf_colors_plot = []
    for i, tf_name in enumerate(TF_NAMES):
        d = anchor_distances_per_tf.get(tf_name, float("nan"))
        if not np.isnan(d):
            tf_names_plot.append(tf_name)
            tf_dists_plot.append(d)
            tf_colors_plot.append(colors_per_tf[i])

    if tf_dists_plot:
        x_pos = np.arange(len(tf_names_plot))
        bars = ax.bar(x_pos, tf_dists_plot, color=tf_colors_plot)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tf_names_plot, rotation=45, ha="right")
        for bar, val in zip(bars, tf_dists_plot):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(tf_dists_plot) * 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_ylabel("Mean L2 Distance to Anchor")
    ax.set_title("Anchor Proximity per TF")

    # --- Panel C: Prototype utilization histogram ---
    ax = axes[1, 0]
    if n_prototypes > 0:
        sorted_frac = np.sort(frac_per_proto)[::-1]
        ax.bar(np.arange(n_prototypes), sorted_frac, color="#4e79a7", width=1.0)
        ax.axhline(y=0.01, color="red", linestyle="--", linewidth=1.0, label="1% threshold")
        ax.set_xlabel("Prototype (sorted by usage)")
        ax.set_ylabel("Fraction of Total Assignment Mass")
        ax.legend(fontsize=8)
    ax.set_title(f"Prototype Utilization ({utilized}/{n_prototypes} active)")

    # --- Panel D: TF accuracy comparison (standard vs motif head) ---
    ax = axes[1, 1]
    acc_labels = ["Standard Head", "Motif Head"]
    acc_vals = [standard_acc, motif_acc]
    bar_colors = ["#4e79a7", "#e15759"]
    x_pos = np.arange(len(acc_labels))
    bars = ax.bar(x_pos, acc_vals, color=bar_colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(acc_labels, fontsize=10)
    ax.set_ylabel("TF Classification Accuracy")
    ax.set_ylim(0, max(max(acc_vals) * 1.3, 0.1))
    for bar, val in zip(bars, acc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("TF Accuracy: Standard vs Motif Head")

    fig.suptitle(
        "A2 Validation: Open-Vocabulary Motif Embedding Quality",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fig_path = output_dir / "a2_motif_embedding_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
