"""
BEACON Analysis Utilities

Tools for evaluation, statistical testing, profile metrics,
motif extraction, variant scoring, and cooperative binding analysis.
"""

from .motif_extraction import PerSlotMotifExtractor
from .variant_scoring import VariantScorer, VariantEffect
from .evaluation import (
    load_model, load_named_model, load_test_data, get_test_path,
    run_model_batch, per_sample_pearson, hungarian_accuracy_per_site,
    hungarian_confusion_matrix, gc_content, json_serialize,
    DATASET_CONFIGS, MODEL_CHECKPOINTS, ABLATION_CHECKPOINTS, TF_FAMILIES,
    BASE_DIR,
)
from .statistical import bootstrap_ci, paired_wilcoxon, mann_whitney, bonferroni, cohens_d
from .profile_metrics import (
    jensen_shannon_divergence, multinomial_nll,
    per_sample_jsd, per_sample_mnll, site_auprc,
)
from .cooperative import (
    attention_pair_analysis, slot_utilization_analysis,
    position_distribution_analysis, profile_shape_similarity,
    COOPERATIVE_PAIRS,
)
from .position_accuracy import (
    hungarian_position_errors, per_tf_position_stats,
    position_error_by_signal, position_error_by_gc,
    systematic_offset_analysis,
)
from .saliency import (
    compute_gradient_x_input, attention_vs_gradient_correlation,
    gradient_enrichment_at_motifs,
)
from .slot_embeddings import (
    extract_slot_embeddings, compute_rsa,
    cross_seed_embedding_consistency, linear_probe_cv,
    tf_family_clustering_metrics,
)
from .denovo_motifs import (
    compute_per_tf_gradient_importance, extract_denovo_pwm,
    compare_to_jaspar, full_motif_discovery_pipeline,
)
from .throughput import benchmark_inference, benchmark_memory, count_parameters
from .cross_cell import (
    motif_level_transfer, attention_pattern_transfer,
    cell_specific_vs_shared_features,
)

__all__ = [
    # Motif / variant
    "PerSlotMotifExtractor",
    "VariantScorer",
    "VariantEffect",
    # Evaluation
    "load_model", "load_named_model", "load_test_data", "get_test_path",
    "run_model_batch", "per_sample_pearson",
    "hungarian_accuracy_per_site", "hungarian_confusion_matrix",
    "gc_content", "json_serialize",
    "DATASET_CONFIGS", "MODEL_CHECKPOINTS", "ABLATION_CHECKPOINTS",
    "TF_FAMILIES", "BASE_DIR",
    # Statistical
    "bootstrap_ci", "paired_wilcoxon", "mann_whitney", "bonferroni", "cohens_d",
    # Profile metrics
    "jensen_shannon_divergence", "multinomial_nll",
    "per_sample_jsd", "per_sample_mnll", "site_auprc",
    # Cooperative
    "attention_pair_analysis", "slot_utilization_analysis",
    "position_distribution_analysis", "profile_shape_similarity",
    "COOPERATIVE_PAIRS",
    # Position accuracy
    "hungarian_position_errors", "per_tf_position_stats",
    "position_error_by_signal", "position_error_by_gc",
    "systematic_offset_analysis",
    # Saliency
    "compute_gradient_x_input", "attention_vs_gradient_correlation",
    "gradient_enrichment_at_motifs",
    # Slot embeddings
    "extract_slot_embeddings", "compute_rsa",
    "cross_seed_embedding_consistency", "linear_probe_cv",
    "tf_family_clustering_metrics",
    # De novo motifs
    "compute_per_tf_gradient_importance", "extract_denovo_pwm",
    "compare_to_jaspar", "full_motif_discovery_pipeline",
    # Throughput
    "benchmark_inference", "benchmark_memory", "count_parameters",
    # Cross-cell
    "motif_level_transfer", "attention_pattern_transfer",
    "cell_specific_vs_shared_features",
]
