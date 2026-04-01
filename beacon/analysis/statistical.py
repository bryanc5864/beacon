"""
Statistical utilities for BEACON evaluation.

Provides bootstrap confidence intervals, significance tests (Wilcoxon,
Mann-Whitney), effect sizes (Cohen's d), and multiple comparison correction.
"""

import numpy as np
from scipy.stats import norm, wilcoxon, mannwhitneyu


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, statistic=np.mean, seed=42):
    """Compute BCa (bias-corrected and accelerated) bootstrap confidence interval.

    Args:
        data: 1-D array of observations.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        statistic: Function to compute the statistic (default np.mean).
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: mean, ci_lower, ci_upper, ci_level, n_bootstrap, n_samples.
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)
    observed = statistic(data)

    # Bootstrap replicates
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        boot_stats[b] = statistic(data[rng.randint(0, n, size=n)])

    # Bias correction factor
    z0 = norm.ppf(np.clip(np.mean(boot_stats < observed), 1e-10, 1 - 1e-10))

    # Acceleration factor via jackknife
    jackknife_stats = np.empty(n)
    for i in range(n):
        jackknife_stats[i] = statistic(np.delete(data, i))
    jack_mean = jackknife_stats.mean()
    diff = jack_mean - jackknife_stats
    denom = np.sum(diff ** 2) ** 1.5
    a = np.sum(diff ** 3) / (6 * denom) if denom > 0 else 0.0

    # Adjusted percentiles
    alpha = (1 - ci) / 2
    z_lo, z_hi = norm.ppf(alpha), norm.ppf(1 - alpha)

    def _adj(z_alpha):
        num = z0 + z_alpha
        den = 1 - a * num
        return norm.cdf(z0 + num / den) if den != 0 else 0.5

    p_lower = np.clip(_adj(z_lo), 0.001, 0.999)
    p_upper = np.clip(_adj(z_hi), 0.001, 0.999)

    return {
        "mean": float(observed),
        "ci_lower": float(np.percentile(boot_stats, p_lower * 100)),
        "ci_upper": float(np.percentile(boot_stats, p_upper * 100)),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def paired_wilcoxon(x, y, alternative='greater'):
    """Paired Wilcoxon signed-rank test.

    Args:
        x, y: Paired observations (same length).
        alternative: 'greater' tests x > y.

    Returns:
        dict with statistic, pvalue.
    """
    try:
        stat, pval = wilcoxon(x, y, alternative=alternative)
    except ValueError:
        stat, pval = 0.0, 1.0
    return {"statistic": float(stat), "pvalue": float(pval)}


def mann_whitney(x, y, alternative='greater'):
    """Mann-Whitney U test for independent samples.

    Returns:
        dict with statistic, pvalue.
    """
    try:
        stat, pval = mannwhitneyu(x, y, alternative=alternative)
    except ValueError:
        stat, pval = 0.0, 1.0
    return {"statistic": float(stat), "pvalue": float(pval)}


def bonferroni(pvalue, n_tests):
    """Bonferroni-corrected p-value."""
    return min(float(pvalue) * n_tests, 1.0)


def cohens_d(x, y):
    """Cohen's d effect size for paired samples."""
    diff = np.asarray(x) - np.asarray(y)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
