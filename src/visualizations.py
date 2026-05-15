"""
visualizations.py
=================
Professional, presentation-ready visualisation module for the
E-commerce CRO A/B Test Analysis.

All charts follow a consistent visual language:
  - Clean, minimal grid (seaborn ``whitegrid`` theme)
  - Colour-blind-safe palette (Control = steel blue, Treatment = coral)
  - Saved as 300 dpi PNG for crisp rendering in reports and slides

Author : Senior Data Analyst
Project: E-commerce Conversion Rate Optimisation — A/B Test Analysis
"""

import logging
import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------
PALETTE = {
    "control":   "#4878CF",   # steel blue
    "treatment": "#E87676",   # muted coral / salmon
}
FONT_FAMILY  = "DejaVu Sans"
TITLE_SIZE   = 16
LABEL_SIZE   = 13
TICK_SIZE    = 11
ANNOTATION_SIZE = 12
DPI          = 300

# Apply global style once at import time
sns.set_theme(style="whitegrid", font=FONT_FAMILY)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    TITLE_SIZE,
    "axes.labelsize":    LABEL_SIZE,
    "xtick.labelsize":   TICK_SIZE,
    "ytick.labelsize":   TICK_SIZE,
    "figure.dpi":        80,   # screen preview; saved at DPI=300
})

MetricsDict = Dict[str, Union[float, int]]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create *path* (and any missing parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_conversion_rates(metrics: MetricsDict, save_path: str) -> str:
    """Create and save a bar chart of Control vs. Treatment conversion rates.

    The chart includes:
      - Colour-coded bars for each group.
      - Asymmetric error bars representing the 95 % CI for each group's
        conversion rate (``ci_control_lo/hi`` and ``ci_treat_lo/hi``).
      - Data labels above each bar showing the exact conversion rate (%).
      - A horizontal dashed line at the control baseline for visual reference.
      - A light annotation band showing the difference and its 95 % CI.
      - An inset text box with the p-value and statistical verdict.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary as returned by
        ``stats_analysis.calculate_confidence_intervals``.
    save_path : str
        Directory (e.g. ``results/``) where the PNG will be saved.

    Returns
    -------
    str
        Full path to the saved PNG file.
    """
    _ensure_dir(save_path)

    # ── unpack metrics ───────────────────────────────────────────────────────
    r_ctrl  = metrics["rate_control"]
    r_trt   = metrics["rate_treatment"]
    p_value = metrics["p_value"]
    z_stat  = metrics["z_stat"]
    n_ctrl  = metrics["n_control"]
    n_trt   = metrics["n_treatment"]

    # Asymmetric error bars for individual group CIs
    err_ctrl = np.array([
        [r_ctrl - metrics["ci_control_lo"]],
        [metrics["ci_control_hi"] - r_ctrl],
    ])
    err_trt = np.array([
        [r_trt - metrics["ci_treat_lo"]],
        [metrics["ci_treat_hi"] - r_trt],
    ])

    # ── figure layout ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("white")

    groups   = ["Control\n(Old Page)", "Treatment\n(New Page)"]
    rates    = [r_ctrl, r_trt]
    errors   = [err_ctrl, err_trt]
    colours  = [PALETTE["control"], PALETTE["treatment"]]
    x_pos    = [0, 1]

    # ── bars ─────────────────────────────────────────────────────────────────
    bars = ax.bar(
        x_pos, rates,
        color=colours,
        width=0.45,
        zorder=3,
        alpha=0.88,
        linewidth=1.2,
        edgecolor="white",
    )

    # ── error bars ───────────────────────────────────────────────────────────
    for xi, rate, err in zip(x_pos, rates, errors):
        ax.errorbar(
            xi, rate,
            yerr=err,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.8,
            capsize=7,
            capthick=1.8,
            zorder=5,
        )

    # ── data labels above bars ───────────────────────────────────────────────
    for xi, rate, bar in zip(x_pos, rates, bars):
        ax.text(
            xi, rate + 0.0028,
            f"{rate * 100:.3f}%",
            ha="center", va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color="#222222",
            zorder=6,
        )

    # ── control baseline dashed line ─────────────────────────────────────────
    ax.axhline(
        r_ctrl,
        color=PALETTE["control"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.55,
        zorder=2,
        label=f"Control baseline ({r_ctrl * 100:.3f}%)",
    )

    # ── difference annotation band ───────────────────────────────────────────
    diff = metrics["rate_diff"]
    ci_l = metrics["ci_lower"]
    ci_u = metrics["ci_upper"]

    mid_x = 1.55
    ax.annotate(
        "",
        xy=(mid_x, r_trt), xytext=(mid_x, r_ctrl),
        arrowprops=dict(
            arrowstyle="<->",
            color="#666666",
            lw=1.5,
        ),
        annotation_clip=False,
    )
    ax.text(
        mid_x + 0.07,
        (r_ctrl + r_trt) / 2,
        f"Δ = {diff * 100:+.3f}%\n95% CI\n[{ci_l * 100:.3f}%,\n {ci_u * 100:.3f}%]",
        ha="left", va="center",
        fontsize=9.5,
        color="#444444",
        clip_on=False,
    )

    # ── p-value verdict box ───────────────────────────────────────────────────
    alpha_level = 0.05
    verdict     = "Fail to Reject H₀" if p_value >= alpha_level else "Reject H₀"
    sig_colour  = "#CC0000" if p_value < alpha_level else "#2A7A2A"
    box_text    = (
        f"Two-Proportions Z-Test\n"
        f"Z-stat : {z_stat:+.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"α = {alpha_level}\n"
        f"Verdict: {verdict}"
    )
    props = dict(boxstyle="round,pad=0.55", facecolor="#F5F5F5",
                 edgecolor=sig_colour, linewidth=1.6, alpha=0.92)
    ax.text(
        0.02, 0.97, box_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=props,
        color=sig_colour,
        fontfamily="monospace",
    )

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))

    y_max = max(rates) + max(r_ctrl - metrics["ci_control_lo"],
                              r_trt  - metrics["ci_treat_lo"]) * 4
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.55, 2.1)

    ax.set_ylabel("Conversion Rate  (%)", labelpad=10)
    ax.set_xlabel("")

    ax.set_title(
        "A/B Test — Conversion Rate: Control vs. Treatment\n"
        f"(n_control = {n_ctrl:,}  |  n_treatment = {n_trt:,})",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=14,
    )

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=PALETTE["control"],   label=f"Control — Old Page  ({r_ctrl * 100:.3f}%)"),
        mpatches.Patch(color=PALETTE["treatment"], label=f"Treatment — New Page  ({r_trt * 100:.3f}%)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── save ─────────────────────────────────────────────────────────────────
    output_file = os.path.join(save_path, "conversion_rates_comparison.png")
    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Plot saved → %s", output_file)
    return output_file


def plot_sample_sizes(metrics: MetricsDict, save_path: str) -> str:
    """Create and save a horizontal bar chart showing group sample sizes.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from ``calculate_confidence_intervals``.
    save_path : str
        Directory where the PNG will be saved.

    Returns
    -------
    str
        Full path to the saved PNG file.
    """
    _ensure_dir(save_path)

    n_ctrl = metrics["n_control"]
    n_trt  = metrics["n_treatment"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("white")

    groups = ["Control\n(Old Page)", "Treatment\n(New Page)"]
    values = [n_ctrl, n_trt]
    colours = [PALETTE["control"], PALETTE["treatment"]]

    bars = ax.barh(groups, values, color=colours, height=0.45, alpha=0.88,
                   edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center", ha="left",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color="#222222",
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Number of Users", labelpad=8)
    ax.set_title("Sample Size per Group", fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    output_file = os.path.join(save_path, "sample_sizes.png")
    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Sample-size plot saved → %s", output_file)
    return output_file


def plot_p_value_distribution(metrics: MetricsDict, save_path: str) -> str:
    """Visualise where the observed p-value falls on the standard-normal distribution.

    This plot provides an intuitive explanation of the hypothesis test:
    it shows the full Z-distribution, shades the rejection regions at α=0.05,
    and marks the observed Z-statistic.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from ``calculate_confidence_intervals``.
    save_path : str
        Directory where the PNG will be saved.

    Returns
    -------
    str
        Full path to the saved PNG file.
    """
    from scipy.stats import norm as _norm

    _ensure_dir(save_path)

    z_stat  = metrics["z_stat"]
    p_value = metrics["p_value"]
    alpha   = 0.05
    z_crit  = _norm.ppf(1 - alpha / 2)   # ≈ 1.96

    x = np.linspace(-4.5, 4.5, 800)
    y = _norm.pdf(x)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    # Full distribution
    ax.plot(x, y, color="#333333", linewidth=2, zorder=4)

    # Rejection regions (α/2 on each tail)
    ax.fill_between(x, y, where=(x <= -z_crit), color="#E87676", alpha=0.35,
                    label=f"Rejection region (α/2 = {alpha/2})", zorder=3)
    ax.fill_between(x, y, where=(x >=  z_crit), color="#E87676", alpha=0.35, zorder=3)

    # Area under curve to the right of |z_stat| (p-value region)
    ax.fill_between(x, y, where=(x >= abs(z_stat)),  color="#FFD700", alpha=0.45,
                    label=f"p-value region ({p_value:.4f})", zorder=3)
    ax.fill_between(x, y, where=(x <= -abs(z_stat)), color="#FFD700", alpha=0.45, zorder=3)

    # Critical value lines
    for zc in [-z_crit, z_crit]:
        ax.axvline(zc, color="#E87676", linewidth=1.4, linestyle="--", zorder=5)
        ax.text(zc, _norm.pdf(zc) + 0.007, f"±{z_crit:.2f}",
                ha="center", fontsize=9.5, color="#CC0000")

    # Observed Z-stat line
    ax.axvline(z_stat, color="#1A6B1A", linewidth=2.0, linestyle="-", zorder=6,
               label=f"Observed Z = {z_stat:.4f}")
    ax.text(z_stat, _norm.pdf(z_stat) + 0.015, f"Z = {z_stat:.4f}",
            ha="center", fontsize=10, color="#1A6B1A", fontweight="bold")

    verdict = "Fail to Reject H₀" if p_value >= alpha else "Reject H₀"
    verdict_color = "#2A7A2A" if p_value >= alpha else "#CC0000"
    ax.set_title(
        f"Standard Normal Distribution — Hypothesis Test Visualisation\n"
        f"p-value = {p_value:.4f}  |  Verdict: {verdict}",
        fontsize=TITLE_SIZE, fontweight="bold", pad=14,
        color=verdict_color,
    )

    ax.set_xlabel("Z-score", labelpad=8)
    ax.set_ylabel("Probability Density", labelpad=8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-4.5, 4.5)

    output_file = os.path.join(save_path, "z_distribution.png")
    fig.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Z-distribution plot saved → %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_prep      import load_data, clean_data
    from src.stats_analysis import calculate_confidence_intervals

    _df      = clean_data(load_data("data/raw/ab_data.csv"))
    _metrics = calculate_confidence_intervals(_df)

    plot_conversion_rates(_metrics, "results/")
    plot_sample_sizes(_metrics, "results/")
    plot_p_value_distribution(_metrics, "results/")
    print("All plots saved to results/")
