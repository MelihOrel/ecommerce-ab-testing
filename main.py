"""
main.py
=======
Entry-point for the E-commerce Conversion Rate Optimisation A/B Test Analysis.

Execution pipeline
------------------
1. Load raw data from  data/raw/ab_data.csv
2. Clean & validate the DataFrame  (alignment filter + duplicate removal)
3. Run Two-Proportions Z-Test and compute 95 % Confidence Intervals
4. Generate and save publication-quality charts to  results/
5. Print a formatted "Executive Report" to the terminal

Usage
-----
    python main.py
    python main.py --data data/raw/ab_data.csv --results results/ --alpha 0.05

Author : Senior Data Analyst
Project: E-commerce Conversion Rate Optimisation — A/B Test Analysis
"""

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the project root is on the Python path when run directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_prep      import load_data, clean_data
from src.stats_analysis import calculate_confidence_intervals
from src.visualizations import (
    plot_conversion_rates,
    plot_sample_sizes,
    plot_p_value_distribution,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on Windows)
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init(autoreset=True)
    _GREEN  = colorama.Fore.GREEN
    _RED    = colorama.Fore.RED
    _YELLOW = colorama.Fore.YELLOW
    _CYAN   = colorama.Fore.CYAN
    _BOLD   = colorama.Style.BRIGHT
    _RESET  = colorama.Style.RESET_ALL
except ImportError:
    _GREEN = _RED = _YELLOW = _CYAN = _BOLD = _RESET = ""


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _divider(char: str = "═", width: int = 70) -> str:
    return char * width


def _section(title: str) -> str:
    pad = max(0, 70 - len(title) - 4)
    return f"\n{_BOLD}{'─' * 2}  {title}  {'─' * pad}{_RESET}"


def _kv(label: str, value: str, colour: str = "") -> str:
    return f"  {label:<38s}{colour}{value}{_RESET}"


def print_executive_report(metrics: dict, alpha: float, saved_files: list) -> None:
    """Print a formatted Executive Report to stdout.

    The report is deliberately written in plain language suitable for
    non-technical stakeholders (Product Managers, C-suite).

    Parameters
    ----------
    metrics     : dict  — output of ``calculate_confidence_intervals``
    alpha       : float — significance level used
    saved_files : list  — list of paths to saved output files
    """
    p_value    = metrics["p_value"]
    z_stat     = metrics["z_stat"]
    r_ctrl     = metrics["rate_control"]
    r_trt      = metrics["rate_treatment"]
    diff       = metrics["rate_diff"]
    ci_l       = metrics["ci_lower"]
    ci_u       = metrics["ci_upper"]
    n_ctrl     = metrics["n_control"]
    n_trt      = metrics["n_treatment"]
    pooled     = metrics["pooled_prob"]
    ci_level   = metrics["ci_level"]

    reject     = p_value < alpha
    verdict    = "✅  REJECT the Null Hypothesis"  if reject else "❌  FAIL TO REJECT the Null Hypothesis"
    v_colour   = _RED   if reject else _GREEN

    print()
    print(_BOLD + _CYAN + _divider("═") + _RESET)
    print(_BOLD + _CYAN + "  E-COMMERCE CRO A/B TEST — EXECUTIVE REPORT".center(70) + _RESET)
    print(_BOLD + _CYAN + _divider("═") + _RESET)

    # ── Experiment Overview ────────────────────────────────────────────────
    print(_section("EXPERIMENT OVERVIEW"))
    print(_kv("Hypothesis (H₀):",   "p_treatment = p_control  (no difference)"))
    print(_kv("Hypothesis (H₁):",   "p_treatment ≠ p_control  (two-sided)"))
    print(_kv("Test type:",          "Two-Proportions Z-Test"))
    print(_kv("Significance level:", f"α = {alpha}  ({ci_level}% confidence)"))

    # ── Sample Statistics ─────────────────────────────────────────────────
    print(_section("SAMPLE STATISTICS"))
    print(_kv("Control group  — sample size:", f"{n_ctrl:>10,}"))
    print(_kv("Treatment group — sample size:", f"{n_trt:>10,}"))
    print(_kv("Total users analysed:",          f"{n_ctrl + n_trt:>10,}"))
    print(_kv("Pooled conversion probability:", f"{pooled:.6f}  ({pooled * 100:.4f}%)"))

    # ── Conversion Rates ─────────────────────────────────────────────────
    print(_section("CONVERSION RATES"))
    print(_kv("Control   (Old Page)  conversion rate:", f"{r_ctrl * 100:.4f}%"))
    print(_kv("Treatment (New Page)  conversion rate:", f"{r_trt  * 100:.4f}%"))
    print(_kv("Observed lift  (Δ = treatment − control):",
              f"{diff * 100:+.4f}%",
              _RED if diff < 0 else _GREEN))
    print(_kv("Relative lift:",
              f"{(diff / r_ctrl) * 100:+.2f}%",
              _RED if diff < 0 else _GREEN))

    # ── Statistical Results ───────────────────────────────────────────────
    print(_section("STATISTICAL RESULTS"))
    print(_kv("Z-statistic:",                       f"{z_stat:>+10.4f}"))
    print(_kv("p-value (two-sided):",               f"{p_value:>10.4f}"))
    print(_kv(f"{ci_level}% CI for Δ (lower):",     f"{ci_l * 100:>+.4f}%"))
    print(_kv(f"{ci_level}% CI for Δ (upper):",     f"{ci_u * 100:>+.4f}%"))

    # ── Verdict ──────────────────────────────────────────────────────────
    print()
    print(_divider("─"))
    print(f"  {_BOLD}STATISTICAL VERDICT:{_RESET}  {v_colour}{_BOLD}{verdict}{_RESET}")
    print(_divider("─"))

    # ── Business Recommendation ───────────────────────────────────────────
    print(_section("BUSINESS RECOMMENDATION"))
    if not reject:
        print(
            f"\n  The new landing page did NOT produce a statistically significant\n"
            f"  change in conversion rate at the α = {alpha} level.\n\n"
            f"  • The observed difference in conversion rates is {diff * 100:+.4f}%,\n"
            f"    which is indistinguishable from random noise given the data.\n\n"
            f"  • The {ci_level}% confidence interval for the true difference is\n"
            f"    [{ci_l * 100:.4f}%, {ci_u * 100:.4f}%], which straddles zero,\n"
            f"    meaning we cannot rule out the absence of any effect.\n\n"
            f"  📌 RECOMMENDATION: Do NOT roll out the new page at this time.\n"
            f"     The current evidence does not support the hypothesis that\n"
            f"     the new design improves conversion.  Consider:\n"
            f"       – Investigating UX/copy changes for a future iteration.\n"
            f"       – Running the experiment longer to increase statistical power.\n"
            f"       – Segmenting results by device, channel, or user cohort.\n"
        )
    else:
        direction = "increased" if diff > 0 else "decreased"
        print(
            f"\n  The new landing page produced a statistically significant\n"
            f"  {direction} in conversion rate at the α = {alpha} level.\n\n"
            f"  • The observed lift is {diff * 100:+.4f}% (relative: {(diff/r_ctrl)*100:+.2f}%).\n\n"
            f"  • The {ci_level}% confidence interval [{ci_l * 100:.4f}%, {ci_u * 100:.4f}%]\n"
            f"    does not include zero, confirming statistical significance.\n\n"
            f"  📌 RECOMMENDATION: {'Roll out' if diff > 0 else 'Do NOT roll out'} "
            f"the new page to all users.\n"
        )

    # ── Output files ─────────────────────────────────────────────────────
    print(_section("OUTPUT FILES"))
    for f in saved_files:
        print(f"  📄  {f}")

    print()
    print(_BOLD + _CYAN + _divider("═") + _RESET)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E-commerce CRO A/B Test Analysis pipeline."
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "ab_data.csv"),
        help="Path to the raw CSV file  (default: data/raw/ab_data.csv)",
    )
    parser.add_argument(
        "--results",
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory for saved charts  (default: results/)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level  (default: 0.05)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0   = time.time()

    logger.info("━━━  A/B Test Analysis pipeline starting  ━━━")

    # ── Step 1: Load ─────────────────────────────────────────────────────
    logger.info("[1/4] Loading raw data…")
    df_raw = load_data(args.data)

    # ── Step 2: Clean ─────────────────────────────────────────────────────
    logger.info("[2/4] Cleaning data…")
    df_clean = clean_data(df_raw)

    # ── Step 3: Statistical analysis ─────────────────────────────────────
    logger.info("[3/4] Running statistical analysis…")
    metrics = calculate_confidence_intervals(df_clean, alpha=args.alpha)

    # ── Step 4: Visualisations ────────────────────────────────────────────
    logger.info("[4/4] Generating visualisations…")
    saved_files = []
    saved_files.append(plot_conversion_rates(metrics,       args.results))
    saved_files.append(plot_sample_sizes(metrics,           args.results))
    saved_files.append(plot_p_value_distribution(metrics,   args.results))

    # ── Executive Report ─────────────────────────────────────────────────
    print_executive_report(metrics, alpha=args.alpha, saved_files=saved_files)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.2f seconds.", elapsed)


if __name__ == "__main__":
    main()
