"""
stats_analysis.py
=================
Core statistical testing module for the E-commerce CRO A/B Test Analysis.

This module implements a rigorous, two-sided Two-Proportions Z-Test to
evaluate whether the new landing page produces a statistically significant
change in conversion rate relative to the control (old) page.

Why a Two-Proportions Z-Test?
------------------------------
* The outcome variable ``converted`` is *binary* (Bernoulli-distributed).
* Both groups have very large sample sizes (N > 10 000), satisfying the
  Central Limit Theorem conditions for the normal approximation to hold.
* We test the difference in *proportions* (conversion rates), which is
  exactly the estimand of a Z-test for two proportions.
* The test is two-sided because we are agnostic about the direction of any
  effect — we want to detect both improvements *and* regressions.

Hypotheses
----------
  H₀ : p_treatment − p_control  = 0   (no difference in conversion rates)
  H₁ : p_treatment − p_control ≠ 0   (conversion rates differ)

Significance level: α = 0.05

Author : Senior Data Analyst
Project: E-commerce Conversion Rate Optimisation — A/B Test Analysis
"""

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Type alias for the metrics dict
MetricsDict = Dict[str, Union[float, int]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perform_z_test(df: pd.DataFrame) -> MetricsDict:
    """Perform a Two-Proportions Z-Test on the A/B test data.

    The function calculates per-group conversion counts and sample sizes,
    derives the pooled conversion probability under H₀, and uses
    ``statsmodels.stats.proportion.proportions_ztest`` to obtain the
    Z-statistic and two-sided p-value.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame containing at minimum the columns
        ``group`` (values: ``'control'`` / ``'treatment'``) and
        ``converted`` (values: ``0`` / ``1``).

    Returns
    -------
    dict
        A metrics dictionary with the following keys:

        ==================  ================================================
        Key                 Description
        ==================  ================================================
        ``n_control``       Sample size of the control group
        ``n_treatment``     Sample size of the treatment group
        ``conv_control``    Number of conversions in the control group
        ``conv_treatment``  Number of conversions in the treatment group
        ``rate_control``    Conversion rate of the control group  (0–1)
        ``rate_treatment``  Conversion rate of the treatment group (0–1)
        ``pooled_prob``     Pooled conversion probability under H₀
        ``z_stat``          Z-statistic of the test
        ``p_value``         Two-sided p-value of the test
        ``rate_diff``       rate_treatment − rate_control  (observed lift)
        ==================  ================================================

    Raises
    ------
    ValueError
        If either group is absent from the DataFrame.
    """
    logger.info("Running Two-Proportions Z-Test…")

    # ── per-group aggregation ────────────────────────────────────────────────
    groups = df.groupby("group")["converted"]

    control_group   = groups.get_group("control")
    treatment_group = groups.get_group("treatment")

    n_control   = len(control_group)
    n_treatment = len(treatment_group)

    conv_control   = int(control_group.sum())
    conv_treatment = int(treatment_group.sum())

    rate_control   = conv_control   / n_control
    rate_treatment = conv_treatment / n_treatment

    # ── pooled probability (used internally by statsmodels) ─────────────────
    pooled_prob = (conv_control + conv_treatment) / (n_control + n_treatment)

    # ── statsmodels proportions_ztest ────────────────────────────────────────
    # We pass [treatment, control] so a positive z-stat means treatment > control.
    count  = np.array([conv_treatment, conv_control])
    nobs   = np.array([n_treatment,   n_control])

    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    rate_diff = rate_treatment - rate_control

    metrics: MetricsDict = {
        "n_control":       n_control,
        "n_treatment":     n_treatment,
        "conv_control":    conv_control,
        "conv_treatment":  conv_treatment,
        "rate_control":    rate_control,
        "rate_treatment":  rate_treatment,
        "pooled_prob":     pooled_prob,
        "z_stat":          z_stat,
        "p_value":         p_value,
        "rate_diff":       rate_diff,
    }

    logger.info(
        "Z-Test complete — Z-stat: %.4f | p-value: %.4f | "
        "control rate: %.4f | treatment rate: %.4f",
        z_stat, p_value, rate_control, rate_treatment,
    )

    return metrics


def calculate_confidence_intervals(
    df: pd.DataFrame, alpha: float = 0.05
) -> MetricsDict:
    """Calculate the (1−α) confidence interval for the difference in conversion rates.

    The CI is computed via the *normal approximation* method, consistent with
    the Z-test framework applied in :func:`perform_z_test`.

    The standard error of the difference in proportions is:

    .. math::

        SE_{diff} = \\sqrt{\\frac{p_c (1-p_c)}{n_c} + \\frac{p_t (1-p_t)}{n_t}}

    And the two-sided CI is:

    .. math::

        (p_t - p_c) \\pm z_{\\alpha/2} \\cdot SE_{diff}

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (same format as for :func:`perform_z_test`).
    alpha : float, optional
        Significance level.  Defaults to ``0.05`` (95 % CI).

    Returns
    -------
    dict
        Extends the Z-test metrics dictionary with:

        =================  =============================================
        Key                Description
        =================  =============================================
        ``ci_lower``       Lower bound of the CI for rate_diff
        ``ci_upper``       Upper bound of the CI for rate_diff
        ``ci_level``       Confidence level expressed as a percentage
        ``se_diff``        Standard error of the difference
        ``ci_control_lo``  Lower bound of the CI for control rate alone
        ``ci_control_hi``  Upper bound of the CI for control rate alone
        ``ci_treat_lo``    Lower bound of the CI for treatment rate alone
        ``ci_treat_hi``    Upper bound of the CI for treatment rate alone
        =================  =============================================
    """
    logger.info("Calculating %d%% confidence intervals…", int((1 - alpha) * 100))

    # ── base metrics from Z-test ─────────────────────────────────────────────
    base = perform_z_test(df)

    p_c  = base["rate_control"]
    p_t  = base["rate_treatment"]
    n_c  = base["n_control"]
    n_t  = base["n_treatment"]

    # ── SE of the difference ─────────────────────────────────────────────────
    se_diff = np.sqrt((p_c * (1 - p_c) / n_c) + (p_t * (1 - p_t) / n_t))

    # ── z critical value ─────────────────────────────────────────────────────
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)

    diff    = base["rate_diff"]
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    # ── individual group CIs (Wilson / normal approximation via statsmodels) ─
    ci_c_lo, ci_c_hi = proportion_confint(
        base["conv_control"],   n_c, alpha=alpha, method="normal"
    )
    ci_t_lo, ci_t_hi = proportion_confint(
        base["conv_treatment"], n_t, alpha=alpha, method="normal"
    )

    extra: MetricsDict = {
        "ci_lower":      ci_low,
        "ci_upper":      ci_high,
        "ci_level":      int((1 - alpha) * 100),
        "se_diff":       se_diff,
        "ci_control_lo": ci_c_lo,
        "ci_control_hi": ci_c_hi,
        "ci_treat_lo":   ci_t_lo,
        "ci_treat_hi":   ci_t_hi,
    }

    metrics = {**base, **extra}

    logger.info(
        "%d%% CI for rate difference: [%.5f, %.5f]",
        extra["ci_level"], ci_low, ci_high,
    )

    return metrics


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_prep import load_data, clean_data
    _df = clean_data(load_data("data/raw/ab_data.csv"))
    _metrics = calculate_confidence_intervals(_df)
    for k, v in _metrics.items():
        print(f"  {k:<20s}: {v}")
