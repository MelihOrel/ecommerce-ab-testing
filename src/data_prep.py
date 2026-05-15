"""
data_prep.py
============
Data loading and cleaning utilities for the E-commerce CRO A/B Test Analysis.

This module handles all data ingestion and preprocessing steps, ensuring the
downstream statistical analysis operates on a clean, validated dataset. The
cleaning strategy follows best practices for A/B test data integrity:

  - Misaligned group/page assignments are removed (e.g., a control user who
    was accidentally exposed to the new page would contaminate the control
    baseline).
  - Duplicate user_id records are removed to prevent inflated sample sizes
    and biased conversion counts.

Author : Senior Data Analyst
Project: E-commerce Conversion Rate Optimisation — A/B Test Analysis
"""

import logging
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw A/B test CSV from disk into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the CSV file (e.g. ``data/raw/ab_data.csv``).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with columns:
        ``user_id``, ``timestamp``, ``group``, ``landing_page``, ``converted``.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not point to an existing file.
    ValueError
        If required columns are missing from the CSV.
    """
    logger.info("Loading raw data from: %s", file_path)

    required_columns = {"user_id", "timestamp", "group", "landing_page", "converted"}

    df = pd.read_csv(file_path)

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info("Raw data loaded successfully — shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw A/B test DataFrame and return a validated copy.

    Cleaning steps applied in order
    --------------------------------
    1. **Alignment filter** — In real-world A/B test pipelines, tracking
       systems occasionally assign users to the wrong page variant.  We
       enforce the strict invariant:

       * ``group == 'control'``   →  ``landing_page == 'old_page'``
       * ``group == 'treatment'`` →  ``landing_page == 'new_page'``

       Rows that violate this invariant represent measurement errors and are
       dropped before any statistical inference.

    2. **Duplicate user_id removal** — Each user must appear at most once.
       If a user has multiple records (e.g. due to session replays or
       logging bugs), only their *first* occurrence (by position) is kept.
       Keeping duplicates would artificially inflate N and distort the
       conversion rate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a reset integer index.

    Notes
    -----
    The function logs a detailed cleaning report so that every dropped row
    can be audited in the execution log.
    """
    logger.info("Starting data cleaning pipeline…")
    initial_rows = len(df)

    # ------------------------------------------------------------------
    # Step 1 — Alignment filter
    # Keep only rows where group↔page assignment is consistent.
    # ------------------------------------------------------------------
    valid_mask = (
        ((df["group"] == "control")   & (df["landing_page"] == "old_page")) |
        ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
    )
    misaligned_count = (~valid_mask).sum()
    df_clean = df[valid_mask].copy()

    logger.info(
        "Alignment filter: removed %d misaligned rows "
        "(control↔new_page or treatment↔old_page).",
        misaligned_count,
    )

    # ------------------------------------------------------------------
    # Step 2 — Duplicate user_id removal
    # Keep the first occurrence; drop subsequent duplicates.
    # ------------------------------------------------------------------
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset="user_id", keep="first")
    duplicate_count = before_dedup - len(df_clean)

    logger.info(
        "Duplicate removal: removed %d duplicate user_id records.",
        duplicate_count,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    final_rows = len(df_clean)
    total_removed = initial_rows - final_rows

    logger.info(
        "Cleaning complete — initial rows: %d | removed: %d | final rows: %d",
        initial_rows,
        total_removed,
        final_rows,
    )

    df_clean = df_clean.reset_index(drop=True)
    return df_clean


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _df_raw = load_data("data/raw/ab_data.csv")
    _df_clean = clean_data(_df_raw)
    print(_df_clean.head())
    print("Groups:\n", _df_clean["group"].value_counts())
    print("Converted:\n", _df_clean.groupby("group")["converted"].mean())
