"""
Utility functions for the marine litter analysis pipeline.

This module provides common utilities including:
- Configuration loading
- Logging setup
- Statistical helpers
- Validation functions
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def get_time_unit_from_cadence(cadence: str) -> str:
    """
    Convert pandas frequency/cadence code to readable time unit.

    Parameters
    ----------
    cadence : str
        Pandas frequency code (e.g., 'D', 'W', 'MS', 'M', 'Q', 'Y')

    Returns
    -------
    str
        Readable time unit ('days', 'weeks', 'months', 'quarters', 'years')

    Examples
    --------
    >>> get_time_unit_from_cadence('D')
    'days'
    >>> get_time_unit_from_cadence('W')
    'weeks'
    >>> get_time_unit_from_cadence('MS')
    'months'
    """
    cadence_map = {
        'D': 'days',
        'W': 'weeks',
        'W-SUN': 'weeks',
        'W-MON': 'weeks',
        'W-TUE': 'weeks',
        'W-WED': 'weeks',
        'W-THU': 'weeks',
        'W-FRI': 'weeks',
        'W-SAT': 'weeks',
        'M': 'months',
        'MS': 'months',
        'ME': 'months',
        'Q': 'quarters',
        'QS': 'quarters',
        'Y': 'years',
        'YS': 'years',
        'YE': 'years',
        'A': 'years',
        'AS': 'years',
    }

    # Extract base frequency (remove modifiers like '-SUN')
    base_cadence = cadence.split('-')[0] if '-' in cadence else cadence

    unit = cadence_map.get(base_cadence)
    if unit is None:
        logging.getLogger('marine_litter.utils').warning(
            f"Unknown cadence '{cadence}', defaulting to 'periods'"
        )
        return 'periods'

    return unit


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, logs only to console.
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    format_string : str, optional
        Custom format string for log messages.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[logging.StreamHandler()]
    )

    logger = logging.getLogger('marine_litter')

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    np.random.seed(seed)
    # If using other libraries with random state, set them here
    # e.g., random.seed(seed), torch.manual_seed(seed), etc.


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
    name: str = "DataFrame"
) -> None:
    """
    Validate that a DataFrame has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : List[str]
        List of required column names.
    optional_columns : List[str], optional
        List of optional column names (not validated but logged if missing).
    name : str
        Name of the DataFrame for error messages.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing_required = set(required_columns) - set(df.columns)
    if missing_required:
        raise ValueError(
            f"{name} is missing required columns: {missing_required}. "
            f"Available columns: {list(df.columns)}"
        )

    if optional_columns:
        missing_optional = set(optional_columns) - set(df.columns)
        if missing_optional:
            logging.getLogger('marine_litter').warning(
                f"{name} is missing optional columns: {missing_optional}"
            )


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Apply Bonferroni correction for multiple testing.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.
    alpha : float
        Significance level.

    Returns
    -------
    np.ndarray
        Boolean array indicating which tests pass correction.
    """
    n_tests = len(p_values)
    return p_values < (alpha / n_tests)


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh'
) -> np.ndarray:
    """
    Apply False Discovery Rate correction (Benjamini-Hochberg or Benjamini-Yekutieli).

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.
    alpha : float
        Significance level.
    method : str
        'bh' for Benjamini-Hochberg or 'by' for Benjamini-Yekutieli.

    Returns
    -------
    np.ndarray
        Boolean array indicating which tests pass correction.
    """
    # Sort p-values
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # Compute thresholds
    if method == 'bh':
        thresholds = (np.arange(1, n_tests + 1) / n_tests) * alpha
    elif method == 'by':
        c_m = np.sum(1.0 / np.arange(1, n_tests + 1))
        thresholds = (np.arange(1, n_tests + 1) / (n_tests * c_m)) * alpha
    else:
        raise ValueError(f"Unknown FDR method: {method}")

    # Find largest i where p[i] <= threshold[i]
    rejected = np.zeros(n_tests, dtype=bool)
    passing = sorted_pvals <= thresholds
    if passing.any():
        max_idx = np.where(passing)[0][-1]
        rejected[sorted_indices[:max_idx + 1]] = True

    return rejected


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> tuple:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    statistic_func : callable
        Function that computes the statistic on the data.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    tuple
        (point_estimate, lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)

    point_estimate = statistic_func(data)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    index: bool = True,
    **kwargs
) -> None:
    """
    Save DataFrame to CSV with automatic directory creation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Output path.
    index : bool
        Whether to write row index.
    **kwargs
        Additional arguments passed to df.to_csv().
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index, **kwargs)
    logging.getLogger('marine_litter').info(f"Saved DataFrame to {path}")


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """
    Compute Variance Inflation Factor (VIF) for detecting multicollinearity.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.Series
        VIF values for each feature.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )
    return vif_data


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great circle distance between two points in kilometers.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point (degrees).
    lat2, lon2 : float
        Latitude and longitude of second point (degrees).

    Returns
    -------
    float
        Distance in kilometers.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in kilometers
    r = 6371

    return c * r


def is_north_of(
    lat_point: float,
    lat_reference: float,
    tolerance: float = 0.1
) -> bool:
    """
    Check if a point is north of a reference point.

    Parameters
    ----------
    lat_point : float
        Latitude of point to check.
    lat_reference : float
        Latitude of reference point.
    tolerance : float
        Tolerance in degrees (points within tolerance are considered same latitude).

    Returns
    -------
    bool
        True if point is north of reference.
    """
    return lat_point > (lat_reference + tolerance)
