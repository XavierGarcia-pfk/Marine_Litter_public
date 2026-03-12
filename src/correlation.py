"""
Correlation analysis for litter-hydrology relationships.

This module provides functions to:
- Compute lagged correlations (Spearman/Pearson)
- Bootstrap confidence intervals
- Multiple testing correction
- Parallel computation for efficiency
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr

from utils import bonferroni_correction, fdr_correction, bootstrap_confidence_interval

logger = logging.getLogger('marine_litter.correlation')


def compute_lagged_correlation(
    litter_series: pd.Series,
    hydro_series: pd.Series,
    lags: np.ndarray,
    method: str = 'spearman'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation between litter and hydrology at multiple lags.

    Parameters
    ----------
    litter_series : pd.Series
        Litter time series.
    hydro_series : pd.Series
        Hydrology time series.
    lags : np.ndarray
        Array of lag values (in same units as series frequency).
    method : str
        'spearman' or 'pearson'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (correlations, p_values) arrays of same shape as lags.
    """
    correlations = np.zeros(len(lags))
    p_values = np.ones(len(lags))

    # Align series
    common_dates = litter_series.index.intersection(hydro_series.index)
    litter_aligned = litter_series.loc[common_dates]
    hydro_aligned = hydro_series.loc[common_dates]

    for i, lag in enumerate(lags):
        # Shift hydrology by lag
        hydro_lagged = hydro_aligned.shift(lag)

        # Remove NaN
        valid_idx = ~(litter_aligned.isna() | hydro_lagged.isna())

        if valid_idx.sum() < 3:  # Need at least 3 points
            continue

        litter_valid = litter_aligned[valid_idx]
        hydro_valid = hydro_lagged[valid_idx]

        try:
            if method == 'spearman':
                corr, pval = spearmanr(litter_valid, hydro_valid)
            elif method == 'pearson':
                corr, pval = pearsonr(litter_valid, hydro_valid)
            else:
                raise ValueError(f"Unknown method: {method}")

            correlations[i] = corr
            p_values[i] = pval
        except Exception as e:
            logger.warning(f"Correlation failed at lag {lag}: {e}")
            continue

    return correlations, p_values


def find_optimal_lag(
    correlations: np.ndarray,
    p_values: np.ndarray,
    lags: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Find optimal lag with maximum absolute correlation.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation values.
    p_values : np.ndarray
        P-values.
    lags : np.ndarray
        Lag values.
    alpha : float
        Significance level.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'lag', 'correlation', 'p_value'.
    """
    # Find maximum absolute correlation
    abs_corr = np.abs(correlations)
    max_idx = np.argmax(abs_corr)

    return {
        'lag': lags[max_idx],
        'correlation': correlations[max_idx],
        'p_value': p_values[max_idx],
        'significant': p_values[max_idx] < alpha
    }


def bootstrap_lag_correlation(
    litter_series: pd.Series,
    hydro_series: pd.Series,
    lag: int,
    method: str = 'spearman',
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for correlation at a specific lag.

    Parameters
    ----------
    litter_series : pd.Series
        Litter time series.
    hydro_series : pd.Series
        Hydrology time series.
    lag : int
        Lag value.
    method : str
        'spearman' or 'pearson'.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level.
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'point_estimate', 'lower_ci', 'upper_ci'.
    """
    # Align and lag
    common_dates = litter_series.index.intersection(hydro_series.index)
    litter_aligned = litter_series.loc[common_dates]
    hydro_aligned = hydro_series.loc[common_dates].shift(lag)

    valid_idx = ~(litter_aligned.isna() | hydro_aligned.isna())
    litter_valid = litter_aligned[valid_idx].values
    hydro_valid = hydro_aligned[valid_idx].values

    if len(litter_valid) < 3:
        return {'point_estimate': 0, 'lower_ci': 0, 'upper_ci': 0}

    # Define correlation function
    def corr_func(indices):
        if method == 'spearman':
            corr, _ = spearmanr(litter_valid[indices], hydro_valid[indices])
        else:
            corr, _ = pearsonr(litter_valid[indices], hydro_valid[indices])
        return corr

    # Bootstrap
    np.random.seed(random_state)
    bootstrap_corrs = []

    n = len(litter_valid)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            bootstrap_corrs.append(corr_func(indices))
        except:
            continue

    bootstrap_corrs = np.array(bootstrap_corrs)

    # Compute CI
    point_estimate = corr_func(np.arange(n))
    alpha = 1 - confidence
    lower_ci = np.percentile(bootstrap_corrs, 100 * alpha / 2)
    upper_ci = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))

    return {
        'point_estimate': point_estimate,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }


def compute_port_river_correlations(
    litter_df: pd.DataFrame,
    hydro_df: pd.DataFrame,
    port_id: str,
    river_ids: List[str],
    lags: np.ndarray,
    rolling_window: int = 30,
    litter_col: str = 'litter_count',
    hydro_col: str = 'precip_mm',
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute correlations between one port's litter and multiple rivers' hydrology.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data with MultiIndex (date, port_id).
    hydro_df : pd.DataFrame
        Hydrology data.
    port_id : str
        Port identifier.
    river_ids : List[str]
        River identifiers.
    lags : np.ndarray
        Lag range.
    rolling_window : int
        Rolling window size for hydrology smoothing.
    litter_col : str
        Litter column name.
    hydro_col : str
        Hydrology column name.
    method : str
        Correlation method.

    Returns
    -------
    pd.DataFrame
        Correlation results with columns: river_id, lag, correlation, p_value.
    """
    # Get litter for this port
    port_litter = litter_df.loc[
        (slice(None), port_id), litter_col
    ].droplevel('port_id')

    results = []

    for river_id in river_ids:
        # Get hydrology for this river
        river_hydro = hydro_df[hydro_df['river_id'] == river_id].copy()

        if river_hydro.empty:
            logger.warning(f"No hydrology for river {river_id}, skipping")
            continue

        # Apply rolling window
        river_series = river_hydro.set_index('date')[hydro_col].sort_index()
        river_series_smooth = river_series.rolling(rolling_window, min_periods=1).mean()

        # Compute lagged correlations
        correlations, p_values = compute_lagged_correlation(
            port_litter, river_series_smooth, lags, method
        )

        # Store results
        for i, lag in enumerate(lags):
            results.append({
                'port_id': port_id,
                'river_id': river_id,
                'lag': lag,
                'correlation': correlations[i],
                'p_value': p_values[i],
                'rolling_window': rolling_window
            })

    return pd.DataFrame(results)


def compute_all_correlations(
    litter_df: pd.DataFrame,
    hydro_df: pd.DataFrame,
    river_port_mapping: Dict[str, List[Tuple[str, float]]],
    lags: np.ndarray,
    rolling_windows: List[int],
    litter_col: str = 'litter_count',
    hydro_col: str = 'precip_mm',
    method: str = 'spearman',
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Compute correlations for all ports and rivers (parallel).

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data.
    hydro_df : pd.DataFrame
        Hydrology data.
    river_port_mapping : Dict
        Port to rivers mapping.
    lags : np.ndarray
        Lag range.
    rolling_windows : List[int]
        Rolling window sizes to test.
    litter_col : str
        Litter column.
    hydro_col : str
        Hydrology column.
    method : str
        Correlation method.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    pd.DataFrame
        All correlation results.
    """
    logger.info("Computing correlations for all ports (parallel)")

    # Create tasks
    tasks = []
    for port_id, river_info in river_port_mapping.items():
        river_ids = [rid for rid, _ in river_info]
        for rolling_window in rolling_windows:
            tasks.append((port_id, river_ids, rolling_window))

    # Parallel computation
    def compute_task(port_id, river_ids, rolling_window):
        return compute_port_river_correlations(
            litter_df, hydro_df, port_id, river_ids, lags,
            rolling_window, litter_col, hydro_col, method
        )

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(compute_task)(pid, rids, rw) for pid, rids, rw in tasks
    )

    # Combine
    all_results = pd.concat(results, ignore_index=True)

    logger.info(f"Computed {len(all_results)} correlation values")

    return all_results


def apply_multiple_testing_correction(
    corr_df: pd.DataFrame,
    method: str = 'fdr_bh',
    alpha: float = 0.05,
    p_value_col: str = 'p_value'
) -> pd.DataFrame:
    """
    Apply multiple testing correction to correlation results.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation results with p-values.
    method : str
        'bonferroni' or 'fdr_bh'.
    alpha : float
        Significance level.
    p_value_col : str
        P-value column name.

    Returns
    -------
    pd.DataFrame
        Results with 'significant' column added.
    """
    df = corr_df.copy()

    # Handle empty DataFrame
    if len(df) == 0:
        logger.warning("Empty correlation results - no multiple testing correction applied")
        df['significant'] = []
        return df

    # Check if p_value column exists
    if p_value_col not in df.columns:
        logger.error(f"Column '{p_value_col}' not found in results. Available columns: {list(df.columns)}")
        raise KeyError(f"Column '{p_value_col}' not found. Check correlation computation.")

    p_values = df[p_value_col].values

    if method == 'bonferroni':
        significant = bonferroni_correction(p_values, alpha)
    elif method == 'fdr_bh':
        significant = fdr_correction(p_values, alpha, method='bh')
    elif method == 'none':
        significant = p_values < alpha
    else:
        raise ValueError(f"Unknown correction method: {method}")

    df['significant'] = significant
    df['corrected_alpha'] = alpha / len(p_values) if method == 'bonferroni' else alpha

    n_significant = significant.sum()
    logger.info(
        f"Multiple testing correction ({method}): "
        f"{n_significant}/{len(df)} tests significant at α={alpha}"
    )

    return df


def summarize_best_lags(
    corr_df: pd.DataFrame,
    group_by: List[str] = ['port_id', 'river_id', 'rolling_window']
) -> pd.DataFrame:
    """
    Summarize best lag for each port-river-window combination.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation results.
    group_by : List[str]
        Columns to group by.

    Returns
    -------
    pd.DataFrame
        Summary with best lag per group.
    """
    def find_best_lag(group):
        # Find maximum absolute correlation
        abs_corr = group['correlation'].abs()
        best_idx = abs_corr.idxmax()
        best_row = group.loc[best_idx]

        return pd.Series({
            'best_lag': best_row['lag'],
            'best_correlation': best_row['correlation'],
            'best_p_value': best_row['p_value'],
            'significant': best_row.get('significant', False),
            'n_observations': len(group)
        })

    summary = corr_df.groupby(group_by).apply(find_best_lag).reset_index()

    logger.info(f"Summarized {len(summary)} port-river combinations")

    return summary
