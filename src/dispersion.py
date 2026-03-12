"""
Lagrangian dispersion model comparison and skill assessment.

This module provides functions to:
- Compare dispersion model outputs with observed litter
- Compute skill metrics (correlation, ROC, Brier score)
- Assess binary event prediction performance
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss,
    confusion_matrix, f1_score
)

logger = logging.getLogger('marine_litter.dispersion')


def align_dispersion_with_litter(
        dispersion_df: pd.DataFrame,
        litter_df: pd.DataFrame,
        port_id_col: str = 'port_id',
        date_col: str = 'date',
        dispersion_col: str = 'settled_particles'
) -> pd.DataFrame:
    """
    Align dispersion model outputs with litter observations.

    Parameters
    ----------
    dispersion_df : pd.DataFrame
        Dispersion aggregates with columns: date, port_id, dispersion metric.
    litter_df : pd.DataFrame
        Litter data with MultiIndex (date, port_id).
    port_id_col : str
        Port identifier column.
    date_col : str
        Date column.
    dispersion_col : str
        Dispersion metric column name.

    Returns
    -------
    pd.DataFrame
        Aligned data with columns: date, port_id, litter_count, dispersion_metric.
    """
    logger.info("Aligning dispersion outputs with litter observations")

    # Ensure date column is datetime
    dispersion_df = dispersion_df.copy()
    dispersion_df[date_col] = pd.to_datetime(dispersion_df[date_col])

    # Set index for joining
    dispersion_indexed = dispersion_df.set_index([date_col, port_id_col])

    # Join with litter
    litter_indexed = litter_df.copy()
    if not isinstance(litter_indexed.index, pd.MultiIndex):
        litter_indexed = litter_indexed.set_index([date_col, port_id_col])

    aligned = litter_indexed.join(
        dispersion_indexed[[dispersion_col]],
        how='inner'
    )

    # Remove rows with NaN
    aligned = aligned.dropna(subset=['litter_count', dispersion_col])

    logger.info(f"Aligned {len(aligned)} date-port pairs")

    return aligned.reset_index()


def compute_lagged_dispersion_correlation(
        litter_series: pd.Series,
        dispersion_series: pd.Series,
        lags: List[int],
        method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute correlation between litter and dispersion at multiple lags.

    Parameters
    ----------
    litter_series : pd.Series
        Litter time series with DatetimeIndex.
    dispersion_series : pd.Series
        Dispersion time series with DatetimeIndex.
    lags : List[int]
        Lag values (in same units as series frequency).
    method : str
        'spearman' or 'pearson'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: lag, correlation, p_value.
    """
    results = []

    for lag in lags:
        # Shift dispersion by lag
        dispersion_lagged = dispersion_series.shift(lag)

        # Align
        common_idx = litter_series.index.intersection(dispersion_lagged.index)
        if len(common_idx) < 3:
            continue

        litter_aligned = litter_series.loc[common_idx]
        dispersion_aligned = dispersion_lagged.loc[common_idx]

        # Remove NaN
        valid = ~(litter_aligned.isna() | dispersion_aligned.isna())
        if valid.sum() < 3:
            continue

        try:
            if method == 'spearman':
                corr, pval = spearmanr(
                    litter_aligned[valid],
                    dispersion_aligned[valid]
                )
            else:
                corr, pval = pearsonr(
                    litter_aligned[valid],
                    dispersion_aligned[valid]
                )

            results.append({
                'lag': lag,
                'correlation': corr,
                'p_value': pval
            })
        except Exception as e:
            logger.warning(f"Correlation failed at lag {lag}: {e}")
            continue

    return pd.DataFrame(results)


def compute_skill_metrics(
        aligned_df: pd.DataFrame,
        litter_col: str = 'litter_count',
        dispersion_col: str = 'settled_particles'
) -> Dict[str, float]:
    """
    Compute comprehensive skill metrics.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned litter and dispersion data.
    litter_col : str
        Litter column name.
    dispersion_col : str
        Dispersion metric column name.

    Returns
    -------
    Dict[str, float]
        Dictionary of skill metrics.
    """
    logger.info("Computing dispersion model skill metrics")

    litter = aligned_df[litter_col].values
    dispersion = aligned_df[dispersion_col].values

    metrics = {}

    # Correlation
    try:
        corr_s, pval_s = spearmanr(litter, dispersion)
        corr_p, pval_p = pearsonr(litter, dispersion)
        metrics['spearman_corr'] = corr_s
        metrics['spearman_pval'] = pval_s
        metrics['pearson_corr'] = corr_p
        metrics['pearson_pval'] = pval_p
    except Exception as e:
        logger.warning(f"Correlation computation failed: {e}")

    # RMSE and MAE
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Normalize to same scale for RMSE/MAE
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    litter_scaled = scaler.fit_transform(litter.reshape(-1, 1)).flatten()
    dispersion_scaled = scaler.fit_transform(dispersion.reshape(-1, 1)).flatten()

    metrics['rmse'] = np.sqrt(mean_squared_error(litter_scaled, dispersion_scaled))
    metrics['mae'] = mean_absolute_error(litter_scaled, dispersion_scaled)
    metrics['r2'] = np.corrcoef(litter, dispersion)[0, 1] ** 2

    logger.info(f"Skill metrics: Spearman ρ={metrics.get('spearman_corr', np.nan):.3f}, "
                f"RMSE={metrics['rmse']:.3f}")

    return metrics


def assess_binary_event_prediction(
        aligned_df: pd.DataFrame,
        litter_col: str = 'litter_count',
        dispersion_col: str = 'settled_particles',
        litter_threshold_quantile: float = 0.75,
        dispersion_threshold_quantile: float = 0.75
) -> Dict[str, float]:
    """
    Assess binary event prediction (high litter events).

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned data.
    litter_col : str
        Litter column.
    dispersion_col : str
        Dispersion column.
    litter_threshold_quantile : float
        Quantile for defining high litter events.
    dispersion_threshold_quantile : float
        Quantile for dispersion threshold.

    Returns
    -------
    Dict[str, float]
        Binary prediction metrics.
    """
    logger.info("Assessing binary event prediction performance")

    litter = aligned_df[litter_col].values
    dispersion = aligned_df[dispersion_col].values

    # Define high events
    litter_threshold = np.quantile(litter, litter_threshold_quantile)
    dispersion_threshold = np.quantile(dispersion, dispersion_threshold_quantile)

    litter_binary = (litter >= litter_threshold).astype(int)
    dispersion_binary = (dispersion >= dispersion_threshold).astype(int)

    metrics = {}

    # Confusion matrix
    cm = confusion_matrix(litter_binary, dispersion_binary)
    tn, fp, fn, tp = cm.ravel()

    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)

    # Hit rate, false alarm rate
    metrics['hit_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = metrics['hit_rate']
    metrics['f1_score'] = f1_score(litter_binary, dispersion_binary)

    # ROC AUC (using continuous values)
    try:
        metrics['roc_auc'] = roc_auc_score(litter_binary, dispersion)
    except Exception as e:
        logger.warning(f"ROC AUC computation failed: {e}")
        metrics['roc_auc'] = np.nan

    # Brier score
    try:
        # Normalize dispersion to [0, 1] for Brier score
        disp_normalized = (dispersion - dispersion.min()) / (dispersion.max() - dispersion.min() + 1e-10)
        metrics['brier_score'] = brier_score_loss(litter_binary, disp_normalized)
    except Exception as e:
        logger.warning(f"Brier score computation failed: {e}")
        metrics['brier_score'] = np.nan

    logger.info(f"Binary prediction: Hit rate={metrics['hit_rate']:.3f}, "
                f"FAR={metrics['false_alarm_rate']:.3f}, F1={metrics['f1_score']:.3f}")

    return metrics


def compute_port_specific_skill(
        aligned_df: pd.DataFrame,
        litter_col: str = 'litter_count',
        dispersion_col: str = 'settled_particles',
        port_id_col: str = 'port_id'
) -> pd.DataFrame:
    """
    Compute skill metrics separately for each port.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned data with port_id column.
    litter_col : str
        Litter column.
    dispersion_col : str
        Dispersion column.
    port_id_col : str
        Port identifier column.

    Returns
    -------
    pd.DataFrame
        Skill metrics per port.
    """
    logger.info("Computing port-specific skill metrics")

    port_metrics = []

    for port_id, group in aligned_df.groupby(port_id_col):
        if len(group) < 10:
            logger.warning(f"Port {port_id}: insufficient data ({len(group)} points)")
            continue

        try:
            metrics = compute_skill_metrics(group, litter_col, dispersion_col)
            metrics['port_id'] = port_id
            metrics['n_observations'] = len(group)

            port_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"Skill computation failed for port {port_id}: {e}")
            continue

    return pd.DataFrame(port_metrics)


def compare_dispersion_with_litter_timeseries(
        aligned_df: pd.DataFrame,
        port_id: str,
        litter_col: str = 'litter_count',
        dispersion_col: str = 'settled_particles'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detailed comparison for a single port.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned data.
    port_id : str
        Port to analyze.
    litter_col : str
        Litter column.
    dispersion_col : str
        Dispersion column.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (time_series_comparison, skill_metrics)
    """
    port_data = aligned_df[aligned_df['port_id'] == port_id].copy()

    if len(port_data) < 3:
        logger.warning(f"Insufficient data for port {port_id}")
        return pd.DataFrame(), {}

    # Sort by date
    port_data = port_data.sort_values('date')

    # Compute skill metrics
    skill = compute_skill_metrics(port_data, litter_col, dispersion_col)
    skill['port_id'] = port_id

    # Prepare time series comparison
    comparison = port_data[['date', litter_col, dispersion_col]].copy()

    # Normalize for visual comparison
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    comparison['litter_normalized'] = scaler.fit_transform(
        comparison[[litter_col]]
    )
    comparison['dispersion_normalized'] = scaler.fit_transform(
        comparison[[dispersion_col]]
    )

    return comparison, skill


def generate_skill_report(
        all_skill_metrics: Dict[str, float],
        port_skill_metrics: pd.DataFrame,
        output_path: Optional[str] = None
) -> str:
    """
    Generate a text summary report of skill assessment.

    Parameters
    ----------
    all_skill_metrics : Dict
        Overall skill metrics.
    port_skill_metrics : pd.DataFrame
        Port-specific metrics.
    output_path : str, optional
        Path to save report.

    Returns
    -------
    str
        Report text.
    """
    report = []
    report.append("=" * 60)
    report.append("LAGRANGIAN DISPERSION MODEL SKILL ASSESSMENT")
    report.append("=" * 60)
    report.append("")

    report.append("Overall Skill Metrics:")
    report.append("-" * 40)
    for key, value in all_skill_metrics.items():
        report.append(f"  {key:25s}: {value:.4f}")
    report.append("")

    report.append("Port-Specific Performance:")
    report.append("-" * 40)
    for _, row in port_skill_metrics.iterrows():
        report.append(f"\n  Port: {row['port_id']}")
        report.append(f"    Observations: {row['n_observations']}")
        report.append(f"    Spearman ρ:   {row.get('spearman_corr', np.nan):.3f}")
        report.append(f"    RMSE:         {row.get('rmse', np.nan):.3f}")

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Skill report saved to {output_path}")

    return report_text
