"""
Preprocessing for temporal alignment and aggregation.

This module handles:
- Resampling time series to consistent cadence
- Aggregating litter and effort data
- Aligning multiple time series
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger('marine_litter.preprocessing')


def resample_litter_data(
    litter_df: pd.DataFrame,
    cadence: str = 'W',
    port_id_col: str = 'port_id',
    date_col: str = 'date',
    agg_funcs: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Resample litter data to consistent temporal cadence.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter observations with date and port_id.
    cadence : str
        Pandas offset alias ('D' for daily, 'W' for weekly, 'M' for monthly).
    port_id_col : str
        Name of port identifier column.
    date_col : str
        Name of date column.
    agg_funcs : Dict[str, str], optional
        Aggregation functions for each column.
        Default: sum for counts/weights.

    Returns
    -------
    pd.DataFrame
        Resampled litter data with MultiIndex (date, port_id).
    """
    logger.info(f"Resampling litter data to cadence: {cadence}")

    # Detect available litter measurement columns
    potential_litter_cols = {
        'litter_count': 'sum',
        'litter_weight_kg': 'sum',
        'Numero_individus': 'sum',   # Catalan name if not translated
        'Pes_total': 'sum',          # Catalan name if not translated
        'count': 'sum',
        'weight': 'sum',
        'items': 'sum'
    }

    if agg_funcs is None:
        # Build agg_funcs from available columns
        agg_funcs = {}
        for col, agg in potential_litter_cols.items():
            if col in litter_df.columns:
                agg_funcs[col] = agg

        # Add any numeric columns that aren't date or port_id
        for col in litter_df.columns:
            if col not in [date_col, port_id_col] and col not in agg_funcs:
                if pd.api.types.is_numeric_dtype(litter_df[col]):
                    agg_funcs[col] = 'sum'

        if not agg_funcs:
            raise ValueError(
                f"No litter measurement columns found. "
                f"Available columns: {list(litter_df.columns)}. "
                f"Expected one of: {list(potential_litter_cols.keys())}"
            )

        logger.info(f"Auto-detected litter columns to aggregate: {list(agg_funcs.keys())}")

    # Set date as index
    df = litter_df.copy()

    # Make sure date_col and port_id_col exist
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available: {list(df.columns)}")
    if port_id_col not in df.columns:
        raise ValueError(f"Port ID column '{port_id_col}' not found. Available: {list(df.columns)}")

    df = df.set_index(date_col)

    # Group by port and resample
    resampled = df.groupby(port_id_col).resample(cadence).agg(agg_funcs)

    # Fill NaN with zeros (no litter observed = zero litter)
    resampled = resampled.fillna(0)

    logger.info(f"Resampled to {len(resampled)} records with columns: {list(resampled.columns)}")

    return resampled


def resample_effort_data(
    effort_df: pd.DataFrame,
    cadence: str = 'W',
    port_id_col: str = 'port_id',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Resample effort data to consistent temporal cadence.

    Parameters
    ----------
    effort_df : pd.DataFrame
        Trawling effort with date and port_id.
    cadence : str
        Pandas offset alias.
    port_id_col : str
        Name of port identifier column.
    date_col : str
        Name of date column.

    Returns
    -------
    pd.DataFrame
        Resampled effort data with MultiIndex (date, port_id).
    """
    logger.info(f"Resampling effort data to cadence: {cadence}")

    df = effort_df.copy()
    df = df.set_index(date_col)

    # Aggregate effort (sum hours/hauls)
    agg_funcs = {'trawling_hours': 'sum'}
    if 'area_swept' in df.columns:
        agg_funcs['area_swept'] = 'sum' # Aggregate total area for the period

    if 'hauls' in df.columns:
        agg_funcs['hauls'] = 'sum'

    resampled = df.groupby(port_id_col).resample(cadence).agg(agg_funcs)

    # Fill NaN with zeros (no effort = zero effort)
    resampled = resampled.fillna(0)

    # effort_rate will now default to area_swept if available
    resampled['effort_rate'] = resampled['area_swept'] if 'area_swept' in resampled.columns else resampled['trawling_hours']

    logger.info(f"Resampled effort to {len(resampled)} records")

    return resampled


def resample_hydro_data(
    hydro_df: pd.DataFrame,
    cadence: str = 'W',
    river_id_col: str = 'river_id',
    date_col: str = 'date',
    measurements: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Resample hydrology data to consistent temporal cadence.

    Parameters
    ----------
    hydro_df : pd.DataFrame
        Hydrology data with date and river_id.
    cadence : str
        Pandas offset alias.
    river_id_col : str
        Name of river/station identifier column.
    date_col : str
        Name of date column.
    measurements : List[str], optional
        Columns to aggregate. Default: ['precip_mm', 'discharge_m3s']

    Returns
    -------
    pd.DataFrame
        Resampled hydrology data with MultiIndex (date, river_id).
    """
    logger.info(f"Resampling hydrology data to cadence: {cadence}")

    if measurements is None:
        measurements = [col for col in ['precip_mm', 'discharge_m3s'] if col in hydro_df.columns]

    df = hydro_df.copy()
    df = df.set_index(date_col)

    # Aggregate: sum for precipitation, mean for discharge (could be configurable)
    agg_funcs = {}
    if 'precip_mm' in measurements:
        agg_funcs['precip_mm'] = 'sum'  # Total precipitation over period
    if 'discharge_m3s' in measurements:
        agg_funcs['discharge_m3s'] = 'mean'  # Average discharge

    resampled = df.groupby(river_id_col).resample(cadence).agg(agg_funcs)

    # Forward-fill small gaps (up to 2 periods), then interpolate
    # Apply ffill and interpolate directly to avoid index structure issues
    resampled = resampled.ffill(limit=2).interpolate(limit=5)

    # Ensure clean MultiIndex with proper names
    if isinstance(resampled.index, pd.MultiIndex):
        # Current structure is (river_id, date)
        # Swap to (date, river_id) for consistency with litter data
        if resampled.index.nlevels == 2:
            resampled = resampled.swaplevel(0, 1).sort_index()
            resampled.index.names = ['date', 'river_id']

    logger.info(f"Resampled hydrology to {len(resampled)} records")
    logger.info(f"Index structure: {resampled.index.names if isinstance(resampled.index, pd.MultiIndex) else 'single-level'}")

    return resampled


def align_litter_effort_hydro(
    litter_resampled: pd.DataFrame,
    effort_resampled: pd.DataFrame,
    hydro_resampled: pd.DataFrame,
    river_port_mapping: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align litter, effort, and hydrology data by merging on date and port/river mapping.

    Parameters
    ----------
    litter_resampled : pd.DataFrame
        Resampled litter data (MultiIndex: date, port_id).
    effort_resampled : pd.DataFrame
        Resampled effort data (MultiIndex: date, port_id).
    hydro_resampled : pd.DataFrame
        Resampled hydrology data (MultiIndex: date, river_id).
    river_port_mapping : Dict[str, List[str]]
        Mapping from port_id to list of relevant river_ids.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (aligned_df, hydro_features_df)
        aligned_df: Combined litter + effort per port/date
        hydro_features_df: Hydrology features per port/date/river
    """
    logger.info("Aligning litter, effort, and hydrology data")

    # Merge litter and effort on (date, port_id)
    aligned_df = litter_resampled.join(effort_resampled, how='left')

    # Fill missing effort with small value
    if 'effort_rate' in aligned_df.columns:
        aligned_df['effort_rate'] = aligned_df['effort_rate'].fillna(0.1)

    logger.info(f"Aligned litter + effort: {len(aligned_df)} records")

    # Create hydrology features per port
    # For each port, get measurements from relevant rivers
    hydro_features = []

    for port_id, river_ids in river_port_mapping.items():
        # Get hydrology for these rivers
        port_hydro = hydro_resampled.loc[
            (slice(None), river_ids), :
        ].reset_index()

        # Pivot to have one row per date with columns for each river
        for _, row in port_hydro.iterrows():
            record = {
                'date': row['date'],
                'port_id': port_id,
                'river_id': row['river_id']
            }

            for col in hydro_resampled.columns:
                record[col] = row[col]

            hydro_features.append(record)

    hydro_features_df = pd.DataFrame(hydro_features)

    logger.info(f"Created hydrology features: {len(hydro_features_df)} records")

    return aligned_df, hydro_features_df


def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Add temporal features for modeling (year, month, day of year, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column or DatetimeIndex.
    date_col : str
        Name of date column (ignored if df has DatetimeIndex).

    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features.
    """
    df = df.copy()

    # Get date series
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        dates = pd.to_datetime(df[date_col])

    # Add features
    df['year'] = dates.year
    df['month'] = dates.month
    df['day_of_year'] = dates.dayofyear
    df['week_of_year'] = dates.isocalendar().week

    # Cyclic encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
    df['doy_sin'] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * dates.dayofyear / 365.25)

    logger.info("Added temporal features")

    return df


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = 'interpolate',
    max_gap: int = 5
) -> pd.DataFrame:
    """
    Handle missing data in time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values.
    strategy : str
        'interpolate', 'ffill', 'drop', or 'zero'.
    max_gap : int
        Maximum gap size to interpolate/fill.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing data handled.
    """
    df = df.copy()

    if strategy == 'interpolate':
        df = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
    elif strategy == 'ffill':
        df = df.fillna(method='ffill', limit=max_gap)
    elif strategy == 'zero':
        df = df.fillna(0)
    elif strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} missing values remain after handling")

    return df


def compute_effort_proxy(
    litter_df: pd.DataFrame,
    method: str = 'constant'
) -> pd.Series:
    """
    Compute effort proxy when effort data is unavailable.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data (with index).
    method : str
        'constant' (all equal), 'haul_based' (if haul_id available).

    Returns
    -------
    pd.Series
        Effort proxy values.
    """
    if method == 'constant':
        logger.warning("Using constant effort proxy (all periods equal)")
        return pd.Series(1.0, index=litter_df.index)
    elif method == 'haul_based' and 'haul_id' in litter_df.columns:
        # Count unique hauls per period as proxy
        return litter_df.groupby(level=[0, 1])['haul_id'].nunique()
    else:
        logger.warning("Unknown effort proxy method, using constant")
        return pd.Series(1.0, index=litter_df.index)
