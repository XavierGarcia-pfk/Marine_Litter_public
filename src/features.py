"""
Feature engineering for marine litter analysis.

This module provides functions to create:
- Rolling window aggregations (mean, max, cumulative)
- Distributed lag features
- Interaction features
"""

import logging
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger('marine_litter.features')


def create_rolling_features(
    series: pd.Series,
    windows: List[int],
    functions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create rolling window features for a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series with DatetimeIndex.
    windows : List[int]
        List of window sizes (in same units as series frequency).
    functions : List[str], optional
        Functions to compute: 'mean', 'max', 'sum', 'std'.
        Default: ['mean', 'max', 'sum']

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features as columns.
    """
    if functions is None:
        functions = ['mean', 'max', 'sum']

    features = pd.DataFrame(index=series.index)

    # Get series name, with fallback to 'value'
    if hasattr(series, 'name') and series.name is not None:
        base_name = series.name
    else:
        base_name = 'value'

    for window in windows:
        for func in functions:
            col_name = f"{base_name}_roll{window}_{func}"

            if func == 'mean':
                features[col_name] = series.rolling(window, min_periods=1).mean()
            elif func == 'max':
                features[col_name] = series.rolling(window, min_periods=1).max()
            elif func == 'sum':
                features[col_name] = series.rolling(window, min_periods=1).sum()
            elif func == 'std':
                features[col_name] = series.rolling(window, min_periods=1).std()
            else:
                logger.warning(f"Unknown function: {func}, skipping")

    logger.debug(f"Created {len(features.columns)} rolling features")

    return features


def create_lag_features(
    series: pd.Series,
    lags: Union[List[int], np.ndarray],
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Create lagged features for a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series with DatetimeIndex.
    lags : List[int] or np.ndarray
        Lag values (in same units as series frequency).
    fill_value : float
        Value to use for initial periods where lag is undefined due to shift.
        Note: NaN values in the original series are preserved.

    Returns
    -------
    pd.DataFrame
        DataFrame with lagged series as columns.
    """
    features = pd.DataFrame(index=series.index)
    base_name = series.name or 'value'

    for lag in lags:
        col_name = f"{base_name}_lag{lag}"
        shifted = series.shift(lag)

        # Only fill NaN values created by shift (at the beginning)
        # Preserve NaN values from original series
        if lag > 0:
            # Fill only the first 'lag' positions
            shifted.iloc[:lag] = fill_value

        features[col_name] = shifted

    logger.debug(f"Created {len(lags)} lag features")

    return features


def create_distributed_lag_matrix(
    hydro_df: pd.DataFrame,
    litter_dates: pd.DatetimeIndex,
    river_id: str,
    lags: np.ndarray,
    measurement_col: str = 'precip_mm',
    river_id_col: str = 'river_id',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Create distributed lag matrix for one river.

    For each date in litter_dates, create features for all lags
    of the hydrology measurement.

    Parameters
    ----------
    hydro_df : pd.DataFrame
        Hydrology data.
    litter_dates : pd.DatetimeIndex
        Target dates for alignment.
    river_id : str
        River identifier.
    lags : np.ndarray
        Array of lag values.
    measurement_col : str
        Column to lag (e.g., 'precip_mm').
    river_id_col : str
        River ID column name.
    date_col : str
        Date column name.

    Returns
    -------
    pd.DataFrame
        Matrix with shape (len(litter_dates), len(lags)).
    """
    # Filter for this river
    river_data = hydro_df[hydro_df[river_id_col] == river_id].copy()
    river_data = river_data.set_index(date_col)[measurement_col].sort_index()

    # Create lag matrix
    lag_matrix = np.zeros((len(litter_dates), len(lags)))

    for i, target_date in enumerate(litter_dates):
        for j, lag in enumerate(lags):
            lagged_date = target_date - pd.Timedelta(days=int(lag))

            # Find closest hydrology measurement
            if lagged_date in river_data.index:
                lag_matrix[i, j] = river_data.loc[lagged_date]
            else:
                # Interpolate or use nearest
                try:
                    # Find nearest date within reasonable range
                    date_diff = np.abs((river_data.index - lagged_date).days)
                    if date_diff.min() <= 7:  # Within 7 days
                        nearest_idx = date_diff.argmin()
                        lag_matrix[i, j] = river_data.iloc[nearest_idx]
                    else:
                        lag_matrix[i, j] = 0.0
                except:
                    lag_matrix[i, j] = 0.0

    # Convert to DataFrame
    col_names = [f"lag_{lag}" for lag in lags]
    df = pd.DataFrame(lag_matrix, index=litter_dates, columns=col_names)

    logger.debug(f"Created distributed lag matrix for river {river_id}: {df.shape}")

    return df


def create_port_hydro_features(
    litter_df: pd.DataFrame,
    hydro_df: pd.DataFrame,
    port_id: str,
    river_ids: List[str],
    lags: np.ndarray,
    rolling_windows: List[int],
    measurement_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create comprehensive hydrology features for one port.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data (for date alignment).
    hydro_df : pd.DataFrame
        Hydrology data.
    port_id : str
        Port identifier.
    river_ids : List[str]
        Relevant river IDs for this port.
    lags : np.ndarray
        Lag range.
    rolling_windows : List[int]
        Rolling window sizes.
    measurement_cols : List[str], optional
        Hydrology columns to process.

    Returns
    -------
    pd.DataFrame
        Feature matrix for this port.
    """
    if measurement_cols is None:
        measurement_cols = ['precip_mm', 'discharge_m3s']
        measurement_cols = [col for col in measurement_cols if col in hydro_df.columns]

    # Get litter dates for this port
    port_litter = litter_df[litter_df.index.get_level_values('port_id') == port_id]
    litter_dates = port_litter.index.get_level_values('date').unique()

    all_features = pd.DataFrame(index=litter_dates)

    # For each river
    for river_id in river_ids:
        river_data = hydro_df[hydro_df['river_id'] == river_id]

        if river_data.empty:
            logger.warning(f"No hydrology data for river {river_id}, skipping")
            continue

        # For each measurement type
        for meas_col in measurement_cols:
            if meas_col not in river_data.columns:
                continue

            # Create time series for this river-measurement
            series = river_data.set_index('date')[meas_col].sort_index()

            # Reindex to litter dates (with interpolation)
            series_aligned = series.reindex(
                litter_dates,
                method='nearest',
                limit=7  # Max 7 periods gap
            ).fillna(0)

            series_aligned.name = f"{river_id}_{meas_col}"

            # Rolling features
            rolling_feats = create_rolling_features(
                series_aligned,
                windows=rolling_windows,
                functions=['mean', 'max']
            )
            all_features = all_features.join(rolling_feats)

            # Distributed lags on rolling means
            for window in rolling_windows:
                roll_series = series_aligned.rolling(window, min_periods=1).mean()
                roll_series.name = f"{river_id}_{meas_col}_roll{window}"

                # Create lags for this rolled series
                lag_feats = create_lag_features(
                    roll_series,
                    lags=lags,
                    fill_value=0.0
                )
                all_features = all_features.join(lag_feats)

    logger.info(f"Created {len(all_features.columns)} hydro features for port {port_id}")

    return all_features


def create_all_port_features(
    litter_df: pd.DataFrame,
    hydro_df: pd.DataFrame,
    river_port_mapping: dict,
    lags: np.ndarray,
    rolling_windows: List[int],
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Create hydrology features for all ports in parallel.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data with MultiIndex (date, port_id).
    hydro_df : pd.DataFrame
        Hydrology data.
    river_port_mapping : Dict
        Mapping from port_id to list of (river_id, distance).
    lags : np.ndarray
        Lag range.
    rolling_windows : List[int]
        Rolling window sizes.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    pd.DataFrame
        Feature matrix for all ports with MultiIndex (date, port_id).
    """
    logger.info("Creating hydrology features for all ports (parallel)")

    ports = list(river_port_mapping.keys())

    # Parallel processing
    def process_port(port_id):
        river_ids = [rid for rid, _ in river_port_mapping[port_id]]
        return port_id, create_port_hydro_features(
            litter_df, hydro_df, port_id, river_ids, lags, rolling_windows
        )

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_port)(port_id) for port_id in ports
    )

    # Combine results
    all_features = []
    for port_id, port_features in results:
        port_features['port_id'] = port_id
        all_features.append(port_features.reset_index())

    combined = pd.concat(all_features, ignore_index=True)
    combined = combined.set_index(['date', 'port_id']).sort_index()

    logger.info(f"Created feature matrix: {combined.shape}")

    return combined


def select_best_lag_features(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    method: str = 'correlation',
    n_features: int = 20
) -> List[str]:
    """
    Select most relevant lag features using univariate selection.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    target_series : pd.Series
        Target variable (aligned with features_df).
    method : str
        Selection method: 'correlation', 'mutual_info'.
    n_features : int
        Number of features to select.

    Returns
    -------
    List[str]
        Selected feature names.
    """
    from scipy.stats import spearmanr

    # Align target with features
    aligned_target = target_series.reindex(features_df.index)

    # Remove rows with NaN in target
    valid_idx = ~aligned_target.isna()
    X = features_df.loc[valid_idx]
    y = aligned_target.loc[valid_idx]

    if method == 'correlation':
        # Compute correlation for each feature
        correlations = {}
        for col in X.columns:
            try:
                corr, _ = spearmanr(X[col], y, nan_policy='omit')
                correlations[col] = abs(corr)
            except:
                correlations[col] = 0.0

        # Sort and select top n
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in sorted_features[:n_features]]

    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression

        # Handle NaNs
        X_clean = X.fillna(0)

        mi_scores = mutual_info_regression(X_clean, y, random_state=42)
        mi_dict = dict(zip(X.columns, mi_scores))

        sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in sorted_features[:n_features]]

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Selected {len(selected)} features using {method}")

    return selected


def add_interaction_features(
    features_df: pd.DataFrame,
    interactions: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Add interaction features (products of pairs of features).

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    interactions : List[Tuple[str, str]]
        List of feature pairs to interact.

    Returns
    -------
    pd.DataFrame
        Feature matrix with interactions added.
    """
    df = features_df.copy()

    for feat1, feat2 in interactions:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            logger.debug(f"Added interaction: {interaction_name}")

    return df


def add_temporal_features(
    df: pd.DataFrame,
    include_cyclic: bool = True,
    include_doy: bool = False,
    include_trend: bool = False
) -> pd.DataFrame:
    """
    Add temporal features for controlling seasonality and trends.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    include_cyclic : bool
        Add cyclic month features (sin/cos).
    include_doy : bool
        Add day-of-year feature.
    include_trend : bool
        Add linear time trend.

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features added.
    """
    df_out = df.copy()

    if not isinstance(df_out.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, cannot add temporal features")
        return df_out

    # Extract temporal components
    df_out['month'] = df_out.index.month
    df_out['year'] = df_out.index.year

    # Cyclic encoding of month (for seasonality)
    if include_cyclic:
        # Sin/cos transformation: December (12) connects to January (1)
        df_out['month_sin'] = np.sin(2 * np.pi * df_out['month'] / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * df_out['month'] / 12)
        logger.debug("Added cyclic month features (sin/cos)")

    # Day of year (1-365/366)
    if include_doy:
        df_out['day_of_year'] = df_out.index.dayofyear
        logger.debug("Added day_of_year feature")

    # Linear time trend (for long-term changes)
    if include_trend:
        # Normalize to 0-1 range
        time_numeric = (df_out.index - df_out.index.min()).days
        df_out['time_trend'] = time_numeric / time_numeric.max()
        logger.debug("Added time_trend feature")

    return df_out

    logger.info(f"Added {len(interactions)} interaction features")

    return df


def select_best_xema_station(port_id, litter_series, hydro_df, mapping):
    """
    Given a port, test all mapped XEMA stations and return the ID of the best one.
    """
    from scipy import stats
    best_corr = -1
    best_station = None
    
    # Get candidate stations from your mapping
    candidates = mapping.get(port_id, [])
    
    for station_id, dist in candidates:
        station_data = hydro_df[hydro_df['river_id'] == station_id]['precip_mm']
        # Align and compute correlation
        corr, _ = stats.spearmanr(litter_series, station_data.reindex(litter_series.index))
        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_station = station_id
            
    return best_station


def add_cumulative_precipitation_features(
        haul_df: pd.DataFrame,
        precip_df: pd.DataFrame,
        port_station_mapping: Dict,
        windows: List[int] = [4, 8, 12, 16, 20]
) -> pd.DataFrame:
    """
    For each haul, add cumulative precipitation over different time windows.

    Parameters
    ----------
    haul_df : pd.DataFrame
        Haul-level data with 'date', 'port_id'
    precip_df : pd.DataFrame
        Precipitation data with 'date', 'station_id', 'precip_mm'
    port_station_mapping : Dict
        Mapping from port_id to station_id
    windows : List[int]
        Time windows in weeks

    Returns
    -------
    pd.DataFrame
        Haul data with added precipitation features
    """
    haul_df = haul_df.copy()

    for port_id in haul_df['port_id'].unique():
        # Get station for this port
        station_id = port_station_mapping.get(port_id)
        if not station_id:
            continue

        # Get precipitation for this station
        station_precip = precip_df[precip_df['station_id'] == station_id].copy()
        station_precip = station_precip.set_index('date')['precip_mm'].sort_index()

        # For each haul at this port
        port_mask = haul_df['port_id'] == port_id

        for window in windows:
            days = window * 7

            # Calculate cumulative precip for each haul
            def calc_cum_precip(haul_date):
                start_date = haul_date - pd.Timedelta(days=days)
                period_precip = station_precip[
                    (station_precip.index >= start_date) &
                    (station_precip.index < haul_date)
                    ]
                return period_precip.sum() if len(period_precip) > 0 else 0

            haul_df.loc[port_mask, f'precip_cum_{window}wk'] = (
                haul_df.loc[port_mask, 'date'].apply(calc_cum_precip)
            )

    return haul_df
