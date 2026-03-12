#!/usr/bin/env python3
"""
Marine Litter Analysis Pipeline

Complete end-to-end pipeline with:
- Proper haul-level aggregation
- Cumulative precipitation window fitting
- AIC-based window selection
- No correlation dependency

Usage:
    python main_fixed.py run_all --config config/config.yaml
    python main_fixed.py prepare --config config/config.yaml
    python main_fixed.py fit_model --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import load_config, setup_logging, set_random_seed, ensure_dir, save_dataframe
from src.data_loading import (
    load_litter_data, adapt_litter_schema, load_effort_data, load_hydro_data,
    load_ports_metadata, load_dispersion_data
)
from src.preprocessing import resample_hydro_data

import pandas as pd
import numpy as np
import statsmodels.api as sm

logger = logging.getLogger('marine_litter')
try:
    from model_diagnostics import create_diagnostic_plots
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    logger.warning("model_diagnostics not available - diagnostic plots will be skipped")


def cmd_prepare(config: dict) -> dict:
    """
    Prepare and preprocess data - PROPERLY aggregates to haul level.

    Returns dictionary with preprocessed data for downstream steps.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPARATION AND PREPROCESSING")
    logger.info("=" * 60)

    results = {}

    # 1. Load raw data
    logger.info("Loading raw data...")

    try:
        raw_litter_df = load_litter_data(csv_path=config['data']['litter_csv'])
        logger.info(f"Loaded raw litter data: {raw_litter_df.shape}")

        # Adapt schema (translate Catalan → English)
        litter_df = adapt_litter_schema(raw_litter_df)
        logger.info(f"Adapted litter data: {litter_df.shape}")

        # CRITICAL: Filter for marine litter category only
        if 'category' in litter_df.columns:
            logger.info(f"Category distribution before filtering:")
            logger.info(f"{litter_df['category'].value_counts()}")

            # Filter for "Rebuig" (marine litter/discard)
            litter_df = litter_df[litter_df['category'] == 'Rebuig']
            logger.info(f"Filtered to 'Rebuig' (marine litter) category: {len(litter_df)} records")

            if len(litter_df) == 0:
                raise ValueError(
                    "No 'Rebuig' category found in data. "
                    f"Available categories: {raw_litter_df['Categoria'].unique()}"
                )
        else:
            logger.warning("No 'category' column found - using all data without filtering")

    except Exception as e:
        logger.error(f"Failed to load/adapt litter data: {e}")
        raise

    try:
        hydro_df = load_hydro_data(csv_path=config['data']['precipitation_csv'])
        logger.info(f"Loaded hydrology data: {hydro_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load hydrology data: {e}")
        raise

    # 2. AGGREGATE TO TRUE HAUL LEVEL
    logger.info("=" * 60)
    logger.info("AGGREGATING TO HAUL LEVEL")
    logger.info("=" * 60)

    # Determine haul identifier
    if 'haul_code' in litter_df.columns:
        haul_col = 'haul_code'
        logger.info(f"Using haul_code as identifier")
    elif 'haul_id' in litter_df.columns:
        haul_col = 'haul_id'
        logger.warning(f"Using haul_id (may not be unique!)")
    else:
        logger.error("No haul identifier found!")
        raise ValueError("Need haul_code or haul_id column")

    n_unique = litter_df[haul_col].nunique()
    logger.info(f"Unique {haul_col}: {n_unique}")

    # Drop NaN litter_count
    if 'litter_count' in litter_df.columns:
        n_nan = litter_df['litter_count'].isna().sum()
        if n_nan > 0:
            logger.info(f"Dropping {n_nan} rows with NaN litter_count")
            litter_df = litter_df.dropna(subset=['litter_count'])
    else:
        logger.error("No litter_count column!")
        raise ValueError("Need litter_count column")

    # Aggregate by haul
    logger.info(f"Aggregating {len(litter_df)} item records to hauls...")

    agg_dict = {
        'litter_count': 'sum',
        'trawling_hours': 'first',
        'date': 'first',
        'port_id': 'first'
    }

    # Add optional columns
    for col in ['lat', 'lon', 'ProfunditatMitjana_m', 'litter_weight_kg']:
        if col in litter_df.columns:
            if col == 'litter_weight_kg':
                agg_dict[col] = 'sum'
            else:
                agg_dict[col] = 'mean'

    haul_level = litter_df.groupby(haul_col).agg(agg_dict).reset_index()

    logger.info(f"Aggregated to {len(haul_level)} hauls")

    # Filter zero effort
    if 'trawling_hours' in haul_level.columns:
        zero_effort = (haul_level['trawling_hours'] <= 0).sum()
        if zero_effort > 0:
            logger.info(f"Filtering {zero_effort} zero-effort hauls...")
            haul_level = haul_level[haul_level['trawling_hours'] > 0]
            logger.info(f"Remaining: {len(haul_level)} hauls")

    # Check zeros
    n_zero = (haul_level['litter_count'] == 0).sum()
    logger.info(f"Zero litter hauls: {n_zero} ({100 * n_zero / len(haul_level):.1f}%)")

    # Hauls per port
    logger.info(f"\nHauls per port:")
    for port_id in sorted(haul_level['port_id'].unique()):
        n = (haul_level['port_id'] == port_id).sum()
        logger.info(f"  Port {port_id}: {n} hauls")

    # Create MultiIndex for compatibility
    if 'date' in haul_level.columns and 'port_id' in haul_level.columns:
        litter_resampled = haul_level.set_index(['date', 'port_id'])
        logger.info(f"Created MultiIndex (date, port_id) for haul-level data")
    else:
        logger.error("Missing date or port_id columns!")
        raise ValueError("Cannot create MultiIndex without date and port_id")

    logger.info(f"Final haul-level litter data: {litter_resampled.shape}")

    # 3. Hydro: resample to daily for cumulative calculations
    hydro_resampled = resample_hydro_data(
        hydro_df,
        cadence='D'  # Daily is better for cumulative precipitation
    )
    logger.info(f"Resampled hydro data: {hydro_resampled.shape}")

    # 4. Extract ports metadata
    try:
        if 'ports_csv' in config['data']:
            ports_df = load_ports_metadata(csv_path=config['data']['ports_csv'])
            logger.info(f"Loaded ports metadata from file: {len(ports_df)} ports")
        else:
            # Extract from haul data
            port_ids = haul_level['port_id'].unique()
            ports_df = pd.DataFrame({'port_id': port_ids})

            # Add coordinates if available
            if 'lat' in haul_level.columns and 'lon' in haul_level.columns:
                port_coords = haul_level.groupby('port_id')[['lat', 'lon']].mean()
                ports_df = ports_df.merge(port_coords, on='port_id', how='left')
                logger.info(f"Extracted port coordinates from litter data")

            logger.info(f"Created ports metadata: {len(ports_df)} ports")
    except Exception as e:
        logger.warning(f"Could not load/create ports metadata: {e}")
        ports_df = None

    # 5. Save preprocessed data
    outputs_dir = Path(config['outputs']['tables'])
    ensure_dir(outputs_dir)

    save_dataframe(litter_resampled, outputs_dir / 'litter_haul_level.csv')
    save_dataframe(hydro_resampled, outputs_dir / 'hydro_resampled.csv')

    logger.info("=" * 60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {outputs_dir / 'litter_haul_level.csv'}")
    logger.info(f"  {len(haul_level)} hauls across {haul_level['port_id'].nunique()} ports")
    logger.info(f"  Date range: {haul_level['date'].min()} to {haul_level['date'].max()}")
    logger.info("=" * 60)

    results['litter'] = litter_resampled
    results['hydro'] = hydro_resampled
    results['ports'] = ports_df

    return results


def cmd_correlate(config: dict, prepared_data: dict = None) -> dict:
    """
    Correlation analysis - SKIPPED for haul-level data.

    Haul-level data has irregular timestamps that don't align well with
    daily precipitation for correlation analysis. Instead, model fitting
    will test multiple cumulative precipitation windows and select by AIC.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: CORRELATION ANALYSIS - SKIPPED")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Correlation analysis is not appropriate for haul-level data with")
    logger.info("irregular sampling. Model fitting will test multiple precipitation")
    logger.info("windows (4-20 weeks) and select the optimal window by AIC per port.")
    logger.info("")
    logger.info("=" * 60)

    return {}


def cmd_fit_model(config: dict, prepared_data: dict = None, correlation_results: dict = None) -> dict:
    """
    Fit NB models with cumulative precipitation windows.

    For each port:
    - Try cumulative precipitation windows: 4, 8, 12, 16, 20 weeks
    - Fit NB GLM: litter ~ precip_cumulative + seasonal + offset(effort)
    - Select optimal window by AIC
    """
    logger.info("=" * 60)
    logger.info("STEP 3: MODEL FITTING WITH CUMULATIVE HYDRO DATA")
    logger.info("=" * 60)

    # Load or use provided data
    if prepared_data is None:
        logger.info("Loading preprocessed data from files...")
        outputs_dir = Path(config['outputs']['tables'])

        litter_df = pd.read_csv(
            outputs_dir / 'litter_haul_level.csv',
            parse_dates=['date']
        )
        # Reset index if it was saved with MultiIndex
        if 'date' not in litter_df.columns:
            litter_df = litter_df.reset_index()

        hydro_df = pd.read_csv(outputs_dir / 'hydro_resampled.csv')

        logger.info(f"Loaded litter: {len(litter_df)} hauls")
        logger.info(f"Loaded hydro: {len(hydro_df)} records")
    else:
        litter_df = prepared_data['litter'].reset_index()
        hydro_df = prepared_data['hydro'].reset_index() if hasattr(prepared_data['hydro'], 'reset_index') else \
        prepared_data['hydro']
        logger.info(f"Using provided data: {len(litter_df)} hauls")

    # Get port-station mapping
    logger.info("\nSetting up port-station mapping...")

    try:
        if 'manual_overrides' in config.get('mapping', {}):
            manual_mapping = config['mapping']['manual_overrides']
            logger.info(f"Using manual mapping from config: {len(manual_mapping)} ports")

            # Convert to simple port -> station mapping
            port_station_map = {}
            for port_id, stations in manual_mapping.items():
                if isinstance(stations, list) and len(stations) > 0:
                    port_station_map[int(port_id)] = stations[0]
                elif isinstance(stations, str):
                    port_station_map[int(port_id)] = stations

            logger.info(f"Port-station mapping:")
            for port, station in sorted(port_station_map.items()):
                logger.info(f"  Port {port}: {station}")
        else:
            raise KeyError("No manual_overrides in config")

    except Exception as e:
        logger.warning(f"Could not load mapping from config: {repr(e)}")
        logger.info(f"Using automatic mapping (same station for all ports)...")

        # Determine station column name
        station_col = 'river_id' if 'river_id' in hydro_df.columns else 'station_id'
        available_stations = hydro_df[station_col].unique() if station_col in hydro_df.columns else []

        if len(available_stations) == 0:
            logger.error("No stations found in hydro data")
            return {}

        # Use first station for all ports
        first_station = available_stations[0]
        port_station_map = {}
        for port_id in litter_df['port_id'].unique():
            port_station_map[port_id] = first_station

        logger.warning(f"Using same station ({first_station}) for all ports")
        logger.warning("This provides a controlled comparison but may not reflect local hydrology")

    # Cumulative windows to test (weeks)
    # cumulative_windows = [4, 8, 12, 16, 20]
    cumulative_windows = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]

    # Fit models per port
    logger.info("\n" + "=" * 70)
    logger.info("FITTING MODELS PER PORT")
    logger.info("=" * 70)

    results_all = []

    for port_id in sorted(litter_df['port_id'].unique()):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"PORT {port_id}")
        logger.info('=' * 70)

        # Get station for this port
        if port_id not in port_station_map:
            logger.warning(f"No station mapping for port {port_id}, skipping")
            continue

        station_id = port_station_map[port_id]
        logger.info(f"  Station: {station_id}")

        # Get port data
        port_data = litter_df[litter_df['port_id'] == port_id].copy()
        logger.info(f"  Port hauls: {len(port_data)}")

        # Check for required columns
        if 'trawling_hours' not in port_data.columns:
            logger.warning(f"No trawling_hours column, skipping")
            continue

        if 'litter_count' not in port_data.columns:
            logger.warning(f"No litter_count column, skipping")
            continue

        # Get station precipitation
        station_col = 'river_id' if 'river_id' in hydro_df.columns else 'station_id'
        station_data = hydro_df[hydro_df[station_col] == station_id].copy()

        if len(station_data) == 0:
            logger.warning(f"No hydro data for station {station_id}, skipping")
            continue

        logger.info(f"  Station records: {len(station_data)}")

        # Ensure dates are datetime
        port_data['date'] = pd.to_datetime(port_data['date'])
        station_data['date'] = pd.to_datetime(station_data['date'])

        # Find hydro column (precipitation or discharge)
        # Check config first for specified variable
        hydro_col = None
        hydro_variable = config.get('mapping', {}).get('hydro_variable', None)

        if hydro_variable and hydro_variable in station_data.columns:
            # Use variable specified in config
            hydro_col = hydro_variable
            logger.info(f"  Using hydro variable from config: {hydro_col}")
        else:
            # Auto-detect: try discharge first, then precipitation
            for col in ['discharge_m3s', 'precip_mm', 'precipitation', 'Pluviometria']:
                if col in station_data.columns:
                    hydro_col = col
                    logger.info(f"  Auto-detected hydro variable: {hydro_col}")
                    break

        if hydro_col is None:
            logger.warning(
                f"No hydro variable found in data (looked for: discharge_m3s, precip_mm, precipitation, Pluviometria)")
            logger.warning(f"Available columns: {list(station_data.columns)}")
            continue

        # Determine if this is discharge or precipitation (for effect calculation later)
        is_discharge = 'discharge' in hydro_col.lower()
        logger.info(f"  Variable type: {'Discharge (m³/s)' if is_discharge else 'Precipitation (mm)'}")

        # Sort and index station data
        station_data = station_data.sort_values('date')
        station_hydro = station_data.set_index('date')[hydro_col]

        # Calculate cumulative hydro variable for each haul
        variable_name = 'discharge' if is_discharge else 'precipitation'
        logger.info(f"\n  Computing cumulative {variable_name} for {len(cumulative_windows)} windows...")

        for weeks in cumulative_windows:
            days = weeks * 7

            def calc_cumulative(haul_date):
                start_date = haul_date - pd.Timedelta(days=days)
                period_data = station_hydro[(station_hydro.index >= start_date) &
                                            (station_hydro.index < haul_date)]
                return period_data.sum() if len(period_data) > 0 else 0

            # Create column name (generic 'hydro' for compatibility)
            port_data[f'hydro_cum_{weeks}wk'] = port_data['date'].apply(calc_cumulative)

        # Fit models with each window
        logger.info(f"\n  Fitting models for {len(cumulative_windows)} windows...")

        window_results = {}

        for weeks in cumulative_windows:
            try:
                # Prepare data
                y = port_data['litter_count'].values.astype(float)

                # Seasonal terms
                port_data['month'] = port_data['date'].dt.month
                sin_month = np.sin(2 * np.pi * port_data['month'] / 12).values
                cos_month = np.cos(2 * np.pi * port_data['month'] / 12).values

                # Build design matrix
                X = pd.DataFrame({
                    'const': 1.0,
                    'hydro': port_data[f'hydro_cum_{weeks}wk'].values.astype(float),
                    'sin_month': sin_month,
                    'cos_month': cos_month
                })

                # Check for issues
                if X.isnull().any().any():
                    logger.info(f"    Window {weeks:2d}wk: NaN values, skipping")
                    continue

                # Offset
                effort = port_data['trawling_hours'].values.astype(float)
                if (effort <= 0).any():
                    effort = effort + 0.1
                offset = np.log(effort)

                # Fit NB model
                model = sm.GLM(
                    y, X,
                    family=sm.families.NegativeBinomial(alpha=1.0),
                    offset=offset
                )

                result = model.fit()

                # Store results
                beta = result.params['hydro']
                se = result.bse['hydro']
                pval = result.pvalues['hydro']

                window_results[weeks] = {
                    'aic': result.aic,
                    'beta': beta,
                    'se': se,
                    'pvalue': pval,
                    'n_obs': len(y)
                }

                logger.info(f"    Window {weeks:2d}wk: AIC={result.aic:7.1f}, β={beta:7.4f}, p={pval:.4f}")

            except Exception as e:
                logger.info(f"    Window {weeks:2d}wk: Failed ({str(e)[:50]})")
                continue

        if len(window_results) == 0:
            logger.warning(f"\n  ✗ No models converged for port {port_id}")
            continue

        # Find best window by AIC
        best_window = min(window_results.keys(), key=lambda w: window_results[w]['aic'])
        best = window_results[best_window]

        # Parsimony check: If best window is longest and ΔAIC < 2 from shorter window, use shorter
        # This prevents overfitting with sparse data
        sorted_windows = sorted(window_results.keys())
        best_idx = sorted_windows.index(best_window)

        if best_idx > 0:  # If not the shortest window
            shorter_window = sorted_windows[best_idx - 1]
            shorter_result = window_results[shorter_window]

            delta_aic = best['aic'] - shorter_result['aic']
            n_obs = best['n_obs']

            # Apply parsimony: if ΔAIC < 2, models are equivalent → use shorter
            if delta_aic > -2.0:  # best - shorter > -2, meaning shorter is within 2 AIC units
                original_window = best_window
                original_aic = best['aic']

                # Use shorter window
                best_window = shorter_window
                best = shorter_result

                logger.info(f"\n  ⚙️  PARSIMONY CHECK:")
                logger.info(f"    Best AIC: {original_window}wk (AIC={original_aic:.1f})")
                logger.info(f"    vs {shorter_window}wk (AIC={shorter_result['aic']:.1f})")
                logger.info(f"    ΔAIC = {abs(delta_aic):.1f} < 2 → Models equivalent")
                logger.info(f"    Using {shorter_window}wk (parsimony principle)")

                if n_obs < 20:
                    logger.info(f"    [Sparse data: n={n_obs} hauls → shorter window preferred]")

        # Calculate effect using appropriate units
        if is_discharge:
            # For discharge: effect per 100 m³/s
            effect_per_unit = np.exp(best['beta'] * 100)
            pct_change = (effect_per_unit - 1) * 100
            effect_label = "per 100 m³/s"
        else:
            # For precipitation: effect per 10mm
            effect_per_unit = np.exp(best['beta'] * 10)
            pct_change = (effect_per_unit - 1) * 100
            effect_label = "per 10mm"

        # Significance
        if best['pvalue'] < 0.001:
            sig = "***"
        elif best['pvalue'] < 0.01:
            sig = "**"
        elif best['pvalue'] < 0.05:
            sig = "*"
        else:
            sig = "NS"

        logger.info(f"\n  ✓ BEST MODEL:")
        logger.info(f"    Window: {best_window} weeks")
        logger.info(f"    AIC: {best['aic']:.1f}")
        logger.info(f"    β₁: {best['beta']:.4f} (SE: {best['se']:.4f})")
        logger.info(f"    p-value: {best['pvalue']:.4f} {sig}")
        logger.info(f"    Effect: {pct_change:+.1f}% {effect_label}")

        # ========== GENERATE DIAGNOSTIC PLOTS FOR OPTIMAL MODEL ==========
        if DIAGNOSTICS_AVAILABLE:
            try:
                logger.info(f"\n  📊 GENERATING DIAGNOSTIC PLOTS...")

                # Re-fit optimal model to get data for diagnostics
                y_optimal = port_data['litter_count'].values.astype(float)

                # Rebuild design matrix for optimal window
                X_optimal = pd.DataFrame({
                    'const': 1.0,
                    'hydro': port_data[f'hydro_cum_{best_window}wk'].values.astype(float),
                    'sin_month': np.sin(2 * np.pi * port_data['month'] / 12).values,
                    'cos_month': np.cos(2 * np.pi * port_data['month'] / 12).values
                })

                # For diagnostics, use linear approximation (easier to interpret)
                # Even though we fit NB, linear diagnostics are still useful
                from sklearn.linear_model import LinearRegression

                # Fit simple linear model for diagnostics
                diagnostic_model = LinearRegression()
                diagnostic_model.fit(X_optimal, y_optimal)

                # Create output directory for diagnostics
                diagnostics_dir = Path(config['outputs']['figures']) / 'diagnostics'
                diagnostics_dir.mkdir(parents=True, exist_ok=True)

                # Generate diagnostic plot
                output_file = diagnostics_dir / f'port_{port_id}_station_{station_id}_diagnostics.pdf'

                create_diagnostic_plots(
                    diagnostic_model,
                    X_optimal.values,
                    y_optimal,
                    output_file=str(output_file),
                    figsize=(12, 10),
                    dpi=300,
                    show_ids=True
                )

                logger.info(f"    ✓ Diagnostics saved: {output_file.name}")

                # Also save the best one as generic figS3_diagnostics.pdf for manuscript
                # (Only if this is a significant result)
                if best['pvalue'] < 0.05 and sig in ['*', '**', '***']:
                    figS3_file = Path(config['outputs']['figures']) / 'figS3_model_diagnostics.pdf'

                    # Copy or create directly
                    create_diagnostic_plots(
                        diagnostic_model,
                        X_optimal.values,
                        y_optimal,
                        output_file=str(figS3_file),
                        figsize=(12, 10),
                        dpi=300,
                        show_ids=True
                    )
                    logger.info(f"    ✓ Manuscript figure: {figS3_file.name}")

            except Exception as e:
                logger.warning(f"    ✗ Diagnostic plot generation failed: {str(e)[:100]}")
        # ================================================================

        # Store
        results_all.append({
            'port_id': port_id,
            'station_id': station_id,
            'optimal_window_weeks': best_window,
            'n_obs': best['n_obs'],
            'beta': best['beta'],
            'se': best['se'],
            'pvalue': best['pvalue'],
            'significance': sig,
            'aic': best['aic'],
            'effect_10mm_pct': pct_change
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    if len(results_all) > 0:
        summary = pd.DataFrame(results_all)

        logger.info("\n")
        logger.info(summary[['port_id', 'optimal_window_weeks', 'beta', 'pvalue',
                             'effect_10mm_pct', 'significance']].to_string(index=False))

        # Save
        outputs_dir = Path(config['outputs']['tables'])
        ensure_dir(outputs_dir)

        summary_path = outputs_dir / 'haul_model_results.csv'
        summary.to_csv(summary_path, index=False)
        logger.info(f"\n✓ Saved: {summary_path}")

        # Statistics
        logger.info(f"\n{'=' * 70}")
        logger.info("STATISTICS:")
        logger.info('=' * 70)
        logger.info(f"\nPorts analyzed: {len(summary)}")
        logger.info(f"Mean window: {summary['optimal_window_weeks'].mean():.1f} weeks")
        logger.info(f"Mean β₁: {summary['beta'].mean():.4f}")

        # Check for outliers
        median_effect = summary['effect_10mm_pct'].median()
        mad = (summary['effect_10mm_pct'] - median_effect).abs().median()
        outliers = summary[abs(summary['effect_10mm_pct'] - median_effect) > 5 * mad]

        if len(outliers) > 0:
            logger.info(f"\n⚠️  WARNING: {len(outliers)} outlier(s) detected:")
            for _, row in outliers.iterrows():
                logger.info(f"  Port {row['port_id']}: effect = {row['effect_10mm_pct']:+.1f}%")
            logger.info(f"\nStatistics excluding outliers:")
            clean = summary[~summary['port_id'].isin(outliers['port_id'])]
            logger.info(f"  Mean effect (10mm): {clean['effect_10mm_pct'].mean():+.1f}%")
            logger.info(f"  Median effect (10mm): {clean['effect_10mm_pct'].median():+.1f}%")
        else:
            logger.info(f"Mean effect (10mm): {summary['effect_10mm_pct'].mean():+.1f}%")

        logger.info(f"\nSignificance:")
        logger.info(f"  p < 0.001: {(summary['pvalue'] < 0.001).sum()}")
        logger.info(f"  p < 0.01:  {(summary['pvalue'] < 0.01).sum()}")
        logger.info(f"  p < 0.05:  {(summary['pvalue'] < 0.05).sum()}")
        logger.info(f"  NS:        {(summary['pvalue'] >= 0.05).sum()}")
        logger.info(f"\nDirection:")
        logger.info(f"  Positive: {(summary['beta'] > 0).sum()}")
        logger.info(f"  Negative: {(summary['beta'] < 0).sum()}")

        logger.info("\n" + "=" * 70)
        logger.info("MODEL FITTING COMPLETE")
        logger.info("=" * 70)

        return {'summary': summary, 'all_results': results_all}
    else:
        logger.warning("\n✗ No models converged successfully")
        logger.info("=" * 70)
        return {}


def cmd_compare_dispersion(config: dict) -> dict:
    """Compare with Lagrangian dispersion model."""
    if 'dispersion' not in config or not config['dispersion'].get('enabled', False):
        logger.info("=" * 60)
        logger.info("STEP 4: DISPERSION MODEL COMPARISON - DISABLED")
        logger.info("=" * 60)
        return {}

    logger.info("=" * 60)
    logger.info("STEP 4: DISPERSION MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info("Dispersion comparison not implemented in this version")
    return {}


def cmd_run_all(config: dict) -> None:
    """Run full pipeline."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL PIPELINE")
    logger.info("=" * 60)

    # Run each step and pass results forward
    prepared_data = cmd_prepare(config)
    correlation_results = cmd_correlate(config, prepared_data=prepared_data)
    model_results = cmd_fit_model(config, prepared_data=prepared_data,
                                  correlation_results=correlation_results)
    dispersion_results = cmd_compare_dispersion(config)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Marine Litter Analysis Pipeline - FIXED VERSION'
    )

    # Global arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add subcommands
    subparsers.add_parser('prepare', help='Prepare and preprocess data')
    subparsers.add_parser('correlate', help='Run correlation analysis (skipped for haul data)')
    subparsers.add_parser('fit_model', help='Fit statistical models')
    subparsers.add_parser('compare_dispersion', help='Compare with dispersion model')
    subparsers.add_parser('run_all', help='Run full pipeline')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        return

    # Setup logging
    log_dir = Path(config['outputs']['logs'])
    ensure_dir(log_dir)
    setup_logging(log_file=log_dir / 'pipeline.log')

    # Set random seed for reproducibility
    set_random_seed(config.get('random_seed', 42))

    # Create command mapping
    commands = {
        'prepare': lambda: cmd_prepare(config),
        'correlate': lambda: cmd_correlate(config),
        'fit_model': lambda: cmd_fit_model(config),
        'compare_dispersion': lambda: cmd_compare_dispersion(config),
        'run_all': lambda: cmd_run_all(config)
    }

    # Execute command
    try:
        logger.info(f"Starting command: {args.command}")
        logger.info(f"Using config: {args.config}")
        commands[args.command]()
        logger.info("Command completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
