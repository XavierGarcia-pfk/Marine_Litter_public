"""
Data loading and standardization for marine litter analysis.

This module provides functions to load data from various sources
(CSV, SQLite, shapefiles) and standardize them into expected schemas.

Handles Catalan and Spanish column names commonly found in regional datasets.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import sqlite3
from shapely.geometry import Point

from src.utils import validate_dataframe_schema

logger = logging.getLogger('marine_litter.data_loading')


def load_litter_data(
    csv_path: Optional[str] = None,
    db_path: Optional[str] = None,
    table_name: str = 'marine_litter'
) -> pd.DataFrame:
    """
    Load marine litter observation data.

    Expected output schema:
        - date (datetime)
        - port_id (str)
        - lat (float, optional)
        - lon (float, optional)
        - litter_count (int)
        - litter_weight_kg (float, optional)
        - category (str, optional)
        - haul_id (str, optional)

    Parameters
    ----------
    csv_path : str, optional
        Path to CSV file.
    db_path : str, optional
        Path to SQLite database.
    table_name : str
        Name of table in database.

    Returns
    -------
    pd.DataFrame
        Standardized litter data.
    """
    # Load from CSV or database
    if csv_path:
        logger.info(f"Loading litter data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    elif db_path:
        logger.info(f"Loading litter data from database: {db_path}")
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    else:
        raise ValueError("Must provide either csv_path or db_path")

    # Adapter: map your actual columns to expected schema
    # This is dataset-specific and should be configured
    logger.info(f"Loaded litter data with shape: {df.shape}")
    logger.debug(f"Columns: {list(df.columns)}")

    return df


def adapt_litter_schema(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Adapt litter DataFrame to standard schema.

    Handles Catalan column names commonly found in Spanish/Catalan datasets:
    - 'Data' → 'date' (Date in Catalan)
    - 'CodiPort' → 'port_id' (Port Code in Catalan)
    - 'Numero_individus' → 'litter_count' (Number of individuals)
    - 'Pes_total' → 'litter_weight_kg' (Total weight)
    - 'Temps_pesca' → 'trawling_hours' (Fishing time)

    Parameters
    ----------
    df : pd.DataFrame
        Raw litter data.
    column_mapping : Dict[str, str], optional
        Mapping from source columns to target schema.
        If None, uses default Catalan translations.
        Example: {'Data': 'date', 'CodiPort': 'port_id', 'Count': 'litter_count'}

    Returns
    -------
    pd.DataFrame
        Standardized litter DataFrame.
    """
    # Default Catalan to English column mapping
    default_mapping = {
        'Data': 'date',              # Date (Catalan)
        'CodiPort': 'port_id',       # Port Code (Catalan)
        'Numero_individus': 'litter_count',   # Number of individuals
        'Pes_total': 'litter_weight_kg',      # Total weight
        'Temps_pesca': 'trawling_hours',      # Fishing time
        'Categoria': 'category',      # Category
        'Lance': 'haul_id',          # Haul/Lance
        'Latitud': 'lat',            # Latitude
        'Longitud': 'lon',           # Longitude

        # Xavier's specific dataset columns (fisheries monitoring data)
        'NumeroTotal': 'litter_count',              # Total number of items
        'NumeroTotalCalculat': 'litter_count',      # Calculated total number
        'PesTotal_g': 'litter_weight_g',            # Total weight in grams
        'PesTotalCalculat_g': 'litter_weight_g',    # Calculated total weight in grams
        'Duracio_h': 'trawling_hours',              # Duration in hours
        'LatitudMitjana': 'lat',                    # Average latitude
        'LongitudMitjana': 'lon',                   # Average longitude
        'NumeroPesca': 'haul_id',                   # Fishing/haul number
        'CodiPesca': 'haul_code',                   # Fishing/haul code
    }

    # Use provided mapping or default
    mapping_to_use = column_mapping if column_mapping is not None else default_mapping

    # Only rename columns that exist in the DataFrame
    actual_mapping = {k: v for k, v in mapping_to_use.items() if k in df.columns}

    # Handle duplicate targets: if multiple source columns map to same target,
    # keep only the first one (prioritize in order of appearance)
    seen_targets = {}
    filtered_mapping = {}

    # Define priority order for duplicate targets
    priority_order = {
        'litter_count': ['NumeroTotalCalculat', 'NumeroTotal'],  # Prefer calculated
        'litter_weight_g': ['PesTotalCalculat_g', 'PesTotal_g'],  # Prefer calculated
    }

    # First, handle prioritized columns
    for target, sources in priority_order.items():
        for source in sources:
            if source in actual_mapping and actual_mapping[source] == target:
                if target not in seen_targets:
                    filtered_mapping[source] = target
                    seen_targets[target] = source
                    logger.debug(f"Selected {source} → {target} (priority)")
                break

    # Then add remaining mappings that don't conflict
    for source, target in actual_mapping.items():
        if source not in filtered_mapping:  # Not already handled
            if target not in seen_targets:  # Target not already used
                filtered_mapping[source] = target
                seen_targets[target] = source
            else:
                logger.debug(f"Skipping {source} → {target} (duplicate target, keeping {seen_targets[target]})")

    if filtered_mapping:
        logger.debug(f"Applying column mapping: {filtered_mapping}")
        df = df.rename(columns=filtered_mapping)

    # Convert weight from grams to kilograms if needed
    if 'litter_weight_g' in df.columns:
        df['litter_weight_kg'] = df['litter_weight_g'] / 1000.0
        logger.info(f"Converted litter_weight_g to litter_weight_kg (n={len(df)} records)")
        df = df.drop(columns=['litter_weight_g'])

    # Ensure date is datetime (handle both original 'Data' and renamed 'date')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Data' in df.columns:
        # In case mapping wasn't applied
        df['date'] = pd.to_datetime(df['Data'])
        if 'Data' != 'date':
            df = df.drop(columns=['Data'])

    # Required columns
    required = ['date', 'port_id']
    validate_dataframe_schema(df, required, name="litter_df")

    # Check for litter measurement columns (not required, just informative)
    litter_measurement_cols = ['litter_count', 'litter_weight_kg', 'Numero_individus', 'Pes_total']
    found_measurements = [col for col in litter_measurement_cols if col in df.columns]

    if not found_measurements:
        logger.warning(
            f"No standard litter measurement columns found. "
            f"Available columns: {list(df.columns)}. "
            f"The pipeline will use any numeric columns for analysis."
        )
    else:
        logger.info(f"Found litter measurement columns: {found_measurements}")

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f"Adapted litter data: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Final columns: {list(df.columns)}")

    return df


def load_effort_data(
    csv_path: str,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load trawling effort data.

    Handles Catalan column names:
    - 'Data' → 'date'
    - 'CodiPort' → 'port_id'
    - 'Temps_pesca' → 'trawling_hours' (Fishing time)
    - 'Nombre_lances' → 'hauls' (Number of hauls)

    Expected schema:
        - date (datetime)
        - port_id (str)
        - trawling_hours (float)
        - hauls (int, optional)

    Parameters
    ----------
    csv_path : str
        Path to effort CSV file.
    column_mapping : Dict[str, str], optional
        Column name mapping. If None, uses default Catalan translations.

    Returns
    -------
    pd.DataFrame
        Standardized effort data.
    """
    logger.info(f"Loading effort data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Default Catalan to English mapping
    default_mapping = {
        'Data': 'date',
        'CodiPort': 'port_id',
        'Temps_pesca': 'trawling_hours',
        'Nombre_lances': 'hauls',
    }

    # Use provided mapping or default
    mapping_to_use = column_mapping if column_mapping is not None else default_mapping

    # Only rename columns that exist
    actual_mapping = {k: v for k, v in mapping_to_use.items() if k in df.columns}

    if actual_mapping:
        logger.debug(f"Applying column mapping: {actual_mapping}")
        df = df.rename(columns=actual_mapping)

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Data' in df.columns:
        df['date'] = pd.to_datetime(df['Data'])
        df = df.drop(columns=['Data'])

    # Required columns
    required = ['date', 'port_id', 'trawling_hours']
    validate_dataframe_schema(df, required, name="effort_df")

    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f"Loaded effort data: {len(df)} records")

    return df


def load_hydro_data(
    csv_path: str,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load river/meteorological data (precipitation, discharge).

    Handles Catalan/Spanish column names:
    - 'Data' or 'Fecha' → 'date'
    - 'Estacio' or 'Codigo_estacion' → 'river_id' (Station code)
    - 'Precipitacio' or 'Precipitacion' → 'precip_mm'
    - 'Cabal' or 'Caudal' → 'discharge_m3s' (Flow rate)
    - 'Latitud' → 'lat'
    - 'Longitud' → 'lon'

    Expected schema:
        - date (datetime)
        - river_id (str or Station_Code)
        - precip_mm (float, optional)
        - discharge_m3s (float, optional)
        - lat (float, optional)
        - lon (float, optional)

    Parameters
    ----------
    csv_path : str
        Path to hydrology CSV file.
    column_mapping : Dict[str, str], optional
        Column name mapping. If None, uses default Catalan/Spanish translations.

    Returns
    -------
    pd.DataFrame
        Standardized hydrology data.
    """
    logger.info(f"Loading hydrology data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Default Catalan/Spanish to English mapping
    default_mapping = {
        # Generic Catalan/Spanish names
        'Data': 'date',
        'Fecha': 'date',
        'Estacio': 'river_id',
        'Codigo_estacion': 'river_id',
        'Station_Code': 'river_id',
        'Precipitacio': 'precip_mm',
        'Precipitacion': 'precip_mm',
        'Cabal': 'discharge_m3s',
        'Caudal': 'discharge_m3s',
        'Latitud': 'lat',
        'Longitud': 'lon',
        # XEMA (Catalan Meteorological Network) specific names
        'DATA_LECTURA': 'date',           # Reading date
        'CODI_ESTACIO': 'river_id',       # Station code
        'VALOR_LECTURA': 'precip_mm',     # Reading value (will need to filter by variable)
        'CODI_VARIABLE': 'variable_code', # Variable code (35 = precipitation)
    }

    # Use provided mapping or default
    mapping_to_use = column_mapping if column_mapping is not None else default_mapping

    # Only rename columns that exist
    actual_mapping = {k: v for k, v in mapping_to_use.items() if k in df.columns}

    if actual_mapping:
        logger.debug(f"Applying column mapping: {actual_mapping}")
        df = df.rename(columns=actual_mapping)

    # Special handling for XEMA (Catalan Meteorological Network) format
    # XEMA data is in long format with variable codes
    if 'variable_code' in df.columns:
        logger.info("Detected XEMA format data")

        # Filter for precipitation data (code 35 is precipitation)
        # Common codes: 35 = precipitation, 32 = temperature, 30 = wind speed
        precip_codes = [35, '35']
        if df['variable_code'].dtype == object:
            # Variable code might be numeric or string
            df_precip = df[df['variable_code'].astype(str).isin(['35'])]
        else:
            df_precip = df[df['variable_code'].isin(precip_codes)]

        if len(df_precip) > 0:
            df = df_precip
            logger.info(f"Filtered to precipitation data: {len(df)} records")
        else:
            logger.warning(f"No precipitation data found (variable code 35). "
                          f"Available codes: {df['variable_code'].unique()}")

        # Drop the variable_code column as it's now redundant
        df = df.drop(columns=['variable_code'], errors='ignore')

    # Ensure date is datetime (handle various formats including XEMA format)
    if 'date' in df.columns:
        # Try to parse with automatic format detection (handles "01/07/2017 12:00:00 AM")
        try:
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        except:
            # Fallback to default parsing
            df['date'] = pd.to_datetime(df['date'])
    elif 'Data' in df.columns:
        df['date'] = pd.to_datetime(df['Data'], dayfirst=True)
        df = df.drop(columns=['Data'])
    elif 'Fecha' in df.columns:
        df['date'] = pd.to_datetime(df['Fecha'], dayfirst=True)
        df = df.drop(columns=['Fecha'])

    # Required: date and river_id, plus at least one measurement
    required = ['date', 'river_id']
    validate_dataframe_schema(df, required, name="hydro_df")

    # Clean up unnecessary columns from XEMA format
    xema_drop_cols = ['ID', 'CODI_ESTAT', 'CODI_BASE', 'DATA_EXTREM']
    df = df.drop(columns=[col for col in xema_drop_cols if col in df.columns], errors='ignore')

    # Check for at least one measurement column
    measurements = ['precip_mm', 'discharge_m3s']
    if not any(col in df.columns for col in measurements):
        available = list(df.columns)
        raise ValueError(
            f"hydro_df must contain at least one of: {measurements}. "
            f"Available columns after mapping: {available}. "
            f"If using XEMA data, make sure VALOR_LECTURA was mapped to precip_mm."
        )

    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f"Loaded hydro data: {len(df)} records, {df['river_id'].nunique()} unique stations")

    return df


def load_ports_metadata(
    csv_path: Optional[str] = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load port metadata.

    Handles Catalan/Spanish column names:
    - 'CodiPort' or 'Codigo_puerto' → 'port_id'
    - 'Nom' or 'Nombre' → 'name'
    - 'Latitud' → 'lat'
    - 'Longitud' → 'lon'

    Expected schema:
        - port_id (str)
        - lat (float)
        - lon (float)
        - name (str, optional)

    Parameters
    ----------
    csv_path : str, optional
        Path to ports CSV.
    gdf : gpd.GeoDataFrame, optional
        GeoDataFrame with port geometries.
    column_mapping : Dict[str, str], optional
        Column name mapping. If None, uses default Catalan/Spanish translations.

    Returns
    -------
    pd.DataFrame
        Port metadata.
    """
    if csv_path:
        logger.info(f"Loading ports metadata from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Default Catalan/Spanish to English mapping
        default_mapping = {
            'CodiPort': 'port_id',
            'Codigo_puerto': 'port_id',
            'Nom': 'name',
            'Nombre': 'name',
            'NomPort': 'name',
            'Latitud': 'lat',
            'Longitud': 'lon',
        }

        # Use provided mapping or default
        mapping_to_use = column_mapping if column_mapping is not None else default_mapping

        # Only rename columns that exist
        actual_mapping = {k: v for k, v in mapping_to_use.items() if k in df.columns}

        if actual_mapping:
            logger.debug(f"Applying column mapping: {actual_mapping}")
            df = df.rename(columns=actual_mapping)

    elif gdf is not None:
        logger.info("Using provided ports GeoDataFrame")
        # Extract lat/lon from geometry
        df = gdf.copy()
        if 'geometry' in df.columns:
            df['lon'] = df.geometry.x
            df['lat'] = df.geometry.y
        df = pd.DataFrame(df.drop(columns='geometry', errors='ignore'))

        # Apply column mapping if provided
        if column_mapping:
            actual_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            if actual_mapping:
                df = df.rename(columns=actual_mapping)
    else:
        raise ValueError("Must provide either csv_path or gdf")

    required = ['port_id', 'lat', 'lon']
    validate_dataframe_schema(df, required, name="ports_df")

    logger.info(f"Loaded {len(df)} ports")

    return df


def load_xema_stations(
    csv_path: str,
    active_only: bool = True
) -> pd.DataFrame:
    """
    Load meteorological stations or discharge point metadata with coordinates.

    Handles both:
    - XEMA format: 'Codi', 'Latitud', 'Longitud', 'Estat actual'
    - Generic format: 'source_id', 'lat', 'lon', 'source_type'

    Parameters
    ----------
    csv_path : str
        Path to stations CSV file.
    active_only : bool
        If True, only load active stations (XEMA format only).

    Returns
    -------
    pd.DataFrame
        Stations with columns: station_id, lat, lon, name (optional)
    """
    logger.info(f"Loading stations from: {csv_path}")

    # Try to detect format by reading first row
    try:
        # Try XEMA format first (has header row to skip)
        df_test = pd.read_csv(csv_path, nrows=1, skiprows=1)
        if 'Codi' in df_test.columns:
            # XEMA format
            logger.info("Detected XEMA stations format")
            df = pd.read_csv(csv_path, skiprows=1)

            # Filter to active stations if requested
            if active_only and 'Estat actual' in df.columns:
                df = df[df['Estat actual'] == 'Operativa']
                logger.info(f"Filtered to active stations: {len(df)}")

            # Create clean dataframe
            stations_df = pd.DataFrame({
                'station_id': df['Codi'].str.strip(),
                'lat': pd.to_numeric(df['Latitud'], errors='coerce'),
                'lon': pd.to_numeric(df['Longitud'], errors='coerce')
            })

            # Add station name if available
            if 'Estació [Codi]' in df.columns:
                stations_df['name'] = df['Estació [Codi]']

        else:
            raise ValueError("Not XEMA format, trying generic...")

    except:
        # Try generic format (coastal discharge, etc.)
        logger.info("Detected generic source format (e.g., coastal discharge)")
        df = pd.read_csv(csv_path)

        # Map generic column names
        if 'source_id' in df.columns and 'lat' in df.columns and 'lon' in df.columns:
            stations_df = pd.DataFrame({
                'station_id': df['source_id'].astype(str),
                'lat': pd.to_numeric(df['lat'], errors='coerce'),
                'lon': pd.to_numeric(df['lon'], errors='coerce')
            })

            # Add name if available
            if 'source_name' in df.columns:
                stations_df['name'] = df['source_name']
            elif 'name' in df.columns:
                stations_df['name'] = df['name']

        else:
            raise ValueError(
                f"Unrecognized format. Expected either:\n"
                f"  - XEMA: 'Codi', 'Latitud', 'Longitud'\n"
                f"  - Generic: 'source_id', 'lat', 'lon'\n"
                f"Got columns: {list(df.columns)}"
            )

    # Remove any stations with missing coordinates
    stations_df = stations_df.dropna(subset=['lat', 'lon'])

    # Remove duplicates (keep first occurrence)
    stations_df = stations_df.drop_duplicates(subset=['station_id'])

    logger.info(f"Loaded {len(stations_df)} stations with coordinates")
    logger.info(f"  Latitude range: {stations_df['lat'].min():.4f} to {stations_df['lat'].max():.4f}")
    logger.info(f"  Longitude range: {stations_df['lon'].min():.4f} to {stations_df['lon'].max():.4f}")

    return stations_df


def load_wastewater_data(
    csv_path: str,
    column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Load wastewater treatment plant data.

    Expected schema:
        - plant_id (str)
        - lat (float)
        - lon (float)
        - capacity_m3_day (float, optional)
        - river_id (str, optional)

    Parameters
    ----------
    csv_path : str
        Path to wastewater CSV.
    column_mapping : Dict[str, str], optional
        Column name mapping.

    Returns
    -------
    pd.DataFrame
        Wastewater plant data.
    """
    logger.info(f"Loading wastewater data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if column_mapping:
        df = df.rename(columns=column_mapping)

    required = ['plant_id', 'lat', 'lon']
    validate_dataframe_schema(df, required, name="wastewater_df")

    logger.info(f"Loaded {len(df)} wastewater plants")

    return df


def load_dispersion_data(
    particles_path: Optional[str] = None,
    aggregates_path: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load Lagrangian dispersion model outputs.

    Particles schema:
        - date (datetime)
        - lat (float)
        - lon (float)
        - depth_m (float)
        - river_id (str)
        - settled_flag (bool)
        - particle_id (int)

    Aggregates schema:
        - date (datetime)
        - port_id (str)
        - settled_particles (int)
        - flux_index (float)

    Parameters
    ----------
    particles_path : str, optional
        Path to particle trajectories CSV.
    aggregates_path : str, optional
        Path to aggregated dispersion CSV.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (particles_df, aggregates_df) or (None, None) if not available.
    """
    particles_df = None
    aggregates_df = None

    if particles_path and Path(particles_path).exists():
        logger.info(f"Loading dispersion particles from: {particles_path}")
        particles_df = pd.read_csv(particles_path)
        particles_df['date'] = pd.to_datetime(particles_df['date'])
        logger.info(f"Loaded {len(particles_df)} particle records")

    if aggregates_path and Path(aggregates_path).exists():
        logger.info(f"Loading dispersion aggregates from: {aggregates_path}")
        aggregates_df = pd.read_csv(aggregates_path)
        aggregates_df['date'] = pd.to_datetime(aggregates_df['date'])
        logger.info(f"Loaded {len(aggregates_df)} aggregate records")

    return particles_df, aggregates_df


def load_shapefile(
    file_path: str,
    target_crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Load and reproject a shapefile.

    Parameters
    ----------
    file_path : str
        Path to shapefile.
    target_crs : str
        Target coordinate reference system.

    Returns
    -------
    gpd.GeoDataFrame
        Loaded and reprojected GeoDataFrame.
    """
    logger.info(f"Loading shapefile: {file_path}")
    gdf = gpd.read_file(file_path)

    if gdf.crs is not None and gdf.crs != target_crs:
        logger.debug(f"Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    logger.info(f"Loaded shapefile with {len(gdf)} features")

    return gdf


def create_mock_data(output_dir: str = "data/mock") -> Dict[str, str]:
    """
    Create mock datasets for testing the pipeline.

    Parameters
    ----------
    output_dir : str
        Directory to save mock data files.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping data type to file path.
    """
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Mock litter data
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='W')
    ports = ['Barcelona', 'Blanes', 'La Ràpita', "L'Ametlla de Mar"]

    litter_data = []
    for date in dates:
        for port in ports:
            litter_data.append({
                'date': date,
                'port_id': port,
                'litter_count': max(0, int(np.random.poisson(50) + np.random.randn() * 10)),
                'litter_weight_kg': max(0, np.random.exponential(20))
            })

    litter_df = pd.DataFrame(litter_data)
    litter_path = f"{output_dir}/litter.csv"
    litter_df.to_csv(litter_path, index=False)
    logger.info(f"Created mock litter data: {litter_path}")

    # Mock effort data
    effort_data = []
    for date in dates:
        for port in ports:
            effort_data.append({
                'date': date,
                'port_id': port,
                'trawling_hours': max(1, np.random.poisson(50)),
                'hauls': max(1, int(np.random.poisson(10)))
            })

    effort_df = pd.DataFrame(effort_data)
    effort_path = f"{output_dir}/effort.csv"
    effort_df.to_csv(effort_path, index=False)
    logger.info(f"Created mock effort data: {effort_path}")

    # Mock hydrology data
    dates_daily = pd.date_range('2019-01-01', '2023-12-31', freq='D')
    rivers = ['Tordera', 'Llobregat', 'Besòs', 'Ebro']

    hydro_data = []
    for date in dates_daily:
        for river in rivers:
            # Simulate seasonal pattern with noise
            doy = date.dayofyear
            seasonal = 10 * np.sin(2 * np.pi * doy / 365.25) + 15
            precip = max(0, seasonal + np.random.exponential(5))

            hydro_data.append({
                'date': date,
                'river_id': river,
                'precip_mm': precip,
                'discharge_m3s': precip * 2 + np.random.randn() * 5
            })

    hydro_df = pd.DataFrame(hydro_data)
    hydro_path = f"{output_dir}/hydrology.csv"
    hydro_df.to_csv(hydro_path, index=False)
    logger.info(f"Created mock hydrology data: {hydro_path}")

    # Mock ports metadata
    port_locs = {
        'Barcelona': (41.38, 2.17),
        'Blanes': (41.67, 2.80),
        'La Ràpita': (40.62, 0.59),
        "L'Ametlla de Mar": (40.89, 0.80)
    }

    ports_data = [
        {'port_id': pid, 'lat': lat, 'lon': lon, 'name': pid}
        for pid, (lat, lon) in port_locs.items()
    ]

    ports_df = pd.DataFrame(ports_data)
    ports_path = f"{output_dir}/ports.csv"
    ports_df.to_csv(ports_path, index=False)
    logger.info(f"Created mock ports data: {ports_path}")

    return {
        'litter': litter_path,
        'effort': effort_path,
        'hydrology': hydro_path,
        'ports': ports_path
    }
