"""
River-to-port spatial mapping.

This module handles the spatial matching of rivers/stations to ports,
considering distance, direction (preferring upstream/north sources),
and manual overrides.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import haversine_distance, is_north_of

logger = logging.getLogger('marine_litter.mapping')


def map_rivers_to_ports(
        ports_df: pd.DataFrame,
        rivers_df: pd.DataFrame,
        max_distance_km: float = 200,
        n_closest: int = 3,
        prefer_north: bool = True,
        manual_overrides: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Map rivers to ports based on proximity and circulation patterns.

    Parameters
    ----------
    ports_df : pd.DataFrame
        Port metadata with columns: port_id, lat, lon.
    rivers_df : pd.DataFrame
        River/station metadata with columns: river_id, lat, lon.
    max_distance_km : float
        Maximum distance to consider (kilometers).
    n_closest : int
        Number of closest rivers to map per port.
    prefer_north : bool
        Prefer rivers north of the port (following general circulation).
    manual_overrides : Dict[str, List[str]], optional
        Manual port_id -> [river_ids] mappings to override automatic matching.

    Returns
    -------
    Dict[str, List[Tuple[str, float]]]
        Mapping from port_id to list of (river_id, distance_km) tuples.
    """
    logger.info(f"Mapping rivers to ports (max_distance={max_distance_km} km, n_closest={n_closest})")

    mapping = {}

    for _, port in ports_df.iterrows():
        port_id = port['port_id']

        # Ensure port_id is integer (not float)
        if isinstance(port_id, (float, np.floating)):
            port_id = int(port_id)

        port_lat = port['lat']
        port_lon = port['lon']

        # Check for manual override
        if manual_overrides and port_id in manual_overrides:
            # Get distances for manual rivers
            manual_rivers = []
            for river_id in manual_overrides[port_id]:
                river_row = rivers_df[rivers_df['river_id'] == river_id]
                if not river_row.empty:
                    river_lat = river_row.iloc[0]['lat']
                    river_lon = river_row.iloc[0]['lon']
                    dist = haversine_distance(port_lat, port_lon, river_lat, river_lon)
                    manual_rivers.append((river_id, dist))

            mapping[port_id] = sorted(manual_rivers, key=lambda x: x[1])
            logger.info(f"Port {port_id}: using manual override with {len(manual_rivers)} rivers")
            continue

        # Compute distances to all rivers
        distances = []
        for _, river in rivers_df.iterrows():
            river_id = river['river_id']
            river_lat = river['lat']
            river_lon = river['lon']

            dist = haversine_distance(port_lat, port_lon, river_lat, river_lon)

            # Apply distance filter
            if dist > max_distance_km:
                continue

            # Apply directional preference
            if prefer_north:
                if is_north_of(river_lat, port_lat, tolerance=0.1):
                    # Boost northern rivers (reduce effective distance)
                    effective_dist = dist * 0.8
                else:
                    # Penalize southern rivers
                    effective_dist = dist * 1.2
            else:
                effective_dist = dist

            distances.append((river_id, dist, effective_dist))

        # Sort by effective distance and take top n_closest
        distances.sort(key=lambda x: x[2])
        selected = [(rid, d) for rid, d, _ in distances[:n_closest]]

        mapping[port_id] = selected

        river_names = ", ".join([f"{rid} ({d:.1f} km)" for rid, d in selected])
        logger.info(f"Port {port_id}: mapped to {len(selected)} rivers: {river_names}")

    return mapping


def create_river_port_dataframe(
        mapping: Dict[str, List[Tuple[str, float]]],
        rivers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert mapping dictionary to a tidy DataFrame.

    Parameters
    ----------
    mapping : Dict[str, List[Tuple[str, float]]]
        Port to rivers mapping.
    rivers_df : pd.DataFrame
        River metadata for additional info.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: port_id, river_id, distance_km, river_lat, river_lon.
    """
    records = []

    for port_id, rivers in mapping.items():
        for river_id, distance_km in rivers:
            # Get river metadata
            river_row = rivers_df[rivers_df['river_id'] == river_id]

            if not river_row.empty:
                record = {
                    'port_id': port_id,
                    'river_id': river_id,
                    'distance_km': distance_km,
                    'river_lat': river_row.iloc[0]['lat'],
                    'river_lon': river_row.iloc[0]['lon']
                }
                records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Created river-port mapping DataFrame with {len(df)} pairs")

    return df


def get_relevant_rivers_for_port(
        port_id: str,
        mapping: Dict[str, List[Tuple[str, float]]]
) -> List[str]:
    """
    Get list of river IDs relevant to a specific port.

    Parameters
    ----------
    port_id : str
        Port identifier.
    mapping : Dict[str, List[Tuple[str, float]]]
        Port to rivers mapping.

    Returns
    -------
    List[str]
        List of river IDs.
    """
    if port_id not in mapping:
        logger.warning(f"Port {port_id} not found in mapping")
        return []

    return [river_id for river_id, _ in mapping[port_id]]


def validate_mapping(
        mapping: Dict[str, List[Tuple[str, float]]],
        ports_df: pd.DataFrame,
        rivers_df: pd.DataFrame
) -> bool:
    """
    Validate that mapping is complete and sensible.

    Parameters
    ----------
    mapping : Dict
        Port to rivers mapping.
    ports_df : pd.DataFrame
        Port metadata.
    rivers_df : pd.DataFrame
        River metadata.

    Returns
    -------
    bool
        True if validation passes.
    """
    all_ports = set(ports_df['port_id'])
    mapped_ports = set(mapping.keys())

    missing = all_ports - mapped_ports
    if missing:
        logger.error(f"Ports without river mapping: {missing}")
        return False

    # Check that all river_ids exist
    all_rivers = set(rivers_df['river_id'])
    for port_id, rivers in mapping.items():
        for river_id, _ in rivers:
            if river_id not in all_rivers:
                logger.error(f"Invalid river_id '{river_id}' in mapping for port {port_id}")
                return False

    # Check for empty mappings
    for port_id, rivers in mapping.items():
        if not rivers:
            logger.warning(f"Port {port_id} has no mapped rivers")

    logger.info("Mapping validation passed")
    return True


def inverse_distance_weighting(
        values: np.ndarray,
        distances: np.ndarray,
        power: float = 2.0
) -> float:
    """
    Compute inverse distance weighted average.

    Parameters
    ----------
    values : np.ndarray
        Values to average (e.g., precipitation from multiple stations).
    distances : np.ndarray
        Distances corresponding to each value.
    power : float
        Power for inverse distance (higher = more weight to closer stations).

    Returns
    -------
    float
        Weighted average.
    """
    # Avoid division by zero
    distances = np.maximum(distances, 0.1)

    weights = 1.0 / (distances ** power)
    weighted_avg = np.sum(values * weights) / np.sum(weights)

    return weighted_avg


def aggregate_river_measurements_for_port(
        hydro_df: pd.DataFrame,
        port_id: str,
        mapping: Dict[str, List[Tuple[str, float]]],
        method: str = 'idw',
        date_col: str = 'date',
        value_col: str = 'precip_mm'
) -> pd.Series:
    """
    Aggregate river measurements for a port using mapping.

    Parameters
    ----------
    hydro_df : pd.DataFrame
        Hydrology data with columns: date, river_id, value_col.
    port_id : str
        Port identifier.
    mapping : Dict
        Port to rivers mapping.
    method : str
        Aggregation method: 'mean', 'idw' (inverse distance weighted), 'nearest'.
    date_col : str
        Name of date column.
    value_col : str
        Name of value column to aggregate.

    Returns
    -------
    pd.Series
        Time series of aggregated values for the port.
    """
    if port_id not in mapping:
        raise ValueError(f"Port {port_id} not in mapping")

    river_info = mapping[port_id]
    river_ids = [rid for rid, _ in river_info]

    # Filter hydrology data for relevant rivers
    port_hydro = hydro_df[hydro_df['river_id'].isin(river_ids)].copy()

    if method == 'mean':
        # Simple average across rivers
        result = port_hydro.groupby(date_col)[value_col].mean()

    elif method == 'idw':
        # Inverse distance weighted
        distance_map = {rid: dist for rid, dist in river_info}

        def weighted_avg(group):
            values = group[value_col].values
            river_ids_in_group = group['river_id'].values
            distances = np.array([distance_map[rid] for rid in river_ids_in_group])
            return inverse_distance_weighting(values, distances)

        result = port_hydro.groupby(date_col).apply(weighted_avg)

    elif method == 'nearest':
        # Use only the nearest river
        nearest_river = river_info[0][0]  # First in sorted list
        result = port_hydro[port_hydro['river_id'] == nearest_river].set_index(date_col)[value_col]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return result
