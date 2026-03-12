"""
Visualization functions for marine litter analysis.

This module provides plotting functions for:
- Lag-correlation curves
- Heatmaps (lag × window)
- Time series comparisons
- Model diagnostics
- Partial dependence plots
- Dispersion comparison plots
- Spatial maps
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd

logger = logging.getLogger('marine_litter.visualization')


def setup_plotting_style(style: str = 'seaborn-v0_8-darkgrid') -> None:
    """
    Set up consistent plotting style.

    Parameters
    ----------
    style : str
        Matplotlib style name.
    """
    plt.style.use(style)
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_lag_correlation_curve(
    corr_df: pd.DataFrame,
    port_id: str,
    river_id: str,
    rolling_window: int,
    output_path: Optional[str] = None,
    show_ci: bool = False,
    time_unit: str = 'periods'
) -> plt.Figure:
    """
    Plot lag-correlation curve for one port-river pair.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation results with columns: lag, correlation, p_value, significant.
    port_id : str
        Port identifier.
    river_id : str
        River identifier.
    rolling_window : int
        Rolling window size used.
    output_path : str, optional
        Path to save figure.
    show_ci : bool
        Whether to show confidence intervals.
    time_unit : str
        Time unit for labels ('days', 'weeks', 'months', 'periods').

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Filter data
    subset = corr_df[
        (corr_df['port_id'] == port_id) &
        (corr_df['river_id'] == river_id) &
        (corr_df['rolling_window'] == rolling_window)
    ].copy()

    if subset.empty:
        logger.warning(f"No data for {port_id}-{river_id}-{rolling_window}")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot correlation vs lag
    ax.plot(subset['lag'], subset['correlation'],
           'o-', linewidth=2, markersize=4, color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Shade significant regions
    if 'significant' in subset.columns:
        significant = subset[subset['significant']]
        if not significant.empty:
            ax.scatter(significant['lag'], significant['correlation'],
                      color='red', s=80, alpha=0.7, zorder=5,
                      label='Significant (FDR corrected)')

    # Mark best lag
    best_idx = subset['correlation'].abs().idxmax()
    best = subset.loc[best_idx]
    ax.axvline(best['lag'], color='crimson', linestyle=':',
              alpha=0.7, linewidth=2.5,
              label=f"Best lag = {best['lag']:.0f} (ρ = {best['correlation']:.3f})")

    # Add confidence interval if available
    if show_ci and 'lower_ci' in subset.columns and 'upper_ci' in subset.columns:
        ax.fill_between(subset['lag'], subset['lower_ci'], subset['upper_ci'],
                       alpha=0.2, color='steelblue', label='95% CI')

    # Labels and formatting
    ax.set_xlabel(f"Lag ({time_unit})", fontsize=13)
    ax.set_ylabel("Spearman Correlation", fontsize=13)
    ax.set_title(f"Lag-Correlation: {port_id} Litter vs {river_id} Precipitation\n"
                f"(Rolling window: {rolling_window} {time_unit})", fontsize=14, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    # Add p-value annotation for best lag
    if 'p_value' in subset.columns:
        pval_text = f"p = {best['p_value']:.4f}" if best['p_value'] >= 0.0001 else "p < 0.0001"
        ax.text(0.02, 0.98, pval_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    port_id: Optional[str] = None,
    river_id: Optional[str] = None,
    output_path: Optional[str] = None,
    time_unit: str = 'periods'
) -> plt.Figure:
    """
    Plot heatmap of correlation vs lag and rolling window.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation results.
    port_id : str, optional
        Filter by port.
    river_id : str, optional
        Filter by river.
    output_path : str, optional
        Path to save figure.
    time_unit : str
        Time unit for labels ('days', 'weeks', 'months', 'periods').

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Filter data
    data = corr_df.copy()
    if port_id:
        data = data[data['port_id'] == port_id]
    if river_id:
        data = data[data['river_id'] == river_id]

    if data.empty:
        logger.warning("No data for heatmap")
        return None

    # Pivot data
    pivot = data.pivot_table(
        values='correlation',
        index='lag',
        columns='rolling_window',
        aggfunc='mean'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0,
        vmin=-0.6, vmax=0.6,
        cbar_kws={'label': 'Spearman Correlation', 'shrink': 0.8},
        ax=ax,
        linewidths=0
    )

    # Labels
    ax.set_xlabel(f"Rolling Window ({time_unit})", fontsize=13)
    ax.set_ylabel(f"Lag ({time_unit})", fontsize=13)

    title = "Correlation Heatmap: Lag × Rolling Window"
    if port_id:
        title += f"\n{port_id}"
    if river_id:
        title += f" - {river_id}"
    ax.set_title(title, fontsize=14, pad=15)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")

    return fig


def plot_litter_timeseries(
    litter_df: pd.DataFrame,
    port_ids: Optional[List[str]] = None,
    value_col: str = 'litter_count',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot litter time series for multiple ports.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data with MultiIndex (date, port_id).
    port_ids : List[str], optional
        Ports to plot. If None, plots all.
    value_col : str
        Column to plot.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if port_ids is None:
        port_ids = litter_df.index.get_level_values('port_id').unique()

    n_ports = len(port_ids)
    n_cols = 2
    n_rows = (n_ports + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_ports > 1 else [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_ports))

    for i, port_id in enumerate(port_ids):
        ax = axes[i]

        # Extract port data
        port_data = litter_df.loc[(slice(None), port_id), value_col].droplevel('port_id')

        # Plot
        ax.plot(port_data.index, port_data.values,
               color=colors[i], linewidth=1.5, alpha=0.7)
        ax.fill_between(port_data.index, 0, port_data.values,
                       alpha=0.3, color=colors[i])

        # Labels
        ax.set_title(f"{port_id}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3)

        # Summary statistics
        mean_val = port_data.mean()
        ax.axhline(mean_val, color='red', linestyle='--',
                  alpha=0.5, linewidth=1, label=f'Mean: {mean_val:.1f}')
        ax.legend(fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Litter Time Series by Port", fontsize=16, y=1.00)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved time series plot to {output_path}")

    return fig


def plot_hydrology_timeseries(
    hydro_df: pd.DataFrame,
    river_ids: Optional[List[str]] = None,
    value_col: str = 'precip_mm',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot hydrology time series for multiple rivers.

    Parameters
    ----------
    hydro_df : pd.DataFrame
        Hydrology data with MultiIndex (date, river_id).
    river_ids : List[str], optional
        Rivers to plot.
    value_col : str
        Column to plot.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if river_ids is None:
        river_ids = hydro_df.index.get_level_values('river_id').unique()

    fig, ax = plt.subplots(figsize=(15, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(river_ids)))

    for i, river_id in enumerate(river_ids):
        river_data = hydro_df.loc[(slice(None), river_id), value_col].droplevel('river_id')

        # Plot with rolling mean overlay
        ax.plot(river_data.index, river_data.values,
               color=colors[i], alpha=0.3, linewidth=0.8)

        # Rolling mean
        rolling = river_data.rolling(30, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values,
               color=colors[i], linewidth=2, label=river_id)

    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=13)
    ax.set_title(f"Hydrology Time Series (30-day rolling mean)", fontsize=14, pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hydrology plot to {output_path}")

    return fig


def plot_model_residuals(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model residual diagnostics.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    fitted_values : np.ndarray
        Fitted values.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel("Fitted Values", fontsize=11)
    axes[0, 0].set_ylabel("Residuals", fontsize=11)
    axes[0, 0].set_title("Residuals vs Fitted", fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q Plot", fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scale-Location
    standardized_resid = residuals / np.std(residuals)
    axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_resid)),
                      alpha=0.5, s=20)
    axes[1, 0].set_xlabel("Fitted Values", fontsize=11)
    axes[1, 0].set_ylabel("√|Standardized Residuals|", fontsize=11)
    axes[1, 0].set_title("Scale-Location", fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Histogram of Residuals
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel("Residuals", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].set_title("Residual Histogram", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Residual Diagnostics", fontsize=16, y=1.00)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residual plots to {output_path}")

    return fig


def plot_partial_dependence(
    feature_values: np.ndarray,
    partial_dependence: np.ndarray,
    feature_name: str,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot partial dependence for a feature.

    Parameters
    ----------
    feature_values : np.ndarray
        Feature values (X-axis).
    partial_dependence : np.ndarray
        Partial dependence values (Y-axis).
    feature_name : str
        Feature name for labels.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(feature_values, partial_dependence, linewidth=3, color='steelblue')
    ax.fill_between(feature_values, partial_dependence,
                    alpha=0.3, color='steelblue')

    ax.set_xlabel(feature_name.replace('_', ' ').title(), fontsize=13)
    ax.set_ylabel("Partial Dependence", fontsize=13)
    ax.set_title(f"Partial Dependence: {feature_name}", fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved partial dependence plot to {output_path}")

    return fig


def plot_dispersion_comparison(
    comparison_df: pd.DataFrame,
    port_id: str,
    litter_col: str = 'litter_count',
    dispersion_col: str = 'settled_particles',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot litter vs dispersion model comparison.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Aligned litter and dispersion data.
    port_id : str
        Port identifier.
    litter_col : str
        Litter column name.
    dispersion_col : str
        Dispersion column name.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Normalize for visual comparison
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    litter_norm = scaler.fit_transform(comparison_df[[litter_col]]).flatten()
    dispersion_norm = scaler.fit_transform(comparison_df[[dispersion_col]]).flatten()

    dates = comparison_df['date']

    # Top panel: Time series
    axes[0].plot(dates, litter_norm, 'o-', label='Observed Litter (normalized)',
                color='steelblue', linewidth=2, markersize=4, alpha=0.7)
    axes[0].plot(dates, dispersion_norm, 's-', label='Dispersion Model (normalized)',
                color='coral', linewidth=2, markersize=4, alpha=0.7)

    axes[0].set_ylabel("Normalized Value", fontsize=12)
    axes[0].set_title(f"Litter vs Dispersion Model - {port_id}",
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: Scatter plot
    axes[1].scatter(comparison_df[dispersion_col], comparison_df[litter_col],
                   alpha=0.6, s=50, color='purple')

    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        comparison_df[dispersion_col], comparison_df[litter_col]
    )

    x_line = np.array([comparison_df[dispersion_col].min(),
                      comparison_df[dispersion_col].max()])
    y_line = slope * x_line + intercept

    axes[1].plot(x_line, y_line, 'r--', linewidth=2,
                label=f'R² = {r_value**2:.3f}, p = {p_value:.4f}')

    axes[1].set_xlabel(dispersion_col.replace('_', ' ').title(), fontsize=12)
    axes[1].set_ylabel(litter_col.replace('_', ' ').title(), fontsize=12)
    axes[1].set_title("Correlation Plot", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dispersion comparison to {output_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance from model.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance with columns: feature, coefficient (or importance).
    top_n : int
        Number of top features to plot.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Sort and select top features
    importance_df = importance_df.sort_values('abs_coef', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['green' if x > 0 else 'red' for x in importance_df['coefficient']]

    ax.barh(range(len(importance_df)), importance_df['coefficient'],
           color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.axvline(0, color='k', linestyle='-', linewidth=1)

    ax.set_xlabel("Coefficient", fontsize=13)
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")

    return fig


def plot_spatial_map(
    ports_df: pd.DataFrame,
    rivers_df: pd.DataFrame,
    mapping: Dict[str, List[Tuple[str, float]]],
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spatial map of ports and rivers with connections.

    Parameters
    ----------
    ports_df : pd.DataFrame
        Port metadata with lat, lon.
    rivers_df : pd.DataFrame
        River metadata with lat, lon.
    mapping : Dict
        Port to rivers mapping.
    output_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        logger.warning("Cartopy not available, using simple matplotlib plot")
        has_cartopy = False

    if has_cartopy:
        projection = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(12, 10),
                              subplot_kw={'projection': projection})

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)

        # Set extent
        all_lats = np.concatenate([ports_df['lat'].values, rivers_df['lat'].values])
        all_lons = np.concatenate([ports_df['lon'].values, rivers_df['lon'].values])
        margin = 0.5
        ax.set_extent([all_lons.min() - margin, all_lons.max() + margin,
                      all_lats.min() - margin, all_lats.max() + margin])

        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        transform = None

    # Plot connections
    for port_id, rivers in mapping.items():
        port = ports_df[ports_df['port_id'] == port_id].iloc[0]

        for river_id, distance in rivers:
            river = rivers_df[rivers_df['river_id'] == river_id]
            if river.empty:
                continue
            river = river.iloc[0]

            # Draw connection line
            if has_cartopy:
                ax.plot([port['lon'], river['lon']],
                       [port['lat'], river['lat']],
                       color='gray', alpha=0.4, linewidth=1,
                       transform=transform, zorder=1)
            else:
                ax.plot([port['lon'], river['lon']],
                       [port['lat'], river['lat']],
                       color='gray', alpha=0.4, linewidth=1, zorder=1)

    # Plot ports
    if has_cartopy:
        ax.scatter(ports_df['lon'], ports_df['lat'],
                  s=200, c='blue', marker='o', edgecolors='black',
                  linewidths=2, label='Ports', transform=transform, zorder=3)
    else:
        ax.scatter(ports_df['lon'], ports_df['lat'],
                  s=200, c='blue', marker='o', edgecolors='black',
                  linewidths=2, label='Ports', zorder=3)

    # Annotate ports
    for _, port in ports_df.iterrows():
        ax.text(port['lon'], port['lat'] + 0.05, port['port_id'],
               fontsize=10, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot rivers
    if has_cartopy:
        ax.scatter(rivers_df['lon'], rivers_df['lat'],
                  s=150, c='red', marker='^', edgecolors='black',
                  linewidths=1.5, label='Rivers', transform=transform, zorder=2)
    else:
        ax.scatter(rivers_df['lon'], rivers_df['lat'],
                  s=150, c='red', marker='^', edgecolors='black',
                  linewidths=1.5, label='Rivers', zorder=2)

    # Annotate rivers
    for _, river in rivers_df.iterrows():
        ax.text(river['lon'], river['lat'] - 0.05, river['river_id'],
               fontsize=9, ha='center', style='italic',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    ax.set_title("Port-River Spatial Mapping", fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best')

    if not has_cartopy:
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spatial map to {output_path}")

    return fig
