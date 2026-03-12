#!/usr/bin/env python3
"""
Test Multiple Stations Per Port

Systematically tests different precipitation stations for each port
to identify optimal hydrological connections and assess sensitivity.

Usage:
    python test_multiple_stations.py --config config_base.yml --stations stations_to_test.yml

Output:
    - Individual results per station in outputs_STATION/
    - Summary comparison table
    - Visualization of station effects per port
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import subprocess
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import ensure_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_stations_to_test(config_path: str) -> dict:
    """
    Load station testing configuration.

    Expected format (stations_to_test.yml):

    stations_to_test:
      63100:  # Port ID
        - US  # Station 1 to test
        - UU  # Station 2 to test
        - UW  # Station 3 to test
      63210:
        - MR
        - UA
      64200:  # Barcelona - test multiple
        - XL
        - X4
        - YQ
        - WT
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('stations_to_test', {})


def update_config_for_station(base_config: dict, port_id: int, station: str,
                              output_dir: str) -> dict:
    """
    Create modified config for specific port-station combination.

    Args:
        base_config: Base configuration dict
        port_id: Port to test
        station: Station to use for this port
        output_dir: Output directory for this test

    Returns:
        Modified config dict
    """
    # Deep copy to avoid modifying original
    import copy
    config = copy.deepcopy(base_config)

    # Update manual overrides to use ONLY this station for ONLY this port
    if 'mapping' not in config:
        config['mapping'] = {}

    # CRITICAL FIX: Replace manual_overrides with ONLY the test port
    # (not "keep others as is" - that causes all ports to be analyzed!)
    config['mapping']['manual_overrides'] = {
        port_id: [station]
    }

    logger.debug(f"Config manual_overrides set to: {config['mapping']['manual_overrides']}")

    # Update output directory
    config['outputs']['tables'] = f"{output_dir}/tables"
    config['outputs']['figures'] = f"{output_dir}/figures"
    config['outputs']['models'] = f"{output_dir}/models"
    if 'logs' in config['outputs']:
        config['outputs']['logs'] = f"{output_dir}/logs"

    return config


def run_analysis_for_station(config_path: str, port_id: int, station: str,
                             output_base: str = "outputs_tests",
                             main_script: str = "main.py") -> dict:
    """
    Run analysis for one port-station combination.

    Args:
        config_path: Path to base config file
        port_id: Port ID to test
        station: Station ID to use
        output_base: Base directory for outputs
        main_script: Path to main.py script

    Returns:
        dict with results summary
    """
    # Create output directory for this test
    output_dir = Path(output_base) / f"port_{port_id}_station_{station}"
    ensure_dir(output_dir)

    logger.info(f"=" * 70)
    logger.info(f"Testing: Port {port_id} with Station {station}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"=" * 70)

    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Update config for this test
    test_config = update_config_for_station(base_config, port_id, station, str(output_dir))

    # Log the key parts of the config for debugging
    logger.info(f"Config for this test:")
    logger.info(f"  manual_overrides: {test_config.get('mapping', {}).get('manual_overrides', {})}")
    logger.info(f"  output tables: {test_config.get('outputs', {}).get('tables', 'NOT SET')}")

    # Save temporary config
    temp_config_path = output_dir / "config_temp.yml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(test_config, f)

    logger.info(f"Saved temp config to: {temp_config_path}")

    # For first test, show the full config for debugging
    if not hasattr(run_analysis_for_station, '_shown_sample_config'):
        logger.info("=" * 60)
        logger.info("SAMPLE TEMP CONFIG (first test only):")
        logger.info("=" * 60)
        with open(temp_config_path, 'r') as f:
            logger.info(f.read())
        logger.info("=" * 60)
        run_analysis_for_station._shown_sample_config = True

    # Run analysis
    try:
        cmd = [
            sys.executable,  # Use same Python interpreter
            main_script,  # Path to main.py
            "--config", str(temp_config_path),
            "run_all"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        logger.info(f"Command completed with return code: {result.returncode}")

        # Always log stdout/stderr for debugging
        if result.stdout:
            logger.debug("STDOUT:")
            logger.debug(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        if result.stderr:
            logger.debug("STDERR:")
            logger.debug(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

        if result.returncode != 0:
            logger.error(f"Analysis failed for port {port_id}, station {station}")
            logger.error(result.stderr)
            return {
                'port_id': port_id,
                'station': station,
                'success': False,
                'error': result.stderr[:200]
            }

        # Check what files were actually created
        logger.info("Checking output directory...")
        tables_dir = output_dir / "tables"
        if tables_dir.exists():
            created_files = list(tables_dir.glob("*.csv"))
            logger.info(f"  Found {len(created_files)} CSV files in tables/:")
            for f in created_files:
                logger.info(f"    - {f.name} ({f.stat().st_size} bytes)")
        else:
            logger.warning(f"  Tables directory doesn't exist: {tables_dir}")

        # Load results
        results_path = output_dir / "tables" / "haul_model_results.csv"
        logger.info(f"Looking for results at: {results_path}")

        if results_path.exists():
            results_df = pd.read_csv(results_path)

            logger.debug(f"Loaded results from {results_path}")
            logger.debug(f"  Shape: {results_df.shape}")
            logger.debug(
                f"  Ports in file: {results_df['port_id'].unique() if 'port_id' in results_df.columns else 'NO PORT_ID'}")
            logger.debug(f"  Looking for port: {port_id}")

            # Extract results for this port
            port_results = results_df[results_df['port_id'] == port_id]

            logger.debug(f"  Found {len(port_results)} rows for port {port_id}")

            if len(port_results) > 0:
                row = port_results.iloc[0]

                # Find effect column flexibly
                effect_col = find_effect_column(results_df)
                if effect_col is None:
                    logger.warning(f"No effect column found in results for {port_id}, {station}")
                    effect_val = np.nan
                else:
                    effect_val = row[effect_col]

                # Try to get beta (might not exist in newer versions)
                beta_val = row.get('beta', effect_val if effect_col else np.nan)

                return {
                    'port_id': port_id,
                    'station': station,
                    'success': True,
                    'optimal_window_weeks': row['optimal_window_weeks'],
                    effect_col if effect_col else 'effect': effect_val,  # Use actual column name
                    'pvalue': row['pvalue'],
                    'significance': row['significance'],
                    'aic': row.get('aic', np.nan),
                    'n_obs': row.get('n_obs', np.nan),
                    'output_dir': str(output_dir)
                }
            else:
                # Port not found - this shouldn't happen with fixed config
                ports_found = results_df['port_id'].unique() if 'port_id' in results_df.columns else []
                error_msg = f"Port {port_id} not in results. Found ports: {ports_found}"
                logger.error(error_msg)
                logger.error(f"This suggests config wasn't properly filtered to test port only")
                return {
                    'port_id': port_id,
                    'station': station,
                    'success': False,
                    'error': error_msg
                }
        else:
            logger.error(f"Results file not found: {results_path}")
            logger.error("This means main script ran but didn't create expected output")
            logger.error("Possible causes:")
            logger.error("  1. No hauls for this port/station combination")
            logger.error("  2. Main script uses different output filename")
            logger.error("  3. Analysis was skipped for this port")

            # Check if ANY csv files exist
            if tables_dir.exists():
                all_files = list(tables_dir.glob("*"))
                logger.error(f"Files that DO exist in {tables_dir}:")
                for f in all_files:
                    logger.error(f"  - {f.name}")

            return {
                'port_id': port_id,
                'station': station,
                'success': False,
                'error': "Results file not created - check logs for details"
            }

    except subprocess.TimeoutExpired:
        logger.error(f"Analysis timed out for port {port_id}, station {station}")
        return {
            'port_id': port_id,
            'station': station,
            'success': False,
            'error': "Timeout (>5 minutes)"
        }
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        return {
            'port_id': port_id,
            'station': station,
            'success': False,
            'error': str(e)[:200]
        }


def create_summary_table(all_results: list) -> pd.DataFrame:
    """
    Create summary table comparing all tests.

    Args:
        all_results: List of result dicts from all tests

    Returns:
        DataFrame with comparison
    """
    if not all_results:
        logger.warning("No results to summarize")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    logger.info(f"Creating summary from {len(df)} results")
    logger.debug(f"Available columns: {df.columns.tolist()}")

    # Find the effect/beta column (name might vary depending on main script version)
    effect_col = None
    for possible_name in ['beta', 'effect', 'effect_10mm_pct', 'effect_pct', 'effect_100m3s_pct']:
        if possible_name in df.columns:
            effect_col = possible_name
            logger.debug(f"Using '{effect_col}' as effect size column")
            break

    # Sort by port, then by effect (to see range per port)
    if effect_col and 'port_id' in df.columns:
        df = df.sort_values(['port_id', effect_col], ascending=[True, False])
    elif 'port_id' in df.columns:
        logger.warning(f"No effect column found in {df.columns.tolist()}, sorting by port_id only")
        df = df.sort_values(['port_id'])
    else:
        logger.warning("Could not find port_id column, returning unsorted")

    return df


def find_effect_column(df: pd.DataFrame) -> str:
    """
    Find the effect/beta column in dataframe (handles different naming schemes).

    Args:
        df: DataFrame to search

    Returns:
        Column name, or None if not found
    """
    for possible_name in ['beta', 'effect', 'effect_10mm_pct', 'effect_pct', 'effect_100m3s_pct']:
        if possible_name in df.columns:
            return possible_name
    return None


def create_comparison_report(summary_df: pd.DataFrame, output_path: str):
    """
    Create detailed comparison report.

    Args:
        summary_df: Summary DataFrame
        output_path: Where to save report
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("STATION SENSITIVITY ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Find effect column name (flexible)
    effect_col = find_effect_column(summary_df)
    if effect_col is None:
        logger.error("Cannot create report: no effect/beta column found")
        logger.error(f"Available columns: {summary_df.columns.tolist()}")
        return

    logger.debug(f"Using '{effect_col}' as effect column in report")

    # Overall summary
    successful = summary_df[summary_df['success']].shape[0]
    total = summary_df.shape[0]

    report_lines.append(f"Tests completed: {successful}/{total}")
    report_lines.append(f"Ports analyzed: {summary_df['port_id'].nunique()}")
    report_lines.append("")

    # Per-port analysis
    for port_id in sorted(summary_df['port_id'].unique()):
        port_data = summary_df[summary_df['port_id'] == port_id]
        port_success = port_data[port_data['success']]

        report_lines.append("=" * 80)
        report_lines.append(f"PORT {port_id}")
        report_lines.append("=" * 80)
        report_lines.append(f"Stations tested: {len(port_data)}")
        report_lines.append(f"Successful: {len(port_success)}")
        report_lines.append("")

        if len(port_success) == 0:
            report_lines.append("  ⚠️  No successful results for this port")
            report_lines.append("")
            continue

        # Summary statistics
        effects = port_success[effect_col].values

        report_lines.append(f"Effect range: {effects.min():.1f}% to {effects.max():.1f}%")
        report_lines.append(f"Direction: {(effects > 0).sum()} positive, {(effects < 0).sum()} negative")
        report_lines.append("")

        # Individual results
        report_lines.append("Results by station:")
        report_lines.append("")

        for _, row in port_success.iterrows():
            sig = row['significance'] if pd.notna(row.get('significance')) else 'NS'
            report_lines.append(
                f"  Station {row['station']:6s}: "
                f"effect = {row[effect_col]:7.1f}%, "
                f"p = {row['pvalue']:.4f} {sig:3s}, "
                f"window = {row['optimal_window_weeks']:2.0f}wk"
            )

        report_lines.append("")

        # Interpretation
        if len(port_success) > 1:
            # Check for agreement
            all_positive = (effects > 0).all()
            all_negative = (effects < 0).all()
            mixed = not (all_positive or all_negative)

            if all_positive:
                report_lines.append("  ✓ CONSISTENT: All stations show positive effects")
                report_lines.append("    → Robust litter delivery signal")
            elif all_negative:
                report_lines.append("  ✓ CONSISTENT: All stations show negative effects")
                report_lines.append("    → Robust washout signal")
            elif mixed:
                report_lines.append("  ⚠️  INCONSISTENT: Stations show mixed effects")
                report_lines.append("    → Hydrological connectivity matters!")
                report_lines.append("    → Choose station with mechanistic connection")

                # Suggest best station
                # Priority: (1) Positive effects (delivery), (2) Significant, (3) Largest effect
                positive_results = port_success[port_success[effect_col] > 0]
                negative_results = port_success[port_success[effect_col] < 0]

                if len(positive_results) > 0:
                    # Prefer positive effects (litter delivery)
                    sig_positive = positive_results[positive_results['pvalue'] < 0.05]

                    if len(sig_positive) > 0:
                        # Best: positive AND significant
                        best = sig_positive.loc[sig_positive[effect_col].idxmax()]
                        report_lines.append(f"    → Suggested: Station {best['station']} (positive + significant)")
                    else:
                        # Use largest positive even if not significant
                        best = positive_results.loc[positive_results[effect_col].idxmax()]
                        if best['pvalue'] < 0.10:
                            report_lines.append(
                                f"    → Suggested: Station {best['station']} (positive + borderline p={best['pvalue']:.3f})")
                        else:
                            report_lines.append(f"    → Suggested: Station {best['station']} (only positive effect)")
                else:
                    # All negative - warn user!
                    report_lines.append("    ⚠️  WARNING: All stations show negative effects (washout)")
                    report_lines.append("    → No station shows litter delivery")
                    report_lines.append(
                        "    → Consider: (1) Different station types, (2) Port may not receive precipitation-driven litter")
                    # Still suggest least negative
                    best = negative_results.loc[negative_results[effect_col].idxmax()]
                    report_lines.append(
                        f"    → Least negative: Station {best['station']} (effect={best[effect_col]:.1f}%)")

            # Check window consistency
            windows = port_success['optimal_window_weeks'].values
            if windows.std() > 4:
                report_lines.append(f"  ⚠️  Window variability: {windows.min():.0f}-{windows.max():.0f} weeks")
                report_lines.append("    → Different transport timescales")

        report_lines.append("")

    # Overall recommendations
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")

    for port_id in sorted(summary_df['port_id'].unique()):
        port_success = summary_df[(summary_df['port_id'] == port_id) & (summary_df['success'])]

        if len(port_success) == 0:
            continue

        # Find best station by criteria
        # Priority: (1) Positive effects (litter delivery), (2) Significant, (3) Largest effect
        positive_results = port_success[port_success[effect_col] > 0]
        negative_results = port_success[port_success[effect_col] < 0]

        if len(positive_results) > 0:
            # Prefer positive effects (litter delivery)
            sig_positive = positive_results[positive_results['pvalue'] < 0.05]

            if len(sig_positive) > 0:
                # Best: positive AND significant
                best = sig_positive.loc[sig_positive[effect_col].idxmax()]
                report_lines.append(
                    f"Port {port_id}: Use station {best['station']} "
                    f"(effect = {best[effect_col]:+.1f}%, p = {best['pvalue']:.4f}) "
                    f"[positive + significant]"
                )
            else:
                # Use largest positive even if not significant
                best = positive_results.loc[positive_results[effect_col].idxmax()]
                if best['pvalue'] < 0.10:
                    report_lines.append(
                        f"Port {port_id}: Use station {best['station']} "
                        f"(effect = {best[effect_col]:+.1f}%, p = {best['pvalue']:.4f}) "
                        f"[positive + borderline]"
                    )
                else:
                    report_lines.append(
                        f"Port {port_id}: Use station {best['station']} "
                        f"(effect = {best[effect_col]:+.1f}%, p = {best['pvalue']:.4f}) "
                        f"[only positive]"
                    )
        elif len(negative_results) > 0:
            # All negative - warn and suggest least negative
            best = negative_results.loc[negative_results[effect_col].idxmax()]
            report_lines.append(
                f"Port {port_id}: ⚠️  No positive effects found! "
                f"Least negative: {best['station']} "
                f"(effect = {best[effect_col]:+.1f}%)"
            )
            report_lines.append(
                f"             → Consider testing different station types or sources"
            )
        else:
            # No results at all
            report_lines.append(f"Port {port_id}: No successful results")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report saved: {output_path}")


def create_visualization(summary_df: pd.DataFrame, output_path: str):
    """
    Create visualization comparing stations per port.

    Args:
        summary_df: Summary DataFrame
        output_path: Where to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        logger.info(f"Creating visualization: {output_path}")

        # Filter to successful results only
        df = summary_df[summary_df['success']].copy()

        if len(df) == 0:
            logger.warning("No successful results to plot")
            return

        # Find effect column name (flexible)
        effect_col = find_effect_column(df)
        if effect_col is None:
            logger.error("Cannot create visualization: no effect column found")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return

        logger.debug(f"Using '{effect_col}' for visualization")

        # Create figure - WIDER TO FIT LABELS
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # ← Changed from (15, 5)

        # Get sorted ports
        ports = sorted(df['port_id'].unique())
        n_ports = len(ports)

        # Create port to x-position mapping
        port_to_x = {port: i for i, port in enumerate(ports)}

        # Panel A: Effect by station for each port
        ax = axes[0]

        for port in ports:
            port_data = df[df['port_id'] == port]
            x_pos = port_to_x[port]
            x = [x_pos] * len(port_data)
            y = port_data[effect_col].values
            colors = ['blue' if val > 0 else 'red' for val in y]
            markers = ['o' if p < 0.05 else 'x' for p in port_data['pvalue'].values]

            for xi, yi, c, m in zip(x, y, colors, markers):
                ax.scatter(xi, yi, c=c, marker=m, s=100, alpha=0.7)

        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xticks(range(n_ports))
        ax.set_xticklabels(ports, rotation=45, ha='right', fontsize=10)  # ← Added rotation and ha
        ax.set_xlabel('Port ID', fontsize=11)
        ax.set_ylabel('Effect (% per 10mm)', fontsize=11)
        ax.set_title('Station Variability by Port', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Panel B: Effect coefficient distribution
        # *** THIS WAS THE BUG - Fixed by using numerical positions ***
        ax = axes[1]

        for port in ports:
            port_data = df[df['port_id'] == port]
            x_pos = port_to_x[port]  # ← FIX: Use numerical position
            effects_vals = port_data[effect_col].values
            x = [x_pos] * len(effects_vals)  # ← FIX: Repeat position for each point
            ax.scatter(x, effects_vals, s=100, alpha=0.6)

        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xticks(range(n_ports))
        ax.set_xticklabels(ports, rotation=45, ha='right', fontsize=10)  # ← Fixed labels
        ax.set_xlabel('Port ID', fontsize=11)
        ax.set_ylabel('Effect (%)', fontsize=11)
        ax.set_title('Effect Range by Port', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Panel C: Window distribution
        # *** THIS WAS ALSO BUGGY - Fixed similarly ***
        ax = axes[2]

        for port in ports:
            port_data = df[df['port_id'] == port]
            x_pos = port_to_x[port]  # ← FIX: Use numerical position
            windows = port_data['optimal_window_weeks'].values
            x = [x_pos] * len(windows)  # ← FIX: Repeat position for each point
            ax.scatter(x, windows, s=100, alpha=0.6)

        ax.set_xticks(range(n_ports))
        ax.set_xticklabels(ports, rotation=45, ha='right', fontsize=10)  # ← Fixed labels
        ax.set_xlabel('Port ID', fontsize=11)
        ax.set_ylabel('Optimal Window (weeks)', fontsize=11)
        ax.set_title('Temporal Window Variability', fontsize=12)
        ax.set_ylim(0, 22)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()  # ← Important!
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved: {output_path}")
        plt.close()

    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test multiple precipitation stations per port'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Base configuration file (e.g., config.yml)'
    )

    parser.add_argument(
        '--stations',
        type=str,
        required=True,
        help='Stations to test configuration (YAML file with stations_to_test)'
    )

    parser.add_argument(
        '--main-script',
        type=str,
        default='main.py',
        help='Path to main.py script (default: main.py in current directory)'
    )

    parser.add_argument(
        '--output-base',
        type=str,
        default='outputs_station_tests',
        help='Base directory for test outputs (default: outputs_station_tests)'
    )

    parser.add_argument(
        '--ports',
        type=int,
        nargs='+',
        help='Specific ports to test (default: all ports in stations config)'
    )

    args = parser.parse_args()

    # Validate main script exists
    main_script_path = Path(args.main_script)
    if not main_script_path.exists():
        logger.error(f"Main script not found: {args.main_script}")
        logger.error(f"Please provide correct path using --main-script argument")
        logger.error(f"Example: --main-script ~/programs/python/MarineLitter/main.py")
        return

    logger.info(f"Using main script: {main_script_path.absolute()}")

    # Load configurations
    logger.info(f"Loading base config: {args.config}")
    logger.info(f"Loading stations config: {args.stations}")

    stations_to_test = load_stations_to_test(args.stations)

    if not stations_to_test:
        logger.error("No stations to test found in config!")
        return

    # Filter to specific ports if requested
    if args.ports:
        stations_to_test = {
            p: s for p, s in stations_to_test.items()
            if p in args.ports
        }

    logger.info(f"Testing {len(stations_to_test)} ports")
    for port, stations in stations_to_test.items():
        logger.info(f"  Port {port}: {len(stations)} stations")

    # Create output directory
    output_base = Path(args.output_base)
    ensure_dir(output_base)

    # Run tests
    all_results = []

    total_tests = sum(len(stations) for stations in stations_to_test.values())
    current_test = 0

    for port_id, stations in stations_to_test.items():
        for station in stations:
            current_test += 1
            logger.info(f"\n[Test {current_test}/{total_tests}]")

            result = run_analysis_for_station(
                args.config,
                port_id,
                station,
                args.output_base,
                args.main_script
            )

            all_results.append(result)

    # Create summary
    logger.info("\n" + "=" * 70)
    logger.info("Creating summary analysis...")
    logger.info("=" * 70)

    summary_df = create_summary_table(all_results)

    # Save summary table
    summary_path = output_base / "station_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary table saved: {summary_path}")

    # Create detailed report
    report_path = output_base / "station_comparison_report.txt"
    create_comparison_report(summary_df, report_path)

    # Create visualization
    figure_path = output_base / "station_comparison_figure.png"
    create_visualization(summary_df, figure_path)

    # Print quick summary
    logger.info("\n" + "=" * 70)
    logger.info("QUICK SUMMARY")
    logger.info("=" * 70)

    successful = summary_df[summary_df['success']]

    for port_id in sorted(stations_to_test.keys()):
        port_results = successful[successful['port_id'] == port_id]

        if len(port_results) == 0:
            logger.info(f"\nPort {port_id}: No successful results")
            continue

        # Find effect column flexibly
        effect_col = find_effect_column(port_results)
        if effect_col:
            effects = port_results[effect_col].values
        else:
            effects = None

        logger.info(f"\nPort {port_id}:")
        logger.info(f"  Stations tested: {len(port_results)}")

        if effects is not None:
            logger.info(f"  Effect range: {effects.min():+.1f}% to {effects.max():+.1f}%")
            logger.info(f"  Direction: {(effects > 0).sum()} positive, {(effects < 0).sum()} negative")

            if (effects > 0).all():
                logger.info(f"  ✓ Consistent: All positive")
            elif (effects < 0).all():
                logger.info(f"  ✓ Consistent: All negative")
            else:
                logger.info(f"  ⚠️  Mixed effects - check report for details")
        else:
            logger.info(f"  (No effect data available)")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_base}")
    logger.info(f"Summary table: {summary_path}")
    logger.info(f"Detailed report: {report_path}")
    logger.info(f"Visualization: {figure_path}")


if __name__ == '__main__':
    main()
