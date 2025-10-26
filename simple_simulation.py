#!/usr/bin/env python3
"""
Simple Water Network Simulation Script
Created for easy single-step simulation based on config file

Purpose: Run ONE simulation without batch processing or parallelization
Usage: python simple_simulation.py --config configs/v7.1/ctown_7v1__EPYNET_config.ini
"""

import argparse
import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from epynet import Network
import matplotlib.pyplot as plt


def load_config(config_path):
    """Load configuration file

    Args:
        config_path (str): Path to .ini config file

    Returns:
        ConfigParser: Loaded config object
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ConfigParser()
    config.read(config_path)
    print(f"✓ Loaded config: {config_path}\n")
    return config


def print_config_summary(config):
    """Print a summary of the configuration"""
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    for section in config.sections():
        print(f"\n[{section}]")
        for key, value in config.items(section):
            print(f"  {key:25s} = {value}")
    print("\n" + "=" * 60 + "\n")


def sample_from_range(low, high):
    """Sample a random value from a range

    Args:
        low (float): Lower bound
        high (float): Upper bound

    Returns:
        float: Random value between low and high
    """
    return low + np.random.random() * (high - low)


def run_single_simulation(config, sample_demands=True, sample_reservoir_heads=True,
                         sample_tank_levels=False, sample_pipe_roughness=False):
    """Run a single EPANET simulation with parameters from config

    Args:
        config (ConfigParser): Configuration object
        sample_demands (bool): Whether to randomly sample junction demands
        sample_reservoir_heads (bool): Whether to randomly sample reservoir heads
        sample_tank_levels (bool): Whether to randomly sample tank levels
        sample_pipe_roughness (bool): Whether to randomly sample pipe roughness

    Returns:
        dict: Simulation results containing pressures, heads, flows, etc.
    """
    # Get INP file path from config
    inp_path = config.get('general', 'wn_inp_path')

    if not os.path.isfile(inp_path):
        raise FileNotFoundError(f"INP file not found: {inp_path}")

    print(f"Loading water network: {inp_path}")
    wn = Network(inp_path)

    # Apply random parameters based on config
    print("\nApplying simulation parameters:")
    print("-" * 60)

    # 1. Sample junction demands
    if sample_demands and config.has_section('junction'):
        demand_lo = config.getfloat('junction', 'demand_lo')
        demand_hi = config.getfloat('junction', 'demand_hi')

        print(f"Sampling demands: [{demand_lo:.6f}, {demand_hi:.6f}]")
        for junction in wn.junctions:
            sampled_demand = sample_from_range(demand_lo, demand_hi)
            junction.basedemand = sampled_demand
        print(f"  → Set {len(wn.junctions)} junction demands")

    # 2. Sample reservoir heads
    if sample_reservoir_heads and config.has_section('reservoir'):
        head_lo = config.getfloat('reservoir', 'head_lo')
        head_hi = config.getfloat('reservoir', 'head_hi')

        print(f"Sampling reservoir heads: [{head_lo:.2f}, {head_hi:.2f}] m")
        for reservoir in wn.reservoirs:
            sampled_head = sample_from_range(head_lo, head_hi)
            reservoir.head = sampled_head
            print(f"  → Reservoir '{reservoir.uid}': {sampled_head:.2f} m")

    # 3. Sample tank levels (optional)
    if sample_tank_levels and config.has_section('tank') and len(wn.tanks) > 0:
        level_lo = config.getfloat('tank', 'level_lo')
        level_hi = config.getfloat('tank', 'level_hi')

        print(f"Sampling tank levels: [{level_lo:.2f}, {level_hi:.2f}] m")
        for tank in wn.tanks:
            sampled_level = sample_from_range(level_lo, level_hi)
            tank.initlevel = sampled_level
            print(f"  → Tank '{tank.uid}': {sampled_level:.2f} m")

    # 4. Sample pipe roughness (optional)
    if sample_pipe_roughness and config.has_section('pipe'):
        roughness_lo = config.getfloat('pipe', 'roughness_lo')
        roughness_hi = config.getfloat('pipe', 'roughness_hi')

        print(f"Sampling pipe roughness: [{roughness_lo:.2f}, {roughness_hi:.2f}]")
        for pipe in wn.pipes:
            sampled_roughness = sample_from_range(roughness_lo, roughness_hi)
            pipe.roughness = sampled_roughness
        print(f"  → Set {len(wn.pipes)} pipe roughness values")

    print("-" * 60)

    # Run the simulation
    print("\nRunning EPANET simulation...")
    try:
        wn.solve()
        print("✓ Simulation completed successfully!\n")
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return None

    # Extract results
    results = extract_results(wn)

    return results


def extract_results(wn):
    """Extract simulation results from the water network

    Args:
        wn (Network): EPANET network object after simulation

    Returns:
        dict: Dictionary containing all results
    """
    results = {}

    # Junction results
    results['junctions'] = pd.DataFrame({
        'id': [j.uid for j in wn.junctions],
        'demand': [j.demand for j in wn.junctions],
        'head': [j.head for j in wn.junctions],
        'pressure': [j.pressure for j in wn.junctions],
        'elevation': [j.elevation for j in wn.junctions],
    })

    # Pipe results
    results['pipes'] = pd.DataFrame({
        'id': [p.uid for p in wn.pipes],
        'flow': [p.flow for p in wn.pipes],
        'velocity': [p.velocity for p in wn.pipes],
        'headloss': [p.headloss for p in wn.pipes],
        'diameter': [p.diameter for p in wn.pipes],
        'length': [p.length for p in wn.pipes],
        'roughness': [p.roughness for p in wn.pipes],
    })

    # Reservoir results
    if len(wn.reservoirs) > 0:
        results['reservoirs'] = pd.DataFrame({
            'id': [r.uid for r in wn.reservoirs],
            'head': [r.head for r in wn.reservoirs],
            'demand': [r.demand for r in wn.reservoirs],
        })

    # Tank results
    if len(wn.tanks) > 0:
        results['tanks'] = pd.DataFrame({
            'id': [t.uid for t in wn.tanks],
            'head': [t.head for t in wn.tanks],
            'pressure': [t.pressure for t in wn.tanks],
            'demand': [t.demand for t in wn.tanks],
            'level': [t.level for t in wn.tanks],
        })

    return results


def print_results_summary(results):
    """Print a summary of simulation results

    Args:
        results (dict): Results dictionary
    """
    print("=" * 60)
    print("Simulation Results Summary")
    print("=" * 60)

    # Junction summary
    junc_df = results['junctions']
    print(f"\n📍 Junctions ({len(junc_df)} nodes):")
    print(f"  Pressure range: {junc_df['pressure'].min():.2f} - {junc_df['pressure'].max():.2f} m")
    print(f"  Pressure mean:  {junc_df['pressure'].mean():.2f} m")
    print(f"  Head range:     {junc_df['head'].min():.2f} - {junc_df['head'].max():.2f} m")
    print(f"  Total demand:   {junc_df['demand'].sum():.6f} L/s")

    # Pipe summary
    pipe_df = results['pipes']
    print(f"\n🔧 Pipes ({len(pipe_df)} pipes):")
    print(f"  Flow range:     {pipe_df['flow'].min():.6f} - {pipe_df['flow'].max():.6f} L/s")
    print(f"  Velocity range: {pipe_df['velocity'].min():.3f} - {pipe_df['velocity'].max():.3f} m/s")
    print(f"  Max headloss:   {pipe_df['headloss'].max():.3f} m")

    # Reservoir summary
    if 'reservoirs' in results:
        res_df = results['reservoirs']
        print(f"\n💧 Reservoirs ({len(res_df)} sources):")
        for idx, row in res_df.iterrows():
            print(f"  {row['id']:15s}: Head={row['head']:6.2f} m, Supply={-row['demand']:8.6f} L/s")

    # Tank summary
    if 'tanks' in results:
        tank_df = results['tanks']
        print(f"\n🏗️  Tanks ({len(tank_df)} tanks):")
        for idx, row in tank_df.iterrows():
            print(f"  {row['id']:15s}: Level={row['level']:6.2f} m, Pressure={row['pressure']:6.2f} m")

    print("\n" + "=" * 60 + "\n")


def save_results(results, output_dir='simulation_results'):
    """Save results to CSV files

    Args:
        results (dict): Results dictionary
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save each component to CSV
    for component, df in results.items():
        output_path = os.path.join(output_dir, f"{component}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    print(f"\n✓ All results saved to: {output_dir}/")


def plot_pressure_distribution(results, save_path=None):
    """Plot pressure distribution across junctions

    Args:
        results (dict): Results dictionary
        save_path (str, optional): Path to save the plot
    """
    junc_df = results['junctions']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Water Network Simulation Results', fontsize=16, fontweight='bold')

    # 1. Pressure histogram
    ax = axes[0, 0]
    ax.hist(junc_df['pressure'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pressure (m)', fontsize=11)
    ax.set_ylabel('Number of Junctions', fontsize=11)
    ax.set_title('Pressure Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Head histogram
    ax = axes[0, 1]
    ax.hist(junc_df['head'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Head (m)', fontsize=11)
    ax.set_ylabel('Number of Junctions', fontsize=11)
    ax.set_title('Head Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Demand histogram
    ax = axes[1, 0]
    ax.hist(junc_df['demand'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Demand (L/s)', fontsize=11)
    ax.set_ylabel('Number of Junctions', fontsize=11)
    ax.set_title('Demand Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Pressure vs Elevation scatter
    ax = axes[1, 1]
    scatter = ax.scatter(junc_df['elevation'], junc_df['pressure'],
                        c=junc_df['pressure'], cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Elevation (m)', fontsize=11)
    ax.set_ylabel('Pressure (m)', fontsize=11)
    ax.set_title('Pressure vs Elevation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Pressure (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Water Network Simulation')
    parser.add_argument('--config', type=str, default='configs/v7.1/ctown_7v1__EPYNET_config.ini',
                       help='Path to configuration file (.ini)')
    parser.add_argument('--sample_demands', action='store_true', default=True,
                       help='Randomly sample junction demands from config range')
    parser.add_argument('--sample_reservoir_heads', action='store_true', default=True,
                       help='Randomly sample reservoir heads from config range')
    parser.add_argument('--sample_tank_levels', action='store_true', default=False,
                       help='Randomly sample tank levels from config range')
    parser.add_argument('--sample_pipe_roughness', action='store_true', default=False,
                       help='Randomly sample pipe roughness from config range')
    parser.add_argument('--show_config', action='store_true', default=False,
                       help='Show full configuration before simulation')
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save results to CSV files')
    parser.add_argument('--plot', action='store_true', default=False,
                       help='Generate and display plots')
    parser.add_argument('--output_dir', type=str, default='simulation_results',
                       help='Directory to save results and plots')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Show config if requested
    if args.show_config:
        print_config_summary(config)

    # Run simulation
    results = run_single_simulation(
        config,
        sample_demands=args.sample_demands,
        sample_reservoir_heads=args.sample_reservoir_heads,
        sample_tank_levels=args.sample_tank_levels,
        sample_pipe_roughness=args.sample_pipe_roughness
    )

    if results is None:
        print("Simulation failed. Exiting.")
        return

    # Print results summary
    print_results_summary(results)

    # Save results if requested
    if args.save_results:
        save_results(results, args.output_dir)

    # Plot if requested
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'pressure_distribution.png') if args.save_results else None
        plot_pressure_distribution(results, save_path=plot_path)


if __name__ == '__main__':
    main()
