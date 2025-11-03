#!/usr/bin/env python3
"""
Script to read and inspect generated simulation data from zarr files.
This script helps you check demand values, pressure values, and other simulation results.

Usage:
    python inspect_simulation_data.py --data_path datasets/ctown.zip
    python inspect_simulation_data.py --data_path datasets/ctown.zip --feature demand
    python inspect_simulation_data.py --data_path datasets/ctown.zip --subset train --show_samples 5
"""

import argparse
import zarr
import numpy as np
import os


def inspect_zarr_structure(data_path):
    """Display the structure of the zarr file"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {data_path}")
    print(f"{'='*80}\n")

    # Open the zarr file
    if data_path.endswith('.zip'):
        root = zarr.open(store=data_path, mode='r')
    else:
        root = zarr.open(store=zarr.DirectoryStore(data_path), mode='r')

    # Print the tree structure
    print("Dataset Structure:")
    print(root.tree())
    print()

    # Print available features
    print("Available Features:")
    for key in root.group_keys():
        if key != 'tmp':
            print(f"  - {key}")
    print()

    # Print metadata
    if 'config' in root.attrs:
        print("Configuration:")
        config = root.attrs['config']
        for section, values in config.items():
            print(f"  [{section}]")
            for k, v in values.items():
                print(f"    {k} = {v}")
    print()

    if 'ordered_names_by_attr' in root.attrs:
        print("Node/Link Names by Attribute:")
        ordered_names = root.attrs['ordered_names_by_attr']
        for attr, names in ordered_names.items():
            print(f"  {attr}: {len(names)} items")
            if len(names) <= 10:
                print(f"    {names}")
            else:
                print(f"    First 5: {names[:5]}")
                print(f"    Last 5: {names[-5:]}")
    print()

    return root


def inspect_feature_data(root, feature='pressure', subset='train'):
    """Inspect specific feature data (pressure, demand, flow, etc.)"""

    if feature not in root.group_keys():
        print(f"ERROR: Feature '{feature}' not found in dataset!")
        print(f"Available features: {list(root.group_keys())}")
        return

    feature_group = root[feature]

    print(f"\n{'='*80}")
    print(f"Feature: {feature.upper()}")
    print(f"{'='*80}\n")

    # Check available subsets
    available_subsets = list(feature_group.group_keys())
    print(f"Available subsets: {available_subsets}\n")

    if subset not in available_subsets:
        print(f"ERROR: Subset '{subset}' not found!")
        print(f"Using first available subset: {available_subsets[0]}")
        subset = available_subsets[0]

    # Get the data array
    data_array = feature_group[subset]

    print(f"Subset: {subset}")
    print(f"Shape: {data_array.shape} (num_scenarios, num_nodes/links)")
    print(f"Data type: {data_array.dtype}")
    print(f"Chunks: {data_array.chunks}")
    print()

    # Print statistics from attributes
    if 'mean' in feature_group.attrs:
        print("Training Statistics (from attributes):")
        print(f"  Mean:  {feature_group.attrs['mean']:.6f}")
        print(f"  Std:   {feature_group.attrs['std']:.6f}")
        print(f"  Min:   {feature_group.attrs['min']:.6f}")
        print(f"  Max:   {feature_group.attrs['max']:.6f}")
        if 'cv' in feature_group.attrs:
            print(f"  CV:    {feature_group.attrs['cv']:.6f}")
        print()

    # Compute statistics for current subset
    print(f"Actual {subset.capitalize()} Statistics:")
    data = data_array[:]
    print(f"  Mean:  {np.mean(data):.6f}")
    print(f"  Std:   {np.std(data):.6f}")
    print(f"  Min:   {np.min(data):.6f}")
    print(f"  Max:   {np.max(data):.6f}")
    print()

    # Check for negative values (important for pressure)
    if feature == 'pressure':
        num_negative = np.sum(data < 0)
        if num_negative > 0:
            print(f"⚠️  WARNING: Found {num_negative} negative pressure values!")
            print(f"   This represents {100*num_negative/data.size:.2f}% of all values")
            print(f"   Minimum pressure: {np.min(data):.2f}")

            # Find scenarios with negative pressure
            scenarios_with_neg = np.any(data < 0, axis=1)
            num_scenarios_with_neg = np.sum(scenarios_with_neg)
            print(f"   Scenarios with negative pressure: {num_scenarios_with_neg}/{data.shape[0]}")
            print()
        else:
            print(f"✓ No negative pressure values found")
            print()

    # Show sample data
    print(f"Sample Data (first 3 scenarios):")
    for i in range(min(3, data.shape[0])):
        print(f"  Scenario {i}: min={np.min(data[i]):.2f}, max={np.max(data[i]):.2f}, mean={np.mean(data[i]):.2f}")
    print()

    return data


def compare_demand_ranges(root, config_path=None):
    """Compare demand values with configuration ranges"""

    if 'demand' not in root.group_keys():
        print("Demand data not found in dataset")
        return

    demand_group = root['demand']

    # Get train data
    if 'train' in demand_group.group_keys():
        demand_data = demand_group['train'][:]
    else:
        print("Train data not found, cannot analyze demands")
        return

    print(f"\n{'='*80}")
    print("DEMAND VALUE ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Demand Statistics:")
    print(f"  Shape: {demand_data.shape}")
    print(f"  Min:   {np.min(demand_data):.8f}")
    print(f"  Max:   {np.max(demand_data):.8f}")
    print(f"  Mean:  {np.mean(demand_data):.8f}")
    print(f"  Std:   {np.std(demand_data):.8f}")
    print()

    # Get config if available
    if 'config' in root.attrs:
        config = root.attrs['config']
        if 'junction' in config:
            demand_lo = float(config['junction'].get('demand_lo', 'N/A'))
            demand_hi = float(config['junction'].get('demand_hi', 'N/A'))
            print(f"Configuration Ranges:")
            print(f"  demand_lo: {demand_lo}")
            print(f"  demand_hi: {demand_hi}")
            print()

            # Check if demands are within expected range
            if demand_lo != 'N/A' and demand_hi != 'N/A':
                within_range = np.all((demand_data >= demand_lo) & (demand_data <= demand_hi))
                if within_range:
                    print(f"✓ All demands are within configured range [{demand_lo}, {demand_hi}]")
                else:
                    print(f"⚠️  Some demands are outside configured range!")
                    print(f"   Expected: [{demand_lo}, {demand_hi}]")
                    print(f"   Actual: [{np.min(demand_data)}, {np.max(demand_data)}]")
    print()


def main():
    parser = argparse.ArgumentParser(description='Inspect simulation data from zarr files')
    parser.add_argument('--data_path', type=str, default='datasets/ctown.zip',
                        help='Path to zarr file (.zip or directory)')
    parser.add_argument('--feature', type=str, default=None,
                        help='Feature to inspect (pressure, demand, flow, velocity, head)')
    parser.add_argument('--subset', type=str, default='train',
                        help='Data subset (train, valid, test)')
    parser.add_argument('--show_structure', action='store_true',
                        help='Show full dataset structure')
    parser.add_argument('--analyze_demands', action='store_true',
                        help='Analyze demand values')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.data_path):
        print(f"ERROR: File not found: {args.data_path}")
        print(f"\nPlease generate data first by running:")
        print(f"  python gnn_pressure_estimation/scenegenv7.py --config <config_file>")
        return

    # Open and inspect structure
    root = inspect_zarr_structure(args.data_path)

    # Inspect specific feature if requested
    if args.feature:
        inspect_feature_data(root, feature=args.feature, subset=args.subset)
    else:
        # Inspect all features
        for feature in root.group_keys():
            if feature != 'tmp':
                inspect_feature_data(root, feature=feature, subset=args.subset)

    # Analyze demands if requested
    if args.analyze_demands or args.feature == 'demand':
        compare_demand_ranges(root)

    print(f"\n{'='*80}")
    print("Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
