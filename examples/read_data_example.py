"""
Simple examples of how to read and work with generated simulation data.

This file shows various ways to load and inspect the zarr-based simulation results.
"""

import zarr
import numpy as np


# Example 1: Open a zarr file and explore structure
def example_open_and_explore():
    """Basic example: open a zarr file and print structure"""

    # Open the zarr file (can be .zip or directory)
    data_path = "datasets/ctown.zip"
    root = zarr.open(store=data_path, mode='r')

    # Print the hierarchical structure
    print("Dataset structure:")
    print(root.tree())

    # List available features
    print("\nAvailable features:")
    for key in root.group_keys():
        print(f"  - {key}")

    return root


# Example 2: Read pressure data
def example_read_pressure(data_path="datasets/ctown.zip"):
    """Read and analyze pressure data"""

    root = zarr.open(store=data_path, mode='r')

    # Access pressure data for training set
    pressure_train = root['pressure']['train']

    print(f"Pressure data shape: {pressure_train.shape}")
    print(f"  - Number of scenarios: {pressure_train.shape[0]}")
    print(f"  - Number of nodes: {pressure_train.shape[1]}")

    # Load all data into memory (be careful with large datasets)
    pressure_array = pressure_train[:]

    print(f"\nPressure statistics:")
    print(f"  Min: {np.min(pressure_array):.2f}")
    print(f"  Max: {np.max(pressure_array):.2f}")
    print(f"  Mean: {np.mean(pressure_array):.2f}")
    print(f"  Std: {np.std(pressure_array):.2f}")

    # Check for negative pressure
    if np.any(pressure_array < 0):
        print(f"\n⚠️  WARNING: Found negative pressure values!")
        print(f"  Minimum pressure: {np.min(pressure_array):.2f}")
        num_negative = np.sum(pressure_array < 0)
        print(f"  Count: {num_negative}/{pressure_array.size} values")

    return pressure_array


# Example 3: Read demand data
def example_read_demand(data_path="datasets/ctown.zip"):
    """Read and analyze demand data"""

    root = zarr.open(store=data_path, mode='r')

    # Check if demand data exists
    if 'demand' not in root.group_keys():
        print("Demand data not found in dataset")
        return None

    # Access demand data for training set
    demand_train = root['demand']['train']

    print(f"Demand data shape: {demand_train.shape}")
    print(f"  - Number of scenarios: {demand_train.shape[0]}")
    print(f"  - Number of junctions: {demand_train.shape[1]}")

    # Load data
    demand_array = demand_train[:]

    print(f"\nDemand statistics:")
    print(f"  Min: {np.min(demand_array):.6f}")
    print(f"  Max: {np.max(demand_array):.6f}")
    print(f"  Mean: {np.mean(demand_array):.6f}")
    print(f"  Std: {np.std(demand_array):.6f}")

    # Show first scenario
    print(f"\nFirst scenario demand values:")
    print(f"  {demand_array[0][:10]}...")  # First 10 junctions

    return demand_array


# Example 4: Read specific scenarios
def example_read_specific_scenarios(data_path="datasets/ctown.zip", scenario_ids=[0, 1, 2]):
    """Read specific scenarios instead of all data"""

    root = zarr.open(store=data_path, mode='r')

    # Read only specific scenarios (memory efficient)
    pressure_train = root['pressure']['train']

    for scenario_id in scenario_ids:
        scenario_pressure = pressure_train[scenario_id]
        print(f"\nScenario {scenario_id}:")
        print(f"  Shape: {scenario_pressure.shape}")
        print(f"  Min pressure: {np.min(scenario_pressure):.2f}")
        print(f"  Max pressure: {np.max(scenario_pressure):.2f}")
        print(f"  Mean pressure: {np.mean(scenario_pressure):.2f}")


# Example 5: Get node/link names
def example_get_node_names(data_path="datasets/ctown.zip"):
    """Get the names of nodes/links corresponding to data columns"""

    root = zarr.open(store=data_path, mode='r')

    # Get ordered names
    if 'ordered_names_by_attr' in root.attrs:
        ordered_names = root.attrs['ordered_names_by_attr']

        print("Available attribute names:")
        for attr, names in ordered_names.items():
            print(f"\n{attr}:")
            print(f"  Number of items: {len(names)}")
            print(f"  First 5: {names[:5]}")
            print(f"  Last 5: {names[-5:]}")

        return ordered_names
    else:
        print("Node/link names not found in metadata")
        return None


# Example 6: Compare train/valid/test splits
def example_compare_splits(data_path="datasets/ctown.zip"):
    """Compare statistics across train/valid/test splits"""

    root = zarr.open(store=data_path, mode='r')

    if 'pressure' not in root.group_keys():
        print("Pressure data not found")
        return

    pressure_group = root['pressure']

    for split in ['train', 'valid', 'test']:
        if split in pressure_group.group_keys():
            data = pressure_group[split][:]
            print(f"\n{split.upper()} split:")
            print(f"  Shape: {data.shape}")
            print(f"  Min: {np.min(data):.2f}")
            print(f"  Max: {np.max(data):.2f}")
            print(f"  Mean: {np.mean(data):.2f}")
            print(f"  Std: {np.std(data):.2f}")


# Example 7: Check configuration
def example_check_config(data_path="datasets/ctown.zip"):
    """Read the configuration used to generate the data"""

    root = zarr.open(store=data_path, mode='r')

    if 'config' in root.attrs:
        config = root.attrs['config']

        print("Generation Configuration:")
        for section, values in config.items():
            print(f"\n[{section}]")
            for k, v in values.items():
                print(f"  {k} = {v}")

    if 'args' in root.attrs:
        args = root.attrs['args']
        print("\n\nGeneration Arguments:")
        for k, v in args.items():
            if v is not None:
                print(f"  {k} = {v}")


# Example 8: Batch processing (memory efficient)
def example_batch_processing(data_path="datasets/ctown.zip", batch_size=10):
    """Process data in batches to save memory"""

    root = zarr.open(store=data_path, mode='r')
    pressure_train = root['pressure']['train']

    total_scenarios = pressure_train.shape[0]
    num_batches = (total_scenarios + batch_size - 1) // batch_size

    print(f"Processing {total_scenarios} scenarios in {num_batches} batches")

    all_mins = []
    all_maxs = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_scenarios)

        # Load only one batch at a time
        batch_data = pressure_train[start_idx:end_idx]

        batch_min = np.min(batch_data)
        batch_max = np.max(batch_data)

        all_mins.append(batch_min)
        all_maxs.append(batch_max)

        print(f"  Batch {batch_idx+1}/{num_batches}: "
              f"min={batch_min:.2f}, max={batch_max:.2f}")

    print(f"\nOverall min: {min(all_mins):.2f}")
    print(f"Overall max: {max(all_maxs):.2f}")


if __name__ == '__main__':
    print("=" * 80)
    print("Simulation Data Reading Examples")
    print("=" * 80)

    # Note: Update the data_path to your actual data file
    data_path = "datasets/ctown.zip"

    print("\n\nExample 1: Open and explore")
    print("-" * 80)
    try:
        example_open_and_explore()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have generated data first!")

    # Uncomment to run other examples:

    # print("\n\nExample 2: Read pressure data")
    # print("-" * 80)
    # example_read_pressure(data_path)

    # print("\n\nExample 3: Read demand data")
    # print("-" * 80)
    # example_read_demand(data_path)

    # print("\n\nExample 4: Read specific scenarios")
    # print("-" * 80)
    # example_read_specific_scenarios(data_path, scenario_ids=[0, 1, 2])

    # print("\n\nExample 5: Get node names")
    # print("-" * 80)
    # example_get_node_names(data_path)

    # print("\n\nExample 6: Compare splits")
    # print("-" * 80)
    # example_compare_splits(data_path)

    # print("\n\nExample 7: Check configuration")
    # print("-" * 80)
    # example_check_config(data_path)

    # print("\n\nExample 8: Batch processing")
    # print("-" * 80)
    # example_batch_processing(data_path, batch_size=10)
