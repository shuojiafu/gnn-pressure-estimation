# How to Read and Inspect Simulation Data

This guide shows you how to read the generated simulation files and check demand, pressure, and other values.

## Data Storage Format

The simulation data is stored in **Zarr** format, which is a high-performance array storage format. The data can be stored as:
- **Zip file**: `datasets/ctown.zip` (compressed, portable)
- **Directory**: `datasets/ctown/zarrays/` (faster access)

## Data Structure

After generation, your data will be organized as:

```
datasets/ctown.zip
├── pressure/
│   ├── train/    # Training data
│   ├── valid/    # Validation data
│   └── test/     # Test data
├── demand/       # (if generated)
│   ├── train/
│   ├── valid/
│   └── test/
├── head/         # (if generated)
├── flow/         # (if generated)
└── velocity/     # (if generated)
```

Each array has shape: `[num_scenarios, num_nodes_or_links]`

## Method 1: Using the Inspection Script (Easiest)

We've created a ready-to-use script:

```bash
# Basic inspection - shows all features
python inspect_simulation_data.py --data_path datasets/ctown.zip

# Inspect specific feature
python inspect_simulation_data.py --data_path datasets/ctown.zip --feature pressure

# Inspect demand values specifically
python inspect_simulation_data.py --data_path datasets/ctown.zip --feature demand --analyze_demands

# Look at validation set
python inspect_simulation_data.py --data_path datasets/ctown.zip --feature pressure --subset valid
```

The script will show:
- Dataset structure
- Statistics (min, max, mean, std)
- Warnings for negative pressure
- Configuration used for generation
- Node/link names

## Method 2: Python Code (Programmatic)

### Quick Start - Read Pressure Data

```python
import zarr
import numpy as np

# Open the zarr file
root = zarr.open(store="datasets/ctown.zip", mode='r')

# Read pressure data for training set
pressure_train = root['pressure']['train']

# Load into memory
pressure_array = pressure_train[:]

# Check statistics
print(f"Shape: {pressure_array.shape}")
print(f"Min pressure: {np.min(pressure_array):.2f}")
print(f"Max pressure: {np.max(pressure_array):.2f}")
print(f"Mean pressure: {np.mean(pressure_array):.2f}")

# Check for negative pressure
if np.any(pressure_array < 0):
    print(f"⚠️  WARNING: Negative pressure detected!")
    print(f"   Min: {np.min(pressure_array):.2f}")
```

### Read Demand Data

```python
import zarr

root = zarr.open(store="datasets/ctown.zip", mode='r')

# Check if demand data exists
if 'demand' in root.group_keys():
    demand_train = root['demand']['train']
    demand_array = demand_train[:]

    print(f"Demand shape: {demand_array.shape}")
    print(f"Min demand: {np.min(demand_array):.6f}")
    print(f"Max demand: {np.max(demand_array):.6f}")
    print(f"Mean demand: {np.mean(demand_array):.6f}")

    # Show first scenario
    print(f"\nFirst scenario demands:")
    print(demand_array[0])
else:
    print("Demand data not generated")
```

### Read Specific Scenarios (Memory Efficient)

```python
import zarr

root = zarr.open(store="datasets/ctown.zip", mode='r')

# Read only specific scenarios
scenario_id = 0
pressure_scenario = root['pressure']['train'][scenario_id]

print(f"Scenario {scenario_id} pressure:")
print(f"  Min: {np.min(pressure_scenario):.2f}")
print(f"  Max: {np.max(pressure_scenario):.2f}")

# Or read a range
pressure_first_10 = root['pressure']['train'][0:10]
print(f"\nFirst 10 scenarios shape: {pressure_first_10.shape}")
```

### Get Node/Link Names

```python
import zarr

root = zarr.open(store="datasets/ctown.zip", mode='r')

# Get ordered names for each attribute
if 'ordered_names_by_attr' in root.attrs:
    names = root.attrs['ordered_names_by_attr']

    # Get pressure node names
    pressure_nodes = names['pressure']
    print(f"Pressure is measured at {len(pressure_nodes)} nodes:")
    print(pressure_nodes[:10])  # First 10 nodes

    # Get demand node names (if exists)
    if 'demand' in names:
        demand_nodes = names['demand']
        print(f"\nDemand is set at {len(demand_nodes)} junctions:")
        print(demand_nodes[:10])
```

### Check Generation Configuration

```python
import zarr

root = zarr.open(store="datasets/ctown.zip", mode='r')

# Read the configuration used for generation
if 'config' in root.attrs:
    config = root.attrs['config']

    # Check junction demand range
    if 'junction' in config:
        print("Junction configuration:")
        print(f"  demand_lo: {config['junction']['demand_lo']}")
        print(f"  demand_hi: {config['junction']['demand_hi']}")
        print(f"  ele_lo: {config['junction']['ele_lo']}")
        print(f"  ele_hi: {config['junction']['ele_hi']}")

    # Check reservoir configuration
    if 'reservoir' in config:
        print("\nReservoir configuration:")
        print(f"  head_lo: {config['reservoir']['head_lo']}")
        print(f"  head_hi: {config['reservoir']['head_hi']}")

# Read generation arguments
if 'args' in root.attrs:
    args = root.attrs['args']
    print(f"\nPressure bounds used:")
    print(f"  Lower bound: {args.get('pressure_lowerbound', 'None')}")
    print(f"  Upper bound: {args.get('pressure_upperbound', 'None')}")
```

## Method 3: Using Jupyter Notebook

For interactive exploration, you can use Jupyter:

```python
# In a Jupyter notebook
import zarr
import numpy as np
import matplotlib.pyplot as plt

# Load data
root = zarr.open(store="datasets/ctown.zip", mode='r')
pressure = root['pressure']['train'][:]

# Plot histogram of all pressure values
plt.figure(figsize=(10, 6))
plt.hist(pressure.flatten(), bins=50)
plt.xlabel('Pressure')
plt.ylabel('Frequency')
plt.title('Pressure Distribution Across All Scenarios')
plt.axvline(x=0, color='r', linestyle='--', label='Zero pressure')
plt.legend()
plt.show()

# Plot one scenario
plt.figure(figsize=(10, 6))
plt.plot(pressure[0], 'o-')
plt.xlabel('Node Index')
plt.ylabel('Pressure')
plt.title('Pressure at Each Node (Scenario 0)')
plt.grid(True)
plt.show()
```

## Common Issues and Solutions

### Issue 1: File Not Found

```
ERROR: File not found: datasets/ctown.zip
```

**Solution**: Generate the data first:
```bash
python gnn_pressure_estimation/scenegenv7.py \
    --config configs/v7.1/ctown_7v1__EPYNET_config.ini \
    --pressure_lowerbound 5.0 \
    --pressure_upperbound 100.0
```

### Issue 2: Negative Pressure Values

```
⚠️  WARNING: Found negative pressure values!
```

**Solution**: This means your parameter ranges allow hydraulically invalid scenarios. See the main README or the pressure limitation guide for how to fix the configuration.

### Issue 3: Demand Data Not Found

```
Demand data not found in dataset
```

**Solution**: Demand data is only generated if you use `--gen_demand` flag:
```bash
python gnn_pressure_estimation/scenegenv7.py \
    --config configs/v7.1/ctown_7v1__EPYNET_config.ini \
    --gen_demand
```

### Issue 4: Out of Memory

If you have a large dataset and run out of memory:

```python
# Don't load all at once
# pressure = root['pressure']['train'][:]  # ❌ May use too much memory

# Instead, process in batches
batch_size = 100
for i in range(0, total_scenarios, batch_size):
    batch = root['pressure']['train'][i:i+batch_size]
    # Process batch...
```

## Examples Directory

See `examples/read_data_example.py` for more detailed examples:

```bash
python examples/read_data_example.py
```

This file contains 8 different examples showing various ways to work with the data.

## Quick Reference

### Essential Imports
```python
import zarr
import numpy as np
```

### Open File
```python
root = zarr.open(store="path/to/file.zip", mode='r')
```

### Access Data
```python
# List features
root.group_keys()  # ['pressure', 'demand', 'flow', ...]

# Access specific feature and split
data = root['pressure']['train']

# Get shape without loading
data.shape  # (num_scenarios, num_nodes)

# Load into memory
array = data[:]

# Load specific scenarios
scenarios_0_to_9 = data[0:10]
scenario_5 = data[5]
```

### Get Metadata
```python
# Configuration
config = root.attrs['config']

# Node/link names
names = root.attrs['ordered_names_by_attr']

# Generation arguments
args = root.attrs['args']
```

## Next Steps

1. **Inspect your data**: Run the inspection script to see what you have
2. **Check for issues**: Look for negative pressure, out-of-range values
3. **Adjust configuration**: If needed, update your config and regenerate
4. **Train models**: Use the data with the training scripts

For more information on fixing pressure issues, see the configuration guide.
