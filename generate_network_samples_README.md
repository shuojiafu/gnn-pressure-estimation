# Generate Network Samples Script

This script generates multiple EPANET inp files with randomized parameters for water distribution network simulation.

## Features

The script randomizes the following parameters according to specific rules:

### Node Parameters
- **Junction Demand**: Only modifies junctions with existing non-zero demands. Random values are generated within the range [0, max_demand] where max_demand is the maximum demand observed in the original network.

- **Reservoir Total Head**: Randomized between 0.5 to 2 times the original reservoir head values, simulating various operating conditions.

- **Tank Level**: Randomized between the tank's minimum and maximum level constraints defined in the original network.

### Link Parameters
- **Pump Speed**: Randomized between 0.8 to 1.2, representing typical operational variations.

- **Pump Status**: Each pump has an 80% probability of being in the open state during simulation (0.8 open probability).

- **Valve Setting**: Randomized within their type-specific ranges:
  - PRV/PSV (Pressure valves): 0 to max(2×current_setting, 100)
  - FCV (Flow control): 0 to max(2×current_setting, 10)
  - TCV (Throttle control): 0 to 1

- **Valve Status**: Each valve has an 80% probability of being in the open state (0.8 open probability).

## Installation

Required Python packages:
```bash
pip install numpy wntr tqdm
```

## Usage

### Basic Usage
```bash
python generate_network_samples.py --input file.inp --num_samples 1000
```

### Full Command with All Options
```bash
python generate_network_samples.py \
    --input file.inp \
    --output_dir network_samples \
    --num_samples 1000 \
    --open_probability 0.8 \
    --prefix network
```

### Arguments

- `--input` (required): Path to the original EPANET inp file
- `--output_dir` (optional): Directory to save generated inp files (default: `network_samples`)
- `--num_samples` (optional): Number of samples to generate (default: 1000)
- `--open_probability` (optional): Probability that pumps and valves are open (default: 0.8)
- `--prefix` (optional): Prefix for output filenames (default: `network`)

## Example

Generate 1000 different network configurations:
```bash
python generate_network_samples.py \
    --input inputs/ctown.inp \
    --output_dir generated_networks \
    --num_samples 1000 \
    --prefix ctown
```

This will create files named:
- `ctown_0001.inp`
- `ctown_0002.inp`
- ...
- `ctown_1000.inp`

in the `generated_networks/` directory.

## Output

Each generated inp file contains:
- Randomized junction demands (only for junctions with existing demands)
- Randomized reservoir heads (0.5-2× original values)
- Randomized tank levels (within min-max constraints)
- Randomized pump speeds (0.8-1.2) and statuses (80% open)
- Randomized valve settings (type-specific ranges) and statuses (80% open)

## Technical Details

The script uses the WNTR (Water Network Tool for Resilience) library to:
1. Load the original EPANET inp file
2. Identify components with existing non-zero values (for demands)
3. Calculate appropriate ranges for each parameter
4. Generate random variations while respecting network constraints
5. Write modified networks to new inp files

## Notes

- The script preserves all other network properties (pipe lengths, diameters, roughness, etc.)
- Only junctions with existing demands are modified; zero-demand junctions remain at zero
- All random values respect the physical constraints defined in the original network
- The 0.8 open probability for pumps and valves reflects typical operational conditions
