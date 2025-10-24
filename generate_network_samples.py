#!/usr/bin/env python3
"""
Generate multiple EPANET inp files with randomized parameters.

This script generates N different inp files by modifying various network parameters:
- Node parameters: junction demand, reservoir total head, tank level
- Link parameters: pump speed/status, valve setting/status

Usage:
    python generate_network_samples.py --input file.inp --output_dir samples/ --num_samples 1000
"""

import os
import argparse
import numpy as np
import wntr
from pathlib import Path
from tqdm import tqdm


def get_junctions_with_demands(wn):
    """Get list of junctions that have non-zero base demands."""
    junctions_with_demands = []
    for junction_name in wn.junction_name_list:
        junction = wn.get_node(junction_name)
        if hasattr(junction, 'base_demand') and junction.base_demand != 0:
            junctions_with_demands.append(junction_name)
    return junctions_with_demands


def get_parameter_ranges(wn):
    """Calculate parameter ranges for randomization based on network properties."""
    ranges = {}

    # Junction demand ranges (for junctions with existing demands)
    junctions_with_demands = get_junctions_with_demands(wn)
    if junctions_with_demands:
        demands = []
        demand_ranges = {}
        for junction_name in junctions_with_demands:
            junction = wn.get_node(junction_name)
            demand = junction.base_demand
            # Get demand pattern if exists
            if junction.demand_timeseries_list:
                demand_pattern = junction.demand_timeseries_list[0]
                if hasattr(demand_pattern, 'pattern_name') and demand_pattern.pattern_name:
                    pattern = wn.get_pattern(demand_pattern.pattern_name)
                    demand = demand * pattern.multipliers[0]
            demands.append(demand)
            # Store individual junction demand for min/max calculation
            demand_ranges[junction_name] = demand

        # For each junction, we'll use 0 as min (can have no demand) and max from the network
        ranges['junction_demand'] = {
            'junctions': junctions_with_demands,
            'min': 0,
            'max': max(demands) if demands else 0
        }

    # Reservoir total head ranges (0.5 to 2 times original)
    reservoir_heads = {}
    for reservoir_name in wn.reservoir_name_list:
        reservoir = wn.get_node(reservoir_name)
        head = reservoir.base_head if hasattr(reservoir, 'base_head') else reservoir.head_timeseries.base_value
        reservoir_heads[reservoir_name] = {
            'min': head * 0.5,
            'max': head * 2.0,
            'original': head
        }
    ranges['reservoir_head'] = reservoir_heads

    # Tank level ranges (between min and max levels)
    tank_levels = {}
    for tank_name in wn.tank_name_list:
        tank = wn.get_node(tank_name)
        tank_levels[tank_name] = {
            'min': tank.min_level,
            'max': tank.max_level
        }
    ranges['tank_level'] = tank_levels

    # Pump speed ranges (0.8 to 1.2)
    pump_speeds = {}
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        pump_speeds[pump_name] = {
            'min': 0.8,
            'max': 1.2
        }
    ranges['pump_speed'] = pump_speeds

    # Valve setting ranges (between min and max for each valve type)
    valve_settings = {}
    for valve_name in wn.valve_name_list:
        valve = wn.get_link(valve_name)
        # Different valve types have different setting ranges
        valve_type = valve.valve_type

        if valve_type == 'PRV' or valve_type == 'PSV':  # Pressure valves
            # Setting is in pressure units, use reasonable range
            current_setting = valve.initial_setting if hasattr(valve, 'initial_setting') else 0
            valve_settings[valve_name] = {
                'min': 0,
                'max': max(current_setting * 2, 100),  # Allow up to 2x current or 100
                'type': valve_type
            }
        elif valve_type == 'FCV':  # Flow control valve
            current_setting = valve.initial_setting if hasattr(valve, 'initial_setting') else 0
            valve_settings[valve_name] = {
                'min': 0,
                'max': max(current_setting * 2, 10),  # Allow up to 2x current or 10
                'type': valve_type
            }
        elif valve_type == 'TCV':  # Throttle control valve
            valve_settings[valve_name] = {
                'min': 0,
                'max': 1,  # Throttle coefficient
                'type': valve_type
            }
        else:  # GPV, PBV, or other
            current_setting = valve.initial_setting if hasattr(valve, 'initial_setting') else 0
            valve_settings[valve_name] = {
                'min': 0,
                'max': max(current_setting * 2, 1),
                'type': valve_type
            }
    ranges['valve_setting'] = valve_settings

    return ranges


def randomize_network(wn, ranges, open_probability=0.8):
    """
    Create a randomized version of the network.

    Args:
        wn: WNTR WaterNetworkModel
        ranges: Dictionary of parameter ranges
        open_probability: Probability that pumps and valves are open (default 0.8)

    Returns:
        Modified WaterNetworkModel
    """
    # Randomize junction demands (only for junctions with existing demands)
    if 'junction_demand' in ranges and ranges['junction_demand']['junctions']:
        for junction_name in ranges['junction_demand']['junctions']:
            junction = wn.get_node(junction_name)
            # Random demand between min and max
            new_demand = np.random.uniform(
                ranges['junction_demand']['min'],
                ranges['junction_demand']['max']
            )
            # Modify demand using demand_timeseries_list
            if junction.demand_timeseries_list:
                junction.demand_timeseries_list[0].base_value = new_demand
            else:
                # If no demand timeseries exists, create one
                junction.add_demand(new_demand)

    # Randomize reservoir total heads (0.5 to 2 times original)
    if 'reservoir_head' in ranges:
        for reservoir_name, head_range in ranges['reservoir_head'].items():
            reservoir = wn.get_node(reservoir_name)
            new_head = np.random.uniform(head_range['min'], head_range['max'])
            if hasattr(reservoir, 'base_head'):
                reservoir.base_head = new_head
            else:
                reservoir.head_timeseries.base_value = new_head

    # Randomize tank levels (between min and max)
    if 'tank_level' in ranges:
        for tank_name, level_range in ranges['tank_level'].items():
            tank = wn.get_node(tank_name)
            new_level = np.random.uniform(level_range['min'], level_range['max'])
            tank.init_level = new_level

    # Randomize pump speeds (0.8 to 1.2) and status (0.8 probability open)
    if 'pump_speed' in ranges:
        for pump_name in ranges['pump_speed'].keys():
            pump = wn.get_link(pump_name)
            # Randomize speed
            new_speed = np.random.uniform(0.8, 1.2)
            pump.speed_timeseries.base_value = new_speed
            # Randomize status (0.8 probability of being open)
            is_open = np.random.random() < open_probability
            pump.initial_status = wntr.network.LinkStatus.Open if is_open else wntr.network.LinkStatus.Closed

    # Randomize valve settings and status (0.8 probability open)
    if 'valve_setting' in ranges:
        for valve_name, setting_range in ranges['valve_setting'].items():
            valve = wn.get_link(valve_name)
            # Randomize setting
            new_setting = np.random.uniform(setting_range['min'], setting_range['max'])
            valve.initial_setting = new_setting
            # Randomize status (0.8 probability of being open)
            is_open = np.random.random() < open_probability
            valve.initial_status = wntr.network.LinkStatus.Open if is_open else wntr.network.LinkStatus.Closed

    return wn


def generate_network_samples(input_file, output_dir, num_samples=1000, open_probability=0.8, prefix="network"):
    """
    Generate multiple randomized network samples.

    Args:
        input_file: Path to the original EPANET inp file
        output_dir: Directory to save generated inp files
        num_samples: Number of samples to generate (default 1000)
        open_probability: Probability that pumps and valves are open (default 0.8)
        prefix: Prefix for output filenames (default "network")
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the original network
    print(f"Loading network from {input_file}...")
    wn_original = wntr.network.WaterNetworkModel(input_file)

    # Calculate parameter ranges
    print("Calculating parameter ranges...")
    ranges = get_parameter_ranges(wn_original)

    # Print summary
    print("\nParameter ranges:")
    if 'junction_demand' in ranges and ranges['junction_demand']['junctions']:
        print(f"  Junction demands: {len(ranges['junction_demand']['junctions'])} junctions with demands")
        print(f"    Range: [{ranges['junction_demand']['min']:.4f}, {ranges['junction_demand']['max']:.4f}]")
    if 'reservoir_head' in ranges:
        print(f"  Reservoir heads: {len(ranges['reservoir_head'])} reservoirs")
        for res_name, res_range in ranges['reservoir_head'].items():
            print(f"    {res_name}: [{res_range['min']:.2f}, {res_range['max']:.2f}] (original: {res_range['original']:.2f})")
    if 'tank_level' in ranges:
        print(f"  Tank levels: {len(ranges['tank_level'])} tanks")
        for tank_name, tank_range in ranges['tank_level'].items():
            print(f"    {tank_name}: [{tank_range['min']:.2f}, {tank_range['max']:.2f}]")
    if 'pump_speed' in ranges:
        print(f"  Pump speeds: {len(ranges['pump_speed'])} pumps")
        print(f"    Range: [0.8, 1.2]")
        print(f"    Open probability: {open_probability}")
    if 'valve_setting' in ranges:
        print(f"  Valve settings: {len(ranges['valve_setting'])} valves")
        print(f"    Open probability: {open_probability}")

    # Generate samples
    print(f"\nGenerating {num_samples} network samples...")
    successful_samples = 0

    for i in tqdm(range(num_samples)):
        try:
            # Create a copy of the original network
            wn = wntr.network.WaterNetworkModel(input_file)

            # Randomize parameters
            wn = randomize_network(wn, ranges, open_probability)

            # Save to file
            output_file = output_path / f"{prefix}_{i+1:04d}.inp"
            wntr.network.write_inpfile(wn, str(output_file))

            successful_samples += 1

        except Exception as e:
            print(f"\nError generating sample {i+1}: {e}")
            continue

    print(f"\nSuccessfully generated {successful_samples}/{num_samples} network samples")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate multiple EPANET inp files with randomized parameters'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the original EPANET inp file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='network_samples',
        help='Directory to save generated inp files (default: network_samples)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--open_probability',
        type=float,
        default=0.8,
        help='Probability that pumps and valves are open (default: 0.8)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='network',
        help='Prefix for output filenames (default: network)'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return

    # Validate open probability
    if not 0 <= args.open_probability <= 1:
        print("Error: open_probability must be between 0 and 1")
        return

    # Generate samples
    generate_network_samples(
        input_file=args.input,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        open_probability=args.open_probability,
        prefix=args.prefix
    )


if __name__ == '__main__':
    main()
