#!/usr/bin/env python3
"""
Simplified Scenario Generator
Based on scenegenv7.py but without batching/zarr complexity

Purpose: Generate individual scenario .pkl files for water network simulations
Usage: python generate_scenarios.py --config configs/v7.1/ctown_7v1__EPYNET_config.ini
"""

import sys
import os
import argparse
import pickle
import numpy as np
from configparser import ConfigParser
from time import time
from tqdm import tqdm
import ray

# Import from existing codebase
from generator.EPYNET.TokenGeneratorByRange import TokenGenerator
from generator.EPYNET.Executorv7 import WDNExecutor, WDNRayExecutor


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate Water Network Scenarios')

    # Main config
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration .ini file")

    # Parallelization
    parser.add_argument("--executors", type=int, default=1,
                       help="Number of parallel workers (1=sequential, >1=parallel with Ray)")
    parser.add_argument("--scenarios_per_executor", type=int, default=10,
                       help="Number of scenarios each executor processes at once")

    # Parameter sampling flags
    parser.add_argument("--gen_demand", action='store_true', default=False,
                       help="Sample junction demands")
    parser.add_argument("--gen_res_total_head", action='store_true', default=False,
                       help="Sample reservoir total heads")
    parser.add_argument("--gen_elevation", action='store_true', default=False,
                       help="Sample nodal elevations")
    parser.add_argument("--gen_roughness", action='store_true', default=False,
                       help="Sample pipe roughness")
    parser.add_argument("--gen_diameter", action='store_true', default=False,
                       help="Sample pipe diameters")
    parser.add_argument("--gen_length", action='store_true', default=False,
                       help="Sample pipe lengths")
    parser.add_argument("--gen_minorloss", action='store_true', default=False,
                       help="Sample pipe minor losses")
    parser.add_argument("--gen_tank_level", action='store_true', default=False,
                       help="Sample tank levels")
    parser.add_argument("--gen_pump_speed", action='store_true', default=False,
                       help="Sample pump speeds")
    parser.add_argument("--gen_valve_setting", action='store_true', default=False,
                       help="Sample valve settings")

    # Output settings
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: from config storage_dir)")
    parser.add_argument("--att", type=str, default="pressure,head",
                       help="Attributes to extract (e.g., 'pressure,head,flow,velocity')")

    # Filtering/quality control
    parser.add_argument("--pressure_lowerbound", type=float, default=None,
                       help="Minimum acceptable pressure (m)")
    parser.add_argument("--pressure_upperbound", type=float, default=None,
                       help="Maximum acceptable pressure (m)")
    parser.add_argument("--accept_warning_code", action='store_true', default=False,
                       help="Accept EPANET warning codes")
    parser.add_argument("--allow_error", action='store_true', default=False,
                       help="Save scenarios even if they have errors")

    # Other settings
    parser.add_argument("--debug", action='store_true', default=False,
                       help="Print debug information")
    parser.add_argument("--remove_pattern", action='store_true', default=True,
                       help="Remove patterns from INP file")
    parser.add_argument("--remove_control", action='store_true', default=False,
                       help="Remove controls from INP file")
    parser.add_argument("--remove_rule", action='store_true', default=False,
                       help="Remove rules from INP file")
    parser.add_argument("--convert_results_by_flow_unit", type=str, default="LPS",
                       help="Convert results to flow unit (LPS, LPM, MLD, CMH, CMD, None)")

    # Additional flags with defaults
    parser.add_argument("--init_valve_state", type=int, default=1,
                       help="Initial valve state")
    parser.add_argument("--init_pipe_state", type=int, default=None,
                       help="Initial pipe state")
    parser.add_argument("--replace_nonzero_basedmd", action='store_true', default=False)
    parser.add_argument("--update_totalhead_method", type=str, default=None)
    parser.add_argument("--ele_kmean_init", type=str, default="k-means++")
    parser.add_argument("--update_elevation_method", type=str, default="ran_cluster")
    parser.add_argument("--ele_std", type=float, default=1.0)
    parser.add_argument("--mean_cv_threshold", type=float, default=None)
    parser.add_argument("--neighbor_std_threshold", type=float, default=None)
    parser.add_argument("--skip_resevoir_result", action='store_true', default=False)
    parser.add_argument("--flowrate_threshold", type=float, default=None)

    # Pump/valve/tank additional options
    parser.add_argument("--gen_pump_init_status", action='store_true', default=False)
    parser.add_argument("--gen_pump_length", action='store_true', default=False)
    parser.add_argument("--gen_valve_init_status", action='store_true', default=False)
    parser.add_argument("--gen_valve_diameter", action='store_true', default=False)
    parser.add_argument("--gen_tank_elevation", action='store_true', default=False)
    parser.add_argument("--gen_tank_diameter", action='store_true', default=False)

    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration file"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = ConfigParser()
    config.read(config_path)
    print(f"\n{'='*80}")
    print(f"Configuration loaded: {config_path}")
    print(f"{'='*80}\n")

    # Print key info
    print(f"  INP file:        {config.get('general', 'wn_inp_path')}")
    print(f"  Num scenarios:   {config.getint('general', 'num_scenarios')}")
    print(f"  Storage dir:     {config.get('general', 'storage_dir')}")
    print()

    return config


def save_scenario(scenario_data, scenario_id, output_dir):
    """Save a single scenario to a pickle file

    Args:
        scenario_data (dict): Scenario data containing simulation results
        scenario_id (int): Scenario ID
        output_dir (str): Output directory
    """
    filename = f"scenario_{scenario_id:06d}.pkl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(scenario_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath


def generate_scenarios_sequential(token_generator, executor, num_scenarios, output_dir, args):
    """Generate scenarios sequentially (no parallelization)

    Args:
        token_generator: Token generator object
        executor: WDN executor object
        num_scenarios (int): Number of scenarios to generate
        output_dir (str): Output directory
        args: Command-line arguments

    Returns:
        tuple: (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    print(f"\n{'='*80}")
    print(f"Generating {num_scenarios} scenarios (sequential mode)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    for scenario_id in tqdm(range(num_scenarios), desc="Generating scenarios"):
        # Generate random parameters (tokens)
        tokens = token_generator.generate_a_token()

        # Run simulation
        sim_results, error, node_indices = executor.epynet_simulate2(tokens, scenario_id)

        # Check if simulation was successful
        if not error or args.allow_error:
            # Package the data
            scenario_data = {
                'id': scenario_id,
                'tokens': tokens,
                'results': sim_results,
                'node_indices': node_indices,
                'error': error,
            }

            # Save to file
            save_scenario(scenario_data, scenario_id, output_dir)
            successful += 1
        else:
            failed += 1
            if args.debug:
                print(f"  Scenario {scenario_id} failed simulation")

    return successful, failed


def generate_scenarios_parallel(token_generator, executor_class, config, valve_type_dict,
                                num_scenarios, output_dir, args):
    """Generate scenarios in parallel using Ray

    Args:
        token_generator: Token generator object
        executor_class: Ray executor class
        config: Configuration object
        valve_type_dict: Valve type dictionary
        num_scenarios (int): Number of scenarios to generate
        output_dir (str): Output directory
        args: Command-line arguments

    Returns:
        tuple: (successful_count, failed_count)
    """
    successful = 0
    failed = 0

    print(f"\n{'='*80}")
    print(f"Generating {num_scenarios} scenarios (parallel mode)")
    print(f"Executors: {args.executors}")
    print(f"Scenarios per executor: {args.scenarios_per_executor}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.executors, ignore_reinit_error=True)

    # Create Ray executors
    featlen_dict = token_generator.featlen
    executors = [executor_class.remote(featlen_dict, config, valve_type_dict, args)
                 for _ in range(args.executors)]

    # Generate all tokens upfront
    print("Generating parameter tokens...")
    all_tokens = np.array([token_generator.generate_a_token() for _ in range(num_scenarios)])

    # Process in batches
    batch_size = args.scenarios_per_executor
    num_batches = (num_scenarios + batch_size - 1) // batch_size

    futures = []
    scenario_id = 0

    print("Submitting tasks to executors...")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_scenarios)
        batch_tokens = all_tokens[start_idx:end_idx]
        batch_ids = list(range(start_idx, end_idx))

        # Round-robin assignment to executors
        executor = executors[batch_idx % args.executors]
        future = executor.simulate.remote(batch_tokens, batch_ids)
        futures.append((future, batch_ids))

    # Collect results
    print("Processing results...")
    with tqdm(total=num_scenarios, desc="Scenarios completed") as pbar:
        for future, batch_ids in futures:
            batch_results, node_indices = ray.get(future)

            # batch_results is a dict with keys like 'pressure', 'head'
            # Each value is a 2D array: [num_scenarios_in_batch, num_nodes]
            if len(batch_results) == 0:
                failed += len(batch_ids)
                pbar.update(len(batch_ids))
                continue

            # Extract number of scenarios in this batch
            first_key = list(batch_results.keys())[0]
            num_in_batch = batch_results[first_key].shape[0]

            # Save each scenario
            for i in range(num_in_batch):
                scenario_id = batch_ids[i]

                # Extract this scenario's results
                scenario_results = {}
                for key, values in batch_results.items():
                    scenario_results[key] = values[i]  # Extract row i

                scenario_data = {
                    'id': scenario_id,
                    'tokens': all_tokens[scenario_id],
                    'results': scenario_results,
                    'node_indices': node_indices,
                    'error': False,
                }

                save_scenario(scenario_data, scenario_id, output_dir)
                successful += 1
                pbar.update(1)

            # Account for failed scenarios in this batch
            if num_in_batch < len(batch_ids):
                failed += len(batch_ids) - num_in_batch
                pbar.update(len(batch_ids) - num_in_batch)

    # Shutdown Ray
    ray.shutdown()

    return successful, failed


def main():
    """Main function"""
    start_time = time()

    # Parse arguments
    args = parse_arguments()

    # Load config
    config = load_config(args.config)

    # Get parameters from config
    num_scenarios = config.getint('general', 'num_scenarios')
    output_dir = args.output_dir if args.output_dir else config.get('general', 'storage_dir')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Print generation settings
    print(f"Generation Settings:")
    print(f"  Demand:          {'YES' if args.gen_demand else 'NO'}")
    print(f"  Reservoir heads: {'YES' if args.gen_res_total_head else 'NO'}")
    print(f"  Elevation:       {'YES' if args.gen_elevation else 'NO'}")
    print(f"  Pipe roughness:  {'YES' if args.gen_roughness else 'NO'}")
    print(f"  Pipe diameter:   {'YES' if args.gen_diameter else 'NO'}")
    print(f"  Tank levels:     {'YES' if args.gen_tank_level else 'NO'}")
    print(f"  Pump speeds:     {'YES' if args.gen_pump_speed else 'NO'}")
    print()

    # Initialize token generator
    print("Initializing token generator...")
    token_generator = TokenGenerator(config, args)
    valve_type_dict = token_generator.valve_type_dict

    # Choose execution mode
    if args.executors == 1:
        # Sequential mode
        print("Mode: Sequential (single-threaded)")

        # Initialize executor
        featlen_dict = token_generator.featlen
        executor = WDNExecutor(featlen_dict, config, valve_type_dict, args)

        # Generate scenarios
        successful, failed = generate_scenarios_sequential(
            token_generator, executor, num_scenarios, output_dir, args
        )
    else:
        # Parallel mode with Ray
        print(f"Mode: Parallel ({args.executors} workers)")

        # Generate scenarios
        successful, failed = generate_scenarios_parallel(
            token_generator, WDNRayExecutor, config, valve_type_dict,
            num_scenarios, output_dir, args
        )

    # Print summary
    elapsed_time = time() - start_time
    print(f"\n{'='*80}")
    print(f"Generation Complete!")
    print(f"{'='*80}")
    print(f"  Successful:  {successful}/{num_scenarios} ({100*successful/num_scenarios:.1f}%)")
    print(f"  Failed:      {failed}/{num_scenarios} ({100*failed/num_scenarios:.1f}%)")
    print(f"  Time:        {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"  Output:      {output_dir}/")
    print(f"{'='*80}\n")

    # Save metadata
    metadata = {
        'num_scenarios_requested': num_scenarios,
        'num_successful': successful,
        'num_failed': failed,
        'args': vars(args),
        'config_path': args.config,
        'elapsed_time': elapsed_time,
    }

    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}\n")

    # Example: How to load a scenario
    if successful > 0:
        print("Example: Loading a scenario file")
        print("-" * 80)
        example_file = os.path.join(output_dir, "scenario_000000.pkl")
        with open(example_file, 'rb') as f:
            scenario = pickle.load(f)

        print(f"  File: {example_file}")
        print(f"  Keys: {list(scenario.keys())}")
        print(f"  Results keys: {list(scenario['results'].keys())}")
        print(f"  Number of nodes: {len(scenario['node_indices'])}")

        for key, values in scenario['results'].items():
            print(f"  {key:15s}: shape={values.shape}, min={values.min():.4f}, max={values.max():.4f}")
        print()


if __name__ == '__main__':
    main()
