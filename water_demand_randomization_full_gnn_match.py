#%%
"""
Complete GNN-matching water network scenario generator.

This script EXACTLY replicates the randomization approach from:
- ConfigCreator.py (config generation)
- TokenGeneratorByRange.py (parameter randomization)
- scenegenv7.py (simulation orchestration)
- Executorv7.py (EPANET simulation)

Key features:
1. Global parameter ranges (not per-element)
2. Spatial clustering for correlated randomization
3. All hydraulic parameters supported
4. Pattern/control removal for steady-state
"""

import os
import math
import pickle
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import wntr
import networkx as nx
from sklearn.cluster import KMeans

# ========= User settings (matching scenegenv7.py args) =========
file_path   = "water_pressure_simulation/ctown.inp"
out_dir     = "Water_quality_data_new/ctown_data"
num_events  = 3
seed        = 42

# DEMAND SETTINGS
gen_demand = True
demand_quantile_lo = 0        # 0-100 or None for absolute min
demand_quantile_hi = 60       # 0-100 or None for absolute max
use_demand_quantiles = True

# ELEVATION SETTINGS
gen_elevation = False          # Set True to randomize elevations
elevation_quantile_lo = None   # None uses min
elevation_quantile_hi = None   # None uses max
use_elevation_quantiles = False
update_elevation_method = "ran_cluster"  # "ran_cluster" | "ran" | "ran_local"
ele_std = 1.0                  # Cluster standard deviation

# PIPE SETTINGS
gen_pipe_roughness = False
pipe_roughness_quantile_lo = None
pipe_roughness_quantile_hi = None
use_pipe_roughness_quantiles = False

gen_pipe_diameter = False
pipe_diameter_quantile_lo = None
pipe_diameter_quantile_hi = None
use_pipe_diameter_quantiles = False

gen_pipe_length = False
pipe_length_quantile_lo = None
pipe_length_quantile_hi = None
use_pipe_length_quantiles = False

gen_pipe_minorloss = False
pipe_minorloss_quantile_lo = None
pipe_minorloss_quantile_hi = None
use_pipe_minorloss_quantiles = False

# PUMP SETTINGS
gen_pump_init_status = True
pump_open_prob = 0.8

gen_pump_speed = True
pump_speed_quantile_lo = None
pump_speed_quantile_hi = None
use_pump_speed_quantiles = False
pump_speed_lo = 0.8
pump_speed_hi = 1.2

gen_pump_length = False
pump_length_quantile_lo = None
pump_length_quantile_hi = None
use_pump_length_quantiles = False

# TANK SETTINGS
gen_tank_level = True
tank_level_quantile_lo = 0
tank_level_quantile_hi = 30
use_tank_level_quantiles = True

gen_tank_elevation = False
tank_elevation_quantile_lo = None
tank_elevation_quantile_hi = None
use_tank_elevation_quantiles = False

gen_tank_diameter = False
tank_diameter_quantile_lo = None
tank_diameter_quantile_hi = None
use_tank_diameter_quantiles = False

# VALVE SETTINGS
gen_valve_init_status = True
valve_open_prob = 0.8

gen_valve_setting = False
gen_valve_diameter = False
valve_diameter_quantile_lo = None
valve_diameter_quantile_hi = None
use_valve_diameter_quantiles = False

# RESERVOIR SETTINGS
gen_res_total_head = True
res_head_lo = 15.0
res_head_hi = 30.0
update_totalhead_method = "add_max_elevation"  # "add_max_elevation" | None
head_add_ele = True

# CLUSTERING SETTINGS
use_spatial_clustering = True  # Enable k-means spatial correlation
num_clusters_lo = 4
num_clusters_hi = 50

# VALIDATION SETTINGS
pressure_ok_lo = 0.0
pressure_ok_hi = 151.0
accept_warning_code = False
skip_reservoir_result = False

# NETWORK MODIFICATION
remove_patterns = True
remove_controls = False
remove_rules = False
# =================================

os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

# ========= HELPER FUNCTIONS (matching ConfigCreator.py) =========

def get_range(arr: np.ndarray, lo: Optional[float], hi: Optional[float],
              is_quantile: bool = False) -> Tuple[float, float]:
    """
    Get range of a hydraulic parameter.
    Matches ConfigCreator.py:73-92
    """
    if is_quantile:
        assert lo is not None and hi is not None
        data_lo = float(np.percentile(arr, lo))
        data_hi = float(np.percentile(arr, hi))
    else:
        data_lo = float(np.min(arr)) if lo is None else float(lo)
        data_hi = float(np.max(arr)) if hi is None else float(hi)
    return data_lo, data_hi


def get_node_coordinates(wn: wntr.network.WaterNetworkModel,
                         do_normalize: bool = True) -> Dict[str, np.ndarray]:
    """Extract and normalize node coordinates. Matches TokenGeneratorByRange.py:167-177"""
    G = wn.to_graph()
    pos_dict = nx.get_node_attributes(G, "pos")

    node_coords = {}
    for node_name in wn.node_name_list:
        if node_name in pos_dict:
            node_coords[node_name] = np.array(pos_dict[node_name])

    if do_normalize and len(node_coords) > 0:
        all_coords = np.array(list(node_coords.values()))
        norm = np.linalg.norm(all_coords) + 1e-12
        node_coords = {k: v / norm for k, v in node_coords.items()}

    return node_coords


def compute_values_by_range(tokens: np.ndarray, range_lo: float, range_hi: float) -> np.ndarray:
    """
    Compute continuous values by range.
    Matches TokenGeneratorByRange.py:74-77
    """
    return range_lo + tokens * (range_hi - range_lo)


def compute_values_by_ran_cluster(tokens: np.ndarray, range_lo: float, range_hi: float,
                                   coords: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute values with spatial clustering (k-means).
    Matches TokenGeneratorByRange.py:99-165
    """
    num_elements = len(coords)

    # Random number of clusters
    n_clusters = int(rng.integers(num_clusters_lo, num_clusters_hi + 1))
    n_clusters = min(n_clusters, num_elements)

    if n_clusters < num_elements:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=int(rng.integers(0, 10000)),
                       n_init='auto')
        labels = kmeans.fit_predict(coords)
    else:
        # Each element is its own cluster
        labels = np.arange(num_elements)

    # Generate cluster centroids
    cluster_values = rng.uniform(range_lo, range_hi, size=n_clusters)

    # Map to elements
    base_values = cluster_values[labels]

    # Add local perturbation
    signs = rng.choice([-1.0, 1.0], size=num_elements)
    perturbations = signs * tokens * sigma

    new_values = base_values + perturbations
    new_values = np.clip(new_values, range_lo, range_hi)

    return new_values


def compute_boolean_values(tokens: np.ndarray, open_prob: float) -> np.ndarray:
    """Compute boolean status. Matches TokenGeneratorByRange.py:79-82"""
    return (tokens < open_prob).astype(float)


# ========= CONFIGURATION COMPUTATION (matching ConfigCreator.py) =========

def compute_global_ranges(wn: wntr.network.WaterNetworkModel) -> Dict:
    """
    Compute global parameter ranges across all elements.
    Matches ConfigCreator.py:95-217
    """
    ranges = {}

    # Junction ranges
    if len(wn.junction_name_list) > 0:
        if gen_demand:
            demands = []
            for j in wn.junction_name_list:
                jn = wn.get_node(j)
                total = sum(float(ts.base_value) for ts in jn.demand_timeseries_list)
                if total > 0:
                    demands.append(total)
            if len(demands) > 0:
                demands = np.array(demands)
                ranges['demand'] = get_range(demands, demand_quantile_lo,
                                            demand_quantile_hi, use_demand_quantiles)

        if gen_elevation:
            elevations = [wn.get_node(j).elevation for j in wn.junction_name_list]
            elevations = np.array(elevations)
            ranges['elevation'] = get_range(elevations, elevation_quantile_lo,
                                           elevation_quantile_hi, use_elevation_quantiles)

    # Pipe ranges
    if len(wn.pipe_name_list) > 0:
        if gen_pipe_roughness:
            roughness = [wn.get_link(p).roughness for p in wn.pipe_name_list]
            roughness = np.array(roughness)
            ranges['pipe_roughness'] = get_range(roughness, pipe_roughness_quantile_lo,
                                                pipe_roughness_quantile_hi,
                                                use_pipe_roughness_quantiles)

        if gen_pipe_diameter:
            diameters = [wn.get_link(p).diameter for p in wn.pipe_name_list]
            diameters = np.array(diameters)
            ranges['pipe_diameter'] = get_range(diameters, pipe_diameter_quantile_lo,
                                               pipe_diameter_quantile_hi,
                                               use_pipe_diameter_quantiles)

        if gen_pipe_length:
            lengths = [wn.get_link(p).length for p in wn.pipe_name_list]
            lengths = np.array(lengths)
            ranges['pipe_length'] = get_range(lengths, pipe_length_quantile_lo,
                                             pipe_length_quantile_hi,
                                             use_pipe_length_quantiles)

        if gen_pipe_minorloss:
            minorlosses = [wn.get_link(p).minor_loss for p in wn.pipe_name_list]
            minorlosses = np.array(minorlosses)
            ranges['pipe_minorloss'] = get_range(minorlosses, pipe_minorloss_quantile_lo,
                                                pipe_minorloss_quantile_hi,
                                                use_pipe_minorloss_quantiles)

    # Pump ranges
    if len(wn.pump_name_list) > 0:
        if gen_pump_speed:
            speeds = [wn.get_link(p).speed_timeseries.base_value for p in wn.pump_name_list]
            speeds = np.array(speeds)
            if use_pump_speed_quantiles:
                ranges['pump_speed'] = get_range(speeds, pump_speed_quantile_lo,
                                                pump_speed_quantile_hi, True)
            else:
                ranges['pump_speed'] = (pump_speed_lo, pump_speed_hi)

        if gen_pump_length:
            # Note: WNTR doesn't have pump.length, skip or use default
            ranges['pump_length'] = (0.0, 0.0)

    # Tank ranges
    if len(wn.tank_name_list) > 0:
        if gen_tank_level:
            min_levels = [wn.get_node(t).min_level for t in wn.tank_name_list]
            max_levels = [wn.get_node(t).max_level for t in wn.tank_name_list]
            min_levels = np.array(min_levels)
            max_levels = np.array(max_levels)
            lo, _ = get_range(min_levels, tank_level_quantile_lo, 100,
                             use_tank_level_quantiles)
            _, hi = get_range(max_levels, 0, tank_level_quantile_hi,
                             use_tank_level_quantiles)
            ranges['tank_level'] = (lo, hi)

        if gen_tank_elevation:
            elevations = [wn.get_node(t).elevation for t in wn.tank_name_list]
            elevations = np.array(elevations)
            ranges['tank_elevation'] = get_range(elevations, tank_elevation_quantile_lo,
                                                 tank_elevation_quantile_hi,
                                                 use_tank_elevation_quantiles)

        if gen_tank_diameter:
            diameters = [wn.get_node(t).diameter for t in wn.tank_name_list]
            diameters = np.array(diameters)
            ranges['tank_diameter'] = get_range(diameters, tank_diameter_quantile_lo,
                                               tank_diameter_quantile_hi,
                                               use_tank_diameter_quantiles)

    # Reservoir ranges
    if len(wn.reservoir_name_list) > 0 and gen_res_total_head:
        ranges['reservoir_head'] = (res_head_lo, res_head_hi)

    print("\n=== Global Parameter Ranges ===")
    for key, (lo, hi) in ranges.items():
        print(f"{key:20s}: [{lo:10.4f}, {hi:10.4f}]")
    print()

    return ranges


# ========= MAIN GENERATION LOOP =========

# Load base network and compute ranges
base_wn = wntr.network.WaterNetworkModel(file_path)
global_ranges = compute_global_ranges(base_wn)

# Get node coordinates for spatial clustering
node_coords_dict = get_node_coordinates(base_wn, do_normalize=True)

saved = 0
attempts = 0

while saved < num_events:
    attempts += 1
    wn = wntr.network.WaterNetworkModel(file_path)

    # Remove patterns/controls/rules (matching Executorv7.py:96-107)
    if remove_patterns:
        for pattern_name in list(wn.pattern_name_list):
            try:
                wn.remove_pattern(pattern_name)
            except:
                pass

    if remove_controls:
        wn.remove_all_controls()

    if remove_rules:
        for rule_name in list(wn.rule_name_list):
            try:
                wn.remove_rule(rule_name)
            except:
                pass

    # ========= RANDOMIZE JUNCTION PARAMETERS =========
    if gen_demand and 'demand' in global_ranges:
        lo, hi = global_ranges['demand']
        num_junctions = len(wn.junction_name_list)

        if use_spatial_clustering and update_elevation_method == "ran_cluster":
            # Spatial clustering approach
            coords = np.array([node_coords_dict.get(j, np.zeros(2))
                              for j in wn.junction_name_list])
            tokens = rng.random(num_junctions)
            new_demands = compute_values_by_ran_cluster(tokens, lo, hi, coords, ele_std)
        else:
            # Simple uniform random
            tokens = rng.random(num_junctions)
            new_demands = compute_values_by_range(tokens, lo, hi)

        for j, new_demand in zip(wn.junction_name_list, new_demands):
            jn = wn.get_node(j)
            if len(jn.demand_timeseries_list) > 0:
                jn.demand_timeseries_list[0].base_value = max(0.0, float(new_demand))

    if gen_elevation and 'elevation' in global_ranges:
        lo, hi = global_ranges['elevation']
        num_junctions = len(wn.junction_name_list)

        if use_spatial_clustering and update_elevation_method == "ran_cluster":
            coords = np.array([node_coords_dict.get(j, np.zeros(2))
                              for j in wn.junction_name_list])
            tokens = rng.random(num_junctions)
            new_elevations = compute_values_by_ran_cluster(tokens, lo, hi, coords, ele_std)
        else:
            tokens = rng.random(num_junctions)
            new_elevations = compute_values_by_range(tokens, lo, hi)

        for j, new_ele in zip(wn.junction_name_list, new_elevations):
            wn.get_node(j).elevation = float(new_ele)

    # ========= RANDOMIZE PIPE PARAMETERS =========
    if gen_pipe_roughness and 'pipe_roughness' in global_ranges:
        lo, hi = global_ranges['pipe_roughness']
        tokens = rng.random(len(wn.pipe_name_list))
        new_values = compute_values_by_range(tokens, lo, hi)
        for p, val in zip(wn.pipe_name_list, new_values):
            wn.get_link(p).roughness = max(1e-12, float(val))

    if gen_pipe_diameter and 'pipe_diameter' in global_ranges:
        lo, hi = global_ranges['pipe_diameter']
        tokens = rng.random(len(wn.pipe_name_list))
        new_values = compute_values_by_range(tokens, lo, hi)
        for p, val in zip(wn.pipe_name_list, new_values):
            wn.get_link(p).diameter = max(1e-12, float(val))

    if gen_pipe_length and 'pipe_length' in global_ranges:
        lo, hi = global_ranges['pipe_length']
        tokens = rng.random(len(wn.pipe_name_list))
        new_values = compute_values_by_range(tokens, lo, hi)
        for p, val in zip(wn.pipe_name_list, new_values):
            wn.get_link(p).length = max(1e-12, float(val))

    if gen_pipe_minorloss and 'pipe_minorloss' in global_ranges:
        lo, hi = global_ranges['pipe_minorloss']
        tokens = rng.random(len(wn.pipe_name_list))
        new_values = compute_values_by_range(tokens, lo, hi)
        for p, val in zip(wn.pipe_name_list, new_values):
            wn.get_link(p).minor_loss = max(0.0, float(val))

    # ========= RANDOMIZE PUMP PARAMETERS =========
    if len(wn.pump_name_list) > 0:
        if gen_pump_init_status:
            tokens = rng.random(len(wn.pump_name_list))
            statuses = compute_boolean_values(tokens, pump_open_prob)
            for p, status in zip(wn.pump_name_list, statuses):
                wn.get_link(p).initial_status = 'OPEN' if status > 0.5 else 'CLOSED'

        if gen_pump_speed and 'pump_speed' in global_ranges:
            lo, hi = global_ranges['pump_speed']
            tokens = rng.random(len(wn.pump_name_list))
            new_speeds = compute_values_by_range(tokens, lo, hi)
            for p, speed in zip(wn.pump_name_list, new_speeds):
                try:
                    wn.get_link(p).speed_timeseries.base_value = float(speed)
                except:
                    pass

    # ========= RANDOMIZE TANK PARAMETERS =========
    if len(wn.tank_name_list) > 0:
        if gen_tank_level and 'tank_level' in global_ranges:
            lo, hi = global_ranges['tank_level']
            tokens = rng.random(len(wn.tank_name_list))
            new_levels = compute_values_by_range(tokens, lo, hi)
            for t, level in zip(wn.tank_name_list, new_levels):
                tank = wn.get_node(t)
                wn.get_node(t).init_level = float(np.clip(level, tank.min_level,
                                                          tank.max_level))

        if gen_tank_elevation and 'tank_elevation' in global_ranges:
            lo, hi = global_ranges['tank_elevation']
            tokens = rng.random(len(wn.tank_name_list))
            new_elevations = compute_values_by_range(tokens, lo, hi)
            for t, ele in zip(wn.tank_name_list, new_elevations):
                wn.get_node(t).elevation = float(ele)

        if gen_tank_diameter and 'tank_diameter' in global_ranges:
            lo, hi = global_ranges['tank_diameter']
            tokens = rng.random(len(wn.tank_name_list))
            new_diameters = compute_values_by_range(tokens, lo, hi)
            for t, dia in zip(wn.tank_name_list, new_diameters):
                wn.get_node(t).diameter = max(1e-12, float(dia))

    # ========= RANDOMIZE VALVE PARAMETERS =========
    if len(wn.valve_name_list) > 0:
        if gen_valve_init_status:
            tokens = rng.random(len(wn.valve_name_list))
            statuses = compute_boolean_values(tokens, valve_open_prob)
            for v, status in zip(wn.valve_name_list, statuses):
                wn.get_link(v).initial_status = 'OPEN' if status > 0.5 else 'CLOSED'

    # ========= RANDOMIZE RESERVOIR PARAMETERS =========
    if len(wn.reservoir_name_list) > 0 and gen_res_total_head:
        lo, hi = global_ranges.get('reservoir_head', (res_head_lo, res_head_hi))
        tokens = rng.random(len(wn.reservoir_name_list))
        random_pressures = compute_values_by_range(tokens, lo, hi)

        for r, pressure in zip(wn.reservoir_name_list, random_pressures):
            res = wn.get_node(r)
            if update_totalhead_method == "add_max_elevation":
                # Add max junction elevation (matching Executorv7.py:308-311)
                elevations = [wn.get_node(j).elevation for j in wn.junction_name_list]
                max_ele = max(elevations)
                res.head_timeseries.base_value = float(max_ele + pressure)
            else:
                res.head_timeseries.base_value = float(pressure)

    # ========= RUN SIMULATION =========
    try:
        ht = wn.options.time.hydraulic_timestep
        wn.options.time.duration = ht
        wn.options.time.report_timestep = ht

        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()

        # Validation (matching Executorv7.py:368-391)
        pressures = results.node["pressure"].iloc[-1]

        if skip_reservoir_result:
            # Remove reservoir pressures from validation
            pressures = pressures.drop(wn.reservoir_name_list, errors='ignore')

        if (pressures.min() < pressure_ok_lo) or (pressures.max() > pressure_ok_hi):
            continue

        # Check for NaN
        if pressures.isna().any():
            continue

        # Build output data
        Gd = wn.to_graph()
        G = nx.MultiGraph(Gd)

        heads = results.node["head"].iloc[-1]
        node_data = pd.DataFrame({"pressure": pressures, "head": heads})

        # Realized demands
        realized = {}
        for j in wn.junction_name_list:
            jn = wn.get_node(j)
            val = sum(float(ts.base_value) for ts in jn.demand_timeseries_list)
            realized[j] = val
        node_data["demand_t0"] = pd.Series(realized)

        # Link data
        link_cols = {}
        for key in ("flowrate", "velocity"):
            if key in results.link:
                link_cols[key] = results.link[key].iloc[-1]
        link_data = pd.DataFrame(link_cols) if link_cols else pd.DataFrame(
            index=wn.link_name_list)

        # Save
        out_path = os.path.join(out_dir, f"data_{saved}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump((G, link_data, node_data), f)

        saved += 1
        print(f"✓ Saved {saved}/{num_events} (attempt {attempts})")

    except Exception as e:
        if attempts % 10 == 0:
            print(f"  Attempt {attempts}: Simulation failed - {e}")
        continue

print(f"\n{'='*60}")
print(f"Complete! Generated {num_events} valid scenarios in {attempts} attempts.")
print(f"Success rate: {100 * num_events / attempts:.1f}%")
print(f"{'='*60}")
