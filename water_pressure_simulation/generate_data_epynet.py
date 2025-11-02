import os
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if not os.path.exists('water_pressure_simulation/ctown_data'):
    os.makedirs('water_pressure_simulation/ctown_data')
    print(f"Folder created successfully.")
else:
    print(f"Folder already exists.")

import math
import pickle
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import wntr
import networkx as nx
from epynet import Network
from epynet import epanet2
from gnn_pressure_estimation.generator.EPYNET import epynet_utils as eutils

# ========= User settings =========
file_path = "water_pressure_simulation/ctown.inp"
out_dir = "water_pressure_simulation/ctown_data"
num_events = 1  # how many valid PKLs to produce
seed = 42

# DEMAND SETTINGS (matching GNN approach)
demand_quantile_lo = 0  # Lower quantile (0-100) or use None for absolute min
demand_quantile_hi = 60  # Upper quantile (0-100) or use None for absolute max
use_quantiles = False  # True: use quantiles, False: use absolute min/max

pump_open_prob = 0.8  # pump OPEN probability
valve_open_prob = 0.8  # valve OPEN probability
pump_speed_multiplier_lo, pump_speed_multiplier_hi = 0.8, 1.2  # pump speed multiplier range
reservoir_scale_lo, reservoir_scale_hi = 0.5, 2.0
pressure_ok_lo, pressure_ok_hi = 0.0, 151.0  # discard trials if outside
# =================================

os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()


# ----- small helpers -----
def _first_multiplier(ts) -> float:
    """Get first pattern multiplier (for single-snapshot simulation)."""
    p = ts.pattern
    if (p is None) or (len(p.multipliers) == 0):
        return 1.0
    return float(p.multipliers[0])


def _compute_global_demand_range(wn: wntr.network.WaterNetworkModel) -> Tuple[float, float]:
    """
    Compute GLOBAL demand range across ALL junctions (matching GNN approach).

    This mimics ConfigCreator.py:127-128 where:
        base_demands = wn.junctions.basedemand.to_numpy()  # ALL junctions
        demand_lo, demand_hi = get_range(base_demands, lo, hi, is_quantile)

    Returns:
        (demand_lo, demand_hi): Single global range used for ALL junctions
    """
    all_base_demands = []

    for j in wn.junction_name_list:
        jn = wn.get_node(j)
        # Sum base demands across all demand categories (similar to epynet's single basedemand)
        total_base = sum(float(ts.base_value) for ts in jn.demand_timeseries_list)
        if total_base > 0:
            all_base_demands.append(total_base)

    if len(all_base_demands) == 0:
        return (0.0, 0.0)

    all_base_demands = np.array(all_base_demands)

    # Matching get_range() function from ConfigCreator.py:73-92
    if use_quantiles:
        demand_lo = float(np.percentile(all_base_demands, demand_quantile_lo))
        demand_hi = float(np.percentile(all_base_demands, demand_quantile_hi))
    else:
        demand_lo = float(all_base_demands.min())
        demand_hi = float(all_base_demands.max())

    print(f"Global demand range: [{demand_lo:.4f}, {demand_hi:.4f}]")
    return (demand_lo, demand_hi)


def _get_valve_bounds_from_inp(wn_epynet, valve) -> Tuple[float, float] | None:
    """
    Get min/max bounds for valve setting from the INP file.

    For valves in EPANET, the setting interpretation depends on valve type:
    - PRV (Pressure Reducing Valve): setting is pressure (psi or m)
    - PSV (Pressure Sustaining Valve): setting is pressure (psi or m)
    - FCV (Flow Control Valve): setting is flow rate
    - TCV (Throttle Control Valve): setting is loss coefficient
    - PBV (Pressure Breaker Valve): setting is pressure (psi or m)
    - GPV (General Purpose Valve): setting is loss coefficient

    We'll read the original setting from the INP file and create a range around it.
    """
    try:
        valve_type = valve.valve_type
        original_setting = float(valve.setting)

        # Handle edge case: if original setting is 0 or very small, use default ranges
        if abs(original_setting) < 1e-6:
            if valve_type in ['PRV', 'PSV', 'PBV']:  # Pressure valves
                return (10.0, 100.0)  # Default pressure range
            elif valve_type == 'FCV':  # Flow Control Valve
                return (0.1, 10.0)  # Default flow range
            elif valve_type in ['TCV', 'GPV']:  # Loss coefficient valves
                return (0.0, 10.0)  # Default loss coefficient range
            else:
                return (0.1, 10.0)  # Default fallback

        # Define reasonable ranges based on valve type
        # We'll use a percentage of the original setting to create the range
        if valve_type in ['PRV', 'PSV', 'PBV']:  # Pressure valves
            # Pressure range: 50% to 150% of original setting
            min_val = max(0.1, original_setting * 0.5)
            max_val = max(min_val + 0.1, original_setting * 1.5)  # Ensure max > min
        elif valve_type == 'FCV':  # Flow Control Valve
            # Flow range: 50% to 150% of original setting
            min_val = max(0.001, original_setting * 0.5)
            max_val = max(min_val + 0.001, original_setting * 1.5)
        elif valve_type in ['TCV', 'GPV']:  # Loss coefficient valves
            # Loss coefficient range: use reasonable bounds
            min_val = max(0.0, original_setting * 0.5)
            max_val = max(min_val + 0.1, original_setting * 2.0)
        else:
            # Default fallback
            min_val = max(0.001, original_setting * 0.5)
            max_val = max(min_val + 0.1, original_setting * 1.5)

        # Final sanity check
        if max_val <= min_val:
            max_val = min_val + 1.0

        return (min_val, max_val)
    except Exception as e:
        print(f"Warning: Could not get valve bounds for {valve.uid}: {e}")
        return None


def solve_epynet(wn):
    """
    Run EPANET hydraulic simulation using epynet.
    Returns error code (0 = success, >0 = warning/error).
    """
    def ENrunH(ep):
        """Runs a single period hydraulic analysis."""
        ierr = ep._lib.EN_runH(ep.ph, epanet2.ctypes.byref(ep._current_simulation_time))
        return ierr

    wn.ep.ENopenH()
    wn.ep.ENinitH(0)
    code = ENrunH(wn.ep)
    wn.ep.ENcloseH()
    return code


# ----- Compute SINGLE GLOBAL demand range from pristine network -----
base_wn_wntr = wntr.network.WaterNetworkModel(file_path)
global_demand_lo, global_demand_hi = _compute_global_demand_range(base_wn_wntr)

# ----- generate until we have num_events valid files -----
saved = 0
attempts = 0
custom_base_index = 100

while saved < num_events:
    attempts += 1
    wn = Network(file_path)

    # Remove patterns, rules, and controls for steady-state simulation
    # (similar to Executorv7.py)
    try:
        patterns = wn.patterns
        if len(patterns) > 0:
            for p in patterns:
                try:
                    eutils.ENdeletepattern(wn, p.uid)
                except:
                    pass
    except:
        pass

    try:
        eutils.ENdeleteallrules(wn)
    except:
        pass

    try:
        eutils.ENdeleteallcontrols(wn)
    except:
        pass

    # Create custom patterns for junctions (for demand control)
    for i, _ in enumerate(wn.junctions):
        pattern_id = str(custom_base_index + i)
        if pattern_id not in [p.uid for p in wn.patterns]:
            wn.add_pattern(pattern_id, values=[1.0])

    # 1) Junction demand: EACH junction randomized within SAME GLOBAL [min,max]
    for i, junc in enumerate(wn.junctions):
        # Random target within GLOBAL range
        if np.isclose(global_demand_lo, global_demand_hi):
            target = global_demand_lo
        else:
            target = float(rng.uniform(global_demand_lo, global_demand_hi))
        target = max(0.0, target)

        # Set base demand and pattern
        junc.basedemand = 1.0
        pattern_id = str(custom_base_index + i)
        junc.pattern = pattern_id
        junc.pattern.values = [target]

        # Set pattern for all demand categories (EPANET 2.2 compatibility)
        eutils.ENsetdemandpatterntoallcategories(wn, junc.index, junc.basedemand, junc.pattern.index)

    # 2) Tank levels in [min_level, max_level]
    for tank in wn.tanks:
        min_level = float(tank.minlevel)
        max_level = float(tank.maxlevel)
        if max_level > min_level:
            new_level = float(rng.uniform(min_level, max_level))
            eutils.set_object_value_wo_ierror(tank, epanet2.EN_TANKLEVEL, new_level)

    # 3) Reservoir head × [0.5, 2.0]
    custom_res_pattern_base_index = custom_base_index + len(wn.junctions)
    for i, res in enumerate(wn.reservoirs):
        # Create pattern for reservoir
        pattern_id = str(custom_res_pattern_base_index + i)
        if pattern_id not in [p.uid for p in wn.patterns]:
            wn.add_pattern(pattern_id, values=[1.0])

        # Get base elevation
        base_head = float(res.elevation)
        new_head = base_head * float(rng.uniform(reservoir_scale_lo, reservoir_scale_hi))

        # Set elevation to 1.0 and use pattern to control head
        res.set_object_value(epanet2.EN_ELEVATION, 1.0)
        p_index = wn.ep.ENgetpatternindex(pattern_id)
        wn.ep.ENsetpattern(p_index, [new_head])
        res.set_object_value(epanet2.EN_PATTERN, p_index)

    # 4) Pumps: status (p=0.8 OPEN), speed multiplier ∈ [0.8,1.2]
    pump_speed_multipliers = {}
    for pump in wn.pumps:
        # Set status
        if rng.random() < pump_open_prob:
            pump.initstatus = 1  # OPEN
        else:
            pump.initstatus = 0  # CLOSED

        # Set speed multiplier (epynet allows direct pump.speed setting)
        speed_multiplier = float(rng.uniform(pump_speed_multiplier_lo, pump_speed_multiplier_hi))
        pump.speed = speed_multiplier
        pump_speed_multipliers[pump.uid] = speed_multiplier

    # 5) Valves: status (p=0.8 OPEN), and if OPEN, randomize setting within INP bounds
    valve_settings = {}

    # Build a temporary graph to check connectivity when closing valves
    tmp_graph = eutils.get_networkx_graph(wn=wn, include_reservoir=True, graph_type="undirected")

    for valve in wn.valves:
        # Determine if valve should be open or closed
        should_open = rng.random() < valve_open_prob

        if not should_open:
            # Try closing the valve - check if graph remains connected
            tmp_graph.remove_edge(valve.from_node.uid, valve.to_node.uid)
            if nx.is_connected(tmp_graph):
                valve.initstatus = 0  # CLOSED
                valve_settings[valve.uid] = None
            else:
                # Keep valve open to maintain connectivity
                tmp_graph.add_edge(valve.from_node.uid, valve.to_node.uid)
                valve.initstatus = 1  # OPEN
                should_open = True
        else:
            valve.initstatus = 1  # OPEN

        # If valve is open, randomize its setting within bounds from INP
        if should_open:
            bounds = _get_valve_bounds_from_inp(wn, valve)
            if bounds is not None:
                lo, hi = bounds
                try:
                    new_setting = float(rng.uniform(lo, hi))
                    eutils.set_object_value_wo_ierror(valve, epanet2.EN_INITSETTING, new_setting)
                    valve_settings[valve.uid] = new_setting
                except Exception as e:
                    print(f"Warning: Could not set valve setting for {valve.uid}: {e}")
                    valve_settings[valve.uid] = float(valve.setting)
            else:
                valve_settings[valve.uid] = float(valve.setting)

    # 6) Set simulation parameters for single-step simulation (matching Executorv7.py:193-199)
    wn.ep.ENsettimeparam(epanet2.EN_DURATION, 1)
    wn.ep.ENsettimeparam(epanet2.EN_QUALSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTART, 1)
    wn.ep.ENsettimeparam(epanet2.EN_REPORTSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_REPORTSTART, 1)
    wn.ep.ENsettimeparam(epanet2.EN_RULESTEP, 1)

    # 7) Run simulation
    try:
        code = solve_epynet(wn)
        if code > 6:  # Error codes > 6 are serious errors
            print(f"Simulation failed (attempt {attempts}) with error code {code}")
            continue
    except Exception as e:
        print(f"Simulation failed (attempt {attempts}): {e}")
        continue

    # 8) Extract results
    try:
        pressures = wn.nodes.pressure.values
        heads = wn.nodes.head.values
        node_ids = wn.nodes.uid.tolist()
    except Exception as e:
        print(f"Failed to extract node results (attempt {attempts}): {e}")
        continue

    # 9) Pressure filter: discard if any pressure outside [0,151]
    if (pressures.min() < pressure_ok_lo) or (pressures.max() > pressure_ok_hi):
        print(f"Pressure out of bounds (attempt {attempts}): [{pressures.min():.2f}, {pressures.max():.2f}]")
        continue

    # 10) Build DIRECTED graph + dataframes and save
    try:
        # Create graph using wntr for consistency with original format
        wn_wntr = wntr.network.WaterNetworkModel(file_path)
        G = wn_wntr.to_graph()  # directed MultiDiGraph

        # Build node dataframe
        node_data = pd.DataFrame({
            "pressure": pressures,
            "head": heads
        }, index=node_ids)

        # Add realized demand (base * pattern multiplier)
        realized = {}
        for i, junc in enumerate(wn.junctions):
            pattern_val = junc.pattern.values[0] if junc.pattern else 1.0
            realized[junc.uid] = float(junc.basedemand * pattern_val)
        node_data["demand_t0"] = pd.Series(realized)

        # Add elevations
        elevations = {}
        for node in wn.nodes:
            elevations[node.uid] = float(node.elevation)
        node_data["elevation"] = pd.Series(elevations)

        # Build link dataframe
        try:
            link_flows = wn.links.flow.values
            link_velocities = wn.links.velocity.values
            link_ids = wn.links.uid.tolist()

            link_data = pd.DataFrame({
                "flowrate": link_flows,
                "velocity": link_velocities
            }, index=link_ids)
        except:
            link_data = pd.DataFrame(index=[l.uid for l in wn.links])

        # Add link parameters
        link_params = {
            'diameter': {},
            'length': {},
            'roughness': {},
            'status': {},
            'link_type': {}
        }

        # Pipes
        for pipe in wn.pipes:
            link_params['diameter'][pipe.uid] = float(pipe.diameter)
            link_params['length'][pipe.uid] = float(pipe.length)
            link_params['roughness'][pipe.uid] = float(pipe.roughness)
            link_params['status'][pipe.uid] = 'OPEN' if pipe.initstatus else 'CLOSED'
            link_params['link_type'][pipe.uid] = 'PIPE'

        # Pumps
        for pump in wn.pumps:
            link_params['diameter'][pump.uid] = None
            link_params['length'][pump.uid] = None
            link_params['roughness'][pump.uid] = None
            link_params['status'][pump.uid] = 'OPEN' if pump.initstatus else 'CLOSED'
            link_params['link_type'][pump.uid] = 'PUMP'

            # Add pump speed
            if pump.uid in pump_speed_multipliers:
                link_data.loc[pump.uid, 'pump_speed'] = pump_speed_multipliers[pump.uid]
            else:
                link_data.loc[pump.uid, 'pump_speed'] = 1.0

        # Valves
        for valve in wn.valves:
            try:
                link_params['diameter'][valve.uid] = float(valve.diameter)
            except:
                link_params['diameter'][valve.uid] = None
            link_params['length'][valve.uid] = None
            link_params['roughness'][valve.uid] = None
            link_params['status'][valve.uid] = 'OPEN' if valve.initstatus else 'CLOSED'
            link_params['link_type'][valve.uid] = f'VALVE_{valve.valve_type}'

            # Add valve setting
            if valve.uid in valve_settings:
                link_data.loc[valve.uid, 'valve_setting'] = valve_settings[valve.uid]
            else:
                link_data.loc[valve.uid, 'valve_setting'] = float(valve.setting)

        # Add all link parameters to link_data
        for param_name, param_dict in link_params.items():
            link_data[param_name] = pd.Series(param_dict)

        # Save to pickle
        out_path = os.path.join(out_dir, f"data_{saved}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump((G, link_data, node_data), f)

        saved += 1
        print(f"Saved {saved}/{num_events} (attempt {attempts})")

    except Exception as e:
        print(f"Failed to build output data (attempt {attempts}): {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nComplete! Generated {num_events} valid scenarios in {attempts} attempts.")
