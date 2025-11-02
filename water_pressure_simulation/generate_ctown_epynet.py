#%%
import os
import math
import pickle
from typing import Dict, Tuple
from epynet import Network
from epynet import epanet2
import numpy as np
import pandas as pd
import networkx as nx

# ========= User settings =========
file_path   = "water_pressure_simulation/ctown.inp"  # Path to your input file
out_dir     = "water_pressure_simulation/ctown_data"  # Output directory

# Print paths for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for input file at: {file_path}")
print(f"Output directory: {out_dir}")

# Check if input file exists
if not os.path.exists(file_path):
    print(f"\nERROR: Input file not found at: {file_path}")
    print(f"Please make sure the path is correct relative to: {os.getcwd()}")
    raise FileNotFoundError(f"Cannot find input file: {file_path}")

num_events  =  1             # how many valid PKLs to produce
seed        = 42

# DEMAND SETTINGS (matching GNN approach)
demand_quantile_lo = 0        # Lower quantile (0-100) or use None for absolute min
demand_quantile_hi = 60       # Upper quantile (0-100) or use None for absolute max
use_quantiles = False          # True: use quantiles, False: use absolute min/max

pump_open_prob  = 0.8          # pump OPEN probability
valve_open_prob = 0.8          # valve OPEN probability
pump_speed_lo, pump_speed_hi = 0.8, 1.2   # pump speed range
reservoir_scale_lo, reservoir_scale_hi = 0.5, 2.0
pressure_ok_lo, pressure_ok_hi = 0.0, 151.0  # discard trials if outside
# =================================

os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

# ----- small helpers -----
def _set_epanet_value(obj, param_code, value):
    """
    Set EPANET parameter value directly via low-level API.
    This bypasses read-only property restrictions in EPyNet.
    Based on epynet_utils.set_object_value_wo_ierror

    Args:
        obj: EPyNet Node or Link object
        param_code: EPANET parameter code (e.g., epanet2.EN_TANKLEVEL)
        value: Value to set
    """
    from epynet import Node, Link

    wn = obj.network()
    wn.solved = False
    obj._values[param_code] = value

    # Call the appropriate EPANET C library function
    if isinstance(obj, Node):
        ierr = wn.ep._lib.EN_setnodevalue(
            wn.ep.ph,
            epanet2.ctypes.c_int(obj.index),
            epanet2.ctypes.c_int(param_code),
            epanet2.ctypes.c_float(value)
        )
    else:  # Link
        ierr = wn.ep._lib.EN_setlinkvalue(
            wn.ep.ph,
            epanet2.ctypes.c_int(obj.index),
            epanet2.ctypes.c_int(param_code),
            epanet2.ctypes.c_float(value)
        )

    if ierr != 0:
        raise Exception(f"EPANET error {ierr} setting {obj.uid} param {param_code} to {value}")

def _first_multiplier(pattern) -> float:
    """Get first pattern multiplier (for single-snapshot simulation)."""
    if pattern is None:
        return 1.0
    try:
        values = pattern.values
        if values is None or len(values) == 0:
            return 1.0
        return float(values[0])
    except:
        return 1.0

def _compute_global_demand_range(wn: Network) -> Tuple[float, float]:
    """
    Compute GLOBAL demand range across ALL junctions (matching GNN approach).

    This mimics ConfigCreator.py:127-128 where:
        base_demands = wn.junctions.basedemand.to_numpy()  # ALL junctions
        demand_lo, demand_hi = get_range(base_demands, lo, hi, is_quantile)

    Returns:
        (demand_lo, demand_hi): Single global range used for ALL junctions
    """
    all_base_demands = []

    for junc in wn.junctions:
        base_demand = float(junc.basedemand)
        if base_demand > 0:
            all_base_demands.append(base_demand)

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

def _valve_setting_bounds_if_available(v) -> Tuple[float, float] | None:
    """Return (min,max) if the valve exposes explicit bounds, else None (we'll skip setting)."""
    try:
        current_setting = float(v.setting)
        if current_setting > 0:
            # Return a range around current setting
            return (current_setting * 0.5, current_setting * 1.5)
    except:
        pass
    return None

def _build_directed_graph(wn: Network, include_reservoir: bool = True) -> nx.MultiDiGraph:
    """Build a directed MultiDiGraph from EPyNet water network."""
    G = nx.MultiDiGraph()

    # Add nodes
    node_list = []
    collection = wn.junctions if not include_reservoir else wn.nodes
    for node in collection:
        node_list.append(node.uid)

    # Add edges from pipes
    for pipe in wn.pipes:
        if (pipe.from_node.uid in node_list) and (pipe.to_node.uid in node_list):
            G.add_edge(pipe.from_node.uid, pipe.to_node.uid, weight=1., length=pipe.length)

    # Add edges from pumps
    for pump in wn.pumps:
        if (pump.from_node.uid in node_list) and (pump.to_node.uid in node_list):
            G.add_edge(pump.from_node.uid, pump.to_node.uid, weight=1., length=0.)

    # Add edges from valves
    for valve in wn.valves:
        if (valve.from_node.uid in node_list) and (valve.to_node.uid in node_list):
            G.add_edge(valve.from_node.uid, valve.to_node.uid, weight=1., length=0.)

    return G

# ----- Compute SINGLE GLOBAL demand range from pristine network -----
base_wn = Network(file_path)
global_demand_lo, global_demand_hi = _compute_global_demand_range(base_wn)
# Close the base network to free up file handles
try:
    base_wn.ep.ENclose()
except:
    pass
del base_wn

# ----- generate until we have num_events valid files -----
saved = 0
attempts = 0
while saved < num_events:
    attempts += 1
    # Create unique report/binary files for each attempt to avoid file locking on Windows
    import tempfile
    temp_dir = tempfile.gettempdir()
    rpt_file = os.path.join(temp_dir, f"epynet_{os.getpid()}_{attempts}.rpt")
    bin_file = os.path.join(temp_dir, f"epynet_{os.getpid()}_{attempts}.bin")

    wn = Network(file_path)
    # Override the default report/binary file paths
    wn.rptfile = rpt_file
    wn.binfile = bin_file

    # 1) Junction demand: EACH junction randomized within SAME GLOBAL [min,max]
    #    This matches TokenGeneratorByRange.py:275-287 where:
    #        new_values = range_lo + tokens * (range_hi - range_lo)
    #    All junctions use the SAME range_lo and range_hi

    for junc in wn.junctions:
        base_demand = float(junc.basedemand)

        if base_demand <= 0:
            continue

        # Random target within GLOBAL range (not per-junction range!)
        if np.isclose(global_demand_lo, global_demand_hi):
            target = global_demand_lo
        else:
            target = float(rng.uniform(global_demand_lo, global_demand_hi))
        target = max(0.0, target)

        # Get pattern multiplier
        pattern = junc.pattern if hasattr(junc, 'pattern') else None
        m = _first_multiplier(pattern)
        m_safe = 1.0 if m <= 0.0 else m

        # Set new base demand: target = new_base * m
        new_base = target / m_safe
        junc.basedemand = float(max(0.0, new_base))

    # 2) Tank levels in [min_level, max_level]
    # Based on Executorv7.py:238
    for tank in wn.tanks:
        lo, hi = float(tank.minlevel), float(tank.maxlevel)
        if hi > lo:
            tank_level = float(rng.uniform(lo, hi))
            _set_epanet_value(tank, epanet2.EN_TANKLEVEL, tank_level)

    # 3) Reservoir head × [0.5, 2.0]
    # Based on Executorv7.py:303-315 - they use patterns for reservoirs
    for res in wn.reservoirs:
        try:
            current_head = float(res.head) if hasattr(res, 'head') else float(res.elevation)
            new_head = current_head * float(rng.uniform(reservoir_scale_lo, reservoir_scale_hi))
            # Use set_object_value method like in Executorv7.py line 304
            res.set_object_value(epanet2.EN_ELEVATION, new_head)
        except Exception as e:
            print(f"Warning: Could not set reservoir head: {e}")

    # 4) Pumps: status (p=0.8 OPEN), speed ∈ [0.8,1.2]
    # Based on Executorv7.py:225 and 230 - direct assignment works!
    for pump in wn.pumps:
        # Set initial status - direct assignment (Executorv7.py:225)
        pump.initstatus = 1 if rng.random() < pump_open_prob else 0

        # Set pump speed - direct assignment (Executorv7.py:230)
        pump.speed = float(rng.uniform(pump_speed_lo, pump_speed_hi))

    # 5) Valves: status (p=0.8 OPEN), setting within explicit [min,max] if available
    # Based on Executorv7.py:249 and 272
    for valve in wn.valves:
        # Set initial status - direct assignment (Executorv7.py:249)
        valve.initstatus = 1 if rng.random() < valve_open_prob else 0

        # Set valve setting - needs low-level API (Executorv7.py:272)
        bounds = _valve_setting_bounds_if_available(valve)
        if bounds is not None:
            lo, hi = bounds
            try:
                new_setting = float(rng.uniform(lo, hi))
                _set_epanet_value(valve, epanet2.EN_INITSETTING, new_setting)
            except Exception:
                pass

    # 6) One-step hydraulic simulation (matching GNN approach - Executorv7.py:193-199)
    # Set all time parameters to 1 second for single-snapshot steady-state
    wn.ep.ENsettimeparam(epanet2.EN_DURATION, 1)
    wn.ep.ENsettimeparam(epanet2.EN_HYDSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_QUALSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTART, 1)
    wn.ep.ENsettimeparam(epanet2.EN_REPORTSTEP, 1)
    wn.ep.ENsettimeparam(epanet2.EN_REPORTSTART, 1)
    wn.ep.ENsettimeparam(epanet2.EN_RULESTEP, 1)

    # Run simulation
    try:
        wn.solve()
    except Exception as e:
        print(f"Simulation failed: {e}")
        # Close and cleanup before abandoning
        try:
            wn.ep.ENclose()
        except:
            pass
        try:
            if os.path.exists(rpt_file):
                os.remove(rpt_file)
            if os.path.exists(bin_file):
                os.remove(bin_file)
        except:
            pass
        continue  # abandon this trial

    # 7) Pressure filter: discard if any pressure outside [0,151]
    pressures = wn.nodes.pressure.values
    if (pressures.min() < pressure_ok_lo) or (pressures.max() > pressure_ok_hi):
        # Close and cleanup before abandoning
        try:
            wn.ep.ENclose()
        except:
            pass
        try:
            if os.path.exists(rpt_file):
                os.remove(rpt_file)
            if os.path.exists(bin_file):
                os.remove(bin_file)
        except:
            pass
        continue  # abandon this trial

    # 8) Build DIRECTED graph + dataframes and save
    G = _build_directed_graph(wn, include_reservoir=True)

    # Node data
    heads = wn.nodes.head.values
    node_ids = wn.nodes.uid.tolist()
    node_data = pd.DataFrame({
        "pressure": wn.nodes.pressure.values,
        "head": heads
    }, index=node_ids)

    # Realized demand at t0 (base * first_multiplier)
    realized = {}
    for junc in wn.junctions:
        base_demand = float(junc.basedemand)
        pattern = junc.pattern if hasattr(junc, 'pattern') else None
        multiplier = _first_multiplier(pattern)
        realized[junc.uid] = base_demand * multiplier
    node_data["demand_t0"] = pd.Series(realized)

    # Elevations
    elevations = {}
    for node in wn.nodes:
        if hasattr(node, 'elevation'):
            elevations[node.uid] = node.elevation
        else:
            # For reservoirs, use head as elevation
            elevations[node.uid] = node.head if hasattr(node, 'head') else 0.0
    node_data["elevation"] = pd.Series(elevations)

    # Link simulation outputs
    link_ids = wn.links.uid.tolist()
    link_data = pd.DataFrame({
        "flowrate": wn.links.flow.values,
        "velocity": wn.links.velocity.values
    }, index=link_ids)

    # Add link input features (parameters that were randomized)
    link_params = {
        'diameter': {},
        'length': {},
        'roughness': {},
        'status': {},
        'link_type': {}
    }

    # Pipes
    for pipe in wn.pipes:
        link_params['diameter'][pipe.uid] = pipe.diameter
        link_params['length'][pipe.uid] = pipe.length
        link_params['roughness'][pipe.uid] = pipe.roughness
        link_params['status'][pipe.uid] = str(pipe.initstatus)
        link_params['link_type'][pipe.uid] = 'PIPE'

    # Pumps
    for pump in wn.pumps:
        link_params['diameter'][pump.uid] = None  # Pumps don't have diameter
        link_params['length'][pump.uid] = pump.length if hasattr(pump, 'length') else None
        link_params['roughness'][pump.uid] = None
        link_params['status'][pump.uid] = str(pump.initstatus)
        link_params['link_type'][pump.uid] = 'PUMP'
        # Add pump-specific parameter
        try:
            link_data.loc[pump.uid, 'pump_speed'] = pump.speed
        except:
            link_data.loc[pump.uid, 'pump_speed'] = None

    # Valves
    for valve in wn.valves:
        try:
            link_params['diameter'][valve.uid] = valve.diameter
        except:
            link_params['diameter'][valve.uid] = None
        link_params['length'][valve.uid] = None
        link_params['roughness'][valve.uid] = None
        link_params['status'][valve.uid] = str(valve.initstatus)
        link_params['link_type'][valve.uid] = f'VALVE_{valve.valve_type}'
        # Add valve-specific parameters
        if hasattr(valve, 'setting'):
            link_data.loc[valve.uid, 'valve_setting'] = valve.setting
        else:
            link_data.loc[valve.uid, 'valve_setting'] = None

    # Add all link parameters to link_data
    for param_name, param_dict in link_params.items():
        link_data[param_name] = pd.Series(param_dict)

    out_path = os.path.join(out_dir, f"data_{saved}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((G, link_data, node_data), f)

    saved += 1
    print(f"Saved {saved}/{num_events} (attempt {attempts})")

    # Close network and clean up temp files
    try:
        wn.ep.ENclose()
    except:
        pass

    # Clean up temporary report and binary files
    try:
        if os.path.exists(rpt_file):
            os.remove(rpt_file)
        if os.path.exists(bin_file):
            os.remove(bin_file)
    except:
        pass

print(f"\nComplete! Generated {num_events} valid scenarios in {attempts} attempts.")
