#%%
import os
import math
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import wntr
import networkx as nx

# ========= User settings =========
file_path   = "water_pressure_simulation/ctown.inp"
out_dir     = "Water_quality_data_new/ctown_data"
num_events  = 3               # how many valid PKLs to produce
seed        = 42

# DEMAND SETTINGS (matching GNN approach)
demand_quantile_lo = 0        # Lower quantile (0-100) or use None for absolute min
demand_quantile_hi = 60       # Upper quantile (0-100) or use None for absolute max
use_quantiles = True          # True: use quantiles, False: use absolute min/max

pump_open_prob  = 0.8          # pump OPEN probability
valve_open_prob = 0.8          # valve OPEN probability
pump_speed_lo, pump_speed_hi = 0.8, 1.2   # pump speed range
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

def _valve_setting_bounds_if_available(v) -> Tuple[float, float] | None:
    """Return (min,max) if the valve exposes explicit bounds, else None (we'll skip setting)."""
    for lo_attr, hi_attr in (("min_setting", "max_setting"), ("setting_min", "setting_max")):
        if hasattr(v, lo_attr) and hasattr(v, hi_attr):
            try:
                lo = float(getattr(v, lo_attr))
                hi = float(getattr(v, hi_attr))
                if hi >= lo:
                    return lo, hi
            except Exception:
                pass
    return None

# ----- Compute SINGLE GLOBAL demand range from pristine network -----
base_wn = wntr.network.WaterNetworkModel(file_path)
global_demand_lo, global_demand_hi = _compute_global_demand_range(base_wn)

# ----- generate until we have num_events valid files -----
saved = 0
attempts = 0
while saved < num_events:
    attempts += 1
    wn = wntr.network.WaterNetworkModel(file_path)

    # 1) Junction demand: EACH junction randomized within SAME GLOBAL [min,max]
    #    This matches TokenGeneratorByRange.py:275-287 where:
    #        new_values = range_lo + tokens * (range_hi - range_lo)
    #    All junctions use the SAME range_lo and range_hi

    for j in wn.junction_name_list:
        jn = wn.get_node(j)
        bases = [float(ts.base_value) for ts in jn.demand_timeseries_list]

        if len(bases) == 0 or sum(bases) <= 0:
            continue

        # Random target within GLOBAL range (not per-junction range!)
        if np.isclose(global_demand_lo, global_demand_hi):
            target = global_demand_lo
        else:
            target = float(rng.uniform(global_demand_lo, global_demand_hi))
        target = max(0.0, target)

        # Get first pattern multipliers (for single-snapshot)
        m = np.array([_first_multiplier(ts) for ts in jn.demand_timeseries_list], dtype=float)
        m_safe = np.where(m <= 0.0, 1.0, m)

        # Proportional weights
        w = np.array(bases, dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            continue

        # Distribute target demand proportionally: sum(new_base_i * m_i) = target
        new_bases = (target * (w / m_safe)) / w_sum
        for ts, nb in zip(jn.demand_timeseries_list, new_bases):
            ts.base_value = float(max(0.0, nb))

    # 2) Tank levels in [min_level, max_level]
    for tname in wn.tank_name_list:
        t = wn.get_node(tname)
        lo, hi = float(t.min_level), float(t.max_level)
        if hi > lo:
            t.init_level = float(rng.uniform(lo, hi))

    # 3) Reservoir head × [0.5, 2.0]
    for rname in wn.reservoir_name_list:
        r = wn.get_node(rname)
        base_head = float(r.head_timeseries.base_value)
        r.head_timeseries.base_value = base_head * float(rng.uniform(reservoir_scale_lo, reservoir_scale_hi))

    # 4) Pumps: status (p=0.8 OPEN), speed ∈ [0.8,1.2]
    for pname in wn.pump_name_list:
        p = wn.get_link(pname)
        p.initial_status = 'OPEN' if rng.random() < pump_open_prob else 'CLOSED'
        try:
            p.speed = float(rng.uniform(pump_speed_lo, pump_speed_hi))
        except Exception:
            pass

    # 5) Valves: status (p=0.8 OPEN), setting within explicit [min,max] if available
    for vname in wn.valve_name_list:
        v = wn.get_link(vname)
        v.initial_status = 'OPEN' if rng.random() < valve_open_prob else 'CLOSED'
        bounds = _valve_setting_bounds_if_available(v)
        if bounds is not None and hasattr(v, "setting"):
            lo, hi = bounds
            try:
                v.setting = float(rng.uniform(lo, hi))
            except Exception:
                pass

    # 6) One-step hydraulic simulation
    ht = wn.options.time.hydraulic_timestep
    wn.options.time.duration = ht
    wn.options.time.report_timestep = ht

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    # 7) Pressure filter: discard if any pressure outside [0,151]
    pressures = results.node["pressure"].iloc[-1]
    if (pressures.min() < pressure_ok_lo) or (pressures.max() > pressure_ok_hi):
        continue  # abandon this trial

    # 8) Build UNDIRECTED graph + dataframes and save
    Gd = wn.to_graph()      # directed MultiDiGraph
    G = nx.MultiGraph(Gd)   # Convert to UNDIRECTED MultiGraph

    heads = results.node["head"].iloc[-1]
    node_data = pd.DataFrame({"pressure": pressures, "head": heads})

    # Realized demand at t0 (base * first_multiplier per category)
    realized = {}
    for j in wn.junction_name_list:
        jn = wn.get_node(j)
        val = 0.0
        for ts in jn.demand_timeseries_list:
            val += float(ts.base_value) * _first_multiplier(ts)
        realized[j] = val
    node_data["demand_t0"] = pd.Series(realized)

    link_cols = {}
    for key in ("flowrate", "velocity"):
        if key in results.link:
            link_cols[key] = results.link[key].iloc[-1]
    link_data = pd.DataFrame(link_cols) if link_cols else pd.DataFrame(index=wn.link_name_list)

    out_path = os.path.join(out_dir, f"data_{saved}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((G, link_data, node_data), f)

    saved += 1
    print(f"Saved {saved}/{num_events} (attempt {attempts})")

print(f"\nComplete! Generated {num_events} valid scenarios in {attempts} attempts.")
