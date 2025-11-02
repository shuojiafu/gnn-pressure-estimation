# WNTR to EPyNet Migration Summary

## Overview
This document summarizes the migration of the water pressure simulation script from WNTR to EPyNet.

**Primary Reason for Migration**: WNTR doesn't allow customized pump speeds, while EPyNet does.

## Files Created
- `generate_ctown_epynet.py` - EPyNet version of the simulation script

## Key API Changes

### 1. Network Loading
**WNTR:**
```python
wn = wntr.network.WaterNetworkModel(file_path)
```

**EPyNet:**
```python
wn = Network(file_path)
```

### 2. Demand Handling
**WNTR:**
```python
for j in wn.junction_name_list:
    jn = wn.get_node(j)
    for ts in jn.demand_timeseries_list:
        ts.base_value = new_value
```

**EPyNet:**
```python
for junc in wn.junctions:
    junc.basedemand = new_value
```

EPyNet has a simpler demand structure with a single `basedemand` per junction instead of WNTR's multiple demand timeseries.

### 3. Pattern Multipliers
**WNTR:**
```python
def _first_multiplier(ts) -> float:
    p = ts.pattern
    if (p is None) or (len(p.multipliers) == 0):
        return 1.0
    return float(p.multipliers[0])
```

**EPyNet:**
```python
def _first_multiplier(pattern) -> float:
    if pattern is None:
        return 1.0
    values = pattern.values
    if values is None or len(values) == 0:
        return 1.0
    return float(values[0])
```

### 4. Tank Level Setting
**WNTR:**
```python
t.init_level = float(rng.uniform(lo, hi))
```

**EPyNet:**
```python
eutils.set_object_value_wo_ierror(tank, epanet2.EN_TANKLEVEL, float(rng.uniform(lo, hi)))
```

EPyNet requires using the utility function to properly set tank levels without triggering errors.

### 5. Reservoir Head Setting
**WNTR:**
```python
r.head_timeseries.base_value = new_head
```

**EPyNet:**
```python
eutils.set_object_value_wo_ierror(res, epanet2.EN_ELEVATION, new_head)
```

### 6. Pump Speed Setting (KEY FEATURE!)
**WNTR:**
```python
# Limited or no support for custom pump speeds
p.speed = value  # May not work as expected
```

**EPyNet:**
```python
# Full support for custom pump speeds
eutils.set_object_value_wo_ierror(pump, epanet2.EN_PUMPSPEED, new_speed)
```

This is the primary reason for migration - EPyNet allows full customization of pump speeds.

### 7. Pump Status Setting
**WNTR:**
```python
p.initial_status = 'OPEN'  # or 'CLOSED'
```

**EPyNet:**
```python
eutils.set_object_value_wo_ierror(pump, epanet2.EN_INITSTATUS, 1)  # 1=OPEN, 0=CLOSED
```

### 8. Valve Settings
**WNTR:**
```python
v.initial_status = 'OPEN'
v.setting = new_setting
```

**EPyNet:**
```python
eutils.set_object_value_wo_ierror(valve, epanet2.EN_INITSTATUS, 1)
eutils.set_object_value_wo_ierror(valve, epanet2.EN_INITSETTING, new_setting)
```

### 9. Simulation Time Parameters
**WNTR:**
```python
wn.options.time.duration = 1
wn.options.time.hydraulic_timestep = 1
wn.options.time.quality_timestep = 1
# ... etc
```

**EPyNet:**
```python
wn.ep.ENsettimeparam(epanet2.EN_DURATION, 1)
wn.ep.ENsettimeparam(epanet2.EN_HYDSTEP, 1)
wn.ep.ENsettimeparam(epanet2.EN_QUALSTEP, 1)
# ... etc
```

### 10. Running Simulation
**WNTR:**
```python
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()
```

**EPyNet:**
```python
wn.solve()
```

EPyNet has a much simpler simulation interface.

### 11. Accessing Results
**WNTR:**
```python
pressures = results.node["pressure"].iloc[-1]
heads = results.node["head"].iloc[-1]
flowrates = results.link["flowrate"].iloc[-1]
```

**EPyNet:**
```python
pressures = wn.nodes.pressure.values
heads = wn.nodes.head.values
flowrates = wn.links.flow.values
```

EPyNet provides direct access to results without needing to index time series.

### 12. Graph Generation
**WNTR:**
```python
G = wn.to_graph()
```

**EPyNet:**
```python
G = eutils.get_networkx_graph(wn, include_reservoir=True, graph_type='multi_directed')
```

EPyNet requires using the custom utility function from `epynet_utils.py`.

### 13. Node/Link Iteration
**WNTR:**
```python
for j in wn.junction_name_list:
    jn = wn.get_node(j)

for pname in wn.pump_name_list:
    p = wn.get_link(pname)
```

**EPyNet:**
```python
for junc in wn.junctions:
    # direct iteration

for pump in wn.pumps:
    # direct iteration
```

EPyNet provides direct iteration over collections without needing name lists.

### 14. Elevation Access
**WNTR:**
```python
for node_name in wn.node_name_list:
    node = wn.get_node(node_name)
    if hasattr(node, 'elevation'):
        elevations[node_name] = node.elevation
    else:
        elevations[node_name] = node.head_timeseries.base_value
```

**EPyNet:**
```python
for node in wn.nodes:
    if hasattr(node, 'elevation'):
        elevations[node.uid] = node.elevation
    else:
        elevations[node.uid] = node.head if hasattr(node, 'head') else 0.0
```

### 15. Link Properties
**WNTR:**
```python
link_data.loc[pname, 'pump_speed'] = pump.speed_timeseries.base_value
```

**EPyNet:**
```python
link_data.loc[pump.uid, 'pump_speed'] = pump.speed
```

EPyNet uses direct property access instead of timeseries objects.

## Settings Preserved
All original settings remain unchanged:
- Demand range computation (global approach)
- Tank level randomization
- Reservoir scaling
- Pump open probability (0.8)
- Valve open probability (0.8)
- Pump speed range (0.8 to 1.2)
- Pressure validation range (0.0 to 151.0)
- Random seed (42)
- Number of events (1)

## Benefits of EPyNet
1. **Custom pump speeds** - Full control over pump speed settings
2. **Simpler API** - Direct property access and iteration
3. **Better performance** - Direct EPANET API calls
4. **Consistent with existing codebase** - The project already uses EPyNet extensively

## Usage
Run the migrated script:
```bash
cd /home/user/gnn-pressure-estimation
python water_pressure_simulation/generate_ctown_epynet.py
```

The script will generate pickle files in `water_pressure_simulation/ctown_data/` with the same structure as the WNTR version.

## Data Structure
Each pickle file contains a tuple `(G, link_data, node_data)`:
- `G`: NetworkX MultiDiGraph
- `link_data`: DataFrame with columns: flowrate, velocity, diameter, length, roughness, status, link_type, pump_speed (for pumps), valve_setting (for valves)
- `node_data`: DataFrame with columns: pressure, head, demand_t0, elevation
