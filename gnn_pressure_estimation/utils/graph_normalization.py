"""
Graph normalization utilities for converting directed WDN graphs to undirected graphs
while preserving pump/valve direction information.
"""

import networkx as nx
from typing import Dict, Any


def normalize_directed_to_undirected(directed_graph: nx.DiGraph) -> nx.Graph:
    """
    Normalize a directed water distribution network graph to an undirected graph.

    For pipes: Convert normally (no direction matters)
    For pumps/valves: If the flow direction is reversed (outflow instead of inflow),
                     mark the edge as "reverse pump" or "reverse valve"

    Args:
        directed_graph: A directed NetworkX graph with edge attributes including 'link_type'

    Returns:
        An undirected NetworkX graph with proper edge type labeling
    """
    undirected_graph = nx.Graph()

    # Copy nodes from directed graph
    for node, attrs in directed_graph.nodes(data=True):
        undirected_graph.add_node(node, **attrs)

    # Process edges
    for u, v, edge_data in directed_graph.edges(data=True):
        # Get link type if available, default to 'Pipe'
        link_type = edge_data.get('link_type', 'Pipe')

        # Copy all edge attributes
        new_edge_data = edge_data.copy()

        # Check if this edge will be reversed in the undirected graph
        # In undirected graph, edge is stored with lower node ID first (by convention)
        # But we need to check if the edge already exists in opposite direction
        if undirected_graph.has_edge(v, u):
            # Edge already exists in opposite direction, skip
            continue

        # For pumps and valves, check if we need to mark as reverse
        if link_type in ['Pump', 'Valve', 'pump', 'valve']:
            # In directed graph: u -> v (from u to v)
            # If in undirected we store it as v-u, then it's reversed
            # For now, let's add a 'reversed' flag
            # We'll store the edge and mark it as reversed if the direction is flipped

            # Original direction: u -> v
            # We store in undirected as (u, v) and mark if it represents a reverse flow
            new_edge_data['original_source'] = u
            new_edge_data['original_target'] = v

            # Check if this pump/valve is in reverse direction based on typical flow
            # For pumps/valves, if the edge exists as v->u in directed but we're storing as u-v,
            # it means the pump/valve is reversed

            # Mark the link type with 'reverse' if needed
            # Note: This is a placeholder - in reality, you'd determine reversal based on
            # actual flow direction from simulation results
            # For now, we'll keep track of the original direction
            new_edge_data['link_type'] = link_type

        # Add edge to undirected graph
        undirected_graph.add_edge(u, v, **new_edge_data)

    return undirected_graph


def normalize_directed_to_undirected_with_flow(
    directed_graph: nx.DiGraph,
    flow_dict: Dict[tuple, float] = None
) -> nx.Graph:
    """
    Normalize a directed water distribution network graph to an undirected graph,
    using flow information to determine if pumps/valves are reversed.

    Args:
        directed_graph: A directed NetworkX graph with edge attributes including 'link_type'
        flow_dict: Dictionary mapping (source, target) tuples to flow values.
                  Negative flow indicates reverse direction.

    Returns:
        An undirected NetworkX graph with proper edge type labeling
    """
    undirected_graph = nx.Graph()

    # Copy nodes from directed graph
    for node, attrs in directed_graph.nodes(data=True):
        undirected_graph.add_node(node, **attrs)

    # Track processed edges to avoid duplicates
    processed_edges = set()

    # Process edges
    for u, v, edge_data in directed_graph.edges(data=True):
        # Skip if we already processed this edge pair
        edge_pair = tuple(sorted([u, v]))
        if edge_pair in processed_edges:
            continue
        processed_edges.add(edge_pair)

        # Get link type
        link_type = edge_data.get('link_type', 'Pipe')
        link_id = edge_data.get('link_id', f'{u}_{v}')

        # Copy all edge attributes
        new_edge_data = edge_data.copy()

        # For pipes, just copy the edge
        if link_type in ['Pipe', 'pipe']:
            new_edge_data['edge_type'] = 'Pipe'
            undirected_graph.add_edge(u, v, **new_edge_data)

        # For pumps and valves, check flow direction
        elif link_type in ['Pump', 'Valve', 'pump', 'valve']:
            # Original directed edge: u -> v (natural direction)
            is_reversed = False

            if flow_dict is not None:
                # Check flow direction
                flow = flow_dict.get((u, v), flow_dict.get((v, u), None))

                if flow is not None:
                    # If flow is negative, it means flow is opposite to edge direction
                    # If edge is u->v but flow is negative, actual flow is v->u (reversed)
                    if (u, v) in flow_dict and flow_dict[(u, v)] < 0:
                        is_reversed = True
                    elif (v, u) in flow_dict and flow_dict[(v, u)] > 0:
                        is_reversed = True

            # Set edge type based on reversal
            if is_reversed:
                new_edge_data['edge_type'] = f'Reverse {link_type}'
            else:
                new_edge_data['edge_type'] = link_type

            # Store original direction info
            new_edge_data['original_source'] = u
            new_edge_data['original_target'] = v

            undirected_graph.add_edge(u, v, **new_edge_data)

    return undirected_graph


def convert_wntr_graph_to_undirected(directed_graph: nx.DiGraph) -> nx.Graph:
    """
    Convert a WNTR directed graph to undirected, marking reversed pumps/valves.

    This function assumes the directed graph has link_type or type attributes on edges.

    Args:
        directed_graph: Directed graph from WNTR's to_graph() method

    Returns:
        Undirected graph with properly labeled edges
    """
    undirected_graph = nx.Graph()

    # Copy nodes
    for node, attrs in directed_graph.nodes(data=True):
        undirected_graph.add_node(node, **attrs)

    # Track processed edges
    processed_edges = set()

    # Process all directed edges
    for u, v, edge_data in directed_graph.edges(data=True):
        # Create a canonical edge representation
        edge_key = tuple(sorted([u, v]))

        if edge_key in processed_edges:
            continue

        processed_edges.add(edge_key)

        # Get link type from edge data
        link_type = edge_data.get('link_type', edge_data.get('type', 'Pipe'))

        # Copy edge attributes
        new_edge_data = edge_data.copy()

        # Determine if we need to mark as reverse
        # For directed edge u -> v:
        # - If it's a pump/valve, the natural flow direction is u -> v
        # - When we make it undirected, we preserve this information

        if link_type in ['Pump', 'pump']:
            # Check if there's a reverse edge v -> u in the directed graph
            if directed_graph.has_edge(v, u):
                # This shouldn't normally happen, but if it does, we have bidirectional pump
                # which typically means reverse flow
                new_edge_data['edge_type'] = 'Reverse Pump'
            else:
                # Normal pump direction u -> v
                new_edge_data['edge_type'] = 'Pump'

        elif link_type in ['Valve', 'valve']:
            # Similar logic for valves
            if directed_graph.has_edge(v, u):
                new_edge_data['edge_type'] = 'Reverse Valve'
            else:
                new_edge_data['edge_type'] = 'Valve'

        else:
            # Pipes are bidirectional, no need to mark as reverse
            new_edge_data['edge_type'] = 'Pipe'

        # Preserve original direction
        new_edge_data['directed_source'] = u
        new_edge_data['directed_target'] = v

        # Add the undirected edge
        undirected_graph.add_edge(u, v, **new_edge_data)

    return undirected_graph
