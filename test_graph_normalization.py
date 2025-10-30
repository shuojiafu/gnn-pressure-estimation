#!/usr/bin/env python3
"""
Test script for graph normalization functionality.
"""

import networkx as nx
from gnn_pressure_estimation.utils.graph_normalization import convert_wntr_graph_to_undirected


def test_graph_normalization():
    """Test that directed graph is properly converted to undirected with correct edge types."""

    # Create a simple directed graph with different link types
    G_directed = nx.DiGraph()

    # Add nodes
    G_directed.add_node('J1')
    G_directed.add_node('J2')
    G_directed.add_node('J3')
    G_directed.add_node('J4')

    # Add edges with different types
    # Pipe: J1 -> J2 (bidirectional, should stay as Pipe)
    G_directed.add_edge('J1', 'J2', link_type='Pipe', link_id='P1', weight=1.0)

    # Pump: J2 -> J3 (unidirectional, should stay as Pump)
    G_directed.add_edge('J2', 'J3', link_type='Pump', link_id='PU1', weight=1.0)

    # Valve: J3 -> J4 (unidirectional, should stay as Valve)
    G_directed.add_edge('J3', 'J4', link_type='Valve', link_id='V1', weight=1.0)

    print("Directed Graph:")
    print(f"  Nodes: {list(G_directed.nodes())}")
    print(f"  Edges: {list(G_directed.edges(data=True))}")
    print()

    # Convert to undirected
    G_undirected = convert_wntr_graph_to_undirected(G_directed)

    print("Undirected Graph:")
    print(f"  Nodes: {list(G_undirected.nodes())}")
    print(f"  Edges:")
    for u, v, data in G_undirected.edges(data=True):
        edge_type = data.get('edge_type', 'Unknown')
        link_id = data.get('link_id', 'Unknown')
        print(f"    {u} - {v}: type={edge_type}, id={link_id}")
    print()

    # Verify edge types
    assert G_undirected.has_edge('J1', 'J2'), "Pipe edge missing"
    assert G_undirected.has_edge('J2', 'J3'), "Pump edge missing"
    assert G_undirected.has_edge('J3', 'J4'), "Valve edge missing"

    # Check edge types
    edge_12 = G_undirected.get_edge_data('J1', 'J2')
    edge_23 = G_undirected.get_edge_data('J2', 'J3')
    edge_34 = G_undirected.get_edge_data('J3', 'J4')

    assert edge_12['edge_type'] == 'Pipe', f"Expected Pipe, got {edge_12['edge_type']}"
    assert edge_23['edge_type'] == 'Pump', f"Expected Pump, got {edge_23['edge_type']}"
    assert edge_34['edge_type'] == 'Valve', f"Expected Valve, got {edge_34['edge_type']}"

    print("✓ All tests passed!")
    print("\nSummary:")
    print("  - Pipes maintain their type as 'Pipe'")
    print("  - Pumps are labeled as 'Pump' (or 'Reverse Pump' if flow is reversed)")
    print("  - Valves are labeled as 'Valve' (or 'Reverse Valve' if flow is reversed)")


if __name__ == '__main__':
    test_graph_normalization()
