"""
Example: Converting an existing directed graph to undirected with 5 edge types

This script demonstrates how to use the WaterNetworkGraphConverter when you
already have a directed graph G.
"""

import networkx as nx
from gnn_pressure_estimation.utils.graph_converter import (
    WaterNetworkGraphConverter,
    convert_directed_to_undirected
)


# Example 1: Simple usage with existing directed graph
def example_basic():
    """Basic example with a directed graph."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage with Directed Graph")
    print("=" * 70 + "\n")

    # Assume you already have a directed graph G
    G = nx.DiGraph()

    # Add some nodes
    G.add_node('A', elevation=100)
    G.add_node('B', elevation=95)
    G.add_node('C', elevation=90)
    G.add_node('D', elevation=85)

    # Add edges with component types
    G.add_edge('A', 'B', component_type='pipe', length=100, diameter=0.3)
    G.add_edge('B', 'C', component_type='pump', length=0, power=50)
    G.add_edge('C', 'D', component_type='valve', length=0, diameter=0.2)
    G.add_edge('A', 'D', component_type='pipe', length=150, diameter=0.25)

    print(f"Original directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Convert to undirected graph with 5 edge types
    undirected_G = convert_directed_to_undirected(G)

    print(f"\nResult: {undirected_G.number_of_nodes()} nodes, {undirected_G.number_of_edges()} edges")

    return undirected_G


# Example 2: Using the converter class with statistics
def example_with_stats():
    """Example with statistics and visualization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: With Statistics")
    print("=" * 70 + "\n")

    # Assume you already have a directed graph G
    G = nx.DiGraph()

    # Add nodes
    for i in range(10):
        G.add_node(f'N{i}', elevation=100-i*5)

    # Add pipes (will stay undirected)
    G.add_edge('N0', 'N1', component_type='pipe', length=100)
    G.add_edge('N1', 'N2', component_type='pipe', length=120)
    G.add_edge('N2', 'N3', component_type='pipe', length=80)

    # Add pumps (will get reverse edges)
    G.add_edge('N3', 'N4', component_type='pump', power=30)
    G.add_edge('N5', 'N6', component_type='pump', power=40)

    # Add valves (will get reverse edges)
    G.add_edge('N4', 'N5', component_type='valve', setting=0.8)
    G.add_edge('N7', 'N8', component_type='valve', setting=0.5)

    # Add more pipes
    G.add_edge('N6', 'N7', component_type='pipe', length=90)
    G.add_edge('N8', 'N9', component_type='pipe', length=110)

    print(f"Original directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Create converter and convert
    converter = WaterNetworkGraphConverter()
    undirected_G = converter.convert_to_undirected(G)

    # Get statistics
    stats = converter.get_edge_type_statistics(undirected_G)
    print("\nEdge Type Statistics:")
    for edge_type, count in stats.items():
        print(f"  {edge_type}: {count}")

    # Visualize
    print("\n" + converter.visualize_edge_types(undirected_G))

    return undirected_G


# Example 3: Inspecting specific edges
def example_inspect_edges():
    """Example showing how to inspect edges by type."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Inspecting Edges by Type")
    print("=" * 70 + "\n")

    # Your directed graph G
    G = nx.DiGraph()
    G.add_node('Source', node_type='reservoir')
    G.add_node('Junction1', node_type='junction')
    G.add_node('Junction2', node_type='junction')
    G.add_node('Tank', node_type='tank')

    G.add_edge('Source', 'Junction1', component_type='pipe', length=200)
    G.add_edge('Junction1', 'Junction2', component_type='pump', power=60)
    G.add_edge('Junction2', 'Tank', component_type='valve', diameter=0.3)

    # Convert
    undirected_G = convert_directed_to_undirected(G)

    # Inspect edges by type
    print("\nInspecting edges by type:")
    print("-" * 70)

    for u, v, data in undirected_G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        direction = data.get('direction', 'N/A')
        component = data.get('component_type', 'N/A')
        print(f"{u:15s} -> {v:15s} | Type: {edge_type:15s} | Dir: {direction:10s}")

    print("\n" + "-" * 70)

    # Filter edges by type
    print("\nFilter only pump edges:")
    pump_edges = [(u, v, d) for u, v, d in undirected_G.edges(data=True)
                  if d.get('edge_type') == 'pump']
    for u, v, d in pump_edges:
        print(f"  {u} -> {v}")

    print("\nFilter only pump_reverse edges:")
    pump_reverse_edges = [(u, v, d) for u, v, d in undirected_G.edges(data=True)
                          if d.get('edge_type') == 'pump_reverse']
    for u, v, d in pump_reverse_edges:
        print(f"  {u} <- {v}")

    return undirected_G


# Example 4: Working with your actual directed graph
def example_your_graph(G):
    """
    Use this function with your actual directed graph G.

    Args:
        G: Your existing nx.DiGraph with component_type attributes

    Returns:
        Undirected graph with 5 edge types
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Converting Your Directed Graph G")
    print("=" * 70 + "\n")

    print(f"Your graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # Method 1: Using convenience function
    undirected_G = convert_directed_to_undirected(G)

    # Method 2: Using converter class (if you need statistics)
    # converter = WaterNetworkGraphConverter()
    # undirected_G = converter.convert_to_undirected(G)
    # stats = converter.get_edge_type_statistics(undirected_G)
    # print(converter.visualize_edge_types(undirected_G))

    print(f"\nConverted graph: {undirected_G.number_of_nodes()} nodes, {undirected_G.number_of_edges()} edges")

    return undirected_G


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRAPH CONVERSION EXAMPLES")
    print("=" * 70)

    # Run examples
    G1 = example_basic()
    G2 = example_with_stats()
    G3 = example_inspect_edges()

    print("\n" + "=" * 70)
    print("TO USE WITH YOUR OWN GRAPH:")
    print("=" * 70)
    print("""
# If you have a directed graph G:
from gnn_pressure_estimation.utils.graph_converter import convert_directed_to_undirected

# Assuming your graph G has edges with 'component_type' attribute
undirected_G = convert_directed_to_undirected(G)

# If your edges use a different attribute name (e.g., 'edge_type'):
undirected_G = convert_directed_to_undirected(G, component_type_attr='edge_type')

# The result will have:
# - Pipes: 1 undirected edge (type='pipe')
# - Pumps: 2 edges (type='pump' and type='pump_reverse')
# - Valves: 2 edges (type='valve' and type='valve_reverse')
    """)

    print("\n" + "=" * 70)
    print("QUICK REFERENCE:")
    print("=" * 70)
    print("""
Edge Types in Undirected Graph:
1. pipe          - Original pipe edge (undirected)
2. pump          - Forward pump direction (original)
3. pump_reverse  - Reverse pump direction (added)
4. valve         - Forward valve direction (original)
5. valve_reverse - Reverse valve direction (added)

Each edge has attributes:
- edge_type: One of the 5 types above
- direction: 'original', 'forward', or 'reverse'
- component_type: 'pipe', 'pump', or 'valve'
- Plus all original edge attributes (length, diameter, etc.)
    """)
    print("=" * 70 + "\n")
