"""
Graph Converter for Water Network Analysis

This module converts a directed water network graph into an undirected graph with
specialized edge types for pumps and valves.

Edge Types (5 total):
1. pipe - undirected pipe connections
2. pump - forward pump direction (from_node → to_node)
3. pump_reverse - reverse pump direction (to_node → from_node)
4. valve - forward valve direction (from_node → to_node)
5. valve_reverse - reverse valve direction (to_node → from_node)
"""

import networkx as nx
from typing import Dict, Any, Optional
from copy import deepcopy


class WaterNetworkGraphConverter:
    """
    Converts directed water network graphs to undirected graphs with specialized edge types.

    This converter handles the asymmetric nature of pumps and valves by creating
    reverse edge types, while treating pipes as truly undirected connections.
    """

    # Edge type constants
    EDGE_TYPE_PIPE = "pipe"
    EDGE_TYPE_PUMP = "pump"
    EDGE_TYPE_PUMP_REVERSE = "pump_reverse"
    EDGE_TYPE_VALVE = "valve"
    EDGE_TYPE_VALVE_REVERSE = "valve_reverse"

    def __init__(self):
        """Initialize the graph converter."""
        self.edge_types = [
            self.EDGE_TYPE_PIPE,
            self.EDGE_TYPE_PUMP,
            self.EDGE_TYPE_PUMP_REVERSE,
            self.EDGE_TYPE_VALVE,
            self.EDGE_TYPE_VALVE_REVERSE
        ]

    def convert_to_undirected(
        self,
        directed_graph: nx.DiGraph,
        component_type_attr: str = 'component_type'
    ) -> nx.Graph:
        """
        Convert a directed water network graph to an undirected graph with specialized edge types.

        Args:
            directed_graph: NetworkX directed graph with water network components
            component_type_attr: Name of the edge attribute that stores component type
                               ('pipe', 'pump', or 'valve')

        Returns:
            NetworkX undirected graph with 5 edge types:
            - pipe: undirected pipes
            - pump: forward pump direction
            - pump_reverse: reverse pump direction
            - valve: forward valve direction
            - valve_reverse: reverse valve direction

        Example:
            >>> converter = WaterNetworkGraphConverter()
            >>> directed_g = nx.DiGraph()
            >>> directed_g.add_edge('node1', 'node2', component_type='pipe', length=100)
            >>> directed_g.add_edge('node2', 'node3', component_type='pump', length=0)
            >>> undirected_g = converter.convert_to_undirected(directed_g)
        """
        # Create a new undirected MultiGraph to handle multiple edges between nodes
        undirected_graph = nx.MultiGraph()

        # Copy all nodes with their attributes
        for node, attrs in directed_graph.nodes(data=True):
            undirected_graph.add_node(node, **attrs)

        # First pass: Add all edges from directed graph with their original types
        print("=" * 60)
        print("Processing Directed Graph Edges")
        print("=" * 60)

        for u, v, edge_data in directed_graph.edges(data=True):
            # Get component type from edge attributes
            component_type = edge_data.get(component_type_attr, 'pipe').lower()

            # Copy all edge attributes
            forward_attrs = deepcopy(edge_data)

            # Set edge type based on component type
            if component_type == 'pipe':
                forward_attrs['edge_type'] = self.EDGE_TYPE_PIPE
                forward_attrs['direction'] = 'original'
            elif component_type == 'pump':
                forward_attrs['edge_type'] = self.EDGE_TYPE_PUMP
                forward_attrs['direction'] = 'forward'
            elif component_type == 'valve':
                forward_attrs['edge_type'] = self.EDGE_TYPE_VALVE
                forward_attrs['direction'] = 'forward'
            else:
                # Unknown type - treat as pipe
                forward_attrs['edge_type'] = self.EDGE_TYPE_PIPE
                forward_attrs['direction'] = 'original'

            # Add the edge
            undirected_graph.add_edge(u, v, **forward_attrs)
            print(f"Added edge: {u} -> {v} | Type: {forward_attrs['edge_type']}")

        # Second pass: Add reverse edges only for pumps and valves (not pipes)
        print("\n" + "=" * 60)
        print("Adding Reverse Edges for Pumps and Valves")
        print("=" * 60)

        for u, v, edge_data in directed_graph.edges(data=True):
            component_type = edge_data.get(component_type_attr, 'pipe').lower()

            if component_type == 'pipe':
                # Pipes: do nothing (already added as undirected)
                print(f"Skipped reverse for pipe: {u} -> {v} (pipes are undirected)")

            elif component_type == 'pump':
                # Pumps: add reverse edge
                reverse_attrs = deepcopy(edge_data)
                reverse_attrs['edge_type'] = self.EDGE_TYPE_PUMP_REVERSE
                reverse_attrs['direction'] = 'reverse'
                undirected_graph.add_edge(v, u, **reverse_attrs)
                print(f"Added reverse: {v} <- {u} | Type: {reverse_attrs['edge_type']}")

            elif component_type == 'valve':
                # Valves: add reverse edge
                reverse_attrs = deepcopy(edge_data)
                reverse_attrs['edge_type'] = self.EDGE_TYPE_VALVE_REVERSE
                reverse_attrs['direction'] = 'reverse'
                undirected_graph.add_edge(v, u, **reverse_attrs)
                print(f"Added reverse: {v} <- {u} | Type: {reverse_attrs['edge_type']}")

        print("=" * 60)
        print(f"Total edges in undirected graph: {undirected_graph.number_of_edges()}")
        print("=" * 60 + "\n")

        return undirected_graph

    def convert_from_wntr(self, wn_model) -> nx.Graph:
        """
        Convert a WNTR WaterNetworkModel directly to undirected graph with specialized edge types.

        Args:
            wn_model: WNTR WaterNetworkModel object

        Returns:
            NetworkX undirected graph with 5 edge types

        Example:
            >>> import wntr
            >>> wn = wntr.network.WaterNetworkModel('network.inp')
            >>> converter = WaterNetworkGraphConverter()
            >>> undirected_g = converter.convert_from_wntr(wn)
        """
        # Create a new undirected MultiGraph
        undirected_graph = nx.MultiGraph()

        # Add all nodes from the water network
        for node_name in wn_model.node_name_list:
            node = wn_model.get_node(node_name)
            undirected_graph.add_node(
                node_name,
                node_type=type(node).__name__,
                elevation=getattr(node, 'elevation', 0)
            )

        # First pass: Add all edges with their original types
        print("=" * 60)
        print("Processing WNTR Water Network Model")
        print("=" * 60)

        # Process pipes - undirected
        for pipe in wn_model.pipes:
            undirected_graph.add_edge(
                pipe.start_node_name,
                pipe.end_node_name,
                component_type='pipe',
                edge_type=self.EDGE_TYPE_PIPE,
                direction='original',
                length=pipe.length,
                diameter=pipe.diameter,
                roughness=pipe.roughness,
                component_name=pipe.name
            )
            print(f"Added pipe: {pipe.start_node_name} -> {pipe.end_node_name}")

        # Process pumps - forward direction
        for pump in wn_model.pumps:
            undirected_graph.add_edge(
                pump.start_node_name,
                pump.end_node_name,
                component_type='pump',
                edge_type=self.EDGE_TYPE_PUMP,
                direction='forward',
                length=0.0,
                component_name=pump.name,
                pump_type=pump.pump_type
            )
            print(f"Added pump: {pump.start_node_name} -> {pump.end_node_name}")

        # Process valves - forward direction
        for valve in wn_model.valves:
            undirected_graph.add_edge(
                valve.start_node_name,
                valve.end_node_name,
                component_type='valve',
                edge_type=self.EDGE_TYPE_VALVE,
                direction='forward',
                length=0.0,
                diameter=valve.diameter,
                component_name=valve.name,
                valve_type=valve.valve_type
            )
            print(f"Added valve: {valve.start_node_name} -> {valve.end_node_name}")

        # Second pass: Add reverse edges only for pumps and valves
        print("\n" + "=" * 60)
        print("Adding Reverse Edges for Pumps and Valves")
        print("=" * 60)

        # Add reverse edges for pumps
        for pump in wn_model.pumps:
            undirected_graph.add_edge(
                pump.end_node_name,
                pump.start_node_name,
                component_type='pump',
                edge_type=self.EDGE_TYPE_PUMP_REVERSE,
                direction='reverse',
                length=0.0,
                component_name=pump.name,
                pump_type=pump.pump_type
            )
            print(f"Added pump reverse: {pump.end_node_name} <- {pump.start_node_name}")

        # Add reverse edges for valves
        for valve in wn_model.valves:
            undirected_graph.add_edge(
                valve.end_node_name,
                valve.start_node_name,
                component_type='valve',
                edge_type=self.EDGE_TYPE_VALVE_REVERSE,
                direction='reverse',
                length=0.0,
                diameter=valve.diameter,
                component_name=valve.name,
                valve_type=valve.valve_type
            )
            print(f"Added valve reverse: {valve.end_node_name} <- {valve.start_node_name}")

        print("=" * 60)
        print(f"Total edges in undirected graph: {undirected_graph.number_of_edges()}")
        print("=" * 60 + "\n")

        return undirected_graph

    def convert_from_epynet(self, wn_epynet) -> nx.Graph:
        """
        Convert an EPYNET Network object to undirected graph with specialized edge types.

        Args:
            wn_epynet: EPYNET Network object

        Returns:
            NetworkX undirected graph with 5 edge types

        Example:
            >>> from epynet import Network
            >>> wn = Network('network.inp')
            >>> converter = WaterNetworkGraphConverter()
            >>> undirected_g = converter.convert_from_epynet(wn)
        """
        # Create a new undirected MultiGraph
        undirected_graph = nx.MultiGraph()

        # Add all nodes
        for node in wn_epynet.nodes:
            undirected_graph.add_node(
                node.uid,
                node_type=type(node).__name__,
                elevation=getattr(node, 'elevation', 0)
            )

        # First pass: Add all edges with their original types
        print("=" * 60)
        print("Processing EPYNET Network")
        print("=" * 60)

        # Process pipes - undirected
        for pipe in wn_epynet.pipes:
            undirected_graph.add_edge(
                pipe.from_node.uid,
                pipe.to_node.uid,
                component_type='pipe',
                edge_type=self.EDGE_TYPE_PIPE,
                direction='original',
                length=pipe.length,
                diameter=getattr(pipe, 'diameter', 0),
                roughness=getattr(pipe, 'roughness', 0),
                component_name=pipe.uid
            )
            print(f"Added pipe: {pipe.from_node.uid} -> {pipe.to_node.uid}")

        # Process pumps - forward direction
        for pump in wn_epynet.pumps:
            undirected_graph.add_edge(
                pump.from_node.uid,
                pump.to_node.uid,
                component_type='pump',
                edge_type=self.EDGE_TYPE_PUMP,
                direction='forward',
                length=0.0,
                component_name=pump.uid,
                speed=getattr(pump, 'speed', 1.0),
                status=getattr(pump, 'status', 'OPEN')
            )
            print(f"Added pump: {pump.from_node.uid} -> {pump.to_node.uid}")

        # Process valves - forward direction
        for valve in wn_epynet.valves:
            undirected_graph.add_edge(
                valve.from_node.uid,
                valve.to_node.uid,
                component_type='valve',
                edge_type=self.EDGE_TYPE_VALVE,
                direction='forward',
                length=0.0,
                diameter=getattr(valve, 'diameter', 0),
                component_name=valve.uid,
                setting=getattr(valve, 'setting', 0),
                status=getattr(valve, 'status', 'OPEN')
            )
            print(f"Added valve: {valve.from_node.uid} -> {valve.to_node.uid}")

        # Second pass: Add reverse edges only for pumps and valves
        print("\n" + "=" * 60)
        print("Adding Reverse Edges for Pumps and Valves")
        print("=" * 60)

        # Add reverse edges for pumps
        for pump in wn_epynet.pumps:
            undirected_graph.add_edge(
                pump.to_node.uid,
                pump.from_node.uid,
                component_type='pump',
                edge_type=self.EDGE_TYPE_PUMP_REVERSE,
                direction='reverse',
                length=0.0,
                component_name=pump.uid,
                speed=getattr(pump, 'speed', 1.0),
                status=getattr(pump, 'status', 'OPEN')
            )
            print(f"Added pump reverse: {pump.to_node.uid} <- {pump.from_node.uid}")

        # Add reverse edges for valves
        for valve in wn_epynet.valves:
            undirected_graph.add_edge(
                valve.to_node.uid,
                valve.from_node.uid,
                component_type='valve',
                edge_type=self.EDGE_TYPE_VALVE_REVERSE,
                direction='reverse',
                length=0.0,
                diameter=getattr(valve, 'diameter', 0),
                component_name=valve.uid,
                setting=getattr(valve, 'setting', 0),
                status=getattr(valve, 'status', 'OPEN')
            )
            print(f"Added valve reverse: {valve.to_node.uid} <- {valve.from_node.uid}")

        print("=" * 60)
        print(f"Total edges in undirected graph: {undirected_graph.number_of_edges()}")
        print("=" * 60 + "\n")

        return undirected_graph

    def get_edge_type_statistics(self, graph: nx.Graph) -> Dict[str, int]:
        """
        Get statistics about edge types in the graph.

        Args:
            graph: NetworkX graph with edge_type attribute

        Returns:
            Dictionary mapping edge type to count

        Example:
            >>> stats = converter.get_edge_type_statistics(undirected_g)
            >>> print(stats)
            {'pipe': 120, 'pump': 5, 'pump_reverse': 5, 'valve': 10, 'valve_reverse': 10}
        """
        stats = {edge_type: 0 for edge_type in self.edge_types}

        for u, v, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            if edge_type in stats:
                stats[edge_type] += 1
            else:
                stats['unknown'] = stats.get('unknown', 0) + 1

        return stats

    def visualize_edge_types(self, graph: nx.Graph) -> str:
        """
        Create a text-based visualization of edge type distribution.

        Args:
            graph: NetworkX graph with edge_type attribute

        Returns:
            Formatted string showing edge type statistics
        """
        stats = self.get_edge_type_statistics(graph)
        total_edges = sum(stats.values())

        output = [
            "=" * 60,
            "Water Network Graph Edge Type Distribution",
            "=" * 60,
            f"Total Edges: {total_edges}",
            f"Total Nodes: {graph.number_of_nodes()}",
            "-" * 60,
        ]

        for edge_type in self.edge_types:
            count = stats.get(edge_type, 0)
            percentage = (count / total_edges * 100) if total_edges > 0 else 0
            bar = "█" * int(percentage / 2)  # Scale bar to 50 chars max
            output.append(f"{edge_type:20s}: {count:5d} ({percentage:5.1f}%) {bar}")

        if 'unknown' in stats:
            count = stats['unknown']
            percentage = (count / total_edges * 100) if total_edges > 0 else 0
            output.append(f"{'unknown':20s}: {count:5d} ({percentage:5.1f}%)")

        output.append("=" * 60)

        return "\n".join(output)


# Convenience functions
def convert_directed_to_undirected(directed_graph: nx.DiGraph,
                                  component_type_attr: str = 'component_type') -> nx.Graph:
    """
    Convenience function to convert a directed graph to undirected with specialized edge types.

    Args:
        directed_graph: NetworkX directed graph
        component_type_attr: Attribute name for component type

    Returns:
        Undirected NetworkX graph with 5 edge types
    """
    converter = WaterNetworkGraphConverter()
    return converter.convert_to_undirected(directed_graph, component_type_attr)


def convert_wntr_to_undirected(wn_model) -> nx.Graph:
    """
    Convenience function to convert WNTR model to undirected graph with specialized edge types.

    Args:
        wn_model: WNTR WaterNetworkModel

    Returns:
        Undirected NetworkX graph with 5 edge types
    """
    converter = WaterNetworkGraphConverter()
    return converter.convert_from_wntr(wn_model)


def convert_epynet_to_undirected(wn_epynet) -> nx.Graph:
    """
    Convenience function to convert EPYNET Network to undirected graph with specialized edge types.

    Args:
        wn_epynet: EPYNET Network object

    Returns:
        Undirected NetworkX graph with 5 edge types
    """
    converter = WaterNetworkGraphConverter()
    return converter.convert_from_epynet(wn_epynet)


if __name__ == "__main__":
    # Example usage
    print("Water Network Graph Converter")
    print("=" * 60)
    print("\nThis module converts directed water network graphs to undirected")
    print("graphs with 5 specialized edge types:")
    print("  1. pipe          - undirected pipe connections")
    print("  2. pump          - forward pump direction")
    print("  3. pump_reverse  - reverse pump direction")
    print("  4. valve         - forward valve direction")
    print("  5. valve_reverse - reverse valve direction")
    print("\nUsage examples:")
    print("-" * 60)
    print("\n# From WNTR:")
    print("import wntr")
    print("from graph_converter import convert_wntr_to_undirected")
    print("wn = wntr.network.WaterNetworkModel('network.inp')")
    print("undirected_g = convert_wntr_to_undirected(wn)")
    print("\n# From EPYNET:")
    print("from epynet import Network")
    print("from graph_converter import convert_epynet_to_undirected")
    print("wn = Network('network.inp')")
    print("undirected_g = convert_epynet_to_undirected(wn)")
    print("\n# From existing directed graph:")
    print("from graph_converter import convert_directed_to_undirected")
    print("undirected_g = convert_directed_to_undirected(directed_g)")
