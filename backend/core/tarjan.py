"""Custom Iterative Tarjan Algorithms for SCCs, Bridges, and Articulation Points."""
from typing import Dict, List, Set, Tuple, Any

class IterativeTarjan:
    """Purely iterative implementations of Tarjan's algorithms avoiding recursion limits."""

    @staticmethod
    def strongly_connected_components(adjacency: Dict[str, List[str]]) -> List[List[str]]:
        """Iterative Tarjan's algorithm for Strongly Connected Components (SCC).

        Returns list of strongly connected components as lists of node identifiers.
        """
        index = 0
        indices: Dict[str, int] = {}
        lowlinks: Dict[str, int] = {}
        on_stack: Set[str] = set()
        stack: List[str] = []
        sccs: List[List[str]] = []

        all_nodes = list(adjacency.keys())

        for root in all_nodes:
            if root in indices:
                continue

            # Explicit call stack: (node, neighbor_iterator_index)
            call_stack: List[Tuple[str, int]] = [(root, 0)]
            indices[root] = index
            lowlinks[root] = index
            index += 1
            stack.append(root)
            on_stack.add(root)

            while call_stack:
                node, nbr_idx = call_stack[-1]
                neighbors = adjacency.get(node, [])

                if nbr_idx < len(neighbors):
                    neighbor = neighbors[nbr_idx]
                    # Update index for next iteration when we return to this node
                    call_stack[-1] = (node, nbr_idx + 1)

                    if neighbor not in indices:
                        indices[neighbor] = index
                        lowlinks[neighbor] = index
                        index += 1
                        stack.append(neighbor)
                        on_stack.add(neighbor)
                        call_stack.append((neighbor, 0))
                    elif neighbor in on_stack:
                        lowlinks[node] = min(lowlinks[node], indices[neighbor])
                else:
                    # Finished exploring all neighbors of node
                    call_stack.pop()

                    # If node is a root of an SCC, pop the component from stack
                    if lowlinks[node] == indices[node]:
                        scc: List[str] = []
                        while True:
                            w = stack.pop()
                            on_stack.remove(w)
                            scc.append(w)
                            if w == node:
                                break
                        sccs.append(scc)

                    # Propagate lowlink to parent in call stack
                    if call_stack:
                        parent = call_stack[-1][0]
                        lowlinks[parent] = min(lowlinks[parent], lowlinks[node])

        return sccs

    @staticmethod
    def bridges_and_articulation_points(
        nodes: List[str],
        edges: List[Tuple[str, str]]
    ) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """Iterative Tarjan's algorithm for finding bridge edges and articulation point nodes.

        Bridges and articulation points are critical topological connectors.
        Returns:
            (bridges_set, articulation_points_set)
        """
        # Build undirected adjacency for bridge/articulation analysis
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for u, v in edges:
            if u in adj and v in adj and u != v:
                adj[u].append(v)
                adj[v].append(u)

        discovery: Dict[str, int] = {}
        low: Dict[str, int] = {}
        parent: Dict[str, Optional[str]] = {}
        time = 0

        bridges: Set[Tuple[str, str]] = set()
        articulation_points: Set[str] = set()
        children_count: Dict[str, int] = {n: 0 for n in nodes}

        for root in nodes:
            if root in discovery:
                continue

            parent[root] = None
            discovery[root] = time
            low[root] = time
            time += 1

            # (node, neighbor_index)
            dfs_stack: List[Tuple[str, int]] = [(root, 0)]

            while dfs_stack:
                u, nbr_idx = dfs_stack[-1]
                neighbors = adj[u]

                if nbr_idx < len(neighbors):
                    v = neighbors[nbr_idx]
                    dfs_stack[-1] = (u, nbr_idx + 1)

                    if v not in discovery:
                        parent[v] = u
                        children_count[u] = children_count.get(u, 0) + 1
                        discovery[v] = time
                        low[v] = time
                        time += 1
                        dfs_stack.append((v, 0))
                    elif v != parent.get(u):
                        low[u] = min(low[u], discovery[v])
                else:
                    # Post-visit u
                    dfs_stack.pop()
                    if dfs_stack:
                        p = dfs_stack[-1][0]
                        low[p] = min(low[p], low[u])

                        # Bridge condition: low[u] > discovery[p]
                        if low[u] > discovery[p]:
                            # Store canonical sorted tuple
                            edge = (min(p, u), max(p, u))
                            bridges.add(edge)

                        # Articulation point condition for non-root
                        if parent.get(p) is not None and low[u] >= discovery[p]:
                            articulation_points.add(p)

            # Articulation point condition for root of DFS tree: 2 or more children
            if children_count.get(root, 0) > 1:
                articulation_points.add(root)

        return bridges, articulation_points

tarjan = IterativeTarjan()
