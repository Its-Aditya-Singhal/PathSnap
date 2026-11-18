"""Per-trip road network in local metres + shortest-path queries."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .geo import LocalProjector, bearing_xy


class RoadNetwork:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, proj: LocalProjector,
                 sample_step: float = 10.0):
        self.proj = proj
        self.nx, self.ny = proj.project(nodes.lon.values, nodes.lat.values)
        self.u = arcs.u.to_numpy()
        self.v = arcs.v.to_numpy()
        self.n_nodes, self.n_arcs = len(nodes), len(arcs)

        self.ax, self.ay = self.nx[self.u], self.ny[self.u]
        self.bx, self.by = self.nx[self.v], self.ny[self.v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)

        # Directed node graph; parallel arcs keep the shortest one.
        e = pd.DataFrame({"u": self.u, "v": self.v, "w": np.maximum(self.length, 1e-3),
                          "arc": np.arange(self.n_arcs)})
        e = e[e.u != e.v].sort_values("w").drop_duplicates(["u", "v"])
        self.graph = csr_matrix((e.w.values, (e.u.values, e.v.values)),
                                shape=(self.n_nodes, self.n_nodes))
        self.edge_arc = {(a, b): arc for a, b, arc in zip(e.u.values, e.v.values, e.arc.values)}

        # Densified sample points along every arc for fast spatial lookup.
        n = np.maximum(np.ceil(self.length / sample_step).astype(int), 1) + 1
        arc_idx = np.repeat(np.arange(self.n_arcs), n)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n])
        self.sample_arc = arc_idx
        sx = self.ax[arc_idx] + frac * (self.bx - self.ax)[arc_idx]
        sy = self.ay[arc_idx] + frac * (self.by - self.ay)[arc_idx]
        self.sample_step = sample_step
        self.tree = cKDTree(np.c_[sx, sy])

    def sp_from(self, sources, limit=np.inf):
        """Shortest node-to-node distances (metres) from each source node.

        Returns a (len(sources), n_nodes) array; inf where unreachable
        within `limit`.
        """
        return dijkstra(self.graph, directed=True, indices=np.asarray(sources), limit=limit)

    def path_arcs(self, src_node, dst_node, limit=np.inf):
        """Arc ids on the shortest path src_node -> dst_node ([] if same / unreachable)."""
        if src_node == dst_node:
            return []
        dist, pred = dijkstra(self.graph, directed=True, indices=src_node,
                              limit=limit, return_predecessors=True)
        if not np.isfinite(dist[dst_node]):
            return None
        nodes = [dst_node]
        while nodes[-1] != src_node:
            nodes.append(pred[nodes[-1]])
        nodes.reverse()
        return [self.edge_arc[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]
