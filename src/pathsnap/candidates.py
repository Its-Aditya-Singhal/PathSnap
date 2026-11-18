"""Candidate road segments for every GPS observation."""
from typing import Optional

import numpy as np
import pandas as pd

from .geo import point_segment
from .network import RoadNetwork


def generate_candidates(px, py, net: RoadNetwork, radius: float = 50.0, k: int = 8,
                        allowed_arcs: Optional[np.ndarray] = None) -> pd.DataFrame:
    """For each GPS point return up to `k` arcs within `radius` metres.

    Output columns: point, arc, dist (perpendicular m), t (0..1 along arc),
    rank (0 = nearest). Points with no candidate simply have no rows.
    `allowed_arcs` restricts the search (used for label derivation).
    """
    mask = None
    if allowed_arcs is not None:
        mask = np.zeros(net.n_arcs, bool)
        mask[allowed_arcs] = True

    # Sample points are <= sample_step apart, so any arc within `radius`
    # has a sample within radius + step/2.
    hits = net.tree.query_ball_point(np.c_[px, py], r=radius + net.sample_step / 2)
    pts, arcs = [], []
    for i, h in enumerate(hits):
        if not h:
            continue
        a = np.unique(net.sample_arc[h])
        if mask is not None:
            a = a[mask[a]]
        pts.append(np.full(len(a), i))
        arcs.append(a)
    if not pts:
        return pd.DataFrame(columns=["point", "arc", "dist", "t", "rank"])
    p = np.concatenate(pts)
    a = np.concatenate(arcs)
    d, t = point_segment(px[p], py[p], net.ax[a], net.ay[a], net.bx[a], net.by[a])
    df = pd.DataFrame({"point": p, "arc": a, "dist": d, "t": t})
    df = df[df.dist <= radius].sort_values(["point", "dist"], kind="stable")
    df["rank"] = df.groupby("point").cumcount()
    return df[df["rank"] < k].reset_index(drop=True)
