"""Features for each (GPS point, candidate arc) pair.

Only GPS + road network are used, never the reference route, so the same
function runs unchanged on evaluator trips.
"""
import numpy as np
import pandas as pd

from .geo import angle_diff, bearing_xy, point_segment
from .pipeline import TripContext

FEATURES = [
    "dist", "rank", "n_cands", "dist_minus_min", "dist_ratio_min",
    "t_offset", "arc_length",
    "heading_diff", "heading_cos", "gps_disp", "gps_speed",
    "win_dist_mean", "win_dist_max", "win_heading_diff",
    "has_reverse", "out_deg_v", "in_deg_u", "dt_prev",
]


def _gps_kinematics(ctx: TripContext, half: int = 2):
    """Heading / displacement / speed from points [i-half, i+half]."""
    T = len(ctx.px)
    i0 = np.clip(np.arange(T) - half, 0, T - 1)
    i1 = np.clip(np.arange(T) + half, 0, T - 1)
    dx, dy = ctx.px[i1] - ctx.px[i0], ctx.py[i1] - ctx.py[i0]
    disp = np.hypot(dx, dy)
    dt = np.maximum(ctx.ts[i1] - ctx.ts[i0], 1e-3)
    dt_prev = np.r_[0.0, np.diff(ctx.ts)]
    return bearing_xy(dx, dy), disp, disp / dt, dt_prev


def build_features(ctx: TripContext, cands: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    net = ctx.net
    p = cands["point"].to_numpy()
    a = cands["arc"].to_numpy()
    T = len(ctx.px)
    f = pd.DataFrame(index=cands.index)

    d = cands["dist"].to_numpy()
    f["dist"] = d
    f["rank"] = cands["rank"].to_numpy()
    g = cands.groupby("point")["dist"]
    f["n_cands"] = g.transform("size").to_numpy()
    dmin = g.transform("min").to_numpy()
    f["dist_minus_min"] = d - dmin
    f["dist_ratio_min"] = (d + 1.0) / (dmin + 1.0)
    f["t_offset"] = cands["t"].to_numpy()
    f["arc_length"] = net.length[a]

    heading, disp, speed, dt_prev = _gps_kinematics(ctx)
    hd = angle_diff(heading[p], net.bearing[a])
    f["heading_diff"] = hd
    f["heading_cos"] = np.cos(np.radians(hd))
    f["gps_disp"] = disp[p]
    f["gps_speed"] = speed[p]

    # Temporal support: how well neighbouring GPS points also fit this arc.
    wd = []
    for o in range(-window, window + 1):
        if o == 0:
            continue
        q = np.clip(p + o, 0, T - 1)
        dd, _ = point_segment(ctx.px[q], ctx.py[q], net.ax[a], net.ay[a], net.bx[a], net.by[a])
        wd.append(dd)
    wd = np.stack(wd)
    f["win_dist_mean"] = wd.mean(0)
    f["win_dist_max"] = wd.max(0)
    hw, _, _, _ = _gps_kinematics(ctx, half=window)
    f["win_heading_diff"] = angle_diff(hw[p], net.bearing[a])

    rev = np.array([(net.v[x], net.u[x]) in net.edge_arc for x in a])
    f["has_reverse"] = rev.astype(float)
    indptr = net.graph.indptr
    out_deg = np.diff(indptr)
    in_deg = np.bincount(net.graph.indices, minlength=net.n_nodes)
    f["out_deg_v"] = out_deg[net.v[a]]
    f["in_deg_u"] = in_deg[net.u[a]]
    f["dt_prev"] = np.log1p(dt_prev[p])   # log: one gap is 153,064 s
    return f[FEATURES]
