"""Evaluation metrics for point-level and route-level matching."""
import numpy as np
import pandas as pd

from .network import RoadNetwork


def top1_accuracy(pred_arc, label_arc):
    m = label_arc >= 0
    return float(np.mean(pred_arc[m] == label_arc[m])) if m.any() else np.nan


def candidate_miss_rate(cands: pd.DataFrame, label_arc):
    """Fraction of labelled points whose reference arc is not among their candidates."""
    lab = pd.DataFrame({"point": np.arange(len(label_arc)), "arc": label_arc})
    lab = lab[lab.arc >= 0]
    hit = lab.merge(cands[["point", "arc"]], on=["point", "arc"], how="left", indicator=True)
    return float((hit["_merge"] == "left_only").mean()) if len(lab) else np.nan


def route_continuity(net: RoadNetwork, pred_arc, px, py, slack: float = 50.0):
    """Share of consecutive matched-point pairs whose arcs are physically consistent:
    same arc, or the second is reachable from the first within d_gps + slack metres."""
    idx = np.flatnonzero(pred_arc >= 0)
    if len(idx) < 2:
        return np.nan
    a, b = pred_arc[idx[:-1]], pred_arc[idx[1:]]
    d_gps = np.hypot(px[idx[1:]] - px[idx[:-1]], py[idx[1:]] - py[idx[:-1]])
    ok = (a == b) | (net.v[a] == net.u[b])
    todo = np.flatnonzero(~ok)
    if len(todo):
        src = np.unique(net.v[a[todo]])
        sp = net.sp_from(src, limit=float(d_gps[todo].max() + slack))
        row = {s: i for i, s in enumerate(np.atleast_1d(src))}
        sp = np.atleast_2d(sp)
        for i in todo:
            ok[i] = sp[row[net.v[a[i]]], net.u[b[i]]] <= d_gps[i] + slack
    return float(ok.mean())


def route_overlap(net: RoadNetwork, pred_route, true_route):
    """Length-weighted precision / recall / F1 of predicted vs reference arc sets.
    Undirected: an arc and its reverse twin count as the same road."""
    def key(arcs):
        arcs = np.unique(np.asarray(arcs, int))
        return {tuple(sorted((net.u[a], net.v[a]))): net.length[a] for a in arcs}
    P, R = key(pred_route), key(true_route)
    inter = sum(R[k] for k in P.keys() & R.keys())
    prec = inter / max(sum(P.values()), 1e-9)
    rec = inter / max(sum(R.values()), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(route_precision=prec, route_recall=rec, route_f1=f1)
