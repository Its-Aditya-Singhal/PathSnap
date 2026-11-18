"""Server-side scoring of a prediction against the reference route.

Runs on Laptop 1 only, AFTER the prediction is made. The reference route is
read here and nowhere in the prediction path; only aggregate metrics leave
this module (never the reference arcs or geometry).
"""
from dataclasses import replace

import numpy as np

from .inference import Prediction
from .io import _read
from .labels import derive_labels
from .metrics import route_continuity, route_overlap
from .pipeline import TripContext


def load_reference_route(trip_dir, trip_id):
    path = trip_dir / f"{trip_id}.route"
    if not path.exists():
        return None
    return _read(path, ["arc"], int)["arc"].to_numpy()


def score(ctx: TripContext, pred: Prediction, ref_route) -> dict:
    T = len(ctx.px)
    out = dict(n_points=T, matched_fraction=float(np.mean(pred.point_arc >= 0)),
               n_breaks=pred.n_breaks, has_reference=ref_route is not None)
    out["continuity_hmm"] = route_continuity(ctx.net, pred.point_arc, ctx.px, ctx.py)
    out["continuity_clf"] = route_continuity(ctx.net, pred.clf_arc, ctx.px, ctx.py)
    if ref_route is None:
        return out

    ref_ctx = replace(ctx, trip=replace(ctx.trip, route=ref_route))
    lab = derive_labels(ref_ctx).arc
    m = lab >= 0
    out["n_labelled"] = int(m.sum())
    out["top1_hmm"] = float(np.mean(pred.point_arc[m] == lab[m]))
    out["top1_clf"] = float(np.mean(pred.clf_arc[m] == lab[m]))

    a = pred.clf_arc
    clf_route = [int(x) for i, x in enumerate(a) if x >= 0 and (i == 0 or x != a[i - 1])]
    for tag, r in (("hmm", pred.route), ("clf", clf_route)):
        for k, v in route_overlap(ctx.net, r, ref_route).items():
            out[f"{k}_{tag}"] = float(v)
    return out
