"""Per-GPS-point reference labels derived from the ordered `.route`.

`.route` lists arcs in driving order, not one per GPS point. We assign each
point to a route arc by running the same HMM, but with candidates restricted
to arcs that are on the reference route. This respects driving order and
direction (unlike plain nearest-arc), which matters on two-way roads, loops
and U-turns. Points with no route arc within `radius` get label -1 and are
excluded from point-level metrics.

Used for TRAINING LABELS and SERVER-SIDE SCORING only, never as a model input.
"""
from dataclasses import dataclass

import numpy as np

from .candidates import generate_candidates
from .hmm import HMMConfig, ViterbiDecoder, gaussian_log_emission
from .pipeline import TripContext


@dataclass
class PointLabels:
    arc: np.ndarray     # (T,) reference arc per point, -1 = unlabelled
    dist: np.ndarray    # (T,) distance from GPS point to its label arc (m), nan if none


def derive_labels(ctx: TripContext, radius: float = 150.0, sigma: float = 10.0,
                  cfg: HMMConfig = None) -> PointLabels:
    route = ctx.trip.route
    if route is None:
        raise ValueError("trip has no .route; labels need the reference route")
    cfg = cfg or HMMConfig(sigma=sigma, beta=5.0)
    cands = generate_candidates(ctx.px, ctx.py, ctx.net, radius=radius, k=1000,
                                allowed_arcs=np.unique(route))
    res = ViterbiDecoder(ctx.net, cfg).decode(ctx.px, ctx.py, ctx.ts, cands,
                                              gaussian_log_emission(cands, cfg.sigma))
    dist = np.full(len(ctx.px), np.nan)
    ok = res.cand >= 0
    dist[ok] = cands["dist"].to_numpy()[res.cand[ok]]
    return PointLabels(res.arc, dist)
