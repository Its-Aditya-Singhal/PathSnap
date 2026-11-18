"""Inference: raw trip (track + road network, NO route) -> matched route.

This is the only code path the API uses to make a prediction. It never
reads a .route file; evaluation against the reference route happens
separately in `pathsnap.serve_eval`.

Artifact bundle (artifacts/pathsnap_model.joblib), written by
scripts/export_artifacts.py:
  model_name, model (fitted classifier or None), features, hmm (HMMConfig dict),
  emission_weight, dist_sigma, radius, k, created, trained_on
Emission = emission_weight * log P_clf + Gaussian distance term (dist_sigma, optional).
Without a bundle the predictor falls back to the Gaussian-emission HMM baseline.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .candidates import generate_candidates
from .config import ARTIFACTS_DIR
from .features import FEATURES, build_features
from .hmm import HMMConfig, ViterbiDecoder, gaussian_log_emission, hybrid_log_emission, reconstruct_route
from .io import Trip
from .pipeline import TripContext, prepare

BUNDLE_PATH = ARTIFACTS_DIR / "pathsnap_model.joblib"


@dataclass
class Prediction:
    route: list                 # ordered predicted arc ids
    point_arc: np.ndarray       # (T,) matched arc per GPS point (-1 = unmatched)
    clf_arc: np.ndarray         # (T,) classifier-only argmax arc per point (-1 = none)
    segment: np.ndarray
    n_breaks: int
    timings_ms: dict = field(default_factory=dict)
    candidate_arcs: list = field(default_factory=list)   # distinct arcs considered (for display)


class Predictor:
    def __init__(self, bundle_path: Path = BUNDLE_PATH):
        if Path(bundle_path).exists():
            b = joblib.load(bundle_path)
            self.bundle_path = str(bundle_path)
        else:
            b = dict(model_name="hmm_baseline", model=None, features=FEATURES,
                     hmm=HMMConfig().to_dict(), emission_weight=1.0, radius=50.0, k=8,
                     created=None, trained_on=None)
            self.bundle_path = None
        assert list(b["features"]) == FEATURES, "artifact was trained with a different feature list"
        self.model_name = b["model_name"]
        self.model = b["model"]
        self.cfg = HMMConfig(**b["hmm"])
        self.emission_weight = float(b["emission_weight"])
        self.dist_sigma = b.get("dist_sigma")
        self.radius, self.k = float(b["radius"]), int(b["k"])
        self.meta = {k: b.get(k) for k in ("model_name", "created", "trained_on", "emission_weight",
                                             "dist_sigma", "radius", "k", "hmm")}

    def predict(self, trip: Trip, ctx: TripContext = None) -> tuple[Prediction, TripContext]:
        assert trip.route is None, "predictor must not receive the reference route"
        tm = {}
        t0 = time.perf_counter()
        ctx = ctx or prepare(trip)
        tm["prepare"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cands = generate_candidates(ctx.px, ctx.py, ctx.net, radius=self.radius, k=self.k)
        tm["candidates"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.model is None:
            log_e = gaussian_log_emission(cands, self.cfg.sigma)
            prob = np.exp(log_e)
        else:
            X = build_features(ctx, cands).to_numpy(np.float32)
            prob = self.model.predict_proba(X)[:, 1]
            log_e = hybrid_log_emission(cands, prob, self.emission_weight, self.dist_sigma)
        tm["features_and_model"] = time.perf_counter() - t0

        T = len(ctx.px)
        clf_arc = np.full(T, -1)
        if len(cands):
            c = cands.assign(p=prob)
            best = c.loc[c.groupby("point")["p"].idxmax()]
            clf_arc[best.point.to_numpy()] = best.arc.to_numpy()

        t0 = time.perf_counter()
        res = ViterbiDecoder(ctx.net, self.cfg).decode(ctx.px, ctx.py, ctx.ts, cands, log_e)
        route = reconstruct_route(ctx.net, res.arc, res.segment)
        tm["viterbi_and_route"] = time.perf_counter() - t0

        pred = Prediction(route=[int(a) for a in route], point_arc=res.arc, clf_arc=clf_arc,
                          segment=res.segment, n_breaks=int(res.n_breaks),
                          timings_ms={k: round(1000 * v, 1) for k, v in tm.items()},
                          candidate_arcs=sorted(int(a) for a in cands.arc.unique()) if len(cands) else [])
        return pred, ctx


def route_polylines(trip: Trip, route) -> list:
    """Chain consecutive arcs sharing a node into [[lat, lon], ...] polylines."""
    if not len(route):
        return []
    u = trip.arcs.u.to_numpy()
    v = trip.arcs.v.to_numpy()
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    lines, cur = [], [int(u[route[0]])]
    for prev, a in zip([None, *route[:-1]], route):
        if prev is not None and u[a] != v[prev]:
            lines.append(cur)
            cur = [int(u[a])]
        cur.append(int(v[a]))
    lines.append(cur)
    return [[[round(float(lat[n]), 6), round(float(lon[n]), 6)] for n in line] for line in lines]


def arc_segments(trip: Trip, arcs) -> list:
    """[[lat, lon], [lat, lon]] per arc; a two-way road (u,v)/(v,u) is drawn once."""
    u = trip.arcs.u.to_numpy()
    v = trip.arcs.v.to_numpy()
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    seen, out = set(), []
    for a in arcs:
        key = tuple(sorted((int(u[a]), int(v[a]))))
        if key in seen:
            continue
        seen.add(key)
        out.append([[round(float(lat[n]), 6), round(float(lon[n]), 6)] for n in key])
    return out
