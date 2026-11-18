"""PathSnap API (Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment:
  PATHSNAP_DATA      folder holding the partitions (default ../Student folder)
  PATHSNAP_DATASETS  comma-separated partitions to serve
                     (default: evaluator_test,student_test - whichever exist)
  PATHSNAP_EVAL_DIR  optional extra folder of trips (served as dataset "custom"),
                     e.g. new trips handed out on the day
  PATHSNAP_BUNDLE    model bundle path (default artifacts/pathsnap_model.joblib)

The training partition is never served. Predictions are made without the
.route file; the route is read afterwards, server-side, only to compute
scores. Reference arcs/geometry never leave this process.

Every request is logged to the console and to results/logs/api_requests.jsonl.
"""
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.config import DATA_ROOT, LOGS_DIR
from pathsnap.inference import BUNDLE_PATH, Predictor, arc_segments, route_polylines
from pathsnap.io import load_trip
from pathsnap.serve_eval import load_reference_route, score

BUNDLE = Path(os.environ.get("PATHSNAP_BUNDLE", BUNDLE_PATH))
LOG_FILE = LOGS_DIR / "api_requests.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")


def _discover():
    names = [n.strip() for n in os.environ.get("PATHSNAP_DATASETS", "evaluator_test,student_test").split(",")]
    sets = {}
    for n in names:
        if n and n != "train" and (DATA_ROOT / n).is_dir():
            sets[n] = DATA_ROOT / n
    extra = os.environ.get("PATHSNAP_EVAL_DIR")
    if extra and Path(extra).is_dir():
        sets["custom"] = Path(extra)
    return {k: {p.name: p for p in sorted(d.iterdir()) if p.is_dir()} for k, d in sets.items()}


app = FastAPI(title="PathSnap map-matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor(BUNDLE)
DATASETS = _discover()
DEFAULT = next(iter(DATASETS), None)
STARTED = datetime.now(timezone.utc).isoformat()
RECENT = deque(maxlen=50)
log.info("model=%s bundle=%s", predictor.model_name, predictor.bundle_path or "(none, HMM baseline)")
for k, v in DATASETS.items():
    log.info("serving dataset %-15s %3d trips", k, len(v))


def _jsonable(d):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def _append_log(rec):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _trip_dir(dataset, trip_id):
    ds = dataset or DEFAULT
    if ds not in DATASETS:
        raise HTTPException(404, f"unknown dataset {ds!r}; available: {list(DATASETS)}")
    if trip_id not in DATASETS[ds]:
        raise HTTPException(404, f"unknown trip_id {trip_id!r} in dataset {ds!r}")
    return ds, DATASETS[ds][trip_id]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = 1000 * (time.perf_counter() - t0)
    client = request.client.host if request.client else "?"
    path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    log.info("%s %s %s -> %d (%.0f ms)", client, request.method, path, response.status_code, ms)
    _append_log(dict(ts=datetime.now(timezone.utc).isoformat(), client=client, method=request.method,
                     path=path, status=response.status_code, ms=round(ms, 1)))
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.meta, "started": STARTED,
            "datasets": {k: len(v) for k, v in DATASETS.items()}, "default_dataset": DEFAULT,
            "n_trips": sum(len(v) for v in DATASETS.values())}


@app.get("/datasets")
def datasets():
    return {"datasets": [{"name": k, "n_trips": len(v)} for k, v in DATASETS.items()], "default": DEFAULT}


@lru_cache(maxsize=None)
def _trip_summary(ds, tid):
    trip = load_trip(DATASETS[ds][tid], with_route=False)
    tr = trip.track
    return dict(trip_id=tid, n_points=len(tr), duration_min=round(float(tr.t.iloc[-1] - tr.t.iloc[0]) / 60, 1),
                n_road_arcs=len(trip.arcs), center=[round(float(tr.lat.mean()), 5), round(float(tr.lon.mean()), 5)])


@app.get("/trips")
def trips(dataset: str | None = Query(None, description="dataset name; default = first served")):
    ds = dataset or DEFAULT
    if ds not in DATASETS:
        raise HTTPException(404, f"unknown dataset {ds!r}")
    return {"dataset": ds, "trips": [_trip_summary(ds, t) for t in DATASETS[ds]]}


@app.get("/activity")
def activity():
    """Most recent predictions handled by this server (newest first)."""
    return {"recent": list(RECENT)[::-1]}


@app.post("/predict/{trip_id}")
def predict(trip_id: str, dataset: str | None = Query(None)):
    ds, d = _trip_dir(dataset, trip_id)
    t0 = time.perf_counter()
    trip = load_trip(d, with_route=False)              # prediction never sees .route
    pred, ctx = predictor.predict(trip)
    pred_ms = 1000 * (time.perf_counter() - t0)

    t1 = time.perf_counter()
    metrics = _jsonable(score(ctx, pred, load_reference_route(d, trip_id)))   # server-side only
    eval_ms = 1000 * (time.perf_counter() - t1)

    log.info("PREDICT dataset=%s trip=%s model=%s points=%d route_arcs=%d top1=%s route_f1=%s pred=%.0fms",
             ds, trip_id, predictor.model_name, metrics["n_points"], len(pred.route),
             f"{metrics['top1_hmm']:.4f}" if metrics.get("top1_hmm") is not None else "n/a",
             f"{metrics['route_f1_hmm']:.4f}" if metrics.get("route_f1_hmm") is not None else "n/a",
             pred_ms)
    rec = dict(ts=datetime.now(timezone.utc).isoformat(), event="predict", dataset=ds, trip=trip_id,
               model=predictor.model_name, pred_ms=round(pred_ms, 1), metrics=metrics)
    _append_log(rec)
    RECENT.append(dict(ts=rec["ts"], dataset=ds, trip=trip_id, pred_ms=rec["pred_ms"],
                       n_points=metrics["n_points"], top1=metrics.get("top1_hmm"),
                       route_f1=metrics.get("route_f1_hmm")))

    tr = trip.track
    return {
        "dataset": ds,
        "trip_id": trip_id,
        "model": predictor.model_name,
        "gps": np.round(tr[["lat", "lon"]].to_numpy(), 6).tolist(),
        "gps_t": np.round(tr.t.to_numpy() - tr.t.iloc[0], 1).tolist(),
        "matched": (pred.point_arc >= 0).astype(int).tolist(),
        "predicted_route_arcs": pred.route,
        "predicted_route_lines": route_polylines(trip, pred.route),
        "candidate_road_lines": arc_segments(trip, pred.candidate_arcs),
        "metrics": metrics,
        "timings_ms": {**pred.timings_ms, "prediction_total": round(pred_ms, 1),
                       "server_evaluation": round(eval_ms, 1),
                       "per_point": round(pred_ms / max(len(tr), 1), 3)},
    }
