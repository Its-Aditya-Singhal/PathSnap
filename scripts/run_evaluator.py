"""Run the frozen PathSnap model on an unseen partition and record the results.

The model bundle (artifacts/pathsnap_model.joblib) was selected and frozen
before this partition was opened; nothing here trains, tunes or selects.
Prediction uses `with_route=False`; the .route file is read afterwards,
only to score the prediction.

    python scripts/run_evaluator.py [--partition evaluator_test]
Writes results/evaluation/<partition>_per_trip.csv and <partition>_summary.json
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pathsnap.config import DATA_ROOT, RESULTS_DIR
from pathsnap.inference import Predictor
from pathsnap.io import list_trips, load_trip
from pathsnap.serve_eval import load_reference_route, score

OUT = RESULTS_DIR / "evaluation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="evaluator_test")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pr = Predictor()
    rows = []
    for d in list_trips(a.partition, DATA_ROOT):
        trip = load_trip(d, a.partition, with_route=False)
        t0 = time.perf_counter()
        pred, ctx = pr.predict(trip)
        ms = 1000 * (time.perf_counter() - t0)
        m = score(ctx, pred, load_reference_route(d, d.name))
        rows.append(dict(trip_id=d.name, **m, route_arcs=len(pred.route), pred_ms=round(ms, 1)))
        print(f"  {d.name}: points={m['n_points']} top1={m.get('top1_hmm', np.nan):.4f} "
              f"routeF1={m.get('route_f1_hmm', np.nan):.4f} cont={m['continuity_hmm']:.4f} {ms:.0f}ms")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{a.partition}_per_trip.csv", index=False)

    w = df.get("n_labelled")
    def pooled(col):
        return float(np.average(df[col], weights=w)) if w is not None and col in df else None
    summary = dict(
        partition=a.partition, model=pr.model_name, created=datetime.now(timezone.utc).isoformat(),
        n_trips=len(df), n_points=int(df.n_points.sum()),
        top1_hmm_pooled=pooled("top1_hmm"), top1_nearest_pooled=pooled("top1_clf"),
        route_f1_hmm_mean=float(df.route_f1_hmm.mean()) if "route_f1_hmm" in df else None,
        route_precision_hmm_mean=float(df.route_precision_hmm.mean()) if "route_precision_hmm" in df else None,
        route_recall_hmm_mean=float(df.route_recall_hmm.mean()) if "route_recall_hmm" in df else None,
        continuity_hmm_mean=float(df.continuity_hmm.mean()),
        pred_ms_mean=float(df.pred_ms.mean()),
        ms_per_point=float(df.pred_ms.sum() / df.n_points.sum()),
    )
    (OUT / f"{a.partition}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
