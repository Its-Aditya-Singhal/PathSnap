"""Build the (point, candidate) feature tables for train and student_test.

Output: results/cache/<partition>_candidates.pkl with columns
  trip, point, arc, <FEATURES...>, y (1 if arc == reference label), label_arc
Points without a reference label (label_arc == -1) are kept but flagged;
training drops them.

    python scripts/build_dataset.py [--radius 50 --k 8]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.config import RESULTS_DIR
from pathsnap.features import FEATURES, build_features
from pathsnap.io import iter_trips
from pathsnap.labels import derive_labels
from pathsnap.pipeline import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    out = RESULTS_DIR / "cache"
    out.mkdir(parents=True, exist_ok=True)

    for part in ("train", "student_test"):
        tables, t0 = [], time.perf_counter()
        for trip in iter_trips((part,)):
            ctx = prepare(trip)
            lab = derive_labels(ctx)
            cands = generate_candidates(ctx.px, ctx.py, ctx.net, radius=args.radius, k=args.k)
            X = build_features(ctx, cands)
            df = pd.concat([cands[["point", "arc"]], X], axis=1)
            df.insert(0, "trip", trip.trip_id)
            df["label_arc"] = lab.arc[df.point.to_numpy()]
            df["y"] = (df.arc.to_numpy() == df.label_arc.to_numpy()).astype(int)
            tables.append(df)
        full = pd.concat(tables, ignore_index=True)
        full.to_pickle(out / f"{part}_candidates.pkl")
        print(f"{part}: {full.trip.nunique()} trips, {len(full):,} rows, "
              f"pos rate {full.y.mean():.3f}, {time.perf_counter() - t0:.0f}s")
    (out / "dataset_config.json").write_text(json.dumps(
        {"radius": args.radius, "k": args.k, "features": FEATURES}, indent=2))


if __name__ == "__main__":
    main()
