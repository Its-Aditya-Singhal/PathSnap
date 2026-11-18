"""Baseline: nearest-arc vs HMM/Viterbi (Gaussian distance emission).

Also derives and saves the per-point reference labels used by every later
model. Runs on train + student_test only.

    python scripts/run_hmm_baseline.py [--beta 5 --sigma 5 --radius 50 --k 8]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.config import RESULTS_DIR
from pathsnap.hmm import HMMConfig, ViterbiDecoder, gaussian_log_emission, reconstruct_route
from pathsnap.io import iter_trips
from pathsnap.labels import derive_labels
from pathsnap.metrics import candidate_miss_rate, route_continuity, route_overlap, top1_accuracy
from pathsnap.pipeline import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gap", type=float, default=60.0)
    args = ap.parse_args()
    cfg = HMMConfig(beta=args.beta, sigma=args.sigma, gap_s=args.gap)

    lab_dir = RESULTS_DIR / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    rows, label_tables = [], {}
    for trip in iter_trips():
        t0 = time.perf_counter()
        ctx = prepare(trip)
        lab = derive_labels(ctx)
        cands = generate_candidates(ctx.px, ctx.py, ctx.net, radius=args.radius, k=args.k)

        nearest = np.full(len(ctx.px), -1)
        first = cands[cands["rank"] == 0]
        nearest[first.point.to_numpy()] = first.arc.to_numpy()

        t1 = time.perf_counter()
        res = ViterbiDecoder(ctx.net, cfg).decode(ctx.px, ctx.py, ctx.ts, cands,
                                                  gaussian_log_emission(cands, cfg.sigma))
        dec_s = time.perf_counter() - t1
        route = reconstruct_route(ctx.net, res.arc, res.segment)

        rows.append(dict(
            partition=trip.partition, trip=trip.trip_id, n_points=len(ctx.px),
            labelled_frac=float((lab.arc >= 0).mean()),
            label_dist_p50=float(np.nanmedian(lab.dist)),
            cands_per_point=len(cands) / len(ctx.px),
            candidate_miss_rate=candidate_miss_rate(cands, lab.arc),
            top1_nearest=top1_accuracy(nearest, lab.arc),
            top1_hmm=top1_accuracy(res.arc, lab.arc),
            continuity_nearest=route_continuity(ctx.net, nearest, ctx.px, ctx.py),
            continuity_hmm=route_continuity(ctx.net, res.arc, ctx.px, ctx.py),
            n_breaks=res.n_breaks, decode_s=dec_s, total_s=time.perf_counter() - t0,
            **route_overlap(ctx.net, route, trip.route),
        ))
        label_tables.setdefault(trip.partition, []).append(pd.DataFrame(
            {"trip": trip.trip_id, "point": np.arange(len(ctx.px)),
             "label_arc": lab.arc, "label_dist": lab.dist}))
        r = rows[-1]
        print(f"{trip.partition:12s} {trip.trip_id} n={r['n_points']:6d} "
              f"top1 nearest={r['top1_nearest']:.3f} hmm={r['top1_hmm']:.3f} "
              f"routeF1={r['route_f1']:.3f} {r['total_s']:.1f}s", flush=True)

    for part, tables in label_tables.items():
        pd.concat(tables).to_csv(lab_dir / f"{part}_point_labels.csv.gz", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hmm_baseline_per_trip.csv", index=False)
    summary = df.groupby("partition").mean(numeric_only=True)
    summary.to_csv(RESULTS_DIR / "hmm_baseline_summary.csv")
    (RESULTS_DIR / "hmm_baseline_config.json").write_text(json.dumps(
        {**cfg.to_dict(), "radius": args.radius, "k": args.k}, indent=2))
    print("\n", summary.T.round(4).to_string())


if __name__ == "__main__":
    main()
