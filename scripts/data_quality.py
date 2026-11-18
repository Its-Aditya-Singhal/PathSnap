"""Missing-value and data-quality checks on the raw trip files.

    python scripts/data_quality.py [--partitions train student_test]

Writes results/eda/data_quality_per_trip.csv and results/eda/data_quality_summary.csv.
Only train and student_test are analysed (evaluator data stays untouched).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.config import DATA_ROOT, RESULTS_DIR  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402

OUT = RESULTS_DIR / "eda"
FILES = {"track": ["lon", "lat", "t"], "nodes": ["lon", "lat"], "arcs": ["u", "v"], "route": ["arc"]}


def check_trip(d: Path, part: str) -> dict:
    tid = d.name
    row = dict(partition=part, trip=tid)
    raw = {}
    for ext, cols in FILES.items():
        p = d / f"{tid}.{ext}"
        row[f"{ext}_present"] = p.exists()
        if not p.exists():
            continue
        df = pd.read_csv(p, sep=r"\s+", header=None, names=cols)
        raw[ext] = df
        row[f"{ext}_rows"] = len(df)
        row[f"{ext}_missing"] = int(df.isna().sum().sum())
        row[f"{ext}_non_numeric"] = int(sum(pd.to_numeric(df[c], errors="coerce").isna().sum() for c in cols)
                                        - df.isna().sum().sum())
    tr = raw["track"]
    dt = np.diff(tr.t.to_numpy())
    row.update(
        track_dup_rows=int(tr.duplicated().sum()),
        track_dup_timestamps=int(tr.t.duplicated().sum()),
        track_non_increasing_t=int((dt <= 0).sum()),
        track_gaps_over_10s=int((dt > 10).sum()),
        track_gaps_over_60s=int((dt > 60).sum()),
        track_max_gap_s=float(dt.max()) if len(dt) else 0.0,
        track_out_of_range=int(((tr.lat.abs() > 90) | (tr.lon.abs() > 180)).sum()),
    )
    n_nodes = len(raw["nodes"])
    arcs = raw["arcs"]
    row["arcs_bad_node_ref"] = int(((arcs.u < 0) | (arcs.u >= n_nodes) | (arcs.v < 0) | (arcs.v >= n_nodes)).sum())
    row["arcs_self_loops"] = int((arcs.u == arcs.v).sum())
    if "route" in raw:
        r = raw["route"].arc.to_numpy()
        row["route_bad_arc_ref"] = int(((r < 0) | (r >= len(arcs))).sum())
        ok = (r >= 0) & (r < len(arcs))
        rr = r[ok]
        row["route_disconnected_steps"] = int((arcs.v.to_numpy()[rr[:-1]] != arcs.u.to_numpy()[rr[1:]]).sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    a = ap.parse_args()
    rows = [check_trip(d, p) for p in a.partitions for d in list_trips(p, DATA_ROOT)]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "data_quality_per_trip.csv", index=False)
    num = df.select_dtypes("number").columns
    summ = df.groupby("partition")[list(num)].sum().T
    summ.loc["trips"] = df.groupby("partition").size()
    summ.loc["track_max_gap_s"] = df.groupby("partition").track_max_gap_s.max()
    summ.loc["trips_with_gap_over_60s"] = df.assign(g=df.track_gaps_over_60s > 0).groupby("partition").g.sum()
    summ.to_csv(OUT / "data_quality_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
