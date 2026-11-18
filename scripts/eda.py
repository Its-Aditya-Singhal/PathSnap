"""Exploratory data analysis of the trips, networks and candidate set.

Everything is computed from the raw files and the candidate cache (no numbers
are typed in). Modelling decisions were made on train; student_test is shown
only as a descriptive comparison.

    python scripts/eda.py
Outputs:
  results/eda/per_trip_stats.csv, results/eda/eda_summary.json
  results/plots/eda_*.png
  results/maps/eda_train_<id>.html   GPS vs reference route (+ local network)
"""
import json

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import MAPS_DIR, PLOTS_DIR, RESULTS_DIR
from pathsnap.io import iter_trips
from pathsnap.pipeline import prepare

OUT = RESULTS_DIR / "eda"
BLUE, ORANGE, GRAY, GREEN = "#2a78d6", "#eb6834", "#52514e", "#1f9e6e"
GRID, MUTED = "#e4e3df", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
    "legend.frameon": False,
})


def save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name)
    plt.close(fig)
    print("wrote", PLOTS_DIR / name)


def trip_stats(trip):
    ctx = prepare(trip)
    tr = trip.track
    dt = np.diff(tr.t.to_numpy())
    step = np.hypot(np.diff(ctx.px), np.diff(ctx.py))
    speed = step / np.where(dt > 0, dt, np.nan)
    pair = set(zip(trip.arcs.u, trip.arcs.v))
    two_way = np.mean([(v, u) in pair for u, v in zip(trip.arcs.u, trip.arcs.v)])
    r = trip.route
    return dict(
        trip=trip.trip_id, partition=trip.partition, n_points=len(tr),
        duration_min=(tr.t.iloc[-1] - tr.t.iloc[0]) / 60,
        dt_median_s=float(np.median(dt)), dt_p95_s=float(np.percentile(dt, 95)),
        n_gaps_over_60s=int((dt > 60).sum()),
        speed_median_mps=float(np.nanmedian(speed)), speed_p95_mps=float(np.nanpercentile(speed, 95)),
        gps_path_km=float(step.sum() / 1000),
        n_nodes=len(trip.nodes), n_arcs=len(trip.arcs), two_way_arc_frac=float(two_way),
        arc_len_median_m=float(np.median(ctx.net.length)),
        route_arcs=int(len(r)), route_km=float(ctx.net.length[r].sum() / 1000),
        route_unique_arcs=int(len(np.unique(r))),
        lat=float(tr.lat.mean()), lon=float(tr.lon.mean()),
    ), ctx, dt, speed


def candidate_stats(part):
    df = pd.read_pickle(RESULTS_DIR / "cache" / f"{part}_candidates.pkl")
    per_pt = df.groupby(["trip", "point"]).agg(n=("arc", "size"), has_true=("y", "max"),
                                               labelled=("label_arc", "max"))
    lab = per_pt[per_pt.labelled >= 0]
    pos = df[df.y == 1]
    return df, dict(
        candidate_rows=int(len(df)), points_with_candidates=int(len(per_pt)),
        cands_per_point_mean=float(per_pt.n.mean()),
        labelled_points=int(len(lab)),
        candidate_recall=float(lab.has_true.mean()),       # true arc inside the k=8 / 50 m set
        positive_rate=float(df.y.mean()),
        gps_error_median_m=float(pos.dist.median()), gps_error_p95_m=float(pos.dist.quantile(0.95)),
        true_is_nearest=float((pos["rank"] == 0).mean()),
    )


def eda_map(trip, ctx, path):
    tr = trip.track
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    net = folium.FeatureGroup("Road network (trip file)", show=True)
    lat, lon = trip.nodes.lat.to_numpy(), trip.nodes.lon.to_numpy()
    # draw only arcs near the track to keep the file small
    hits = ctx.net.tree.query_ball_point(np.c_[ctx.px, ctx.py], r=150)
    near = np.unique(ctx.net.sample_arc[np.concatenate([np.asarray(h, int) for h in hits])])
    for a in near:
        u, v = trip.arcs.u.iat[a], trip.arcs.v.iat[a]
        folium.PolyLine([(lat[u], lon[u]), (lat[v], lon[v])], color=GRAY, weight=1, opacity=0.5).add_to(net)
    net.add_to(fmap)
    ref = folium.FeatureGroup("Reference route (ground truth)")
    for a in trip.route:
        u, v = trip.arcs.u.iat[a], trip.arcs.v.iat[a]
        folium.PolyLine([(lat[u], lon[u]), (lat[v], lon[v])], color=GREEN, weight=5, opacity=0.8).add_to(ref)
    ref.add_to(fmap)
    g = folium.FeatureGroup("GPS points")
    for la, lo in zip(tr.lat, tr.lon):
        folium.CircleMarker((la, lo), radius=2, color=BLUE, weight=0, fill=True, fill_opacity=0.7).add_to(g)
    g.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    fmap.save(str(path))
    print("wrote", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    rows, dts, speeds, keep = [], {"train": [], "student_test": []}, {"train": [], "student_test": []}, {}
    for trip in iter_trips():
        s, ctx, dt, sp = trip_stats(trip)
        rows.append(s)
        dts[trip.partition].append(dt)
        speeds[trip.partition].append(sp[np.isfinite(sp)])
        if trip.partition == "train":
            keep[trip.trip_id] = (trip, ctx, s)
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "per_trip_stats.csv", index=False)

    summary = {}
    for part, g in st.groupby("partition"):
        df, cs = candidate_stats(part)
        summary[part] = dict(
            n_trips=int(len(g)), total_points=int(g.n_points.sum()),
            points_per_trip_median=float(g.n_points.median()),
            duration_min_median=float(g.duration_min.median()),
            dt_median_s=float(np.median(np.concatenate(dts[part]))),
            trips_with_gaps_over_60s=int((g.n_gaps_over_60s > 0).sum()),
            speed_median_mps=float(np.median(np.concatenate(speeds[part]))),
            arcs_per_network_median=float(g.n_arcs.median()),
            route_km_median=float(g.route_km.median()),
            two_way_arc_frac_mean=float(g.two_way_arc_frac.mean()),
            **cs)
        if part == "train":
            train_df = df
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # 1. where the trips are
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for part, c in (("train", BLUE), ("student_test", ORANGE)):
        g = st[st.partition == part]
        ax.scatter(g.lon, g.lat, s=18, color=c, label=f"{part} ({len(g)} trips)", alpha=0.8,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Trip locations (mean GPS position): each trip has its own local network")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "eda_trip_locations.png")

    # 2. trip size / sampling / speed
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.0), gridspec_kw=dict(wspace=0.35))
    tr_st = st[st.partition == "train"]
    axs[0].hist(tr_st.n_points, bins=20, color=BLUE)
    axs[0].set_title("GPS points per trip"); axs[0].set_xlabel("points")
    dt_all = np.concatenate(dts["train"])
    axs[1].hist(np.clip(dt_all, 0, 30), bins=30, color=BLUE)
    axs[1].set_title("Sampling interval (clipped at 30 s)"); axs[1].set_xlabel("seconds")
    sp_all = np.concatenate(speeds["train"])
    axs[2].hist(np.clip(sp_all, 0, 50), bins=40, color=BLUE)
    axs[2].set_title("Implied speed (clipped at 50 m/s)"); axs[2].set_xlabel("m/s")
    for a in axs:
        a.set_ylabel("count")
    fig.suptitle("Train trips (80)", fontsize=10, fontweight="bold", y=1.04)
    save(fig, "eda_trips.png")

    # 3. GPS error & candidate set
    pos = train_df[train_df.y == 1]
    per_pt = train_df[train_df.label_arc >= 0].groupby(["trip", "point"]).arc.size()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.0), gridspec_kw=dict(wspace=0.35))
    axs[0].hist(pos.dist, bins=50, range=(0, 50), color=BLUE)
    axs[0].axvline(pos.dist.median(), color=ORANGE, lw=2)
    axs[0].annotate(f"median {pos.dist.median():.1f} m", (pos.dist.median(), axs[0].get_ylim()[1] * 0.9),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED)
    axs[0].set_title("GPS distance to the true arc"); axs[0].set_xlabel("metres")
    vc = per_pt.value_counts().sort_index()
    axs[1].bar(vc.index, vc.values, color=BLUE, width=0.7)
    axs[1].set_title("Candidates per point (r=50 m, k=8)"); axs[1].set_xlabel("candidates")
    rk = pos["rank"].value_counts(normalize=True).sort_index()
    axs[2].bar(rk.index, rk.values, color=BLUE, width=0.7)
    axs[2].set_title("Distance rank of the true arc"); axs[2].set_xlabel("rank (0 = nearest)")
    axs[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axs[0].set_ylabel("points"); axs[1].set_ylabel("points"); axs[2].set_ylabel("share of points")
    save(fig, "eda_candidates.png")

    # 4. key features: true vs other candidates
    lab = train_df[train_df.label_arc >= 0]
    feats = [("dist", "distance to arc (m)", (0, 50)), ("heading_diff", "heading difference (deg)", (0, 180)),
             ("dist_minus_min", "distance minus nearest (m)", (0, 40)), ("win_dist_mean", "window mean distance (m)", (0, 50))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.6))
    for a, (f, lbl, rng) in zip(axs, feats):
        for yv, c, name in ((0, GRAY, "other candidate"), (1, BLUE, "true arc")):
            a.hist(lab.loc[lab.y == yv, f].clip(*rng), bins=40, range=rng, density=True, color=c,
                   alpha=0.6 if yv == 0 else 0.75, label=name)
        a.set_title(lbl, fontsize=9); a.set_yticks([])
    axs[0].legend(fontsize=7)
    fig.suptitle("Feature distributions: true arc vs other candidates (train)", fontsize=10,
                 fontweight="bold", y=1.06)
    save(fig, "eda_features.png")

    # 5. Folium maps for three train trips (short / median / long)
    ordered = sorted(keep.values(), key=lambda r: r[2]["n_points"])
    for trip, ctx, s in (ordered[0], ordered[len(ordered) // 2], ordered[-1]):
        eda_map(trip, ctx, MAPS_DIR / f"eda_train_{trip.trip_id}.html")


if __name__ == "__main__":
    main()
