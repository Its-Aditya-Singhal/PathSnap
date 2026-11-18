"""Folium maps of the exported model's predictions on student_test.

For each trip: raw GPS (blue), reference route (green) and the predicted
route (orange). The reference is read only for drawing/scoring, after the
prediction is made with `with_route=False`.

    python scripts/make_maps.py [--partition student_test]
Writes results/maps/pred_<partition>_<id>.html, results/maps/index_<partition>.html
and results/maps/index.html (links to every partition index).
"""
import argparse

import folium

from pathsnap.config import DATA_ROOT, MAPS_DIR
from pathsnap.inference import Predictor, route_polylines
from pathsnap.io import list_trips, load_trip
from pathsnap.serve_eval import load_reference_route, score

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1f9e6e"


def trip_map(trip, pred_route, ref_route, m):
    tr = trip.track
    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if ref_route is not None:
        g = folium.FeatureGroup("Reference route")
        for line in route_polylines(trip, list(ref_route)):
            folium.PolyLine(line, color=GREEN, weight=8, opacity=0.6).add_to(g)
        g.add_to(fmap)
    g = folium.FeatureGroup("Predicted route")
    for line in route_polylines(trip, pred_route):
        folium.PolyLine(line, color=ORANGE, weight=3.5, opacity=0.95).add_to(g)
    g.add_to(fmap)
    g = folium.FeatureGroup("GPS points")
    for la, lo in zip(tr.lat, tr.lon):
        folium.CircleMarker((la, lo), radius=2, color=BLUE, weight=0, fill=True, fill_opacity=0.6).add_to(g)
    g.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    title = (f"<b>Trip {trip.trip_id}</b> &middot; Top-1 {m.get('top1_hmm', float('nan')):.3f} "
             f"&middot; route F1 {m.get('route_f1_hmm', float('nan')):.3f} &middot; "
             f"<span style='color:{GREEN}'>reference</span> / <span style='color:{ORANGE}'>predicted</span>"
             f" / <span style='color:{BLUE}'>GPS</span>")
    fmap.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:6px 10px;"
        f"border-radius:4px;font:13px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'>{title}</div>"))
    fmap.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    return fmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    a = ap.parse_args()
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    pr = Predictor()
    print("model:", pr.model_name)
    links = []
    for d in list_trips(a.partition, DATA_ROOT):
        trip = load_trip(d, a.partition, with_route=False)
        pred, ctx = pr.predict(trip)
        ref = load_reference_route(d, d.name)
        m = score(ctx, pred, ref)
        out = MAPS_DIR / f"pred_{a.partition}_{d.name}.html"
        trip_map(trip, pred.route, ref, m).save(str(out))
        links.append((d.name, out.name, m))
        print(f"  {d.name}: top1={m.get('top1_hmm', float('nan')):.4f} "
              f"routeF1={m.get('route_f1_hmm', float('nan')):.4f} -> {out.name}")
    rows = "".join(f"<tr><td><a href='{f}'>{t}</a></td><td>{m.get('top1_hmm', float('nan')):.3f}</td>"
                   f"<td>{m.get('route_f1_hmm', float('nan')):.3f}</td></tr>" for t, f, m in links)
    (MAPS_DIR / f"index_{a.partition}.html").write_text(
        f"<html><body style='font:14px sans-serif'><p><a href='index.html'>all partitions</a></p>"
        f"<h3>PathSnap predictions ({a.partition}, "
        f"model {pr.model_name})</h3><table cellpadding=4><tr><th>trip</th><th>Top-1</th>"
        f"<th>route F1</th></tr>{rows}</table></body></html>")
    parts = sorted(p.stem.removeprefix("index_") for p in MAPS_DIR.glob("index_*.html"))
    eda = sorted(p.name for p in MAPS_DIR.glob("eda_*.html"))
    (MAPS_DIR / "index.html").write_text(
        "<html><body style='font:14px sans-serif'><h3>PathSnap maps</h3><h4>Predictions</h4><ul>"
        + "".join(f"<li><a href='index_{q}.html'>{q}</a></li>" for q in parts)
        + "</ul><h4>EDA (train)</h4><ul>" + "".join(f"<li><a href='{e}'>{e}</a></li>" for e in eda)
        + "</ul></body></html>")


if __name__ == "__main__":
    main()
