"""PathSnap evaluator console (Laptop 2).

    streamlit run ui/app.py
    PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py

The UI talks to the API only. It has no data files, no model and no
reference route: everything shown comes from API responses.
"""
import os
import time
from datetime import datetime

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

GPS_C, ROUTE_C, ROAD_C, INK, MUTED = "#2a78d6", "#e8590c", "#8a929c", "#1b1f24", "#5f6874"
TILES = {
    "Light": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
              "Esri, HERE, Garmin, OpenStreetMap contributors"),
    "Streets": ("OpenStreetMap", None),
    "Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "Esri World Imagery"),
}

st.set_page_config(page_title="PathSnap · Evaluator console", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}
.ps-header {display:flex; align-items:center; justify-content:space-between; gap:16px;
  background:#ffffff; border:1px solid #e3e6ea; border-radius:14px; padding:16px 22px; margin-bottom:14px;}
.ps-brand {display:flex; align-items:center; gap:14px;}
.ps-logo {width:42px; height:42px; border-radius:11px; background:linear-gradient(135deg,#1f5fbf,#2a78d6);
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:19px;}
.ps-title {font-size:21px; font-weight:700; color:#1b1f24; line-height:1.15;}
.ps-sub {font-size:13px; color:#5f6874;}
.ps-pill {display:inline-flex; align-items:center; gap:8px; font-size:12.5px; color:#1b1f24;
  background:#f1f4f8; border:1px solid #e3e6ea; border-radius:999px; padding:6px 12px; margin-left:6px;}
.ps-dot {width:8px; height:8px; border-radius:50%; background:#1f9e6e; box-shadow:0 0 0 3px #1f9e6e22;}
.ps-dot.off {background:#c92a2a; box-shadow:0 0 0 3px #c92a2a22;}
.ps-kpi {background:#fff; border:1px solid #e3e6ea; border-radius:12px; padding:14px 16px; min-height:112px;}
.ps-kpi .l {white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:12px; color:#5f6874; text-transform:uppercase; letter-spacing:.04em; font-weight:600;}
.ps-kpi .v {white-space:nowrap; font-size:25px; font-weight:700; color:#1b1f24; margin-top:2px; font-variant-numeric:tabular-nums;}
.ps-kpi .s {font-size:12.5px; color:#5f6874; margin-top:2px;}
.ps-kpi .s b {color:#1f7a55;}
.ps-card {background:#fff; border:1px solid #e3e6ea; border-radius:12px; padding:14px 18px;}
.ps-legend {display:flex; gap:18px; flex-wrap:wrap; font-size:12.5px; color:#5f6874; margin:6px 2px 0;}
.ps-legend span {display:inline-flex; align-items:center; gap:6px;}
.ps-sw {display:inline-block; width:18px; height:4px; border-radius:2px;}
.ps-swd {display:inline-block; width:9px; height:9px; border-radius:50%;}
.ps-steps {display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:6px 0 12px;}
.ps-step {background:#fff; border:1px solid #cfd8e3; border-radius:10px; padding:8px 12px; font-size:13px;}
.ps-step b {color:#1f5fbf; margin-right:4px;}
.ps-arrow {color:#8a929c;}
.ps-empty {background:#fff; border:1px dashed #cfd5dc; border-radius:12px; padding:48px 24px;
  text-align:center; color:#5f6874;}
.ps-empty h4 {color:#1b1f24; margin-bottom:6px;}
[data-testid="stSidebar"] {background:#ffffff; border-right:1px solid #e3e6ea;}
[data-testid="stTabs"] button p {font-size:14.5px; font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API helpers
def api_get(path, **params):
    r = requests.get(f"{st.session_state.api}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120, show_spinner=False)
def get_trips(api_url, dataset):
    r = requests.get(f"{api_url}/trips", params={"dataset": dataset}, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json()["trips"])


def predict(dataset, trip_id):
    t0 = time.perf_counter()
    r = requests.post(f"{st.session_state.api}/predict/{trip_id}", params={"dataset": dataset}, timeout=600)
    rtt = 1000 * (time.perf_counter() - t0)
    r.raise_for_status()
    return {**r.json(), "round_trip_ms": rtt}


def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f}%"


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='ps-kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Connection")
    st.session_state.api = st.text_input(
        "API URL", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"),
        help="Laptop 1 address, e.g. http://192.168.1.20:8000").rstrip("/")
    try:
        health = api_get("/health")
        online = True
    except Exception as e:  # noqa: BLE001
        health, online, err = None, False, e

    if online:
        sets = list(health["datasets"])
        st.markdown("### Dataset")
        dataset = st.selectbox("Dataset served by the API", sets,
                               index=sets.index(health["default_dataset"]) if health["default_dataset"] in sets else 0,
                               format_func=lambda s: f"{s}  ·  {health['datasets'][s]} trips",
                               label_visibility="collapsed")
        st.markdown("### Map")
        basemap = st.segmented_control("Basemap", list(TILES), default="Light", label_visibility="collapsed") or "Light"
        show_gps = st.toggle("Raw GPS points", value=True)
        show_cands = st.toggle("Candidate roads (within 50 m)", value=True)
        st.divider()
        if st.button("Refresh connection", width="stretch"):
            get_trips.clear()
            st.rerun()
        st.caption("This console holds no data or model. Every number on screen is returned by the API.")

# ------------------------------------------------------------------ header
if online:
    model = health["model"]["model_name"]
    status = (f"<span class='ps-pill'><span class='ps-dot'></span>API online</span>"
              f"<span class='ps-pill'>Model&nbsp;<b>{model}</b></span>"
              f"<span class='ps-pill'>{st.session_state.api}</span>")
else:
    status = "<span class='ps-pill'><span class='ps-dot off'></span>API offline</span>"
st.markdown(f"""
<div class='ps-header'>
  <div class='ps-brand'><div class='ps-logo'>P</div>
    <div><div class='ps-title'>PathSnap</div>
    <div class='ps-sub'>GPS map-matching &amp; route reconstruction · Hidden Markov Model + Viterbi</div></div></div>
  <div>{status}</div>
</div>""", unsafe_allow_html=True)

if not online:
    st.error(f"Cannot reach the API at **{st.session_state.api}**.\n\n"
             "Start it on Laptop 1 with `uvicorn api.main:app --host 0.0.0.0 --port 8000`, "
             f"then set the URL in the sidebar.\n\n`{err}`")
    st.stop()

trips = get_trips(st.session_state.api, dataset)
st.session_state.setdefault("results", {})
st.session_state.setdefault("batch", {})


# ------------------------------------------------------------------ map
def build_map(res):
    tiles, attr = TILES[basemap]
    gps = res["gps"]
    fmap = folium.Map(tiles=tiles, attr=attr, control_scale=True, zoom_control=True, prefer_canvas=True,
                      scrollWheelZoom=False)
    if show_cands and res.get("candidate_road_lines"):
        g = folium.FeatureGroup("Candidate roads")
        for seg in res["candidate_road_lines"]:
            folium.PolyLine(seg, color=ROAD_C, weight=2, opacity=0.55).add_to(g)
        g.add_to(fmap)
    if show_gps:
        g = folium.FeatureGroup("GPS points")
        step = max(1, len(gps) // 2500)
        ts = res.get("gps_t") or [None] * len(gps)
        for i in range(0, len(gps), step):
            lat, lon = gps[i]
            folium.CircleMarker((lat, lon), radius=2, color=GPS_C, weight=0, fill=True, fill_opacity=0.5,
                                tooltip=f"point {i} · t+{ts[i]:.0f}s" if ts[i] is not None else None).add_to(g)
        g.add_to(fmap)
    g = folium.FeatureGroup("Predicted route")
    for line in res["predicted_route_lines"]:
        folium.PolyLine(line, color="#ffffff", weight=9, opacity=0.9).add_to(g)
        folium.PolyLine(line, color=ROUTE_C, weight=5, opacity=0.95,
                        tooltip=f"Predicted route · {len(res['predicted_route_arcs'])} arcs").add_to(g)
    g.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    pts = [p for line in res["predicted_route_lines"] for p in line] + gps
    fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                     [max(p[0] for p in pts), max(p[1] for p in pts)]], padding=(20, 20))
    return fmap


def legend():
    st.markdown(f"""<div class='ps-legend'>
      <span><span class='ps-sw' style='background:{ROUTE_C}'></span>Predicted route</span>
      <span><span class='ps-swd' style='background:{GPS_C}'></span>Raw GPS</span>
      <span><span class='ps-sw' style='background:{ROAD_C}'></span>Candidate roads</span>
      <span>● Start (green) · ■ End (red)</span></div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ tabs
tab_trip, tab_batch, tab_model, tab_log = st.tabs(
    ["Trip reconstruction", "Batch evaluation", "Model & method", "Server activity"])

# ---------------------------------------------------------------- trip tab
with tab_trip:
    left, right = st.columns([1, 3.2], gap="large")
    with left:
        st.markdown("#### Trips")
        st.caption(f"Dataset: {dataset} · {len(trips)} trips")
        if trips.empty:
            st.warning("The API serves no trips for this dataset.")
            st.stop()
        labels = {r.trip_id: f"{r.trip_id}  ·  {r.n_points:,} pts"
                  for r in trips.itertuples()}
        trip_id = st.selectbox("Trip", list(labels), format_func=labels.get, key=f"trip_{dataset}")
        info = trips.set_index("trip_id").loc[trip_id]
        st.markdown(f"""<div class='ps-card' style='font-size:13.5px; line-height:1.9'>
            <b>GPS points</b><span style='float:right'>{info.n_points:,}</span><br>
            <b>Duration</b><span style='float:right'>{info.duration_min:.1f} min</span><br>
            <b>Road arcs in network</b><span style='float:right'>{info.n_road_arcs:,}</span><br>
            <b>Centre</b><span style='float:right'>{info.center[0]:.3f}, {info.center[1]:.3f}</span>
            </div>""", unsafe_allow_html=True)
        st.write("")
        run = st.button("Reconstruct route", type="primary", width="stretch", icon=":material/route:")
        key = (dataset, trip_id)
        if run:
            with st.spinner(f"API is matching trip {trip_id}…"):
                try:
                    st.session_state.results[key] = predict(dataset, trip_id)
                except Exception as e:  # noqa: BLE001
                    st.error(f"API error: {e}")
        done = [t for (d, t) in st.session_state.results if d == dataset]
        if done:
            st.caption("Reconstructed this session: " + ", ".join(done))

    res = st.session_state.results.get((dataset, trip_id))
    with right:
        if res is None:
            st.markdown("""<div class='ps-empty'><h4>No reconstruction yet</h4>
                Choose a trip and press <b>Reconstruct route</b>. The API projects the GPS points,
                finds candidate roads, runs the HMM / Viterbi decoder and scores the result on the server.
                </div>""", unsafe_allow_html=True)
        else:
            m, tm = res["metrics"], res["timings_ms"]
            c = st.columns(5)
            gain = (m["top1_hmm"] - m["top1_clf"]) if m.get("top1_hmm") is not None and m.get("top1_clf") is not None else None
            kpi(c[0], "Top-1 accuracy", pct(m.get("top1_hmm")),
                f"<b>+{100 * gain:.1f} pts</b> vs nearest road" if gain is not None else "no reference route")
            kpi(c[1], "Route F1", pct(m.get("route_f1_hmm")),
                f"P {pct(m.get('route_precision_hmm'))} · R {pct(m.get('route_recall_hmm'))}")
            kpi(c[2], "Continuity", pct(m.get("continuity_hmm")), f"nearest road {pct(m.get('continuity_clf'))}")
            kpi(c[3], "Latency", f"{tm['prediction_total']:.0f} ms", f"{tm['per_point']:.2f} ms per point")
            kpi(c[4], "Route", f"{len(res['predicted_route_arcs']):,} arcs",
                f"{m['n_points']:,} points · {m['n_breaks']} breaks")
            st.write("")
            components.html(build_map(res).get_root().render(), height=600)
            legend()
            st.write("")

            a, b = st.columns([1.15, 1], gap="large")
            with a:
                st.markdown("##### Nearest road vs HMM / Viterbi")
                rows = [("Top-1 accuracy", "top1"), ("Route F1", "route_f1"), ("Continuity", "continuity")]
                cmp = pd.DataFrame([{"Metric": n, "Nearest road per point": pct(m.get(f"{k}_clf")),
                                     "HMM / Viterbi": pct(m.get(f"{k}_hmm"))} for n, k in rows])
                st.dataframe(cmp, hide_index=True, width="stretch")
                st.caption("Nearest road = choose the closest arc for each point independently. "
                           "HMM / Viterbi = choose the most likely connected sequence. Scores are computed by "
                           "the API against the reference route, which never leaves Laptop 1.")
            with b:
                st.markdown("##### Where the time goes")
                stages = {"prepare": "Load & project", "candidates": "Candidate search",
                          "features_and_model": "Emission scores", "viterbi_and_route": "Viterbi + route"}
                tdf = pd.DataFrame([{"Stage": v, "ms": tm.get(k, 0.0)} for k, v in stages.items()])
                chart = alt.Chart(tdf).mark_bar(color="#1f5fbf", cornerRadiusEnd=4, height=18).encode(
                    x=alt.X("ms:Q", title="milliseconds", axis=alt.Axis(grid=True, gridColor="#eef0f3")),
                    y=alt.Y("Stage:N", sort=list(stages.values()), title=None),
                    tooltip=["Stage", alt.Tooltip("ms:Q", format=".1f")])
                text = chart.mark_text(align="left", dx=4, color=MUTED, fontSize=11).encode(
                    text=alt.Text("ms:Q", format=".0f"))
                st.altair_chart((chart + text).properties(height=170), width="stretch")
                st.caption(f"Round trip UI ↔ API: {res['round_trip_ms']:.0f} ms "
                           f"(server scoring {tm['server_evaluation']:.0f} ms).")
            with st.expander("Raw API response (metrics & timings)"):
                st.json({"dataset": res["dataset"], "trip_id": res["trip_id"], "model": res["model"],
                         "metrics": m, "timings_ms": tm})

# ---------------------------------------------------------------- batch tab
with tab_batch:
    st.markdown(f"#### Evaluate every trip in `{dataset}`")
    st.caption("Sends each trip to the API one by one and aggregates the scores it returns. "
               "Top-1 is pooled over all scored GPS points; route metrics are averaged per trip.")
    go = st.button(f"Run all {len(trips)} trips", type="primary", icon=":material/play_arrow:")
    if go:
        rows, bar = [], st.progress(0.0, text="Starting…")
        for i, tid in enumerate(trips.trip_id):
            bar.progress(i / len(trips), text=f"Trip {tid} ({i + 1}/{len(trips)})")
            try:
                r = predict(dataset, tid)
                st.session_state.results[(dataset, tid)] = r
                rows.append(dict(trip_id=tid, **r["metrics"], route_arcs=len(r["predicted_route_arcs"]),
                                 pred_ms=r["timings_ms"]["prediction_total"]))
            except Exception as e:  # noqa: BLE001
                st.error(f"{tid}: {e}")
        bar.progress(1.0, text="Done")
        st.session_state.batch[dataset] = dict(df=pd.DataFrame(rows), at=datetime.now().strftime("%H:%M:%S"))

    bt = st.session_state.batch.get(dataset)
    if bt is None:
        st.markdown("<div class='ps-empty'><h4>No batch run yet</h4>Press the button to score the whole "
                    "dataset through the API.</div>", unsafe_allow_html=True)
    else:
        df = bt["df"]
        has_ref = "top1_hmm" in df and df.top1_hmm.notna().any()
        c = st.columns(5)
        if has_ref:
            w = df.n_labelled
            pooled = (df.top1_hmm * w).sum() / w.sum()
            pooled_nn = (df.top1_clf * w).sum() / w.sum()
            kpi(c[0], "Top-1 (pooled)", pct(pooled), f"<b>+{100 * (pooled - pooled_nn):.1f} pts</b> vs nearest road")
            kpi(c[1], "Route F1 (mean)", pct(df.route_f1_hmm.mean()),
                f"P {pct(df.route_precision_hmm.mean())} · R {pct(df.route_recall_hmm.mean())}")
        else:
            kpi(c[0], "Top-1", "—", "no reference routes")
            kpi(c[1], "Route F1", "—", "no reference routes")
        kpi(c[2], "Continuity (mean)", pct(df.continuity_hmm.mean(), 2), "connected consecutive matches")
        kpi(c[3], "GPS points", f"{int(df.n_points.sum()):,}", f"{len(df)} trips")
        kpi(c[4], "Mean prediction", f"{df.pred_ms.mean():.0f} ms",
            f"{df.pred_ms.sum() / df.n_points.sum():.2f} ms per point")
        st.write("")
        g, t = st.columns([1, 1.25], gap="large")
        if has_ref:
            with g:
                st.markdown("##### Top-1 accuracy per trip")
                base = alt.Chart(df).encode(y=alt.Y("trip_id:N", title=None, sort=None))
                bars = base.mark_bar(color="#1f5fbf", cornerRadiusEnd=4, height=14).encode(
                    x=alt.X("top1_hmm:Q", title="Top-1 accuracy (HMM / Viterbi)", scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%", grid=True, gridColor="#eef0f3")),
                    tooltip=[alt.Tooltip("trip_id:N", title="Trip"),
                             alt.Tooltip("top1_hmm:Q", title="Top-1 HMM", format=".1%"),
                             alt.Tooltip("top1_clf:Q", title="Top-1 nearest", format=".1%"),
                             alt.Tooltip("route_f1_hmm:Q", title="Route F1", format=".1%")])
                rule = alt.Chart(pd.DataFrame({"x": [pooled]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(x="x:Q")
                st.altair_chart((bars + rule).properties(height=32 * len(df) + 30), width="stretch")
                st.caption("Dashed line: pooled Top-1 across all trips.")
        with t:
            st.markdown(f"##### Per-trip results · run at {bt['at']}")
            show = pd.DataFrame({
                "Trip": df.trip_id, "Points": df.n_points,
                "Top-1 nearest": df.get("top1_clf"), "Top-1 HMM": df.get("top1_hmm"),
                "Route F1": df.get("route_f1_hmm"), "Continuity": df.continuity_hmm, "ms": df.pred_ms})
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="percent") for c in
                ["Top-1 nearest", "Top-1 HMM", "Route F1", "Continuity"]} | {
                "ms": st.column_config.NumberColumn("Pred. ms", format="%.0f")})
            st.download_button("Download CSV", df.to_csv(index=False).encode(),
                               file_name=f"pathsnap_{dataset}_results.csv", mime="text/csv",
                               icon=":material/download:")

# ---------------------------------------------------------------- model tab
with tab_model:
    meta = health["model"]
    hmm = meta.get("hmm") or {}
    st.markdown("#### How a trip is reconstructed")
    steps = ["Project GPS to metres", "Candidate roads (k ≤ 8, ≤ 50 m)", "Emission scores",
             "HMM transitions", "Viterbi decoding", "Fill gaps · output route"]
    st.markdown("<div class='ps-steps'>" + "<span class='ps-arrow'>→</span>".join(
        f"<span class='ps-step'><b>{i + 1}</b>{s}</span>" for i, s in enumerate(steps)) + "</div>",
        unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Deployed model")
        st.dataframe(pd.DataFrame({"Value": [
            meta.get("model_name"), meta.get("trained_on") or "—", (meta.get("created") or "—")[:19].replace("T", " "),
            f"{meta.get('radius', 0):.0f} m", str(meta.get("k")),
            "classifier probability + distance" if meta.get("dist_sigma") else "Gaussian distance (σ)",
        ]}, index=["Name", "Trained on", "Exported (UTC)", "Candidate radius", "Max candidates / point",
                   "Emission"]), width="stretch")
    with b:
        st.markdown("##### HMM parameters (tuned with Optuna TPE on train)")
        labels = {"sigma": ("σ  GPS noise", "m"), "beta": ("β  transition scale", "m"),
                  "gap_s": ("Chain break after gap", "s"), "back_tol": ("Backward tolerance", "m"),
                  "max_speed": ("Max speed", "m/s"), "sp_slack": ("Shortest-path slack", "m"),
                  "sp_cap": ("Shortest-path cap", "m")}
        st.dataframe(pd.DataFrame({"Value": [f"{hmm[k]:.2f} {u}" for k, (n, u) in labels.items() if k in hmm]},
                                  index=[n for k, (n, u) in labels.items() if k in hmm]), width="stretch")
    st.markdown("##### Integrity")
    st.markdown("""
- Prediction is made from the GPS track and road network only; the API never passes the reference route to the model.
- The reference route is read on Laptop 1 after prediction, only to compute the scores shown here.
- This console has no data, no model and no reference geometry; it renders what the API returns.
- The model and its settings were frozen before any test or evaluator trip was scored.
""")

# ---------------------------------------------------------------- activity tab
with tab_log:
    st.markdown("#### Recent predictions handled by the API")
    if st.button("Refresh", icon=":material/refresh:"):
        pass
    try:
        act = pd.DataFrame(api_get("/activity")["recent"])
    except Exception as e:  # noqa: BLE001
        act = pd.DataFrame()
        st.error(e)
    if act.empty:
        st.info("No predictions yet in this API session.")
    else:
        act["ts"] = pd.to_datetime(act.ts).dt.tz_convert(None).dt.strftime("%H:%M:%S") + " UTC"
        st.dataframe(act.rename(columns={"ts": "Time", "dataset": "Dataset", "trip": "Trip", "pred_ms": "Pred. ms",
                                         "n_points": "Points", "top1": "Top-1", "route_f1": "Route F1"}),
                     hide_index=True, width="stretch", column_config={
                         "Top-1": st.column_config.NumberColumn(format="percent"),
                         "Route F1": st.column_config.NumberColumn(format="percent")})
    st.caption("The full log (every request) is printed in the API terminal on Laptop 1 and saved to "
               "results/logs/api_requests.jsonl.")
