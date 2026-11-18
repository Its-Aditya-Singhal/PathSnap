# PathSnap demo guide (Phase 2)

How to read this: **Do** is what you click or type. **Say** is roughly what to tell the evaluator. Say it in your own words; don't read it out.

The demo takes about 10 minutes, and Q&A follows. Before you go in, practise it once from start to finish.

---

## 0. Setup (do this 15 minutes before)

### Laptop 1: API (the "server")
```bash
cd pathsnap
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Keep this terminal **visible and large**. The evaluator is required to see the API logs here.
- Find the IP: `ipconfig getifaddr en0` (Mac) or `ipconfig` (Windows). Last time it was 10.20.200.167, but it changes with the network.
- Firewall: allow incoming connections on port 8000 (on a Mac, click **Allow** when the popup appears).
- Check it works: open `http://<IP>:8000/health` in a browser. It should say `"status": "ok"` and list `evaluator_test: 10`.

### Laptop 2: UI (the "client")
```bash
cd pathsnap
source .venv/bin/activate
PATHSNAP_API_URL=http://<LAPTOP1-IP>:8000 streamlit run ui/app.py
```
- The browser opens `http://localhost:8501`. The sidebar should show **API connected** in green.
- If it shows red: both laptops must be on the same Wi-Fi (college Wi-Fi sometimes blocks device-to-device traffic, so use a phone hotspot), and check the firewall.
- **Fallback:** run both on one laptop. Leave `PATHSNAP_API_URL` unset and it uses localhost.

### Have these open in tabs (Laptop 2)
1. The UI: `localhost:8501`
2. `http://<IP>:8000/docs` (the Swagger API page)
3. `results/maps/pred_evaluator_test_00000067.html` (a map file)
4. The report PDF

---

## 1. Opening (30 seconds)

**Say:** "Our problem is map-matching. A car's GPS is noisy, so the points don't sit on the road. Given the GPS track and the road network, we recover exactly which road segments, or arcs, the car drove. Phase 1 was building and comparing models. Phase 2 is serving the final model as an API on this laptop, with a UI on the other laptop that calls it over the network."

## 2. The API terminal (1 minute), on Laptop 1

**Do:** point at the uvicorn terminal.

**Say:**
- "This is a FastAPI server. When it started it loaded our frozen model, the tuned HMM, and it serves the evaluator_test and student_test trips. It never serves train."
- "Every request that comes in is printed here: the client IP, the endpoint, the status code and the time taken. For predictions it also prints the trip, the number of points, Top-1 and route F1. The same log is saved to `results/logs/api_requests.jsonl`."

## 3. Swagger /docs (1 minute)

**Do:** open `<IP>:8000/docs`.

**Say:** "These are the 5 endpoints. `/health` is the status check. `/datasets` and `/trips` list what is available. `/predict/{trip_id}` runs the model. `/activity` returns the recent predictions."

**Do:** expand POST /predict, click Try it out, enter trip `00000067` and dataset `evaluator_test`, then Execute.

**Say:** "The response is JSON: the GPS points, the predicted arc IDs, the route geometry to draw, the metrics, and the timing. Look at Laptop 1: the request just appeared in the log."

**Point at the Laptop 1 terminal.** This is the moment that proves the client–server setup.

## 4. UI tab 1, Trip reconstruction (3 minutes), the main part

**Do:** in the sidebar, pick dataset **evaluator_test** and trip **00000067**, then click **Reconstruct route**.

**Say while it runs:** "The UI doesn't have the model. It sends an HTTP request to Laptop 1, and the server does the prediction."

**When the result is up:**
- Metric cards: "Top-1 is 99.5%, meaning 99.5% of GPS points were matched to the correct arc. Route F1 is 99.7%, the overlap between our predicted route and the true one. Continuity is 100%: the route has no jumps. Prediction took about 84 ms for 916 GPS points."
- Map: "Blue dots are the raw GPS points, grey lines are the candidate roads we considered, and the coloured line is our predicted route. Zoom in at a junction: the GPS points are off the road, but the route stays on real roads and takes legal turns."
- **Important line:** "The ground truth never leaves the server. The model predicts without the route file. Only afterwards does the server read the reference to score it, and it sends back only the numbers."

**Then show a harder trip:** pick **00000015** (84.5%). Be honest about it. "This is our weakest evaluator trip. The errors come where roads run parallel or on two-way roads, where both directions are equally close to the GPS."

## 5. UI tab 2, Batch evaluation (1 minute)

**Do:** go to the Batch evaluation tab and click **Run all**.

**Say:** "This sends all 10 evaluator trips to the API one by one. Overall: Top-1 92.0%, route F1 99.1%, continuity 99.96%, over 19,674 points. That's close to our student_test score of 92.9%, so the model generalises to trips it has never seen."

**Point at Laptop 1:** 10 PREDICT lines just appeared.

## 6. UI tab 3, Model & method (1 minute)

**Say:** "This explains the pipeline: candidates within 50 m, 18 features, the classifiers, and the HMM with Viterbi. It also shows the model comparison table from Phase 1, and why we picked the tuned HMM."

## 7. UI tab 4, Server activity (30 seconds)

**Say:** "This reads the `/activity` endpoint, so the UI can show what the server has handled."

## 8. Maps and results files (1 minute, if time allows)

- Open `results/maps/pred_evaluator_test_00000067.html`. "This is the Folium map. It has a layer control for GPS, predicted and reference, and an index page listing all trips."
- Mention: `results/evaluation/` has the per-trip CSVs from `scripts/run_evaluator.py`.

## 9. Closing (20 seconds)

**Say:** "To sum up: Phase 1 compared 7 classifiers plus an HMM baseline, then tuned them with Bayesian optimisation. The tuned HMM won on train and was frozen before we looked at the test sets. Phase 2 serves it as an API with logging, plus a 4-tab UI. On the new evaluator trips it gets 92% Top-1 and 99.1% route F1."

---

## What's new in Phase 2 (if they ask "what did you add?")

| Added | What it is |
|---|---|
| evaluator_test integration | API serves it automatically; `scripts/run_evaluator.py` scores the frozen model on it once (results/evaluation/) |
| Multi-dataset API | `/datasets`, `?dataset=` on every endpoint, optional `PATHSNAP_EVAL_DIR` for new trips handed out on the day ("custom") |
| Request logging | Every request on the console + `results/logs/api_requests.jsonl`; `/activity` endpoint |
| Ground-truth protection | Prediction never reads `.route`; scoring happens server-side; reference geometry never sent |
| Candidate roads in the response | The UI can show which roads the model considered |
| CORS + `PATHSNAP_API_URL` | UI can run on a different laptop |
| Professional 4-tab UI | Trip reconstruction, Batch evaluation, Model & method, Server activity |
| Maps per partition | `scripts/make_maps.py --partition evaluator_test` |
| Data-quality script | `scripts/data_quality.py`: missing values, duplicates, gaps, broken references |
| Report material | Screenshots in `results/screenshots/`, README updated |

---

## Likely questions and short answers

**Data and preprocessing**
- *What's in a trip?* There are 4 files. `.track` is lon, lat and time. `.nodes` holds the road node coordinates. `.arcs` holds directed road segments, each u→v. `.route` is the true arc sequence and only exists for labelled trips.
- *Missing values?* There are none. We checked with `data_quality.py`: no missing values, no duplicates, no bad references. The real issue is time gaps: 69 gaps over 60 s in train, and the longest is 153,064 s.
- *Why project coordinates?* Lat/lon degrees are not metres. We project each trip with an azimuthal-equidistant projection centred on the trip, so distances come out in metres.
- *Why isn't nearest-road enough?* The true arc is the nearest arc only 55% of the time. 92% of arcs are two-way, so each road has two arcs at the same distance.

**Candidates and features**
- *Candidates?* Up to 8 arcs within 50 m of each point. We find them with a KD-tree over points sampled every 10 m along each arc. The true arc is in this set 96% of the time on train and 98.6% on student_test.
- *Features?* There are 18:
  - distance and rank, compared with the nearest candidate
  - heading difference between GPS and arc
  - speed
  - where the point falls along the arc and the arc length
  - whether the road is two-way, and junction degree
  - ±3-point window averages
- *Labels?* Binary: 1 if the candidate is the true arc for that point.

**Models**
- *Which models?* Logistic regression, Gaussian NB, kNN, random forest, LightGBM, XGBoost and an MLP, plus the distance-only HMM baseline.
- *Best classifier?* LightGBM and XGBoost tied, with test Macro F1 0.925 and log loss 0.104. GNB was worst (log loss 0.62) because it assumes the features are independent.
- *How did you validate?* 5-fold GroupKFold, grouped by trip, so no trip is split across folds. Otherwise neighbouring points would leak.
- *Overfitting?* CV and test F1 are almost equal for every model. LightGBM's train loss keeps falling, 0.415 → 0.069, while validation flattens at about 0.14. That's mild late overfitting, but validation never rises.

**Bayesian optimisation**
- *What is it?* Optuna's TPE sampler. Unlike grid search, it learns from past trials which regions of the search space look good and samples more there. The pruner stops bad trials early.
- *What did you tune?*
  - LightGBM: 7 parameters, 30 trials, minimising CV log loss
  - the HMM's σ and β: 50 trials, maximising train Top-1
  - the hybrid: 50 trials
- *Result?*
  - The HMM's σ went from 5 to 21.8 m, which matches the 95th-percentile GPS error of 17.2 m.
  - The HMM went from 0.924 to 0.929 Top-1.
  - LightGBM + HMM went from 0.902 to 0.916.

**HMM / Viterbi (they will ask this)**
- *Hidden states?* The candidate arcs for each GPS point.
- *Emission?* How likely the GPS point is given the arc: a Gaussian of the perpendicular distance, −½(d/σ)².
- *Transition?* How likely it is to move from arc A to arc B. We compare the shortest driving distance along roads with the straight-line GPS distance, scoring −|d_route − d_gps|/β. If they are similar, the move is plausible. We disallow going backwards on an arc (more than 15 m) and speeds over 60 m/s.
- *Viterbi?* Dynamic programming that finds the single most likely sequence of states. It works in log space, so it's sums, not products.
- *Gaps / disconnected?* If the gap is over 60 s, or no candidate is reachable, we break the chain and start a new one. Then we join the pieces with a shortest path.
- *Did HMM help?* Massively. Nearest-road scores 0.525 and with the HMM it's 0.924. LightGBM goes from 0.872 to 0.902, and continuity from about 0.97 to 0.999.
- *Why did the plain HMM beat classifier + HMM?* The classifier is overconfident at exactly the ambiguous spots, such as the two directions of a road, and it overrides the sequence logic. Tuning shrank the classifier weight to 0.1. We chose by train Top-1 (0.8563 vs 0.8544) **before** looking at the test sets.

**API / UI / integrity**
- *What is an API?* A program that waits for requests over the network and replies with data, here as JSON. The UI is just a client and doesn't contain the model.
- *Why FastAPI?* It's fast and simple, has automatic `/docs`, and validates inputs.
- *Can you predict a new trip they give you now?* Yes. Put the folder anywhere, set `PATHSNAP_EVAL_DIR=<folder>` and restart the API. It shows up as dataset "custom". The trips need `.track`, `.nodes` and `.arcs`; without `.route` the metrics show as N/A, but the route is still predicted.
- *Did you train on the test data?* No. Train only. The selection rule was fixed in advance, and the test sets were scored once.
- *Are predictions hard-coded?* No. The API runs the model live. You can see the prediction time in the log, and any trip can be picked.
- *Speed?* About 0.2 ms per GPS point, so a 30-minute trip takes under 0.5 s.

**Limitations / future work**
- Parallel roads closer together than the GPS error, and direction at the start of a chain.
- Long gaps: the route between chains is a shortest-path guess.
- Future work: learn transition probabilities, use the road type or speed limit, and run online (streaming) Viterbi.

---

## If something breaks
- UI shows "API unreachable" → check the IP, use the same network, or fall back to one laptop.
- Port busy → `lsof -i :8000`, then kill that process, or use `--port 8001` and update the URL.
- Keep the screenshots in `results/screenshots/` as a last-resort backup to show.
