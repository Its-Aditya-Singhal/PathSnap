# PathSnap — Map-Matching & Route Reconstruction

Noisy GPS trajectories → candidate road arcs → classifier probabilities → HMM/Viterbi decoding →
reconstructed route. Served by a FastAPI backend (Laptop 1) to a Streamlit evaluator UI (Laptop 2).

## Pipeline

1. **Projection.** Each trip comes with its own road network (`.nodes`, `.arcs`), and trips are spread worldwide. Every trip is projected to a local azimuthal-equidistant frame centred on its own GPS track, so all distances are in metres.
2. **Candidates.** For each GPS point, take up to `k=8` arcs within `50 m`, found with a KD-tree over points sampled every 10 m along each arc.
3. **Labels.** For each GPS point the true arc is derived by an HMM restricted to the arcs of the reference `.route`. The script is `scripts/build_dataset.py`.
4. **Features.** 18 per point–candidate features: distance, distance rank and margin, heading agreement, windowed distance and heading, GPS speed, arc length, connectivity, and others. See `src/pathsnap/features.py`.
5. **Classifiers.** Seven model families predict P(candidate is the true arc): logistic regression, Gaussian NB, kNN, random forest, LightGBM, XGBoost and an MLP. They are compared with 5-fold **GroupKFold by trip** on the 80 train trips.
6. **HMM/Viterbi.** Transitions follow Newson & Krumm (2009): `-|d_route − d_gps| / β`, where `d_route` is the shortest-path network distance. Emissions are `w · log P_clf + (−½ (d/σ)²)`. The chain breaks when the time gap exceeds 60 s or no candidate is reachable. The output is a connected arc sequence, with gaps filled by shortest paths.
7. **Bayesian HPO.** Optuna TPE runs on **train trips only**:
   - LightGBM hyperparameters, minimising 5-fold GroupKFold log loss
   - HMM and emission parameters, maximising train Top-1 with out-of-fold classifier probabilities

   The final system is chosen by its train objective. student_test is used only for reporting.

## Setup

```bash
brew install libomp                      # macOS only: needed by LightGBM / XGBoost
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export PATHSNAP_DATA="/path/to/Student folder"   # optional, default ../Student folder
```

## Reproduce Phase 1 (run from the repo root)

```bash
python scripts/run_hmm_baseline.py       # Gaussian-emission HMM baseline
python scripts/build_dataset.py          # labels + candidates + features -> results/cache, results/labels
python scripts/eda.py                    # EDA stats, plots, Folium maps
python scripts/train_models.py           # 7 classifiers: GroupKFold CV + student_test eval
python scripts/plot_results.py           # comparison plots
python scripts/tune.py --stage clf --trials 30 --timeout 3000
python scripts/tune.py --stage hmm --trials 50
python scripts/tune.py --stage final     # frozen selection + student_test report
python scripts/export_artifacts.py       # artifacts/pathsnap_model.joblib
python scripts/make_maps.py              # prediction maps for student_test
```

## Evaluate held-out partitions (frozen model, no refitting)

```bash
python scripts/run_evaluator.py --partition evaluator_test   # -> results/evaluation/evaluator_test_*.csv/json
python scripts/run_evaluator.py --partition student_test
python scripts/make_maps.py --partition evaluator_test       # -> results/maps/pred_evaluator_test_*.html
```

| Output | Where |
|---|---|
| Model comparison (CV + student_test) | `results/model_comparison.csv`, `results/cv_folds.csv` |
| Optuna studies (all trials, best params, compute cost) | `results/tuning/*_trials.csv`, `*_best.json`, `optuna.db` |
| Tuned vs untuned, classifier-only vs + HMM | `results/tuning/final_comparison.csv` |
| Plots | `results/plots/` |
| Folium maps | `results/maps/` (open `index.html`) |
| Held-out evaluation (per trip + summary) | `results/evaluation/` |
| Data-quality checks (missing values, duplicates, gaps) | `results/eda/data_quality_*.csv` (`python scripts/data_quality.py`) |
| UI, API and map screenshots | `results/screenshots/` |
| Technical report | submitted as a separate PDF (added to this repo by the team at submission) |
| Demo script and Q&A notes | `docs/DEMO_GUIDE.md`, `docs/PHASE1_QA.md`, `docs/PathSnap_Explained_QA.pdf` |
| Final model bundle | `artifacts/pathsnap_model.joblib`, `artifacts/pathsnap_config.json` |

## Results

Final system: **`hmm_baseline_tuned`** (Gaussian-emission HMM with Optuna-tuned σ = 21.8 m, β = 30.6). It was chosen by pooled train Top-1 with out-of-fold emissions: 0.856, against 0.854 for the tuned LightGBM + HMM hybrid. The choice was frozen in `results/tuning/selection.json` before student_test was evaluated.

student_test (10 trips, 19,160 points), from `results/tuning/final_comparison.csv`:

| System | Top-1 (per point) | Top-1 + HMM | Route F1 + HMM | Continuity + HMM |
|---|---|---|---|---|
| HMM baseline (default σ = β = 5) | 0.525 (nearest arc) | 0.924 | 0.986 | 0.999 |
| **HMM baseline (tuned)** | 0.525 (nearest arc) | **0.929** | **0.988** | 0.999 |
| LightGBM default + HMM default | 0.872 | 0.902 | 0.989 | 0.999 |
| LightGBM tuned + HMM default | 0.874 | 0.916 | 0.989 | 0.999 |
| LightGBM tuned + HMM tuned (hybrid) | 0.874 | 0.927 | 0.988 | 0.999 |

Optuna compute:

| Study | Trials | Pruned | Compute | Best |
|---|---|---|---|---|
| LightGBM | 30 | 10 | 2,553 s | CV log loss 0.1133, against 0.1147 untuned |
| HMM baseline | 50 | 0 | 154 s | train Top-1 0.856 |
| Hybrid | 50 | 0 | 165 s | train Top-1 0.854, best classifier weight ≈ 0.10 |

All seven classifiers (CV + student_test) are in `results/model_comparison.csv`. The best per-point classifiers are LightGBM, XGBoost and RF, with test Macro F1 of about 0.925.

## Evaluator test results

These are the frozen `hmm_baseline_tuned` bundle run with no changes on the 10 evaluator trips (19,674 points) handed out for Phase 2. The figures come from `results/evaluation/evaluator_test_summary.json`, with the per-trip CSV beside it. The evaluator data was not used for training, tuning or model selection.

| Partition | Trips | Points | Top-1 nearest arc | **Top-1 + HMM** (pooled) | Route F1 (mean) | Route P / R | Continuity | Latency |
|---|---|---|---|---|---|---|---|---|
| evaluator_test | 10 | 19,674 | 0.472 | **0.920** | **0.991** | 0.986 / 0.996 | 0.9996 | 0.22 ms/point |
| student_test | 10 | 19,160 | 0.525 | **0.929** | **0.988** | 0.980 / 0.997 | 0.9992 | 0.18 ms/point |

## Phase 2: two-laptop deployment

**Laptop 1: API.** It holds the data, the model and the ground truth.

```bash
cd pathsnap && source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0                                 # this laptop's LAN IP
```

| Env var | Meaning |
|---|---|
| `PATHSNAP_DATA` | folder holding the partitions (default `../Student folder`) |
| `PATHSNAP_DATASETS` | partitions to serve (default `evaluator_test,student_test`; `train` is never served) |
| `PATHSNAP_EVAL_DIR` | optional extra folder of trips, served as dataset `custom` (for new cases handed out on the day) |
| `PATHSNAP_BUNDLE` | model bundle (default `artifacts/pathsnap_model.joblib`) |

| Endpoint | Returns |
|---|---|
| `GET /health` | status, model metadata, datasets served |
| `GET /datasets` | dataset names and trip counts |
| `GET /trips?dataset=` | trip id, point count, duration, road-arc count, map centre |
| `POST /predict/{trip_id}?dataset=` | GPS, predicted route arcs and polylines, candidate roads, server-side metrics, timings |
| `GET /activity` | the last 50 predictions handled |

Interactive docs are at `http://<laptop1-ip>:8000/docs`. Each request is logged to the console as a `PREDICT dataset=… trip=… top1=… route_f1=…` line and also appended to `results/logs/api_requests.jsonl`.

**Laptop 2: UI.** It holds no data files and no model.

```bash
PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py   # http://localhost:8501
```

The UI has four tabs:
- **Trip reconstruction.** Pick a trip and press **Reconstruct route**. It shows Top-1 with its gain over nearest-road, route F1 with P/R, continuity, latency, and a map of the GPS points, candidate roads and the reconstructed route. Light, Streets and Satellite basemaps are available.
- **Batch evaluation.** Runs every trip through the API and shows the pooled scores, a per-trip chart and a downloadable CSV.
- **Model & method.** The pipeline, the deployed model and the tuned HMM parameters.
- **Server activity.** Recent predictions, read from the API.

Both laptops must be on the same network, and the API laptop's firewall must allow port 8000. To run everything on one laptop, leave `PATHSNAP_API_URL` unset; the UI then uses `http://localhost:8000`.

## Integrity

- The prediction path (`pathsnap.inference`) loads trips with `with_route=False` and asserts that no route is passed in. Reference routes are read only in `pathsnap.serve_eval`, after the prediction is made, and only aggregate metrics leave the API.
- No student_test or evaluator data is used for training, tuning, choosing the search space, or choosing the model.
- Every number and plot is produced by the scripts above from the raw data.

## Known limitations

- Labels come from a distance-based HMM constrained to the reference route. They therefore favour distance-based matching, which partly explains why the Gaussian-emission HMM is a strong baseline.
- Each road is usually two directed arcs at the same distance. On train, the true arc is the nearest candidate for only about 55% of points, so most remaining errors are direction confusions on two-way roads.
