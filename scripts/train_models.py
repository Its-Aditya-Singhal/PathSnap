"""Train and compare candidate classifiers.

For each model:
  1. 5-fold GroupKFold CV on the 80 train trips (groups = trip) -> CV metrics.
  2. Refit on all train trips.
  3. Student-test evaluation: classifier-only and classifier + HMM/Viterbi.
Also evaluates the Gaussian-emission HMM baseline (no classifier).

Outputs (results/):
  model_comparison.csv   one row per model
  cv_folds.csv           per-fold CV metrics
  student_test_per_trip.csv
  histories/<model>.json real train/val loss curves (MLP, LightGBM, XGBoost; CV fold 0)
  confusion/<model>.json
Fitted models: results/cache/models/<model>.joblib

    python scripts/train_models.py [--models logreg gnb ...] [--folds 5]
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.config import RESULTS_DIR, SEED
from pathsnap.evaluate import candidate_metrics, evaluate_partition, labelled
from pathsnap.features import FEATURES
from pathsnap.models import MODELS, fit_model

CACHE = RESULTS_DIR / "cache"


def upsert_csv(path, rows, key):
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, dtype={key: str})
        new = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["hmm_baseline", *MODELS])
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    train = pd.read_pickle(CACHE / "train_candidates.pkl")
    test = pd.read_pickle(CACHE / "student_test_candidates.pkl")
    tr = labelled(train).reset_index(drop=True)
    X, y, groups = tr[FEATURES].to_numpy(np.float32), tr.y.to_numpy(), tr.trip.to_numpy()
    print(f"train rows {len(tr):,} ({tr.trip.nunique()} trips), test rows {len(test):,}", flush=True)

    for d in ("histories", "confusion"):
        (RESULTS_DIR / d).mkdir(exist_ok=True)
    (CACHE / "models").mkdir(parents=True, exist_ok=True)

    for name in args.models:
        print(f"\n=== {name}", flush=True)
        if name == "hmm_baseline":
            res, per_trip = evaluate_partition(None, test)
            row = dict(model=name, family="HMM only (Gaussian distance emission)",
                       config="sigma=5, beta=5", **res)
            upsert_csv(RESULTS_DIR / "model_comparison.csv", [row], "model")
            per_trip.insert(0, "model", name)
            upsert_csv(RESULTS_DIR / "student_test_per_trip.csv", per_trip.to_dict("records"), "model")
            print({k: round(v, 4) for k, v in res.items()}, flush=True)
            continue

        spec = MODELS[name]
        folds, hist0 = [], None
        for f, (i_tr, i_va) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups)):
            t0 = time.perf_counter()
            model, hist = fit_model(name, X[i_tr], y[i_tr], X[i_va], y[i_va])
            fit_s = time.perf_counter() - t0
            prob = model.predict_proba(X[i_va])[:, 1]
            m = candidate_metrics(tr.iloc[i_va], prob)
            folds.append(dict(model=name, fold=f, fit_s=fit_s, **m))
            if f == 0:
                hist0 = hist
            print(f"  fold {f}: macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f} "
                  f"top1={m['top1_cand']:.4f} fit={fit_s:.0f}s", flush=True)
        fd = pd.DataFrame(folds)

        t0 = time.perf_counter()
        model, _ = fit_model(name, X, y)
        fit_all_s = time.perf_counter() - t0
        joblib.dump(model, CACHE / "models" / f"{name}.joblib", compress=3)

        res, per_trip = evaluate_partition(model, test)
        cm = res.pop("confusion_matrix")
        row = dict(
            model=name, family=spec["family"], config=json.dumps(spec["params"]),
            cv_macro_f1=fd.macro_f1.mean(), cv_macro_f1_std=fd.macro_f1.std(),
            cv_log_loss=fd.log_loss.mean(), cv_top1=fd.top1_cand.mean(),
            train_fit_s=fit_all_s, **res, seed=SEED,
        )
        upsert_csv(RESULTS_DIR / "model_comparison.csv", [row], "model")
        upsert_csv(RESULTS_DIR / "cv_folds.csv", folds, "model")
        per_trip.insert(0, "model", name)
        upsert_csv(RESULTS_DIR / "student_test_per_trip.csv", per_trip.to_dict("records"), "model")
        (RESULTS_DIR / "confusion" / f"{name}.json").write_text(json.dumps(cm))
        if hist0:
            (RESULTS_DIR / "histories" / f"{name}.json").write_text(json.dumps(hist0))
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
