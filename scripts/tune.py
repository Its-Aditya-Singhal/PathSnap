"""Bayesian hyperparameter optimisation (Optuna TPE) — TRAIN TRIPS ONLY.

No student_test or evaluator data is read in the `clf` or `hmm` stages; the
search spaces, objectives and the final model choice are all decided on
train. student_test is touched only in the `final` stage, after everything
is frozen, to report the result.

Stages (run in order; each reads the previous stage's outputs):

  clf    LightGBM hyperparameters. Objective = mean candidate log loss over
         5-fold GroupKFold on the 80 train trips (groups = trip), with median
         pruning after each fold. Then out-of-fold (OOF) probabilities are
         produced with the best parameters, so every train trip gets emissions
         from a model that never saw it.
  hmm    Two studies on the 80 train trips, objective = pooled point Top-1
         after Viterbi decoding:
           hmm_baseline  Gaussian distance emission: sigma, beta
           hmm_hybrid    OOF classifier emission: clf_weight, optional distance
                         term (dist_sigma), beta
         Trial 0 of each study is the untuned default, for reference.
  final  Picks the system with the best TRAIN objective, refits the tuned
         LightGBM on all train trips, and reports student_test results for
         untuned vs tuned, classifier-only vs classifier + HMM.

Outputs: results/tuning/ (trials CSVs, best params, selection.json,
final_comparison.csv, Optuna sqlite db) and results/plots/tuning_*.png

    python scripts/tune.py --stage clf --trials 30
    python scripts/tune.py --stage hmm --trials 60
    python scripts/tune.py --stage final
"""
import argparse
import json
import multiprocessing as mp
import time
from dataclasses import replace

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import generate_candidates
from pathsnap.config import DATA_ROOT, PLOTS_DIR, RESULTS_DIR, SEED
from pathsnap.evaluate import candidate_metrics, evaluate_partition, labelled, load_point_labels
from pathsnap.features import FEATURES
from pathsnap.hmm import HMMConfig, ViterbiDecoder, hybrid_log_emission, reconstruct_route
from pathsnap.io import load_trip
from pathsnap.metrics import route_overlap
from pathsnap.models import MODELS
from pathsnap.pipeline import prepare

CACHE = RESULTS_DIR / "cache"
OUT = RESULTS_DIR / "tuning"
STORAGE = f"sqlite:///{OUT / 'optuna.db'}"
FOLDS = 5
BLUE, ORANGE, GRAY, GRID, MUTED = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#52514e"


def make_study(name, direction):
    if name in optuna.study.get_all_study_names(STORAGE):     # fresh study on every run
        optuna.delete_study(study_name=name, storage=STORAGE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    return optuna.create_study(study_name=name, storage=STORAGE, direction=direction,
                               sampler=optuna.samplers.TPESampler(seed=SEED), pruner=pruner)


def save_study(study, name, extra=None):
    df = study.trials_dataframe()
    df["duration_s"] = df["duration"].dt.total_seconds()
    df.drop(columns=["duration"]).to_csv(OUT / f"{name}_trials.csv", index=False)
    done = df[df.state == "COMPLETE"]
    rec = dict(study=name, direction=study.direction.name, best_value=study.best_value,
               best_params=study.best_params, best_trial=study.best_trial.number,
               best_user_attrs=study.best_trial.user_attrs,
               n_trials=len(df), n_complete=int(len(done)),
               n_pruned=int((df.state == "PRUNED").sum()),
               total_compute_s=float(df.duration_s.sum()),
               mean_trial_s=float(done.duration_s.mean()), **(extra or {}))
    try:
        rec["param_importance"] = optuna.importance.get_param_importances(study)
    except Exception as e:                       # needs >1 complete trial with variation
        rec["param_importance"] = f"n/a ({e})"
    (OUT / f"{name}_best.json").write_text(json.dumps(rec, indent=2, default=float))
    plot_history(df, name, study.direction.name == "MINIMIZE")
    print(json.dumps({k: rec[k] for k in ("best_value", "best_params", "n_trials", "n_pruned",
                                          "total_compute_s")}, default=float), flush=True)
    return rec


def plot_history(df, name, minimize):
    done = df[df.state == "COMPLETE"].sort_values("number")
    best = done.value.cummin() if minimize else done.value.cummax()
    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=150)
    ax.scatter(done.number, done.value, s=14, color=BLUE, alpha=0.7, label="trial", zorder=3)
    ax.step(done.number, best, where="post", color=ORANGE, lw=2, label="best so far", zorder=4)
    pr = df[df.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, [ax.get_ylim()[1 if minimize else 0]] * len(pr), marker="x", s=12,
                   color=GRAY, label="pruned", zorder=3)
    ax.set_xlabel("trial")
    ax.set_ylabel("CV log loss" if name.startswith("clf") else "train Top-1 (pooled)")
    ax.set_title(f"Optuna TPE: {name}", fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"tuning_{name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- stage: clf
def lgbm_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 900, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
    )


def stage_clf(n_trials, timeout):
    train = pd.read_pickle(CACHE / "train_candidates.pkl").reset_index(drop=True)
    tr = labelled(train).reset_index(drop=True)
    X, y, g = tr[FEATURES].to_numpy(np.float32), tr.y.to_numpy(), tr.trip.to_numpy()
    splits = list(GroupKFold(n_splits=FOLDS).split(X, y, g))
    factory = MODELS["lightgbm"]["factory"]
    print(f"[clf] {len(tr):,} labelled rows, {tr.trip.nunique()} trips, {FOLDS} folds", flush=True)

    def objective(trial):
        params = lgbm_space(trial)
        losses, f1s, top1s = [], [], []
        for f, (i_tr, i_va) in enumerate(splits):
            m = factory(**params).fit(X[i_tr], y[i_tr])
            cm = candidate_metrics(tr.iloc[i_va], m.predict_proba(X[i_va])[:, 1])
            losses.append(cm["log_loss"]); f1s.append(cm["macro_f1"]); top1s.append(cm["top1_cand"])
            trial.report(float(np.mean(losses)), f)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_macro_f1_std", float(np.std(f1s, ddof=1)))
        trial.set_user_attr("cv_top1_cand", float(np.mean(top1s)))
        print(f"  trial {trial.number}: logloss={np.mean(losses):.4f} macroF1={np.mean(f1s):.4f}",
              flush=True)
        return float(np.mean(losses))

    study = make_study("clf_lightgbm", "minimize")
    study.enqueue_trial({**MODELS["lightgbm"]["params"], "min_child_samples": 50, "subsample": 0.8,
                         "colsample_bytree": 0.8, "reg_lambda": 1.0})    # untuned defaults = trial 0
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    save_study(study, "clf_lightgbm",
               dict(objective="mean 5-fold GroupKFold candidate log loss (train trips only)"))

    # out-of-fold probabilities for ALL candidate rows (incl. unlabelled points) of each train trip
    best = study.best_params
    trips = np.array(sorted(train.trip.unique()))
    oof = np.full(len(train), np.nan)
    Xall = train[FEATURES].to_numpy(np.float32)
    for f, (i_tr, i_va) in enumerate(splits):
        va_trips = np.unique(g[i_va])
        m = factory(**best).fit(X[i_tr], y[i_tr])
        rows = np.flatnonzero(train.trip.isin(va_trips).to_numpy())
        oof[rows] = m.predict_proba(Xall[rows])[:, 1]
        print(f"  OOF fold {f}: {len(va_trips)} trips", flush=True)
    assert not np.isnan(oof).any() and len(trips) == 80
    np.save(CACHE / "train_oof_prob.npy", oof)

    t0 = time.perf_counter()
    final = factory(**best).fit(X, y)
    fit_s = time.perf_counter() - t0
    joblib.dump(final, CACHE / "models" / "lightgbm_tuned.joblib", compress=3)
    print(f"[clf] refit on all train in {fit_s:.1f}s -> lightgbm_tuned.joblib", flush=True)


# ---------------------------------------------------------------- stage: hmm
_G = {}


def _load_train_trips():
    train = pd.read_pickle(CACHE / "train_candidates.pkl").reset_index(drop=True)
    oof = np.load(CACHE / "train_oof_prob.npy")
    labels = load_point_labels("train")
    out = []
    for tid, rows in train.groupby("trip", sort=True):
        ctx = prepare(load_trip(DATA_ROOT / "train" / tid, "train"))
        cands = generate_candidates(ctx.px, ctx.py, ctx.net, radius=50.0, k=8)
        assert np.array_equal(cands.arc.to_numpy(), rows.arc.to_numpy()), tid
        out.append((tid, ctx, cands, oof[rows.index.to_numpy()], labels[tid]))
    out.sort(key=lambda r: -len(r[1].px))          # big trips first for load balance
    return out


def _eval_trip(args):
    i, cfg, use_clf, clf_weight, dist_sigma = args
    tid, ctx, cands, prob, lab = _G["trips"][i]
    log_e = hybrid_log_emission(cands, prob if use_clf else None, clf_weight, dist_sigma)
    res = ViterbiDecoder(ctx.net, HMMConfig(**cfg)).decode(ctx.px, ctx.py, ctx.ts, cands, log_e)
    m = lab >= 0
    route = reconstruct_route(ctx.net, res.arc, res.segment)
    f1 = route_overlap(ctx.net, route, ctx.trip.route)["route_f1"]
    return int((res.arc[m] == lab[m]).sum()), int(m.sum()), float(f1), int(res.n_breaks)


def train_hmm_score(pool, cfg, use_clf, clf_weight=1.0, dist_sigma=None):
    jobs = [(i, cfg.to_dict(), use_clf, clf_weight, dist_sigma) for i in range(len(_G["trips"]))]
    r = pool.map(_eval_trip, jobs, chunksize=1)
    hits, n = sum(a for a, *_ in r), sum(b for _, b, *_ in r)
    return hits / n, float(np.mean([x[2] for x in r])), int(sum(x[3] for x in r))


def stage_hmm(n_trials):
    t0 = time.perf_counter()
    _G["trips"] = _load_train_trips()
    print(f"[hmm] loaded {len(_G['trips'])} train trips in {time.perf_counter() - t0:.0f}s", flush=True)
    base_cfg = HMMConfig()
    with mp.get_context("fork").Pool(min(10, mp.cpu_count())) as pool:

        def obj_baseline(trial):
            cfg = replace(base_cfg, sigma=trial.suggest_float("sigma", 2.0, 30.0, log=True),
                          beta=trial.suggest_float("beta", 1.0, 50.0, log=True))
            top1, rf1, nb = train_hmm_score(pool, cfg, use_clf=False, dist_sigma=cfg.sigma)
            trial.set_user_attr("train_route_f1", rf1); trial.set_user_attr("n_breaks", nb)
            print(f"  baseline trial {trial.number}: top1={top1:.4f} routeF1={rf1:.4f} {trial.params}",
                  flush=True)
            return top1

        def obj_hybrid(trial):
            w = trial.suggest_float("clf_weight", 0.05, 5.0, log=True)
            use_dist = trial.suggest_categorical("use_dist", [True, False])
            ds = trial.suggest_float("dist_sigma", 2.0, 60.0, log=True) if use_dist else None
            cfg = replace(base_cfg, beta=trial.suggest_float("beta", 1.0, 50.0, log=True))
            top1, rf1, nb = train_hmm_score(pool, cfg, use_clf=True, clf_weight=w, dist_sigma=ds)
            trial.set_user_attr("train_route_f1", rf1); trial.set_user_attr("n_breaks", nb)
            print(f"  hybrid trial {trial.number}: top1={top1:.4f} routeF1={rf1:.4f} {trial.params}",
                  flush=True)
            return top1

        s1 = make_study("hmm_baseline", "maximize")
        s1.enqueue_trial({"sigma": base_cfg.sigma, "beta": base_cfg.beta})
        s1.optimize(obj_baseline, n_trials=n_trials)
        save_study(s1, "hmm_baseline", dict(objective="pooled train Top-1 after Viterbi (80 trips)"))

        s2 = make_study("hmm_hybrid", "maximize")
        s2.enqueue_trial({"clf_weight": 1.0, "use_dist": False, "beta": base_cfg.beta})
        s2.optimize(obj_hybrid, n_trials=n_trials)
        save_study(s2, "hmm_hybrid", dict(objective="pooled train Top-1 after Viterbi (80 trips), "
                                                    "OOF tuned-LightGBM emissions"))


# ---------------------------------------------------------------- stage: final
def stage_final():
    b = json.loads((OUT / "hmm_baseline_best.json").read_text())
    h = json.loads((OUT / "hmm_hybrid_best.json").read_text())
    c = json.loads((OUT / "clf_lightgbm_best.json").read_text())
    hp = h["best_params"]
    base_cfg = replace(HMMConfig(), **b["best_params"])
    hyb_cfg = replace(HMMConfig(), beta=hp["beta"])
    hyb_sigma = hp.get("dist_sigma") if hp.get("use_dist") else None

    # selection uses TRAIN objective only
    chosen = "lightgbm_tuned+hmm" if h["best_value"] >= b["best_value"] else "hmm_baseline_tuned"
    sel = dict(chosen=chosen, rule="highest pooled train Top-1 after Viterbi (OOF emissions)",
               train_top1_hmm_baseline_tuned=b["best_value"], train_top1_hybrid_tuned=h["best_value"],
               hmm_baseline=dict(hmm=base_cfg.to_dict()),
               hybrid=dict(model="lightgbm_tuned", lgbm_params=c["best_params"], hmm=hyb_cfg.to_dict(),
                           emission_weight=hp["clf_weight"], dist_sigma=hyb_sigma))
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
    print("[final] selection:", chosen, flush=True)

    # report on student_test (frozen choices; nothing below feeds back into tuning)
    test = pd.read_pickle(CACHE / "student_test_candidates.pkl")
    lgb_default = joblib.load(CACHE / "models" / "lightgbm.joblib")
    lgb_tuned = joblib.load(CACHE / "models" / "lightgbm_tuned.joblib")
    runs = [
        ("hmm_baseline (default)", None, HMMConfig(), 1.0, None),
        ("hmm_baseline (tuned)", None, base_cfg, 1.0, None),
        ("lightgbm default + HMM default", lgb_default, HMMConfig(), 1.0, None),
        ("lightgbm tuned + HMM default", lgb_tuned, HMMConfig(), 1.0, None),
        ("lightgbm tuned + HMM tuned (hybrid)", lgb_tuned, hyb_cfg, hp["clf_weight"], hyb_sigma),
    ]
    rows, per_trip_all = [], []
    for name, model, cfg, w, ds in runs:
        res, per_trip = evaluate_partition(model, test, cfg=cfg, clf_weight=w, dist_sigma=ds)
        res.pop("confusion_matrix", None)
        rows.append(dict(system=name, beta=cfg.beta, sigma=cfg.sigma if model is None else None,
                         clf_weight=w if model is not None else None, dist_sigma=ds, **res))
        per_trip.insert(0, "system", name)
        per_trip_all.append(per_trip)
        print(f"  {name}: top1_clf={res['top1_clf']:.4f} top1_hmm={res['top1_hmm']:.4f} "
              f"routeF1_hmm={res['route_f1_hmm']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "final_comparison.csv", index=False)
    pd.concat(per_trip_all).to_csv(OUT / "final_per_trip.csv", index=False)
    print("[final] wrote", OUT / "final_comparison.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clf", "hmm", "final"], required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds (clf stage)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    {"clf": lambda: stage_clf(a.trials, a.timeout), "hmm": lambda: stage_hmm(a.trials),
     "final": stage_final}[a.stage]()


if __name__ == "__main__":
    main()
