"""Shared evaluation: candidate-level metrics and full student-test evaluation
(classifier-only vs classifier + HMM/Viterbi)."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score

from .candidates import generate_candidates
from .config import DATA_ROOT, RESULTS_DIR
from .features import FEATURES
from .hmm import HMMConfig, ViterbiDecoder, gaussian_log_emission, hybrid_log_emission, reconstruct_route
from .io import load_trip
from .metrics import route_continuity, route_overlap
from .pipeline import prepare


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.label_arc >= 0]


def candidate_metrics(df: pd.DataFrame, prob: np.ndarray) -> dict:
    """Metrics on labelled candidate rows.
    Macro F1/precision/recall: binary 'is true arc' task at threshold 0.5, macro over both classes.
    top1_cand: per GPS point, is the highest-probability candidate the reference arc?"""
    y = df.y.to_numpy()
    pred = (prob >= 0.5).astype(int)
    tmp = pd.DataFrame({"g": df.trip.astype(str) + ":" + df.point.astype(str), "p": prob, "y": y})
    best = tmp.loc[tmp.groupby("g", sort=False)["p"].idxmax(), "y"]
    return dict(
        macro_f1=f1_score(y, pred, average="macro"),
        macro_precision=precision_score(y, pred, average="macro", zero_division=0),
        macro_recall=recall_score(y, pred, average="macro", zero_division=0),
        log_loss=log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        top1_cand=float(best.mean()),
    )


def load_point_labels(partition: str) -> dict:
    lab = pd.read_csv(RESULTS_DIR / "labels" / f"{partition}_point_labels.csv.gz", dtype={"trip": str})
    return {t: g.sort_values("point").label_arc.to_numpy() for t, g in lab.groupby("trip")}


def evaluate_partition(model, df: pd.DataFrame, partition: str = "student_test",
                       cfg: HMMConfig = HMMConfig(), radius: float = 50.0, k: int = 8,
                       clf_weight: float = 1.0, dist_sigma=None) -> dict:
    """Run every trip end-to-end. model=None -> Gaussian-emission HMM baseline.
    With a model the emission is hybrid_log_emission(prob, clf_weight, dist_sigma).

    Point-level Top-1 is pooled over all labelled GPS points (points without
    a candidate count as wrong). Route metrics are averaged over trips."""
    labels = load_point_labels(partition)
    hit_clf = hit_hmm = n_lab = 0
    n_pts, t_model, t_hmm = 0, 0.0, 0.0
    trip_rows = []
    for tid, rows in df.groupby("trip", sort=True):
        trip = load_trip(DATA_ROOT / partition / tid, partition)
        ctx = prepare(trip)
        cands = generate_candidates(ctx.px, ctx.py, ctx.net, radius=radius, k=k)
        assert np.array_equal(cands.arc.to_numpy(), rows.arc.to_numpy()), "candidate order mismatch"
        T = len(ctx.px)

        t0 = time.perf_counter()
        if model is None:
            prob = np.exp(gaussian_log_emission(cands, cfg.sigma))
            log_e = gaussian_log_emission(cands, cfg.sigma)
        else:
            prob = model.predict_proba(rows[FEATURES].to_numpy(np.float32))[:, 1]
            log_e = hybrid_log_emission(cands, prob, clf_weight, dist_sigma)
        t_model += time.perf_counter() - t0

        # classifier-only: argmax per point, no sequence model
        clf_arc = np.full(T, -1)
        c = cands.assign(p=prob)
        best = c.loc[c.groupby("point")["p"].idxmax()]
        clf_arc[best.point.to_numpy()] = best.arc.to_numpy()

        t0 = time.perf_counter()
        res = ViterbiDecoder(ctx.net, cfg).decode(ctx.px, ctx.py, ctx.ts, cands, log_e)
        t_hmm += time.perf_counter() - t0

        lab = labels[tid]
        m = lab >= 0
        hit_clf += int((clf_arc[m] == lab[m]).sum())
        hit_hmm += int((res.arc[m] == lab[m]).sum())
        n_lab += int(m.sum())
        n_pts += T

        clf_route = [int(a) for i, a in enumerate(clf_arc) if a >= 0 and (i == 0 or a != clf_arc[i - 1])]
        hmm_route = reconstruct_route(ctx.net, res.arc, res.segment)
        ro_c = route_overlap(ctx.net, clf_route, trip.route)
        ro_h = route_overlap(ctx.net, hmm_route, trip.route)
        trip_rows.append(dict(
            trip=tid,
            top1_clf=float(np.mean(clf_arc[m] == lab[m])), top1_hmm=float(np.mean(res.arc[m] == lab[m])),
            continuity_clf=route_continuity(ctx.net, clf_arc, ctx.px, ctx.py),
            continuity_hmm=route_continuity(ctx.net, res.arc, ctx.px, ctx.py),
            route_f1_clf=ro_c["route_f1"], route_f1_hmm=ro_h["route_f1"], n_breaks=res.n_breaks))

    per_trip = pd.DataFrame(trip_rows)
    out = dict(
        top1_clf=hit_clf / n_lab, top1_hmm=hit_hmm / n_lab,
        continuity_clf=per_trip.continuity_clf.mean(), continuity_hmm=per_trip.continuity_hmm.mean(),
        route_f1_clf=per_trip.route_f1_clf.mean(), route_f1_hmm=per_trip.route_f1_hmm.mean(),
        infer_ms_per_point_model=1000 * t_model / n_pts, infer_ms_per_point_hmm=1000 * t_hmm / n_pts,
    )
    if model is not None:
        lab_df = labelled(df)
        p = model.predict_proba(lab_df[FEATURES].to_numpy(np.float32))[:, 1]
        cm = candidate_metrics(lab_df, p)
        out.update({f"test_{k}": v for k, v in cm.items()})
        out["confusion_matrix"] = confusion_matrix(lab_df.y, (p >= 0.5).astype(int)).tolist()
    return out, per_trip
