"""HMM map-matching with Viterbi decoding.

Hidden state at GPS point t  = one of its candidate arcs (with projection offset).
Emission  log P(obs_t | arc)  = pluggable: Gaussian on distance (baseline) or a
                                trained classifier's probability.
Transition log P(arc_t | arc_{t-1}) = -|d_route - d_gps| / beta   (Newson & Krumm 2009)
   d_route = network distance between the two projected positions,
   d_gps   = straight-line distance between the two GPS points.
Breaks: the chain is cut (and decoding restarts) when the time gap exceeds
`gap_s` or when no candidate at t is reachable from any candidate at t-1.
Points with no candidates are skipped (arc = -1) and bridged.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass
class HMMConfig:
    beta: float = 5.0          # transition scale (m)
    sigma: float = 5.0         # baseline Gaussian emission std (m)
    gap_s: float = 60.0        # time gap that forces a chain break
    back_tol: float = 15.0     # allowed backward jitter along the same arc (m)
    max_speed: float = 60.0    # m/s, bounds plausible route distance per second
    sp_slack: float = 200.0    # extra route distance allowed beyond max_speed*dt (m)
    sp_cap: float = 3000.0     # hard cap on shortest-path search radius (m)

    def to_dict(self):
        return asdict(self)


def gaussian_log_emission(cands: pd.DataFrame, sigma: float) -> np.ndarray:
    return -0.5 * (cands["dist"].to_numpy() / sigma) ** 2


def prob_log_emission(prob: np.ndarray, points: np.ndarray, eps: float = 1e-6,
                      weight: float = 1.0) -> np.ndarray:
    """Classifier probabilities -> per-point normalised log emissions.

    `weight` scales emission against transition (1.0 = plain log-prob).
    Values < 1 soften over-confident classifiers so the route prior can
    still overrule them; it is tuned on train only."""
    p = np.clip(prob, eps, None)
    s = pd.Series(p).groupby(points).transform("sum").to_numpy()
    return weight * np.log(p / s)


def hybrid_log_emission(cands: pd.DataFrame, prob=None, clf_weight: float = 1.0,
                        dist_sigma=None) -> np.ndarray:
    """Emission used by the final system:
        clf_weight * log P_clf(arc | point)  +  (-0.5 * (dist / dist_sigma)^2)
    Either term can be switched off (prob=None / clf_weight=0, or dist_sigma=None).
    With prob=None it reduces to the Gaussian baseline."""
    out = np.zeros(len(cands))
    if prob is not None and clf_weight:
        out += prob_log_emission(prob, cands["point"].to_numpy(), weight=clf_weight)
    if dist_sigma:
        out += gaussian_log_emission(cands, dist_sigma)
    return out


@dataclass
class DecodeResult:
    arc: np.ndarray        # (T,) matched arc per GPS point, -1 if none
    cand: np.ndarray       # (T,) row index into candidates, -1 if none
    segment: np.ndarray    # (T,) chain id, -1 if none
    n_breaks: int


class ViterbiDecoder:
    def __init__(self, net: RoadNetwork, cfg: HMMConfig = HMMConfig()):
        self.net, self.cfg = net, cfg
        self._sp_cache: dict[int, np.ndarray] = {}

    # ---- shortest paths with a one-step cache (consecutive points share sources)
    def _sp_rows(self, nodes):
        new = [n for n in np.unique(nodes) if n not in self._sp_cache]
        if new:
            rows = self.net.sp_from(new, limit=self.cfg.sp_cap)
            for n, r in zip(new, np.atleast_2d(rows)):
                self._sp_cache[n] = r
        return self._sp_cache

    def log_transition(self, A, tA, B, tB, d_gps, dt):
        net, cfg = self.net, self.cfg
        lenA, lenB = net.length[A], net.length[B]
        cache = self._sp_rows(net.v[A])
        d_nodes = np.stack([cache[n][net.u[B]] for n in net.v[A]])
        d_route = (1 - tA)[:, None] * lenA[:, None] + d_nodes + (tB * lenB)[None, :]

        same = A[:, None] == B[None, :]
        fwd = (tB * lenB)[None, :] - (tA * lenA)[:, None]
        on_same = same & (fwd >= -cfg.back_tol)
        d_route = np.where(on_same, np.maximum(fwd, 0.0), d_route)

        limit = min(cfg.max_speed * max(dt, 1.0) + cfg.sp_slack, cfg.sp_cap)
        logp = -np.abs(d_route - d_gps) / cfg.beta
        return np.where(np.isfinite(d_route) & (d_route <= limit), logp, -np.inf)

    def decode(self, px, py, ts, cands: pd.DataFrame, log_emis: np.ndarray) -> DecodeResult:
        T = len(px)
        pts = cands["point"].to_numpy()
        arcs = cands["arc"].to_numpy()
        toff = cands["t"].to_numpy()
        start = np.searchsorted(pts, np.arange(T), "left")
        end = np.searchsorted(pts, np.arange(T), "right")

        back = {}                      # t -> argmax predecessor index per candidate
        prev_of = np.full(T, -1)       # previous decoded point in the same chain
        seg_ends = []                  # (t_end, delta_end)
        delta, prev_t = None, None

        for t in range(T):
            s, e = start[t], end[t]
            if s == e:
                continue
            emis = log_emis[s:e]
            new_chain = prev_t is None or (ts[t] - ts[prev_t]) > self.cfg.gap_s
            if not new_chain:
                ps, pe = start[prev_t], end[prev_t]
                d_gps = float(np.hypot(px[t] - px[prev_t], py[t] - py[prev_t]))
                trans = self.log_transition(arcs[ps:pe], toff[ps:pe], arcs[s:e], toff[s:e],
                                            d_gps, ts[t] - ts[prev_t])
                scores = delta[:, None] + trans
                best = np.argmax(scores, axis=0)
                cand_delta = scores[best, np.arange(e - s)] + emis
                if np.all(~np.isfinite(cand_delta)):
                    new_chain = True
                else:
                    back[t], prev_of[t], delta = best, prev_t, cand_delta
            if new_chain:
                if prev_t is not None:
                    seg_ends.append((prev_t, delta))
                delta = emis.copy()
            prev_t = t
            # keep only sources still needed next step
            if len(self._sp_cache) > 64:
                keep = set(self.net.v[arcs[s:e]])
                self._sp_cache = {k: v for k, v in self._sp_cache.items() if k in keep}
        if prev_t is not None:
            seg_ends.append((prev_t, delta))

        cand_idx = np.full(T, -1)
        segment = np.full(T, -1)
        for sid, (t, d) in enumerate(seg_ends):
            j = int(np.argmax(d))
            while True:
                cand_idx[t] = start[t] + j
                segment[t] = sid
                if t not in back:
                    break
                j = int(back[t][j])
                t = prev_of[t]
        arc = np.where(cand_idx >= 0, arcs[np.maximum(cand_idx, 0)], -1)
        return DecodeResult(arc, cand_idx, segment, max(len(seg_ends) - 1, 0))


def reconstruct_route(net: RoadNetwork, arc: np.ndarray, segment: np.ndarray,
                      limit: float = 3000.0) -> list[int]:
    """Point-wise arcs -> connected arc sequence (fills gaps with shortest paths)."""
    route, prev, prev_seg = [], None, None
    for a, sgm in zip(arc, segment):
        if a < 0:
            continue
        if prev is not None and a != prev:
            if sgm == prev_seg:
                mid = net.path_arcs(net.v[prev], net.u[a], limit=limit)
                if mid:
                    route.extend(mid)
            route.append(int(a))
        elif prev is None:
            route.append(int(a))
        prev, prev_seg = a, sgm
    return route
