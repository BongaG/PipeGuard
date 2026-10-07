import numpy as np
from .config import Config


def feature_names(nodes):
    names = []
    for n in nodes:
        names += [f"{n}_res_shift", f"{n}_res_min", f"{n}_res_slope", f"{n}_raw_shift", f"{n}_std", f"{n}_res_frac"]
    names += ["imbalance_shift", "imbalance_now", "inlet_shift", "metered_shift", "max_rate"]
    return names


def window_features(pressure, twin, inlet, metered, t, cfg=Config, base_end=None):
    w = cfg.WINDOW_S
    b = cfg.BASELINE_S
    lo = max(t - w + 1, 0)
    bhi = max(t - w + 1, 1) if base_end is None else max(min(base_end, t - w + 1), 1)
    blo = max(bhi - b, 0)
    res = pressure - twin
    win_res = res[lo:t + 1]
    base_res = np.median(res[blo:bhi], axis=0)
    win_p = pressure[lo:t + 1]
    base_p = np.median(pressure[blo:bhi], axis=0)
    x = np.arange(len(win_res)) - (len(win_res) - 1) / 2
    denom = max(float((x ** 2).sum()), 1.0)
    slope = (x[:, None] * (win_res - win_res.mean(axis=0))).sum(axis=0) / denom
    res_frac = win_res.mean(axis=0) / np.maximum(twin[lo:t + 1].mean(axis=0), 1.0)
    per_node = np.stack([
        win_res.mean(axis=0) - base_res,
        win_res.min(axis=0) - base_res,
        slope,
        win_p.mean(axis=0) - base_p,
        win_p.std(axis=0),
        res_frac,
    ], axis=1).ravel()
    imb = inlet - metered
    imb_base = np.median(imb[blo:bhi])
    rate = np.abs(np.diff(pressure[max(lo - 1, 0):t + 1], axis=0)).max() if t > 0 else 0.0
    glob = np.array([
        imb[lo:t + 1].mean() - imb_base,
        imb[lo:t + 1].mean(),
        inlet[lo:t + 1].mean() - np.median(inlet[blo:bhi]),
        metered[lo:t + 1].mean() - np.median(metered[blo:bhi]),
        rate,
    ])
    return np.concatenate([per_node, glob])


def episode_windows(ep, stride=1, cfg=Config, start=None, base_end=None):
    p, tw, inl, met = ep["pressure"], ep["twin_pressure"], ep["inlet_flow"], ep["metered_flow"]
    T = len(inl)
    start = cfg.WINDOW_S + cfg.BASELINE_S if start is None else start
    ts = list(range(start, T, stride))
    X = np.array([window_features(p, tw, inl, met, t, cfg, base_end) for t in ts])
    return ts, X
