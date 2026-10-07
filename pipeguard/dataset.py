import numpy as np
from .config import Config
from .simulator import TelemetrySimulator, EVENT_CLASSES
from .features import episode_windows, window_features


def build_dataset(n_episodes, seed, duration=300, stride=1, keep_episodes=False):
    sim = TelemetrySimulator(seed=seed)
    rng = np.random.default_rng(seed + 1)
    X, X_loc, y_event, y_pipe, ep_id, t_since, sizes = [], [], [], [], [], [], []
    episodes = []
    for e in range(n_episodes):
        ep = sim.episode(duration)
        sc = ep["scenario"]
        ts, Xe = episode_windows(ep, stride=stride)
        X.append(Xe)
        Xl = Xe.copy()
        if sc.kind in ("leak", "burst"):
            for k, t in enumerate(ts):
                if t >= sc.onset:
                    be = int(sc.onset) + int(rng.integers(*Config.LATCH_OFFSET_S))
                    Xl[k] = window_features(ep["pressure"], ep["twin_pressure"], ep["inlet_flow"], ep["metered_flow"], t, Config, be)
        X_loc.append(Xl)
        y_event.append(ep["labels"][ts])
        y_pipe.append(np.array([sc.pipe or "none"] * len(ts)))
        ep_id.append(np.full(len(ts), e))
        onset = sc.onset if sc.onset is not None else np.inf
        t_since.append(np.array(ts) - onset)
        sizes.append(np.full(len(ts), sc.share))
        if keep_episodes:
            episodes.append({"scenario": sc.to_dict(), "ts": ts, "X": Xe, "X_loc": Xl, "labels": ep["labels"], "pressure": ep["pressure"], "true_pressure": ep["true_pressure"], "twin_pressure": ep["twin_pressure"], "inlet_flow": ep["inlet_flow"], "metered_flow": ep["metered_flow"], "leak_flow": ep["leak_flow"]})
    data = {
        "X": np.vstack(X),
        "X_loc": np.vstack(X_loc),
        "y_event": np.concatenate(y_event),
        "y_pipe": np.concatenate(y_pipe),
        "episode": np.concatenate(ep_id),
        "t_since": np.concatenate(t_since),
        "share": np.concatenate(sizes),
    }
    return data, episodes


def class_counts(y):
    return {EVENT_CLASSES[i]: int((y == i).sum()) for i in range(len(EVENT_CLASSES))}
