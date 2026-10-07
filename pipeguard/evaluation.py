import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .config import Config
from .dataset import build_dataset
from .detector import LeakDetector, StaticThresholdDetector
from .network import PipelineNetwork
from .simulator import EVENT_CLASSES, TelemetrySimulator
from .features import window_features

LEAKISH = [EVENT_CLASSES.index("leak"), EVENT_CLASSES.index("burst")]


def _binary(y):
    return np.isin(y, LEAKISH).astype(int)


def _bin_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def nominal_pressure(net):
    d = {k: j.base_demand / 1000.0 for k, j in net.junctions.items()}
    r = net.solve(d)
    return np.array([r["pressure_kpa"][n] for n in net.sensor_nodes])


def window_level(det, data, net):
    out = {}
    y = data["y_event"]
    settled = data["t_since"] < 0
    settled |= data["t_since"] >= 3
    for name in ("xgb", "mlp"):
        pred = det.predict_event(data["X"], name)
        p, r, f, _ = precision_recall_fscore_support(y[settled], pred[settled], labels=range(4), zero_division=0)
        out[name] = {
            "multiclass_accuracy": float(accuracy_score(y[settled], pred[settled])),
            "per_class": {EVENT_CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i in range(4)},
            "macro_f1": float(f.mean()),
            "binary": _bin_metrics(_binary(y[settled]), _binary(pred[settled])),
            "confusion": confusion_matrix(y[settled], pred[settled], labels=range(4)).tolist(),
        }
    return out


def threshold_level(episodes, net, drop_kpa=15.0):
    thr = StaticThresholdDetector(nominal_pressure(net), drop_kpa)
    yt, yp = [], []
    for ep in episodes:
        for t in ep["ts"]:
            since = t - (ep["scenario"]["onset"] if ep["scenario"]["onset"] is not None else np.inf)
            if 0 <= since < 3:
                continue
            yt.append(int(ep["labels"][t] in LEAKISH))
            yp.append(int(thr.predict_alarm(ep["pressure"][t])))
    return _bin_metrics(np.array(yt), np.array(yp))


def localisation(det, data):
    mask = np.isin(data["y_event"], LEAKISH) & (data["t_since"] >= 10)
    X = data["X_loc"][mask]
    truth = data["y_pipe"][mask]
    out = {}
    for name in ("mlp", "xgb"):
        proba = det.predict_pipe_proba(X, name)
        order = np.argsort(-proba, axis=1)
        idx = np.array([det.pipes.index(p) for p in truth])
        top1 = (order[:, 0] == idx).mean()
        top2 = ((order[:, 0] == idx) | (order[:, 1] == idx)).mean()
        per_pipe = {}
        for k, pipe in enumerate(det.pipes):
            m = idx == k
            per_pipe[pipe] = float((order[m, 0] == k).mean()) if m.any() else None
        out[name] = {"top1": float(top1), "top2": float(top2), "per_pipe": per_pipe}
    return out


def ulp_trigger_time(p, onset, cfg=Config):
    T = len(p)
    start = int(np.ceil(onset))
    for t in range(max(start, 31), T):
        ref = p[t - 31:t - 1].mean(axis=0)
        if (np.abs(p[t] - ref) > cfg.ULP_DEVIATION_KPA).any() or (np.abs(p[t] - p[t - 1]) > cfg.ULP_RATE_KPA_S).any():
            return t
    return None


def uplink_latency(rng, cfg=Config):
    lat = rng.uniform(*cfg.LORA_LATENCY_S)
    if rng.random() < cfg.LORA_LOSS:
        lat += rng.uniform(*cfg.GSM_LATENCY_S)
    return lat


def streaming(det, episodes, cfg=Config, seed=11):
    rng = np.random.default_rng(seed)
    records = []
    false_alarms = 0
    monitored_s = 0.0
    for ep in episodes:
        sc = ep["scenario"]
        pred = det.predict_event(ep["X"], "xgb")
        flag = np.isin(pred, LEAKISH).astype(int)
        run = np.convolve(flag, np.ones(cfg.ALARM_CONSECUTIVE, dtype=int), mode="full")[:len(flag)]
        alarm_idx = np.where(run >= cfg.ALARM_CONSECUTIVE)[0]
        ts = np.array(ep["ts"])
        if sc["kind"] in ("leak", "burst"):
            pre = alarm_idx[ts[alarm_idx] < sc["onset"]]
            if len(pre):
                false_alarms += 1
            monitored_s += max(sc["onset"] - ts[0], 0)
            post = alarm_idx[ts[alarm_idx] >= sc["onset"]]
            rec = {"kind": sc["kind"], "pipe": sc["pipe"], "share_pct": 100 * sc["share"], "detected": bool(len(post))}
            if len(post):
                t_alarm = float(ts[post[0]])
                ulp = ulp_trigger_time(ep["pressure"], sc["onset"], cfg)
                sched = sc["onset"] + rng.uniform(0, cfg.SCHEDULED_UPLOAD_S)
                data_at = min(sched, ulp) if ulp is not None else sched
                rec["algorithmic_s"] = t_alarm + 1 - sc["onset"]
                rec["ulp_triggered"] = ulp is not None
                rec["total_s"] = max(t_alarm + 1, data_at) - sc["onset"] + uplink_latency(rng, cfg)
                ta = int(ts[post[0]])
                be = ta - cfg.WINDOW_S + 1
                args = (ep["pressure"], ep["twin_pressure"], ep["inlet_flow"], ep["metered_flow"])
                xa = window_features(*args, ta, cfg, be)
                loc = det.predict_pipe(xa, "mlp")[0]
                t10 = min(ta + 10, len(ep["inlet_flow"]) - 1)
                loc10 = det.predict_pipe(window_features(*args, t10, cfg, be), "mlp")[0]
                rec["loc_at_alarm_correct"] = loc == sc["pipe"]
                rec["loc_10s_correct"] = loc10 == sc["pipe"]
            records.append(rec)
        else:
            if len(alarm_idx):
                false_alarms += 1
            monitored_s += ts[-1] - ts[0]
            records.append({"kind": sc["kind"], "detected": bool(len(alarm_idx))})
    ev = [r for r in records if r["kind"] in ("leak", "burst")]
    det_ev = [r for r in ev if r["detected"]]
    summary = {
        "event_episodes": len(ev),
        "event_detection_rate": len(det_ev) / max(len(ev), 1),
        "non_event_episodes": len([r for r in records if r["kind"] not in ("leak", "burst")]),
        "non_event_false_alarm_episodes": len([r for r in records if r["kind"] not in ("leak", "burst") and r["detected"]]),
        "false_alarms_per_hour": false_alarms / max(monitored_s / 3600.0, 1e-9),
        "latency": {},
        "localisation_at_alarm": float(np.mean([r["loc_at_alarm_correct"] for r in det_ev])) if det_ev else None,
        "localisation_after_10s": float(np.mean([r["loc_10s_correct"] for r in det_ev])) if det_ev else None,
    }
    for kind in ("leak", "burst"):
        rs = [r for r in det_ev if r["kind"] == kind]
        if rs:
            tot = np.array([r["total_s"] for r in rs])
            alg = np.array([r["algorithmic_s"] for r in rs])
            summary["latency"][kind] = {
                "n": len(rs),
                "detection_rate": len(rs) / max(len([r for r in ev if r["kind"] == kind]), 1),
                "algorithmic_median_s": float(np.median(alg)),
                "total_median_s": float(np.median(tot)),
                "total_mean_s": float(tot.mean()),
                "total_p90_s": float(np.percentile(tot, 90)),
                "within_10s": float((tot <= 10).mean()),
                "within_60s": float((tot <= 60).mean()),
                "ulp_trigger_rate": float(np.mean([r["ulp_triggered"] for r in rs])),
            }
    return summary, records


def size_sensitivity(records):
    bins = [(10, 20), (20, 30), (30, 40), (40, 60), (60, 80), (80, 100)]
    out = []
    for lo, hi in bins:
        rs = [r for r in records if r["kind"] in ("leak", "burst") and lo <= r["share_pct"] < hi + (hi == 100)]
        if not rs:
            continue
        det = [r for r in rs if r["detected"]]
        out.append({
            "range": f"{lo:g}-{hi:g}%",
            "mid": (lo + hi) / 2,
            "n": len(rs),
            "detection_rate": len(det) / len(rs),
            "median_latency_s": float(np.median([r["total_s"] for r in det])) if det else None,
        })
    return out


def failsafe(det=None, n_bursts=200, seed=21, cfg=Config):
    rng = np.random.default_rng(seed)
    sim = TelemetrySimulator(seed=seed)
    net = sim.net
    hz = cfg.FAILSAFE_SAMPLE_HZ
    results = []
    for _ in range(n_bursts):
        sc = sim.random_scenario(300)
        while sc.kind != "burst":
            sc = sim.random_scenario(300)
        ep = sim.episode(300, sc)
        tp = ep["true_pressure"]
        t_hi = np.arange(0, 300, 1.0 / hz)
        p_hi = np.stack([np.interp(t_hi, np.arange(300), tp[:, i]) for i in range(tp.shape[1])], axis=1)
        p_hi = p_hi + rng.normal(0, cfg.SENSOR_NOISE_KPA, p_hi.shape)
        ref = tp[int(sc.onset) - 30:int(sc.onset) - 1].mean(axis=0)
        drop = ref[None, :] - p_hi
        start = int(sc.onset * hz)
        closed = {}
        trip_t = {}
        for i, node in enumerate(sim.nodes):
            cnt = 0
            for k in range(start, len(t_hi)):
                cnt = cnt + 1 if drop[k, i] > cfg.FAILSAFE_DROP_KPA else 0
                if cnt >= cfg.FAILSAFE_HOLD_SAMPLES:
                    trip_t[node] = t_hi[k]
                    closed[node] = t_hi[k] + 0.05 + rng.uniform(*cfg.VALVE_CLOSE_S)
                    break
        target = net.pipes[sc.pipe].end
        leak_pipes = {net.pipe_into[n]: t for n, t in closed.items()}
        isolating = [t for pid, t in leak_pipes.items() if _is_upstream_or_same(net, pid, sc.pipe)]
        rec = {"pipe": sc.pipe, "share_pct": 100 * sc.share, "triggered": bool(closed), "isolated": bool(isolating)}
        if isolating:
            t_close = min(isolating)
            rec["time_to_close_s"] = t_close - sc.onset
            lf = ep["leak_flow"]
            lost_before = float(lf[int(sc.onset):int(np.ceil(t_close))].sum()) / 1000.0
            rec["volume_lost_m3"] = lost_before
            rec["volume_12_6h_m3"] = float(lf[-1]) * 12.6 * 3600 / 1000.0
            unnecessary = [n for n in closed if not _is_downstream_or_same(net, n, sc.pipe) and net.pipe_into[n] != sc.pipe]
            rec["unnecessary_closures"] = len(unnecessary)
            if det is not None:
                tt = int(np.ceil(min(trip_t.values()))) + 1
                x = window_features(ep["pressure"], ep["twin_pressure"], ep["inlet_flow"], ep["metered_flow"], tt, cfg, tt - cfg.WINDOW_S)
                rec["reconcile_correct"] = det.predict_pipe(x, "mlp")[0] == sc.pipe
        results.append(rec)
    iso = [r for r in results if r["isolated"]]
    by_pipe = {}
    for pid in net.pipes:
        rs = [r for r in results if r["pipe"] == pid]
        if rs:
            by_pipe[pid] = sum(r["isolated"] for r in rs) / len(rs)
    return {
        "bursts": len(results),
        "isolation_rate": len(iso) / len(results),
        "isolation_rate_by_pipe": by_pipe,
        "time_to_close_median_s": float(np.median([r["time_to_close_s"] for r in iso])) if iso else None,
        "time_to_close_p90_s": float(np.percentile([r["time_to_close_s"] for r in iso], 90)) if iso else None,
        "volume_lost_median_m3": float(np.median([r["volume_lost_m3"] for r in iso])) if iso else None,
        "volume_saved_vs_manual_pct": float(100 * (1 - np.sum([r["volume_lost_m3"] for r in iso]) / np.sum([r["volume_12_6h_m3"] for r in iso]))) if iso else None,
        "unnecessary_closure_rate": float(np.mean([r["unnecessary_closures"] > 0 for r in iso])) if iso else None,
        "reconcile_accuracy": float(np.mean([r["reconcile_correct"] for r in iso if "reconcile_correct" in r])) if det is not None and iso else None,
    }, results


def _path_pipes(net, node):
    out = []
    while node in net.pipe_into:
        pid = net.pipe_into[node]
        out.append(pid)
        node = net.pipes[pid].start
    return out


def _is_upstream_or_same(net, pid, leak_pipe):
    return pid == leak_pipe or pid in _path_pipes(net, net.pipes[leak_pipe].start)


def _is_downstream_or_same(net, node, leak_pipe):
    return leak_pipe in _path_pipes(net, node)


def energy(episodes, cfg=Config):
    wakes, seconds = 0, 0
    for ep in episodes:
        if ep["scenario"]["kind"] != "normal":
            continue
        p = ep["pressure"]
        T = len(p)
        for i in range(p.shape[1]):
            quiet_until = 0
            for t in range(31, T):
                if t < quiet_until:
                    continue
                ref = p[t - 31:t - 1, i].mean()
                if abs(p[t, i] - ref) > cfg.ULP_DEVIATION_KPA or abs(p[t, i] - p[t - 1, i]) > cfg.ULP_RATE_KPA_S:
                    wakes += 1
                    quiet_until = t + cfg.ULP_REFRACTORY_S
            seconds += T - 31
    false_wakes_per_hour = wakes / max(seconds / 3600.0, 1e-9)
    sched_per_hour = 3600.0 / cfg.SCHEDULED_UPLOAD_S
    active_s_per_hour = (sched_per_hour + false_wakes_per_hour) * cfg.ACTIVE_WAKE_S
    duty = active_s_per_hour / 3600.0
    avg_ma = duty * cfg.ACTIVE_MA + (1 - duty) * cfg.SLEEP_ULP_MA
    return {
        "false_ulp_wakes_per_node_hour": false_wakes_per_hour,
        "duty_cycle_pct": 100 * duty,
        "average_current_ma": avg_ma,
        "battery_days_duty_cycled": cfg.BATTERY_MAH / avg_ma / 24.0,
        "battery_days_always_on": cfg.BATTERY_MAH / cfg.ALWAYS_ON_MA / 24.0,
        "energy_reduction_pct": 100 * (1 - avg_ma / cfg.ALWAYS_ON_MA),
    }


def inference_timing(det, X, repeats=2000):
    import time
    x = X[:1]
    det.predict_event(x, "xgb")
    t0 = time.perf_counter()
    for _ in range(repeats // 10):
        det.predict_event(x, "xgb")
        det.predict_pipe(x, "mlp")
    return (time.perf_counter() - t0) / (repeats // 10) * 1000.0


def _response_run(eng, rng, kind, where, percent, online, keep_trace=False, cfg=Config):
    """One scenario on the live engine: time to alert, time until the faulty pipe's valve is shut,
    and the pressure the junctions that lose water receive before, during and after."""
    net = eng.net
    eng.reset()
    for _ in range(cfg.WINDOW_S + cfg.BASELINE_S + 10 + int(rng.integers(0, 30))):
        eng.tick()
    if not online:
        eng.set_link(False)
    if kind == "leak":
        target, first = where, net.pipes[where].end
        eng.inject("leak", pipe=where, percent=percent)
    else:
        target, first = net.pipe_into[where], where
        eng.inject("burst", node=where, percent=percent)
    onset = eng.t
    affected = [net.sensor_nodes.index(n) for n in net.downstream_nodes(first)]
    others = [i for i in range(len(net.sensor_nodes)) if i not in affected]
    pre = np.array(eng.P)[-10:]
    before = pre.mean(axis=0)
    uplink = uplink_latency(rng, cfg) if online else 0.0
    downlink = uplink_latency(rng, cfg) if online else 0.0
    trace, detected, located, closed_by, pending, closed_at, p_detect = [], None, None, None, None, None, None
    for _ in range(90):
        eng.tick()
        elapsed = eng.t - onset
        trace.append(eng.last["pressure"].copy())
        if detected is None and eng.alert:
            detected = float(eng.alert["latency_s"]) + uplink
            located = target in str(eng.alert["pipe"]).split(", ")
            p_detect = eng.last["pressure"].copy()
            if online:
                # the cloud's close command (automatic for bursts, the operator's click for leaks) goes out once the
                # readings have reached the cloud, and only reaches the valve one downlink later
                pending = (eng.alert["pipe"], detected + downlink)
                closed_by = "cloud auto-close" if kind == "burst" else "operator on alert"
            else:
                closed_by = "edge fail-safe"
        if pending and elapsed >= pending[1]:
            eng.set_valve(pending[0], "close")
            pending = None
        if closed_at is None and eng.valves[target] == "closed":
            closed_at = elapsed
        if closed_at is not None and elapsed >= closed_at + 5:
            break
    trace = np.array(trace)
    rec = {"kind": kind, "where": where, "percent": percent, "online": online, "detected": detected is not None, "located": located}
    if detected is not None and closed_at is not None:
        closed_s = float(closed_at)
        fault = trace[:max(closed_at, 1)]
        rec.update({
            "detection_s": detected,
            "valve_closed_s": closed_s,
            "detect_to_closed_s": closed_s - detected,
            "closed_by": closed_by,
            "before_kpa": float(before[affected].mean()),
            "at_detection_kpa": float(p_detect[affected].mean()),
            "lowest_kpa": float(fault[:, affected].mean(axis=1).min()),
            "after_close_kpa": float(trace[-1, affected].mean()),
            "others_change_kpa": float(trace[-1, others].mean() - before[others].mean()) if others else 0.0,
        })
    if keep_trace:
        rec["trace"] = {
            "nodes": [net.sensor_nodes[i] for i in affected] + ([net.sensor_nodes[others[0]]] if others else []),
            "kpa": [[round(float(v), 1) for v in col] for col in np.vstack([pre, trace])[:, affected + others[:1]].T],
            "onset_idx": 9,
            "detected_s": detected,
            "closed_s": rec.get("valve_closed_s"),
        }
    return rec


def response_test(repeats=5, seed=31, cfg=Config):
    from .db import Database
    from .engine import LiveEngine
    rng = np.random.default_rng(seed)
    class ManualCloud(cfg):
        AUTO_ISOLATE_BURSTS = False

    eng = LiveEngine(Database(":memory:"), ManualCloud, seed=seed)
    net = eng.net
    groups = [("leak", f"Leak {cfg.LEAK_PERCENT:g}%", list(net.pipes), cfg.LEAK_PERCENT, True)]
    for pct in cfg.BURST_PERCENTS:
        groups.append(("burst", f"Burst {pct:g}%", list(net.junctions), pct, True))
    for pct in cfg.BURST_PERCENTS:
        groups.append(("burst", f"Burst {pct:g}%, cloud link down", list(net.junctions), pct, False))
    runs, summary, examples = [], [], {}
    for kind, label, places, pct, online in groups:
        rs = []
        for where in places:
            for r in range(repeats):
                keep = r == 0 and online and (kind, where, pct) in (("leak", "P3", cfg.LEAK_PERCENT), ("burst", "J5", max(cfg.BURST_PERCENTS)))
                rec = _response_run(eng, rng, kind, where, pct, online, keep, cfg)
                if "trace" in rec:
                    examples[label] = dict(rec.pop("trace"), where=where)
                rec["group"] = label
                rs.append(rec)
        ok = [r for r in rs if "valve_closed_s" in r]
        det = np.array([r["detection_s"] for r in ok])
        clo = np.array([r["valve_closed_s"] for r in ok])
        summary.append({
            "group": label, "runs": len(rs),
            "detection_rate": float(np.mean([r["detected"] for r in rs])),
            "located_rate": float(np.mean([bool(r["located"]) for r in rs if r["detected"]])) if any(r["detected"] for r in rs) else None,
            "detection_median_s": float(np.median(det)), "detection_p90_s": float(np.percentile(det, 90)),
            "detection_min_s": float(det.min()), "detection_max_s": float(det.max()),
            "valve_closed_median_s": float(np.median(clo)), "valve_closed_p90_s": float(np.percentile(clo, 90)),
            "detect_to_closed_median_s": float(np.median(clo - det)),
            "before_kpa": float(np.median([r["before_kpa"] for r in ok])),
            "at_detection_kpa": float(np.median([r["at_detection_kpa"] for r in ok])),
            "lowest_kpa": float(np.median([r["lowest_kpa"] for r in ok])),
            "after_close_kpa": float(np.median([r["after_close_kpa"] for r in ok])),
            "others_change_kpa": float(np.median([r["others_change_kpa"] for r in ok])),
        })
        runs += rs
    return {"repeats": repeats, "summary": summary, "examples": examples}, runs


def run_all(n_test=300, seed=99, out_dir=Config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    net = PipelineNetwork()
    det = LeakDetector.load()
    data, episodes = build_dataset(n_test, seed=seed, stride=1, keep_episodes=True)
    results = {
        "test_episodes": n_test,
        "test_windows": int(len(data["y_event"])),
        "window_level": window_level(det, data, net),
        "static_threshold": threshold_level(episodes, net),
        "localisation": localisation(det, data),
    }
    stream, records = streaming(det, episodes)
    results["streaming"] = stream
    results["size_sensitivity"] = size_sensitivity(records)
    fs, fs_records = failsafe(det)
    results["failsafe"] = fs
    results["energy"] = energy(episodes)
    results["inference_ms"] = inference_timing(det, data["X"])
    results["response"], _ = response_test()
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results, data, episodes, records, fs_records
