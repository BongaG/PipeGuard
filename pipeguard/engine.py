import copy
import math
import threading
import time
from collections import deque
import numpy as np
from .config import Config
from .network import PipelineNetwork, Leak
from .simulator import TelemetrySimulator, EVENT_CLASSES, burst_wave_amp
from .features import window_features
from .detector import LeakDetector

LEAKISH = {"leak", "burst"}


class EdgeNode:
    def __init__(self, node, valve_pipe, cfg):
        self.node = node
        self.valve_pipe = valve_pipe
        self.cfg = cfg
        self.awake_until = -1
        self.next_upload = 0
        self.wakes = 0
        self.history = deque(maxlen=32)
        self.low_count = 0
        self.failsafe_fired = False

    def step(self, t, pressure, link_up):
        cfg = self.cfg
        reason = None
        if len(self.history) >= 31:
            ref = float(np.mean(list(self.history)[-31:-1]))
            dev = abs(pressure - ref)
            rate = abs(pressure - self.history[-1])
            if t > self.awake_until and (dev > cfg.ULP_DEVIATION_KPA or rate > cfg.ULP_RATE_KPA_S):
                reason = "interrupt"
                self.awake_until = t + cfg.ULP_REFRACTORY_S
                self.wakes += 1
            drop = ref - pressure
            if drop > cfg.FAILSAFE_DROP_KPA:
                self.low_count += 1
            else:
                self.low_count = 0
        if t >= self.next_upload:
            reason = reason or "scheduled"
            self.next_upload = t + cfg.SCHEDULED_UPLOAD_S
        self.history.append(pressure)
        hold = max(1, round(cfg.FAILSAFE_HOLD_SAMPLES / cfg.FAILSAFE_SAMPLE_HZ))
        fire = False
        if self.low_count >= hold and not self.failsafe_fired and (link_up is False or not cfg.FAILSAFE_ONLY_OFFLINE):
            self.failsafe_fired = True
            fire = True
        return reason, fire

    def state(self, t):
        return "awake" if t <= self.awake_until or t % self.cfg.SCHEDULED_UPLOAD_S < self.cfg.ACTIVE_WAKE_S else "asleep"


class LiveEngine:
    def __init__(self, db, cfg=Config, seed=None):
        self.cfg = cfg
        self.db = db
        self.net = PipelineNetwork()
        self.helper = TelemetrySimulator(self.net, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.nodes = self.net.sensor_nodes
        self.detector = LeakDetector.load(cfg.MODEL_DIR)
        self.lock = threading.RLock()
        self.running = False
        self.thread = None
        self.reset()

    def reset(self):
        with self.lock:
            n = len(self.nodes)
            self.t = 0
            self.mult = 1.0
            self.drift = self.rng.uniform(-self.cfg.SENSOR_DRIFT_KPA, self.cfg.SENSOR_DRIFT_KPA, n)
            self.meter_bias = self.rng.normal(0, 0.01, n)
            self.P = deque(maxlen=self.cfg.BUFFER_S)
            self.TW = deque(maxlen=self.cfg.BUFFER_S)
            self.IN = deque(maxlen=self.cfg.BUFFER_S)
            self.ME = deque(maxlen=self.cfg.BUFFER_S)
            self.RX = deque(maxlen=self.cfg.BUFFER_S)
            self.leaks = []
            self.spikes = {}
            self.valves = {p: "open" for p in self.net.pipes}
            self.valve_timers = {}
            self.link_up = True
            self.link_mode = "LoRaWAN"
            self.overrides = {}
            self.edges = {n: EdgeNode(n, self.net.pipe_into[n], self.cfg) for n in self.nodes}
            self.flag_run = 0
            self.clear_run = 0
            self.alert = None
            self.latched_base = None
            self.last_proba = [1.0, 0.0, 0.0, 0.0]
            self.last = {}
            self.log = deque(maxlen=40)
            self._log("System started, all valves open, link on LoRaWAN")

    def _log(self, text, level="info"):
        self.log.appendleft({"t": self.t, "wall": time.time(), "text": text, "level": level})

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            start = time.time()
            try:
                self.tick()
            except Exception as exc:
                self._log(f"Engine error: {exc}", "error")
            time.sleep(max(self.cfg.TICK_S - (time.time() - start), 0.05))

    def closed_pipes(self):
        return [p for p, s in self.valves.items() if s == "closed"]

    def tick(self):
        with self.lock:
            cfg = self.cfg
            t = self.t
            self.mult = float(np.clip(self.mult + self.rng.normal(0, 0.004), 0.6, 1.4))
            demand_true, expected = {}, {}
            closed = self.closed_pipes()
            cut = self.net.isolated_nodes(closed)
            for node in self.nodes:
                base = self.net.junctions[node].base_demand / 1000.0 * self.mult
                factor = 1.0
                if node in self.spikes:
                    s = self.spikes[node]
                    prog = min((t - s["onset"]) / s["ramp"], 1.0)
                    factor = 1.0 + (s["factor"] - 1.0) * prog
                if node in cut:
                    factor = 0.0
                demand_true[node] = base * factor * (1 + self.rng.normal(0, 0.02))
                expected[node] = base * factor
            leaks = []
            for lk in self.leaks:
                prog = min((t - lk["onset"]) / lk["ramp"], 1.0)
                leaks.append(Leak(lk["pipe"], lk["fraction"], lk["share"] * prog))
            res = self.net.solve(demand_true, leaks, closed)
            demand_meter = {n: expected[n] * res["supply"][n] * (1 + self.meter_bias[i]) for i, n in enumerate(self.nodes)}
            twin = self.net.solve(demand_meter, closed_pipes=closed)
            wave = np.zeros(len(self.nodes))
            for lk, lost in zip(self.leaks, res["leak_flows_lps"]):
                if lk["kind"] == "burst":
                    dt = t - lk["onset"]
                    if 0 <= dt < 20:
                        wave += burst_wave_amp(lost) * math.exp(-dt / 2.5) * math.cos(2 * math.pi * dt / 3.0) * self.helper._wave_weights(lk["pipe"])
            p = np.array([res["pressure_kpa"][n] for n in self.nodes]) - wave
            p = np.maximum(p + self.drift + self.rng.normal(0, cfg.SENSOR_NOISE_KPA, len(self.nodes)), 0.0)
            sources = ["sim"] * len(self.nodes)
            now = time.time()
            for i, node in enumerate(self.nodes):
                ov = self.overrides.get(node)
                if ov and now - ov["wall"] < cfg.TELEMETRY_OVERRIDE_S:
                    p[i] = ov["value"]
                    sources[i] = "device"
            tw = np.array([twin["pressure_kpa"][n] for n in self.nodes])
            inlet = res["inlet_flow_lps"] * (1 + self.rng.normal(0, cfg.FLOW_NOISE_FRAC))
            metered = sum(demand_meter.values()) * 1000.0
            self.P.append(p)
            self.TW.append(tw)
            self.IN.append(inlet)
            self.ME.append(metered)
            received = np.array([demand_meter[n] * 1000.0 for n in self.nodes])
            self.RX.append(received)
            edge_state = {}
            for i, node in enumerate(self.nodes):
                reason, fire = self.edges[node].step(t, p[i], self.link_up)
                edge_state[node] = reason
                if reason == "interrupt":
                    self._log(f"{node} edge node woke on a pressure interrupt")
                if fire:
                    self._log(f"{node} edge fail-safe closed valve on {self.edges[node].valve_pipe} without the cloud", "danger")
                    self._actuate(self.edges[node].valve_pipe, "close", f"edge:{node}")
                    self._raise_edge_alert(node)
            self._advance_valves()
            self._detect()
            self.last = {
                "pressure": p, "twin": tw, "inlet": inlet, "metered": metered,
                "received": received, "demand": np.array([expected[n] * 1000.0 for n in self.nodes]),
                "leak_flow": res["leak_flow_lps"], "pipe_flow": res["pipe_flow_lps"], "sources": sources, "edge": edge_state,
            }
            rows = [(now, n, float(p[i]), float(tw[i]), sources[i]) for i, n in enumerate(self.nodes)]
            self.db.executemany("INSERT INTO readings (ts, node, pressure_kpa, twin_kpa, source) VALUES (?,?,?,?,?)", rows)
            self.db.execute("INSERT INTO flows (ts, inlet_lps, metered_lps) VALUES (?,?,?)", (now, inlet, metered))
            if t % 300 == 0:
                self.db.prune(now - 3600)
            self.t += 1

    def _advance_valves(self):
        for pipe, until in list(self.valve_timers.items()):
            if self.t >= until:
                state = self.valves[pipe]
                self.valves[pipe] = "closed" if state == "closing" else "open"
                del self.valve_timers[pipe]
                self._log(f"Valve on {pipe} is now {self.valves[pipe]}")

    def _actuate(self, pipe, action, actor):
        if action == "close" and self.valves[pipe] in ("open", "opening"):
            self.valves[pipe] = "closing"
            self.valve_timers[pipe] = self.t + math.ceil(self.rng.uniform(*self.cfg.VALVE_CLOSE_S))
        elif action == "open" and self.valves[pipe] in ("closed", "closing"):
            self.valves[pipe] = "opening"
            self.valve_timers[pipe] = self.t + math.ceil(self.rng.uniform(*self.cfg.VALVE_CLOSE_S))
            for e in self.edges.values():
                if e.valve_pipe == pipe:
                    e.failsafe_fired = False
                    e.low_count = 0
        else:
            return False
        self.db.execute("INSERT INTO valve_events (ts, pipe, action, actor) VALUES (?,?,?,?)", (time.time(), pipe, action, actor))
        return True

    def _onset_latency(self):
        if not self.leaks:
            return None
        return float(self.t + 1 - min(lk["onset"] for lk in self.leaks))

    def _raise_edge_alert(self, node):
        pipe = self.edges[node].valve_pipe
        if self.alert:
            if self.alert.get("detected_by") == "edge fail-safe":
                self.alert["edge_closed"].append(pipe)
                self.alert["pipe"] = ", ".join(self.alert["edge_closed"])
                self.db.execute("UPDATE alerts SET pipe=? WHERE id=?", (self.alert["pipe"], self.alert["id"]))
            return
        aid = self.db.execute(
            "INSERT INTO alerts (opened_at, kind, pipe, detected_by, latency_s, note) VALUES (?,?,?,?,?,?)",
            (time.time(), "burst", self.edges[node].valve_pipe, "edge fail-safe", self._onset_latency(), "Cloud link down, local threshold tripped"),
        )
        self.alert = {"id": aid, "kind": "burst", "pipe": pipe, "pipe_confidence": None, "event_confidence": None, "detected_by": "edge fail-safe", "opened_t": self.t, "latency_s": self._onset_latency(), "edge_closed": [pipe]}

    def _detect(self):
        cfg = self.cfg
        if self.detector is None or not self.link_up or len(self.P) < cfg.WINDOW_S + cfg.BASELINE_S:
            self.last_proba = None
            return
        P, TW = np.array(self.P), np.array(self.TW)
        IN, ME = np.array(self.IN), np.array(self.ME)
        t = len(IN) - 1
        x = window_features(P, TW, IN, ME, t, cfg)
        proba = self.detector.predict_event_proba(x, "xgb")[0]
        self.last_proba = [float(v) for v in proba]
        kind = EVENT_CLASSES[int(np.argmax(proba))]
        if kind in LEAKISH:
            self.flag_run += 1
            self.clear_run = 0
        else:
            self.flag_run = 0
            self.clear_run += 1
        if self.alert is None and self.flag_run >= cfg.ALARM_CONSECUTIVE:
            self.latched_base = self.t - cfg.WINDOW_S + 1
            self._open_alert(kind, float(proba.max()), P, TW, IN, ME, t)
        elif self.alert is not None:
            # once a valve has moved the hydraulics are outside what the model was trained on, so keep the location found before
            valves_moved = any(s != "open" for s in self.valves.values())
            if self.latched_base is not None and not valves_moved and (self.t - self.alert["opened_t"]) % 5 == 0:
                idx = self.latched_base - (self.t - t)
                if idx > cfg.BASELINE_S // 2:
                    self._localise(P, TW, IN, ME, t, idx)
            if kind == "burst" and self.alert["kind"] == "leak":
                self.alert["kind"] = "burst"
                self.db.execute("UPDATE alerts SET kind=? WHERE id=?", ("burst", self.alert["id"]))
            if self.clear_run >= cfg.ALERT_CLEAR_S and not self.leaks:
                self.resolve_alert("Pressure returned to the twin's expected profile")

    def _open_alert(self, kind, conf, P, TW, IN, ME, t):
        aid = self.db.execute(
            "INSERT INTO alerts (opened_at, kind, detected_by, event_confidence, latency_s) VALUES (?,?,?,?,?)",
            (time.time(), kind, "cloud model", conf, self._onset_latency()),
        )
        self.alert = {"id": aid, "kind": kind, "pipe": None, "pipe_confidence": None, "event_confidence": conf, "detected_by": "cloud model", "opened_t": self.t, "latency_s": self._onset_latency()}
        self._localise(P, TW, IN, ME, t, self.latched_base - (self.t - t))
        lat = self.alert["latency_s"]
        extra = f" {lat:.0f} s after onset" if lat is not None else ""
        where = f"at {self.alert['junction']} (fed by {self.alert['pipe']})" if kind == "burst" else f"on {self.alert['pipe']}"
        self._log(f"{kind.title()} detected{extra}, most likely {where}", "danger")
        if kind == "burst" and self.cfg.AUTO_ISOLATE_BURSTS and (self.alert["pipe_confidence"] or 0) >= self.cfg.AUTO_ISOLATE_CONFIDENCE:
            if self._actuate(self.alert["pipe"], "close", "cloud auto-isolate"):
                self._log(f"Cloud closed the valve on {self.alert['pipe']} automatically", "danger")

    def _localise(self, P, TW, IN, ME, t, base_end):
        x = window_features(P, TW, IN, ME, t, self.cfg, base_end)
        proba = self.detector.predict_pipe_proba(x, "mlp")[0]
        order = np.argsort(-proba)
        pipe = self.detector.pipes[int(order[0])]
        conf = float(proba[order[0]])
        if self.alert.get("pipe") and self.alert["pipe"] != pipe and self.alert.get("detected_by") == "cloud model":
            self._log(f"Location refined from {self.alert['pipe']} to {pipe} as more data arrived", "warn")
        self.alert["pipe"] = pipe
        self.alert["junction"] = self.net.pipes[pipe].end
        self.alert["pipe_confidence"] = conf
        self.alert["ranking"] = [{"pipe": self.detector.pipes[int(i)], "p": float(proba[i])} for i in order[:3]]
        self.db.execute("UPDATE alerts SET pipe=?, pipe_confidence=?, kind=? WHERE id=?", (pipe, conf, self.alert["kind"], self.alert["id"]))

    def resolve_alert(self, note="Resolved by operator"):
        with self.lock:
            if not self.alert:
                return False
            self.db.execute("UPDATE alerts SET status='resolved', closed_at=?, note=COALESCE(note || '. ', '') || ? WHERE id=?", (time.time(), note, self.alert["id"]))
            self._log(f"Alert #{self.alert['id']} resolved. {note}")
            self.alert = None
            self.latched_base = None
            self.flag_run = 0
            return True

    def inject(self, kind, pipe=None, node=None, percent=None, fraction=None, factor=None):
        with self.lock:
            if kind in LEAKISH:
                if kind == "leak":
                    if pipe not in self.net.pipes:
                        raise ValueError("Unknown pipe")
                    first = self.net.pipes[pipe].end
                    percent = float(percent if percent is not None else self.cfg.LEAK_PERCENT)
                    fraction = float(fraction or self.rng.uniform(0.2, 0.8))
                    ramp = float(self.rng.uniform(10, 45))
                else:
                    if node not in self.net.junctions:
                        raise ValueError("Unknown junction")
                    first = node
                    pipe = self.net.pipe_into[node]
                    percent = float(percent if percent is not None else max(self.cfg.BURST_PERCENTS))
                    fraction = 1.0
                    ramp = float(self.rng.uniform(0.5, 2.0))
                if not 0 < percent <= 100:
                    raise ValueError("percent must be between 0 and 100")
                affected = self.net.downstream_nodes(first)
                self.leaks.append({"kind": kind, "pipe": pipe, "node": first, "affected": affected, "share": percent / 100.0, "percent": percent, "fraction": fraction, "onset": self.t, "ramp": ramp})
                where = f"on {pipe}" if kind == "leak" else f"at {first}"
                self._log(f"Test scenario: {kind} {where} cutting {percent:g}% of the water to {', '.join(affected)}", "warn")
            elif kind == "demand_spike":
                if node not in self.net.junctions:
                    raise ValueError("Unknown node")
                factor = float(factor or 2.0)
                self.spikes[node] = {"factor": factor, "onset": self.t, "ramp": float(self.rng.uniform(3, 20))}
                self._log(f"Test scenario: metered demand at {node} rising to {factor:.1f}x", "warn")
            else:
                raise ValueError("Unknown scenario")

    def repair(self):
        with self.lock:
            self.leaks = []
            self.spikes = {}
            self._log("All test scenarios cleared, pipes marked as repaired")

    def set_valve(self, pipe, action, actor="operator"):
        with self.lock:
            if pipe not in self.valves:
                raise ValueError("Unknown pipe")
            ok = self._actuate(pipe, action, actor)
            if ok:
                self._log(f"Operator asked to {action} the valve on {pipe}")
            return ok

    def set_link(self, up):
        with self.lock:
            self.link_up = bool(up)
            self.link_mode = "LoRaWAN" if up else "offline"
            self._log("Cloud link restored" if up else "Cloud link cut, edge nodes now on local fail-safe", "warn" if not up else "info")
            if up:
                self._reconcile_edge_closures()

    def _reconcile_edge_closures(self):
        a = self.alert
        if not a or a.get("detected_by") != "edge fail-safe" or self.detector is None:
            return
        cfg = self.cfg
        P, TW = np.array(self.P), np.array(self.TW)
        IN, ME = np.array(self.IN), np.array(self.ME)
        last = len(IN) - 1
        trip = last - (self.t - 1 - a["opened_t"])
        if trip - cfg.WINDOW_S < cfg.BASELINE_S // 2:
            return
        t_eval = min(trip + 1, last)
        self._localise(P, TW, IN, ME, t_eval, trip - cfg.WINDOW_S + 1)
        keep = a["pipe"]
        self._log(f"Cloud re-checked the edge trip and located the burst on {keep}", "warn")
        for pipe in a["edge_closed"]:
            if pipe != keep and self.valves.get(pipe) in ("closed", "closing"):
                self._actuate(pipe, "open", "cloud reconcile")
                self._log(f"Cloud reopened {pipe}, it was closed by the edge but is not the burst pipe")
        if self.valves.get(keep) == "open":
            self._actuate(keep, "close", "cloud reconcile")
        a["edge_closed"] = [keep]

    def ingest(self, node, value, age_s=0.0):
        value = float(value)
        age_s = float(age_s or 0.0)
        if not math.isfinite(value) or not math.isfinite(age_s) or age_s < 0:
            raise ValueError("Bad reading")
        wall = time.time() - age_s
        with self.lock:
            if node not in self.net.junctions:
                raise ValueError("Unknown node")
            cur = self.overrides.get(node)
            if cur is None or wall >= cur["wall"]:
                self.overrides[node] = {"value": value, "wall": wall}
        if age_s > 0:
            self.db.execute("INSERT INTO readings (ts, node, pressure_kpa, source) VALUES (?,?,?,?)", (wall, node, value, "device-buffered"))

    def snapshot(self, seconds=180):
        with self.lock:
            if not self.last:
                return {"ready": False}
            P = np.array(self.P)[-seconds:]
            TW = np.array(self.TW)[-seconds:]
            RX = np.array(self.RX)[-seconds:]
            ts =list(range(self.t - len(P), self.t))
            nodes = []
            for i, n in enumerate(self.nodes):
                nodes.append({
                    "id": n,
                    "pressure": round(float(self.last["pressure"][i]), 1),
                    "twin": round(float(self.last["twin"][i]), 1),
                    "residual": round(float(self.last["pressure"][i] - self.last["twin"][i]), 1),
                    "received_lps": round(float(self.last["received"][i]), 2),
                    "demand_lps": round(float(self.last["demand"][i]), 2),
                    "edge": self.edges[n].state(self.t - 1),
                    "wakes": self.edges[n].wakes,
                    "source": self.last["sources"][i],
                    "valve_pipe": self.edges[n].valve_pipe,
                })
            return {
                "ready": True,
                "t": self.t,
                "model_loaded": self.detector is not None,
                "link_up": self.link_up,
                "link_mode": self.link_mode,
                "demand_multiplier": round(self.mult, 3),
                "inlet_lps": round(self.last["inlet"], 2),
                "metered_lps": round(self.last["metered"], 2),
                "imbalance_lps": round(self.last["inlet"] - self.last["metered"], 2),
                "lost_lps": round(self.last["leak_flow"], 2),
                "proba": dict(zip(EVENT_CLASSES, [round(v, 3) for v in self.last_proba])) if self.last_proba else None,
                "nodes": nodes,
                "valves": dict(self.valves),
                "pipe_flow": {k: round(v, 2) for k, v in self.last["pipe_flow"].items()},
                "alert": copy.deepcopy(self.alert),
                "scenarios": {"leaks": [dict(l) for l in self.leaks], "spikes": {k: dict(v) for k, v in self.spikes.items()}},
                "log": list(self.log)[:15],
                "series": {
                    "t": ts,
                    "pressure": {n: [round(float(v), 1) for v in P[:, i]] for i, n in enumerate(self.nodes)},
                    "twin": {n: [round(float(v), 1) for v in TW[:, i]] for i, n in enumerate(self.nodes)},
                    "received": {n: [round(float(v), 2) for v in RX[:, i]] for i, n in enumerate(self.nodes)},
                    "inlet": [round(float(v), 2) for v in list(self.IN)[-seconds:]],
                    "metered": [round(float(v), 2) for v in list(self.ME)[-seconds:]],
                },
            }
