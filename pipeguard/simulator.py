import math
import numpy as np
from .network import PipelineNetwork, Leak, KPA_PER_M
from .config import Config

EVENT_CLASSES = ["normal", "demand_spike", "leak", "burst"]


def burst_wave_amp(lost_lps):
    return 1.2 * min(lost_lps, 25.0)


class Scenario:
    """A leak sits on `pipe` at `fraction` along it. A burst sits at junction `node`, which is the
    end of `pipe`. `share` is how much of the water heading past that point is lost."""

    def __init__(self, kind="normal", onset=None, pipe=None, fraction=0.5, share=0.0, ramp_s=1.0, node=None, spike_node=None, spike_factor=1.0):
        self.kind = kind
        self.onset = onset
        self.pipe = pipe
        self.fraction = fraction
        self.share = share
        self.ramp_s = ramp_s
        self.node = node
        self.spike_node = spike_node
        self.spike_factor = spike_factor

    @classmethod
    def burst_at(cls, net, node, share, onset, ramp_s=1.0):
        return cls("burst", onset=onset, pipe=net.pipe_into[node], fraction=1.0, share=share, ramp_s=ramp_s, node=node)

    def to_dict(self):
        return dict(vars(self))


class TelemetrySimulator:
    def __init__(self, network=None, seed=None, cfg=Config):
        self.net = network or PipelineNetwork()
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.nodes = self.net.sensor_nodes

    def random_scenario(self, duration):
        r = self.rng.random()
        onset = float(self.rng.uniform(90, duration - 90))
        if r < 0.2:
            return Scenario("normal")
        if r < 0.4:
            node = str(self.rng.choice(self.nodes))
            return Scenario("demand_spike", onset=onset, spike_node=node, spike_factor=float(self.rng.uniform(1.8, 3.5)), ramp_s=float(self.rng.uniform(3, 20)))
        if r < 0.75:
            pipe = str(self.rng.choice(list(self.net.pipes.keys())))
            return Scenario("leak", onset=onset, pipe=pipe, fraction=float(self.rng.uniform(0.1, 0.9)), share=float(self.rng.uniform(0.10, 0.35)), ramp_s=float(self.rng.uniform(10, 45)))
        node = str(self.rng.choice(self.nodes))
        return Scenario.burst_at(self.net, node, float(self.rng.uniform(0.50, 1.0)), onset, float(self.rng.uniform(0.5, 2.0)))

    def episode(self, duration=300, scenario=None):
        sc = scenario or self.random_scenario(duration)
        cfg = self.cfg
        n = len(self.nodes)
        base_mult = self.rng.uniform(0.4, 1.6)
        walk = np.cumsum(self.rng.normal(0, 0.003, duration))
        node_jitter = self.rng.normal(0, 0.02, (duration, n))
        drift = self.rng.uniform(-cfg.SENSOR_DRIFT_KPA, cfg.SENSOR_DRIFT_KPA, n)
        meter_bias = self.rng.normal(0, 0.01, n)
        true_p = np.zeros((duration, n))
        meas_p = np.zeros((duration, n))
        twin_p = np.zeros((duration, n))
        inlet = np.zeros(duration)
        metered = np.zeros(duration)
        leak_flow = np.zeros(duration)
        labels = np.zeros(duration, dtype=int)
        cache_twin = {}
        for t in range(duration):
            mult = max(base_mult + walk[t], 0.2)
            demand_true = {}
            expected = {}
            for i, node in enumerate(self.nodes):
                base = self.net.junctions[node].base_demand / 1000.0 * mult
                factor = 1.0
                if sc.kind == "demand_spike" and sc.onset is not None and t >= sc.onset and node == sc.spike_node:
                    prog = min((t - sc.onset) / max(sc.ramp_s, 0.1), 1.0)
                    factor = 1.0 + (sc.spike_factor - 1.0) * prog
                demand_true[node] = base * factor * (1 + node_jitter[t, i])
                expected[node] = base * factor
            leaks = []
            if sc.kind in ("leak", "burst") and t >= sc.onset:
                prog = min((t - sc.onset) / max(sc.ramp_s, 0.1), 1.0)
                leaks.append(Leak(sc.pipe, sc.fraction, sc.share * prog))
            res = self.net.solve(demand_true, leaks)
            demand_meter = {node: expected[node] * res["supply"][node] * (1 + meter_bias[i]) for i, node in enumerate(self.nodes)}
            key = tuple(round(v, 7) for v in demand_meter.values())
            twin = cache_twin.get(key) or self.net.solve(demand_meter)
            cache_twin[key] = twin
            for i, node in enumerate(self.nodes):
                true_p[t, i] = res["pressure_kpa"][node]
                twin_p[t, i] = twin["pressure_kpa"][node]
            inlet[t] = res["inlet_flow_lps"] * (1 + self.rng.normal(0, cfg.FLOW_NOISE_FRAC))
            metered[t] = sum(demand_meter.values()) * 1000.0
            leak_flow[t] = res["leak_flow_lps"]
            if sc.onset is not None and t >= sc.onset and sc.kind != "normal":
                # a leak still ramping in below EVENT_MIN_CUT is indistinguishable from noise, so it stays labelled normal
                cut_now = sum(lk.share for lk in leaks)
                if sc.kind == "demand_spike" and t - sc.onset < self.cfg.DEMAND_EVENT_S:
                    labels[t] = EVENT_CLASSES.index(sc.kind)
                elif sc.kind in ("leak", "burst") and cut_now >= self.cfg.EVENT_MIN_CUT:
                    labels[t] = EVENT_CLASSES.index(sc.kind)
        if sc.kind == "burst":
            t0 = int(math.ceil(sc.onset))
            tt = np.arange(duration) - t0
            wave = np.where(tt >= 0, np.exp(-tt / 2.5) * np.cos(2 * math.pi * tt / 3.0), 0.0)
            amp = burst_wave_amp(float(leak_flow.max()))
            true_p = true_p - (amp * wave)[:, None] * self._wave_weights(sc.pipe)[None, :]
        meas_p = np.maximum(true_p + drift[None, :] + self.rng.normal(0, cfg.SENSOR_NOISE_KPA, true_p.shape), 0.0)
        return {
            "scenario": sc,
            "nodes": list(self.nodes),
            "pressure": meas_p,
            "true_pressure": true_p,
            "twin_pressure": twin_p,
            "inlet_flow": inlet,
            "metered_flow": metered,
            "leak_flow": leak_flow,
            "labels": labels,
        }

    def _wave_weights(self, pipe_id):
        pipe = self.net.pipes[pipe_id]
        w = []
        for node in self.nodes:
            d = self._hops(node, pipe.end)
            w.append(1.0 / (1 + d))
        return np.array(w)

    def _hops(self, a, b):
        def path(x):
            p = [x]
            while x in self.net.pipe_into:
                x = self.net.pipes[self.net.pipe_into[x]].start
                p.append(x)
            return p
        pa, pb = path(a), path(b)
        common = next(x for x in pa if x in pb)
        return pa.index(common) + pb.index(common)
