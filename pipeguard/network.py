from dataclasses import dataclass

G = 9.81
KPA_PER_M = 9.81


@dataclass
class Junction:
    id: str
    elevation: float
    base_demand: float
    x: float
    y: float


@dataclass
class Pipe:
    id: str
    start: str
    end: str
    length: float
    diameter: float
    roughness: float = 130.0


@dataclass
class Leak:
    pipe: str
    fraction: float
    share: float


class PipelineNetwork:
    def __init__(self):
        self.reservoir = {"id": "R1", "head": 62.0, "x": 60, "y": 200}
        self.junctions = {
            j.id: j
            for j in [
                Junction("J1", 12.0, 6.8, 215, 200),
                Junction("J2", 14.0, 5.8, 400, 200),
                Junction("J3", 15.5, 7.8, 570, 90),
                Junction("J4", 17.0, 5.3, 730, 90),
                Junction("J5", 13.0, 7.3, 570, 280),
                Junction("J6", 14.5, 6.3, 730, 280),
                Junction("J7", 11.0, 5.8, 300, 390),
                Junction("J8", 12.5, 4.9, 480, 420),
            ]
        }
        self.pipes = {
            p.id: p
            for p in [
                Pipe("P1", "R1", "J1", 450, 0.375),
                Pipe("P2", "J1", "J2", 380, 0.300),
                Pipe("P3", "J2", "J3", 420, 0.200),
                Pipe("P4", "J3", "J4", 360, 0.150),
                Pipe("P5", "J2", "J5", 400, 0.200),
                Pipe("P6", "J5", "J6", 350, 0.150),
                Pipe("P7", "J1", "J7", 500, 0.200),
                Pipe("P8", "J7", "J8", 380, 0.150),
            ]
        }
        self.sensor_nodes = list(self.junctions.keys())
        self.inlet_pipe = "P1"
        self.children = {}
        self.pipe_into = {}
        for p in self.pipes.values():
            self.children.setdefault(p.start, []).append(p.end)
            self.pipe_into[p.end] = p.id
        self.order = self._topological()

    def _topological(self):
        order, stack = [], [self.reservoir["id"]]
        while stack:
            n = stack.pop(0)
            order.append(n)
            stack.extend(self.children.get(n, []))
        return order

    @staticmethod
    def headloss(pipe, flow, length=None):
        length = pipe.length if length is None else length
        q = abs(flow)
        if q < 1e-9:
            return 0.0
        return 10.67 * length * q ** 1.852 / (pipe.roughness ** 1.852 * pipe.diameter ** 4.87)

    def solve(self, demands, leaks=None, closed_pipes=None):
        """Each leak spills `share` of the water passing its point, so every junction after it
        receives that much less. The reservoir still sends what the junctions ask for, and the
        difference is lost. Pressure at a short-supplied junction follows the orifice law
        (delivered flow goes with the square root of pressure), so it falls with supply squared."""
        leaks = leaks or []
        isolated = self.isolated_nodes(closed_pipes)
        root = self.reservoir["id"]
        need = {}
        for n in reversed(self.order):
            if n == root:
                continue
            need[n] = 0.0 if n in isolated else demands.get(n, 0.0) + sum(need[c] for c in self.children.get(n, []))
        supply = {root: 1.0}
        heads = {root: self.reservoir["head"]}
        pipe_flow = {}
        leak_flows = [0.0 for _ in leaks]
        for n in self.order:
            if n == root:
                continue
            pid = self.pipe_into[n]
            pipe = self.pipes[pid]
            if n in isolated:
                supply[n] = 0.0
                pipe_flow[pid] = 0.0
                heads[n] = self.junctions[n].elevation
                continue
            f = supply[pipe.start]
            total = need[n]
            pipe_flow[pid] = f * total
            h = heads[pipe.start]
            pos = 0.0
            for i, lk in sorted([(i, lk) for i, lk in enumerate(leaks) if lk.pipe == pid], key=lambda t: t[1].fraction):
                h -= self.headloss(pipe, f * total, pipe.length * (lk.fraction - pos))
                share = min(max(lk.share, 0.0), 1.0)
                leak_flows[i] = f * share * total
                f *= 1.0 - share
                pos = lk.fraction
            h -= self.headloss(pipe, f * total, pipe.length * (1 - pos))
            heads[n] = h
            supply[n] = f
        pressures = {n: max(heads[n] - j.elevation, 0.0) * KPA_PER_M * supply[n] ** 2 for n, j in self.junctions.items()}
        return {
            "pressure_kpa": pressures,
            "inlet_flow_lps": pipe_flow.get(self.inlet_pipe, 0.0) * 1000.0,
            "leak_flow_lps": sum(leak_flows) * 1000.0,
            "leak_flows_lps": [v * 1000.0 for v in leak_flows],
            "pipe_flow_lps": {k: v * 1000.0 for k, v in pipe_flow.items()},
            "delivered_lps": {n: 0.0 if n in isolated else demands.get(n, 0.0) * supply[n] * 1000.0 for n in self.junctions},
            "supply": {n: supply[n] for n in self.junctions},
        }

    def isolated_nodes(self, closed_pipes):
        closed = set(closed_pipes or [])
        out = set()
        for n in self.order:
            if n == self.reservoir["id"]:
                continue
            pid = self.pipe_into[n]
            if pid in closed or self.pipes[pid].start in out:
                out.add(n)
        return out

    def downstream_nodes(self, node):
        out, stack = [], [node]
        while stack:
            n = stack.pop(0)
            out.append(n)
            stack.extend(self.children.get(n, []))
        return out

    def upstream_pipe_of(self, node):
        return self.pipe_into.get(node)

    def topology(self):
        return {
            "reservoir": self.reservoir,
            "junctions": [vars(j) for j in self.junctions.values()],
            "pipes": [vars(p) for p in self.pipes.values()],
        }
