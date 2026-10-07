import os
import tempfile
import numpy as np
import pytest
from pipeguard.network import PipelineNetwork, Leak
from pipeguard.simulator import TelemetrySimulator, Scenario
from pipeguard.features import episode_windows, feature_names
from pipeguard.config import Config


def base_demands(net):
    return {k: j.base_demand / 1000.0 for k, j in net.junctions.items()}


def test_leak_cuts_water_to_junctions_after_it():
    net = PipelineNetwork()
    d = base_demands(net)
    before = net.solve(d)
    after = net.solve(d, [Leak("P3", 0.5, 0.25)])
    assert before["inlet_flow_lps"] == pytest.approx(50.0, abs=0.01)
    assert after["inlet_flow_lps"] == pytest.approx(before["inlet_flow_lps"])
    for n in ("J3", "J4"):
        assert after["delivered_lps"][n] == pytest.approx(0.75 * before["delivered_lps"][n])
        assert after["pressure_kpa"][n] < before["pressure_kpa"][n] - 80
    for n in ("J1", "J2", "J5", "J6", "J7", "J8"):
        assert after["delivered_lps"][n] == pytest.approx(before["delivered_lps"][n])
    assert after["leak_flow_lps"] == pytest.approx(0.25 * (d["J3"] + d["J4"]) * 1000)


def test_full_burst_cuts_junction_and_those_after_it():
    net = PipelineNetwork()
    r = net.solve(base_demands(net), [Leak(net.pipe_into["J5"], 1.0, 1.0)])
    assert r["delivered_lps"]["J5"] == 0 and r["delivered_lps"]["J6"] == 0
    assert r["pressure_kpa"]["J5"] == 0 and r["pressure_kpa"]["J6"] == 0
    assert r["delivered_lps"]["J3"] > 0


def test_closed_valve_isolates_and_stops_leak():
    net = PipelineNetwork()
    r = net.solve(base_demands(net), [Leak("P3", 0.5, 0.25)], closed_pipes=["P3"])
    assert r["pressure_kpa"]["J3"] == 0 and r["pressure_kpa"]["J4"] == 0
    assert r["leak_flow_lps"] == 0
    assert net.isolated_nodes(["P3"]) == {"J3", "J4"}


def test_features_shape():
    sim = TelemetrySimulator(seed=3)
    ep = sim.episode(150, Scenario("leak", onset=100, pipe="P6", share=0.25, ramp_s=10))
    ts, X = episode_windows(ep)
    assert X.shape == (len(ts), len(feature_names(sim.nodes)))
    assert np.isfinite(X).all()


@pytest.fixture
def client():
    class TestConfig(Config):
        START_ENGINE = False
        DATABASE = os.path.join(tempfile.mkdtemp(), "test.db")
    from pipeguard import create_app
    app = create_app(TestConfig)
    for _ in range(80):
        app.engine.tick()
    return app.test_client(), app


def test_api_state_and_scenario(client):
    c, app = client
    s = c.get("/api/state").get_json()
    assert s["ready"] and len(s["nodes"]) == 8
    assert c.post("/api/scenario", json={"kind": "burst", "node": "J6", "percent": 100}).status_code == 200
    for _ in range(10):
        app.engine.tick()
    s = c.get("/api/state").get_json()
    if s["model_loaded"]:
        assert s["alert"] is not None
    assert c.post("/api/scenario", json={"kind": "leak", "pipe": "P99"}).status_code == 400
    assert c.post("/api/scenario", json={"kind": "burst", "node": "J99"}).status_code == 400


def test_edge_failsafe_closes_valve_when_offline(client):
    c, app = client
    c.post("/api/link", json={"up": False})
    c.post("/api/scenario", json={"kind": "burst", "node": "J8", "percent": 100})
    for _ in range(8):
        app.engine.tick()
    s = c.get("/api/state").get_json()
    assert s["valves"]["P8"] in ("closing", "closed")


def test_telemetry_ingest(client):
    c, _ = client
    r = c.post("/api/telemetry", json=[{"node": "J2", "pressure_kpa": 450.0}])
    assert r.get_json()["accepted"] == 1
    assert c.post("/api/telemetry", json={"bad": 1}).status_code == 400
    assert c.post("/api/telemetry", json={"node": "J2", "pressure_kpa": "nan"}).status_code == 400


def test_buffered_telemetry_does_not_override_fresh(client):
    c, app = client
    batch = [
        {"node": "J3", "pressure_kpa": 300.0, "age_s": 60},
        {"node": "J3", "pressure_kpa": 310.0, "age_s": 30},
        {"node": "J3", "pressure_kpa": 441.0},
    ]
    assert c.post("/api/telemetry", json=batch).get_json()["accepted"] == 3
    app.engine.tick()
    j3 = next(n for n in c.get("/api/state").get_json()["nodes"] if n["id"] == "J3")
    assert j3["pressure"] == 441.0 and j3["source"] == "device"
    rows = app.db.query("SELECT pressure_kpa FROM readings WHERE node='J3' AND source='device-buffered' ORDER BY ts")
    assert [r["pressure_kpa"] for r in rows] == [300.0, 310.0]


def test_scenarios_cut_water_to_the_right_junctions(client):
    c, app = client
    assert c.post("/api/scenario", json={"kind": "leak", "pipe": "P3"}).status_code == 200
    assert c.post("/api/scenario", json={"kind": "burst", "node": "J5", "percent": 50}).status_code == 200
    leak, burst = app.engine.leaks
    assert leak["percent"] == 25 and leak["affected"] == ["J3", "J4"]
    assert burst["percent"] == 50 and burst["affected"] == ["J5", "J6"] and burst["pipe"] == "P5"
    for _ in range(50):
        app.engine.tick()
    nodes = {n["id"]: n for n in c.get("/api/state").get_json()["nodes"]}
    assert nodes["J3"]["received_lps"] == pytest.approx(0.75 * nodes["J3"]["demand_lps"], rel=0.03)
    assert nodes["J6"]["received_lps"] == pytest.approx(0.5 * nodes["J6"]["demand_lps"], rel=0.03)
    assert nodes["J1"]["received_lps"] == pytest.approx(nodes["J1"]["demand_lps"], rel=0.03)
    assert c.post("/api/scenario", json={"kind": "burst", "node": "J6", "percent": 150}).status_code == 400
    assert c.post("/api/scenario", json={"kind": "burst", "node": "J6", "percent": "x"}).status_code == 400


def test_bad_query_args_and_unknown_incident(client):
    c, _ = client
    assert c.get("/api/state?seconds=abc").status_code == 200
    assert c.get("/api/readings/J1?minutes=x").status_code == 200
    assert c.post("/api/alerts/99999/resolve").status_code == 404
