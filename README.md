# PipeGuard: pipeline pressure and leak detection

PipeGuard is the software prototype for the paper *Enhancing IoT system for monitoring urban water availability* (Group 21, PRJT302, Durban University of Technology). It implements the pipeline pressure and leak detection method selected in the paper:

- a digital twin of a district metered area (8 junctions, 8 pipes, 50 L/s at design demand) solved with the Hazen-Williams equation
- simulated ESP32 edge nodes that sample pressure in deep sleep, wake on sharp changes and run a local fail-safe that closes a solenoid valve when the cloud link is down
- an XGBoost classifier that labels each 10 second window as normal, demand change, leak or burst
- an MLP that ranks which pipe segment is leaking
- a Flask dashboard with a live network schematic, charts, incident log, valve control and an evaluation page
- a REST endpoint so a real ESP32 (or the included emulator) can push readings into the same pipeline

All results in the paper come from `scripts/evaluate.py`, which tests the trained models on 300 simulated episodes the models never saw during training.

## 1. Set up

You need Python 3.10 or newer.

```bash
cd pipeguard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate` instead.

## 2. Run the dashboard

A trained model is already in `models/`, so you can run straight away.

```bash
python run.py
```

Open http://localhost:5000. The model starts making predictions after the first 70 seconds, because it needs a 60 second baseline plus one 10 second window.

To make the simulation run 10 times faster while you demo:

```bash
PIPEGUARD_TICK_S=0.1 python run.py
```

On Windows PowerShell use `$env:PIPEGUARD_TICK_S="0.1"; python run.py`.

## 3. Demo walkthrough

1. Wait until the model reading panel shows probabilities.
2. Pick pipe P3 and click **Start leak**. J3 and J4, the junctions after P3, now get 25% less water, and that water pours out of the pipe as unaccounted flow. Within a few seconds the incident card shows *Leak on P3* and the pipe turns red and dashed.
3. Click **Close valve on P3**. J3 and J4 go grey (isolated) and the unaccounted flow drops back to about zero.
4. Click **Repair all pipes**, open the P3 valve in the valve table and mark the incident resolved.
5. Pick junction J6 and click **Raise metered demand**. Pressures drop, but the model reads it as a demand change, not a leak, because the smart meters account for the extra flow.
6. Click **Cut cloud link**, then pick junction J6, choose **100% cut** and click **Start burst**. J6 loses all its water and the edge nodes close valves on their own within about 2 to 3 seconds.
7. Click **Restore cloud link**. The cloud re-checks the edge trip, keeps the correct valve closed and reopens any valve that the edge closed unnecessarily.
8. Open **Incidents** to see every alert and valve action, and **Evaluation** for the offline test results and figures.

### How leaks and bursts work in the simulation

The zone takes 50 L/s at design demand. A leak or burst spills a share of the water passing that point, so every junction after it gets that much less. The reservoir still sends what the junctions ask for, and the difference is the unaccounted flow. A junction that gets less water also loses pressure: flow out of an outlet goes with the square root of pressure, so 75% of the water means about 56% of the pressure. The dashboard shows each junction's water in L/s (received against needed) with its pressure in kPa underneath.

## 4. Retrain and re-evaluate

```bash
python scripts/train.py 800
python scripts/evaluate.py
```

Training builds 800 episodes and takes about 2 to 3 minutes. Evaluation runs 300 unseen test episodes and writes `results/metrics.json` plus the figures.

Evaluation also runs a **response test** on the live engine. Every scenario from the dashboard (a 25% leak on each pipe, and 50% and 100% bursts at each junction, with the cloud link up and down) is run 5 times. For each run it records:

- how long after the fault starts the alert is raised
- how long until the faulty pipe's valve is fully closed (bursts close automatically, by the cloud or by the edge node when the link is down; for leaks this models the operator pressing **Close valve** as soon as the alert appears)
- the pressure in kPa at the junctions that lose water: before the fault, when the alert is raised, at its lowest and after the valve closes, plus how much the other junctions' pressure changes

Cloud timings include the modelled radio delay (0.6 to 1.8 s each way). The results are at the top of the Evaluation page and in `fig_detection_time.png` and `fig_pressure_kpa.png`.

Training uses random seed 7 and testing uses seed 99, so the test episodes never overlap the training ones. Change the episode count to trade time for accuracy.

## 5. Send readings from a device

Any device can post JSON to `/api/telemetry`. A reading overrides the simulated value for that node for 3 seconds.

```bash
curl -X POST http://localhost:5000/api/telemetry \
  -H "Content-Type: application/json" \
  -d '{"node": "J3", "pressure_kpa": 441.2}'
```

A reading may carry `age_s`, how many seconds ago it was taken. The ESP32 uses this when it uploads readings it buffered while offline: those are stored in the database as `device-buffered`, and only a fresh reading overrides the live value.

The emulator sends one reading a second and then simulates a pressure drop:

```bash
python scripts/device_emulator.py --node J3 --drop-after 60 --drop-kpa 25
```

`firmware/edge_node/edge_node.ino` is an Arduino sketch for an ESP32 with a 0 to 1 MPa pressure transducer on GPIO34, a relay on GPIO26 and a comparator wake line on GPIO33. It uses Wi-Fi HTTP for the uplink. LoRaWAN and GSM are represented in the evaluation by their latency and packet loss settings in `pipeguard/config.py`.

## 6. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | Live snapshot: pressures, twin values, flows, valves, model probabilities, open incident, recent series |
| GET | `/api/topology` | Junctions, pipes and reservoir |
| POST | `/api/telemetry` | Push a reading or a list of readings `{node, pressure_kpa, age_s optional}` |
| POST | `/api/scenario` | Start a test scenario: `{kind: leak, pipe, percent}` cuts the water to every junction after the pipe (default 25%), `{kind: burst, node, percent}` cuts the water to that junction and every junction after it (dashboard offers 50 or 100), or `{kind: demand_spike, node, factor}` |
| POST | `/api/repair` | Clear all test scenarios |
| POST | `/api/valve` | `{pipe, action: open or close}` |
| POST | `/api/link` | `{up: true or false}` to cut or restore the cloud link |
| GET | `/api/alerts` | Incident history |
| POST | `/api/alerts/<id>/resolve` | Resolve an incident |
| GET | `/api/readings/<node>?minutes=10` | Stored readings for one node |
| GET | `/api/metrics` | Contents of `results/metrics.json` |

## 7. Project layout

```
pipeguard/
  run.py                      starts the Flask server
  pipeguard/
    config.py                 every tunable setting in one place
    network.py                pipe network and Hazen-Williams solver
    simulator.py              sensor telemetry, leaks, bursts, demand changes
    features.py               10 s window features against the twin
    detector.py               XGBoost, MLP and static threshold detectors
    dataset.py                builds labelled windows from simulated episodes
    engine.py                 live loop, edge nodes, alerts, valves
    evaluation.py             all offline tests used in the paper
    figures.py                result charts
    db.py                     SQLite storage
    routes.py                 pages and API
    templates/, static/       dashboard front end (Chart.js and fonts bundled, works offline)
  scripts/                    train.py, evaluate.py, device_emulator.py
  firmware/edge_node/         ESP32 sketch
  models/                     trained detector
  results/                    metrics.json and figures from the last evaluation
  docs/figures/               architecture diagram and dashboard screenshots
  tests/                      pytest suite
```

Run the tests with `pytest -q`.

## 8. Limits to state honestly

- The results are from simulation, not a physical pipe rig. The twin, sensor noise, drift, packet loss and valve timings are modelled, and the numbers will change on real hardware.
- The network is branched (a tree). Looped networks need a full EPANET or WNTR solver in place of `network.py`.
- A leak is modelled as a fixed share of the water heading past it, and the reservoir keeps sending the same total. In a real network a leak usually draws extra water from the reservoir as well, and the share it takes depends on the hole and the pressure.
- With the cloud link down, every junction that loses pressure trips its own valve, so a burst also closes the valves after it (a burst at J1 closes all eight). Those extra closures cost no water because those junctions are already cut off, and the cloud reopens them when the link returns.
