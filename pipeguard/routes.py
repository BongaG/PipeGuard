import json
import os
import time
from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory, abort

bp = Blueprint("main", __name__)


def _engine():
    return current_app.engine


def _int_arg(name, default, lo, hi):
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, lo), hi)


def _metrics():
    path = os.path.join(current_app.config["RESULTS_DIR"], "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@bp.route("/")
def dashboard():
    eng = _engine()
    cfg = current_app.config
    return render_template("dashboard.html", topology=eng.net.topology(), model_loaded=eng.detector is not None, leak_percent=cfg["LEAK_PERCENT"], burst_percents=cfg["BURST_PERCENTS"])


@bp.route("/alerts")
def alerts():
    rows = current_app.db.query("SELECT * FROM alerts ORDER BY id DESC LIMIT 200")
    valves = current_app.db.query("SELECT * FROM valve_events ORDER BY id DESC LIMIT 50")
    return render_template("alerts.html", alerts=rows, valves=valves)


@bp.route("/evaluation")
def evaluation():
    figs = []
    rdir = current_app.config["RESULTS_DIR"]
    if os.path.isdir(rdir):
        figs = sorted(f for f in os.listdir(rdir) if f.endswith(".png"))
    return render_template("evaluation.html", m=_metrics(), figures=figs)


@bp.route("/results/<path:name>")
def result_file(name):
    return send_from_directory(current_app.config["RESULTS_DIR"], name)


@bp.route("/api/topology")
def api_topology():
    return jsonify(_engine().net.topology())


@bp.route("/api/state")
def api_state():
    return jsonify(_engine().snapshot(_int_arg("seconds", 180, 10, 600)))


@bp.route("/api/scenario", methods=["POST"])
def api_scenario():
    body = request.get_json(force=True, silent=True) or {}
    try:
        _engine().inject(body.get("kind"), pipe=body.get("pipe"), node=body.get("node"), percent=body.get("percent"), factor=body.get("factor"))
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True})


@bp.route("/api/repair", methods=["POST"])
def api_repair():
    _engine().repair()
    return jsonify({"ok": True})


@bp.route("/api/valve", methods=["POST"])
def api_valve():
    body = request.get_json(force=True, silent=True) or {}
    action = body.get("action")
    if action not in ("open", "close"):
        return jsonify({"ok": False, "error": "action must be open or close"}), 400
    try:
        ok = _engine().set_valve(body.get("pipe"), action)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": ok})


@bp.route("/api/link", methods=["POST"])
def api_link():
    body = request.get_json(force=True, silent=True) or {}
    _engine().set_link(bool(body.get("up", True)))
    return jsonify({"ok": True})


@bp.route("/api/alerts/<int:aid>/resolve", methods=["POST"])
def api_resolve(aid):
    eng = _engine()
    if not current_app.db.query("SELECT id FROM alerts WHERE id=?", (aid,)):
        return jsonify({"ok": False, "error": "Unknown incident"}), 404
    if eng.alert and eng.alert["id"] == aid:
        eng.resolve_alert()
    else:
        current_app.db.execute("UPDATE alerts SET status='resolved', closed_at=COALESCE(closed_at, ?) WHERE id=?", (time.time(), aid))
    return jsonify({"ok": True})


@bp.route("/api/alerts")
def api_alerts():
    return jsonify(current_app.db.query("SELECT * FROM alerts ORDER BY id DESC LIMIT 100"))


@bp.route("/api/telemetry", methods=["POST"])
def api_telemetry():
    body = request.get_json(force=True, silent=True) or {}
    items = body if isinstance(body, list) else [body]
    accepted = 0
    for item in items:
        try:
            _engine().ingest(item["node"], item["pressure_kpa"], item.get("age_s", 0.0))
            accepted += 1
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    if not accepted:
        return jsonify({"ok": False, "error": "send node and pressure_kpa"}), 400
    return jsonify({"ok": True, "accepted": accepted})


@bp.route("/api/readings/<node>")
def api_readings(node):
    minutes = _int_arg("minutes", 10, 1, 60)
    rows = current_app.db.query("SELECT ts, pressure_kpa, twin_kpa, source FROM readings WHERE node=? AND ts>=? ORDER BY ts", (node, time.time() - minutes * 60))
    return jsonify(rows)


@bp.route("/api/metrics")
def api_metrics():
    m = _metrics()
    if m is None:
        abort(404)
    return jsonify(m)
