import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import Config
from .detector import LeakDetector
from .features import window_features
from .simulator import TelemetrySimulator, Scenario, EVENT_CLASSES

INK = "#14283a"
WATER = "#1f6f8b"
DANGER = "#b83227"
WARN = "#c98a0b"
MUTED = "#5b6b78"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
})


def fig_pressure_response(out_dir, cfg=Config):
    det = LeakDetector.load()
    sim = TelemetrySimulator(seed=2024)
    sc = Scenario("leak", onset=150, pipe="P4", fraction=0.5, share=0.25, ramp_s=20)
    ep = sim.episode(300, sc)
    args = (ep["pressure"], ep["twin_pressure"], ep["inlet_flow"], ep["metered_flow"])
    alarm, run = None, 0
    for t in range(cfg.WINDOW_S + cfg.BASELINE_S, 300):
        k = EVENT_CLASSES[int(det.predict_event(window_features(*args, t, cfg), "xgb")[0])]
        run = run + 1 if k in ("leak", "burst") else 0
        if run >= cfg.ALARM_CONSECUTIVE:
            alarm = t + 1
            break
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    idx = {n: i for i, n in enumerate(ep["nodes"])}
    t = np.arange(300)
    for node, col in (("J3", WATER), ("J4", DANGER), ("J6", MUTED)):
        axes[0].plot(t, ep["pressure"][:, idx[node]], color=col, lw=1.2, label=f"{node} measured")
        axes[0].plot(t, ep["twin_pressure"][:, idx[node]], color=col, lw=1, ls="--", alpha=.8)
    axes[0].axvline(sc.onset, color=INK, lw=.8, ls=":")
    lo, hi = axes[0].get_ylim()
    axes[0].text(sc.onset - 3, lo + (hi - lo) * 0.45, "leak starts", fontsize=8, color=INK, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    if alarm:
        axes[0].axvline(alarm, color=DANGER, lw=.9)
        axes[0].text(alarm + 4, lo + (hi - lo) * 0.45, f"alarm at +{alarm - sc.onset:.0f} s", fontsize=8, color=DANGER, bbox=dict(fc="white", ec="none", pad=1))
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Pressure (kPa)")
    axes[0].set_title("(a) Measured (solid) against digital twin (dashed)", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    axes[1].plot(t, ep["inlet_flow"], color=WATER, lw=1.2, label="Inlet flow")
    axes[1].plot(t, ep["metered_flow"], color=WARN, lw=1.2, ls="--", label="Delivered (metered)")
    axes[1].fill_between(t, ep["metered_flow"], ep["inlet_flow"], where=t >= sc.onset, color=DANGER, alpha=.15, label="Unaccounted flow")
    axes[1].axvline(sc.onset, color=INK, lw=.8, ls=":")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Flow (L/s)")
    axes[1].set_title("(b) Water sent against water delivered", fontsize=9)
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_pressure_response.png"))
    plt.close(fig)


def fig_model_comparison(results, out_dir):
    names = [("XGBoost", results["window_level"]["xgb"]["binary"]), ("MLP", results["window_level"]["mlp"]["binary"]), ("Static threshold", results["static_threshold"])]
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("false_positive_rate", "False positive rate")]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = .26
    x = np.arange(len(metrics))
    for k, (n, m) in enumerate(names):
        vals = [m[key] * 100 for key, _ in metrics]
        ax.bar(x + (k - 1) * w, vals, w, label=n, color=[WATER, INK, WARN][k])
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylabel("Percent")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_model_comparison.png"))
    plt.close(fig)


def _cloud_rows(results):
    return [r for r in results["response"]["summary"] if "link down" not in r["group"]]


def _bar_labels(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()), ha="center", va="bottom", fontsize=9, color=INK)


def fig_detection_time(results, out_dir):
    rows = _cloud_rows(results)
    x = np.arange(len(rows))
    w = .36
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _bar_labels(ax, ax.bar(x - w / 2, [r["detection_median_s"] for r in rows], w, color=DANGER, label="Detected"), "{:.1f} s")
    _bar_labels(ax, ax.bar(x + w / 2, [r["valve_closed_median_s"] for r in rows], w, color=WATER, label="Valve closed"), "{:.1f} s")
    ax.set_xticks(x, [r["group"] for r in rows])
    ax.set_ylabel("Seconds after it starts")
    ax.set_title("How fast leaks and bursts are detected and stopped", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")
    ax.margins(y=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_detection_time.png"))
    plt.close(fig)


def fig_pressure_kpa(results, out_dir):
    rows = _cloud_rows(results)
    x = np.arange(len(rows))
    w = .26
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, (key, label, col) in enumerate([("before_kpa", "Normal", WATER), ("lowest_kpa", "During leak or burst", DANGER), ("after_close_kpa", "After valve closed", MUTED)]):
        _bar_labels(ax, ax.bar(x + (k - 1) * w, [max(r[key], 0) for r in rows], w, color=col, label=label), "{:.0f}")
    ax.set_xticks(x, [r["group"] for r in rows])
    ax.set_ylabel("Pressure at affected junctions (kPa)")
    ax.set_title("Pressure the affected junctions receive", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")
    ax.margins(y=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_pressure_kpa.png"))
    plt.close(fig)


def make_figures(results, episodes, records, fs_records, out_dir=Config.RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig_pressure_response(out_dir)
    fig_model_comparison(results, out_dir)
    if "response" in results:
        fig_detection_time(results, out_dir)
        fig_pressure_kpa(results, out_dir)
