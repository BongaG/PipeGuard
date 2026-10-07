import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeguard.dataset import build_dataset, class_counts
from pipeguard.detector import LeakDetector
from pipeguard.network import PipelineNetwork


def main(n_episodes=800, seed=7):
    t0 = time.time()
    data, _ = build_dataset(n_episodes, seed=seed, stride=2)
    print("windows", len(data["y_event"]), class_counts(data["y_event"]), f"{time.time() - t0:.1f}s")
    net = PipelineNetwork()
    det = LeakDetector(net.pipes.keys())
    det.fit(data["X"], data["y_event"], data["y_pipe"], data["X_loc"])
    det.save()
    print("trained and saved", f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    main(n)
