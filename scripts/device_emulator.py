import argparse
import json
import random
import time
import urllib.request


def send(url, node, value):
    body = json.dumps({"node": node, "pressure_kpa": round(value, 2)}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser(description="Send pressure readings to PipeGuard the way an edge node would")
    ap.add_argument("--url", default="http://localhost:5000/api/telemetry")
    ap.add_argument("--node", default="J3")
    ap.add_argument("--base", type=float, default=441.0)
    ap.add_argument("--drop-after", type=int, default=60)
    ap.add_argument("--drop-kpa", type=float, default=25.0)
    ap.add_argument("--seconds", type=int, default=180)
    args = ap.parse_args()
    for t in range(args.seconds):
        value = args.base + random.gauss(0, 0.5)
        if t >= args.drop_after:
            value -= min((t - args.drop_after) / 20.0, 1.0) * args.drop_kpa
        status = send(args.url, args.node, value)
        print(f"t={t:4d}s {args.node} {value:7.2f} kPa -> {status}")
        time.sleep(1)


if __name__ == "__main__":
    main()
