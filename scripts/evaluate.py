import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeguard.evaluation import run_all
from pipeguard.figures import make_figures


def main():
    results, data, episodes, records, fs_records = run_all()
    make_figures(results, episodes, records, fs_records)
    print(json.dumps({k: v for k, v in results.items() if k not in ("localisation",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
