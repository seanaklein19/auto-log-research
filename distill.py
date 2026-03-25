"""Run QKV distill on the most recent training run.

Usage: uv run distill.py [path/to/run.parquet]
"""

import glob
import sys

from qkv.distill import distill_run


def main():
    if len(sys.argv) > 1:
        pq = sys.argv[1]
    else:
        files = sorted(glob.glob("qkv_logs/run_*.parquet"))
        if not files:
            print("No parquet files found in qkv_logs/", file=sys.stderr)
            sys.exit(1)
        pq = files[-1]

    print(f"Distilling: {pq}\n")
    record = distill_run(pq)
    print(record.to_briefing())


if __name__ == "__main__":
    main()
