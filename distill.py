"""Run QKV distill on the most recent training run.

Usage: uv run distill.py [--llm] [path/to/run.parquet]
"""

import glob
import sys

from qkv.distill import distill_run


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_llm = "--llm" in sys.argv

    if args:
        pq = args[0]
    else:
        files = sorted(glob.glob("qkv_logs/run_*.parquet"))
        if not files:
            print("No parquet files found in qkv_logs/", file=sys.stderr)
            sys.exit(1)
        pq = files[-1]

    print(f"Distilling: {pq}" + (" (with LLM analysis)" if use_llm else ""))
    print()
    record = distill_run(pq, use_llm=use_llm)
    print(record.to_briefing())

    if record.findings:
        print("\n--- LLM Findings ---")
        for f in record.findings:
            print(f"  [{f.get('severity', '?')}] {f.get('finding', '?')}")
            print(f"    Evidence: {f.get('evidence', '')}")
            if f.get('actionable'):
                print(f"    ** Actionable **")


if __name__ == "__main__":
    main()
