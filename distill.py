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

    if record.events:
        print(f"\nEvents: {len(record.events)}")
        for ev in record.events[:10]:
            print(f"  Step {ev.get('step', '?')}: {ev.get('type', '?')} "
                  f"(metric={ev.get('metric', '?')}, z={ev.get('z_score', 0):.1f})")

    if record.activation_health:
        print("\nActivation health issues:")
        for ah in record.activation_health:
            print(f"  {ah['layer']}: {', '.join(ah.get('issues', []))}")

    if record.layer_trends:
        growing = [lt for lt in record.layer_trends if lt.get("trend") == "growing"]
        shrinking = [lt for lt in record.layer_trends if lt.get("trend") == "shrinking"]
        if growing or shrinking:
            print("\nLayer norm trends:")
            for lt in growing[:5]:
                print(f"  {lt['name']} ({lt.get('kind', '?')}): growing +{lt.get('change_pct', 0):.0f}%")
            for lt in shrinking[:5]:
                print(f"  {lt['name']} ({lt.get('kind', '?')}): shrinking {lt.get('change_pct', 0):.0f}%")


if __name__ == "__main__":
    main()
