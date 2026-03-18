# auto-log-research

**Autonomous AI research with deep experiment tracking.** An AI agent doesn't just tweak hyperparameters -- it investigates training dynamics, forms hypotheses, and runs targeted experiments. Every run logs per-step metrics (loss curves, LR schedules, MFU, step times) to simple JSONL files. The agent digs into this data between runs, generates plots, spots anomalies, and uses what it learns to decide what to try next.

Built on top of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), which pioneered the idea of giving an AI agent a training setup and letting it experiment autonomously overnight. This fork adds the missing piece: **the agent can actually see what happened during training**, not just the final number.

## What's different from autoresearch?

| | autoresearch | auto-log-research |
|---|---|---|
| **What the agent sees** | Final val_bpb number | Full training curves, LR schedules, MFU, step times, cross-run comparisons |
| **How it decides** | Was the number lower? | Why was it lower? What do the dynamics tell me? |
| **Experiment tracking** | TSV with 5 columns | Structured archives with per-step metrics + agent research notes |
| **Analysis** | None -- agent flies blind | Automated baseline stats + agent-driven investigation with Python |
| **Knowledge accumulation** | Reset every run | `insights.md` (validated findings + current best tracking) + `ideas_queue.md` (prioritized hypotheses) |
| **Best tracking** | Compares to last run | Always compares against all-time best, always builds on best commit |
| **Research quality** | Hyperparameter search | Hypothesis-driven experimentation |

The core insight: **an agent that understands *why* a change helped will make better next moves than one that only knows *whether* it helped.**

## How it works

The agent runs in a loop. Each iteration:

1. **Re-read instructions** -- re-read `program.md` to stay on track (context gets long)
2. **Train** (5 min) -- run the model on a single GPU, logging per-step metrics to `metrics.jsonl`
3. **Analyze** -- automated stats: loss trajectory, convergence detection, cross-run comparison
4. **Investigate** -- the agent writes Python to dig deeper: plots loss curves, compares trajectories across runs, checks where differences emerge
5. **Synthesize** -- writes findings to the run's `analysis.md`: what was surprising, what patterns emerge
6. **Learn** -- updates `insights.md` (what works) and `ideas_queue.md` (what to try next)
7. **Decide** -- compare against all-time best: keep only if it beats the historical best, otherwise revert to best commit
8. **Implement** -- pick next idea from queue, modify `train.py`, commit, repeat

Every step is tracked. Every decision has a paper trail. The agent builds up a knowledge base that gets sharper with each experiment.

### What gets logged

Per training step (to `metrics.jsonl`):
- Training loss (raw + smoothed)
- Learning rate schedule, momentum, weight decay
- Step time, tokens/sec, MFU

Per run (to `.auto-log-research/<commit>/`):
- `run_summary.json` -- all hyperparameters + final metrics
- `analysis.md` -- automated stats + agent's investigation notes
- `run.log` -- full training output
- `metrics.jsonl` -- per-step metrics
- `*.png` -- auto-generated and custom plots

Across runs:
- `results.tsv` -- master experiment log
- `insights.md` -- validated hypotheses + current best commit/val_bpb (agent curates this)
- `ideas_queue.md` -- prioritized experiment queue

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py

# 5. Run post-training analysis
uv run analyze.py
```

## Running the autonomous agent

Spin up Claude Code (or your preferred agent) in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The agent reads `program.md`, sets up a branch, runs the baseline, and enters the autonomous experiment loop. It will keep going indefinitely -- leave it overnight and come back to a log of experiments, each with full analysis.

### Slurm environments

If you're running on a Slurm cluster, the agent uses:
```bash
srun --account=gdpgroup --partition=gdpgroup-a6000 --qos=gdpgroup-main --gpus=1 --mem=64G uv run train.py > run.log 2>&1
```

Modify the account/partition/QoS in `program.md` to match your cluster.

## Project structure

```
prepare.py          -- constants, data prep + runtime utilities (do not modify)
train.py            -- model, optimizer, training loop (agent modifies this)
analyze.py          -- post-run analysis + bookkeeping (do not modify)
program.md          -- agent instructions (human modifies this)
pyproject.toml      -- dependencies

.auto-log-research/ -- experiment tracking (gitignored)
  results.tsv           master experiment log
  insights.md           agent's validated knowledge base + current best tracking
  ideas_queue.md        prioritized experiment queue
  <commit>/             per-run archive
    analysis.md             baseline stats + agent investigation
    run_summary.json        config + metrics
    run.log                 training output
    metrics.jsonl           per-step metrics
```

## Design choices

- **Hypothesis-driven, not grid search.** The agent has access to full training dynamics. It should use them to form and test hypotheses, not just enumerate hyperparameter combinations.
- **Knowledge accumulates.** `insights.md` and `ideas_queue.md` persist across runs. The agent builds a growing understanding of what works for this specific model/data/compute setup.
- **Always build on the best.** The agent tracks the all-time best commit and val_bpb. Failed experiments revert to the best, not just the previous run. This prevents drift away from good configurations.
- **Everything is logged.** Per-step metrics go to `metrics.jsonl`. `analyze.py` computes baseline statistics. The agent adds deeper analysis. Nothing is lost.
- **Single file to modify.** The agent only touches `train.py`. Everything is fair game: architecture, optimizer, hyperparameters, training loop, batch size, model size.
- **Fixed time budget.** Training always runs for exactly 5 minutes. This makes experiments directly comparable regardless of what the agent changes.

## Based on

This project builds on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The training code, model architecture (GPT with Muon optimizer), and core experiment loop come from there. auto-log-research adds the experiment tracking, analysis pipeline, and agentic investigation layer.

## License

MIT
