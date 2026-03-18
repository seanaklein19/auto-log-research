# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `prepare.py` -- fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` -- the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize tracking**: Create `.auto-log-research/` directory with empty `insights.md` and `ideas_queue.md`. In `insights.md`, add a "Current Best" section at the top:
   ```
   ## Current Best
   commit: (none yet -- baseline pending)
   val_bpb: (none)
   ```
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**Run command** (always use this exact command):
```bash
uv run train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` -- this is the only training file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Write and run Python analysis scripts to explore training dynamics.
- Generate plots with matplotlib and save them to `.auto-log-research/<commit>/`.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Modify `analyze.py`. It handles bookkeeping automatically.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time -- it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically. Remember you are running on A6000 GPUs (48GB VRAM) via Slurm with `--mem=64G` system memory.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

It also writes `run_summary.json` with config + final metrics, and `metrics.jsonl` with per-step metrics (loss, LR, MFU, step time, etc.).

You can extract the key metric from the log file:
```bash
grep "^val_bpb:" run.log
```

## Tracking and Analysis

All experiment data lives in `.auto-log-research/`:

```
.auto-log-research/
  results.tsv              # Master log (append-only TSV)
  insights.md              # Your validated insights + CURRENT BEST tracking
  ideas_queue.md           # Queue of ideas to try next
  <commit>/
    analysis.md            # Baseline stats + YOUR investigation notes (you must write to this!)
    run_summary.json       # Config + final metrics
    run.log                # Full training output
    metrics.jsonl          # Per-step metrics (loss, LR, MFU, step time, etc.)
    loss_curve.png         # Auto-generated: loss vs step (overlaid with prev run)
    lr_and_schedule.png    # Auto-generated: LR, weight decay, momentum schedules
    gpu_perf.png           # Auto-generated: MFU and step time vs step
    *.png                  # Any additional plots you generate
```

**results.tsv** has 5 tab-separated columns:
```
commit	val_bpb	memory_gb	status	description
```

`analyze.py` appends rows with status=`pending`. You update the status to `keep` or `discard` after deciding.

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

**LOOP FOREVER:**

### 0. REFRESH: Re-read program.md

At the start of every iteration, re-read this file (`program.md`) to remind yourself of the full process. Context gets long and you will forget steps otherwise. This takes 2 seconds and prevents wasted experiments.

### 1. ANALYZE: Read the analysis

After each run, run `analyze.py` and read the analysis it produces:

```bash
uv run analyze.py 2>/dev/null
```

Then read `.auto-log-research/<commit>/analysis.md`.

### 2. INVESTIGATE: Dig deeper with Python

`analyze.py` auto-generates comparison plots (loss curves, LR schedules, GPU perf) overlaid with the previous run. **Start by looking at these plots** in `.auto-log-research/<commit>/`.

Then dig deeper. Each run's per-step metrics are saved as `metrics.jsonl` in the archive. Each line is a JSON object with keys: `step`, `train/loss`, `train/loss_smooth`, `train/lr_multiplier`, `train/muon_momentum`, `train/weight_decay`, `train/progress`, `perf/step_time_ms`, `perf/tokens_per_sec`, `perf/mfu_percent`.

```python
# Example: Load a run's step-level metrics
import json
commit = "abc1234"  # short commit hash
history = [json.loads(l) for l in open(f".auto-log-research/{commit}/metrics.jsonl")]
losses = [h["train/loss_smooth"] for h in history]
```

```python
# Example: Find when two runs diverge
import json
cur = [json.loads(l) for l in open(f".auto-log-research/{cur_commit}/metrics.jsonl")]
prev = [json.loads(l) for l in open(f".auto-log-research/{prev_commit}/metrics.jsonl")]

for i in range(min(len(cur), len(prev))):
    delta = cur[i]["train/loss_smooth"] - prev[i]["train/loss_smooth"]
    if abs(delta) > 0.01:
        print(f"Divergence at step {i}: delta={delta:.4f}")
        break
```

Things to look for:
- **Loss trajectory shape**: Is it still decreasing at the end? Plateaued? Diverging?
- **Learning rate sensitivity**: How does the loss change during warmup vs cooldown?
- **GPU utilization**: Is MFU stable? Any step-time spikes?
- **Cross-run patterns**: Do certain types of changes have consistent effects?
- **Where differences emerge**: At what training step does this run diverge from the previous one?

### 3. SYNTHESIZE: Write your findings to analysis.md

**MANDATORY**: Open `.auto-log-research/<commit>/analysis.md` and APPEND your investigation notes at the bottom (under the "Agent Investigation Notes" header). This is NOT optional. Every run must have your written analysis. Write:
- What you observed in the loss curves and plots
- How this run compares to the previous one
- What was surprising or expected
- What this tells you about what to try next

If you skip this step, your research has no memory and you'll repeat mistakes.

### 4. UPDATE your research notes

**insights.md** -- Your validated knowledge base. Things you've confirmed work or don't work:
- "Increasing matrix_lr beyond 0.06 causes gradient instability after step 400"
- "Depth 10 with aspect_ratio 64 hits OOM on A6000"
- "Weight decay 0.2 -> 0.3 gave +0.002 bpb improvement"

If you disproved a hypothesis, delete it. Keep this file clean and accurate.

**ideas_queue.md** -- Your prioritized list of what to try next:
- Delete ideas you just tried (if you've explored the full range)
- Remove ideas that no longer make sense given new learnings
- Add new ideas inspired by your investigation
- Keep them ordered by expected impact

### 5. DECIDE: Keep or reject? (compare against ALL-TIME BEST)

**CRITICAL**: Do NOT compare against the previous run. Compare against the **all-time best val_bpb** recorded in `insights.md` under "Current Best".

- If this run's val_bpb **is lower than the current best**: this is a NEW BEST.
  - Update `results.tsv`: change status from `pending` to `keep`
  - Update `insights.md`: set the "Current Best" section to this commit and val_bpb
  - Leave the commit as is.
- If this run's val_bpb **is equal to or higher than the current best**: DISCARD.
  - Update `results.tsv`: change status from `pending` to `discard`
  - **Checkout back to the current best**: `git checkout <best_commit> -- train.py` to restore the best version of train.py. Then commit the revert.

This ensures you always build on top of the historically best configuration, not just the last thing you tried.

### 6. IMPLEMENT: Make your next change

Read `insights.md` and `ideas_queue.md`. Pick the top idea. Modify `train.py`. Think about what the investigation told you -- don't just try random things.

Git commit your change with a clear description.

### 7. RUN: Launch the experiment

```bash
uv run train.py > run.log 2>&1
```

### 8. POST-RUN: Run analysis

```bash
uv run analyze.py 2>/dev/null
```

### 9. REPEAT

Go back to step 0.

## Important Rules

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: fix trivial issues and re-run, or skip fundamentally broken ideas. Log crashes in results.tsv.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?". The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder -- re-read your insights, look at the loss curves more carefully, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

**Be a researcher, not a search script**: Don't just grid-search hyperparameters. Investigate why things work. Look at the auto-generated plots. Load `metrics.jsonl` files and compare loss trajectories across runs. Form hypotheses and test them. The per-step data is there for a reason -- use it.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
