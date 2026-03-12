# InstaWall Sweep — Transfer & Run Guide

Target machine: **RTX 4070 Super 12 GB VRAM**

---

## 1. Transfer the project

Copy the entire `InstaWall/` folder to the target machine, including:
- `data/` (images + masks)
- `primary_model_training/` (all source files)
- `run_sweep.py`
- `start_sweep.bat`
- `requirements.txt` (if present) or note the deps below

---

## 2. Set up Python environment

```bat
conda create -n instawall python=3.10
conda activate instawall
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers timm rich
```

Verify GPU is visible:
```bat
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 3. Set up Discord notifications (optional but recommended)

1. In your Discord server: **Server Settings > Integrations > Webhooks > New Webhook**
2. Choose a channel, copy the webhook URL
3. Open `run_sweep.py` in Notepad and paste it:
   ```
   DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."
   ```
4. Optionally adjust how often heartbeat summaries are sent (default: every 5 completions):
   ```
   DISCORD_HEARTBEAT_EVERY = 5
   ```

**Notifications you'll receive:**
- Sweep started (experiment count + tiers)
- Each experiment launched (tier, slot, PID)
- Each experiment completed (best IoU, epochs, time, overall progress)
- Each experiment failed (exit code + time) — check `train.log` for details
- Heartbeat summary every N completions (overall best IoU so far)
- Sweep complete (total time, top 3 IoU results)

## 4. Edit start_sweep.bat (one line)

Open `start_sweep.bat` in Notepad and confirm:
```
set CONDA_ENV=instawall
```
Change if you used a different env name.

---

## 4. Running the sweep

Double-click `start_sweep.bat` — it will run all 32 experiments in tier order.

Or open a terminal and pass arguments:

| Command | What it does |
|---|---|
| `start_sweep.bat` | Run all pending experiments (auto-resumes) |
| `start_sweep.bat --tier 0` | Tier 0 only — sanity check (30 epochs, ~5 min) |
| `start_sweep.bat --tier 0 1` | Tiers 0 and 1 |
| `start_sweep.bat --only tex_b0` | All experiments with "tex_b0" in the name |
| `start_sweep.bat --dry-run` | List all 32 experiments, don't run anything |
| `start_sweep.bat --summary` | Print results table sorted by best IoU |

**Always run `--tier 0` first** as a sanity check before leaving it to run overnight.

---

## 5. What the sweep runs

32 experiments across 7 tiers (tier = priority order):

| Tier | Focus | Experiments |
|---|---|---|
| 0 | Sanity check | 1 run, 30 epochs |
| 1 | Baselines | segformer b0/b1, global-context b0/b1 |
| 2 | Texture ablation | crop size (32/64/128), n crops (1/2/3), gc variant, b1 |
| 3 | Architecture depth | b2, b3 variants |
| 4 | Training tricks | focal loss, focal schedule, mask dilation 3/5, no grayscale aug |
| 5 | 4-class variants | same models without --merge-terrain |
| 6 | Best model candidates | b2/b3 combos with best tricks from tiers 2–4 |

Parallelism: up to **2 b0 experiments run simultaneously**, all others sequential.
Sequential runs wait until all b0 slots are empty to avoid VRAM contention.

---

## 6. Batch sizes (tuned for 4070 Super 12 GB)

| Variant | Batch size |
|---|---|
| b0 | 8 |
| b1 | 6 |
| b2 | 4 |
| b3 | 3 |

If you get OOM errors, open `run_sweep.py` and reduce the relevant `_BATCH` value.

---

## 7. Monitoring while running

The **Rich live dashboard** refreshes every 3 seconds in the terminal window showing:
- Each experiment's status (pending / running / completed / failed)
- Current epoch and best IoU so far
- Elapsed time and estimated time remaining

Individual run logs are saved to:
```
outputs/sweep/<experiment_name>/train.log
```

Open a second terminal to tail a specific run:
```bat
type outputs\sweep\tex_b0_c64_n2\train.log
```

Status file (safe to read while running):
```
outputs/sweep/sweep_status.json
```

---

## 8. Resuming after interruption

If the sweep stops for any reason (power, Ctrl+C, crash), just run `start_sweep.bat` again.
Already-completed experiments are skipped automatically. Interrupted runs restart from scratch.

---

## 9. Checking results

```bat
start_sweep.bat --summary
```

Prints a table sorted by best validation IoU with delta vs the baseline.
Also saves `outputs/sweep/results_summary.csv` for spreadsheet analysis.

---

## 10. Stopping gracefully

Press **Ctrl+C** in the terminal window.
Running subprocesses are terminated, status is saved, and the next run will resume.

---

## 11. Key file locations

| File | Purpose |
|---|---|
| `run_sweep.py` | Orchestrator — edit `_BATCH` for VRAM tuning |
| `start_sweep.bat` | Double-click launcher |
| `outputs/sweep/sweep_status.json` | Live status of all experiments |
| `outputs/sweep/<name>/train.log` | Full stdout for each run |
| `outputs/sweep/<name>/history.json` | Per-epoch loss/IoU for each run |
| `outputs/sweep/<name>/checkpoint_best.pt` | Best checkpoint for each run |
| `outputs/sweep/results_summary.csv` | Final results table (written by --summary) |
