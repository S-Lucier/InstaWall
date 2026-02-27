#!/usr/bin/env python3
"""
Hyperparameter sweep orchestrator for InstaWall wall segmentation.

Usage:
    python run_sweep.py                     # run all pending experiments (resumes)
    python run_sweep.py --tier 0 1          # run only tiers 0 and 1
    python run_sweep.py --only tex_b0       # run experiments whose name contains "tex_b0"
    python run_sweep.py --dry-run           # list all experiments and exit
    python run_sweep.py --summary           # print results table sorted by best IoU
    python run_sweep.py --output-base PATH  # override OUTPUT_BASE
"""

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# USER CONFIGURATION — edit before transfer to target machine
# ============================================================

IMAGE_DIR   = "data/foundry_to_mask/Map_Images"
MASK_DIR    = "data/foundry_to_mask/line_masks"
META_FILE   = "data/foundry_to_mask/grid_metadata.json"
WATABOU_DIR = None   # set to path string if available on target machine
OUTPUT_BASE = "outputs/sweep"

# ============================================================

# Shared base flags for all runs (4070 Super 12 GB optimised, 3-class)
_BASE = (
    "--merge-terrain --tile-context-cells 1 --num-workers 4 --ema-decay 0.999 "
    "--save-interval 100 --tiles-per-image 8 --epochs 500 --early-stopping 75"
)

# Same base without --merge-terrain for 4-class tier
_BASE_4CLS = (
    "--tile-context-cells 1 --num-workers 4 --ema-decay 0.999 "
    "--save-interval 100 --tiles-per-image 8 --epochs 500 --early-stopping 75"
)

# Batch sizes by SegFormer variant (tuned for 4070 Super 12 GB VRAM budget)
_BATCH: Dict[str, str] = {
    "b0": "--batch-size 8",
    "b1": "--batch-size 6",
    "b2": "--batch-size 4",
    "b3": "--batch-size 3",
}

# Model flag strings
_SEG: Dict[str, str] = {
    v: f"--model segformer --segformer-variant {v}" for v in ("b0", "b1", "b2", "b3")
}
_GC: Dict[str, str] = {
    v: f"--model segformer_gc --segformer-variant {v}" for v in ("b0", "b1")
}
_TEX: Dict[str, str] = {
    v: f"--model segformer_texture --segformer-variant {v}" for v in ("b0", "b1", "b2", "b3")
}

# Default texture crop settings
_CROP = "--texture-crop-size 64 --texture-crops-per-class 2"


def _j(*parts: str) -> str:
    """Join non-empty strings with spaces."""
    return " ".join(p for p in parts if p)


EXPERIMENTS: List[Dict] = [
    # ── Tier 0: Sanity (~5 min) ───────────────────────────────────────────────
    dict(
        name="sanity_b0", tier=0, slot="b0",
        cli_args=_j(_BASE, _SEG["b0"], _BATCH["b0"], "--epochs 30 --early-stopping 0"),
        description="Sanity check, 30 epochs only",
    ),

    # ── Tier 1: Baselines (~45 min) ───────────────────────────────────────────
    dict(
        name="base_b0", tier=1, slot="b0",
        cli_args=_j(_BASE, _SEG["b0"], _BATCH["b0"]),
        description="Baseline segformer b0",
    ),
    dict(
        name="base_b1", tier=1, slot="seq",
        cli_args=_j(_BASE, _SEG["b1"], _BATCH["b1"]),
        description="Baseline segformer b1",
    ),
    dict(
        name="gc_b0", tier=1, slot="b0",
        cli_args=_j(_BASE, _GC["b0"], _BATCH["b0"]),
        description="Global-context segformer b0",
    ),
    dict(
        name="gc_b1", tier=1, slot="seq",
        cli_args=_j(_BASE, _GC["b1"], _BATCH["b1"]),
        description="Global-context segformer b1",
    ),

    # ── Tier 2: Core Texture Ablation (~75 min) ───────────────────────────────
    dict(
        name="tex_b0_c64_n2", tier=2, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP),
        description="Texture b0, crop=64, n=2 (default)",
    ),
    dict(
        name="tex_b0_c32_n2", tier=2, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"],
                    "--texture-crop-size 32 --texture-crops-per-class 2"),
        description="Texture b0, crop=32, n=2",
    ),
    dict(
        name="tex_b0_c128_n2", tier=2, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"],
                    "--texture-crop-size 128 --texture-crops-per-class 2"),
        description="Texture b0, crop=128, n=2",
    ),
    dict(
        name="tex_b0_c64_n1", tier=2, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"],
                    "--texture-crop-size 64 --texture-crops-per-class 1"),
        description="Texture b0, crop=64, n=1",
    ),
    dict(
        name="tex_b0_c64_n3", tier=2, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"],
                    "--texture-crop-size 64 --texture-crops-per-class 3"),
        description="Texture b0, crop=64, n=3",
    ),
    dict(
        name="tex_b0_gc_c64_n2", tier=2, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP, "--global-context"),
        description="Texture b0 + global context, crop=64",
    ),
    dict(
        name="tex_b1_c64_n2", tier=2, slot="seq",
        cli_args=_j(_BASE, _TEX["b1"], _BATCH["b1"], _CROP),
        description="Texture b1, crop=64, n=2",
    ),

    # ── Tier 3: Architecture Depth (~7 hrs) ───────────────────────────────────
    dict(
        name="base_b2", tier=3, slot="seq",
        cli_args=_j(_BASE, _SEG["b2"], _BATCH["b2"]),
        description="Baseline segformer b2",
    ),
    dict(
        name="tex_b2_c64_n2", tier=3, slot="seq",
        cli_args=_j(_BASE, _TEX["b2"], _BATCH["b2"], _CROP),
        description="Texture b2, crop=64, n=2",
    ),
    dict(
        name="tex_b1_c64_n3", tier=3, slot="seq",
        cli_args=_j(_BASE, _TEX["b1"], _BATCH["b1"],
                    "--texture-crop-size 64 --texture-crops-per-class 3"),
        description="Texture b1, crop=64, n=3",
    ),
    dict(
        name="tex_b3_c64_n2", tier=3, slot="seq",
        cli_args=_j(_BASE, _TEX["b3"], _BATCH["b3"], _CROP),
        description="Texture b3, crop=64, n=2",
    ),

    # ── Tier 4: Training Tricks with Texture (~2.5 hrs) ───────────────────────
    dict(
        name="tex_b0_focal", tier=4, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP,
                    "--focal-loss --focal-gamma 2.0"),
        description="Texture b0, focal loss",
    ),
    dict(
        name="tex_b0_focal_sched", tier=4, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP, "--focal-schedule"),
        description="Texture b0, focal schedule",
    ),
    dict(
        name="tex_b0_dil3", tier=4, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP, "--mask-dilation 3"),
        description="Texture b0, dilation=3",
    ),
    dict(
        name="tex_b0_dil5", tier=4, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP, "--mask-dilation 5"),
        description="Texture b0, dilation=5",
    ),
    dict(
        name="tex_b0_no_gray", tier=4, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP, "--no-grayscale-aug"),
        description="Texture b0, no grayscale aug",
    ),
    dict(
        name="tex_b1_focal_sched", tier=4, slot="seq",
        cli_args=_j(_BASE, _TEX["b1"], _BATCH["b1"], _CROP, "--focal-schedule"),
        description="Texture b1, focal schedule",
    ),
    dict(
        name="tex_b1_dil3", tier=4, slot="seq",
        cli_args=_j(_BASE, _TEX["b1"], _BATCH["b1"], _CROP, "--mask-dilation 3"),
        description="Texture b1, dilation=3",
    ),
    dict(
        name="tex_b0_gc_focal", tier=4, slot="b0",
        cli_args=_j(_BASE, _TEX["b0"], _BATCH["b0"], _CROP,
                    "--global-context --focal-schedule"),
        description="Texture b0, gc, focal schedule",
    ),

    # ── Tier 5: 4-Class Variants (~1 hr) ──────────────────────────────────────
    dict(
        name="base_b0_4cls", tier=5, slot="b0",
        cli_args=_j(_BASE_4CLS, _SEG["b0"], _BATCH["b0"]),
        description="Baseline b0, 4 classes",
    ),
    dict(
        name="gc_b0_4cls", tier=5, slot="b0",
        cli_args=_j(_BASE_4CLS, _GC["b0"], _BATCH["b0"]),
        description="GC b0, 4 classes",
    ),
    dict(
        name="tex_b0_4cls", tier=5, slot="b0",
        cli_args=_j(_BASE_4CLS, _TEX["b0"], _BATCH["b0"], _CROP),
        description="Texture b0, 4 classes",
    ),
    dict(
        name="tex_b1_4cls", tier=5, slot="seq",
        cli_args=_j(_BASE_4CLS, _TEX["b1"], _BATCH["b1"], _CROP),
        description="Texture b1, 4 classes",
    ),

    # ── Tier 6: Best Model Candidates (~8 hrs) ────────────────────────────────
    dict(
        name="tex_b2_gc_n3", tier=6, slot="seq",
        cli_args=_j(_BASE, _TEX["b2"], _BATCH["b2"],
                    "--texture-crop-size 64 --texture-crops-per-class 3 --global-context"),
        description="Texture b2, gc, n=3",
    ),
    dict(
        name="tex_b2_dil3_focal", tier=6, slot="seq",
        cli_args=_j(_BASE, _TEX["b2"], _BATCH["b2"],
                    _CROP, "--mask-dilation 3 --focal-schedule"),
        description="Texture b2, dilation=3, focal",
    ),
    dict(
        name="tex_b3_focal_sched", tier=6, slot="seq",
        cli_args=_j(_BASE, _TEX["b3"], _BATCH["b3"], _CROP, "--focal-schedule"),
        description="Texture b3, focal schedule",
    ),
    dict(
        name="tex_b1_gc_c64_n3", tier=6, slot="seq",
        cli_args=_j(_BASE, _TEX["b1"], _BATCH["b1"],
                    "--texture-crop-size 64 --texture-crops-per-class 3 --global-context"),
        description="Texture b1, gc, n=3",
    ),
]

# Total: 32 experiments across 7 tiers (0–6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_path(output_base: str) -> Path:
    return Path(output_base) / "sweep_status.json"


def _load_status(output_base: str) -> Dict:
    p = _status_path(output_base)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {"started_at": datetime.now().isoformat(), "runs": {}}


def _save_status(status: Dict, output_base: str):
    p = _status_path(output_base)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, p)


def _read_history(run_dir: Path) -> Optional[Dict]:
    hp = run_dir / "history.json"
    if hp.exists():
        try:
            with open(hp) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _best_iou_from_history(history: Optional[Dict]) -> Optional[float]:
    if history is None:
        return None
    ious = history.get("val_iou", [])
    return float(max(ious)) if ious else None


def _epoch_from_history(history: Optional[Dict]) -> Optional[int]:
    if history is None:
        return None
    ious = history.get("val_iou", [])
    return len(ious) if ious else None


def _extract_model_info(cli_args: str) -> Tuple[str, str]:
    """Return (model_type, variant) parsed from cli_args."""
    model = "segformer"
    variant = "b0"
    m = re.search(r'--model\s+(\S+)', cli_args)
    if m:
        model = m.group(1)
    v = re.search(r'--segformer-variant\s+(\S+)', cli_args)
    if v:
        variant = v.group(1)
    return model, variant


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


# ── Dry run ───────────────────────────────────────────────────────────────────

def cmd_dry_run(experiments: List[Dict]):
    total = len(experiments)
    b0_count = sum(1 for e in experiments if e["slot"] == "b0")
    seq_count = sum(1 for e in experiments if e["slot"] == "seq")

    print(f"{'#':>4}  {'Tier':>4}  {'Slot':>4}  {'Name':<25}  Description")
    print("-" * 80)
    for i, exp in enumerate(experiments):
        print(f"{i+1:>4}  {exp['tier']:>4}  {exp['slot']:>4}  {exp['name']:<25}  {exp['description']}")

    print()
    print(f"Total: {total} experiments  (b0-parallel: {b0_count}, sequential: {seq_count})")
    by_tier: Dict[int, int] = {}
    for e in experiments:
        by_tier[e["tier"]] = by_tier.get(e["tier"], 0) + 1
    for tier in sorted(by_tier):
        print(f"  Tier {tier}: {by_tier[tier]} runs")


# ── Summary ───────────────────────────────────────────────────────────────────

def cmd_summary(experiments: List[Dict], output_base: str):
    status = _load_status(output_base)

    rows = []
    for exp in experiments:
        name = exp["name"]
        run_st = status["runs"].get(name, {})
        state = run_st.get("status", "pending")

        run_dir = Path(output_base) / name
        history = _read_history(run_dir)
        best_iou = _best_iou_from_history(history) or run_st.get("best_iou")
        epochs   = _epoch_from_history(history) or run_st.get("epochs")
        elapsed  = run_st.get("elapsed_s")

        model, variant = _extract_model_info(exp["cli_args"])
        rows.append({
            "name":        name,
            "model":       model,
            "variant":     variant,
            "tier":        exp["tier"],
            "status":      state,
            "best_iou":    best_iou,
            "epochs":      epochs,
            "elapsed_s":   elapsed,
            "description": exp["description"],
        })

    rows.sort(key=lambda r: r["best_iou"] if r["best_iou"] is not None else -1.0,
              reverse=True)

    baseline_ious = [r["best_iou"] for r in rows
                     if r["name"].startswith("base_") and r["best_iou"] is not None]
    best_baseline = max(baseline_ious) if baseline_ious else None

    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(box=box.SIMPLE_HEAVY, title="Sweep Results")
        table.add_column("Rank",        width=5,  justify="right")
        table.add_column("Name",        width=22)
        table.add_column("Model",       width=18)
        table.add_column("Var",         width=4)
        table.add_column("Status",      width=10)
        table.add_column("Best IoU",    width=9,  justify="right")
        table.add_column("Epochs",      width=7,  justify="right")
        table.add_column("Time",        width=8)
        table.add_column("vs Baseline", width=11, justify="right")

        STATUS_STYLE = {"completed": "green", "running": "yellow",
                        "failed": "red", "pending": "dim"}

        for i, r in enumerate(rows):
            rank      = str(i + 1) if r["best_iou"] is not None else "-"
            iou_str   = f"{r['best_iou']:.4f}" if r["best_iou"] is not None else ""
            epoch_str = str(r["epochs"]) if r["epochs"] is not None else ""
            time_str  = _fmt_elapsed(r["elapsed_s"]) if r["elapsed_s"] is not None else ""
            if best_baseline is not None and r["best_iou"] is not None:
                delta_str = f"{r['best_iou'] - best_baseline:+.4f}"
            else:
                delta_str = ""

            style = STATUS_STYLE.get(r["status"], "")
            table.add_row(rank, r["name"], r["model"], r["variant"], r["status"],
                          iou_str, epoch_str, time_str, delta_str,
                          style=style)

        console.print(table)

    except ImportError:
        print(f"{'Rank':>5}  {'Name':<22}  {'Status':<10}  {'Best IoU':>9}  "
              f"{'Epochs':>7}  {'vs Baseline':>11}")
        print("-" * 72)
        for i, r in enumerate(rows):
            rank      = str(i + 1) if r["best_iou"] is not None else "-"
            iou_str   = f"{r['best_iou']:.4f}" if r["best_iou"] is not None else "-"
            epoch_str = str(r["epochs"] or "-")
            if best_baseline is not None and r["best_iou"] is not None:
                delta_str = f"{r['best_iou'] - best_baseline:+.4f}"
            else:
                delta_str = ""
            print(f"{rank:>5}  {r['name']:<22}  {r['status']:<10}  "
                  f"{iou_str:>9}  {epoch_str:>7}  {delta_str:>11}")

    # Save CSV
    csv_path = Path(output_base) / "results_summary.csv"
    try:
        Path(output_base).mkdir(parents=True, exist_ok=True)
        fieldnames = ["name", "model", "variant", "tier", "status",
                      "best_iou", "epochs", "elapsed_s", "description"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in fieldnames})
        print(f"\nCSV saved to {csv_path}")
    except Exception as e:
        print(f"Could not save CSV: {e}")


# ── Live dashboard ────────────────────────────────────────────────────────────

def _build_dashboard(experiments: List[Dict], running: Dict,
                     status: Dict, output_base: str):
    """Build a Rich renderable for the live dashboard."""
    from rich.table import Table
    from rich.console import Group
    from rich.text import Text
    from rich import box

    STATUS_STYLE = {
        "completed": "green",
        "running":   "bold yellow",
        "failed":    "bold red",
        "pending":   "dim",
    }

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, expand=False)
    table.add_column("#",        width=3,  justify="right")
    table.add_column("Name",     width=22)
    table.add_column("Model",    width=18)
    table.add_column("Var",      width=4)
    table.add_column("T",        width=2,  justify="right")
    table.add_column("Status",   width=10)
    table.add_column("Epoch",    width=6,  justify="right")
    table.add_column("Best IoU", width=9,  justify="right")
    table.add_column("Elapsed",  width=8)

    n_completed = n_running = n_failed = n_pending = 0
    elapsed_completed: List[float] = []

    for i, exp in enumerate(experiments):
        name     = exp["name"]
        run_st   = status["runs"].get(name, {})
        state    = run_st.get("status", "pending")
        model, variant = _extract_model_info(exp["cli_args"])

        run_dir  = Path(output_base) / name
        history  = _read_history(run_dir)
        best_iou = _best_iou_from_history(history) or run_st.get("best_iou")
        epoch    = _epoch_from_history(history) or None

        if state == "completed":
            n_completed += 1
            elapsed_s = run_st.get("elapsed_s", 0.0)
            elapsed_completed.append(elapsed_s)
            elapsed_str = _fmt_elapsed(elapsed_s)
        elif state == "running":
            n_running += 1
            started_at = run_st.get("started_at", "")
            if started_at:
                try:
                    elapsed_s = (datetime.now() -
                                 datetime.fromisoformat(started_at)).total_seconds()
                except Exception:
                    elapsed_s = 0.0
            else:
                elapsed_s = 0.0
            elapsed_str = _fmt_elapsed(elapsed_s)
        elif state == "failed":
            n_failed += 1
            elapsed_s = run_st.get("elapsed_s", 0.0)
            elapsed_str = _fmt_elapsed(elapsed_s)
        else:
            n_pending += 1
            elapsed_str = ""

        style    = STATUS_STYLE.get(state, "")
        iou_str  = f"{best_iou:.4f}" if best_iou is not None else ""
        ep_str   = str(epoch) if epoch is not None else ""
        status_t = Text(state, style=style)

        table.add_row(str(i + 1), name, model, variant, str(exp["tier"]),
                      status_t, ep_str, iou_str, elapsed_str)

    total   = len(experiments)
    n_done  = n_completed + n_failed
    n_rem   = total - n_done
    if elapsed_completed:
        eta_s   = (sum(elapsed_completed) / len(elapsed_completed)) * n_rem
        eta_str = _fmt_elapsed(eta_s)
    else:
        eta_str = "?"

    summary = Text.from_markup(
        f"[green]{n_completed}[/] done  "
        f"[yellow]{n_running}[/] running  "
        f"[dim]{n_pending}[/] pending  "
        f"[red]{n_failed}[/] failed  "
        f"  ~{eta_str} remaining"
    )

    return Group(table, summary)


# ── Sweep runner ──────────────────────────────────────────────────────────────

def cmd_run(experiments: List[Dict], output_base: str):
    """Run the sweep orchestrator."""
    # Sort by (tier, original index) for deterministic order
    sorted_exps = sorted(enumerate(experiments), key=lambda x: (x[1]["tier"], x[0]))
    sorted_exps = [e for _, e in sorted_exps]

    Path(output_base).mkdir(parents=True, exist_ok=True)

    status = _load_status(output_base)
    if "started_at" not in status:
        status["started_at"] = datetime.now().isoformat()
    if "runs" not in status:
        status["runs"] = {}

    # Any previously "running" run was interrupted — reset to pending
    for name, rs in status["runs"].items():
        if rs.get("status") == "running":
            rs["status"] = "pending"
    _save_status(status, output_base)

    # Build pending queue (skip completed and failed)
    skip_states = ("completed", "failed")
    pending_queue: List[Dict] = [
        e for e in sorted_exps
        if status["runs"].get(e["name"], {}).get("status") not in skip_states
    ]

    running: Dict[str, Dict] = {}  # name -> {proc, slot, start_time, log_fh}

    # ── Rich setup ────────────────────────────────────────────────────────────
    try:
        from rich.live import Live
        from rich.console import Console
        _use_rich = True
    except ImportError:
        _use_rich = False
        print("rich not installed — using compact status every 30 s")

    last_fallback_print = 0.0

    # ── Launch helper ─────────────────────────────────────────────────────────
    def launch(exp: Dict):
        name    = exp["name"]
        run_dir = Path(output_base) / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "primary_model_training.train",
            "--image-dir", IMAGE_DIR,
            "--mask-dir",  MASK_DIR,
            "--output-dir", str(run_dir),
        ]
        if META_FILE:
            cmd += ["--metadata-file", META_FILE]
        if WATABOU_DIR:
            cmd += ["--watabou-dir", WATABOU_DIR]
        cmd += shlex.split(exp["cli_args"])

        log_fh = open(run_dir / "train.log", "w", buffering=1, encoding="utf-8")
        proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

        now = datetime.now().isoformat()
        running[name] = {
            "proc":       proc,
            "slot":       exp["slot"],
            "start_time": time.time(),
            "log_fh":     log_fh,
        }
        status["runs"][name] = {
            "status":     "running",
            "pid":        proc.pid,
            "started_at": now,
        }
        _save_status(status, output_base)
        print(f"  [+] Launched {name}  (slot={exp['slot']}, pid={proc.pid})")

    # ── Completion checker ────────────────────────────────────────────────────
    def check_finished():
        to_remove = []
        for name, run in list(running.items()):
            retcode = run["proc"].poll()
            if retcode is None:
                continue

            elapsed_s = time.time() - run["start_time"]
            run["log_fh"].close()

            run_dir  = Path(output_base) / name
            history  = _read_history(run_dir)
            best_iou = _best_iou_from_history(history)
            epochs   = _epoch_from_history(history)

            if retcode == 0:
                state   = "completed"
                iou_str = f"{best_iou:.4f}" if best_iou is not None else "?"
                print(f"  [ok] {name}  IoU={iou_str}  "
                      f"epoch={epochs}  time={_fmt_elapsed(elapsed_s)}")
            else:
                state = "failed"
                print(f"  [!!] {name}  FAILED (exit={retcode}, "
                      f"time={_fmt_elapsed(elapsed_s)})")

            status["runs"][name] = {
                "status":    state,
                "best_iou":  best_iou,
                "epochs":    epochs,
                "elapsed_s": elapsed_s,
                "exit_code": retcode,
            }
            _save_status(status, output_base)
            to_remove.append(name)

        for name in to_remove:
            del running[name]

    # ── Signal handler ────────────────────────────────────────────────────────
    def handle_interrupt(sig, frame):
        print("\nInterrupt received — stopping all running experiments...")
        for name, run in running.items():
            run["proc"].terminate()
            print(f"  Terminated {name}")

        deadline = time.time() + 30
        for name, run in running.items():
            wait_s = max(0.0, deadline - time.time())
            try:
                run["proc"].wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                run["proc"].kill()
            run["log_fh"].close()
            # Reset to pending so the next run picks them up
            status["runs"][name] = {"status": "pending"}

        _save_status(status, output_base)
        print(f"Status saved to {_status_path(output_base)}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        signal.signal(signal.SIGTERM, handle_interrupt)
    except (OSError, AttributeError):
        pass  # SIGTERM not available on all Windows configurations

    # ── Main loop ─────────────────────────────────────────────────────────────
    live_ctx = None
    if _use_rich:
        from rich.live import Live
        from rich.console import Console
        live_ctx = Live(refresh_per_second=0.33, console=Console())
        live_ctx.start()

    try:
        while pending_queue or running:
            check_finished()

            b0_active  = sum(1 for r in running.values() if r["slot"] == "b0")
            seq_active = sum(1 for r in running.values() if r["slot"] == "seq")

            still_pending: List[Dict] = []
            for exp in pending_queue:
                if exp["slot"] == "b0" and b0_active < 2 and seq_active == 0:
                    launch(exp)
                    b0_active += 1
                elif exp["slot"] == "seq" and seq_active == 0 and b0_active == 0:
                    launch(exp)
                    seq_active += 1
                else:
                    still_pending.append(exp)

            pending_queue = still_pending

            if live_ctx is not None:
                live_ctx.update(
                    _build_dashboard(sorted_exps, running, status, output_base)
                )
            elif time.time() - last_fallback_print >= 30:
                n_done = sum(
                    1 for r in status["runs"].values()
                    if r.get("status") == "completed"
                )
                print(f"  Status: {n_done}/{len(sorted_exps)} completed, "
                      f"{len(running)} running, {len(pending_queue)} pending")
                last_fallback_print = time.time()

            time.sleep(3)

    finally:
        if live_ctx is not None:
            live_ctx.stop()

    n_done   = sum(1 for r in status["runs"].values() if r.get("status") == "completed")
    n_failed = sum(1 for r in status["runs"].values() if r.get("status") == "failed")
    print(f"\nSweep finished: {n_done} completed, {n_failed} failed")
    print(f"Results: {_status_path(output_base)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="InstaWall hyperparameter sweep orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tier", type=int, nargs="+",
                        help="Run only the specified tier numbers")
    parser.add_argument("--only", type=str,
                        help="Run only experiments whose name contains this substring")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all experiments and exit without running")
    parser.add_argument("--summary", action="store_true",
                        help="Print results table sorted by best IoU then exit")
    parser.add_argument("--output-base", default=OUTPUT_BASE,
                        help=f"Override output base directory (default: {OUTPUT_BASE})")
    args = parser.parse_args()

    # Apply filters
    experiments = list(EXPERIMENTS)
    if args.tier:
        experiments = [e for e in experiments if e["tier"] in args.tier]
    if args.only:
        experiments = [e for e in experiments if args.only in e["name"]]

    if args.dry_run:
        cmd_dry_run(experiments)
        return

    if args.summary:
        cmd_summary(EXPERIMENTS, args.output_base)
        return

    if not experiments:
        print("No experiments match the given filters.")
        return

    cmd_run(experiments, args.output_base)


if __name__ == "__main__":
    main()
