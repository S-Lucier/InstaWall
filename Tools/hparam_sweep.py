"""
Hyperparameter sweep tool for wall segmentation training.

Runs multiple short training trials and reports a comparison table.
Edit the TRIALS list at the bottom to define what to test.

Usage:
    python Tools/hparam_sweep.py
    python Tools/hparam_sweep.py --epochs 15 --trials-file my_sweep.json

Output:
    outputs/hparam_sweep/<timestamp>/   - one subdir per trial with checkpoints
    outputs/hparam_sweep/<timestamp>/summary.md  - markdown comparison table
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# =============================================================================
# EDIT THIS to define your sweep
# =============================================================================

# Base args shared by all trials. Override per-trial below.
BASE_ARGS = [
    '--model', 'segformer_gc',
    '--segformer-variant', 'b2',
    '--batch-size', '4',
    '--merge-terrain',
    '--watabou-dir', 'data/watabou_to_mask',
    '--watabou-prob', '0.26',
    '--metadata-file', 'data/foundry_to_mask/Map_Images/metadata.json',
    '--save-interval', '9999',   # don't clutter disk with interval checkpoints
    '--early-stopping', '0',     # disable early stopping for fair comparison
]

# Each trial: (name, extra_args_list)
TRIALS = [
    ('baseline',        []),
    ('focal_g2',        ['--focal-loss', '--focal-gamma', '2.0']),
    ('focal_g3',        ['--focal-loss', '--focal-gamma', '3.0']),
    ('dilation_3',      ['--mask-dilation', '3']),
    ('dilation_5',      ['--mask-dilation', '5']),
    ('focal+dilation',  ['--focal-loss', '--focal-gamma', '2.0', '--mask-dilation', '5']),
]

# =============================================================================


def parse_history(output_dir: Path):
    """Load val IoU history from a completed trial."""
    history_path = output_dir / 'history.json'
    if not history_path.exists():
        return []
    with open(history_path) as f:
        h = json.load(f)
    return h.get('val_iou', [])


def run_trial(name: str, extra_args: list, base_args: list, output_dir: Path,
              epochs: int, seed: int) -> dict:
    """Run a single training trial and return result dict."""
    trial_dir = output_dir / name
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, '-u', '-m', 'primary_model_training.train',
        *base_args,
        *extra_args,
        '--epochs', str(epochs),
        '--output-dir', str(trial_dir),
        '--seed', str(seed),
    ]

    print(f'\n{"=" * 60}')
    print(f'Trial: {name}')
    print(f'Extra args: {extra_args if extra_args else "(none)"}')
    print(f'{"=" * 60}')

    start = time.time()
    log_path = trial_dir / 'train.log'

    with open(log_path, 'w') as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()

    elapsed = time.time() - start

    # Find the actual output subdir (trainer creates a timestamp subdir)
    subdirs = [d for d in trial_dir.iterdir() if d.is_dir()]
    run_dir = subdirs[0] if subdirs else trial_dir

    val_ious = parse_history(run_dir)
    best_iou = max(val_ious) if val_ious else 0.0
    best_epoch = val_ious.index(best_iou) + 1 if val_ious else 0
    final_iou = val_ious[-1] if val_ious else 0.0

    return {
        'name': name,
        'extra_args': ' '.join(extra_args) if extra_args else '(baseline)',
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'final_iou': final_iou,
        'elapsed_s': elapsed,
        'run_dir': str(run_dir),
        'exit_code': proc.returncode,
    }


def write_summary(results: list, output_dir: Path, epochs: int):
    """Write a markdown summary table."""
    lines = [
        f'# Hyperparameter Sweep Summary',
        f'',
        f'Epochs per trial: **{epochs}**',
        f'',
        f'| Trial | Extra Args | Best Val IoU | Best Epoch | Final IoU | Time |',
        f'|-------|------------|:------------:|:----------:|:---------:|------|',
    ]
    for r in sorted(results, key=lambda x: -x['best_iou']):
        mins = r['elapsed_s'] / 60
        status = '' if r['exit_code'] == 0 else f' (exit {r["exit_code"]})'
        lines.append(
            f'| {r["name"]} | `{r["extra_args"]}` | **{r["best_iou"]:.4f}** | '
            f'{r["best_epoch"]} | {r["final_iou"]:.4f} | {mins:.1f}m{status} |'
        )

    summary = '\n'.join(lines) + '\n'
    summary_path = output_dir / 'summary.md'
    summary_path.write_text(summary, encoding='utf-8')

    print('\n' + '=' * 60)
    print('SWEEP COMPLETE')
    print('=' * 60)
    print(summary)
    print(f'Summary saved to: {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for wall segmentation')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs per trial (default: 20)')
    parser.add_argument('--output-dir', default='outputs/hparam_sweep',
                        help='Output base directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for all trials')
    parser.add_argument('--trials-file', default=None,
                        help='JSON file with custom trials list (overrides TRIALS in script)')
    parser.add_argument('--hibernate', action='store_true',
                        help='Hibernate the computer after all trials complete')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = TRIALS
    if args.trials_file:
        with open(args.trials_file) as f:
            raw = json.load(f)
        trials = [(t['name'], t['args']) for t in raw]

    print(f'Running {len(trials)} trials x {args.epochs} epochs each')
    print(f'Output: {output_dir}')

    results = []
    for name, extra_args in trials:
        result = run_trial(
            name=name,
            extra_args=extra_args,
            base_args=BASE_ARGS,
            output_dir=output_dir,
            epochs=args.epochs,
            seed=args.seed,
        )
        results.append(result)
        # Print running best after each trial
        best = max(results, key=lambda r: r['best_iou'])
        print(f'\nRunning best: {best["name"]} @ {best["best_iou"]:.4f}')

    write_summary(results, output_dir, args.epochs)

    if args.hibernate:
        print('\nHibernating in 30 seconds... (close this window to cancel)')
        time.sleep(30)
        subprocess.run(['shutdown', '/h'], check=True)


if __name__ == '__main__':
    main()
