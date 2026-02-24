"""
Watches a training process and hibernates when done or if stalled.
Usage: python Tools/training_watcher.py --pid 28768 --log path/to/train.log --timeout 40
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def pid_alive(pid):
    result = subprocess.run(
        ['tasklist', '/FI', f'PID eq {pid}'],
        capture_output=True, text=True
    )
    return str(pid) in result.stdout


def get_output_size(output_dir):
    """Sum of all file sizes in output_dir — grows as history.json and checkpoints are written."""
    total = 0
    for f in Path(output_dir).iterdir():
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output-dir', required=True, help='Training output directory to watch for file changes')
    parser.add_argument('--timeout', type=int, default=40, help='Minutes without progress before hibernating')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--hibernate-delay', type=int, default=30, help='Seconds to wait before hibernating after training finishes')
    args = parser.parse_args()

    print(f'Watching PID {args.pid}')
    print(f'Output dir: {args.output_dir}')
    print(f'Hibernate if stalled for {args.timeout} min or process exits (delay: {args.hibernate_delay}s).')
    sys.stdout.flush()

    last_size = 0
    last_progress = time.time()

    while True:
        if not pid_alive(args.pid):
            print(f'Training process exited. Hibernating in {args.hibernate_delay}s...')
            sys.stdout.flush()
            time.sleep(args.hibernate_delay)
            subprocess.run(['shutdown', '/h'])
            return

        size = get_output_size(args.output_dir)
        if size != last_size:
            last_size = size
            last_progress = time.time()

        stalled = (time.time() - last_progress) / 60
        print(f'  PID {args.pid} alive | dir {last_size} bytes | stalled {stalled:.1f}/{args.timeout} min')
        sys.stdout.flush()

        if stalled >= args.timeout:
            print(f'No progress for {args.timeout} min — hung. Hibernating in {args.hibernate_delay}s...')
            sys.stdout.flush()
            time.sleep(args.hibernate_delay)
            subprocess.run(['shutdown', '/h'])
            return

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
