"""
Watches a training process and hibernates when done or if stalled.
Usage: python Tools/training_watcher.py --pid 28768 --log path/to/train.log --timeout 40
"""
import argparse
import os
import subprocess
import sys
import time


def pid_alive(pid):
    result = subprocess.run(
        ['tasklist', '/FI', f'PID eq {pid}'],
        capture_output=True, text=True
    )
    return str(pid) in result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--timeout', type=int, default=40, help='Minutes without log progress before hibernating')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    args = parser.parse_args()

    print(f'Watching PID {args.pid}')
    print(f'Log: {args.log}')
    print(f'Hibernate if stalled for {args.timeout} min or process exits.')
    sys.stdout.flush()

    last_size = 0
    last_progress = time.time()

    while True:
        if not pid_alive(args.pid):
            print('Training process exited. Hibernating in 30s...')
            sys.stdout.flush()
            time.sleep(30)
            subprocess.run(['shutdown', '/h'])
            return

        if os.path.exists(args.log):
            size = os.path.getsize(args.log)
            if size != last_size:
                last_size = size
                last_progress = time.time()

        stalled = (time.time() - last_progress) / 60
        print(f'  PID {args.pid} alive | log {last_size} bytes | stalled {stalled:.1f}/{args.timeout} min')
        sys.stdout.flush()

        if stalled >= args.timeout:
            print(f'No progress for {args.timeout} min — hung. Hibernating in 30s...')
            sys.stdout.flush()
            time.sleep(30)
            subprocess.run(['shutdown', '/h'])
            return

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
