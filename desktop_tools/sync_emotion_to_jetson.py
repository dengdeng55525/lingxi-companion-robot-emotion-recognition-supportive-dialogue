#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Sync VM emotion JSON to Jetson for RoboEyes.')
    p.add_argument('--src', default=str(Path.home() / '.ros' / 'latest_emotion.json'))
    p.add_argument('--jetson-user', default='clbrobot')
    p.add_argument('--jetson-host', default='192.168.210.91')
    p.add_argument('--jetson-target', default='/home/clbrobot/RoboEyes/latest_emotion_vm.json')
    p.add_argument('--interval', type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)

    print('[INFO] src:', src)
    print('[INFO] target: {}@{}:{}'.format(args.jetson_user, args.jetson_host, args.jetson_target))

    last_mtime_ns = -1
    while True:
        try:
            st = src.stat()
            mtime_ns = st.st_mtime_ns
        except FileNotFoundError:
            time.sleep(max(0.05, args.interval))
            continue

        if mtime_ns == last_mtime_ns:
            time.sleep(max(0.05, args.interval))
            continue

        last_mtime_ns = mtime_ns

        remote_tmp = args.jetson_target + '.tmp'
        remote_final = args.jetson_target

        scp_cmd = [
            'scp', '-q',
            str(src),
            '{}@{}:{}'.format(args.jetson_user, args.jetson_host, remote_tmp),
        ]
        mv_cmd = [
            'ssh', '-o', 'BatchMode=yes',
            '{}@{}'.format(args.jetson_user, args.jetson_host),
            'mv "{}" "{}"'.format(remote_tmp, remote_final),
        ]

        try:
            subprocess.check_call(scp_cmd)
            subprocess.check_call(mv_cmd)
        except subprocess.CalledProcessError as exc:
            print('[WARN] sync failed:', exc)

        time.sleep(max(0.05, args.interval))


if __name__ == '__main__':
    main()
