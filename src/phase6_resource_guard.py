"""Remote Phase 6 transfer bound, independent of model/engine responsiveness."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def counters():
    ports = [p for p in Path('/sys/class/net').iterdir() if p.name != 'lo']
    return tuple(sum(int((p / 'statistics' / name).read_text()) for p in ports)
                 for name in ('rx_bytes', 'tx_bytes'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir
    while not (directory / '.worker.lock').exists():
        time.sleep(.5)
    pid = int((directory / '.worker.lock').read_text())
    while True:
        if (directory / 'worker.done.json').exists() or (directory / 'error.json').exists():
            return
        try:
            rx, tx = counters()
            # Rental-lifetime namespace counters are deliberately NOT reset.
            # 10GB of the 120GB ingress envelope remains for image/control
            # traffic outside this namespace; 1GB egress remains for teardown.
            exceeded = rx >= 110 * 10**9 or tx >= 7 * 10**9
        except Exception:
            exceeded = True
            rx = tx = None
        if exceeded:
            error = directory / 'worker.error.json'
            temp = error.with_suffix('.tmp')
            temp.write_text(json.dumps({'status': 'failed', 'error': 'transfer_envelope_guard',
                                        'rx_bytes': rx, 'tx_bytes': tx, 'unix': time.time()}))
            temp.replace(error)
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.5)


if __name__ == '__main__':
    main()
