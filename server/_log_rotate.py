"""
stdin → rotating-file shim for the bash-based Pi stream daemon.

The server-side process self-rotates because it owns the Python logger.
The Pi-side daemon is bash + rpicam-vid, so we put Python back into the
write path: the daemon's stdout pipes here, and each line is emitted
through a RotatingFileHandler that rotates at maxBytes.

Usage: python3 -u _log_rotate.py <log_path> [max_bytes] [backup_count]
"""

import logging
import logging.handlers
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: _log_rotate.py <log_path> [max_bytes] [backup_count]",
            file=sys.stderr,
        )
        return 2

    log_path = sys.argv[1]
    max_bytes = int(sys.argv[2]) if len(sys.argv) > 2 else 10 * 1024 * 1024
    backup_count = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger = logging.getLogger("stream")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for line in sys.stdin:
        line = line.rstrip("\n")
        if line:
            logger.info(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
