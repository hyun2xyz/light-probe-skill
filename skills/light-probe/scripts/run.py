#!/usr/bin/env python3
"""Run Light Probe with its isolated Python, without shell activation."""
from pathlib import Path
import subprocess
import sys
from bootstrap import python_path


def main():
    python = python_path()
    if not python.is_file():
        print('Quick-mode environment is missing. Run scripts/bootstrap.py first.', file=sys.stderr)
        return 1
    return subprocess.call([str(python), str(Path(__file__).with_name('light_probe.py')), *sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
