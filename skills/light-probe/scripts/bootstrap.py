#!/usr/bin/env python3
"""Create/check a private quick-mode environment. Does not download AI models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def python_path():
    return ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check only; no install or download')
    args = parser.parse_args()
    if sys.version_info < (3,10):
        parser.error('Python 3.10 or newer is required')
    python = python_path()
    if args.check and not python.is_file():
        print(json.dumps({'ready':False,'reason':'Run scripts/bootstrap.py first','python':str(python)}))
        return 1
    if not args.check:
        if not python.is_file():
            venv.EnvBuilder(with_pip=True).create(ROOT/'.venv')
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT/'requirements.txt')], check=True)
    check = subprocess.run([str(python), '-c',
        'import json,sys,numpy,PIL,cv2; print(json.dumps({"ready":True,"python":sys.executable,"numpy":numpy.__version__,"Pillow":PIL.__version__,"OpenCV":cv2.__version__}))'])
    return check.returncode


if __name__ == '__main__':
    raise SystemExit(main())
