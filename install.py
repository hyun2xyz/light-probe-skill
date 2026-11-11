#!/usr/bin/env python3
"""Install the self-contained skill; never change an LLM's permissions or config."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parent / 'skills' / 'light-probe'
IGNORE = shutil.ignore_patterns('.venv', '__pycache__', '*.pyc', '.DS_Store', 'outputs')


def install(target, replace=False):
    target = Path(target).expanduser().absolute()
    if target.name != 'light-probe':
        raise ValueError('Target must be a directory named light-probe')
    if target.is_symlink():
        raise ValueError('Target is a symlink; use its owning installer or choose another target')
    if target.exists() and not replace:
        raise FileExistsError('Skill already exists. Review it first; --replace creates a backup')
    if SOURCE == target or SOURCE in target.parents or target in SOURCE.parents:
        raise ValueError('Target must be outside the source skill')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.light-probe-stage-', dir=target.parent))
    backup = None
    try:
        shutil.copytree(SOURCE, stage, dirs_exist_ok=True, ignore=IGNORE)
        if not (stage/'SKILL.md').is_file():
            raise RuntimeError('Incomplete skill source')
        if target.exists():
            # Keep old SKILL.md files outside the client's skill discovery root.
            backup_root = target.parent.parent/'light-probe-backups'
            backup_root.mkdir(parents=True,exist_ok=True)
            backup = backup_root/('light-probe-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            target.rename(backup)
        try:
            stage.rename(target)
        except Exception:
            if backup is not None and not target.exists():
                backup.rename(target)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target, backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--client', choices=['codex', 'claude'])
    group.add_argument('--target', type=Path, help='Custom skill folder, ending in light-probe')
    parser.add_argument('--replace', action='store_true', help='Back up an existing skill before replacement')
    parser.add_argument('--setup', action='store_true', help='Create an isolated Python environment and install quick-mode dependencies')
    args = parser.parse_args()
    base = '.agents' if args.client == 'codex' else '.claude'
    target = args.target or Path.home()/base/'skills'/'light-probe'
    try:
        target, backup = install(target, args.replace)
        print(json.dumps({'status':'installed', 'skill':str(target), 'backup':str(backup) if backup else None}, ensure_ascii=False), flush=True)
        if args.setup:
            return subprocess.call([sys.executable, str(target/'scripts/bootstrap.py')])
        print('Next: run scripts/bootstrap.py inside the installed skill with Python 3.10+.')
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Install failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
