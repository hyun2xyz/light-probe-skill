#!/usr/bin/env python3
"""Check or explicitly download all pinned DiffusionLight components.

No action defaults to --check. Only --download imports the optional Hub SDK
and permits network access. HF manages its own cache, resume files and locks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from model_registry import MODELS, acquire_lock, file_issue, file_path, hub_root, model_status


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download(root: Path, status_path: Path) -> dict:
    """Download exact filenames/revisions via the official SDK, then hash them.

    Existing invalid files are preserved and reported, not silently overwritten.
    Model snapshots may be symlinks (Unix) or regular files (Windows).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install requirements-diffusion.txt before --download") from exc

    root = root.expanduser()
    completed = 0
    total = sum(item["size"] for item in MODELS)
    started = time.time()
    current = ""

    def report(status: str, message: str, **extra) -> None:
        atomic_json(status_path, {
            "status": status, "pid": os.getpid(), "current_file": current,
            "completed_bytes": completed, "total_bytes": total,
            "message": message, "started_at": started, "updated_at": time.time(), **extra,
        })

    lock = acquire_lock(root / ".locks" / "light-probe-download.lock",
                        "Another Light Probe download owns this cache; wait for it to exit")
    try:
        report("running", "Checking and downloading the pinned model inventory")
        for item in MODELS:
            current = f"{item['repo']}/{item['filename']}"
            target = file_path(item, root)
            # A broken symlink is an existing invalid entry, too. Do not replace it.
            if os.path.lexists(target):
                report("running", f"Verifying cached {current}")
                issue = file_issue(target, item, verify_hashes=True)
                if issue:
                    raise RuntimeError(f"Existing cache file failed validation ({issue}); preserved: {target}")
            else:
                report("running", f"Downloading {current}")
                print(f"Downloading {current} ({item['size']} bytes)", flush=True)
                try:
                    downloaded = Path(hf_hub_download(
                        repo_id=item["repo"], filename=item["filename"], revision=item["revision"],
                        cache_dir=str(root), local_files_only=False,
                    ))
                except Exception as exc:
                    # SDK errors may contain temporary signed URLs; do not log them.
                    raise RuntimeError(
                        f"Hub download failed for {current} ({type(exc).__name__}). "
                        "Check network access, disk space, HF_HUB_OFFLINE and model access; "
                        "rerun --download to resume."
                    ) from None
                if downloaded.resolve() != target.resolve():
                    raise RuntimeError(f"SDK returned an unexpected snapshot path for {current}")
                report("running", f"Verifying downloaded {current}")
                issue = file_issue(target, item, verify_hashes=True)
                if issue:
                    raise RuntimeError(f"Downloaded file failed validation ({issue}); preserved: {target}")
            completed += item["size"]
            print(f"Verified {current}", flush=True)
            report("running", f"Verified {current}")
        result = model_status(root)
        if not result["ready"]:
            raise RuntimeError("Model inventory changed during download; rerun --check")
        # Each file was hash-verified above; avoid rereading all weights twice.
        result["validation"] = "hashes"
        current = ""
        report("done", "All pinned model files downloaded and hash-verified", model_status=result)
        return result
    except BaseException as exc:
        report("cancelled" if isinstance(exc, KeyboardInterrupt) else "error",
               str(exc), error=type(exc).__name__)
        raise
    finally:
        lock.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Read-only local readiness check (default)")
    action.add_argument("--download", action="store_true", help="Explicitly download missing pinned files")
    parser.add_argument("--verify", action="store_true", help="With --check, hash all files (reads about 11 GB)")
    parser.add_argument("--hub-root", type=Path, default=None, help="Override the Hugging Face Hub cache root")
    parser.add_argument("--status-file", type=Path, default=None,
                        help="Download status JSON; default: <hub-root>/light-probe-download-status.json")
    args = parser.parse_args()
    if args.verify and args.download:
        parser.error("--download always verifies hashes; use --verify with --check")
    if args.status_file is not None and not args.download:
        parser.error("--status-file is only used with --download; --check does not write files")
    root = (args.hub_root or hub_root()).expanduser()
    try:
        if args.download:
            status_path = (args.status_file or root / "light-probe-download-status.json").expanduser()
            result = download(root, status_path)
        else:
            result = model_status(root, verify_hashes=args.verify)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    except KeyboardInterrupt:
        print("Download interrupted; rerun --download to resume.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
