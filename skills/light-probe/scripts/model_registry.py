"""Pinned offline model inventory; no third-party imports or network access.

Sizes and digests are from the official Hub metadata at the pinned revisions.
LFS weights use SHA256; small Git files use the Git blob SHA1 format.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

# Each file: (path, bytes, digest algorithm, expected digest).
_INVENTORY = {
    'base': ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b', (
        ('model_index.json', 609, 'git-sha1',
         '6cc5138ddb67e9309c9b4e058e33e52087f1d215'),
        ('scheduler/scheduler_config.json', 479, 'git-sha1',
         'e5bc8421e047838523be7acfb6720f167f7382f6'),
        ('unet/config.json', 1680, 'git-sha1',
         'c8714c90f0e2409156da42781954416cb7df36af'),
        ('unet/diffusion_pytorch_model.fp16.safetensors', 5135149760, 'sha256',
         '83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139'),
        ('text_encoder/config.json', 565, 'git-sha1',
         'cde352ada4bb95fdad2fc503b8121257cef215a6'),
        ('text_encoder/model.fp16.safetensors', 246144152, 'sha256',
         '660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd'),
        ('text_encoder_2/config.json', 575, 'git-sha1',
         'da1848b5ed17b676f021578838f12c4023b86379'),
        ('text_encoder_2/model.fp16.safetensors', 1389382176, 'sha256',
         'ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4'),
        ('tokenizer/tokenizer_config.json', 737, 'git-sha1',
         '2e8612a429492973fe60635b3f44a28b065cfac0'),
        ('tokenizer/vocab.json', 1059962, 'git-sha1',
         '469be27c5c010538f845f518c4f5e8574c78f7c8'),
        ('tokenizer/merges.txt', 524619, 'git-sha1',
         '76e821f1b6f0a9709293c3b6b51ed90980b3166b'),
        ('tokenizer_2/tokenizer_config.json', 725, 'git-sha1',
         'a8438e020c4497a429240d6b89e0bf9a6e2ffa92'),
        ('tokenizer_2/vocab.json', 1059962, 'git-sha1',
         '469be27c5c010538f845f518c4f5e8574c78f7c8'),
        ('tokenizer_2/merges.txt', 524619, 'git-sha1',
         '76e821f1b6f0a9709293c3b6b51ed90980b3166b'),
        ('tokenizer/special_tokens_map.json', 472, 'git-sha1',
         '2c2130b544c0c5a72d5d00da071ba130a9800fb2'),
        ('tokenizer_2/special_tokens_map.json', 460, 'git-sha1',
         'ae0c5be6f35217e51c4c000fd325d8de0294e99c'),
    )),
    'controlnet': ('diffusers/controlnet-depth-sdxl-1.0', '17bb97973f29801224cd66f192c5ffacf82648b4', (
        ('config.json', 1274, 'git-sha1',
         '9e36326b51f8c1d0b35cc74aa75cfd2504cc8d5f'),
        ('diffusion_pytorch_model.fp16.safetensors', 2502139134, 'sha256',
         '66a6813e6bd7270ecfe68206a59ddd605a011ae85321188376605c66e0a4f303'),
    )),
    'vae': ('madebyollin/sdxl-vae-fp16-fix', '207b116dae70ace3637169f1ddd2434b91b3a8cd', (
        ('config.json', 631, 'git-sha1',
         'a0fc2b4620b60f1e599418c4de1c3d5e45db875a'),
        ('diffusion_pytorch_model.safetensors', 334643238, 'sha256',
         '1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c'),
    )),
    'depth': ('Intel/dpt-large', 'bc15f29aa3a80d532f2ed650b5e16ac48d8958f9', (
        ('config.json', 942, 'git-sha1',
         'ee679a1bf4c5dc149c47ff8db3855ed787161240'),
        ('model.safetensors', 1367456044, 'sha256',
         'c173e7fd575c4d6c2164621063fed3a0e7e2eb6adbfb7c1845b364bc0fed9ce8'),
        ('preprocessor_config.json', 285, 'git-sha1',
         'a0fbbe1170f71fba7cee165aad8b0d68185def61'),
    )),
    'turbo': ('DiffusionLight/Flickr2K', 'ed635477e5a742e363e54a6ed2193a9a43153113', (
        ('pytorch_lora_weights.safetensors', 23396024, 'sha256',
         'acc0781a04d2341cc6307c8fcfa78768e6f50c1c79ebcdad6f22893207f8b16d'),
    )),
    'exposure': ('DiffusionLight/DiffusionLight', 'bda34823e932601b3eb9aff0276f00925cdeea46', (
        ('pytorch_lora_weights.safetensors', 23396024, 'sha256',
         '6bf601e30c08ce4f1eea6e09e1780bd8ba2588986eaee8379672350707dcddaa'),
    )),
}
COMPONENTS = {key: (repo, revision, tuple(row[0] for row in files))
              for key, (repo, revision, files) in _INVENTORY.items()}
MODELS = tuple(
    {"component": key, "repo": repo, "revision": revision, "filename": name,
     "size": size, "algorithm": algorithm, "digest": digest}
    for key, (repo, revision, files) in _INVENTORY.items()
    for name, size, algorithm, digest in files
)


def hub_root() -> Path:
    """Match HF cache defaults without importing its optional SDK.

    LIGHT_PROBE_HF_HUB is accepted as a legacy alias. Environment values must
    be set before model loading; the engine rejects switching a loaded cache.
    """
    for name in ("LIGHT_PROBE_HUB_ROOT", "LIGHT_PROBE_HF_HUB", "HF_HUB_CACHE",
                 "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return Path(os.environ[name]).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def snapshot_path(key: str, root: Path | None = None) -> Path:
    repo, revision, _ = COMPONENTS[key]
    return (hub_root() if root is None else Path(root).expanduser()) / (
        "models--" + repo.replace("/", "--")) / "snapshots" / revision


def file_path(item: dict, root: Path) -> Path:
    return snapshot_path(item["component"], root) / item["filename"]


def file_issue(path: Path, item: dict, verify_hashes: bool = False) -> str | None:
    """Read-only validation, including regular-file Windows snapshot caches."""
    try:
        if not path.is_file():
            return "missing"
        size = path.stat().st_size
        if size != item["size"]:
            return f"size mismatch: {size} != {item['size']}"
        if verify_hashes:
            algorithm = item["algorithm"]
            digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
            if algorithm == "git-sha1":
                digest.update(f"blob {size}\0".encode("ascii"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["digest"]:
                return f"{algorithm} mismatch"
    except OSError as exc:
        return f"unreadable ({type(exc).__name__})"
    return None


def model_status(root: Path | None = None, verify_hashes: bool = False) -> dict:
    """Check every required local file; never import ML packages or download."""
    root = hub_root() if root is None else Path(root).expanduser()
    missing = []
    available_bytes = 0
    for item in MODELS:
        issue = file_issue(file_path(item, root), item, verify_hashes)
        if issue:
            missing.append(f"{item['repo']}/{item['filename']} ({issue})")
        else:
            available_bytes += item["size"]
    return {"ready": not missing, "missing": missing, "cache_root": str(root),
            "offline": True, "validation": "hashes" if verify_hashes else "sizes",
            "file_count": len(MODELS), "available_files": len(MODELS) - len(missing),
            "available_bytes": available_bytes, "total_bytes": sum(m["size"] for m in MODELS)}


def gpu_lock_path() -> Path:
    # getuid/fcntl are not available on Windows. Avoid usernames in the filename.
    user_key = hashlib.sha256(str(Path.home()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"light-probe-gpu-{user_key}.lock"


def acquire_lock(path: Path, busy_message: str):
    """Optional cross-platform OS lock; a crashed process releases its lock."""
    try:
        from filelock import FileLock, Timeout
    except ImportError as exc:
        raise RuntimeError("Install requirements-diffusion.txt to use DiffusionLight") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(busy_message) from exc
    return lock
