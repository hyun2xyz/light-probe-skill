#!/usr/bin/env python3
"""No model download/inference: cache, SDK-contract and process-lock regressions."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import download_models
import model_registry as registry

SCRIPTS = Path(__file__).resolve().parent
CACHE_ENV = ("LIGHT_PROBE_HUB_ROOT", "LIGHT_PROBE_HF_HUB", "HF_HUB_CACHE",
             "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME")


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "cache"

    def test_cache_precedence_and_standard_defaults(self):
        env = {key: value for key, value in os.environ.items() if key not in CACHE_ENV}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(registry.hub_root(), Path("~/.cache/huggingface/hub").expanduser())
            os.environ["XDG_CACHE_HOME"] = str(self.root / "xdg")
            self.assertEqual(registry.hub_root(), self.root / "xdg/huggingface/hub")
            os.environ["HF_HOME"] = str(self.root / "hf")
            self.assertEqual(registry.hub_root(), self.root / "hf/hub")
            for variable in reversed(CACHE_ENV[:4]):
                os.environ[variable] = str(self.root / variable)
                self.assertEqual(registry.hub_root(), self.root / variable)

    def test_inventory_covers_all_components_with_immutable_revisions(self):
        pairs = {(m["component"], m["filename"]) for m in registry.MODELS}
        self.assertEqual(len(registry.MODELS), 25)
        self.assertEqual(sum(m["size"] for m in registry.MODELS), 11024885148)
        for key, (repo, revision, filenames) in registry.COMPONENTS.items():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
            self.assertEqual({name for component, name in pairs if component == key}, set(filenames))
            for item in (m for m in registry.MODELS if m["component"] == key):
                self.assertEqual((item["repo"], item["revision"]), (repo, revision))
                self.assertGreater(item["size"], 0)
                self.assertRegex(item["digest"], r"^[0-9a-f]{64}$" if item["algorithm"] == "sha256" else r"^[0-9a-f]{40}$")

    def test_check_is_read_only_stdlib_only_and_defaults_to_no_download(self):
        code = '''
import importlib.abc, json, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','diffusers','huggingface_hub','filelock','fcntl','numpy','PIL'}:
            raise AssertionError('Optional/platform import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, sys.argv[1])
import download_models
sys.argv = ['download_models.py', '--hub-root', sys.argv[2]]
raise SystemExit(download_models.main())
'''
        result = subprocess.run([sys.executable, "-S", "-c", code, str(SCRIPTS), str(self.root)],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 1, result.stderr)
        status = json.loads(result.stdout)
        self.assertFalse(status["ready"])
        self.assertEqual(len(status["missing"]), 25)
        self.assertFalse(self.root.exists())

    def test_hash_check_detects_equal_size_corruption(self):
        self.root.mkdir()
        path = self.root / "fixture.bin"
        path.write_bytes(b"test")
        for algorithm, digest in (("sha256", hashlib.sha256(b"test").hexdigest()),
                                  ("git-sha1", hashlib.sha1(b"blob 4\0test").hexdigest())):
            item = {"size": 4, "algorithm": algorithm, "digest": digest}
            self.assertIsNone(registry.file_issue(path, item, True))
            path.write_bytes(b"nope")
            self.assertIsNone(registry.file_issue(path, item, False))
            self.assertIn("mismatch", registry.file_issue(path, item, True))
            path.write_bytes(b"test")

    def fake_inventory(self):
        # Preserve every real repository/revision/path, but use tiny contents.
        models = []
        content = {}
        for item in registry.MODELS:
            data = (item["repo"] + "/" + item["filename"]).encode("utf-8")
            content[(item["repo"], item["filename"])] = data
            models.append({**item, "size": len(data), "algorithm": "sha256",
                           "digest": hashlib.sha256(data).hexdigest()})
        return tuple(models), content

    def mock_download(self, models, sdk):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(registry, "MODELS", models))
        stack.enter_context(patch.object(download_models, "MODELS", models))
        stack.enter_context(patch.dict(sys.modules, {"huggingface_hub": sdk}))
        stack.enter_context(patch.object(download_models, "acquire_lock", return_value=types.SimpleNamespace(release=lambda: None)))
        stack.enter_context(redirect_stdout(io.StringIO()))

    def test_sdk_download_covers_every_file_reuses_cache_and_preserves_partial(self):
        models, content = self.fake_inventory()
        calls = []
        def fake_hub_download(**kwargs):
            calls.append(kwargs)
            item = next(item for item in models if (item["repo"], item["filename"]) == (kwargs["repo_id"], kwargs["filename"]))
            target = registry.file_path(item, Path(kwargs["cache_dir"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content[(item["repo"], item["filename"])])
            return str(target)
        self.mock_download(models, types.SimpleNamespace(hf_hub_download=fake_hub_download))
        existing = registry.file_path(models[0], self.root)
        existing.parent.mkdir(parents=True)
        existing.write_bytes(content[(models[0]["repo"], models[0]["filename"])])
        partial = self.root / "untouched.incomplete"
        partial.write_bytes(b"SDK owns this")
        status_file = self.root / "status.json"
        result = download_models.download(self.root, status_file)
        self.assertTrue(result["ready"])
        self.assertEqual(result["validation"], "hashes")
        self.assertEqual(len(calls), len(models) - 1)
        for call in calls:
            expected = next(m for m in models if (m["repo"], m["filename"]) == (call["repo_id"], call["filename"]))
            self.assertEqual(call["revision"], expected["revision"])
            self.assertFalse(call["local_files_only"])
        self.assertEqual(partial.read_bytes(), b"SDK owns this")
        self.assertEqual(json.loads(status_file.read_text(encoding="utf-8"))["status"], "done")

    def test_corrupt_existing_file_is_preserved_without_sdk_call(self):
        models, _ = self.fake_inventory()
        def forbidden(**kwargs):
            self.fail("Corrupt cache must not trigger replacement/download")
        self.mock_download(models, types.SimpleNamespace(hf_hub_download=forbidden))
        path = registry.file_path(models[0], self.root)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * models[0]["size"])
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            download_models.download(self.root, self.root / "status.json")
        self.assertEqual(path.read_bytes(), b"x" * models[0]["size"])
        self.assertEqual(json.loads((self.root / "status.json").read_text(encoding="utf-8"))["status"], "error")

    @unittest.skipUnless(importlib.util.find_spec("filelock"), "optional filelock not installed")
    def test_lock_excludes_another_process_and_releases(self):
        path = self.root / "gpu.lock"
        lock = registry.acquire_lock(path, "busy")
        code = '''
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from model_registry import acquire_lock
try:
    lock = acquire_lock(Path(sys.argv[2]), 'busy')
except RuntimeError:
    raise SystemExit(0)
else:
    lock.release()
    raise SystemExit(1)
'''
        try:
            result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS), str(path)],
                                    capture_output=True, text=True, encoding="utf-8", timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            lock.release()
        registry.acquire_lock(path, "busy").release()
        self.assertIsInstance(registry.gpu_lock_path(), Path)

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "quick image dependencies not installed")
    def test_engine_import_and_missing_models_do_not_load_ml_or_platform_modules(self):
        code = '''
import importlib.abc, os, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','diffusers','huggingface_hub','filelock','fcntl'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, sys.argv[1])
os.environ['LIGHT_PROBE_HUB_ROOT'] = sys.argv[2]
import diffusion_engine
assert not diffusion_engine.model_status()['ready']
try:
    diffusion_engine._load_pipeline()
except diffusion_engine.ModelUnavailableError:
    pass
else:
    raise AssertionError('Missing models accepted')
'''
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS), str(self.root)],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
