import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('installer',Path(__file__).resolve().parents[1]/'install.py')
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def test_install_refuses_overwrite_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'skills'/'light-probe'
            result,backup=installer.install(target)
            self.assertIsNone(backup)
            self.assertTrue((result/'SKILL.md').is_file())
            self.assertFalse((result/'.venv').exists())
            (target/'user-note.txt').write_text('preserve me',encoding='utf-8')
            with self.assertRaises(FileExistsError):installer.install(target)
            _,backup=installer.install(target,replace=True)
            self.assertNotEqual(backup.parent,target.parent)
            self.assertEqual((backup/'user-note.txt').read_text(encoding='utf-8'),'preserve me')
            self.assertTrue((target/'assets/fonts/OFL.txt').is_file())

    def test_unsafe_target_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):installer.install(Path(tmp)/'documents')


if __name__=='__main__':unittest.main()
