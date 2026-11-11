"""Build a portable skill ZIP without local environments, models, or outputs."""
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'skills/light-probe'
EXCLUDE={'.venv','__pycache__','outputs','.DS_Store'}


def main():
    out=ROOT/'dist'
    out.mkdir(exist_ok=True)
    archive=out/'light-probe.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SOURCE.rglob('*')):
            relative=p.relative_to(SOURCE)
            if any(part in EXCLUDE for part in relative.parts):continue
            if p.is_symlink():raise ValueError(f'Symlink is not distributable: {relative}')
            if p.is_file() and p.suffix not in {'.pyc','.log','.safetensors','.ckpt','.pt','.pth'}:
                z.write(p,(Path('light-probe')/relative).as_posix())
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/'SHA256SUMS').write_text(f'{digest}  light-probe.zip\n',encoding='utf-8')
    print(f'{archive} ({archive.stat().st_size} bytes)')


if __name__=='__main__':main()
