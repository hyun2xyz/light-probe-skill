"""Exercise an installed skill in a temporary path, including a non-ASCII filename."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    python=ROOT/'skills/light-probe/.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.run([str(python),'-m','unittest','discover','-s','skills/light-probe/scripts','-p','test_*.py','-v'],cwd=ROOT,check=True)
    subprocess.run([str(python),'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
    with tempfile.TemporaryDirectory(prefix='light-probe-smoke-') as tmp:
        root=Path(tmp)
        skill=root/'installed'/'light-probe'
        subprocess.run([sys.executable,str(ROOT/'install.py'),'--target',str(skill)],check=True)
        subprocess.run([str(python),str(ROOT/'examples/make_demo.py'),'--out',str(root/'source')],check=True)
        photo=root/'source'/'실습 사진.png'
        shutil.copy2(root/'source/room.png',photo)
        subprocess.run([str(python),str(skill/'scripts/light_probe.py'),'analyze',str(photo),'--out',str(root/'result'),'--display','all','--context',str(root/'source/context.json')],check=True,stdout=subprocess.PIPE)
        data=json.loads((root/'result/result.json').read_text(encoding='utf-8'))
        assert len(data['object_overlay']['masks'])==3
        assert data['scattering']['hits']
        subprocess.run([str(python),'-c','from PIL import Image; import sys; Image.open(sys.argv[1]).verify()',str(root/'result/annotated.png')],check=True)
        print(json.dumps({'smoke':'passed','objects':3,'non_ascii_path':True,'isolated_install':True}))


if __name__=='__main__':main()
