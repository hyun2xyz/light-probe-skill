"""Export a real analysis as an original/result pair with portable provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True,help='Public provenance JSON, no local paths')
    parser.add_argument('--evidence',type=Path,required=True,help='Folder containing the actual generated probe/ball')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('Output folder must be new or empty')
    source=json.loads(args.source.read_text(encoding='utf-8'))
    result=json.loads(args.result.read_text(encoding='utf-8'))
    if result.get('mode')!='diffusion':parser.error('This exporter expects a completed DiffusionLight analysis')
    raw=args.result.parent/'input.png'
    annotated=args.result.parent/'annotated.png'
    with Image.open(annotated) as image:output=image.convert('RGB')
    with Image.open(raw) as image:
        photo=image.convert('RGB').resize((output.width-40,output.height-40),Image.Resampling.LANCZOS)
    original=Image.new('RGB',output.size,'white')
    original.paste(photo,(20,20))
    pair=Image.new('RGB',(output.width*2+16,output.height),'white')
    pair.paste(original,(0,0));pair.paste(output,(output.width+16,0))
    args.out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(raw,args.out/'input.png')
    original.save(args.out/'original.jpg',quality=93,subsampling=0)
    output.save(args.out/'result.jpg',quality=93,subsampling=0)
    pair.save(args.out/'comparison.jpg',quality=93,subsampling=0)
    # Whitelist analysis data so private cache paths never enter the public repo.
    keys=('mode','quality','device','elapsed_ms','render_elapsed_ms','width','height','anchor','display',
          'seed','steps_per_exposure','exposures','lights','note','optics','object_overlay','scattering')
    data={k:result[k] for k in keys if k in result}
    data['input_sha256']=digest(raw)
    data['source']=source
    data['generated_evidence']={}
    for name in ['ball_ev00.png','probe_preview.png']:
        origin=args.evidence/name
        if not origin.is_file():raise FileNotFoundError('Missing actual inference artifact: '+name)
        shutil.copy2(origin,args.out/name)
        data['generated_evidence'][name]=digest(origin)
    for name,value in [('analysis.json',data),('context.json',result.get('context') or {})]:
        (args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(args.out/'comparison.jpg')


if __name__=='__main__':main()
