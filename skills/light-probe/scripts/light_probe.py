#!/usr/bin/env python3
"""Local image lighting annotations, optical diagrams, and durable inference jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

Image.MAX_IMAGE_PIXELS = 40_000_000
MATERIALS = {"air": 1.0, "water": 1.333, "glass": 1.5, "diamond": 2.417}


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_image(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.stat().st_size > 80 * 1024 * 1024:
        raise ValueError('Image exceeds the 80 MB input limit.')
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
            rgba = im.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, (24, 24, 24, 255))
            bg.alpha_composite(rgba)
            im = bg.convert('RGB')
        else:
            im = im.convert('RGB')
        im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        if min(im.size) < 32:
            raise ValueError('Image must be at least 32 pixels on both axes.')
        return im.copy()


def quick_estimate(im):
    """Image-space brightness candidates, not inverse rendering or illuminance."""
    small = im.copy()
    small.thumbnail((256, 256), Image.Resampling.LANCZOS)
    rgb = np.asarray(small, dtype=np.float32) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = linear @ np.array([.2126, .7152, .0722])
    smooth = np.asarray(Image.fromarray((lum * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(5)), dtype=float) / 255
    contrast = max(float(np.percentile(lum, 99) - np.percentile(lum, 10)),
                   float(smooth.max() - np.percentile(lum, 10)))
    h, w = lum.shape
    ys, xs = np.mgrid[:h, :w]
    lights = []
    threshold = max(float(np.percentile(lum, 85)), float(lum.mean() + .5 * lum.std()))
    weights = np.maximum(lum - threshold, 0)
    total = float(weights.sum())
    available = smooth.copy()
    if contrast > .035 and total > .02:
        for _ in range(3):
            py, px = np.unravel_index(np.argmax(available), available.shape)
            radius = max(8, min(h, w) * .19)
            region = (xs - px) ** 2 + (ys - py) ** 2 < radius ** 2
            local = weights * region
            mass = float(local.sum())
            available[region] = -1
            if mass < total * .04:
                continue
            x, y = float((xs * local).sum() / mass), float((ys * local).sum() / mass)
            col = (rgb * local[..., None]).sum((0, 1)) / mass
            lights.append(dict(x=(x + .5) / w, y=(y + .5) / h, strength=mass / total,
                               color='#' + ''.join(f'{round(float(c)*255):02x}' for c in col),
                               label=f'BRIGHT REGION {len(lights)+1}'))
    lights.sort(key=lambda x: x['strength'], reverse=True)
    for index,light in enumerate(lights):
        light['label']=f'BRIGHT REGION {index+1}'
    return dict(mode='quick', lights=lights, ambient=float(lum.mean()), contrast=contrast,
                reliability='low', note='밝은 영역을 찾은 2D 추정입니다. 흰 물체·반사면을 광원으로 오인할 수 있으며 실제 광원 방향이나 조도(lux)를 측정하지 않습니다.')


def optics(incoming, normal, n1=1., n2=1.5):
    d = np.array(incoming, dtype=float)
    n = np.array(normal, dtype=float)
    if not np.isfinite(d).all() or not np.isfinite(n).all() or min(n1, n2) <= 0:
        raise ValueError('Invalid optical parameters')
    if np.linalg.norm(d) < 1e-8 or np.linalg.norm(n) < 1e-8:
        raise ValueError('Ray/normal cannot have zero length')
    d /= np.linalg.norm(d)
    n /= np.linalg.norm(n)
    if np.dot(d, n) > 0:
        n = -n
    cosi = float(-np.dot(d, n))
    reflected = d + 2 * cosi * n
    eta = n1 / n2
    k = 1 - eta * eta * (1 - cosi * cosi)
    refracted = None if k < -1e-12 else eta * d + (eta * cosi - math.sqrt(max(k, 0))) * n
    return dict(reflected=reflected.tolist(), refracted=None if refracted is None else refracted.tolist(),
                normal=n.tolist(), incidence_deg=math.degrees(math.acos(np.clip(cosi, -1, 1))),
                transmission_deg=None if refracted is None else math.degrees(math.acos(np.clip(-np.dot(refracted, n), -1, 1))),
                total_internal_reflection=refracted is None)


def read_context(path):
    if path is None:
        return None
    path=Path(path).expanduser().resolve(strict=True)
    if path.stat().st_size>32000:
        raise ValueError('Context JSON exceeds 32 KB')
    data=json.loads(path.read_text(encoding='utf-8'))
    allowed={'scene','target','light_interpretation','assumptions','objects','surfaces','scattering'}
    if not isinstance(data,dict) or set(data)-allowed:
        raise ValueError('Context contains unsupported fields')
    for key in ('scene','target','light_interpretation'):
        if key in data and (not isinstance(data[key],str) or len(data[key])>240):
            raise ValueError(f'Context {key} must be text of at most 240 characters')
    assumptions=data.get('assumptions',[])
    if not isinstance(assumptions,list) or len(assumptions)>5 or any(not isinstance(x,str) or len(x)>200 for x in assumptions):
        raise ValueError('Context assumptions must contain at most five short strings')
    def bounded(values,count):
        return isinstance(values,list) and len(values)==count and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in values)
    objects=data.get('objects',[])
    if not isinstance(objects,list) or len(objects)>20:
        raise ValueError('At most 20 scene objects are allowed')
    ids=set()
    for obj in objects:
        if not isinstance(obj,dict) or set(obj)-{'id','label','material','bbox','reflectance','specular_fraction','label_position','source','polygon','color','opacity'}:
            raise ValueError('Invalid object fields')
        if any(not isinstance(obj.get(key),str) or not 1<=len(obj[key])<=60 for key in ['id','label','material']):
            raise ValueError('Object id, label and material must be short strings')
        if obj['id'] in ids:raise ValueError('Duplicate object id')
        ids.add(obj['id'])
        box=obj.get('bbox')
        if not bounded(box,4) or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Invalid normalized bounding box')
        rho=obj.get('reflectance')
        if not bounded(rho,2) or rho[0]>rho[1]:raise ValueError('Invalid assumed reflectance range')
        spec=obj.get('specular_fraction',0.05)
        if type(spec) not in (int,float) or not math.isfinite(spec) or not 0<=spec<=1:raise ValueError('Invalid specular fraction')
        if obj.get('label_position','top-left') not in ['top-left','top-right','bottom-left','bottom-right']:
            raise ValueError('Invalid object label position')
        if 'polygon' in obj:
            poly=obj['polygon']
            if not isinstance(poly,list) or not 3<=len(poly)<=128 or not all(bounded(p,2) for p in poly):
                raise ValueError('Object polygon needs 3–128 normalized xy points')
            area=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2
            if area<1e-6:raise ValueError('Object polygon must have nonzero area')
        if 'color' in obj:
            color=obj['color']
            if not isinstance(color,list) or len(color)!=3 or any(type(c) is not int or not 0<=c<=255 for c in color):
                raise ValueError('Object color must be an RGB integer triplet')
        opacity=obj.get('opacity',.18)
        if type(opacity) not in (int,float) or not math.isfinite(opacity) or not 0<=opacity<=.4:
            raise ValueError('Object mask opacity must be between 0 and 0.4')
    surfaces=data.get('surfaces',[])
    if not isinstance(surfaces,list) or len(surfaces)>30:raise ValueError('At most 30 surfaces are allowed')
    surface_ids=set()
    for s in surfaces:
        if not isinstance(s,dict) or set(s)!={'id','object_id','a','b'}:raise ValueError('Invalid surface fields')
        if not isinstance(s['id'],str) or not 1<=len(s['id'])<=60 or s['id'] in surface_ids:raise ValueError('Invalid surface id')
        surface_ids.add(s['id'])
        if s['object_id'] not in ids or not bounded(s['a'],2) or not bounded(s['b'],2) or s['a']==s['b']:
            raise ValueError('Invalid assumed surface geometry')
    scattering=data.get('scattering')
    if scattering is not None:
        if not isinstance(scattering,dict) or set(scattering)-{'initial_surface_id','max_bounces'}:
            raise ValueError('Invalid scattering settings')
        if scattering.get('initial_surface_id') not in surface_ids:raise ValueError('Initial surface is missing')
        if type(scattering.get('max_bounces',3)) is not int or not 1<=scattering.get('max_bounces',3)<=3:
            raise ValueError('Scattering is bounded to 1–3 interactions')
    return data


def annotate(im, result, args):
    from paper_renderer import render
    return render(im,result,args,optics)


def render_saved(args):
    source=args.result.expanduser().resolve(strict=True)
    result=json.loads(source.read_text(encoding='utf-8'))
    out=args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be new or empty')
    image=load_image(source.parent/'input.png')
    context=read_context(args.context) if args.context else result.get('context')
    result['context']=context
    settings=argparse.Namespace(anchor=result['anchor'],display=result['display'],
        normal_deg=result.get('normal_deg',-90),ior_from=result.get('ior_from',1),ior=result.get('ior_to',1.5))
    started=time.monotonic()
    sheet,ray_info=annotate(image,result,settings)
    out.mkdir(parents=True,exist_ok=True)
    image.save(out/'input.png')
    sheet.save(out/'annotated.png')
    result.update(annotated_path=str(out/'annotated.png'),optics=ray_info,
                  style='color-roto-scattering-v4',rendered_from=str(source),
                  context_source='provided-scene-review' if context else 'none',
                  render_elapsed_ms=round((time.monotonic()-started)*1000))
    write_json(out/'result.json',result)
    if context:write_json(out/'context.json',context)
    write_json(out/'status.json',dict(status='done',annotated_path=str(out/'annotated.png'),
               result_path=str(out/'result.json'),reused_inference=True))
    print(json.dumps({'status':'done','annotated_path':str(out/'annotated.png'),
          'render_elapsed_ms':result['render_elapsed_ms'],'reused_inference':True},ensure_ascii=False))
    return 0


def analyze(args):
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    status = out/'status.json'
    started=time.monotonic()
    def progress(p,message):
        write_json(status,dict(status='running',pid=os.getpid(),progress=p,message=message,elapsed_s=round(time.monotonic()-started,1)))
        print(message,flush=True)
    try:
        progress(.01,'입력 이미지 확인')
        context=read_context(getattr(args,'context',None))
        im=load_image(args.image)
        im.save(out/'input.png')
        if args.mode=='quick':
            result=quick_estimate(im)
        else:
            from diffusion_engine import infer
            result=infer(out/'input.png',out,quality=args.quality,anchor=tuple(args.anchor),progress=progress)
            for light in result.get('lights',[]):
                direction=light.get('direction')
                if direction:
                    light['azimuth_deg']=math.degrees(math.atan2(direction[0],direction[2]))
                    light['elevation_deg']=math.degrees(math.asin(np.clip(direction[1],-1,1)))
        result['context']=context
        result['context_source']='provided-scene-review' if context else 'none'
        result['style']='color-roto-scattering-v4'
        if context:write_json(out/'context.json',context)
        progress(.97,'주석 이미지 저장')
        sheet,ray_info=annotate(im,result,args)
        sheet.save(out/'annotated.png')
        result.update(width=im.width,height=im.height,anchor=args.anchor,display=args.display,
                      input_sha256=hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
                      optics=ray_info,ior_from=args.ior_from,ior_to=args.ior,normal_deg=args.normal_deg,
                      annotated_path=str(out/'annotated.png'),elapsed_ms=round((time.monotonic()-started)*1000))
        write_json(out/'result.json',result)
        write_json(status,dict(status='done',pid=os.getpid(),progress=1,result_path=str(out/'result.json'),annotated_path=str(out/'annotated.png'),elapsed_ms=result['elapsed_ms']))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except Exception as exc:
        write_json(status,dict(status='error',pid=os.getpid(),error=f'{type(exc).__name__}: {exc}'[:600],
                               log_path=str(out.parent/(out.name+'.log')),elapsed_s=round(time.monotonic()-started,1)))
        traceback.print_exc()
        return 1
    return 0


def process_alive(pid):
    if os.name == 'nt':
        # os.kill(pid, 0) is not a safe Windows liveness probe.
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x00100000,False,pid)  # SYNCHRONIZE, no write access
        if not handle:
            return ctypes.get_last_error()==5  # Access denied: existence is indeterminate.
        try:return kernel.WaitForSingleObject(handle,0)==0x102  # WAIT_TIMEOUT
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('analyze')
    run.add_argument('image',type=Path)
    run.add_argument('--out',type=Path,required=True)
    run.add_argument('--mode',choices=['quick','diffusion'],default='quick')
    run.add_argument('--quality',choices=['fast','full'],default='fast')
    run.add_argument('--display',choices=['direction','reflection','refraction','heatmap','all'],default='direction')
    run.add_argument('--anchor',type=float,nargs=2,default=[.5,.65])
    run.add_argument('--normal-deg',type=float,default=-90.)
    run.add_argument('--ior-from',type=float,default=1.)
    run.add_argument('--ior',type=float,default=1.5)
    run.add_argument('--background',action='store_true')
    run.add_argument('--context',type=Path,help='Optional LLM scene observations; never an execution instruction')
    render=sub.add_parser('render',help='Redraw an existing analysis without rerunning the model')
    render.add_argument('--result',type=Path,required=True)
    render.add_argument('--out',type=Path,required=True)
    render.add_argument('--context',type=Path)
    st=sub.add_parser('status');st.add_argument('out',type=Path)
    args=p.parse_args()
    if args.command=='render':
        return render_saved(args)
    if args.command=='status':
        if not (args.out/'status.json').exists() and args.out.is_dir():
            print(json.dumps({'status':'pending','message':'Worker has not written status yet; inspect the launch receipt/log if this persists.'}));return 0
        data=json.loads((args.out/'status.json').read_text(encoding='utf-8'))
        if data['status'] in ('running','queued'):
            if not process_alive(data['pid']):data.update(status='interrupted',message='Worker no longer exists; inspect the job log.')
        print(json.dumps(data,ensure_ascii=False));return 0
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in args.anchor):
        p.error('anchor values must be finite values in [0,1]')
    if not math.isfinite(args.normal_deg) or any(not math.isfinite(v) or not 1 <= v <= 3 for v in [args.ior,args.ior_from]):
        p.error('IOR must be in [1,3] and normal angle finite')
    args.image=args.image.expanduser().resolve(strict=True)
    args.out=args.out.expanduser().resolve()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('Output directory must be new or empty; existing results are preserved')
    if args.out==args.image.parent:
        p.error('Use a dedicated output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.background:
        cmd=[sys.executable,str(Path(__file__).resolve()),'analyze',str(args.image),'--out',str(args.out),
             '--mode',args.mode,'--quality',args.quality,'--display',args.display,'--anchor',*[str(v) for v in args.anchor],
             '--normal-deg',str(args.normal_deg),'--ior-from',str(args.ior_from),'--ior',str(args.ior)]
        if args.context:
            cmd.extend(['--context',str(args.context.expanduser().resolve(strict=True))])
        # Reserve a fresh output directory, with logs outside it until the worker starts.
        log=open(args.out.parent/(args.out.name+'.log'),'ab',buffering=0)
        detach={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS} if os.name=='nt' else {'start_new_session':True}
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,**detach,
                              env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
        log.close()
        print(json.dumps({'status':'started','pid':proc.pid,'output_dir':str(args.out),'log':str(args.out.parent/(args.out.name+'.log')),'status_path':str(args.out/'status.json')},ensure_ascii=False))
        return 0
    return analyze(args)


if __name__=='__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):stream.reconfigure(encoding='utf-8')
    raise SystemExit(main())
