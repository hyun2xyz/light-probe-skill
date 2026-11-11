"""Color scene with translucent object silhouettes and explanatory light transport."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

INK = '#151515'
WHITE = '#ffffff'
MASK_PALETTE = [(164,181,183), (186,169,156), (164,180,157),
                (184,161,170), (170,166,190), (185,179,149)]


def visible_masks(objects, size):
    """Rasterize supplied silhouettes, back to front; never invent box masks."""
    w,h=size
    scale=2
    cover=Image.new('L',(w*scale,h*scale))
    masks={}
    for obj in reversed(objects):
        polygon=obj.get('polygon')
        if not polygon:continue
        mask=Image.new('L',cover.size)
        points=[(round(x*(w*scale-1)),round(y*(h*scale-1))) for x,y in polygon]
        ImageDraw.Draw(mask).polygon(points,fill=255)
        visible=ImageChops.subtract(mask,cover)
        masks[obj['id']]=visible.resize(size,Image.Resampling.LANCZOS)
        cover=ImageChops.lighter(mask,cover)
    return masks


def object_masks(scene, objects):
    masks=visible_masks(objects,scene.size)
    metadata=[]
    for index,obj in enumerate(objects):
        mask=masks.get(obj['id'])
        if mask is None:continue
        color=tuple(obj.get('color',MASK_PALETTE[index%len(MASK_PALETTE)]))
        opacity=obj.get('opacity',.18)
        tint=Image.new('RGB',scene.size,color)
        alpha=mask.point(lambda v:round(v*opacity))
        scene=Image.composite(tint,scene,alpha)
        # Fine inner contour follows only the visible silhouette, not its bbox.
        edge=ImageChops.subtract(mask,mask.filter(ImageFilter.MinFilter(3)))
        scene=Image.composite(tint,scene,edge.point(lambda v:round(v*.72)))
        metadata.append({'object_id':obj['id'],'source':obj.get('source','provided'),
                         'geometry':'provided_polygon','opacity':opacity,'color':list(color)})
    return scene,metadata


def face(size, role='serif'):
    choices = {
        'serif': ['/System/Library/Fonts/Supplemental/AppleMyungjo.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'],
        'mono': ['/System/Library/Fonts/Menlo.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'],
        'korean': ['/System/Library/Fonts/AppleSDGothicNeo.ttc',
                   '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'],
    }
    bundled = Path(__file__).resolve().parents[1]/'assets/fonts/NanumGothic-Regular.ttf'
    choices[role].append(str(bundled))
    for path in choices[role]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def segment(draw, a, b, width=2, dash=None, halo=True):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b-a))
    if length < .01:
        return
    if dash:
        d=(b-a)/length
        for start in np.arange(0, length, sum(dash)):
            end=min(start+dash[0],length)
            segment(draw,a+d*start,a+d*end,width,halo=halo)
        return
    pts=[tuple(a),tuple(b)]
    if halo:
        draw.line(pts, fill=WHITE, width=width+4)
    draw.line(pts, fill=INK, width=width)


def ray(draw, a, b, width=2, dash=None):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    length=float(np.linalg.norm(b-a))
    if length<3:
        return
    d=(b-a)/length
    p=np.array([-d[1],d[0]])
    segment(draw,a,b,width,dash)
    head=min(15,length*.22)
    pts=[tuple(b),tuple(b-d*head+p*head*.4),tuple(b-d*head-p*head*.4)]
    draw.line([pts[1],pts[0],pts[2]],fill=WHITE,width=width+5)
    draw.polygon(pts,fill=INK)


def sun(draw, point, primary=True):
    x,y=point
    r=12 if primary else 6
    outer=24 if primary else 13
    draw.ellipse((x-r-4,y-r-4,x+r+4,y+r+4),fill=WHITE)
    draw.ellipse((x-r,y-r,x+r,y+r),fill=WHITE,outline=INK,width=2)
    for angle in np.linspace(0,2*math.pi,8,endpoint=False):
        v=np.array([math.cos(angle),math.sin(angle)])
        segment(draw,point+v*(r+5),point+v*outer,2)
    if primary:
        draw.ellipse((x-3,y-3,x+3,y+3),fill=INK)


def tag(draw, point, text, width, height, size=17, side='right'):
    f=face(size,'korean' if any(ord(c)>255 for c in text) else 'mono')
    box=draw.textbbox((0,0),text,font=f)
    tw=box[2]-box[0]
    th=box[3]-box[1]
    x=point[0]+30 if side=='right' else point[0]-tw-35
    x=max(8,min(x,width-tw-20))
    y=max(8,min(point[1]-th-19,height-th-16))
    draw.rectangle((x-7,y-5,x+tw+7,y+th+6),fill=WHITE)
    draw.text((x,y-box[1]),text,font=f,fill=INK)


def source_marker(origin, source, width, height):
    """Extend a direction marker along the same ray, with comfortable edge clearance."""
    origin=np.asarray(origin,dtype=float)
    source=np.asarray(source,dtype=float)
    delta=source-origin
    inset=max(40,min(width,height)*.08)
    limit=1.5
    for axis,extent in enumerate((width,height)):
        if abs(delta[axis])<1e-9:
            continue
        edge=extent-inset if delta[axis]>0 else inset
        limit=min(limit,(edge-origin[axis])/delta[axis])
    # Preserve the angle; a marker already at the edge may need to move inward.
    return origin+delta*max(0.,limit)


def object_labels(draw, objects, width, height):
    for obj in objects:
        x0,y0,x1,y1=np.array(obj['bbox'])*np.array([width,height,width,height])
        rho=obj['reflectance']
        lines=[f"{obj['label']} / {obj['material']}",f"반사율 {rho[0]*100:.0f}–{rho[1]*100:.0f}%"]
        f=face(18,'korean')
        tw=max(f.getlength(line) for line in lines)+18
        th=53
        corner=obj.get('label_position','top-left')
        x=x1-tw-8 if corner.endswith('right') else x0+8
        y=y1-th-8 if corner.startswith('bottom') else y0+8
        x=max(8,min(x,width-tw-8));y=max(8,min(y,height-th-8))
        draw.rectangle((x,y,x+tw,y+th),fill=WHITE)
        for i,line in enumerate(lines):draw.text((x+8,y+5+i*23),line,font=f,fill=INK)


def scattering_overlay(draw, transport, objects, width, height):
    # Show only a legible subset; the complete bounded energy ledger stays in JSON.
    segments=[]
    for bounce in (1,2,3):
        candidates=[s for s in transport['segments'] if s['bounce']==bounce]
        segments.extend(sorted(candidates,key=lambda s:(not s['hit'],-s['energy']))[:6])
    transport['visualization']={'selected_segments':len(segments),'unhit_rays_shortened':True}
    for s in sorted(segments,key=lambda s:s['energy']):
        if s['energy']<.003:continue
        dash=(10,6) if s['kind']=='specular' else (2,7)
        a,b=np.array(s['a']),np.array(s['b'])
        if not s['hit']:
            delta=b-a;length=np.linalg.norm(delta)
            if length>0:b=a+delta*min(1,min(width,height)*.20/length)
        ray(draw,a,b,1 if s['energy']<.08 else 2,dash)
    labels={obj['id']:obj['label'] for obj in objects}
    # One label per collision depth avoids a label on every diffuse sample.
    shown=set()
    # Depth-first traversal gives a consistent B1 -> B2 -> B3 example path.
    for hit in transport['hits']:
        p=np.array(hit['point']);x,y=p
        draw.ellipse((x-4,y-4,x+4,y+4),fill=WHITE,outline=INK,width=1)
        if hit['bounce'] in shown:continue
        shown.add(hit['bounce'])
        text=f"B{hit['bounce']} / {labels.get(hit['object_id'],hit['object_id'])}"
        if hit['bounce']==1:text+=' · 산란 2D'
        tag(draw,p+[0,40],text,width,height,size=15,side='left' if x>.7*width else 'right')


def render(im, result, args, optics_fn):
    target_w=min(1200,max(800,im.width))
    target_h=round(im.height*target_w/im.width)
    scene=im.resize((target_w,target_h),Image.Resampling.LANCZOS).convert('RGB')
    w,h=scene.size
    context=result.get('context') or {}
    objects=context.get('objects',[])
    scene,mask_info=object_masks(scene,objects)
    result['object_overlay']={'style':'muted-silhouettes','masks':mask_info,
        'unmasked_object_ids':[o['id'] for o in objects if not o.get('polygon')],
        'occlusion_order':'back-to-front'}
    draw=ImageDraw.Draw(scene)
    if args.display=='heatmap':
        # Brightness contours retain the source photograph's colors.
        import cv2
        values=np.asarray(scene.convert('L').resize((256,256)))
        for level in (64,128,192):
            contours,_=cv2.findContours((values>level).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c)<150:continue
                pts=[(float(p[0][0])*w/256,float(p[0][1])*h/256) for p in c]
                if len(pts)>2:
                    draw.line(pts+[pts[0]],fill=WHITE,width=3)
                    draw.line(pts+[pts[0]],fill=INK,width=1)
    scatter_settings=context.get('scattering')
    origin=np.array([args.anchor[0]*w,args.anchor[1]*h])
    lights=result.get('lights',[])
    points=[]
    for index,light in enumerate(lights):
        source=np.array([float(light['x'])*w,float(light['y'])*h])
        # Visible glyph stays inside the frame; ray geometry still uses source.
        marker=np.array([np.clip(source[0],30,w-30),np.clip(source[1],30,h-30)])
        if index==0 and result['mode']=='diffusion':
            marker=source_marker(origin,source,w,h)
        points.append((source,marker))
        if scatter_settings and index>0:continue
        if args.display in ('direction','all','heatmap') or index==0:
            ray(draw,marker if index==0 else source,origin,width=3 if index==0 else 1,dash=None if index==0 else (4,7))
    ray_info=None
    if args.display in ('reflection','refraction','all') and lights and not scatter_settings:
        incoming=origin-points[0][0]
        if np.linalg.norm(incoming)>1:
            angle=math.radians(args.normal_deg)
            ray_info=optics_fn(incoming,[math.cos(angle),math.sin(angle)],args.ior_from,args.ior)
            normal=np.array(ray_info['normal'])
            tangent=np.array([-normal[1],normal[0]])
            radius=min(w,h)*.24
            segment(draw,origin-tangent*radius,origin+tangent*radius,1)
            ray(draw,origin,origin+normal*radius*.7,1,(2,7))
            tag(draw,origin+normal*radius*.7,'N',w,h,size=14)
            if args.display in ('reflection','all') or ray_info['total_internal_reflection']:
                end=origin+np.array(ray_info['reflected'])*radius
                ray(draw,origin,end,2,(12,7))
                tag(draw,end,'R / TIR' if ray_info['total_internal_reflection'] else 'R',w,h,size=15,side='left')
            if args.display in ('refraction','all') and ray_info['refracted'] is not None:
                end=origin+np.array(ray_info['refracted'])*radius
                ray(draw,origin,end,2,(2,6))
                tag(draw,end,'T',w,h,size=15,side='left')
    result.pop('scattering',None)
    if scatter_settings and lights:
        from scattering import trace_scattering
        transport=trace_scattering(objects,context['surfaces'],origin.tolist(),(origin-points[0][0]).tolist(),
            w,h,scatter_settings['initial_surface_id'],max_bounces=scatter_settings.get('max_bounces',3))
        result['scattering']=transport
        scattering_overlay(draw,transport,objects,w,h)
    for index,(_,marker) in enumerate(points):
        if scatter_settings and index>0:continue
        if index==0:
            sun(draw,marker)
            label='S1 / LIGHT IN'
        else:
            x,y=marker
            draw.ellipse((x-5,y-5,x+5,y+5),fill=WHITE,outline=INK,width=2)
            label=f'L{index+1}'
        tag(draw,marker,label,w,h,size=16,side='left' if marker[0]>.72*w else 'right')
    x,y=origin
    draw.ellipse((x-8,y-8,x+8,y+8),fill=WHITE,outline=INK,width=2)
    draw.ellipse((x-3,y-3,x+3,y+3),fill=INK)
    if not scatter_settings:tag(draw,origin+[4,40],'P / TARGET',w,h,size=14)
    object_labels(draw,objects,w,h)

    # Image-only figure: explanations and assumptions remain in result.json.
    margin=20
    sheet=Image.new('RGB',(w+margin*2,h+margin*2),WHITE)
    sheet.paste(scene,(margin,margin))
    ImageDraw.Draw(sheet).rectangle((margin,margin,margin+w-1,margin+h-1),outline=INK,width=1)
    return sheet,ray_info
