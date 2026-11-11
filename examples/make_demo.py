"""Make a synthetic teaching scene; no third-party photograph is bundled."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('Use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    w,h=900,700
    yy,xx=np.mgrid[0:h,0:w]
    base=np.array([56,72,71])+np.clip(1-np.hypot((xx-790)/1000,(yy-100)/900),0,1)[...,None]*65
    im=Image.fromarray(np.uint8(base));d=ImageDraw.Draw(im)
    d.polygon([(0,465),(900,465),(900,700),(0,700)],fill=(88,76,67))
    d.rectangle((650,50,847,403),fill=(208,209,194))
    d.rectangle((663,64,834,389),fill=(248,242,216))
    d.line([(749,64),(749,389)],fill=(132,140,132),width=6)
    d.line([(663,228),(834,228)],fill=(132,140,132),width=6)
    d.polygon([(650,403),(834,389),(529,680),(190,700)],fill=(142,126,99))
    table=[(.17,.68),(.76,.60),(.94,.83),(.29,.92)]
    d.polygon([(int(x*w),int(y*h)) for x,y in table],fill=(160,114,84))
    d.polygon([(261,644),(288,640),(288,699),(270,699)],fill=(76,62,51))
    d.polygon([(825,580),(846,580),(838,699),(821,699)],fill=(76,62,51))
    d.ellipse((340,377,450,483),fill=(116,139,148))
    d.ellipse((340,377,450,404),fill=(163,179,180))
    im.save(args.out/'room.png')
    context={
      'scene':'직접 그린 합성 실내 장면. 창문·벽·테이블로 구성한 수업용 도식.',
      'target':'테이블 중앙',
      'light_interpretation':'오른쪽 밝은 창이 광원 후보. quick의 밝기 기반 탐색을 확인하는 합성 예제.',
      'assumptions':['합성 장면이며 사진 모델 정확도 평가가 아님','반사율은 설명용 입력','2D 선분에서만 산란 계산'],
      'objects':[
       {'id':'wall','label':'벽','material':'무광 도장','bbox':[.02,.025,.65,.65],'polygon':[[0,0],[1,0],[1,.665],[0,.665]],'reflectance':[.3,.6],'color':[150,177,177],'label_position':'top-left','source':'synthetic'},
       {'id':'window','label':'창','material':'유리','bbox':[650/w,50/h,847/w,403/h],'polygon':[[650/w,50/h],[847/w,50/h],[847/w,403/h],[650/w,403/h]],'reflectance':[.05,.1],'color':[185,180,154],'label_position':'top-left','source':'synthetic'},
       {'id':'table','label':'테이블','material':'목재','bbox':[.17,.60,.94,.92],'polygon':table,'reflectance':[.25,.45],'color':[178,160,178],'label_position':'bottom-left','source':'synthetic'},
      ],
      'surfaces':[
       {'id':'table_plane','object_id':'table','a':[.2,.65],'b':[.9,.65]},
       {'id':'wall_plane','object_id':'wall','a':[.15,.18],'b':[.15,.60]},
       {'id':'window_plane','object_id':'window','a':[.72,.12],'b':[.72,.58]}],
      'scattering':{'initial_surface_id':'table_plane','max_bounces':3}}
    (args.out/'context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(args.out/'room.png')


if __name__=='__main__':main()
