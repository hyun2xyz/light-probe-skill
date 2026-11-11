import math
import os
import subprocess
import sys
import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image, ImageDraw
from light_probe import process_alive,optics,quick_estimate,analyze,annotate,read_context,render_saved
from paper_renderer import source_marker,object_masks,visible_masks


class PhysicsAndCandidates(unittest.TestCase):
    def test_liveness_does_not_signal_process(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)'])
        try:
            self.assertTrue(process_alive(child.pid))
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=10)
        self.assertFalse(process_alive(child.pid))
        self.assertTrue(process_alive(os.getpid()))

    def test_bright_region(self):
        image=Image.new('RGB',(400,300),(30,30,30))
        ImageDraw.Draw(image).ellipse((20,20,100,100),fill='white')
        r=quick_estimate(image)
        self.assertTrue(r['lights'])
        self.assertLess(abs(r['lights'][0]['x']-.15),.05)
        self.assertLess(abs(r['lights'][0]['y']-.20),.05)

    def test_uniform_abstains(self):
        for color in ['black','white',(128,128,128)]:
            self.assertEqual(quick_estimate(Image.new('RGB',(200,200),color))['lights'],[])

    def test_reflection(self):
        r=optics([1,1],[0,-1])
        np.testing.assert_allclose(r['reflected'],[2**-.5,-2**-.5],atol=1e-8)
        self.assertAlmostEqual(r['incidence_deg'],45)

    def test_snell(self):
        r=optics([.5,math.sqrt(.75)],[0,-1],1,1.5)
        self.assertAlmostEqual(math.sin(math.radians(r['transmission_deg'])),1/3)

    def test_total_internal_reflection(self):
        r=optics([math.sin(math.radians(60)),.5],[0,-1],1.5,1)
        self.assertTrue(r['total_internal_reflection'])
        self.assertIsNone(r['refracted'])

    def test_normal_incidence(self):
        r=optics([0,1],[0,-1],1,1.333)
        np.testing.assert_allclose(r['refracted'],[0,1],atol=1e-8)

    def test_corrupt_input_has_error_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'broken.png').write_bytes(b'not an image')
            args=argparse.Namespace(image=root/'broken.png',out=root/'output',mode='quick')
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analyze(args),1)
            self.assertEqual(json.loads((root/'output/status.json').read_text(encoding='utf-8'))['status'],'error')
            self.assertFalse((root/'output/annotated.png').exists())

    def test_every_display_preserves_source_color(self):
        im=Image.new('RGB',(120,80),(210,80,35))
        result={'mode':'quick','lights':[{'x':.85,'y':.2,'strength':1}]}
        for display in ['direction','reflection','refraction','all','heatmap']:
            args=argparse.Namespace(anchor=[.5,.65],display=display,normal_deg=-90,ior_from=1,ior=1.5)
            sheet,_=annotate(im,result,args)
            rgb=np.asarray(sheet)
            original_pixels=np.all(rgb==[210,80,35],axis=-1)
            self.assertGreater(float(original_pixels.mean()),.75)
        self.assertEqual(im.getpixel((0,0)),(210,80,35))

    def test_saved_render_preserves_inference_and_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'prior';source.mkdir()
            Image.new('RGB',(100,80),(70,90,130)).save(source/'input.png')
            data={'mode':'diffusion','quality':'fast','lights':[],'anchor':[.5,.5],
                  'display':'direction','elapsed_ms':47000,'ior_from':1,'ior_to':1.5,'normal_deg':-90}
            result=source/'result.json';result.write_text(json.dumps(data));before=result.read_bytes()
            context=root/'context.json';context.write_text(json.dumps({'scene':'창이 보이는 방','assumptions':['광원은 추정']}))
            args=argparse.Namespace(result=result,out=root/'new',context=context)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(render_saved(args),0)
            revised=json.loads((root/'new/result.json').read_text(encoding='utf-8'))
            self.assertEqual(revised['elapsed_ms'],47000)
            self.assertEqual(revised['context']['scene'],'창이 보이는 방')
            self.assertEqual(result.read_bytes(),before)
            with self.assertRaises(ValueError):render_saved(args)

    def test_context_is_bounded_data(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'context.json';p.write_text(json.dumps({'command':'not executable'}))
            with self.assertRaises(ValueError):read_context(p)

    def test_reflectance_cannot_exceed_one(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'context.json'
            obj={'id':'wall','label':'벽','material':'도장','bbox':[0,0,1,1],'reflectance':[.4,1.4]}
            p.write_text(json.dumps({'objects':[obj]}))
            with self.assertRaises(ValueError):read_context(p)

    def test_sun_moves_toward_source_without_changing_angle(self):
        origin=np.array([512.,665.6]);source=np.array([790.,430.])
        marker=source_marker(origin,source,1024,1024)
        self.assertGreater(marker[0],source[0])
        self.assertLess(marker[1],source[1])
        original=source-origin;extended=marker-origin
        self.assertAlmostEqual(original[0]*extended[1]-original[1]*extended[0],0.,places=6)
        self.assertLessEqual(marker[0],1024*.92)
        edge=source_marker(origin,[1024.,0.],1024,1024)
        self.assertTrue(np.all(edge>=1024*.08))
        self.assertTrue(np.all(edge<=1024*.92))

    def test_silhouette_tint_preserves_outside_and_source(self):
        original=Image.new('RGB',(100,100),(190,80,30))
        obj={'id':'triangle','polygon':[[.1,.1],[.9,.1],[.1,.9]],
             'color':[150,170,180],'opacity':.2}
        image,meta=object_masks(original,[obj])
        self.assertEqual(image.getpixel((80,80)),(190,80,30))
        np.testing.assert_allclose(image.getpixel((25,25)),[182,98,60],atol=1)
        self.assertEqual(original.getpixel((25,25)),(190,80,30))
        self.assertEqual(meta[0]['geometry'],'provided_polygon')

    def test_foreground_masks_do_not_stack_tints(self):
        back={'id':'back','polygon':[[0,0],[1,0],[1,1],[0,1]],'color':[180,160,150]}
        front={'id':'front','polygon':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'color':[150,170,180]}
        masks=visible_masks([back,front],(100,100))
        self.assertEqual(masks['back'].getpixel((50,50)),0)
        self.assertEqual(masks['front'].getpixel((50,50)),255)
        photo=Image.new('RGB',(100,100),(190,80,30))
        combined,_=object_masks(photo,[back,front])
        foreground,_=object_masks(photo,[front])
        self.assertEqual(combined.getpixel((50,50)),foreground.getpixel((50,50)))

    def test_missing_silhouette_never_becomes_rectangle_mask(self):
        photo=Image.new('RGB',(100,100),(190,80,30))
        revised,meta=object_masks(photo,[{'id':'unknown','bbox':[.1,.1,.9,.9]}])
        self.assertEqual(revised.tobytes(),photo.tobytes())
        self.assertEqual(meta,[])

    def test_invalid_mask_geometry_is_rejected(self):
        base={'id':'wall','label':'벽','material':'도장','bbox':[0,0,1,1],'reflectance':[.3,.6]}
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'context.json'
            for field,value in [('polygon',[[0,0],[1,1]]),
                                ('polygon',[[0,0],[.5,.5],[1,1]]),
                                ('polygon',[[0,0],[1,0],[float('nan'),1]]),
                                ('polygon',[[0,0],[1,0],[1,1.1]]),
                                ('color',[128,190,256]),('opacity',.8)]:
                with self.subTest(field=field,value=value):
                    p.write_text(json.dumps({'objects':[{**base,field:value}]}))
                    with self.assertRaises(ValueError):read_context(p)


if __name__=='__main__':unittest.main()
