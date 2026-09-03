import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
W,H=1080,1920
BG=Image.open('bg_fit.png').convert('RGB')
_a=np.array(BG).astype(int); _lum=_a.mean(axis=2)
_m=np.zeros((H,W),bool); _m[103:1374, 99:981]=True
GLASS=(_lum>42)&_m
GLASS=ndimage.binary_opening(GLASS,iterations=2)
GLASS=ndimage.binary_erosion(GLASS,iterations=3)
GLASS_IMG=Image.fromarray((GLASS*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
rng=np.random.default_rng(7)
NS=130
sx=rng.uniform(99,981,NS); sy=rng.uniform(-600,1374,NS)
ln=rng.uniform(34,90,NS); sp=rng.uniform(30,58,NS)
op=rng.uniform(120,210,NS); wd=rng.choice([1,1,2],NS)
TILT=0.13
def rain_layer(f):
    lay=Image.new('L',(W,H),0); d=ImageDraw.Draw(lay)
    for i in range(NS):
        y=(sy[i]+sp[i]*f)%(1374+700)-600
        x=sx[i]+TILT*(y-sy[i])
        x=99+((x-99)%882)
        d.line([(x,y),(x-TILT*ln[i],y+ln[i])],fill=int(op[i]),width=int(wd[i]))
    lay=lay.filter(ImageFilter.GaussianBlur(0.6))
    lay=Image.fromarray((np.array(lay).astype(float)*np.array(GLASS_IMG).astype(float)/255).astype(np.uint8))
    col=Image.new('RGB',(W,H),(158,176,190))
    out=Image.new('RGBA',(W,H),(0,0,0,0))
    out.paste(col,(0,0),lay)
    return out
