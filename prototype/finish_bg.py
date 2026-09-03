import sys, glob, subprocess, datetime, numpy as np
sys.path.insert(0,'.')
sys.path.insert(0,'/Users/michelheyaca/Desktop/clonecut')
from PIL import Image, ImageDraw
from rain import BG, rain_layer, W, H
from sing import font
FPS=24
TEXT="this is our song. it's called goodbye party"
CHAR_W, CHAR_TOP, TEXT_Y = 850, 480, 1690
fs=sorted(glob.glob('/tmp/seq_motion/rgba-*.png'))
x0=y0=10**9; x1=y1=-1
for f in fs[::4]:
    al=np.array(Image.open(f))[...,3]; ys,xs=np.where(al>16)
    x0,x1=min(x0,xs.min()),max(x1,xs.max()); y0,y1=min(y0,ys.min()),max(y1,ys.max())
cw,ch=x1-x0,y1-y0; scale=CHAR_W/cw
print(f'char bbox {cw}x{ch} scale {scale:.3f}')
fnt=font(44)
out=f'/Users/michelheyaca/Desktop/clonecut/out/doc-rain-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4'
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
    '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-ss','41.2','-t','9.0',
    '-i','/Users/michelheyaca/Desktop/clonecut/music/goodbyparty 1.4.wav',
    '-map','0:v','-map','1:a','-af','afade=t=out:st=8.5:d=0.5',
    '-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-shortest',out],
    stdin=subprocess.PIPE)
N=len(fs)
for i,f in enumerate(fs):
    # background drifts in slower than the character -> a little parallax
    z=1.0+0.018*(i/(N-1))
    bw,bh=int(W*z),int(H*z)
    bgf=BG.resize((bw,bh),Image.LANCZOS).crop(((bw-W)//2,(bh-H)//2,(bw-W)//2+W,(bh-H)//2+H)).convert('RGBA')
    bgf.alpha_composite(rain_layer(i))
    im=Image.open(f).crop((x0,y0,x1,y1)).resize((int(cw*scale),int(ch*scale)),Image.LANCZOS)
    bgf.alpha_composite(im,((W-im.width)//2,CHAR_TOP))
    canvas=bgf.convert('RGB')
    d=ImageDraw.Draw(canvas)
    tw=d.textlength(TEXT,font=fnt)
    d.text(((W-tw)/2,TEXT_Y),TEXT,font=fnt,fill=(235,235,235))
    p.stdin.write(canvas.tobytes())
p.stdin.close(); p.wait()
print('wrote',out)
