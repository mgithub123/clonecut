import sys, os, shutil, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,'/Users/michelheyaca/Desktop/clonecut')
import lipsync
FPS=24; N=216; GATE=0.14; LOW=0.055
ORDER=['CLOSED','SMALL','OO','FV','EE','OH','AH','WIDE']; IX={n:i for i,n in enumerate(ORDER)}
VOX='/Users/michelheyaca/Desktop/LuckyDog-Harmony/goodbyparty 1.4 vox.wav'
HEAD=Image.open('/tmp/DocHEAD/frames/part-0001.tga').convert('RGBA')
BODY=Image.open('/tmp/DocBODY/frames/part-0001.tga').convert('RGBA')
from post import bg_mask
from scipy import ndimage as _nd
def unmul(im):
    """Art drawn on white -> straight RGBA. An edge pixel is a blend of the black
    outline and white, so alpha = 1 - luminance/255 and the white can be divided out."""
    rgb=np.array(im.convert('RGB')).astype(np.float64)
    bg=bg_mask(rgb.astype(int),736)
    near=_nd.binary_dilation(bg,iterations=3)&~bg
    lum=rgb.mean(axis=2)
    al=np.ones(lum.shape)
    al[near]=np.clip(1.0-lum[near]/255.0,0.0,1.0)
    al[bg]=0.0
    out=rgb.copy()
    m=near&(al>0.02)
    if m.any():
        aa=al[m][:,None]
        out[m]=np.clip((rgb[m]-255.0*(1.0-aa))/aa,0,255)
    a4=np.dstack([out.astype(np.uint8),(al*255).astype(np.uint8)])
    return Image.fromarray(a4)
HEAD=unmul(HEAD); BODY=unmul(BODY)

# ---- mouth track (unchanged from good8) ----
env=lipsync.envelope(VOX,41.2,9,FPS)
onsets=[0.24,0.59,0.8,0.99,1.18,1.38,1.56,1.91,2.3,2.62,2.84,3.05,3.27,3.45,3.64,3.82,4.03,4.39,4.69,4.97,5.22,5.4,5.84,6.04,6.38,6.63,6.81,7.05,7.53,7.8,8.3,8.69]
vseq="AH EE EE EE EE EE EE OH AH AH OO OH EE AH AH OO OO OH OH OH EE OH OO OH EE AH OO OO OH OH OH EE".split()
of=[int(round(t*FPS)) for t in onsets]
fv=['OH']*N
for i,s in enumerate(of):
    e=of[i+1] if i+1<len(of) else N
    for f in range(max(0,s),min(N,e)): fv[f]=vseq[i]
loud=np.where(env<GATE,0.0,(env-GATE)/(1-GATE))
def shape(f):
    v=fv[f]
    if v=='OO': return IX['OO']
    if v=='AH': return IX['WIDE'] if loud[f]>0.62 else IX['AH']
    if v=='EE': return IX['EE'] if loud[f]>0.35 else IX['FV']
    return IX['OH']
mouth=np.array([IX['CLOSED'] if env[f]<GATE else shape(f) for f in range(N)])
for i,s in enumerate(of):
    if s>=N or vseq[i]!='OO': continue
    nxt=of[i+1] if i+1<len(of) else N
    for f in range(s,min(s+8,nxt,N)):
        if env[f]>=LOW: mouth[f]=IX['OO']
        else: break
m2=mouth.copy(); i=1
while i<N:
    if m2[i]!=m2[i-1]: m2[i:i+3]=m2[i]; i+=3
    else: i+=1
mouth=np.concatenate([m2[2:],np.full(2,m2[-1])])

# ---- blinks / tears ----
EYES=[(1849,454,1902,508),(1941,454,1994,508)]
CURVE=[0.62,0.18,0.0,0.38,0.82]
blink=np.ones(N)
for s in (26,82,142,192):
    for k,v in enumerate(CURVE):
        if 0<=s+k<N: blink[s+k]=v
TEARS=[(30,1868,1857),(60,1972,1984),(105,1868,1857),(135,1972,1984),(175,1868,1857),(200,1972,1984)]
WELL,FALL,Y0,Y1=4,18,516,630
def tears_at(f):
    out=[]
    for (s,xt,xb) in TEARS:
        k=f-s
        if k<0 or k>=WELL+FALL: continue
        if k<WELL: p=0.0; r=0.35+0.65*(k+1)/WELL
        else:
            p=(k-WELL+1)/FALL; p=p*p*0.35+p*0.65; r=1.0
        out.append((xt+(xb-xt)*p, Y0+(Y1-Y0)*p, r))
    return out

# ---- motion ----
BEATS=[0.06,0.43,0.80,1.18,1.55,1.92,2.29,2.66,3.03,3.41,3.78,4.15,4.52,4.89,5.26,5.63,6.01,6.38,6.75,7.12,7.49,7.86,8.23,8.61,8.98]
PULSE=BEATS[::2]                       # half-time, every ~0.74s
def pulse_env(t, times, dur=0.42):
    v=0.0
    for b in times:
        d=t-b
        if 0<=d<dur:
            x=d/dur
            v=max(v, np.sin(np.pi*x)*(1-x)**0.6)
    return v
MOUTH_PLATE={}
def mouth_img(mi):
    if mi not in MOUTH_PLATE:
        nm=ORDER[mi]
        if nm=='CLOSED': MOUTH_PLATE[mi]=None
        else: MOUTH_PLATE[mi]=Image.open(f'/tmp/gm2/{nm}.png').convert('RGBA')
    return MOUTH_PLATE[mi]
FACE_W=127; CX,LIPY,MAXH=1920,568,74
PCT={'CLOSED':0.45,'SMALL':0.48,'OO':0.30,'FV':0.55,'EE':0.58,'OH':0.55,'AH':0.58,'WIDE':0.62}
CLOSED_IMG=Image.open('/tmp/gm2/CLOSED.png').convert('RGBA')
def head_with_face(f):
    h=HEAD.copy()
    nm=ORDER[int(mouth[f])]
    m=CLOSED_IMG if nm=='CLOSED' else mouth_img(int(mouth[f]))
    tw=int(FACE_W*PCT[nm]); s=min(tw/m.width, MAXH/m.height)
    w,hh=max(1,int(m.width*s)),max(1,int(m.height*s))
    h.paste(m.resize((w,hh),Image.LANCZOS),(CX-w//2,LIPY),m.resize((w,hh),Image.LANCZOS))
    bl=blink[f]
    if bl<0.999:
        a=np.array(h)
        for (x0,y0,x1,y1) in EYES:
            patch=a[y0:y1+1,x0:x1+1].copy(); ht=y1-y0+1
            nh=max(0,int(round(ht*bl)))
            a[y0:y1+1,x0:x1+1,:3]=0; a[y0:y1+1,x0:x1+1,3]=255
            if nh>0:
                sq=np.array(Image.fromarray(patch).resize((patch.shape[1],nh),Image.LANCZOS))
                t=y0+(ht-nh)//2; a[t:t+nh,x0:x1+1]=sq
        h=Image.fromarray(a)
    d=ImageDraw.Draw(h)
    for (x,y,r) in tears_at(f):
        w2,h2=11*r,15*r
        d.ellipse([x-w2/2,y-h2/2,x+w2/2,y+h2/2],fill=(191,217,236,255),outline=(0,0,0,255),width=max(1,int(2*r)))
    return h

PHRASE=[]
_i=0
while _i<N:
    if mouth[_i]==IX['CLOSED']:
        _j=_i
        while _j<N and mouth[_j]==IX['CLOSED']: _j+=1
        if _j-_i>=5: PHRASE.append(_i)
        _i=_j
    else: _i+=1
print('phrase ends at frames',PHRASE)
def settle(t):
    v=0.0
    for pf in PHRASE:
        d=t-pf/FPS
        if 0<=d<0.95:
            x=d/0.95
            v+=np.sin(np.pi*x)*(1-x)**0.5
    return v
CROP=(1590,100,2250,800)
OUT='/tmp/seq_motion'; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
HEAD_PIVOT=(1920,640)
for f in range(N):
    t=f/FPS
    p=pulse_env(t,PULSE)
    breath=np.sin(2*np.pi*t/4.2)
    body_dy = 2.2*p + 1.6*breath
    head_dy = 4.0*pulse_env(t-0.05,PULSE) + 1.6*breath + 1.1*np.sin(2*np.pi*t/3.1+0.7)
    head_dx = 1.8*np.sin(2*np.pi*t/5.3)
    head_rot= 1.25*np.sin(2*np.pi*t/6.1+1.2) - 0.9*p + 1.7*settle(t)
    head_dy = head_dy + 3.2*settle(t)
    canvas=Image.new('RGBA',(3840,2160),(0,0,0,0))
    b=BODY.copy()
    canvas.alpha_composite(b,(0,int(round(body_dy))))
    h=head_with_face(f)
    h=h.rotate(head_rot,resample=Image.BICUBIC,center=HEAD_PIVOT)
    canvas.alpha_composite(h,(int(round(head_dx)),int(round(head_dy))))

    zoom=1.0+0.045*(f/(N-1))
    if zoom!=1.0:
        cx,cy=1920,600
        w,hh=canvas.size
        canvas=canvas.resize((int(w*zoom),int(hh*zoom)),Image.LANCZOS)
        ox,oy=int(cx*zoom-cx),int(cy*zoom-cy)
        canvas=canvas.crop((ox,oy,ox+w,oy+hh))
    canvas.crop(CROP).save(OUT+'/rgba-%04d.png'%(f+1))
print('frames written to',OUT)
