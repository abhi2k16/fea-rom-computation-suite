"""Figure for Section 7.5 (several cores): schematic of how jobs land on cores, the two-core measurements of
Table 7.6, and the many-core scaling and job-size sweeps. Reads experiments/*.jsonl, writes
figures/ch06_multicore.pdf and tables/numbers_ch06_cores.tex. Only up to 8 of the 12 logical CPUs are used, so
that the machine stays available for other work."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
E=os.path.join(ROOT,'experiments')+os.sep
L=lambda f:[json.loads(l) for l in open(E+f)][1:]
R=[json.loads(l) for l in open(E+'multicore.jsonl')]
def g(case,nj,be,bl): return [r for r in R if r['case']==case and r['n_jobs']==nj and r['backend']==be and r['blas']==bl][0]['time']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; RED='#c0392b'; GRN='#2e8b57'; GREY='#8a96a3'; J='#9cc3e6'; J2='#6fa8dc'; ORG='#f4b183'
fig=plt.figure(figsize=(6.2,5.8))
# ---------- (a) schematic ----------
H=L('multicore_scaling_heavy.jsonl'); hv={r['n_jobs']:r['time'] for r in H if r['case']=='heavy'}
fig_h=8.4
ax=fig.add_axes([0.0,0.55,1.0,0.45]); ax.set_xlim(0,14); ax.set_ylim(4.6,16.6); ax.axis('off')
ax.text(0.1,16.3,'(a) How the jobs land on the cores',fontweight='bold',fontsize=8.2,color=NAVY,va='center')
LH=0.42; SC=0.60
def blk(x,y,w,c,h=None): ax.add_patch(Rectangle((x,y),w,LH-0.05,fc=c,ec='white' if h is None else '#999',lw=0.5,hatch=h))
OTHER='#d9dde2'
def lanes_group(x0,ytop,title,n,speed_note,fin,jobs=8,startup=0.0,tag=''):
    ax.text(x0,ytop+0.35,title,fontsize=7.2,fontweight='bold',va='center',color='#222')
    per=jobs//n
    for k in range(n):
        y=ytop-(k+1)*LH
        ax.text(x0+0.85,y+LH/2-0.02,'core %d'%(k+1) if n<=4 or k in (0,n-1) else ('\u22ee' if k==1 else ''),fontsize=5.8,ha='right',va='center',color='#666')
        x=0.0
        if startup>0: blk(x0+1.0,y,startup*SC,GREY); x=startup
        for i in range(per):
            blk(x0+1.0+x*SC,y,(1-0.05)*SC,J if (i+k)%2==0 else J2); x+=1
        if fin>x: blk(x0+1.0+x*SC,y,(fin-x)*SC,OTHER,'////')
    xe=x0+1.0+fin*SC; ax.plot([xe,xe],[ytop-n*LH+0.02,ytop-0.02],color='#444',lw=0.8,ls=':')
    ax.text(xe+0.15,ytop-n*LH/2,speed_note,fontsize=6.4,va='center',color='#333')
sp=lambda n:hv[1]/hv[n]
y=15.2
lanes_group(0.1,y,'serial loop (1 core)',1,'speed-up 1.00',8.0); y-=1.0+0.1
lanes_group(0.1,y,'2 processes',2,'%.2f'%sp(2),8/sp(2),startup=0.25); y-=2*LH+0.7
lanes_group(0.1,y,'4 processes',4,'%.2f'%sp(4),8/sp(4),startup=0.35); y-=4*LH+0.7
lanes_group(0.1,y,'8 processes',8,'%.2f'%sp(8),8/sp(8),startup=0.5); y-=8*LH+0.7
ax.text(0.1,y+0.35,'Time to scale, 12-CPU system, %.1f s solves.'%(hv[1]/48),fontsize=6.2,color='#555',va='center')
# right column: threads and BLAS on the two-core system
l1=[(0,.45,RED),(.45,.5,GRN),(.95,.45,RED),(1.4,.5,GRN)]; l2=[(.45,.45,RED),(.9,.5,GRN),(1.4,.45,RED),(1.85,.5,GRN)]
def small(x0,ytop,title,lns,note,tot):
    ax.text(x0,ytop+0.35,title,fontsize=7.2,fontweight='bold',va='center',color='#222')
    for k,lane in enumerate(lns):
        yy=ytop-(k+1)*LH*1.2
        ax.text(x0+0.85,yy+LH/2-0.02,'core %d'%(k+1),fontsize=5.8,ha='right',va='center',color='#666')
        for a,w,c in lane: ax.add_patch(Rectangle((x0+1.0+a*1.3,yy),w*1.3,LH*1.1-0.05,fc=c,ec='white',lw=0.5))
    ax.plot([x0+1.0+tot*1.3]*2,[ytop-2.4*LH-0.05,ytop-0.05],color='#444',lw=0.8,ls=':')
    ax.text(x0+1.0,ytop-2.4*LH-0.35,note,fontsize=6.4,va='top',color='#333',linespacing=1.2)
tk=g('plate',1,'process','1')/g('plate',2,'thread','1')
ob=g('dense',1,'process','1')/g('dense',2,'process','2'); o1=g('dense',1,'process','1')/g('dense',2,'process','1')
small(7.4,15.2,'2 threads (two-core system)',[l1,l2],'speed-up %.2f: the interpreter lock (GIL)\nserialises the Python code (red)'%tk,2.35)
ORG2='#f4b183'
lb=[(0,.2,GREY)]+[(0.2+i*0.38,0.38,J if i%2==0 else ORG2) for i in range(8)]
small(7.4,11.6,'2 processes \u00d7 2 BLAS threads',[lb,lb],'speed-up %.2f, against %.2f with one thread\neach: more threads than cores (oversubscription)'%(ob,o1),3.24)
ents=((J,'job'),(GREY,'pool start-up'),(OTHER,'start-up and contention (remainder)'),(RED,'Python code, one thread at a time'),(GRN,'compiled library call'),(ORG2,'second BLAS thread'))
for k,(c,t) in enumerate(ents):
    yy=8.3-k*0.55
    ax.add_patch(Rectangle((7.4,yy-0.15),0.4,0.3,fc=c,ec='#999',lw=0.4,hatch='////' if c==OTHER else None)); ax.text(7.95,yy,t,fontsize=6.3,va='center')
tiny0={r['n_jobs']:r['time'] for r in H if r['case']=='tiny'}
# ---------- (b) two-core measured ----------
a2=fig.add_axes([0.175,0.085,0.31,0.40])
items=[('2 processes',g('plate',1,'process','1')/g('plate',2,'process','1')),('4 processes',g('plate',1,'process','1')/g('plate',4,'process','1')),('2 threads',g('plate',1,'process','1')/g('plate',2,'thread','1')),
 ('2 BLAS threads',g('dense',1,'process','1')/g('dense',1,'process','2')),('2 processes',g('dense',1,'process','1')/g('dense',2,'process','1')),('2 proc. + 2 BLAS',g('dense',1,'process','1')/g('dense',2,'process','2')),
 ('2 processes',g('tiny',1,'process','1')/g('tiny',2,'process','1')),('2 threads',g('tiny',1,'process','1')/g('tiny',2,'thread','1'))]
xs=[0,1,2,4,5,6,8,9]
a2.barh(xs,[v-1 for _,v in items],left=1,color=[GRN if v>1 else RED for _,v in items],height=0.7)
a2.set_xscale('log'); a2.set_xlim(0.003,6); a2.set_ylim(9.7,-1.9)
a2.axvline(1,color='#333',lw=1); a2.axvline(2,color=GREY,lw=0.9,ls='--')
a2.text(2.1,-1.35,'ideal (2 cores)',fontsize=6.2,color=GREY,va='center',ha='left')
for x,(l,v) in zip(xs,items): a2.text(max(v,1)*1.15,x,'%.2f'%v if v>=0.1 else '%.3f'%v,va='center',ha='left',fontsize=6.3)
a2.set_yticks(xs); a2.set_yticklabels([i[0] for i in items],fontsize=6.2)
for y,t in ((-0.95,'plate model, 12 solves'),(3.05,'dense solve, 12 solves'),(7.05,'trivial jobs, 400 per run')):
    a2.text(0.0032,y,t,fontsize=6.4,color='#444',va='center',ha='left',fontweight='bold')
a2.set_xlabel('speed-up on 2 cores (log scale)',fontsize=7); a2.tick_params(axis='x',labelsize=6.3)
a2.set_title('(b) Two-core system (Table 7.6)',fontweight='bold',fontsize=8,loc='left',color=NAVY)
for s in ('top','right'): a2.spines[s].set_visible(False)
a2.grid(axis='x',alpha=0.25,which='both',lw=0.4)
# ---------- (c) job size ----------
D={}
for r in L('multicore_jobsize.jsonl')+L('multicore_jobsize_large.jsonl'): D[(r['n_elements'],r['n_jobs'])]=r['time']/r['n_items']
nes=sorted({k[0] for k in D}); tj=[D[n,1] for n in nes]; hv_t=D[20736,1]
a3=fig.add_axes([0.62,0.085,0.365,0.40])
a3.axhspan(0,1,color=RED,alpha=0.07,lw=0); a3.axhspan(1,4,color=GRN,alpha=0.05,lw=0)
a3.axhline(1,color='#333',lw=1)
for nw,c,m,off in ((4,J2,'o',-0.22),(8,GRN,'s',0.12)):
    sp=[D[n,1]/D[n,nw] for n in nes]
    a3.plot(tj,sp,m+'-',color=c,ms=3.5,lw=1.3)
    a3.text(tj[-1]*1.18,sp[-1],'%d workers'%nw,fontsize=6.4,color=c,va='center',ha='left')
for nw in (2,3,5,6,7): a3.plot([hv_t],[hv[1]/hv[nw]],'o',color=GREY,ms=2.6,mfc='white',mew=0.8,zorder=1)
a3.text(hv_t*1.0,0.95*0+1.2,'',fontsize=1)
a3.annotate('2\u20137 workers\n(one size)',xy=(hv_t,1.97),xytext=(0.04,3.0),fontsize=6.0,color='#555',arrowprops=dict(arrowstyle='-',color='#aaa',lw=0.6),va='center')
for n,t in zip(nes,tj):
    if n in (576,5184,20736,57600): a3.text(t,0.06,'%.1fk'%(n/1000),fontsize=5.6,color='#555',ha='center',va='bottom',rotation=0)
a3.text(0.5,0.62,'pool slower than serial loop',fontsize=6.4,color=RED,va='center',ha='left')
a3.text(0.5,0.38,'start-up: %.0f s (2 workers) to %.0f s (8)'%(tiny0[2],tiny0[8]),fontsize=6.0,color='#555',va='center',ha='left')
a3.text(0.3,3.55,'pool pays off',fontsize=6.4,color=GRN,va='center')
a3.axhspan(2.3,2.7,xmin=0.45,color=GREY,alpha=0.18,lw=0)
a3.text(tj[-1]*1.0,2.05,'plateau ≈ 2.5',fontsize=6.2,color='#555',ha='center',va='top')
a3.set_xscale('log'); a3.set_xlim(0.025,40); a3.set_xticks([0.1,1,10]); a3.set_xticklabels(['0.1','1','10']); a3.set_ylim(0,3.9)
a3.set_xlabel('time per solve on one worker (s); k = elements',fontsize=6.3); a3.set_ylabel('speed-up over serial loop',fontsize=7)
a3.tick_params(labelsize=6.3)
a3.set_title('(c) Many-core system',fontweight='bold',fontsize=8,loc='left',color=NAVY)
a3.grid(alpha=0.25,which='both',lw=0.4)
for s in ('top','right'): a3.spines[s].set_visible(False)

fig.savefig(os.path.join(ROOT,'figures','ch06_multicore.pdf'),bbox_inches='tight',pad_inches=0.03)
# ---- macros ----
heavy_ms=1000*hv[1]/48
tiny={r['n_jobs']:r['time'] for r in H if r['case']=='tiny'}
big=[n for n in nes if n>=30000]
plateau=np.mean([D[n,1]/D[n,8] for n in big])
def cross(nw):
    pairs=list(zip(nes,[D[n,1]/D[n,nw] for n in nes]))
    for (n0,s0),(n1,s1) in zip(pairs,pairs[1:]):
        if s0<1<=s1: return 1000*D[n0,1],1000*D[n1,1]
lo4,hi4=cross(4); lo8,hi8=cross(8)
m={'cmCpus':'%d'%12,'cmMaxWorkers':'8','cmSpdTwo':'%.1f'%(hv[1]/hv[2]),'cmSpdFour':'%.1f'%(hv[1]/hv[4]),'cmSpdEight':'%.1f'%(hv[1]/hv[8]),
 'cmPlateau':'%.1f'%plateau,'cmBreakLo':'%.0f'%max(lo4,lo8),'cmBreakHi':'%.0f'%min(hi4,hi8),
 'cmStartTwo':'%.0f'%tiny[2],'cmStartEight':'%.0f'%tiny[8],'cmHeavySolve':'%.1f'%(heavy_ms/1000),
 'cmBigElems':'{:,}'.format(max(nes)).replace(',','\\,'),'cmBigSolve':'%.1f'%D[max(nes),1],'cmBaseVar':'%.0f'%(100*(D[20736,1]*48/hv[1]-1))}
open(os.path.join(ROOT,'tables','numbers_ch06_cores.tex'),'w').write("".join("\\newcommand{\\%s}{%s}\n"%(k,v) for k,v in m.items()))
print(m)
