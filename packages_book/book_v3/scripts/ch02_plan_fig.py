import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
HERE=os.path.dirname(os.path.abspath(__file__))
src=os.path.join(HERE,'..','..','..','fea_engine','src')
if os.path.isdir(src): sys.path.insert(0,src)
from fea_engine import grading as G
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; ORG='#d9822b'; GREY='#8a96a3'
def title(ax,t): ax.set_title(t,fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
fig=plt.figure(figsize=(6.8,5.3))
gs=fig.add_gridspec(2,3,height_ratios=[1.15,1],hspace=0.45,wspace=0.12,left=0.08,right=0.99,top=0.93,bottom=0.05)
# (a) exponent against target
ax=fig.add_subplot(gs[0,0:2]); tg=np.linspace(1.0,2.1,200)
ax.axvspan(1.0,1.2,color=GRN,alpha=.10); ax.axvspan(1.2,1.5,color=ORG,alpha=.10); ax.axvspan(1.5,2.1,color=RED,alpha=.08)
ax.plot(tg,np.log2(1+tg),color=GREY,lw=3.5,alpha=.5,label=r'closed form $p=\log_2(1+\rho^\ast)$')
for n,m,c in zip((6,12,20),('o','s','^'),(GRN,BLUE,ORG)):
    t=np.array([1.05,1.1,1.2,1.3,1.5,1.8,2.0]); ax.plot(t,[G.grade_for_growth_ratio(n,x) for x in t],m,color=c,ms=4.5,mfc='none',label=f'package, $n={n}$')
ax.axvline(1.2,color='#555',ls=':',lw=1); ax.axvline(1.5,color=RED,ls=':',lw=1)
ax.text(1.1,2.03,'target 1.2\nused here',color='#555',fontsize=7,ha='center',va='top'); ax.text(1.35,2.03,'warns\nabove 1.5',color=RED,fontsize=7,ha='center',va='top')
ax.text(1.1,0.97,'gentle',color=GRN,fontsize=7.5,ha='center',va='bottom'); ax.text(1.35,0.97,'caution',color=ORG,fontsize=7.5,ha='center',va='bottom'); ax.text(1.8,0.97,'abrupt size change',color=RED,fontsize=7.5,ha='center',va='bottom')
ax.set_xlim(1.0,2.1); ax.set_ylim(0.95,2.15); ax.set_xlabel(r'target growth ratio $\rho^\ast$'); ax.set_ylabel('exponent $p$ returned')
ax.legend(frameon=False,fontsize=7,loc='lower right',bbox_to_anchor=(1.0,0.12)); title(ax,'(a) The exponent depends on the target, not on $n$')
for s in ['top','right']: ax.spines[s].set_visible(False)
# (b) rings
R,Ro,n=0.35,1.0,12
cases=[('uniform',1.0,GREY),('target 1.2',G.grade_for_growth_ratio(n,1.2),GRN),('target 2.0',G.grade_for_growth_ratio(n,2.0),RED)]
ax=fig.add_subplot(gs[0,2]); ax.set_aspect('equal'); ax.axis('off'); title(ax,'(b) Rings around a hole')
for k,(lab,p,c) in enumerate(cases):
    ox=0.0; oy=-k*1.08
    x=G.graded_partition(0.0,1.0,n,p); r=R+(Ro-R)*x
    for ri in r: 
        th=np.linspace(0,np.pi/2,60); ax.plot(ox+ri*np.cos(th),oy+ri*np.sin(th),color=c,lw=.8)
    ax.plot([ox+R,ox+Ro],[oy,oy],color=c,lw=.8); ax.plot([ox,ox],[oy+R,oy+Ro],color=c,lw=.8)
    ax.add_patch(Wedge((ox,oy),R,0,90,fc='#ddd',ec='none'))
    ax.text(ox+Ro+0.06,oy+0.5,lab+f'\n$p={p:.2f}$',fontsize=7.5,color=c,va='center')
ax.set_xlim(-0.05,1.9); ax.set_ylim(-2.25,1.1)
# (c) strips
ax=fig.add_subplot(gs[1,:]); title(ax,'(c) The partitions of a 12-element ring: uniform and five targets')
rows=[('uniform',1.0)]+[(f'target {t}',G.grade_for_growth_ratio(n,t)) for t in (1.1,1.2,1.3,1.5,2.0)]
for k,(lab,p) in enumerate(rows):
    x=G.graded_partition(0.0,1.0,n,p); y=-k
    col=GREY if k==0 else (GRN if lab=='target 1.2' else (RED if lab=='target 2.0' else BLUE))
    ax.plot([0,1],[y,y],color=col,lw=.7); ax.plot(x,np.full_like(x,y),'|',color=col,ms=11,mew=1.3)
    ax.text(-0.02,y,lab,ha='right',va='center',fontsize=7.5,color=col); ax.text(1.02,y,f'$p={p:.2f}$',ha='left',va='center',fontsize=7.5,color=col)
ax.set_xlim(-0.22,1.12); ax.set_ylim(-5.6,0.8); ax.axis('off')
ax.annotate('',(0.5,-5.35),(0.0,-5.35),arrowprops=dict(arrowstyle='<-',color='#555',lw=.8)); ax.text(0.25,-5.5,'hole boundary: smallest elements',fontsize=7,ha='center',va='top',color='#555')
fig.savefig(os.path.join(HERE,'..','figures','ch02_plan.pdf'))
print([ (l,round(p,3)) for l,p in rows])
