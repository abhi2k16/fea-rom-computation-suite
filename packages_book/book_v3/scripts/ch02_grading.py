import os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
src=os.path.join(HERE,'..','..','..','fea_engine','src')
if os.path.isdir(src): sys.path.insert(0,src)
from fea_engine import grading as G
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GREY='#8a96a3'; GRN='#2e8b57'; ORG='#d9822b'
n=20
ps=np.linspace(1,2.4,60); gr=np.array([G.growth_ratio_of_partition(G.graded_partition(0.0,1.0,n,p)) for p in ps])
p12=G.grade_for_growth_ratio(n,1.2); r12=G.growth_ratio_of_partition(G.graded_partition(0.0,1.0,n,p12))
p2=2.0; r2=G.growth_ratio_of_partition(G.graded_partition(0.0,1.0,n,p2))
fig=plt.figure(figsize=(6.8,4.4)); gs=fig.add_gridspec(2,2,height_ratios=[1.15,1],hspace=0.55,wspace=0.28,left=0.08,right=0.99,top=0.93,bottom=0.1)
ax=fig.add_subplot(gs[0,0])
ax.plot(ps,gr,color=BLUE,lw=2)
ax.axhline(1.2,color=GRN,ls='--',lw=1); ax.text(2.4,1.2-0.05,'rule of thumb 1.2',color=GRN,fontsize=7,ha='right',va='top')
ax.axhline(1.5,color=ORG,ls='--',lw=1); ax.text(2.4,1.5+0.04,'solver warns above 1.5',color=ORG,fontsize=7,ha='right',va='bottom')
ax.plot([1],[gr[0]],'o',color=NAVY,ms=4); ax.text(1.0,0.62,'uniform',color=NAVY,fontsize=7.5,va='center',ha='left')
ax.plot([p12],[r12],'o',color=GRN,ms=5,zorder=5); ax.text(p12+0.05,r12-0.12,f'target 1.2: $p={p12:.2f}$',color=GRN,fontsize=7.5,va='top',ha='left')
ax.plot([p2],[r2],'o',color=RED,ms=5,zorder=5); ax.text(p2-0.07,r2+0.15,f'$p=2$:\nratio {r2:.1f}',color=RED,fontsize=7.5,ha='right',va='bottom')
ax.set_xlabel('exponent $p$ of $x_i=L(i/n)^p$'); ax.set_ylabel('largest ratio of adjacent sizes'); ax.set_ylim(0.5,gr.max()*1.05); ax.set_xlim(0.95,2.4)
ax.set_title('(a) The exponent sets the growth ratio ($n=20$)',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax=fig.add_subplot(gs[0,1])
i=np.arange(1,n+1)
for p,c,l,o in [(1.0,GREY,'uniform',-0.27),(p12,GRN,f'target 1.2 ($p={p12:.2f}$)',0.0),(p2,RED,'$p=2$',0.27)]:
    h=np.diff(G.graded_partition(0.0,1.0,n,p)); ax.bar(i+o,h/h[0],0.26,color=c,label=l)
ax.set_yscale('log'); ax.set_ylim(0.8,300); ax.set_xlabel('element number $i$'); ax.set_ylabel('size $h_i/h_1$ (log scale)'); ax.legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,0.98))
ax.set_title('(b) Element sizes along the partition',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax=fig.add_subplot(gs[1,:])
rows=[('uniform',1.0,GREY),(f'target 1.2  ($p={p12:.2f}$, ratio {r12:.2f})',p12,GRN),(f'$p=2$  (ratio {r2:.1f})',p2,RED)]
for k,(lab,p,c) in enumerate(rows):
    x=G.graded_partition(0.0,1.0,n,p); y=-k
    ax.plot([0,1],[y,y],color=c,lw=.8); ax.plot(x,np.full_like(x,y),'|',color=c,ms=13,mew=1.3)
    ax.text(-0.015,y,lab,ha='right',va='center',fontsize=7.5,color=c)
ax.set_xlim(-0.45,1.02); ax.set_ylim(-2.6,0.5); ax.axis('off')
ax.text(0.5,0.45,'(c) The same 20 elements on a unit length',fontsize=8.5,fontweight='bold',color=NAVY,ha='center')
ax.annotate('',(0.6,-2.45),(0.0,-2.45),arrowprops=dict(arrowstyle='<->',color='#555',lw=.8))
ax.text(0.3,-2.55,'refine here (hole, notch, fillet)',fontsize=7.2,ha='center',va='top',color='#555')
fig.savefig(os.path.join(HERE,'..','figures','ch02_grading.pdf'))
print(p12,r12,r2)
