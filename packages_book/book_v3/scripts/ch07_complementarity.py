import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; VIO='#6a3d9a'; GREY='#8a96a3'; GRN='#2e8b57'; ORG='#d9822b'
# --- small PDAS example: a string under load pressed onto a floor
n=24; x=np.linspace(0,1,n+2)[1:-1]; h=1/(n+1)
K=(np.diag(2*np.ones(n))-np.diag(np.ones(n-1),1)-np.diag(np.ones(n-1),-1))/h**2*0.02
f=-np.ones(n)/6.24
c=-0.72+0.40*np.exp(-((x-0.30)/0.05)**2)+0.55*np.exp(-((x-0.70)/0.05)**2)
cp=np.mean(np.diag(K))
A=np.zeros(n,bool); hist=[]
for k in range(8):
    idx=np.where(A)[0]; m=len(idx)
    if m:
        Bm=np.zeros((m,n)); Bm[np.arange(m),idx]=1
        S=np.block([[K,-Bm.T],[Bm,np.zeros((m,m))]]); rhs=np.r_[f,c[idx]]
        sol=np.linalg.solve(S,rhs); w=sol[:n]; lam=np.zeros(n); lam[idx]=sol[n:]
    else: w=np.linalg.solve(K,f); lam=np.zeros(n)
    g=w-c; Anew=(lam-cp*g)>0
    hist.append((A.copy(),w.copy(),lam.copy(),g.copy(),Anew.copy()))
    if (Anew==A).all(): break
    A=Anew
print(len(hist),[a.sum() for a,*_ in hist],[ (g<-1e-9).sum() for _,_,_,g,_ in hist],[(l<-1e-9).sum() for _,_,l,_,_ in hist])
fig=plt.figure(figsize=(6.8,3.6))
gs=fig.add_gridspec(2,3,width_ratios=[1.15,1,1],hspace=0.55,wspace=0.28,left=0.07,right=0.99,top=0.9,bottom=0.1)
ax=fig.add_subplot(gs[:,0])
ax.set_title('(a) Contact law',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
ax.fill_between([-1.2,0],[0,0],[3.2,3.2],color=BLUE,alpha=.08)
ax.text(-1.2,3.1,'penetration $g<0$',color=BLUE,fontsize=7,va='top')
ax.plot([0,0],[3.2,0],color=RED,lw=2.4); ax.plot([0,3.2],[0,0],color=RED,lw=2.4)
ax.plot([-1.2,0.0],[2.4,0.0],color=BLUE,lw=1.6,ls='--')
ax.axhline(0,color='k',lw=.6); ax.axvline(0,color='k',lw=.6)
ax.text(0.12,2.5,'closed: $g=0$, $\\lambda>0$',color=RED,fontsize=7.5)
ax.text(1.6,0.2,'open: $g>0$, $\\lambda=0$',color=RED,fontsize=7.5,ha='center')
ax.text(-1.2,1.55,'penalty $\\lambda=-k_p\\,g$:\nforce only with penetration',color=BLUE,fontsize=7.5,va='top')
ax.plot([0],[0],'o',color='k',ms=3)
ax.text(1.0,1.7,'$\\lambda\\geq0,\\ g\\geq0,$\n$\\lambda\\,g=0$',color=NAVY,fontsize=9)
ax.set_xlim(-1.3,3.2); ax.set_ylim(-0.3,3.3); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('gap $g$'); ax.set_ylabel('contact force $\\lambda$')
for s in ['top','right']: ax.spines[s].set_visible(False)
cols=[(0,1),(0,2),(1,1),(1,2)]
titles=['0. guess: no active set','1. solve with active set,\n   then reclassify','2. solve again,\n   reclassify','3. active set unchanged: stop']
sel=list(range(min(4,len(hist))))
if len(hist)>4: sel=[0,1,2,len(hist)-1]
for q,(r,cc) in zip(sel,cols):
    pass
ax_list=[]
for p,k in enumerate(sel):
    r,cc=[(0,1),(0,2),(1,1),(1,2)][p]
    a=fig.add_subplot(gs[r,cc]); A_,w,lam,g,An=hist[k]
    a.fill_between(x,c,-1.2,color='#ddd',zorder=1); a.plot(x,c,color='#666',lw=1)
    a.plot(x,w,color=BLUE,lw=1.6,zorder=3)
    pen=g<-1e-9
    a.plot(x[pen],w[pen],'o',color=RED,ms=2.6,zorder=4)
    a.plot(x[A_],w[A_],'s',color=GRN,ms=2.4,zorder=4,mfc='none')
    a.set_xticks([]); a.set_yticks([]); a.set_ylim(-1.15,0.1)
    a.set_title(f'({chr(98+p)}) '+['$k=0$: $\\mathcal{A}=\\emptyset$, bodies overlap',f'$k={k}$: hold the penetrating points',f'$k={k}$: release or add by $\\lambda-c_pg$',f'$k={k}$: $\\mathcal{{A}}$ unchanged, stop'][p],fontsize=7,fontweight='bold',color=NAVY,loc='left')
    for s in a.spines.values(): s.set_linewidth(.5)
fig.text(0.40,0.005,'green squares: active set $\\mathcal{A}$; red dots: penetration; grey: obstacle',fontsize=7,color='#333')
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch07_complementarity.pdf'),bbox_inches='tight')
