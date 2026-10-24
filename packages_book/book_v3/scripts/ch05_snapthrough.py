import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; VIO='#6a3d9a'; GREY='#8a96a3'; GRN='#2e8b57'; ORG='#d9822b'
# snap-through path (a, b)
u=np.linspace(0,2.0,400); lam=2*u-2.4*u**2+0.8*u**3
# snap-back path (c, d): parametric
sg=np.linspace(-0.12,1.12,1800); th=np.radians(55+140*np.exp(-((sg-.5)/.17)**2))
ds_=0.0085
U=np.cumsum(np.cos(th))*ds_; L=np.cumsum(np.sin(th))*ds_; U=U-U[0]; L=L-L[0]
U=2.2*(U-U.min()); L=0.5*(L-L.min())+0.7
def style(ax,title,xl=True):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('displacement $u$',fontsize=8); ax.set_ylabel('load factor $\\lambda$',fontsize=8)
    ax.set_title(title,fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
    for s in ['top','right']: ax.spines[s].set_visible(False)
fig,axs=plt.subplots(2,2,figsize=(6.6,5.6)); plt.subplots_adjust(hspace=0.5,wspace=0.28)
# limit points of snap-through
uc=[(4.8-np.sqrt(4.8**2-4*2.4*2))/(2*2.4),(4.8+np.sqrt(4.8**2-4*2.4*2))/(2*2.4)]
lc=[2*x-2.4*x**2+0.8*x**3 for x in uc]
ax=axs[0,0]; style(ax,'(a) Load control: stops at the limit load')
ax.plot(u,lam,color=BLUE,lw=2)
ax.plot(uc[0],lc[0],'o',color=RED,ms=5); ax.text(uc[0]-0.05,lc[0]+0.08,'limit point\n$\\lambda_{max}$',color=RED,ha='right',fontsize=7.5)
for l in np.linspace(0.1,lc[0],5)[:-1]:
    uu=np.interp(l,lam[u<uc[0]],u[u<uc[0]]); ax.plot([uu],[l],'.',color=ORG,ms=5)
ax.annotate('',(1.78,lc[0]+0.0),(uc[0]+0.04,lc[0]+0.02),arrowprops=dict(arrowstyle='-|>',color=ORG,lw=1.4,connectionstyle='arc3,rad=-0.25'))
ax.text(1.1,lc[0]+0.34,'no solution above $\\lambda_{max}$:\nthe iterate jumps or diverges',color=ORG,ha='center',fontsize=7.5)
ax.set_ylim(-0.05,1.1)
ax=axs[0,1]; style(ax,'(b) Displacement control: follows a snap-through')
ax.plot(u,lam,color=BLUE,lw=2)
for x in np.linspace(0.1,1.95,12):
    ax.plot([x,x],[0,2*x-2.4*x**2+0.8*x**3],color=GRN,lw=.6,alpha=.6); ax.plot(x,2*x-2.4*x**2+0.8*x**3,'o',color=GRN,ms=3)
ax.text(1.0,0.07,'prescribe $u$, solve for $\\lambda$:\none solution per step',color=GRN,fontsize=7.5,ha='center'); ax.set_ylim(-0.05,1.1)
ax=axs[1,0]; style(ax,'(c) ... but not a snap-back')
ax.plot(U,L,color=BLUE,lw=2)
xp=0.5*(U[0]+U[-1])
idx=np.where(np.diff(np.sign(U-xp)))[0]
ax.axvline(xp,color=GRN,lw=1,ls='--')
for i in idx: ax.plot(xp,np.interp(xp,[U[i],U[i+1]][::1] if U[i+1]>U[i] else [U[i+1],U[i]],[L[i],L[i+1]] if U[i+1]>U[i] else [L[i+1],L[i]]),'o',color=RED,ms=5,zorder=6)
ax.text(0.02,0.98,'one prescribed $u$ meets the path\nthree times: the load is not unique',color=RED,fontsize=7.5,va='top',transform=ax.transAxes)
ax=axs[1,1]; style(ax,'(d) Arc length: marches along the path')
ax.plot(U,L,color=BLUE,lw=2)
ds=np.hypot(np.diff(U),np.diff(L)); s=np.concatenate([[0],np.cumsum(ds)])
pts=np.arange(0,s[-1],0.55)
for p in pts: j=np.searchsorted(s,p); ax.plot(U[j],L[j],'o',color=VIO,ms=3.2)
j=np.searchsorted(s,pts[3]); th=np.linspace(0,2*np.pi,100)
ax.plot(U[j]+0.85*np.cos(th),L[j]+0.85*np.sin(th),color=VIO,lw=1,ls='--')
ax.text(0.02,0.98,'step length fixed:\n$\\|\\Delta u,\\Delta\\lambda\\|=\\Delta_L$\n$\\lambda$ is an unknown',color=VIO,fontsize=7.5,ha='left',va='top',transform=ax.transAxes)

for a_ in (axs[1,0],axs[1,1]):
    a_.set_xlim(U.min()-0.3,U.max()+2.6); W=U.max()-U.min()+2.9; a_.set_ylim(L.min()-0.4,L.min()-0.4+W/1.31)
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch05_snapthrough.pdf'),bbox_inches='tight')
print(U.max(),L.max(),L.min())
