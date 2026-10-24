import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; ORG='#d9822b'; GREY='#8a96a3'; VIO='#6a3d9a'
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures')
def title(ax,t): ax.set_title(t,fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
def clean(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
# ---------------------------------------------------------------- B.1 three ways to a derivative
f=lambda x: np.exp(x)/np.sqrt(np.sin(x)**3+np.cos(x)**3)
x0=np.pi/4
# exact derivative by mpmath-free analytic: use complex step with tiny h as reference
import mpmath as mp
mp.mp.dps=40
fm=lambda x: mp.e**x/mp.sqrt(mp.sin(x)**3+mp.cos(x)**3)
ref=float(mp.diff(fm,mp.pi/4))
hs=np.logspace(-16,0,200)
fwd=np.abs((f(x0+hs)-f(x0))/hs-ref)/abs(ref)
cen=np.abs((f(x0+hs)-f(x0-hs))/(2*hs)-ref)/abs(ref)
cs=np.abs(np.imag(f(x0+1j*hs))/hs-ref)/abs(ref)
fig,axs=plt.subplots(1,2,figsize=(6.8,3.0),gridspec_kw=dict(width_ratios=[1.5,1],wspace=0.3)); plt.subplots_adjust(left=0.11,right=0.99,top=0.88,bottom=0.17)
ax=axs[0]
ax.loglog(hs,np.maximum(fwd,1e-17),color=ORG,lw=1.8,label='forward difference'); ax.loglog(hs,np.maximum(cen,1e-17),color=BLUE,lw=1.8,label='central difference'); ax.loglog(hs,np.maximum(cs,1e-17),color=GRN,lw=2,label='complex step')
ax.axhline(1.2e-16,color=RED,lw=1.4,ls='--'); ax.text(3e-16,3.6e-16,'automatic differentiation: rounding error only, no step size',color=RED,fontsize=7,va='bottom')
ax.set_ylim(5e-17,10); ax.set_xlim(1e-16,1); ax.set_xlabel('step $h$'); ax.set_ylabel('relative error')
ax.legend(frameon=False,fontsize=7,loc='lower left',bbox_to_anchor=(0.0,0.12))
ax.text(2e-4,1e-9,'truncation:\nstep too large',fontsize=7,color='#555',ha='center'); ax.text(2e-13,1e-3,'rounding: subtraction\nof nearly equal numbers',fontsize=7,color='#555',ha='center')
title(ax,'(a) Error against step size'); clean(ax)
ax=axs[1]; n=12
labels=['central\ndifference','complex\nstep','reverse AD\n(1 fwd + $n$ bwd)']; vals=[2*n,n,n+1]; cols=[BLUE,GRN,RED]
ax.bar(range(3),vals,color=cols,width=0.6)
for i,v in enumerate(vals): ax.text(i,v+0.5,str(v),ha='center',fontsize=8,color=cols[i])
ax.set_xticks(range(3)); ax.set_xticklabels(labels,fontsize=7); ax.set_yticks([]); ax.set_ylabel('passes for a full tangent ($n=12$)')
title(ax,'(b) Cost of an $n\\times n$ tangent'); ax.spines['left'].set_visible(False); clean(ax)
fig.savefig(os.path.join(OUT,'appB_derivs.pdf'))
b1=(hs[np.argmin(cen)],cen.min(),cs.min())
# ---------------------------------------------------------------- B.2 additive correction and explicit calibration
Fint=lambda u:u+0.4*u**3; dF=lambda u:1+1.2*u**2; Fext=2.0
from scipy.optimize import brentq
u_unc=brentq(lambda u:Fext-Fint(u),0,3); wm=0.8; gap=Fext-Fint(wm)
phi=lambda u:np.tanh(2*u); dphi=lambda u:2/np.cosh(2*u)**2; th=gap/phi(wm)
fth=lambda u:th*phi(u); dfth=lambda u:th*dphi(u)
u_cor=brentq(lambda u:Fext-Fint(u)-fth(u),0,3)
fig,axs=plt.subplots(1,2,figsize=(6.8,3.1),gridspec_kw=dict(width_ratios=[1.35,1],wspace=0.28)); plt.subplots_adjust(left=0.07,right=0.99,top=0.88,bottom=0.17)
ax=axs[0]; us=np.linspace(0,1.9,200)
ax.plot(us,Fint(us),color=BLUE,lw=2); ax.plot(us,Fint(us)+fth(us),color=GRN,lw=2); ax.plot([wm,wm],[0,Fext],color=GREY,lw=.7,ls=':'); ax.plot([u_unc,u_unc],[0,Fext],color=GREY,lw=.7,ls=':')
ax.axhline(Fext,color=GREY,ls='--',lw=1); ax.text(1.88,Fext-0.08,'external load $F_{ext}$',color='#555',fontsize=7.5,ha='right',va='top')
ax.plot([u_unc],[Fext],'o',color=BLUE,ms=5); ax.text(u_unc+0.04,Fext+0.12,'uncorrected\nequilibrium',color=BLUE,fontsize=7.5,va='bottom')
ax.plot([wm],[Fext],'o',color=GRN,ms=5); ax.text(wm-0.04,Fext+0.12,'reference state $w_m$ =\ncorrected equilibrium',color=GRN,fontsize=7.5,ha='right',va='bottom')
ax.annotate('',(wm,Fint(wm)+fth(wm)),(wm,Fint(wm)),arrowprops=dict(arrowstyle='<->',color=RED,lw=1.6)); ax.text(wm-0.06,(Fint(wm)+Fint(wm)+fth(wm))/2,r'$f_\theta(w_m)=$'+'\n'+r'$F_{ext}-F_{int}(w_m)$',color=RED,fontsize=7.5,va='center',ha='right')
ax.text(1.62,Fint(1.62)-0.4,'$F_{int}(u)$',color=BLUE,fontsize=8,ha='center'); ax.text(1.28,Fint(1.28)+fth(1.28)+0.15,r'$F_{int}+f_\theta$',color=GRN,fontsize=8,ha='right')
ax.set_xlabel('displacement $u$'); ax.set_ylabel('force'); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0,4.0); ax.set_xlim(0,1.9)
title(ax,'(a) The correction moves the equilibrium'); clean(ax)
ax=axs[1]
def newton(use_j):
    u=0.2; r=[]
    for k in range(9):
        res=Fext-Fint(u)-fth(u); r.append(abs(res))
        K=dF(u)+(dfth(u) if use_j else 0.0); u=u+res/K
    return np.array(r)
rf,rd=newton(True),newton(False)
ax.semilogy(range(9),np.maximum(rf,1e-17),'o-',color=GRN,lw=1.8,ms=4,label='tangent $K_T+\\partial f_\\theta/\\partial u$'); ax.semilogy(range(9),np.maximum(rd,1e-17),'s--',color=ORG,lw=1.8,ms=4,label='Jacobian of the correction dropped')
ax.set_xlabel('Newton iteration'); ax.set_ylabel('residual'); ax.legend(frameon=False,fontsize=7,loc='lower left'); ax.set_ylim(1e-14,3)
title(ax,'(b) Why the tangent needs $\\partial f_\\theta/\\partial u$'); clean(ax)
fig.savefig(os.path.join(OUT,'appB_correction.pdf'))
b2=(u_unc,u_cor,wm,th,rf[:6],rd[:6])
# ---------------------------------------------------------------- B.3 adjoint / implicit function
def ustar(theta):
    u=0.5
    for _ in range(60):
        res=Fext-Fint(u)-theta*phi(u); u+=res/(dF(u)+theta*dphi(u))
    return u
um=1.25
L=lambda theta:0.5*(ustar(theta)-um)**2
th0=0.1
u0=ustar(th0); Keff=dF(u0)+th0*dphi(u0); g=u0-um
lam=-g/Keff; dLdth_adj=lam*(-phi(u0))*(-1)  # dL/dtheta = lambda^T * dG/dtheta' , G=Fext-Fint-th*phi: dG/dth=-phi, Keff^T lam=-g -> dL/dth = lam*(-(-phi))?
# direct: du/dth = -(dG/du)^-1 dG/dth ; dG/du=-Keff ; dG/dth=-phi -> du/dth = -phi/Keff ; dL/dth = g*du/dth
dudth=-phi(u0)/Keff; dL_impl=g*dudth
eps=1e-6; dL_fd=(L(th0+eps)-L(th0-eps))/(2*eps)
ths=np.linspace(-0.2,0.9,200); us_=np.array([ustar(t) for t in ths]); Ls=np.array([L(t) for t in ths])
fig,axs=plt.subplots(1,3,figsize=(6.8,2.9),gridspec_kw=dict(width_ratios=[1,1,0.8],wspace=0.42)); plt.subplots_adjust(left=0.07,right=0.99,top=0.86,bottom=0.2)
ax=axs[0]; ax.plot(ths,us_,color=BLUE,lw=2); ax.plot([th0],[u0],'o',color=NAVY,ms=5)
tt=np.array([th0-0.25,th0+0.25]); ax.plot(tt,u0+dudth*(tt-th0),color=RED,lw=1.6)
ax.axhline(um,color=GREY,ls='--',lw=1); ax.text(0.9,um+0.015,'measured $u_m$',color='#555',fontsize=7,ha='right')
ax.text(0.04,0.06,'slope $\\mathrm{d}u^\\ast/\\mathrm{d}\\theta$\n$=-K_{\\rm eff}^{-1}\\,\\partial f_\\theta/\\partial\\theta$',color=RED,fontsize=7,va='bottom',ha='left',transform=ax.transAxes)
ax.set_xlabel(r'parameter $\theta$'); ax.set_ylabel(r'converged solution $u^\ast(\theta)$'); ax.set_xticks([]); ax.set_yticks([]); title(ax,'(a) Solution'); clean(ax)
ax=axs[1]; ax.plot(ths,Ls,color=BLUE,lw=2); ax.plot([th0],[L(th0)],'o',color=NAVY,ms=5)
ax.plot(tt,L(th0)+dL_impl*(tt-th0),color=RED,lw=1.6)
ax.text(0.04,0.95,'slope $=\\lambda\\,\\partial f_\\theta/\\partial\\theta$\nfrom one adjoint solve',color=RED,fontsize=7,va='top',ha='left',transform=ax.transAxes)
ax.set_xlabel(r'parameter $\theta$'); ax.set_ylabel(r'loss $L(u^\ast(\theta))$'); ax.set_xticks([]); ax.set_yticks([]); title(ax,'(b) Loss'); clean(ax)
ax=axs[2]; k=6
ax.bar([0,1],[k,1],color=[ORG,GRN],width=0.6); ax.text(0,k+0.2,f'{k}',ha='center',fontsize=8,color=ORG); ax.text(1,1.2,'1',ha='center',fontsize=8,color=GRN)
ax.set_xticks([0,1]); ax.set_xticklabels(['unroll\nNewton','adjoint\nsolve'],fontsize=7); ax.set_yticks([]); ax.set_ylim(0,k+1.5); ax.set_ylabel('states stored'); ax.spines['left'].set_visible(False); title(ax,'(c) Memory'); clean(ax)
fig.savefig(os.path.join(OUT,'appB_adjoint.pdf'))
b3=(dL_impl,dL_fd)
# ---------------------------------------------------------------- B.4 metric-consistent step
h=np.r_[np.full(10,0.01),np.full(10,0.1)]; xnode=np.r_[0,np.cumsum(h)]; xc=0.5*(xnode[1:]+xnode[:-1])
grad=np.ones(20)   # equal gradient per cell value
mass=h             # lumped mass
eta=1.0
plain=eta*grad; metric=eta*mass*grad
fig,axs=plt.subplots(2,1,figsize=(6.8,3.4),gridspec_kw=dict(height_ratios=[0.55,1],hspace=0.55)); plt.subplots_adjust(left=0.1,right=0.99,top=0.9,bottom=0.15)
ax=axs[0]
for i in range(21): ax.plot([xnode[i]]*2,[0,1],color=NAVY,lw=.8)
ax.plot([0,xnode[-1]],[0,0],color=NAVY,lw=1.2); ax.plot([0,xnode[-1]],[1,1],color=NAVY,lw=1.2)
ax.text(xnode[10]/2,1.45,'fine cells, $h=0.01$',ha='center',fontsize=8,color=NAVY); ax.text((xnode[10]+xnode[-1])/2,1.45,'coarse cells, $h=0.1$',ha='center',fontsize=8,color=NAVY)
ax.set_ylim(-0.1,2); ax.axis('off'); ax.set_title('(a) A line of cells with a field parameter in each',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
ax=axs[1]; idx=np.arange(20); w=0.38
ax.bar(idx-w/2,plain/plain.max(),w,color=GREY,label='plain step $\\eta\\,\\nabla L$ (same in every cell)')
ax.bar(idx+w/2,metric/metric.max(),w,color=GRN,label='metric step $\\eta\\,\\mathbf{M}\\,\\nabla L$ (proportional to the cell size)')
ax.set_xticks([]); ax.set_ylabel('step (relative)'); ax.set_xlabel('cell (same equal gradient in all cells)')
ax.legend(frameon=False,fontsize=7.5,loc='upper left',bbox_to_anchor=(0.0,1.05)); ax.set_ylim(0,1.45); clean(ax)
ax.set_title('(b) One optimiser step for the same gradient',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
fig.savefig(os.path.join(OUT,'appB_metric.pdf'))
print(b1); print(b2[0],b2[1],b2[3]); print(b2[4]); print(b2[5]); print(b3, metric[-1]/metric[0])
