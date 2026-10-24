import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; ORG='#d9822b'; GREY='#8a96a3'
def amp_cd(x):   # central difference, state (u_n, u_{n-1}) for u''+u=0 with x=w dt
    return np.array([[2-x**2,-1],[1,0]])
def amp_nm(x,beta=0.25,gam=0.5):  # Newmark for u''+u=0, state (u,v*dt,a*dt^2) -> use standard 2x2 on (u, v dt)
    # a_{n+1}= -u_{n+1}; u_{n+1}=u+dt v+dt^2[(1/2-beta)a+beta a_{n+1}]; v_{n+1}=v+dt[(1-gam)a+gam a_{n+1}]
    # in units dt=1, w=x : a=-x^2 u
    k=x**2; M=np.array([[1+beta*k,0],[gam*k,1]]); R=np.array([[1-(0.5-beta)*k,1],[-(1-gam)*k,1]])
    return np.linalg.solve(M,R)
xs=np.linspace(0.01,6.0,1200)
def rho_ang(f):
    r=[];a=[]
    for x in xs:
        ev=np.linalg.eigvals(f(x)); i=np.argmax(abs(ev)); r.append(abs(ev[i])); a.append(abs(np.angle(ev[i])))
    return np.array(r),np.array(a)
r_cd,a_cd=rho_ang(amp_cd); r_nm,a_nm=rho_ang(amp_nm)
per_cd=xs/a_cd-1; per_nm=xs/a_nm-1   # numerical period / exact period - 1  (T_num/T = x/angle)
fig,axs=plt.subplots(1,2,figsize=(6.8,3.1)); plt.subplots_adjust(wspace=0.28,left=0.08,right=0.99,top=0.88,bottom=0.17)
ax=axs[0]
ax.axvspan(2,6,color=RED,alpha=.08); ax.axvspan(0,2,color=GRN,alpha=.07)
ax.plot(xs,r_nm,color=BLUE,lw=2.4,); ax.plot(xs,np.minimum(r_cd,2.2),color=ORG,lw=2,ls='--')
ax.axhline(1,color=GREY,lw=.6,ls=':'); ax.axvline(2,color=RED,lw=1)
ax.text(0.1,2.2,'critical step at $\\omega\\Delta t=2$:\n$\\Delta t_c=2/\\omega_{max}$',color=RED,fontsize=7.5,va='top')
ax.text(2.2,1.95,'central difference\n(off the scale above 2.2)',color=ORG,fontsize=7.5,va='top')
ax.text(1.0,0.45,'explicit\nstable',ha='center',color=GRN,fontsize=7.5); ax.text(4.0,0.35,'explicit: spectral radius $>1$,\nthe error grows each step',ha='center',color=RED,fontsize=7.5)
ax.text(5.9,1.06,'Newmark, average acceleration:\nspectral radius $=1$ for every $\\Delta t$',color=BLUE,fontsize=7.5,ha='right',va='bottom')
ax.set_ylim(0,2.3); ax.set_xlim(0,6); ax.set_xlabel(r'$\omega\,\Delta t$ ($\omega$ = highest frequency of the discrete system)'); ax.set_ylabel('spectral radius of the step')
ax.set_title('(a) Stability',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
ax=axs[1]; m=xs<=2
ax.plot(xs[m],per_cd[m]*100,color=ORG,lw=2,ls='--'); mm=xs<=3.2
ax.plot(xs[mm],per_nm[mm]*100,color=BLUE,lw=2.4)
ax.axvline(2,color=RED,lw=1); ax.axhline(0,color=GREY,lw=.6,ls=':')
ax.text(1.0,-18,'explicit: period\ntoo short',color=ORG,fontsize=7.5,ha='left',va='top')
ax.text(0.15,48,'Newmark: period too long;\nstable, but inaccurate\nfor large $\\omega\\Delta t$',color=BLUE,fontsize=7.5,ha='left',va='top')
ax.set_xlim(0,3.2); ax.set_ylim(-42,62); ax.set_xlabel(r'$\omega\,\Delta t$'); ax.set_ylabel('period error (%)')
ax.set_title('(b) Accuracy: period error',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch04_stability.pdf'))
print(r_cd[xs<2].max(),r_nm.min(),r_nm.max(),per_cd[np.argmin(abs(xs-1.0))],per_nm[np.argmin(abs(xs-1.0))])
