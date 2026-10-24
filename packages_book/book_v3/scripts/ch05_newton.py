import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; VIO='#6a3d9a'; GREY='#8a96a3'
f=lambda u:2.5*(1-np.exp(-1.2*u)); kt=lambda u:3.0*np.exp(-1.2*u); F=2.0
u=[0.2]
for _ in range(6): u.append(u[-1]+(F-f(u[-1]))/kt(u[-1]))
r=[abs(F-f(x)) for x in u]
us=np.linspace(0,2.4,200)
fig=plt.figure(figsize=(6.6,5.4))
gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],hspace=0.42,wspace=0.28)
# (a) one iteration
ax=fig.add_subplot(gs[0,:]) 
ax.plot(us,f(us),color=BLUE,lw=2,label=r'internal force $f_{int}(u)$')
ax.axhline(F,color=GREY,ls='--',lw=1); ax.text(2.38,F+0.09,r'load level $\lambda f_{ref}$',color='#555',fontsize=8,ha='right')
u0,u1=u[0],u[1]
ut=np.linspace(u0-0.15,u1+0.1,10); ax.plot(ut,f(u0)+kt(u0)*(ut-u0),color=VIO,lw=1.6)
ax.plot([u0,u0],[f(u0),F],color=RED,lw=2.2); ax.annotate('residual\n'+r'$r^0=\lambda f_{ref}-f_{int}(u^0)$',(u0,(f(u0)+F)/2),(1.0,0.75),color=RED,fontsize=8,va='center',arrowprops=dict(arrowstyle='-',color=RED,lw=.8))
ax.annotate('',(u1,F),(u0,F),arrowprops=dict(arrowstyle='-|>',color=VIO,lw=1.4))
ax.text((u0+u1)/2,F+0.2,r'$\Delta u=K_T^{-1}r^0$',color=VIO,ha='center',fontsize=8,fontweight='bold')
ax.plot([u1,u1],[0,F],color=GREY,ls=':',lw=.9)
ax.plot([u0],[f(u0)],'o',color=NAVY,ms=4,zorder=5)
ax.plot([u1],[f(u1)],'s',color=NAVY,ms=4,zorder=5); ax.plot([u1,u1],[f(u1),F],color=RED,lw=1.2,alpha=.7)
ax.text(u1+0.05,(f(u1)+F)/2-0.02,r'smaller residual $r^1$',color=RED,fontsize=7.5,va='center')
ax.set_xticks([u0,u1]); ax.set_xticklabels(['$u^0$','$u^1=u^0+\\Delta u$']); ax.set_yticks([])
ax.text(1.0,0.25,r'tangent $K_T(u^0)=\mathrm{d}f_{int}/\mathrm{d}u$',color=VIO,fontsize=8,rotation=0,ha='left')
ax.set_xlim(0,2.4); ax.set_ylim(0,2.8); ax.set_ylabel('force')
ax.set_title('(a) One Newton iteration: follow the tangent to the load level',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
# (b) iterates
ax=fig.add_subplot(gs[1,0]); ax.plot(us,f(us),color=BLUE,lw=1.8); ax.axhline(F,color=GREY,ls='--',lw=1)
for i in range(3):
    ut=np.linspace(u[i],u[i+1],5); ax.plot([u[i],u[i+1]],[f(u[i]),F],color=VIO,lw=1.1,alpha=.9)
    ax.plot([u[i],u[i]],[f(u[i]),F],color=RED,lw=.9,alpha=.7)
for i in range(4):
    ax.plot(u[i],f(u[i]),'o',color=NAVY,ms=3.5)
    if i<3: ax.text(u[i],-0.27,f'$u^{i}$',ha='center',fontsize=7.5,color=NAVY)
ax.set_xlim(0,2.4); ax.set_ylim(-0.1,2.8); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('(b) Successive iterates',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
# (c) convergence
ax=fig.add_subplot(gs[1,1]); it=np.arange(len(r)); ax.semilogy(it[:5],r[:5],'o-',color=RED,lw=1.6,ms=4)
ax.set_xlabel('iteration $k$'); ax.set_ylabel(r'$|r^k|$'); ax.set_xticks(it[:5])
ax.set_title('(c) Residual per iteration',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
ax.text(0.97,0.95,'correct digits roughly\ndouble per iteration\nnear the solution',transform=ax.transAxes,fontsize=7,color='#333',ha='right',va='top')
for s in ['top','right']: ax.spines[s].set_visible(False)
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch05_newton.pdf'),bbox_inches='tight')
print(u[:5],r[:6])
