"""Chapter 4: lowest axial frequency of a fixed-free thin bar with consistent, lumped and averaged mass (figure ch04_mass.pdf, macros dyMass*)."""
import os, numpy as np, scipy.linalg as sl, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fea_engine import FESystem, Material, Quad4PlaneStress, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.geometry import generate_mesh
from fea_engine.material import EI_beam, Section
from fea_engine.elements import Beam2DEulerBernoulli
import fea_engine.convergence as cv
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; GREY='#8a96a3'
E,rho,L=1e9,1000.0,1.0; ex=(np.pi/2)**2*E/(rho*L**2)
ns=[4,8,16,32,64]; err={'c':[],'l':[],'a':[]}
d=lambda A: np.asarray(A.toarray() if hasattr(A,"toarray") else A)
for n in ns:
    mesh=rectangle_mesh(Lx=L,Ly=0.1,nx=n,ny=1); s=FESystem(mesh,Quad4PlaneStress(),thickness=0.1)
    s.assemble_stiffness(D_plane_stress(Material(E=E,nu=0.0,rho=rho)),thickness=0.1)
    s.assemble_mass(rho*np.eye(2),thickness=0.1); s.assemble_lumped_mass(rho*np.eye(2),thickness=0.1)
    mesh.select_nodes(x=0.0,name="root"); s.fix_dofs("root",["ux"]); s.fix_dofs(np.arange(len(mesh.nodes)),["uy"])
    fr=s.free_dofs; K=d(s.K)[np.ix_(fr,fr)]; Mc=d(s.M)[np.ix_(fr,fr)]; Ml=d(s.M_lumped)[np.ix_(fr,fr)]
    for k,M in (('c',Mc),('l',Ml),('a',0.5*(Mc+Ml))): err[k].append(sl.eigh(K,M,eigvals_only=True)[0]/ex-1)
# (a) Euler-Bernoulli cantilever, first frequency (the test beam of this chapter)
Eb,rb,Ab,Ib,Lb=210e9,7800.0,0.01,8.33e-6,2.0
fex=1.8751040687**2/(2*np.pi*Lb**2)*np.sqrt(Eb*Ib/(rb*Ab))
def f1(n):
    b=FESystem(generate_mesh(dim=1,L=Lb,n=n),Beam2DEulerBernoulli())
    b.assemble_stiffness(EI_beam(Material(E=Eb,nu=0.3,rho=rb),Section(A=Ab,I=Ib))); b.assemble_mass(rb*Ab)
    b.fix_dofs([0],["uy","rz"]); return b.solve_modal(n_modes=1)[0][0]
stb=cv.run_study(f1,[2,4,8,16],exact=fex)
h=1/np.array(ns)
fig,axs=plt.subplots(1,2,figsize=(6.8,3.2)); ax=axs[1]
ax.loglog(h,np.abs(err['c']),'o-',color=BLUE,lw=1.8,label='consistent, error $>0$, order 2')
ax.loglog(h,np.abs(err['l']),'s--',color=RED,lw=1.4,mfc='none',label='lumped, error $<0$, order 2')
ax.loglog(h,np.abs(err['a']),'^-',color=GRN,lw=1.8,label='average, order 4')
ax.set_xlabel('element size $h_e/L$'); ax.set_ylabel('relative error of $\\lambda_1$'); ax.legend(frameon=True,facecolor='white',edgecolor='none',fontsize=7.2,loc='lower right'); ax.set_title('(b) Bar, consistent and lumped mass',fontsize=8.5,color=NAVY)
ax0=axs[0]; ax0.loglog(stb.h,stb.errors/fex,'o-',color=BLUE,label='observed order %.1f'%stb.order); ax0.set_xticks([1/16,1/8,1/4,1/2]); ax0.set_xticklabels(['1/16','1/8','1/4','1/2']); ax0.minorticks_off(); ax0.set_xlabel('element size $h_e/L$'); ax0.set_ylabel('relative error of $f_1$'); ax0.set_title('(a) Euler--Bernoulli cantilever',fontsize=8.5,color=NAVY); ax0.legend(frameon=False,fontsize=7.5)
for ax_ in axs:
    for sp in ('top','right'): ax_.spines[sp].set_visible(False)
    ax_.grid(alpha=0.25,which='both',lw=0.4)
fig.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures' if os.environ.get('FINAL') else 'preview_figs')
fig.savefig(os.path.join(out,'ch04_mass.pdf')); fig.savefig(os.path.join(out,'ch04_mass.png'),dpi=130)
oc,ol,oa=[cv.observed_order(h,np.abs(err[k])) for k in 'cla']
sci=lambda x: ('$%.1f\\times10^{%d}$'%(abs(x)/10**np.floor(np.log10(abs(x))),int(np.floor(np.log10(abs(x))))))
mac={'cdyMassConsCoarse':'%.2f\\,\\%%'%(100*err['c'][0]),'cdyMassLumpCoarse':'%.2f\\,\\%%'%(100*abs(err['l'][0])),'cdyMassConsFine':sci(err['c'][-1]),'cdyMassLumpFine':sci(err['l'][-1]),'cdyMassAvgFine':sci(err['a'][-1]),'cdyMassConsOrder':'%.1f'%oc,'cdyMassLumpOrder':'%.1f'%ol,'cdyMassAvgOrder':'%.1f'%oa,'cdyMassNFine':str(ns[-1]),'cdyBeamFreqOrder':'%.1f'%stb.order}
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','tables','numbers_ch04_mass.tex'),'w').write(''.join('\\newcommand{\\%s}{%s\\xspace}\n'%(k,v) for k,v in mac.items()))
print(mac)
print('fine',err['c'][-1],err['l'][-1],err['a'][-1],'orders',oc,ol,oa,'coarse',err['c'][0],err['l'][0])
