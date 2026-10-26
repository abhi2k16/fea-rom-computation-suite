"""Chapter 4: lowest axial frequency of a fixed-free thin bar with consistent, lumped and averaged mass (figure ch04_mass.pdf, macros dyMass*)."""
import os, numpy as np, scipy.linalg as sl, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fea_engine import FESystem, Material, Quad4PlaneStress, D_plane_stress
from fea_engine.mesh import rectangle_mesh
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
h=1/np.array(ns)
fig,ax=plt.subplots(figsize=(4.8,3.3))
ax.loglog(h,np.abs(err['c']),'o-',color=BLUE,lw=1.8,label='consistent mass, error $>0$, order 2')
ax.loglog(h,np.abs(err['l']),'s--',color=RED,lw=1.4,mfc='none',label='lumped mass, error $<0$, order 2')
ax.loglog(h,np.abs(err['a']),'^-',color=GRN,lw=1.8,label='average of the two, order 4')
ax.set_xlabel('element size $h/L$'); ax.set_ylabel('relative error of $\\lambda_1$'); ax.legend(frameon=False,fontsize=7.5)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
ax.grid(alpha=0.25,which='both',lw=0.4); fig.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures' if os.environ.get('FINAL') else 'preview_figs')
fig.savefig(os.path.join(out,'ch04_mass.pdf')); fig.savefig(os.path.join(out,'ch04_mass.png'),dpi=130)
oc,ol,oa=[cv.observed_order(h,np.abs(err[k])) for k in 'cla']
print('fine',err['c'][-1],err['l'][-1],err['a'][-1],'orders',oc,ol,oa,'coarse',err['c'][0],err['l'][0])
