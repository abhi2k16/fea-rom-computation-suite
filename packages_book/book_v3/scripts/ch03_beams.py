"""Chapter 3: beam checks. (a) Euler-Bernoulli first cantilever frequency (cubic Hermite, order 4);
(b) Reissner beam against the Timoshenko tip deflection for L/h from 2 to 1000 (shear effect, no shear locking);
(c) convergence of the Reissner tip deflection for a stocky beam (L/h = 3).
Writes figures/ch03_beams.pdf (or preview_figs/ without FINAL=1) and tables/numbers_ch03_beams.tex."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fea_engine import FESystem, Material
from fea_engine.geometry import generate_mesh
from fea_engine.material import EI_beam, Section
from fea_engine.mesh import Mesh
from fea_engine.elements import Beam2DReissner, Beam2DEulerBernoulli
import fea_engine.convergence as cv
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; GREY='#8a96a3'
HERE=os.path.dirname(os.path.abspath(__file__))
def sci(x):
    x=max(abs(x),1e-16); e=int(np.floor(np.log10(x))); return '$%.1f\\times10^{%d}$'%(x/10**e,e)
mac={}
# (a) Euler-Bernoulli
E,rho,A,I,L=210e9,7800.0,0.01,8.33e-6,2.0
lam1=1.8751040687; f_exact=lam1**2/(2*np.pi*L**2)*np.sqrt(E*I/(rho*A))
def f1(n):
    b=FESystem(generate_mesh(dim=1,L=L,n=n),Beam2DEulerBernoulli())
    b.assemble_stiffness(EI_beam(Material(E=E,nu=0.3,rho=rho),Section(A=A,I=I))); b.assemble_mass(rho*A)
    b.fix_dofs([0],["uy","rz"]); return b.solve_modal(n_modes=1)[0][0]
st1=cv.run_study(f1,[2,4,8,16],exact=f_exact)
Et,nu,Lt,P=210e9,0.3,1.0,1000.0; Gt=Et/(2*(1+nu)); bw=0.1
def eb_tip(n,A_,I_):
    b=FESystem(generate_mesh(dim=1,L=Lt,n=n),Beam2DEulerBernoulli())
    b.assemble_stiffness(EI_beam(Material(E=Et,nu=0.3,rho=rho),Section(A=A_,I=I_)))
    b.fix_dofs([0],["uy","rz"]); b.add_nodal_force([n],"uy",P); return b.solve_static().component("uy",nodes=[n])[0]
A0,I0=1.0e-3,8.33e-7
eb1=eb_tip(1,A0,I0); eb_ref=P*Lt**3/(3*Et*I0)
mac['cecEBTipErr']=sci(abs(eb1/eb_ref-1)); mac['cecEBFreqOrder']='%.1f'%st1.order; mac['cecEBFreqHz']='%.2f'%f_exact
# (b),(c) Reissner
def tipR(n,A_,I_):
    x=np.linspace(0,Lt,n+1).reshape(-1,1)
    s=FESystem(Mesh(nodes=np.hstack([x,0*x]),elements=np.array([[i,i+1] for i in range(n)]),dim=1),Beam2DReissner())
    s.fix_dofs([0],["ux","uy","rz"]); s.assemble_stiffness((Et,Gt,A_,I_,1.0)); s.add_nodal_force([n],"uy",P)
    return s.solve_static().component("uy",nodes=[n])[0]
def sect(ld): h=Lt/ld; return bw*h, bw*h**3/12
lds=np.array([2,3,5,10,20,50,100,300,1000.]); NB=8
rat=[];ebr=[]
for ld in lds:
    A_,I_=sect(ld); ex=P*Lt**3/(3*Et*I_)+P*Lt/(Gt*A_)
    rat.append(tipR(NB,A_,I_)/ex); ebr.append(P*Lt**3/(3*Et*I_)/ex)
rat=np.array(rat); ebr=np.array(ebr)
LD=3; A_,I_=sect(LD); ex=P*Lt**3/(3*Et*I_)+P*Lt/(Gt*A_)
st2=cv.run_study(lambda n: tipR(n,A_,I_),[2,4,8,16],exact=ex)
mac['cecTimoOrder']='%.1f'%st2.order; mac['cecTimoLD']=str(LD)
mac['cecTimoShearShare']='%.1f'%(100*(1-P*Lt**3/(3*Et*I_)/ex))
mac['cecTimoEBLowLD']='%d'%lds[0]; mac['cecTimoEBLow']='%.0f'%(100*(1-ebr[0]))
mac['cecTimoNB']=str(NB); mac['cecTimoFlatMin']='%.1f'%(100*rat.min()); mac['cecTimoFlatMax']='%.1f'%(100*rat.max())
mac['cecTimoThinLD']='%d'%lds[-1]
out=os.path.join(HERE,'..','figures' if os.environ.get('FINAL') else 'preview_figs')
fig,ax=plt.subplots(1,3,figsize=(8.4,3.0),gridspec_kw={'width_ratios':[1,1.15,1]})
a=ax[0]; a.loglog(st1.h,st1.errors/abs(st1.exact if hasattr(st1,'exact') else f_exact),'o-',color=BLUE,label='observed order %.1f'%st1.order)
a.set_xlabel('element size $h/L$'); a.set_ylabel('relative error'); a.set_title('(a) Euler--Bernoulli,\nfirst frequency',fontsize=8.5,color=NAVY); a.legend(frameon=False,fontsize=7.5)
a=ax[1]; a.semilogx(lds,ebr,'s--',color=RED,mfc='none',label='Euler--Bernoulli (no shear)'); a.semilogx(lds,rat,'o-',color=BLUE,label='Reissner element, %d elements'%NB)
a.axhline(1,color=GREY,lw=0.8); a.set_ylim(0.8,1.03); a.set_xlabel('slenderness $L/h$'); a.set_ylabel('tip deflection / Timoshenko')
a.set_title('(b) Reissner beam, thick to thin',fontsize=8.5,color=NAVY); a.legend(frameon=False,fontsize=7.2,loc='lower right')
a=ax[2]; a.loglog(st2.h,st2.errors/abs(ex),'o-',color=BLUE,label='observed order %.1f'%st2.order)
a.set_xlabel('element size $h/L$'); a.set_ylabel('relative error'); a.set_title('(c) Reissner tip deflection,\n$L/h=%d$'%LD,fontsize=8.5,color=NAVY); a.legend(frameon=False,fontsize=7.5)
for a in (ax[0],ax[2]):
    a.set_xticks([1/16,1/8,1/4,1/2]); a.set_xticklabels(['1/16','1/8','1/4','1/2']); a.minorticks_off()
for a in ax:
    for sp in ('top','right'): a.spines[sp].set_visible(False)
    a.grid(alpha=0.25,which='both',lw=0.4)
fig.tight_layout(); fig.savefig(os.path.join(out,'ch03_beams.pdf')); fig.savefig(os.path.join(out,'ch03_beams.png'),dpi=130)
open(os.path.join(HERE,'..','tables','numbers_ch03_beams_new.tex' if not os.environ.get('FINAL') else 'numbers_ch03_beams.tex'.join(['',''])),'w') if False else None
tabdir=os.path.join(HERE,'..','tables')
open(os.path.join(tabdir,'numbers_ch03_beams.tex' if os.environ.get('FINAL') else 'preview_numbers_ch03_beams.tex'),'w').write(''.join('\\newcommand{\\%s}{%s\\xspace}\n'%(k,v) for k,v in mac.items()))
print(mac); print('ratios',np.round(rat,4),'EB',np.round(ebr,4))
