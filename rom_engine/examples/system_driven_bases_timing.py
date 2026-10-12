# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
system_driven_bases_timing.py -- how fast, and at what offline cost, are the system-driven reduced models?

A parameter sweep of static load levels (1 to 4 beam thicknesses of mid-deflection, loaded along mode 1) on the
geometrically nonlinear clamped-clamped beam of fea_engine, solved by

  * the full-order load-stepped Newton solver,
  * Galerkin on 3 vibration modes and on the modal-derivative basis (both still call the FULL-order force),
  * a polynomial reduced force identified from tangent probes (enhanced enforced displacement), no full-order calls,
  * a 3-coordinate quadratic manifold with ECSW trained on lifted states, only 10-20% of the elements evaluated.

For each: relative error against the full-order solution (worst case over the sweep), online time per query, offline
time, speed-up per query, effective speed-up over the sweep (offline cost charged: rom_engine.effective_speedup) and the
break-even number of queries. Results are printed as JSON. Wall-clock numbers depend on the machine, NumPy/BLAS build
and thread count; run it where you will use it.

    python examples/system_driven_bases_timing.py [n_elem=100] [n_sweep=8]
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tests"), os.path.join(HERE, "..", "src")]   # fixtures use fea_engine
from nonlinear_beam_fixtures import nonlinear_beam_model
from rom_engine import (static_modal_derivatives, augment_with_derivatives, dual_modes,
                        enhanced_enforced_displacement_rom, manifold_from_derivatives, lifted_training_states,
                        ManifoldECSW, QuadraticManifoldROM, effective_speedup, break_even_queries)

n_elem=int(sys.argv[1]) if len(sys.argv)>1 else 100
n_sweep=int(sys.argv[2]) if len(sys.argv)>2 else 8
b=nonlinear_beam_model(n_elem=n_elem, n_modes=3)
K0,M,Phi=b["K0"],b["M"],b["Phi"]; n=b["n_free"]
h=np.sqrt(12*8.333e-9/1e-4); mid=int(np.argmax(abs(Phi[:,0])))
def newton(res, jac, x0, scale, tol=1e-7, max_it=40):
    """Newton with a relative residual tolerance; stops at the round-off floor (stagnation) instead of failing."""
    x=np.array(x0,float); prev=np.inf
    for it in range(max_it):
        r=res(x); rel=np.linalg.norm(r)/scale
        if rel<=tol or (it>=3 and rel<1e-5 and rel>=0.7*prev): return x
        prev=rel; x=x-np.linalg.solve(jac(x),r)
    raise RuntimeError("Newton did not converge")

def fom_static(F, steps=4, tol=1e-7):
    """Full-order load-stepped Newton on the same f_int / K_t callables the reduced models use."""
    u=np.zeros(n)
    for lam in np.linspace(0,1,steps+1)[1:]:
        u=newton(lambda x:b["f_int"](x)-lam*F, b["K_t"], u, np.linalg.norm(lam*F), tol)
    return u
b["static"]=fom_static
loads=[K0@(k*h/Phi[mid,0]*Phi[:,0]) for k in np.linspace(1.0,4.0,n_sweep)]

def best(fn, reps=3):
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); out=fn(); ts.append(time.perf_counter()-t)
    return min(ts), out

# full-order reference
t_fom, U_fom = best(lambda: [b["static"](F) for F in loads], reps=2)
t_fom_each=t_fom/len(loads)

off={}
t=time.perf_counter(); md=static_modal_derivatives(K0,Phi,tangent_fn=b["K_t"],length_scale=b["L"]); t_md=time.perf_counter()-t
t=time.perf_counter(); V,info=augment_with_derivatives(Phi,md,M=M); t_aug=time.perf_counter()-t
off["modal derivatives"]=t_md+t_aug
t=time.perf_counter(); D,dinfo=dual_modes(K0,M,Phi,b["static"],[0.5*h,3*h,6*h],dominant=0); off["dual modes"]=time.perf_counter()-t
# EED on V: probe coordinates at the sizes they actually take (derivative coordinates grow like the square of a mode amplitude)
scale=h/np.abs(V[:,0]).max()
Kr=V.T@K0@V
man0=manifold_from_derivatives(Phi,md,M)
def mode_scales(a):
    c=V.T@(M@man0.map(np.array([a,0.0,0.0])))          # V-coordinates of the lifted single-mode state
    sc=np.abs(c)/a; sc[:3]=[1.0,0.3,0.3]
    return sc
t=time.perf_counter(); eed,einfo=enhanced_enforced_displacement_rom(V,b["K_t"],[0.5*scale,1.0*scale,2.0*scale,4.0*scale],K_r=Kr,mode_scales=mode_scales); off["EED polynomial"]=time.perf_counter()-t+off["modal derivatives"]
# manifold + ECSW
man=manifold_from_derivatives(Phi,md,M)
amp=4*h/np.abs(Phi[:,0]).max()
Q,Ul=lifted_training_states(man,[amp,0.3*amp,0.3*amp],n_samples=40)
free=b["free"]
Phi_f=np.zeros((b["n"],3)); Phi_f[free]=Phi
Om_f=np.zeros((b["n"],3,3)); Om_f[free]=man.Omega
from rom_engine import QuadraticManifold
man_f=QuadraticManifold(Phi_f,Om_f)
t=time.perf_counter(); ecsw=ManifoldECSW.fit(man_f,Q,b["elem_dofs"],b["elem_force"],tol=1e-4); off["manifold ECSW"]=time.perf_counter()-t+off["modal derivatives"]-t_aug

def galerkin(W,F):
    return newton(lambda x:W.T@(b["f_int"](W@x)-F), lambda x:W.T@b["K_t"](W@x)@W, np.zeros(W.shape[1]), np.linalg.norm(W.T@F))

def err(Us): return max(np.linalg.norm(u-uf)/np.linalg.norm(uf) for u,uf in zip(Us,U_fom))

res={}
def run(name,fn,coords):
    try:
        t,Us=best(fn); res[name]=dict(coords=coords,t_each=t/len(loads),err=err(Us))
    except Exception as e:
        res[name]=dict(coords=coords,failed=str(e),t_each=float("nan"),err=float("nan"))

run("Galerkin, 3 modes (full force)", lambda:[Phi@galerkin(Phi,F) for F in loads],3)
run("Galerkin, modal derivatives (full force)", lambda:[V@galerkin(V,F) for F in loads],V.shape[1])
def eed_sweep():
    out=[];q=None
    for F in loads:
        r=V.T@F
        q=np.linalg.solve(Kr,r) if q is None else q
        q=newton(lambda x:Kr@x+eed.force(x)-r, lambda x:Kr+eed.jacobian(x), q, np.linalg.norm(r))
        out.append(V@q)
    return out
run("polynomial force (EED), modal-derivative basis", eed_sweep, V.shape[1])
man_full=QuadraticManifold(Phi_f,Om_f)
def man_ecsw():
    rom=QuadraticManifoldROM(man_f,hyper=ecsw); out=[]
    Ff=lambda F: np.pad(F,(0,0)) 
    for F in loads:
        Fm=np.zeros(b["n"]); Fm[free]=F
        q=rom.static_solve(Fm); out.append(man_f.map(q)[free])
    return out
run("quadratic manifold + ECSW", man_ecsw, 3)
res["_meta"]=dict(n_elem=n_elem,n_free=n,n_sweep=n_sweep,t_fom_each=t_fom_each,ecsw_selected=ecsw.n_selected,ecsw_total=ecsw.n_total,
                  offline=off, eed_probes=einfo["n_tangent_evals"], basis=V.shape[1], dual_solves=dinfo["n_solves"])
for name,r in res.items():
    if name=="_meta": continue
    o={"Galerkin, 3 modes (full force)":0.0,"Galerkin, modal derivatives (full force)":off["modal derivatives"],
       "polynomial force (EED), modal-derivative basis":off["EED polynomial"],"quadratic manifold + ECSW":off["manifold ECSW"]}[name]
    r["offline"]=o
    if "failed" in r: continue
    r["speedup_per_query"]=t_fom_each/r["t_each"]
    r["eff_speedup_M"]=effective_speedup(n_sweep,r["t_each"],o,t_fom_each)
    r["break_even"]=break_even_queries(r["t_each"],o,t_fom_each)
print(json.dumps(res,indent=1,default=float))
