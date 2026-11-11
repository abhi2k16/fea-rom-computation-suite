"""Extra figures of Chapter 12 (F1: why Loewner draws fail; F3: harmonic content of the Duffing backbone; F2: the fitted restoring force and the training range; F4: where the operator-inference error sits). Run: PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests python packages_book/book_v3/scripts/ch12_figs.py
Writes figures/ch12_loewnerfail.pdf, figures/ch12_harm.pdf, figures/ch12_fnl.pdf, figures/ch12_opinfblocks.pdf and tables/numbers_ch12_fig.tex (macro prefix cnf, no digits in names). Same data and draws (seed 100+d) as scripts/ch12_assets.py, no noise."""
import os
import importlib.util,numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
spec=importlib.util.spec_from_file_location("a",os.path.join(os.path.dirname(os.path.abspath(__file__)),"ch12_assets.py")); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
ff,lf,LoewnerROM,eigh=a.ff,a.lf,a.LoewnerROM,a.eigh
fx=ff.damped_cantilever_beam_system(n=20); free=fx["free_dofs"]; ix=np.ix_(free,free)
K,M,C=fx["K"][ix],fx["M"][ix],fx["C"][ix]; sysobj,n_dof=fx["sys"],fx["n_dof"]
wn=np.sqrt(np.abs(eigh(K,M,eigvals_only=True))); fu=wn/2/np.pi; fmin,fmax=.7*fu[0],1.3*fu[3]
ft,et,_=lf.ground_truth_modes(M,C,K,fx["alpha"],fx["beta"],fmin,fmax); nm=len(ft)
tip=n_dof-2; F=np.zeros(n_dof); F[tip]=1000.
r0=np.random.default_rng(7); npool=100; fp=np.linspace(fmin,fmax,npool)*(1+1e-3*r0.standard_normal(npool)); op=2*np.pi*fp
xp=np.array([sysobj.solve_harmonic(w,F)[tip] for w in op])
fd=np.geomspace(fmin,fmax,1500); xd=np.array([sysobj.solve_harmonic(2*np.pi*f,F)[tip] for f in fd])
E=np.zeros((30,nm)); sets=[]
for d in range(30):
    rng=np.random.default_rng(100+d); ni=nm+3; idx=rng.choice(npool,2*ni,replace=False); ia,ib=idx[:ni],idx[ni:]
    rom=LoewnerROM.fit(op[ia],op[ib],xp[ia],xp[ib]); inb=(rom.f>=fmin)&(rom.f<=fmax)&(rom.eta>=0)&(rom.eta<=1); fi=rom.f[inb]
    E[d]=[min(abs(fi-t))/t if len(fi) else 1 for t in ft]; sets.append((ia,ib))
fail=E.max(1)>=.02
dist=[min(abs(fp[np.r_[s[0],s[1]]]-ft[0])) for s in sets]; hw=et[0]*ft[0]/2
dd=np.array(dist); ok=~fail; _=("dist to mode1 (Hz) fail:",np.round(np.array(dist)[fail],1)," ok min/median:",np.round(min(np.array(dist)[~fail]),1),np.round(np.median(np.array(dist)[~fail]),1)," half-width",round(hw,2), "spacing",round(np.median(np.diff(fp)),1))
d0=int(np.where(fail)[0][0]); ia,ib=sets[d0]
plt.rcParams.update({"font.size":7,"axes.grid":True,"grid.alpha":.3})
fig,ax=plt.subplots(1,2,figsize=(6.3,2.7),gridspec_kw={"width_ratios":[1.4,1]})
ax[0].loglog(fd,abs(xd),"k-",lw=.8,label="response")
ax[0].plot(fp[ia],abs(xp[ia]),"o",ms=3,color="C0",label="set $\\alpha$"); ax[0].plot(fp[ib],abs(xp[ib]),"s",ms=3,color="C1",label="set $\\beta$")
for t in ft: ax[0].axvline(t,color="C3",ls=":",lw=.8)
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("tip response"); ax[0].legend(fontsize=6,loc="lower left"); ax[0].set_title("(a) one failing draw",fontsize=7)
rng=np.random.default_rng(0)
for j in range(nm):
    x=j+1+.12*rng.standard_normal(30)*0+np.linspace(-.25,.25,30)
    ax[1].semilogy(x[~fail],E[~fail,j],"o",ms=2.5,color="C0"); ax[1].semilogy(x[fail],E[fail,j],"o",ms=3,color="C3")
ax[1].axhline(.02,color="k",ls="--",lw=.8); ax[1].set_xticks(range(1,nm+1)); ax[1].set_xlabel("mode"); ax[1].set_ylabel("relative frequency error"); ax[1].set_title("(b) 30 draws, no noise",fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures","ch12_loewnerfail.pdf"))

NUM={"cnfNFail":str(int(fail.sum())),"cnfFOne":f"{ft[0]:.0f}","cnfHalfWidth":f"{hw:.1f}","cnfGridHz":f"{np.median(np.diff(fp)):.0f}",
 "cnfNSamp":str(2*(nm+3)),"cnfDistFailLo":f"{dd[fail].min():.0f}","cnfDistFailHi":f"{dd[fail].max():.0f}","cnfDistOkMed":f"{np.median(dd[ok]):.0f}",
 "cnfOthersMax":"$%.1f\\times10^{%d}$"%tuple([float(f"{E[:,1:].max():.1e}".split("e")[0]),int(f"{E[:,1:].max():.1e}".split("e")[1])]),
 "cnfMissedFirst":str(int(np.sum(E[fail].argmax(1)==0)))}

# ---------------------------------------------------------------- F3: harmonic content of the Duffing solution
from scipy.integrate import solve_ivp
from rom_engine.nnm import HarmonicBalanceSystem, solve_nnm_backbone
lam3, k33, a3 = 100., 5e4, 0.03
class Cub:
    def predict(self, q): return k33 * np.asarray(q) ** 3
    def jacobian(self, q): return np.diag(3 * k33 * np.asarray(q) ** 2)
def meas(x):
    st = lambda t, y: y[1]; st.direction = 1
    r_ = solve_ivp(lambda t, y: [y[1], -lam3 * y[0] - k33 * y[0] ** 3], [0, 3.2], [x, 0], events=st, rtol=1e-12, atol=1e-14)
    return 2 * np.pi / np.mean(np.diff(r_.t_events[0]))
res = {}
for H in (1, 3, 5, 7):
    hb = HarmonicBalanceSystem(np.array([lam3]), n_harmonics=H, n_time_samples=max(128, 8 * H + 8))
    am, om, Z, ok = solve_nnm_backbone(hb, Cub(), (0.005, a3), n_points=6, domain="q_nl")
    z = Z[-1, :, 0]; peak = z[0] + sum(z[2 * k - 1] for k in range(1, H + 1)); wr = meas(peak)
    res[H] = (z, om[-1], abs(om[-1] - wr) / wr, peak)
z7, w7, _, pk = res[7]; c = z7[1::2]
tau = np.linspace(0, 2 * np.pi, 400); q7 = z7[0] + sum(c[k - 1] * np.cos(k * tau) for k in range(1, 8)); q1 = c[0] * np.cos(tau)
r_ = solve_ivp(lambda t, y: [y[1], -lam3 * y[0] - k33 * y[0] ** 3], [0, 2 * np.pi / w7], [pk, 0], t_eval=np.linspace(0, 2 * np.pi / w7, 25), rtol=1e-12, atol=1e-14)
qi = np.interp(r_.t * w7, tau, q1)
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax[0].plot(tau / np.pi, 100 * (q7 - q1) / pk, "C0-", label="7 harmonics"); ax[0].plot(r_.t * w7 / np.pi, 100 * (r_.y[0] - qi) / pk, "ko", ms=2.5, label="time integration")
ax[0].set_xlabel("$\\omega t/\\pi$"); ax[0].set_ylabel("(q $-$ first harmonic) / q(0)  (%)"); ax[0].legend(fontsize=5.5, loc="lower center"); ax[0].set_title("(a) what one harmonic misses", fontsize=7)
ax[1].bar(["1", "3", "5"], [abs(c[k - 1]) / abs(c[0]) for k in (1, 3, 5)], color="C0"); ax[1].set_yscale("log"); ax[1].set_xlabel("harmonic"); ax[1].set_ylabel("amplitude / first"); ax[1].set_title("(b) harmonic content", fontsize=7)
ax[2].semilogy([1, 3, 5, 7], [res[k][2] for k in (1, 3, 5, 7)], "C2o-"); ax[2].set_xlabel("harmonics in the model"); ax[2].set_ylabel("relative frequency error"); ax[2].set_xticks([1, 3, 5, 7]); ax[2].set_title("(c) error of the backbone", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "ch12_harm.pdf"))
def sci_(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
NUM.update({"cnfHarmDiff": f"{100 * np.max(abs(q7 - q1)) / pk:.1f}", "cnfHarmThree": sci_(abs(c[2]) / abs(c[0])), "cnfHarmFive": sci_(abs(c[4]) / abs(c[0])),
            "cnfErrFive": sci_(res[5][2]), "cnfHarmIntMax": sci_(np.max(abs(r_.y[0] - np.interp(r_.t * w7, tau, q7))) / pk)})


# ---------------------------------------------------------------- F2: the fitted restoring force and the training range
N_E = 16; fxb = a.clamped(N_E, 3); Vb, Mb, fb, freeb = fxb["V"], fxb["M_ff"], fxb["freq_hz"], fxb["free_dofs"]
Lamb = (2 * np.pi * fb) ** 2; midb = list(freeb).index((N_E // 2) * 3 + 1); shb = np.zeros(len(freeb)); shb[midb] = -1; fhb = Vb.T @ shb
loadsb = np.linspace(0, 6000, 13); ufb = [fxb["fom_solver"](Pq * shb, n_steps=30) for Pq in loadsb]
qlt = np.array([Pq * fhb / Lamb for Pq in loadsb]); qft = np.array([Vb.T @ (Mb @ u_) for u_ in ufb])
def run(frac, ns):
    stg = a.AppliedLoadStrategy(frac, 0.01, ns, np.random.default_rng(0)); ql_, qn_, F_ = stg.generate(Vb, Mb, fb, fxb["mode_shape_peaks"], fxb["fom_solver"])
    ic = a.PolynomialModalROM(3).fit(qn_, F_); mf_ = a.MultiFidelitySurrogate().fit(ql_, F_); qp_ = np.zeros(3); ei_, em_, qi_, qm_ = [], [], [], []
    for Pq, u_ in zip(loadsb[1:], ufb[1:]):
        q_, _ = ic.predict(F_ext=Pq * fhb, Lambda=Lamb, q0=qp_); qp_ = q_; qm0 = mf_.reconstruct_q_nl(Pq * fhb / Lamb, Lamb); qi_.append(q_); qm_.append(qm0)
        ref = u_[midb]; ei_.append(abs((Vb @ q_)[midb] - ref) / abs(ref)); em_.append(abs((Vb @ qm0)[midb] - ref) / abs(ref))
    return ql_, qn_, F_, np.array(qi_), np.array(qm_), max(ei_), max(em_)
DEF, WIDE, NS = (0.3, 3.0), (0.3, 5.0), (10, 20, 45, 90)
rd = {n_: run(DEF, n_) for n_ in NS}; rw = {n_: run(WIDE, n_) for n_ in NS}
ql, qn, Fd, qi, qm, _, _ = rd[45]
assert (ql > 0).all() and abs(qlt[:, 1]).max() < 1e-12 and qlt[-1, 2] < 0      # test sweep: q_l2 = 0, q_l3 < 0, training all positive
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.6))
ax[0].plot(ql[:, 0] * 1e3, ql[:, 2] * 1e3, "o", ms=2.5, color="0.5", label="training"); ax[0].plot(qlt[:, 0] * 1e3, qlt[:, 2] * 1e3, "C3-", label="test sweep")
ax[0].set_xlabel("$q_{l,1}$ (mm)"); ax[0].set_ylabel("$q_{l,3}$ (mm)"); ax[0].legend(fontsize=5.5); ax[0].set_title("(a) training loads", fontsize=7)
fnl_t = Lamb * (qlt - qft); qip = np.vstack([np.zeros(3), qi]); qmp = np.vstack([np.zeros(3), qm])
ax[1].plot(qn[:, 0] * 1e3, Fd[:, 0], ".", ms=3, color="0.6", label="training"); ax[1].plot(qft[:, 0] * 1e3, fnl_t[:, 0], "k-", label="full order")
ax[1].plot(qip[:, 0] * 1e3, (Lamb * (qlt - qip))[:, 0], "C0--", label="polynomial"); ax[1].plot(qmp[:, 0] * 1e3, (Lamb * (qlt - qmp))[:, 0], "C1-.", label="RBF")
ax[1].set_xlabel("$q_{nl,1}$ (mm)"); ax[1].set_ylabel("$f_{nl,1}$"); ax[1].legend(fontsize=5.5); ax[1].set_title("(b) the fitted force", fontsize=7)
ax[2].semilogy(NS, [rd[n_][5] for n_ in NS], "C0o--", label="polynomial"); ax[2].semilogy(NS, [rd[n_][6] for n_ in NS], "C1s-.", label="RBF, default range"); ax[2].semilogy(NS, [rw[n_][6] for n_ in NS], "C2^-", label="RBF, wider range")
ax[2].set_xlabel("training solves"); ax[2].set_ylabel("largest error, mid-span"); ax[2].legend(fontsize=5.5); ax[2].set_title("(c) more data", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "ch12_fnl.pdf"))
NUM.update({"cnfRangeLo": f"{DEF[0]:g}", "cnfRangeHi": f"{DEF[1]:g}", "cnfWideHi": f"{WIDE[1]:g}",
            "cnfTrMax": f"{rd[45][0][:, 0].max() * 1e3:.1f}", "cnfTestMax": f"{qlt[-1, 0] * 1e3:.1f}", "cnfWideMax": f"{rw[45][0][:, 0].max() * 1e3:.1f}",
            "cnfTrQnl": f"{rd[45][1][:, 0].max() * 1e3:.1f}", "cnfTestQnl": f"{qft[-1, 0] * 1e3:.1f}",
            "cnfWideMfsTen": sci_(rw[10][6]), "cnfWideMfsFortyFive": sci_(rw[45][6]), "cnfWideMfsNinety": sci_(rw[90][6]), "cnfWideIceNinety": sci_(rw[90][5])})


# ---------------------------------------------------------------- F4: where the operator-inference error sits
toi, craig, infer = a.toi, a.craig_bampton_transformation, a.infer_contact_operators
Ks, Ms = toi.system(); NN, BND, RR = toi.N, toi.BND, toi.R; nb = len(BND)
itr = np.setdiff1d(np.arange(NN), BND); Kii, Mii, Kib = Ks[np.ix_(itr, itr)], Ms[np.ix_(itr, itr)], Ks[np.ix_(itr, BND)]
w2_, V_ = a.eigh(Kii, Mii); Phi_, _ = np.linalg.qr(V_[:, :RR]); Psi_ = -np.linalg.solve(Kii, Kib)
U_, A_, F_ = toi.training_data(Ks, Ms, Phi_); T_ = craig(BND, NN, Phi_, Psi_); Mg, Kg = T_.T @ Ms @ T_, T_.T @ Ks @ T_
mods = {}
for nz in (0.0, 1e-2):
    An_ = A_ + nz * np.abs(A_).max() * np.random.default_rng(5).standard_normal(A_.shape); mods[nz] = infer(U_, An_, F_, BND, Phi_, Psi_, max_nfev=60)
def blocks(nz, key, Xg):
    d = getattr(mods[nz], key) - Xg; n_ = np.linalg.norm(Xg)
    return [np.linalg.norm(d[:nb, :nb]) / n_, np.linalg.norm(d[:nb, nb:]) / n_, np.linalg.norm(d[nb:, nb:]) / n_]
fig = plt.figure(figsize=(6.3, 2.8)); gs4 = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 0.07], hspace=0.55)
ax = [fig.add_subplot(gs4[0, 0]), fig.add_subplot(gs4[0, 1]), fig.add_subplot(gs4[0, 2])]; cax = fig.add_subplot(gs4[1, 0:2])
Kmax = np.abs(Kg).max(); L_ = lambda X: np.log10(np.abs(X) / Kmax + 1e-16)
im = ax[0].imshow(L_(Kg), vmin=-8, vmax=0, cmap="viridis"); ax[0].set_title("(a) $K$ of the Galerkin model", fontsize=6.5)
ax[1].imshow(L_(mods[1e-2].K - Kg), vmin=-8, vmax=0, cmap="viridis"); ax[1].set_title("(b) error of inferred $K$, noise $10^{-2}$", fontsize=6.5)
for c_ in range(2):
    ax[c_].axhline(nb - .5, color="w", lw=.6); ax[c_].axvline(nb - .5, color="w", lw=.6); ax[c_].set_xticks([0, 2, 6, 11]); ax[c_].set_yticks([0, 2, 6, 11])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("$\\log_{10}(|\\cdot|/\\max|K|)$", fontsize=6)
x_ = np.arange(6); b0 = blocks(0.0, "M", Mg) + blocks(0.0, "K", Kg); b1 = blocks(1e-2, "M", Mg) + blocks(1e-2, "K", Kg)
ax[2].bar(x_ - .2, b0, .4, label="exact data"); ax[2].bar(x_ + .2, b1, .4, label="noise $10^{-2}$"); ax[2].set_yscale("log"); ax[2].set_xticks(x_)
ax[2].set_xticklabels(["M bb", "M bq", "M qq", "K bb", "K bq", "K qq"], fontsize=6); ax[2].legend(fontsize=5.5); ax[2].set_ylabel("block error / $\\|X\\|_F$"); ax[2].set_title("(c) where the error sits", fontsize=6.5)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.14, wspace=0.35); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "ch12_opinfblocks.pdf"))
assert np.linalg.norm(Kg[:nb, nb:]) < 1e-10 * np.linalg.norm(Kg)                       # the coupling block of K is zero by construction
NUM.update({"cnfKbbZero": sci_(b0[3]), "cnfKbqZero": sci_(b0[4]), "cnfKqqZero": sci_(b0[5]), "cnfKbbNoise": sci_(b1[3]), "cnfKbqNoise": sci_(b1[4]), "cnfKqqNoise": sci_(b1[5]),
            "cnfMbbZero": sci_(b0[0]), "cnfMbbShare": f"{100 * np.linalg.norm(Mg[:nb, :nb]) / np.linalg.norm(Mg):.0f}"})

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tables","numbers_ch12_fig.tex"),"w") as fh:
    fh.write("% generated by scripts/ch12_figs.py -- do not edit\n")
    for k,v in NUM.items(): fh.write("\\newcommand{\\%s}{%s\\xspace}\n"%(k,v))
print(NUM)
