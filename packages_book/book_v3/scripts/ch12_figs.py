"""Extra figures of Chapter 12 (F1: why Loewner draws fail; F3: harmonic content of the Duffing backbone; F2: the fitted restoring force and the training range; F4: where the operator-inference error sits; F5: error against load for three training ranges; F6: singular values of the Loewner pencil; V1: active modes and the polynomial floor; V7: settings as macros and the printed output of Listing 12.1). Run: PYTHONPATH=fea_engine/src:rom_engine/src:rom_engine/tests python packages_book/book_v3/scripts/ch12_figs.py
Writes figures/ch12_loewnerfail.pdf, figures/ch12_harm.pdf, figures/ch12_fnl.pdf, figures/ch12_opinfblocks.pdf, figures/ch12_range.pdf, figures/ch12_loewnersv.pdf and tables/numbers_ch12_fig.tex (macro prefix cnf, no digits in names). Same data and draws (seed 100+d) as scripts/ch12_assets.py, no noise."""
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
    rng=np.random.default_rng(100+d); rng.standard_normal(npool); rng.standard_normal(npool)   # the noise draws of ch12_assets.py exp_a (zero here) come before the choice
    ni=nm+3; idx=rng.choice(npool,2*ni,replace=False); ia,ib=idx[:ni],idx[ni:]
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
 "cnfDistOkMax":f"{dd[ok].max():.0f}","cnfOthersOk":"$%.1f\\times10^{%d}$"%(float(f"{E[ok][:,1:].max():.1e}".split("e")[0]),int(f"{E[ok][:,1:].max():.1e}".split("e")[1])),
 "cnfOthersFail":"%.2f"%E[fail][:,1:].max(),
 "cnfMissedFirst":str(int(np.sum(E[fail].argmax(1)==0)))}

import re
_T = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "numbers_ch12.tex")).read()
assert int(fail.sum()) == 30 - int(re.search(r"\\cldZeroFound\}\{(\d+)", _T).group(1))          # same draws as Table 12.3
assert int(np.sum(E[fail].argmax(1) == 0)) == int(fail.sum())
# hysteretic damping with the same modal damping ratios: the response is rational in omega^2
w2_, Phi_ = eigh(K, M); w_ = np.sqrt(w2_); zeta_ = (fx["alpha"] + fx["beta"] * w_ ** 2) / (2 * w_)
tipf = list(free).index(n_dof - 2); Fh = np.zeros(len(free)); Fh[tipf] = 1000.
xh = np.array([(Phi_ @ (1 / (w2_ * (1 + 2j * zeta_) - o ** 2) * (Phi_.T @ Fh)))[tipf] for o in op])
pole = np.sqrt(w2_ * (1 + 2j * zeta_)); fth = pole.real / 2 / np.pi; eth = pole.imag / pole.real; mh = (fth >= fmin) & (fth <= fmax); fth, eth = fth[mh], eth[mh]
okh = 0; deh = []
for d in range(30):
    rng = np.random.default_rng(100 + d); rng.standard_normal(npool); rng.standard_normal(npool); idx = rng.choice(npool, 2 * (nm + 3), replace=False); ia, ib = idx[:nm + 3], idx[nm + 3:]
    rom = LoewnerROM.fit(op[ia], op[ib], xh[ia], xh[ib]); inb = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1); fi, ei = rom.f[inb], rom.eta[inb]
    ef = [abs(fi[np.argmin(abs(fi - t))] - t) / t for t in fth]
    if max(ef) < .02: okh += 1; deh.append(max(abs(ei[np.argmin(abs(fi - t))] - e_) for t, e_ in zip(fth, eth)))
assert okh == 30
# single degree of freedom, viscous: the eigenvalue depends on the two samples
m_, k_, c_ = 1.0, 1e4, 4.0; xs = lambda w: 1 / (k_ - w ** 2 * m_ + 1j * w * c_)
def sdof(wa, wb): r_ = LoewnerROM.fit(np.array([wa]), np.array([wb]), np.array([xs(wa)]), np.array([xs(wb)])); return r_.eta[0]
zs = c_ / (2 * np.sqrt(k_ * m_)); zr = zs / np.sqrt(1 - zs ** 2)
NUMH = {"cnfZetaDiffPct": f"{100 * (1 / np.sqrt(1 - (et.max() / np.sqrt(1 + et.max() ** 2)) ** 2) - 1):.1f}", "cnfHystEta": "$%.1f\\times10^{%d}$" % (float(f"{np.median(deh):.1e}".split("e")[0]), int(f"{np.median(deh):.1e}".split("e")[1])),
        "cnfSdofZeta": f"{zr:.4f}", "cnfSdofWaOne": "90", "cnfSdofWbOne": "110", "cnfSdofWaTwo": "60", "cnfSdofWbTwo": "140",
        "cnfSdofEtaOne": f"{sdof(90., 110.):.4f}", "cnfSdofEtaTwo": f"{sdof(60., 140.):.4f}"}

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


# ---------------------------------------------------------------- F5: error against load for three training ranges
q1N = (fhb / Lamb)[0]
def run5(frac, ns=45):
    stg = a.AppliedLoadStrategy(frac, 0.01, ns, np.random.default_rng(0)); ql_, qn_, F_ = stg.generate(Vb, Mb, fb, fxb["mode_shape_peaks"], fxb["fom_solver"])
    ic = a.PolynomialModalROM(3).fit(qn_, F_); mf_ = a.MultiFidelitySurrogate().fit(ql_, F_); qp_ = np.zeros(3); ei_, em_ = [], []
    for Pq, u_ in zip(loadsb[1:], ufb[1:]):
        q_, _ = ic.predict(F_ext=Pq * fhb, Lambda=Lamb, q0=qp_); qp_ = q_; ref = u_[midb]
        ei_.append(abs((Vb @ q_)[midb] - ref) / abs(ref)); em_.append(abs((Vb @ mf_.reconstruct_q_nl(Pq * fhb / Lamb, Lamb))[midb] - ref) / abs(ref))
    return ql_[:, 0].max() / q1N, np.array(ei_), np.array(em_)
FR5 = ((0.3, 1.7), (0.3, 3.0), (0.3, 5.0)); r5 = [run5(fr_) for fr_ in FR5]
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
for a_, fr_, (eq_, ei_, em_) in zip(ax, FR5, r5):
    a_.semilogy(loadsb[1:], ei_, "C0o--", ms=3, label="polynomial"); a_.semilogy(loadsb[1:], em_, "C1s-.", ms=3, label="RBF"); a_.axvspan(0, eq_, color="0.9"); a_.set_xlabel("load (N)")
    a_.set_title(f"targets {fr_[0]:g}$-${fr_[1]:g}: cloud to {round(eq_, -2):.0f} N", fontsize=6.5)
ax[0].set_ylabel("relative error, mid-span"); ax[0].legend(fontsize=5.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "ch12_range.pdf"))
allice = np.concatenate([r_[1] for r_ in r5])
NUM.update({"cnfEqOne": f"{round(r5[0][0], -2):.0f}", "cnfEqTwo": f"{round(r5[1][0], -2):.0f}", "cnfEqThree": f"{round(r5[2][0], -2):.0f}",
            "cnfIceFlatLo": sci_(allice.min()), "cnfIceFlatHi": sci_(allice.max()), "cnfMfsInLo": sci_(r5[0][2][loadsb[1:] <= r5[0][0]].max()),
            "cnfMfsOutHi": sci_(r5[0][2][-1]), "cnfMfsWideEnd": sci_(r5[2][2][-1]), "cnfMfsWideStart": sci_(r5[2][2][0])})


# ---------------------------------------------------------------- F6: singular values of the Loewner matrix E (same data and draws as F1)
def pencilE(ia, ib, x):
    wa2, wb2 = op[ia] ** 2, op[ib] ** 2; return (x[ia][:, None] - x[ib][None, :]) / (wa2[:, None] - wb2[None, :])
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True); SV = {}
for a_, n_ in zip(ax, (7, 20)):
    for nz, c_ in ((0.0, "C0"), (1e-4, "C2"), (1e-2, "C3")):
        S_ = []
        for d_ in range(30):
            rng_ = np.random.default_rng(100 + d_); xn_ = xp + nz * np.abs(xp) * (rng_.standard_normal(npool) + 1j * rng_.standard_normal(npool)) / np.sqrt(2)   # noise first, as in ch12_assets.py
            idx_ = rng_.choice(npool, 2 * n_, replace=False); ia_, ib_ = idx_[:n_], idx_[n_:]
            sv_ = np.linalg.svd(pencilE(ia_, ib_, xn_), compute_uv=False); S_.append(sv_ / sv_[0])
        S_ = np.array(S_); k_ = np.arange(1, n_ + 1); SV[(n_, nz)] = np.median(S_, 0)
        a_.semilogy(k_, SV[(n_, nz)], "o-", ms=3, color=c_, label="noise 0" if nz == 0 else f"noise {nz:g}"); a_.fill_between(k_, S_.min(0), S_.max(0), color=c_, alpha=.15)
    a_.axvline(nm + .5, color="k", ls=":", lw=.8); a_.set_xlabel("index"); a_.set_title(f"$n={n_}$ per set", fontsize=7)
ax[0].set_ylabel("$\\sigma_k/\\sigma_1$ of $\\tilde E$"); ax[0].legend(fontsize=5.5); ax[0].set_ylim(1e-16, 2)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "ch12_loewnersv.pdf"))
m0 = SV[(7, 0.0)]
NUM.update({"cnfSvTwo": sci_(m0[1]), "cnfSvThree": sci_(m0[2]), "cnfSvFour": sci_(m0[3]), "cnfSvFive": sci_(m0[4]), "cnfSvTailTwenty": sci_(SV[(20, 1e-2)][9])})


# ---------------------------------------------------------------- V1: active modes under the central load, 5 and 6 modes, low-load errors
def modes_run(nm_, ns_):
    fx_ = a.clamped(N_E, nm_); V_, M_, f_, free_ = fx_["V"], fx_["M_ff"], fx_["freq_hz"], fx_["free_dofs"]; Lam_ = (2 * np.pi * f_) ** 2
    mid_ = list(free_).index((N_E // 2) * 3 + 1); sh_ = np.zeros(len(free_)); sh_[mid_] = -1; fh_ = V_.T @ sh_
    stg = a.AppliedLoadStrategy((0.3, 3.0), 0.01, ns_, np.random.default_rng(0)); ql_, qn_, F_ = stg.generate(V_, M_, f_, fx_["mode_shape_peaks"], fx_["fom_solver"])
    ic = a.PolynomialModalROM(nm_).fit(qn_, F_); qp_ = np.zeros(nm_); ei_, el_ = [], []
    for Pq in loadsb[1:]:
        u_ = fx_["fom_solver"](Pq * sh_, n_steps=30); q_, _ = ic.predict(F_ext=Pq * fh_, Lambda=Lam_, q0=qp_); qp_ = q_
        ei_.append(abs((V_ @ q_)[mid_] - u_[mid_]) / abs(u_[mid_])); el_.append(abs((V_ @ (Pq * fh_ / Lam_))[mid_] - u_[mid_]) / abs(u_[mid_]))
    return fh_, np.array(ei_), np.array(el_), len(ic.quad_idx) + len(ic.cub_idx)
fh3, ei3, el3, _ = modes_run(3, 45); fh4, _, _, _ = modes_run(4, 45)
fh5, ei5, el5, nt5 = modes_run(5, 90); _, ei5s, _, _ = modes_run(5, 45); _, ei6, _, nt6 = modes_run(6, 45)
assert abs(fh4[1]) < 1e-10 * abs(fh4[0]) and abs(fh4[3]) < 1e-10 * abs(fh4[0])       # modes 2 and 4 are inactive under the central load
NUM.update({"cnfLoadImg": f"{abs(fh3[2] / fh3[0]):.2f}", "cnfTermsFive": str(nt5), "cnfTermsSix": str(nt6),
            "cnfIceFiveFortyFive": sci_(ei5s.max()), "cnfIceFiveNinety": sci_(ei5.max()), "cnfIceSixFortyFive": f"{ei6.max():.1f}",
            "cnfLowLoad": f"{loadsb[1]:.0f}", "cnfLowIceThree": sci_(ei3[0]), "cnfLowLinThree": sci_(el3[0]), "cnfLowIceFive": sci_(ei5[0]), "cnfLowLinFive": sci_(el5[0])})


# ---------------------------------------------------------------- Exercise 2: samples chosen from the response instead of at random (noise-free, viscous damping)
def loewner_errors(idx):
    idx = np.array(sorted(idx)); ia, ib = idx[0::2], idx[1::2]; n_ = min(len(ia), len(ib)); ia, ib = ia[:n_], ib[:n_]
    rom = LoewnerROM.fit(op[ia], op[ib], xp[ia], xp[ib]); inb = (rom.f >= fmin) & (rom.f <= fmax) & (rom.eta >= 0) & (rom.eta <= 1); fi = rom.f[inb]
    return [min(abs(fi - t)) / t if len(fi) else 9.0 for t in ft]
NSAMP = 2 * (nm + 3); mag = abs(xp)
large = list(np.argsort(-mag)[:NSAMP]); errL = loewner_errors(large)
pk_ = [i for i in range(1, npool - 1) if mag[i] > mag[i - 1] and mag[i] > mag[i + 1]]
sel = sorted({j for i in pk_ for j in (i - 1, i, i + 1)}); rest = [i for i in range(npool) if i not in sel]
sel = sel + [rest[j] for j in np.linspace(0, len(rest) - 1, NSAMP - len(sel)).astype(int)]; errP = loewner_errors(sel)
assert max(errL) > 0.02 and max(errP) < 0.02 and len(pk_) == nm
NUMX = {"cnfExLargeMax": f"{fp[large].max():.0f}", "cnfExLargeErrThree": sci_(errL[2]), "cnfExLargeErrFour": sci_(errL[3]), "cnfExPeakErr": sci_(max(errP))}
NUM.update(NUMH); NUM.update(NUMX)

# ---------------------------------------------------------------- V7: settings typed in the text become macros (asserted against ch12_assets.py), small-load sweep, printed output of the listing
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch12_assets.py")).read()
for _frag in ("n_steps=30", "np.linspace(0.005, 0.03, 6)", "noises = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)", "for nz in (0.0, 1e-3, 1e-2, 1e-1, 5e-1)", "c = np.array([-0.02]); h, steps = 2e-3, 1500",
              "f[N - 1] = -0.3", "(1e-4, 0.01)", "max(ef) < 0.02", "1e-3 * rng0.standard_normal(n_pool)", "damped_cantilever_beam_system(n=20)", "for ns in (10, 20, 45, 90)"):
    assert _frag in _src, _frag
def sci10(x):
    m_, e_ = f"{x:.0e}".split("e"); return f"10^{{{int(e_)}}}" if m_ == "1" else f"{x:g}"
loads_small = np.linspace(0, 900, 7)[1:]; qp_ = np.zeros(3); eis, els = [], []
icS = a.PolynomialModalROM(3).fit(rd[45][1], rd[45][2]); Kff_ = np.asarray(fxb["K"])[np.ix_(freeb, freeb)]
for Pq in loads_small:
    u_ = fxb["fom_solver"](Pq * shb, n_steps=30); q_, _ = icS.predict(F_ext=Pq * fhb, Lambda=Lamb, q0=qp_); qp_ = q_
    eis.append(abs((Vb @ q_)[midb] - u_[midb]) / abs(u_[midb])); els.append(abs((Vb @ (Pq * fhb / Lamb))[midb] - u_[midb]) / abs(u_[midb]))
NUMV = {"cnfNSamMax": "90", "cnfNSamA": "10", "cnfNSamB": "20", "cnfNElemCant": "20", "cnfJitterPct": "0.1", "cnfSuccTol": "2", "cnfNIncr": "30",
        "cnfSmallMax": f"{loads_small[-1]:.0f}", "cnfSmallDefl": f"{abs(u_[midb]) * 1e3:.1f}", "cnfSmallLin": sci_(max(els)), "cnfSmallIce": sci_(max(eis)),
        "cnfRatioMfsIce": f"{rd[45][6] / rd[45][5]:.0f}", "cnfNAmps": "6", "cnfAmpLo": "0.005", "cnfBeamAmp": "0.01", "cnfActive": "2",
        "cnfLwNzA": sci10(1e-4), "cnfLwNzB": sci10(1e-3), "cnfLwNzC": sci10(1e-2), "cnfLwNzD": sci10(1e-1),
        "cnfOiNzA": sci10(1e-3), "cnfOiNzB": sci10(1e-2), "cnfOiNzC": sci10(1e-1), "cnfOiNzD": sci10(5e-1), "cnfOiModes": str(toi.R),
        "cnfObst": "-0.02", "cnfForce": "-0.3", "cnfStepH": "2\\times10^{-3}", "cnfSteps": "1500",
        "cnfPeakOne": f"{res[1][3]:.5f}", "cnfPeakThree": f"{res[3][3]:.5f}", "cnfPeakSeven": f"{res[7][3]:.5f}"}
assert abs(u_[midb] * 1e3 + 2.67) < 0.05 and toi.R == 10 and toi.N == 30
# the listing of Sec. 12.3 is run exactly as printed in ch12.tex and its output is what the chapter shows
import io, contextlib, re as _re
_tex = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapters", "ch12.tex"), encoding="utf-8").read()
_m = _re.search(r"\\begin\{lstlisting\}\[caption=\{A cubic oscillator[^\n]*\n(.*?)\\end\{lstlisting\}", _tex, _re.S)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    exec(compile(_m.group(1), "listing", "exec"), {"__name__": "listing"})
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "ch12_try_output.txt"), "w").write(_buf.getvalue())
NUM.update(NUMV)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","tables","numbers_ch12_fig.tex"),"w") as fh:
    fh.write("% generated by scripts/ch12_figs.py -- do not edit\n")
    for k,v in NUM.items(): fh.write("\\newcommand{\\%s}{%s\\xspace}\n"%(k,v))
print(NUM)
