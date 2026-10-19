# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
ch14_assets.py -- figures and number macros of Chapter 14 (reproduction studies).

The reproduction drivers are NOT part of the repository: they are scripts in the author's study folders (one folder per
reproduced paper, see STUDIES). This script therefore does two separate things.

  default   REDRAW. Read the saved outputs (npz / json / csv) of the study folders and redraw the chapter's figures in the
            book's style; compute every number quoted in a caption or in the text from those saved arrays (never typed
            in), and compare a few of them with the numbers of the written reports. Takes seconds.
  --rerun   RE-RUN. Copy a study folder to a scratch directory (the author's results are never overwritten), run its
            driver commands there with `--jobs` worker processes (single-threaded BLAS in each worker), then redraw from the
            scratch copy. The commands are the ones of the reports; they take minutes to hours (see RERUN) and were
            executed in full only on the author's 12-core machine. They need numpy, scipy and matplotlib, and the two
            packages on PYTHONPATH.

Writes into book/: figures/ch14_*.pdf; tables/numbers_ch14.tex.

Usage (from packages_book/book/scripts):
    python ch14_assets.py                                  # redraw everything that has saved data
    python ch14_assets.py --only mfs,ice                   # a subset
    python ch14_assets.py --study-root D:/Physical_ML/ROM_formulations
    python ch14_assets.py --rerun --only mfs --jobs 10     # re-run, then redraw (multi-core)
Studies without saved numeric data (the fluid-structure study keeps only images) are listed in NOT_REDRAWN and are quoted
from their report in the chapter.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
FIG, TAB = os.path.join(BOOK, "figures"), os.path.join(BOOK, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, VIOLET, GREEN, ORANGE, GREY = "#2b4c9b", "#6a3d9a", "#2e8b57", "#d2691e", "#555555"
NUM, CHECKS = {}, []
NOT_REDRAWN = {"fsi": "plate identification (images and summary only)", "fdpinn_net": "network training (needs PyTorch and a GPU)"}

# study folder (below --study-root) and the commands of its report (run inside the folder), with the time on the author's machine
STUDIES = {"mfs": "Multi_Fidelity_NL_Structural_ROM", "hyrom": "NonLin-HyROM", "ice": "ICE-ROM/validation",
           "fdpinn": "FD-PINN_dynamics", "hyb": "contact-hybrid-rom/repro", "asm": "MOR_struct_contact_geomNL/assembly2d",
           "bal": "projection_MOR_contact/validation", "frino": "FE_ROM_Neural_Op/frino_repro"}
RERUN = {"mfs": [["flat_beam_fem.py"], ["mfs_nlrom_reproduction.py"]],                           # ~10 s, ~130 s on 10 workers
         "ice": [["ice_rom_fit.py"], ["reproduce_flat_beam.py"]],                                  # ~8 min
         "hyb": [["rope_paper_repro.py", "--skip-a", "--contact-margin", "1", "--eps-u", "1e-10", "--layers", "3", "--tag", "_L3e10"]],   # ~5 min
         "bal": [["v0_checks.py"], ["v2_dynamic.py", "40"], ["v3_twobody.py", "24"]]}              # 1, 20, 10 min


def num(k, v):
    NUM[k] = v


def sci(x):
    if x == 0:
        return "0"
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(float(x))).normalize(); e = d.adjusted(); m = (d.scaleb(-e)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if m >= 10:
        m, e = Decimal("1.0"), e + 1
    return f"${m}\\times10^{{{e}}}$"


def pct(x, n=2):
    return f"{x:.{n}g}\\,\\%"


def check(what, mine, report, tol):
    """record a comparison of a number computed here with the number printed in the written report"""
    ok = abs(mine - report) <= tol * max(abs(report), 1e-300)
    CHECKS.append((what, mine, report, ok))


def r2(ref, x):
    return 1.0 - np.sum((ref - x) ** 2) / np.sum((ref - ref.mean()) ** 2)


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name, flush=True)


def rerun(study, root, jobs, scratch):
    src = os.path.join(root, STUDIES[study]); parts = STUDIES[study].split("/")
    dst = os.path.join(scratch, parts[0])
    if not os.path.exists(dst):
        shutil.copytree(os.path.join(root, parts[0]), dst, ignore=shutil.ignore_patterns("*.pdf", "__pycache__", "archive*"))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MFS_NWORKERS=str(jobs))
    wd = os.path.join(scratch, STUDIES[study])
    for cmd in RERUN[study]:
        print("run", study, cmd, flush=True)
        subprocess.run([sys.executable, "-u", *cmd], cwd=wd, env=env, check=True)
    return wd


# ------------------------------------------------------------------------------------------------ MFS-NLROM (He et al.)
def fig_mfs(d):
    z = np.load(os.path.join(d, "results", "mfs_nlrom_reproduction_results.npz"), allow_pickle=True)
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.2))
    ax[0].plot(z["Pvals"], z["w_lin"] * 1e3, "--", color=GREY, label="linear")
    ax[0].plot(z["Pvals"], z["w_nl_fe"] * 1e3, "-", color="k", lw=2.5, alpha=0.35, label="nonlinear FE")
    ax[0].plot(z["Pvals"], z["w_mfs"] * 1e3, "o", color=VIOLET, ms=3.5, label="MFS-NLROM")
    ax[0].plot(z["Pvals"], z["w_ice"] * 1e3, "x", color=ORANGE, ms=4, label="ICE")
    ax[0].set(xlabel="pressure $P$ (Pa)", ylabel="centre displacement (mm)", title="(a) pressure sweep"); ax[0].legend(fontsize=7)
    t = np.arange(z["q_nl_fe_case_a"].shape[0])
    ax[1].plot(t, z["q_nl_fe_case_a"][:, 0] * 1e3, color="k", lw=2.5, alpha=0.35, label="FE")
    ax[1].plot(t, z["q_nl_mfs_case_a"][:, 0] * 1e3, color=VIOLET, lw=1, label="MFS-NLROM")
    ax[1].plot(t, z["q_nl_ice_case_a"][:, 0] * 1e3, "--", color=ORANGE, lw=1, label="ICE")
    ax[1].set(xlabel="time step", ylabel="modal coordinate $q_1$ ($10^{-3}$)", title="(b) free decay, case a"); ax[1].legend(fontsize=7)
    a = z["amps_backbone"] * 1e6
    ax[2].plot(a, z["omega_fe_backbone"] / 2 / np.pi, "s-", color="k", ms=4, label="FE free vibration")
    ax[2].plot(a, z["omega_mfs_backbone"] / 2 / np.pi, "o--", color=VIOLET, ms=4, label="MFS-NLROM (HBM)")
    ax[2].plot(a, z["omega_ice_backbone"] / 2 / np.pi, "x:", color=ORANGE, ms=5, label="ICE (HBM)")
    ax[2].set(xlabel="master-mode amplitude ($10^{-6}$)", ylabel="frequency (Hz)", title="(c) backbone"); ax[2].legend(fontsize=7)
    save(fig, "ch14_mfs.pdf")
    f1 = z["freq_hz"]; num("cvfMfsRtwoMfs", f"{float(z['r2_static_mfs']):.4f}"); num("cvfMfsRtwoIce", f"{float(z['r2_static_ice']):.4f}")
    ra = r2(z["q_nl_fe_case_a"][:, 0], z["q_nl_mfs_case_a"][:, 0]); rb = r2(z["q_nl_fe_case_b"][:, 0], z["q_nl_mfs_case_b"][:, 0])
    num("cvfMfsDecayA", f"{ra:.4f}"); num("cvfMfsDecayB", f"{rb:.4f}"); check("MFS free decay R2 case a", ra, 0.9971, 1e-3); check("MFS free decay R2 case b", rb, 0.9898, 1e-3)
    be = np.max(np.abs(z["omega_mfs_backbone"] / z["omega_fe_backbone"] - 1)) * 100
    wl = 2 * np.pi * f1[0]; hard = (z["omega_fe_backbone"][-1] / wl - 1) * 100
    num("cvfMfsBackErr", pct(be)); num("cvfMfsHardening", pct(hard, 3)); check("MFS backbone error (%)", be, 0.96, 0.02); check("MFS hardening range (%)", hard, 22, 0.05)
    sp = float(z["t_fe_mean"] / z["t_mfs_mean"]); num("cvfMfsSpeed", sci(sp)); check("MFS static speed ratio", sp, 1.7e4, 0.05)
    num("cvfMfsTfe", f"{float(z['t_fe_mean']):.2f}"); num("cvfMfsTmfs", sci(float(z["t_mfs_mean"])))


# ------------------------------------------------------------------------------------------------ hybrid modal ROM (Yang et al.)
def fig_hyrom(d):
    c2 = json.load(open(os.path.join(d, "results_case2", "case2_backbone_check.json")))["results"]
    c1 = json.load(open(os.path.join(d, "results_case1", "case1_backbone_check.json")))["results"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1))
    a2 = np.array([r["q1_0"] for r in c2 if not r["failed"]]); o = np.argsort(a2)
    fo = np.array([r["6modes"]["ratio_fom"] for r in c2 if not r["failed"]])[o]; ro = np.array([r["6modes"]["ratio_rom"] for r in c2 if not r["failed"]])[o]
    ax[0].plot(a2[o], (fo - 1) * 100, "s-", color="k", ms=4, label="full-order"); ax[0].plot(a2[o], (ro - 1) * 100, "o--", color=VIOLET, ms=4, label="ROM, 6 modes")
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set(xlabel="release amplitude $q_1(0)$ (mm, signed)", ylabel="$f/f_{\\rm lin}-1$ (%)", title="(a) cylindrical panel"); ax[0].legend(fontsize=7)
    a1 = np.array([max(abs(np.array(r["q1_range_fom"]))) for r in c1]); o = np.argsort(a1)
    fo = np.array([r["ratio_fom"] for r in c1])[o]; ro = np.array([r["ratio_rom"] for r in c1])[o]
    ax[1].plot(a1[o] * 1e3, (fo - 1) * 100, "s-", color="k", ms=4, label="full-order"); ax[1].plot(a1[o] * 1e3, (ro - 1) * 100, "o--", color=VIOLET, ms=4, label="ROM")
    ax[1].axhline(0, color=GREY, lw=0.8)
    ax[1].set(xlabel="orbit amplitude (mm)", ylabel="$f/f_{\\rm lin}-1$ (%)", title="(b) plate with hole"); ax[1].legend(fontsize=7)
    save(fig, "ch14_hyrom.pdf")
    pb = max(abs(r["6modes"]["f_rom_hz"] / r["f_fom_hz"] - 1) for r in c2 if not r["failed"]) * 100
    num("cvfHyPanelBack", pct(pb, 2)); check("panel backbone, 6 modes (%)", pb, 0.09, 0.15)
    last = c1[-1]; num("cvfHyPlateRatio", f"{(last['ratio_rom'] - 1) / (last['ratio_fom'] - 1):.1f}")
    print("hyrom plate over-hardening ratio by amplitude:", [round((r["ratio_rom"] - 1) / (r["ratio_fom"] - 1), 2) for r in c1])


# ------------------------------------------------------------------------------------------------ ICE (Hollkamp et al.)
def fig_ice(d):
    z = np.load(os.path.join(d, "reproduce_flat_beam_results.npz"))
    fig, ax = plt.subplots(2, 2, figsize=(8.8, 5.0))
    for i, lv in enumerate(("low", "high")):
        t = z[f"{lv}_t"]; m = t <= 0.2
        ax[i, 0].plot(t[m], z[f"{lv}_v_fom"][m], color="k", lw=2.5, alpha=0.35, label="full-order"); ax[i, 0].plot(t[m], z[f"{lv}_v_rom"][m], color=VIOLET, lw=0.9, label="one-mode ROM")
        ax[i, 0].set(xlabel="time (s)", ylabel="displacement at $x=4$ in", title=f"({'ab'[i]}) {lv} level, time history"); ax[i, 0].legend(fontsize=7)
        ax[i, 1].loglog(z[f"{lv}_f_fom"][1:], z[f"{lv}_Pxx_fom"][1:], color="k", lw=2.5, alpha=0.35); ax[i, 1].loglog(z[f"{lv}_f_rom"][1:], z[f"{lv}_Pxx_rom"][1:], color=VIOLET, lw=0.9)
        ax[i, 1].set(xlabel="frequency (Hz)", ylabel="PSD", title=f"({'cd'[i]}) {lv} level, spectrum", xlim=(5, 3000))
        e = np.sqrt(np.mean((z[f"{lv}_v_rom"] - z[f"{lv}_v_fom"]) ** 2)) / np.sqrt(np.mean(z[f"{lv}_v_fom"] ** 2)) * 100
        num(f"cvfIce{lv.capitalize()}Rms", pct(e, 3)); print("ice rms error", lv, e)
    save(fig, "ch14_ice.pdf")


# ------------------------------------------------------------------------------------------------ FD-PINN band truncation
def fig_fdpinn(d):
    rows = list(csv.DictReader(open(os.path.join(d, "band_study.csv"))))
    bands = [float(k) for k in rows[0] if k not in ("case", "model")]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    col = {"case1": BLUE, "case2": GREEN, "case3": ORANGE}
    for r in rows:
        if r["model"] == "full" and r["case"] in col:
            ax[0].loglog(bands, [float(r[str(b).rstrip("0").rstrip(".") if str(b).endswith(".0") else str(b)]) for b in bands], "o-", color=col[r["case"]], ms=3.5, label=r["case"])
    ax[0].axvline(5, color=GREY, ls=":", lw=1); ax[0].text(5.2, ax[0].get_ylim()[1] * 0.3, "5 Hz band of the paper", fontsize=7, color=GREY)
    ax[0].set(xlabel="band width (Hz)", ylabel="normalised error of the band-limited field", title="(a) cost of truncating the spectrum"); ax[0].legend(fontsize=7)
    rr = [1, 2, 3, 5, 10]
    for c in ("case1", "case2", "case3"):
        full = [float(r["5"]) for r in rows if r["case"] == c and r["model"] == "full"][0]
        ax[1].semilogy(rr, [float([r for r in rows if r["case"] == c and r["model"] == f"rom{k}"][0]["5"]) for k in rr], "o-", color=col[c], ms=3.5, label=c)
        ax[1].axhline(full, color=col[c], ls=":", lw=1)
    ax[1].set(xlabel="ROM basis size $r$", ylabel="error at the 5 Hz band", title="(b) frequency-domain ROM against full order (dotted)"); ax[1].legend(fontsize=7)
    save(fig, "ch14_fdpinn.pdf")
    for c, k in (("case1", "One"), ("case2", "Two"), ("case3", "Three")):
        v = [float(r["5"]) for r in rows if r["case"] == c and r["model"] == "full"][0]; num(f"cvfFdBand{k}", sci(v))
    check("FD-PINN band cost case 1", float([r for r in rows if r['case'] == 'case1' and r['model'] == 'full'][0]["5"]), 3.8e-5, 0.02)
    ten = [abs(float([r for r in rows if r["case"] == c and r["model"] == "rom10"][0]["5"]) / float([r for r in rows if r["case"] == c and r["model"] == "full"][0]["5"]) - 1) for c in ("case1", "case2", "case3")]
    num("cvfFdTen", pct(max(ten) * 100, 2)); check("FD-PINN rom10 vs full (<0.1 %)", max(ten) * 100, 0.05, 1.0)


# ------------------------------------------------------------------------------------------------ hybrid hyper-reduction, rope (Fauque et al.)
def fig_hyb(d):
    z = json.load(open(os.path.join(d, "results", "rope_results_L3e10.json")))
    g = z["greedy_eta"]; k = np.arange(1, len(g["hr_u2"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1))
    ax[0].semilogy(k, g["hr_u2"], "o-", color=VIOLET, ms=3, label="hybrid (HR)"); ax[0].semilogy(k, g["pb_u2"], "s--", color=BLUE, ms=3, label="projection (PB)")
    ax[0].set(xlabel="greedy iteration", ylabel="displacement error (%, 2-norm)", title="(a) displacement"); ax[0].legend(fontsize=7)
    ax[1].semilogy(k, g["hr_lam2"], "o-", color=VIOLET, ms=3); ax[1].semilogy(k, g["pb_lam2"], "s--", color=BLUE, ms=3)
    ax[1].set(xlabel="greedy iteration", ylabel="contact-force error (%, 2-norm)", title="(b) contact force")
    ax[2].plot(k, g["hr_unknowns"], "o-", color=VIOLET, ms=3, label="HR unknowns"); ax[2].plot(k, g["pb_unknowns"], "s--", color=BLUE, ms=3, label="PB unknowns")
    ax[2].plot(k, g["rid_nodes"], ":", color=GREY, label="RID nodes"); ax[2].set(xlabel="greedy iteration", ylabel="count", title="(c) size of the reduced problem"); ax[2].legend(fontsize=7)
    save(fig, "ch14_hyb.pdf")
    f = z["fig12"]; num("cvfHybDispMean", pct(f["disp"]["mean_pct"], 2)); num("cvfHybDispMax", pct(f["disp"]["max_pct"], 2))
    num("cvfHybForceMean", pct(f["force_rid"]["mean_pct"], 2)); num("cvfHybForceRecon", pct(f["force_recon"]["mean_pct"], 2))
    print("hyb final greedy-eta errors:", f["disp"], f["force_rid"], f["force_recon"], "(report: 0.011/0.084, 0.45/9.6, 10.3/36.1)")
    check("rope displacement mean (%)", f["disp"]["mean_pct"], 0.011, 0.05); check("rope displacement max (%)", f["disp"]["max_pct"], 0.084, 0.05)


# ------------------------------------------------------------------------------------------------ assembly reduction (Nikiema et al.), 2D
def fig_asm(d):
    rows = json.load(open(os.path.join(d, "retry_results.json")))["rows"]
    rows = [r for r in rows if r["method"] in ("POD", "DEIM", "UDEIM", "ECSW")]
    meth = ["POD", "ECSW", "DEIM", "UDEIM"]; sty = {"POD": ("o", BLUE), "ECSW": ("s", GREEN), "DEIM": ("^", ORANGE), "UDEIM": ("v", VIOLET)}
    eps = ["1e-05", "1e-06", "1e-09"]
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for i, m in enumerate(meth):
        for j, e in enumerate(eps):
            r = [x for x in rows if x["method"] == m and x["eps"] == e][0]
            xpos = j + (i - 1.5) * 0.15
            if r["converged"]:
                ax[0].semilogy(xpos, max(r["err_last_pct"], 1e-4), sty[m][0], color=sty[m][1], ms=7, label=m if j == 0 else None)
            else:
                ax[0].semilogy(xpos, 1.0, "x", color=sty[m][1], ms=9, mew=2, label=m if j == 0 else None)
                ax[0].text(xpos, 1.5 + (2.5 if m == "UDEIM" else 0), f"fails at\nincr. {r['failed_inc']}", ha="center", fontsize=6, color=sty[m][1])
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["$10^{-5}$", "$10^{-6}$", "$10^{-9}$"]); ax[0].set_ylim(2e-4, 40)
    ax[0].set(xlabel="truncation tolerance $\\varepsilon$", ylabel="error of the last increment (%)", title="(a) 2D assembly, plain Newton"); ax[0].legend(fontsize=7, ncol=2, loc="lower left")
    for i, m in enumerate(meth):
        xs = [r["n_elements"] for r in rows if r["method"] == m and r["converged"]]; ys = [r["err_last_pct"] for r in rows if r["method"] == m and r["converged"]]
        ax[1].loglog(xs, np.maximum(ys, 1e-4), sty[m][0], color=sty[m][1], ms=7, label=m)
    ax[1].set(xlabel="elements evaluated per iteration (of 324)", ylabel="error of the last increment (%)", title="(b) converged runs"); ax[1].legend(fontsize=7)
    save(fig, "ch14_asm.pdf")
    nf = sum(1 for r in rows if not r["converged"]); num("cvfAsmFail", str(nf))
    num("cvfAsmPodErr", pct([r for r in rows if r["method"] == "POD" and r["eps"] == "1e-05"][0]["err_last_pct"], 2))
    print("asm failures (DEIM/UDEIM at 1e-6 expected):", [(r["method"], r["eps"], r["failed_inc"]) for r in rows if not r["converged"]])
    check("2D POD error at 1e-5 (%)", [r for r in rows if r["method"] == "POD" and r["eps"] == "1e-05"][0]["err_last_pct"], 0.94, 0.01)


# ------------------------------------------------------------------------------------------------ projection-based contact ROM (Balajewicz et al.)
def fig_bal(d):
    v = json.load(open(os.path.join(d, "results", "v2_dynamic_40.json")))
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.1))
    names = {"1,1,1": "$\\alpha=(1,1,1)$", "1,1,0": "$\\alpha=(1,1,0)$", "1,0,0": "$\\alpha=(1,0,0)$", "true": "true error"}
    for key, c in zip(names, (BLUE, GREEN, VIOLET, GREY)):
        e = v["greedy"][key]["emax"]; ax[0].semilogy(range(1, len(e) + 1), e, "o-", ms=3, color=c, label=names[key])
    ax[0].set(xlabel="greedy iteration", ylabel="maximum relative error (%)", title="(a) greedy sampling"); ax[0].legend(fontsize=7)
    f = v["final"]; h = np.array(f["hdm_0"]["traj"])[:, 0]; n = np.array(f["nmf_0"]["traj"])[:, 0]; s = np.array(f["svd_0"]["traj"])[:, 0]
    t = np.arange(len(h)) * 0.005
    ax[1].plot(t, h, color="k", lw=2.5, alpha=0.35, label="full-order"); ax[1].plot(t, n, color=VIOLET, lw=1, label="NMF dual basis")
    ax[1].set(xlabel="time (s)", ylabel="response", title="(b) NMF dual basis"); ax[1].legend(fontsize=7)
    ax[2].plot(t, h, color="k", lw=2.5, alpha=0.35, label="full-order"); ax[2].plot(t, s, color=ORANGE, lw=1, label="SVD dual basis")
    ax[2].set(xlabel="time (s)", ylabel="response", title="(c) SVD dual basis"); ax[2].legend(fontsize=7)
    save(fig, "ch14_bal.pdf")
    e = v["greedy"]["1,0,0"]["emax"]; num("cvfBalRed", f"{e[0] / e[-1]:.0f}"); num("cvfBalNmf", pct(f["nmf_0"]["err"], 2))
    num("cvfBalSvd", f"{f['svd_0']['err']:.0f}\\,\\%")
    print("bal dynamic: err nmf/svd (raw)", f["nmf_0"]["err"], f["svd_0"]["err"], "min lam", f["nmf_0"]["min_lam"], f["svd_0"]["min_lam"], "speedup", v["speedup"])
    check("Balajewicz dynamic greedy reduction (alpha 1,0,0)", e[0] / e[-1], 288, 0.02)
    num("cvfBalSpeed", f"{v['speedup']:.1f}"); check("dynamic speed-up", v["speedup"], 14.7, 0.03)


# ------------------------------------------------------------------------------------------------ FRINO (Yang et al. 2025)
def fig_frino(d):
    rows = list(csv.DictReader(open(os.path.join(d, "results", "sweeps.csv")))); ts = [r for r in rows if r["group"] == "trainset"]
    labs = {"set_paper8": "8 tones", "set_paper8+tones": "40 tones", "set_rich": "200 waveforms"}
    cats = [("low_fine", "low\nsingle tone"), ("high_fine", "high\nsingle tone"), ("multi_fine", "multi-\nfrequency"), ("random_fine", "random")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); w = 0.26
    for i, (r, c) in enumerate(zip(ts, (BLUE, VIOLET, GREEN))):
        ax.bar(np.arange(4) + (i - 1) * w, [float(r[k]) for k, _ in cats], w, color=c, label=labs[r["tag"]])
    for x, v, nm in ((2, 0.17, "pub."), (3, 0.098, "pub.")):
        ax.hlines(v, x - 0.4, x + 0.4, color="k", ls="--", lw=1.2); ax.text(x + 0.42, v, nm, fontsize=7, va="center")
    ax.hlines(0.018, -0.4, 0.4, color="k", ls="--", lw=1.2); ax.text(0.42, 0.018, "pub.", fontsize=7, va="center")
    ax.set_xticks(range(4)); ax.set_xticklabels([c for _, c in cats], fontsize=8); ax.set(ylabel="mean relative $L_2$ error", title="Neural operator, error by test load and training set"); ax.legend(fontsize=7)
    save(fig, "ch14_frino.pdf")
    r = ts[1]; num("cvfFrinoLow", sci(float(r["low_fine"]))); num("cvfFrinoMulti", f"{float(r['multi_fine']):.3f}"); num("cvfFrinoRandom", f"{float(r['random_fine']):.3f}")
    check("FRINO multi-frequency (40 tones)", float(r["multi_fine"]), 0.443, 0.01); check("FRINO random (40 tones)", float(r["random_fine"]), 0.691, 0.01)


FIGS = {"mfs": fig_mfs, "hyrom": fig_hyrom, "ice": fig_ice, "fdpinn": fig_fdpinn, "hyb": fig_hyb, "asm": fig_asm, "bal": fig_bal, "frino": fig_frino}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", default=os.path.normpath(os.path.join(BOOK, "..", "..", "..", "ROM_formulations")))
    ap.add_argument("--only", default=",".join(FIGS)); ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--scratch", default=os.path.join(BOOK, "scripts", "_scratch_ch14"))
    a = ap.parse_args()
    print(platform.platform(), "numpy", np.__version__, "jobs", a.jobs, flush=True)
    for s in a.only.split(","):
        root = a.study_root
        if a.rerun:
            if s not in RERUN:
                print("no rerun commands for", s, "- redrawing from the saved data"); d = os.path.join(root, STUDIES[s])
            else:
                os.makedirs(a.scratch, exist_ok=True); d = rerun(s, root, a.jobs, a.scratch)
        else:
            d = os.path.join(root, STUDIES[s])
        if not os.path.isdir(d):
            print("MISSING study folder", d, "- skipped"); continue
        try:
            FIGS[s](d)
        except Exception as ex:                                    # a missing or changed output must not hide the other figures
            print("FAILED", s, type(ex).__name__, ex)
    num("cvfMachine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPUs, NumPy {np.__version__}")
    with open(os.path.join(TAB, "numbers_ch14.tex"), "w") as fh:
        fh.write("% generated by scripts/ch14_assets.py -- do not edit\n")
        for k, v in NUM.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}\\xspace}}\n")
    print("\nconsistency with the written reports (saved data against report text):")
    for w, m, r, ok in CHECKS:
        print(f"  {'ok  ' if ok else 'DIFF'} {w}: data {m:.4g}, report {r:.4g}")
    print("wrote", len(NUM), "macros")


if __name__ == "__main__":
    main()
