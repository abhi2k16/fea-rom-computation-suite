# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
system_driven_bases_timing.py -- how accurate, how fast, and at what offline cost are the system-driven reduced
models, for a static load sweep and for a transient run, on a beam or a shell?

MODELS. `beam`: the geometrically nonlinear clamped-clamped Beam2DCorotational chain of fea_engine (size = number of
elements). `shell`: a clamped square plate of Shell4MITCCorotational shells (size = elements per side). Both come from
the fixtures in tests/ as plain arrays and callables, so the reduced models see no fea_engine object.

REDUCED MODELS TIMED, each against a full-order solution:
  * Galerkin on 3 vibration modes and on the modal-derivative basis (both still call the FULL-order force),
  * a polynomial reduced force identified from tangent probes (enhanced enforced displacement), no full-order calls,
  * a 3-coordinate quadratic manifold with ECSW trained on lifted states (a small fraction of the elements).

STATIC: a sweep of load levels (1 to 4 thicknesses of peak deflection, load shaped like mode 1 for the beam, a uniform transverse pressure for the plate), full-order
load-stepped Newton as reference.
TRANSIENT: free undamped vibration released from the static state at 3 thicknesses, two linear periods of mode 1,
full-order average-acceleration Newmark-Newton as reference (reduced: the same Newmark with the reduced force for
the Galerkin and polynomial models, explicit RK4 for the manifold). Error is the worst-case relative error of the
displacement field over all time steps.

For each model: error, online time per query (static) or per run (transient), offline time, speed-up, effective
speed-up with the offline cost charged (rom_engine.effective_speedup) and the break-even number of queries. Printed as
JSON. Wall-clock numbers depend on the machine, the NumPy/BLAS build and the thread count: run it where you will use it.

    python examples/system_driven_bases_timing.py beam 100 8
    python examples/system_driven_bases_timing.py shell 4 6 [--no-eed] [--steps-per-period=50]

Shells are slow to assemble (a Python loop over elements). `--cache=DIR` keeps every finished step on disk, so an
interrupted run resumes where it stopped instead of starting over.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tests"), os.path.join(HERE, "..", "src")]      # fixtures use fea_engine
from rom_engine import (static_modal_derivatives, augment_with_derivatives, enhanced_enforced_displacement_rom,
                        manifold_from_derivatives, lifted_training_states, ManifoldECSW, QuadraticManifold,
                        QuadraticManifoldROM, effective_speedup, break_even_queries)


def newton(res, jac, x0, scale, tol=1e-7, max_it=40):
    """Newton with a relative residual tolerance; stops at the round-off floor (stagnation) instead of failing."""
    x = np.array(x0, float)
    prev = np.inf
    for it in range(max_it):
        r = res(x)
        rel = np.linalg.norm(r) / scale
        if rel <= tol or (it >= 3 and rel < 1e-5 and rel >= 0.7 * prev):
            return x
        prev = rel
        x = x - np.linalg.solve(jac(x), r)
    raise RuntimeError("Newton did not converge")


def best(fn, reps):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), out


def build(kind, size):
    if kind == "beam":
        from nonlinear_beam_fixtures import nonlinear_beam_model
        b = nonlinear_beam_model(n_elem=size, n_modes=3)
        b["h"] = float(np.sqrt(12 * 8.333e-9 / 1e-4))
        b["length"] = b["L"]
    else:
        from shell_plate_fixtures import nonlinear_shell_plate_model
        b = nonlinear_shell_plate_model(n=size)
        b["length"] = b["a"]
    return b


def newmark(M_r, force, tangent, q0, qd0, dt, n_steps, scale, ckpt=None):
    """Average-acceleration Newmark-Newton for M_r q'' + force(q) = 0 (free, undamped). Returns (history of q, wall
    time). With `ckpt` (a file path) the state is saved every few steps and a rerun resumes from it, so a long
    full-order run can be split over several shorter processes; the returned time adds up the pieces."""
    import pickle
    q, v = np.array(q0, float), np.array(qd0, float)
    a = np.linalg.solve(M_r, -force(q))
    hist, spent = [q.copy()], 0.0
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, "rb") as fh:
            hist, q, v, a, spent = pickle.load(fh)
    t_start = time.perf_counter()
    while len(hist) <= n_steps:
        def res(x):
            an = (x - q - dt * v) * 4 / dt ** 2 - a
            return M_r @ an + force(x)
        qn = newton(res, lambda x: 4 / dt ** 2 * M_r + tangent(x), q + dt * v + 0.25 * dt ** 2 * a, scale)
        an = (qn - q - dt * v) * 4 / dt ** 2 - a
        v = v + 0.5 * dt * (a + an)
        q, a = qn, an
        hist.append(q.copy())
        if ckpt and len(hist) % 5 == 0:
            with open(ckpt + ".tmp", "wb") as fh:
                pickle.dump((hist, q, v, a, spent + time.perf_counter() - t_start), fh)
            os.replace(ckpt + ".tmp", ckpt)
    return np.array(hist), spent + time.perf_counter() - t_start


class Cache:
    """Small on-disk store so a long run can be resumed after an interruption (every expensive step is kept)."""

    def __init__(self, path):
        self.path = path
        if path:
            os.makedirs(path, exist_ok=True)

    def get(self, name, fn):
        import pickle
        f = os.path.join(self.path, name + ".pkl") if self.path else None
        if f and os.path.exists(f):
            with open(f, "rb") as fh:
                return pickle.load(fh)
        val = fn()
        if f:
            with open(f, "wb") as fh:
                pickle.dump(val, fh)
        return val


class MemoTangent:
    """Wraps a tangent-stiffness callable with a disk-backed memo, so the many probe states of the polynomial-force
    identification survive an interruption (only for resumable runs; harmless otherwise)."""

    def __init__(self, fn, path):
        import pickle, signal
        self.fn, self.path, self.memo, self.new, self.spent = fn, path, {}, 0, 0.0
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                self.memo, self.spent = pickle.load(fh)
        if path:
            signal.signal(signal.SIGTERM, lambda *_: (self.save(), sys.exit(124)))

    def save(self):
        import pickle
        if self.path and self.new:
            with open(self.path + ".tmp", "wb") as fh:
                pickle.dump((self.memo, self.spent), fh)
            os.replace(self.path + ".tmp", self.path)
            self.new = 0

    def __call__(self, u):
        key = np.asarray(u, float).tobytes()
        if key not in self.memo:
            t0 = time.perf_counter()
            self.memo[key] = self.fn(u)
            self.spent += time.perf_counter() - t0          # cumulative over resumed runs
            self.new += 1
            if self.new >= 20:
                self.save()
        return self.memo[key]


def main(kind="beam", size=100, n_sweep=8, run_eed=True, run_transient=True, cache_dir=None, steps_per_period=100):
    from rom_engine.nonlinear_rom import PolynomialModalROM
    b = build(kind, size)
    cache = Cache(cache_dir)
    K0, M, Phi, n, h = b["K0"], b["M"], b["Phi"], b["n_free"], b["h"]
    tr = b.get("translational", np.ones(n, bool))          # amplitudes are measured on displacement DOFs only
    peak = float(np.abs(Phi[tr, 0]).max())
    pattern = b.get("load_pattern")          # shells: uniform transverse load; beams: the load that excites mode 1 alone
    if pattern is None:
        load_of = lambda k: K0 @ (k * h / peak * Phi[:, 0])
    else:
        peak_lin = float(np.abs(np.linalg.solve(K0, pattern)[tr]).max())
        load_of = lambda k: pattern * (k * h / peak_lin)       # linear peak displacement = k thicknesses
    loads = [load_of(k) for k in np.linspace(1.0, 4.0, n_sweep)]
    reps = 3 if kind == "beam" else 1
    free = b["free"] if "free" in b else np.arange(n)

    def fom_static(F, steps=4):
        """Load-stepped Newton; if a step fails the number of load steps is doubled (up to 64) and the solve repeated."""
        while True:
            try:
                u = np.zeros(n)
                for lam in np.linspace(0, 1, steps + 1)[1:]:
                    u = newton(lambda x: b["f_int"](x) - lam * F, b["K_t"], u, np.linalg.norm(lam * F))
                return u
            except RuntimeError:
                if steps >= 64:
                    raise
                steps *= 2

    def galerkin(W, F):
        return newton(lambda x: W.T @ (b["f_int"](W @ x) - F), lambda x: W.T @ b["K_t"](W @ x) @ W,
                      np.zeros(W.shape[1]), np.linalg.norm(W.T @ F))

    # ---------------------------------------------------------------- offline (cached)
    def make_md():
        t = time.perf_counter()
        md = static_modal_derivatives(K0, Phi, tangent_fn=b["K_t"], length_scale=b["length"])
        V, _ = augment_with_derivatives(Phi, md, M=M)
        return dict(theta=md.theta, V=V, t=time.perf_counter() - t)
    mdd = cache.get("md", make_md)
    V, theta = mdd["V"], mdd["theta"]
    off = {"modal derivatives": mdd["t"]}
    Kr = V.T @ K0 @ V
    man0 = manifold_from_derivatives(Phi, theta, M)
    scale = h / float(np.abs(V[tr, 0]).max())

    def mode_scales(a):
        c = V.T @ (M @ man0.map(np.array([a, 0.0, 0.0])))
        sc = np.abs(c) / a
        sc[:3] = [1.0, 0.3, 0.3]
        return sc

    eed = None
    if run_eed:
        def make_eed():
            t = time.perf_counter()
            memo = MemoTangent(b["K_t"], os.path.join(cache_dir, "eed_probes.pkl") if cache_dir else None)
            m, _ = enhanced_enforced_displacement_rom(V, memo, [0.25 * scale, 0.5 * scale, scale, 2 * scale],
                                                      K_r=Kr, mode_scales=mode_scales)
            memo.save()
            fit = time.perf_counter() - t
            return dict(coeffs=m.coeffs, t=memo.spent + (fit if not cache_dir else 0.0))
        ed = cache.get("eed", make_eed)
        eed = PolynomialModalROM(V.shape[1])
        eed.coeffs = ed["coeffs"]
        off["EED polynomial"] = ed["t"] + off["modal derivatives"]
    amp = 4 * h / peak
    Q, _ = lifted_training_states(man0, [amp, 0.3 * amp, 0.3 * amp], n_samples=40)
    Phi_f = np.zeros((b["n"], Phi.shape[1])); Phi_f[free] = Phi
    Om_f = np.zeros((b["n"], Phi.shape[1], Phi.shape[1])); Om_f[free] = man0.Omega
    M_f = np.zeros((b["n"], b["n"])); M_f[np.ix_(free, free)] = M          # the dynamics need the mass on the same DOFs
    man_f = QuadraticManifold(Phi_f, Om_f, M_f)

    def make_ecsw():
        t = time.perf_counter()
        e = ManifoldECSW.fit(man_f, Q, b["elem_dofs"], b["elem_force"], tol=1e-4)
        return dict(weights=e.weights, t=time.perf_counter() - t)
    ew = cache.get("ecsw", make_ecsw)
    ecsw = ManifoldECSW(man_f, b["elem_dofs"], b["elem_force"], ew["weights"])
    off["manifold ECSW"] = ew["t"] + off["modal derivatives"]
    offline_of = {"Galerkin, 3 modes (full force)": 0.0,
                  "Galerkin, modal derivatives (full force)": off["modal derivatives"],
                  "polynomial force (EED)": off.get("EED polynomial"), "quadratic manifold + ECSW": off["manifold ECSW"]}

    def finish(res, t_fom, n_q):
        for name, r in res.items():
            if "failed" in r:
                continue
            r["offline"] = offline_of[name]
            r["speedup"] = t_fom / r["t"]
            r["effective_speedup"] = effective_speedup(n_q, r["t"], r["offline"], t_fom)
            r["break_even"] = break_even_queries(r["t"], r["offline"], t_fom)
        return res

    def timed(phase, res, name, fn, coords, per, errfn):
        def go():
            try:
                t, outv = best(fn, reps)
                if isinstance(outv, tuple):                  # (result, accumulated wall time) from a resumable run
                    outv, t = outv
                return dict(coords=coords, t=t / per, err=errfn(outv))
            except Exception as e:                      # a failed solve is a result, not a crash
                return dict(coords=coords, failed=str(e))
        res[name] = cache.get(f"{phase}-{name.split(',')[0].split(' ')[0]}-{coords}", go)

    out = dict(model=kind, size=size, n_free=n, offline=off, basis=V.shape[1],
               ecsw=dict(selected=ecsw.n_selected, total=ecsw.n_total))

    # ---------------------------------------------------------------- static sweep
    def make_static_ref():
        t, U = best(lambda: [fom_static(F) for F in loads], 1 if kind == "shell" else 2)
        return dict(t=t, U=U)
    sref = cache.get("static_ref", make_static_ref)
    t_fom, U_fom = sref["t"], sref["U"]
    err = lambda Us: max(np.linalg.norm(u - uf) / np.linalg.norm(uf) for u, uf in zip(Us, U_fom))
    res = {}
    timed("static", res, "Galerkin, 3 modes (full force)", lambda: [Phi @ galerkin(Phi, F) for F in loads], 3, len(loads), err)
    timed("static", res, "Galerkin, modal derivatives (full force)", lambda: [V @ galerkin(V, F) for F in loads],
          V.shape[1], len(loads), err)
    if eed is not None:
        def eed_sweep():
            outl, q = [], None
            for F in loads:
                r = V.T @ F
                q = np.linalg.solve(Kr, r) if q is None else q
                q = newton(lambda x: Kr @ x + eed.force(x) - r, lambda x: Kr + eed.jacobian(x), q, np.linalg.norm(r))
                outl.append(V @ q)
            return outl
        timed("static", res, "polynomial force (EED)", eed_sweep, V.shape[1], len(loads), err)

    def man_sweep():
        rom = QuadraticManifoldROM(man_f, hyper=ecsw)
        outl = []
        for F in loads:
            Fm = np.zeros(b["n"]); Fm[free] = F
            outl.append(man_f.map(rom.static_solve(Fm))[free])
        return outl
    timed("static", res, "quadratic manifold + ECSW", man_sweep, Phi.shape[1], len(loads), err)
    out["static"] = dict(t_fom_per_query=t_fom / len(loads), n_queries=len(loads),
                         models=finish(res, t_fom / len(loads), len(loads)))

    # ---------------------------------------------------------------- transient run
    if run_transient:
        w1 = np.sqrt(b["omega2"][0])
        T = 2 * np.pi / w1
        n_steps, dt = 2 * steps_per_period, T / steps_per_period
        u0 = cache.get("u0", lambda: fom_static(load_of(3.0)))

        def make_tref():
            sc = np.linalg.norm(M @ u0) * w1 ** 2 + 1e-300
            U, t = newmark(M, b["f_int"], b["K_t"], u0, np.zeros(n), dt, n_steps, sc,
                           ckpt=os.path.join(cache_dir, "ck_fom.pkl") if cache_dir else None)
            return dict(t=t, U=U)
        tref = cache.get(f"transient_ref-{steps_per_period}", make_tref)
        t_fom_run, U_ref = tref["t"], tref["U"]
        err_t = lambda U: max(np.linalg.norm(U[i] - U_ref[i]) for i in range(len(U_ref))) / np.max(np.linalg.norm(U_ref, axis=1))
        resT = {}

        def galerkin_run(W):
            Mr = W.T @ M @ W
            q0 = W.T @ (M @ u0)
            Qh, el = newmark(Mr, lambda x: W.T @ b["f_int"](W @ x), lambda x: W.T @ b["K_t"](W @ x) @ W, q0,
                             np.zeros(W.shape[1]), dt, n_steps, np.linalg.norm(Mr @ q0) * w1 ** 2 + 1e-300,
                             ckpt=os.path.join(cache_dir, f"ck_gal{W.shape[1]}.pkl") if cache_dir else None)
            return Qh @ W.T, el
        ph = f"transient{steps_per_period}"
        timed(ph, resT, "Galerkin, 3 modes (full force)", lambda: galerkin_run(Phi), 3, 1, err_t)
        timed(ph, resT, "Galerkin, modal derivatives (full force)", lambda: galerkin_run(V), V.shape[1], 1, err_t)
        if eed is not None:
            def eed_run():
                Mr = V.T @ M @ V
                q0 = V.T @ (M @ u0)
                Qh, el = newmark(Mr, lambda x: Kr @ x + eed.force(x), lambda x: Kr + eed.jacobian(x), q0,
                                 np.zeros(V.shape[1]), dt, n_steps, np.linalg.norm(Mr @ q0) * w1 ** 2 + 1e-300)
                return Qh @ V.T, el
            timed(ph, resT, "polynomial force (EED)", eed_run, V.shape[1], 1, err_t)

        def man_run():
            rom = QuadraticManifoldROM(man_f, hyper=ecsw)
            q0 = Phi.T @ (M @ u0)
            _, Qh, _ = rom.simulate(lambda t: np.zeros(b["n"]), q0, np.zeros(Phi.shape[1]), dt, n_steps)
            return np.array([man_f.map(q)[free] for q in Qh])
        timed(ph, resT, "quadratic manifold + ECSW", man_run, Phi.shape[1], 1, err_t)
        out["transient"] = dict(t_fom_run=t_fom_run, n_steps=n_steps, dt=dt,
                                u0_peak_thicknesses=float(np.abs(u0[tr]).max() / h), models=finish(resT, t_fom_run, 1))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kind = args[0] if args else "beam"
    size = int(args[1]) if len(args) > 1 else (100 if kind == "beam" else 4)
    sweep = int(args[2]) if len(args) > 2 else 8
    opt = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(k + "=")), d)
    print(json.dumps(main(kind, size, sweep, run_eed="--no-eed" not in sys.argv,
                          run_transient="--no-transient" not in sys.argv, cache_dir=opt("--cache"),
                          steps_per_period=int(opt("--steps-per-period", 100))), indent=1, default=float))
