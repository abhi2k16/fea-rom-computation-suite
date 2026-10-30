"""Second worker-count sweep for Section 7.5 (run after multicore_scaling.py).

The first sweep (150 ms jobs) was dominated by process start-up on Windows, where each worker is a freshly
spawned interpreter that imports NumPy/SciPy. This sweep separates the two effects:

  heavy : 48 plate-model solves of about one second each, so computation dominates start-up
  tiny  : 48 trivial jobs, which measure the pool start-up cost alone

    python multicore_scaling_heavy.py          # 1 .. os.cpu_count() workers
    python multicore_scaling_heavy.py 1 2 4 6  # or a chosen list

Writes experiments/multicore_scaling_heavy.jsonl."""
import sys, os, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "fea_engine", "src"))
N_ITEMS = 48


def heavy_job(E):
    import numpy as np
    from fea_engine import FESystem
    from fea_engine.material import Material, D_plane_stress
    from fea_engine.mesh import rectangle_mesh
    from fea_engine.elements import ELEMENT_REGISTRY
    mesh = rectangle_mesh(4.0, 1.0, 288, 72)
    s = FESystem(mesh, ELEMENT_REGISTRY["quad4_plane_stress"](), sparse=True)
    s.assemble_stiffness(D_plane_stress(Material(E=E, nu=0.3)), vectorized=True)
    left = np.where(mesh.nodes[:, 0] < 1e-12)[0]; right = np.where(mesh.nodes[:, 0] > 4 - 1e-12)[0]
    s.fix_dofs(left, [0, 1]); s.add_nodal_force(right, 1, -1000.0)
    return float(np.abs(s.solve_static()).max())


def tiny_job(x):
    import math
    return math.sin(x) ** 2


def child(case, nj):
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from multicore_scaling_heavy import heavy_job, tiny_job   # importable by spawned workers
    from fea_engine.batch import map as fe_map
    fn, items, reps = ((heavy_job, [150e9 + 5e9 * i for i in range(N_ITEMS)], 2) if case == "heavy"
                       else (tiny_job, list(range(N_ITEMS)), 3))
    fe_map(fn, items[:2], n_jobs=1)
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = fe_map(fn, items, n_jobs=nj, backend="process")
        best = min(best, time.perf_counter() - t)
    print(json.dumps(dict(case=case, n_jobs=nj, backend="process", blas="1", time=best, n_items=N_ITEMS, check=r[0])))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3])); sys.exit()
    ncpu = os.cpu_count()
    counts = [int(a) for a in sys.argv[1:]] or list(range(1, ncpu + 1))
    env = dict(os.environ, **{k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    out = os.path.join(HERE, "multicore_scaling_heavy.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps(dict(machine=platform.processor() or platform.machine(), logical_cpus=ncpu,
                                system=platform.platform(), python=platform.python_version())) + "\n")
        for case in ("tiny", "heavy"):
            for nj in counts:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", case, str(nj)],
                                   env=env, capture_output=True, text=True)
                line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else None
                if line is None:
                    print("failed", case, nj, p.stderr[-500:]); continue
                f.write(line + "\n"); f.flush(); print(line)
    print("wrote", out)
