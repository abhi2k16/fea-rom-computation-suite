"""Job-size sweep for Section 7.5: how does the multi-core benefit change with problem size?

For each mesh size, 48 plate-model solves are run with 1, 4, 8 and 12 worker processes (BLAS limited to one
thread). Output: experiments/multicore_jobsize.jsonl, one line per (mesh, workers) plus a machine header line.

    python multicore_jobsize.py                 # all sizes, workers 1 4 8 12 (about 15 minutes)
    python multicore_jobsize.py --workers 1 6   # choose worker counts
    python multicore_jobsize.py --quick         # three small sizes, for a smoke test
"""
import sys, os, json, time, platform, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "fea_engine", "src"))
N_ITEMS = 48
SIZES = [(48, 12), (96, 24), (144, 36), (192, 48), (288, 72)]     # elements: 576 ... 20 736
WORKERS = [1, 4, 8, 12]


def job(args):
    E, nx, ny = args
    import numpy as np
    from fea_engine import FESystem
    from fea_engine.material import Material, D_plane_stress
    from fea_engine.mesh import rectangle_mesh
    from fea_engine.elements import ELEMENT_REGISTRY
    mesh = rectangle_mesh(4.0, 1.0, nx, ny)
    s = FESystem(mesh, ELEMENT_REGISTRY["quad4_plane_stress"](), sparse=True)
    s.assemble_stiffness(D_plane_stress(Material(E=E, nu=0.3)), vectorized=True)
    left = np.where(mesh.nodes[:, 0] < 1e-12)[0]; right = np.where(mesh.nodes[:, 0] > 4 - 1e-12)[0]
    s.fix_dofs(left, [0, 1]); s.add_nodal_force(right, 1, -1000.0)
    return float(np.abs(s.solve_static()).max())


def child(nx, ny, nj, reps):
    sys.path.insert(0, SRC); sys.path.insert(0, HERE)
    from multicore_jobsize import job                      # importable by spawned workers
    from fea_engine.batch import map as fe_map
    items = [(150e9 + 5e9 * i, nx, ny) for i in range(N_ITEMS)]
    fe_map(job, items[:2], n_jobs=1)                       # warm-up
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = fe_map(job, items, n_jobs=nj, backend="process")
        best = min(best, time.perf_counter() - t)
    print(json.dumps(dict(nx=nx, ny=ny, n_elements=nx * ny, n_jobs=nj, time=best, n_items=N_ITEMS, check=r[0])))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--child":
        child(int(a[1]), int(a[2]), int(a[3]), int(a[4])); sys.exit()
    workers = WORKERS; sizes = SIZES; reps = 2
    if "--workers" in a:
        workers = [int(x) for x in a[a.index("--workers") + 1:] if x.isdigit()]
    if "--quick" in a:
        sizes = SIZES[:3]; reps = 1
    env = dict(os.environ, **{k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    out = os.path.join(HERE, "multicore_jobsize.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps(dict(machine=platform.processor() or platform.machine(), logical_cpus=os.cpu_count(),
                                system=platform.platform(), python=platform.python_version())) + "\n")
        for nx, ny in sizes:
            for nj in workers:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", str(nx), str(ny), str(nj), str(reps)],
                                   env=env, capture_output=True, text=True)
                line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else None
                if line is None:
                    print("failed", nx, ny, nj, p.stderr[-400:]); continue
                f.write(line + "\n"); f.flush(); print(line)
    print("wrote", out)
