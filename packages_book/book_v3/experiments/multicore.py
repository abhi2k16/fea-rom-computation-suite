"""Multi-core study for the 'Using several cores' section. Run as: python multicore.py <case> <n_jobs> <backend>
Prints one JSON line. The BLAS thread count is set from the environment (OPENBLAS_NUM_THREADS) by the caller."""
import sys, json, time, os
sys.path.insert(0, "/sessions/affectionate-epic-mccarthy/mnt/computation-suite/fea_engine/src")
import numpy as np

def plate_job(E):
    from fea_engine import FESystem
    from fea_engine.material import Material, D_plane_stress
    from fea_engine.mesh import rectangle_mesh
    from fea_engine.elements import ELEMENT_REGISTRY
    mesh = rectangle_mesh(4.0, 1.0, 96, 24)
    s = FESystem(mesh, ELEMENT_REGISTRY["quad4_plane_stress"](), sparse=True)
    s.assemble_stiffness(D_plane_stress(Material(E=E, nu=0.3)), vectorized=True)
    left = np.where(mesh.nodes[:, 0] < 1e-12)[0]; right = np.where(mesh.nodes[:, 0] > 4 - 1e-12)[0]
    s.fix_dofs(left, [0, 1]); s.add_nodal_force(right, 1, -1000.0)
    U = s.solve_static()
    return float(np.abs(U).max())

def tiny_job(x):
    return float(np.sin(x) ** 2)

def dense_job(seed):
    rng = np.random.default_rng(seed); A = rng.standard_normal((1200, 1200)); A = A @ A.T + 1200 * np.eye(1200)
    return float(np.linalg.solve(A, np.ones(1200)).sum())

if __name__ == "__main__":
    from fea_engine.batch import map as fe_map
    case, nj, backend = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fn, items = {"plate": (plate_job, [150e9 + 5e9 * i for i in range(12)]),
                 "tiny": (tiny_job, list(range(400))),
                 "dense": (dense_job, list(range(12)))}[case]
    fe_map(fn, items[:2], n_jobs=1)                     # warm-up: imports, caches
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); r = fe_map(fn, items, n_jobs=nj, backend=backend); best = min(best, time.perf_counter() - t)
    print(json.dumps(dict(case=case, n_jobs=nj, backend=backend, blas=os.environ.get("OPENBLAS_NUM_THREADS", "default"),
                          time=best, n_items=len(items), check=r[0])))
