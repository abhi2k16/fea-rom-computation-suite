import time
import numpy as np
from fea_engine import FESystem, Material, MultiBlockMesh
from fea_engine.material import D_plane_stress
from fea_engine.elements import Quad4PlaneStressTL

def plate(nx, ny, L=6.0, H=2.0):                       # nx * ny Quad4 elements of a plate
    x, y = np.meshgrid(np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1))
    nodes = np.column_stack([x.ravel(), y.ravel()])
    n = lambda i, j: j * (nx + 1) + i
    el = np.array([[n(i, j), n(i + 1, j), n(i + 1, j + 1), n(i, j + 1)]
                   for j in range(ny) for i in range(nx)])
    fes = FESystem(MultiBlockMesh(nodes=nodes, blocks={"q": el}, dim=2),
                   {"q": Quad4PlaneStressTL()}, sparse=False)
    return fes, {"q": D_plane_stress(Material(E=2.1e11, nu=0.3))}

def timed(fn, repeats=1):                              # best wall time of repeats runs
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return out, best

for nx, ny in ((12, 4), (24, 8), (72, 24)):
    fes, mat = plate(nx, ny)
    u = np.random.default_rng(0).normal(size=fes.n_dof) * 0.02
    kw = dict(thickness=0.5)
    fes.assemble_internal_force(u, mat, vectorized=True, **kw)            # warm-up of the caches
    f_loop, t_loop = timed(lambda: fes.assemble_internal_force(u, mat, **kw))               # the default loop
    f_bat, t_bat = timed(lambda: fes.assemble_internal_force(u, mat, vectorized=True, **kw), 5)  # batched
    diff = np.linalg.norm(f_bat - f_loop) / np.linalg.norm(f_loop)
    print(f"{nx*ny:5d} elements: loop {1e3*t_loop:7.1f} ms, batched {1e3*t_bat:6.2f} ms, "
          f"x{t_loop / t_bat:3.0f}, difference {diff:.0e}")
print(fes.batch_support_report())
print(fes.last_assembly_modes)
