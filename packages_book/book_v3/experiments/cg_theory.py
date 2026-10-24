"""Torch-free study for the PyTorch section: iterations of Jacobi-preconditioned CG, condition number and float32
accuracy for a cantilever plane-stress plate. Same recurrence as fea_engine.torch_sparse_solver._cg_torch, in NumPy."""
import sys, json, time
sys.path.insert(0, "/sessions/affectionate-epic-mccarthy/mnt/computation-suite/fea_engine/src")
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
from fea_engine import FESystem
from fea_engine.material import Material, D_plane_stress
from fea_engine.mesh import rectangle_mesh
from fea_engine.elements import ELEMENT_REGISTRY

def pcg(K, b, dtype, tol, jacobi=True, maxit=20000):
    K = K.astype(dtype); b = b.astype(dtype); d = K.diagonal()
    pre = (lambda r: r / d) if jacobi else (lambda r: r)
    x = np.zeros_like(b); r = b - K @ x; z = pre(r); p = z.copy(); rz = r @ z; bn = np.linalg.norm(b)
    for k in range(maxit):
        Ap = K @ p; a = rz / (p @ Ap); x = x + a * p; r = r - a * Ap
        if np.linalg.norm(r) / bn < tol: return x, k + 1, True
        z = pre(r); rzn = r @ z; p = z + (rzn / rz) * p; rz = rzn
    return x, maxit, False

rows = []
for ny in (8, 16, 32, 64):
    nx = 4 * ny
    mesh = rectangle_mesh(4.0, 1.0, nx, ny)
    s = FESystem(mesh, ELEMENT_REGISTRY["quad4_plane_stress"](), sparse=True)
    s.assemble_stiffness(D_plane_stress(Material(E=210e9, nu=0.3)))
    left = np.where(mesh.nodes[:, 0] < 1e-12)[0]; right = np.where(mesh.nodes[:, 0] > 4 - 1e-12)[0]
    s.fix_dofs(left, [0, 1]); s.add_nodal_force(right, 1, -1000.0)
    free = s.free_dofs
    Kff = sp.csr_matrix(s._as_solve_matrix(s.K)[np.ix_(free, free)]); Ff = s.F[free]
    n = Kff.shape[0]
    xref = spl.spsolve(Kff.tocsc(), Ff)
    D = sp.diags(1 / np.sqrt(Kff.diagonal())); Ks = D @ Kff @ D
    lmax = spl.eigsh(Kff, k=1, which="LM", return_eigenvectors=False)[0]
    lmin = spl.eigsh(Kff.tocsc(), k=1, sigma=0, which="LM", return_eigenvectors=False)[0]
    lmaxs = spl.eigsh(Ks, k=1, which="LM", return_eigenvectors=False)[0]
    lmins = spl.eigsh(Ks.tocsc(), k=1, sigma=0, which="LM", return_eigenvectors=False)[0]
    _, itp, okp = pcg(Kff, Ff, np.float64, 1e-10, True)
    _, itc, okc = pcg(Kff, Ff, np.float64, 1e-10, False)
    x32, it32, ok32 = pcg(Kff, Ff, np.float32, 1e-10, True, maxit=4 * itp)
    x32b, it32b, ok32b = pcg(Kff, Ff, np.float32, 1e-5, True)
    e32 = np.linalg.norm(x32b - xref) / np.linalg.norm(xref)
    rows.append(dict(n=n, nnz=int(Kff.nnz), kappa=lmax / lmin, kappa_j=lmaxs / lmins, it_cg=itc, it_pcg=itp,
                     f32_conv_1e10=bool(ok32), f32_err_at_1e5=float(e32), it32_1e5=it32b,
                     bytes_csr=int(Kff.data.nbytes + Kff.indices.nbytes + Kff.indptr.nbytes), bytes_dense=8 * n * n))
    print(rows[-1], flush=True)
json.dump(rows, open("/tmp/cg_theory.json", "w"), indent=1)
