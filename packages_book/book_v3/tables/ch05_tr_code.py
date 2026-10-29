import numpy as np
from fea_engine import elements as elmod
from fea_engine.mesh import Mesh
from fea_engine.solver import FESystem
from fea_engine.loads import LoadPattern, TimeHistoryLoad
from fea_engine.damping import RayleighDamping
from fea_engine import nonlinear_solver as nls

E, A, h0, rho = 210e9, 2e-4, 0.10, 7850.0
L0 = np.hypot(1.0, h0)
mesh = Mesh(nodes=np.array([[-1.0, 0.0], [0.0, h0], [1.0, 0.0]]),
            elements=np.array([[0, 1], [1, 2]]), dim=2)
fes = FESystem(mesh, elmod.TrussTL2D())
fes.fix_dofs([0, 2], [0, 1])
fes.assemble_mass(rho * np.eye(2), thickness=A)            # consistent mass
fes.assemble_damping(RayleighDamping(alpha=0.0, beta=0.0))   # no damping
fes.assemble_stiffness((E, A))
free = np.asarray(fes.free_dofs)
w2 = np.linalg.eigvals(np.linalg.solve(fes.M[np.ix_(free, free)], fes.K[np.ix_(free, free)])).real
T = 2 * np.pi / np.sqrt(w2.min())                          # period of the vertical vibration of the apex
P_dyn = 8 / 27 * E * A * h0**3 / L0**3                     # critical step load from the energy argument

pattern = LoadPattern(node_ids=np.array([1]), dof_index=1)
for frac in (0.95, 1.05):
    load = TimeHistoryLoad(pattern=pattern, time_fn=lambda t, P=frac * P_dyn: -P)
    t, U = nls.solve_nonlinear_transient(fes, (E, A), load, 3 * T, T / 200, tol=1e-9, max_iter=50)
    d = -U[:, 3]                                           # apex deflection history
    print(f"step load {frac:.2f} P_dyn: largest deflection {1e3 * d.max():.2f} mm,",
          "snaps through" if d.max() > h0 else "stays on the first branch")
