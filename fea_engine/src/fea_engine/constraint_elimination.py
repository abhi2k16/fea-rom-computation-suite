# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
constraint_elimination.py -- linear constraints in nonlinear analysis by an elimination matrix ``u = T q``.

``FESystem.add_constraint`` / ``tie`` relations ``sum_i c_i u_i = 0`` were refused by the nonlinear drivers. ``EliminatedSystem`` removes them
exactly: with the independent DOFs ``q`` it uses ``u = T q`` (``T`` sparse, ``n_dof x n_q``: identity rows for independent DOFs, rows of
``-coefficients`` for dependent DOFs, zero rows for DOFs fixed by ``fix_dofs``), so that

    f_q = T^T f_int(T q),     K_q = T^T K_T(T q) T,     F_q = T^T F,

and any driver that works on an ``FESystem`` runs on the object unchanged (it presents ``n_dof = n_q``, ``free_dofs = range(n_q)``,
``F``, ``assemble_internal_force``, ``assemble_tangent_stiffness`` with ``sparse=True`` giving CSR, ``commit_all_states``,
``update_iter_states``). ``solve_nonlinear_static`` and ``solve_nonlinear_arc_length`` do this automatically when the system has
constraints and return displacements in the full numbering; for ``solve_nonlinear_displacement_control`` build the object with
``fes.eliminate_constraints()``, pass it as the system (``control_dof`` is then a number in ``q``, see ``independent_dofs``), and
expand the result with ``expand``.

Scope: the constraints must be homogeneous (right-hand side 0) and the prescribed values zero, which is what the nonlinear drivers
assume of Dirichlet conditions; otherwise ``NotImplementedError``. The transient driver is not supported (it needs reduced mass and
damping matrices).
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import numpy as np
import scipy.sparse as sp

__all__ = ["EliminatedSystem"]


class EliminatedSystem:
    """View of an ``FESystem`` in the independent coordinates ``q`` of its linear constraints (see the module docstring)."""

    def __init__(self, fes):
        from .boundary import reduce_constraints
        if not fes._constraints:
            raise ValueError("EliminatedSystem: the system has no constraints (add_constraint / tie); use it directly")
        if any(abs(v) > 0.0 for v in fes.fixed_dof_values.values()):
            raise NotImplementedError("EliminatedSystem: non-zero prescribed values are not supported by the nonlinear drivers")
        if any(v != 0.0 for _, _, v in fes._constraints):
            raise NotImplementedError("EliminatedSystem: constraints with a non-zero right-hand side are not supported "
                                      "(the nonlinear drivers start from zero displacement and scale only the load)")
        self.fes = fes
        n = fes.n_dof
        fixed = fes.fixed_dofs_array
        rows = [dict(zip(d.tolist(), c.tolist())) for d, c, _ in fes._constraints]
        slave, mat, g = reduce_constraints(rows, [0.0] * len(rows), {int(d): 0.0 for d in fixed})
        free = fes.free_dofs
        is_slave = np.zeros(n, bool)
        is_slave[slave] = True
        keep = free[~is_slave[free]]
        kpos = {int(d): i for i, d in enumerate(keep)}
        r, c, v = list(keep), list(range(len(keep))), [1.0] * len(keep)
        for s in slave:
            for m, coef in mat[int(s)].items():
                if m not in kpos:
                    raise ValueError(f"constraint refers to DOF {m}, which is prescribed or dependent")
                r.append(int(s)); c.append(kpos[m]); v.append(-coef)
        if any(abs(g[int(s)]) > 1e-14 for s in slave):
            raise NotImplementedError("EliminatedSystem: the constraints reduce to a non-zero offset")
        self.T = sp.csr_matrix((v, (r, c)), shape=(n, len(keep)))
        self.independent_dofs = keep                       # global DOF of each q entry
        self.dependent_dofs = np.asarray(slave, dtype=int)
        self.n_dof = len(keep)
        self.free_dofs = np.arange(self.n_dof)
        self.fixed_dofs = set()
        self.fixed_dof_values = {}
        self.sparse = fes.sparse

    # ------------------------------------------------------------------ mapping
    def expand(self, q):
        """Full displacement vector(s) ``u = T q``; ``q`` is (n_q,) or (steps, n_q)."""
        q = np.asarray(q, float)
        return (self.T @ q.T).T if q.ndim == 2 else self.T @ q

    def reduce(self, u):
        """Independent coordinates of a full displacement vector (the entries at ``independent_dofs``)."""
        return np.asarray(u, float)[..., self.independent_dofs]

    @property
    def F(self):
        return self.T.T @ self.fes.F

    # ------------------------------------------------------------------ element states (path-dependent elements)
    @property
    def state(self):
        return self.fes.state

    @property
    def iter_state(self):
        return self.fes.iter_state

    @iter_state.setter
    def iter_state(self, value):
        self.fes.iter_state = value

    def commit_all_states(self, q, mat, **kwargs):
        return self._guarded(self.fes.commit_all_states, self.expand(q), mat, **kwargs)

    def update_iter_states(self, q, dq, mat, **kwargs):
        return self._guarded(self.fes.update_iter_states, self.expand(q), self.expand(dq), mat, **kwargs)

    # ------------------------------------------------------------------ force and tangent in q
    def _guarded(self, fn, *args, **kwargs):
        """Call a FESystem method with its 'constraints not supported' guard lifted (they are eliminated here)."""
        old = self.fes._constraints_eliminated
        self.fes._constraints_eliminated = True
        try:
            return fn(*args, **kwargs)
        finally:
            self.fes._constraints_eliminated = old

    def assemble_internal_force(self, q, mat, **kwargs):
        return self.T.T @ self._guarded(self.fes.assemble_internal_force, self.expand(q), mat, **kwargs)

    def assemble_tangent_stiffness(self, q, mat, sparse=False, **kwargs):
        K = self._guarded(self.fes.assemble_tangent_stiffness, self.expand(q), mat, sparse=sparse, **kwargs)
        if sparse:
            return (self.T.T @ K @ self.T).tocsr()
        A = self.T.T @ K                                   # ndarray
        return np.asarray((self.T.T @ A.T).T)
