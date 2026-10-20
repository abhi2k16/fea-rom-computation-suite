import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("blu", ["#0B2545", "#134074", "#2F7FC1", "#8DB9E8", "#F4F9FF"])
from fea_engine import FESystem, Quad4Laplace
from fea_engine.mesh import hole_in_rectangle_mesh

BG, INK, ACC = "#0B2545", "#FFFFFF", "#8DA9C4"
Lx, Ly, R = 3.0, 2.0, 0.45
mesh = hole_in_rectangle_mesh(Lx, Ly, (1.2, 1.0), R, 6, 24)
X, E = mesh.nodes, mesh.elements

# real eigenmodes of the membrane on this mesh (scalar Laplace, clamped outer edge)
fes = FESystem(mesh, Quad4Laplace(), sparse=True)
fes.assemble_stiffness(1.0); fes.assemble_lumped_mass(1.0)
edge = np.flatnonzero((X[:, 0] < 1e-9) | (X[:, 0] > Lx - 1e-9) | (X[:, 1] < 1e-9) | (X[:, 1] > Ly - 1e-9))
fes.fix_dofs(edge, ["u"])
free = np.asarray(fes.free_dofs)
K = sp.csr_matrix(fes.K)[free][:, free]
M = sp.diags(sp.csr_matrix(fes.M_lumped).diagonal()[free]).tocsr()
w, V = spl.eigsh(K, k=3, M=M, sigma=0, which="LM")
modes = []
for k in range(3):
    u = np.zeros(len(X)); u[free] = V[:, k]; modes.append(u)

fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)          # A4
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_facecolor(BG)
serif = "DejaVu Serif"

# --- left: the full-order mesh
def draw_mesh(box, field=None, lw=0.5, cmap="Blues", alpha=1.0):
    a = fig.add_axes(box); a.set_aspect("equal"); a.axis("off"); a.set_facecolor("none")
    polys = X[E]
    if field is None:
        a.add_collection(PolyCollection(polys, facecolors="none", edgecolors=INK, linewidths=lw))
    else:
        f = field[E].mean(axis=1)
        a.add_collection(PolyCollection(polys, array=f, cmap=cmap, edgecolors="none", linewidths=0, alpha=alpha))
    a.set_xlim(-0.05, Lx + 0.05); a.set_ylim(-0.05, Ly + 0.05)
    return a
draw_mesh([0.07, 0.40, 0.52, 0.31], lw=0.55)
ax.text(0.07, 0.372, "full-order model", color=ACC, fontsize=10, family=serif, style="italic")

# --- arrow
ax.annotate("", xy=(0.665, 0.555), xytext=(0.595, 0.555),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=18))

# --- right: three modes
for i, k in enumerate((0, 1, 2)):
    u = modes[k]
    f = np.abs(u) / np.abs(u).max()
    draw_mesh([0.70, 0.62 - i * 0.105, 0.24, 0.09], field=f, cmap=CM)
ax.text(0.70, 0.372, "reduced basis", color=ACC, fontsize=10, family=serif, style="italic")

# --- type
ax.text(0.07, 0.905, "ARCHITECTING", color=INK, fontsize=40, family=serif, weight="bold")
ax.text(0.07, 0.862, "COMPUTATIONAL", color=INK, fontsize=40, family=serif, weight="bold")
ax.text(0.07, 0.819, "MECHANICS", color=INK, fontsize=40, family=serif, weight="bold")
ax.plot([0.07, 0.30], [0.795, 0.795], color=ACC, lw=2.2)
ax.text(0.07, 0.76, "From-Scratch Finite Element Analysis", color=ACC, fontsize=15, family=serif)
ax.text(0.07, 0.735, "and Reduced-Order Modeling in Python", color=ACC, fontsize=15, family=serif)
ax.plot([0.07, 0.93], [0.30, 0.30], color=ACC, lw=0.6)
ax.text(0.07, 0.26, "fea_engine  ·  rom_engine", color=INK, fontsize=13, family="DejaVu Sans Mono")
ax.text(0.07, 0.235, "Verified implementation of reduced-order models\nover finite element analysis for structures", color=ACC, fontsize=11, family=serif, linespacing=1.5, va="top")
ax.text(0.07, 0.07, "ABHIJEET KUMAR", color=INK, fontsize=18, family=serif, weight="bold")
fig.savefig("/tmp/cover/cover.pdf", facecolor=BG)
fig.savefig("/tmp/cover/cover.png", dpi=150, facecolor=BG)
print("ok")
