import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("blu", ["#040C16", "#0B2A4A", "#2F7FC1", "#8DB9E8", "#F4F9FF"])
from fea_engine import FESystem, Quad4Laplace
from fea_engine.mesh import hole_in_rectangle_mesh

BG, INK, ACC = "#040C16", "#FFFFFF", "#8DA9C4"
bgc = np.array([4, 12, 22]) / 255.0
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
grad = np.linspace(0, 1, 256)[:, None, None]                      # vertical gradient: darker top, faint blue glow low
top, low = np.array([3, 9, 18]) / 255.0, np.array([8, 28, 52]) / 255.0
ax.imshow(top + (low - top) * grad[::-1] ** 1.8 * np.ones((1, 2, 1)), extent=[0, 1, 0, 1], aspect="auto", zorder=0)

# original base art: the first membrane mode of the plate with a hole, drawn as a glowing wireframe surface
from mpl_toolkits.mplot3d.art3d import Line3DCollection
mesh_b = hole_in_rectangle_mesh(Lx, Ly, (1.5, 1.0), 0.40, 10, 40)
Xb, Eb = mesh_b.nodes, mesh_b.elements
fb = FESystem(mesh_b, Quad4Laplace(), sparse=True)
fb.assemble_stiffness(1.0); fb.assemble_lumped_mass(1.0)
eb = np.flatnonzero((Xb[:, 0] < 1e-9) | (Xb[:, 0] > Lx - 1e-9) | (Xb[:, 1] < 1e-9) | (Xb[:, 1] > Ly - 1e-9))
fb.fix_dofs(eb, ["u"]); frb = np.asarray(fb.free_dofs)
Kb = sp.csr_matrix(fb.K)[frb][:, frb]; Mb = sp.diags(sp.csr_matrix(fb.M_lumped).diagonal()[frb]).tocsr()
wb, Vb = spl.eigsh(Kb, k=2, M=Mb, sigma=0, which="LM")
zb = np.zeros(len(Xb)); zb[frb] = Vb[:, 1]; zb = zb / np.abs(zb).max()
a3 = fig.add_axes([-0.05, 0.14, 1.10, 0.36], projection="3d", facecolor="none")
a3.set_axis_off(); a3.view_init(elev=24, azim=-72)
segs, cols = [], []
for e in Eb:
    for i0 in range(4):
        p, q = e[i0], e[(i0 + 1) % 4]
        segs.append([(Xb[p, 0], Xb[p, 1], 0.9 * zb[p]), (Xb[q, 0], Xb[q, 1], 0.9 * zb[q])])
        cols.append(0.5 * (zb[p] + zb[q]))
cols = (np.array(cols) + 1) / 2
for lw, al in ((3.0, 0.05), (1.4, 0.12), (0.45, 0.85)):
    lc = Line3DCollection(segs, linewidths=lw, alpha=al, colors=CM(0.25 + 0.75 * cols))
    a3.add_collection3d(lc)
a3.set_xlim(0, Lx); a3.set_ylim(0, Ly); a3.set_zlim(-1.0, 1.4); a3.set_box_aspect((3.0, 2.0, 1.1))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# --- left: the full-order mesh
def draw_mesh(box, field=None, lw=0.5, cmap="Blues", alpha=1.0):
    a = fig.add_axes(box); a.set_aspect("equal"); a.axis("off"); a.set_facecolor("none")
    polys = X[E]
    if field is None:
        a.add_collection(PolyCollection(polys, facecolors="none", edgecolors=INK, linewidths=lw))
    else:
        f = field[E].mean(axis=1)
        fc = CM(f.clip(0, 1)); fc[:, 3] = np.clip(f * 1.6, 0, 1) ** 0.8
        a.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="none", linewidths=0))
    a.set_xlim(-0.05, Lx + 0.05); a.set_ylim(-0.05, Ly + 0.05)
    return a
draw_mesh([0.07, 0.50, 0.50, 0.245], lw=0.55)
ax.text(0.07, 0.478, "full-order model", color=ACC, fontsize=10, family=serif, style="italic")

# --- arrow
ax.annotate("", xy=(0.655, 0.625), xytext=(0.590, 0.625),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=18))

# --- right: three modes
for i, k in enumerate((0, 1, 2)):
    u = modes[k]
    f = np.abs(u) / np.abs(u).max()
    draw_mesh([0.675, 0.675 - i * 0.087, 0.20, 0.08], field=f, cmap=CM)
ax.text(0.675, 0.492, "reduced basis", color=ACC, fontsize=10, family=serif, style="italic")

# --- type
ax.text(0.07, 0.935, "IMPLEMENTING", color=INK, fontsize=29, family=serif, weight="bold")
ax.text(0.07, 0.901, "REDUCED-ORDER MODELING", color=INK, fontsize=29, family=serif, weight="bold")
ax.text(0.07, 0.867, "IN COMPUTATIONAL", color=INK, fontsize=29, family=serif, weight="bold")
ax.text(0.07, 0.833, "MECHANICS", color=INK, fontsize=29, family=serif, weight="bold")
ax.plot([0.07, 0.30], [0.815, 0.815], color=ACC, lw=2.2)
ax.text(0.07, 0.789, "Theory, Algorithms, and From-Scratch", color=ACC, fontsize=15, family=serif)
ax.text(0.07, 0.765, "Implementation in Python", color=ACC, fontsize=15, family=serif)
ax.text(0.07, 0.135, "fea_engine  ·  rom_engine", color=INK, fontsize=13, family="DejaVu Sans Mono")
ax.text(0.07, 0.108, "Verified implementation of reduced-order models\nover finite element analysis for structures", color=ACC, fontsize=11, family=serif, linespacing=1.5, va="top")
ax.text(0.07, 0.045, "ABHIJEET KUMAR", color=INK, fontsize=18, family=serif, weight="bold")
fig.savefig("/tmp/cover/cover.pdf", facecolor=BG)
fig.savefig("/tmp/cover/cover.png", dpi=150, facecolor=BG)
print("ok")
