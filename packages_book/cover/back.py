import numpy as np, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from fea_engine.mesh import hole_in_rectangle_mesh

BG, INK, ACC = "#040C16", "#FFFFFF", "#8DA9C4"
serif, mono = "DejaVu Serif", "DejaVu Sans Mono"
fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
grad = np.linspace(0, 1, 256)[:, None, None]
top, low = np.array([3, 9, 18]) / 255.0, np.array([8, 28, 52]) / 255.0
ax.imshow(top + (low - top) * grad ** 1.8 * np.ones((1, 2, 1)), extent=[0, 1, 0, 1], aspect="auto", zorder=0)

# faint mesh band at the foot of the page (same plate-with-hole mesh as the front)
m = hole_in_rectangle_mesh(3.0, 2.0, (1.2, 1.0), 0.45, 6, 24)
X, E = m.nodes, m.elements
W = 0.34; sx = W / 3.0; sy = sx * 8.27 / 11.69; x0, y0 = 0.09, 0.035
segs = []
for e in E:
    for i in range(4):
        p, q = e[i], e[(i + 1) % 4]
        segs.append([(x0 + sx * X[p, 0], y0 + sy * X[p, 1]), (x0 + sx * X[q, 0], y0 + sy * X[q, 1])])
for lw, al in ((2.2, 0.05), (0.4, 0.40)):
    ax.add_collection(LineCollection(segs, colors="#8DB9E8", linewidths=lw, alpha=al, zorder=1))

L = 0.09
ax.text(L, 0.90, "From mesh to reduced model,", color=INK, fontsize=24, family=serif, weight="bold")
ax.text(L, 0.867, "every step in the open.", color=INK, fontsize=24, family=serif, weight="bold")
ax.plot([L, L + 0.23], [0.846, 0.846], color=ACC, lw=2.2)

blurb = ("This book shows how a reduced-order modelling package is built on top of a finite element "
         "package for the analysis of structures, and checks, one claim at a time, that both do what "
         "is said of them. It is written for readers who already know the finite element method and "
         "the idea of model reduction, and who want to see how the two meet in working code.")
ax.text(L, 0.815, "\n".join(textwrap.wrap(blurb, 62)), color=INK, fontsize=12, family=serif, linespacing=1.7, va="top")

items = [("Full-order models", "beams, plates, shells and solids; nonlinear statics, dynamics, contact"),
         ("Reduced-order models", "POD, balanced truncation, Krylov, frequency-domain, component mode synthesis, hyper-reduction"),
         ("Verification", "nine published studies rebuilt from the packages, with an honest account of how each came out"),
         ("Runnable listings", "every Try it listing was executed and its output printed under the code")]
y = 0.665
for head, body in items:
    ax.text(L, y, head, color=INK, fontsize=12, family=serif, weight="bold", va="top")
    ax.text(L, y - 0.020, "\n".join(textwrap.wrap(body, 70)), color=ACC, fontsize=10.5, family=serif, linespacing=1.5, va="top")
    y -= 0.062 if len(body) <= 70 else 0.078

ax.plot([L, 1 - L], [0.355, 0.355], color=ACC, lw=0.6, alpha=0.5)
ax.text(L, 0.337, "ABOUT THE AUTHOR", color=ACC, fontsize=9.5, family=serif, weight="bold", va="top")
bio = ("Abhijeet Kumar is a mechanical engineer and computational researcher with a Ph.D. from IIT Bombay, "
       "working on computational mechanics, finite element analysis and scientific machine learning.")
ax.text(L, 0.318, "\n".join(textwrap.wrap(bio, 74)), color=INK, fontsize=10.5, family=serif, linespacing=1.6, va="top")
ax.text(L, 0.255, "github.com/abhi2k16", color=ACC, fontsize=10, family=mono, va="top")
ax.text(L, 0.236, "linkedin.com/in/abhijeet-kumar-phd-a689b429", color=ACC, fontsize=10, family=mono, va="top")
ax.text(1 - L, 0.090, "fea_engine 1.0.1\nrom_engine 0.2.0", color=INK, fontsize=10, family=mono, ha="right", va="top", linespacing=1.6)
ax.text(1 - L, 0.030, "code under the MIT license", color=ACC, fontsize=8.5, family=serif, ha="right", style="italic")
fig.savefig("/tmp/back.pdf", facecolor=BG); fig.savefig("/tmp/back.png", dpi=150, facecolor=BG)
