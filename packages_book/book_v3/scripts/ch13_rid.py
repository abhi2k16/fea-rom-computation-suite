import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; VIO='#6a3d9a'; GREY='#8a96a3'; GRN='#2e8b57'; ORG='#d9822b'
nx,ny=10,7
# selected dofs (interpolation points of the bases): nodes (i,j) with j=0 bottom row
sel=[(3,1),(4,2),(5,1),(6,2),(7,1),(5,3)]
rid=set()
for (i,j) in sel:
    for di in (-1,0):
        for dj in (-1,0):
            ci,cj=i+di,j+dj
            if 0<=ci<nx and 0<=cj<ny: rid.add((ci,cj))
# contact zone: bottom nodes i=4..8 touch obstacle; add their cells
for i in range(3,8):
    for di in (-1,0):
        if 0<=i+di<nx: rid.add((i+di,0))
nodes=[(i,j) for i in range(nx+1) for j in range(ny+1)]
def adj(i,j): return [(i+di,j+dj) for di in (-1,0) for dj in (-1,0) if 0<=i+di<nx and 0<=j+dj<ny]
P=[n for n in nodes if all(c in rid for c in adj(*n))]
ext=[n for n in nodes if any(c in rid for c in adj(*n)) and n not in P]
contact=[(i,0) for i in range(3,8)]
fig,axs=plt.subplots(1,3,figsize=(6.9,3.3)); plt.subplots_adjust(wspace=0.08,left=0.01,right=0.99,top=0.92,bottom=0.25)
def base(ax,title):
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.4,nx+0.4); ax.set_ylim(-0.8,ny+0.3)
    ax.set_title(title,fontsize=8,fontweight='bold',color=NAVY,loc='left',pad=2)
    ax.plot([-0.3,nx+0.3],[-0.45,-0.45],color=RED,lw=3,solid_capstyle='butt')
def cells(ax,fill=None,only=None):
    for i in range(nx):
        for j in range(ny):
            f='white'; ec='#cfd5db'
            if only is not None and (i,j) in only: f=fill; ec=VIO
            ax.add_patch(Rectangle((i,j),1,1,fc=f,ec=ec,lw=.6))
def arrows(ax,col=RED):
    for (i,j) in contact: ax.annotate('',(i,-0.38),(i,-0.05-0.0),arrowprops=dict(arrowstyle='-|>',color=col,lw=1.1,mutation_scale=7))
# (a) full model
ax=axs[0]; base(ax,'(a) Full-order model'); cells(ax)
for (i,j) in nodes: ax.plot(i,j,'.',color='#9aa5b0',ms=3)
arrows(ax); ax.text(0.5,-0.02,'every element and every\ncontact row is evaluated',ha='center',va='top',fontsize=8,color='#333',transform=ax.transAxes)
# (b) selection
ax=axs[1]; base(ax,'(b) Chosen dofs and their elements'); cells(ax,fill='#e8def5',only=rid)
for (i,j) in sel: ax.plot(i,j,'*',color=ORG,ms=8,zorder=5)
arrows(ax); ax.text(0.5,-0.02,'stars: dofs chosen from the bases;\nRID (shaded): their elements\nand the contact zone',ha='center',va='top',fontsize=8,color='#333',transform=ax.transAxes)
# (c) hybrid reduced problem
ax=axs[2]; base(ax,'(c) Hybrid reduced problem'); cells(ax,fill='#e8def5',only=rid)
for (i,j) in nodes:
    if (i,j) not in P and (i,j) not in ext: ax.plot(i,j,'.',color='#c9d0d6',ms=2)
for (i,j) in P: ax.plot(i,j,'o',color=VIO,ms=4.2,zorder=5)
for (i,j) in ext: ax.plot(i,j,'o',mfc='white',mec=VIO,ms=4.2,mew=1,zorder=5)
arrows(ax); 
ax.text(0.5,-0.02,'filled: balance equation at $P$;\nhollow: interface, taken from $\\mathbf{V}\\mathbf{q}$;\narrows: contact forces on the RID',ha='center',va='top',fontsize=8,color='#333',transform=ax.transAxes)
fig.text(0.5,0.015,'Contact forces on the RID are unknowns of the reduced problem together with the amplitudes $\\mathbf{q}$: no dual basis is needed.',ha='center',fontsize=8,color=NAVY,fontweight="bold")
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch13_rid.pdf'),bbox_inches='tight')
print(len(rid),len(P),len(ext))
