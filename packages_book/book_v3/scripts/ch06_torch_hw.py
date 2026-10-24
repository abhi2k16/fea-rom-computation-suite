import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; ORG='#d9822b'; GREY='#8a96a3'
def box(ax,x,y,w,h,fc,ec,lw=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.12',fc=fc,ec=ec,lw=lw))
fig=plt.figure(figsize=(6.8,3.7))
ax=fig.add_axes([0.01,0.02,0.52,0.95]); ax.set_xlim(0,10); ax.set_ylim(0,10.5); ax.axis('off')
ax.text(0,10.2,'(a) Where data and arithmetic live (typical orders)',fontweight='bold',fontsize=8,va='top',color=NAVY)
# host
box(ax,0.0,1.0,3.9,7.9,'#eef4fa',BLUE,1.3); ax.text(1.95,8.55,'Host: CPU',ha='center',va='center',fontweight='bold',color=BLUE)
for i in range(4):
    for j in range(2):
        box(ax,0.35+i*0.88,6.0-j*1.2,0.72,0.95,'#cfe2f3',BLUE,0.8)
ax.text(1.95,7.45,'few fast cores, each with\nits own cache and thread',ha='center',va='center',fontsize=7)
box(ax,0.35,2.0,3.2,1.5,'white',BLUE,0.8); ax.text(1.95,2.75,'RAM: NumPy arrays,\nSciPy sparse matrix',ha='center',va='center',fontsize=7)
ax.text(1.95,1.35,'typically 10$^1$-10$^2$ GB, ~10$^2$ GB/s',ha='center',va='center',fontsize=6.5,color='#555')
# device
box(ax,6.1,1.0,3.9,7.9,'#fbefe8',ORG,1.3); ax.text(8.05,8.55,'Device: GPU',ha='center',va='center',fontweight='bold',color=ORG)
for i in range(7):
    for j in range(4):
        ax.add_patch(Rectangle((6.35+i*0.5,5.1-j*0.4),0.36,0.3,fc='#f6cfae',ec=ORG,lw=0.5))
ax.text(8.05,7.55,'thousands of simple units,\nsame instruction on many data',ha='center',va='center',fontsize=7)
box(ax,6.35,2.0,3.4,1.5,'white',ORG,0.8); ax.text(8.05,2.75,'GPU memory: tensors\n(CSR arrays, vectors)',ha='center',va='center',fontsize=7)
ax.text(8.05,1.35,'typically 10$^1$ GB, ~10$^2$-10$^3$ GB/s',ha='center',va='center',fontsize=6.5,color='#555')
# link
ax.annotate('',(6.05,3.1),(3.95,3.1),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
ax.annotate('',(3.95,2.4),(6.05,2.4),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
ax.text(5.0,4.1,'PCIe link',ha='center',color=RED,fontweight='bold',fontsize=7.5)
ax.text(5.0,3.5,'copy in',ha='center',color=RED,fontsize=7); ax.text(5.0,1.7,'copy out',ha='center',color=RED,fontsize=7)
ax.text(5.0,0.35,'the link (~10$^1$ GB/s) is slower than either memory',ha='center',fontsize=7,color=RED)
# (b) speedup from table
d=np.array([1088,4224,16640,66048.]); sp=np.array([7.0,40.4,275,1680]); cpu=np.array([18.7,86.9,221,1894]); cu=np.array([65.3,237,346,904])
ax2=fig.add_axes([0.63,0.17,0.355,0.68])
ax2.axhline(1,color='#444',lw=1); ax2.axhspan(0.01,1,color='#f3d9d6',alpha=0.5,lw=0); 
ax2.plot(d,sp/cpu,'o-',color=BLUE,lw=1.8,ms=4,label='CG on CPU threads')
ax2.plot(d,sp/cu,'s-',color=ORG,lw=1.8,ms=4,label='CG on GPU (CUDA)')
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_ylim(0.08,3); ax2.set_xlim(7e2,1e5); ax2.set_xticks([1e3,1e4,1e5])
ax2.set_xlabel('free degrees of freedom'); ax2.set_ylabel('SciPy time / torch time')
ax2.text(8.5e2,1.12,'torch faster',fontsize=7,va='bottom',color=GRN); ax2.text(8.5e2,0.9,'torch slower',fontsize=7,va='top',color=RED)
ax2.text(8.5e2,0.88*0.88**0,'',fontsize=7)
ax2.set_title('(b) Same solve, same plate',fontweight='bold',fontsize=8,loc='left',color=NAVY)
ax2.legend(frameon=False,fontsize=7,loc='lower right'); ax2.grid(alpha=0.25,which='both',lw=0.4)
for s in ('top','right'): ax2.spines[s].set_visible(False)
fig.savefig('preview_figs/ch06_torch_hw.pdf'); fig.savefig('preview_figs/ch06_torch_hw.png',dpi=130)
