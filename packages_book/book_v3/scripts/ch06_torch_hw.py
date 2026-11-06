import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; ORG='#d9822b'; GREY='#6b7785'
def box(ax,x,y,w,h,fc,ec,lw=1.0): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.12',fc=fc,ec=ec,lw=lw))
def num(ax,x,y,n,c=NAVY):
    ax.add_patch(plt.Circle((x,y),0.27,fc=c,ec='none')); ax.text(x,y-0.02,str(n),ha='center',va='center',color='white',fontsize=7,fontweight='bold')
fig=plt.figure(figsize=(7.2,4.3))  # Figure: why a GPU does not always win
ax=fig.add_axes([0.01,0.02,0.58,0.96]); ax.set_xlim(0,10); ax.set_ylim(-0.7,11); ax.axis('off')
ax.text(0,10.8,'(a) What one torch solve does, and where the time goes',fontweight='bold',fontsize=8,va='top',color=NAVY)
# host / device
box(ax,0,6.3,3.5,3.6,'#eef4fa',BLUE,1.3); ax.text(1.75,9.55,'Host: CPU',ha='center',va='center',fontweight='bold',color=BLUE,fontsize=7.5)
ax.text(1.75,8.5,'matrix and load as\nNumPy / SciPy arrays',ha='center',va='center',fontsize=7)
box(ax,6.5,6.3,3.5,3.6,'#fbefe8',ORG,1.3); ax.text(8.25,9.55,'Device: GPU',ha='center',va='center',fontweight='bold',color=ORG,fontsize=7.5)
ax.text(8.25,8.5,'the same data as tensors;\nmany small operations\nper iteration',ha='center',va='center',fontsize=7)
for i in range(7):
    for j in range(2): ax.add_patch(Rectangle((6.8+i*0.43,6.55+j*0.3),0.32,0.22,fc='#f6cfae',ec=ORG,lw=0.5))
# arrows + steps
ax.annotate('',(6.45,8.9),(3.55,8.9),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
ax.annotate('',(3.55,7.1),(6.45,7.1),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.6))
num(ax,5.0,9.55,2,RED); ax.text(5.0,9.15,'copy in',ha='center',fontsize=7,color=RED)
num(ax,5.0,6.45,4,RED); ax.text(5.0,6.8,'copy out',ha='center',fontsize=7,color=RED)
num(ax,0.35,6.7,1,RED); ax.text(0.65,6.7,'build tensors',fontsize=6.5,color=RED,va='center')
num(ax,6.78,7.5,3,GRN); ax.text(7.12,7.5,'iterate',fontsize=6.5,color=GRN,va='center')
ax.text(5.0,5.75,'the PCIe link is slower than the GPU memory (about 112 GB/s on the GTX 1050 used here)',ha='center',fontsize=6,color=GREY)
ax.text(5.0,5.35,'(memory bandwidth from the manufacturer\'s specification, not measured here)',ha='center',fontsize=5.6,color=GREY)
# time bars (schematic)
ax.text(0,4.85,'Schematic of the time of one solve (not to scale). The GPU wins only when\nthe arithmetic it saves is larger than the overhead it adds.',fontsize=7,fontweight='bold',color=NAVY,va='center')
X0=2.5
def bar(y,label,parts):
    ax.text(X0-0.15,y,label,ha='right',va='center',fontsize=7)
    x=X0
    for w,c,t in parts:
        ax.add_patch(Rectangle((x,y-0.28),w,0.56,fc=c,ec='white',lw=0.8))
        x+=w
    return x
ax.text(0,3.85,'small problem',fontsize=7,color=GREY,va='center',style='italic')
e1=bar(3.4,'SciPy (CPU)',[(0.7,GRN,'')]); ax.text(e1+0.1,3.4,'fastest',fontsize=6.5,va='center',color=GRN)
e2=bar(2.75,'torch (GPU)',[(2.1,RED,''),(0.15,GRN,'')]); ax.text(e2+0.1,2.75,'slower: overhead dominates',fontsize=6.5,va='center',color=RED)
ax.text(0,1.95,'large problem',fontsize=7,color=GREY,va='center',style='italic')
e3=bar(1.5,'SciPy (CPU)',[(7.4,GRN,'')])
e4=bar(0.85,'torch (GPU)',[(2.1,RED,''),(2.5,GRN,'')]); ax.text(e4+0.1,0.85,'faster',fontsize=6.5,va='center',color=GRN)
ax.add_patch(Rectangle((2.5,0.0),0.25,0.18,fc=RED,ec='none')); ax.text(2.85,0.09,'overhead: steps 1, 2, 4 and the start-up of every GPU operation',fontsize=6,va='center')
ax.add_patch(Rectangle((2.5,-0.45),0.25,0.18,fc=GRN,ec='none')); ax.text(2.85,-0.36,'arithmetic (the part a GPU speeds up)',fontsize=6,va='center')
# (b) measured
d=np.array([1088,4224,16640,66048.]); sp=np.array([7.0,40.4,275,1680]); cpu=np.array([18.7,86.9,221,1894]); cu=np.array([65.3,237,346,904])
ax2=fig.add_axes([0.675,0.17,0.30,0.70])
ax2.plot(d,sp,'o-',color=GRN,lw=1.8,ms=4); ax2.plot(d,cpu,'^-',color=BLUE,lw=1.5,ms=4); ax2.plot(d,cu,'s-',color=ORG,lw=1.8,ms=4)
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_xlim(7e2,1.2e5); ax2.set_ylim(4,4000); ax2.set_xticks([1e3,1e4,1e5])
ax2.set_xlabel('free degrees of freedom'); ax2.set_ylabel('time of the solve (ms)')
ax2.set_title('(b) Measured, plate model',fontweight='bold',fontsize=8,loc='left',color=NAVY)
ax2.legend(['SciPy','CG, CPU threads','CG, GPU (CUDA)'],frameon=False,fontsize=6.5,loc='lower right')
ax2.text(8.5e2,2300,'GPU: high start-up cost,\nslower growth; it passes SciPy\nonly on the largest mesh',fontsize=6.5,color=ORG,va='center')
ax2.grid(alpha=0.25,which='both',lw=0.4)
for s in ('top','right'): ax2.spines[s].set_visible(False)
import os; _d=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures')
fig.savefig(os.path.join(_d,'ch06_torch_hw.pdf'))
