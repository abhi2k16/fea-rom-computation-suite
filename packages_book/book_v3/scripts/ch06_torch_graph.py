import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GRN='#2e8b57'; ORG='#d9822b'; GREY='#8a96a3'
fig,axs=plt.subplots(1,2,figsize=(6.8,3.3),gridspec_kw={'width_ratios':[1,1.05]}); plt.subplots_adjust(left=0.01,right=0.99,top=0.9,bottom=0.02,wspace=0.06)
def node(ax,x,y,t,fc,ec,w=1.5,h=0.8):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0,rounding_size=0.15',fc=fc,ec=ec,lw=1.2)); ax.text(x,y,t,ha='center',va='center',fontsize=8)
def arr(ax,a,b,c,rad=0.0,ls='-'):
    ax.annotate('',b,a,arrowprops=dict(arrowstyle='-|>',color=c,lw=1.5,shrinkA=0,shrinkB=0,connectionstyle=f'arc3,rad={rad}',linestyle=ls))
# (a) graph of a chain
ax=axs[0]; ax.set_xlim(0,10); ax.set_ylim(0,9); ax.axis('off')
ax.set_title('(a) A graph is recorded as the code runs',fontweight='bold',fontsize=8,loc='left',color=NAVY)
node(ax,1.0,6.6,'$\\theta$','#e4eef8',BLUE,w=1.2,h=0.8); node(ax,1.0,4.2,'$\\mathbf{u}$','#e4eef8',BLUE,w=1.2,h=0.8)
node(ax,4.0,5.6,'$\\mathbf{g}=\\theta\\mathbf{u}$','#fbefe8',ORG,w=1.9); node(ax,6.9,5.6,'$\\mathbf{f}=\\tanh\\mathbf{g}$','#fbefe8',ORG,w=2.1); node(ax,9.2,5.6,'$L$','#eaf5ee',GRN,w=1.1)
for a_,b_ in [((1.6,6.6),(3.3,5.95)),((1.6,4.2),(3.3,5.3))]: arr(ax,a_,b_,NAVY)
arr(ax,(4.95,5.75),(5.85,5.75),NAVY); arr(ax,(7.95,5.75),(8.65,5.75),NAVY)
for a_,b_ in [((3.5,5.95),(1.7,6.4)),((3.5,5.2),(1.7,4.0))]: arr(ax,a_,b_,RED,ls='--')
arr(ax,(5.85,5.45),(4.95,5.45),RED,ls='--'); arr(ax,(8.65,5.45),(7.95,5.45),RED,ls='--')
ax.text(0.2,8.5,'solid: forward, values are computed and each\noperation stores what its backward step needs',fontsize=7,color=NAVY,ha='left',va='center')
ax.text(0.2,7.7,'dashed: backward, chain rule in reverse order',fontsize=7,color=RED,ha='left',va='center')
ax.text(5.0,3.3,'$\\partial L/\\partial\\mathbf{f}=\\mathbf{w}$  $\\rightarrow$  $\\partial L/\\partial\\mathbf{g}=(1-\\mathbf{f}^2)\\odot\\mathbf{w}$  $\\rightarrow$  $\\partial L/\\partial\\theta,\\ \\partial L/\\partial\\mathbf{u}$',fontsize=7,color=RED,ha='center',va='center')
ax.text(5.0,1.6,'Example: $L=\\mathbf{w}^T\\tanh(\\theta\\mathbf{u})$. Calling L.backward() fills\nthe gradient of every leaf in one reverse sweep,\nat a cost of the order of one forward evaluation.',fontsize=7,ha='center',va='center')
# (b) implicit solve
ax=axs[1]; ax.set_xlim(0,11); ax.set_ylim(0,9); ax.axis('off')
ax.set_title('(b) Through a Newton solve: one adjoint solve',fontweight='bold',fontsize=8,loc='left',color=NAVY)
node(ax,0.9,6.4,'$\\theta$','#e4eef8',BLUE,w=1.1)
ax.add_patch(FancyBboxPatch((2.0,5.0),5.0,3.0,boxstyle='round,pad=0,rounding_size=0.2',fc='#f4f4f4',ec=GREY,lw=1.2,ls='--'))
ax.text(4.5,7.55,'Newton loop (not recorded)',ha='center',va='center',fontsize=7,color='#555')
for x,t in [(2.9,'$\\mathbf{u}_0$'),(4.5,'$\\mathbf{u}_1$'),(6.1,'$\\cdots$')]: node(ax,x,6.4,t,'#ececec',GREY,w=1.0,h=0.7)
for x in (3.5,5.1): ax.annotate('',(x+0.5,6.4),(x,6.4),arrowprops=dict(arrowstyle='-|>',color=GREY,lw=1.2))
ax.text(4.5,5.4,'$\\mathbf{K}\\Delta\\mathbf{u}=-\\mathbf{r}$ at each step',ha='center',va='center',fontsize=7,color='#555')
node(ax,9.4,6.4,'$\\mathbf{u}^*$:  $\\mathbf{r}=0$','#fbefe8',ORG,w=2.6,h=0.9)
arr(ax,(1.5,6.4),(2.0,6.4),NAVY); arr(ax,(7.0,6.4),(8.1,6.4),NAVY)
node(ax,9.4,3.6,'$L(\\mathbf{u}^*)$','#eaf5ee',GRN,w=1.8); arr(ax,(9.4,5.95),(9.4,4.0),NAVY)
arr(ax,(8.45,3.6),(0.9,3.6),RED,ls='--')
node(ax,4.7,2.5,'adjoint solve  $\\mathbf{K}^T\\boldsymbol{\\lambda}=\\partial L/\\partial\\mathbf{u}$','#fdecea',RED,w=5.6,h=0.8)
ax.text(4.7,1.2,'$\\mathrm{d}L/\\mathrm{d}\\theta=-\\boldsymbol{\\lambda}^T\\,\\partial\\mathbf{r}/\\partial\\theta$',fontsize=8,color=RED,ha='center',va='center')
ax.text(4.7,0.35,'the backward sweep does not repeat the iterations',fontsize=7,color=RED,ha='center',va='center')
fig.savefig('preview_figs/ch06_torch_graph.pdf'); fig.savefig('preview_figs/ch06_torch_graph.png',dpi=130)
