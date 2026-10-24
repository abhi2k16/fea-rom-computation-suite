import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; ORG='#d9822b'; GRN='#2e8b57'; VIO='#6a3d9a'; GREY='#8a96a3'
K=np.array([[4.0,3.2],[3.2,3.0]]); 
th=np.radians(8); v=np.array([np.cos(th),np.sin(th)])
u=np.array([2.2,2.1])
E=v*(v@u); G=v*(v@K@u)/(v@K@v)
nK=lambda e:np.sqrt(e@K@e); n2=lambda e:np.linalg.norm(e)
eE,eG=u-E,u-G
fig=plt.figure(figsize=(6.6,3.3)); gs=fig.add_gridspec(1,2,width_ratios=[1.7,1],wspace=0.18,left=0.02,right=0.99,top=0.9,bottom=0.12)
ax=fig.add_subplot(gs[0]); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(a) Same subspace, two projections',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
t=np.linspace(-1.2,5.6,2); ax.plot(t*v[0],t*v[1],color=VIO,lw=2); ax.text(5.6*v[0]+0.05,5.6*v[1]-0.1,r'span$(\mathbf{V})$',color=VIO,fontsize=8.5,ha='left')
phi=np.linspace(0,2*np.pi,300)
ax.plot(u[0]+n2(eE)*np.cos(phi),u[1]+n2(eE)*np.sin(phi),color=ORG,lw=1,ls=':')
# energy ellipse: (x-u)^T K (x-u)=nK(eG)^2
L=np.linalg.cholesky(K); R=np.linalg.inv(L.T)   # x-u = R z, |z|=nK(eG)
z=nK(eG)*np.stack([np.cos(phi),np.sin(phi)]); ell=(R@z).T+u
ax.plot(ell[:,0],ell[:,1],color=GRN,lw=1,ls=':')
ax.plot([u[0],E[0]],[u[1],E[1]],color=ORG,lw=1.8); ax.plot([u[0],G[0]],[u[1],G[1]],color=GRN,lw=1.8)
# right-angle mark at E
a=eE/n2(eE); b=v; s=0.16
ax.add_patch(Polygon([E+s*a,E+s*a+s*b,E+s*b],closed=False,fill=False,ec=ORG,lw=.9))
ax.plot(*u,'o',color=NAVY,ms=5,zorder=5); ax.plot(*E,'o',color=ORG,ms=5,zorder=5); ax.plot(*G,'o',color=GRN,ms=5,zorder=5)
ax.text(u[0]-0.1,u[1]+0.1,r'$\mathbf{u}$ (full-order solution)',color=NAVY,fontsize=8,ha='right',fontweight='bold')
ax.annotate('Euclidean projection\n'+r'$\mathbf{V}\mathbf{V}^T\mathbf{u}$: error $\perp$ subspace'+'\n(what the POD error bounds)',E,(E[0]-2.0,E[1]-1.4),color=ORG,fontsize=7.5,ha='center',arrowprops=dict(arrowstyle='-',color=ORG,lw=.8))
ax.annotate('Galerkin solution $\\mathbf{V}\\mathbf{q}$:\nerror $K$-orthogonal to\nthe subspace (best in energy norm)',G,(G[0]+1.9,G[1]-1.6),color=GRN,fontsize=7.5,ha='center',arrowprops=dict(arrowstyle='-',color=GRN,lw=.8))
ax.text(u[0]-n2(eE)*0.9,u[1]+n2(eE)*0.95,'circle: equal\nEuclidean error',color=ORG,fontsize=7,ha='right',va='bottom')
ax.text(ell[:,0].max()+0.05,u[1]+0.3,'ellipse: equal\nenergy error',color=GRN,fontsize=7,ha='left')
ax.set_xlim(-2.0,6.5); ax.set_ylim(-1.9,4.6)
# (b) errors
b_=fig.add_subplot(gs[1]); vals=np.array([[n2(eE),n2(eG)],[nK(eE),nK(eG)]])
w=0.34; xs=np.arange(2)
b_.bar(xs-w/2,vals[:,0],w,color=ORG,label='Euclidean projection'); b_.bar(xs+w/2,vals[:,1],w,color=GRN,label='Galerkin solution')
for i in range(2):
    for j,c in enumerate([ORG,GRN]): b_.text(xs[i]+(j-.5)*w,vals[i,j]+0.05,f'{vals[i,j]:.2f}',ha='center',fontsize=7,color=c)
b_.set_xticks(xs); b_.set_xticklabels([r'Euclidean norm'+'\n'+r'$\|\mathbf{u}-\hat{\mathbf{u}}\|_2$',r'energy norm'+'\n'+r'$\|\mathbf{u}-\hat{\mathbf{u}}\|_K$'],fontsize=7.5)
b_.set_yticks([]); b_.legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,1.02)); b_.set_ylim(0,vals.max()*1.45)
for s in ['top','right','left']: b_.spines[s].set_visible(False)
b_.set_title('(b) Each is best in its own norm',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch08_projection.pdf'),bbox_inches='tight')
print(vals)
