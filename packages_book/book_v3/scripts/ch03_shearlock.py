import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; GREY='#8a96a3'
b,h,k=1.0,0.5,0.4           # half length, half height, curvature (exaggerated)
def exact(x,y): return x+k*x*y, y-k*x**2/2
def q4(x,y):    return x+k*x*y, y
def edge(f,p0,p1,n=60):
    t=np.linspace(0,1,n); x=p0[0]+(p1[0]-p0[0])*t; y=p0[1]+(p1[1]-p0[1])*t; return f(x,y)
def draw(ax,f,col,fill):
    pts=[]
    for p0,p1 in [((-b,-h),(b,-h)),((b,-h),(b,h)),((b,h),(-b,h)),((-b,h),(-b,-h))]:
        X,Y=edge(f,p0,p1); pts+=list(zip(X,Y))
    ax.add_patch(Polygon(pts,closed=True,fc=fill,ec=col,lw=1.6,zorder=2))
    # material lines: vertical sections and mid-line
    for xs in np.linspace(-b,b,5)[1:-1]:
        X,Y=edge(f,(xs,-h),(xs,h)); ax.plot(X,Y,color=col,lw=0.5,alpha=.5,zorder=3)
    X,Y=edge(f,(-b,0),(b,0)); ax.plot(X,Y,color=col,lw=0.5,alpha=.5,zorder=3)
def setup(ax,title,sub):
    ax.set_aspect('equal'); ax.set_xlim(-1.55,1.55); ax.set_ylim(-1.55,0.9); ax.axis('off')
    ax.set_title(title,fontsize=8.5,fontweight='bold',color=NAVY,loc='left',pad=2)
    ax.text(0,-0.78,sub,ha='center',va='top',fontsize=7.2,color='#333',linespacing=1.3)
fig=plt.figure(figsize=(6.6,5.6))
gs=fig.add_gridspec(2,3,height_ratios=[1.0,1],hspace=0.3,wspace=0.05)
gb=fig.add_gridspec(2,3,height_ratios=[1.0,1],hspace=0.3,wspace=0.45)
# (a) undeformed
ax=fig.add_subplot(gs[0,0]); setup(ax,'(a) Undeformed element','width $2b$, height $2h$\ncorners are $90^\\circ$')
draw(ax,lambda x,y:(x,y),NAVY,'#eaf1f8')
ax.annotate('',(-b,0.68),(b,0.68),arrowprops=dict(arrowstyle='<->',color=GREY,lw=.8)); ax.text(0,0.72,'$2b$',ha='center',fontsize=7.5,color=GREY)
ax.annotate('',(1.18,-h),(1.18,h),arrowprops=dict(arrowstyle='<->',color=GREY,lw=.8)); ax.text(1.24,0,'$2h$',va='center',fontsize=7.5,color=GREY)
for (x,y) in [(-b,-h),(b,-h),(b,h),(-b,h)]:
    ax.add_patch(Arc((x,y),0.28,0.28,theta1={(-b,-h):0,(b,-h):90,(b,h):180,(-b,h):270}[(x,y)],theta2={(-b,-h):90,(b,-h):180,(b,h):270,(-b,h):360}[(x,y)],color=NAVY,lw=.7))
    ax.plot([x+(0.07 if x<0 else -0.07)],[y+(0.07 if y<0 else -0.07)],'.',color=NAVY,ms=2)
# (b) exact
ax=fig.add_subplot(gs[0,1]); setup(ax,'(b) Exact bending','edges curve; sections stay\nperpendicular to the axis')
draw(ax,lambda x,y:(x,y),GREY,'none')
draw(ax,exact,BLUE,'#e3effa')
ax.text(0,-1.45,'$\\gamma=0$ everywhere',color=BLUE,fontsize=8,ha='center',fontweight='bold')
# right-angle marker at top-right corner
cx,cy=exact(b,h); 
ax.plot([cx-0.12,cx-0.12-0.0],[cy,cy],alpha=0)
# (c) Q4
ax=fig.add_subplot(gs[0,2]); setup(ax,'(c) Bilinear element','edges stay straight;\ncorners are sheared')
draw(ax,lambda x,y:(x,y),GREY,'none')
draw(ax,q4,RED,'#fbe9e7')
cx,cy=q4(b,h)
ax.add_patch(Arc((cx,cy),0.5,0.5,theta1=180,theta2=270-np.degrees(np.arctan(k*b)),color=RED,lw=1.1))
# angle between top edge (pointing left) and side edge (pointing down-right-tilted)
ax.text(cx-0.12,cy-0.34,'$90^\\circ\\!-\\!\\gamma$',color=RED,fontsize=7.5,ha='center')
ax.text(0,-1.45,'parasitic shear $\\gamma=\\kappa x$',color=RED,fontsize=8,ha='center',fontweight='bold')
# bottom row
ax=fig.add_subplot(gb[1,0:2]); x=np.linspace(-b,b,100)
ax.plot(x,0*x,color=BLUE,lw=1.8,label='exact bending')
ax.plot(x,k*x,color=RED,lw=1.8,label='bilinear element')
ax.axvline(0,color=GREY,lw=.6,ls=':'); ax.set_xlabel('position along the element, $x$'); ax.set_ylabel('shear strain $\\gamma$')
ax.set_xticks([-b,0,b]); ax.set_xticklabels(['$-b$','0','$b$']); ax.set_yticks([0]); ax.set_ylim(-0.6,0.6)
ax.fill_between(x,0,k*x,color=RED,alpha=.12); ax.legend(frameon=False,fontsize=7.5,loc='upper left')
ax.set_title('(d) The shear strain that should be zero',fontsize=8.5,fontweight='bold',color=NAVY,loc='left',pad=2)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.text(0.05,-0.45,'zero only on the line $x=0$',fontsize=7,color='#333',ha='left')
ax=fig.add_subplot(gb[1,2]); r=np.logspace(0,1.3,50); nu=0.3
ax.loglog(r,r**2/(2*(1+nu)),color=RED,lw=1.8)
ax.axhline(1,color=GREY,lw=.6,ls=':')
ax.set_xlabel('aspect ratio $b/h$'); ax.set_ylabel('shear / bending energy')
ax.set_title('(e) Cost of the error',fontsize=8.5,fontweight='bold',color=NAVY,loc='left',pad=2)
ax.text(1.05,90,'grows as $(b/h)^2$:\nthin elements\nlock',fontsize=7,color=RED,va='top')
for s in ['top','right']: ax.spines[s].set_visible(False)
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch03_shearlock.pdf'),bbox_inches='tight')
