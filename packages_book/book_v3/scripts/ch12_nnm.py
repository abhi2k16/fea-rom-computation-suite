import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
NAVY='#1b3a5c'; BLUE='#2f7fc1'; RED='#c0392b'; VIO='#6a3d9a'; GREY='#8a96a3'; GRN='#2e8b57'; ORG='#d9822b'
lam,k3=100.0,5e4; w1=np.sqrt(lam)
back=lambda A: np.sqrt(lam+0.75*k3*A**2)/w1
fig,axs=plt.subplots(1,2,figsize=(6.8,3.5)); plt.subplots_adjust(wspace=0.22,left=0.07,right=0.99,top=0.88,bottom=0.15)
# (a) backbone and continuation
ax=axs[0]; A=np.linspace(0,0.06,300)
ax.axvline(1,color=GREY,ls='--',lw=1); ax.text(1.005,0.054,'linear\nfrequency',color='#555',fontsize=7,va='top')
ax.plot(back(A),A,color=BLUE,lw=2.2)
Ap=np.array([0.0,0.012,0.024,0.036,0.048,0.060])
ax.plot(back(Ap),Ap,'o',color=NAVY,ms=4,zorder=5)
# predictor-corrector from point 2
i=1; w0,a0=back(Ap[i]),Ap[i]; da=1e-4; tang=np.array([(back(a0+da)-back(a0-da))/(2*da),1.0]); tang/=np.linalg.norm(tang*np.array([1/0.1,1.0]))
step=0.0105
wp,ap=w0+tang[0]*step,a0+tang[1]*step
ax.annotate('',(wp,ap),(w0,a0),arrowprops=dict(arrowstyle='-|>',color=ORG,lw=1.6,shrinkA=3,shrinkB=0))
ax.plot([wp],[ap],'s',color=ORG,ms=4.5,zorder=6)
ax.annotate('',(back(Ap[i+1]),Ap[i+1]),(wp,ap),arrowprops=dict(arrowstyle='-|>',color=GRN,lw=1.4,shrinkA=0,shrinkB=3,linestyle='-'))
ax.text(w0+0.13,0.0125,'predictor: step along\nthe tangent',color=ORG,fontsize=7,ha='left',va='center')
ax.text(back(Ap[i+1])+0.17,0.0325,'corrector: Newton back\nonto the curve (amplitude fixed)',color=GRN,fontsize=7,ha='left',va='center')
ax.text(1.0+0.025,0.0,'start: linear mode',color=NAVY,fontsize=7,va='center')
ax.text(back(0.06)-0.02,0.0605,'backbone (hardening)',color=BLUE,fontsize=8,ha='right',va='bottom',fontweight='bold')
ax.set_xlabel(r'frequency $\omega/\omega_1$'); ax.set_ylabel('amplitude of the first harmonic'); ax.set_xticks([1]); ax.set_xticklabels(['1']); ax.set_yticks([])
ax.set_xlim(0.95,back(0.06)+0.15); ax.set_ylim(-0.003,0.072)
ax.set_title('(a) Free vibration: the backbone',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s in ['top','right']: ax.spines[s].set_visible(False)
# (b) forced response with a fold
ax=axs[1]; c=1.5; F=0.8
ws=np.linspace(0.9,1.75,1500)*w1
lo=[];mid=[];hi=[]
for w in ws:
    # ((lam-w^2+.75k3 y)^2 + c^2 w^2) y = F^2, y=A^2  -> cubic in y
    p=np.array([ (0.75*k3)**2, 2*(0.75*k3)*(lam-w**2), (lam-w**2)**2+c**2*w**2, -F**2 ])
    r=np.roots(p); r=np.sort(r[(abs(r.imag)<1e-9)&(r.real>0)].real)
    A_=np.sqrt(r)
    if len(A_)==1: (hi if w<1.25*w1 else lo).append((w/w1,A_[0]))
    else:
        lo.append((w/w1,A_[0])); mid.append((w/w1,A_[1])); hi.append((w/w1,A_[2]))
lo,mid,hi=[np.array(q) for q in (lo,mid,hi)]
# single-root region for w beyond window belongs to the lowest branch, fix assignments
Aa=np.linspace(0,0.09,300); ax.plot(back(Aa),Aa,color=BLUE,lw=1,ls='--',alpha=.8); ax.text(back(0.075)+0.02,0.080,'backbone',color=BLUE,fontsize=7,ha='left')
# full curve = hi branch up to the left fold + mid + lo branch beyond right fold
allpts=np.vstack([hi,mid[::-1],lo]) if False else None
ax.plot(hi[:,0],hi[:,1],color=NAVY,lw=2); ax.plot(lo[:,0],lo[:,1],color=NAVY,lw=2); ax.plot(mid[:,0],mid[:,1],color=NAVY,lw=1.4,ls=':')
fr=mid[:,0].max(); fl=mid[:,0].min()
ax.plot([fr],[mid[np.argmax(mid[:,0]),1]],'o',color=RED,ms=5,zorder=6); ax.plot([fl],[mid[np.argmin(mid[:,0]),1]],'o',color=RED,ms=5,zorder=6)
ax.text(fr+0.015,mid[np.argmax(mid[:,0]),1]+0.004,'fold',color=RED,fontsize=7.5,va='bottom')
ax.text(fl-0.015,mid[np.argmin(mid[:,0]),1]-0.002,'fold',color=RED,fontsize=7.5,va='top',ha='right')
# sweep up in omega: follow hi to fold then jump down to lo
ax.annotate('',(fr+0.004,lo[np.argmin(abs(lo[:,0]-fr)),1]),(fr,mid[np.argmax(mid[:,0]),1]-0.001),arrowprops=dict(arrowstyle='-|>',color=ORG,lw=1.6,ls='--'))
ax.text(fr+0.03,0.043,'stepping $\\omega$ up:\nthe amplitude jumps\nand skips the fold',color=ORG,fontsize=7,ha='left',va='top')
ax.text(0.97,0.106,'pseudo-arclength: follow the curve\nthrough the folds',color=GRN,fontsize=7,ha='left',va='top',fontweight='bold')
ax.set_xlabel(r'frequency $\omega/\omega_1$'); ax.set_ylabel('amplitude'); ax.set_yticks([]); ax.set_xticks([1]); ax.set_xticklabels(['1'])
ax.set_xlim(0.95,back(0.09)+0.1); ax.set_ylim(0,0.108)
ax.set_title('(b) Forced response: a fold in amplitude',fontsize=8.5,fontweight='bold',color=NAVY,loc='left')
for s_ in ['top','right']: ax.spines[s_].set_visible(False)
import os; fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures','ch12_nnm.pdf'),bbox_inches='tight')
print(fl,fr,hi[:,1].max())
