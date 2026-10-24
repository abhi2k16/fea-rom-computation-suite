import json
R=[json.loads(l) for l in open('/sessions/affectionate-epic-mccarthy/mnt/computation-suite/packages_book/book_v3/experiments/multicore.jsonl')]
def g(case,nj,be,bl): return [r for r in R if r['case']==case and r['n_jobs']==nj and r['backend']==be and r['blas']==bl][0]['time']
rows=[];base={}
def row(label,case,nj,be,bl,ref):
    t=g(case,nj,be,bl); rows.append("%s & %d & %s & %.2f & %.2f & %.0f\\,\\%% \\\\"%(label,nj,bl,t,ref/t,100*ref/t/nj))
t1=g('plate',1,'process','1')
rows.append("\\multicolumn{6}{@{}l}{\\emph{(a) twelve plate solves, $n=5\\,000$ dofs each; time in s}} \\\\")
for bl in('1','2'):
    for nj,be,lab in((1,'process','serial'),(2,'process','2 processes'),(4,'process','4 processes'),(2,'thread','2 threads')):
        t=g('plate',nj,be,bl); rows.append("%s & %d & %s & %.2f & %.2f & %.0f\\,\\%% \\\\"%(lab,nj,bl,t,t1/t,100*t1/t/nj))
rows.append("\\midrule\\multicolumn{6}{@{}l}{\\emph{(b) twelve dense solves, $1200\\times1200$ each}} \\\\")
d1=g('dense',1,'process','1')
for bl,nj,lab in(('1',1,'serial'),('2',1,'serial, 2 BLAS threads'),('1',2,'2 processes'),('2',2,'2 processes, 2 BLAS threads')):
    t=g('dense',nj,'process',bl); rows.append("%s & %d & %s & %.2f & %.2f & %.0f\\,\\%% \\\\"%(lab,nj,bl,t,d1/t,100*d1/t/nj))
rows.append("\\midrule\\multicolumn{6}{@{}l}{\\emph{(c) 400 jobs of about 1 microsecond each; time in ms}} \\\\")
tt=g('tiny',1,'process','1')
for nj,be,lab in((1,'process','serial'),(2,'process','2 processes'),(2,'thread','2 threads')):
    t=g('tiny',nj,be,'1'); rows.append("%s & %d & 1 & %.1f & %.3f & %.1f\\,\\%% \\\\"%(lab,nj,t*1e3,tt/t,100*tt/t/nj))
tab="\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nconfiguration & jobs & BLAS threads & time & speed-up & efficiency \\\\\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n"
open('tables/ch06_multicore.tex','w').write(tab)
m={'mcPlateSpeed':"%.1f"%(t1/g('plate',2,'process','1')),'mcPlateFour':"%.1f"%(t1/g('plate',4,'process','1')),
 'mcDenseBlas':"%.2f"%(d1/g('dense',1,'process','2')),'mcDenseProc':"%.2f"%(d1/g('dense',2,'process','1')),
 'mcDenseOver':"%.2f"%(d1/g('dense',2,'process','2')),'mcTinySlow':"%.0f"%(g('tiny',2,'process','1')/tt),
 'mcTinyThread':"%.0f"%(g('tiny',2,'thread','1')/tt),'mcPlateOne':"%.0f"%(1000*t1/12)}
open('tables/numbers_ch06_multicore.tex','w').write("".join("\\newcommand{\\%s}{%s}\n"%(k,v) for k,v in m.items()))
print(m); print(tab)
