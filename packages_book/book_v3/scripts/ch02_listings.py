"""Run the two code listings of Chapter 2 and save their output for the book.

The code is read from chapters/ch02.tex (the lstlisting blocks labelled lst:me-nodal and lst:me-try),
so the printed output cannot drift from the listed code.
"""
import io, os, re, sys, contextlib, warnings
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BOOK = os.path.dirname(HERE)
for p in (os.path.join(BOOK, "..", "..", "fea_engine", "src"),):
    if os.path.isdir(p): sys.path.insert(0, os.path.abspath(p))
src = open(os.path.join(BOOK, "chapters", "ch02.tex"), encoding="utf-8").read()

def block(label):
    m = re.search(r"\\begin\{lstlisting\}\[[^\]]*label=\{" + label + r"\}\]\n(.*?)\\end\{lstlisting\}", src, re.S)
    return m.group(1)

for label, out in (("lst:me-nodal", "ch02_nodal_out.txt"), ("lst:me-try", "ch02_try_out.txt")):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(block(label), label, "exec"), {"__name__": "__listing__"})
    open(os.path.join(BOOK, "tables", out), "w", encoding="utf-8").write(buf.getvalue())
    print("wrote", out); print(buf.getvalue())
