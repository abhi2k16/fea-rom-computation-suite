"""Report (never delete) the generated number macros that no .tex file of the book uses.
Run from the book root:  python scripts/report_unused_macros.py [prefix ...]      (default prefixes: ch10 ch11)
For each prefix it reads tables/numbers_<prefix>*.tex, looks for every macro name in all other .tex files of the book
(chapters, appendices, tables, figures/tikz, macros.tex), prints a summary and writes UNUSED_MACROS.md
(name = value, grouped by file). Nothing in tables/ is changed, so the build is unaffected.
A macro counts as used when its name appears as \\name not followed by a letter, in any .tex file that is not a
numbers_<prefix>*.tex file."""
import glob, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prefixes = sys.argv[1:] or ["ch10", "ch11"]
alltex = [f for f in glob.glob(os.path.join(ROOT, "**", "*.tex"), recursive=True)]
lines = ["# Unused generated macros (report only)", "",
         "Written by `scripts/report_unused_macros.py`. The macros stay in `tables/numbers_*.tex`; this file only lists them.", ""]
for pref in prefixes:
    files = sorted(glob.glob(os.path.join(ROOT, "tables", f"numbers_{pref}*.tex")))
    own = {os.path.abspath(f) for f in files}
    text = "\n".join(open(f, errors="ignore").read() for f in alltex if os.path.abspath(f) not in own)
    total = unused = 0
    lines += [f"## numbers_{pref}*.tex", ""]
    for f in files:
        defs = re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\\xspace\}", open(f).read())
        un = [(n, v) for n, v in defs if not re.search(r"\\" + n + r"(?![A-Za-z])", text)]
        total += len(defs); unused += len(un)
        lines += [f"### {os.path.basename(f)}: {len(un)} unused of {len(defs)}", ""]
        lines += [f"- `{n}` = {v}" for n, v in un] + [""]
    print(f"{pref}: {unused} unused of {total} macros")
open(os.path.join(ROOT, "UNUSED_MACROS.md"), "w").write("\n".join(lines))
print("wrote UNUSED_MACROS.md")
