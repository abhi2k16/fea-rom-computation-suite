# Plan: Kleppmann items 1-5 and 9 in Chapters 1-7 (awaiting approval)

Method: as in Chapters 8-13. For each chapter: run a short script, record its real output, edit the chapter, build, check. Scripts stay out of the repo. Numbers in listings come from the run; timings are single runs and are said to be.

Items applied in every chapter: (1) story/concrete-problem opening in "The problem"-type first section, box kept; (3) naive version first, with a measured cost or failure; (4) one runnable "Try it" listing with printed output (extend the existing listing with its output where one exists); (5) 2-3 sidenotes for definitions/detours; (9) prose summary; (2) one running example per chapter, named in the first section.

| Ch | Story opening | Naive first | Try it (output to print) | Sidenotes | Running example |
|---|---|---|---|---|---|
| 1 Architecture | adding one element should not mean editing the solver | what touching solver, assembler and I/O would cost (files and lines changed for a new element, counted) | register a new element with the one registry line, run the conformance test, print what passed | registry; extension point; backend | bar element |
| 2 Meshes | a mesh that passes the area check but is wrong | hand-built node list with a reversed element: area fine, Jacobian negative | `check_quality` on the bad and the good mesh; the stray-node mask on a hole mesh | node ordering; Jacobian; conformity | plate with a hole |
| 3 Elements | a thin beam that comes out 100x too stiff | fully integrated Q4 on a thin cantilever | tip deflection ratio against the closed form: full vs selective-reduced integration | weak form; locking; patch test | thin cantilever |
| 4 Dynamics | an explicit run that blows up after a step change | explicit with a step above the critical one | critical step, amplification, Newmark vs explicit on the same beam | stability vs accuracy; critical step; modal truncation | cantilever |
| 5 Nonlinearity | load control that cannot pass the limit load | load control on the von Mises truss | load vs displacement control, printed failure and success (extend the existing listing) | tangent; limit point; state variable | von Mises truss |
| 6 Acceleration | a Python loop that dominates the run | element-by-element assembly | loop vs batched assembly: time and difference of the matrices | batching; chunking; backend | plate of the chapter |
| 7 Contact | a penalty spring that lets a membrane sink in | penalty with three stiffnesses | penalty vs active set: penetration and force (extend existing listing) | KKT; active set; gap | membrane on an obstacle |

Order and checkpoints: 3, 5, 7 first (the strongest naive-vs-better contrasts), then 1, 2, 4, 6. One build per chapter; a review stop after Chapter 3 (style check) and after Chapter 7 (page count, build, summary list).

Rules kept: no design-lesson or removal arguments; Gmsh removal not discussed; "report" numbers stay "report" numbers; the authorship text is untouched; the code-map "Code" column stays; script names are not mentioned in chapters.

Risks: a naive version may not show the expected failure (then I report what I measured and change the story, not the number); Chapter 2 has 20 pages already, so I add only the Try it and one story paragraph there; Chapter 6 timings vary by tens of percent, stated in the text.

Estimated effect: +10 to +14 pages in total.
