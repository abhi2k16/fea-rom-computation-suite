# packages_book

Planning material for the book *Architecting Computational Mechanics: From-Scratch Finite Element Analysis and Reduced-Order Modeling in Python*, based on the two packages of this repository (`fea_engine`, `rom_engine`).

- `plan/` the book plan, written in LaTeX in the same style as `paper/`.
  Build: `cd plan && latexmk -pdf main` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
  `plan/refs.bib` was copied from `paper/refs.bib` and extended with contact and active-set references.
- `book/` is not created yet; it will hold the chapters once the plan is accepted.
