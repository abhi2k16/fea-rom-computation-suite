# Book v2: reader-oriented theory context, Chapter 3 plan (for approval)

Goal: a reader who knows FEA and ROM in outline should be able to follow every term of Chapter 3 without a textbook open. Add short plain-language context at the point where a term is first needed, not a theory chapter. Source of context: standard texts (Zienkiewicz-Taylor, Bathe, Hughes, Cook et al., Belytschko et al.), checked by search before any citation is added. v1 (`book/`, `book_draft.pdf`) stays untouched; v2 is `book_v2/` -> `book_draft_v2.pdf`.

## Additions (about +6 pages)
1. 3.1 Weak form: sidenote "Strong form, weak form, virtual work" (one physical picture: equilibrium tested against every admissible virtual displacement; why the weak form needs only first derivatives). New short subsection "An element in five steps": interpolation (shape functions), isoparametric map and Jacobian, B and D, Gauss quadrature, assembly. Worked 2-node bar: K and consistent M by hand, then checked against the package (a run listing).
2. 3.1 sidenote: plane stress vs plane strain, what the D matrix holds, when each is the right model.
3. 3.1 Mass: consistent vs lumped in words (distributed vs point masses), with the 2-node bar matrices; what HRZ lumping does.
4. 3.2 Families: table "Which element for which structure" (continuum, beam, plate, shell: assumptions, dofs, thickness range). Sidenotes: Euler-Bernoulli vs Timoshenko/Reissner; Kirchhoff vs Mindlin; what MITC means in one paragraph.
5. 3.3 Locking: sidenote "What locking is" (spurious strain energy the interpolation cannot avoid; parasitic shear in a 4-node bending beam). Table "Full, reduced, selective reduced integration": points used, rank of K_e, failure mode (locking vs hourglass). Sidenote "Incompressibility and Poisson ratio" (bulk modulus, lambda -> infinity as nu -> 0.5). Two-sentence mixed/B-bar idea before the Hughes element.
6. 3.4 Verification: sidenotes "Convergence order and mesh refinement" (h-refinement, why order p+1 in displacement and 2p in eigenvalues), "Natural frequency and eigenvalue" (K phi = lambda M phi).
7. Glossary (Appendix C) gets the new terms; every new term is defined once and used consistently afterwards.

## Not done
No new derivations of shape functions or time integrators (the Preface says the book assumes them); no change to existing numbers, tables or figures; Chapters 1-2 and 4-14 untouched until Chapter 3 is approved as a pattern.

## Steps
copy `book/` to `book_v2/`; write the additions in `book_v2/chapters/ch03.tex`; run the new bar listing; search to verify definitions and citations; build `book_draft_v2.pdf`; check pages visually.
