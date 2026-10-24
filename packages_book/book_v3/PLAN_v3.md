# Plan for version 3 of the book

Working title unchanged: *Implementing Reduced-Order Modeling In Computational Mechanics*.
Baseline: `book_v2` (341 pages, `book_draft_v2_full_i.pdf`). `book_v2` and `book` (v1) stay as they are; v3 is written in this folder only.
Status: decisions 1 to 5 confirmed. Phase 0 and Phase 1 done (see section 10). Next: Phase 2.

## 1. Why a third version

Two independent readings of v2 (my section-level analysis and the editorial review) agree on the main point: the book should let a reader understand the mathematics and algorithms first, and then see how `fea_engine` and `rom_engine` implement them. v2 has this in Part III but not consistently in Chapters 1, 2, 6 and 14, and it does not separate what belongs to the *method* from what belongs to *this implementation*.

Measured in v2: Chapter 2 has about 7,000 words and about 170 package names (a user manual in places); Chapter 14 has about 24,000 words, 107 figures and 77 tables; Chapter 13 holds two topics (hyper-reduction and contact ROMs); Chapter 6 interrupts the path from nonlinearity to contact; Chapter 1 has 14 forward references.

## 2. Guiding rules (apply to every chapter)

R1. **Teaching sequence.** Engineering question, mathematical model, numerical method (algorithm), small worked example, implementation note, verification, interpretation and limits. A teaching order, not seven mandatory headings.
R2. **Package names stay out of the first two sections of a chapter.** Words and symbols there. Names, signatures and code paths go to the implementation note and the code map.
R3. **Implementation note box.** A new deep-blue box (`implnote`, same blue as code anchors) holds package-specific mechanics: classes, registries, file paths, API behaviour. Main text can be read without it.
R4. **Limits are tagged.** Every item in a limits box starts with a tag: *[Method]* (true of the method anywhere) or *[This implementation]* (true of the package as released). Examples: SOAR rescaling and the `auto` backend probe are *[This implementation]*; the loose a priori bound of balanced truncation is *[Method]*.
R5. **Replacement test.** For each section: if another implementation replaced the package, would the reader still understand the mathematics, algorithm, assumptions and interpretation? If not, either strengthen the explanation or label the section as software design.
R6. **Tables have a purpose line and an interpretation line.** Before: the question the table answers. After: what it shows and why it matters.
R7. **Opening box.** Replace the 200 to 250 word "In this chapter" box with three short parts: *Question*, *Result*, *Needs from earlier chapters*. "The problem" section keeps the story.
R8. **Notation** is introduced where first used. A short notation table before each major derivation, with sizes and units.
R9. **One reference model** is named in Chapter 3 and used wherever the physics allows (see 5). New models appear only when a phenomenon needs them.
R10. **Diagrams answer questions** (assembly, Newton loop, offline/online ROM workflow, interface reduction, hyper-reduction). At least one per chapter in Parts I to III where none exists.

## 3. Proposed chapter structure

| v3 | Title (working) | From v2 | Main change |
|---|---|---|---|
| 1 | Design of a decoupled engine | 1 | Open with the problem and one small worked example. Keep layers, element interface, new element, dependence rule. Move assembly mathematics to Chapter 3. Move the conformance test to an implementation note. Move the `auto` backend cost to Chapter 7. |
| 2 | Meshes | 2 | Concepts only: what a mesh is, topology and ordering, quality, grading, conformity, node sets. Target about 3,500 words. Generators table, exchange formats, visualisation and the dispatcher move to a new Appendix E, *Mesh tools*. |
| 3 | Linear elements and assembly | 3 | Add the path weak form, element matrix, local-to-global mapping, global assembly, boundary conditions (formula plus diagram). Define the reference model. |
| 4 | Dynamics | 4 | Governing equation and the five response types first, then solvers. Choosing-a-method table moves to the start. |
| 5 | Nonlinearity | 5 | Add Newton-loop diagram; keep residual/tangent/Newton first (already so). |
| 6 | Contact | 7 | Moves one place earlier. Add a short bridge from nonlinear equilibrium (gap, complementarity). |
| 7 | Acceleration | 6 | Moves after contact. Add the `auto` backend cost from Chapter 1. PyTorch condensed to about 400 words, with the full records in Appendix A. |
| 8 | Projection (POD, Galerkin) | 8 | Principal worked example of the full-order to reduced-order step. Add offline/online diagram. |
| 9 | Systems and control | 9 | Open with the change of viewpoint (state approximation to input-output map) and when each method fits. |
| 10 | Frequency-domain ROMs | 10 | Separate heuristic indicators from certified bounds into two sections, with the conditions of validity of the bound. |
| 11 | Substructuring | 11 | Two-component example before the general formulation. Interface diagram. |
| 12 | Non-intrusive and nonlinear ROMs | 12 | Add a section and table: which problem each approach (Loewner, polynomial/RBF modal models, harmonic balance) solves. |
| 13 | Hyper-reduction (ECSW) | 13, first part | Split from old 13. Revisit which parts stay exact. |
| 14 | Reduced contact models | 13, second part | Split from old 13. Start from the full-order contact problem; state exactly which constraints stay exact and which approximations enter. |
| 15 | Verification and the pipeline | 14, first part | What "verified" means, the one-table summary, the integrated pipeline. About 6,000 words. |
| Part V | Reproduction studies | 14, nine studies | Each study opens with a summary card (see 4). Text first kept as in v2, then trimmed in a later pass. |
| App. A | Differentiable assembly and the PyTorch backend | A | Receives the full PyTorch records of old Chapter 6. |
| App. B to D | Conventions, glossary, test map | B, C, D | Renumbered references; glossary updated. |
| App. E | Mesh tools | new, from 2 | Generators, formats, visualisation, dispatcher, Abaqus import example. |

Part structure: I Foundations (1-2), II Full-order physics (3-7), III Reduced-order modelling (8-14), IV Verification (15), V Reproduction studies.

## 4. Reproduction-study summary card (for each of the nine studies)

Four lines at the top: *Original question*, *Method reproduced*, *Evidence* (what was compared), *Outcome* with two flags: **result status** (reproduced / partly / not established) and **number status** (regenerated at the tag / quoted from the author's report). The "quoted, not regenerated" rule also goes into the Chapter 15 opening box.

## 5. Reference model (R9)

v2 uses a different model almost every chapter. v3.0 does not change numbers. It adds a table, "Models used in this book", in Getting started and in Chapter 3: which model, which chapters, why a different one is needed (truss for snap-through, membrane for contact, chain for hyper-reduction). Aligning mesh sizes would change many generated numbers and is left as an optional later step (v3.1).

## 6. Front matter and navigation

- Preface "Outline of the book", Getting started overview table and dependency figure, Reading guide: updated to the new numbering and order.
- Short roadmap figure at the start of each Part.
- Colours: keep `colors.tex`; add the implementation-note colour and the two limit tags.

## 7. Work phases and checks

| Phase | Content | Check |
|---|---|---|
| 0 | Copy `book_v2` sources into `book_v3`; add `implnote`, limit-tag macros, new opening box | Builds, 341 pages, 0 undefined |
| 1 | Chapter 15 opening box and nine summary cards; status rules | Every study has both flags |
| 2 | Chapters 1, 2, 3: restructure, new Appendix E | Replacement test passed per section; word counts as targets |
| 3 | Reorder 6 and 7; move `auto` cost; condense PyTorch | All cross references resolve; no number macro lost |
| 4 | Split Chapter 13; Chapter 15 and Part V | Same results as v2; numbers unchanged |
| 5 | Part III chapters 9 to 12: openers, tags, purpose lines | Limits all tagged |
| 6 | Front matter, figures, Getting started update | Reading paths correct |
| 7 | Rebuild book and reviewer preview; check every chapter page type | 0 undefined; a reviewer read of two chapters |

Each phase is built and checked before the next; v2 is the reference for every number.

## 8. Decisions needed

1. Reorder: contact before acceleration (Chapter 6 / 7 swap). Recommended: yes.
2. Split old Chapter 13 into two chapters. Recommended: yes.
3. Move the nine studies into a Part V of their own. Recommended: yes, keeping the text first and trimming later.
4. Reference model: table only in v3.0, regenerate numbers later. Recommended: table only.
5. Start with Phase 0 and 1 after your confirmation.

## 9. Risks

- Renumbering chapters changes every "Chapter n" in the reviewer preview and its generated pages; they are regenerated from the chapter order.
- Moving measurements between chapters (`auto` backend) requires moving the figure, table and number macros together.
- Splitting Chapter 13 and trimming Chapter 14 can drop a claim that other chapters cite; a reference search is part of each phase.

## 10. Progress log

- **Phase 0 (done).** `book_v2` sources copied into `book_v3`. New file `v3.tex` (loaded after `colors.tex`) with: `implnote` box, `\lmethod` and `\limpl` tags for limits items, `\chopen{question}{result}{needs}` opening box, `\studycard{...}` for reproduction studies. Builds with 0 undefined references (344 pages, v2 had 341).
- **Phase 1 (done).** Chapter 14 opening box now says the numbers of the nine studies are quoted from the author's reports and not regenerated at the tag. Each of the nine studies opens with a summary card: original question, method reproduced, evidence, result status, number status. Result status: reproduced (Balajewicz 2016; He 2023 for the beam and shell wing, partly for the solid wing), partly (Yang 2019, Hollkamp 2008, Liang 2024, Yang 2025, Fauque 2018, Nikiema 2026), and Liu 2026 reproduced against the author's own reference model. Number status is the same for all nine.
- The numbers are taken from Table 14.2 of v2 and from the text of the chapter; no number was changed.
