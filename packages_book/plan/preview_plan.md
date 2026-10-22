# Preview editions of the book (plan, for approval)

Source: `book_v2/` (the version with reader context). Two editions from one set of sources, built in `preview/`: `book_preview_reviewer.pdf` (about 45 pages) and `book_preview_public.pdf` (about 16 pages) plus 15 LinkedIn slides (1080 x 1350 px PNG and one PDF). Nothing new is invented: every figure, number and quotation comes from the book or its macros.

## Reviewer edition (about 45 pages)
1. Front matter as in the book, up to the contents: front and back cover, title page, copyright (with a preview notice), license for code, About the author, Preface, Reading guide, list of abbreviations, full table of contents (page numbers refer to the full draft), list of figures and tables (about 12 pages).
2. A whole-book spread (2 pages): the architecture figure (fea_engine to rom_engine by plain arrays), the six levels of evidence, the nine reproduction studies named in one table, four reading paths.
3. One two-page spread per chapter (28 pages), same layout each time:
   - the question it answers (the opening problem, 2 to 3 sentences);
   - key concepts, 5 to 6 terms with a one-line meaning (from the new context boxes and glossary);
   - one signature figure with its caption;
   - one headline number with its reference and the stated limit;
   - a short code excerpt (at most 12 lines) or the names of the main classes;
   - what the reader can do afterwards (from the chapter's opening box).
4. Request page: what the reviewer receives (full draft, repository tag, the scripts behind the numbers), the kind of feedback sought (accuracy of definitions, depth for the target reader, structure, missing topics), a timeline, and contact.
5. Review terms (one page, drafted for a lawyer to check; not legal advice): confidentiality, no copying or sharing of the full draft, use of feedback, attribution, no warranty, and how to withdraw.

## Per-chapter content (headline numbers are taken from the book's macros at build time)
| Ch | Hook | Signature figure | Headline (to be confirmed from the book) |
|---|---|---|---|
| 1 | Add an element without touching a solver | ch01 flow / dispatch | no concrete element classes named in the three solver modules |
| 2 | An area check passes on a mesh that is wrong | ch02 quality / anatomy | the conformity defect and its fix |
| 3 | A plate element 200 times too stiff | ch03 bending / beams | tip deflection ratio 0.005 against 1.059 |
| 4 | An explicit step that blows up | ch04 modal / explicit | critical step and amplification above it |
| 5 | Load control fails at a limit point | ch05 vonmises | displacement control converges to about 1e-13 |
| 6 | The loop is the bottleneck | ch06 scaling | batched against loop speed-up |
| 7 | Why a penalty is the wrong tool | ch07 penalty vs exact | exact active-set result against penalty |
| 8 | A parametric beam, thousands of queries | ch08 pod / sweep | error on unseen parameters, online time |
| 9 | Reduce through the input-output map | ch09 compare | accuracy at equal state count |
| 10 | Choose frequencies greedily, bound the error | ch10 greedy / bound | greedy against uniform training |
| 11 | Reduce the parts, join them | ch11 cb / component | Craig-Bampton against Guyan at equal size |
| 12 | Models from data | ch12 nlrom / nnm | nonlinear against linear modal error |
| 13 | Make the nonlinear force cheap | ch13 ecsw | selected elements against full mesh |
| 14 | What "verified" means, nine reproductions | ch14 pipeline | pipeline error on unseen parameters and loads |

## Public edition (about 16 pages) and LinkedIn slides
- Public PDF: cover, a one-page "what the book is and who it is for", the contents, one page per chapter (hook, one figure, four key concepts, one sentence of result), a page on how the numbers were checked, About the author, and a call to follow or request the full draft. No terms page, no review questions, no code excerpts.
- 15 slides (cover plus one per chapter): a large figure, a one-line hook, three keywords. Plus three post texts (launch, a figure-led post, a "what I got wrong and checked" post), no hashtags beyond five, written for you to edit.
- The public edition names no unreleased results beyond those printed in the preview.

## Build
New `preview/` folder with `main_reviewer.tex`, `main_public.tex`, `highlights.tex` (one macro per chapter so both editions share the text), figures read from `book_v2/figures`. A small script renders the slides from the same figures. Contents and lists are regenerated from the full book's .toc so page numbers match the full draft. Checks: every number in the highlights matches the book; no page mentions the repository tag as released (it is local); links work.

## Decisions I will make unless told otherwise
Colours and fonts follow the cover (deep blue and white); preview notice on every page footer of the reviewer edition; "Draft for review" mark; the full draft's copyright page is kept, with the preview wording added.
