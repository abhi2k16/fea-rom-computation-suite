# Plan: make the prose read less templated (awaiting approval)

Scope: the passages listed in the review (front matter, Ch. 1-14, Appendices), plus every other place with the same pattern. Counts in the current source: "After this chapter the reader can" 14, "held-out" 64, "was not re-run" 8, "Redrawn by" 8, "Plot of the reproduction report" 90, "Fails when" 38, "What was not reproduced, and why" 9, "single runs" 12.

## Rules
1. Every number, check and caveat stays. Only the voice changes. A limitation is kept, then said once, in the place it matters, not repeated in every caption, box and summary.
2. No new claims. Anything I cannot verify in the source or the reports is not added.
3. Edit in place, chapter by chapter, with a diff summary per chapter. Build after each group; no new warnings.
4. Terminology stays fixed where it is a definition (verification, validation, reproduced / partly / not reproduced).

## Changes by pattern
| Pattern | Now | Change |
|---|---|---|
| Chapter opening box | same three-part frame in all 14 chapters, list (i)(ii)(iii) | Keep a short box, but write each opening as its own paragraph: start from the problem the chapter solves (a question, a failure, a number). Replace the (i)-(iii) list by one or two sentences of what the reader can do afterwards; list form only where tasks are really separate (Ch. 12, 13). |
| "Limits" boxes | parallel bullets with bold lead-ins | Keep the box. Shorten to the 3-5 limits that change how a result is read; merge repeats ("single runs", "small models", "one machine") into one sentence per chapter; plain sentences instead of bold-label bullets where possible. |
| Summaries | recap of every claim | 3-5 sentences: what the chapter established, one surprise, one open point. |
| "Fails when / Cost / Code / Verified in" blocks (38) | on every algorithm | Keep the structure (it is useful for reference) but cut empty or generic entries, and reword the ones that repeat the same wording. Open question to you below. |
| Evidence caveats | "held-out" (64), "single run", "not re-run", "sandbox (2 CPUs)" on nearly every number | Say it once per section; use "unseen parameters" / "test queries" / "new loads" where that is what is meant; drop the qualifier where the sentence already says it. |
| Ch. 14 captions | "Plot of the reproduction report." (90), "Redrawn by ... not re-run for this book" (8) | State once in 14.3 that the plots come from the reports and which were redrawn; captions then say what the plot shows (and the source only where it differs). |
| Ch. 14 section frame | "What was not reproduced, and why" in all 9 studies | Rename per study to its actual content (e.g. "What the publication leaves open"), write as prose, keep the content. |
| Front matter | "so that no part of the book is a stub", "template for the others" | Shorter reading guide in plain first person plural or neutral voice; keep the five-things-to-read-with-a-number list, in a sentence. |
| Stock phrases | "the design promises", "is not a property of the code", "the book quotes only ...", "not a bound / a screening tool" | Replace with concrete statements; keep each idea once. |

## Order of work
1. Front matter and Ch. 1-2 (pilot). Show before/after for review; adjust the style from your comments.
2. Ch. 3-7.
3. Ch. 8-13.
4. Ch. 14 (largest: the nine study files, captions, 14.1-14.3, 14.13-14.15).
5. Appendices A-D, final build, page count, list of edited passages.

## Decisions I need from you
- Keep the chapter-opening box in every chapter (rewritten), or drop it from some?
- Keep the "Fails when / Cost / Code / Verified in" blocks (trimmed), or reduce them to a table row or a sentence?
- Voice: neutral ("the package computes ...") or first person plural ("we compute ...")?
- Should the book stay as unusually explicit about its own limits as now (just less repeated)?

Note: this changes style, not provenance. The book's own text on authorship (one author, assistant-written exercises and references to be verified) is not touched.

## Decisions (from you)
- Opening box: keep, rewritten per chapter.
- Fails when / Cost / Code / Verified in blocks: removed; the content moves into one short paragraph after each algorithm (pseudocode stays).
- Voice: natural (plain, occasional "we"; no fixed rule).
- Limits: keep the explicitness, remove repetition.

## Status
- Pilot done: front matter, Ch. 1, Ch. 2 frame (opening box, algorithm blocks, Limits, Summary). Book 311 pages, clean build. Backups of the originals in /tmp only (git diff is the record).
- Ch. 3-7 done (same changes as the pilot: opening box, 11 algorithm blocks folded into prose, Limits merged, light edits). Book 310 pages, clean build.
- Script mentions ("Drawn by scripts/...") removed from all chapters, captions and tables; kept only in Appendix D and the README.
- Ch. 8 done as the trial for the Kleppmann items: story opening (1), naive loop first (3), runnable listing with real output (4, trial), sidenotes (5), prose summary (9), running beam example (2, trial). Book 309 pages. Items 2 and 4 await approval before Ch. 9-13.
- Ch. 9-13 done with items 1, 2, 3, 4, 5, 9 (approved after Ch. 8): story openings, naive baseline first, runnable "Try it" listings with real output (scripts kept in /tmp/hz2 only), sidenotes, algorithm blocks folded into prose, "held-out" reworded, Limits trimmed, prose summaries. Book 314 pages, clean build.
- Ch. 14 done (opening box, 14.3 voice, 90 captions reworded by removing the repeated report phrase, nine "What was not reproduced" headings renamed per study, pipeline algdet folded, Limits and Summary rewritten) and Appendices A-D (algdet folded, "held-out" reworded; Appendix D lists package tests only, no script names; README unchanged). Book 313 pages, clean build.
- Package fix: hole_in_rectangle_mesh now conforming for off-centre holes (one near-hole size for all four quadrants); regression test and CHANGELOG entry added; Ch. 2 paragraph, table (all conforming), Fig. caption and Limits updated.
- Kleppmann items 1-5 and 9 applied to Ch. 1-7 (story openings, naive first, Try it listings with real output, sidenotes, prose summaries). Done in the order 3, 5, 7, 4, 6, 1, 2. Ch. 2 summary corrected for the hole-mesh fix. Book 322 pages, clean build.
