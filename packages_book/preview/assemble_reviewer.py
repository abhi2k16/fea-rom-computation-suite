"""Assemble the reviewer preview: front matter (pages 1-21 of the full draft build with the preview notice)
+ highlights pages, with a 'Draft for review' stamp on every page after the two covers."""
import io, sys, os
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color
FN = int(os.environ.get('FRONT_N','21')); front = PdfReader(sys.argv[1]); hl = PdfReader(sys.argv[2]); out = sys.argv[3]; ab = PdfReader(sys.argv[4])
W = PdfWriter()
pages = [front.pages[i] for i in range(FN)] + [ab.pages[0]] + list(hl.pages)
def stamp(w, h):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFillColor(Color(0.36, 0.42, 0.49)); c.setFont("Helvetica", 7.5)
    c.drawRightString(w - 42, h - 28, "Draft for review  |  confidential  |  not for redistribution")
    c.save(); buf.seek(0); return PdfReader(buf).pages[0]
for i, p in enumerate(pages):
    if i >= (FN if os.environ.get("FRONT_STAMPED") else 2):
        p.merge_page(stamp(float(p.mediabox.width), float(p.mediabox.height)))
    W.add_page(p)
W.add_metadata({"/Title": "Implementing Reduced-Order Modeling In Computational Mechanics: reviewer preview",
                "/Author": "Abhijeet Kumar", "/Subject": "Preview edition for reviewers (confidential)"})
W.add_outline_item("Cover", 0); W.add_outline_item("Front matter", 2); W.add_outline_item("Contents", 16)
W.add_outline_item("Graphical abstract", FN); W.add_outline_item("Why this book", FN+1); W.add_outline_item("The whole book", FN+2); W.add_outline_item("Chapter highlights", FN+4)
W.add_outline_item("Request the full draft", len(pages) - 2)
with open(out, "wb") as f: W.write(f)
print("pages", len(pages))
