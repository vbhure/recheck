from fpdf import FPDF
from pypdf import PdfReader
import re
p = FPDF(); p.add_page(); p.set_font("Courier", size=10)
lines = ["DEPARTMENT OF VETERANS AFFAIRS","Regional Office","",
 "Date of Notification: March 14, 2026","File Number: 12-345-678","",
 "RATING DECISION",
 "  1. Post-traumatic stress disorder (DC 9411) ....... 50%",
 "  2. Limitation of flexion, right knee (DC 5260) .... 30%",
 "  3. Limitation of extension, left knee (DC 5261) ... 20%",
 "  4. Tinnitus (DC 6260) ............................. 10%","",
 "COMBINED EVALUATION FOR COMPENSATION: 70%"]
for L in lines: p.cell(0,5,L,new_x="LMARGIN",new_y="NEXT")
p.output("letter.pdf")
txt = "\n".join(pg.extract_text() for pg in PdfReader("letter.pdf").pages)
print("--- extracted text ---"); print(txt)
print("--- regex ---")
print("lines:", re.findall(r"\(DC\s*(\d{4})\)[^\d%]*?(\d{1,3})%", txt))
print("combined:", re.search(r"COMBINED EVALUATION[^:]*:\s*(\d{1,3})%", txt).group(1))
print("date:", re.search(r"Date of Notification:\s*([A-Z][a-z]+ \d{1,2}, \d{4})", txt).group(1))
