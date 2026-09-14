"""Deep review, files lane: letters as files, PDFs, encodings and output.

Each test fails on 12705f8 and passes with its fix, except the controls,
which say so.

  FILES-1  pypdf raises KeyError, NotImplementedError, RecursionError and
           others - not only its own PyPdfError - on a malformed PDF. Those
           escaped the extract node: the case was left 'open' with no reason,
           the report said "the run stopped", and a second sweep listed it as
           a stopped run rather than a letter that could not be read.
  FILES-2  A scanned letter with an OCR text layer - an image, and the OCR's
           reading of it drawn invisibly (text render mode 3) - was read as
           a text letter. An OCR misread of "70%" as "80%" in the stated
           combined evaluation turned a POTENTIAL DISCREPANCY into NO
           DISCREPANCY FOUND at exit 0, for a page whose image states 70%.
  FILES-3  A PDF whose font maps one glyph to a lone UTF-16 surrogate (a
           malformed ToUnicode map) put that surrogate into a condition name.
           The Strands session could not write a question naming it, so the
           case was left "awaiting" an answer its session did not hold, and a
           finished case of such a letter could not be printed.
  FILES-4  A .txt transcript saved as UTF-16 with a byte-order mark - what
           Windows PowerShell 5.1's `Get-Content letter > letter.txt` writes -
           was refused as "could not be read (UnicodeDecodeError)".
  FILES-5  A finished audit whose report holds a character the output
           encoding cannot represent (report redirected to a file on Windows:
           the ANSI code page) exited 3 with only a codec error, though the
           case was complete; `show` failed the same way.
  FILES-6  A control character in a condition name (ESC) reached the terminal
           raw: "ESC[8m" concealed every following line of the report,
           verdict included, and cursor controls can overwrite printed figures.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from _support import EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED, EXIT_OK, LETTERS, case_json, main
from recheck.case import CaseStore
from recheck.graph import ScannedDocument, read_document
from recheck.sweep import outcome_from_case

TABULAR = (LETTERS / "01_tabular.txt").read_text(encoding="utf-8")


def _pdf(objects: list[bytes]) -> bytes:
    out, offsets = bytearray(b"%PDF-1.7\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _stream(data: bytes, extra: bytes = b"") -> bytes:
    return b"<< /Length %d " % len(data) + extra + b">>\nstream\n" + data + b"\nendstream"


def _one_page(content: bytes, *, font: bytes = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
              stream_extra: bytes = b"", resources: bytes = b"", extra_objects: tuple[bytes, ...] = ()) -> bytes:
    """A one-page PDF (stdlib only) drawing `content` with font /F1 (object 5)."""
    return _pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> " + resources + b" >> >>",
        _stream(content, stream_extra),
        font,
        *extra_objects,
    ])


def _literal(text: str) -> bytes:
    return b"(" + text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1") + b")"


def _show(lines: list[str]) -> bytes:
    return b" ".join(_literal(line) + b" Tj T*" for line in lines)


# --------------------------------------------------------------------------
# FILES-1: a malformed PDF pypdf fails on with a non-pypdf exception
# --------------------------------------------------------------------------

MALFORMED = {
    # A Type0 font without /DescendantFonts: KeyError.
    "type0_font": _one_page(b"BT /F1 9 Tf 72 700 Td (A) Tj ET",
                            font=b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H >>"),
    # A content stream with a filter pypdf does not know: NotImplementedError.
    "unknown_filter": _one_page(b"BT /F1 9 Tf 72 700 Td (A) Tj ET", stream_extra=b"/Filter /Bogus "),
    # Deeply nested arrays in page content: RecursionError.
    "nested_arrays": _one_page(b"BT /F1 9 Tf 72 700 Td " + b"[" * 2000 + b"(x)" + b"]" * 2000 + b" TJ ET"),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_a_pdf_pypdf_fails_on_with_a_plain_exception_is_a_letter_that_could_not_be_read(tmp_path, name):
    """Before: exit 3 with "KeyError: '/DescendantFonts'", the case 'open'
    with no reason and no digest; `show` said the run stopped, and a second
    sweep listed "the run stopped with the case in state 'open'"."""
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(MALFORMED[name])
    store = tmp_path / "runs"
    code, out, err = main("audit", path, "--case", "bad", store=store)
    assert code == EXIT_CANNOT_PROCEED and "Traceback" not in out + err
    assert f"{name}.pdf could not be read" in out + err
    raw = case_json(store, "bad")
    assert raw["status"] == "unparsed" and raw["document_sha256"] is not None
    outcome = outcome_from_case(CaseStore(store), "bad", reused=True)
    assert "could not be read" in outcome.detail and "run stopped" not in outcome.detail


# --------------------------------------------------------------------------
# FILES-2: a scan with an OCR text layer
# --------------------------------------------------------------------------

def _ocr_scan(path, lines):
    """What a scanner's "searchable PDF" is: the page image, and the OCR text over it, invisible."""
    from fpdf import FPDF
    from PIL import Image

    pdf = FPDF()
    pdf.add_page()
    image = io.BytesIO()
    Image.new("L", (850, 1100), 255).save(image, format="PNG")  # stands in for the scanned page
    pdf.image(image, x=0, y=0, w=210, h=297)
    pdf.set_font("Courier", size=9)
    pdf.text_mode = "INVISIBLE"
    for number, line in enumerate(lines):
        pdf.text(10, 12 + 5 * number, line)
    pdf.output(str(path))
    return path


def test_a_scan_with_an_ocr_text_layer_is_refused_like_an_image(tmp_path):
    """Before: the OCR text was read as the letter's text, and a misread
    stated figure (80% where the page says 70%) gave NO DISCREPANCY FOUND,
    exit 0. The same letter as text is a POTENTIAL DISCREPANCY."""
    lines = TABULAR.splitlines()
    misread = [line.replace("70%", "80%") if line.startswith("COMBINED") else line for line in lines]
    assert misread != lines
    path = _ocr_scan(tmp_path / "scan.pdf", misread)
    with pytest.raises(ScannedDocument, match="OCR"):
        read_document(path)
    code, out, err = main("audit", path, "--case", "scan", "--brief", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED and "Traceback" not in out + err
    assert "DISCREPANCY" not in out and "recomputed final degree" not in out
    assert case_json(tmp_path / "runs", "scan")["status"] == "unparsed"


@pytest.mark.parametrize("mode", [3, 7])
def test_invisible_text_anywhere_in_the_text_layer_is_refused(tmp_path, mode):
    """Visible text with one invisible rating row: the row is not on the page.
    Mode 3 draws nothing; mode 7 only adds the glyphs to the clipping path."""
    lines = TABULAR.splitlines()
    content = (b"BT /F1 9 Tf 20 800 Td 12 TL " + _show(lines[:-1]) + b" ET "
               b"BT %d Tr /F1 9 Tf 20 100 Td (  5. Limitation of flexion, left elbow ..... 40%%) Tj ET " % mode
               + b"BT 0 Tr /F1 9 Tf 20 80 Td " + _show(lines[-1:]) + b" ET")
    path = tmp_path / "hidden.pdf"
    path.write_bytes(_one_page(content))
    with pytest.raises(ScannedDocument, match="not drawn on the page"):
        read_document(path)


def test_control_a_render_mode_restored_before_any_text_is_still_read(tmp_path):
    """Control (passes before and after): mode 3 set inside q/Q, or inside a
    Form XObject, does not make the page's visible text invisible, and blanks
    drawn in mode 3 are not text."""
    form = b"q 3 Tr Q 3 Tr"
    content = (b"q 3 Tr Q /X1 Do BT 3 Tr /F1 9 Tf 20 20 Td (  ) Tj [( ) -200 ( )] TJ ET "
               b"BT 0 Tr /F1 9 Tf 20 800 Td 12 TL " + _show(TABULAR.splitlines()) + b" ET")
    path = tmp_path / "visible.pdf"
    path.write_bytes(_one_page(content, resources=b"/XObject << /X1 6 0 R >>", extra_objects=(
        _stream(form, b"/Type /XObject /Subtype /Form /BBox [0 0 1 1] "),)))
    assert "COMBINED EVALUATION FOR COMPENSATION: 70%" in read_document(path)
    code, out, _ = main("audit", path, "--case", "visible", "--brief", store=tmp_path / "runs")
    assert code == EXIT_OK and "POTENTIAL DISCREPANCY" in out


# --------------------------------------------------------------------------
# FILES-3: a lone surrogate from a malformed ToUnicode map
# --------------------------------------------------------------------------

def _pdf_with_a_lone_surrogate(letter: str, after: str) -> bytes:
    """The letter as a text PDF whose glyph 0x01, drawn just after `after`, maps to U+D800."""
    cmap = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /X def\n"
            b"1 begincodespacerange <00> <FF> endcodespacerange\n"
            b"1 beginbfchar <01> <D800> endbfchar\n"
            b"endcmap CMapName currentdict /CMap defineresource pop end end")
    shown = []
    for line in letter.splitlines():
        if after in line:
            head, tail = line.split(after, 1)
            shown.append(_literal(head + after) + b" Tj (\x01) Tj " + _literal(tail) + b" Tj T*")
        else:
            shown.append(_literal(line) + b" Tj T*")
    content = b"BT /F1 9 Tf 20 800 Td 12 TL " + b" ".join(shown) + b" ET"
    return _one_page(content, font=b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /ToUnicode 6 0 R >>",
                     extra_objects=(_stream(cmap),))


def test_a_lone_surrogate_in_pdf_text_does_not_reach_the_case(tmp_path):
    path = tmp_path / "letter.pdf"
    path.write_bytes(_pdf_with_a_lone_surrogate(TABULAR, "Tinn"))
    text = read_document(path)
    assert "Tinn�itus" in text
    text.encode("utf-8")  # raised UnicodeEncodeError: surrogates not allowed


def test_a_question_naming_a_condition_with_a_lone_surrogate_can_be_answered(tmp_path):
    """Before: exit 3 "UnicodeEncodeError: 'utf-8' codec can't encode
    character '\\ud800'"; the case said awaiting_human, its session held no
    question, and resume refused it forever."""
    letter = (LETTERS / "05_missing_side.txt").read_text(encoding="utf-8")
    path = tmp_path / "question.pdf"
    path.write_bytes(_pdf_with_a_lone_surrogate(letter, "degenerative"))
    store = tmp_path / "runs"
    code, out, err = main("audit", path, "--case", "q", store=store)
    assert code == EXIT_AWAITING_HUMAN, out + err
    code, out, err = main("resume", "--case", "q", "--answer", "1=left,2=right", store=store)
    assert code == EXIT_OK, out + err
    assert case_json(store, "q")["status"] == "complete"
    code, out, err = main("show", "--case", "q", "--brief", store=store)
    assert code == EXIT_OK and "�" in out
