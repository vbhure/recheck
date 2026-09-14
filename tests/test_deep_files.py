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
