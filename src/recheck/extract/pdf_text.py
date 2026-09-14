"""PDF text extraction in a child process that can be killed.

Byte, page, stream and character caps bound what a PDF may hold, not how
long pypdf may work on it. pypdf re-parses a Form XObject every time a page
invokes it (up to 5,000 times) and loads font resources before any hook can
run, and it swallows exceptions at the form level. A 2.7 KB PDF invoking a
1.9 MB form 40 times held a sweep for 95 s; bigger forms scale to hours. No
cap counts that work and nothing inside the reader can interrupt it, so the
work runs in a separate process with a wall-clock budget, and the process is
killed when the budget runs out.

The child is a fresh interpreter (sys.executable -I -c) that imports only
this module, pypdf and the standard library, given the parent's sys.path.
multiprocessing's "spawn" was the obvious alternative, but it re-imports the
parent's main script in the child: a script that reads a letter without an
"if __name__ == '__main__'" guard failed with a RuntimeError, and a guarded
one still re-imported the Strands SDK (a second of the budget) before pypdf
read a byte. On Windows, killing a virtual environment's python.exe also
ends the interpreter it launched (checked), so the kill is complete. The
parent waits in short steps, so Ctrl+C is not held until the budget runs
out, and the child ends itself at its budget, so a parent killed outright
does not leave pypdf working.

The limits arrive as arguments rather than being read from recheck.graph's
constants, so a test that lowers a constant in the parent is honoured in the
child too.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_pages: int
    max_chars: int
    max_stream_bytes: int
    max_content_bytes: int


@dataclass(frozen=True)
class Outcome:
    """What the child reported.

    kind is "text" (value: the text), "too_large" (value: the reason),
    "invisible" (value: the reason; text drawn invisibly, as an OCR layer is),
    "error" (value: the exception pypdf raised; the parent re-raises a pypdf error or OSError as
    itself and anything else as a PdfReadError naming it),
    "timeout" (the budget ran out and the child was killed) or "died" (the
    child ended without a readable report; value: why).
    """

    kind: str
    value: object = None


# The request is unpickled before sys.path is extended, so it holds builtins only.
_BOOTSTRAP = (
    "import pickle, sys\n"
    "request = pickle.load(sys.stdin.buffer)\n"
    "sys.path[:0] = request['path']\n"
    "from recheck.extract.pdf_text import _serve\n"
    "_serve(request)\n"
)


def extract(data: bytes, name: str, limits: Limits, seconds: float) -> Outcome:
    """Extract the text of a PDF in a child process, within seconds of wall-clock time."""
    request = pickle.dumps({"path": list(sys.path), "data": data, "name": name, "seconds": seconds,
                            "limits": dataclasses.asdict(limits)})
    # Popen and short waits rather than subprocess.run: a Ctrl+C in the parent
    # was held until the child's whole budget ran out, and the finally kills
    # the child however the parent leaves. The child also ends itself at its
    # budget (see _serve), so a parent killed outright does not leave pypdf
    # working for hours.
    proc = subprocess.Popen([sys.executable, "-I", "-c", _BOOTSTRAP], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    deadline = time.monotonic() + seconds
    stdout = stderr = b""
    try:
        pending: bytes | None = request
        while True:
            try:
                stdout, stderr = proc.communicate(input=pending, timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pending = None
                if time.monotonic() >= deadline:
                    return Outcome("timeout")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    try:
        # Our own child's report: plain values and pypdf's exception objects.
        kind, value, records = pickle.loads(stdout)
    except Exception:  # noqa: BLE001 - killed, crashed, or wrote something else
        last = stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        return Outcome("died", last[0])
    _replay(records)
    return Outcome(kind, value)


class _Collect(logging.Handler):
    """Keeps the child's log records (pypdf's warnings about a malformed file), up to LIMIT.

    All of them were replayed: a 271 KB PDF with a deeply nested junk
    dictionary printed 44,264 warning lines (5.9 MB) before its report. The
    rest are counted, and one record says how many were not shown.
    """

    LIMIT = 20

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) >= self.LIMIT:
            self.dropped += 1
            return
        # Flattened as logging.handlers.QueueHandler does, so it pickles.
        record.msg, record.args, record.exc_info, record.exc_text = record.getMessage(), None, None, None
        self.records.append(record)


def _replay(records: list[logging.LogRecord]) -> None:
    """Log the child's records in this process, under its logging configuration.

    Read in-process, pypdf's warnings went through the CLI's handlers and
    levels. The child has none of that configuration, and would print them
    in its own format straight to the terminal.
    """
    for record in records:
        logger = logging.getLogger(record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


def _serve(request: dict) -> None:
    """The child: read the PDF and write one pickled report to stdout."""
    # An orphan - the parent killed outright - ends at its budget.
    import os
    import threading

    stop = threading.Timer(float(request.get("seconds", 60.0)) + 1.0, os._exit, (70,))
    stop.daemon = True
    stop.start()
    report = sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing else may write into the report
    collect = _Collect()
    logging.getLogger().addHandler(collect)
    try:
        kind, value = read(request["data"], request["name"], Limits(**request["limits"]))
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which re-raises it
        kind, value = "error", exc
    if collect.dropped:
        collect.records.append(logging.makeLogRecord({
            "name": "pypdf", "levelno": logging.WARNING, "levelname": "WARNING",
            "msg": f"{collect.dropped:,} more pypdf warnings about {request['name']} not shown",
        }))
    try:
        payload = pickle.dumps((kind, value, collect.records))
    except Exception as exc:  # noqa: BLE001 - an exception or record that does not pickle
        payload = pickle.dumps(("error", RuntimeError(f"{type(value).__name__}: {value} ({exc})"), []))
    report.write(payload)
    report.flush()


class _InvisibleText:
    """Counts text drawn invisibly - render mode 3, or 7 (clipping only) - via pypdf's operator visitors.

    That is how OCR software lays its text over a scanned image: the page
    shows the image, and the extracted text is the OCR's reading of it. Read
    as a text layer, a scan whose OCR misread "70%" as "80%" was reported as
    NO DISCREPANCY FOUND at exit 0 for a letter whose image states 70%.
    The render mode belongs to the graphics state: q/Q save and restore it,
    and so does the implicit q/Q around a Form XObject (Do). A string of
    blanks draws nothing either way, so it does not count.
    """

    SHOW = (b"Tj", b"TJ", b"'", b'"')
    INVISIBLE = (3, 7)

    def __init__(self) -> None:
        self.mode, self.shown = 0, 0
        self.saved: list[int] = []
        self.forms: list[tuple[int, int]] = []

    def before(self, operator: bytes, operands: list, *_: object) -> None:
        if operator == b"q":
            self.saved.append(self.mode)
        elif operator == b"Q":
            self.mode = self.saved.pop() if self.saved else 0
        elif operator == b"Do":
            self.forms.append((len(self.saved), self.mode))
        elif operator == b"Tr" and operands:
            try:
                self.mode = int(operands[0])
            except (TypeError, ValueError):
                pass
        elif operator in self.SHOW and self.mode in self.INVISIBLE and _glyphs(operands):
            self.shown += 1

    def after(self, operator: bytes, *_: object) -> None:
        if operator == b"Do" and self.forms:
            depth, self.mode = self.forms.pop()
            del self.saved[depth:]


def _glyphs(operands: list) -> bool:
    """Whether a text-showing operator's operands hold a string that is not all blanks (TJ: inside its array)."""
    for operand in operands:
        if isinstance(operand, list) and _glyphs(operand):
            return True
        if isinstance(operand, (str, bytes)) and operand.strip():
            return True
    return False


def read(data: bytes, name: str, limits: Limits) -> tuple[str, str]:
    """("text", text), ("too_large", reason) or ("invisible", reason). Raises whatever pypdf raises.

    Runs in the child; callable directly only where no time budget is needed.
    """
    from pypdf import PdfReader, apply_configuration

    # Scoped to this read: every decompression filter pypdf applies while
    # loading the document and extracting its text stops at the budget,
    # raising LimitReachedError (a PyPdfError, so "could not be read").
    budget = {
        "zlib_maximum_output_length": limits.max_stream_bytes,
        "lzw_maximum_output_length": limits.max_stream_bytes,
        "run_length_maximum_output_length": limits.max_stream_bytes,
        "array_based_stream_maximum_output_length": limits.max_stream_bytes,
    }
    with apply_configuration(**budget):
        reader = PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
        if page_count > limits.max_pages:
            return "too_large", f"{name} has {page_count} pages, over the {limits.max_pages}-page limit."
        pages: list[str] = []
        total = content = 0
        for page in reader.pages:
            # Inflating is cheap and pypdf keeps the result; PARSING the
            # content is the cost, so the budget is checked before it.
            try:
                stream = page.get_contents()
                content += len(stream.get_data()) if stream is not None else 0
            except (AttributeError, KeyError, TypeError):
                # A malformed /Contents (a number, a bare dictionary).
                # extract_text fails on the same construction and reads
                # the page as empty, parsing nothing - so it costs nothing.
                pass
            if content > limits.max_content_bytes:
                return "too_large", (
                    f"{name}'s page content inflates to more than {limits.max_content_bytes / 1e6:.0f} MB. "
                    f"A rating decision is a few pages; refusing rather than parsing it."
                )
            invisible = _InvisibleText()
            pages.append(page.extract_text(visitor_operand_before=invisible.before,
                                           visitor_operand_after=invisible.after) or "")
            if invisible.shown:
                return "invisible", (
                    f"{name} carries text that is not drawn on the page (PDF text render mode 3 or 7), which is how "
                    f"OCR software lays its reading over a scanned image. Recheck does not read OCR text, which "
                    f"can misread a figure; supply a text-layer PDF or a .txt transcript."
                )
            total += len(pages[-1]) + 1
            if total > limits.max_chars:  # stop at the page that crosses it
                return "too_large", (
                    f"{name} yields more than {limits.max_chars:,} characters of text. A rating "
                    f"decision is a few pages; refusing rather than parsing it."
                )
    return "text", "\n".join(pages)
