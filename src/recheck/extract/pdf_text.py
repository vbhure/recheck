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
ends the interpreter it launched (checked), so the kill is complete.

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
    "error" (value: the exception pypdf raised, to be re-raised as itself),
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
    request = pickle.dumps({"path": list(sys.path), "data": data, "name": name,
                            "limits": dataclasses.asdict(limits)})
    try:
        # subprocess.run kills the child when the timeout expires.
        done = subprocess.run([sys.executable, "-I", "-c", _BOOTSTRAP], input=request, capture_output=True,
                              timeout=seconds, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return Outcome("timeout")
    try:
        # Our own child's report: plain values and pypdf's exception objects.
        kind, value, records = pickle.loads(done.stdout)
    except Exception:  # noqa: BLE001 - killed, crashed, or wrote something else
        last = done.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"exit code {done.returncode}"]
        return Outcome("died", last[0])
    _replay(records)
    return Outcome(kind, value)


class _Collect(logging.Handler):
    """Keeps the child's log records (pypdf's warnings about a malformed file)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
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
    report = sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing else may write into the report
    collect = _Collect()
    logging.getLogger().addHandler(collect)
    try:
        kind, value = read(request["data"], request["name"], Limits(**request["limits"]))
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which re-raises it
        kind, value = "error", exc
    try:
        payload = pickle.dumps((kind, value, collect.records))
    except Exception as exc:  # noqa: BLE001 - an exception or record that does not pickle
        payload = pickle.dumps(("error", RuntimeError(f"{type(value).__name__}: {value} ({exc})"), []))
    report.write(payload)
    report.flush()


def read(data: bytes, name: str, limits: Limits) -> tuple[str, str]:
    """("text", text) or ("too_large", reason). Raises whatever pypdf raises.

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
            pages.append(page.extract_text() or "")
            total += len(pages[-1]) + 1
            if total > limits.max_chars:  # stop at the page that crosses it
                return "too_large", (
                    f"{name} yields more than {limits.max_chars:,} characters of text. A rating "
                    f"decision is a few pages; refusing rather than parsing it."
                )
    return "text", "\n".join(pages)
