"""Document ingestion and final-degree conversion.

These cover three claims the README makes that were previously untested:
the PDF path, the property behaviour of the conversion to the final degree,
and the refusal to proceed on an unparseable document.

The rounding property tests existed in an earlier form and were lost when the
engine was rewritten for pairwise integer combination. Restored here, because
final_degree is the last transformation before the number a veteran reads.
"""

from __future__ import annotations

import pathlib

import pytest

from recheck.cfr.combine import final_degree
from recheck.extract.deterministic import parse
from recheck.graph import ScannedDocument, read_document


# --------------------------------------------------------------------------
# Conversion to the final degree of disability
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "combined,expected",
    [
        (0, 0), (4, 0), (5, 10), (6, 10), (14, 10), (15, 20),
        (44, 40), (45, 50), (52, 50), (65, 70), (74, 70), (75, 80),
        (76, 80), (94, 90), (95, 100), (99, 100), (100, 100),
    ],
)
def test_boundaries(combined, expected):
    """4.25(a): nearest number divisible by 10, values ending in 5 upward."""
    assert final_degree(combined) == expected


def test_values_ending_in_five_always_round_up():
    """Explicitly pinned because it is the rule most often implemented wrong.

    Banker's rounding would send 75 to 80 but 65 to 60. The regulation says
    5 is "adjusted upward", always.
    """
    for tens in range(0, 10):
        value = tens * 10 + 5
        assert final_degree(value) == (tens + 1) * 10


def test_monotonic_across_the_whole_range():
    """A higher combined value can never produce a lower final degree."""
    previous = -1
    for combined in range(0, 101):
        current = final_degree(combined)
        assert current >= previous, f"final_degree({combined}) dropped below its predecessor"
        previous = current


def test_result_is_always_a_multiple_of_ten_in_range():
    for combined in range(0, 101):
        result = final_degree(combined)
        assert result % 10 == 0
        assert 0 <= result <= 100


def test_conversion_never_moves_a_value_by_more_than_five():
    """Sanity bound: this is a rounding step, not a rescaling."""
    for combined in range(0, 101):
        assert abs(final_degree(combined) - combined) <= 5


# --------------------------------------------------------------------------
# PDF ingestion
# --------------------------------------------------------------------------

LETTER_LINES = [
    "*** SYNTHETIC DOCUMENT - NOT A REAL VA DECISION ***",
    "DEPARTMENT OF VETERANS AFFAIRS",
    "",
    "                    RATING DECISION",
    "  1. Post-traumatic stress disorder (DC 9411) ....... 60%",
    "  2. Limitation of flexion, right knee (DC 5260) .... 20%",
    "  3. Limitation of flexion, left knee (DC 5260) ..... 10%",
    "  4. Tinnitus (DC 6260) ............................. 10%",
    "",
    "COMBINED EVALUATION FOR COMPENSATION: 70%",
]


def _write_text_pdf(path: pathlib.Path) -> pathlib.Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    for line in LETTER_LINES:
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))
    return path


def test_text_layer_pdf_is_ingested(tmp_path):
    path = _write_text_pdf(tmp_path / "letter.pdf")
    text = read_document(path)
    assert "RATING DECISION" in text
    assert "COMBINED EVALUATION" in text


def test_ratings_parse_identically_from_pdf_and_from_text(tmp_path):
    """The PDF path must not change the extracted facts."""
    pdf_extraction = parse(read_document(_write_text_pdf(tmp_path / "letter.pdf")))
    txt_path = tmp_path / "letter.txt"
    txt_path.write_text("\n".join(LETTER_LINES), encoding="utf-8")
    txt_extraction = parse(read_document(txt_path))

    assert [r.percent for r in pdf_extraction.ratings] == [r.percent for r in txt_extraction.ratings]
    assert pdf_extraction.stated_combined == txt_extraction.stated_combined == 70
    assert [r.laterality for r in pdf_extraction.ratings] == [r.laterality for r in txt_extraction.ratings]


def test_image_only_pdf_is_refused_by_name_not_silently_empty(tmp_path):
    """A scanned letter must produce a clear refusal.

    Returning empty text would send an "extraction failed" downstream with no
    explanation, and the cause - that OCR is out of scope - would be invisible
    to the person holding the document.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(120, 120, 120)
    pdf.rect(20, 20, 100, 60, style="F")
    path = tmp_path / "scanned.pdf"
    pdf.output(str(path))

    with pytest.raises(ScannedDocument, match="requires OCR"):
        read_document(path)


def test_missing_file_raises_rather_than_returning_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_document(tmp_path / "absent.txt")


# --------------------------------------------------------------------------
# Unparseable input
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "content,reason_fragment",
    [
        ("", "no rating lines matched"),
        ("Dear veteran,\n\nThank you for your enquiry.\n", "no rating lines matched"),
        (
            "RATING DECISION\n  1. Tinnitus (DC 6260) ..... 10%\n",
            "no combined evaluation statement found",
        ),
    ],
)
def test_unparseable_documents_report_a_reason(content, reason_fragment):
    extraction = parse(content)
    assert not extraction.ok
    assert extraction.unparsed_reason is not None
    assert reason_fragment in extraction.unparsed_reason


def test_percentages_outside_the_decision_section_are_not_extracted():
    """The historical-percentage and rating-criteria trap, pinned directly."""
    letter = (
        "DECISION\n\n"
        "Evaluation of tinnitus, currently evaluated as 30 percent disabling, "
        "is increased to 10 percent effective January 1, 2026.\n\n"
        "Your combined evaluation for compensation is 10 percent.\n\n"
        "REASONS FOR DECISION\n\n"
        "An evaluation of 70 percent is assigned only where there is severe "
        "impairment. A 100 percent evaluation requires total impairment.\n"
    )
    extraction = parse(letter)
    percents = [r.percent for r in extraction.ratings]
    assert percents == [10], f"picked up historical or criteria percentages: {percents}"
    assert 30 not in percents
    assert 70 not in percents
    assert 100 not in percents
