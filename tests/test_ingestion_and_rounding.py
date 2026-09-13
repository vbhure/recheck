"""Document ingestion, extraction, and final-degree conversion.

Extraction is the first thing that can silently lose a rating or invent a
side, and final_degree is the last transformation before the number a
veteran reads. Both are deterministic, so both are pinned directly.

Extraction defects regression-locked here:

  E1  The stop-heading search was unanchored: "with x-ray evidence of
      arthritis" inside a rating line matched EVIDENCE, cut the decision
      section there, and every later rating was silently dropped.
  E2  Sides were grounded against the whole source LINE, so "pain on the left
      side" in the sentence before a side-less knee made it a left knee.
  E3  Non-extremity hints were checked before anatomy, so "Radiculopathy,
      right lower extremity, associated with lumbosacral strain" became
      "none" ("lumbosacral") and dropped a 4.26 pair.
  E4  A linked condition's side was read as the rated condition's: "Left knee
      strain, secondary to right knee strain" is a LEFT knee.
  E5  Handedness is not a side: "(right hand dominant)" is not a right wrist.
  E6  Two identical numbered rows (two separately rated scars) are two
      evaluations; deduplicating them loses a rating.
  E7  Evidence for a hard-wrapped condition must point at the line where the
      condition starts, not "none recorded" and not a later line.
"""

from __future__ import annotations

import pathlib

import pytest

from recheck.cfr.combine import final_degree
from recheck.classify import derive_laterality
from recheck.extract.deterministic import _classify_extremity, parse
from recheck.graph import ScannedDocument, read_document


# --------------------------------------------------------------------------
# Conversion to the final degree of disability
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "combined,expected",
    [(0, 0), (4, 0), (5, 10), (6, 10), (14, 10), (15, 20), (44, 40), (45, 50), (52, 50), (65, 70),
     (74, 70), (75, 80), (76, 80), (94, 90), (95, 100), (99, 100), (100, 100)],
)
def test_boundaries(combined, expected):
    """4.25(a): nearest number divisible by 10, values ending in 5 upward."""
    assert final_degree(combined) == expected


def test_values_ending_in_five_always_round_up():
    """Banker's rounding would send 75 to 80 but 65 to 60. The regulation says
    5 is "adjusted upward", always."""
    for tens in range(10):
        assert final_degree(tens * 10 + 5) == (tens + 1) * 10


def test_monotonic_multiple_of_ten_and_within_five():
    previous = -1
    for combined in range(101):
        result = final_degree(combined)
        assert result >= previous and result % 10 == 0 and 0 <= result <= 100
        assert abs(result - combined) <= 5
        previous = result


@pytest.mark.parametrize("value", [-1, 101, 110])
def test_an_impossible_combined_value_is_refused(value):
    with pytest.raises(ValueError):
        final_degree(value)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

PROSE = """DECISION

The examiner noted pain on the left side. Service connection for limitation of flexion of the
knee is granted with an evaluation of 10 percent effective January 9, 2026.

Service connection for degenerative arthritis of the right knee with x-ray evidence of
arthritis is granted with an evaluation of 20 percent effective January 9, 2026.

Service connection for radiculopathy, right lower extremity, associated with lumbosacral strain
is granted with an evaluation of 20 percent effective January 9, 2026.

Service connection for left knee strain, secondary to right knee strain, is granted
with an evaluation of 10 percent effective January 9, 2026.

Service connection for tinnitus is granted with an evaluation of 10 percent.

EVIDENCE

An evaluation of 70 percent is assigned only where there is severe impairment.

Your combined evaluation for compensation is 70 percent.
"""


def test_prose_extraction_keeps_every_rating_and_its_first_line():
    extraction = parse(PROSE)
    got = [(r.percent, r.source_line_number, r.condition) for r in extraction.ratings]
    assert got == [
        (10, 3, "limitation of flexion of the knee"),                                     # E7
        (20, 6, "degenerative arthritis of the right knee with x-ray evidence of arthritis"),  # E1
        (20, 9, "radiculopathy, right lower extremity, associated with lumbosacral strain"),
        (10, 12, "left knee strain, secondary to right knee strain"),
        (10, 15, "tinnitus"),
    ]
    assert extraction.stated_combined == 70
    assert 70 not in [r.percent for r in extraction.ratings], "criteria under EVIDENCE are not ratings"


def test_the_evidence_line_for_a_wrapped_condition_contains_the_condition():
    first = parse(PROSE).ratings[0]
    assert "limitation of flexion of the" in first.source_line
    assert "knee is granted" in first.source_line


def test_a_side_in_an_adjacent_sentence_does_not_leak():
    """E2."""
    knee = parse(PROSE).ratings[0]
    assert "left" in knee.source_line.lower(), "precondition: the adjacent sentence is on the same line"
    assert derive_laterality(knee.condition) == "unknown"


def test_x_ray_evidence_inside_a_tabular_row_does_not_truncate():
    """E1, tabular form."""
    letter = (
        "RATING DECISION\n"
        "  1. Right knee strain with x-ray evidence of arthritis (DC 5003) ...... 10%\n"
        "  2. Left knee strain (DC 5260) ........................................ 10%\n"
        "  3. Tinnitus (DC 6260) ................................................ 10%\n"
        "\nEVIDENCE\n  4. Not a rating ....... 90%\n"
        "\nCOMBINED EVALUATION FOR COMPENSATION: 30%\n"
    )
    assert [r.percent for r in parse(letter).ratings] == [10, 10, 10]


def test_identical_tabular_rows_are_both_kept():
    """E6."""
    letter = (
        "RATING DECISION\n"
        "  1. Scar, painful (DC 7804) ...... 10%\n"
        "  2. Scar, painful (DC 7804) ...... 10%\n"
        "\nCOMBINED EVALUATION FOR COMPENSATION: 20%\n"
    )
    ratings = parse(letter).ratings
    assert [(r.condition, r.percent, r.source_line_number) for r in ratings] == [
        ("Scar, painful (DC 7804)", 10, 2), ("Scar, painful (DC 7804)", 10, 3)]


@pytest.mark.parametrize(
    "condition,group,side",
    [
        ("Radiculopathy, right lower extremity, associated with lumbosacral strain", "lower", "right"),  # E3
        ("Left knee strain, secondary to right knee strain", "lower", "left"),                           # E4
        ("knee strain secondary to right ankle injury", "lower", "unknown"),                             # E4
        ("Carpal tunnel syndrome, wrist (right hand dominant)", "upper", "unknown"),                      # E5
        ("Left wrist strain (right hand dominant)", "upper", "left"),                                    # E5
        ("Limitation of flexion, right knee (DC 5260)", "lower", "right"),
        ("Strain of both knees", "lower", "both"),
        ("Bilateral knee strain", "lower", "both"),
        ("Post-traumatic stress disorder (DC 9411)", "none", "unknown"),
    ],
)
def test_group_and_side_are_read_from_the_rated_condition_itself(condition, group, side):
    assert _classify_extremity(condition) == group
    assert derive_laterality(condition) == side


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
    assert [r.percent for r in parse(letter).ratings] == [10]


@pytest.mark.parametrize(
    "content,reason_fragment",
    [
        ("", "no rating lines matched"),
        ("Dear veteran,\n\nThank you for your enquiry.\n", "no rating lines matched"),
        ("RATING DECISION\n  1. Tinnitus (DC 6260) ..... 10%\n", "no combined evaluation statement found"),
    ],
)
def test_unparseable_documents_report_a_reason(content, reason_fragment):
    extraction = parse(content)
    assert not extraction.ok
    assert reason_fragment in (extraction.unparsed_reason or "")


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
    text = read_document(_write_text_pdf(tmp_path / "letter.pdf"))
    assert "RATING DECISION" in text and "COMBINED EVALUATION" in text


def test_ratings_parse_identically_from_pdf_and_from_text(tmp_path):
    """The PDF path must not change the extracted facts."""
    pdf = parse(read_document(_write_text_pdf(tmp_path / "letter.pdf")))
    txt_path = tmp_path / "letter.txt"
    txt_path.write_text("\n".join(LETTER_LINES), encoding="utf-8")
    txt = parse(read_document(txt_path))

    def facts(extraction):
        return [(r.percent, r.extremity_group, derive_laterality(r.condition)) for r in extraction.ratings]

    assert facts(pdf) == facts(txt) == [(60, "none", "unknown"), (20, "lower", "right"),
                                        (10, "lower", "left"), (10, "none", "unknown")]
    assert pdf.stated_combined == txt.stated_combined == 70


def test_image_only_pdf_is_refused_by_name_not_silently_empty(tmp_path):
    """Returning empty text would surface as an unexplained extraction
    failure; the cause - OCR is out of scope - must be named."""
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
