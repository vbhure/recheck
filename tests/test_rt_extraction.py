"""Red-team findings against extraction, each with a regression lock.

Every letter below was rebuilt from the red team's description. Each test
fails on the release candidate (af714cb) and passes after the fix.

  ARITH-F1        A rating sentence in an unsupported wording was skipped -
                  including "An evaluation of 30 percent is assigned from
                  ...", the later stage of a staged rating - and a figure was
                  computed on the partial list. Every percentage in the
                  decision section must now be accounted for.
  ARITH-F2        A staged rating in prose counted both stages as separate
                  disabilities; a tabular row with two stages used the first.
  ARITH-F5        Two identical prose rating sentences were deduplicated to
                  one, so two separately rated scars counted once.
  ARITH-F7        "Your previous combined evaluation ... is 30 percent" was
                  taken as the stated value.
  ARITH-F9        Rows after a stand-alone "Evidence" heading were cut while
                  the numbering stayed contiguous.
  MODEL-RT-P2P6-01  A wrapped (or bare-CR) FINAL numbered row was dropped:
                  the contiguity check cannot see a missing last row.
  MODEL-RT-P2P6-02 / SECRETS-F2  With no sentence-ending period in the
                  letterhead, the veteran's name, file number and date became
                  part of the first condition name - stored, printed and sent
                  to the model provider.
  MODEL-RT-P2P6-04  A ligature, soft hyphen or zero-width character inside a
                  limb word hid it from the lexicon and the "none" veto.
  FILES-P4-01 / SECRETS-F3  A percentage over 100 was persisted, and the next
                  load of the case raised CaseCorrupt as a traceback, exit 1.
  FILES-P4-05     A 68 KB PDF inflating to 70 MB passed the byte and page
                  caps; the prose patterns were quadratic in sentence length.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import zlib

import pytest

from _support import (
    EXIT_CANNOT_PROCEED,
    EXIT_OK,
    LETTERS,
    batch,
    case_json,
    item,
    main,
    prompt_text,
    run_audit,
    scripted,
)
from recheck.case import CaseStore
from recheck.extract.deterministic import parse
from recheck.graph import DocumentTooLarge, ExtractNode, open_case, read_document

HEADER = (
    "*** SYNTHETIC TEST LETTER - NOT A REAL VA DECISION ***\n"
    "DEPARTMENT OF VETERANS AFFAIRS\n"
    "Regional Office\n\n"
    "Name: R. SYNTHETIC\n"
    "File Number: 00-000-002\n"
    "Date of Notification: April 2, 2026\n\n"
    "DECISION\n\n"
)


def prose(*sentences: str, stated: int) -> str:
    body = "\n\n".join(sentences)
    return f"{HEADER}{body}\n\nYour combined evaluation for compensation is {stated} percent.\n"


def tabular(*rows: str, stated: int, after: str = "") -> str:
    return ("RATING DECISION\n\n" + "\n".join(rows) + after
            + f"\n\nCOMBINED EVALUATION FOR COMPENSATION: {stated}%\n")


def refused(text: str) -> str:
    """The refusal reason; fails if the letter was read."""
    extraction = parse(text)
    got = [(r.condition, r.percent) for r in extraction.ratings]
    assert not extraction.ok and got == [], f"expected a refusal, read {got} stated {extraction.stated_combined}"
    return extraction.unparsed_reason or ""


def audit(tmp_path, name: str, text: str, *extra: str) -> tuple[int, str]:
    letter = tmp_path / f"{name}.txt"
    letter.write_text(text, encoding="utf-8")
    code, out, err = main("audit", letter, "--case", name, *extra, store=tmp_path / "runs")
    return code, out + err


# --------------------------------------------------------------------------
# ARITH-F1: every percentage in the decision section is accounted for
# --------------------------------------------------------------------------

PROSE_DROPPED = prose(
    "Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
    "Service connection for limitation of flexion of the left knee is granted with an evaluation of 10 percent.",
    "Service connection for limitation of flexion of the right knee is granted. An evaluation of 10 "
    "percent is assigned effective January 9, 2024.",
    stated=50,
)
STAGED_THREE = prose(
    "Service connection for post-traumatic stress disorder is granted with an evaluation of 40 percent.",
    "Service connection for left knee strain is granted with an evaluation of 20 percent effective "
    "January 9, 2024. An evaluation of 30 percent is assigned from March 1, 2025.",
    stated=60,
)


def test_a_rating_sentence_in_an_unread_wording_refuses_the_letter():
    """Read: 30% PTSD and 10% left knee, recomputed 40% against a correct 50%."""
    assert "10 percent" in refused(PROSE_DROPPED)


def test_the_later_stage_of_a_staged_rating_refuses_the_letter():
    """Read: the superseded 20% knee, recomputed 50% against a correct 60%."""
    assert "30 percent" in refused(STAGED_THREE)


@pytest.mark.parametrize(
    "sentence",
    [
        "Service connection for tinnitus is granted with an evaluation of ten percent.",
        "Service connection for tinnitus is granted with an evaluation of 10%.",
        "Service connection for tinnitus is granted with an evaluation of 10 per-\ncent.",
        "A 10 percent evaluation is assigned for tinnitus.",
        # How a continued rating is worded. The prior-value clause is only
        # history in front of a rating that was read; alone it is the rating.
        "Evaluation of tinnitus, which is currently 10 percent disabling, is continued.",
    ],
)
def test_percentages_in_wordings_recheck_does_not_read_are_refused_not_dropped(sentence):
    letter = prose("Service connection for asthma is granted with an evaluation of 30 percent.", sentence, stated=40)
    refused(letter)


def test_a_prior_value_is_history_only_directly_before_the_rating_it_precedes():
    """ "X, currently evaluated as 50 percent disabling, and Y is increased to
    10 percent" names two ratings; reading the 50 as history drops one."""
    refused(prose("Evaluation of asthma, currently evaluated as 50 percent disabling, and tinnitus is "
                  "increased to 10 percent.", stated=60))


def test_prior_values_before_the_rating_are_still_read_as_history():
    letter = prose(
        "Evaluation of bronchial asthma, currently evaluated as\n30 percent disabling, is increased to 60 percent.",
        "Your claim for right De Quervain's tenosynovitis, previously 10 percent, is increased to 20 percent.",
        stated=70,
    )
    extraction = parse(letter)
    assert [(r.condition, r.percent) for r in extraction.ratings] == [
        ("bronchial asthma", 60), ("right De Quervain's tenosynovitis", 20)]
    assert extraction.stated_combined == 70


def test_the_refusals_reach_the_cli_as_could_not_read_exit_3(tmp_path):
    for name, text in (("dropped", PROSE_DROPPED), ("staged", STAGED_THREE)):
        code, output = audit(tmp_path, name, text)
        assert code == EXIT_CANNOT_PROCEED, output
        assert "COULD NOT READ THE LETTER" in output
        assert "DISCREPANCY" not in output and "Traceback" not in output


# --------------------------------------------------------------------------
# ARITH-F2 and ARITH-F5: one condition rated twice is not guessed at
# --------------------------------------------------------------------------

def test_a_staged_prose_rating_is_not_counted_as_two_disabilities():
    """Counted: 40, 20 and 30 -> 70% against a correct 60%."""
    letter = prose(
        "Service connection for post-traumatic stress disorder is granted with an evaluation of 40 percent.",
        "Service connection for limitation of flexion of the left knee is granted with an evaluation of "
        "20 percent effective January 9, 2024.",
        "Evaluation of limitation of flexion of the left knee is increased to 30 percent effective March 1, 2025.",
        stated=60,
    )
    assert "more than once" in refused(letter)


def test_a_tabular_row_with_two_stages_is_refused():
    """Read: the first (superseded) 20%, recomputed 50% against a correct 60%."""
    letter = tabular("  1. Post-traumatic stress disorder ...... 40%",
                     "  2. Limitation of flexion of the left knee ...... 20% from January 9, 2024; "
                     "30% from March 1, 2025", stated=60)
    assert "30%" in refused(letter)


def test_two_identical_prose_rating_sentences_are_not_collapsed_into_one():
    """Two separately rated tender scars counted once: 40% against a correct 50%."""
    scar = "Service connection for tender scar is granted with an evaluation of 10 percent."
    letter = prose("Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
                   scar, scar, "Service connection for tinnitus is granted with an evaluation of 10 percent.",
                   stated=50)
    assert "more than once" in refused(letter)


# --------------------------------------------------------------------------
# ARITH-F7: the stated combined evaluation
# --------------------------------------------------------------------------

def test_a_previous_combined_evaluation_is_not_the_stated_value():
    letter = (HEADER + "Your previous combined evaluation for compensation is 30 percent.\n\n"
              "Service connection for post-traumatic stress disorder is granted with an evaluation of 50 percent.\n\n"
              "Service connection for tinnitus is granted with an evaluation of 10 percent.\n\n"
              "Your combined evaluation for compensation is 60 percent.\n")
    extraction = parse(letter)
    assert extraction.ok and extraction.stated_combined == 60


def test_two_different_current_combined_statements_are_refused():
    letter = tabular("  1. Post-traumatic stress disorder ...... 30%", stated=30,
                     after="\n\nYour combined evaluation for compensation is 40 percent.")
    assert "more than one combined evaluation" in refused(letter)


# --------------------------------------------------------------------------
# ARITH-F9: what a stop heading cuts
# --------------------------------------------------------------------------

def test_rows_after_a_mid_list_evidence_heading_are_not_silently_cut():
    """Read: rows 1-2, contiguous; recomputed 40% against a correct 50%."""
    letter = tabular("  1. Post-traumatic stress disorder ...... 30%",
                     "  2. Left knee strain ...... 10%", "",
                     "Evidence", "  VA examination of March 2, 2026",
                     "  3. Right knee strain ...... 10%", stated=50)
    assert "Evidence" in refused(letter)


def test_a_prose_rating_after_a_stop_heading_is_not_silently_cut():
    letter = prose("Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.",
                   "EVIDENCE",
                   "Service connection for left knee strain is granted with an evaluation of 10 percent.",
                   stated=40)
    refused(letter)


def test_a_restatement_under_reasons_for_decision_is_still_accepted():
    """REASONS FOR DECISION repeats the decision; that is not a cut list."""
    letter = (HEADER
              + "Service connection for post-traumatic stress disorder is granted with an evaluation of 30 percent.\n\n"
              + "REASONS FOR DECISION\n\n"
              + "Service connection for post-traumatic stress disorder has been granted with an evaluation of "
              + "30 percent. A 50 percent evaluation requires occupational impairment with reduced reliability.\n\n"
              + "Your combined evaluation for compensation is 30 percent.\n")
    extraction = parse(letter)
    assert extraction.ok and [(r.condition, r.percent) for r in extraction.ratings] == [
        ("post-traumatic stress disorder", 30)]


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-01: a missing LAST row
# --------------------------------------------------------------------------

WRAPPED_LAST = tabular("  1. Post-traumatic stress disorder ...... 50%",
                       "  2. Left knee strain with limitation of flexion and",
                       "     instability ..... 30%", stated=70)


def test_a_wrapped_final_row_refuses_the_letter():
    """Read: row 1 only, recomputed 50% against a stated 70%, exit 0."""
    assert "line 5" in refused(WRAPPED_LAST)


def test_a_bare_carriage_return_inside_the_final_row_refuses_the_letter(tmp_path):
    path = tmp_path / "cr.txt"
    path.write_bytes(b"RATING DECISION\n\n  1. Left wrist strain ...... 30%\n  2. Right wrist\rstrain ..... 30%\n\n"
                     b"COMBINED EVALUATION FOR COMPENSATION: 50%\n")
    refused(read_document(path))


def test_a_final_numbered_line_without_a_percentage_refuses_the_letter():
    letter = tabular("  1. Post-traumatic stress disorder ...... 50%", "  2. Left knee strain with instability",
                     stated=70)
    assert "numbered lines" in refused(letter)


def test_a_wrapped_final_row_reaches_the_cli_as_exit_3(tmp_path):
    code, output = audit(tmp_path, "lastrow", WRAPPED_LAST, "--brief")
    assert code == EXIT_CANNOT_PROCEED and "COULD NOT READ THE LETTER" in output
    assert "DISCREPANCY" not in output


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-02 / SECRETS-F2: letterhead text in a condition name
# --------------------------------------------------------------------------

BLEED = (
    "*** SYNTHETIC TEST LETTER - NOT A REAL VA DECISION ***\n"
    "DEPARTMENT OF VETERANS AFFAIRS\nRegional Office\n\n"
    "Name: Jane Q Veteran\nFile Number: 123-45-6789\nDate of Notification: August 19, 2026\n\n"
    "DECISION\n\n"
    "Left cubital tunnel syndrome is continued as 20 percent disabling.\n\n"
    "Your claim for right De Quervain's tenosynovitis, previously 10 percent, is increased to 20 percent.\n\n"
    "Your combined evaluation for compensation is 40 percent.\n"
)


def test_letterhead_text_does_not_become_part_of_a_condition_name():
    extraction = parse(BLEED)
    assert [r.condition for r in extraction.ratings] == [
        "Left cubital tunnel syndrome", "right De Quervain's tenosynovitis"]


def test_letterhead_with_no_blank_lines_does_not_bleed_either():
    letter = BLEED.replace("\n\n", "\n")
    assert [r.condition for r in parse(letter).ratings] == [
        "Left cubital tunnel syndrome", "right De Quervain's tenosynovitis"]


def test_a_condition_name_carrying_a_file_number_is_refused_not_used():
    """No label, no heading, no blank line: the name itself shows the bleed."""
    letter = ("Veteran Jane Q Sample 123-45-6789\n"
              "Tinnitus is continued as 10 percent disabling.\n\n"
              "Your combined evaluation for compensation is 10 percent.\n")
    assert "long number" in refused(letter)


def test_only_condition_names_reach_the_model_prompt(tmp_path):
    """The prompt carried 'Name: Jane Q Veteran File Number: 123-45-6789 ...'."""
    letter = tmp_path / "bleed.txt"
    letter.write_text(BLEED, encoding="utf-8")
    factory = scripted(batch(item("Left cubital tunnel syndrome", "upper"),
                             item("right De Quervain's tenosynovitis", "upper")))
    store, _ = run_audit(tmp_path / "runs", "bleed", letter, factory, classifier="scripted")
    (model,) = factory.models
    sent = prompt_text(model)
    assert "cubital tunnel" in sent
    for leak in ("Jane", "Veteran", "File Number", "123-45-6789", "August", "percent", "DECISION"):
        assert leak not in sent, f"{leak!r} reached the model prompt: {sent!r}"
    assert not re.search(r"\d", sent), sent
    stored = json.dumps(store.load("bleed").ratings)
    assert "Jane" not in stored and "123-45-6789" not in stored


# --------------------------------------------------------------------------
# MODEL-RT-P2P6-04: invisible and look-alike characters in limb words
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "written",
    ["ﬁnger",          # the "fi" ligature, as pypdf often extracts it
     "fin­ger",        # soft hyphen
     "fin​ger",        # zero-width space
     "ｆｉｎｇｅｒ"],  # full-width letters
)
def test_a_limb_word_with_invisible_or_compatibility_characters_is_still_recognised(written):
    letter = tabular("  1. Left wrist strain ...... 30%", f"  2. Tenosynovitis, right {written} ...... 30%", stated=60)
    extraction = parse(letter)
    assert [(r.condition, r.extremity_group) for r in extraction.ratings] == [
        ("Left wrist strain", "upper"), ("Tenosynovitis, right finger", "upper")]


def test_a_dotted_capital_i_is_read_as_the_letter_it_shows():
    letter = tabular("  1. Left HİP strain ...... 20%", "  2. Right hip strain ...... 10%", stated=30)
    assert [r.extremity_group for r in parse(letter).ratings] == ["lower", "lower"]


@pytest.mark.parametrize("written", ["кnee", "Tenosynovitis, right kneе", "Нір"])
def test_look_alike_letters_from_another_alphabet_are_refused(written):
    """NFKC cannot fold Cyrillic into Latin. "кnee" reads "knee" to a person
    and matches nothing in the lexicon or the veto."""
    letter = tabular("  1. Left wrist strain ...... 30%", f"  2. Right {written} strain ...... 30%", stated=60)
    refused(letter)


def test_other_alphabets_after_a_stop_heading_do_not_refuse_the_letter():
    """Evidence text is not read for facts; a Greek mu in "5 μg/dL" is fine there."""
    letter = tabular("  1. Bronchial asthma ...... 60%", stated=60,
                     after="\n\nREASONS FOR DECISION\n\nTheophylline level of 5 μg/mL.")
    assert parse(letter).ok


def test_a_ligature_limb_word_is_not_dropped_by_a_model_none(tmp_path):
    """Before: the replayed fixture's 'none' was accepted, 50%, a false
    POTENTIAL DISCREPANCY, exit 0. The ASCII control gives 60%, no discrepancy."""
    name = "Tenosynovitis, right ﬁnger"
    fixture = tmp_path / "none.json"
    fixture.write_text(json.dumps({"classifications": [
        {"condition": name, "extremity_group": "none", "confidence": 0.9}]}), encoding="utf-8")
    text = tabular("  1. Left wrist strain ...... 30%", f"  2. {name} ...... 30%", stated=60)
    code, output = audit(tmp_path, "lig", text, "--scripted", "--classifications", fixture)
    assert code == EXIT_OK, output
    assert "NO DISCREPANCY FOUND" in output
    assert case_json(tmp_path / "runs", "lig")["recomputed_degree"] == 60


# --------------------------------------------------------------------------
# FILES-P4-01 / SECRETS-F3: percentages over 100
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "DECISION\n\n  1. Tinnitus ............ 999%\n  2. Asthma ............ 60%\n\n"
        "COMBINED EVALUATION FOR COMPENSATION: 70%\n",
        tabular("  1. Bronchial asthma ...... 150%", stated=70),
        tabular("  1. Bronchial asthma ...... 101%", stated=70),
        tabular("  1. Bronchial asthma ...... 60%", stated=150),
        prose("Service connection for migraine is granted with an evaluation of 999 percent.", stated=90),
    ],
    ids=["row-999", "row-150", "row-101", "combined-150", "prose-999"],
)
def test_a_percentage_over_100_is_refused_at_extraction(tmp_path, text):
    assert "100%" in refused(text)
    code, output = audit(tmp_path, "over", text, "--brief")
    assert code == EXIT_CANNOT_PROCEED, output
    assert "COULD NOT READ THE LETTER" in output and "Traceback" not in output


# --------------------------------------------------------------------------
# FILES-P4-05: bounded work, not only bounded bytes
# --------------------------------------------------------------------------

def _raw_pdf(contents: list[bytes]) -> bytes:
    """A minimal PDF with one FlateDecode content stream per page (stdlib only)."""
    n = len(contents)
    font = 3 + 2 * n
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>".encode()]
    for i, content in enumerate(contents):
        packed = zlib.compress(content, 9)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                       f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode())
        objects.append(f"<< /Length {len(packed)} /Filter /FlateDecode >>\nstream\n".encode() + packed + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    out, offsets = bytearray(b"%PDF-1.7\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _text_page(text: bytes) -> bytes:
    return b"BT /F1 10 Tf 72 700 Td (" + text + b") Tj ET"


# Short sentences, so that on the unfixed code these tests FAIL (the bomb is
# read and parsed) instead of hanging in the old quadratic prose patterns.
BOMB_TEXT = b"ab. " * 750_000


def test_a_pdf_stream_that_inflates_far_beyond_a_letter_is_refused_quickly(tmp_path):
    """Before: 68 KB on disk, 70 MB inflated, 5-6 GB and over ten minutes.
    3 MB is enough to show the cap: pypdf's own default allows 75 MB."""
    from pypdf.errors import PyPdfError

    path = tmp_path / "bomb.pdf"
    path.write_bytes(_raw_pdf([_text_page(BOMB_TEXT)]))
    assert path.stat().st_size < 50_000
    started = time.monotonic()
    with pytest.raises(PyPdfError):
        read_document(path)
    assert time.monotonic() - started < 10


def test_pdf_text_is_capped_page_by_page(tmp_path, monkeypatch):
    monkeypatch.setattr("recheck.graph.MAX_DOCUMENT_CHARS", 1500)
    path = tmp_path / "long.pdf"
    path.write_bytes(_raw_pdf([_text_page(b"A" * 1000)] * 3))
    with pytest.raises(DocumentTooLarge, match="characters"):
        read_document(path)


def test_pdf_page_content_is_capped_across_pages(tmp_path, monkeypatch):
    """Each page under the per-stream cap, together far over it: a hundred
    pages of drawing operators ran for over five minutes."""
    monkeypatch.setattr("recheck.graph.MAX_PDF_CONTENT_BYTES", 50_000)
    path = tmp_path / "drawing.pdf"
    path.write_bytes(_raw_pdf([b"0 0 m 1 1 l S\n" * 2_000] * 3))  # 28 KB each, no text
    with pytest.raises(DocumentTooLarge, match="page content"):
        read_document(path)


@pytest.mark.parametrize("contents", [b"/Contents 42     ", b"/Contents <<>>   "])
def test_a_malformed_page_content_entry_is_still_a_clean_refusal(tmp_path, contents):
    """Measuring page content for the budget must not turn what pypdf reads as
    an empty page into an AttributeError escaping the extract node."""
    path = tmp_path / "malformed.pdf"
    path.write_bytes(_raw_pdf([_text_page(b"RATING DECISION")]).replace(b"/Contents 4 0 R", contents))
    code, out, err = main("audit", path, "--case", "malformed", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "requires OCR" in out + err and "Traceback" not in out + err


def test_text_documents_are_capped_by_characters_too(tmp_path, monkeypatch):
    monkeypatch.setattr("recheck.graph.MAX_DOCUMENT_CHARS", 1000)
    path = tmp_path / "long.txt"
    path.write_text("RATING DECISION\n" + "x" * 2000, encoding="utf-8")
    with pytest.raises(DocumentTooLarge, match="characters"):
        read_document(path)


def test_a_pdf_bomb_reaches_the_cli_as_could_not_read(tmp_path):
    path = tmp_path / "bomb.pdf"
    path.write_bytes(_raw_pdf([_text_page(BOMB_TEXT)]))
    code, out, err = main("audit", path, "--case", "bomb", store=tmp_path / "runs")
    assert code == EXIT_CANNOT_PROCEED
    assert "could not be read" in out + err and "Traceback" not in out + err


def test_parsing_a_long_sentence_is_linear_not_quadratic():
    """The prose patterns began "(?P<condition>[^.]*?)": 10 KB with no period
    took seconds, 40 KB over a minute, a megabyte hours."""
    text = "DECISION\n\n" + "word " * 4_000 + "\n\nYour combined evaluation for compensation is 10 percent.\n"
    started = time.monotonic()
    parse(text)
    assert time.monotonic() - started < 5


# --------------------------------------------------------------------------
# Agreed interface: the digest of the bytes a case was read from
# --------------------------------------------------------------------------

def _digest_recorded(tmp_path, source) -> object:
    store = CaseStore(tmp_path / "runs")
    open_case(store, "d", str(source))
    recorded = []
    save = store.save

    def spy(case):
        recorded.append(getattr(case, "document_sha256", "not set"))
        return save(case)

    store.save = spy  # type: ignore[method-assign]
    asyncio.run(ExtractNode(store, "d", str(source)).invoke_async(None))
    return recorded[-1]


def test_extraction_records_the_sha256_of_the_bytes_it_read(tmp_path):
    letter = tmp_path / "agrees.txt"
    letter.write_bytes((LETTERS / "06_agrees.txt").read_bytes())
    assert _digest_recorded(tmp_path, letter) == hashlib.sha256(letter.read_bytes()).hexdigest()


def test_a_refused_letter_still_records_its_digest_and_a_missing_one_records_none(tmp_path):
    letter = tmp_path / "wrapped.txt"
    letter.write_text(WRAPPED_LAST, encoding="utf-8")
    assert _digest_recorded(tmp_path, letter) == hashlib.sha256(letter.read_bytes()).hexdigest()
    assert _digest_recorded(tmp_path / "other", tmp_path / "absent.txt") is None
