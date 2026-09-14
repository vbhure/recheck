"""Release-candidate fixes found open by the deep review, each with its regression test.

  SEC-4  `--fresh` on a case whose directory is a link (a Windows junction, or
         a symbolic link) renamed the link, then walked through it clearing the
         read-only attribute of every file it points to - files outside the
         store - and left a .discarded-* link behind.
  CLI-5  A .txt letter saved as Windows-1252 ('ANSI') was refused with only the
         codec's message; it is still refused, and now says to save it as UTF-8.
"""

from __future__ import annotations

import os
import pathlib
import stat

from _support import EXIT_CANNOT_PROCEED, EXIT_OK, main, tabular_letter
from recheck.case import CaseStore

SETTLED = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20), ("Left knee strain", 10),
           ("Tinnitus", 10)]


def _link(link: pathlib.Path, target: pathlib.Path) -> None:
    """A directory link anyone who can write the store can make: a junction on Windows."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _is_link(path: pathlib.Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


# --------------------------------------------------------------------------
# SEC-4: discarding a case directory that is a link removes the link only
# --------------------------------------------------------------------------

def test_fresh_on_a_case_directory_that_is_a_link_removes_only_the_link(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    private = outside / "private.txt"
    private.write_text("not the store's to touch", encoding="utf-8")
    (outside / "case.json").write_text("{}", encoding="utf-8")
    for path in (private, outside / "case.json"):
        os.chmod(path, stat.S_IREAD)
    store = tmp_path / "runs"
    store.mkdir()
    _link(store / "c1", outside)
    assert CaseStore(store).exists("c1")  # precondition: the link reaches a case file

    letter = tabular_letter(tmp_path / "c1.txt", SETTLED, stated=70)
    try:
        code, out, err = main("audit", letter, "--case", "c1", "--fresh", store=store)
        assert code == EXIT_OK, out + err
        for path in (private, outside / "case.json"):
            assert not os.stat(path).st_mode & stat.S_IWRITE, f"{path.name} outside the store was made writable"
        assert private.read_text(encoding="utf-8") == "not the store's to touch"
        leftovers = [p.name for p in store.iterdir() if p.name.startswith(".discarded-")]
        assert leftovers == [], f"the discarded link was left in the store: {leftovers}"
        assert not _is_link(store / "c1") and (store / "c1" / "case.json").is_file()
    finally:
        for path in (private, outside / "case.json"):
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_discarding_a_link_whose_target_is_gone_removes_the_link(tmp_path):
    store = tmp_path / "runs"
    store.mkdir()
    target = tmp_path / "gone"
    target.mkdir()
    _link(store / "c2", target)
    target.rmdir()
    CaseStore(store).discard("c2")
    assert list(store.iterdir()) == []


# --------------------------------------------------------------------------
# CLI-5: a .txt letter that is not UTF-8 is refused in words a reviewer can act on
# --------------------------------------------------------------------------

def _ansi_letter(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """A letter as Notepad or Word save it as 'ANSI' plain text: Windows-1252, where a
    right single quotation mark is the byte 0x92 - not UTF-8."""
    path = tabular_letter(tmp_path / name, SETTLED, stated=70)
    text = path.read_text(encoding="utf-8").replace("RATING DECISION", "The veteran\u2019s RATING DECISION")
    path.write_bytes(text.encode("cp1252"))
    assert b"\x92" in path.read_bytes()
    return path


def test_a_letter_saved_as_windows_1252_is_refused_with_a_save_as_utf8_instruction(tmp_path):
    store = tmp_path / "runs"
    letter = _ansi_letter(tmp_path, "ansi.txt")
    code, out, err = main("audit", letter, "--case", "ansi", store=store)
    flat = " ".join((out + err).split())
    assert code == EXIT_CANNOT_PROCEED, out + err
    assert "save the letter as UTF-8" in flat, out + err
    # Refused, not guessed: nothing was read from it.
    assert "DISCREPANCY" not in out and "recomputed" not in out
    case = CaseStore(store).load("ansi")
    assert case.status == "unparsed" and case.ratings == []
    assert "save the letter as UTF-8" in case.extraction_failure()

    # The sweep's triage row says the same.
    folder = tmp_path / "letters"
    folder.mkdir()
    _ansi_letter(folder, "ansi2.txt")
    code, out, _ = main("sweep", folder, store=store)
    assert code == EXIT_CANNOT_PROCEED
    assert "save the letter as UTF-8" in " ".join(out.split()), out


def test_control_the_same_letter_saved_as_utf8_is_read(tmp_path):
    letter = _ansi_letter(tmp_path, "utf8.txt")
    letter.write_bytes(letter.read_bytes().decode("cp1252").encode("utf-8"))
    code, out, err = main("audit", letter, "--case", "utf8", "--brief", store=tmp_path / "runs")
    assert code == EXIT_OK and "DISCREPANCY" in out, out + err
