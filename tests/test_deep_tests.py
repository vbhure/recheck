"""Mutation-testing regressions: safety checks no existing test would miss.

Each test here was written against a mutation of 12705f8 that removed or
weakened one safety check and left the whole suite passing (see the review
lane "tests"). Every test fails with its mutation applied and passes
without it; the mutation is named in each docstring.
"""

from __future__ import annotations

import json

import pytest

from _support import run_audit, tabular_letter
from recheck.case import CaseCorrupt

PAIR = [("Post-traumatic stress disorder", 60), ("Right knee strain", 20),
        ("Limitation of motion of the knee", 10), ("Tinnitus", 10)]


def _open_case(tmp_path):
    letter = tabular_letter(tmp_path / "pair.txt", PAIR, stated=70)
    store, _ = run_audit(tmp_path / "runs", "pair", letter)
    assert store.load("pair").status == "awaiting_human"
    return store


def _edit(tmp_path, change) -> None:
    path = tmp_path / "runs" / "pair" / "case.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


# --------------------------------------------------------------------------
# case.json values are checked on a case with no result to re-derive
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [("extremity_group", "torso"), ("laterality", "north")])
def test_an_impossible_fact_on_an_open_case_is_refused(tmp_path, field, value):
    """Mutations load_group_value / load_side_value. The existing tamper test
    edits a COMPLETE case, where the re-derived figure no longer matches and
    the load refuses it anyway; the value check itself was never exercised.
    On an open case nothing is re-derived: a group of "torso" on the knee of
    unstated side made it neither asked nor paired, so the question lost it."""
    store = _open_case(tmp_path)
    _edit(tmp_path, lambda raw: raw["decisions"][2].update({field: value}))
    with pytest.raises(CaseCorrupt, match="impossible"):
        store.load("pair")
