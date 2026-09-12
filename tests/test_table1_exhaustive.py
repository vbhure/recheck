"""Exhaustive validation against every published cell of 38 CFR 4.25 Table I.

fixtures/cfr425_table1_points.json holds all 684 (running_value, next_rating,
published_combined) triples, extracted from the official eCFR API and
committed so this test runs offline and reproducibly.

This is an EXTERNAL oracle. None of these values were computed by Recheck;
they are the government's own published numbers. A reviewer can re-extract
them with tools/fetch_cfr.py and diff.
"""
import json
import pathlib

import pytest

from recheck.cfr.combine import combine_step

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "cfr425_table1_points.json"
POINTS = [tuple(p) for p in json.loads(FIXTURE.read_text())]


def test_fixture_is_the_expected_size():
    """Guards against a truncated or re-extracted-and-shrunk fixture."""
    assert len(POINTS) == 684


@pytest.mark.parametrize("running,rating,published", POINTS)
def test_every_published_cell(running, rating, published):
    assert combine_step(running, rating) == published


def test_no_cell_ever_reaches_one_hundred():
    """Partial disabilities can never combine to total disability."""
    for running, rating, published in POINTS:
        assert published < 100
