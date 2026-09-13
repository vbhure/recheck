"""AI JUSTIFICATION GATE - where the model is, and is not, worth a call.

Recheck uses a model for one judgment only: which extremity group a condition
name belongs to, for names the deterministic lexicon abstains on. This file is
the standing evidence for where that line falls. It rests on two
measurements, because an earlier version rested on one and it did not hold.

WHAT WENT WRONG BEFORE. The gate measured a 29-word lexicon against 138
rating-schedule titles, found it resolved 22 (15.9%), and offered that as the
case for a model. But the schedule is a closed, enumerable list: a lexicon of
the schedule's own anatomy covers it. The low number showed only that the
lexicon was small. And across the whole caseload sweep the model's
contribution came down to three schedule words (median, musculospiral,
sciatic) that belonged in the lexicon all along.

(a) RATING-SCHEDULE VOCABULARY - fixtures/va_condition_names.json
    138 titles from 38 CFR Part 4, labelled by diagnostic-code range: labels
    this project does not control. The lexicon now carries the schedule's
    anatomy (bones, joints, named peripheral nerves, the extremity tables'
    headings - every entry checked against the eCFR text, see
    recheck.extract.deterministic). Measured:
        arm or leg names   90: 83 resolved,  7 abstained, 0 wrong
        other names        48: 30 resolved, 18 abstained, 0 wrong
        all               138: 113 resolved (81.9%), 25 abstained, 0 wrong
    CLAIM: the schedule's vocabulary does not need a model, and none is used
    for it. The 7 arm/leg abstentions are deliberate: musculocutaneous names
    an arm nerve (DC 8517) and a leg nerve (DC 8522); ilio-inguinal serves the
    groin; pronation and supination are also foot movements.

(b) LETTER PHRASINGS - fixtures/va_letter_phrasings.json
    70 names in the style letters use - clinical, eponymous, colloquial
    ("cubital tunnel syndrome", "De Quervain's tenosynovitis", "meralgia
    paresthetica"). LABELS ARE INTERNAL: assigned by this project from
    anatomy, written by the same project that maintains the lexicon, and not
    an external oracle. Four entries anatomy cannot settle (Raynaud's, tinea
    pedis, ganglion cyst, peripheral neuropathy) are marked ambiguous and not
    scored. Measured:
        arm or leg names   47: 12 resolved, 35 abstained, 0 wrong
        other names        19:  8 resolved, 11 abstained, 0 wrong
    What this shows, and no more: a lexicon limited to the schedule's own
    vocabulary abstains on most of these names, and never asserts a wrong
    group. An independent review added 26 common limb words and resolved 29
    of the 47 with still no wrong assertion - so a bigger lexicon closes much
    of the gap too. The line Recheck draws is a policy (the schedule's
    vocabulary is deterministic; names outside it go to the classifier, and
    everything the classifier says is gated), not a proof that a model is
    needed.

WHAT IS NOT CLAIMED. That a model classifies set (b) correctly: no live model
has been run, and every "AI" decision in the fixtures is a replayed fixture.
That no deterministic approach could cover set (b): a larger lexicon or a
clinical terminology would cover much of it. That set (b) is representative
of real letters: it was written for this project.

History: the review that measured the 26-word expansion also found schedule
words ("patellofemoral", "iliotibial", "Achilles", "Morton's") listed here as
names the lexicon should abstain on. They are the schedule's vocabulary, so
they were moved into the lexicon - and a test pinning their abstention was
removed, because it failed exactly when the lexicon followed its own policy.

THE SAFETY PROPERTIES, on both sets:
    - the lexicon never asserts a wrong group; it abstains instead, because a
      wrong "upper" or "lower" feeds 38 CFR 4.26 with no model call and no
      question;
    - the lexicon never says "none" for a name containing arm, leg or
      peripheral-nerve vocabulary (the same markers that veto a model's
      "none"), because "not an arm or leg" silently removes a 4.26 pair;
    - it asserts no group for a name anatomy does not settle.

The bounds below are tolerant, so adding a correct term does not break the
build; the zero-wrong assertions are exact.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from recheck.extract.deterministic import EXTREMITY_MARKERS, _classify_extremity
from recheck.schema import ClassificationBatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
SCHEDULE = json.loads((FIXTURES / "va_condition_names.json").read_text(encoding="utf-8"))
PHRASINGS_FILE = json.loads((FIXTURES / "va_letter_phrasings.json").read_text(encoding="utf-8"))
PHRASINGS = PHRASINGS_FILE["conditions"]

# The value the lexicon returns when it has no opinion. Distinct from "none",
# which is a positive finding that a condition is not of an arm or leg.
ABSTAIN = "unrecognised"
LIMB = ("upper", "lower")


def _scored(rows):
    return [r for r in rows if r["group"] in ("upper", "lower", "none")]


def _outcomes(rows, groups):
    """(resolved correctly, abstained, wrong) over rows whose label is in groups."""
    resolved = abstained = 0
    wrong = []
    for row in _scored(rows):
        if row["group"] not in groups:
            continue
        said = _classify_extremity(row["name"])
        if said == ABSTAIN:
            abstained += 1
        elif said == row["group"]:
            resolved += 1
        else:
            wrong.append((row["name"], row["group"], said))
    return resolved, abstained, wrong


# --------------------------------------------------------------------------
# The two datasets
# --------------------------------------------------------------------------

def test_schedule_set_is_the_externally_labelled_138():
    assert len(SCHEDULE) == 138
    assert {c["group"] for c in SCHEDULE} == {"upper", "lower", "none"}
    assert sum(c["group"] in LIMB for c in SCHEDULE) == 90


def test_letter_set_discloses_that_its_labels_are_internal():
    assert 50 <= len(PHRASINGS) <= 70
    assert "INTERNAL" in PHRASINGS_FILE["_comment"]
    for row in PHRASINGS:
        assert row["labels"] == "internal - authored for this project, not an external oracle"
        assert row["group"] in ("upper", "lower", "none", "ambiguous")
        assert row["basis"].strip(), f"{row['name']} has no anatomical basis"
    scored = _scored(PHRASINGS)
    limb_share = sum(r["group"] in LIMB for r in scored) / len(scored)
    assert 0.6 <= limb_share <= 0.8, "roughly two thirds arm or leg names, one third not"


# --------------------------------------------------------------------------
# (a) The schedule's vocabulary: deterministic, no model needed
# --------------------------------------------------------------------------

def test_a_lexicon_covers_the_schedules_arm_and_leg_vocabulary():
    """Measured 83 of 90. The model is not needed for the schedule's own names."""
    resolved, abstained, wrong = _outcomes(SCHEDULE, LIMB)
    assert wrong == []
    assert resolved >= 78, f"schedule arm/leg coverage fell to {resolved}/90"
    assert abstained <= 12


def test_a_lexicon_covers_most_of_the_schedule_overall():
    """Measured 113 of 138 (81.9%)."""
    resolved, abstained, wrong = _outcomes(SCHEDULE, ("upper", "lower", "none"))
    assert wrong == []
    assert 105 <= resolved <= 138
    assert resolved / len(SCHEDULE) >= 0.75


@pytest.mark.parametrize(
    "name,group",
    [
        ("Scapulohumeral articulation, ankylosis.", "upper"),
        ("Radius and ulna, nonunion.", "upper"),
        ("Median nerve, paralysis.", "upper"),
        ("Neuralgia, musculospiral nerve (radial).", "upper"),
        ("Upper radicular group, paralysis.", "upper"),
        ("Genu recurvatum.", "lower"),
        ("Os calcis or astragalus, malunion.", "lower"),
        ("Cartilage, semilunar, removal.", "lower"),
        ("Sciatic nerve, paralysis.", "lower"),
        ("External popliteal nerve (common peroneal), paralysis.", "lower"),
        ("Neuritis, anterior crural (femoral) nerve.", "lower"),
    ],
)
def test_a_schedule_names_the_old_gate_sent_to_the_model_are_now_deterministic(name, group):
    """The old gate pinned these as proof a model was needed. They were not."""
    assert _classify_extremity(name) == group


@pytest.mark.parametrize(
    "name",
    [
        "Musculocutaneous nerve, paralysis.",       # DC 8517 arm, DC 8522 leg
        "Supination and pronation, impairment.",    # also movements of the foot
        "Ilio-inguinal nerve, paralysis.",          # listed with the leg nerves; serves the groin
        "femoral hernia",                           # "femoral" is not always a leg
        "obturator hernia",
        "status post median sternotomy",            # "median" is not always the median nerve
        "radial keratotomy residuals",              # nor "radial" the radial nerve
        "circumflex artery stenosis",               # a coronary artery
        "semilunar valve disease",                  # the heart, not the knee
        "chalazion of the tarsal plate",            # the eyelid, not the foot
    ],
)
def test_a_words_that_name_more_than_one_thing_are_left_to_abstain(name):
    """Every term was admitted only if it names one extremity group wherever it
    appears. These are the ones left out, and why."""
    assert _classify_extremity(name) not in LIMB


# --------------------------------------------------------------------------
# (b) Letter phrasings: the lexicon abstains - the model's job
# --------------------------------------------------------------------------

def test_b_lexicon_never_misassigns_letter_style_arm_and_leg_names():
    """Measured 12 resolved, 35 abstained of 47. Only the zero-wrong property is
    pinned: coverage may rise as schedule vocabulary is added, and must."""
    resolved, abstained, wrong = _outcomes(PHRASINGS, LIMB)
    assert wrong == []
    assert abstained + resolved == 47


def test_b_non_extremity_names_are_never_misassigned():
    """Measured 8 of 19 resolved, 11 abstained. Abstaining here costs a model
    call, never a wrong pairing."""
    resolved, abstained, wrong = _outcomes(PHRASINGS, ("none",))
    assert wrong == []
    assert resolved + abstained == 19


def test_b_ambiguous_names_get_no_group_from_the_lexicon():
    for row in PHRASINGS:
        if row["group"] == "ambiguous":
            assert _classify_extremity(row["name"]) == ABSTAIN, row["name"]


@pytest.mark.parametrize(
    "name,group",
    [("Morton's neuroma", "lower"),            # DC 5279 "(Morton's disease)"
     ("patellofemoral pain syndrome", "lower"),  # DC 5257 "patellofemoral complex"
     ("Achilles tendinopathy", "lower"),         # "tendo achillis"
     ("iliotibial band syndrome", "lower")],     # 38 CFR 4.73 "iliotibial (Maissiat's) band"
)
def test_b_schedule_words_found_in_the_letter_set_are_deterministic(name, group):
    """Names a review showed are the schedule's own vocabulary: lexicon, not model."""
    assert _classify_extremity(name) == group


@pytest.mark.parametrize(
    "name",
    ["cervical strain with radiculopathy",
     "diabetes mellitus with peripheral neuropathy",
     "degenerative disc disease of the lumbar spine with radiculopathy"],
)
def test_a_non_extremity_hint_never_outweighs_nerve_vocabulary(name):
    """A hint ("cervical strain", "diabetes", "spine") next to nerve vocabulary
    means no opinion, never "none": "none" would silently drop a 4.26 pair.
    "cervical strain" was added as a hint and briefly did exactly that."""
    assert _classify_extremity(name) == ABSTAIN


def test_a_saphenous_vein_graft_is_not_a_leg_disability():
    """"saphenous" as a bare word called a heart bypass graft a leg disability."""
    assert _classify_extremity("coronary artery disease, status post bypass with saphenous vein graft") != "lower"
    assert _classify_extremity("Neuralgia, internal saphenous nerve") == "lower"


def test_b_lumbosacral_radiculopathy_is_not_called_none():
    """A real defect this set found. The hint was the bare word "lumbosacral",
    which called a leg disability rated under the sciatic nerve "not an arm or
    leg" and would have dropped its 4.26 pair without asking anyone."""
    assert _classify_extremity("lumbosacral radiculopathy") == ABSTAIN
    assert _classify_extremity("Lumbosacral strain") == "none"


# --------------------------------------------------------------------------
# Safety properties on both sets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rows", [SCHEDULE, PHRASINGS], ids=["schedule", "letter-phrasings"])
def test_lexicon_never_asserts_a_wrong_group(rows):
    """The property that makes the lexicon a safe fast path.

    An "upper" or "lower" from the lexicon feeds 4.26 pairing with no model
    call and no question, so a single false positive could apply the
    bilateral factor to the wrong conditions silently.
    """
    _, _, wrong = _outcomes(rows, ("upper", "lower", "none"))
    assert wrong == [], f"lexicon asserted a wrong group for: {wrong[:5]}"


@pytest.mark.parametrize("rows", [SCHEDULE, PHRASINGS], ids=["schedule", "letter-phrasings"])
def test_lexicon_never_says_none_for_a_name_with_extremity_vocabulary(rows):
    """The same markers that veto a model's "none" (recheck.classify) hold the
    lexicon to the same standard."""
    offenders = [r["name"] for r in rows
                 if _classify_extremity(r["name"]) == "none" and EXTREMITY_MARKERS.search(r["name"].lower())]
    assert offenders == []


def test_abstention_is_distinct_from_a_positive_none():
    """"I know this is not an arm or leg" and "I do not know this name" must
    not be the same value; conflating them once over-reported coverage and
    spent model calls on names the lexicon already knew."""
    assert _classify_extremity("tinnitus") == "none"
    assert _classify_extremity("cubital tunnel syndrome") == ABSTAIN
    assert _classify_extremity("right knee, limitation of flexion") == "lower"


# --------------------------------------------------------------------------
# The showcase letters use names from set (b), not the schedule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stem", ["07_clinical_terms", "08_clinical_terms_no_side"])
def test_showcase_letters_need_the_classifier_and_their_fixtures_are_schema_only(stem):
    from recheck.extract.deterministic import parse

    letter = parse((FIXTURES / "letters" / f"{stem}.txt").read_text(encoding="utf-8"))
    raw = json.loads((FIXTURES / "classifications" / f"{stem}.json").read_text(encoding="utf-8"))
    batch = ClassificationBatch.model_validate(raw)  # extra="forbid": no side, no rationale
    abstained = [r.condition for r in letter.ratings if r.extremity_group == ABSTAIN]
    assert sorted(c.condition for c in batch.classifications) == sorted(abstained)
    assert len(abstained) == 2
    for c in batch.classifications:
        assert c.extremity_group == "upper" and c.confidence >= 0.85
    for entry in raw["classifications"]:
        assert set(entry) == {"condition", "extremity_group", "confidence"}


def test_gate_report_runs():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "gate_report.py")],
                            capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert result.returncode == 0, result.stderr
    assert "(a) RATING-SCHEDULE VOCABULARY" in result.stdout
    assert "(b) LETTER PHRASINGS" in result.stdout
    assert "wrong-group assertions across both sets: 0" in result.stdout
