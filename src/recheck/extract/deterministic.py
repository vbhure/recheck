"""Deterministic extraction of ratings from VA decision letter text.

This is a genuine best-effort parser, not a strawman. It exists to answer one
question honestly: how far does deterministic parsing actually get? Anything
the model is later asked to do must be something this parser demonstrably
cannot do.

It uses no model, no network and no I/O beyond the string it is handed.

It FAILS CLOSED. A red team showed that reading most of a letter is worse
than reading none of it: a rating sentence in an unsupported wording, the
second stage of a staged rating, a wrapped last table row or a row after a
mid-list "Evidence" heading was silently skipped, and a confident figure was
reported on the partial list. So parse() accounts for every percentage in
the decision section - each must belong to a rating it read, a combined
statement, or a prior value named inside that rating's own sentence - and
refuses the letter otherwise. Refusing costs a manual review; a partial
reading costs a wrong rating.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

# "none" means recognised as a non-extremity condition. "unrecognised"
# means the lexicon has no opinion - a genuinely different state, and the
# only one that justifies a model call.
ExtremityGroup = Literal["upper", "lower", "none", "unrecognised"]

# 4.26(a): "arms" and "legs" mean the upper and lower extremities AS A WHOLE,
# so a wrist and a forearm belong to the same extremity group.
#
# The vocabulary is the rating schedule's OWN, checked against the eCFR text
# of 38 CFR Part 4: diagnostic-code titles, table headings, the named bones of
# the extremity tables and the named peripheral nerves. That vocabulary is
# finite, so deterministic code covers it and no model call is spent on it.
# The model is for what the schedule does not say - the clinical, eponymous
# and colloquial names letters actually use ("cubital tunnel syndrome").
#
# A word goes in only if it names ONE extremity group wherever it appears.
# Left out on purpose, each because it also names something else:
#   musculocutaneous  DC 8517 is an arm nerve, DC 8522 a leg nerve
#   median, radial    "median sternotomy", "radial keratotomy" (phrase only)
#   femoral           "femoral hernia" is not a leg disability (phrase only)
#   obturator         "obturator hernia" (phrase only)
#   circumflex        the circumflex coronary artery (phrase only)
#   semilunar         the heart's semilunar valves; the wrist's semilunar bone
#   tarsal            the tarsal plate of the eyelid
#   pronation, supination (DC 5213)  both are also movements of the foot
#   biceps, triceps   biceps femoris and triceps surae are leg muscles
#   ilio-inguinal     DC 8530 sits with the leg nerves, but the nerve serves
#                     the groin; abstaining costs at most one question
#   muscle group numerals (38 CFR 4.73)  matched as substrings, "muscle group
#                     i" is also the start of "muscle group ix" and "muscle
#                     group injury", and XIX-XXIII are the torso and neck
UPPER_TERMS = {
    # plain words for the arm
    "arm", "arms", "forearm", "forearms", "elbow", "elbows", "wrist", "wrists",
    "hand", "hands", "finger", "fingers", "thumb", "thumbs", "shoulder",
    "shoulders",
    # bones and joints of the shoulder and arm tables, DC 5200-5230; clavicle
    # and scapula are rated there with major/minor values (DC 5203)
    "scapulohumeral", "humerus", "humeral", "clavicle", "scapula", "radius",
    "ulna", "ulnar",
    # amputation of digits, DC 5126-5156, is measured at the metacarpal
    "metacarpal",
    # DC 8514 / 8614 / 8714 musculospiral (radial) nerve
    "musculospiral",
}
LOWER_TERMS = {
    # plain words for the leg
    "leg", "legs", "thigh", "thighs", "knee", "knees", "ankle", "ankles",
    "foot", "feet", "toe", "toes", "hip", "hips",
    # bones and joints of the hip, knee, ankle and foot tables, DC 5250-5284
    "femur", "tibia", "tibial", "fibula", "patella", "patellar",
    "astragalus", "astragalectomy", "subastragalar", "calcis", "metatarsal",
    "metatarsalgia", "hallux", "flatfoot", "forefoot", "acetabulum",
    # DC 5269 plantar fasciitis; DC 5262 "medial tibial stress syndrome
    # (MTSS), or shin splints"
    "plantar", "shin",
    # DC 8520-8525 and their neuritis (86xx) and neuralgia (87xx) forms:
    # sciatic, external / internal popliteal, superficial / deep peroneal,
    # anterior / posterior tibial. (DC 8527 internal saphenous is a phrase
    # below: a "saphenous vein graft" belongs to a heart bypass.)
    "sciatic", "popliteal", "peroneal",
    # DC 5257 "patellofemoral complex"; 38 CFR 4.73 "iliotibial (Maissiat's)
    # band"; the foot tables' "tendo achillis"; DC 5279 "Metatarsalgia,
    # anterior (Morton's disease)"
    "patellofemoral", "iliotibial", "achilles", "achillis", "morton",
}
UPPER_PHRASES = (
    "upper extremity", "upper extremities",
    # DC 8510-8513 upper, middle, lower and all radicular groups - the
    # schedule has radicular groups only for the arm
    "radicular group",
    # DC 8515, 8518, 8519 (and 86xx / 87xx)
    "median nerve", "circumflex nerve", "long thoracic nerve",
)
LOWER_PHRASES = (
    "lower extremity", "lower extremities",
    # DC 5258 / 5259 "Cartilage, semilunar" (the knee's menisci)
    "cartilage, semilunar", "semilunar cartilage",
    # DC 5263; DC 5278 "Claw foot (pes cavus)"; "pes planus" is 4.26(a)'s
    # own example of a disability of the foot
    "genu recurvatum", "pes cavus", "pes planus",
    # DC 8526 anterior crural (femoral) nerve; DC 8527 internal saphenous
    # nerve; DC 8528 obturator nerve
    "anterior crural", "internal saphenous", "saphenous nerve", "obturator nerve",
)

# Conditions that are never extremity disabilities - but only when nothing in
# the name points at an arm or a leg. "Radiculopathy, right lower extremity,
# associated with lumbosacral strain" is a leg disability; checking these
# hints first used to call it "none" and silently drop a 4.26 pair.
#
# Hints are substrings, so each names a condition or an organ, not a region.
# The bare "lumbosacral" used to be one: it called "lumbosacral radiculopathy"
# - a leg disability rated under the sciatic nerve - "none". The hint is now
# the DC 5237 title, "lumbosacral or cervical strain".
NON_EXTREMITY_HINTS = (
    "tinnitus", "post-traumatic stress", "posttraumatic stress", "ptsd",
    "hearing", "migraine", "lumbosacral strain", "cervical strain", "spine",
    "sleep apnea", "diabetes",
    # ear and other sense organs, 38 CFR 4.87 and 4.87a (DC 6200-6276)
    "otitis", "otosclerosis", "vestibular", "meniere", "auricle", "tympanic",
    "smell", "taste",
    # nose, sinuses and lungs, 38 CFR 4.97 (DC 6510-6847)
    "sinusitis", "rhinitis", "asthma", "bronchitis", "lung", "pulmonary",
    "pneumonitis", "pneumoconiosis", "asbestosis", "pleural",
    "kyphoscoliosis", "pectus", "chest wall",
    # mental disorders, 38 CFR 4.130 (DC 9201-9440)
    "schizophreni", "obsessive compulsive", "depressive", "anxiety", "bipolar",
)

# Vocabulary that makes "not an arm or leg" implausible. The lexicon never
# says "none" for a name containing it, and a model's "none" is vetoed on it
# (recheck.classify). "Cervical strain with radiculopathy" and "diabetes
# mellitus with peripheral neuropathy" carry a non-extremity hint AND nerve
# vocabulary; calling them "none" would silently drop a 4.26 pair.
# Whole words for short body parts (so "pharmacological" is not an "arm"),
# stems for the anatomical and neurological vocabulary.
EXTREMITY_MARKERS = re.compile(
    r"\b(?:arms?|elbows?|forearms?|wrists?|hands?|fingers?|thumbs?|shoulders?|"
    r"legs?|thighs?|knees?|ankles?|foot|feet|toes?|hips?|heels?)\b"
    r"|nerve|neuritis|neuralgia|paralysis|radicul|neuropath|extremit|carpal|tarsal|"
    r"metacarp|metatars|phalan|hallux|patell|tibia|fibula|femor|humer|radius|ulna|"
    r"calcane|achilles|plantar|amputat|muscle group"
)

# Clauses that name a DIFFERENT condition the rated one is linked to. The
# rated condition's own anatomy and side come before them: in "Left knee
# strain, secondary to right knee strain" the rated knee is the left one.
# The list is closed, so a link worded another way is not stripped; "Left hip
# strain, caused by right knee disability" read as both sides and entered the
# 4.26 factor. recheck.classify reads two sides with no wording naming both
# as unknown, and _lexical_group abstains on a limb beside a non-extremity
# condition, for the wordings still missing here.
_LINKED_CLAUSE = re.compile(
    r"[,;]?\s*\(?\b(?:secondary to|associated with|due to|claimed as|incident to|aggravated by|"
    r"as a result of|resulting from|caused by|related to|because of|attributable to|in connection with|"
    r"compensating for|following|worsened by|exacerbated by)\b.*$",
    re.I,
)
# Handedness is not a side: "(right hand dominant)", "(major)", "right-handed",
# "right hand-dominant", "right handed", "right hand is dominant".
_HANDEDNESS = re.compile(
    r"\((?:[^)]*\b(?:dominant|major|minor|handed)\b[^)]*)\)"
    r"|\b(?:right|left)[- ]hand(?:ed)?(?:[- ]|\s+is\s+)dominant\b|\b(?:right|left)[- ]handed\b",
    re.I,
)


def primary_clause(condition: str) -> str:
    """The part of a condition name that describes the rated condition itself."""
    text = _HANDEDNESS.sub(" ", condition)
    text = _LINKED_CLAUSE.sub("", text)
    return " ".join(text.split()).strip(" ,;")


# A link in wording _LINKED_CLAUSE does not list still usually ends in one of
# these words ("Scar, abdomen, onset after right knee injury"). Nothing is
# stripped at them - "Scar from shell fragment wound, right thigh" rates the
# thigh - but a fact that only the text after one supplies is not taken from
# the name: read whole, that scar was a right leg disability and entered the
# 4.26 factor. See _classify_extremity and recheck.classify.derive_laterality.
_OPEN_LINK = re.compile(r"\b(?:by|after|since|subsequent to|from)\b", re.I)


def before_open_link(primary: str) -> str | None:
    """The text of a primary clause before a word that may open a linked condition, if it has one."""
    match = _OPEN_LINK.search(primary)
    return primary[: match.start()] if match else None


def linked_clause(condition: str) -> str:
    """Whatever primary_clause removed (the linked condition), if anything."""
    text = " ".join(_HANDEDNESS.sub(" ", condition).split())
    match = _LINKED_CLAUSE.search(text)
    return match.group(0) if match else ""

LEXICON_SIZE = (
    len(UPPER_TERMS) + len(LOWER_TERMS) + len(UPPER_PHRASES) + len(LOWER_PHRASES)
    + len(NON_EXTREMITY_HINTS)
)

# Percentages after these headings describe rating CRITERIA, not the
# veteran's assigned evaluations. Matched only as a heading on its own line:
# an unanchored match cut the ratings list at "with x-ray evidence of
# arthritis" inside a rating line and silently dropped every later rating.
#
# What follows a heading is not simply ignored, though. A letter with a
# stand-alone "Evidence" line in the middle of its table had its later rows
# cut, the numbering stayed contiguous, and a figure was computed on the
# rows above the heading. parse() therefore refuses a letter whose text after
# the heading holds a rating it did not read above it; a restatement of one
# it did read (REASONS FOR DECISION often repeats the decision) is fine.
STOP_HEADINGS = ("REASONS FOR DECISION", "EVIDENCE", "REFERENCES")
_STOP_HEADING = re.compile(
    r"^[ \t]*(?:" + "|".join(STOP_HEADINGS) + r")[ \t]*:?[ \t]*$", re.I | re.M
)

# One numbered row, on one line. Whitespace is [^\S\n] (any space but a line
# break, so a pdftotext form feed still starts a row), never \s: "^\s*" let
# every blank line start a match that ran to the end of the blank run, and
# 20,000 blank lines took ten seconds. The name must end in a character that
# is neither a dot nor a space (optionally followed by one dot, "etc."), so
# the leader cannot be retried from every dot or space of a long run: a row
# with 20,000 leader dots and no percentage took a minute.
_ROW_TAIL = (
    r"(?P<condition>[^\n]*?[^.\s]\.?)[^\S\n]*\.{3,}[^\S\n]*(?P<pct>\d{1,3})[^\S\n]*%"
)
_TABULAR = re.compile(r"^[^\S\n]*(?P<row>\d+)\.[^\S\n]*" + _ROW_TAIL, re.MULTILINE)
# Every line that starts like a numbered row. The contiguity check compares
# row numbers with 1..n, so it cannot see a missing LAST row: a final row
# wrapped onto a second line was dropped and the rating computed without it.
_NUMBERED_LINE = re.compile(r"^[^\S\n]*\d+\.(?!\d)", re.M)
# A line after a stop heading that is laid out like a rating row: leader dots
# and a percentage. A REASONS section may restate the table; anything else
# laid out as a row (an unnumbered or wrapped row, a row numbered on from the
# table) is a row the heading cut off.
_LEADER = re.compile(r"\.[^\S\n]?\.[^\S\n]?\.")
_TAIL_ROW = re.compile(r"[^\S\n]*(?:(?P<row>\d+)[.)][^\S\n]*)?" + _ROW_TAIL + r"[^\S\n]*$")

# Prose forms, most specific first. "increased to" must win over "currently
# evaluated as" in the same sentence, or historical values get captured.
#
# Only the anchor is a pattern; the condition is the text between the last
# period before it and the anchor. The earlier patterns captured it with a
# leading "(?P<condition>[^.]*?)", which the regex engine retries from every
# start position - quadratic in sentence length: a 30 KB letter with no
# period took a minute to parse, and a megabyte would take hours.
_PROSE_ANCHORS = (
    re.compile(r"\bis increased to\s+(?P<pct>\d{1,3})\s+percent", re.I),
    re.compile(r"\bis continued as\s+(?P<pct>\d{1,3})\s+percent", re.I),
    re.compile(r"\bwith an evaluation of\s+(?P<pct>\d{1,3})\s+percent", re.I),
    # A 0 percent evaluation written without a percentage has no mark for the
    # completeness check to count, so it was left out of the list silently.
    re.compile(r"\bwith a\s+(?P<pct>noncompensable)\s+evaluation", re.I),
    re.compile(r"\bis continued as\s+(?P<pct>noncompensable)\b", re.I),
)

# Words are separated by \s+, not a space: letters are hard-wrapped, and "Your
# combined evaluation for\ncompensation is 70 percent." was not found at all.
_COMBINED = [
    re.compile(r"COMBINED\s+EVALUATION\s+FOR\s+COMPENSATION\s*:?\s*(\d{1,3})\s*%", re.I),
    re.compile(r"combined\s+evaluation\s+for\s+compensation\s+is\s+(\d{1,3})\s+percent", re.I),
]
# "Your previous combined evaluation for compensation is 30 percent" is
# history, not the statement under review. Taking the first match anywhere
# reported it as the stated value and printed a false discrepancy.
_HISTORICAL_QUALIFIER = re.compile(r"\b(?:previous|prior|former)\s+$", re.I)

# Every percent-like mark, whatever number (or none) stands before it. The
# completeness check first looked for whole tokens - digits, then "%" or
# "percent" - and a value it did not recognise as a token was not counted,
# so it was dropped silently: "A 10-percent evaluation", "ten (10) percent",
# "ten-percent", "10 pct.", the Arabic percent sign. Counting the MARK
# instead means a percentage can only be accounted for or refused, however
# its number is written. Substrings count too ("percentage"): in the decision
# section a refusal costs a manual review, a missed value a wrong figure.
_PERCENT_MARK = re.compile(
    r"[%٪‰‱⁒]"
    r"|(?<![A-Za-z])per[-‐-―\s]*cent"
    r"|(?<![A-Za-z])pct(?![A-Za-z])",
    re.I,
)
# "10 percent each" and "10 and 20 percent, respectively" give one value to
# several conditions, or several values to several conditions. The anchors
# read one condition and one value, so the letter was read as ONE rating.
_DISTRIBUTIVE = re.compile(r"\b(?:each|respectively)\b", re.I)

# A prior value named inside the sentence that assigns the current one:
# "Evaluation of X, currently evaluated as 30 percent disabling, is increased
# to 60 percent" or "Your claim for X, previously 10 percent, is increased to
# 20 percent". It is accounted for only when it sits directly in front of the
# rating parse() read (_AFTER_PRIOR). On its own, "X, which is currently 10
# percent disabling, is continued" is how a CONTINUED rating is worded, and
# "PTSD, currently evaluated as 50 percent disabling, and tinnitus is
# increased to 10 percent" names a second rating; treating either as
# history would drop an evaluation.
_PRIOR_VALUE = re.compile(
    r",\s*(?:currently evaluated as|which is currently|previously evaluated as|previously rated as|previously)"
    r"\s+(?P<pct>\d{1,3})\s+percent(?:\s+disabling)?,?",
    re.I,
)
_AFTER_PRIOR = re.compile(r"\s*(?:(?:is|has been)\s+(?:granted|continued|increased|assigned)\s*)?")

# A line that cannot be the middle of a wrapped rating sentence: blank, a
# heading with no lower-case letter ("DECISION", "DEPARTMENT OF VETERANS
# AFFAIRS"), or a "Label: value" line ("Name: ...", "File Number: ..."). Only
# periods used to end sentences, so a letterhead with none ran straight into
# the first rating, and "Name: Jane Q Veteran File Number: 123-45-6789 ...
# DECISION Left cubital tunnel syndrome" became a condition name - stored,
# printed, and sent to the model provider.
_LABEL_LINE = re.compile(r"^[ \t]*[A-Za-z][A-Za-z .'/#-]{0,40}:(?:\s|$)")
# A line with no lower-case letter is not always a heading, though. Breaking
# at every one split "limitation of flexion of the left knee\n(DC 5260)\nis
# granted with an evaluation of 30 percent" after the code, and the condition
# became "is granted": a figure computed on a rating with no anatomy. Such a
# line is a heading only when it carries a heading word; otherwise it joins
# the sentence it sits in when that is visible (a parenthetical after an
# unfinished line, or a next line that starts in lower case), and in every
# other case the text after it is marked as possibly starting mid-sentence
# (see _blocks), which parse() refuses rather than guesses at.
_HEADING_WORD = re.compile(
    r"\b(?:DECISIONS?|EVIDENCE|REASONS?|REFERENCES?|DEPARTMENT|AFFAIRS|ISSUES?|SUMMARY|NOTIFICATION|"
    r"INTRODUCTION|CONCLUSIONS?|FINDINGS?|BACKGROUND|ANALYSIS)\b|\*\*\*"
)
_PARENTHETICAL_LINE = re.compile(
    r"^\s*(?:\([^()\n]*\)|(?:DC|diagnostic codes?)\s*\d{4}(?:\s*[-/]\s*\d{4})*)\s*[.,;]?\s*$", re.I
)
_SENTENCE_END = re.compile(r"[.;!?][)\"'\]]*$")
# Words a wrapped sentence can end a line on and carry on after: "Service
# connection for" / "Left knee strain is granted". Matched in lower case only,
# on purpose: wrapped prose ends a line on a lower-case "for" or "is", while an
# address ends on a state code or a unit letter ("Gary, IN", "Portland, OR",
# "Apt A"), which read as "in", "or" and "a" and joined a name and address
# block to the rating below it.
_JOINING_TAIL = re.compile(
    r"\b(?:a|an|the|of|for|to|and|or|with|in|on|at|by|from|as|is|was|are|were|be|been|has|have|under|"
    r"including|include|includes)\s*$"
)

# A condition name longer than this is text the sentence split failed to
# separate. It is also the model schema's limit (recheck.schema).
MAX_CONDITION_CHARS = 200
# Diagnostic codes are the only long numbers a condition name legitimately
# carries: "Limitation of flexion, right knee (DC 5260)", "DC 5010-5260".
_CODE_REFERENCE = re.compile(r"\b(?:DCs?|diagnostic codes?)\s*\d{4}(?:\s*(?:[-/,]|and)\s*\d{4})*", re.I)
# No condition name starts with a verb, a conjunction or a preposition; a
# captured one that does is the tail of a sentence split from its start.
_NOT_A_NAME_START = re.compile(
    r"(?:is|are|was|were|has|have|had|been|being|granted|continued|increased|assigned|and|or|nor|but|of|"
    r"with|without|to|for|in|on|at|by|from|as)\b(?![-'])",
    re.I,
)


@dataclass
class ExtractedRating:
    """One assigned evaluation, with where in the letter it was found.

    Which side a condition is on is not recorded here. It is derived from the
    condition text in exactly one place, recheck.classify.derive_laterality.
    """

    condition: str
    percent: int
    extremity_group: ExtremityGroup
    source_line: str
    source_line_number: int


@dataclass
class Extraction:
    ratings: list[ExtractedRating] = field(default_factory=list)
    stated_combined: int | None = None
    unparsed_reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.ratings) and self.stated_combined is not None


def _normalise(text: str) -> str:
    """Fold the text to what a reader sees, before anything matches on it.

    The lexicon, the "none" veto and the side reader match ASCII words. pypdf
    commonly extracts "finger" with the "fi" ligature, and a soft hyphen or a
    zero-width space inside "knee" is invisible on the page. Each made the
    word unrecognisable, so a model's "none" for "Tenosynovitis, right
    finger" passed the veto and dropped a 4.26 pair while the report printed
    the word "finger". Compatibility decomposition folds ligatures, full-width
    letters and non-breaking spaces; combining marks and invisible format
    characters are dropped, so "HİP" reads "HIP" and "kn<ZWSP>ee" reads "knee".
    Enclosing marks (a keycap or circle drawn around a letter) go the same way.

    The modifier-letter apostrophes (U+02BB-U+02BD) are letters to Unicode,
    so "Quervainʼs" was a word mixing two alphabets and refused; a reader
    sees an apostrophe, so they become one.
    """
    if text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text.translate(_APOSTROPHES))
    kept = "".join(ch for ch in decomposed if unicodedata.category(ch) not in ("Mn", "Me", "Cf"))
    return unicodedata.normalize("NFC", kept)


_APOSTROPHES = str.maketrans({"ʻ": "'", "ʼ": "'", "ʽ": "'"})
# Renders as a space but is not whitespace to Python or to the word matches:
# "kn<U+2800>ee" is neither the word "knee" nor two words to anything here.
_BLANK_LOOKALIKES = ("⠀",)


def _script(char: str) -> str:
    return "LATIN" if char.isascii() else unicodedata.name(char, "UNKNOWN").split(" ", 1)[0]


def _lookalike_problem(text: str) -> str | None:
    """Why normalised decision text is still ambiguous to read, if it is.

    Normalisation cannot fold a Cyrillic "к" into a Latin "k": they are
    different letters that look the same. "кnee" defeats every word match
    while a reviewer reads "knee", and "pеrcent" would hide a percentage from
    the completeness check. English decision letters have no reason to mix
    alphabets inside a word or to use another script's digits, so either is
    refused rather than read.
    """
    if text.isascii():
        return None
    if any(blank in text for blank in _BLANK_LOOKALIKES):
        return ("the decision section contains a character that looks like a space but is not one (braille "
                "blank), so what Recheck matches may not be what a reader sees")
    if any(ch.isdigit() and not ch.isascii() for ch in text):
        return "the decision section contains digits from a non-Latin script, which can read as a different number"
    for word in re.findall(r"[^\W\d_]+", text):
        if not word.isascii() and len({_script(ch) for ch in word}) > 1:
            return ("a word in the decision section mixes letters from different alphabets (look-alike characters), "
                    "so what Recheck matches may not be what a reader sees")
    return None


def _name_problem(condition: str) -> str | None:
    """Why a captured condition name cannot be what the letter rates, if it cannot."""
    if not condition or not re.search(r"[^\W\d_]{2,}", _CODE_REFERENCE.sub(" ", condition)):
        return "a rating statement names no condition"
    if _NOT_A_NAME_START.match(condition):
        # "is granted", "and instability": the statement was split from the
        # line that holds its condition, and what is left has no anatomy.
        return ("a rating statement's condition name starts mid-sentence, so the rest of the name is on a "
                "line Recheck did not join to it")
    if len(condition) > MAX_CONDITION_CHARS:
        return (f"a condition name runs to {len(condition)} characters, so the rating statement "
                f"could not be separated from the text around it")
    if ":" in condition:
        return "a condition name contains a 'label:' - letterhead or other letter text ran into it"
    if re.search(r"%|\bper[-\s]*cent", condition, re.I):
        return "a condition name contains another percentage, so the statement holds more than one value"
    if re.search(r"\d{3,}", _CODE_REFERENCE.sub(" ", condition)):
        return ("a condition name contains a long number that is not a diagnostic code (a date, a file "
                "number or other letter text)")
    if any(ch.isalpha() and _script(ch) != "LATIN" for ch in condition):
        return "a condition name contains letters outside the Latin alphabet (possible look-alike characters)"
    return None


def _split_scope(text: str) -> tuple[str, str, str | None]:
    """(decision section, the text after it, the heading that ended it)."""
    match = _STOP_HEADING.search(text)
    if not match:
        return text, "", None
    return text[: match.start()], text[match.start():], match.group(0).strip().rstrip(":").strip()


def _open_lead_in(run: str) -> bool:
    """Whether the unfinished sentence ending `run` has a lead-in whose condition name has not ended."""
    partial = re.split(r"[.;!?]\s", run)[-1]
    matches = list(_LEAD_IN.finditer(partial))
    return bool(matches) and not re.search(r"\b(?:is|are|was|were|has been|have been)\b|\d",
                                           partial[matches[0].end():])


def _blocks(text: str) -> list[tuple[str, bool]]:
    """Group lines into runs a wrapped sentence can span: (text, may start mid-sentence).

    Blank lines and "Label:" lines always end a run. A line with no lower-case
    letter ends one only when it cannot be the middle of a sentence; see
    _HEADING_WORD. When such a line does end a run although the line before
    it was unfinished, or when the line itself is unfinished and not a
    heading, the next run is flagged: its first words may be the end of a
    condition name that began above ("SERVICE CONNECTION FOR LEFT KNEE" /
    "STRAIN IS GRANTED ...") or a letterhead line may sit above it. parse()
    refuses a rating whose name would start at such a point.
    """
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []
    soft = False  # the run being built may begin mid-sentence
    unfinished = False  # the previous line did not end its sentence
    bridged = None  # (blocks so far, sentence offset) of the one break joined for an open lead-in
    lines = text.split("\n")
    for number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _LABEL_LINE.match(line):
            if current:
                blocks.append((" ".join(current), soft))
                current = []
            if stripped:
                blocks.append((line, False))
            unfinished = False
            continue
        heading = parenthetical = continues = False
        if re.search(r"[a-z]", line):
            joins = True
            # An unfinished line that does not grammatically lead into the
            # next ("Dear Mr. Right," / "Knee strain is continued ...", or a
            # name and address block) is not the start of that sentence.
            # Joined, "Right" became the knee's side and the whole salutation
            # was sent to the model as part of the condition name. The next
            # run is marked as possibly starting mid-sentence, and parse()
            # refuses a rating there that has no lead-in.
            #
            # A name the lead-in has opened ("Service connection for right" /
            # "Achilles tendonitis is granted", "chronic" / "PTSD") may bridge
            # ONE such break. A salutation or address sits on a line of its
            # own, so reaching a rating from a subject line ("Your Claim for
            # Compensation" / "Dear Ms Right," / "Knee strain is continued")
            # takes a second break, and that one splits.
            if current and unfinished and re.match(r"[A-Z]", stripped) and not _JOINING_TAIL.search(current[-1]):
                run = " ".join(current)
                sentence_start = len(run) - len(re.split(r"[.;!?]\s", run)[-1])
                if _open_lead_in(run) and bridged != (len(blocks), sentence_start):
                    bridged = (len(blocks), sentence_start)
                else:
                    blocks.append((run, soft))
                    current = []
        else:
            heading = bool(_HEADING_WORD.search(line))
            parenthetical = bool(_PARENTHETICAL_LINE.match(line))
            following = lines[number + 1].lstrip() if number + 1 < len(lines) else ""
            continues = bool(re.match(r"[a-z]", following))
            joins = not heading and (continues or (parenthetical and unfinished and bool(current)))
        # A heading word between an unfinished line and one that continues in
        # lower case ("Service connection for" / "LEFT KNEE - SEE EVIDENCE" /
        # "strain is granted") may be inside the sentence after all.
        mid_sentence_heading = heading and unfinished and continues
        if joins:
            if not current:
                soft = unfinished
            current.append(line)
        else:
            if current:
                blocks.append((" ".join(current), soft))
                current = []
            blocks.append((line, unfinished and not heading))
        # A parenthetical that stands alone qualifies the text above it; it
        # does not start a condition name below it.
        unfinished = ((not heading or mid_sentence_heading) and not (parenthetical and not joins)
                      and not _SENTENCE_END.search(stripped))
    if current:
        blocks.append((" ".join(current), soft))
    return blocks


def _sentences(text: str) -> list[tuple[str, bool]]:
    """Split into whitespace-normalised sentences: (sentence, may start mid-sentence).

    VA letters are hard-wrapped, so a clause can straddle a newline. Patterns
    must see "with an evaluation of" as contiguous. Splitting on sentence
    boundaries - and never joining across a blank line, a heading or a
    "Label:" line - stops a condition capture running backwards into the
    letterhead. Only the first sentence of a flagged run carries the flag.
    """
    sentences: list[tuple[str, bool]] = []
    for block, soft in _blocks(text):
        flat = re.sub(r"\s+", " ", block)
        # Protect the period in a middle initial ("Name: R. SYNTHETIC") and in
        # abbreviations like "No." so they do not end a sentence.
        flat = re.sub(r"\b([A-Z])\.\s", r"\1<DOT> ", flat)
        parts = [p.replace("<DOT>", ".").strip() for p in re.split(r"(?<=[.;])\s+", flat)]
        sentences.extend((part, soft and index == 0) for index, part in enumerate(p for p in parts if p))
    return sentences


def _marks(text: str, start: int, end: int) -> set[int]:
    """Offsets of the percent marks inside text[start:end] - one matched percentage."""
    return {mark.start() for mark in _PERCENT_MARK.finditer(text, start, end)}


def _prose_rating(sentence: str) -> tuple[str, int, set[int], int] | None:
    """(raw condition span, percent, offsets of the percent marks it accounts for, span start), or None."""
    for anchor in _PROSE_ANCHORS:
        match = anchor.search(sentence)
        if match is None:
            continue
        start = sentence.rfind(".", 0, match.start()) + 1
        span = sentence[start:match.start()]
        accounted = _marks(sentence, match.start("pct"), match.end())
        for prior in _PRIOR_VALUE.finditer(span):
            if _AFTER_PRIOR.fullmatch(span, prior.end()):
                accounted |= _marks(sentence, start + prior.start("pct"), start + prior.end())
        value = match.group("pct")
        return span, 0 if value.isalpha() else int(value), accounted, start
    return None  # most specific anchor wins for a given sentence


def _combined_marks(text: str) -> set[int]:
    return {offset for pattern in _COMBINED for m in pattern.finditer(text)
            for offset in _marks(text, m.start(1), m.end())}


def _unaccounted(text: str, accounted: set[int]) -> tuple[int, str] | None:
    """(offset, value as written) of the first percentage no reading accounts for, or None."""
    for mark in _PERCENT_MARK.finditer(text):
        if mark.start() not in accounted:
            # Show the value with its number ("10-percent", "ten (10) percent").
            before = re.search(r"(?:\d+(?:\.\d+)?|[A-Za-z]+)[^\S\n]*[-‐-―]?[^\S\n]*(?:\(\d+\)[^\S\n]*)?$",
                               text[max(0, mark.start() - 24):mark.start()])
            return mark.start(), ((before.group(0) if before else "") + mark.group(0)).strip()
    return None


_KEY_FILLER = {"a", "an", "the", "of", "your"}


def _key(condition: str) -> str:
    """The condition a name refers to, for telling a repeat from a new rating.

    "left knee strain" and then "the left knee strain", or a name with and
    without "(DC 5260)", were different keys, so one knee's staged rating
    counted twice. REASONS FOR DECISION restating "limitation of flexion of
    the right knee" for the row "Limitation of flexion, right knee (DC 5260)",
    or adding "(PTSD)", was refused as a rating the heading cut. Codes,
    articles, "of", punctuation, case, and an acronym that only repeats the
    initials of the words before it are therefore ignored. Word order and
    every other word are kept, so a different side or site stays different.
    """
    text = _CODE_REFERENCE.sub(" ", condition)
    # Only the last few words before an acronym can spell it. Passing all the
    # text before it re-split the whole name for every acronym: quadratic, and
    # 438 s for a 256 KB letter of "(AB)" repeats.
    text = _ACRONYM.sub(
        lambda m: " " if _abbreviates(m.group(1), text[max(0, m.start() - 600):m.start()]) else m.group(0), text)
    return " ".join(word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _KEY_FILLER)


_ACRONYM = re.compile(r"\(\s*([A-Za-z]{2,8})\s*\)")


def _codes(condition: str) -> frozenset[str]:
    """The diagnostic codes a condition name cites (empty when it cites none)."""
    return frozenset(re.findall(r"\d{4}", " ".join(m.group(0) for m in _CODE_REFERENCE.finditer(condition))))


def _abbreviates(acronym: str, before: str) -> bool:
    """Whether acronym is spelled by leading letters of the words just before it.

    "post-traumatic stress disorder (PTSD)", "gastroesophageal reflux
    disease (GERD)": each word gives the acronym a non-empty prefix of
    itself, in order, and the acronym ends at the last word.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", before)][-len(acronym):]
    letters = acronym.lower()

    def covers(i: int, j: int) -> bool:  # letters[i:] by the words words[j:]
        if i == len(letters):
            return j == len(words)
        if j == len(words):
            return False
        word = words[j]
        return any(covers(i + n, j + 1) for n in range(1, min(len(word), len(letters) - i) + 1)
                   if word[:n] == letters[i:i + n])

    return any(covers(0, start) for start in range(len(words)))


def _lexical_group(text: str) -> ExtremityGroup:
    low = text.lower()
    words = set(re.findall(r"[a-z]+", low))
    upper = any(p in low for p in UPPER_PHRASES) or bool(words & UPPER_TERMS)
    lower = any(p in low for p in LOWER_PHRASES) or bool(words & LOWER_TERMS)
    if upper and lower:
        return "unrecognised"  # "hand and foot" - not the lexicon's call
    if (upper or lower) and any(hint in low for hint in NON_EXTREMITY_HINTS):
        # "Major depressive disorder worsened by right knee injury": the limb
        # word may belong to a linked condition in wording _LINKED_CLAUSE does
        # not list, and calling a mental disorder a leg disability put it in
        # the 4.26 factor. Neither answer is the lexicon's call.
        return "unrecognised"
    if upper:
        return "upper"
    if lower:
        return "lower"
    if any(hint in low for hint in NON_EXTREMITY_HINTS):
        return "unrecognised" if EXTREMITY_MARKERS.search(low) else "none"
    return "unrecognised"


def _classify_extremity(condition: str) -> ExtremityGroup:
    """The lexicon's verdict on a condition name: upper, lower, none, or no opinion.

    Anatomy in the rated condition itself decides first. When the rated
    condition names no anatomy but a linked clause does ("radiculopathy
    associated with lumbar spine", "strain secondary to a knee injury"), the
    lexicon has no opinion - it does not guess from the other condition.
    """
    text = primary_clause(condition)
    primary = _lexical_group(text)
    before = before_open_link(text)
    if before is not None and _lexical_group(before) != primary:
        return "unrecognised"  # the group comes from text that may name another condition
    if primary != "unrecognised":
        return primary
    if linked_clause(condition):
        return "unrecognised"
    return primary


# "Your claim for X ... is increased to" is a lead-in like "Service connection
# for X"; without it the condition read "Your claim for right De Quervain's
# tenosynovitis, previously 10 percent".
_LEAD_IN = re.compile(r"(?:service connection for|evaluation of|claim for)\s+", re.I)
_TRAILING_VERB = re.compile(r"\s+(?:is|has been) (?:granted|continued|increased|assigned)\b.*$", re.I)


def _clean(condition: str) -> str:
    """Reduce a matched span to just the condition name.

    Three failure modes this guards against, all real:
      - the capture running backwards into the letterhead, so the condition
        reads "SYNTHETIC File Number: 00-000-002 ...". Anchoring to the LAST
        lead-in phrase trims it when there is one; _sentences and
        _name_problem cover the letters where there is not.
      - trailing verb phrases ("... is granted") ending up in the condition
        name, which pollutes the evidence shown to a reviewer.
      - a prior value ("currently evaluated as 30 percent disabling",
        "previously 10 percent") left inside the name.
    """
    condition = re.sub(r"\s+", " ", condition).strip()
    condition = _PRIOR_VALUE.sub("", condition)
    matches = list(_LEAD_IN.finditer(condition))
    if matches:
        condition = condition[matches[-1].end():]
    condition = _TRAILING_VERB.sub("", condition)
    return condition.strip(" .,—-")


class _LineIndex:
    """Map a phrase found in whitespace-normalised text back to letter lines.

    Letters are hard-wrapped, so a condition can start on one line and end
    on the next. Searching raw lines for the phrase misses it, and the
    evidence then reads "none recorded" - or, worse, points at the wrong
    line. Instead the text is flattened once, keeping for every character the
    offset it came from.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        # Split at "\n" only, as line_of counts: splitlines() also breaks at a
        # form feed (a pdftotext page break), \v, \x1c-\x1e, \x85, U+2028 and
        # U+2029, and every later rating's source line was another line's text.
        self.lines = text.split("\n")
        flat: list[str] = []
        origin: list[int] = []
        newlines: list[int] = []
        previous_space = False
        for offset, char in enumerate(text):
            if char == "\n":
                newlines.append(offset)
            if char.isspace():
                if previous_space:
                    continue
                flat.append(" ")
                previous_space = True
            else:
                flat.append(char.lower())
                previous_space = False
            origin.append(offset)
        self.flat = "".join(flat)
        self.origin = origin
        # Line numbers by bisection. Counting newlines from the top for every
        # rating made a long letter quadratic.
        self.newlines = newlines
        self.cursor = 0

    def line_of(self, offset: int) -> int:
        return bisect.bisect_left(self.newlines, offset) + 1

    def locate(self, phrase: str) -> tuple[str, int]:
        """(source text, first line number) for the next occurrence of phrase."""
        needle = " ".join(phrase.lower().split())
        if not needle:
            return "", 0
        # Search forward from the previous hit so repeated wording ("limitation
        # of flexion of the ...") maps to successive lines, then from the top.
        for probe in (needle, needle[:32]):
            at = self.flat.find(probe, self.cursor)
            if at == -1:
                at = self.flat.find(probe)
            if at != -1:
                end = min(at + len(probe), len(self.origin)) - 1
                first, last = self.line_of(self.origin[at]), self.line_of(self.origin[end])
                self.cursor = at + len(probe)
                span = " ".join(line.strip() for line in self.lines[first - 1:last])
                return span, first
        return phrase.strip()[:120], 0


_PARTIAL = "refusing rather than computing on a partial list"


def _refuse(result: Extraction, reason: str) -> Extraction:
    result.ratings = []
    result.unparsed_reason = reason
    return result


def _where(index: _LineIndex, sentence: str) -> str:
    _, line = index.locate(sentence)
    return f"letter line {line}" if line else "the decision section"


def _only_restatements(tail: str, read: dict[str, list[tuple[int, int | None, frozenset[str]]]]) -> bool:
    """Whether every rating-shaped statement after a stop heading repeats one read above it.

    A repeat matches on the condition key (see _key) and the percentage. A
    row must also carry the number of the row it repeats: "3. Scar, left
    knee ... 10%" after a table whose row 2 reads the same is row 3 of a
    list the heading cut, not a restatement of row 2. Any other line laid
    out as a row - leader dots and a percentage - is not a restatement
    either: an unnumbered row, a wrapped row's second line, a row with two
    stages.
    """
    def repeats(condition: str, percent: int, row: int | None = None, is_row: bool = False) -> bool:
        # A row read from prose has no number to match; a sentence repeats
        # a row whatever its number. _key ignores diagnostic codes, so a
        # statement under a DIFFERENT code ("scar of the left knee (DC 7805)"
        # after the row "Scar, left knee (DC 7804)") is a second evaluation
        # the heading cut, not a restatement: where both name codes, one must
        # cite all the codes of the other. A hyphenated code (DC 5003-5260)
        # names one evaluation, and a restatement may cite either part;
        # disjoint or partly overlapping codes are a second evaluation.
        codes = _codes(condition)
        return any(p == percent and (not is_row or r is None or r == row)
                   and (not c or not codes or c <= codes or codes <= c)
                   for p, r, c in read.get(_key(condition), []))

    for line in tail.split("\n"):
        if not (_LEADER.search(line) and _PERCENT_MARK.search(line)):
            continue
        match = _TAIL_ROW.match(line)
        if match is None or len(_PERCENT_MARK.findall(line)) != 1:
            return False
        row = int(match.group("row")) if match.group("row") else None
        if not repeats(_clean(match.group("condition")), int(match.group("pct")), row, is_row=True):
            return False
    for sentence, _ in _sentences(tail):
        found = _prose_rating(sentence)
        if found is not None and not repeats(_clean(found[0]), found[1]):
            return False
    return True


def parse(text: str) -> Extraction:
    """Extract ratings and the stated combined evaluation from letter text.

    Refuses (ratings empty, unparsed_reason set) whenever the letter cannot be
    read completely and unambiguously. See the module docstring.
    """
    result = Extraction()
    text = _normalise(text)
    scope, tail, heading = _split_scope(text)
    # The decision section only: its words become facts. Evidence text after
    # a stop heading may fairly write micrograms with a Greek mu beside a Latin g.
    problem = _lookalike_problem(scope)
    if problem:
        return _refuse(result, f"{problem}; refusing rather than guessing")
    index = _LineIndex(text)
    # What was read, by condition key: (percent, row number, or None for prose).
    read: dict[str, list[tuple[int, int | None, frozenset[str]]]] = {}
    sentences = _sentences(scope)
    for sentence, _ in sentences:
        if _DISTRIBUTIVE.search(sentence) and _PERCENT_MARK.search(sentence):
            return _refuse(result, (
                f"a statement in {_where(index, sentence)} shares percentages between conditions ('each' or "
                f"'respectively'), which Recheck would read as one rating; {_PARTIAL}"
            ))

    # Every numbered row is its own evaluation, even when two rows read
    # identically (two separately rated scars): tabular rows are never deduped.
    rows: list[int] = []
    accounted: set[int] = set()
    for match in _TABULAR.finditer(scope):
        rows.append(int(match.group("row")))
        accounted |= _marks(scope, match.start("pct"), match.end())
        condition = _clean(match.group("condition"))
        number = index.line_of(match.start("condition"))
        problem = _name_problem(condition)
        if problem:
            return _refuse(result, f"{problem} (letter line {number}); {_PARTIAL}")
        percent = int(match.group("pct"))
        source = index.lines[number - 1].strip() if number <= len(index.lines) else condition
        read.setdefault(_key(condition), []).append((percent, rows[-1], _codes(condition)))
        result.ratings.append(
            ExtractedRating(condition, percent, _classify_extremity(condition), source, number)
        )

    if rows:
        # A numbered list with a gap means a row did not match the pattern (a
        # wrapped line, a missing leader). Computing on the rows that did match
        # would silently drop an evaluation, so the letter is refused instead.
        if rows != list(range(1, len(rows) + 1)):
            missing = sorted(set(range(1, max(rows) + 1)) - set(rows))
            return _refuse(result, (
                f"numbered evaluations are not contiguous (rows read: {rows}; missing: {missing or 'order'}); "
                f"{_PARTIAL}"
            ))
        accounted |= _combined_marks(scope)
        stray = _unaccounted(scope, accounted)
        if stray is not None:
            # A last row wrapped onto a second line, a second stage in a
            # row ("20% from ...; 30% from ..."), a stray value.
            offset, written = stray
            return _refuse(result, (
                f"a percentage on letter line {index.line_of(offset)} ({written}) "
                f"is not part of any rating row Recheck could read; {_PARTIAL}"
            ))
        numbered = len(_NUMBERED_LINE.findall(scope))
        if numbered != len(rows):
            return _refuse(result, (
                f"{numbered} numbered lines but {len(rows)} rating rows matched the rating pattern; {_PARTIAL}"
            ))
    else:
        # Sentence-scoped matching. Letters are hard-wrapped, so anchors are
        # applied to whitespace-normalised sentences rather than raw text -
        # otherwise a line break inside "with an\nevaluation of" defeats the
        # match, and an unanchored capture bleeds backwards through the header.
        pending: list[tuple[str, int]] = []
        for sentence, may_start_mid_sentence in sentences:
            accounted = _combined_marks(sentence)
            found = _prose_rating(sentence)
            if found is not None:
                span, percent, marks, start = found
                accounted |= marks
            stray = _unaccounted(sentence, accounted)
            if stray is not None:
                # A rating in a wording Recheck does not read ("An
                # evaluation of 30 percent is assigned from ...", often
                # the later stage of a staged rating) was skipped here,
                # and the figure computed without it.
                return _refuse(result, (
                    f"a percentage in {_where(index, sentence)} ({stray[1]}) is not part of any rating "
                    f"statement Recheck could read; {_PARTIAL}"
                ))
            if found is None:
                continue
            condition = _clean(span)
            # Before the name check: in a run that may start mid-sentence the
            # captured "name" is often the wrapped tail ("Diagnostic Code
            # 5260", "January 9, 2026"), and the name check blamed a name the
            # letter does state. Both refuse; only the reason differs.
            if may_start_mid_sentence and start == 0 and not _LEAD_IN.search(span):
                return _refuse(result, (
                    f"the rating statement in {_where(index, sentence)} follows a line with no lower-case "
                    f"letters, or an unfinished line, that may be the start of its condition name, a "
                    f"salutation or letterhead; {_PARTIAL}"
                ))
            problem = _name_problem(condition)
            if problem:
                return _refuse(result, f"{problem}; {_PARTIAL}")
            if _key(condition) in read:
                # Two statements for one condition are either a staged rating
                # (20 percent, then 30 percent from a later date: one
                # disability, counted twice) or two separately rated
                # conditions with identical names (counted once when repeats
                # were deduplicated). The words do not say which.
                return _refuse(result, (
                    f"the decision section rates {condition!r} more than once (a staged rating, or two "
                    f"evaluations the letter does not tell apart); refusing rather than choosing a reading"
                ))
            read[_key(condition)] = [(percent, None, _codes(condition))]
            pending.append((condition, percent))
        for condition, percent in pending:
            source, number = index.locate(condition)
            result.ratings.append(
                ExtractedRating(condition, percent, _classify_extremity(condition), source, number)
            )

    over = [r.percent for r in result.ratings if r.percent > 100]
    if over:
        return _refuse(result, f"an evaluation of {over[0]}% was read, but no evaluation can exceed 100%; "
                               f"refusing rather than computing on it")

    stated: set[int] = set()
    for pattern in _COMBINED:
        for match in pattern.finditer(text):
            if _HISTORICAL_QUALIFIER.search(text, max(0, match.start() - 24), match.start()):
                continue
            stated.add(int(match.group(1)))
    if len(stated) > 1:
        return _refuse(result, (
            f"the letter states more than one combined evaluation ({', '.join(f'{v}%' for v in sorted(stated))}); "
            f"refusing rather than choosing which one is under review"
        ))
    # The decision section refuses a combined statement that carries a second
    # percentage; after a stop heading nothing did, and "is 20 percent until
    # February 28, 2026, and 30 percent thereafter" was read as 20 percent.
    # Sentences are split within paragraphs, not by _sentences: its line
    # grouping broke that statement at "until\nFebruary 28" and hid the stage.
    for paragraph in re.split(r"\n[^\S\n]*\n", tail):
        for sentence in re.split(r"(?<=[.;])\s+", " ".join(paragraph.split())):
            marks = _combined_marks(sentence)
            if marks and _unaccounted(sentence, marks) is not None:
                return _refuse(result, (
                    f"the combined evaluation statement after the {heading!r} heading holds more than one "
                    f"percentage (a staged combined evaluation?); refusing rather than choosing which one is "
                    f"under review"
                ))
    result.stated_combined = next(iter(stated), None)
    if result.stated_combined is not None and result.stated_combined > 100:
        return _refuse(result, f"a combined evaluation of {result.stated_combined}% was read, but no combined "
                               f"evaluation can exceed 100%; refusing rather than comparing against it")

    if tail and result.ratings and not _only_restatements(tail, read):
        return _refuse(result, (
            f"a rating statement after the {heading!r} heading is not one read above it, so the heading "
            f"may have cut the list of evaluations; {_PARTIAL}"
        ))

    if not result.ratings:
        result.unparsed_reason = "no rating lines matched any known tabular or prose pattern"
    elif result.stated_combined is None:
        result.unparsed_reason = "no combined evaluation statement found"
    return result
