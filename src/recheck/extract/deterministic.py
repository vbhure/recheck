"""Deterministic extraction of ratings from VA decision letter text.

This is a genuine best-effort parser, not a strawman. It exists to answer one
question honestly: how far does deterministic parsing actually get? Anything
the model is later asked to do must be something this parser demonstrably
cannot do.

It uses no model, no network and no I/O beyond the string it is handed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# "none" means recognised as a non-extremity condition. "unrecognised"
# means the lexicon has no opinion - a genuinely different state, and the
# only one that justifies a model call.
ExtremityGroup = Literal["upper", "lower", "none", "unrecognised"]

# 4.26(a): "arms" and "legs" mean the upper and lower extremities AS A WHOLE,
# so a wrist and a forearm belong to the same extremity group.
UPPER_TERMS = {
    "arm", "arms", "forearm", "forearms", "elbow", "wrist", "wrists", "hand",
    "hands", "finger", "fingers", "thumb", "shoulder",
}
LOWER_TERMS = {
    "leg", "legs", "thigh", "knee", "knees", "ankle", "ankles", "foot", "feet",
    "toe", "toes", "hip",
}
UPPER_PHRASES = ("upper extremity", "upper extremities")
LOWER_PHRASES = ("lower extremity", "lower extremities")

# Conditions that are never extremity disabilities - but only when nothing in
# the name points at an arm or a leg. "Radiculopathy, right lower extremity,
# associated with lumbosacral strain" is a leg disability; checking these
# hints first used to call it "none" and silently drop a 4.26 pair.
NON_EXTREMITY_HINTS = (
    "tinnitus", "post-traumatic stress", "ptsd", "hearing", "migraine",
    "lumbosacral", "spine", "sleep apnea", "diabetes",
)

# Clauses that name a DIFFERENT condition the rated one is linked to. The
# rated condition's own anatomy and side come before them: in "Left knee
# strain, secondary to right knee strain" the rated knee is the left one.
_LINKED_CLAUSE = re.compile(
    r"[,;]?\s*\(?\b(?:secondary to|associated with|due to|claimed as|incident to|aggravated by|"
    r"as a result of|resulting from)\b.*$",
    re.I,
)
# Handedness is not a side: "(right hand dominant)", "(major)", "right-handed".
_HANDEDNESS = re.compile(
    r"\((?:[^)]*\b(?:dominant|major|minor|handed)\b[^)]*)\)"
    r"|\b(?:right|left)[- ]hand(?:ed)? dominant\b|\b(?:right|left)-handed\b",
    re.I,
)


def primary_clause(condition: str) -> str:
    """The part of a condition name that describes the rated condition itself."""
    text = _HANDEDNESS.sub(" ", condition)
    text = _LINKED_CLAUSE.sub("", text)
    return " ".join(text.split()).strip(" ,;")


def linked_clause(condition: str) -> str:
    """Whatever primary_clause removed (the linked condition), if anything."""
    text = " ".join(_HANDEDNESS.sub(" ", condition).split())
    match = _LINKED_CLAUSE.search(text)
    return match.group(0) if match else ""

LEXICON_SIZE = len(UPPER_TERMS) + len(LOWER_TERMS) + len(UPPER_PHRASES) + len(LOWER_PHRASES)

# Percentages after these headings describe rating CRITERIA, not the
# veteran's assigned evaluations. Matched only as a heading on its own line:
# an unanchored match cut the ratings list at "with x-ray evidence of
# arthritis" inside a rating line and silently dropped every later rating.
STOP_HEADINGS = ("REASONS FOR DECISION", "EVIDENCE", "REFERENCES")
_STOP_HEADING = re.compile(
    r"^[ \t]*(?:" + "|".join(STOP_HEADINGS) + r")[ \t]*:?[ \t]*$", re.I | re.M
)

_TABULAR = re.compile(
    r"^\s*\d+\.\s*(?P<condition>.+?)\s*\.{3,}\s*(?P<pct>\d{1,3})\s*%",
    re.MULTILINE,
)

# Prose forms, most specific first. "increased to" must win over "currently
# evaluated as" in the same sentence, or historical values get captured.
_PROSE_PATTERNS = [
    re.compile(r"(?P<condition>[^.]*?)\bis increased to\s+(?P<pct>\d{1,3})\s+percent", re.I),
    re.compile(r"(?P<condition>[^.]*?)\bis continued as\s+(?P<pct>\d{1,3})\s+percent", re.I),
    re.compile(r"(?P<condition>[^.]*?)\bwith an evaluation of\s+(?P<pct>\d{1,3})\s+percent", re.I),
]

_COMBINED = [
    re.compile(r"COMBINED EVALUATION FOR COMPENSATION\s*:?\s*(\d{1,3})\s*%", re.I),
    re.compile(r"combined evaluation for compensation is\s+(\d{1,3})\s+percent", re.I),
]


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


def _decision_scope(text: str) -> str:
    """Trim explanatory sections whose percentages are not assigned ratings."""
    match = _STOP_HEADING.search(text)
    return text[: match.start()] if match else text


def _sentences(text: str) -> list[str]:
    """Split into whitespace-normalised sentences.

    VA letters are hard-wrapped, so a clause can straddle a newline. Patterns
    must see "with an evaluation of" as contiguous. Splitting on sentence
    boundaries also prevents a non-greedy condition capture from running
    backwards into the letterhead.
    """
    flat = re.sub(r"\s+", " ", text)
    # Protect the period in a middle initial ("Name: R. SYNTHETIC") and in
    # abbreviations like "No." so they do not end a sentence.
    flat = re.sub(r"\b([A-Z])\.\s", r"\1<DOT> ", flat)
    parts = re.split(r"(?<=[.;])\s+", flat)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _lexical_group(text: str) -> ExtremityGroup:
    low = text.lower()
    words = set(re.findall(r"[a-z]+", low))
    upper = any(p in low for p in UPPER_PHRASES) or bool(words & UPPER_TERMS)
    lower = any(p in low for p in LOWER_PHRASES) or bool(words & LOWER_TERMS)
    if upper and lower:
        return "unrecognised"  # "hand and foot" - not the lexicon's call
    if upper:
        return "upper"
    if lower:
        return "lower"
    if any(hint in low for hint in NON_EXTREMITY_HINTS):
        return "none"
    return "unrecognised"


def _classify_extremity(condition: str) -> ExtremityGroup:
    """The lexicon's verdict on a condition name: upper, lower, none, or no opinion.

    Anatomy in the rated condition itself decides first. When the rated
    condition names no anatomy but a linked clause does ("radiculopathy
    associated with lumbar spine", "strain secondary to a knee injury"), the
    lexicon has no opinion - it does not guess from the other condition.
    """
    primary = _lexical_group(primary_clause(condition))
    if primary != "unrecognised":
        return primary
    if linked_clause(condition):
        return "unrecognised"
    return primary



_LEAD_IN = re.compile(r"(?:service connection for|evaluation of)\s+", re.I)
_TRAILING_VERB = re.compile(r"\s+is (?:granted|continued|increased|assigned)\b.*$", re.I)


def _clean(condition: str) -> str:
    """Reduce a matched span to just the condition name.

    Two failure modes this guards against, both real:
      - the non-greedy capture running backwards into the letterhead, so the
        condition reads "SYNTHETIC File Number: 00-000-002 ...". Anchoring to
        the LAST lead-in phrase fixes it.
      - trailing verb phrases ("... is granted") ending up in the condition
        name, which pollutes the evidence shown to a reviewer.
    """
    condition = re.sub(r"\s+", " ", condition).strip()
    condition = re.sub(r",?\s*currently evaluated as \d{1,3} percent disabling,?", "", condition, flags=re.I)
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
        self.lines = text.splitlines()
        flat: list[str] = []
        origin: list[int] = []
        previous_space = False
        for offset, char in enumerate(text):
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
        self.cursor = 0

    def _line_of(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1

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
                first, last = self._line_of(self.origin[at]), self._line_of(self.origin[end])
                self.cursor = at + len(probe)
                span = " ".join(line.strip() for line in self.lines[first - 1:last])
                return span, first
        return phrase.strip()[:120], 0


def parse(text: str) -> Extraction:
    """Extract ratings and the stated combined evaluation from letter text."""
    result = Extraction()
    scope = _decision_scope(text)
    index = _LineIndex(text)
    seen: set[tuple[str, int]] = set()

    # Every numbered row is its own evaluation, even when two rows read
    # identically (two separately rated scars): tabular rows are never deduped.
    for match in _TABULAR.finditer(scope):
        condition = _clean(match.group("condition"))
        percent = int(match.group("pct"))
        number = scope.count("\n", 0, match.start("condition")) + 1
        source = index.lines[number - 1].strip() if number <= len(index.lines) else condition
        result.ratings.append(
            ExtractedRating(condition, percent, _classify_extremity(condition), source, number)
        )

    if not result.ratings:
        # Sentence-scoped matching. Letters are hard-wrapped, so patterns are
        # applied to whitespace-normalised sentences rather than raw text -
        # otherwise a line break inside "with an\nevaluation of" defeats the
        # match, and an unanchored capture bleeds backwards through the header.
        for sentence in _sentences(scope):
            for pattern in _PROSE_PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                condition = _clean(match.group("condition"))
                percent = int(match.group("pct"))
                key = (condition.lower(), percent)
                if not condition or key in seen:
                    break
                seen.add(key)
                source, number = index.locate(condition)
                result.ratings.append(
                    ExtractedRating(condition, percent, _classify_extremity(condition), source, number)
                )
                break  # most specific pattern wins for a given sentence

    for pattern in _COMBINED:
        match = pattern.search(text)
        if match:
            result.stated_combined = int(match.group(1))
            break

    if not result.ratings:
        result.unparsed_reason = "no rating lines matched any known tabular or prose pattern"
    elif result.stated_combined is None:
        result.unparsed_reason = "no combined evaluation statement found"
    return result
