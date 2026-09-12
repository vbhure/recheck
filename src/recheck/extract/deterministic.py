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

Laterality = Literal["left", "right", "bilateral", "unknown", "not_applicable"]
ExtremityGroup = Literal["upper", "lower", "none"]

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

# Conditions that are never extremity disabilities.
NON_EXTREMITY_HINTS = (
    "tinnitus", "post-traumatic stress", "ptsd", "hearing", "migraine",
    "lumbosacral", "spine", "sleep apnea", "diabetes",
)

LEXICON_SIZE = len(UPPER_TERMS) + len(LOWER_TERMS) + len(UPPER_PHRASES) + len(LOWER_PHRASES)

# Percentages after these headings describe rating CRITERIA, not the
# veteran's assigned evaluations.
STOP_HEADINGS = ("REASONS FOR DECISION", "EVIDENCE", "REFERENCES")

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
    condition: str
    percent: int
    laterality: Laterality
    extremity_group: ExtremityGroup
    source_line: str
    source_line_number: int


@dataclass
class Extraction:
    ratings: list[ExtractedRating] = field(default_factory=list)
    stated_combined: int | None = None
    ambiguities: list[str] = field(default_factory=list)
    unparsed_reason: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.ratings) and self.stated_combined is not None


def _decision_scope(text: str) -> str:
    """Trim explanatory sections whose percentages are not assigned ratings."""
    cut = len(text)
    upper = text.upper()
    for heading in STOP_HEADINGS:
        index = upper.find(heading)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


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


def _classify_extremity(condition: str) -> ExtremityGroup:
    low = condition.lower()
    if any(hint in low for hint in NON_EXTREMITY_HINTS):
        return "none"
    if any(p in low for p in UPPER_PHRASES):
        return "upper"
    if any(p in low for p in LOWER_PHRASES):
        return "lower"
    words = set(re.findall(r"[a-z]+", low))
    if words & UPPER_TERMS:
        return "upper"
    if words & LOWER_TERMS:
        return "lower"
    return "none"


def _classify_laterality(condition: str, group: ExtremityGroup) -> Laterality:
    if group == "none":
        return "not_applicable"
    low = condition.lower()
    if re.search(r"\bbilateral\b", low):
        return "bilateral"
    has_left = re.search(r"\bleft\b", low) is not None
    has_right = re.search(r"\bright\b", low) is not None
    if has_left and has_right:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return "unknown"


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


def parse(text: str) -> Extraction:
    """Extract ratings and the stated combined evaluation from letter text."""
    result = Extraction()
    scope = _decision_scope(text)
    lines = text.splitlines()

    def locate(fragment: str) -> tuple[str, int]:
        head = fragment.strip().split("\n")[0][:38]
        if head:
            for number, line in enumerate(lines, start=1):
                if head in line:
                    return line.strip(), number
        return fragment.strip()[:120], 0

    seen: set[tuple[str, int]] = set()

    for match in _TABULAR.finditer(scope):
        condition = _clean(match.group("condition"))
        percent = int(match.group("pct"))
        key = (condition.lower(), percent)
        if key in seen:
            continue
        seen.add(key)
        src, number = locate(match.group(0))
        group = _classify_extremity(condition)
        result.ratings.append(
            ExtractedRating(condition, percent, _classify_laterality(condition, group), group, src, number)
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
                src, number = locate(condition)
                group = _classify_extremity(condition)
                result.ratings.append(
                    ExtractedRating(condition, percent, _classify_laterality(condition, group), group, src, number)
                )
                break  # most specific pattern wins for a given sentence

    for pattern in _COMBINED:
        match = pattern.search(text)
        if match:
            result.stated_combined = int(match.group(1))
            break

    for rating in result.ratings:
        if rating.extremity_group != "none" and rating.laterality == "unknown":
            result.ambiguities.append(
                f"'{rating.condition}' is a {rating.extremity_group} extremity condition but the "
                f"letter does not state left or right; 4.26 eligibility cannot be determined."
            )

    if not result.ratings:
        result.unparsed_reason = "no rating lines matched any known tabular or prose pattern"
    elif result.stated_combined is None:
        result.unparsed_reason = "no combined evaluation statement found"
    return result


def candidate_bilateral_pairs(ratings: list[ExtractedRating]) -> list[tuple[int, int]]:
    """Index pairs satisfying 4.26: same group, opposite sides, both compensable."""
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(ratings):
        for j in range(i + 1, len(ratings)):
            b = ratings[j]
            if a.extremity_group == "none" or a.extremity_group != b.extremity_group:
                continue
            if {a.laterality, b.laterality} != {"left", "right"}:
                continue
            if a.percent < 10 or b.percent < 10:
                continue
            pairs.append((i, j))
    return pairs
