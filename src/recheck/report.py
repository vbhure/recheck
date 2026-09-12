"""The report: what a reviewer reads.

Language discipline is a product requirement, not a style preference.
Recheck is an audit aid. It does not adjudicate, and it has no access to the
evidence, the examination findings, or the rating criteria the VA applied. It
compares the arithmetic it can verify against the value the letter states.

So the verdict is always framed as:

    POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED

and never as "the VA is wrong". A difference can have legitimate causes this
tool cannot see, including a different and equally lawful judgment about
which conditions are paired extremities - which is exactly the ambiguity the
interrupt exists to surface.
"""

from __future__ import annotations

from recheck.case import Case
from recheck.provenance import Actor

RULE = "=" * 74
THIN = "-" * 74

DISCLAIMER = (
    "Recheck verifies combined-rating arithmetic against 38 CFR 4.25 and 4.26. "
    "It is not legal advice, not a VA adjudication, and not a claims or appeals "
    "service. It cannot see the medical evidence or the rating criteria applied, "
    "so a difference is a question to raise with an accredited representative - "
    "not a conclusion that the decision is wrong."
)


def verdict(case: Case) -> tuple[str, str]:
    """Return (headline, explanation). Never asserts the VA erred."""
    if case.recomputed_degree is None:
        return (
            "INCOMPLETE - NO RECOMPUTATION",
            "Recheck could not establish enough facts to recompute this evaluation.",
        )
    if case.stated_combined is None:
        return (
            "NO STATED VALUE TO COMPARE",
            f"Recheck computes {case.recomputed_degree}% but the letter did not state a "
            f"combined evaluation.",
        )
    if case.recomputed_degree == case.stated_combined:
        return (
            "NO DISCREPANCY FOUND",
            f"The stated combined evaluation of {case.stated_combined}% matches Recheck's "
            f"recomputation under 38 CFR 4.25/4.26.",
        )
    direction = "higher" if case.recomputed_degree > case.stated_combined else "lower"
    return (
        "POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED",
        f"The letter states {case.stated_combined}%. Applying 38 CFR 4.25 and 4.26 to the "
        f"individual evaluations as printed gives {case.recomputed_degree}%, which is "
        f"{direction}. This is a question for an accredited representative, not a finding "
        f"of error - the difference may rest on a judgment about paired extremities that "
        f"Recheck surfaced rather than resolved.",
    )


def render(case: Case, *, show_trace: bool = True) -> str:
    out: list[str] = []
    out.append(RULE)
    out.append("RECHECK - combined rating verification")
    out.append(f"case {case.case_id}   source: {case.source_path}")
    out.append(RULE)

    if show_trace and case.trace:
        out.append("")
        out.append("DECISION TRACE - who decided what")
        out.append(THIN)
        out.append(case.load_trace().render())

    decisions = case.load_decisions()
    if decisions:
        out.append("")
        out.append("EVALUATIONS AS EXTRACTED")
        out.append(THIN)
        out.append(f"  {'#':<3}{'%':<6}{'group/side':<18}{'decided by':<15}condition")
        for index, d in enumerate(decisions):
            side = f"{d.extremity_group}/{d.laterality}"
            flag = "  <-- needs human" if d.needs_human else ""
            out.append(
                f"  {index:<3}{str(d.percent) + '%':<6}{side:<18}{d.decided_by.value:<15}"
                f"{d.condition[:30]}{flag}"
            )

    out.append("")
    out.append("ARITHMETIC")
    out.append(THIN)
    out.append(f"  stated combined evaluation     {_pct(case.stated_combined)}")
    out.append(f"  recomputed combined value      {_pct(case.recomputed_combined)}")
    out.append(f"  recomputed final degree        {_pct(case.recomputed_degree)}")
    out.append(f"  38 CFR 4.26 bilateral factor   {'applied' if case.bilateral_applied else 'not applied'}")
    if case.alternative_degree is not None:
        out.append(f"  alternative without/with 4.26  {case.alternative_degree}%")
    if case.bilateral_note:
        out.append(f"  note: {case.bilateral_note}")

    headline, explanation = verdict(case)
    out.append("")
    out.append(RULE)
    out.append(headline)
    out.append(RULE)
    for line in _wrap(explanation, 72):
        out.append(line)

    counts = {a.value: len(case.load_trace().by_actor(a)) for a in Actor}
    out.append("")
    out.append(
        f"ownership: {counts['DETERMINISTIC']} deterministic, {counts['AI']} AI, "
        f"{counts['HUMAN']} human decision(s)"
    )
    out.append("")
    for line in _wrap(DISCLAIMER, 72):
        out.append(line)
    out.append(RULE)
    return "\n".join(out)


def _pct(value: int | None) -> str:
    return "not established" if value is None else f"{value}%"


def _wrap(text: str, width: int) -> list[str]:
    words, out, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out
