"""The report: what a reviewer reads.

Language discipline is a product requirement, not a style preference.
Recheck is an audit aid. It does not adjudicate, and it has no access to the
evidence, the examination findings, or the rating criteria the VA applied. It
compares the arithmetic it can verify against the value the letter states.

So a difference is always framed as:

    POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED

and never as "the VA is wrong". And a result that depends on a fact nobody
established is reported as UNDETERMINED, with the ratings it could be - never
as agreement.
"""

from __future__ import annotations

import pathlib
from collections import Counter

from recheck.case import RULE_426D_ACTION, Case
from recheck.provenance import Actor, wrap

RULE = "=" * 74
THIN = "-" * 74

DISCLAIMER = (
    "Recheck checks combined-rating arithmetic against 38 CFR 4.25 and 4.26 as currently "
    "in force. It reads the narrative decision letter, not the rating code sheet. It is not "
    "legal advice, not a VA adjudication, and not a claims or appeals service. It cannot see "
    "the medical evidence or the rating criteria applied, so a difference is a question to "
    "raise in review - not a conclusion that the decision is wrong."
)


def _or(values) -> str:
    items = [f"{v}%" for v in values]
    if not items:
        return "not established"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def verdict(case: Case, *, question_open: bool = True) -> tuple[str, str]:
    """Return (headline, explanation). Never asserts the VA erred.

    `question_open` is False for a case whose status says it awaits an answer
    but whose Strands session holds no question an answer could reach; it is
    not presented as awaiting one.
    """
    stated = f"{case.stated_combined}%" if case.stated_combined is not None else "no value"
    if case.status == "awaiting_human" and not question_open:
        return (
            "QUESTION NO LONGER OPEN - NO RECOMPUTATION",
            f"The letter states {stated}. This case was waiting on an answer, but its question is no "
            f"longer open in the Strands session, so no answer can be accepted. Nothing was recomputed; "
            f"audit the letter again with --fresh.",
        )
    if case.status == "awaiting_human":
        return (
            "AWAITING YOUR ANSWER",
            f"The letter states {stated}. Depending on facts the letter does not establish, "
            f"the rating could be {_or(case.possible_degrees)}. Answer the question to finish "
            f"this case.",
        )
    if case.status == "undetermined":
        reason = case.undetermined_reason or "the facts needed are not established"
        if case.possible_degrees:
            could = f"The rating could be {_or(case.possible_degrees)}; the letter states {stated}."
        else:
            # Over the enumeration limit there are no possible degrees, and
            # this read "The rating could be not established".
            # The reason says why (too many unknown facts, or more arm and leg
            # ratings than the 4.26(d) search is verified for - where no fact
            # need be unknown at all).
            could = f"No possible ratings were computed. The letter states {stated}."
        return (
            "UNDETERMINED - NOT COMPUTED",
            # Only the first letter is raised: capitalize() lowercased the rest.
            f"{reason[:1].upper() + reason[1:]}. {could} Recheck does not pick one.",
        )
    if case.status == "unparsed":
        reason = case.extraction_failure() or "no assigned evaluations or no combined evaluation statement were found"
        # Capitalising the first letter turned "scanned.pdf has no extractable
        # text layer" into "Scanned.pdf ...", a different file on a
        # case-sensitive filesystem. A reason that starts with the file name
        # keeps it exactly as on disk.
        if not reason.startswith(pathlib.Path(case.source_path).name):
            reason = reason[:1].upper() + reason[1:]
        return ("COULD NOT READ THE LETTER", reason.rstrip(".") + ".")
    if case.status == "ready":
        return ("RUN DID NOT FINISH - NO RECOMPUTATION",
                "The facts Recheck needs are established"
                + (" and your answers are on file" if case.human_answers else "")
                + ", but the run stopped before the arithmetic. Run resume for this case, without "
                  "an answer, to finish it.")
    if case.status != "complete" or case.recomputed_degree is None:
        # Only the compute node produces a figure, and only a "complete" case
        # has one. A run that stopped (a filesystem error, a kill) was
        # reported as "could not establish enough facts", a cause nobody
        # determined.
        return ("RUN DID NOT FINISH - NO RECOMPUTATION",
                f"The run stopped with the case in state {case.status!r}, before a result was reached. "
                f"Nothing was recomputed; audit the letter again with --fresh.")
    if case.stated_combined is None:
        return ("NO STATED VALUE TO COMPARE",
                f"Recheck computes {case.recomputed_degree}% but the letter did not state a "
                f"combined evaluation.")

    basis = "the evaluations as printed"
    if case.human_answers:
        basis += " and the facts you supplied for " + ", ".join(f"[{i}]" for i in case.human_answers)
    # A result that rests on a model's (or a replayed fixture's) extremity
    # group says so, as it does for the reviewer's facts. It used to read as
    # pure arithmetic on "the evaluations as printed".
    ai = [i for i, d in enumerate(case.load_decisions()) if d.group_by is Actor.AI]
    if ai:
        source = "replayed from a fixture" if (case.classifier or "").startswith("scripted") else "from a live model"
        basis += (" and the AI classification of " + ", ".join(f"[{i}]" for i in ai)
                  + f" ({source})")
    caveat = ""
    if case.has_trace_action(RULE_426D_ACTION):
        # With an unknown fact the arithmetic assumes, the prior rule's figure
        # is for that assumption: the facts can settle the result under
        # 4.26(d) and still leave the prior rule's figure open.
        assumption = (" with the fact the arithmetic assumes"
                      if case.has_trace_action("Arithmetic shown with an assumed fact") else "")
        prior = (f"; for a decision period before that date the prior rule gives {case.alternative_degree}%"
                 f"{assumption}" if case.alternative_degree is not None else "")
        caveat = (f" This result depends on the 38 CFR 4.26(d) exception, in force from April 16, 2023"
                  f"{prior}. Check the period the decision covers.")
    if case.recomputed_degree == case.stated_combined:
        return (
            "NO DISCREPANCY FOUND",
            f"Applying 38 CFR 4.25 and 4.26 to {basis} gives {case.recomputed_degree}%, the same "
            f"as the {stated} the letter states.{caveat}",
        )
    higher = case.recomputed_degree > case.stated_combined
    caution = (
        "" if higher else
        " A LOWER recomputation is not an opportunity: raising it could prompt VA to review "
        "the rating downward. Weigh that before acting."
    )
    return (
        "POTENTIAL DISCREPANCY - HUMAN REVIEW RECOMMENDED",
        f"The letter states {stated}. Applying 38 CFR 4.25 and 4.26 to {basis} gives "
        f"{case.recomputed_degree}%, which is {'higher' if higher else 'lower'}.{caveat} Check it against "
        f"the rating code sheet and claims file: this is a question to raise in review, not a "
        f"finding of error, and the difference may rest on facts or judgments Recheck cannot "
        f"see.{caution}",
    )


def classifier_note(case: Case) -> str:
    label = case.classifier or "none"
    if label.startswith("scripted"):
        return f"AI decisions replayed from a committed fixture ({label.split(':', 1)[-1].strip()}) - no model was called"
    if label == "none":
        return "no classifier: terms outside the lexicon are left unknown"
    return f"AI decisions from a live model ({label})"


def timeline_line(case: Case) -> str | None:
    if not case.timeline:
        return None
    groups: list[tuple[int, list[str]]] = []
    for step in case.timeline:
        if groups and groups[-1][0] == step["pid"]:
            groups[-1][1].append(step["node"])
        else:
            groups.append((step["pid"], [step["node"]]))
    return "  ->  ".join(f"{', '.join(nodes)} [process {pid}]" for pid, nodes in groups)


def ownership_line(case: Case) -> str:
    decisions = case.load_decisions()
    groups = Counter(d.group_by.value if d.group_by else "unknown" for d in decisions)
    relevant = [d for d in decisions if d.extremity_group != "none"]
    sides = Counter(d.side_by.value if d.side_by else "unknown" for d in relevant)

    def fmt(counter: Counter, labels: dict[str, str]) -> str:
        return ", ".join(f"{counter[k]} {v}" for k, v in labels.items() if counter[k]) or "none needed"

    return (
        "who established what: extremity group - "
        + fmt(groups, {"DETERMINISTIC": "lexicon", "AI": "AI", "HUMAN": "reviewer", "unknown": "unknown"})
        + "; side - "
        + fmt(sides, {"DETERMINISTIC": "from the letter", "HUMAN": "reviewer", "unknown": "unknown"})
        + "; arithmetic - deterministic"
    )


def render(case: Case, *, show_trace: bool = True, notice: str | None = None, question_open: bool = True) -> str:
    """The report. `notice` is a warning printed under the header, for a case
    whose report cannot be taken as it stands (its letter changed, its
    question was lost); `question_open` is passed to verdict()."""
    out: list[str] = [RULE, "RECHECK - combined rating verification",
                      f"case {case.case_id}   letter: {pathlib.Path(case.source_path).name}",
                      classifier_note(case), RULE]
    if notice:
        out += wrap(notice, 72) + [RULE]

    # A scripted run replays committed answers; every place an AI decision is
    # shown says so, or a demo on the zero-model path reads as a model run.
    # The evaluations table used to say plain [AI] under a trace that said
    # "replayed fixture".
    ai_label = "AI - replayed fixture" if (case.classifier or "").startswith("scripted") else "AI"
    if show_trace and case.trace:
        out += ["", "DECISION TRACE - who decided what", THIN, case.load_trace().render(ai_label=ai_label)]

    decisions = case.load_decisions()
    if decisions:
        out += ["", "EVALUATIONS  (who established each fact in brackets)", THIN]
        for index, d in enumerate(decisions):
            lines = wrap(d.condition, 64) or [""]
            out.append(f"  [{index}] {str(d.percent) + '%':<5}{lines[0]}")
            out += [f"{'':11}{line}" for line in lines[1:]]
            facts = f"extremity group: {d.extremity_group} [{_by(d.group_by, ai_label=ai_label)}]"
            if d.extremity_group != "none":
                facts += f"   side: {d.laterality} [{_by(d.side_by, side=True)}]"
            out.append(f"{'':11}{facts}")

    def row(label: str, value: object) -> str:
        return f"  {label:<34}{value}"

    out += ["", "ARITHMETIC", THIN, row("stated combined evaluation", _pct(case.stated_combined))]
    if case.status == "complete":
        out += [
            row("recomputed combined value", case.recomputed_combined),
            row("recomputed final degree", _pct(case.recomputed_degree)),
            row("38 CFR 4.26 bilateral factor", "applied" if case.bilateral_applied else "not applied"),
        ]
        if case.alternative_degree is not None and case.alternative_degree != case.recomputed_degree:
            if case.bilateral_applied and case.has_trace_action(RULE_426D_ACTION):
                # 4.26(d) left SOME bilateral disabilities out. The alternative
                # is every one of them in the factor, not no factor: it was
                # labelled "without the factor" where that figure differs.
                label = "final degree without 4.26(d)"
            else:
                label = f"final degree {'without' if case.bilateral_applied else 'with'} the factor"
            out.append(row(label, f"{case.alternative_degree}%"))
        if case.immaterial_unknowns:
            listed = ", ".join(f"[{i}]" for i in case.immaterial_unknowns)
            out.append(f"  unknown facts for {listed} could not change this result, so nobody was asked")
    elif case.status == "undetermined" and not case.possible_degrees:
        out.append(row("possible final degrees", "not computed (see the reason below)"))
    else:
        out.append(row("possible final degrees", _or(case.possible_degrees)))

    headline, explanation = verdict(case, question_open=question_open)
    out += ["", RULE, headline, RULE] + wrap(explanation, 72)
    out += ["", *wrap(ownership_line(case), 72)]
    timeline = timeline_line(case)
    if timeline:
        out += wrap("graph nodes: " + timeline, 72)
    out += [""] + wrap(DISCLAIMER, 72) + [RULE]
    return "\n".join(out)


def _by(actor: Actor | None, *, side: bool = False, ai_label: str = "AI") -> str:
    if actor is None:
        return "not established"
    if actor is Actor.DETERMINISTIC:
        return "letter" if side else "lexicon"
    return {"AI": ai_label, "HUMAN": "reviewer"}[actor.value]


def _pct(value: int | None) -> str:
    return "not established" if value is None else f"{value}%"
