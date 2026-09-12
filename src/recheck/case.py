"""Durable case state.

Two different things have to survive a process boundary, and conflating them
would be a mistake:

  GRAPH STATE    which node is pending, and which interrupt is outstanding.
                 Owned by Strands (FileSessionManager + graph.serialize_state).
  DOMAIN STATE   the extracted ratings, the classifications, who decided
                 each one, the human's answers, and the trace. Owned here.

Keeping domain state out of the model provider and out of the agent
framework is deliberate: persistence must not be coupled to either, so the
provider stays replaceable and a case remains readable without Strands
installed. The case file is plain JSON on purpose - a reviewer can open it.

No PII is stored, because none is extracted. The letters are synthetic and
the only values retained are condition names, percentages, line numbers and
decisions.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any

from recheck.classify import Decision
from recheck.provenance import Actor, Entry, Trace

SCHEMA_VERSION = 1


@dataclass
class Case:
    """Everything Recheck knows about one letter."""

    case_id: str
    source_path: str
    stated_combined: int | None = None
    ratings: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    human_answers: dict[str, str] = field(default_factory=dict)
    recomputed_combined: int | None = None
    recomputed_degree: int | None = None
    bilateral_applied: bool = False
    bilateral_note: str | None = None
    alternative_degree: int | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    status: str = "open"
    schema_version: int = SCHEMA_VERSION

    # -- trace bridging --------------------------------------------------
    def load_trace(self) -> Trace:
        restored = Trace()
        for raw in self.trace:
            restored.entries.append(
                Entry(
                    actor=Actor(raw["actor"]),
                    action=raw["action"],
                    detail=raw["detail"],
                    value=raw.get("value"),
                    confidence=raw.get("confidence"),
                    evidence=raw.get("evidence"),
                    rule=raw.get("rule"),
                )
            )
        return restored

    def store_trace(self, trace: Trace) -> None:
        self.trace = [
            {
                "actor": e.actor.value,
                "action": e.action,
                "detail": e.detail,
                "value": e.value,
                "confidence": e.confidence,
                "evidence": e.evidence,
                "rule": e.rule,
            }
            for e in trace.entries
        ]

    def load_decisions(self) -> list[Decision]:
        return [
            Decision(
                condition=d["condition"],
                percent=d["percent"],
                extremity_group=d["extremity_group"],
                laterality=d["laterality"],
                decided_by=Actor(d["decided_by"]),
                confidence=d.get("confidence"),
                needs_human=d["needs_human"],
                reason=d.get("reason"),
                evidence=d.get("evidence"),
            )
            for d in self.decisions
        ]

    def store_decisions(self, decisions: list[Decision]) -> None:
        self.decisions = [
            {**asdict(d), "decided_by": d.decided_by.value} for d in decisions
        ]


class CaseStore:
    """Filesystem-backed case storage. One directory per case."""

    def __init__(self, root: str | pathlib.Path = "runs") -> None:
        self.root = pathlib.Path(root)

    def dir_for(self, case_id: str) -> pathlib.Path:
        _validate_case_id(case_id)
        return self.root / case_id

    def path_for(self, case_id: str) -> pathlib.Path:
        return self.dir_for(case_id) / "case.json"

    def session_dir(self, case_id: str) -> pathlib.Path:
        """Where Strands keeps graph and interrupt state for this case."""
        return self.dir_for(case_id) / "session"

    def exists(self, case_id: str) -> bool:
        return self.path_for(case_id).exists()

    def save(self, case: Case) -> pathlib.Path:
        path = self.path_for(case.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot leave a truncated case.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(case), indent=1), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, case_id: str) -> Case:
        path = self.path_for(case_id)
        if not path.exists():
            raise FileNotFoundError(f"no such case: {case_id} (looked in {path})")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CaseCorrupt(f"case {case_id} is not valid JSON: {exc}") from exc
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CaseCorrupt(
                f"case {case_id} has schema_version {version!r}, this build expects "
                f"{SCHEMA_VERSION}. Refusing to guess at an incompatible case."
            )
        known = set(Case.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise CaseCorrupt(f"case {case_id} has unexpected fields: {sorted(unknown)}")
        return Case(**raw)


class CaseCorrupt(Exception):
    """A case file exists but cannot be trusted."""


def _validate_case_id(case_id: str) -> None:
    """Reject anything that could escape the store directory.

    Case ids arrive from the command line, so "../../etc/passwd" and absolute
    paths must not be usable as identifiers.
    """
    if not case_id or len(case_id) > 64:
        raise ValueError("case id must be 1-64 characters")
    if not all(c.isalnum() or c in "-_" for c in case_id):
        raise ValueError(
            f"case id may contain only letters, digits, '-' and '_'; got {case_id!r}"
        )
