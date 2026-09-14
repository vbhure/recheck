"""Durable case state.

Two different things have to survive a process boundary, and conflating them
would be a mistake:

  GRAPH STATE    which node is pending, and which interrupt is outstanding.
                 Owned by Strands: FileSessionManager persists it under
                 <store>/<case>/session and restores it when the graph is built.
  DOMAIN STATE   the extracted ratings, the established facts and who
                 established each one, the reviewer's answers, and the trace.
                 Owned here, in <store>/<case>/case.json.

Keeping domain state out of the model provider and out of the agent
framework is deliberate: a case remains readable without Strands installed,
and the provider stays replaceable. The case file is plain JSON on purpose - a
reviewer can open it.

Status is one of STATUSES; the compute gate and the compute node accept only
"ready" and "complete":

  open -> extracted -> classified -> ready -> complete
                          |      \\-> awaiting_human -> awaiting_human (follow-up) | ready | undetermined
                          \\-> undetermined (too many possibilities to try)
  unparsed, undetermined: terminal, nothing computed

No PII is extracted by design. What is kept is the letter's path and
SHA-256, condition names, percentages, evidence lines and line numbers, and
decisions. Names and evidence lines are the letter's own text, so treat a
store built from real letters as sensitive.

The case file is also UNTRUSTED INPUT. Anyone who can write to the store can
edit it, and a red team did: a "complete" status with a made-up
recomputed_degree was printed by `show` and `sweep` as the output of the 38
CFR engine; a type-confused field crashed the whole sweep; a forged case_id
redirected writes into a different case and destroyed its result. So
CaseStore.load checks every field's type and value, checks that the file
names the case it was read for, and re-derives a completed result from the
stored facts before anything can report it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import stat
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from recheck.classify import Decision
from recheck.materiality import assess, evaluate_for_report
from recheck.provenance import Actor, Entry, Trace

# 3: document_sha256 added, so a case can tell when its letter has changed.
SCHEMA_VERSION = 3

STATUSES = (
    "open", "extracted", "classified", "awaiting_human", "ready", "complete", "unparsed", "undetermined",
)


@dataclass
class Case:
    """Everything Recheck knows about one letter."""

    case_id: str
    source_path: str
    #: SHA-256 of the letter's bytes as the extract node read them. Resume and
    #: show never re-read the letter, so without this a report kept naming a
    #: file whose current content it contradicted.
    document_sha256: str | None = None
    classifier: str = "none"
    stated_combined: int | None = None
    ratings: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    human_answers: dict[str, str] = field(default_factory=dict)
    rejected_answer: str | None = None
    immaterial_unknowns: list[int] = field(default_factory=list)
    possible_degrees: list[int] = field(default_factory=list)
    undetermined_reason: str | None = None
    recomputed_combined: int | None = None
    recomputed_degree: int | None = None
    bilateral_applied: bool = False
    bilateral_note: str | None = None
    alternative_degree: int | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    status: str = "open"
    schema_version: int = SCHEMA_VERSION

    def extraction_failure(self) -> str | None:
        """Why the document was refused, as the extract node recorded it."""
        for raw in reversed(self.trace):
            if raw.get("action") == "Extraction FAILED":
                return raw.get("detail")
        return None

    def has_trace_action(self, action: str) -> bool:
        return any(raw.get("action") == action for raw in self.trace)

    def letter_changed(self) -> bool | None:
        """Has the letter changed since it was audited?

        True or False when the fingerprint can be compared; None when it
        cannot - no fingerprint was recorded, or the file is gone or
        unreadable. A missing letter is not a change: resume deliberately
        works from the persisted state alone.
        """
        if self.document_sha256 is None:
            return None
        return document_fingerprint(self.source_path, expected=self.document_sha256)

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
        def actor(value: str | None) -> Actor | None:
            return Actor(value) if value else None

        return [
            Decision(
                condition=d["condition"],
                percent=d["percent"],
                extremity_group=d["extremity_group"],
                laterality=d["laterality"],
                group_by=actor(d.get("group_by")),
                side_by=actor(d.get("side_by")),
                confidence=d.get("confidence"),
                note=d.get("note"),
                evidence=d.get("evidence"),
            )
            for d in self.decisions
        ]

    def store_decisions(self, decisions: list[Decision]) -> None:
        self.decisions = [
            {
                **asdict(d),
                "group_by": d.group_by.value if d.group_by else None,
                "side_by": d.side_by.value if d.side_by else None,
            }
            for d in decisions
        ]


# The fingerprint is compared against a letter that was at most this large
# when it was read (graph.MAX_DOCUMENT_BYTES); anything bigger has changed, and
# is not read in full just to prove it.
_FINGERPRINT_READ_LIMIT = 8 * 1024 * 1024


def document_fingerprint(path: str | pathlib.Path, *, expected: str) -> bool | None:
    """Compare a file's SHA-256 with `expected`: True if it CHANGED, None if unreadable."""
    try:
        p = pathlib.Path(path)
        if not p.is_file():
            return None
        if p.stat().st_size > _FINGERPRINT_READ_LIMIT:
            return True
        digest = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest() != expected


class CaseBusy(Exception):
    """Another process is working on this case right now."""


@dataclass(frozen=True)
class CaseSnapshot:
    """case.json bytes (None if absent) and the session tree (None if absent) of one case."""

    case_id: str
    case_json: bytes | None
    session: tuple[tuple[str, ...], dict[str, bytes]] | None


#: What a snapshot may hold in memory. A case's session is a few JSON files of
#: a few kilobytes.
_SESSION_READ_LIMIT = 8 * 1024 * 1024


def _is_link(path: pathlib.Path) -> bool:
    """A symbolic link, or any other reparse point: a Windows junction is not a symlink to pathlib."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _read_tree(root: pathlib.Path) -> tuple[tuple[str, ...], dict[str, bytes]] | None:
    """(sub-directories, {file: bytes}) under root, as POSIX relative paths; None if root is absent.

    Links are refused, not followed, and so is more than _SESSION_READ_LIMIT.
    rglob followed a junction planted in a session: every resume read the
    linked tree into memory, and after a resume that failed - a read-only
    case.json is enough - restore() wrote the linked files back as real ones,
    copying a directory from outside the store into it. Strands never writes
    a link into a session.
    """
    if _is_link(root):
        raise OSError(f"{root} is a link; a case's session is never one")
    if not root.is_dir():
        return None
    directories, files, size = [], {}, 0
    pending = [root]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            relative = path.relative_to(root).as_posix()
            if _is_link(path):
                raise OSError(f"the session under {root} holds a link, {relative}; a case's session never does")
            if path.is_dir():
                directories.append(relative)
                pending.append(path)
                continue
            with path.open("rb") as handle:
                data = handle.read(_SESSION_READ_LIMIT - size + 1)
            size += len(data)
            if size > _SESSION_READ_LIMIT:
                raise OSError(f"the session under {root} holds more than {_SESSION_READ_LIMIT // 2**20} MB; "
                              f"a case's session is a few small files")
            files[relative] = data
    return tuple(sorted(directories)), files


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

    @contextlib.contextmanager
    def lock(self, case_id: str) -> Iterator[None]:
        """Hold this case exclusively for one command, or raise CaseBusy.

        Two `resume` processes started together on one case interleaved their
        writes: one race in three left case.json as invalid JSON, a stranded
        question, or answers on file with no figure. Only one process may act
        on a case at a time; the second is refused, not queued.

        The lock is an OS byte-range lock (msvcrt on Windows, flock on POSIX)
        on <store>/<case>.lock - beside the case directory, not in it, so that
        discarding the directory is not blocked by the lock's own open file.
        The OS releases it when the process dies, so a killed run never leaves
        a case locked. The file is removed on release so the store holds only
        cases; on POSIX the inode is re-checked after locking, because a
        process that opened the file just before it was removed would
        otherwise hold a lock on an orphan.
        """
        path = self.root / f"{self.dir_for(case_id).name}.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        fd = _acquire(path, case_id)
        try:
            yield
        finally:
            if _UNLINK_WHILE_LOCKED:
                # POSIX: remove the file while still holding the lock, then
                # close. Released first, a second process could lock the old
                # file and pass its inode re-check before the removal, and a
                # third could then create and lock a new file - two holders.
                with contextlib.suppress(OSError):
                    path.unlink()
                _release(fd)
            else:
                # Windows cannot remove a file another handle has open, so a
                # waiting process never holds an orphan; close first.
                _release(fd)
                with contextlib.suppress(OSError):
                    path.unlink()

    def discard(self, case_id: str) -> None:
        """Remove a case entirely - domain state and Strands session - to re-audit it.

        The directory is first renamed out of the way in one atomic step.
        Removing it in place deleted the session and then failed on a
        read-only or locked case.json: a traceback, a half-deleted case still
        "awaiting" a question its session no longer held, and a sweep that
        stopped there. If the rename fails, nothing has been touched and the
        OSError says why; once it succeeds the case is gone, and a file left
        inside the renamed directory cannot bring it back.
        """
        directory = self.dir_for(case_id)
        if not directory.exists():
            return
        trash = self.root / f".discarded-{directory.name}-{uuid.uuid4().hex[:12]}"
        os.replace(directory, trash)
        _remove_tree(trash)

    def snapshot(self, case_id: str) -> "CaseSnapshot":
        """Everything a run can change on disk for this case: case.json and its Strands session.

        A resume that raised part-way - a locked or read-only case.json, a full
        disk, Ctrl+C - left the session with the interrupt already consumed
        and no node waiting on it, so the reviewer's question could never be
        answered again without discarding the case. Taken before the run and
        restored if it raises (CaseStore.restore), the question stays open.
        Held in memory: the session is a few small JSON files, and a snapshot
        on disk would lengthen paths that are already near Windows' limit and
        outlive a killed process.
        """
        session = self.session_dir(case_id)
        path = self.path_for(case_id)
        return CaseSnapshot(case_id, path.read_bytes() if path.exists() else None, _read_tree(session))

    def restore(self, snapshot: "CaseSnapshot") -> None:
        """Put back what snapshot() recorded, touching only what differs.

        case.json is rewritten only if its bytes changed: a read-only file the
        run failed to replace is left as it is. The session directory, if it
        changed, is moved aside in one step and rebuilt from the snapshot.
        """
        session = self.session_dir(snapshot.case_id)
        if _read_tree(session) != snapshot.session:
            # No longer than "session": Strands' deepest file sits close to
            # the Windows path limit, and the aside copy must still be removable.
            aside = session.with_name(f".r{uuid.uuid4().hex[:5]}")
            if session.exists():
                os.replace(session, aside)
            if snapshot.session is not None:
                directories, files = snapshot.session
                session.mkdir(parents=True, exist_ok=True)
                for relative in directories:
                    (session / relative).mkdir(parents=True, exist_ok=True)
                for relative, data in files.items():
                    (session / relative).parent.mkdir(parents=True, exist_ok=True)
                    (session / relative).write_bytes(data)
            if aside.exists():
                _remove_tree(aside)
        path = self.path_for(snapshot.case_id)
        current = path.read_bytes() if path.exists() else None
        if current == snapshot.case_json:
            return
        if snapshot.case_json is None:
            path.unlink()
            return
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".case.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(snapshot.case_json)
            _replace_with_retry(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def save(self, case: Case) -> pathlib.Path:
        path = self.path_for(case.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot leave a truncated case.
        # The temporary file is unique: two writers sharing "case.json.tmp"
        # interleaved into invalid JSON.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".case.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(case), indent=1))
            _replace_with_retry(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return path

    def _same_directory(self, recorded: object, requested: str) -> bool:
        """Does the id the file names reach the directory it was read from?

        On a case-insensitive filesystem (Windows, macOS) "PAIR" and "pair"
        are one directory. Compared exactly, a case audited as PAIR could not
        be loaded as pair: `show --case pair` was refused, and a sweep that
        met PAIR.txt and pair.pdf treated the answered case as corrupt and
        discarded it. An id that differs only in letter case is accepted when
        it names the very same directory - so a write by the id on file lands
        where the case was read - and refused anywhere it would not, which is
        the redirect the exact check exists to stop.
        """
        if not isinstance(recorded, str) or recorded.casefold() != requested.casefold():
            return False
        try:
            return os.path.samefile(self.dir_for(recorded), self.dir_for(requested))
        except (OSError, ValueError):
            return False

    def load(self, case_id: str) -> Case:
        """Read and validate a case. The Case carries the id as the file records it,
        which differs from `case_id` only in letter case (see _same_directory)."""
        path = self.path_for(case_id)
        if not path.exists():
            raise FileNotFoundError(f"no such case: {case_id} (looked in {path})")
        # Not only JSONDecodeError. Bytes that are not UTF-8, or a number of
        # more than 4,300 digits, raise a plain ValueError, and a file of a few
        # thousand "[" raises RecursionError: show and resume printed a
        # traceback, and a sweep with --fresh would not discard such a case.
        try:
            raw = json.loads(_read_with_retry(path))
        except (ValueError, RecursionError) as exc:
            raise CaseCorrupt(f"case {case_id} is not valid JSON: {type(exc).__name__}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CaseCorrupt(f"case {case_id} is not a JSON object")
        version = raw.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise CaseCorrupt(
                f"case {case_id} has schema_version {version!r}, this build expects "
                f"{SCHEMA_VERSION}. Refusing to guess at an incompatible case; re-audit it."
            )
        known = set(Case.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise CaseCorrupt(f"case {case_id} has unexpected fields: {sorted(unknown)}")
        if raw.get("status") not in STATUSES:
            raise CaseCorrupt(f"case {case_id} has an unknown status {raw.get('status')!r}")
        # Every node loads a case by its directory and saves it by the id the
        # file names, so a case.json edited to name another case wrote into
        # that case and destroyed its result. The file must name the case it
        # was read for.
        if raw.get("case_id") != case_id and not self._same_directory(raw.get("case_id"), case_id):
            raise CaseCorrupt(f"case {case_id} names a different case id {raw.get('case_id')!r}")
        _check_shape(case_id, raw)
        try:
            case = Case(**raw)
            decisions = case.load_decisions()
            case.load_trace()
        except (TypeError, KeyError, ValueError) as exc:
            raise CaseCorrupt(f"case {case_id} has malformed content: {exc}") from exc
        # Values, not just shape. A hand-edited "percent": "ten" made a later
        # node raise mid-run, and a side of "north" was silently treated as
        # unpaired - a changed result from a tampered file.
        for index, d in enumerate(decisions):
            problem = None
            if type(d.percent) is not int or not 0 <= d.percent <= 100:
                problem = f"percent {d.percent!r}"
            elif d.extremity_group not in ("upper", "lower", "none", "unknown"):
                problem = f"extremity group {d.extremity_group!r}"
            elif d.laterality not in ("left", "right", "both", "unknown"):
                problem = f"side {d.laterality!r}"
            elif d.confidence is not None and not _is_unit_interval(d.confidence):
                problem = f"confidence {d.confidence!r}"
            if problem:
                raise CaseCorrupt(f"case {case_id} decision [{index}] has an impossible {problem}")
        for value in (case.stated_combined, case.recomputed_degree, case.recomputed_combined,
                      case.alternative_degree, *case.possible_degrees):
            if value is not None and not _is_percentage(value):
                raise CaseCorrupt(f"case {case_id} has an impossible percentage {value!r}")
        if any(not 0 <= i < len(decisions) for i in case.immaterial_unknowns):
            raise CaseCorrupt(f"case {case_id} lists an unknown fact for a condition it does not have")
        _check_as_read(case_id, case, decisions)
        _check_provenance(case_id, case, decisions)
        _check_possibilities(case_id, case, decisions)
        _check_result(case_id, case, decisions)
        return case


def _check_as_read(case_id: str, case: Case, decisions: list[Decision]) -> None:
    """The values read from the letter are recorded alike in the facts, the ratings and the trace.

    _check_result ties a figure to the stored facts, but the stated value is
    the other half of every verdict. A case.json whose stated_combined alone
    was edited from 70 to 80 turned "POTENTIAL DISCREPANCY" into "NO
    DISCREPANCY FOUND" at exit 0, directly under a trace that still said
    "Stated combined evaluation: 70%"; a percentage edited in the facts alone
    was computed as "the evaluations as printed". The extract and classify
    nodes write each value into the facts and the trace in the same save, so
    the two must agree. A letter the extract node could not read is
    "unparsed" and nothing else: marked "ready", `resume` reported "Recheck
    computes 0%" under "Extraction FAILED". As in _check_result, a file edited
    consistently in every place - the trace entries removed or edited too -
    gives a consistent report of false values, which only re-reading the
    letter can catch.
    """
    if case.has_trace_action("Extraction FAILED") and case.status != "unparsed":
        raise CaseCorrupt(f"case {case_id} records that its letter could not be read, but its status is "
                          f"{case.status!r}")
    stated = [e.get("value") for e in case.trace if e.get("action") == "Stated combined evaluation"]
    if len(stated) > 1 or any(v != f"{case.stated_combined}%" for v in stated):
        raise CaseCorrupt(f"case {case_id} records a stated combined evaluation of {case.stated_combined!r}, but its "
                          f"trace records {', '.join(map(str, stated))} as printed in the letter")
    facts = [(d.condition, d.percent, d.evidence) for d in decisions]
    extracted = [(e.get("detail"), e.get("value"), e.get("evidence")) for e in case.trace
                 if e.get("action") == "Extracted rating"]
    if extracted and extracted != [(c, f"{p}%", ev) for c, p, ev in facts]:
        raise CaseCorrupt(f"case {case_id} has conditions or evaluations that differ from the ones its trace "
                          f"records as extracted from the letter")
    if case.ratings and decisions and [
        (r.get("condition"), r.get("percent"),
         f"line {r.get('source_line_number')}" if r.get("source_line_number") else None) for r in case.ratings
    ] != facts:
        raise CaseCorrupt(f"case {case_id} has conditions or evaluations that differ from the ratings extracted "
                          f"from its letter")


def _check_provenance(case_id: str, case: Case, decisions: list[Decision]) -> None:
    """Every fact must be one its recorded owner could have established.

    _check_result re-derives a figure from the stored facts, so the facts are
    what a tamper changes - and who established them is what the report
    prints beside them. An open question edited to give the knee of unstated
    side "left", attributed to the letter, and marked "ready" was finished by
    `resume` with no answer: POTENTIAL DISCREPANCY, 80%, "Applying 38 CFR 4.25
    and 4.26 to the evaluations as printed", exit 0, and "side: left
    [letter]" for a name that states no side.

    Deterministic code owns the side a name states and the lexicon's group,
    and both are functions of the name, so they are read again here rather
    than trusted. A fact attributed to the reviewer must be the answer on
    file; one attributed to the model must be for a name the lexicon does not
    recognise, sendable to the model, and pass the confidence floor and the
    veto. What a model said cannot be checked after the fact, and a file
    edited to name a different condition is a consistent report of false
    facts (see _check_result).
    """
    from recheck.classify import _letters_outside_latin1, _markers_in, _unsendable, derive_laterality
    from recheck.extract.deterministic import _classify_extremity
    from recheck.schema import CONFIDENCE_FLOOR

    for key in case.human_answers:
        if int(key) >= len(decisions):
            raise CaseCorrupt(f"case {case_id} has an answer for condition {key[:12]}, which it does not have")
    for index, d in enumerate(decisions):
        lexicon = _classify_extremity(d.condition)
        stated = derive_laterality(d.condition)
        answer = case.human_answers.get(str(index))
        problem = None
        if (d.extremity_group == "unknown") != (d.group_by is None):
            problem = f"extremity group {d.extremity_group!r} established by {_actor_name(d.group_by)}"
        elif (d.laterality == "unknown") != (d.side_by is None):
            problem = f"side {d.laterality!r} established by {_actor_name(d.side_by)}"
        elif lexicon != "unrecognised" and (d.extremity_group, d.group_by) != (lexicon, Actor.DETERMINISTIC):
            problem = f"extremity group {d.extremity_group!r} for a name the lexicon reads as {lexicon!r}"
        elif lexicon == "unrecognised" and d.group_by is Actor.DETERMINISTIC:
            problem = "extremity group attributed to the lexicon, which does not recognise the name"
        elif d.group_by is Actor.AI and (
            d.confidence is None or d.confidence < CONFIDENCE_FLOOR
            or _unsendable(d.condition) is not None
            or (d.extremity_group == "none" and (_markers_in(d.condition) or _letters_outside_latin1(d.condition)))
        ):
            problem = "extremity group attributed to a model classification Recheck would not have used"
        elif d.side_by is Actor.AI:
            problem = "side attributed to a model, which never establishes one"
        elif stated != "unknown" and d.laterality != stated and not (d.extremity_group == "none"
                                                                    and d.laterality == "unknown"):
            problem = f"side {d.laterality!r} for a name that states {stated!r}"
        elif stated == "unknown" and d.side_by is Actor.DETERMINISTIC:
            problem = f"side {d.laterality!r} attributed to the letter, which does not state one for this name"
        elif Actor.HUMAN in (d.group_by, d.side_by) and answer is None:
            problem = "a fact attributed to the reviewer, with no answer on file"
        elif answer is not None and answer != f"{d.extremity_group}-{d.laterality}":
            problem = f"an answer on file ({answer[:40]!r}) that is not its extremity group and side"
        if problem:
            raise CaseCorrupt(f"case {case_id} decision [{index}] has {problem}")


def _actor_name(actor: Actor | None) -> str:
    return actor.value if actor else "nobody"


#: The trace action the compute node records when 4.26(d) decides a result.
RULE_426D_ACTION = "38 CFR 4.26(d) decides this result"


def _check_result(case_id: str, case: Case, decisions: list[Decision]) -> None:
    """A figure on file must be the one 38 CFR 4.25 and 4.26 give for the facts on file.

    The report and the triage print recomputed_degree as the engine's output.
    A case.json edited to status "complete" and 90% was printed as "Applying
    38 CFR 4.25 and 4.26 ... gives 90%" for a letter whose facts give 70% or
    80% and whose compute node never ran. The compute node already refuses to
    trust committed state; this is the same rule for everything that reads a
    result. It ties the figure to the stored facts - someone who also edits
    the facts gets a consistent report of false facts, which only re-reading
    the letter can catch.
    """
    figures = (case.recomputed_combined, case.recomputed_degree, case.alternative_degree)
    computed = [e for e in case.trace if _is_result_entry(e)]
    if case.status != "complete":
        if any(v is not None for v in figures) or case.bilateral_applied:
            raise CaseCorrupt(f"case {case_id} carries a recomputed result but its status is {case.status!r}")
        if computed:
            raise CaseCorrupt(f"case {case_id} has arithmetic in its trace but its status is {case.status!r}")
        return
    try:
        derived = _derive(decisions)
    except ValueError as exc:
        # The engine refuses facts it is not verified for (more bilateral
        # disabilities than its 4.26(d) search covers). Raised from here, it
        # was not CaseCorrupt, and a sweep that met such a file stopped
        # without a triage for any document.
        raise CaseCorrupt(f"case {case_id} has facts the 38 CFR engine refuses: {exc}") from exc
    if derived is None:
        raise CaseCorrupt(f"case {case_id} is marked complete, but the facts on file could change its result")
    final, combined, applied, alternative, decided_by_426d, steps, notes = derived
    recorded = (case.recomputed_degree, case.recomputed_combined, case.bilateral_applied, case.alternative_degree)
    if recorded != (final, combined, applied, alternative):
        raise CaseCorrupt(f"case {case_id} records a result that 38 CFR 4.25 and 4.26 do not give for its facts")
    # The 4.26(d) effective-date caveat is read from this trace entry; it must
    # be there exactly when 4.26(d) decides the result.
    if case.has_trace_action(RULE_426D_ACTION) != decided_by_426d:
        raise CaseCorrupt(f"case {case_id} has a trace that disagrees with its result about 38 CFR 4.26(d)")
    # The full report prints the decision trace, figures included. An entry
    # edited to "Final degree of disability: 90%" printed that line under the
    # correct 80% verdict. Every figure the compute node wrote into the trace
    # must be the one the engine gives for these facts.
    if not _trace_agrees(computed, final, combined, steps, notes):
        raise CaseCorrupt(f"case {case_id} has a decision trace whose arithmetic 38 CFR 4.25 and 4.26 do not give "
                          f"for its facts")


#: Trace actions the compute node records from the evaluation, and nothing else does.
FINAL_DEGREE_ACTION = "Final degree of disability"
RESULT_ACTIONS = (FINAL_DEGREE_ACTION, "Arithmetic", "Note")

# Every action a Recheck node writes into a trace. A forged entry under any
# other name ("Final degree: 90%", "Result: VA erred") printed in the report
# at exit 0, because only near-copies of the result actions were checked.
# tests/test_rt_final.py re-derives this set from the source, so it cannot
# drift from what the nodes actually write.
TRACE_ACTIONS = frozenset({
    "Extraction FAILED", "Stated combined evaluation", "Extracted rating", "Extremity group",
    "Extremity group UNKNOWN", "Classification REJECTED", "Classification NOT USED", "Side",
    "Human answer REJECTED", "Answer", "Assessment", "Unknown facts cannot change the result",
    "Question for a reviewer", "Result UNDETERMINED", "Arithmetic REFUSED",
    "Arithmetic shown with an assumed fact", RULE_426D_ACTION, "Bilateral factor applies", "Arithmetic",
    "Note", FINAL_DEGREE_ACTION,
})


def _action_key(action: object) -> str:
    """An action name with case, spacing and punctuation removed."""
    return re.sub(r"[^0-9a-z]", "", action.casefold()) if isinstance(action, str) else ""


_RESULT_KEYS = {_action_key(a) for a in RESULT_ACTIONS}


def _is_result_entry(entry: dict[str, Any]) -> bool:
    """Does this trace entry present itself as the compute node's arithmetic?

    Matched loosely on purpose: the report prints "<action>: <value>", so an
    entry named "Final degree of disability." or "FINAL DEGREE OF DISABILITY"
    reads exactly like the real one, and must be checked like it.
    """
    return _action_key(entry.get("action")) in _RESULT_KEYS


def _trace_agrees(computed: list[dict[str, Any]], final: int, combined: int,
                  steps: tuple, notes: tuple) -> bool:
    """The result entries of a complete case's trace, checked against the evaluation.

    Every such entry must be spelled exactly as the compute node spells it
    and be DETERMINISTIC. Arithmetic and Note entries must reproduce the
    evaluation's steps and notes exactly and in order (none at all is
    allowed: an absent entry prints no figure). Each final-degree entry must
    name the final degree as its value, and its detail must start its numbers
    with the combined value; the only other numbers it may carry are the 10
    and 5 of the 4.25(a) rounding rule it states.
    """
    if any(e.get("action") not in RESULT_ACTIONS or e.get("actor") != Actor.DETERMINISTIC.value
           for e in computed):
        return False
    arithmetic = tuple((e.get("detail"), e.get("value"), e.get("rule")) for e in computed
                       if e.get("action") == "Arithmetic")
    if arithmetic and arithmetic != steps:
        return False
    written_notes = tuple(e.get("detail") for e in computed if e.get("action") == "Note")
    if written_notes and written_notes != notes:
        return False
    for entry in computed:
        if entry.get("action") != FINAL_DEGREE_ACTION:
            continue
        if entry.get("value") != f"{final}%":
            return False
        numbers = re.findall(r"\d+", entry.get("detail") or "")
        if numbers[:1] != [str(combined)] or any(n not in ("10", "5") for n in numbers[1:]):
            return False
    return True


def _check_possibilities(case_id: str, case: Case, decisions: list[Decision]) -> None:
    """Possible degrees and "nobody was asked" indices on file are the ones assess gives for the facts.

    The report and the triage print both as Recheck's findings. Edited on
    file, an open question read "could be 90%" for a letter whose facts give
    70% or 80%, and a complete case said "unknown facts for [1] could not
    change this result, so nobody was asked" of a condition with no unknown
    fact.
    """
    m = _assessed(decisions)
    if case.immaterial_unknowns and case.immaterial_unknowns != list(m.unknown):
        raise CaseCorrupt(f"case {case_id} lists an unknown fact its conditions do not have")
    possible = case.possible_degrees
    if case.status == "awaiting_human":
        wrong = possible != list(m.possible)
    elif case.status == "ready":
        # Kept from the question the reviewer answered; the answers pick one of them.
        wrong = bool(possible) and m.settled and not set(m.possible) <= set(possible)
    elif case.status == "complete":
        # Its result is re-derived by _check_result; the report prints no possible degrees for it.
        wrong = False
    else:
        # None at all is allowed: an undetermined case whose possibilities were
        # not enumerated has none, and a case that never reached assess has none.
        wrong = bool(possible) and possible != list(m.possible)
    if wrong:
        raise CaseCorrupt(f"case {case_id} records possible final degrees {possible[:8]} that its facts do not give")


_ASSESSED: dict[tuple[tuple[int, str, str], ...], Any] = {}


def _assessed(decisions: list[Decision]):
    """materiality.assess for these facts, cached: a sweep loads each case several times,
    and only the percentage, group and side of each condition enter it."""
    key = tuple((d.percent, d.extremity_group, d.laterality) for d in decisions)
    if key not in _ASSESSED:
        if len(_ASSESSED) > 512:
            _ASSESSED.clear()
        _ASSESSED[key] = assess(decisions)
    return _ASSESSED[key]


_DERIVED: dict[tuple[tuple[int, str, str], ...], tuple | None] = {}


def _derive(decisions: list[Decision]) -> tuple | None:
    """(final, combined, factor applied, alternative, 4.26(d) decides, steps, notes) for
    the facts on file, or None if they do not settle a result.

    Steps are (detail, running value, rule), as the compute node writes them
    into the trace. Cached, because a sweep loads each case several times and
    only the percentage, group and side of each condition enter the
    arithmetic and its wording.
    """
    key = tuple((d.percent, d.extremity_group, d.laterality) for d in decisions)
    if key not in _DERIVED:
        m = _assessed(decisions)
        # Not only unknown facts: a single evaluation naming both sides can
        # leave M21-1's reading open with every fact known.
        if not m.settled:
            result = None
        else:
            evaluation, _ = evaluate_for_report(decisions)
            result = (evaluation.final_degree, evaluation.combined_value, evaluation.bilateral_applied,
                      evaluation.alternative_final_degree, bool(evaluation.excluded_under_426d),
                      tuple((s.detail, s.running_after, s.rule) for s in evaluation.steps),
                      tuple(evaluation.notes))
        if len(_DERIVED) > 512:
            _DERIVED.clear()
        _DERIVED[key] = result
    return _DERIVED[key]


_ACTORS = {a.value for a in Actor}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _check_shape(case_id: str, raw: dict[str, Any]) -> None:
    """Every field has the type the code that reads it assumes.

    A number where a list or text belongs (possible_degrees: 5, a condition of
    5, a timeline entry with no pid) crashed show, resume and - because the
    triage renders every case - the whole sweep, with a traceback.
    """

    def bad(what: str) -> CaseCorrupt:
        return CaseCorrupt(f"case {case_id} has malformed content: {what}")

    def optional_str(value: Any) -> bool:
        return value is None or isinstance(value, str)

    def optional_int(value: Any) -> bool:
        return value is None or type(value) is int

    if not isinstance(raw.get("source_path"), str) or not raw["source_path"]:
        raise bad("source_path must be text")
    for name in ("classifier", "status"):
        if name in raw and not isinstance(raw[name], str):
            raise bad(f"{name} must be text")
    for name in ("rejected_answer", "undetermined_reason", "bilateral_note"):
        if not optional_str(raw.get(name)):
            raise bad(f"{name} must be text or null")
    sha = raw.get("document_sha256")
    if sha is not None and not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        raise bad("document_sha256 must be 64 lowercase hex digits or null")
    for name in ("stated_combined", "recomputed_combined", "recomputed_degree", "alternative_degree"):
        if not optional_int(raw.get(name)):
            raise bad(f"{name} must be a whole number or null")
    if "bilateral_applied" in raw and type(raw["bilateral_applied"]) is not bool:
        raise bad("bilateral_applied must be true or false")
    for name in ("ratings", "decisions", "trace", "timeline", "possible_degrees", "immaterial_unknowns"):
        if name in raw and not isinstance(raw[name], list):
            raise bad(f"{name} must be a list")
    for name in ("possible_degrees", "immaterial_unknowns"):
        if any(type(v) is not int for v in raw.get(name, [])):
            raise bad(f"{name} must list whole numbers")
    answers = raw.get("human_answers", {})
    if not isinstance(answers, dict) or not all(
        isinstance(k, str) and k.isascii() and k.isdigit() and isinstance(v, str) for k, v in answers.items()
    ):
        raise bad("human_answers must map condition indices to text")
    if not all(isinstance(r, dict) for r in raw.get("ratings", [])):
        raise bad("each rating must be an object")
    for index, d in enumerate(raw.get("decisions", [])):
        if not isinstance(d, dict):
            raise bad(f"decision [{index}] must be an object")
        missing = {"condition", "percent", "extremity_group", "laterality"} - set(d)
        if missing:
            raise bad(f"decision [{index}] is missing {sorted(missing)}")
        if not isinstance(d["condition"], str):
            raise bad(f"decision [{index}] condition must be text")
        for name in ("group_by", "side_by"):
            # Type first: a list or object here raised "unhashable type" from
            # the membership test, a traceback that stopped the whole sweep.
            if d.get(name) is not None and not _is_actor(d.get(name)):
                raise bad(f"decision [{index}] {name} {d.get(name)!r} is not an actor")
        for name in ("note", "evidence"):
            if not optional_str(d.get(name)):
                raise bad(f"decision [{index}] {name} must be text or null")
    for index, e in enumerate(raw.get("trace", [])):
        if not isinstance(e, dict):
            raise bad(f"trace entry [{index}] must be an object")
        if not _is_actor(e.get("actor")):
            raise bad(f"trace entry [{index}] has an unknown actor {e.get('actor')!r}")
        if not (isinstance(e.get("action"), str) and isinstance(e.get("detail"), str)):
            raise bad(f"trace entry [{index}] action and detail must be text")
        if e.get("action") not in TRACE_ACTIONS:
            raise bad(f"trace entry [{index}] has an action no Recheck node writes: {e.get('action')[:60]!r}")
        value = e.get("value")
        if not (value is None or isinstance(value, str) or type(value) is int):
            raise bad(f"trace entry [{index}] value must be text, a whole number or null")
        for name in ("evidence", "rule"):
            if not optional_str(e.get(name)):
                raise bad(f"trace entry [{index}] {name} must be text or null")
        if e.get("confidence") is not None and not _is_unit_interval(e.get("confidence")):
            raise bad(f"trace entry [{index}] confidence must be between 0 and 1")
    for index, step in enumerate(raw.get("timeline", [])):
        if not (isinstance(step, dict) and isinstance(step.get("node"), str) and type(step.get("pid")) is int):
            raise bad(f"timeline entry [{index}] must name a node and a process id")


def _is_actor(value: Any) -> bool:
    return isinstance(value, str) and value in _ACTORS


def _is_percentage(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 100


def _is_unit_interval(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _acquire(path: pathlib.Path, case_id: str) -> int:
    busy = f"case {case_id} is busy in another process; nothing was changed. Try again when it has finished."
    for _ in range(20):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise CaseBusy(busy) from exc
        if os.name != "nt":
            try:
                current = os.stat(path).st_ino
            except FileNotFoundError:
                current = None
            if current != os.fstat(fd).st_ino:
                os.close(fd)  # locked a file the previous holder had just removed; try again
                continue
        return fd
    raise CaseBusy(busy)


#: Whether lock() removes its file before closing it (see lock()).
_UNLINK_WHILE_LOCKED = os.name != "nt"


def _release(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            with contextlib.suppress(OSError):
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def _replace_with_retry(source: str | pathlib.Path, target: pathlib.Path) -> None:
    """os.replace, retried briefly: on Windows it fails while another process
    has the target open for reading (a `show`, or a sweep's status view)."""
    for attempt in range(40):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


def _read_with_retry(path: pathlib.Path) -> str:
    """Read case.json, retried briefly: on Windows, opening it while another
    process replaces it fails with "Permission denied" for a moment."""
    for attempt in range(40):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)
    raise AssertionError("unreachable")


def _remove_tree(path: pathlib.Path) -> None:
    """Best-effort removal of a discarded case; read-only files are made writable first."""
    try:
        shutil.rmtree(path)
        return
    except OSError:
        pass
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(root, name), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    shutil.rmtree(path, ignore_errors=True)


class CaseCorrupt(Exception):
    """A case file exists but cannot be trusted."""


def _validate_case_id(case_id: str) -> None:
    """Reject anything that could escape the store directory.

    Case ids arrive from the command line, so "../../etc/passwd" and absolute
    paths must not be usable as identifiers.

    Nor may one start with "-": every printed command is `--case <id>`, and
    `resume --case -x` (or a sweep id of "--help" from a file named
    --help.txt) is parsed as an option, so the command Recheck printed failed
    when pasted.
    """
    if not isinstance(case_id, str) or not case_id or len(case_id) > 64:
        raise ValueError("case id must be 1-64 characters")
    if not all(c.isascii() and (c.isalnum() or c in "-_") for c in case_id):
        raise ValueError(
            f"case id may contain only ASCII letters, digits, '-' and '_'; got {case_id!r}"
        )
    if case_id.startswith("-"):
        raise ValueError(f"case id may not start with '-'; got {case_id!r}")
    if case_id.lower() in _WINDOWS_RESERVED:
        raise ValueError(f"case id {case_id!r} is a reserved device name on Windows")


_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
