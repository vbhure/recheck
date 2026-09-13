"""Shared helpers for the test suite. Not a test module.

Letters used to exercise the model path are written into tmp_path by the
tests themselves rather than read from fixtures/letters/07-08, which are
regenerated independently. Condition names chosen for those tests are
INVENTED ("Zorblatt syndrome") on purpose: the tests are about the mechanics
of the model boundary, and a real term could be absorbed by a future lexicon
expansion, silently turning a model-path test into a lexicon-path test.
require_lexicon_abstains() makes that failure loud instead of silent.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

from strands import Agent

from recheck.case import CaseStore
from recheck.classify import SYSTEM_PROMPT, _markers_in
from recheck.extract.deterministic import _classify_extremity
from recheck.graph import build_graph, open_case, outstanding_interrupt
from recheck.models.scripted import ScriptedModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
LETTERS = ROOT / "fixtures" / "letters"

EXIT_OK, EXIT_AWAITING_HUMAN, EXIT_CANNOT_PROCEED = 0, 2, 3

# Invented condition names. See the module docstring for why.
UNLISTED = "Zorblatt syndrome"                     # no anatomy, no extremity vocabulary
UNLISTED_NERVE = "Neuritis of the zorblatt nerve"  # extremity vocabulary, no lexicon anatomy


def require_lexicon_abstains(*names: str) -> None:
    for name in names:
        assert _classify_extremity(name) == "unrecognised", (
            f"test precondition: the lexicon now recognises {name!r}; this test is about the "
            f"model path and needs a name the lexicon abstains on"
        )


def require_no_extremity_markers(name: str) -> None:
    assert not _markers_in(name), f"test precondition: {name!r} contains extremity vocabulary"


# --------------------------------------------------------------------------
# Letters
# --------------------------------------------------------------------------

def tabular_letter(path: pathlib.Path, rows: list[tuple[str, int]], stated: int) -> pathlib.Path:
    """A minimal tabular rating decision the deterministic parser reads."""
    lines = ["*** SYNTHETIC TEST LETTER - NOT A REAL VA DECISION ***", "", "RATING DECISION", ""]
    for number, (name, percent) in enumerate(rows, 1):
        lines.append(f"  {number}. {name} {'.' * max(3, 48 - len(name))} {percent}%")
    lines += ["", f"COMBINED EVALUATION FOR COMPENSATION: {stated}%", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Scripted model: the real Strands structured-output path, no network
# --------------------------------------------------------------------------

def item(condition: str, group: str, confidence: float = 0.9, **extra: Any) -> dict:
    return {"condition": condition, "extremity_group": group, "confidence": confidence, **extra}


def batch(*items: dict) -> dict:
    return {"classifications": list(items)}


def scripted(payload: Any = None, *, responder=None, stop_reason: str = "tool_use",
             timeout_s: float | None = None, model_cls=ScriptedModel):
    """An agent factory whose Strands Agent runs a ScriptedModel.

    Every model it builds is kept on `factory.models`, so a test can inspect
    exactly what was sent (ScriptedModel.calls) and how many calls were made.
    """
    models: list[ScriptedModel] = []

    def make():
        model = model_cls(payload=payload, responder=responder, stop_reason=stop_reason)
        models.append(model)
        return Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)

    make.models = models  # type: ignore[attr-defined]
    if timeout_s is not None:
        make.timeout_s = timeout_s  # type: ignore[attr-defined]
    return make


def prompt_text(model: ScriptedModel) -> str:
    """Everything the model was sent in the user turn(s), flattened."""
    return "\n".join(
        block.get("text", "")
        for call in model.calls
        for message in call["messages"]
        for block in message.get("content", [])
        if isinstance(block, dict)
    )


# --------------------------------------------------------------------------
# Running the graph in-process
# --------------------------------------------------------------------------

def run_audit(store_root: pathlib.Path, case_id: str, letter: pathlib.Path, factory=None,
              classifier: str = "none"):
    store = CaseStore(store_root)
    open_case(store, case_id, str(letter), classifier)
    result = asyncio.run(build_graph(store, case_id, str(letter), factory).invoke_async("audit"))
    return store, result


def answer(store: CaseStore, case_id: str, response: Any):
    """Resume from the persisted Strands session with a reviewer's answer."""
    case = store.load(case_id)
    graph = build_graph(store, case_id, case.source_path, None)
    pending = outstanding_interrupt(graph)
    assert pending is not None, "expected an outstanding interrupt in the restored session"
    return asyncio.run(
        graph.invoke_async([{"interruptResponse": {"interruptId": pending.id, "response": response}}])
    )


def nodes_run(store: CaseStore, case_id: str) -> list[str]:
    return [step["node"] for step in store.load(case_id).timeline]


def actions(store: CaseStore, case_id: str) -> list[str]:
    return [entry["action"] for entry in store.load(case_id).trace]


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------

def main(*args: Any, store: pathlib.Path) -> tuple[int, str, str]:
    """recheck.cli.main in this process: fast, for exit codes and output."""
    from recheck.cli import main as cli_main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli_main(["--store", str(store), *[str(a) for a in args]])
    return code, out.getvalue(), err.getvalue()


def cli(*args: Any, store: pathlib.Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """recheck.cli in a brand-new interpreter: for anything about process boundaries."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "recheck.cli", "--store", str(store), *[str(a) for a in args]],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ROOT), timeout=timeout,
    )


def case_json(store_root: pathlib.Path, case_id: str) -> dict:
    return json.loads((store_root / case_id / "case.json").read_text(encoding="utf-8"))
