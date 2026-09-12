"""Live model provider adapters.

Deliberately not wired yet. The zero-model path is the architecture; a live
provider is an adapter at the edge, and adding one is the LAST step of the
build rather than a prerequisite for any other part.

Keeping this a loud failure rather than a silent fallback matters: a demo
that quietly ran against no classifier while claiming a live model would be
dishonest, and the CLI already has an explicit --scripted flag for the
zero-cost path.
"""

from __future__ import annotations


def build_agent_factory(provider: str):
    raise NotImplementedError(
        f"live provider {provider!r} is not wired in this build. "
        f"Use --scripted for the zero-model path, which exercises the same "
        f"Strands structured-output code path with no network and no cost."
    )
