"""ScriptedModel - a deterministic Model implementation for the zero-model path.

This is not a mock that bypasses Strands. It implements `stream()` and emits
the same tool-use event sequence a real provider emits, so structured output
is parsed and validated by Strands' own machinery. That matters: the
adversarial tests in tests/test_boundary_adversarial.py drive malformed
payloads through the REAL validation path, not around it.

It exists so that:
  - the full graph runs with no network, no credentials and no paid inference
  - the demo has an emergency path if a provider is unavailable
  - tests are fast and deterministic

The real provider is an adapter selected at the edge (see recheck.models.factory).
It is not the architecture.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, Callable, TypeVar

from strands.models.model import Model
from strands.tools.structured_output import convert_pydantic_to_tool_spec

T = TypeVar("T")

# A responder receives the tool name and the outgoing messages, and returns
# the raw payload the "model" claims to produce. Returning a str allows tests
# to emit deliberately invalid JSON.
Responder = Callable[[str, list[dict[str, Any]]], Any]


class ScriptedModel(Model):
    """Replays a scripted structured-output payload through the real code path."""

    def __init__(
        self,
        responder: Responder | None = None,
        payload: Any | None = None,
        *,
        stop_reason: str = "tool_use",
        model_id: str = "scripted",
    ) -> None:
        if responder is None and payload is None:
            raise ValueError("ScriptedModel needs either a responder or a payload")
        self._responder = responder
        self._payload = payload
        self._stop_reason = stop_reason
        self._config: dict[str, Any] = {"model_id": model_id}
        self.calls: list[dict[str, Any]] = []

    # -- Model ABC -------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[dict[str, Any]]:
        tool_name = "structured_output"
        if tool_specs:
            tool_name = tool_specs[0].get("name", tool_name)

        self.calls.append(
            {"tool": tool_name, "messages": messages, "system_prompt": system_prompt}
        )

        payload = (
            self._responder(tool_name, messages)
            if self._responder is not None
            else self._payload
        )
        raw = payload if isinstance(payload, str) else json.dumps(payload)

        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "scripted-1", "name": tool_name}}
            }
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": raw}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": self._stop_reason}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }

    async def structured_output(
        self,
        output_model: type[T],
        prompt: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Delegate to the inherited stream-based implementation.

        Mirrors how real providers implement this, so the tool spec is derived
        from the Pydantic model exactly as in production.
        """
        tool_spec = convert_pydantic_to_tool_spec(output_model)
        from strands.event_loop import streaming

        event: dict[str, Any] = {}
        async for event in streaming.process_stream(
            self.stream(prompt, [tool_spec], system_prompt, **kwargs)
        ):
            yield event

        stop_reason, message, _, _ = event["stop"]
        if stop_reason != "tool_use":
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "tool_use".')
        for block in message["content"]:
            if "toolUse" in block:
                yield {"output": output_model(**block["toolUse"]["input"])}
                return
        raise ValueError("scripted model emitted no toolUse block")
