"""Live model provider seam.

A provider is an ADAPTER at the edge of the system, not a dependency of it.
Everything in this module can be built, configured and verified without ever
making an inference call, and the zero-model path remains the canonical
verification route.

Design constraints, all of them load-bearing:

  SECRETS ARE NOT OURS. Recheck never reads, stores, logs or prints a
  credential. Each provider resolves its own via its own standard mechanism
  (the AWS credential chain, ANTHROPIC_API_KEY). Preflight reports only
  whether a credential could be RESOLVED - never a value, never a prefix.

  ONE CALL PER LETTER. The model classifies a batch of condition names in a
  single structured call. There is no agent loop, no tool use, and no
  re-prompting: `limits={"turns": 1}` in recheck.classify bounds it, because
  Strands otherwise retries a structured output that fails validation.

  MINIMAL PROMPT. Only condition names are sent. The letter, the
  percentages, the stated combined evaluation and the file number never
  leave the machine - they are not needed for the classification task, so
  they are not transmitted.

  FAILURE IS UNIFORM. Timeout, API error, unavailable provider, malformed
  output and unsupported content all produce the same outcome: the affected
  conditions route to human review. Nothing is fabricated or substituted.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Mapping

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_TOKENS = 1024
# Classification is a labelling task, not a generative one. Low temperature
# is about reproducibility, not quality.
DEFAULT_TEMPERATURE = 0.0

#: provider id -> (import path, class name, pip extra, default model id)
PROVIDERS: dict[str, tuple[str, str, str | None, str]] = {
    "bedrock": (
        "strands.models.bedrock",
        "BedrockModel",
        None,  # bundled with strands-agents
        "global.anthropic.claude-haiku-4-5",
    ),
    "anthropic": (
        "strands.models.anthropic",
        "AnthropicModel",
        "strands-agents[anthropic]",
        "claude-haiku-4-5",
    ),
    "ollama": (
        "strands.models.ollama",
        "OllamaModel",
        "strands-agents[ollama]",
        "llama3.2:3b",
    ),
}


class ProviderError(RuntimeError):
    """Any provider-side failure. Callers treat every subclass identically."""


class ProviderNotConfigured(ProviderError):
    """Configuration is absent or invalid. Raised before any network use."""


class ProviderTimeout(ProviderError):
    """The provider did not answer within the configured budget."""


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model_id: str
    region: str | None = None
    host: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    def describe(self) -> str:
        """A one-line summary safe to print. Contains no credential material."""
        where = self.region or self.host or "provider default"
        return (
            f"{self.provider}:{self.model_id} ({where}, timeout {self.timeout_s:g}s, "
            f"max_tokens {self.max_tokens}, temperature {self.temperature:g})"
        )


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def load_config(
    provider: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Build configuration from arguments and environment. No network use.

    Environment variables:
        RECHECK_PROVIDER     bedrock | anthropic | ollama
        RECHECK_MODEL_ID     provider-specific model identifier
        RECHECK_REGION       AWS region (bedrock only)
        RECHECK_OLLAMA_HOST  e.g. http://localhost:11434 (ollama only)
        RECHECK_TIMEOUT_S    wall-clock budget for one classification call
        RECHECK_MAX_TOKENS   output cap

    Credentials are deliberately NOT read here. Providers resolve their own.
    """
    env = os.environ if env is None else env
    name = (provider or env.get("RECHECK_PROVIDER") or "").strip().lower()
    if not name:
        raise ProviderNotConfigured(
            "no provider selected. Pass --model <provider> or set RECHECK_PROVIDER. "
            f"Known providers: {', '.join(sorted(PROVIDERS))}. "
            "For a zero-cost run, use --scripted instead."
        )
    if name not in PROVIDERS:
        raise ProviderNotConfigured(
            f"unknown provider {name!r}. Known providers: {', '.join(sorted(PROVIDERS))}."
        )

    _, _, _, default_model = PROVIDERS[name]
    model_id = (env.get("RECHECK_MODEL_ID") or default_model).strip()
    if not model_id:
        raise ProviderNotConfigured("RECHECK_MODEL_ID resolved to an empty string")

    region = (env.get("RECHECK_REGION") or env.get("AWS_REGION") or "").strip() or None
    host = (env.get("RECHECK_OLLAMA_HOST") or "").strip() or None
    if name == "ollama" and host is None:
        host = "http://localhost:11434"

    return ProviderConfig(
        provider=name,
        model_id=model_id,
        region=region,
        host=host,
        timeout_s=_positive_float(env, "RECHECK_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        max_tokens=int(_positive_float(env, "RECHECK_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        temperature=DEFAULT_TEMPERATURE,
    )


def _positive_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ProviderNotConfigured(f"{key}={raw!r} is not a number") from exc
    if value <= 0:
        raise ProviderNotConfigured(f"{key} must be greater than zero, got {value}")
    return value


def preflight(config: ProviderConfig) -> list[Check]:
    """Verify configuration WITHOUT inference.

    Every check here is free. Credential resolution reads local config and,
    on some AWS setups, instance metadata; neither is a billable operation
    and neither invokes a model. No check prints a credential.
    """
    checks: list[Check] = [
        Check("provider known", True, f"{config.provider}"),
        Check("model id present", bool(config.model_id), config.model_id or "(empty)"),
    ]

    import_path, class_name, extra, _ = PROVIDERS[config.provider]
    try:
        import importlib

        module = importlib.import_module(import_path)
        getattr(module, class_name)
        checks.append(Check("provider SDK importable", True, f"{import_path}.{class_name}"))
    except Exception as exc:
        hint = f"install it with: pip install '{extra}'" if extra else str(exc)[:70]
        checks.append(Check("provider SDK importable", False, hint))
        return checks

    if config.provider == "bedrock":
        if not config.region:
            checks.append(
                Check("region configured", False, "set RECHECK_REGION or AWS_REGION")
            )
        else:
            checks.append(Check("region configured", True, config.region))
        try:
            import boto3

            creds = boto3.Session(region_name=config.region).get_credentials()
            if creds is None:
                checks.append(
                    Check(
                        "credentials resolvable",
                        False,
                        "the AWS credential chain resolved nothing; run `aws configure`",
                    )
                )
            else:
                # Report the METHOD, never any key material.
                checks.append(
                    Check("credentials resolvable", True, f"via {creds.method}")
                )
        except Exception as exc:
            checks.append(Check("credentials resolvable", False, f"{type(exc).__name__}"))

    elif config.provider == "anthropic":
        present = bool(os.environ.get("ANTHROPIC_API_KEY"))
        checks.append(
            Check(
                "credentials resolvable",
                present,
                "ANTHROPIC_API_KEY is set" if present else "ANTHROPIC_API_KEY is not set",
            )
        )

    elif config.provider == "ollama":
        checks.append(Check("host configured", bool(config.host), config.host or "(none)"))
        checks.append(
            Check("credentials resolvable", True, "not applicable - local provider")
        )

    checks.append(
        Check("single-call contract", True, "one structured classification call per letter")
    )
    checks.append(
        Check("prompt minimisation", True, "condition names only; no percentages, no letter text")
    )
    return checks


def build_model(config: ProviderConfig) -> Any:
    """Instantiate the provider's Model. Construction makes no inference call."""
    import importlib

    import_path, class_name, extra, _ = PROVIDERS[config.provider]
    try:
        module = importlib.import_module(import_path)
    except ImportError as exc:
        hint = f" Install it with: pip install '{extra}'" if extra else ""
        raise ProviderNotConfigured(
            f"provider {config.provider!r} requires {import_path}, which is not installed.{hint}"
        ) from exc
    cls = getattr(module, class_name)

    if config.provider == "bedrock":
        kwargs: dict[str, Any] = {
            "model_id": config.model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        if config.region:
            kwargs["region_name"] = config.region
        return cls(**kwargs)
    if config.provider == "anthropic":
        return cls(
            model_id=config.model_id,
            max_tokens=config.max_tokens,
            params={"temperature": config.temperature},
        )
    if config.provider == "ollama":
        return cls(host=config.host, model_id=config.model_id, temperature=config.temperature)
    raise ProviderNotConfigured(f"no constructor mapping for {config.provider!r}")


class BoundedAgent:
    """Wraps a Strands Agent with a wall-clock budget.

    Strands bounds TURNS; this bounds TIME. A provider that accepts a
    connection and then stalls would otherwise hang the run, which for a
    tool people invoke from a terminal is indistinguishable from a crash.

    On timeout the underlying invocation is signalled to cancel and
    ProviderTimeout is raised. recheck.classify treats any exception the same
    way: the affected conditions route to human review.
    """

    def __init__(self, agent: Any, timeout_s: float) -> None:
        self._agent = agent
        self._timeout_s = timeout_s

    def __call__(self, prompt: str, **kwargs: Any) -> Any:
        cancel = threading.Event()
        kwargs.setdefault("cancel_signal", cancel)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="recheck-model") as pool:
            future = pool.submit(self._agent, prompt, **kwargs)
            try:
                return future.result(timeout=self._timeout_s)
            except FutureTimeout as exc:
                cancel.set()
                raise ProviderTimeout(
                    f"provider did not answer within {self._timeout_s:g}s"
                ) from exc


AgentFactory = Callable[[], Any]


def build_agent_factory(
    provider: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AgentFactory:
    """Return a factory producing a bounded, single-call classification agent.

    Raises ProviderNotConfigured eagerly, before any run begins, so a
    misconfiguration surfaces immediately rather than mid-workflow.
    """
    config = load_config(provider, env)
    failures = [c for c in preflight(config) if not c.ok]
    if failures:
        summary = "; ".join(f"{c.name}: {c.detail}" for c in failures)
        raise ProviderNotConfigured(
            f"provider {config.provider!r} is not ready - {summary}. "
            f"Run `recheck preflight --model {config.provider}` for detail, "
            f"or use --scripted for a zero-cost run."
        )

    def make() -> BoundedAgent:
        from strands import Agent

        from recheck.classify import SYSTEM_PROMPT

        agent = Agent(
            model=build_model(config),
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,  # framework chatter is not product output
        )
        return BoundedAgent(agent, config.timeout_s)

    make.config = config  # type: ignore[attr-defined]
    return make
