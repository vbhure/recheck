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

  ONE CALL PER LETTER, BOUNDED. The model classifies a batch of condition
  names in a single structured call, and only when the lexicon abstains on
  something. `limits={"turns": 1}` in recheck.classify bounds retries (Strands
  otherwise retries a structured output that fails validation), and a
  wall-clock budget bounds time.

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

    # The wall-clock budget in recheck.classify cancels the call, but a client
    # blocked in a socket read can hold the process open after that. So the
    # same budget is given to each provider's own client, with retries off:
    # one call per letter means one attempt.
    if config.provider == "bedrock":
        from botocore.config import Config

        kwargs: dict[str, Any] = {
            "model_id": config.model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "boto_client_config": Config(
                read_timeout=config.timeout_s,
                connect_timeout=min(10.0, config.timeout_s),
                # total_max_attempts counts the first try; max_attempts would allow a retry
                retries={"total_max_attempts": 1, "mode": "standard"},
            ),
        }
        if config.region:
            kwargs["region_name"] = config.region
        return cls(**kwargs)
    if config.provider == "anthropic":
        return cls(
            client_args={"timeout": config.timeout_s, "max_retries": 0},
            model_id=config.model_id,
            max_tokens=config.max_tokens,
            params={"temperature": config.temperature},
        )
    if config.provider == "ollama":
        return cls(host=config.host, model_id=config.model_id, temperature=config.temperature,
                   ollama_client_args={"timeout": config.timeout_s})
    raise ProviderNotConfigured(f"no constructor mapping for {config.provider!r}")


AgentFactory = Callable[[], Any]


def build_agent_factory(
    provider: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AgentFactory:
    """Return a factory producing the classification agent for this provider.

    The factory carries `timeout_s`; recheck.classify enforces it as a
    wall-clock budget around the one structured call (asyncio.wait_for plus
    Strands' cancel_signal), and bounds turns with limits={"turns": 1}.

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

    def make() -> Any:
        from strands import Agent

        from recheck.classify import SYSTEM_PROMPT

        return Agent(
            model=build_model(config),
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,  # framework chatter is not product output
        )

    make.config = config  # type: ignore[attr-defined]
    make.timeout_s = config.timeout_s  # type: ignore[attr-defined]
    return make
