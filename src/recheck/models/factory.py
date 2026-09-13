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

  PREFLIGHT SHOWS WHERE THE CALL GOES. Configuration printed by Recheck has
  any user:password in a URL redacted, and an SDK endpoint override
  (ANTHROPIC_BASE_URL, AWS_ENDPOINT_URL_BEDROCK_RUNTIME, AWS_ENDPOINT_URL) is
  shown, and refused unless it is https or loopback: the provider SDK sends
  the credential to it.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

DEFAULT_TIMEOUT_S = 30.0
# Sized for the largest answer the schema admits: 40 classifications echoing
# 200-character names is about 10,600 characters of JSON, some 3,500 tokens.
# At the earlier 1,024 such a letter was sent and truncated every time - a
# paid call that could only end "unknown" (red team MODEL-RT-P2P6-10). A cap
# is not a spend: a provider bills the tokens produced, not the ceiling.
DEFAULT_MAX_TOKENS = 4096
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
    """Configuration is absent or invalid. Raised before any inference call.

    Not "before any network use": resolving a credential can itself be a
    network lookup (a container credential endpoint the operator configured,
    or EC2 instance metadata when they opt in - see preflight).
    """


def _redact_url(url: str) -> str:
    """The URL with anything before the last "@" of its authority replaced.

    RECHECK_OLLAMA_HOST=https://vso:PASSWORD@host:443 was printed whole by
    preflight and by every live audit (red team SECRETS-F8). Parsing is
    deliberately forgiving: a missing scheme or a malformed port must not let
    the credential through, so no URL parser gets to decide.
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        scheme, rest = "", url
    authority_end = min((i for i in (rest.find(c) for c in "/?#") if i >= 0), default=len(rest))
    authority, tail = rest[:authority_end], rest[authority_end:]
    if "@" not in authority:
        # A password holding "/" or "?" puts the "@" after authority_end.
        if "@" not in rest:
            return url
        authority, tail = rest, ""
    host = authority.rpartition("@")[2]
    return f"{scheme}{sep}***@{host}{tail}"


def _is_https_or_loopback(url: str) -> bool:
    """https to any host, or http to this machine. Anything unparseable is neither."""
    try:
        parts = urlsplit(url)
        scheme, host = parts.scheme.lower(), parts.hostname or ""
        if scheme == "https" and host:
            return True
        return scheme == "http" and (host == "localhost" or ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return False


# The SDK variables that move each provider's endpoint, in the SDK's order of
# precedence. Preflight said "provider default" while ANTHROPIC_BASE_URL sent
# the API key to http://attacker.example (red team SECRETS-F9).
ENDPOINT_VARIABLES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_BASE_URL",),
    "bedrock": ("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_ENDPOINT_URL"),
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model_id: str
    region: str | None = None
    host: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    endpoint: str | None = None           # an SDK endpoint override, if one is set
    endpoint_variable: str | None = None  # the variable that set it

    def describe(self) -> str:
        """A one-line summary safe to print. Contains no credential material."""
        parts = [p for p in (self.region, _redact_url(self.host) if self.host else None) if p]
        if self.endpoint:
            parts.append(f"endpoint {_redact_url(self.endpoint)} from {self.endpoint_variable}")
        where = ", ".join(parts) or "provider default"
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
        RECHECK_PROVIDER     bedrock | anthropic | ollama - the default for
                             `recheck preflight` only. audit and sweep call a
                             live model only when --model is passed, so a
                             variable alone never starts paid calls.
        RECHECK_MODEL_ID     provider-specific model identifier
        RECHECK_REGION       AWS region (bedrock only)
        RECHECK_OLLAMA_HOST  e.g. http://localhost:11434 (ollama only)
        RECHECK_TIMEOUT_S    wall-clock budget for one classification call
        RECHECK_MAX_TOKENS   output cap

    The provider SDKs' own endpoint overrides (ENDPOINT_VARIABLES) are read so
    that preflight can show them; they are not ours to set.

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

    endpoint = endpoint_variable = None
    # botocore ignores its endpoint variables when told to; so does this.
    ignored = name == "bedrock" and str(env.get("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "")).strip().lower() == "true"
    for variable in () if ignored else ENDPOINT_VARIABLES.get(name, ()):
        value = (env.get(variable) or "").strip()
        if value:
            endpoint, endpoint_variable = value, variable
            break

    return ProviderConfig(
        provider=name,
        model_id=model_id,
        region=region,
        host=host,
        timeout_s=_positive_float(env, "RECHECK_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        max_tokens=int(_positive_float(env, "RECHECK_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        temperature=DEFAULT_TEMPERATURE,
        endpoint=endpoint,
        endpoint_variable=endpoint_variable,
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

    Every check here is free, and none invokes a model. No check prints a
    credential.

    AWS credential resolution reads local configuration. It does NOT probe
    EC2 instance metadata unless the operator opts in with
    AWS_EC2_METADATA_DISABLED=false: off EC2, botocore's probe of
    169.254.169.254 was a connection attempt, and a second or two of delay,
    before a refusal that is meant to need no network (red team SECRETS-F6).
    """
    checks: list[Check] = [
        Check("provider known", True, f"{config.provider}"),
        Check("model id present", bool(config.model_id), config.model_id or "(empty)"),
    ]

    if config.provider in ENDPOINT_VARIABLES:
        # Checked before the SDK import so it is reported whether or not the
        # SDK is installed: it is configuration, and the credential follows it.
        if config.endpoint is None:
            checks.append(Check("endpoint", True, "provider default"))
        else:
            safe = _is_https_or_loopback(config.endpoint)
            detail = f"overridden by {config.endpoint_variable}: {_redact_url(config.endpoint)}"
            if not safe:
                detail += " - not https, so the credential would cross the network unencrypted"
            checks.append(Check("endpoint", safe, detail))

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
            import botocore.session

            session = botocore.session.Session()
            imds = os.environ.get("AWS_EC2_METADATA_DISABLED", "").strip().lower() == "false"
            if not imds:
                session.get_component("credential_provider").remove("iam-role")
            creds = session.get_credentials()
            if creds is None:
                not_consulted = "" if imds else (
                    " (instance metadata was not consulted; on EC2 set AWS_EC2_METADATA_DISABLED=false "
                    "to allow it)"
                )
                checks.append(
                    Check(
                        "credentials resolvable",
                        False,
                        f"the AWS credential chain resolved nothing; run `aws configure`{not_consulted}",
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
        checks.append(Check("host configured", bool(config.host), _redact_url(config.host) if config.host else "(none)"))
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
    wall-clock budget around the one structured call (the call runs on its
    own thread and event loop, and is told to stop through Strands'
    cancel_signal), and bounds turns with limits={"turns": 1}.

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
