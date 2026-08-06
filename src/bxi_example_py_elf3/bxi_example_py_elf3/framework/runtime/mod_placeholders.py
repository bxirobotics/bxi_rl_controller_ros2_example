"""Safe, fixed-value placeholder expansion for Mod YAML string values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import platform
import re
import socket
from types import MappingProxyType

from .runtime_requirements import runtime_platform_tag, runtime_python_tag


PlaceholderProvider = Callable[[], str]


def _primary_ipv4_address() -> str:
    """Return the preferred local IPv4 address without invoking a command."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_probe:
            # UDP connect only asks the kernel to select a route and source
            # address.  It does not send a packet until send() is called.
            route_probe.connect(("192.0.2.1", 9))
            address = str(route_probe.getsockname()[0])
            if address and address != "0.0.0.0":
                return address
    except OSError:
        pass

    try:
        candidates = socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=socket.AF_INET,
            type=socket.SOCK_DGRAM,
        )
    except OSError:
        candidates = ()
    for candidate in candidates:
        address = str(candidate[4][0])
        if address and not address.startswith("127.") and address != "0.0.0.0":
            return address
    return "127.0.0.1"


# This is the single extension point for built-in Mod YAML placeholders.  A
# provider must only return already-computed process or platform metadata.  It
# must never invoke a shell, evaluate YAML text, or run a command supplied by a
# Mod.  Values are evaluated once for each loaded manifest.
PRESET_PLACEHOLDER_PROVIDERS: Mapping[str, PlaceholderProvider] = MappingProxyType(
    {
        "bxi.system.name": platform.system,
        "bxi.system.release": platform.release,
        "bxi.system.machine": platform.machine,
        "bxi.system.hostname": platform.node,
        "bxi.system.ip": _primary_ipv4_address,
        "bxi.python.version": platform.python_version,
        "bxi.runtime.platform": runtime_platform_tag,
        "bxi.runtime.python_tag": runtime_python_tag,
    }
)


_PLACEHOLDER_PATTERN = re.compile(
    r"\$\$\{(?P<escaped>bxi\.[^{}]*)\}"
    r"|\$\{(?P<name>bxi\.[^{}]*)\}"
)


def expand_mod_placeholders(value: object, *, context: str) -> object:
    """Recursively expand preset placeholders in Mod YAML string values.

    Mapping keys are deliberately not expanded because they define the Mod
    schema.  ``$${bxi.system.name}`` escapes one placeholder and produces the
    literal text ``${bxi.system.name}``.  Non-BXI forms such as ``${HOME}`` and
    ``$(uname)`` are left untouched.
    """

    resolved: dict[str, str] = {}
    return _expand_value(
        value,
        PRESET_PLACEHOLDER_PROVIDERS,
        resolved,
        context,
    )


def _expand_value(
    value: object,
    providers: Mapping[str, PlaceholderProvider],
    resolved: dict[str, str],
    context: str,
) -> object:
    if isinstance(value, str):
        return _expand_string(value, providers, resolved, context)
    if isinstance(value, list):
        return [
            _expand_value(item, providers, resolved, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        return {
            key: _expand_value(item, providers, resolved, f"{context}.{key}")
            for key, item in value.items()
        }
    return value


def _expand_string(
    value: str,
    providers: Mapping[str, PlaceholderProvider],
    resolved: dict[str, str],
    context: str,
) -> str:
    def replace(match: re.Match[str]) -> str:
        escaped = match.group("escaped")
        if escaped is not None:
            return f"${{{escaped}}}"
        name = match.group("name")
        assert name is not None
        provider = providers.get(name)
        if provider is None:
            choices = ", ".join(f"${{{item}}}" for item in providers)
            raise ValueError(
                f"{context} uses unknown Mod placeholder ${{{name}}}; "
                f"available placeholders: {choices}"
            )
        replacement = resolved.get(name)
        if replacement is None:
            try:
                replacement = provider()
            except Exception as exc:
                raise ValueError(
                    f"{context} could not resolve Mod placeholder "
                    f"${{{name}}}: {exc}"
                ) from exc
            if not isinstance(replacement, str):
                raise TypeError(
                    f"Mod placeholder provider {name!r} returned "
                    f"{type(replacement).__name__}, expected str"
                )
            resolved[name] = replacement
        return replacement

    return _PLACEHOLDER_PATTERN.sub(replace, value)


__all__ = ["PRESET_PLACEHOLDER_PROVIDERS", "expand_mod_placeholders"]
