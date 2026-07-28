"""Internal Transition registry and configuration compiler."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from bxi_example_py_elf3.framework.mod_api.transition import TransitionPlan, TransitionPlugin


_plugins: dict[str, type[TransitionPlugin]] = {}
_builtins_registered = False
TransitionPluginSnapshot: TypeAlias = dict[str, type[TransitionPlugin]]


def register_transition_plugin(plugin: type[TransitionPlugin]) -> None:
    type_name = getattr(plugin, "type_name", "")
    if not type_name:
        raise TypeError(f"{plugin.__name__} must define type_name")
    existing = _plugins.get(type_name)
    if existing is not None and existing is not plugin:
        raise TypeError(
            f"duplicate transition type '{type_name}': "
            f"{existing.__name__} and {plugin.__name__}"
        )
    _plugins[type_name] = plugin


def _ensure_builtin_plugins() -> None:
    global _builtins_registered
    if _builtins_registered:
        return
    from bxi_example_py_elf3.framework.transitions import BUILTIN_TRANSITION_PLUGINS

    for plugin in BUILTIN_TRANSITION_PLUGINS.values():
        register_transition_plugin(plugin)
    _builtins_registered = True


def snapshot_transition_plugins() -> TransitionPluginSnapshot:
    return dict(_plugins)


def restore_transition_plugins(snapshot: TransitionPluginSnapshot) -> None:
    _plugins.clear()
    _plugins.update(snapshot)


def release_transition_plugins(module_prefixes: Sequence[str]) -> None:
    for type_name, plugin in tuple(_plugins.items()):
        if not any(
            plugin.__module__ == prefix or plugin.__module__.startswith(f"{prefix}.")
            for prefix in module_prefixes
        ):
            continue
        _plugins.pop(type_name, None)


def compile_transition(name: str, raw: Mapping[str, object]) -> TransitionPlan:
    _ensure_builtin_plugins()
    type_name = raw.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise ValueError(f"transition profile '{name}' must define string field 'type'")
    plugin = _plugins.get(type_name)
    if plugin is None:
        available = ", ".join(sorted(_plugins)) or "<none>"
        raise ValueError(
            f"unknown transition type '{type_name}' in profile '{name}'; "
            f"available: {available}"
        )
    return plugin.compile(name, raw)


__all__ = [
    "TransitionPluginSnapshot",
    "compile_transition",
    "register_transition_plugin",
    "release_transition_plugins",
    "restore_transition_plugins",
    "snapshot_transition_plugins",
]
