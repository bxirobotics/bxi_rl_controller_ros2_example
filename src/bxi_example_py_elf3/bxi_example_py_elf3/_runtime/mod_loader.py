"""Internal Mod discovery, loading, configuration composition and resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import cast

import yaml

from bxi_example_py_elf3.mod_api.mod import (
    ModDefinition,
    ModLoadContext,
    StateBuildContext,
    StateFactory,
)
from bxi_example_py_elf3._runtime.resource_manager import ResourceManager
from bxi_example_py_elf3.mod_api.state import RobotControlState
from bxi_example_py_elf3._runtime.transition import (
    register_transition_plugin,
    release_transition_plugins,
    restore_transition_plugins,
    snapshot_transition_plugins,
)


ConfigMap = dict[str, object]
_python_export_owners: dict[str, str] = {}


@dataclass(frozen=True)
class LoadedMod:
    id: str
    version: str
    root: Path
    manifest_path: Path
    requires: tuple[str, ...]


@dataclass(frozen=True)
class _Requirement:
    id: str
    version: str | None = None


@dataclass(frozen=True)
class _DiscoveredMod:
    id: str
    version: str
    enabled: bool
    root: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    requires: tuple[_Requirement, ...]


@dataclass
class _PythonExportSession:
    names: tuple[str, ...]
    added_roots: tuple[str, ...]
    owners: Mapping[str, str]
    _active: bool = True

    def close(self) -> None:
        if not self._active:
            return
        _remove_module_prefixes(self.names)
        for name, owner in self.owners.items():
            if _python_export_owners.get(name) == owner:
                _python_export_owners.pop(name, None)
        for root in self.added_roots:
            try:
                sys.path.remove(root)
            except ValueError:
                pass
        self._active = False


@dataclass
class ModRuntime:
    config: ConfigMap
    state_factories: dict[str, StateFactory]
    resources: ResourceManager
    mods: tuple[LoadedMod, ...]
    disabled_mods: tuple[LoadedMod, ...]
    _module_prefixes: tuple[str, ...] = field(repr=False)
    _python_exports: _PythonExportSession = field(repr=False)

    def close(self) -> None:
        try:
            self.resources.close()
        finally:
            _remove_module_prefixes(self._module_prefixes)
            release_transition_plugins(self._module_prefixes)
            self._python_exports.close()


def load_mod_runtime(
    base_config: Mapping[str, object],
    *,
    built_in_root: Path,
    extra_roots: Sequence[Path] = (),
) -> ModRuntime:
    discovered = _discover_mods((built_in_root, *extra_roots))
    enabled = {mod_id: mod for mod_id, mod in discovered.items() if mod.enabled}
    disabled_ids = set(discovered) - set(enabled)
    if not enabled:
        raise ValueError("no enabled Mods found")
    for mod in enabled.values():
        for requirement in mod.requires:
            if requirement.id in disabled_ids:
                raise ValueError(
                    f"Mod '{mod.id}' requires disabled Mod '{requirement.id}'"
                )
    ordered = _dependency_order(enabled)
    transition_plugins = snapshot_transition_plugins()
    python_exports = _prepare_python_exports(ordered)
    resources = ResourceManager()
    definitions: dict[str, ModDefinition] = {}
    loaded_modules: list[ModuleType] = []

    try:
        for mod in ordered:
            definition, module = _load_definition(mod, resources)
            definitions[mod.id] = definition
            loaded_modules.append(module)
            for type_name, plugin in definition.transition_plugins.items():
                if type_name != plugin.type_name:
                    raise ValueError(
                        f"Mod '{mod.id}' transition key '{type_name}' does not "
                        f"match plugin type_name '{plugin.type_name}'"
                    )
                register_transition_plugin(plugin)
        config, factories = _compose_config(base_config, ordered, definitions)
    except Exception:
        resources.close()
        _remove_module_prefixes(
            tuple(module.__name__.split(".", 1)[0] for module in loaded_modules)
        )
        restore_transition_plugins(transition_plugins)
        python_exports.close()
        raise

    loaded = tuple(
        LoadedMod(
            mod.id,
            mod.version,
            mod.root,
            mod.manifest_path,
            tuple(requirement.id for requirement in mod.requires),
        )
        for mod in ordered
    )
    disabled = tuple(
        LoadedMod(
            mod.id,
            mod.version,
            mod.root,
            mod.manifest_path,
            tuple(requirement.id for requirement in mod.requires),
        )
        for mod in discovered.values()
        if not mod.enabled
    )
    return ModRuntime(
        config=config,
        state_factories=factories,
        resources=resources,
        mods=loaded,
        disabled_mods=disabled,
        _module_prefixes=tuple(
            module.__name__.split(".", 1)[0] for module in loaded_modules
        ),
        _python_exports=python_exports,
    )


def _discover_mods(roots: Sequence[Path]) -> dict[str, _DiscoveredMod]:
    result: dict[str, _DiscoveredMod] = {}
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.exists():
            continue
        manifests = [root / "mod.yaml"] if (root / "mod.yaml").is_file() else []
        manifests.extend(root.rglob("mod.yaml"))
        for manifest_path in sorted(set(manifests)):
            manifest = _yaml_mapping(manifest_path)
            schema = manifest.get("schema", 1)
            if schema != 1:
                raise ValueError(f"{manifest_path}: unsupported Mod schema {schema!r}")
            mod_id = _required_string(manifest, "id", manifest_path)
            if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)+", mod_id):
                raise ValueError(
                    f"{manifest_path}: invalid namespaced Mod id '{mod_id}'"
                )
            version = _required_string(manifest, "version", manifest_path)
            _version_tuple(version)
            api = manifest.get("api", 1)
            if api != 1:
                raise ValueError(f"{manifest_path}: unsupported Mod API {api!r}")
            enabled = manifest.get("enable", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"{manifest_path}: 'enable' must be a boolean")
            requires = _read_requirements(manifest.get("requires"), manifest_path)
            previous = result.get(mod_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate Mod '{mod_id}': {previous.root} and {manifest_path.parent}"
                )
            result[mod_id] = _DiscoveredMod(
                id=mod_id,
                version=version,
                enabled=enabled,
                root=manifest_path.parent,
                manifest_path=manifest_path,
                manifest=manifest,
                requires=requires,
            )
    if not result:
        raise ValueError(f"no Mods found in: {', '.join(map(str, roots))}")
    return result


def _dependency_order(mods: Mapping[str, _DiscoveredMod]) -> list[_DiscoveredMod]:
    ordered: list[_DiscoveredMod] = []
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(mod_id: str) -> None:
        if mod_id in visited:
            return
        if mod_id in visiting:
            cycle = " -> ".join((*visiting, mod_id))
            raise ValueError(f"Mod dependency cycle: {cycle}")
        mod = mods.get(mod_id)
        if mod is None:
            parent = visiting[-1] if visiting else "<root>"
            raise ValueError(f"Mod '{parent}' requires missing Mod '{mod_id}'")
        visiting.append(mod_id)
        for requirement in mod.requires:
            dependency = mods.get(requirement.id)
            if dependency is None:
                raise ValueError(
                    f"Mod '{mod.id}' requires missing Mod '{requirement.id}'"
                )
            if requirement.version is not None and not _version_matches(
                dependency.version, requirement.version
            ):
                raise ValueError(
                    f"Mod '{mod.id}' requires '{requirement.id}' "
                    f"version '{requirement.version}', found '{dependency.version}'"
                )
            visit(requirement.id)
        visiting.pop()
        visited.add(mod_id)
        ordered.append(mod)

    for mod_id in sorted(mods):
        visit(mod_id)
    return ordered


def _prepare_python_exports(
    mods: Sequence[_DiscoveredMod],
) -> _PythonExportSession:
    exports: dict[str, tuple[str, str]] = {}
    for mod in mods:
        raw_exports = mod.manifest.get("python_exports", ())
        if not isinstance(raw_exports, Sequence) or isinstance(
            raw_exports, (str, bytes)
        ):
            raise ValueError(f"{mod.manifest_path}: python_exports must be a list")
        for item in raw_exports:
            if not isinstance(item, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", item
            ):
                raise ValueError(
                    f"{mod.manifest_path}: invalid exported Python package {item!r}"
                )
            package_init = mod.root / item / "__init__.py"
            if not package_init.is_file():
                raise FileNotFoundError(
                    f"{mod.manifest_path}: exported package is missing: {item}"
                )
            previous = exports.get(item)
            if previous is not None:
                raise ValueError(
                    f"Python package '{item}' is exported by '{previous[0]}' "
                    f"and '{mod.id}'"
                )
            exports[item] = (mod.id, str(mod.root))
            loaded_owner = _python_export_owners.get(item)
            if loaded_owner is not None:
                raise ValueError(
                    f"Python package '{item}' was already loaded from Mod "
                    f"'{loaded_owner}'"
                )
            if loaded_owner is None and item in sys.modules:
                raise ValueError(
                    f"Mod '{mod.id}' cannot replace already imported package '{item}'"
                )

    names = tuple(exports)
    roots = tuple(dict.fromkeys(root for _, root in exports.values()))
    added_roots: list[str] = []
    for root in roots:
        if root not in sys.path:
            sys.path.insert(0, root)
            added_roots.append(root)
    for name, (owner, _) in exports.items():
        _python_export_owners[name] = owner
    return _PythonExportSession(
        names,
        tuple(added_roots),
        {name: owner for name, (owner, _) in exports.items()},
    )


def _load_definition(
    mod: _DiscoveredMod,
    resources: ResourceManager,
) -> tuple[ModDefinition, ModuleType]:
    entrypoint_value = mod.manifest.get("entrypoint")
    if entrypoint_value is None and (mod.root / "plugin.py").is_file():
        entrypoint_value = "plugin:create_mod"
    if entrypoint_value is None:
        package = _create_dynamic_package(mod)
        try:
            definition = _load_convention_definition(mod, package)
        except Exception:
            _remove_module_prefixes((package.__name__,))
            raise
        return definition, package
    if not isinstance(entrypoint_value, str) or not entrypoint_value:
        raise ValueError(f"{mod.manifest_path}: entrypoint must be a string")
    entrypoint = entrypoint_value
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            f"{mod.manifest_path}: entrypoint must look like 'plugin:create_mod'"
        )
    if (
        not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            module_name,
        )
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name)
    ):
        raise ValueError(f"{mod.manifest_path}: invalid entrypoint '{entrypoint}'")
    package = _create_dynamic_package(mod)
    try:
        module = _load_mod_module(mod, package, module_name)
        factory = getattr(module, function_name, None)
        if not callable(factory):
            raise TypeError(
                f"{mod.manifest_path}: entrypoint is not callable: {entrypoint}"
            )
        definition = factory(ModLoadContext(mod.id, mod.root, resources))
        if not isinstance(definition, ModDefinition):
            raise TypeError(
                f"{mod.manifest_path}: entrypoint must return ModDefinition"
            )
    except Exception:
        _remove_module_prefixes((package.__name__,))
        raise
    return definition, package


def _create_dynamic_package(mod: _DiscoveredMod) -> ModuleType:
    package_name = f"_bxi_mod_{mod.id.encode('utf-8').hex()}"
    if package_name in sys.modules:
        raise RuntimeError(
            f"Mod '{mod.id}' is already loaded in this process; "
            "restart the process to load it again"
        )
    package = ModuleType(package_name)
    package.__path__ = [str(mod.root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return package


def _load_mod_module(
    mod: _DiscoveredMod,
    package: ModuleType,
    module_name: str,
) -> ModuleType:
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
        module_name,
    ):
        raise ValueError(f"{mod.manifest_path}: invalid module name '{module_name}'")
    full_name = f"{package.__name__}.{module_name}"
    loaded = sys.modules.get(full_name)
    if loaded is not None:
        return loaded
    relative_module = Path(*module_name.split("."))
    module_file = mod.root / relative_module.with_suffix(".py")
    submodule_search_locations = None
    if not module_file.is_file():
        module_file = mod.root / relative_module / "__init__.py"
        if not module_file.is_file():
            raise FileNotFoundError(
                f"{mod.manifest_path}: module does not exist: {module_name}"
            )
        submodule_search_locations = [str(module_file.parent)]
    spec = importlib.util.spec_from_file_location(
        full_name,
        module_file,
        submodule_search_locations=submodule_search_locations,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Mod module: {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise
    return module


def _load_convention_definition(
    mod: _DiscoveredMod,
    package: ModuleType,
) -> ModDefinition:
    raw_states = _mapping(mod.manifest.get("states"), f"{mod.id}.states")
    factories: dict[str, StateFactory] = {}
    for local_name, raw_state in raw_states.items():
        state_config = _mapping(raw_state, f"{mod.id}.states.{local_name}")
        reference = state_config.get("factory")
        if not isinstance(reference, str):
            raise ValueError(
                f"{mod.manifest_path}: state '{local_name}' needs "
                "factory: module:Class when no entrypoint is used"
            )
        module_name, separator, class_name = reference.partition(":")
        if (
            not separator
            or not class_name
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", class_name)
        ):
            raise ValueError(
                f"{mod.manifest_path}: state '{local_name}' factory must look "
                "like 'state:WaveState'"
            )
        module = _load_mod_module(mod, package, module_name)
        state_type = getattr(module, class_name, None)
        if not isinstance(state_type, type) or not issubclass(
            state_type, RobotControlState
        ):
            raise TypeError(
                f"{mod.manifest_path}: factory '{reference}' must name a "
                "RobotControlState class"
            )
        factories[local_name] = _convention_state_factory(state_type)
    return ModDefinition(state_factories=factories)


def _convention_state_factory(
    state_type: type[RobotControlState],
) -> StateFactory:
    params_type = getattr(state_type, "Params", None)

    def build(context: StateBuildContext) -> RobotControlState:
        if params_type is None:
            return state_type(context.name, context.state_id)
        params = context.dataclass_params(params_type)
        return state_type(context.name, context.state_id, params)

    return build


def _remove_module_prefixes(prefixes: Sequence[str]) -> None:
    for module_name in tuple(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)


def _compose_config(
    base_config: Mapping[str, object],
    mods: Sequence[_DiscoveredMod],
    definitions: Mapping[str, ModDefinition],
) -> tuple[ConfigMap, dict[str, StateFactory]]:
    config: ConfigMap = {
        key: value for key, value in base_config.items() if key != "mod_paths"
    }
    states: dict[str, object] = {}
    remote_events: dict[str, object] = {}
    speed_profiles: dict[str, object] = {}
    transition_profiles = dict(
        _mapping(config.get("transition_profiles"), "transition_profiles")
    )
    factories: dict[str, StateFactory] = {}
    routes: list[tuple[_DiscoveredMod, Mapping[str, object]]] = []

    for mod in mods:
        definition = definitions[mod.id]
        raw_states = _mapping(mod.manifest.get("states"), f"{mod.id}.states")
        declared = set(raw_states)
        provided = set(definition.state_factories)
        if declared != provided:
            raise ValueError(
                f"Mod '{mod.id}' state manifest/factory mismatch: "
                f"missing_factories={sorted(declared - provided)}, "
                f"undeclared_factories={sorted(provided - declared)}"
            )
        mod_speed_profiles = _mapping(
            mod.manifest.get("speed_profiles"), f"{mod.id}.speed_profiles"
        )
        for local_name, raw_profile in mod_speed_profiles.items():
            _validate_local_name(mod.id, local_name, "speed profile")
            canonical = _qualify(mod.id, local_name)
            _insert_unique(speed_profiles, canonical, raw_profile, "speed profile")
        mod_transition_profiles = _mapping(
            mod.manifest.get("transition_profiles"),
            f"{mod.id}.transition_profiles",
        )
        for local_name, raw_profile in mod_transition_profiles.items():
            _validate_local_name(mod.id, local_name, "transition profile")
            canonical = _qualify(mod.id, local_name)
            _insert_unique(
                transition_profiles, canonical, raw_profile, "transition profile"
            )
        for local_name, raw_state in raw_states.items():
            _validate_local_name(mod.id, local_name, "state")
            canonical = _qualify(mod.id, local_name)
            state_config = dict(_mapping(raw_state, f"{mod.id}.states.{local_name}"))
            state_config.pop("factory", None)
            _normalize_state_manifest_shorthand(
                state_config,
                f"{mod.id}.states.{local_name}",
            )
            speed_profile = state_config.get("speed_profile")
            if isinstance(speed_profile, str) and "/" not in speed_profile:
                state_config["speed_profile"] = _qualify(mod.id, speed_profile)
            configured_transitions = state_config.get("transitions")
            if configured_transitions not in (None, {}):
                raise ValueError(
                    f"Mod '{mod.id}' state '{local_name}' must declare edges in routes"
                )
            state_config.setdefault("transitions", {})
            _insert_unique(states, canonical, state_config, "state")
            factories[canonical] = definition.state_factories[local_name]
        raw_events = _mapping(mod.manifest.get("events"), f"{mod.id}.events")
        for local_name, raw_event in raw_events.items():
            _validate_local_name(mod.id, local_name, "event")
            canonical = _qualify(mod.id, local_name)
            _insert_unique(remote_events, canonical, raw_event, "event")
        raw_routes = mod.manifest.get("routes", ())
        if not isinstance(raw_routes, Sequence) or isinstance(raw_routes, (str, bytes)):
            raise ValueError(f"{mod.id}.routes must be a list")
        for index, raw_route in enumerate(raw_routes):
            routes.append((mod, _mapping(raw_route, f"{mod.id}.routes[{index}]")))

    for mod, route in routes:
        from_name = _qualify(mod.id, _required_string(route, "from", mod.manifest_path))
        source = states.get(from_name)
        if source is None:
            raise ValueError(
                f"Mod '{mod.id}' route references unknown source '{from_name}'"
            )
        source_map = cast(dict[str, object], source)
        transitions = source_map.setdefault("transitions", {})
        if not isinstance(transitions, dict):
            raise ValueError(f"state '{from_name}'.transitions must be a map")
        event_name = _qualify(
            mod.id, _required_string(route, "event", mod.manifest_path)
        )
        if event_name not in remote_events:
            raise ValueError(
                f"Mod '{mod.id}' route references unknown event '{event_name}'"
            )
        on_event = transitions.setdefault("on_event", {})
        if not isinstance(on_event, dict):
            raise ValueError(f"state '{from_name}'.transitions.on_event must be a map")
        if event_name in on_event:
            raise ValueError(
                f"duplicate route for state '{from_name}', event '{event_name}'"
            )
        target = route.get("to")
        action = route.get("action")
        if target is None and action is None:
            raise ValueError(
                f"route '{from_name}'/'{event_name}' needs 'to' or 'action'"
            )
        rule: dict[str, object] = {}
        if target is not None:
            if not isinstance(target, str):
                raise ValueError(
                    f"route '{from_name}'/'{event_name}'.to must be a string"
                )
            target_name = _qualify(mod.id, target)
            if target_name not in states:
                raise ValueError(
                    f"Mod '{mod.id}' route targets unknown state '{target_name}'"
                )
            rule["to"] = target_name
        if action is not None:
            if not isinstance(action, str):
                raise ValueError(
                    f"route '{from_name}'/'{event_name}'.action must be a string"
                )
            rule["action"] = action
        if "transition" in route:
            rule["transition"] = _normalize_transition_reference(
                mod.id,
                route["transition"],
                transition_profiles,
            )
        if "delay" in route:
            rule["delay"] = route["delay"]
        on_event[event_name] = rule

    config["states"] = states
    config["remote_events"] = remote_events
    config["speed_profiles"] = speed_profiles
    config["transition_profiles"] = transition_profiles
    _validate_remote_inputs(states, remote_events)
    _resolve_state_manifest_indexes(states)
    for state_name, raw_state in states.items():
        state = cast(Mapping[str, object], raw_state)
        speed_profile = state.get("speed_profile")
        if speed_profile is not None and speed_profile not in speed_profiles:
            raise ValueError(
                f"state '{state_name}' references unknown speed profile "
                f"'{speed_profile}'"
            )
    initial = config.get("initial_state")
    if not isinstance(initial, str) or initial not in states:
        raise ValueError(f"initial_state references unavailable Mod state: {initial!r}")
    return config, factories


def _validate_remote_inputs(
    states: Mapping[str, object],
    remote_events: Mapping[str, object],
) -> None:
    bindings: dict[str, tuple[str, int | None]] = {}
    for event_name, raw_event in remote_events.items():
        if isinstance(raw_event, str):
            slot = raw_event
            value = None
        elif isinstance(raw_event, Mapping):
            unknown = set(raw_event) - {"slot", "value"}
            if unknown:
                raise ValueError(
                    f"remote event '{event_name}' has unknown fields: "
                    f"{sorted(unknown)}"
                )
            slot = raw_event.get("slot")
            value = raw_event.get("value")
        else:
            raise ValueError(
                f"remote event '{event_name}' must be a slot string or map"
            )
        if not isinstance(slot, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", slot
        ):
            raise ValueError(
                f"remote event '{event_name}' must define a valid slot name"
            )
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"remote event '{event_name}'.value must be an integer")
        bindings[event_name] = (slot, value)

    used_events: set[str] = set()
    for state_name, raw_state in states.items():
        state = _mapping(raw_state, f"states.{state_name}")
        transitions = _mapping(
            state.get("transitions"), f"states.{state_name}.transitions"
        )
        on_event = _mapping(
            transitions.get("on_event"),
            f"states.{state_name}.transitions.on_event",
        )
        active_bindings: list[tuple[str, str, int | None]] = []
        for event_name in on_event:
            binding = bindings.get(event_name)
            if binding is None:
                raise ValueError(
                    f"state '{state_name}' references undeclared remote event "
                    f"'{event_name}'"
                )
            used_events.add(event_name)
            slot, value = binding
            for other_event, other_slot, other_value in active_bindings:
                if slot != other_slot:
                    continue
                if value is None or other_value is None or value == other_value:
                    first_binding = (
                        f"{other_slot}=any change"
                        if other_value is None
                        else f"{other_slot}={other_value}"
                    )
                    second_binding = (
                        f"{slot}=any change" if value is None else f"{slot}={value}"
                    )
                    raise ValueError(
                        f"remote input conflict in state '{state_name}': events "
                        f"'{other_event}' ({first_binding}) and '{event_name}' "
                        f"({second_binding}) can trigger together"
                    )
            active_bindings.append((event_name, slot, value))

    unused = sorted(set(bindings) - used_events)
    if unused:
        raise ValueError(f"remote events have no routes: {unused}")


def _resolve_state_manifest_indexes(states: Mapping[str, object]) -> None:
    explicit_owners: dict[int, str] = {}
    automatic_states: list[
        tuple[int, str, dict[str, object], Mapping[str, object]]
    ] = []

    for state_name, raw_state in states.items():
        state = _mapping(raw_state, f"states.{state_name}")
        manifest = _mapping(state.get("manifest"), f"states.{state_name}.manifest")
        priority = manifest.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(
                f"state '{state_name}' manifest priority must be an integer"
            )
        index = manifest.get("index")
        if index is None:
            automatic_states.append(
                (priority, state_name, cast(dict[str, object], state), manifest)
            )
            continue
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                f"state '{state_name}' manifest index must be a non-negative integer"
            )
        previous = explicit_owners.get(index)
        if previous is not None:
            raise ValueError(
                f"duplicate explicit state manifest index {index}: "
                f"'{previous}' and '{state_name}'"
            )
        explicit_owners[index] = state_name

    allocated_indexes = set(explicit_owners)
    next_index = 0
    for _, _, state, manifest in sorted(
        automatic_states,
        key=lambda item: (-item[0], item[1]),
    ):
        while next_index in allocated_indexes:
            next_index += 1
        allocated_indexes.add(next_index)
        updated_manifest = dict(manifest)
        updated_manifest["index"] = next_index
        state["manifest"] = updated_manifest
        next_index += 1


_STATE_MANIFEST_FIELDS = (
    "label",
    "priority",
    "index",
    "group",
    "icon",
    "confirm",
    "confirm_message",
)


def _normalize_state_manifest_shorthand(
    state: dict[str, object],
    context: str,
) -> None:
    manifest = dict(_mapping(state.get("manifest"), f"{context}.manifest"))
    for name in _STATE_MANIFEST_FIELDS:
        if name not in state:
            continue
        value = state.pop(name)
        if name in manifest and manifest[name] != value:
            raise ValueError(
                f"{context} declares conflicting '{name}' shorthand and manifest values"
            )
        manifest[name] = value
    if manifest or "manifest" in state:
        state["manifest"] = manifest


def _qualify(mod_id: str, local_or_canonical: str) -> str:
    return (
        local_or_canonical
        if "/" in local_or_canonical
        else f"{mod_id}/{local_or_canonical}"
    )


def _validate_local_name(mod_id: str, name: str, kind: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"Mod '{mod_id}' has invalid local {kind} name '{name}'")


def _normalize_transition_reference(
    mod_id: str,
    raw: object,
    profiles: Mapping[str, object],
) -> object:
    if isinstance(raw, str):
        local_name = _qualify(mod_id, raw)
        return local_name if local_name in profiles else raw
    if isinstance(raw, Mapping):
        normalized = dict(raw)
        profile = normalized.get("profile")
        if isinstance(profile, str):
            local_name = _qualify(mod_id, profile)
            if local_name in profiles:
                normalized["profile"] = local_name
        return normalized
    return raw


def _read_requirements(value: object, path: Path) -> tuple[_Requirement, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path}: requires must be a list")
    requirements: list[_Requirement] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            requirements.append(_Requirement(item))
        elif isinstance(item, Mapping):
            requirement_id = item.get("id")
            if not isinstance(requirement_id, str) or not requirement_id:
                raise ValueError(f"{path}: requires[{index}].id must be a string")
            version = item.get("version")
            if version is not None and not isinstance(version, str):
                raise ValueError(f"{path}: requires[{index}].version must be a string")
            requirements.append(_Requirement(requirement_id, version))
        else:
            raise ValueError(f"{path}: requires[{index}] must be a string or map")
    return tuple(requirements)


def _version_matches(version: str, specifier: str) -> bool:
    actual = _version_tuple(version)
    for raw_clause in specifier.split(","):
        clause = raw_clause.strip()
        match = re.fullmatch(r"(==|!=|>=|<=|>|<)\s*(\d+(?:\.\d+)*)", clause)
        if match is None:
            raise ValueError(f"unsupported Mod version constraint: {specifier!r}")
        expected = _version_tuple(match.group(2))
        width = max(len(actual), len(expected))
        left = actual + (0,) * (width - len(actual))
        right = expected + (0,) * (width - len(expected))
        operator = match.group(1)
        passed = {
            "==": left == right,
            "!=": left != right,
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
        }[operator]
        if not passed:
            return False
    return True


def _version_tuple(version: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)*", version):
        raise ValueError(f"Mod versions must be numeric dot versions: {version!r}")
    return tuple(int(part) for part in version.split("."))


def _yaml_mapping(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as input_file:
        value: object = yaml.safe_load(input_file) or {}
    return _mapping(value, str(path))


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed map")
    return cast(Mapping[str, object], value)


def _required_string(mapping: Mapping[str, object], name: str, path: Path) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: '{name}' must be a non-empty string")
    return value


def _insert_unique(
    target: dict[str, object], key: str, value: object, kind: str
) -> None:
    if key in target:
        raise ValueError(f"duplicate {kind} '{key}'")
    target[key] = value


__all__ = [
    "LoadedMod",
    "ModDefinition",
    "ModLoadContext",
    "ModRuntime",
    "ResourceHandle",
    "ResourceKey",
    "ResourceLoadContext",
    "ResourceManager",
    "StateBuildContext",
    "StateFactory",
    "load_mod_runtime",
]
