"""Internal Mod discovery, loading, configuration composition and resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import ctypes
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import cast

import yaml

from bxi_example_py_elf3.mod_api_version import (
    MOD_API_VERSION,
    parse_numeric_version,
    parse_version_constraint,
    version_matches,
)
from bxi_example_py_elf3.mod_api.mod import (
    ModDefinition,
    ModLoadContext,
    StateBuildContext,
    StateFactory,
)
from bxi_example_py_elf3.mod_api.node import NodeFactory
from bxi_example_py_elf3._runtime.mod_nodes import ModNodeSpec
from bxi_example_py_elf3._runtime.resource_manager import ResourceManager
from bxi_example_py_elf3._runtime.runtime_requirements import (
    RuntimeRequirementReport,
    RuntimeRequirements,
    check_runtime_requirements,
    read_runtime_requirements,
)
from bxi_example_py_elf3.mod_api.state import RobotControlState
from bxi_example_py_elf3._runtime.transition import (
    register_transition_plugin,
    release_transition_plugins,
    restore_transition_plugins,
    snapshot_transition_plugins,
)


ConfigMap = dict[str, object]
_python_export_owners: dict[str, str] = {}
_process_vendor_library_handles: list[ctypes.CDLL] = []


@dataclass(frozen=True)
class LoadedMod:
    id: str
    version: str
    root: Path
    manifest_path: Path
    requires: tuple[str, ...]
    status: str
    error: str | None = None
    warnings: tuple[str, ...] = ()


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
    conflicts: tuple[str, ...]
    runtime_requirements: RuntimeRequirements


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
class _VendorSession:
    added_python_paths: list[str] = field(default_factory=list)
    loaded_libraries: list[ctypes.CDLL] = field(default_factory=list)
    _active: bool = True

    def activate(
        self,
        mod: _DiscoveredMod,
        report: RuntimeRequirementReport,
        owner: str,
    ) -> tuple[str | None, tuple[str, ...]]:
        if not report.uses_vendor:
            return None, ()

        warning = (
            f"{owner} uses bundled dependencies in-process; Python modules and "
            "native symbols are process-global, cannot be unloaded, and may "
            "conflict with other Mods"
        )
        for library in report.vendor_libraries:
            try:
                handle = ctypes.CDLL(
                    str(library),
                    mode=getattr(ctypes, "RTLD_GLOBAL", 0),
                )
            except OSError as exc:
                return (
                    f"cannot load bundled system library '{library.name}': {exc}",
                    (warning,),
                )
            self.loaded_libraries.append(handle)
            _process_vendor_library_handles.append(handle)

        for root in reversed(report.vendor_python_paths):
            path = str(root)
            if path not in sys.path:
                sys.path.insert(0, path)
                self.added_python_paths.append(path)
        return None, (warning,)

    def close(self) -> None:
        if not self._active:
            return
        for path in reversed(self.added_python_paths):
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        self.added_python_paths.clear()
        # The process-global handle list deliberately keeps native libraries
        # loaded. Removing import paths does not make native symbols isolatable.
        self.loaded_libraries.clear()
        self._active = False


@dataclass
class ModRuntime:
    config: ConfigMap
    state_factories: dict[str, StateFactory]
    resources: ResourceManager
    mods: tuple[LoadedMod, ...]
    disabled_mods: tuple[LoadedMod, ...]
    unavailable_mods: tuple[LoadedMod, ...]
    node_specs: tuple[ModNodeSpec, ...]
    _module_prefixes: tuple[str, ...] = field(repr=False)
    _python_exports: _PythonExportSession = field(repr=False)
    _vendor_session: _VendorSession = field(repr=False)

    def close(self) -> None:
        try:
            self.resources.close()
        finally:
            try:
                _remove_module_prefixes(self._module_prefixes)
                release_transition_plugins(self._module_prefixes)
            finally:
                try:
                    self._python_exports.close()
                finally:
                    self._vendor_session.close()


def load_mod_runtime(
    base_config: Mapping[str, object],
    *,
    built_in_root: Path,
    extra_roots: Sequence[Path] = (),
) -> ModRuntime:
    # Mods can live in a deployment directory owned by a different user than
    # the runtime process. Never create __pycache__ entries beside Mod code:
    # root-owned bytecode would make a later unprivileged update impossible.
    sys.dont_write_bytecode = True
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

    runtime_reports = {
        mod.id: check_runtime_requirements(mod.runtime_requirements, mod.root)
        for mod in enabled.values()
    }
    unavailable_reasons = {
        mod_id: "; ".join(report.errors)
        for mod_id, report in runtime_reports.items()
        if not report.available
    }
    for mod in enabled.values():
        if mod.id in unavailable_reasons:
            continue
        for requirement in mod.requires:
            reason = unavailable_reasons.get(requirement.id)
            if reason is not None:
                raise ValueError(
                    f"Mod '{mod.id}' requires unavailable Mod "
                    f"'{requirement.id}': {reason}"
                )

    candidates = {
        mod_id: mod
        for mod_id, mod in enabled.items()
        if mod_id not in unavailable_reasons
    }
    ordered_candidates = _dependency_order(candidates)
    vendor_session = _VendorSession()
    mod_warnings = {
        mod_id: report.warnings for mod_id, report in runtime_reports.items()
    }
    ordered: list[_DiscoveredMod] = []
    try:
        for mod in ordered_candidates:
            for requirement in mod.requires:
                reason = unavailable_reasons.get(requirement.id)
                if reason is not None:
                    raise ValueError(
                        f"Mod '{mod.id}' requires unavailable Mod "
                        f"'{requirement.id}': {reason}"
                    )
            activation_error, activation_warnings = vendor_session.activate(
                mod,
                runtime_reports[mod.id],
                f"Mod '{mod.id}'",
            )
            mod_warnings[mod.id] = (
                *runtime_reports[mod.id].warnings,
                *activation_warnings,
            )
            if activation_error is not None:
                unavailable_reasons[mod.id] = activation_error
                continue
            ordered.append(mod)
        _validate_mod_conflicts({mod.id: mod for mod in ordered})
        python_exports = _prepare_python_exports(ordered)
    except Exception:
        vendor_session.close()
        raise

    transition_plugins = snapshot_transition_plugins()
    resources = ResourceManager()
    definitions: dict[str, ModDefinition] = {}
    loaded_modules: list[ModuleType] = []
    node_specs: list[ModNodeSpec] = []

    try:
        for mod in ordered:
            definition, module = _load_definition(mod, resources)
            definitions[mod.id] = definition
            loaded_modules.append(module)
            package = sys.modules[module.__name__.split(".", 1)[0]]
            node_specs.extend(
                _load_mod_node_specs(
                    mod,
                    package,
                    vendor_session=vendor_session,
                )
            )
            for type_name, plugin in definition.transition_plugins.items():
                if type_name != plugin.type_name:
                    raise ValueError(
                        f"Mod '{mod.id}' transition key '{type_name}' does not "
                        f"match plugin type_name '{plugin.type_name}'"
                    )
                register_transition_plugin(plugin)
        resources.preload_eager()
        config, factories = _compose_config(base_config, ordered, definitions)
        _validate_mod_node_states(node_specs, config)
    except Exception:
        resources.close()
        _remove_module_prefixes(
            tuple(module.__name__.split(".", 1)[0] for module in loaded_modules)
        )
        restore_transition_plugins(transition_plugins)
        python_exports.close()
        vendor_session.close()
        raise

    loaded = tuple(
        LoadedMod(
            mod.id,
            mod.version,
            mod.root,
            mod.manifest_path,
            tuple(requirement.id for requirement in mod.requires),
            "loaded",
            warnings=mod_warnings.get(mod.id, ()),
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
            "disabled",
        )
        for mod in discovered.values()
        if not mod.enabled
    )
    unavailable = tuple(
        LoadedMod(
            mod.id,
            mod.version,
            mod.root,
            mod.manifest_path,
            tuple(requirement.id for requirement in mod.requires),
            "unavailable",
            unavailable_reasons[mod.id],
            mod_warnings.get(mod.id, ()),
        )
        for mod in discovered.values()
        if mod.id in unavailable_reasons
    )
    return ModRuntime(
        config=config,
        state_factories=factories,
        resources=resources,
        mods=loaded,
        disabled_mods=disabled,
        unavailable_mods=unavailable,
        node_specs=tuple(node_specs),
        _module_prefixes=tuple(
            module.__name__.split(".", 1)[0] for module in loaded_modules
        ),
        _python_exports=python_exports,
        _vendor_session=vendor_session,
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
            required_header = {
                "schema",
                "id",
                "name",
                "version",
                "api",
                "enable",
                "entrypoint",
                "visibility",
                "requires",
                "conflicts",
                "python_exports",
                "runtime_requirements",
            }
            missing_header = required_header - set(manifest)
            if missing_header:
                raise ValueError(
                    f"{manifest_path}: missing explicit Mod fields: "
                    f"{sorted(missing_header)}"
                )
            allowed_fields = required_header | {
                "events",
                "speed_profiles",
                "transition_profiles",
                "states",
                "routes",
                "actions",
                "nodes",
            }
            unknown_fields = set(manifest) - allowed_fields
            if unknown_fields:
                raise ValueError(
                    f"{manifest_path}: unknown Mod fields: {sorted(unknown_fields)}"
                )

            schema = manifest["schema"]
            if schema != 1:
                raise ValueError(f"{manifest_path}: unsupported Mod schema {schema!r}")
            mod_id = _required_string(manifest, "id", manifest_path)
            if not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)+", mod_id):
                raise ValueError(
                    f"{manifest_path}: invalid namespaced Mod id '{mod_id}'"
                )
            name = _required_string(manifest, "name", manifest_path)
            if not name.strip():
                raise ValueError(f"{manifest_path}: 'name' must not be blank")
            version = _required_string(manifest, "version", manifest_path)
            parse_numeric_version(version)
            api = manifest["api"]
            if not isinstance(api, str) or not api:
                raise ValueError(
                    f"{manifest_path}: 'api' must be a non-empty version constraint"
                )
            try:
                api_compatible = version_matches(MOD_API_VERSION, api)
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path}: invalid Mod API constraint {api!r}: {exc}"
                ) from exc
            if not api_compatible:
                raise ValueError(
                    f"{manifest_path}: requires Mod API {api!r}, "
                    f"framework provides {MOD_API_VERSION!r}"
                )
            enabled = manifest["enable"]
            if not isinstance(enabled, bool):
                raise ValueError(f"{manifest_path}: 'enable' must be a boolean")
            entrypoint = manifest["entrypoint"]
            if entrypoint is not None and (
                not isinstance(entrypoint, str) or not entrypoint
            ):
                raise ValueError(
                    f"{manifest_path}: 'entrypoint' must be null or a non-empty string"
                )
            visibility = manifest["visibility"]
            if visibility not in ("public", "protected"):
                raise ValueError(
                    f"{manifest_path}: 'visibility' must be 'public' or 'protected'"
                )
            requires = _read_requirements(manifest["requires"], manifest_path)
            runtime_requirements = read_runtime_requirements(
                manifest["runtime_requirements"],
                f"{manifest_path}: runtime_requirements",
            )
            conflicts = _read_mod_id_list(
                manifest["conflicts"], "conflicts", manifest_path
            )
            if mod_id in conflicts:
                raise ValueError(f"{manifest_path}: Mod cannot conflict with itself")
            _read_mod_id_list(
                manifest["python_exports"],
                "python_exports",
                manifest_path,
                python_names=True,
            )
            previous = result.get(mod_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate Mod '{mod_id}': {previous.root} and {manifest_path.parent}"
                )
            result[mod_id] = _DiscoveredMod(
                id=mod_id,
                version=version,
                enabled=enabled,
                root=manifest_path.resolve().parent,
                manifest_path=manifest_path,
                manifest=manifest,
                requires=requires,
                conflicts=conflicts,
                runtime_requirements=runtime_requirements,
            )
    if not result:
        raise ValueError(f"no Mods found in: {', '.join(map(str, roots))}")
    return result


def _validate_mod_conflicts(mods: Mapping[str, _DiscoveredMod]) -> None:
    for mod in mods.values():
        for conflict_id in mod.conflicts:
            if conflict_id not in mods:
                continue
            raise ValueError(
                f"enabled Mods conflict: '{mod.id}' declares conflict with "
                f"'{conflict_id}'"
            )


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
            if requirement.version is not None and not version_matches(
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
        raw_exports = mod.manifest["python_exports"]
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
    entrypoint_value = mod.manifest["entrypoint"]
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


def _load_mod_node_specs(
    mod: _DiscoveredMod,
    package: ModuleType,
    *,
    vendor_session: _VendorSession | None = None,
    load_process_factories: bool = False,
) -> list[ModNodeSpec]:
    raw_nodes = _mapping(mod.manifest.get("nodes"), f"{mod.id}.nodes")
    specs: list[ModNodeSpec] = []
    for local_name, raw_node in raw_nodes.items():
        _validate_local_name(mod.id, local_name, "node")
        context = f"{mod.id}.nodes.{local_name}"
        node = _mapping(raw_node, context)
        allowed_fields = {
            "entrypoint",
            "execution",
            "lifecycle",
            "states",
            "params",
            "manifest",
            "restart",
            "runtime_requirements",
        }
        unknown_fields = set(node) - allowed_fields
        if unknown_fields:
            raise ValueError(f"{context} has unknown fields: {sorted(unknown_fields)}")

        entrypoint = node.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError(f"{context}.entrypoint must be a non-empty string")
        module_name, function_name = _parse_entrypoint(
            entrypoint,
            f"{context}.entrypoint",
        )
        _validate_mod_module_exists(mod, module_name)

        execution = node.get("execution", "in_process")
        if execution not in ("in_process", "process"):
            raise ValueError(f"{context}.execution must be 'in_process' or 'process'")
        lifecycle = node.get("lifecycle", "mod")
        if lifecycle not in ("mod", "state"):
            raise ValueError(f"{context}.lifecycle must be 'mod' or 'state'")

        raw_states = node.get("states", ())
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
            raise ValueError(f"{context}.states must be a list")
        states: list[str] = []
        for state in raw_states:
            if not isinstance(state, str) or not state:
                raise ValueError(f"{context}.states entries must be non-empty strings")
            states.append(_qualify(mod.id, state))
        if lifecycle == "state" and not states:
            raise ValueError(f"{context}.states is required for state lifecycle")
        if lifecycle == "mod" and states:
            raise ValueError(f"{context}.states is only valid for state lifecycle")

        runtime_requirements = (
            read_runtime_requirements(
                node["runtime_requirements"],
                f"{context}.runtime_requirements",
            )
            if "runtime_requirements" in node
            else RuntimeRequirements((), (), ())
        )
        requirement_report = check_runtime_requirements(
            runtime_requirements,
            mod.root,
        )
        unavailable_error = (
            "; ".join(requirement_report.errors)
            if not requirement_report.available
            else None
        )
        runtime_warnings = requirement_report.warnings
        factory: NodeFactory | None = None
        should_load_factory = unavailable_error is None and (
            execution == "in_process" or load_process_factories
        )
        if should_load_factory:
            if execution == "in_process" and requirement_report.uses_vendor:
                if vendor_session is None:
                    raise RuntimeError(
                        f"{context} needs a vendor session for in-process loading"
                    )
                activation_error, activation_warnings = vendor_session.activate(
                    mod,
                    requirement_report,
                    f"Mod node '{_qualify(mod.id, local_name)}'",
                )
                runtime_warnings = (*runtime_warnings, *activation_warnings)
                if activation_error is not None:
                    unavailable_error = activation_error
            if unavailable_error is None:
                module = _load_mod_module(mod, package, module_name)
                raw_factory = getattr(module, function_name, None)
                if not callable(raw_factory):
                    raise TypeError(
                        f"{context}.entrypoint is not callable: {entrypoint}"
                    )
                factory = cast(NodeFactory, raw_factory)

        params = dict(_mapping(node.get("params"), f"{context}.params"))
        manifest = dict(_mapping(node.get("manifest"), f"{context}.manifest"))
        label = manifest.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{context}.manifest.label must be a non-empty string")
        reserved_manifest_fields = {
            "id",
            "entrypoint",
            "execution",
            "lifecycle",
            "states",
            "status",
            "restart_attempts",
            "error",
            "warnings",
        }
        conflicts = set(manifest) & reserved_manifest_fields
        if conflicts:
            raise ValueError(
                f"{context}.manifest uses reserved fields: {sorted(conflicts)}"
            )

        restart = _mapping(node.get("restart"), f"{context}.restart")
        if execution != "process" and restart:
            raise ValueError(f"{context}.restart is only valid for process execution")
        restart_unknown = set(restart) - {"max_attempts", "delay"}
        if restart_unknown:
            raise ValueError(
                f"{context}.restart has unknown fields: {sorted(restart_unknown)}"
            )
        max_attempts = restart.get("max_attempts", 3)
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 0
        ):
            raise ValueError(
                f"{context}.restart.max_attempts must be a non-negative integer"
            )
        restart_delay = restart.get("delay", 1.0)
        if (
            isinstance(restart_delay, bool)
            or not isinstance(restart_delay, (int, float))
            or restart_delay < 0.0
        ):
            raise ValueError(f"{context}.restart.delay must be a non-negative number")

        node_id = _qualify(mod.id, local_name)
        node_name = re.sub(r"[^A-Za-z0-9_]", "_", node_id)
        if not re.match(r"[A-Za-z_]", node_name):
            node_name = f"node_{node_name}"
        specs.append(
            ModNodeSpec(
                id=node_id,
                mod_id=mod.id,
                local_name=local_name,
                node_name=node_name,
                mod_root=mod.root,
                manifest_path=mod.manifest_path,
                entrypoint=entrypoint,
                execution=execution,
                lifecycle=lifecycle,
                states=tuple(dict.fromkeys(states)),
                params=params,
                manifest=manifest,
                restart_max_attempts=max_attempts,
                restart_delay=float(restart_delay),
                factory=factory,
                unavailable_error=unavailable_error,
                warnings=runtime_warnings,
            )
        )
    return specs


def _validate_mod_module_exists(mod: _DiscoveredMod, module_name: str) -> None:
    relative_module = Path(*module_name.split("."))
    if (mod.root / relative_module.with_suffix(".py")).is_file():
        return
    if (mod.root / relative_module / "__init__.py").is_file():
        return
    raise FileNotFoundError(
        f"{mod.manifest_path}: module does not exist: {module_name}"
    )


def _parse_entrypoint(reference: str, context: str) -> tuple[str, str]:
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"{context} must look like 'module:function'")
    if (
        not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            module_name,
        )
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name)
    ):
        raise ValueError(f"{context} is invalid: {reference!r}")
    return module_name, function_name


def _validate_mod_node_states(
    specs: Sequence[ModNodeSpec],
    config: Mapping[str, object],
) -> None:
    states = _mapping(config.get("states"), "states")
    for spec in specs:
        unknown = sorted(set(spec.states) - set(states))
        if unknown:
            raise ValueError(
                f"Mod node '{spec.id}' references unknown states: {unknown}"
            )


def load_process_node_spec(
    manifest_path: Path,
    local_name: str,
) -> tuple[ModNodeSpec, str]:
    """Load one node spec inside the dedicated child-process runner."""

    sys.dont_write_bytecode = True
    manifest_path = manifest_path.resolve()
    discovered = _discover_mods((manifest_path.parent,))
    mod = next(
        (
            item
            for item in discovered.values()
            if item.manifest_path.resolve() == manifest_path
        ),
        None,
    )
    if mod is None:
        raise ValueError(f"Mod manifest was not discovered: {manifest_path}")
    package = _create_dynamic_package(mod)
    try:
        specs = _load_mod_node_specs(
            mod,
            package,
            load_process_factories=True,
        )
        spec = next((item for item in specs if item.local_name == local_name), None)
        if spec is None:
            raise ValueError(f"Mod '{mod.id}' has no node '{local_name}'")
        if spec.unavailable_error is not None:
            raise RuntimeError(
                f"Mod node '{spec.id}' is unavailable: {spec.unavailable_error}"
            )
        if spec.factory is None:
            raise RuntimeError(f"Mod node '{spec.id}' has no process factory")
    except Exception:
        _remove_module_prefixes((package.__name__,))
        raise
    return spec, package.__name__


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
    event_rules: list[tuple[_DiscoveredMod, str, Mapping[str, object]]] = []
    actions: list[dict[str, object]] = []

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
            event_rules.append(
                (
                    mod,
                    "route",
                    _mapping(raw_route, f"{mod.id}.routes[{index}]"),
                )
            )
        raw_actions = mod.manifest.get("actions", ())
        if not isinstance(raw_actions, Sequence) or isinstance(
            raw_actions, (str, bytes)
        ):
            raise ValueError(f"{mod.id}.actions must be a list")
        for index, raw_action in enumerate(raw_actions):
            event_rules.append(
                (
                    mod,
                    "action",
                    _mapping(raw_action, f"{mod.id}.actions[{index}]"),
                )
            )

    for mod, rule_kind, event_rule in event_rules:
        allowed_fields = (
            {"from", "event", "to", "transition", "delay"}
            if rule_kind == "route"
            else {"from", "event", "action", "manifest"}
        )
        unknown_fields = set(event_rule) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"Mod '{mod.id}' {rule_kind} has unknown fields: "
                f"{sorted(unknown_fields)}"
            )
        from_name = _qualify(
            mod.id,
            _required_string(event_rule, "from", mod.manifest_path),
        )
        source = states.get(from_name)
        if source is None:
            raise ValueError(
                f"Mod '{mod.id}' {rule_kind} references unknown source '{from_name}'"
            )
        source_map = cast(dict[str, object], source)
        transitions = source_map.setdefault("transitions", {})
        if not isinstance(transitions, dict):
            raise ValueError(f"state '{from_name}'.transitions must be a map")
        event_name = _qualify(
            mod.id,
            _required_string(event_rule, "event", mod.manifest_path),
        )
        if event_name not in remote_events:
            raise ValueError(
                f"Mod '{mod.id}' {rule_kind} references unknown event '{event_name}'"
            )
        on_event = transitions.setdefault("on_event", {})
        if not isinstance(on_event, dict):
            raise ValueError(f"state '{from_name}'.transitions.on_event must be a map")
        if event_name in on_event:
            raise ValueError(
                f"duplicate route/action for state '{from_name}', event '{event_name}'"
            )
        rule: dict[str, object] = {}
        if rule_kind == "route":
            target = _required_string(event_rule, "to", mod.manifest_path)
            target_name = _qualify(mod.id, target)
            if target_name not in states:
                raise ValueError(
                    f"Mod '{mod.id}' route targets unknown state '{target_name}'"
                )
            rule["to"] = target_name
        else:
            action = _required_string(event_rule, "action", mod.manifest_path)
            rule["action"] = action
            manifest = dict(
                _mapping(
                    event_rule.get("manifest"),
                    f"{mod.id}.actions manifest",
                )
            )
            label = manifest.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"Mod '{mod.id}' action '{from_name}'/'{event_name}' "
                    "manifest.label must be a non-empty string"
                )
            reserved_manifest_fields = {"from", "event", "action", "manifest"}
            conflicting_fields = set(manifest) & reserved_manifest_fields
            if conflicting_fields:
                raise ValueError(
                    f"Mod '{mod.id}' action manifest uses reserved fields: "
                    f"{sorted(conflicting_fields)}"
                )
            actions.append(
                {
                    "from": from_name,
                    "event": event_name,
                    "action": action,
                    "manifest": manifest,
                }
            )
        if "transition" in event_rule:
            rule["transition"] = _normalize_transition_reference(
                mod.id,
                event_rule["transition"],
                transition_profiles,
            )
        if "delay" in event_rule:
            rule["delay"] = event_rule["delay"]
        on_event[event_name] = rule

    config["states"] = states
    config["remote_events"] = remote_events
    config["speed_profiles"] = speed_profiles
    config["transition_profiles"] = transition_profiles
    config["actions"] = actions
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
        raise ValueError(f"remote events have no routes or actions: {unused}")


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
            if version is not None:
                try:
                    parse_version_constraint(version)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: requires[{index}].version is invalid: {exc}"
                    ) from exc
            requirements.append(_Requirement(requirement_id, version))
        else:
            raise ValueError(f"{path}: requires[{index}] must be a string or map")
    return tuple(requirements)


def _read_mod_id_list(
    value: object,
    field: str,
    path: Path,
    *,
    python_names: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path}: {field} must be a list")
    pattern = (
        r"[A-Za-z_][A-Za-z0-9_]*" if python_names else r"[a-z0-9]+(?:[._-][a-z0-9]+)+"
    )
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not re.fullmatch(pattern, item):
            kind = "Python package name" if python_names else "namespaced Mod id"
            raise ValueError(f"{path}: {field}[{index}] must be a valid {kind}")
        if item in result:
            raise ValueError(f"{path}: duplicate {field} entry '{item}'")
        result.append(item)
    return tuple(result)


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
