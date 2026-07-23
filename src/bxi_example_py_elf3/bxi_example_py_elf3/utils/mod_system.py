from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
import hashlib
import importlib.util
from itertools import count
from pathlib import Path
import re
import sys
import warnings
from types import ModuleType, UnionType
from typing import Generic, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

from bxi_example_py_elf3.utils.robot_state_base import RobotControlState
from bxi_example_py_elf3.utils.transition_core import (
    TransitionPluginSnapshot,
    release_transition_plugins,
    restore_transition_plugins,
    snapshot_transition_plugins,
)


ResourceT = TypeVar("ResourceT")
ParamT = TypeVar("ParamT")
ParamsT = TypeVar("ParamsT")
ResourceFactory = Callable[["ResourceLoadContext"], ResourceT]
StateFactory = Callable[["StateBuildContext"], RobotControlState]
ConfigMap = dict[str, object]
_python_export_owners: dict[str, str] = {}
_python_export_tokens: dict[str, object] = {}
_python_path_refcounts: dict[str, int] = {}
_python_paths_added_by_mods: set[str] = set()
_module_generations = count()


@dataclass(frozen=True)
class ResourceKey(Generic[ResourceT]):
    """A statically typed, globally unique resource identity."""

    id: str

    def __post_init__(self) -> None:
        if not self.id or "/" not in self.id:
            raise ValueError(f"resource id must be namespaced: {self.id!r}")


@dataclass(frozen=True)
class ResourceLoadContext:
    mod_id: str
    mod_root: Path
    _record_path: Callable[[Path], None] = field(repr=False, compare=False)

    def asset(self, relative_path: str) -> Path:
        path = (self.mod_root / relative_path).resolve()
        assets_root = (self.mod_root / "assets").resolve()
        if assets_root not in path.parents:
            raise ValueError(
                f"resource in '{self.mod_id}' must come from its assets folder: "
                f"{relative_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"resource asset does not exist in '{self.mod_id}': {relative_path}"
            )
        self._record_path(path)
        return path


@dataclass
class _ResourceProvider(Generic[ResourceT]):
    key: ResourceKey[ResourceT]
    owner: str
    root: Path
    factory: ResourceFactory[ResourceT]
    instance: ResourceT | None = None


class ResourceHandle(Generic[ResourceT]):
    def __init__(self, manager: "ResourceManager", key: ResourceKey[ResourceT]):
        self._manager = manager
        self._key = key

    @property
    def key(self) -> ResourceKey[ResourceT]:
        return self._key

    def get(self) -> ResourceT:
        return self._manager.get(self._key)


class ResourceManager:
    """Lazily creates and caches resources without adding dynamic ctx attributes."""

    def __init__(self) -> None:
        self._providers: dict[str, _ResourceProvider[object]] = {}
        self._loaded_paths: set[Path] = set()

    def register(
        self,
        key: ResourceKey[ResourceT],
        *,
        owner: str,
        root: Path,
        factory: ResourceFactory[ResourceT],
    ) -> None:
        previous = self._providers.get(key.id)
        if previous is not None:
            raise ValueError(
                f"duplicate resource '{key.id}' from '{previous.owner}' and '{owner}'"
            )
        provider = _ResourceProvider(key, owner, root, factory)
        self._providers[key.id] = cast(_ResourceProvider[object], provider)

    def handle(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        if key.id not in self._providers:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        return ResourceHandle(self, key)

    def get(self, key: ResourceKey[ResourceT]) -> ResourceT:
        provider = self._providers.get(key.id)
        if provider is None:
            raise ValueError(f"no loaded Mod provides resource '{key.id}'")
        if provider.instance is None:
            context = ResourceLoadContext(
                mod_id=provider.owner,
                mod_root=provider.root,
                _record_path=self._loaded_paths.add,
            )
            provider.instance = provider.factory(context)
        return cast(ResourceT, provider.instance)

    @property
    def loaded_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._loaded_paths))

    def close(self) -> None:
        first_error: Exception | None = None
        for provider in reversed(tuple(self._providers.values())):
            instance = provider.instance
            if instance is None:
                continue
            close = getattr(instance, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    provider.instance = None
            else:
                provider.instance = None
        if first_error is not None:
            raise first_error


@dataclass(frozen=True)
class StateBuildContext:
    name: str
    state_id: int
    params: Mapping[str, object]
    _consumed: set[str] = field(default_factory=set, compare=False, repr=False)

    def int_param(self, name: str, default: int) -> int:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"state '{self.name}' param '{name}' must be an integer")
        return value

    def float_param(self, name: str, default: float) -> float:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"state '{self.name}' param '{name}' must be a number")
        return float(value)

    def string_param(self, name: str, default: str) -> str:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if not isinstance(value, str):
            raise ValueError(f"state '{self.name}' param '{name}' must be a string")
        return value

    def bool_param(self, name: str, default: bool) -> bool:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"state '{self.name}' param '{name}' must be a boolean")
        return value

    def param(self, name: str, expected: type[ParamT], default: ParamT) -> ParamT:
        self._consumed.add(name)
        value = self.params.get(name, default)
        if not isinstance(value, expected):
            raise ValueError(
                f"state '{self.name}' param '{name}' must be " f"{expected.__name__}"
            )
        return value

    def dataclass_params(self, params_type: type[ParamsT]) -> ParamsT:
        """Build a typed parameter object while retaining strict YAML validation."""
        if not isinstance(params_type, type) or not is_dataclass(params_type):
            raise TypeError("dataclass_params() expects a dataclass type")
        try:
            annotations = get_type_hints(params_type)
        except (NameError, TypeError) as exc:
            raise TypeError(
                f"state '{self.name}' cannot resolve annotations for "
                f"{params_type.__name__}: {exc}"
            ) from exc

        values: dict[str, object] = {}
        for parameter in fields(params_type):
            name = parameter.name
            self._consumed.add(name)
            if name not in self.params:
                if (
                    parameter.default is MISSING
                    and parameter.default_factory is MISSING
                ):
                    raise ValueError(
                        f"state '{self.name}' is missing required param '{name}'"
                    )
                continue
            values[name] = _typed_dataclass_value(
                self.name,
                name,
                self.params[name],
                annotations.get(name, parameter.type),
            )
        return params_type(**values)

    def finish(self) -> None:
        unknown = set(self.params) - self._consumed
        if unknown:
            raise ValueError(
                f"state '{self.name}' has unknown params: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ModDefinition:
    state_factories: Mapping[str, StateFactory] = field(default_factory=dict)


class ModLoadContext:
    def __init__(self, mod_id: str, mod_root: Path, resources: ResourceManager):
        self.mod_id = mod_id
        self.mod_root = mod_root
        self.resources = resources

    def register_resource(
        self,
        key: ResourceKey[ResourceT],
        factory: ResourceFactory[ResourceT],
    ) -> None:
        self.resources.register(
            key,
            owner=self.mod_id,
            root=self.mod_root,
            factory=factory,
        )

    def resource(self, key: ResourceKey[ResourceT]) -> ResourceHandle[ResourceT]:
        return self.resources.handle(key)


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
    root: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    requires: tuple[_Requirement, ...]


@dataclass
class _PythonExportSession:
    names: tuple[str, ...]
    roots: tuple[str, ...]
    token: object
    _previous_modules: dict[str, ModuleType]
    _previous_exports: dict[str, tuple[str, object] | None]
    _active: bool = True

    def commit(self) -> None:
        self._previous_modules.clear()
        self._previous_exports.clear()

    def rollback(self) -> None:
        if not self._active:
            return
        _remove_owned_python_exports(self.names, self.token)
        sys.modules.update(self._previous_modules)
        for name, previous in self._previous_exports.items():
            if previous is None:
                _python_export_owners.pop(name, None)
                _python_export_tokens.pop(name, None)
            else:
                owner, token = previous
                _python_export_owners[name] = owner
                _python_export_tokens[name] = token
        self._release_paths()
        self._active = False

    def close(self) -> None:
        if not self._active:
            return
        _remove_owned_python_exports(self.names, self.token)
        self._release_paths()
        self._active = False

    def _release_paths(self) -> None:
        for root in self.roots:
            remaining = _python_path_refcounts[root] - 1
            if remaining > 0:
                _python_path_refcounts[root] = remaining
                continue
            _python_path_refcounts.pop(root, None)
            if root in _python_paths_added_by_mods:
                _python_paths_added_by_mods.remove(root)
                try:
                    sys.path.remove(root)
                except ValueError:
                    pass


@dataclass
class ModRuntime:
    config: ConfigMap
    state_factories: dict[str, StateFactory]
    resources: ResourceManager
    mods: tuple[LoadedMod, ...]
    watched_paths: tuple[Path, ...]
    _modules: tuple[ModuleType, ...] = field(repr=False)
    _module_prefixes: tuple[str, ...] = field(repr=False)
    _python_exports: _PythonExportSession = field(repr=False)
    _transition_plugins: TransitionPluginSnapshot = field(repr=False)

    def close(self) -> None:
        try:
            self.resources.close()
        finally:
            _remove_module_prefixes(self._module_prefixes)
            release_transition_plugins(
                self._transition_plugins,
                self._module_prefixes,
            )
            self._python_exports.close()


def load_mod_runtime(
    base_config: Mapping[str, object],
    *,
    built_in_root: Path,
    extra_roots: Sequence[Path] = (),
) -> ModRuntime:
    discovered = _discover_mods((built_in_root, *extra_roots))
    ordered = _dependency_order(discovered)
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
        config, factories = _compose_config(base_config, ordered, definitions)
    except Exception:
        resources.close()
        _remove_module_prefixes(
            tuple(module.__name__.split(".", 1)[0] for module in loaded_modules)
        )
        restore_transition_plugins(transition_plugins)
        python_exports.rollback()
        raise

    watched: set[Path] = set()
    for mod in ordered:
        watched.update(path for path in mod.root.rglob("*") if path.is_file())
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
    python_exports.commit()
    return ModRuntime(
        config=config,
        state_factories=factories,
        resources=resources,
        mods=loaded,
        watched_paths=tuple(sorted(watched)),
        _modules=tuple(loaded_modules),
        _module_prefixes=tuple(
            module.__name__.split(".", 1)[0] for module in loaded_modules
        ),
        _python_exports=python_exports,
        _transition_plugins=transition_plugins,
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
            requires = _read_requirements(manifest.get("requires"), manifest_path)
            previous = result.get(mod_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate Mod '{mod_id}': {previous.root} and {manifest_path.parent}"
                )
            result[mod_id] = _DiscoveredMod(
                id=mod_id,
                version=version,
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
            if loaded_owner is not None and loaded_owner != mod.id:
                raise ValueError(
                    f"Python package '{item}' was already loaded from Mod "
                    f"'{loaded_owner}'"
                )
            if loaded_owner is None and item in sys.modules:
                raise ValueError(
                    f"Mod '{mod.id}' cannot replace already imported package '{item}'"
                )

    token = object()
    names = tuple(exports)
    roots = tuple(dict.fromkeys(root for _, root in exports.values()))
    previous_modules = {
        module_name: module
        for module_name, module in tuple(sys.modules.items())
        if any(
            module_name == name or module_name.startswith(f"{name}.") for name in names
        )
    }
    previous_exports = {
        name: (
            (_python_export_owners[name], _python_export_tokens[name])
            if name in _python_export_owners
            else None
        )
        for name in names
    }
    for root in roots:
        count = _python_path_refcounts.get(root, 0)
        if count == 0 and root not in sys.path:
            sys.path.insert(0, root)
            _python_paths_added_by_mods.add(root)
        _python_path_refcounts[root] = count + 1
    _remove_module_prefixes(names)
    for name, (owner, _) in exports.items():
        _python_export_owners[name] = owner
        _python_export_tokens[name] = token
    return _PythonExportSession(
        names,
        roots,
        token,
        previous_modules,
        previous_exports,
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
    generation = max(
        (path.stat().st_mtime_ns for path in mod.root.rglob("*") if path.is_file()),
        default=0,
    )
    nonce = next(_module_generations)
    digest = hashlib.sha256(
        f"{mod.root}:{generation}:{nonce}".encode("utf-8")
    ).hexdigest()[:12]
    package_name = f"_bxi_mod_{re.sub('[^a-zA-Z0-9_]', '_', mod.id)}_{digest}"
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
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
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


def _remove_owned_python_exports(names: Sequence[str], token: object) -> None:
    owned = [name for name in names if _python_export_tokens.get(name) is token]
    _remove_module_prefixes(owned)
    for name in owned:
        _python_export_tokens.pop(name, None)
        _python_export_owners.pop(name, None)


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
    declared_indexes: set[int] = set()
    indexed_states: list[tuple[str, dict[str, object], Mapping[str, object], int]] = []

    # Validate and reserve every explicitly declared index before assigning new
    # ones. This prevents an earlier conflict from taking an index that a later
    # state legitimately declares.
    for state_name, raw_state in states.items():
        state = _mapping(raw_state, f"states.{state_name}")
        manifest = _mapping(state.get("manifest"), f"states.{state_name}.manifest")
        index = manifest.get("index")
        if index is None:
            continue
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                f"state '{state_name}' manifest index must be a non-negative integer"
            )
        indexed_states.append(
            (state_name, cast(dict[str, object], state), manifest, index)
        )
        declared_indexes.add(index)

    owners: dict[int, str] = {}
    allocated_indexes = set(declared_indexes)
    for state_name, state, manifest, index in indexed_states:
        previous = owners.get(index)
        if previous is None:
            owners[index] = state_name
            continue

        new_index = index + 1
        while new_index in allocated_indexes:
            new_index += 1
        allocated_indexes.add(new_index)
        owners[new_index] = state_name
        updated_manifest = dict(manifest)
        updated_manifest["index"] = new_index
        state["manifest"] = updated_manifest
        warnings.warn(
            f"state manifest index conflict: '{previous}' keeps index {index}; "
            f"'{state_name}' was reassigned to index {new_index}",
            RuntimeWarning,
            stacklevel=2,
        )


_STATE_MANIFEST_FIELDS = (
    "label",
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


def _typed_dataclass_value(
    state_name: str,
    parameter_name: str,
    value: object,
    annotation: object,
) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        allows_none = type(None) in arguments
        candidates = tuple(item for item in arguments if item is not type(None))
        if value is None and allows_none:
            return None
        if len(candidates) == 1:
            return _typed_dataclass_value(
                state_name,
                parameter_name,
                value,
                candidates[0],
            )
        raise TypeError(
            f"state '{state_name}' param '{parameter_name}' uses unsupported "
            f"union annotation {annotation!r}"
        )

    expected = annotation
    valid = False
    converted = value
    expected_name = getattr(expected, "__name__", repr(expected))
    if expected is float:
        valid = not isinstance(value, bool) and isinstance(value, (int, float))
        if valid:
            converted = float(cast(float, value))
        expected_name = "a number"
    elif expected is int:
        valid = not isinstance(value, bool) and isinstance(value, int)
        expected_name = "an integer"
    elif expected is bool:
        valid = isinstance(value, bool)
        expected_name = "a boolean"
    elif expected is str:
        valid = isinstance(value, str)
        expected_name = "a string"
    elif isinstance(expected, type):
        valid = isinstance(value, expected)
    else:
        raise TypeError(
            f"state '{state_name}' param '{parameter_name}' uses unsupported "
            f"annotation {annotation!r}"
        )
    if not valid:
        raise ValueError(
            f"state '{state_name}' param '{parameter_name}' must be {expected_name}"
        )
    return converted


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
