"""Side-effect-free inspection of an installed ELF3 state-machine package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from pathlib import Path
from typing import cast

from bxi_example_py_elf3.framework.mod_api import ModDefinition, RobotControlState
from bxi_example_py_elf3.framework.runtime.mod_loader import (
    _compose_config,
    _dependency_order,
    _discover_mods,
    _qualify,
    _validate_mod_conflicts,
)
from bxi_example_py_elf3.framework.runtime.state_builder import configured_states
from bxi_example_py_elf3.framework.runtime.state_machine import (
    RobotStateMachine,
    load_state_machine_config,
)


class _OfflineState(RobotControlState):
    """Metadata-only state used to reuse the runtime graph compiler."""

    def __init__(self, name: str, state_id: int) -> None:
        RobotControlState.__init__(self, name, state_id)

    def on_update(self, ctx: object, dt: float) -> None:
        del ctx, dt


class _NullLogger:
    def debug(self, message: str) -> None:
        del message

    def info(self, message: str) -> None:
        del message

    def warning(self, message: str) -> None:
        del message

    def error(self, message: str) -> None:
        del message


def _unavailable_factory(_state: object) -> RobotControlState:
    raise RuntimeError("offline state factories must never be called")


class StateMachineInspector:
    """Parse a state-machine graph without starting ROS or loading Mod code.

    The inspector reads the base YAML and each ``mod.yaml`` only. It does not
    import Mod entrypoints, check host runtime dependencies, create ROS nodes,
    start child processes, or load startup resources and inference models.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        config_path: str | Path,
        *,
        built_in_mod_root: str | Path,
        extra_mod_roots: Sequence[str | Path] | None = None,
        package_name: str | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.built_in_mod_root = Path(built_in_mod_root).expanduser().resolve()
        self.package_name = package_name

        base_config = load_state_machine_config(str(self.config_path))
        if extra_mod_roots is None:
            extra_mod_roots = self._configured_mod_roots(base_config)
        self.extra_mod_roots = tuple(
            self._resolve_extra_root(root) for root in extra_mod_roots
        )

        discovered = _discover_mods(
            (self.built_in_mod_root, *self.extra_mod_roots)
        )
        enabled = {key: mod for key, mod in discovered.items() if mod.enabled}
        disabled_ids = set(discovered) - set(enabled)
        if not enabled:
            raise ValueError("no enabled Mods found")
        for mod in enabled.values():
            for requirement in mod.requires:
                if requirement.id in disabled_ids:
                    raise ValueError(
                        f"Mod '{mod.id}' requires disabled Mod "
                        f"'{requirement.id}'"
                    )

        ordered = _dependency_order(enabled)
        _validate_mod_conflicts(enabled)
        definitions = {
            mod.id: ModDefinition(
                state_factories={
                    local_name: _unavailable_factory
                    for local_name in self._mapping(
                        mod.manifest.get("states"), f"{mod.id}.states"
                    )
                }
            )
            for mod in ordered
        }
        config, _factories = _compose_config(base_config, ordered, definitions)

        self._config = config
        self._ordered_mods = tuple(ordered)
        self._disabled_mods = tuple(
            mod for mod in discovered.values() if not mod.enabled
        )
        self._graph = self._compile_graph(config, ordered)
        self._nodes = self._node_snapshots(ordered, config)

    @classmethod
    def from_package(
        cls,
        package_name: str = "bxi_example_py_elf3",
        *,
        config: str | Path = "config/elf3_state_machine.yaml",
        mods: str | Path = "mods",
        extra_mod_roots: Sequence[str | Path] | None = None,
    ) -> "StateMachineInspector":
        """Locate an installed package through the sourced ament index."""

        from ament_index_python.packages import get_package_share_path

        share = get_package_share_path(package_name)
        return cls(
            share / config,
            built_in_mod_root=share / mods,
            extra_mod_roots=extra_mod_roots,
            package_name=package_name,
        )

    @property
    def config(self) -> Mapping[str, object]:
        """Return the composed static configuration."""

        return copy.deepcopy(self._config)

    def snapshot(self, *, include_graph: bool = True) -> dict[str, object]:
        """Return a JSON-safe declaration snapshot similar to runtime status."""

        result: dict[str, object] = {
            "schema_version": self.SCHEMA_VERSION,
            "offline": True,
            "package": self.package_name,
            "initial_state": self._config.get("initial_state"),
            "mods": [
                self._mod_snapshot(mod, "enabled") for mod in self._ordered_mods
            ]
            + [
                self._mod_snapshot(mod, "disabled")
                for mod in self._disabled_mods
            ],
            "nodes": copy.deepcopy(self._nodes),
        }
        if include_graph:
            result["graph"] = copy.deepcopy(self._graph)
        return result

    def graph(self) -> dict[str, object]:
        """Return only the compiled static graph."""

        return copy.deepcopy(self._graph)

    def _configured_mod_roots(
        self, base_config: Mapping[str, object]
    ) -> tuple[str | Path, ...]:
        raw = base_config.get("mod_paths", ())
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("mod_paths must be a list of directory strings")
        return tuple(cast(list[str], raw))

    def _resolve_extra_root(self, root: str | Path) -> Path:
        path = Path(root).expanduser()
        if not path.is_absolute():
            path = self.config_path.parent / path
        return path.resolve()

    @classmethod
    def _compile_graph(
        cls,
        config: Mapping[str, object],
        mods: Sequence[object],
    ) -> dict[str, object]:
        states: dict[str, RobotControlState] = {}
        for state in configured_states(config):
            descriptor = _OfflineState(state.name, state.state_id)
            descriptor.speed_profile_name = state.speed_profile
            descriptor.inference_hz = state.inference_hz
            descriptor.manifest.update(state.manifest)
            states[state.name] = descriptor

        graph_config = copy.deepcopy(dict(config))
        graph_config["graph"] = {"validate": False, "export": {}}
        machine = RobotStateMachine(
            cast(object, None),
            graph_config,
            states,
            logger=_NullLogger(),
            enter_initial=False,
        )
        graph = cast(
            dict[str, object],
            machine.snapshot(include_graph=True)["graph"],
        )

        behavior_by_state: dict[str, str | None] = {}
        for mod in mods:
            manifest = getattr(mod, "manifest")
            mod_id = cast(str, getattr(mod, "id"))
            raw_states = cls._mapping(manifest.get("states"), f"{mod_id}.states")
            for local_name, raw_state in raw_states.items():
                state_config = cls._mapping(
                    raw_state, f"{mod_id}.states.{local_name}"
                )
                factory = state_config.get("factory")
                behavior_by_state[_qualify(mod_id, local_name)] = (
                    factory.rpartition(":")[2]
                    if isinstance(factory, str) and ":" in factory
                    else None
                )
        for state in cast(list[dict[str, object]], graph["states"]):
            state["behavior"] = behavior_by_state.get(cast(str, state["name"]))
        return graph

    @classmethod
    def _node_snapshots(
        cls,
        mods: Sequence[object],
        config: Mapping[str, object],
    ) -> list[dict[str, object]]:
        known_states = set(cls._mapping(config.get("states"), "states"))
        result: list[dict[str, object]] = []
        known_nodes: set[str] = set()
        pending_dependencies: list[tuple[str, tuple[str, ...]]] = []

        for mod in mods:
            manifest = getattr(mod, "manifest")
            mod_id = cast(str, getattr(mod, "id"))
            raw_nodes = cls._mapping(manifest.get("nodes"), f"{mod_id}.nodes")
            for local_name, raw_node in raw_nodes.items():
                node = cls._mapping(raw_node, f"{mod_id}.nodes.{local_name}")
                node_id = _qualify(mod_id, local_name)
                states = tuple(
                    _qualify(mod_id, item)
                    for item in cls._string_list(
                        node.get("states", ()), f"{node_id}.states"
                    )
                )
                unknown_states = sorted(set(states) - known_states)
                if unknown_states:
                    raise ValueError(
                        f"Mod node '{node_id}' references unknown states: "
                        f"{unknown_states}"
                    )
                dependencies = tuple(
                    _qualify(mod_id, item)
                    for item in cls._string_list(
                        node.get("depends_on", ()), f"{node_id}.depends_on"
                    )
                )
                node_manifest = dict(
                    cls._mapping(node.get("manifest"), f"{node_id}.manifest")
                )
                result.append(
                    {
                        "id": node_id,
                        "runtime": node.get("runtime", "python"),
                        "execution": node.get("execution", "in_process"),
                        "lifecycle": node.get("lifecycle", "startup"),
                        "states": list(states),
                        "depends_on": list(dependencies),
                        "status": "declared",
                        "error": None,
                        "warnings": [],
                        **node_manifest,
                    }
                )
                known_nodes.add(node_id)
                pending_dependencies.append((node_id, dependencies))

        for node_id, dependencies in pending_dependencies:
            unknown = sorted(set(dependencies) - known_nodes)
            if unknown:
                raise ValueError(
                    f"Mod node '{node_id}' depends on unknown nodes: {unknown}"
                )
        return result

    @staticmethod
    def _mod_snapshot(mod: object, status: str) -> dict[str, object]:
        return {
            "id": getattr(mod, "id"),
            "version": getattr(mod, "version"),
            "status": status,
            "error": None,
            "warnings": [],
        }

    @staticmethod
    def _mapping(value: object, context: str) -> Mapping[str, object]:
        if value is None:
            return {}
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            raise ValueError(f"{context} must be a string-keyed map")
        return cast(Mapping[str, object], value)

    @staticmethod
    def _string_list(value: object, context: str) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{context} must be a list")
        result: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"{context}[{index}] must be a string")
            result.append(item)
        return tuple(result)


__all__ = ["StateMachineInspector"]
