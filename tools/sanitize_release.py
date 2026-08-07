#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
from typing import cast

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "bxi_example_py_elf3"
sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

from bxi_example_py_elf3.framework.mod_api_version import (  # noqa: E402
    MOD_API_VERSION,
    parse_numeric_version,
    parse_version_constraint,
    version_matches,
)
from bxi_example_py_elf3.framework.platform.cpu_affinity import (  # noqa: E402
    CpuAffinityRole,
    read_cpu_affinity,
)


DEFAULT_MOD_ROOT = Path("src/bxi_example_py_elf3/mods")
PUBLIC_DEV_ONLY_PATHS = {
    Path("tools/sanitize_release.py"),
    Path("tools/README.md"),
    Path(".github/workflows/sync_public_main.yml"),
}
REQUIRED_MOD_FIELDS = {
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
ALLOWED_MOD_FIELDS = REQUIRED_MOD_FIELDS | {
    "events",
    "runtime_profiles",
    "speed_profiles",
    "transition_profiles",
    "states",
    "routes",
    "actions",
    "nodes",
}


@dataclass(frozen=True)
class ModInfo:
    id: str
    root: Path
    requires: tuple[str, ...]
    protected: bool


def load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as input_file:
        value: object = yaml.safe_load(input_file) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Mod manifest must be a map: {path}")
    return cast(Mapping[str, object], value)


def validate_runtime_requirements(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a map")
    expected = {
        "python": ("import", r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"),
        "ros": ("package", r"[a-z][a-z0-9_]*"),
        "system": ("library", r"[A-Za-z0-9][A-Za-z0-9_.+-]*"),
    }
    if set(value) != set(expected):
        raise ValueError(f"{context} must contain exactly {sorted(expected)}")
    for category, (field, pattern) in expected.items():
        entries = value[category]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"{context}.{category} must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != {field}:
                raise ValueError(
                    f"{context}.{category}[{index}] must contain only '{field}'"
                )
            name = entry[field]
            if not isinstance(name, str) or not re.fullmatch(pattern, name):
                raise ValueError(f"{context}.{category}[{index}].{field} is invalid")


def validate_runtime_profiles(value: object, context: str) -> set[str]:
    """Validate the manifest subset parsed by runtime_profiles.py.

    The release sanitizer intentionally stays independent from ROS runtime
    imports, so this mirrors the language-neutral profile schema used by the
    framework loader.
    """

    if value is None:
        return set()
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a map")

    names: set[str] = set()
    for name, raw_profile in value.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*", name
        ):
            raise ValueError(f"{context} has invalid profile name: {name!r}")
        profile_context = f"{context}.{name}"
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"{profile_context} must be a map")
        if "candidates" in raw_profile:
            if set(raw_profile) != {"candidates"}:
                raise ValueError(
                    f"{profile_context} cannot combine candidates with "
                    "single-candidate fields"
                )
            candidates = raw_profile["candidates"]
            if (
                not isinstance(candidates, Sequence)
                or isinstance(candidates, (str, bytes))
                or not candidates
            ):
                raise ValueError(
                    f"{profile_context}.candidates must be a non-empty list"
                )
            for index, candidate in enumerate(candidates):
                validate_runtime_candidate(
                    candidate,
                    f"{profile_context}.candidates[{index}]",
                )
        else:
            validate_runtime_candidate(raw_profile, profile_context)
        names.add(name)
    return names


def validate_runtime_candidate(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a map")
    allowed = {
        "mode",
        "root",
        "python",
        "executable_paths",
        "library_paths",
        "isolated",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")

    mode = value.get("mode")
    if mode not in ("host", "vendor", "portable"):
        raise ValueError(f"{context}.mode must be 'host', 'vendor' or 'portable'")
    root = value.get("root")
    python = value.get("python")
    if mode == "portable":
        if not isinstance(root, str) or not root:
            raise ValueError(f"{context}.root is required for portable mode")
        if python is not None and (not isinstance(python, str) or not python):
            raise ValueError(f"{context}.python must be a non-empty relative path")
    elif root is not None or python is not None:
        raise ValueError(f"{context}.root/python are only valid for portable mode")

    for field in ("executable_paths", "library_paths"):
        entries = value.get(field, ())
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"{context}.{field} must be a list")
        if not all(isinstance(entry, str) for entry in entries):
            raise ValueError(f"{context}.{field} entries must be strings")
        if mode != "portable" and entries:
            raise ValueError(f"{context}.{field} is only valid for portable mode")
    isolated = value.get("isolated", mode == "portable")
    if not isinstance(isolated, bool):
        raise ValueError(f"{context}.isolated must be a boolean")


def validate_node_declaration(
    node: Mapping[str, object],
    context: str,
) -> None:
    allowed_fields = {
        "entrypoint",
        "runtime",
        "execution",
        "lifecycle",
        "states",
        "params",
        "arguments",
        "remappings",
        "namespace",
        "manifest",
        "restart",
        "runtime_requirements",
        "interpreter",
        "environment",
        "cwd",
        "depends_on",
        "shutdown",
        "runtime_profile",
        "scheduling",
    }
    unknown_fields = set(node) - allowed_fields
    if unknown_fields:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown_fields)}")
    entrypoint = node.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ValueError(f"{context}.entrypoint must be a non-empty string")
    runtime = node.get("runtime", "python")
    if runtime not in ("python", "executable", "ros", "command"):
        raise ValueError(
            f"{context}.runtime must be 'python', 'executable', 'ros' or 'command'"
        )
    if runtime == "python" and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
        r"[A-Za-z_][A-Za-z0-9_]*",
        entrypoint,
    ):
        raise ValueError(f"{context}.entrypoint must look like 'module:function'")
    if runtime in ("executable", "command"):
        relative = Path(entrypoint)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValueError(f"{context}.entrypoint is not a safe relative path")
    if runtime == "ros" and not re.fullmatch(
        r"[a-z][a-z0-9_]*:[A-Za-z0-9_][A-Za-z0-9_.+-]*",
        entrypoint,
    ):
        raise ValueError(f"{context}.entrypoint must look like 'package:executable'")
    interpreter = node.get("interpreter")
    if runtime == "command":
        if interpreter is not None and (
            not isinstance(interpreter, str) or not interpreter.strip()
        ):
            raise ValueError(f"{context}.interpreter must be a non-empty string")
    elif interpreter is not None:
        raise ValueError(f"{context}.interpreter requires runtime command")
    execution = node.get(
        "execution",
        "in_process" if runtime == "python" else "process",
    )
    if execution not in ("in_process", "process"):
        raise ValueError(f"{context}.execution is invalid")
    if runtime != "python" and execution != "process":
        raise ValueError(f"{context}.execution must be 'process' for {runtime}")
    scheduling = node.get("scheduling", {})
    if not isinstance(scheduling, Mapping):
        raise ValueError(f"{context}.scheduling must be a map")
    unknown_scheduling = set(scheduling) - {"cpu_affinity"}
    if unknown_scheduling:
        raise ValueError(
            f"{context}.scheduling has unknown fields: {sorted(unknown_scheduling)}"
        )
    if execution != "process" and "scheduling" in node:
        raise ValueError(f"{context}.scheduling requires process execution")
    read_cpu_affinity(
        scheduling.get("cpu_affinity"),
        f"{context}.scheduling.cpu_affinity",
        default=CpuAffinityRole.SHARED,
    )
    runtime_profile = node.get("runtime_profile")
    if runtime_profile is not None and (
        not isinstance(runtime_profile, str) or not runtime_profile
    ):
        raise ValueError(f"{context}.runtime_profile must be a non-empty string")
    lifecycle = node.get("lifecycle", "mod")
    if lifecycle not in ("mod", "state"):
        raise ValueError(f"{context}.lifecycle is invalid")
    states = node.get("states", ())
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ValueError(f"{context}.states must be a list")
    if not all(isinstance(state, str) and state for state in states):
        raise ValueError(f"{context}.states entries must be non-empty strings")
    if lifecycle == "state" and not states:
        raise ValueError(f"{context}.states is required for state lifecycle")
    if lifecycle == "mod" and states:
        raise ValueError(f"{context}.states is only valid for state lifecycle")
    arguments = node.get("arguments", ())
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise ValueError(f"{context}.arguments must be a list")
    if not all(isinstance(argument, str) for argument in arguments):
        raise ValueError(f"{context}.arguments entries must be strings")
    remappings = node.get("remappings", {})
    if not isinstance(remappings, Mapping) or not all(
        isinstance(source, str)
        and bool(source)
        and isinstance(target, str)
        and bool(target)
        for source, target in remappings.items()
    ):
        raise ValueError(f"{context}.remappings must map non-empty strings")
    namespace = node.get("namespace", "")
    if not isinstance(namespace, str) or (
        namespace
        and not re.fullmatch(
            r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*",
            namespace,
        )
    ):
        raise ValueError(f"{context}.namespace is invalid")
    params = node.get("params", {})
    if runtime == "command" and (params or remappings or namespace):
        raise ValueError(
            f"{context} command nodes do not accept params, remappings or namespace"
        )
    cwd = node.get("cwd")
    if runtime == "command":
        if cwd is not None and (
            not isinstance(cwd, str)
            or not cwd
            or Path(cwd).is_absolute()
            or any(part in ("", "..") for part in Path(cwd).parts)
        ):
            raise ValueError(f"{context}.cwd must be a safe relative path")
    elif cwd is not None:
        raise ValueError(f"{context}.cwd requires runtime command")
    depends_on = node.get("depends_on", ())
    if not isinstance(depends_on, Sequence) or isinstance(depends_on, (str, bytes)):
        raise ValueError(f"{context}.depends_on must be a list")
    if not all(isinstance(item, str) and item for item in depends_on):
        raise ValueError(f"{context}.depends_on entries must be non-empty strings")
    environment = node.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ValueError(f"{context}.environment must be a map")
    if runtime != "command" and environment:
        raise ValueError(f"{context}.environment requires runtime command")
    for name, edit in environment.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError(f"{context}.environment has an invalid name")
        if isinstance(edit, str):
            continue
        if not isinstance(edit, Mapping) or set(edit) - {
            "set",
            "prepend",
            "append",
            "separator",
            "existing_only",
        }:
            raise ValueError(f"{context}.environment.{name} is invalid")
        for field in ("prepend", "append"):
            entries = edit.get(field, ())
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                raise ValueError(f"{context}.environment.{name}.{field} is invalid")
            if not all(isinstance(item, str) for item in entries):
                raise ValueError(f"{context}.environment.{name}.{field} is invalid")
        if "set" in edit and not isinstance(edit["set"], str):
            raise ValueError(f"{context}.environment.{name}.set is invalid")
        if "separator" in edit and (
            not isinstance(edit["separator"], str) or not edit["separator"]
        ):
            raise ValueError(f"{context}.environment.{name}.separator is invalid")
        if "existing_only" in edit and not isinstance(edit["existing_only"], bool):
            raise ValueError(f"{context}.environment.{name}.existing_only is invalid")
    for mapping_name in ("params", "manifest", "restart", "shutdown"):
        value = node.get(mapping_name, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"{context}.{mapping_name} must be a map")
    manifest = cast(Mapping[str, object], node.get("manifest", {}))
    if (
        not isinstance(manifest.get("label"), str)
        or not cast(str, manifest.get("label")).strip()
    ):
        raise ValueError(f"{context}.manifest.label must be a non-empty string")
    restart = cast(Mapping[str, object], node.get("restart", {}))
    if execution != "process" and restart:
        raise ValueError(f"{context}.restart requires process execution")
    if set(restart) - {"max_attempts", "delay", "non_retryable_exit_codes"}:
        raise ValueError(f"{context}.restart has unknown fields")
    max_attempts = restart.get("max_attempts", 3)
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 0
    ):
        raise ValueError(f"{context}.restart.max_attempts is invalid")
    delay = restart.get("delay", 1.0)
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
        raise ValueError(f"{context}.restart.delay is invalid")
    non_retryable_exit_codes = restart.get("non_retryable_exit_codes", ())
    if not isinstance(non_retryable_exit_codes, Sequence) or isinstance(
        non_retryable_exit_codes, (str, bytes)
    ):
        raise ValueError(
            f"{context}.restart.non_retryable_exit_codes must be a list"
        )
    if not all(
        not isinstance(exit_code, bool)
        and isinstance(exit_code, int)
        and 1 <= exit_code <= 255
        for exit_code in non_retryable_exit_codes
    ):
        raise ValueError(
            f"{context}.restart.non_retryable_exit_codes entries must be "
            "integers from 1 to 255"
        )
    shutdown = cast(Mapping[str, object], node.get("shutdown", {}))
    if execution != "process" and shutdown:
        raise ValueError(f"{context}.shutdown requires process execution")
    if set(shutdown) - {"signal", "terminate_after", "kill_after"}:
        raise ValueError(f"{context}.shutdown has unknown fields")
    if shutdown.get("signal", "SIGTERM") not in {
        "SIGHUP",
        "SIGINT",
        "SIGQUIT",
        "SIGTERM",
    }:
        raise ValueError(f"{context}.shutdown.signal is invalid")
    terminate_after = shutdown.get("terminate_after")
    kill_after = shutdown.get("kill_after", 3.0)
    for field, value in (
        ("terminate_after", terminate_after),
        ("kill_after", kill_after),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        ):
            raise ValueError(f"{context}.shutdown.{field} is invalid")
    if terminate_after is not None and terminate_after >= kill_after:
        raise ValueError(
            f"{context}.shutdown.terminate_after must be less than kill_after"
        )
    if "runtime_requirements" in node:
        validate_runtime_requirements(
            node["runtime_requirements"],
            f"{context}.runtime_requirements",
        )


def discover_mods(source_root: Path, mod_roots: Sequence[Path]) -> dict[str, ModInfo]:
    mods: dict[str, ModInfo] = {}
    for raw_root in mod_roots:
        root = raw_root if raw_root.is_absolute() else source_root / raw_root
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("mod.yaml")):
            manifest = load_yaml(manifest_path)
            missing_fields = REQUIRED_MOD_FIELDS - set(manifest)
            if missing_fields:
                raise ValueError(
                    f"missing explicit Mod fields in {manifest_path}: "
                    f"{sorted(missing_fields)}"
                )
            unknown_fields = set(manifest) - ALLOWED_MOD_FIELDS
            if unknown_fields:
                raise ValueError(
                    f"unknown Mod fields in {manifest_path}: "
                    f"{sorted(unknown_fields)}"
                )
            if manifest["schema"] != 1:
                raise ValueError(f"unsupported Mod schema: {manifest_path}")
            api = manifest["api"]
            if not isinstance(api, str) or not api:
                raise ValueError(f"invalid Mod API constraint: {manifest_path}")
            try:
                api_compatible = version_matches(MOD_API_VERSION, api)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Mod API constraint in {manifest_path}: {exc}"
                ) from exc
            if not api_compatible:
                raise ValueError(
                    f"Mod API mismatch in {manifest_path}: requires {api!r}, "
                    f"framework provides {MOD_API_VERSION!r}"
                )
            mod_id = manifest.get("id")
            if not isinstance(mod_id, str) or not re.fullmatch(
                r"[a-z0-9]+(?:[._-][a-z0-9]+)+", mod_id
            ):
                raise ValueError(f"invalid Mod id: {manifest_path}")
            if mod_id in mods:
                raise ValueError(f"duplicate Mod id in release tree: {mod_id}")
            if not isinstance(manifest["name"], str) or not manifest["name"].strip():
                raise ValueError(f"invalid Mod name: {manifest_path}")
            version = manifest["version"]
            if not isinstance(version, str):
                raise ValueError(f"invalid Mod version: {manifest_path}")
            try:
                parse_numeric_version(version)
            except ValueError as exc:
                raise ValueError(f"invalid Mod version: {manifest_path}") from exc
            if not isinstance(manifest["enable"], bool):
                raise ValueError(f"enable must be a boolean: {manifest_path}")
            entrypoint = manifest["entrypoint"]
            if entrypoint is not None and (
                not isinstance(entrypoint, str) or not entrypoint
            ):
                raise ValueError(
                    f"entrypoint must be null or a non-empty string: {manifest_path}"
                )
            if manifest["visibility"] not in ("public", "protected"):
                raise ValueError(
                    f"visibility must be public or protected: {manifest_path}"
                )
            python_exports = manifest["python_exports"]
            if (
                not isinstance(python_exports, Sequence)
                or isinstance(python_exports, (str, bytes))
                or not all(
                    isinstance(item, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item)
                    for item in python_exports
                )
            ):
                raise ValueError(f"invalid python_exports: {manifest_path}")
            validate_runtime_requirements(
                manifest["runtime_requirements"],
                f"{manifest_path}: runtime_requirements",
            )
            runtime_profile_names = validate_runtime_profiles(
                manifest.get("runtime_profiles"),
                f"{manifest_path}: runtime_profiles",
            )
            raw_requires = manifest["requires"]
            if not isinstance(raw_requires, Sequence) or isinstance(
                raw_requires, (str, bytes)
            ):
                raise ValueError(f"requires must be a list: {manifest_path}")
            requires: list[str] = []
            for item in raw_requires:
                if isinstance(item, str):
                    requires.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    requirement_version = item.get("version")
                    if requirement_version is not None:
                        if not isinstance(requirement_version, str):
                            raise ValueError(
                                f"invalid requirement version in {manifest_path}: "
                                f"{requirement_version!r}"
                            )
                        try:
                            parse_version_constraint(requirement_version)
                        except ValueError as exc:
                            raise ValueError(
                                f"invalid requirement version in {manifest_path}: "
                                f"{requirement_version!r}"
                            ) from exc
                    requires.append(cast(str, item["id"]))
                else:
                    raise ValueError(
                        f"invalid requirement in {manifest_path}: {item!r}"
                    )
            raw_conflicts = manifest["conflicts"]
            if not isinstance(raw_conflicts, Sequence) or isinstance(
                raw_conflicts, (str, bytes)
            ):
                raise ValueError(f"conflicts must be a list: {manifest_path}")
            conflicts: list[str] = []
            for conflict in raw_conflicts:
                if not isinstance(conflict, str) or not re.fullmatch(
                    r"[a-z0-9]+(?:[._-][a-z0-9]+)+", conflict
                ):
                    raise ValueError(
                        f"invalid conflict in {manifest_path}: {conflict!r}"
                    )
                if conflict == mod_id:
                    raise ValueError(f"Mod conflicts with itself: {manifest_path}")
                if conflict in conflicts:
                    raise ValueError(
                        f"duplicate conflict in {manifest_path}: {conflict}"
                    )
                conflicts.append(conflict)
            for collection_name, reference_fields in (
                ("routes", ("from", "to", "event")),
                ("actions", ("from", "event")),
            ):
                raw_rules = manifest.get(collection_name, ())
                if not isinstance(raw_rules, Sequence) or isinstance(
                    raw_rules, (str, bytes)
                ):
                    raise ValueError(
                        f"{collection_name} must be a list: {manifest_path}"
                    )
                for rule in raw_rules:
                    if not isinstance(rule, Mapping):
                        raise ValueError(
                            f"invalid {collection_name} entry in "
                            f"{manifest_path}: {rule!r}"
                        )
                    for field in reference_fields:
                        reference = rule.get(field)
                        if not isinstance(reference, str) or "/" not in reference:
                            continue
                        owner = reference.split("/", 1)[0]
                        if owner != mod_id:
                            requires.append(owner)
            raw_nodes = manifest.get("nodes", {})
            if not isinstance(raw_nodes, Mapping):
                raise ValueError(f"nodes must be a map: {manifest_path}")
            for node_name, raw_node in raw_nodes.items():
                if not isinstance(node_name, str) or not isinstance(raw_node, Mapping):
                    raise ValueError(
                        f"invalid node declaration in {manifest_path}: "
                        f"{node_name!r}={raw_node!r}"
                    )
                validate_node_declaration(
                    raw_node,
                    f"{manifest_path}: nodes.{node_name}",
                )
                runtime_profile = raw_node.get("runtime_profile")
                if (
                    isinstance(runtime_profile, str)
                    and runtime_profile not in runtime_profile_names
                ):
                    raise ValueError(
                        f"{manifest_path}: nodes.{node_name}.runtime_profile "
                        f"references unknown profile {runtime_profile!r}"
                    )
                raw_states = raw_node.get("states", ())
                if not isinstance(raw_states, Sequence) or isinstance(
                    raw_states, (str, bytes)
                ):
                    raise ValueError(
                        f"nodes.{node_name}.states must be a list: {manifest_path}"
                    )
                for reference in raw_states:
                    if not isinstance(reference, str) or "/" not in reference:
                        continue
                    owner = reference.split("/", 1)[0]
                    if owner != mod_id:
                        requires.append(owner)
            mods[mod_id] = ModInfo(
                id=mod_id,
                root=manifest_path.parent.relative_to(source_root),
                requires=tuple(dict.fromkeys(requires)),
                protected=manifest["visibility"] == "protected",
            )
    if not mods:
        raise ValueError("no Mods found for release")
    return mods


def copy_release_tree(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root:
        raise ValueError("output directory must not be the repository root")
    if output_root.exists():
        shutil.rmtree(output_root)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {
            ".git",
            ".wiki",
            ".agents",
            ".codex",
            "build",
            "install",
            "log",
            "dist",
            "update",
            ".update",
            "__pycache__",
            ".pytest_cache",
        }
        return {name for name in names if name in ignored}

    shutil.copytree(source_root, output_root, ignore=ignore)


def dependency_closure(mods: Mapping[str, ModInfo], roots: set[str]) -> set[str]:
    closure: set[str] = set()

    def visit(mod_id: str) -> None:
        if mod_id in closure:
            return
        mod = mods.get(mod_id)
        if mod is None:
            raise ValueError(f"release references missing Mod dependency: {mod_id}")
        closure.add(mod_id)
        for dependency in mod.requires:
            visit(dependency)

    for mod_id in sorted(roots):
        visit(mod_id)
    return closure


def sanitize_release(
    source_root: Path,
    output_root: Path,
    mod_roots: Sequence[Path],
    explicit_excludes: set[str],
    self_check: bool,
) -> None:
    mods = discover_mods(source_root, mod_roots)
    excluded = {mod.id for mod in mods.values() if mod.protected}
    excluded.update(explicit_excludes)
    unknown = excluded - set(mods)
    if unknown:
        raise ValueError(f"cannot exclude unknown Mods: {sorted(unknown)}")

    public_roots = set(mods) - excluded
    required_public = dependency_closure(mods, public_roots)
    protected_dependencies = required_public & excluded
    if protected_dependencies:
        users = {
            mod.id: sorted(set(mod.requires) & protected_dependencies)
            for mod in mods.values()
            if not mod.protected and set(mod.requires) & protected_dependencies
        }
        raise ValueError(
            "public Mods depend on excluded Mods: "
            f"dependencies={sorted(protected_dependencies)}, users={users}"
        )

    copy_release_tree(source_root, output_root)
    for mod_id in sorted(excluded):
        shutil.rmtree(output_root / mods[mod_id].root)
    for path in PUBLIC_DEV_ONLY_PATHS:
        target = output_root / path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    if self_check:
        for mod_id in excluded:
            if (output_root / mods[mod_id].root).exists():
                raise RuntimeError(f"excluded Mod remains in release: {mod_id}")
        output_mods = discover_mods(output_root, mod_roots)
        if set(output_mods) != public_roots:
            raise RuntimeError(
                "public Mod set mismatch: "
                f"expected={sorted(public_roots)}, actual={sorted(output_mods)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a public tree by removing protected Mod folders."
    )
    parser.add_argument(
        "--mod-root",
        action="append",
        default=None,
        help="Mod root relative to the repository; can be repeated",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="additional Mod id to exclude; can be repeated",
    )
    parser.add_argument("--out", default="dist/public_release")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path.cwd()
    mod_roots = (
        tuple(Path(path) for path in args.mod_root)
        if args.mod_root
        else (DEFAULT_MOD_ROOT,)
    )
    output_root = Path(args.out)
    sanitize_release(
        source_root,
        output_root,
        mod_roots,
        set(args.exclude),
        args.self_check,
    )
    print(f"public release tree generated at: {output_root}")


if __name__ == "__main__":
    main()
