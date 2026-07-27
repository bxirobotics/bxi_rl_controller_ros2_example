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

from bxi_example_py_elf3.mod_api_version import (  # noqa: E402
    MOD_API_VERSION,
    parse_numeric_version,
    parse_version_constraint,
    version_matches,
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
                if "runtime_requirements" in raw_node:
                    validate_runtime_requirements(
                        raw_node["runtime_requirements"],
                        f"{manifest_path}: nodes.{node_name}.runtime_requirements",
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
