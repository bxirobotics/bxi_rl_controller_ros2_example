#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import cast

import yaml


DEFAULT_MOD_ROOT = Path("src/bxi_example_py_elf3/mods")
PUBLIC_DEV_ONLY_PATHS = {
    Path("tools/sanitize_release.py"),
    Path("tools/README.md"),
    Path(".github/workflows/sync_public_main.yml"),
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


def discover_mods(source_root: Path, mod_roots: Sequence[Path]) -> dict[str, ModInfo]:
    mods: dict[str, ModInfo] = {}
    for raw_root in mod_roots:
        root = raw_root if raw_root.is_absolute() else source_root / raw_root
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("mod.yaml")):
            manifest = load_yaml(manifest_path)
            mod_id = manifest.get("id")
            if not isinstance(mod_id, str) or not mod_id:
                raise ValueError(f"invalid Mod id: {manifest_path}")
            if mod_id in mods:
                raise ValueError(f"duplicate Mod id in release tree: {mod_id}")
            raw_requires = manifest.get("requires", ())
            if not isinstance(raw_requires, Sequence) or isinstance(
                raw_requires, (str, bytes)
            ):
                raise ValueError(f"requires must be a list: {manifest_path}")
            requires: list[str] = []
            for item in raw_requires:
                if isinstance(item, str):
                    requires.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    requires.append(cast(str, item["id"]))
                else:
                    raise ValueError(
                        f"invalid requirement in {manifest_path}: {item!r}"
                    )
            raw_routes = manifest.get("routes", ())
            if not isinstance(raw_routes, Sequence) or isinstance(
                raw_routes, (str, bytes)
            ):
                raise ValueError(f"routes must be a list: {manifest_path}")
            for route in raw_routes:
                if not isinstance(route, Mapping):
                    raise ValueError(f"invalid route in {manifest_path}: {route!r}")
                for field in ("from", "to", "event"):
                    reference = route.get(field)
                    if not isinstance(reference, str) or "/" not in reference:
                        continue
                    owner = reference.split("/", 1)[0]
                    if owner != mod_id:
                        requires.append(owner)
            mods[mod_id] = ModInfo(
                id=mod_id,
                root=manifest_path.parent.relative_to(source_root),
                requires=tuple(dict.fromkeys(requires)),
                protected=manifest.get("visibility", "public") == "protected",
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
