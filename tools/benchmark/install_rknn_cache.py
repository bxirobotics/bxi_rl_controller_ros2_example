#!/usr/bin/env python3
"""Install benchmark RKNN cache files beside their source ONNX models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "tools/benchmark/cache/rknn"


def _same_file_contents(left: Path, right: Path) -> bool:
    if not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False

    chunk_size = 1024 * 1024
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(chunk_size)
            right_chunk = right_file.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def install_cache(cache_root: Path, *, dry_run: bool) -> int:
    if not cache_root.is_dir():
        print(
            f"error: RKNN cache directory does not exist: {cache_root}", file=sys.stderr
        )
        return 2

    cache_files = sorted(cache_root.rglob("*.rknn"))
    if not cache_files:
        print(f"error: no .rknn files found under {cache_root}", file=sys.stderr)
        return 1

    installed = 0
    unchanged = 0
    skipped = 0
    failed = 0

    for source in cache_files:
        relative = source.relative_to(cache_root)
        destination = ROOT / relative
        source_manifest = Path(str(source) + ".build.json")
        destination_manifest = Path(str(destination) + ".build.json")
        onnx_path = destination.with_suffix(".onnx")

        try:
            destination.resolve().relative_to(ROOT)
        except ValueError:
            print(f"SKIP  unsafe cache path: {relative}")
            skipped += 1
            continue

        if not onnx_path.is_file():
            print(f"SKIP  no matching ONNX: {onnx_path.relative_to(ROOT)}")
            skipped += 1
            continue
        if not source_manifest.is_file():
            print(
                f"SKIP  no RKNN build contract: {source_manifest.relative_to(cache_root)}"
            )
            skipped += 1
            continue

        try:
            model_matches = _same_file_contents(source, destination)
            manifest_matches = _same_file_contents(
                source_manifest, destination_manifest
            )
            if model_matches and manifest_matches:
                print(f"SAME  {destination.relative_to(ROOT)}")
                unchanged += 1
                continue

            action = "WOULD INSTALL" if dry_run else "INSTALL"
            print(f"{action:<13} {destination.relative_to(ROOT)}")
            if not dry_run:
                if not model_matches:
                    _atomic_copy(source, destination)
                if not manifest_matches:
                    _atomic_copy(source_manifest, destination_manifest)
            installed += 1
        except OSError as exc:
            print(f"FAIL  {destination.relative_to(ROOT)}: {exc}", file=sys.stderr)
            failed += 1

    verb = "would install" if dry_run else "installed"
    print(
        f"Summary: {installed} {verb}, {unchanged} unchanged, "
        f"{skipped} skipped, {failed} failed"
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy cached RKNN models beside their corresponding repository ONNX models. "
            "Existing adjacent RKNN files are atomically updated."
        )
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"RKNN cache root (default: {DEFAULT_CACHE.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be installed without changing files",
    )
    args = parser.parse_args()
    return install_cache(args.cache.expanduser().resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
