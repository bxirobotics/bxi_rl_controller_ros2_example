#!/usr/bin/env python3
"""Synchronize ELF3 MuJoCo body inertials from their authoritative URDFs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = (
    (
        REPOSITORY_ROOT
        / "src/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml",
        REPOSITORY_ROOT / "resources/elf3_dof31/urdf/elf3.urdf",
    ),
    (
        REPOSITORY_ROOT
        / "src/bxi_example_py_elf3/data/mujoco_simulation/elf3_hand.xml",
        REPOSITORY_ROOT / "resources/elf3_dof31_hand/urdf/elf3.urdf",
    ),
)

_BODY_INERTIAL_PATTERN = re.compile(
    r'(?P<body><body\s+name="(?P<name>[^"]+)"[^>]*>\n)'
    r'(?P<indent>[ \t]*)<inertial\b[^>]*/>',
)


@dataclass(frozen=True, slots=True)
class UrdfInertial:
    position: str
    mass: str
    full_inertia: str

    def as_mjcf(self, indent: str) -> str:
        return (
            f'{indent}<inertial pos="{self.position}" mass="{self.mass}" '
            f'fullinertia="{self.full_inertia}"/>'
        )


def _normalized_triplet(value: str | None) -> str:
    fields = "0 0 0" if value is None else value
    parts = fields.split()
    if len(parts) != 3:
        raise ValueError(f"expected three values, got {fields!r}")
    return " ".join(parts)


def load_urdf_inertials(path: Path) -> dict[str, UrdfInertial]:
    root = ElementTree.parse(path).getroot()
    result: dict[str, UrdfInertial] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        name = link.attrib["name"]
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if origin is None or mass is None or inertia is None:
            raise ValueError(f"URDF link {name!r} has an incomplete inertial")
        rpy = _normalized_triplet(origin.attrib.get("rpy"))
        if any(float(value) != 0.0 for value in rpy.split()):
            raise ValueError(
                f"URDF link {name!r} uses inertial rpy={rpy}; rotate its inertia "
                "into the link frame before emitting MJCF fullinertia"
            )
        full_inertia = " ".join(
            inertia.attrib[field]
            for field in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        )
        result[name] = UrdfInertial(
            position=_normalized_triplet(origin.attrib.get("xyz")),
            mass=mass.attrib["value"],
            full_inertia=full_inertia,
        )
    return result


def render_synced_mjcf(mjcf_path: Path, urdf_path: Path) -> tuple[str, int]:
    source = mjcf_path.read_text(encoding="utf-8")
    urdf_inertials = load_urdf_inertials(urdf_path)
    replaced: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        inertial = urdf_inertials.get(name)
        if inertial is None:
            return match.group(0)
        replaced.add(name)
        return match.group("body") + inertial.as_mjcf(match.group("indent"))

    rendered = _BODY_INERTIAL_PATTERN.sub(replace, source)
    missing = sorted(set(urdf_inertials) - replaced)
    if missing:
        raise ValueError(
            f"{mjcf_path}: URDF inertial links missing from MJCF bodies: {missing}"
        )

    mjcf_root = ElementTree.fromstring(rendered)
    extra = sorted(
        body.attrib["name"]
        for body in mjcf_root.findall(".//body")
        if body.find("inertial") is not None
        and body.attrib["name"] not in urdf_inertials
    )
    if extra:
        raise ValueError(
            f"{mjcf_path}: MJCF inertial bodies missing from URDF: {extra}"
        )
    return rendered, len(replaced)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail if an MJCF file differs from its URDF (default)",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="rewrite MJCF inertials from the URDF",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for mjcf_path, urdf_path in DEFAULT_PAIRS:
        rendered, count = render_synced_mjcf(mjcf_path, urdf_path)
        current = mjcf_path.read_text(encoding="utf-8")
        if current != rendered:
            stale.append(mjcf_path)
            if args.write:
                mjcf_path.write_text(rendered, encoding="utf-8")
                print(f"updated {mjcf_path.relative_to(REPOSITORY_ROOT)} ({count} links)")
            else:
                print(
                    f"stale {mjcf_path.relative_to(REPOSITORY_ROOT)}",
                    file=sys.stderr,
                )
        else:
            print(f"ok {mjcf_path.relative_to(REPOSITORY_ROOT)} ({count} links)")

    if stale and not args.write:
        print("run tools/sync_mujoco_inertials.py --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
