"""Version of the stable API exposed to Mods."""

from __future__ import annotations

import re


MOD_API_VERSION = "4.0.0"


def parse_numeric_version(version: str) -> tuple[int, ...]:
    """Parse the numeric dot versions used by the Mod system."""
    if not re.fullmatch(r"\d+(?:\.\d+)*", version):
        raise ValueError(f"version must be a numeric dot version: {version!r}")
    return tuple(int(part) for part in version.split("."))


def parse_version_constraint(
    constraint: str,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Parse a comma-separated Mod version constraint."""
    clauses: list[tuple[str, tuple[int, ...]]] = []
    for raw_clause in constraint.split(","):
        clause = raw_clause.strip()
        match = re.fullmatch(r"(==|!=|>=|<=|>|<)\s*(\d+(?:\.\d+)*)", clause)
        if match is None:
            raise ValueError(f"unsupported version constraint: {constraint!r}")
        clauses.append((match.group(1), parse_numeric_version(match.group(2))))
    return tuple(clauses)


def version_matches(version: str, constraint: str) -> bool:
    """Return whether a numeric dot version satisfies a constraint."""
    actual = parse_numeric_version(version)
    for operator, expected in parse_version_constraint(constraint):
        width = max(len(actual), len(expected))
        left = actual + (0,) * (width - len(actual))
        right = expected + (0,) * (width - len(expected))
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


__all__ = ["MOD_API_VERSION"]
