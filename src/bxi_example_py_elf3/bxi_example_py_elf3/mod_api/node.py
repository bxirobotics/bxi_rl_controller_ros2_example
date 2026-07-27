from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ModNode(Protocol):
    """Minimum contract returned by a Mod node factory."""

    def destroy_node(self) -> object:
        ...


@dataclass(frozen=True)
class NodeBuildContext:
    """Stable construction context for a Python node owned by a Mod."""

    mod_id: str
    node_id: str
    node_name: str
    mod_root: Path
    params: Mapping[str, object]
    arguments: tuple[str, ...] = ()
    remappings: Mapping[str, str] = field(default_factory=dict)
    namespace: str = ""

    def asset(self, relative_path: str) -> Path:
        path = (self.mod_root / relative_path).resolve()
        assets_root = (self.mod_root / "assets").resolve()
        if assets_root not in path.parents:
            raise ValueError(
                f"node in '{self.mod_id}' must use files from its assets folder: "
                f"{relative_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"node asset does not exist in '{self.mod_id}': {relative_path}"
            )
        return path


NodeFactory = Callable[[NodeBuildContext], ModNode]


__all__ = ["ModNode", "NodeBuildContext", "NodeFactory"]
