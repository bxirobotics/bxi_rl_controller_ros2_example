from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bxi_example_py_elf3.framework.mod_api.transition import (
    ConfigReader,
    SingleClassTransition,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class InstantTransition(SingleClassTransition):
    type_name = "instant"

    @classmethod
    def from_config(
        cls,
        name: str,
        raw: Mapping[str, object],
    ) -> "InstantTransition":
        reader = ConfigReader(raw, name)
        reader.finish()
        return cls(name, 0.0)

    def apply(self, ctx: "RobotControlContext", dt: float, progress: float) -> None:
        pass
