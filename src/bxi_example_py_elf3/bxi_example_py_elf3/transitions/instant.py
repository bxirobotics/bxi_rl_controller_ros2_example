from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bxi_example_py_elf3.utils.transition_core import (
    ConfigReader,
    SingleClassTransition,
)

if TYPE_CHECKING:
    from bxi_example_py_elf3.bxi_example_demo import BxiExample


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

    def apply(self, ctx: "BxiExample", dt: float, progress: float) -> None:
        pass
