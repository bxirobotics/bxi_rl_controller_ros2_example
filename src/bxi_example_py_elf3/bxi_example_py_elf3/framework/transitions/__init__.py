"""Explicit registry of framework-provided Transition plugins."""

from bxi_example_py_elf3.framework.mod_api.transition import TransitionPlugin

from .entry_gain_ramp import EntryGainRampTransition
from .hold import HoldTransition
from .instant import InstantTransition
from .running_blend import RunningBlendTransition
from .sequence import SequenceTransition


BUILTIN_TRANSITION_PLUGINS: dict[str, type[TransitionPlugin]] = {
    plugin.type_name: plugin
    for plugin in (
        EntryGainRampTransition,
        HoldTransition,
        InstantTransition,
        RunningBlendTransition,
        SequenceTransition,
    )
}


__all__ = ["BUILTIN_TRANSITION_PLUGINS"]
