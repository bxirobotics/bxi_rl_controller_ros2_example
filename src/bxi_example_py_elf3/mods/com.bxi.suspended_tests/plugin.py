"""Lightweight entrypoint for the suspended motor-test Mod."""

from dataclasses import dataclass

from .state import (
    SuspendedLimbTestState,
    SuspendedRunningState,
    SuspendedVibrationState,
)


@dataclass(frozen=True)
class SuspendedTestsMod:
    """Minimal Mod definition used by the lightweight controller branch."""

    button_states: tuple


def create_button_states():
    return (
        SuspendedRunningState(),
        SuspendedVibrationState(),
        SuspendedLimbTestState(),
    )


def create_mod(context=None):
    return SuspendedTestsMod(button_states=create_button_states())
