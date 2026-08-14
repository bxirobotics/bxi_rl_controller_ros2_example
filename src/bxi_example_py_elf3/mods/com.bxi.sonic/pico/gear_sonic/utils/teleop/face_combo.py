"""Edge detection for overlapping PICO face-button combinations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceComboEdges:
    """Pressed and rising-edge state for the manager's face-button combos."""

    ax_pressed: bool
    by_pressed: bool
    start_pressed: bool
    ax_rising: bool
    by_rising: bool
    start_rising: bool
    rearmed: bool
    release_required: bool


class FaceComboEdgeDetector:
    """Detect combo edges while preventing a start combo from leaking into AX/BY.

    ``A+B+X+Y`` necessarily contains both ``A+X`` and ``B+Y``.  Once the start
    combo fires, all combo edges remain disarmed until one sampled frame shows
    all four face buttons released.  This makes the release an explicit part of
    the input protocol instead of relying on the relative timing of two loops.
    """

    def __init__(self) -> None:
        self._previous_ax = False
        self._previous_by = False
        self._previous_start = False
        self._release_required = False

    @property
    def release_required(self) -> bool:
        return self._release_required

    def update(self, a: bool, b: bool, x: bool, y: bool) -> FaceComboEdges:
        a = bool(a)
        b = bool(b)
        x = bool(x)
        y = bool(y)

        ax_pressed = a and x
        by_pressed = b and y
        start_pressed = a and b and x and y
        all_released = not (a or b or x or y)

        rearmed = False
        if self._release_required and all_released:
            self._release_required = False
            rearmed = True

        edges_enabled = not self._release_required
        start_rising = (
            edges_enabled and start_pressed and not self._previous_start
        )
        ax_rising = edges_enabled and ax_pressed and not self._previous_ax
        by_rising = edges_enabled and by_pressed and not self._previous_by

        if start_rising:
            # The start command owns this sample.  AX/BY are merely subsets of
            # ABXY and must not become independent commands until full release.
            ax_rising = False
            by_rising = False
            self._release_required = True

        self._previous_ax = ax_pressed
        self._previous_by = by_pressed
        self._previous_start = start_pressed

        return FaceComboEdges(
            ax_pressed=ax_pressed,
            by_pressed=by_pressed,
            start_pressed=start_pressed,
            ax_rising=ax_rising,
            by_rising=by_rising,
            start_rising=start_rising,
            rearmed=rearmed,
            release_required=self._release_required,
        )

