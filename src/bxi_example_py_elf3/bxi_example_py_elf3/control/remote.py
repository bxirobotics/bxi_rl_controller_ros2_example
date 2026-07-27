"""Controller-independent processing of persistent or momentary buttons."""


class RemoteButtonEdge:
    """Convert a MotionCommands button value into one action per press.

    ``toggle`` matches this repository's C++ gamepad node: the published value
    changes once for each physical press. ``momentary`` matches keyboard-style
    sources that publish 1 while held and 0 after release.

    The first sample, and the first sample after a publisher gap, only
    synchronizes state. This prevents a stale persistent button value from
    starting a robot motion when a controller node is restarted.
    """

    MODES = frozenset(("toggle", "momentary"))

    def __init__(self, mode="toggle", resync_sec=0.5):
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError("button mode must be 'toggle' or 'momentary'")
        if float(resync_sec) <= 0.0:
            raise ValueError("resync_sec must be > 0")
        self.mode = mode
        self.resync_sec = float(resync_sec)
        self.previous = None
        self.last_sample_at = 0.0

    def update(self, value, now):
        current = bool(value)
        now = float(now)
        if (
            self.previous is None
            or (
                self.last_sample_at > 0.0
                and now - self.last_sample_at > self.resync_sec
            )
        ):
            self.previous = current
            self.last_sample_at = now
            return False

        if self.mode == "toggle":
            activated = current != self.previous
        else:
            activated = current and not self.previous
        self.previous = current
        self.last_sample_at = now
        return activated

    def reset(self):
        self.previous = None
        self.last_sample_at = 0.0
