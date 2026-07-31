"""Framework-wide logger identities and subprocess stream routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import os
from queue import Empty, SimpleQueue
import re
import selectors
import subprocess
from threading import Event, Lock, Thread
import time
from typing import BinaryIO

from bxi_example_py_elf3.framework.mod_api.context import LoggerLike
from bxi_example_py_elf3.framework.platform.cpu_affinity import (
    configure_current_thread,
)


_LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "fatal": 50,
}
_WAKEUP = object()


def _display_scope(scope: str) -> str:
    if scope.startswith("framework."):
        return "fw." + scope.removeprefix("framework.")
    if scope.startswith("mod."):
        return scope.removeprefix("mod.")
    return scope


@dataclass(frozen=True)
class SubprocessLoggingConfig:
    max_line_bytes: int = 16_384
    max_lines_per_sec: int = 200


@dataclass(frozen=True)
class LoggingConfig:
    default_level: str = "info"
    levels: Mapping[str, str] = field(default_factory=dict)
    subprocess: SubprocessLoggingConfig = SubprocessLoggingConfig()

    @classmethod
    def from_mapping(cls, raw: object) -> "LoggingConfig":
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("logging must be a YAML map")
        unknown = set(raw) - {"default_level", "levels", "subprocess"}
        if unknown:
            raise ValueError(f"logging contains unknown fields: {sorted(unknown)}")

        default_level = _read_level(
            raw.get("default_level", "info"),
            "logging.default_level",
        )
        raw_levels = raw.get("levels", {})
        if not isinstance(raw_levels, Mapping) or not all(
            isinstance(key, str) and key for key in raw_levels
        ):
            raise ValueError("logging.levels must be a string-keyed map")
        levels = {
            _normalize_scope(key): _read_level(value, f"logging.levels.{key}")
            for key, value in raw_levels.items()
        }

        raw_subprocess = raw.get("subprocess", {})
        if not isinstance(raw_subprocess, Mapping):
            raise ValueError("logging.subprocess must be a YAML map")
        subprocess_unknown = set(raw_subprocess) - {
            "max_line_bytes",
            "max_lines_per_sec",
        }
        if subprocess_unknown:
            raise ValueError(
                "logging.subprocess contains unknown fields: "
                f"{sorted(subprocess_unknown)}"
            )
        max_line_bytes = _positive_integer(
            raw_subprocess.get("max_line_bytes", 16_384),
            "logging.subprocess.max_line_bytes",
        )
        max_lines_per_sec = _positive_integer(
            raw_subprocess.get("max_lines_per_sec", 200),
            "logging.subprocess.max_lines_per_sec",
        )
        return cls(
            default_level=default_level,
            levels=levels,
            subprocess=SubprocessLoggingConfig(
                max_line_bytes=max_line_bytes,
                max_lines_per_sec=max_lines_per_sec,
            ),
        )

    def level_for(self, scope: str) -> str:
        normalized = _normalize_scope(scope)
        matches = (
            (prefix, level)
            for prefix, level in self.levels.items()
            if normalized == prefix or normalized.startswith(prefix + ".")
        )
        selected = max(
            matches,
            key=lambda item: len(item[0]),
            default=("", self.default_level),
        )
        return selected[1]


class ScopedLoggers:
    """Create cached loggers with stable logical scopes and concise names."""

    def __init__(
        self,
        root: LoggerLike,
        config: LoggingConfig,
        *,
        logger_factory: Callable[[str], LoggerLike] | None = None,
    ) -> None:
        self._logger_factory = logger_factory or getattr(root, "get_child")
        self.config = config
        self._cache: dict[str, LoggerLike] = {}
        self._lock = Lock()

    def get(self, scope: str) -> LoggerLike:
        normalized = _normalize_scope(scope)
        with self._lock:
            logger = self._cache.get(normalized)
            if logger is not None:
                return logger
            logger = self._logger_factory(_display_scope(normalized))
            set_level = getattr(logger, "set_level")
            set_level(_LEVELS[self.config.level_for(normalized)])
            self._cache[normalized] = logger
            return logger

    def framework(self, component: str) -> LoggerLike:
        return self.get(f"framework.{component}")

    def mod(self, mod_id: str) -> LoggerLike:
        return self.get(f"mod.{mod_id}")

    def state(self, state_name: str) -> LoggerLike:
        mod_id, separator, local_name = state_name.partition("/")
        if not separator:
            raise ValueError(f"state name is not Mod-qualified: {state_name}")
        return self.get(f"mod.{mod_id}.state.{local_name}")

    def node(self, mod_id: str, local_name: str) -> LoggerLike:
        return self.get(f"mod.{mod_id}.node.{local_name}")


@dataclass
class _StreamState:
    fd: int
    stream: BinaryIO
    mod_id: str
    node_name: str
    stream_name: str
    output_fd: int
    buffer: bytearray = field(default_factory=bytearray)
    window_started: float = field(default_factory=time.monotonic)
    emitted_in_window: int = 0
    dropped_in_window: int = 0

    @property
    def prefix(self) -> bytes:
        stream = "out" if self.stream_name == "stdout" else "err"
        return f"[{self.mod_id}/{self.node_name}:{stream}] ".encode("utf-8")


class SubprocessLogRouter:
    """Drain every managed process without blocking a control or child thread."""

    def __init__(
        self,
        config: SubprocessLoggingConfig,
        *,
        cpu_affinity: frozenset[int],
    ) -> None:
        self._config = config
        self._cpu_affinity = cpu_affinity
        self._pending: SimpleQueue[_StreamState] = SimpleQueue()
        self._stop = Event()
        self._ready = Event()
        self._startup_error: BaseException | None = None
        self._wake_read_fd, self._wake_write_fd = os.pipe2(
            os.O_NONBLOCK | os.O_CLOEXEC
        )
        self._thread = Thread(
            target=self._run,
            name="bxi-subprocess-logs",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            os.close(self._wake_read_fd)
            os.close(self._wake_write_fd)
            raise RuntimeError(
                f"cannot initialize subprocess log router: {self._startup_error}"
            ) from self._startup_error

    def register(
        self,
        process: subprocess.Popen[bytes],
        *,
        mod_id: str,
        node_name: str,
    ) -> None:
        for stream_name, output_fd in (("stdout", 1), ("stderr", 2)):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue
            fd = stream.fileno()
            os.set_blocking(fd, False)
            self._pending.put(
                _StreamState(
                    fd,
                    stream,
                    mod_id,
                    node_name,
                    stream_name,
                    output_fd,
                )
            )
        self._wake()

    def close(self) -> None:
        self._stop.set()
        self._wake()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("subprocess log router did not stop within timeout")
        os.close(self._wake_read_fd)
        os.close(self._wake_write_fd)

    def _run(self) -> None:
        selector = selectors.DefaultSelector()
        selector.register(self._wake_read_fd, selectors.EVENT_READ, _WAKEUP)
        try:
            configure_current_thread(self._cpu_affinity, realtime_priority=0)
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._ready.set()
        if self._startup_error is not None:
            selector.close()
            return

        try:
            while not self._stop.is_set():
                self._register_pending(selector)
                for key, _mask in selector.select():
                    if key.data is _WAKEUP:
                        self._drain_wakeup()
                        self._register_pending(selector)
                    else:
                        self._read_stream(selector, key.data)
            self._register_pending(selector)
            for key in tuple(selector.get_map().values()):
                if key.data is not _WAKEUP:
                    self._drain_and_close(selector, key.data)
        finally:
            selector.close()

    def _wake(self) -> None:
        try:
            os.write(self._wake_write_fd, b"\0")
        except BlockingIOError:
            pass

    def _drain_wakeup(self) -> None:
        while True:
            try:
                if not os.read(self._wake_read_fd, 4096):
                    return
            except BlockingIOError:
                return

    def _register_pending(self, selector: selectors.BaseSelector) -> None:
        while True:
            try:
                state = self._pending.get_nowait()
            except Empty:
                return
            selector.register(state.fd, selectors.EVENT_READ, state)

    def _read_stream(
        self,
        selector: selectors.BaseSelector,
        state: _StreamState,
    ) -> None:
        try:
            chunk = os.read(state.fd, 65_536)
        except BlockingIOError:
            return
        except OSError:
            chunk = b""
        if not chunk:
            self._flush_remainder(state)
            self._close_stream(selector, state)
            return
        state.buffer.extend(chunk)
        self._emit_complete_lines(state)

    def _emit_complete_lines(self, state: _StreamState) -> None:
        limit = self._config.max_line_bytes
        while True:
            newline = state.buffer.find(b"\n")
            if newline >= 0 and newline + 1 <= limit:
                line = bytes(state.buffer[: newline + 1])
                del state.buffer[: newline + 1]
                self._emit(state, line)
                continue
            if len(state.buffer) >= limit:
                line = bytes(state.buffer[:limit]) + b" [line continued]\n"
                del state.buffer[:limit]
                self._emit(state, line)
                continue
            return

    def _flush_remainder(self, state: _StreamState) -> None:
        if state.buffer:
            self._emit(state, bytes(state.buffer) + b"\n")
            state.buffer.clear()
        self._emit_drop_summary(state)

    def _drain_and_close(
        self,
        selector: selectors.BaseSelector,
        state: _StreamState,
    ) -> None:
        while True:
            try:
                chunk = os.read(state.fd, 65_536)
            except (BlockingIOError, OSError):
                break
            if not chunk:
                break
            state.buffer.extend(chunk)
            self._emit_complete_lines(state)
        self._flush_remainder(state)
        self._close_stream(selector, state)

    def _emit(self, state: _StreamState, line: bytes) -> None:
        now = time.monotonic()
        if now - state.window_started >= 1.0:
            self._emit_drop_summary(state)
            state.window_started = now
            state.emitted_in_window = 0
            state.dropped_in_window = 0
        if state.emitted_in_window >= self._config.max_lines_per_sec:
            state.dropped_in_window += 1
            return
        state.emitted_in_window += 1
        self._write(state.output_fd, state.prefix + line)

    def _emit_drop_summary(self, state: _StreamState) -> None:
        if state.dropped_in_window <= 0:
            return
        self._write(
            state.output_fd,
            state.prefix
            + f"dropped {state.dropped_in_window} excessive log lines\n".encode(
                "utf-8"
            ),
        )

    @staticmethod
    def _write(fd: int, value: bytes) -> None:
        try:
            remaining = memoryview(value)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    return
                remaining = remaining[written:]
        except OSError:
            pass

    @staticmethod
    def _close_stream(
        selector: selectors.BaseSelector,
        state: _StreamState,
    ) -> None:
        try:
            selector.unregister(state.fd)
        except (KeyError, ValueError):
            pass
        try:
            state.stream.close()
        except (OSError, ValueError):
            pass


def _normalize_scope(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.]+", "_", value.strip("."))
    if not normalized:
        raise ValueError("logger scope must not be empty")
    return normalized


def _read_level(value: object, context: str) -> str:
    if not isinstance(value, str) or value not in _LEVELS:
        raise ValueError(f"{context} must be one of: {', '.join(_LEVELS)}")
    return value


def _positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


__all__ = [
    "LoggingConfig",
    "ScopedLoggers",
    "SubprocessLogRouter",
    "SubprocessLoggingConfig",
]
