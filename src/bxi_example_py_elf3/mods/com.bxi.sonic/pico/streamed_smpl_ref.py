"""Policy-owned merger for the live SONIC SMPL reference stream.

The PICO bridge forwards complete rolling source chunks, but it does not own a
playback clock.  The control thread gathers ``current + [0..9]`` here and moves
the cursor only after the corresponding policy inference succeeds.  This is
the same gather/infer/advance ordering used by the official SONIC deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import sys

import numpy as np


WINDOW = 10
HISTORY_FRAMES = 5
MAX_GAP_FRAMES = 200
MAX_STREAM_EPOCH = (1 << 63) - 1


@dataclass
class IncomingChunk:
    frame_indices: np.ndarray
    term1_local: np.ndarray
    root_quat: np.ndarray
    wrist: np.ndarray
    head_joint_pos: np.ndarray


@dataclass
class MergeResult:
    did_catchup_reset: bool = False
    frame_offset_adjustment: int = 0
    frame_step: int = 1


def classify_frame_progress(
    newest_frame: int,
    previous_newest_frame: int | None,
) -> str:
    """Classify source progress without treating a repeated chunk as fresh."""
    if previous_newest_frame is None or newest_frame > previous_newest_frame:
        return "forward"
    if newest_frame == previous_newest_frame:
        return "duplicate"
    return "restart"


def new_stream_epoch(previous: int | None = None) -> int:
    """Return a positive process-unique epoch that changes after every reset."""
    while True:
        epoch = secrets.randbelow(MAX_STREAM_EPOCH) + 1
        if epoch != previous:
            return epoch


class StreamedSmplRefMerger:
    """Merge rolling source chunks while preserving an ordered policy cursor."""

    def __init__(
        self,
        history_frames: int = HISTORY_FRAMES,
        max_gap_frames: int = MAX_GAP_FRAMES,
        catch_up_enabled: bool = True,
    ) -> None:
        self.history_frames = int(history_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.catch_up_enabled = bool(catch_up_enabled)
        self.reset()

    def reset(self) -> None:
        self.stream_epoch = new_stream_epoch(
            getattr(self, "stream_epoch", None)
        )
        self.term1_local = np.zeros((0, 72), dtype=np.float32)
        self.root_quat = np.zeros((0, 4), dtype=np.float32)
        self.wrist = np.zeros((0, 6), dtype=np.float32)
        self.head_joint_pos = np.zeros((0, 2), dtype=np.float32)
        self.stream_window_start = 0
        self.current_frame = 0
        self.frame_step = 1
        self.total_merges = 0
        self.catchup_count = 0
        self.last_playback_held = True
        self.last_hold_reason = "waiting_for_window"
        self.last_consumed_local_frame = 0
        self.playback_hold_count = 0

    @property
    def timesteps(self) -> int:
        return int(self.term1_local.shape[0])

    @staticmethod
    def _calculate_frame_step(frame_indices: np.ndarray) -> int:
        if frame_indices.shape[0] < 2:
            return 1
        step = abs(int(frame_indices[1]) - int(frame_indices[0]))
        return step if step > 0 else 1

    def _calculate_sliding_window(
        self,
        incoming_frame_start: int,
        incoming_frame_end: int,
        frame_step: int,
    ) -> tuple[int, int, bool]:
        if self.timesteps <= 0:
            return incoming_frame_start, 0, True

        global_playback_frame = self.stream_window_start + frame_step * max(
            0,
            self.current_frame - self.history_frames,
        )
        max_gap_frames = (
            self.max_gap_frames + self.history_frames
            if self.catch_up_enabled
            else sys.maxsize
        )
        stream_window_end = (
            self.stream_window_start + frame_step * (self.timesteps - 1)
        )

        if incoming_frame_start <= self.stream_window_start:
            return incoming_frame_start, 0, True
        if incoming_frame_end <= stream_window_end:
            return incoming_frame_start, 0, True

        tentative_window_start = min(
            global_playback_frame,
            incoming_frame_start,
        )
        delta_to_incoming = incoming_frame_start - tentative_window_start
        tentative_merge_dst = delta_to_incoming // frame_step
        large_gap_from_old = incoming_frame_start > stream_window_end + frame_step

        if tentative_merge_dst > max_gap_frames or large_gap_from_old:
            return incoming_frame_start, 0, True
        return tentative_window_start, tentative_merge_dst, False

    def merge(self, chunk: IncomingChunk) -> MergeResult:
        had_existing_stream = self.timesteps > 0
        frame_step = self._calculate_frame_step(chunk.frame_indices)
        incoming_frame_start = int(chunk.frame_indices[0])
        incoming_frame_end = int(chunk.frame_indices[-1])

        new_window_start, merge_dst_frame, did_catchup = (
            self._calculate_sliding_window(
                incoming_frame_start,
                incoming_frame_end,
                frame_step,
            )
        )
        new_len = merge_dst_frame + int(chunk.frame_indices.shape[0])
        new_term1 = np.zeros((new_len, 72), dtype=np.float32)
        new_root = np.zeros((new_len, 4), dtype=np.float32)
        new_wrist = np.zeros((new_len, 6), dtype=np.float32)
        new_head = np.zeros((new_len, 2), dtype=np.float32)

        old_window_start = self.stream_window_start
        if merge_dst_frame > 0 and self.timesteps > 0:
            old_window_end = old_window_start + frame_step * self.timesteps
            overlap_start_global = max(new_window_start, old_window_start)
            overlap_end_global = min(incoming_frame_start, old_window_end)
            if overlap_start_global < overlap_end_global:
                copy_src_idx = (
                    overlap_start_global - old_window_start
                ) // frame_step
                copy_dst_idx = (
                    overlap_start_global - new_window_start
                ) // frame_step
                copy_count = (
                    overlap_end_global - overlap_start_global
                ) // frame_step
                if copy_count > 0:
                    src = slice(copy_src_idx, copy_src_idx + copy_count)
                    dst = slice(copy_dst_idx, copy_dst_idx + copy_count)
                    new_term1[dst] = self.term1_local[src]
                    new_root[dst] = self.root_quat[src]
                    new_wrist[dst] = self.wrist[src]
                    new_head[dst] = self.head_joint_pos[src]

        n_in = int(chunk.frame_indices.shape[0])
        incoming_dst = slice(merge_dst_frame, merge_dst_frame + n_in)
        new_term1[incoming_dst] = chunk.term1_local
        new_root[incoming_dst] = chunk.root_quat
        new_wrist[incoming_dst] = chunk.wrist
        new_head[incoming_dst] = chunk.head_joint_pos

        window_shift = (new_window_start - old_window_start) // frame_step
        self.term1_local = new_term1
        self.root_quat = new_root
        self.wrist = new_wrist
        self.head_joint_pos = new_head
        self.stream_window_start = new_window_start
        self.frame_step = frame_step
        self.total_merges += 1

        if did_catchup:
            self.current_frame = 0
            self.catchup_count += 1
            if had_existing_stream:
                self.stream_epoch = new_stream_epoch(self.stream_epoch)
            frame_offset_adjustment = 0
        else:
            self.current_frame = max(0, self.current_frame - window_shift)
            frame_offset_adjustment = window_shift

        return MergeResult(
            did_catchup_reset=did_catchup,
            frame_offset_adjustment=frame_offset_adjustment,
            frame_step=frame_step,
        )

    def build_smpl_ref(
        self,
        *,
        source_age_ms: float = 0.0,
        source_stale: bool = False,
    ) -> dict[str, np.ndarray] | None:
        """Gather a complete ten-frame window without moving the cursor."""
        if self.timesteps < WINDOW:
            return None

        current_local = self.current_frame
        idx = current_local + np.arange(WINDOW, dtype=np.int64)
        current_global_frame = (
            self.stream_window_start + current_local * self.frame_step
        )
        newest_global_frame = self.stream_window_start + (
            self.timesteps - 1
        ) * self.frame_step
        lead_frames = (
            newest_global_frame - current_global_frame
        ) // self.frame_step
        held = current_local + 1 + WINDOW >= self.timesteps
        return {
            "term1_local": np.ascontiguousarray(
                self.term1_local[idx], dtype=np.float32
            ),
            "root_quat": np.ascontiguousarray(
                self.root_quat[idx], dtype=np.float32
            ),
            "wrist": np.ascontiguousarray(self.wrist[idx], dtype=np.float32),
            "head_joint_pos": np.ascontiguousarray(
                self.head_joint_pos[idx], dtype=np.float32
            ),
            "frame_index": np.asarray([current_global_frame], dtype=np.int64),
            "newest_frame_index": np.asarray(
                [newest_global_frame], dtype=np.int64
            ),
            "lead_frames": np.asarray([lead_frames], dtype=np.int32),
            "valid_horizon": np.asarray([WINDOW], dtype=np.int32),
            "clamp_slots": np.asarray([0], dtype=np.int32),
            "stream_epoch": np.asarray([self.stream_epoch], dtype=np.int64),
            "source_age_ms": np.asarray(
                [max(0.0, source_age_ms)], dtype=np.float32
            ),
            "source_stale": np.asarray([int(source_stale)], dtype=np.uint8),
            "playback_hold": np.asarray([int(held)], dtype=np.uint8),
        }

    def advance_after_successful_tick(self) -> bool:
        """Advance at most once while retaining the protected complete tail."""
        if self.timesteps < WINDOW:
            return False

        self.last_consumed_local_frame = self.current_frame
        candidate = self.current_frame + 1
        advanced = candidate + WINDOW < self.timesteps
        if advanced:
            self.current_frame = candidate
            self.last_playback_held = False
            self.last_hold_reason = "advanced_after_successful_tick"
        else:
            self.last_playback_held = True
            self.last_hold_reason = "protected_tail"
            self.playback_hold_count += 1
        return advanced


__all__ = [
    "HISTORY_FRAMES",
    "IncomingChunk",
    "MAX_GAP_FRAMES",
    "MergeResult",
    "StreamedSmplRefMerger",
    "WINDOW",
    "classify_frame_progress",
    "new_stream_epoch",
]
