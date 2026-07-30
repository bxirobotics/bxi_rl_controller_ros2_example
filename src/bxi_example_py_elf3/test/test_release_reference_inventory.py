import hashlib
from pathlib import Path

import numpy as np


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "sonic_reference"
MODEL = (
    Path(__file__).resolve().parents[1]
    / "data/sonic_model/elf3_step28800_smpl/model_step_028800_smpl.onnx"
)
MODEL_SHA256 = "26dc3e96adfb894850b409e43f06178c79b74167c719f626eadbb9df3fcacd06"


def test_release_contains_only_the_self_collected_standing_reference():
    relative_files = {
        path.relative_to(REFERENCE_ROOT).as_posix()
        for path in REFERENCE_ROOT.rglob("*.npz")
    }
    assert relative_files == {
        "elf3_pico_stand_clean_001/stream_reference.npz"
    }


def test_release_reference_does_not_embed_workstation_paths():
    reference = (
        REFERENCE_ROOT
        / "elf3_pico_stand_clean_001"
        / "stream_reference.npz"
    )
    with np.load(reference, allow_pickle=False) as archive:
        embedded_strings = [
            str(archive[key].tolist())
            for key in archive.files
            if archive[key].dtype.kind in "US"
        ]

    assert set(embedded_strings) == {
        "elf3_pico_stand_clean_001",
        "self_collected://elf3_pico_stand_clean_001",
    }


def test_release_model_has_no_training_environment_traces():
    payload = MODEL.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == MODEL_SHA256

    lower_payload = payload.lower()
    forbidden_markers = (
        b"/home/",
        b"/data/",
        b"wandb",
        b"tensorboard",
        b"dataset",
        b"checkpoint",
        b"optimizer_state",
    )
    assert all(marker not in lower_payload for marker in forbidden_markers)
