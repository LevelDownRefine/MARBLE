"""Data integrity and CPU loading for public author artifacts."""

import hashlib
import io
import json
import pickle
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_manifest():
    return json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))


def verify_file(path, metadata):
    """Reject missing, truncated, or changed files before deserialization."""
    path = Path(path)
    assert "bytes" in metadata and "sha256" in metadata
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"File size mismatch: {path}")
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class CPUUnpickler(pickle.Unpickler):
    """Read the hash-verified author data with tensors mapped to CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda content: torch.load(
                io.BytesIO(content), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_author_data(path):
    with Path(path).open("rb") as handle:
        return CPUUnpickler(handle).load()
