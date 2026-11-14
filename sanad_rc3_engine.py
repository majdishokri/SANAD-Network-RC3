"""Thin API wrapper around the exact immutable SANAD RC3 frozen inference."""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from rc3 import FrozenRC3

class SANADRC3Engine:
    """Identical frozen RC3; accepts precomputed, validated Sionna-format traces."""
    def __init__(self):
        self._engine = FrozenRC3()

    def predict(self, trace: Mapping, stream: str = "injected") -> dict:
        if stream not in ("healthy", "injected"):
            raise ValueError("Stream must be healthy/injected")
        return self._engine.predict_trace(trace, stream)

    def predict_npz(self, path: str | Path, stream: str = "injected") -> dict:
        with np.load(path, allow_pickle=False) as z:
            trace = {key: z[key] for key in z.files}
        return self.predict(trace, stream)
