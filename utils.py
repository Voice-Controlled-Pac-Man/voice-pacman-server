from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
from typing import Any

from constants import MODEL_PATH

_backend = None
_backend_lock = threading.Lock()


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass(frozen=True)
class OnnxBackend:
    session: Any
    input_name: str

    def predict_proba(self, audio_batch: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: audio_batch})[0]
        return _softmax(logits, axis=1).astype(np.float32, copy=False)


def get_inference_backend():
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                onnx_path = Path(MODEL_PATH)
                if not onnx_path.exists():
                    raise FileNotFoundError(
                        f"ONNX model not found at {onnx_path}. "
                        f"Expected it at the project root as '{MODEL_PATH}'."
                    )

                import onnxruntime
                session = onnxruntime.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                input_name = session.get_inputs()[0].name
                _backend = OnnxBackend(session=session, input_name=input_name)
    return _backend



