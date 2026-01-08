import time
import threading
from collections import deque
from pathlib import Path

import numpy as np

from constants import CONFIDENCE_THRESHOLD, DEBUG, LABELS, SAMPLE_RATE
from utils import get_inference_backend


class AudioStreamProcessor:
    """Process audio stream from WebSocket and run inference."""
    
    def __init__(self):
        self.buffer = deque(maxlen=SAMPLE_RATE)
        self.lock = threading.Lock()
        self.backend = get_inference_backend()
    
    def add_audio_chunk(self, audio_data: np.ndarray):        
        if audio_data.ndim > 1:
            raise ValueError("Audio data must be 1D")
        
        with self.lock:
            self.buffer.extend(audio_data)
    
    def plot_buffer(self):
        """Plot the current buffer contents."""
        if not DEBUG:
            return

        with self.lock:
            if len(self.buffer) == 0:
                return
            
            buffer_array = np.array(list(self.buffer), dtype=np.float32)

        # Lazy imports so production deployments don't require matplotlib.
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            print(f"⚠️  Plotting disabled (missing optional deps): {e}")
            return

        Path("plots").mkdir(exist_ok=True)

        t = np.arange(buffer_array.shape[0], dtype=np.float32) / float(SAMPLE_RATE)
        plt.figure(figsize=(12, 3))
        plt.plot(t, buffer_array, linewidth=1)
        plt.grid(True)
        plt.title("Audio Buffer")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig("plots/Audio Buffer.png")
        plt.close()
    
    def get_audio(self):
        with self.lock:
            if len(self.buffer) != SAMPLE_RATE:
                empty_samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
                empty_samples[:len(self.buffer)] = np.array(list(self.buffer), dtype=np.float32)
                return empty_samples
            return np.array(list(self.buffer), dtype=np.float32)
    
    def clear_buffer(self):
        with self.lock:
            self.buffer.clear()
    
    def process_audio(self):
        audio_samples = self.get_audio()
        
        if len(audio_samples) != SAMPLE_RATE:
            return None

        # Shape: (batch=1, channels=1, samples=16000)
        waveform_batch = audio_samples.astype(np.float32)[None, None, :]

        start_time = time.time()
        probabilities = self.backend.predict_proba(waveform_batch)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        if DEBUG:
            print(f"⏱️  Inference time: {inference_time:.2f}ms")
        
        return probabilities[0]
    
    def find_action_with_threshold(self, probabilities):
        best_action_idx = probabilities.argmax()
        best_action = LABELS[best_action_idx]
        best_confidence = probabilities[best_action_idx].item()
        return best_action, best_confidence if best_confidence >= CONFIDENCE_THRESHOLD else None

    def check_other_detected(self, probabilities):
        other_idx = LABELS.index("other")
        other_confidence = float(probabilities[other_idx])
        return other_confidence >= CONFIDENCE_THRESHOLD
