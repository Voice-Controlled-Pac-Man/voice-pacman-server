import sounddevice as sd
import numpy as np
import torch
import torch.nn.functional as F
from model import PacManCNN
from datasets import LABELS
import threading
import time
from collections import deque

# Configuration
SAMPLE_RATE = 16000
BUFFER_DURATION = 2.0  # Keep 2 seconds in buffer (more than 1 second needed)
INFERENCE_INTERVAL = 0.05  # Run inference every 100ms
CONFIDENCE_THRESHOLD = 0.95  # Confidence threshold for action detection
SAMPLES_PER_SECOND = SAMPLE_RATE
SAMPLES_IN_BUFFER = int(BUFFER_DURATION * SAMPLES_PER_SECOND)
SAMPLES_FOR_INFERENCE = int(1.0 * SAMPLES_PER_SECOND)  # 1 second

# Action labels (we care about these)
ACTION_LABELS = ["up", "down", "left", "right"]
LABEL_LIST = list(LABELS.keys())

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
print("Loading model...")
model = PacManCNN(num_classes=6)
model.load_state_dict(torch.load("training/models/best_pacman_model_with_background_noise.pth", map_location=device))
model.eval()
model.to(device)
print("✅ Model loaded!")


class AudioStreamProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=SAMPLES_IN_BUFFER)
        self.lock = threading.Lock()
        self.running = False
        self.stream = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input stream."""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        # Add to buffer (thread-safe)
        with self.lock:
            self.buffer.extend(audio_data)
    
    def get_last_second(self):
        """Extract the last 1 second from buffer."""
        with self.lock:
            if len(self.buffer) < SAMPLES_FOR_INFERENCE:
                return None
            
            # Get last SAMPLES_FOR_INFERENCE samples
            samples = list(self.buffer)[-SAMPLES_FOR_INFERENCE:]
            return np.array(samples, dtype=np.float32)
    
    def clear_buffer(self):
        """Clear the audio buffer."""
        with self.lock:
            self.buffer.clear()
    
    def start_stream(self):
        """Start the audio input stream."""
        self.running = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.05),  # 50ms blocks
            dtype=np.float32
        )
        self.stream.start()
        print("🎤 Audio stream started")
    
    def stop_stream(self):
        """Stop the audio input stream."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("🛑 Audio stream stopped")
    
    def process_audio(self):
        """Process audio buffer and run inference."""
        # Get last second of audio
        audio_samples = self.get_last_second()
        
        if audio_samples is None:
            return None
        
        # Convert to tensor: [channels, samples]
        waveform = torch.from_numpy(audio_samples).unsqueeze(0).float()
        
        # Ensure exactly 16000 samples (should be, but just in case)
        if waveform.shape[1] < SAMPLES_FOR_INFERENCE:
            waveform = F.pad(waveform, (0, SAMPLES_FOR_INFERENCE - waveform.shape[1]))
        elif waveform.shape[1] > SAMPLES_FOR_INFERENCE:
            waveform = waveform[:, :SAMPLES_FOR_INFERENCE]
        
        # Add batch dimension: [1, channels, samples]
        waveform_batch = waveform.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(waveform_batch)
            probabilities = F.softmax(output, dim=1)
        
        return probabilities[0]  # Return probabilities for single sample


def find_action_with_threshold(probabilities, threshold=CONFIDENCE_THRESHOLD):
    """
    Check if any action label exceeds the confidence threshold.
    Returns the action with the highest confidence among those exceeding threshold.
    
    Args:
        probabilities: Tensor with probabilities for each class
        threshold: Confidence threshold
    
    Returns:
        Tuple (action_label, confidence) if threshold exceeded, else (None, None)
    """
    best_action = None
    best_confidence = 0.0
    
    for action_label in ACTION_LABELS:
        label_idx = LABEL_LIST.index(action_label)
        confidence = probabilities[label_idx].item()
        
        if confidence >= threshold and confidence > best_confidence:
            best_action = action_label
            best_confidence = confidence
    
    if best_action is not None:
        return best_action, best_confidence
    
    return None, None


def check_other_detected(probabilities, threshold=CONFIDENCE_THRESHOLD):
    """
    Check if "other" class exceeds the confidence threshold.
    
    Args:
        probabilities: Tensor with probabilities for each class
        threshold: Confidence threshold
    
    Returns:
        True if "other" exceeds threshold, False otherwise
    """
    other_idx = LABEL_LIST.index("other")
    other_confidence = probabilities[other_idx].item()
    return other_confidence >= threshold


def main():
    """Main function for streaming audio processing."""
    processor = AudioStreamProcessor()
    
    print("="*60)
    print("🎮 VOICE PACMAN - REAL-TIME STREAMING")
    print("="*60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Inference interval: {INFERENCE_INTERVAL*1000:.0f} ms")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Action labels: {', '.join(ACTION_LABELS)}")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    
    try:
        # Start audio stream
        processor.start_stream()
        
        # Give buffer time to fill up
        time.sleep(0.5)
        
        print("\n🎤 Listening... (say 'up', 'down', 'left', or 'right')\n")
        
        last_action_time = 0
        action_cooldown = 0.5  # Prevent duplicate detections within 0.5 seconds
        
        # Main processing loop
        while processor.running:
            start_time = time.time()
            
            # Process audio
            probabilities = processor.process_audio()
            
            if probabilities is not None:
                # Check for action
                action, confidence = find_action_with_threshold(probabilities, CONFIDENCE_THRESHOLD)
                
                current_time = time.time()
                
                if action is not None:
                    # Check cooldown to avoid duplicate detections
                    if current_time - last_action_time >= action_cooldown:
                        print(f"🎯 ACTION DETECTED: {action.upper()} ({confidence*100:.1f}%)")
                        processor.clear_buffer()
                        last_action_time = current_time
                elif check_other_detected(probabilities, CONFIDENCE_THRESHOLD):
                    # Clear buffer when "other" is detected
                    processor.clear_buffer()
                
                # Optional: Show top prediction (for debugging)
                # predicted_idx = torch.argmax(probabilities).item()
                # predicted_label = LABEL_LIST[predicted_idx]
                # predicted_prob = probabilities[predicted_idx].item()
                # if predicted_prob > 0.3:  # Only show if confident enough
                #     print(f"   [{predicted_label}: {predicted_prob*100:.1f}%]", end='\r')
            
            # Sleep to maintain inference interval
            elapsed = time.time() - start_time
            sleep_time = max(0, INFERENCE_INTERVAL - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.stop_stream()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
