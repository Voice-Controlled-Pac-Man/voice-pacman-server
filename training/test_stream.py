import sounddevice as sd
import numpy as np
import torch
import torch.nn.functional as F
from model import PacManCNN
from datasets import LABELS
import threading
import time
from collections import deque

SAMPLE_RATE = 16000
BUFFER_DURATION = 2.0
INFERENCE_INTERVAL = 0.05
CONFIDENCE_THRESHOLD = 0.85
SAMPLES_PER_SECOND = SAMPLE_RATE
SAMPLES_IN_BUFFER = int(BUFFER_DURATION * SAMPLES_PER_SECOND)
SAMPLES_FOR_INFERENCE = int(1.0 * SAMPLES_PER_SECOND)

ACTION_LABELS = ["up", "down", "left", "right"]
LABEL_LIST = list(LABELS.keys())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading model...")
model = PacManCNN(num_classes=6)
model.load_state_dict(torch.load("training/models/best_pacman_model_without_augmentation.pth", map_location=device))
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
        if status:
            print(f"⚠️  Audio status: {status}")
        
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        with self.lock:
            self.buffer.extend(audio_data)
    
    def get_last_second(self):
        with self.lock:
            if len(self.buffer) < SAMPLES_FOR_INFERENCE:
                return None
            
            samples = list(self.buffer)[-SAMPLES_FOR_INFERENCE:]
            return np.array(samples, dtype=np.float32)
    
    def clear_buffer(self):
        with self.lock:
            self.buffer.clear()
    
    def start_stream(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.05),
            dtype=np.float32
        )
        self.stream.start()
        print("🎤 Audio stream started")
    
    def stop_stream(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("🛑 Audio stream stopped")
    
    def process_audio(self):
        audio_samples = self.get_last_second()
        
        if audio_samples is None:
            return None
        
        waveform = torch.from_numpy(audio_samples).unsqueeze(0).float()
        
        if waveform.shape[1] < SAMPLES_FOR_INFERENCE:
            waveform = F.pad(waveform, (0, SAMPLES_FOR_INFERENCE - waveform.shape[1]))
        elif waveform.shape[1] > SAMPLES_FOR_INFERENCE:
            waveform = waveform[:, :SAMPLES_FOR_INFERENCE]
        
        waveform_batch = waveform.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(waveform_batch)
            probabilities = F.softmax(output, dim=1)
        
        return probabilities[0]


def find_action_with_threshold(probabilities, threshold=CONFIDENCE_THRESHOLD):
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
    other_idx = LABEL_LIST.index("other")
    other_confidence = probabilities[other_idx].item()
    return other_confidence >= threshold


def main():
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
        processor.start_stream()
        
        time.sleep(0.5)
        
        print("\n🎤 Listening... (say 'up', 'down', 'left', or 'right')\n")
        
        last_action_time = 0
        action_cooldown = 0.5
        
        while processor.running:
            start_time = time.time()
            
            probabilities = processor.process_audio()
            
            if probabilities is not None:
                probs_str = " | ".join([f"{label}: {probabilities[i].item()*100:5.1f}%" 
                                        for i, label in enumerate(LABEL_LIST)])
                print(f"{probs_str}", end='\r')
                
                action, confidence = find_action_with_threshold(probabilities, CONFIDENCE_THRESHOLD)
                
                current_time = time.time()
                
                if action is not None:
                    if current_time - last_action_time >= action_cooldown:
                        print(f"\n🎯 ACTION DETECTED: {action.upper()} ({confidence*100:.1f}%)")
                        processor.clear_buffer()
                        last_action_time = current_time
                elif check_other_detected(probabilities, CONFIDENCE_THRESHOLD):
                    processor.clear_buffer()
            
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
