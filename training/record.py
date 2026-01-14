import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime

DEFAULT_DURATION = 1
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_DIR = "custom_recordings"

AVAILABLE_LABELS = ["up", "down", "left", "right", "other", "silence"]


def record_audio(duration=DEFAULT_DURATION, sample_rate=DEFAULT_SAMPLE_RATE):
    print(f"\n🎤 Recording {duration} second(s)...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("✅ Recording finished")
    return recording


def save_recording(recording, label, output_dir=DEFAULT_OUTPUT_DIR, sample_rate=DEFAULT_SAMPLE_RATE):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label_dir = output_path / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{label}_{timestamp}.wav"
    filepath = label_dir / filename
    
    sf.write(str(filepath), recording, sample_rate)
    
    return filepath


def get_label_from_console():
    print("\n" + "="*50)
    print("Available labels:")
    for i, label in enumerate(AVAILABLE_LABELS, 1):
        print(f"  {i}. {label}")
    print("="*50)
    
    while True:
        user_input = input("\nEnter label (number or name, 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            return None
        
        try:
            label_idx = int(user_input) - 1
            if 0 <= label_idx < len(AVAILABLE_LABELS):
                return AVAILABLE_LABELS[label_idx]
            else:
                print(f"❌ Invalid number. Please enter 1-{len(AVAILABLE_LABELS)}")
        except ValueError:
            if user_input in AVAILABLE_LABELS:
                return user_input
            else:
                print(f"❌ Invalid label. Please enter one of: {', '.join(AVAILABLE_LABELS)}")


def record_and_save(duration=DEFAULT_DURATION, sample_rate=DEFAULT_SAMPLE_RATE, 
                    output_dir=DEFAULT_OUTPUT_DIR):
    recording = record_audio(duration, sample_rate)
    
    label = get_label_from_console()
    if label is None:
        return None
    
    filepath = save_recording(recording, label, output_dir, sample_rate)
    print(f"\n💾 Saved to: {filepath}")
    
    return filepath


def main():
    print("="*60)
    print("🎙️  Audio Recording Tool")
    print("="*60)
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    print(f"Duration: {DEFAULT_DURATION} second(s)")
    print(f"Sample rate: {DEFAULT_SAMPLE_RATE} Hz")
    print("\nPress Enter to start recording, or 'q' to quit")
    
    count = 0
    
    while True:
        user_input = input("\n[Press Enter to record, 'q' to quit]: ").strip().lower()
        
        if user_input == 'q':
            print(f"\n👋 Recorded {count} sample(s). Goodbye!")
            break
        
        try:
            filepath = record_and_save()
            if filepath is not None:
                count += 1
                print(f"📊 Total recordings: {count}")
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()