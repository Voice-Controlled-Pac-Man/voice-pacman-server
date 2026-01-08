import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Default settings
DEFAULT_DURATION = 1
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_OUTPUT_DIR = "custom_recordings"

# Available labels (from datasets.py)
AVAILABLE_LABELS = ["up", "down", "left", "right", "other", "silence"]


def record_audio(duration=DEFAULT_DURATION, sample_rate=DEFAULT_SAMPLE_RATE):
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording
    
    Returns:
        numpy array with audio data
    """
    print(f"\n🎤 Recording {duration} second(s)...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("✅ Recording finished")
    return recording


def save_recording(recording, label, output_dir=DEFAULT_OUTPUT_DIR, sample_rate=DEFAULT_SAMPLE_RATE):
    """
    Save recording to file with label.
    
    Args:
        recording: Audio data as numpy array
        label: Label for the recording
        output_dir: Directory to save recordings
        sample_rate: Sample rate of the recording
    
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for label if it doesn't exist
    label_dir = output_path / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
    filename = f"{label}_{timestamp}.wav"
    filepath = label_dir / filename
    
    # Save audio file
    sf.write(str(filepath), recording, sample_rate)
    
    return filepath


def get_label_from_console():
    """
    Prompt user for label from console.
    
    Returns:
        Label string
    """
    print("\n" + "="*50)
    print("Available labels:")
    for i, label in enumerate(AVAILABLE_LABELS, 1):
        print(f"  {i}. {label}")
    print("="*50)
    
    while True:
        user_input = input("\nEnter label (number or name, 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            return None
        
        # Check if input is a number
        try:
            label_idx = int(user_input) - 1
            if 0 <= label_idx < len(AVAILABLE_LABELS):
                return AVAILABLE_LABELS[label_idx]
            else:
                print(f"❌ Invalid number. Please enter 1-{len(AVAILABLE_LABELS)}")
        except ValueError:
            # Check if input is a valid label name
            if user_input in AVAILABLE_LABELS:
                return user_input
            else:
                print(f"❌ Invalid label. Please enter one of: {', '.join(AVAILABLE_LABELS)}")


def record_and_save(duration=DEFAULT_DURATION, sample_rate=DEFAULT_SAMPLE_RATE, 
                    output_dir=DEFAULT_OUTPUT_DIR):
    """
    Record audio, get label from console, and save to file.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording
        output_dir: Directory to save recordings
    
    Returns:
        Path to saved file or None if user quit
    """
    # Record audio
    recording = record_audio(duration, sample_rate)
    
    # Get label from console
    label = get_label_from_console()
    if label is None:
        return None
    
    # Save recording
    filepath = save_recording(recording, label, output_dir, sample_rate)
    print(f"\n💾 Saved to: {filepath}")
    
    return filepath


def main():
    """
    Main loop for recording multiple samples.
    """
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