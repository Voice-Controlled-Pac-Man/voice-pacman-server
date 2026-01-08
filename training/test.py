
import torch.nn.functional as F
from model import PacManCNN
import torchaudio
import torch
from datasets import LABELS
from pathlib import Path
import random
from utils import pad_or_truncate_waveform, play_tensor, plot_waveform

CUSTOM_RECORDINGS_DIR = "custom_recordings"
MODEL_PATH = "models/best_pacman_model_with_background_noise.pth"
SAMPLE_RATE = 16000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = PacManCNN(num_classes=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)


def get_all_custom_recordings(recordings_dir=CUSTOM_RECORDINGS_DIR):
    """
    Scan the custom_recordings directory and return all audio files with their labels.
    
    Args:
        recordings_dir: Directory containing custom recordings
    
    Returns:
        List of tuples (filepath, label)
    """
    recordings = []
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        return recordings
    
    for label_dir in recordings_path.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            wav_files = list(label_dir.glob("*.wav"))
            for wav_file in wav_files:
                recordings.append((wav_file, label))
    
    return recordings


def load_random_recording(recordings_dir=CUSTOM_RECORDINGS_DIR):
    """
    Randomly pick and load a recording from custom_recordings directory.
    
    Args:
        recordings_dir: Directory containing custom recordings
    
    Returns:
        Tuple (waveform, label, filepath) or None if no recordings found
    """
    recordings = get_all_custom_recordings(recordings_dir)
    
    if not recordings:
        return None
    
    filepath, label = random.choice(recordings)
    
    waveform, sample_rate = torchaudio.load(str(filepath))
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    waveform = pad_or_truncate_waveform(waveform)
    
    return waveform, label, filepath


def display_probabilities(probabilities, true_label=None):
    """
    Display probabilities in a formatted way.
    
    Args:
        probabilities: Tensor with probabilities for each class
        true_label: Optional true label for comparison
    """
    print("\n" + "="*60)
    print("PREDICTION PROBABILITIES:")
    print("="*60)
    
    # Convert LABELS dict to list (maintaining order)
    label_list = list(LABELS.keys())
    
    # Sort by probability (descending)
    probs_with_labels = [(label_list[i], prob.item()) for i, prob in enumerate(probabilities[0])]
    probs_with_labels.sort(key=lambda x: x[1], reverse=True)
    
    for label, prob in probs_with_labels:
        marker = "✓" if true_label and label == true_label else " "
        print(f"{marker} {label:10s}: {prob*100:6.2f}%")
    
    print("="*60)
    
    # Show predicted label
    predicted_idx = torch.argmax(probabilities[0]).item()
    predicted_label = label_list[predicted_idx]
    predicted_prob = probabilities[0][predicted_idx].item()
    
    print(f"\n🎯 Predicted: {predicted_label} ({predicted_prob*100:.2f}%)")
    if true_label:
        print(f"📝 True label: {true_label}")
        if predicted_label == true_label:
            print("✅ Correct prediction!")
        else:
            print("❌ Incorrect prediction")


def test_random_recording(recordings_dir=CUSTOM_RECORDINGS_DIR):
    """
    Randomly pick a custom recording, play it, plot it, and display probabilities.
    
    Args:
        recordings_dir: Directory containing custom recordings
    """
    # Load random recording
    result = load_random_recording(recordings_dir)
    
    if result is None:
        print(f"❌ No recordings found in '{recordings_dir}' directory.")
        print("   Please record some samples first using record.py")
        return
    
    waveform, true_label, filepath = result
    
    print("="*60)
    print("🎵 TESTING CUSTOM RECORDING")
    print("="*60)
    print(f"📁 File: {filepath}")
    print(f"🏷️  Label: {true_label}")
    print("="*60)
    
    # Plot waveform
    print("\n📊 Plotting waveform...")
    plot_title = f"Custom Recording - {true_label}"
    # Ensure plots directory exists
    Path("plots").mkdir(exist_ok=True)
    plot_waveform(waveform, SAMPLE_RATE, title=plot_title)
    print(f"✅ Plot saved to plots/{plot_title}.png")
    
    # Play audio
    print("\n🔊 Playing audio...")
    play_tensor(waveform, SAMPLE_RATE)
    
    # Prepare waveform for model (add batch dimension: [1, channels, samples])
    waveform_batch = waveform.unsqueeze(0).to(device)
    
    # Get predictions
    print("\n🤖 Running model inference...")
    with torch.no_grad():
        output = model(waveform_batch)
        probabilities = F.softmax(output, dim=1)
    
    # Display probabilities
    display_probabilities(probabilities, true_label)


if __name__ == "__main__":
    test_random_recording()





