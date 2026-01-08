from pathlib import Path
import sounddevice as sd
import torch
import matplotlib.pyplot as plt

from constants import LABELS
from training.model import PacManCNN

def play_tensor(waveform, sample_rate=16000):
    audio_data = waveform.detach().cpu().squeeze().numpy()
    
    print("🔊 Odtwarzanie...")
    sd.play(audio_data, sample_rate)
    sd.wait()
    print("✅ Koniec.")


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    figure.savefig(f"plots/{title}.png")
    return figure


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    figure.savefig(f"plots/{title}.png")
    return figure

def pad_or_truncate_waveform(waveform):
    if waveform.shape[1] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    elif waveform.shape[1] > 16000:
        waveform = waveform[:, :16000]
    return waveform


def convert_to_onnx(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PacManCNN(num_classes=len(LABELS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 16000)
    
    onnx_path = "model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['logits'],
        dynamic_axes={
            'audio': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    return onnx_path