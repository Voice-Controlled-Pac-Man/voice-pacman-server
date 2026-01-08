from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import random
from tqdm import tqdm
import torch


WANTED_LABELS = [
    "up",
    "down",
    "left",
    "right",
]

OTHER_LABEL = "other"
SILENCE_LABEL = "silence"

LABELS = {
    **{label: label for label in WANTED_LABELS},
    OTHER_LABEL: OTHER_LABEL,
    SILENCE_LABEL: SILENCE_LABEL,
}

class PacmanDataset(Dataset):
    """Dataset for Pacman game."""

    def __init__(self, root, url, download, subset, limit=None) -> None:
        self.root = root
        self.url = url
        self.download = download
        self.subset = subset
        self.limit = limit

        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=self.root,
            url=self.url,
            download=self.download,
            subset=self.subset,
        )

        self.wanted_samples = []
        self.other_samples = []
        self.silence_samples = []
        self.length = len(self.dataset)
        
        for file in tqdm(self.dataset, total=self.length, desc="Processing dataset"):
            waveform, sample_rate, label, _, _ = file
            if sample_rate != 16000:
                print("WARNING: Waveform has wrong sample rate")
                continue
            if self.limit and len(self.wanted_samples) >= self.limit:
                break
            if label in WANTED_LABELS:
                self.wanted_samples.append((waveform, label))
            else:
                self.other_samples.append((waveform, OTHER_LABEL))

        target_count_per_label = len(self.wanted_samples) // len(WANTED_LABELS)

        random.shuffle(self.other_samples)
        self.other_samples = self.other_samples[:target_count_per_label]

        for _ in range(target_count_per_label):
            self.silence_samples.append((torch.zeros(1, 16000, dtype=torch.float32), SILENCE_LABEL))

        self.samples = self.wanted_samples + self.other_samples + self.silence_samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        waveform, label = self.samples[index]
        # Pad or truncate to exactly 16000 samples
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        elif waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
        # Convert label to index
        label_to_idx = {label: idx for idx, label in enumerate(LABELS.keys())}
        label_idx = label_to_idx[label]
        return waveform.clone(), label_idx


class BackgroundNoiseDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = []

        noise_path = Path(self.root) / "SpeechCommands" / "speech_commands_v0.02" / "_background_noise_"

        files = list(noise_path.glob("*.wav"))
        for file in files:
            waveform, sample_rate = torchaudio.load(file)
            if sample_rate != 16000 or (waveform.shape[1] < 16000):
                print("WARNING: Waveform has wrong sample rate or length")
                continue
            self.samples.append(waveform)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # Clone the waveform to ensure it has its own storage
        return random.choice(self.samples).clone()