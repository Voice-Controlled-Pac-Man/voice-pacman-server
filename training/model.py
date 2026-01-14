import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchaudio

class BackgroundNoiseAugmentation(nn.Module):
    def __init__(self, background_noise_dataset):
        super().__init__()
        self.background_noise_dataset = background_noise_dataset

    def forward(self, waveform):
        if random.random() < 0.7:
            noise = self.background_noise_dataset[0].to(waveform.device)
            start = random.randint(0, noise.shape[1] - waveform.shape[2])
            noise = noise[:, start:start+waveform.shape[2]]

            snr = random.uniform(0.05, 0.2)
            waveform = waveform + (noise * snr)

        return waveform


class SmartRandomShiftAugumentation(nn.Module):
    def __init__(self, sample_rate=16000, window_size_ms=50, energy_threshold_percentile=10):
        super().__init__()
        self.sr = sample_rate
        self.window_size = int(window_size_ms * sample_rate / 1000)
        self.energy_threshold_percentile = energy_threshold_percentile

    def forward(self, waveform):
        if random.random() < 0.7:
            return waveform

        batch_size, channels, length = waveform.shape
        
        shifted_waveforms = []
        
        for i in range(batch_size):
            wav = waveform[i]
            
            if channels > 1:
                mono_wav = torch.mean(wav, dim=0)
            else:
                mono_wav = wav.squeeze(0)
            
            window_size = min(self.window_size, length // 4)
            if window_size < 10:
                shifted_waveforms.append(wav)
                continue
            
            num_windows = length - window_size + 1
            energies = torch.zeros(num_windows, device=wav.device)
            
            for j in range(num_windows):
                window = mono_wav[j:j + window_size]
                energies[j] = torch.sqrt(torch.mean(window ** 2))
            
            energy_threshold = torch.quantile(energies, self.energy_threshold_percentile / 100.0)
            
            peak_energy = torch.max(energies)
            relative_threshold = peak_energy * 0.1
            final_threshold = torch.max(energy_threshold, relative_threshold).item()
            
            voice_end_idx = length - 1
            for idx in range(num_windows - 1, -1, -1):
                if energies[idx] > final_threshold:
                    voice_end_idx = idx + window_size - 1
                    break
            
            available_shift = length - voice_end_idx - 1
            
            if available_shift > 0:
                shift_amount = random.randint(0, available_shift)
                
                shifted = torch.zeros_like(wav)
                shifted[:, shift_amount:length] = wav[:, :length - shift_amount]
            else:
                shifted = wav
            
            shifted_waveforms.append(shifted)
        
        return torch.stack(shifted_waveforms, dim=0)


class PacManCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=40, n_fft=400, hop_length=160
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 8))
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(4, 4))
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(32 * 7 * 22, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.melspec(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout3(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x