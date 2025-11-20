"""
Speech enhancement algorithm.

- Loads clean speech from LibriSpeech (torchaudio.datasets.LIBRISPEECH).
- Creates noisy mixtures on-the-fly by adding random Gaussian noise at random SNRs.
- Processes audio in STFT domain and trains a small U-Net-like CNN to estimate a mask
  on the magnitude spectrogram. Reconstructs waveform using noisy phase.

Requirements:
  pip install torch torchaudio librosa numpy soundfile

References:
- torchaudio.datasets.LIBRISPEECH (downloads LibriSpeech). See torchaudio docs.
"""

import math
import random
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import librosa

# --------------------------- Config ---------------------------
class Config:
    root = './data'               # where LibriSpeech will be stored
    subset = 'train-clean-100'    # can be changed to other splits
    sample_rate = 16000
    n_fft = 512
    hop_length = 128
    win_length = 512
    batch_size = 8
    epochs = 20
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    min_snr = -5.0   # dB
    max_snr = 15.0   # dB
    checkpoint_dir = './checkpoints'

cfg = Config()
Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

# --------------------------- Dataset ---------------------------
class LibriSpeechNoisy(torch.utils.data.Dataset):
    """Wraps torchaudio LibriSpeech and synthesizes noisy mixtures on the fly."""
    def __init__(self, root, url='train-clean-100', sample_rate=16000, min_snr=-5.0, max_snr=15.0):
        super().__init__()
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=True)
        self.sample_rate = sample_rate
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __len__(self):
        return len(self.dataset)

    def _fix_len(self, wav):
        # convert to mono and resample if needed
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        return librosa.resample(wav, orig_sr=self.orig_sr, target_sr=self.sample_rate) if self.orig_sr != self.sample_rate else wav

    def __getitem__(self, idx):
        # torchaudio LIBRISPEECH returns (waveform, sample_rate, speaker_id, chapter_id, utterance_id)
        waveform, sr, speaker_id, chapter_id, utterance_id, transcript = self.dataset[idx]

        waveform = waveform.squeeze().numpy()
        # ensure sample rate
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        # normalize
        waveform = waveform / (np.abs(waveform).max() + 1e-8)
        # synthesize Gaussian noise and mix at random SNR
        snr = random.uniform(self.min_snr, self.max_snr)
        rms_clean = np.sqrt(np.mean(waveform**2) + 1e-12)
        desired_rms_noise = rms_clean / (10**(snr/20))
        noise = np.random.randn(len(waveform)).astype(np.float32)
        rms_noise = np.sqrt(np.mean(noise**2) + 1e-12)
        noise = noise * (desired_rms_noise / (rms_noise + 1e-12))
        mixture = waveform + noise
        # return as float32 tensors
        return torch.from_numpy(mixture).float(), torch.from_numpy(waveform).float()

# --------------------------- STFT / ISTFT helpers ---------------------------

def spectrogram_torch(waveform, n_fft=512, hop_length=128, win_length=512):
    # waveform: [B, T]
    return torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)

def istft_torch(stft_complex, hop_length=128, win_length=512):
    return torch.istft(stft_complex, n_fft=stft_complex.size(-2)*2 - 2, hop_length=hop_length, win_length=win_length)

# --------------------------- Model ---------------------------
class SmallUNet(nn.Module):
    """A lightweight U-Net on log-magnitude spectrograms with fixed skip connection sizes."""
    def __init__(self, n_fft_bins):
        super().__init__()
        C = 32
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(C, C*2, 3, padding=1, stride=2), nn.BatchNorm2d(C*2), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(C*2, C*4, 3, padding=1, stride=2), nn.BatchNorm2d(C*4), nn.ReLU())

        # Middle
        self.mid = nn.Sequential(nn.Conv2d(C*4, C*4, 3, padding=1), nn.BatchNorm2d(C*4), nn.ReLU())

        # Decoder with output_padding=1 to fix size mismatches
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1, output_padding=1),
                                  nn.BatchNorm2d(C*2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(C*2, C, 4, stride=2, padding=1, output_padding=1),
                                  nn.BatchNorm2d(C), nn.ReLU())
        self.dec1 = nn.Conv2d(C, 1, 1)

    def forward(self, x):
        # x: [B, 1, F, T]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.mid(e3)

        d3 = self.dec3(m)
        # crop if necessary to match encoder size (just in case)
        if d3.size(2) != e2.size(2) or d3.size(3) != e2.size(3):
            d3 = d3[:, :, :e2.size(2), :e2.size(3)]
        d3 = d3 + e2

        d2 = self.dec2(d3)
        if d2.size(2) != e1.size(2) or d2.size(3) != e1.size(3):
            d2 = d2[:, :, :e1.size(2), :e1.size(3)]
        out = torch.sigmoid(self.dec1(d2))
        return out
# --------------------------- Training utilities ---------------------------

def collate_fn(batch):
    # batch: list of (mixture, clean)
    mixtures, cleans = zip(*batch)
    # pad to max length in batch
    max_len = max([m.shape[0] for m in mixtures])
    mix_padded = torch.stack([F.pad(m, (0, max_len - m.shape[0])) for m in mixtures])
    clean_padded = torch.stack([F.pad(c, (0, max_len - c.shape[0])) for c in cleans])
    return mix_padded, clean_padded


def prepare_batch(mixture_wav, clean_wav, cfg):
    window = torch.hann_window(cfg.n_fft).to(cfg.device)
    stft_mix = torch.stft(mixture_wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                          win_length=cfg.win_length, window=window, return_complex=True)
    stft_clean = torch.stft(clean_wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                            win_length=cfg.win_length, window=window, return_complex=True)
    mag_mix = torch.abs(stft_mix)
    mag_clean = torch.abs(stft_clean)
    log_mag = torch.log1p(mag_mix).unsqueeze(1)
    return stft_mix, stft_clean, log_mag, mag_clean

# --------------------------- Main training function ---------------------------

def train():
    # dataset & loader
    ds = LibriSpeechNoisy(root=cfg.root, url=cfg.subset, sample_rate=cfg.sample_rate, min_snr=cfg.min_snr, max_snr=cfg.max_snr)
    loader = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)

    # model
    n_fft_bins = cfg.n_fft // 2 + 1
    model = SmallUNet(n_fft_bins).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for i, (mix_wav, clean_wav) in enumerate(loader):
            mix_wav = mix_wav.to(cfg.device)
            clean_wav = clean_wav.to(cfg.device)
            stft_mix, stft_clean, log_mag, mag_clean = prepare_batch(mix_wav, clean_wav, cfg)
            log_mag = log_mag.to(cfg.device)
            mag_clean = mag_clean.to(cfg.device)
            # forward
            est_mask = model(log_mag)
            est_mag = est_mask.squeeze(1) * torch.abs(stft_mix)
            # loss on magnitude (MSE)
            loss = criterion(est_mag, mag_clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs}  Step {i+1}/{len(loader)}  Loss {running_loss/50:.4f}")
                running_loss = 0.0

        # save checkpoint
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch+1}, os.path.join(cfg.checkpoint_dir, f'model_epoch_{epoch+1}.pt'))

    print('Training finished')

# --------------------------- Inference helper ---------------------------

def enhance_waveform(waveform, model, cfg):
    # Convert to tensor on device
    waveform = torch.tensor(waveform, dtype=torch.float32, device=cfg.device).unsqueeze(0)

    # Window for STFT/ISTFT
    window = torch.hann_window(cfg.win_length, device=cfg.device)

    # STFT
    stft_mix = torch.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True
    )

    # Magnitude + phase
    mag_mix = torch.abs(stft_mix)
    phase_mix = torch.angle(stft_mix)

    # 🔥 FIX: Add channel dimension for U-Net
    mag_mix = mag_mix.unsqueeze(1)    # [B, 1, F, T]

    # Forward pass through the model
    est_mask = model(mag_mix)         # → [B, 1, F, T]

    # Remove channel dimension
    est_mask = est_mask.squeeze(1)    # → [B, F, T]

    # Apply mask
    est_complex = est_mask * stft_mix

    # ISTFT
    enhanced = torch.istft(
        est_complex,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window
    )

    return enhanced.squeeze().cpu().detach().numpy()

# --------------------------- Run ---------------------------
if __name__ == '__main__':
    print('Device:', cfg.device)
    train()
