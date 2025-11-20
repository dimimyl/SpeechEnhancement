"""
Run speech enhancement inference using a trained model and an open-source LibriSpeech sample.

This script:
  1. Loads a pretrained model checkpoint from ./checkpoints/
  2. Downloads a LibriSpeech test-clean sample
  3. Adds synthetic Gaussian noise at a specified SNR
  4. Runs the model to enhance the noisy speech
  5. Saves the enhanced waveform as 'enhanced_from_librispeech.wav'

Requirements:
  pip install torch torchaudio librosa numpy soundfile
"""

import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

from main import SmallUNet, enhance_waveform, Config  # assumes main.py is in the same folder

cfg = Config()

# --------------------------- Helper functions ---------------------------
def load_trained_model(checkpoint_path):
    n_fft_bins = cfg.n_fft // 2 + 1
    model = SmallUNet(n_fft_bins)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.device)
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return model


def get_noisy_sample(snr_db=5.0):
    sample_ds = torchaudio.datasets.LIBRISPEECH(root=cfg.root, url="test-clean", download=True)

    # Correct for your torchaudio version (6 return values)
    waveform, sr, transcript, speaker_id, chapter_id, utterance_id = sample_ds[0]

    waveform = waveform.squeeze().numpy()

    if sr != cfg.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.sample_rate)

    # Add Gaussian noise at desired SNR
    rms_clean = np.sqrt(np.mean(waveform**2))
    noise = np.random.randn(len(waveform)).astype(np.float32)
    rms_noise = np.sqrt(np.mean(noise**2))
    desired_rms_noise = rms_clean / (10 ** (snr_db / 20))
    noise *= desired_rms_noise / (rms_noise + 1e-12)

    noisy_waveform = waveform + noise
    print(f"Generated noisy sample at {snr_db} dB SNR from LibriSpeech")
    return noisy_waveform, waveform

# --------------------------- Main ---------------------------
def main():
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # choose last available checkpoint
    checkpoints = sorted(Path(cfg.checkpoint_dir).glob('model_epoch_*.pt'))
    if not checkpoints:
        raise FileNotFoundError("No trained model found in ./checkpoints/. Run main.py to train first.")
    checkpoint_path = str(checkpoints[-1])

    # load model
    model = load_trained_model(checkpoint_path)

    # get sample and enhance
    noisy, clean = get_noisy_sample(snr_db=5.0)
    enhanced = enhance_waveform(noisy, model, cfg)

    # save results
    sf.write("noisy_from_librispeech.wav", noisy, cfg.sample_rate)
    sf.write("enhanced_from_librispeech.wav", enhanced, cfg.sample_rate)
    sf.write("clean_reference.wav", clean, cfg.sample_rate)
    print("Saved noisy, clean reference, and enhanced audio files.")

if __name__ == '__main__':
    main()
