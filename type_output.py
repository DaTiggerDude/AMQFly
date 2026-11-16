"""
Live "what song is this" listener -- for personal fun/curiosity only.

This captures system audio in real time and periodically runs the last few
seconds through your trained fly_amq_model.pt, printing a running guess to
the terminal. It does NOT interact with AMQ's website, detect game state, or
submit answers -- you'd read the guess yourself and type your own answer in,
same as always. This is meant to let you see how the model does against your
own memory for fun, not to automate play in a real match.

SETUP (one-time):
  1. Install a loopback audio device so Python can "hear" your system audio
     output (not just your microphone):
       - Mac: install BlackHole (https://existential.audio/blackhole/) or
         Loopback (rogueamoeba.com). Route AMQ's tab/browser audio through it.
       - Windows: enable "Stereo Mix" in sound settings, or install VB-Cable.
     Then set your system output (or just AMQ's browser tab) to route through
     that virtual device, AND select that same device as the input below.

  2. pip install sounddevice --break-system-packages

  3. Run: python3 live_listener.py
     (edit DEVICE_NAME below if it doesn't find your loopback device automatically)
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd

from fly_amq_classifier import load_model, mel_features_from_waveform

MODEL_PATH = "fly_amq_model.pt"
DEVICE_NAME = "BlackHole 2ch"       # e.g. "BlackHole 2ch" -- None = use default input device
POLL_INTERVAL = 2.0      # seconds between guesses
WINDOW_SECONDS = 6.0     # should roughly match hparams['segment_seconds'] from training
TOP_K = 3
OUTPUT_FILE = "current_guess.txt"   # written only when confidence clears the threshold below
CONFIDENCE_THRESHOLD = 0.90          # only write to file when top guess is >= this confident


def find_device(name):
    if name is None:
        return None
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if name.lower() in d["name"].lower():
            return i
    print(f"Warning: couldn't find a device matching '{name}'. Available devices:")
    print(sd.query_devices())
    return None


def main():
    model, class_names, hparams = load_model(MODEL_PATH)
    sr = hparams.get("sr", 16000)
    n_mels = hparams["n_mels"]
    max_frames = hparams["max_frames"]
    hop_length = hparams.get("hop_length", 512)

    device_idx = find_device(DEVICE_NAME)
    window_samples = int(WINDOW_SECONDS * sr)

    # rolling buffer of the most recent audio
    buffer = np.zeros(window_samples, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            print(status)
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        buffer = np.concatenate([buffer, mono])[-window_samples:]

    print(f"Listening on device: {sd.query_devices(device_idx)['name'] if device_idx is not None else 'default'}")
    print("Ctrl+C to stop.\n")

    with sd.InputStream(device=device_idx, channels=1, samplerate=sr, callback=audio_callback):
        try:
            while True:
                time.sleep(POLL_INTERVAL)

                feat = mel_features_from_waveform(
                    buffer, sr, n_mels=n_mels, hop_length=hop_length, max_frames=max_frames
                )
                X = torch.tensor(feat).unsqueeze(0)

                model.eval()
                with torch.no_grad():
                    logits = model(X)
                    probs = F.softmax(logits, dim=-1).squeeze(0)

                k = min(TOP_K, len(class_names))
                top_probs, top_idx = torch.topk(probs, k)

                guesses = ", ".join(
                    f"{class_names[i]} ({p*100:.0f}%)"
                    for p, i in zip(top_probs.tolist(), top_idx.tolist())
                )
                line = f"[{time.strftime('%H:%M:%S')}] {guesses}"
                print(line)

                top_confidence = top_probs[0].item()
                if top_confidence >= CONFIDENCE_THRESHOLD:
                    with open(OUTPUT_FILE, "w") as f:
                        f.write(line + "\n")
                    print(f"  -> confidence {top_confidence*100:.0f}% >= threshold, wrote to {OUTPUT_FILE}")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()