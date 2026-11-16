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

from fly_amq_classifier_real import load_model, mel_features_from_waveform

MODEL_PATH = "fly_amq_model_real.pt"
DEVICE_NAME = None       # e.g. "BlackHole 2ch" -- None = use default input device
WINDOW_SECONDS = 6.0     # should roughly match hparams['segment_seconds'] from training
TOP_K = 3
OUTPUT_FILE = "current_guess.txt"   # written only when confidence clears the threshold below
CONFIDENCE_THRESHOLD = 0.90          # only write to file when top guess is >= this confident

# --- song-boundary detection settings ---
# AMQ plays songs back-to-back with no silence gap, so we can't key off volume
# dropping out. Instead we watch for a SUDDEN CHANGE in the audio's frequency
# content ("fingerprint") -- a song transition is one of the few moments where
# that content jumps abruptly, rather than drifting smoothly like it does
# within a single song.
CHECK_INTERVAL = 0.5            # how often (seconds) to compare fingerprints / check confidence
FINGERPRINT_WINDOW = 1.0        # seconds of audio each fingerprint summarizes
NOVELTY_HISTORY = 12            # how many recent novelty scores to use as a baseline
NOVELTY_STD_MULTIPLIER = 2.0    # flag a boundary when novelty exceeds baseline_mean + this*std
MIN_NOVELTY_HISTORY = 6         # need at least this many samples before trusting the baseline
COOLDOWN_SECONDS = 3.0          # ignore further boundary triggers for this long after one fires

# --- early-exit guessing settings ---
# Rather than always waiting a fixed number of seconds after a boundary, we
# check the model's confidence as fresh audio comes in and guess as soon as
# it's confident -- so easy/distinctive songs get answered fast, while
# harder/similar-sounding ones naturally get more time before answering.
MIN_SECONDS_BEFORE_CHECKING = 2.0   # don't even try guessing on less than this much audio
EARLY_EXIT_CONFIDENCE = 0.90        # guess immediately once top confidence reaches this
# WINDOW_SECONDS above also acts as a hard cap: if we reach a full window's
# worth of fresh audio without hitting EARLY_EXIT_CONFIDENCE, guess anyway
# rather than waiting forever.


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


def spectral_fingerprint(y, sr, n_bands=20):
    """
    A cheap, small summary of a short audio chunk's frequency content:
    average log-energy in n_bands mel bands. Fast enough to compute several
    times a second, and similar chunks (same song) should give similar
    fingerprints, while a song transition should produce a visibly different one.
    """
    import librosa
    if len(y) < 512:
        return np.zeros(n_bands, dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bands, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.mean(axis=1).astype(np.float32)  # average over time -> (n_bands,)


def guess_from_buffer(model, class_names, song_buffer, sr, n_mels, hop_length, max_frames, top_k=TOP_K):
    feat = mel_features_from_waveform(
        song_buffer, sr, n_mels=n_mels, hop_length=hop_length, max_frames=max_frames
    )
    X = torch.tensor(feat).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    k = min(top_k, len(class_names))
    top_probs, top_idx = torch.topk(probs, k)
    return top_probs, top_idx


def main():
    model, class_names, hparams = load_model(MODEL_PATH)
    sr = hparams.get("sr", 16000)
    n_mels = hparams["n_mels"]
    max_frames = hparams["max_frames"]
    hop_length = hparams.get("hop_length", 512)

    device_idx = find_device(DEVICE_NAME)
    window_samples = int(WINDOW_SECONDS * sr)
    fingerprint_samples = int(FINGERPRINT_WINDOW * sr)

    # rolling buffer of the most recent audio (used continuously for boundary detection)
    buffer = np.zeros(window_samples, dtype=np.float32)
    # separate buffer that starts EMPTY and grows from zero at each detected
    # boundary -- this is what actually gets fed to the model, so it never
    # contains leftover audio from the previous song.
    song_buffer = np.zeros(0, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer, song_buffer
        if status:
            print(status)
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        buffer = np.concatenate([buffer, mono])[-window_samples:]
        song_buffer = np.concatenate([song_buffer, mono])[-window_samples:]

    print(f"Listening on device: {sd.query_devices(device_idx)['name'] if device_idx is not None else 'default'}")
    print("Ctrl+C to stop.\n")

    from collections import deque
    novelty_history = deque(maxlen=NOVELTY_HISTORY)
    prev_fingerprint = None
    cooldown_remaining = 0.0
    pending_guess = False   # a boundary fired and we haven't locked in a guess yet

    with sd.InputStream(device=device_idx, channels=1, samplerate=sr, callback=audio_callback):
        try:
            while True:
                time.sleep(CHECK_INTERVAL)
                cooldown_remaining = max(0.0, cooldown_remaining - CHECK_INTERVAL)

                recent_chunk = buffer[-fingerprint_samples:]
                fingerprint = spectral_fingerprint(recent_chunk, sr)

                if prev_fingerprint is not None:
                    novelty = float(np.linalg.norm(fingerprint - prev_fingerprint))

                    if len(novelty_history) >= MIN_NOVELTY_HISTORY:
                        baseline_mean = np.mean(novelty_history)
                        baseline_std = np.std(novelty_history) + 1e-6
                        threshold = baseline_mean + NOVELTY_STD_MULTIPLIER * baseline_std

                        if novelty > threshold and cooldown_remaining <= 0.0:
                            print(f"[{time.strftime('%H:%M:%S')}] (spectral jump detected -- "
                                  f"probably a new song; novelty={novelty:.1f} vs threshold={threshold:.1f})")
                            cooldown_remaining = COOLDOWN_SECONDS
                            pending_guess = True
                            song_buffer = np.zeros(0, dtype=np.float32)   # start fresh for the new song

                    novelty_history.append(novelty)

                prev_fingerprint = fingerprint

                if pending_guess:
                    seconds_collected = len(song_buffer) / sr
                    have_full_window = seconds_collected >= WINDOW_SECONDS

                    if seconds_collected >= MIN_SECONDS_BEFORE_CHECKING:
                        top_probs, top_idx = guess_from_buffer(
                            model, class_names, song_buffer, sr, n_mels, hop_length, max_frames
                        )
                        top_confidence = top_probs[0].item()

                        should_commit = (top_confidence >= EARLY_EXIT_CONFIDENCE) or have_full_window

                        if should_commit:
                            guesses = ", ".join(
                                f"{class_names[i]} ({p*100:.0f}%)"
                                for p, i in zip(top_probs.tolist(), top_idx.tolist())
                            )
                            line = f"[{time.strftime('%H:%M:%S')}] ({seconds_collected:.1f}s heard) {guesses}"
                            print(line)

                            if top_confidence >= CONFIDENCE_THRESHOLD:
                                with open(OUTPUT_FILE, "w") as f:
                                    f.write(line + "\n")
                                print(f"  -> confidence {top_confidence*100:.0f}% >= threshold, wrote to {OUTPUT_FILE}")

                            pending_guess = False   # done until the next detected boundary
                        else:
                            print(f"[{time.strftime('%H:%M:%S')}] ({seconds_collected:.1f}s heard) "
                                  f"still unsure ({class_names[top_idx[0].item()]} @ {top_confidence*100:.0f}%) -- waiting for more audio")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()