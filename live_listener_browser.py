"""
Same guessing logic as live_listener.py, but audio comes from a browser tab
(via tab_capture.html) over a local WebSocket instead of from sounddevice /
BlackHole. Run this, then open tab_capture.html in Chrome and click Start.

Guesses are also BROADCAST back over the same WebSocket as text messages, so
a userscript running on the quiz page can type them into the answer box.
Binary messages = audio from the browser; text messages = guesses to the page.

Manual toggle: press Enter in this terminal the moment a new song starts and
the listener resets and starts guessing. Type "a" then Enter to switch the
automatic spectral-jump detection on or off while it's running.

pip install websockets --break-system-packages   (websockets >= 10 for broadcast)
"""

import asyncio
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import websockets

from fly_amq_classifier_real import load_model, mel_features_from_waveform

MODEL_PATH = "fly_amq_model_real.pt"
WS_HOST = "localhost"
WS_PORT = 8765

WINDOW_SECONDS = 6.0
TOP_K = 3
OUTPUT_FILE = "current_guess.txt"
CONFIDENCE_THRESHOLD = 0.90

CHECK_INTERVAL = 0.5
FINGERPRINT_WINDOW = 1.0
NOVELTY_HISTORY = 12
NOVELTY_STD_MULTIPLIER = 2.0
MIN_NOVELTY_HISTORY = 6
COOLDOWN_SECONDS = 3.0

MIN_SECONDS_BEFORE_CHECKING = 1.5
EARLY_EXIT_CONFIDENCE = 0.85

# --- manual toggle ---
MANUAL_TRIGGER = True   # press Enter in this terminal when a new song starts
AUTO_DETECT = False     # starting state of automatic detection; type "a" + Enter to flip it live

# --- send guesses to the web page ---
SEND_TO_PAGE = True     # broadcast the top guess to connected pages over the WebSocket

# every connected WebSocket (the audio-capture tab and/or the quiz page userscript)
clients = set()


def spectral_fingerprint(y, sr, n_bands=20):
    import librosa
    if len(y) < 512:
        return np.zeros(n_bands, dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_bands, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.mean(axis=1).astype(np.float32)


def guess_from_buffer(model, class_names, song_buffer, sr, n_mels, hop_length, max_frames, top_k=TOP_K):
    feat = mel_features_from_waveform(song_buffer, sr, n_mels=n_mels, hop_length=hop_length, max_frames=max_frames)
    X = torch.tensor(feat).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1).squeeze(0)
    k = min(top_k, len(class_names))
    return torch.topk(probs, k)


class ListenerState:
    """Holds the rolling buffers and boundary-detection state, updated as
    audio chunks arrive from the browser over the WebSocket."""

    def __init__(self, sr, window_samples):
        self.sr = sr
        self.window_samples = window_samples
        self.buffer = np.zeros(window_samples, dtype=np.float32)
        self.song_buffer = np.zeros(0, dtype=np.float32)
        self.novelty_history = deque(maxlen=NOVELTY_HISTORY)
        self.prev_fingerprint = None
        self.cooldown_remaining = 0.0
        self.pending_guess = False
        self.auto_detect = AUTO_DETECT
        self.last_sent = None   # last guess pushed to the page, to avoid retyping the same answer

    def start_new_song(self):
        """Reset so the next guess uses only audio heard from this moment on."""
        self.cooldown_remaining = COOLDOWN_SECONDS
        self.pending_guess = True
        self.song_buffer = np.zeros(0, dtype=np.float32)
        self.last_sent = None

    def push_audio(self, mono_chunk):
        self.buffer = np.concatenate([self.buffer, mono_chunk])[-self.window_samples:]
        self.song_buffer = np.concatenate([self.song_buffer, mono_chunk])[-self.window_samples:]


async def audio_receiver(websocket, state):
    """Reads incoming binary Float32 PCM messages from the browser and pushes
    them into the shared buffers. Text messages are ignored."""
    async for message in websocket:
        if isinstance(message, (bytes, bytearray)):
            chunk = np.frombuffer(message, dtype=np.float32)
            state.push_audio(chunk)


async def guessing_loop(model, class_names, hparams, state):
    sr = hparams.get("sr", 16000)
    n_mels = hparams["n_mels"]
    max_frames = hparams["max_frames"]
    hop_length = hparams.get("hop_length", 512)
    fingerprint_samples = int(FINGERPRINT_WINDOW * sr)

    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        state.cooldown_remaining = max(0.0, state.cooldown_remaining - CHECK_INTERVAL)

        if state.auto_detect:
            recent_chunk = state.buffer[-fingerprint_samples:]
            fingerprint = spectral_fingerprint(recent_chunk, sr)
        else:
            fingerprint = None
            state.prev_fingerprint = None   # don't compare against a stale fingerprint after re-enabling
            state.novelty_history.clear()

        if state.auto_detect and state.prev_fingerprint is not None:
            novelty = float(np.linalg.norm(fingerprint - state.prev_fingerprint))

            if len(state.novelty_history) >= MIN_NOVELTY_HISTORY:
                baseline_mean = np.mean(state.novelty_history)
                baseline_std = np.std(state.novelty_history) + 1e-6
                threshold = baseline_mean + NOVELTY_STD_MULTIPLIER * baseline_std

                if novelty > threshold and state.cooldown_remaining <= 0.0:
                    print(f"[{time.strftime('%H:%M:%S')}] (spectral jump detected -- "
                          f"probably a new song; novelty={novelty:.1f} vs threshold={threshold:.1f})")
                    state.start_new_song()

            state.novelty_history.append(novelty)

        state.prev_fingerprint = fingerprint

        if state.pending_guess:
            seconds_collected = len(state.song_buffer) / sr
            have_full_window = seconds_collected >= WINDOW_SECONDS

            if seconds_collected >= MIN_SECONDS_BEFORE_CHECKING:
                top_probs, top_idx = guess_from_buffer(
                    model, class_names, state.song_buffer, sr, n_mels, hop_length, max_frames
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

                    # push the top guess to any connected page, but only when it changes
                    best = class_names[top_idx[0].item()]
                    if SEND_TO_PAGE and clients and best != state.last_sent:
                        websockets.broadcast(clients, best)   # sent as a text frame
                        state.last_sent = best
                        print(f"  -> sent to {len(clients)} connected page(s): {best}")

                    # if top_confidence >= CONFIDENCE_THRESHOLD:
                    #     try:
                    #         # utf-8 explicitly: Windows defaults to cp1252, which can't encode
                    #         # characters like the star in "Tenbatsu Angel Rabbie☆"
                    #         with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    #             f.write(line + "\n")
                    #         print(f"  -> confidence {top_confidence*100:.0f}% >= threshold, wrote to {OUTPUT_FILE}")
                    #     except OSError as e:
                    #         print(f"  -> couldn't write {OUTPUT_FILE}: {e}")
                    #
                    # state.pending_guess = False
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ({seconds_collected:.1f}s heard) "
                          f"still unsure ({class_names[top_idx[0].item()]} @ {top_confidence*100:.0f}%) -- waiting")


async def manual_trigger_loop(state):
    """Terminal controls: Enter = a new song just started; "a" + Enter = toggle auto-detect."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, input)
        except EOFError:
            return
        cmd = line.strip().lower()
        if cmd == "":
            print(f"[{time.strftime('%H:%M:%S')}] (manual trigger -- new song)")
            state.start_new_song()
        elif cmd == "a":
            state.auto_detect = not state.auto_detect
            print(f"[{time.strftime('%H:%M:%S')}] auto-detect {'ON' if state.auto_detect else 'OFF'}")
        else:
            print('Unknown command. Enter = new song, "a" + Enter = toggle auto-detect.')


async def main():
    model, class_names, hparams = load_model(MODEL_PATH)
    sr = hparams.get("sr", 16000)
    window_samples = int(WINDOW_SECONDS * sr)
    state = ListenerState(sr, window_samples)

    if MANUAL_TRIGGER:
        asyncio.create_task(manual_trigger_loop(state))
        print('Press Enter when a new song starts. Type "a" + Enter to toggle auto-detect '
              f"(currently {'ON' if state.auto_detect else 'OFF'}).")

    async def handler(websocket):
        clients.add(websocket)
        print(f"Browser connected ({len(clients)} total) -- listening for audio.")
        try:
            await audio_receiver(websocket, state)
        finally:
            clients.discard(websocket)
            print(f"Browser disconnected ({len(clients)} remaining).")

    asyncio.create_task(guessing_loop(model, class_names, hparams, state))

    print(f"WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
    print("Now open tab_capture.html in Chrome and click Start Capture.\n")

    async with websockets.serve(handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())