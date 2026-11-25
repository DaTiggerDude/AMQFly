"""
Audio augmentation + random-crop sampling for training the AMQ classifier.

Pure numpy (no torch / librosa imports), so it can be tested on its own.

Why this exists:
  * PARTIAL WINDOWS: the live listener may guess after only 2-3 seconds of a
    song. If the model only ever trained on full 6-second windows, those early
    guesses are out-of-distribution. Random crop lengths fix that.
  * AUGMENTATION: with ~1 file per class the model memorizes the exact
    training audio. Small random changes (noise, speed, EQ tilt, masking,
    bleed from the previous song) force it to learn what's actually
    distinctive about each song, so it copes with AMQ's different
    encoding / loudness / start point / capture path.

Note: there is deliberately no "random gain" augmentation. The mel features
are computed with power_to_db(ref=max) and then standardized, which already
removes overall volume, so gain changes would do nothing.
"""

import numpy as np

MIN_CROP_SAMPLES = 1024  # skip crops too short to featurize


# -----------------------------------------------------------------------
# Waveform-level helpers
# -----------------------------------------------------------------------

def random_crop(y, sr, rng, min_seconds=1.5, max_seconds=6.0, full_prob=0.4, region=(0.0, 1.0)):
    """
    Cut a random-length crop out of waveform y.

    The crop length is max_seconds with probability full_prob (so full-window
    performance stays strong), otherwise uniform in [min_seconds, max_seconds].
    region=(start_frac, end_frac) restricts where in the song the crop can fall
    (used to keep a held-out part of each song out of training).
    """
    n = len(y)
    lo = int(region[0] * n)
    hi = max(int(region[1] * n), lo + 1)
    span = hi - lo

    max_len = int(max_seconds * sr)
    min_len = min(int(min_seconds * sr), max_len)

    if rng.random() < full_prob:
        crop_len = max_len
    else:
        crop_len = int(rng.integers(min_len, max_len + 1))
    crop_len = min(crop_len, span)

    start = lo + int(rng.integers(0, span - crop_len + 1))
    return y[start:start + crop_len]


def change_speed(y, rate):
    """Tape-style speed change (pitch and tempo move together) via linear resampling."""
    if len(y) < 2 or abs(rate - 1.0) < 1e-3:
        return y
    new_len = max(int(len(y) / rate), 2)
    new_idx = np.linspace(0, len(y) - 1, new_len)
    return np.interp(new_idx, np.arange(len(y)), y).astype(np.float32)


def add_noise(y, rng, snr_db_range=(15.0, 35.0)):
    """Add white noise at a random signal-to-noise ratio."""
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-8:
        return y
    snr_db = rng.uniform(*snr_db_range)
    noise_rms = rms / (10 ** (snr_db / 20))
    return (y + rng.normal(0.0, noise_rms, size=len(y))).astype(np.float32)


def add_previous_song_bleed(y, other, rng, min_seconds=0.3, max_seconds=1.5, sr=16000):
    """
    Replace the first fraction of a second or two of y with audio from a
    different song. This mimics pressing Enter slightly early (or the
    listener's buffer still holding the tail of the previous song), so the
    model learns to focus on the part that is actually the new song.
    """
    bleed_len = int(rng.uniform(min_seconds, max_seconds) * sr)
    bleed_len = min(bleed_len, len(y) // 2, len(other))
    if bleed_len <= 0:
        return y

    start = int(rng.integers(0, len(other) - bleed_len + 1))
    chunk = other[start:start + bleed_len].astype(np.float32)

    # roughly match loudness to the main audio so the bleed isn't trivially quiet/loud
    y_rms = float(np.sqrt(np.mean(y ** 2)))
    c_rms = float(np.sqrt(np.mean(chunk ** 2)))
    if y_rms > 1e-8 and c_rms > 1e-8:
        chunk = chunk * (y_rms / c_rms) * rng.uniform(0.7, 1.3)

    out = y.copy()
    out[:bleed_len] = chunk
    return out


def augment_waveform(y, rng, other=None, sr=16000,
                     bleed_prob=0.25, speed_prob=0.5, speed_range=(0.95, 1.05),
                     noise_prob=0.7, snr_db_range=(15.0, 35.0)):
    if other is not None and rng.random() < bleed_prob:
        y = add_previous_song_bleed(y, other, rng, sr=sr)
    if rng.random() < speed_prob:
        y = change_speed(y, rng.uniform(*speed_range))
    if rng.random() < noise_prob:
        y = add_noise(y, rng, snr_db_range)
    return y


# -----------------------------------------------------------------------
# Feature-level (mel spectrogram) augmentation
# -----------------------------------------------------------------------

def augment_features(feat, rng, tilt_std=0.3, band_std=0.1,
                     freq_mask_prob=0.5, max_freq_mask=4,
                     time_mask_prob=0.5, max_time_mask=10):
    """
    feat: (time, n_mels), already standardized. Applies a random EQ tilt plus
    small per-band offsets (mimics different encoders / EQ / capture chains),
    then optional SpecAugment-style frequency and time masking.
    """
    out = feat.copy()
    n_frames, n_bands = out.shape

    tilt = rng.normal(0.0, tilt_std) * np.linspace(-1.0, 1.0, n_bands)
    band = rng.normal(0.0, band_std, size=n_bands)
    out += (tilt + band)[None, :]

    if rng.random() < freq_mask_prob and n_bands > max_freq_mask:
        w = int(rng.integers(1, max_freq_mask + 1))
        f0 = int(rng.integers(0, n_bands - w + 1))
        out[:, f0:f0 + w] = 0.0

    if rng.random() < time_mask_prob and n_frames > max_time_mask:
        w = int(rng.integers(1, max_time_mask + 1))
        t0 = int(rng.integers(0, n_frames - w + 1))
        out[t0:t0 + w, :] = 0.0

    return out.astype(np.float32)


# -----------------------------------------------------------------------
# Epoch-level sampling
# -----------------------------------------------------------------------

def build_epoch_data(songs, feature_fn, sr, rng, crops_per_song=24,
                     min_crop_seconds=1.5, max_crop_seconds=6.0, full_crop_prob=0.4,
                     region=(0.0, 1.0), augment=True):
    """
    Draw a fresh set of random crops from every song.

    songs:       list of (waveform, label_index)
    feature_fn:  callable(waveform) -> (time, n_mels) float32 array
    Returns (X, y) as numpy arrays: X (N, time, n_mels), y (N,) int64.
    """
    X, y = [], []

    for idx, (wave, label) in enumerate(songs):
        # songs from OTHER classes, used for the "previous song bleed" augmentation
        other_idx = [j for j, (_, l) in enumerate(songs) if l != label]

        for _ in range(crops_per_song):
            crop = random_crop(
                wave, sr, rng,
                min_seconds=min_crop_seconds, max_seconds=max_crop_seconds,
                full_prob=full_crop_prob, region=region,
            )

            if augment:
                other = songs[int(rng.choice(other_idx))][0] if other_idx else None
                crop = augment_waveform(crop, rng, other=other, sr=sr)

            if len(crop) < MIN_CROP_SAMPLES:
                continue

            feat = feature_fn(crop)
            if augment:
                feat = augment_features(feat, rng)

            X.append(feat)
            y.append(label)

    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
