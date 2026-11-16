"""
Connectome-constrained "fly brain" audio classifier — sample / starting point.

Idea:
  1. Use a sparse graph (ideally the real FlyWire connectome, or a synthetic
     fly-like graph as a stand-in) as a FIXED wiring topology.
  2. Build a small recurrent network whose connections are masked by that
     topology, but whose weights are trainable.
  3. Feed short audio training clips in as a time series of "sensory" input.
  4. Attach a trainable linear readout over N output classes (your anime list).
  5. Train end-to-end with backprop on labeled training clips.

This is a toy/research-style sketch, not a plug-and-play tool. You will need
to supply your own labeled audio training clips and, if you want the real connectome,
FlyWire/CAVE credentials (see NOTES at the bottom).

Requires: torch, numpy, librosa
    pip install torch numpy librosa --break-system-packages
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------
# 1. WIRING: synthetic fly-like graph (swap in real FlyWire data if you have it)
# -----------------------------------------------------------------------

def make_synthetic_fly_graph(n_neurons=2000, avg_out_degree=25, seed=0, target_row_scale=1.0):
    """
    Build a sparse random directed graph with connectivity statistics
    loosely resembling the fly connectome (sparse, heavy-tailed synapse
    counts, mostly local + some long-range hubs). This is NOT real fly
    data -- it's a stand-in so the code runs without needing FlyWire access.

    target_row_scale: after building the graph, weights are rescaled so a
    typical neuron's *summed* incoming input is around this value. Raw
    synapse-count-like weights (lognormal, mean ~1.6 each, ~25 inputs per
    neuron) sum to ~40 per neuron unscaled -- that's large enough to
    immediately saturate a tanh activation, which stalls learning. Rescaling
    keeps the network in a regime where it can actually learn.
    """
    rng = np.random.default_rng(seed)
    n_edges = n_neurons * avg_out_degree
    src = rng.integers(0, n_neurons, size=n_edges)
    dst = rng.integers(0, n_neurons, size=n_edges)
    # lognormal synapse "weights" (counts), matches the heavy-tailed pattern
    # reported for real connectomes
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=n_edges)
    # drop self-loops
    mask = src != dst
    src, dst, weights = src[mask], dst[mask], weights[mask]

    adj = torch.zeros(n_neurons, n_neurons)
    adj[dst, src] = torch.tensor(weights, dtype=torch.float32)  # adj[i,j] = j -> i

    # Rescale so each row's (each neuron's) summed absolute input is roughly
    # target_row_scale, instead of growing with avg_out_degree.
    row_sums = adj.abs().sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero for isolated neurons
    adj = adj * (target_row_scale / row_sums)

    return adj


def load_real_flywire_graph(neuron_ids=None):
    """
    Placeholder for loading the real connectome. In practice you'd use the
    `fafbseg` / `caveclient` packages against the FlyWire CAVE API, e.g.:

        from caveclient import CAVEclient
        client = CAVEclient('flywire_fafb_production')
        edges = client.materialize.query_table('synapses_nt_v1', ...)

    then pivot into a (post_id x pre_id) weighted adjacency matrix, restricted
    to whichever neuron subset you care about (e.g. Johnston's Organ neurons
    -> AMMC -> mushroom body -> your custom readout).
    This requires a registered FlyWire/CAVE account and API token.
    """
    raise NotImplementedError("Plug in caveclient / fafbseg calls here.")


# -----------------------------------------------------------------------
# 2. MODEL: connectome-masked recurrent network + trainable readout
# -----------------------------------------------------------------------

class ConnectomeRNN(nn.Module):
    def __init__(self, adjacency: torch.Tensor, n_input_channels: int, n_classes: int,
                 n_input_neurons: int = 128, dt=1.0, tau=10.0):
        super().__init__()
        n = adjacency.shape[0]
        self.n_neurons = n
        self.dt = dt
        self.tau = tau

        # Fixed sparsity mask from the connectome graph (nonzero = allowed connection)
        mask = (adjacency != 0).float()
        self.register_buffer("mask", mask)

        # Trainable weights, but only nonzero where the connectome says a
        # synapse exists. Initialize from the connectome's own weight scale.
        init_w = adjacency.clone()
        self.W = nn.Parameter(init_w)

        # Map audio features onto a subset of "input" neurons
        # (stand-in for Johnston's Organ neurons receiving vibration input)
        self.n_input_neurons = n_input_neurons
        self.input_proj = nn.Linear(n_input_channels, n_input_neurons)

        # Trainable readout from full population activity to class logits
        self.readout = nn.Linear(n, n_classes)

    def forward(self, audio_features):
        # audio_features: (batch, time, n_input_channels)
        batch, T, _ = audio_features.shape
        device = audio_features.device
        h = torch.zeros(batch, self.n_neurons, device=device)

        # project audio into the first n_input_neurons "sensory" units per timestep
        drive = self.input_proj(audio_features)  # (batch, T, n_input_neurons)

        alpha = self.dt / self.tau
        effective_W = self.W * self.mask  # enforce connectome sparsity every step

        for t in range(T):
            external = torch.zeros(batch, self.n_neurons, device=device)
            external[:, :self.n_input_neurons] = drive[:, t, :]

            recurrent_input = h @ effective_W.T
            h = (1 - alpha) * h + alpha * torch.tanh(recurrent_input + external)

        logits = self.readout(h)
        return logits


# -----------------------------------------------------------------------
# 3. AUDIO FEATURES: chop a song into overlapping windows, then featurize each
# -----------------------------------------------------------------------

def mel_features_from_waveform(y, sr, n_mels=40, hop_length=512, max_frames=200):
    """Turn a raw waveform array into a fixed-size (time, n_mels) tensor."""
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T  # (time, n_mels)
    if mel_db.shape[0] < max_frames:
        pad = max_frames - mel_db.shape[0]
        mel_db = np.pad(mel_db, ((0, pad), (0, 0)))
    else:
        mel_db = mel_db[:max_frames]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def audio_to_segments_with_times(path, sr=16000, n_mels=40, hop_length=512, max_frames=200,
                                  segment_seconds=6.0, segment_overlap=0.5, min_segment_seconds=2.0):
    """
    Like audio_to_segment_features, but also returns each window's fractional
    position within the song (0.0 = start, 1.0 = end). This lets us split
    train/val by TIME WITHIN each song, rather than by whole file -- which is
    the right setup for AMQ-style recognition, where the model is *supposed*
    to already know the song and just needs to recognize an arbitrary snippet
    of it. We hold out snippets from a part of the song it hasn't trained on,
    to check it can still recognize the song from there.
    """
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    seg_len = int(segment_seconds * sr)
    hop = int(seg_len * (1 - segment_overlap))
    hop = max(hop, 1)
    min_len = int(min_segment_seconds * sr)
    total_len = len(y)

    if total_len <= seg_len:
        windows = [(y, 0.0)]
    else:
        windows = []
        start = 0
        while start < total_len:
            chunk = y[start:start + seg_len]
            if len(chunk) >= min_len:
                frac_pos = start / total_len  # 0.0 (song start) .. ~1.0 (song end)
                windows.append((chunk, frac_pos))
            start += hop

    return [
        (mel_features_from_waveform(w, sr, n_mels=n_mels, hop_length=hop_length, max_frames=max_frames), frac)
        for w, frac in windows
    ]


# -----------------------------------------------------------------------
# 4. DATA LOADING: auto-discover training clips from a folder-per-class layout
# -----------------------------------------------------------------------

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def scan_dataset_folder(root_dir):
    """
    Expects a layout like:

        training clips/
          ShowA_OP/
            song1.mp3
            song2.wav
          ShowB_OP/
            clip1.mp3
          ShowC_ED/
            clip1.flac

    Each subfolder name becomes a class label. Returns:
        dataset      -> list of (filepath, label_index)
        class_names  -> list of subfolder names, index-aligned with labels
    """
    import os

    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"No such folder: {root_dir}")

    class_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    if not class_names:
        raise ValueError(
            f"No subfolders found in {root_dir}. Organize training clips as "
            f"{root_dir}/<AnimeName>/<clip files>."
        )

    dataset = []
    for label_idx, name in enumerate(class_names):
        class_dir = os.path.join(root_dir, name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(AUDIO_EXTS)]
        if not files:
            print(f"Warning: no audio files found in {class_dir}")
        for f in files:
            dataset.append((os.path.join(class_dir, f), label_idx))

    print(f"Found {len(dataset)} training clips across {len(class_names)} classes: {class_names}")
    return dataset, class_names


# -----------------------------------------------------------------------
# 5. TRAINING LOOP
# -----------------------------------------------------------------------

def train_from_folder(root_dir="training clips", n_mels=40, max_frames=200,
                       epochs=100, val_fraction=0.0, seed=0,
                       segment_seconds=6.0, segment_overlap=0.5,
                       batch_size=16, lr=1e-3, grad_clip=1.0):
    dataset, class_names = scan_dataset_folder(root_dir)
    n_classes = len(class_names)

    # Split each song TEMPORALLY: the first (1 - val_fraction) portion of its
    # timeline supplies training segments, the last val_fraction portion
    # supplies validation segments. This matches the actual AMQ task -- the
    # model is meant to already know the song, it just needs to recognize an
    # arbitrary snippet of it. So we test "can it recognize a snippet from a
    # part of the song it never specifically trained on," not "can it
    # recognize an entirely unseen song."
    def build_temporal_split():
        train_feats, train_labels = [], []
        val_feats, val_labels = [], []
        for path, label in dataset:
            try:
                segments = audio_to_segments_with_times(
                    path, n_mels=n_mels, max_frames=max_frames,
                    segment_seconds=segment_seconds, segment_overlap=segment_overlap,
                )
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            for feat, frac_pos in segments:
                if frac_pos >= (1 - val_fraction):
                    val_feats.append(feat)
                    val_labels.append(label)
                else:
                    train_feats.append(feat)
                    train_labels.append(label)

        def to_tensors(feats, labels):
            X = torch.tensor(np.stack(feats)) if feats else torch.empty(0, max_frames, n_mels)
            y = torch.tensor(labels, dtype=torch.long)
            return X, y

        return to_tensors(train_feats, train_labels), to_tensors(val_feats, val_labels)

    (X_train, y_train), (X_val, y_val) = build_temporal_split()
    if val_fraction > 0:
        print(f"Expanded into {len(X_train)} training segments, {len(X_val)} val segments "
              f"(val = last {int(val_fraction*100)}% of each song's timeline)")
    else:
        print(f"Expanded into {len(X_train)} training segments (using full songs, no held-out portion)")

    if len(X_train) == 0:
        raise RuntimeError("No training clips loaded successfully -- check file paths/formats.")

    adjacency = make_synthetic_fly_graph(n_neurons=2000, avg_out_degree=25)
    model = ConnectomeRNN(adjacency, n_input_channels=n_mels, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(X_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss, epoch_correct = 0.0, 0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)
            epoch_correct += (logits.argmax(-1) == yb).sum().item()

        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train

        msg = f"epoch {epoch:03d}  loss {train_loss:.4f}  train_acc {train_acc:.2f}"
        if X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()
            msg += f"  val_acc {val_acc:.2f}"
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(msg)

    return model, class_names


# -----------------------------------------------------------------------
# 6. SAVE / LOAD
# -----------------------------------------------------------------------

def save_model(model, class_names, path="fly_amq_model.pt", **hparams):
    """
    Saves everything needed to reload and use the model later:
      - model weights (state_dict)
      - the connectome adjacency shape/mask info (needed to rebuild the
        architecture before loading weights into it)
      - class_names, so predictions can be mapped back to show names
      - the audio feature hyperparameters used during training (n_mels,
        max_frames, segment_seconds, etc.) so inference uses matching settings
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "mask": model.mask,              # connectome sparsity pattern, needed to rebuild
        "n_neurons": model.n_neurons,
        "n_input_neurons": model.n_input_neurons,
        "class_names": class_names,
        "hparams": hparams,              # e.g. n_mels, max_frames, segment_seconds
    }, path)
    print(f"Saved model to {path}")


def load_model(path="fly_amq_model.pt"):
    """
    Rebuilds a ConnectomeRNN with the saved architecture and loads the
    trained weights back into it. Returns (model, class_names, hparams).
    """
    checkpoint = torch.load(path, weights_only=False)

    n_neurons = checkpoint["n_neurons"]
    n_input_neurons = checkpoint["n_input_neurons"]
    hparams = checkpoint["hparams"]
    n_classes = len(checkpoint["class_names"])

    # rebuild an empty adjacency shell just to construct the model object;
    # its actual values get overwritten by load_state_dict right after
    dummy_adjacency = torch.zeros(n_neurons, n_neurons)
    model = ConnectomeRNN(
        dummy_adjacency,
        n_input_channels=hparams["n_mels"],
        n_classes=n_classes,
        n_input_neurons=n_input_neurons,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {path} ({n_classes} classes)")
    return model, checkpoint["class_names"], hparams


# -----------------------------------------------------------------------
# 7. INFERENCE: identify a snippet using a saved model
# -----------------------------------------------------------------------

def predict_snippet(model, class_names, hparams, audio_path, top_k=3, verbose=True):
    """
    Loads an audio snippet (e.g. a 20-second AMQ clip), scans it using the
    SAME window size/overlap the model was trained on, runs every window
    through the model, and averages the resulting probabilities. This uses
    the whole snippet as evidence instead of only its first few seconds,
    which matters a lot once snippets are longer than one training window.
    """
    import librosa

    sr = hparams.get("sr", 16000)
    n_mels = hparams["n_mels"]
    max_frames = hparams["max_frames"]
    hop_length = hparams.get("hop_length", 512)
    segment_seconds = hparams.get("segment_seconds", 6.0)
    segment_overlap = hparams.get("segment_overlap", 0.5)
    min_segment_seconds = hparams.get("min_segment_seconds", 2.0)

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    seg_len = int(segment_seconds * sr)
    hop = max(int(seg_len * (1 - segment_overlap)), 1)
    min_len = int(min_segment_seconds * sr)

    windows, starts = [], []
    if len(y) <= seg_len:
        windows, starts = [y], [0]
    else:
        start = 0
        while start < len(y):
            chunk = y[start:start + seg_len]
            if len(chunk) >= min_len:
                windows.append(chunk)
                starts.append(start)
            start += hop

    feats = [
        mel_features_from_waveform(w, sr, n_mels=n_mels, hop_length=hop_length, max_frames=max_frames)
        for w in windows
    ]
    X = torch.tensor(np.stack(feats))  # (n_windows, T, n_mels)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=-1)   # (n_windows, n_classes) -- one guess per window
        avg_probs = probs.mean(dim=0)       # aggregate evidence across the whole snippet

    k = min(top_k, len(class_names))
    top_probs, top_idx = torch.topk(avg_probs, k)

    if verbose:
        print(f"\nScanned {len(windows)} windows ({segment_seconds}s each) from {audio_path}")
        print("Per-window top guess (sanity check -- should mostly agree):")
        for i, (wp, s) in enumerate(zip(probs, starts)):
            wi = wp.argmax().item()
            t0 = s / sr
            print(f"  {t0:5.1f}s-{t0 + segment_seconds:5.1f}s: {class_names[wi]}  ({wp[wi].item()*100:.1f}%)")

        print("\nFinal prediction (averaged across all windows):")
        for rank, (p, idx) in enumerate(zip(top_probs.tolist(), top_idx.tolist()), start=1):
            print(f"  {rank}. {class_names[idx]}   ({p * 100:.1f}%)")

    best_idx = top_idx[0].item()
    return class_names[best_idx], top_probs[0].item()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        # usage: python3 fly_amq_classifier.py predict path/to/snippet.mp3 [model.pt]
        if len(sys.argv) < 3:
            print("Usage: python3 fly_amq_classifier.py predict <snippet_audio_path> [model_path]")
            sys.exit(1)
        snippet_path = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else "fly_amq_model.pt"

        model, class_names, hparams = load_model(model_path)
        predict_snippet(model, class_names, hparams, snippet_path)

    else:
        # usage: python3 fly_amq_classifier.py [clips_folder]
        folder = sys.argv[1] if len(sys.argv) > 1 else "training clips"
        model, class_names = train_from_folder(root_dir=folder)
        save_model(
            model, class_names, path="fly_amq_model.pt",
            n_mels=40, max_frames=200, segment_seconds=6.0, segment_overlap=0.5,
            sr=16000, hop_length=512,
        )


"""
NOTES
-----
1. Real connectome data: to swap in actual FlyWire connectivity instead of
   the synthetic graph, you need a registered FlyWire/CAVE account, then use
   `caveclient` or `fafbseg` to pull synapse tables and build a weighted
   adjacency matrix over the neuron subset you want (e.g. Johnston's Organ ->
   AMMC -> mushroom body). See https://caveclient.readthedocs.io.

2. Scale: the real connectome has ~140,000 neurons and ~50 million synapses.
   Running the full thing as a dense recurrent loop (as above) will be very
   slow/memory-heavy -- for anything beyond a small toy example you'll want
   sparse tensor ops (torch.sparse) or a proper spiking-network simulator
   (e.g. Brian2, NEST) rather than this dense-matrix RNN.

3. Biological honesty: nothing about the connectome's topology gives this
   network any innate skill at recognizing anime songs. The connectome only
   supplies a fixed sparsity pattern; all task-relevant learning happens in
   the trainable weights and readout, same as any other neural net. Treat
   this as "a biologically-inspired sparse RNN," not literally a fly hearing
   music.
"""