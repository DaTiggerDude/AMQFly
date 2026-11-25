"""
Whole-connectome version: instead of sampling ~2000 neurons, this loads
EVERY neuron and synapse in your connections CSV and builds a SPARSE
recurrent network over the whole thing. Reuses the audio pipeline and
save/load/predict functions from fly_amq_classifier.py.

This will be much slower and more memory-hungry than the sampled version --
see the warnings printed at load time. Start with a short test run (a couple
epochs) to gauge how long a full training run will actually take on your
machine before committing to 100 epochs.

Usage:
    python3 fly_amq_classifier_fullbrain.py train clips connections_princeton.csv
    python3 fly_amq_classifier_fullbrain.py predict path/to/snippet.mp3 [model.pt]

Requires: torch, numpy, librosa, pandas
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fly_amq_classifier import (
    mel_features_from_waveform,
    audio_to_segments_with_times,
    scan_dataset_folder,
    save_model,
    load_model,
    predict_snippet,
)
from fly_amq_classifier_real import _find_column, PRE_COL_CANDIDATES, POST_COL_CANDIDATES, WEIGHT_COL_CANDIDATES


# -----------------------------------------------------------------------
# 1. LOAD THE WHOLE CONNECTOME (sparse, every neuron in the file)
# -----------------------------------------------------------------------

def load_full_flywire_graph(connections_csv, target_row_scale=1.0):
    print(f"Loading connectome data from {connections_csv} ...")
    df = pd.read_csv(connections_csv)

    pre_col = _find_column(df, PRE_COL_CANDIDATES, "presynaptic neuron ID")
    post_col = _find_column(df, POST_COL_CANDIDATES, "postsynaptic neuron ID")
    try:
        weight_col = _find_column(df, WEIGHT_COL_CANDIDATES, "synapse weight/count")
    except ValueError:
        print("No weight/synapse-count column found -- treating every edge as weight 1.")
        weight_col = None

    print(f"Loaded {len(df):,} raw synaptic connections.")

    all_ids = pd.unique(pd.concat([df[pre_col], df[post_col]]))
    n = len(all_ids)
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}
    print(f"Whole graph: {n:,} neurons, {len(df):,} edges.")

    if n > 50_000:
        print(f"WARNING: {n:,} neurons is large. Training will be slow and memory-heavy "
              f"-- see the module docstring. Consider testing with epochs=2 first to time it.")

    pre_idx = torch.tensor(df[pre_col].map(id_to_idx).to_numpy().copy(), dtype=torch.long)
    post_idx = torch.tensor(df[post_col].map(id_to_idx).to_numpy().copy(), dtype=torch.long)
    weights = torch.tensor(
        df[weight_col].to_numpy(dtype=np.float32).copy() if weight_col else np.ones(len(df), dtype=np.float32),
        dtype=torch.float32,
    )

    # Rescale so each neuron's summed incoming input is ~target_row_scale,
    # same reasoning as the dense versions: keeps tanh from saturating.
    row_sums = torch.zeros(n).scatter_add_(0, post_idx, weights.abs())
    row_sums[row_sums == 0] = 1.0
    weights = weights * (target_row_scale / row_sums[post_idx])

    edge_index = torch.stack([post_idx, pre_idx], dim=0)  # (2, E): adj[post, pre]
    return edge_index, weights, n


# -----------------------------------------------------------------------
# 2. SPARSE CONNECTOME MODEL
# -----------------------------------------------------------------------

class SparseConnectomeRNN(nn.Module):
    """
    train_recurrent=False (default): the connectome's synapse weights are FIXED.
    Only input_proj (audio -> input neurons) and readout (neuron activity ->
    class scores) are trained, so the wiring acts as a fixed "reservoir".

    train_recurrent=True: every synapse weight is trainable too. DON'T use this
    on the whole brain: the backward pass of torch.sparse.mm then builds a
    dense n x n matrix (138,584^2 floats = ~72 GB), which is what caused
    "not enough memory: tried to allocate 76822100224 bytes". It is only
    safe for small graphs (a few thousand neurons).
    """
    def __init__(self, edge_index, edge_weight_init, n_neurons, n_input_channels, n_classes,
                 n_input_neurons=128, dt=1.0, tau=10.0, train_recurrent=False):
        super().__init__()
        self._adj_cache = None
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau = tau
        self.n_input_neurons = n_input_neurons

        self.register_buffer("edge_index", edge_index)     # fixed sparsity pattern
        # synapse strengths; same state_dict key whether frozen or trainable
        self.edge_weight = nn.Parameter(edge_weight_init.clone(), requires_grad=train_recurrent)

        self.input_proj = nn.Linear(n_input_channels, n_input_neurons)
        self.readout = nn.Linear(n_neurons, n_classes)

    def _sparse_adj(self):
        # frozen weights never change, so build the sparse matrix once and reuse it
        if not self.edge_weight.requires_grad:
            if self._adj_cache is None:
                self._adj_cache = torch.sparse_coo_tensor(
                    self.edge_index, self.edge_weight.detach(), size=(self.n_neurons, self.n_neurons)
                ).coalesce()
            return self._adj_cache
        return torch.sparse_coo_tensor(
            self.edge_index, self.edge_weight, size=(self.n_neurons, self.n_neurons)
        ).coalesce()

    def forward(self, audio_features):
        batch, T, _ = audio_features.shape
        device = audio_features.device
        h = torch.zeros(batch, self.n_neurons, device=device)
        drive = self.input_proj(audio_features)
        alpha = self.dt / self.tau

        sparse_adj = self._sparse_adj()

        for t in range(T):
            external = torch.zeros(batch, self.n_neurons, device=device)
            external[:, :self.n_input_neurons] = drive[:, t, :]

            # sparse (n,n) @ dense (n,batch) -> (n,batch); transpose to (batch,n)
            recurrent_input = torch.sparse.mm(sparse_adj, h.T).T
            h = (1 - alpha) * h + alpha * torch.tanh(recurrent_input + external)

        return self.readout(h)


# -----------------------------------------------------------------------
# 3. TRAINING (mirrors train_from_folder, smaller batch size by default)
# -----------------------------------------------------------------------

def train_from_folder_fullbrain(root_dir, connections_csv, n_mels=40, max_frames=200,
                                 epochs=20, val_fraction=0.0, seed=0, segment_seconds=6.0,
                                 sr=16000, hop_length=512, batch_size=4, lr=1e-3, grad_clip=1.0,
                                 augment=True, min_crop_seconds=1.5, full_crop_prob=0.4,
                                 crops_per_song_per_epoch=24, train_recurrent=False):
    """
    Same training scheme as train_from_folder_real in fly_amq_classifier_real.py:
    fresh random crops of every song each epoch (lengths between
    min_crop_seconds and segment_seconds, so the model can recognize a song from
    a couple of seconds), plus augmentation (noise, slight speed change, bleed
    from another song, EQ tilt, frequency/time masking -- see audio_augment.py).

    Set augment=False and min_crop_seconds=segment_seconds to train on plain
    full-length windows only. val_fraction > 0 holds out the last part of every
    song for validation; the default 0.0 trains on whole songs.

    train_recurrent=False keeps the connectome's synapse weights fixed (see
    SparseConnectomeRNN) -- required for the whole brain to fit in memory.
    """
    import librosa
    from audio_augment import build_epoch_data

    dataset, class_names = scan_dataset_folder(root_dir)
    n_classes = len(class_names)

    songs = []
    for path, label in dataset:
        try:
            wave, _ = librosa.load(path, sr=sr, mono=True)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        songs.append((wave, label))

    if not songs:
        raise RuntimeError("No training clips loaded successfully -- check file paths/formats.")
    total_minutes = sum(len(w) for w, _ in songs) / sr / 60
    print(f"Loaded {len(songs)} songs ({total_minutes:.1f} min of audio).")

    def feature_fn(wave):
        return mel_features_from_waveform(
            wave, sr, n_mels=n_mels, hop_length=hop_length, max_frames=max_frames
        )

    rng = np.random.default_rng(seed)
    train_region = (0.0, 1.0 - val_fraction)

    X_val = y_val = None
    if val_fraction > 0:
        seg_len = int(segment_seconds * sr)
        hop = max(seg_len // 2, 1)
        val_feats, val_labels = [], []
        for wave, label in songs:
            held_out = wave[int((1.0 - val_fraction) * len(wave)):]
            for start in range(0, max(len(held_out) - seg_len, 0) + 1, hop):
                chunk = held_out[start:start + seg_len]
                if len(chunk) < sr:
                    continue
                val_feats.append(feature_fn(chunk))
                val_labels.append(label)
        if val_feats:
            X_val = torch.tensor(np.stack(val_feats))
            y_val = torch.tensor(val_labels, dtype=torch.long)
            print(f"Held out the last {int(val_fraction * 100)}% of each song: "
                  f"{len(X_val)} validation windows.")

    print(f"Training on random {min_crop_seconds}-{segment_seconds}s crops, "
          f"{crops_per_song_per_epoch} per song per epoch, "
          f"augmentation {'ON' if augment else 'OFF'}.")

    edge_index, edge_weight, n_neurons = load_full_flywire_graph(connections_csv)
    model = SparseConnectomeRNN(edge_index, edge_weight, n_neurons, n_input_channels=n_mels,
                                n_classes=n_classes, train_recurrent=train_recurrent)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Synapse weights {'TRAINABLE' if train_recurrent else 'FROZEN'}; "
          f"training {sum(p.numel() for p in trainable):,} parameters.")
    optimizer = torch.optim.Adam(trainable, lr=lr)

    for epoch in range(epochs):
        t0 = time.time()

        X_np, y_np = build_epoch_data(
            songs, feature_fn, sr, rng, crops_per_song=crops_per_song_per_epoch,
            min_crop_seconds=min_crop_seconds, max_crop_seconds=segment_seconds,
            full_crop_prob=full_crop_prob, region=train_region, augment=augment,
        )
        X_train, y_train = torch.from_numpy(X_np), torch.from_numpy(y_np)
        n_train = len(X_train)

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
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_val).argmax(-1) == y_val).float().mean().item()
            msg += f"  val_acc {val_acc:.2f}"
        print(f"{msg}  ({time.time() - t0:.1f}s)")

    return model, class_names


# -----------------------------------------------------------------------
# 4. SAVE/LOAD note
# -----------------------------------------------------------------------
# load_model() from fly_amq_classifier.py rebuilds a *dense* ConnectomeRNN,
# which won't work for this sparse model's checkpoint. We reimplement a
# matching save/load pair here instead.

def save_model_sparse(model, class_names, path="fly_amq_model_fullbrain.pt", **hparams):
    torch.save({
        "model_state_dict": model.state_dict(),
        "edge_index": model.edge_index,
        "n_neurons": model.n_neurons,
        "n_input_neurons": model.n_input_neurons,
        "class_names": class_names,
        "hparams": hparams,
    }, path)
    print(f"Saved model to {path}")


def load_model_sparse(path="fly_amq_model_fullbrain.pt"):
    checkpoint = torch.load(path, weights_only=False)
    n_neurons = checkpoint["n_neurons"]
    n_input_neurons = checkpoint["n_input_neurons"]
    hparams = checkpoint["hparams"]
    n_classes = len(checkpoint["class_names"])
    edge_index = checkpoint["edge_index"]

    dummy_weights = torch.zeros(edge_index.shape[1])
    model = SparseConnectomeRNN(
        edge_index, dummy_weights, n_neurons,
        n_input_channels=hparams["n_mels"], n_classes=n_classes, n_input_neurons=n_input_neurons,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {path} ({n_classes} classes, {n_neurons:,} neurons)")
    return model, checkpoint["class_names"], hparams


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        if len(sys.argv) < 3:
            print("Usage: python3 fly_amq_classifier_fullbrain.py predict <snippet_audio_path> [model_path]")
            sys.exit(1)
        snippet_path = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else "fly_amq_model_fullbrain.pt"
        model, class_names, hparams = load_model_sparse(model_path)
        predict_snippet(model, class_names, hparams, snippet_path)

    else:
        if len(sys.argv) < 4 or sys.argv[1] != "train":
            print("Usage: python3 fly_amq_classifier_fullbrain.py train <clips_folder> <connections_csv>")
            sys.exit(1)
        folder = sys.argv[2]
        csv_path = sys.argv[3]

        model, class_names = train_from_folder_fullbrain(root_dir=folder, connections_csv=csv_path)
        save_model_sparse(
            model, class_names, path="fly_amq_model_fullbrain.pt",
            n_mels=40, max_frames=200, segment_seconds=6.0, segment_overlap=0.5,
            sr=16000, hop_length=512,
        )