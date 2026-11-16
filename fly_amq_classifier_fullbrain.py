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
    python3 fly_amq_classifier_fullbrain.py train training clips connections_princeton.csv
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
    def __init__(self, edge_index, edge_weight_init, n_neurons, n_input_channels, n_classes,
                 n_input_neurons=128, dt=1.0, tau=10.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.tau = tau
        self.n_input_neurons = n_input_neurons

        self.register_buffer("edge_index", edge_index)     # fixed sparsity pattern
        self.edge_weight = nn.Parameter(edge_weight_init.clone())  # trainable synapse strengths

        self.input_proj = nn.Linear(n_input_channels, n_input_neurons)
        self.readout = nn.Linear(n_neurons, n_classes)

    def _sparse_adj(self):
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
                                 epochs=20, seed=0, segment_seconds=6.0, segment_overlap=0.5,
                                 batch_size=4, lr=1e-3, grad_clip=1.0):
    dataset, class_names = scan_dataset_folder(root_dir)
    n_classes = len(class_names)

    train_feats, train_labels = [], []
    for path, label in dataset:
        try:
            segments = audio_to_segments_with_times(
                path, n_mels=n_mels, max_frames=max_frames,
                segment_seconds=segment_seconds, segment_overlap=segment_overlap,
            )
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        for feat, _frac_pos in segments:
            train_feats.append(feat)
            train_labels.append(label)

    X_train = torch.tensor(np.stack(train_feats))
    y_train = torch.tensor(train_labels, dtype=torch.long)
    print(f"Expanded into {len(X_train)} training segments (using full songs).")

    if len(X_train) == 0:
        raise RuntimeError("No training clips loaded successfully -- check file paths/formats.")

    edge_index, edge_weight, n_neurons = load_full_flywire_graph(connections_csv)
    model = SparseConnectomeRNN(edge_index, edge_weight, n_neurons, n_input_channels=n_mels, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = len(X_train)
    for epoch in range(epochs):
        t0 = time.time()
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
        elapsed = time.time() - t0
        print(f"epoch {epoch:03d}  loss {train_loss:.4f}  train_acc {train_acc:.2f}  ({elapsed:.1f}s)")

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
