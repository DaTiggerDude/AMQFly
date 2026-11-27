"""
Same pipeline as fly_amq_classifier.py, but the "brain" wiring comes from the
REAL FlyWire connectome (FAFB v783 connections export) instead of a random
synthetic graph. Everything else -- audio processing, training loop, saving,
inference -- is reused directly from fly_amq_classifier.py so the two stay
in sync and you don't have two copies of the same logic drifting apart.

Class names are restored to the ORIGINAL anime titles (from manifest.json,
written by the download script) instead of the sanitized folder names, so
saved models and predictions show e.g. "Uchitama?! Have You Seen My Tama?"
rather than "Uchitama_! Have You Seen My Tama_".

Usage (quote the clips folder if it contains a space):
    python3 fly_amq_classifier_real.py train "training clips" connections_princeton.csv
    python3 fly_amq_classifier_real.py predict path/to/snippet.mp3 [model.pt]

Requires: torch, numpy, librosa, pandas
    pip install pandas --break-system-packages
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fly_amq_classifier import (
    ConnectomeRNN,
    mel_features_from_waveform,
    audio_to_segments_with_times,
    scan_dataset_folder,
    save_model,
    load_model,
    predict_snippet,
)

# Column name candidates seen across different FlyWire connections exports --
# different releases/tools have used different naming conventions.
PRE_COL_CANDIDATES = ["pre_root_id", "pre_pt_root_id", "pre_id", "pre"]
POST_COL_CANDIDATES = ["post_root_id", "post_pt_root_id", "post_id", "post"]
WEIGHT_COL_CANDIDATES = ["syn_count", "weight", "n_synapses", "count", "cleft_scores"]


def load_folder_to_original(clips_dir):
    """
    Reads manifest.json from the clips folder and returns a dict mapping the
    (sanitized) folder name on disk -> the original animeENName from the JSON.
    Returns an empty dict if there is no manifest, in which case folder names
    are used as class names unchanged.
    """
    path = os.path.join(clips_dir, "manifest.json")
    if not os.path.exists(path):
        print("No manifest.json found -- using folder names as class names.")
        return {}
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    mapping = {m["folder"]: m["animeENName"] for m in manifest.values()}
    print(f"Loaded manifest with {len(manifest)} entries "
          f"covering {len(mapping)} anime folders.")
    return mapping


def _find_column(df, candidates, description):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a {description} column. Looked for {candidates}, "
        f"but the CSV has columns: {list(df.columns)}. Update the "
        f"*_COL_CANDIDATES lists in this file to match."
    )


def sample_connected_neurons(df, pre_col, post_col, n_neurons, seed=0, max_rounds=40):
    """
    Randomly picking neurons independently tends to produce a nearly-empty
    subgraph -- real connectomes are sparse and locally clustered, so two
    random neurons rarely happen to be wired together. Instead, this grows
    the neuron set outward from a random seed by repeatedly pulling in real
    synaptic neighbors (snowball/BFS sampling), guaranteeing the resulting
    subgraph is actually connected enough for signal to propagate through it.
    """
    rng = np.random.default_rng(seed)
    all_ids = pd.unique(pd.concat([df[pre_col], df[post_col]]))

    selected = set()
    frontier = {rng.choice(all_ids)}
    selected |= frontier

    for _ in range(max_rounds):
        if len(selected) >= n_neurons or not frontier:
            break
        edge_mask = df[pre_col].isin(frontier) | df[post_col].isin(frontier)
        touched = pd.unique(pd.concat([df.loc[edge_mask, pre_col], df.loc[edge_mask, post_col]]))
        new_nodes = [n for n in touched if n not in selected]
        rng.shuffle(new_nodes)

        remaining = n_neurons - len(selected)
        new_nodes = new_nodes[:remaining]
        selected.update(new_nodes)
        frontier = set(new_nodes)

    selected = np.array(list(selected))
    print(f"Snowball-sampled {len(selected)} connected neurons "
          f"(target was {n_neurons}; fewer means the neighborhood ran out of new connections).")
    return selected


def load_real_flywire_graph(connections_csv, n_neurons=2000, seed=0, target_row_scale=1.0,
                             n_input_neurons=128):
    """
    Loads a real FlyWire connectivity CSV, extracts a CONNECTED neighborhood
    of n_neurons neurons (via snowball sampling -- see sample_connected_neurons),
    keeps only the edges between neurons in that set, and builds a dense
    adjacency matrix in the same format/scale make_synthetic_fly_graph() would
    have produced -- so it's a drop-in swap for ConnectomeRNN.

    Neurons are additionally ORDERED by their out-degree within the sampled
    subgraph (highest first), so the first n_input_neurons -- which is where
    ConnectomeRNN injects the audio signal -- are the neurons most likely to
    actually have somewhere for that signal to go.
    """
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

    sample_ids = sample_connected_neurons(df, pre_col, post_col, n_neurons, seed=seed)
    id_set = set(sample_ids)
    n = len(sample_ids)

    # keep only edges where BOTH ends are in our sampled neuron set
    mask = df[pre_col].isin(id_set) & df[post_col].isin(id_set)
    sub = df[mask]
    print(f"Kept {len(sub):,} edges between the {n} sampled neurons.")

    if len(sub) == 0:
        raise RuntimeError(
            "No edges survived sampling -- try a larger n_neurons, or this "
            "connections file may use a different ID scheme than expected."
        )

    # Order neurons by out-degree (descending) within this subgraph, so the
    # highest-out-degree neurons end up first -- those become the "input
    # neurons" ConnectomeRNN injects audio signal into.
    out_degree = sub.groupby(pre_col).size()
    degree_order = sorted(sample_ids, key=lambda rid: -out_degree.get(rid, 0))
    id_to_idx = {rid: i for i, rid in enumerate(degree_order)}

    n_zero_out = sum(1 for rid in sample_ids if out_degree.get(rid, 0) == 0)
    print(f"{n_zero_out}/{n} sampled neurons have zero outgoing connections within the subgraph.")
    top_input_out_degrees = [out_degree.get(rid, 0) for rid in degree_order[:n_input_neurons]]
    print(f"Out-degrees of the {n_input_neurons} chosen input neurons: "
          f"min={min(top_input_out_degrees)}, max={max(top_input_out_degrees)}, "
          f"mean={np.mean(top_input_out_degrees):.1f}")

    adj = torch.zeros(n, n)
    pre_idx = sub[pre_col].map(id_to_idx).to_numpy().copy()
    post_idx = sub[post_col].map(id_to_idx).to_numpy().copy()
    weights = (
        sub[weight_col].to_numpy(dtype=np.float32).copy()
        if weight_col else np.ones(len(sub), dtype=np.float32)
    )

    # adj[i, j] = strength of connection FROM neuron j TO neuron i (post, pre)
    adj[post_idx, pre_idx] = torch.tensor(weights, dtype=torch.float32)

    # Same rescaling as the synthetic graph: keep each neuron's summed input
    # near target_row_scale so it doesn't immediately saturate tanh.
    row_sums = adj.abs().sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    adj = adj * (target_row_scale / row_sums)

    return adj, np.array(degree_order)


def train_from_folder_real(root_dir, connections_csv, n_neurons=2000,
                            n_mels=40, max_frames=200, epochs=100, val_fraction=0.0,
                            seed=0, segment_seconds=6.0, segment_overlap=0.5,
                            batch_size=16, lr=1e-3, grad_clip=1.0):
    dataset, class_names = scan_dataset_folder(root_dir)

    # Swap sanitized folder names for the original titles. Order is unchanged,
    # so label index i still refers to the same class; only the display name
    # differs. Folders missing from the manifest keep their folder name.
    folder_to_original = load_folder_to_original(root_dir)
    class_names = [folder_to_original.get(c, c) for c in class_names]
    n_classes = len(class_names)

    def build_temporal_split():
        train_feats, train_labels = [], []
        val_feats, val_labels = [], []
        for path, label in dataset:
            # skip leftovers from interrupted downloads / non-audio files
            if str(path).endswith((".part", ".tmp", ".json")):
                continue
            try:
                segments = audio_to_segments_with_times(
                    path, n_mels=n_mels, max_frames=max_frames,
                    segment_seconds=segment_seconds, segment_overlap=segment_overlap,
                )
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
            for feat, frac_pos in segments:
                if val_fraction > 0 and frac_pos >= (1 - val_fraction):
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
    print(f"Expanded into {len(X_train)} training segments"
          + (f", {len(X_val)} val segments" if val_fraction > 0 else " (using full songs)"))

    if len(X_train) == 0:
        raise RuntimeError("No training clips loaded successfully -- check file paths/formats.")

    adjacency, sample_ids = load_real_flywire_graph(connections_csv, n_neurons=n_neurons, seed=seed)
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
        if val_fraction > 0 and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()
            msg += f"  val_acc {val_acc:.2f}"
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(msg)

    return model, class_names


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        if len(sys.argv) < 3:
            print("Usage: python3 fly_amq_classifier_real.py predict <snippet_audio_path> [model_path]")
            sys.exit(1)
        snippet_path = sys.argv[2]
        model_path = sys.argv[3] if len(sys.argv) > 3 else "fly_amq_model_real.pt"
        model, class_names, hparams = load_model(model_path)
        predict_snippet(model, class_names, hparams, snippet_path)

    else:
        # usage: python3 fly_amq_classifier_real.py train "training clips" connections_princeton.csv
        if len(sys.argv) < 4 or sys.argv[1] != "train":
            print('Usage: python3 fly_amq_classifier_real.py train "<clips_folder>" <connections_csv>')
            sys.exit(1)
        folder = sys.argv[2]
        csv_path = sys.argv[3]

        model, class_names = train_from_folder_real(root_dir=folder, connections_csv=csv_path)
        save_model(
            model, class_names, path="fly_amq_model_real.pt",
            n_mels=40, max_frames=200, segment_seconds=6.0, segment_overlap=0.5,
            sr=16000, hop_length=512,
        )