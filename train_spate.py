"""Stage 1: train the SPATE ConvLSTM autoencoder and save the context embeddings.

The embeddings (one 64-d vector per weekday-hour slot, 7 x 24 = 168 slots) are
the context vectors consumed by `context_encoder.py` (stage 2).

Usage:
    python train_spate.py --dataset geolife
    python train_spate.py --dataset porto

Paths (override with environment variables):
    OUTPUT_ROOT   root folder holding one sub-folder per dataset (default ./processed)

Expected inputs in $OUTPUT_ROOT/<dataset>/:
    geolife: stacked_weekday_matrices13.pkl, weekday_date_indices13.pkl
    porto:   padded_trajmatrices.pkl,        padded_weekday_date_indices.pkl

Outputs in $OUTPUT_ROOT/<dataset>/:
    spate_knn_embeddings_<dataset>.npy      -> read by context_encoder.py
    weekday_date_indices.pkl                -> read by context_encoder.py (only written if missing)
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from spate_autoencoder import convLSTM_autoencoder
from spate_loss import make_spate_knn_loss, temporal_weights

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "./processed")

# Hyperparameters (same values as the original notebook).
MAX_DAYS = 13            # GeoLife stacked-matrix files are suffixed with this
EMBEDDING_DIM = 64
K_NEIGHBOURS = 512
TEMPORAL_DECAY = 20      # `l` in temporal_weights
SPATE_WEIGHT = 0.8
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1
PATIENCE = 10
MIN_DELTA = 1e-4

DATASETS = {
    "geolife": {
        "matrices": f"stacked_weekday_matrices{MAX_DAYS}.pkl",
        "weekday_indices": f"weekday_date_indices{MAX_DAYS}.pkl",
        "epochs": 210,
    },
    "porto": {
        "matrices": "padded_trajmatrices.pkl",
        "weekday_indices": "padded_weekday_date_indices.pkl",
        "epochs": 200,
    },
}


def find_file(dataset_dir, filename):
    candidates = [dataset_dir / filename, dataset_dir / "316_3M" / filename]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing {filename}. Searched: " + ", ".join(map(str, candidates))
    )


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_input(matrices):
    """(7, days, 24, H, W) -> min-max scaled (168, days, H, W, 1)."""
    # Stack every (weekday, hour) slot along a new first axis -> (7 * 24, days, H, W)
    stacked_data = np.stack(
        [matrices[i, :, j, :, :] for i in range(7) for j in range(24)], axis=0
    )
    print("stacked_data shape:", stacked_data.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    stacked_data_2d = stacked_data.reshape(stacked_data.shape[0], -1)
    scaled_data = scaler.fit_transform(stacked_data_2d).reshape(stacked_data.shape)

    X = tf.expand_dims(scaled_data, axis=-1)  # add channel axis
    print("X shape:", X.shape)
    return X


def run_dataset(dataset, epochs=None, k=None):
    cfg = DATASETS[dataset]
    dataset_dir = Path(OUTPUT_ROOT) / dataset
    epochs = epochs or cfg["epochs"]
    k = k or K_NEIGHBOURS

    print(f"\n========== {dataset.upper()} SPATE AUTOENCODER ==========")
    matrices_path = find_file(dataset_dir, cfg["matrices"])
    weekday_path = find_file(dataset_dir, cfg["weekday_indices"])
    matrices = load_pickle(matrices_path)
    weekday_date_indices = load_pickle(weekday_path)
    print(f"matrices: {matrices_path} {matrices.shape}")

    X = prepare_input(matrices)
    n, t, h, w, c = X.shape
    print(n, t, h, w, c)

    b = temporal_weights(t, l=TEMPORAL_DECAY)
    print(f"temporal weights shape : {b.shape}")

    # A fresh model per dataset (input shape taken from the data).
    encoder, autoencoder = convLSTM_autoencoder(
        input_shape=(t, h, w, c), embedding_dim=EMBEDDING_DIM
    )
    autoencoder.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    autoencoder.compile(
        optimizer=optimizer,
        loss=make_spate_knn_loss(k=k, b=b, spate_weight=SPATE_WEIGHT),
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True,
    )

    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
    )

    spate_knn_embeddings = encoder.predict(X)
    print("embeddings shape:", spate_knn_embeddings.shape)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_path = dataset_dir / f"spate_knn_embeddings_{dataset}.npy"
    np.save(out_path, spate_knn_embeddings)
    print(f"Saved {out_path}")

    # context_encoder.py looks for this file name in the dataset folder.
    canonical_weekday = dataset_dir / "weekday_date_indices.pkl"
    if not canonical_weekday.exists():
        with open(canonical_weekday, "wb") as f:
            pickle.dump(weekday_date_indices, f)
        print(f"Saved {canonical_weekday}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the SPATE ConvLSTM autoencoder for one dataset."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the dataset default (geolife 210, porto 200)")
    parser.add_argument("--k", type=int, default=None,
                        help=f"override the number of neighbours (default {K_NEIGHBOURS})")
    args = parser.parse_args()
    run_dataset(args.dataset, epochs=args.epochs, k=args.k)


if __name__ == "__main__":
    main()
