# STC-Traj2vec

Code for the paper [**"STC-Traj2vec: Spatio-Temporal Context Integrated Trajectory Representation Learning for Clustering"**](https://www.sciencedirect.com/science/article/pii/S095070512600835X).

Two-stage pipeline that learns trajectory embeddings with weekday-hour context, for the **GeoLife** and **Porto** datasets.

```
Stage 1  train_spate.py        SPATE ConvLSTM autoencoder  ->  spate_knn_embeddings_<dataset>.npy
                                                                (168 context vectors: 7 weekdays x 24 hours)
                                         |
Stage 2  context_encoder.py    contextual trajectory encoder (MoCo contrastive)  ->  trajectory embeddings
```

- **Stage 1** trains a ConvLSTM autoencoder on stacked weekday-hour trajectory-density grids with a SPATE (space-time autocorrelation) loss. The 64-d bottleneck of each of the 168 weekday-hour slots is the **context vector**.
- **Stage 2** looks up the context vector for every (weekday, hour) a trajectory visits and feeds it, together with the point-embedded trajectory, into the context auto-encoder.

## Files

| File | Purpose |
|---|---|
| `spate_loss.py` | Spatial/temporal weights, space-time expectations, k-NN local Moran's I, SPATE and `make_spate_knn_loss(k, b)` |
| `spate_autoencoder.py` | `convLSTM_autoencoder` (encoder + autoencoder) |
| `train_spate.py` | Stage 1 CLI: `--dataset geolife\|porto` |
| `context_encoder.py` | Stage 2 CLI: `--dataset geolife\|porto` |

SPATE helper functions are adapted from [spate-gan](https://github.com/konstantinklemmer/spate-gan).

## Setup

```bash
pip install -r requirements.txt

export OUTPUT_ROOT=./processed          # one sub-folder per dataset
# Stage 2 only (raw input paths, used for logging):
export INPUT_GEOLIFE=/data/geolife.csv
export INPUT_PORTO=/data/train.csv
```

## Data layout

```
$OUTPUT_ROOT/
├── geolife/
│   ├── stacked_weekday_matrices13.pkl      # stage 1 input  (7, days, 24, H, W)
│   ├── weekday_date_indices13.pkl          # stage 1 input
│   └── traj_grid_lists.pkl                 # stage 2 input
└── porto/
    ├── padded_trajmatrices.pkl             # stage 1 input
    ├── padded_weekday_date_indices.pkl     # stage 1 input
    └── traj_grid_lists.pkl                 # stage 2 input
```

## Run

```bash
# Stage 1: SPATE context vectors
python train_spate.py --dataset geolife
python train_spate.py --dataset porto

# Stage 2: context encoder
python context_encoder.py --dataset geolife
python context_encoder.py --dataset porto
```

### Stage 1 outputs (`$OUTPUT_ROOT/<dataset>/`)

- `spate_knn_embeddings_<dataset>.npy`: context vectors, shape `(168, 64)`; read by stage 2
- `weekday_date_indices.pkl`: copied under the name stage 2 expects (only if missing)

### Stage 2 inputs

Each dataset folder must contain:

- `traj_grid_lists.pkl`
- `weekday_date_indices.pkl`
- `spate_knn_embeddings_<dataset>.npy`

### Stage 2 outputs (`$OUTPUT_ROOT/<dataset>/output/`)

- `trajectory_embeddings_<dataset>.npy`, `trajectory_ids_<dataset>.pkl`
- `query_encoder_<dataset>.weights.h5`, `training_history_<dataset>.pkl`
- `traj_seq_embed_*`, `traj_context_embeddings_*`, `context_groups_*`, `traj_day_hour_pairs_*`, `empty_trajectories_*`

**Data availability:** Data can be made available on request
