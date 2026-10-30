# STC-Traj2vec
code for the paper "STC-Traj2vec:Spatio-Temporal Context Integrated Trajectory Representation Learning for Clustering" (https://www.sciencedirect.com/science/article/pii/S095070512600835X)




## Requirements
Each dataset folder (`$OUTPUT_ROOT/geolife/`, `$OUTPUT_ROOT/porto/`) must contain:
- `traj_grid_lists.pkl`
- `weekday_date_indices.pkl`
- `spate_knn_embeddings_<dataset>.npy`

## Run
```bash
export INPUT_GEOLIFE=/data/geolife.csv
export INPUT_PORTO=/data/train.csv
export OUTPUT_ROOT=./processed

python context_encoder.py --dataset geolife
python context_encoder.py --dataset porto
```

Results are saved to `$OUTPUT_ROOT/<dataset>/output/`.
