import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from pyproj import Transformer




data_path = "data/geolife"

data_file = os.path.join(data_path, "geolife.csv")

LNG_MIN = 116.08
LNG_MAX = 116.69
LAT_MIN = 39.66
LAT_MAX = 40.27


def preprocess_trajectories(
    df,
    lat_min,
    lat_max,
    lng_min,
    lng_max,
    min_dur=60,
    max_dur=2880,
    min_timesteps=20,
    max_timesteps=1000,
):
    # Filter spatially and remove missing values.
    df = df.copy()
    df = df[
        df["lat"].between(lat_min, lat_max)
        & df["lng"].between(lng_min, lng_max)
    ]
    df = df.dropna()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.date
    df = df[df["datetime"].dt.year != 2000]

    trajectories = []
    tids = []
    tid = 0
    count = 0

    for _, tr in df.groupby(["uid", "tid"]):
        count += 1
        tr = tr.sort_values("datetime")
        tr.index = pd.to_datetime(tr["datetime"])

        dur = (tr.index.max() - tr.index.min()).total_seconds() / 60

        if min_dur <= dur <= max_dur and min_timesteps <= len(tr) <= max_timesteps:
            tr["tid"] = tid
            trajectories.append(tr[["tid", "lat", "lng", "datetime"]])
            tids.append(tid)
            tid += 1

    print(f"Number of original trajectories: {count}")
    print(f"Number of filtered trajectories: {len(trajectories)}")

    return tids, trajectories, df


def project_trajectories(trajectories, epsg="EPSG:32650"):
    # Project WGS84 coordinates to UTM Zone 50N.
    transformer = Transformer.from_crs(
        "EPSG:4326",
        epsg,
        always_xy=True,
    )

    projected_trajectories = []

    for tr in trajectories:
        tr = tr.copy()
        tr["x"], tr["y"] = transformer.transform(
            tr["lng"].values,
            tr["lat"].values,
        )
        projected_trajectories.append(tr)

    x_min, y_min = transformer.transform(LNG_MIN, LAT_MIN)
    x_max, y_max = transformer.transform(LNG_MAX, LAT_MAX)

    print(f"Projected x range: {x_min:.2f} - {x_max:.2f}")
    print(f"Projected y range: {y_min:.2f} - {y_max:.2f}")

    return projected_trajectories


def input_to_trajmatrix(trajectories):
    # Convert projected trajectories to matrix input format.
    trajmatrix_data = []

    for traj in trajectories:
        if traj.empty:
            continue

        trajmatrix_data.append(
            {
                "tid": traj["tid"].iloc[0],
                "timestamps": traj["datetime"]
                .dt.strftime("%Y-%m-%d %H:%M:%S")
                .tolist(),
                "x": traj["x"].tolist(),
                "y": traj["y"].tolist(),
            }
        )

    return trajmatrix_data


class ProjectedTrajMatrix:

    def __init__(
        self,
        trajectories,
        start_date,
        end_date,
        grid_size=(128, 128),
    ):
        self.trajectories = trajectories
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.grid_size = grid_size
        self.weekdays = list(range(7))

        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32650",
            always_xy=True,
        )

        self._init_grid_bounds()

    def _init_grid_bounds(self):
        self.x_min, self.y_min = self.transformer.transform(
            LNG_MIN,
            LAT_MIN,
        )
        self.x_max, self.y_max = self.transformer.transform(
            LNG_MAX,
            LAT_MAX,
        )

    def _xy_to_grid(self, x, y):
        grid_x = int(
            (x - self.x_min)
            / ((self.x_max - self.x_min) / self.grid_size[0])
        )
        grid_y = int(
            (y - self.y_min)
            / ((self.y_max - self.y_min) / self.grid_size[1])
        )

        grid_x = min(max(grid_x, 0), self.grid_size[0] - 1)
        grid_y = min(max(grid_y, 0), self.grid_size[1] - 1)

        return grid_x, grid_y

    def create_matrices(self):
        print(f"x_min: {self.x_min}, y_min: {self.y_min}")
        print(f"x_max: {self.x_max}, y_max: {self.y_max}")

        matrices = {weekday: {} for weekday in self.weekdays}
        grid_lists = []
        weekday_counts = {weekday: 0 for weekday in self.weekdays}
        date_indices = []
        traj_ids = []

        for traj in self.trajectories:
            timestamps = pd.to_datetime(traj["timestamps"])
            xs = traj["x"]
            ys = traj["y"]
            tid = traj["tid"]
            traj_grid_list = []

            for ts, x, y in zip(timestamps, xs, ys):
                if not (self.start_date <= ts <= self.end_date):
                    continue

                hour = ts.hour
                weekday = ts.weekday()
                date = ts.date()

                grid_x, grid_y = self._xy_to_grid(x, y)
                date_index = (date - self.start_date.date()).days + 1

                if not any(item[1] == date_index for item in date_indices):
                    date_indices.append((date, date_index))

                if date_index not in matrices[weekday]:
                    matrices[weekday][date_index] = np.zeros(
                        (24, *self.grid_size),
                        dtype=np.float32,
                    )
                    weekday_counts[weekday] += 1

                matrices[weekday][date_index][hour, grid_x, grid_y] += 1
                traj_grid_list.append(
                    (date_index, hour, grid_x, grid_y)
                )

            if traj_grid_list:
                grid_lists.append(traj_grid_list)
                traj_ids.append(tid)

        print(f"Counts for each weekday: {weekday_counts}")
        print(f"Number of date indices: {len(date_indices)}")

        return matrices, traj_ids, grid_lists, date_indices


def save_pickle(obj, path):
    # Save an object as a pickle file.
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    print(f"Saved: {path}")


def main():
    # Load and preprocess Geolife.
    df = pd.read_csv(data_file)
    print(f"Geolife CSV shape: {df.shape}")

    tids, trajectories, df = preprocess_trajectories(
        df,
        LAT_MIN,
        LAT_MAX,
        LNG_MIN,
        LNG_MAX,
    )

    print(f"Number of trajectories: {len(trajectories)}")

    os.makedirs(data_path, exist_ok=True)

    save_pickle(
        trajectories,
        os.path.join(data_path, "all_trajectories.pkl"),
    )

    # Select trajectories from 2009.
    sample_traj = [
        tr[["tid", "lat", "lng", "datetime"]]
        for tr in trajectories
        if tr["datetime"].dt.year.iloc[0] != 2007
    ]

    min_date = pd.to_datetime("2009-01-01")
    max_date = pd.to_datetime("2009-03-31")

    filtered_trajectories = []

    for trajectory in sample_traj:
        tdate = pd.to_datetime(trajectory["datetime"].dt.date.iloc[0])

        if min_date <= tdate <= max_date:
            filtered_trajectories.append(trajectory)

    print(
        f"Number of trajectories within date range: "
        f"{len(filtered_trajectories)}"
    )

    excluded_tids = [656, 1194, 2109]

    sample_trajectories = [
        traj
        for traj in filtered_trajectories
        if traj["tid"].iloc[0] not in excluded_tids
    ]

    print(f"Final number of Geolife trajectories: {len(sample_trajectories)}")

    save_pickle(
        sample_trajectories,
        os.path.join(data_path, "sample_trajectories.pkl"),
    )

    # Project trajectories.
    projected_trajectories = project_trajectories(
        sample_trajectories
    )

    save_pickle(
        projected_trajectories,
        os.path.join(data_path, "projected_trajectories.pkl"),
    )

    # Create spatiotemporal matrices.
    trajmatrix_data = input_to_trajmatrix(projected_trajectories)

    traj_matrix = ProjectedTrajMatrix(
        trajmatrix_data,
        min_date,
        max_date,
        grid_size=(128, 128),
    )

    (
        trajmatrices,
        traj_ids,
        traj_grid_lists,
        date_indices,
    ) = traj_matrix.create_matrices()

    # Sort matrices by weekday and date.
    sorted_trajmatrices = {}
    sorted_weekday_date_indices = {}

    for weekday, date_matrices in trajmatrices.items():
        sorted_matrices = dict(sorted(date_matrices.items()))
        sorted_trajmatrices[weekday] = sorted_matrices
        sorted_weekday_date_indices[weekday] = list(sorted_matrices.keys())

    traj_matrices = dict(sorted(sorted_trajmatrices.items()))

    all_traj_matrices = []
    all_date_indices = []

    for weekday, matrices_by_date in traj_matrices.items():
        all_traj_matrices.append(list(matrices_by_date.values()))
        all_date_indices.append(list(matrices_by_date.keys()))

        print(
            f"Weekday {weekday}: "
            f"{len(matrices_by_date)} matrices"
        )

    padded_trajmatrices = tf.keras.preprocessing.sequence.pad_sequences(
        all_traj_matrices,
        padding="post",
        dtype="float32",
    )

    padded_weekday_date_indices = tf.keras.preprocessing.sequence.pad_sequences(
        all_date_indices,
        padding="post",
        dtype="int32",
        value=0,
    )

    print(f"Padded trajectory matrices shape: {padded_trajmatrices.shape}")
    print(
        "Padded weekday-date indices shape: "
        f"{padded_weekday_date_indices.shape}"
    )

    save_pickle(
        traj_ids,
        os.path.join(data_path, "traj_ids.pkl"),
    )

    save_pickle(
        traj_grid_lists,
        os.path.join(data_path, "traj_grid_lists.pkl"),
    )

    save_pickle(
        date_indices,
        os.path.join(data_path, "date_indices.pkl"),
    )

    save_pickle(
        padded_trajmatrices,
        os.path.join(data_path, "stacked_weekday_matrices.pkl"),
    )

    save_pickle(
        padded_weekday_date_indices,
        os.path.join(data_path, "weekday_date_indices.pkl"),
    )

    print("Geolife preprocessing completed.")


if __name__ == "__main__":
    main()
