import os
import ast
import math
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf


porto_dir = "data/porto"

START_DATE = "2014-01-01 00:00:21"
END_DATE = "2014-02-01 00:33:18"
GRID_SIZE = (128, 128)


def save_pickle(obj, path):
    # Save an object as a pickle file.
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


def load_porto_data():
    # Load and clean the Porto dataset.
    train_path = os.path.join(porto_dir, "train.csv")
    taxidf = pd.read_csv(train_path)

    print(f"Original Porto shape: {taxidf.shape}")

    taxidf = taxidf[
        (taxidf["MISSING_DATA"] != True)
        & (taxidf["POLYLINE"] != "[]")
    ]

    porto_taxidf = taxidf[
        ["TRIP_ID", "TIMESTAMP", "POLYLINE"]
    ].copy()

    porto_taxidf["TIMESTAMP"] = pd.to_datetime(
        porto_taxidf["TIMESTAMP"],
        unit="s",
    )

    porto_taxidf["LENGTH"] = porto_taxidf["POLYLINE"].apply(
        lambda x: len(ast.literal_eval(x))
    )

    print(f"Porto trajectories after filtering: {len(porto_taxidf)}")

    return porto_taxidf


def filter_porto_date_range(
    porto_taxidf,
    start_date="2014-01-01 00:00:00",
    end_date="2014-01-31 23:59:59",
):
    # Select the required date range.
    mask = (
        (porto_taxidf["TIMESTAMP"] >= start_date)
        & (porto_taxidf["TIMESTAMP"] <= end_date)
    )

    filtered = porto_taxidf.loc[mask].copy()

    print(f"Number after date filtering: {len(filtered)}")

    return filtered


def parse_polyline(polyline):
    # Parse a Porto polyline string.
    if pd.isna(polyline) or polyline in ("[]", ""):
        return []

    try:
        return ast.literal_eval(polyline)
    except (ValueError, SyntaxError, TypeError):
        return []


def create_trips(filtered_porto_taxidf):
    # Convert Porto polylines into trajectory DataFrames.
    trips = []

    for _, row in filtered_porto_taxidf.iterrows():
        coordinates = parse_polyline(row["POLYLINE"])

        if not coordinates:
            continue

        timestamps = [
            row["TIMESTAMP"] + pd.Timedelta(seconds=15 * i)
            for i in range(len(coordinates))
        ]

        trips.append(
            pd.DataFrame(
                {
                    "TRIP_ID": row["TRIP_ID"],
                    "datetime": timestamps,
                    "lat": [point[1] for point in coordinates],
                    "lon": [point[0] for point in coordinates],
                }
            )
        )

    print(f"Successfully processed {len(trips)} trips.")

    return trips


def resample_trips(
    trips,
    resolution="2min",
    min_timesteps=20,
    max_timesteps=100,
):
    # Resample and filter trajectories.
    resampled_trips = []

    for trip in trips:
        trip = trip.copy()
        trip["datetime"] = pd.to_datetime(trip["datetime"])
        trip = trip.set_index("datetime")

        trip = trip.resample(resolution).last()
        trip = trip.dropna(subset=["lat", "lon"])
        trip["datetime"] = trip.index

        if min_timesteps <= len(trip) <= max_timesteps:
            resampled_trips.append(trip.reset_index(drop=True))

    print(f"Trips after resampling and filtering: {len(resampled_trips)}")

    return resampled_trips


def project_trajectories(trips):
    # Project WGS84 coordinates to Web Mercator.
    earth_radius = 6378137.0
    origin_shift = math.pi * earth_radius

    projected_trips = []

    for trip in trips:
        trip = trip.copy()

        lon = trip["lon"].to_numpy()
        lat = np.clip(trip["lat"].to_numpy(), -85.05112878, 85.05112878)

        trip["x"] = lon * origin_shift / 180.0
        trip["y"] = (
            np.log(
                np.tan(
                    (90.0 + lat) * math.pi / 360.0
                )
            )
            / (math.pi / 180.0)
            * origin_shift
            / 180.0
        )

        projected_trips.append(trip)

    print(f"Projected trajectories: {len(projected_trips)}")

    return projected_trips


def get_projection_bounds(projected_trips):
    # Calculate global projected coordinate bounds.
    all_x = np.concatenate(
        [trip["x"].to_numpy() for trip in projected_trips]
    )
    all_y = np.concatenate(
        [trip["y"].to_numpy() for trip in projected_trips]
    )

    return {
        "x_range": (all_x.min(), all_x.max()),
        "y_range": (all_y.min(), all_y.max()),
    }


def input_to_trajmatrix(projected_trips):
    # Convert projected trajectories to matrix input format.
    trajmatrix_data = []

    for trip in projected_trips:
        if trip.empty:
            continue

        trajmatrix_data.append(
            {
                "tid": trip["TRIP_ID"].iloc[0],
                "datetime": pd.to_datetime(trip["datetime"])
                .dt.strftime("%Y-%m-%d %H:%M:%S")
                .tolist(),
                "lat": trip["lat"].tolist(),
                "lon": trip["lon"].tolist(),
                "x": trip["x"].tolist(),
                "y": trip["y"].tolist(),
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
        projected_bounds=None,
    ):
        self.trajectories = trajectories
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.grid_size = grid_size
        self.weekdays = list(range(7))

        if projected_bounds is None:
            self._init_grid_bounds()
        else:
            self.x_min, self.x_max = projected_bounds["x_range"]
            self.y_min, self.y_max = projected_bounds["y_range"]

    def _init_grid_bounds(self):
        # Calculate grid bounds from projected trajectories.
        all_x = np.concatenate(
            [np.asarray(traj["x"]) for traj in self.trajectories]
        )
        all_y = np.concatenate(
            [np.asarray(traj["y"]) for traj in self.trajectories]
        )

        self.x_min = all_x.min()
        self.x_max = all_x.max()
        self.y_min = all_y.min()
        self.y_max = all_y.max()

    def _xy_to_grid(self, x, y):
        # Map projected coordinates to grid cells.
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
        # Create weekday-date-hour spatial matrices.
        matrices = {
            weekday: {}
            for weekday in self.weekdays
        }

        grid_lists = []
        traj_ids = []
        weekday_counts = {
            weekday: 0
            for weekday in self.weekdays
        }

        unique_dates = set()

        for traj in self.trajectories:
            timestamps = pd.to_datetime(traj["datetime"])

            for ts in timestamps:
                if self.start_date <= ts <= self.end_date:
                    unique_dates.add(ts.date())

        sorted_dates = sorted(unique_dates)
        date_to_index = {
            date: index + 1
            for index, date in enumerate(sorted_dates)
        }

        for traj in self.trajectories:
            timestamps = pd.to_datetime(traj["datetime"])
            xs = traj["x"]
            ys = traj["y"]
            tid = traj["tid"]

            traj_grid_list = []
            weekday_visited = set()

            for ts, x, y in zip(timestamps, xs, ys):
                if not (self.start_date <= ts <= self.end_date):
                    continue

                date = ts.date()
                hour = ts.hour
                weekday = ts.weekday()
                date_index = date_to_index[date]

                grid_x, grid_y = self._xy_to_grid(x, y)

                if date_index not in matrices[weekday]:
                    matrices[weekday][date_index] = np.zeros(
                        (24, *self.grid_size),
                        dtype=np.float32,
                    )

                matrices[weekday][date_index][
                    hour, grid_x, grid_y
                ] += 1

                traj_grid_list.append(
                    (date_index, hour, grid_x, grid_y)
                )
                weekday_visited.add(weekday)

            if traj_grid_list:
                grid_lists.append(traj_grid_list)
                traj_ids.append(tid)

                for weekday in weekday_visited:
                    weekday_counts[weekday] += 1

        print(f"Trajectory coverage by weekday: {weekday_counts}")
        print(f"Unique dates: {len(sorted_dates)}")

        return matrices, traj_ids, grid_lists, list(date_to_index.values())


def main():
    # Run the Porto preprocessing pipeline.
    os.makedirs(porto_dir, exist_ok=True)

    porto_taxidf = load_porto_data()

    filtered_porto_taxidf = filter_porto_date_range(
        porto_taxidf
    )

    trips = create_trips(
        filtered_porto_taxidf
    )

    save_pickle(
        trips,
        os.path.join(porto_dir, "trips_list_jan2014.pkl"),
    )

    trips = resample_trips(trips)

    save_pickle(
        trips,
        os.path.join(porto_dir, "trips_jan2014.pkl"),
    )

    projected_trips = project_trajectories(trips)

    save_pickle(
        projected_trips,
        os.path.join(porto_dir, "projected_trajectories.pkl"),
    )

    projection_bounds = get_projection_bounds(
        projected_trips
    )

    print(
        f"X range: {projection_bounds['x_range'][0]:.2f} - "
        f"{projection_bounds['x_range'][1]:.2f} m"
    )
    print(
        f"Y range: {projection_bounds['y_range'][0]:.2f} - "
        f"{projection_bounds['y_range'][1]:.2f} m"
    )

    trajmatrix_data = input_to_trajmatrix(
        projected_trips
    )

    traj_matrix = ProjectedTrajMatrix(
        trajmatrix_data,
        START_DATE,
        END_DATE,
        grid_size=GRID_SIZE,
        projected_bounds=projection_bounds,
    )

    (
        trajmatrices,
        traj_ids,
        traj_grid_lists,
        date_indices,
    ) = traj_matrix.create_matrices()

    sorted_trajmatrices = {}

    for weekday, date_matrices in trajmatrices.items():
        sorted_trajmatrices[weekday] = dict(
            sorted(date_matrices.items())
        )

    all_traj_matrices = []
    all_date_indices = []

    for weekday, matrices_by_date in sorted_trajmatrices.items():
        if not matrices_by_date:
            continue

        all_traj_matrices.append(
            list(matrices_by_date.values())
        )
        all_date_indices.append(
            list(matrices_by_date.keys())
        )

        print(
            f"Weekday {weekday}: "
            f"{len(matrices_by_date)} matrices"
        )

    padded_trajmatrices = tf.keras.preprocessing.sequence.pad_sequences(
        all_traj_matrices,
        padding="post",
        dtype="float32",
    )

    padded_weekday_date_indices = (
        tf.keras.preprocessing.sequence.pad_sequences(
            all_date_indices,
            padding="post",
            dtype="int32",
            value=0,
        )
    )

    print(f"Trajectory matrices shape: {padded_trajmatrices.shape}")
    print(
        "Weekday-date indices shape: "
        f"{padded_weekday_date_indices.shape}"
    )

    save_pickle(
        traj_ids,
        os.path.join(porto_dir, "traj_ids.pkl"),
    )

    save_pickle(
        traj_grid_lists,
        os.path.join(porto_dir, "traj_grid_lists.pkl"),
    )

    save_pickle(
        date_indices,
        os.path.join(porto_dir, "date_indices.pkl"),
    )

    save_pickle(
        padded_trajmatrices,
        os.path.join(porto_dir, "padded_trajmatrices.pkl"),
    )

    save_pickle(
        padded_weekday_date_indices,
        os.path.join(
            porto_dir,
            "padded_weekday_date_indices.pkl",
        ),
    )

    print("Porto preprocessing completed.")


if __name__ == "__main__":
    main()
