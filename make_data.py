#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized STC-Traj2vec data preparation pipeline.

Reference behavior:
    - make_geolife_data (1).ipynb
    - make_porto_data.ipynb

The pipeline standardizes both datasets to:
    tid | datetime | lat | lon | x | y

and preserves the STC-Traj2vec artifacts:
    traj_ids.pkl
    traj_grid_lists.pkl
    date_indices.pkl
    padded_trajmatrices.pkl
    padded_weekday_date_indices.pkl

Examples:
    python make_data.py --dataset geolife \
        --input /path/to/geolife.csv \
        --output /path/to/processed/geolife

    python make_data.py --dataset porto \
        --input /path/to/train.csv \
        --output /path/to/processed/porto
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str

    # Raw input columns
    trajectory_id_col: str
    user_id_col: Optional[str]
    timestamp_col: str
    lat_col: str
    lon_col: str

    # Porto-specific input handling
    polyline_col: Optional[str] = None
    missing_data_col: Optional[str] = None
    unix_timestamp: bool = False
    polyline_point_interval_seconds: int = 15

    # Filtering
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    # (lat_min, lat_max, lon_min, lon_max)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Date range used by the context-matrix stage.
    matrix_start_date: Optional[str] = None
    matrix_end_date: Optional[str] = None

    min_duration_min: Optional[float] = None
    max_duration_min: Optional[float] = None
    min_timesteps: Optional[int] = None
    max_timesteps: Optional[int] = None

    # Temporal processing
    resample: Optional[str] = None

    # Projection/grid
    projection: str = "epsg"
    target_crs: Optional[str] = None
    grid_size: Tuple[int, int] = (128, 128)
    bounds_mode: str = "trajectory"       # geographic | trajectory

    # Date-index convention used by the original notebooks:
    # GeoLife -> elapsed days from matrix start + 1
    # Porto   -> consecutive unique observed dates + 1
    date_index_mode: str = "unique"

    # GeoLife manually excluded these IDs after selecting Jan-Mar 2009.
    exclude_ids: Tuple[int, ...] = ()


GEOLIFE_CONFIG = DatasetConfig(
    name="geolife",
    trajectory_id_col="tid",
    user_id_col="uid",
    timestamp_col="datetime",
    lat_col="lat",
    lon_col="lng",

    spatial_bounds=(39.66, 40.27, 116.08, 116.69),
    start_date="2009-01-01",
    end_date="2009-03-31 23:59:59",
    matrix_start_date="2009-01-01",
    matrix_end_date="2009-03-31 23:59:59",

    min_duration_min=60,
    max_duration_min=2880,
    min_timesteps=20,
    max_timesteps=1000,

    # The notebook calls tr.resample('5min') but does not assign the result,
    # so the effective notebook behavior is no resampling.
    resample=None,

    projection="epsg",
    target_crs="EPSG:32650",
    grid_size=(128, 128),
    bounds_mode="geographic",
    date_index_mode="elapsed",
    exclude_ids=(656, 1194, 2109),
)


PORTO_CONFIG = DatasetConfig(
    name="porto",
    trajectory_id_col="TRIP_ID",
    user_id_col=None,
    timestamp_col="TIMESTAMP",
    lat_col="lat",
    lon_col="lon",

    polyline_col="POLYLINE",
    missing_data_col="MISSING_DATA",
    unix_timestamp=True,
    polyline_point_interval_seconds=15,

    # The notebook first selects trips starting in Jan 2014.
    start_date="2014-01-01 00:00:00",
    end_date="2014-01-31 23:59:59",

    # The resampled trajectory can extend into Feb 1, so the matrix stage
    # uses the notebook's explicit matrix end.
    matrix_start_date="2014-01-01 00:00:21",
    matrix_end_date="2014-02-01 00:33:18",

    min_duration_min=None,
    max_duration_min=None,
    min_timesteps=20,
    max_timesteps=100,

    resample="2min",

    projection="mercator",
    target_crs="EPSG:3857",
    grid_size=(128, 128),
    bounds_mode="trajectory",
    date_index_mode="unique",
)


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# GeoLife
# ---------------------------------------------------------------------------

def geolife_clean_plt(
    root: Path,
    user_id: str,
    input_filepath: str,
    traj_id: int,
) -> List[List[str]]:
    """Read one GeoLife .plt file, skipping the six-line header."""
    import csv

    path = root / user_id / "Trajectory" / input_filepath
    with open(path, "rt") as fin:
        rows = [list(row) for row in csv.reader(fin)][6:]

    for row in rows:
        row.insert(0, traj_id)
        row.insert(0, user_id)

    return rows


def geolife_raw_to_df(raw_dir: Path) -> pd.DataFrame:
    """
    Convert the raw GeoLife directory into the CSV-style representation
    used by the notebook.
    """
    data_root = raw_dir
    if (raw_dir / "Geolife Trajectories 1.3" / "Data").exists():
        data_root = raw_dir / "Geolife Trajectories 1.3" / "Data"

    col_names = [
        "uid", "tid", "lat", "lng", "-", "Alt", "dayNo", "date", "time"
    ]
    data: List[List[str]] = []

    user_dirs = sorted(
        [p for p in data_root.iterdir()
         if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )

    for user_dir in user_dirs:
        trajectory_dir = user_dir / "Trajectory"
        if not trajectory_dir.exists():
            continue

        files = sorted(
            p.name for p in trajectory_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".plt"
        )

        for traj_id, filename in enumerate(files):
            data.extend(
                geolife_clean_plt(
                    data_root, user_dir.name, filename, traj_id
                )
            )

    df = pd.DataFrame(data, columns=col_names)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

    # Exact timezone conversion used in the GeoLife notebook.
    df["datetime"] = (
        df["datetime"]
        .dt.tz_localize("GMT")
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )

    return df.drop(columns=["date", "time"])


def load_geolife(path: Path) -> pd.DataFrame:
    if path.is_dir():
        df = geolife_raw_to_df(path)
    else:
        df = load_csv(path)

    required = {"uid", "tid", "lat", "lng", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"GeoLife input is missing columns: {sorted(missing)}"
        )

    return df


def standardize_geolife(df: pd.DataFrame) -> List[pd.DataFrame]:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lon"])

    # Notebook explicitly removes year 2000.
    df = df[df["datetime"].dt.year != 2000]

    trajectories = []
    for (uid, tid), tr in df.groupby(["uid", "tid"], sort=False):
        tr = tr.sort_values("datetime").copy()

        # Make IDs unique across users while retaining the original tid.
        tr["tid"] = str(uid) + "_" + tr["tid"].astype(str)

        trajectories.append(
            tr[["tid", "lat", "lon", "datetime"]].reset_index(drop=True)
        )

    return trajectories


# ---------------------------------------------------------------------------
# Porto
# ---------------------------------------------------------------------------

def extrapolate_timestamps(
    start_timestamp: Any,
    length: int,
    resolution_seconds: int = 15,
) -> List[pd.Timestamp]:
    """Reproduce the Porto notebook's 15-second timestamp extrapolation."""
    start_dt = pd.to_datetime(start_timestamp)
    return [
        start_dt + pd.Timedelta(seconds=i * resolution_seconds)
        for i in range(length)
    ]


def parse_polyline(polyline_str: Any) -> List[List[float]]:
    """Safely parse a Porto POLYLINE string."""
    if pd.isna(polyline_str) or polyline_str in ("[]", ""):
        return []

    try:
        value = ast.literal_eval(polyline_str)
    except (ValueError, SyntaxError, TypeError):
        return []

    return value if isinstance(value, list) else []


def load_porto(path: Path, cfg: DatasetConfig) -> pd.DataFrame:
    """
    Reproduce the Porto notebook:
      MISSING_DATA != True
      POLYLINE != []
      Unix timestamp -> datetime
      POLYLINE -> timestamped lat/lon points
    """
    df = load_csv(path)

    required = {"TRIP_ID", "TIMESTAMP", "POLYLINE", "MISSING_DATA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Porto input is missing columns: {sorted(missing)}"
        )

    df = df[df["MISSING_DATA"] != True].copy()
    df = df[df["POLYLINE"] != "[]"].copy()

    base = pd.DataFrame()
    base["TRIP_ID"] = df["TRIP_ID"]
    base["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"].apply(
            lambda x: pd.Timestamp.fromtimestamp(float(x))
        )
    )
    base["POLYLINE"] = df["POLYLINE"]

    # Notebook filters the original trip start timestamp to Jan 2014.
    if cfg.start_date:
        base = base[base["TIMESTAMP"] >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        base = base[base["TIMESTAMP"] <= pd.Timestamp(cfg.end_date)]

    records: List[Dict[str, Any]] = []

    for _, row in base.iterrows():
        coordinates = parse_polyline(row["POLYLINE"])
        if not coordinates:
            continue

        timestamps = extrapolate_timestamps(
            row["TIMESTAMP"],
            len(coordinates),
            cfg.polyline_point_interval_seconds,
        )

        for ts, coordinate in zip(timestamps, coordinates):
            if len(coordinate) < 2:
                continue

            # Porto POLYLINE stores [lon, lat].
            lon, lat = float(coordinate[0]), float(coordinate[1])

            records.append({
                "tid": row["TRIP_ID"],
                "datetime": ts,
                "lat": lat,
                "lon": lon,
            })

    if not records:
        raise ValueError("No Porto trajectory points were produced.")

    return pd.DataFrame(records)


def standardize_porto(df: pd.DataFrame) -> List[pd.DataFrame]:
    trajectories = []

    for tid, tr in df.groupby("tid", sort=False):
        tr = tr.sort_values("datetime").copy()
        trajectories.append(
            tr[["tid", "lat", "lon", "datetime"]].reset_index(drop=True)
        )

    return trajectories


# ---------------------------------------------------------------------------
# Common preprocessing
# ---------------------------------------------------------------------------

def filter_spatial(
    trajectories: List[pd.DataFrame],
    bounds: Optional[Tuple[float, float, float, float]],
) -> List[pd.DataFrame]:
    if bounds is None:
        return trajectories

    lat_min, lat_max, lon_min, lon_max = bounds
    output = []

    for tr in trajectories:
        mask = (
            tr["lat"].between(lat_min, lat_max, inclusive="both")
            & tr["lon"].between(lon_min, lon_max, inclusive="both")
        )
        tr2 = tr.loc[mask].copy()

        if not tr2.empty:
            output.append(tr2.reset_index(drop=True))

    return output


def resample_trajectory(
    tr: pd.DataFrame,
    resolution: Optional[str],
) -> pd.DataFrame:
    """
    Porto notebook behavior:
        index = datetime
        resample('2min').last()
        drop missing lat/lon
        datetime = index

    GeoLife notebook's resample call was not assigned, so its effective
    behavior is preserved by resolution=None.
    """
    if resolution is None:
        return tr.sort_values("datetime").reset_index(drop=True)

    x = tr.sort_values("datetime").copy()
    x.index = pd.to_datetime(x["datetime"])

    x = x.resample(resolution).last()
    x = x.dropna(subset=["lat", "lon"])
    x["datetime"] = x.index

    return x.reset_index(drop=True)


def filter_trajectories(
    trajectories: List[pd.DataFrame],
    cfg: DatasetConfig,
) -> List[pd.DataFrame]:
    output = []

    for tr in trajectories:
        if tr.empty:
            continue

        tr = tr.sort_values("datetime").copy()
        first_dt = pd.to_datetime(tr["datetime"].iloc[0])

        # The notebooks select trajectories by their start date.
        if cfg.start_date and first_dt < pd.Timestamp(cfg.start_date):
            continue
        if cfg.end_date and first_dt > pd.Timestamp(cfg.end_date):
            continue

        duration_min = (
            tr["datetime"].max() - tr["datetime"].min()
        ).total_seconds() / 60.0

        if (
            cfg.min_duration_min is not None
            and duration_min < cfg.min_duration_min
        ):
            continue

        if (
            cfg.max_duration_min is not None
            and duration_min > cfg.max_duration_min
        ):
            continue

        if (
            cfg.min_timesteps is not None
            and len(tr) < cfg.min_timesteps
        ):
            continue

        if (
            cfg.max_timesteps is not None
            and len(tr) > cfg.max_timesteps
        ):
            continue

        output.append(tr.reset_index(drop=True))

    return output


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class WebMercatorProjection:
    """Web Mercator EPSG:3857 implementation used by the Porto notebook."""

    earth_radius = 6378137.0
    origin_shift = math.pi * earth_radius

    @classmethod
    def lat_lon_to_mercator(
        cls,
        lat: float,
        lon: float,
    ) -> Tuple[float, float]:
        x = lon * cls.origin_shift / 180.0

        # Keep the projection numerically stable.
        lat = max(min(float(lat), 89.5), -89.5)

        y = math.log(
            math.tan((90.0 + lat) * math.pi / 360.0)
        ) / (math.pi / 180.0)
        y *= cls.origin_shift / 180.0

        return x, y


def project_trajectories(
    trajectories: List[pd.DataFrame],
    cfg: DatasetConfig,
) -> List[pd.DataFrame]:
    """Add projected x/y coordinates."""
    output = []

    if cfg.projection == "mercator":
        for tr in trajectories:
            tr2 = tr.copy()
            xy = [
                WebMercatorProjection.lat_lon_to_mercator(
                    float(lat), float(lon)
                )
                for lat, lon in zip(tr2["lat"], tr2["lon"])
            ]

            tr2["x"] = [p[0] for p in xy]
            tr2["y"] = [p[1] for p in xy]
            output.append(tr2)

        return output

    if cfg.projection == "epsg":
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise ImportError(
                "GeoLife projection requires pyproj. "
                "Install with: pip install pyproj"
            ) from exc

        transformer = Transformer.from_crs(
            "EPSG:4326",
            cfg.target_crs,
            always_xy=True,
        )

        for tr in trajectories:
            tr2 = tr.copy()

            x, y = transformer.transform(
                tr2["lon"].to_numpy(),
                tr2["lat"].to_numpy(),
            )

            tr2["x"] = np.asarray(x)
            tr2["y"] = np.asarray(y)
            output.append(tr2)

        return output

    raise ValueError(f"Unsupported projection: {cfg.projection}")


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def geographic_bounds_to_projected(
    cfg: DatasetConfig,
) -> Dict[str, Tuple[float, float]]:
    """Project a WGS84 bounding box into the selected coordinate system."""
    if cfg.spatial_bounds is None:
        raise ValueError(
            "spatial_bounds is required for bounds_mode='geographic'."
        )

    lat_min, lat_max, lon_min, lon_max = cfg.spatial_bounds

    if cfg.projection == "epsg":
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            cfg.target_crs,
            always_xy=True,
        )
        x0, y0 = transformer.transform(lon_min, lat_min)
        x1, y1 = transformer.transform(lon_max, lat_max)

    elif cfg.projection == "mercator":
        x0, y0 = WebMercatorProjection.lat_lon_to_mercator(
            lat_min, lon_min
        )
        x1, y1 = WebMercatorProjection.lat_lon_to_mercator(
            lat_max, lon_max
        )

    else:
        raise ValueError(f"Unsupported projection: {cfg.projection}")

    return {
        "x_range": (min(x0, x1), max(x0, x1)),
        "y_range": (min(y0, y1), max(y0, y1)),
    }


def trajectory_projected_bounds(
    trajectories: List[pd.DataFrame],
) -> Dict[str, Tuple[float, float]]:
    all_x = np.concatenate(
        [tr["x"].to_numpy(dtype=float) for tr in trajectories]
    )
    all_y = np.concatenate(
        [tr["y"].to_numpy(dtype=float) for tr in trajectories]
    )

    return {
        "x_range": (float(all_x.min()), float(all_x.max())),
        "y_range": (float(all_y.min()), float(all_y.max())),
    }


class ProjectedTrajMatrix:
    """
    Common implementation of the matrix-building stage.

    STC-Traj2vec-compatible trajectory representation:
        (date_index, hour, grid_x, grid_y)

    Context representation:
        one matrix per weekday/date:
            (24, H, W)
    """

    def __init__(
        self,
        trajectories: List[pd.DataFrame],
        start_date: str,
        end_date: str,
        grid_size: Tuple[int, int] = (128, 128),
        projected_bounds: Optional[
            Dict[str, Tuple[float, float]]
        ] = None,
        date_index_mode: str = "unique",
    ):
        self.trajectories = trajectories
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.grid_size = tuple(grid_size)
        self.date_index_mode = date_index_mode
        self.weekdays = list(range(7))

        if projected_bounds is not None:
            self.x_min, self.x_max = projected_bounds["x_range"]
            self.y_min, self.y_max = projected_bounds["y_range"]
        else:
            self._init_grid_bounds_from_trajectories()

    def _init_grid_bounds_from_trajectories(self) -> None:
        all_x = np.concatenate(
            [tr["x"].to_numpy(dtype=float) for tr in self.trajectories]
        )
        all_y = np.concatenate(
            [tr["y"].to_numpy(dtype=float) for tr in self.trajectories]
        )

        self.x_min = float(all_x.min())
        self.x_max = float(all_x.max())
        self.y_min = float(all_y.min())
        self.y_max = float(all_y.max())

    def _xy_to_grid(
        self,
        x: float,
        y: float,
    ) -> Tuple[int, int]:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Invalid projected grid bounds.")

        grid_x = int(
            (float(x) - self.x_min)
            / ((self.x_max - self.x_min) / self.grid_size[0])
        )
        grid_y = int(
            (float(y) - self.y_min)
            / ((self.y_max - self.y_min) / self.grid_size[1])
        )

        grid_x = min(
            max(grid_x, 0),
            self.grid_size[0] - 1,
        )
        grid_y = min(
            max(grid_y, 0),
            self.grid_size[1] - 1,
        )

        return grid_x, grid_y

    def create_matrices(self):
        print(
            f"x_min: {self.x_min}, y_min: {self.y_min}\n"
            f"x_max: {self.x_max}, y_max: {self.y_max}"
        )

        # Collect unique dates represented in the matrix range.
        unique_dates = set()

        for tr in self.trajectories:
            ts = pd.to_datetime(tr["datetime"])
            mask = (
                (ts >= self.start_date)
                & (ts <= self.end_date)
            )
            unique_dates.update(
                ts.loc[mask].dt.date.tolist()
            )

        sorted_dates = sorted(unique_dates)

        if not sorted_dates:
            raise ValueError(
                "No trajectory points fall inside the matrix date range."
            )

        if self.date_index_mode == "elapsed":
            date_to_index = {
                date: (date - self.start_date.date()).days + 1
                for date in sorted_dates
            }
        elif self.date_index_mode == "unique":
            date_to_index = {
                date: idx + 1
                for idx, date in enumerate(sorted_dates)
            }
        else:
            raise ValueError(
                "date_index_mode must be 'elapsed' or 'unique'."
            )

        matrices: Dict[int, Dict[int, np.ndarray]] = {
            wd: {} for wd in self.weekdays
        }

        traj_ids = []
        traj_grid_lists = []
        weekday_counts = {wd: 0 for wd in self.weekdays}

        for tr in self.trajectories:
            tid = tr["tid"].iloc[0]
            timestamps = pd.to_datetime(tr["datetime"])

            grid_list = []
            weekday_visited = set()

            for ts, x, y in zip(
                timestamps,
                tr["x"].to_numpy(),
                tr["y"].to_numpy(),
            ):
                if ts < self.start_date or ts > self.end_date:
                    continue

                date = ts.date()
                date_index = date_to_index[date]
                hour = int(ts.hour)
                weekday = int(ts.weekday())

                grid_x, grid_y = self._xy_to_grid(x, y)

                if date_index not in matrices[weekday]:
                    matrices[weekday][date_index] = np.zeros(
                        (24, *self.grid_size),
                        dtype=np.float32,
                    )

                matrices[weekday][date_index][
                    hour, grid_x, grid_y
                ] += 1.0

                grid_list.append(
                    (date_index, hour, grid_x, grid_y)
                )
                weekday_visited.add(weekday)

            if grid_list:
                traj_ids.append(tid)
                traj_grid_lists.append(grid_list)

                for wd in weekday_visited:
                    weekday_counts[wd] += 1

        print("\n=== Trajectory Coverage by Weekday ===")
        for wd in self.weekdays:
            print(
                f"weekday {wd}: "
                f"{weekday_counts[wd]} trajectories"
            )

        print(f"\nUnique dates: {len(sorted_dates)}")
        print(
            f"Date range: "
            f"{min(sorted_dates)} to {max(sorted_dates)}"
        )

        return (
            matrices,
            traj_ids,
            traj_grid_lists,
            list(range(1, len(sorted_dates) + 1)),
        )


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad_weekday_matrices(
    matrices: Dict[int, Dict[int, np.ndarray]],
    grid_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy equivalent of the notebook's tf.keras.pad_sequences stage.

    Returns:
        padded_trajmatrices:
            (7, max_days, 24, H, W)

        padded_weekday_date_indices:
            (7, max_days), padded with zero.
    """
    h, w = grid_size

    max_days = max(
        [len(matrices.get(wd, {})) for wd in range(7)]
        or [0]
    )

    padded_trajmatrices = np.zeros(
        (7, max_days, 24, h, w),
        dtype=np.float32,
    )

    padded_weekday_date_indices = np.zeros(
        (7, max_days),
        dtype=np.int32,
    )

    for wd in range(7):
        items = sorted(
            matrices.get(wd, {}).items()
        )

        for j, (date_index, matrix) in enumerate(items):
            padded_trajmatrices[wd, j] = matrix
            padded_weekday_date_indices[
                wd, j
            ] = int(date_index)

    return (
        padded_trajmatrices,
        padded_weekday_date_indices,
    )


# ---------------------------------------------------------------------------
# Statistics / saving
# ---------------------------------------------------------------------------

def trajectory_statistics(
    trajectories: List[pd.DataFrame],
) -> Dict[str, Any]:
    if not trajectories:
        return {
            "num_trajectories": 0,
            "avg_timesteps": 0,
            "min_timesteps": 0,
            "max_timesteps": 0,
            "avg_duration_minutes": 0,
            "min_duration_minutes": 0,
            "max_duration_minutes": 0,
        }

    lengths = np.asarray(
        [len(t) for t in trajectories],
        dtype=int,
    )

    durations = np.asarray(
        [
            (
                t["datetime"].max()
                - t["datetime"].min()
            ).total_seconds() / 60.0
            for t in trajectories
        ],
        dtype=float,
    )

    return {
        "num_trajectories": int(len(trajectories)),
        "avg_timesteps": float(lengths.mean()),
        "min_timesteps": int(lengths.min()),
        "max_timesteps": int(lengths.max()),
        "avg_duration_minutes": float(durations.mean()),
        "min_duration_minutes": float(durations.min()),
        "max_duration_minutes": float(durations.max()),
    }


def save_outputs(
    output_dir: Path,
    all_trajectories: List[pd.DataFrame],
    filtered_trajectories: List[pd.DataFrame],
    projected_trajectories: List[pd.DataFrame],
    traj_ids: List[Any],
    traj_grid_lists: List[List[Tuple[int, int, int, int]]],
    date_indices: List[int],
    padded_trajmatrices: np.ndarray,
    padded_weekday_date_indices: np.ndarray,
    config: DatasetConfig,
) -> None:
    ensure_dir(output_dir)

    # Useful intermediate files.
    save_pickle(
        all_trajectories,
        output_dir / "all_trajectories.pkl",
    )
    save_pickle(
        filtered_trajectories,
        output_dir / "filtered_trajectories.pkl",
    )
    save_pickle(
        filtered_trajectories,
        output_dir / "sample_trajectories.pkl",
    )
    save_pickle(
        projected_trajectories,
        output_dir / "projected_trajectories.pkl",
    )

    # Current STC-Traj2vec artifacts.
    save_pickle(
        traj_ids,
        output_dir / "traj_ids.pkl",
    )
    save_pickle(
        traj_grid_lists,
        output_dir / "traj_grid_lists.pkl",
    )
    save_pickle(
        date_indices,
        output_dir / "date_indices.pkl",
    )
    save_pickle(
        padded_trajmatrices,
        output_dir / "padded_trajmatrices.pkl",
    )
    save_pickle(
        padded_weekday_date_indices,
        output_dir / "padded_weekday_date_indices.pkl",
    )

    # GeoLife notebook naming convention.
    max_days = padded_trajmatrices.shape[1]
    save_pickle(
        padded_trajmatrices,
        output_dir
        / f"stacked_weekday_matrices{max_days}.pkl",
    )
    save_pickle(
        padded_weekday_date_indices,
        output_dir
        / f"weekday_date_indices{max_days}.pkl",
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            asdict(config),
            f,
            indent=2,
            default=list,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset: str,
    input_path: str,
    output_dir: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    base_config = (
        GEOLIFE_CONFIG
        if dataset == "geolife"
        else PORTO_CONFIG
    )

    cfg = DatasetConfig(
        **asdict(base_config)
    )

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)

    output_path = Path(output_dir)
    ensure_dir(output_path)

    print("=" * 70)
    print(
        f"STC-Traj2vec data preparation: "
        f"{cfg.name}"
    )
    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print(
        f"Grid  : "
        f"{cfg.grid_size[0]} x {cfg.grid_size[1]}"
    )

    input_file = Path(input_path)

    # 1. Dataset-specific loading.
    if cfg.name == "geolife":
        raw_df = load_geolife(input_file)
        all_trajectories = standardize_geolife(raw_df)
    else:
        raw_df = load_porto(
            input_file,
            cfg,
        )
        all_trajectories = standardize_porto(raw_df)

    print(
        f"\nRaw/standardized trajectories: "
        f"{len(all_trajectories)}"
    )

    # 2. Dataset spatial filtering.
    trajectories = filter_spatial(
        all_trajectories,
        cfg.spatial_bounds,
    )
    print(
        f"After spatial filtering: "
        f"{len(trajectories)}"
    )

    # 3. Temporal resampling.
    resampled = []
    for tr in trajectories:
        tr2 = resample_trajectory(
            tr,
            cfg.resample,
        )
        if not tr2.empty:
            resampled.append(tr2)

    trajectories = resampled

    # 4. Duration/timestep/start-date filtering.
    trajectories = filter_trajectories(
        trajectories,
        cfg,
    )

    # 5. Remove the three GeoLife IDs excluded by the notebook.
    if cfg.exclude_ids:
        excluded = {
            str(x) for x in cfg.exclude_ids
        }

        trajectories = [
            tr for tr in trajectories
            if (
                str(tr["tid"].iloc[0]) not in excluded
                and str(tr["tid"].iloc[0]).split("_")[-1]
                not in excluded
            )
        ]

    print(
        f"After temporal/timestep filtering: "
        f"{len(trajectories)}"
    )

    if not trajectories:
        raise ValueError(
            "No trajectories remain after preprocessing."
        )

    # 6. Projection.
    projected = project_trajectories(
        trajectories,
        cfg,
    )

    projected_bounds = trajectory_projected_bounds(
        projected
    )

    print("\nProjected trajectory bounds:")
    print(projected_bounds)

    # 7. Matrix date range.
    matrix_start = (
        cfg.matrix_start_date
        if cfg.matrix_start_date is not None
        else str(
            min(
                t["datetime"].min()
                for t in projected
            )
        )
    )

    matrix_end = (
        cfg.matrix_end_date
        if cfg.matrix_end_date is not None
        else str(
            max(
                t["datetime"].max()
                for t in projected
            )
        )
    )

    # 8. Grid bounds.
    if cfg.bounds_mode == "geographic":
        grid_bounds = geographic_bounds_to_projected(
            cfg
        )
    elif cfg.bounds_mode == "trajectory":
        grid_bounds = projected_bounds
    else:
        raise ValueError(
            "bounds_mode must be 'geographic' "
            "or 'trajectory'."
        )

    print("\nGrid bounds:")
    print(grid_bounds)

    # 9. Build STC-Traj2vec matrices.
    matrix_builder = ProjectedTrajMatrix(
        projected,
        matrix_start,
        matrix_end,
        grid_size=cfg.grid_size,
        projected_bounds=grid_bounds,
        date_index_mode=cfg.date_index_mode,
    )

    (
        matrices,
        traj_ids,
        traj_grid_lists,
        date_indices,
    ) = matrix_builder.create_matrices()

    # 10. Pad by weekday, exactly as the notebook does.
    (
        padded_trajmatrices,
        padded_weekday_date_indices,
    ) = pad_weekday_matrices(
        matrices,
        cfg.grid_size,
    )

    # 11. Save.
    save_outputs(
        output_dir=output_path,
        all_trajectories=all_trajectories,
        filtered_trajectories=trajectories,
        projected_trajectories=projected,
        traj_ids=traj_ids,
        traj_grid_lists=traj_grid_lists,
        date_indices=date_indices,
        padded_trajmatrices=padded_trajmatrices,
        padded_weekday_date_indices=(
            padded_weekday_date_indices
        ),
        config=cfg,
    )

    stats = trajectory_statistics(
        trajectories
    )

    print("\n=== Final statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"traj_ids: {len(traj_ids)}")
    print(
        f"traj_grid_lists: "
        f"{len(traj_grid_lists)}"
    )
    print(
        f"date_indices: "
        f"{len(date_indices)}"
    )
    print(
        "padded_trajmatrices:",
        padded_trajmatrices.shape,
    )
    print(
        "padded_weekday_date_indices:",
        padded_weekday_date_indices.shape,
    )

    return {
        "config": cfg,
        "statistics": stats,
        "traj_ids": traj_ids,
        "traj_grid_lists": traj_grid_lists,
        "date_indices": date_indices,
        "padded_trajmatrices": padded_trajmatrices,
        "padded_weekday_date_indices": (
            padded_weekday_date_indices
        ),
        "projected_bounds": grid_bounds,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Geolife or Porto data "
            "for STC-Traj2vec."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["geolife", "porto"],
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV path. For GeoLife, "
            "a raw dataset directory is also accepted."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory.",
    )

    # Optional overrides.
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--matrix-start-date")
    parser.add_argument("--matrix-end-date")
    parser.add_argument(
        "--min-duration",
        type=float,
    )
    parser.add_argument(
        "--max-duration",
        type=float,
    )
    parser.add_argument(
        "--min-timesteps",
        type=int,
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
    )
    parser.add_argument("--resample")
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--bounds-mode",
        choices=["geographic", "trajectory"],
    )
    parser.add_argument("--target-crs")
    parser.add_argument(
        "--projection",
        choices=["epsg", "mercator"],
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    overrides = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "matrix_start_date": (
            args.matrix_start_date
        ),
        "matrix_end_date": (
            args.matrix_end_date
        ),
        "min_duration_min": (
            args.min_duration
        ),
        "max_duration_min": (
            args.max_duration
        ),
        "min_timesteps": (
            args.min_timesteps
        ),
        "max_timesteps": (
            args.max_timesteps
        ),
        "resample": args.resample,
        "grid_size": (
            tuple(args.grid_size)
            if args.grid_size
            else None
        ),
        "bounds_mode": args.bounds_mode,
        "target_crs": args.target_crs,
        "projection": args.projection,
    }

    run_pipeline(
        dataset=args.dataset,
        input_path=args.input,
        output_dir=args.output,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
