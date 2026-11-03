"""SPATE (space-time autocorrelation) loss.

Helper functions adapted from https://github.com/konstantinklemmer/spate-gan

The loss is MSE plus a penalty on the space-time autocorrelation (SPATE,
computed with k-nearest-neighbour local Moran's I) of the squared residuals.
Use `make_spate_knn_loss(k, b)` to build a Keras-compatible loss function.
"""

import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------
# Spatial / temporal weights
# --------------------------------------------------------------------------
def crs_to_tf_sparse(matrix):
    """Convert a compressed sparse row (CSR) matrix to a TensorFlow sparse tensor."""
    coo = matrix.tocoo()
    indices = np.array([coo.row, coo.col]).T
    return tf.sparse.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)


def make_sparse_weight_matrix(h, w, rook=False):
    """Create a sparse spatial weight matrix as a TensorFlow sparse tensor.

    Args:
        h (int): Grid height.
        w (int): Grid width.
        rook (bool): Use rook contiguity instead of queen contiguity.

    Returns:
        tf.SparseTensor: Spatial weight matrix.
    """
    from libpysal.weights import lat2W  # imported lazily; only needed here

    w_matrix = lat2W(h, w, rook=rook)
    return crs_to_tf_sparse(w_matrix.sparse)


def temporal_weights(n, l):
    """Temporal weights with exponential decay.

    Output: weights of shape (n, n).
    """
    weights = tf.exp(-tf.cast(tf.range(1, n, dtype=tf.float32)[::-1], tf.float32) / l)
    weights = tf.reshape(weights, (1, 1, -1))

    # Adjust b (decay parameter)
    l = 1 / -tf.math.log(weights[0, 0, 0]) * (n - 1)

    # Calculate weights with exponential decay
    time_range = tf.range(0, n, dtype=tf.float32)
    stacked_diffs = tf.stack([tf.abs(time_range - t) for t in tf.range(0, n, dtype=tf.float32)])
    weights = tf.exp(-stacked_diffs / l)
    # Add an extra dimension to make weights 3D
    # weights = tf.expand_dims(weights, axis=-1)

    return weights


# --------------------------------------------------------------------------
# Space-time expectations
# --------------------------------------------------------------------------
def st_ex(x, weights):
    """Expected values assuming space-time independence and sequential calculation.

    For every time step t >= 1 the expectation at each cell is a temporally
    weighted sum of the previous frames, scaled by the total mass of frame t.
    The first time step (index 0) has no history, so it gets no expectation.

    Args:
        x: Input grid data, tensor of shape [time_steps, height, width].
        weights: Temporal distance weights.

    Returns:
        Tensor of shape [time_steps - 1, height, width].
    """
    n, h, w = x.shape

    exp_val_list = []
    for t in range(1, n):
        # Weights for the t preceding frames, reshaped to broadcast over the grid.
        weighted_x = tf.reshape(weights[0, 0, -t:], [1, 1, -1]) * x[:t, :, :]
        sum_weighted_x = tf.reduce_sum(weighted_x, axis=0)  # sum across time
        sum_x_t = tf.reduce_sum(x[t, :, :])  # total of the current frame

        exp_val_t = (tf.reshape(sum_weighted_x, [-1]) * sum_x_t) / tf.reduce_sum(
            tf.reshape(weighted_x, [-1])
        )
        exp_val_list.append(tf.reshape(exp_val_t, [h, w]))

    return tf.stack(exp_val_list, axis=0)


# --------------------------------------------------------------------------
# Local Moran's I with k-nearest neighbours
# --------------------------------------------------------------------------
def morans_i_knn(x, k, threshold=0.1):
    """Local Moran's I using k-nearest neighbours on each grid.

    Args:
        x: Tensor of shape [days, hours, height, width].
    """
    x_np = x.numpy()
    days, hours, height, width = x_np.shape

    coords = np.array([(i, j) for i in range(height) for j in range(width)])
    x_reshaped = x_np.reshape(days * hours, height * width)

    all_mi_results = []
    for day_hour_data in x_reshaped:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Sparse weight matrix from the k-nearest neighbours (equal weights)
        n = len(day_hour_data)
        row_indices = np.repeat(np.arange(n), k)
        col_indices = indices.flatten()
        data = np.ones(n * k, dtype=np.float32)

        w_sparse = tf.sparse.SparseTensor(
            indices=np.stack([row_indices, col_indices], axis=-1),
            values=data,
            dense_shape=[n, n],
        )

        x_tf = tf.convert_to_tensor(day_hour_data, dtype=tf.float32)
        n = tf.shape(x_tf)[0]
        n_1 = tf.cast(n - 1, tf.float32)
        z = x_tf - tf.reduce_mean(x_tf)
        sx = tf.math.reduce_std(x_tf)
        z = z / (sx + 1e-8)
        den = tf.reduce_sum(tf.square(z))
        zl = tf.sparse.sparse_dense_matmul(w_sparse, tf.reshape(z, [-1, 1]))
        zl = tf.reshape(zl, [-1])
        mi = (n_1 * z * zl) / (den + 1e-8)

        all_mi_results.append(mi)

    final_mi_result = tf.stack(all_mi_results, axis=0)
    return tf.reshape(final_mi_result, (days, hours, height, width))


def morans_i_knn_mean(x, cus_mean, k):
    """Local Moran's I using k-nearest neighbours and a custom mean.

    Args:
        x: Tensor of shape [height, width].
        cus_mean: Expected values (flattened) used in place of the grid mean.
        k: Number of nearest neighbours.
    """
    x_np = x.numpy()
    height, width = x_np.shape

    coords = np.array([(i, j) for i in range(height) for j in range(width)])
    x_reshaped = np.reshape(x_np, (height * width))

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    n = len(x_reshaped)
    row_indices = np.repeat(np.arange(n), k)
    col_indices = indices.flatten()
    data = np.ones(n * k, dtype=np.float32)

    w_sparse = tf.sparse.SparseTensor(
        indices=np.stack([row_indices, col_indices], axis=-1),
        values=data,
        dense_shape=[n, n],
    )

    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    n = tf.shape(x_tf)[0]
    n_1 = tf.cast(n - 1, tf.float32)
    z = x_tf - cus_mean
    sx = tf.math.reduce_std(x_tf)
    z = z / (sx + 1e-8)
    den = tf.reduce_sum(tf.square(z))
    zl = tf.sparse.sparse_dense_matmul(w_sparse, tf.reshape(z, [-1, 1]))
    zl = tf.reshape(zl, [-1])
    z = tf.reshape(z, [-1])

    mi = (n_1 * z * zl) / (den + 1e-8)
    return np.reshape(mi, (height, width))


# --------------------------------------------------------------------------
# SPATE
# --------------------------------------------------------------------------
@tf.function
def get_spates_knn(x, k, b, method="skw"):
    """SPATE (space-time autocorrelation) for a 3D spatial grid using k-NN.

    Args:
        x: Input tensor of shape [time_steps, height, width].
        k: Number of nearest neighbours.
        b: Temporal distance weights (from `temporal_weights`).
        method: 'skw', 'k' or 'kw'. Defaults to 'skw'.

    Returns:
        SPATE tensor of shape [time_steps (- 1 for 'skw'), height, width].
    """
    time_steps, height, width = x.shape
    x_means = st_ex(x, b)

    spates_list = []
    for i in range(time_steps - 1 if method == "skw" else time_steps):
        morans_i_result = morans_i_knn_mean(
            x[i if method != "skw" else i + 1, :, :],
            tf.reshape(x_means[i, :, :], [-1]),
            k,
        )
        spates_list.append(morans_i_result)

    return tf.stack(spates_list, axis=0)


@tf.function
def make_spates_knn(x, k, b, method="skw"):
    """SPATE for a batch of 3D spatial grids using k-NN.

    Args:
        x: Input tensor of shape [batch_size, time_steps, height, width, n_channel].
        k: Number of nearest neighbours.
        b: Temporal distance weights (from `temporal_weights`).
        method: 'skw', 'k' or 'kw'. Defaults to 'skw'.

    Returns:
        SPATE tensor of shape [batch_size, time_steps, height, width, n_channel]
        (time_steps - 1 for 'skw'), min-max normalised per sample.
    """
    batch_size, time_steps, height, width, n_channel = x.shape

    if batch_size is None:
        batch_size = 0

    spates_list = []
    for i in range(batch_size):
        for j in range(n_channel):
            sgrid = x[i, :, :, :, j]
            spates_list.append(get_spates_knn(sgrid, k, b, method))

    spates = tf.stack(spates_list, axis=0)

    if method == "skw":
        time_steps -= 1

    spates = tf.reshape(spates, [batch_size, time_steps, height, width, n_channel])

    # Normalize
    variance = tf.math.reduce_variance(spates)
    if variance < 1e-6:
        spates = tf.zeros_like(spates)
    else:
        min_spates = tf.reduce_min(spates, axis=[1, 2, 3, 4], keepdims=True)
        max_spates = tf.reduce_max(spates, axis=[1, 2, 3, 4], keepdims=True)
        spates = (spates - min_spates) / (max_spates - min_spates)

    return spates


def make_spate_knn_loss(k, b, spate_weight=0.8, method="skw"):
    """Build the loss: (1 - l) * MSE + l * sum(|SPATE of squared residuals|).

    The original notebook read `k` and `b` from module globals; here they are
    passed in explicitly so the same code works for any dataset.

    Args:
        k: Number of nearest neighbours.
        b: Temporal distance weights (from `temporal_weights`).
        spate_weight: Weight `l` of the SPATE term (default 0.8).
        method: SPATE method, default 'skw'.

    Returns:
        A loss function `loss(y_true, y_pred)` for `model.compile`.
    """

    def spate_knn_loss(y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

        residuals = y_true - y_pred
        spates_residuals = make_spates_knn(residuals ** 2, k, b, method=method)
        spates_loss = tf.reduce_sum(tf.abs(spates_residuals))  # minimise |SPATE|

        l = spate_weight
        return (1 - l) * mse + l * spates_loss

    return spate_knn_loss
