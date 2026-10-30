import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, Input
from tensorflow.keras.utils import register_keras_serializable

# Update these paths or set INPUT_GEOLIFE, INPUT_PORTO and OUTPUT_ROOT as environment variables.
INPUT_GEOLIFE = os.environ.get("INPUT_GEOLIFE", "/data/geolife.csv")
INPUT_PORTO = os.environ.get("INPUT_PORTO", "train.csv")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "./processed")

H_DIM = 128
W_DIM = 128
T_DIM = 24
POINT_EMBEDDING_DIM = 96
EPOCHS = 16
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
PATIENCE = 10


def get_dataset_config(dataset):
    raw_input = INPUT_GEOLIFE if dataset == "geolife" else INPUT_PORTO
    dataset_dir = Path(OUTPUT_ROOT) / dataset
    output_dir = dataset_dir / "output"
    checkpoint_dir = dataset_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset": dataset,
        "raw_input": Path(raw_input),
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
    }


def find_dataset_file(dataset_dir, filename, legacy_weekday=False):
    candidates = [dataset_dir / filename, dataset_dir / "316_3M" / filename]
    if legacy_weekday and filename == "weekday_date_indices.pkl":
        candidates.extend([
            dataset_dir / "weekday_date_indices13.pkl",
            dataset_dir / "316_3M" / "weekday_date_indices13.pkl",
        ])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing {filename}. Searched: " + ", ".join(map(str, candidates))
    )


def load_dataset_files(config):
    dataset_dir = config["dataset_dir"]
    dataset = config["dataset"]
    grid_path = find_dataset_file(dataset_dir, "traj_grid_lists.pkl")
    weekday_path = find_dataset_file(
        dataset_dir, "weekday_date_indices.pkl", legacy_weekday=True
    )
    # Each dataset must have its own SPATE embedding file.
    spate_path = find_dataset_file(
        dataset_dir, f"spate_knn_embeddings_{dataset}.npy"
    )
    with open(grid_path, "rb") as f:
        grid_lists = pickle.load(f)
    with open(weekday_path, "rb") as f:
        weekday_date_indices = pickle.load(f)
    spate_embeddings = np.load(spate_path)
    if spate_embeddings.ndim == 2 and spate_embeddings.shape[0] == 168:
        spate_embeddings = spate_embeddings.reshape(7, 24, -1)
    if spate_embeddings.ndim != 3 or spate_embeddings.shape[:2] != (7, 24):
        raise ValueError(
            f"Expected SPATE embeddings with shape (7, 24, D); got {spate_embeddings.shape}"
        )
    return grid_lists, weekday_date_indices, spate_embeddings, {
        "grid": grid_path, "weekday": weekday_path, "spate": spate_path
    }


def save_pickle(path, value):
    with open(path, "wb") as f:
        pickle.dump(value, f)


def build_trajectory_inputs(grid_lists):
    point_embedding = PointEmbedding(
        H_DIM, W_DIM, T_DIM, POINT_EMBEDDING_DIM
    )
    traj_seq_embed = {}
    traj_date_hours = {}
    empty_trajs = []
    for traj_id, traj in enumerate(grid_lists):
        if not traj:
            empty_trajs.append(traj_id)
            continue
        h_list, w_list, hour_list = [], [], []
        date_hours = {}
        for point in traj:
            date_index, hour, h, w = point
            h_list.append(h)
            w_list.append(w)
            hour_list.append(hour)
            date_hours.setdefault(date_index, []).append(hour)
        traj_seq_embed[traj_id] = point_embedding((
            tf.constant(h_list, dtype=tf.int32),
            tf.constant(w_list, dtype=tf.int32),
            tf.constant(hour_list, dtype=tf.int32),
        ))
        traj_date_hours[traj_id] = {
            date: sorted(set(hours)) for date, hours in date_hours.items()
        }
    return traj_seq_embed, traj_date_hours, empty_trajs


class PointEmbedding(tf.keras.layers.Layer):
    def __init__(self, h_dim, w_dim, t_dim, embedding_dim, **kwargs):
        super(PointEmbedding, self).__init__(**kwargs)
        self.h_embedding = tf.keras.layers.Embedding(input_dim=h_dim, output_dim=embedding_dim // 3)
        self.w_embedding = tf.keras.layers.Embedding(input_dim=w_dim, output_dim=embedding_dim // 3)
        self.t_embedding = tf.keras.layers.Embedding(input_dim=t_dim, output_dim=embedding_dim // 3)
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        """
        Args:
            inputs: A tuple of 3 tensors:
                h_indices: (batch_size,) or (sequence_length,)
                w_indices: same shape
                t_indices: same shape (hour of day)

        Returns:
            tf.Tensor: (batch_size, embedding_dim)
        """
        h_indices, w_indices, t_indices = inputs
        h_emb = self.h_embedding(h_indices)
        w_emb = self.w_embedding(w_indices)
        t_emb = self.t_embedding(t_indices)

        return tf.concat([h_emb, w_emb, t_emb], axis=-1)

class TrajectoryContextMapper:
    """
    A class to process trajectory data, map contexts (day, hour) to trajectory
    indices, and generate context-aware embeddings.

    This class loads trajectory data, processes it to create mappings based on
    the day of the week and hour, and can generate contextual embeddings for
    trajectories using pre-trained spatial-temporal embeddings.

    Args:
        data_dir (str): The directory where the data files are located.
        weekday_indices_filename (str): The filename for the pickled weekday date indices.
        traj_date_hours (dict): A dictionary where keys are trajectory IDs and
                                values are dictionaries mapping dates to lists of hours.
                                Example: {traj_id: {date: [hour1, hour2]}}
    """

    def __init__(self, data_dir, weekday_indices_filename, traj_date_hours):
        self.data_dir = data_dir
        self.weekday_indices_filename = weekday_indices_filename
        self.traj_date_hours = traj_date_hours
        self.weekday_date_indices = self._load_weekday_indices()
        self.weekday_dict = self._create_weekday_dict()
        self.date_hour_pairs, self.day_hour_pairs = self._generate_hour_pairs()
        self.day_hour_pairs_list = self._create_day_hour_pairs_list()
        self.context_to_indices = self.build_context_groups()

    def _load_weekday_indices(self):
        """Loads the weekday date indices from a pickle file."""
        try:
            with open(os.path.join(self.data_dir, self.weekday_indices_filename), 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Error: The file {self.weekday_indices_filename} was not found in {self.data_dir}")
            return []

    def _create_weekday_dict(self):
        """Creates a dictionary mapping date indices to weekdays (0-6)."""
        weekday_dict = {}
        for i, date_indices in enumerate(self.weekday_date_indices):
            for date_index in date_indices:
                weekday_dict[date_index] = i
        return weekday_dict

    def _generate_hour_pairs(self):
        """Generates (date, hour) and (weekday, hour) pairs for each trajectory."""
        date_hour_pairs = {}
        day_hour_pairs = {}
        for traj_id, date_hours in self.traj_date_hours.items():
            pairs, w_pairs = [], []
            for date, hours in date_hours.items():
                if date in self.weekday_dict:
                    for hour in hours:
                        pairs.append((date, hour))
                        w_pairs.append((self.weekday_dict[date], hour))
            date_hour_pairs[traj_id] = pairs
            day_hour_pairs[traj_id] = w_pairs
        return date_hour_pairs, day_hour_pairs

    def _create_day_hour_pairs_list(self):
        """Preserves trajectory IDs for sequence/context alignment."""
        return {traj_id: list(pairs) for traj_id, pairs in self.day_hour_pairs.items()}

    def build_context_groups(self):
        """
        Builds a dictionary mapping (day, hour) contexts to a list of trajectory indices.

        Returns:
            dict: A dictionary where keys are (day, hour) tuples and values are lists
                  of trajectory indices that occur at that context.
        """
        context_to_indices = {}
        for traj_id, day_hour_pairs in self.day_hour_pairs_list.items():
            for day, hour in day_hour_pairs:
                if (day, hour) not in context_to_indices:
                    context_to_indices[(day, hour)] = []
                context_to_indices[(day, hour)].append(traj_id)
        return context_to_indices

    def get_context_groups(self):
        """
        Returns the created context groups.
        """
        return self.context_to_indices

    def get_context_embeddings(self, spate_embeddings):
        """
        Generates contextual embeddings for each trajectory.

        This method retrieves a context embedding for each time point in a trajectory
        based on its weekday and hour, using a pre-loaded matrix of spatio-temporal
        embeddings.

        Args:
            spate_embeddings (np.ndarray): A 3D numpy array of shape (7, 24, embedding_dim)
                                           containing the embedding for each day and hour.

        Returns:
            dict: A dictionary mapping each trajectory index to its corresponding
                  contextual embedding tensor. The shape of the tensor for each
                  trajectory will be (1, num_time_points, embedding_dim).
        """
        traj_context = {}
        for traj_id, pairs in self.day_hour_pairs_list.items():
            if not pairs:
                continue
            context_indices = tf.constant(pairs, dtype=tf.int32)
            context_embed = tf.gather_nd(spate_embeddings, context_indices)
            traj_context[traj_id] = tf.expand_dims(context_embed, axis=0)
        return traj_context


class LocalSelfAttention(layers.Layer):
    def __init__(self, output_dim, local_window_size, **kwargs):
        super().__init__(**kwargs)
        if local_window_size % 2 == 0:
            raise ValueError("`local_window_size` must be an odd number.")

        self.output_dim = output_dim
        self.local_window_size = local_window_size
        self.supports_masking = True

        initializer = initializers.HeNormal()

        # Define projection layers
        self.query_dense = layers.Dense(output_dim, kernel_initializer=initializer, name='query_dense')
        self.key_dense = layers.Dense(output_dim, kernel_initializer=initializer, name='key_dense')
        self.value_dense = layers.Dense(output_dim, kernel_initializer=initializer, name='value_dense')

    def build(self, input_shape):
        # Build dense layers with appropriate input shapes
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        super().build(input_shape)

    @tf.function
    def call(self, inputs, mask=None):
        # inputs: (batch_size, seq_len, feature_dim)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        q = self.query_dense(inputs)  # (B, L, D)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)

        pad = self.local_window_size // 2
        padded_k = tf.pad(k, [[0, 0], [pad, pad], [0, 0]])
        padded_v = tf.pad(v, [[0, 0], [pad, pad], [0, 0]])

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            padded_mask = tf.pad(mask, [[0, 0], [pad, pad]], constant_values=0.0)
        else:
            padded_mask = tf.ones((batch_size, seq_len + 2 * pad), dtype=tf.float32)

        def compute_step(i):
            q_i = q[:, i:i+1, :]  # (B, 1, D)
            k_window = padded_k[:, i:i+self.local_window_size, :]  # (B, W, D)
            v_window = padded_v[:, i:i+self.local_window_size, :]
            m_window = padded_mask[:, i:i+self.local_window_size]  # (B, W)

            scores = tf.matmul(q_i, k_window, transpose_b=True)  # (B, 1, W)
            scores += (1. - tf.expand_dims(m_window, 1)) * -1e9  # Masking
            weights = tf.nn.softmax(scores, axis=-1)  # (B, 1, W)
            output_i = tf.matmul(weights, v_window)  # (B, 1, D)
            return tf.squeeze(output_i, axis=1)  # (B, D)

        output = tf.map_fn(
            compute_step,
            elems=tf.range(seq_len),
            fn_output_signature=tf.float32
        )  # (L, B, D)

        output = tf.transpose(output, perm=[1, 0, 2])  # (B, L, D)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "local_window_size": self.local_window_size
        })
        return config


import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class ContextualTrajectoryEncoder(tf.keras.Model):
    def __init__(self, lstm_units=[64, 32], attention_dim=32, output_dim=64, local_window_size=5, **kwargs):
        super().__init__(**kwargs)
        he_init = initializers.HeNormal()
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.attention_dim = attention_dim
        self.local_window_size = local_window_size
        self.lstm1 = layers.LSTM(lstm_units[0], return_sequences=True, kernel_initializer=he_init, mask_zero=True)
        self.lstm2 = layers.LSTM(lstm_units[1], return_sequences=True, kernel_initializer=he_init, mask_zero=True)
        #self.masking = layers.Masking(mask_value=0.0)

        # GlobalAveragePooling1D respects the mask automatically, so it will only
        # average over the non-padded parts of the sequence.
        self.masked_average_pooling = layers.GlobalAveragePooling1D()

        self.projection = layers.Dense(attention_dim, kernel_initializer=he_init)
        self.projection_act = layers.LeakyReLU(negative_slope=0.01)

        self.attention = LocalSelfAttention(attention_dim, local_window_size)
        self.norm = layers.LayerNormalization()

        self.embedding_proj = layers.Dense(output_dim, kernel_initializer=he_init)
        self.embedding_proj_act = layers.LeakyReLU(negative_slope=0.01)

        self.ctx_output_proj = layers.Dense(output_dim, kernel_initializer=he_init)
        self.ctx_output_proj_act = layers.LeakyReLU(negative_slope=0.01)

        self.gate_dense = layers.Dense(output_dim, activation='sigmoid', kernel_initializer=he_init)
        self.fusion_norm = layers.LayerNormalization()

        self.final_dense = layers.Dense(output_dim, kernel_initializer=he_init)
        self.final_act = layers.LeakyReLU(negative_slope=0.01)

    def build(self, input_shape):
        seq_input_shape, ctx_input_shape = input_shape
        self.embedding_proj.build((None, self.lstm_units[1]))
        self.ctx_output_proj.build((None, self.output_dim))
        super().build(input_shape)

    def call(self, inputs, training=False):
        traj_seq, context_embed = inputs

        x = traj_seq

        #x = self.masking(traj_seq)
        #mask = self.masking.compute_mask(traj_seq)

        lstm_out = self.lstm1(x, training=training)
        lstm_out = self.lstm2(lstm_out, training=training)

        attn_out = self.attention(lstm_out)
        residual = self.projection(lstm_out)
        residual = self.projection_act(residual)
        x = self.norm(attn_out + residual)

        x = self.masked_average_pooling(x)
        traj_embedding = self.embedding_proj(x)
        traj_embedding = self.embedding_proj_act(traj_embedding)

        if context_embed is not None:
            weighted_ctx = self.attend_context(traj_embedding, context_embed)
            if weighted_ctx.shape[-1] != traj_embedding.shape[-1]:
                weighted_ctx = self.ctx_output_proj(weighted_ctx)
                weighted_ctx = self.ctx_output_proj_act(weighted_ctx)

            gate_input = tf.concat([traj_embedding, weighted_ctx], axis=-1)
            gate = self.gate_dense(gate_input)
            fused = gate * weighted_ctx + (1.0 - gate) * traj_embedding
            fused = self.fusion_norm(fused + traj_embedding)
        else:
            fused = traj_embedding

        output = self.final_dense(fused)
        output = self.final_act(output)
        return tf.nn.l2_normalize(output, axis=-1)

    def attend_context(self, traj_emb, ctx_emb):
        traj_emb_exp = tf.expand_dims(traj_emb, axis=1)
        scores = tf.matmul(traj_emb_exp, ctx_emb, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        weighted_ctx = tf.matmul(weights, ctx_emb)
        return tf.squeeze(weighted_ctx, axis=1)

    '''
    def masked_average_pooling(self, inputs, mask):
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        masked = inputs * mask
        summed = tf.reduce_sum(masked, axis=1)
        count = tf.reduce_sum(mask, axis=1)
        count = tf.where(tf.equal(count, 0.), 1e-9, count)
        return summed / count
    '''

    def get_config(self):
        config = super().get_config()
        config.update({
            'lstm_units': self.lstm_units,
            'attention_dim': self.attention_dim,
            'output_dim': self.output_dim,
            'local_window_size': self.local_window_size,
            # Add other parameters if necessary for serialization
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the layers with their configurations
        encoder = cls(**config)
        return encoder
        


import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class MoCoTrajectoryContrastive(tf.keras.Model):
    def __init__(self, encoder_q, encoder_k, embedding_dim=64, queue_size=4096, momentum=0.999, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.encoder_k.trainable = False

        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.queue = tf.Variable(tf.math.l2_normalize(tf.zeros((queue_size, embedding_dim)), axis=-1), trainable=False, name="moco_queue")
        self.queue_ptr = tf.Variable(0, dtype=tf.int32, trainable=False, name="queue_pointer")

        # Optional: initialization for any extra layers (if added later)
        self.he_init = initializers.HeNormal()
        self.leaky_relu = layers.LeakyReLU(negative_slope=0.2)

    def build(self, input_shape):
        self.encoder_q.build(input_shape)
        self.encoder_k.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        pos_seq, pos_ctx = inputs
        q = self.encoder_q((pos_seq, pos_ctx), training=training)
        q = tf.nn.l2_normalize(q, axis=-1)

        if training:
            self.update_momentum_encoder()

        # Dropout-based augmentation
        if training:
            pos_seq = self.apply_batch_dropout(pos_seq, dropout_rate=0.5)

        k = self.encoder_k((pos_seq, pos_ctx), training=False)
        k = tf.stop_gradient(tf.nn.l2_normalize(k, axis=-1))

        # Positive logit
        l_pos = tf.reduce_sum(q * k, axis=-1, keepdims=True)

        # Negative logits from queue
        queue_keys = tf.stop_gradient(self.queue)
        l_neg = tf.matmul(q, queue_keys, transpose_b=True)

        # Combine and scale
        logits = tf.concat([l_pos, l_neg], axis=1) / self.temperature
        labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)

        if training:
            self.dequeue_and_enqueue(k)

        #self.add_metric(loss, name="loss", aggregation="mean")
        return {'loss': loss}


    @property
    def metrics(self):
        return [self.loss_tracker]

    def apply_batch_dropout(self, sequence, dropout_rate=0.1):
        mask = tf.cast(tf.random.uniform(tf.shape(sequence)[:-1]) > dropout_rate, sequence.dtype)
        return sequence * tf.expand_dims(mask, axis=-1)

    def dequeue_and_enqueue(self, keys):
        keys = tf.stop_gradient(keys)
        batch_size = tf.shape(keys)[0]
        ptr = self.queue_ptr
        end_ptr = (ptr + batch_size) % self.queue_size

        def enqueue_wrap():
            num_wrap = self.queue_size - ptr
            self.queue[ptr:].assign(keys[:num_wrap])
            self.queue[:end_ptr].assign(keys[num_wrap:])
            return end_ptr

        def enqueue_linear():
            self.queue[ptr:end_ptr].assign(keys)
            return end_ptr

        new_ptr = tf.cond(end_ptr > ptr, enqueue_linear, enqueue_wrap)
        self.queue_ptr.assign(new_ptr)

    def update_momentum_encoder(self):
        for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
            if qw.shape == kw.shape:
                kw.assign(self.momentum * kw + (1.0 - self.momentum) * qw)

    def train_step(self, data):
        pos_seq, pos_ctx = data
        with tf.GradientTape() as tape:
            outputs = self((pos_seq, pos_ctx), training=True)
            loss = outputs['loss']
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _initialize_key_encoder(self):
        for q_layer, k_layer in zip(self.encoder_q.layers, self.encoder_k.layers):
            k_layer.set_weights(q_layer.get_weights())

    def get_config(self):
        return {
            'encoder_q_config': self.encoder_q.get_config(),
            'encoder_k_config': self.encoder_k.get_config(),
            "embedding_dim": self.embedding_dim,
            "queue_size": self.queue_size,
            "momentum": self.momentum,
            "temperature": self.temperature,
        }

    @classmethod
    def from_config(cls, config):
        encoder_q = tf.keras.models.model_from_config(config.pop('encoder_q_config'))
        encoder_k = tf.keras.models.model_from_config(config.pop('encoder_k_config'))
        return cls(encoder_q=encoder_q, encoder_k=encoder_k, **config)


class ContextualizedTrajEmbed:
    def __init__(self,
                 seq_dim=32,
                 ctx_dim=64, # Added context dimension parameter
                 lstm_units=[64, 32],
                 attention_dim=32,
                 output_dim=64,
                 local_window_size=5,
                 momentum=0.999,
                 temperature=0.07,
                 queue_size=2048):

        self.config = {
            'seq_dim': seq_dim,
            'ctx_dim': ctx_dim,  # Added to config
            'lstm_units': lstm_units,
            'attention_dim': attention_dim,
            'output_dim': output_dim,
            'local_window_size': local_window_size,
            'embedding_dim': output_dim,
            'momentum': momentum,
            'temperature': temperature,
            'queue_size': queue_size
        }

        print("Model Configuration:")
        for k, v in self.config.items():
            print(f"  {k}: {v}")

        self.query_encoder = self._build_encoder()
        self.key_encoder = tf.keras.models.clone_model(self.query_encoder)
        self.key_encoder.trainable = False

        self.moco_model = self._build_moco_model()

    def _build_encoder(self):
        # Define the input layers with dynamic sequence length (None) and feature dimensions
        traj_input = Input(shape=(None, self.config['seq_dim']), name='trajectory_sequence')
        context_input = Input(shape=(None, self.config['ctx_dim']), name='context_embedding')

        # Instantiate the ContextualTrajectoryEncoder with the defined inputs
        encoder_output = ContextualTrajectoryEncoder(
            lstm_units=self.config['lstm_units'],
            attention_dim=self.config['attention_dim'],
            output_dim=self.config['output_dim'],
            local_window_size=self.config['local_window_size']
        )((traj_input, context_input)) # Pass the inputs as a tuple

        # Create the model
        encoder_model = tf.keras.Model(inputs=[traj_input, context_input], outputs=encoder_output)
        return encoder_model


    def _build_moco_model(self):
        return MoCoTrajectoryContrastive(
            encoder_q=self.query_encoder,
            encoder_k=self.key_encoder,
            embedding_dim=self.config['embedding_dim'],
            queue_size=self.config['queue_size'],
            momentum=self.config['momentum'],
            temperature=self.config['temperature']
        )

    def _pad_sequences(self, sequences, max_len=None):
        if isinstance(sequences[0], tf.Tensor):
            sequences = [s.numpy() for s in sequences]
        if not max_len:
            max_len = max(len(seq) for seq in sequences)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, padding='pre', dtype='float32', value=0.0, maxlen=max_len
        )

    def train(self, traj_sequences, traj_context, num_epochs=16, learning_rate=1e-4, batch_size=32, patience=10, min_delta=0.001, log_dir="./logs"):

        if isinstance(traj_sequences, dict):
            valid_ids = sorted(set(traj_sequences) & set(traj_context))
            traj_sequences = [traj_sequences[i] for i in valid_ids]
            context_list = [tf.squeeze(traj_context[i], axis=0).numpy() for i in valid_ids]
        else:
            context_list = [tf.squeeze(c, axis=0).numpy() for c in traj_context.values()]

        if not traj_sequences or not context_list:
            raise ValueError("No trajectories have both sequence and context embeddings.")

        dataset_size = len(traj_sequences)
        max_seq_len = max(len(seq) for seq in traj_sequences)
        max_ctx_len = max(len(c) for c in context_list)
        padded_seqs = self._pad_sequences(traj_sequences, max_seq_len)
        padded_ctxs = self._pad_sequences(context_list, max_ctx_len)

        print(f"Shape of padded_seqs: {padded_seqs.shape}")
        print(f"Shape of padded_ctxs: {padded_ctxs.shape}")


        dataset = tf.data.Dataset.from_tensor_slices((padded_seqs, padded_ctxs))
        dataset = dataset.shuffle(buffer_size=dataset_size).batch(batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Implement learning rate decay
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate=learning_rate,
        #    decay_steps=int(dataset_size / batch_size) * 30, # Decay every 30 epochs
        #    decay_rate=0.1, # Reduce by a factor of 10
        #    staircase=True
        #)

        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.moco_model.compile(optimizer=optimizer,metrics=[])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,min_delta=min_delta,restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ]

        self.moco_model._initialize_key_encoder()
        print("Key encoder initialized.")

        print("\nStarting training...")
        history = self.moco_model.fit(dataset, epochs=num_epochs, callbacks=callbacks)
        return history

    def predict(self, traj_sequences, traj_context):
        if self.query_encoder is None or not self.query_encoder.built:
            print("Model is not trained or built.")
            return None

        if isinstance(traj_sequences, dict):
            valid_ids = sorted(set(traj_sequences) & set(traj_context))
            traj_sequences = [traj_sequences[i] for i in valid_ids]
            context_list = [tf.squeeze(traj_context[i], axis=0).numpy() for i in valid_ids]
        else:
            context_list = [tf.squeeze(c, axis=0).numpy() for c in traj_context.values()]

        if not traj_sequences or not context_list:
            raise ValueError("No trajectories have both sequence and context embeddings.")

        max_seq_len = max(len(seq) for seq in traj_sequences)
        padded_seqs = self._pad_sequences(traj_sequences, max_seq_len)
        max_ctx_len = max(c.shape[0] for c in context_list)
        padded_ctxs = self._pad_sequences(context_list, max_ctx_len)

        seq_tensor = tf.convert_to_tensor(padded_seqs, dtype=tf.float32)
        ctx_tensor = tf.convert_to_tensor(padded_ctxs, dtype=tf.float32)

        return self.query_encoder((seq_tensor, ctx_tensor), training=False)

def run_dataset(dataset):
    config = get_dataset_config(dataset)
    print(f"\n========== {dataset.upper()} CONTEXT ENCODER ==========")
    print(f"Raw input configured for preprocessing: {config['raw_input']}")
    print(f"Processed directory: {config['dataset_dir']}")

    grid_lists, weekday_date_indices, spate_embeddings, input_files = load_dataset_files(config)
    print(f"Loaded trajectories: {len(grid_lists)}")
    print(f"Dataset-specific SPATE embeddings: {spate_embeddings.shape}")
    for key, path in input_files.items():
        print(f"{key}: {path}")

    traj_seq_embed, traj_date_hours, empty_trajs = build_trajectory_inputs(grid_lists)
    mapper = TrajectoryContextMapper(
        data_dir=str(input_files["weekday"].parent),
        weekday_indices_filename=input_files["weekday"].name,
        traj_date_hours=traj_date_hours,
    )
    context_groups = mapper.get_context_groups()
    day_hour_pairs = mapper.date_hour_pairs
    traj_context_embeddings = mapper.get_context_embeddings(spate_embeddings)

    valid_ids = sorted(set(traj_seq_embed) & set(traj_context_embeddings))
    if not valid_ids:
        raise ValueError(f"No trajectory/context pairs found for {dataset}.")

    output_dir = config["output_dir"]
    suffix = dataset
    save_pickle(output_dir / f"traj_seq_embed_{suffix}.pkl", traj_seq_embed)
    save_pickle(output_dir / f"traj_day_hour_pairs_{suffix}.pkl", day_hour_pairs)
    save_pickle(output_dir / f"context_groups_{suffix}.pkl", context_groups)
    save_pickle(output_dir / f"traj_context_embeddings_{suffix}.pkl", traj_context_embeddings)
    save_pickle(output_dir / f"empty_trajectories_{suffix}.pkl", empty_trajs)

    print(f"Trajectories with aligned sequence/context inputs: {len(valid_ids)}")
    model = ContextualizedTrajEmbed(
        seq_dim=POINT_EMBEDDING_DIM,
        ctx_dim=int(spate_embeddings.shape[-1]),
        lstm_units=[64, 32],
        attention_dim=32,
        output_dim=64,
        local_window_size=5,
        momentum=0.999,
        temperature=0.07,
        queue_size=2048,
    )
    history = model.train(
        traj_sequences=traj_seq_embed,
        traj_context=traj_context_embeddings,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        log_dir=str(config["checkpoint_dir"] / "tensorboard"),
    )
    trajectory_embeddings = model.predict(
        traj_sequences=traj_seq_embed,
        traj_context=traj_context_embeddings,
    )
    np.save(output_dir / f"trajectory_embeddings_{suffix}.npy", trajectory_embeddings.numpy())
    save_pickle(output_dir / f"trajectory_ids_{suffix}.pkl", valid_ids)
    model.query_encoder.save_weights(
        output_dir / f"query_encoder_{suffix}.weights.h5"
    )
    save_pickle(output_dir / f"training_history_{suffix}.pkl", history.history)
    print(f"Saved outputs to: {output_dir}")
    print(f"Trajectory embedding shape: {trajectory_embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the contextual trajectory encoder for one dataset."
    )
    parser.add_argument("--dataset", choices=["geolife", "porto"], required=True)
    args = parser.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
