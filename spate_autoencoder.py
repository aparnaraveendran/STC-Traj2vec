"""ConvLSTM autoencoder whose bottleneck gives the SPATE context embeddings."""

from tensorflow.keras.layers import (
    Input, ConvLSTM2D, Dropout, BatchNormalization, Flatten, Dense,
    Reshape, Conv3DTranspose,
)
from tensorflow.keras.models import Model
import tensorflow as tf


def convLSTM_autoencoder(input_shape=(13, 128, 128, 1), embedding_dim=64):
    """ConvLSTM autoencoder that reconstructs the full spatio-temporal sequence.

    Includes small-normal initialization and dilation in the encoder.

    Args:
        input_shape: (time_steps, height, width, channels).
        embedding_dim: Size of the bottleneck (the context embedding).

    Returns:
        (encoder, autoencoder) Keras models sharing weights.
    """
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3, seed=None)
    act = 'relu'
    drop1 = 0.2
    drop2 = 0.3

    # ------- ENCODER -------
    inputs = Input(shape=input_shape)

    x = ConvLSTM2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        dilation_rate=2,
        activation=act,
        kernel_initializer=init,
        return_sequences=True
    )(inputs)
    x = Dropout(drop1)(x)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        dilation_rate=2,
        activation=act,
        kernel_initializer=init,
        return_sequences=False
    )(x)
    x = Dropout(drop1)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Bottleneck (embedding layer)
    embedding = Dense(embedding_dim, activation=act, kernel_initializer=init, name="Embedding_Layer")(x)

    # ------- DECODER -------
    x = Dense(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], activation=act, kernel_initializer=init)(embedding)
    x = Reshape((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))(x)  # back to original dimensions

    x = Dropout(drop2)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), padding='same', dilation_rate=1, activation=act, kernel_initializer=init)(x)

    x = Dropout(drop2)(x)
    x = BatchNormalization()(x)
    x = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), padding='same', dilation_rate=1, kernel_initializer=init, activation="sigmoid")(x)

    # ------- MODELS -------
    autoencoder = Model(inputs, x, name='ConvLSTM_Autoencoder')
    encoder = Model(inputs, embedding, name='ConvLSTM_Encoder')

    return encoder, autoencoder
