import pickle
import time
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, MultiHeadAttention, LayerNormalization, Add, Layer
from umap.parametric_umap import ParametricUMAP
import numpy as np
import random

# Patch for TensorFlow
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------

class ParametricUMAPEncoder:
    def __init__(self, num_components, embedding_np, y_tensor, trained=False, seed=23):
        self.num_components = num_components
        self.embedding_np = embedding_np
        self.y_tensor = y_tensor
        self.trained = trained
        self.seed = seed
        
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        print("seed:", self.seed)

        # Convert to tensor
        self.embedding_np = tf.convert_to_tensor(self.embedding_np)
        self.y_tensor = tf.convert_to_tensor(self.y_tensor)

        # Define the enhanced encoder network
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(3072,)),
            layers.Dense(units=512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(units=64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dense(units=self.num_components)
        ])

        # Initialize reducer with the encoder
        self.reducer = ParametricUMAP(encoder=self.encoder, n_components=self.num_components)

        # Load weights if already trained
        if self.trained:
            self._load_weights()
        else:
            self.fit()

    def fit(self):
        start_time = time.time()
        self.reducer.fit(self.embedding_np, y=self.y_tensor)
        self.encoder.save_weights('encoder.weights.h5')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time} seconds")

    def transform(self):
        start_time = time.time()
        embedding_np = self.reducer.transform(self.embedding_np)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Transforming time: {execution_time} seconds")
        return embedding_np

    def _load_weights(self):
        self.encoder.load_weights('encoder.weights.h5')


