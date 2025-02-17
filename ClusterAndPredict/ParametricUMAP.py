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
import os

# Patch for TensorFlow
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------

class ParametricUMAPEncoder:
    def __init__(self, num_components, embedding_np, y_tensor, trained=False, seed=23,
                 weights_path='encoder.weights.h5'):
        self.num_components = num_components
        self.embedding_np = embedding_np
        self.y_tensor = y_tensor
        self.trained = trained
        self.seed = seed
        self.weights_path = weights_path
        
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
            weights_path = 'encoder.weights.h5'
            if os.path.exists(weights_path):
                self._load_weights()
            else:
                print(f"Weights file not found at {weights_path}. Running fit() instead.")
                self.fit()
        else:
            self.fit()

    def fit(self):
        print("Num GPUs Available during fit: ", len(tf.config.list_physical_devices('GPU')))
        start_time = time.time()
        self.reducer.fit(self.embedding_np, y=self.y_tensor)
        self.encoder.save_weights('encoder.weights.h5')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training time: {execution_time} seconds")

        self.trained = True

    def transform(self, new_data=None):
        if new_data is None:
            new_data = self.embedding_np
        start_time = time.time()
        new_data = tf.convert_to_tensor(new_data)
        # embedding_np = self.reducer.transform(self.embedding_np)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Transforming time: {execution_time} seconds")
        return self.reducer.transform(new_data)

    def _load_weights(self):
        self.encoder.load_weights('encoder.weights.h5')

    def save(self, filepath: str):
        """
        Serialize the entire Keras model (with structure + weights) as an .h5 file
        If you only want to save the weights, go ahead and use self.encoder.save_weights().
        But save() is the most complete, containing the network structure and so on.
        """
        print(f"Saving entire Keras model to {filepath} ...")
        self.encoder.save(filepath)
        print("Done saving model.")

    @classmethod
    def load(cls, filepath: str, num_components=100, seed=23):
        """
        Load the trained Keras model from the filepath (some .h5 file).
        Then reinitialize ParametricUMAPEncoder and replace the encoder with the loaded model.
        Note: For transforms, we pass dummy to embedding_np/y_tensor at init here, because it doesn't need to be fit again.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ParametricUMAPEncoder.load: {filepath} does not exist!")

        print(f"Loading entire Keras model from {filepath} ...")
        # Create dummy embedding and y_tensor
        dummy_embedding = tf.zeros((1, 3072), dtype=tf.float32)
        dummy_y = tf.zeros((1,), dtype=tf.float32)

        # Initialize ParametricUMAPEncoder with dummy embedding and y_tensor
        obj = cls(num_components=num_components,
                  embedding_np=dummy_embedding,
                  y_tensor=dummy_y,
                  trained=True,
                  seed=seed)
        # Load the model
        loaded_model = tf.keras.models.load_model(filepath)
        # Replace the encoder with the loaded model
        obj.encoder = loaded_model
        # Reinitialize the reducer with the loaded encoder
        obj.reducer = ParametricUMAP(encoder=obj.encoder, n_components=obj.num_components)

        print("Done loading Keras model; ParametricUMAPEncoder is ready.")
        return obj

