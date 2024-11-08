import pickle
import time
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from umap.parametric_umap import ParametricUMAP

# Patch for TensorFlow
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

class ParametricUMAPEncoder:
    def __init__(self, num_components, embedding_np, y_tensor, trained=False):
        self.num_components = num_components
        self.embedding_np = embedding_np
        self.y_tensor = y_tensor
        self.trained = trained

        # Convert to tensor
        self.embedding_np = tf.convert_to_tensor(self.embedding_np)
        self.y_tensor = tf.convert_to_tensor(self.y_tensor)

        # Define the enhanced encoder network
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(3072,)),
            layers.Dense(units=512, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(units=512, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(units=256, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(units=128, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
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

# # Example usage
# if __name__ == "__main__":
#     # Load embedding_np and y_tensor
#     with open('variables.pkl', 'rb') as f:
#         embedding_np, y_tensor = pickle.load(f)

#     num_components = 100
#     trained = False  # Set to True if the model has already been trained

#     param_umap_encoder = ParametricUMAPEncoder(num_components, embedding_np, y_tensor, trained=trained)
#     embedding_np_transformed = param_umap_encoder.transform()
