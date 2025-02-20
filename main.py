import pickle
import time
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, MultiHeadAttention, LayerNormalization, \
    Add, Layer
from umap.parametric_umap import ParametricUMAP
import numpy as np
import random
import os
import boto3
import tempfile
from botocore.exceptions import ClientError

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
                 s3_bucket="sagemaker-us-east-1-390403859474", s3_key="umap/my-param-umap-weights.h5"):
        self.num_components = num_components
        self.embedding_np = embedding_np
        self.y_tensor = y_tensor
        self.trained = trained
        self.seed = seed
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key

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
            loaded = self._try_load_from_s3()
            if not loaded:
                print(f"Weights file not found. Running fit() instead.")
                self.fit()

                print("Now saving newly trained parametric UMAP to S3 ...")
                self.save_to_s3(self.s3_bucket, self.s3_key)
        else:
            self.fit()

            print("Now saving newly trained parametric UMAP to S3 ...")
            self.save_to_s3(self.s3_bucket, self.s3_key)

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

    #
    # def _load_weights(self):
    #     self.encoder.load_weights('encoder.weights.h5')

    def save_to_s3(self, bucket: str, key: str):
        """
        Save the trained Keras model to S3.
        """
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.encoder.save(tmp_path)
            s3.upload_file(tmp_path, bucket, key)
            print(f"[ParametricUMAPEncoder] Saved model to s3://{bucket}/{key}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _try_load_from_s3(self) -> bool:
        """
        Try to load the trained Keras model from S3. Return True if successful, False otherwise.
        """
        if not self.s3_bucket or not self.s3_key:
            print("[ParametricUMAPEncoder] s3_bucket or s3_key not provided, skip loading from S3.")
            return False
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            print(f"[ParametricUMAPEncoder] Trying to load from s3://{self.s3_bucket}/{self.s3_key}")
            s3.download_file(self.s3_bucket, self.s3_key, tmp_path)
            # Load the model
            loaded_model = keras.models.load_model(tmp_path)
            self.encoder = loaded_model
            # Initialize the reducer
            self.reducer = ParametricUMAP(encoder=self.encoder, n_components=self.num_components)
            print(f"[ParametricUMAPEncoder] Loaded model from s3://{self.s3_bucket}/{self.s3_key}")
            return True
        except ClientError as e:
            # If the file does not exist or cannot be accessed, return False
            if e.response['Error']['Code'] == '404':
                print("[ParametricUMAPEncoder] S3 file not found.")
            else:
                print(f"[ParametricUMAPEncoder] Error while loading from S3: {e}")
            return False
        except Exception as e:
            print(f"[ParametricUMAPEncoder] Unexpected error loading from S3: {e}")
            return False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def load_from_s3(cls, bucket: str, key: str, num_components=100, seed=23):
        """
        Load the trained Keras model from S3.
        """
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            s3.download_file(bucket, key, tmp_path)
            # Create a dummy object to load the model
            dummy_embedding = tf.zeros((1, 3072), dtype=tf.float32)
            dummy_y = tf.zeros((1,), dtype=tf.float32)
            obj = cls(num_components, dummy_embedding, dummy_y, trained=True, seed=seed)
            # load_model
            loaded_model = keras.models.load_model(tmp_path)
            obj.encoder = loaded_model
            # Initialize the reducer
            obj.reducer = ParametricUMAP(
                encoder=obj.encoder,
                n_components=obj.num_components
            )
            print(f"[ParametricUMAPEncoder] Loaded model from s3://{bucket}/{key}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return obj

