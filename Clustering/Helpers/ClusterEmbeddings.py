import chromadb
import joblib
import umap
import hdbscan
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()


class ClusterEmbeddings:
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")

    def __init__(self, min_cluster_size: int, min_samples: int, time_stamp: str):
        self.og_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_" + time_stamp,
        )
        self.reduced_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_reduced_" + time_stamp,
            metadata={"hnsw:space": "cosine"}
        )
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.time = time_stamp

    def cluster(self) -> (pd.DataFrame, float):
        np.random.seed(23)

        reduced_collection = self.reduced_collection.get(include=['embeddings', 'documents', 'metadatas'])
        # Cluster the embeddings using HDBSCAN
        hdbscan_object = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True, approx_min_span_tree=False).fit(reduced_collection['embeddings'])
        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True, approx_min_span_tree=False).fit_predict(reduced_collection['embeddings'])
        # pickle the hdbscan object
        with open(f"/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Models/hdbscan_model_{self.time}.pkl", "wb") as f:
            joblib.dump(hdbscan_object, f)

        # Print % of -1 clusters
        percentage_of_no_clusters = len(hdbscan_labels[hdbscan_labels == -1]) / len(hdbscan_labels)

        # Create empty dataframe
        df = pd.DataFrame()
        df["cluster"] = hdbscan_labels
        metadata = reduced_collection['metadatas']
        metadata = [metadata[i]['claim'] for i in range(len(metadata))]
        df["claim"] = metadata
        metadata = reduced_collection['metadatas']
        metadata = [metadata[i]['veracity'] for i in range(len(metadata))]
        df['veracity'] = metadata

        # Unique clusters
        unique_clusters = df['cluster'].unique()
        print("Number of clusters: " + str(len(unique_clusters)))
        return df, percentage_of_no_clusters

