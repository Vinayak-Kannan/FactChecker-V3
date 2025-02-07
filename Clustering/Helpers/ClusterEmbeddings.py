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
from pinecone import Pinecone, QueryResponse

load_dotenv()


class ClusterEmbeddings:
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path="./../Clustering/Chroma")
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

    def __init__(self, min_cluster_size: int, min_samples: int, time_stamp: str, num_components: int):

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.time = time_stamp
        self.num_components = num_components

    def cluster(self) -> (pd.DataFrame, float):
        np.random.seed(23)

        query_reduced = self.reduced_collection.query(
            vector=[0] * self.num_components,
            top_k=9999,
            include_values=True,
            include_metadata=True,
        )['matches']
        query_reduced_vectors = [query_reduced[i]['values'] for i in range(len(query_reduced))]
        query_reduced_claims = [query_reduced[i]['id'] for i in range(len(query_reduced))]
        query_reduced_veracity = [query_reduced[i]['metadata']['veracity'] for i in range(len(query_reduced))]

        query_original = self.og_collection.query(
            vector=[0] * 3072,
            top_k=9999,
            include_values=True,
            include_metadata=True,
        )['matches']
        query_original_vectors = [query_original[i]['values'] for i in range(len(query_original))]
        query_original_claims = [query_original[i]['id'] for i in range(len(query_original))]
        query_original_veracity = [query_original[i]['metadata']['veracity'] for i in range(len(query_original))]

        hdbscan_object = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True, approx_min_span_tree=False).fit(query_reduced_vectors)
        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, prediction_data=True, approx_min_span_tree=False).fit_predict(query_reduced_vectors)
        # pickle the hdbscan object
        # with open(f"/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Models/hdbscan_model_{self.time}.pkl", "wb") as f:
        #     joblib.dump(hdbscan_object, f)

        # Print % of -1 clusters
        percentage_of_no_clusters = len(hdbscan_labels[hdbscan_labels == -1]) / len(hdbscan_labels)

        # Create empty dataframe
        df = pd.DataFrame()
        df["cluster"] = hdbscan_labels
        # metadata = reduced_collection['metadata']
        # metadata = [metadata[i]['claim'] for i in range(len(metadata))]
        df["claim"] = query_reduced_claims
        # metadata = reduced_collection['metadata']
        # metadata = [metadata[i]['veracity'] for i in range(len(metadata))]
        df['veracity'] = query_reduced_veracity

        # Unique clusters
        unique_clusters = df['cluster'].unique()
        # print("Number of clusters: " + str(len(unique_clusters)))
        return df, percentage_of_no_clusters

