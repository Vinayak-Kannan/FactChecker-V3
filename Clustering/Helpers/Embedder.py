import glob
from datetime import datetime

import joblib
import numpy as np
import umap
from joblib import load
from openai import OpenAI
import pandas as pd
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv
import os

from umap import UMAP

load_dotenv()


class Embedder:
    random_seed = None
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(
        path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

    def __init__(self, n_neighbors: int, min_dist: float, num_components: int, no_umap: bool, time_stamp: str,
                 random_seed: bool = False):
        self.embedding_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_embeddings_unchanged",
            metadata={"hnsw:space": "cosine"}
        )

        self.og_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_" + time_stamp,
            metadata={"hnsw:space": "cosine"}
        )
        self.reduced_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_reduced_" + time_stamp,
            metadata={"hnsw:space": "cosine"}
        )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.num_components = num_components
        self.no_umap = no_umap
        self.time = time_stamp
        self.random_seed = random_seed

    def __get_embedding(self, text: str) -> np.ndarray:
        model = "text-embedding-3-large"
        query_original = self.embedding_collection.get(ids=[text], include=['embeddings', 'documents', 'metadatas'])
        if query_original and len(query_original['embeddings']) > 0:
            return np.array(query_original['embeddings'][0])

        query = self.og_collection.get(ids=[text], include=['embeddings', 'documents', 'metadatas'])
        if query and len(query['embeddings']) > 0:
            return np.array(query['embeddings'][0])
        else:
            # print("Had to create embedding...")
            result = self.client.embeddings.create(input=[text], model=model).data[
                0].embedding  # extra_body={'dimensions': 256}
            self.embedding_collection.upsert(
                documents=[text],
                embeddings=[result],
                metadatas=[{"claim": text}],
                ids=[text]
            )
            return np.array(result)

    def embed_column_ground_truth(self, df: pd.DataFrame, claim_column_name: str, veracity_column_name: str,
                                  supervised_label_column_name: str,
                                  insert_into_collection: bool = True, supervised_umap: bool = False) -> None:
        embeddings = df[claim_column_name].apply(lambda x: self.__get_embedding(x))
        embeddings = np.array(embeddings.tolist(), dtype=np.float64)
        veracity = df[veracity_column_name]
        # Cast veracity to float
        veracity = veracity.astype(float)

        target_classify = df[supervised_label_column_name]
        target_classify = target_classify.astype(int)

        if not self.no_umap:
            # Reduce using UMAP
            reducer = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, n_components=self.num_components,
                                n_jobs=1,
                                random_state=23 if self.random_seed else None)
            embeddings = np.array(embeddings.tolist())
            embeddings_reduced_object = None
            if supervised_umap:
                embeddings_reduced_object = reducer.fit(embeddings, y=target_classify)
            else:
                embeddings_reduced_object = reducer.fit(embeddings)
            embeddings_reduced = reducer.transform(embeddings)
            # pickle the urlmap object
            with open(
                    f"/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Models/urlmap_object_{self.time}.pkl",
                    "wb") as f:
                joblib.dump(embeddings_reduced_object, f)

        for i, embedding in (enumerate(embeddings)):
            claim_text = df[claim_column_name].iloc[i]
            veracity = float(df[veracity_column_name].iloc[i])

            if insert_into_collection:
                self.og_collection.upsert(
                    documents=[claim_text],
                    embeddings=[embedding.tolist()],
                    metadatas=[{"claim": claim_text, "veracity": veracity}],
                    ids=[claim_text]
                )

                self.reduced_collection.upsert(
                    documents=[claim_text],
                    embeddings=[embeddings_reduced[i].tolist()],
                    metadatas=[{"claim": claim_text, "veracity": veracity}],
                    ids=[claim_text]
                )

    def embed_claim_to_predict(self, claim: str, get_reduced_dimesions: bool) -> np.ndarray:
        if not get_reduced_dimesions or self.no_umap:
            return self.__get_embedding(claim)
        else:
            umap_model: UMAP = load(
                str(f'/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Models/urlmap_object_{self.time}.pkl'))
            print("Loaded UMAP model...")
            return umap_model.transform([self.__get_embedding(claim)])

    def reduce_embedding(self, embeddings: np.ndarray) -> np.ndarray:
        umap_model: UMAP = load(
            str(f'/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Models/urlmap_object_{self.time}.pkl'))
        return umap_model.transform(embeddings)
