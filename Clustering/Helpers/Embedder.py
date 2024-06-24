import glob
import re
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
from pinecone import Pinecone, ServerlessSpec, FetchResponse

from umap import UMAP

load_dotenv()


class Embedder:
    random_seed = None
    api_key = os.getenv("OPEN_AI_KEY")
    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(
        path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

    def __init__(self, n_neighbors: int, min_dist: float, num_components: int, no_umap: bool, time_stamp: str,
                 random_seed: bool = False):
        self.embedding_collection = self.pc.Index("factchecker")
        # self.embedding_collection_namespace = "climate_claims_embeddings"
        # self.embedding_collection = self.chroma_client.get_collection(
        #     name="climate_claims_embeddings_unchanged"
        # )

        # self.og_collection = self.pc.Index("climate-claims-" + time_stamp)
        # self.reduced_collection = self.pc.Index("climate-claims-reduced-" + time_stamp)

        # self.og_collection = self.chroma_client.get_or_create_collection(
        #     name="climate_claims_" + time_stamp,
        #     metadata={"hnsw:space": "cosine"}
        # )
        # self.reduced_collection = self.chroma_client.get_or_create_collection(
        #     name="climate_claims_reduced_" + time_stamp,
        #     metadata={"hnsw:space": "cosine"}
        # )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.num_components = num_components
        self.no_umap = no_umap
        self.time = time_stamp
        self.random_seed = random_seed

    def format_text(self, text: str) -> str:
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.replace(" ", "_")
        # Substring text to the first 500 characters
        text = text[:500]
        return text

    def __get_embedding(self, text: str) -> np.ndarray:
        model = "text-embedding-3-large"
        og_text = text
        text = self.format_text(text)
        if len(text) > 500:
            raise ValueError("Text is too long. Please provide a text with less than 500 characters.")
        query_original: FetchResponse = self.embedding_collection.fetch(ids=[text])
        query_original = query_original['vectors']
        if query_original and len(query_original.keys()) > 0:
            return np.array(query_original[text]['values'])
        else:
            print(text)
            print("Had to create embedding...")
            # raise ValueError("Should not be here...")
            embedding_collection_old = self.chroma_client.get_collection(
                name="climate_claims_embeddings_unchanged"
            )
            old_data = embedding_collection_old.get(ids=[og_text], include=['embeddings', 'documents', 'metadatas'])
            train_df = pd.read_csv('/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Transformed Data/train_df.csv')
            test_df = pd.read_csv('/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Transformed Data/test_df.csv')

            # Ground Truth
            ground_truth = pd.read_csv(
                "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/VKs Copy of Cleaned - Google Fact Check Explorer - Climate.xlsx - Corrected.csv")
            ground_truth = ground_truth.dropna(subset=['Text'])
            ground_truth = ground_truth[ground_truth['Text'] != '']
            ground_truth = ground_truth.drop_duplicates(subset=['Text'])
            ground_truth['Synthetic'] = [False for i in range(len(ground_truth))]

            # EPA
            epa_who_data = pd.read_csv(
                "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/climate_change_epa_who.csv")
            epa_who_data['Category'] = -1
            epa_who_data['Numerical Rating'] = 3
            epa_who_data = epa_who_data.dropna(subset=['Text'])
            epa_who_data = epa_who_data[epa_who_data['Text'] != '']
            epa_who_data = epa_who_data.drop_duplicates(subset=['Text'])
            epa_who_data['Synthetic'] = [False for i in range(len(epa_who_data))]

            # Card Data
            card_data = pd.read_csv(
                "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/card_train_with_score.csv")
            card_data['claim'] = '1_1'
            card_data = card_data.rename(columns={'text': 'Text', 'claim': 'Category'})
            # To the card_data, add a 'Numerical Rating' column with value 1
            card_data['Numerical Rating'] = 1
            card_data = card_data.dropna(subset=['Text'])
            card_data = card_data[card_data['Text'] != '']
            card_data = card_data[card_data['score'] >= 0.8]
            card_data = card_data.drop_duplicates(subset=['Text'])
            card_data['Synthetic'] = [False for i in range(len(card_data))]

            # Concat train and test df
            df = pd.concat([train_df, test_df, ground_truth, epa_who_data, card_data])
            # Drop duplicates in the df
            df = df.drop_duplicates(subset=['Text'])
            # Filter df and get Numerical Rating for the claim
            df = df[df['Text'] == og_text]
            # Get the numerical rating
            rating = int(df['Numerical Rating'].values[0])
            self.embedding_collection.upsert(
                vectors=[{
                    "id": text,
                    "values": old_data['embeddings'][0],
                    "metadata": {
                        "claim": og_text,
                        "veracity": rating
                    }
                }],
            )
            #
            #
            # result = self.client.embeddings.create(input=[text], model=model).data[
            #     0].embedding  # extra_body={'dimensions': 256}
            # self.embedding_collection.upsert(
            #     documents=[text],
            #     embeddings=[result],
            #     metadatas=[{"claim": text}],
            #     ids=[text]
            # )
            return np.array(old_data['embeddings'][0])

    def embed_column_ground_truth(self, df: pd.DataFrame, claim_column_name: str, veracity_column_name: str,
                                  supervised_label_column_name: str,
                                  insert_into_collection: bool = True, supervised_umap: bool = False) -> None:
        embeddings = df[claim_column_name].apply(lambda x: self.__get_embedding(x))
        embeddings = np.array(embeddings.tolist(), dtype=np.float64)
        veracity = df[veracity_column_name]
        # Cast veracity to float
        veracity = veracity.astype(float)

        target_classify = df[supervised_label_column_name]
        target_classify = np.array(target_classify.astype(int))

        embeddings_reduced = embeddings

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
            embeddings_reduced = embeddings_reduced_object.transform(embeddings)
            # pickle the urlmap object
            # with open(
            #         f"../../Models/urlmap_object_{self.time}.pkl",
            #         "wb") as f:
            #     joblib.dump(embeddings_reduced_object, f)

        for i, embedding in (enumerate(embeddings)):
            claim_text = df[claim_column_name].iloc[i]
            veracity = float(df[veracity_column_name].iloc[i])

            if insert_into_collection:
                claim_text_clean = self.format_text(claim_text)
                # self.og_collection.upsert(
                #     vectors=[
                #         {
                #             "id": claim_text_clean,
                #             "values": embedding.tolist(),
                #             "metadata": {"claim": claim_text, "veracity": veracity}
                #          },
                #     ],
                #     # namespace=self.embedding_collection_namespace + self.time
                # )
                #
                # self.reduced_collection.upsert(
                #     vectors=[
                #         {
                #             "id": claim_text_clean,
                #             "values": embeddings_reduced[i].tolist(),
                #             "metadata": {"claim": claim_text, "veracity": veracity}
                #         }
                #     ],
                #     # namespace=self.embedding_collection_namespace + self.time
                # )

                #
                # self.og_collection.upsert(
                #     documents=[claim_text],
                #     embeddings=[embedding.tolist()],
                #     metadatas=[{"claim": claim_text, "veracity": veracity}],
                #     ids=[claim_text]
                # )
                #
                # self.reduced_collection.upsert(
                #     documents=[claim_text],
                #     embeddings=[embeddings_reduced[i].tolist()],
                #     metadatas=[{"claim": claim_text, "veracity": veracity}],
                #     ids=[claim_text]
                # )

    def embed_claim_to_predict(self, claim: str, get_reduced_dimesions: bool) -> np.ndarray:
        if not get_reduced_dimesions or self.no_umap:
            return self.__get_embedding(claim)
        # else:
        #     umap_model: UMAP = load(
        #         str(f'../../Models/urlmap_object_{self.time}.pkl'))
        #     print("Loaded UMAP model...")
        #     return umap_model.transform([self.__get_embedding(claim)])

    # def reduce_embedding(self, embeddings: np.ndarray) -> np.ndarray:
    #     umap_model: UMAP = load(
    #         str(f'../../Models/urlmap_object_{self.time}.pkl'))
    #     return umap_model.transform(embeddings)
