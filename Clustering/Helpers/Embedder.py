import glob
import re
from datetime import datetime

import joblib
import numpy as np
import umap
from joblib import load
from openai import OpenAI
import pandas as pd
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
    # chroma_client = chromadb.PersistentClient(
    #     path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")
    chroma_client = None
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

    def __init__(self, n_neighbors: int, min_dist: float, num_components: int, no_umap: bool, time_stamp: str,
                 random_seed: bool = False):
        self.embedding_collection = self.pc.Index("factchecker")
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

    def __get_embedding(self, text: str, veracity: int) -> np.ndarray:
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
            result = self.client.embeddings.create(input=[text], model=model).data[0].embedding
            print("Uploading claim to pinecone: ", text)
            self.embedding_collection.upsert(
                vectors=[{
                    "id": text,
                    "values": np.array(result),
                    "metadata": {
                        "claim": og_text,
                        "veracity": veracity
                    }
                }],
            )
            return np.array(result)
            # raise ValueError("Had to create embedding...")
            # This code is commented out to prevent the creation of embeddings. This may be needed in the future.
            # embedding_collection_old = self.chroma_client.get_collection(
            #     name="climate_claims_embeddings_unchanged"
            # )
            # old_data = embedding_collection_old.get(ids=[og_text], include=['embeddings', 'documents', 'metadatas'])
            # train_df = pd.read_csv('/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Transformed Data/train_df.csv')
            # test_df = pd.read_csv('/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Transformed Data/test_df.csv')
            #
            # # Ground Truth
            # ground_truth = pd.read_csv(
            #     "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/VKs Copy of Cleaned - Google Fact Check Explorer - Climate.xlsx - Corrected.csv")
            # ground_truth = ground_truth.dropna(subset=['Text'])
            # ground_truth = ground_truth[ground_truth['Text'] != '']
            # ground_truth = ground_truth.drop_duplicates(subset=['Text'])
            # ground_truth['Synthetic'] = [False for i in range(len(ground_truth))]
            #
            # # EPA
            # epa_who_data = pd.read_csv(
            #     "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/climate_change_epa_who.csv")
            # epa_who_data['Category'] = -1
            # epa_who_data['Numerical Rating'] = 3
            # epa_who_data = epa_who_data.dropna(subset=['Text'])
            # epa_who_data = epa_who_data[epa_who_data['Text'] != '']
            # epa_who_data = epa_who_data.drop_duplicates(subset=['Text'])
            # epa_who_data['Synthetic'] = [False for i in range(len(epa_who_data))]
            #
            # # Card Data
            # card_data = pd.read_csv(
            #     "/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Raw Data/Climate/card_train_with_score.csv")
            # card_data['claim'] = '1_1'
            # card_data = card_data.rename(columns={'text': 'Text', 'claim': 'Category'})
            # # To the card_data, add a 'Numerical Rating' column with value 1
            # card_data['Numerical Rating'] = 1
            # card_data = card_data.dropna(subset=['Text'])
            # card_data = card_data[card_data['Text'] != '']
            # card_data = card_data[card_data['score'] >= 0.8]
            # card_data = card_data.drop_duplicates(subset=['Text'])
            # card_data['Synthetic'] = [False for i in range(len(card_data))]
            #
            # # Concat train and test df
            # df = pd.concat([train_df, test_df, ground_truth, epa_who_data, card_data])
            # # Drop duplicates in the df
            # df = df.drop_duplicates(subset=['Text'])
            # # Filter df and get Numerical Rating for the claim
            # df = df[df['Text'] == og_text]
            # # Get the numerical rating
            # rating = int(df['Numerical Rating'].values[0])
            # self.embedding_collection.upsert(
            #     vectors=[{
            #         "id": text,
            #         "values": old_data['embeddings'][0],
            #         "metadata": {
            #             "claim": og_text,
            #             "veracity": rating
            #         }
            #     }],
            # )
            # #
            # #
            # # result = self.client.embeddings.create(input=[text], model=model).data[
            # #     0].embedding  # extra_body={'dimensions': 256}
            # # self.embedding_collection.upsert(
            # #     documents=[text],
            # #     embeddings=[result],
            # #     metadatas=[{"claim": text}],
            # #     ids=[text]
            # # )
            # return np.array(old_data['embeddings'][0])

    def embed_claim_to_predict(self, claim: str, get_reduced_dimesions: bool, veracity: int) -> np.ndarray:
        if not get_reduced_dimesions or self.no_umap:
            return self.__get_embedding(claim, veracity)
        

    def embed_claims_batch(self, claims: list, veracity: list) -> np.ndarray:
        embeddings = []
        texts = []
        batch_size = 75
        
        # Prepare texts
        for claim, ver in zip(claims, veracity):
            text = self.format_text(claim)
            texts.append(text)
        
        # Fetch embeddings in batches
        success = True
        for i in range(0, len(texts), batch_size):
            print("Getting embeddings for batch ", i, " out of ", len(texts))
            batch_texts = texts[i:i + batch_size]
            query_original: FetchResponse = self.embedding_collection.fetch(ids=batch_texts)
            query_original = query_original['vectors']
            
            if query_original and len(query_original.keys()) > 0:
                for text in batch_texts:
                    if text not in query_original:
                        # Get veracity based on index in texts
                        loc_ver = veracity[texts.index(text)]
                        self.__get_embedding(text, veracity=loc_ver)
                        # Break and restart the loop
                        success = False
                    else:
                        embeddings.append(np.array(query_original[text]['values']))
            else:
                raise ValueError("Nothing returned from Pinecone")
        
        if len(embeddings) != len(texts) and success:
            raise ValueError("Length of embeddings and texts do not match.")        

        if not success:
            print("Restarting...")
            return self.embed_claims_batch(claims, veracity)
        
        return np.array(embeddings)
      
