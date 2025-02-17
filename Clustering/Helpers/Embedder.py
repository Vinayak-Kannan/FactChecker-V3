import glob
import re
from datetime import datetime
from pathlib import Path

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
import hashlib
import boto3
from io import BytesIO
import json
import logging

load_dotenv()

print("=== Debug Info ===")
print("Current working directory:", os.getcwd())
print("OpenAI API Key loaded:", os.getenv("OPENAI_API_KEY") is not None)
if os.getenv("OPENAI_API_KEY") is None:
    # Try to load .env file directly
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    print("Looking for .env at:", env_path)
    print("File exists:", env_path.exists())
    if env_path.exists():
        load_dotenv(env_path)
        print("Loaded .env file, OpenAI API Key now loaded:", os.getenv("OPENAI_API_KEY") is not None)

class Embedder:
    random_seed = None
    # chroma_client = chromadb.PersistentClient(
    #     path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")
    chroma_client = None
    
    def __init__(self, n_neighbors: int, min_dist: float, num_components: int, no_umap: bool, time_stamp: str,
                 random_seed: bool = False,
                 s3_bucket: str = None,
                 existing_umap_model: umap.UMAP = None):
        # Load API key and initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        print("API Key in __init__:", bool(self.api_key))  # Debug print
        if self.api_key is None:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
        self.embedding_collection = self.pc.Index("factchecker")
        
        # Other initializations
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.num_components = num_components
        self.no_umap = no_umap
        self.time = time_stamp
        self.random_seed = random_seed

        # Load UMAP model if provided
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET", "default-bucket")
        self.umap_model = existing_umap_model
        self.current_umap_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": num_components,
            "random_state": random_seed if random_seed else None
        }

    def _get_umap_model_key(self):
        """generate a unique key for the UMAP model based on its parameters"""
        params_str = json.dumps(self.current_umap_params, sort_keys=True)
        return f"umap_models/{hashlib.md5(params_str.encode()).hexdigest()}.joblib"

    # def _save_umap_model(self):
    #     """save UMAP model to S3"""
    #     if not self.umap_model:
    #         return
    #
    #     buffer = BytesIO()
    #     joblib.dump(self.umap_model, buffer)
    #     buffer.seek(0)
    #
    #     s3 = boto3.client('s3')
    #     try:
    #         s3.upload_fileobj(buffer, self.s3_bucket, self._get_umap_model_key())
    #         logging.info(f"UMAP model saved to s3://{self.s3_bucket}/{self._get_umap_model_key()}")
    #     except Exception as e:
    #         logging.error(f"Failed to save UMAP model: {str(e)}")
    #
    # def _load_umap_model(self):
    #     """load UMAP model from S3"""
    #     s3 = boto3.client('s3')
    #     try:
    #         response = s3.get_object(Bucket=self.s3_bucket, Key=self._get_umap_model_key())
    #         buffer = BytesIO(response['Body'].read())
    #         model = joblib.load(buffer)
    #         logging.info(f"Loaded UMAP model from s3://{self.s3_bucket}/{self._get_umap_model_key()}")
    #         return model
    #     except s3.exceptions.NoSuchKey:
    #         logging.info("No existing UMAP model found")
    #         return None
    #     except Exception as e:
    #         logging.error(f"Failed to load UMAP model: {str(e)}")
    #         return None

    def get_reduced_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """apply UMAP to reduce the dimensionality of the embeddings"""
        if self.no_umap:
            return embeddings

        if embeddings.shape[1] != self.num_components and not self.no_umap:
            raise ValueError(f"Input parameters {embeddings.shape[1]} and {self.num_components} not matching")

        # check if UMAP model is already loaded
        if not self.umap_model:
            self.umap_model = self._load_umap_model()

        if not self.umap_model:
            logging.info("Training new UMAP model...")
            self.umap_model = umap.UMAP(**self.current_umap_params)
            reduced_embeddings = self.umap_model.fit_transform(embeddings)
            self._save_umap_model()
        else:
            logging.info("Using existing UMAP model")
            reduced_embeddings = self.umap_model.transform(embeddings)

        return reduced_embeddings


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
        batch_size = 50
        
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
                for text in batch_texts:
                    loc_ver = veracity[texts.index(text)]
                    self.__get_embedding(text, veracity=loc_ver)
                    success = False
                # raise ValueError("Nothing returned from Pinecone")
        
        if len(embeddings) != len(texts) and success:
            raise ValueError("Length of embeddings and texts do not match.")        

        if not success:
            print("Restarting...")
            return self.embed_claims_batch(claims, veracity)
        
        return np.array(embeddings)
      
