import os
import glob
from collections import Counter
from datetime import datetime

import chromadb
import numpy as np
import umap
from matplotlib import pyplot as plt

from Clustering.Helpers.Embedder import Embedder
import hdbscan
import pandas as pd
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
from umap.parametric_umap import ParametricUMAP


class ClaimClassifier:
    model = None
    chroma_client = chromadb.PersistentClient(
        path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")

    def __init__(self, n_neighbors: int, min_dist: float, num_components: int, path_to_model: str, time_stamp: str,
                 min_cluster_size: int = 5, min_samples: int = 1):
        # Look through path_to_model and load model with the latest timestamp in the filename
        # The files will follow the naming convention hdbscan_model_%Y-%m-%d %H:%M:%S
        # list_of_files = glob.glob(os.path.join(path_to_model, 'hdbscan_model_*'))
        # Parse the timestamps from the filenames and find the file with the latest timestamp with .pkl extension
        # latest_file = max(list_of_files, key=lambda x: datetime.strptime(x.split('_')[-1], '%Y-%m-%d %H:%M:%S.pkl'))
        # Load the model from the file
        self.model = load(path_to_model + 'hdbscan_model_' + time_stamp + '.pkl')
        self.reduced_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_reduced_" + time_stamp,
        )
        self.og_collection = self.chroma_client.get_or_create_collection(
            name="climate_claims_" + time_stamp,
            metadata={"hnsw:space": "cosine"}
        )

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.time = time_stamp
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.num_components = num_components

    def classify_v2(self, claim: str, EmbeddingObject: Embedder, threshold: float, claim_check_threshold: float,
                    show_graph: bool, k: int, use_weightage: bool) -> (float, float, float):
        if threshold < claim_check_threshold:
            return None, None

        np.random.seed(11)
        temp_df = pd.DataFrame()
        temp_df["text"] = [claim]
        temp_df["veracity"] = [0]
        claims_embeddings = EmbeddingObject.embed_claim_to_predict(claim, get_reduced_dimesions=False)

        # Get the current embeddings from reduced_collection
        current_embeddings = self.og_collection.get(include=['embeddings', 'documents', 'metadatas'])
        current_embeddings_predict = np.array(current_embeddings['embeddings'])

        reducer = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.num_components, min_dist=self.min_dist,
                            random_state=23 if Embedder.random_seed else None, n_jobs=1)
        embedding_np = np.concatenate((np.array([claims_embeddings]), current_embeddings_predict), axis=0)
        embedding_np = reducer.fit_transform(embedding_np)

        # Create a dataframe from the embeddings
        claims = [claim]
        veracities = [0]
        predict = [True]
        for metadata in current_embeddings["metadatas"]:
            claims.append(metadata["claim"])
            veracities.append(metadata["veracity"])
            predict.append(False)

        embeddings = []
        for embedding in embedding_np:
            embeddings.append(embedding)

        output_df = pd.DataFrame()
        output_df["embeddings"] = embeddings
        output_df["text"] = claims
        output_df["veracity"] = veracities
        output_df["predict"] = predict

        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
                                         prediction_data=True,
                                         approx_min_span_tree=False).fit_predict(
            embeddings)
        output_df["cluster"] = hdbscan_labels

        if show_graph:
            self.__create_classify_graph_v2(output_df)

        mean, sd, confidence = self.__run_knn(output_df, "text", "cluster", "predict", k, use_weightage)

        return mean, sd, confidence

    def classify_v2_batch(self, claims: list[str], EmbeddingObject: Embedder, k: int, use_weightage: bool, supervised_umap: bool, parametric_umap: bool) -> (
            list[float], list[float], list[float]):
        temp_df = pd.DataFrame()

        claims_embeddings = []
        for claim in claims:
            claims_embeddings.append(EmbeddingObject.embed_claim_to_predict(claim, get_reduced_dimesions=False))

        # Get the current embeddings from reduced_collection
        current_embeddings = self.og_collection.get(include=['embeddings', 'documents', 'metadatas'])
        current_embeddings_predict = np.array(current_embeddings['embeddings'])

        temp_df["text"] = claims + [metadata["claim"] for metadata in current_embeddings["metadatas"]]
        temp_df["veracity"] = [-1] * len(claims) + [metadata["veracity"] for metadata in current_embeddings["metadatas"]]
        temp_df["predict"] = [True] * len(claims) + [False] * len(current_embeddings["metadatas"])

        claims_embeddings = np.array(claims_embeddings)
        # claims_embeddings = np.squeeze(claims_embeddings, axis=1)
        embedding_np = np.concatenate((claims_embeddings, current_embeddings_predict), axis=0)

        reducer = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.num_components, min_dist=self.min_dist,
                            random_state=23 if Embedder.random_seed else None, n_jobs=1)
        if parametric_umap:
            reducer = ParametricUMAP(n_neighbors=self.n_neighbors, n_components=self.num_components, min_dist=self.min_dist,
                                     random_state=23 if Embedder.random_seed else None, n_jobs=1)

        if supervised_umap:
            embedding_np = reducer.fit_transform(embedding_np, y=temp_df["veracity"].tolist())
        else:
            embedding_np = reducer.fit_transform(embedding_np)

        temp_df["embeddings"] = embedding_np.tolist()

        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
                                         prediction_data=True,
                                         approx_min_span_tree=False).fit_predict(embedding_np)
        temp_df["cluster"] = hdbscan_labels

        labels, sds, confidences = self.__run_knn(temp_df, "text", "cluster", "predict", k,
                                                  use_weightage, batch_mode=True)

        return labels, sds, confidences

    def __create_classify_graph_v2(self, claim_df: pd.DataFrame) -> None:
        # Create a graph of the clusters
        # reducer = umap.UMAP(n_components=2, random_state=23)
        reduced_embeddings_predict = np.array(claim_df["embeddings"].tolist())
        # reduced_embeddings_predict = np.array(reducer.fit_transform(reduced_embeddings_predict))

        # Create a dataframe from the embeddings
        current_df = pd.DataFrame()

        current_df["claim"] = claim_df["text"]
        current_df["x"] = reduced_embeddings_predict[:, 0]
        current_df["y"] = reduced_embeddings_predict[:, 1]
        current_df["predict"] = claim_df["predict"]
        current_df["cluster"] = claim_df["cluster"]

        # Plot the clusters. Each cluster should be a different color.
        plt.figure(figsize=(10, 10))

        # First plot the points with 'predict' being false
        scatter_false = plt.scatter(current_df[current_df['predict'] == False]['x'],
                                    current_df[current_df['predict'] == False]['y'],
                                    c=current_df[current_df['predict'] == False]['cluster'])

        # Then plot the points with 'predict' being true
        scatter_true = plt.scatter(current_df[current_df['predict'] == True]['x'],
                                   current_df[current_df['predict'] == True]['y'],
                                   c=current_df[current_df['predict'] == True]['cluster'],
                                   edgecolor='red')

        plt.title('Clusters')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(*scatter_true.legend_elements(), title="Clusters")
        plt.show()

        return

    def __run_knn(self, output_df: pd.DataFrame, claim_col: str, cluster_col: str, predict_col: str, k: int,
                  use_weightage: bool, batch_mode: bool = False):
        df_with_only_predictions = output_df[output_df[predict_col] == True]
        df_no_predictions = output_df[output_df[predict_col] == False]
        labels = []
        sds = []
        confidences_final = []

        for i in range(len(df_with_only_predictions)):
            if df_with_only_predictions.iloc[i][cluster_col] == -1:
                if batch_mode:
                    labels.append(4)
                    sds.append(0)
                    confidences_final.append(1)
                    continue
                else:
                    return 4, 0, 1
            # Filter the output_df to only include the cluster that the current claim is in
            cluster = df_with_only_predictions.iloc[i][cluster_col]
            cluster_df = df_no_predictions[df_no_predictions[cluster_col] == cluster]

            # TODO: SHOULDN't NEED THIS SET
            claims = set(cluster_df[claim_col].tolist())
            if len(claims) == 0:
                if batch_mode:
                    labels.append(4)
                    sds.append(0)
                    confidences_final.append(1)
                    continue
                else:
                    return 4, 0, 1

            # Convert set to list
            claims = list(claims)
            # Get embeddings from chroma
            cluster_data = self.reduced_collection.get(ids=claims, include=['embeddings', 'documents', 'metadatas'])

            # Extract embeddings from the cluster data
            claim_test = [data['claim'] for data in cluster_data['metadatas']]
            veracities = [data['veracity'] for data in cluster_data['metadatas']]
            embeddings = [data for data in cluster_data['embeddings']]

            # Create a KNN model
            k_neighbor_value = min(k, len(embeddings))
            if use_weightage:
                k_neighbor_value = len(embeddings)
            knn = KNeighborsClassifier(n_neighbors=k_neighbor_value, metric='cosine',
                                       weights='distance' if use_weightage else 'uniform')
            # Fit the model with the embeddings
            knn.fit(embeddings, veracities)

            # Get the current claim's embedding
            current_claim_embedding = df_with_only_predictions.iloc[i]['embeddings']

            # Get the indices of the 5 nearest neighbors
            _, indices = knn.kneighbors([current_claim_embedding])

            # Get the closest 5 claims
            closest_veracities = [veracities[index] for index in indices[0]]
            closest_veracities = np.array([float(veracity) for veracity in closest_veracities])

            label_counts = Counter(closest_veracities)
            most_common_label = label_counts.most_common(1)[0][0]

            # Calculate the average veracity for the closest claims
            average_veracity = sum(closest_veracities) / len(veracities)
            # Calculate the sd for the closest claims
            sd = np.std(closest_veracities)

            count_most_common_label = label_counts.most_common(1)[0][1]
            confidence = count_most_common_label / min(min(k, len(embeddings)), len(embeddings))

            if use_weightage:
                most_common_label = knn.predict([current_claim_embedding])[0]
                confidences = knn.predict_proba([current_claim_embedding])
                if len(confidences[0]) == 1:
                    confidence = 1
                elif int(most_common_label) == 1:
                    confidence = confidences[0][0]
                else:
                    confidence = confidences[0][1]

                if label_counts.most_common(1)[0][0] != most_common_label:
                    print(f"Predicted: {most_common_label}")
                    print(f"Confidences: {confidences}")

            if label_counts.most_common(1)[0][0] != most_common_label:
                print(f"Predicted (Non Weighted): {label_counts.most_common(1)[0][0]}")

            if not batch_mode:
                return most_common_label, sd, confidence
            else:
                labels.append(int(most_common_label))
                sds.append(sd)
                confidences_final.append(confidence)
        return labels, sds, confidences_final
