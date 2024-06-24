import os
import glob
from collections import Counter
from datetime import datetime

import chromadb
import numpy as np
import umap
from matplotlib import pyplot as plt
from pinecone import Pinecone, FetchResponse

from Clustering.Helpers.Embedder import Embedder
import hdbscan
import pandas as pd
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense


class ClaimClassifier:
    model = None
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
    chroma_client = chromadb.PersistentClient(
        path="./../../Clustering/Clustering/Chroma")

    def __init__(self, EmbeddingObject: Embedder, n_neighbors: int, min_dist: float, num_components: int, path_to_model: str, time_stamp: str,
                 min_cluster_size: int = 5, min_samples: int = 1,):
        self.EmbeddingObject = EmbeddingObject

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.time = time_stamp
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.num_components = num_components


    def classify_v2_batch(self, train_df: pd.DataFrame, claims: list[str], k: int, use_weightage: bool, supervised_umap: bool, parametric_umap: bool, threshold_break: float, break_further: bool, seed: int, use_hdbscan: bool, use_umap: bool) -> (
            list[float], list[float], list[float]):
        np.random.seed(seed)

        temp_df = pd.DataFrame()

        # Drop duplicates in claims
        claims = list(set(claims))
        #
        claims_embeddings = []
        for claim in claims:
            claims_embeddings.append(self.EmbeddingObject.embed_claim_to_predict(claim, get_reduced_dimesions=False))

        old_claims = train_df['Text'].tolist()
        old_veracity = train_df['Numerical Rating'].tolist()
        old_predict = [False] * len(old_claims)
        current_embeddings_predict = []
        for i in range(len(old_claims)):
            current_embeddings_predict.append(self.EmbeddingObject.embed_claim_to_predict(old_claims[i], get_reduced_dimesions=False))

        # Find indices of old_claims that are in claims and drop from old_claims, old_veracity, old_predict [likely not needed]
        indices_to_drop = []
        for i, claim in enumerate(old_claims):
            if claim in claims:
                indices_to_drop.append(i)
        old_claims = [old_claims[i] for i in range(len(old_claims)) if i not in indices_to_drop]
        old_veracity = [old_veracity[i] for i in range(len(old_veracity)) if i not in indices_to_drop]
        old_predict = [old_predict[i] for i in range(len(old_predict)) if i not in indices_to_drop]
        current_embeddings_predict = np.delete(current_embeddings_predict, indices_to_drop, axis=0)

        temp_df["text"] = claims + old_claims
        temp_df["veracity"] = [-1 for _ in range(len(claims))] + old_veracity
        temp_df["predict"] = [True for _ in range(len(claims))] + old_predict
        temp_df["predicted_veracity"] = [-1 for _ in range(len(claims))] + old_veracity

        # Drop duplicates in text column in temp_df
        temp_df = temp_df.drop_duplicates(subset=["text"])

        claims_embeddings = np.array(claims_embeddings)
        # claims_embeddings = np.squeeze(claims_embeddings, axis=1)
        embedding_np = np.concatenate((claims_embeddings, current_embeddings_predict), axis=0)

        if use_umap:
            reducer = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.num_components, min_dist=self.min_dist,
                                random_state=seed, n_jobs=1)

            y_tensor = temp_df["veracity"].astype(int).tolist()
            if parametric_umap:
                encoder = keras.Sequential([
                    keras.layers.InputLayer(input_shape=(3072, )),
                    keras.layers.Dense(units=256, activation="relu"),
                    keras.layers.Dense(units=256, activation="relu"),
                    keras.layers.Dense(units=self.num_components),
                ])
                encoder.summary()
                reducer = ParametricUMAP(encoder=encoder, dims=(3072, ), n_components=self.num_components)
                embedding_np = tf.convert_to_tensor(embedding_np)
                y_tensor = tf.convert_to_tensor(y_tensor)

            if supervised_umap:
                embedding_np = reducer.fit_transform(embedding_np, y=y_tensor)
                if parametric_umap:
                    print(reducer._history)
                    fig, ax = plt.subplots()
                    ax.plot(reducer._history['loss'])
                    ax.set_ylabel('Cross Entropy')
                    ax.set_xlabel('Epoch')
            else:
                embedding_np = reducer.fit_transform(embedding_np)

        temp_df["embeddings"] = embedding_np.tolist()

        if use_hdbscan:
            hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
                                             approx_min_span_tree=False).fit_predict(embedding_np)
            temp_df["cluster"] = hdbscan_labels
            print(temp_df.groupby(['cluster', 'predict'])['veracity'].value_counts())
            output = hdbscan_labels
            i = 0
            while break_further and i < 50:
                print("breaking further...")
                output, break_further = self.break_clusters_down(output, veracities=y_tensor, embeddings=embedding_np, threshold=threshold_break)
                # print value counts of output
                temp_df["cluster"] = output
                print(temp_df.groupby(['cluster', 'predict'])['veracity'].value_counts())
                i += 1

            # Filter temp_df where predict is False
            temp_df_no_predictions = temp_df[temp_df["predict"] == False]
            print("Number of clusters: " + str(len(temp_df_no_predictions["cluster"].value_counts())))
        else:
            temp_df["cluster"] = [1 for _ in range(len(temp_df))]

        # Drop duplicates in text column in temp_df
        temp_df = temp_df.drop_duplicates(subset=["text"])
        labels, sds, confidences, temp_df = self.__run_knn(temp_df, "text", "cluster", "predict", "embeddings", "veracity", k,
                                                  use_weightage, batch_mode=True)


        return labels, sds, confidences, temp_df

    def break_clusters_down(self, labels, veracities: list[int], embeddings: list[list[float]], threshold: float):
        if not isinstance(labels, list):
            labels = labels.tolist()
        # If there is any label with count greater than threshold except -1, run hdbscan on that cluster. Return the new labels
        label_counts = Counter(labels)
        change_needed = False
        for label, count in label_counts.items():
            if label != -1:
                # Get the veracities of the claims with the label
                veracities_with_label = [veracities[i] for i in range(len(veracities)) if labels[i] == label]
                # Calculate the percentage of veracity true (3) and false (1)
                veracity_counts = Counter(veracities_with_label)
                total = len(veracities_with_label)
                number_to_predict = veracity_counts.get(-1, 0)
                if number_to_predict == total:
                    number_to_predict = 0
                percent_true = (veracity_counts.get(3, 0) / (total - number_to_predict))

                if (percent_true < 0.5 and percent_true > 1 - threshold) or (percent_true >= 0.5 and percent_true < threshold) or count > 1000000:
                    change_needed = True
                    # Get the embeddings of the claims with the label
                    embeddings_with_label = [embeddings[i] for i in range(len(embeddings)) if labels[i] == label]
                    # Run hdbscan on the embeddings
                    hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                                     min_samples=self.min_samples,
                                                     approx_min_span_tree=False).fit_predict(embeddings_with_label)
                    # Convert hdbscan_labels to a list
                    hdbscan_labels = hdbscan_labels.tolist()
                    max_label_current = max(labels)
                    print(max_label_current)
                    new_labels = []
                    for orig_label in hdbscan_labels:
                        if orig_label != -1:
                            new_labels.append(orig_label + max_label_current + 1)
                        else:
                            new_labels.append(-1)
                    # Update the labels with the new labels. If the label number already exists in labels, assign a new label number
                    for i in range(len(labels)):
                        if labels[i] == label:
                            labels[i] = new_labels[0]
                            new_labels = new_labels[1:]

        return labels, change_needed

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

    def __run_knn(self, output_df: pd.DataFrame, claim_col: str, cluster_col: str, predict_col: str, embedding_col: str, veracity_col: str, k: int,
                  use_weightage: bool, batch_mode: bool = False):
        df_with_only_predictions = output_df[output_df[predict_col] == True]
        df_no_predictions = output_df[output_df[predict_col] == False]
        labels = []
        sds = []
        confidences_final = []

        for i in range(len(df_with_only_predictions)):
            claim = df_with_only_predictions.iloc[i][claim_col]
            index = output_df[output_df["text"] == claim].index[0]

            if df_with_only_predictions.iloc[i][cluster_col] == -1:
                if batch_mode:
                    labels.append(4)
                    sds.append(0)
                    confidences_final.append(1)
                    # Find index of claim in output_df and add predicted_veracity
                    output_df.at[index, "predicted_veracity"] = 4
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
                    labels.append(5)
                    sds.append(0)
                    confidences_final.append(1)
                    output_df.at[index, "predicted_veracity"] = 5
                    continue
                else:
                    return 4, 0, 1

            # Convert set to list
            claims = list(claims)
            # Find the embeddings and veracity of the claims in the cluster using output_df
            cluster_data = output_df[output_df[claim_col].isin(claims)]
            claim_test = [cluster_data.iloc[i][claim_col] for i in range(len(cluster_data))]
            veracities = [cluster_data.iloc[i][veracity_col] for i in range(len(cluster_data))]
            embeddings = [cluster_data.iloc[i][embedding_col] for i in range(len(cluster_data))]

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
                output_df.at[index, "predicted_veracity"] = int(most_common_label)
        return labels, sds, confidences_final, output_df
