import os
import uuid
from operator import itemgetter

import chromadb
import gensim
import nltk
import pandas as pd
from dotenv import load_dotenv
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from nltk.corpus import stopwords
from openai import OpenAI
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from Clustering.Helpers.ClusterEmbeddings import ClusterEmbeddings
from Clustering.Helpers.Embedder import Embedder
from Scraping.Helpers.ClaimClassifier import ClaimClassifier

load_dotenv()


class ClusterAndPredict:
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 1, n_neighbors: int = 200, min_dist: float = 0,
                 num_components: int = 100, no_umap: bool = False, k=5, supervised_umap: bool = False,
                 random_seed: bool = False,
                 use_weightage: bool = False,
                 supervised_label_column_name: str = 'Numerical Rating',
                 claim_column_name: str = 'Text',
                 veracity_column_name: str = 'Numerical Rating',
                 parametric_umap: bool = False,
                 train_df: pd.DataFrame = pd.DataFrame()):
        self.train_text = None
        self.EmbedderObject = None
        self.ClusterEmbeddingsObject = None
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.num_components = num_components
        self.no_umap = no_umap
        self.k = k
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.claim_column_name = claim_column_name
        self.veracity_column_name = veracity_column_name
        self.supervised_umap = supervised_umap
        self.random_seed = random_seed
        self.supervised_label_column_name = supervised_label_column_name
        self.use_weightage = use_weightage
        self.parametric_umap = parametric_umap

        self.chroma_client = chromadb.PersistentClient(
            path="/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Clustering/Chroma")

        # Randomly split the data into train and test
        self.train_df = train_df  # df.sample(frac=self.train_percentage, replace=False, random_state=23)
        # self.test_df = test_df # df.drop(self.train_df.index)
        self.clusters_df = pd.DataFrame()
        self.predicted_means = []
        self.predicted_sds = []
        self.confidences = []
        self.actual_veracities = []
        # Create uuid
        unique_uuid = uuid.uuid4()
        self.time_stamp = str(unique_uuid)

        # Performance Metrics
        self.accuracy = 0
        self.percentage_of_fours = 0
        self.accuracy_not_including_fours = 0
        self.percentage_of_no_clusters_in_ground_truth = 0
        self.ground_truth_df = pd.DataFrame()
        self.precision_on_three = 0
        self.recall_on_three = 0
        self.precision_on_three_excluding_fours = 0
        self.recall_on_three_excluding_fours = 0
        self.average_confidence_for_3 = 0

        self.accuracy_90_confidence = 0
        self.accuracy_80_confidence = 0
        self.accuracy_70_confidence = 0
        self.accuracy_60_confidence = 0
        self.percentage_90_confidence = 0
        self.percentage_80_confidence = 0
        self.percentage_70_confidence = 0
        self.percentage_60_confidence = 0

        # OpenAI
        api_key = os.getenv("OPEN_AI_KEY")
        self.client = OpenAI(api_key=api_key)

    def get_params(self, deep=True):
        return {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'num_components': self.num_components,
            'no_umap': self.no_umap,
            'claim_column_name': self.claim_column_name,
            'veracity_column_name': self.veracity_column_name,
            'train_df': self.train_df
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __cluster_ground_truth(self):
        self.EmbedderObject.embed_column_ground_truth(df=self.train_df, claim_column_name=self.claim_column_name,
                                                      veracity_column_name=self.veracity_column_name,
                                                      insert_into_collection=True, supervised_umap=self.supervised_umap,
                                                      supervised_label_column_name=self.supervised_label_column_name)
        clustered_data, self.percentage_of_no_clusters_in_ground_truth = self.ClusterEmbeddingsObject.cluster()
        self.ground_truth_df = clustered_data

    def fit(self, X: list, y: list):
        self.ClusterEmbeddingsObject = ClusterEmbeddings(min_cluster_size=self.min_cluster_size,
                                                         min_samples=self.min_samples, time_stamp=self.time_stamp)
        self.EmbedderObject = Embedder(n_neighbors=self.n_neighbors, min_dist=self.min_dist,
                                       num_components=self.num_components, no_umap=self.no_umap,
                                       time_stamp=self.time_stamp, random_seed=self.random_seed)
        self.__cluster_ground_truth()
        ClaimClassifierObject = ClaimClassifier(
            path_to_model='/Users/vinayakkannan/Desktop/Projects/FactChecker/FactChecker/Clustering/Models/',
            time_stamp=self.time_stamp, min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
            min_dist=self.min_dist, num_components=self.num_components, n_neighbors=self.n_neighbors)
        # Loop through the test data and classify each claim
        # predicted_mean = []
        # predicted_sd = []
        # predicted_confidence = []

        # cluster_df columns - text, veracity, predict, embeddings, cluster
        predicted_mean, predicted_sd, predicted_confidence, cluster_df = ClaimClassifierObject.classify_v2_batch(X,
                                                                                                                 self.EmbedderObject,
                                                                                                                 self.k,
                                                                                                                 self.use_weightage,
                                                                                                                 self.supervised_umap,
                                                                                                                 self.parametric_umap)
        self.clusters_df = cluster_df
        # for i, claim in enumerate(X):
        #     mean, sd, confidence = ClaimClassifierObject.classify_v2(claim, self.EmbedderObject, 1, 0.8, False, self.k, self.use_weightage)
        #     if confidence < 1:
        #         print(f"Actual: {y[i]}")
        #     predicted_mean.append(int(mean))
        #     predicted_sd.append(sd)
        #     predicted_confidence.append(confidence)

        self.predicted_means = predicted_mean
        self.predicted_sds = predicted_sd
        self.confidences = predicted_confidence

        self.train_text = X
        self.actual_veracities = y

    def score(self, X_test, y_test):
        # mse = mean_squared_error(self.actual_veracities, self.predicted_means)
        accuracy = metrics.accuracy_score(self.actual_veracities, self.predicted_means)
        # self.chroma_client.delete_collection('climate_claims_' + self.time_stamp)
        # self.chroma_client.delete_collection('climate_claims_reduced_' + self.time_stamp)
        # Loop through predicted_means and count the number of 4's
        num_of_fours = self.predicted_means.count(4)
        print(self.predicted_means.count(5))
        self.accuracy = accuracy
        self.percentage_of_fours = num_of_fours / len(self.predicted_means)

        accuracy_not_including_fours_values_count_correct = 0
        accuracy_not_including_fours_values_count_total = 0


        for i, value in enumerate(self.predicted_means):
            claim = self.train_text[i]
            if not self.clusters_df.loc[self.clusters_df['text'] == claim, 'predict'].values[0]:
                raise ValueError("Predicted value is False")
            cluster = self.clusters_df.loc[self.clusters_df['text'] == claim, 'cluster'].values[0]
            if len(self.clusters_df.loc[self.clusters_df['text'] == claim, 'text'].values) > 1:
                # print(self.clusters_df.loc[self.clusters_df['text'] == claim, 'text'].values)
                # print(self.clusters_df.loc[self.clusters_df['text'] == claim, 'veracity'].values)
                print("Yeah this shouldn't happen lol")
            value = self.clusters_df.loc[self.clusters_df['text'] == claim, 'predicted_veracity'].values[0]
            self.clusters_df.loc[self.clusters_df['text'] == claim, 'veracity'] = self.actual_veracities[i]
            # Note, we use 5 for claims where they are to be predicted and clustered on themselves in the df
            if value == 4 and cluster != -1:
                raise ValueError("Cluster should be -1 if predicted value is 4")

            if value != 4:
                accuracy_not_including_fours_values_count_total += 1
                if self.actual_veracities[i] == value:
                    accuracy_not_including_fours_values_count_correct += 1

        if accuracy_not_including_fours_values_count_total == 0:
            self.accuracy_not_including_fours = 0
        else:
            self.accuracy_not_including_fours = accuracy_not_including_fours_values_count_correct / accuracy_not_including_fours_values_count_total

        # Fill in cluster accuracy and number wrong in clusters_df
        self.clusters_df['num_correct_in_cluster'] = 0
        self.clusters_df['total_in_cluster'] = 0
        self.clusters_df['cluster_accuracy'] = 0
        self.clusters_df['num_correct_in_cluster'] = (
                    (self.clusters_df['veracity'] == self.clusters_df['predicted_veracity']))
        self.clusters_df['num_correct_in_cluster'] = self.clusters_df.groupby('cluster')[
            'num_correct_in_cluster'].transform('sum')
        self.clusters_df['total_in_cluster'] = self.clusters_df.groupby('cluster')['cluster'].transform('size')
        # Fill Nan in total_in_cluster with 1
        self.clusters_df['total_in_cluster'] = self.clusters_df['total_in_cluster'].fillna(0)
        self.clusters_df['cluster_accuracy'] = self.clusters_df['num_correct_in_cluster'].div(self.clusters_df['total_in_cluster'], fill_value=1)
        # In cluster_accuracy, fill inf or Nan with 1
        self.clusters_df['cluster_accuracy'] = self.clusters_df['cluster_accuracy'].replace([float('inf'), float('nan')], 1)

        # Calculate precision and recall for veracity 3
        precision = 0
        recall = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i, value in enumerate(self.predicted_means):
            if value == 3:
                if self.actual_veracities[i] == 3:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if self.actual_veracities[i] == 3:
                    false_negatives += 1

        if true_positives + false_positives != 0:
            precision = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives != 0:
            recall = true_positives / (true_positives + false_negatives)

        self.precision_on_three = precision
        self.recall_on_three = recall

        precision_no_fours = 0
        recall_no_fours = 0
        true_positives_no_fours = 0
        false_positives_no_fours = 0
        false_negatives_no_fours = 0
        average_confidence_for_3 = []

        for i, value in enumerate(self.predicted_means):
            if value == 3:
                average_confidence_for_3.append(self.confidences[i])
                if self.actual_veracities[i] == 3:
                    true_positives_no_fours += 1
                else:
                    false_positives_no_fours += 1
            elif value == 1:
                if self.actual_veracities[i] == 3:
                    false_negatives_no_fours += 1

        if len(average_confidence_for_3) > 0:
            self.average_confidence_for_3 = sum(average_confidence_for_3) / len(average_confidence_for_3)

        if true_positives_no_fours + false_positives_no_fours != 0:
            precision_no_fours = true_positives_no_fours / (true_positives_no_fours + false_positives_no_fours)
        if true_positives_no_fours + false_negatives_no_fours != 0:
            recall_no_fours = true_positives_no_fours / (true_positives_no_fours + false_negatives_no_fours)
            print(true_positives_no_fours, false_positives_no_fours, (true_positives_no_fours + false_negatives_no_fours))
            print(recall_no_fours, 1 / (1 + 0), true_positives_no_fours / (true_positives_no_fours + false_negatives_no_fours))

        self.precision_on_three_excluding_fours = precision_no_fours
        self.recall_on_three_excluding_fours = recall_no_fours

        # Store confidence scores and report accuracy based on confidence. 5 means confidence not high enough
        predictions_90_confidence = []
        predictions_80_confidence = []
        predictions_70_confidence = []
        predictions_60_confidence = []
        for i, value in enumerate(self.predicted_means):
            if self.confidences[i] >= 0.9:
                predictions_90_confidence.append(value)
            else:
                predictions_90_confidence.append(5)

            if self.confidences[i] >= 0.8:
                predictions_80_confidence.append(value)
            else:
                predictions_80_confidence.append(5)

            if self.confidences[i] >= 0.7:
                predictions_70_confidence.append(value)
            else:
                predictions_70_confidence.append(5)

            if self.confidences[i] >= 0.6:
                predictions_60_confidence.append(value)
            else:
                predictions_60_confidence.append(5)

        self.accuracy_90_confidence = metrics.accuracy_score(self.actual_veracities, predictions_90_confidence)
        self.accuracy_80_confidence = metrics.accuracy_score(self.actual_veracities, predictions_80_confidence)
        self.accuracy_70_confidence = metrics.accuracy_score(self.actual_veracities, predictions_70_confidence)
        self.accuracy_60_confidence = metrics.accuracy_score(self.actual_veracities, predictions_60_confidence)

        self.percentage_90_confidence = 1 - (predictions_90_confidence.count(5) / len(predictions_90_confidence))
        self.percentage_80_confidence = 1 - (predictions_80_confidence.count(5) / len(predictions_80_confidence))
        self.percentage_70_confidence = 1 - (predictions_70_confidence.count(5) / len(predictions_70_confidence))
        self.percentage_60_confidence = 1 - (predictions_60_confidence.count(5) / len(predictions_60_confidence))

        return (0.5 * self.precision_on_three + 0.5 * self.recall_on_three) / 0.5

    def print_all_performance_metrics(self) -> None:
        self.score([], [])
        print(self.actual_veracities)
        print(self.predicted_means)
        print(f'Accuracy: {self.accuracy}')
        print(f'Accuracy not including no clusters in test data: {self.accuracy_not_including_fours}')
        print(f'Percentage of no clusters in test data: {self.percentage_of_fours}')
        print(f'Percentage of no clusters in ground truth: {self.percentage_of_no_clusters_in_ground_truth}')
        print(f'Precision on veracity 3: {self.precision_on_three}')
        print(f'Recall on veracity 3: {self.recall_on_three}')
        print('Average confidence for 3: ', self.average_confidence_for_3)
        print(f'Precision on veracity 3 excluding 4s: {self.precision_on_three_excluding_fours}')
        print(f'Recall on veracity 3 excluding 4s: {self.recall_on_three_excluding_fours}')
        print(f'Accuracy at 90% confidence: {self.accuracy_90_confidence}')
        print(f'Accuracy at 80% confidence: {self.accuracy_80_confidence}')
        print(f'Accuracy at 70% confidence: {self.accuracy_70_confidence}')
        print(f'Accuracy at 60% confidence: {self.accuracy_60_confidence}')
        print(f'Percentage of 90% confidence: {self.percentage_90_confidence}')
        print(f'Percentage of 80% confidence: {self.percentage_80_confidence}')
        print(f'Percentage of 70% confidence: {self.percentage_70_confidence}')
        print(f'Percentage of 60% confidence: {self.percentage_60_confidence}')

    def get_all_performance_metrics(self) -> object:
        self.score([], [])
        return {
            'accuracy': self.accuracy,
            'accuracy_not_including_fours': self.accuracy_not_including_fours,
            'percentage_of_fours': self.percentage_of_fours,
            'percentage_of_no_clusters_in_ground_truth': self.percentage_of_no_clusters_in_ground_truth,
            'precision_on_three': self.precision_on_three,
            'recall_on_three': self.recall_on_three,
            'average_confidence_for_3': self.average_confidence_for_3,
            'precision_on_three_excluding_fours': self.precision_on_three_excluding_fours,
            'recall_on_three_excluding_fours': self.recall_on_three_excluding_fours,
            'accuracy_90_confidence': self.accuracy_90_confidence,
            'accuracy_80_confidence': self.accuracy_80_confidence,
            'accuracy_70_confidence': self.accuracy_70_confidence,
            'accuracy_60_confidence': self.accuracy_60_confidence,
            'percentage_90_confidence': self.percentage_90_confidence,
            'percentage_80_confidence': self.percentage_80_confidence,
            'percentage_70_confidence': self.percentage_70_confidence,
            'percentage_60_confidence': self.percentage_60_confidence,
            'cluster_df': self.clusters_df,
        }

    def plot_confidence_histogram(self) -> None:
        # Filter out indices where the predicted mean is 4 from the confidence list
        confidence_filtered = [self.confidences[i] for i, value in enumerate(self.predicted_means) if value != 4]
        # Plot the histogram
        import matplotlib.pyplot as plt
        plt.hist(confidence_filtered, bins=10)
        plt.show()
        return

    def sent_to_words(self, sentences):
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        for sentence in sentences:
            yield [word for word in gensim.utils.simple_preprocess(str(sentence), deacc=True) if
                   word not in stop_words]

    def report_ground_truth_data(self, create_cluster_labels: bool = False) -> (pd.DataFrame, pd.DataFrame):
        # Ground truth data has columns 'cluster', 'claim', 'veracity'
        # Create a summarized dataframe for each 'cluster' and the count of true / false in each cluster
        # Create a dataframe to store the counts
        cluster_counts = pd.DataFrame()
        cluster_counts['cluster'] = self.ground_truth_df['cluster'].unique()
        cluster_counts['count_false'] = 0
        cluster_counts['count_true'] = 0
        # Loop through the ground truth dataframe and count the number of true and false in each cluster
        for i in range(len(self.ground_truth_df)):
            if self.ground_truth_df.iloc[i]['veracity'] == 4:
                continue
            cluster = self.ground_truth_df.iloc[i]['cluster']
            if self.ground_truth_df.iloc[i]['veracity'] == 3:
                cluster_counts.loc[cluster_counts['cluster'] == cluster, 'count_true'] += 1
            else:
                cluster_counts.loc[cluster_counts['cluster'] == cluster, 'count_false'] += 1

        if not create_cluster_labels:
            return self.ground_truth_df, cluster_counts

        self.ground_truth_df['tfidf_top_3'] = ""
        self.ground_truth_df['cluster_label'] = ""

        # Create documents, where each document is all the claims in a cluster concatenated together
        documents = []
        cluster_nums = []
        for cluster in cluster_counts['cluster']:
            cluster_nums.append(cluster)
            claims = self.ground_truth_df[self.ground_truth_df['cluster'] == cluster]['claim']
            documents.append(" ".join(claims))

        # Tokenize the documents
        tokenized_docs = list(self.sent_to_words(documents))

        # Remove stopwords and words that appear only once
        dictionary = gensim.corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

        # Step 3: Train the TF-IDF model
        tfidf = TfidfModel(corpus, smartirs='ntc')

        # Step 4: Apply the model to the entire corpus
        corpus_tfidf = tfidf[corpus]

        results = []
        # Step 5: Find the terms with the highest TF-IDF score in each document
        for doc in corpus_tfidf:
            sorted_doc = sorted(doc, key=itemgetter(1), reverse=True)
            # Get the top 3 terms with highest TF-IDF scores
            top_terms = sorted_doc[:10]
            cluster_local = cluster_nums.pop(0)
            # Append the top 3 terms to the tfidf_top_3 column in the ground truth dataframe
            self.ground_truth_df.loc[self.ground_truth_df['cluster'] == cluster_local, 'tfidf_top_3'] = \
                f"{dictionary[top_terms[0][0]]}, {dictionary[top_terms[1][0]]}, {dictionary[top_terms[2][0]]}, {dictionary[top_terms[3][0]]}, {dictionary[top_terms[4][0]]}"

            # Number of claims in the cluster
            len_cluster = self.ground_truth_df[self.ground_truth_df['cluster'] == cluster_local].shape[0]
            print(dictionary[top_terms[0][0]])

            result = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f""" Write a one sentence description for a topic cluster of claims that is as 
                        different as possible from other topic clusters. All topic clusters are related to climate 
                        change. Your topic cluster name should be as different from the other topic clusters which are
                        listed below:
                        {", ".join(results)}
                        
                        Up to 10 of the cluster's claims are: 
                        {", ".join(self.ground_truth_df[self.ground_truth_df['cluster'] == cluster_local]['claim'].sample(
                            min(10, len_cluster)).tolist())
                        }
                        
                        The top 5 terms with the highest TF-IDF scores you should use in the description for 
                        this cluster are: {dictionary[top_terms[0][0]]}, {dictionary[top_terms[1][0]]}, 
                        {dictionary[top_terms[2][0]]}, {dictionary[top_terms[3][0]]}, {dictionary[top_terms[4][0]]}
                        
                        Your one sentence description should be at most 10 words and use the TF-IDF terms. Do not write more than 10 words.
                        """
                    },
                ]
            ).choices[0].message.content

            results.append(result)

            self.ground_truth_df.loc[self.ground_truth_df['cluster'] == cluster_local, 'cluster_label'] = result

        return self.ground_truth_df, cluster_counts

    def get_accuracy(self):
        self.score([], [])
        return self.accuracy

    def get_was_supervised(self):
        return self.supervised_umap
