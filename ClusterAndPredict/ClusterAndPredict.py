import os
import uuid
from operator import itemgetter

import gensim
import nltk
import pandas as pd
from dotenv import load_dotenv
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from nltk.corpus import stopwords
from openai import OpenAI
from pinecone import ServerlessSpec, Pinecone
from sklearn import metrics

from Clustering.Helpers.Embedder import Embedder
from Scraping.Helpers.ClaimClassifier import ClaimClassifier

load_dotenv()

"""
1 - False
3 - True
4 - No prediction
5 - Predicts fell into their own cluster only. No predicts
"""

class ClusterAndPredict:
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 1, n_neighbors: int = 200, min_dist: float = 0,
                 num_components: int = 100, no_umap: bool = False, k=5, supervised_umap: bool = False,
                 random_seed: bool = False,
                 use_weightage: bool = False,
                 supervised_label_column_name: str = 'Numerical Rating',
                 claim_column_name: str = 'Text',
                 veracity_column_name: str = 'Numerical Rating',
                 parametric_umap: bool = False,
                 threshold_break: float = 0.8,
                 break_further: bool = True,
                 random_seed_val: int = 23,
                 use_hdbscan: bool = True,
                 train_df: pd.DataFrame = pd.DataFrame()):
        pd.set_option('future.no_silent_downcasting', True)
        self.test_text = None
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
        self.random_seed_val = random_seed_val
        self.use_hdbscan = use_hdbscan

        # self.chroma_client = chromadb.PersistentClient(
        #     path="./../../Clustering/Clustering/Chroma")
        self.chroma_client = None

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
        # Filter out non-numeric values in uuid
        self.time_stamp = ''.join(filter(lambda x: x.isdigit(), self.time_stamp))


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
        self.precision_on_one = 0
        self.recall_on_one = 0
        self.precision_on_one_excluding_fours = 0
        self.recall_on_one_excluding_fours = 0
        self.precision = 0
        self.recall = 0
        self.precision_no_fours, self.recall_no_fours = 0, 0
        self.average_confidence_for_3 = 0

        self.accuracy_90_confidence = 0
        self.accuracy_80_confidence = 0
        self.accuracy_70_confidence = 0
        self.accuracy_60_confidence = 0
        self.percentage_90_confidence = 0
        self.percentage_80_confidence = 0
        self.percentage_70_confidence = 0
        self.percentage_60_confidence = 0

        # Recursive Breaking
        self.threshold_break = threshold_break
        self.break_further = break_further

        # OpenAI
        api_key = os.getenv("OPEN_AI_KEY")
        self.client = OpenAI(api_key=api_key)

        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))


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
            'threshold_break': self.threshold_break,
            'break_further': self.break_further,
            'train_df': self.train_df
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X: list, y: list):
        
        self.EmbedderObject = Embedder(n_neighbors=self.n_neighbors, min_dist=self.min_dist,
                                       num_components=self.num_components, no_umap=self.no_umap,
                                       time_stamp=self.time_stamp, random_seed=self.random_seed)
        # self.__cluster_ground_truth()
        ClaimClassifierObject = ClaimClassifier(
            EmbeddingObject=self.EmbedderObject,
            path_to_model='../../Clustering/Models/',
            time_stamp=self.time_stamp, min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
            min_dist=self.min_dist, num_components=self.num_components, n_neighbors=self.n_neighbors)

        print("Fitting")
        # cluster_df columns - text, veracity, predict, predicted_veracity, embeddings, cluster
        predicted_mean, predicted_sd, predicted_confidence, cluster_df = ClaimClassifierObject.classify_v2_batch(
         self.train_df,
         X,
         y,
         self.k,
         self.use_weightage,
         self.supervised_umap,
         self.parametric_umap,
         self.threshold_break,
         self.break_further,
         self.random_seed_val,
         self.use_hdbscan,
         not self.no_umap
        )
        self.clusters_df = cluster_df

        self.predicted_means = predicted_mean
        self.predicted_sds = predicted_sd
        self.confidences = predicted_confidence

        self.test_text = X
        self.actual_veracities = y

    def score(self, _, __):
        self.accuracy = self.calculate_accuracy(self.clusters_df)
        self.accuracy_not_including_fours = self.calculate_accuracy_excluding_no_predict(self.clusters_df)
        self.percentage_of_fours, self.percentage_of_no_clusters_in_ground_truth = self.calculate_percentage_of_four_and_five(self.clusters_df)


        for i, value in enumerate(self.predicted_means):
            claim = self.test_text[i]
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
            # if value == 4 and cluster != -1:
            #     raise ValueError("Cluster should be -1 if predicted value is 4")


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

        # clusters_df columsn - text, veracity, predict, predicted_veracity, embeddings, cluster, num_correct_in_cluster, total_in_cluster, cluster_accuracy
        self.precision_on_three, self.recall_on_three = self.calculate_precision_recall_for_a_value(self.clusters_df, 3)
        self.precision_on_one, self.recall_on_one = self.calculate_precision_recall_for_a_value(self.clusters_df, 1)
        self.precision, self.recall = self.calculate_precision_recall(self.clusters_df)
        self.precision_on_three_excluding_fours, self.recall_on_three_excluding_fours = self.calculate_precision_recall_for_three_excluding_no_predict(self.clusters_df)
        self.precision_on_one_excluding_fours, self.recall_on_one_excluding_fours = self.calculate_precision_recall_for_one_excluding_no_predict(self.clusters_df)
        self.precision_no_fours, self.recall_no_fours = self.calculate_weighted_precision_recall_excluding_no_predict(self.clusters_df)

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

        return (0.5 * self.precision_on_three_excluding_fours + 0.5 * self.recall_on_three_excluding_fours) / 0.5

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
        # print(f'Accuracy at 90% confidence: {self.accuracy_90_confidence}')
        # print(f'Accuracy at 80% confidence: {self.accuracy_80_confidence}')
        # print(f'Accuracy at 70% confidence: {self.accuracy_70_confidence}')
        # print(f'Accuracy at 60% confidence: {self.accuracy_60_confidence}')
        # print(f'Percentage of 90% confidence: {self.percentage_90_confidence}')
        # print(f'Percentage of 80% confidence: {self.percentage_80_confidence}')
        # print(f'Percentage of 70% confidence: {self.percentage_70_confidence}')
        # print(f'Percentage of 60% confidence: {self.percentage_60_confidence}')

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
            'precision_on_one': self.precision_on_one,
            'recall_on_one': self.recall_on_one,
            'precision_on_one_excluding_fours': self.precision_on_one_excluding_fours,
            'recall_on_one_excluding_fours': self.recall_on_one_excluding_fours,
            'precision': self.precision,
            'recall': self.recall,
            'precision_no_fours': self.precision_no_fours,
            'recall_no_fours': self.recall_no_fours,
            # 'accuracy_90_confidence': self.accuracy_90_confidence,
            # 'accuracy_80_confidence': self.accuracy_80_confidence,
            # 'accuracy_70_confidence': self.accuracy_70_confidence,
            # 'accuracy_60_confidence': self.accuracy_60_confidence,
            # 'percentage_90_confidence': self.percentage_90_confidence,
            # 'percentage_80_confidence': self.percentage_80_confidence,
            # 'percentage_70_confidence': self.percentage_70_confidence,
            # 'percentage_60_confidence': self.percentage_60_confidence,
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

    # Helper functions for metrics
    def calculate_accuracy(self, cluster_df):
        # clusters_df columns - text, veracity, predict, predicted_veracity, embeddings, cluster, num_correct_in_cluster, total_in_cluster, cluster_accuracy
        num_correct = 0

        cluster_df = cluster_df[cluster_df['predict']]

        for _, row in cluster_df.iterrows():
            if int(row['veracity']) == int(row['predicted_veracity']):
                num_correct += 1

        if len(cluster_df) == 0:
            raise ValueError("No claims in cluster_df")
        return num_correct / len(cluster_df)

    def calculate_accuracy_excluding_no_predict(self, cluster_df):
        # clusters_df columns - text, veracity, predict, predicted_veracity, embeddings, cluster, num_correct_in_cluster, total_in_cluster, cluster_accuracy

        cluster_df = cluster_df[cluster_df['predict']]
        # Filter where predicted veracity equals 1 or 3
        cluster_df = cluster_df[cluster_df['predicted_veracity'].isin([1, 3])]

        return self.calculate_accuracy(cluster_df)

    def calculate_percentage_of_four_and_five(self, cluster_df):
        original_cluster_df = cluster_df.copy(deep=True)
        cluster_df = cluster_df[cluster_df['predict']]
        original_length = len(cluster_df)
        cluster_df = cluster_df[cluster_df['predicted_veracity'].isin([1, 3])]

        original_cluster_df = original_cluster_df[original_cluster_df['predict'] == False]
        original_length_predict_false = len(original_cluster_df)
        original_cluster_df = original_cluster_df[original_cluster_df['predicted_veracity'].isin([1, 3])]

        return 1 - (len(cluster_df) / original_length), 1 - (len(original_cluster_df) / original_length_predict_false)

    def calculate_precision_recall_for_a_value(self, cluster_df, value: int):
        if len(cluster_df) == 0:
            raise ValueError("No claims in cluster_df")
        
        # Replace all 5 in predicted_veracity with 4
        cluster_df.loc[cluster_df['predicted_veracity'] == 5, 'predicted_veracity'] = 4

        # Filter df to where predict is true
        cluster_df = cluster_df[cluster_df['predict']]
        # Cast veracity and predicted_veracity as int
        cluster_df.loc[:, 'veracity'] = cluster_df['veracity'].astype(int)
        cluster_df.loc[:, 'predicted_veracity'] = cluster_df['predicted_veracity'].astype(int)
        
        # Filter df to where veracity is value
        cluster_df = cluster_df[cluster_df['veracity'] == value]
        # Replace all vlaues where predicted_veracity is 4 with 1 if value is 3 and 3 if value is 1
        if value == 3:
            cluster_df.loc[cluster_df['predicted_veracity'] == 4, 'predicted_veracity'] = 1
        else:
            cluster_df.loc[cluster_df['predicted_veracity'] == 4, 'predicted_veracity'] = 3

        # # Check if veracity and predicted_veracity only contain 0 and 1
        # if not cluster_df['veracity'].isin([1, 0]).all() or not cluster_df['predicted_veracity'].isin([1, 0]).all():
        #     mapped_values = cluster_df['veracity'].map({1: 0, 3: 1, "1": 0, "3": 1})
        #     print(cluster_df['veracity'].value_counts())
        #     cluster_df.loc[:, 'veracity'] = mapped_values.astype(int)
        #     mapped_values_predicted = cluster_df['predicted_veracity'].map({1: 0, 3: 1, "1": 0, "3": 1})
        #     print(cluster_df['predicted_veracity'].value_counts())
        #     cluster_df.loc[:, 'predicted_veracity'] = mapped_values_predicted.astype(int)
        
        # pos_value = 0
        # if value == 3:
        #     pos_value = 1

        # If count of pos_value in veracity is 0, return NaN for precision and recall
        if cluster_df['veracity'].value_counts().get(value) == None:
            print("happened")
            return "No values with this veracity were possible to predict", "No values with this veracity were possible to predict"
        
        print("ERROR DEBUG")
        print(cluster_df['veracity'].value_counts())
        print(cluster_df['predicted_veracity'].value_counts())
        precision = metrics.precision_score(cluster_df['veracity'], cluster_df['predicted_veracity'], average='binary', pos_label=value)
        recall = metrics.recall_score(cluster_df['veracity'], cluster_df['predicted_veracity'], average='binary', pos_label=value)
        
        # Get count of true positives
        true_positives = cluster_df[(cluster_df['veracity'] == value) & (cluster_df['predicted_veracity'] == value)].shape[0]
        if (precision == 0 or recall == 0):
            if true_positives == 0:
                print("sad...")
                # raise ValueError("Precision or recall is 0 (actually)")
            else:
                # Print value counts for veracity and predicted_veracity
                print("here")
                print(cluster_df['veracity'].value_counts())
                print(cluster_df['predicted_veracity'].value_counts())
                raise ValueError("Precision or recall is 0 (error)")

        return precision, recall
    
    def calculate_precision_recall(self, cluster_df):
        if len(cluster_df) == 0:
            raise ValueError("No claims in cluster_df")
        
        # Filter df to where predict is true
        cluster_df = cluster_df[cluster_df['predict']]
        cluster_df.loc[:, 'veracity'] = cluster_df['veracity'].astype(int)
        cluster_df.loc[:, 'predicted_veracity'] = cluster_df['predicted_veracity'].astype(int)
        cluster_df.loc[(cluster_df['predicted_veracity'] == 4) & (cluster_df['veracity'] == 3), 'predicted_veracity'] = 1
        cluster_df.loc[(cluster_df['predicted_veracity'] == 4) & (cluster_df['veracity'] == 1), 'predicted_veracity'] = 3


        # Check if veracity and predicted_veracity only contain 0 and 1
        # if not cluster_df['veracity'].isin([1, 0]).all() or not cluster_df['predicted_veracity'].isin([1, 0]).all():
        #     mapped_values = cluster_df['veracity'].map({1: 0, 3: 1, "1": 0, "3": 1})
        #     cluster_df.loc[:, 'veracity'] = mapped_values.astype(int)
        #     mapped_values_predicted = cluster_df['predicted_veracity'].map({1: 0, 3: 1, "1": 0, "3": 1})
        #     cluster_df.loc[:, 'predicted_veracity'] = mapped_values_predicted.astype(int)
        
        precision = metrics.precision_score(cluster_df['veracity'], cluster_df['predicted_veracity'], average='weighted')
        recall = metrics.recall_score(cluster_df['veracity'], cluster_df['predicted_veracity'], average='weighted')

        # Get count of true positives where veracity equals predicted_veracity
        true_positives = cluster_df[(cluster_df['veracity'] == cluster_df['predicted_veracity'])].shape[0]
        
        if precision == 0 or recall == 0:
            if true_positives == 0:
                print("sad...")
                # raise ValueError("Precision or recall is 0 (actually)")
            else:
                # Print value counts for veracity and predicted_veracity
                print(cluster_df['veracity'].value_counts())
                print(cluster_df['predicted_veracity'].value_counts())
                raise ValueError("Precision or recall is 0 (error)")

        return precision, recall

    def calculate_precision_recall_for_three_excluding_no_predict(self, cluster_df):
        cluster_df = cluster_df[cluster_df['predict']]
        cluster_df.loc[:, 'veracity'] = cluster_df['veracity'].astype(int)
        cluster_df.loc[:, 'predicted_veracity'] = cluster_df['predicted_veracity'].astype(int)
        cluster_df = cluster_df[cluster_df['predicted_veracity'].isin([1, 3])]
        return self.calculate_precision_recall_for_a_value(cluster_df, 3)
    
    def calculate_precision_recall_for_one_excluding_no_predict(self, cluster_df):
        cluster_df = cluster_df[cluster_df['predict']]
        cluster_df.loc[:, 'veracity'] = cluster_df['veracity'].astype(int)
        cluster_df.loc[:, 'predicted_veracity'] = cluster_df['predicted_veracity'].astype(int)
        cluster_df = cluster_df[cluster_df['predicted_veracity'].isin([1, 3])]
        return self.calculate_precision_recall_for_a_value(cluster_df, 1)
    
    def calculate_weighted_precision_recall_excluding_no_predict(self, cluster_df):
        cluster_df = cluster_df[cluster_df['predict']]
        cluster_df.loc[:, 'veracity'] = cluster_df['veracity'].astype(int)
        cluster_df.loc[:, 'predicted_veracity'] = cluster_df['predicted_veracity'].astype(int)
        cluster_df = cluster_df[cluster_df['predicted_veracity'].isin([1, 3])]
        return self.calculate_precision_recall(cluster_df)

