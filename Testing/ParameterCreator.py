from itertools import product
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Test script for process_single_claim
from ClusterAndPredict.ClusterAndPredict import ClusterAndPredict
import pandas as pd
import boto3
from io import BytesIO


class ParameterCreator:
    def __init__(self):
        self.parameters = {}
        train_df = ParameterCreator.load_s3_data()

        vals = {
            # HDBSCAN parameters
            'min_cluster_size': [5],
            'min_samples': [2],
            'use_hdbscan': [True],

            # UMAP parameters
            'n_neighbors': [int(train_df.shape[0] - 2)],
            'min_dist': [0],
            'num_components': [100],

            # UMAP options
            'no_umap': [False],
            'parametric_umap': [True],
            'supervised_umap': [True],

            # Data column specifications
            'claim_column_name': ['Text'],
            'veracity_column_name': ['Numerical Rating'],
            'supervised_label_column_name': ['Numerical Rating'],

            # Random seed options
            'random_seed': [True],
            'random_seed_val': [23],

            # Other pipeline options
            'use_weightage': [True],
            'k': [15000],
            'threshold_break': [0.9],
            'break_further': [True],
            'size_of_dataset': [1],
            'use_only_CARD': [True]
        }

        # Generate all combinations of parameter values
        keys, values = zip(*vals.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        # Store all parameter combinations
        self.parameters = param_combinations

    def get_parameters(self):
        return self.parameters


    def load_s3_data() -> pd.DataFrame:
        """
        Load and merge all CSV files from configured S3 location
        Returns:
            pd.DataFrame: Combined training data
        """
        s3_bucket = "sagemaker-us-east-1-390403859474"
        s3_prefix = "processed_files/"
        s3_client = boto3.client('s3') if s3_bucket else None

        if not s3_client:
            raise ValueError("S3 client not initialized")

        all_dfs = []
        paginator = s3_client.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.json'):
                        # Read CSV content directly into memory
                        response = s3_client.get_object(
                            Bucket=s3_bucket,
                            Key=obj['Key']
                        )
                        df = pd.read_json(BytesIO(response['Body'].read()))
                        df = ParameterCreator.clean_columns_for_s3(df)
                        all_dfs.append(df)

            # Combine all DataFrames
            train_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Successfully loaded {len(train_df)} training records")
            return train_df

        except Exception as e:
            print(f"S3 data loading failed: {str(e)}")
            raise


    def clean_columns_for_s3(cluster_df):
        # Loop through all 'predicted_veracity' and 1 and 3 to True and False and 4 and 5 to No prediction in a new column called 'cleaned_predicted_veracity'
        # cluster_df['cleaned_predicted_veracity'] = cluster_df['predicted_veracity'].map({1: 'False', 3: 'True', 4: 'No prediction', 5: 'No prediction'})
        cluster_df['cleaned_veracity'] = cluster_df['veracity'].map(
            {1: 'False', 3: 'True', 4: 'No prediction', 5: 'No prediction'})
        # Capatalize text column first letter
        cluster_df['text'] = cluster_df['text'].str.capitalize()
        cluster_df['id'] = cluster_df['text'].str[:100].str.capitalize()
        return cluster_df
