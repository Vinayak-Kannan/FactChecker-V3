import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Test script for process_single_claim
from ClusterAndPredict.ClusterAndPredict import ClusterAndPredict
import pandas as pd
from Testing.ParameterCreator import ParameterCreator
import boto3
from io import BytesIO
import json
import sys, logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def main(test_claim:str):
    # 1. Create sample training data
    s3_bucket = "sagemaker-us-east-1-390403859474"

    params = ParameterCreator().get_parameters()

    train_df = load_s3_data()

    if train_df.empty:
        print("Cannot load data from S3 bucket")
        return

    # add test_claim to train_df
    new_record_df = pd.DataFrame({'text': [test_claim], 'veracity': [1]})

    # 2. Initialize and fit the model
    for param in params:
        percentage = 0.75
        use_only_card = param['use_only_CARD']
        size_of_dataset = param['size_of_dataset']
        del param['size_of_dataset']
        del param['use_only_CARD']
        model = ClusterAndPredict(**param, train_df=train_df)
        model.fit(new_record_df['text'].tolist(), new_record_df['veracity'].tolist())
        object_output = model.get_all_performance_metrics()
        cluster_df = object_output['cluster_df']
        print("This is the cluster_df", cluster_df)


    cluster_df = model.generate_explanations_and_similar_for_each_claim(cluster_df, "predict", "cluster", "text")

    # get the test_claim from the cluster_df
    filtered_df = cluster_df[cluster_df['text'] == test_claim]
    filtered_dict = filtered_df.to_dict(orient='records')
    print(filtered_dict)
    if filtered_dict:
        result = filtered_dict[0]
    else:
        result = {}

    # 3. Test data
    
    result_dict = {
        "claim": result.get("text", test_claim),
        "prediction": result.get("predicted_veracity", "Error"),
        "cluster_name": result.get("cluster_name", "N/A"),
        "explanation": result.get("detailed_explanation", "N/A"),
        "similar_claims": result.get("similar_claims", "N/A")
    }

    print("Output Results:")
    print(json.dumps(result_dict))

    return 0


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
                    df = clean_columns_for_s3(df)
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
    cluster_df['cleaned_veracity'] = cluster_df['veracity'].map({1: 'False', 3: 'True', 4: 'No prediction', 5: 'No prediction'})
    # Capatalize text column first letter
    cluster_df['text'] = cluster_df['text'].str.capitalize()
    cluster_df['id'] = cluster_df['text'].str[:100].str.capitalize()
    return cluster_df


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_claim = sys.argv[1]
    else:
        test_claim = "Default test claim"
    result = main(test_claim)
    # turn result into JSON
    print(json.dumps(result))