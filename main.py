import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Test script for process_single_claim
from ClusterAndPredict.ClusterAndPredict import ClusterAndPredict
import pandas as pd
from Testing.ParameterCreator import ParameterCreator
import boto3
from io import BytesIO
import json
import sys, logging
import joblib
import hashlib
import argparse

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def main(test_claim: str, force_retrain=False):
    timings = {}
    total_start = time.time()

    # 1. Load S3 data
    start = time.time()
    train_df = load_s3_data()
    timings['Load S3 data'] = time.time() - start

    if train_df.empty:
        print("Cannot load data from S3 bucket")
        return

    # 2. Prepare new record DataFrame
    start = time.time()
    new_record_df = pd.DataFrame({'text': [test_claim], 'veracity': [1]})
    timings['Prepare new record DataFrame'] = time.time() - start

    # 3. Get parameters from ParameterCreator
    start = time.time()
    params = ParameterCreator().get_parameters()
    timings['Get parameters'] = time.time() - start

    # 4. For each param, create and fit model
    for param in params:
        # 清理不需要的字段
        percentage = 0.75
        use_only_card = param['use_only_CARD']
        size_of_dataset = param['size_of_dataset']
        del param['size_of_dataset']
        del param['use_only_CARD']

        print("Creating a new model for param:", param)
        start = time.time()
        model = ClusterAndPredict(**param, train_df=train_df, force_retrain=force_retrain)
        timings['Create ClusterAndPredict object'] = time.time() - start

        print("Fitting the model with new data...")
        start = time.time()
        model.fit(
            new_record_df['text'].tolist(),
            new_record_df['veracity'].tolist()
        )
        timings['Model fit'] = time.time() - start

        start = time.time()
        object_output = model.get_all_performance_metrics()
        cluster_df = object_output['cluster_df']
        timings['Get performance metrics'] = time.time() - start

        print("This is the cluster_df", cluster_df)

    # 5. Generate explanations
    start = time.time()
    cluster_df = model.generate_explanations_and_similar_for_each_claim(cluster_df, "predict", "cluster", "text")
    timings['Generate explanations'] = time.time() - start

    # 6. Extract result for test_claim
    start = time.time()
    filtered_df = cluster_df[cluster_df['text'] == test_claim]
    filtered_dict = filtered_df.to_dict(orient='records')
    if filtered_dict:
        result = filtered_dict[0]
    else:
        result = {}
    timings['Extract result for test_claim'] = time.time() - start

    # 7. Prepare final result dictionary
    result_dict = {
        "claim": result.get("text", test_claim),
        "prediction": result.get("predicted_veracity", "Error"),
        "cluster_name": result.get("cluster_name", "N/A"),
        "explanation": result.get("detailed_explanation", "N/A"),
        "similar_claims": result.get("similar_claims", "N/A")
    }

    print("Output Results:")
    print(json.dumps(result_dict, indent=2))

    total_time = time.time() - total_start
    timings['Total time'] = total_time

    # 统一打印各部分的运行时间
    print("\n=== Timing Summary ===")
    for key, duration in timings.items():
        print(f"{key}: {duration:.3f} seconds")

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
                    # Read JSON content directly into memory
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
    # Adjust columns as needed (例如大写首字母等)
    cluster_df['cleaned_veracity'] = cluster_df['veracity'].map(
        {1: 'False', 3: 'True', 4: 'No prediction', 5: 'No prediction'})
    cluster_df['text'] = cluster_df['text'].str.capitalize()
    cluster_df['id'] = cluster_df['text'].str[:100].str.capitalize()
    return cluster_df


# if __name__ == '__main__':

#     if len(sys.argv) > 1:
#         test_claim = sys.argv[1]
#     else:
#         test_claim = "Default test claim"
#     result = main(test_claim)
#     print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if weights exist")
    parser.add_argument("test_claim", nargs="?", default="Default test claim", help="Test claim text")
    args = parser.parse_args()

    # 根据命令行参数设置 force_retrain 参数
    result = main(args.test_claim, force_retrain=args.force_retrain)
    print(json.dumps(result))