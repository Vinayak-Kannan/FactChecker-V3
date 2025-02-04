import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Test script for process_single_claim
from ClusterAndPredict.ClusterAndPredict import ClusterAndPredict
import pandas as pd

def test_single_claim_processing():
    # 1. Create sample training data
    train_data = {
        'Text': [
            "Global warming is caused by human activities.",
            "Climate change is a natural cycle.",
            "The Earth is getting warmer every year.",
            "CO2 emissions are rising globally.",
            "Sea levels are rising due to climate change."
        ],
        'Numerical Rating': [3, 1, 3, 3, 3]  # 3 for True, 1 for False
    }
    train_df = pd.DataFrame(train_data)

    # 2. Initialize and fit the model
    model = ClusterAndPredict(train_df=train_df)
    model.fit(train_df['Text'].tolist(), train_df['Numerical Rating'].tolist())

    # 3. Test cases
    test_claims = [
        "Human activities are the main cause of global warming",  # Similar to existing true claim
        "Climate change is completely natural and not man-made",  # Similar to existing false claim
        "Completely unrelated claim about something else"  # Different topic
    ]

    print("\n=== Testing Basic Predictions ===")
    for claim in test_claims:
        print("\nTesting claim:", claim)
        try:
            result = model.process_single_claim(claim)
            print(result, 'process single claim result')
            print("Prediction:", result['prediction'])
            print("Confidence:", f"{result['confidence']*100:.2f}%")
            print("Basic explanation:", result['explanation'])
        except Exception as e:
            print(f"Error processing claim: {str(e)}")

    print("\n=== Testing Detailed Predictions ===")
    for claim in test_claims:
        print("\nTesting claim:", claim)
        
        result = model.process_single_claim(claim, generate_detailed_explanation=True)
        print("Prediction:", result['prediction'])
        print("Confidence:", f"{result['confidence']*100:.2f}%")
        print("Cluster:", result.get('cluster', 'N/A'))
        print("Cluster name:", result.get('cluster_name', 'N/A'))
        print("Detailed explanation:", result.get('detailed_explanation', 'N/A'))
        print("Similar claims:", result.get('similar_claims', 'N/A'))   

if __name__ == "__main__":
    test_single_claim_processing()